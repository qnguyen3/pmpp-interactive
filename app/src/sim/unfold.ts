// The matrices of section 19.3 (book pp. 463–472): a convolutional layer read as
// one matrix product Y = F · B.
//
// What it models, for one layer with C input maps of H × W, M output maps, K × K
// filters, a batch of N samples, and (stride 1, no padding) outputs of
// H_out × W_out = (H − K + 1) × (W − K + 1):
//   - F, the M × C × K × K filter array, read as an M × (C·K·K) matrix (no copy,
//     p. 465);
//   - B, one sample's input unfolded into a (C·K·K) × (H_out·W_out) matrix, and its
//     transpose (one row per output pixel, the orientation cudnnIm2Col and
//     ggml_im2col produce; ch19-facts A4, D2);
//   - Eq. 19.5 both ways: which X element a B entry copies, and every B entry that
//     copies a given X element;
//   - copies per input pixel, the expansion ratio of Eq. 19.4 and its K² limit;
//   - element and byte counts, with the three limits the lesson flags (a 32-bit int
//     index, cuDNN's documented tensor cap, Metal's documented minimum buffer cap);
//   - per-layer GEMM shapes, 16 × 16 output tiles, padding, phases, and requested
//     global loads per output for the direct kernel and the implicit one;
//   - the frames of the "Unfolding the input into a matrix" scene.
//
// A stride other than 1 is ours, not the book's (the lesson labels it Beyond): it
// exists to show that the duplication comes from overlapping windows, since with a
// stride equal to K (a vision patch embedding) nothing is duplicated (ch19-facts E1).
//
// Left out: padding, dilation and groups; data types other than FP32 for byte
// counts; caches (load counts are what the kernel requests, not DRAM traffic).

import { tr as trText } from '../lib/locale.ts';

export interface ConvShape {
  C: number;
  H: number;
  W: number;
  K: number;
  M: number;
  N: number;
  /** Ours (Beyond): 1 in the book. */
  stride?: number;
}

export const strideOf = (s: ConvShape) => s.stride ?? 1;
export function outDims(s: ConvShape): { Hout: number; Wout: number } {
  const S = strideOf(s);
  return { Hout: Math.floor((s.H - s.K) / S) + 1, Wout: Math.floor((s.W - s.K) / S) + 1 };
}
/** Output pixels per map: the number of columns of B. */
export const pixels = (s: ConvShape) => {
  const { Hout, Wout } = outDims(s);
  return Hout * Wout;
};
/** Terms per output: the number of rows of B and columns of F. */
export const ckk = (s: ConvShape) => s.C * s.K * s.K;
export const valid = (s: ConvShape) => s.K >= 1 && s.K <= s.H && s.K <= s.W && s.C >= 1 && s.M >= 1 && s.N >= 1;

// ---------------------------------------------------------------- the book's example (Fig. 19.2 / 19.8)

export const BOOK_SHAPE: ConvShape = { C: 3, H: 3, W: 3, K: 2, M: 2, N: 1 };
/** X[c][h][w], flattened row-major (Fig. 19.2b, p. 456). */
export const BOOK_X = [
  1, 2, 0, 1, 1, 3, 0, 2, 2, // X0
  0, 2, 1, 0, 3, 2, 1, 1, 0, // X1
  1, 2, 1, 0, 1, 3, 3, 3, 2, // X2
];
/** F[m][c][p][q], flattened row-major (Fig. 19.2b). */
export const BOOK_F = [
  1, 1, 2, 2, 1, 1, 1, 1, 0, 1, 1, 0, // output map 0
  1, 0, 0, 1, 2, 1, 2, 1, 1, 2, 2, 0, // output map 1
];
/** Y[m][h][w] as printed (Figs. 19.2b and 19.8). */
export const BOOK_Y = [14, 20, 15, 24, 12, 24, 17, 26];

// ---------------------------------------------------------------- the layer, three ways

/** Fig. 19.4's loop nest (with an optional stride, ours): the reference every other form is checked against. */
export function convDirect(s: ConvShape, X: readonly number[], F: readonly number[]): number[] {
  const { Hout, Wout } = outDims(s);
  const S = strideOf(s);
  const Y = new Array(s.N * s.M * Hout * Wout).fill(0);
  for (let n = 0; n < s.N; n++)
    for (let m = 0; m < s.M; m++)
      for (let h = 0; h < Hout; h++)
        for (let w = 0; w < Wout; w++) {
          let acc = 0;
          for (let c = 0; c < s.C; c++)
            for (let p = 0; p < s.K; p++)
              for (let q = 0; q < s.K; q++)
                acc += X[((n * s.C + c) * s.H + h * S + p) * s.W + w * S + q] * F[((m * s.C + c) * s.K + p) * s.K + q];
          Y[((n * s.M + m) * Hout + h) * Wout + w] = acc;
        }
  return Y;
}

/** Where B[u, v] comes from: Eq. 19.5 (p. 470), with the patch corner scaled by the stride. */
export interface BSource {
  u: number;
  v: number;
  /** Input channel: u / K². */
  c: number;
  /** Offset inside the patch: (u % K²) / K and (u % K²) % K. */
  p: number;
  q: number;
  /** Output pixel v's row and column: v / W_out and v % W_out. */
  oh: number;
  ow: number;
  /** Patch corner in the input map (= oh, ow at stride 1). */
  h0: number;
  w0: number;
  /** The X element: X[n, c, h, w]. */
  h: number;
  w: number;
  /** Flat index within one sample's X (C × H × W, row-major). */
  flat: number;
}

export function bToX(s: ConvShape, u: number, v: number): BSource {
  const { Wout } = outDims(s);
  const S = strideOf(s);
  const KK = s.K * s.K;
  const c = Math.floor(u / KK);
  const r = u % KK;
  const p = Math.floor(r / s.K);
  const q = r % s.K;
  const oh = Math.floor(v / Wout);
  const ow = v % Wout;
  const h0 = oh * S;
  const w0 = ow * S;
  const h = h0 + p;
  const w = w0 + q;
  return { u, v, c, p, q, oh, ow, h0, w0, h, w, flat: (c * s.H + h) * s.W + w };
}

/** Every (u, v) whose B entry copies X[n, c, h, w]: Eq. 19.5 run backwards. */
export function xToB(s: ConvShape, c: number, h: number, w: number): { u: number; v: number; p: number; q: number }[] {
  const { Hout, Wout } = outDims(s);
  const S = strideOf(s);
  const out: { u: number; v: number; p: number; q: number }[] = [];
  for (let p = 0; p < s.K; p++)
    for (let q = 0; q < s.K; q++) {
      const dh = h - p;
      const dw = w - q;
      if (dh < 0 || dw < 0 || dh % S || dw % S) continue;
      const oh = dh / S;
      const ow = dw / S;
      if (oh >= Hout || ow >= Wout) continue;
      out.push({ u: c * s.K * s.K + p * s.K + q, v: oh * Wout + ow, p, q });
    }
  return out.sort((a, b) => a.v - b.v || a.u - b.u);
}

/** One sample's B, (C·K·K) × (H_out·W_out), row-major. */
export function unfold(s: ConvShape, X: readonly number[], n = 0): number[] {
  const R = ckk(s);
  const P = pixels(s);
  const base = n * s.C * s.H * s.W;
  const B = new Array(R * P);
  for (let u = 0; u < R; u++) for (let v = 0; v < P; v++) B[u * P + v] = X[base + bToX(s, u, v).flat];
  return B;
}

/** One sample's Bᵀ, (H_out·W_out) × (C·K·K): one row per output pixel. */
export function unfoldT(s: ConvShape, X: readonly number[], n = 0): number[] {
  const R = ckk(s);
  const P = pixels(s);
  const B = unfold(s, X, n);
  const T = new Array(R * P);
  for (let u = 0; u < R; u++) for (let v = 0; v < P; v++) T[v * R + u] = B[u * P + v];
  return T;
}

/** Row-major A (m × k) times B (k × n). */
export function matmul(A: readonly number[], B: readonly number[], m: number, k: number, n: number): number[] {
  const C = new Array(m * n).fill(0);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) {
      let acc = 0;
      for (let t = 0; t < k; t++) acc += A[i * k + t] * B[t * n + j];
      C[i * n + j] = acc;
    }
  return C;
}

/** Y for one sample as F · B: M × P, which is Y[n, m, :, :] in row-major order (p. 466). */
export const convAsGemm = (s: ConvShape, X: readonly number[], F: readonly number[], n = 0) =>
  matmul(F, unfold(s, X, n), s.M, ckk(s), pixels(s));

/**
 * The project's linear layer, Y = X · Wᵀ with W stored (out, in) row-major
 * (labs/ch06/lab4_linear). Used to check which arguments lab 19.1 passes.
 */
export function linear(X: readonly number[], W: readonly number[], batch: number, inF: number, outF: number): number[] {
  const Y = new Array(batch * outF).fill(0);
  for (let b = 0; b < batch; b++)
    for (let o = 0; o < outF; o++) {
      let acc = 0;
      for (let i = 0; i < inF; i++) acc += X[b * inF + i] * W[o * inF + i];
      Y[b * outF + o] = acc;
    }
  return Y;
}

// ---------------------------------------------------------------- duplication

/** How many B entries copy each pixel of one input map (the same for every channel), H × W. */
export function copiesPerPixel(s: ConvShape): number[] {
  const out = new Array(s.H * s.W).fill(0);
  for (let h = 0; h < s.H; h++) for (let w = 0; w < s.W; w++) out[h * s.W + w] = xToB(s, 0, h, w).length;
  return out;
}

/** Copies after only the first `columns` columns of B are filled (the scene's counters). */
export function copiesAfterColumns(s: ConvShape, columns: number): number[] {
  const out = new Array(s.H * s.W).fill(0);
  const { Wout } = outDims(s);
  const S = strideOf(s);
  for (let v = 0; v < columns; v++) {
    const oh = Math.floor(v / Wout);
    const ow = v % Wout;
    for (let p = 0; p < s.K; p++) for (let q = 0; q < s.K; q++) out[(oh * S + p) * s.W + ow * S + q]++;
  }
  return out;
}

/** Eq. 19.4: B's elements over X's, per sample. Tends to K² at stride 1 when maps are much larger than filters. */
export const expansion = (s: ConvShape) => (ckk(s) * pixels(s)) / (s.C * s.H * s.W);
/** The ratio Eq. 19.4 approaches. */
export const limitRatio = (s: ConvShape) => s.K * s.K;

/** The ratio for square maps of side `side` (any C: it cancels). */
export const ratioAt = (K: number, side: number, stride = 1) => expansion({ C: 1, H: side, W: side, K, M: 1, N: 1, stride });

// ---------------------------------------------------------------- sizes and limits

/** 2^31 − 1: the largest index a 32-bit signed int holds. */
export const INT32_MAX = 2 ** 31 - 1;
/**
 * cuDNN: "The total size of a tensor ... is limited to 2 Giga-elements" (ch19-facts A1).
 * The docs do not say whether that is 2·10^9 or 2^31; the flag uses the lower.
 */
export const CUDNN_TENSOR_CAP = 2e9;
/** Apple: maxBufferLength "is at least 256 MB (268,435,456 bytes)" (ch19-facts C2). A floor, not any device's value. */
export const METAL_MIN_MAX_BUFFER = 268_435_456;
export const FP32 = 4;
export const MiB = 1024 * 1024;

export type FlagId = 'int32' | 'cudnn' | 'metal';
export interface Flag {
  id: FlagId;
  /** Which count crossed it. */
  what: 'sample' | 'batch';
}
export interface Sizes {
  xSample: number;
  xBatch: number;
  bSample: number;
  bBatch: number;
  bBytesSample: number;
  bBytesBatch: number;
  xBytesBatch: number;
  flags: Flag[];
}

export function limitFlags(elements: number): FlagId[] {
  const out: FlagId[] = [];
  if (elements > INT32_MAX) out.push('int32');
  if (elements >= CUDNN_TENSOR_CAP) out.push('cudnn');
  if (elements * FP32 > METAL_MIN_MAX_BUFFER) out.push('metal');
  return out;
}

export function sizes(s: ConvShape): Sizes {
  const xSample = s.C * s.H * s.W;
  const bSample = ckk(s) * pixels(s);
  const flags: Flag[] = [];
  const one = limitFlags(bSample);
  for (const id of one) flags.push({ id, what: 'sample' }); // i18n-ignore: identifier compared by the chapter, which shows its own text
  for (const id of limitFlags(bSample * s.N)) if (!one.includes(id)) flags.push({ id, what: 'batch' }); // i18n-ignore: identifier compared by the chapter, which shows its own text
  return {
    xSample,
    xBatch: xSample * s.N,
    bSample,
    bBatch: bSample * s.N,
    bBytesSample: bSample * FP32,
    bBytesBatch: bSample * s.N * FP32,
    xBytesBatch: xSample * s.N * FP32,
    flags,
  };
}

// ---------------------------------------------------------------- GEMM shapes per layer

export interface GemmShape {
  /** Rows of Y and F: M. */
  rows: number;
  /** The reduction: C·K·K. */
  inner: number;
  /** Columns of Y and B: P per sample, or N·P for the whole batch at once. */
  cols: number;
  /** How many GEMMs: N (one per sample) or 1 (whole batch). */
  gemms: number;
  /** T × T output tiles (thread blocks) per GEMM, and in all. */
  tilesPerGemm: number;
  tilesTotal: number;
  /** Share of tile slots that hold no output (ceil padding at the edges of Y). */
  padding: number;
  /** Phases of the tiled loop: ceil(C·K·K / T). */
  phases: number;
  /** Global loads one thread requests: the direct kernel (one X and one F per term) and the implicit GEMM (one of each per phase). */
  loadsDirect: number;
  loadsImplicit: number;
  /** The whole-batch product comes out M × N × P (CNHW): reaching NCHW needs a permute. */
  needsPermute: boolean;
}

export function gemmShape(s: ConvShape, opts: { tile?: number; wholeBatch?: boolean } = {}): GemmShape {
  const T = opts.tile ?? 16;
  const whole = !!opts.wholeBatch;
  const rows = s.M;
  const inner = ckk(s);
  const cols = pixels(s) * (whole ? s.N : 1);
  const gemms = whole ? 1 : s.N;
  const tr = Math.ceil(rows / T);
  const tc = Math.ceil(cols / T);
  const tilesPerGemm = tr * tc;
  const phases = Math.ceil(inner / T);
  return {
    rows,
    inner,
    cols,
    gemms,
    tilesPerGemm,
    tilesTotal: tilesPerGemm * gemms,
    padding: 1 - (rows * cols) / (tr * T * tc * T),
    phases,
    loadsDirect: 2 * inner,
    loadsImplicit: 2 * phases,
    needsPermute: whole && s.N > 1,
  };
}

// ---------------------------------------------------------------- presets

export type PresetId = 'book' | 'c1' | 'c3' | 'c5' | 'early' | 'middle' | 'late' | 'pointwise';
export interface Preset {
  id: PresetId;
  label: string;
  /** Where the shape comes from. */
  from: string;
  shape: ConvShape;
}

/**
 * The book's example and LeNet-5's three convolutional layers (p. 458), and the
 * four layers lab 19.1 benchmarks (labs/ch19/lab1_convgemm/main.cpp, at batch 8).
 * The lab's shapes are ours: an early layer (few channels, a large map), a middle
 * and a late one (more channels, smaller maps), and a 1 × 1 layer.
 */
export const PRESETS: Preset[] = [
  { id: 'book', get label() { return trText("the book's example", 'ví dụ của sách'); }, from: 'Fig. 19.2, p. 456', shape: BOOK_SHAPE },
  { id: 'c1', get label() { return trText('LeNet C1', 'LeNet C1'); }, from: 'p. 458', shape: { C: 1, H: 32, W: 32, K: 5, M: 6, N: 1 } },
  { id: 'c3', get label() { return trText('LeNet C3', 'LeNet C3'); }, from: 'p. 458', shape: { C: 6, H: 14, W: 14, K: 5, M: 16, N: 1 } },
  { id: 'c5', get label() { return trText('LeNet C5', 'LeNet C5'); }, from: 'p. 458', shape: { C: 16, H: 5, W: 5, K: 5, M: 120, N: 1 } },
  { id: 'early', get label() { return trText("lab's early layer", 'lớp đầu của lab'); }, get from() { return trText('lab 19.1', 'bài thực hành 19.1'); }, shape: { C: 3, H: 224, W: 224, K: 3, M: 64, N: 8 } },
  { id: 'middle', get label() { return trText("lab's middle layer", 'lớp giữa của lab'); }, get from() { return trText('lab 19.1', 'bài thực hành 19.1'); }, shape: { C: 64, H: 56, W: 56, K: 3, M: 64, N: 8 } },
  { id: 'late', get label() { return trText("lab's late layer", 'lớp cuối của lab'); }, get from() { return trText('lab 19.1', 'bài thực hành 19.1'); }, shape: { C: 256, H: 14, W: 14, K: 3, M: 256, N: 8 } },
  { id: 'pointwise', get label() { return trText("lab's 1 × 1 layer", 'lớp 1 × 1 của lab'); }, get from() { return trText('lab 19.1', 'bài thực hành 19.1'); }, shape: { C: 256, H: 14, W: 14, K: 1, M: 256, N: 8 } },
];
export const preset = (id: PresetId) => PRESETS.find((p) => p.id === id)!;

// ---------------------------------------------------------------- scene frames

/** Small deterministic integers for drawings and checks (exact in floating point). */
export function makeData(len: number, seed: number, lo = 1, hi = 3): number[] {
  let x = seed >>> 0 || 1;
  return Array.from({ length: len }, () => {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
    return lo + ((x >>> 16) % (hi - lo + 1));
  });
}

/** One term of the dot product for B column v: which X it copies and which F weight multiplies it. */
export interface Term {
  u: number;
  x: number;
  f: number;
  src: BSource;
}

/** The C·K·K terms of output (m, v) in u order: row m of F against column v of B. */
export function dotTerms(s: ConvShape, X: readonly number[], F: readonly number[], m: number, v: number, n = 0): Term[] {
  const R = ckk(s);
  const base = n * s.C * s.H * s.W;
  return Array.from({ length: R }, (_, u) => {
    const src = bToX(s, u, v);
    return { u, x: X[base + src.flat], f: F[m * R + u], src };
  });
}

/**
 * Frames of "Unfolding the input into a matrix", from the book's numbers:
 * the columns of B in fill order, the copy counters after each column, the
 * F · B product, and the ratio curves of the last beat.
 */
export const UNFOLD_SCENE = (() => {
  const s = BOOK_SHAPE;
  const B = unfold(s, BOOK_X);
  const P = pixels(s);
  const R = ckk(s);
  const Y = matmul(BOOK_F, B, s.M, R, P);
  const term0 = dotTerms(s, BOOK_X, BOOK_F, 0, 0);
  return {
    shape: s,
    B,
    Y,
    rows: R,
    cols: P,
    /** The three per-channel partial sums of output (0, 0): 7 + 5 + 2. */
    perChannel: [0, 1, 2].map((c) => term0.filter((t) => t.src.c === c).reduce((a, t) => a + t.x * t.f, 0)),
    copiesAfter: Array.from({ length: P + 1 }, (_, k) => copiesAfterColumns(s, k)),
    entries: B.length,
    inputs: BOOK_X.length,
    /** Ratio against map side, K = 2, 3, 5, sides 3 to 64. */
    curves: [2, 3, 5].map((K) => ({ K, points: Array.from({ length: 62 }, (_, i) => ({ side: i + 3, ratio: K <= i + 3 ? ratioAt(K, i + 3) : NaN })) })),
    c1Ratio: expansion(preset('c1').shape),
  };
})();
