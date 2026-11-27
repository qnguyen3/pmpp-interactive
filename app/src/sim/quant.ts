// Quantisers, packings and the small address and unpacking models behind sections X2.2,
// X2.3 and X2.7 and the scene "One block, packed, shipped and unpacked". Group 2 owns this
// file. Other groups use the fixed API below (quantizeQ4_0, dequantQ4_0, quantizeQ8_0,
// dequantQ8_0, rng, sampleWeights) and do not edit it.
//
// Nothing here is in the book (module X2 is ours). Sources, all in notes/today/x2-facts.md:
//   D1  llama.cpp's reference quantisers, "for deterministic creation of model files":
//       Q8_0  d = amax / 127, id = d ? 1/d : 0, q = roundf(x·id), d stored as FP16;
//       Q4_0  the value of largest magnitude WITH its sign, d = max / -8, q = MIN(15,
//             (int8_t)(x·id + 8.5f)), d stored as FP16, byte j = q[j] | q[j+16] << 4,
//             value (q - 8)·d;
//       Q4_1  d = (max - min) / 15, m = min, q = MIN(15, (int8_t)((x - min)·id + 0.5f)),
//             value q·d + m, d and m stored as FP16.
//   D2  Q4_K: 64-weight chunks whose 32 bytes carry the first 32 weights in the low nibbles
//       and the next 32 in the high nibbles; get_scale_min_k4 for the 6-bit scales and mins.
//   D3  llama.cpp CUDA reads a 32-bit word of quants, masks it with 0x0F0F0F0F for weights
//       j..j+3 and shifts it by 4 for j+16..j+19; A5: it assembles that word from two
//       16-bit loads (get_int_b2) because a block is only 2-byte aligned.
//   D4  llama.cpp Metal masks a 16-bit word four ways with no shift and divides the matching
//       activations by 1, 256, 16 and 4096 first; the -8 is folded in as -8·d·sum(y).
//   D5  MLX affine: documented s = (max - min)/(2^b - 1), bias = min, q = round((w - β)/s);
//       coded: the larger-magnitude edge is the bias, the scale may be negative, and the
//       scale is nudged to edge / q0 so that code -q0 lands on zero. Codes LSB-first, 8 per
//       uint32 at 4 bits.
//   E2  error orders of magnitude; E4 the folding of the offset out of the inner loop.
//
// Float32 arithmetic is emulated with Math.fround at every step the C code rounds, roundf is
// half away from zero, and (int8_t) of a non-negative float truncates. FP16 rounding of a
// stored scale comes from minifloat.ts (round to nearest even: the IEEE default, and our
// assumption for llama.cpp's GGML_FP32_TO_FP16, which the facts pass did not read).
// Choices that are ours, not a project's: the generators (mulberry32, Box-Muller, Student t
// from three squared normals, scaled to unit variance), the outlier demo block, and in
// errorSweep the double-precision arithmetic (FP16 scales still rounded).
//
// What it leaves out: time, caches, anything about a GPU; llama.cpp's importance-matrix
// quantisers (make_qx_quants, D1), which choose d differently for the same layout; the
// packing of MLX's 3-, 5- and 6-bit codes; MLX's eps (we use 1e-7; only an all-equal group
// can reach it); what a negative value cast to uint8 does in MLX's kernel (we clamp at 0;
// the facts show it cannot happen outside such degenerate groups).

import { FP16, encode, round as roundTo } from './minifloat';
import { tr } from '../lib/locale.ts';

export const f32 = Math.fround;
/** C's roundf: to nearest, halves away from zero. Exact for every float input. */
export const roundf = (x: number) => (x < 0 ? -Math.floor(-x + 0.5) : Math.floor(x + 0.5));
/** The value an FP16 store keeps (round to nearest even). */
export const fp16 = (x: number) => roundTo(FP16, x);
/** The FP16 bit pattern a store writes. */
export const fp16Bits = (x: number) => encode(FP16, x);
/** The float an FP16 bit pattern decodes to. */
export function fp16FromBits(b: number): number {
  const s = b & 0x8000 ? -1 : 1;
  const e = (b >> 10) & 0x1f;
  const m = b & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

// ---------------------------------------------------------------- seeded weights
/** mulberry32: a seeded uniform generator on [0, 1). */
export function rng(seed: number): () => number {
  let h = seed >>> 0;
  return () => {
    h = (h + 0x6d2b79f5) >>> 0;
    let t = Math.imul(h ^ (h >>> 15), h | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** One standard normal by Box-Muller. */
export function gauss(r: () => number): number {
  const u = 1 - r();
  const v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
export type Dist = 'normal' | 't3';
export const DIST_LABEL: Record<Dist, string> = {
  get normal() {
    return tr('normal', 'chuẩn');
  },
  get t3() {
    return tr('Student t, 3 degrees of freedom', 'Student t, 3 bậc tự do');
  },
};

/**
 * n float32 weights with standard deviation sigma: normal, or Student t with 3 degrees of
 * freedom (heavy tails) scaled to the same variance (a raw t3 has variance 3).
 */
export function sampleWeights(seed: number, n: number, dist: Dist = 'normal', sigma = 1): number[] {
  const r = rng(seed);
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const z = gauss(r);
    if (dist === 'normal') out[i] = f32(sigma * z);
    else {
      const a = gauss(r), b = gauss(r), c = gauss(r);
      out[i] = f32((sigma * z) / Math.sqrt((a * a + b * b + c * c) / 3) / Math.sqrt(3));
    }
  }
  return out;
}

// ---------------------------------------------------------------- llama.cpp Q8_0 (D1)
export interface Q8_0Block {
  /** The float32 d the codes were computed with. */
  d: number;
  /** FP16 bits of the stored d. */
  dBits: number;
  /** 32 signed codes, -127..127. */
  codes: number[];
  /** The 34 bytes of block_q8_0: d little-endian, then the codes as two's complement bytes. */
  bytes: number[];
}

const absMax = (x: number[]) => x.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
const need32 = (v: number[]) => {
  if (v.length !== 32) throw new Error(`one block is 32 weights, got ${v.length}`);
};
const le16 = (b: number) => [b & 0xff, (b >> 8) & 0xff];

/** quantize_row_q8_0_ref on one block of 32. */
export function quantizeQ8_0(values: number[]): Q8_0Block {
  need32(values);
  const x = values.map(f32);
  const d = f32(absMax(x) / 127);
  const id = d ? f32(1 / d) : 0;
  const codes = x.map((v) => roundf(f32(v * id)));
  const dBits = fp16Bits(d);
  return { d, dBits, codes, bytes: [...le16(dBits), ...codes.map((q) => q & 0xff)] };
}
/** dequantize_row_q8_0: q·d with d read back from FP16. */
export function dequantQ8_0(b: { dBits: number; codes: number[] }): number[] {
  const d = fp16FromBits(b.dBits);
  return b.codes.map((q) => f32(q * d));
}

// ---------------------------------------------------------------- llama.cpp Q4_0 (D1)
export interface Q4_0Block {
  /** The float32 d the codes were computed with: the signed extreme over -8. */
  d: number;
  /** FP16 bits of the stored d. */
  dBits: number;
  /** 32 stored codes, 0..15; weight j's value is (codes[j] - 8)·d. */
  codes: number[];
  /** The 18 bytes of block_q4_0: d little-endian, then qs[16] with qs[j] = codes[j] | codes[j+16] << 4. */
  bytes: number[];
}

/** The value of largest magnitude with its sign, and where it is: the first one wins a tie, as in the C loop. */
export function signedExtreme(x: number[]): { index: number; value: number } {
  let amax = 0;
  let max = 0;
  let index = -1;
  x.forEach((v, i) => {
    if (amax < Math.abs(v)) {
      amax = Math.abs(v);
      max = v;
      index = i;
    }
  });
  return { index, value: max };
}

/** quantize_row_q4_0_ref on one block of 32. */
export function quantizeQ4_0(values: number[]): Q4_0Block {
  need32(values);
  const x = values.map(f32);
  const { value: max } = signedExtreme(x);
  const d = f32(max / -8); // an all-zero block gives -0: 0 / -8
  const id = d ? f32(1 / d) : 0;
  const codes = x.map((v) => Math.min(15, Math.trunc(f32(f32(v * id) + 8.5))));
  const dBits = fp16Bits(d);
  return { d, dBits, codes, bytes: [...le16(dBits), ...packQ4_0(codes)] };
}
/** dequantize_row_q4_0: (q - 8)·d with d read back from FP16. */
export function dequantQ4_0(b: { dBits: number; codes: number[] }): number[] {
  const d = fp16FromBits(b.dBits);
  return b.codes.map((q) => f32((q - 8) * d));
}
/** A Q4_0 block back from its 18 bytes. */
export function parseQ4_0(bytes: number[]): { dBits: number; codes: number[] } {
  return { dBits: bytes[0] | (bytes[1] << 8), codes: unpackQ4_0(bytes.slice(2, 18)) };
}

export interface Q4_1Block {
  d: number;
  m: number;
  dBits: number;
  mBits: number;
  /** 0..15; value codes[j]·d + m. */
  codes: number[];
}
/** quantize_row_q4_1_ref on one block of 32. */
export function quantizeQ4_1(values: number[]): Q4_1Block {
  need32(values);
  const x = values.map(f32);
  let min = 3.4028234663852886e38;
  let max = -3.4028234663852886e38;
  for (const v of x) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const d = f32((max - min) / 15);
  const id = d ? f32(1 / d) : 0;
  const codes = x.map((v) => Math.min(15, Math.trunc(f32(f32(f32(v - min) * id) + 0.5))));
  return { d, m: min, dBits: fp16Bits(d), mBits: fp16Bits(min), codes };
}
export function dequantQ4_1(b: { dBits: number; mBits: number; codes: number[] }): number[] {
  const d = fp16FromBits(b.dBits);
  const m = fp16FromBits(b.mBits);
  return b.codes.map((q) => f32(f32(q * d) + m));
}

/** Row helpers: a row is a whole number of 32-weight blocks. */
export function quantizeRowQ4_0(row: number[]): Q4_0Block[] {
  if (row.length % 32) throw new Error('a Q4_0 row is a whole number of 32-weight blocks');
  return Array.from({ length: row.length / 32 }, (_, b) => quantizeQ4_0(row.slice(32 * b, 32 * b + 32)));
}
export function quantizeRowQ8_0(row: number[]): Q8_0Block[] {
  if (row.length % 32) throw new Error('a Q8_0 row is a whole number of 32-weight blocks');
  return Array.from({ length: row.length / 32 }, (_, b) => quantizeQ8_0(row.slice(32 * b, 32 * b + 32)));
}

// ---------------------------------------------------------------- the stepper's six rules
export type Rule = 'q8_0' | 'q4_0' | 'sym4' | 'q4_1' | 'mlx-doc' | 'mlx-code';
export const RULES: Rule[] = ['q8_0', 'q4_0', 'sym4', 'q4_1', 'mlx-doc', 'mlx-code'];
export const RULE_LABEL: Record<Rule, string> = {
  q8_0: 'llama.cpp Q8_0', // i18n-ignore: format name
  q4_0: 'llama.cpp Q4_0', // i18n-ignore: format name
  get sym4() {
    return tr('symmetric ±7 (textbook)', 'đối xứng ±7 (sách giáo khoa)');
  },
  q4_1: 'llama.cpp Q4_1', // i18n-ignore: format name
  get 'mlx-doc'() {
    return tr('MLX affine 4-bit, as documented', 'MLX affine 4-bit, theo tài liệu');
  },
  get 'mlx-code'() {
    return tr('MLX affine 4-bit, as coded', 'MLX affine 4-bit, theo mã nguồn');
  },
};
export const RULE_FACT: Record<Rule, string> = {
  q8_0: 'D1',
  q4_0: 'D1',
  get sym4() {
    return tr('ours (E2 "sym4")', 'của chúng tôi (E2 "sym4")');
  },
  q4_1: 'D1',
  'mlx-doc': 'D5',
  'mlx-code': 'D5',
};

export interface Trace {
  rule: Rule;
  values: number[];
  /** Step 1: what sets the scale. */
  extreme: { index: number; value: number; minIndex: number; min: number; maxIndex: number; max: number };
  /** Step 2: the float scale the codes are computed with, and the offset (llama.cpp m, MLX bias; 0 when symmetric). */
  scale: number;
  offset: number;
  /** Q4_0's fixed code offset, 8; 0 elsewhere. Value = (code - codeZero)·scale + offset. */
  codeZero: number;
  /** What dequantisation uses: the same, or the FP16-rounded scale and offset. */
  scaleStored: number;
  offsetStored: number;
  /** Step 3: each weight divided by the scale (after removing the offset where there is one). */
  divided: number[];
  /** Step 4: the integer before clamping (Q4_0 adds 8.5 and truncates). */
  rounded: number[];
  /** Step 5: the stored code, and which weights the clamp changed. */
  codes: number[];
  clipped: boolean[];
  /** Step 6: what dequantisation returns, and the error. */
  dequant: number[];
  err: number[];
  /** The stored code range. */
  codeMin: number;
  codeMax: number;
  /** Every value the block can represent, ascending. */
  levels: number[];
  /** sqrt(Σ err²) / sqrt(Σ w²). */
  rel: number;
  /** MLX as coded only: the code that sits on zero (-q0), or null; and what it dequantises to after storage. */
  zeroCode: number | null;
  zeroValue: number | null;
}

const MLX_EPS = 1e-7;

/** One block through one rule, every intermediate kept for the stepper. */
export function quantTrace(rule: Rule, values: number[], fp16Scale = true): Trace {
  const x = values.map(f32);
  const ext = signedExtreme(x);
  let minIndex = 0;
  let maxIndex = 0;
  x.forEach((v, i) => {
    if (v < x[minIndex]) minIndex = i;
    if (v > x[maxIndex]) maxIndex = i;
  });
  const extreme = { ...ext, minIndex, min: x[minIndex], maxIndex, max: x[maxIndex] };
  const store = (v: number) => (fp16Scale ? fp16(v) : v);
  let scale = 0;
  let offset = 0;
  let codeZero = 0;
  let codeMin = 0;
  let codeMax = 15;
  let divided: number[];
  let rounded: number[];
  let codes: number[];
  let zeroCode: number | null = null;

  if (rule === 'q8_0' || rule === 'sym4') {
    const qmax = rule === 'q8_0' ? 127 : 7;
    codeMin = -qmax;
    codeMax = qmax;
    scale = f32(absMax(x) / qmax);
    const id = scale ? f32(1 / scale) : 0;
    divided = x.map((v) => f32(v * id));
    rounded = divided.map(roundf);
    codes = rounded.map((q) => Math.max(-qmax, Math.min(qmax, q)));
  } else if (rule === 'q4_0') {
    codeZero = 8;
    scale = f32(ext.value / -8);
    const id = scale ? f32(1 / scale) : 0;
    divided = x.map((v) => f32(v * id));
    rounded = divided.map((v) => Math.trunc(f32(v + 8.5)));
    codes = rounded.map((q) => Math.min(15, q));
  } else if (rule === 'q4_1') {
    scale = f32((extreme.max - extreme.min) / 15);
    offset = extreme.min;
    const id = scale ? f32(1 / scale) : 0;
    divided = x.map((v) => f32(f32(v - offset) * id));
    rounded = divided.map((v) => Math.trunc(f32(v + 0.5)));
    codes = rounded.map((q) => Math.min(15, q));
  } else if (rule === 'mlx-doc') {
    scale = f32((extreme.max - extreme.min) / 15);
    offset = extreme.min;
    divided = x.map((v) => (scale ? f32(f32(v - offset) / scale) : 0));
    rounded = divided.map(roundf);
    codes = rounded.map((q) => Math.max(0, Math.min(15, q)));
  } else {
    // MLX affine_quantize, as the kernel computes it (D5).
    const nbins = 15;
    let s = Math.max(f32((extreme.max - extreme.min) / nbins), MLX_EPS);
    const side = Math.abs(extreme.min) > Math.abs(extreme.max);
    s = side ? s : -s;
    const edge = side ? extreme.min : extreme.max;
    const q0 = roundf(f32(edge / s));
    const atZero = q0 === 0;
    scale = f32(atZero ? s : edge / q0);
    offset = atZero ? 0 : edge;
    zeroCode = atZero ? null : -q0;
    divided = x.map((v) => f32(f32(v - offset) / scale));
    rounded = divided.map(roundf);
    codes = rounded.map((q) => Math.max(0, Math.min(nbins, q)));
  }
  const scaleStored = store(scale);
  const offsetStored = store(offset);
  const value = (q: number) => f32(f32((q - codeZero) * scaleStored) + offsetStored);
  const dequant = codes.map(value);
  const err = dequant.map((v, i) => v - x[i]);
  const levels: number[] = [];
  for (let q = codeMin; q <= codeMax; q++) levels.push(value(q));
  levels.sort((a, b) => a - b);
  const norm = Math.sqrt(x.reduce((s, v) => s + v * v, 0));
  const rel = norm ? Math.sqrt(err.reduce((s, e) => s + e * e, 0)) / norm : 0;
  return {
    rule, values: x, extreme, scale, offset, codeZero, scaleStored, offsetStored, divided, rounded, codes,
    clipped: codes.map((q, i) => q !== rounded[i]), dequant, err, codeMin, codeMax, levels, rel, zeroCode,
    zeroValue: zeroCode == null ? null : value(zeroCode),
  };
}

/**
 * The stepper's block: 32 seeded weights (standard deviation 0.25), with an outlier of the
 * given magnitude at weight 5, negative ('-'), positive ('+'), or negative at 5 with its
 * mirror image at weight 21 ('±'). Weights 5 and 21 share byte 5 of a Q4_0 block.
 */
export type OutlierSign = '+' | '-' | '±';
export const DEMO_SIGMA = 0.25;
export const DEMO_SEED = 7;
export const OUTLIER_AT = 5;
export const MIRROR_AT = 21;
export function demoBlock(dist: Dist = 'normal', outlier = 0.8, sign: OutlierSign = '±', seed = DEMO_SEED): number[] {
  const w = sampleWeights(seed, 32, dist, DEMO_SIGMA);
  if (outlier > 0) {
    w[OUTLIER_AT] = f32(sign === '+' ? outlier : -outlier);
    if (sign === '±') w[MIRROR_AT] = f32(outlier);
  }
  return w;
}
/** The block the scene and the X2.3 explorers draw: the stepper's default. */
export const SCENE_BLOCK = demoBlock();

// ---------------------------------------------------------------- error against bits and blocks (E2's shape)
export type Scheme = 'signed-extreme' | 'symmetric' | 'affine';
export const SCHEME_LABEL: Record<Scheme, string> = {
  get 'signed-extreme'() {
    return tr('Q4_0-style: signed extreme over -2^(b-1)', 'kiểu Q4_0: giá trị cực trị có dấu chia cho -2^(b-1)');
  },
  get symmetric() {
    return tr('symmetric: ±(2^(b-1) - 1)', 'đối xứng: ±(2^(b-1) - 1)');
  },
  get affine() {
    return tr('affine: min and max (MLX as documented)', 'affine: min và max (MLX theo tài liệu)');
  },
};
export type Gran = number | 'row' | 'tensor';
export const GRANS: Gran[] = [16, 32, 64, 128, 256, 'row', 'tensor'];
export const granLabel = (g: Gran) =>
  g === 'row' ? tr('one per row', 'một cho mỗi hàng') : g === 'tensor' ? tr('one per tensor', 'một cho mỗi tensor') : tr(`block ${g}`, `khối ${g}`);

export interface SweepPoint {
  bits: number;
  gran: Gran;
  /** Weights sharing one scale. */
  group: number;
  /** Code bits plus an FP16 scale (and an FP16 bias for affine) per group. */
  bpw: number;
  /** Relative RMS error ‖w - ŵ‖ / ‖w‖. */
  rel: number;
}

/** Quantise and dequantise one group in place into out, by a b-bit scheme with FP16 scales. Double arithmetic. */
function quantGroup(x: Float64Array, lo: number, hi: number, bits: number, scheme: Scheme, out: Float64Array) {
  if (scheme === 'affine') {
    let mn = Infinity;
    let mx = -Infinity;
    for (let i = lo; i < hi; i++) {
      if (x[i] < mn) mn = x[i];
      if (x[i] > mx) mx = x[i];
    }
    const top = 2 ** bits - 1;
    const s = (mx - mn) / top;
    const s16 = fp16(s);
    const m16 = fp16(mn);
    for (let i = lo; i < hi; i++) {
      const q = s ? Math.max(0, Math.min(top, roundf((x[i] - mn) / s))) : 0;
      out[i] = q * s16 + m16;
    }
    return;
  }
  let amax = 0;
  let max = 0;
  for (let i = lo; i < hi; i++)
    if (amax < Math.abs(x[i])) {
      amax = Math.abs(x[i]);
      max = x[i];
    }
  if (scheme === 'symmetric') {
    const qmax = 2 ** (bits - 1) - 1;
    const d = amax / qmax;
    const id = d ? 1 / d : 0;
    const d16 = fp16(d);
    for (let i = lo; i < hi; i++) out[i] = Math.max(-qmax, Math.min(qmax, roundf(x[i] * id))) * d16;
    return;
  }
  const half = 2 ** (bits - 1);
  const d = max / -half;
  const id = d ? 1 / d : 0;
  const d16 = fp16(d);
  const top = 2 ** bits - 1;
  for (let i = lo; i < hi; i++) out[i] = (Math.min(top, Math.trunc(x[i] * id + half + 0.5)) - half) * d16;
}

/** Relative RMS error of a rows × cols matrix quantised with one scale per `group` weights along each row (or per row, or per tensor). */
export function quantError(w: Float64Array, rows: number, cols: number, bits: number, scheme: Scheme, gran: Gran): number {
  const out = new Float64Array(w.length);
  if (gran === 'tensor') quantGroup(w, 0, w.length, bits, scheme, out);
  else {
    const g = gran === 'row' ? cols : gran;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c += g) quantGroup(w, r * cols + c, r * cols + Math.min(cols, c + g), bits, scheme, out);
  }
  let e = 0;
  let n = 0;
  for (let i = 0; i < w.length; i++) {
    e += (w[i] - out[i]) ** 2;
    n += w[i] ** 2;
  }
  return Math.sqrt(e / n);
}

export interface SweepOpts {
  dist: Dist;
  scheme: Scheme;
  bits?: number[];
  grans?: Gran[];
  seed?: number;
  rows?: number;
  cols?: number;
}
export const SWEEP_BITS = [2, 3, 4, 5, 6, 7, 8];
const sampleCache = new Map<string, Float64Array>();
function sweepSample(dist: Dist, seed: number, n: number): Float64Array {
  const k = `${dist}|${seed}|${n}`;
  let s = sampleCache.get(k);
  if (!s) {
    s = Float64Array.from(sampleWeights(seed, n, dist, 1));
    sampleCache.set(k, s);
  }
  return s;
}

/** Error for every (bits, granularity) pair on one seeded matrix (default 16 rows of 4096). */
export function errorSweep(o: SweepOpts): SweepPoint[] {
  const rows = o.rows ?? 16;
  const cols = o.cols ?? 4096;
  const w = sweepSample(o.dist, o.seed ?? 1, rows * cols);
  const out: SweepPoint[] = [];
  for (const gran of o.grans ?? GRANS)
    for (const bits of o.bits ?? SWEEP_BITS) {
      const group = gran === 'tensor' ? rows * cols : gran === 'row' ? cols : gran;
      const overhead = (o.scheme === 'affine' ? 32 : 16) / group;
      out.push({ bits, gran, group, bpw: bits + overhead, rel: quantError(w, rows, cols, bits, o.scheme, gran) });
    }
  return out;
}

// ---------------------------------------------------------------- packing and where each weight lives
/** Q4_0's qs: byte j holds weight j in its low nibble and weight j + 16 in its high nibble (D1). */
export function packQ4_0(codes: number[]): number[] {
  return Array.from({ length: 16 }, (_, j) => (codes[j] & 0xf) | ((codes[j + 16] & 0xf) << 4));
}
export function unpackQ4_0(qs: number[]): number[] {
  return Array.from({ length: 32 }, (_, j) => (j < 16 ? qs[j] & 0xf : qs[j - 16] >> 4));
}
/** MLX 4-bit: 8 codes per uint32, the first in the 4 least significant bits (D5). */
export function packMlx4(codes: number[]): number[] {
  const words: number[] = [];
  for (let w = 0; w < codes.length / 8; w++) {
    let v = 0;
    for (let k = 0; k < 8; k++) v |= (codes[8 * w + k] & 0xf) << (4 * k);
    words.push(v >>> 0);
  }
  return words;
}
export function unpackMlx4(words: number[]): number[] {
  return words.flatMap((v) => Array.from({ length: 8 }, (_, k) => (v >>> (4 * k)) & 0xf));
}
/** Q4_K's qs over a 256-weight super-block: chunk c's 32 bytes carry weights 64c..64c+31 low and 64c+32..64c+63 high (D2). */
export function packQ4K(codes: number[]): number[] {
  return Array.from({ length: codes.length / 2 }, (_, b) => {
    const c = Math.floor(b / 32);
    const l = b % 32;
    return (codes[64 * c + l] & 0xf) | ((codes[64 * c + 32 + l] & 0xf) << 4);
  });
}
export function unpackQ4K(qs: number[]): number[] {
  const out: number[] = [];
  for (let c = 0; c < qs.length / 32; c++) {
    for (let l = 0; l < 32; l++) out[64 * c + l] = qs[32 * c + l] & 0xf;
    for (let l = 0; l < 32; l++) out[64 * c + 32 + l] = qs[32 * c + l] >> 4;
  }
  return out;
}

export type Layout = 'q4_0' | 'mlx4' | 'q4_k';
export const LAYOUT_LABEL: Record<Layout, string> = {
  get q4_0() {
    return tr('llama.cpp Q4_0 (32 weights, 16 bytes of quants)', 'llama.cpp Q4_0 (32 trọng số, 16 byte mã lượng tử)');
  },
  get mlx4() {
    return tr('MLX 4-bit (32 weights, four uint32 words)', 'MLX 4-bit (32 trọng số, bốn word uint32)');
  },
  get q4_k() {
    return tr('llama.cpp Q4_K (one 64-weight chunk, 32 bytes)', 'llama.cpp Q4_K (một chunk 64 trọng số, 32 byte)');
  },
};
/** Weights the explorer shows per layout. */
export const LAYOUT_WEIGHTS: Record<Layout, number> = { q4_0: 32, mlx4: 32, q4_k: 64 };

export interface Place {
  /** Byte offset inside the quants (MLX: inside the words, little-endian). */
  byte: number;
  high: boolean;
  /** MLX only: which uint32 and which bits. */
  word?: number;
  bits?: [number, number];
}
/** Where weight j lives. */
export function placeOf(layout: Layout, j: number): Place {
  if (layout === 'q4_0') return { byte: j % 16, high: j >= 16 };
  if (layout === 'q4_k') return { byte: 32 * Math.floor(j / 64) + (j % 32), high: j % 64 >= 32 };
  const word = j >> 3;
  const k = j % 8;
  // Byte within memory assumes a little-endian machine (ours: the facts give the word order only).
  return { byte: 4 * word + (k >> 1), high: k % 2 === 1, word, bits: [4 * k, 4 * k + 3] };
}
/** The two weights in a byte: [low nibble, high nibble]. */
export function weightsInByte(layout: Layout, byte: number): [number, number] {
  if (layout === 'q4_0') return [byte, byte + 16];
  if (layout === 'q4_k') {
    const c = Math.floor(byte / 32);
    return [64 * c + (byte % 32), 64 * c + 32 + (byte % 32)];
  }
  return [2 * byte, 2 * byte + 1];
}

// ---------------------------------------------------------------- Q4_K's 6-bit scales and mins (D2)
/** get_scale_min_k4: scale and min j (0..7) from the 12 bytes of scales. */
export function getScaleMinK4(j: number, q: number[]): { sc: number; m: number } {
  if (j < 4) return { sc: q[j] & 63, m: q[j + 4] & 63 };
  return { sc: (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4), m: (q[j + 4] >> 4) | ((q[j] >> 6) << 4) };
}
/** The inverse of getScaleMinK4 (ours): 8 six-bit scales and 8 six-bit mins into 12 bytes. */
export function packScaleMinK4(sc: number[], m: number[]): number[] {
  const q = new Array<number>(12).fill(0);
  for (let j = 0; j < 4; j++) {
    q[j] = (sc[j] & 63) | ((sc[j + 4] >> 4) << 6);
    q[j + 4] = (m[j] & 63) | ((m[j + 4] >> 4) << 6);
  }
  for (let j = 4; j < 8; j++) q[j + 4] = (sc[j] & 0xf) | ((m[j] & 0xf) << 4);
  return q;
}
/** Where get_scale_min_k4 finds the bits of scale j and min j, for the table in X2.2. */
export function scaleMinSources(j: number): { sc: string; m: string } {
  if (j < 4) return { sc: tr(`byte ${j}, bits 0–5`, `byte ${j}, bit 0–5`), m: tr(`byte ${j + 4}, bits 0–5`, `byte ${j + 4}, bit 0–5`) };
  return {
    sc: tr(`byte ${j + 4}, bits 0–3, and byte ${j - 4}, bits 6–7`, `byte ${j + 4}, bit 0–3, và byte ${j - 4}, bit 6–7`),
    m: tr(`byte ${j + 4}, bits 4–7, and byte ${j}, bits 6–7`, `byte ${j + 4}, bit 4–7, và byte ${j}, bit 6–7`),
  };
}

// ---------------------------------------------------------------- blocks end to end (A5, C5)
export const Q4_0_BYTES = 18;
/** Byte addresses of block i's d and qs in an array of 18-byte blocks starting at base. */
export function blockAddress(i: number, base = 0, blockBytes = Q4_0_BYTES): { start: number; d: number; qs: number; end: number } {
  const start = base + i * blockBytes;
  return { start, d: start, qs: start + 2, end: start + blockBytes };
}
export const isAligned = (addr: number, width: number) => addr % width === 0;
/** The loads that read block i's 16 bytes of quants at a given width, and whether each is aligned. */
export function qsLoads(i: number, width: number, base = 0): { addr: number; ok: boolean }[] {
  const { qs } = blockAddress(i, base);
  return Array.from({ length: 16 / width }, (_, k) => ({ addr: qs + k * width, ok: isAligned(qs + k * width, width) }));
}

// ---------------------------------------------------------------- unpacking one word (D3, D4)
export type UnpackVariant = 'cuda32' | 'metal16';
export interface UnpackStep {
  op: 'load' | 'mask' | 'shift' | 'sub8' | 'dot';
  /** The word as it stands after this step (32 or 16 bits). */
  word: number;
  width: 32 | 16;
  /** Bit ranges this step singles out, with the weight each holds. */
  fields: { lo: number; hi: number; weight: number; value: number }[];
  /** What this step produces, per weight. */
  out: { weight: number; value: number; note: string }[];
  text: string;
}

const hex = (v: number, digits: number) => '0x' + (v >>> 0).toString(16).toUpperCase().padStart(digits, '0');

/**
 * Unpack one word of a Q4_0 block's quants, the way llama.cpp's CUDA kernel does
 * ('cuda32': word i of 4, assembled from two 16-bit loads, mask 0x0F0F0F0F, shift by 4,
 * minus 8) or its Metal kernel does ('metal16': word i of 8, four masks, no shift,
 * activations pre-divided by 1, 256, 16, 4096, then -8·sum(y) folded in).
 * y is the block's 32 float32 activations.
 */
export function unpackTrace(variant: UnpackVariant, qs: number[], i: number, y: number[]): UnpackStep[] {
  const codes = unpackQ4_0(qs);
  if (variant === 'cuda32') {
    const lo16 = qs[4 * i] | (qs[4 * i + 1] << 8);
    const hi16 = qs[4 * i + 2] | (qs[4 * i + 3] << 8);
    const v = (lo16 | (hi16 << 16)) >>> 0;
    const vi0 = (v & 0x0f0f0f0f) >>> 0;
    const vi1 = ((v >>> 4) & 0x0f0f0f0f) >>> 0;
    const lowW = [0, 1, 2, 3].map((k) => 4 * i + k);
    const highW = lowW.map((w) => w + 16);
    const all = [...lowW, ...highW];
    const signed = all.map((w) => codes[w] - 8);
    const dot = all.reduce((s, w, k) => f32(s + f32(signed[k] * y[w])), 0);
    return [
      {
        op: 'load', word: v, width: 32,
        fields: [0, 1, 2, 3].flatMap((k) => [
          { lo: 8 * k, hi: 8 * k + 3, weight: lowW[k], value: codes[lowW[k]] },
          { lo: 8 * k + 4, hi: 8 * k + 7, weight: highW[k], value: codes[highW[k]] },
        ]),
        out: [],
        text: tr(`Two 16-bit loads, ${hex(lo16, 4)} from qs[${4 * i}] and ${hex(hi16, 4)} from qs[${4 * i + 2}], make the 32-bit word ${hex(v, 8)}: bytes ${4 * i} to ${4 * i + 3} of the quants.`, `Hai lần load 16-bit, ${hex(lo16, 4)} từ qs[${4 * i}] và ${hex(hi16, 4)} từ qs[${4 * i + 2}], tạo thành word 32-bit ${hex(v, 8)}: các byte ${4 * i} đến ${4 * i + 3} của mã lượng tử.`),
      },
      {
        op: 'mask', word: vi0, width: 32,
        fields: [0, 1, 2, 3].map((k) => ({ lo: 8 * k, hi: 8 * k + 3, weight: lowW[k], value: codes[lowW[k]] })),
        out: lowW.map((w) => ({ weight: w, value: codes[w], note: tr('low nibble', 'nibble thấp') })),
        text: tr(`v & 0x0F0F0F0F = ${hex(vi0, 8)}: one byte per lane, holding weights ${lowW[0]} to ${lowW[3]}, codes ${lowW.map((w) => codes[w]).join(', ')}. Four neighbours, ready to meet four neighbouring int8 activations.`, `v & 0x0F0F0F0F = ${hex(vi0, 8)}: mỗi lane một byte, chứa các trọng số ${lowW[0]} đến ${lowW[3]}, mã ${lowW.map((w) => codes[w]).join(', ')}. Bốn phần tử liền kề, sẵn sàng gặp bốn activation int8 liền kề.`),
      },
      {
        op: 'shift', word: vi1, width: 32,
        fields: [0, 1, 2, 3].map((k) => ({ lo: 8 * k, hi: 8 * k + 3, weight: highW[k], value: codes[highW[k]] })),
        out: highW.map((w) => ({ weight: w, value: codes[w], note: tr('high nibble', 'nibble cao') })),
        text: tr(`(v >> 4) & 0x0F0F0F0F = ${hex(vi1, 8)}: weights ${highW[0]} to ${highW[3]}, codes ${highW.map((w) => codes[w]).join(', ')}. The same mask after one shift, and these four meet the activations 16 further on.`, `(v >> 4) & 0x0F0F0F0F = ${hex(vi1, 8)}: các trọng số ${highW[0]} đến ${highW[3]}, mã ${highW.map((w) => codes[w]).join(', ')}. Cùng mặt nạ đó sau một lần dịch bit, và bốn trọng số này gặp các activation cách 16 vị trí về phía sau.`),
      },
      {
        op: 'sub8', word: vi1, width: 32,
        fields: [],
        out: all.map((w, k) => ({ weight: w, value: signed[k], note: `${codes[w]} - 8` })),
        text: tr(`Subtract 8 from each code: ${signed.join(', ')}. llama.cpp's CUDA kernel does not do this per weight; it subtracts 8 times the activations' sum once per block (section X2.5).`, `Trừ 8 khỏi mỗi mã: ${signed.join(', ')}. Kernel CUDA của llama.cpp không làm việc này cho từng trọng số; nó trừ 8 lần tổng các activation một lần cho mỗi khối (mục X2.5).`),
      },
      {
        op: 'dot', word: vi1, width: 32, fields: [],
        out: all.map((w, k) => ({ weight: w, value: f32(signed[k] * y[w]), note: `(${signed[k]}) × y[${w}]` })),
        text: tr(`These eight codes times their activations add up to ${dot.toPrecision(6)}, still to be multiplied by the block's d.`, `Tám mã này nhân với các activation của chúng cộng lại được ${dot.toPrecision(6)}, vẫn còn phải nhân với d của khối.`),
      },
    ];
  }
  const u = qs[2 * i] | (qs[2 * i + 1] << 8);
  const W = [2 * i, 2 * i + 1, 2 * i + 16, 2 * i + 17];
  const MASK = [0x000f, 0x0f00, 0x00f0, 0xf000];
  const DIV = [1, 256, 16, 4096];
  const LO = [0, 8, 4, 12];
  const masked = MASK.map((m) => u & m);
  const yl = W.map((w, k) => f32(y[w] / DIV[k]));
  const prods = masked.map((m, k) => f32(yl[k] * m));
  const acc = prods.reduce((s, p) => f32(s + p), 0);
  const sumy = W.reduce((s, w) => f32(s + y[w]), 0);
  return [
    {
      op: 'load', word: u, width: 16,
      fields: W.map((w, k) => ({ lo: LO[k], hi: LO[k] + 3, weight: w, value: codes[w] })),
      out: [],
      text: tr(`One 16-bit load from qs[${2 * i}]: ${hex(u, 4)}, bytes ${2 * i} and ${2 * i + 1}, four weights: ${W.join(', ')}.`, `Một lần load 16-bit từ qs[${2 * i}]: ${hex(u, 4)}, các byte ${2 * i} và ${2 * i + 1}, bốn trọng số: ${W.join(', ')}.`),
    },
    ...MASK.map((m, k): UnpackStep => ({
      op: 'mask', word: masked[k], width: 16,
      fields: [{ lo: LO[k], hi: LO[k] + 3, weight: W[k], value: codes[W[k]] }],
      out: [{ weight: W[k], value: masked[k], note: `${codes[W[k]]} × ${DIV[k]}` }],
      text: tr(`u & ${hex(m, 4)} = ${masked[k]}: weight ${W[k]}'s code ${codes[W[k]]}, left where it sits, so it counts ${DIV[k]} times over. No shift.`, `u & ${hex(m, 4)} = ${masked[k]}: mã ${codes[W[k]]} của trọng số ${W[k]}, để nguyên tại chỗ, nên nó được tính gấp ${DIV[k]} lần. Không dịch bit.`),
    })),
    {
      op: 'dot', word: u, width: 16, fields: [],
      out: W.map((w, k) => ({ weight: w, value: prods[k], note: `${masked[k]} × y[${w}]/${DIV[k]}` })),
      text: tr(`The activations were divided by 1, 256, 16 and 4096 when they were loaded, so each product is exactly code × activation: ${prods.map((p) => p.toPrecision(4)).join(', ')}. Sum ${acc.toPrecision(6)}; the 8 comes off later as 8 × (sum of these activations) = 8 × ${sumy.toPrecision(5)}.`, `Các activation đã được chia cho 1, 256, 16 và 4096 khi load, nên mỗi tích đúng bằng mã × activation: ${prods.map((p) => p.toPrecision(4)).join(', ')}. Tổng ${acc.toPrecision(6)}; số 8 được trừ sau dưới dạng 8 × (tổng các activation này) = 8 × ${sumy.toPrecision(5)}.`),
    },
  ];
}

// ---------------------------------------------------------------- one block's dot product, three ways (E4)
export interface BlockDot {
  /** Σ dequantised weight × activation, float32 left to right. */
  dequantised: number;
  /** d·Σ q·y - 8·d·Σ y, the folded form the kernels use (E4). */
  folded: number;
  /** Σ original float32 weight × activation. */
  exact: number;
  sumQY: number;
  sumY: number;
  d: number;
}
export function blockDotQ4_0(b: { dBits: number; codes: number[] }, y: number[], w: number[]): BlockDot {
  const d = fp16FromBits(b.dBits);
  const deq = dequantQ4_0(b);
  let dequantised = 0;
  let sumQY = 0;
  let sumY = 0;
  let exact = 0;
  for (let j = 0; j < 32; j++) {
    dequantised = f32(dequantised + f32(deq[j] * y[j]));
    sumQY = f32(sumQY + f32(b.codes[j] * y[j]));
    sumY = f32(sumY + y[j]);
    exact = f32(exact + f32(f32(w[j]) * y[j]));
  }
  const folded = f32(f32(d * sumQY) - f32(f32(8 * d) * sumY));
  return { dequantised, folded, exact, sumQY, sumY, d };
}

/** Seeded activations for a block (standard normal, float32). */
export const sceneActivations = (seed = 11) => sampleWeights(seed, 32, 'normal', 1);

// ---------------------------------------------------------------- generic matrix quantise-dequantise (qstep.ts uses it)
export type MatFormat = 'f32' | 'q8_0' | 'q4_0' | 'affine3' | 'affine2';
export const MAT_FORMAT_LABEL: Record<MatFormat, string> = {
  f32: 'F32',
  q8_0: 'Q8_0',
  q4_0: 'Q4_0',
  get affine3() {
    return tr('affine 3-bit, group 64', 'affine 3-bit, nhóm 64');
  },
  get affine2() {
    return tr('affine 2-bit, group 64', 'affine 2-bit, nhóm 64');
  },
};
/** Dequantised copy of a rows × cols matrix quantised along each row. Affine is MLX's documented rule with FP16 scale and bias. */
export function fakeQuant(w: number[], rows: number, cols: number, fmt: MatFormat): number[] {
  if (fmt === 'f32') return w.slice();
  const out = new Array<number>(w.length);
  for (let r = 0; r < rows; r++) {
    const row = w.slice(r * cols, (r + 1) * cols);
    let deq: number[];
    if (fmt === 'q8_0') deq = quantizeRowQ8_0(row).flatMap(dequantQ8_0);
    else if (fmt === 'q4_0') deq = quantizeRowQ4_0(row).flatMap(dequantQ4_0);
    else {
      const bits = fmt === 'affine3' ? 3 : 2;
      const g = Math.min(64, cols);
      const x = Float64Array.from(row);
      const o = new Float64Array(cols);
      for (let c = 0; c < cols; c += g) quantGroup(x, c, Math.min(cols, c + g), bits, 'affine', o);
      deq = Array.from(o, f32);
    }
    for (let c = 0; c < cols; c++) out[r * cols + c] = deq[c];
  }
  return out;
}

/** The lowest-error point whose bits per weight fit a budget, or null. */
export function bestUnder(points: SweepPoint[], budget: number): SweepPoint | null {
  let best: SweepPoint | null = null;
  for (const p of points) if (p.bpw <= budget + 1e-9 && (!best || p.rel < best.rel)) best = p;
  return best;
}
