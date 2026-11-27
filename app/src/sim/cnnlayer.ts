// Chapter 19, sections 19.1 and 19.2: the convolutional layer as the book defines it.
//
// What it models (notes/ch19.md):
//  - The forward path of one layer, Figs. 19.3 and 19.4 (pp. 457, 459):
//    Y[n, m, h, w] = sum over c, p, q of X[n, c, h + p, w + q] * F[m, c, p, q],
//    with no ghost cells (H_out = H - K + 1, p. 457), no activation (p. 456) and the
//    filter not flipped. All arrays are flat and row-major, highest dimension first (p. 458).
//  - The book's example of Fig. 19.2 (p. 456): three 3 x 3 input maps, two output maps,
//    six 2 x 2 filters. Its printed outputs (14 20 15 24 / 12 24 17 26) are what the
//    loop nest gives; the tests pin that.
//  - A per-term trace of one output (c, p, q, the two operands, the running sum), with an
//    optional 180-degree rotation of every filter. The rotation is ours, not the book's: it
//    turns the book's formula (cross-correlation, which PyTorch, cuDNN and MIOpen compute:
//    notes/today/ch07-facts.md D1) into a true convolution, to show the two differ.
//  - LeNet-5's three convolution layers C1, C3 and C5 (p. 458): filters, weights,
//    multiply-adds, and the ideal traffic of a batch (every tensor touched once).
//  - The roles of the seven loops of Fig. 19.4 (p. 460).
//  - What the direct kernel of Fig. 19.7 fetches from global memory, per thread and per
//    block (section 19.2, p. 463, and the derived counts in the notes).
//
// What it leaves out:
//  - No timing, no caches: "reads" are what a kernel asks of global memory, not what DRAM
//    delivers. A cache may serve repeated reads (the lesson says so where counts appear).
//  - Integers, as the book uses for readability (footnote, p. 457).
//
// Erasable TypeScript only, importing only lib/locale.ts: the browser checks import this file directly.

import { tr } from '../lib/locale.ts';

export interface LayerShape {
  /** Input feature maps (channels). */
  C: number;
  /** Input height and width. */
  H: number;
  W: number;
  /** Output feature maps. */
  M: number;
  /** Filter side (square filters). */
  K: number;
}

/** No ghost cells: the output loses K - 1 rows and columns (p. 457). */
export const outDim = (size: number, K: number): number => size - K + 1;
export const hOut = (s: LayerShape): number => outDim(s.H, s.K);
export const wOut = (s: LayerShape): number => outDim(s.W, s.K);

// ---------------------------------------------------------------- the book's example

/** Fig. 19.2: C = 3 input maps of 3 x 3, M = 2 output maps, 2 x 2 filters. */
export const BOOK_LAYER: LayerShape = { C: 3, H: 3, W: 3, M: 2, K: 2 };
/** X[c][h][w], flat. X0 = [1 2 0; 1 1 3; 0 2 2], X1 = [0 2 1; 0 3 2; 1 1 0], X2 = [1 2 1; 0 1 3; 3 3 2]. */
export const BOOK_X: readonly number[] = [
  1, 2, 0, 1, 1, 3, 0, 2, 2,
  0, 2, 1, 0, 3, 2, 1, 1, 0,
  1, 2, 1, 0, 1, 3, 3, 3, 2,
];
/** F[m][c][p][q], flat. Output 0: [1 1; 2 2], [1 1; 1 1], [0 1; 1 0]. Output 1: [1 0; 0 1], [2 1; 2 1], [1 2; 2 0]. */
export const BOOK_F: readonly number[] = [
  1, 1, 2, 2, 1, 1, 1, 1, 0, 1, 1, 0,
  1, 0, 0, 1, 2, 1, 2, 1, 1, 2, 2, 0,
];
/** The outputs as printed in Fig. 19.2 (and again in Fig. 19.8): Y0 = [14 20; 15 24], Y1 = [12 24; 17 26]. */
export const BOOK_Y_PRINTED: readonly number[] = [14, 20, 15, 24, 12, 24, 17, 26];

// ---------------------------------------------------------------- flat indices

/** X[c, h, w] for one sample, row-major. */
export const xAt = (s: LayerShape, c: number, h: number, w: number): number => (c * s.H + h) * s.W + w;
/** X[n, c, h, w] for a batch, row-major. */
export const xAtN = (s: LayerShape, n: number, c: number, h: number, w: number): number => ((n * s.C + c) * s.H + h) * s.W + w;
/** F[m, c, p, q], row-major. */
export const fAt = (s: LayerShape, m: number, c: number, p: number, q: number): number => ((m * s.C + c) * s.K + p) * s.K + q;
/** Y[m, h, w] for one sample, row-major. */
export const yAt = (s: LayerShape, m: number, h: number, w: number): number => (m * hOut(s) + h) * wOut(s) + w;
/** Y[n, m, h, w] for a batch, row-major. */
export const yAtN = (s: LayerShape, n: number, m: number, h: number, w: number): number => ((n * s.M + m) * hOut(s) + h) * wOut(s) + w;

// ---------------------------------------------------------------- the loop nest

export interface Term {
  /** Position in the c, p, q loop, from 0. */
  k: number;
  c: number;
  p: number;
  q: number;
  /** Flat index of the input value read: X[c, h + p, w + q]. */
  xi: number;
  /** Flat index of the filter weight read. With flip, F[m, c, K-1-p, K-1-q]. */
  fi: number;
  x: number;
  f: number;
  product: number;
  /** Running sum after this term. */
  sum: number;
}

/**
 * Every multiply-add of one output Y[m, h, w], in the order of Fig. 19.3's c, p, q loops.
 * With `flip`, each filter is rotated 180 degrees first (a true convolution; ours, not the book's).
 */
export function terms(s: LayerShape, X: readonly number[], F: readonly number[], m: number, h: number, w: number, flip = false): Term[] {
  const out: Term[] = [];
  let sum = 0;
  let k = 0;
  for (let c = 0; c < s.C; c++)
    for (let p = 0; p < s.K; p++)
      for (let q = 0; q < s.K; q++) {
        const xi = xAt(s, c, h + p, w + q);
        const fi = flip ? fAt(s, m, c, s.K - 1 - p, s.K - 1 - q) : fAt(s, m, c, p, q);
        const product = X[xi] * F[fi];
        sum += product;
        out.push({ k: k++, c, p, q, xi, fi, x: X[xi], f: F[fi], product, sum });
      }
  return out;
}

/** One output, Y[m, h, w]. */
export function output(s: LayerShape, X: readonly number[], F: readonly number[], m: number, h: number, w: number, flip = false): number {
  const t = terms(s, X, F, m, h, w, flip);
  return t.length ? t[t.length - 1].sum : 0;
}

/** Fig. 19.3: the whole layer for one sample, Y as a flat M x H_out x W_out array. */
export function forward(s: LayerShape, X: readonly number[], F: readonly number[], flip = false): number[] {
  const Y: number[] = [];
  for (let m = 0; m < s.M; m++)
    for (let h = 0; h < hOut(s); h++)
      for (let w = 0; w < wOut(s); w++) Y.push(output(s, X, F, m, h, w, flip));
  return Y;
}

/** Fig. 19.4: a batch of N samples. X is N x C x H x W, Y is N x M x H_out x W_out; F is shared. */
export function forwardBatched(s: LayerShape, N: number, X: readonly number[], F: readonly number[]): number[] {
  const per = s.C * s.H * s.W;
  const Y: number[] = [];
  for (let n = 0; n < N; n++) Y.push(...forward(s, X.slice(n * per, (n + 1) * per), F));
  return Y;
}

/** Per output contribution of each input map: the three numbers that add up to one output (Eq. 19.1). */
export function perInputMap(s: LayerShape, X: readonly number[], F: readonly number[], m: number, h: number, w: number, flip = false): number[] {
  const t = terms(s, X, F, m, h, w, flip);
  return Array.from({ length: s.C }, (_, c) => t.filter((x) => x.c === c).reduce((a, x) => a + x.product, 0));
}

/** Which outputs change when every filter is rotated 180 degrees: flat Y indices. */
export function changedByFlip(s: LayerShape, X: readonly number[], F: readonly number[]): number[] {
  const a = forward(s, X, F);
  const b = forward(s, X, F, true);
  return a.flatMap((v, i) => (v === b[i] ? [] : [i]));
}

/** True when rotating this filter by 180 degrees leaves it unchanged. */
export function symmetricUnderRotation(s: LayerShape, F: readonly number[], m: number, c: number): boolean {
  for (let p = 0; p < s.K; p++)
    for (let q = 0; q < s.K; q++) if (F[fAt(s, m, c, p, q)] !== F[fAt(s, m, c, s.K - 1 - p, s.K - 1 - q)]) return false;
  return true;
}

// ---------------------------------------------------------------- shapes and LeNet-5

export interface Shapes {
  X: number[];
  F: number[];
  Y: number[];
  /** One K x K filter per (output map, input map) pair (p. 456). */
  filters: number;
  weights: number;
  /** Multiply-adds for one sample: M * H_out * W_out * C * K * K. */
  macsPerSample: number;
}

/** Shapes of X, F and Y, with a leading N when a batch size is given (p. 459: F has none). */
export function shapes(s: LayerShape, N?: number): Shapes {
  const lead = N == null ? [] : [N];
  return {
    X: [...lead, s.C, s.H, s.W],
    F: [s.M, s.C, s.K, s.K],
    Y: [...lead, s.M, hOut(s), wOut(s)],
    filters: s.M * s.C,
    weights: s.M * s.C * s.K * s.K,
    macsPerSample: s.M * hOut(s) * wOut(s) * s.C * s.K * s.K,
  };
}

export interface BatchTraffic {
  N: number;
  macs: number;
  /** Ideal bytes: every element of X, F and Y moved once for the whole batch. */
  xBytes: number;
  fBytes: number;
  yBytes: number;
  total: number;
  /** F's share of the ideal bytes. */
  fShare: number;
  /** F's bytes divided among the samples of the batch. */
  fBytesPerSample: number;
  /** 2 FLOP per multiply-add over the ideal bytes. */
  flopPerByte: number;
}

/** Ideal traffic of one layer for a batch of N (every tensor touched exactly once). */
export function batchTraffic(s: LayerShape, N: number, bytesPerValue = 4): BatchTraffic {
  const sh = shapes(s);
  const xBytes = N * s.C * s.H * s.W * bytesPerValue;
  const fBytes = sh.weights * bytesPerValue;
  const yBytes = N * s.M * hOut(s) * wOut(s) * bytesPerValue;
  const total = xBytes + fBytes + yBytes;
  const macs = N * sh.macsPerSample;
  return { N, macs, xBytes, fBytes, yBytes, total, fShare: fBytes / total, fBytesPerSample: fBytes / N, flopPerByte: (2 * macs) / total };
}

export type StageKind = 'input' | 'conv' | 'subsampling' | 'full' | 'gaussian';
export interface Stage {
  id: string;
  kind: StageKind;
  /** What the stage produces, as Fig. 19.1 labels it. */
  out: string;
  /** Only the convolution layers are computed here. */
  layer?: LayerShape;
}

/** Fig. 19.1 (p. 455) with the layer sizes of p. 458. */
export const LENET: readonly Stage[] = [
  { id: 'INPUT', kind: 'input', out: '1 × 32 × 32' },
  { id: 'C1', kind: 'conv', out: '6 × 28 × 28', layer: { C: 1, H: 32, W: 32, M: 6, K: 5 } },
  { id: 'S2', kind: 'subsampling', out: '6 × 14 × 14' },
  { id: 'C3', kind: 'conv', out: '16 × 10 × 10', layer: { C: 6, H: 14, W: 14, M: 16, K: 5 } },
  { id: 'S4', kind: 'subsampling', out: '16 × 5 × 5' },
  { id: 'C5', kind: 'conv', out: '120 × 1 × 1', layer: { C: 16, H: 5, W: 5, M: 120, K: 5 } },
  { id: 'F6', kind: 'full', out: '84' },
  { id: 'OUTPUT', kind: 'gaussian', out: '10' },
];
export const lenetLayer = (id: string): LayerShape | undefined => LENET.find((x) => x.id === id)?.layer;

// ---------------------------------------------------------------- the loops of Fig. 19.4

export type LoopVar = 'n' | 'm' | 'h' | 'w' | 'c' | 'p' | 'q';
export type LoopRole = 'independent' | 'accumulates';
export interface Loop {
  v: LoopVar;
  /** Line of Fig. 19.4 that opens the loop. */
  line: number;
  over: string;
  role: LoopRole;
}

/** p. 460: n, m, h and w write different Y elements; c, p and q all add into one. */
export const FIG_19_4_LOOPS: readonly Loop[] = [
  { v: 'n', line: 4, get over() { return tr('samples in the batch', 'các mẫu trong batch'); }, role: 'independent' },
  { v: 'm', line: 5, get over() { return tr('output feature maps', 'các bản đồ đặc trưng đầu ra'); }, role: 'independent' },
  { v: 'h', line: 6, get over() { return tr('output rows', 'các hàng đầu ra'); }, role: 'independent' },
  { v: 'w', line: 7, get over() { return tr('output columns', 'các cột đầu ra'); }, role: 'independent' },
  { v: 'c', line: 9, get over() { return tr('input feature maps', 'các bản đồ đặc trưng đầu vào'); }, role: 'accumulates' },
  { v: 'p', line: 10, get over() { return tr('filter rows', 'các hàng của bộ lọc'); }, role: 'accumulates' },
  { v: 'q', line: 11, get over() { return tr('filter columns', 'các cột của bộ lọc'); }, role: 'accumulates' },
];

/** Total independent iterations (p. 460): N * M * H_out * W_out. */
export const parallelism = (s: LayerShape, N: number): number => N * s.M * hOut(s) * wOut(s);

/** Grades a classification of the seven loops; returns the loops marked wrongly. */
export function gradeLoops(marks: Partial<Record<LoopVar, LoopRole>>): LoopVar[] {
  return FIG_19_4_LOOPS.filter((l) => marks[l.v] !== l.role).map((l) => l.v);
}

// ---------------------------------------------------------------- what the direct kernel reads

export interface DirectRead {
  /** Block of the grid: x = output map m, y = tile, z = sample n (Fig. 19.6). */
  bx: number;
  by: number;
  bz: number;
  tx: number;
  ty: number;
  /** Thread number inside the block, row-major (ty * tile + tx). */
  thread: number;
  h: number;
  w: number;
  /** Step of the thread's c, p, q loop. */
  step: number;
  c: number;
  p: number;
  q: number;
  /** Flat indices into X (with the sample) and F. */
  xi: number;
  fi: number;
  x: number;
  f: number;
  /** The thread's register sum after this step. */
  sum: number;
}

export interface DirectBlock {
  bx: number;
  by: number;
  bz: number;
  reads: DirectRead[];
  /** How many times each element of X is fetched by this block. */
  xHeat: number[];
  xReads: number;
  fReads: number;
  distinctX: number;
  distinctF: number;
  /** Distinct F addresses per step: 1 means every thread of the block read the same weight. */
  fAddressesPerStep: number[];
}

export interface DirectTrace {
  tile: number;
  grid: { x: number; y: number; z: number };
  blocks: DirectBlock[];
  /** Fetches of each X element over the whole grid. */
  xHeat: number[];
  xReads: number;
  fReads: number;
  distinctX: number;
  /** Values the kernel stores, flat N x M x H_out x W_out. */
  Y: number[];
  flops: number;
  /** Bytes requested from global memory by the loads (4 bytes each); the stores are not counted. */
  loadBytes: number;
  /** flops / loadBytes: what the kernel asks of memory per operation. */
  ratio: number;
}

/**
 * Every global load of the direct kernel (Fig. 19.7, with bounds checks) for a batch of N,
 * thread by thread. The grid is Fig. 19.5's with ceiling division: x = M, y = tiles per map,
 * z = N. Threads past the edge of the map read nothing.
 */
export function directTrace(s: LayerShape, X: readonly number[], F: readonly number[], tile: number, N = 1): DirectTrace {
  const Ho = hOut(s);
  const Wo = wOut(s);
  const Wgrid = Math.ceil(Wo / tile);
  const Hgrid = Math.ceil(Ho / tile);
  const grid = { x: s.M, y: Wgrid * Hgrid, z: N };
  const blocks: DirectBlock[] = [];
  const xHeat = new Array<number>(X.length).fill(0);
  const Y = new Array<number>(N * s.M * Ho * Wo).fill(0);
  for (let bz = 0; bz < grid.z; bz++)
    for (let by = 0; by < grid.y; by++)
      for (let bx = 0; bx < grid.x; bx++) {
        const reads: DirectRead[] = [];
        const heat = new Array<number>(X.length).fill(0);
        const steps = s.C * s.K * s.K;
        for (let ty = 0; ty < tile; ty++)
          for (let tx = 0; tx < tile; tx++) {
            const h = Math.floor(by / Wgrid) * tile + ty;
            const w = (by % Wgrid) * tile + tx;
            if (h >= Ho || w >= Wo) continue;
            let sum = 0;
            let step = 0;
            for (let c = 0; c < s.C; c++)
              for (let p = 0; p < s.K; p++)
                for (let q = 0; q < s.K; q++) {
                  const xi = xAtN(s, bz, c, h + p, w + q);
                  const fi = fAt(s, bx, c, p, q);
                  sum += X[xi] * F[fi];
                  heat[xi]++;
                  xHeat[xi]++;
                  reads.push({ bx, by, bz, tx, ty, thread: ty * tile + tx, h, w, step: step++, c, p, q, xi, fi, x: X[xi], f: F[fi], sum });
                }
            Y[yAtN(s, bz, bx, h, w)] = sum;
          }
        const fAddressesPerStep = Array.from({ length: steps }, (_, k) => new Set(reads.filter((r) => r.step === k).map((r) => r.fi)).size);
        blocks.push({
          bx, by, bz, reads, xHeat: heat,
          xReads: reads.length,
          fReads: reads.length,
          distinctX: new Set(reads.map((r) => r.xi)).size,
          distinctF: new Set(reads.map((r) => r.fi)).size,
          fAddressesPerStep,
        });
      }
  const xReads = blocks.reduce((a, b) => a + b.xReads, 0);
  const fReads = blocks.reduce((a, b) => a + b.fReads, 0);
  const flops = 2 * xReads;
  const loadBytes = 4 * (xReads + fReads);
  return {
    tile, grid, blocks, xHeat, xReads, fReads,
    distinctX: xHeat.filter((v) => v > 0).length,
    Y, flops, loadBytes, ratio: flops / loadBytes,
  };
}

/** The reads of one block at one step of the c, p, q loop, one per thread. */
export const readsAtStep = (b: DirectBlock, step: number): DirectRead[] => b.reads.filter((r) => r.step === step);
/** The reads of one thread of one block, in loop order. */
export const readsOfThread = (b: DirectBlock, thread: number): DirectRead[] => b.reads.filter((r) => r.thread === thread);

/** Global loads per output of the direct kernel: one X and one F per multiply-add (derived, notes §19.2). */
export const directLoadsPerOutput = (s: LayerShape): number => 2 * s.C * s.K * s.K;
/** FLOP per byte the direct kernel requests: 2 FLOP per multiply-add over 8 bytes of loads. */
export const directRatio = (s: LayerShape): number => (2 * s.C * s.K * s.K) / (4 * directLoadsPerOutput(s));
/**
 * The same ratio when each thread computes `maps` output maps at one pixel and loads each X
 * value once for all of them: C·K² X loads plus maps·C·K² F loads for maps·C·K² multiply-adds.
 * Ours (a variation for a question); with maps = 1 it is directRatio.
 */
export const directRatioMaps = (s: LayerShape, maps: number): number =>
  (2 * maps * s.C * s.K * s.K) / (4 * (s.C * s.K * s.K * (1 + maps)));

// ---------------------------------------------------------------- lesson regions of lab files

/**
 * The part of a lab file between `[lesson:<name>]` and `[/lesson:<name>]` marker comments.
 * Several regions of the same name are joined with a blank line. Marker lines of any
 * region are dropped. With no complete region, the whole file comes back and `found` is false.
 */
export function lessonRegion(text: string, name: string): { code: string; found: boolean } {
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  const open = `[lesson:${name}]`; // i18n-ignore: marker searched for in lab source, not reader text
  const close = `[/lesson:${name}]`;
  const parts: string[][] = [];
  let at = -1;
  lines.forEach((l, k) => {
    if (at < 0 && l.includes(open)) at = k;
    else if (at >= 0 && l.includes(close)) {
      parts.push(lines.slice(at + 1, k));
      at = -1;
    }
  });
  if (!parts.length) return { code: text.replace(/\s+$/, ''), found: false };
  const marker = /\[\/?lesson:[\w.-]+\]/;
  const code = parts
    .map((p) => p.filter((l) => !marker.test(l)).join('\n').replace(/^\s*\n/, '').replace(/\s+$/, ''))
    .join('\n\n');
  return { code, found: true };
}
