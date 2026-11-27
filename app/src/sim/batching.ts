// The arithmetic of batching for chapter 22, section 22.5 (book pp. 539–540).
//
// What the book says (notes/ch22.md, §22.5): batching the queries of several users turns
// the projections' vector-matrix products into one matrix product. That product takes
// longer than one vector-matrix product but much less than all of them together, because
// each weight fetched serves every row. Its two throughput formulas are
//   with batching:    queries / (latency of the matrix product)
//   without batching: queries / (sum of the latencies of the vector-matrix products)
// and bookThroughput() is exactly those.
//
// Everything else here is ours, from notes/today/ch22-facts.md, group E:
//   E1  the intensity of one decode-step GEMM, I(B) = 2BKN / (w·KN + a·B·(K + N)) with w
//       bytes per weight and a bytes per activation, and the batch B* at which it reaches
//       a device threshold T (closed form and by search);
//   E2  the per-row pieces (bias, residual, softmax, argmax, embedding), whose intensity
//       does not grow with the batch;
//   E3  tile padding: a GEMM whose block tile is P rows tall computes ceil(B / P) · P rows
//       whatever B is (M2's tile is 16 rows, M3's is 128);
//   E4  the book's two formulas and the notes' 8-at-3x example.
//
// The decode step is lab 22.2's (labs/ch22/lab2_batching/decode.h): embed the tokens, then
// per layer one d → d linear layer, a bias add and a residual add, then an LM head tied to
// the embedding (d → vocab), a row softmax and an argmax. It is a list of kernels, each
// timed at the roofline bound of its own ideal traffic, time = max(bytes / bandwidth,
// operations / peak), one after another with no overlap between kernels. No launch cost,
// no caches, no attention (attention proper does not batch: ch22-facts A6). NVIDIA calls
// this comparison of intensity with a threshold "a simplified rule of thumb" (A4).
//
// Erasable TypeScript only, importing only lib/locale.ts: the browser checks import this file directly.

import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- devices
export interface Device {
  /** Peak arithmetic throughput, GFLOP/s. */
  peak: number;
  /** Peak global-memory bandwidth, GB/s. */
  bw: number;
  /**
   * The threshold (FLOP/B) as the source prints it, when the source rounds peak / bw. The
   * model then uses threshold × bw as the arithmetic peak, so that its knee and the
   * printed threshold agree.
   */
  threshold?: number;
}

/**
 * The book's example device (ch. 5 p. 94, notes/ch05.md): 66.9 TFLOPS, 3.35 TB/s, and the
 * threshold the book prints for it, 20.0 FLOP/B (66.9 / 3.35 = 19.97, rounded).
 * Used only as the book's example.
 */
export const BOOK_DEVICE: Device = { peak: 66900, bw: 3350, threshold: 20 };

/** The device's threshold in FLOP/B: where the roofline's slope meets its flat top. */
export function thresholdOf(dev: Device): number {
  return dev.threshold ?? dev.peak / dev.bw;
}

/** The arithmetic peak the model uses, GFLOP/s: threshold × bandwidth. */
export function effectivePeak(dev: Device): number {
  return thresholdOf(dev) * dev.bw;
}

// ---------------------------------------------------------------- precision and padding
/** Bytes per weight the explorer offers: float32, 16-bit, 8-bit and 4-bit weights. */
export type WeightBytes = 4 | 2 | 1 | 0.5;
export const WEIGHT_BYTES: WeightBytes[] = [4, 2, 1, 0.5];

export interface Precision {
  /** Bytes per weight. */
  weight: number;
  /** Bytes per activation (inputs, outputs, logits, bias). */
  act: number;
}

/**
 * The model's precision for a choice of weight bytes. Activations are float32 with
 * float32 weights and 16-bit otherwise (our choice for the model: 8- and 4-bit weights
 * with 16-bit activations).
 */
export function precisionFor(weight: WeightBytes): Precision {
  return { weight, act: weight === 4 ? 4 : 2 };
}

/** Rows a GEMM's block tile computes at once: 1 (no padding), M2's 16, M3's 128 (ch22-facts E3). */
export type Padding = 1 | 16 | 128;
export const PADDINGS: Padding[] = [1, 16, 128];

/** Rows a GEMM actually computes for a batch of B under padding P. */
export function paddedRows(B: number, pad: Padding): number {
  return pad > 1 ? Math.ceil(B / pad) * pad : B;
}

// ---------------------------------------------------------------- shapes
/** Lab 22.2's decode step: `layers` blocks of (d → d linear, bias, residual), then an LM head d → vocab tied to the embedding. */
export interface StepShape {
  kind: 'step';
  d: number;
  layers: number;
  vocab: number;
}
/** One projection, K inputs → N outputs, on its own. */
export interface ProjShape {
  kind: 'proj';
  k: number;
  n: number;
}
export type Shape = StepShape | ProjShape;

/**
 * The decode step the explorer assumes for lab 22.2 (M8) when its 'step' rows carry no
 * shape (fields d, layers, vocab): the driver's default (labs/ch22/lab2_batching/main.cpp,
 * bench(): vocab 32000, 2 layers, d = 4096, halved on a device with little memory). Keep
 * the two in step.
 */
export const M8_DEFAULT: StepShape = { kind: 'step', d: 4096, layers: 2, vocab: 32000 };

/** What lab 22.2 calls each GEMM a step can use, and where it comes from. */
export const M8_GEMMS: Record<string, string> = {
  get m2() {
    return tr("M2 (lab 6.4's tiled kernel)", 'M2 (kernel phân ô của lab 6.4)');
  },
  get m3() {
    return tr("M3 (lab 15.6's register-tiled kernel)", 'M3 (kernel phân ô thanh ghi của lab 15.6)');
  },
  get gemv() {
    return tr("lab 22.1's gather (batches up to 8)", 'phép gather của lab 22.1 (batch tối đa 8)');
  },
};
/** A 4096 → 4096 float projection: the section's worked example. */
export const PROJ_4096: ProjShape = { kind: 'proj', k: 4096, n: 4096 };
/** Chapter 20's example layer (p. 505) and the layer labs 5.3 and 15.6 time: 1024 → 4096. */
export const PROJ_1024: ProjShape = { kind: 'proj', k: 1024, n: 4096 };

/** The weight matrices one step multiplies by, as [K, N, how many per step]. */
export function gemmsOf(shape: Shape): [number, number, number][] {
  if (shape.kind === 'proj') return [[shape.k, shape.n, 1]];
  const { d, layers, vocab } = shape;
  return [
    [d, d, layers], // one linear layer per block
    [d, vocab, 1], // the LM head (its weight is the embedding table)
  ];
}

/** Bytes of weights one step's GEMMs stream: every layer's weight and the LM head's, once. */
export function weightBytes(shape: Shape, prec: Precision): number {
  return gemmsOf(shape).reduce((s, [k, n, c]) => s + prec.weight * k * n * c, 0);
}

// ---------------------------------------------------------------- one GEMM (E1)
/** Useful operations of Y[B×N] = X[B×K]·Wᵀ: 2·B·K·N. */
export const gemmFlops = (K: number, N: number, B: number) => 2 * B * K * N;

/** Minimum bytes: every weight, input and output once. */
export const gemmBytes = (K: number, N: number, B: number, prec: Precision) => prec.weight * K * N + prec.act * B * (K + N);

/** I(B) in FLOP/B (E1). */
export function intensity(K: number, N: number, B: number, prec: Precision): number {
  return gemmFlops(K, N, B) / gemmBytes(K, N, B, prec);
}

/** Where I(B) levels off as B grows: 2KN / (a·(K + N)). */
export function intensityLimit(K: number, N: number, prec: Precision): number {
  return (2 * K * N) / (prec.act * (K + N));
}

/**
 * B* for a set of GEMMs run in one step, closed form: the batch at which their combined
 * operations per byte reach T. Solving 2·B·ΣKN = T·(w·ΣKN + a·B·Σ(K + N)) gives
 * B* = T·w·ΣKN / (2·ΣKN − T·a·Σ(K + N)); null when the denominator is not positive (the
 * GEMMs never reach T, however large the batch). For one GEMM in float32 this is E1's
 * 2T / (1 − 2T(K + N)/(KN)).
 */
export function bStarOf(gemms: [number, number, number][], T: number, prec: Precision): number | null {
  let kn = 0;
  let kpn = 0;
  for (const [k, n, c] of gemms) {
    kn += k * n * c;
    kpn += (k + n) * c;
  }
  const denom = 2 * kn - T * prec.act * kpn;
  return denom > 0 ? (T * prec.weight * kn) / denom : null;
}

/** B* for one K → N projection. */
export function bStar(K: number, N: number, T: number, prec: Precision): number | null {
  return bStarOf([[K, N, 1]], T, prec);
}

/** The smallest whole batch whose I(B) reaches T, by search; null if none up to `max`. */
export function bStarSearch(K: number, N: number, T: number, prec: Precision, max = 1 << 24): number | null {
  if (intensity(K, N, max, prec) < T) return null;
  let lo = 1;
  let hi = max;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (intensity(K, N, mid, prec) >= T) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

// ---------------------------------------------------------------- the per-row pieces (E2)
/** Bias add over a B×N output: B·N operations; read Y, write Y, read the bias once. */
export function biasIntensity(B: number, N: number, prec: Precision): number {
  return (B * N) / (prec.act * (2 * B * N + N));
}
/** Residual add: read two, write one, one addition per element. 1/12 FLOP/B in float32. */
export function residualIntensity(prec: Precision): number {
  return 1 / (3 * prec.act);
}
/** Operations per element of a row softmax in this model: scale, max, subtract, exponential, sum, divide, rounded down to 5 (ours; E2 only needs it constant). */
export const SOFTMAX_OPS = 5;

// ---------------------------------------------------------------- the decode step
export type PieceKey = 'proj' | 'head' | 'rowwise' | 'softmax' | 'argmax' | 'embed';
export const PIECES: { key: PieceKey; label: string; perRow: boolean }[] = [
  { key: 'proj', get label() { return tr('layer projections', 'các phép chiếu của lớp'); }, perRow: false },
  { key: 'head', get label() { return tr('LM head', 'LM head'); }, perRow: false },
  { key: 'rowwise', get label() { return tr('bias and residual adds', 'cộng bias và residual'); }, perRow: true },
  { key: 'softmax', get label() { return tr('softmax over the vocabulary', 'softmax trên từ vựng'); }, perRow: true },
  { key: 'argmax', get label() { return tr('argmax', 'argmax'); }, perRow: true },
  { key: 'embed', get label() { return tr('embedding lookup', 'tra cứu embedding'); }, perRow: true },
];

export interface Kernel {
  piece: PieceKey;
  /** Useful operations. */
  flops: number;
  /** Operations actually executed (padding included). */
  work: number;
  /** Minimum bytes moved. */
  bytes: number;
  /** How many times the step runs it. */
  count: number;
}

/** Every kernel of one step at batch B. A projection shape has only its one GEMM. */
export function stepKernels(shape: Shape, B: number, prec: Precision, pad: Padding): Kernel[] {
  const gemm = (piece: PieceKey, K: number, N: number, count: number): Kernel => ({
    piece,
    flops: gemmFlops(K, N, B),
    work: gemmFlops(K, N, paddedRows(B, pad)),
    bytes: gemmBytes(K, N, B, prec),
    count,
  });
  if (shape.kind === 'proj') return [gemm('proj', shape.k, shape.n, 1)];
  const { d, layers, vocab } = shape;
  const a = prec.act;
  return [
    gemm('proj', d, d, layers), // T = H W_l^T
    gemm('head', d, vocab, 1), // logits = H E^T
    { piece: 'rowwise', flops: B * d, work: B * d, bytes: a * (2 * B * d + d), count: layers }, // T += b_l: read T, write T, read b_l once
    { piece: 'rowwise', flops: B * d, work: B * d, bytes: 3 * a * B * d, count: layers }, // H += T: read two, write one
    { piece: 'softmax', flops: SOFTMAX_OPS * B * vocab, work: SOFTMAX_OPS * B * vocab, bytes: 2 * a * B * vocab, count: 1 }, // read logits, write probs
    { piece: 'argmax', flops: B * vocab, work: B * vocab, bytes: a * B * vocab + 4 * B, count: 1 }, // read logits, write a token id per row
    { piece: 'embed', flops: 0, work: 0, bytes: (prec.weight + a) * B * d + 4 * B, count: 1 }, // read B token ids and B rows of E, write H
  ];
}

export interface PieceTime {
  key: PieceKey;
  label: string;
  perRow: boolean;
  seconds: number;
  /** Seconds the piece would take on bandwidth alone, and on arithmetic alone. */
  memory: number;
  compute: number;
  bound: 'memory' | 'compute';
}

export interface StepResult {
  batch: number;
  pieces: PieceTime[];
  /** Time per step: every kernel at its own roofline bound, one after another. */
  seconds: number;
  /** B / seconds: one token per request per step. */
  tokensPerSecond: number;
  /** What each user waits between tokens: one step. */
  perToken: number;
  /** The GEMMs (projections and LM head): traffic time, arithmetic time and their roofline time. */
  gemmMemory: number;
  gemmCompute: number;
  gemmSeconds: number;
  /** The per-row pieces together. */
  rowSeconds: number;
  /** Useful operations per byte of the GEMMs together at this batch. */
  intensity: number;
}

/** Time of one kernel at its roofline bound, all its repetitions. */
function kernelTimes(k: Kernel, dev: Device): { memory: number; compute: number; seconds: number } {
  const memory = (k.bytes / (dev.bw * 1e9)) * k.count;
  const compute = (k.work / (effectivePeak(dev) * 1e9)) * k.count;
  return { memory, compute, seconds: Math.max(memory, compute) };
}

/** One decode step at batch B on a device. */
export function stepAt(shape: Shape, B: number, prec: Precision, pad: Padding, dev: Device): StepResult {
  const ks = stepKernels(shape, B, prec, pad);
  const pieces: PieceTime[] = [];
  let gemmMemory = 0;
  let gemmCompute = 0;
  let gemmSeconds = 0;
  let rowSeconds = 0;
  let gflops = 0;
  let gbytes = 0;
  for (const p of PIECES) {
    const mine = ks.filter((k) => k.piece === p.key);
    if (!mine.length) continue;
    let memory = 0;
    let compute = 0;
    let seconds = 0;
    for (const k of mine) {
      const t = kernelTimes(k, dev);
      memory += t.memory;
      compute += t.compute;
      seconds += t.seconds;
      if (!p.perRow) {
        gflops += k.flops * k.count;
        gbytes += k.bytes * k.count;
      }
    }
    pieces.push({ key: p.key, label: p.label, perRow: p.perRow, seconds, memory, compute, bound: compute > memory ? 'compute' : 'memory' });
    if (p.perRow) rowSeconds += seconds;
    else {
      gemmMemory += memory;
      gemmCompute += compute;
      gemmSeconds += seconds;
    }
  }
  const seconds = gemmSeconds + rowSeconds;
  return {
    batch: B,
    pieces,
    seconds,
    tokensPerSecond: B / seconds,
    perToken: seconds,
    gemmMemory,
    gemmCompute,
    gemmSeconds,
    rowSeconds,
    intensity: gflops / gbytes,
  };
}

/** The step's B*: where its GEMMs together reach the device threshold (padding ignored). */
export function stepBStar(shape: Shape, prec: Precision, dev: Device): number | null {
  return bStarOf(gemmsOf(shape), thresholdOf(dev), prec);
}

/** Batches the explorer plots: every whole batch from 1 to max (so tile steps show). */
export function batchRange(max = 1024): number[] {
  return Array.from({ length: max }, (_, i) => i + 1);
}

/** Step results for every batch in `batches`. */
export function stepCurve(shape: Shape, prec: Precision, pad: Padding, dev: Device, batches = batchRange()): StepResult[] {
  return batches.map((B) => stepAt(shape, B, prec, pad, dev));
}

/**
 * The step in relative units, for drawings that must not show device timings: time is
 * measured in passes over the weights (one unit = the time to stream every layer's weight
 * and the LM head's once at the device's bandwidth), so only the threshold T matters.
 */
export function relativeStep(shape: Shape, B: number, prec: Precision, pad: Padding, T: number) {
  const r = stepAt(shape, B, prec, pad, { peak: T, bw: 1 });
  const unit = weightBytes(shape, prec) / 1e9;
  const rel = (s: number) => s / unit;
  return {
    batch: B,
    /** Weights and activations of the GEMMs streaming in, alone. */
    memory: rel(r.gemmMemory),
    /** The GEMMs' arithmetic alone. */
    compute: rel(r.gemmCompute),
    /** The GEMMs at their roofline bound. */
    gemm: rel(r.gemmSeconds),
    /** Bias, residual, softmax, argmax and embedding together. */
    row: rel(r.rowSeconds),
    step: rel(r.seconds),
    tokensPerUnit: B / rel(r.seconds),
  };
}

// ---------------------------------------------------------------- the book's formulas (E4)
/**
 * p. 539's two throughputs for `queries` requests: one matrix product taking `gemm`
 * seconds, against one vector-matrix product per query taking `gemv` seconds each.
 */
export function bookThroughput(queries: number, gemm: number, gemv: number) {
  const batched = queries / gemm;
  const separate = queries / (queries * gemv);
  return {
    batched,
    separate,
    /** How many times more queries per second batching gives. */
    gain: batched / separate,
    /** How many times longer each query's own projection takes. */
    latencyFactor: gemm / gemv,
  };
}

// ---------------------------------------------------------------- measured knees
export interface Point {
  batch: number;
  seconds: number;
}

/**
 * The first measured batch at which time per step has grown to `factor` times its value
 * at the smallest batch measured; null when no batch gets there (no knee in the range) or
 * there are fewer than two points.
 */
export function kneeOf(rows: Point[], factor = 1.5): number | null {
  const pts = rows.filter((r) => isFinite(r.seconds) && r.seconds > 0).sort((a, b) => a.batch - b.batch);
  if (pts.length < 2) return null;
  const base = pts[0].seconds;
  const hit = pts.slice(1).find((r) => r.seconds >= factor * base);
  return hit ? hit.batch : null;
}

// ---------------------------------------------------------------- reading the reader's reports
// The shapes of reports/runs/<backend>.json that this file needs (lib/reports.ts has the
// full types; this file imports only lib/locale.ts).
export interface RowLike {
  name: string;
  [k: string]: string | number;
}
export interface RunLike {
  generated?: string;
  device?: Record<string, unknown>;
  results: { lab: string; variant: string; bench: RowLike[] }[];
}
export type ReportsLike = Record<string, RunLike>;

export const LAB_M8 = 'ch22/lab2_batching';
export const LAB_ROOFLINE = 'ch05/lab1_roofline';
export const LAB_GEMM = 'ch05/lab3_gemm';
export const LAB_LINEAR = 'ch15/lab6_linear';

const numOf = (v: unknown): number | undefined => (typeof v === 'number' && isFinite(v) ? v : undefined);
/** Backends that ran on a GPU: neither emulator counts. */
export const gpuBackends = (r: ReportsLike) => Object.keys(r).filter((b) => b !== 'emu' && b !== 'race');
function solutionRows(run: RunLike | undefined, lab: string): RowLike[] {
  return run ? run.results.filter((x) => x.lab === lab && x.variant === 'solution').flatMap((x) => x.bench) : [];
}

export interface MeasuredDevice extends Device {
  backend: string;
  name: string;
  date: string;
}

/** Each GPU backend's peak and bandwidth as lab 5.1 measured them ('roofline' row). */
export function measuredDevices(reports: ReportsLike): MeasuredDevice[] {
  return gpuBackends(reports).flatMap((b) => {
    const r = solutionRows(reports[b], LAB_ROOFLINE).find((x) => x.name === 'roofline');
    const peak = numOf(r?.compute_gflops);
    const bw = numOf(r?.bandwidth_gb_per_s);
    if (!peak || !bw) return [];
    return [{ backend: b, name: String(reports[b].device?.name ?? b), date: String(reports[b].generated ?? '').slice(0, 10), peak, bw }];
  });
}

export interface Series {
  key: string;
  /** Which lab, which kernel. */
  label: string;
  backend: string;
  points: Point[];
  /** kneeOf(points, 1.5). */
  knee: number | null;
}

const seconds = (r: RowLike): number | undefined => numOf(r.seconds) ?? (numOf(r.ms) != null ? numOf(r.ms)! / 1e3 : undefined);

function group(backend: string, rows: { kernel: string; batch: number; seconds: number }[], label: (k: string) => string, prefix: string): Series[] {
  const kernels = [...new Set(rows.map((r) => r.kernel))];
  return kernels.map((k) => {
    const points = rows
      .filter((r) => r.kernel === k)
      .map((r) => ({ batch: r.batch, seconds: r.seconds }))
      .sort((a, b) => a.batch - b.batch);
    return { key: `${backend}:${prefix}:${k}`, label: label(k), backend, points, knee: kneeOf(points) };
  });
}

/** Lab 22.2's 'step' rows on one backend, one series per GEMM kernel. */
export function stepSeries(reports: ReportsLike, backend: string): Series[] {
  const rows = solutionRows(reports[backend], LAB_M8)
    .filter((r) => r.name === 'step')
    .flatMap((r) => {
      const batch = numOf(r.batch);
      const s = seconds(r);
      if (!batch || !s) return [];
      return [{ kernel: String(r.gemm ?? r.kernel ?? 'gemm'), batch, seconds: s }];
    });
  return group(backend, rows, (k) => tr(`lab 22.2, ${M8_GEMMS[k] ?? k}`, `lab 22.2, ${M8_GEMMS[k] ?? k}`), 'm8');
}

/** Lab 5.3's 'linear_layer' rows and lab 15.6's kernel rows for a K → N layer, one series per kernel. */
export function projectionSeries(reports: ReportsLike, backend: string, K: number, N: number): Series[] {
  const run = reports[backend];
  const l53 = solutionRows(run, LAB_GEMM)
    .filter((r) => r.name === 'linear_layer' && numOf(r.k) === K && numOf(r.l) === N)
    .flatMap((r) => (numOf(r.batch) && seconds(r) ? [{ kernel: 'tiled', batch: numOf(r.batch)!, seconds: seconds(r)! }] : []));
  const l156 = solutionRows(run, LAB_LINEAR)
    .filter((r) => numOf(r.in) === K && numOf(r.out) === N && typeof r.kernel === 'string')
    .flatMap((r) => (numOf(r.batch) && seconds(r) ? [{ kernel: String(r.kernel), batch: numOf(r.batch)!, seconds: seconds(r)! }] : []));
  return [
    ...group(backend, l53, () => tr('lab 5.3, tiled (M2)', 'lab 5.3, phân ô (M2)'), 'l53'),
    ...group(backend, l156, (k) => tr(`lab 15.6, ${k === 'vendor' ? 'vendor library' : k.toUpperCase()}`, `lab 15.6, ${k === 'vendor' ? 'thư viện của hãng' : k.toUpperCase()}`), 'l156'),
  ];
}

/**
 * The decode step lab 22.2 timed on a backend, when its 'step' rows say (fields d, layers,
 * vocab); otherwise M8_DEFAULT.
 */
export function m8Shape(reports: ReportsLike, backend?: string): StepShape {
  const backends = backend ? [backend] : gpuBackends(reports);
  for (const b of backends) {
    const r = solutionRows(reports[b], LAB_M8).find((x) => x.name === 'step');
    const d = numOf(r?.d);
    const layers = numOf(r?.layers);
    const vocab = numOf(r?.vocab);
    if (d && layers && vocab) return { kind: 'step', d, layers, vocab };
  }
  return M8_DEFAULT;
}
