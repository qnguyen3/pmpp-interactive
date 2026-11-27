// Row reductions as the real backends write them (module X3, section X3.3). Not in the book:
// the kernels compared here are read in notes/today/x3-facts.md (B2, B3) and
// notes/today/ch10-facts.md (C4, D1 to D4) at the commits listed there.
//
// What it models:
//   - One softmax row in float32 (every operation rounded with Math.fround), computed two ways:
//       'twopass' max, then sum of exponentials, then the write, as three loops over the row
//                 (lab 10.3's softmaxRowsKernel; the order of reductions MLX Metal and
//                 llama.cpp use, x3 B3, ch10 D1, D3);
//       'online'  MLX CUDA's kernel (x3 B3): per chunk of N_READS values a new running max and
//                 the normaliser rescaled by exp(prevmax - maxval), then one write pass. Chapter
//                 20's (m, D) recurrence applied to a row.
//     Each step records the running state, the elements read and written so far and the
//     exponentials computed so far.
//   - The same online normaliser in base 2 (MLX CUDA's decode attention, x3 B5): the scale is
//     folded into log2 e and exp2 replaces exp.
//   - KERNELS: which reductions each project's kernel performs and which exponential it calls,
//     where the facts say; 'not read' where they are silent.
//
// What it leaves out: threads, warps and the per-warp merge (the cross-lane reduction of the
// online state is collapsed into one sequential update); the exponential is Math.exp rounded
// to float32, not fast::exp, expf or __expf; caches (a row that fits in registers or cache
// may cost less than its reads suggest). The start value of MLX CUDA's running max is not in
// the facts; this model starts at minus infinity, so the first rescale multiplies 0 by 0.

import { tr } from '../lib/locale.ts';

const f = Math.fround;
const fexp = (x: number) => f(Math.exp(f(x)));
const fexp2 = (x: number) => f(Math.pow(2, f(x)));

/** MLX CUDA's per-thread chunk (its N_READS template argument); 4 here, as in the stepper. */
export const N_READS = 4;
/** Bytes per float32 element. */
export const FLOAT_BYTES = 4;

/**
 * The stepper's and the scene's row: 16 logits in 4 chunks. The max rises in chunk 1 (to 3.1),
 * stays in chunk 2 (2.2 < 3.1, so the rescale factor is exactly 1) and rises again in chunk 3 (to 4).
 */
export const ROW16 = [1.2, -0.5, 0.8, 2.0, 0.3, 3.1, -1.0, 1.5, 2.2, 0.6, -0.2, 1.1, 4.0, 0.0, 2.7, -2.0];

export type PathId = 'twopass' | 'online';
export const PATH_LABEL: Record<PathId, string> = {
  get twopass() { return tr('max, then sum (three loops, as lab 10.3)', 'max, rồi sum (ba vòng lặp, như lab 10.3)'); },
  get online() { return tr('online max and sum (MLX CUDA)', 'max và sum kiểu online (MLX CUDA)'); },
};

export type Op = 'max' | 'sum' | 'inv' | 'write' | 'chunk-max' | 'rescale' | 'add';

export interface Counters {
  /** Elements read from the row so far. */
  reads: number;
  /** Elements written so far. */
  writes: number;
  /** Exponentials computed so far. */
  exps: number;
}

export interface RowStep {
  path: PathId;
  /** 1-based sweep over the row: twopass 1 (max), 2 (sum), 3 (write); online 1 (max and sum), 2 (write). */
  pass: number;
  op: Op;
  /** Chunk index (0-based), -1 for the reciprocal step. */
  chunk: number;
  cols: number[];
  /** Running max after the step. */
  max: number;
  /** Running sum (twopass) or normaliser (online) after the step; after 'inv' still the sum. */
  sum: number;
  /** Online 'chunk-max' and 'rescale': the max before this chunk. */
  prevMax?: number;
  /** Online 'rescale': exp(prevMax - max), and the normaliser before it was applied. */
  factor?: number;
  before?: number;
  /** 'inv': 1 / sum. */
  inv?: number;
  counters: Counters;
  /** Outputs written so far (null where not yet written). */
  out: (number | null)[];
}

export interface RowTrace {
  path: PathId;
  row: number[];
  steps: RowStep[];
  out: number[];
  max: number;
  sum: number;
  inv: number;
  counters: Counters;
  /** How many times the row was read in full (reads / n). */
  rowReads: number;
  /** How many chunks raised the online max after the first (0 for twopass). */
  rises: number;
}

const chunksOf = (n: number, w: number) =>
  Array.from({ length: Math.ceil(n / w) }, (_, k) => Array.from({ length: Math.min(w, n - k * w) }, (_, i) => k * w + i));

/** Lab 10.3's order: three loops over the row (max, sum of exponentials, write). */
export function twoPass(row: number[], nReads = N_READS): RowTrace {
  const x = row.map(f);
  const n = x.length;
  const chunks = chunksOf(n, nReads);
  const steps: RowStep[] = [];
  const c: Counters = { reads: 0, writes: 0, exps: 0 };
  const out: (number | null)[] = new Array(n).fill(null);
  let m = -Infinity;
  let s = 0;
  const push = (pass: number, op: Op, chunk: number, cols: number[], extra: Partial<RowStep> = {}) =>
    steps.push({ path: 'twopass', pass, op, chunk, cols, max: m, sum: s, counters: { ...c }, out: out.slice(), ...extra });
  chunks.forEach((cols, k) => {
    for (const i of cols) m = Math.max(m, x[i]);
    c.reads += cols.length;
    push(1, 'max', k, cols);
  });
  chunks.forEach((cols, k) => {
    for (const i of cols) s = f(s + fexp(f(x[i] - m)));
    c.reads += cols.length;
    c.exps += cols.length;
    push(2, 'sum', k, cols);
  });
  const inv = f(1 / s);
  push(2, 'inv', -1, [], { inv });
  chunks.forEach((cols, k) => {
    for (const i of cols) out[i] = f(fexp(f(x[i] - m)) * inv);
    c.reads += cols.length;
    c.exps += cols.length;
    c.writes += cols.length;
    push(3, 'write', k, cols);
  });
  return { path: 'twopass', row: x, steps, out: out as number[], max: m, sum: s, inv, counters: { ...c }, rowReads: c.reads / n, rises: 0 };
}

/** MLX CUDA's order (x3 B3): per chunk the new max, the rescale, the new terms; then one write pass. */
export function online(row: number[], nReads = N_READS): RowTrace {
  const x = row.map(f);
  const n = x.length;
  const chunks = chunksOf(n, nReads);
  const steps: RowStep[] = [];
  const c: Counters = { reads: 0, writes: 0, exps: 0 };
  const out: (number | null)[] = new Array(n).fill(null);
  let maxval = -Infinity;
  let normalizer = 0;
  let rises = 0;
  const push = (pass: number, op: Op, chunk: number, cols: number[], extra: Partial<RowStep> = {}) =>
    steps.push({ path: 'online', pass, op, chunk, cols, max: maxval, sum: normalizer, counters: { ...c }, out: out.slice(), ...extra });
  chunks.forEach((cols, k) => {
    const prevmax = maxval;
    for (const i of cols) maxval = Math.max(maxval, x[i]);
    if (k > 0 && maxval > prevmax) rises++;
    c.reads += cols.length;
    push(1, 'chunk-max', k, cols, { prevMax: prevmax });
    // normalizer = normalizer * softmax_exp(prevmax - maxval): computed for every chunk, risen or not.
    const factor = fexp(f(prevmax - maxval));
    const before = normalizer;
    normalizer = f(normalizer * factor);
    c.exps += 1;
    push(1, 'rescale', k, cols, { prevMax: prevmax, factor, before });
    for (const i of cols) normalizer = f(normalizer + fexp(f(x[i] - maxval)));
    c.exps += cols.length;
    push(1, 'add', k, cols);
  });
  const inv = f(1 / normalizer);
  push(1, 'inv', -1, [], { inv });
  chunks.forEach((cols, k) => {
    for (const i of cols) out[i] = f(fexp(f(x[i] - maxval)) * inv);
    c.reads += cols.length;
    c.exps += cols.length;
    c.writes += cols.length;
    push(2, 'write', k, cols);
  });
  return { path: 'online', row: x, steps, out: out as number[], max: maxval, sum: normalizer, inv, counters: { ...c }, rowReads: c.reads / n, rises };
}

export const runPath = (p: PathId, row: number[] = ROW16, nReads = N_READS) => (p === 'twopass' ? twoPass(row, nReads) : online(row, nReads));

/** Softmax in float64, the reference both paths are compared with. */
export function softmax64(row: number[]): number[] {
  const m = Math.max(...row);
  const e = row.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export const maxAbsDiff = (a: number[], b: number[]) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

/** Distance between two float32 values in units in the last place (same sign assumed). */
export function ulps(a: number, b: number): number {
  const buf = new DataView(new ArrayBuffer(8));
  buf.setFloat32(0, a);
  buf.setFloat32(4, b);
  return Math.abs(buf.getInt32(0) - buf.getInt32(4));
}
export const maxUlps = (a: number[], b: number[]) => Math.max(...a.map((v, i) => ulps(v, b[i])));

/** Bytes moved for a row of n float32 values read `reads` times and written `writes` times. */
export const rowBytes = (n: number, reads: number, writes: number) => n * FLOAT_BYTES * (reads + writes);

/** Steps at which the stepper asks for a prediction: every rescale after the first chunk. */
export const predictionSteps = (t: RowTrace) =>
  t.steps.flatMap((s, i) => (s.op === 'rescale' && s.chunk > 0 ? [i] : []));

/** Is a predicted normaliser close enough to the model's? 2% relative, plus a small absolute slack. */
export const predictionOk = (predicted: number, actual: number) => Math.abs(predicted - actual) <= 0.02 * Math.abs(actual) + 5e-4;

// ---------------------------------------------------------------- base 2 (MLX CUDA decode, x3 B5)
export interface BaseResult {
  /** Running max after every element, in the units the kernel keeps it in. */
  maxes: number[];
  /** The normaliser at the end. */
  D: number;
  probs: number[];
}

/** One element at a time, in base e: s = score * scale, m and D as chapter 20 keeps them. */
export function onlineBaseE(scores: number[], scale: number): BaseResult {
  let m = -Infinity;
  let D = 0;
  const s = scores.map((v) => f(v * f(scale)));
  const maxes: number[] = [];
  for (const v of s) {
    const nm = Math.max(m, v);
    D = f(f(D * fexp(f(m - nm))) + fexp(f(v - nm)));
    m = nm;
    maxes.push(m);
  }
  return { maxes, D, probs: s.map((v) => f(fexp(f(v - m)) / D)) };
}

/**
 * The same in base 2, as MLX CUDA's kernel_sdpav writes it (x3 B5): scale_log2 = scale * M_LOG2E,
 * the score multiplied by scale_log2 (our reading), the running max kept in those units, and
 * factor = exp2f(max_score - new_max), exp_score = exp2f(score - new_max).
 */
export function onlineBase2(scores: number[], scale: number): BaseResult {
  const scaleLog2 = f(f(scale) * f(Math.LOG2E));
  let m = -Infinity;
  let D = 0;
  const s = scores.map((v) => f(v * scaleLog2));
  const maxes: number[] = [];
  for (const v of s) {
    const nm = Math.max(m, v);
    D = f(f(D * fexp2(f(m - nm))) + fexp2(f(v - nm)));
    m = nm;
    maxes.push(m);
  }
  return { maxes, D, probs: s.map((v) => f(fexp2(f(v - m)) / D)) };
}

// ---------------------------------------------------------------- the classification table
export type Structure = 'max-then-sum' | 'online' | 'one-sum' | 'not-read';
export const STRUCTURE_LABEL: Record<Structure, string> = {
  get 'max-then-sum'() { return tr('max, then sum', 'max, rồi sum'); },
  get online() { return tr('online max and sum in one pass', 'max và sum kiểu online trong một lượt'); },
  get 'one-sum'() { return tr('one sum only', 'chỉ một sum'); },
  get 'not-read'() { return tr('not read in the facts', 'chưa đọc trong tệp facts'); },
};
export type ExpFn = 'expf' | '__expf' | 'fast::exp' | 'precise::exp' | 'exp' | 'none' | 'not-read';
export const EXP_LABEL: Record<ExpFn, string> = {
  expf: 'expf',
  __expf: '__expf',
  'fast::exp': 'fast::exp',
  'precise::exp': 'precise::exp',
  get exp() { return tr('exp (Metal, default math mode)', 'exp (Metal, chế độ toán mặc định)'); },
  get none() { return tr('no exponential', 'không có hàm mũ'); },
  get 'not-read'() { return tr('not read in the facts', 'chưa đọc trong tệp facts'); },
};

export interface KernelRow {
  id: string;
  project: string;
  backend: 'CUDA and HIP' | 'CUDA' | 'Metal' | 'MPS';
  op: 'softmax' | 'RMSNorm';
  kernel: string;
  file: string;
  structure: Structure;
  exp: ExpFn;
  /** Where the answer comes from. */
  source: string;
  /** One line on why, shown after checking. */
  why: string;
}

export const KERNELS: KernelRow[] = [
  {
    id: 'llama-cuda-softmax', project: 'llama.cpp', backend: 'CUDA and HIP', op: 'softmax', kernel: 'soft_max_f32', file: 'ggml/src/ggml-cuda/softmax.cu',
    structure: 'max-then-sum', exp: 'expf', source: 'ch10 D1', // i18n-ignore: facts-item citation
    get why() { return tr('block_reduce MAX into buf_iw, expf(vals[col] - max_val), a barrier, then block_reduce SUM.', 'block_reduce MAX vào buf_iw, expf(vals[col] - max_val), một rào chắn, rồi block_reduce SUM.'); },
  },
  {
    id: 'llama-metal-softmax', project: 'llama.cpp', backend: 'Metal', op: 'softmax', kernel: 'kernel_soft_max', file: 'ggml/src/ggml-metal/kernels/softmax.metal',
    structure: 'max-then-sum', exp: 'exp', source: 'x3 B3', // i18n-ignore: facts-item citation
    get why() { return tr('simd_max with -INFINITY padding, then exp(...) and simd_sum with 0 padding: plain exp, so its accuracy follows the compile options.', 'simd_max với phần đệm -INFINITY, rồi exp(...) và simd_sum với phần đệm 0: exp thường, nên độ chính xác của nó theo các tùy chọn biên dịch.'); },
  },
  {
    id: 'mlx-metal-softmax', project: 'MLX', backend: 'Metal', op: 'softmax', kernel: 'softmax_single_row', file: 'mlx/backend/metal/kernels/softmax.h',
    structure: 'max-then-sum', exp: 'fast::exp', source: 'ch10 D3, x3 B3', // i18n-ignore: facts-item citation
    get why() { return tr('simd_max, broadcast through local_max[0], then the same with simd_sum; softmax_exp is fast::exp.', 'simd_max, phát (broadcast) qua local_max[0], rồi làm tương tự với simd_sum; softmax_exp là fast::exp.'); },
  },
  {
    id: 'mlx-cuda-softmax', project: 'MLX', backend: 'CUDA', op: 'softmax', kernel: 'softmax (softmax.cu)', file: 'mlx/backend/cuda/softmax.cu', // i18n-ignore: kernel name, shown as code
    structure: 'online', exp: '__expf', source: 'x3 B3', // i18n-ignore: facts-item citation
    get why() { return tr('per chunk: new max, normalizer * softmax_exp(prevmax - maxval), then the new terms; softmax_exp is __expf.', 'mỗi chunk: max mới, normalizer * softmax_exp(prevmax - maxval), rồi các số hạng mới; softmax_exp là __expf.'); },
  },
  {
    id: 'torch-cuda-softmax', project: 'PyTorch', backend: 'CUDA', op: 'softmax', kernel: 'persistent softmax (rows up to 2048)', file: 'aten/src/ATen/native/cuda/PersistentSoftmax.cuh', // i18n-ignore: kernel name, shown as code
    structure: 'not-read', exp: 'not-read', source: 'x3 B3, ch10 D4', // i18n-ignore: facts-item citation
    get why() { return tr('the facts record the routing (warp per row, at most 2048 elements and 8192 bytes) and "only __shfl_ instructions", not the order of its reductions or its exponential.', 'tệp facts ghi lại cách định tuyến (một warp mỗi hàng, tối đa 2048 phần tử và 8192 byte) và "chỉ các lệnh __shfl_", không ghi thứ tự các phép rút gọn hay hàm mũ của nó.'); },
  },
  {
    id: 'torch-mps-softmax', project: 'PyTorch', backend: 'MPS', op: 'softmax', kernel: 'SoftMax.metal', file: 'aten/src/ATen/native/mps/kernels/SoftMax.metal',
    structure: 'not-read', exp: 'precise::exp', source: 'x3 B3', // i18n-ignore: facts-item citation
    get why() { return tr('precise::exp throughout, with simd_max and simd_sum; whether the max and the sum are separate passes was not recorded.', 'precise::exp xuyên suốt, với simd_max và simd_sum; việc max và sum có là các lượt riêng hay không thì chưa được ghi lại.'); },
  },
  {
    id: 'llama-cuda-rmsnorm', project: 'llama.cpp', backend: 'CUDA and HIP', op: 'RMSNorm', kernel: 'rms_norm_f32', file: 'ggml/src/ggml-cuda/norm.cu',
    structure: 'one-sum', exp: 'none', source: 'x3 B2', // i18n-ignore: facts-item citation
    get why() { return tr('one block_reduce SUM of squares, then rsqrtf(mean + eps): no maximum and no exponential.', 'một block_reduce SUM các bình phương, rồi rsqrtf(mean + eps): không có phép lấy max và không có hàm mũ.'); },
  },
];
