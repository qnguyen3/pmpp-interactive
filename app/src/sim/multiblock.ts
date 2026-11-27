// Many blocks (chapter 10, sections 10.9 and 10.10). One model for the segment
// calculator, the commit-order explorer, the coarsening explorer and the
// "commit-order" scene:
//
//   - segments: block b of a launch with `block` threads and coarsening factor C
//     starts at 2·C·block·b (Fig. 10.18 line 03 with C = 1, Fig. 10.20 line 03), so a
//     launch needs ceil(N / (2·C·block)) blocks and the last one is usually partly
//     filled; each thread loads 2·C elements strided by the block size (Fig. 10.20
//     lines 04 to 07), guarded by index < N as exercise 5 asks (p. 250);
//   - block sums in float32 (rounding after every addition, Math.fround): each
//     thread adds its loads in order, then the block's partial sums meet in a fixed
//     halving tree (thread t with thread t + m/2, the order of Fig. 10.10's loop);
//   - commits: the atomic design adds the block sums into one total in a given order
//     (p. 245: atomics give mutual exclusion, not order); the two-pass design writes
//     them to an array indexed by block and reduces it in a fixed tree (pp. 245-246);
//     seeded orders, distinct bit patterns over k runs, a float64 reference;
//   - Fig. 10.21's step accounting (pp. 247-248): per block, 2C - 1 fully used steps
//     and log2(block) underused tree steps; blocks run in whole waves of R; totals,
//     commits and a "too few blocks" flag; presets.
//
// What this leaves out (the lessons say so beside every interactive):
//   - A commit order is a seeded permutation, not a timing model. Nothing here says
//     which orders a GPU produces or how often.
//   - The step accounting is the book's: every step costs the same, blocks run in
//     whole waves, no memory time, no launch overhead, no atomic cost.
//
// Erasable TypeScript only, and no value imports: the browser checks import this
// file directly in Node.

import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- float32 helpers
export const f32 = Math.fround;

const view = new DataView(new ArrayBuffer(4));
/** The 32 bits of a float32, as an unsigned integer. */
export function f32bits(x: number): number {
  view.setFloat32(0, x);
  return view.getUint32(0);
}
/** A float32 written as its bit pattern, e.g. 0x420a6666. */
export function f32hex(x: number): string {
  return '0x' + f32bits(x).toString(16).padStart(8, '0');
}
/** Bit pattern mapped to an integer that orders like the float. */
function ordered(x: number): number {
  const b = f32bits(x);
  return b >= 0x80000000 ? -(b - 0x80000000) : b;
}
/** How many float32 steps apart two values are: 0 when equal, 1 when adjacent. */
export function ulpDistance(a: number, b: number): number {
  return Math.abs(ordered(f32(a)) - ordered(f32(b)));
}
/** The shortest decimal that reads back as the same float32. */
export function f32short(x: number): string {
  const v = f32(x);
  for (let p = 1; p <= 9; p++) {
    const s = v.toPrecision(p);
    if (f32(Number(s)) === v) return tidy(s);
  }
  return tidy(v.toPrecision(9));
}
/** A float64 value for display, up to 12 significant digits. */
export function f64short(x: number): string {
  return tidy(String(Number(x.toPrecision(12))));
}
function tidy(s: string): string {
  let [m, e] = s.split('e');
  if (m.includes('.')) m = m.replace(/0+$/, '').replace(/\.$/, '');
  return (m + (e ? 'e' + e : '')).replace(/^-/, '−');
}

/** Neumaier's compensated sum in float64: the reference no float32 order is expected to match. */
export function sum64(values: ArrayLike<number>): number {
  let s = 0;
  let c = 0;
  for (let i = 0; i < values.length; i++) {
    const x = values[i];
    const t = s + x;
    c += Math.abs(s) >= Math.abs(x) ? s - t + x : x - t + s;
    s = t;
  }
  return s + c;
}

// ---------------------------------------------------------------- seeded data
/** mulberry32: a small seedable generator. Same seed, same numbers, everywhere. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Logit-like mean and spread (our choice, not a measured model): normal, mean −1, deviation 2.5. */
export const LOGIT_MEAN = -1;
export const LOGIT_SD = 2.5;

/** n float32 logit-like values from a seeded generator (Box-Muller). */
export function logits(n: number, seed: number): Float32Array {
  const u = rng(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const a = Math.max(u(), 1e-12);
    const b = u();
    const m = Math.sqrt(-2 * Math.log(a));
    out[i] = LOGIT_MEAN + LOGIT_SD * m * Math.cos(2 * Math.PI * b);
    if (i + 1 < n) out[i + 1] = LOGIT_MEAN + LOGIT_SD * m * Math.sin(2 * Math.PI * b);
  }
  return out;
}

/** A permutation of 0..n-1 (Fisher-Yates on the seeded generator). */
export function seededOrder(n: number, seed: number): number[] {
  const u = rng(seed);
  const p = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(u() * (i + 1));
    const t = p[i];
    p[i] = p[j];
    p[j] = t;
  }
  return p;
}

// ---------------------------------------------------------------- 10.9: segments
export interface SegmentInfo {
  N: number;
  block: number;
  coarse: number;
  /** Elements one block covers: 2·C·block. */
  perBlock: number;
  /** ceil(N / perBlock); 0 for an empty input. */
  blocks: number;
  /** Where the last block's segment starts. */
  lastStart: number;
  /** Real elements (index < N) in the last block's segment. */
  lastReal: number;
  /** Slots of the last segment that lie past the end of the input. */
  pastEnd: number;
}

/** Block b's first element: 2·C·block·b (Fig. 10.18 line 03; Fig. 10.20 line 03). */
export const segmentStart = (b: number, block: number, coarse = 1): number => 2 * coarse * block * b;

export function segments(N: number, block: number, coarse = 1): SegmentInfo {
  const perBlock = 2 * coarse * block;
  const blocks = N > 0 ? Math.ceil(N / perBlock) : 0;
  const lastStart = blocks > 0 ? segmentStart(blocks - 1, block, coarse) : 0;
  const lastReal = blocks > 0 ? N - lastStart : 0;
  return { N, block, coarse, perBlock, blocks, lastStart, lastReal, pastEnd: blocks > 0 ? perBlock - lastReal : 0 };
}

export interface Load {
  /** Which of the thread's 2·C loads: 0 is line 05 of Fig. 10.20, 1.. the loop of line 07. */
  k: number;
  index: number;
  inRange: boolean;
}

/** The elements thread t of block b loads: segment + t + k·block for k = 0 .. 2C-1. */
export function threadLoads(N: number, block: number, coarse: number, b: number, t: number): Load[] {
  const start = segmentStart(b, block, coarse) + t;
  return Array.from({ length: 2 * coarse }, (_, k) => {
    const index = start + k * block;
    return { k, index, inRange: index < N };
  });
}

export interface BlockView {
  b: number;
  start: number;
  /** One past the segment's last slot. */
  end: number;
  /** Real elements in this segment. */
  real: number;
  /** Threads with at least one real element. */
  busyThreads: number;
  /** Threads that load nothing real but must still take part (and contribute 0). */
  idleThreads: number;
  /** Loads that would read past N without the exercise-5 guard. */
  pastEnd: number;
}

export function blockView(N: number, block: number, coarse: number, b: number): BlockView {
  const s = segments(N, block, coarse);
  const start = segmentStart(b, block, coarse);
  const real = Math.max(0, Math.min(s.perBlock, N - start));
  const busy = Math.min(block, real);
  return { b, start, end: start + s.perBlock, real, busyThreads: busy, idleThreads: block - busy, pastEnd: s.perBlock - real };
}

// ---------------------------------------------------------------- 10.9: block sums
/** Thread t's float32 partial sum: its loads in order, out-of-range ones contributing 0 (exercise 5). */
export function threadSum(values: ArrayLike<number>, block: number, coarse: number, b: number, t: number): number {
  const N = values.length;
  let acc = 0;
  const start = segmentStart(b, block, coarse) + t;
  for (let k = 0; k < 2 * coarse; k++) {
    const i = start + k * block;
    acc = f32(acc + (i < N ? values[i] : 0));
  }
  return acc;
}

/** A fixed halving tree in float32: element j absorbs element j + m/2, then the half, and so on. Pads to a power of two with 0. */
export function halvingTree(xs: ArrayLike<number>): number {
  if (xs.length === 0) return 0;
  let m = 1;
  while (m < xs.length) m *= 2;
  const a = new Float64Array(m);
  for (let i = 0; i < xs.length; i++) a[i] = f32(xs[i]);
  for (let stride = m / 2; stride >= 1; stride /= 2) for (let j = 0; j < stride; j++) a[j] = f32(a[j] + a[j + stride]);
  return a[0];
}

/** Every level of the halving tree, from the padded input down to the single total. */
export function halvingLevels(xs: ArrayLike<number>): number[][] {
  let m = 1;
  while (m < xs.length) m *= 2;
  let level = Array.from({ length: m }, (_, i) => (i < xs.length ? f32(xs[i]) : 0));
  const out = [level];
  while (level.length > 1) {
    const half = level.length / 2;
    level = level.slice(0, half).map((v, j) => f32(v + level[j + half]));
    out.push(level);
  }
  return out;
}

/** One block's float32 sum: each thread's partial sum, then the halving tree over the block. */
export function blockSum(values: ArrayLike<number>, block: number, coarse: number, b: number): number {
  const partial = new Float64Array(block);
  for (let t = 0; t < block; t++) partial[t] = threadSum(values, block, coarse, b, t);
  return halvingTree(partial);
}

/** Every block's float32 sum, in block order. */
export function blockSums(values: ArrayLike<number>, block: number, coarse = 1): number[] {
  const { blocks } = segments(values.length, block, coarse);
  return Array.from({ length: blocks }, (_, b) => blockSum(values, block, coarse, b));
}

// ---------------------------------------------------------------- 10.9: commits
export type Design = 'atomic' | 'two-pass';
export const DESIGN_LABEL: Record<Design, string> = {
  get atomic() {
    return tr('atomic commit (Fig. 10.18)', 'ghi kết quả bằng phép toán nguyên tử (Fig. 10.18)');
  },
  get 'two-pass'() {
    return tr('partials, then a fixed second pass', 'tổng từng phần, rồi một lượt thứ hai cố định');
  },
};

export interface CommitTrace {
  /** The order the leaders' sums reach the total (atomic), or block order (two-pass). */
  order: number[];
  /** Atomic: the float32 total after each commit. Two-pass: empty. */
  running: number[];
  total: number;
}

/**
 * The atomic design for a given commit order: the total starts at 0 (the host must
 * zero it before the launch) and each leader adds its block's sum, rounding to
 * float32 every time. 'fixed' is the two-pass design: the sums sit in an array
 * indexed by block and one block reduces it in the fixed halving tree.
 */
export function commit(sums: number[], order: number[] | 'fixed'): CommitTrace {
  if (order === 'fixed') return { order: sums.map((_, i) => i), running: [], total: halvingTree(sums) };
  let total = 0;
  const running = order.map((b) => (total = f32(total + sums[b])));
  return { order, running, total };
}

/** The seed base of run k's commit order: run k uses seededOrder(B, RUN_SEED + k). */
export const RUN_SEED = 1009;

/** Run k of a design: the atomic design commits in run k's seeded order; the two-pass design ignores the order. */
export function runOf(sums: number[], design: Design, run: number): CommitTrace {
  return design === 'two-pass' ? commit(sums, 'fixed') : commit(sums, seededOrder(sums.length, RUN_SEED + run));
}

export interface History {
  totals: number[];
  /** Distinct float32 bit patterns among the totals, in order of first appearance. */
  distinct: string[];
}

/** Runs 0 .. runs-1 of a design: each total and the distinct bit patterns among them. */
export function history(sums: number[], design: Design, runs: number): History {
  const totals = Array.from({ length: runs }, (_, k) => runOf(sums, design, k).total);
  const distinct: string[] = [];
  for (const t of totals) {
    const h = f32hex(t);
    if (!distinct.includes(h)) distinct.push(h);
  }
  return { totals, distinct };
}

/** The float64 sum of the float32 inputs. */
export const reference = (values: ArrayLike<number>): number => sum64(values);

/** The first run after run 0 whose atomic total differs from run 0's, or -1 within `limit` runs. */
export function firstDifferentRun(sums: number[], limit = 64): number {
  const first = f32bits(runOf(sums, 'atomic', 0).total);
  for (let k = 1; k < limit; k++) if (f32bits(runOf(sums, 'atomic', k).total) !== first) return k;
  return -1;
}

// ---------------------------------------------------------------- 10.10: step accounting
export interface CoarsenInput {
  /** Elements per row (the whole input when rows = 1). */
  N: number;
  /** Threads per block (a power of two). */
  block: number;
  /** Coarsening factor: each thread loads 2·C elements. */
  C: number;
  /** Blocks the device runs at once. */
  R: number;
  /** Independent rows, each reduced by its own blocks (default 1). */
  rows?: number;
}

export interface CoarsenResult {
  input: CoarsenInput;
  /** Elements one block covers, 2·C·block. */
  perBlock: number;
  blocksPerRow: number;
  /** Blocks launched: rows x blocks per row. */
  blocks: number;
  /** ceil(blocks / R). */
  waves: number;
  /** Blocks in the last wave. */
  lastWave: number;
  /** Fully used steps per block: 2C - 1 additions with every thread busy. */
  fullPerBlock: number;
  /** Underused steps per block: the tree, log2(block). */
  underPerBlock: number;
  stepsPerBlock: number;
  /** Totals over the whole launch: waves x per block. */
  fullSteps: number;
  underSteps: number;
  totalSteps: number;
  /** One atomic commit per block (Fig. 10.18). */
  commits: number;
  /** Busy block slots over all slots offered, wave by wave. */
  busyShare: number;
  /** Fewer blocks than the device runs at once: the book's warning (p. 248). */
  tooFew: boolean;
}

export const log2 = (n: number): number => Math.round(Math.log2(n));

export function coarsenSteps(input: CoarsenInput): CoarsenResult {
  const { N, block, C, R } = input;
  const rows = input.rows ?? 1;
  const perBlock = 2 * C * block;
  const blocksPerRow = Math.max(1, Math.ceil(N / perBlock));
  const blocks = rows * blocksPerRow;
  const waves = Math.ceil(blocks / R);
  const lastWave = blocks - (waves - 1) * R;
  const fullPerBlock = 2 * C - 1;
  const underPerBlock = log2(block);
  const stepsPerBlock = fullPerBlock + underPerBlock;
  return {
    input,
    perBlock,
    blocksPerRow,
    blocks,
    waves,
    lastWave,
    fullPerBlock,
    underPerBlock,
    stepsPerBlock,
    fullSteps: waves * fullPerBlock,
    underSteps: waves * underPerBlock,
    totalSteps: waves * stepsPerBlock,
    commits: blocks,
    busyShare: blocks / (waves * R),
    tooFew: blocks < R,
  };
}

/** Beyond this factor a row is already one block: more only adds idle loads. */
export const maxUsefulC = (N: number, block: number): number => Math.max(1, Math.ceil(N / (2 * block)));

/** The step accounting for every C from 1 to maxC. */
export function sweepC(base: Omit<CoarsenInput, 'C'>, maxC: number): CoarsenResult[] {
  return Array.from({ length: Math.max(1, maxC) }, (_, i) => coarsenSteps({ ...base, C: i + 1 }));
}

/** The factor with the fewest total steps (the smaller factor on a tie). */
export function fewestStepsC(base: Omit<CoarsenInput, 'C'>, maxC: number): number {
  let best = 1;
  let steps = Infinity;
  for (const r of sweepC(base, maxC))
    if (r.totalSteps < steps) {
      steps = r.totalSteps;
      best = r.input.C;
    }
  return best;
}

/**
 * The book's rule (p. 248): the largest factor that still launches at least as many
 * blocks as the device runs at once. 1 when even C = 1 launches too few.
 */
export function justEnoughC(base: Omit<CoarsenInput, 'C'>, maxC: number): number {
  let best = 1;
  for (const r of sweepC(base, maxC)) if (r.blocks >= base.R) best = r.input.C;
  return best;
}

// ---------------------------------------------------------------- presets
/** Fig. 10.21 (pp. 247-248): 32 elements, 8-thread blocks, one block at a time; C = 1 against C = 2. */
export const FIG1021 = { N: 32, block: 8, R: 1 } as const;

/** Chapter 6's hypothetical device runs 264 blocks at once (book p. 143, as that chapter's example). */
export const CH6_DEVICE_R = 264;

/**
 * Vocabulary sizes read from public model configs (notes/today/ch09-facts.md, D7).
 * The softmax presets reduce one row of this many logits per block.
 */
export const VOCABS = [
  { name: 'GPT-2', size: 50257 },
  { name: 'Qwen3-8B', size: 151936 }, // i18n-ignore: model name
  { name: 'gpt-oss-20b', size: 201088 }, // i18n-ignore: model name
] as const;

/** The softmax presets' row length: Qwen3-8B's vocabulary. */
export const SOFTMAX_ROW = 151936;
export const SOFTMAX_BLOCK = 256;
export const SOFTMAX_BATCHES = [1, 8, 64] as const;

/** One 256-thread block per row of SOFTMAX_ROW logits: C is whatever makes 2·C·256 cover the row. */
export function softmaxPreset(batch: number, R: number = CH6_DEVICE_R): CoarsenInput {
  return { N: SOFTMAX_ROW, block: SOFTMAX_BLOCK, C: maxUsefulC(SOFTMAX_ROW, SOFTMAX_BLOCK), R, rows: batch };
}

// ---------------------------------------------------------------- the scene's example
/**
 * The commit-order scene: 60 logits, 4-thread blocks, so 8 segments of 8 and a last
 * block with 4 real elements. DATA_SEED is chosen (and pinned by the tests) so that
 * runs 0 and 1 of the atomic design end on different float32 totals.
 */
export const SCENE = { N: 60, block: 4, coarse: 1, dataSeed: 14 } as const;

export function sceneData() {
  const values = logits(SCENE.N, SCENE.dataSeed);
  const seg = segments(SCENE.N, SCENE.block, SCENE.coarse);
  const sums = blockSums(values, SCENE.block, SCENE.coarse);
  const runA = runOf(sums, 'atomic', 0);
  const runB = runOf(sums, 'atomic', 1);
  const twoPass = runOf(sums, 'two-pass', 0);
  return { values, seg, sums, runA, runB, twoPass, ref: reference(values) };
}
