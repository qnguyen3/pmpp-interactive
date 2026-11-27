// One kernel launch, for chapter 2, section 2.6: how many blocks an expression in
// the execution configuration asks for, what a launch with that many blocks does
// to an array of n elements, and what the result looks like when the blocks run a
// few at a time in some order. It runs the launch calculator, the "find the n that
// breaks it" bug hunt, the "any order, any device" explorer, the narrated animation
// "One launch, start to finish" and the numbers in the section's questions.
//
// Sources for the rules (the lesson cites them where they are used):
//  - the book, section 2.6, pp. 37 to 39: the block count is the ceiling of n over
//    the block size, written ceil(n/256.0) so that the division is not an integer
//    division; n = 1000 gives 4 blocks and 1024 threads of which 24 do nothing, 750
//    gives 3 blocks, 4000 gives 16, 2,000,000 gives 7813; blocks may run in any order;
//  - the book, Fig. 2.15, p. 42 (exercise 9): the integer form (N + 128 - 1)/128;
//  - notes/today/ch02-facts.md A8: NVIDIA's guide teaches the integer form; what
//    CUDA does with a grid dimension of zero is not documented in the pages read;
//  - notes/today/ch02-facts.md B4: HIP documents hipErrorInvalidValue for a zero
//    grid dimension;
//  - labs/common/include/pmpp/emu/emu_runtime.h (launch_impl): the course emulator
//    sets cudaErrorInvalidConfiguration for a grid of zero blocks;
//  - notes/today/ch02-facts.md A18: no order is guaranteed between blocks.
//
// What this leaves out:
//  - A device is one number, S: the blocks it runs at once. Real devices assign
//    blocks to SMs under several resource limits (chapter 4), and how many run
//    together depends on the device and the kernel. S is a parameter of this model
//    and not a property of any device.
//  - Blocks run in waves of S: every block of a wave starts from the memory as it
//    was before the wave and all of them finish together. A real device starts a
//    block whenever an earlier one finishes.
//  - Inside one block the threads run one after another in index order. A real
//    block gives no such order (book, chapter 4, p. 74). The model does this so
//    that the order of blocks is the only thing that varies.
//  - No time, no SMs, no warps, no occupancy, no limits on threads per block or on
//    blocks per grid (src/sim/grid.ts has those for chapter 3).
//  - n and the block size are C++ ints small enough that n + B - 1 does not
//    overflow; the model does not show what an overflow would do.
//  - Memory a kernel never writes keeps its initial contents. The model starts C
//    at a value you give it; a real allocation is not cleared and holds whatever
//    was there (notes/today/ch02-facts.md A2).
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- the block-count expression
export type ExprKey = 'int-ceil' | 'book-ceil' | 'trunc' | 'ceil-trunc' | 'plus-one';

export interface Expression {
  key: ExprKey;
  /** The expression with the names n and B. */
  generic: string;
  /** The expression as it is written in a launch, with the block size as a literal. */
  source: (B: number) => string;
}

export const EXPRESSIONS: readonly Expression[] = [
  { key: 'int-ceil', generic: '(n + B - 1) / B', source: (B) => `(n + ${B} - 1) / ${B}` },
  { key: 'book-ceil', generic: 'ceil(n / (double)B)', source: (B) => `ceil(n / ${B}.0)` }, // i18n-ignore: C++ expression
  { key: 'trunc', generic: 'n / B', source: (B) => `n / ${B}` },
  { key: 'ceil-trunc', generic: 'ceil(n / B)', source: (B) => `ceil(n / ${B})` },
  { key: 'plus-one', generic: 'n / B + 1', source: (B) => `n / ${B} + 1` },
];
export const expression = (key: ExprKey): Expression => EXPRESSIONS.find((e) => e.key === key)!;

export type CType = 'int' | 'double' | 'unsigned';

/** One step of evaluating the expression the way C++ does. */
export interface EvalStep {
  /** What is evaluated, with the numbers filled in. */
  text: string;
  value: number;
  type: CType;
  /** The C++ rule that produced the value. */
  rule: string;
}

export interface BlockCount {
  key: ExprKey;
  n: number;
  B: number;
  /** The value that reaches the launch as the number of blocks. */
  blocks: number;
  /** The fewest blocks whose threads cover n elements: the mathematical ceiling of n / B. */
  needed: number;
  steps: EvalStep[];
}

const checkSizes = (n: number, B: number) => {
  if (!Number.isInteger(n) || n < 0) throw new RangeError(`n must be a whole number, 0 or more (got ${n})`);
  if (!Number.isInteger(B) || B < 1) throw new RangeError(`the block size must be a whole number, 1 or more (got ${B})`);
};

/** The mathematical ceiling of n / B, in integers only. */
export const ceilDiv = (n: number, B: number): number => {
  checkSizes(n, B);
  return Math.floor(n / B) + (n % B === 0 ? 0 : 1);
};

/** How a value of a C++ type is written here: a double keeps its point, an integer has none. */
export const showValue = (value: number, type: CType): string => {
  if (type !== 'double') return String(value);
  if (Number.isInteger(value)) return value + '.0';
  return String(Number(value.toPrecision(7)));
};

const TO_COUNT =
  'the number of blocks is a field of a dim3, an unsigned integer, so the double is converted on the way in'; // i18n-ignore: translated where it is used (rule: tr(TO_COUNT, ...))

/**
 * Evaluates one of the five forms for int n and block size B with C++ semantics:
 * a division of two ints drops the fraction, ceil works on doubles, and the launch
 * takes an unsigned whole number.
 */
export function blockCount(key: ExprKey, n: number, B: number): BlockCount {
  checkSizes(n, B);
  const q = Math.trunc(n / B); // int / int
  const exact = n / B;
  const dropped = n % B === 0 ? tr('the division is exact', 'phép chia là chính xác') : tr(`the exact quotient is ${showValue(exact, 'double')}; the fraction is dropped`, `thương chính xác là ${showValue(exact, 'double')}; phần thập phân bị bỏ`);
  const steps: EvalStep[] = [];
  let blocks: number;
  switch (key) {
    case 'int-ceil': {
      const sum = n + B - 1;
      steps.push({ text: `${n} + ${B} - 1`, value: sum, type: 'int', rule: tr('int arithmetic', 'số học int') });
      blocks = Math.trunc(sum / B);
      steps.push({
        text: `${sum} / ${B}`,
        value: blocks,
        type: 'int',
        rule:
          tr('both operands are int, so the division drops the fraction', 'cả hai toán hạng đều là int, nên phép chia bỏ phần thập phân') +
          (sum % B === 0 ? tr(' (there is none here)', ' (ở đây không có)') : tr(` (${showValue(sum / B, 'double')} becomes ${blocks})`, ` (${showValue(sum / B, 'double')} thành ${blocks})`)),
      });
      break;
    }
    case 'book-ceil': {
      steps.push({
        text: `${n} / ${B}.0`,
        value: exact,
        type: 'double',
        rule: tr(`${B}.0 is a double, so n is converted to double and nothing is dropped`, `${B}.0 là một double, nên n được chuyển sang double và không có gì bị bỏ`),
      });
      blocks = Math.ceil(exact);
      steps.push({ text: `ceil(${showValue(exact, 'double')})`, value: blocks, type: 'double', rule: tr('rounds up to a whole number, still a double', 'làm tròn lên thành một số nguyên, vẫn là double') });
      steps.push({ text: tr(`${showValue(blocks, 'double')} as a block count`, `${showValue(blocks, 'double')} làm số khối`), value: blocks, type: 'unsigned', rule: tr(TO_COUNT, 'số khối là một trường của một dim3, một số nguyên unsigned, nên double được chuyển đổi khi truyền vào') });
      break;
    }
    case 'trunc': {
      blocks = q;
      steps.push({ text: `${n} / ${B}`, value: q, type: 'int', rule: tr(`both operands are int: ${dropped}`, `cả hai toán hạng đều là int: ${dropped}`) });
      break;
    }
    case 'ceil-trunc': {
      steps.push({ text: `${n} / ${B}`, value: q, type: 'int', rule: tr(`both operands are int: ${dropped}`, `cả hai toán hạng đều là int: ${dropped}`) });
      blocks = q;
      steps.push({
        text: `ceil(${q})`,
        value: q,
        type: 'double',
        rule: tr('the int is converted to double; it is already whole, so ceil has nothing to round', 'int được chuyển sang double; nó đã là số nguyên, nên ceil không có gì để làm tròn'),
      });
      steps.push({ text: tr(`${showValue(q, 'double')} as a block count`, `${showValue(q, 'double')} làm số khối`), value: q, type: 'unsigned', rule: tr(TO_COUNT, 'số khối là một trường của một dim3, một số nguyên unsigned, nên double được chuyển đổi khi truyền vào') });
      break;
    }
    case 'plus-one': {
      steps.push({ text: `${n} / ${B}`, value: q, type: 'int', rule: tr(`both operands are int: ${dropped}`, `cả hai toán hạng đều là int: ${dropped}`) });
      blocks = q + 1;
      steps.push({ text: `${q} + 1`, value: blocks, type: 'int', rule: tr('one more block, whether or not anything was dropped', 'thêm một khối, dù có gì bị bỏ hay không') });
      break;
    }
  }
  return { key, n, B, blocks, needed: ceilDiv(n, B), steps };
}

// ---------------------------------------------------------------- the launch
/** A run of element indices, both ends included. */
export interface Range {
  first: number;
  last: number;
}
const span = (first: number, end: number): Range[] => (end > first ? [{ first, last: end - 1 }] : []);
export const rangeSize = (r: Range): number => r.last - r.first + 1;

export type Backend = 'cuda' | 'hip' | 'emu';
export type ZeroOutcome = 'not documented' | 'error, documented' | 'error';

/** What a launch with zero blocks does, as far as each source says. */
export interface ZeroBlocks {
  backend: Backend;
  name: string;
  outcome: ZeroOutcome;
  /** The error named by the source, or what the source leaves unsaid. */
  detail: string;
}

export const ZERO_BLOCKS: readonly ZeroBlocks[] = [
  {
    backend: 'cuda',
    name: 'CUDA',
    outcome: 'not documented', // i18n-ignore: identifier also shown to the reader
    get detail() {
      return tr(
        "NVIDIA's pages describe a launch that fails when a dimension exceeds the maximum; they do not mention zero",
        'Các trang của NVIDIA mô tả một lần khởi chạy thất bại khi một chiều vượt quá mức tối đa; chúng không nhắc đến số không',
      );
    },
  },
  { backend: 'hip', name: 'HIP', outcome: 'error, documented', detail: 'hipErrorInvalidValue' }, // i18n-ignore: identifier also shown to the reader
  {
    backend: 'emu',
    get name() {
      return tr('course emulator', 'trình giả lập (emulator) của khóa học');
    },
    outcome: 'error', detail: 'cudaErrorInvalidConfiguration',
  },
];

export interface LaunchResult {
  n: number;
  B: number;
  blocks: number;
  guard: boolean;
  /** False when the count is 0: there is no grid to run. */
  launched: boolean;
  /** Per backend, what a count of 0 does; null when there is at least one block. */
  zero: readonly ZeroBlocks[] | null;
  /** Threads the launch creates: blocks times B. */
  threads: number;
  /** Threads whose index is below n: with the guard, the ones that pass it. */
  working: number;
  /** Threads whose index is n or more. With the guard they do nothing; without it they write out of range. */
  surplus: number;
  /** Threads the guard stops (0 when the guard is off). */
  idle: number;
  /** Elements of the array that no thread processes. */
  unprocessed: Range[];
  unprocessedCount: number;
  /** Whole blocks in which no thread has an element below n. */
  emptyBlocks: number;
  /** Indices past the end of the array that threads write when the guard is off. */
  outOfRange: Range[];
  outOfRangeCount: number;
  /** Every element processed and nothing written out of range: what a comparison with the CPU loop tests. */
  correct: boolean;
}

/** What a launch of `blocks` blocks of B threads does to an array of n elements, one element per thread. */
export function launch(n: number, B: number, blocks: number, guard = true): LaunchResult {
  checkSizes(n, B);
  if (!Number.isInteger(blocks) || blocks < 0) throw new RangeError(`the block count must be a whole number, 0 or more (got ${blocks})`);
  const threads = blocks * B;
  const working = Math.min(threads, n);
  const surplus = threads - working;
  const unprocessed = span(threads, n);
  const outOfRange = guard ? [] : span(n, threads);
  const needed = ceilDiv(n, B);
  return {
    n,
    B,
    blocks,
    guard,
    launched: blocks > 0,
    zero: blocks === 0 ? ZERO_BLOCKS : null,
    threads,
    working,
    surplus,
    idle: guard ? surplus : 0,
    unprocessed,
    unprocessedCount: n - working,
    emptyBlocks: Math.max(0, blocks - needed),
    outOfRange,
    outOfRangeCount: guard ? 0 : surplus,
    correct: working === n && (guard || surplus === 0),
  };
}

/** The launch you get from one of the five expressions. */
export const launchWith = (key: ExprKey, n: number, B: number, guard = true): LaunchResult =>
  launch(n, B, blockCount(key, n, B).blocks, guard);

/** Block k of a launch: the indices its threads compute and how many of them are below n. */
export interface BlockWork {
  block: number;
  /** Index computed by the block's thread 0, and by its last thread. */
  first: number;
  last: number;
  /** Threads of this block whose index is below n. */
  working: number;
}
export const blockWork = (n: number, B: number, k: number): BlockWork => ({
  block: k,
  first: k * B,
  last: k * B + B - 1,
  working: Math.max(0, Math.min(B, n - k * B)),
});

/** The global index a thread computes: blockIdx.x * blockDim.x + threadIdx.x. */
export const globalIndex = (block: number, B: number, thread: number): number => block * B + thread;

// ---------------------------------------------------------------- testing an expression
export interface Verdict {
  n: number;
  blocks: number;
  /** A test that compares every element with the CPU loop passes. */
  passes: boolean;
  /** The launch contains a block in which no thread works. */
  wastes: boolean;
  result: LaunchResult;
}
export const verdict = (key: ExprKey, n: number, B: number): Verdict => {
  const result = launchWith(key, n, B);
  return { n, blocks: result.blocks, passes: result.correct, wastes: result.emptyBlocks > 0, result };
};

/** The smallest n from 1 up to `limit` for which the test fails, or null when there is none. */
export function smallestFailing(key: ExprKey, B: number, limit = 4 * B): number | null {
  for (let n = 1; n <= limit; n++) if (!verdict(key, n, B).passes) return n;
  return null;
}
/** The smallest n from 1 up to `limit` for which a whole block does nothing, or null when there is none. */
export function smallestWasting(key: ExprKey, B: number, limit = 4 * B): number | null {
  for (let n = 1; n <= limit; n++) if (verdict(key, n, B).wastes) return n;
  return null;
}

// ---------------------------------------------------------------- order and concurrency
export type Order = 'ascending' | 'descending' | 'shuffled';
export const ORDERS: readonly Order[] = ['ascending', 'descending', 'shuffled'];

export interface Schedule {
  blocks: number;
  /** Blocks the device runs at once. */
  S: number;
  /** Every block index once, in the order the device takes them. */
  order: number[];
  /** The same order cut into groups of at most S. */
  waves: number[][];
}

/** A small seeded generator (mulberry32), so a shuffle can be repeated. */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The order blocks are taken in, and the waves that order makes on a device that
 * runs S at once. A shuffled order that happens to come out ascending is rotated
 * by one place, so that "shuffled" always differs from "ascending" when there is
 * more than one block.
 */
export function schedule(blocks: number, S: number, order: Order, seed = 1): Schedule {
  if (!Number.isInteger(blocks) || blocks < 0) throw new RangeError(`blocks must be a whole number, 0 or more (got ${blocks})`);
  if (!Number.isInteger(S) || S < 1) throw new RangeError(`S must be a whole number, 1 or more (got ${S})`);
  const list = Array.from({ length: blocks }, (_, k) => k);
  if (order === 'descending') list.reverse();
  if (order === 'shuffled') {
    const next = rng(seed);
    for (let i = list.length - 1; i > 0; i--) {
      const j = Math.floor(next() * (i + 1));
      [list[i], list[j]] = [list[j], list[i]];
    }
    if (blocks > 1 && list.every((b, k) => b === k)) list.push(list.shift()!);
  }
  const waves: number[][] = [];
  for (let i = 0; i < list.length; i += S) waves.push(list.slice(i, i + S));
  return { blocks, S, order: list, waves };
}

/** Waves a device that runs S blocks at once needs for a grid: the ceiling of blocks / S. */
export const waveCount = (blocks: number, S: number): number => Math.ceil(blocks / S);

export type KernelKind = 'independent' | 'chained';
export interface KernelInfo {
  kind: KernelKind;
  /** The statement a thread executes when its index passes the guard. */
  body: string;
}
export const KERNELS: readonly KernelInfo[] = [
  { kind: 'independent', body: 'C[i] = A[i] + B[i]' },
  { kind: 'chained', body: 'C[i] = C[i-1] + A[i]' },
];

export interface KernelData {
  n: number;
  /** Threads per block. */
  B: number;
  A: number[];
  Bv: number[];
  /** What C holds before the launch. NaN stands for "whatever the allocation held". */
  C0: number[];
}

/** Small whole numbers, so sums can be checked by eye. C starts at `initial` everywhere. */
export function makeData(n: number, B: number, initial = 0): KernelData {
  checkSizes(n, B);
  return {
    n,
    B,
    A: Array.from({ length: n }, (_, i) => (i * 7 + 3) % 10),
    Bv: Array.from({ length: n }, (_, i) => (i * 4 + 1) % 9),
    C0: Array.from({ length: n }, () => initial),
  };
}

/** What a sequential loop over i = 0 .. n-1 leaves in C. */
export function sequential(kind: KernelKind, d: KernelData): number[] {
  const C = d.C0.slice();
  for (let i = 0; i < d.n; i++) C[i] = kind === 'independent' ? d.A[i] + d.Bv[i] : (i > 0 ? C[i - 1] : 0) + d.A[i];
  return C;
}

export interface KernelRun {
  kind: KernelKind;
  /** C after wave 0, after wave 1, and so on; `after[w]` is the state once wave w has finished. */
  after: number[][];
  /** For each wave, the elements written in it. */
  written: number[][];
  /** C when the grid has finished (the initial contents if there were no waves). */
  final: number[];
  /** What the sequential loop gives. */
  expected: number[];
  /** Elements where the two differ. */
  wrong: number[];
  matches: boolean;
}

const sameValue = (a: number, b: number) => a === b || (Number.isNaN(a) && Number.isNaN(b));

/**
 * Runs a kernel wave by wave. A thread works only if its index is below n (the
 * guard). Every block of a wave reads other blocks' elements as they were before
 * the wave; inside a block, threads run in index order. For the chained kernel the
 * thread with i = 0 reads 0 in place of C[-1].
 */
export function runKernel(kind: KernelKind, sched: Schedule, d: KernelData): KernelRun {
  const C = d.C0.slice();
  const after: number[][] = [];
  const written: number[][] = [];
  for (const wave of sched.waves) {
    const before = C.slice();
    const wrote: number[] = [];
    for (const b of wave) {
      for (let t = 0; t < d.B; t++) {
        const i = globalIndex(b, d.B, t);
        if (i >= d.n) continue; // the guard
        if (kind === 'independent') C[i] = d.A[i] + d.Bv[i];
        else {
          // C[i-1] belongs to this block unless this is the block's first thread.
          const prev = i === 0 ? 0 : t > 0 ? C[i - 1] : before[i - 1];
          C[i] = prev + d.A[i];
        }
        wrote.push(i);
      }
    }
    after.push(C.slice());
    written.push(wrote.sort((x, y) => x - y));
  }
  const expected = sequential(kind, d);
  const wrong = expected.flatMap((v, i) => (sameValue(v, C[i]) ? [] : [i]));
  return { kind, after, written, final: C, expected, wrong, matches: wrong.length === 0 };
}

/** Every order of `blocks` blocks (use for small counts only). */
export function permutations(blocks: number): number[][] {
  if (blocks <= 0) return [[]];
  const out: number[][] = [];
  const go = (rest: number[], acc: number[]) => {
    if (!rest.length) out.push(acc);
    rest.forEach((b, i) => go([...rest.slice(0, i), ...rest.slice(i + 1)], [...acc, b]));
  };
  go(Array.from({ length: blocks }, (_, k) => k), []);
  return out;
}

/** A schedule for an explicit order, cut into waves of S. */
export function scheduleOf(order: number[], S: number): Schedule {
  const waves: number[][] = [];
  for (let i = 0; i < order.length; i += S) waves.push(order.slice(i, i + S));
  return { blocks: order.length, S, order: order.slice(), waves };
}

/** How many different final arrays a kernel can produce over every order of its blocks and every S from 1 to the block count. */
export function distinctResults(kind: KernelKind, blocks: number, d: KernelData): { runs: number; distinct: number } {
  const seen = new Set<string>();
  let runs = 0;
  for (const order of permutations(blocks))
    for (let S = 1; S <= Math.max(1, blocks); S++) {
      seen.add(runKernel(kind, scheduleOf(order, S), d).final.join(','));
      runs++;
    }
  return { runs, distinct: seen.size };
}

// ---------------------------------------------------------------- the book's numbers
/** The book's four worked values of n, with 256 threads per block (pp. 38 to 39). */
export const BOOK_BLOCK = 256;
export const BOOK_NS: readonly number[] = [1000, 750, 4000, 2000000];

/** Book exercise 9 (Fig. 2.15, p. 42): N = 200000, blocks of 128, (N + 128 - 1)/128 blocks. */
export function ex9() {
  const N = 200000;
  const threadsPerBlock = 128;
  const r = launchWith('int-ceil', N, threadsPerBlock);
  return {
    N,
    /** (a) */ threadsPerBlock,
    /** (b) */ threads: r.threads,
    /** (c) */ blocks: r.blocks,
    /** (d) threads that execute the index line: every thread. */ line2: r.threads,
    /** (e) threads that execute the assignment: those that pass the guard. */ line4: r.working,
  };
}

/**
 * The launch the narrated animation draws: 22 elements, blocks of 8, a device that
 * runs 2 at once. With this seed the shuffled order is block 2, block 0, block 1.
 */
export const SCENE = { n: 22, B: 8, S: 2, seed: 1 } as const;
