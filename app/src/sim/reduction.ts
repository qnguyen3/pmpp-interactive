// Chapter 10, sections 10.3 to 10.6: one block of N/2 threads summing N values, at most
// 2048. It runs the steppers, warp maps, request counters and the barrier hunt of those
// four sections, and the numbers in their questions.
//
// Four models, all of the same kernels:
//  1. iterations(kind, N, width): the loop of the simple (Fig. 10.5) or convergent
//     (Fig. 10.8) kernel, one iteration at a time: stride, active threads, and every
//     warp full, divergent or idle at a given warp width, with the book's accounting of
//     thread slots, results and efficiency (pp. 230 to 234).
//  2. requests(kind, N, width): the global-memory requests of the simple, convergent and
//     shared-memory (Fig. 10.10) kernels, by a counting rule of ours that reproduces the
//     book's three counts (pp. 235 to 237).
//  3. runSpec(spec, values): exercise 3 (p. 249) as a choice of four expressions in
//     Fig. 10.8's loop, run one iteration at a time.
//  4. runInPlace / runShared: every thread of one block as its own list of source-level
//     operations, under four thread orders, with or without the barriers, recording reads
//     of values that are not yet final, reads of values already overwritten, reads past
//     the end of an array, and blocks that can never leave a barrier.
//
// Sources (the lessons cite them where they are used):
//  - the book, §10.3 to §10.6, pp. 227 to 237 (notes/ch10.md): the kernels, the
//    256-element example, the formulas for slots (p. 231, p. 234) and requests (p. 235),
//    N + 1 accesses and N/32 + 1 requests for the shared-memory kernel (pp. 236 to 237),
//    exercises 1, 2, 3 and 6 (pp. 249 to 250);
//  - the kernels as they run here are labs/ch10/snippets/s103_inplace.cu,
//    s104_mirrored.cu and s106_barrier_hunt.cu; reduction.test.ts ties every line number
//    below to those files;
//  - the barrier rule: the book's, ch. 4 §4.3 (every thread of the block reaches the same
//    barrier, or none does), which is also Metal's (notes/today/ch05-facts.md C2). NVIDIA
//    calls a barrier under a non-uniform condition undefined behaviour and lets exited
//    threads off (ch04 facts A6).
//
// What this leaves out, and what the app says wherever it is shown:
//  - No timing at all. A warp "costs" its full width in any iteration in which one of its
//    lanes is active: the book's accounting, not a measurement.
//  - The request rule (one aligned piece of `segment` floats per warp per access) is ours.
//    Real memory systems, transaction sizes and caches differ (chapter 6), and the course's
//    emulators model none of it.
//  - A thread advances one source-level operation at a time; an addition reads its two
//    operands and writes its result in one step. "Lockstep" runs every thread one
//    operation per round, all reads of a round before all writes. Real threads interleave
//    instruction by instruction, in warps.
//  - Shared memory nobody wrote reads as NaN ("?"), the worst case. On a device it holds
//    whatever was there before.
//  - A block stuck at a barrier stands for undefined behaviour, which on a GPU may hang or
//    may carry on and compute something wrong.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.
import { tr } from '../lib/locale.ts';
import type { Schedule } from './tiled';
import type { WarpState } from './warps';

export type { Schedule, WarpState };

// ---------------------------------------------------------------- constants
/** The warp width the book uses throughout (pp. 230, 238). */
export const BOOK_WARP = 32;
/** One block of at most 1024 threads, so at most 2048 values (p. 228). */
export const MAX_THREADS = 1024;
/** The figures the book prints on p. 234 that its own formula on p. 231 contradicts. */
export const BOOK_PRINTED = { simpleSlots: 736, divergentBefore: 10, divergentAfter: 5 } as const;
/** Exercise 6 (p. 250). */
export const EXERCISE_6: readonly number[] = [6, 2, 7, 4, 5, 8, 3, 1];
/** Sixteen values for the 16-element figures: the leaves of Fig. 10.3(a) then of Fig. 10.3(b). */
export const FIG_VALUES: readonly number[] = [3, 1, 7, 0, 4, 1, 6, 3, 4, 7, 2, 3, 8, 5, 9, 6];

export const isPow2 = (n: number): boolean => Number.isInteger(n) && n >= 1 && (n & (n - 1)) === 0;
const sum = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0);

/** Throws unless one block can reduce N values: N a power of two from 2 to 2048. */
export function checkN(N: number): void {
  if (!isPow2(N) || N < 2 || N > 2 * MAX_THREADS)
    throw new Error(`N must be a power of two from 2 to ${2 * MAX_THREADS}, not ${N}`);
}
function checkWidth(width: number): void {
  if (!Number.isInteger(width) || width < 1) throw new Error(`a warp width is a whole number, 1 or more, not ${width}`);
}

// ---------------------------------------------------------------- the loops
export type InPlaceKind = 'simple' | 'convergent';
export type Kind = InPlaceKind | 'shared';
export const KIND_LABEL: Record<Kind, string> = {
  get simple() { return tr('simple (Fig. 10.5)', 'đơn giản (Fig. 10.5)'); },
  get convergent() { return tr('convergent (Fig. 10.8)', 'hội tụ (Fig. 10.8)'); },
  get shared() { return tr('shared memory (Fig. 10.10)', 'bộ nhớ chia sẻ (Fig. 10.10)'); },
};

/**
 * The strides of the loop for a block of `threads` threads, in order.
 * simple: 1, 2, 4, ... while stride <= blockDim.x. convergent: blockDim.x, ..., 1.
 * shared: blockDim.x/2, ..., 1 (the load did the first level).
 */
export function strides(kind: Kind, threads: number): number[] {
  const out: number[] = [];
  if (kind === 'simple') for (let s = 1; s <= threads; s *= 2) out.push(s);
  else for (let s = kind === 'convergent' ? threads : threads / 2; s >= 1; s = Math.floor(s / 2)) out.push(s);
  return out;
}

/** The active test: threadIdx.x % stride == 0 (simple) or threadIdx.x < stride. */
export const isActive = (kind: Kind, t: number, stride: number): boolean => (kind === 'simple' ? t % stride === 0 : t < stride);

/** The element thread t owns in an in-place kernel: 2t (simple) or t (convergent). */
export const ownedCell = (kind: InPlaceKind, t: number): number => (kind === 'simple' ? 2 * t : t);

/** The thread that owns element c, or null when no thread does (it is only ever read). */
export function ownerOf(kind: InPlaceKind, c: number, threads: number): number | null {
  if (kind === 'simple') return c % 2 === 0 && c / 2 < threads ? c / 2 : null;
  return c < threads ? c : null;
}

/** The array after each iteration, done in order with every barrier respected (exercise 6). */
export function arraysAfter(kind: InPlaceKind, values: readonly number[]): number[][] {
  checkN(values.length);
  const T = values.length / 2;
  const a = [...values];
  return strides(kind, T).map((s) => {
    for (let t = 0; t < T; t++) if (isActive(kind, t, s)) a[ownedCell(kind, t)] += a[ownedCell(kind, t) + s];
    return [...a];
  });
}

// ---------------------------------------------------------------- 1. warps and efficiency
export interface WarpRow {
  warp: number;
  /** One entry per lane: true active, false inactive, null a lane with no thread. */
  lanes: (boolean | null)[];
  active: number;
  /** Lanes that hold a thread. */
  real: number;
  state: WarpState;
}
export interface Iteration {
  /** 1-based, as the book counts on pp. 230 to 234. */
  n: number;
  stride: number;
  active: number;
  warps: WarpRow[];
  activeWarps: number;
  divergentWarps: number;
  /** Thread slots this iteration consumes: every warp with an active lane, at full width. */
  slots: number;
}
export interface Trace {
  kind: InPlaceKind;
  N: number;
  width: number;
  threads: number;
  warpCount: number;
  iterations: Iteration[];
  slots: number;
  /** Additions performed: N - 1. */
  results: number;
  efficiency: number;
  /** Iterations in which at least one warp diverges. */
  divergentIterations: number;
}

/** The book's accounting of one block's loop at one warp width (pp. 230 to 234). */
export function iterations(kind: InPlaceKind, N: number, width: number): Trace {
  checkN(N);
  checkWidth(width);
  const T = N / 2;
  const warpCount = Math.ceil(T / width);
  const its: Iteration[] = strides(kind, T).map((stride, k) => {
    const warps: WarpRow[] = [];
    for (let w = 0; w < warpCount; w++) {
      const lanes: (boolean | null)[] = [];
      let active = 0;
      let real = 0;
      for (let l = 0; l < width; l++) {
        const t = w * width + l;
        if (t >= T) {
          lanes.push(null);
          continue;
        }
        real++;
        const a = isActive(kind, t, stride);
        if (a) active++;
        lanes.push(a);
      }
      warps.push({ warp: w, lanes, active, real, state: active === 0 ? 'idle' : active === real ? 'full' : 'divergent' });
    }
    const activeWarps = warps.filter((w) => w.active > 0).length;
    return {
      n: k + 1,
      stride,
      active: sum(warps.map((w) => w.active)),
      warps,
      activeWarps,
      divergentWarps: warps.filter((w) => w.state === 'divergent').length,
      slots: activeWarps * width,
    };
  });
  const slots = sum(its.map((i) => i.slots));
  const results = sum(its.map((i) => i.active));
  return {
    kind,
    N,
    width,
    threads: T,
    warpCount,
    iterations: its,
    slots,
    results,
    efficiency: results / slots,
    divergentIterations: its.filter((i) => i.divergentWarps > 0).length,
  };
}

/** Divergent warps in one iteration (1-based), for exercises 1 and 2 and the quiz. */
export const divergentAt = (kind: InPlaceKind, N: number, width: number, n: number): number =>
  iterations(kind, N, width).iterations[n - 1].divergentWarps;

/**
 * The book's own formulas, written out term by term as printed, for 32-wide warps and
 * N >= 64 (N/64 warps). p. 231: (N/64*6 + N/64*1/2 + N/64*1/4 + ... + 1) * 32.
 * p. 234: (N/64*1 + N/64*1/2 + ... + 1 + 5*1) * 32. p. 235: (N/64*5*2 + N/64*1 + N/64*1/2
 * + ... + 1) * 3. pp. 236 to 237: N/32 + 1.
 */
const halving = (from: number): number => {
  let s = 0;
  for (let x = from; x >= 1; x /= 2) s += x;
  return s;
};
export const bookFormula = {
  simpleSlots: (N: number): number => (N / 64 * 6 + halving(N / 128)) * 32,
  convergentSlots: (N: number): number => (halving(N / 64) + 5) * 32,
  simpleRequests: (N: number): number => (N / 64 * 5 * 2 + halving(N / 64)) * 3,
  sharedRequests: (N: number): number => N / 32 + 1,
};

// ---------------------------------------------------------------- 2. global-memory requests
export type AccessName = 'own' | 'partner' | 'write' | 'load-lo' | 'load-hi' | 'store';
export const ACCESS_LABEL: Record<AccessName, string> = {
  get own() { return tr('read own element', 'đọc phần tử của mình'); },
  get partner() { return tr('read partner', 'đọc phần tử đối tác'); },
  get write() { return tr('write own element', 'ghi phần tử của mình'); },
  get 'load-lo'() { return tr('load input[t]', 'nạp input[t]'); },
  get 'load-hi'() { return tr('load input[t + blockDim.x]', 'nạp input[t + blockDim.x]'); },
  get store() { return tr('write *output', 'ghi *output'); },
};
export interface WarpAccess {
  name: AccessName;
  array: 'input' | 'output';
  /** The elements the warp's active lanes touch in this access. */
  cells: number[];
  /** The aligned pieces of `segment` floats those elements fall in. */
  segments: number[];
}
export interface RequestWarp {
  warp: number;
  accesses: WarpAccess[];
  requests: number;
}
export interface RequestStep {
  label: string;
  phase: 'load' | 'iteration' | 'store';
  /** Iteration number (1-based), 0 for the load and the store. */
  n: number;
  stride: number | null;
  /** Only the warps that make at least one request. */
  warps: RequestWarp[];
  requests: number;
  byAccess: Partial<Record<AccessName, number>>;
}
export interface RequestTrace {
  kind: Kind;
  N: number;
  width: number;
  segment: number;
  steps: RequestStep[];
  total: number;
}

export const segmentsOf = (cells: readonly number[], segment: number): number[] =>
  [...new Set(cells.map((c) => Math.floor(c / segment)))].sort((a, b) => a - b);

/**
 * Global-memory requests by our rule: in each access (one load or store instruction), a
 * warp makes one request per aligned piece of `segment` floats that its active lanes
 * touch. The in-place kernels' last read of input[0] and write of *output are left out,
 * as the book's formula leaves them out; the shared-memory kernel's one write is counted,
 * as the book counts it (N + 1).
 */
export function requests(kind: Kind, N: number, width: number, segment = 32): RequestTrace {
  checkN(N);
  checkWidth(width);
  const T = N / 2;
  const warpCount = Math.ceil(T / width);
  const lanesOf = (w: number) => {
    const out: number[] = [];
    for (let t = w * width; t < Math.min(T, (w + 1) * width); t++) out.push(t);
    return out;
  };
  const acc = (name: AccessName, array: 'input' | 'output', cells: number[]): WarpAccess => ({
    name, array, cells, segments: segmentsOf(cells, segment),
  });
  const finish = (label: string, phase: RequestStep['phase'], n: number, stride: number | null, warps: RequestWarp[]): RequestStep => {
    const byAccess: Partial<Record<AccessName, number>> = {};
    for (const w of warps) for (const a of w.accesses) byAccess[a.name] = (byAccess[a.name] ?? 0) + a.segments.length;
    return { label, phase, n, stride, warps, requests: sum(warps.map((w) => w.requests)), byAccess };
  };
  const steps: RequestStep[] = [];
  if (kind === 'shared') {
    const warps: RequestWarp[] = [];
    for (let w = 0; w < warpCount; w++) {
      const ts = lanesOf(w);
      const accesses = [acc('load-lo', 'input', ts), acc('load-hi', 'input', ts.map((t) => t + T))];
      warps.push({ warp: w, accesses, requests: sum(accesses.map((a) => a.segments.length)) });
    }
    steps.push(finish(tr('load and first addition', 'nạp và phép cộng đầu tiên'), 'load', 0, null, warps));
    strides('shared', T).forEach((s, k) => steps.push(finish(tr(`iteration ${k + 1}, stride ${s}`, `vòng lặp ${k + 1}, bước nhảy ${s}`), 'iteration', k + 1, s, [])));
    const store = acc('store', 'output', [0]);
    steps.push(finish(tr('thread 0 writes the total', 'luồng 0 ghi tổng'), 'store', 0, null, [{ warp: 0, accesses: [store], requests: 1 }]));
  } else {
    strides(kind, T).forEach((s, k) => {
      const warps: RequestWarp[] = [];
      for (let w = 0; w < warpCount; w++) {
        const ts = lanesOf(w).filter((t) => isActive(kind, t, s));
        if (!ts.length) continue;
        const own = ts.map((t) => ownedCell(kind, t));
        const accesses = [acc('own', 'input', own), acc('partner', 'input', own.map((c) => c + s)), acc('write', 'input', own)];
        warps.push({ warp: w, accesses, requests: sum(accesses.map((a) => a.segments.length)) });
      }
      steps.push(finish(tr(`iteration ${k + 1}, stride ${s}`, `vòng lặp ${k + 1}, bước nhảy ${s}`), 'iteration', k + 1, s, warps));
    });
  }
  return { kind, N, width, segment, steps, total: sum(steps.map((s) => s.requests)) };
}

// ---------------------------------------------------------------- 3. exercise 3: a kernel from parts
// Fig. 10.8's loop (stride from blockDim.x down to 1, halving, a barrier after the if) with
// four parts to choose. B is blockDim.x; the input has 2B elements.
export type OwnedKey = 't' | '2t' | 't+B' | '2t+1';
export type TestKey = 'lt' | 'ge' | 'mirror' | 'mod';
export type PartnerKey = 'plus' | 'minus';
export type WriterKey = 'first' | 'last' | 'middle';
export interface Spec {
  owned: OwnedKey;
  test: TestKey;
  partner: PartnerKey;
  writer: WriterKey;
}
export const OWNED: Record<OwnedKey, { code: string; at: (t: number, B: number) => number }> = {
  t: { code: 'threadIdx.x', at: (t) => t },
  '2t': { code: '2*threadIdx.x', at: (t) => 2 * t },
  't+B': { code: 'threadIdx.x + blockDim.x', at: (t, B) => t + B },
  '2t+1': { code: '2*threadIdx.x + 1', at: (t) => 2 * t + 1 },
};
export const TESTS: Record<TestKey, { code: string; on: (t: number, s: number, B: number) => boolean }> = {
  lt: { code: 'threadIdx.x < stride', on: (t, s) => t < s },
  ge: { code: 'threadIdx.x >= stride', on: (t, s) => t >= s },
  mirror: { code: 'threadIdx.x >= blockDim.x - stride', on: (t, s, B) => t >= B - s },
  mod: { code: 'threadIdx.x % stride == 0', on: (t, s) => t % s === 0 },
};
export const PARTNERS: Record<PartnerKey, { code: string; at: (i: number, s: number) => number }> = {
  plus: { code: 'input[i + stride]', at: (i, s) => i + s },
  minus: { code: 'input[i - stride]', at: (i, s) => i - s },
};
export const WRITERS: Record<WriterKey, { code: string; thread: (B: number) => number; cell: (B: number) => number }> = {
  first: { code: 'if (threadIdx.x == 0) *output = input[0];', thread: () => 0, cell: () => 0 },
  last: { code: 'if (threadIdx.x == blockDim.x - 1) *output = input[2*blockDim.x - 1];', thread: (B) => B - 1, cell: (B) => 2 * B - 1 },
  middle: { code: 'if (threadIdx.x == 0) *output = input[blockDim.x];', thread: () => 0, cell: (B) => B },
};
/** Fig. 10.8 itself. */
export const CONVERGENT_SPEC: Spec = { owned: 't', test: 'lt', partner: 'plus', writer: 'first' };
/** Our answer to exercise 3: labs/ch10/snippets/s104_mirrored.cu. */
export const MIRROR_SPEC: Spec = { owned: 't+B', test: 'mirror', partner: 'minus', writer: 'last' };

export interface SpecOp {
  t: number;
  own: number;
  partner: number;
  /** The partner index is outside the input (an unsigned i - stride wraps around). */
  oob: boolean;
}
export interface SpecIteration {
  n: number;
  stride: number;
  ops: SpecOp[];
  /** Elements read in this iteration that another thread writes in the same iteration. */
  conflicts: number[];
  /** The input after the iteration (reads of an iteration see the array before it). */
  after: number[];
}
export interface SpecRun {
  spec: Spec;
  B: number;
  iterations: SpecIteration[];
  output: number;
  total: number;
  /** The last element holds the total at the end. */
  totalInLast: boolean;
  oob: number;
  conflicts: number;
  correct: boolean;
}

export function runSpec(spec: Spec, values: readonly number[]): SpecRun {
  checkN(values.length);
  const N = values.length;
  const B = N / 2;
  const a = [...values];
  const total = sum(values);
  let oob = 0;
  let conflicts = 0;
  const its: SpecIteration[] = strides('convergent', B).map((s, k) => {
    const ops: SpecOp[] = [];
    for (let t = 0; t < B; t++) {
      if (!TESTS[spec.test].on(t, s, B)) continue;
      const own = OWNED[spec.owned].at(t, B);
      const partner = PARTNERS[spec.partner].at(own, s);
      ops.push({ t, own, partner, oob: partner < 0 || partner >= N || own >= N });
    }
    const written = new Map(ops.map((o) => [o.own, o.t]));
    const clash = new Set<number>();
    for (const o of ops) {
      const w = written.get(o.partner);
      if (w !== undefined && w !== o.t) clash.add(o.partner);
    }
    const before = [...a];
    for (const o of ops) a[o.own] = (before[o.own] ?? NaN) + (o.oob ? NaN : before[o.partner]);
    oob += ops.filter((o) => o.oob).length;
    conflicts += clash.size;
    return { n: k + 1, stride: s, ops, conflicts: [...clash].sort((x, y) => x - y), after: [...a] };
  });
  const output = a[WRITERS[spec.writer].cell(B)];
  return {
    spec,
    B,
    iterations: its,
    output,
    total,
    totalInLast: a[N - 1] === total,
    oob,
    conflicts,
    correct: output === total && oob === 0 && conflicts === 0,
  };
}

// ---------------------------------------------------------------- 4. threads, one operation at a time
export type Arr = 'input' | 'shared' | 'output';
/** Which value a cell holds: the reduction it belongs to (0 or 1) and the level: 0 before the loop, k written in iteration k. */
export interface Version {
  phase: number;
  level: number;
}
export type OpKind = 'skip' | 'add' | 'max' | 'load' | 'load-max' | 'load-diff' | 'read-max' | 'barrier' | 'store';
export interface Read {
  arr: Arr;
  index: number;
  /** The version a correct run finds here; null when nothing needs checking (a thread's own element, the input of a shared-memory kernel). */
  expect: Version | null;
}
export interface Op {
  kind: OpKind;
  /** Source lines to highlight, numbered from 1 = the function header. */
  lines: number[];
  /** Which reduction: 0, or 1 for the second of two. */
  phase: number;
  /** Loop iteration (1-based), 0 before the loop, -1 after it. */
  iter: number;
  stride: number;
  reads: Read[];
  write?: { arr: Arr; index: number; version: Version };
}

export type IssueKind = 'stale' | 'overwritten' | 'oob' | 'deadlock';
export interface Issue {
  kind: IssueKind;
  step: number;
  thread: number;
  message: string;
}
export interface StepRead extends Read {
  value: number;
  found: Version | null;
  issue?: IssueKind;
}
export interface TStep {
  index: number;
  /** The thread, -1 for a barrier opening, -2 for a block that can never move again. */
  thread: number;
  op?: Op;
  reads: StepRead[];
  write?: { arr: Arr; index: number; value: number; version: Version };
  issue?: Issue;
  /** Barrier opening: its line and the threads let go. */
  site?: number;
  released?: number[];
}
export type Outcome = 'correct' | 'wrong' | 'deadlock' | 'oob';
export const OUTCOME_LABEL: Record<Outcome, string> = {
  get correct() { return tr('correct', 'đúng'); },
  get wrong() { return tr('wrong answer', 'sai kết quả'); },
  get deadlock() { return tr('stuck at a barrier', 'kẹt tại rào chắn'); },
  get oob() { return tr('out of bounds', 'vượt biên'); },
};
export interface ThreadRun {
  /** The function in labs/ch10/snippets that this run models. */
  kernel: string;
  file: string;
  threads: number;
  schedule: Schedule;
  /** Each thread's list of operations. */
  program: Op[][];
  initial: { input: number[]; shared: number[] };
  steps: TStep[];
  issues: Issue[];
  /** What *output holds at the end: NaN if nothing (or garbage) was written. */
  output: number;
  expected: number;
  outcome: Outcome;
}

// The line numbers below are pinned to the snippet files by reduction.test.ts.
export const INPLACE_FILE = 'ch10/snippets/s103_inplace.cu';
export const INPLACE_KERNEL: Record<InPlaceKind, string> = {
  simple: 'SimpleSumReductionKernel',
  convergent: 'ConvergentSumReductionKernel',
};
export const INPLACE_LINES = { index: 2, loop: 3, test: 4, add: 5, barrier: 7, storeTest: 9, store: 10 } as const;

export type Variant = 'book' | 'bottom' | 'none' | 'inside-if' | 'le' | 'two' | 'two-race';
export const VARIANTS: readonly Variant[] = ['book', 'bottom', 'none', 'inside-if', 'le', 'two', 'two-race'];
export const HUNT_FILE = 'ch10/snippets/s106_barrier_hunt.cu';
interface VariantDef {
  kernel: string;
  label: string;
  /** Barrier placement: at the top of the loop body, at the bottom, inside the if, or none. */
  barrier: 'top' | 'bottom' | 'if' | 'none';
  le: boolean;
  /** Two reductions (max, then a sum of x - max), and whether a barrier follows the read of the max. */
  two: null | { barrierAfterRead: boolean };
  lines: {
    load: number[];
    barrier?: number;
    test: number;
    add: number;
    store: number;
    afterLoop?: number;
    readMax?: number;
    afterRead?: number;
    load2?: number[];
    barrier2?: number;
    test2?: number;
    add2?: number;
  };
}
export const VARIANT: Record<Variant, VariantDef> = {
  book: {
    kernel: 'SharedMemorySumReductionKernel', get label() { return tr('as in the book (barrier at the top of the loop)', 'như trong sách (rào chắn ở đầu vòng lặp)'); },
    barrier: 'top', le: false, two: null, lines: { load: [4], barrier: 6, test: 7, add: 8, store: 12 },
  },
  bottom: {
    kernel: 'BarrierAtBottomKernel', get label() { return tr('barrier at the bottom of the loop only', 'chỉ có rào chắn ở cuối vòng lặp'); },
    barrier: 'bottom', le: false, two: null, lines: { load: [4], test: 6, add: 7, barrier: 9, store: 12 },
  },
  none: {
    kernel: 'NoBarrierKernel', get label() { return tr('no barrier', 'không có rào chắn'); },
    barrier: 'none', le: false, two: null, lines: { load: [4], test: 6, add: 7, store: 11 },
  },
  'inside-if': {
    kernel: 'BarrierInsideIfKernel', get label() { return tr('barrier inside the if', 'rào chắn bên trong if'); },
    barrier: 'if', le: false, two: null, lines: { load: [4], test: 6, barrier: 7, add: 8, store: 12 },
  },
  le: {
    kernel: 'LessOrEqualKernel', get label() { return tr('active test <= instead of <', 'phép kiểm tra luồng còn hoạt động dùng <= thay vì <'); },
    barrier: 'top', le: true, two: null, lines: { load: [4], barrier: 6, test: 7, add: 8, store: 12 },
  },
  two: {
    kernel: 'MaxThenSumKernel', get label() { return tr('max, then sum, one scratch array, barrier after the read', 'max, rồi tổng, một mảng nháp, có rào chắn sau lần đọc'); },
    barrier: 'top', le: false, two: { barrierAfterRead: true },
    lines: { load: [4, 5], barrier: 7, test: 8, add: 9, afterLoop: 12, readMax: 13, afterRead: 14, load2: [15], barrier2: 17, test2: 18, add2: 19, store: 23 },
  },
  'two-race': {
    kernel: 'MaxThenSumNoBarrierKernel', get label() { return tr('max, then sum, one scratch array, no barrier after the read', 'max, rồi tổng, một mảng nháp, không có rào chắn sau lần đọc'); },
    barrier: 'top', le: false, two: { barrierAfterRead: false },
    lines: { load: [4, 5], barrier: 7, test: 8, add: 9, afterLoop: 12, readMax: 13, load2: [14], barrier2: 16, test2: 17, add2: 18, store: 22 },
  },
};

const cmpVersion = (a: Version, b: Version): number => a.phase - b.phase || a.level - b.level;
const range = (n: number): number[] => Array.from({ length: n }, (_, i) => i);

/** Last iteration j < n of `ss` in which test(owner, ss[j-1]) holds, as a level (0 if none). */
function lastLevel(ss: readonly number[], n: number, on: (stride: number) => boolean): number {
  let level = 0;
  for (let j = 1; j < n; j++) if (on(ss[j - 1])) level = j;
  return level;
}

/** Each thread's operations for an in-place kernel, as s103_inplace.cu writes them. */
function inPlaceProgram(kind: InPlaceKind, T: number, barrier: boolean): Op[][] {
  const L = INPLACE_LINES;
  const ss = strides(kind, T);
  return range(T).map((t) => {
    const ops: Op[] = [];
    const i = ownedCell(kind, t);
    ss.forEach((s, k) => {
      const n = k + 1;
      if (isActive(kind, t, s)) {
        const c = i + s;
        const owner = ownerOf(kind, c, T);
        const level = owner == null ? 0 : lastLevel(ss, n, (st) => isActive(kind, owner, st));
        ops.push({
          kind: 'add', lines: [L.add], phase: 0, iter: n, stride: s,
          reads: [{ arr: 'input', index: i, expect: null }, { arr: 'input', index: c, expect: { phase: 0, level } }],
          write: { arr: 'input', index: i, version: { phase: 0, level: n } },
        });
      } else ops.push({ kind: 'skip', lines: [L.test], phase: 0, iter: n, stride: s, reads: [] });
      if (barrier) ops.push({ kind: 'barrier', lines: [L.barrier], phase: 0, iter: n, stride: s, reads: [] });
    });
    if (t === 0)
      ops.push({
        kind: 'store', lines: [L.store], phase: 0, iter: -1, stride: 0,
        reads: [{ arr: 'input', index: 0, expect: null }],
        write: { arr: 'output', index: 0, version: { phase: 0, level: ss.length } },
      });
    return ops;
  });
}

/** Each thread's operations for a variant of the shared-memory kernel, as s106_barrier_hunt.cu writes them. */
function sharedProgram(v: Variant, B: number): Op[][] {
  const d = VARIANT[v];
  const ln = d.lines;
  const ss = strides('shared', B);
  const L = ss.length;
  const on = (t: number, s: number) => (d.le ? t <= s : t < s);
  const tree = (t: number, phase: number, lines: { barrier?: number; test: number; add: number }, ops: Op[]) => {
    ss.forEach((s, k) => {
      const n = k + 1;
      const bar = (): Op => ({ kind: 'barrier', lines: [lines.barrier!], phase, iter: n, stride: s, reads: [] });
      if (d.barrier === 'top') ops.push(bar());
      if (on(t, s)) {
        if (d.barrier === 'if') ops.push(bar());
        const c = t + s;
        const expect: Version | null = c < B ? { phase, level: lastLevel(ss, n, (st) => on(c, st)) } : null;
        ops.push({
          kind: phase === 0 && d.two ? 'max' : 'add', lines: [lines.add], phase, iter: n, stride: s,
          reads: [{ arr: 'shared', index: t, expect: null }, { arr: 'shared', index: c, expect }],
          write: { arr: 'shared', index: t, version: { phase, level: n } },
        });
      } else ops.push({ kind: 'skip', lines: [lines.test], phase, iter: n, stride: s, reads: [] });
      if (d.barrier === 'bottom') ops.push(bar());
    });
  };
  return range(B).map((t) => {
    const ops: Op[] = [];
    ops.push({
      kind: d.two ? 'load-max' : 'load', lines: ln.load, phase: 0, iter: 0, stride: 0,
      reads: [{ arr: 'input', index: t, expect: null }, { arr: 'input', index: t + B, expect: null }],
      write: { arr: 'shared', index: t, version: { phase: 0, level: 0 } },
    });
    tree(t, 0, { barrier: ln.barrier, test: ln.test, add: ln.add }, ops);
    let phase = 0;
    if (d.two) {
      phase = 1;
      ops.push({ kind: 'barrier', lines: [ln.afterLoop!], phase: 0, iter: -1, stride: 0, reads: [] });
      ops.push({ kind: 'read-max', lines: [ln.readMax!], phase: 0, iter: -1, stride: 0, reads: [{ arr: 'shared', index: 0, expect: { phase: 0, level: L } }] });
      if (d.two.barrierAfterRead) ops.push({ kind: 'barrier', lines: [ln.afterRead!], phase: 0, iter: -1, stride: 0, reads: [] });
      ops.push({
        kind: 'load-diff', lines: ln.load2!, phase: 1, iter: 0, stride: 0, reads: [],
        write: { arr: 'shared', index: t, version: { phase: 1, level: 0 } },
      });
      tree(t, 1, { barrier: ln.barrier2, test: ln.test2!, add: ln.add2! }, ops);
    }
    if (t === 0)
      ops.push({
        kind: 'store', lines: [ln.store], phase, iter: -1, stride: 0,
        reads: [{ arr: 'shared', index: 0, expect: { phase, level: L } }],
        write: { arr: 'output', index: 0, version: { phase, level: L } },
      });
    return ops;
  });
}

function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A value as the steppers print it: "?" for garbage (NaN). */
export const fmtValue = (v: number): string => (Number.isNaN(v) ? '?' : String(Math.round(v * 1000) / 1000));
const arrName = (arr: Arr, kernel: string): string =>
  arr === 'shared' ? (kernel.startsWith('MaxThenSum') ? 'buf_s' : 'input_s') : arr === 'output' ? '*output' : 'input';
const cellName = (arr: Arr, index: number, kernel: string): string =>
  arr === 'output' ? '*output' : `${arrName(arr, kernel)}[${index}]`;
export const threadList = (ts: readonly number[]): string => {
  if (!ts.length) return tr('no thread', 'không luồng nào');
  const sorted = [...ts].sort((a, b) => a - b);
  const contiguous = sorted.every((t, i) => i === 0 || t === sorted[i - 1] + 1);
  if (sorted.length === 1) return tr(`thread ${sorted[0]}`, `luồng ${sorted[0]}`);
  if (contiguous && sorted.length > 2) return tr(`threads ${sorted[0]} to ${sorted[sorted.length - 1]}`, `các luồng ${sorted[0]} đến ${sorted[sorted.length - 1]}`);
  return tr(`threads ${sorted.slice(0, -1).join(', ')} and ${sorted[sorted.length - 1]}`, `các luồng ${sorted.slice(0, -1).join(', ')} và ${sorted[sorted.length - 1]}`);
};

interface Machine {
  input: number[];
  shared: number[];
  output: number[];
  ver: Record<Arr, (Version | null)[]>;
  regs: { a: number; b: number; m: number }[];
}

/** Runs a program under one thread order and records every step. */
function execute(
  program: Op[][],
  machine: Machine,
  schedule: Schedule,
  seed: number,
  kernel: string,
  describeVersion: (v: Version, arr: Arr, index: number) => string,
): { steps: TStep[]; issues: Issue[] } {
  const T = program.length;
  const pc = new Array<number>(T).fill(0);
  const waiting = new Array<boolean>(T).fill(false);
  const steps: TStep[] = [];
  const issues: Issue[] = [];
  const rand = mulberry(seed);
  const arrays: Record<Arr, number[]> = { input: machine.input, shared: machine.shared, output: machine.output };

  const flag = (s: TStep, kind: IssueKind, message: string) => {
    const issue: Issue = { kind, step: s.index, thread: s.thread, message };
    if (!s.issue) s.issue = issue;
    issues.push(issue);
  };

  /** Performs the reads of thread t's next operation and works out what it writes. */
  const prepare = (t: number): { step: TStep; commit: () => void } => {
    const op = program[t][pc[t]];
    const step: TStep = { index: steps.length, thread: t, op, reads: [] };
    steps.push(step);
    pc[t]++;
    if (op.kind === 'barrier') {
      waiting[t] = true;
      return { step, commit: () => undefined };
    }
    for (const r of op.reads) {
      const arr = arrays[r.arr];
      const inside = r.index >= 0 && r.index < arr.length;
      const value = inside ? arr[r.index] : NaN;
      const found = inside ? machine.ver[r.arr][r.index] : null;
      const sr: StepRead = { ...r, value, found };
      step.reads.push(sr);
      const where = cellName(r.arr, r.index, kernel);
      if (!inside) {
        sr.issue = 'oob';
        flag(step, 'oob', tr(`thread ${t} read ${where}, past the end of the ${arr.length}-element array`, `luồng ${t} đọc ${where}, vượt quá cuối mảng ${arr.length} phần tử`));
      } else if (r.expect) {
        if (!found || cmpVersion(found, r.expect) < 0) {
          sr.issue = 'stale';
          flag(step, 'stale', tr(`thread ${t} read ${where} before ${describeVersion(r.expect, r.arr, r.index)} was there`, `luồng ${t} đọc ${where} trước khi ${describeVersion(r.expect, r.arr, r.index)} có ở đó`) +
            (found ? tr(`; it found ${describeVersion(found, r.arr, r.index)}`, `; nó thấy ${describeVersion(found, r.arr, r.index)}`) : tr('; nobody had written it yet', '; chưa ai ghi vào đó')));
        } else if (cmpVersion(found, r.expect) > 0) {
          sr.issue = 'overwritten';
          flag(step, 'overwritten', tr(
            `thread ${t} needed ${describeVersion(r.expect, r.arr, r.index)} in ${where}, but it had already been overwritten with ${describeVersion(found, r.arr, r.index)}`,
            `luồng ${t} cần ${describeVersion(r.expect, r.arr, r.index)} trong ${where}, nhưng ô đó đã bị ghi đè bằng ${describeVersion(found, r.arr, r.index)}`,
          ));
        }
      }
    }
    const v = step.reads.map((r) => r.value);
    const reg = machine.regs[t];
    let out: number | undefined;
    switch (op.kind) {
      case 'add':
      case 'load':
        out = v[0] + v[1];
        break;
      case 'max':
        out = Number.isNaN(v[0]) || Number.isNaN(v[1]) ? NaN : Math.max(v[0], v[1]);
        break;
      case 'load-max':
        reg.a = v[0];
        reg.b = v[1];
        out = Number.isNaN(v[0]) || Number.isNaN(v[1]) ? NaN : Math.max(v[0], v[1]);
        break;
      case 'load-diff':
        out = reg.a - reg.m + (reg.b - reg.m);
        break;
      case 'read-max':
        reg.m = v[0];
        break;
      case 'store':
        out = v[0];
        break;
      default:
        break;
    }
    if (op.write && out !== undefined) step.write = { ...op.write, value: out };
    return {
      step,
      commit: () => {
        if (!step.write) return;
        arrays[step.write.arr][step.write.index] = step.write.value;
        machine.ver[step.write.arr][step.write.index] = step.write.version;
      },
    };
  };

  const live = (t: number) => pc[t] < program[t].length;
  const runnable = (t: number) => live(t) && !waiting[t];
  /** The book's rule: a barrier opens when every thread of the block waits at the same call. */
  const release = (): boolean => {
    if (!waiting.some(Boolean)) return false;
    const sites = new Set(range(T).filter((t) => waiting[t]).map((t) => program[t][pc[t] - 1].lines[0]));
    if (sites.size !== 1 || !waiting.every(Boolean)) return false;
    const site = [...sites][0];
    steps.push({ index: steps.length, thread: -1, reads: [], site, released: range(T) });
    waiting.fill(false);
    return true;
  };

  const order = range(T);
  if (schedule === 'reversed') order.reverse();
  for (let guard = 0; guard < 1e6; guard++) {
    if (!order.some(live)) break;
    if (schedule === 'lockstep') {
      const round = order.filter(runnable).map(prepare);
      for (const r of round) r.commit();
    } else if (schedule === 'shuffled') {
      const cands = order.filter(runnable);
      if (cands.length) prepare(cands[Math.floor(rand() * cands.length)]).commit();
    } else {
      for (const t of order) while (runnable(t)) prepare(t).commit();
    }
    if (order.some(runnable)) continue;
    if (release()) continue;
    if (order.some(live)) {
      // Nothing can move: some threads wait at a barrier that cannot open.
      const bySite = new Map<number, number[]>();
      for (const t of order) if (waiting[t]) {
        const line = program[t][pc[t] - 1].lines[0];
        bySite.set(line, [...(bySite.get(line) ?? []), t]);
      }
      const ended = order.filter((t) => !live(t));
      const parts = [...bySite.entries()].map(([line, ts]) => tr(`${threadList(ts)} ${ts.length === 1 ? 'waits' : 'wait'} at the barrier on line ${line}`, `${threadList(ts)} chờ tại rào chắn ở dòng ${line}`));
      const step: TStep = { index: steps.length, thread: -2, reads: [] };
      steps.push(step);
      flag(step, 'deadlock',
        parts.join('; ') + (ended.length ? tr(`; ${threadList(ended)} ended without reaching it`, `; ${threadList(ended)} kết thúc mà không đến được đó`) : '') +
          tr('. Every thread of the block must reach the same barrier, so the waiting threads can never leave it', '. Mọi luồng của khối phải đến cùng một rào chắn, nên các luồng đang chờ không bao giờ rời khỏi được'));
      break;
    }
  }
  return { steps, issues };
}

const near = (a: number, b: number): boolean => Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b));
function outcomeOf(issues: Issue[], output: number, expected: number): Outcome {
  if (issues.some((i) => i.kind === 'deadlock')) return 'deadlock';
  if (issues.some((i) => i.kind === 'oob')) return 'oob';
  return near(output, expected) ? 'correct' : 'wrong';
}

export interface InPlaceOptions {
  barrier?: boolean;
  schedule: Schedule;
  seed?: number;
}

/** One block of an in-place kernel (Fig. 10.5 or 10.8) on these values, thread by thread. */
export function runInPlace(kind: InPlaceKind, values: readonly number[], opts: InPlaceOptions): ThreadRun {
  checkN(values.length);
  const T = values.length / 2;
  const program = inPlaceProgram(kind, T, opts.barrier ?? true);
  const ss = strides(kind, T);
  const machine: Machine = {
    input: [...values],
    shared: [],
    output: [NaN],
    ver: { input: values.map(() => ({ phase: 0, level: 0 })), shared: [], output: [null] },
    regs: range(T).map(() => ({ a: NaN, b: NaN, m: NaN })),
  };
  const describe = (v: Version, _arr: Arr, index: number) => {
    const owner = ownerOf(kind, index, T);
    if (v.level === 0) return tr('the original value', 'giá trị ban đầu');
    return tr(`thread ${owner}'s partial from iteration ${v.level} (stride ${ss[v.level - 1]})`, `tổng riêng phần của luồng ${owner} từ vòng lặp ${v.level} (bước nhảy ${ss[v.level - 1]})`);
  };
  const kernel = INPLACE_KERNEL[kind];
  const { steps, issues } = execute(program, machine, opts.schedule, opts.seed ?? 1, kernel, describe);
  const output = machine.output[0];
  const expected = sum(values);
  return {
    kernel, file: INPLACE_FILE, threads: T, schedule: opts.schedule, program,
    initial: { input: [...values], shared: [] }, steps, issues, output, expected,
    outcome: outcomeOf(issues, output, expected),
  };
}

/** What a variant must produce: the sum, or for the two reductions the sum of x - max. */
export function expectedOf(v: Variant, values: readonly number[]): number {
  return VARIANT[v].two ? sum(values) - values.length * Math.max(...values) : sum(values);
}

/** One block of a variant of the shared-memory kernel (s106_barrier_hunt.cu) on these values. */
export function runShared(v: Variant, values: readonly number[], schedule: Schedule, seed = 1): ThreadRun {
  checkN(values.length);
  const B = values.length / 2;
  const d = VARIANT[v];
  const program = sharedProgram(v, B);
  const ss = strides('shared', B);
  const machine: Machine = {
    input: [...values],
    shared: new Array<number>(B).fill(NaN),
    output: [NaN],
    ver: { input: values.map(() => null), shared: new Array(B).fill(null), output: [null] },
    regs: range(B).map(() => ({ a: NaN, b: NaN, m: NaN })),
  };
  const what = (phase: number) => (d.two ? (phase === 0 ? 'max' : 'sum') : 'sum');
  const describe = (ver: Version, _arr: Arr, index: number) => {
    if (d.two && ver.phase === 0 && ver.level === ss.length && index === 0) return tr('the max', 'giá trị max');
    if (ver.level === 0) return tr(`thread ${index}'s first value of the ${what(ver.phase)}`, `giá trị đầu tiên của luồng ${index} cho ${what(ver.phase) === 'max' ? 'max' : 'tổng'}`);
    return tr(`thread ${index}'s ${what(ver.phase)} partial from iteration ${ver.level} (stride ${ss[ver.level - 1]})`, `${what(ver.phase) === 'max' ? 'max' : 'tổng'} riêng phần của luồng ${index} từ vòng lặp ${ver.level} (bước nhảy ${ss[ver.level - 1]})`);
  };
  const { steps, issues } = execute(program, machine, schedule, seed, d.kernel, describe);
  const output = machine.output[0];
  const expected = expectedOf(v, values);
  return {
    kernel: d.kernel, file: HUNT_FILE, threads: B, schedule, program,
    initial: { input: [...values], shared: new Array<number>(B).fill(NaN) }, steps, issues, output, expected,
    outcome: outcomeOf(issues, output, expected),
  };
}

export type ThreadStatus = 'not started' | 'running' | 'waiting' | 'finished';
export interface RunState {
  input: number[];
  shared: number[];
  output: number;
  /** Per thread: index of its last step in run.steps, or -1. */
  last: number[];
  status: ThreadStatus[];
  /** Per thread: operations done. */
  done: number[];
}

/** The state after the first n steps, for a stepper. */
export function stateAt(run: ThreadRun, n: number): RunState {
  const input = [...run.initial.input];
  const shared = [...run.initial.shared];
  let output = NaN;
  const T = run.threads;
  const last = new Array<number>(T).fill(-1);
  const done = new Array<number>(T).fill(0);
  const waiting = new Array<boolean>(T).fill(false);
  for (let i = 0; i < Math.min(n, run.steps.length); i++) {
    const s = run.steps[i];
    if (s.thread === -1) {
      waiting.fill(false);
      continue;
    }
    if (s.thread < 0) continue;
    last[s.thread] = i;
    done[s.thread]++;
    if (s.op?.kind === 'barrier') waiting[s.thread] = true;
    if (s.write) {
      if (s.write.arr === 'input') input[s.write.index] = s.write.value;
      else if (s.write.arr === 'shared') shared[s.write.index] = s.write.value;
      else output = s.write.value;
    }
  }
  const status: ThreadStatus[] = range(T).map((t) =>
    done[t] === 0 ? 'not started' : done[t] >= run.program[t].length ? 'finished' : waiting[t] ? 'waiting' : 'running', // i18n-ignore: identifier also shown to the reader
  );
  return { input, shared, output, last, status, done };
}

export interface HistoryRow {
  /** 0 for the input as it starts, else the iteration. */
  n: number;
  stride: number;
  values: number[];
  /** Elements written in this iteration so far. */
  written: boolean[];
  /** Elements written from a read that was not final. */
  bad: boolean[];
  /** The additions of this iteration so far: the two elements read and the one written. */
  adds: { thread: number; own: number; partner: number }[];
}

/** For an in-place run: the input row, then one row per iteration with what that iteration wrote so far (Fig. 10.6's layout). */
export function historyAt(run: ThreadRun, n: number): HistoryRow[] {
  const kind: InPlaceKind = run.kernel === INPLACE_KERNEL.simple ? 'simple' : 'convergent';
  const ss = strides(kind, run.threads);
  const rows: HistoryRow[] = [{
    n: 0, stride: 0, values: [...run.initial.input],
    written: run.initial.input.map(() => false), bad: run.initial.input.map(() => false), adds: [],
  }];
  const byIter = ss.map(() => [] as TStep[]);
  for (let i = 0; i < Math.min(n, run.steps.length); i++) {
    const s = run.steps[i];
    if (s.write?.arr === 'input' && s.op && s.op.iter > 0) byIter[s.op.iter - 1].push(s);
  }
  ss.forEach((stride, k) => {
    const prev = rows[k];
    const row: HistoryRow = {
      n: k + 1, stride, values: [...prev.values],
      written: prev.values.map(() => false), bad: prev.values.map(() => false), adds: [],
    };
    for (const s of byIter[k]) {
      row.values[s.write!.index] = s.write!.value;
      row.written[s.write!.index] = true;
      if (s.issue) row.bad[s.write!.index] = true;
      row.adds.push({ thread: s.thread, own: s.reads[0].index, partner: s.reads[1].index });
    }
    rows.push(row);
  });
  return rows;
}

/** A run's first problem, as a step number (1-based, for "go to the first failure"), or null. */
export const firstIssueStep = (run: ThreadRun): number | null => (run.issues.length ? run.issues[0].step + 1 : null);
