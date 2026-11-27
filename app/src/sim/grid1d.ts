// Thread-level model of a one-dimensional kernel, for chapter 2, section 2.5: the
// line-by-line stepper, the thread index explorer and the "Where the loop went"
// scene all draw from it, and the section's questions take their numbers from it.
//
// The kernel is the book's vecAddKernel (Fig. 2.10, p. 35):
//
//     int i = blockIdx.x * blockDim.x + threadIdx.x;   // line 1
//     if (i < n) {                                     // line 2
//         C[i] = A[i] + B[i];                          // line 3
//     }
//
// Sources for the rules:
//  - the book, section 2.5, pp. 33 to 37: the global index (Fig. 2.9), one private
//    i per thread, the guard, and the worked case n = 100 with blocks of 32;
//  - the book, section 2.6, pp. 38 to 39: the block count as a ceiling division and
//    the worked case n = 1000 with blocks of 256;
//  - the book, exercises 2 and 3, p. 41: two elements per thread, adjacent or in two
//    sections per block. The book prints no answers; the two mappings here are our
//    derivation (notes/ch02.md, "Exercises").
//
// What this leaves out:
//  - Warps, scheduling order and timing. stepKernel() advances every thread by one
//    source line together. That is a drawing convention, not how a GPU schedules
//    threads, and nothing in the result depends on it: no thread reads what another
//    thread writes.
//  - Real memory. An index at or past n is reported as out of range. On a GPU such
//    an access is not documented to be reported at all (notes/today/ch02-facts.md,
//    A6); what value a stray load returns is unknown, so the model gives it none.
//  - Types. Indices are plain integers; the unsigned built-ins and the int they are
//    stored in are not modelled.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

/** The book's upper limit on threads per block "on current systems" (p. 33). The explorer's slider stops here. */
import { tr } from '../lib/locale.ts';

export const BOOK_MAX_BLOCK = 1024;
/** The book's advice: a block size that is a multiple of 32 "for hardware efficiency reasons" (p. 34). */
export const BOOK_BLOCK_MULTIPLE = 32;

export type MappingKind = 'one' | 'two-adjacent' | 'two-sections';
export const MAPPING_KINDS: readonly MappingKind[] = ['one', 'two-adjacent', 'two-sections'];

export interface Owner {
  block: number;
  thread: number;
  /** 0 for the thread's first element, 1 for its second. */
  slot: number;
}

export interface Mapping {
  kind: MappingKind;
  label: string;
  /** The book exercise this mapping answers (p. 41), or null for the chapter's own kernel. */
  exercise: number | null;
  /** Elements one thread is responsible for. */
  perThread: 1 | 2;
  /** The index of a thread's first element, as C source. */
  firstExpr: string;
  /** The index of its second element in terms of i, or null. */
  secondExpr: string | null;
  /** The bounds test each access needs, as C source, in the order of the accesses. */
  guards: string[];
  first: (block: number, thread: number, blockDim: number) => number;
  /** Every element the thread is responsible for, in range or not. */
  elements: (block: number, thread: number, blockDim: number) => number[];
  /** The thread responsible for element e. */
  owner: (e: number, blockDim: number) => Owner;
  /** Fewest blocks whose threads cover n elements. */
  blocksFor: (n: number, blockDim: number) => number;
}

const ceilDiv = (a: number, b: number) => Math.floor((a + b - 1) / b);

const MAPPINGS: Record<MappingKind, Mapping> = {
  one: {
    kind: 'one',
    get label() {
      return tr('one element per thread', 'mỗi luồng một phần tử');
    },
    exercise: 1,
    perThread: 1,
    firstExpr: 'blockIdx.x * blockDim.x + threadIdx.x', // i18n-ignore: C index expression
    secondExpr: null,
    guards: ['i < n'],
    first: (b, t, B) => b * B + t,
    elements: (b, t, B) => [b * B + t],
    owner: (e, B) => ({ block: Math.floor(e / B), thread: e % B, slot: 0 }),
    blocksFor: (n, B) => ceilDiv(n, B),
  },
  'two-adjacent': {
    kind: 'two-adjacent',
    get label() {
      return tr('two adjacent elements per thread', 'mỗi luồng hai phần tử liền kề');
    },
    exercise: 2,
    perThread: 2,
    firstExpr: '(blockIdx.x * blockDim.x + threadIdx.x) * 2', // i18n-ignore: C index expression
    secondExpr: 'i + 1',
    guards: ['i < n', 'i + 1 < n'],
    first: (b, t, B) => (b * B + t) * 2,
    elements: (b, t, B) => [(b * B + t) * 2, (b * B + t) * 2 + 1],
    owner: (e, B) => {
      const g = Math.floor(e / 2);
      return { block: Math.floor(g / B), thread: g % B, slot: e % 2 };
    },
    blocksFor: (n, B) => ceilDiv(ceilDiv(n, 2), B),
  },
  'two-sections': {
    kind: 'two-sections',
    get label() {
      return tr('two sections per block', 'mỗi khối hai đoạn');
    },
    exercise: 3,
    perThread: 2,
    firstExpr: 'blockIdx.x * blockDim.x * 2 + threadIdx.x', // i18n-ignore: C index expression
    secondExpr: 'i + blockDim.x', // i18n-ignore: C index expression
    guards: ['i < n', 'i + blockDim.x < n'],
    first: (b, t, B) => b * B * 2 + t,
    elements: (b, t, B) => [b * B * 2 + t, b * B * 2 + t + B],
    owner: (e, B) => {
      const r = e % (2 * B);
      return { block: Math.floor(e / (2 * B)), thread: r % B, slot: Math.floor(r / B) };
    },
    blocksFor: (n, B) => ceilDiv(n, 2 * B),
  },
};

/** The element(s) a thread owns under one of the three mappings, with the guard each needs. */
export const mapping = (kind: MappingKind): Mapping => MAPPINGS[kind];

export interface Config {
  /** Number of elements. */
  n: number;
  /** Threads per block, blockDim.x. */
  blockDim: number;
  /** Blocks launched. Left out: the fewest that cover n. */
  blocks?: number;
  /** Whether the kernel tests its indices against n. Left out: it does. */
  guard?: boolean;
  /** Left out: one element per thread. */
  mapping?: MappingKind;
}

const isCount = (v: number) => Number.isInteger(v) && v >= 1;
/** n and blockDim are whole numbers of at least 1 (and blocks, when given). */
export const validConfig = (c: Config): boolean =>
  isCount(c.n) && isCount(c.blockDim) && (c.blocks === undefined || isCount(c.blocks));

/** Fewest blocks of blockDim threads that cover n elements: the ceiling division of section 2.6. */
export const blocksNeeded = (n: number, blockDim: number, kind: MappingKind = 'one'): number =>
  mapping(kind).blocksFor(n, blockDim);

export interface Thread {
  /** Position in the whole grid, counting block after block: block * blockDim + thread. */
  g: number;
  /** blockIdx.x */
  block: number;
  /** threadIdx.x */
  thread: number;
  /** The index the kernel computes: the thread's first element. */
  i: number;
  /** Every element the mapping gives this thread, in range or not. */
  elements: number[];
}

/** Every thread of the launch, block after block. */
export function threads(n: number, blockDim: number, blocks?: number, kind: MappingKind = 'one'): Thread[] {
  const m = mapping(kind);
  const count = blocks ?? m.blocksFor(n, blockDim);
  const out: Thread[] = [];
  for (let b = 0; b < count; b++)
    for (let t = 0; t < blockDim; t++)
      out.push({ g: b * blockDim + t, block: b, thread: t, i: m.first(b, t, blockDim), elements: m.elements(b, t, blockDim) });
  return out;
}

/** The index arithmetic for one thread, written out with its numbers: "3 × 256 + 5 = 773". */
export function working(block: number, thread: number, blockDim: number, kind: MappingKind = 'one'): string {
  const i = mapping(kind).first(block, thread, blockDim);
  if (kind === 'one') return `${block} × ${blockDim} + ${thread} = ${i}`;
  if (kind === 'two-adjacent') return `(${block} × ${blockDim} + ${thread}) × 2 = ${i}`;
  return `${block} × ${blockDim} × 2 + ${thread} = ${i}`;
}

export interface Counts {
  blocks: number;
  /** blocks × blockDim */
  launched: number;
  /** Threads with at least one element in range. */
  working: number;
  /** Threads with no element in range: every access of theirs fails the guard. */
  idle: number;
  /** Threads with one element in range and one out of it (two-element mappings only). */
  partial: number;
}

/**
 * Thread counts in closed form, so that a launch of any size can be counted
 * without listing its threads (the book's exercise 9 has 200,064 of them).
 */
export function counts(c: Config): Counts {
  const kind = c.mapping ?? 'one';
  const B = c.blockDim;
  const blocks = c.blocks ?? blocksNeeded(c.n, B, kind);
  const launched = blocks * B;
  let workingThreads: number;
  let full: number; // threads with every element in range
  if (kind === 'one') {
    workingThreads = Math.min(launched, c.n);
    full = workingThreads;
  } else if (kind === 'two-adjacent') {
    workingThreads = Math.min(launched, Math.ceil(c.n / 2));
    full = Math.min(launched, Math.floor(c.n / 2));
  } else {
    // Whole blocks cover 2B elements each; the block that n falls in is partly used.
    const whole = Math.min(blocks, Math.floor(c.n / (2 * B)));
    const rest = whole < blocks ? c.n - whole * 2 * B : 0; // elements in the partly used block
    workingThreads = whole * B + Math.min(B, rest);
    full = whole * B + Math.max(0, rest - B);
  }
  return { blocks, launched, working: workingThreads, idle: launched - workingThreads, partial: workingThreads - full };
}

export interface Coverage extends Counts {
  /** Per element 0 to n - 1: the grid positions (Thread.g) of the threads that write it. */
  writers: number[][];
  /** Threads with no element in range. */
  idleThreads: Thread[];
  /** Threads with one element in range and one out of it. */
  partialThreads: Thread[];
  /** Indices at or past n that some thread's mapping reaches: what the guard stops. Sorted, no repeats. */
  beyond: number[];
  /** Indices at or past n that are really touched: none with the guard on, all of `beyond` with it off. */
  outOfRange: number[];
  /** Elements below n that no thread writes (a launch with too few blocks). */
  unwritten: number[];
}

/** Who writes each element, which threads have nothing to do, and what an unguarded kernel would touch. */
export function coverage(c: Config): Coverage {
  const kind = c.mapping ?? 'one';
  const all = threads(c.n, c.blockDim, c.blocks, kind);
  const writers: number[][] = Array.from({ length: c.n }, () => []);
  const idleThreads: Thread[] = [];
  const partialThreads: Thread[] = [];
  const beyond = new Set<number>();
  for (const th of all) {
    let inside = 0;
    for (const e of th.elements) {
      if (e < c.n) {
        writers[e].push(th.g);
        inside++;
      } else beyond.add(e);
    }
    if (inside === 0) idleThreads.push(th);
    else if (inside < th.elements.length) partialThreads.push(th);
  }
  const sorted = [...beyond].sort((a, b) => a - b);
  const blocks = c.blocks ?? blocksNeeded(c.n, c.blockDim, kind);
  return {
    blocks,
    launched: all.length,
    working: all.length - idleThreads.length,
    idle: idleThreads.length,
    partial: partialThreads.length,
    writers,
    idleThreads,
    partialThreads,
    beyond: sorted,
    outOfRange: c.guard === false ? sorted : [],
    unwritten: writers.flatMap((w, e) => (w.length ? [] : [e])),
  };
}

// ---------------------------------------------------------------- data and the sequential loop
export interface Data {
  A: number[];
  B: number[];
}
/** Small whole numbers, so that every sum can be checked by eye. The same for every caller. */
export function makeData(n: number): Data {
  return {
    A: Array.from({ length: n }, (_, i) => ((i * 3 + 1) % 7) + 1),
    B: Array.from({ length: n }, (_, i) => ((i * 5 + 2) % 9) + 1),
  };
}

export interface Trip {
  /** The loop variable on this trip. */
  i: number;
  a: number;
  b: number;
  /** What the trip stores in C[i]. */
  c: number;
}
/** The sequential loop of Fig. 2.4, one entry per trip: for (i = 0; i < n; ++i) C[i] = A[i] + B[i]. */
export function loopView(n: number, data: Data = makeData(n)): Trip[] {
  return Array.from({ length: n }, (_, i) => ({ i, a: data.A[i], b: data.B[i], c: data.A[i] + data.B[i] }));
}

// ---------------------------------------------------------------- the kernel, line by line
/** 0: nothing has run. 1: i computed. 2: guard tested. 3: loads and store done. */
export type Line = 0 | 1 | 2 | 3;
export interface Access {
  array: 'A' | 'B' | 'C';
  index: number;
  /** index < n */
  inRange: boolean;
  /** The value read or written, or null for a load from outside the array. */
  value: number | null;
}
export interface ThreadState {
  g: number;
  block: number;
  thread: number;
  /** The thread's private i: null until line 1 has run. */
  i: number | null;
  /** Result of `i < n`: null until line 2 has run, and null throughout when the kernel has no guard. */
  pass: boolean | null;
  /** Whether the thread runs line 3: it passed the guard, or there is no guard. */
  runsBody: boolean;
  /** The two loads of line 3, once it has run. */
  loads: Access[];
  /** The store of line 3, once it has run. */
  store: Access | null;
}
export interface KernelState {
  line: Line;
  threads: ThreadState[];
  /** C[0 .. n-1]: null where nothing has been stored yet. */
  C: (number | null)[];
  /** Indices at or past n that line 3 touched. Empty unless the guard is off. Sorted. */
  outOfRange: number[];
}

/**
 * The state of every thread after `line` source lines of vecAddKernel, with all
 * threads advanced together. One element per thread only: this is the chapter's kernel.
 */
export function stepKernel(c: Config, line: Line, data: Data = makeData(c.n)): KernelState {
  const guard = c.guard !== false;
  const C: (number | null)[] = Array.from({ length: c.n }, () => null);
  const outOfRange: number[] = [];
  const states = threads(c.n, c.blockDim, c.blocks, 'one').map((th): ThreadState => {
    const st: ThreadState = { g: th.g, block: th.block, thread: th.thread, i: null, pass: null, runsBody: false, loads: [], store: null };
    if (line >= 1) st.i = th.i;
    const inRange = th.i < c.n;
    if (line >= 2 && guard) st.pass = inRange;
    st.runsBody = guard ? inRange : true;
    if (line >= 3 && st.runsBody) {
      const a = inRange ? data.A[th.i] : null;
      const b = inRange ? data.B[th.i] : null;
      st.loads = [
        { array: 'A', index: th.i, inRange, value: a },
        { array: 'B', index: th.i, inRange, value: b },
      ];
      st.store = { array: 'C', index: th.i, inRange, value: a !== null && b !== null ? a + b : null };
      if (inRange) C[th.i] = st.store.value;
      else outOfRange.push(th.i);
    }
    return st;
  });
  return { line, threads: states, C, outOfRange: outOfRange.sort((x, y) => x - y) };
}

/** The whole kernel run to its end: what it leaves in C[0 .. n-1]. */
export const runKernel = (c: Config, data: Data = makeData(c.n)): (number | null)[] => stepKernel(c, 3, data).C;

// ---------------------------------------------------------------- block sizes
/** The nearest multiple of `multiple` to v, kept between `multiple` and the largest multiple not above max. */
export function snapBlock(v: number, multiple: number = BOOK_BLOCK_MULTIPLE, max: number = BOOK_MAX_BLOCK): number {
  const top = Math.max(multiple, Math.floor(max / multiple) * multiple);
  return Math.min(top, Math.max(multiple, Math.round(v / multiple) * multiple));
}

/** "0 to 255" for a run of indices, "7" for a single one, "none" for an empty run. */
export function span(indices: number[]): string {
  if (!indices.length) return 'none';
  const lo = indices[0];
  const hi = indices[indices.length - 1];
  return lo === hi ? String(lo) : tr(`${lo} to ${hi}`, `${lo} đến ${hi}`);
}
