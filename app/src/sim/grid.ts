// Launch geometry for chapter 3, section 3.1: what a pair of dim3 values creates,
// whether a device would accept it, what the book calls each block and thread, and
// how many blocks a 1D launch needs for n elements. It runs the grid explorer, the
// find-the-thread drill, the four-formula sizing explorer, the limits tables and
// the numbers in the questions of the "Before you start" page and of section 3.1.
//
// Sources for the rules (the lessons cite them where they are used):
//  - the book, section 3.1, pp. 45 to 48: a grid is a 3D array of blocks, a block a
//    3D array of threads; at most 1024 threads in a block; gridDim.x from 1 to
//    2^31 - 1, gridDim.y and gridDim.z from 1 to 65535; the labels Block (y, x) and
//    Thread (z, y, x);
//  - notes/today/ch05-facts.md A5 and notes/today/ch03-facts.md A3: NVIDIA's
//    documented per-dimension caps on a block, 1024 by 1024 by 64, which the book
//    does not state;
//  - notes/today/ch03-facts.md A2: threads of a block are numbered x fastest, then
//    y, then z;
//  - notes/today/ch03-facts.md A4 and B4: the error a rejected launch sets.
//
// What this leaves out:
//  - Limits are tables. Nothing here is a device: there are no warps, no
//    scheduling, no memory, and a "legal" verdict says only that the numbers pass
//    the chosen table.
//  - Limits that depend on more than the two dim3 values are not modelled: shared
//    memory per block; the registers a kernel uses, which can make a CUDA launch
//    under 1024 threads fail with cudaErrorLaunchOutOfResources (ch08-facts A4), and
//    on HIP the block-size bound a kernel is compiled for (ch08-facts B4); HIP's rule
//    that gridDim times blockDim fits in 32 bits per dimension; Metal's per-pipeline
//    thread limit.
//  - A dim3 field is treated as an unsigned 32-bit integer. What a compiler does
//    with a negative or fractional argument is not modelled.
//  - The ceil form is evaluated in double precision, as C++ would, and then taken
//    as an integer. The conversion from double to unsigned that the C++ code
//    relies on is not modelled beyond that.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

export interface Dim3 {
  x: number;
  y: number;
  z: number;
}
export type Axis = 'x' | 'y' | 'z';
export const AXES: readonly Axis[] = ['x', 'y', 'z'];

/** Like the CUDA constructor: components you leave out are 1. */
export const dim3 = (x = 1, y = 1, z = 1): Dim3 => ({ x, y, z });

/** A value a dim3 field can hold: an unsigned 32-bit integer. */
export const isField = (v: number): boolean => Number.isInteger(v) && v >= 0 && v <= 0xffffffff;
export const isDim3 = (d: Dim3): boolean => AXES.every((a) => isField(d[a]));

/** x * y * z. A grid at its limits holds more blocks than a double counts exactly, hence bigint. */
export const volume = (d: Dim3): bigint => BigInt(d.x) * BigInt(d.y) * BigInt(d.z);
export const threadsPerBlock = (block: Dim3): bigint => volume(block);
export const blockCount = (grid: Dim3): bigint => volume(grid);
export const totalThreads = (grid: Dim3, block: Dim3): bigint => volume(grid) * volume(block);

/** How many of the three dimensions are larger than 1 (at least 1: a single thread is still 1D). */
export const dimensionality = (d: Dim3): number => Math.max(1, AXES.filter((a) => d[a] > 1).length);

// ---------------------------------------------------------------- limits
export interface Limits {
  /** Most threads one block may hold (the product of its three dimensions), or null when the source gives none. */
  maxThreadsPerBlock: number | null;
  /** Cap on each dimension of a block, or null when the source gives none. */
  maxBlockDim: Dim3 | null;
  /** Cap on each dimension of a grid, in blocks, or null when the source gives none. */
  maxGridDim: Dim3 | null;
}

const INT31 = 2 ** 31 - 1;

/** The book's rule (p. 47): a product limit on the block, ranges for the grid, nothing per block dimension. */
export const BOOK_RULE: Limits = {
  maxThreadsPerBlock: 1024,
  maxBlockDim: null,
  maxGridDim: { x: INT31, y: 65535, z: 65535 },
};

/** NVIDIA's documented limits, the same for every listed compute capability (ch05-facts A5). */
export const CUDA_DOCUMENTED: Limits = {
  maxThreadsPerBlock: 1024,
  maxBlockDim: { x: 1024, y: 1024, z: 64 },
  maxGridDim: { x: INT31, y: 65535, z: 65535 },
};

const positiveInt = (v: unknown): number | null => {
  if (v == null || typeof v === 'boolean' || v === '') return null;
  const n = Number(v);
  return Number.isInteger(n) && n > 0 ? n : null;
};
const triple = (d: Record<string, unknown>, keys: [string, string, string]): Dim3 | null => {
  const [x, y, z] = keys.map((k) => positiveInt(d[k]));
  return x != null && y != null && z != null ? { x, y, z } : null;
};

/**
 * Limits from a device record in reports/runs/<backend>.json. CUDA and HIP devices
 * report maxThreadsPerBlock, maxThreadsDim.* and maxGridSize.*; a Metal device
 * reports only maxThreadsPerThreadgroup.* (per dimension; the total is a property
 * of each compiled kernel and is not in the device record). Null when the record
 * holds none of these.
 */
export function limitsFromReport(device: Record<string, unknown>): Limits | null {
  const limits: Limits = {
    maxThreadsPerBlock: positiveInt(device['maxThreadsPerBlock']),
    maxBlockDim:
      triple(device, ['maxThreadsDim.x', 'maxThreadsDim.y', 'maxThreadsDim.z']) ??
      triple(device, ['maxThreadsPerThreadgroup.width', 'maxThreadsPerThreadgroup.height', 'maxThreadsPerThreadgroup.depth']),
    maxGridDim: triple(device, ['maxGridSize.x', 'maxGridSize.y', 'maxGridSize.z']),
  };
  return limits.maxThreadsPerBlock == null && !limits.maxBlockDim && !limits.maxGridDim ? null : limits;
}

export type Clause = 'zero' | 'block-threads' | 'block-dim' | 'grid-dim';
export interface Violation {
  clause: Clause;
  /** The quantity that breaks the clause: "blockDim.z", "gridDim.y" or "threads per block". */
  field: string;
  value: bigint;
  /** The largest value allowed, or 1 for the 'zero' clause (the smallest allowed). */
  limit: number;
}
export type Unchecked = 'block-threads' | 'block-dim' | 'grid-dim';
export interface Verdict {
  /** No clause of the limit set is broken. */
  legal: boolean;
  violations: Violation[];
  /** Clauses the limit set has no number for. A legal verdict does not cover them. */
  unchecked: Unchecked[];
}

/** Every clause of `limits` that the launch breaks. */
export function checkLaunch(grid: Dim3, block: Dim3, limits: Limits): Verdict {
  const violations: Violation[] = [];
  const unchecked: Unchecked[] = [];
  for (const [name, d] of [['gridDim', grid], ['blockDim', block]] as const)
    for (const a of AXES)
      if (d[a] < 1) violations.push({ clause: 'zero', field: `${name}.${a}`, value: BigInt(d[a]), limit: 1 });

  if (limits.maxThreadsPerBlock == null) unchecked.push('block-threads');
  else if (volume(block) > BigInt(limits.maxThreadsPerBlock))
    violations.push({ clause: 'block-threads', field: tr('threads per block', 'số luồng mỗi khối'), value: volume(block), limit: limits.maxThreadsPerBlock });

  if (!limits.maxBlockDim) unchecked.push('block-dim');
  else
    for (const a of AXES)
      if (block[a] > limits.maxBlockDim[a])
        violations.push({ clause: 'block-dim', field: `blockDim.${a}`, value: BigInt(block[a]), limit: limits.maxBlockDim[a] }); // i18n-ignore: code identifier

  if (!limits.maxGridDim) unchecked.push('grid-dim');
  else
    for (const a of AXES)
      if (grid[a] > limits.maxGridDim[a])
        violations.push({ clause: 'grid-dim', field: `gridDim.${a}`, value: BigInt(grid[a]), limit: limits.maxGridDim[a] }); // i18n-ignore: code identifier

  return { legal: violations.length === 0, violations, unchecked };
}

const big = (v: bigint | number) => v.toLocaleString('en-US');

/** One sentence per broken clause, for the explorer's verdict. */
export function describeViolation(v: Violation): string {
  switch (v.clause) {
    case 'zero':
      return tr(`${v.field} is ${big(v.value)}. Every dimension must be at least 1.`, `${v.field} là ${big(v.value)}. Mọi chiều phải ít nhất là 1.`);
    case 'block-threads':
      return tr(`A block would hold ${big(v.value)} threads. The limit is ${big(v.limit)}.`, `Một khối sẽ chứa ${big(v.value)} luồng. Giới hạn là ${big(v.limit)}.`);
    case 'block-dim':
      return tr(`${v.field} is ${big(v.value)}. The cap for that dimension of a block is ${big(v.limit)}.`, `${v.field} là ${big(v.value)}. Giới hạn cho chiều đó của một khối là ${big(v.limit)}.`);
    case 'grid-dim':
      return tr(`${v.field} is ${big(v.value)}. The cap for that dimension of a grid is ${big(v.limit)}.`, `${v.field} là ${big(v.value)}. Giới hạn cho chiều đó của một lưới là ${big(v.limit)}.`);
  }
}

/** What a rejected launch leaves in the runtime's error state (ch03-facts A4 and B4). Both enumerators have the value 9. */
export const LAUNCH_ERROR = {
  cuda: { name: 'cudaErrorInvalidConfiguration', value: 9, query: 'cudaGetLastError()' },
  hip: { name: 'hipErrorInvalidConfiguration', value: 9, query: 'hipGetLastError()' },
} as const;

// ---------------------------------------------------------------- coordinates and labels
/** Is `i` a coordinate inside `dims` (0 <= i < dims on every axis)? */
export const inside = (i: Dim3, dims: Dim3): boolean => AXES.every((a) => Number.isInteger(i[a]) && i[a] >= 0 && i[a] < dims[a]);
export const sameIndex = (a: Dim3, b: Dim3): boolean => a.x === b.x && a.y === b.y && a.z === b.z;

/** The book's label for a thread (p. 48): highest dimension first, (threadIdx.z, threadIdx.y, threadIdx.x). */
export const threadLabel = (t: Dim3): [number, number, number] => [t.z, t.y, t.x];
/** The book's label for a block (p. 48): (blockIdx.y, blockIdx.x). */
export const blockLabel = (b: Dim3): [number, number] => [b.y, b.x];
export const threadFromLabel = ([z, y, x]: readonly [number, number, number]): Dim3 => ({ x, y, z });
export const blockFromLabel = ([y, x]: readonly [number, number]): Dim3 => ({ x, y, z: 0 });

export const formatThreadLabel = (t: Dim3): string => `Thread (${threadLabel(t).join(',')})`; // i18n-ignore: identifier also shown to the reader
/**
 * "Block (y,x)" as the book writes it. The book only labels 2D grids; for a grid
 * with more than one layer in z the same highest-first rule gives (z,y,x), which is
 * our extension.
 */
export const formatBlockLabel = (b: Dim3, grid: Dim3 = dim3()): string =>
  grid.z > 1 ? `Block (${b.z},${b.y},${b.x})` : `Block (${blockLabel(b).join(',')})`; // i18n-ignore: identifier also shown to the reader
/** The same index in the order of the dim3 constructor. */
export const formatDim3 = (d: Dim3): string => `(${d.x}, ${d.y}, ${d.z})`;

/**
 * A thread's number inside its block: x moves fastest, then y, then z (NVIDIA's
 * documented linearisation, ch03-facts A2). threadIdx.y has a stride of blockDim.x
 * and threadIdx.z a stride of blockDim.x * blockDim.y.
 */
export const linearThread = (t: Dim3, block: Dim3): number => t.x + t.y * block.x + t.z * block.x * block.y;
/** The inverse of linearThread. */
export const threadAt = (n: number, block: Dim3): Dim3 => ({
  x: n % block.x,
  y: Math.floor(n / block.x) % block.y,
  z: Math.floor(n / (block.x * block.y)),
});
/** The arithmetic of linearThread written out, e.g. "2 + 0×4 + 1×8 = 10". */
export const linearThreadWorking = (t: Dim3, block: Dim3): string =>
  `${t.x} + ${t.y}×${block.x} + ${t.z}×${block.x * block.y} = ${linearThread(t, block)}`;

/** The global coordinate of a thread on each axis: blockIdx * blockDim + threadIdx (chapter 2, p. 34, per axis). */
export const globalCoord = (blockIdx: Dim3, blockDim: Dim3, threadIdx: Dim3): Dim3 => ({
  x: blockIdx.x * blockDim.x + threadIdx.x,
  y: blockIdx.y * blockDim.y + threadIdx.y,
  z: blockIdx.z * blockDim.z + threadIdx.z,
});

// ---------------------------------------------------------------- the find-the-thread drill
export interface Target {
  block: Dim3;
  thread: Dim3;
}
export const FIG31 = { grid: dim3(2, 2, 1), block: dim3(4, 2, 2) } as const;
/** The book's own example on p. 48: Thread (1,0,2) of Block (1,0). */
export const BOOK_TARGET: Target = { block: blockFromLabel([1, 0]), thread: threadFromLabel([1, 0, 2]) };

/** A small deterministic generator (mulberry32), so a seed always gives the same drill. */
function rng(seed: number): () => number {
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
 * Would a reader who reads the label in constructor order (x first) land on a
 * different cell? True when the thread's x and z differ and the block's x and y
 * differ, so both halves of the label test the reversal.
 */
export const discriminates = (t: Target): boolean => t.thread.x !== t.thread.z && t.block.x !== t.block.y;

/**
 * `count` targets inside the launch. The first is the book's example when the
 * launch contains it. The rest are drawn from `seed`, each different from the
 * others and, where the launch allows, each one a label that reads differently
 * when its order is reversed.
 */
export function drillTargets(grid: Dim3, block: Dim3, count: number, seed: number): Target[] {
  const out: Target[] = [];
  const has = (t: Target) => out.some((o) => sameIndex(o.block, t.block) && sameIndex(o.thread, t.thread));
  if (inside(BOOK_TARGET.block, grid) && inside(BOOK_TARGET.thread, block)) out.push(BOOK_TARGET);
  const next = rng(seed);
  const pick = (n: number) => Math.floor(next() * n);
  const cells = Number(totalThreads(grid, block));
  for (let tries = 0; out.length < Math.min(count, cells) && tries < 4000; tries++) {
    const t: Target = {
      block: { x: pick(grid.x), y: pick(grid.y), z: pick(grid.z) },
      thread: { x: pick(block.x), y: pick(block.y), z: pick(block.z) },
    };
    // After many failed draws the launch has too few discriminating cells: take any new one.
    if (!has(t) && (discriminates(t) || tries > 2000)) out.push(t);
  }
  return out;
}

export type Reading = 'right' | 'reversed' | 'other';
/**
 * How a click in the drill relates to the target, separately for the block and for
 * the thread. 'reversed' means the reader took the book's label in constructor
 * order: (x, y) for a block, (x, y, z) for a thread.
 */
export function misread(target: Target, clicked: Target): { block: Reading; thread: Reading } {
  const [by, bx] = blockLabel(target.block);
  const [tz, ty, tx] = threadLabel(target.thread);
  const one = (want: Dim3, got: Dim3, swapped: Dim3): Reading =>
    sameIndex(want, got) ? 'right' : sameIndex(swapped, got) ? 'reversed' : 'other';
  return {
    block: one(target.block, clicked.block, { x: by, y: bx, z: target.block.z }),
    thread: one(target.thread, clicked.thread, { x: tz, y: ty, z: tx }),
  };
}

// ---------------------------------------------------------------- sizing a 1D grid
export type FormulaKey = 'truncate' | 'plus-one' | 'round-up' | 'ceil';
export interface Formula {
  key: FormulaKey;
  /** The C++ expression, with n the element count and b the block size, both int. */
  code: string;
}
export const FORMULAS: readonly Formula[] = [
  { key: 'truncate', code: 'n / b' },
  { key: 'plus-one', code: 'n / b + 1' },
  { key: 'round-up', code: '(n + b - 1) / b' },
  { key: 'ceil', code: 'ceil(n / (double)b)' },
];

/** Blocks each formula launches for n elements and blocks of b threads (n >= 0, b >= 1, C++ int division). */
export function sizeGrid(n: number, b: number, formula: FormulaKey): number {
  switch (formula) {
    case 'truncate':
      return Math.floor(n / b);
    case 'plus-one':
      return Math.floor(n / b) + 1;
    case 'round-up':
      return Math.floor((n + b - 1) / b);
    case 'ceil':
      return Math.ceil(n / b);
  }
}
/** The form the book's exercise 3 uses (p. 65): (n - 1) / b + 1. For n >= 1 it equals the round-up form. */
export const sizeGridMinusOne = (n: number, b: number): number => Math.floor((n - 1) / b) + 1;

export interface Coverage {
  blocks: number;
  /** blocks * b. */
  threads: number;
  /** Threads that pass the i < n test and do work. */
  working: number;
  /** Threads that fail the i < n test. */
  surplus: number;
  /** Elements no thread reaches. */
  uncovered: number;
  /** Blocks in which no thread passes the test. */
  idleBlocks: number;
}

/** What a 1D launch of `blocks` blocks of `b` threads does to n elements, one element per thread. */
export function coverage(n: number, b: number, blocks: number): Coverage {
  const threads = blocks * b;
  const working = Math.min(n, threads);
  // Block k works when its first thread, k * b, is below n: that is ceil(n / b) blocks.
  const needed = Math.ceil(n / b);
  return {
    blocks,
    threads,
    working,
    surplus: threads - working,
    uncovered: n - working,
    idleBlocks: Math.max(0, blocks - needed),
  };
}

/** How many threads of block k (0-based) of a 1D launch pass the i < n test. */
export const workingInBlock = (n: number, b: number, k: number): number => Math.max(0, Math.min(b, n - k * b));
