// Chapter 17, page 17.2 ("Formats decide the kernel"): a small-matrix model of the
// book's sparse storage formats and of the SpMV thread mapping each one implies. It
// runs the format explorer, the "Two threads, one y[0]" stepper, the narrated scene
// "Pad, turn, flatten" and the numbers in the page's questions.
//
// Sources (the lesson cites them where it uses them):
//  - notes/ch17.md: the running example, Fig. 17.1 (p. 402); COO, Figs. 17.3 to 17.6
//    (pp. 404 to 406); CSR, Figs. 17.7 to 17.9 and the iteration-by-iteration reads of
//    p. 410; ELL, Figs. 17.10 to 17.12 (pp. 410 to 414), index t*numRows + row and
//    row = i % numRows (p. 413); hybrid ELL-COO, Fig. 17.13 (pp. 414 to 415, the tail of
//    a long row goes to COO); JDS, Fig. 17.14 (p. 416, rows sorted longest first, as the
//    figure shows; notes "Unclear" item 3); CSC, Figs. 17.16 to 17.18 and the reads of
//    p. 420; exercises 1 and 2 (p. 422).
//  - notes/today/ch17-facts.md A6 and B3: NVIDIA's Sliced ELL and AMD's ELL store
//    column index -1 in a pad slot, and a loop over the full width tests it; E1: every
//    array of the figures rebuilt from the matrices alone.
//  - Fig. 17.13 prints no values (notes "Unclear"): the values 1 to 18 in row order are
//    ours. Fig. 17.14 prints letters: the values a = 1 to o = 15 are ours.
//
// What it models: the arrays each format stores; for each thread of an SpMV kernel and
// each iteration of its loop, the array index it reads (or that it is idle), the x
// element it reads, the y element its work goes to and whether that update is atomic;
// per-thread trip counts, from which divergence is counted with divergence.ts loopBatch;
// runs of consecutive addresses within a group of threads, counted with coalesce.ts runs.
// A float32 reference SpMV (Math.fround after every operation) checks that every
// format's trace computes y = A·x + y. For the atomics stepper: two threads doing
// load, add, store on one y element, with and without the atomic, and float32 sums of
// three terms in every order.
//
// What it leaves out, and the page says so: time, caches and the coalescing unit (runs
// of consecutive addresses stand in for sectors or lines); the warp width (threads are
// grouped in fours, or one group per row, as in the book's figures); the cost of an
// atomic; any order of atomics other than the one a schedule names.

import { loopBatch } from './divergence';
import { runs } from './coalesce';

// ---------------------------------------------------------------- matrices

export interface Entry {
  r: number;
  c: number;
  v: number;
  /** What a cell shows: its value, or a letter for Fig. 17.14. */
  label: string;
}
export interface Matrix {
  rows: number;
  cols: number;
  /** Non-zeros in row order, columns ascending within a row. */
  entries: Entry[];
  /** Labels are letters (Fig. 17.14) rather than the values. */
  letters?: boolean;
}

const byRow = (a: Entry, b: Entry) => a.r - b.r || a.c - b.c;
const byCol = (a: Entry, b: Entry) => a.c - b.c || a.r - b.r;

/** A matrix from (row, column, value[, label]) cells. Rejects duplicates and cells outside it. */
export function matrix(rows: number, cols: number, cells: readonly (readonly [number, number, number, string?])[], letters = false): Matrix {
  const seen = new Set<string>();
  const entries = cells.map(([r, c, v, label]) => {
    if (r < 0 || r >= rows || c < 0 || c >= cols) throw new Error(`cell (${r}, ${c}) is outside a ${rows} × ${cols} matrix`);
    const k = `${r},${c}`;
    if (seen.has(k)) throw new Error(`cell (${r}, ${c}) given twice`);
    seen.add(k);
    return { r, c, v, label: label ?? String(v) };
  });
  return { rows, cols, entries: entries.sort(byRow), letters };
}

/** A matrix from its dense rows; zeros are not stored. */
export function fromDense(rows: readonly (readonly number[])[]): Matrix {
  const cells: [number, number, number][] = [];
  rows.forEach((row, r) => row.forEach((v, c) => { if (v !== 0) cells.push([r, c, v]); }));
  return matrix(rows.length, rows[0]?.length ?? 0, cells);
}

/** The dense picture: value or 0. */
export const toDense = (m: Matrix): number[][] => {
  const d = Array.from({ length: m.rows }, () => Array<number>(m.cols).fill(0));
  for (const e of m.entries) d[e.r][e.c] = e.v;
  return d;
};

export const nnz = (m: Matrix): number => m.entries.length;
export const rowLists = (m: Matrix): Entry[][] => {
  const out: Entry[][] = Array.from({ length: m.rows }, () => []);
  for (const e of [...m.entries].sort(byRow)) out[e.r].push(e);
  return out;
};
export const colLists = (m: Matrix): Entry[][] => {
  const out: Entry[][] = Array.from({ length: m.cols }, () => []);
  for (const e of [...m.entries].sort(byCol)) out[e.c].push(e);
  return out;
};
export const rowLengths = (m: Matrix): number[] => rowLists(m).map((l) => l.length);
/** K: the length of the longest row. */
export const longestRow = (m: Matrix): number => Math.max(0, ...rowLengths(m));
export const entryAt = (m: Matrix, r: number, c: number): Entry | undefined => m.entries.find((e) => e.r === r && e.c === c);

const LETTERS = 'abcdefghijklmnopqrstuvwxyz';
/**
 * Click a cell: remove its non-zero, or add one. A new value is the smallest positive
 * integer the matrix does not use yet; in a lettered matrix the label is the first unused letter.
 */
export function toggle(m: Matrix, r: number, c: number): Matrix {
  if (entryAt(m, r, c)) return { ...m, entries: m.entries.filter((e) => e.r !== r || e.c !== c) };
  const used = new Set(m.entries.map((e) => e.v));
  let v = 1;
  while (used.has(v)) v++;
  let label = String(v);
  if (m.letters) {
    const taken = new Set(m.entries.map((e) => e.label));
    label = [...LETTERS].find((l) => !taken.has(l)) ?? String(v);
  }
  return { ...m, entries: [...m.entries, { r, c, v, label }].sort(byRow) };
}

/** Fig. 17.1 (p. 402): the chapter's running example. */
export const FIG17_1 = matrix(4, 4, [
  [0, 0, 1], [0, 1, 7],
  [1, 0, 5], [1, 2, 3], [1, 3, 9],
  [2, 1, 2], [2, 2, 8],
  [3, 3, 6],
]);

/** Exercise 1 (p. 422). */
export const EXERCISE1 = fromDense([
  [1, 0, 7, 0],
  [0, 0, 8, 0],
  [0, 4, 3, 0],
  [2, 0, 0, 1],
]);

/** Fig. 17.13 (p. 415): positions read from the figure; it prints no values, so 1 to 18 in row order are ours. */
export const FIG17_13_COLUMNS: readonly (readonly number[])[] = [[0], [0, 1, 3, 5, 7], [2, 4], [0, 3], [4], [1, 5], [0, 2, 4, 6], [7]];
export const FIG17_13 = (() => {
  let v = 0;
  return matrix(8, 8, FIG17_13_COLUMNS.flatMap((cs, r) => cs.map((c) => [r, c, ++v] as [number, number, number])));
})();

/** Fig. 17.14 (p. 416): letters a to o; their values 1 to 15 are ours. */
export const FIG17_14_COLUMNS: readonly (readonly number[])[] = [[0], [0, 1, 3, 4], [2, 4], [0, 3, 5], [1, 4], [0, 2, 5]];
export const FIG17_14 = (() => {
  let k = 0;
  return matrix(6, 6, FIG17_14_COLUMNS.flatMap((cs, r) => cs.map((c) => { k++; return [r, c, k, LETTERS[k - 1]] as [number, number, number, string]; })), true);
})();

export interface Preset {
  id: string;
  label: string;
  /** Where it comes from in the book. */
  source: string;
  m: Matrix;
}
export const PRESETS: readonly Preset[] = [
  { id: 'fig17_1', label: 'Fig. 17.1', source: 'p. 402, the running example', m: FIG17_1 }, // i18n-ignore: identifier also shown to the reader; S172 translates it by this English key
  { id: 'ex1', label: 'Exercise 1', source: 'p. 422', m: EXERCISE1 }, // i18n-ignore: identifier also shown to the reader; S172 translates it by this English key
  { id: 'fig17_13', label: 'Fig. 17.13', source: 'p. 415, two long rows; values ours', m: FIG17_13 }, // i18n-ignore: identifier also shown to the reader; S172 translates it by this English key
  { id: 'fig17_14', label: 'Fig. 17.14', source: 'p. 416, letters; values ours', m: FIG17_14 }, // i18n-ignore: identifier also shown to the reader; S172 translates it by this English key
];

// ---------------------------------------------------------------- formats

export interface COO {
  kind: 'coo';
  rowIdx: number[];
  colIdx: number[];
  value: number[];
  /** The non-zero stored at each position. */
  at: Entry[];
}
export interface CSR {
  kind: 'csr';
  /** numRows + 1 entries; the last one is the end marker of the last row. */
  rowPtrs: number[];
  colIdx: number[];
  value: number[];
  at: Entry[];
}
export interface ELL {
  kind: 'ell';
  numRows: number;
  /** Distance between a row's consecutive elements: numRows, or more when rows are padded for alignment. */
  stride: number;
  /** Slots per row: the longest row's length. */
  width: number;
  nnzPerRow: number[];
  /** Column-major: element t of row r is at t*stride + r. Pads hold padMarker. */
  colIdx: number[];
  /** Pads hold 0. */
  value: number[];
  /** The non-zero stored in each slot, or null for a pad. */
  at: (Entry | null)[];
  padMarker: number;
}
export interface Hybrid {
  kind: 'hyb';
  /** The ELL width asked for. */
  K: number;
  ell: ELL;
  /** The tails of the rows longer than K, in row order. */
  coo: COO;
}
export interface JDS {
  kind: 'jds';
  /** row[p]: the original row of sorted position p (longest first, ties in original order). */
  row: number[];
  /** iterPtr[t]: where iteration t's elements begin; one more entry than iterations. */
  iterPtr: number[];
  colIdx: number[];
  value: number[];
  at: Entry[];
}
export interface CSC {
  kind: 'csc';
  colPtrs: number[];
  rowIdx: number[];
  value: number[];
  at: Entry[];
}
export type Format = COO | CSR | ELL | Hybrid | JDS | CSC;
export type FormatId = Format['kind'];
export const FORMAT_NAME: Record<FormatId, string> = {
  coo: 'COO', csr: 'CSR', ell: 'ELL', hyb: 'Hybrid ELL-COO', jds: 'JDS', csc: 'CSC', // i18n-ignore: identifier also shown to the reader; S172 translates it by this English key
};

function cooOf(list: readonly Entry[]): COO {
  return { kind: 'coo', rowIdx: list.map((e) => e.r), colIdx: list.map((e) => e.c), value: list.map((e) => e.v), at: [...list] };
}

/** Fig. 17.6 (p. 406): the same eight non-zeros in another order, as positions in the row-ordered list. */
export const FIG17_6_ORDER: readonly number[] = [4, 7, 5, 6, 0, 2, 3, 1];

/** A seeded shuffle of 0..n-1, the same on every machine. */
export function shuffledOrder(n: number, seed = 17): number[] {
  const out = Array.from({ length: n }, (_, i) => i);
  let s = seed >>> 0 || 1;
  const next = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/** COO in row order (Fig. 17.3), or in any order given as positions in the row-ordered list. */
export function toCOO(m: Matrix, order?: readonly number[]): COO {
  const list = [...m.entries].sort(byRow);
  if (!order) return cooOf(list);
  if (order.length !== list.length || new Set(order).size !== list.length || order.some((k) => k < 0 || k >= list.length))
    throw new Error('order must be a permutation of the non-zeros');
  return cooOf(order.map((k) => list[k]));
}

export function toCSR(m: Matrix): CSR {
  const lists = rowLists(m);
  const rowPtrs = [0];
  for (const l of lists) rowPtrs.push(rowPtrs[rowPtrs.length - 1] + l.length);
  const at = lists.flat();
  return { kind: 'csr', rowPtrs, colIdx: at.map((e) => e.c), value: at.map((e) => e.v), at };
}

function ellOf(lists: readonly Entry[][], numRows: number, stride: number, width: number, padMarker: number): ELL {
  if (stride < numRows) throw new Error('stride must be at least numRows');
  const n = stride * width;
  const colIdx = Array<number>(n).fill(padMarker);
  const value = Array<number>(n).fill(0);
  const at = Array<Entry | null>(n).fill(null);
  lists.forEach((l, r) => l.forEach((e, t) => {
    const i = t * stride + r;
    colIdx[i] = e.c;
    value[i] = e.v;
    at[i] = e;
  }));
  return { kind: 'ell', numRows, stride, width, nnzPerRow: lists.map((l) => l.length), colIdx, value, at, padMarker };
}

export interface EllOptions {
  /** Default numRows (the book). Larger pads whole rows so every iteration starts aligned. */
  stride?: number;
  /** What a pad slot's column index holds: -1, as the libraries do (facts A6), or 0. */
  padMarker?: number;
}
/** Pad every row to the longest, store column-major (Fig. 17.10). */
export function toELL(m: Matrix, { stride = m.rows, padMarker = -1 }: EllOptions = {}): ELL {
  return ellOf(rowLists(m), m.rows, stride, longestRow(m), padMarker);
}

/** The smallest stride at least `rows` that is a multiple of `multiple` (8 floats is 32 bytes). */
export const alignedStride = (rows: number, multiple: number): number => Math.ceil(rows / multiple) * multiple;

/** ELL of width at most K, the tail of every longer row in a COO part (Fig. 17.13). */
export function toHybrid(m: Matrix, K: number, { padMarker = -1 }: { padMarker?: number } = {}): Hybrid {
  const lists = rowLists(m);
  const keep = lists.map((l) => l.slice(0, K));
  const spill = lists.flatMap((l) => l.slice(K));
  const width = Math.max(0, ...keep.map((l) => l.length));
  return { kind: 'hyb', K, ell: ellOf(keep, m.rows, m.rows, width, padMarker), coo: cooOf(spill) };
}

/** Sort rows longest first (stable), then store column-major without padding (Fig. 17.14). */
export function toJDS(m: Matrix): JDS {
  const lists = rowLists(m);
  const row = lists.map((_, r) => r).sort((a, b) => lists[b].length - lists[a].length || a - b);
  const K = Math.max(0, ...lists.map((l) => l.length));
  const iterPtr = [0];
  const at: Entry[] = [];
  for (let t = 0; t < K; t++) {
    for (const r of row) if (lists[r].length > t) at.push(lists[r][t]);
    iterPtr.push(at.length);
  }
  return { kind: 'jds', row, iterPtr, colIdx: at.map((e) => e.c), value: at.map((e) => e.v), at };
}

/** CSR with rows and columns swapped (Fig. 17.16). */
export function toCSC(m: Matrix): CSC {
  const lists = colLists(m);
  const colPtrs = [0];
  for (const l of lists) colPtrs.push(colPtrs[colPtrs.length - 1] + l.length);
  const at = lists.flat();
  return { kind: 'csc', colPtrs, rowIdx: at.map((e) => e.r), value: at.map((e) => e.v), at };
}

export interface BuildOptions {
  /** Hybrid's ELL width. Default: the longest row (pure ELL). */
  K?: number;
  stride?: number;
  padMarker?: number;
  /** COO order (positions in the row-ordered list). */
  order?: readonly number[];
}
export function build(id: FormatId, m: Matrix, o: BuildOptions = {}): Format {
  switch (id) {
    case 'coo': return toCOO(m, o.order);
    case 'csr': return toCSR(m);
    case 'ell': return toELL(m, { stride: o.stride, padMarker: o.padMarker });
    case 'hyb': return toHybrid(m, o.K ?? longestRow(m), { padMarker: o.padMarker });
    case 'jds': return toJDS(m);
    case 'csc': return toCSC(m);
  }
}

// ---------------------------------------------------------------- storage

export interface Storage {
  /** Value slots, pads included. */
  values: number;
  /** Column and row indices, pads included. */
  indices: number;
  /** Row or column pointers, per-row counts, the JDS row and iterPtr arrays. */
  pointers: number;
  /** Slots that hold no non-zero (counted in values and in indices). */
  pads: number;
  total: number;
  /** m·n: the dense matrix. */
  dense: number;
  /** Exercise 2's count for this format in m, n, z, K. */
  formula: string;
}

export function storage(f: Format, m: Matrix): Storage {
  const z = nnz(m);
  const dense = m.rows * m.cols;
  const mk = (values: number, indices: number, pointers: number, pads: number, formula: string): Storage =>
    ({ values, indices, pointers, pads, total: values + indices + pointers, dense, formula });
  switch (f.kind) {
    case 'coo': return mk(z, 2 * z, 0, 0, '3z');
    case 'csr': return mk(z, z, m.rows + 1, 0, '2z + m + 1');
    case 'ell': {
      const slots = f.stride * f.width;
      return mk(slots, slots, f.numRows, slots - z, f.stride === f.numRows ? '2·m·K + m' : '2·stride·K + m'); // i18n-ignore: formula
    }
    case 'hyb': {
      const slots = f.ell.stride * f.ell.width;
      const s = f.coo.value.length;
      return mk(slots + s, slots + 2 * s, f.ell.numRows, slots - (z - s), '2·m·K + m + 3s');
    }
    case 'jds': return mk(z, z, m.rows + f.iterPtr.length, 0, '2z + m + K + 1');
    case 'csc': return mk(z, z, m.cols + 1, 0, '2z + n + 1');
  }
}

/** Exercise 2 (p. 422): integers needed by each format; ELL and JDS need K, the longest row. */
export function exercise2Count(m: number, z: number, K: number) {
  return { coo: 3 * z, csr: 2 * z + m + 1, ell: 2 * m * K, ellWithCounts: 2 * m * K + m, jds: 2 * z + m + K + 1 };
}

// ---------------------------------------------------------------- kernels

export type Mapping = 'per non-zero' | 'per row' | 'per sorted row' | 'per column';
/** Which array a step's indices point into. */
export type ArrayName = 'coo' | 'csr' | 'ell' | 'jds' | 'csc';
export type EllLoop = 'nnzPerRow' | 'full';

export interface Lane {
  thread: number;
  /** The index of value (and colIdx or rowIdx) this lane reads in this step; null when idle. */
  index: number | null;
  /** The slot read is a pad (ELL looping its full width). */
  pad: boolean;
  /** The x element read in this step, or null. */
  x: number | null;
  /** The y element this thread's work goes to: its own row, or the row of its non-zero. */
  y: number | null;
  /** This step updates y[y] with an atomic add. */
  atomic: boolean;
  /** The multiplication done in this step: value read and x index, or null. */
  v: number | null;
  xi: number | null;
}
export interface Step {
  launch: number;
  /** Loop iteration within the launch. */
  t: number;
  array: ArrayName;
  lanes: Lane[];
}
export interface Launch {
  name: string;
  mapping: Mapping;
  threads: number;
  array: ArrayName;
  atomic: boolean;
  /** Loop iterations each thread runs. */
  trips: number[];
  /** Of those, iterations spent on pad slots. */
  padTrips: number[];
}
export interface Trace {
  launches: Launch[];
  steps: Step[];
}

const idle = (thread: number, y: number | null = null): Lane => ({ thread, index: null, pad: false, x: null, y, atomic: false, v: null, xi: null });

function cooLaunch(f: COO, name: string, launch: number): { L: Launch; steps: Step[] } {
  const n = f.value.length;
  const lanes: Lane[] = f.value.map((v, i) => ({ thread: i, index: i, pad: false, x: f.colIdx[i], y: f.rowIdx[i], atomic: true, v, xi: f.colIdx[i] }));
  return {
    L: { name, mapping: 'per non-zero', threads: n, array: 'coo', atomic: true, trips: Array(n).fill(1), padTrips: Array(n).fill(0) }, // i18n-ignore: identifier also shown to the reader; S172 translates it by this English key
    steps: n ? [{ launch, t: 0, array: 'coo', lanes }] : [],
  };
}

function ellLaunch(f: ELL, loop: EllLoop, name: string, launch: number): { L: Launch; steps: Step[] } {
  const R = f.numRows;
  const len = f.nnzPerRow;
  const trips = loop === 'full' ? Array(R).fill(f.width) : [...len];
  const steps: Step[] = [];
  for (let t = 0; t < f.width; t++) {
    const lanes: Lane[] = [];
    for (let r = 0; r < R; r++) {
      if (loop === 'nnzPerRow' && t >= len[r]) { lanes.push(idle(r, r)); continue; }
      const i = t * f.stride + r;
      const pad = f.at[i] === null;
      const col = f.colIdx[i];
      // The full-width loop reads colIdx[i] and goes on only when col >= 0 (facts A6).
      const reads = col >= 0;
      lanes.push({ thread: r, index: i, pad, x: reads ? col : null, y: r, atomic: false, v: reads ? f.value[i] : null, xi: reads ? col : null });
    }
    steps.push({ launch, t, array: 'ell', lanes });
  }
  return {
    L: { name, mapping: 'per row', threads: R, array: 'ell', atomic: false, trips, padTrips: loop === 'full' ? len.map((l) => f.width - l) : Array(R).fill(0) }, // i18n-ignore: identifier also shown to the reader; S172 translates it by this English key
    steps,
  };
}

export interface TraceOptions {
  /** ELL (and the hybrid's ELL part): stop at nnzPerRow[row] (the book) or loop the full width and test col >= 0. */
  ellLoop?: EllLoop;
}

/** What each thread of the format's SpMV kernel reads and updates, iteration by iteration. */
export function spmvTrace(f: Format, { ellLoop = 'nnzPerRow' }: TraceOptions = {}): Trace {
  switch (f.kind) {
    case 'coo': {
      const { L, steps } = cooLaunch(f, 'SpMV/COO', 0);
      return { launches: [L], steps };
    }
    case 'csr': {
      const R = f.rowPtrs.length - 1;
      const len = Array.from({ length: R }, (_, r) => f.rowPtrs[r + 1] - f.rowPtrs[r]);
      const K = Math.max(0, ...len);
      const steps: Step[] = [];
      for (let t = 0; t < K; t++)
        steps.push({
          launch: 0, t, array: 'csr',
          lanes: len.map((l, r) => {
            if (t >= l) return idle(r, r);
            const i = f.rowPtrs[r] + t;
            return { thread: r, index: i, pad: false, x: f.colIdx[i], y: r, atomic: false, v: f.value[i], xi: f.colIdx[i] };
          }),
        });
      return { launches: [{ name: 'SpMV/CSR', mapping: 'per row', threads: R, array: 'csr', atomic: false, trips: len, padTrips: Array(R).fill(0) }], steps }; // i18n-ignore: identifier also shown to the reader; S172 translates it by this English key
    }
    case 'ell': {
      const { L, steps } = ellLaunch(f, ellLoop, 'SpMV/ELL', 0);
      return { launches: [L], steps };
    }
    case 'hyb': {
      const a = ellLaunch(f.ell, ellLoop, 'SpMV/ELL part', 0); // i18n-ignore: identifier also shown to the reader; S172 translates it by this English key
      const b = cooLaunch(f.coo, 'SpMV/COO part', 1); // i18n-ignore: identifier also shown to the reader; S172 translates it by this English key
      return { launches: [a.L, b.L], steps: [...a.steps, ...b.steps] };
    }
    case 'jds': {
      const R = f.row.length;
      const K = f.iterPtr.length - 1;
      const count = (t: number) => f.iterPtr[t + 1] - f.iterPtr[t];
      const trips = Array.from({ length: R }, (_, p) => { let n = 0; while (n < K && p < count(n)) n++; return n; });
      const steps: Step[] = [];
      for (let t = 0; t < K; t++)
        steps.push({
          launch: 0, t, array: 'jds',
          lanes: f.row.map((orig, p) => {
            if (p >= count(t)) return idle(p, orig);
            const i = f.iterPtr[t] + p;
            return { thread: p, index: i, pad: false, x: f.colIdx[i], y: orig, atomic: false, v: f.value[i], xi: f.colIdx[i] };
          }),
        });
      return { launches: [{ name: 'SpMV/JDS', mapping: 'per sorted row', threads: R, array: 'jds', atomic: false, trips, padTrips: Array(R).fill(0) }], steps }; // i18n-ignore: identifier also shown to the reader; S172 translates it by this English key
    }
    case 'csc': {
      const C = f.colPtrs.length - 1;
      const len = Array.from({ length: C }, (_, c) => f.colPtrs[c + 1] - f.colPtrs[c]);
      const K = Math.max(0, ...len);
      const steps: Step[] = [];
      for (let t = 0; t < K; t++)
        steps.push({
          launch: 0, t, array: 'csc',
          lanes: len.map((l, c) => {
            // x[col] is loaded once, before the loop (Fig. 17.18 line 04): shown at t = 0.
            const x = t === 0 ? c : null;
            if (t >= l) return { ...idle(c), x };
            const i = f.colPtrs[c] + t;
            return { thread: c, index: i, pad: false, x, y: f.rowIdx[i], atomic: true, v: f.value[i], xi: c };
          }),
        });
      return { launches: [{ name: 'SpMV/CSC', mapping: 'per column', threads: C, array: 'csc', atomic: true, trips: len, padTrips: Array(C).fill(0) }], steps }; // i18n-ignore: identifier also shown to the reader; S172 translates it by this English key
    }
  }
}

// ---------------------------------------------------------------- what a trace costs

export interface LaunchSummary {
  name: string;
  mapping: Mapping;
  threads: number;
  /** Iterations the groups run, summed over groups (each group runs as long as its longest lane). */
  groupIterations: number;
  /** Lane-iterations that multiply a non-zero. */
  useful: number;
  /** Lane-iterations spent on pad slots. */
  pad: number;
  /** Lane-iterations a lane waits while others in its group go on (divergence). */
  idle: number;
}
export interface TraceSummary {
  launches: LaunchSummary[];
  /** Atomic updates of y, in all. */
  atomicUpdates: number;
  /** y elements that more than one thread updates atomically. */
  sharedY: number;
  /** For each y element, how many threads update it atomically. */
  atomicThreadsPerY: number[];
  groupIterations: number;
  useful: number;
  pad: number;
  idle: number;
}

/** Divergence (divergence.ts loopBatch over groups of `width` threads), pad work and atomics. */
export function summarize(trace: Trace, width: number, rows: number): TraceSummary {
  const launches = trace.launches.map((L): LaunchSummary => {
    const b = loopBatch(L.trips, width);
    const pad = L.padTrips.reduce((a, x) => a + x, 0);
    return { name: L.name, mapping: L.mapping, threads: L.threads, groupIterations: b.spent, useful: b.useful - pad, pad, idle: b.idle };
  });
  const hit: Set<number>[] = Array.from({ length: rows }, () => new Set());
  let atomicUpdates = 0;
  for (const s of trace.steps)
    for (const l of s.lanes)
      if (l.atomic && l.y != null) {
        atomicUpdates++;
        hit[l.y].add(s.launch * 1e6 + l.thread);
      }
  const per = hit.map((h) => h.size);
  const sum = (k: 'groupIterations' | 'useful' | 'pad' | 'idle') => launches.reduce((a, l) => a + l[k], 0);
  return {
    launches, atomicUpdates, sharedY: per.filter((n) => n > 1).length, atomicThreadsPerY: per,
    groupIterations: sum('groupIterations'), useful: sum('useful'), pad: sum('pad'), idle: sum('idle'),
  };
}

/** Lanes of a step cut into groups of `width` threads, in thread order. */
export const groupsOf = <T>(lanes: readonly T[], width: number): T[][] =>
  Array.from({ length: Math.ceil(lanes.length / width) }, (_, g) => lanes.slice(g * width, (g + 1) * width));

/**
 * For each group, how many runs of consecutive addresses its matrix reads form in this
 * step (coalesce.ts runs over 4-byte elements); 0 for a group with nothing to read.
 */
export const matrixRuns = (step: Step, width: number): number[] =>
  groupsOf(step.lanes, width).map((g) => {
    const addr = g.filter((l) => l.index != null).map((l) => l.index! * 4);
    return addr.length ? runs(addr, 4) : 0;
  });

/** The same count for the reads of x. */
export const xRuns = (step: Step, width: number): number[] =>
  groupsOf(step.lanes, width).map((g) => {
    const addr = g.filter((l) => l.x != null).map((l) => l.x! * 4);
    return addr.length ? runs(addr, 4) : 0;
  });

/** Idle lanes in a step (threads with nothing to read). CSC's x load before the loop does not count as work. */
export const idleLanes = (step: Step): number => step.lanes.filter((l) => l.index == null).length;

// ---------------------------------------------------------------- arithmetic

const f32 = Math.fround;

/** y = A·x + y in float32, row by row, columns in order. */
export function spmvRef(m: Matrix, x: readonly number[], y0: readonly number[]): number[] {
  const y = y0.map(f32);
  for (const l of rowLists(m)) {
    if (!l.length) continue;
    let sum = 0;
    for (const e of l) sum = f32(sum + f32(e.v * x[e.c]));
    y[l[0].r] = f32(y[l[0].r] + sum);
  }
  return y;
}

/**
 * Run a trace in float32: steps in order, lanes in thread order. Atomic lanes add into y
 * at once; the others keep a sum per thread and add it to y when their launch ends.
 */
export function runTrace(trace: Trace, x: readonly number[], y0: readonly number[]): number[] {
  const y = y0.map(f32);
  trace.launches.forEach((L, li) => {
    const sum = Array<number>(L.threads).fill(0);
    const owner = Array<number | null>(L.threads).fill(null);
    for (const s of trace.steps) {
      if (s.launch !== li) continue;
      for (const l of s.lanes) {
        if (l.y != null && !L.atomic) owner[l.thread] = l.y;
        if (l.v == null || l.xi == null) continue;
        const p = f32(l.v * x[l.xi]);
        if (l.atomic) y[l.y!] = f32(y[l.y!] + p);
        else sum[l.thread] = f32(sum[l.thread] + p);
      }
    }
    if (!L.atomic) owner.forEach((r, t) => { if (r != null) y[r] = f32(y[r] + sum[t]); });
  });
  return y;
}

/** Sum terms in the given order, in float32, starting from `start`. */
export function floatSum(terms: readonly number[], order: readonly number[], start = 0): number {
  let s = f32(start);
  for (const k of order) s = f32(s + f32(terms[k]));
  return s;
}

/** Every order of 0..n-1. */
export function permutations(n: number): number[][] {
  if (n === 0) return [[]];
  return permutations(n - 1).flatMap((p) => Array.from({ length: n }, (_, k) => [...p.slice(0, k), n - 1, ...p.slice(k)]));
}

/** Three products that one y element receives in the stepper's second panel. */
export const ORDER_TERMS: readonly number[] = [1e8, 1, -1e8];

/** Results of summing the terms in every order, ascending, without repeats. */
export const floatOutcomes = (terms: readonly number[], start = 0): number[] =>
  [...new Set(permutations(terms.length).map((o) => floatSum(terms, o, start)))].sort((a, b) => a - b);

// ---------------------------------------------------------------- two threads, one y element

/** Fig. 17.1 row 0 with x all ones: the two products thread 0 and thread 1 add into y[0]. */
export const ROW0_PRODUCTS: readonly [number, number] = [1, 7];

export type RmwOp = 'load' | 'add' | 'store' | 'atomic add'; // i18n-ignore: identifier also shown to the reader; S172 translates it by this English key
export interface RmwEvent {
  thread: number;
  op: RmwOp;
  /** y in memory after the step. */
  y: number;
  /** The thread's register after the step. */
  reg: number;
}
export interface RmwState {
  y: number;
  /** Per thread: 0 = load next, 1 = add next, 2 = store next, 3 = done. */
  pc: number[];
  reg: (number | null)[];
  events: RmwEvent[];
}

export const rmwInit = (y0 = 0, threads = 2): RmwState => ({ y: y0, pc: Array(threads).fill(0), reg: Array(threads).fill(null), events: [] });
export const rmwDone = (s: RmwState): boolean => s.pc.every((p) => p >= 3);
export const RMW_NEXT = ['load y', 'add its product', 'store y', 'done'] as const; // i18n-ignore: identifier also shown to the reader; S172 translates it by this English key

/**
 * One step of one thread. Without the atomic a step is one of load, add, store; with it
 * the three happen as one indivisible step, so no other thread can come between them.
 */
export function rmwStep(s: RmwState, thread: number, products: readonly number[], atomic: boolean): RmwState {
  const pc = [...s.pc];
  const reg = [...s.reg];
  let y = s.y;
  if (pc[thread] >= 3) return s;
  let op: RmwOp;
  if (atomic) {
    reg[thread] = f32(y + products[thread]);
    y = reg[thread]!;
    pc[thread] = 3;
    op = 'atomic add'; // i18n-ignore: identifier also shown to the reader; S172 translates it by this English key
  } else if (pc[thread] === 0) {
    reg[thread] = y;
    pc[thread] = 1;
    op = 'load';
  } else if (pc[thread] === 1) {
    reg[thread] = f32(reg[thread]! + products[thread]);
    pc[thread] = 2;
    op = 'add';
  } else {
    y = reg[thread]!;
    pc[thread] = 3;
    op = 'store';
  }
  return { y, pc, reg, events: [...s.events, { thread, op, y, reg: reg[thread]! }] };
}

/** Run a schedule (a list of thread numbers; a finished thread's turn is skipped). */
export function cooUpdates(schedule: readonly number[], { products = ROW0_PRODUCTS, atomic = false, y0 = 0 }: { products?: readonly number[]; atomic?: boolean; y0?: number } = {}): RmwState {
  let s = rmwInit(y0, products.length);
  for (const t of schedule) s = rmwStep(s, t, products, atomic);
  return s;
}

/** Every interleaving of the threads' steps, as schedules. */
export function interleavings(stepsPerThread: readonly number[]): number[][] {
  const out: number[][] = [];
  const left = [...stepsPerThread];
  const cur: number[] = [];
  const go = () => {
    if (left.every((n) => n === 0)) { out.push([...cur]); return; }
    left.forEach((n, t) => {
      if (!n) return;
      left[t]--;
      cur.push(t);
      go();
      cur.pop();
      left[t]++;
    });
  };
  go();
  return out;
}

/** Final y over every interleaving, ascending, without repeats. */
export function cooOutcomes({ products = ROW0_PRODUCTS, atomic = false, y0 = 0 }: { products?: readonly number[]; atomic?: boolean; y0?: number } = {}): number[] {
  const per = atomic ? 1 : 3;
  return [...new Set(interleavings(products.map(() => per)).map((s) => cooUpdates(s, { products, atomic, y0 }).y))].sort((a, b) => a - b);
}
