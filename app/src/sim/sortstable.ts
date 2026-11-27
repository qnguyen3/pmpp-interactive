// Key-value sorting basics for chapter 14, section 14.1 (pp. 330-331): what a sort must
// deliver, what stability adds, and the cascaded sort of p. 330 (sort by the secondary key
// first, then stably by the primary key), which PyTorch uses to sort a few rows of a
// (rows, vocabulary) tensor (notes/today/ch14-facts.md D5).
//
// What it models: a stable sort (equal keys keep their input order, by construction:
// ties are broken by input position) and an unstable one. "Unstable" here is one
// specific network, the bitonic network of section 14.9 (src/sim/sortnet.ts), padded to
// a power of two. An unstable sort promises nothing, so another one could reorder ties
// differently or, by luck, not at all; this one is fixed so that the lessons can show
// what it does.
//
// What it leaves out: memory, threads, timing. Keys are plain numbers; probabilities are
// written as short decimals, not bf16 bit patterns.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.
import { tr } from '../lib/locale.ts';
import { byValue, networkSort, type Item } from './sortnet';

export type Dir = 'asc' | 'desc';

/** Stable: ties keep input order (sorted on (key, input position)). */
export function stableSort<T>(items: T[], key: (t: T) => number, dir: Dir): T[] {
  const s = dir === 'asc' ? 1 : -1;
  return items
    .map((t, i) => ({ t, i, k: key(t) }))
    .sort((a, b) => (a.k !== b.k ? s * (a.k - b.k) : a.i - b.i))
    .map((x) => x.t);
}

/** Unstable: the bitonic network of section 14.9, comparing keys only. */
export function unstableSort<T>(items: T[], key: (t: T) => number, dir: Dir): T[] {
  return networkSort(items, dir === 'asc' ? (a, b) => key(a) < key(b) : (a, b) => key(a) > key(b));
}

// ---------------------------------------------------------------- candidates: ties
export type SortKind = 'stable' | 'network';

/**
 * A candidate list (probability, token) sorted largest first. `byIndex` sorts (value,
 * index) pairs instead, which makes any sort put equal values in increasing token order.
 */
export function sortCandidates(items: Item[], kind: SortKind, byIndex: boolean): Item[] {
  if (kind === 'network') return networkSort(items, byValue(byIndex));
  if (!byIndex) return stableSort(items, (t) => t.key, 'desc');
  return items.slice().sort((a, b) => (a.key !== b.key ? b.key - a.key : a.id - b.id));
}

/** The tokens of the section 14.1 explorer: increasing, as a row arrives in token order. */
export const CANDIDATE_TOKENS = [3, 8, 15, 21, 40, 52, 77, 90];
const cands = (ps: number[]): Item[] => ps.map((key, i) => ({ key, id: CANDIDATE_TOKENS[i] }));
export const CANDIDATE_PRESETS: { id: string; label: string; items: Item[] }[] = [
  {
    id: 'two',
    get label() {
      return tr('ties at two values', 'trùng nhau ở hai giá trị');
    },
    items: cands([0.15, 0.2, 0.15, 0.05, 0.2, 0.15, 0.08, 0.02]),
  },
  {
    id: 'equal',
    get label() {
      return tr('all equal', 'tất cả bằng nhau');
    },
    items: cands([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]),
  },
  {
    id: 'none',
    get label() {
      return tr('no ties', 'không có giá trị trùng');
    },
    items: cands([0.3, 0.05, 0.12, 0.2, 0.08, 0.15, 0.03, 0.07]),
  },
];

// ---------------------------------------------------------------- the cascade
/** One entry of a flattened (rows, vocabulary) batch. */
export interface BatchItem {
  row: number;
  token: number;
  p: number;
}
export type PassKey = 'value' | 'row';
export interface Pass {
  key: PassKey;
  stable: boolean;
}
/** Value passes sort largest first; row passes sort by increasing row id. */
export function applyPass(items: BatchItem[], pass: Pass): BatchItem[] {
  const key = pass.key === 'value' ? (t: BatchItem) => t.p : (t: BatchItem) => t.row;
  const dir: Dir = pass.key === 'value' ? 'desc' : 'asc';
  return pass.stable ? stableSort(items, key, dir) : unstableSort(items, key, dir);
}

/** Runs the passes in order and returns the list after each one (index 0 is the input). */
export function cascade(items: BatchItem[], passes: Pass[]): BatchItem[][] {
  const out = [items.slice()];
  for (const p of passes) out.push(applyPass(out[out.length - 1], p));
  return out;
}

export interface RowCheck {
  row: number;
  /** The row's items fill exactly its own block of positions, rows in order. */
  contiguous: boolean;
  /** The row's values, in output order, never increase. */
  sorted: boolean;
  ok: boolean;
}

/** Is every row contiguous (in row order) and sorted largest first? */
export function rowChecks(items: BatchItem[], rows: number, perRow: number): RowCheck[] {
  return Array.from({ length: rows }, (_, r) => {
    const at = items.flatMap((t, i) => (t.row === r ? [i] : []));
    const contiguous = at.length === perRow && at.every((pos, k) => pos === r * perRow + k);
    const ps = at.map((i) => items[i].p);
    const sorted = ps.every((v, k) => k === 0 || ps[k - 1] >= v);
    return { row: r, contiguous, sorted, ok: contiguous && sorted };
  });
}

/** The section 14.1 batch: two rows of six tokens, flattened row by row. */
export const BATCH_ROWS = 2;
export const BATCH_PER_ROW = 6;
export const BATCH: BatchItem[] = [
  [0.1, 0.3, 0.05, 0.25, 0.2, 0.1],
  [0.4, 0.05, 0.15, 0.1, 0.25, 0.05],
].flatMap((ps, row) => ps.map((p, token) => ({ row, token, p })));

// ---------------------------------------------------------------- the book's list
/** The book's tuples (age, income in thousands of dollars), p. 330. */
export interface Person {
  age: number;
  income: number;
}
export const BOOK_LIST: Person[] = [
  { age: 30, income: 150 },
  { age: 32, income: 80 },
  { age: 22, income: 45 },
  { age: 29, income: 80 },
];
