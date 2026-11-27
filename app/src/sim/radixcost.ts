// What a whole radix sort costs as the digit width r (section 14.7, pp. 342-344) and the
// keys per thread c (section 14.8, pp. 344-345) change: the numbers behind the two
// explorers on those pages.
//
// What it models. The local-sort design of section 14.6 run as three kernels per pass
// (count each block's digits, scan the bucket table, then sort locally and scatter: the
// split of exercise 1, p. 347, and of PyTorch's MPS sort, ch14-facts D6). For a problem
// of `rows` rows of `rowLength` keys, sorted row by row with no block spanning two rows:
//   - passes = ceil(key bits / r), three launches each;
//   - blocks = rows x ceil(rowLength / (threads x c)); table entries = 2^r x blocks;
//   - global traffic, counted, not measured (ours): per pass the count kernel reads every
//     key, the scatter kernel reads and writes every (key, 4-byte value) pair, and every
//     table entry is touched four times (written by the count, read and written by the
//     scan, read by the scatter). The one-bit sort of section 14.5 is given for reference
//     as Fig. 14.7 with pairs: read and write every pair, plus the bits array written,
//     scanned (read and written once) and read back;
//   - shared memory per block: the tile of c x threads (key, value) pairs plus 2^r 4-byte
//     counts (the scan's own scratch is not counted);
//   - runs and memory units per warp store from radixlocal.ts, on a seeded sample of a
//     few blocks of keys, averaged over every pass of the sort (each pass sees the keys as
//     the earlier passes left them).
//
// What it leaves out. Caches, occupancy, registers, latency, timing, the scan's own
// hierarchy (a large table is scanned in several levels, chapter 11). Device limits come
// only from the reader's reports, never from this file.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.
import {
  BOOK_KEYS,
  passCount,
  softmaxKeys,
  storeTotals,
  uniformKeys,
  type Design,
} from './radixlocal';
import { tr } from '../lib/locale.ts';

export type Dist = 'uniform' | 'softmax';

export interface Problem {
  id: string;
  label: string;
  rows: number;
  rowLength: number;
  dist: Dist;
}

/** One vocabulary row (ch14-facts: vocabulary sizes are settled in ch09-facts D7). */
export const VOCAB = 151936;

export const PROBLEMS: Problem[] = [
  {
    id: 'n20',
    get label() {
      return tr('2²⁰ random keys', '2²⁰ khóa ngẫu nhiên');
    },
    rows: 1,
    rowLength: 2 ** 20,
    dist: 'uniform',
  },
  {
    id: 'n24',
    get label() {
      return tr('2²⁴ random keys', '2²⁴ khóa ngẫu nhiên');
    },
    rows: 1,
    rowLength: 2 ** 24,
    dist: 'uniform',
  },
  {
    id: 'row',
    get label() {
      return tr('one 151,936-token row', 'một hàng 151,936 token');
    },
    rows: 1,
    rowLength: VOCAB,
    dist: 'softmax',
  },
  {
    id: 'batch',
    get label() {
      return tr('64 such rows', '64 hàng như thế');
    },
    rows: 64,
    rowLength: VOCAB,
    dist: 'softmax',
  },
];

/** The book's sixteen 4-bit keys as a problem (Figs. 14.10 to 14.13). */
export const BOOK_PROBLEM: Problem = {
  id: 'book',
  get label() {
    return tr("the book's sixteen keys", 'mười sáu khóa của sách');
  },
  rows: 1,
  rowLength: BOOK_KEYS.length,
  dist: 'uniform',
};

export const passes = passCount;
export const LAUNCHES_PER_PASS = 3;
/** Each table entry is written by the count, read and written by the scan, read by the scatter. */
export const TABLE_TOUCHES = 4;

export const keyBytes = (keyBits: number) => keyBits / 8;
/** A key and a 4-byte value (the token index). */
export const pairBytes = (keyBits: number) => keyBits / 8 + 4;

export const totalKeys = (p: Pick<Problem, 'rows' | 'rowLength'>) => p.rows * p.rowLength;
export const blocksPerRow = (rowLength: number, threads: number, c: number) => Math.ceil(rowLength / (threads * c));
export const blocks = (p: Pick<Problem, 'rows' | 'rowLength'>, threads: number, c: number) =>
  p.rows * blocksPerRow(p.rowLength, threads, c);
export const tableEntries = (p: Pick<Problem, 'rows' | 'rowLength'>, r: number, threads: number, c: number) =>
  2 ** r * blocks(p, threads, c);
/** Keys a block owns divided by its buckets: what one local bucket holds on average. */
export const keysPerBucket = (threads: number, c: number, r: number) => (threads * c) / 2 ** r;
/** The block's tile of (key, value) pairs in shared memory. */
export const tileBytes = (threads: number, c: number, pair = 8) => threads * c * pair;
/** The tile plus the block's 2^r counts. */
export const sharedBytes = (threads: number, c: number, r: number, pair = 8) => tileBytes(threads, c, pair) + 2 ** r * 4;

export interface Traffic {
  passes: number;
  launches: number;
  /** Bytes per pass spent on keys and values. */
  keysPerPass: number;
  /** Bytes per pass spent on the table (or the bits array of the one-bit sort). */
  tablePerPass: number;
  /** Bytes over the whole sort. */
  total: number;
  /** total / keys. */
  perKey: number;
}

export function localTraffic(p: Pick<Problem, 'rows' | 'rowLength'>, keyBits: number, r: number, threads: number, c: number): Traffic {
  const n = totalKeys(p);
  const P = passes(keyBits, r);
  const keysPerPass = n * (keyBytes(keyBits) + 2 * pairBytes(keyBits));
  const tablePerPass = tableEntries(p, r, threads, c) * 4 * TABLE_TOUCHES;
  const total = P * (keysPerPass + tablePerPass);
  return { passes: P, launches: P * LAUNCHES_PER_PASS, keysPerPass, tablePerPass, total, perKey: total / n };
}

/** Fig. 14.7 with (key, value) pairs, one bit per pass, launched once per pass as the book draws it. */
export function directTraffic(p: Pick<Problem, 'rows' | 'rowLength'>, keyBits: number): Traffic {
  const n = totalKeys(p);
  const P = passes(keyBits, 1);
  const keysPerPass = n * 2 * pairBytes(keyBits);
  const tablePerPass = n * 4 * 4; // bits[i] written, scanned (read, written), read back
  const total = P * (keysPerPass + tablePerPass);
  return { passes: P, launches: P, keysPerPass, tablePerPass, total, perKey: total / n };
}

// ---------------------------------------------------------------- sampled runs and units

export interface SampleSpec {
  dist: Dist;
  keyBits: 32 | 16;
  rowLength: number;
  r: number;
  threads: number;
  c: number;
  design?: Design;
  warpWidth?: number;
  unitBytes?: number;
  seed?: number;
}

export interface Sampled {
  /** Keys in the sample and the blocks they make. */
  n: number;
  blocks: number;
  passes: number;
  stores: number;
  meanRuns: number;
  meanUnits: number;
  /** Units a fully coalesced store of the same lanes needs. */
  meanMin: number;
  ratio: number;
}

/** A few blocks' worth of keys: at least 2 blocks, at most 16, about 16,384 keys, never more than a row. */
export function sampleSize(rowLength: number, threads: number, c: number): { n: number; blocks: number } {
  const per = threads * c;
  const b = Math.max(2, Math.min(16, Math.floor(16384 / per)));
  const n = Math.min(rowLength, b * per);
  return { n, blocks: Math.ceil(n / per) };
}

const keyCache = new Map<string, number[]>();
export function sampleKeys(dist: Dist, keyBits: 32 | 16, n: number, rowLength: number, seed: number): number[] {
  const k = `${dist}/${keyBits}/${n}/${rowLength}/${seed}`;
  let v = keyCache.get(k);
  if (!v) {
    v = dist === 'uniform' ? uniformKeys(n, seed, keyBits) : softmaxKeys(n, seed, keyBits, rowLength);
    if (keyCache.size > 32) keyCache.clear();
    keyCache.set(k, v);
  }
  return v;
}

/** Runs and units per warp store, averaged over every pass of a sort of the sample. */
export function sampled(s: SampleSpec): Sampled {
  const design = s.design ?? 'local';
  const W = s.warpWidth ?? 32;
  const unit = s.unitBytes ?? 32;
  const seed = s.seed ?? 1;
  const elem = keyBytes(s.keyBits);
  const c = design === 'direct' ? 1 : s.c;
  const { n, blocks: nb } = sampleSize(s.rowLength, s.threads, c);
  let a = sampleKeys(s.dist, s.keyBits, n, s.rowLength, seed);
  const P = passes(s.keyBits, s.r);
  let stores = 0, runs = 0, units = 0, min = 0;
  for (let p = 0; p < P; p++) {
    const t = storeTotals(design, a, p * s.r, s.r, s.threads, c, W, elem, unit);
    stores += t.stores;
    runs += t.runs;
    units += t.units;
    min += t.min;
    a = t.out;
  }
  return { n, blocks: nb, passes: P, stores, meanRuns: runs / stores, meanUnits: units / stores, meanMin: min / stores, ratio: units / min };
}

// ---------------------------------------------------------------- sweeps for the explorers

export const R_CHOICES = [1, 2, 3, 4, 5, 6, 7, 8, 16];
export const C_CHOICES = [1, 2, 4, 8, 16, 32];

export interface Row {
  r: number;
  c: number;
  passes: number;
  launches: number;
  blocks: number;
  tableEntries: number;
  /** Table entries per key: 1 means the table is as long as the data. */
  tableToKeys: number;
  keysPerBucket: number;
  sharedBytes: number;
  traffic: Traffic;
  local: Sampled;
}

export function row(p: Problem, keyBits: 32 | 16, r: number, threads: number, c: number, seed = 1): Row {
  const n = totalKeys(p);
  const te = tableEntries(p, r, threads, c);
  return {
    r,
    c,
    passes: passes(keyBits, r),
    launches: passes(keyBits, r) * LAUNCHES_PER_PASS,
    blocks: blocks(p, threads, c),
    tableEntries: te,
    tableToKeys: te / n,
    keysPerBucket: keysPerBucket(threads, c, r),
    sharedBytes: sharedBytes(threads, c, r, pairBytes(keyBits)),
    traffic: localTraffic(p, keyBits, r, threads, c),
    local: sampled({ dist: p.dist, keyBits, rowLength: p.rowLength, r, threads, c, seed }),
  };
}

/** Section 14.7: every digit width at one key per thread. */
export function sweepR(p: Problem, keyBits: 32 | 16, threads: number, seed = 1): Row[] {
  return R_CHOICES.map((r) => row(p, keyBits, r, threads, 1, seed));
}

/** Section 14.8: every coarsening factor at one digit width. */
export function sweepC(p: Problem, keyBits: 32 | 16, r: number, threads: number, seed = 1): Row[] {
  return C_CHOICES.map((c) => row(p, keyBits, r, threads, c, seed));
}

/** The direct scatter's runs and units for the same sample, for comparison. */
export function sampledDirect(p: Problem, keyBits: 32 | 16, r: number, threads: number, seed = 1): Sampled {
  return sampled({ dist: p.dist, keyBits, rowLength: p.rowLength, r, threads, c: 1, design: 'direct', seed });
}

// ---------------------------------------------------------------- lab 14.2's measurements

/** A benchmark row as reports/runs/<backend>.json stores it. */
export interface LabRow {
  name: string;
  [k: string]: string | number;
}

/** The path of lab 14.2 in the reports, and the row names its benchmark writes. */
export const LAB2 = 'ch14/lab2_multibit';

const numOf = (v: unknown): number | undefined => (typeof v === 'number' && isFinite(v) && v > 0 ? v : undefined);

/** Seconds of lab 14.2's local-sort design at one digit width and keys per thread, if it ran. */
export function measuredLocal(rows: readonly LabRow[], r: number, c: number): number | undefined {
  const hits = rows.filter((x) => x.name === 'local' && Number(x.radix_bits) === r && Number(x.keys_per_thread) === c);
  const s = hits.map((x) => numOf(x.seconds)).filter((v): v is number => v != null);
  return s.length ? Math.min(...s) : undefined;
}

/** Seconds of the provided one-bit direct-scatter sort, if it ran. */
export function measuredOneBit(rows: readonly LabRow[]): number | undefined {
  const s = rows.filter((x) => x.name === 'onebit_global').map((x) => numOf(x.seconds)).filter((v): v is number => v != null);
  return s.length ? Math.min(...s) : undefined;
}

/** The fastest of several measured settings, or undefined when none ran. */
export function fastest(entries: readonly { at: number; seconds: number | undefined }[]): { at: number; seconds: number } | undefined {
  let best: { at: number; seconds: number } | undefined;
  for (const e of entries) if (e.seconds != null && (!best || e.seconds < best.seconds)) best = { at: e.at, seconds: e.seconds };
  return best;
}
