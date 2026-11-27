// LSD radix sort for chapter 14, sections 14.4 and 14.5 (pp. 334-339, Figs. 14.4 to 14.7
// and 14.10), and the narrated scene "Sorting one bit at a time".
//
// What it models, exactly:
//  - A pass (section 14.4): a stable partition of the keys by one digit of `bits` bits at
//    `shift`. Keys go to the bucket of their digit, buckets in digit order, each bucket in
//    input order. The "unstable" variant fills the highest non-empty bucket in reverse
//    input order, the one bucket of a 1-bit pass, so that exactly one thing changes.
//  - A sort: ceil(keyBits / radixBits) passes, least significant digit first, or most
//    significant first (the wrong order for global passes). Every pass's mapping is kept,
//    and every key keeps its identity (its input position), so a key can be followed even
//    where two keys are equal.
//  - One parallel pass (section 14.5, Figs. 14.5 to 14.7): one thread per key in blocks of
//    blockDim threads. Each thread extracts its bit; an exclusive scan of the bits gives
//    "ones before"; entry n of the scan (the n bits and a zero after them) is the total;
//    a zero goes to i - onesBefore and a one to n - total + onesBefore. Defect variants:
//    ones use the zero formula; the scan has n entries, so reading entry n is out of range;
//    the total is taken from the last of n entries (Fig. 14.6's reading, right only when
//    the last key's bit is 0); an inclusive scan. Each reports collisions (two threads,
//    one slot), holes (a slot nobody writes), out-of-range reads and writes.
//
// What it leaves out: the grid-wide scan is one step that every thread sees complete at
// once; there are no warps, no memory and no time; whatever an out-of-range read returns
// is not modelled (the destinations that depend on it are unknown).
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

/** The book's sixteen 4-bit keys (Figs. 14.4 to 14.6 and 14.8 to 14.13). */
import { tr } from '../lib/locale.ts';

export const BOOK_KEYS: number[] = [
  0b1100, 0b0011, 0b0110, 0b1001, 0b1111, 0b1000, 0b0101, 0b1010,
  0b1001, 0b0110, 0b1011, 0b1101, 0b0100, 0b1010, 0b0111, 0b0000,
];

export const bin = (k: number, bits: number) => (k >>> 0).toString(2).padStart(bits, '0');
export const digit = (key: number, shift: number, bits: number) => (key >>> shift) & ((1 << bits) - 1);

export interface PassResult {
  shift: number;
  bits: number;
  input: number[];
  output: number[];
  /** dst[i]: where input position i lands. */
  dst: number[];
  /** src[j]: which input position output j came from. */
  src: number[];
  /** Bucket sizes by digit value. */
  counts: number[];
  /** Start of each bucket in the output. */
  starts: number[];
  unstable: boolean;
}

/** One pass: a stable partition by one digit (or the unstable variant). */
export function partitionPass(keys: number[], shift: number, bits: number, unstable = false): PassResult {
  const R = 1 << bits;
  const counts = Array(R).fill(0);
  for (const k of keys) counts[digit(k, shift, bits)]++;
  const starts = Array(R).fill(0);
  for (let d = 1; d < R; d++) starts[d] = starts[d - 1] + counts[d - 1];
  let top = R - 1;
  while (top > 0 && counts[top] === 0) top--;
  const dst: number[] = Array(keys.length);
  const seen = Array(R).fill(0);
  keys.forEach((k, i) => {
    const d = digit(k, shift, bits);
    const r = seen[d]++;
    dst[i] = unstable && d === top ? starts[d] + counts[d] - 1 - r : starts[d] + r;
  });
  const output: number[] = Array(keys.length);
  const src: number[] = Array(keys.length);
  dst.forEach((j, i) => {
    output[j] = keys[i];
    src[j] = i;
  });
  return { shift, bits, input: keys.slice(), output, dst, src, counts, starts, unstable };
}

/** Largest k such that the list is in order by its low k bits (k = keyBits: fully sorted). */
export function sortedLow(keys: number[], keyBits: number): number {
  // Not monotone in k: a list sorted by its low two bits is usually not sorted by bit 0.
  for (let k = keyBits; k >= 1; k--) {
    const m = k === 32 ? 0xffffffff : (1 << k) - 1;
    let ok = true;
    for (let i = 0; i + 1 < keys.length && ok; i++) if (((keys[i] & m) >>> 0) > ((keys[i + 1] & m) >>> 0)) ok = false;
    if (ok) return k;
  }
  return 0;
}

/** First i with keys[i] > keys[i + 1], or -1 when the list is sorted. */
export function firstDisorder(keys: number[]): number {
  for (let i = 0; i + 1 < keys.length; i++) if (keys[i] >>> 0 > keys[i + 1] >>> 0) return i;
  return -1;
}

export type Order = 'lsd' | 'msd';
export interface SortOptions {
  order?: Order;
  /** 1-based pass to make unstable, or null. */
  unstablePass?: number | null;
}
export interface SortPass extends PassResult {
  /** 1-based. */
  index: number;
  /** Identity (original input position) of each input slot and each output slot. */
  inIds: number[];
  outIds: number[];
  /** The output is in order by its low sortedBy bits (computed from the data). */
  sortedBy: number;
}
export interface SortRun {
  keys: number[];
  keyBits: number;
  radixBits: number;
  order: Order;
  unstablePass: number | null;
  passes: SortPass[];
  result: number[];
  sorted: boolean;
  firstDisorder: number;
  /** Identity of each output slot (original input position). */
  resultIds: number[];
  /** Equal keys are in input order (what a sort with payloads needs). */
  stable: boolean;
  /** First output position j where equal keys j and j + 1 are out of input order, or -1. */
  firstTieSwap: number;
}

export const passCount = (keyBits: number, radixBits: number) => Math.ceil(keyBits / radixBits);

/** The digit positions of the passes, in the order they run. */
export function shifts(keyBits: number, radixBits: number, order: Order): number[] {
  const s = Array.from({ length: passCount(keyBits, radixBits) }, (_, p) => p * radixBits);
  return order === 'lsd' ? s : s.reverse();
}

/** A whole radix sort, pass by pass. */
export function lsdSort(keys: number[], keyBits: number, radixBits: number, opts: SortOptions = {}): SortRun {
  const order = opts.order ?? 'lsd';
  const unstablePass = opts.unstablePass ?? null;
  let cur = keys.slice();
  let ids = keys.map((_, i) => i);
  const passes: SortPass[] = [];
  shifts(keyBits, radixBits, order).forEach((shift, p) => {
    const bits = Math.min(radixBits, keyBits - shift);
    const r = partitionPass(cur, shift, bits, unstablePass === p + 1);
    const outIds = r.src.map((i) => ids[i]);
    passes.push({ ...r, index: p + 1, inIds: ids, outIds, sortedBy: sortedLow(r.output, keyBits) });
    cur = r.output;
    ids = outIds;
  });
  const fd = firstDisorder(cur);
  let firstTieSwap = -1;
  for (let j = 0; j + 1 < cur.length && firstTieSwap < 0; j++) if (cur[j] === cur[j + 1] && ids[j] > ids[j + 1]) firstTieSwap = j;
  return {
    keys: keys.slice(), keyBits, radixBits, order, unstablePass, passes, result: cur, sorted: fd < 0, firstDisorder: fd,
    resultIds: ids, stable: firstTieSwap < 0, firstTieSwap,
  };
}

/** In a pass, the key the stepper asks about: the middle key of the highest bucket. */
export function promptKey(p: PassResult): number {
  const R = p.counts.length;
  let top = R - 1;
  while (top > 0 && p.counts[top] === 0) top--;
  const want = p.starts[top] + Math.floor(p.counts[top] / 2);
  const i = p.dst.findIndex((d) => d === want);
  return i < 0 ? 0 : i;
}

/** The bit-reversed value of a key: what four 1-bit passes from the top sort by. */
export function reverseBits(k: number, bits: number): number {
  let r = 0;
  for (let b = 0; b < bits; b++) if ((k >>> b) & 1) r |= 1 << (bits - 1 - b);
  return r;
}

/** Deterministic pseudo-random keys (a small LCG), so a "shuffle" can be reproduced. */
export function randomKeys(n: number, bits: number, seed: number): number[] {
  let s = seed >>> 0 || 1;
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    out.push((s >>> 8) & ((1 << bits) - 1));
  }
  return out;
}

/** Parse up to `count` keys of `bits` bits, written in binary ("0101") or decimal ("5"). */
export function parseKeys(text: string, bits: number, count: number): { keys: number[] } | { error: string } {
  const parts = text.trim().split(/[\s,]+/).filter(Boolean);
  if (parts.length !== count) return { error: tr(`type exactly ${count} keys (you typed ${parts.length})`, `hãy nhập đúng ${count} khóa (bạn đã nhập ${parts.length})`) };
  const max = (1 << bits) - 1;
  const keys: number[] = [];
  for (const p of parts) {
    let v: number;
    if (/^[01]+$/.test(p) && p.length === bits) v = parseInt(p, 2);
    else if (/^\d+$/.test(p)) v = Number(p);
    else return { error: tr(`"${p}" is not a ${bits}-bit key`, `"${p}" không phải là khóa ${bits} bit`) };
    if (v > max) return { error: tr(`${p} does not fit in ${bits} bits`, `${p} không vừa trong ${bits} bit`) };
    keys.push(v);
  }
  return { keys };
}

// ---------------------------------------------------------------- one parallel pass (14.5)
export type Defect = 'none' | 'zero-formula' | 'scan-n' | 'last-as-total' | 'inclusive';
export const DEFECT_LABEL: Record<Defect, string> = {
  get none() {
    return tr('as corrected', 'bản đã sửa');
  },
  get 'zero-formula'() {
    return tr('ones use the zero formula', 'phần tử có bit 1 dùng công thức của bit 0');
  },
  get 'scan-n'() {
    return tr('scan of n entries', 'scan trên n phần tử');
  },
  get 'last-as-total'() {
    return tr('total from the last of n entries', 'lấy tổng từ phần tử cuối trong n phần tử');
  },
  get inclusive() {
    return tr('inclusive instead of exclusive scan', 'scan bao gồm thay vì loại trừ');
  },
};

export interface ThreadRow {
  block: number;
  thread: number;
  i: number;
  key: number;
  bit: number;
  /** What the thread reads from scan[i]: ones before it (or whatever the defect gives). */
  onesBefore: number;
  /** What the thread takes as the total; null when its read is out of range. */
  total: number | null;
  /** Destination; null when it depends on an out-of-range read. */
  dst: number | null;
  /** The formula with this thread's numbers. */
  formula: string;
  /** The total was read from past the end of the scan array. */
  oobRead: boolean;
  /** The destination is outside [0, n). */
  oobWrite: boolean;
}
export interface ParallelPass {
  n: number;
  bit: number;
  blockDim: number;
  defect: Defect;
  keys: number[];
  bits: number[];
  /** The scan array after the scan (n + 1 entries, or n for 'scan-n' and 'last-as-total'). */
  scan: number[];
  /** The true number of ones. */
  ones: number;
  threads: ThreadRow[];
  /** Output slots 0..n-1: the key written there when exactly one thread wrote it, else null. */
  out: (number | null)[];
  /** Which threads wrote each slot. */
  writers: number[][];
  collisions: number[];
  holes: number[];
  oobReads: number[];
  oobWrites: number[];
  /** Threads whose destination is unknown (it depends on an out-of-range read). */
  unknown: number[];
  /** The output is the stable partition and nothing went out of range. */
  correct: boolean;
  /** What a correct pass produces. */
  expected: number[];
}

export function exclusiveScan(xs: number[]): number[] {
  let s = 0;
  return xs.map((x) => {
    const r = s;
    s += x;
    return r;
  });
}
export function inclusiveScan(xs: number[]): number[] {
  let s = 0;
  return xs.map((x) => (s += x));
}

/** One 1-bit pass as Fig. 14.7 computes it, one thread per key, with an optional defect. */
export function parallelPass(keys: number[], bit: number, blockDim = 4, defect: Defect = 'none'): ParallelPass {
  const n = keys.length;
  const bits = keys.map((k) => digit(k, bit, 1));
  const ones = bits.reduce((a, b) => a + b, 0);
  const shortScan = defect === 'scan-n' || defect === 'last-as-total';
  const scan = shortScan ? exclusiveScan(bits) : defect === 'inclusive' ? inclusiveScan([...bits, 0]) : exclusiveScan([...bits, 0]);
  const totalRead: number | null = defect === 'scan-n' ? null : defect === 'last-as-total' ? scan[n - 1] : scan[n];
  const threads: ThreadRow[] = keys.map((key, i) => {
    const b = bits[i];
    const before = scan[i];
    let dst: number | null;
    let formula: string;
    if (b === 0 || defect === 'zero-formula') {
      dst = i - before;
      formula = `${i} − ${before} = ${dst}`;
    } else if (totalRead == null) {
      dst = null;
      formula = `${n} − ? + ${before} = ?`;
    } else {
      dst = n - totalRead + before;
      formula = `${n} − ${totalRead} + ${before} = ${dst}`;
    }
    return {
      block: Math.floor(i / blockDim),
      thread: i % blockDim,
      i,
      key,
      bit: b,
      onesBefore: before,
      total: totalRead,
      dst,
      formula,
      oobRead: defect === 'scan-n',
      oobWrite: dst != null && (dst < 0 || dst >= n),
    };
  });
  const writers: number[][] = Array.from({ length: n }, () => []);
  for (const t of threads) if (t.dst != null && !t.oobWrite) writers[t.dst].push(t.i);
  const out = writers.map((w) => (w.length === 1 ? keys[w[0]] : null));
  const collisions = writers.flatMap((w, s) => (w.length > 1 ? [s] : []));
  const holes = writers.flatMap((w, s) => (w.length === 0 ? [s] : []));
  const oobReads = threads.filter((t) => t.oobRead).map((t) => t.i);
  const oobWrites = threads.filter((t) => t.oobWrite).map((t) => t.i);
  const unknown = threads.filter((t) => t.dst == null).map((t) => t.i);
  const expected = partitionPass(keys, bit, 1).output;
  const correct =
    collisions.length === 0 && holes.length === 0 && oobReads.length === 0 && oobWrites.length === 0 && out.every((k, s) => k === expected[s]);
  return { n, bit, blockDim, defect, keys: keys.slice(), bits, scan, ones, threads, out, writers, collisions, holes, oobReads, oobWrites, unknown, correct, expected };
}

/** How many contiguous runs of output addresses a group of consecutive threads writes. */
export function writeRuns(dsts: number[]): number {
  let runs = 0;
  const sorted = [...dsts].sort((a, b) => a - b);
  for (let k = 0; k < sorted.length; k++) if (k === 0 || sorted[k] !== sorted[k - 1] + 1) runs++;
  return runs;
}
