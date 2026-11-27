// The book's co-rank function (Fig. 13.5, p. 310) and the stable merge it serves
// (Fig. 13.2, p. 305), small enough to draw. Chapter 13, section 13.3.
//
// What it models, exactly: the integer logic of Fig. 13.5 on small sorted int
// arrays, including the order in which the bounds tests short-circuit (so the
// reads it reports are the reads the code makes), and the merge of Fig. 13.2 with
// its `A[i] <= B[j]` tie rule. Every element carries a tag (which array, which
// index), so a merge that keeps keys sorted but reorders equal keys can be seen.
//
// What it leaves out: memory, timing, warps. It is the algorithm, not the machine;
// app/src/sim/mergeaccess.ts adds the warp-level access picture on top of it.
//
// The `swapped` tie rule exchanges `>` and `>=` in the two tests (the bug in the
// section's tie-rule question). It still converges, to the co-rank that sends
// ties to B, so keys stay sorted while stability is lost.

import { tr } from '../lib/locale.ts';

export type Src = 'A' | 'B';

/** One element of a merge, tagged with where it came from. */
export interface Tagged {
  key: number;
  src: Src;
  /** Index in its own input array. */
  idx: number;
}

export type TieRule = 'book' | 'swapped';
export type Verdict = 'i-too-high' | 'j-too-high' | 'done';

export interface Read {
  arr: Src;
  index: number;
}

/** One of the two tests of Fig. 13.5, as evaluated. */
export interface TestEval {
  /** The bounds part (i > 0 && j < n, or j > 0 && i < m) held, so the comparison ran. */
  inBounds: boolean;
  /** The two values compared (A[i-1] and B[j]; or B[j-1] and A[i]) when the comparison ran. */
  left?: number;
  right?: number;
  /** The whole condition. */
  holds: boolean;
}

export interface CoRankState {
  i: number;
  j: number;
  iLow: number;
  jLow: number;
}

/** One trip round the loop: the state when the tests ran, what they found, and the move made. */
export interface CoRankStep extends CoRankState {
  test: Verdict;
  /** Elements read by the tests, in program order (short-circuiting respected). */
  reads: Read[];
  /** Test 1 ("i too high"), always evaluated. */
  t1: TestEval;
  /** Test 2 ("j too high"), evaluated only when test 1 is false (else-if). */
  t2: TestEval | null;
  /** How far i and j moved, when they moved. */
  delta?: number;
  /** Line of the co_rank snippet (labs/ch13/snippets/merge_basic.cu, numbered as Fig. 13.5) that decided the step. */
  line: number;
}

export interface CoRankTrace {
  k: number;
  rule: TieRule;
  init: CoRankState;
  steps: CoRankStep[];
  /** The co-rank i; j is k - i. */
  result: number;
  /** Times i moved (steps that were not "done"). */
  moves: number;
  /** Candidates for i at the start: w + 1, with w = min(k, m) - max(0, k - n). */
  candidates: number;
  /** Our bound on moves, ceil(log2(w + 1)): the bit length of w. */
  bound: number;
}

/** Lines of the snippet's co_rank that each verdict corresponds to (book numbering, Fig. 13.5). */
export const CORANK_LINES: Record<Verdict | 'init', number[]> = {
  init: [2, 3, 4, 5],
  'i-too-high': [9, 10, 11, 12, 13],
  'j-too-high': [14, 15, 16, 17, 18],
  done: [19, 20, 23],
};

/** The stable merge of Fig. 13.2: on equal keys the element of A goes first. */
export function mergeStable(A: readonly number[], B: readonly number[], aBase = 0, bBase = 0): Tagged[] {
  const out: Tagged[] = [];
  let i = 0;
  let j = 0;
  while (i < A.length && j < B.length) {
    if (A[i] <= B[j]) {
      out.push({ key: A[i], src: 'A', idx: aBase + i });
      i++;
    } else {
      out.push({ key: B[j], src: 'B', idx: bBase + j });
      j++;
    }
  }
  for (; i < A.length; i++) out.push({ key: A[i], src: 'A', idx: aBase + i });
  for (; j < B.length; j++) out.push({ key: B[j], src: 'B', idx: bBase + j });
  return out;
}

/** Both tests at (i, j), in the order and with the short-circuiting of Fig. 13.5. */
export function evaluate(
  i: number,
  j: number,
  A: readonly number[],
  B: readonly number[],
  rule: TieRule = 'book',
): { verdict: Verdict; t1: TestEval; t2: TestEval | null; reads: Read[] } {
  const m = A.length;
  const n = B.length;
  const reads: Read[] = [];
  // Test 1: i > 0 && j < n && A[i-1] > B[j]   (swapped: >=)
  const t1: TestEval = { inBounds: i > 0 && j < n, holds: false };
  if (t1.inBounds) {
    reads.push({ arr: 'A', index: i - 1 }, { arr: 'B', index: j });
    t1.left = A[i - 1];
    t1.right = B[j];
    t1.holds = rule === 'book' ? A[i - 1] > B[j] : A[i - 1] >= B[j];
  }
  if (t1.holds) return { verdict: 'i-too-high', t1, t2: null, reads }; // i18n-ignore: verdict id; the chapter shows its own text
  // Test 2: j > 0 && i < m && B[j-1] >= A[i]   (swapped: >)
  const t2: TestEval = { inBounds: j > 0 && i < m, holds: false };
  if (t2.inBounds) {
    reads.push({ arr: 'B', index: j - 1 }, { arr: 'A', index: i });
    t2.left = B[j - 1];
    t2.right = A[i];
    t2.holds = rule === 'book' ? B[j - 1] >= A[i] : B[j - 1] > A[i];
  }
  return { verdict: t2.holds ? 'j-too-high' : 'done', t1, t2, reads };
}

/** ceil(log2(w + 1)) for an integer w >= 0, computed exactly: the bit length of w. */
export const bitLength = (w: number) => (w <= 0 ? 0 : 32 - Math.clz32(w));

/** Fig. 13.5, one trip round the loop at a time. */
export function coRankTrace(k: number, A: readonly number[], B: readonly number[], rule: TieRule = 'book'): CoRankTrace {
  const m = A.length;
  const n = B.length;
  if (!Number.isInteger(k) || k < 0 || k > m + n) throw new RangeError(`k = ${k} is outside 0..${m + n}`);
  let i = Math.min(k, m);
  let j = k - i;
  let iLow = Math.max(0, k - n);
  let jLow = Math.max(0, k - m);
  const init = { i, j, iLow, jLow };
  const steps: CoRankStep[] = [];
  for (;;) {
    const ev = evaluate(i, j, A, B, rule);
    const step: CoRankStep = { i, j, iLow, jLow, test: ev.verdict, reads: ev.reads, t1: ev.t1, t2: ev.t2, line: 0 };
    if (ev.verdict === 'i-too-high') {
      const delta = (i - iLow + 1) >> 1;
      if (delta === 0) throw new Error('co_rank made no progress: are both inputs sorted?');
      step.delta = delta;
      step.line = 9;
      jLow = j;
      j += delta;
      i -= delta;
    } else if (ev.verdict === 'j-too-high') {
      const delta = (j - jLow + 1) >> 1;
      if (delta === 0) throw new Error('co_rank made no progress: are both inputs sorted?');
      step.delta = delta;
      step.line = 14;
      iLow = i;
      i += delta;
      j -= delta;
    } else {
      step.line = 20;
    }
    steps.push(step);
    if (ev.verdict === 'done') break;
    if (steps.length > 64) throw new Error('co_rank did not converge');
  }
  const w = Math.min(k, m) - Math.max(0, k - n);
  return { k, rule, init, steps, result: i, moves: steps.length - 1, candidates: w + 1, bound: bitLength(w) };
}

/** The co-rank of k in A: how many of the first k merged outputs come from A. */
export const coRank = (k: number, A: readonly number[], B: readonly number[], rule: TieRule = 'book') =>
  coRankTrace(k, A, B, rule).result;

/**
 * The candidates for i still possible after the tests of steps 0..upto (inclusive)
 * have run: a failed "i too high" test at i rules out i and everything above, a
 * failed "j too high" test rules out i and everything below. Drawn by the scene as
 * the window that shrinks.
 */
export function knownRange(trace: CoRankTrace, upto: number): [number, number] {
  let lo = trace.init.iLow;
  let hi = trace.init.i;
  for (let s = 0; s <= upto && s < trace.steps.length; s++) {
    const st = trace.steps[s];
    if (st.test === 'i-too-high') hi = Math.min(hi, st.i - 1);
    else if (st.test === 'j-too-high') lo = Math.max(lo, st.i + 1);
    else lo = hi = st.i;
  }
  return [lo, hi];
}

/** What happens if two threads split at output k with i forced (Fig. 13.4, p. 309). */
export interface Split {
  k: number;
  i: number;
  j: number;
  /** Both tests pass: i is the co-rank. */
  valid: boolean;
  /** The test that fires, which says which way to move. */
  failing: 'i-too-high' | 'j-too-high' | null;
  t1: TestEval;
  t2: TestEval | null;
  /** What the first thread writes, C[0..k): the merge of A[0..i) and B[0..j). */
  prefix: Tagged[];
  /** What the second thread writes, C[k..m+n): the merge of the rest. */
  suffix: Tagged[];
  /** prefix + suffix has its keys in order. */
  sortedKeys: boolean;
  /** prefix + suffix equals the stable merge, tags included. */
  stable: boolean;
}

/** The feasible range of i for output position k: max(0, k - n) to min(k, m). */
export const iRange = (k: number, m: number, n: number): [number, number] => [Math.max(0, k - n), Math.min(k, m)];

export const sameMerge = (x: readonly (Tagged | null)[], y: readonly Tagged[]) =>
  x.length === y.length && x.every((e, p) => e != null && e.key === y[p].key && e.src === y[p].src && e.idx === y[p].idx);

export function splitAt(k: number, i: number, A: readonly number[], B: readonly number[]): Split {
  const m = A.length;
  const n = B.length;
  const [lo, hi] = iRange(k, m, n);
  if (!Number.isInteger(i) || i < lo || i > hi) throw new RangeError(`i = ${i} is outside ${lo}..${hi} for k = ${k}`);
  const j = k - i;
  const ev = evaluate(i, j, A, B, 'book');
  const prefix = mergeStable(A.slice(0, i), B.slice(0, j));
  const suffix = mergeStable(A.slice(i), B.slice(j), i, j);
  const all = [...prefix, ...suffix];
  return {
    k, i, j,
    valid: ev.verdict === 'done',
    failing: ev.verdict === 'done' ? null : ev.verdict,
    t1: ev.t1,
    t2: ev.t2,
    prefix,
    suffix,
    sortedKeys: all.every((e, p) => p === 0 || all[p - 1].key <= e.key),
    stable: sameMerge(all, mergeStable(A, B)),
  };
}

/** One thread's share of the merge: outputs [kCurr, kNext), inputs A[iCurr, iNext) and B[jCurr, jNext). */
export interface Slice {
  t: number;
  kCurr: number;
  kNext: number;
  iCurr: number;
  iNext: number;
  jCurr: number;
  jNext: number;
}

/**
 * The basic kernel's partition (Fig. 13.9): thread t owns outputs from
 * t·perThread to (t+1)·perThread, both clamped to m + n (the book clamps only
 * k_next; clamping k_curr is ours), and finds its inputs with two co-ranks.
 */
export function partition(A: readonly number[], B: readonly number[], threads: number, perThread: number, rule: TieRule = 'book'): Slice[] {
  const total = A.length + B.length;
  const out: Slice[] = [];
  for (let t = 0; t < threads; t++) {
    const kCurr = Math.min(t * perThread, total);
    const kNext = Math.min((t + 1) * perThread, total);
    const iCurr = coRank(kCurr, A, B, rule);
    const iNext = coRank(kNext, A, B, rule);
    out.push({ t, kCurr, kNext, iCurr, iNext, jCurr: kCurr - iCurr, jNext: kNext - iNext });
  }
  return out;
}

/**
 * C as the threads write it: each thread merges its own two input slices with
 * Fig. 13.2's `<=` into its own output slice. Positions no thread writes are null.
 */
export function mergeByThreads(A: readonly number[], B: readonly number[], slices: readonly Slice[]): (Tagged | null)[] {
  const C: (Tagged | null)[] = Array.from({ length: A.length + B.length }, () => null);
  for (const s of slices) {
    const part = mergeStable(A.slice(s.iCurr, s.iNext), B.slice(s.jCurr, s.jNext), s.iCurr, s.jCurr);
    part.forEach((e, p) => {
      C[s.kCurr + p] = e;
    });
  }
  return C;
}

// ---------------------------------------------------------------- presets

/** A small deterministic generator (mulberry32), so a seed reproduces a preset. */
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

/** n sorted integers drawn from lo..hi. */
export function sortedRandom(r: () => number, n: number, lo: number, hi: number): number[] {
  return Array.from({ length: n }, () => lo + Math.floor(r() * (hi - lo + 1))).sort((x, y) => x - y);
}

export type PresetId = 'running' | 'duplicates' | 'a-below' | 'b-below' | 'equal' | 'a-empty' | 'random';

export const PRESETS: { id: PresetId; label: string; make: (seed: number) => { A: number[]; B: number[] } }[] = [
  {
    id: 'running',
    get label() {
      return tr('The running example', 'Ví dụ xuyên suốt');
    },
    make: () => ({ A: [1, 7, 8, 9, 10], B: [7, 10, 10, 12] }),
  },
  {
    id: 'duplicates',
    get label() {
      return tr('Many duplicates (keys 0 to 3)', 'Nhiều khóa trùng (khóa 0 đến 3)');
    },
    make: (seed) => {
      const r = rng(seed);
      return { A: sortedRandom(r, 8, 0, 3), B: sortedRandom(r, 7, 0, 3) };
    },
  },
  {
    id: 'a-below',
    get label() {
      return tr('All of A below B', 'Toàn bộ A nhỏ hơn B');
    },
    make: () => ({ A: [1, 2, 3, 4, 5], B: [6, 7, 8, 9] }),
  },
  {
    id: 'b-below',
    get label() {
      return tr('All of B below A', 'Toàn bộ B nhỏ hơn A');
    },
    make: () => ({ A: [6, 7, 8, 9, 10], B: [1, 2, 3, 4] }),
  },
  {
    id: 'equal',
    get label() {
      return tr('All keys equal', 'Mọi khóa bằng nhau');
    },
    make: () => ({ A: [5, 5, 5, 5, 5], B: [5, 5, 5, 5] }),
  },
  {
    id: 'a-empty',
    get label() {
      return tr('A empty', 'A rỗng');
    },
    make: () => ({ A: [], B: [2, 4, 6, 8, 10] }),
  },
  {
    id: 'random',
    get label() {
      return tr('Random', 'Ngẫu nhiên');
    },
    make: (seed) => {
      const r = rng(seed);
      const m = 3 + Math.floor(r() * 8);
      const n = 3 + Math.floor(r() * 8);
      return { A: sortedRandom(r, m, 0, 20), B: sortedRandom(r, n, 0, 20) };
    },
  },
];

export const MAX_INPUT = 16;

/** Parse "1, 7, 8" into a sorted int array, or say why not. */
export function parseSorted(text: string, max = MAX_INPUT): { ok: true; values: number[] } | { ok: false; error: string } {
  const t = text.trim();
  if (t === '') return { ok: true, values: [] };
  const parts = t.split(/[\s,]+/).filter((s) => s !== '');
  if (parts.length > max) return { ok: false, error: tr(`at most ${max} numbers`, `tối đa ${max} số`) };
  const values: number[] = [];
  for (const p of parts) {
    if (!/^-?\d+$/.test(p)) return { ok: false, error: tr(`"${p}" is not a whole number`, `"${p}" không phải số nguyên`) };
    const v = Number(p);
    if (Math.abs(v) > 999) return { ok: false, error: tr('keep numbers between -999 and 999', 'giữ các số trong khoảng -999 đến 999') };
    values.push(v);
  }
  for (let q = 1; q < values.length; q++)
    if (values[q - 1] > values[q]) return { ok: false, error: tr(`not sorted: ${values[q - 1]} comes before ${values[q]}`, `chưa sắp xếp: ${values[q - 1]} đứng trước ${values[q]}`) };
  return { ok: true, values };
}
