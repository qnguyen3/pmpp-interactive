// A warp-level model of the basic merge kernel (Fig. 13.9, p. 313) for chapter 13,
// section 13.3: who reads and writes what at each step of merge_sequential, how
// many memory segments one warp's accesses touch, and what the co-rank searches
// cost. Built on corank.ts (the algorithm) and on chapter 6's coalesce.ts
// (warpAccess and the vendors' segment sizes), which it uses read-only.
//
// What it models:
//   - The launch: `threads` threads in a 1D grid; line 03 computes the outputs per
//     thread either as printed (an integer quotient, so `ceil` does nothing:
//     ch13-facts A4) or corrected ((total + threads - 1) / threads).
//   - Each thread's two co_rank calls on global memory, with the reads the code
//     makes (from coRankTrace, short-circuiting respected).
//   - Each thread's merge_sequential, one output per step: in the main loop a
//     thread loads A[i] and B[j] and stores one of them; in the tail it loads and
//     stores one element. Step s of every thread is lined up, as if the warp ran
//     step s together.
//
// What it leaves out, and what the lesson says wherever it is shown:
//   - One instruction at a time, no caches: a segment fetched at step s is not
//     remembered at step s + 1, although each thread walks its own slice in order
//     and a real L1 or L2 may serve later steps from lines fetched earlier.
//   - Lanes in different loops (main, tail of A, tail of B) execute at different
//     times (divergence); the main loop's A-or-B choice may compile to a select
//     (ours, unchecked). The model idealises the warp as doing step s together and
//     counts the paths.
//   - No timing, no occupancy, no latency hiding. Arrays start on a segment boundary.

import { tr } from '../lib/locale.ts';
import { coRankTrace, mergeStable, rng, type Read, type Tagged } from './corank';
import { warpAccess, type WarpAccess } from './coalesce';

export type Formula = 'printed' | 'corrected';

/** Line 03 of Fig. 13.9: outputs per thread. As printed, the division is an integer one and `ceil` changes nothing. */
export function perThreadCount(total: number, threads: number, formula: Formula): number {
  if (!(threads > 0)) throw new RangeError('threads must be positive');
  return formula === 'printed' ? Math.floor(total / threads) : Math.floor((total + threads - 1) / threads);
}

/** Threads a launch needs for a chosen number of outputs per thread. */
export const threadsFor = (total: number, perThread: number) => Math.max(1, Math.ceil(total / perThread));

export type Path = 'A' | 'B' | 'tailA' | 'tailB' | 'idle';

/** One step of one thread's merge_sequential. */
export interface Op {
  path: Exclude<Path, 'idle'>;
  /** Global loads, in program order: A[i] and B[j] in the main loop, one element in a tail loop. */
  reads: Read[];
  /** The element of C stored. */
  write: number;
  /** The element stored, with its source. */
  value: Tagged;
}

export interface ThreadWork {
  t: number;
  kCurr: number;
  kNext: number;
  iCurr: number;
  iNext: number;
  jCurr: number;
  jNext: number;
  /** Global reads made by this thread's two co_rank calls. */
  searchReads: number;
  /** Its merge_sequential, one entry per output. */
  ops: Op[];
}

export interface BasicLayout {
  A: readonly number[];
  B: readonly number[];
  total: number;
  threads: number;
  formula: Formula;
  perThread: number;
  work: ThreadWork[];
  /** Outputs no thread writes. */
  unwritten: number[];
  /** C as the threads leave it (null where nothing was written). */
  C: (Tagged | null)[];
}

/** One thread's merge_sequential (Fig. 13.2) on its slices, step by step. */
function mergeOps(A: readonly number[], B: readonly number[], iCurr: number, iNext: number, jCurr: number, jNext: number, kCurr: number): Op[] {
  const m = iNext - iCurr;
  const n = jNext - jCurr;
  const ops: Op[] = [];
  let i = 0;
  let j = 0;
  let k = kCurr;
  while (i < m && j < n) {
    const ra: Read = { arr: 'A', index: iCurr + i };
    const rb: Read = { arr: 'B', index: jCurr + j };
    if (A[iCurr + i] <= B[jCurr + j]) {
      ops.push({ path: 'A', reads: [ra, rb], write: k++, value: { key: A[iCurr + i], src: 'A', idx: iCurr + i } });
      i++;
    } else {
      ops.push({ path: 'B', reads: [ra, rb], write: k++, value: { key: B[jCurr + j], src: 'B', idx: jCurr + j } });
      j++;
    }
  }
  if (i === m) {
    for (; j < n; j++) ops.push({ path: 'tailB', reads: [{ arr: 'B', index: jCurr + j }], write: k++, value: { key: B[jCurr + j], src: 'B', idx: jCurr + j } });
  } else {
    for (; i < m; i++) ops.push({ path: 'tailA', reads: [{ arr: 'A', index: iCurr + i }], write: k++, value: { key: A[iCurr + i], src: 'A', idx: iCurr + i } });
  }
  return ops;
}

/** The basic kernel on inputs A and B with `threads` threads. */
export function basicLayout(A: readonly number[], B: readonly number[], threads: number, formula: Formula = 'corrected'): BasicLayout {
  const total = A.length + B.length;
  const perThread = perThreadCount(total, threads, formula);
  const work: ThreadWork[] = [];
  const C: (Tagged | null)[] = Array.from({ length: total }, () => null);
  for (let t = 0; t < threads; t++) {
    // Our corrected kernel clamps both; as printed, k_curr never passes the end anyway.
    const kCurr = Math.min(t * perThread, total);
    const kNext = Math.min((t + 1) * perThread, total);
    const a = coRankTrace(kCurr, A, B);
    const b = coRankTrace(kNext, A, B);
    const reads = (x: typeof a) => x.steps.reduce((s, st) => s + st.reads.length, 0);
    const iCurr = a.result;
    const iNext = b.result;
    const ops = mergeOps(A, B, iCurr, iNext, kCurr - iCurr, kNext - iNext, kCurr);
    for (const op of ops) C[op.write] = op.value;
    work.push({ t, kCurr, kNext, iCurr, iNext, jCurr: kCurr - iCurr, jNext: kNext - iNext, searchReads: reads(a) + reads(b), ops });
  }
  const unwritten = C.flatMap((e, p) => (e == null ? [p] : []));
  return { A, B, total, threads, formula, perThread, work, unwritten, C };
}

export interface LaneStep {
  lane: number;
  /** Thread index, or -1 for a lane past the last thread. */
  t: number;
  path: Path;
  reads: Read[];
  write: number | null;
}

export interface WarpStep {
  lanes: LaneStep[];
  /** One warp's loads of A at this step, its loads of B, and its stores to C. */
  readsA: WarpAccess;
  readsB: WarpAccess;
  writes: WarpAccess;
  /** Distinct paths among the lanes that have work. */
  paths: Exclude<Path, 'idle'>[];
  /** Lanes with work at this step. */
  active: number;
}

/** What one warp's lanes do at step s of merge_sequential. */
export function warpStep(layout: BasicLayout, warp: number, warpWidth: number, s: number, elemBytes = 4, segmentBytes = 32): WarpStep {
  const lanes: LaneStep[] = [];
  for (let l = 0; l < warpWidth; l++) {
    const t = warp * warpWidth + l;
    const op = t < layout.threads ? layout.work[t].ops[s] : undefined;
    lanes.push(op ? { lane: l, t, path: op.path, reads: op.reads, write: op.write } : { lane: l, t: t < layout.threads ? t : -1, path: 'idle', reads: [], write: null });
  }
  const bytesOf = (arr: 'A' | 'B') => lanes.flatMap((x) => x.reads.filter((r) => r.arr === arr).map((r) => r.index * elemBytes));
  const order: Exclude<Path, 'idle'>[] = ['A', 'B', 'tailA', 'tailB'];
  const seen = new Set(lanes.map((x) => x.path));
  return {
    lanes,
    readsA: warpAccess(bytesOf('A'), elemBytes, segmentBytes),
    readsB: warpAccess(bytesOf('B'), elemBytes, segmentBytes),
    writes: warpAccess(lanes.flatMap((x) => (x.write == null ? [] : [x.write * elemBytes])), elemBytes, segmentBytes),
    paths: order.filter((p) => seen.has(p)),
    active: lanes.filter((x) => x.path !== 'idle').length,
  };
}

export interface SearchCost {
  /** co_rank calls: two per thread. */
  calls: number;
  /** Global reads by all co_rank calls, counted from their traces. */
  searchReads: number;
  /** Global loads by all merge_sequential calls (two per main-loop step, one per tail step). */
  mergeReads: number;
  outputs: number;
  searchPerOutput: number;
  mergePerOutput: number;
}

export function searchCost(layout: BasicLayout): SearchCost {
  const searchReads = layout.work.reduce((s, w) => s + w.searchReads, 0);
  const mergeReads = layout.work.reduce((s, w) => s + w.ops.reduce((x, o) => x + o.reads.length, 0), 0);
  const outputs = layout.total - layout.unwritten.length;
  return {
    calls: 2 * layout.threads,
    searchReads,
    mergeReads,
    outputs,
    searchPerOutput: outputs ? searchReads / outputs : 0,
    mergePerOutput: outputs ? mergeReads / outputs : 0,
  };
}

/** Does the layout write the stable merge of A and B? */
export function layoutCorrect(layout: BasicLayout): boolean {
  const ref = mergeStable(layout.A, layout.B);
  return layout.C.every((e, p) => e != null && e.key === ref[p].key && e.src === ref[p].src && e.idx === ref[p].idx);
}

// ---------------------------------------------------------------- data

export type DataId = 'distinct' | 'duplicates' | 'a-before-b' | 'interleaved';

export const DATA: { id: DataId; label: string }[] = [
  {
    id: 'distinct',
    get label() {
      return tr('Distinct keys, random interleaving', 'Khóa khác nhau, xen kẽ ngẫu nhiên');
    },
  },
  {
    id: 'duplicates',
    get label() {
      return tr('Many duplicates (keys 0 to 7)', 'Nhiều khóa trùng (khóa 0 đến 7)');
    },
  },
  {
    id: 'a-before-b',
    get label() {
      return tr('All of A before all of B', 'Toàn bộ A trước toàn bộ B');
    },
  },
  {
    id: 'interleaved',
    get label() {
      return tr('Perfectly interleaved (A even, B odd)', 'Xen kẽ hoàn hảo (A chẵn, B lẻ)');
    },
  },
];

/** Two sorted inputs with m = floor(total / 2) and n = total - m. */
export function makeData(id: DataId, total: number, seed = 1): { A: number[]; B: number[] } {
  const m = Math.floor(total / 2);
  const n = total - m;
  if (id === 'a-before-b') return { A: Array.from({ length: m }, (_, q) => q), B: Array.from({ length: n }, (_, q) => m + q) };
  if (id === 'interleaved') return { A: Array.from({ length: m }, (_, q) => 2 * q), B: Array.from({ length: n }, (_, q) => 2 * q + 1) };
  const r = rng(seed);
  if (id === 'duplicates') {
    const draw = (len: number) => Array.from({ length: len }, () => Math.floor(r() * 8)).sort((x, y) => x - y);
    return { A: draw(m), B: draw(n) };
  }
  // distinct: the keys 0..total-1, each sent to A or B at random (m of them to A)
  const keys = Array.from({ length: total }, (_, q) => q);
  for (let q = total - 1; q > 0; q--) {
    const p = Math.floor(r() * (q + 1));
    [keys[q], keys[p]] = [keys[p], keys[q]];
  }
  return { A: keys.slice(0, m).sort((x, y) => x - y), B: keys.slice(m).sort((x, y) => x - y) };
}
