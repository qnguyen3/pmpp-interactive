// Top-k by MSD radix select, for module X1, section X1.4 (ours: the book describes no
// selection algorithm; the MSD idea is §14.9, p. 346, and the shared-memory histogram with
// atomics is chapter 9's Fig. 9.10, pp. 213-214). The same algorithm ships as llama.cpp's
// top_k_radix_cuda and Metal kernel_top_k (notes/today/ch14-facts.md D2, ch12-facts D1, D2,
// ch09-facts D4) and as PyTorch's multi-block mbtopk (ch14-facts D5).
//
// What it models, exactly:
//  - Keys: a logit's float32 bits through floatkey.ts's keyAsc (CUB's transform with the
//    -0.0 step, ch14-facts A3, E1). Everything after that is integer bit operations.
//  - One block of `blockThreads` threads owns the row. In each counting pass, thread t reads
//    columns t, t + blockThreads, ... (one "round" per blockThreads columns). A thread whose
//    key's decided bits equal the prefix found so far adds 1 to the bin of the current 8-bit
//    digit (a shared-memory atomic); the others do nothing. The histogram has 256 bins.
//  - The bin walk goes from bin 255 down until the running count reaches the k still needed.
//    That crossing bin's digit extends the prefix; keys in bins above it are in, keys in bins
//    below are out. Only the crossing bin is undecided. When every key of the crossing bin is
//    needed (its count equals the k still needed), the set is decided and the select stops
//    early; otherwise it runs at most four passes, after which the prefix is the k-th key K*.
//  - The gather: one more pass in rounds of blockThreads columns. A key whose decided bits are
//    above the prefix is kept; a key equal to it ("a tie") is kept while the tie budget lasts,
//    lowest columns first; slots come from a ballot and a popcount of the lanes below
//    (chapter 12, Fig. 12.3, pp. 292-293) plus running counts across rounds: the stable filter
//    of §12.5 (pp. 297-298). Output: exactly k (column, logit) pairs in ascending column order.
//  - Warps are a display grouping: `warpWidth` lanes per warp (8, 32 or 64), so a block of 16
//    threads is two warps of 8, or one warp with 16 or 48 lanes that have no thread. The model
//    records which lanes take each branch, not what divergence costs.
//  - Threshold bisection: one counting pass per key bit, 32 in all, each reading the row.
//  - Cost: bytes a design must move per row, counted, not measured. LSD radix sort: four 8-bit
//    passes, each reading the 4-byte key to count and reading and writing the 8-byte
//    (key, index) pair to scatter, taken from radixcost.ts (lab 14.3's sort, which has no
//    bucket table: one block per row). Radix select: four histogram reads and one gather read
//    of each 4-byte key, plus 8 bytes per kept pair. Bisection: 32 counting reads and the same
//    gather.
//
// What it leaves out: time, caches, occupancy, latency, the cost of divergence and of atomics,
// several blocks per row, NaN payloads (a NaN is ordered by its bits, as in CUB), and keys
// wider or narrower than 32 bits.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.
import { bitsOf, fromBits, hex, keyAsc } from './floatkey';
import { keyBytes, pairBytes, passes as radixPasses } from './radixcost';
import { tr } from '../lib/locale.ts';

export { hex };
export const DIGIT_BITS = 8;
export const BINS = 1 << DIGIT_BITS;
export const KEY_BITS = 32;
/** Passes an 8-bit radix select needs at most over a 32-bit key. */
export const MAX_PASSES = KEY_BITS / DIGIT_BITS;

/** The bits decided once the digits above `shift` are fixed: 0 before any pass, all 32 after the last. */
export const maskAbove = (shift: number) => (shift >= 32 ? 0 : (0xffffffff << shift) >>> 0);
export const digitOf = (key: number, shift: number) => ((key >>> 0) >>> shift) & (BINS - 1);
/** Hex of the decided bits only, the rest shown as dots: 0xC040····. */
export function prefixHex(prefix: number, mask: number): string {
  const h = hex(prefix).slice(2);
  const m = hex(mask).slice(2);
  return '0x' + h.split('').map((c, i) => (m[i] === '0' ? '·' : c)).join('');
}

// ---------------------------------------------------------------- bin walk
export interface WalkStep {
  bin: number;
  count: number;
  /** Keys in the bins above this one. */
  above: number;
  /** above + count. */
  running: number;
}
export interface Walk {
  steps: WalkStep[];
  /** The bin where the running count reaches `need`. */
  bin: number;
  /** Keys in bins above it: all of them are in. */
  above: number;
  /** Keys still needed from the crossing bin: need - above. */
  stillNeeded: number;
}

/**
 * Walk the bins from the highest down until the running count reaches `need`. Only nonempty
 * bins are listed. `need` must be at least 1 and at most the histogram's total.
 */
export function walkBins(hist: readonly number[], need: number): Walk {
  const total = hist.reduce((a, b) => a + b, 0);
  if (need < 1 || need > total) throw new Error(`need ${need} outside 1..${total}`);
  const steps: WalkStep[] = [];
  let above = 0;
  for (let b = hist.length - 1; b >= 0; b--) {
    if (!hist[b]) continue;
    const running = above + hist[b];
    steps.push({ bin: b, count: hist[b], above, running });
    if (running >= need) return { steps, bin: b, above, stillNeeded: need - above };
    above = running;
  }
  throw new Error('unreachable');
}

// ---------------------------------------------------------------- warps (display)
export interface WarpView {
  warp: number;
  /** One entry per lane of the display warp: the block thread in that lane, or null. */
  threads: (number | null)[];
}
export function warpsOf(blockThreads: number, warpWidth: number): WarpView[] {
  const n = Math.ceil(blockThreads / warpWidth);
  return Array.from({ length: n }, (_, w) => ({
    warp: w,
    threads: Array.from({ length: warpWidth }, (_, l) => (w * warpWidth + l < blockThreads ? w * warpWidth + l : null)),
  }));
}

// ---------------------------------------------------------------- the trace
export interface CountLane {
  thread: number;
  /** Column read this round, or null past the end of the row. */
  col: number | null;
  /** The key's decided bits equal the prefix: this lane does the atomic. */
  match: boolean;
  /** The bin it adds 1 to, when it matches. */
  digit: number | null;
}
export interface WarpTest {
  warp: number;
  /** Lanes with a thread and a column. */
  present: number;
  /** Lanes that pass the prefix test. */
  active: number;
  /** Some present lanes pass the test and some do not: the warp splits at the branch. */
  diverged: boolean;
}
export interface CountRound {
  round: number;
  lanes: CountLane[];
  warps: WarpTest[];
  /** The histogram after this round's atomics. */
  histAfter: number[];
}
export interface Pass {
  index: number;
  /** Position of this pass's digit: 24, 16, 8, 0. */
  shift: number;
  prefixIn: number;
  maskIn: number;
  /** k still to place among the keys that match prefixIn. */
  needIn: number;
  /** Keys that match prefixIn (they take part in this pass). */
  candidates: number;
  rounds: CountRound[];
  hist: number[];
  walk: Walk;
  bin: number;
  /** Keys in bins above the crossing bin (in). */
  above: number;
  /** Matching keys in bins below it (out). */
  below: number;
  /** Keys in the crossing bin (undecided). */
  inBin: number;
  prefixOut: number;
  maskOut: number;
  /** k still needed from the crossing bin. */
  needOut: number;
  /** Every key of the crossing bin is needed: the set is decided, stop. */
  early: boolean;
  /** Rounds in which some warp split at the prefix test. */
  divergedWarps: number;
}
export type GatherClass = 'above' | 'tie' | 'out' | 'none';
export interface GatherLane {
  thread: number;
  col: number | null;
  cls: GatherClass;
  /** Rank among the ties in column order, for a tie. */
  tieRank: number | null;
  keep: boolean;
  /** Output slot, for a kept key. */
  slot: number | null;
}
export interface GatherWarp {
  warp: number;
  /** Ballot of "is a tie", one entry per lane of the display warp. */
  tieBallot: boolean[];
  /** Ballot of "is kept". */
  keepBallot: boolean[];
  tieCount: number;
  keepCount: number;
  /** Ties and kept keys in the warps before this one, this round. */
  tieOffset: number;
  keepOffset: number;
}
export interface GatherRound {
  round: number;
  lanes: GatherLane[];
  warps: GatherWarp[];
  tiesBefore: number;
  keptBefore: number;
  tiesAfter: number;
  keptAfter: number;
}
export interface OutPair {
  col: number;
  logit: number;
  key: number;
}
export interface SelectTrace {
  n: number;
  k: number;
  blockThreads: number;
  warpWidth: number;
  logits: number[];
  bits: number[];
  keys: number[];
  passes: Pass[];
  /** The decided prefix after the last pass; with mask 0xFFFFFFFF it is K*, the k-th key. */
  prefix: number;
  mask: number;
  /** The select ran all four passes and knows K* itself. */
  exact: boolean;
  /** Keys above the prefix in their decided bits: all kept. */
  greater: number;
  /** Keys equal to the prefix in their decided bits. */
  equal: number;
  /** How many of those are kept (the tie budget). */
  need: number;
  gather: GatherRound[];
  out: OutPair[];
}

export const rounds = (n: number, blockThreads: number) => Math.ceil(n / blockThreads);

/** Runs the select on a row of logits (rounded to float32). k is clamped to 1..n. */
export function selectTrace(logits: readonly number[], kIn: number, blockThreads = 16, warpWidth = 32): SelectTrace {
  const n = logits.length;
  if (n < 1) throw new Error('empty row');
  const k = Math.max(1, Math.min(n, Math.floor(kIn)));
  const bits = logits.map(bitsOf);
  const keys = bits.map(keyAsc);
  const views = warpsOf(blockThreads, warpWidth);
  const R = rounds(n, blockThreads);
  const passes: Pass[] = [];
  let prefix = 0;
  let mask = 0;
  let need = k;
  for (let index = 0; index < MAX_PASSES; index++) {
    const shift = KEY_BITS - DIGIT_BITS * (index + 1);
    const hist = new Array(BINS).fill(0);
    const rs: CountRound[] = [];
    let candidates = 0;
    let divergedWarps = 0;
    for (let r = 0; r < R; r++) {
      const lanes: CountLane[] = [];
      for (let t = 0; t < blockThreads; t++) {
        const c = r * blockThreads + t;
        if (c >= n) {
          lanes.push({ thread: t, col: null, match: false, digit: null });
          continue;
        }
        const match = ((keys[c] & mask) >>> 0) === prefix;
        const digit = match ? digitOf(keys[c], shift) : null;
        if (digit != null) {
          hist[digit]++;
          candidates++;
        }
        lanes.push({ thread: t, col: c, match, digit });
      }
      const warps = views.map((v): WarpTest => {
        const own = v.threads.filter((t): t is number => t != null).map((t) => lanes[t]).filter((l) => l.col != null);
        const active = own.filter((l) => l.match).length;
        return { warp: v.warp, present: own.length, active, diverged: active > 0 && active < own.length };
      });
      divergedWarps += warps.filter((w) => w.diverged).length;
      rs.push({ round: r, lanes, warps, histAfter: hist.slice() });
    }
    const walk = walkBins(hist, need);
    const inBin = hist[walk.bin];
    const below = candidates - walk.above - inBin;
    const prefixOut = (prefix | (walk.bin << shift)) >>> 0;
    const maskOut = maskAbove(shift);
    const early = walk.stillNeeded === inBin && shift > 0;
    passes.push({
      index, shift, prefixIn: prefix, maskIn: mask, needIn: need, candidates, rounds: rs, hist, walk,
      bin: walk.bin, above: walk.above, below, inBin, prefixOut, maskOut, needOut: walk.stillNeeded, early, divergedWarps,
    });
    prefix = prefixOut;
    mask = maskOut;
    need = walk.stillNeeded;
    if (early) break;
  }

  // The gather: a stable filter with a tie budget.
  const cls = (c: number): GatherClass => {
    const m = (keys[c] & mask) >>> 0;
    return m > prefix ? 'above' : m === prefix ? 'tie' : 'out';
  };
  const gather: GatherRound[] = [];
  const out: OutPair[] = [];
  let ties = 0;
  let kept = 0;
  let greater = 0;
  let equal = 0;
  for (let r = 0; r < R; r++) {
    const lanes: GatherLane[] = [];
    for (let t = 0; t < blockThreads; t++) {
      const c = r * blockThreads + t;
      const g = c < n ? cls(c) : 'none';
      if (g === 'above') greater++;
      if (g === 'tie') equal++;
      lanes.push({ thread: t, col: c < n ? c : null, cls: g, tieRank: null, keep: false, slot: null });
    }
    // Ballot 1: which lanes hold a tie; a tie's rank is the ties before it in column order.
    const warps: GatherWarp[] = [];
    let tieOffset = 0;
    for (const v of views) {
      const tieBallot = v.threads.map((t) => t != null && lanes[t].cls === 'tie');
      let below = 0;
      v.threads.forEach((t, l) => {
        if (t != null && tieBallot[l]) {
          const rank = ties + tieOffset + below;
          lanes[t].tieRank = rank;
          below++;
        }
      });
      warps.push({ warp: v.warp, tieBallot, keepBallot: [], tieCount: below, keepCount: 0, tieOffset, keepOffset: 0 });
      tieOffset += below;
    }
    for (const l of lanes) l.keep = l.cls === 'above' || (l.cls === 'tie' && (l.tieRank as number) < need);
    // Ballot 2: which lanes keep their key; a kept key's slot is the kept keys before it.
    let keepOffset = 0;
    for (const w of warps) {
      const v = views[w.warp];
      w.keepBallot = v.threads.map((t) => t != null && lanes[t].keep);
      let below = 0;
      v.threads.forEach((t, l) => {
        if (t != null && w.keepBallot[l]) {
          lanes[t].slot = kept + keepOffset + below;
          below++;
        }
      });
      w.keepCount = below;
      w.keepOffset = keepOffset;
      keepOffset += below;
    }
    for (const l of lanes) if (l.keep) out.push({ col: l.col as number, logit: fromBits(bits[l.col as number]), key: keys[l.col as number] });
    gather.push({ round: r, lanes, warps, tiesBefore: ties, keptBefore: kept, tiesAfter: ties + tieOffset, keptAfter: kept + keepOffset });
    ties += tieOffset;
    kept += keepOffset;
  }
  return {
    n, k, blockThreads, warpWidth, logits: bits.map(fromBits), bits, keys, passes, prefix, mask,
    exact: mask === 0xffffffff, greater, equal, need, gather, out,
  };
}

export type Decided = 'in' | 'out' | 'open';
/**
 * Where a column stands once `passes` passes have run (0: nothing decided): in (its decided
 * bits are above the prefix), out (below), or open (equal: still in the undecided bin).
 */
export function statusAfter(tr: SelectTrace, passes: number, col: number): Decided {
  if (passes <= 0) return 'open';
  const p = tr.passes[Math.min(passes, tr.passes.length) - 1];
  const m = (tr.keys[col] & p.maskOut) >>> 0;
  return m > p.prefixOut ? 'in' : m < p.prefixOut ? 'out' : 'open';
}
export function countStatus(tr: SelectTrace, passes: number): Record<Decided, number> {
  const c = { in: 0, out: 0, open: 0 };
  for (let i = 0; i < tr.n; i++) c[statusAfter(tr, passes, i)]++;
  return c;
}

/** The reference: stable sort by value, largest first, ties by lower column; cut at k; back in column order. */
export function referenceTopK(logits: readonly number[], kIn: number): number[] {
  const n = logits.length;
  const k = Math.max(1, Math.min(n, Math.floor(kIn)));
  const keys = logits.map((v) => keyAsc(bitsOf(v)));
  return keys
    .map((key, col) => ({ key, col }))
    .sort((a, b) => (a.key !== b.key ? b.key - a.key : a.col - b.col))
    .slice(0, k)
    .map((x) => x.col)
    .sort((a, b) => a - b);
}

/** The k-th largest key, by sorting: what the select finds without sorting. */
export function kthKey(logits: readonly number[], kIn: number): number {
  const k = Math.max(1, Math.min(logits.length, Math.floor(kIn)));
  return logits.map((v) => keyAsc(bitsOf(v))).sort((a, b) => b - a)[k - 1];
}

/** transformers' TopKLogitsWarper keeps every token not below the k-th value (x1-facts D4): possibly more than k. */
export function keepTiesTopK(logits: readonly number[], k: number): number[] {
  const kth = [...logits].sort((a, b) => b - a)[k - 1];
  return logits.flatMap((v, i) => (v < kth ? [] : [i]));
}

// ---------------------------------------------------------------- bisection
export interface BisectStep {
  bit: number;
  candidate: number;
  /** Keys at or above the candidate threshold: one read of the whole row. */
  count: number;
  accept: boolean;
  threshold: number;
}
/** Threshold bisection: the largest T with at least k keys at or above it, one bit per pass. */
export function bisectionTrace(logits: readonly number[], kIn: number): { steps: BisectStep[]; threshold: number } {
  const k = Math.max(1, Math.min(logits.length, Math.floor(kIn)));
  const keys = logits.map((v) => keyAsc(bitsOf(v)));
  const steps: BisectStep[] = [];
  let T = 0;
  for (let bit = KEY_BITS - 1; bit >= 0; bit--) {
    const candidate = (T | (2 ** bit)) >>> 0;
    const count = keys.filter((key) => key >= candidate).length;
    const accept = count >= k;
    if (accept) T = candidate;
    steps.push({ bit, candidate, count, accept, threshold: T });
  }
  return { steps, threshold: T };
}

// ---------------------------------------------------------------- cost
export interface Cost {
  id: 'sort' | 'select' | 'bisect';
  label: string;
  /** Passes over the row. */
  passes: number;
  /** Bytes per key over all passes, without the output. */
  perKey: number;
  /** Bytes for the k output pairs (column, logit). */
  outBytes: number;
  /** Bytes for one row. */
  perRow: number;
  /** Bytes for the batch. */
  total: number;
  /** Blocks launched: one per row. */
  blocks: number;
}
/** The output: one 4-byte column and one 4-byte logit per kept token. */
export const OUT_PAIR_BYTES = 8;
/** LSD radix sort of (key, index) pairs, 8-bit digits: lab 14.3's traffic per radixcost.ts. */
export const SORT_PASSES = radixPasses(KEY_BITS, DIGIT_BITS);
export const SORT_PER_KEY = SORT_PASSES * (keyBytes(KEY_BITS) + 2 * pairBytes(KEY_BITS));
/** Four histogram reads and one gather read of the 4-byte key. */
export const SELECT_PER_KEY = (MAX_PASSES + 1) * keyBytes(KEY_BITS);
/** One counting read per key bit and the same gather read. */
export const BISECT_PASSES = KEY_BITS;
export const BISECT_PER_KEY = (BISECT_PASSES + 1) * keyBytes(KEY_BITS);

export function costs(V: number, kIn: number, batch: number): Cost[] {
  const k = Math.max(1, Math.min(V, Math.floor(kIn)));
  const mk = (id: Cost['id'], label: string, passes: number, perKey: number, outBytes: number): Cost => {
    const perRow = perKey * V + outBytes;
    return { id, label, passes, perKey, outBytes, perRow, total: perRow * batch, blocks: batch };
  };
  return [
    mk('sort', tr('LSD radix sort (lab 14.3), then cut at k', 'LSD radix sort (bài thực hành 14.3), rồi cắt tại k'), SORT_PASSES, SORT_PER_KEY, 0),
    mk('select', tr('radix select (lab X1.1)', 'radix select (bài thực hành X1.1)'), MAX_PASSES + 1, SELECT_PER_KEY, k * OUT_PAIR_BYTES),
    mk('bisect', tr('threshold bisection', 'tìm ngưỡng bằng cách chia đôi'), BISECT_PASSES + 1, BISECT_PER_KEY, k * OUT_PAIR_BYTES),
  ];
}
/** Sort bytes over select bytes for one row. */
export const trafficRatio = (V: number, k: number) => {
  const [s, sel] = costs(V, k, 1);
  return s.perRow / sel.perRow;
};

// ---------------------------------------------------------------- measured (lab X1.1)
export const LAB_X11 = 'x1/lab1_topk';
export interface LabRow {
  name: string;
  [k: string]: string | number;
}
export interface Measured {
  rows: number;
  cols: number;
  k: number;
  select: number;
  sort: number;
  speedup: number;
}
const numField = (r: LabRow, names: string[]) => {
  for (const n of names) if (typeof r[n] === 'number' && isFinite(r[n] as number)) return r[n] as number;
  return undefined;
};
/**
 * Pairs lab X1.1's benchmark rows: a row whose name mentions "select" and one whose name
 * mentions "sort" (and not "select"), with the same rows, cols and k, each with seconds.
 */
export function measuredPairs(rows: readonly LabRow[]): Measured[] {
  const shape = (r: LabRow) => ({
    rows: numField(r, ['rows', 'batch']),
    cols: numField(r, ['cols', 'vocab', 'V']),
    k: numField(r, ['k', 'K', 'topk']),
    s: numField(r, ['seconds', 'time_s', 's']),
  });
  const sel = rows.filter((r) => /select/i.test(r.name));
  const srt = rows.filter((r) => /sort/i.test(r.name) && !/select/i.test(r.name) && !/cpu/i.test(r.name));
  const out: Measured[] = [];
  for (const a of sel) {
    const A = shape(a);
    if (A.rows == null || A.cols == null || A.s == null) continue;
    const b = srt.find((r) => {
      const B = shape(r);
      return B.rows === A.rows && B.cols === A.cols && (B.k == null || A.k == null || B.k === A.k) && B.s != null;
    });
    if (!b) continue;
    const S = shape(b).s as number;
    out.push({ rows: A.rows, cols: A.cols, k: A.k ?? NaN, select: A.s, sort: S, speedup: S / A.s });
  }
  return out;
}

// ---------------------------------------------------------------- rows for the stepper and the scene
function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function shuffle<T>(xs: T[], rand: () => number): T[] {
  const a = xs.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
const f32 = Math.fround;
/** 3.0 plus m units in its last place (2^-22): keys 0xC04000mm, sharing their top 24 bits. */
export const near3 = (m: number) => f32(3 + m * 2 ** -22);

/**
 * The scene's row (and the stepper's first preset): 64 logits for k = 12 built so that the
 * select needs all four passes and ends with ties at K*. Pass 1 puts 6 keys above the
 * crossing byte; pass 2 two more; pass 3 one more; pass 4 one, then K* has three ties of which
 * two are needed.
 */
export const SCENE_K = 12;
export const SCENE_ROW: number[] = (() => {
  const rand = mulberry(7);
  const big = [9.5, 11.25, 14.0, 17.5, 21.0, 26.0]; // [8, 32): key top byte 0xC1, above the cluster's 0xC0
  const mid = [5.0, 6.5]; // [4, 8): second byte above the cluster's 0x40
  const third = [f32(3.01)]; // third byte above the cluster's 0x00
  const cluster = [200, 150, 150, 150, 90, 40, 12].map(near3); // one above, three tied, three below
  const low2 = [2.25, 2.5, 2.75]; // same top byte, second byte below 0x40: out in pass 2
  const rest: number[] = [];
  while (big.length + mid.length + third.length + cluster.length + low2.length + rest.length < 64) {
    const v = rand() < 0.35 ? -(0.5 + rand() * 6) : 0.05 + rand() * 1.8; // top bytes below 0xC0: out in pass 1
    rest.push(f32(Math.round(v * 100) / 100));
  }
  return shuffle([...big, ...mid, ...third, ...cluster, ...low2, ...rest], rand);
})();

export interface StepPreset {
  id: string;
  label: string;
  logits: number[];
  /** The k the preset is made for. */
  k: number;
}
const randomRow = (seed: number, f: (r: () => number) => number) => {
  const rand = mulberry(seed);
  return Array.from({ length: 64 }, () => f32(Math.round(f(rand) * 100) / 100));
};
export const STEP_PRESETS: StepPreset[] = [
  {
    id: 'scene',
    get label() {
      return tr('the animation\'s row', 'hàng của hoạt cảnh');
    },
    logits: SCENE_ROW, k: SCENE_K,
  },
  {
    id: 'random',
    get label() {
      return tr('random', 'ngẫu nhiên');
    },
    logits: randomRow(11, (r) => -4 + r() * 12), k: 10,
  },
  {
    id: 'peaked',
    get label() {
      return tr('peaked', 'có đỉnh nhọn');
    },
    logits: (() => {
      const rand = mulberry(23);
      const xs = Array.from({ length: 64 }, () => f32(Math.round((-6 + rand() * 4) * 100) / 100));
      [[5, 12.5], [19, 9.75], [40, 8.5], [51, 6.0]].forEach(([c, v]) => (xs[c] = v));
      return xs;
    })(),
    k: 3,
  },
  {
    id: 'ties',
    get label() {
      return tr('ties straddling the k-th', 'các giá trị bằng nhau vắt qua vị trí thứ k');
    },
    logits: (() => {
      const rand = mulberry(31);
      const xs = Array.from({ length: 64 }, () => f32(Math.round((-3 + rand() * 4) * 100) / 100));
      [3, 9, 14, 22, 27, 33, 41, 50].forEach((c, i) => (xs[c] = 4 + i * 0.5)); // 8 clear winners
      [6, 18, 30, 37, 58].forEach((c) => (xs[c] = 1.5)); // 5 ties at ranks 9 to 13
      return xs;
    })(),
    k: 10,
  },
  {
    id: 'top8',
    get label() {
      return tr('top 8 bits shared', 'chung 8 bit cao nhất');
    },
    logits: randomRow(41, (r) => 2 + r() * 5.99), k: 10,
  },
  {
    id: 'top16',
    get label() {
      return tr('top 16 bits shared', 'chung 16 bit cao nhất');
    },
    logits: (() => {
      const rand = mulberry(43);
      return Array.from({ length: 64 }, () => f32(3 + Math.floor(rand() * 2 ** 15) * 2 ** -22)); // [3, 3.0078): 0xC040....
    })(),
    k: 10,
  },
  {
    id: 'top24',
    get label() {
      return tr('top 24 bits shared', 'chung 24 bit cao nhất');
    },
    logits: (() => {
      const rand = mulberry(47);
      return Array.from({ length: 64 }, () => near3(Math.floor(rand() * 256)));
    })(),
    k: 10,
  },
  {
    id: 'negative',
    get label() {
      return tr('negatives only', 'chỉ số âm');
    },
    logits: randomRow(53, (r) => -8 + r() * 7.5), k: 10,
  },
];

// ---------------------------------------------------------------- the questions' rows
/** q-hf-ties: logits [6, 5, 5, 4, 1] and k = 2. */
export const HF_TIES = { logits: [6, 5, 5, 4, 1], k: 2 };
/** The bin-walk example: the top four nonempty bins hold 2, 5, 30 and 100 keys; k = 20. */
export const WALK_EXAMPLE = (() => {
  const hist = new Array(BINS).fill(0);
  hist[250] = 2;
  hist[200] = 5;
  hist[150] = 30;
  hist[100] = 100;
  return { hist, k: 20 };
})();
