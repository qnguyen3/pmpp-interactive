// The stable filter family of chapter 12 (book §12.5, pp. 297-298; §12.6, pp. 298-300;
// exercises 1 and 2, p. 302).
//
// A stable filter cannot take its output slot from a counter: a kept key needs the
// number of kept keys before it, which is an exclusive scan of 0/1 keep flags (Fig. 12.7).
// This file holds:
//   1. keep flags from three sources (the explorer's three skins): keys with a predicate,
//      logits against a threshold (a min-p row: logit >= max + ln p), and probabilities
//      sorted in descending order against p (top-p: exclusive cumulative mass < p);
//   2. the exclusive scan, the total (last offset + last keep, p. 298), the earlier
//      keepers behind any offset, and the scatter;
//   3. blockBinaryScan: the same offsets for one block computed the way lab 12.3 part B
//      and lab 12.4 compute them, from warp votes (rank = popcount of the vote masked to
//      the lanes below, as in §12.3) and one scan of the warp totals (chapter 11's
//      scan-scan-add, §11.4 pp. 261-265);
//   4. bytes of global traffic per input key for three designs, as functions of the kept
//      fraction. That accounting is ours, not the book's: the book asks for the "major
//      advantages" of a single kernel (exercise 2) without counting them.
//
// What it leaves out: time, caches, the scan's own internal traffic (stated where it
// matters), and anything about how the hardware executes the vote.
//
// Erasable TypeScript only.

import { tr } from '../lib/locale.ts';

/** Fig. 12.7's keep row (p. 297): k1, k3, k4, k6, k8, k10, k11 and k14 are kept. */
export const BOOK_KEEP: readonly number[] = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0];
export const BOOK_KEYS: readonly string[] = Array.from({ length: 16 }, (_, i) => `k${i}`);

// ---------------------------------------------------------------- 2. scan and scatter
/** Exclusive scan of the flags: offset[i] = flags[0] + ... + flags[i-1]. */
export function exclusiveScan(flags: readonly number[]): number[] {
  const out: number[] = [];
  let run = 0;
  for (const f of flags) {
    out.push(run);
    run += f;
  }
  return out;
}

/** The output size the last thread writes in Fig. 12.8: its exclusive offset plus its own keep (p. 298). */
export function total(keep: readonly number[], offsets: readonly number[]): number {
  const n = keep.length;
  return n ? offsets[n - 1] + keep[n - 1] : 0;
}

/** The kept keys before position i: the ones its offset counts. */
export function earlierKeepers(keep: readonly number[], i: number): number[] {
  const out: number[] = [];
  for (let j = 0; j < i; j++) if (keep[j]) out.push(j);
  return out;
}

/** Every kept item at its offset. Items whose keep is 0 are never written; their offset is unused. */
export function scatter<T>(items: readonly T[], keep: readonly number[], offsets: readonly number[]): T[] {
  const out: T[] = new Array<T>(total(keep, offsets));
  for (let i = 0; i < items.length; i++) if (keep[i]) out[offsets[i]] = items[i];
  return out;
}

export interface Filtered<T> {
  keep: number[];
  offsets: number[];
  total: number;
  output: T[];
}
export function stableFilter<T>(items: readonly T[], keep: readonly number[]): Filtered<T> {
  const offsets = exclusiveScan(keep);
  return { keep: [...keep], offsets, total: total(keep, offsets), output: scatter(items, keep, offsets) };
}

// ---------------------------------------------------------------- 1. the three skins
/** A small seeded generator (the same LCG family the course's emulator uses for its shuffle). */
function lcg(seed: number) {
  let r = (Math.imul(seed, 2654435761) + 12345) >>> 0;
  return () => {
    r = (Math.imul(r, 1664525) + 1013904223) >>> 0;
    return (r >>> 8) / 16777216;
  };
}

/** A row of n logits for the min-p skin, seeded, one decimal each, in token-id order (unsorted). */
export function makeLogits(n = 16, seed = 3): number[] {
  const r = lcg(seed);
  return Array.from({ length: n }, () => Math.round((r() * 8 - 2) * 10) / 10);
}

/** min-p as llama.cpp's host sampler writes it: keep a logit at or above max + ln p (ch12-facts D3). */
export const minPThreshold = (logits: readonly number[], p: number) => Math.max(...logits) + Math.log(p);
export const keepAtLeast = (logits: readonly number[], threshold: number): number[] => logits.map((l) => (l >= threshold ? 1 : 0));

/** Probabilities for the top-p skin: a softmax of seeded logits, sorted in descending order, rounded to 3 decimals. */
export function makeSortedProbs(n = 16, seed = 5): number[] {
  const r = lcg(seed);
  const logits = Array.from({ length: n }, () => r() * 5);
  const m = Math.max(...logits);
  const e = logits.map((l) => Math.exp(l - m));
  const s = e.reduce((a, b) => a + b, 0);
  const p = e.map((x) => Math.round((x / s) * 1000) / 1000).sort((a, b) => b - a);
  // Make the rounded values sum to exactly 1 by adjusting the largest.
  const err = Math.round((1 - p.reduce((a, b) => a + b, 0)) * 1000) / 1000;
  p[0] = Math.round((p[0] + err) * 1000) / 1000;
  return p;
}

/** Cumulative mass before each position (an exclusive scan of the probabilities), rounded to 3 decimals. */
export function massBefore(probs: readonly number[]): number[] {
  const out: number[] = [];
  let run = 0;
  for (const p of probs) {
    out.push(Math.round(run * 1000) / 1000);
    run += p;
  }
  return out;
}

/** top-p over probabilities sorted in descending order: keep while the mass before you is below p. */
export const keepTopP = (probs: readonly number[], p: number): number[] => massBefore(probs).map((m) => (m < p ? 1 : 0));

/** True when the kept set is a run of ones followed only by zeros. */
export function isPrefix(keep: readonly number[]): boolean {
  let seenZero = false;
  for (const k of keep) {
    if (!k) seenZero = true;
    else if (seenZero) return false;
  }
  return true;
}
/** Length of the leading run of ones: for a prefix, the one cut index that replaces the whole filter. */
export function cutIndex(keep: readonly number[]): number {
  let i = 0;
  while (i < keep.length && keep[i]) i++;
  return i;
}

// ---------------------------------------------------------------- 3. one block, from votes
export const popcount = (x: number) => {
  let v = x >>> 0;
  let c = 0;
  while (v) {
    v &= v - 1;
    c++;
  }
  return c;
};
/** Bits 0 .. lane-1 set: the book's (1 << laneIdx()) - 1, written to stay defined at lane 31. */
export const lanesBelow = (lane: number) => (lane >= 32 ? 0xffffffff : (2 ** lane - 1)) >>> 0;

export interface BlockScan {
  warpWidth: number;
  warps: number;
  /** Per warp: the vote, bit l set when lane l keeps its key (a ballot in uniform control flow). */
  masks: number[];
  /** Per thread: kept lanes below it in its warp, popcount(mask & lanesBelow(lane)). */
  ranks: number[];
  /** Per warp: popcount(mask). */
  warpTotals: number[];
  /** Exclusive scan of the warp totals: where each warp's kept keys start in the block's output. */
  warpStarts: number[];
  /** Per thread: warpStarts[warp] + rank. Meaningful only where keep = 1, like any offset. */
  offsets: number[];
  total: number;
}

/**
 * The block-local exclusive scan of 0/1 flags from warp votes. One block holds keep.length
 * threads; warpWidth is at most 32 here (the explorer's toy uses 8). The tests check that
 * the offsets equal exclusiveScan(keep) for every input.
 */
export function blockBinaryScan(keep: readonly number[], warpWidth: number): BlockScan {
  if (!(warpWidth >= 1 && warpWidth <= 32)) throw new Error(`warp width ${warpWidth}`);
  const warps = Math.ceil(keep.length / warpWidth);
  const masks = Array.from({ length: warps }, (_, w) => {
    let m = 0;
    for (let l = 0; l < warpWidth; l++) if (keep[w * warpWidth + l]) m = (m | (2 ** l)) >>> 0;
    return m >>> 0;
  });
  const ranks = keep.map((_, t) => popcount(masks[Math.floor(t / warpWidth)] & lanesBelow(t % warpWidth)));
  const warpTotals = masks.map(popcount);
  const warpStarts = exclusiveScan(warpTotals);
  const offsets = keep.map((_, t) => warpStarts[Math.floor(t / warpWidth)] + ranks[t]);
  return { warpWidth, warps, masks, ranks, warpTotals, warpStarts, offsets, total: total(warpTotals, warpStarts) };
}

/** A mask as the book prints one (p. 292): bit 0, lane 0, on the right. */
export const maskBits = (mask: number, width: number) => (mask >>> 0).toString(2).padStart(width, '0').slice(-width);

// ---------------------------------------------------------------- 4. bytes per key (ours)
export type TrafficDesign = 'three-kernels' | 'staged' | 'single-pass';
export const TRAFFIC_DESIGNS: TrafficDesign[] = ['three-kernels', 'staged', 'single-pass'];
export const TRAFFIC_LABEL: Record<TrafficDesign, string> = {
  get 'three-kernels'() {
    return tr('exercise 1: keep flags, scan, scatter (three kernels)', 'bài tập 1: cờ giữ lại, scan, rải (ba kernel)');
  },
  get staged() {
    return tr('count per block, scan the block counts, staged scatter (three kernels)', 'đếm theo khối, scan các số đếm của khối, rải có gom tạm (ba kernel)');
  },
  get 'single-pass'() {
    return tr('exercise 2: one kernel with a single-pass scan', 'bài tập 2: một kernel với scan một lượt');
  },
};

export interface Traffic {
  design: TrafficDesign;
  /** Each kernel's reads and writes, in bytes per input key. */
  parts: { kernel: string; reads: number; writes: number }[];
  reads: number;
  writes: number;
  total: number;
}

/**
 * Global-memory bytes per input key for 4-byte keys, flags and offsets, with a fraction f
 * of the keys kept and keysPerBlock keys per block (block size times coarsening factor).
 * Counted: every array a kernel reads or writes in full, once. Not counted: the scan's own
 * internal traffic beyond reading its input and writing its output, caches, and the
 * single pass's look-back reads of earlier blocks' flags, which are a few words per block.
 */
export function traffic(design: TrafficDesign, f: number, keysPerBlock = 256): Traffic {
  if (!(f >= 0 && f <= 1)) throw new Error(`fraction ${f}`);
  const B = keysPerBlock;
  const parts =
    design === 'three-kernels'
      ? [
          { kernel: tr('keep flags: read each key, write its flag', 'cờ giữ lại: đọc từng khóa, ghi cờ của nó'), reads: 4, writes: 4 },
          { kernel: tr('exclusive scan: read the flags, write the offsets', 'scan loại trừ: đọc các cờ, ghi các độ lệch'), reads: 4, writes: 4 },
          { kernel: tr('scatter: read key, flag and offset, write the kept keys', 'rải: đọc khóa, cờ và độ lệch, ghi các khóa được giữ'), reads: 12, writes: 4 * f },
        ]
      : design === 'staged'
        ? [
            { kernel: tr('count: read each key, write one count per block', 'đếm: đọc từng khóa, ghi một số đếm cho mỗi khối'), reads: 4, writes: 4 / B },
            { kernel: tr('scan the block counts', 'scan các số đếm của khối'), reads: 4 / B, writes: 4 / B },
            { kernel: tr('staged scatter: read each key and its block offset, write the kept keys', 'rải có gom tạm: đọc từng khóa và độ lệch khối của nó, ghi các khóa được giữ'), reads: 4 + 4 / B, writes: 4 * f },
          ]
        : [{ kernel: tr('one kernel: read each key once, write each kept key once', 'một kernel: đọc mỗi khóa một lần, ghi mỗi khóa được giữ một lần'), reads: 4, writes: 4 * f }];
  const reads = parts.reduce((s, p) => s + p.reads, 0);
  const writes = parts.reduce((s, p) => s + p.writes, 0);
  return { design, parts, reads, writes, total: reads + writes };
}
