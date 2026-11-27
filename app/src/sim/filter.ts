// Keys, keep flags and the unstable filter family of chapter 12, sections 12.1 to 12.4.
// Section 12.1's "Stable or unstable?" explorer, section 12.2's "One counter, many threads"
// explorer and beats 1 to 3 of the scene "One vote, one atomic" run on it.
//
// What it rests on (notes/ch12.md and the facts files):
//   - p. 289-290: a filter removes the keys that fail a test and compacts the rest; a
//     stable filter keeps their input order, an unstable one does not (Fig. 12.1, whose
//     kept keys are k1, k3, k4, k6, k8, k10, k11 and k14).
//   - p. 290-291, Fig. 12.2: each kept key bumps one counter with fetch_add(1) and is
//     stored at the old value. Atomics on one address complete in no fixed order
//     (ch09-facts A3 for NVIDIA, B1 for AMD), so the slots go out in an order nobody chose.
//     Here that order is one draw of a seeded shuffle. It stands for "some order the
//     hardware picked"; it is not a model of any hardware.
//   - Fig. 12.3 and the ballot form of lab 12.1: one atomic per warp that has a kept key.
//     Fig. 12.3 as printed may see a warp's passing lanes as several converged snapshots,
//     each with its own atomic (ch12-facts A1).
//   - Fig. 12.6 (p. 296): one public atomic per block, issued by thread 0 even when the
//     block kept nothing, plus one block-scope atomic per kept key on the private counter.
//
// Counts only: nothing here models time, caches, queues, or what a compiler may already
// have aggregated (ch12-facts A7, B4).
//
// Erasable TypeScript only, and the one import is lib/locale.ts (itself import-free): the browser
// checks import this file directly.

import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- keys and keep flags
/** Fig. 12.1: sixteen keys k0 to k15; the shaded ones are kept. */
export const FIG12_1 = {
  n: 16,
  kept: [1, 3, 4, 6, 8, 10, 11, 14],
  /** The book's unstable panel, output slot by output slot. One possible order of many. */
  bookUnstable: [4, 10, 3, 1, 6, 14, 11, 8],
} as const;

/** Keep flags for n keys with the given indices kept. */
export function keepFrom(n: number, kept: readonly number[]): boolean[] {
  const keep = Array.from({ length: n }, () => false);
  for (const k of kept) {
    if (!Number.isInteger(k) || k < 0 || k >= n) throw new Error(`key ${k} is not one of 0..${n - 1}`);
    keep[k] = true;
  }
  return keep;
}

/** Indices of the kept keys. */
export const keptOf = (keep: readonly boolean[]): number[] => keep.flatMap((k, i) => (k ? [i] : []));

/** The stable filter: kept keys in input order. This is also the CPU reference order. */
export function stableFilter(keep: readonly boolean[]): number[] {
  return keptOf(keep);
}

/** A small seeded generator (mulberry32), so every "random" result can be reproduced. */
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

/**
 * The order in which the kept keys' atomics complete: one draw of a seeded shuffle of the
 * kept indices. Run 1 and run 2 of a kernel are two seeds.
 */
export function arrivalOrder(keep: readonly boolean[], seed: number): number[] {
  const order = keptOf(keep);
  const r = rng(seed);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}

/**
 * Fig. 12.2's output: the counter starts at 0 and the k-th atomic to complete returns k, so
 * slot k holds the k-th key to arrive. `arrival` must be the kept indices in some order.
 */
export function unstableFilter(keep: readonly boolean[], arrival: readonly number[]): number[] {
  const kept = keptOf(keep);
  const sorted = [...arrival].sort((a, b) => a - b);
  if (sorted.length !== kept.length || sorted.some((k, i) => k !== kept[i]))
    throw new Error('the arrival order must contain every kept key exactly once');
  return [...arrival];
}

/** For each output slot, the index of the key stored there; inverted: the slot of each key. */
export function slotOf(output: readonly number[]): Map<number, number> {
  return new Map(output.map((key, slot) => [key, slot]));
}

/**
 * Pairs of input-to-output arrows that cross: pairs of kept keys whose output order is the
 * reverse of their input order. Zero exactly when the filter kept the order.
 */
export function crossings(output: readonly number[]): number {
  let n = 0;
  for (let a = 0; a < output.length; a++) for (let b = a + 1; b < output.length; b++) if (output[a] > output[b]) n++;
  return n;
}

/** Same keys and the same count, in any order: what every run of an unstable filter guarantees. */
export function sameKeys(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  const x = [...a].sort((p, q) => p - q);
  const y = [...b].sort((p, q) => p - q);
  return x.every((v, i) => v === y[i]);
}

export interface Overwrite {
  /** Output slot, which is also the position in the one array an in-place filter uses. */
  slot: number;
  /** The kept key written there. */
  writer: number;
  /** The key that lived at this position before: key number `slot`. */
  held: number;
  /** That key is kept and moves to another slot, so it must be read before this write. */
  heldMovesTo: number | null;
  /**
   * What the write asks of the key it lands on. 'own': a key written over itself, nothing to
   * protect. 'moved': a kept key bound elsewhere, which must be read and moved first.
   * 'tested': a dropped key, which must still be read first, because a thread tests it.
   */
  kind: 'own' | 'moved' | 'tested';
}

/**
 * A stable filter done in place: key `writer` lands in position `slot`, where key `slot`
 * used to live. Every old key must be read (to test it, and to move it if kept) before its
 * position is overwritten: the hazard of p. 290, which section 12.7 orders.
 */
export function inPlaceOverwrites(keep: readonly boolean[]): Overwrite[] {
  const out = stableFilter(keep);
  const where = slotOf(out);
  return out.map((writer, slot) => {
    const heldMovesTo = keep[slot] && slot !== writer ? where.get(slot)! : null;
    return { slot, writer, held: slot, heldMovesTo, kind: slot === writer ? 'own' : heldMovesTo != null ? 'moved' : 'tested' };
  });
}

// ---------------------------------------------------------------- input patterns
/** How the kept keys are laid out in the input. */
export type Pattern = 'random' | 'runs' | 'spread' | 'sorted';
export const PATTERNS: readonly Pattern[] = ['random', 'runs', 'spread', 'sorted'];
export const PATTERN_LABEL: Record<Pattern, string> = {
  get random() {
    return tr('independent random keys', 'khóa ngẫu nhiên độc lập');
  },
  get runs() {
    return tr('runs of kept keys', 'các dải khóa được giữ liền nhau');
  },
  get spread() {
    return tr('evenly spread (at 50 percent, every other key)', 'rải đều (ở 50 phần trăm, cứ cách một khóa)');
  },
  get sorted() {
    return tr('sorted: the kept keys come first', 'đã sắp xếp: các khóa được giữ đứng trước');
  },
};
/** Pattern "runs": the input is cut into runs of this many keys, each kept or dropped as a whole. */
export const RUN_LENGTH = 50;

/** Counts kept keys over ranges of the input without materialising 16 million flags. */
export interface KeepSource {
  n: number;
  /** Kept keys among indices lo .. hi - 1. */
  keptIn(lo: number, hi: number): number;
}

/** A source over explicit flags (prefix sums). */
export function fromFlags(keep: readonly boolean[]): KeepSource {
  const pre = new Int32Array(keep.length + 1);
  for (let i = 0; i < keep.length; i++) pre[i + 1] = pre[i] + (keep[i] ? 1 : 0);
  const clip = (v: number) => Math.max(0, Math.min(keep.length, v));
  return { n: keep.length, keptIn: (lo, hi) => (hi <= lo ? 0 : pre[clip(hi)] - pre[clip(lo)]) };
}

/** Independent keys, each kept with probability `fraction` (one seeded draw). */
export function randomFlags(n: number, fraction: number, seed = 1): boolean[] {
  const r = rng(seed);
  return Array.from({ length: n }, () => r() < fraction);
}

/**
 * A concrete input of n keys with about `fraction` of them kept, laid out by `pattern`.
 *  - random: independent keys (explicit flags; meant for n up to a few million).
 *  - runs: runs of RUN_LENGTH keys, each run kept with probability `fraction` (seeded).
 *  - spread: key i is kept when floor((i + 1) f) > floor(i f): evenly spaced, every other
 *    key at f = 0.5, every key at f = 1.
 *  - sorted: the first round(f n) keys are kept, the rest dropped.
 */
export function patternSource(pattern: Pattern, n: number, fraction: number, seed = 1): KeepSource {
  const f = Math.max(0, Math.min(1, fraction));
  switch (pattern) {
    case 'random':
      return fromFlags(randomFlags(n, f, seed));
    case 'sorted': {
      const k = Math.round(f * n);
      return { n, keptIn: (lo, hi) => Math.max(0, Math.min(hi, k, n) - Math.max(lo, 0)) };
    }
    case 'spread':
      // floor(i f) counts the kept keys among 0 .. i - 1.
      return { n, keptIn: (lo, hi) => (hi <= lo ? 0 : Math.floor(Math.min(hi, n) * f + 1e-9) - Math.floor(Math.max(lo, 0) * f + 1e-9)) };
    case 'runs': {
      const runs = Math.ceil(n / RUN_LENGTH);
      const r = rng(seed);
      const on = Array.from({ length: runs }, () => r() < f);
      return {
        n,
        keptIn: (lo, hi) => {
          lo = Math.max(lo, 0);
          hi = Math.min(hi, n);
          let s = 0;
          for (let run = Math.floor(lo / RUN_LENGTH); run * RUN_LENGTH < hi; run++) {
            if (!on[run]) continue;
            s += Math.min(hi, (run + 1) * RUN_LENGTH, n) - Math.max(lo, run * RUN_LENGTH);
          }
          return s;
        },
      };
    }
  }
}

// ---------------------------------------------------------------- atomics on the counter
export interface AtomicCounts {
  n: number;
  kept: number;
  warps: number;
  blocks: number;
  /** Fig. 12.2: one atomic on the public counter per kept key. */
  perKey: number;
  /** The ballot form of section 12.3 (lab 12.1): one per warp that has a kept key. */
  perWarp: number;
  /**
   * Fig. 12.3 as printed when each warp's passing lanes reach __activemask() as `snapshots`
   * converged groups: one atomic per group, so min(snapshots, kept) per warp.
   */
  perSnapshot: number;
  /** Fig. 12.6: one public atomic per block, issued even when the block kept nothing. */
  perBlock: number;
  /** Fig. 12.6's block-scope atomics on the private counter in shared memory: one per kept key. */
  privatePerKey: number;
  /** Fig. 12.6 with a warp vote in front of the private counter: one per warp with a kept key. */
  privatePerWarp: number;
}

function checkShape(warp: number, block: number) {
  if (!Number.isInteger(warp) || warp < 1) throw new Error('warp width must be a positive integer');
  if (!Number.isInteger(block) || block < warp || block % warp !== 0) throw new Error('block size must be a multiple of the warp width');
}

/** Atomic operations each kernel issues for a concrete input. Warps never straddle blocks. */
export function atomicCounts(src: KeepSource, warp: number, block: number, snapshots = 1): AtomicCounts {
  checkShape(warp, block);
  const { n } = src;
  const warps = Math.ceil(n / warp);
  let busy = 0;
  let split = 0;
  for (let w = 0; w < warps; w++) {
    const k = src.keptIn(w * warp, Math.min(n, (w + 1) * warp));
    if (k > 0) busy++;
    split += Math.min(k, snapshots);
  }
  const kept = src.keptIn(0, n);
  return {
    n,
    kept,
    warps,
    blocks: Math.ceil(n / block),
    perKey: kept,
    perWarp: busy,
    perSnapshot: split,
    perBlock: Math.ceil(n / block),
    privatePerKey: kept,
    privatePerWarp: busy,
  };
}

export interface Expected {
  /** Mean atomics on the public counter for independent keys kept with probability f. */
  perKey: number;
  perWarp: number;
  perBlock: number;
}

/**
 * Expected counts for n independent keys, each kept with probability f. A warp of W lanes
 * has no kept key with probability (1 - f)^W, so the vote form issues
 * (n / W)(1 - (1 - f)^W) atomics; a last, partial warp of r lanes counts 1 - (1 - f)^r.
 */
export function expectedAtomics(n: number, fraction: number, warp: number, block: number): Expected {
  checkShape(warp, block);
  const f = Math.max(0, Math.min(1, fraction));
  const full = Math.floor(n / warp);
  const rest = n - full * warp;
  const busy = (lanes: number) => 1 - Math.pow(1 - f, lanes);
  return {
    perKey: n * f,
    perWarp: full * busy(warp) + (rest > 0 ? busy(rest) : 0),
    perBlock: Math.ceil(n / block),
  };
}

/** Fractions the explorer's slider steps through: fine near zero, where the vote form loses its edge. */
export const FRACTIONS: readonly number[] = [
  0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
];
/** The points the explorer's curves are drawn through (log axis: no zero). */
export const CURVE_FRACTIONS: readonly number[] = Array.from({ length: 31 }, (_, i) => Math.pow(10, -3 + (3 * i) / 30));

export interface CurvePoint {
  fraction: number;
  perKey: number;
  perWarp: number;
  perBlock: number;
}

/** The three counts across fractions, for a pattern: expectations for random keys, concrete counts otherwise. */
export function curve(pattern: Pattern, n: number, warp: number, block: number, fractions: readonly number[] = CURVE_FRACTIONS, seed = 1): CurvePoint[] {
  return fractions.map((f) => {
    if (pattern === 'random') return { fraction: f, ...expectedAtomics(n, f, warp, block) };
    const c = atomicCounts(patternSource(pattern, n, f, seed), warp, block);
    return { fraction: f, perKey: c.perKey, perWarp: c.perWarp, perBlock: c.perBlock };
  });
}

// ---------------------------------------------------------------- the scene's toy warp
/** Eight lanes of a toy warp, each with its key; the scene keeps keys below the threshold. */
export const SCENE = {
  keys: [7, 2, 3, 9, 1, 4, 8, 6],
  threshold: 5,
  /** The counter's value when the warp arrives. */
  counter: 40,
  /** Two runs of Fig. 12.2 on the same warp: seeds for two different completion orders. */
  seeds: [3, 8],
} as const;

/** The scene's keep flags: key < threshold, the predicate lab 12.1 uses. */
export const sceneKeep = (): boolean[] => SCENE.keys.map((k) => k < SCENE.threshold);

export interface Served {
  /** Lane whose atomic this is. */
  lane: number;
  /** The value fetch_add returned to it: its output slot. */
  slot: number;
}

/** Fig. 12.2 on the scene's warp: the counter serves the passing lanes one at a time, in arrival order. */
export function serveOneByOne(keep: readonly boolean[], counter: number, seed: number): Served[] {
  return arrivalOrder(keep, seed).map((lane, k) => ({ lane, slot: counter + k }));
}
