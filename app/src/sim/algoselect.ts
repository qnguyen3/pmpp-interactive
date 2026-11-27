// Algorithm selection for one block scan on P execution units (chapter 22, section 22.1).
// It runs the explorer "Which scan for this hardware?" and the question q-scan-units.
//
// From the book (notes/ch22.md, p. 530): Brent-Kung does fewer operations, Kogge-Stone
// exposes more parallelism and finishes in fewer steps; which to pick depends on the target
// hardware and on how far hybrids soften the extra work, for example a parallel algorithm
// combined with a sequential one through thread coarsening.
//
// Nothing here re-implements a scan. Every count comes from chapter 11's tested models:
//  - prefixnet.ts: closedForm (additions), span (steps with unlimited units), and
//    stepsWithUnits (a step costs ceil(charge / P) time units under one of three charging
//    rules: its additions, whole warps, or every thread of the block).
//  - coarsescan.ts: coarsenedSteps and coarsenedWork, the book's counts for the three-phase
//    coarsened scan of N elements by T threads (pp. 269-270): 2N/T + log2T - 1 steps and
//    2N + T·log2T - T - N/T additions.
// What this file adds: the coarsened scan's thread count for P units, the comparison, the
// winner and the values of P where the winner changes.
//
// Ours, not the book's:
//  - The coarsened scan runs T threads on P units with T the largest power of two that is at
//    most both P and N, one thread per unit: its time is chapter 11's step count for T
//    threads. Using every unit is its best choice (2N/T + log2T falls while T < 2N·ln2).
//    With T = 1 it is the sequential loop, so the loop's counts are used (chapter 11's
//    three-phase formula would count an add phase that a single thread does not have).
//  - The coarsened scan's time ignores the charging rule. Its first and third phases keep
//    every thread busy, and its middle phase is a Kogge-Stone scan of T values on T units,
//    which costs one time unit per step under every rule once T is at least a warp (tested).
//
// What it leaves out (chapter 11's model): memory traffic and its latency, barriers, the
// time a step takes beyond its count, and any overlap between steps.
//
// Erasable TypeScript only.

import { closedForm, lg, span, stepsWithUnits, type Charge } from './prefixnet';
import { coarsenedSteps, coarsenedWork } from './coarsescan';
import { tr } from '../lib/locale.ts';

export type AlgoId = 'kogge-stone' | 'brent-kung' | 'coarsened';
export const ALGOS: AlgoId[] = ['kogge-stone', 'brent-kung', 'coarsened'];
export const ALGO_LABEL: Record<AlgoId, string> = {
  'kogge-stone': 'Kogge-Stone',
  'brent-kung': 'Brent-Kung',
  get coarsened() {
    return tr('Coarsened three-phase', 'Ba pha có gộp việc');
  },
};

export interface AlgoRow {
  id: AlgoId;
  /** Time units with P execution units. */
  time: number;
  /** Additions the scan performs. */
  additions: number;
  /** Steps with unlimited execution units. */
  span: number;
  /** Threads the scan runs. */
  threads: number;
}

const isPow2 = (n: number) => Number.isInteger(n) && n >= 1 && (n & (n - 1)) === 0;

/** Threads the coarsened scan runs on P units: the largest power of two at most min(P, N). */
export function coarsenedThreads(n: number, p: number): number {
  const m = Math.max(1, Math.min(Math.floor(p), n));
  return 2 ** Math.floor(Math.log2(m));
}

/** One scan's counts for N elements on P units. N must be a power of two of at least 2. */
export function row(id: AlgoId, n: number, p: number, charge: Charge, warp: number): AlgoRow {
  if (!isPow2(n) || n < 2) throw new Error('N must be a power of two, at least 2');
  if (!(p >= 1)) throw new Error('P must be at least 1');
  if (id === 'coarsened') {
    const t = coarsenedThreads(n, p);
    if (t === 1) return { id, time: span('sequential', n), additions: closedForm('sequential', n), span: span('sequential', n), threads: 1 };
    const tn = coarsenedThreads(n, n);
    return { id, time: coarsenedSteps(n, t).total, additions: coarsenedWork(n, t).total, span: coarsenedSteps(n, tn).total, threads: t };
  }
  return {
    id,
    time: stepsWithUnits(id, n, p, charge, warp),
    additions: closedForm(id, n),
    span: span(id, n),
    threads: id === 'kogge-stone' ? n : n / 2,
  };
}

export function compare(n: number, p: number, charge: Charge, warp: number): Record<AlgoId, AlgoRow> {
  return {
    'kogge-stone': row('kogge-stone', n, p, charge, warp),
    'brent-kung': row('brent-kung', n, p, charge, warp),
    coarsened: row('coarsened', n, p, charge, warp),
  };
}

/** The faster of the two networks, or 'tie'. */
export function networkWinner(rows: Record<AlgoId, AlgoRow>): 'kogge-stone' | 'brent-kung' | 'tie' {
  const ks = rows['kogge-stone'].time;
  const bk = rows['brent-kung'].time;
  return ks < bk ? 'kogge-stone' : bk < ks ? 'brent-kung' : 'tie';
}

/** The fastest of all three, or 'tie' when two share the lowest time. */
export function overallWinner(rows: Record<AlgoId, AlgoRow>): AlgoId | 'tie' {
  const best = Math.min(...ALGOS.map((a) => rows[a].time));
  const at = ALGOS.filter((a) => rows[a].time === best);
  return at.length === 1 ? at[0] : 'tie';
}

/** The coarsened scan's time over the better network's: 1 means as fast, below 1 faster. */
export function hybridRatio(rows: Record<AlgoId, AlgoRow>): number {
  return rows.coarsened.time / Math.min(rows['kogge-stone'].time, rows['brent-kung'].time);
}

export interface Change {
  /** The first P (an integer) at which the network winner differs from the one at P - 1. */
  p: number;
  from: 'kogge-stone' | 'brent-kung' | 'tie';
  to: 'kogge-stone' | 'brent-kung' | 'tie';
}

/** Every integer P from 2 to maxP at which the faster network changes. */
export function crossovers(n: number, charge: Charge, warp: number, maxP = n): Change[] {
  const out: Change[] = [];
  let prev = networkWinner(compare(n, 1, charge, warp));
  for (let p = 2; p <= maxP; p++) {
    const ks = stepsWithUnits('kogge-stone', n, p, charge, warp);
    const bk = stepsWithUnits('brent-kung', n, p, charge, warp);
    const w = ks < bk ? 'kogge-stone' : bk < ks ? 'brent-kung' : 'tie';
    if (w !== prev) out.push({ p, from: prev, to: w });
    prev = w;
  }
  return out;
}

/** The largest P such that Brent-Kung is strictly faster than Kogge-Stone for every P from 1 to it, or null. */
export function brentKungUpTo(n: number, charge: Charge, warp: number, maxP = n): number | null {
  let upTo: number | null = null;
  for (let p = 1; p <= maxP; p++) {
    const ks = stepsWithUnits('kogge-stone', n, p, charge, warp);
    const bk = stepsWithUnits('brent-kung', n, p, charge, warp);
    if (bk < ks) upTo = p;
    else break;
  }
  return upTo;
}

/** The smallest P from which Kogge-Stone is strictly faster than Brent-Kung for every larger P up to maxP, or null. */
export function koggeStoneFrom(n: number, charge: Charge, warp: number, maxP = n): number | null {
  let from: number | null = null;
  for (let p = maxP; p >= 1; p--) {
    const ks = stepsWithUnits('kogge-stone', n, p, charge, warp);
    const bk = stepsWithUnits('brent-kung', n, p, charge, warp);
    if (ks < bk) from = p;
    else break;
  }
  return from;
}

export interface CurvePoint {
  p: number;
  'kogge-stone': number;
  'brent-kung': number;
  coarsened: number;
}

/**
 * Times against P for every power of two from 1 to N: the explorer's P. The coarsened scan as
 * chapter 11 counts it needs a power-of-two thread count, so the explorer offers only those.
 */
export function curve(n: number, charge: Charge, warp: number): CurvePoint[] {
  const out: CurvePoint[] = [];
  for (let p = 1; p <= n; p *= 2) {
    const r = compare(n, p, charge, warp);
    out.push({ p, 'kogge-stone': r['kogge-stone'].time, 'brent-kung': r['brent-kung'].time, coarsened: r.coarsened.time });
  }
  return out;
}

/** The coarsened scan's time minus the better network's: positive means it is behind. */
export const hybridLag = (rows: Record<AlgoId, AlgoRow>) =>
  rows.coarsened.time - Math.min(rows['kogge-stone'].time, rows['brent-kung'].time);

/** log2 of N, for slider positions. */
export const log2N = (n: number) => lg(n);

/** Slider positions: N from 2^6 to 2^14 (the explorer's range). */
export const N_MIN_LOG = 6;
export const N_MAX_LOG = 14;
