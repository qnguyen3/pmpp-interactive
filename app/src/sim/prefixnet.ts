// Prefix networks for chapter 11: the sequential loop, Kogge-Stone and Brent-Kung written as
// steps of adders, and the counts that sections 11.5 and 11.10 reason with. It runs the
// "Work and steps as N grows" explorer (11.5), the "Up-sweep, down-sweep" stepper and the
// "Inside one warp" explorer (11.10), and the narrated scene "Up-sweep and down-sweep".
//
// From the book (notes/ch11.md; pp. 265-267 and 282-285 also read directly in the PDF):
//  - Fig. 11.1, p. 254: the loop does N - 1 additions; p. 266 counts it as about N steps.
//  - Fig. 11.2, pp. 255-256: Kogge-Stone. The step with stride s adds position i - s into
//    position i for every i >= s: 15 + 14 + 12 + 8 = 49 adders on 16 positions.
//  - §11.5, pp. 265-267: work N·log2N - (N - 1); on hardware "closer to N·log2N" because
//    threads that stop adding still use resources until their warp completes; steps with P
//    execution units (N·log2N)/P; 512 elements: 4097 additions against 511; 1024 elements
//    and 32 units: 320 steps, 3.2 times fewer than the loop's 1024.
//  - §11.10, Figs. 11.19 and 11.20, pp. 282-285: Brent-Kung. The reduction tree updates the
//    positions 2n - 1, then 4n - 1, 8n - 1, 16n - 1; the reverse tree adds 7 into 11, then 3,
//    7, 11 into 5, 9, 13, then every odd position into its right neighbour; 2N - 2 - log2N
//    additions; Fig. 11.20's shading counts the partial sums a position still needs.
//  - Exercises 1 and 6, p. 286: both networks on 4 6 7 1 2 8 5 2.
//
// Ours, not the book's:
//  - Brent-Kung's thread mapping in a block: thread t performs the t-th adder of a step, at
//    idx = (t + 1)·2·stride - 1, with N/2 threads for N elements. That is what
//    labs/ch11/snippets/s1110_brent_kung.cu does (our answer to exercise 7).
//  - Inside one warp each lane holds one value, so the lane that adds is the lane that owns
//    the destination position, for both networks.
//  - Three ways to charge a step to P execution units: its additions only; whole warps (a
//    warp with any adding lane is charged all its lanes, the reason p. 266 gives); every
//    thread of the block (the book's own estimate for Kogge-Stone, (N·log2N)/P). A step costs
//    ceil(charge / P) time units and the next step starts when it has finished.
//  - The step count 2·log2N - 1 for Brent-Kung is counted from Fig. 11.19's structure.
//  - Fig. 11.20's "needs": the chain of ranges a position still has to add, read from the
//    ranges every position holds. It reproduces every printed cell but three (columns 9 and
//    11 of the Initial row, column 9 of Level 1), which are printed one shade too dark.
//
// What it leaves out: the time a step takes (all steps cost the same), memory and its
// latency, barriers and shuffles, and any overlap between steps.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

// ---------------------------------------------------------------- networks
import { tr } from '../lib/locale.ts';

export type NetId = 'sequential' | 'kogge-stone' | 'brent-kung';
export const NET_LABEL: Record<NetId, string> = {
  get sequential() {
    return tr('sequential loop', 'vòng lặp tuần tự');
  },
  'kogge-stone': 'Kogge-Stone',
  'brent-kung': 'Brent-Kung',
};

/** dst += src, reading the value src held before the step. */
export interface Adder {
  src: number;
  dst: number;
  /** The thread that performs it in a block-level kernel (see the header). */
  thread: number;
}
export type Phase = 'loop' | 'sweep' | 'up' | 'down';
export interface Step {
  adders: Adder[];
  phase: Phase;
  stride: number;
  /** Down-sweep only: level 1, 2, ... of the reverse tree (Fig. 11.20's rows). */
  level?: number;
}
export interface Network {
  id: NetId;
  n: number;
  steps: Step[];
  /** Threads a block-level kernel uses: 1 for the loop, N for Kogge-Stone, N/2 for Brent-Kung. */
  threads: number;
}

export const isPow2 = (n: number) => Number.isInteger(n) && n >= 1 && (n & (n - 1)) === 0;
/** log2 of a power of two. */
export const lg = (n: number) => Math.round(Math.log2(n));

/** Fig. 11.1: one addition per step, i - 1 into i. */
export function sequential(n: number): Network {
  const steps: Step[] = [];
  for (let i = 1; i < n; i++) steps.push({ adders: [{ src: i - 1, dst: i, thread: 0 }], phase: 'loop', stride: 1 });
  return { id: 'sequential', n, steps, threads: 1 };
}

/** Fig. 11.2: strides 1, 2, 4, ...; every position i >= stride adds position i - stride. Thread i owns position i. */
export function koggeStone(n: number): Network {
  const steps: Step[] = [];
  for (let s = 1; s < n; s *= 2) {
    const adders: Adder[] = [];
    for (let i = s; i < n; i++) adders.push({ src: i - s, dst: i, thread: i });
    steps.push({ adders, phase: 'sweep', stride: s });
  }
  return { id: 'kogge-stone', n, steps, threads: n };
}

/** Fig. 11.19: the reduction tree (up-sweep), then the reverse tree (down-sweep). N must be a power of two. */
export function brentKung(n: number): Network {
  if (!isPow2(n) || n < 2) throw new Error('Brent-Kung needs a power of two, at least 2');
  const steps: Step[] = [];
  for (let s = 1; s < n; s *= 2) {
    const adders: Adder[] = [];
    for (let t = 0; (t + 1) * 2 * s - 1 < n; t++) {
      const idx = (t + 1) * 2 * s - 1;
      adders.push({ src: idx - s, dst: idx, thread: t });
    }
    steps.push({ adders, phase: 'up', stride: s });
  }
  let level = 1;
  for (let s = n / 4; s >= 1; s /= 2) {
    const adders: Adder[] = [];
    for (let t = 0; (t + 1) * 2 * s - 1 + s < n; t++) {
      const idx = (t + 1) * 2 * s - 1;
      adders.push({ src: idx, dst: idx + s, thread: t });
    }
    steps.push({ adders, phase: 'down', stride: s, level: level++ });
  }
  return { id: 'brent-kung', n, steps, threads: n / 2 };
}

export function network(id: NetId, n: number): Network {
  return id === 'sequential' ? sequential(n) : id === 'kogge-stone' ? koggeStone(n) : brentKung(n);
}

/** Number of up-sweep steps of a Brent-Kung network (0 for the others). */
export const upSteps = (net: Network) => net.steps.filter((s) => s.phase === 'up').length;

// ---------------------------------------------------------------- running a network
/** The array before any step (index 0) and after each step. Every adder of a step reads the values from before it. */
export function run(net: Network, values: readonly number[]): number[][] {
  if (values.length !== net.n) throw new Error(`need ${net.n} values`);
  const out: number[][] = [[...values]];
  for (const step of net.steps) {
    const prev = out[out.length - 1];
    const next = [...prev];
    for (const a of step.adders) next[a.dst] = prev[a.dst] + prev[a.src];
    out.push(next);
  }
  return out;
}

/** An inclusive scan the way Fig. 11.1 computes it, as the reference. */
export function inclusiveScan(values: readonly number[]): number[] {
  const out: number[] = [];
  let run = 0;
  for (const v of values) out.push((run += v));
  return out;
}

/** The inputs x(lo)..x(hi) a position holds. */
export interface Range {
  lo: number;
  hi: number;
}
export const rangeLabel = (r: Range) => (r.lo === r.hi ? `x${r.lo}` : `x${r.lo}..x${r.hi}`);

/**
 * The range every position holds before any step and after each step. Throws if an adder
 * would join two ranges that are not adjacent: in both networks they always are (p. 283).
 */
export function ranges(net: Network): Range[][] {
  const out: Range[][] = [Array.from({ length: net.n }, (_, i) => ({ lo: i, hi: i }))];
  for (const step of net.steps) {
    const prev = out[out.length - 1];
    const next = prev.map((r) => ({ ...r }));
    for (const a of step.adders) {
      const s = prev[a.src];
      const d = prev[a.dst];
      if (s.hi + 1 !== d.lo) throw new Error(`step joins ${rangeLabel(s)} and ${rangeLabel(d)}`);
      next[a.dst] = { lo: s.lo, hi: d.hi };
    }
    out.push(next);
  }
  return out;
}

/** Positions that hold their final value, x0..x(i), after k steps. */
export function finalAfter(net: Network, k: number): number[] {
  const r = ranges(net)[k];
  return r.flatMap((x, i) => (x.lo === 0 && x.hi === i ? [i] : []));
}

/** The step (1-based) after which each position first holds its final value; 0 if it starts final. */
export function finalAt(net: Network): number[] {
  const all = ranges(net);
  return Array.from({ length: net.n }, (_, i) => all.findIndex((r) => r[i].lo === 0));
}

/**
 * Positions that one adder of the step writes and another adder of the same step reads. An
 * in-place kernel with one barrier per step is safe only when this is empty: otherwise a
 * thread may overwrite a value before another thread has read it (pp. 257-258).
 */
export function readWriteOverlap(step: Step): number[] {
  const written = new Map(step.adders.map((a) => [a.dst, a]));
  const out = new Set<number>();
  for (const a of step.adders) {
    const w = written.get(a.src);
    if (w && w !== a) out.add(a.src);
  }
  return [...out].sort((x, y) => x - y);
}

// ---------------------------------------------------------------- counts and closed forms
export const additions = (net: Network) => net.steps.reduce((sum, s) => sum + s.adders.length, 0);
export const addersPerStep = (net: Network) => net.steps.map((s) => s.adders.length);

/** The book's closed forms (pp. 253-254, 265-266, 284). N a power of two for the parallel ones. */
export function closedForm(id: NetId | 'naive', n: number): number {
  switch (id) {
    case 'sequential':
      return n - 1;
    case 'naive':
      return (n * (n - 1)) / 2;
    case 'kogge-stone':
      return n * lg(n) - (n - 1);
    case 'brent-kung':
      return 2 * n - 2 - lg(n);
  }
}
export const CLOSED_FORM_TEXT: Record<NetId | 'naive', string> = {
  sequential: 'N − 1',
  naive: 'N(N − 1)/2',
  'kogge-stone': 'N·log₂N − (N − 1)', // i18n-ignore: closed-form formula
  'brent-kung': '2N − 2 − log₂N', // i18n-ignore: closed-form formula
};

/**
 * Steps with unlimited execution units. The loop: about N, as the book counts it (p. 266:
 * "approximately N steps"). Kogge-Stone: log2N. Brent-Kung: log2N up, log2N - 1 down.
 */
export function span(id: NetId, n: number): number {
  return id === 'sequential' ? n : id === 'kogge-stone' ? lg(n) : 2 * lg(n) - 1;
}

// ---------------------------------------------------------------- steps with P units
/** One step's load: how many adders, and the threads lo..hi that perform them (always a contiguous run). */
export interface StepLoad {
  adds: number;
  lo: number;
  hi: number;
}

/**
 * Per-step loads without building the network, so N can reach 2^20. Kogge-Stone: threads
 * stride..N-1. Brent-Kung: threads 0..adds-1. tests check these against the networks.
 */
export function loads(id: 'kogge-stone' | 'brent-kung', n: number): StepLoad[] {
  const out: StepLoad[] = [];
  if (id === 'kogge-stone') {
    for (let s = 1; s < n; s *= 2) out.push({ adds: n - s, lo: s, hi: n - 1 });
    return out;
  }
  for (let s = 1; s < n; s *= 2) out.push({ adds: n / (2 * s), lo: 0, hi: n / (2 * s) - 1 });
  for (let s = n / 4; s >= 1; s /= 2) out.push({ adds: n / (2 * s) - 1, lo: 0, hi: n / (2 * s) - 2 });
  return out;
}

export const threadsOf = (id: NetId, n: number) => (id === 'sequential' ? 1 : id === 'kogge-stone' ? n : n / 2);

export type Charge = 'adds' | 'warps' | 'threads';
export const CHARGE_LABEL: Record<Charge, string> = {
  get adds() {
    return tr('its additions', 'các phép cộng thực có của nó');
  },
  get warps() {
    return tr('whole warps', 'trọn warp');
  },
  get threads() {
    return tr('every thread (the book)', 'mọi luồng (theo sách)');
  },
};

/** Lane slots one step is charged under a rule. */
export function chargeOf(load: StepLoad, charge: Charge, threads: number, warp: number): number {
  if (charge === 'adds') return load.adds;
  if (charge === 'threads') return threads;
  if (load.adds === 0) return 0;
  return warp * (Math.floor(load.hi / warp) - Math.floor(load.lo / warp) + 1);
}

/** Lane slots the whole scan is charged: its resource use under the rule. */
export function slotsTotal(id: 'kogge-stone' | 'brent-kung', n: number, charge: Charge, warp = 32): number {
  return loads(id, n).reduce((sum, l) => sum + chargeOf(l, charge, threadsOf(id, n), warp), 0);
}

/**
 * Steps with P execution units: each step costs ceil(charge / P), and a step starts when the
 * previous one has finished. The loop is the sequential reference: N steps, whatever P.
 * Under 'threads' Kogge-Stone gives log2N · ceil(N/P), which is the book's (N·log2N)/P
 * whenever P divides N.
 */
export function stepsWithUnits(id: NetId, n: number, p: number, charge: Charge, warp = 32): number {
  if (id === 'sequential') return span('sequential', n);
  const t = threadsOf(id, n);
  return loads(id, n).reduce((sum, l) => sum + Math.ceil(chargeOf(l, charge, t, warp) / p), 0);
}

/** The book's estimate for Kogge-Stone, (N·log2N)/P (p. 266). */
export const bookKoggeStoneSteps = (n: number, p: number) => (n * lg(n)) / p;

/**
 * The smallest number of execution units with which the network needs fewer steps than the
 * loop, or null if none up to `maxP` does. Steps never grow with P, so a binary search works.
 */
export function breakEven(id: 'kogge-stone' | 'brent-kung', n: number, charge: Charge, warp = 32, maxP = 1 << 22): number | null {
  const loop = span('sequential', n);
  if (stepsWithUnits(id, n, maxP, charge, warp) >= loop) return null;
  let lo = 1;
  let hi = maxP;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (stepsWithUnits(id, n, mid, charge, warp) < loop) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

// ---------------------------------------------------------------- one warp
export interface WarpStep {
  phase: Phase;
  stride: number;
  /** Lanes that add in this step (lane = destination position). */
  lanes: number[];
  adds: number;
  idle: number;
}
export interface WarpView {
  id: 'kogge-stone' | 'brent-kung';
  width: number;
  steps: WarpStep[];
  additions: number;
  /** Lane slots the warp occupies: every step issues for all its lanes. */
  slots: number;
  idleSlots: number;
  /** The step after which the scan is complete. */
  finishesAt: number;
}

/** One warp scanning one value per lane: who adds and who idles in each step (pp. 284-285). */
export function warpView(id: 'kogge-stone' | 'brent-kung', width: number): WarpView {
  const net = network(id, width);
  const steps = net.steps.map((s) => {
    const lanes = s.adders.map((a) => a.dst).sort((a, b) => a - b);
    return { phase: s.phase, stride: s.stride, lanes, adds: lanes.length, idle: width - lanes.length };
  });
  const adds = steps.reduce((x, s) => x + s.adds, 0);
  return { id, width, steps, additions: adds, slots: steps.length * width, idleSlots: steps.length * width - adds, finishesAt: steps.length };
}

// ---------------------------------------------------------------- Fig. 11.20
/**
 * The positions whose ranges a position still has to add, nearest first: what x(lo-1)
 * holds, then what the position before that range holds, and so on down to x0. Position 14
 * after the reduction tree: 13 (x12..x13), 11 (x8..x11), 7 (x0..x7), as on pp. 283-284.
 */
export function needsChain(state: readonly Range[], i: number): number[] {
  const out: number[] = [];
  let lo = state[i].lo;
  while (lo > 0) {
    out.push(lo - 1);
    lo = state[lo - 1].lo;
  }
  return out;
}
export const needs = (state: readonly Range[]) => state.map((_, i) => needsChain(state, i).length);

export interface ReverseLevel {
  /** "Initial" (after the reduction tree), then "Level 1", "Level 2", ... */
  label: string;
  /** Network steps completed at this row. */
  step: number;
  ranges: Range[];
  needs: number[];
  /** Positions this level wrote (empty for Initial). */
  changed: number[];
}

/** Fig. 11.20's rows for a Brent-Kung network of N elements. */
export function reverseLevels(n: number): ReverseLevel[] {
  const net = brentKung(n);
  const all = ranges(net);
  const up = upSteps(net);
  const rows: ReverseLevel[] = [{ get label() { return tr('Initial', 'Ban đầu'); }, step: up, ranges: all[up], needs: needs(all[up]), changed: [] }];
  for (let k = up; k < net.steps.length; k++)
    rows.push({
      label: tr(`Level ${net.steps[k].level}`, `Mức ${net.steps[k].level}`),
      step: k + 1,
      ranges: all[k + 1],
      needs: needs(all[k + 1]),
      changed: net.steps[k].adders.map((a) => a.dst),
    });
  return rows;
}

/**
 * Fig. 11.20's shading as printed (p. 284), 0 = white, 1 = light grey, 2 = dark grey,
 * 3 = black (legend on p. 283), rows Initial, Level 1, Level 2, Level 3. From notes/ch11.md.
 */
export const FIG_11_20_PRINTED: readonly (readonly number[])[] = [
  [0, 0, 1, 0, 1, 1, 2, 0, 1, 2, 2, 2, 2, 2, 3, 0],
  [0, 0, 1, 0, 1, 1, 2, 0, 1, 2, 2, 0, 1, 1, 2, 0],
  [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];

/** Cells where the printed shading disagrees with the needs the ranges imply. */
export function printedDiffers(): { row: number; col: number; printed: number; needs: number }[] {
  const rows = reverseLevels(16);
  const out: { row: number; col: number; printed: number; needs: number }[] = [];
  rows.forEach((r, k) => r.needs.forEach((v, col) => {
    if (FIG_11_20_PRINTED[k][col] !== v) out.push({ row: k, col, printed: FIG_11_20_PRINTED[k][col], needs: v });
  }));
  return out;
}

// ---------------------------------------------------------------- the book's inputs
/** Exercises 1 and 6 (p. 286). */
export const EX6: readonly number[] = [4, 6, 7, 1, 2, 8, 5, 2];
/** Fig. 11.6's input row (p. 262): sixteen values whose scan the book prints. */
export const FIG116: readonly number[] = [2, 1, 3, 1, 0, 4, 1, 2, 0, 3, 1, 2, 5, 3, 1, 2];

/** A Qwen3-8B vocabulary row (notes/today/ch09-facts.md D7) and the power of two just below it. */
export const VOCAB_QWEN3_8B = 151936;
export const VOCAB_POW2 = 1 << 17;
