// Odd-even transposition sort, chapter 14 section 14.2 (pp. 331-333, Figs. 14.1 and
// 14.2). Runs the section's stepper, its "neighbouring pairs at once" bug hunt and the
// narrated scene "Why odd-even sort needs N steps".
//
// What it models, exactly:
//  - One step of Fig. 14.2: floor(n/2) threads are launched (the book launches half as
//    many threads as elements); thread t owns i = 2t + p, where p is 0 on an even step
//    and 1 on an odd step, and works only if i + 1 < n. It compares data[i] with
//    data[i+1] and swaps them when the first is larger; a thread that swaps sets the flag.
//    Pairs within a step are disjoint, so the order of the threads cannot matter, and the
//    model runs them one after another.
//  - The host loop the book does not show (notes/ch14.md, "Unclear in the PDF" item 7),
//    with two stopping rules: the book read literally ("until no elements are swapped":
//    stop after the first quiet step) and ours (stop after two steps in a row, one even and
//    one odd in either order, are both quiet). Only the second guarantees a sorted list:
//    [1, 3, 2, 4] makes no swap on its first even step. Lab 14.1's loop, which reads the
//    flag once per even-odd pair of steps, is pairedLaunches() below.
//  - The naive alternative, where thread t owns (t, t+1) for every t in the same step.
//    Here each thread's two loads and two stores are separate operations, executed in an
//    interleaving the caller picks, so a lost or duplicated value can be shown.
//
// What it leaves out: one thread's step (or one load or store, in the bug hunt) is
// atomic; real threads interleave at the instruction level and run in warps. No timing,
// no launch cost, no cost of divergence at the swap branch (undocumented on Metal,
// notes/today/ch04-facts.md C4).
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

export type Parity = 0 | 1;

/** What one thread did in one step. */
export interface ThreadStep {
  t: number;
  /** First index of its pair: 2t + parity. */
  i: number;
  /** i + 1 < n: the pair is in range and the thread compares it. */
  active: boolean;
  /** The two values compared (when active). */
  a?: number;
  b?: number;
  swapped: boolean;
}

export interface StepResult {
  /** 1-based step number in a run. */
  n: number;
  parity: Parity;
  before: number[];
  after: number[];
  threads: ThreadStep[];
  comparisons: number;
  swaps: number;
  /** The flag after the step: some thread swapped. */
  flag: boolean;
}

/** Threads launched per step: half as many as elements (the book's launch, p. 331). */
export const threadsFor = (n: number) => Math.floor(n / 2);

/** One launch of Fig. 14.2 on `list` with the given parity. */
export function step(list: number[], parity: Parity, num = 1): StepResult {
  const after = list.slice();
  const threads: ThreadStep[] = [];
  let comparisons = 0;
  let swaps = 0;
  for (let t = 0; t < threadsFor(list.length); t++) {
    const i = 2 * t + parity;
    const active = i + 1 < list.length;
    if (!active) {
      threads.push({ t, i, active, swapped: false });
      continue;
    }
    comparisons++;
    const a = after[i];
    const b = after[i + 1];
    const swapped = a > b;
    if (swapped) {
      after[i] = b;
      after[i + 1] = a;
      swaps++;
    }
    threads.push({ t, i, active, a, b, swapped });
  }
  return { n: num, parity, before: list.slice(), after, threads, comparisons, swaps, flag: swaps > 0 };
}

export const isAscending = (xs: number[]) => xs.every((v, k) => k === 0 || xs[k - 1] <= v);

/** 'book': stop after the first step with no swap. 'ours': stop after two quiet steps in a row (one even, one odd, in either order). */
export type StopRule = 'book' | 'ours';
export const STOP_LABEL: Record<StopRule, string> = {
  get book() {
    return tr('after one quiet step (the book, literally)', 'sau một bước yên lặng (đúng nguyên văn sách)');
  },
  get ours() {
    return tr('after two quiet steps in a row, one even and one odd (ours)', 'sau hai bước yên lặng liên tiếp, một chẵn và một lẻ (của khóa học này)');
  },
};

export interface Run {
  input: number[];
  rule: StopRule;
  rows: StepResult[];
  /** Steps the host launched before stopping. */
  launched: number;
  /** The list when the host stopped. */
  out: number[];
  /** Is it sorted? (Under the book's literal rule it may not be.) */
  sorted: boolean;
  /** The first step after which the list was sorted (0: it already was); null if never. */
  sortedAfter: number | null;
  comparisons: number;
  swaps: number;
}

/** The host loop: alternate steps from `start` until the stopping rule fires. */
export function runToSorted(list: number[], rule: StopRule, start: Parity = 0, maxSteps = 4 * list.length + 8): Run {
  const rows: StepResult[] = [];
  let cur = list.slice();
  let quiet = 0;
  let sortedAfter: number | null = isAscending(cur) ? 0 : null;
  if (list.length >= 2) {
    for (let s = 0; s < maxSteps; s++) {
      const r = step(cur, ((start + s) % 2) as Parity, s + 1);
      rows.push(r);
      cur = r.after;
      if (sortedAfter === null && isAscending(cur)) sortedAfter = s + 1;
      quiet = r.flag ? 0 : quiet + 1;
      if (quiet >= (rule === 'book' ? 1 : 2)) break;
    }
  }
  return {
    input: list.slice(),
    rule,
    rows,
    launched: rows.length,
    out: cur,
    sorted: isAscending(cur),
    sortedAfter,
    comparisons: rows.reduce((s, r) => s + r.comparisons, 0),
    swaps: rows.reduce((s, r) => s + r.swaps, 0),
  };
}

/**
 * Lab 14.1's host loop: clear the flag, launch an even step and the odd step after it, read
 * the flag once for the pair, and stop when the pair was quiet. Half the reads of
 * runToSorted(list, 'ours'), at the price of up to one more step, because two quiet steps
 * that straddle two pairs (odd, then even) do not stop it. Returns the steps launched.
 */
export function pairedLaunches(list: number[], maxSteps = 4 * list.length + 8): number {
  if (list.length < 2) return 0;
  let cur = list.slice();
  let launched = 0;
  while (launched < maxSteps) {
    const even = step(cur, 0, launched + 1);
    const odd = step(even.after, 1, launched + 2);
    cur = odd.after;
    launched += 2;
    if (!even.flag && !odd.flag) break;
  }
  return launched;
}

/** A fixed number of alternating steps, whatever happens (for the scene). */
export function walk(list: number[], steps: number, start: Parity = 0): StepResult[] {
  const rows: StepResult[] = [];
  let cur = list.slice();
  for (let s = 0; s < steps; s++) {
    const r = step(cur, ((start + s) % 2) as Parity, s + 1);
    rows.push(r);
    cur = r.after;
  }
  return rows;
}

/** The book's bound (p. 333): N steps of N/2 comparisons. */
export const bookBound = (n: number) => (n * n) / 2;
/** N log2 N, the comparison-sort scale it is compared with. */
export const nLogN = (n: number) => n * Math.log2(n);

/** Fig. 14.1 (p. 332): the input and the six rows after it. */
export const FIG_14_1 = [4, 7, 2, 3, 8, 5, 9, 6];
export const FIG_14_1_ROWS = [
  [4, 7, 2, 3, 8, 5, 9, 6],
  [4, 7, 2, 3, 5, 8, 6, 9],
  [4, 2, 7, 3, 5, 6, 8, 9],
  [2, 4, 3, 7, 5, 6, 8, 9],
  [2, 3, 4, 5, 7, 6, 8, 9],
  [2, 3, 4, 5, 6, 7, 8, 9],
  [2, 3, 4, 5, 6, 7, 8, 9],
];
export const LARGEST_FIRST = [8, 1, 2, 3, 4, 5, 6, 7];
export const TRAP = [1, 3, 2, 4, 5, 6, 7, 8];

export const PRESETS: { id: string; label: string; values: number[] }[] = [
  {
    id: 'fig',
    get label() {
      return tr('Fig. 14.1: 4 7 2 3 8 5 9 6', 'Hình 14.1: 4 7 2 3 8 5 9 6');
    },
    values: FIG_14_1,
  },
  {
    id: 'reversed',
    get label() {
      return tr('reversed: 9 8 7 6 5 4 3 2', 'đảo ngược: 9 8 7 6 5 4 3 2');
    },
    values: [9, 8, 7, 6, 5, 4, 3, 2],
  },
  {
    id: 'sorted',
    get label() {
      return tr('already sorted', 'đã sắp xếp sẵn');
    },
    values: [2, 3, 4, 5, 6, 7, 8, 9],
  },
  {
    id: 'largest',
    get label() {
      return tr('largest first: 8 1 2 3 4 5 6 7', 'lớn nhất đứng đầu: 8 1 2 3 4 5 6 7');
    },
    values: LARGEST_FIRST,
  },
  {
    id: 'trap',
    get label() {
      return tr('the trap: 1 3 2 4 5 6 7 8', 'cái bẫy: 1 3 2 4 5 6 7 8');
    },
    values: TRAP,
  },
];

/** Reads "4 7 2 3" or "4, 7, 2, 3": up to `max` non-negative integers below 100; null if malformed. */
export function parseValues(text: string, max = 8): number[] | null {
  const parts = text.split(/[\s,]+/).filter(Boolean);
  if (parts.length < 2 || parts.length > max) return null;
  const xs = parts.map(Number);
  return xs.every((v) => Number.isInteger(v) && v >= 0 && v < 100) ? xs : null;
}

// ---------------------------------------------------------------- neighbouring pairs at once
/** The four operations of one naive thread: load data[t], load data[t+1] (and compare), store both. */
export type OpKind = 'loadA' | 'loadB' | 'storeA' | 'storeB';
export const OPS: OpKind[] = ['loadA', 'loadB', 'storeA', 'storeB'];

export type Interleaving = { kind: 'in-order' } | { kind: 'reversed' } | { kind: 'race'; at: number };

export interface Regs {
  a?: number;
  b?: number;
  /** Decided after the second load. */
  swap?: boolean;
  done: boolean;
}

export interface OverlapEvent {
  t: number;
  op: OpKind;
  /** The element read or written. */
  index: number;
  /** The value read or written; undefined for a store that was skipped (no swap). */
  value?: number;
  skipped: boolean;
  /** The memory after this event, and every thread's registers. */
  data: number[];
  regs: Regs[];
}

export interface OverlapRun {
  input: number[];
  events: OverlapEvent[];
  out: number[];
  /** Values with fewer copies in the output than in the input. */
  lost: number[];
  /** Values with more copies in the output than in the input. */
  duplicated: number[];
  permutation: boolean;
}

/** The order in which the n-1 naive threads run their four operations. */
export function schedule(n: number, il: Interleaving): { t: number; op: OpKind }[] {
  const T = Math.max(0, n - 1);
  const whole = (t: number) => OPS.map((op) => ({ t, op }));
  if (il.kind === 'in-order') return Array.from({ length: T }, (_, t) => whole(t)).flat();
  if (il.kind === 'reversed') return Array.from({ length: T }, (_, k) => whole(T - 1 - k)).flat();
  // Two threads share element `at`: thread at-1 owns (at-1, at), thread at owns (at, at+1).
  // Both load before either stores; the left one stores first.
  const L = il.at - 1;
  const R = il.at;
  if (L < 0 || R >= T) throw new Error('race needs 1 <= at <= n - 2');
  const out: { t: number; op: OpKind }[] = [];
  for (let t = 0; t < L; t++) out.push(...whole(t));
  out.push({ t: L, op: 'loadA' }, { t: L, op: 'loadB' }, { t: R, op: 'loadA' }, { t: R, op: 'loadB' });
  out.push({ t: L, op: 'storeA' }, { t: L, op: 'storeB' }, { t: R, op: 'storeA' }, { t: R, op: 'storeB' });
  for (let t = R + 1; t < T; t++) out.push(...whole(t));
  return out;
}

/** Runs one naive step, thread t owning (t, t+1) for every t, in the given interleaving. */
export function overlapStep(list: number[], il: Interleaving): OverlapRun {
  const data = list.slice();
  const T = Math.max(0, list.length - 1);
  const regs: Regs[] = Array.from({ length: T }, () => ({ done: false }));
  const events: OverlapEvent[] = [];
  for (const { t, op } of schedule(list.length, il)) {
    const r = regs[t];
    let index = t;
    let value: number | undefined;
    let skipped = false;
    if (op === 'loadA') value = r.a = data[t];
    else if (op === 'loadB') {
      index = t + 1;
      value = r.b = data[t + 1];
      r.swap = (r.a as number) > r.b;
    } else if (op === 'storeA') {
      if (r.swap) value = data[t] = r.b as number;
      else skipped = true;
    } else {
      index = t + 1;
      if (r.swap) value = data[t + 1] = r.a as number;
      else skipped = true;
      r.done = true;
    }
    events.push({ t, op, index, value, skipped, data: data.slice(), regs: regs.map((x) => ({ ...x })) });
  }
  const count = (xs: number[]) => xs.reduce((m, v) => m.set(v, (m.get(v) ?? 0) + 1), new Map<number, number>());
  const cin = count(list);
  const cout = count(data);
  const lost = [...cin].filter(([v, c]) => (cout.get(v) ?? 0) < c).map(([v]) => v);
  const duplicated = [...cout].filter(([v, c]) => (cin.get(v) ?? 0) < c).map(([v]) => v);
  return { input: list.slice(), events, out: data, lost, duplicated, permutation: lost.length === 0 && duplicated.length === 0 };
}
