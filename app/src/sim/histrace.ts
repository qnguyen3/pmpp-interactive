// Interleavings of a plain increment on one bin, for section 9.2's stepper and
// enumerator.
//
// Each of n threads (2 to 4) runs the book's three steps on bins[x] (Figs. 9.4, 9.5,
// pp. 205-206):  Old <- bins[x];  New <- Old + 1;  bins[x] <- New.
// A schedule is the order in which threads take their next step: [0, 0, 1, 0, 1, 1]
// means thread 1 (index 0) loads and adds, thread 2 loads, thread 1 stores, thread 2
// adds and stores. In atomic mode a thread's three steps are one indivisible step, so a
// schedule names each thread once.
//
// Besides the numbers, the model tracks WHICH increments each value contains: a load
// copies the set held in memory, the add puts the thread's own increment in, a store
// writes the set back. The final set names exactly the updates that survived, so the
// stepper can say whose update was overwritten, even when a later stale store happens
// to restore it.
//
// What it simplifies: one source-level step at a time, no timing, no caches. Formally a
// concurrent plain ++ is a data race and undefined behaviour (notes/today/ch09-facts.md
// A2, A3); the lost update is the typical outcome, not a promise.

import { tr } from '../lib/locale.ts';

export type Op = 'load' | 'add' | 'store' | 'atomic';
export const OPS: Op[] = ['load', 'add', 'store'];
/** The book's wording for each step. */
export const OP_TEXT: Record<Op, string> = {
  load: 'Old ← bins[x]', // i18n-ignore: the book's pseudocode, shown as written
  add: 'New ← Old + 1', // i18n-ignore: the book's pseudocode, shown as written
  store: 'bins[x] ← New', // i18n-ignore: the book's pseudocode, shown as written
  get atomic() {
    return tr('atomic: Old ← bins[x]; bins[x] ← Old + 1', 'nguyên tử: Old ← bins[x]; bins[x] ← Old + 1');
  },
};

export interface Options {
  /** Number of threads, 2 to 4. */
  n: number;
  /** Make each thread's three steps one indivisible step. */
  atomic?: boolean;
  /** bins[x] before anyone runs. The book starts at 0. */
  initial?: number;
}

export interface Regs {
  old: number | null;
  new: number | null;
}

export interface Step {
  /** 1-based, as in the book's tables. */
  time: number;
  thread: number;
  op: Op;
  /** The value in parentheses in the book's tables: the value loaded, computed or stored. */
  shown: number;
  /** bins[x] after this step. */
  memory: number;
  /** For a store: the threads whose increments were in memory and are not in what was written. */
  erased: number[];
  /** Every thread's registers after this step. */
  regs: Regs[];
}

export interface Run {
  steps: Step[];
  /** bins[x] after the last step of the schedule. */
  final: number;
  /** Every thread has finished. */
  complete: boolean;
  /** Threads whose increment is in the final value. */
  survived: number[];
  /** Threads that finished but whose increment is not in the final value. */
  lost: number[];
  /** Complete and nothing lost. */
  correct: boolean;
}

const stepsPerThread = (o: Options) => (o.atomic ? 1 : 3);
const members = (mask: number, n: number) => Array.from({ length: n }, (_, t) => t).filter((t) => mask & (1 << t));
const popcount = (m: number) => {
  let c = 0;
  for (let x = m; x; x &= x - 1) c++;
  return c;
};

function check(o: Options) {
  if (!Number.isInteger(o.n) || o.n < 1 || o.n > 8) throw new Error(`n = ${o.n}: the model takes 1 to 8 threads`);
}

/** How many steps each thread has taken in a (possibly partial) schedule. */
export function progressOf(schedule: readonly number[], o: Options): number[] {
  check(o);
  const done = new Array<number>(o.n).fill(0);
  for (const t of schedule) {
    if (!Number.isInteger(t) || t < 0 || t >= o.n) throw new Error(`thread ${t} does not exist`);
    done[t] += 1;
    if (done[t] > stepsPerThread(o)) throw new Error(`thread ${t + 1} has no step left`);
  }
  return done;
}

/** The step a thread would take next, or null if it has finished. */
export function nextOp(schedule: readonly number[], thread: number, o: Options): Op | null {
  const k = progressOf(schedule, o)[thread];
  if (o.atomic) return k === 0 ? 'atomic' : null;
  return k < 3 ? OPS[k] : null;
}

export function run(schedule: readonly number[], o: Options): Run {
  const done = progressOf(schedule, o);
  const n = o.n;
  const initial = o.initial ?? 0;
  const pos = new Array<number>(n).fill(0);
  const oldSet = new Array<number>(n).fill(0);
  const newSet = new Array<number>(n).fill(0);
  const regs: Regs[] = Array.from({ length: n }, () => ({ old: null, new: null }));
  let mem = 0; // the set of increments the memory value contains
  const steps: Step[] = [];
  schedule.forEach((t, k) => {
    let op: Op;
    let shown: number;
    let erased: number[] = [];
    if (o.atomic) {
      op = 'atomic';
      oldSet[t] = mem;
      newSet[t] = mem | (1 << t);
      mem = newSet[t];
      regs[t] = { old: initial + popcount(oldSet[t]), new: initial + popcount(newSet[t]) };
      shown = regs[t].new!;
    } else {
      op = OPS[pos[t]];
      if (op === 'load') {
        oldSet[t] = mem;
        regs[t] = { old: initial + popcount(mem), new: null };
        shown = regs[t].old!;
      } else if (op === 'add') {
        newSet[t] = oldSet[t] | (1 << t);
        regs[t] = { ...regs[t], new: regs[t].old! + 1 };
        shown = regs[t].new!;
      } else {
        erased = members(mem & ~newSet[t], n);
        mem = newSet[t];
        shown = regs[t].new!;
      }
    }
    pos[t] += 1;
    steps.push({ time: k + 1, thread: t, op, shown, memory: initial + popcount(mem), erased, regs: regs.map((r) => ({ ...r })) });
  });
  const complete = done.every((d) => d === stepsPerThread(o));
  const finished = done.map((d, t) => (d === stepsPerThread(o) ? t : -1)).filter((t) => t >= 0);
  const survived = members(mem, n);
  const lost = finished.filter((t) => !(mem & (1 << t)));
  return { steps, final: initial + popcount(mem), complete, survived, lost, correct: complete && lost.length === 0 };
}

// ------------------------------------------------------------------ presets
export type PresetId = 't1-first' | 'overlap-t1' | 't2-first' | 'overlap-t2' | 'lockstep';

export interface Preset {
  id: PresetId;
  label: string;
  /** Where its outcome is drawn in the book, if anywhere. */
  book: string | null;
  /** Threads it needs: the book's tables have two; lockstep takes any count. */
  n: number | null;
  schedule: (n: number) => number[];
}

export const PRESETS: Preset[] = [
  {
    id: 't1-first',
    get label() {
      return tr('thread 1 first', 'luồng 1 chạy trước');
    },
    book: 'Fig. 9.4(a)', // i18n-ignore: citation
    n: 2, schedule: () => [0, 0, 0, 1, 1, 1],
  },
  {
    id: 'overlap-t1',
    get label() {
      return tr('overlapped, thread 1 ahead', 'xen kẽ, luồng 1 đi trước');
    },
    book: 'Fig. 9.4(b)', // i18n-ignore: citation
    n: 2, schedule: () => [0, 0, 1, 0, 1, 1],
  },
  {
    id: 't2-first',
    get label() {
      return tr('thread 2 first', 'luồng 2 chạy trước');
    },
    book: 'Fig. 9.5(a)', // i18n-ignore: citation
    n: 2, schedule: () => [1, 1, 1, 0, 0, 0],
  },
  {
    id: 'overlap-t2',
    get label() {
      return tr('overlapped, thread 2 ahead', 'xen kẽ, luồng 2 đi trước');
    },
    book: 'Fig. 9.5(b)', // i18n-ignore: citation
    n: 2, schedule: () => [1, 1, 0, 1, 0, 0],
  },
  {
    id: 'lockstep',
    get label() {
      return tr('lockstep warp', 'warp chạy đồng bộ (lockstep)');
    },
    book: null,
    n: null,
    // Every thread loads, then every thread adds, then every thread stores.
    schedule: (n) => [0, 1, 2].flatMap(() => Array.from({ length: n }, (_, t) => t)),
  },
];

/** A preset's schedule in atomic mode: each thread once, in the order it first appears. */
export function atomicOrder(schedule: readonly number[]): number[] {
  return [...new Set(schedule)];
}

// ------------------------------------------------------------------ every interleaving
export interface Census {
  n: number;
  atomic: boolean;
  /** Number of interleavings that keep each thread's own order: (3n)! / (3!)^n, or n! in atomic mode. */
  interleavings: number;
  /** How many end at each final value (index = final value; the bin starts at 0). */
  byFinal: number[];
  /** How many end at n. */
  correct: number;
}

/**
 * Runs every interleaving of n threads on a bin that starts at 0. Brute force over a
 * bitmask state: 369,600 complete runs for n = 4.
 */
export function enumerate(n: number, opts: { atomic?: boolean } = {}): Census {
  check({ n });
  const atomic = !!opts.atomic;
  const per = atomic ? 1 : 3;
  const byFinal = new Array<number>(n + 1).fill(0);
  const pos = new Array<number>(n).fill(0);
  const oldSet = new Array<number>(n).fill(0);
  let total = 0;
  const go = (mem: number, left: number) => {
    if (left === 0) {
      byFinal[popcount(mem)] += 1;
      total += 1;
      return;
    }
    for (let t = 0; t < n; t++) {
      const p = pos[t];
      if (p === per) continue;
      pos[t] = p + 1;
      if (atomic) go(mem | (1 << t), left - 1);
      else if (p === 0) {
        const keep = oldSet[t];
        oldSet[t] = mem;
        go(mem, left - 1);
        oldSet[t] = keep;
      } else if (p === 1) go(mem, left - 1);
      else go(oldSet[t] | (1 << t), left - 1);
      pos[t] = p;
    }
  };
  go(0, n * per);
  return { n, atomic, interleavings: total, byFinal, correct: byFinal[n] };
}

/** (3n)! / (3!)^n, the closed form for the number of interleavings. */
export function interleavingCount(n: number, per = 3): number {
  let r = 1;
  let k = 0;
  for (let t = 0; t < n; t++) for (let j = 1; j <= per; j++) r = (r * ++k) / j;
  return Math.round(r);
}
