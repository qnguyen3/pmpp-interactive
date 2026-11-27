// Chapter 4, sections 4.4 and 4.5: one warp executing a small structured program.
// It runs the stepper "One fetch, many lanes" of section 4.4, the divergence
// simulator, the exercise 1 reveal and the narrated animation "A warp at a branch"
// of section 4.5, and the numbers in their questions.
//
// Sources for the rules (the lessons cite them where they are used):
//  - the book, section 4.4, pp. 76 to 79: one instruction is fetched and executed
//    for all threads of a warp; the lanes differ only in the contents of their
//    registers (the sidebar's add r1, r2, r3);
//  - the book, section 4.5, pp. 79 to 81: when threads of a warp take different
//    paths the hardware makes one pass per path, and in each pass the threads on
//    the other path are inactive (Fig. 4.9: threads 0 to 23 execute A, 24 to 31
//    execute B); in a loop with per-thread trip counts, threads that have finished
//    are inactive while the others go on (Fig. 4.10); before Volta one pass ran to
//    completion before the other, since then the passes may be interleaved;
//  - the book, exercise 1, pp. 90 to 91: the kernel foo_kernel and its launch;
//  - notes/ch04.md, exercises: "SIMD efficiency" and "active warp" are used by the
//    exercise and defined nowhere in the chapter. The definitions used here are
//    ours: efficiency = lanes that execute the step / warp width; a warp is active
//    for a statement when at least one of its threads executes it.
//
// What this leaves out:
//  - Cost. Every step is one instruction issued to the warp and costs one
//    "warp-step" whatever the number of active lanes. That is the book's picture,
//    not a measurement. NVIDIA publishes no cost for a divergent pass and says
//    hardware "may optimize masked lanes" in ways a program cannot see (facts A9);
//    AMD says the ALUs still process the full warp (facts B7); Apple documents
//    neither (facts C4). Lab 4.2 measures it.
//  - The compiler. A short branch may be compiled into a select with no divergence
//    at all; nothing documents when (facts A9). Here a branch is always a branch.
//  - Branch overhead. The test is one step; the branch itself costs nothing. So select()
//    is never cheaper than ifElse() here (one step more in a split warp), and whether a
//    select wins for short paths on a device is outside the model. No vendor documents
//    that cost.
//  - Loop overhead. A loop costs its body only; the test and the jump are not
//    counted as steps.
//  - Scheduling. "Interleaved" alternates the two paths one instruction at a time.
//    Real hardware with independent thread scheduling may mix them in any way; the
//    totals are the same in this model whatever the order.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- programs
/** `cost` straight-line instructions that every lane executes. */
export interface OpStmt {
  kind: 'op';
  label: string;
  cost: number;
}
/** if (cond) A else B. `taken[lane]` is the condition's value for that lane. */
export interface IfStmt {
  kind: 'if';
  label: string;
  taken: boolean[];
  /** Instructions of A and of B. A path of 0 instructions is no pass at all. */
  costA: number;
  costB: number;
}
/** A loop whose body is `cost` instructions and whose trip count differs per lane. */
export interface LoopStmt {
  kind: 'loop';
  label: string;
  trips: number[];
  cost: number;
}
export type Stmt = OpStmt | IfStmt | LoopStmt;

/** One full warp (every lane holds a thread) and the statements it executes in order. */
export interface Program {
  width: number;
  stmts: Stmt[];
}

export type Order = 'serialised' | 'interleaved';
export const ORDER_LABEL: Record<Order, string> = {
  get serialised() {
    return tr('one path to completion, then the other', 'chạy hết một đường rồi mới đến đường kia');
  },
  get interleaved() {
    return tr('interleaved', 'xen kẽ');
  },
};

/** One instruction issued to the warp: fetched once, executed by the active lanes. */
export interface Issue {
  /** Index into Program.stmts. */
  stmt: number;
  part: 'op' | 'A' | 'B' | 'body';
  /** Which instruction of that part, from 0. */
  k: number;
  /** Loop iteration, from 0; null outside a loop. */
  iter: number | null;
  label: string;
  active: boolean[];
  count: number;
  /** count / width */
  efficiency: number;
}

export interface StmtSummary {
  stmt: number;
  kind: Stmt['kind'];
  /** Passes the warp makes over this statement: paths with lanes on them, or loop iterations. */
  passes: number;
  /** True when some issue of this statement had an idle lane. */
  divergent: boolean;
  steps: number;
  /** Loops only: iterations in which every lane works, and iterations in which some wait. */
  uniformIterations?: number;
  divergentIterations?: number;
}

export interface Trace {
  width: number;
  issues: Issue[];
  /** Warp-steps spent: one per issue. */
  spent: number;
  /** Lane-steps that did work. */
  useful: number;
  /** spent * width */
  capacity: number;
  /** Lane-steps in which a lane waited: capacity - useful. */
  idle: number;
  /** Warp-steps the same work takes when no lane ever waits: useful / width. */
  uniform: number;
  /** useful / capacity */
  efficiency: number;
  /** spent / uniform: how many times the uniform cost this warp paid. */
  overhead: number;
  stmts: StmtSummary[];
}

const issue = (stmt: number, part: Issue['part'], k: number, iter: number | null, label: string, active: boolean[]): Issue => {
  const count = active.filter(Boolean).length;
  return { stmt, part, k, iter, label, active, count, efficiency: active.length ? count / active.length : 0 };
};

const fit = (mask: boolean[], width: number): boolean[] => Array.from({ length: width }, (_, i) => !!mask[i]);

/** Executes the program on one warp and lists every instruction issued, in order. */
export function run(program: Program, order: Order = 'serialised'): Trace {
  const { width } = program;
  const issues: Issue[] = [];
  const stmts: StmtSummary[] = [];
  program.stmts.forEach((s, si) => {
    const before = issues.length;
    let passes = 0;
    let extra: Partial<StmtSummary> = {};
    if (s.kind === 'op') {
      const all = Array.from({ length: width }, () => true);
      for (let k = 0; k < s.cost; k++) issues.push(issue(si, 'op', k, null, s.label, all));
      passes = s.cost > 0 ? 1 : 0;
    } else if (s.kind === 'if') {
      const a = fit(s.taken, width);
      const b = a.map((v) => !v);
      const runA = a.some(Boolean) ? s.costA : 0;
      const runB = b.some(Boolean) ? s.costB : 0;
      passes = Number(runA > 0) + Number(runB > 0);
      const A = (k: number) => issue(si, 'A', k, null, 'A', a);
      const B = (k: number) => issue(si, 'B', k, null, 'B', b);
      if (order === 'serialised') {
        for (let k = 0; k < runA; k++) issues.push(A(k));
        for (let k = 0; k < runB; k++) issues.push(B(k));
      } else {
        for (let k = 0; k < Math.max(runA, runB); k++) {
          if (k < runA) issues.push(A(k));
          if (k < runB) issues.push(B(k));
        }
      }
    } else {
      const trips = Array.from({ length: width }, (_, i) => Math.max(0, Math.floor(s.trips[i] ?? 0)));
      const longest = Math.max(0, ...trips);
      const shortest = Math.min(...trips);
      for (let it = 0; it < longest; it++) {
        const active = trips.map((t) => t > it);
        for (let k = 0; k < s.cost; k++) issues.push(issue(si, 'body', k, it, 'A', active));
      }
      passes = s.cost > 0 ? longest : 0;
      extra = { uniformIterations: shortest, divergentIterations: longest - shortest };
    }
    const mine = issues.slice(before);
    stmts.push({ stmt: si, kind: s.kind, passes, divergent: mine.some((x) => x.count < width), steps: mine.length, ...extra });
  });
  const spent = issues.length;
  const useful = issues.reduce((n, x) => n + x.count, 0);
  const capacity = spent * width;
  const uniform = width ? useful / width : 0;
  return {
    width,
    issues,
    spent,
    useful,
    capacity,
    idle: capacity - useful,
    uniform,
    efficiency: capacity ? useful / capacity : 1,
    overhead: uniform ? spent / uniform : 1,
    stmts,
  };
}

// ---------------------------------------------------------------- conditions
/** threadIdx.x < T for the lanes of the warp that starts at thread `first`. */
export const maskBelow = (width: number, T: number, first = 0): boolean[] =>
  Array.from({ length: width }, (_, i) => first + i < T);
/** i % m == 0 for the lanes of the warp whose lane 0 has index `first`. */
export const maskModulo = (width: number, m: number, first = 0): boolean[] =>
  Array.from({ length: width }, (_, i) => (first + i) % m === 0);
export const countTaken = (mask: boolean[]): number => mask.filter(Boolean).length;

/** The shape of Fig. 4.9: evaluate the condition, then A or B, then C. */
export const ifElse = (width: number, taken: boolean[], costA: number, costB: number, cond = 'condition'): Program => ({
  width,
  stmts: [
    { kind: 'op', label: cond, cost: 1 },
    { kind: 'if', label: cond, taken: fit(taken, width), costA, costB },
    { kind: 'op', label: 'C', cost: 1 },
  ],
});

/** The shape of Fig. 4.10: read N, then loop N times. */
export const loop = (width: number, trips: number[], cost = 1): Program => ({
  width,
  stmts: [
    { kind: 'op', label: 'N = a[threadIdx.x]', cost: 1 },
    { kind: 'loop', label: 'for (i = 0; i < N; ++i) A', trips: Array.from({ length: width }, (_, i) => trips[i] ?? 0), cost },
  ],
});

/**
 * "Compute both, then select": every lane executes A and B and keeps one result.
 * No lane waits; every lane does work it throws away.
 */
export const select = (width: number, costA: number, costB: number, cond = 'condition'): Program => ({
  width,
  stmts: [
    { kind: 'op', label: cond, cost: 1 },
    { kind: 'op', label: 'A', cost: costA },
    { kind: 'op', label: 'B', cost: costB },
    { kind: 'op', label: 'select', cost: 1 }, // i18n-ignore: pseudo-instruction name, like A, B and C
    { kind: 'op', label: 'C', cost: 1 },
  ],
});
/** Lane-steps a select spends on results it discards. */
export const discardedBySelect = (taken: boolean[], costA: number, costB: number): number => {
  const a = countTaken(taken);
  return (taken.length - a) * costA + a * costB;
};

/** Fig. 4.9 as printed (p. 80): a warp of 32, the test threadIdx.x < 24. */
export const FIG49 = { width: 32, threshold: 24 } as const;
/** The trip counts printed under Fig. 4.10's eight columns, left to right (p. 81). */
export const FIG410_PRINTED: readonly number[] = [8, 6, 7, 4, 5, 6, 8, 7];
/** A full warp's worth of trip counts: the figure's eight printed values, repeated. */
export const fig410Trips = (width: number): number[] =>
  Array.from({ length: width }, (_, i) => FIG410_PRINTED[i % FIG410_PRINTED.length]);

/** Which lanes are still iterating in each iteration of a loop: rows of Fig. 4.10's grid. */
export const loopRows = (trips: readonly number[]): boolean[][] =>
  Array.from({ length: Math.max(0, ...trips) }, (_, it) => trips.map((t) => t > it));

// ---------------------------------------------------------------- many warps of loops
export interface WarpLoop {
  trips: number[];
  /** Iterations this warp runs: its longest lane. */
  longest: number;
  /** Lane-iterations that did work. */
  useful: number;
  /** Lane-iterations spent waiting. */
  idle: number;
}
export interface LoopBatch {
  width: number;
  warps: WarpLoop[];
  /** Sum over warps of the iterations each one runs. */
  spent: number;
  useful: number;
  idle: number;
  /** useful / width: what the same trips cost when no lane waits. */
  uniform: number;
  efficiency: number;
}

/** One thread per element of `trips`, cut into warps in order (a 1D launch). */
export function loopBatch(trips: readonly number[], width: number): LoopBatch {
  const warps: WarpLoop[] = [];
  for (let i = 0; i < trips.length; i += width) {
    const t = trips.slice(i, i + width);
    const longest = Math.max(0, ...t);
    const useful = t.reduce((a, b) => a + b, 0);
    warps.push({ trips: t, longest, useful, idle: longest * t.length - useful });
  }
  const spent = warps.reduce((a, w) => a + w.longest, 0);
  const useful = warps.reduce((a, w) => a + w.useful, 0);
  const idle = warps.reduce((a, w) => a + w.idle, 0);
  return { width, warps, spent, useful, idle, uniform: width ? useful / width : 0, efficiency: useful + idle ? useful / (useful + idle) : 1 };
}

/** A small deterministic generator, so "shuffled" is the same on every machine. */
function rng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}
/** Trip counts spread evenly from lo to hi, one per thread, in a seeded shuffled order. */
export function shuffledTrips(count: number, lo: number, hi: number, seed = 7): number[] {
  const out = Array.from({ length: count }, (_, i) => lo + Math.floor((i * (hi - lo + 1)) / count));
  const next = rng(seed);
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}
export const sortedTrips = (trips: readonly number[]): number[] => [...trips].sort((a, b) => a - b);

// ---------------------------------------------------------------- one fetch, many lanes
export interface Regs {
  r1: number | null;
  r2: number | null;
  r3: number | null;
}
export interface Instr {
  /** As an assembly listing would show it. */
  text: string;
  dst: keyof Regs;
  /** What it means for the thread with global index i. */
  meaning: string;
  value: (r: Regs, i: number) => number;
}
/** The two input arrays of the toy program: small numbers that differ per thread. */
export const inputA = (i: number): number => i + 1;
export const inputB = (i: number): number => 10 * ((i % 8) + 1);

/**
 * Three instructions with no branch. The last is the book's own example,
 * add r1, r2, r3 (sidebar, pp. 78 to 79); the two loads are ours, so that r2 and
 * r3 hold something different in every lane.
 */
export const STRAIGHT_LINE: readonly Instr[] = [
  { text: 'load r2, a[i]', dst: 'r2', meaning: 'r2 = a[i]', value: (_r, i) => inputA(i) }, // i18n-ignore: instruction text (code)
  { text: 'load r3, b[i]', dst: 'r3', meaning: 'r3 = b[i]', value: (_r, i) => inputB(i) }, // i18n-ignore: instruction text (code)
  { text: 'add r1, r2, r3', dst: 'r1', meaning: 'r1 = r2 + r3', value: (r) => (r.r2 ?? 0) + (r.r3 ?? 0) }, // i18n-ignore: instruction text (code)
];
export const straightLine = (width: number): Program => ({
  width,
  stmts: STRAIGHT_LINE.map((x): OpStmt => ({ kind: 'op', label: x.text, cost: 1 })),
});

export interface WarpRun {
  warp: number;
  width: number;
  /** Global index of the thread in lane 0. */
  first: number;
  /** Next instruction to fetch: one program counter for the whole warp. */
  pc: number;
  /** Instructions fetched so far. */
  fetches: number;
  /** Lane-operations carried out so far: fetches * width. */
  laneOps: number;
  /** One private register set per lane. */
  regs: Regs[];
}
export const startWarp = (warp: number, width: number): WarpRun => ({
  warp,
  width,
  first: warp * width,
  pc: 0,
  fetches: 0,
  laneOps: 0,
  regs: Array.from({ length: width }, () => ({ r1: null, r2: null, r3: null })),
});
export const finished = (w: WarpRun, program: readonly Instr[] = STRAIGHT_LINE): boolean => w.pc >= program.length;
/** Fetches the instruction at the warp's program counter once and applies it to every lane's own registers. */
export function step(w: WarpRun, program: readonly Instr[] = STRAIGHT_LINE): WarpRun {
  if (finished(w, program)) return w;
  const ins = program[w.pc];
  return {
    ...w,
    pc: w.pc + 1,
    fetches: w.fetches + 1,
    laneOps: w.laneOps + w.width,
    regs: w.regs.map((r, lane) => ({ ...r, [ins.dst]: ins.value(r, w.first + lane) })),
  };
}

// ---------------------------------------------------------------- exercise 1
/** The launch of exercise 1 (p. 91): N = 1024 in blocks of 128 threads. */
export const EX1 = { n: 1024, blockDim: 128 } as const;
/** Line 3 of foo_kernel guards line 4. */
export const ex1Line4 = (threadIdx: number): boolean => threadIdx < 40 || threadIdx >= 104;
/** Line 6 guards line 7. */
export const ex1Line7 = (i: number): boolean => i % 2 === 0;
/** The loop on line 9 runs 5 - (i % 3) times. */
export const ex1Trips = (i: number): number => 5 - (i % 3);

export interface StatementInWarp {
  mask: boolean[];
  count: number;
  efficiency: number;
  /** At least one thread executes the statement. */
  active: boolean;
  /** Some threads execute it and some do not. */
  divergent: boolean;
}
export interface Ex1Warp {
  block: number;
  warp: number;
  /** threadIdx.x of lane 0. */
  first: number;
  /** Real threads in this warp. */
  real: number;
  line4: StatementInWarp;
  line7: StatementInWarp;
  trips: number[];
  uniformIterations: number;
  divergentIterations: number;
}
export interface Ex1 {
  width: number;
  blocks: number;
  warpsPerBlock: number;
  warps: number;
  line4: { active: number; divergent: number };
  line7: { active: number; divergent: number };
  /** The loop, over the whole grid: iterations every warp runs with all lanes, and iterations in which some warp has lanes waiting. */
  loop: { iterations: number; uniform: number; divergent: number };
  detail: Ex1Warp[];
}

const inWarp = (mask: boolean[], width: number): StatementInWarp => {
  const count = countTaken(mask);
  return { mask, count, efficiency: count / width, active: count > 0, divergent: count > 0 && count < mask.length };
};

/**
 * Exercise 1's three statements for every warp of the launch. A 1D block is cut in
 * threadIdx.x order, so lane l of warp w is thread w * width + l (warps.ts, tested there).
 */
export function exercise1(width = 32, n: number = EX1.n, blockDim: number = EX1.blockDim): Ex1 {
  const blocks = Math.ceil(n / blockDim);
  const perBlock = Math.ceil(blockDim / width);
  const detail: Ex1Warp[] = [];
  for (let b = 0; b < blocks; b++)
    for (let w = 0; w < perBlock; w++) {
      const first = w * width;
      const real = Math.min(width, blockDim - first);
      const tx = Array.from({ length: real }, (_, l) => first + l);
      const trips = tx.map((t) => ex1Trips(b * blockDim + t));
      const shortest = Math.min(...trips);
      const longest = Math.max(...trips);
      detail.push({
        block: b,
        warp: w,
        first,
        real,
        line4: inWarp(tx.map(ex1Line4), width),
        line7: inWarp(tx.map((t) => ex1Line7(b * blockDim + t)), width),
        trips,
        uniformIterations: shortest,
        divergentIterations: longest - shortest,
      });
    }
  const tally = (pick: (w: Ex1Warp) => StatementInWarp) => ({
    active: detail.filter((w) => pick(w).active).length,
    divergent: detail.filter((w) => pick(w).divergent).length,
  });
  const iterations = Math.max(...detail.map((w) => w.uniformIterations + w.divergentIterations));
  const uniform = Math.min(...detail.map((w) => w.uniformIterations));
  return {
    width,
    blocks,
    warpsPerBlock: perBlock,
    warps: detail.length,
    line4: tally((w) => w.line4),
    line7: tally((w) => w.line7),
    loop: { iterations, uniform, divergent: iterations - uniform },
    detail,
  };
}
