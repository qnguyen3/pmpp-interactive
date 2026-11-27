// Chapter 4, section 4.3: the block-wide barrier. Two models.
//
// timeline(durations): threads that need different times to reach one barrier.
// It gives the moment of release, each thread's wait, and the share of all thread
// time spent waiting (the book's Fig. 4.3 and exercise 4).
//
// run(program, rule, order): small per-thread programs made of work, barrier calls
// and returns, run until every thread has ended or nothing can move. It reports,
// step by step, who waits at which call site, whether the waiting threads can be
// released, and whether a thread started a phase before another had finished the
// one before it. It runs the bug hunt "Is this barrier legal?".
//
// Sources for the rules (the lesson cites them where they are used):
//  - the book, section 4.3, pp. 70 to 72: a thread is held at the call until every
//    thread of its block has arrived; a barrier makes all threads finish one phase
//    before any starts the next; a barrier must be executed by all threads of the
//    block or by none; two calls in two branches are two different barriers
//    (Fig. 4.4); misuse can give wrong results or a deadlock;
//  - notes/today/ch04-facts.md A6: NVIDIA's wording, "all non-exited threads ...
//    reach the same __syncthreads*() intrinsic call ... or exit"; C3: Apple's
//    "a thread that has ended no longer participates or blocks remaining threads".
//
// What this leaves out:
//  - "Stuck" in this model stands for what the vendors call undefined behaviour.
//    A real device may hang, or may carry on and compute something wrong. The
//    model always hangs, because that is the outcome you can look at.
//  - A thread moves a whole stretch at a time, from where it is to its next
//    barrier call or its end. Real threads interleave instruction by instruction
//    and run in warps.
//  - One thread order per run (lowest index first, or highest first). A run with
//    nothing flagged shows that this order went well, not that every order would.
//  - No timing: run() has no clock, and timeline() knows only the durations you
//    give it. Nothing here says what a barrier costs.
//  - Programs are written by hand to mirror the kernels in labs/ch04/snippets;
//    the model does not parse C.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- timeline
export interface Timeline {
  /** When the last thread arrives and all are released. */
  release: number;
  /** How long each thread waits at the barrier. */
  waits: number[];
  /** Sum of the waits. */
  waiting: number;
  /** Thread time from start to release: threads x release. */
  total: number;
  /** waiting / total. */
  share: number;
  /** The threads that arrive last (they wait for nobody). */
  slowest: number[];
}

/** Threads that reach one barrier after the given times: who waits, and for how long. */
export function timeline(durations: readonly number[]): Timeline {
  const release = durations.length ? Math.max(...durations) : 0;
  const waits = durations.map((d) => release - d);
  const waiting = waits.reduce((a, b) => a + b, 0);
  const total = durations.length * release;
  return {
    release,
    waits,
    waiting,
    total,
    share: total > 0 ? waiting / total : 0,
    slowest: durations.flatMap((d, i) => (d === release ? [i] : [])),
  };
}

/** The eight times of the book's exercise 4 (p. 91), in microseconds. */
export const EXERCISE_4: readonly number[] = [2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9];

// ---------------------------------------------------------------- programs
export type Op =
  /** A stretch of work that belongs to a phase of the algorithm. */
  | { kind: 'work'; phase: number; line: number }
  /** A call of __syncthreads() at one place in the source. */
  | { kind: 'barrier'; site: number }
  /** An explicit return. Running off the end of the list is a return too. */
  | { kind: 'exit'; line: number };

export interface Program {
  /** One list of operations per thread of the block. */
  threads: Op[][];
}

/** The two readings of "who must be at the barrier". */
export type Rule = 'all-arrive' | 'exited-excluded';
export const RULE_LABEL: Record<Rule, string> = {
  get 'all-arrive'() {
    return tr('every thread of the block must arrive', 'mọi luồng của khối phải đến');
  },
  get 'exited-excluded'() {
    return tr('threads that have ended do not block', 'các luồng đã kết thúc không làm rào chắn phải chờ');
  },
};
export type Order = 'ascending' | 'descending';

export type Status = 'ready' | 'waiting' | 'ended';
export interface ThreadState {
  status: Status;
  /** The call site (source line) the thread waits at, or null. */
  site: number | null;
  /** The last source line the thread executed, or null before it starts. */
  line: number | null;
  /** The phase of the last work it did, or null. */
  phase: number | null;
  /** How many barrier calls it has been released from. */
  passed: number;
}

/** A thread began a phase while another still had work of an earlier phase to do. */
export interface Ahead {
  thread: number;
  phase: number;
  /** The threads that still had earlier work. */
  behind: number[];
}

export interface Step {
  kind: 'advance' | 'release' | 'stuck' | 'finished';
  /** advance: the thread that moved. */
  thread?: number;
  /** advance: the lines it executed, in order. release: the call site that opened. */
  lines?: number[];
  site?: number;
  /** release: the threads let go. */
  released?: number[];
  /** release and stuck: threads that ended without reaching this call. */
  absent?: number[];
  /** stuck: why nothing can move. */
  reason?: 'different-sites' | 'ended-thread';
  /** advance: the thread ran ahead of the others during this step. */
  ahead?: Ahead;
  /** Who waits at which call site after this step. */
  waiting: { site: number; threads: number[] }[];
  /** Every thread after this step. */
  states: ThreadState[];
}

export interface RunResult {
  steps: Step[];
  outcome: 'finished' | 'stuck';
  reason?: 'different-sites' | 'ended-thread';
  /** Number of times a barrier opened. */
  releases: number;
  /** Releases that let threads go although some thread had ended without arriving. */
  partialReleases: number;
  /** Every time a thread ran ahead. */
  ahead: Ahead[];
  /** Call sites that at least one thread reached. */
  sites: number[];
}

const snapshot = (st: ThreadState[]): ThreadState[] => st.map((s) => ({ ...s }));
const waitingBySite = (st: ThreadState[]) => {
  const m = new Map<number, number[]>();
  st.forEach((s, t) => {
    if (s.status === 'waiting') m.set(s.site!, [...(m.get(s.site!) ?? []), t]);
  });
  return [...m.entries()].sort((a, b) => a[0] - b[0]).map(([site, threads]) => ({ site, threads }));
};

/**
 * Runs a block. Each ready thread in turn moves to its next barrier call or to its
 * end. When no thread is ready, the waiting threads are released if the rule allows
 * it, and otherwise the block is stuck.
 */
export function run(program: Program, rule: Rule, order: Order = 'ascending'): RunResult {
  const n = program.threads.length;
  const pc = new Array<number>(n).fill(0);
  const st: ThreadState[] = Array.from({ length: n }, () => ({ status: 'ready' as Status, site: null, line: null, phase: null, passed: 0 }));
  const steps: Step[] = [];
  const ahead: Ahead[] = [];
  const sites = new Set<number>();
  let releases = 0;
  let partialReleases = 0;
  const turn = Array.from({ length: n }, (_, i) => (order === 'ascending' ? i : n - 1 - i));

  /** Threads other than t that still have work of a phase below `phase` to do. */
  const behindOf = (t: number, phase: number) =>
    turn
      .filter((u) => u !== t && program.threads[u].slice(pc[u]).some((o) => o.kind === 'work' && o.phase < phase))
      .sort((a, b) => a - b);

  for (let guard = 0; guard < 10_000; guard++) {
    // 1. Every ready thread moves as far as it can.
    for (const t of turn) {
      if (st[t].status !== 'ready') continue;
      const lines: number[] = [];
      let flagged: Ahead | undefined;
      const ops = program.threads[t];
      while (st[t].status === 'ready') {
        const op = ops[pc[t]];
        if (!op) {
          st[t].status = 'ended';
          break;
        }
        pc[t]++;
        if (op.kind === 'work') {
          const behind = behindOf(t, op.phase);
          if (behind.length && !flagged) flagged = { thread: t, phase: op.phase, behind };
          st[t].phase = op.phase;
          st[t].line = op.line;
          lines.push(op.line);
        } else if (op.kind === 'barrier') {
          st[t].status = 'waiting';
          st[t].site = op.site;
          st[t].line = op.site;
          sites.add(op.site);
          lines.push(op.site);
        } else {
          st[t].status = 'ended';
          st[t].line = op.line;
          lines.push(op.line);
        }
      }
      if (flagged) ahead.push(flagged);
      steps.push({ kind: 'advance', thread: t, lines, ahead: flagged, waiting: waitingBySite(st), states: snapshot(st) });
    }

    // 2. Nobody is ready. Either all have ended, or the waiting threads are judged.
    const groups = waitingBySite(st);
    const ended = st.flatMap((s, t) => (s.status === 'ended' ? [t] : []));
    if (!groups.length) {
      steps.push({ kind: 'finished', waiting: [], states: snapshot(st) });
      return { steps, outcome: 'finished', releases, partialReleases, ahead, sites: [...sites].sort((a, b) => a - b) };
    }
    const stuck = (reason: 'different-sites' | 'ended-thread'): RunResult => {
      steps.push({ kind: 'stuck', reason, absent: ended, waiting: groups, states: snapshot(st) });
      return { steps, outcome: 'stuck', reason, releases, partialReleases, ahead, sites: [...sites].sort((a, b) => a - b) };
    };
    // Two call sites are two barriers: no rule lets threads at different calls release each other.
    if (groups.length > 1) return stuck('different-sites');
    if (ended.length && rule === 'all-arrive') return stuck('ended-thread');
    const { site, threads } = groups[0];
    for (const t of threads) {
      st[t].status = 'ready';
      st[t].site = null;
      st[t].passed++;
    }
    releases++;
    if (ended.length) partialReleases++;
    steps.push({ kind: 'release', site, released: threads, absent: ended, waiting: [], states: snapshot(st) });
  }
  throw new Error('the program did not settle');
}

// ---------------------------------------------------------------- the five kernels
// Each mirrors a file in labs/ch04/snippets (same name, .cu). Line numbers are
// lines of that file; a unit test checks that every barrier line holds a
// __syncthreads() call and every exit line a return.
export type KernelId = 'incorrect_barrier' | 'barrier_uniform' | 'barrier_per_thread' | 'barrier_early_return' | 'barrier_loop' | 'barrier_guarded';

export interface KernelCase {
  id: KernelId;
  /** Lines of the snippet that hold a __syncthreads() call. */
  sites: number[];
  /** Builds the block's program: `threads` threads, in block `block` of the grid. */
  build: (threads: number, block: number) => Program;
}

const each = (threads: number, f: (t: number) => Op[]): Program => ({ threads: Array.from({ length: threads }, (_, t) => f(t)) });

export const KERNELS: Record<KernelId, KernelCase> = {
  // Fig. 4.4: even threads call the barrier on line 5, odd threads the one on line 9.
  incorrect_barrier: {
    id: 'incorrect_barrier',
    sites: [5, 9],
    build: (threads) =>
      each(threads, (t) =>
        t % 2 === 0
          ? [{ kind: 'work', phase: 0, line: 4 }, { kind: 'barrier', site: 5 }]
          : [{ kind: 'work', phase: 0, line: 8 }, { kind: 'barrier', site: 9 }],
      ),
  },
  // if (blockIdx.x > 0) __syncthreads(): the whole block calls it, or none of it does.
  barrier_uniform: {
    id: 'barrier_uniform',
    sites: [6],
    build: (threads, block) =>
      each(threads, () =>
        block > 0
          ? [{ kind: 'work', phase: 0, line: 4 }, { kind: 'barrier', site: 6 }, { kind: 'work', phase: 1, line: 8 }]
          : [{ kind: 'work', phase: 0, line: 4 }, { kind: 'work', phase: 1, line: 8 }],
      ),
  },
  // if (threadIdx.x > 0) __syncthreads(): thread 0 walks past.
  barrier_per_thread: {
    id: 'barrier_per_thread',
    sites: [6],
    build: (threads) =>
      each(threads, (t) =>
        t > 0
          ? [{ kind: 'work', phase: 0, line: 4 }, { kind: 'barrier', site: 6 }, { kind: 'work', phase: 1, line: 8 }]
          : [{ kind: 'work', phase: 0, line: 4 }, { kind: 'work', phase: 1, line: 8 }],
      ),
  },
  // Out-of-range threads return on line 4, before the barrier. The last thread of the block is out of range.
  barrier_early_return: {
    id: 'barrier_early_return',
    sites: [6],
    build: (threads) =>
      each(threads, (t) =>
        t < threads - 1
          ? [{ kind: 'work', phase: 0, line: 5 }, { kind: 'barrier', site: 6 }, { kind: 'work', phase: 1, line: 7 }]
          : [{ kind: 'exit', line: 4 }],
      ),
  },
  // A barrier inside a loop that odd threads go round twice and even threads once.
  barrier_loop: {
    id: 'barrier_loop',
    sites: [7],
    build: (threads) =>
      each(threads, (t) => {
        const ops: Op[] = [];
        for (let k = 0; k <= t % 2; k++) ops.push({ kind: 'work', phase: k, line: 6 }, { kind: 'barrier', site: 7 });
        ops.push({ kind: 'work', phase: 2, line: 9 });
        return ops;
      }),
  },
  // The early return rewritten: the range test guards the work, and every thread calls the barrier.
  barrier_guarded: {
    id: 'barrier_guarded',
    sites: [6],
    build: (threads) =>
      each(threads, (t) =>
        t < threads - 1
          ? [{ kind: 'work', phase: 0, line: 5 }, { kind: 'barrier', site: 6 }, { kind: 'work', phase: 1, line: 7 }]
          : [{ kind: 'barrier', site: 6 }],
      ),
  },
};

/** One sentence for what a run showed, in the order of severity. */
export type Finding = 'stuck-sites' | 'stuck-ended' | 'ran-ahead' | 'partial' | 'clean';
export function finding(r: RunResult): Finding {
  if (r.outcome === 'stuck') return r.reason === 'different-sites' ? 'stuck-sites' : 'stuck-ended';
  if (r.ahead.length) return 'ran-ahead';
  if (r.partialReleases) return 'partial';
  return 'clean';
}
