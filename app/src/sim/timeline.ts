// Two-lane execution model for chapter 2, section 2.2 (book pp. 24–25, Fig. 2.3).
//
// From the book: execution starts as one host thread; a kernel call launches a
// grid of device threads; when all threads of a grid finish, the grid
// terminates (p. 24). The book's figure does not make clear whether the host
// waits while a grid runs, and this chapter never says whether a kernel call
// blocks (notes/ch02.md, "Unclear in the PDF").
//
// From current documentation (notes/today/ch02-facts.md, A5, B3, C4):
//   - A kernel launch is asynchronous: the host does not wait for the kernel to
//     complete, or even to start.
//   - Launches without a stream go into one default stream, and operations in a
//     stream run in the order they were enqueued.
//   - A blocking call (cudaDeviceSynchronize, a cudaMemcpy) is what waits.
// HIP documents the same model; in Metal, commit returns at once and
// waitUntilCompleted blocks.
//
// The model: a short program of host statements. The host lane executes them
// one after another. A launch costs the host LAUNCH_TICKS and puts a grid in a
// queue; the device takes grids from the queue one at a time, in launch order.
// A blocking call returns once every grid launched before it has finished.
//
// What the model leaves out:
//   - More than one stream, events, and anything that lets two grids overlap.
//   - Real durations. A tick is a unit you set with a slider; nothing here is
//     measured. LAUNCH_TICKS, HANDOFF_TICKS and CALL_TICKS are drawing
//     conventions: the documentation says only that the launch returns before
//     the kernel completes "or even starts", not how long either gap is.
//   - What the host thread does while blocked (it may sleep or spin).
//   - Errors. A launch that fails is the subject of section 2.6.
//
// No imports and no syntax that needs compiling away beyond types, so the
// browser checks can load this file directly.

export type Stmt =
  | { kind: 'host'; ticks: number }
  | { kind: 'launch'; grid: string; ticks: number }
  | { kind: 'wait' }
  | { kind: 'read' };

/** Host time one launch statement takes. It does not depend on the grid. */
export const LAUNCH_TICKS = 1;
/** Time between a launch returning and the device starting the grid, when the device is idle. */
export const HANDOFF_TICKS = 1;
/** The least host time a blocking call or a read takes. */
export const CALL_TICKS = 1;

export type HostState = 'running' | 'blocked';

export interface HostSpan {
  /** Index of the statement in the program. */
  stmt: number;
  kind: Stmt['kind'];
  from: number;
  to: number;
  state: HostState;
}

export interface GridRun {
  grid: string;
  /** Index of the launch statement. */
  stmt: number;
  /** When the launch statement returned and the grid entered the queue. */
  enqueued: number;
  start: number;
  end: number;
}

export interface DeviceSpan {
  from: number;
  to: number;
  /** The grid that is running, or null when the device is idle. */
  grid: string | null;
}

export interface Timeline {
  program: Stmt[];
  host: HostSpan[];
  grids: GridRun[];
  device: DeviceSpan[];
  /** When the last host statement returns. */
  hostEnd: number;
  /** When the last of host and device finishes. */
  end: number;
}

/** Runs a program and returns what each lane did and when. */
export function run(program: Stmt[]): Timeline {
  const host: HostSpan[] = [];
  const grids: GridRun[] = [];
  let t = 0; // host clock
  let deviceFree = 0; // when the device finishes everything queued so far
  program.forEach((s, stmt) => {
    const from = t;
    if (s.kind === 'host') {
      t += Math.max(0, s.ticks);
      host.push({ stmt, kind: s.kind, from, to: t, state: 'running' });
    } else if (s.kind === 'launch') {
      t += LAUNCH_TICKS;
      const start = Math.max(t + HANDOFF_TICKS, deviceFree);
      const end = start + Math.max(1, s.ticks);
      deviceFree = end;
      grids.push({ grid: s.grid, stmt, enqueued: t, start, end });
      host.push({ stmt, kind: s.kind, from, to: t, state: 'running' });
    } else if (s.kind === 'wait') {
      // Every grid launched so far was launched before this call.
      t = Math.max(t + CALL_TICKS, deviceFree);
      host.push({ stmt, kind: s.kind, from, to: t, state: 'blocked' });
    } else {
      t += CALL_TICKS;
      host.push({ stmt, kind: s.kind, from, to: t, state: 'running' });
    }
  });
  const end = Math.max(t, deviceFree);
  const device: DeviceSpan[] = [];
  let at = 0;
  for (const g of grids) {
    if (g.start > at) device.push({ from: at, to: g.start, grid: null });
    device.push({ from: g.start, to: g.end, grid: g.grid });
    at = g.end;
  }
  if (end > at) device.push({ from: at, to: end, grid: null });
  return { program, host, grids, device, hostEnd: t, end };
}

export type GridState = 'not launched' | 'queued' | 'running' | 'finished'; // i18n-ignore: identifier also shown to the reader

/** The state of a grid at a moment in time. */
export function gridStateAt(tl: Timeline, grid: string, time: number): GridState {
  const g = tl.grids.find((x) => x.grid === grid);
  if (!g || time < g.enqueued) return 'not launched'; // i18n-ignore: identifier also shown to the reader
  if (time < g.start) return 'queued';
  if (time < g.end) return 'running';
  return 'finished';
}

/** The host time at which `done` statements have returned (0 for none). */
export function timeAfter(tl: Timeline, done: number): number {
  return done <= 0 ? 0 : tl.host[Math.min(done, tl.host.length) - 1].to;
}

export interface StepReport {
  /** How many statements have returned. */
  done: number;
  /** Host time at that moment. */
  time: number;
  /** What the host was doing during the statement that just returned (null before the first). */
  host: HostSpan | null;
  /** State of every grid of the program at that moment, in launch order. */
  grids: { grid: string; state: GridState }[];
}

/** What is true at the moment `done` statements have returned. */
export function report(tl: Timeline, done: number): StepReport {
  const n = Math.max(0, Math.min(done, tl.host.length));
  const time = timeAfter(tl, n);
  return {
    done: n,
    time,
    host: n > 0 ? tl.host[n - 1] : null,
    grids: tl.grids.map((g) => ({ grid: g.grid, state: gridStateAt(tl, g.grid, time) })),
  };
}

/**
 * Is the host certain to see the grid's finished results when statement `stmt`
 * begins? Only if a blocking call that came after the grid's launch has already
 * returned. Timing that happens to work out does not count.
 */
export function hostCanRead(tl: Timeline, grid: string, stmt: number): boolean {
  const g = tl.grids.find((x) => x.grid === grid);
  if (!g) return false;
  return tl.program.some((s, i) => s.kind === 'wait' && i > g.stmt && i < stmt);
}

export type ReadOutcome = 'guaranteed' | 'finished by timing only' | 'not finished';

/** For a statement, what reading each already-launched grid's results would find. */
export function readOutcomes(tl: Timeline, stmt: number): { grid: string; outcome: ReadOutcome }[] {
  const time = stmt < tl.host.length ? tl.host[stmt].from : tl.hostEnd;
  return tl.grids
    .filter((g) => g.stmt < stmt)
    .map((g) => ({
      grid: g.grid,
      outcome: hostCanRead(tl, g.grid, stmt)
        ? ('guaranteed' as const)
        : gridStateAt(tl, g.grid, time) === 'finished'
          ? ('finished by timing only' as const) // i18n-ignore: identifier, translated by its consumer
          : ('not finished' as const), // i18n-ignore: identifier, translated by its consumer
    }));
}

/** Total time during which the host is doing its own work while a grid is running. */
export function overlap(tl: Timeline): number {
  let total = 0;
  for (const h of tl.host) {
    if (h.state !== 'running') continue;
    for (const d of tl.device) {
      if (d.grid === null) continue;
      total += Math.max(0, Math.min(h.to, d.to) - Math.max(h.from, d.from));
    }
  }
  return total;
}

// ---------------------------------------------------------------- the program of section 2.2
export interface Options {
  /** A blocking call follows each launch (true), or the host goes straight on (false). */
  waitAfterLaunch: boolean;
  /** Launch B immediately after A, with nothing between them. */
  backToBack: boolean;
  /** How long each grid runs on the device. */
  gridTicks: number;
  /** How long each piece of host work takes. */
  hostTicks: number;
}

export const BOOK_PICTURE: Options = { waitAfterLaunch: true, backToBack: false, gridTicks: 6, hostTicks: 3 };

/**
 * The program of the book's Fig. 2.3: host work, KernelA, host work, KernelB,
 * then the host reads the results. The two switches decide where the host waits.
 */
export function program(o: Options): Stmt[] {
  const host: Stmt = { kind: 'host', ticks: o.hostTicks };
  const a: Stmt = { kind: 'launch', grid: 'A', ticks: o.gridTicks };
  const b: Stmt = { kind: 'launch', grid: 'B', ticks: o.gridTicks };
  const wait: Stmt = { kind: 'wait' };
  const read: Stmt = { kind: 'read' };
  if (o.backToBack) return o.waitAfterLaunch ? [host, a, b, wait, read] : [host, a, b, host, read];
  return o.waitAfterLaunch ? [host, a, wait, host, b, wait, read] : [host, a, host, b, host, read];
}

export type Backend = 'cuda' | 'hip' | 'metal';
export const BACKENDS: Backend[] = ['cuda', 'hip', 'metal'];
export const BACKEND_LABEL: Record<Backend, string> = { cuda: 'CUDA', hip: 'HIP', metal: 'Metal' };

/** The blocking call each backend documents (ch02-facts A5, B3, C4). */
export const WAIT_CALL: Record<Backend, string> = {
  cuda: 'cudaDeviceSynchronize',
  hip: 'hipDeviceSynchronize',
  metal: 'waitUntilCompleted',
};

/**
 * A statement as source text for a backend. The picture does not change with
 * the backend; only these labels do. `waitsFor` lists the grids a wait covers,
 * which Metal needs because waitUntilCompleted belongs to one command buffer.
 */
export function stmtText(s: Stmt, backend: Backend, waitsFor: string[] = []): string {
  if (s.kind === 'host') return '/* host work */';
  if (s.kind === 'read') return '/* host reads the results */';
  if (s.kind === 'launch')
    return backend === 'metal' ? `[cmd${s.grid} commit];` : `Kernel${s.grid}<<<nBlk, nTid>>>(args);`; // i18n-ignore: source code
  if (backend === 'metal') return waitsFor.map((g) => `[cmd${g} waitUntilCompleted];`).join(' ');
  return WAIT_CALL[backend] + '();';
}

/** For a wait statement, the grids launched since the previous wait (the ones it is there for). */
export function waitCovers(prog: Stmt[], stmt: number): string[] {
  const out: string[] = [];
  for (let i = stmt - 1; i >= 0 && prog[i].kind !== 'wait'; i--) {
    const s = prog[i];
    if (s.kind === 'launch') out.unshift(s.grid);
  }
  return out;
}
