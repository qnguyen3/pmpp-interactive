// Many small kernels at once: a timeline model for section 22.3 (p. 538). The book's
// response to the small activities that Amdahl's law leaves behind is to run them in
// parallel with each other, for example as several small kernels on the GPU at the
// same time, which CUDA expresses with streams (chapter 23, §23.3). This model shows
// when that helps, and adds what the book leaves out: the host's cost per launch, and
// how each backend lets independent kernels overlap.
//
// k kernels, each filling a fraction f of the device for a time d, are issued by a
// host that spends L on each launch. Modes:
//   'stream'   one stream: each kernel starts once it is launched and the one before
//              it has finished.
//   'streams'  one stream per independent chain, mapped round-robin onto Q hardware
//              queues (HIP: at most GPU_MAX_HW_QUEUES per device, 4 by default, with
//              further streams sharing them round-robin; ch22-facts B1). Q may be
//              Infinity: one queue per stream.
//   'encoder'  Metal: every dispatch is encoded into one command buffer (L each),
//              which is committed after the last one. In a concurrent encoder the
//              dispatches may overlap, and a memory barrier goes before a dispatch
//              that reads the output of one encoded since the last barrier; a
//              barrier holds back everything after it until every dispatch before
//              it has finished (ch22-facts C1, D8). A serial encoder runs its
//              dispatches one after another (ch02-facts C9).
//   'graph'    a graph captured from one stream: the host pays one launch for the
//              whole sequence, and the kernels run one after another (ch22-facts A3).
// With `pairs`, every odd-numbered kernel reads the output of the kernel before it.
//
// What the model assumes (the lesson states it next to the explorer):
//   - Kernels run side by side while their fills add up to at most 1; otherwise a
//     kernel waits. CUDA's "may run concurrently" is modelled as "does".
//   - A launch, and the encoding of a dispatch, cost the same constant L. A graph's
//     launch costs one L. A graph's kernels and a stream's kernels run back to back
//     with no gap once they may start.
//   - A hardware queue shared by several streams runs its kernels one after another,
//     in the order they were launched. That order is our assumption: AMD's page says
//     streams share queues, not how a shared queue orders them.
//   - The Metal command buffer is not split: the device starts only at the commit.
//   - No memory contention, no copies, no cost for a barrier itself.
//   - All times are relative (d = 1 in the lesson), never device times.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

export type Mode = 'stream' | 'streams' | 'encoder' | 'graph';

export interface LanesInput {
  /** Number of kernels. */
  k: number;
  /** Fraction of the device each kernel fills while it runs, 0 < f <= 1. */
  f: number;
  /** How long each kernel runs once started. */
  d: number;
  /** Host time per launch, or per encoded dispatch, in the same units. */
  L: number;
  mode: Mode;
  /** 'streams': hardware queues the streams share, round-robin. Infinity (the default): one per stream. */
  queues?: number;
  /** Every odd-numbered kernel reads the output of the kernel before it. */
  pairs?: boolean;
  /** 'encoder': dispatch type concurrent (true, the default) or serial (false). */
  concurrent?: boolean;
  /** 'encoder' with pairs: encode every producer first, then every consumer, instead of in issue order. */
  grouped?: boolean;
}

export interface KernelRun {
  /** Kernel number, in issue order. */
  i: number;
  /** Device row it is drawn in: its queue or stream, or the first free row of an encoder. */
  row: number;
  /** The stream it is on ('stream' and 'graph': 0; 'encoder': none, -1). */
  stream: number;
  /** The kernel whose output it reads, or null. */
  dependsOn: number | null;
  start: number;
  end: number;
}

export interface HostSpan {
  kind: 'launch' | 'encode' | 'graph';
  /** The kernel launched or encoded; null for a graph launch. */
  i: number | null;
  start: number;
  end: number;
}

export interface LanesResult {
  input: LanesInput;
  host: HostSpan[];
  /** Indexed by kernel number. */
  kernels: KernelRun[];
  /** One label per device row. */
  rows: string[];
  /** Times at which a Metal barrier lets the dispatches after it start. */
  barriers: number[];
  /** Metal: when the command buffer is committed. */
  commit: number | null;
  /** When the host has issued its last launch (or committed). */
  hostEnd: number;
  /** When the last kernel ends. */
  total: number;
  /** Device work done: k f d. */
  busy: number;
  /** busy / total: the average fraction of the device in use. */
  utilisation: number;
  /** Most kernels running at one moment. */
  maxConcurrent: number;
  /** Time before hostEnd during which no kernel runs: the device waiting for the host. */
  idleWhileLaunching: number;
  /** Fraction of the device in use, as a step function: `fill` holds from `t` to the next point. */
  occupancy: { t: number; fill: number }[];
}

/** HIP's default number of hardware queues per device per process (GPU_MAX_HW_QUEUES; ch22-facts B1). */
export const HIP_DEFAULT_QUEUES = 4;

const EPS = 1e-9;

/** The kernel whose output kernel `i` reads, or null. */
const producer = (i: number, pairs: boolean) => (pairs && i % 2 === 1 ? i - 1 : null);

/** Fraction of the device in use at time `t` by the runs given. */
function fillAt(t: number, runs: { start: number; end: number }[], f: number): number {
  let s = 0;
  for (const r of runs) if (r.start <= t + EPS && t < r.end - EPS) s += f;
  return s;
}

/** Earliest start at or after `t0` at which a kernel of fill f and length d fits beside the runs already placed. */
function earliestFit(t0: number, f: number, d: number, placed: { start: number; end: number }[]): number {
  const candidates = [t0, ...placed.map((r) => r.end).filter((e) => e > t0)].sort((a, b) => a - b);
  for (const c of candidates) {
    const checks = [c, ...placed.map((r) => r.start).filter((s) => s > c + EPS && s < c + d - EPS)];
    if (checks.every((x) => fillAt(x, placed, f) + f <= 1 + EPS)) return c;
  }
  return Math.max(t0, ...placed.map((r) => r.end));
}

/** The order the host issues (or encodes) the kernels in. */
export function issueOrder(input: LanesInput): number[] {
  const all = Array.from({ length: input.k }, (_, i) => i);
  if (input.mode === 'encoder' && input.pairs && input.grouped && input.concurrent !== false)
    return [...all.filter((i) => producer(i, true) === null), ...all.filter((i) => producer(i, true) !== null)];
  return all;
}

export function simulate(input: LanesInput): LanesResult {
  const k = Math.max(1, Math.round(input.k));
  const f = Math.min(1, Math.max(EPS, input.f));
  const { d, L, mode } = input;
  const pairs = !!input.pairs;
  const concurrent = mode === 'encoder' && input.concurrent !== false;
  const order = issueOrder({ ...input, k });

  // Host lane.
  const host: HostSpan[] = [];
  let hostEnd: number;
  if (mode === 'graph') {
    host.push({ kind: 'graph', i: null, start: 0, end: L });
    hostEnd = L;
  } else {
    order.forEach((i, p) => host.push({ kind: mode === 'encoder' ? 'encode' : 'launch', i, start: p * L, end: (p + 1) * L }));
    hostEnd = k * L;
  }
  const commit = mode === 'encoder' ? hostEnd : null;
  const readyAt = new Map<number, number>();
  order.forEach((i, p) => readyAt.set(i, mode === 'graph' ? L : mode === 'encoder' ? hostEnd : (p + 1) * L));

  // Streams and queues.
  const nStreams = pairs ? Math.ceil(k / 2) : k;
  const streamOf = (i: number) => (mode === 'streams' ? (pairs ? Math.floor(i / 2) : i) : mode === 'encoder' ? -1 : 0);
  const q = mode === 'streams' ? Math.max(1, Math.min(input.queues ?? Infinity, nStreams)) : 1;
  const queueOf = (i: number) => (mode === 'streams' ? streamOf(i) % q : 0);
  const inOrder = mode !== 'encoder' || !concurrent; // one FIFO per queue (or the whole device)

  const runs: KernelRun[] = new Array(k);
  const placed: { start: number; end: number }[] = [];
  const queueFree = new Map<number, number>();
  const barriers: number[] = [];
  let gate = 0;
  let sinceBarrier = new Set<number>();
  for (const i of order) {
    const dep = producer(i, pairs);
    let t0 = readyAt.get(i)!;
    if (inOrder) t0 = Math.max(t0, queueFree.get(queueOf(i)) ?? 0);
    else {
      if (dep !== null && sinceBarrier.has(dep)) {
        gate = Math.max(gate, ...placed.map((r) => r.end));
        barriers.push(gate);
        sinceBarrier = new Set();
      }
      t0 = Math.max(t0, gate);
      sinceBarrier.add(i);
    }
    if (dep !== null && runs[dep]) t0 = Math.max(t0, runs[dep].end);
    const start = earliestFit(t0, f, d, placed);
    const run = { i, row: queueOf(i), stream: streamOf(i), dependsOn: dep, start, end: start + d };
    runs[i] = run;
    placed.push(run);
    if (inOrder) queueFree.set(queueOf(i), run.end);
  }

  // Rows: a concurrent encoder has no queues, so each dispatch takes the first row free at its start.
  let rows: string[];
  if (mode === 'encoder' && concurrent) {
    const rowEnds: number[] = [];
    for (const r of [...runs].sort((a, b) => a.start - b.start || a.i - b.i)) {
      let row = rowEnds.findIndex((e) => e <= r.start + EPS);
      if (row < 0) row = rowEnds.push(0) - 1;
      rowEnds[row] = r.end;
      r.row = row;
    }
    rows = rowEnds.map((_, j) => (j === 0 ? 'encoder' : ''));
  } else if (mode === 'streams') {
    const label = isFinite(input.queues ?? Infinity) ? tr('queue', 'hàng đợi') : 'stream';
    rows = Array.from({ length: q }, (_, j) => `${label} ${j}`);
  } else rows = [mode === 'graph' ? 'graph' : mode === 'encoder' ? tr('serial encoder', 'encoder tuần tự') : 'stream'];

  // Totals and occupancy.
  const total = Math.max(hostEnd, ...runs.map((r) => r.end));
  const busy = k * f * d;
  const events = runs.flatMap((r) => [
    { t: r.start, df: f, dn: 1 },
    { t: r.end, df: -f, dn: -1 },
  ]);
  events.sort((a, b) => a.t - b.t || a.dn - b.dn); // ends before starts at the same moment
  const occupancy: { t: number; fill: number }[] = [{ t: 0, fill: 0 }];
  let fill = 0;
  let n = 0;
  let maxConcurrent = 0;
  for (const e of events) {
    fill += e.df;
    n += e.dn;
    maxConcurrent = Math.max(maxConcurrent, n);
    const v = Math.abs(fill) < EPS ? 0 : fill;
    const last = occupancy[occupancy.length - 1];
    if (Math.abs(last.t - e.t) < EPS) last.fill = v;
    else occupancy.push({ t: e.t, fill: v });
  }
  // Device idle before hostEnd: hostEnd minus the union of kernel runs clipped to [0, hostEnd].
  let covered = 0;
  let reach = 0;
  for (const r of [...runs].sort((a, b) => a.start - b.start)) {
    const s = Math.max(r.start, reach);
    const e = Math.min(r.end, hostEnd);
    if (e > s) covered += e - s;
    reach = Math.max(reach, r.end);
  }
  return {
    input: { ...input, k },
    host,
    kernels: runs,
    rows,
    barriers,
    commit,
    hostEnd,
    total,
    busy,
    utilisation: busy / total,
    maxConcurrent,
    idleWhileLaunching: Math.max(0, hostEnd - covered),
    occupancy,
  };
}

/** The modes compared side by side in the lesson, for the same kernels and launch cost. */
export const COMPARED: { key: string; label: string; mode: Mode; queues?: number; concurrent?: boolean }[] = [
  {
    key: 'stream',
    get label() {
      return tr('One stream', 'Một stream');
    },
    mode: 'stream',
  },
  {
    key: 'streams',
    get label() {
      return tr('A stream each, unlimited queues', 'Mỗi kernel một stream, không giới hạn hàng đợi');
    },
    mode: 'streams',
    queues: Infinity,
  },
  {
    key: 'streams4',
    get label() {
      return tr(`A stream each, ${HIP_DEFAULT_QUEUES} queues (HIP's default)`, `Mỗi kernel một stream, ${HIP_DEFAULT_QUEUES} hàng đợi (mặc định của HIP)`);
    },
    mode: 'streams',
    queues: HIP_DEFAULT_QUEUES,
  },
  {
    key: 'encoder',
    get label() {
      return tr('Metal concurrent encoder', 'Encoder đồng thời của Metal');
    },
    mode: 'encoder',
    concurrent: true,
  },
  {
    key: 'serial',
    get label() {
      return tr('Metal serial encoder', 'Encoder tuần tự của Metal');
    },
    mode: 'encoder',
    concurrent: false,
  },
  {
    key: 'graph',
    get label() {
      return tr('Graph, launched once', 'Graph, khởi chạy một lần');
    },
    mode: 'graph',
  },
];

/** Every compared mode for the same k, f, d, L and dependencies. */
export function compare(input: Omit<LanesInput, 'mode' | 'queues' | 'concurrent'>): (LanesResult & { key: string; label: string })[] {
  return COMPARED.map((c) => ({
    key: c.key,
    label: c.label,
    ...simulate({ ...input, mode: c.mode, queues: c.queues, concurrent: c.concurrent }),
  }));
}

// ---------------------------------------------------------------- slider stops

/** Kernel counts offered: 1 to 16. */
export const KERNEL_COUNTS: number[] = Array.from({ length: 16 }, (_, i) => i + 1);
/** Fill fractions offered: 5% to 100% in steps of 5%. */
export const FILLS: number[] = Array.from({ length: 20 }, (_, i) => Number(((i + 1) * 0.05).toFixed(2)));
/** Launch cost as a multiple of a kernel's duration. */
export const LAUNCH_RATIOS: number[] = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5];
