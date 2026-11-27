// Who waits: one or two iterations of the book's multi-GPU Jacobi loop (chapter 23) for one
// rank, in the four versions the chapter builds:
//   'mpi'      MPI without overlap (Fig. 23.7, p. 549): one kernel in the default stream,
//              cudaDeviceSynchronize, two blocking MPI_Sendrecv, a blocking cudaMemcpy.
//   'streams'  MPI with three streams and events (Fig. 23.14, p. 557).
//   'nccl'     NCCL send/receive queued in the boundary streams (Fig. 23.20, p. 568).
//   'nvshmem'  one kernel that puts its boundary rows into the neighbours' halos as it
//              computes them, and a barrier on the stream (Figs. 23.23, 23.25, pp. 571-573).
//
// Lanes: the host, the top, bottom and internal streams (the one stream of 'mpi' and
// 'nvshmem' is drawn in the internal lane), and two links, one per exchange.
//
// What the model assumes (the lesson states it beside the explorer):
//   - One rank whose two neighbours behave exactly like it. A partner is therefore always
//     ready when this rank is, and the halo this rank receives in an exchange arrives when
//     its own send in that exchange ends.
//   - The "top exchange" is the book's first one: it sends row 1 up and receives the bottom
//     halo (row ny - 1) from below. The "bottom exchange" sends row ny - 2 down and receives
//     the top halo (row 0). Each exchange has a link of its own.
//   - The GPU is SLOTS block slots. A kernel is a number of equal-length blocks: the
//     internal kernel INTERNAL_BLOCKS blocks (three full waves, so it takes `internal` time
//     alone), a boundary kernel BOUNDARY_BLOCKS blocks of length `boundary`. A one-kernel
//     version launches all those blocks at once, boundary rows first. A slot that frees
//     goes to the pending kernel of highest priority, then to the one issued first. Nothing
//     is preempted (ch23-facts A2).
//   - Streams run their operations in order. An event record completes when its stream
//     reaches it; a stream wait waits for the record that was the event's latest when the
//     wait was queued, and passes at once if the event was never recorded (A5).
//   - Every host API call costs HOST_CALL; a group call costs four. A blocking call returns
//     when what it waits for has happened. MPI_Sendrecv starts its exchange at once and
//     returns when it ends; MPI_Allreduce takes ALLREDUCE. NCCL's own kernels take no slots.
//   - NVSHMEM: the puts of a boundary row travel as soon as the blocks that compute that row
//     finish, and nvshmemx_barrier_all_on_stream completes when every put has been
//     delivered, plus BARRIER.
//   - With a pageable l2norm_h, cudaMemcpyAsync is synchronous: the host waits in the call
//     until the copy is done (A6 says "might be synchronous"; the model takes the worse case).
//   - A kernel adds its block sums to the norm as each block finishes (Fig. 23.1). Memset,
//     copy and event operations take no slots. All times are unitless.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

export type Version = 'mpi' | 'streams' | 'nccl' | 'nvshmem';
export type Priority = 'equal' | 'higher';
export type Issue = 'boundary-first' | 'internal-first';
export type Defect = 'none' | 'reset-waits' | 'compute-waits' | 'sync-top' | 'exchange-waits' | 'barrier' | 'pageable';
export type StreamId = 'top' | 'bottom' | 'internal';
export type Lane = 'host' | StreamId | 'linkTop' | 'linkBottom';
export type Part = 'top' | 'bottom' | 'internal';
export type Which = 'top' | 'bottom';
export type EventName = 'resetL2' | 'computeTop' | 'computeBottom' | 'exchangeTop' | 'exchangeBottom';

export const SLOTS = 8;
export const INTERNAL_BLOCKS = 24;
export const BOUNDARY_BLOCKS = 2;
export const HOST_CALL = 0.02;
export const MEMSET = 0.04;
export const COPY = 0.06;
export const ALLREDUCE = 0.3;
export const BARRIER = 0.04;

export interface Durations {
  /** The internal kernel's time with the whole GPU to itself. */
  internal: number;
  /** Each boundary kernel's time (one wave of its blocks). */
  boundary: number;
  /** The two exchanges, each on its own link. */
  exTop: number;
  exBottom: number;
}
/** The durations the explorer and the scene start from. */
export const DEFAULTS: Durations = { internal: 6, boundary: 1, exTop: 4, exBottom: 4 };

export interface LoopInput extends Durations {
  version: Version;
  /** Boundary streams' priority relative to the internal stream (the book: higher, p. 558). */
  priority?: Priority;
  /** Order of the three launches on the host (the book: boundary kernels first). */
  issue?: Issue;
  /** At most one ordering removed: the bug hunt. Ignored where it does not apply. */
  defect?: Defect;
  iterations?: number;
}

export const VERSIONS: Record<Version, { label: string; fig: string; pages: string }> = {
  mpi: { get label() { return tr('MPI, no overlap', 'MPI, không chồng lấp'); }, fig: 'Fig. 23.7', pages: '548–549' },
  streams: { get label() { return tr('MPI with streams', 'MPI với stream'); }, fig: 'Fig. 23.14', pages: '556–562' },
  nccl: { label: 'NCCL', fig: 'Fig. 23.20', pages: '563–568' },
  nvshmem: { label: 'NVSHMEM', fig: 'Fig. 23.25', pages: '568–574' },
};

export const DEFECTS: Record<Exclude<Defect, 'none'>, { label: string; where: string; versions: Version[] }> = {
  'reset-waits': { get label() { return tr('drop the two waits on resetL2', 'bỏ hai lần chờ trên resetL2'); }, get where() { return tr('lines 06–07', 'dòng 06–07'); }, versions: ['streams', 'nccl'] },
  'compute-waits': { get label() { return tr('drop the waits on computeTop and computeBottom', 'bỏ các lần chờ trên computeTop và computeBottom'); }, get where() { return tr('lines 19–20', 'dòng 19–20'); }, versions: ['streams', 'nccl'] },
  'sync-top': { get label() { return tr('drop the host sync before the top exchange', 'bỏ lần đồng bộ của host trước trao đổi phía trên'); }, get where() { return tr('Fig. 23.14 line 27', 'Fig. 23.14 dòng 27'); }, versions: ['streams'] },
  'exchange-waits': { get label() { return tr('drop the waits on exchangeTop and exchangeBottom', 'bỏ các lần chờ trên exchangeTop và exchangeBottom'); }, get where() { return tr('Fig. 23.20 lines 43–44', 'Fig. 23.20 dòng 43–44'); }, versions: ['nccl'] },
  barrier: { get label() { return tr('drop nvshmemx_barrier_all_on_stream', 'bỏ nvshmemx_barrier_all_on_stream'); }, get where() { return tr('Fig. 23.25 line 18', 'Fig. 23.25 dòng 18'); }, versions: ['nvshmem'] },
  pageable: { get label() { return tr('allocate l2norm_h with malloc (pageable)', 'cấp phát l2norm_h bằng malloc (pageable)'); }, where: 'p. 562', versions: ['streams', 'nccl', 'nvshmem'] },
};

export function defectsFor(v: Version): Defect[] {
  return ['none', ...(Object.keys(DEFECTS) as Exclude<Defect, 'none'>[]).filter((d) => DEFECTS[d].versions.includes(v))];
}

// ---------------------------------------------------------------- the host programs

type OpSpec =
  | { k: 'memset' }
  | { k: 'copy' }
  | { k: 'record'; ev: EventName }
  | { k: 'wait'; ev: EventName }
  | { k: 'kernel'; part: Part | 'whole' }
  | { k: 'exchange'; which: Which }
  | { k: 'barrier' };

type Act =
  | { k: 'op'; stream: StreamId; op: OpSpec; block?: boolean }
  | { k: 'sync'; stream: StreamId }
  | { k: 'devsync' }
  | { k: 'sendrecv'; which: Which }
  | { k: 'allreduce' };

export interface CallSpec {
  /** Line(s) of the version's figure. */
  line: string;
  /** What the call is, short. */
  label: string;
  /** API calls it makes (host time = cost x HOST_CALL). */
  cost: number;
  act: Act;
}

const op = (line: string, label: string, stream: StreamId, o: OpSpec, cost = 1, block = false): CallSpec => ({
  line, label, cost, act: { k: 'op', stream, op: o, block },
});

/** The host's calls for one iteration of a version, in program order. */
export function program(v: Version, issue: Issue = 'boundary-first', defect: Defect = 'none'): CallSpec[] {
  const d = defectsFor(v).includes(defect) ? defect : 'none';
  if (v === 'mpi')
    return [
      op('04', 'cudaMemset', 'internal', { k: 'memset' }),
      op('07', 'launch_jacobi_kernel', 'internal', { k: 'kernel', part: 'whole' }),
      { line: '08', label: 'cudaDeviceSynchronize', cost: 1, act: { k: 'devsync' } },
      { line: '13–15', label: 'MPI_Sendrecv (top)', cost: 1, act: { k: 'sendrecv', which: 'top' } }, // i18n-ignore: identifier also shown to the reader
      { line: '16–18', label: 'MPI_Sendrecv (bottom)', cost: 1, act: { k: 'sendrecv', which: 'bottom' } }, // i18n-ignore: identifier also shown to the reader
      op('21', 'cudaMemcpy', 'internal', { k: 'copy' }, 1, true),
      { line: '24', label: 'MPI_Allreduce', cost: 1, act: { k: 'allreduce' } }, // i18n-ignore: API name
    ];
  if (v === 'nvshmem')
    return [
      op('04', 'cudaMemsetAsync', 'internal', { k: 'memset' }),
      op('09', tr('launch_jacobi_kernel (computes and puts)', 'launch_jacobi_kernel (tính toán và put)'), 'internal', { k: 'kernel', part: 'whole' }),
      op('12–13', 'cudaMemcpyAsync', 'internal', { k: 'copy' }, 1, d === 'pageable'),
      { line: '14', label: 'cudaStreamSynchronize(stream)', cost: 1, act: { k: 'sync', stream: 'internal' } },
      { line: '15', label: 'MPI_Allreduce', cost: 1, act: { k: 'allreduce' } }, // i18n-ignore: API name
      ...(d === 'barrier' ? [] : [op('18', 'nvshmemx_barrier_all_on_stream', 'internal', { k: 'barrier' })]),
    ];
  // Figs. 23.14 and 23.20 share lines 04-22.
  const reset = [
    op('04', 'cudaMemsetAsync', 'internal', { k: 'memset' }),
    op('05', 'cudaEventRecord(resetL2)', 'internal', { k: 'record', ev: 'resetL2' }),
    ...(d === 'reset-waits'
      ? []
      : [
          op('06', 'cudaStreamWaitEvent(topStream, resetL2)', 'top', { k: 'wait', ev: 'resetL2' }),
          op('07', 'cudaStreamWaitEvent(bottomStream, resetL2)', 'bottom', { k: 'wait', ev: 'resetL2' }),
        ]),
  ];
  const boundary = [
    op('10', tr('launch top kernel', 'khởi chạy kernel phía trên'), 'top', { k: 'kernel', part: 'top' }),
    op('11', 'cudaEventRecord(computeTop)', 'top', { k: 'record', ev: 'computeTop' }),
    op('12–13', tr('launch bottom kernel', 'khởi chạy kernel phía dưới'), 'bottom', { k: 'kernel', part: 'bottom' }),
    op('14', 'cudaEventRecord(computeBottom)', 'bottom', { k: 'record', ev: 'computeBottom' }),
  ];
  const internal = [op('15–16', tr('launch internal kernel', 'khởi chạy kernel bên trong'), 'internal', { k: 'kernel', part: 'internal' })];
  const copy = [
    ...(d === 'compute-waits'
      ? []
      : [
          op('19', 'cudaStreamWaitEvent(internalStream, computeTop)', 'internal', { k: 'wait', ev: 'computeTop' }),
          op('20', 'cudaStreamWaitEvent(internalStream, computeBottom)', 'internal', { k: 'wait', ev: 'computeBottom' }),
        ]),
    op('21–22', 'cudaMemcpyAsync', 'internal', { k: 'copy' }, 1, d === 'pageable'),
  ];
  const head = [...reset, ...(issue === 'internal-first' ? [...internal, ...boundary] : [...boundary, ...internal]), ...copy];
  if (v === 'streams')
    return [
      ...head,
      ...(d === 'sync-top' ? [] : [{ line: '27', label: 'cudaStreamSynchronize(topStream)', cost: 1, act: { k: 'sync', stream: 'top' } } as CallSpec]),
      { line: '28–30', label: 'MPI_Sendrecv (top)', cost: 1, act: { k: 'sendrecv', which: 'top' } }, // i18n-ignore: identifier also shown to the reader
      { line: '31', label: 'cudaStreamSynchronize(bottomStream)', cost: 1, act: { k: 'sync', stream: 'bottom' } },
      { line: '32–34', label: 'MPI_Sendrecv (bottom)', cost: 1, act: { k: 'sendrecv', which: 'bottom' } }, // i18n-ignore: identifier also shown to the reader
      { line: '37', label: 'cudaStreamSynchronize(internalStream)', cost: 1, act: { k: 'sync', stream: 'internal' } },
      { line: '38', label: 'MPI_Allreduce', cost: 1, act: { k: 'allreduce' } }, // i18n-ignore: API name
    ];
  return [
    ...head,
    op('27–30', tr('ncclGroupStart, ncclSend, ncclRecv, ncclGroupEnd (top)', 'ncclGroupStart, ncclSend, ncclRecv, ncclGroupEnd (phía trên)'), 'top', { k: 'exchange', which: 'top' }, 4),
    op('31', 'cudaEventRecord(exchangeTop)', 'top', { k: 'record', ev: 'exchangeTop' }),
    op('32–35', tr('ncclGroupStart, ncclSend, ncclRecv, ncclGroupEnd (bottom)', 'ncclGroupStart, ncclSend, ncclRecv, ncclGroupEnd (phía dưới)'), 'bottom', { k: 'exchange', which: 'bottom' }, 4),
    op('36', 'cudaEventRecord(exchangeBottom)', 'bottom', { k: 'record', ev: 'exchangeBottom' }),
    { line: '39', label: 'cudaStreamSynchronize(internalStream)', cost: 1, act: { k: 'sync', stream: 'internal' } },
    { line: '40', label: 'MPI_Allreduce', cost: 1, act: { k: 'allreduce' } }, // i18n-ignore: API name
    ...(d === 'exchange-waits'
      ? []
      : [
          op('43', 'cudaStreamWaitEvent(internalStream, exchangeTop)', 'internal', { k: 'wait', ev: 'exchangeTop' }),
          op('44', 'cudaStreamWaitEvent(internalStream, exchangeBottom)', 'internal', { k: 'wait', ev: 'exchangeBottom' }),
        ]),
  ];
}

// ---------------------------------------------------------------- results

export interface Span {
  start: number;
  end: number;
}

export type SegKind =
  | 'call' // host: queueing an operation; the call returns at once
  | 'blocked' // host: inside a call that waits for something
  | 'memset'
  | 'copy'
  | 'pending' // a kernel at the head of its stream, waiting for block slots
  | 'kernel'
  | 'wait' // a stream held by an event wait
  | 'exchange'
  | 'barrier';

export interface Seg {
  lane: Lane;
  kind: SegKind;
  iter: number;
  start: number;
  end: number;
  /** Figure line(s) of the call that made it. */
  line: string;
  label: string;
  part?: Part | 'whole';
  /** Waits: the event, and when the record it waits for completed. */
  event?: EventName;
  from?: number;
}

export interface Mark {
  lane: StreamId;
  iter: number;
  at: number;
  event: EventName;
  line: string;
}

export interface BlockRun {
  slot: number;
  start: number;
  end: number;
  part: Part;
  iter: number;
  whole: boolean;
}

/** A stream wait on an event, including those that passed at once (for drawing the arrows). */
export interface WaitRun {
  lane: StreamId;
  iter: number;
  event: EventName;
  line: string;
  /** When the wait reached the head of its stream, and when it let the stream go on. */
  reached: number;
  end: number;
  /** When the record it waited for completed (null: the event had never been recorded). */
  from: number | null;
  /** The lane of that record. */
  fromLane: StreamId | null;
}

/** One queued operation, for drawing insertion. */
export interface Insertion {
  iter: number;
  lane: StreamId;
  at: number;
  line: string;
  kind: OpSpec['k'];
  label: string;
}

export type ViolationKind = 'reset' | 'copy' | 'send' | 'halo' | 'stall';
export interface Violation {
  kind: ViolationKind;
  iter: number;
  /** The ordering broken (see edges()), or null for the pageable stall. */
  edge: EdgeId | null;
  lane: Lane;
  at: number;
  until: number;
  text: string;
}

export interface IterStats {
  iter: number;
  hostStart: number;
  hostEnd: number;
  hostBlocked: number;
  memset: Span;
  copy: Span;
  allreduce: Span;
  /** When the rows of each part were computed (the kernels, or a one-kernel version's blocks). */
  top: Span;
  bottom: Span;
  internal: Span;
  exTop: Span;
  exBottom: Span;
  /** Bottom exchange start minus the end of the bottom rows' computation. */
  bottomGap: number;
  /** Both exchanges end no later than the internal rows' computation. */
  hidden: boolean;
}

export interface LoopResult {
  input: Required<LoopInput>;
  lanes: Lane[];
  segs: Seg[];
  marks: Mark[];
  waits: WaitRun[];
  blocks: BlockRun[];
  insertions: Insertion[];
  iters: IterStats[];
  /** From the end of one pass through the host loop to the end of the next (iteration 2). */
  iterTime: number;
  /** Host time inside waiting calls, in the last iteration. */
  hostBlocked: number;
  /** In the last iteration. */
  bottomGap: number;
  hidden: boolean;
  violations: Violation[];
  /** When the last thing (host or device) finished. */
  end: number;
}

// ---------------------------------------------------------------- the simulation

const EPS = 1e-9;

interface Blk {
  op: Op;
  part: Part;
  dur: number;
  slot: number;
  start: number | null;
  end: number | null;
}
interface Op {
  iter: number;
  stream: StreamId;
  spec: OpSpec;
  line: string;
  label: string;
  inserted: number;
  reached: number | null;
  start: number | null;
  end: number | null;
  target: Op | null;
  prio: number;
  seq: number;
  blocks: Blk[];
  next: number;
  partEnd: Partial<Record<Part, number>>;
}
interface Xch {
  iter: number;
  which: Which;
  start: number;
  end: number;
}
interface HostCall {
  iter: number;
  spec: CallSpec;
  start: number;
  end: number | null;
  op: Op | null;
  ready: number; // earliest return
  done: (t: number) => boolean;
}

const r2 = (x: number) => (Math.round(x * 100) / 100).toFixed(2);

export function simulate(inp: LoopInput): LoopResult {
  const input: Required<LoopInput> = {
    internal: inp.internal,
    boundary: inp.boundary,
    exTop: inp.exTop,
    exBottom: inp.exBottom,
    version: inp.version,
    priority: inp.priority ?? 'higher',
    issue: inp.issue ?? 'boundary-first',
    defect: defectsFor(inp.version).includes(inp.defect ?? 'none') ? (inp.defect ?? 'none') : 'none',
    iterations: inp.iterations ?? 2,
  };
  const { version, priority, issue, defect, iterations } = input;
  const one = version === 'mpi' || version === 'nvshmem';
  const exDur = (w: Which) => (w === 'top' ? input.exTop : input.exBottom);

  const plan = program(version, issue, defect);
  const calls: { iter: number; spec: CallSpec }[] = [];
  for (let i = 1; i <= iterations; i++) for (const c of plan) calls.push({ iter: i, spec: c });

  const streams: Record<StreamId, Op[]> = { top: [], bottom: [], internal: [] };
  const lastRecord: Partial<Record<EventName, Op>> = {};
  const slots: (Blk | null)[] = Array.from({ length: SLOTS }, () => null);
  let pending: Op[] = [];
  const allBlocks: Blk[] = [];
  const xchs: Xch[] = [];
  const done: HostCall[] = [];
  const insertions: Insertion[] = [];
  let seq = 0;

  const makeBlocks = (o: Op, part: Part | 'whole'): Blk[] => {
    const b = (p: Part, n: number, dur: number) => Array.from({ length: n }, (): Blk => ({ op: o, part: p, dur, slot: -1, start: null, end: null }));
    const top = () => b('top', BOUNDARY_BLOCKS, input.boundary);
    const bottom = () => b('bottom', BOUNDARY_BLOCKS, input.boundary);
    const internal = () => b('internal', INTERNAL_BLOCKS, (input.internal * SLOTS) / INTERNAL_BLOCKS);
    if (part === 'whole') return [...top(), ...bottom(), ...internal()];
    return part === 'top' ? top() : part === 'bottom' ? bottom() : internal();
  };
  const isDone = (o: Op, t: number) => o.end !== null && o.end <= t + EPS;
  const streamIdle = (s: StreamId, t: number) => streams[s].every((o) => o.inserted <= t + EPS && isDone(o, t));

  let cur: HostCall | null = null;
  let pc = 0;

  const startCall = (t: number): HostCall => {
    const c = calls[pc];
    const a = c.spec.act;
    const t1 = t + c.spec.cost * HOST_CALL;
    const hc: HostCall = { iter: c.iter, spec: c.spec, start: t, end: null, op: null, ready: t1, done: () => true };
    if (a.k === 'op') {
      const o: Op = {
        iter: c.iter, stream: a.stream, spec: a.op, line: c.spec.line, label: c.spec.label, inserted: t1,
        reached: null, start: null, end: null, target: null, prio: 0, seq: 0, blocks: [], next: 0, partEnd: {},
      };
      if (a.op.k === 'wait') o.target = lastRecord[a.op.ev] ?? null;
      if (a.op.k === 'record') lastRecord[a.op.ev] = o;
      if (a.op.k === 'kernel') {
        o.blocks = makeBlocks(o, a.op.part);
        o.prio = !one && a.stream !== 'internal' && priority === 'higher' ? 1 : 0;
        o.seq = seq++;
      }
      streams[a.stream].push(o);
      insertions.push({ iter: c.iter, lane: a.stream, at: t1, line: c.spec.line, kind: a.op.k, label: c.spec.label });
      hc.op = o;
      hc.done = a.block ? (u) => u >= t1 - EPS && isDone(o, u) : (u) => u >= t1 - EPS;
    } else if (a.k === 'sync') {
      hc.done = (u) => u >= t1 - EPS && streamIdle(a.stream, u);
    } else if (a.k === 'devsync') {
      hc.done = (u) => u >= t1 - EPS && streamIdle('top', u) && streamIdle('bottom', u) && streamIdle('internal', u);
    } else if (a.k === 'sendrecv') {
      // The neighbours call their matching MPI_Sendrecv at the same moment.
      const x: Xch = { iter: c.iter, which: a.which, start: t1, end: t1 + exDur(a.which) };
      xchs.push(x);
      hc.ready = x.end;
      hc.done = (u) => u >= x.end - EPS;
    } else {
      const e = t + ALLREDUCE;
      hc.ready = e;
      hc.done = (u) => u >= e - EPS;
    }
    return hc;
  };

  let t = 0;
  let stuck = false;
  for (let guard = 0; guard < 100000; guard++) {
    let changed = true;
    while (changed) {
      changed = false;
      // Blocks retire; a kernel ends with its last block; a one-kernel NVSHMEM launch starts
      // the puts of a boundary row when the blocks of that row have all finished.
      for (let k = 0; k < SLOTS; k++) {
        const b = slots[k];
        if (!b || (b.end as number) > t + EPS) continue;
        slots[k] = null;
        changed = true;
        const o = b.op;
        const mine = o.blocks.filter((x) => x.part === b.part);
        if (o.partEnd[b.part] === undefined && mine.every((x) => x.end !== null && x.end <= t + EPS)) {
          o.partEnd[b.part] = Math.max(...mine.map((x) => x.end as number));
          if (version === 'nvshmem' && b.part !== 'internal') {
            const w = b.part as Which;
            xchs.push({ iter: o.iter, which: w, start: o.partEnd[b.part] as number, end: (o.partEnd[b.part] as number) + exDur(w) });
          }
        }
        if (o.end === null && o.next === o.blocks.length && o.blocks.every((x) => x.end !== null && x.end <= t + EPS))
          o.end = Math.max(...o.blocks.map((x) => x.end as number));
      }
      // Streams: start what has reached the head of its stream.
      for (const s of ['internal', 'top', 'bottom'] as StreamId[]) {
        const h = streams[s].find((o) => !isDone(o, t));
        if (!h || h.inserted > t + EPS || h.start !== null) continue;
        if (h.reached === null) {
          h.reached = t;
          changed = true;
        }
        const sp = h.spec;
        if (sp.k === 'memset' || sp.k === 'copy') {
          h.start = t;
          h.end = t + (sp.k === 'memset' ? MEMSET : COPY);
          changed = true;
        } else if (sp.k === 'record') {
          h.start = h.end = t;
          changed = true;
        } else if (sp.k === 'wait') {
          if (!h.target || isDone(h.target, t)) {
            h.start = h.end = t;
            changed = true;
          }
        } else if (sp.k === 'kernel') {
          if (!pending.includes(h)) {
            pending.push(h);
            changed = true;
          }
        } else if (sp.k === 'exchange') {
          h.start = t;
          h.end = t + exDur(sp.which);
          xchs.push({ iter: h.iter, which: sp.which, start: h.start, end: h.end });
          changed = true;
        } else {
          // nvshmemx_barrier_all_on_stream: every put of this iteration, and (identical
          // neighbours) every put into this rank, delivered.
          const deliveries = xchs.filter((x) => x.iter === h.iter).map((x) => x.end);
          h.start = t;
          h.end = Math.max(t, ...deliveries) + BARRIER;
          changed = true;
        }
      }
      // The GPU: freed slots go to the highest-priority pending kernel, then the one issued first.
      pending.sort((a, b) => b.prio - a.prio || a.seq - b.seq);
      for (let k = 0; k < SLOTS; k++) {
        if (slots[k]) continue;
        const o = pending.find((p) => p.next < p.blocks.length);
        if (!o) break;
        const b = o.blocks[o.next++];
        b.slot = k;
        b.start = t;
        b.end = t + b.dur;
        slots[k] = b;
        allBlocks.push(b);
        if (o.start === null) o.start = t;
        changed = true;
      }
      pending = pending.filter((p) => p.next < p.blocks.length);
      // The host.
      if (cur && cur.done(t)) {
        cur.end = t;
        done.push(cur);
        cur = null;
        pc++;
        changed = true;
      }
      if (!cur && pc < calls.length) {
        cur = startCall(t);
        changed = true;
      }
    }
    // The next moment anything can happen.
    let next = Infinity;
    const later = (x: number | null | undefined) => {
      if (x !== null && x !== undefined && x > t + EPS && x < next) next = x;
    };
    for (const b of slots) if (b) later(b.end);
    for (const s of ['internal', 'top', 'bottom'] as StreamId[])
      for (const o of streams[s]) {
        later(o.inserted);
        if (o.start !== null) later(o.end);
      }
    for (const x of xchs) later(x.end);
    if (cur) later(cur.ready);
    if (next === Infinity) {
      stuck = !!cur || pc < calls.length || (['internal', 'top', 'bottom'] as StreamId[]).some((s) => !streamIdle(s, t));
      break;
    }
    t = next;
  }
  if (stuck) throw new Error('haloloop: the schedule deadlocked');

  // ---------------------------------------------------------------- drawing material
  const segs: Seg[] = [];
  const marks: Mark[] = [];
  const waits: WaitRun[] = [];
  for (const h of done) {
    const a = h.spec.act;
    const blocking = a.k !== 'op' || !!a.block;
    segs.push({ lane: 'host', kind: blocking ? 'blocked' : 'call', iter: h.iter, start: h.start, end: h.end as number, line: h.spec.line, label: h.spec.label });
  }
  for (const s of ['top', 'bottom', 'internal'] as StreamId[])
    for (const o of streams[s]) {
      const base = { lane: s as Lane, iter: o.iter, line: o.line, label: o.label };
      const sp = o.spec;
      const st = o.start as number;
      const en = o.end as number;
      if (sp.k === 'memset' || sp.k === 'copy') segs.push({ ...base, kind: sp.k, start: st, end: en });
      else if (sp.k === 'record') marks.push({ lane: s, iter: o.iter, at: en, event: sp.ev, line: o.line });
      else if (sp.k === 'wait') {
        waits.push({
          lane: s, iter: o.iter, event: sp.ev, line: o.line, reached: o.reached as number, end: en,
          from: o.target ? (o.target.end as number) : null, fromLane: o.target ? o.target.stream : null,
        });
        if (en > (o.reached as number) + EPS)
          segs.push({ ...base, kind: 'wait', start: o.reached as number, end: en, event: sp.ev, from: o.target?.end ?? undefined });
      } else if (sp.k === 'kernel') {
        if (st > (o.reached as number) + EPS) segs.push({ ...base, kind: 'pending', start: o.reached as number, end: st, part: sp.part });
        segs.push({ ...base, kind: 'kernel', start: st, end: en, part: sp.part });
      } else if (sp.k === 'exchange') segs.push({ ...base, kind: 'exchange', start: st, end: en });
      else segs.push({ ...base, kind: 'barrier', start: o.reached as number, end: en });
    }
  for (const x of xchs)
    segs.push({
      lane: x.which === 'top' ? 'linkTop' : 'linkBottom', kind: 'exchange', iter: x.iter, start: x.start, end: x.end,
      line: '', label: x.which === 'top' ? tr('top exchange', 'trao đổi phía trên') : tr('bottom exchange', 'trao đổi phía dưới'),
    });
  const blocks: BlockRun[] = allBlocks.map((b) => ({
    slot: b.slot, start: b.start as number, end: b.end as number, part: b.part, iter: b.op.iter, whole: b.op.spec.k === 'kernel' && b.op.spec.part === 'whole',
  }));

  // ---------------------------------------------------------------- per iteration
  const kernelsOf = (i: number) => (['top', 'bottom', 'internal'] as StreamId[]).flatMap((s) => streams[s].filter((o) => o.iter === i && o.spec.k === 'kernel'));
  const partSpan = (i: number, p: Part): Span => {
    const bs = allBlocks.filter((b) => b.op.iter === i && b.part === p);
    return { start: Math.min(...bs.map((b) => b.start as number)), end: Math.max(...bs.map((b) => b.end as number)) };
  };
  const opSpan = (i: number, k: OpSpec['k']): Span => {
    const o = streams.internal.find((x) => x.iter === i && x.spec.k === k) as Op;
    return { start: o.start as number, end: o.end as number };
  };
  const xSpan = (i: number, w: Which): Span => {
    const x = xchs.find((e) => e.iter === i && e.which === w) as Xch;
    return { start: x.start, end: x.end };
  };
  const iters: IterStats[] = [];
  for (let i = 1; i <= iterations; i++) {
    const hs = done.filter((h) => h.iter === i);
    const ar = hs.find((h) => h.spec.act.k === 'allreduce') as HostCall;
    const top = partSpan(i, 'top');
    const bottom = partSpan(i, 'bottom');
    const internal = partSpan(i, 'internal');
    const exTop = xSpan(i, 'top');
    const exBottom = xSpan(i, 'bottom');
    iters.push({
      iter: i,
      hostStart: hs[0].start,
      hostEnd: hs[hs.length - 1].end as number,
      hostBlocked: hs.filter((h) => h.spec.act.k !== 'op' || h.spec.act.block).reduce((s, h) => s + ((h.end as number) - h.start), 0),
      memset: opSpan(i, 'memset'),
      copy: opSpan(i, 'copy'),
      allreduce: { start: ar.start, end: ar.end as number },
      top, bottom, internal, exTop, exBottom,
      bottomGap: Math.max(0, exBottom.start - bottom.end),
      hidden: Math.max(exTop.end, exBottom.end) <= internal.end + EPS,
    });
  }

  // ---------------------------------------------------------------- what went wrong
  const violations: Violation[] = [];
  const PART_NAME: Record<Part, string> = { top: tr('top', 'phía trên'), bottom: tr('bottom', 'phía dưới'), internal: tr('internal', 'bên trong') };
  for (const it of iters) {
    const i = it.iter;
    for (const k of kernelsOf(i)) {
      const part = (k.spec as { part: Part | 'whole' }).part;
      // The norm is zeroed before any block of the kernel adds to it.
      const early = k.blocks.filter((b) => (b.end as number) < it.memset.end - EPS);
      if (early.length) {
        const first = early.reduce((a, b) => ((a.end as number) <= (b.end as number) ? a : b));
        const p = first.part;
        violations.push({
          kind: 'reset', iter: i, edge: `reset-${p}` as EdgeId, lane: k.stream, at: first.end as number, until: it.memset.end, // i18n-ignore: edge id
          text: tr(
            `Iteration ${i}: a block of the ${PART_NAME[p]} ${part === 'whole' ? 'rows' : 'kernel'} adds its squared residues to the norm at ${r2(first.end as number)}, but the memset zeroes it only at ${r2(it.memset.end)}. That part is lost: the norm comes out too small.`,
            `Vòng lặp ${i}: một khối của ${part === 'whole' ? 'các hàng' : 'kernel'} ${PART_NAME[p]} cộng các phần dư bình phương của nó vào chuẩn tại ${r2(first.end as number)}, nhưng memset chỉ đặt chuẩn về 0 tại ${r2(it.memset.end)}. Phần đó bị mất: chuẩn tính ra nhỏ hơn thực tế.`,
          ),
        });
      }
      // Every kernel has added its part before the copy reads the norm.
      if (it.copy.start < (k.end as number) - EPS) {
        const p = part === 'whole' ? 'internal' : part;
        violations.push({
          kind: 'copy', iter: i, edge: `${p}-copy` as EdgeId, lane: 'internal', at: it.copy.start, until: k.end as number, // i18n-ignore: edge id
          text: tr(
            `Iteration ${i}: the norm is copied to the host at ${r2(it.copy.start)}, but the ${PART_NAME[p]} kernel adds its last part only at ${r2(k.end as number)}. The host reduces a norm that is missing it.`,
            `Vòng lặp ${i}: chuẩn được sao chép về host tại ${r2(it.copy.start)}, nhưng kernel ${PART_NAME[p]} chỉ cộng phần cuối của nó tại ${r2(k.end as number)}. Host rút gọn một chuẩn còn thiếu phần đó.`,
          ),
        });
      }
    }
    // Rows are computed before they are sent.
    for (const w of ['top', 'bottom'] as Which[]) {
      const x = w === 'top' ? it.exTop : it.exBottom;
      const rows = w === 'top' ? it.top : it.bottom;
      if (x.start < rows.end - EPS)
        violations.push({
          kind: 'send', iter: i, edge: `${w}-exchange` as EdgeId, lane: w === 'top' ? 'linkTop' : 'linkBottom', at: x.start, until: rows.end, // i18n-ignore: edge id
          text: tr(
            `Iteration ${i}: the ${w} exchange starts sending at ${r2(x.start)}, but the ${w} kernel finishes the row only at ${r2(rows.end)}. The neighbour receives a row that is not yet computed.`,
            `Vòng lặp ${i}: trao đổi ${w === 'top' ? 'phía trên' : 'phía dưới'} bắt đầu gửi tại ${r2(x.start)}, nhưng kernel ${w === 'top' ? 'phía trên' : 'phía dưới'} chỉ tính xong hàng đó tại ${r2(rows.end)}. Bên láng giềng nhận một hàng chưa được tính.`,
          ),
        });
    }
    // The halos of iteration i-1 have arrived before iteration i reads them. The top halo
    // arrives with the bottom exchange, the bottom halo with the top exchange.
    if (i > 1) {
      const prev = iters[i - 2];
      const readers: [Which, Span, Span][] = [
        ['top', it.top, prev.exBottom],
        ['bottom', it.bottom, prev.exTop],
      ];
      for (const [w, rows, arrive] of readers)
        if (rows.start < arrive.end - EPS)
          violations.push({
            kind: 'halo', iter: i, edge: 'exchange-next', lane: one ? 'internal' : w, at: rows.start, until: arrive.end,
            text: tr(
              `Iteration ${i}: the ${w} ${one ? 'rows start' : 'kernel starts'} at ${r2(rows.start)} and read${one ? '' : 's'} the ${w} halo, which arrives only at ${r2(arrive.end)} (with the ${w === 'top' ? 'bottom' : 'top'} exchange of iteration ${i - 1}). ${one ? 'They read' : 'It reads'} stale values.`,
              `Vòng lặp ${i}: ${one ? 'các hàng' : 'kernel'} ${w === 'top' ? 'phía trên' : 'phía dưới'} bắt đầu tại ${r2(rows.start)} và đọc vùng biên (halo) ${w === 'top' ? 'phía trên' : 'phía dưới'}, vốn chỉ đến tại ${r2(arrive.end)} (qua trao đổi ${w === 'top' ? 'phía dưới' : 'phía trên'} của vòng lặp ${i - 1}). ${one ? 'Chúng đọc' : 'Nó đọc'} giá trị cũ.`,
            ),
          });
    }
    // The copy is done before the host reduces the norm.
    if (it.allreduce.start < it.copy.end - EPS)
      violations.push({
        kind: 'copy', iter: i, edge: 'copy-allreduce', lane: 'host', at: it.allreduce.start, until: it.copy.end,
        text: tr(
          `Iteration ${i}: MPI_Allreduce starts at ${r2(it.allreduce.start)}, before the copy of the norm ends at ${r2(it.copy.end)}.`,
          `Vòng lặp ${i}: MPI_Allreduce bắt đầu tại ${r2(it.allreduce.start)}, trước khi lần sao chép chuẩn kết thúc tại ${r2(it.copy.end)}.`,
        ),
      });
    // A pageable host buffer turns the asynchronous copy into a wait.
    if (defect === 'pageable') {
      const h = done.find((x) => x.iter === i && x.spec.act.k === 'op' && x.spec.act.op.k === 'copy') as HostCall;
      violations.push({
        kind: 'stall', iter: i, edge: null, lane: 'host', at: h.start, until: h.end as number,
        text: tr(
          `Iteration ${i}: cudaMemcpyAsync into pageable memory is synchronous: the host waits in it from ${r2(h.start)} to ${r2(h.end as number)}, and nothing after it can be issued until then.`,
          `Vòng lặp ${i}: cudaMemcpyAsync vào bộ nhớ phân trang (pageable) là đồng bộ: host chờ trong lời gọi đó từ ${r2(h.start)} đến ${r2(h.end as number)}, và không lệnh nào sau nó được phát cho tới lúc đó.`,
        ),
      });
    }
  }
  violations.sort((a, b) => a.at - b.at);

  const lastIt = iters[iters.length - 1];
  const end = Math.max(
    ...segs.map((s) => s.end),
    ...marks.map((m) => m.at),
  );
  return {
    input,
    lanes: one ? ['host', 'internal', 'linkTop', 'linkBottom'] : ['host', 'top', 'bottom', 'internal', 'linkTop', 'linkBottom'],
    segs, marks, waits, blocks, insertions, iters,
    iterTime: iters.length > 1 ? lastIt.hostEnd - iters[iters.length - 2].hostEnd : lastIt.hostEnd,
    hostBlocked: lastIt.hostBlocked,
    bottomGap: lastIt.bottomGap,
    hidden: lastIt.hidden,
    violations,
    end,
  };
}

// ---------------------------------------------------------------- the dependency puzzle

export type EdgeId =
  | 'reset-top'
  | 'reset-bottom'
  | 'reset-internal'
  | 'top-copy'
  | 'bottom-copy'
  | 'internal-copy'
  | 'top-exchange'
  | 'bottom-exchange'
  | 'copy-allreduce'
  | 'exchange-next';

/**
 * How an ordering is enforced:
 *   stream    both operations are in the same stream, in that order
 *   event     an event recorded in one stream and a wait queued in another
 *   host      the host waits for a stream (or the device) before issuing the second
 *   blocking  the call that does the first returns only when it is complete
 *   barrier   a barrier queued in the stream (NVSHMEM)
 *   kernel    program order inside one kernel (NVSHMEM's puts)
 */
export type Mechanism = 'stream' | 'event' | 'host' | 'blocking' | 'barrier' | 'kernel';

export interface Edge {
  id: EdgeId;
  first: string;
  then: string;
  mechanism: Mechanism;
  /** The figure lines that enforce it. */
  lines: string;
}

export const EDGE_IDS: EdgeId[] = [
  'reset-top', 'reset-bottom', 'reset-internal', 'top-copy', 'bottom-copy', 'internal-copy',
  'top-exchange', 'bottom-exchange', 'copy-allreduce', 'exchange-next',
];

const WHAT: Record<EdgeId, [string, string]> = {
  get 'reset-top'(): [string, string] {
    return [tr('the norm is zeroed', 'chuẩn được đặt về 0'), tr('the top kernel adds to it', 'kernel phía trên cộng vào nó')];
  },
  get 'reset-bottom'(): [string, string] {
    return [tr('the norm is zeroed', 'chuẩn được đặt về 0'), tr('the bottom kernel adds to it', 'kernel phía dưới cộng vào nó')];
  },
  get 'reset-internal'(): [string, string] {
    return [tr('the norm is zeroed', 'chuẩn được đặt về 0'), tr('the internal kernel adds to it', 'kernel bên trong cộng vào nó')];
  },
  get 'top-copy'(): [string, string] {
    return [tr('the top kernel has added its part', 'kernel phía trên đã cộng phần của nó'), tr('the norm is copied to the host', 'chuẩn được sao chép về host')];
  },
  get 'bottom-copy'(): [string, string] {
    return [tr('the bottom kernel has added its part', 'kernel phía dưới đã cộng phần của nó'), tr('the norm is copied to the host', 'chuẩn được sao chép về host')];
  },
  get 'internal-copy'(): [string, string] {
    return [tr('the internal kernel has added its part', 'kernel bên trong đã cộng phần của nó'), tr('the norm is copied to the host', 'chuẩn được sao chép về host')];
  },
  get 'top-exchange'(): [string, string] {
    return [tr('the top kernel has computed row 1', 'kernel phía trên đã tính hàng 1'), tr('the top exchange sends it', 'trao đổi phía trên gửi hàng đó đi')];
  },
  get 'bottom-exchange'(): [string, string] {
    return [tr('the bottom kernel has computed row ny − 2', 'kernel phía dưới đã tính hàng ny − 2'), tr('the bottom exchange sends it', 'trao đổi phía dưới gửi hàng đó đi')];
  },
  get 'copy-allreduce'(): [string, string] {
    return [tr('the copy of the norm has landed in l2norm_h', 'lần sao chép chuẩn đã về tới l2norm_h'), tr('MPI_Allreduce reads it', 'MPI_Allreduce đọc nó')];
  },
  get 'exchange-next'(): [string, string] {
    return [tr('both exchanges have delivered their halos', 'cả hai trao đổi đã chuyển giao vùng biên (halo) của chúng'), tr("the next iteration's kernels read them", 'các kernel của vòng lặp kế tiếp đọc chúng')];
  },
};

const HOW: Record<Version, Record<EdgeId, [Mechanism, string]>> = {
  mpi: {
    'reset-top': ['stream', '04, 07'],
    'reset-bottom': ['stream', '04, 07'],
    'reset-internal': ['stream', '04, 07'],
    'top-copy': ['stream', '07, 21'],
    'bottom-copy': ['stream', '07, 21'],
    'internal-copy': ['stream', '07, 21'],
    'top-exchange': ['host', '08'],
    'bottom-exchange': ['host', '08'],
    'copy-allreduce': ['blocking', '21'],
    'exchange-next': ['blocking', '13–18'],
  },
  streams: {
    'reset-top': ['event', '05, 06'],
    'reset-bottom': ['event', '05, 07'],
    'reset-internal': ['stream', '04, 15'],
    'top-copy': ['event', '11, 19'],
    'bottom-copy': ['event', '14, 20'],
    'internal-copy': ['stream', '15, 21'],
    'top-exchange': ['host', '27'],
    'bottom-exchange': ['host', '31'],
    'copy-allreduce': ['host', '37'],
    'exchange-next': ['blocking', '28–34'],
  },
  nccl: {
    'reset-top': ['event', '05, 06'],
    'reset-bottom': ['event', '05, 07'],
    'reset-internal': ['stream', '04, 15'],
    'top-copy': ['event', '11, 19'],
    'bottom-copy': ['event', '14, 20'],
    'internal-copy': ['stream', '15, 21'],
    'top-exchange': ['stream', '10, 27–30'],
    'bottom-exchange': ['stream', '12, 32–35'],
    'copy-allreduce': ['host', '39'],
    'exchange-next': ['event', '31, 36, 43–44'],
  },
  nvshmem: {
    'reset-top': ['stream', '04, 09'],
    'reset-bottom': ['stream', '04, 09'],
    'reset-internal': ['stream', '04, 09'],
    'top-copy': ['stream', '09, 12'],
    'bottom-copy': ['stream', '09, 12'],
    'internal-copy': ['stream', '09, 12'],
    'top-exchange': ['kernel', 'Fig. 23.23'],
    'bottom-exchange': ['kernel', 'Fig. 23.23'],
    'copy-allreduce': ['host', '14'],
    'exchange-next': ['barrier', '18'],
  },
};

/** Each ordering one iteration needs, and what enforces it in the version's figure. */
export function edges(v: Version): Edge[] {
  return EDGE_IDS.map((id) => ({ id, first: WHAT[id][0], then: WHAT[id][1], mechanism: HOW[v][id][0], lines: HOW[v][id][1] }));
}

/** The orderings whose mechanism differs between two versions. */
export function changedEdges(from: Version, to: Version): EdgeId[] {
  return EDGE_IDS.filter((id) => HOW[from][id][0] !== HOW[to][id][0]);
}

// ---------------------------------------------------------------- the three launches of Fig. 23.14

/**
 * Global rows each launch of Fig. 23.14 writes. A launch gets `rows` rows starting at
 * global row `first`; Fig. 23.1's guard (y < rows - 1 with y counted from 1) admits local
 * rows 1 to rows - 2 (ours, from the notes).
 */
export function launchRows(ny: number): { top: number[]; bottom: number[]; internal: number[] } {
  const written = (first: number, rows: number) => {
    const out: number[] = [];
    for (let y = 1; y < rows - 1; y++) out.push(first + y);
    return out;
  };
  return { top: written(0, 3), bottom: written(ny - 3, 3), internal: written(1, ny - 2) };
}
