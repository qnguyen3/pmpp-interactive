// The chunked copy-in, compute, copy-out pipeline of section 23.3, as a small
// discrete-event schedule. The book overlaps a halo exchange with the internal rows
// of a stencil (pp. 553-563); this model applies the same idea to one GPU and the
// host-device link, the staged loop NVIDIA prints in its Best Practices Guide 10.1.2
// (ch23-facts A7). It drives the section's stepper, its two predictions, the
// "pinned" scene and the "your device" preset.
//
// The job: copy-in, kernel and copy-out with totals a, b and c (unitless). It is cut
// into n chunks (equal, or [k*items/n, (k+1)*items/n) of an element count, the
// lab's and the snippet's split, with empty chunks skipped). The j-th chunk queued
// goes to stream j mod S. Each operation also costs a fixed overhead o, a figure no
// vendor publishes (E3).
//
// CUDA and HIP view. The host thread queues the operations in issue order, depth
// first (chunk by chunk) or breadth first (all copy-ins, all kernels, all
// copy-outs), and with pinned memory every call returns at once. An operation
// starts when all of these hold:
//   - the operation before it in its stream has finished (p. 556);
//   - with a limited number of hardware queues (CUDA 8, A8; HIP 4, ch22 B1), the
//     operation before it in its queue has finished. Streams map round-robin onto
//     queues, and kernels and copies have separate queues, as CUDA documents its
//     "compute and copy engine connections" (A8). HIP's mapping of copies is not
//     documented; the model assumes the same;
//   - its resource is free, and every operation issued to that resource before it
//     has started: each resource serves its operations in issue order (strict FIFO,
//     our assumption; NVIDIA does not document it for one copy engine, E2). The
//     resources are one kernel at a time on the GPU, and either a copy engine per
//     direction or one copy engine shared by both (A7).
// With pageable host memory a copy makes the host thread wait for its stream and
// then take part in the copy until it ends, the worst case the documentation
// allows (A6: "may synchronize with the stream and stage the copy"; B4).
//
// Metal view (Apple silicon, C4, C5). A shared buffer is the GPU's memory, so the
// copy-in becomes the CPU writing a chunk into one of K buffers and the copy-out
// the CPU reading a finished chunk back. One CPU thread does, for chunk j: wait
// until slot j mod K's last command buffer has completed, read that chunk out
// (drain), write chunk j in (fill), commit. The GPU runs the command buffers one
// at a time in commit order. After the loop it drains what is still in flight,
// oldest first.
//
// Everything is relative time; nothing here is a device figure.
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

export type Order = 'depth' | 'breadth';
export type HostMem = 'pinned' | 'pageable';
export type View = 'cuda' | 'metal';
export type Kind = 'in' | 'kernel' | 'out';
export type Lane = 'host' | 'h2d' | 'compute' | 'd2h' | 'copy' | 'cpu' | 'gpu';
export type Cause = 'stream' | 'queue' | 'engine' | 'slot';

/** CUDA_DEVICE_MAX_CONNECTIONS default (ch23-facts A8). */
export const CUDA_QUEUES = 8;
/** GPU_MAX_HW_QUEUES default (ch22-facts B1, cited in ch23-facts B1). */
export const HIP_QUEUES = 4;
export const MAX_CHUNKS = 64;
export const MAX_STREAMS = 16;

export interface PipeConfig {
  /** Total copy-in time of the whole job (Metal: CPU fill). */
  a: number;
  /** Total kernel time. */
  b: number;
  /** Total copy-out time (Metal: CPU drain). 0: nothing is copied back. */
  c: number;
  /** Chunks, 1 to 64. */
  n: number;
  /** Streams (CUDA, HIP) or buffers in flight K (Metal). */
  streams: number;
  /** Copy engines: one per direction, or one shared by both. CUDA and HIP view only. */
  engines: 1 | 2;
  order: Order;
  /** Fixed cost added to every operation; no vendor publishes it (E3). */
  overhead: number;
  host: HostMem;
  /** Hardware queues the streams share round-robin; Infinity: one per stream. */
  queues: number;
  view: View;
  /** Optional element count: chunks become [k*items/n, (k+1)*items/n), empty ones skipped. */
  items?: number;
}

export const DEFAULTS: PipeConfig = {
  a: 8, b: 16, c: 8, n: 8, streams: 8, engines: 2, order: 'depth', overhead: 0, host: 'pinned', queues: Infinity, view: 'cuda',
};

// ---------------------------------------------------------------- the snippets
/** The files the stepper shows, under labs/. */
export const SNIPPET = {
  cuda: 'ch23/snippets/staged_loop.cu',
  metal: 'ch23/snippets/staged_loop_metal.mm',
  metalKernel: 'ch23/snippets/staged_loop.metal',
};
/** Lines of the snippets that issue each operation (pinned by S233.snippets.test.ts). */
export const LINES = {
  depth: { in: 47, kernel: 48, out: 49 },
  breadth: { in: 56, kernel: 57, out: 58 },
  /** The runtime call inside copyIn, compute and copyOut. */
  api: { in: 35, kernel: 38, out: 41 },
  metal: { drain: 31, fill: 32, commit: 41, tail: 45, wait: 22, read: 23 },
};

// ---------------------------------------------------------------- results
export interface Wait {
  /** When the operation could otherwise have started. */
  at: number;
  cause: Cause;
  /** The operation it waits for. */
  on: number;
  text: string;
}
export interface Op {
  /** Issue order. */
  id: number;
  kind: Kind;
  /** j: the chunk's number among the chunks queued. */
  chunk: number;
  /** Stream (CUDA, HIP) or buffer slot (Metal). */
  stream: number;
  /** Hardware queue (equal to the stream when queues are unlimited); -1 on Metal. */
  queue: number;
  lane: Lane;
  issue: number;
  start: number;
  end: number;
  /** Snippet lines that issued it: the call site first. */
  lines: number[];
  wait: Wait | null;
  /** Metal drains: issued after the loop. */
  tail?: boolean;
}
export interface HostSpan {
  from: number;
  to: number;
  /** blocked: in a pageable copy, waiting for its stream; staging: taking part in it; waiting: Metal, for a slot. */
  kind: 'blocked' | 'staging' | 'waiting';
  op: number;
  lane: Lane;
  text: string;
  lines: number[];
}
export interface PipeEvent {
  t: number;
  kind: 'issue' | 'start' | 'end' | 'wait' | 'block' | 'stage' | 'return' | 'idle';
  op: number | null;
  text: string;
  lines: number[];
}
export interface Step {
  t: number;
  events: PipeEvent[];
}
export interface Condition {
  id: string;
  text: string;
  holds: boolean;
}
export interface ClosedForm {
  name: string;
  formula: string;
  value: number;
  conditions: Condition[];
  holds: boolean;
  /** The schedule's total equals the formula's value. */
  matches: boolean;
}
export interface Schedule {
  cfg: PipeConfig;
  /** Lanes in drawing order. */
  lanes: Lane[];
  ops: Op[];
  host: HostSpan[];
  /** Fractions of the data, one per chunk queued. */
  shares: number[];
  total: number;
  /** The unsplit job: one copy-in, one kernel, one copy-out. */
  serial: number;
  speedup: number;
  busy: Partial<Record<Lane, number>>;
  bottleneck: Lane;
  /** Idle time of the bottleneck lane, and where it is. */
  bubble: number;
  bubbles: { from: number; to: number }[];
  steps: Step[];
  e1: ClosedForm | null;
  nvidia: ClosedForm | null;
  /** Metal: no closed form, only the lower bound n·max(fill + drain, kernel). */
  lowerBound: number | null;
}

const EPS = 1e-9;
const r9 = (x: number) => Math.round(x * 1e9) / 1e9;
const fmtT = (x: number) => String(Number(x.toFixed(3)));

export const LANE_NAME: Record<Lane, string> = {
  get host() {
    return tr('Host thread', 'Luồng host');
  },
  get h2d() {
    return tr('Copy-in engine', 'Copy engine chiều vào');
  },
  get compute() {
    return tr('Compute', 'Tính toán');
  },
  get d2h() {
    return tr('Copy-out engine', 'Copy engine chiều ra');
  },
  get copy() {
    return tr('Copy engine (both ways)', 'Copy engine (cả hai chiều)');
  },
  get cpu() {
    return tr('CPU: drain, then fill', 'CPU: rút, rồi nạp');
  },
  gpu: 'GPU',
};

/** Fractions of the data in each chunk that is queued (empty chunks skipped). */
export function chunkShares(n: number, items?: number): number[] {
  const k = Math.max(1, Math.floor(n));
  if (items == null) return Array.from({ length: k }, () => 1 / k);
  const N = Math.max(0, Math.floor(items));
  const out: number[] = [];
  for (let c = 0; c < k; c++) {
    const len = Math.floor(((c + 1) * N) / k) - Math.floor((c * N) / k);
    if (len > 0) out.push(len / N);
  }
  return out;
}

const STAGES: Kind[] = ['in', 'kernel', 'out'];
const total = (cfg: PipeConfig, kind: Kind) => (kind === 'in' ? cfg.a : kind === 'kernel' ? cfg.b : cfg.c);
const present = (cfg: PipeConfig) => STAGES.filter((k) => total(cfg, k) > 0);

/** "chunk 2's copy-in", "chunk 0's kernel". */
export function opLabel(op: Pick<Op, 'kind' | 'chunk'>, view: View = 'cuda'): string {
  const what = view === 'metal'
    ? { in: tr('fill', 'lần nạp'), kernel: tr('command buffer', 'command buffer'), out: tr('drain', 'lần rút') }[op.kind]
    : { in: tr('copy-in', 'lần sao chép vào'), kernel: 'kernel', out: tr('copy-out', 'lần sao chép ra') }[op.kind];
  return tr(`chunk ${op.chunk}'s ${what}`, `${what} của chunk ${op.chunk}`);
}
const doing = (op: Op, view: View) =>
  view === 'metal'
    ? {
        in: tr(`filling chunk ${op.chunk}`, `đang nạp chunk ${op.chunk}`),
        kernel: tr(`running chunk ${op.chunk}'s command buffer`, `đang chạy command buffer của chunk ${op.chunk}`),
        out: tr(`draining chunk ${op.chunk}`, `đang rút chunk ${op.chunk}`),
      }[op.kind]
    : {
        in: tr(`copying in chunk ${op.chunk}`, `đang sao chép chunk ${op.chunk} vào`),
        kernel: tr(`running chunk ${op.chunk}'s kernel`, `đang chạy kernel của chunk ${op.chunk}`),
        out: tr(`copying out chunk ${op.chunk}`, `đang sao chép chunk ${op.chunk} ra`),
      }[op.kind];

function laneFor(cfg: PipeConfig, kind: Kind): Lane {
  if (cfg.view === 'metal') return kind === 'kernel' ? 'gpu' : 'cpu';
  if (kind === 'kernel') return 'compute';
  if (cfg.engines === 1) return 'copy';
  return kind === 'in' ? 'h2d' : 'd2h';
}

/** Hardware queue of chunk j's stream. */
const queueOf = (cfg: PipeConfig, stream: number) => (isFinite(cfg.queues) ? stream % cfg.queues : stream);

// ---------------------------------------------------------------- the schedule
export function simulate(input: PipeConfig): Schedule {
  const cfg: PipeConfig = {
    ...input,
    n: Math.max(1, Math.min(MAX_CHUNKS, Math.floor(input.n))),
    streams: Math.max(1, Math.floor(input.streams)),
    a: Math.max(0, input.a), b: Math.max(0, input.b), c: Math.max(0, input.c),
    overhead: Math.max(0, input.overhead),
  };
  const shares = chunkShares(cfg.n, cfg.items);
  const dur = (kind: Kind, j: number) => total(cfg, kind) * shares[j] + cfg.overhead;
  const ops: Op[] = [];
  const host: HostSpan[] = [];
  if (cfg.view === 'metal') metalSchedule(cfg, shares, dur, ops, host);
  else cudaSchedule(cfg, shares, dur, ops, host);

  const end = Math.max(0, ...ops.map((o) => o.end), ...host.map((h) => h.to));
  const stages = present(cfg);
  const serial = r9(cfg.a + cfg.b + cfg.c + stages.length * cfg.overhead);
  const lanes: Lane[] = cfg.view === 'metal'
    ? ['cpu', 'gpu']
    : ['host', ...(cfg.engines === 2 ? (['h2d', 'compute', 'd2h'] as Lane[]) : (['copy', 'compute'] as Lane[]))];

  // Busy time per lane; the bottleneck is the busiest resource.
  const busy: Partial<Record<Lane, number>> = {};
  for (const l of lanes) busy[l] = 0;
  for (const o of ops) busy[o.lane] = (busy[o.lane] ?? 0) + (o.end - o.start);
  for (const h of host) if (h.lane === 'host') busy.host = (busy.host ?? 0) + (h.to - h.from);
  for (const l of lanes) busy[l] = r9(busy[l] ?? 0);
  const candidates = lanes.filter((l) => l !== 'host' || cfg.host === 'pageable');
  const bottleneck = candidates.reduce((best, l) => ((busy[l] ?? 0) > (busy[best] ?? 0) + EPS ? l : best), candidates.find((l) => l !== 'host') ?? candidates[0]);
  const intervals = (bottleneck === 'host'
    ? host.filter((h) => h.lane === 'host').map((h) => ({ from: h.from, to: h.to }))
    : ops.filter((o) => o.lane === bottleneck).map((o) => ({ from: o.start, to: o.end }))
  ).sort((x, y) => x.from - y.from);
  const bubbles: { from: number; to: number }[] = [];
  let cur = 0;
  for (const iv of intervals) {
    if (iv.from > cur + EPS) bubbles.push({ from: r9(cur), to: r9(iv.from) });
    cur = Math.max(cur, iv.to);
  }
  if (end > cur + EPS) bubbles.push({ from: r9(cur), to: r9(end) });
  const bubble = r9(bubbles.reduce((s, b) => s + b.to - b.from, 0));

  const sched: Schedule = {
    cfg, lanes, ops, host, shares,
    total: r9(end), serial, speedup: end > 0 ? serial / end : 1,
    busy, bottleneck, bubble, bubbles,
    steps: [], e1: null, nvidia: null, lowerBound: null,
  };
  sched.steps = buildSteps(sched);
  if (cfg.view === 'cuda') {
    sched.e1 = e1Form(sched);
    sched.nvidia = nvidiaForm(sched);
  } else {
    const m = shares.length;
    const per = (k: Kind) => (total(cfg, k) > 0 ? total(cfg, k) / m + cfg.overhead : 0);
    sched.lowerBound = r9(m * Math.max(per('in') + per('out'), per('kernel')));
  }
  return sched;
}

interface Constraint { t: number; cause: Cause | 'issue'; on: number }
const PREFER: (Cause | 'issue')[] = ['stream', 'queue', 'engine', 'slot', 'issue'];

function cudaSchedule(cfg: PipeConfig, shares: number[], dur: (k: Kind, j: number) => number, ops: Op[], host: HostSpan[]) {
  const m = shares.length;
  const stages = present(cfg);
  const issueList: { kind: Kind; j: number }[] = [];
  if (cfg.order === 'depth') for (let j = 0; j < m; j++) for (const k of stages) issueList.push({ kind: k, j });
  else for (const k of stages) for (let j = 0; j < m; j++) issueList.push({ kind: k, j });
  const lines = cfg.order === 'depth' ? LINES.depth : LINES.breadth;

  const streamLast = new Map<number, number>();
  const queueLast = new Map<string, number>();
  const laneLast = new Map<Lane, number>();
  let hostT = 0;
  for (const { kind, j } of issueList) {
    const stream = j % cfg.streams;
    const queue = queueOf(cfg, stream);
    const lane = laneFor(cfg, kind);
    const qKey = (kind === 'kernel' ? 'k' : 'c') + queue;
    const id = ops.length;
    const cons: Constraint[] = [{ t: hostT, cause: 'issue', on: -1 }];
    const sPrev = streamLast.get(stream);
    if (sPrev != null) cons.push({ t: ops[sPrev].end, cause: 'stream', on: sPrev });
    const qPrev = isFinite(cfg.queues) ? queueLast.get(qKey) : undefined;
    if (qPrev != null && qPrev !== sPrev) cons.push({ t: ops[qPrev].end, cause: ops[qPrev].stream === stream ? 'stream' : 'queue', on: qPrev });
    const lPrev = laneLast.get(lane);
    if (lPrev != null) cons.push({ t: ops[lPrev].end, cause: 'engine', on: lPrev });
    const start = Math.max(...cons.map((c) => c.t));
    const binding = [...cons].sort((x, y) => y.t - x.t || PREFER.indexOf(x.cause) - PREFER.indexOf(y.cause))[0];
    const op: Op = {
      id, kind, chunk: j, stream, queue, lane,
      issue: r9(hostT), start: r9(start), end: r9(start + dur(kind, j)),
      lines: [lines[kind], LINES.api[kind]], wait: null,
    };
    if (binding.cause !== 'issue' && start > hostT + EPS) {
      const others = cons.filter((c) => c !== binding);
      const at = Math.max(...others.map((c) => c.t));
      if (start > at + EPS) op.wait = { at: r9(at), cause: binding.cause, on: binding.on, text: '' };
    }
    ops.push(op);
    if (op.wait) op.wait.text = waitText(op, ops, cfg);
    streamLast.set(stream, id);
    if (isFinite(cfg.queues)) queueLast.set(qKey, id);
    laneLast.set(lane, id);
    if (cfg.host === 'pageable' && kind !== 'kernel') {
      const api = kind === 'in' ? tr('cudaMemcpyAsync (host to device)', 'cudaMemcpyAsync (host sang device)') : tr('cudaMemcpyAsync (device to host)', 'cudaMemcpyAsync (device sang host)');
      if (op.start > op.issue + EPS)
        host.push({
          from: op.issue, to: op.start, kind: 'blocked', op: id, lane: 'host', lines: op.lines,
          text: tr(
            `The host thread is stuck in ${api} for ${opLabel(op)}: the buffer is pageable, so the call first waits for ${binding.cause === 'stream' ? `stream ${stream} to finish ${opLabel(ops[binding.on])}` : binding.cause === 'queue' ? `queue ${queue} to finish ${opLabel(ops[binding.on])}` : `the ${LANE_NAME[lane].toLowerCase()} to finish ${opLabel(ops[binding.on])}`}.`,
            `Luồng host bị kẹt trong ${api} cho ${opLabel(op)}: buffer là bộ nhớ phân trang (pageable), nên lời gọi trước hết chờ ${binding.cause === 'stream' ? `stream ${stream} làm xong ${opLabel(ops[binding.on])}` : binding.cause === 'queue' ? `hàng đợi ${queue} làm xong ${opLabel(ops[binding.on])}` : `${LANE_NAME[lane].toLowerCase()} làm xong ${opLabel(ops[binding.on])}`}.`,
          ),
        });
      host.push({
        from: op.start, to: op.end, kind: 'staging', op: id, lane: 'host', lines: op.lines,
        text: tr(
          `The host thread takes part in ${opLabel(op)}, staging it through pinned memory; the call returns only when the copy has finished.`,
          `Luồng host tham gia vào ${opLabel(op)}, trung chuyển dữ liệu qua bộ nhớ ghim (pinned memory); lời gọi chỉ trả về khi lần sao chép đã xong.`,
        ),
      });
      hostT = op.end;
    }
  }
}

function waitText(op: Op, ops: Op[], cfg: PipeConfig): string {
  const w = op.wait!;
  const on = ops[w.on];
  const running = on.start <= w.at + EPS && on.end > w.at + EPS;
  const state = running
    ? tr(`is still ${doing(on, cfg.view)}`, `vẫn ${doing(on, cfg.view)}`)
    : on.start > w.at + EPS
      ? tr(`has not yet started ${opLabel(on, cfg.view)}`, `chưa bắt đầu ${opLabel(on, cfg.view)}`)
      : tr(`has not finished ${opLabel(on, cfg.view)}`, `chưa làm xong ${opLabel(on, cfg.view)}`);
  const who = opLabel(op, cfg.view);
  const head = who.charAt(0).toUpperCase() + who.slice(1) + tr(' waits: ', ' chờ: ');
  switch (w.cause) {
    case 'stream':
      return head + tr(`stream ${op.stream} ${state}.`, `stream ${op.stream} ${state}.`);
    case 'queue':
      return head + tr(
        `hardware queue ${op.queue} also holds stream ${on.stream}, which ${state} (a false dependency between unrelated streams).`,
        `hàng đợi phần cứng ${op.queue} cũng chứa stream ${on.stream}, mà stream này ${state} (một phụ thuộc giả giữa các stream không liên quan đến nhau).`,
      );
    case 'engine': {
      if (cfg.view === 'metal')
        return head + tr(
          `the GPU runs command buffers one at a time, in commit order, and it ${state}.`,
          `GPU chạy lần lượt từng command buffer một, theo thứ tự commit, và nó ${state}.`,
        );
      const gpu = op.lane === 'compute';
      const eng = gpu ? tr('the GPU', 'GPU') : tr(`the ${LANE_NAME[op.lane].toLowerCase()}`, `bộ ${LANE_NAME[op.lane].toLowerCase()}`);
      if (running) return head + (gpu ? tr(`the GPU runs one kernel at a time, and it ${state}.`, `GPU chạy mỗi lần một kernel, và nó ${state}.`) : `${eng} ${state}.`);
      return head + tr(
        `${eng} ${gpu ? 'runs kernels' : 'serves operations'} in the order they were issued, and ${opLabel(on)}, queued ahead of it, ${on.start > w.at + EPS ? 'has not started' : 'has not finished'}${on.wait ? ` (it waits for ${opLabel(ops[on.wait.on])})` : ''}.`,
        `${eng} ${gpu ? 'chạy các kernel' : 'phục vụ các thao tác'} theo thứ tự chúng được phát lệnh, và ${opLabel(on)}, được xếp hàng trước nó, ${on.start > w.at + EPS ? 'chưa bắt đầu' : 'chưa làm xong'}${on.wait ? ` (nó đang chờ ${opLabel(ops[on.wait.on])})` : ''}.`,
      );
    }
    case 'slot':
      return head + tr(`buffer ${op.stream} ${state}.`, `buffer ${op.stream} ${state}.`);
  }
}

function metalSchedule(cfg: PipeConfig, shares: number[], dur: (k: Kind, j: number) => number, ops: Op[], host: HostSpan[]) {
  const m = shares.length;
  const K = cfg.streams;
  const slotChunk: number[] = Array(K).fill(-1);
  const slotGpu: number[] = Array(K).fill(-1);
  let cpu = 0;
  let gpuLast = -1;
  const drain = (k: number, tail: boolean) => {
    const j = slotChunk[k];
    const g = slotGpu[k];
    const site = tail ? LINES.metal.tail : LINES.metal.drain;
    if (g >= 0 && ops[g].end > cpu + EPS) {
      host.push({
        from: r9(cpu), to: ops[g].end, kind: 'waiting', op: g, lane: 'cpu', lines: [site, LINES.metal.wait],
        text: tr(
          `The CPU waits in waitUntilCompleted: buffer ${k} is still in use by ${opLabel(ops[g], 'metal')}, and it must not be touched until that has completed.`,
          `CPU chờ trong waitUntilCompleted: buffer ${k} vẫn đang được ${opLabel(ops[g], 'metal')} dùng, và không được đụng vào nó cho tới khi việc đó hoàn tất.`,
        ),
      });
      cpu = ops[g].end;
    }
    if (cfg.c > 0) {
      const d = dur('out', j);
      ops.push({
        id: ops.length, kind: 'out', chunk: j, stream: k, queue: -1, lane: 'cpu',
        issue: r9(cpu), start: r9(cpu), end: r9(cpu + d), lines: [site, LINES.metal.read], wait: null, tail,
      });
      cpu += d;
    }
    slotChunk[k] = -1;
  };
  for (let j = 0; j < m; j++) {
    const k = j % K;
    if (slotChunk[k] >= 0) drain(k, false);
    if (cfg.a > 0) {
      const d = dur('in', j);
      ops.push({ id: ops.length, kind: 'in', chunk: j, stream: k, queue: -1, lane: 'cpu', issue: r9(cpu), start: r9(cpu), end: r9(cpu + d), lines: [LINES.metal.fill], wait: null });
      cpu += d;
    }
    if (cfg.b > 0) {
      const gEnd = gpuLast >= 0 ? ops[gpuLast].end : 0;
      const start = Math.max(cpu, gEnd);
      const op: Op = { id: ops.length, kind: 'kernel', chunk: j, stream: k, queue: -1, lane: 'gpu', issue: r9(cpu), start: r9(start), end: r9(start + dur('kernel', j)), lines: [LINES.metal.commit], wait: null };
      if (gEnd > cpu + EPS) op.wait = { at: r9(cpu), cause: 'engine', on: gpuLast, text: '' };
      ops.push(op);
      if (op.wait) op.wait.text = waitText(op, ops, cfg);
      gpuLast = op.id;
      slotGpu[k] = op.id;
    } else slotGpu[k] = -1;
    slotChunk[k] = j;
  }
  for (let i = 0; i < K; i++) {
    const k = (m + i) % K;
    if (slotChunk[k] >= 0) drain(k, true);
  }
}

// ---------------------------------------------------------------- stepping
function startText(op: Op, cfg: PipeConfig): string {
  if (cfg.view === 'metal') {
    if (op.kind === 'in') return tr(`The CPU fills buffer ${op.stream} with chunk ${op.chunk}.`, `CPU nạp chunk ${op.chunk} vào buffer ${op.stream}.`);
    if (op.kind === 'out')
      return tr(
        `The CPU reads chunk ${op.chunk}'s result out of buffer ${op.stream}${op.tail ? ' (after the loop)' : ''}.`,
        `CPU đọc kết quả của chunk ${op.chunk} ra khỏi buffer ${op.stream}${op.tail ? ' (sau vòng lặp)' : ''}.`,
      );
    return tr(`The GPU starts chunk ${op.chunk}'s command buffer (buffer ${op.stream}).`, `GPU bắt đầu command buffer của chunk ${op.chunk} (buffer ${op.stream}).`);
  }
  const where = op.kind === 'kernel' ? '' : tr(` on the ${LANE_NAME[op.lane].toLowerCase()}`, ` trên ${LANE_NAME[op.lane].toLowerCase()}`);
  const who = opLabel(op);
  return tr(
    `${who.charAt(0).toUpperCase() + who.slice(1)} starts${where} (stream ${op.stream}).`,
    `${who.charAt(0).toUpperCase() + who.slice(1)} bắt đầu${where} (stream ${op.stream}).`,
  );
}

function buildSteps(s: Schedule): Step[] {
  const { cfg, ops, host } = s;
  const ev: (PipeEvent & { rank: number })[] = [];
  const push = (e: PipeEvent, rank: number) => ev.push({ ...e, t: r9(e.t), rank });
  if (cfg.view === 'cuda' && ops.length) {
    const first = cfg.host === 'pinned' ? ops : ops.filter((o) => o.issue <= EPS);
    push({
      t: 0, kind: 'issue', op: null, lines: [],
      text: cfg.host === 'pinned'
        ? tr(
            `The host thread queues all ${ops.length} operations at time 0; every call returns at once.`,
            `Luồng host xếp hàng cả ${ops.length} thao tác tại thời điểm 0; mọi lời gọi trả về ngay.`,
          )
        : tr(
            `The host thread starts queuing. With pageable buffers each copy holds it until the copy is done, so it gets only ${first.length} operation${first.length === 1 ? '' : 's'} queued before it is stuck.`,
            `Luồng host bắt đầu xếp hàng. Với buffer phân trang (pageable), mỗi lần sao chép giữ luồng lại cho tới khi lần sao chép xong, nên nó chỉ kịp xếp hàng ${first.length} thao tác trước khi bị kẹt.`,
          ),
    }, 0);
  }
  for (const o of ops) {
    const who = opLabel(o, cfg.view);
    push({ t: o.end, kind: 'end', op: o.id, lines: o.lines, text: tr(`${who.charAt(0).toUpperCase() + who.slice(1)} finishes.`, `${who.charAt(0).toUpperCase() + who.slice(1)} xong.`) }, 1);
    push({ t: o.start, kind: 'start', op: o.id, lines: o.lines, text: startText(o, cfg) }, 3);
    if (o.wait) push({ t: o.wait.at, kind: 'wait', op: o.id, lines: o.lines, text: o.wait.text }, 5);
    if (cfg.host === 'pageable' && cfg.view === 'cuda' && o.kind === 'kernel' && o.issue > EPS)
      push({ t: o.issue, kind: 'issue', op: o.id, lines: o.lines, text: tr(`Only now can the host thread queue ${opLabel(o)}.`, `Đến lúc này luồng host mới xếp hàng được ${opLabel(o)}.`) }, 4);
  }
  for (const h of host) {
    if (h.kind === 'staging') {
      push({ t: h.from, kind: 'stage', op: h.op, lines: h.lines, text: h.text }, 4);
      push({ t: h.to, kind: 'return', op: h.op, lines: h.lines, text: tr(`The call for ${opLabel(ops[h.op])} returns.`, `Lời gọi cho ${opLabel(ops[h.op])} trả về.`) }, 2);
    } else push({ t: h.from, kind: h.kind === 'blocked' ? 'block' : 'wait', op: h.op, lines: h.lines, text: h.text }, 4);
  }
  ev.sort((x, y) => x.t - y.t || x.rank - y.rank || (x.op ?? -1) - (y.op ?? -1));
  const steps: Step[] = [];
  for (const { rank: _rank, ...e } of ev) {
    const last = steps[steps.length - 1];
    if (last && Math.abs(last.t - e.t) < EPS) last.events.push(e);
    else steps.push({ t: e.t, events: [e] });
  }
  return steps;
}

/** The step showing at time t: the last one at or before it (null before the first). */
export function stepAt(s: Schedule, t: number): number | null {
  let k: number | null = null;
  for (let i = 0; i < s.steps.length; i++) if (s.steps[i].t <= t + EPS) k = i;
  return k;
}

/** The operation a step is about, for highlighting: its last start, else its last event with one. */
export function stepOp(step: Step): number | null {
  const starts = step.events.filter((e) => e.kind === 'start' && e.op != null);
  if (starts.length) return starts[starts.length - 1].op;
  const any = step.events.filter((e) => e.op != null);
  return any.length ? any[any.length - 1].op : null;
}

// ---------------------------------------------------------------- closed forms
/** Per-chunk times of equal chunks, overhead included, for the stages present. */
export function perChunk(cfg: PipeConfig, m: number) {
  const f = (k: Kind) => (total(cfg, k) > 0 ? total(cfg, k) / m + cfg.overhead : 0);
  return { a: f('in'), b: f('kernel'), c: f('out') };
}

/**
 * The smallest distance, in chunks queued, between two chunks that share a stream
 * or a hardware queue (Infinity when no two do).
 */
export function sharingGap(cfg: PipeConfig, m: number): number {
  const lastIn = new Map<number, number>();
  let d = Infinity;
  for (let j = 0; j < m; j++) {
    const g = queueOf(cfg, j % cfg.streams);
    const p = lastIn.get(g);
    if (p != null) d = Math.min(d, j - p);
    lastIn.set(g, j);
  }
  return d;
}

function e1Conditions(s: Schedule): Condition[] {
  const { cfg } = s;
  const m = s.shares.length;
  const p = perChunk(cfg, m);
  const sum = p.a + p.b + p.c;
  const M = Math.max(p.a, p.b, p.c);
  const d = sharingGap(cfg, m);
  const equal = cfg.items == null || (cfg.items >= cfg.n && cfg.items % cfg.n === 0);
  const enough = cfg.order === 'depth' ? d * M >= sum - EPS : d === Infinity;
  const fmt = (x: number) => fmtT(x);
  return [
    {
      id: 'engines',
      text: cfg.engines === 2
        ? tr('a copy engine for each direction', 'mỗi chiều có một copy engine')
        : cfg.a === 0 || cfg.c === 0
          ? tr('only one direction is copied', 'chỉ sao chép một chiều')
          : tr('one copy engine for both directions', 'một copy engine cho cả hai chiều'),
      holds: cfg.engines === 2 || cfg.a === 0 || cfg.c === 0,
    },
    { id: 'pinned', text: cfg.host === 'pinned' ? tr('pinned host memory', 'bộ nhớ host ghim (pinned)') : tr('pageable host memory', 'bộ nhớ host phân trang (pageable)'), holds: cfg.host === 'pinned' },
    { id: 'equal', text: equal ? tr('equal chunks', 'các chunk bằng nhau') : tr('unequal chunks', 'các chunk không bằng nhau'), holds: equal },
    {
      id: 'streams',
      text: cfg.order === 'depth'
        ? d === Infinity
          ? tr('no two chunks share a stream or queue', 'không có hai chunk nào chung một stream hay hàng đợi')
          : tr(
              `chunks sharing a stream or queue are ${d} apart, and ${d} × ${fmt(M)} ${enough ? '≥' : '<'} ${fmt(sum)}`,
              `các chunk chung một stream hay hàng đợi cách nhau ${d}, và ${d} × ${fmt(M)} ${enough ? '≥' : '<'} ${fmt(sum)}`,
            )
        : d === Infinity
          ? tr('each stream and queue holds one chunk', 'mỗi stream và hàng đợi chứa một chunk')
          : tr('issued breadth first with chunks sharing a stream or queue', 'phát lệnh theo chiều rộng trước, với các chunk chung một stream hay hàng đợi'),
      holds: enough,
    },
  ];
}

function e1Form(s: Schedule): ClosedForm | null {
  const m = s.shares.length;
  if (!s.ops.length) return null;
  const p = perChunk(s.cfg, m);
  const sum = p.a + p.b + p.c;
  const M = Math.max(p.a, p.b, p.c);
  const value = r9(sum + (m - 1) * M);
  const conditions = e1Conditions(s);
  return {
    name: tr('Pipeline with enough streams (E1)', 'Pipeline với đủ stream (E1)'),
    formula: `a + b + c + (n − 1)·max(a, b, c) = ${fmtT(sum)} + ${m - 1} × ${fmtT(M)}`,
    value, conditions, holds: conditions.every((c) => c.holds), matches: Math.abs(value - s.total) < 1e-6,
  };
}

function nvidiaForm(s: Schedule): ClosedForm | null {
  const { cfg } = s;
  if (!(cfg.a > 0 && cfg.b > 0)) return null;
  const m = s.shares.length;
  const tT = cfg.a;
  const tE = cfg.b;
  const value = r9(tE >= tT ? tE + tT / m : tT + tE / m);
  const conditions = [
    ...e1Conditions(s),
    {
      id: 'oneway',
      text: cfg.c === 0 ? tr('nothing is copied back, as in NVIDIA\'s example', 'không sao chép gì ngược về, như trong ví dụ của NVIDIA') : tr('the result is copied back too', 'kết quả cũng được sao chép ngược về'),
      holds: cfg.c === 0,
    },
    { id: 'overhead', text: cfg.overhead === 0 ? tr('no per-operation cost', 'không có chi phí cho mỗi thao tác') : tr('a per-operation cost', 'có chi phí cho mỗi thao tác'), holds: cfg.overhead === 0 },
  ];
  return {
    name: tr("NVIDIA's rough estimate (A7)", 'Ước lượng thô của NVIDIA (A7)'),
    formula: tE >= tT ? `tE + tT/nStreams = ${fmtT(tE)} + ${fmtT(tT)}/${m}` : `tT + tE/nStreams = ${fmtT(tT)} + ${fmtT(tE)}/${m}`,
    value, conditions, holds: conditions.every((c) => c.holds), matches: Math.abs(value - s.total) < 1e-6,
  };
}

// ---------------------------------------------------------------- sweeps
/** Total time for each chunk count, the other settings fixed. */
export function sweep(cfg: PipeConfig, ns: number[]): { n: number; total: number }[] {
  return ns.map((n) => ({ n, total: simulate({ ...cfg, n }).total }));
}

/** The chunk count from 1 to 64 with the smallest total (the smallest such n on a tie, to 1e-6). */
export function bestChunks(cfg: PipeConfig): { n: number; total: number } {
  let best = { n: 1, total: Infinity };
  for (let n = 1; n <= MAX_CHUNKS; n++) {
    const t = simulate({ ...cfg, n }).total;
    if (t < best.total - 1e-6) best = { n, total: t };
  }
  return best;
}

// ---------------------------------------------------------------- presets
export interface Preset {
  id: string;
  label: string;
  cfg: PipeConfig;
}
const P = (o: Partial<PipeConfig>): PipeConfig => ({ ...DEFAULTS, ...o });
export const PRESETS: Preset[] = [
  { id: 'best-practices', get label() { return tr('Best Practices example', 'Ví dụ trong Best Practices'); }, cfg: P({ a: 8, b: 16, c: 0, n: 4, streams: 4 }) },
  { id: 'equal', get label() { return tr('equal stages', 'các giai đoạn bằng nhau'); }, cfg: P({ a: 12, b: 12, c: 12, n: 6, streams: 6 }) },
  { id: 'copy-bound', get label() { return tr('copy-bound', 'bị giới hạn bởi sao chép'); }, cfg: P({ a: 16, b: 4, c: 8, n: 8, streams: 8 }) },
  { id: 'compute-bound', get label() { return tr('compute-bound', 'bị giới hạn bởi tính toán'); }, cfg: P({ a: 4, b: 24, c: 4, n: 8, streams: 8 }) },
  { id: 'p-pipeline', get label() { return tr("the prediction's case", 'trường hợp của phần dự đoán'); }, cfg: P({ a: 8, b: 16, c: 8, n: 8, streams: 8 }) },
];
/** Where the stepper starts: not the prediction's case, so the read-outs do not give it away. */
export const START: PipeConfig = P({ a: 12, b: 16, c: 8, n: 4, streams: 4 });
/** The prediction before the one-engine switch: 4 chunks of 1, 2, 1, one stream each. */
export const ONE_ENGINE: PipeConfig = P({ a: 4, b: 8, c: 4, n: 4, streams: 4, engines: 1, order: 'depth' });
/** The scene's four-chunk case, and q-bottleneck's. */
export const FOUR_CHUNKS: PipeConfig = P({ a: 4, b: 8, c: 4, n: 4, streams: 4 });
export const Q_BOTTLENECK: PipeConfig = P({ a: 8, b: 12, c: 4, n: 4, streams: 4 });

// ---------------------------------------------------------------- the reader's device
export interface LabParts {
  a: number;
  b: number;
  c: number;
  /** Seconds per unit: the largest part is scaled to 24 units. */
  unit: number;
  seconds: { a: number; b: number; c: number };
  /** CUDA only: asyncEngineCount when it is 1 or 2 (A7). ROCm's is a constant 8 and says nothing (B3). */
  engines?: 1 | 2;
}
export interface LabStaged {
  /** Streams (CUDA, HIP) or buffers (Metal). */
  streams: number;
  chunks: number;
  order: Order;
  host: HostMem;
  seconds: number;
  /** The same time in the preset's units. */
  units: number;
}
const numOf = (r: Record<string, unknown>, ...keys: string[]) => {
  for (const k of keys) {
    const v = r[k];
    if (typeof v === 'number' && isFinite(v)) return v;
  }
  return undefined;
};

/**
 * Lab 23.1's parts row (h2d_s, kernel_s, d2h_s; on Metal fill_s, kernel_s, drain_s)
 * as totals for the stepper, the largest part scaled to 24 units. `backend` is the
 * report's backend: only on 'cuda' does async_engines count copy engines.
 */
export function fromReports(row: Record<string, unknown>, view: View, backend = ''): LabParts | null {
  const sa = view === 'metal' ? numOf(row, 'fill_s') : numOf(row, 'h2d_s');
  const sb = numOf(row, 'kernel_s');
  const sc = view === 'metal' ? numOf(row, 'drain_s') : numOf(row, 'd2h_s');
  if (sa == null || sb == null || sc == null) return null;
  const big = Math.max(sa, sb, sc);
  if (!(big > 0)) return null;
  const unit = big / 24;
  const out: LabParts = { a: sa / unit, b: sb / unit, c: sc / unit, unit, seconds: { a: sa, b: sb, c: sc } };
  const eng = numOf(row, 'async_engines');
  if (view === 'cuda' && backend === 'cuda' && (eng === 1 || eng === 2)) out.engines = eng;
  return out;
}

/** The lab's rows: its parts, and every staged (CUDA family) or in-flight (Metal) measurement. */
export function parseLabRows(rows: Record<string, unknown>[], view: View, backend = ''): { parts: LabParts | null; staged: LabStaged[] } {
  const partsRow = rows.find((r) => r.name === 'parts') ?? rows.find((r) => fromReports(r, view, backend) != null);
  const parts = partsRow ? fromReports(partsRow, view, backend) : null;
  const staged: LabStaged[] = [];
  if (parts)
    for (const r of rows) {
      if (!(r.name === 'staged' || r.name === 'inflight')) continue;
      const streams = numOf(r, 'streams', 'buffers');
      const chunks = numOf(r, 'chunks');
      const seconds = numOf(r, 'seconds');
      if (streams == null || chunks == null || seconds == null) continue;
      const order: Order = String(r.order ?? 'depth') === 'breadth' ? 'breadth' : 'depth';
      const host: HostMem = String(r.host ?? 'pinned') === 'pageable' ? 'pageable' : 'pinned';
      staged.push({ streams, chunks, order, host, seconds, units: seconds / parts.unit });
    }
  return { parts, staged };
}

// ---------------------------------------------------------------- the pinned scene
/**
 * A pageable copy staged through pinned memory (pp. 562-563): the CPU copies each
 * piece into a staging buffer and the copy engine moves it on. With `buffers`
 * staging buffers, the CPU may fill one while the engine empties another. The call
 * returns when the last piece has been moved (the worst case, A6). Unitless.
 */
export function stagingPlan(bytes: number, stagingSize: number, opts: { hostRate?: number; dmaRate?: number; buffers?: number } = {}) {
  const hostRate = opts.hostRate ?? 1;
  const dmaRate = opts.dmaRate ?? 1;
  const buffers = Math.max(1, opts.buffers ?? 2);
  const pieces: { k: number; size: number; buffer: number; hostFrom: number; hostTo: number; dmaFrom: number; dmaTo: number }[] = [];
  const count = bytes > 0 && stagingSize > 0 ? Math.ceil(bytes / stagingSize) : 0;
  let hostT = 0;
  for (let k = 0; k < count; k++) {
    const size = Math.min(stagingSize, bytes - k * stagingSize);
    const freeAt = k >= buffers ? pieces[k - buffers].dmaTo : 0;
    const hostFrom = Math.max(hostT, freeAt);
    const hostTo = hostFrom + size * hostRate;
    const dmaFrom = Math.max(hostTo, k > 0 ? pieces[k - 1].dmaTo : 0);
    const dmaTo = dmaFrom + size * dmaRate;
    pieces.push({ k, size, buffer: k % buffers, hostFrom: r9(hostFrom), hostTo: r9(hostTo), dmaFrom: r9(dmaFrom), dmaTo: r9(dmaTo) });
    hostT = hostTo;
  }
  const returnsAt = count ? pieces[count - 1].dmaTo : 0;
  return {
    pieces,
    /** The call returns when the last piece has been moved; the host is held until then. */
    returnsAt,
    /** Of that, time the host spent copying (the rest it waited for a staging buffer or for the engine). */
    hostCopying: r9(pieces.reduce((s, p) => s + p.hostTo - p.hostFrom, 0)),
    /** The same bytes moved straight from pinned memory by the engine. */
    pinnedTime: r9(bytes * dmaRate),
  };
}

/** What the scene "Why an asynchronous copy needs pinned memory" draws. */
export function pinnedScene() {
  const pages = 8;
  return {
    pages,
    /** The physical frame each virtual page of the malloc buffer sits in, before the OS moves one. */
    frames: [3, 9, 0, 6, 11, 4, 8, 1],
    frameCount: 12,
    /** The page the OS moves in part 1, and the frame it moves to. */
    moved: { page: 2, to: 10 },
    staging: 2,
    plan: stagingPlan(pages, 2, { hostRate: 0.5, dmaRate: 1, buffers: 2 }),
    pageable: simulate({ ...FOUR_CHUNKS, host: 'pageable' }),
    pinned: simulate(FOUR_CHUNKS),
  };
}

// ---------------------------------------------------------------- Fig. 23.13
export interface HaloBar {
  lane: 'gpu' | 'net';
  label: string;
  from: number;
  to: number;
}
/**
 * One stencil iteration on one GPU of the multi-GPU job (pp. 553-555, Fig. 23.13).
 * Without overlap: compute every point, then exchange the top and bottom halos, then
 * reduce the norm. With overlap: the boundary rows are computed first, beside the
 * internal rows as the figure draws them, each exchange starts as soon as it can
 * (the bottom one after the top one: MPI_Sendrecv blocks, p. 562), and the reduction
 * waits for everything (p. 555).
 */
export function haloLanes(t: { internal: number; boundary: number; exchange: number; reduce: number }, overlap: boolean) {
  const bars: HaloBar[] = [];
  if (!overlap) {
    bars.push({ lane: 'gpu', label: 'compute all points', from: 0, to: t.internal }); // i18n-ignore: identifier also shown to the reader
    bars.push({ lane: 'net', label: 'halo exchange (top)', from: t.internal, to: t.internal + t.exchange }); // i18n-ignore: identifier also shown to the reader
    bars.push({ lane: 'net', label: 'halo exchange (bottom)', from: t.internal + t.exchange, to: t.internal + 2 * t.exchange }); // i18n-ignore: identifier also shown to the reader
  } else {
    bars.push({ lane: 'gpu', label: 'compute internal', from: 0, to: t.internal }); // i18n-ignore: identifier also shown to the reader
    bars.push({ lane: 'gpu', label: 'compute top', from: 0, to: t.boundary }); // i18n-ignore: identifier also shown to the reader
    bars.push({ lane: 'gpu', label: 'compute bottom', from: 0, to: t.boundary }); // i18n-ignore: identifier also shown to the reader
    bars.push({ lane: 'net', label: 'halo exchange (top)', from: t.boundary, to: t.boundary + t.exchange }); // i18n-ignore: identifier also shown to the reader
    bars.push({ lane: 'net', label: 'halo exchange (bottom)', from: t.boundary + t.exchange, to: t.boundary + 2 * t.exchange }); // i18n-ignore: identifier also shown to the reader
  }
  const ready = Math.max(...bars.map((b) => b.to));
  bars.push({ lane: 'net', label: 'reduce L2 norm', from: ready, to: ready + t.reduce }); // i18n-ignore: identifier also shown to the reader
  const total = ready + t.reduce;
  const exchangeEnd = overlap ? t.boundary + 2 * t.exchange : t.internal + 2 * t.exchange;
  return {
    bars,
    total: r9(total),
    /** The exchange is hidden when it ends before the internal rows do (p. 554). */
    hidden: overlap && exchangeEnd <= t.internal + EPS,
    /** How far the exchange runs past the internal compute. */
    exposed: r9(Math.max(0, exchangeEnd - t.internal)),
  };
}
export const HALO_BOOK = { internal: 10, boundary: 2.5, exchange: 3, reduce: 2 };
