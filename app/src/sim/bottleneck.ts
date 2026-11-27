// Chapter 6, section 6.9: a toy bottleneck model for the "bottleneck loop" explorer,
// and the roofline arithmetic for the "stopping rule" panel that places the reader's
// measured kernels against the roof lab 5.1 measured.
//
// The rule the explorer illustrates is the book's (p. 153): an optimisation spends
// more of one resource to relieve another; relieving a resource that is not the
// bottleneck gains nothing, and spending the bottleneck resource makes things worse.
// Tiling is the book's example: it helps when global bandwidth limits the kernel and
// hurts when occupancy is already limited by shared memory.
//
// The model is ours, illustrative, and not a predictor:
//  - run time is the largest of four resource times (global memory, shared memory,
//    arithmetic and instructions, barriers); real resources overlap imperfectly;
//  - global memory moves at full rate only with enough bytes in flight, which grows
//    with occupancy, with the width of each load and with independent loads per warp
//    (6.2, 6.3, 6.6). Below that it is latency-bound, linearly. The constant NEED is
//    made up;
//  - occupancy counts whole blocks against three budgets of a made-up SM (block
//    slots, shared memory, registers), the arithmetic of 4.7 and 5.6;
//  - arithmetic hides pipeline latency only above a made-up occupancy; barrier time
//    does not depend on occupancy; shared memory has no latency, only passes;
//  - a grid that does not fill the device, or ends in a partial wave, stretches every
//    resource's time by ceil(waves) / waves (the tail effect, 4.2 and p. 143);
//  - each optimisation multiplies a few numbers by a fixed factor in the direction
//    the chapter gives. The factors are invented; only their directions are the book's.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

export type Resource = 'global' | 'shared' | 'compute' | 'barrier';
export const RESOURCES: readonly Resource[] = ['global', 'shared', 'compute', 'barrier'];
export const RESOURCE_NAME: Record<Resource, string> = {
  get global() { return tr('Global memory', 'Bộ nhớ toàn cục'); },
  get shared() { return tr('Shared memory', 'Bộ nhớ chia sẻ'); },
  get compute() { return tr('Arithmetic and instructions', 'Số học và lệnh'); },
  get barrier() { return tr('Waiting at barriers', 'Chờ ở rào chắn'); },
};

/** The made-up SM every toy kernel runs on. */
export const DEVICE = {
  sms: 4,
  maxBlocksPerSM: 8,
  /** Shared memory per SM, in arbitrary units. */
  sharedPerSM: 100,
  /** Registers per SM, in arbitrary units. */
  regsPerSM: 100,
  /** Bytes in flight needed to saturate the bus: occupancy x load width x independent loads (made up). */
  need: 0.5,
  /** Occupancy needed to hide pipeline latency (made up). */
  needCompute: 0.25,
};

export interface ToyKernel {
  /** Global traffic for data the kernel must touch at least once. */
  uniqueBytes: number;
  /** Global traffic spent re-reading data that other threads or blocks also load. Tiling and coarsening cut it. */
  reuseBytes: number;
  /** Shared-memory requests, in time units at one pass per request. */
  sharedWork: number;
  /** Average passes per shared-memory request: 1 means no bank conflicts. */
  conflict: number;
  /** Useful arithmetic. */
  flops: number;
  /** Instruction cost of the load and store instructions at 4 bytes each (divided by the load width). */
  loadInstr: number;
  /** Loop branches and counter updates (unrolling removes them). */
  branchWork: number;
  /** Index arithmetic. */
  indexWork: number;
  /** Time spent waiting at barriers. */
  barrierWork: number;
  /** Shared memory per block, in units of DEVICE.sharedPerSM. */
  sharedPerBlock: number;
  /** Registers per block, in units of DEVICE.regsPerSM. */
  regsPerBlock: number;
  /** Floats per load instruction: 1 or 4. */
  vec: number;
  /** Independent loads in flight per warp: 1, or 2 after unrolling. */
  ilp: number;
  /** Blocks in the grid. */
  gridBlocks: number;
  /** How many times each optimisation has been applied. */
  applied: Partial<Record<OpId, number>>;
}

export type Limiter = 'blocks' | 'shared' | 'regs';

export interface Evaluation {
  times: Record<Resource, number>;
  total: number;
  bottleneck: Resource;
  blocksPerSM: number;
  occupancy: number;
  limiter: Limiter;
  /** Fraction of the bus the kernel can keep busy. */
  memRate: number;
  waves: number;
  /** ceil(waves) / waves: 1 when the grid fills whole waves. */
  tail: number;
}

const EPS = 1e-9;
const fl = (x: number) => Math.floor(x + EPS);

export function evaluate(k: ToyKernel): Evaluation {
  const bySlots = DEVICE.maxBlocksPerSM;
  const byShared = k.sharedPerBlock > 0 ? fl(DEVICE.sharedPerSM / k.sharedPerBlock) : Infinity;
  const byRegs = k.regsPerBlock > 0 ? fl(DEVICE.regsPerSM / k.regsPerBlock) : Infinity;
  const blocksPerSM = Math.max(0, Math.min(bySlots, byShared, byRegs));
  // Report the budget that binds; slots win a tie (nothing to fix there).
  const limiter: Limiter = blocksPerSM === bySlots ? 'blocks' : blocksPerSM === byShared ? 'shared' : 'regs';
  const occupancy = blocksPerSM / DEVICE.maxBlocksPerSM;
  const resident = blocksPerSM * DEVICE.sms;
  const waves = resident > 0 ? k.gridBlocks / resident : Infinity;
  const tail = waves > 0 && isFinite(waves) ? Math.ceil(waves - EPS) / waves : Infinity;
  const memRate = Math.min(1, (occupancy * k.vec * k.ilp) / DEVICE.need);
  const hide = Math.min(1, occupancy / DEVICE.needCompute);
  const raw: Record<Resource, number> = {
    global: memRate > 0 ? (k.uniqueBytes + k.reuseBytes) / memRate : Infinity,
    shared: k.sharedWork * k.conflict,
    compute: hide > 0 ? (k.flops + k.loadInstr / k.vec + k.branchWork + k.indexWork) / hide : Infinity,
    barrier: k.barrierWork,
  };
  const times = Object.fromEntries(RESOURCES.map((r) => [r, raw[r] * tail])) as Record<Resource, number>;
  let bottleneck: Resource = 'global';
  for (const r of RESOURCES) if (times[r] > times[bottleneck] + EPS) bottleneck = r;
  return { times, total: times[bottleneck], bottleneck, blocksPerSM, occupancy, limiter, memRate, waves, tail };
}

// ---------------------------------------------------------------- optimisations
export type OpId = 'tile' | 'pad' | 'vector' | 'double-buffer' | 'coarsen' | 'unroll';

export interface Op {
  id: OpId;
  label: string;
  /** What it does in this model, and the page that gives the direction. */
  effect: string;
  page: string;
  /** How many times it may be applied. */
  max: number;
}

export const OPS: readonly Op[] = [
  {
    id: 'tile',
    get label() { return tr('Tiling', 'Phân ô (tiling)'); },
    get effect() { return tr('re-read global traffic halved; shared memory per block doubled (a first tile adds shared traffic and barriers)', 'lưu lượng toàn cục đọc lại giảm một nửa; bộ nhớ chia sẻ mỗi khối tăng gấp đôi (ô đầu tiên thêm lưu lượng chia sẻ và rào chắn)'); },
    page: 'p. 153',
    max: 3,
  },
  {
    id: 'pad',
    get label() { return tr('Padding', 'Đệm thêm (padding)'); },
    get effect() { return tr('bank conflicts gone; shared memory per block up by 1/32; a little more index arithmetic', 'hết xung đột bank; bộ nhớ chia sẻ mỗi khối tăng 1/32; thêm một chút số học chỉ số'); },
    page: 'pp. 140–141',
    max: 1,
  },
  {
    id: 'vector',
    get label() { return tr('Vector loads', 'Load vector'); },
    get effect() { return tr('four floats per load instruction, so four times the bytes in flight and a quarter of the load instructions; registers up by a quarter', 'bốn float mỗi lệnh load, nên số byte đang truyền gấp bốn và số lệnh load còn một phần tư; thanh ghi tăng một phần tư'); },
    page: 'p. 139; facts A11', // i18n-ignore: citation
    max: 1,
  },
  {
    id: 'double-buffer',
    get label() { return tr('Double buffering', 'Double buffering (hai buffer)'); },
    get effect() { return tr('barrier waiting halved (one barrier instead of two); shared memory per block doubled', 'thời gian chờ rào chắn giảm một nửa (một rào chắn thay vì hai); bộ nhớ chia sẻ mỗi khối tăng gấp đôi'); },
    page: 'p. 147',
    max: 1,
  },
  {
    id: 'coarsen',
    get label() { return tr('Coarsening ×2', 'Gộp việc cho luồng (coarsening) ×2'); },
    get effect() { return tr('re-read global traffic cut by a quarter; index arithmetic cut by a quarter; registers up by a quarter; half as many blocks', 'lưu lượng toàn cục đọc lại giảm một phần tư; số học chỉ số giảm một phần tư; thanh ghi tăng một phần tư; số khối còn một nửa'); },
    page: 'pp. 142–143',
    max: 3,
  },
  {
    id: 'unroll',
    get label() { return tr('Unrolling', 'Trải vòng lặp (unrolling)'); },
    get effect() { return tr('loop branches gone; two independent loads in flight per warp', 'hết các nhánh vòng lặp; hai lần load độc lập đang truyền mỗi warp'); },
    page: 'pp. 144–145',
    max: 1,
  },
];
export const opById = (id: OpId): Op => OPS.find((o) => o.id === id)!;

/** The shared memory, shared traffic and barriers a kernel gets from its first tile. */
export const FIRST_TILE = { sharedPerBlock: 10, sharedWork: 20, barrierWork: 8 };

export function canApply(k: ToyKernel, op: OpId): { ok: boolean; why?: string } {
  const n = k.applied[op] ?? 0;
  const o = opById(op);
  if (n >= o.max) return { ok: false, why: o.max === 1 ? tr('already applied', 'đã áp dụng rồi') : tr(`applied ${n} times already (the limit here)`, `đã áp dụng ${n} lần (giới hạn ở đây)`) };
  if (op === 'pad' && k.sharedPerBlock === 0) return { ok: false, why: tr('no shared memory to pad', 'không có bộ nhớ chia sẻ để đệm') };
  if (op === 'double-buffer' && k.barrierWork === 0) return { ok: false, why: tr('no barriers to remove', 'không có rào chắn để bỏ') };
  return { ok: true };
}

export function applyOp(k: ToyKernel, op: OpId): ToyKernel {
  if (!canApply(k, op).ok) return k;
  const n: ToyKernel = { ...k, applied: { ...k.applied, [op]: (k.applied[op] ?? 0) + 1 } };
  switch (op) {
    case 'tile':
      n.reuseBytes = k.reuseBytes / 2;
      if (k.sharedPerBlock === 0) {
        n.sharedPerBlock = FIRST_TILE.sharedPerBlock;
        n.sharedWork = k.sharedWork + FIRST_TILE.sharedWork;
        n.barrierWork = k.barrierWork + FIRST_TILE.barrierWork;
      } else n.sharedPerBlock = k.sharedPerBlock * 2;
      break;
    case 'pad':
      n.conflict = 1;
      n.sharedPerBlock = (k.sharedPerBlock * 33) / 32;
      n.indexWork = k.indexWork + 1;
      break;
    case 'vector':
      n.vec = 4;
      n.regsPerBlock = k.regsPerBlock * 1.25;
      break;
    case 'double-buffer':
      n.barrierWork = k.barrierWork / 2;
      n.sharedPerBlock = k.sharedPerBlock * 2;
      break;
    case 'coarsen':
      n.reuseBytes = k.reuseBytes * 0.75;
      n.indexWork = k.indexWork * 0.75;
      n.regsPerBlock = k.regsPerBlock * 1.25;
      n.gridBlocks = k.gridBlocks / 2;
      break;
    case 'unroll':
      n.branchWork = 0;
      n.ilp = 2;
      break;
  }
  return n;
}

export type StepVerdict = 'faster' | 'same' | 'slower';
export function compare(before: number, after: number): StepVerdict {
  const rel = (after - before) / before;
  return rel < -1e-6 ? 'faster' : rel > 1e-6 ? 'slower' : 'same';
}

export interface Step {
  op: OpId;
  before: Evaluation;
  after: Evaluation;
  verdict: StepVerdict;
  /** True when the step lowered a resource time other than the bottleneck's and left the total unchanged. */
  relievedNonBottleneck: boolean;
}

/** Apply a sequence of optimisations from a starting kernel; one Step per applicable op. */
export function run(start: ToyKernel, ops: OpId[]): { kernel: ToyKernel; steps: Step[] } {
  let k = start;
  const steps: Step[] = [];
  for (const op of ops) {
    if (!canApply(k, op).ok) continue;
    const before = evaluate(k);
    k = applyOp(k, op);
    const after = evaluate(k);
    const verdict = compare(before.total, after.total);
    const lowered = RESOURCES.some((r) => r !== before.bottleneck && after.times[r] < before.times[r] - EPS);
    steps.push({ op, before, after, verdict, relievedNonBottleneck: verdict === 'same' && lowered });
  }
  return { kernel: k, steps };
}

// ---------------------------------------------------------------- starting kernels
const base = (p: Omit<ToyKernel, 'applied' | 'vec' | 'ilp' | 'conflict'> & Partial<ToyKernel>): ToyKernel => ({
  vec: 1,
  ilp: 1,
  conflict: 1,
  applied: {},
  ...p,
});

export interface Preset {
  id: string;
  label: string;
  about: string;
  kernel: ToyKernel;
}

export const PRESETS: readonly Preset[] = [
  {
    id: 'naive',
    get label() { return tr('Naive matmul', 'Matmul ngây thơ'); },
    get about() { return tr('One thread per output, no shared memory, full occupancy. Most of its global traffic re-reads inputs other threads also load.', 'Một luồng cho mỗi đầu ra, không dùng bộ nhớ chia sẻ, occupancy đầy đủ. Phần lớn lưu lượng toàn cục của nó là đọc lại các đầu vào mà luồng khác cũng load.'); },
    kernel: base({
      uniqueBytes: 10, reuseBytes: 70, sharedWork: 0, flops: 24, loadInstr: 12, branchWork: 6, indexWork: 6,
      barrierWork: 0, sharedPerBlock: 0, regsPerBlock: 10, gridBlocks: 256,
    }),
  },
  {
    id: 'shared-limited',
    get label() { return tr('Tiled, limited by shared memory', 'Phân ô, bị giới hạn bởi bộ nhớ chia sẻ'); },
    get about() { return tr('A tiled kernel whose shared memory allows only 3 of 8 block slots. Too few requests in flight: global memory is latency-bound.', 'Một kernel phân ô có bộ nhớ chia sẻ chỉ cho phép 3 trên 8 chỗ khối. Quá ít yêu cầu đang truyền: bộ nhớ toàn cục bị giới hạn bởi độ trễ.'); },
    kernel: base({
      uniqueBytes: 10, reuseBytes: 20, sharedWork: 24, flops: 18, loadInstr: 6, branchWork: 4, indexWork: 2,
      barrierWork: 10, sharedPerBlock: 30, regsPerBlock: 10, gridBlocks: 96,
    }),
  },
  {
    id: 'unpadded',
    get label() { return tr('Corner-turned, unpadded', 'Corner-turned, không đệm'); },
    get about() { return tr('A tiled kernel whose corner-turned store conflicts on the banks: on average 4 passes per shared request.', 'Một kernel phân ô có lần store corner-turned bị xung đột bank: trung bình 4 lượt cho mỗi yêu cầu chia sẻ.'); },
    kernel: base({
      uniqueBytes: 10, reuseBytes: 15, sharedWork: 20, conflict: 4, flops: 18, loadInstr: 6, branchWork: 4,
      indexWork: 2, barrierWork: 10, sharedPerBlock: 16, regsPerBlock: 10, gridBlocks: 192,
    }),
  },
  {
    id: 'vecadd',
    get label() { return tr('Vector addition', 'Cộng vector'); },
    get about() { return tr('One add per 12 bytes, nothing reused, full occupancy, bandwidth-bound. Its grid fills four waves.', 'Một phép cộng cho mỗi 12 byte, không tái sử dụng gì, occupancy đầy đủ, bị giới hạn bởi băng thông. Lưới của nó lấp đầy bốn đợt (wave).'); },
    kernel: base({
      uniqueBytes: 60, reuseBytes: 0, sharedWork: 0, flops: 6, loadInstr: 12, branchWork: 0, indexWork: 2,
      barrierWork: 0, sharedPerBlock: 0, regsPerBlock: 8, gridBlocks: 128,
    }),
  },
  {
    id: 'exchange',
    get label() { return tr('Neighbour exchange, two barriers', 'Trao đổi với hàng xóm, hai rào chắn'); },
    get about() { return tr('Section 6.7’s loop: read a neighbour’s value from a shared buffer, write your own, two barriers per iteration.', 'Vòng lặp của mục 6.7: đọc giá trị của hàng xóm từ một buffer chia sẻ, ghi giá trị của mình, hai rào chắn mỗi vòng lặp.'); },
    kernel: base({
      uniqueBytes: 20, reuseBytes: 10, sharedWork: 25, flops: 10, loadInstr: 4, branchWork: 4, indexWork: 2,
      barrierWork: 50, sharedPerBlock: 10, regsPerBlock: 10, gridBlocks: 128,
    }),
  },
];
export const presetById = (id: string): Preset => PRESETS.find((p) => p.id === id) ?? PRESETS[0];

// ---------------------------------------------------------------- roofline arithmetic (section 6.9, p. 154)
export interface RoofPoint {
  /** FLOP per byte, from the counts given. */
  intensity: number;
  /** The roof at that intensity, GFLOP/s: min(peak, bandwidth x intensity). */
  roof: number;
  /** What the kernel achieved, GFLOP/s. */
  achieved: number;
  /** achieved / roof. */
  fraction: number;
  /** Which part of the roof applies. */
  regime: 'memory' | 'compute';
}

export function roofFraction(flop: number, bytes: number, seconds: number, peakGflops: number, bwGBs: number): RoofPoint {
  const intensity = flop / bytes;
  const memRoof = bwGBs * intensity;
  const roof = Math.min(peakGflops, memRoof);
  const achieved = flop / seconds / 1e9;
  return { intensity, roof, achieved, fraction: achieved / roof, regime: memRoof < peakGflops ? 'memory' : 'compute' };
}

export type StopVerdict = 'stop' | 'room' | 'above';
/** Above 110 % of the roof the counts, not the kernel, are what to question. */
export const ABOVE = 1.1;
export function stopVerdict(fraction: number, threshold: number): StopVerdict {
  if (fraction > ABOVE) return 'above';
  return fraction >= threshold ? 'stop' : 'room';
}
/** How much faster the kernel could get at most, if it reached the roof. */
export const headroom = (fraction: number) => (fraction > 0 ? Math.max(1, 1 / fraction) : Infinity);

/**
 * Global-memory bytes a tiled GEMM loads, by the book's counting (chapter 5): every
 * element a block uses is loaded once per block, stores and caches ignored. P (B x L)
 * = M (B x K) times an operand of K x L, tile width T. M is loaded once per column of
 * output tiles, the second operand once per row of output tiles. For square W x W with
 * T dividing W this is 8 W^3 / T bytes, intensity T / 4 FLOP/B. T = 1 is the naive
 * kernel's 0.25 FLOP/B.
 */
export function tiledGemmBytes(B: number, K: number, L: number, T: number): number {
  return 4 * (B * K * Math.ceil(L / T) + K * L * Math.ceil(B / T));
}

// ---------------------------------------------------------------- placing lab rows
export interface LabPoint {
  /** Lab number as the reader types it, e.g. "5.2". */
  lab: string;
  label: string;
  kind: 'stream' | 'gemm';
  flop: number;
  bytes: number;
  seconds: number;
  /** Batch size of a linear-layer row, when it has one. */
  batch?: number;
  tile?: number;
}

const pos = (v: unknown): number | undefined => (typeof v === 'number' && isFinite(v) && v > 0 ? v : undefined);
/** "ch06/lab2_vector" -> "6.2". */
export function labNum(lab: string): string {
  const m = /^ch0*(\d+)\/lab(\d+)/.exec(lab);
  return m ? `${m[1]}.${m[2]}` : lab;
}
/** Row fields that name a kernel variant. "solution" and "work" are the build variant the runner stamps on rows, not a name. */
const TAGS = ['kernel', 'variant', 'mode', 'config', 'version', 'layout'];
const NOT_TAGS = new Set(['solution', 'work']);

/**
 * Turn one benchmark row into a point with model counts, or null when the row does not
 * carry what the counts need. Two shapes are recognised:
 *  - a streaming add (a row of a vector lab, with an element count): 1 FLOP and 12
 *    bytes per element, the book's 1/12 FLOP/B (chapter 5, p. 96);
 *  - a tiled GEMM: a square one (width, tile) or a linear layer (batch, k or in, l or
 *    out, tile), counted with tiledGemmBytes.
 * Seconds come from the row, or from its GFLOP/s or GB/s and the model's counts.
 */
export function placeRow(lab: string, row: Record<string, unknown>): LabPoint | null {
  const num = labNum(lab);
  const name = String(row.name ?? '');
  const tags = TAGS.map((t) => row[t]).filter((x): x is string => typeof x === 'string' && x !== '' && !NOT_TAGS.has(x));
  const tag = [...new Set(tags)].join(', ');
  const label = name + (tag ? ` (${tag})` : '');
  const isVec = /vec/i.test(lab + ' ' + name) && !/matmul|gemm|linear/i.test(lab + ' ' + name);
  const elems = pos(row.elements) ?? pos(row.n);
  if (isVec && elems) {
    const bytes = 12 * elems;
    const gbs = pos(row.gb_per_s);
    const seconds = pos(row.seconds) ?? (gbs ? bytes / (gbs * 1e9) : undefined);
    if (!seconds) return null;
    return { lab: num, label, kind: 'stream', flop: elems, bytes, seconds };
  }
  const T = pos(row.tile) ?? pos(row.tile_width);
  const W = pos(row.width);
  const B = pos(row.batch);
  const K = pos(row.k) ?? pos(row.in) ?? pos(row.in_features);
  const L = pos(row.l) ?? pos(row.out) ?? pos(row.out_features);
  const dims: [number, number, number] | null = B && K && L ? [B, K, L] : W ? [W, W, W] : null;
  if (!dims || !T) return null;
  const [b, k, l] = dims;
  const flop = 2 * b * k * l;
  const bytes = tiledGemmBytes(b, k, l, T);
  const gf = pos(row.gflops);
  const seconds = pos(row.seconds) ?? (gf ? flop / (gf * 1e9) : undefined);
  if (!seconds) return null;
  const size = B ? tr(`batch ${b}`, `batch ${b}`) : tr(`width ${b.toLocaleString('en-US')}`, `chiều rộng ${b.toLocaleString('en-US')}`);
  return { lab: num, label: tr(`${label}, ${size}, tile ${T}`, `${label}, ${size}, ô ${T}`), kind: 'gemm', flop, bytes, seconds, batch: B, tile: T };
}

/** The labs the stopping-rule panel reads, by id prefix, and which of their rows it keeps. */
export const STOP_LABS: readonly { prefix: string; keep: (p: LabPoint) => boolean; why: string }[] = [
  { prefix: 'ch05/lab2_', keep: (p) => p.tile === 16, get why() { return tr("chapter 5's tiled kernel at tile 16, this chapter's starting point", 'kernel phân ô của chương 5 với ô 16, điểm xuất phát của chương này'); } },
  { prefix: 'ch05/lab3_', keep: (p) => p.batch === 1 || p.batch === 256, get why() { return tr('the linear layer before this chapter, at batch 1 and 256', 'lớp tuyến tính trước chương này, ở batch 1 và 256'); } },
  { prefix: 'ch06/lab2_', keep: () => true, get why() { return tr('vector addition with 16-byte loads', 'cộng vector với các lần load 16 byte'); } },
  {
    prefix: 'ch06/lab4_',
    keep: (p) => p.tile === 16 && (p.batch === 1 || p.batch === 256),
    get why() { return tr('the linear layer in the framework layout, baseline and corner-turned, at batch 1 and 256, tile 16', 'lớp tuyến tính theo bố cục của framework, bản cơ sở và bản corner-turned, ở batch 1 và 256, ô 16'); },
  },
  { prefix: 'ch06/lab5_', keep: () => true, get why() { return tr('tiled matmul with two barriers and double-buffered', 'matmul phân ô với hai rào chắn và bản double-buffered'); } },
];

/** Every placeable row of the stopping-rule labs, from (lab id, row) pairs. */
export function stopPoints(rows: { lab: string; row: Record<string, unknown> }[]): LabPoint[] {
  const out: LabPoint[] = [];
  for (const { lab, row } of rows) {
    const spec = STOP_LABS.find((s) => lab.startsWith(s.prefix));
    if (!spec) continue;
    const p = placeRow(lab, row);
    if (p && spec.keep(p)) out.push(p);
  }
  return out;
}
