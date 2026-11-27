// One warp, lane by lane: the shuffles of section 10.7, the warp reduction of Fig. 10.13,
// and the block-level kernels of Figs. 10.10, 10.14 and 10.16 counted operation by
// operation. Sections 10.7 and 10.8 and the scene "Lanes exchanging values" all run on it.
//
// The shuffle rules are the documented ones, not a guess at hardware:
//   - CUDA guide 5.4.6 (notes/today/infra-facts.md A3; ch10-facts.md A3): a source past
//     the end of its subsection is not wrapped and the caller keeps its own value (down,
//     up); xor may read earlier subsections but not later ones; __shfl_sync reads lane
//     srcLane % width of the caller's own subsection; reading a lane that does not take
//     part is undefined.
//   - HIP (infra B3) documents the same for down, up and xor, and says nothing about an
//     out-of-range srcLane for __shfl. Metal (infra C2) has no mask and no width; its
//     simd_shuffle_down leaves the top lanes unchanged in the same way.
// An undefined value is null and stays null through every addition it enters.
//
// Nothing here models time, lockstep execution or divergence cost. Block-level functions
// work one iteration at a time (every thread of an iteration, then the next iteration).
//
// The one import is lib/locale.ts (itself import-free), so the browser checks can still import
// this file directly in Node.

import { tr } from '../lib/locale.ts';

export const BOOK_WARP = 32;

export const isPow2 = (n: number): boolean => Number.isInteger(n) && n >= 1 && (n & (n - 1)) === 0;
export const log2 = (n: number): number => Math.round(Math.log2(n));

const add = (a: number | null, b: number | null): number | null => (a == null || b == null ? null : a + b);
const range = (n: number) => Array.from({ length: n }, (_, i) => i);

// ---------------------------------------------------------------- one shuffle
/** down, up, xor: __shfl_down_sync, __shfl_up_sync, __shfl_xor_sync. idx: __shfl_sync (by source lane). */
export type Op = 'down' | 'up' | 'xor' | 'idx';
export const OPS: Op[] = ['down', 'up', 'xor', 'idx'];
export const OP_FN: Record<Op, string> = {
  down: '__shfl_down_sync',
  up: '__shfl_up_sync',
  xor: '__shfl_xor_sync',
  idx: '__shfl_sync',
};
export const OP_PARAM: Record<Op, string> = { down: 'delta', up: 'delta', xor: 'laneMask', idx: 'srcLane' };

export interface ShuffleOptions {
  /** The width argument: a power of two from 1 to the warp's width. Default: the warp's width. */
  width?: number;
  /** Lanes named in the mask. Default: every lane. */
  mask?: boolean[];
  /** Lanes that have returned before the call. Default: none. */
  exited?: boolean[];
}

/**
 * moved: received another lane's value. own: the source is out of range, so the lane
 * keeps its own value (a documented rule, not an accident). undefined: the source lane
 * takes no part in the call. absent: this lane itself does not call (exited, or not in
 * the mask).
 */
export type Outcome = 'moved' | 'own' | 'undefined' | 'absent';

export interface LaneResult {
  lane: number;
  /** Named in the mask and not exited. */
  calls: boolean;
  /** The lane the arithmetic points at (index in the warp), before the out-of-range rule; may lie outside the warp. */
  target: number | null;
  /** The lane actually read: target when in range, the lane itself when not, null when it does not call. */
  src: number | null;
  outcome: Outcome;
  /** What the call returns to this lane; null when undefined or absent. */
  value: number | null;
}

/** Where lane `lane` reads from, by the documented rule for each function. */
export function sourceOf(op: Op, lane: number, param: number, width: number): { target: number; inRange: boolean } {
  const base = lane - (lane % width);
  const li = lane % width;
  switch (op) {
    case 'down':
      return { target: base + li + param, inRange: li + param < width };
    case 'up':
      return { target: base + li - param, inRange: li - param >= 0 };
    case 'xor': {
      // Earlier subsections may be read; a later one gives the caller its own value.
      const t = lane ^ param;
      return { target: t, inRange: t < base + width };
    }
    case 'idx':
      // srcLane outside [0, width - 1] reads srcLane % width of the caller's own subsection.
      return { target: base + (((param % width) + width) % width), inRange: true };
  }
}

export function checkWidth(width: number, warp: number): string | null {
  if (!isPow2(width) || width > warp) return `width must be a power of two from 1 to ${warp}; ${width} is not (the guide: "Other values will produce undefined results")`; // i18n-ignore: error text thrown for invalid input, not shown
  return null;
}

export function shuffle(op: Op, values: (number | null)[], param: number, opts: ShuffleOptions = {}): LaneResult[] {
  const warp = values.length;
  const width = opts.width ?? warp;
  const bad = checkWidth(width, warp);
  if (bad) throw new Error(bad);
  const calls = range(warp).map((l) => (opts.mask ? !!opts.mask[l] : true) && !(opts.exited && opts.exited[l]));
  return range(warp).map((lane): LaneResult => {
    if (!calls[lane]) return { lane, calls: false, target: null, src: null, outcome: 'absent', value: null };
    const { target, inRange } = sourceOf(op, lane, param, width);
    if (!inRange) return { lane, calls: true, target, src: lane, outcome: 'own', value: values[lane] };
    if (target < 0 || target >= warp || !calls[target]) return { lane, calls: true, target, src: target, outcome: 'undefined', value: null };
    return { lane, calls: true, target, src: target, outcome: 'moved', value: values[target] };
  });
}

/** The mask as a hex literal: 32 bits for a 32-lane warp, 64 bits with a suffix for a 64-lane one. */
export function maskHex(mask: boolean[]): string {
  let v = 0n;
  mask.forEach((on, i) => {
    if (on) v |= 1n << BigInt(i);
  });
  const digits = Math.max(1, Math.ceil(mask.length / 4));
  return '0x' + v.toString(16).padStart(digits, '0') + (mask.length > 32 ? 'ull' : '');
}

/** The call as it would be written in CUDA (HIP spells it the same with a 64-bit mask). */
export function callText(op: Op, param: number, width: number, mask: boolean[]): string {
  const w = width !== mask.length ? `, ${width}` : '';
  return `v = ${OP_FN[op]}(${maskHex(mask)}, v, ${param}${w});`;
}

// ---------------------------------------------------------------- warp_reduce, step by step
export type Pattern = 'down' | 'xor';

export interface TraceOptions {
  /** down: Fig. 10.13 (__shfl_down_sync). xor: the butterfly (__shfl_xor_sync). */
  pattern: Pattern;
  /** The width the loop was written for: it starts at loopWidth / 2. Default: the hardware width. */
  loopWidth?: number;
  /** Lanes in the hardware warp. Default: values.length. */
  hardwareWidth?: number;
  /**
   * Lanes from `from` on hold no data: with 'return' they returned before the loop, with
   * 'zero' they stay and contribute 0, the identity of addition.
   */
  tail?: { from: number; mode: 'return' | 'zero' } | null;
}

export interface LaneStep {
  lane: number;
  /** Executes this step (has not returned). */
  live: boolean;
  outcome: Outcome;
  src: number | null;
  before: number | null;
  received: number | null;
  after: number | null;
  /** This lane's value after the step reaches lane 0 (down) or is wanted by every lane (xor). */
  inCone: boolean;
  /** In the cone, and adds two operands that both hold real elements from different lanes. */
  useful: boolean;
}

export interface TraceStep {
  stride: number;
  lanes: LaneStep[];
}

export interface Trace {
  pattern: Pattern;
  loopWidth: number;
  hardwareWidth: number;
  initial: (number | null)[];
  /** Lanes whose starting value reaches the lanes that need the result. */
  initialCone: boolean[];
  live: boolean[];
  steps: TraceStep[];
  final: (number | null)[];
  /** The sum the reduction should produce: every element that holds data. */
  total: number;
  /** Lane 0's result, null when undefined. */
  lane0: number | null;
  /** Lanes whose result equals the total. */
  holders: number[];
  /** Lane-steps executed, those whose result is used, and the rest. */
  laneOps: number;
  useful: number;
  wasted: number;
}

export function warpReduceTrace(values: (number | null)[], o: TraceOptions): Trace {
  const hw = o.hardwareWidth ?? values.length;
  const loopWidth = o.loopWidth ?? hw;
  if (values.length !== hw) throw new Error(`need ${hw} values, got ${values.length}`);
  if (!isPow2(loopWidth) || !isPow2(hw)) throw new Error('widths must be powers of two');
  const tail = o.tail ?? null;
  const live = range(hw).map((i) => !(tail && tail.mode === 'return' && i >= tail.from));
  const holds = range(hw).map((i) => !(tail && i >= tail.from));
  const initial = range(hw).map((i) => (tail && tail.mode === 'zero' && i >= tail.from ? 0 : values[i]));
  const op: Op = o.pattern === 'down' ? 'down' : 'xor';

  let cur = initial.slice();
  // How many real elements each lane's value stands for (a lane adding itself counts twice),
  // and, element by element, how many times each lane's original value is in it.
  let cnt: number[] = range(hw).map((i) => (holds[i] ? 1 : 0));
  let parts: number[][] = range(hw).map((i) => range(hw).map((j) => (i === j && holds[i] ? 1 : 0)));
  const raw: { stride: number; res: LaneResult[]; before: (number | null)[]; cntBefore: number[]; after: (number | null)[] }[] = [];
  for (let s = loopWidth / 2; s >= 1; s /= 2) {
    const res = shuffle(op, cur, s, { exited: live.map((x) => !x) });
    const after = cur.map((v, i) => (live[i] ? add(v, res[i].value) : v));
    const src = (i: number) => (live[i] && res[i].src != null && res[i].src! < hw ? res[i].src! : -1);
    const nc = cnt.map((c, i) => (src(i) >= 0 ? c + cnt[src(i)] : c));
    const np = parts.map((p, i) => (src(i) >= 0 ? p.map((x, j) => x + parts[src(i)][j]) : p));
    raw.push({ stride: s, res, before: cur, cntBefore: cnt, after });
    cur = after;
    cnt = nc;
    parts = np;
  }

  // The cone, from the end backwards: which lane-steps feed the lanes that need the result.
  let needed = new Set<number>(o.pattern === 'down' ? [0] : range(hw).filter((i) => live[i]));
  const steps: TraceStep[] = new Array(raw.length);
  for (let k = raw.length - 1; k >= 0; k--) {
    const { stride, res, before, cntBefore, after } = raw[k];
    const next = new Set(needed);
    const lanes = range(hw).map((i): LaneStep => {
      const r = res[i];
      const inCone = live[i] && needed.has(i);
      const useful = inCone && r.outcome === 'moved' && r.src !== i && cntBefore[i] > 0 && cntBefore[r.src!] > 0;
      if (inCone && r.src != null && r.src !== i) next.add(r.src);
      return { lane: i, live: live[i], outcome: r.outcome, src: r.src, before: before[i], received: r.value, after: after[i], inCone, useful };
    });
    steps[k] = { stride, lanes };
    needed = next;
  }

  const total = initial.reduce<number>((a, v, i) => a + (holds[i] ? (v ?? 0) : 0), 0);
  const laneOps = steps.reduce((a, st) => a + st.lanes.filter((l) => l.live).length, 0);
  const useful = steps.reduce((a, st) => a + st.lanes.filter((l) => l.useful).length, 0);
  return {
    pattern: o.pattern,
    loopWidth,
    hardwareWidth: hw,
    initial,
    initialCone: range(hw).map((i) => live[i] && needed.has(i)),
    live,
    steps,
    final: cur,
    total,
    lane0: cur[0],
    // A lane holds the total when it is defined and contains every element exactly once
    // (not when its number merely happens to be equal).
    holders: range(hw).filter((i) => live[i] && cur[i] != null && parts[i].every((x, j) => x === (holds[j] ? 1 : 0))),
    laneOps,
    useful,
    wasted: laneOps - useful,
  };
}

/** Small integers, the same for every lesson that wants numbers you can add by hand. */
export const smallValues = (n: number): number[] => range(n).map((i) => ((i * 7 + 3) % 9) + 1);
export const ones = (n: number): number[] => new Array(n).fill(1);

// ---------------------------------------------------------------- shared memory, iteration by iteration
/** A read or write of a shared slot that no barrier separates from another thread's access. */
export interface Race {
  kind: 'read-after-write' | 'write-after-read' | 'write-after-write';
  slot: number;
  /** The thread whose access comes second in the source. */
  thread: number;
  /** The other thread. */
  other: number;
  where: string;
}

/**
 * Shared memory with a barrier epoch: an access by one thread to a slot another thread
 * touched in the same epoch (no barrier in between) is a race, unless both only read.
 * A value read in a race is undefined.
 */
class Shared {
  vals: (number | null)[];
  writer: { thread: number; epoch: number }[];
  readers: Map<number, Set<number>> = new Map();
  epoch = 0;
  races: Race[] = [];
  /** Reads past the end of the array: undefined values. */
  pastEnd: { slot: number; thread: number; where: string }[] = [];
  reads = 0;
  writes = 0;
  constructor(size: number) {
    this.vals = new Array(size).fill(null);
    this.writer = range(size).map(() => ({ thread: -1, epoch: -1 }));
  }
  barrier() {
    this.epoch++;
    this.readers.clear();
  }
  read(slot: number, t: number, where: string): number | null {
    this.reads++;
    if (slot >= this.vals.length) {
      this.pastEnd.push({ slot, thread: t, where });
      return null;
    }
    const w = this.writer[slot];
    let v = this.vals[slot];
    if (w.thread >= 0 && w.thread !== t && w.epoch === this.epoch) {
      this.races.push({ kind: 'read-after-write', slot, thread: t, other: w.thread, where });
      v = null;
    }
    if (!this.readers.has(slot)) this.readers.set(slot, new Set());
    this.readers.get(slot)!.add(t);
    return v;
  }
  write(slot: number, t: number, v: number | null, where: string) {
    this.writes++;
    const w = this.writer[slot];
    if (w.thread >= 0 && w.thread !== t && w.epoch === this.epoch)
      this.races.push({ kind: 'write-after-write', slot, thread: t, other: w.thread, where });
    for (const r of this.readers.get(slot) ?? [])
      if (r !== t) this.races.push({ kind: 'write-after-read', slot, thread: t, other: r, where });
    this.vals[slot] = v;
    this.writer[slot] = { thread: t, epoch: this.epoch };
  }
}

// ---------------------------------------------------------------- the hybrid kernel, Fig. 10.14
/**
 * printed: the shared loop runs while stride >= warp width (Fig. 10.14).
 * early: it stops one level sooner (stride > warp width), leaving twice as many values as lanes.
 * double: the shared loop runs all the way to stride 1, and warp_reduce runs as well.
 * early2: stops one level sooner, and each lane of warp 0 adds two slots, t and t + width (ours).
 */
export type Handoff = 'printed' | 'early' | 'double' | 'early2';
export const HANDOFFS: Handoff[] = ['printed', 'early', 'double', 'early2'];

export interface HybridResult {
  blockDim: number;
  width: number;
  variant: Handoff;
  postLoopBarrier: boolean;
  /** The strides of the shared-memory iterations, in order. */
  strides: number[];
  barriers: number;
  sharedReads: number;
  sharedWrites: number;
  /** Shuffle-and-add steps of the one warp that finishes. */
  shuffleSteps: number;
  /** input_s at the hand-off, the slots that still hold a partial sum not yet added into a lower slot, and what warp 0 reads. */
  handoff: { values: (number | null)[]; live: boolean[]; reads: number[][] };
  warp: Trace;
  result: number | null;
  expected: number;
  correct: boolean;
  races: Race[];
  /** Slots read past the end of input_s (early2 when the block has fewer than two warps). */
  pastEnd: number;
}

export function hybrid(values: number[], blockDim: number, width: number, variant: Handoff = 'printed', opts: { postLoopBarrier?: boolean } = {}): HybridResult {
  if (!isPow2(blockDim) || !isPow2(width) || blockDim < width) throw new Error('blockDim and width must be powers of two with blockDim >= width');
  if (values.length !== 2 * blockDim) throw new Error(`need ${2 * blockDim} values`);
  const B = blockDim;
  const W = width;
  const postLoopBarrier = opts.postLoopBarrier ?? true;
  const sh = new Shared(B);
  for (let t = 0; t < B; t++) sh.write(t, t, values[t] + values[t + B], 'line 04'); // i18n-ignore: internal access-site tag, not shown
  const keep = (s: number) => (variant === 'printed' ? s >= W : variant === 'double' ? s >= 1 : s > W);
  const strides: number[] = [];
  for (let s = B / 2; s >= 1 && keep(s); s /= 2) {
    sh.barrier();
    strides.push(s);
    // Every active thread reads before any writes: one iteration is one step in time.
    const got = range(s).map((t) => add(sh.read(t, t, 'line 08'), sh.read(t + s, t, 'line 08'))); // i18n-ignore: internal access-site tag, not shown
    range(s).forEach((t) => sh.write(t, t, got[t], 'line 08')); // i18n-ignore: internal access-site tag, not shown
  }
  const liveTo = strides.length ? strides[strides.length - 1] : B;
  const snapshot = sh.vals.slice();
  if (postLoopBarrier) sh.barrier();
  const reads = range(W).map((t) => (variant === 'early2' ? [t, t + W] : [t]));
  const lanes = reads.map((rs, t) => rs.map((k) => sh.read(k, t, 'line 13')).reduce<number | null>((a, v) => add(a, v), 0)); // i18n-ignore: internal access-site tag, not shown
  const warp = warpReduceTrace(lanes, { pattern: 'down' });
  const expected = values.reduce((a, v) => a + v, 0);
  const shown = Math.min(B, 2 * W);
  return {
    blockDim: B,
    width: W,
    variant,
    postLoopBarrier,
    strides,
    barriers: strides.length + (postLoopBarrier ? 1 : 0),
    sharedReads: sh.reads,
    sharedWrites: sh.writes,
    shuffleSteps: log2(W),
    handoff: { values: snapshot.slice(0, shown), live: range(shown).map((k) => k < liveTo), reads },
    warp,
    result: warp.lane0,
    expected,
    correct: warp.lane0 === expected && sh.races.length === 0 && sh.pastEnd.length === 0,
    races: sh.races,
    pastEnd: sh.pastEnd.length,
  };
}

// ---------------------------------------------------------------- the two-stage kernel, Fig. 10.16
export type SlotState = 'written' | 'garbage' | 'past';

export interface TwoStageResult {
  blockDim: number;
  width: number;
  guarded: boolean;
  /** Warps in the block, so partial sums from stage 1. */
  warps: number;
  /** Entries of partialSums_s: BLOCK_DIM / WARP_SIZE as printed, BLOCK_DIM / 32 in the guarded version. */
  arraySize: number;
  /** Stage 1: each warp's sum, in lane 0. */
  warpSums: number[];
  /** The slots warp 0 can touch: max(width, arraySize) of them. Garbage and past-the-end slots read as undefined. */
  slots: { index: number; state: SlotState; value: number | null }[];
  /** Stage 2: what each lane of warp 0 starts with; read: the slot it reads, or null when it pads with 0. */
  lanes: { lane: number; read: number | null; value: number | null }[];
  stage2: Trace;
  result: number | null;
  expected: number;
  correct: boolean;
  barriers: number;
  sharedReads: number;
  sharedWrites: number;
}

/** The narrowest warp any of the three backends documents or assumes (ch10-facts B2): size shared arrays for it. */
export const NARROWEST_WARP = 32;

export function twoStage(values: number[], blockDim: number, width: number, guarded: boolean): TwoStageResult {
  if (!isPow2(blockDim) || !isPow2(width) || blockDim < width) throw new Error('blockDim and width must be powers of two with blockDim >= width');
  if (values.length !== 2 * blockDim) throw new Error(`need ${2 * blockDim} values`);
  const B = blockDim;
  const W = width;
  const warps = B / W;
  const arraySize = guarded ? Math.max(1, Math.floor(B / Math.min(NARROWEST_WARP, W))) : B / W;
  const warpSums = range(warps).map((w) => {
    const v = range(W).map((l) => values[w * W + l] + values[w * W + l + B]);
    return warpReduceTrace(v, { pattern: 'down' }).lane0!;
  });
  // As printed, every lane of warp 0 reads a slot, so draw as many as there are lanes;
  // guarded, only the array itself matters.
  const n = guarded ? arraySize : Math.max(W, arraySize);
  const slots = range(n).map((k) => ({
    index: k,
    state: (k >= arraySize ? 'past' : k < warps ? 'written' : 'garbage') as SlotState,
    value: k < arraySize && k < warps ? warpSums[k] : null,
  }));
  const lanes = range(W).map((l) => (guarded && l >= warps ? { lane: l, read: null, value: 0 } : { lane: l, read: l, value: slots[l].value }));
  const stage2 = warpReduceTrace(lanes.map((x) => x.value), { pattern: 'down', tail: guarded && warps < W ? { from: warps, mode: 'zero' } : null });
  const expected = values.reduce((a, v) => a + v, 0);
  return {
    blockDim: B,
    width: W,
    guarded,
    warps,
    arraySize,
    warpSums,
    slots,
    lanes,
    stage2,
    result: stage2.lane0,
    expected,
    correct: stage2.lane0 === expected,
    barriers: 1,
    sharedReads: guarded ? warps : W,
    sharedWrites: warps,
  };
}

// ---------------------------------------------------------------- what each kernel costs
/** shared: Fig. 10.10. hybrid: Fig. 10.14. twostage: Fig. 10.16 with the guarded stage-2 read. */
export type Kernel = 'shared' | 'hybrid' | 'twostage';
export const KERNELS: Kernel[] = ['shared', 'hybrid', 'twostage'];
export const KERNEL_LABEL: Record<Kernel, string> = {
  get shared() {
    return tr('shared memory (Fig. 10.10)', 'bộ nhớ chia sẻ (Fig. 10.10)');
  },
  get hybrid() {
    return tr('hybrid (Fig. 10.14)', 'kết hợp (Fig. 10.14)');
  },
  get twostage() {
    return tr('two-stage (Fig. 10.16)', 'hai giai đoạn (Fig. 10.16)');
  },
};

export interface CostStep {
  label: string;
  what: 'load' | 'shared' | 'shuffle';
  /** A __syncthreads() comes right before this step. */
  barrierBefore: boolean;
  /** Warps that execute the step. */
  warps: number;
  /** Lane slots spent: every lane of every warp that executes the step. */
  slots: number;
  /** Additions whose result reaches the block's sum and that combine real data. */
  useful: number;
}

export interface Costs {
  kernel: Kernel;
  blockDim: number;
  width: number;
  steps: CostStep[];
  barriers: number;
  sharedReads: number;
  sharedWrites: number;
  /** Shuffle-and-add steps one after another on the way to the result. */
  shuffleRounds: number;
  /** Warp-wide shuffle instructions issued in the whole block. */
  warpShuffles: number;
  slots: number;
  useful: number;
  wasted: number;
  correct: boolean;
}

export function blockCosts(kernel: Kernel, blockDim: number, width: number): Costs {
  const B = blockDim;
  const W = width;
  if (!isPow2(B) || !isPow2(W) || B < W) throw new Error('blockDim and width must be powers of two with blockDim >= width');
  const warps = B / W;
  const input = ones(2 * B);
  // Step labels are getters: S108 builds the costs in a memo keyed on the block shape only.
  const steps: CostStep[] = [
    {
      get label() {
        return tr('each thread adds its two elements', 'mỗi luồng cộng hai phần tử của nó');
      },
      what: 'load', // i18n-ignore: union member, not shown
      barrierBefore: false,
      warps,
      slots: warps * W,
      useful: B,
    },
  ];
  const sharedStep = (s: number, barrier: boolean): CostStep => ({
    get label() {
      return tr(`stride ${s} in shared memory`, `bước nhảy ${s} trong bộ nhớ chia sẻ`);
    },
    what: 'shared', // i18n-ignore: union member, not shown
    barrierBefore: barrier,
    warps: Math.ceil(s / W),
    slots: Math.ceil(s / W) * W,
    useful: s,
  });
  const shuffleSteps = (t: Trace, nWarps: number, label: string, firstBarrier: boolean): CostStep[] =>
    t.steps.map((st, k) => ({
      get label() {
        return tr(`${label}: shuffle down ${st.stride}`, `${label === 'every warp' ? 'mọi warp' : label}: shuffle down ${st.stride}`); // i18n-ignore: 'every warp' is the English key the Vietnamese is looked up by
      },
      what: 'shuffle' as const,
      barrierBefore: firstBarrier && k === 0,
      warps: nWarps,
      slots: nWarps * W,
      useful: nWarps * st.lanes.filter((l) => l.useful).length,
    }));
  let barriers = 0;
  let sharedReads = 0;
  let sharedWrites = 0;
  let shuffleRounds = 0;
  let warpShuffles = 0;
  let correct = false;
  if (kernel === 'shared') {
    sharedWrites += B;
    for (let s = B / 2; s >= 1; s /= 2) {
      steps.push(sharedStep(s, true));
      barriers++;
      sharedReads += 2 * s;
      sharedWrites += s;
    }
    sharedReads += 1; // thread 0 reads input_s[0]
    correct = true;
  } else if (kernel === 'hybrid') {
    const h = hybrid(input, B, W, 'printed');
    for (const s of h.strides) steps.push(sharedStep(s, true));
    steps.push(...shuffleSteps(h.warp, 1, 'warp 0', true)); // i18n-ignore: same in both languages; the label getter in shuffleSteps translates it
    barriers = h.barriers;
    sharedReads = h.sharedReads;
    sharedWrites = h.sharedWrites;
    shuffleRounds = h.shuffleSteps;
    warpShuffles = h.shuffleSteps;
    correct = h.correct;
  } else {
    const r = twoStage(input, B, W, true);
    const stage1 = warpReduceTrace(range(W).map(() => 2), { pattern: 'down' });
    steps.push(...shuffleSteps(stage1, warps, 'every warp', false)); // i18n-ignore: English key; the label getter in shuffleSteps translates it
    steps.push(...shuffleSteps(r.stage2, 1, 'warp 0', true)); // i18n-ignore: same in both languages; the label getter in shuffleSteps translates it
    barriers = r.barriers;
    sharedReads = r.sharedReads;
    sharedWrites = r.sharedWrites;
    shuffleRounds = 2 * log2(W);
    warpShuffles = (warps + 1) * log2(W);
    correct = r.correct;
  }
  const slots = steps.reduce((a, s) => a + s.slots, 0);
  const useful = steps.reduce((a, s) => a + s.useful, 0);
  return { kernel, blockDim: B, width: W, steps, barriers, sharedReads, sharedWrites, shuffleRounds, warpShuffles, slots, useful, wasted: slots - useful, correct };
}

export const BLOCK_SIZES = [64, 128, 256, 512, 1024];
