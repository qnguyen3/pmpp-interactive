// Halo tiling for 2D convolution: chapter 7, section 7.5 (book pp. 172–177,
// Figs. 7.11 to 7.14), as two models that section 7.5's interactives share.
//
// 1. Counting. Input and output tile sizes, the book's arithmetic-intensity
//    formula of p. 176 and its table (Fig. 7.14), the two thread organisations of
//    p. 173 (block = input tile, as Fig. 7.12; block = output tile, Exercise 11),
//    and the memory segments one warp touches when it loads or stores a tile row.
//    The segment rule is NVIDIA's documented one (32-byte sectors; notes/today/
//    ch06-facts.md A2, A3) or AMD's cache lines (B1), through sim/coalesce.ts.
//
// 2. A thread-level model of the Fig. 7.12 kernel for one block: every thread
//    loads one input-tile element (a zero for a ghost cell), waits at the barrier,
//    and then, if it is not on the outer ring, reads its patch from the shared
//    tile tap by tap and stores one output. Switches remove the barrier, the ghost
//    zeros, or the -FILTER_RADIUS offset, and several thread orders interleave the
//    threads, so that each defect can be watched failing (or hiding).
//
// What it leaves out, and what the app says wherever this model is shown:
//   - A thread's unit of progress is one source-level operation (a load, one
//     filter tap, a store); hardware interleaves instructions and runs warps.
//     "Lockstep" is an idealised single warp; a real block has several warps and
//     no promised order between them.
//   - Uninitialised shared memory is NaN, the worst case. On a device it holds
//     whatever was there before, which can look harmless.
//   - No timing, no caches, no bandwidth. The counting functions are the book's
//     model: an interior block, every load and store going to DRAM.
//   - The segment counts assume one instruction at a time and no cache.

import { tr } from '../lib/locale.ts';
import { warpAccess, type WarpAccess } from './coalesce';
import type { Schedule } from './tiled';

// ---------------------------------------------------------------- tile geometry

export interface TileGeometry {
  /** Filter radius r and filter width m = 2r + 1. */
  r: number;
  m: number;
  /** Edge of the input tile (output tile plus a halo of r on every side). */
  inTile: number;
  /** Edge of the output tile; 0 when the input tile is too small to have one. */
  outTile: number;
  inElems: number;
  outElems: number;
  /** Input elements outside the output tile's footprint: the halo. */
  haloElems: number;
  /** Input elements loaded per output element (Infinity without an output tile). */
  ratio: number;
}

function geometry(inTile: number, outTile: number, r: number): TileGeometry {
  const out = Math.max(0, outTile);
  const inElems = inTile * inTile;
  const outElems = out * out;
  return {
    r,
    m: 2 * r + 1,
    inTile,
    outTile: out,
    inElems,
    outElems,
    haloElems: inElems - outElems,
    ratio: outElems ? inElems / outElems : Infinity,
  };
}

/** Starting from the input tile (organisation 1, Fig. 7.12: the block is the input tile). */
export const tileGeometry = (inTile: number, r: number): TileGeometry => geometry(inTile, inTile - 2 * r, r);

/** Starting from the output tile (organisation 2, Exercise 11, or the book's t). */
export const fromOutTile = (t: number, r: number): TileGeometry => geometry(t + 2 * r, t, r);

// ---------------------------------------------------------------- arithmetic intensity (p. 176)

/**
 * FLOP per byte of global traffic for one interior block of the Fig. 7.12 kernel,
 * output tile t x t, filter m x m: t²·m²·2 FLOP over (t+m-1)² loads and t² stores
 * (book p. 176). `bytes` is the element size; the book uses 4.
 */
export function tiledIntensity(t: number, m: number, bytes = 4): number {
  if (t <= 0) return 0;
  return (t * t * m * m * 2) / (((t + m - 1) ** 2 + t * t) * bytes);
}

/** The factor that multiplies the ideal: 1 / (½(1 + (m-1)/t)² + ½). Tends to 1 as t grows. */
export function attenuation(t: number, m: number): number {
  if (t <= 0) return 0;
  return 1 / (0.5 * (1 + (m - 1) / t) ** 2 + 0.5);
}

/** Each input and output touched once: m²/4 FLOP/B for 4-byte elements (p. 167). */
export const idealIntensity = (m: number, bytes = 4): number => (2 * m * m) / (2 * bytes);

/** Fig. 7.7, F and N both from global memory: 0.25 FLOP/B for 4-byte elements (p. 167). */
export const basicIntensity = (bytes = 4): number => 2 / (2 * bytes);

/** Fig. 7.9, F in constant memory, one N load per tap: 0.5 FLOP/B for 4-byte elements (p. 171). */
export const constMemIntensity = (bytes = 4): number => 2 / bytes;

/** How many times less global traffic the tiled kernel asks for than the constant-memory kernel. */
export const trafficRatioVsConst = (t: number, m: number): number => tiledIntensity(t, m) / constMemIntensity();

/** Columns and rows of Fig. 7.14 (p. 177). */
export const FIG714_T = [8, 16, 28, 32, 64] as const;
export const FIG714_M = [3, 5, 7, 9, 11, 13] as const;

/** The values printed in Fig. 7.14, per filter width: five tile sizes, then the ideal. */
export const FIG714_PUBLISHED: Record<number, readonly number[]> = {
  3: [1.76, 1.99, 2.1, 2.11, 2.18, 2.25],
  5: [3.85, 4.88, 5.42, 5.52, 5.87, 6.25],
  7: [6.03, 8.48, 9.9, 10.17, 11.16, 12.25],
  9: [8.1, 12.46, 15.27, 15.8, 17.88, 20.25],
  11: [9.98, 16.62, 21.29, 22.22, 25.89, 30.25],
  13: [11.66, 20.8, 27.79, 29.23, 35.06, 42.25],
};

export interface Fig714Row {
  m: number;
  /** One value per column of FIG714_T. */
  values: number[];
  ideal: number;
}

/** Fig. 7.14 recomputed from the formula, for any element size. */
export function fig714(bytes = 4): Fig714Row[] {
  return FIG714_M.map((m) => ({
    m,
    values: FIG714_T.map((t) => tiledIntensity(t, m, bytes)),
    ideal: idealIntensity(m, bytes),
  }));
}

// ---------------------------------------------------------------- the two thread organisations (p. 173)

export type Organisation = 1 | 2;

export interface OrgInfo extends TileGeometry {
  which: Organisation;
  /** Threads in one block. */
  threads: number;
  /** Bytes of shared memory for the input tile. */
  sharedBytes: number;
  /** Threads with nothing to do in the compute phase. */
  idleCompute: number;
  /** Threads that compute an output. */
  computing: number;
  /** Most input-tile elements one thread loads with a flat loop (i = tid; i < in²; i += threads). */
  loadItersFlat: number;
  /** The same with a 2D loop striding by the block's width and height in each direction. */
  loadIters2D: number;
}

/**
 * The block for an output tile t x t and radius r. Organisation 1 (Fig. 7.12):
 * one thread per input-tile element, the outer ring idle while computing.
 * Organisation 2 (Exercise 11): one thread per output, looping to load.
 */
export function organisation(t: number, r: number, which: Organisation, bytes = 4): OrgInfo {
  const g = fromOutTile(t, r);
  const threads = which === 1 ? g.inElems : g.outElems;
  const flat = threads > 0 ? Math.ceil(g.inElems / threads) : 0;
  const per = which === 1 ? 1 : t > 0 ? Math.ceil(g.inTile / t) ** 2 : 0;
  return {
    ...g,
    which,
    threads,
    sharedBytes: g.inElems * bytes,
    idleCompute: which === 1 ? g.inElems - g.outElems : 0,
    computing: g.outElems,
    loadItersFlat: which === 1 ? 1 : flat,
    loadIters2D: per,
  };
}

/** For organisation 2's picture: the pass (0, 1, ...) of a flat loop in which each input-tile element is loaded. */
export function flatLoadPass(t: number, r: number): number[] {
  const g = fromOutTile(t, r);
  const threads = g.outElems;
  return Array.from({ length: g.inElems }, (_, i) => (threads ? Math.floor(i / threads) : 0));
}

// ---------------------------------------------------------------- one warp's row access (inefficiency 2, p. 177)

export interface RowAccess extends WarpAccess {
  /** Each block row the warp covers: first element's byte address and how many lanes are active. */
  rows: { startBytes: number; lanes: number }[];
}

/**
 * The segments one warp touches when its active lanes read or write consecutive
 * elements of a tile row. `startBytes` is the byte address of the first active
 * lane's element. A block row `blockWidth` threads wide is narrower than a warp of
 * `warpSize` lanes when warpSize > blockWidth; the warp then also covers the next
 * block rows, `rowPitchBytes` further on each (an image row apart).
 */
export function warpSectors(
  startBytes: number,
  activeLanes: number,
  warpSize: number,
  bytesPerElem = 4,
  opts: { blockWidth?: number; rowPitchBytes?: number; segmentBytes?: number } = {},
): RowAccess {
  const blockWidth = opts.blockWidth ?? 32;
  const seg = opts.segmentBytes ?? 32;
  const nRows = Math.max(1, Math.floor(warpSize / blockWidth));
  const rows: RowAccess['rows'] = [];
  const addrs: number[] = [];
  for (let k = 0; k < nRows; k++) {
    const s = startBytes + k * (opts.rowPitchBytes ?? 0);
    rows.push({ startBytes: s, lanes: activeLanes });
    for (let i = 0; i < activeLanes; i++) addrs.push(s + i * bytesPerElem);
  }
  return { ...warpAccess(addrs, bytesPerElem, seg), rows };
}

export type RowLayout = 'fig712' | 'tile32';

export interface ConvRowParams {
  /** 'fig712': a 32-wide block is the input tile, the output tile is 32 - 2r (organisation 1).
   *  'tile32': input, output and block all 32 wide (section 7.6's kernel). */
  layout: RowLayout;
  op: 'load' | 'store';
  bx: number;
  r: number;
  /** Image width in elements, and the image row whose tile row the warp handles. */
  width: number;
  row: number;
  warpSize: number;
  segmentBytes: number;
  bytes?: number;
}

/** First column and number of in-image columns of a warp's tile-row access. */
export function convRowSpan(p: ConvRowParams): { firstCol: number; lanes: number } {
  const block = 32;
  const out = p.layout === 'fig712' ? block - 2 * p.r : block;
  let lo: number;
  let n: number;
  if (p.op === 'load') {
    lo = p.layout === 'fig712' ? p.bx * out - p.r : p.bx * block;
    n = block;
  } else {
    lo = p.bx * out;
    n = out;
  }
  const a = Math.max(0, lo);
  const b = Math.min(p.width, lo + n);
  return { firstCol: a, lanes: Math.max(0, b - a) };
}

/** The segments a warp touches loading its input-tile row or storing its output row. */
export function convRowAccess(p: ConvRowParams): RowAccess & { firstCol: number } {
  const bytes = p.bytes ?? 4;
  const { firstCol, lanes } = convRowSpan(p);
  const start = (p.row * p.width + firstCol) * bytes;
  return {
    ...warpSectors(start, lanes, p.warpSize, bytes, { blockWidth: 32, rowPitchBytes: p.width * bytes, segmentBytes: p.segmentBytes }),
    firstCol,
  };
}

// ---------------------------------------------------------------- thread-level model of Fig. 7.12

export interface ConvProblem {
  width: number;
  height: number;
  r: number;
  m: number;
  /** Input image, row-major, small integers. */
  N: number[];
  /** Filter, m x m row-major, small integers, asymmetric in both directions. */
  F: number[];
  /** The book's formula (no flip) with zero ghost cells, computed directly. Exact. */
  ref: number[];
}

/** A small problem whose sums can be checked by hand. */
export function makeProblem(width: number, height: number, r: number): ConvProblem {
  const m = 2 * r + 1;
  const N: number[] = [];
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) N.push(((y * 5 + x * 7 + ((x * y) % 4) + 2) % 9) + 1);
  const F: number[] = [];
  for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) F.push(((i * 3 + j * 2) % 5) + 1);
  const ref: number[] = [];
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      let s = 0;
      for (let i = 0; i < m; i++)
        for (let j = 0; j < m; j++) {
          const yy = y - r + i;
          const xx = x - r + j;
          if (yy >= 0 && yy < height && xx >= 0 && xx < width) s += F[i * m + j] * N[yy * width + xx];
        }
      ref.push(s);
    }
  return { width, height, r, m, N, F, ref };
}

export interface ConvConfig {
  /** IN_TILE_DIM: the block is inTile x inTile threads. */
  inTile: number;
  /** FILTER_RADIUS. */
  r: number;
  /** __syncthreads() between the load phase and the compute phase. */
  barrier: boolean;
  /** The else branch that stores 0.0f for a ghost cell. */
  zeroGhost: boolean;
  /** The "- FILTER_RADIUS" in the col and row computations. */
  loadOffset: boolean;
}

export const CONV_CORRECT: Omit<ConvConfig, 'inTile' | 'r'> = { barrier: true, zeroGhost: true, loadOffset: true };

export const outTileOf = (cfg: Pick<ConvConfig, 'inTile' | 'r'>): number => cfg.inTile - 2 * cfg.r;

/** Blocks in x and y: one per output tile, rounded up (the book does not show the launch). */
export function gridOf(cfg: ConvConfig, prob: Pick<ConvProblem, 'width' | 'height'>): { x: number; y: number } {
  const o = outTileOf(cfg);
  return { x: Math.ceil(prob.width / o), y: Math.ceil(prob.height / o) };
}

export interface ThreadInfo {
  t: number;
  tx: number;
  ty: number;
  /** The kernel's row and col: the input element it loads, and for an active thread the output it computes. */
  row: number;
  col: number;
  /** Whether (row, col) is inside the image (else the load is a ghost cell). */
  inImage: boolean;
  tileRow: number;
  tileCol: number;
  /** Computes an output in the compute phase. */
  active: boolean;
  /** Why an inactive thread is idle: on the outer ring of the block, or its output is outside the image. */
  reason?: 'ring' | 'outside';
}

export function threadInfo(cfg: ConvConfig, prob: Pick<ConvProblem, 'width' | 'height'>, bx: number, by: number, tx: number, ty: number): ThreadInfo {
  const O = outTileOf(cfg);
  const off = cfg.loadOffset ? cfg.r : 0;
  const col = bx * O + tx - off;
  const row = by * O + ty - off;
  const inImage = row >= 0 && row < prob.height && col >= 0 && col < prob.width;
  const tileCol = tx - cfg.r;
  const tileRow = ty - cfg.r;
  const inner = tileCol >= 0 && tileCol < O && tileRow >= 0 && tileRow < O;
  return {
    t: ty * cfg.inTile + tx,
    tx,
    ty,
    row,
    col,
    inImage,
    tileRow,
    tileCol,
    active: inner && inImage,
    reason: !inner ? 'ring' : !inImage ? 'outside' : undefined,
  };
}

export type ConvOp = 'load' | 'wait' | 'release' | 'idle' | 'tap' | 'store';

export type ConvViolationKind =
  | 'read-before-write' // the slot's owner has not loaded it yet
  | 'unwritten-slot'; // nobody ever writes this slot (a ghost cell without its zero)

export interface ConvViolation {
  kind: ConvViolationKind;
  step: number;
  thread: number;
  message: string;
}

export interface ConvStep {
  index: number;
  /** Thread index ty*inTile+tx, or -1 for the barrier opening (all threads at once). */
  thread: number;
  tx: number;
  ty: number;
  op: ConvOp;
  /** The global element a load reads (inside: false for a ghost cell) or a store writes. */
  global?: { array: 'N' | 'P'; row: number; col: number; index: number; inside: boolean };
  /** The shared-tile slot a load writes, unless a ghost load wrote nothing. */
  write?: { r: number; c: number; value: number; ghost: boolean };
  /** One filter tap: the slot read, what it held, and the filter weight. */
  tap?: { fRow: number; fCol: number; slotR: number; slotC: number; value: number; weight: number; written: boolean };
  /** The thread's running sum after this step. */
  pvalue?: number;
  /** For an idle step: why the thread computes nothing. */
  reason?: 'ring' | 'outside';
  violation?: ConvViolation;
}

export interface ConvTrace {
  bx: number;
  by: number;
  steps: ConvStep[];
  violations: ConvViolation[];
  stores: { index: number; value: number; thread: number }[];
}

function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Runs one block under one thread order and records every operation. */
export function runTiledBlock(cfg: ConvConfig, prob: ConvProblem, bx: number, by: number, schedule: Schedule, seed = 1): ConvTrace {
  const IN = cfg.inTile;
  const R = cfg.r;
  const m = 2 * R + 1;
  const nT = IN * IN;
  const W = prob.width;
  const info = Array.from({ length: nT }, (_, t) => threadInfo(cfg, prob, bx, by, t % IN, Math.floor(t / IN)));

  type PlanOp = { op: 'load' | 'wait' | 'idle' | 'tap' | 'store'; fRow?: number; fCol?: number };
  const plans: PlanOp[][] = info.map((ti) => {
    const p: PlanOp[] = [{ op: 'load' }];
    if (cfg.barrier) p.push({ op: 'wait' });
    if (ti.active) {
      for (let fRow = 0; fRow < m; fRow++) for (let fCol = 0; fCol < m; fCol++) p.push({ op: 'tap', fRow, fCol });
      p.push({ op: 'store' });
    } else p.push({ op: 'idle' });
    return p;
  });

  // Each thread writes only its own slot. Will this slot be written at all?
  const willWrite = info.map((ti) => ti.inImage || cfg.zeroGhost);
  const slots = Array.from({ length: nT }, () => ({ value: NaN, written: false }));
  const pc = new Array(nT).fill(0);
  const waiting = new Array(nT).fill(false);
  const pvalue = new Array(nT).fill(0);
  const steps: ConvStep[] = [];
  const violations: ConvViolation[] = [];
  const stores: ConvTrace['stores'] = [];
  const rand = mulberry(seed);
  const name = (t: number) => tr(`thread (tx ${t % IN}, ty ${Math.floor(t / IN)})`, `luồng (tx ${t % IN}, ty ${Math.floor(t / IN)})`);

  const exec = (t: number) => {
    const ti = info[t];
    const p = plans[t][pc[t]];
    const s: ConvStep = { index: steps.length, thread: t, tx: ti.tx, ty: ti.ty, op: p.op };
    steps.push(s);
    if (p.op === 'wait') {
      waiting[t] = true;
      return;
    }
    if (p.op === 'load') {
      const idx = ti.row * W + ti.col;
      s.global = { array: 'N', row: ti.row, col: ti.col, index: idx, inside: ti.inImage };
      if (ti.inImage) {
        slots[t] = { value: prob.N[idx], written: true };
        s.write = { r: ti.ty, c: ti.tx, value: prob.N[idx], ghost: false };
      } else if (cfg.zeroGhost) {
        slots[t] = { value: 0, written: true };
        s.write = { r: ti.ty, c: ti.tx, value: 0, ghost: true };
      }
    } else if (p.op === 'idle') {
      s.reason = ti.reason;
    } else if (p.op === 'tap') {
      const sr = ti.tileRow + p.fRow!;
      const sc = ti.tileCol + p.fCol!;
      const slot = slots[sr * IN + sc];
      const weight = prob.F[p.fRow! * m + p.fCol!];
      s.tap = { fRow: p.fRow!, fCol: p.fCol!, slotR: sr, slotC: sc, value: slot.value, weight, written: slot.written };
      if (!slot.written) {
        const owner = sr * IN + sc;
        const v: ConvViolation = willWrite[owner]
          ? {
              kind: 'read-before-write',
              step: s.index,
              thread: t,
              message: tr(
                `${name(t)} read N_s[${sr}][${sc}] before ${owner === t ? 'it' : name(owner)} had loaded it`,
                `${name(t)} đọc N_s[${sr}][${sc}] trước khi ${owner === t ? 'chính nó' : name(owner)} nạp giá trị vào đó`,
              ),
            }
          : {
              kind: 'unwritten-slot',
              step: s.index,
              thread: t,
              message: tr(
                `${name(t)} read N_s[${sr}][${sc}], a ghost cell that nobody set to 0, so it holds whatever was there before`,
                `${name(t)} đọc N_s[${sr}][${sc}], một ô ma không ai đặt về 0, nên nó chứa bất cứ giá trị nào có ở đó từ trước`,
              ),
            };
        s.violation = v;
        violations.push(v);
      }
      pvalue[t] += weight * slot.value;
    } else if (p.op === 'store') {
      const idx = ti.row * W + ti.col;
      s.global = { array: 'P', row: ti.row, col: ti.col, index: idx, inside: true };
      stores.push({ index: idx, value: pvalue[t], thread: t });
    }
    s.pvalue = pvalue[t];
    pc[t]++;
  };

  const live = (t: number) => pc[t] < plans[t].length;
  const runnable = (t: number) => live(t) && !waiting[t];
  const release = () => {
    if (!waiting.some(Boolean)) return;
    if (!waiting.every((w, t) => w || !live(t))) return;
    steps.push({ index: steps.length, thread: -1, tx: -1, ty: -1, op: 'release' });
    for (let t = 0; t < nT; t++)
      if (waiting[t]) {
        waiting[t] = false;
        pc[t]++;
      }
  };

  const order = Array.from({ length: nT }, (_, i) => i);
  if (schedule === 'reversed') order.reverse();
  let guard = 0;
  while (order.some(live)) {
    if (++guard > 1e6) throw new Error('simulation did not terminate');
    if (schedule === 'lockstep') {
      for (const t of order) if (runnable(t)) exec(t);
    } else if (schedule === 'shuffled') {
      const c = order.filter(runnable);
      if (c.length) exec(c[Math.floor(rand() * c.length)]);
    } else {
      for (const t of order) while (runnable(t)) exec(t);
    }
    release();
  }
  return { bx, by, steps, violations, stores };
}

export interface ConvBlockState {
  /** The shared tile, one slot per thread: its value, and whether its owner has written it. */
  slots: { value: number; written: boolean; ghost: boolean }[];
  pvalue: number[];
  /** Per thread: index of the last step it executed, or -1. */
  last: number[];
  waiting: boolean[];
  /** Finished: stored its output, or found it has none. */
  done: boolean[];
  stores: ConvTrace['stores'];
  /** The barrier has opened. */
  released: boolean;
}

/** The block after its first n steps, for the stepper and the narrated scene. */
export function stateAt(cfg: ConvConfig, trace: ConvTrace, n: number): ConvBlockState {
  const nT = cfg.inTile * cfg.inTile;
  const st: ConvBlockState = {
    slots: Array.from({ length: nT }, () => ({ value: NaN, written: false, ghost: false })),
    pvalue: new Array(nT).fill(0),
    last: new Array(nT).fill(-1),
    waiting: new Array(nT).fill(false),
    done: new Array(nT).fill(false),
    stores: [],
    released: false,
  };
  for (let i = 0; i < Math.min(n, trace.steps.length); i++) {
    const s = trace.steps[i];
    if (s.thread < 0) {
      st.waiting.fill(false);
      st.released = true;
      continue;
    }
    st.last[s.thread] = i;
    if (s.write) st.slots[s.write.r * cfg.inTile + s.write.c] = { value: s.write.value, written: true, ghost: s.write.ghost };
    if (s.op === 'wait') st.waiting[s.thread] = true;
    if (s.pvalue !== undefined) st.pvalue[s.thread] = s.pvalue;
    if (s.op === 'idle') st.done[s.thread] = true;
    if (s.op === 'store') {
      st.done[s.thread] = true;
      st.stores.push({ index: s.global!.index, value: s.pvalue ?? 0, thread: s.thread });
    }
  }
  return st;
}

export interface ConvRunResult {
  P: number[];
  /** Outputs written with a value that differs from the reference. */
  wrong: number[];
  /** Outputs no block wrote. */
  missing: number[];
  traces: ConvTrace[];
  violations: (ConvViolation & { bx: number; by: number })[];
  correct: boolean;
}

/** Runs every block, one after another, and compares P with the reference. */
export function runAll(cfg: ConvConfig, prob: ConvProblem, schedule: Schedule, seed = 1): ConvRunResult {
  const g = gridOf(cfg, prob);
  const P = new Array(prob.width * prob.height).fill(NaN);
  const written = new Array(P.length).fill(false);
  const traces: ConvTrace[] = [];
  const violations: ConvRunResult['violations'] = [];
  for (let by = 0; by < g.y; by++)
    for (let bx = 0; bx < g.x; bx++) {
      const tr = runTiledBlock(cfg, prob, bx, by, schedule, seed + by * g.x + bx);
      traces.push(tr);
      for (const s of tr.stores)
        if (s.index >= 0 && s.index < P.length) {
          P[s.index] = s.value;
          written[s.index] = true;
        }
      for (const v of tr.violations) violations.push({ ...v, bx, by });
    }
  const wrong: number[] = [];
  const missing: number[] = [];
  for (let i = 0; i < P.length; i++) {
    if (!written[i]) missing.push(i);
    else if (!(P[i] === prob.ref[i])) wrong.push(i);
  }
  return { P, wrong, missing, traces, violations, correct: wrong.length === 0 && missing.length === 0 };
}
