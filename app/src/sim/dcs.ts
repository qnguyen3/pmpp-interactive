// Direct Coulomb summation on a small 2D slice (book §§21.2 to 21.4, pp. 515–522).
//
// What it models:
//   - A slice of W × H grid points at integer (i, j), unit grid spacing, and a few charged
//     atoms in a thin layer a fixed height above the slice, so no distance is zero.
//   - The float64 direct sum at every point, and float32 sums that follow each kernel's own
//     expression and update order (Math.fround after every operation), so that the order
//     dependence of the scatter kernel's atomics can be shown and tested.
//   - The five thread-to-point mappings the chapter discusses: scatter (Fig. 21.5), gather
//     (Fig. 21.6), the coarsened kernel with line 05 as printed (Fig. 21.8), the corrected
//     adjacent ownership, and the blockDim.x stride (Fig. 21.10); for any block shape,
//     coarsening factor and slice size: writers per point, conflicts, unwritten points,
//     whether an atomic is needed or a data race occurs, and each mapping's result.
//   - Distinct constant-memory addresses one warp instruction reads.
//   - The lanes' byte addresses of every write statement of one warp (or one float4 store
//     per thread), passed to coalesce.ts (imported read-only).
//   - Operation counts per atom for a coarsening factor F, counted from the listings, with
//     the book's printed figures beside them (notes/today/ch21-facts.md E1).
//   - The lockstep timelines that the "Scatter and gather, step by step" scene draws.
//
// What it leaves out, and what the app says wherever this model is used:
//   - 2D only, unit spacing. No time, no caches, no DRAM. Segments are counted one
//     instruction at a time.
//   - Same-address atomics are applied one per step in a fixed order the model chooses;
//     NVIDIA documents that they are serialised in an undefined order (infra A1) and
//     documents no hardware combining.
//   - A non-atomic += by several lanes of one warp to one address is modelled as "every
//     lane loads the old value, the highest lane's store survives". Which lane wins on a
//     GPU is not defined. Non-atomic updates from different statements or warps are
//     modelled as all landing, as on the course's CPU emulator; a GPU may lose some.

import { runs, warpAccess, type WarpAccess } from './coalesce';
import { tr } from '../lib/locale.ts';

const f = Math.fround;

// ---------------------------------------------------------------- atoms

export interface Atom {
  /** Position in grid units; the slice is the plane z = 0 and point (i, j) sits at (i, j, 0). */
  x: number;
  y: number;
  z: number;
  /** Charge. */
  q: number;
}

/** How far above the slice the atoms sit, in grid spacings. */
export const LAYER_HEIGHT = 0.75;

/** mulberry32: a small, well-mixed 32-bit generator, so a seed gives the same atoms everywhere. */
export function rng(seed: number): () => number {
  let h = seed >>> 0;
  return () => {
    h = (h + 0x6d2b79f5) >>> 0;
    let t = Math.imul(h ^ (h >>> 15), h | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Round to a float32 with few decimal digits, so positions read cleanly and are exact in the kernel's arithmetic. */
const clean = (v: number, step = 1 / 16) => f(Math.round(v / step) * step);

/**
 * n atoms at seeded random positions over a W × H slice, in a layer LAYER_HEIGHT above
 * it, with charges of either sign and magnitude 0.25 to 1. Every coordinate and charge is
 * a float32 with a short binary expansion.
 */
export function seededAtoms(n: number, W: number, H: number, seed: number): Atom[] {
  const r = rng(seed * 7919 + 17);
  const out: Atom[] = [];
  for (let k = 0; k < n; k++) {
    const x = clean(-0.4 + r() * (W - 0.2));
    const y = clean(-0.4 + r() * (H - 0.2));
    const mag = clean(0.25 + r() * 0.75);
    const q = r() < 0.5 ? -mag : mag;
    out.push({ x, y, z: f(LAYER_HEIGHT), q });
  }
  return out;
}

/** An atom moved to (x, y), kept in the layer and on the same float32 grid as seededAtoms. */
export function movedAtom(a: Atom, x: number, y: number): Atom {
  return { ...a, x: clean(x), y: clean(y) };
}

// ---------------------------------------------------------------- sums

/** The exact contribution of one atom at point (i, j), in float64. */
export function term64(a: Atom, i: number, j: number): number {
  const dx = i - a.x;
  const dy = j - a.y;
  const dz = 0 - a.z;
  return a.q / Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/** The float64 direct sum at every point, row-major (j * W + i). */
export function directSum(atoms: Atom[], W: number, H: number): number[] {
  const out = new Array<number>(W * H).fill(0);
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) for (const a of atoms) out[j * W + i] += term64(a, i, j);
  return out;
}

/** Sum of |contribution| at a point: the scale against which float32 error is judged. */
export function absScale(atoms: Atom[], i: number, j: number): number {
  return atoms.reduce((s, a) => s + Math.abs(term64(a, i, j)), 0);
}

/** One atom's contribution as Fig. 21.6 computes it in float32: q / sqrtf(dx*dx + dy*dy + dz*dz). */
export function termGather(a: Atom, i: number, j: number): number {
  const dx = f(f(i) - a.x);
  const dy = f(f(j) - a.y);
  const dz = f(0 - a.z);
  const r2 = f(f(f(dx * dx) + f(dy * dy)) + f(dz * dz));
  return f(a.q / f(Math.sqrt(r2)));
}

/** One atom's contribution as Fig. 21.5 computes it: dy2 and dz2 hoisted, charge / sqrtf(dx*dx + dy2 + dz2). */
export function termScatter(a: Atom, i: number, j: number): number {
  const dz = f(0 - a.z);
  const dz2 = f(dz * dz);
  const dy = f(f(j) - a.y);
  const dy2 = f(dy * dy);
  const dx = f(f(i) - a.x);
  const r2 = f(f(f(dx * dx) + dy2) + dz2);
  return f(a.q / f(Math.sqrt(r2)));
}

/**
 * One atom's contribution to point m of a coarsened thread whose first point is x0, as
 * Figs. 21.8 and 21.10 compute it: dx_m = dx0 + m·step (step 1 for adjacent points,
 * blockDim.x for the stride), then charge / sqrtf(dx_m*dx_m + (dy*dy + dz*dz)).
 */
export function termCoarse(a: Atom, x0: number, m: number, step: number, j: number): number {
  const dx0 = f(f(x0) - a.x);
  const dxm = m === 0 ? dx0 : f(dx0 + f(m * step));
  const dy = f(f(j) - a.y);
  const dz = f(0 - a.z);
  const dysqdzsq = f(f(dy * dy) + f(dz * dz));
  return f(a.q / f(Math.sqrt(f(f(dxm * dxm) + dysqdzsq))));
}

/** A float32 running sum of `values` taken in `order` (indices into values), starting from `start`. */
export function sumInOrder(values: number[], order: number[], start = 0): number {
  let s = f(start);
  for (const k of order) s = f(s + values[k]);
  return s;
}

// ---------------------------------------------------------------- the five mappings

export type KernelId = 'scatter' | 'gather' | 'printed' | 'adjacent' | 'strided';

export interface KernelInfo {
  id: KernelId;
  label: string;
  fig: string;
  coarsened: boolean;
  /** The index expression a thread starts from, as the listing writes it. */
  index: string;
}

export const KERNELS: Record<KernelId, KernelInfo> = {
  scatter: { id: 'scatter', get label() { return tr('scatter', 'scatter'); }, fig: 'Fig. 21.5', coarsened: false, index: 'n = (blockIdx.x*blockDim.x + threadIdx.x)*4' },
  gather: { id: 'gather', get label() { return tr('gather', 'gather'); }, fig: 'Fig. 21.6', coarsened: false, index: 'i = blockIdx.x*blockDim.x + threadIdx.x' },
  printed: { id: 'printed', get label() { return tr('coarsened, line 05 as printed', 'đã gộp việc, dòng 05 như bản in'); }, get fig() { return tr('Fig. 21.8 as printed', 'Fig. 21.8 như bản in'); }, coarsened: true, index: 'i = blockIdx.x*blockDim.x*F + threadIdx.x; writes i, i+1, …' },
  adjacent: { id: 'adjacent', get label() { return tr('coarsened, adjacent (corrected)', 'đã gộp việc, liền kề (đã sửa)'); }, get fig() { return tr('Fig. 21.8 corrected', 'Fig. 21.8 đã sửa'); }, coarsened: true, index: 'i = (blockIdx.x*blockDim.x + threadIdx.x)*F; writes i, i+1, …' },
  strided: { id: 'strided', get label() { return tr('coarsened, blockDim.x stride', 'đã gộp việc, bước nhảy blockDim.x'); }, fig: 'Fig. 21.10', coarsened: true, index: 'i = blockIdx.x*blockDim.x*F + threadIdx.x; writes i, i+blockDim.x, …' },
};
export const KERNEL_ORDER: KernelId[] = ['scatter', 'gather', 'printed', 'adjacent', 'strided'];

export interface MapConfig {
  kernel: KernelId;
  /** Slice size in points. */
  W: number;
  H: number;
  atoms: Atom[];
  /** blockDim.x; scatter launches 1D blocks of this many threads over the atoms. */
  blockX: number;
  /** blockDim.y (ignored by scatter). */
  blockY: number;
  /** Coarsening factor (coarsened kernels only). */
  F: number;
  /** Scatter only: true for atomicAdd as the listing has it, false for a plain +=. */
  atomic?: boolean;
  /**
   * Lanes per warp, for the lost-update model and the constant-address count. Default 32,
   * NVIDIA's warp width and the book's assumption (ch05-facts B5: 64 on AMD Instinct).
   */
  warpWidth?: number;
}

export interface ThreadInfo {
  /** Linear id over the whole launch: block-major, then threadIdx in linear order. */
  id: number;
  bx: number;
  by: number;
  tx: number;
  ty: number;
  /** Warp within the launch (blocks are split into warps by linear thread index) and lane. */
  warp: number;
  lane: number;
  /** Scatter: the thread's atom (or -1 for a thread past the last atom). */
  atom?: number;
  /** Gather and coarsened: the kernel's i (first point) and j. */
  i?: number;
  j?: number;
  /** Points this thread writes, as flat indices j*W + i, in statement order. */
  points: number[];
  /** Writes the kernel would make outside the slice (stopped by a bound check, or out of range as printed). */
  outside: number;
  /** Atom indices this thread reads. */
  reads: number[];
}

export type Verdict = 'one-writer' | 'atomic' | 'race';

export interface MapResult {
  config: MapConfig;
  launch: { gridX: number; gridY: number; blockX: number; blockY: number; threads: number };
  threads: ThreadInfo[];
  /** Per point: ids of the threads that write it. */
  writers: number[][];
  /** Points with more than one writer, and with none. */
  conflicts: number;
  unwritten: number;
  /** Threads that write nothing (the padding of Fig. 21.9, or scatter threads past the last atom). */
  idle: number;
  /** One writer per point; atomics needed and present; or a data race. */
  verdict: Verdict;
  /** What the kernel leaves in each point (float32, starting from zero). */
  potentials: number[];
  /** float64 direct sum. */
  direct: number[];
  /** True when every point is within float32 tolerance of the direct sum. */
  correct: boolean;
  /** Points whose result is wrong. */
  wrong: number;
  /** Distinct constant-memory addresses one warp instruction reads (the launch's first warp). */
  constAddrsPerWarp: number;
  /** Atom-point pairs one busy thread evaluates. */
  pairsPerThread: number;
}

const ceilDiv = (a: number, b: number) => Math.ceil(a / b);

/** The thread-to-point mapping of a kernel, and what it leaves in the slice. */
export function mapKernel(cfg: MapConfig): MapResult {
  const { kernel, W, H, atoms, blockX, blockY } = cfg;
  const F = KERNELS[kernel].coarsened ? cfg.F : 1;
  const ww = cfg.warpWidth ?? 32;
  const atomic = kernel === 'scatter' ? cfg.atomic !== false : false;
  const nA = atoms.length;
  const threads: ThreadInfo[] = [];
  let gridX: number, gridY: number, bdy: number;
  if (kernel === 'scatter') {
    gridX = Math.max(1, ceilDiv(nA, blockX));
    gridY = 1;
    bdy = 1;
  } else {
    gridX = Math.max(1, ceilDiv(W, blockX * F));
    gridY = Math.max(1, ceilDiv(H, blockY));
    bdy = blockY;
  }
  const perBlock = blockX * bdy;
  const warpsPerBlock = ceilDiv(perBlock, ww);
  let id = 0;
  for (let by = 0; by < gridY; by++)
    for (let bx = 0; bx < gridX; bx++)
      for (let ty = 0; ty < bdy; ty++)
        for (let tx = 0; tx < blockX; tx++) {
          const L = ty * blockX + tx;
          const warp = (by * gridX + bx) * warpsPerBlock + Math.floor(L / ww);
          const lane = L % ww;
          const t: ThreadInfo = { id: id++, bx, by, tx, ty, warp, lane, points: [], outside: 0, reads: [] };
          if (kernel === 'scatter') {
            const a = bx * blockX + tx;
            t.atom = a < nA ? a : -1;
            if (a < nA) {
              t.reads = [a];
              for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) t.points.push(j * W + i);
            }
          } else {
            const j = by * blockY + ty;
            const i0 =
              kernel === 'adjacent' ? (bx * blockX + tx) * F
              : kernel === 'gather' ? bx * blockX + tx
              : bx * blockX * F + tx; // printed and strided share line 05
            t.i = i0;
            t.j = j;
            const step = kernel === 'strided' ? blockX : 1;
            for (let m = 0; m < F; m++) {
              const i = i0 + m * step;
              if (j < H && i < W) t.points.push(j * W + i);
              else t.outside++;
            }
            if (t.points.length) t.reads = atoms.map((_, k) => k);
          }
          threads.push(t);
        }

  const writers: number[][] = Array.from({ length: W * H }, () => []);
  for (const t of threads) for (const p of t.points) writers[p].push(t.id);
  const conflicts = writers.filter((w) => w.length > 1).length;
  const unwritten = writers.filter((w) => w.length === 0).length;
  const idle = threads.filter((t) => t.points.length === 0).length;
  const verdict: Verdict = conflicts === 0 ? 'one-writer' : atomic ? 'atomic' : 'race';

  // What each kernel leaves behind.
  const direct = directSum(atoms, W, H);
  const potentials = new Array<number>(W * H).fill(0);
  if (kernel === 'scatter') {
    for (let p = 0; p < W * H; p++) {
      const i = p % W;
      const j = Math.floor(p / W);
      const ws = writers[p].map((tid) => threads[tid]);
      if (atomic) {
        // Every update lands; the model applies them in thread order (one possible order).
        potentials[p] = sumInOrder(ws.map((t) => termScatter(atoms[t.atom!], i, j)), ws.map((_, k) => k));
      } else {
        // Lockstep: in each warp every lane loads the same old value; the highest lane's store survives.
        const lastOfWarp = new Map<number, ThreadInfo>();
        for (const t of ws) lastOfWarp.set(t.warp, t);
        let s = 0;
        for (const t of [...lastOfWarp.values()].sort((a, b) => a.warp - b.warp)) s = f(s + termScatter(atoms[t.atom!], i, j));
        potentials[p] = s;
      }
    }
  } else {
    for (const t of threads) {
      const step = kernel === 'strided' ? blockX : 1;
      t.points.forEach((p) => {
        const i = p % W;
        const j = Math.floor(p / W);
        const m = (i - t.i!) / step;
        let e = 0;
        for (const a of atoms) e = f(e + (kernel === 'gather' ? termGather(a, i, j) : termCoarse(a, t.i!, m, step, j)));
        potentials[p] = f(potentials[p] + e); // all writers land (see the header)
      });
    }
  }
  let wrong = 0;
  for (let p = 0; p < W * H; p++) {
    const i = p % W;
    const j = Math.floor(p / W);
    if (Math.abs(potentials[p] - direct[p]) > 1e-5 * absScale(atoms, i, j) + 1e-12) wrong++;
  }
  const first = threads.filter((t) => t.warp === 0 && t.points.length > 0);
  const constAddrsPerWarp = kernel === 'scatter' ? new Set(first.map((t) => t.atom)).size : first.length ? 1 : 0;
  const pairsPerThread = Math.max(0, ...threads.map((t) => t.points.length * (kernel === 'scatter' ? 1 : nA)));
  return {
    config: cfg,
    launch: { gridX, gridY, blockX, blockY: bdy, threads: threads.length },
    threads,
    writers,
    conflicts,
    unwritten,
    idle,
    verdict,
    potentials,
    direct,
    correct: wrong === 0,
    wrong,
    constAddrsPerWarp,
    pairsPerThread,
  };
}

// ---------------------------------------------------------------- write statements and coalescing

export type Ownership = 'adjacent' | 'strided' | 'float4';

export interface WriteParams {
  ownership: Ownership;
  /** Coarsening factor. float4 needs a multiple of 4. */
  F: number;
  /** Lanes in the warp (wavefront, SIMD-group). */
  warpWidth: number;
  blockX: number;
  blockY: number;
  /** Elements per slice row (the row pitch). */
  rowPitch: number;
  /** Segment size in bytes (32 for NVIDIA sectors, 64 or 128 for AMD lines, a choice for Apple). */
  segmentBytes: number;
  /** Which warp of block (0, 0). Default 0. */
  warp?: number;
}

export interface StatementLane {
  lane: number;
  tx: number;
  ty: number;
  /** Element index into the slice (row-major, rowPitch per row) and its byte address. */
  element: number;
  byte: number;
}

export interface Statement {
  /** Write statement m (0 .. F−1), or float4 store m (0 .. F/4 − 1). */
  m: number;
  lanes: StatementLane[];
  bytesPerLane: number;
  access: WarpAccess;
  /** Separate runs of contiguous bytes. */
  runs: number;
}

export interface WriteResult {
  statements: Statement[];
  /** Summed over the statements. */
  transactions: number;
  bytesFetched: number;
  bytesNeeded: number;
  /** Every lane's address is a multiple of its access size (float4 needs 16 bytes). */
  aligned: boolean;
}

/**
 * The byte addresses one warp of block (0, 0) writes in each write statement of the
 * coarsened kernel. Lane l of the warp is linear thread L = warp·width + l of the block,
 * with tx = L mod blockDim.x and ty = L div blockDim.x, so a warp wider than blockDim.x
 * covers several rows. Adjacent: thread owns tx·F .. tx·F + F − 1, statement m writes
 * tx·F + m. Stride: statement m writes tx + m·blockDim.x (Fig. 21.10). float4: F/4 stores
 * of 16 bytes at tx·F + 4m (Exercise 4).
 */
export function writeStatements(p: WriteParams): WriteResult {
  const w = p.warp ?? 0;
  const perBlock = p.blockX * p.blockY;
  const lanesOf = (fn: (tx: number, ty: number) => number): StatementLane[] => {
    const out: StatementLane[] = [];
    for (let l = 0; l < p.warpWidth; l++) {
      const L = w * p.warpWidth + l;
      if (L >= perBlock) break;
      const tx = L % p.blockX;
      const ty = Math.floor(L / p.blockX);
      const element = ty * p.rowPitch + fn(tx, ty);
      out.push({ lane: l, tx, ty, element, byte: element * 4 });
    }
    return out;
  };
  const statements: Statement[] = [];
  if (p.ownership === 'float4') {
    if (p.F % 4 !== 0) throw new Error('float4 stores need a coarsening factor that is a multiple of 4');
    for (let m = 0; m < p.F / 4; m++) {
      const lanes = lanesOf((tx) => tx * p.F + 4 * m);
      const addrs = lanes.map((l) => l.byte);
      statements.push({ m, lanes, bytesPerLane: 16, access: warpAccess(addrs, 16, p.segmentBytes), runs: runs(addrs, 16) });
    }
  } else {
    for (let m = 0; m < p.F; m++) {
      const lanes = lanesOf((tx) => (p.ownership === 'adjacent' ? tx * p.F + m : tx + m * p.blockX));
      const addrs = lanes.map((l) => l.byte);
      statements.push({ m, lanes, bytesPerLane: 4, access: warpAccess(addrs, 4, p.segmentBytes), runs: runs(addrs, 4) });
    }
  }
  const aligned = statements.every((s) => s.lanes.every((l) => l.byte % s.bytesPerLane === 0));
  return {
    statements,
    transactions: statements.reduce((n, s) => n + s.access.transactions, 0),
    bytesFetched: statements.reduce((n, s) => n + s.access.bytesFetched, 0),
    bytesNeeded: statements.reduce((n, s) => n + s.access.bytesNeeded, 0),
    aligned,
  };
}

// ---------------------------------------------------------------- operation counts (E1, Exercise 2)

export interface OpCounts {
  /** Constant-memory loads. */
  loads: number;
  sub: number;
  add: number;
  /** Multiplications with the m·gridspacing products hoisted out of the atom loop. */
  mul: number;
  /** Multiplications as the listing writes them. */
  mulUnhoisted: number;
  div: number;
  sqrt: number;
  /** Loop branches. */
  branch: number;
}

/** Per atom, F threads of Fig. 21.6 (one point each), counted from the listing. */
export function opsGather(F: number): OpCounts {
  return { loads: 4 * F, sub: 3 * F, add: 3 * F, mul: 3 * F, mulUnhoisted: 3 * F, div: F, sqrt: F, branch: F };
}

/**
 * Per atom, one coarsened thread with F points (Fig. 21.8 generalised), counted from the
 * listing: 4 loads; dx0, dy, dz; F − 1 additions for dx1.., one for dysqdzsq, F inside the
 * roots, F accumulates; dy·dy, dz·dz and F squares, plus the m·gridspacing products for
 * m ≥ 2 when they are not hoisted.
 */
export function opsCoarsened(F: number): OpCounts {
  return {
    loads: 4,
    sub: 3,
    add: F - 1 + 1 + F + F,
    mul: F + 2,
    mulUnhoisted: F + 2 + Math.max(0, F - 2),
    div: F,
    sqrt: F,
    branch: 1,
  };
}

/** Arithmetic operations without square roots (the book counts none). */
export const arithmetic = (o: OpCounts, hoisted = false) => o.sub + o.add + (hoisted ? o.mul : o.mulUnhoisted) + o.div;

/** The book's printed counts for four grid points (p. 521), which do not match the listings (E1). */
export const BOOK_OPS_F4 = {
  gather: { loads: 16, sub: 12, add: 12, mul: 12, div: 12, sqrt: null, total: 48 },
  coarsened: { loads: 4, sub: 3, add: 11, mul: 6, div: 4, sqrt: null, total: 24 },
} as const;

// ---------------------------------------------------------------- the scene: lockstep timelines

/**
 * The scene's problem: a row of six points and four atoms. Positions and charges are
 * chosen once; every number the scene draws is computed from them below.
 */
export const SCENE = {
  W: 6,
  atoms: [
    { x: 0.75, y: 0.5, z: LAYER_HEIGHT, q: 0.8125 },
    { x: 2.25, y: -0.5, z: LAYER_HEIGHT, q: -0.4375 },
    { x: 3.5, y: 0.75, z: LAYER_HEIGHT, q: 0.9375 },
    { x: 4.75, y: -0.25, z: LAYER_HEIGHT, q: -0.6875 },
  ] as Atom[],
  /** The two orders in which the scene's two runs apply each point's queue of updates. */
  runA: [0, 1, 2, 3],
  runB: [2, 0, 3, 1],
};

export interface ScatterStep {
  /** The point every thread updates at this step (lockstep). */
  point: number;
  /** Each thread's float32 contribution (thread k holds atom k). */
  values: number[];
  /** The atoms' constant-memory addresses read at this step (one per lane). */
  constAddrs: number[];
}

/** Fig. 21.5 in lockstep on a row of W points: at step s every thread updates point s. */
export function scatterTimeline(atoms: Atom[], W: number): ScatterStep[] {
  return Array.from({ length: W }, (_, s) => ({
    point: s,
    values: atoms.map((a) => termScatter(a, s, 0)),
    constAddrs: atoms.map((_, k) => 4 * k),
  }));
}

export interface GatherStep {
  /** The atom every lane reads at this step (one address, broadcast). */
  atom: number;
  constAddr: number;
  /** Each point's register after this step. */
  registers: number[];
}

/** Fig. 21.6 in lockstep on a row of W points: at step s every thread adds atom s into its own register. */
export function gatherTimeline(atoms: Atom[], W: number): GatherStep[] {
  const reg = new Array<number>(W).fill(0);
  return atoms.map((a, s) => {
    for (let i = 0; i < W; i++) reg[i] = f(reg[i] + termGather(a, i, 0));
    return { atom: s, constAddr: 4 * s, registers: [...reg] };
  });
}

/** The float32 total of point p after all scatter updates, applied in `order` (thread indices). */
export function scatterTotal(atoms: Atom[], p: number, order: number[]): number {
  return sumInOrder(atoms.map((a) => termScatter(a, p, 0)), order);
}

/** Points whose float32 scatter total depends on which of the two orders is used. */
export function orderSensitive(atoms: Atom[], W: number, a: number[], b: number[]): { point: number; a: number; b: number }[] {
  const out: { point: number; a: number; b: number }[] = [];
  for (let p = 0; p < W; p++) {
    const x = scatterTotal(atoms, p, a);
    const y = scatterTotal(atoms, p, b);
    if (x !== y) out.push({ point: p, a: x, b: y });
  }
  return out;
}

/** The tally the scene's last part shows. */
export function sceneTally(atoms: Atom[], W: number) {
  const sc = scatterTimeline(atoms, W);
  return {
    scatterUpdates: sc.reduce((n, s) => n + s.values.length, 0),
    scatterAddresses: new Set(sc.map((s) => s.point)).size,
    updatesPerAddress: atoms.length,
    scatterConstPerStep: new Set(sc[0].constAddrs).size,
    gatherAdds: atoms.length * W,
    gatherWrites: W,
    gatherConstPerStep: 1,
  };
}

/** A float32 shown with enough digits to tell neighbouring floats apart. */
export const f32text = (v: number) => v.toPrecision(9);
