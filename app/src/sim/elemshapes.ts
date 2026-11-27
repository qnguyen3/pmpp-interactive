// Which thread writes which element, and by which path, for three elementwise launch styles
// read in the X3 facts pass (notes/today/x3-facts.md B1):
//
//   per-element  llama.cpp CUDA unary_op_kernel: i = blockDim.x*blockIdx.x + threadIdx.x,
//                if (i >= k) return; one element. Launch (k + B - 1) / B blocks of B.
//   binary_vv    MLX CUDA: index = thread rank; if ((index + 1) * N_READS > size) a scalar loop
//                over index*N_READS .. size-1, else load_vector<N_READS>, which reinterprets the
//                pointer as an aligned N-wide vector only if is_aligned<N>(ptr), else a scalar loop.
//   grid-stride  llama.cpp Metal kernel_swiglu: for (i0 = tgpig.x*ntg.x + tpitg.x; i0 < ne0;
//                i0 += ntg.x*ntgs.x), a fixed grid of ntgs.x groups of ntg.x threads.
//
// What this model leaves out: time, caches and the cost of anything; only which accesses are
// made. MLX's own launch computation was not read in the facts pass, so binary_vv's launch here
// is ours: ceil(n / N_READS) threads, rounded up to whole blocks. Only the loads of input `a`
// are modelled (store_vector's body was not read). The pointer offset applies to the view of
// every operand; the allocation itself is taken to be aligned to every vector width used.

import { tr } from '../lib/locale.ts';

export type Style = 'per-element' | 'binary_vv' | 'grid-stride';
export const STYLES: { id: Style; label: string; who: string }[] = [
  {
    id: 'per-element',
    get label() {
      return tr('one thread per element', 'mỗi luồng một phần tử');
    },
    who: 'llama.cpp CUDA unary_op_kernel', // i18n-ignore: library and kernel name
  },
  {
    id: 'binary_vv',
    get label() {
      return tr('N_READS per thread', 'N_READS phần tử mỗi luồng');
    },
    who: 'MLX CUDA binary_vv', // i18n-ignore: library and kernel name
  },
  {
    id: 'grid-stride',
    get label() {
      return tr('grid-stride loop', 'vòng lặp grid-stride');
    },
    who: 'llama.cpp Metal kernel_swiglu', // i18n-ignore: library and kernel name
  },
];
export const N_READS_CHOICES = [1, 2, 4, 8] as const;
export const BLOCK_CHOICES = [32, 64, 128, 256] as const;
/** Width of the warps drawn: CUDA's warp. Metal's SIMD-group width is read from the pipeline instead. */
export const WARP = 32;
/** Threadgroups in the grid-stride launch: our choice, fixed, to show a grid smaller than n. */
export const STRIDE_GROUPS = 2;

export interface LaunchCfg {
  style: Style;
  n: number;
  block: number;
  /** Elements per thread for binary_vv. */
  nReads: number;
  /** Offset of the view from an aligned allocation, in floats. */
  offset: number;
}

/** How one thread spent its time. */
export type Path = 'vector' | 'tail' | 'misaligned' | 'scalar' | 'idle';
export const PATH_LABEL: Record<Path, string> = {
  get vector() {
    return tr('vector load', 'nạp vector');
  },
  get tail() {
    return tr('scalar tail', 'phần đuôi vô hướng');
  },
  get misaligned() {
    return tr('scalar, misaligned', 'vô hướng, lệch căn chỉnh');
  },
  get scalar() {
    return tr('one element', 'một phần tử');
  },
  get idle() {
    return tr('idle after the guard', 'rảnh sau điều kiện chặn');
  },
};

/** One load instruction of `a` as a warp issues it: the element index each active lane reads. */
export interface LoadInstr {
  /** Which load of the thread this is (0 for a single vector load, 0..N-1 in a scalar loop). */
  k: number;
  lanes: { lane: number; element: number; width: number }[];
}

export interface ThreadPlan {
  t: number;
  warp: number;
  /** The first branch the thread takes: the main path, or the tail / guard side. */
  branch: 'main' | 'tail' | 'out' | 'loop';
  path: Path;
  /** Elements written, in order. */
  elements: number[];
  /** Iterations of a grid-stride loop (or 1 / 0 for the other styles). */
  iterations: number;
}

export interface WarpInfo {
  w: number;
  threads: [number, number];
  paths: Path[];
  /** The warp's threads do not all take the same branch (or, in a loop, the same trip count). */
  diverges: boolean;
  loads: LoadInstr[];
}

export interface Launch {
  cfg: LaunchCfg;
  threads: number;
  blocks: number;
  aligned: boolean;
  plans: ThreadPlan[];
  /** For each element, the thread that wrote it (-1 if none). */
  writer: number[];
  /** For each element, how many threads wrote it. */
  writes: number[];
  /** For each element, the path that wrote it. */
  pathOf: (Path | null)[];
  warps: WarpInfo[];
}

const ceilDiv = (a: number, b: number) => Math.floor((a + b - 1) / b);

/** MLX's is_aligned<N>(ptr) for a float view `offset` floats past an aligned allocation. */
export const isAligned = (offset: number, nReads: number) => (offset * 4) % (4 * nReads) === 0;

/** Threads launched for a style: whole blocks, as the CUDA launches compute them. */
export function threadsFor(cfg: LaunchCfg): number {
  if (cfg.style === 'per-element') return ceilDiv(cfg.n, cfg.block) * cfg.block;
  if (cfg.style === 'binary_vv') return ceilDiv(ceilDiv(cfg.n, cfg.nReads), cfg.block) * cfg.block;
  return STRIDE_GROUPS * cfg.block;
}

function planThread(cfg: LaunchCfg, t: number, threads: number, aligned: boolean): ThreadPlan {
  const { n, nReads } = cfg;
  const warp = Math.floor(t / WARP);
  if (cfg.style === 'per-element') {
    return t < n
      ? { t, warp, branch: 'main', path: 'scalar', elements: [t], iterations: 1 }
      : { t, warp, branch: 'out', path: 'idle', elements: [], iterations: 0 };
  }
  if (cfg.style === 'binary_vv') {
    if ((t + 1) * nReads > n) {
      const elements: number[] = [];
      for (let i = t * nReads; i < n; i++) elements.push(i);
      return { t, warp, branch: 'tail', path: elements.length ? 'tail' : 'idle', elements, iterations: elements.length ? 1 : 0 };
    }
    const elements = Array.from({ length: nReads }, (_, j) => t * nReads + j);
    return { t, warp, branch: 'main', path: aligned ? 'vector' : 'misaligned', elements, iterations: 1 };
  }
  const elements: number[] = [];
  for (let i = t; i < n; i += threads) elements.push(i);
  return { t, warp, branch: 'loop', path: elements.length ? 'scalar' : 'idle', elements, iterations: elements.length };
}

/** The loads of `a` a warp issues, lane by lane, for its threads' plans. */
function warpLoads(cfg: LaunchCfg, plans: ThreadPlan[]): LoadInstr[] {
  const out: LoadInstr[] = [];
  if (cfg.style === 'binary_vv') {
    const vec = plans.filter((p) => p.path === 'vector');
    if (vec.length) out.push({ k: 0, lanes: vec.map((p) => ({ lane: p.t % WARP, element: p.elements[0], width: cfg.nReads })) });
    // Scalar loops, misaligned or tail: load k of every thread that has a k-th element.
    const scalar = plans.filter((p) => p.path === 'misaligned' || p.path === 'tail');
    const maxK = Math.max(0, ...scalar.map((p) => p.elements.length));
    for (let k = 0; k < maxK; k++) {
      const lanes = scalar.filter((p) => p.elements.length > k).map((p) => ({ lane: p.t % WARP, element: p.elements[k], width: 1 }));
      if (lanes.length) out.push({ k, lanes });
    }
    return out;
  }
  const maxK = Math.max(0, ...plans.map((p) => p.elements.length));
  for (let k = 0; k < maxK; k++) {
    const lanes = plans.filter((p) => p.elements.length > k).map((p) => ({ lane: p.t % WARP, element: p.elements[k], width: 1 }));
    if (lanes.length) out.push({ k, lanes });
  }
  return out;
}

export function launch(cfg: LaunchCfg): Launch {
  const threads = threadsFor(cfg);
  const aligned = cfg.style !== 'binary_vv' || isAligned(cfg.offset, cfg.nReads);
  const plans = Array.from({ length: threads }, (_, t) => planThread(cfg, t, threads, aligned));
  const writer = new Array<number>(cfg.n).fill(-1);
  const writes = new Array<number>(cfg.n).fill(0);
  const pathOf = new Array<Path | null>(cfg.n).fill(null);
  for (const p of plans)
    for (const e of p.elements) {
      writer[e] = p.t;
      writes[e]++;
      pathOf[e] = p.path;
    }
  const warps: WarpInfo[] = [];
  for (let w = 0; w * WARP < threads; w++) {
    const ps = plans.slice(w * WARP, (w + 1) * WARP);
    const key = (p: ThreadPlan) => (cfg.style === 'grid-stride' ? String(p.iterations) : p.branch);
    warps.push({
      w,
      threads: [w * WARP, w * WARP + ps.length - 1],
      paths: [...new Set(ps.map((p) => p.path))],
      diverges: new Set(ps.map(key)).size > 1,
      loads: warpLoads(cfg, ps),
    });
  }
  return { cfg, threads, blocks: threads / cfg.block, aligned, plans, writer, writes, pathOf, warps };
}

/** How many elements the scalar tail writes, and in how many threads. */
export function tailSummary(l: Launch): { elements: number; threads: number } {
  const tail = l.plans.filter((p) => p.path === 'tail');
  return { elements: tail.reduce((s, p) => s + p.elements.length, 0), threads: tail.length };
}

/** The warps whose threads split between branches. */
export const divergentWarps = (l: Launch) => l.warps.filter((w) => w.diverges).map((w) => w.w);

/**
 * What one load instruction touches, in addresses of the underlying allocation (element +
 * offset, in floats): the lowest and highest float, how many floats in that span are read,
 * and whether the lanes read one unbroken run.
 */
export function footprint(instr: LoadInstr, offset: number): { lo: number; hi: number; read: number; contiguous: boolean } {
  const spans = instr.lanes.map((x) => [x.element + offset, x.element + offset + x.width - 1] as const).sort((a, b) => a[0] - b[0]);
  const lo = spans[0][0];
  const hi = spans[spans.length - 1][1];
  const read = spans.reduce((s, [a, b]) => s + (b - a + 1), 0);
  let contiguous = true;
  for (let i = 1; i < spans.length; i++) if (spans[i][0] !== spans[i - 1][1] + 1) contiguous = false;
  return { lo, hi, read, contiguous };
}

/** One summary line per path, for the readout under the explorer. */
export function pathCounts(l: Launch): Record<Path, { threads: number; elements: number }> {
  const out = Object.fromEntries((Object.keys(PATH_LABEL) as Path[]).map((p) => [p, { threads: 0, elements: 0 }])) as Record<
    Path,
    { threads: number; elements: number }
  >;
  for (const p of l.plans) {
    out[p.path].threads++;
    out[p.path].elements += p.elements.length;
  }
  return out;
}
