// The thread-to-data layout of the coarsened GEMM kernel in section 15.3
// (book Figs. 15.2, 15.3 and 15.5, pp. 352–356; notes/ch15.md).
//
// What it models, as plain index arithmetic taken from the kernel:
//   - which block-level output tile a block owns (bRow, bCol, Fig. 15.3 lines 05–06);
//   - which thread-level tile a thread owns inside it (ty, tx, tRow, tCol, lines 09–13),
//     and the inverse: who owns a given element of C;
//   - what a launch with a two-dimensional block does to a kernel that reads only
//     threadIdx.x (notes, "Unclear in the PDF" item 7);
//   - loadTile's mapping of threads to tile elements (Fig. 15.5) for any tile shape and
//     thread count, with scalar loads or with one float4 per access (the book's
//     Exercise 1; the float4 layout is the notes' sketch, not the book's code);
//   - for one warp's load instruction: the global byte addresses, a count of 32-byte
//     sectors (NVIDIA's documented unit, notes/today/ch06-facts.md A2), whether each
//     lane takes the load branch or the zero-fill branch, and whether the warp diverges.
//
// What it leaves out: time, caches, the order of warps, and anything about how a real
// memory system merges requests beyond counting distinct aligned 32-byte sectors per
// instruction. Matrices are assumed to start on a 256-byte boundary (what cudaMalloc
// guarantees, infra-facts A6). The warp width is a parameter; WARP below is the book's
// assumption (32 lanes, NVIDIA), not a fact about every GPU.

import { tr } from '../lib/locale.ts';

export interface TileConfig {
  /** Block-level output tile: bM rows by bN columns. */
  bM: number;
  bN: number;
  /** Depth of the block-level input tiles (A tile bM x bK, B tile bK x bN). */
  bK: number;
  /** Thread-level output tile: tM rows by tN columns. */
  tM: number;
  tN: number;
  /** NUM_THREADS_PER_BLOCK. */
  threads: number;
}

/** The book's running example (notes, "Running example"): 128 x 128 x 8, 8 x 8 per thread, 256 threads. */
export const BOOK: TileConfig = { bM: 128, bN: 128, bK: 8, tM: 8, tN: 8, threads: 256 };

/** The book's warp width (32 threads, NVIDIA). AMD wavefronts can be 64 wide; Metal reads it from the pipeline. */
export const WARP = 32;
/** NVIDIA's unit of global-memory access (ch06-facts A2). */
export const SECTOR = 32;

const cdiv = (a: number, b: number) => Math.floor((a + b - 1) / b);

// ---------------------------------------------------------------- the grid and one thread's tile
export function gridDims(M: number, N: number, cfg: TileConfig = BOOK) {
  const x = cdiv(N, cfg.bN);
  const y = cdiv(M, cfg.bM);
  return { x, y, blocks: x * y };
}

/** Fig. 15.3 lines 05–06. */
export function blockOrigin(bx: number, by: number, cfg: TileConfig = BOOK) {
  return { bRow: by * cfg.bM, bCol: bx * cfg.bN };
}

export interface ThreadTile {
  t: number;
  tilesPerBlockX: number;
  ty: number;
  tx: number;
  tRow: number;
  tCol: number;
}

/** Fig. 15.3 lines 09–13: thread t's tile inside the block tile. */
export function threadTile(t: number, cfg: TileConfig = BOOK): ThreadTile {
  const tilesPerBlockX = cfg.bN / cfg.tN;
  const ty = Math.floor(t / tilesPerBlockX);
  const tx = t % tilesPerBlockX;
  return { t, tilesPerBlockX, ty, tx, tRow: ty * cfg.tM, tCol: tx * cfg.tN };
}

export interface Ownership extends ThreadTile {
  bx: number;
  by: number;
  bRow: number;
  bCol: number;
  /** First and last row and column of C the thread owns (they may lie outside C). */
  rows: [number, number];
  cols: [number, number];
  /** Fig. 15.3 lines 37–38: elements remaining from the tile's origin, clamped to 0. */
  maxRow: number;
  maxCol: number;
  /** Elements writeTile actually stores. */
  writes: number;
  /** Rows of A_s and columns of B_s the thread reads (its thread-level input tiles). */
  aRows: [number, number];
  bCols: [number, number];
}

export function owns(M: number, N: number, bx: number, by: number, t: number, cfg: TileConfig = BOOK): Ownership {
  const { bRow, bCol } = blockOrigin(bx, by, cfg);
  const tt = threadTile(t, cfg);
  const r0 = bRow + tt.tRow;
  const c0 = bCol + tt.tCol;
  const maxRow = r0 < M ? M - r0 : 0;
  const maxCol = c0 < N ? N - c0 : 0;
  return {
    ...tt,
    bx,
    by,
    bRow,
    bCol,
    rows: [r0, r0 + cfg.tM - 1],
    cols: [c0, c0 + cfg.tN - 1],
    maxRow,
    maxCol,
    writes: Math.min(maxRow, cfg.tM) * Math.min(maxCol, cfg.tN),
    aRows: [tt.tRow, tt.tRow + cfg.tM - 1],
    bCols: [tt.tCol, tt.tCol + cfg.tN - 1],
  };
}

/** The inverse: which block and thread own element (r, c) of C, and where it sits in the thread's tile. */
export function ownerOf(r: number, c: number, cfg: TileConfig = BOOK) {
  const bx = Math.floor(c / cfg.bN);
  const by = Math.floor(r / cfg.bM);
  const ty = Math.floor((r - by * cfg.bM) / cfg.tM);
  const tx = Math.floor((c - bx * cfg.bN) / cfg.tN);
  return { bx, by, t: ty * (cfg.bN / cfg.tN) + tx, ty, tx, row: (r - by * cfg.bM) % cfg.tM, col: (c - bx * cfg.bN) % cfg.tN };
}

// ---------------------------------------------------------------- loadTile (Fig. 15.5)
export type LoadMode = 'scalar' | 'float4';

export interface LoadPlan {
  height: number;
  width: number;
  threads: number;
  mode: LoadMode;
  /** Elements per access: 1, or 4 for a float4. */
  vec: number;
  /** Accesses that make up one row of the tile (width, or width/4). */
  perRow: number;
  rowsPerSubTile: number;
  numSubtiles: number;
  /** Where the book's arithmetic needs divisibility it does not check; empty when the shape is fine. */
  problems: string[];
}

/**
 * loadTile's plan for a height x width tile and a block of `threads` threads.
 * Scalar mode is Fig. 15.5 line for line. In float4 mode a row is width/4 accesses;
 * when a sub-tile would be taller than the tile, the threads beyond its last row do
 * nothing (the notes' caveat for Exercise 1).
 */
export function loadPlan(height: number, width: number, threads: number, mode: LoadMode = 'scalar'): LoadPlan {
  const vec = mode === 'float4' ? 4 : 1;
  const problems: string[] = [];
  if (width % vec) problems.push(tr(`the tile is ${width} wide, not a multiple of ${vec}`, `ô (tile) rộng ${width}, không phải bội số của ${vec}`));
  const perRow = Math.max(1, Math.floor(width / vec));
  if (threads % perRow) problems.push(tr(`${perRow} accesses per row do not divide ${threads} threads`, `${threads} luồng không chia hết cho ${perRow} lần truy cập mỗi hàng`));
  const rowsPerSubTile = Math.max(1, Math.floor(threads / perRow));
  const numSubtiles = mode === 'scalar' ? Math.floor(height / rowsPerSubTile) : cdiv(height, rowsPerSubTile);
  if (mode === 'scalar' && height % rowsPerSubTile) problems.push(tr(`${rowsPerSubTile} rows per sub-tile do not divide ${height} rows`, `${height} hàng không chia hết cho ${rowsPerSubTile} hàng mỗi ô con`));
  return { height, width, threads, mode, vec, perRow, rowsPerSubTile, numSubtiles, problems };
}

/** The tile position thread t fills in sub-tile `sub` (Fig. 15.5 lines 08–09), or null if it has none. */
export function slotOf(plan: LoadPlan, t: number, sub: number): { row: number; col: number } | null {
  const row = sub * plan.rowsPerSubTile + Math.floor(t / plan.perRow);
  const col = (t % plan.perRow) * plan.vec;
  return row < plan.height ? { row, col } : null;
}

/** For every element of the tile: which thread and which sub-tile fill it (-1 if none). */
export function loaderMap(plan: LoadPlan): { thread: number[]; sub: number[]; times: number[] } {
  const n = plan.height * plan.width;
  const thread = new Array<number>(n).fill(-1);
  const sub = new Array<number>(n).fill(-1);
  const times = new Array<number>(n).fill(0);
  for (let s = 0; s < plan.numSubtiles; s++)
    for (let t = 0; t < plan.threads; t++) {
      const at = slotOf(plan, t, s);
      if (!at) continue;
      for (let j = 0; j < plan.vec; j++) {
        const col = at.col + j;
        if (col >= plan.width) continue;
        const i = at.row * plan.width + col;
        thread[i] = t;
        sub[i] = s;
        times[i] += 1;
      }
    }
  return { thread, sub, times };
}

/** Load instructions (scalar or float4) each thread issues for one A tile and one B tile together. */
export function loadsPerThreadPerTilePair(cfg: TileConfig = BOOK, mode: LoadMode = 'scalar'): number {
  const a = loadPlan(cfg.bM, cfg.bK, cfg.threads, mode);
  const b = loadPlan(cfg.bK, cfg.bN, cfg.threads, mode);
  const count = (p: LoadPlan) => {
    let n = 0;
    for (let s = 0; s < p.numSubtiles; s++) if (slotOf(p, 0, s)) n++;
    return n;
  };
  return count(a) + count(b);
}

// ---------------------------------------------------------------- one warp's load instruction
export interface Problem {
  M: number;
  N: number;
  K: number;
}

export type Branch = 'load' | 'zero' | 'partial';

export interface LaneAccess {
  lane: number;
  t: number;
  /** Position in the tile. */
  row: number;
  col: number;
  /** Position in the source matrix (A or B). */
  gRow: number;
  gCol: number;
  /** Byte offset of the first element from the matrix's start. */
  byte: number;
  /** Elements this lane is responsible for (1, or 4 in float4 mode). */
  vec: number;
  /** Which elements are inside the matrix: loaded; the rest are zero-filled. */
  inside: number;
  /**
   * Which side of loadTile's if the lane takes: 'load' (all its elements inside the
   * matrix), 'zero' (none: it writes 0.0f) or, in float4 mode only, 'partial' (the
   * scalar fallback for a vector that crosses the edge).
   */
  branch: Branch;
  /** float4 mode: the address is not a multiple of 16 bytes. */
  misaligned: boolean;
}

export interface WarpLoad {
  which: 'A' | 'B';
  sub: number;
  warp: number;
  lanes: LaneAccess[];
  /** Distinct 32-byte sectors the loading lanes touch, ascending (sector index = byte ÷ 32). */
  sectors: number[];
  /** Contiguous byte runs the loading lanes read, as [start, end). */
  runs: [number, number][];
  /** Lanes that take the zero-fill branch (or, in float4 mode, the partial fallback). */
  zeroLanes: number;
  /** Lanes on both sides of the branch. */
  diverges: boolean;
}

/** A matrix of `cols` columns, row-major, starting at byte 0: the byte offset of (r, c). */
const byteOf = (r: number, c: number, cols: number) => (r * cols + c) * 4;

/** The tile pairs the kernel loops over: ceil(K / bK) (Fig. 15.3 line 20). */
export const tilePairs = (K: number, cfg: TileConfig = BOOK) => cdiv(K, cfg.bK);

/** The A or B tile's plan and its window in the source matrix for block (bx, by), tile pair `tile`. */
export function tileWindow(which: 'A' | 'B', p: Problem, bx: number, by: number, tile: number, cfg: TileConfig = BOOK, mode: LoadMode = 'scalar') {
  const { bRow, bCol } = blockOrigin(bx, by, cfg);
  if (which === 'A') {
    // loadTile(&A[bRow*K + tile*bK], K, M - bRow, K - tile*bK, A_s, bK, bM, bK)
    return { plan: loadPlan(cfg.bM, cfg.bK, cfg.threads, mode), row0: bRow, col0: tile * cfg.bK, ld: p.K, maxRow: p.M - bRow, maxCol: p.K - tile * cfg.bK };
  }
  // loadTile(&B[tile*bK*N + bCol], N, K - tile*bK, N - bCol, B_s, bN, bK, bN)
  return { plan: loadPlan(cfg.bK, cfg.bN, cfg.threads, mode), row0: tile * cfg.bK, col0: bCol, ld: p.N, maxRow: p.K - tile * cfg.bK, maxCol: p.N - bCol };
}

/** Merge sorted-or-not byte intervals into contiguous runs. */
function mergeRuns(iv: [number, number][]): [number, number][] {
  const s = [...iv].sort((a, b) => a[0] - b[0]);
  const out: [number, number][] = [];
  for (const [lo, hi] of s) {
    const last = out[out.length - 1];
    if (last && lo <= last[1]) last[1] = Math.max(last[1], hi);
    else out.push([lo, hi]);
  }
  return out;
}

/** Distinct aligned `sectorBytes` sectors covered by byte intervals [start, end). */
export function sectorsOf(iv: [number, number][], sectorBytes = SECTOR): number[] {
  const set = new Set<number>();
  for (const [lo, hi] of iv) for (let s = Math.floor(lo / sectorBytes); s <= Math.floor((hi - 1) / sectorBytes); s++) set.add(s);
  return [...set].sort((a, b) => a - b);
}

/** One warp's load instruction for sub-tile `sub` of the A or B tile. */
export function warpLoad(
  which: 'A' | 'B',
  p: Problem,
  bx: number,
  by: number,
  tile: number,
  sub: number,
  warp: number,
  cfg: TileConfig = BOOK,
  mode: LoadMode = 'scalar',
  warpWidth = WARP,
): WarpLoad {
  const w = tileWindow(which, p, bx, by, tile, cfg, mode);
  const lanes: LaneAccess[] = [];
  const iv: [number, number][] = [];
  for (let lane = 0; lane < warpWidth; lane++) {
    const t = warp * warpWidth + lane;
    if (t >= cfg.threads) break;
    const at = slotOf(w.plan, t, sub);
    if (!at) continue;
    const vec = w.plan.vec;
    let inside = 0;
    for (let j = 0; j < vec; j++) if (at.row < w.maxRow && at.col + j < w.maxCol) inside++;
    const gRow = w.row0 + at.row;
    const gCol = w.col0 + at.col;
    const byte = byteOf(gRow, gCol, w.ld);
    const branch: Branch = inside === vec ? 'load' : inside === 0 ? 'zero' : 'partial';
    lanes.push({ lane, t, row: at.row, col: at.col, gRow, gCol, byte, vec, inside, branch, misaligned: vec > 1 && byte % 16 !== 0 });
    if (branch === 'load') iv.push([byte, byte + 4 * vec]);
    else if (branch === 'partial') for (let j = 0; j < inside; j++) iv.push([byte + 4 * j, byte + 4 * j + 4]);
  }
  const kinds = new Set(lanes.map((l) => l.branch));
  return {
    which,
    sub,
    warp,
    lanes,
    sectors: sectorsOf(iv),
    runs: mergeRuns(iv),
    zeroLanes: lanes.filter((l) => l.branch !== 'load').length,
    diverges: kinds.size > 1,
  };
}

/** Every warp instruction of one tile's load: sectors in total and how many warps diverge at the bounds check. */
export function tileLoadSummary(
  which: 'A' | 'B',
  p: Problem,
  bx: number,
  by: number,
  tile: number,
  cfg: TileConfig = BOOK,
  mode: LoadMode = 'scalar',
  warpWidth = WARP,
) {
  const plan = tileWindow(which, p, bx, by, tile, cfg, mode).plan;
  const warps = Math.ceil(cfg.threads / warpWidth);
  let instructions = 0;
  let sectors = 0;
  let divergent = 0;
  let zeroFilled = 0;
  for (let s = 0; s < plan.numSubtiles; s++)
    for (let w = 0; w < warps; w++) {
      const wl = warpLoad(which, p, bx, by, tile, s, w, cfg, mode, warpWidth);
      if (!wl.lanes.length) continue;
      instructions++;
      sectors += wl.sectors.length;
      if (wl.diverges) divergent++;
      zeroFilled += wl.lanes.reduce((n, l) => n + l.vec - l.inside, 0);
    }
  return { instructions, sectors, divergent, zeroFilled };
}

/** Elements of the last A tile (columns of A beyond K) that loadTile fills with zeros, per row. */
export const lastTileZeroColumns = (K: number, cfg: TileConfig = BOOK) => tilePairs(K, cfg) * cfg.bK - K;

// ---------------------------------------------------------------- a two-dimensional block by mistake
/**
 * The kernel reads only threadIdx.x. Launched with a dimX x dimY block, thread (x, y)
 * behaves as thread x of the intended one-dimensional block, whatever y is: dimY copies
 * of the first dimX threads, and nobody playing threads dimX to 255. NUM_THREADS_PER_BLOCK
 * stays the compile-time 256, so loadTile's arithmetic is unchanged.
 */
export function launch2D(dimX: number, dimY: number, cfg: TileConfig = BOOK) {
  const patches = (cfg.bM / cfg.tM) * (cfg.bN / cfg.tN);
  const writers = new Array<number>(patches).fill(0);
  const seen = new Set<number>();
  for (let y = 0; y < dimY; y++)
    for (let x = 0; x < dimX; x++) {
      const t = x; // threadIdx.y is never read
      seen.add(t);
      if (t < patches) writers[t] += 1;
    }
  const a = loadPlan(cfg.bM, cfg.bK, cfg.threads);
  const b = loadPlan(cfg.bK, cfg.bN, cfg.threads);
  const filled = (plan: LoadPlan) => {
    const got = new Array<boolean>(plan.height * plan.width).fill(false);
    for (const t of seen) for (let s = 0; s < plan.numSubtiles; s++) {
      const at = slotOf(plan, t, s);
      if (at) got[at.row * plan.width + at.col] = true;
    }
    return got;
  };
  const aLoaded = filled(a);
  const bLoaded = filled(b);
  const perRow = cfg.bN / cfg.tN;
  const rowsWritten = new Set<number>();
  writers.forEach((n, p) => {
    if (n) for (let r = 0; r < cfg.tM; r++) rowsWritten.add(Math.floor(p / perRow) * cfg.tM + r);
  });
  // A patch is right only if every A_s row and B_s column it reads was loaded.
  let correctPatches = 0;
  writers.forEach((n, p) => {
    if (!n) return;
    const tt = threadTile(p, cfg);
    let ok = true;
    for (let r = tt.tRow; r < tt.tRow + cfg.tM && ok; r++) for (let k = 0; k < cfg.bK; k++) if (!aLoaded[r * cfg.bK + k]) ok = false;
    for (let c = tt.tCol; c < tt.tCol + cfg.tN && ok; c++) for (let k = 0; k < cfg.bK; k++) if (!bLoaded[k * cfg.bN + c]) ok = false;
    if (ok) correctPatches++;
  });
  return {
    threads: dimX * dimY,
    distinctThreads: seen.size,
    writers,
    patchesWritten: writers.filter((n) => n > 0).length,
    rowsNeverWritten: cfg.bM - rowsWritten.size,
    aLoaded,
    bLoaded,
    aLoadedCount: aLoaded.filter(Boolean).length,
    bLoadedCount: bLoaded.filter(Boolean).length,
    correctPatches,
  };
}
