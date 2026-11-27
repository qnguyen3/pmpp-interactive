// Chapter 7, sections 7.1 and 7.2: the definition of convolution and the basic
// one-thread-per-output kernel. It runs the sliding-window explorer, the patch and
// ghost counter, the thread-to-taps stepper, the divergence explorer and every frame
// of the narrated animation "One thread per output", and pins the numbers their
// questions use.
//
// Sources for the rules (the lessons cite them where they are used):
//  - the book, section 7.1, pp. 160 to 164: y[i] = sum over j = -r..r of f[j+r]*x[i+j]
//    (no flip), the 2D form, ghost cells defaulting to 0, and the two other boundary
//    policies it names (replicate the edge, wrap around), with the worked numbers of
//    Figs. 7.1 to 7.5;
//  - the book, Exercises 1 to 6, p. 180; the closed forms are the answers worked in
//    notes/ch07.md (Exercises table);
//  - the book, section 7.2, pp. 164 to 166 and Figs. 7.6 and 7.7: outCol and outRow
//    from blockIdx, blockDim and threadIdx, the patch corner (outRow - r, outCol - r),
//    the ghost test of line 09, and the 16 x 16 toy image with 4 x 4 blocks;
//  - the book, chapter 4 section 4.4, and notes/today/ch04-facts.md A8: a block is
//    linearised x fastest and cut into consecutive runs of the warp width. The same
//    cut is used for every width given here; AMD documents the order but not the cut
//    in one sentence, and Metal calls the cut implementation-defined (ch04 facts B6,
//    C2), so nothing here describes a Metal device;
//  - notes/today/ch07-facts.md D1: "flip" is what textbook convolution does and what
//    PyTorch, cuDNN, MIOpen and MLX do not do by default.
//
// What this leaves out:
//  - Time. Nothing here knows how long anything takes; there are no caches, no memory
//    traffic and no predication.
//  - Divergence means that the lanes of one warp disagree at a branch at least once.
//    It says nothing about what that disagreement costs.
//  - The kernel modelled by basicTaps and warpDivergence is the book's Fig. 7.7 with an
//    output guard added in front of the loops (ours: the printed kernel has none), so a
//    thread whose output is outside the image does no taps at all.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- boundary policies
export type Policy = 'zero' | 'replicate' | 'wrap';
export const POLICIES: readonly Policy[] = ['zero', 'replicate', 'wrap'];
export const POLICY_LABEL: Record<Policy, string> = {
  get zero() {
    return tr('zero ghost cells', 'ô ma bằng 0');
  },
  get replicate() {
    return tr('replicate the edge', 'nhân bản giá trị biên');
  },
  get wrap() {
    return tr('wrap around', 'quay vòng');
  },
};

export interface Opts {
  /** What a missing input element is. Default: zero, the book's default (p. 162). */
  policy?: Policy;
  /** Reverse the filter: textbook convolution, not the book's (or PyTorch's) form. */
  flip?: boolean;
}

/** Radius of a filter with this many taps; filters have an odd number of taps (p. 160). */
export function radiusOf(taps: number): number {
  if (!Number.isInteger(taps) || taps < 1 || taps % 2 !== 1) throw new Error(`a filter needs an odd number of taps, not ${taps}`);
  return (taps - 1) / 2;
}

/**
 * Which real element stands in for logical index `at` of an array of `n` elements, or
 * null when the policy says it is a zero ghost cell.
 */
export function sourceIndex(at: number, n: number, policy: Policy): number | null {
  if (at >= 0 && at < n) return at;
  if (policy === 'zero') return null;
  if (policy === 'replicate') return at < 0 ? 0 : n - 1;
  return ((at % n) + n) % n;
}

// ---------------------------------------------------------------- the book's data
/** Figs. 7.1 to 7.3 (pp. 160 to 162). */
export const BOOK_1D = { x: [8, 2, 5, 4, 1, 7, 3], f: [1, 3, 5, 3, 1] } as const;
/** Exercise 2 (p. 180). */
export const EX2 = { x: [4, 1, 3, 2, 3], f: [2, 1, 4] } as const;

export interface Ex3Filter {
  key: 'a' | 'b' | 'c' | 'd' | 'e';
  f: readonly number[];
  /** As the book prints it. */
  printed: string;
}
/** Exercise 3's five filters (p. 180). */
export const EX3_FILTERS: readonly Ex3Filter[] = [
  { key: 'a', f: [0, 1, 0], printed: '[0, 1, 0]' },
  { key: 'b', f: [0, 0, 1], printed: '[0, 0, 1]' },
  { key: 'c', f: [1, 0, 0], printed: '[1, 0, 0]' },
  { key: 'd', f: [-0.5, 0, 0.5], printed: '[-1/2, 0, 1/2]' },
  { key: 'e', f: [1 / 3, 1 / 3, 1 / 3], printed: '[1/3, 1/3, 1/3]' },
];

/** An asymmetric filter of our own, for showing what a flip changes. */
export const ASYMMETRIC = [3, 2, 1] as const;

/** A row-major image (or filter). */
export interface Grid {
  width: number;
  height: number;
  data: readonly number[];
}
export const at = (g: Grid, row: number, col: number): number => g.data[row * g.width + col];

/** Fig. 7.4 (p. 163): the 7 x 7 input N and the 5 x 5 filter. */
export const BOOK_2D: { N: Grid; F: Grid } = {
  N: {
    width: 7,
    height: 7,
    data: [
      1, 2, 3, 4, 5, 6, 7,
      2, 3, 4, 5, 6, 7, 8,
      3, 4, 5, 6, 7, 8, 9,
      4, 5, 6, 7, 8, 5, 6,
      5, 6, 7, 8, 5, 6, 7,
      6, 7, 8, 9, 0, 1, 2,
      7, 8, 9, 0, 1, 2, 3,
    ],
  },
  F: {
    width: 5,
    height: 5,
    data: [
      1, 2, 3, 2, 1,
      2, 3, 4, 3, 2,
      3, 4, 5, 4, 3,
      2, 3, 4, 3, 2,
      1, 2, 3, 2, 1,
    ],
  },
};

// ---------------------------------------------------------------- 1D
export interface Tap1D {
  /** Offset from the centre, -r..r. */
  j: number;
  /** Which filter element multiplies this input: j + r, or r - j when flipped. */
  fIndex: number;
  weight: number;
  /** Logical input index i + j; may be outside the array. */
  at: number;
  ghost: boolean;
  /** The real element read, or null for a zero ghost cell. */
  src: number | null;
  value: number;
  product: number;
}
export interface Window1D {
  i: number;
  r: number;
  taps: Tap1D[];
  sum: number;
  /** Whether any tap fell outside the array, so the policy matters for this output. */
  touchesGhost: boolean;
}

export function window1d(x: readonly number[], f: readonly number[], i: number, o: Opts = {}): Window1D {
  const r = radiusOf(f.length);
  const policy = o.policy ?? 'zero';
  const taps: Tap1D[] = [];
  let sum = 0;
  for (let j = -r; j <= r; j++) {
    const fIndex = o.flip ? r - j : j + r;
    const weight = f[fIndex];
    const pos = i + j;
    const ghost = pos < 0 || pos >= x.length;
    const src = sourceIndex(pos, x.length, policy);
    const value = src == null ? 0 : x[src];
    const product = weight * value;
    sum += product;
    taps.push({ j, fIndex, weight, at: pos, ghost, src, value, product });
  }
  return { i, r, taps, sum, touchesGhost: taps.some((t) => t.ghost) };
}

/** One output per input, with ghost cells by the policy (the book's form, p. 160). */
export const conv1d = (x: readonly number[], f: readonly number[], o: Opts = {}): number[] =>
  x.map((_, i) => window1d(x, f, i, o).sum);

/** No ghost cells at all: only outputs whose window lies inside x (PyTorch's padding=0). */
export function valid1d(x: readonly number[], f: readonly number[], o: Opts = {}): number[] {
  const r = radiusOf(f.length);
  const out: number[] = [];
  for (let i = r; i < x.length - r; i++) out.push(window1d(x, f, i, o).sum);
  return out;
}

// ---------------------------------------------------------------- 2D
export interface Tap2D {
  fRow: number;
  fCol: number;
  /** Logical input position: row - ry + fRow, col - rx + fCol. */
  inRow: number;
  inCol: number;
  ghost: boolean;
  /** Flat index into N of the element read, or null for a zero ghost cell. */
  nIndex: number | null;
  /** Flat index into F of the weight used. */
  fIndex: number;
  weight: number;
  value: number;
  product: number;
}
export interface Patch2D {
  row: number;
  col: number;
  ry: number;
  rx: number;
  taps: Tap2D[];
  sum: number;
  ghostTaps: number;
}

/** The patch of output (row, col): every tap, its ghost flag, its product, and the sum. */
export function patch2d(N: Grid, F: Grid, row: number, col: number, o: Opts = {}): Patch2D {
  const ry = radiusOf(F.height);
  const rx = radiusOf(F.width);
  const policy = o.policy ?? 'zero';
  const taps: Tap2D[] = [];
  let sum = 0;
  for (let fRow = 0; fRow < F.height; fRow++) {
    for (let fCol = 0; fCol < F.width; fCol++) {
      const inRow = row - ry + fRow;
      const inCol = col - rx + fCol;
      const ghost = inRow < 0 || inRow >= N.height || inCol < 0 || inCol >= N.width;
      const sr = sourceIndex(inRow, N.height, policy);
      const sc = sourceIndex(inCol, N.width, policy);
      const nIndex = sr == null || sc == null ? null : sr * N.width + sc;
      const fr = o.flip ? F.height - 1 - fRow : fRow;
      const fc = o.flip ? F.width - 1 - fCol : fCol;
      const fIndex = fr * F.width + fc;
      const weight = F.data[fIndex];
      const value = nIndex == null ? 0 : N.data[nIndex];
      const product = weight * value;
      sum += product;
      taps.push({ fRow, fCol, inRow, inCol, ghost, nIndex, fIndex, weight, value, product });
    }
  }
  return { row, col, ry, rx, taps, sum, ghostTaps: taps.filter((t) => t.ghost).length };
}

/** Every output of a 2D convolution, row-major, one per input element. */
export function conv2d(N: Grid, F: Grid, o: Opts = {}): Grid {
  const data: number[] = [];
  for (let row = 0; row < N.height; row++)
    for (let col = 0; col < N.width; col++) data.push(patch2d(N, F, row, col, o).sum);
  return { width: N.width, height: N.height, data };
}

// ---------------------------------------------------------------- counting (Exercises 4 to 6)
export interface GhostCounts {
  /** Distinct missing elements touched by some output. */
  ghostCells: number;
  /** Multiplications if every tap is computed, ghost or not. */
  allTaps: number;
  /** Multiplications if ghost taps are skipped. */
  realTaps: number;
}

/** Brute force for an array of n elements and a filter of m taps. */
export function ghostCounts1d(n: number, m: number): GhostCounts {
  const r = radiusOf(m);
  const ghosts = new Set<number>();
  let allTaps = 0;
  let realTaps = 0;
  for (let i = 0; i < n; i++)
    for (let j = -r; j <= r; j++) {
      allTaps++;
      if (i + j < 0 || i + j >= n) ghosts.add(i + j);
      else realTaps++;
    }
  return { ghostCells: ghosts.size, allTaps, realTaps };
}

/** Exercise 4's closed forms. They hold for n >= r (and n >= 1). */
export function ghostFormula1d(n: number, m: number): GhostCounts {
  const r = radiusOf(m);
  return { ghostCells: m - 1, allTaps: n * m, realTaps: n * m - r * (r + 1) };
}

/**
 * Brute force for an n1 x n2 array (n1 rows of n2) and an m1 x m2 filter (m1 rows of m2).
 * Exercise 5 is the square case ghostCounts2d(n, n, m, m).
 */
export function ghostCounts2d(n1: number, n2: number, m1: number, m2: number): GhostCounts {
  const r1 = radiusOf(m1);
  const r2 = radiusOf(m2);
  // Every out-of-range position some output touches. They form the frame of the
  // (n1 + 2 r1) x (n2 + 2 r2) rectangle, corners included, because the corner output's
  // patch reaches its corner of the frame; count them one by one all the same.
  const ghosts = new Set<string>();
  let allTaps = 0;
  let realTaps = 0;
  for (let row = 0; row < n1; row++)
    for (let col = 0; col < n2; col++)
      for (let a = -r1; a <= r1; a++)
        for (let b = -r2; b <= r2; b++) {
          allTaps++;
          const ir = row + a;
          const ic = col + b;
          if (ir < 0 || ir >= n1 || ic < 0 || ic >= n2) ghosts.add(ir + ',' + ic);
          else realTaps++;
        }
  return { ghostCells: ghosts.size, allTaps, realTaps };
}

/** Exercises 5 and 6's closed forms. They hold for n1 >= r1 and n2 >= r2. */
export function ghostFormula2d(n1: number, n2: number, m1: number, m2: number): GhostCounts {
  const r1 = radiusOf(m1);
  const r2 = radiusOf(m2);
  return {
    ghostCells: (n1 + m1 - 1) * (n2 + m2 - 1) - n1 * n2,
    allTaps: n1 * n2 * m1 * m2,
    realTaps: (n1 * m1 - r1 * (r1 + 1)) * (n2 * m2 - r2 * (r2 + 1)),
  };
}

/** Per output: how many of its taps are ghost taps (row-major), for drawing the thin border. */
export function skippedMap(n1: number, n2: number, m1: number, m2: number): number[] {
  const r1 = radiusOf(m1);
  const r2 = radiusOf(m2);
  const inRange = (i: number, n: number, r: number) => Math.min(n - 1, i + r) - Math.max(0, i - r) + 1;
  const out: number[] = [];
  for (let row = 0; row < n1; row++)
    for (let col = 0; col < n2; col++) out.push(m1 * m2 - inRange(row, n1, r1) * inRange(col, n2, r2));
  return out;
}

// ---------------------------------------------------------------- threads (Figs. 7.6 and 7.7)
export interface ThreadPos {
  bx: number;
  by: number;
  tx: number;
  ty: number;
  /** blockDim.x and blockDim.y */
  bdx: number;
  bdy: number;
}
export interface Mapped {
  outRow: number;
  outCol: number;
  /** Upper-left corner of the input patch: (outRow - r, outCol - r). */
  cornerRow: number;
  cornerCol: number;
}

/** Lines 02 and 03 of Fig. 7.7, and the patch corner the text derives from them (p. 166). */
export function mapThread(t: ThreadPos, r: number): Mapped {
  const outCol = t.bx * t.bdx + t.tx;
  const outRow = t.by * t.bdy + t.ty;
  return { outRow, outCol, cornerRow: outRow - r, cornerCol: outCol - r };
}

/** The inverse: which block and thread own output (row, col). */
export function ownerOf(row: number, col: number, bdx: number, bdy: number): ThreadPos {
  return { bx: Math.floor(col / bdx), by: Math.floor(row / bdy), tx: col % bdx, ty: row % bdy, bdx, bdy };
}

export interface KernelTap {
  fRow: number;
  fCol: number;
  /** Lines 07 and 08: outRow - r + fRow and outCol - r + fCol. */
  inRow: number;
  inCol: number;
  /** Line 09's test: true when the tap is inside the image. */
  inside: boolean;
  /** inRow * width + inCol, or null when the tap is skipped. */
  nIndex: number | null;
  /** fRow * (2r + 1) + fCol, the corrected form of the book's F[fRow][fCol]. */
  fIndex: number;
  /** Only when data was given: the weight, the input and their product (0 when skipped). */
  weight?: number;
  value?: number;
  product?: number;
  /** Pvalue after this tap. */
  pvalue?: number;
}
export interface KernelRun {
  /** False when the output guard stops the thread before the loops (ours, not the book's). */
  inImage: boolean;
  taps: KernelTap[];
  /** Taps where line 09's test fails. */
  skipped: number;
  pvalue: number;
  /** outRow * width + outCol, the corrected form of P[outRow][outCol]; null when guarded out. */
  store: number | null;
}

/**
 * One thread of the basic kernel (Fig. 7.7, corrected, with an output guard in front),
 * tap by tap. With N and F it also accumulates Pvalue the way line 10 does.
 */
export function basicTaps(
  width: number,
  height: number,
  r: number,
  outRow: number,
  outCol: number,
  N?: readonly number[],
  F?: readonly number[],
): KernelRun {
  const inImage = outRow >= 0 && outRow < height && outCol >= 0 && outCol < width;
  if (!inImage) return { inImage, taps: [], skipped: 0, pvalue: 0, store: null };
  const m = 2 * r + 1;
  const taps: KernelTap[] = [];
  let pvalue = 0;
  for (let fRow = 0; fRow < m; fRow++)
    for (let fCol = 0; fCol < m; fCol++) {
      const inRow = outRow - r + fRow;
      const inCol = outCol - r + fCol;
      const inside = inRow >= 0 && inRow < height && inCol >= 0 && inCol < width;
      const nIndex = inside ? inRow * width + inCol : null;
      const fIndex = fRow * m + fCol;
      const tap: KernelTap = { fRow, fCol, inRow, inCol, inside, nIndex, fIndex };
      if (N && F) {
        tap.weight = F[fIndex];
        tap.value = nIndex == null ? 0 : N[nIndex];
        tap.product = nIndex == null ? 0 : tap.weight * tap.value;
        pvalue += tap.product;
        tap.pvalue = pvalue;
      }
      taps.push(tap);
    }
  return { inImage, taps, skipped: taps.filter((t) => !t.inside).length, pvalue, store: outRow * width + outCol };
}

/** Small integers for the 16 x 16 toy image of Fig. 7.6, so sums can be checked by hand. */
export function toyImage(width: number, height: number): Grid {
  const data: number[] = [];
  for (let row = 0; row < height; row++) for (let col = 0; col < width; col++) data.push(((row * 3 + col * 5) % 9) + 1);
  return { width, height, data };
}

/** A pyramid filter of radius r; for r = 2 it is the book's 5 x 5 filter of Fig. 7.4. */
export function pyramidFilter(r: number): Grid {
  const m = 2 * r + 1;
  const data: number[] = [];
  for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) data.push(r + 1 - Math.abs(a - r) + (r + 1 - Math.abs(b - r)) - 1);
  return { width: m, height: m, data };
}

// ---------------------------------------------------------------- which warps diverge
export type WarpClass = 'uniform' | 'ghost' | 'guard' | 'idle';
export const WARP_CLASS_LABEL: Record<WarpClass, string> = {
  get uniform() {
    return tr('every lane takes the same path', 'mọi lane đi cùng một đường');
  },
  get ghost() {
    return tr('lanes disagree at the ghost test', 'các lane không thống nhất tại phép kiểm tra ô ma');
  },
  get guard() {
    return tr('lanes disagree at the output guard', 'các lane không thống nhất tại điều kiện chặn đầu ra');
  },
  get idle() {
    return tr('no lane has an output', 'không lane nào có đầu ra');
  },
};

/** A run of one warp's threads that lies in one row of the grid of threads. */
export interface Segment {
  row: number;
  /** First and last output column of the run (may reach past the image). */
  c0: number;
  c1: number;
}
export interface WarpInfo {
  bx: number;
  by: number;
  /** Warp number inside its block. */
  warp: number;
  /** Threads in the warp (a last warp of a block can be partial). */
  lanes: number;
  /** Threads whose output is inside the image. */
  inImage: number;
  /** Some lanes pass the output guard and some do not. */
  guard: boolean;
  /** The lanes that pass the guard disagree at the ghost test for at least one tap. */
  ghost: boolean;
  cls: WarpClass;
  segments: Segment[];
}

/** Rows (or columns) of the image whose taps all stay inside: r .. n - 1 - r. */
const interior = (i: number, n: number, r: number) => i >= r && i <= n - 1 - r;

/** Classify one warp of block (bx, by). Threads are linearised x fastest and cut into runs of warpSize. */
export function warpInfo(
  width: number,
  height: number,
  blockX: number,
  blockY: number,
  r: number,
  warpSize: number,
  bx: number,
  by: number,
  warp: number,
): WarpInfo {
  const threads = blockX * blockY;
  const first = warp * warpSize;
  const last = Math.min(threads, first + warpSize) - 1;
  const segments: Segment[] = [];
  let inImage = 0;
  const rows: number[] = [];
  let minC = Infinity;
  let maxC = -Infinity;
  for (let ty = Math.floor(first / blockX); ty <= Math.floor(last / blockX); ty++) {
    const tx0 = ty === Math.floor(first / blockX) ? first % blockX : 0;
    const tx1 = ty === Math.floor(last / blockX) ? last % blockX : blockX - 1;
    const row = by * blockY + ty;
    const c0 = bx * blockX + tx0;
    const c1 = bx * blockX + tx1;
    segments.push({ row, c0, c1 });
    if (row < height && c0 < width) {
      const hi = Math.min(c1, width - 1);
      inImage += hi - c0 + 1;
      rows.push(row);
      minC = Math.min(minC, c0);
      maxC = Math.max(maxC, hi);
    }
  }
  const lanes = last - first + 1;
  const guard = inImage > 0 && inImage < lanes;
  // Two lanes skip the same taps exactly when they skip the same rows of the filter
  // and the same columns. Distinct rows skip the same filter rows only when both are
  // interior rows; likewise for columns.
  const rowsDiffer = rows.length > 1 && rows.some((row) => !interior(row, height, r));
  const colsDiffer = maxC > minC && (!interior(minC, width, r) || !interior(maxC, width, r));
  const ghost = inImage > 1 && (rowsDiffer || colsDiffer);
  const cls: WarpClass = guard ? 'guard' : ghost ? 'ghost' : inImage === 0 ? 'idle' : 'uniform';
  return { bx, by, warp, lanes, inImage, guard, ghost, cls, segments };
}

export interface Divergence {
  blocksX: number;
  blocksY: number;
  warpsPerBlock: number;
  warps: number;
  /** Warps whose active lanes disagree at the ghost test (including ones that also split at the guard). */
  ghost: number;
  /** Warps split by the output guard. */
  guard: number;
  /** Warps that diverge at either branch. */
  divergent: number;
  idle: number;
  uniform: number;
  /** Every warp that is not uniform, for drawing (the uniform ones are everything else). */
  marked: WarpInfo[];
}

/**
 * Every warp of the basic kernel on a width x height image with blockX x blockY blocks
 * and a filter of radius r, cut at warpSize. Blocks whose outputs all lie at least r
 * from every edge are counted without being visited: all their warps are uniform.
 */
export function warpDivergence(width: number, height: number, blockX: number, blockY: number, r: number, warpSize: number): Divergence {
  const blocksX = Math.ceil(width / blockX);
  const blocksY = Math.ceil(height / blockY);
  const warpsPerBlock = Math.ceil((blockX * blockY) / warpSize);
  const out: Divergence = { blocksX, blocksY, warpsPerBlock, warps: blocksX * blocksY * warpsPerBlock, ghost: 0, guard: 0, divergent: 0, idle: 0, uniform: 0, marked: [] };
  for (let by = 0; by < blocksY; by++) {
    const rowsInside = by * blockY >= r && (by + 1) * blockY - 1 <= height - 1 - r;
    for (let bx = 0; bx < blocksX; bx++) {
      const colsInside = bx * blockX >= r && (bx + 1) * blockX - 1 <= width - 1 - r;
      if (rowsInside && colsInside) {
        out.uniform += warpsPerBlock;
        continue;
      }
      for (let w = 0; w < warpsPerBlock; w++) {
        const info = warpInfo(width, height, blockX, blockY, r, warpSize, bx, by, w);
        if (info.ghost) out.ghost++;
        if (info.guard) out.guard++;
        if (info.ghost || info.guard) out.divergent++;
        if (info.cls === 'idle') out.idle++;
        if (info.cls === 'uniform') out.uniform++;
        else out.marked.push(info);
      }
    }
  }
  return out;
}

/** Every warp of the blocks that intersect rows [0, rows) and columns [0, cols): the close-up. */
export function warpsInCorner(
  width: number,
  height: number,
  blockX: number,
  blockY: number,
  r: number,
  warpSize: number,
  rows: number,
  cols: number,
): WarpInfo[] {
  const out: WarpInfo[] = [];
  const warpsPerBlock = Math.ceil((blockX * blockY) / warpSize);
  for (let by = 0; by * blockY < rows; by++)
    for (let bx = 0; bx * blockX < cols; bx++)
      for (let w = 0; w < warpsPerBlock; w++) out.push(warpInfo(width, height, blockX, blockY, r, warpSize, bx, by, w));
  return out;
}

/** A rectangle of threads, rows row0..row1 and columns c0..c1, all of one class. */
export interface Strip {
  cls: WarpClass;
  row0: number;
  row1: number;
  c0: number;
  c1: number;
}

/**
 * The marked warps as few rectangles: runs of the same class and the same columns in
 * consecutive rows merge into one. A whole-image edge of 32-wide warps becomes one strip.
 */
export function stripsOf(marked: readonly WarpInfo[]): Strip[] {
  const segs = marked.flatMap((w) => w.segments.map((s) => ({ cls: w.cls, row: s.row, c0: s.c0, c1: s.c1 })));
  segs.sort((a, b) => (a.cls < b.cls ? -1 : a.cls > b.cls ? 1 : a.c0 - b.c0 || a.c1 - b.c1 || a.row - b.row));
  const out: Strip[] = [];
  for (const s of segs) {
    const last = out[out.length - 1];
    if (last && last.cls === s.cls && last.c0 === s.c0 && last.c1 === s.c1 && last.row1 + 1 === s.row) last.row1 = s.row;
    else out.push({ cls: s.cls, row0: s.row, row1: s.row, c0: s.c0, c1: s.c1 });
  }
  return out;
}

// ---------------------------------------------------------------- where the steps live in the lab's source
/**
 * The text between `// @part <name>` and the next `// @part` marker (or the end), as the
 * lab files of chapter 7 mark them. The whole text when the marker is missing.
 */
export function partOf(text: string, name: string): string {
  const lines = text.split('\n');
  const start = lines.findIndex((l) => l.trim() === `// @part ${name}`);
  if (start < 0) return text.replace(/\s+$/, '');
  let end = lines.findIndex((l, k) => k > start && /^\s*\/\/ @part\s/.test(l));
  if (end < 0) end = lines.length;
  return lines.slice(start + 1, end).join('\n').replace(/\s+$/, '');
}

/** From the first line holding `__global__` (or `kernel void`) to the first unindented "}". */
export function firstKernel(text: string): string {
  const lines = text.split('\n');
  const start = lines.findIndex((l) => /__global__|^\s*kernel\s+void/.test(l));
  if (start < 0) return text;
  const end = lines.findIndex((l, k) => k > start && /^\}\s*$/.test(l));
  return lines.slice(start, end < 0 ? lines.length : end + 1).join('\n');
}

/** What each statement of the basic kernel looks like, so a stepper can mark its line. */
export const BASIC_LINES = {
  outCol: /\bint\s+outCol\s*=/,
  outRow: /\bint\s+outRow\s*=/,
  guard: /if\s*\(\s*outRow\s*>=\s*height/,
  init: /\bfloat\s+Pvalue\s*=/,
  inRow: /\bint\s+inRow\s*=/,
  inCol: /\bint\s+inCol\s*=/,
  test: /if\s*\(\s*inRow\s*>=\s*0/,
  accumulate: /Pvalue\s*\+=/,
  store: /\]\s*=\s*Pvalue\s*;/,
} as const;
export type BasicLine = keyof typeof BASIC_LINES;

/** 1-based line numbers of each statement in `code` (absent when not found exactly once). */
export function basicLineNumbers(code: string): Partial<Record<BasicLine, number>> {
  const lines = code.split('\n');
  const out: Partial<Record<BasicLine, number>> = {};
  for (const key of Object.keys(BASIC_LINES) as BasicLine[]) {
    const hits = lines.flatMap((l, k) => (BASIC_LINES[key].test(l) && !l.trim().startsWith('//') ? [k + 1] : []));
    if (hits.length === 1) out[key] = hits[0];
  }
  return out;
}

/**
 * For each lane of a block-sized group of threads walking the taps together (the scene's
 * fifth part): lanes x taps, true where the lane multiplies and false where its tap is a
 * ghost cell and it sits idle.
 */
export function laneActivity(width: number, height: number, r: number, outputs: { row: number; col: number }[]): boolean[][] {
  return outputs.map(({ row, col }) => basicTaps(width, height, r, row, col).taps.map((t) => t.inside));
}
