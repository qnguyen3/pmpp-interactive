// Chapter 4, sections 4.4 and 4.5: how the threads of a block are cut into warps,
// and which warps a boundary check splits. It runs the warp painter and the index
// stepper of section 4.4, the boundary-check counter of section 4.5, and the numbers
// in their questions.
//
// Sources for the rules (the lessons cite them where they are used):
//  - the book, section 4.4, pp. 74 to 76: a block is divided into warps of 32 threads
//    "in most implementations to date"; threads are put in row-major order (x fastest,
//    then y, then z) and cut into consecutive runs; a last warp that is not full is
//    padded with inactive threads (48 threads: 2 warps, 16 padded);
//  - notes/today/ch04-facts.md A8: NVIDIA documents the same order and the same cut
//    ("threadIdx.y has a stride of blockDim.x, and threadIdx.z has a stride of
//    blockDim.x * blockDim.y"; "each warp contains threads of consecutive, increasing
//    thread IDs with the first warp containing thread 0"). The one-line formula below
//    is that rule written out; NVIDIA does not print it in this form;
//  - the book, section 4.5, pp. 81 to 82: a warp diverges at a boundary check when
//    some of its threads pass the test and some do not; the 1003-element vector and
//    the 62 by 76 picture with its four regions;
//  - notes/today/ch04-facts.md E1: the same counts recomputed at widths 32 and 64,
//    which the tests of this file pin.
//
// What this leaves out:
//  - It applies CUDA's documented cut with whatever width you give it. AMD documents
//    the linear order and the lane arithmetic but not, in one sentence, that warps
//    are consecutive runs of that order (facts B6). Metal says the cut into
//    SIMD-groups is implementation-defined (facts C2), so nothing here describes a
//    Metal device: a kernel there has to ask which group it is in.
//  - A warp is "divergent" here when some but not all of its real threads pass the
//    test. Padded lanes are not threads and do not count either way. What a divergent
//    warp costs is the subject of divergence.ts and of lab 4.2, not of this file.
//  - The boundary counts assume one thread per element and the usual guard. Nothing
//    here knows what the kernel does after the test.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

export interface Dims {
  x: number;
  y: number;
  z: number;
}
export const dims = (x = 1, y = 1, z = 1): Dims => ({ x, y, z });

/** The warp width the book reports for CUDA devices "to date" (pp. 74, 77). */
export const BOOK_WIDTH = 32;
/** The most threads a block may have on the book's example device (p. 85). */
export const BOOK_MAX_THREADS = 1024;

export const threadsIn = (d: Dims): number => d.x * d.y * d.z;

/** Position of a thread in the block's linear order: x fastest, then y, then z. */
export const linear = (x: number, y: number, z: number, d: Dims): number => x + y * d.x + z * d.x * d.y;

/** The thread at a position of the linear order. */
export const coordOf = (i: number, d: Dims): Dims => ({
  x: i % d.x,
  y: Math.floor(i / d.x) % d.y,
  z: Math.floor(i / (d.x * d.y)),
});

export const warpOf = (index: number, width: number): number => Math.floor(index / width);
export const laneOf = (index: number, width: number): number => index % width;

export interface Place {
  index: number;
  warp: number;
  lane: number;
}
export const place = (x: number, y: number, z: number, d: Dims, width: number): Place => {
  const index = linear(x, y, z, d);
  return { index, warp: warpOf(index, width), lane: laneOf(index, width) };
};

/** The arithmetic of linear(), written out with the block's own numbers. */
export function linearWorking(x: number, y: number, z: number, d: Dims): string {
  const terms = [String(x)];
  if (d.y > 1 || d.z > 1) terms.push(`${y}·${d.x}`);
  if (d.z > 1) terms.push(`${z}·${d.x}·${d.y}`);
  return terms.length === 1 ? String(x) : `${terms.join(' + ')} = ${linear(x, y, z, d)}`;
}

export const warpsPerBlock = (d: Dims, width: number): number => Math.ceil(threadsIn(d) / width);
/** Lanes of the last warp that hold no thread. */
export const paddedLanes = (d: Dims, width: number): number => warpsPerBlock(d, width) * width - threadsIn(d);

/**
 * How many rows of the block one warp covers: a whole number when blockDim.x divides the
 * width, and the fraction 1/k when a row is exactly k warps (the width divides
 * blockDim.x). Null when warp boundaries fall inside rows in any other way.
 */
export const rowsPerWarp = (d: Dims, width: number): number | null =>
  width % d.x === 0 || d.x % width === 0 ? width / d.x : null;

export interface Cut {
  threads: number;
  width: number;
  warps: number;
  padded: number;
  rowsPerWarp: number | null;
}
export const cut = (d: Dims, width: number): Cut => ({
  threads: threadsIn(d),
  width,
  warps: warpsPerBlock(d, width),
  padded: paddedLanes(d, width),
  rowsPerWarp: rowsPerWarp(d, width),
});

/** The threads of each warp of a block, in lane order. Padded lanes are not listed. */
export function members(d: Dims, width: number): Dims[][] {
  const out: Dims[][] = Array.from({ length: warpsPerBlock(d, width) }, () => []);
  const n = threadsIn(d);
  for (let i = 0; i < n; i++) out[warpOf(i, width)].push(coordOf(i, d));
  return out;
}

/** First and last thread of one warp, and how many real threads it holds. */
export function warpSpan(d: Dims, width: number, warp: number): { first: Dims; last: Dims; count: number } {
  const lo = warp * width;
  const hi = Math.min(threadsIn(d), lo + width) - 1;
  return { first: coordOf(lo, d), last: coordOf(hi, d), count: hi - lo + 1 };
}

/** Warps in a whole launch of equal blocks. */
export const warpsInGrid = (blocks: number, d: Dims, width: number): number => blocks * warpsPerBlock(d, width);

const positiveInt = (v: unknown): number | null => {
  const n = typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN;
  return Number.isInteger(n) && n > 0 ? n : null;
};

/**
 * The warp width a CUDA or HIP device reported (warpSize in reports/runs/<backend>.json).
 * Null for every other backend: a Metal device report cannot hold a width, because the
 * width belongs to a compiled pipeline (facts C2), and the course emulators are not GPUs.
 */
export const reportedWidth = (backend: string, device: Record<string, unknown>): number | null =>
  backend === 'cuda' || backend === 'hip' ? positiveInt(device['warpSize']) : null;

/** The block-size cap a CUDA or HIP device reported, or null. */
export const reportedMaxThreads = (backend: string, device: Record<string, unknown>): number | null =>
  backend === 'cuda' || backend === 'hip' ? positiveInt(device['maxThreadsPerBlock']) : null;

// ---------------------------------------------------------------- boundary checks
/** full: every thread passes the test. idle: none does. divergent: some do and some do not. */
export type WarpState = 'full' | 'divergent' | 'idle';

export const stateOf = (passing: number, real: number): WarpState =>
  passing === 0 ? 'idle' : passing === real ? 'full' : 'divergent';

/** Every warp of one block under a per-thread test. */
export function classifyBlock(
  block: Dims,
  width: number,
  pass: (x: number, y: number, z: number) => boolean,
): { states: WarpState[]; passing: number[]; real: number[] } {
  const w = warpsPerBlock(block, width);
  const passing = new Array<number>(w).fill(0);
  const real = new Array<number>(w).fill(0);
  const n = threadsIn(block);
  for (let i = 0; i < n; i++) {
    const c = coordOf(i, block);
    const k = warpOf(i, width);
    real[k]++;
    if (pass(c.x, c.y, c.z)) passing[k]++;
  }
  return { states: passing.map((p, k) => stateOf(p, real[k])), passing, real };
}

export interface Tally {
  blocks: number;
  warpsPerBlock: number;
  warps: number;
  full: number;
  divergent: number;
  idle: number;
  /** divergent / warps */
  share: number;
}

const emptyTally = (blocks: number, perBlock: number): Tally => ({
  blocks,
  warpsPerBlock: perBlock,
  warps: blocks * perBlock,
  full: 0,
  divergent: 0,
  idle: 0,
  share: 0,
});
const close = (t: Tally): Tally => ({ ...t, share: t.warps ? t.divergent / t.warps : 0 });

/**
 * Every warp of a 2D launch under any test of the thread's column and row, counted
 * the slow, obvious way: block by block, thread by thread. `states[by][bx]` is kept
 * only when `keep` is set, for drawing small launches.
 */
export function classifyGrid(
  gridX: number,
  gridY: number,
  block: Dims,
  width: number,
  pass: (col: number, row: number) => boolean,
  keep = false,
): Tally & { states: WarpState[][][] } {
  const t = emptyTally(gridX * gridY, warpsPerBlock(block, width));
  const states: WarpState[][][] = [];
  for (let by = 0; by < gridY; by++) {
    const row: WarpState[][] = [];
    for (let bx = 0; bx < gridX; bx++) {
      const s = classifyBlock(block, width, (x, y) => pass(bx * block.x + x, by * block.y + y)).states;
      for (const k of s) t[k]++;
      if (keep) row.push(s);
    }
    if (keep) states.push(row);
  }
  return { ...close(t), states };
}

/** A picture (or matrix) with one thread per element and the guard col < width && row < height. */
export interface Picture {
  width: number;
  height: number;
  blockX: number;
  blockY: number;
}

export const gridFor = (p: Picture): { gridX: number; gridY: number } => ({
  gridX: Math.ceil(p.width / p.blockX),
  gridY: Math.ceil(p.height / p.blockY),
});

/**
 * The book's four regions (p. 82, from Fig. 3.5): blocks wholly inside the picture,
 * blocks the right edge cuts, blocks the bottom edge cuts, and the corner block.
 * Every block of a region has the same valid rectangle, so one block stands for all.
 */
export interface Region {
  id: '1' | '2' | '3' | '4';
  where: string;
  blocks: number;
  /** Valid columns and rows inside one block of this region. */
  validX: number;
  validY: number;
  /** The warps of one block. */
  states: WarpState[];
  /** Divergent, idle and full warps of the whole region. */
  divergent: number;
  idle: number;
  full: number;
  warps: number;
}

export function regions(p: Picture, width: number): Region[] {
  const { gridX, gridY } = gridFor(p);
  const fx = Math.floor(p.width / p.blockX);
  const fy = Math.floor(p.height / p.blockY);
  const remX = p.width - fx * p.blockX;
  const remY = p.height - fy * p.blockY;
  const block = dims(p.blockX, p.blockY, 1);
  const make = (id: Region['id'], where: string, blocks: number, validX: number, validY: number): Region => {
    const states = classifyBlock(block, width, (x, y) => x < validX && y < validY).states;
    const count = (s: WarpState) => blocks * states.filter((k) => k === s).length;
    return {
      id,
      where,
      blocks,
      validX,
      validY,
      states,
      divergent: count('divergent'),
      idle: count('idle'),
      full: count('full'),
      warps: blocks * states.length,
    };
  };
  return [
    make('1', tr('wholly inside', 'nằm trọn bên trong'), fx * fy, p.blockX, p.blockY),
    make('2', tr('cut by the right edge', 'bị cạnh phải cắt'), (gridX - fx) * fy, remX, p.blockY),
    make('3', tr('cut by the bottom edge', 'bị cạnh dưới cắt'), fx * (gridY - fy), p.blockX, remY),
    make('4', tr('the corner', 'góc'), (gridX - fx) * (gridY - fy), remX, remY),
  ];
}

export interface Boundary extends Tally {
  gridX: number;
  gridY: number;
  /** Threads launched. */
  threads: number;
  regions: Region[];
}

/** Warps split by the guard col < width && row < height, by regions. */
export function boundary2D(p: Picture, width: number): Boundary {
  const { gridX, gridY } = gridFor(p);
  const rs = regions(p, width);
  const perBlock = warpsPerBlock(dims(p.blockX, p.blockY, 1), width);
  const t = emptyTally(gridX * gridY, perBlock);
  for (const r of rs) {
    t.full += r.full;
    t.divergent += r.divergent;
    t.idle += r.idle;
  }
  return { ...close(t), gridX, gridY, threads: gridX * gridY * p.blockX * p.blockY, regions: rs };
}

/** A vector of n elements in blocks of `block` threads, guard i < n. */
export function boundary1D(n: number, block: number, width: number): Boundary {
  return boundary2D({ width: n, height: 1, blockX: block, blockY: 1 }, width);
}

/**
 * The one warp a 1D boundary can split: the warp that holds element n, if that warp
 * also holds a valid element. Null when n falls exactly on a warp boundary.
 */
export function splitWarp1D(n: number, block: number, width: number): { block: number; warp: number; valid: number; invalid: number } | null {
  const b = Math.floor(n / block);
  const inBlock = n - b * block;
  if (inBlock === 0) return null;
  const w = warpOf(inBlock, width);
  const valid = inBlock - w * width;
  if (valid === 0) return null;
  const real = Math.min(width, block - w * width);
  return { block: b, warp: w, valid, invalid: real - valid };
}

/**
 * Every warp of every block the right edge crosses (regions 2 and 4), whether or not
 * the warp is split. For the picture 200 wide and 150 high this shortcut gives the
 * book's printed 80 (72 + 8); the count of warps that are really split is 75.
 */
export function warpsInRightEdgeBlocks(p: Picture, width: number): number {
  const rs = regions(p, width);
  return rs[1].warps + rs[3].warps;
}

/** The test of a causal mask: a score at (row, col) is kept when col <= row. */
export const causal = (col: number, row: number): boolean => col <= row;

/** The book's boundary examples with the figure each page prints (notes pp. 81 to 82, exercise 3). */
export interface BookCase {
  id: string;
  label: string;
  page: string;
  /** What the book prints, or null where it prints no count. */
  printed: string | null;
  picture: Picture;
}
const vector = (n: number, block: number): Picture => ({ width: n, height: 1, blockX: block, blockY: 1 });
export const BOOK_CASES: readonly BookCase[] = [
  {
    id: 'v1003',
    get label() {
      return tr('vector of 1003, 64-thread blocks', 'vector 1003 phần tử, khối 64 luồng');
    },
    page: '81–82',
    printed: '1 of 32', // i18n-ignore: identifier also shown to the reader
    picture: vector(1003, 64),
  },
  {
    id: 'v100',
    get label() {
      return tr('vector of 100, 64-thread blocks', 'vector 100 phần tử, khối 64 luồng');
    },
    page: '82',
    printed: '1 of 4', // i18n-ignore: identifier also shown to the reader
    picture: vector(100, 64),
  },
  {
    id: 'v1000',
    get label() {
      return tr('vector of 1000, 64-thread blocks', 'vector 1000 phần tử, khối 64 luồng');
    },
    page: '82',
    printed: '1 of 32', // i18n-ignore: identifier also shown to the reader
    picture: vector(1000, 64),
  },
  {
    id: 'v10000',
    get label() {
      return tr('vector of 10,000, 64-thread blocks', 'vector 10,000 phần tử, khối 64 luồng');
    },
    page: '82',
    printed: '1 of 313', // i18n-ignore: identifier also shown to the reader
    picture: vector(10000, 64),
  },
  {
    id: 'ex3',
    get label() {
      return tr('exercise 3: vector of 2000, 512-thread blocks', 'bài tập 3: vector 2000 phần tử, khối 512 luồng');
    },
    page: '91',
    printed: null,
    picture: vector(2000, 512),
  },
  {
    id: 'p76x62',
    get label() {
      return tr('picture 76 wide, 62 high, 16 × 16 blocks', 'ảnh rộng 76, cao 62, khối 16 × 16');
    },
    page: '82',
    printed: '31 of 160', // i18n-ignore: identifier also shown to the reader
    picture: { width: 76, height: 62, blockX: 16, blockY: 16 },
  },
  {
    id: 'p200x150',
    get label() {
      return tr('“200 × 150” read as 200 wide, 150 high', '“200 × 150” hiểu là rộng 200, cao 150');
    },
    page: '82',
    printed: '80 of 1040', // i18n-ignore: identifier also shown to the reader
    picture: { width: 200, height: 150, blockX: 16, blockY: 16 },
  },
  {
    id: 'p150x200',
    get label() {
      return tr('“200 × 150” read as 150 wide, 200 high', '“200 × 150” hiểu là rộng 150, cao 200');
    },
    page: '82',
    printed: '80 of 1040', // i18n-ignore: identifier also shown to the reader
    picture: { width: 150, height: 200, blockX: 16, blockY: 16 },
  },
];

// ---------------------------------------------------------------- widths to choose from
/** A warp width an explorer can offer, and where the number comes from. */
export interface WidthChoice {
  key: string;
  label: string;
  width: number;
  /** The most threads a block may have with this choice. */
  maxThreads: number;
  source: 'book' | 'doc' | 'report';
  /** One sentence saying where both numbers come from. */
  note: string;
}

/**
 * The book's 32, the 64 that HIP documents for gfx9 devices (ch05 facts B5), and the
 * warpSize of every CUDA or HIP device this machine has a report for. A device that
 * does not report maxThreadsPerBlock is capped like the book's example.
 */
export function widthChoices(reports: Record<string, { device: Record<string, unknown> }>): WidthChoice[] {
  const out: WidthChoice[] = [
    {
      key: 'book',
      get label() {
        return tr("the book's 32", '32 của sách');
      },
      width: BOOK_WIDTH,
      maxThreads: BOOK_MAX_THREADS,
      source: 'book',
      get note() {
        return tr(
          "The width the book reports for CUDA devices to date (pp. 74, 77), with its example device's cap of 1024 threads in a block (p. 85).",
          'Độ rộng mà sách ghi cho các device CUDA tính đến nay (pp. 74, 77), cùng giới hạn 1024 luồng trong một khối của device ví dụ trong sách (p. 85).',
        );
      },
    },
    {
      key: 'w64',
      get label() {
        return tr('64, as HIP documents for gfx9', '64, theo tài liệu HIP cho gfx9');
      },
      width: 64,
      maxThreads: BOOK_MAX_THREADS,
      source: 'doc',
      get note() {
        return tr(
          "The width AMD's HIP documentation gives for gfx9 devices (AMD Instinct). The cap of 1024 threads in a block is the book's example, not an AMD figure.",
          'Độ rộng mà tài liệu HIP của AMD đưa ra cho các device gfx9 (AMD Instinct). Giới hạn 1024 luồng trong một khối là ví dụ của sách, không phải con số của AMD.',
        );
      },
    },
  ];
  for (const backend of Object.keys(reports).sort()) {
    const device = reports[backend].device;
    const width = reportedWidth(backend, device);
    if (width == null) continue;
    const cap = reportedMaxThreads(backend, device);
    const name = typeof device['name'] === 'string' && device['name'] ? String(device['name']) : backend;
    out.push({
      key: backend,
      get label() {
        return tr(`yours: ${name} (${backend}), ${width}`, `của bạn: ${name} (${backend}), ${width}`);
      },
      width,
      maxThreads: cap ?? BOOK_MAX_THREADS,
      source: 'report',
      get note() {
        return tr(
          `What this device reported as warpSize` +
            (cap != null ? ` and maxThreadsPerBlock (${cap}).` : `. It reported no maxThreadsPerBlock, so blocks are capped at the book's 1024.`),
          `Giá trị device này báo cáo cho warpSize` +
            (cap != null ? ` và maxThreadsPerBlock (${cap}).` : `. Nó không báo cáo maxThreadsPerBlock, nên khối bị giới hạn ở 1024 như trong sách.`),
        );
      },
    });
  }
  return out;
}
