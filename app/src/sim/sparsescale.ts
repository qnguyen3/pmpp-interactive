// Chapter 17, pages "Before you start" and 17.5: what padding, spilling and sorting cost
// on rows of uneven length. It runs the explorer "Pad, spill or sort" on page 17.5 and
// the byte arithmetic of the start page's hook and questions.
//
// Sources for the rules (the lessons cite them where they are used):
//  - the book, §17.3 to §17.6, pp. 407–418: CSR is two non-zero-length arrays plus
//    numRows + 1 row pointers; ELL pads every row to the longest and stores column-major
//    (Fig. 17.10, i = t*numRows + row); hybrid ELL-COO moves the tail of long rows into
//    COO (Fig. 17.13); JDS sorts rows longest first, stores column-major with no padding
//    and keeps a row array and iterPtr (Fig. 17.14); the sectioned variant pads each
//    section only to its own longest row (p. 417);
//  - the book's worked example, pp. 412–413, and our arithmetic on it (ch17-facts E1):
//    CSR 2·10,000 + 1,001 = 21,001 numbers, ELL 2·1,000·200 = 400,000, ratio 19.05;
//  - exercise 2, p. 422, and the notes' answer sketch: JDS stores 2z + m + K + 1;
//  - Sliced ELL as NVIDIA documents it (ch17-facts A6): slices of consecutive rows, each
//    padded to its own longest row, column-major per slice, nslices + 1 slice offsets;
//  - one thread per row for every per-row kernel, so a thread's trip count is its row's
//    length (Figs. 17.9 and 17.12), counted with divergence.ts loopBatch (chapter 4);
//  - the segment rule of coalesce.ts warpAccess (chapter 6; ch06-facts A2, B1, C1).
//
// Choices that are ours, stated where they matter:
//  - "numbers" counts values and indices as the book does (p. 412). The book's ELL count
//    leaves out nnzPerRow; so does this one (the libraries mark pads with -1 instead,
//    ch17-facts A6, B3).
//  - Bytes: values and column (and COO row) indices take the chosen widths; every array
//    of pointers (rowPtrs, iterPtr, the JDS row array, slice offsets) takes 4 bytes.
//  - The book's example says the other rows have "fewer than 10" non-zeros while the
//    whole matrix is 1% non-zero. Both cannot hold: 999 rows of at most 9 plus one of 200
//    is at most 9,191 non-zeros, below 10,000. The preset keeps the 1% (one row of 200,
//    the others 9 or 10), which changes nothing visible in the ELL count.
//  - Row lengths are generated, deterministically, never read from real matrices. The
//    power-law shape is ours (a Pareto distribution with exponent 1.5, sampled at evenly
//    spaced quantiles).
//
// What it leaves out: time; caches between "every request counted" and "each segment
// once"; the traffic of x, y and the small per-row arrays; sorting and conversion cost.

import { loopBatch, shuffledTrips, type LoopBatch } from './divergence';
import { warpAccess } from './coalesce';
import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- row-length generators
export type Shape = 'uniform' | 'spread' | 'long' | 'power' | 'fig1713';

export interface ShapeParams {
  shape: Shape;
  /** Rows; the matrix is square except for Fig. 17.13, which is 8 × 8 anyway. */
  rows: number;
  /** Mean non-zeros per row: the total is rows × mean (approximately for 'power'). */
  mean: number;
  /** 'long' only: how many rows are long, and how long. */
  longCount: number;
  longLen: number;
  seed?: number;
}

/** Fig. 17.13's row lengths, read from the figure (notes, Figures; ch17-facts E1). */
export const FIG1713_LENGTHS: readonly number[] = [1, 5, 2, 2, 1, 2, 4, 1];

/** A small deterministic generator (the same LCG as divergence.ts), so presets agree everywhere. */
function rng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}
function shuffle<T>(xs: T[], seed: number): T[] {
  const next = rng(seed);
  for (let i = xs.length - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1));
    [xs[i], xs[j]] = [xs[j], xs[i]];
  }
  return xs;
}

/** Columns of the matrix a shape describes: square, as in the book's example. */
export const colsOf = (p: ShapeParams): number => (p.shape === 'fig1713' ? FIG1713_LENGTHS.length : p.rows);

/** Where the long rows sit: evenly spaced, so "one row in 512" means exactly that. */
export const longRowPositions = (rows: number, count: number): number[] =>
  Array.from({ length: Math.max(0, Math.min(count, rows)) }, (_, i) => Math.floor(((i + 0.5) * rows) / count));

/** The power-law exponent of the shape we add (ours). */
export const POWER_EXPONENT = 1.5;

/**
 * Non-zeros per row, one entry per row, in row order.
 *  - uniform: every row has `mean`.
 *  - spread: lengths spread evenly from 1 to 2·mean − 1 (so the mean is `mean`), shuffled.
 *  - long: `longCount` rows of `longLen` at evenly spaced positions; the other rows share
 *    what is left of rows × mean as evenly as integers allow (the book's construction).
 *  - power: a Pareto shape (ours), mean close to `mean`, capped at the column count.
 *  - fig1713: the eight rows of Fig. 17.13.
 */
export function rowLengths(p: ShapeParams): number[] {
  const seed = p.seed ?? 17;
  const rows = Math.max(1, Math.floor(p.rows));
  const cols = colsOf(p);
  const mean = Math.max(0, Math.floor(p.mean));
  switch (p.shape) {
    case 'fig1713':
      return [...FIG1713_LENGTHS];
    case 'uniform':
      return Array.from({ length: rows }, () => Math.min(cols, mean));
    case 'spread':
      return mean < 1 ? Array(rows).fill(0) : shuffledTrips(rows, 1, Math.min(cols, 2 * mean - 1), seed);
    case 'long': {
      const longLen = Math.max(0, Math.min(cols, Math.floor(p.longLen)));
      const at = longRowPositions(rows, Math.floor(p.longCount));
      const isLong = new Set(at);
      const others = rows - isLong.size;
      const rest = Math.max(0, rows * mean - isLong.size * longLen);
      const base = others ? Math.floor(rest / others) : 0;
      let extra = others ? rest - base * others : 0;
      const otherRows = shuffle(
        Array.from({ length: rows }, (_, r) => r).filter((r) => !isLong.has(r)),
        seed,
      );
      const out = Array(rows).fill(0);
      for (const r of at) out[r] = longLen;
      for (const r of otherRows) out[r] = Math.min(cols, base + (extra-- > 0 ? 1 : 0));
      return out;
    }
    case 'power': {
      const a = POWER_EXPONENT;
      const xm = (mean * (a - 1)) / a; // a Pareto(xm, a) has mean a·xm/(a − 1)
      const out = Array.from({ length: rows }, (_, i) => {
        const q = (i + 0.5) / rows;
        return Math.max(1, Math.min(cols, Math.round(xm * Math.pow(q, -1 / a))));
      });
      return shuffle(out, seed);
    }
  }
}

export interface Preset {
  id: 'book' | 'fig1713' | 'uniform' | 'spread' | 'skewed' | 'power';
  label: string;
  params: ShapeParams;
  /** One sentence on where the preset comes from. */
  note: string;
}

export const PRESETS: readonly Preset[] = [
  {
    id: 'book',
    get label() {
      return tr("The book's 1000 × 1000", 'Ví dụ 1000 × 1000 của sách');
    },
    params: { shape: 'long', rows: 1000, mean: 10, longCount: 1, longLen: 200 },
    get note() {
      return tr("The book's worked example (pp. 412–413): 1% non-zero, one row of 200. The others are 9 or 10, so that the total stays 1%; the book's \"fewer than 10\" cannot hold together with 1%.", 'Ví dụ mẫu của sách (pp. 412–413): 1% khác 0, một hàng dài 200. Các hàng khác dài 9 hoặc 10, để tổng vẫn là 1%; cụm "fewer than 10" (ít hơn 10) của sách không thể khớp với 1%.');
    },
  },
  {
    id: 'fig1713',
    get label() {
      return tr('Fig. 17.13', 'Hình 17.13');
    },
    params: { shape: 'fig1713', rows: 8, mean: 2, longCount: 0, longLen: 0 },
    get note() {
      return tr("The eight rows of Fig. 17.13 (p. 415): lengths 1, 5, 2, 2, 1, 2, 4, 1, 18 non-zeros.", 'Tám hàng của Hình 17.13 (p. 415): độ dài 1, 5, 2, 2, 1, 2, 4, 1, 18 phần tử khác 0.');
    },
  },
  {
    id: 'uniform',
    get label() {
      return tr('Uniform, 16 per row', 'Đều, 16 mỗi hàng');
    },
    params: { shape: 'uniform', rows: 1024, mean: 16, longCount: 0, longLen: 0 },
    get note() {
      return tr('Every row has the same length: the case ELL was made for.', 'Mọi hàng có cùng độ dài: trường hợp mà ELL được tạo ra để xử lý.');
    },
  },
  {
    id: 'spread',
    get label() {
      return tr('Spread, 1 to 31', 'Trải, 1 đến 31');
    },
    params: { shape: 'spread', rows: 1024, mean: 16, longCount: 0, longLen: 0 },
    get note() {
      return tr('Lengths spread evenly from 1 to 31, in a shuffled order.', 'Độ dài trải đều từ 1 đến 31, theo thứ tự xáo trộn.');
    },
  },
  {
    id: 'skewed',
    get label() {
      return tr('Skewed: 1 row in 512 is 16× longer', 'Lệch: cứ 512 hàng có 1 hàng dài gấp 16 lần');
    },
    params: { shape: 'long', rows: 4096, mean: 16, longCount: 8, longLen: 256 },
    get note() {
      return tr('Eight rows of 256 in 4096; the others share the rest of 16 per row on average.', 'Tám hàng dài 256 trong 4096 hàng; các hàng còn lại chia phần còn lại để trung bình là 16 mỗi hàng.');
    },
  },
  {
    id: 'power',
    get label() {
      return tr('Power law (ours)', 'Luật lũy thừa (của khóa học này)');
    },
    params: { shape: 'power', rows: 4096, mean: 16, longCount: 0, longLen: 0 },
    get note() {
      return tr('Ours, not the book\'s: a Pareto shape with exponent 1.5, a few very long rows and many short ones.', 'Của khóa học này, không phải của sách: dạng Pareto với số mũ 1.5, vài hàng rất dài và nhiều hàng ngắn.');
    },
  },
];
export const preset = (id: Preset['id']): Preset => PRESETS.find((p) => p.id === id)!;

// ---------------------------------------------------------------- bytes per weight position
/**
 * Bytes of the matrix arrays a CSR product reads per position of the dense matrix: the
 * kept fraction pays a value and a column index, and each row pays one 4-byte row pointer
 * spread over its `cols` positions.
 */
export const bytesPerPosition = (density: number, valueBytes: number, indexBytes: number, cols: number, ptrBytes = 4): number =>
  density * (valueBytes + indexBytes) + (Number.isFinite(cols) && cols > 0 ? ptrBytes / cols : 0);

/** The density below which CSR reads fewer matrix bytes than a dense matrix of `denseBytes` per element. */
export const breakEvenDensity = (denseBytes: number, valueBytes: number, indexBytes: number, cols = Infinity, ptrBytes = 4): number =>
  Math.max(0, (denseBytes - (Number.isFinite(cols) && cols > 0 ? ptrBytes / cols : 0)) / (valueBytes + indexBytes));

// ---------------------------------------------------------------- storage
export interface Widths {
  /** Bytes per value: 4, 2, 1 or 0.5 (4-bit). */
  value: number;
  /** Bytes per column (and COO row) index: 4 or 2. */
  index: number;
}
/** Every array of pointers or row numbers is stored as 4-byte integers. */
export const PTR_BYTES = 4;

export type FormatId = 'dense' | 'csr' | 'ell' | 'hybrid' | 'jds' | 'sell';

export interface Store {
  id: FormatId;
  /** Values and indices stored, counted as the book counts (p. 412). */
  numbers: number;
  bytes: number;
  /** Padding slots (each costs a value and an index). */
  pads: number;
  /** Entries moved to the COO part (hybrid only). */
  coo: number;
  /** Rows of padded storage per slice, or the padded width, where it applies. */
  width?: number;
}

const sum = (xs: readonly number[]) => xs.reduce((a, b) => a + b, 0);
export const nnzOf = sum;
export const maxOf = (xs: readonly number[]): number => xs.reduce((a, b) => (b > a ? b : a), 0);

export function dense(rows: number, cols: number, w: Widths): Store {
  return { id: 'dense', numbers: rows * cols, bytes: rows * cols * w.value, pads: 0, coo: 0 };
}
export function csr(len: readonly number[], w: Widths): Store {
  const nnz = sum(len);
  const rows = len.length;
  return { id: 'csr', numbers: 2 * nnz + rows + 1, bytes: nnz * (w.value + w.index) + (rows + 1) * PTR_BYTES, pads: 0, coo: 0 };
}
export function ell(len: readonly number[], w: Widths): Store {
  const width = maxOf(len);
  const slots = len.length * width;
  return { id: 'ell', numbers: 2 * slots, bytes: slots * (w.value + w.index), pads: slots - sum(len), coo: 0, width };
}
/** Hybrid ELL-COO with ELL width K: each row keeps its first K non-zeros, the rest go to COO. */
export function hybrid(len: readonly number[], K: number, w: Widths): Store {
  const k = Math.max(0, Math.floor(K));
  const slots = len.length * k;
  const coo = sum(len.map((l) => Math.max(0, l - k)));
  const pads = sum(len.map((l) => Math.max(0, k - l)));
  return { id: 'hybrid', numbers: 2 * slots + 3 * coo, bytes: slots * (w.value + w.index) + coo * (w.value + 2 * w.index), pads, coo, width: k };
}
/** How many entries each row sends to COO at width K. */
export const spills = (len: readonly number[], K: number): number[] => len.map((l) => Math.max(0, l - Math.floor(K)));

/** The JDS row array: rows sorted longest first, ties in row order (Fig. 17.14). */
export function sortOrder(len: readonly number[]): number[] {
  return len.map((_, r) => r).sort((a, b) => len[b] - len[a] || a - b);
}
export const sortedLengths = (len: readonly number[]): number[] => sortOrder(len).map((r) => len[r]);

/** iterPtr of JDS: where iteration t's elements start, with the end appended (Fig. 17.14). */
export function iterPtr(len: readonly number[]): number[] {
  const width = maxOf(len);
  const ofLength = Array(width + 1).fill(0);
  for (const l of len) ofLength[l]++;
  // rows that still have a t-th element: those longer than t
  const longer = Array(width + 1).fill(0);
  for (let t = width - 1; t >= 0; t--) longer[t] = longer[t + 1] + ofLength[t + 1];
  const out = [0];
  for (let t = 0; t < width; t++) out.push(out[t] + longer[t]);
  return out;
}
/** Rows that still have a t-th element, for each iteration t: iterPtr[t+1] − iterPtr[t]. */
export const rowsInIteration = (len: readonly number[]): number[] => {
  const ip = iterPtr(len);
  return ip.slice(1).map((v, t) => v - ip[t]);
};

export function jds(len: readonly number[], w: Widths): Store {
  const nnz = sum(len);
  const width = maxOf(len);
  const ptrs = len.length + width + 1; // the row array and iterPtr
  return { id: 'jds', numbers: 2 * nnz + ptrs, bytes: nnz * (w.value + w.index) + ptrs * PTR_BYTES, pads: 0, coo: 0, width };
}

/** Lengths cut into slices of S consecutive rows (the last may be shorter). */
export function slices(len: readonly number[], S: number): number[][] {
  const s = Math.max(1, Math.floor(S));
  const out: number[][] = [];
  for (let i = 0; i < len.length; i += s) out.push(len.slice(i, i + s));
  return out;
}
/** Sliced ELL: each slice of S rows padded to its own longest row; sorted first if asked (the book's sectioned JDS). */
export function sell(len: readonly number[], S: number, sorted: boolean, w: Widths): Store {
  const l = sorted ? sortedLengths(len) : [...len];
  const parts = slices(l, S);
  const slots = sum(parts.map((p) => p.length * maxOf(p)));
  const ptrs = parts.length + 1 + (sorted ? len.length : 0); // slice offsets, and the row array if sorted
  return { id: 'sell', numbers: 2 * slots + ptrs, bytes: slots * (w.value + w.index) + ptrs * PTR_BYTES, pads: slots - sum(len), coo: 0 };
}

/** Slice sizes the explorer offers: powers of two below the row count, then the row count (plain ELL). */
export function sliceChoices(rows: number): number[] {
  const out: number[] = [];
  for (let s = 1; s < rows; s *= 2) out.push(s);
  out.push(rows);
  return out;
}

export interface StorageTable {
  rows: number;
  cols: number;
  nnz: number;
  stores: Record<FormatId, Store>;
}
export function storage(len: readonly number[], cols: number, w: Widths, K: number, S: number, sorted: boolean): StorageTable {
  return {
    rows: len.length,
    cols,
    nnz: sum(len),
    stores: {
      dense: dense(len.length, cols, w),
      csr: csr(len, w),
      ell: ell(len, w),
      hybrid: hybrid(len, K, w),
      jds: jds(len, w),
      sell: sell(len, S, sorted, w),
    },
  };
}

// ---------------------------------------------------------------- warps of a per-row kernel
export interface WarpStats {
  batch: LoopBatch;
  /** Warp-iterations: the sum over warps of each warp's longest trip. */
  spent: number;
  /** Lane-iterations that do work: the non-zeros. */
  useful: number;
  /**
   * useful / (width × spent). A partial last warp counts its missing lanes as idle: the
   * warp occupies its full width either way. (loopBatch's own efficiency leaves them out.)
   */
  efficiency: number;
  /** The longest trip any thread makes. */
  longest: number;
}
export function warpStats(trips: readonly number[], width: number): WarpStats {
  const batch = loopBatch(trips, width);
  return {
    batch,
    spent: batch.spent,
    useful: batch.useful,
    efficiency: batch.spent ? batch.uniform / batch.spent : 1,
    longest: maxOf(trips),
  };
}
export interface WarpTable {
  /** One thread per row in row order: CSR's and ELL's kernels (Figs. 17.9, 17.12). */
  original: WarpStats;
  /** One thread per row, longest first: JDS (Fig. 17.15). */
  sorted: WarpStats;
  /** The ELL part of the hybrid at width K, in row order; the COO part has no loop. */
  hybrid: WarpStats;
}
export function warps(len: readonly number[], width: number, K: number): WarpTable {
  return {
    original: warpStats(len, width),
    sorted: warpStats(sortedLengths(len), width),
    hybrid: warpStats(
      len.map((l) => Math.min(l, Math.max(0, Math.floor(K)))),
      width,
    ),
  };
}

export type LaneState = 'work' | 'done' | 'absent';
/** What each lane of warp `warp` does in iteration t: works, has finished, or has no row. */
export function laneStates(trips: readonly number[], width: number, warp: number, t: number): LaneState[] {
  return Array.from({ length: width }, (_, lane) => {
    const r = warp * width + lane;
    if (r >= trips.length) return 'absent';
    return trips[r] > t ? 'work' : 'done';
  });
}

// ---------------------------------------------------------------- requested segments
export type TrafficFormat = 'csr' | 'ell' | 'jds';
export interface Traffic {
  format: TrafficFormat;
  /** Segments requested, summed over every warp's every iteration, values and indices: no cache. */
  requested: number;
  /** Distinct segments touched over the whole kernel: what a perfect cache would leave. */
  distinct: number;
  /** Bytes of values and indices the threads actually use. */
  needed: number;
  /** needed / (requested × segment size). */
  efficiency: number;
}

/**
 * Element index each active thread reads in iteration t, per format. Every format's
 * value and index arrays are laid out the same way, so one index serves both arrays.
 */
function elementIndex(format: TrafficFormat, len: readonly number[]): (row: number, t: number) => number {
  if (format === 'ell') {
    const rows = len.length;
    return (row, t) => t * rows + row; // Fig. 17.12: i = t*numRows + row
  }
  if (format === 'jds') {
    const ip = iterPtr(len); // `len` is already sorted for JDS
    return (row, t) => ip[t] + row;
  }
  const ptr = [0];
  for (const l of len) ptr.push(ptr[ptr.length - 1] + l);
  return (row, t) => ptr[row] + t; // Fig. 17.9: i from rowPtrs[row]
}

/** Byte address and access size of element i in an array of `bytes`-wide elements (4-bit values read their whole byte). */
const addr = (i: number, bytes: number): [number, number] => (bytes >= 1 ? [i * bytes, bytes] : [Math.floor(i * bytes), 1]);

export function traffic(format: TrafficFormat, len: readonly number[], width: number, segmentBytes: number, w: Widths): Traffic {
  const l = format === 'jds' ? sortedLengths(len) : [...len];
  const at = elementIndex(format, l);
  const seen: [Set<number>, Set<number>] = [new Set(), new Set()];
  let requested = 0;
  for (let first = 0; first < l.length; first += width) {
    const rows = l.slice(first, first + width);
    const longest = maxOf(rows);
    for (let t = 0; t < longest; t++) {
      const idx: number[] = [];
      for (let lane = 0; lane < rows.length; lane++) if (rows[lane] > t) idx.push(at(first + lane, t));
      [w.value, w.index].forEach((bytes, k) => {
        const size = addr(0, bytes)[1];
        const a = warpAccess(
          idx.map((i) => addr(i, bytes)[0]),
          size,
          segmentBytes,
        );
        requested += a.transactions;
        for (const s of a.segments) seen[k].add(s);
      });
    }
  }
  const needed = sum(l) * (w.value + w.index);
  return {
    format,
    requested,
    distinct: seen[0].size + seen[1].size,
    needed,
    efficiency: requested ? needed / (requested * segmentBytes) : 1,
  };
}

// ---------------------------------------------------------------- amortising preprocessing
/**
 * Solver iterations after which a conversion pays for itself. `conversionCost` is in units
 * of one SpMV in the old format; the new SpMV takes `speedRatio` of the old one's time.
 */
export const amortiseIterations = (conversionCost: number, speedRatio: number): number =>
  speedRatio >= 1 ? Infinity : conversionCost / (1 - speedRatio);
