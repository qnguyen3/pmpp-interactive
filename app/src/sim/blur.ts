// A thread-level model of the image blur kernel of chapter 3 (book Fig. 3.8, p. 57;
// the corrected copy is labs/ch03/snippets/blur_kernel.cu).
//
// What it models: one thread per position of a 2D grid of blocks, each running the
// kernel body to completion: the outer guard, the nested loop over the patch, the
// four clauses of the inner test in source order (with short-circuit evaluation),
// the running sum and counter, the float division and the truncating cast, and the
// store to the flat row-major index row*w + col. Any of those pieces can be taken
// out, the output can be the input buffer itself, and the threads can run in
// different orders.
//
// What it leaves out, and what the app says wherever this model is shown:
//   - Threads run one at a time, each to completion, block after block, in the
//     chosen order inside each block. That is what the labs' CPU emulator does. A
//     GPU interleaves threads, and the result of a racy kernel there is not
//     predictable at all.
//   - A read outside the buffer returns 0 here and a write outside the buffer is
//     dropped. On a device both are undefined: you may get stale data, corrupt a
//     neighbouring allocation, or fault.
//   - 0/0 is stored as NaN ("no defined value"). In C++ the float result is not a
//     number and converting it to a byte is undefined behaviour.
//   - One channel, tiny pictures, no timing.

import { tr } from '../lib/locale.ts';

export interface Picture {
  w: number;
  h: number;
  /** Row-major bytes: pixel (row, col) is data[row*w + col]. */
  data: number[];
}

export interface Pos {
  row: number;
  col: number;
}

/** The four clauses of the inner test, in the order the source evaluates them. */
export type Clause = 'rowLo' | 'rowHi' | 'colLo' | 'colHi';
export const CLAUSES: Clause[] = ['rowLo', 'rowHi', 'colLo', 'colHi'];
export const CLAUSE_TEXT: Record<Clause, string> = {
  rowLo: 'curRow >= 0',
  rowHi: 'curRow < h',
  colLo: 'curCol >= 0',
  colHi: 'curCol < w',
};

const holds = (c: Clause, curRow: number, curCol: number, w: number, h: number) =>
  c === 'rowLo' ? curRow >= 0 : c === 'rowHi' ? curRow < h : c === 'colLo' ? curCol >= 0 : curCol < w;

// ---------------------------------------------------------------- pictures
const picture = (w: number, h: number, f: (row: number, col: number) => number): Picture => ({
  w,
  h,
  data: Array.from({ length: w * h }, (_, i) => f(Math.floor(i / w), i % w)),
});

/** Single digits 1 to 9, so that sums and averages can be checked by hand. */
export const numbered = (w: number, h: number) => picture(w, h, (r, c) => ((r * 7 + c * 3 + ((r * c) % 4)) % 9) + 1);

/** A slope with a texture on it: neighbouring pixels differ, so a wrong read shows. */
export const textured = (w: number, h: number) =>
  picture(w, h, (r, c) => Math.min(255, 24 + 18 * c + 8 * r + 16 * ((r + 2 * c) % 3)));

/** All zero except one pixel. 252 is 9 times 28, so the correct smudge is exact. */
export const spot = (w: number, h: number, at: Pos = { row: 2, col: 2 }, value = 252) =>
  picture(w, h, (r, c) => (r === at.row && c === at.col ? value : 0));

export const uniform = (w: number, h: number, value = 180) => picture(w, h, () => value);

// ---------------------------------------------------------------- the patch
/** How many of the indices i-r .. i+r lie in 0 .. n-1. */
export function span(i: number, n: number, r: number): number {
  return Math.max(0, Math.min(i + r, n - 1) - Math.max(i - r, 0) + 1);
}

/** How many values the thread for output pixel (row, col) averages: a row term times a column term. */
export function count(row: number, col: number, w: number, h: number, r: number): number {
  return span(row, h, r) * span(col, w, r);
}

/** count() for every pixel, row-major. */
export function countMap(w: number, h: number, r: number): number[] {
  return Array.from({ length: w * h }, (_, i) => count(Math.floor(i / w), i % w, w, h, r));
}

/** The patch area the kernel would visit if nothing were outside: (2r+1) squared. */
export const area = (r: number) => (2 * r + 1) * (2 * r + 1);

export interface PatchPos {
  blurRow: number;
  blurCol: number;
  curRow: number;
  curCol: number;
  inside: boolean;
  /** The first clause of the inner test that is false, or null when all four hold. */
  failed: Clause | null;
}

/** Every position the nested loop visits for output pixel (row, col), in loop order. */
export function patch(row: number, col: number, w: number, h: number, r: number): PatchPos[] {
  const out: PatchPos[] = [];
  for (let blurRow = -r; blurRow < r + 1; ++blurRow)
    for (let blurCol = -r; blurCol < r + 1; ++blurCol) {
      const curRow = row + blurRow;
      const curCol = col + blurCol;
      const failed = CLAUSES.find((c) => !holds(c, curRow, curCol, w, h)) ?? null;
      out.push({ blurRow, blurCol, curRow, curCol, inside: failed === null, failed });
    }
  return out;
}

/** (float)pixVal / pixels, then the cast to unsigned char: C++ truncates toward zero. */
export function average(pixVal: number, pixels: number): { quotient: number; byte: number } {
  if (pixels === 0) return { quotient: NaN, byte: NaN };
  const quotient = Math.fround(Math.fround(pixVal) / Math.fround(pixels));
  return { quotient, byte: Math.trunc(quotient) };
}

// ---------------------------------------------------------------- one thread, step by step
export interface TraceStep extends PatchPos {
  /** One entry per clause, in source order: true, false, or null when && never reached it. */
  clauses: (boolean | null)[];
  /** Flat index read, and the value found there; null when the position was skipped. */
  index: number | null;
  value: number | null;
  /** Running sum and counter after this iteration. */
  pixVal: number;
  pixels: number;
}

export interface Trace {
  row: number;
  col: number;
  steps: TraceStep[];
  pixVal: number;
  pixels: number;
  rejected: number;
  quotient: number;
  byte: number;
  /** Flat index of the store. */
  outIndex: number;
}

/** The correct kernel, for the thread that owns output pixel (row, col). */
export function trace(pic: Picture, row: number, col: number, r: number): Trace {
  const { w, h, data } = pic;
  let pixVal = 0;
  let pixels = 0;
  const steps: TraceStep[] = patch(row, col, w, h, r).map((p) => {
    let stop = false;
    const clauses = CLAUSES.map((c) => {
      if (stop) return null;
      const ok = holds(c, p.curRow, p.curCol, w, h);
      if (!ok) stop = true;
      return ok;
    });
    let index: number | null = null;
    let value: number | null = null;
    if (p.inside) {
      index = p.curRow * w + p.curCol;
      value = data[index];
      pixVal += value;
      ++pixels;
    }
    return { ...p, clauses, index, value, pixVal, pixels };
  });
  return { row, col, steps, pixVal, pixels, rejected: steps.length - pixels, ...average(pixVal, pixels), outIndex: row * w + col };
}

// ---------------------------------------------------------------- the whole grid, with switches
export type Order = 'forward' | 'reversed' | 'shuffled';
export const ORDERS: Order[] = ['forward', 'reversed', 'shuffled'];
export const ORDER_LABEL: Record<Order, string> = {
  get forward() {
    return tr('in order', 'theo thứ tự');
  },
  get reversed() {
    return tr('reversed', 'đảo ngược');
  },
  get shuffled() {
    return tr('shuffled', 'xáo trộn');
  },
};

export interface BlurOptions {
  /** BLUR_SIZE: the patch is 2*radius+1 on a side. */
  radius: number;
  /** Block shape in threads. The grid is the smallest whole number of blocks that covers the picture. */
  blockW: number;
  blockH: number;
  /** Which clauses of the inner test (line 15) are present. */
  clauses: Record<Clause, boolean>;
  /** 'count' divides by the counter (the book's kernel); 'area' divides by (2r+1) squared. */
  divisor: 'count' | 'area';
  /** The test on line 05 that keeps surplus threads from doing anything. */
  outerGuard: boolean;
  /** out and in are the same buffer. */
  inPlace: boolean;
  /** Order of the threads inside each block; blocks run one after another. */
  order: Order;
  /** Seed for the shuffled order. */
  seed: number;
  /** Run only the first `limit` threads of the order (used by the animation). */
  limit?: number;
}

export const CORRECT: BlurOptions = {
  radius: 1,
  blockW: 4,
  blockH: 4,
  clauses: { rowLo: true, rowHi: true, colLo: true, colHi: true },
  divisor: 'count',
  outerGuard: true,
  inPlace: false,
  order: 'forward',
  seed: 1,
};

export interface BadAccess {
  /** Global coordinates of the thread that did it. */
  thread: Pos;
  op: 'read' | 'write';
  /**
   * 'wrapped': the flat index is inside the buffer but belongs to another row: a
   * valid address holding the wrong pixel. 'outside': the index is not in the buffer.
   */
  kind: 'wrapped' | 'outside';
  index: number;
  /** The (row, col) the source expression was written for. */
  wanted: Pos;
  /** The pixel the index really is, when it is inside the buffer. */
  got: Pos | null;
}

export interface BlurResult {
  /** The output picture, row-major. NaN marks a pixel with no defined value. */
  out: number[];
  /** Which output elements have been stored so far. */
  written: boolean[];
  /** What the correct kernel produces from the same input. */
  reference: number[];
  /** Flat indices where out differs from the reference. */
  wrong: number[];
  accesses: BadAccess[];
  /** Threads whose divisor was 0. */
  zeroDivisions: Pos[];
  /** In place only: reads that found an element some thread had already stored. */
  overwrittenReads: number;
  /** Grid shape in blocks, and thread counts. */
  gridW: number;
  gridH: number;
  threads: number;
  surplus: number;
  /** The order the threads ran in (global coordinates), cut at `limit`. */
  ran: Pos[];
}

/** Linear congruential generator, the same recurrence the labs' emulator shuffles with. */
function lcg(seed: number) {
  let s = (Math.imul(seed >>> 0, 2654435761) + 12345) >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s >>> 8;
  };
}

/**
 * Every thread of the launch as global (row, col), in execution order: blocks in
 * row-major order of blockIdx, and inside a block the threads in row-major order of
 * threadIdx, reversed, or shuffled.
 */
export function threadOrder(w: number, h: number, blockW: number, blockH: number, order: Order, seed = 1): Pos[] {
  const gridW = Math.ceil(w / blockW);
  const gridH = Math.ceil(h / blockH);
  const T = blockW * blockH;
  const next = lcg(seed);
  const out: Pos[] = [];
  for (let by = 0; by < gridH; ++by)
    for (let bx = 0; bx < gridW; ++bx) {
      const ids = Array.from({ length: T }, (_, i) => i);
      if (order === 'reversed') ids.reverse();
      else if (order === 'shuffled')
        for (let i = T - 1; i > 0; --i) {
          const j = next() % (i + 1);
          [ids[i], ids[j]] = [ids[j], ids[i]];
        }
      for (const t of ids) out.push({ row: by * blockH + Math.floor(t / blockW), col: bx * blockW + (t % blockW) });
    }
  return out;
}

function run(pic: Picture, o: BlurOptions) {
  const { w, h } = pic;
  const n = w * h;
  const input = o.inPlace ? pic.data.slice() : pic.data;
  const out = o.inPlace ? input : new Array<number>(n).fill(0);
  const written = new Array<boolean>(n).fill(false);
  const accesses: BadAccess[] = [];
  const zeroDivisions: Pos[] = [];
  let overwrittenReads = 0;
  const at = (index: number): Pos | null => (index >= 0 && index < n ? { row: Math.floor(index / w), col: index % w } : null);

  const all = threadOrder(w, h, o.blockW, o.blockH, o.order, o.seed);
  const ran = o.limit === undefined ? all : all.slice(0, Math.max(0, o.limit));
  for (const thread of ran) {
    const { row, col } = thread;
    if (o.outerGuard && !(col < w && row < h)) continue;
    let pixVal = 0;
    let pixels = 0;
    for (let blurRow = -o.radius; blurRow < o.radius + 1; ++blurRow)
      for (let blurCol = -o.radius; blurCol < o.radius + 1; ++blurCol) {
        const curRow = row + blurRow;
        const curCol = col + blurCol;
        if (!CLAUSES.every((c) => !o.clauses[c] || holds(c, curRow, curCol, w, h))) continue;
        const index = curRow * w + curCol;
        const got = at(index);
        if (!(curRow >= 0 && curRow < h && curCol >= 0 && curCol < w))
          accesses.push({ thread, op: 'read', kind: got ? 'wrapped' : 'outside', index, wanted: { row: curRow, col: curCol }, got });
        if (got && o.inPlace && written[index]) ++overwrittenReads;
        pixVal += got ? input[index] : 0;
        ++pixels;
      }
    const divisor = o.divisor === 'area' ? area(o.radius) : pixels;
    if (divisor === 0) zeroDivisions.push(thread);
    const byte = Number.isNaN(pixVal) ? NaN : average(pixVal, divisor).byte;
    const index = row * w + col;
    const got = at(index);
    if (!(col < w && row < h))
      accesses.push({ thread, op: 'write', kind: got ? 'wrapped' : 'outside', index, wanted: { row, col }, got });
    if (got) {
      out[index] = byte;
      written[index] = true;
    }
  }
  return { out, written, accesses, zeroDivisions, overwrittenReads, ran, threads: all.length };
}

/** What the kernel of Fig. 3.8 produces: the average of the valid part of each patch, truncated. */
export function reference(pic: Picture, radius: number): number[] {
  return run(pic, { ...CORRECT, radius }).out;
}

/** Runs the kernel over the whole picture with the given pieces present or missing. */
export function blurImage(pic: Picture, o: BlurOptions): BlurResult {
  const r = run(pic, o);
  const ref = reference(pic, o.radius);
  const wrong: number[] = [];
  r.out.forEach((v, i) => {
    if (v !== ref[i]) wrong.push(i);
  });
  return {
    ...r,
    reference: ref,
    wrong,
    gridW: Math.ceil(pic.w / o.blockW),
    gridH: Math.ceil(pic.h / o.blockH),
    surplus: r.threads - pic.w * pic.h,
  };
}

/**
 * Where the wrong pixels of a result are, for the explanations: the rows and columns
 * that hold at least one, and the rows strictly between the first and last wrong row
 * that hold none. All sorted.
 */
export function wrongSpan(wrong: number[], w: number): { rows: number[]; cols: number[]; spared: number[] } {
  const sorted = (xs: number[]) => [...new Set(xs)].sort((a, b) => a - b);
  const rows = sorted(wrong.map((i) => Math.floor(i / w)));
  const cols = sorted(wrong.map((i) => i % w));
  const spared: number[] = [];
  for (let r = (rows[0] ?? 0) + 1; r < (rows[rows.length - 1] ?? 0); r++) if (!rows.includes(r)) spared.push(r);
  return { rows, cols, spared };
}

/** Counts of bad accesses by operation and kind, for display. */
export function tally(accesses: BadAccess[]) {
  const n = (op: BadAccess['op'], kind: BadAccess['kind']) => accesses.filter((a) => a.op === op && a.kind === kind).length;
  return {
    readWrapped: n('read', 'wrapped'),
    readOutside: n('read', 'outside'),
    writeWrapped: n('write', 'wrapped'),
    writeOutside: n('write', 'outside'),
  };
}

// ---------------------------------------------------------------- single defects, for the challenge
export type Defect = Clause | 'area' | 'outer' | 'inPlace';
export const DEFECTS: Defect[] = ['rowLo', 'rowHi', 'colLo', 'colHi', 'area', 'outer', 'inPlace'];
export const DEFECT_LABEL: Record<Defect, string> = {
  get rowLo() {
    return tr('curRow >= 0 is missing', 'thiếu curRow >= 0');
  },
  get rowHi() {
    return tr('curRow < h is missing', 'thiếu curRow < h');
  },
  get colLo() {
    return tr('curCol >= 0 is missing', 'thiếu curCol >= 0');
  },
  get colHi() {
    return tr('curCol < w is missing', 'thiếu curCol < w');
  },
  get area() {
    return tr('the divisor is the patch area', 'số chia là diện tích vùng');
  },
  get outer() {
    return tr('the outer guard (line 05) is missing', 'thiếu phép kiểm tra ngoài (dòng 05)');
  },
  get inPlace() {
    return tr('out is the same buffer as in', 'out và in là cùng một buffer');
  },
};

/** The correct kernel with exactly one thing wrong. */
export function withDefect(d: Defect, base: BlurOptions = CORRECT): BlurOptions {
  switch (d) {
    case 'area':
      return { ...base, divisor: 'area' };
    case 'outer':
      return { ...base, outerGuard: false };
    case 'inPlace':
      return { ...base, inPlace: true };
    default:
      return { ...base, clauses: { ...base.clauses, [d]: false } };
  }
}

/** Three different defects, chosen by the seed. */
export function challenge(seed: number): Defect[] {
  const next = lcg(seed * 7 + 3);
  const pool = DEFECTS.slice();
  for (let i = pool.length - 1; i > 0; --i) {
    const j = next() % (i + 1);
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, 3);
}
