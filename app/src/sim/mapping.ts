// Thread-to-data mapping for chapter 3, section 3.2 (book pp. 48-55).
//
// One model for the thread map, the kernel stepper, the flattening explorer, the
// erratum tester and the narrated scene "A grid over a picture".
//
// What it models: which thread of a 2D launch owns which pixel of a height x width
// picture, which threads have no pixel, how many threads of each block work, and
// where a pixel sits in a flat array. Three launch styles:
//   'blocks'        CUDA or HIP: the grid is a whole number of blocks.
//   'threadgroups'  Metal dispatchThreadgroups: the same geometry under Metal's names.
//   'threads'       Metal dispatchThreads: the grid is exactly the data and the
//                   threadgroups on the far edges are smaller.
//
// What it leaves out: time. Nothing here says when a thread runs, how threads are
// packed into warps, or what a surplus thread costs. The trimmed edge groups of the
// 'threads' style follow Apple's description (notes/today/ch03-facts.md, C3) and
// were not measured on a device; that they sit on the right and bottom edges is
// this model's assumption.

import { tr as trText } from '../lib/locale.ts';

export type Style = 'blocks' | 'threadgroups' | 'threads';

export const STYLE_LABEL: Record<Style, string> = {
  get blocks() {
    return trText('CUDA or HIP: whole blocks', 'CUDA hoặc HIP: nguyên khối');
  },
  get threadgroups() {
    return trText('Metal: dispatchThreadgroups', 'Metal: dispatchThreadgroups');
  },
  get threads() {
    return trText('Metal: dispatchThreads', 'Metal: dispatchThreads');
  },
};

export interface Launch {
  /** Rows of the picture (the book's n, the y extent). */
  height: number;
  /** Columns of the picture (the book's m, the x extent). */
  width: number;
  /** Threads per block along x. */
  blockW: number;
  /** Threads per block along y. */
  blockH: number;
  style: Style;
  /** The defect of question q-swapped: x sized from the height, y from the width. */
  swapped?: boolean;
}

/** The book's running example: a 62 x 76 picture under 16 x 16 blocks (p. 49). */
export const BOOK: Launch = { height: 62, width: 76, blockW: 16, blockH: 16, style: 'blocks' };

/** Bytes per pixel of the colour input (the book's CHANNELS, footnote 2 on p. 53). */
export const CHANNELS = 3;

export interface Layout {
  /** Blocks along x and y: gridDim.x and gridDim.y. */
  gridX: number;
  gridY: number;
  blocks: number;
  /** Threads along x and y. */
  threadsX: number;
  threadsY: number;
  threads: number;
  pixels: number;
  /** Columns and rows of the picture that some thread reaches. */
  coveredCols: number;
  coveredRows: number;
  /** Threads that pass the range test. */
  working: number;
  /** Threads with no pixel. */
  surplus: number;
  /** Pixels no thread reaches (0 unless the extents are swapped). */
  unreached: number;
}

const ceilDiv = (a: number, b: number) => Math.ceil(a / b);

export function layout(l: Launch): Layout {
  // The launch sizes x from the width and y from the height (p. 50). Swapped, it does the opposite.
  const forX = l.swapped ? l.height : l.width;
  const forY = l.swapped ? l.width : l.height;
  const gridX = ceilDiv(forX, l.blockW);
  const gridY = ceilDiv(forY, l.blockH);
  const threadsX = l.style === 'threads' ? forX : gridX * l.blockW;
  const threadsY = l.style === 'threads' ? forY : gridY * l.blockH;
  const coveredCols = Math.min(l.width, threadsX);
  const coveredRows = Math.min(l.height, threadsY);
  const pixels = l.height * l.width;
  const working = coveredCols * coveredRows;
  return {
    gridX, gridY, blocks: gridX * gridY,
    threadsX, threadsY, threads: threadsX * threadsY,
    pixels, coveredCols, coveredRows, working,
    surplus: threadsX * threadsY - working,
    unreached: pixels - working,
  };
}

export interface XY {
  x: number;
  y: number;
}

export interface ThreadInfo {
  /** Position in the lattice of threads: x across, y down. */
  x: number;
  y: number;
  blockIdx: XY;
  threadIdx: XY;
  /** The block size given at the launch. */
  blockDim: XY;
  /** The size of this thread's own block; smaller than blockDim only for a trimmed Metal edge group. */
  groupSize: XY;
  row: number;
  col: number;
  colOk: boolean;
  rowOk: boolean;
  /** Passes `col < width && row < height`. */
  works: boolean;
  /** The first clause of the guard that is false. && stops there, so the other is not evaluated. */
  failing: 'col' | 'row' | null;
  /** row*width + col, or null for a thread that fails the guard. */
  grayOffset: number | null;
  /** grayOffset*CHANNELS: where the pixel's red byte is in the colour input. */
  rgbOffset: number | null;
}

/** Everything the thread at lattice position (x, y) computes before it touches memory. */
export function threadAt(l: Launch, x: number, y: number): ThreadInfo {
  const L = layout(l);
  if (x < 0 || y < 0 || x >= L.threadsX || y >= L.threadsY) throw new RangeError(`no thread at (${x}, ${y})`);
  const blockIdx = { x: Math.floor(x / l.blockW), y: Math.floor(y / l.blockH) };
  const threadIdx = { x: x % l.blockW, y: y % l.blockH };
  const blockDim = { x: l.blockW, y: l.blockH };
  const col = blockIdx.x * blockDim.x + threadIdx.x;
  const row = blockIdx.y * blockDim.y + threadIdx.y;
  const colOk = col < l.width;
  const rowOk = row < l.height;
  const works = colOk && rowOk;
  const grayOffset = works ? flat(row, col, l.width) : null;
  const b = blockInfo(l, blockIdx.x, blockIdx.y);
  return {
    x, y, blockIdx, threadIdx, blockDim, groupSize: { x: b.w, y: b.h }, row, col, colOk, rowOk, works,
    failing: !colOk ? 'col' : !rowOk ? 'row' : null,
    grayOffset,
    rgbOffset: grayOffset == null ? null : grayOffset * CHANNELS,
  };
}

/** Lattice position of thread (tx, ty) of block (bx, by). */
export const latticeOf = (l: Launch, bx: number, by: number, tx: number, ty: number): XY => ({
  x: bx * l.blockW + tx,
  y: by * l.blockH + ty,
});

/** The book labels blocks and threads highest dimension first (p. 48): Block (y, x). */
export const bookLabel = (p: XY) => `(${p.y},${p.x})`;

export type Cut = 'full' | 'partial' | 'idle';
/** The book's four areas (Fig. 3.5), plus blocks in which no thread works. */
export type RegionId = '1' | '2' | '3' | '4' | 'idle';

export interface BlockInfo {
  bx: number;
  by: number;
  /** Lattice position of the block's first thread. */
  x0: number;
  y0: number;
  /** Threads this block really has across and down (trimmed for a Metal edge group). */
  w: number;
  h: number;
  launched: number;
  /** Columns and rows of the block whose threads pass the guard. */
  workW: number;
  workH: number;
  working: number;
  cutX: Cut;
  cutY: Cut;
  region: RegionId;
}

const cutOf = (work: number, full: number): Cut => (work >= full ? 'full' : work <= 0 ? 'idle' : 'partial');

export function blockInfo(l: Launch, bx: number, by: number): BlockInfo {
  const L = layout(l);
  if (bx < 0 || by < 0 || bx >= L.gridX || by >= L.gridY) throw new RangeError(`no block (${bx}, ${by})`);
  const x0 = bx * l.blockW;
  const y0 = by * l.blockH;
  const w = Math.min(l.blockW, L.threadsX - x0);
  const h = Math.min(l.blockH, L.threadsY - y0);
  const workW = Math.max(0, Math.min(x0 + w, l.width) - x0);
  const workH = Math.max(0, Math.min(y0 + h, l.height) - y0);
  const cutX = cutOf(workW, l.blockW);
  const cutY = cutOf(workH, l.blockH);
  const region: RegionId =
    cutX === 'idle' || cutY === 'idle' ? 'idle'
    : cutX === 'full' ? (cutY === 'full' ? '1' : '3')
    : cutY === 'full' ? '2' : '4';
  return { bx, by, x0, y0, w, h, launched: w * h, workW, workH, working: workW * workH, cutX, cutY, region };
}

export interface Region {
  region: RegionId;
  blocks: number;
  /** Working threads in each block of the region. */
  working: number;
  /** Threads each block of the region has. */
  launched: number;
  /** blocks * working. */
  total: number;
}

const REGION_ORDER: RegionId[] = ['1', '2', '3', '4', 'idle'];

/** A run of consecutive block columns (or rows) that are cut the same way. */
export interface Band {
  cut: Cut;
  first: number;
  count: number;
}

/**
 * Block columns and block rows grouped by how the picture's edge cuts them. Along an
 * axis the full blocks come first, then at most one partial block, then idle ones.
 */
export function blockBands(l: Launch): { xs: Band[]; ys: Band[] } {
  const L = layout(l);
  return {
    xs: bandsAlong(L.gridX, (bx) => blockInfo(l, bx, 0).cutX),
    ys: bandsAlong(L.gridY, (by) => blockInfo(l, 0, by).cutY),
  };
}

function bandsAlong(n: number, cutAt: (i: number) => Cut): Band[] {
  const out: Band[] = [];
  for (let i = 0; i < n; i++) {
    const cut = cutAt(i);
    const e = out.find((o) => o.cut === cut);
    if (e) e.count++;
    else out.push({ cut, first: i, count: 1 });
  }
  return out;
}

/** The kinds of block in a launch, in the book's order, leaving out kinds with no blocks. */
export function regions(l: Launch): Region[] {
  const found = new Map<RegionId, Region>();
  // Every block of one kind has the same counts, so one representative per pair of bands is enough.
  const { xs, ys } = blockBands(l);
  for (const cx of xs)
    for (const cy of ys) {
      const b = blockInfo(l, cx.first, cy.first);
      const n = cx.count * cy.count;
      const r = found.get(b.region);
      if (r) {
        r.blocks += n; // only 'idle' is reached more than once, and no thread works in any of it
        r.launched = Math.max(r.launched, b.launched);
      } else found.set(b.region, { region: b.region, blocks: n, working: b.working, launched: b.launched, total: 0 });
    }
  return REGION_ORDER.flatMap((id) => {
    const r = found.get(id);
    return r ? [{ ...r, total: r.blocks * r.working }] : [];
  });
}

// ---------------------------------------------------------------- flattening (pp. 50-52, 55)

/** Row-major: all of a row together, rows one after another. `row*width` skips the rows before. */
export const flat = (row: number, col: number, width: number) => row * width + col;
export const unflat = (i: number, width: number) => ({ row: Math.floor(i / width), col: i % width });

/** Column-major: all of a column together. The stride is the height (cuBLAS's IDX2C with ld = rows). */
export const flatColMajor = (row: number, col: number, height: number) => col * height + row;
export const unflatColMajor = (i: number, height: number) => ({ row: i % height, col: Math.floor(i / height) });

/** Three dimensions: planes one after another (p. 55). */
export const flat3 = (plane: number, row: number, col: number, height: number, width: number) =>
  plane * height * width + row * width + col;
export const unflat3 = (i: number, height: number, width: number) => ({
  plane: Math.floor(i / (height * width)),
  row: Math.floor((i % (height * width)) / width),
  col: i % width,
});

/**
 * The jump in the flat array for one step along each dimension of a row-major array of
 * this shape: what torch.Tensor.stride() reports for a contiguous tensor.
 */
export function strides(shape: number[]): number[] {
  const out = new Array<number>(shape.length);
  let size = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    out[i] = size;
    size *= shape[i];
  }
  return out;
}

export interface Dims {
  depth: number;
  height: number;
  width: number;
}
/** 'wrong' is the defect `row*height + col`: the other dimension used as the stride. */
export type Indexing = 'row' | 'col' | 'wrong';

/** Strides for (plane, row, col) under each way of indexing. Column-major is defined for one plane. */
export function stridesFor(d: Dims, kind: Indexing): [number, number, number] {
  const plane = d.height * d.width;
  return kind === 'row' ? [plane, d.width, 1] : kind === 'col' ? [plane, 1, d.height] : [plane, d.height, 1];
}

export function offsetOf(d: Dims, kind: Indexing, plane: number, row: number, col: number): number {
  const s = stridesFor(d, kind);
  return plane * s[0] + row * s[1] + col * s[2];
}

/** The element stored in slot i, or null where the indexing is not one-to-one ('wrong'). */
export function elementAt(d: Dims, kind: Indexing, i: number): { plane: number; row: number; col: number } | null {
  if (kind === 'wrong') return null;
  const plane = Math.floor(i / (d.height * d.width));
  const rest = i % (d.height * d.width);
  const rc = kind === 'row' ? unflat(rest, d.width) : unflatColMajor(rest, d.height);
  return { plane, ...rc };
}

export interface Coverage {
  size: number;
  /** How many elements land on each slot. */
  hits: number[];
  /** Slots written more than once. */
  collisions: number[];
  /** Slots never written. */
  holes: number[];
  /** Elements whose offset is past the end of the array. */
  outOfBounds: { plane: number; row: number; col: number; offset: number }[];
}

/** Write every element once through this indexing and see where the writes land. */
export function indexingCoverage(d: Dims, kind: Indexing): Coverage {
  const size = d.depth * d.height * d.width;
  const hits = new Array<number>(size).fill(0);
  const outOfBounds: Coverage['outOfBounds'] = [];
  for (let plane = 0; plane < d.depth; plane++)
    for (let row = 0; row < d.height; row++)
      for (let col = 0; col < d.width; col++) {
        const offset = offsetOf(d, kind, plane, row, col);
        if (offset < size) hits[offset]++;
        else outOfBounds.push({ plane, row, col, offset });
      }
  const idx = hits.map((_, i) => i);
  return { size, hits, collisions: idx.filter((i) => hits[i] > 1), holes: idx.filter((i) => hits[i] === 0), outOfBounds };
}

// ---------------------------------------------------------------- the grayscale arithmetic

/**
 * 'printed'       line 19 of Fig. 3.4 as printed: (unsigned char) 0.21f*r+0.71f*g+0.07f*b.
 *                 The cast binds to 0.21f alone, which becomes 0, so red contributes nothing.
 * 'printedParen'  the printed constants with the whole sum inside the cast.
 * 'corrected'     the constants of the book's text (p. 52), sum inside the cast.
 */
export type GrayVariant = 'printed' | 'printedParen' | 'corrected';

export const GRAY_WEIGHTS: Record<GrayVariant, [number, number, number]> = {
  printed: [0.21, 0.71, 0.07],
  printedParen: [0.21, 0.71, 0.07],
  corrected: [0.299, 0.587, 0.114],
};

const f32 = Math.fround;

export interface GrayDetail {
  /** The three products, as single-precision values. In 'printed' the first is the integer 0. */
  terms: [number, number, number];
  /** The single-precision sum, added left to right. */
  sum: number;
  /** What the conversion to unsigned char stores: the sum with its fraction dropped. */
  byte: number;
}

/**
 * The value one pixel gets. Arithmetic is done in single precision, one rounded operation
 * at a time and left to right, as the C++ expression is written. A compiler that fuses a
 * multiply and an add, or reorders the additions, can differ from this by one grey level.
 */
export function grayDetail(r: number, g: number, b: number, variant: GrayVariant = 'corrected'): GrayDetail {
  const [wr, wg, wb] = GRAY_WEIGHTS[variant];
  // As printed, (unsigned char)0.21f is 0 and 0*r is the integer 0.
  const tr = variant === 'printed' ? 0 : f32(f32(wr) * r);
  const tg = f32(f32(wg) * g);
  const tb = f32(f32(wb) * b);
  const sum = f32(f32(tr + tg) + tb);
  return { terms: [tr, tg, tb], sum, byte: Math.trunc(sum) };
}

export const gray = (r: number, g: number, b: number, variant: GrayVariant = 'corrected') =>
  grayDetail(r, g, b, variant).byte;

/** A small synthetic colour picture, so the stepper has bytes to read. Any (row, col) works. */
export function pixelAt(row: number, col: number): [number, number, number] {
  return [(37 * row + 11 * col + 50) % 256, (17 * row + 29 * col + 90) % 256, (53 * row + 7 * col + 20) % 256];
}

// ---------------------------------------------------------------- one thread through the kernel

export type GrayOp =
  | 'col' | 'row' | 'guard' | 'grayOffset' | 'rgbOffset' | 'r' | 'g' | 'b' | 'sum' | 'cast' | 'store' | 'done';

export interface GrayStep {
  op: GrayOp;
  /** Register values once the step has run. */
  regs: Record<string, number>;
  /** The register this step wrote. */
  set?: string;
  /** The global-memory access this step made. */
  mem?: { array: 'Pin' | 'Pout'; index: number; value: number; write: boolean };
  /** For the guard: did the thread pass. */
  pass?: boolean;
  /** One sentence with the numbers substituted. */
  text: string;
}

/** The corrected kernel, one source-level step at a time, for the thread at lattice (x, y). */
export function stepsGrayscale(l: Launch, x: number, y: number): GrayStep[] {
  const t = threadAt(l, x, y);
  const regs: Record<string, number> = {};
  const steps: GrayStep[] = [];
  const push = (s: Omit<GrayStep, 'regs'>) => steps.push({ ...s, regs: { ...regs } });

  regs.col = t.col;
  push({ op: 'col', set: 'col',
    text: `col = blockIdx.x*blockDim.x + threadIdx.x = ${t.blockIdx.x}*${t.blockDim.x} + ${t.threadIdx.x} = ${t.col}.` });
  regs.row = t.row;
  push({ op: 'row', set: 'row',
    text: `row = blockIdx.y*blockDim.y + threadIdx.y = ${t.blockIdx.y}*${t.blockDim.y} + ${t.threadIdx.y} = ${t.row}.` });

  const colTest = trText(`col < width is ${t.col} < ${l.width}, ${t.colOk}`, `col < width là ${t.col} < ${l.width}, ${t.colOk}`);
  const rowTest = trText(`row < height is ${t.row} < ${l.height}, ${t.rowOk}`, `row < height là ${t.row} < ${l.height}, ${t.rowOk}`);
  if (!t.works) {
    push({ op: 'guard', pass: false,
      text: t.failing === 'col'
        ? trText(
            `Guard: ${colTest}. The && stops there; the row test is not evaluated. The body is skipped.`,
            `Điều kiện chặn: ${colTest}. Toán tử && dừng ở đó; phép kiểm tra hàng không được tính. Phần thân bị bỏ qua.`,
          )
        : trText(`Guard: ${colTest}; ${rowTest}. The body is skipped.`, `Điều kiện chặn: ${colTest}; ${rowTest}. Phần thân bị bỏ qua.`) });
    push({ op: 'done', text: trText('This thread has no pixel. It returns without reading or writing memory.', 'Luồng này không có điểm ảnh nào. Nó trả về mà không đọc hay ghi bộ nhớ.') });
    return steps;
  }
  push({ op: 'guard', pass: true, text: trText(`Guard: ${colTest}; ${rowTest}. This thread has a pixel.`, `Điều kiện chặn: ${colTest}; ${rowTest}. Luồng này có một điểm ảnh.`) });

  const go = t.grayOffset!;
  const ro = t.rgbOffset!;
  regs.grayOffset = go;
  push({ op: 'grayOffset', set: 'grayOffset',
    text: trText(
      `grayOffset = row*width + col = ${t.row}*${l.width} + ${t.col} = ${go}: ${t.row} full rows of ${l.width} come first.`,
      `grayOffset = row*width + col = ${t.row}*${l.width} + ${t.col} = ${go}: ${t.row} hàng đầy đủ, mỗi hàng ${l.width} phần tử, đứng trước.`,
    ) });
  regs.rgbOffset = ro;
  push({ op: 'rgbOffset', set: 'rgbOffset',
    text: trText(
      `rgbOffset = grayOffset*CHANNELS = ${go}*${CHANNELS} = ${ro}: every pixel before this one took ${CHANNELS} bytes.`,
      `rgbOffset = grayOffset*CHANNELS = ${go}*${CHANNELS} = ${ro}: mỗi điểm ảnh đứng trước điểm này chiếm ${CHANNELS} byte.`,
    ) });

  const [r, g, b] = pixelAt(t.row, t.col);
  const chans: ['r' | 'g' | 'b', number, string][] = [
    ['r', r, trText('red', 'đỏ')],
    ['g', g, trText('green', 'lục')],
    ['b', b, trText('blue', 'lam')],
  ];
  chans.forEach(([name, value, word], k) => {
    regs[name] = value;
    push({ op: name, set: name, mem: { array: 'Pin', index: ro + k, value, write: false },
      text: trText(
        `${name} = Pin[rgbOffset${k ? ' + ' + k : ''}] = Pin[${ro + k}] = ${value}, the ${word} byte.`,
        `${name} = Pin[rgbOffset${k ? ' + ' + k : ''}] = Pin[${ro + k}] = ${value}, byte màu ${word}.`,
      ) });
  });

  const d = grayDetail(r, g, b, 'corrected');
  regs.sum = d.sum;
  push({ op: 'sum', set: 'sum',
    text: trText(
      `The sum is computed in float: 0.299f*${r} + 0.587f*${g} + 0.114f*${b} = ${fmtFloat(d.sum)}.`,
      `Tổng được tính bằng float: 0.299f*${r} + 0.587f*${g} + 0.114f*${b} = ${fmtFloat(d.sum)}.`,
    ) });
  regs.byte = d.byte;
  push({ op: 'cast', set: 'byte',
    text: trText(
      `(unsigned char) applies to the whole sum and drops the fraction: ${fmtFloat(d.sum)} becomes ${d.byte}.`,
      `(unsigned char) áp dụng cho cả tổng và bỏ phần thập phân: ${fmtFloat(d.sum)} thành ${d.byte}.`,
    ) });
  push({ op: 'store', mem: { array: 'Pout', index: go, value: d.byte, write: true },
    text: trText(`Pout[grayOffset] = Pout[${go}] = ${d.byte}. No other thread writes this element.`, `Pout[grayOffset] = Pout[${go}] = ${d.byte}. Không luồng nào khác ghi phần tử này.`) });
  push({ op: 'done', text: trText('The thread is finished: three bytes read, one byte written.', 'Luồng đã xong: đọc ba byte, ghi một byte.') });
  return steps;
}

/** A float as the stepper prints it: up to three decimals, no trailing zeros. */
export const fmtFloat = (v: number) => String(Math.round(v * 1000) / 1000);
