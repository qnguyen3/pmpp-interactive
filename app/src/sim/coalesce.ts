// A warp-level model of global-memory coalescing for chapter 6 (book §6.1, pp. 124–132).
//
// What it models: one load (or store) instruction executed by one warp. Every
// active lane names a byte address and an access size; the hardware serves the
// warp with whole aligned segments of memory, and the model counts the distinct
// segments the lanes touch. That is the rule NVIDIA documents today for compute
// capability 6.0 and later, with 32-byte segments ("sectors") inside 128-byte
// L1/L2 lines (notes/today/ch06-facts.md A2, A3, A4). AMD documents the same
// mechanism with 64- and/or 128-byte cache lines (B1). Apple documents no rule,
// no line size and no sector size (C1), so for Metal the segment size is a choice
// the reader makes, not a fact.
//
// What it leaves out, and what the app says wherever this model is shown:
//   - One instruction at a time. No cache: a segment fetched by one instruction is
//     not remembered by the next, so reuse across loop iterations or warps is not
//     counted (a real L1/L2 may serve part of it; only a measurement says how much).
//   - No timing, no DRAM channels or banks, no queueing.
//   - The segment size is the vendor's documented unit; nothing else about the
//     memory system is modelled.
//
// Used by section 6.1, the "One instruction, thirty-two addresses" and "Turn the
// corner, then pad" scenes, and (read-only) section 6.5. The `warpAccess` contract
// below is fixed: other sections rely on it. It imports only lib/locale.ts, so the browser
// checks can still import it directly.

import { tr } from '../lib/locale.ts';

export interface WarpAccess {
  /** Distinct aligned segments touched, as segment indices (byte address ÷ segmentBytes), ascending. */
  segments: number[];
  /** Number of segments, i.e. memory transactions in the vendor's unit. */
  transactions: number;
  /** transactions × segmentBytes. */
  bytesFetched: number;
  /** Distinct bytes the lanes asked for (a broadcast of one float is 4 bytes). */
  bytesNeeded: number;
  /** bytesNeeded ÷ bytesFetched, 0 for an empty warp. */
  efficiency: number;
}

/** Merged, sorted [start, end) byte intervals covered by the lanes. */
function union(byteAddrs: readonly number[], bytesPerLane: number): [number, number][] {
  const iv = byteAddrs.map((a) => [a, a + bytesPerLane] as [number, number]).sort((x, y) => x[0] - y[0]);
  const out: [number, number][] = [];
  for (const [lo, hi] of iv) {
    const last = out[out.length - 1];
    if (last && lo <= last[1]) last[1] = Math.max(last[1], hi);
    else out.push([lo, hi]);
  }
  return out;
}

/**
 * The segments one warp's access touches. `byteAddrs` holds the byte address of
 * every active lane (leave inactive lanes out); each lane reads `bytesPerLane`
 * bytes from its address. A lane whose bytes straddle a boundary touches both
 * segments.
 */
export function warpAccess(byteAddrs: readonly number[], bytesPerLane: number, segmentBytes: number): WarpAccess {
  const segs = new Set<number>();
  for (const a of byteAddrs) {
    const first = Math.floor(a / segmentBytes);
    const last = Math.floor((a + bytesPerLane - 1) / segmentBytes);
    for (let s = first; s <= last; s++) segs.add(s);
  }
  const segments = [...segs].sort((x, y) => x - y);
  const bytesFetched = segments.length * segmentBytes;
  const bytesNeeded = union(byteAddrs, bytesPerLane).reduce((n, [lo, hi]) => n + hi - lo, 0);
  return {
    segments,
    transactions: segments.length,
    bytesFetched,
    bytesNeeded,
    efficiency: bytesFetched ? bytesNeeded / bytesFetched : 0,
  };
}

/**
 * How many separate runs of consecutive bytes the warp asks for. One run is the
 * book's picture of a coalesced access (p. 126), in any lane order: NVIDIA notes
 * that consecutive lane order is "merely the typical way" (A2).
 */
export function runs(byteAddrs: readonly number[], bytesPerLane: number): number {
  return union(byteAddrs, bytesPerLane).length;
}

// ---------------------------------------------------------------- segment models

export type SegmentModelId = 'nvidia' | 'amd64' | 'amd128' | 'metal';

export interface SegmentModel {
  id: SegmentModelId;
  label: string;
  /** Size of one transaction, or null where the vendor documents none. */
  segmentBytes: number | null;
  /** Cache-line size, where documented. */
  lineBytes: number | null;
  /** What one transaction is called in this vendor's documents. */
  unit: string;
  /** Where the numbers come from. */
  facts: string;
}

export const SEGMENT_MODELS: Record<SegmentModelId, SegmentModel> = {
  nvidia: {
    id: 'nvidia',
    get label() {
      return tr('NVIDIA: 32-byte sectors in 128-byte lines', 'NVIDIA: sector 32 byte trong dòng 128 byte');
    },
    segmentBytes: 32,
    lineBytes: 128,
    unit: 'sector', // i18n-ignore: identifier also shown to the reader (unit key)
    facts: 'notes/today/ch06-facts.md A2, A3, A4', // i18n-ignore: file path
  },
  amd64: {
    id: 'amd64',
    get label() {
      return tr('AMD: 64-byte cache lines', 'AMD: dòng bộ nhớ đệm (cache) 64 byte');
    },
    segmentBytes: 64,
    lineBytes: 64,
    unit: 'line', // i18n-ignore: identifier also shown to the reader (unit key)
    facts: 'notes/today/ch06-facts.md B1', // i18n-ignore: file path
  },
  amd128: {
    id: 'amd128',
    get label() {
      return tr('AMD: 128-byte cache lines', 'AMD: dòng bộ nhớ đệm (cache) 128 byte');
    },
    segmentBytes: 128,
    lineBytes: 128,
    unit: 'line', // i18n-ignore: identifier also shown to the reader (unit key)
    facts: 'notes/today/ch06-facts.md B1', // i18n-ignore: file path
  },
  metal: {
    id: 'metal',
    get label() {
      return tr('Apple: not documented', 'Apple: không có tài liệu');
    },
    segmentBytes: null,
    lineBytes: null,
    unit: 'segment', // i18n-ignore: identifier also shown to the reader (unit key)
    facts: 'notes/today/ch06-facts.md C1', // i18n-ignore: file path
  },
};

// ---------------------------------------------------------------- the book's expressions

export type ExprId = 'n-row' | 'n-col' | 'm-row' | 'stride' | 'offset' | 'ex1-a' | 'ex1-c' | 'ex1-d' | 'ex1-g' | 'ex1-i';

export interface ExprInfo {
  id: ExprId;
  /** The index expression as the kernel writes it. */
  code: string;
  /** Matrix expressions use Width, k, row and the block width; the others a 1D index i. */
  matrix: boolean;
}

export const EXPRESSIONS: Record<ExprId, ExprInfo> = {
  'n-row': { id: 'n-row', code: 'N[k*Width + col]', matrix: true },
  'n-col': { id: 'n-col', code: 'N[col*Width + k]', matrix: true },
  'm-row': { id: 'm-row', code: 'M[row*Width + k]', matrix: true },
  stride: { id: 'stride', code: 'in[i*stride]', matrix: false },
  offset: { id: 'offset', code: 'a[i + offset]', matrix: false },
  'ex1-a': { id: 'ex1-a', code: 'a[i]', matrix: false },
  'ex1-c': { id: 'ex1-c', code: 'b[j*blockDim.x*gridDim.x + i]', matrix: false },
  'ex1-d': { id: 'ex1-d', code: 'c[i*4 + j]', matrix: false },
  'ex1-g': { id: 'ex1-g', code: 'd[i + 8]', matrix: false },
  'ex1-i': { id: 'ex1-i', code: 'e[i*8]', matrix: false },
};

export interface AccessParams {
  /** Matrices are Width × Width (Figs. 6.2 and 6.3). */
  Width: number;
  /** The inner-product loop index. */
  k: number;
  /** Row of P computed by the warp's first lane. */
  row: number;
  /** For in[i*stride], in elements. */
  stride: number;
  /** For a[i + offset], in elements. */
  offset: number;
  warpWidth: number;
  /** col (matrix expressions) or i (the others) of the warp's lane 0. */
  lane0: number;
  /**
   * Threads per block row (blockDim.x). When it is smaller than the warp, one warp
   * covers several rows of the block: lane l has tx = l mod blockDimX and is
   * floor(l / blockDimX) rows further down. Default: at least as wide as the warp.
   */
  blockDimX?: number;
  /** Exercise 1's loop index j (0 to 3). */
  j?: number;
  /** Exercise 1's blockDim.x * gridDim.x. */
  gridThreads?: number;
}

export interface Lane {
  lane: number;
  /** Element index into the array. */
  element: number;
  /** Byte offset from the array's start (arrays are assumed to start on a 256-byte boundary, A3). */
  byte: number;
  /** For matrix expressions: the logical element (r, c) read. */
  r?: number;
  c?: number;
  /** For matrix expressions: the row of P this lane computes and its column. */
  row?: number;
  col?: number;
}

/**
 * Each active lane's address for one execution of the expression. Matrix
 * expressions apply the kernel's guard (row < Width && col < Width), so a Width
 * smaller than the warp leaves lanes inactive, as in the book's four-thread toy.
 */
export function laneAddresses(expr: ExprId, p: AccessParams, elemBytes = 4): Lane[] {
  const out: Lane[] = [];
  const bdx = Math.max(1, Math.min(p.blockDimX ?? p.warpWidth, p.warpWidth));
  for (let l = 0; l < p.warpWidth; l++) {
    if (EXPRESSIONS[expr].matrix) {
      const col = p.lane0 + (l % bdx);
      const row = p.row + Math.floor(l / bdx);
      if (row >= p.Width || col >= p.Width) continue;
      let r: number, c: number, element: number;
      if (expr === 'n-row') [r, c, element] = [p.k, col, p.k * p.Width + col];
      else if (expr === 'n-col') [r, c, element] = [p.k, col, col * p.Width + p.k];
      else [r, c, element] = [row, p.k, row * p.Width + p.k];
      out.push({ lane: l, element, byte: element * elemBytes, r, c, row, col });
    } else {
      const i = p.lane0 + l;
      const j = p.j ?? 0;
      const element =
        expr === 'stride' ? i * p.stride
        : expr === 'offset' ? i + p.offset
        : expr === 'ex1-a' ? i
        : expr === 'ex1-c' ? j * (p.gridThreads ?? 1024) + i
        : expr === 'ex1-d' ? i * 4 + j
        : expr === 'ex1-g' ? i + 8
        : i * 8;
      out.push({ lane: l, element, byte: element * elemBytes });
    }
  }
  return out;
}

/** The access of one expression, in one segment model. */
export function exprAccess(expr: ExprId, p: AccessParams, segmentBytes: number, elemBytes = 4): WarpAccess & { lanes: Lane[]; runs: number } {
  const lanes = laneAddresses(expr, p, elemBytes);
  const addrs = lanes.map((l) => l.byte);
  return { ...warpAccess(addrs, elemBytes, segmentBytes), lanes, runs: runs(addrs, elemBytes) };
}

/**
 * Fraction of fetched bytes used when a warp reads in[i*stride] from an aligned
 * start: the model curve beside lab 6.1's stride sweep.
 */
export function strideEfficiency(stride: number, segmentBytes: number, elemBytes = 4, warpWidth = 32): number {
  const p: AccessParams = { Width: 0, k: 0, row: 0, stride, offset: 0, warpWidth, lane0: 0 };
  return exprAccess('stride', p, segmentBytes, elemBytes).efficiency;
}

// ---------------------------------------------------------------- corner turning (Fig. 6.4)

export type Layout = 'row' | 'col';
export type Arrangement = 'ch5' | 'turned';

export interface TileThread {
  /** Linear thread index in the block, ty·T + tx. */
  t: number;
  tx: number;
  ty: number;
  warp: number;
  lane: number;
  /** Logical element N(r, c) this thread loads. */
  r: number;
  c: number;
  /** Flat index into N under the layout, and its byte offset. */
  element: number;
  byte: number;
  /** Where it goes in the shared tile, which keeps the tile's logical layout: Nds[r - ph·T][c - bx·T]. */
  sharedRow: number;
  sharedCol: number;
}

export interface TileWarp {
  warp: number;
  threads: TileThread[];
  access: WarpAccess;
  runs: number;
}

export interface TileLoadParams {
  /** How N is stored: row-major (N(r, c) at r·Width + c) or column-major (at c·Width + r). */
  layout: Layout;
  /** 'ch5': thread (tx, ty) loads N(ph·T + ty, bx·T + tx), Fig. 6.4(a). 'turned': N(ph·T + tx, bx·T + ty), Fig. 6.4(b). */
  arrangement: Arrangement;
  /** Tile width; the block is T × T threads. */
  T: number;
  /** Block column; the N tile does not depend on the block row. */
  bx: number;
  /** Phase. */
  ph: number;
  Width: number;
  warpWidth: number;
  segmentBytes?: number;
  elemBytes?: number;
}

/** The N-tile load of one block in one phase of the tiled matmul, warp by warp. */
export function tileLoad(q: TileLoadParams): { threads: TileThread[]; warps: TileWarp[] } {
  const seg = q.segmentBytes ?? 32;
  const eb = q.elemBytes ?? 4;
  const threads: TileThread[] = [];
  for (let t = 0; t < q.T * q.T; t++) {
    const tx = t % q.T;
    const ty = Math.floor(t / q.T);
    const r = q.ph * q.T + (q.arrangement === 'ch5' ? ty : tx);
    const c = q.bx * q.T + (q.arrangement === 'ch5' ? tx : ty);
    const element = q.layout === 'row' ? r * q.Width + c : c * q.Width + r;
    threads.push({
      t, tx, ty,
      warp: Math.floor(t / q.warpWidth),
      lane: t % q.warpWidth,
      r, c, element,
      byte: element * eb,
      sharedRow: r - q.ph * q.T,
      sharedCol: c - q.bx * q.T,
    });
  }
  const warps: TileWarp[] = [];
  for (let w = 0; w * q.warpWidth < threads.length; w++) {
    const ts = threads.slice(w * q.warpWidth, (w + 1) * q.warpWidth);
    const addrs = ts.map((x) => x.byte);
    warps.push({ warp: w, threads: ts, access: warpAccess(addrs, eb, seg), runs: runs(addrs, eb) });
  }
  return { threads, warps };
}

// ---------------------------------------------------------------- Exercise 1 (pp. 154–155)

export type Ex1Verdict = 'coalesced' | 'uncoalesced' | 'not applicable';

export interface Ex1Access {
  id: 'a' | 'b' | 'c' | 'd' | 'e' | 'f' | 'g' | 'h' | 'i';
  /** Line of foo_kernel as the book numbers it. */
  line: number;
  code: string;
  space: 'global' | 'shared';
  verdict: Ex1Verdict;
  /** For global accesses: the warp's transactions and efficiency in the chosen model. */
  transactions?: number;
  efficiency?: number;
  runs?: number;
  /** For shared accesses: the word stride between neighbouring lanes (section 6.4 counts the banks). */
  sharedStride?: number;
}

const EX1: { id: Ex1Access['id']; line: number; code: string; expr?: ExprId; sharedStride?: number }[] = [
  { id: 'a', line: 5, code: 'a[i]', expr: 'ex1-a' },
  { id: 'b', line: 5, code: 'a_s[threadIdx.x]', sharedStride: 1 },
  { id: 'c', line: 7, code: 'b[j*blockDim.x*gridDim.x + i]', expr: 'ex1-c' },
  { id: 'd', line: 7, code: 'c[i*4 + j]', expr: 'ex1-d' },
  { id: 'e', line: 7, code: 'bc_s[j*256 + threadIdx.x]', sharedStride: 1 },
  { id: 'f', line: 10, code: 'a_s[threadIdx.x]', sharedStride: 1 },
  { id: 'g', line: 10, code: 'd[i + 8]', expr: 'ex1-g' },
  { id: 'h', line: 11, code: 'bc_s[threadIdx.x*4]', sharedStride: 4 },
  { id: 'i', line: 11, code: 'e[i*8]', expr: 'ex1-i' },
];

/**
 * The nine accesses of foo_kernel, classified. A global access is coalesced when
 * the warp asks for one run of consecutive bytes (the book's rule, p. 126); shared
 * memory needs no coalescing (p. 130). Counts are for the warp whose lane 0 has
 * i = lane0, with blockDim.x·gridDim.x = gridThreads.
 */
export function exercise1(warpWidth = 32, segmentBytes = 32, j = 0, lane0 = 0, gridThreads = 1024): Ex1Access[] {
  return EX1.map((x) => {
    if (!x.expr) return { id: x.id, line: x.line, code: x.code, space: 'shared', verdict: 'not applicable', sharedStride: x.sharedStride }; // i18n-ignore: Ex1Verdict union member; S61 translates it
    const p: AccessParams = { Width: 0, k: 0, row: 0, stride: 1, offset: 0, warpWidth, lane0, j, gridThreads };
    const a = exprAccess(x.expr, p, segmentBytes);
    return {
      id: x.id,
      line: x.line,
      code: x.code,
      space: 'global',
      verdict: a.runs === 1 ? 'coalesced' : 'uncoalesced',
      transactions: a.transactions,
      efficiency: a.efficiency,
      runs: a.runs,
    };
  });
}
