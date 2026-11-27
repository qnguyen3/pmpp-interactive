// The two thread-to-output mappings of chapter 15's GEMM kernel, and what one warp's
// store instruction writes under each (book §15.3 and §15.5, pp. 352–361, Fig. 15.10;
// notes/ch15.md).
//
// What it models, as plain index arithmetic:
//   - the §15.3 mapping (Fig. 15.3 lines 09–13): thread t owns the 8 x 8 patch whose
//     corner is (8·(t / 16), 8·(t % 16)) in the 128 x 128 block tile;
//   - the §15.5 mapping (Fig. 15.10) for a warp width W of 32 or 64, with the formulas
//     lab 15.3 uses:
//         warp = t / W, lane = t % W, LC = W / 8 lanes per lane row,
//         lRow = 4·(lane / LC), lCol = 4·(lane % LC),
//         quadrants QR = 32 rows by QC = 4·LC columns, WC = 128 / (2·QC) warp tiles per row,
//         wRow = 64·(warp / WC), wCol = 2·QC·(warp % WC),
//         sub-tile (qr, qc) starts at (wRow + qr·QR + lRow, wCol + qc·QC + lCol).
//     W = 32 is the book's layout (8 warps as 2 x 4 tiles of 64 x 32, lanes 8 x 4,
//     quadrants 32 x 16, p. 360). W = 64 is our derivation for 64-wide wavefronts
//     (notes/today/ch15-facts.md B1): 4 wavefronts as 2 x 2 tiles of 64 x 64, lanes 8 x 8,
//     quadrants 32 x 32. A thread's logical 8 x 8 tile has element (r, c) in sub-tile
//     (r / 4, c / 4) at offset (r % 4, c % 4) (notes, Exercise 3 sketch);
//   - who owns any element, the inverse of both mappings;
//   - the bytes one warp's store instruction writes for three store patterns: scalar
//     stores from 8 x 8 tiles (§15.3, Fig. 15.7), float4 stores from 8 x 8 tiles (the
//     book's Exercise 2) and float4 stores from four 4 x 4 sub-tiles (§15.5, Exercise 3),
//     and the aligned segments those bytes touch, counted with src/sim/coalesce.ts:
//     NVIDIA's 32-byte sectors (ch06-facts A2), or AMD's 64- or 128-byte cache lines
//     (ch06-facts B1) as options.
//
// What it leaves out: one instruction at a time, so a cache that merges the partly
// written segments of different instructions is not modelled; no timing; Apple documents
// no coalescing rule (ch06-facts C1), so there is no Apple segment size. C starts on a
// 256-byte boundary (what cudaMalloc guarantees, ch06-facts A3) and the block is block
// (0, 0). A float4 store needs a 16-byte-aligned address (ch06-facts A11); the kernel
// takes the vector path only when N is a multiple of 4 and stores scalars otherwise
// (our choice, the labs' rule), so the vector patterns fall back to scalar stores then.

import { warpAccess, type WarpAccess } from './coalesce';
import { tr } from '../lib/locale.ts';

/** The book's running example (notes, "Running example"). */
export const BM = 128;
export const BN = 128;
export const BK = 8;
export const TM = 8;
export const TN = 8;
export const THREADS = 256;
/** Thread tiles per row of the block tile in the §15.3 mapping: bN / tN. */
export const TILES_PER_ROW = BN / TN;

export type Width = 32 | 64;
/** 'coarse': §15.3's 8 x 8 patch per thread. 'remap': §15.5's four 4 x 4 sub-tiles. */
export type Layout = 'coarse' | 'remap';

// ---------------------------------------------------------------- the §15.5 geometry

export interface RemapGeometry {
  W: Width;
  /** Warps per block: 256 / W. */
  warps: number;
  /** Lanes per lane row: W / 8. */
  LC: number;
  /** Lane rows per warp: always 8. */
  laneRows: number;
  /** Quadrant size: QR rows by QC columns. */
  QR: number;
  QC: number;
  /** Warp tiles per row and per column of the block tile. */
  WC: number;
  WR: number;
  /** Warp tile size: 64 rows by 2·QC columns. */
  warpRows: number;
  warpCols: number;
  /** Contiguous bytes one lane row writes with one float4 store each. */
  laneRowBytes: number;
}

export function remapGeometry(W: Width): RemapGeometry {
  const LC = W / 8;
  const QR = 32;
  const QC = 4 * LC;
  const warpCols = 2 * QC;
  const WC = BN / warpCols;
  const warps = THREADS / W;
  return { W, warps, LC, laneRows: 8, QR, QC, WC, WR: warps / WC, warpRows: 2 * QR, warpCols, laneRowBytes: LC * 16 };
}

export interface SubTile {
  qr: number;
  qc: number;
  /** Top-left element of the 4 x 4 sub-tile in the block tile. */
  row: number;
  col: number;
}

export interface RemapThread {
  t: number;
  warp: number;
  lane: number;
  lRow: number;
  lCol: number;
  wRow: number;
  wCol: number;
  /** The four sub-tiles, in the order (0,0), (0,1), (1,0), (1,1). */
  subTiles: SubTile[];
}

/** §15.5: where thread t's four 4 x 4 sub-tiles sit in the block tile. */
export function remapThread(t: number, W: Width): RemapThread {
  const g = remapGeometry(W);
  const warp = Math.floor(t / W);
  const lane = t % W;
  const lRow = Math.floor(lane / g.LC) * 4;
  const lCol = (lane % g.LC) * 4;
  const wRow = Math.floor(warp / g.WC) * g.warpRows;
  const wCol = (warp % g.WC) * g.warpCols;
  const subTiles: SubTile[] = [];
  for (let qr = 0; qr < 2; qr++)
    for (let qc = 0; qc < 2; qc++) subTiles.push({ qr, qc, row: wRow + qr * g.QR + lRow, col: wCol + qc * g.QC + lCol });
  return { t, warp, lane, lRow, lCol, wRow, wCol, subTiles };
}

export interface CoarseThread {
  t: number;
  ty: number;
  tx: number;
  tRow: number;
  tCol: number;
}

/** §15.3 (Fig. 15.3 lines 09–13): thread t's 8 x 8 patch. */
export function coarseThread(t: number): CoarseThread {
  const ty = Math.floor(t / TILES_PER_ROW);
  const tx = t % TILES_PER_ROW;
  return { t, ty, tx, tRow: ty * TM, tCol: tx * TN };
}

/** Block-tile element of thread t's logical element (r, c), 0 ≤ r, c < 8. */
export function elementOf(layout: Layout, W: Width, t: number, r: number, c: number): { row: number; col: number } {
  if (layout === 'coarse') {
    const ct = coarseThread(t);
    return { row: ct.tRow + r, col: ct.tCol + c };
  }
  const rt = remapThread(t, W);
  const s = rt.subTiles[Math.floor(r / 4) * 2 + Math.floor(c / 4)];
  return { row: s.row + (r % 4), col: s.col + (c % 4) };
}

export interface Owner {
  t: number;
  warp: number;
  lane: number;
  /** The element's place in the owner's logical 8 x 8 tile. */
  r: number;
  c: number;
  /** Quadrant of the owner's sub-tile (§15.5); in the §15.3 layout, the 4 x 4 corner of the patch. */
  qr: number;
  qc: number;
}

/** Who owns element (row, col) of the block tile: the inverse of the mapping. */
export function ownerOf(layout: Layout, W: Width, row: number, col: number): Owner {
  if (layout === 'coarse') {
    const t = Math.floor(row / TM) * TILES_PER_ROW + Math.floor(col / TN);
    const r = row % TM;
    const c = col % TN;
    return { t, warp: Math.floor(t / W), lane: t % W, r, c, qr: Math.floor(r / 4), qc: Math.floor(c / 4) };
  }
  const g = remapGeometry(W);
  const warp = Math.floor(row / g.warpRows) * g.WC + Math.floor(col / g.warpCols);
  const rr = row % g.warpRows;
  const cc = col % g.warpCols;
  const qr = Math.floor(rr / g.QR);
  const qc = Math.floor(cc / g.QC);
  const r2 = rr % g.QR;
  const c2 = cc % g.QC;
  const lane = Math.floor(r2 / 4) * g.LC + Math.floor(c2 / 4);
  return { t: warp * W + lane, warp, lane, r: qr * 4 + (r2 % 4), c: qc * 4 + (c2 % 4), qr, qc };
}

/** How many threads write each element of the block tile (every entry should be 1). */
export function ownershipCounts(layout: Layout, W: Width): Uint8Array {
  const n = new Uint8Array(BM * BN);
  for (let t = 0; t < THREADS; t++)
    for (let r = 0; r < TM; r++)
      for (let c = 0; c < TN; c++) {
        const e = elementOf(layout, W, t, r, c);
        n[e.row * BN + e.col]++;
      }
  return n;
}

// ---------------------------------------------------------------- stores

export type StorePattern = 'scalar8x8' | 'float4-8x8' | 'remap4x4';

export interface PatternInfo {
  id: StorePattern;
  label: string;
  layout: Layout;
  /** Asks for float4 stores (taken only when N is a multiple of 4). */
  vector: boolean;
  /** Where the book has it. */
  book: string;
}

export const PATTERNS: Record<StorePattern, PatternInfo> = {
  scalar8x8: {
    id: 'scalar8x8',
    get label() {
      return tr('8 × 8 tiles, scalar stores', 'ô 8 × 8, ghi vô hướng');
    },
    layout: 'coarse',
    vector: false,
    book: '§15.3, Fig. 15.7',
  },
  'float4-8x8': {
    id: 'float4-8x8',
    get label() {
      return tr('8 × 8 tiles, float4 stores', 'ô 8 × 8, ghi float4');
    },
    layout: 'coarse',
    vector: true,
    get book() {
      return tr('Exercise 2', 'Bài tập 2');
    },
  },
  remap4x4: {
    id: 'remap4x4',
    get label() {
      return tr('four 4 × 4 sub-tiles, float4 stores', 'bốn ô con 4 × 4, ghi float4');
    },
    layout: 'remap',
    vector: true,
    get book() {
      return tr('§15.5, Exercise 3', '§15.5, Bài tập 3');
    },
  },
};
export const PATTERN_ORDER: StorePattern[] = ['scalar8x8', 'float4-8x8', 'remap4x4'];

/** The float4 path needs every row to start on a 16-byte boundary: N a multiple of 4. */
export const vectorPath = (pattern: StorePattern, N: number) => PATTERNS[pattern].vector && N % 4 === 0;

/** The logical columns that start a store instruction: 0 and 4 for float4 stores, 0 to 7 for scalars. */
export const storeColumns = (pattern: StorePattern, N: number) => (vectorPath(pattern, N) ? [0, 4] : [0, 1, 2, 3, 4, 5, 6, 7]);

/** Store instructions each thread executes for its tile. */
export const instructionsPerThread = (pattern: StorePattern, N: number) => (vectorPath(pattern, N) ? 16 : 64);

export interface StoreSpec {
  pattern: StorePattern;
  W: Width;
  /** Columns of C (its leading dimension). */
  N: number;
  warp: number;
  /** The instruction: logical row (0 to 7) and first logical column of the thread's 8 x 8 tile. */
  row: number;
  col: number;
  /** 32 (NVIDIA sectors), 64 or 128 (AMD cache lines). */
  segmentBytes: number;
}

export interface StoreLane {
  lane: number;
  t: number;
  /** Matrix row and column of the first element the lane writes. */
  row: number;
  col: number;
  /** Byte address in C. */
  byte: number;
}

export interface StoreResult extends WarpAccess {
  lanes: StoreLane[];
  /** float4 stores were used. */
  vector: boolean;
  /** float4 asked for, but N rules it out: scalar stores instead. */
  fallback: boolean;
  bytesPerLane: number;
  /** Distinct matrix rows the warp writes, ascending. */
  rows: number[];
  /** Separate runs of consecutive bytes. */
  runs: number;
}

/** One store instruction of one warp. */
export function storeInstruction(s: StoreSpec): StoreResult {
  const info = PATTERNS[s.pattern];
  const vector = vectorPath(s.pattern, s.N);
  const col = vector ? s.col - (s.col % 4) : s.col;
  const bytesPerLane = vector ? 16 : 4;
  const lanes: StoreLane[] = [];
  for (let lane = 0; lane < s.W; lane++) {
    const t = s.warp * s.W + lane;
    const e = elementOf(info.layout, s.W, t, s.row, col);
    lanes.push({ lane, t, row: e.row, col: e.col, byte: (e.row * s.N + e.col) * 4 });
  }
  const acc = warpAccess(lanes.map((l) => l.byte), bytesPerLane, s.segmentBytes);
  const sorted = lanes.map((l) => l.byte).sort((a, b) => a - b);
  let runs = 0;
  let end = -1;
  for (const b of sorted) {
    if (b > end) runs++;
    end = Math.max(end, b + bytesPerLane);
  }
  return {
    ...acc,
    lanes,
    vector,
    fallback: info.vector && !vector,
    bytesPerLane,
    rows: [...new Set(lanes.map((l) => l.row))].sort((a, b) => a - b),
    runs,
  };
}

export interface RowStrip {
  row: number;
  /** First segment index of the span, and one past the last. */
  seg0: number;
  seg1: number;
  /** Bytes written in each segment of the span. */
  used: number[];
  /** What each lane writes, as [lo, hi) byte intervals. */
  pieces: { lane: number; lo: number; hi: number }[];
}

/** The instruction's bytes grouped by matrix row: each row is one stretch of memory. */
export function rowStrips(res: StoreResult, segmentBytes: number): RowStrip[] {
  return res.rows.map((row) => {
    const mine = res.lanes.filter((l) => l.row === row);
    const pieces = mine.map((l) => ({ lane: l.lane, lo: l.byte, hi: l.byte + res.bytesPerLane })).sort((a, b) => a.lo - b.lo);
    const seg0 = Math.floor(pieces[0].lo / segmentBytes);
    const seg1 = Math.floor((pieces[pieces.length - 1].hi - 1) / segmentBytes) + 1;
    const used = Array.from({ length: seg1 - seg0 }, (_, k) => {
      const lo = (seg0 + k) * segmentBytes;
      const hi = lo + segmentBytes;
      return pieces.reduce((n, p) => n + Math.max(0, Math.min(hi, p.hi) - Math.max(lo, p.lo)), 0);
    });
    return { row, seg0, seg1, used, pieces };
  });
}

export interface WarpTileTotal {
  instructions: number;
  /** Segments summed over all of the warp's store instructions (no cache merging). */
  transactions: number;
  bytes: number;
  /** The fewest segments those bytes could occupy. */
  minimum: number;
}

/** Every store instruction of one warp for its whole tile, summed. */
export function warpTileTotal(pattern: StorePattern, W: Width, N: number, warp: number, segmentBytes: number): WarpTileTotal {
  let transactions = 0;
  let instructions = 0;
  for (let row = 0; row < TM; row++)
    for (const col of storeColumns(pattern, N)) {
      transactions += storeInstruction({ pattern, W, N, warp, row, col, segmentBytes }).transactions;
      instructions++;
    }
  const bytes = W * TM * TN * 4;
  return { instructions, transactions, bytes, minimum: bytes / segmentBytes };
}

/** The settings the lesson asks the reader to classify: the book's warp, N a multiple of 8, NVIDIA's sectors. */
export const DEFAULT_STORE: Omit<StoreSpec, 'pattern'> = { W: 32, N: 1024, warp: 0, row: 0, col: 0, segmentBytes: 32 };

/** "Coalesced" in the lesson's sense: every byte of every segment touched is a byte asked for. */
export const coalesced = (pattern: StorePattern) => storeInstruction({ ...DEFAULT_STORE, pattern }).efficiency === 1;
