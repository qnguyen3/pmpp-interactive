// Data reuse of a tiled matrix multiplication (book §15.2, pp. 350–352).
//
// One block owns an m x n tile of C, held in its threads' registers, and walks the
// inner dimension in steps of k: each step loads an m x k tile of A and the matching
// k x n tile of B from global memory into shared memory. Per step ("tile pair"):
//   operations  2·m·n·k
//   bytes       e·k·(m + n)      (e bytes per element; the book uses 4)
//   intensity   2·m·n / (e·(m + n)), which for e = 4 is the book's 0.5·m·n/(m + n).
// k cancels. The model counts input-tile loads from global memory only: the output
// tile written once at the end is left out (notes, [inferred]) and caches do not exist.

/** The book's example device threshold: 20 FLOP/B (ch. 5, from the H100's peak figures). */
export const BOOK_THRESHOLD = 20;
/** Bytes per element the explorer offers: float32, 16-bit, 8-bit, 4-bit. */
export const ELEMENT_BYTES = [4, 2, 1, 0.5] as const;

export interface TilePair {
  m: number;
  n: number;
  k: number;
  /** Bytes per element. */
  bytes: number;
  /** Floating-point operations for one tile pair: 2·m·n·k. */
  ops: number;
  /** Bytes loaded from global memory for one tile pair: e·k·(m + n). */
  traffic: number;
  /** ops / traffic, in FLOP/B. */
  intensity: number;
}

const pos = (v: number, what: string) => {
  if (!(v > 0) || !isFinite(v)) throw new Error(`${what} must be a positive number, got ${v}`);
};

export function tilePair(m: number, n: number, k: number, bytes = 4): TilePair {
  pos(m, 'm');
  pos(n, 'n');
  pos(k, 'k');
  pos(bytes, 'bytes per element'); // i18n-ignore: developer error text for an invalid argument
  const ops = 2 * m * n * k;
  const traffic = bytes * k * (m + n);
  return { m, n, k, bytes, ops, traffic, intensity: ops / traffic };
}

/** Intensity of an m x n output tile in FLOP/B; independent of the tile depth k. */
export const intensity = (m: number, n: number, bytes = 4) => tilePair(m, n, 1, bytes).intensity;

export type Regime = 'memory-bound' | 'compute-bound';
export const regime = (fpb: number, threshold: number): Regime => (fpb < threshold ? 'memory-bound' : 'compute-bound');

/**
 * The smallest whole tile height m (>= 1) whose m x n tile reaches `threshold`, or null
 * when no height up to `maxM` does. Solves 2mn >= t·e·(m + n), i.e. m >= t·e·n / (2n − t·e).
 */
export function smallestHeight(n: number, threshold: number, bytes = 4, maxM = Infinity): number | null {
  pos(n, 'n');
  pos(threshold, 'threshold');
  const te = threshold * bytes;
  if (2 * n <= te) return null; // even an infinitely tall tile stays below: the limit is 2n/e
  let m = Math.max(1, Math.ceil((te * n) / (2 * n - te) - 1e-9));
  while (intensity(m, n, bytes) < threshold) m += 1; // guard against rounding
  while (m > 1 && intensity(m - 1, n, bytes) >= threshold) m -= 1;
  return m <= maxM ? m : null;
}

// ---------------------------------------------------------------- the whole grid
export interface GridFetches {
  M: number;
  N: number;
  K: number;
  bM: number;
  bN: number;
  bytes: number;
  /** Blocks down the rows of C (ceil(M / bM)) and across its columns (ceil(N / bN)). */
  blocksM: number;
  blocksN: number;
  /** How many blocks fetch each element of A (one per block in its row band): ceil(N / bN). */
  aFetches: number;
  /** How many blocks fetch each element of B (one per block in its column band): ceil(M / bM). */
  bFetches: number;
  /** Bytes the grid requests from global memory for A and B, with no caches. */
  requested: number;
  /** Bytes if every input element were fetched exactly once. */
  ideal: number;
  /** 2·M·N·K. */
  ops: number;
  /** ops / requested. */
  intensity: number;
}

/** Fetch counts for C = A·B (A M x K, B K x N) computed with bM x bN block tiles. */
export function gridFetches(M: number, N: number, K: number, bM: number, bN: number, bytes = 4): GridFetches {
  for (const [v, w] of [[M, 'M'], [N, 'N'], [K, 'K'], [bM, 'bM'], [bN, 'bN'], [bytes, 'bytes']] as const) pos(v, w);
  const blocksM = Math.ceil(M / bM);
  const blocksN = Math.ceil(N / bN);
  // Every block in a row band reads the whole bM x K panel of A that lies inside the
  // matrix; every block in a column band reads the whole K x bN panel of B.
  const requested = bytes * (M * K * blocksN + K * N * blocksM);
  const ideal = bytes * (M * K + K * N);
  const ops = 2 * M * N * K;
  return { M, N, K, bM, bN, bytes, blocksM, blocksN, aFetches: blocksN, bFetches: blocksM, requested, ideal, ops, intensity: ops / requested };
}

// ---------------------------------------------------------------- the "reuse" scene
export interface RegionBlock {
  /** Order in which the scene lights the block up (row-major). */
  index: number;
  /** Block coordinates inside the region. */
  row: number;
  col: number;
  /** The band of A (rows) and band of B (columns) this block fetches. */
  aBand: number;
  bBand: number;
}

/** The blocks that cover a square region of C with square tiles, in row-major order. */
export function regionBlocks(region: number, tile: number): RegionBlock[] {
  pos(region, 'region');
  pos(tile, 'tile');
  if (region % tile) throw new Error('the region must be a whole number of tiles');
  const per = region / tile;
  const out: RegionBlock[] = [];
  for (let r = 0; r < per; r++) for (let c = 0; c < per; c++) out.push({ index: r * per + c, row: r, col: c, aBand: r, bBand: c });
  return out;
}

export interface RegionCounts {
  /** Fetches of each A band (one entry per tile row of the region). */
  aBand: number[];
  /** Fetches of each B band (one entry per tile column). */
  bBand: number[];
  /** Bytes requested from global memory for one step of k along the inner dimension. */
  requested: number;
  /** Bytes the region's data needs for that step: its A band and its B band, once each. */
  needed: number;
}

/**
 * What the blocks of a region have fetched after the first `upto` of them (row-major
 * order) each loaded one tile pair of depth k. With upto = all blocks this is one full
 * step along the inner dimension.
 */
export function regionCounts(region: number, tile: number, k: number, upto?: number, bytes = 4): RegionCounts {
  const blocks = regionBlocks(region, tile);
  const per = region / tile;
  const n = Math.max(0, Math.min(blocks.length, upto ?? blocks.length));
  const aBand = new Array<number>(per).fill(0);
  const bBand = new Array<number>(per).fill(0);
  let requested = 0;
  for (const b of blocks.slice(0, n)) {
    aBand[b.aBand] += 1;
    bBand[b.bBand] += 1;
    requested += tilePair(tile, tile, k, bytes).traffic;
  }
  return { aBand, bBand, requested, needed: bytes * k * (region + region) };
}
