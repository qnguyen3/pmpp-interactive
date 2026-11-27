// Chapter 13, section 13.6 (pp. 314 to 320): a block-and-tile model of the book's
// tiled merge kernel (Figs. 13.11 to 13.13), and of the exact-load design the book
// leaves as Exercise 3 (p. 316) and that CUB, rocPRIM, MLX and PyTorch MPS ship
// (notes/today/ch13-facts.md A7, B2, D1, D2).
//
// It is independent of sim/corank.ts on purpose. Every split point here comes from a
// stable sequential merge: "how many of the first k outputs come from A". That count
// is the co-rank by definition (p. 308), and ch13-facts E1 checks that the book's
// binary search, CUB's MergePath and this count agree. So the two models check each
// other; neither borrows the other's search.
//
// What the book scheme does, per block (Figs. 13.11 to 13.13):
//  - The block's slice of C is [cCurr, cNext), cCurr = block * ceil((m+n)/blocks);
//    its slices of A and B come from the co-ranks of cCurr and cNext.
//  - Each iteration loads up to `tile` elements of A and up to `tile` of B into shared
//    memory, starting at A_curr + A_consumed and B_curr + B_consumed. The `if` in the
//    load loops (lines 27 and 32, p. 320) stops a load at the end of the block's slice.
//  - The threads write min(tile, C_length - C_completed) outputs, each thread its own
//    tile/threads of them, from co-ranks taken on the two tiles.
//  - A_consumed grows by the number of A elements those outputs used.
//
// The printed formula ('printed') keeps the listing's `ceil(int / int)`: the division
// truncates first, so ceil changes nothing (notes item 14, ch13-facts A4). It applies
// to the block split (lines 05 and 06) and to the iteration count (line 21). With the
// book's own numbers it runs 3 iterations where the text says 4 (p. 317).
//
// Where this model departs from the listing, on purpose (the lesson must say so):
//  - Line 48 advances A_consumed by co_rank(tile_size, A_S, tile_size, B_S, tile_size),
//    which treats every slot of both tiles as loaded. The book says this is wrong only
//    after the last iteration, when it no longer matters (p. 320). It is also wrong on
//    any earlier iteration whose A or B tile is only partly loaded, because the search
//    can then read slots that still hold the previous tile (our own emulation of the
//    listing, not the book's statement; see `staleSlots` below and the preset
//    'shortTileEarly'). The model advances A_consumed by what the iteration's merges
//    actually used, which is what the circular-buffer kernel's line 50 computes.
//  - A block whose slice would start past the end of C gets an empty slice (the
//    listing does not clamp C_curr; the launch configuration keeps it in range).
//
// What it leaves out: time, warps, coalescing, bank conflicts and the cost of the
// searches themselves (only `extraSearchReads` counts reads, for the exact scheme).
// Elements are counted, not bytes.
//
// Erasable TypeScript only, and no imports: the browser checks import this file
// directly in Node.

import { tr } from '../lib/locale.ts';

export type Src = 'A' | 'B';
/** One element with its origin, so that ties stay visible (A1 is A[1]). */
export interface Item {
  key: number;
  src: Src;
  idx: number;
}
export type Formula = 'printed' | 'corrected';
/** The tie rule of merge_sequential (Fig. 13.2, line 06). '<=' is the book's: A wins ties. */
export type TieRule = '<=' | '<';

/** Fig. 13.2 as a function of the comparison. With '<=' it is the book's stable merge. */
export function mergeWith(A: readonly number[], B: readonly number[], rule: TieRule): Item[] {
  const out: Item[] = [];
  let i = 0;
  let j = 0;
  while (i < A.length && j < B.length) {
    const takeA = rule === '<=' ? A[i] <= B[j] : A[i] < B[j];
    if (takeA) {
      out.push({ key: A[i], src: 'A', idx: i });
      i++;
    } else {
      out.push({ key: B[j], src: 'B', idx: j });
      j++;
    }
  }
  for (; j < B.length; j++) out.push({ key: B[j], src: 'B', idx: j });
  for (; i < A.length; i++) out.push({ key: A[i], src: 'A', idx: i });
  return out;
}

/** The book's merge: stable within each list and across lists (pp. 304 to 305). */
export const mergeStable = (A: readonly number[], B: readonly number[]): Item[] => mergeWith(A, B, '<=');

/** "A1", "B0": how the lessons tag an element. */
export const tag = (it: Item): string => it.src + it.idx;

/**
 * The co-rank of k, computed by merging: how many of the first k outputs of the stable
 * merge come from A. k may be 0 to m + n (Observation 2, p. 307).
 */
export function coRankByMerge(k: number, A: readonly number[], B: readonly number[]): number {
  if (k < 0 || k > A.length + B.length) throw new RangeError(`k = ${k} is outside 0..${A.length + B.length}`);
  let i = 0;
  let j = 0;
  for (let t = 0; t < k; t++) {
    if (j >= B.length || (i < A.length && A[i] <= B[j])) i++;
    else j++;
  }
  return i;
}

/**
 * A merge-path binary search for the same number (the form CUB and rocPRIM use,
 * ch13-facts A7, B2): find the first a with B[k-1-a] < A[a]. Returns the co-rank and
 * how many array elements it read (two per probe). Used to count the extra search of
 * the exact-load scheme.
 */
export function mergePathSearch(k: number, A: readonly number[], B: readonly number[]): { i: number; reads: number } {
  const m = A.length;
  const n = B.length;
  if (k < 0 || k > m + n) throw new RangeError(`k = ${k} is outside 0..${m + n}`);
  let lo = Math.max(0, k - n);
  let hi = Math.min(k, m);
  let reads = 0;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    reads += 2;
    if (B[k - 1 - mid] < A[mid]) hi = mid;
    else lo = mid + 1;
  }
  return { i: lo, reads };
}

/** Iterations of the while-loop (Fig. 13.12, line 21), as printed or as the text means. */
export function iterationsFor(cLength: number, tile: number, formula: Formula): number {
  if (cLength <= 0) return 0;
  return formula === 'printed' ? Math.floor(cLength / tile) : Math.ceil(cLength / tile);
}

/** Outputs per block (Fig. 13.11, lines 05 and 06), as printed or as the text means. */
export function outputsPerBlock(total: number, blocks: number, formula: Formula): number {
  return formula === 'printed' ? Math.floor(total / blocks) : Math.ceil(total / blocks);
}

export interface TileIteration {
  /** Global index of the first A element loaded: A_curr + A_consumed. */
  aStart: number;
  bStart: number;
  /** Elements loaded into A_S and B_S this iteration (the tail guard caps them). */
  aLoaded: number;
  bLoaded: number;
  /** Of those, how many this iteration's outputs used. */
  aUsed: number;
  bUsed: number;
  /** Global index of the first output written: C_curr + C_completed. */
  cStart: number;
  /** Outputs written: min(tile, C_length - C_completed). */
  cOut: number;
  /** For each thread, how many of A and of B it merged (Fig. 13.13, lines 37 to 45). */
  threads: { a: number; b: number }[];
  /**
   * Tile slots that the printed line 48 treats as loaded although this iteration did
   * not load them: non-zero when an iteration that is not the block's last has a partly
   * loaded tile. Those slots still hold the previous tile (or nothing, on the first),
   * and its binary search may read them.
   */
  staleSlots: number;
}

export interface TiledBlock {
  block: number;
  cCurr: number;
  cNext: number;
  aCurr: number;
  aNext: number;
  bCurr: number;
  bNext: number;
  iterations: TileIteration[];
  /** Outputs of the block's slice that no iteration wrote (only the printed formula leaves any). */
  missed: number;
}

export interface Totals {
  /** Elements copied from global memory into shared memory. */
  loaded: number;
  /** Elements that went into an output. */
  used: number;
  /** m + n: every input element has to be read once. */
  minimum: number;
}

export interface TiledRun {
  blocks: TiledBlock[];
  totals: Totals & { missed: number };
  /** The merged array; null where no thread ever wrote. */
  output: (Item | null)[];
}

/** Wrap a slice of an input array as tagged items, so the tile merges keep their origin. */
const items = (keys: readonly number[], src: Src, from: number, count: number): Item[] =>
  Array.from({ length: count }, (_, t) => ({ key: keys[from + t], src, idx: from + t }));

/** coRankByMerge on tagged items (the tile contents). */
function coRankItems(k: number, A: readonly Item[], B: readonly Item[]): number {
  let i = 0;
  let j = 0;
  for (let t = 0; t < k; t++) {
    if (j >= B.length || (i < A.length && A[i].key <= B[j].key)) i++;
    else j++;
  }
  return i;
}

/** merge_sequential on tagged items, writing into `out` from `at`. */
function mergeItemsInto(A: readonly Item[], B: readonly Item[], out: (Item | null)[], at: number) {
  let i = 0;
  let j = 0;
  let k = at;
  while (i < A.length && j < B.length) out[k++] = A[i].key <= B[j].key ? A[i++] : B[j++];
  while (j < B.length) out[k++] = B[j++];
  while (i < A.length) out[k++] = A[i++];
}

/**
 * The book's tiled kernel, one block after another. `threads` is blockDim.x and must
 * divide `tile` (the listing assumes it, Fig. 13.13 lines 37 and 38).
 */
export function bookTiled(
  A: readonly number[],
  B: readonly number[],
  blocks: number,
  tile: number,
  formula: Formula,
  threads = 1,
): TiledRun {
  if (!(blocks >= 1 && tile >= 1 && threads >= 1)) throw new RangeError('blocks, tile and threads must be at least 1');
  if (tile % threads !== 0) throw new RangeError(`tile (${tile}) must be a multiple of threads (${threads})`);
  const m = A.length;
  const n = B.length;
  const total = m + n;
  const output: (Item | null)[] = new Array(total).fill(null);
  const per = outputsPerBlock(total, blocks, formula);
  const perThread = tile / threads;
  const out: TiledBlock[] = [];
  let loaded = 0;
  let used = 0;
  let missed = 0;

  for (let b = 0; b < blocks; b++) {
    const cCurr = Math.min(b * per, total);
    const cNext = Math.min((b + 1) * per, total);
    const aCurr = coRankByMerge(cCurr, A, B);
    const aNext = coRankByMerge(cNext, A, B);
    const bCurr = cCurr - aCurr;
    const bNext = cNext - aNext;
    const cLength = cNext - cCurr;
    const aLength = aNext - aCurr;
    const bLength = bNext - bCurr;
    const count = iterationsFor(cLength, tile, formula);
    const its: TileIteration[] = [];
    let cDone = 0;
    let aDone = 0;
    let bDone = 0;
    for (let it = 0; it < count; it++) {
      const aLoaded = Math.max(0, Math.min(tile, aLength - aDone));
      const bLoaded = Math.max(0, Math.min(tile, bLength - bDone));
      const aS = items(A, 'A', aCurr + aDone, aLoaded);
      const bS = items(B, 'B', bCurr + bDone, bLoaded);
      const cOut = Math.min(tile, cLength - cDone);
      const ths: { a: number; b: number }[] = [];
      for (let t = 0; t < threads; t++) {
        const c0 = Math.min(t * perThread, cOut);
        const c1 = Math.min((t + 1) * perThread, cOut);
        const a0 = coRankItems(c0, aS, bS);
        const a1 = coRankItems(c1, aS, bS);
        const b0 = c0 - a0;
        const b1 = c1 - a1;
        mergeItemsInto(aS.slice(a0, a1), bS.slice(b0, b1), output, cCurr + cDone + c0);
        ths.push({ a: a1 - a0, b: b1 - b0 });
      }
      const aUsed = coRankItems(cOut, aS, bS);
      const bUsed = cOut - aUsed;
      const last = it === count - 1;
      its.push({
        aStart: aCurr + aDone,
        bStart: bCurr + bDone,
        aLoaded,
        bLoaded,
        aUsed,
        bUsed,
        cStart: cCurr + cDone,
        cOut,
        threads: ths,
        staleSlots: last ? 0 : tile - aLoaded + (tile - bLoaded),
      });
      loaded += aLoaded + bLoaded;
      used += cOut;
      cDone += cOut;
      aDone += aUsed;
      bDone += bUsed;
    }
    const blockMissed = cLength - cDone;
    missed += blockMissed;
    out.push({ block: b, cCurr, cNext, aCurr, aNext, bCurr, bNext, iterations: its, missed: blockMissed });
  }
  // Outputs past the last block's slice (the printed block split can stop short of m + n).
  missed += total - out[out.length - 1].cNext;
  return { blocks: out, totals: { loaded, used, minimum: total, missed }, output };
}

export interface ExactTile {
  tile: number;
  /** Half-open ranges [start, end) in C, A and B. */
  cRange: [number, number];
  aRange: [number, number];
  bRange: [number, number];
  /** Exactly the elements this tile consumes: loaded = used. */
  loaded: number;
  used: number;
  /** Elements read by the merge-path search for this tile's end, on global memory. */
  extraSearchReads: number;
}

export interface ExactRun {
  tiles: ExactTile[];
  totals: Totals & { extraSearchReads: number };
  output: Item[];
}

/**
 * The exact-load scheme: find the co-rank of every tile boundary first (one search per
 * boundary, on global memory), then load exactly the A and B elements each tile will
 * consume. Exercise 3 (p. 328), the alternative of p. 316, and what the libraries do.
 */
export function exactTiled(A: readonly number[], B: readonly number[], tileOutputs: number): ExactRun {
  if (!(tileOutputs >= 1)) throw new RangeError('tileOutputs must be at least 1');
  const total = A.length + B.length;
  const output: (Item | null)[] = new Array(total).fill(null);
  const tiles: ExactTile[] = [];
  let aStart = 0;
  let loaded = 0;
  let extra = 0;
  for (let t = 0, c0 = 0; c0 < total; t++, c0 += tileOutputs) {
    const c1 = Math.min(c0 + tileOutputs, total);
    const { i: aEnd, reads } = mergePathSearch(c1, A, B);
    const bStart = c0 - aStart;
    const bEnd = c1 - aEnd;
    mergeItemsInto(items(A, 'A', aStart, aEnd - aStart), items(B, 'B', bStart, bEnd - bStart), output, c0);
    tiles.push({
      tile: t,
      cRange: [c0, c1],
      aRange: [aStart, aEnd],
      bRange: [bStart, bEnd],
      loaded: c1 - c0,
      used: c1 - c0,
      extraSearchReads: reads,
    });
    loaded += c1 - c0;
    extra += reads;
    aStart = aEnd;
  }
  return { tiles, totals: { loaded, used: loaded, minimum: total, extraSearchReads: extra }, output: output as Item[] };
}

/** Sorted pseudo-random keys in [0, maxKey), reproducible from the seed (mulberry32). */
export function sortedRandom(count: number, maxKey: number, seed: number): number[] {
  let s = seed >>> 0;
  const next = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let x = s;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
  return Array.from({ length: count }, () => Math.floor(next() * maxKey)).sort((a, b) => a - b);
}

/** Build A and B from a pattern of sources, one letter per output, with keys from `first` up. */
export function fromPattern(pattern: string, first = 10): { A: number[]; B: number[] } {
  const A: number[] = [];
  const B: number[] = [];
  [...pattern].forEach((s, p) => (s === 'A' ? A : B).push(first + p));
  return { A, B };
}

/** The book's running example (Fig. 13.1, p. 304). */
export const RUNNING = { A: [1, 7, 8, 9, 10], B: [7, 10, 10, 12] } as const;

/** The book's example launch (p. 317): m = 33,000, n = 31,000, 16 blocks of 128 threads, tile 1,024. */
export const BOOK_EXAMPLE = { m: 33000, n: 31000, blocks: 16, threads: 128, tile: 1024 } as const;

export interface Preset {
  label: string;
  A: number[];
  B: number[];
  blocks: number;
  tile: number;
  threads: number;
}

// Scene and explorer presets: one block, tile 8, two threads of four outputs each.
// 'uneven': four iterations that use 5+3, 2+6, 7+1 and 2+6 elements of A and B.
// 'allFromA': the first two iterations take every output from A while B still has a
// full tile waiting, so each loads 16 and uses 8: the case that forces 2 x tile.
// 'shortTileEarly': A runs out in iteration 1 of 5, so that iteration loads a partial
// A tile and the printed line 48 would read 6 slots left over from iteration 0.
export const PRESETS: Record<'uneven' | 'allFromA' | 'shortTileEarly', Preset> = {
  uneven: {
    get label() {
      return tr('uneven interleaving', 'xen kẽ không đều');
    },
    ...fromPattern('AABABAAB' + 'BABBBABB' + 'AAABAAAA' + 'BBABBABB'),
    blocks: 1,
    tile: 8,
    threads: 2,
  },
  allFromA: {
    get label() {
      return tr('worst case: whole tiles from A', 'trường hợp xấu nhất: các ô lấy trọn từ A');
    },
    ...fromPattern('A'.repeat(16) + 'B'.repeat(8)),
    blocks: 1,
    tile: 8,
    threads: 2,
  },
  shortTileEarly: {
    get label() {
      return tr('A runs out early', 'A hết sớm');
    },
    ...fromPattern('A'.repeat(10) + 'B'.repeat(30), 0),
    blocks: 1,
    tile: 8,
    threads: 2,
  },
};
