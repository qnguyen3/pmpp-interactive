// Thread coarsening for the histogram, chapter 9 section 9.5 (book pp. 214–217,
// Figs. 9.11 to 9.14). Runs the "Who reads what in iteration c" explorer, the
// "Blocks, copies, commits" calculator and the numbers in the section's questions.
//
// Sources (the lesson cites them where they are used):
//  - notes/ch09.md §9.5: the block segment COARSE_FACTOR·blockDim.x, and line 12 of
//    the two listings: contiguous (Fig. 9.12) i = segment + threadIdx.x·COARSE_FACTOR + c,
//    interleaved (Fig. 9.14) i = segment + c·blockDim.x + threadIdx.x. The drawings
//    (Figs. 9.11, 9.13) use 8 threads and, by the look of them, a factor of 3.
//  - notes/ch09.md exercise 5 (p. 220): 524,288 elements, 128 bins, 1024 threads per
//    block, factor 4.
//  - notes/today/ch09-facts.md A7: NVIDIA serves a warp's global load in 32-byte
//    transactions, 32 of them in the worst case; B3: AMD merges consecutive lanes into
//    64- or 128-byte cache-line requests; C4: Apple documents no rule, so a segment size
//    for Metal is the reader's choice, not a fact.
//  - notes/today/ch05-facts.md B5: AMD warps are 64 or 32 lanes; C3: Metal reports its
//    SIMD-group width per pipeline. The warp width is always a parameter here.
//
// What this leaves out (the lesson says so beside the explorer):
//  - It counts the aligned segments one warp's load touches. No time, no caches (a
//    segment fetched by one load is not remembered by the next), no DRAM.
//  - The image starts on a segment boundary.
//  - A warp is warpWidth consecutive threads of one block, as on every vendor's
//    hardware; the last warp of a block may be partial.
//
// Erasable TypeScript and no imports: the browser check imports this file directly.

import { tr } from '../lib/locale.ts';

export type Partitioning = 'contiguous' | 'interleaved';

/** Line 12 of the two listings, as the book prints them. */
export const LINE12: Record<Partitioning, string> = {
  contiguous: 'unsigned int i = segment + threadIdx.x*COARSE_FACTOR + c;',
  interleaved: 'unsigned int i = segment + c*blockDim.x + threadIdx.x;',
};

/** The book's drawing and listing for each way of handing out elements. */
export const FIGS: Record<Partitioning, { drawing: string; code: string }> = {
  contiguous: { drawing: 'Fig. 9.11', code: 'Fig. 9.12' },
  interleaved: { drawing: 'Fig. 9.13', code: 'Fig. 9.14' },
};

/** First element of a block's segment (line 10 of Figs. 9.12 and 9.14). */
export function segmentStart(blockIdx: number, blockDim: number, coarse: number): number {
  return coarse * blockIdx * blockDim;
}

/** Line 12: the element thread `threadIdx` of block `blockIdx` reads in iteration `c`. */
export function index(p: Partitioning, blockIdx: number, threadIdx: number, c: number, blockDim: number, coarse: number): number {
  const segment = segmentStart(blockIdx, blockDim, coarse);
  return p === 'contiguous' ? segment + threadIdx * coarse + c : segment + c * blockDim + threadIdx;
}

/** The inverse of line 12: which block, thread and iteration read element `i`. */
export function owner(p: Partitioning, i: number, blockDim: number, coarse: number): { block: number; thread: number; c: number } {
  const per = blockDim * coarse;
  const block = Math.floor(i / per);
  const off = i - block * per;
  return p === 'contiguous'
    ? { block, thread: Math.floor(off / coarse), c: off % coarse }
    : { block, thread: off % blockDim, c: Math.floor(off / blockDim) };
}

/** Blocks the launch needs: ceil(n / (blockDim·COARSE_FACTOR)). */
export const blocksFor = (n: number, blockDim: number, coarse: number): number => Math.ceil(n / (blockDim * coarse));

/** Warps in one block. */
export const warpsPerBlock = (blockDim: number, warpWidth: number): number => Math.ceil(blockDim / warpWidth);

// ---------------------------------------------------------------- one warp's load

export interface Machine {
  /** Lanes per warp: a parameter, never a constant (32 NVIDIA; 64 or 32 AMD; reported per pipeline on Metal). */
  warpWidth: number;
  /** 1 for a pixel, 4 for a 32-bit token id. */
  elemBytes: number;
  /** The unit the memory system serves a warp in (ch09-facts A7, B3; Apple: a choice, C4). */
  segmentBytes: number;
}

export interface LoadParams extends Machine {
  p: Partitioning;
  /** Elements in the input. */
  n: number;
  blockDim: number;
  coarse: number;
  block: number;
  /** Which warp of the block (0 = threads 0 to warpWidth − 1). */
  warp: number;
  /** Iteration of the coarsening loop, 0 to coarse − 1. */
  c: number;
}

export interface Lane {
  lane: number;
  thread: number;
  /** Element line 12 gives this thread, in range or not. */
  element: number;
  /** False when the bounds check (element < n) fails. */
  active: boolean;
}

export interface Footprint {
  lanes: Lane[];
  active: number;
  /** Elements the active lanes read, in lane order. */
  elements: number[];
  /** Distinct aligned segments touched (byte address ÷ segmentBytes), ascending. */
  segments: number[];
  transactions: number;
  bytesNeeded: number;
  bytesFetched: number;
  /** bytesNeeded ÷ bytesFetched; 0 when no lane is active. */
  efficiency: number;
  /** Elements between neighbouring lanes: COARSE_FACTOR (contiguous) or 1 (interleaved). */
  laneStride: number;
}

/** The elements and segments one warp touches in iteration c. */
export function warpFootprint(q: LoadParams): Footprint {
  const first = q.warp * q.warpWidth;
  const last = Math.min(q.blockDim, first + q.warpWidth);
  const lanes: Lane[] = [];
  const segs = new Set<number>();
  for (let t = first; t < last; t++) {
    const element = index(q.p, q.block, t, q.c, q.blockDim, q.coarse);
    const active = element < q.n;
    lanes.push({ lane: t - first, thread: t, element, active });
    if (!active) continue;
    const a = element * q.elemBytes;
    for (let s = Math.floor(a / q.segmentBytes); s <= Math.floor((a + q.elemBytes - 1) / q.segmentBytes); s++) segs.add(s);
  }
  const act = lanes.filter((l) => l.active);
  const segments = [...segs].sort((x, y) => x - y);
  const bytesNeeded = act.length * q.elemBytes;
  const bytesFetched = segments.length * q.segmentBytes;
  return {
    lanes,
    active: act.length,
    elements: act.map((l) => l.element),
    segments,
    transactions: segments.length,
    bytesNeeded,
    bytesFetched,
    efficiency: bytesFetched ? bytesNeeded / bytesFetched : 0,
    laneStride: q.p === 'contiguous' ? q.coarse : 1,
  };
}

/** Segments of one warp's load without building the lane list (both orders are ascending in the lane). */
function loadCount(p: Partitioning, n: number, B: number, C: number, b: number, w: number, c: number, m: Machine): { active: number; transactions: number } {
  const first = w * m.warpWidth;
  const last = Math.min(B, first + m.warpWidth);
  let active = 0;
  let transactions = 0;
  let prev = -1;
  for (let t = first; t < last; t++) {
    const e = index(p, b, t, c, B, C);
    if (e >= n) continue;
    active++;
    const a = e * m.elemBytes;
    const lo = Math.floor(a / m.segmentBytes);
    const hi = Math.floor((a + m.elemBytes - 1) / m.segmentBytes);
    const from = Math.max(lo, prev + 1);
    if (hi >= from) transactions += hi - from + 1;
    prev = Math.max(prev, hi);
  }
  return { active, transactions };
}

export interface Totals {
  /** Warp-wide load instructions with at least one active lane. */
  loads: number;
  transactions: number;
  bytesNeeded: number;
  bytesFetched: number;
  efficiency: number;
  /** Fewest and most segments any one load touched. */
  minPerLoad: number;
  maxPerLoad: number;
}

/** Every load of the whole kernel: all blocks, all warps, all iterations. */
export function kernelTotals(q: Machine & { p: Partitioning; n: number; blockDim: number; coarse: number }): Totals {
  const blocks = blocksFor(q.n, q.blockDim, q.coarse);
  const warps = warpsPerBlock(q.blockDim, q.warpWidth);
  let loads = 0;
  let transactions = 0;
  let elements = 0;
  let minPerLoad = Infinity;
  let maxPerLoad = 0;
  for (let b = 0; b < blocks; b++)
    for (let w = 0; w < warps; w++)
      for (let c = 0; c < q.coarse; c++) {
        const r = loadCount(q.p, q.n, q.blockDim, q.coarse, b, w, c, q);
        if (!r.active) continue;
        loads++;
        elements += r.active;
        transactions += r.transactions;
        minPerLoad = Math.min(minPerLoad, r.transactions);
        maxPerLoad = Math.max(maxPerLoad, r.transactions);
      }
  const bytesNeeded = elements * q.elemBytes;
  const bytesFetched = transactions * q.segmentBytes;
  return {
    loads,
    transactions,
    bytesNeeded,
    bytesFetched,
    efficiency: bytesFetched ? bytesNeeded / bytesFetched : 0,
    minPerLoad: loads ? minPerLoad : 0,
    maxPerLoad,
  };
}

// ---------------------------------------------------------------- what coarsening saves

export interface Plan {
  n: number;
  blockDim: number;
  coarse: number;
  bins: number;
  /** ceil(n / (blockDim·coarse)). */
  blocks: number;
  /** Pixels one full block handles. */
  pixelsPerBlock: number;
  /** One private copy per block, each zeroed and committed once. */
  copies: number;
  /** Shared-memory writes spent zeroing the copies: blocks·bins. */
  zeroWrites: number;
  /** Block-scope atomics on the private copies: one per pixel, whatever the factor. */
  sharedAtomics: number;
  /**
   * Most device-scope atomics the commits can issue. A block commits only non-zero bins,
   * and it cannot have more non-zero bins than pixels: sum over blocks of min(bins, pixels).
   */
  commitAtomicsMax: number;
  /** Fig. 9.6, no privatisation: one global atomic per pixel. */
  unprivatisedAtomics: number;
}

export function plan(n: number, blockDim: number, coarse: number, bins: number): Plan {
  const per = blockDim * coarse;
  const blocks = blocksFor(n, blockDim, coarse);
  const lastPixels = n - (blocks - 1) * per;
  const commitAtomicsMax = blocks ? (blocks - 1) * Math.min(bins, per) + Math.min(bins, lastPixels) : 0;
  return {
    n,
    blockDim,
    coarse,
    bins,
    blocks,
    pixelsPerBlock: per,
    copies: blocks,
    zeroWrites: blocks * bins,
    sharedAtomics: n,
    commitAtomicsMax,
    unprivatisedAtomics: n,
  };
}

/** Exercise 5 (p. 220). */
export const EXERCISE5 = { n: 524_288, bins: 128, blockDim: 1024, coarse: 4 } as const;

// ---------------------------------------------------------------- the explorer's choices

export interface SegmentChoice {
  id: 'nvidia' | 'amd64' | 'amd128' | 'apple';
  label: string;
  /** null: the vendor documents none, and the reader picks one. */
  bytes: number | null;
  facts: string;
}

export const SEGMENT_CHOICES: SegmentChoice[] = [
  {
    id: 'nvidia',
    get label() {
      return tr('32 bytes: NVIDIA’s transaction', '32 byte: một giao dịch (transaction) của NVIDIA');
    },
    bytes: 32,
    facts: 'notes/today/ch09-facts.md, A7', // i18n-ignore: facts-file citation
  },
  {
    id: 'amd64',
    get label() {
      return tr('64 bytes: an AMD cache line', '64 byte: một dòng bộ nhớ đệm (cache line) của AMD');
    },
    bytes: 64,
    facts: 'notes/today/ch09-facts.md, B3', // i18n-ignore: facts-file citation
  },
  {
    id: 'amd128',
    get label() {
      return tr('128 bytes: an AMD cache line', '128 byte: một dòng bộ nhớ đệm (cache line) của AMD');
    },
    bytes: 128,
    facts: 'notes/today/ch09-facts.md, B3', // i18n-ignore: facts-file citation
  },
  {
    id: 'apple',
    get label() {
      return tr('Apple: not documented', 'Apple: không có tài liệu');
    },
    bytes: null,
    facts: 'notes/today/ch09-facts.md, C4', // i18n-ignore: facts-file citation
  },
];
