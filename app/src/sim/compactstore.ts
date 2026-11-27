// Who writes each output slot of a stable filter, and with how many store instructions
// (book §12.6, pp. 298-300, Figs. 12.8, 12.9 and 12.10).
//
// Every design here produces the same output: the kept keys in input order, at the
// offsets an exclusive scan of the keep flags gives (stablescan.ts). What differs is which
// thread stores which key, and so which store instructions each warp issues:
//
//   - 'direct'        Fig. 12.8: one key per thread; a thread whose key is kept stores it
//                     at its offset. Within one warp the kept lanes' offsets rise by one,
//                     so each warp's store writes one short run; every warp with a kept
//                     key issues its own store.
//   - 'staged'        Fig. 12.9: the block first places its kept keys in shared memory at
//                     their block-local offsets; after a barrier, threads 0, 1, 2, ... copy
//                     the block's run out, so consecutive lanes write consecutive slots.
//   - 'staged-coarse' Fig. 12.10: the same with c keys per thread; the block's run is up
//                     to blockDim * c long and is copied out in rounds of blockDim.
//   - 'direct-coarse' Not in the book (ours): c keys per thread held contiguously (thread t
//                     holds keys t*c .. t*c+c-1 of the block, the arrangement of chapter
//                     11's coarsened scan, Fig. 11.13, and of the notes' sketch of exercise
//                     4), each thread storing its own kept keys without staging. Round k
//                     stores every lane's k-th key, so the lanes' slots are spread apart by
//                     how many keys each thread kept. CUB also scatters without staging,
//                     at low fractions (ch12-facts A9); how CUB arranges its keys is not
//                     in the facts files, so this design is not claimed to be CUB's.
//
// Each store instruction's segments are counted by chapter 6's warpAccess (coalesce.ts,
// imported read-only) at a chosen segment size: NVIDIA documents 32-byte sectors (ch06-facts
// A2), AMD 64- or 128-byte cache lines (B1), Apple nothing (C1), so for Metal the size is a
// choice, not a fact.
//
// What it leaves out: one store instruction at a time with no cache (a segment touched by
// two instructions is counted twice), no timing, no cost for the shared-memory round trip
// and barrier that staging adds, and no reads (every design reads each key once from
// global memory in the same coalesced way).
//
// Erasable TypeScript only.
import { warpAccess } from './coalesce';
import { BOOK_KEEP, exclusiveScan } from './stablescan';
import { tr } from '../lib/locale.ts';

export type Design = 'direct' | 'staged' | 'staged-coarse' | 'direct-coarse';
export const DESIGNS: Design[] = ['direct', 'staged', 'staged-coarse', 'direct-coarse'];
export const DESIGN_LABEL: Record<Design, string> = {
  get direct() {
    return tr('Fig. 12.8: each kept thread stores its own key', 'Fig. 12.8: mỗi luồng được giữ tự ghi khóa của mình');
  },
  get staged() {
    return tr('Fig. 12.9: stage in shared memory, then copy out', 'Fig. 12.9: trung chuyển qua bộ nhớ chia sẻ, rồi sao chép ra');
  },
  get 'staged-coarse'() {
    return tr('Fig. 12.10: staged and coarsened', 'Fig. 12.10: có trung chuyển và gộp việc');
  },
  get 'direct-coarse'() {
    return tr('coarsened, no staging (ours)', 'gộp việc, không trung chuyển (của khóa học này)');
  },
};
export const isStaged = (d: Design) => d === 'staged' || d === 'staged-coarse';
export const isCoarse = (d: Design) => d === 'staged-coarse' || d === 'direct-coarse';
/** 4-byte keys throughout. */
export const KEY_BYTES = 4;

export interface Config {
  keep: readonly number[];
  blockDim: number;
  warp: number;
  /** Keys per thread; forced to 1 for 'direct' and 'staged'. */
  coarse: number;
  segmentBytes: number;
  design: Design;
}

export interface Store {
  /** 0-based index in issue order: block by block, round by round, warp by warp. */
  id: number;
  block: number;
  warp: number;
  /** Round of the copy loop (staged) or index of the key within the thread (coarsened direct). */
  round: number;
  /** Active lanes, ascending. */
  lanes: number[];
  /** Output slot written by each active lane, in lane order. */
  slots: number[];
  /** Segments touched (byte address / segmentBytes). */
  segments: number[];
  usefulBytes: number;
}

export interface Totals {
  keys: number;
  kept: number;
  blocks: number;
  keysPerBlock: number;
  stores: number;
  /** Sum over store instructions of the segments each touches. */
  segments: number;
  bytesFetched: number;
  bytesUseful: number;
  /** bytesUseful / bytesFetched. */
  efficiency: number;
  /** Bytes the kernel reads: every key once, fully coalesced, the same in every design. */
  readBytes: number;
  /** Store bytes fetched as a share of all global traffic (reads plus store segments). */
  storeShare: number;
  /** The book's contiguous chunks (p. 299): one per block that keeps anything. */
  chunks: number;
  meanChunk: number;
  /** Shared memory a block allocates for staging, in bytes (0 without staging). */
  stagingBytes: number;
  /** Values the block-level scan takes in per key: one per thread over c keys, so 1/c. */
  scanInputsPerKey: number;
}

export interface StoreMap {
  cfg: Config;
  offsets: number[];
  /** Per block: where its run starts in the output and how long it is. */
  runs: { start: number; length: number }[];
  stores: Store[];
  /** For each output slot, the id of the store that writes it. */
  writer: number[];
  totals: Totals;
}

/** Normalised configuration: designs without coarsening get c = 1. */
export function normalise(cfg: Config): Config {
  return { ...cfg, coarse: isCoarse(cfg.design) ? cfg.coarse : 1 };
}

export function storeMap(input: Config): StoreMap {
  const cfg = normalise(input);
  const { keep, blockDim: D, warp: W, coarse: c, segmentBytes } = cfg;
  if (!(D >= 1 && W >= 1 && c >= 1 && segmentBytes >= KEY_BYTES)) throw new Error('bad configuration');
  const n = keep.length;
  const per = D * c;
  const blocks = Math.max(1, Math.ceil(n / per));
  const offsets = exclusiveScan(keep);
  const kept = n ? offsets[n - 1] + keep[n - 1] : 0;
  const runs = Array.from({ length: blocks }, (_, b) => {
    const lo = b * per;
    const hi = Math.min(n, lo + per);
    const start = lo < n ? offsets[lo] : kept;
    const end = hi < n ? offsets[hi] : kept;
    return { start, length: end - start };
  });
  const stores: Store[] = [];
  const warpsPerBlock = Math.ceil(D / W);
  const emit = (block: number, warp: number, round: number, lanes: number[], slots: number[]) => {
    if (!lanes.length) return;
    const a = warpAccess(slots.map((s) => s * KEY_BYTES), KEY_BYTES, segmentBytes);
    stores.push({ id: stores.length, block, warp, round, lanes, slots, segments: a.segments, usefulBytes: a.bytesNeeded });
  };
  for (let b = 0; b < blocks; b++) {
    const base = b * per;
    if (isStaged(cfg.design)) {
      const { start, length } = runs[b];
      const rounds = Math.ceil(length / D);
      for (let r = 0; r < rounds; r++)
        for (let w = 0; w < warpsPerBlock; w++) {
          const lanes: number[] = [];
          const slots: number[] = [];
          for (let l = 0; l < W; l++) {
            const t = w * W + l;
            const j = r * D + t;
            if (t < D && j < length) {
              lanes.push(l);
              slots.push(start + j);
            }
          }
          emit(b, w, r, lanes, slots);
        }
    } else {
      for (let k = 0; k < c; k++)
        for (let w = 0; w < warpsPerBlock; w++) {
          const lanes: number[] = [];
          const slots: number[] = [];
          for (let l = 0; l < W; l++) {
            const t = w * W + l;
            const i = base + t * c + k;
            if (t < D && i < n && keep[i]) {
              lanes.push(l);
              slots.push(offsets[i]);
            }
          }
          emit(b, w, k, lanes, slots);
        }
    }
  }
  const writer = new Array<number>(kept).fill(-1);
  for (const s of stores) for (const slot of s.slots) writer[slot] = s.id;
  const segments = stores.reduce((a, s) => a + s.segments.length, 0);
  const bytesFetched = segments * segmentBytes;
  const bytesUseful = kept * KEY_BYTES;
  const chunks = runs.filter((r) => r.length > 0).length;
  return {
    cfg,
    offsets,
    runs,
    stores,
    writer,
    totals: {
      keys: n,
      kept,
      blocks,
      keysPerBlock: per,
      stores: stores.length,
      segments,
      bytesFetched,
      bytesUseful,
      efficiency: bytesFetched ? bytesUseful / bytesFetched : 0,
      readBytes: n * KEY_BYTES,
      storeShare: bytesFetched + n * KEY_BYTES ? bytesFetched / (bytesFetched + n * KEY_BYTES) : 0,
      chunks,
      meanChunk: chunks ? kept / chunks : 0,
      stagingBytes: isStaged(cfg.design) ? per * KEY_BYTES : 0,
      scanInputsPerKey: 1 / c,
    },
  };
}

// ---------------------------------------------------------------- keep patterns
export type Pattern = 'random' | 'runs' | 'alternate';
export const PATTERN_LABEL: Record<Pattern, string> = {
  get random() {
    return tr('random', 'ngẫu nhiên');
  },
  get runs() {
    return tr('in runs', 'theo dải');
  },
  get alternate() {
    return tr('every other key', 'cứ cách một khóa');
  },
};

function lcg(seed: number) {
  let r = (Math.imul(seed, 2654435761) + 12345) >>> 0;
  return () => {
    r = (Math.imul(r, 1664525) + 1013904223) >>> 0;
    return (r >>> 8) / 16777216;
  };
}

/**
 * Seeded keep flags with about a fraction f kept. 'runs' keeps and drops in runs of
 * random length averaging 16 keys; 'alternate' keeps every other key whatever f is.
 */
export function makeKeep(n: number, f: number, pattern: Pattern, seed = 1): number[] {
  const r = lcg(seed);
  if (pattern === 'alternate') return Array.from({ length: n }, (_, i) => (i % 2 === 1 ? 1 : 0));
  if (pattern === 'random') return Array.from({ length: n }, () => (r() < f ? 1 : 0));
  const out: number[] = [];
  let on = r() < f;
  while (out.length < n) {
    // Mean run length 16 for the kept and dropped runs together, in proportion f.
    const mean = on ? Math.max(1, 32 * f) : Math.max(1, 32 * (1 - f));
    const len = Math.max(1, Math.round(-Math.log(1 - r()) * mean));
    for (let k = 0; k < len && out.length < n; k++) out.push(on && f > 0 ? 1 : 0);
    on = f >= 1 ? true : f <= 0 ? false : !on;
  }
  return out;
}

// ---------------------------------------------------------------- the explorer's preset
/** The explorer opens here; Predict ch12/12.6/p-stores asks about it. */
export const PRESET = {
  n: 1024,
  fraction: 0.5,
  pattern: 'random' as Pattern,
  blockDim: 256,
  warp: 32,
  coarse: 1,
  segmentBytes: 32,
  seed: 1,
};
export function presetMap(design: Design): StoreMap {
  const p = PRESET;
  return storeMap({
    keep: makeKeep(p.n, p.fraction, p.pattern, p.seed),
    blockDim: p.blockDim,
    warp: p.warp,
    coarse: p.coarse,
    segmentBytes: p.segmentBytes,
    design,
  });
}

// ---------------------------------------------------------------- the narrated scene
/**
 * "Gather, then write together": the book's sixteen keys (Fig. 12.7's keep row), two
 * blocks of eight threads, warps of four lanes, and a toy segment of 16 bytes (four keys)
 * so that every boundary is visible. Coarsening by two puts all sixteen keys in one block.
 */
export const SCENE = { blockDim: 8, warp: 4, segmentBytes: 16, coarse: 2 } as const;
export function sceneData() {
  const base = { keep: BOOK_KEEP, blockDim: SCENE.blockDim, warp: SCENE.warp, segmentBytes: SCENE.segmentBytes };
  const direct = storeMap({ ...base, coarse: 1, design: 'direct' });
  const staged = storeMap({ ...base, coarse: 1, design: 'staged' });
  const coarse = storeMap({ ...base, coarse: SCENE.coarse, design: 'staged-coarse' });
  return { keep: [...BOOK_KEEP], offsets: direct.offsets, direct, staged, coarse };
}
