// Shared-memory banks of the tiled Smith-Waterman kernels (book §16.7, pp. 395–396,
// Fig. 16.20; ch16-facts E1).
//
// What it models: for one full interior tile, the slot each lane of a warp touches in
// one access, for every intra-tile step and every warp of the block, and the bank of
// that slot, under three layouts of the W x W int tile:
//   unpadded   word = row·W + col                       (the listings as printed)
//   row gap    word = row·(W + 1) + col                 (one empty word after each tile row)
//   pad()      word = x + (x >> LOG_NUM_BANKS), x = row·W + col   (the book's macro, p. 395)
// Accesses: the compute step's write and its three loads (Figs. 16.9–16.10 square,
// 16.16–16.17 sheared), and store_tile / initialize_tile, which touch slot (row, t).
// A lane whose load goes to global memory (a tile edge) takes no part in the shared access.
// The conflict count is warpBanks from banks.ts: NVIDIA's documented rule (4-byte banks,
// bank = word mod banks, distinct words in one bank serialise).
// Two bank models: 32 banks with 32-lane warps (NVIDIA), and 64 banks with 64-lane
// wavefronts, which is NVIDIA's arithmetic applied to an AMD bank count: AMD does not
// document how a 64-lane access is split for conflicts (ch15 B2), so it is a model.
// Apple documents no banks, so there is no Apple model (lab 16.1 measures padding).
// Left out: timing, and every lane of an edge or partial tile (the tile is full).

import { warpBanks, type BankResult } from './banks';
import { tr } from '../lib/locale.ts';

export type BankKernel = 'square' | 'sheared';
export type BankLayout = 'unpadded' | 'rowgap' | 'pad';
export type BankAccess = 'write' | 'n' | 'w' | 'nw' | 'store' | 'init';

export const COMPUTE_ACCESSES: BankAccess[] = ['write', 'n', 'w', 'nw'];
export const BANK_WIDTHS = [8, 16, 32, 64] as const;

export const LAYOUT_LABEL: Record<BankLayout, string> = {
  get unpadded() {
    return tr('unpadded', 'không đệm');
  },
  get rowgap() {
    return tr('one gap per row', 'một khoảng trống mỗi hàng');
  },
  get pad() {
    return tr("the book's pad()", 'pad() của sách');
  },
};
export const ACCESS_LABEL: Record<BankAccess, string> = {
  get write() {
    return tr('write of the step', 'lần ghi của bước');
  },
  n: 'load_n',
  w: 'load_w',
  nw: 'load_nw',
  store: 'store_tile',
  init: 'initialize_tile',
};

export interface BankModel {
  id: 'nvidia' | 'amd64';
  label: string;
  banks: number;
  lanes: number;
  /** The book's LOG_NUM_BANKS for this bank count. */
  logBanks: number;
}
export const BANK_MODELS: Record<BankModel['id'], BankModel> = {
  nvidia: {
    id: 'nvidia',
    get label() {
      return tr('32 banks, 32-lane warps (NVIDIA)', '32 bank, warp 32 lane (NVIDIA)');
    },
    banks: 32,
    lanes: 32,
    logBanks: 5,
  },
  amd64: {
    id: 'amd64',
    get label() {
      return tr('64 banks, 64-lane wavefronts (an AMD model)', '64 bank, wavefront 64 lane (một mô hình AMD)');
    },
    banks: 64,
    lanes: 64,
    logBanks: 6,
  },
};

/** Word index of tile slot (row, col). */
export function wordOf(layout: BankLayout, W: number, row: number, col: number, logBanks = 5): number {
  if (layout === 'rowgap') return row * (W + 1) + col;
  const x = row * W + col;
  return layout === 'pad' ? x + (x >> logBanks) : x;
}

/** Words the tile needs under a layout: W², W(W + 1), or pad(W² - 1) + 1. */
export function wordsNeeded(layout: BankLayout, W: number, logBanks = 5): number {
  return layout === 'rowgap' ? W * (W + 1) : wordOf(layout, W, W - 1, W - 1, logBanks) + 1;
}

/** Steps (or loop iterations) an access runs for: 2W - 1 or W intra-tile steps; W rows for store and init. */
export function stepsFor(kernel: BankKernel, access: BankAccess, W: number): number {
  if (access === 'store' || access === 'init') return W;
  return kernel === 'square' ? 2 * W - 1 : W;
}

/**
 * The tile slot thread t touches in this access at step (or row) k, or null when it
 * takes no part: idle at this step, or its load goes to global memory (a tile edge).
 */
export function slotOf(kernel: BankKernel, access: BankAccess, W: number, k: number, t: number): [number, number] | null {
  if (access === 'store' || access === 'init') return [k, t];
  if (kernel === 'square') {
    const q = k - t; // q_tile
    if (q < 0 || q >= W) return null;
    switch (access) {
      case 'write': return [t, q];
      case 'n': return t === 0 ? null : [t - 1, q];
      case 'w': return q === 0 ? null : [t, q - 1];
      case 'nw': return t === 0 || q === 0 ? null : [t - 1, q - 1];
    }
  }
  switch (access) {
    case 'write': return [t, k];
    case 'n': return t === 0 || k === 0 ? null : [t - 1, k - 1];
    case 'w': return k === 0 ? null : [t, k - 1];
    case 'nw': return t === 0 || k <= 1 ? null : [t - 1, k - 2];
  }
  return null;
}

export interface LaneUse {
  lane: number;
  thread: number;
  slot: [number, number] | null;
  word: number | null;
  bank: number | null;
}

export interface WarpUse {
  lanes: LaneUse[];
  /** Over the lanes that take part. */
  result: BankResult;
  /** Serialised passes: 1 = conflict-free, 0 = no lane takes part. */
  degree: number;
}

export const warpsFor = (W: number, m: BankModel) => Math.ceil(W / m.lanes);

/** One warp's access at step (or row) k. */
export function warpUse(kernel: BankKernel, access: BankAccess, layout: BankLayout, W: number, m: BankModel, k: number, warp: number): WarpUse {
  const first = warp * m.lanes;
  const n = Math.max(0, Math.min(m.lanes, W - first));
  const lanes: LaneUse[] = Array.from({ length: n }, (_, lane) => {
    const thread = first + lane;
    const slot = slotOf(kernel, access, W, k, thread);
    const word = slot ? wordOf(layout, W, slot[0], slot[1], m.logBanks) : null;
    return { lane, thread, slot, word, bank: word === null ? null : word % m.banks };
  });
  const words = lanes.filter((l) => l.word !== null).map((l) => l.word! * 4);
  const result = warpBanks(words, 4, m.banks);
  return { lanes, result, degree: result.passes };
}

export interface Worst {
  degree: number;
  /** First step and warp where it happens. */
  k: number;
  warp: number;
}

/** The worst conflict of one access over every step and every warp of the block. */
export function worstOf(kernel: BankKernel, access: BankAccess, layout: BankLayout, W: number, m: BankModel): Worst {
  let best: Worst = { degree: 0, k: 0, warp: 0 };
  for (let k = 0; k < stepsFor(kernel, access, W); k++)
    for (let warp = 0; warp < warpsFor(W, m); warp++) {
      const d = warpUse(kernel, access, layout, W, m, k, warp).degree;
      if (d > best.degree) best = { degree: d, k, warp };
    }
  return best;
}

/** The worst over the compute step's write and three loads. */
export function worstCompute(kernel: BankKernel, layout: BankLayout, W: number, m: BankModel): number {
  return Math.max(...COMPUTE_ACCESSES.map((a) => worstOf(kernel, a, layout, W, m).degree));
}

/** Smallest and largest worst-case degree over the compute accesses ("31 to 32"). */
export function computeRange(kernel: BankKernel, layout: BankLayout, W: number, m: BankModel): [number, number] {
  const ds = COMPUTE_ACCESSES.map((a) => worstOf(kernel, a, layout, W, m).degree);
  return [Math.min(...ds), Math.max(...ds)];
}
