// The static arithmetic of the flash-attention kernel of book §20.5 (Figs. 20.8–20.12,
// pp. 495–500): the derived constants of Fig. 20.9, how much shared memory and how many
// registers a configuration needs, whether it fits a device's limit, which lane holds
// which element of Q, where a shuffle reads from on a 64-wide wavefront, which banks the
// transposed K tile hits, and which tiles a causal mask leaves empty. Four interactives
// of section 20.5 run on it ("Fit the tiles", "Which tiles does causality empty?",
// "Who holds Q element dd?", "Writing K^T with and without padding").
//
// Sources: the constants and formulas are the book's (notes/ch20.md, Fig. 20.9); the
// limits are from the facts files (CUDA 48 KB static, ch05 A3; AMD LDS 64 KiB or more,
// ch20 B3; Metal 32 KB, ch20 C1, ch05 C1); the shuffle rule is HIP's source (ch20 B2,
// through shuffle.ts) and the bank rule NVIDIA's (banks.ts). The tile classes are
// FlashAttention-2's (facts A7). Nothing here models time.

import { warpBanks, type BankResult } from './banks';
import { sourceOf } from './shuffle';
import { tr as trText } from '../lib/locale.ts';

// ---------------------------------------------------------------- the book's constants
/** Fig. 20.9 lines 1–7. */
export const BOOK = { d: 128, br: 32, bc: 32, threads: 512, warp: 32, banks: 32 } as const;

export interface PlanInput {
  d: number;
  br: number;
  bc: number;
  /** Threads per block (BLOCK_SIZE). */
  threads: number;
  /** Logical warp width (WARP_SIZE): 32 in the book. */
  warp: number;
  /** Bank count used for the padding of KT_j (2^LOG_NUM_BANKS), or 0 for no padding. */
  banks: number;
}

export interface Plan {
  nWarps: number;
  /** B_r / N_WARPS; must be a whole number of rows per warp. */
  brWarp: number;
  dSize: number;
  /** Floats in each shared array. KT_j includes its padding. */
  floats: { kt: number; ktPad: number; s: number; v: number; total: number };
  bytes: { kt: number; s: number; v: number; total: number };
  /** Floats per thread: O_i, Q_i (B_r_warp × d_size each), m_i, D_i (B_r_warp each). */
  regs: { o: number; q: number; m: number; D: number; total: number };
  /** Reasons this configuration cannot run the book's kernel; empty when it can. */
  problems: string[];
}

const isPow2 = (n: number) => Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
export const log2 = (n: number) => Math.round(Math.log2(n));

/** The padding function of Fig. 20.11: addr(x) = x + (x >> LOG_NUM_BANKS); identity when banks = 0. */
export const addr = (x: number, banks: number) => (banks ? x + (x >> log2(banks)) : x);

export function plan(p: PlanInput): Plan {
  const nWarps = p.threads / p.warp;
  const brWarp = p.br / nWarps;
  const dSize = p.d > p.warp ? p.d / p.warp : 1;
  const ktBase = p.bc * p.d;
  const ktPad = p.banks ? ktBase >> log2(p.banks) : 0;
  const kt = ktBase + ktPad;
  const s = p.br * p.bc;
  const v = p.bc * p.d;
  const total = kt + s + v;
  const whole = Number.isInteger(brWarp) && brWarp >= 1;
  const rw = whole ? brWarp : 0;
  const ds = Number.isInteger(dSize) ? dSize : 0;
  const problems: string[] = [];
  if (!Number.isInteger(nWarps)) problems.push(trText(`${p.threads} threads are not a whole number of ${p.warp}-lane warps`, `${p.threads} luồng không chia hết thành các warp ${p.warp} lane`));
  else if (!whole)
    problems.push(
      trText(
        `B_r_warp = B_r / N_WARPS = ${p.br} / ${nWarps} = ${fmtFrac(brWarp)}: every warp must own a whole number of rows (at least 1)`,
        `B_r_warp = B_r / N_WARPS = ${p.br} / ${nWarps} = ${fmtFrac(brWarp)}: mỗi warp phải giữ một số nguyên hàng (ít nhất 1)`,
      ),
    );
  if (p.d % p.warp !== 0 && p.d > p.warp) problems.push(trText(`d = ${p.d} is not a multiple of the warp width ${p.warp} (the book assumes it is, p. 498)`, `d = ${p.d} không phải bội số của độ rộng warp ${p.warp} (sách giả định là có, p. 498)`));
  if (p.threads > 1024) problems.push(trText(`${p.threads} threads per block exceeds the 1024 that CUDA allows (ch05 A5)`, `${p.threads} luồng mỗi khối vượt quá 1024 mà CUDA cho phép (ch05 A5)`));
  if (p.banks && !isPow2(p.banks)) problems.push(trText('the bank count must be a power of two for the shift in addr()', 'số bank phải là lũy thừa của hai để phép dịch trong addr() đúng'));
  return {
    nWarps,
    brWarp,
    dSize,
    floats: { kt, ktPad, s, v, total },
    bytes: { kt: kt * 4, s: s * 4, v: v * 4, total: total * 4 },
    regs: { o: rw * ds, q: rw * ds, m: rw, D: rw, total: 2 * rw * ds + 2 * rw },
    problems,
  };
}

function fmtFrac(x: number): string {
  return Number.isInteger(x) ? String(x) : x.toFixed(3).replace(/0+$/, '');
}

// ---------------------------------------------------------------- limits
export interface Limit {
  id: 'cuda' | 'amd' | 'metal';
  label: string;
  bytes: number;
  /** Where the number comes from. */
  source: string;
}

/** CUDA: 48 KB of static shared memory per block without an opt-in (ch05 A3). */
export const CUDA_STATIC = 48 * 1024;
/** AMD: a block may use the whole LDS of a CU, 64 KiB at the smallest in AMD's table (ch20 B3, ch08 B3). */
export const AMD_LDS_MIN = 64 * 1024;
/** Metal: 32 KB of threadgroup memory documented for Apple4 to Apple11 (ch05 C1, ch20 C1). */
export const METAL_DOCUMENTED = 32 * 1024;

export function limits(metalBytes?: number, metalSource?: string): Limit[] {
  return [
    { id: 'cuda', label: trText('CUDA, static shared memory', 'CUDA, bộ nhớ chia sẻ (shared memory) tĩnh'), bytes: CUDA_STATIC, source: 'ch05 facts A3' }, // i18n-ignore: citation
    { id: 'amd', label: trText('AMD LDS (smallest in AMD’s table)', 'AMD LDS (nhỏ nhất trong bảng của AMD)'), bytes: AMD_LDS_MIN, source: 'ch20 facts B3' }, // i18n-ignore: citation
    {
      id: 'metal',
      label: metalSource ? trText(`Metal, ${metalSource}`, `Metal, ${metalSource}`) : trText('Metal, documented (Apple4 to Apple11)', 'Metal, theo tài liệu (Apple4 đến Apple11)'),
      bytes: metalBytes ?? METAL_DOCUMENTED,
      source: metalSource ? trText('your report (maxThreadgroupMemoryLength)', 'báo cáo của bạn (maxThreadgroupMemoryLength)') : 'ch05 facts C1', // i18n-ignore: citation
    },
  ];
}

export const fits = (bytes: number, limit: number) => bytes <= limit;

/** The largest of the given B_c values whose tiles fit in `limit` bytes (null if none). */
export function largestFittingBc(base: Omit<PlanInput, 'bc'>, candidates: number[], limit: number): number | null {
  const ok = candidates.filter((bc) => fits(plan({ ...base, bc }).bytes.total, limit));
  return ok.length ? Math.max(...ok) : null;
}

// ---------------------------------------------------------------- who holds Q element dd
/** Figs. 20.10 and 20.12: element dd of a Q row lives in lane dd % W, register slot dd / W. */
export function ownerOf(dd: number, warp = 32): { lane: number; slot: number } {
  return { lane: dd % warp, slot: Math.floor(dd / warp) };
}

/** The elements of one Q row a lane holds, in slot order (lane L: L, L + W, L + 2W, ...). */
export function laneElements(lane: number, d: number, warp = 32): number[] {
  const out: number[] = [];
  for (let dd = lane; dd < d; dd += warp) out.push(dd);
  return out;
}

/**
 * The hardware lane whose value a __shfl / __shfl_sync call returns on HIP: index =
 * (srcLane & (width − 1)) + (self & ~(width − 1)) (amd_warp_functions.h, ch20 B2). With
 * the default width = warpSize = 64, hardware lane 40 asking for lane 5 reads lane 5;
 * with width 32 it reads 37.
 */
export function hipShuffleSource(hwLane: number, srcLane: number, width: number): number {
  return sourceOf('idx', hwLane, srcLane, width).target;
}

/** For a 64-lane wavefront holding two 32-lane logical warps: what each hardware lane receives. */
export function wave64Broadcast(dd: number, width: number): { hw: number; logicalWarp: number; wants: number; gets: number; ok: boolean }[] {
  const src = dd % 32;
  return Array.from({ length: 64 }, (_, hw) => {
    const logicalWarp = Math.floor(hw / 32);
    const wants = logicalWarp * 32 + src;
    const gets = hipShuffleSource(hw, src, width);
    return { hw, logicalWarp, wants, gets, ok: gets === wants };
  });
}

// ---------------------------------------------------------------- banks hit by the K^T store
/**
 * Fig. 20.11: for one jj, consecutive threads dd = 0, 1, ... write KT_j[addr(dd · B_c + jj)].
 * `lanes` threads of one warp; `banks` banks of 4 bytes; padding with LOG_NUM_BANKS = log2(banks).
 */
export function ktStoreBanks(bc: number, jj: number, padded: boolean, banks = 32, lanes = 32): BankResult & { index: number[] } {
  const index = Array.from({ length: lanes }, (_, dd) => (padded ? addr(dd * bc + jj, banks) : dd * bc + jj));
  return { ...warpBanks(index.map((i) => i * 4), 4, banks), index };
}

// ---------------------------------------------------------------- causal tile classes
export type TileClass = 'skip' | 'mask' | 'full';

export interface TileClasses {
  tr: number;
  tc: number;
  /** grid[i][j] for Q panel i and K/V tile j. */
  grid: TileClass[][];
  counts: Record<TileClass, number>;
  total: number;
}

/**
 * For causal attention with N rows in panels of B_r and N columns in tiles of B_c: a tile
 * whose every column lies to the right of every row of the panel is empty ('skip',
 * FlashAttention-2 skips it); one wholly at or below the diagonal needs no mask ('full');
 * the rest cross the diagonal ('mask'). The book's kernel computes every pair (facts A7).
 */
export function tileClasses(n: number, br: number, bc: number): TileClasses {
  const tr = Math.ceil(n / br);
  const tc = Math.ceil(n / bc);
  const counts: Record<TileClass, number> = { skip: 0, mask: 0, full: 0 };
  const grid: TileClass[][] = [];
  for (let i = 0; i < tr; i++) {
    const r0 = i * br;
    const r1 = Math.min(n, (i + 1) * br) - 1;
    const row: TileClass[] = [];
    for (let j = 0; j < tc; j++) {
      const c0 = j * bc;
      const c1 = Math.min(n, (j + 1) * bc) - 1;
      const k: TileClass = c0 > r1 ? 'skip' : c1 <= r0 ? 'full' : 'mask';
      counts[k]++;
      row.push(k);
    }
    grid.push(row);
  }
  return { tr, tc, grid, counts, total: tr * tc };
}

/** Tile counts with the book's integer division (line 10–11 of Fig. 20.9) and with the ceiling the labs use. */
export function tileCounts(n: number, br: number, bc: number) {
  return {
    trBook: Math.floor(n / br),
    tcBook: Math.floor(n / bc),
    tr: Math.ceil(n / br),
    tc: Math.ceil(n / bc),
    /** Rows the book's integer division would never compute. */
    rowsDropped: n - Math.floor(n / br) * br,
  };
}
