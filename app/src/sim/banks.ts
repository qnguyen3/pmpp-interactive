// A shared-memory bank model for chapter 6 (book §6.4, pp. 139–141, and Exercise 5).
//
// What it models: one shared-memory access by one warp, under the rule NVIDIA
// documents for every compute capability (notes/today/infra-facts.md A10,
// notes/today/ch06-facts.md A12): 32 banks, each 4 bytes wide, successive 32-bit
// words in successive banks, so bank = word index mod 32. Lanes that touch
// different words of the same bank are served one after another; lanes that touch
// the same word do not conflict (a read is broadcast; for a write one lane wins),
// and that includes two narrower elements inside one 32-bit word. The number of
// passes is the largest number of distinct words any one bank must deliver.
//
// The same arithmetic is offered with AMD's documented bank counts (32 on CDNA 1
// to 3, 64 on CDNA 4 and RDNA 2 to 4, 4 bytes each; infra B10). It is NOT AMD's
// cost: AMD's LDS serves 16 lanes per cycle, so a wavefront's access takes several
// cycles even without conflicts, and how AMD groups lanes for conflict detection is
// not modelled. Apple documents no bank structure at all (ch06-facts C2, infra
// C13), so there is no Apple model: lab 6.3 measures instead.
//
// Left out everywhere: timing, accesses wider than 4 bytes per lane (NVIDIA splits
// those differently and the facts files do not cover it), and atomics.

import { tr } from '../lib/locale.ts';

export const bankOf = (word: number, banks = 32) => ((word % banks) + banks) % banks;

export interface BankLane {
  lane: number;
  byte: number;
  word: number;
  bank: number;
}

export interface BankUse {
  bank: number;
  /** Lanes that touch this bank. */
  lanes: number[];
  /** Distinct 32-bit words those lanes touch; each one needs its own pass. */
  words: number[];
}

export interface BankResult {
  lanes: BankLane[];
  /** One entry per bank, in bank order. */
  perBank: BankUse[];
  /** Serialised passes: the most distinct words any bank must deliver (1 = conflict-free). */
  passes: number;
  /** Banks touched by at least one lane. */
  banksUsed: number;
  /** Most lanes on any one bank (counts lanes, not words). */
  maxLanesPerBank: number;
  /** Some word is touched by more than one lane. */
  sharedWord: boolean;
}

/**
 * Banks hit by one warp's shared-memory access. `byteAddrs` holds each active lane's
 * byte offset into shared memory (lane = position in the array); elements are
 * `elementBytes` wide, at most one bank wide.
 */
export function warpBanks(byteAddrs: readonly number[], elementBytes = 4, banks = 32, bankBytes = 4): BankResult {
  if (elementBytes > bankBytes) throw new Error('accesses wider than one bank are not modelled');
  const lanes = byteAddrs.map((byte, lane) => {
    const word = Math.floor(byte / bankBytes);
    return { lane, byte, word, bank: bankOf(word, banks) };
  });
  const perBank: BankUse[] = Array.from({ length: banks }, (_, bank) => ({ bank, lanes: [], words: [] }));
  for (const l of lanes) {
    const b = perBank[l.bank];
    b.lanes.push(l.lane);
    if (!b.words.includes(l.word)) b.words.push(l.word);
  }
  const used = perBank.filter((b) => b.lanes.length);
  const words = new Set(lanes.map((l) => l.word));
  return {
    lanes,
    perBank,
    passes: used.length ? Math.max(...used.map((b) => b.words.length)) : 0,
    banksUsed: used.length,
    maxLanesPerBank: used.length ? Math.max(...used.map((b) => b.lanes.length)) : 0,
    sharedWord: words.size < lanes.length,
  };
}

/** a[threadIdx.x * stride] over one warp (Exercise 5's pattern). */
export function strideCase(stride: number, lanes = 32, banks = 32, elementBytes = 4): BankResult {
  return warpBanks(Array.from({ length: lanes }, (_, l) => l * stride * elementBytes), elementBytes, banks);
}

/** Every lane on the same word. */
export function sameWordCase(word = 0, lanes = 32, banks = 32): BankResult {
  return warpBanks(Array.from({ length: lanes }, () => word * 4), 4, banks);
}

/** a[threadIdx.x] with 2-byte elements (a half array): two lanes per word. */
export function halfCase(lanes = 32, banks = 32): BankResult {
  return warpBanks(Array.from({ length: lanes }, (_, l) => l * 2), 2, banks);
}

export interface TileLane {
  lane: number;
  tx: number;
  ty: number;
  /** Array indices [row][col] as written in the source. */
  row: number;
  col: number;
  /** row·pitch + col. */
  word: number;
}

export interface TileAccessParams {
  /** blockDim.x: threads per block row, which decides which threads share a warp. */
  T: number;
  /** Words per row of the shared array (its column count, padding included). */
  pitch: number;
  /** 'x-first' is a[threadIdx.x][threadIdx.y] (the corner-turned store); 'y-first' is a[threadIdx.y][threadIdx.x]. */
  orientation: 'x-first' | 'y-first';
  /** The warp's first thread is (tx0, ty0); the warp takes warpWidth consecutive threads from there. */
  ty0?: number;
  tx0?: number;
  warpWidth?: number;
  banks?: number;
}

/** One warp's access to a 2D float array in shared memory. */
export function tileAccess(q: TileAccessParams): BankResult & { threads: TileLane[] } {
  const w = q.warpWidth ?? 32;
  const first = (q.ty0 ?? 0) * q.T + (q.tx0 ?? 0);
  const threads: TileLane[] = Array.from({ length: w }, (_, lane) => {
    const t = first + lane;
    const tx = t % q.T;
    const ty = Math.floor(t / q.T);
    const [row, col] = q.orientation === 'x-first' ? [tx, ty] : [ty, tx];
    return { lane, tx, ty, row, col, word: row * q.pitch + col };
  });
  return { ...warpBanks(threads.map((t) => t.word * 4), 4, q.banks ?? 32), threads };
}

const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));

/**
 * Exercise 5's closed form for a[threadIdx.x * stride], stride ≥ 1: lane l is in bank
 * (l·stride) mod banks, which repeats every banks/g lanes, g = gcd(stride, banks).
 * With as many lanes as banks: banks/g banks used, g lanes (all on different words) on each.
 */
export function gcdRule(stride: number, banks = 32, lanes = 32): { g: number; banksUsed: number; lanesPerBank: number } {
  const g = gcd(stride, banks);
  const period = banks / g;
  return { g, banksUsed: Math.min(lanes, period), lanesPerBank: Math.ceil(lanes / period) };
}

/** Exercise 5 (p. 155), parts (a) to (g). */
export const EX5_STRIDES = [32, 31, 24, 16, 12, 8, 7] as const;

// ---------------------------------------------------------------- backends

export type BankPresetId = 'nvidia' | 'amd-cdna' | 'amd-cdna4' | 'amd-rdna' | 'apple';

export interface BankPreset {
  id: BankPresetId;
  label: string;
  /** Null where nothing is documented. */
  banks: number | null;
  warpWidth: number | null;
  /** What the vendor says about many lanes on one word. */
  sameWord: string;
  /** Shown beside every count computed under this preset. */
  caveat: string | null;
  facts: string;
}

export const BANK_PRESETS: Record<BankPresetId, BankPreset> = {
  nvidia: {
    id: 'nvidia',
    get label() { return tr('NVIDIA: 32 banks, 32 lanes', 'NVIDIA: 32 bank, 32 lane'); },
    banks: 32,
    warpWidth: 32,
    get sameWord() { return tr('a read of one word is broadcast to every lane that asks; no conflict', 'một lần đọc một word được phát (broadcast) tới mọi lane yêu cầu; không có xung đột'); },
    caveat: null,
    facts: 'notes/today/infra-facts.md A10; ch06-facts.md A12', // i18n-ignore: facts-file paths and item ids
  },
  'amd-cdna': {
    id: 'amd-cdna',
    get label() { return tr('AMD CDNA 1–3: 32 banks, 64 lanes', 'AMD CDNA 1–3: 32 bank, 64 lane'); },
    banks: 32,
    warpWidth: 64,
    get sameWord() { return tr('broadcast from one address is "handled efficiently without conflicts"', 'phát (broadcast) từ một địa chỉ được "xử lý hiệu quả, không có xung đột"'); },
    get caveat() { return tr('AMD serves 16 lanes per cycle, so a 64-lane access takes several cycles anyway; this count is NVIDIA\'s arithmetic on AMD\'s bank number, not AMD\'s cost.', 'AMD phục vụ 16 lane mỗi chu kỳ, nên một truy cập 64 lane dù sao cũng mất vài chu kỳ; con số này là phép tính của NVIDIA áp lên số bank của AMD, không phải chi phí thật của AMD.'); },
    facts: 'notes/today/infra-facts.md B10; ch06-facts.md B4', // i18n-ignore: facts-file paths and item ids
  },
  'amd-cdna4': {
    id: 'amd-cdna4',
    get label() { return tr('AMD CDNA 4: 64 banks, 64 lanes', 'AMD CDNA 4: 64 bank, 64 lane'); },
    banks: 64,
    warpWidth: 64,
    get sameWord() { return tr('broadcast from one address is "handled efficiently without conflicts"', 'phát (broadcast) từ một địa chỉ được "xử lý hiệu quả, không có xung đột"'); },
    get caveat() { return tr('AMD serves 16 lanes per cycle, so a 64-lane access takes several cycles anyway; this count is NVIDIA\'s arithmetic on AMD\'s bank number, not AMD\'s cost.', 'AMD phục vụ 16 lane mỗi chu kỳ, nên một truy cập 64 lane dù sao cũng mất vài chu kỳ; con số này là phép tính của NVIDIA áp lên số bank của AMD, không phải chi phí thật của AMD.'); },
    facts: 'notes/today/infra-facts.md B10; ch05-facts.md B5', // i18n-ignore: facts-file paths and item ids
  },
  'amd-rdna': {
    id: 'amd-rdna',
    get label() { return tr('AMD RDNA 2–4: 64 banks, 32 lanes', 'AMD RDNA 2–4: 64 bank, 32 lane'); },
    banks: 64,
    warpWidth: 32,
    get sameWord() { return tr('broadcast from one address is "handled efficiently without conflicts"', 'phát (broadcast) từ một địa chỉ được "xử lý hiệu quả, không có xung đột"'); },
    get caveat() { return tr('AMD serves 16 lanes per cycle; this count is NVIDIA\'s arithmetic on AMD\'s bank number, not AMD\'s cost.', 'AMD phục vụ 16 lane mỗi chu kỳ; con số này là phép tính của NVIDIA áp lên số bank của AMD, không phải chi phí thật của AMD.'); },
    facts: 'notes/today/infra-facts.md B10; ch05-facts.md B5', // i18n-ignore: facts-file paths and item ids
  },
  apple: {
    id: 'apple',
    get label() { return tr('Apple: nothing documented', 'Apple: không có tài liệu nào'); },
    banks: null,
    warpWidth: null,
    get sameWord() { return tr('Apple advises removing accesses to the same location from many threads of a threadgroup', 'Apple khuyên loại bỏ việc nhiều luồng của một threadgroup cùng truy cập một vị trí'); },
    get caveat() { return tr('Apple documents no bank structure for threadgroup memory; lab 6.3 measures what your Mac does.', 'Apple không công bố cấu trúc bank nào cho bộ nhớ threadgroup; lab 6.3 đo xem máy Mac của bạn làm gì.'); },
    facts: 'notes/today/ch06-facts.md C2; infra-facts.md C13', // i18n-ignore: facts-file paths and item ids
  },
};
