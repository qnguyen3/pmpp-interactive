// Chapter 4, section 4.8: from a device's limits to a launch configuration. The
// rules lab 4.1 tests a configuration against, as a checked function, and one way
// to find a configuration that passes them. It runs the explorer "Derive a launch
// configuration".
//
// (app/src/sim/launch.ts is chapter 2's model of a single launch. This file is the
// chapter 4 model; the two share nothing.)
//
// Sources for the rules (the lesson cites them where they are used):
//  - labs/ch04/lab1_launchconfig/main.cpp, brokenRules(): the rule names below are
//    that function's, and launchconfig.test.ts fails if one is renamed there;
//  - labs/ch04/lab1_launchconfig/solution/kernel.cu, chooseLaunch(): suggest() is
//    the same search;
//  - the book, section 4.8, p. 89: maxThreadsPerBlock, maxThreadsDim, maxGridSize
//    and warpSize are per-device limits to be queried;
//  - the book, section 4.7, pp. 85 to 86: a block size that does not divide the
//    thread slots leaves some empty (768 on the example), and a block so small that
//    the block slots run out first leaves half of them empty (32 on the example);
//  - the book, section 4.4, p. 75: a block that is not whole warps has its last warp
//    padded with lanes that run nothing;
//  - notes/today/ch05-facts.md A5: on CUDA a block is at most 1024 x 1024 x 64 and
//    1024 threads, a grid at most 2^31 - 1 by 65535 by 65535;
//  - notes/today/ch04-facts.md B9: a HIP device may report 0 for a field, and its
//    block-slot figure is derived from the thread slots.
//
// What this leaves out (the lesson says so beside the explorer):
//  - the rules are this course's first default for a one-thread-per-output kernel,
//    not a vendor recommendation. A kernel that keeps data in registers may do
//    better far from full thread slots (section 4.6);
//  - registers and shared memory, which can lower occupancy below what the slots
//    allow (section 4.7): no device field tells you a kernel's register count;
//  - block.z and grid.z are fixed at 1 and the grid is always the smallest that
//    covers the problem, so the lab's rules "two-dimensional", "grid covers the
//    problem" and "no block to spare" hold by construction and are not listed;
//  - Metal: its limits belong to a compiled pipeline, and Apple publishes its own
//    recipe (facts C6). This model is for CUDA and HIP.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';
import type { Limits } from './occupancy';

/** What a device says about launches. null means "not reported": the rule that needs it is skipped. */
export interface LaunchLimits {
  /** warpSize. */
  width: number | null;
  maxThreadsPerBlock: number | null;
  /** maxThreadsDim: x, y, z. */
  maxThreadsDim: [number | null, number | null, number | null];
  /** maxGridSize: x, y, z. */
  maxGridSize: [number | null, number | null, number | null];
  /** maxThreadsPerMultiProcessor. */
  threadSlots: number | null;
  /** maxBlocksPerMultiProcessor. */
  blockSlots: number | null;
  /** True on HIP, where the block-slot figure is thread slots / width (facts B9). */
  blockSlotsDerived: boolean;
}

export type RuleKey = 'per-block' | 'dims' | 'warps' | 'divides' | 'block-slots' | 'rows' | 'grid';

/** The rules in the order they are shown, with the names lab 4.1's tests print. */
export const RULES: readonly { key: RuleKey; name: string }[] = [
  { key: 'per-block', name: 'within maxThreadsPerBlock' }, // i18n-ignore: identifier also shown to the reader (the name lab 4.1 prints)
  { key: 'dims', name: 'within maxThreadsDim' }, // i18n-ignore: identifier also shown to the reader (the name lab 4.1 prints)
  { key: 'warps', name: 'whole number of warps' }, // i18n-ignore: identifier also shown to the reader (the name lab 4.1 prints)
  { key: 'divides', name: 'divides the thread slots' }, // i18n-ignore: identifier also shown to the reader (the name lab 4.1 prints)
  { key: 'block-slots', name: 'block slots do not run out first' }, // i18n-ignore: identifier also shown to the reader (the name lab 4.1 prints)
  { key: 'rows', name: 'no taller than the rows' }, // i18n-ignore: identifier also shown to the reader (the name lab 4.1 prints)
  { key: 'grid', name: 'within maxGridSize' }, // i18n-ignore: identifier also shown to the reader (the name lab 4.1 prints)
];
export const ruleName = (key: RuleKey): string => RULES.find((r) => r.key === key)!.name;

export type RuleStatus = 'pass' | 'fail' | 'skipped';
export interface RuleResult {
  key: RuleKey;
  name: string;
  status: RuleStatus;
  /** The numbers behind the verdict, or which field was not reported. */
  detail: string;
}

export interface BlockShape {
  x: number;
  y: number;
}

export interface LaunchCheck {
  rows: number;
  cols: number;
  block: BlockShape;
  /** The smallest grid of such blocks that covers rows x cols. */
  grid: BlockShape;
  threadsPerBlock: number;
  blocks: number;
  /** Threads the launch creates. */
  launched: number;
  /** Threads that own an output: rows x cols. */
  working: number;
  /** Threads the kernel's guard turns away. */
  surplus: number;
  rules: RuleResult[];
  /** No rule fails. A skipped rule does not count against a configuration. */
  ok: boolean;
  failed: RuleKey[];
  skipped: RuleKey[];
}

const pos = (v: number | null | undefined): number | null => (v != null && Number.isFinite(v) && v > 0 ? Math.floor(v) : null);
const whole = (v: number, min = 1): number => (Number.isFinite(v) ? Math.max(min, Math.floor(v)) : min);
const n = (v: number): string => v.toLocaleString('en-US');

/**
 * One block shape against every rule, each on its own, for a kernel with one thread
 * per element of a rows x cols output: x covers the columns, y covers the rows.
 */
export function checkLaunch(limits: LaunchLimits, rowsIn: number, colsIn: number, blockIn: BlockShape): LaunchCheck {
  const rows = whole(rowsIn);
  const cols = whole(colsIn);
  const block = { x: whole(blockIn.x), y: whole(blockIn.y) };
  const threads = block.x * block.y;
  const grid = { x: Math.ceil(cols / block.x), y: Math.ceil(rows / block.y) };
  const blocks = grid.x * grid.y;
  const launched = blocks * threads;
  const working = rows * cols;

  const perBlock = pos(limits.maxThreadsPerBlock);
  const dimX = pos(limits.maxThreadsDim[0]);
  const dimY = pos(limits.maxThreadsDim[1]);
  const width = pos(limits.width);
  const threadSlots = pos(limits.threadSlots);
  const blockSlots = pos(limits.blockSlots);
  const gridX = pos(limits.maxGridSize[0]);
  const gridY = pos(limits.maxGridSize[1]);

  const out: RuleResult[] = [];
  const add = (key: RuleKey, status: RuleStatus, detail: string) => out.push({ key, name: ruleName(key), status, detail });
  const verdict = (ok: boolean): RuleStatus => (ok ? 'pass' : 'fail');

  if (perBlock == null) add('per-block', 'skipped', tr('maxThreadsPerBlock not reported', 'maxThreadsPerBlock không được báo cáo'));
  else add('per-block', verdict(threads <= perBlock), tr(`${n(threads)} threads, the device allows ${n(perBlock)}`, `${n(threads)} luồng, thiết bị cho phép ${n(perBlock)}`));

  if (dimX == null || dimY == null) add('dims', 'skipped', tr('maxThreadsDim not reported', 'maxThreadsDim không được báo cáo'));
  else add('dims', verdict(block.x <= dimX && block.y <= dimY), tr(`${n(block.x)} x ${n(block.y)}, the device allows ${n(dimX)} x ${n(dimY)}`, `${n(block.x)} x ${n(block.y)}, thiết bị cho phép ${n(dimX)} x ${n(dimY)}`));

  if (width == null) add('warps', 'skipped', tr('warpSize not reported', 'warpSize không được báo cáo'));
  else {
    const over = threads % width;
    add(
      'warps',
      verdict(over === 0),
      over === 0
        ? tr(`${n(threads)} threads are ${n(threads / width)} ${threads / width === 1 ? 'warp' : 'warps'} of ${width}`, `${n(threads)} luồng là ${n(threads / width)} warp, mỗi warp rộng ${width}`)
        : tr(`${n(threads)} threads, the width is ${width}: the last warp has ${width - over} padded lanes`, `${n(threads)} luồng, độ rộng là ${width}: warp cuối có ${width - over} lane đệm`),
    );
  }

  if (threadSlots == null) add('divides', 'skipped', tr('maxThreadsPerMultiProcessor not reported', 'maxThreadsPerMultiProcessor không được báo cáo'));
  else {
    const left = threadSlots % threads;
    add(
      'divides',
      verdict(left === 0),
      left === 0
        ? tr(`${n(threadSlots / threads)} blocks of ${n(threads)} threads are exactly ${n(threadSlots)} thread slots`, `${n(threadSlots / threads)} khối ${n(threads)} luồng vừa đúng ${n(threadSlots)} chỗ luồng`)
        : tr(`${n(threads)} threads per block leave ${n(left)} of ${n(threadSlots)} thread slots empty`, `${n(threads)} luồng mỗi khối để trống ${n(left)} trên ${n(threadSlots)} chỗ luồng`),
    );
  }

  if (threadSlots == null) add('block-slots', 'skipped', tr('maxThreadsPerMultiProcessor not reported', 'maxThreadsPerMultiProcessor không được báo cáo'));
  else if (blockSlots == null) add('block-slots', 'skipped', tr('maxBlocksPerMultiProcessor not reported', 'maxBlocksPerMultiProcessor không được báo cáo'));
  else {
    const reach = threads * blockSlots;
    add(
      'block-slots',
      verdict(reach >= threadSlots),
      reach >= threadSlots
        ? tr(`${n(blockSlots)} block slots x ${n(threads)} threads reach ${n(threadSlots)} thread slots`, `${n(blockSlots)} chỗ khối x ${n(threads)} luồng đạt tới ${n(threadSlots)} chỗ luồng`)
        : tr(`${n(blockSlots)} block slots x ${n(threads)} threads fill ${n(reach)} of ${n(threadSlots)} thread slots`, `${n(blockSlots)} chỗ khối x ${n(threads)} luồng lấp ${n(reach)} trên ${n(threadSlots)} chỗ luồng`),
    );
  }

  add(
    'rows',
    verdict(block.y <= rows),
    block.y <= rows
      ? tr(`block.y is ${n(block.y)} for ${n(rows)} ${rows === 1 ? 'row' : 'rows'}`, `block.y là ${n(block.y)} cho ${n(rows)} hàng`)
      : tr(`block.y is ${n(block.y)} for ${n(rows)} ${rows === 1 ? 'row' : 'rows'}: ${n(block.y - rows)} of every ${n(block.y)} rows of threads own nothing`, `block.y là ${n(block.y)} cho ${n(rows)} hàng: cứ ${n(block.y)} hàng luồng thì ${n(block.y - rows)} hàng không sở hữu phần tử nào`),
  );

  if (gridX == null || gridY == null) add('grid', 'skipped', tr('maxGridSize not reported', 'maxGridSize không được báo cáo'));
  else add('grid', verdict(grid.x <= gridX && grid.y <= gridY), tr(`grid ${n(grid.x)} x ${n(grid.y)}, the device allows ${n(gridX)} x ${n(gridY)}`, `lưới ${n(grid.x)} x ${n(grid.y)}, thiết bị cho phép ${n(gridX)} x ${n(gridY)}`));

  const failed = out.filter((r) => r.status === 'fail').map((r) => r.key);
  const skipped = out.filter((r) => r.status === 'skipped').map((r) => r.key);
  return {
    rows, cols, block, grid, threadsPerBlock: threads, blocks, launched, working, surplus: launched - working,
    rules: out, ok: failed.length === 0, failed, skipped,
  };
}

/** True when blocks of `threads` threads can fill the thread slots; a field not reported skips its test. */
function fillsThreadSlots(limits: LaunchLimits, threads: number): boolean {
  const threadSlots = pos(limits.threadSlots);
  const blockSlots = pos(limits.blockSlots);
  if (threadSlots == null) return true;
  if (threadSlots % threads !== 0) return false;
  if (blockSlots != null && threads * blockSlots < threadSlots) return false;
  return true;
}

/**
 * One configuration that passes every rule, or null when none exists. The search of
 * lab 4.1's reference solution: a block whose rows are whole warps, as tall as the
 * limits and the problem allow, and no wider than a rule demands. With no reported
 * warp width there is nothing to build rows from, and the answer is null.
 */
export function suggest(limits: LaunchLimits, rowsIn: number, colsIn: number): BlockShape | null {
  const rows = whole(rowsIn);
  const cols = whole(colsIn);
  const width = pos(limits.width);
  if (width == null) return null;
  const perBlock = pos(limits.maxThreadsPerBlock) ?? Infinity;
  const dimX = pos(limits.maxThreadsDim[0]) ?? Infinity;
  const dimY = pos(limits.maxThreadsDim[1]) ?? Infinity;
  const gridX = pos(limits.maxGridSize[0]) ?? Infinity;
  const gridY = pos(limits.maxGridSize[1]) ?? Infinity;
  // A block wider than the columns only helps to reach the thread slots, so the search
  // stops there even when the device reports no cap of its own.
  const widest = Math.max(Math.ceil(cols / width) * width, pos(limits.threadSlots) ?? width);
  const maxHeight = Math.min(Math.floor(perBlock / width), dimY, rows);
  for (let h = maxHeight; h >= 1; h--) {
    const maxWidth = Math.min(Math.floor(perBlock / h), dimX, widest);
    for (let w = width; w <= maxWidth; w += width) {
      if (!fillsThreadSlots(limits, w * h)) continue;
      if (Math.ceil(cols / w) > gridX || Math.ceil(rows / h) > gridY) continue;
      return { x: w, y: h };
    }
  }
  return null;
}

/** The per-SM part of a set of launch limits, for the occupancy model. */
export const occupancyLimits = (l: LaunchLimits): Limits => ({
  threadSlots: pos(l.threadSlots),
  blockSlots: pos(l.blockSlots),
  blockSlotsDerived: l.blockSlotsDerived,
  regs: null,
  shared: null,
  width: pos(l.width),
  maxThreadsPerBlock: pos(l.maxThreadsPerBlock),
});

// ---------------------------------------------------------------- devices
/** CUDA's documented launch limits, the same on every listed compute capability (facts ch05 A5). */
export const CUDA_BLOCK_DIMS: [number, number, number] = [1024, 1024, 64];
export const CUDA_GRID_DIMS: [number, number, number] = [2 ** 31 - 1, 65535, 65535];

/**
 * The book's example device: 32 block slots, 2048 thread slots and 1024 threads per
 * block (p. 85), a warp of 32 (p. 74), with CUDA's documented block and grid
 * dimensions. It is the first row of lab 4.1's table of synthetic devices.
 */
export const BOOK_LAUNCH: LaunchLimits = {
  width: 32,
  maxThreadsPerBlock: 1024,
  maxThreadsDim: CUDA_BLOCK_DIMS,
  maxGridSize: CUDA_GRID_DIMS,
  threadSlots: 2048,
  blockSlots: 32,
  blockSlotsDerived: false,
};

/** The device fields this model reads, by the name a report (reports/runs/<backend>.json) uses. */
export const LAUNCH_FIELDS = {
  width: 'warpSize',
  maxThreadsPerBlock: 'maxThreadsPerBlock',
  maxThreadsDim: ['maxThreadsDim.x', 'maxThreadsDim.y', 'maxThreadsDim.z'],
  maxGridSize: ['maxGridSize.x', 'maxGridSize.y', 'maxGridSize.z'],
  threadSlots: 'maxThreadsPerMultiProcessor',
  blockSlots: 'maxBlocksPerMultiProcessor',
} as const;

const reported = (v: unknown): number | null => {
  const x = typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN;
  return Number.isInteger(x) && x > 0 ? x : null;
};

/**
 * Launch limits as one of this machine's device reports gives them. A missing field
 * and a field of 0 both become "not reported". Returns null for the CPU emulators,
 * whose values describe no GPU, and for a report with neither a warp width nor a
 * threads-per-block limit, which is what a Metal device gives (facts C1).
 */
export function launchLimitsFromReport(device: Record<string, unknown>, backend: string): LaunchLimits | null {
  if (backend === 'emu' || backend === 'race' || (typeof device.note === 'string' && device.note.includes('synthetic'))) return null;
  if (!(LAUNCH_FIELDS.width in device) && !(LAUNCH_FIELDS.maxThreadsPerBlock in device)) return null;
  const three = (keys: readonly string[]) => keys.map((k) => reported(device[k])) as [number | null, number | null, number | null];
  return {
    width: reported(device[LAUNCH_FIELDS.width]),
    maxThreadsPerBlock: reported(device[LAUNCH_FIELDS.maxThreadsPerBlock]),
    maxThreadsDim: three(LAUNCH_FIELDS.maxThreadsDim),
    maxGridSize: three(LAUNCH_FIELDS.maxGridSize),
    threadSlots: reported(device[LAUNCH_FIELDS.threadSlots]),
    blockSlots: reported(device[LAUNCH_FIELDS.blockSlots]),
    blockSlotsDerived: backend === 'hip',
  };
}

/** Apple's recipe for a threadgroup (facts C6): w = threadExecutionWidth, h = maxTotalThreadsPerThreadgroup / w. */
export function appleRecipe(threadExecutionWidth: number, maxTotalThreadsPerThreadgroup: number): BlockShape | null {
  const w = pos(threadExecutionWidth);
  const most = pos(maxTotalThreadsPerThreadgroup);
  if (w == null || most == null || most < w) return null;
  return { x: w, y: Math.floor(most / w) };
}

/** The shapes lab 4.1's tests ask about: rows (the batch) by columns (the outputs). */
export const LAB_PROBLEMS: readonly { rows: number; cols: number }[] = [
  { rows: 1, cols: 4096 },
  { rows: 3, cols: 40 },
  { rows: 64, cols: 4096 },
  { rows: 1000, cols: 1000 },
];
