// Chapter 6, section 6.5: thread coarsening. Runs the wave calculator, the
// "who handles what" stepper, the pitfall cards and the numbers in the section's
// questions.
//
// Sources for the rules (the lesson cites them where they are used):
//  - the book, section 6.5, pp. 141 to 143: coarsening factor, coarsening loop,
//    iStart = factor x (global index) and a loop over c; the three pitfalls; the
//    wave arithmetic on a hypothetical device that runs 264 blocks at once
//    (1056 blocks = 4 waves, x8 gives 132 = half a wave; 1584 = 6 waves, x4 gives
//    396 = 1.5 waves);
//  - notes/today/ch06-facts.md A2: NVIDIA's coalescing unit, the 32-byte sector;
//    A11: a warp's float4 load is 16 sectors in one instruction;
//  - notes/today/ch04-facts.md A19: blocks resident at once is the number of SMs
//    times the blocks one SM holds; B9: on HIP maxBlocksPerMultiProcessor is the
//    number of wave slots, not a block-slot limit of its own; B10: a block is
//    rounded up to whole waves before thread slots are divided.
//
// What this leaves out (the lesson says so beside every interactive):
//  - Every block takes the same time, and the device holds a fixed number of
//    blocks at once. Changing the factor does not change that number here, as it
//    would if coarsening raised registers or shared memory per block; the reader
//    lowers it by hand to see that pitfall.
//  - "Wave time" assumes a coarsened block takes factor times as long as an
//    uncoarsened one: it counts the parallelism lost and none of the overhead
//    saved, which is what coarsening is for. It answers "what does a partial wave
//    cost?", not "how fast is the kernel?".
//  - The address model is one warp instruction at a time, with no caches: a
//    sector fetched by one instruction is fetched again by the next.
//
// Erasable TypeScript only, and the only value import is lib/locale.ts: the browser
// checks import this file directly in Node. Sector counts come from coalesce.ts (warpAccess), which
// the lesson and the unit tests apply to the addresses produced here.

import { tr } from '../lib/locale.ts';

/** The book's hypothetical device: it runs 264 blocks at the same time (p. 143). */
export const BOOK_CAPACITY = 264;
/** NVIDIA's coalescing unit (ch06-facts A2). */
export const SECTOR_BYTES = 32;

export interface Grid {
  /** Threads needed: one per `factor` units of work, rounded up. */
  threads: number;
  /** Blocks of `threadsPerBlock`, rounded up. */
  blocks: number;
}

/** How many threads and blocks a coarsened launch needs. */
export function grid(units: number, threadsPerBlock: number, factor: number): Grid {
  const threads = Math.ceil(units / factor);
  return { threads, blocks: Math.ceil(threads / threadsPerBlock) };
}

/** Waves as the book counts them: blocks divided by blocks that run at once (may be fractional). */
export const waves = (blocks: number, capacity: number): number => blocks / capacity;

export interface WaveFill {
  /** Blocks running in each wave, first to last. */
  perWave: number[];
  /** Waves that start, counting a partial last wave as one. */
  waveCount: number;
  /** Whole waves only: no partial last wave. */
  whole: boolean;
  /** Share of block slots left idle over all the waves (0 to 1). */
  idle: number;
  /** Blocks in the partial last wave, or 0. */
  tail: number;
}

/** Which slots each wave fills, assuming equal-duration blocks and a fixed capacity. */
export function waveFill(blocks: number, capacity: number): WaveFill {
  const waveCount = Math.ceil(blocks / capacity);
  const perWave: number[] = [];
  for (let w = 0; w < waveCount; w++) perWave.push(Math.min(capacity, blocks - w * capacity));
  const tail = blocks % capacity;
  return {
    perWave,
    waveCount,
    whole: blocks > 0 && tail === 0,
    idle: waveCount ? 1 - blocks / (waveCount * capacity) : 0,
    tail,
  };
}

/**
 * Time in units of one uncoarsened block, if a coarsened block takes `factor` times
 * as long and coarsening saved nothing: waves that start, times the factor.
 */
export const waveTime = (blocks: number, capacity: number, factor: number): number =>
  Math.ceil(blocks / capacity) * factor;

/** Everything the wave calculator shows for one setting. */
export function calculate(units: number, threadsPerBlock: number, factor: number, capacity: number) {
  const g = grid(units, threadsPerBlock, factor);
  const base = grid(units, threadsPerBlock, 1);
  return {
    ...g,
    waves: waves(g.blocks, capacity),
    fill: waveFill(g.blocks, capacity),
    time: waveTime(g.blocks, capacity, factor),
    baseTime: waveTime(base.blocks, capacity, 1),
  };
}

/** Factors from 1 to `max` that leave only whole waves (every slot of every wave filled). */
export function wholeWaveFactors(units: number, threadsPerBlock: number, capacity: number, max = 64): number[] {
  const out: number[] = [];
  for (let f = 1; f <= max; f++) if (waveFill(grid(units, threadsPerBlock, f).blocks, capacity).whole) out.push(f);
  return out;
}

/** The largest factor up to `max` that leaves only whole waves, or 0 if none does. */
export function largestWholeWaveFactor(units: number, threadsPerBlock: number, capacity: number, max = 64): number {
  const fs = wholeWaveFactors(units, threadsPerBlock, capacity, max);
  return fs.length ? fs[fs.length - 1] : 0;
}

/** The book's two worked cases (p. 143), as work units at 256 threads per block. */
export const BOOK_CASES = [
  { get label() { return tr('1056 blocks', '1056 khối'); }, blocks: 1056, factor: 8 },
  { get label() { return tr('1584 blocks', '1584 khối'); }, blocks: 1584, factor: 4 },
] as const;

// ---------------------------------------------------------------- the reader's device
export interface DeviceSlots {
  /** SMs (CUDA) or what multiProcessorCount holds (HIP). */
  units: number;
  /** maxBlocksPerMultiProcessor: block slots on CUDA, wave slots on HIP (ch04-facts B9). */
  blocksPerUnit: number;
  /** maxThreadsPerMultiProcessor, or 0 when not reported. */
  threadsPerUnit: number;
  /** warpSize. */
  warpWidth: number;
}

/**
 * An upper bound on blocks resident at once for this block size: block slots, and
 * thread slots after rounding the block up to whole warps (ch04-facts B10), times
 * the number of SMs. Registers and shared memory can only lower it.
 */
export function residentUpperBound(d: DeviceSlots, threadsPerBlock: number): number {
  const w = d.warpWidth > 0 ? d.warpWidth : 1;
  const rounded = Math.ceil(threadsPerBlock / w) * w;
  let perUnit = d.blocksPerUnit;
  if (d.threadsPerUnit > 0) perUnit = Math.min(perUnit, Math.floor(d.threadsPerUnit / rounded));
  return Math.max(0, perUnit) * d.units;
}

/** The fields a CUDA or HIP device report must carry for the bound, as numbers; null if any is missing. */
export function deviceSlots(device: Record<string, unknown>): DeviceSlots | null {
  const n = (k: string) => {
    const v = Number(device[k]);
    return isFinite(v) && v > 0 ? Math.floor(v) : 0;
  };
  const d: DeviceSlots = {
    units: n('multiProcessorCount'),
    blocksPerUnit: n('maxBlocksPerMultiProcessor'),
    threadsPerUnit: n('maxThreadsPerMultiProcessor'),
    warpWidth: n('warpSize'),
  };
  return d.units && d.blocksPerUnit ? d : null;
}

// ---------------------------------------------------------------- who handles what
export type Mapping = 'contiguous' | 'strided';

/**
 * The elements one thread handles. Contiguous is the book's toy (p. 142):
 * iStart = factor x thread, then iStart + c. Strided is ours: thread + c x totalThreads.
 */
export function elementsOf(thread: number, factor: number, mapping: Mapping, totalThreads: number): number[] {
  return Array.from({ length: factor }, (_, c) => (mapping === 'contiguous' ? factor * thread + c : thread + c * totalThreads));
}

/** Element index lane `lane` of warp `warp` touches in loop iteration c. */
export function laneElement(lane: number, c: number, factor: number, mapping: Mapping, warpWidth: number, totalThreads: number, warp = 0): number {
  const t = warp * warpWidth + lane;
  return mapping === 'contiguous' ? factor * t + c : t + c * totalThreads;
}

/** Byte addresses one warp's scalar load touches in loop iteration c (4-byte floats). */
export function warpInstruction(c: number, factor: number, mapping: Mapping, warpWidth: number, totalThreads: number, warp = 0): number[] {
  return Array.from({ length: warpWidth }, (_, lane) => 4 * laneElement(lane, c, factor, mapping, warpWidth, totalThreads, warp));
}

/** Byte addresses of one warp's 16-byte vector load: each lane reads its own four floats at once (Fig. 6.12). */
export function warpVectorInstruction(warpWidth: number, warp = 0): number[] {
  return Array.from({ length: warpWidth }, (_, lane) => 16 * (warp * warpWidth + lane));
}

// ---------------------------------------------------------------- pitfall cards
export type Verdict = 'helps' | 'little or no change' | 'can hurt';
export const VERDICTS: Verdict[] = ['helps', 'little or no change', 'can hurt']; // i18n-ignore: identifier also shown to the reader (Verdict union member)

export interface Pitfall {
  id: string;
  scenario: string;
  answer: Verdict;
  why: string;
  pages: string;
}

/** Predict-then-reveal scenarios. Answers are the book's (pages given); the occupancy figures are its example SM. */
export const PITFALLS: Pitfall[] = [
  {
    id: 'relu',
    get scenario() {
      return tr(
        'A ReLU kernel: one element per thread, no shared data, loads already coalesced. Coarsen it by 4 with scalar loads.',
        'Một kernel ReLU: mỗi luồng một phần tử, không có dữ liệu dùng chung, các lần nạp đã được hợp nhất. Gộp việc cho nó theo hệ số 4 với các lần nạp vô hướng.',
      );
    },
    answer: 'little or no change', // i18n-ignore: identifier also shown to the reader (Verdict union member)
    get why() {
      return tr(
        'The threads are independent. There is no parallelisation overhead for coarsening to remove (pitfall 1).',
        'Các luồng độc lập với nhau. Không có chi phí song song hóa nào để việc gộp loại bỏ (cạm bẫy 1).',
      );
    },
    pages: '142–143',
  },
  {
    id: 'vector',
    get scenario() {
      return tr(
        'Vector addition at full occupancy, already moving bytes at about your measured bandwidth. Coarsen it by 4 so that each thread adds four neighbouring elements with one 16-byte load per input and one 16-byte store (Fig. 6.12).',
        'Phép cộng vector ở occupancy đầy đủ, vốn đã chuyển byte ở mức xấp xỉ băng thông bạn đo được. Gộp việc theo hệ số 4 để mỗi luồng cộng bốn phần tử liền kề, với một lần nạp 16 byte cho mỗi đầu vào và một lần ghi 16 byte (Fig. 6.12).',
      );
    },
    answer: 'little or no change', // i18n-ignore: identifier also shown to the reader (Verdict union member)
    get why() {
      return tr(
        'It does remove a real overhead: four load and store instructions become one, which is why the book counts Fig. 6.12 as coarsening (p. 142). But the bytes are the same and moving them was already the limit. Fewer instructions and more bytes per request pay where they were the limit, as below full occupancy (p. 139). Sections 6.3 and 6.9 reach the same verdict, and lab 6.2 measures it.',
        'Nó thật sự loại bỏ một chi phí: bốn lệnh nạp và ghi trở thành một, đó là lý do sách coi Fig. 6.12 là gộp việc (p. 142). Nhưng số byte vẫn như cũ, và việc chuyển chúng vốn đã là giới hạn. Ít lệnh hơn và nhiều byte hơn cho mỗi yêu cầu có lợi ở nơi chúng là giới hạn, như khi dưới occupancy đầy đủ (p. 139). Mục 6.3 và 6.9 đi đến cùng kết luận, và lab 6.2 đo điều đó.',
      );
    },
    pages: '139, 142',
  },
  {
    id: 'matmul-tiles',
    get scenario() {
      return tr(
        "Chapter 5's tiled matrix multiplication on a large matrix. Each block now computes two neighbouring output tiles, so the tile of M it loads serves both. The grid still fills many waves, and occupancy does not change.",
        'Phép nhân ma trận phân ô của chương 5 trên một ma trận lớn. Giờ mỗi khối tính hai ô đầu ra liền kề, nên ô của M mà nó nạp phục vụ cả hai. Lưới vẫn lấp đầy nhiều wave, và occupancy không đổi.',
      );
    },
    answer: 'helps',
    get why() {
      return tr(
        'This removes the overhead the book names for it: neighbouring blocks loading the same input tiles. Each tile of M now crosses from global memory half as often. Chapter 15 takes it further.',
        'Việc này loại bỏ đúng chi phí mà sách nêu cho nó: các khối liền kề nạp cùng các ô đầu vào. Giờ số lần mỗi ô của M phải đi ra từ bộ nhớ toàn cục giảm một nửa. Chương 15 đi xa hơn nữa.',
      );
    },
    pages: '142',
  },
  {
    id: 'below-wave',
    get scenario() {
      return tr(
        "A grid of 1056 blocks on the book's device that runs 264 blocks at once. Coarsen it by 8.",
        'Một lưới 1056 khối trên thiết bị của sách, thiết bị chạy 264 khối cùng lúc. Gộp việc theo hệ số 8.',
      );
    },
    answer: 'can hurt', // i18n-ignore: identifier also shown to the reader (Verdict union member)
    get why() {
      return tr(
        '132 blocks are half a wave: half the device sits idle while each block does eight times the work (pitfall 2).',
        '132 khối là nửa wave: một nửa thiết bị nhàn rỗi trong khi mỗi khối làm gấp tám lần việc (cạm bẫy 2).',
      );
    },
    pages: '143',
  },
  {
    id: 'registers',
    get scenario() {
      return tr(
        "Coarsening by 2 raises registers per thread from 32 to 64 on the book's example SM (65,536 registers, 2048 thread slots), and removes little overhead.",
        'Gộp việc theo hệ số 2 làm số thanh ghi mỗi luồng tăng từ 32 lên 64 trên SM ví dụ của sách (65,536 thanh ghi, 2048 chỗ cho luồng), và loại bỏ được ít chi phí.',
      );
    },
    answer: 'can hurt', // i18n-ignore: identifier also shown to the reader (Verdict union member)
    get why() {
      return tr(
        'Only 1024 threads now fit: occupancy halves. The gain has to exceed what the lost warps were hiding, and here there is little gain (pitfall 3).',
        'Giờ chỉ còn 1024 luồng vừa: occupancy giảm một nửa. Phần lợi phải vượt qua những gì các warp bị mất đang che giấu, và ở đây lợi rất ít (cạm bẫy 3).',
      );
    },
    pages: '143',
  },
];

// ---------------------------------------------------------------- the book's occupancy example (ch. 4, pp. 85 to 86)
/** Threads one SM can hold when each thread needs `regsPerThread` registers, capped by the thread slots. */
export function threadsByRegisters(regsPerSM: number, threadSlots: number, regsPerThread: number): number {
  return Math.min(threadSlots, Math.floor(regsPerSM / regsPerThread));
}
