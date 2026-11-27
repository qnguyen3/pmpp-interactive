// Chapter 4, sections 4.1 to 4.3: what a device is made of, as far as a program can
// ask, and how the blocks of a grid are handed to it. It runs the level explorer of
// section 4.1, the block stepper of section 4.2, the wave explorer and the narrated
// animation "One grid, two devices" of section 4.3, and the numbers in their questions.
//
// Sources for the rules (the lessons cite them where they are used):
//  - the book, section 4.1, p. 68: a device is an array of SMs, each a set of
//    streaming processors; the example device has 132 SMs of 128, in 8 GPCs;
//  - the book, section 4.2, p. 69: all threads of a block go to one SM at the same
//    time; an SM holds several blocks; the runtime keeps a list of blocks that have
//    not started and assigns one whenever an earlier block finishes;
//  - the book, section 4.3, pp. 73 to 74: blocks may run in any order; a wave is a
//    group of blocks running at once; waves = blocks / blocks that run at once; an
//    incomplete last wave leaves capacity idle (660 blocks at 264 at once = 2.5);
//  - the book, sections 4.4, 4.6 and 4.7, pp. 76 and 85: four processing blocks of
//    32 per SM, 2048 resident threads and 32 resident blocks per SM on the example;
//  - notes/today/ch04-facts.md A2: NVIDIA's level between GPC and SM, the TPC, and
//    the counts for the two H100 variants; A5 and B2: no vendor promises an order
//    or a placement; B1, B9, C1: what the same fields mean on AMD, and that Metal
//    reports none of them.
//
// What this leaves out:
//  - A block is an atom: it occupies one slot on one SM from its start to its end.
//    Its threads, warps, registers and shared memory are not modelled; "slots per
//    SM" stands for whatever limit binds (section 4.7 takes that apart).
//  - Every block of a grid asks for the same resources, so every SM has the same
//    number of slots.
//  - Placement is a choice of this model: free slots are filled lowest slot first,
//    going round the SMs. No vendor documents the real policy, and NVIDIA says it
//    can be neither controlled nor queried.
//  - Time is the list of block durations you give it. Nothing here predicts how
//    long a block takes, and a block that is running is never suspended.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- devices
export type Vocabulary = 'cuda' | 'hip' | 'metal';

/** What is known about a device's levels. null means "nobody told us": not reported, or not stated. */
export interface Device {
  name: string;
  /** Where the numbers come from: the book's example, a device report of this machine, or nowhere. */
  source: 'book' | 'report' | 'none';
  vocabulary: Vocabulary;
  /** SMs on CUDA; on HIP what multiProcessorCount holds (compute units, or half of them). */
  units: number | null;
  /** GPCs. No device property reports it. */
  clusters: number | null;
  /** Streaming processors in one SM. No device property reports it. */
  coresPerUnit: number | null;
  /** Processing blocks in one SM. No device property reports it. */
  groupsPerUnit: number | null;
  /** Threads one SM keeps in progress at once. */
  threadsPerUnit: number | null;
  /** Blocks one SM keeps in progress at once. */
  blocksPerUnit: number | null;
  /** Threads in a warp. */
  groupWidth: number | null;
}

/** The book's example device, every number as the book states it (pp. 68, 76, 85). */
export const BOOK_DEVICE: Device = {
  get name() {
    return tr("the book's example", 'ví dụ của sách');
  },
  source: 'book',
  vocabulary: 'cuda',
  units: 132,
  clusters: 8,
  coresPerUnit: 128,
  groupsPerUnit: 4,
  threadsPerUnit: 2048,
  blocksPerUnit: 32,
  groupWidth: 32,
};

/**
 * What NVIDIA publishes today about the device the book's numbers describe
 * (notes/today/ch04-facts.md, A2): a level the book leaves out, the TPC of 2 SMs,
 * and a second variant of the same product with fewer SMs.
 */
export const BOOK_DEVICE_TODAY = { clusters: 8, tpcs: 66, unitsPerTpc: 2, unitsOtherVariant: 114 } as const;

/**
 * AMD's published description of a compute unit of its CDNA family (facts B1): four
 * SIMDs of 16 lanes. It is a statement about a product family, not a device report.
 */
export const AMD_CDNA_DOC = { simdsPerCu: 4, lanesPerSimd: 16 } as const;

const product = (a: number | null, b: number | null): number | null => (a == null || b == null ? null : a * b);
const ratio = (a: number | null, b: number | null): number | null => (a == null || b == null || b === 0 ? null : a / b);

/** Streaming processors on the whole device. */
export const totalCores = (d: Device): number | null => product(d.units, d.coresPerUnit);
/** Threads the whole device keeps in progress at once. */
export const residentThreads = (d: Device): number | null => product(d.units, d.threadsPerUnit);
/** Resident threads for each streaming processor of an SM. */
export const threadsPerCore = (d: Device): number | null => ratio(d.threadsPerUnit, d.coresPerUnit);
/** Streaming processors in one processing block. */
export const coresPerGroup = (d: Device): number | null => ratio(d.coresPerUnit, d.groupsPerUnit);
/** Blocks the whole device keeps in progress at once, when every SM is full. */
export const residentBlocks = (d: Device): number | null => product(d.units, d.blocksPerUnit);

/** parts / groups when it is a whole number, else null: "these cannot all be the same size". */
export const evenSplit = (parts: number, groups: number): number | null =>
  groups > 0 && Number.isInteger(parts / groups) ? parts / groups : null;

/**
 * The report names that are GPU backends. The reports directory also holds the CPU
 * emulators ("emu", "race"), whose device fields are synthetic and must never be
 * shown as a device.
 */
export const GPU_BACKENDS: readonly string[] = ['cuda', 'hip', 'metal'];
export const isGpuBackend = (name: string): boolean => GPU_BACKENDS.includes(name);

const positiveInt = (v: unknown): number | null => {
  const n = typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN;
  return Number.isInteger(n) && n > 0 ? n : null;
};

/**
 * A device as one of this machine's reports describes it (reports/runs/<backend>.json,
 * written by labs/pmpp.py doctor). A missing field and a field of 0 both become null:
 * AMD's reference says some of these fields can read 0 (facts B9). The Metal report has
 * none of them, because MTLDevice has no such properties (facts C1).
 */
export function deviceFromReport(backend: string, raw: Record<string, unknown>): Device {
  const vocabulary: Vocabulary = backend === 'metal' ? 'metal' : backend === 'hip' ? 'hip' : 'cuda';
  const get = (k: string) => (vocabulary === 'metal' ? null : positiveInt(raw[k]));
  return {
    name: typeof raw.name === 'string' && raw.name ? raw.name : backend,
    source: 'report',
    vocabulary,
    units: get('multiProcessorCount'),
    clusters: null,
    coresPerUnit: null,
    groupsPerUnit: null,
    threadsPerUnit: get('maxThreadsPerMultiProcessor'),
    blocksPerUnit: get('maxBlocksPerMultiProcessor'),
    groupWidth: get('warpSize'),
  };
}

/** A backend this machine has no device report for: the names are known, the counts are not. */
export const noReport = (vocabulary: Vocabulary): Device => ({
  get name() {
    return tr('no device report on this machine', 'không có báo cáo thiết bị trên máy này');
  },
  source: 'none',
  vocabulary,
  units: null,
  clusters: null,
  coresPerUnit: null,
  groupsPerUnit: null,
  threadsPerUnit: null,
  blocksPerUnit: null,
  groupWidth: null,
});

// ---------------------------------------------------------------- levels, by vendor
export type LevelId = 'device' | 'cluster' | 'unit' | 'group' | 'core' | 'onchip' | 'global';
export const LEVEL_ORDER: readonly LevelId[] = ['device', 'cluster', 'unit', 'group', 'core', 'onchip', 'global'];

/** Where a count on a level row comes from. */
export type Provenance = 'book' | 'reported' | 'vendor-doc' | 'none';

export interface Level {
  id: LevelId;
  /** The vendor's name for the level, or null where the vendor documents no such level. */
  name: string | null;
  /** The book's (CUDA) short name for the level, so a row without a name can still say what is missing. */
  bookName: string;
  /** How many, or null. */
  count: number | null;
  /** What the count is a count of, e.g. "on the device", "in each SM". */
  per: string;
  provenance: Provenance;
  /** How a program asks for the count on this backend, or null when it cannot. */
  ask: string | null;
  /** What the later sections of the chapter place on this level. */
  holds: { what: string; where: string }[];
  /** One sentence of caution about the row. */
  note: string;
}

/** The software words of each backend: what it calls a block, a warp and the unit a block runs on. */
export const TERMS: Record<Vocabulary, { block: string; warp: string; unit: string | null; api: string }> = {
  cuda: { block: 'block', warp: 'warp', unit: 'SM', api: 'CUDA' },
  hip: { block: 'block', warp: 'warp', unit: 'CU', get api() { return tr('HIP on AMD', 'HIP trên AMD'); } },
  metal: { block: 'threadgroup', warp: 'SIMD-group', unit: null, api: 'Metal' },
};

const NAMES: Record<Vocabulary, Record<LevelId, string | null>> = {
  cuda: {
    device: 'GPU',
    cluster: 'GPC',
    get unit() { return tr('streaming multiprocessor (SM)', 'streaming multiprocessor (SM)'); },
    get group() { return tr('processing block', 'khối xử lý (processing block)'); },
    get core() { return tr('streaming processor', 'bộ xử lý luồng (streaming processor)'); },
    get onchip() { return tr('on-chip memory of an SM', 'bộ nhớ trên chip của một SM'); },
    get global() { return tr('global memory', 'bộ nhớ toàn cục (global memory)'); },
  },
  hip: {
    device: 'GPU',
    get cluster() { return tr('shader engine and shader array', 'shader engine và shader array'); },
    get unit() { return tr('compute unit (CU)', 'compute unit (CU)'); },
    group: 'SIMD',
    get core() { return tr('ALU lane', 'lane ALU'); },
    get onchip() { return tr('registers and LDS of a CU', 'thanh ghi (register) và LDS của một CU'); },
    get global() { return tr('global memory', 'bộ nhớ toàn cục (global memory)'); },
  },
  metal: {
    device: 'MTLDevice',
    cluster: null,
    unit: null,
    group: null,
    core: null,
    get onchip() { return tr('threadgroup memory', 'bộ nhớ threadgroup'); },
    get global() { return tr('device memory (buffers)', 'bộ nhớ device (các buffer)'); },
  },
};

/** The book's short name for each level, used where another vendor has no name of its own. */
export const BOOK_NAMES: Record<LevelId, string> = {
  device: 'GPU',
  cluster: 'GPC',
  unit: 'SM',
  get group() { return tr('processing block', 'khối xử lý (processing block)'); },
  get core() { return tr('streaming processor', 'bộ xử lý luồng (streaming processor)'); },
  get onchip() { return tr('on-chip memory', 'bộ nhớ trên chip'); },
  get global() { return tr('global memory', 'bộ nhớ toàn cục (global memory)'); },
};

/**
 * The levels of a device in its own vendor's words, with a count where one is known,
 * where that count comes from, and how a program would ask for it.
 */
export function levels(d: Device): Level[] {
  const v = d.vocabulary;
  const t = TERMS[v];
  const from = (n: number | null): Provenance => (n == null ? 'none' : d.source === 'book' ? 'book' : 'reported');
  const undocumented = tr(
    'Apple documents threadgroups and SIMD-groups, which are software, and no hardware unit under the device.',
    'Apple có tài liệu về threadgroup và SIMD-group, vốn là phần mềm, và không có tài liệu về đơn vị phần cứng nào bên dưới thiết bị.',
  );
  const rows: Record<LevelId, Omit<Level, 'id' | 'name' | 'bookName'>> = {
    device: {
      count: d.source === 'none' ? null : 1,
      per: tr('selected', 'đang chọn'),
      provenance: d.source === 'book' ? 'book' : d.source === 'report' ? 'reported' : 'none',
      ask:
        v === 'cuda' ? 'cudaGetDeviceCount'
        : v === 'hip' ? 'hipGetDeviceCount'
        : 'MTLCopyAllDevices',
      holds: [{ what: tr(`the grid: every ${t.block} that has not started waits here for room`, `lưới (grid): mọi ${v === 'metal' ? 'threadgroup' : 'khối'} chưa bắt đầu chờ ở đây cho đến khi có chỗ`), where: '4.2' }],
      note: tr('One device at a time: a kernel launch goes to one device.', 'Mỗi lần một thiết bị: một lần khởi chạy kernel đi đến một thiết bị.'),
    },
    cluster: {
      count: d.clusters,
      per: tr('on the device', 'trên thiết bị'),
      provenance: from(d.clusters),
      ask: null,
      holds: v === 'metal' ? [] : [{ what: tr('the blocks of one thread block cluster, scheduled together', 'các khối của một thread block cluster, được lập lịch cùng nhau'), where: '4.2' }],
      note:
        v === 'cuda' ? tr('NVIDIA expands GPC as both "GPU processing cluster" and "graphics processing cluster".', 'NVIDIA viết đầy đủ GPC là cả "GPU processing cluster" lẫn "graphics processing cluster".')
        : v === 'hip' ? tr('Two AMD levels above the CU. Setting them beside a GPC is our comparison, not AMD\'s.', 'Hai tầng của AMD phía trên CU. Đặt chúng cạnh một GPC là phép so sánh của chúng tôi, không phải của AMD.')
        : undocumented,
    },
    unit: {
      count: d.units,
      per: tr('on the device', 'trên thiết bị'),
      provenance: from(d.units),
      ask: v === 'metal' ? null : 'multiProcessorCount',
      holds:
        v === 'metal'
          ? []
          : [
              { what: tr(`whole ${t.block}s, a few at a time`, `trọn vẹn từng khối, mỗi lần vài cái`), where: '4.2' },
              { what: tr(`the ${t.warp}s of those ${t.block}s, more than can execute at once`, `các ${t.warp} của những khối đó, nhiều hơn số có thể thực thi cùng lúc`), where: '4.6' },
              { what: tr('a register file shared out among the resident threads', 'một tệp thanh ghi (register file) được chia cho các luồng đang thường trú'), where: '4.7' },
              { what: tr(`shared memory for each resident ${t.block}`, `bộ nhớ chia sẻ (shared memory) cho mỗi khối đang thường trú`), where: 'chapter 5' }, // i18n-ignore: identifier also shown to the reader
            ],
      note:
        v === 'cuda' ? tr('The one size of this picture that every CUDA device reports.', 'Kích thước duy nhất trong bức tranh này mà mọi thiết bị CUDA đều báo cáo.')
        : v === 'hip' ? tr('The field counts compute units, or half of them when the device runs in workgroup-processor mode.', 'Trường này đếm số compute unit, hoặc một nửa số đó khi thiết bị chạy ở chế độ workgroup-processor.')
        : undocumented,
    },
    group: {
      count: v === 'hip' ? AMD_CDNA_DOC.simdsPerCu : d.groupsPerUnit,
      per: v === 'hip' ? tr('in each CU', 'trong mỗi CU') : tr('in each SM', 'trong mỗi SM'),
      provenance: v === 'hip' ? 'vendor-doc' : from(d.groupsPerUnit),
      ask: null,
      holds: v === 'metal' ? [] : [{ what: tr(`one ${t.warp} at a time, every thread of it on the same instruction`, `mỗi lần một ${t.warp}, mọi luồng của nó cùng ở một lệnh`), where: '4.4' }],
      note:
        v === 'cuda' ? tr('Only the book states this count for its example; NVIDIA\'s public text for that device does not.', 'Chỉ sách nêu con số này cho ví dụ của nó; tài liệu công khai của NVIDIA về thiết bị đó thì không.')
        : v === 'hip' ? tr('AMD\'s description of its CDNA family, not a property of your device.', 'Mô tả của AMD về dòng CDNA của họ, không phải một thuộc tính của thiết bị của bạn.')
        : undocumented,
    },
    core: {
      count: v === 'hip' ? AMD_CDNA_DOC.simdsPerCu * AMD_CDNA_DOC.lanesPerSimd : d.coresPerUnit,
      per: v === 'hip' ? tr('in each CU', 'trong mỗi CU') : tr('in each SM', 'trong mỗi SM'),
      provenance: v === 'hip' ? 'vendor-doc' : from(d.coresPerUnit),
      ask: null,
      holds: v === 'metal' ? [] : [{ what: tr(`one thread of that ${t.warp}, for one instruction`, `một luồng của ${t.warp} đó, cho một lệnh`), where: '4.4' }],
      note:
        v === 'cuda'
          ? tr(
              'Also sold as "CUDA cores"; the pages of NVIDIA\'s current guide read for this course do not say "streaming processor". No property of a CUDA device reports how many an SM has.',
              'Cũng được bán dưới tên "CUDA cores"; các trang trong hướng dẫn hiện hành của NVIDIA mà khóa học này đã đọc không nói "streaming processor". Không thuộc tính nào của thiết bị CUDA báo cáo một SM có bao nhiêu cái.',
            )
        : v === 'hip' ? tr('AMD\'s description of its CDNA family: 16 lanes in each SIMD. Not a property of your device.', 'Mô tả của AMD về dòng CDNA của họ: 16 lane trong mỗi SIMD. Không phải một thuộc tính của thiết bị của bạn.')
        : undocumented,
    },
    onchip: {
      count: null,
      per: tr('a size, not a count', 'một kích thước, không phải một số đếm'),
      provenance: 'none',
      ask: v === 'metal' ? 'maxThreadgroupMemoryLength' : 'sharedMemPerBlock, regsPerMultiprocessor', // i18n-ignore: API property names
      holds:
        v === 'metal'
          ? [{ what: tr('arrays declared threadgroup, one copy per threadgroup', 'các mảng khai báo threadgroup, mỗi threadgroup một bản'), where: 'chapter 5' }] // i18n-ignore: identifier also shown to the reader
          : [
              { what: tr('registers: the automatic variables of every resident thread', 'thanh ghi: các biến tự động của mọi luồng đang thường trú'), where: '4.7' },
              { what: tr(`shared memory: one copy per resident ${t.block}`, `bộ nhớ chia sẻ: mỗi khối đang thường trú một bản`), where: 'chapter 5' }, // i18n-ignore: identifier also shown to the reader
            ],
      note: tr(
        'Sizes belong to chapter 5. What matters here: it is inside the unit a block runs on, so only that block\'s threads can share it.',
        'Kích thước thuộc về chương 5. Điều quan trọng ở đây: nó nằm bên trong đơn vị mà một khối chạy trên đó, nên chỉ các luồng của khối đó mới chia sẻ được nó.',
      ),
    },
    global: {
      count: null,
      per: tr('a size, not a count', 'một kích thước, không phải một số đếm'),
      provenance: 'none',
      ask: v === 'metal' ? 'maxBufferLength' : 'totalGlobalMem',
      holds: [{ what: tr('the arrays behind a kernel\'s pointer arguments', 'các mảng đằng sau các đối số con trỏ của một kernel'), where: '4.6' }],
      note: tr('Off the chip and shared by every thread of the grid. Every access to it is slow next to arithmetic.', 'Nằm ngoài chip và được mọi luồng của lưới chia sẻ. Mỗi lần truy cập nó đều chậm so với phép tính số học.'),
    },
  };
  return LEVEL_ORDER.map((id) => ({ id, name: NAMES[v][id], bookName: BOOK_NAMES[id], ...rows[id] }));
}

// ---------------------------------------------------------------- waves (closed form)
/** Blocks in progress at once when every slot is taken. */
export const capacity = (units: number, slotsPerUnit: number): number => Math.max(0, units * slotsPerUnit);

/** Blocks still on the runtime's list just after launch. */
export const waitingAtLaunch = (blocks: number, cap: number): number => Math.max(0, blocks - cap);

/** The book's count of waves: blocks in the grid over blocks that run at once. */
export const waves = (blocks: number, cap: number): number => (cap > 0 ? blocks / cap : Infinity);

/** How many blocks each wave holds when all blocks take the same time. */
export function waveSizes(blocks: number, cap: number): number[] {
  const out: number[] = [];
  if (cap <= 0) return out;
  for (let left = blocks; left > 0; left -= cap) out.push(Math.min(cap, left));
  return out;
}

export interface Tail {
  /** Blocks in the last wave. */
  last: number;
  /** Slots with nothing to run during the last wave. */
  idle: number;
  /** idle / capacity. */
  idleShare: number;
}
/** The last wave of a grid of equal blocks. */
export function tail(blocks: number, cap: number): Tail {
  const sizes = waveSizes(blocks, cap);
  const last = sizes.length ? sizes[sizes.length - 1] : 0;
  const idle = sizes.length ? cap - last : 0;
  return { last, idle, idleShare: cap > 0 ? idle / cap : 0 };
}

// ---------------------------------------------------------------- durations and orders
/** A small seeded generator (mulberry32), so "uneven" is the same every time. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Every block takes one unit of time. */
export const equalDurations = (blocks: number): number[] => Array.from({ length: blocks }, () => 1);

/** Each block takes 1, 2, 3 or 4 units of time, picked by a seeded generator. */
export function unevenDurations(blocks: number, seed = 4): number[] {
  const r = rng(seed);
  return Array.from({ length: blocks }, () => 1 + Math.floor(r() * 4));
}

/** The block numbers 0 to n - 1 in a seeded random order. */
export function shuffledOrder(n: number, seed = 1): number[] {
  const r = rng(seed);
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ---------------------------------------------------------------- the scheduler
export interface GridSpec {
  /** Blocks in the grid. */
  blocks: number;
  /** SMs on the device. */
  units: number;
  /** Blocks one SM holds at once, for this kernel. */
  slotsPerUnit: number;
  /** Time each block takes; one unit each when left out. */
  durations?: number[];
  /** The order in which the runtime hands blocks out; 0, 1, 2, ... when left out. */
  order?: number[];
}

export interface Placement {
  block: number;
  unit: number;
  slot: number;
  start: number;
  end: number;
}

export interface SchedEvent {
  kind: 'assign' | 'complete';
  block: number;
  unit: number;
  slot: number;
  time: number;
}

export interface Schedule {
  spec: Required<GridSpec>;
  capacity: number;
  /** One per block, indexed by block number. */
  placements: Placement[];
  /** Every assignment and completion, in the order they happen. */
  events: SchedEvent[];
  /** When the last block finishes. */
  makespan: number;
}

/** A binary min-heap on (key, tie). */
class Heap<T> {
  private a: { key: number; tie: number; v: T }[] = [];
  get size() {
    return this.a.length;
  }
  private less(i: number, j: number) {
    return this.a[i].key < this.a[j].key || (this.a[i].key === this.a[j].key && this.a[i].tie < this.a[j].tie);
  }
  push(key: number, tie: number, v: T) {
    const a = this.a;
    a.push({ key, tie, v });
    for (let i = a.length - 1; i > 0; ) {
      const p = (i - 1) >> 1;
      if (!this.less(i, p)) break;
      [a[i], a[p]] = [a[p], a[i]];
      i = p;
    }
  }
  pop(): T {
    const a = this.a;
    const top = a[0];
    const last = a.pop()!;
    if (a.length) {
      a[0] = last;
      for (let i = 0; ; ) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < a.length && this.less(l, m)) m = l;
        if (r < a.length && this.less(r, m)) m = r;
        if (m === i) break;
        [a[i], a[m]] = [a[m], a[i]];
        i = m;
      }
    }
    return top.v;
  }
}

/**
 * Hands the blocks of a grid to the device. A block takes one free slot on one SM
 * and keeps it until it ends; the next block on the list takes a slot as soon as
 * one is free.
 */
export function schedule(spec: GridSpec): Schedule {
  const blocks = Math.max(0, Math.floor(spec.blocks));
  const units = Math.max(1, Math.floor(spec.units));
  const slotsPerUnit = Math.max(1, Math.floor(spec.slotsPerUnit));
  const durations = spec.durations ?? equalDurations(blocks);
  const order = spec.order ?? Array.from({ length: blocks }, (_, i) => i);
  if (durations.length !== blocks) throw new Error('one duration per block is needed');
  if (order.length !== blocks || new Set(order).size !== blocks || order.some((b) => !(b >= 0 && b < blocks)))
    throw new Error('the order must name every block once');
  const cap = capacity(units, slotsPerUnit);

  const placements: Placement[] = new Array(blocks);
  const events: SchedEvent[] = [];
  // Slot s is slot floor(s / units) of SM s % units: the first slot of every SM is used before any second slot.
  const free = new Heap<number>();
  for (let s = 0; s < cap; s++) free.push(s, 0, s);
  const running = new Heap<number>(); // block numbers, by end time then by start order
  let next = 0;
  let seq = 0;
  let makespan = 0;

  const assign = (time: number) => {
    const block = order[next++];
    const s = free.pop();
    const p: Placement = { block, unit: s % units, slot: Math.floor(s / units), start: time, end: time + durations[block] };
    placements[block] = p;
    events.push({ kind: 'assign', block, unit: p.unit, slot: p.slot, time });
    running.push(p.end, seq++, block);
  };

  while (next < blocks && free.size) assign(0);
  while (running.size) {
    const block = running.pop();
    const p = placements[block];
    makespan = Math.max(makespan, p.end);
    events.push({ kind: 'complete', block, unit: p.unit, slot: p.slot, time: p.end });
    free.push(p.slot * units + p.unit, 0, p.slot * units + p.unit);
    if (next < blocks) assign(p.end);
  }
  return { spec: { blocks, units, slotsPerUnit, durations, order }, capacity: cap, placements, events, makespan };
}

export interface SchedState {
  /** Events applied so far. */
  step: number;
  /** Time of the last event applied. */
  time: number;
  /** Blocks not yet started, in the order they will be handed out. */
  pending: number[];
  /** slots[unit][slot] is the block running there, or null. */
  slots: (number | null)[][];
  /** Blocks that have finished, in the order they finished. */
  done: number[];
  /** Blocks running now. */
  resident: number;
}

/** The device after the first n events of a schedule. */
export function stateAt(s: Schedule, n: number): SchedState {
  const { units, slotsPerUnit, order } = s.spec;
  const step = Math.max(0, Math.min(n, s.events.length));
  const slots: (number | null)[][] = Array.from({ length: units }, () => new Array<number | null>(slotsPerUnit).fill(null));
  const done: number[] = [];
  let assigned = 0;
  let time = 0;
  for (let i = 0; i < step; i++) {
    const e = s.events[i];
    time = e.time;
    if (e.kind === 'assign') {
      slots[e.unit][e.slot] = e.block;
      assigned++;
    } else {
      slots[e.unit][e.slot] = null;
      done.push(e.block);
    }
  }
  return { step, time, pending: order.slice(assigned), slots, done, resident: assigned - done.length };
}

export interface Segment {
  from: number;
  to: number;
  /** Blocks running during the whole of [from, to). */
  resident: number;
}

/**
 * Blocks in progress over time, cut at every moment a block starts or ends. With
 * equal blocks each segment is one wave.
 */
export function profile(s: Schedule): Segment[] {
  const delta = new Map<number, number>();
  for (const p of s.placements) {
    if (p.end <= p.start) continue;
    delta.set(p.start, (delta.get(p.start) ?? 0) + 1);
    delta.set(p.end, (delta.get(p.end) ?? 0) - 1);
  }
  const times = [...delta.keys()].sort((a, b) => a - b);
  const out: Segment[] = [];
  let resident = 0;
  for (let i = 0; i + 1 < times.length; i++) {
    resident += delta.get(times[i])!;
    out.push({ from: times[i], to: times[i + 1], resident });
  }
  return out;
}

export interface Usage {
  /** Sum of all block durations. */
  work: number;
  /** capacity x makespan: what the device could have done in the time it took. */
  available: number;
  /** work / available. */
  utilisation: number;
  /** The most blocks ever running at once. */
  peak: number;
  /** Busy time of each SM, summed over its slots. */
  perUnit: number[];
}

/** How much of the device's block capacity a schedule used from launch to the last block's end. */
export function usage(s: Schedule): Usage {
  const perUnit = new Array<number>(s.spec.units).fill(0);
  let work = 0;
  for (const p of s.placements) {
    work += p.end - p.start;
    perUnit[p.unit] += p.end - p.start;
  }
  const available = s.capacity * s.makespan;
  const peak = profile(s).reduce((m, g) => Math.max(m, g.resident), 0);
  return { work, available, utilisation: available > 0 ? work / available : 0, peak, perUnit };
}

/** The order in which blocks finish. */
export const completionOrder = (s: Schedule): number[] => s.events.filter((e) => e.kind === 'complete').map((e) => e.block);

// ---------------------------------------------------------------- the book's cases
/** Fig. 4.5: one grid of 8 blocks, a device that runs 2 at once and one that runs 4. */
export const FIG45 = { blocks: 8, small: { units: 2, slotsPerUnit: 1 }, large: { units: 4, slotsPerUnit: 1 } } as const;
/** Fig. 4.2's toy device: 2 SMs with 3 blocks on each. */
export const FIG42 = { units: 2, slotsPerUnit: 3 } as const;
/**
 * The tail-effect example of p. 73: 660 blocks where 264 run at once. The book
 * does not name the device; 132 SMs with 2 blocks each is one way to get 264.
 */
export const TAIL_EXAMPLE = { blocks: 660, units: 132, slotsPerUnit: 2 } as const;
