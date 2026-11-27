// Address widths for Appendix C, §C.4 and §C.7 (book pp. 625–628).
//
// Three questions, all plain arithmetic:
//   1. How many bytes can an address of a given width name? (reach)
//   2. How large an array can a kernel index with a given index type and element
//      size, before the index or the address runs out? (arrayLimit)
//   3. Does a CPU's user address space plus a GPU's memory fit, end to end, inside
//      a GPU virtual address of a given width? (coverage, smallestWidth)
//
// Sources for the labelled widths (MARKS):
//   - 32 bits, 4 GiB: early CUDA GPUs (book pp. 625–626).
//   - 40 bits, 1 TiB: Fermi's unified address space, "a Terabyte" (Fermi whitepaper,
//     notes/today/appC-facts.md A6).
//   - 48 bits, 256 TiB: x86-64 with 4-level paging (Linux docs, A7); the book's "48
//     bits actually used" (p. 626).
//   - 49 bits: Pascal's GPU virtual addresses (book pp. 627–628; NVIDIA, A11).
//   - 57 bits, 128 PiB: x86-64 with 5-level paging (Linux docs, A7).
//   - 47 and 56 bits: Linux user space by default, and with 5-level paging when a
//     program asks for it (A7).
//
// What the model leaves out, and what the app says wherever it is shown:
//   - "Reach" is addressable bytes only. Real devices and operating systems reserve
//     parts of an address space, and nothing here models that.
//   - Physical and virtual widths are not modelled separately.
//   - The coverage check packs the CPU range and the GPU range end to end. Real
//     layouts are not documented in our sources.
//
// No imports and only erasable TypeScript, so browser checks can load this file
// directly with Node.
import { tr } from '../lib/locale.ts';

export const KiB = 1024;
export const MiB = 1024 * KiB;
export const GiB = 1024 * MiB;
export const TiB = 1024 * GiB;
export const PiB = 1024 * TiB;

/** Bytes an address of `bits` bits can name: 2^bits. Exact up to 2^1023 as a double. */
export function reach(bits: number): number {
  return 2 ** bits;
}

const UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB'];

export interface Binary {
  value: number;
  unit: string;
  /** e.g. "4 GiB", "1.5 TiB", "8 GiB". */
  text: string;
}

/** A byte count in binary units (KiB = 1024 bytes), with at most three significant digits. */
export function binary(bytes: number): Binary {
  if (!isFinite(bytes) || bytes < 0) return { value: NaN, unit: 'B', text: 'n/a' };
  let k = 0;
  let v = bytes;
  while (v >= 1024 && k < UNITS.length - 1) {
    v /= 1024;
    k++;
  }
  let value = Number(v.toPrecision(3));
  // Three significant digits can round 1023.9 up to 1020, never to 1024, but be safe.
  if (value >= 1024 && k < UNITS.length - 1) {
    value = Number((v / 1024).toPrecision(3));
    k++;
  }
  return { value, unit: UNITS[k], text: `${value} ${UNITS[k]}` };
}

/** log2 of a positive byte count (for a log-scale bar). */
export function bitsOf(bytes: number): number {
  return bytes > 0 ? Math.log2(bytes) : 0;
}

export interface Mark {
  bits: number;
  /** The reach, in binary units. */
  label: string;
  what: string;
  /** Where the width comes from. */
  source: string;
}

export const MARKS: Mark[] = [
  { bits: 32, label: '4 GiB', get what() { return tr('early CUDA GPUs', 'các GPU CUDA đời đầu'); }, get source() { return tr('book pp. 625–626', 'sách pp. 625–626'); } }, // i18n-ignore: label is a byte figure in binary units
  { bits: 40, label: '1 TiB', get what() { return tr("Fermi's unified address space", 'không gian địa chỉ hợp nhất của Fermi'); }, get source() { return tr('Fermi whitepaper (appC-facts A6)', 'sách trắng Fermi (appC-facts A6)'); } }, // i18n-ignore: label is a byte figure in binary units
  { bits: 48, label: '256 TiB', get what() { return tr('x86-64, 4-level paging', 'x86-64, phân trang 4 cấp'); }, get source() { return tr('book p. 626; Linux docs (A7)', 'sách p. 626; tài liệu Linux (A7)'); } }, // i18n-ignore: label is a byte figure in binary units
  { bits: 49, label: '512 TiB', get what() { return tr("Pascal's GPU virtual addresses", 'địa chỉ ảo GPU của Pascal'); }, get source() { return tr('book pp. 627–628; NVIDIA (A11)', 'sách pp. 627–628; NVIDIA (A11)'); } }, // i18n-ignore: label is a byte figure in binary units
  { bits: 57, label: '128 PiB', get what() { return tr('x86-64, 5-level paging', 'x86-64, phân trang 5 cấp'); }, get source() { return tr('Linux docs (A7)', 'tài liệu Linux (A7)'); } }, // i18n-ignore: label is a byte figure in binary units
];

// ---------------------------------------------------------------- the index limit

export type IndexType = 'int32' | 'uint32' | 'int64';

export interface IndexInfo {
  key: IndexType;
  /** What a kernel would write. */
  c: string;
  /** Distinct non-negative indices the type can hold, as a power of two: 0 .. 2^log2Count − 1. */
  log2Count: number;
}

export const INDEX: Record<IndexType, IndexInfo> = {
  int32: { key: 'int32', c: 'int', log2Count: 31 },
  uint32: { key: 'uint32', c: 'unsigned int', log2Count: 32 }, // i18n-ignore: C type name
  int64: { key: 'int64', c: 'int64_t', log2Count: 63 },
};

export interface ElementType {
  /** Bytes per element; 0.5 for 4-bit values packed two to a byte. */
  bytes: number;
  label: string;
}

export const ELEMENTS: ElementType[] = [
  { bytes: 4, get label() { return tr('float, 4 bytes', 'float, 4 byte'); } },
  { bytes: 2, get label() { return tr('half or bfloat16, 2 bytes', 'half hoặc bfloat16, 2 byte'); } },
  { bytes: 1, get label() { return tr('int8 or fp8, 1 byte', 'int8 hoặc fp8, 1 byte'); } },
  { bytes: 0.5, get label() { return tr('4-bit, two per byte', '4 bit, hai giá trị mỗi byte'); } },
];

export interface ArrayLimit {
  /** Elements the index type can reach: indices 0 .. elements − 1. */
  elements: number;
  /** Bytes those elements occupy. */
  indexBytes: number;
  /** Bytes the address width can name. */
  addressBytes: number;
  /** The smaller of the two. */
  bytes: number;
  /** Which one binds. Equal limits count as the index. */
  binds: 'index' | 'address';
}

/**
 * The largest array a kernel can index with `index` over elements of `elemBytes`
 * bytes, given `addrBits`-bit addresses (64 by default). With an int index and
 * floats this is 2^31 × 4 bytes = 8 GiB.
 */
export function arrayLimit(index: IndexType, elemBytes: number, addrBits = 64): ArrayLimit {
  const elements = 2 ** INDEX[index].log2Count;
  const indexBytes = elements * elemBytes;
  const addressBytes = reach(addrBits);
  const binds = indexBytes <= addressBytes ? 'index' : 'address';
  return { elements, indexBytes, addressBytes, bytes: Math.min(indexBytes, addressBytes), binds };
}

/** How many times larger `bytes` is than `limit` (for "your device holds N times what an int can index"). */
export function times(bytes: number, limit: number): number {
  return limit > 0 ? bytes / limit : NaN;
}

// ---------------------------------------------------------------- the reader's devices

/** The minimal shape of reports/runs/*.json that this needs (see app/src/lib/reports.ts). */
export type ReportsLike = Record<string, { device: Record<string, string | boolean | number> }>;

export interface MemoryFigure {
  backend: string;
  /** The device name the backend reported. */
  name: string;
  /** The device property it came from. */
  field: 'totalGlobalMem' | 'maxBufferLength' | 'recommendedMaxWorkingSetSize';
  /** What it means, in a few words. */
  meaning: string;
  bytes: number;
}

const MEMORY_FIELDS: { field: MemoryFigure['field']; meaning: string }[] = [
  { field: 'totalGlobalMem', get meaning() { return tr('global memory on the device', 'bộ nhớ toàn cục trên thiết bị'); } },
  { field: 'maxBufferLength', get meaning() { return tr('the largest single buffer', 'buffer đơn lớn nhất'); } },
  { field: 'recommendedMaxWorkingSetSize', get meaning() { return tr('the recommended working set', 'tập làm việc (working set) được khuyến nghị'); } },
];

/** Emulator backends describe the CPU emulator, not a GPU. */
export const isEmulator = (backend: string) => backend === 'emu' || backend === 'race';

/** Memory figures the reader's real devices reported, largest first within a backend. */
export function memoryFigures(reports: ReportsLike): MemoryFigure[] {
  const out: MemoryFigure[] = [];
  for (const backend of Object.keys(reports).sort()) {
    if (isEmulator(backend)) continue;
    const d = reports[backend]?.device ?? {};
    for (const f of MEMORY_FIELDS) {
      const bytes = Number(d[f.field]);
      if (!isFinite(bytes) || bytes <= 0) continue;
      out.push({ backend, name: String(d.name ?? backend), field: f.field, meaning: f.meaning, bytes });
    }
  }
  return out;
}

// ---------------------------------------------------------------- does it all fit (§C.7)

/** CPU user-space widths the C.7 explorer offers (A7). */
export const CPU_WIDTHS = [
  { bits: 47, get label() { return tr('47 bits: Linux user space by default', '47 bit: không gian người dùng mặc định của Linux'); } },
  { bits: 48, get label() { return tr("48 bits: the book's figure, all of 4-level paging", '48 bit: con số của sách, toàn bộ phân trang 4 cấp'); } },
  { bits: 56, get label() { return tr('56 bits: 5-level paging, when a program asks', '56 bit: phân trang 5 cấp, khi một chương trình yêu cầu'); } },
] as const;

/** GPU virtual widths the C.7 explorer offers. */
export const GPU_WIDTHS = [32, 40, 47, 48, 49] as const;

export interface Coverage {
  /** Bytes the GPU's virtual width can name. */
  capacity: number;
  /** The CPU's user range: [0, cpu). */
  cpu: number;
  /** The GPU's memory, placed right after: [cpu, cpu + gpu). */
  gpu: number;
  needed: number;
  fits: boolean;
  /** capacity − needed when it fits, else 0. */
  spare: number;
  /** needed − capacity when it does not fit, else 0. */
  over: number;
  /** Part of the CPU range that falls outside the capacity. */
  cpuOutside: number;
  /** Part of the GPU range that falls outside the capacity. */
  gpuOutside: number;
}

/**
 * Packs a CPU user range of 2^cpuBits bytes and `gpuBytes` of GPU memory end to end
 * and checks the result against a GPU virtual address of `gpuBits` bits.
 */
export function coverage(gpuBits: number, cpuBits: number, gpuBytes: number): Coverage {
  const capacity = reach(gpuBits);
  const cpu = reach(cpuBits);
  const gpu = Math.max(0, gpuBytes);
  const needed = cpu + gpu;
  const fits = needed <= capacity;
  return {
    capacity,
    cpu,
    gpu,
    needed,
    fits,
    spare: fits ? capacity - needed : 0,
    over: fits ? 0 : needed - capacity,
    cpuOutside: Math.max(0, cpu - capacity),
    gpuOutside: Math.min(gpu, Math.max(0, needed - Math.max(capacity, cpu))),
  };
}

/**
 * The smallest address width, in bits, that holds 2^cpuBits bytes of CPU range plus
 * `gpuBytes` of GPU memory. For a 48-bit CPU space and any GPU memory from 1 byte up
 * to 2^48 bytes it is 49: the book's reasoning for Pascal.
 */
export function smallestWidth(cpuBits: number, gpuBytes: number): number {
  const needed = reach(cpuBits) + Math.max(0, gpuBytes);
  let w = 0;
  while (reach(w) < needed) w++;
  return w;
}
