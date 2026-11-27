// Bytes per element and bandwidth ceilings for scan (chapter 11, sections 11.8 and 11.9).
// One model for the "scan ceiling" explorer (11.8), the "bytes per element" explorer
// (11.9), the two measured reveals and the sections' questions.
//
// From the book (notes/ch11.md):
//   - §11.8, p. 272: scanning N floats loads 4N bytes, does N - 1 additions and stores 4N
//     bytes, so its intensity is (N - 1) / 8N, about 1/8 OP/B. On the book's example device
//     (H100: 66.9 TFLOPS, 3.35 TB/s, threshold 20 FLOP/B) the ceiling is 3.35e12 / 8 =
//     419e9 elements per second. More than 80% of bandwidth with no redundant global
//     access counts as well optimised (pp. 272-273).
//   - §11.9, pp. 273-276: the three-kernel scan-scan-add moves (4 + 2/S)·N elements,
//     (16 + 8/S)·N bytes for floats: kernel 1 loads N and stores N, kernel 2 loads and
//     stores N/S, kernel 3 loads N and stores N; ceiling 209e9. The three-kernel
//     reduce-scan-scan moves (3 + 3/S)·N elements, (12 + 12/S)·N bytes: kernel 1 loads N
//     and stores N/S, kernel 2 loads and stores N/S, kernel 3 loads N and stores N;
//     ceiling 279e9. A single kernel can reach the ideal 8 bytes (p. 276).
// From the facts (notes/today/ch11-facts.md A1): NVIDIA prints FP32 as 67 TFLOPS and
// 3.35 TB/s for H100 SXM; the threshold is 20 FLOP/B either way and the book's rounding
// of 419, 209 and 279 is right.
//
// Ours, and labelled so in the lessons:
//   - other element sizes (a half-precision scan) and an extra one-byte output per element
//     (the keep flag of lab 11.6's fused top-p row: 4 bytes in, 4 + 1 out = 9 bytes);
//   - "bookkeeping", the per-block accesses the book's totals leave out: scan-scan-add's
//     kernel 1 also stores one block sum per block and kernel 3 also loads one scanned sum
//     per block; reduce-scan-scan's kernel 3 also loads one scanned sum per block; the
//     single-pass kernel touches, per block, the counter (one atomic read-modify-write,
//     counted as a 4-byte load and a 4-byte store), one flag read (at least) and one flag
//     write (4 bytes each), and one partial sum stored and one loaded (4 bytes each);
//   - reading lab rows: elements per second from n / seconds, or from gb_per_s at the
//     scan's bytes per element; share of the copy rate = elements per second × bytes per
//     element / copy bytes per second.
//
// What this leaves out: caches (every access is charged to global memory), spin-loop
// re-reads of a flag, launch overhead, and anything that is not a byte of traffic.
//
// Erasable TypeScript only, and no value imports: the browser checks import this file
// directly in Node.

import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- the book's example device
/** The book's example device (p. 272). Used only as the book's example. */
export const BOOK_DEVICE = {
  get name() {
    return tr("The book's example (H100 SXM)", 'Ví dụ của sách (H100 SXM)');
  },
  /** Peak global-memory bandwidth, GB/s (3.35 TB/s). */
  bandwidthGBs: 3350,
  /** FP32 peak as the book prints it, TFLOPS. */
  tflopsBook: 66.9,
  /** FP32 peak as NVIDIA prints it for H100 SXM, TFLOPS (facts A1, ch05 A8). */
  tflopsNvidia: 67,
} as const;

/** The book's yardstick (p. 273): above this share of bandwidth, with no redundant accesses, a kernel is well optimised. */
export const WELL_OPTIMISED = 0.8;

/** Compute-bound threshold in FLOP/B: peak operations per second over peak bytes per second. */
export function threshold(tflops: number, bandwidthGBs: number): number {
  return (tflops * 1000) / bandwidthGBs;
}

// ---------------------------------------------------------------- one scan: intensity and ceiling
export type Elem = 'float' | 'half';
export const ELEM_BYTES: Record<Elem, number> = { float: 4, half: 2 };
export const ELEM_LABEL: Record<Elem, string> = {
  get float() {
    return tr('float (4 bytes)', 'float (4 byte)');
  },
  get half() {
    return tr('half (2 bytes)', 'half (2 byte)');
  },
};

/** What the scan writes: its result only, or its result plus a one-byte flag per element (lab 11.6's keep mask). */
export type Written = 'scan' | 'scan+mask';
export const WRITTEN_LABEL: Record<Written, string> = {
  get scan() {
    return tr('the scan only', 'chỉ kết quả scan');
  },
  get 'scan+mask'() {
    return tr('the scan and a one-byte flag', 'kết quả scan và một cờ một byte');
  },
};

export interface ScanBytes {
  /** Bytes loaded per element. */
  load: number;
  /** Bytes stored per element. */
  store: number;
  /** load + store. */
  total: number;
}

/**
 * Bytes per element of a single-pass scan: one element in, one result out (in the same
 * type; the accumulator's width does not move bytes), plus one byte if a flag is written.
 */
export function scanBytes(elem: Elem, written: Written = 'scan'): ScanBytes {
  const e = ELEM_BYTES[elem];
  const store = e + (written === 'scan+mask' ? 1 : 0);
  return { load: e, store, total: e + store };
}

/** Additions per byte for a scan of n elements: n - 1 additions over n·bytesPerElement bytes. */
export function intensity(n: number, bytesPerElement: number): number {
  return (n - 1) / (n * bytesPerElement);
}
/** The same for large n: 1 / bytes per element. */
export const intensityLimit = (bytesPerElement: number) => 1 / bytesPerElement;

/** Elements per second a kernel moving `bytesPerElement` could reach at `bandwidthGBs`. */
export function ceiling(bandwidthGBs: number, bytesPerElement: number): number {
  return (bandwidthGBs * 1e9) / bytesPerElement;
}

// ---------------------------------------------------------------- global scan designs (§11.9)
export type Design = 'single' | 'ssa' | 'rss';
export const DESIGNS: readonly Design[] = ['single', 'ssa', 'rss'];
export const DESIGN_LABEL: Record<Design, string> = {
  get single() {
    return tr('single pass (one kernel)', 'một lượt (một kernel)');
  },
  get ssa() {
    return tr('scan-scan-add (three kernels)', 'scan-scan-add (ba kernel)');
  },
  get rss() {
    return tr('reduce-scan-scan (three kernels)', 'reduce-scan-scan (ba kernel)');
  },
};

export interface Access {
  /** What is touched, e.g. "input", "block sums". */
  what: string;
  kind: 'load' | 'store';
  /** Elements per input element (so N/S is 1/S). */
  perElement: number;
  /** Bytes of each access. */
  bytes: number;
  /** True for the per-block accesses the book's totals leave out (ours). */
  bookkeeping?: boolean;
}
export interface KernelTraffic {
  kernel: string;
  does: string;
  accesses: Access[];
}
export interface DesignTraffic {
  design: Design;
  segment: number;
  elemBytes: number;
  kernels: KernelTraffic[];
  /** Accesses per input element, counting only the accesses included. */
  elementsPerElement: number;
  /** Bytes per input element, counting only the accesses included. */
  bytesPerElement: number;
  /** The book's closed form for these sizes, e.g. "(16 + 8/S)·N bytes" with S filled in. */
  formula: string;
}

/**
 * Every kernel's loads and stores for one design, segment size S (elements per block) and
 * element size. With `bookkeeping` the per-block accesses the book leaves out are added.
 */
export function designTraffic(design: Design, S: number, elemBytes = 4, bookkeeping = false): DesignTraffic {
  const s = 1 / S;
  const e = elemBytes;
  let kernels: KernelTraffic[];
  if (design === 'ssa') {
    kernels = [
      {
        kernel: 'kernel 1', // i18n-ignore: identifier also shown to the reader
        does: tr('scan each block segment', 'scan từng đoạn của khối'),
        accesses: [
          { what: tr('input', 'đầu vào'), kind: 'load', perElement: 1, bytes: e },
          { what: tr('scanned segments', 'các đoạn đã scan'), kind: 'store', perElement: 1, bytes: e },
          { what: tr('block sums', 'tổng của các khối'), kind: 'store', perElement: s, bytes: e, bookkeeping: true },
        ],
      },
      {
        kernel: 'kernel 2', // i18n-ignore: identifier also shown to the reader
        does: tr('scan the block sums (one block)', 'scan các tổng của khối (một khối)'),
        accesses: [
          { what: tr('block sums', 'tổng của các khối'), kind: 'load', perElement: s, bytes: e },
          { what: tr('scanned block sums', 'tổng của các khối đã scan'), kind: 'store', perElement: s, bytes: e },
        ],
      },
      {
        kernel: 'kernel 3', // i18n-ignore: identifier also shown to the reader
        does: tr('add the earlier blocks\' total', 'cộng tổng của các khối trước'),
        accesses: [
          { what: tr('scanned segments', 'các đoạn đã scan'), kind: 'load', perElement: 1, bytes: e },
          { what: tr('scanned block sums', 'tổng của các khối đã scan'), kind: 'load', perElement: s, bytes: e, bookkeeping: true },
          { what: tr('output', 'đầu ra'), kind: 'store', perElement: 1, bytes: e },
        ],
      },
    ];
  } else if (design === 'rss') {
    kernels = [
      {
        kernel: 'kernel 1', // i18n-ignore: identifier also shown to the reader
        does: tr('reduce each block segment', 'rút gọn từng đoạn của khối'),
        accesses: [
          { what: tr('input', 'đầu vào'), kind: 'load', perElement: 1, bytes: e },
          { what: tr('block sums', 'tổng của các khối'), kind: 'store', perElement: s, bytes: e },
        ],
      },
      {
        kernel: 'kernel 2', // i18n-ignore: identifier also shown to the reader
        does: tr('scan the block sums (one block)', 'scan các tổng của khối (một khối)'),
        accesses: [
          { what: tr('block sums', 'tổng của các khối'), kind: 'load', perElement: s, bytes: e },
          { what: tr('scanned block sums', 'tổng của các khối đã scan'), kind: 'store', perElement: s, bytes: e },
        ],
      },
      {
        kernel: 'kernel 3', // i18n-ignore: identifier also shown to the reader
        does: tr('add the earlier blocks\' total to the first element, scan the segment', 'cộng tổng của các khối trước vào phần tử đầu tiên, rồi scan đoạn đó'),
        accesses: [
          { what: tr('input', 'đầu vào'), kind: 'load', perElement: 1, bytes: e },
          { what: tr('scanned block sums', 'tổng của các khối đã scan'), kind: 'load', perElement: s, bytes: e, bookkeeping: true },
          { what: tr('output', 'đầu ra'), kind: 'store', perElement: 1, bytes: e },
        ],
      },
    ];
  } else {
    kernels = [
      {
        kernel: 'one kernel', // i18n-ignore: identifier also shown to the reader
        does: tr('scan, look back, add, store', 'scan, nhìn lại (look back), cộng, ghi'),
        accesses: [
          { what: tr('input', 'đầu vào'), kind: 'load', perElement: 1, bytes: e },
          { what: tr('output', 'đầu ra'), kind: 'store', perElement: 1, bytes: e },
          { what: tr('block counter (atomic read)', 'bộ đếm khối (đọc nguyên tử)'), kind: 'load', perElement: s, bytes: 4, bookkeeping: true },
          { what: tr('block counter (atomic write)', 'bộ đếm khối (ghi nguyên tử)'), kind: 'store', perElement: s, bytes: 4, bookkeeping: true },
          { what: tr('previous block\'s flag (at least one read)', 'cờ của khối trước (ít nhất một lần đọc)'), kind: 'load', perElement: s, bytes: 4, bookkeeping: true },
          { what: tr('previous block\'s partial sum', 'tổng riêng phần của khối trước'), kind: 'load', perElement: s, bytes: 4, bookkeeping: true },
          { what: tr('own partial sum', 'tổng riêng phần của chính khối'), kind: 'store', perElement: s, bytes: 4, bookkeeping: true },
          { what: tr('own flag', 'cờ của chính khối'), kind: 'store', perElement: s, bytes: 4, bookkeeping: true },
        ],
      },
    ];
  }
  const counted = kernels.flatMap((k) => k.accesses).filter((a) => bookkeeping || !a.bookkeeping);
  const elementsPerElement = counted.reduce((t, a) => t + a.perElement, 0);
  const bytesPerElement = counted.reduce((t, a) => t + a.perElement * a.bytes, 0);
  const formula =
    design === 'ssa'
      ? tr(`(${4 * e} + ${2 * e}/S)·N bytes`, `(${4 * e} + ${2 * e}/S)·N byte`)
      : design === 'rss'
        ? tr(`(${3 * e} + ${3 * e}/S)·N bytes`, `(${3 * e} + ${3 * e}/S)·N byte`)
        : tr(`${2 * e}·N bytes`, `${2 * e}·N byte`);
  return { design, segment: S, elemBytes: e, kernels, elementsPerElement, bytesPerElement, formula };
}

/** Accesses counted in a design's total (all of them with bookkeeping, else the book's). */
export const countedAccesses = (t: DesignTraffic, bookkeeping: boolean) =>
  t.kernels.flatMap((k) => k.accesses.map((a) => ({ ...a, kernel: k.kernel }))).filter((a) => bookkeeping || !a.bookkeeping);

/** Ceilings of the three designs side by side, elements per second. */
export function designCeilings(bandwidthGBs: number, S: number, elemBytes = 4, bookkeeping = false): Record<Design, number> {
  const out = {} as Record<Design, number>;
  for (const d of DESIGNS) out[d] = ceiling(bandwidthGBs, designTraffic(d, S, elemBytes, bookkeeping).bytesPerElement);
  return out;
}

/** The book's preset: its device, floats, a large segment (any S of a few thousand rounds the same). */
export const BOOK_PRESET = { bandwidthGBs: BOOK_DEVICE.bandwidthGBs, S: 2048, elem: 'float' as Elem, bookkeeping: false };

// ---------------------------------------------------------------- the fused top-p row (lab 11.6)
/** Lab 11.6 reads a float probability and writes a float exclusive mass and a one-byte keep flag. */
export const TOPP_ROW = { load: 4, store: 4 + 1 } as const;
export const toppBytesPerElement = () => TOPP_ROW.load + TOPP_ROW.store;

// ---------------------------------------------------------------- reading lab rows
/** A bench row as written by the labs (reports/runs/*.json). */
export type Row = Record<string, string | number | boolean>;
const numOf = (v: unknown): number | undefined => (typeof v === 'number' && isFinite(v) ? v : undefined);
const textOf = (r: Row) => [r.name, r.kernel, r.variant, r.design].filter((x) => typeof x === 'string').join(' ');

/** Elements a row processed: `n`, else `elements`. */
export const rowElements = (r: Row) => numOf(r.n) ?? numOf(r.elements);
/** A copy kernel's row: named "copy", or (the chapter 11 labs' convention) with kernel "copy". */
export const isCopy = (r: Row) => r.name === 'copy' || r.kernel === 'copy';
/** The fastest copy row's bandwidth in GB/s, if the rows have one. */
export function copyGBs(rows: Row[]): number | undefined {
  const g = rows.filter(isCopy).map((r) => numOf(r.gb_per_s)).filter((x): x is number => x != null);
  return g.length ? Math.max(...g) : undefined;
}
/** Elements per second of a row: elements_per_s, else n / seconds, else gb_per_s at bytesPerElement. */
export function rowRate(r: Row, bytesPerElement = 8): number | undefined {
  const direct = numOf(r.elements_per_s);
  if (direct) return direct;
  const n = rowElements(r);
  const s = numOf(r.seconds);
  if (n && s) return n / s;
  const g = numOf(r.gb_per_s);
  return g ? (g * 1e9) / bytesPerElement : undefined;
}
/** Share of a copy rate: elements/s × bytes per element over copy bytes/s. */
export function shareOfCopy(elementsPerSec: number, copyGBsValue: number, bytesPerElement = 8): number {
  return (elementsPerSec * bytesPerElement) / (copyGBsValue * 1e9);
}

export interface KernelRate {
  label: string;
  factor: number | undefined;
  n: number | undefined;
  elementsPerSec: number;
  share: number;
  /** True for a register-tiled kernel (its name, kernel or variant mentions "reg" or "tile"). */
  registerTiled: boolean;
}
/** Thread counts per block that a lab's rows ran at, in increasing order. */
export const blockSizes = (rows: Row[]) =>
  [...new Set(rows.map((r) => numOf(r.block)).filter((b): b is number => b != null))].sort((a, b) => a - b);

/**
 * Every non-copy row of a scan lab, as a rate and a share of the copy rate. A row's own
 * `share_of_copy` (the lab's copy at the same launch shape) is used when it has one;
 * otherwise the share is taken against `copy`.
 */
export function kernelRates(rows: Row[], copy: number, bytesPerElement = 8): KernelRate[] {
  return rows.flatMap((r) => {
    if (isCopy(r)) return [];
    const rate = rowRate(r, bytesPerElement);
    if (!rate) return [];
    const factor = numOf(r.coarse) ?? numOf(r.factor) ?? numOf(r.coarse_factor);
    const kernel = typeof r.kernel === 'string' ? r.kernel : String(r.name);
    const block = numOf(r.block);
    const own = numOf(r.share_of_copy);
    return [
      {
        label: [kernel, factor != null ? `factor ${factor}` : '', block != null ? `${block} threads` : ''].filter(Boolean).join(', '),
        factor,
        n: rowElements(r),
        elementsPerSec: rate,
        share: own && own > 0 ? own : shareOfCopy(rate, copy, bytesPerElement),
        registerTiled: /reg|tile/i.test(textOf(r)),
      },
    ];
  });
}
/** The best register-tiled rate, or the best of all if none is marked register-tiled. */
export function bestRegisterTiled(rates: KernelRate[]): { rate: KernelRate; marked: boolean } | undefined {
  const marked = rates.filter((r) => r.registerTiled);
  const pool = marked.length ? marked : rates;
  if (!pool.length) return undefined;
  return { rate: pool.reduce((a, b) => (b.share > a.share ? b : a)), marked: marked.length > 0 };
}

// ---------------------------------------------------------------- this machine's bandwidth
/** The part of a backend's report (reports/runs/<backend>.json) these functions read. */
export interface RunLike {
  generated: string;
  device: Record<string, string | boolean>;
  results: { lab: string; variant: string; bench: Row[] }[];
}
/** Bench rows of the reference solution of every lab whose id starts with `prefix` (e.g. "ch11/lab2_"). */
export function labRows(run: RunLike | undefined, prefix: string): Row[] {
  if (!run) return [];
  return run.results.filter((x) => x.lab.startsWith(prefix) && x.variant === 'solution').flatMap((x) => x.bench);
}
export interface Measured {
  /** GB/s a copy achieved. */
  gbs: number;
  /** Which lab measured it. */
  source: string;
  /** FLOP/B from lab 5.1's own pair of probes (compute over bandwidth), when it ran. */
  threshold?: number;
}
/**
 * The copy bandwidth this backend measured: lab 11.2's copy kernel, else lab 11.1's, else
 * lab 5.1's bandwidth probe. The threshold uses lab 5.1's compute probe over that bandwidth.
 */
export function measuredBandwidth(run: RunLike | undefined): Measured | undefined {
  if (!run) return undefined;
  const roof = labRows(run, 'ch05/lab1_roofline').find((r) => r.name === 'roofline');
  const found = (
    [
      [copyGBs(labRows(run, 'ch11/lab2_')), 'the copy kernel of lab 11.2'],
      [copyGBs(labRows(run, 'ch11/lab1_')), 'the copy kernel of lab 11.1'],
      [roof ? numOf(roof.bandwidth_gb_per_s) : undefined, 'the bandwidth probe of lab 5.1'],
    ] as [number | undefined, string][]
  ).find(([v]) => v != null);
  if (!found) return undefined;
  const compute = roof ? numOf(roof.compute_gflops) : undefined;
  const roofBw = roof ? numOf(roof.bandwidth_gb_per_s) : undefined;
  return { gbs: found[0]!, source: found[1], threshold: compute && roofBw ? compute / roofBw : undefined };
}

/** Which three-kernel design a lab 11.4 row times, from its name, kernel, variant or design field. */
export function rowDesign(r: Row): 'ssa' | 'rss' | undefined {
  const t = textOf(r);
  if (/reduce|rss/i.test(t)) return 'rss';
  if (/scan[-_ ]?scan[-_ ]?add|ssa|add/i.test(t)) return 'ssa';
  return undefined;
}
/**
 * Reduce-scan-scan's speed over scan-scan-add from lab rows at size n (2^26 by default, or the
 * largest size both designs ran at if none matches n): the ratio of their element rates.
 */
export function rssOverSsa(rows: Row[], n = 2 ** 26): { n: number; ssa: number; rss: number; ratio: number } | undefined {
  const timed = rows.flatMap((r) => {
    const d = rowDesign(r);
    const rate = rowRate(r);
    const size = rowElements(r);
    return d && rate && size ? [{ d, rate, size }] : [];
  });
  const sizes = [...new Set(timed.map((t) => t.size))].filter(
    (size) => timed.some((t) => t.size === size && t.d === 'ssa') && timed.some((t) => t.size === size && t.d === 'rss'),
  );
  if (!sizes.length) return undefined;
  const at = sizes.includes(n) ? n : Math.max(...sizes);
  const best = (d: 'ssa' | 'rss') => Math.max(...timed.filter((t) => t.size === at && t.d === d).map((t) => t.rate));
  const ssa = best('ssa');
  const rss = best('rss');
  return { n: at, ssa, rss, ratio: rss / ssa };
}
