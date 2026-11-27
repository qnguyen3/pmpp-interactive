// Long sums in a chosen accumulator format: the model behind appendix A's start-page teaser
// and section A.6 (the two steppers, the summation explorer, the Kahan bug hunt and the
// "Kahan's correction" scene).
//
// Sources:
//  - the book, §A.6, pp. 593–594 (notes/appA.md): the 5-bit example 1 + 1 + 0.25 + 0.25 gives
//    2.0 in order and 2.5 as a tree; a nearby case can make the tree worse; sort ascending
//    before reducing (signs taken into account); Kahan's compensated sum is named, not
//    described. Exercise 4, p. 600, with chapter 10's two tree kernels (Fig. 10.6, adjacent
//    pairs, and Fig. 10.8, thread i adds element i + stride).
//  - the formats: app/src/sim/minifloat.ts (FP32, FP16, BF16 and the book's 5-bit TOY5).
//  - chunked order: chapter 10's coarsened kernel (§10.10, Fig. 10.19): each thread adds
//    elements one block-width apart, then the per-thread sums go through a tree.
//  - Kahan's algorithm and the Neumaier reference are standard numerical analysis, ours to
//    describe (the book only cites Kahan, ref. [5]).
//
// What it models, exactly:
//  - Every addition (and every subtraction inside Kahan) is rounded once to the accumulator
//    format, to nearest with ties to even, with IEEE overflow to infinity. Inputs are rounded
//    to the accumulator format before they are added (half acc; acc += half(x)).
//  - FP32, FP16 and BF16 use fast rounders (Math.fround, and a grid rounder for the other two)
//    on the binary64 sum of two values of the format. That is correctly rounded because binary64
//    has at least 2p + 2 significant bits for p = 24, 11 and 8 (double rounding is then
//    innocuous); fpsum.test.ts checks every rounder against minifloat.round and minifloat.add.
//    Any other format goes through minifloat.add directly.
//  - Tree shapes are chapter 10's: "halves" (element i with element i + m/2, the convergent
//    kernel's order) and "neighbours" (adjacent pairs). Lengths that are not a power of two
//    are padded with zeros, which adds nothing.
//
// What it leaves out: fused multiply-add, flush-to-zero, any compiler reassociation (except
// the one bug-hunt variant that spells it out), round-toward-zero hardware (Metal allows it,
// appA-facts C2), warp-shuffle trees and other tree shapes, and anything about time.
//
// Erasable TypeScript only (the browser checks import this module through the app's server).

import { BF16, FP16, FP32, TOY5, add as mfAdd, maxFinite, round as mfRound, ulp as mfUlp, values as mfValues, type Format } from './minifloat';
import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- formats and rounding
export type Acc = 'fp32' | 'fp16' | 'bf16';
export const ACC_FORMAT: Record<Acc, Format> = { fp32: FP32, fp16: FP16, bf16: BF16 };
export const ACC_LABEL: Record<Acc, string> = { fp32: 'FP32', fp16: 'FP16', bf16: 'BF16' };
export const fmtOf = (f: Acc | Format): Format => (typeof f === 'string' ? ACC_FORMAT[f] : f);

const f64 = new Float64Array(1);
const u32 = new Uint32Array(f64.buffer);
/** floor(log2 a) for a positive, finite binary64 value (little-endian hosts; subnormal doubles count as -1023). */
function exponentOf(a: number): number {
  f64[0] = a;
  return ((u32[1] >>> 20) & 0x7ff) - 1023;
}
const POW2 = new Float64Array(2400);
for (let k = 0; k < POW2.length; k++) POW2[k] = 2 ** (k - 1200);
const MAGIC = 6755399441055744; // 1.5 * 2^52: adding and subtracting C * q rounds to a multiple of q, ties to even

/**
 * Round to nearest even onto a binary grid with m stored mantissa bits, lowest normal exponent
 * emin (subnormals below it) and largest finite value max (beyond: infinity).
 */
function gridRounder(m: number, emin: number, max: number): (x: number) => number {
  return (x: number) => {
    if (x === 0 || x !== x || x === Infinity || x === -Infinity) return x;
    const a = x < 0 ? -x : x;
    let e = exponentOf(a);
    if (e < emin) e = emin;
    const C = MAGIC * POW2[e - m + 1200];
    const r = a + C - C;
    if (r > max) return x < 0 ? -Infinity : Infinity;
    return x < 0 ? -r : r;
  };
}

const ROUNDERS = new Map<Format, (x: number) => number>([
  [FP32, Math.fround],
  [FP16, gridRounder(10, -14, 65504)],
  [BF16, gridRounder(7, -126, maxFinite(BF16))],
]);

/** The function that stores a value in format f (round to nearest even, IEEE overflow). */
export function rounder(f: Acc | Format): (x: number) => number {
  const F = fmtOf(f);
  return ROUNDERS.get(F) ?? ((x: number) => mfRound(F, x));
}

/**
 * a + b in format f: both operands stored in f, then one rounding of their binary64 sum. For
 * formats of at most 25 significant bits that is the correctly rounded sum (binary64 has 2p + 2
 * bits to spare, so rounding twice is innocuous); fpsum.test.ts checks it against
 * minifloat.add, which rounds the exact sum.
 */
export function adder(f: Acc | Format): (a: number, b: number) => number {
  const r = rounder(f);
  return (a, b) => r(r(a) + r(b));
}

/** The values as stored in format f. */
export function stored(values: ArrayLike<number>, f: Acc | Format): Float64Array {
  const r = rounder(f);
  const v = new Float64Array(values.length);
  for (let i = 0; i < v.length; i++) v[i] = r(values[i]);
  return v;
}

/** The exact rounding error of a + b in binary64 (Knuth's TwoSum): a + b = s + err exactly. */
function twoSumErr(a: number, b: number, s: number): number {
  const bb = s - a;
  return a - (s - bb) + (b - bb);
}

// ---------------------------------------------------------------- data
/** mulberry32: a small seedable generator. Same seed, same numbers, everywhere. */
export function rng(seed: number): () => number {
  let h = seed >>> 0;
  return () => {
    h = (h + 0x6d2b79f5) >>> 0;
    let t = Math.imul(h ^ (h >>> 15), h | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type DataKind = 'ones' | 'uniform' | 'mixed' | 'mixed-asc' | 'mixed-desc' | 'cancelling' | 'softmax' | 'big-small';
export const DATA: { id: DataKind; label: string; note: string }[] = [
  {
    id: 'ones',
    get label() {
      return tr('ones', 'toàn số 1');
    },
    get note() {
      return tr('every value is 1', 'mọi giá trị đều là 1');
    },
  },
  {
    id: 'uniform',
    get label() {
      return tr('uniform on [0, 1)', 'đều trên [0, 1)');
    },
    get note() {
      return tr('random values between 0 and 1', 'các giá trị ngẫu nhiên giữa 0 và 1');
    },
  },
  {
    id: 'mixed',
    get label() {
      return tr('mixed magnitudes', 'độ lớn trộn lẫn');
    },
    get note() {
      return tr('log-uniform between 2^-14 and 2^-2, random order', 'log-uniform giữa 2^-14 và 2^-2, thứ tự ngẫu nhiên');
    },
  },
  {
    id: 'mixed-asc',
    get label() {
      return tr('mixed, ascending', 'trộn lẫn, tăng dần');
    },
    get note() {
      return tr('the same values sorted smallest first', 'cùng các giá trị đó, sắp xếp từ nhỏ nhất trước');
    },
  },
  {
    id: 'mixed-desc',
    get label() {
      return tr('mixed, descending', 'trộn lẫn, giảm dần');
    },
    get note() {
      return tr('the same values sorted largest first', 'cùng các giá trị đó, sắp xếp từ lớn nhất trước');
    },
  },
  {
    id: 'cancelling',
    get label() {
      return tr('cancelling', 'triệt tiêu');
    },
    get note() {
      return tr('mixed magnitudes with random signs', 'độ lớn trộn lẫn với dấu ngẫu nhiên');
    },
  },
  {
    id: 'softmax',
    get label() {
      return tr('softmax terms', 'các số hạng softmax');
    },
    get note() {
      return tr('exp(z − max z) for standard-normal logits z', 'exp(z − max z) với logits z phân phối chuẩn tắc');
    },
  },
  {
    id: 'big-small',
    get label() {
      return tr('one big, many small', 'một lớn, nhiều nhỏ');
    },
    get note() {
      return tr('1.0 first, then values below 2^-24, half an FP32 ULP of 1', '1.0 trước, rồi các giá trị dưới 2^-24, tức nửa ULP FP32 của 1');
    },
  },
];
export const DEFAULT_SEED = 7;

function normal(r: () => number): number {
  // Box-Muller; 1 - r() is in (0, 1], so the log is finite.
  return Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
}

/** n values of a kind, stored in `storage` (FP32 unless said otherwise). */
export function generate(kind: DataKind, n: number, seed = DEFAULT_SEED, storage: Acc | Format = FP32): Float64Array {
  const r = rng(seed);
  const store = rounder(storage);
  const x = new Float64Array(n);
  const mixed = () => 2 ** (-14 + 12 * r());
  switch (kind) {
    case 'ones':
      x.fill(1);
      break;
    case 'uniform':
      for (let i = 0; i < n; i++) x[i] = r();
      break;
    case 'mixed':
    case 'mixed-asc':
    case 'mixed-desc':
      for (let i = 0; i < n; i++) x[i] = mixed();
      break;
    case 'cancelling':
      for (let i = 0; i < n; i++) x[i] = (r() < 0.5 ? -1 : 1) * mixed();
      break;
    case 'softmax': {
      let m = -Infinity;
      for (let i = 0; i < n; i++) {
        x[i] = normal(r);
        if (x[i] > m) m = x[i];
      }
      for (let i = 0; i < n; i++) x[i] = Math.exp(x[i] - m);
      break;
    }
    case 'big-small':
      x[0] = 1;
      for (let i = 1; i < n; i++) x[i] = r() * 2 ** -24;
      break;
  }
  for (let i = 0; i < n; i++) x[i] = store(x[i]);
  if (kind === 'mixed-asc') x.sort();
  if (kind === 'mixed-desc') x.sort().reverse();
  return x;
}

// ---------------------------------------------------------------- reference and errors
/** Neumaier's compensated sum in binary64: the reference every result is compared with. */
export function reference(values: ArrayLike<number>): number {
  let s = 0;
  let c = 0;
  for (let i = 0; i < values.length; i++) {
    const x = values[i];
    const t = s + x;
    c += Math.abs(s) >= Math.abs(x) ? s - t + x : x - t + s;
    s = t;
  }
  return s + c;
}
/** The sum of absolute values: the scale every relative error is measured against. */
export function absSum(values: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < values.length; i++) s += Math.abs(values[i]);
  return s;
}

export interface Errors {
  result: number;
  reference: number;
  /** |result - reference| (Infinity when the result overflowed). */
  abs: number;
  /** abs / sum of |x|, which is abs / |reference| for data of one sign. */
  rel: number;
  /** abs in units in the last place of the accumulator format at the reference. */
  ulps: number;
  overflow: boolean;
}
export function errorsOf(result: number, ref: number, scale: number, f: Acc | Format): Errors {
  const overflow = !Number.isFinite(result);
  const abs = overflow ? Infinity : Math.abs(result - ref);
  return {
    result,
    reference: ref,
    abs,
    rel: overflow ? Infinity : scale > 0 ? abs / scale : abs,
    ulps: overflow ? Infinity : abs / mfUlp(fmtOf(f), ref),
    overflow,
  };
}

// ---------------------------------------------------------------- methods
export type Method = 'sequential' | 'sorted' | 'tree' | 'tree-neighbours' | 'chunked' | 'kahan' | 'kahan-chunked';
export const METHODS: { id: Method; label: string }[] = [
  {
    id: 'sequential',
    get label() {
      return tr('sequential loop', 'vòng lặp tuần tự');
    },
  },
  {
    id: 'sorted',
    get label() {
      return tr('sorted by |x|, then the loop', 'sắp xếp theo |x|, rồi vòng lặp');
    },
  },
  {
    id: 'tree',
    get label() {
      return tr('pairwise tree', 'cây từng cặp');
    },
  },
  {
    id: 'chunked',
    get label() {
      return tr('chunked: c per thread, then a tree', 'chia chunk: c phần tử mỗi luồng, rồi một cây');
    },
  },
  {
    id: 'kahan',
    get label() {
      return tr('Kahan loop', 'vòng lặp Kahan');
    },
  },
  {
    id: 'kahan-chunked',
    get label() {
      return tr('Kahan per thread, then a tree', 'Kahan cho từng luồng, rồi một cây');
    },
  },
];

export function sumSequential(values: ArrayLike<number>, f: Acc | Format): number {
  const r = rounder(f);
  let s = 0;
  for (let i = 0; i < values.length; i++) s = r(s + r(values[i]));
  return s;
}

/** Sorted by magnitude, smallest first (the book's pre-sorting with signs taken into account), then the loop. */
export function sortByMagnitude(values: ArrayLike<number>): Float64Array {
  // Two typed-array sorts and a merge: much faster than one sort with a comparator.
  const neg: number[] = [];
  const pos: number[] = [];
  for (let i = 0; i < values.length; i++) (values[i] < 0 ? neg : pos).push(values[i]);
  const P = Float64Array.from(pos).sort();
  const N = Float64Array.from(neg).sort().reverse(); // -0.1, -0.5, -2: smallest magnitude first
  const out = new Float64Array(values.length);
  let i = 0;
  let j = 0;
  for (let k = 0; k < out.length; k++) out[k] = j >= N.length || (i < P.length && P[i] <= -N[j]) ? P[i++] : N[j++];
  return out;
}
export function sumSorted(values: ArrayLike<number>, f: Acc | Format): number {
  return sumSequential(sortByMagnitude(values), f);
}

export type Shape = 'halves' | 'neighbours';
/** A reduction tree in format f (each value stored in f, each pair added in f). Pads to a power of two with zeros. */
export function sumTree(values: ArrayLike<number>, f: Acc | Format, shape: Shape = 'halves'): number {
  const n = values.length;
  if (n === 0) return 0;
  let m = 1;
  while (m < n) m *= 2;
  const r = rounder(f);
  const a = new Float64Array(m);
  for (let i = 0; i < n; i++) a[i] = r(values[i]);
  if (shape === 'halves') {
    for (let stride = m / 2; stride >= 1; stride /= 2) for (let i = 0; i < stride; i++) a[i] = r(a[i] + a[i + stride]);
  } else {
    for (let stride = 1; stride < m; stride *= 2) for (let i = 0; i + stride < m; i += 2 * stride) a[i] = r(a[i] + a[i + stride]);
  }
  return a[0];
}

/** Kahan's compensated loop in format f. */
export function sumKahan(values: ArrayLike<number>, f: Acc | Format): number {
  return kahanVariant('correct', values, f);
}

/**
 * Chunked: T = ceil(n / c) threads; thread t adds elements t, t + T, t + 2T, ... (c of them,
 * one block-width apart, as in chapter 10's coarsened kernel), plainly or with Kahan's
 * compensation, then the T partial sums go through the "halves" tree.
 * c = 1 is the tree; c >= n is the sequential loop (or the Kahan loop).
 */
export function sumChunked(values: ArrayLike<number>, f: Acc | Format, c: number, kahan = false): number {
  return chunkedStored(stored(values, f), f, c, kahan);
}
function chunkedStored(v: Float64Array, f: Acc | Format, c: number, kahan: boolean): number {
  const n = v.length;
  const per = Math.max(1, Math.min(Math.floor(c), n));
  const T = Math.ceil(n / per);
  const r = rounder(f);
  const partial = new Float64Array(T);
  for (let t = 0; t < T; t++) {
    let s = 0;
    if (kahan) {
      let comp = 0;
      for (let i = t; i < n; i += T) {
        const y = r(v[i] - comp);
        const tt = r(s + y);
        comp = r(r(tt - s) - y);
        s = tt;
      }
    } else for (let i = t; i < n; i += T) s = r(s + v[i]);
    partial[t] = s;
  }
  return sumTree(partial, f, 'halves');
}

export interface SumOptions {
  method: Method;
  acc: Acc | Format;
  /** Values per thread for the chunked methods. */
  chunk?: number;
}
export function sum(values: ArrayLike<number>, o: SumOptions): number {
  switch (o.method) {
    case 'sequential':
      return sumSequential(values, o.acc);
    case 'sorted':
      return sumSorted(values, o.acc);
    case 'tree':
      return sumTree(values, o.acc, 'halves');
    case 'tree-neighbours':
      return sumTree(values, o.acc, 'neighbours');
    case 'chunked':
      return sumChunked(values, o.acc, o.chunk ?? 64, false);
    case 'kahan':
      return sumKahan(values, o.acc);
    case 'kahan-chunked':
      return sumChunked(values, o.acc, o.chunk ?? 64, true);
  }
}

/** A sum with its errors against the reference. */
export function measure(values: ArrayLike<number>, o: SumOptions, ref = reference(values), scale = absSum(values)): Errors {
  return errorsOf(sum(values, o), ref, scale, o.acc);
}

// ---------------------------------------------------------------- traces for the plots
export interface RunningPoint {
  /** How many values have been added. */
  i: number;
  /** The sequential loop's accumulator after i additions. */
  acc: number;
  /** The exact sum of the first i values (Neumaier in binary64). */
  exact: number;
}
/** The sequential loop's running sum against the exact running sum, sampled at about `points` places. */
export function runningTrace(values: ArrayLike<number>, f: Acc | Format, points = 200): RunningPoint[] {
  const n = values.length;
  const add = adder(f);
  const every = Math.max(1, Math.floor(n / points));
  const out: RunningPoint[] = [{ i: 0, acc: 0, exact: 0 }];
  let s = 0;
  let es = 0;
  let ec = 0;
  for (let i = 0; i < n; i++) {
    const x = values[i];
    s = add(s, x);
    const t = es + x;
    ec += Math.abs(es) >= Math.abs(x) ? es - t + x : x - t + es;
    es = t;
    if ((i + 1) % every === 0 || i === n - 1) out.push({ i: i + 1, acc: s, exact: es + ec });
  }
  return out;
}

export interface ChunkPoint {
  c: number;
  plain: Errors;
  kahan: Errors;
}
/** Chunked plain and chunked Kahan at c = 1, 2, 4, ... up to min(maxC, n). */
export function chunkSweep(values: ArrayLike<number>, f: Acc | Format, maxC = 4096): ChunkPoint[] {
  const ref = reference(values);
  const scale = absSum(values);
  const v = stored(values, f);
  const out: ChunkPoint[] = [];
  for (let c = 1; c <= Math.min(maxC, values.length); c *= 2)
    out.push({
      c,
      plain: errorsOf(chunkedStored(v, f, c, false), ref, scale, f),
      kahan: errorsOf(chunkedStored(v, f, c, true), ref, scale, f),
    });
  return out;
}

// ---------------------------------------------------------------- the four-value stepper (p. 593)
export const TOY = TOY5;
/** The book's 5-bit format's values (Fig. A.4, denormal convention), ascending. */
export const TOY_VALUES: number[] = mfValues(TOY5);
export const BOOK_FOUR = [1, 1, 0.25, 0.25];
/** Ours: the tree is the worse order (found by brute force while designing; the tests confirm it). */
export const OUR_FOUR = [0.25, 1, 0.25, 2];

export interface ToyAdd {
  left: number;
  right: number;
  exact: number;
  rounded: number;
  /** exact - rounded: what the addition dropped (negative when it rounded up). */
  lost: number;
  /** The exact sum is beyond the format's largest value; the format clamps it (it has no infinity). */
  saturated: boolean;
}
export interface ToyRun {
  order: 'sequential' | 'tree';
  adds: ToyAdd[];
  result: number;
  exact: number;
  /** |result - exact|. */
  error: number;
  saturated: boolean;
}

function toyAdd(a: number, b: number): ToyAdd {
  const rounded = mfAdd(TOY5, a, b);
  const exact = a + b;
  return { left: a, right: b, exact, rounded, lost: exact - rounded, saturated: Math.abs(exact) > maxFinite(TOY5) };
}

/** Four values in the book's 5-bit format, added in order ((a + b) + c) + d or as the tree (a + b) + (c + d). */
export function toySum(values: number[], order: 'sequential' | 'tree'): ToyRun {
  const v = values.map((x) => mfRound(TOY5, x));
  const adds: ToyAdd[] = [];
  let result: number;
  if (order === 'sequential') {
    let s = v[0];
    for (let i = 1; i < v.length; i++) {
      const a = toyAdd(s, v[i]);
      adds.push(a);
      s = a.rounded;
    }
    result = s;
  } else {
    const l = toyAdd(v[0], v[1]);
    const r = toyAdd(v[2], v[3]);
    const top = toyAdd(l.rounded, r.rounded);
    adds.push(l, r, top);
    result = top.rounded;
  }
  const exact = v.reduce((a, b) => a + b, 0);
  return { order, adds, result, exact, error: Math.abs(result - exact), saturated: adds.some((a) => a.saturated) };
}

/** The challenge of p. 594: the tree is strictly less accurate than the loop, with no addition beyond the format's range. */
export function challengeHolds(values: number[]): boolean {
  if (values.length !== 4 || values.some((x) => !TOY_VALUES.includes(x))) return false;
  const s = toySum(values, 'sequential');
  const t = toySum(values, 'tree');
  return !s.saturated && !t.saturated && t.error > s.error;
}

// ---------------------------------------------------------------- the thread-to-data stepper (Exercise 4)
export type Pairing = 'convergent' | 'neighbours';
export interface TreeStep {
  level: number;
  stride: number;
  thread: number;
  /** The cell the thread writes (and reads first). */
  dst: number;
  /** The other cell it reads. */
  src: number;
  a: number;
  b: number;
  /** a + b before rounding. */
  exact: number;
  rounded: number;
  /** exact - rounded, exactly. */
  lost: number;
  /** The cells after this step. */
  cells: number[];
}
export interface TreeTrace {
  pairing: Pairing;
  /** The input as stored in the format. */
  input: number[];
  levels: { stride: number; active: number[] }[];
  steps: TreeStep[];
  result: number;
  /** The exact sum of the stored input. */
  exact: number;
  error: number;
}

/**
 * Chapter 10's two in-place trees with n/2 threads over n values (n a power of two):
 *  convergent (Fig. 10.8): stride n/2, n/4, ..., 1; thread t < stride adds cell t + stride into cell t;
 *  neighbours (Figs. 10.5 and 10.6): stride 1, 2, 4, ...; thread t with t % stride == 0 adds cell 2t + stride into cell 2t.
 * Steps go level by level and, within a level, thread by thread.
 */
export function treeTrace(values: number[], pairing: Pairing, f: Acc | Format): TreeTrace {
  const n = values.length;
  if (n < 2 || (n & (n - 1)) !== 0) throw new Error('treeTrace: a power of two of at least 2 values');
  const r = rounder(f);
  const add = adder(f);
  const input = values.map((x) => r(x));
  const cells = [...input];
  const threads = n / 2;
  const levels: TreeTrace['levels'] = [];
  const steps: TreeStep[] = [];
  const strides: number[] = [];
  if (pairing === 'convergent') for (let s = threads; s >= 1; s /= 2) strides.push(s);
  else for (let s = 1; s <= threads; s *= 2) strides.push(s);
  strides.forEach((stride, level) => {
    const active: number[] = [];
    for (let t = 0; t < threads; t++) {
      const on = pairing === 'convergent' ? t < stride : t % stride === 0;
      if (!on) continue;
      active.push(t);
      const dst = pairing === 'convergent' ? t : 2 * t;
      const src = dst + stride;
      const a = cells[dst];
      const b = cells[src];
      const s = a + b;
      const rounded = add(a, b);
      const lost = Number.isFinite(rounded) ? s - rounded + twoSumErr(a, b, s) : NaN;
      cells[dst] = rounded;
      steps.push({ level, stride, thread: t, dst, src, a, b, exact: s, rounded, lost, cells: [...cells] });
    }
    levels.push({ stride, active });
  });
  const exact = reference(input);
  return { pairing, input, levels, steps, result: cells[0], exact, error: Math.abs(cells[0] - exact) };
}

/**
 * The stepper's starting values per format: eight 1s, five 8s and three values B at which adding 1
 * is a tie that rounds back to B (B = 2^11 in FP16, 2^8 in BF16). Found by a search while
 * designing: on these values the neighbour tree is exact and the convergent tree loses all eight
 * 1s. (No set of 16 values of the book's 5-bit format does this without passing its largest
 * value, 3.5, so the stepper offers FP16 and BF16 only.)
 */
export const EX4_FORMATS: Acc[] = ['fp16', 'bf16'];
export const EX4_VALUES: Record<'fp16' | 'bf16', number[]> = {
  fp16: [1, 1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 2048, 2048, 2048],
  bf16: [1, 1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 256, 256, 256],
};
export type Ex4Order = 'ascending' | 'descending' | 'shuffled';
export function ex4Order(values: number[], order: Ex4Order, seed = 3): number[] {
  const v = [...values];
  if (order === 'ascending') return v.sort((a, b) => a - b);
  if (order === 'descending') return v.sort((a, b) => b - a);
  const r = rng(seed);
  for (let i = v.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [v[i], v[j]] = [v[j], v[i]];
  }
  return v;
}

// ---------------------------------------------------------------- Kahan and the bug hunt
export type KahanVariantName = 'correct' | 'late-c' | 'sign' | 'no-c';
/**
 * The four loops of labs/appA/snippets/kahan.h, statement for statement, in format f:
 *  correct  y = x - c; t = sum + y; c = (t - sum) - y; sum = t;
 *  late-c   y = x - c; t = sum + y; sum = t; c = (t - sum) - y;     (c computed after sum is updated)
 *  sign     y = x + c; t = sum + y; c = (t - sum) - y; sum = t;     (the correction's sign flipped)
 *  no-c     y = x - c; t = sum + y; c = 0;             sum = t;     ((t - sum) - y simplified as if + were associative)
 */
export function kahanVariant(name: KahanVariantName, values: ArrayLike<number>, f: Acc | Format): number {
  const r = rounder(f);
  let s = 0;
  let c = 0;
  for (let i = 0; i < values.length; i++) {
    const x = r(values[i]);
    const y = name === 'sign' ? r(x + c) : r(x - c);
    const t = r(s + y);
    if (name === 'late-c') {
      s = t;
      c = r(r(t - s) - y);
    } else {
      c = name === 'no-c' ? 0 : r(r(t - s) - y);
      s = t;
    }
  }
  return s;
}

/** Which version is in which slot of the bug hunt (and of kahan.h): the order the reader sees. */
export const BUG_HUNT: { version: number; name: KahanVariantName }[] = [
  { version: 1, name: 'sign' }, // i18n-ignore: variant identifier
  { version: 2, name: 'correct' }, // i18n-ignore: variant identifier
  { version: 3, name: 'late-c' }, // i18n-ignore: variant identifier
  { version: 4, name: 'no-c' },
];
export type Verdict = 'kahan' | 'plain' | 'worse';
export const VERDICT_LABEL: Record<Verdict, string> = {
  get kahan() {
    return tr('like Kahan', 'giống Kahan');
  },
  get plain() {
    return tr('like the plain sum', 'giống tổng thường');
  },
  get worse() {
    return tr('worse than plain', 'tệ hơn tổng thường');
  },
};

export interface VariantRun {
  name: KahanVariantName;
  result: Errors;
  verdict: Verdict;
}
/**
 * Runs the four versions and the plain loop on the same data and classifies each version:
 * within 1 ULP of the correct Kahan loop's error: "like Kahan"; otherwise within twice the plain
 * loop's error: "like the plain sum"; otherwise "worse than plain".
 */
export function bugHunt(values: ArrayLike<number>, f: Acc | Format): { plain: Errors; runs: VariantRun[] } {
  const ref = reference(values);
  const scale = absSum(values);
  const plain = errorsOf(sumSequential(values, f), ref, scale, f);
  const kahan = errorsOf(kahanVariant('correct', values, f), ref, scale, f);
  const runs = BUG_HUNT.map(({ name }) => {
    const e = errorsOf(kahanVariant(name, values, f), ref, scale, f);
    const verdict: Verdict = e.ulps <= kahan.ulps + 1 ? 'kahan' : e.ulps <= 2 * plain.ulps ? 'plain' : 'worse';
    return { name, result: e, verdict };
  });
  return { plain, runs };
}
/** The bug hunt's data, in FP32: 1.0, then 65,535 values of 2^-25 (the scene's values, for longer). */
export const BUG_HUNT_N = 65536;
export function bugHuntValues(): number[] {
  return [1, ...Array.from({ length: BUG_HUNT_N - 1 }, () => 2 ** -25)];
}

export interface KahanStep {
  /** 1-based index of the value. */
  i: number;
  x: number;
  y: number;
  t: number;
  /** c after this step. */
  c: number;
  /** The compensated sum after this step. */
  sum: number;
  /** The plain loop's sum after the same values. */
  plain: number;
  /** The exact sum of the values so far. */
  exact: number;
}
/** Kahan's loop with every intermediate, beside the plain loop and the exact sum (for the scene). */
export function kahanTrace(values: ArrayLike<number>, f: Acc | Format): KahanStep[] {
  const add = adder(f);
  const r = rounder(f);
  let s = 0;
  let c = 0;
  let p = 0;
  let es = 0;
  let ec = 0;
  const out: KahanStep[] = [];
  for (let i = 0; i < values.length; i++) {
    const x = r(values[i]);
    const y = add(x, -c);
    const t = add(s, y);
    c = add(add(t, -s), -y);
    s = t;
    p = add(p, x);
    const e = es + x;
    ec += Math.abs(es) >= Math.abs(x) ? es - e + x : x - e + es;
    es = e;
    out.push({ i: i + 1, x, y, t, c, sum: s, plain: p, exact: es + ec });
  }
  return out;
}

/** The scene: an FP32 accumulator holding 1.0, then 4,096 values of 2^-25 (a quarter of the gap above 1). */
export const SCENE = { big: 1, x: 2 ** -25, n: 4096 };
export function sceneValues(): number[] {
  return [SCENE.big, ...Array.from({ length: SCENE.n }, () => SCENE.x)];
}
/** A value near 1 as 1 + k·2^-25, or a small one as k·2^-25 (k an integer for every value in the scene). */
export const units25 = (v: number) => v * 2 ** 25;

// ---------------------------------------------------------------- the start page's teaser
export const TEASER_N = 131072;
export interface TeaserRow {
  id: 'fp32-loop' | 'fp32-tree' | 'fp16-loop' | 'fp16-tree';
  label: string;
  acc: Acc;
  method: Method;
  e: Errors;
}
const TEASER_ROWS: Omit<TeaserRow, 'e'>[] = [
  {
    id: 'fp32-loop',
    get label() {
      return tr('float loop', 'vòng lặp float');
    },
    acc: 'fp32', method: 'sequential',
  },
  {
    id: 'fp32-tree',
    get label() {
      return tr('float tree', 'cây float');
    },
    acc: 'fp32', method: 'tree',
  },
  {
    id: 'fp16-loop',
    get label() {
      return tr('half loop', 'vòng lặp half');
    },
    acc: 'fp16', method: 'sequential',
  },
  {
    id: 'fp16-tree',
    get label() {
      return tr('half tree', 'cây half');
    },
    acc: 'fp16', method: 'tree',
  },
];
/** One softmax denominator of a 131,072-entry row (standard-normal logits, the default seed), summed four ways. */
export function teaser(): { reference: number; rows: TeaserRow[] } {
  const x = generate('softmax', TEASER_N);
  const ref = reference(x);
  const scale = absSum(x);
  const rows = TEASER_ROWS.map((r) => ({ ...r, e: errorsOf(sum(x, { method: r.method, acc: r.acc }), ref, scale, r.acc) }));
  return { reference: ref, rows };
}

// ---------------------------------------------------------------- display helpers
/** A number for display: up to `digits` significant digits, no trailing zeros. */
export function show(v: number, digits = 7): string {
  if (Number.isNaN(v)) return 'NaN';
  if (v === Infinity) return '+inf';
  if (v === -Infinity) return '−inf';
  if (v === 0) return '0';
  const a = Math.abs(v);
  let s: string;
  if (Number.isInteger(v) && a < 1e12) s = String(v);
  else if (a >= 1e-4 && a < 1e9) s = String(Number(v.toPrecision(digits)));
  else s = v.toExponential(Math.max(0, digits - 1)).replace(/\.?0+e/, 'e');
  return s.replace(/^-/, '−');
}
/** A relative error as a percentage with two significant digits (or "0%"). */
export function pct(rel: number): string {
  if (!Number.isFinite(rel)) return 'overflow';
  if (rel === 0) return '0%';
  const p = rel * 100;
  return (p >= 1 ? String(Number(p.toPrecision(3))) : String(Number(p.toPrecision(2)))).replace(/e/, 'e') + '%';
}
/** An error in ULPs for display. */
export function ulpText(u: number): string {
  if (!Number.isFinite(u)) return 'overflow';
  if (u === 0) return '0';
  if (u < 0.01) return u.toExponential(1);
  return String(Number(u.toPrecision(u < 10 ? 2 : 3)));
}
