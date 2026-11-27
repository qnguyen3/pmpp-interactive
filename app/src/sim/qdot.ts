// One quantised block's dot product, three ways, and the accumulators it needs: the model
// behind module X2's sections X2.5 ("Dequantise, or multiply in integers") and X2.6 ("Where
// the accumulation must stay wide"), their interactives and both narrated scenes.
//
// Module X2 is not in the book. Sources (notes/today/x2-facts.md, checked 2026-11-20):
//  - D1: llama.cpp's Q4_0 block (FP16 d, 16 bytes, weight j in the low nibble of byte j and
//    weight j + 16 in its high nibble, value (q - 8) d) and Q8_1 (FP16 d and s = d sum q).
//    Blocks come from quant.ts's quantizeQ4_0 (llama.cpp's _ref rule), imported read-only.
//  - D3: llama.cpp's CUDA and HIP decode. Activations quantised to Q8_1 every token: a 32-wide
//    max, d = amax / 127, q = roundf(x / d) (0 when amax is 0), the integer sum, s = sum * d,
//    d and s stored as FP16. The block dot: vi0 = v & 0x0F0F0F0F, vi1 = (v >> 4) & 0x0F0F0F0F,
//    sumi = dp4a(vi0, u, sumi), sumi = dp4a(vi1, u', sumi), then d4 * (sumi * d8 - 8 * s).
//  - D4: llama.cpp's Metal decode (block_q_n_dot_y): 16-bit words masked with 0x000F, 0x0F00,
//    0x00F0, 0xF000 into four accumulators, activations pre-divided by 1, 256, 16, 4096,
//    result d * (sumy * -8 + acc0 + acc1 + acc2 + acc3). MLX (D5) and PyTorch MPS (D6) do the same.
//  - A2: dp4a semantics (four signed byte products added to a 32-bit integer).
//  - E3: the bounds 516,128; 60,960; 32,512; about 133,144 int32 terms. E4: the offset fold.
//  - C1, A3, B4: what dot(half4), __hfma2 and amd_mixed_dot accumulate in.
//  - FP16's limits (largest finite 65,504, smallest subnormal 2^-24): appendix A facts, A2.
//
// What it models, exactly:
//  - Every float operation is rounded once to its format (Math.fround for FP32; fpsum.ts's
//    rounder for FP16, which minifloat.ts checks). A multiply-add written a + b * c is one
//    rounding of the exact value (an FMA). The product of two float32 values is exact in
//    binary64, and so is its sum with a float32 accumulator at the magnitudes used here, so
//    one Math.fround of the binary64 expression is the FMA's result.
//  - Integers are exact; int32 wraps modulo 2^32 and int16 modulo 2^16 (what a C++20 store
//    of an int into a short does).
//  - "Exact" values are computed in exact binary rational arithmetic (BigInt), then rounded
//    once to binary64 for display.
//  - Operation counts are source-level: an FMA counts as one multiply and one add, a dp4a as
//    one integer operation, a division by a power of two as a multiply. They are not
//    instruction counts and say nothing about time.
// What it leaves out: the lanes of a warp and the order in which real kernels split a row
// among them (a row here is summed block after block), compiler contraction choices beyond
// the ones written, flush-to-zero, and anything about time.
//
// Erasable TypeScript only (the browser checks import this module through the app's server).

import { tr as trText } from '../lib/locale.ts';
import { rng, rounder } from './fpsum';
import { quantizeQ4_0, quantizeQ8_0 } from './quant';

export const QK = 32;
const f32 = Math.fround;
export const f16 = rounder('fp16');
export const FP16_MAX = 65504;
export const FP16_MIN_SUBNORMAL = 2 ** -24;
export const FP16_EXACT_INT = 2048;
export const FP32_EXACT_INT = 2 ** 24;
export const INT16_MAX = 32767;
export const INT32_MAX = 2 ** 31 - 1;

// ---------------------------------------------------------------- exact binary rationals
/** n * 2^e, exactly. */
interface Dy {
  n: bigint;
  e: number;
}
const dv = new DataView(new ArrayBuffer(8));
function dy(x: number): Dy {
  if (x === 0) return { n: 0n, e: 0 };
  if (!Number.isFinite(x)) throw new Error('not finite');
  dv.setFloat64(0, x);
  const hi = dv.getUint32(0);
  const lo = dv.getUint32(4);
  const sign = hi >>> 31 ? -1n : 1n;
  const ex = (hi >>> 20) & 0x7ff;
  let mant = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  let e: number;
  if (ex === 0) e = -1074;
  else {
    mant |= 1n << 52n;
    e = ex - 1075;
  }
  return { n: sign * mant, e };
}
const dyMul = (a: Dy, b: Dy): Dy => ({ n: a.n * b.n, e: a.e + b.e });
function dyAdd(a: Dy, b: Dy): Dy {
  if (a.n === 0n) return b;
  if (b.n === 0n) return a;
  const e = Math.min(a.e, b.e);
  return { n: (a.n << BigInt(a.e - e)) + (b.n << BigInt(b.e - e)), e };
}
function dyNum(a: Dy): number {
  if (a.n === 0n) return 0;
  // Keep 64 significant bits, then let Number() round once.
  const neg = a.n < 0n;
  let n = neg ? -a.n : a.n;
  let e = a.e;
  const bits = n.toString(2).length;
  if (bits > 64) {
    const drop = bits - 64;
    const sticky = n & ((1n << BigInt(drop)) - 1n) ? 1n : 0n;
    n = (n >> BigInt(drop)) | sticky;
    e += drop;
  }
  const v = Number(n) * 2 ** e;
  return neg ? -v : v;
}
const dySame = (a: Dy, b: Dy) => dyAdd(a, { n: -b.n, e: b.e }).n === 0n;

// ---------------------------------------------------------------- blocks and data
/** One llama.cpp Q4_0 block: the FP16 scale as a number, and 16 bytes of nibbles. */
export interface Q4Block {
  d: number;
  qs: ArrayLike<number>;
}

/** Q4_0 blocks of a float32 row, by quant.ts's quantizeQ4_0 (llama.cpp's _ref rule): d is the stored FP16 value. */
export function q4Blocks(w: ArrayLike<number>): Q4Block[] {
  const out: Q4Block[] = [];
  for (let i = 0; i < w.length; i += QK) {
    const b = quantizeQ4_0(Array.prototype.slice.call(w, i, i + QK) as number[]);
    out.push({ d: f16(b.d), qs: b.bytes.slice(2, 18) });
  }
  return out;
}

/** The 32 codes of a block, 0..15, in weight order: weight j < 16 is the low nibble of byte j, weight j + 16 its high nibble. */
export function codesOf(b: Q4Block): number[] {
  const c = new Array<number>(QK);
  for (let j = 0; j < 16; j++) {
    c[j] = b.qs[j] & 0x0f;
    c[j + 16] = b.qs[j] >> 4;
  }
  return c;
}
/** Dequantised weights (q - 8) * d. Exact in float32: an FP16 scale times a 4-bit integer. */
export const dequant = (b: Q4Block): number[] => codesOf(b).map((q) => f32((q - 8) * b.d));

function normal(r: () => number): number {
  return Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
}
/** Standard-normal float32 values. */
export function normals(n: number, seed: number): number[] {
  const r = rng(seed);
  return Array.from({ length: n }, () => f32(normal(r)));
}

export type ActDist = 'normal' | 'outliers';
export const ACT_DIST: { id: ActDist; label: string }[] = [
  { id: 'normal', get label() { return trText('normal', 'chuẩn tắc'); } },
  { id: 'outliers', get label() { return trText('normal, with a few large channels', 'chuẩn tắc, với vài kênh lớn'); } },
];
/** Where the large channels sit in a row of n: one in every 128, at a fixed place in each (our construction, not a measured model). */
export const OUTLIER_EVERY = 128;
export const outlierChannels = (n: number): number[] =>
  Array.from({ length: Math.max(1, n / OUTLIER_EVERY) }, (_, k) => k * OUTLIER_EVERY + ((k * 37 + 5) % OUTLIER_EVERY));
/** How large they are: magnitude 16 to 24, against a standard normal. Ours. */
export const OUTLIER_SIZE = 20;
/** Activations for a row: standard normal, optionally with the large channels. */
export function activations(n: number, seed: number, dist: ActDist): number[] {
  const r = rng(seed ^ 0x5bd1e995);
  const x = Array.from({ length: n }, () => f32(normal(r)));
  if (dist === 'outliers') for (const j of outlierChannels(n)) x[j] = f32((r() < 0.5 ? -1 : 1) * (OUTLIER_SIZE - 4 + 8 * r()));
  return x;
}

/** The stepper's block: 32 float32 weights (quantised to one Q4_0 block) and 32 float32 activations. */
export interface BlockCase {
  w: number[];
  x: number[];
  block: Q4Block;
}
export const CASE_SEED = 11;
/** In the stepper the large channel is weight 5 of the block. */
export const OUTLIER_J = 5;
export function makeCase(outlier = false, seed = CASE_SEED): BlockCase {
  const w = normals(QK, seed);
  const x = normals(QK, seed + 1000);
  if (outlier) x[OUTLIER_J] = f32(OUTLIER_SIZE);
  return { w, x, block: q4Blocks(w)[0] };
}

// ---------------------------------------------------------------- Q8_1 and dp4a
/** One 32-value Q8_1 activation block, by llama.cpp's CUDA rule (quantize_q8_1, D3). */
export interface Q8_1 {
  amax: number;
  /** d = amax / 127 in float32, before it is stored. */
  dFloat: number;
  /** The stored FP16 scale. */
  d: number;
  qs: number[];
  /** The integer sum of the codes. */
  sum: number;
  /** The stored FP16 s = sum * d (computed with the float32 d). */
  s: number;
}
/** roundf: to nearest, halves away from zero. */
export const roundf = (v: number) => (v < 0 ? -Math.round(-v) : Math.round(v)) | 0;
export function quantizeQ8_1(x: ArrayLike<number>): Q8_1 {
  let amax = 0;
  for (let j = 0; j < QK; j++) amax = Math.max(amax, Math.abs(x[j]));
  const dFloat = f32(amax / 127);
  const qs = new Array<number>(QK);
  let sum = 0;
  for (let j = 0; j < QK; j++) {
    qs[j] = amax === 0 ? 0 : roundf(f32(x[j] / dFloat));
    sum += qs[j];
  }
  return { amax, dFloat, d: f16(dFloat), qs, sum, s: f16(f32(sum * dFloat)) };
}

/** A signed byte of a 32-bit word. */
export const byteOf = (word: number, k: number) => ((word << (24 - 8 * k)) >> 24);
/** Four bytes (signed) to a 32-bit word, byte 0 lowest. */
export const wordOf = (b: number[]) => ((b[0] & 255) | ((b[1] & 255) << 8) | ((b[2] & 255) << 16) | ((b[3] & 255) << 24)) | 0;
/** dp4a's arithmetic (A2): four signed byte products added to c, in int32. */
export function dot4(a: number, b: number, c: number): number {
  let s = c | 0;
  for (let k = 0; k < 4; k++) s = (s + byteOf(a, k) * byteOf(b, k)) | 0;
  return s;
}
/** The 32-bit word i (0..3) of a block's qs, bytes 4i..4i+3. */
export const qsWord = (b: Q4Block, i: number) => wordOf([b.qs[4 * i], b.qs[4 * i + 1], b.qs[4 * i + 2], b.qs[4 * i + 3]]);
/** The 16-bit word k (0..7) of a block's qs, bytes 2k and 2k + 1: what the Metal kernel reads. */
export const qsHalfword = (b: Q4Block, k: number) => b.qs[2 * k] | (b.qs[2 * k + 1] << 8);

// ---------------------------------------------------------------- three paths, traced
export type Path = 'A' | 'B' | 'C';
export const PATH_LABEL: Record<Path, string> = {
  get A() { return trText('A. dequantise, then multiply', 'A. giải lượng tử hóa, rồi nhân'); },
  get B() { return trText('B. float, offset folded out', 'B. số thực, offset được tách ra ngoài'); },
  get C() { return trText('C. integer, activations to int8', 'C. số nguyên, activation về int8'); },
};
export interface Counts {
  /** float multiplies (an FMA counts one) */
  mul: number;
  /** float adds (an FMA counts one) */
  add: number;
  /** float compares (the max of the activation quantiser) */
  cmp: number;
  /** integer operations: masks, shifts, subtractions, integer adds; a dp4a counts one */
  int: number;
  /** conversions: integer to float, float to integer, to and from FP16 */
  conv: number;
}
const zero = (): Counts => ({ mul: 0, add: 0, cmp: 0, int: 0, conv: 0 });
const plus = (a: Counts, b: Partial<Counts>): Counts => ({
  mul: a.mul + (b.mul ?? 0), add: a.add + (b.add ?? 0), cmp: a.cmp + (b.cmp ?? 0), int: a.int + (b.int ?? 0), conv: a.conv + (b.conv ?? 0),
});

export type StepKind = 'group' | 'prepare' | 'quantise-max' | 'quantise-round' | 'word' | 'dp4a' | 'fold' | 'correct' | 'rescale';
export interface Lane {
  /** weight index in the block */
  j: number;
  code: number;
  /** what multiplies the activation: the dequantised weight (A), the masked nibble (B), the unsigned nibble byte (C) */
  wv: number;
  /** what it multiplies: the float activation (A), the pre-divided activation (B), the int8 code (C) */
  xv: number;
  /** their product: rounded in A and B (into the accumulator), exact in C */
  prod: number;
}
export interface QStep {
  kind: StepKind;
  /** per row (true) or per token, shared by every row that reads this activation block (false) */
  perRow: boolean;
  lanes?: Lane[];
  /** a 16-bit (B) or 32-bit (C) word of weight codes, its mask and the masked word */
  word?: number;
  mask?: number;
  shift?: number;
  masked?: number;
  /** the running accumulator after the step: float32 in A and B (B shows the sum of its four), int32 in C */
  acc: number;
  /** B's four accumulators */
  accs?: number[];
  /** what this step added to the counts */
  counts: Partial<Counts>;
  /** step-specific values */
  vals?: Record<string, number>;
}
export interface PathRun {
  path: Path;
  steps: QStep[];
  result: number;
  perRow: Counts;
  perToken: Counts;
  q8?: Q8_1;
  /** B's pre-divided activations */
  xs?: number[];
  sumx?: number;
}

const PRE = [1, 1 / 256, 1 / 16, 1 / 4096]; // what masks 0x000F, 0x0F00, 0x00F0, 0xF000 leave each nibble multiplied by
const MASKS16 = [0x000f, 0x0f00, 0x00f0, 0xf000];
/** Which weight each of the four masks of 16-bit word k selects. */
const j16 = (k: number) => [2 * k, 2 * k + 1, 2 * k + 16, 2 * k + 17];
/** Path B's pre-divided activation for weight j (exact: a power of two). */
export function prescaled(x: ArrayLike<number>): number[] {
  const xs = new Array<number>(QK);
  for (let k = 0; k < 8; k++) j16(k).forEach((j, m) => (xs[j] = f32(x[j] * PRE[m])));
  return xs;
}
/** Path B's sum of the block's activations, in float32, in order. */
export function sumOf(x: ArrayLike<number>): number {
  let s = 0;
  for (let j = 0; j < QK; j++) s = f32(s + x[j]);
  return s;
}

function traceA(c: BlockCase): PathRun {
  const { block: b, x } = c;
  const codes = codesOf(b);
  const steps: QStep[] = [];
  let acc = 0;
  for (let g = 0; g < 8; g++) {
    const lanes: Lane[] = [];
    for (let j = 4 * g; j < 4 * g + 4; j++) {
      const wv = f32((codes[j] - 8) * b.d);
      acc = f32(acc + wv * x[j]);
      lanes.push({ j, code: codes[j], wv, xv: x[j], prod: f32(wv * x[j]) });
    }
    // per weight: unpack the nibble, subtract 8, convert, multiply by d, multiply-add
    steps.push({ kind: 'group', perRow: true, lanes, acc, counts: { int: 8, conv: g === 0 ? 5 : 4, mul: 8, add: 4 } });
  }
  return finish('A', steps, acc);
}

function traceB(c: BlockCase): PathRun {
  const { block: b, x } = c;
  const xs = prescaled(x);
  const sumx = sumOf(x);
  const steps: QStep[] = [
    { kind: 'prepare', perRow: false, acc: 0, counts: { add: 31, mul: 24 }, vals: { sumx } },
  ];
  const accs = [0, 0, 0, 0];
  const codes = codesOf(b);
  for (let k = 0; k < 8; k++) {
    const word = qsHalfword(b, k);
    const lanes: Lane[] = j16(k).map((j, m) => {
      const wv = word & MASKS16[m];
      accs[m] = f32(accs[m] + xs[j] * wv);
      return { j, code: codes[j], wv, xv: xs[j], prod: f32(xs[j] * wv) };
    });
    steps.push({ kind: 'word', perRow: true, word, lanes, acc: f32(f32(f32(accs[0] + accs[1]) + accs[2]) + accs[3]), accs: [...accs], counts: { int: 4, conv: 4, mul: 4, add: 4 } });
  }
  // d * (sumy * -8 + acc0 + acc1 + acc2 + acc3), left to right, the first term a multiply-add
  let t = f32(accs[0] + sumx * -8);
  t = f32(t + accs[1]);
  t = f32(t + accs[2]);
  t = f32(t + accs[3]);
  steps.push({ kind: 'fold', perRow: true, acc: t, accs: [...accs], counts: { mul: 1, add: 4 }, vals: { sumx, fold: f32(sumx * -8), t } });
  const result = f32(b.d * t);
  steps.push({ kind: 'rescale', perRow: true, acc: result, counts: { mul: 1, conv: 1 }, vals: { d: b.d } });
  const run = finish('B', steps, result);
  return { ...run, xs, sumx };
}

function traceC(c: BlockCase): PathRun {
  const { block: b, x } = c;
  const q8 = quantizeQ8_1(x);
  const codes = codesOf(b);
  const steps: QStep[] = [
    { kind: 'quantise-max', perRow: false, acc: 0, counts: { cmp: 31, mul: 1, conv: 1 }, vals: { amax: q8.amax, dFloat: q8.dFloat, d: q8.d } },
    { kind: 'quantise-round', perRow: false, acc: 0, counts: { mul: 33, conv: 34, int: 31 }, vals: { sum: q8.sum, s: q8.s } },
  ];
  let sumi = 0;
  for (let i = 0; i < 4; i++) {
    const v = qsWord(b, i);
    for (const shift of [0, 4]) {
      const masked = (v >> shift) & 0x0f0f0f0f;
      const js = [0, 1, 2, 3].map((k) => 4 * i + k + (shift ? 16 : 0));
      const u = wordOf(js.map((j) => q8.qs[j]));
      sumi = dot4(masked, u, sumi);
      const lanes = js.map((j, k) => ({ j, code: codes[j], wv: byteOf(masked, k), xv: q8.qs[j], prod: byteOf(masked, k) * q8.qs[j] }));
      steps.push({ kind: 'dp4a', perRow: true, word: v, shift, mask: 0x0f0f0f0f, masked, lanes, acc: sumi, counts: { int: shift ? 3 : 2 } });
    }
  }
  const corr = f32(f32(sumi * q8.d) - f32(8 * q8.s));
  steps.push({ kind: 'correct', perRow: true, acc: corr, counts: { conv: 3, mul: 2, add: 1 }, vals: { sumi, d8: q8.d, s: q8.s, scaled: f32(sumi * q8.d), off: f32(8 * q8.s) } });
  const result = f32(b.d * corr);
  steps.push({ kind: 'rescale', perRow: true, acc: result, counts: { mul: 1, conv: 1 }, vals: { d: b.d } });
  return { ...finish('C', steps, result), q8 };
}

function finish(path: Path, steps: QStep[], result: number): PathRun {
  let perRow = zero();
  let perToken = zero();
  for (const s of steps) {
    if (s.perRow) perRow = plus(perRow, s.counts);
    else perToken = plus(perToken, s.counts);
  }
  return { path, steps, result, perRow, perToken };
}

export function tracePath(path: Path, c: BlockCase): PathRun {
  return path === 'A' ? traceA(c) : path === 'B' ? traceB(c) : traceC(c);
}

/** The exact dot product with the dequantised weights, rounded once to binary64. */
export function exactDequantDot(b: Q4Block, x: ArrayLike<number>): number {
  return dyNum(exactA(b, x));
}
/** The exact dot product with the original float32 weights, rounded once to binary64. */
export function exactFloatDot(w: ArrayLike<number>, x: ArrayLike<number>): number {
  let s: Dy = { n: 0n, e: 0 };
  for (let j = 0; j < w.length; j++) s = dyAdd(s, dyMul(dy(w[j]), dy(x[j])));
  return dyNum(s);
}
function exactA(b: Q4Block, x: ArrayLike<number>): Dy {
  const codes = codesOf(b);
  let s: Dy = { n: 0n, e: 0 };
  for (let j = 0; j < QK; j++) s = dyAdd(s, dyMul(dyMul(dy(codes[j] - 8), dy(b.d)), dy(x[j])));
  return s;
}
function exactB(b: Q4Block, x: ArrayLike<number>): Dy {
  // d * (sum_j masked_j * x'_j  -  8 * sum_j x_j), every operation exact
  let acc: Dy = { n: 0n, e: 0 };
  let sx: Dy = { n: 0n, e: 0 };
  for (let k = 0; k < 8; k++) {
    const word = qsHalfword(b, k);
    j16(k).forEach((j, m) => {
      acc = dyAdd(acc, dyMul(dy(word & MASKS16[m]), dyMul(dy(x[j]), dy(PRE[m]))));
    });
  }
  for (let j = 0; j < QK; j++) sx = dyAdd(sx, dy(x[j]));
  return dyMul(dy(b.d), dyAdd(acc, dyMul(dy(-8), sx)));
}
/** Paths A and B in exact arithmetic: E4's algebra says they are the same number. */
export function exactPaths(b: Q4Block, x: ArrayLike<number>): { A: number; B: number; same: boolean } {
  const a = exactA(b, x);
  const bb = exactB(b, x);
  return { A: dyNum(a), B: dyNum(bb), same: dySame(a, bb) };
}
/** The Predict in X2.5: over `count` seeded blocks (seeds 1..count), how often float32 paths A and B agree bit for bit, and the largest gap relative to the block's scale. */
export function bitwiseAgreement(count = 40): { same: number; count: number; worstGap: number } {
  let same = 0;
  let worstGap = 0;
  for (let s = 1; s <= count; s++) {
    const c = makeCase(false, s);
    const a = blockA(c.block, c.x);
    const b = blockB(c.block, c.x);
    if (a === b) same++;
    const scale = dequant(c.block).reduce((t, w, j) => t + Math.abs(w * c.x[j]), 0);
    worstGap = Math.max(worstGap, Math.abs(a - b) / scale);
  }
  return { same, count, worstGap };
}
/** Is dividing x by 2^k exact in float32? (Yes, unless the quotient falls below float's normal range.) */
export const preDivisionExact = (x: number, k: number) => f32(x / 2 ** k) * 2 ** k === x;

// ---------------------------------------------------------------- whole rows (fast, untraced)
/** Path B for one block, the same arithmetic as the trace without the bookkeeping. */
export function blockB(b: Q4Block, x: ArrayLike<number>, off = 0): number {
  const accs = [0, 0, 0, 0];
  let sumx = 0;
  for (let j = 0; j < QK; j++) sumx = f32(sumx + x[off + j]);
  for (let k = 0; k < 8; k++) {
    const word = b.qs[2 * k] | (b.qs[2 * k + 1] << 8);
    const js = j16(k);
    for (let m = 0; m < 4; m++) accs[m] = f32(accs[m] + f32(x[off + js[m]] * PRE[m]) * (word & MASKS16[m]));
  }
  let t = f32(accs[0] + sumx * -8);
  t = f32(t + accs[1]);
  t = f32(t + accs[2]);
  t = f32(t + accs[3]);
  return f32(b.d * t);
}
/** Path C for one block, given its activation block's Q8_1. */
export function blockC(b: Q4Block, q8: Q8_1): number {
  let sumi = 0;
  for (let i = 0; i < 4; i++) {
    const v = qsWord(b, i);
    sumi = dot4(v & 0x0f0f0f0f, wordOf(q8.qs.slice(4 * i, 4 * i + 4)), sumi);
    sumi = dot4((v >> 4) & 0x0f0f0f0f, wordOf(q8.qs.slice(4 * i + 16, 4 * i + 20)), sumi);
  }
  return f32(b.d * f32(f32(sumi * q8.d) - f32(8 * q8.s)));
}
/** Path A for one block. */
export function blockA(b: Q4Block, x: ArrayLike<number>, off = 0): number {
  const codes = codesOf(b);
  let acc = 0;
  for (let j = 0; j < QK; j++) acc = f32(acc + f32((codes[j] - 8) * b.d) * x[off + j]);
  return acc;
}

export interface RowDot {
  A: number;
  B: number;
  C: number;
  /** exact, with the dequantised weights */
  exact: number;
  /** exact, with the float32 weights before quantisation */
  exactFloat: number;
  /** sum of |dequantised w_j * x_j|: the scale relative errors are measured against */
  scale: number;
  /** the same with the float32 weights */
  scaleFloat: number;
}
/** One row (n a multiple of 32): each path's block results summed block after block in float32. */
export function rowDot(w: ArrayLike<number>, x: ArrayLike<number>): RowDot {
  const blocks = q4Blocks(w);
  let A = 0, B = 0, C = 0, exact = 0, exactFloat = 0, scale = 0, scaleFloat = 0;
  let cE = 0, cF = 0; // Neumaier compensation for the binary64 references
  blocks.forEach((b, i) => {
    const off = i * QK;
    const xb = Array.prototype.slice.call(x, off, off + QK) as number[];
    A = f32(A + blockA(b, x, off));
    B = f32(B + blockB(b, x, off));
    C = f32(C + blockC(b, quantizeQ8_1(xb)));
    const wd = dequant(b);
    for (let j = 0; j < QK; j++) {
      const p = wd[j] * xb[j];
      const q = w[off + j] * xb[j];
      const t = exact + p;
      cE += Math.abs(exact) >= Math.abs(p) ? exact - t + p : p - t + exact;
      exact = t;
      const u = exactFloat + q;
      cF += Math.abs(exactFloat) >= Math.abs(q) ? exactFloat - u + q : q - u + exactFloat;
      exactFloat = u;
      scale += Math.abs(p);
      scaleFloat += Math.abs(q);
    }
  });
  return { A, B, C, exact: exact + cE, exactFloat: exactFloat + cF, scale, scaleFloat };
}

export interface RowErrors {
  n: number;
  dist: ActDist;
  /** |C - B| / scale per row: what quantising the activations costs */
  actErr: number[];
  /** |B - exact float32-weight dot| / scaleFloat per row: what quantising the weights cost */
  weightErr: number[];
  median: number;
  p95: number;
  weightMedian: number;
}
export const ROW_LENGTHS = [256, 1024, 4096];
export const ROWS = 200;
const quantile = (v: number[], q: number) => {
  const s = [...v].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
};
/** Path C against path B on many seeded rows. */
export function rowErrors(n: number, dist: ActDist, rows = ROWS, seed = 1): RowErrors {
  const actErr: number[] = [];
  const weightErr: number[] = [];
  for (let r = 0; r < rows; r++) {
    const w = normals(n, seed * 7919 + r);
    const x = activations(n, seed * 104729 + r, dist);
    const d = rowDot(w, x);
    actErr.push(Math.abs(d.C - d.B) / d.scale);
    weightErr.push(Math.abs(d.B - d.exactFloat) / d.scaleFloat);
  }
  return { n, dist, actErr, weightErr, median: quantile(actErr, 0.5), p95: quantile(actErr, 0.95), weightMedian: quantile(weightErr, 0.5) };
}
/** Log-spaced histogram bins of relative errors, from 1e-8 to 1, four bins a decade; zeros go into the first bin. */
export function histogram(v: number[], lo = -8, hi = 0, perDecade = 4): { from: number; to: number; count: number }[] {
  const nb = (hi - lo) * perDecade;
  const bins = Array.from({ length: nb }, (_, i) => ({ from: 10 ** (lo + i / perDecade), to: 10 ** (lo + (i + 1) / perDecade), count: 0 }));
  for (const e of v) {
    const k = e <= 0 ? 0 : Math.floor((Math.log10(e) - lo) * perDecade);
    bins[Math.max(0, Math.min(nb - 1, k))].count++;
  }
  return bins;
}

// ---------------------------------------------------------------- bounds (E3)
export const MAX_Q8_PRODUCT = 127 * 127; // 16,129
export const BOUND_Q8Q8 = QK * MAX_Q8_PRODUCT; // 516,128
export const BOUND_Q4_UNSIGNED = QK * 15 * 127; // 60,960
export const BOUND_Q4_SIGNED = QK * 8 * 127; // 32,512
export const DP4A_STEP_Q4_UNSIGNED = 4 * 15 * 127; // 7,620
/** How many terms of size p an int32 holds before the next one overflows. */
export const int32Terms = (p: number) => Math.floor(INT32_MAX / p);

// ---------------------------------------------------------------- accumulators (X2.6)
export type Inner = 'int16' | 'int32' | 'fp16' | 'fp32';
export type Outer = 'fp16' | 'fp32';
export type Pairing = 'q8q8' | 'q4q8';
export type DataSet = 'realistic' | 'worst';
export const INNER: { id: Inner; label: string }[] = [
  { id: 'int16', label: 'int16 (short)' }, // i18n-ignore: type names
  { id: 'int32', label: 'int32' }, // i18n-ignore: type names
  { id: 'fp16', label: 'FP16 (half)' }, // i18n-ignore: type names
  { id: 'fp32', label: 'FP32 (float)' }, // i18n-ignore: type names
];
export const OUTER: { id: Outer; label: string }[] = [
  { id: 'fp16', label: 'FP16 (half)' }, // i18n-ignore: type names
  { id: 'fp32', label: 'FP32 (float)' }, // i18n-ignore: type names
];
export const PAIRING: { id: Pairing; label: string }[] = [
  { id: 'q8q8', get label() { return trText('Q8_0 weights × Q8_1 activations', 'trọng số Q8_0 × activation Q8_1'); } },
  { id: 'q4q8', get label() { return trText('Q4_0 nibbles (0..15) × Q8_1 activations', 'nibble Q4_0 (0..15) × activation Q8_1'); } },
];
/** The largest magnitude an inner accumulator type holds, and the largest integer up to which it is exact. */
export const INNER_LIMIT: Record<Inner, { max: number; exactTo: number }> = {
  int16: { max: INT16_MAX, exactTo: INT16_MAX },
  int32: { max: INT32_MAX, exactTo: INT32_MAX },
  fp16: { max: FP16_MAX, exactTo: FP16_EXACT_INT },
  fp32: { max: 3.4028234663852886e38, exactTo: FP32_EXACT_INT },
};
export const wrap16 = (v: number) => ((v << 16) >> 16);

/** The integer data of one row: per block, 32 weight codes, 32 activation codes, and the scales. */
export interface IntRow {
  pairing: Pairing;
  blocks: { qw: number[]; qx: number[]; dw: number; dx: number; s: number }[];
}
/** Realistic: seeded standard-normal weights and activations, coded as the pairing says. Worst: every code at its extreme, with the realistic scales. */
export function intRow(pairing: Pairing, data: DataSet, n: number, seed = 3): IntRow {
  const w = normals(n, seed);
  const x = normals(n, seed + 1);
  const nb = n / QK;
  const q4 = pairing === 'q4q8' ? q4Blocks(w) : null;
  const blocks = [];
  for (let i = 0; i < nb; i++) {
    const xa = quantizeQ8_1(x.slice(i * QK, (i + 1) * QK));
    const wa = q4 ? null : quantizeQ8_0(w.slice(i * QK, (i + 1) * QK));
    let qw = q4 ? codesOf(q4[i]) : wa!.codes;
    let qx = xa.qs;
    let s = xa.s;
    if (data === 'worst') {
      qw = new Array(QK).fill(q4 ? 15 : 127);
      qx = new Array(QK).fill(127);
      s = f16(f32(QK * 127 * xa.dFloat));
    }
    blocks.push({ qw, qx, dw: q4 ? q4[i].d : f16(wa!.d), dx: xa.d, s });
  }
  return { pairing, blocks };
}

export type Event = 'wrap' | 'inexact' | 'inf';
export interface InnerTrace {
  /** the accumulator after each step */
  values: number[];
  /** the exact running sum after each step */
  exact: number[];
  /** first step (1-based) at which each event happened, if it did */
  first: Partial<Record<Event, number>>;
  /** products per step */
  step: number;
}
/** One block's integer sum, accumulated in `inner`, `step` products at a time (step 4 is a dp4a; float types always add one product at a time). */
export function innerSum(qw: number[], qx: number[], inner: Inner, step: 1 | 4 = 1): InnerTrace {
  const st = inner === 'fp16' || inner === 'fp32' ? 1 : step;
  const values: number[] = [];
  const exact: number[] = [];
  const first: Partial<Record<Event, number>> = {};
  let v = 0;
  let e = 0;
  const mark = (ev: Event, k: number) => {
    if (first[ev] === undefined) first[ev] = k;
  };
  for (let j = 0, k = 1; j < qw.length; j += st, k++) {
    let part = 0;
    for (let t = j; t < j + st; t++) part += qw[t] * qx[t];
    e += part;
    if (inner === 'int16') {
      if (v + part > INT16_MAX || v + part < -INT16_MAX - 1) mark('wrap', k);
      v = wrap16(v + part);
    } else if (inner === 'int32') {
      if (v + part > INT32_MAX || v + part < -INT32_MAX - 1) mark('wrap', k);
      v = (v + part) | 0;
    } else {
      v = inner === 'fp16' ? f16(v + part) : f32(v + part);
      if (!Number.isFinite(v)) mark('inf', k);
      else if (v !== e) mark('inexact', k);
    }
    values.push(v);
    exact.push(e);
  }
  return { values, exact, first, step: st };
}

export interface LadderOpts {
  inner: Inner;
  outer: Outer;
  data: DataSet;
  n: number;
  pairing: Pairing;
  step?: 1 | 4;
  seed?: number;
}
export interface LadderResult {
  /** block 0's inner trace */
  block0: InnerTrace;
  /** how many blocks' inner sums wrapped, overflowed, or rounded */
  wrapped: number;
  overflowed: number;
  rounded: number;
  result: number;
  reference: number;
  scale: number;
  /** |result - reference| / scale; Infinity when the result is not finite */
  relErr: number;
  /** |result - reference|; Infinity when the result is not finite */
  absErr: number;
  /** after every block: the running relative error of the outer sum */
  outerTrace: number[];
  /** after every block: the running absolute error of the outer sum */
  outerAbs: number[];
}
const OUTER_ROUND: Record<Outer, (v: number) => number> = { fp16: (v) => f16(f32(v)), fp32: f32 };
/** A block's float result from its inner sum (D3's two formulas). */
export function blockResult(pairing: Pairing, b: IntRow['blocks'][number], inner: number): number {
  return pairing === 'q8q8' ? f32(f32(b.dw * b.dx) * inner) : f32(b.dw * f32(f32(inner * b.dx) - f32(8 * b.s)));
}
export function accumulate(o: LadderOpts): LadderResult {
  const row = intRow(o.pairing, o.data, o.n, o.seed ?? 3);
  const round = OUTER_ROUND[o.outer];
  let out = 0;
  let ref = 0;
  let refC = 0;
  let scale = 0;
  let wrapped = 0, overflowed = 0, rounded = 0;
  const outerTrace: number[] = [];
  const outerAbs: number[] = [];
  let block0: InnerTrace | null = null;
  for (const b of row.blocks) {
    const tr = innerSum(b.qw, b.qx, o.inner, o.step ?? 1);
    block0 ??= tr;
    if (tr.first.wrap !== undefined) wrapped++;
    if (tr.first.inf !== undefined) overflowed++;
    else if (tr.first.inexact !== undefined) rounded++;
    const exactInner = tr.exact[tr.exact.length - 1];
    const r = blockResult(o.pairing, b, tr.values[tr.values.length - 1]);
    // The reference: the same block formula in binary64 on the exact inner sum.
    const re = o.pairing === 'q8q8' ? b.dw * b.dx * exactInner : b.dw * (exactInner * b.dx - 8 * b.s);
    out = round(out + r);
    const t = ref + re;
    refC += Math.abs(ref) >= Math.abs(re) ? ref - t + re : re - t + ref;
    ref = t;
    scale += Math.abs(re);
    const ae = Number.isFinite(out) ? Math.abs(out - (ref + refC)) : Infinity;
    outerAbs.push(ae);
    outerTrace.push(ae / scale);
  }
  const reference = ref + refC;
  return {
    block0: block0!,
    wrapped,
    overflowed,
    rounded,
    result: out,
    reference,
    scale,
    relErr: Number.isFinite(out) ? Math.abs(out - reference) / scale : Infinity,
    absErr: Number.isFinite(out) ? Math.abs(out - reference) : Infinity,
    outerTrace,
    outerAbs,
  };
}
/** Row lengths of the ladder's slider: 32 to 16,384, doubling. */
export const LADDER_LENGTHS = Array.from({ length: 10 }, (_, k) => 32 * 2 ** k);

/** FP16 rounding of a pre-divided activation (the X2.6 question): 1e-4 / 4096 in half. */
export const halfPrescaled = (x: number, k: number) => f16(f32(x / 2 ** k));

// ---------------------------------------------------------------- the bug spot (four fragments)
export type FragmentId = 'metal-half-dot' | 'cuda-hfma2' | 'short-sum' | 'int-then-float';
export interface FragmentRun {
  id: FragmentId;
  safe: boolean;
  /** relative error on the realistic row, against the exact value the fragment means to compute */
  realistic: number;
  /** what happened on the worst-case row */
  worst: { result: number; reference: number; relErr: number; event: 'none' | 'wrap' | 'inf'; at?: number };
}
export const FRAG_N = 4096;
/**
 * What each fragment of labs/x2/snippets/accumulators.md accumulates in, as runFragment models
 * it. `decl` is the accumulator's declaration as the fragment spells it, `op` the operation
 * that adds into it; qdot.snippets.test.ts checks both against the file.
 */
export const FRAGMENT_MODEL: Record<FragmentId, { decl: string; op: string; inner: 'half-dot4' | 'half2-row' | 'int16' | 'int32'; outer: 'fp32' }> = {
  'metal-half-dot': { decl: 'float sumf', op: 'dot(w, half4(', inner: 'half-dot4', outer: 'fp32' }, // i18n-ignore: code shown as written
  'cuda-hfma2': { decl: '__half2 acc', op: '__hfma2(', inner: 'half2-row', outer: 'fp32' }, // i18n-ignore: code shown as written
  'short-sum': { decl: 'short sumi', op: 'dot4(', inner: 'int16', outer: 'fp32' }, // i18n-ignore: code shown as written
  'int-then-float': { decl: 'int sumi', op: 'dot4(', inner: 'int32', outer: 'fp32' }, // i18n-ignore: code shown as written
};
export interface FragmentText { id: string; title: string; lang: string; code: string }
/** The fragments of labs/x2/snippets/accumulators.md: one "## id" heading, a Title and Language line and one fenced block each. */
export function parseFragments(md: string): FragmentText[] {
  return md.split(/^## /m).slice(1).map((part) => {
    const id = part.slice(0, part.indexOf('\n')).trim();
    const title = /^Title: (.*)$/m.exec(part)?.[1].trim() ?? id;
    const lang = /^Language: (.*)$/m.exec(part)?.[1].trim() ?? '';
    const code = /```[a-z]*\n([\s\S]*?)\n```/.exec(part)?.[1] ?? '';
    return { id, title, lang, code };
  });
}
/** The worst-case row for the fragments: every Q4_0 code 15 (weight 7 d) and every activation code 127, d = 1 on both sides, so the float fragments see weights of 7 and activations of 127. */
function worstRow(n: number) {
  const nb = n / QK;
  return Array.from({ length: nb }, () => ({ qw: new Array(QK).fill(15), qx: new Array(QK).fill(127), dw: 1, dx: 1, s: f16(QK * 127) }));
}
/** Float fragments' worst case: the same row read as floats. */
const worstFloat = (n: number) => ({ w: new Array(n).fill(7), x: new Array(n).fill(127) });

function fragFloat(id: 'metal-half-dot' | 'cuda-hfma2', w: number[], x: number[]): { v: number; at?: number } {
  if (FRAGMENT_MODEL[id].inner === 'half-dot4') {
    // float sumf; sumf += dot(half4 w, half4 x): the four-term dot is a chain of half multiply-adds
    let sumf = 0;
    for (let i = 0; i < w.length; i += 4) {
      let t = 0;
      for (let k = i; k < i + 4; k++) t = f16(t + f16(w[k]) * f16(x[k]));
      sumf = f32(sumf + t);
      if (!Number.isFinite(sumf)) return { v: sumf, at: i / 4 + 1 };
    }
    return { v: sumf };
  }
  // __half2 acc; acc = __hfma2(w2[i], x2[i], acc) over the whole row, then the two halves added in float
  const lane = [0, 0];
  let at: number | undefined;
  for (let i = 0; i < w.length; i++) {
    lane[i & 1] = f16(lane[i & 1] + f16(w[i]) * f16(x[i]));
    if (at === undefined && !Number.isFinite(lane[i & 1])) at = i + 1;
  }
  return { v: f32(lane[0] + lane[1]), at };
}
function fragInt(id: 'short-sum' | 'int-then-float', blocks: IntRow['blocks']): { v: number; at?: number } {
  let sumf = 0;
  let at: number | undefined;
  for (const b of blocks) {
    const tr = innerSum(b.qw, b.qx, FRAGMENT_MODEL[id].inner as 'int16' | 'int32', 4);
    if (at === undefined && tr.first.wrap !== undefined) at = tr.first.wrap;
    sumf = f32(sumf + blockResult('q4q8', b, tr.values[tr.values.length - 1]));
  }
  return { v: sumf, at };
}
const SAFE: Record<FragmentId, boolean> = { 'metal-half-dot': false, 'cuda-hfma2': false, 'short-sum': false, 'int-then-float': true };
export const FRAGMENTS: FragmentId[] = ['metal-half-dot', 'cuda-hfma2', 'short-sum', 'int-then-float'];
export function runFragment(id: FragmentId, n = FRAG_N, seed = 5): FragmentRun {
  const isFloat = id === 'metal-half-dot' || id === 'cuda-hfma2';
  // realistic
  const w = normals(n, seed);
  const x = normals(n, seed + 1);
  const blocks = q4Blocks(w);
  let realistic: number;
  if (isFloat) {
    const wd = blocks.flatMap(dequant);
    let sc = 0;
    for (let j = 0; j < n; j++) sc += Math.abs(wd[j] * x[j]);
    const ex = dyNum(blocks.reduce((s, b, i) => dyAdd(s, exactA(b, x.slice(i * QK, (i + 1) * QK))), { n: 0n, e: 0 } as Dy));
    realistic = Math.abs(fragFloat(id, wd, x).v - ex) / sc;
  } else {
    const row = intRow('q4q8', 'realistic', n, seed);
    let ref = 0, sc = 0;
    for (const b of row.blocks) {
      const e = innerSum(b.qw, b.qx, 'int32', 4).exact[7];
      const re = b.dw * (e * b.dx - 8 * b.s);
      ref += re;
      sc += Math.abs(re);
    }
    realistic = Math.abs(fragInt(id, row.blocks).v - ref) / sc;
  }
  // worst
  let worst: FragmentRun['worst'];
  const perBlock = QK * 7 * 127; // 28,448: the block's value after the -8 correction
  const reference = (n / QK) * perBlock;
  if (isFloat) {
    const { w: ww, x: wx } = worstFloat(n);
    const r = fragFloat(id, ww, wx);
    worst = { result: r.v, reference, relErr: Number.isFinite(r.v) ? Math.abs(r.v - reference) / reference : Infinity, event: Number.isFinite(r.v) ? 'none' : 'inf', at: r.at };
  } else {
    const r = fragInt(id, worstRow(n));
    worst = { result: r.v, reference, relErr: Math.abs(r.v - reference) / reference, event: r.at !== undefined ? 'wrap' : 'none', at: r.at };
  }
  return { id, safe: SAFE[id], realistic, worst };
}

// ---------------------------------------------------------------- scenes
/** The integer-path scene's row: eight blocks, so four lanes take two blocks each (a simplification of the real lane layout). */
export const SCENE_BLOCKS = 8;
export const SCENE_LANES = 4;
export interface SceneRow {
  case0: BlockCase;
  /** per block: path C's and path B's float results */
  c: number[];
  b: number[];
  /** per lane: its running float sums of path C */
  lanes: number[];
  /** the warp's reduction of the lanes, in float32 */
  outC: number;
  /** path B's row result, summed the same way */
  outB: number;
  exact: number;
}
export function integerSceneRow(seed = CASE_SEED): SceneRow {
  const n = SCENE_BLOCKS * QK;
  const w = normals(n, seed);
  const x = normals(n, seed + 1000);
  const blocks = q4Blocks(w);
  const c = blocks.map((b, i) => blockC(b, quantizeQ8_1(x.slice(i * QK, (i + 1) * QK))));
  const bb = blocks.map((b, i) => blockB(b, x, i * QK));
  const lanes = new Array(SCENE_LANES).fill(0);
  const lanesB = new Array(SCENE_LANES).fill(0);
  c.forEach((v, i) => (lanes[i % SCENE_LANES] = f32(lanes[i % SCENE_LANES] + v)));
  bb.forEach((v, i) => (lanesB[i % SCENE_LANES] = f32(lanesB[i % SCENE_LANES] + v)));
  const tree = (v: number[]) => f32(f32(v[0] + v[2]) + f32(v[1] + v[3]));
  let exact = 0;
  blocks.forEach((b, i) => (exact += exactDequantDot(b, x.slice(i * QK, (i + 1) * QK))));
  return {
    case0: { w: w.slice(0, QK), x: x.slice(0, QK), block: blocks[0] },
    c,
    b: bb,
    lanes,
    outC: tree(lanes),
    outB: tree(lanesB),
    exact,
  };
}

/** The narrow-accumulator scene: the worst-case Q8_0 × Q8_1 block, one product at a time, into four accumulators. */
export function narrowScene() {
  const qw = new Array(QK).fill(127);
  const qx = new Array(QK).fill(127);
  const types: Inner[] = ['int16', 'fp16', 'int32', 'fp32'];
  const traces = Object.fromEntries(types.map((t) => [t, innerSum(qw, qx, t, 1)])) as Record<Inner, InnerTrace>;
  const half = accumulate({ inner: 'int32', outer: 'fp16', data: 'realistic', n: 4096, pairing: 'q8q8' });
  const float = accumulate({ inner: 'int32', outer: 'fp32', data: 'realistic', n: 4096, pairing: 'q8q8' });
  return { traces, outerHalf: half.outerAbs, outerFloat: float.outerAbs, half, float };
}

// ---------------------------------------------------------------- display
export function show(v: number, digits = 6): string {
  if (Number.isNaN(v)) return 'NaN';
  if (v === Infinity) return '+inf';
  if (v === -Infinity) return '−inf';
  if (v === 0) return '0';
  const a = Math.abs(v);
  let s: string;
  if (Number.isInteger(v) && a < 1e9) s = v.toLocaleString('en-US');
  else if (a >= 1e-3 && a < 1e7) s = String(Number(v.toPrecision(digits)));
  else s = v.toExponential(Math.max(0, digits - 2)).replace(/\.?0+e/, 'e');
  return s.replace(/^-/, '−');
}
export const hex = (v: number, digits = 8) => '0x' + (v >>> 0).toString(16).toUpperCase().padStart(digits, '0');
export function sci(v: number): string {
  if (!Number.isFinite(v)) return 'overflow';
  if (v === 0) return '0';
  return v.toExponential(1).replace('e-', 'e−');
}
