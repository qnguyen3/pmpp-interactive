// Softmax over a short row in FP32, BF16 or FP16, for appendix A section A.3's "Softmax
// overflow" explorer: where infinities and NaNs come from in practice, and why subtracting
// the row's maximum (lab 10.3, milestone M4) removes them.
//
// Sources: overflow saturates to infinity (book §A.3, p. 590); ∞/∞ and ∞ − ∞ are NaN
// (p. 591); GPUs raise no exception, so a NaN spreads silently (p. 591; appA-facts A7, B6,
// C2). The book never mentions softmax: putting these together is ours (notes/appA.md,
// "Our inferences"). Formats and every rounding come from ./minifloat.
//
// The computation, in the order a kernel does it:
//   1. x_i, the logit as stored in the format;
//   2. z_i = x_i × scale (1/temperature), rounded;
//   3. m = max z_i; d_i = z_i − m, rounded (or d_i = z_i without the subtraction);
//   4. t_i = exp(d_i), computed in binary64 and rounded once;
//   5. s_i = s_{i-1} + t_i, a sequential sum, each addition rounded;
//   6. p_i = t_i / s_n, rounded.
//
// What it leaves out: exp is correctly rounded here, where real exp functions are within a
// few ULPs (expf 2 ULP, Metal exp 4 ulp: ch10-facts A11, C4); the sum is sequential, where
// kernels use trees (other orders change the low bits, section A.6); rounding is IEEE round
// to nearest even with denormals kept. Under Metal's fast math, the behaviour of INF and
// NaN is undefined (appA-facts C2), so this model says nothing about that case.
//
// Erasable TypeScript only.
import { BF16, FP16, FP32, add, div, maxFinite, mul, round, sub, ulp, type Format } from './minifloat';
import { tr } from '../lib/locale.ts';

export const SOFTMAX_FORMATS: Format[] = [FP32, BF16, FP16];

/**
 * The largest exponent argument whose exp still rounds to a finite value is just below this:
 * exp(d) overflows exactly when exp(d) >= the largest value plus half a unit in its last place
 * (round to nearest even sends that tie up, to the next power of two). About 88.72 for FP32
 * and BF16, about 11.09 for FP16. Our arithmetic on the formats of appA-facts A2.
 */
export function overflowThreshold(f: Format): number {
  const max = maxFinite(f);
  return Math.log(max + ulp(f, max) / 2);
}

export type Quantity = 'z' | 'd' | 't' | 's' | 'p';
export type Cause =
  | 'scaled-overflow' // x · scale left the format's range
  | 'inf-minus-inf' // subtracting an infinite maximum from an infinite logit
  | 'exp-overflow' // exp(d) is larger than the format's largest value
  | 'sum-overflow' // the running sum passed the largest value although every term was finite
  | 'inf-over-inf' // an infinite term divided by an infinite sum
  | 'nan-propagated'; // a NaN went in, so a NaN came out

export interface Event {
  kind: 'inf' | 'nan';
  /** Which element (0-based). */
  index: number;
  quantity: Quantity;
  cause: Cause;
}

export interface SoftmaxElement {
  x: number;
  z: number;
  d: number;
  t: number;
  /** The running sum after this element's term. */
  s: number;
  p: number;
  /** exp(d) was positive but too small for the format: the term rounded to 0. */
  underflow: boolean;
}

export interface SoftmaxTrace {
  f: Format;
  scale: number;
  subtractMax: boolean;
  rows: SoftmaxElement[];
  /** The maximum of z (also when it is not subtracted). */
  max: number;
  denom: number;
  /** The first infinity and the first NaN, in the order the kernel computes. */
  firstInf: Event | null;
  firstNan: Event | null;
  /** Σ p_i in binary64 (NaN if any p is NaN). */
  sumP: number;
  /** Every probability is finite and they sum to 1 within the rounding the format allows. */
  ok: boolean;
}

/** The rounding slack a sum of n probabilities can show: n half-units at 1, plus one for the denominator. */
export const sumTolerance = (f: Format, n: number) => (n + 1) * ulp(f, 1);

export function softmaxTrace(logits: number[], scale: number, f: Format, subtractMax: boolean): SoftmaxTrace {
  const n = logits.length;
  const x = logits.map((v) => round(f, v));
  const z = x.map((v) => mul(f, v, scale));
  let max = -Infinity;
  for (const v of z) if (v > max) max = v;
  const d = z.map((v) => (subtractMax ? sub(f, v, max) : v));
  const t = d.map((v) => round(f, Math.exp(v)));
  const s: number[] = [];
  let acc = 0;
  for (let i = 0; i < n; i++) {
    acc = i === 0 ? t[0] : add(f, acc, t[i]);
    s.push(acc);
  }
  const denom = n ? s[n - 1] : 0;
  const p = t.map((v) => div(f, v, denom));

  // Find the first infinity and the first NaN, in computation order: all z (and the max), then
  // per element d, t, s, then all p.
  let firstInf: Event | null = null;
  let firstNan: Event | null = null;
  const note = (kind: 'inf' | 'nan', index: number, quantity: Quantity, cause: Cause) => {
    if (kind === 'inf' && !firstInf) firstInf = { kind, index, quantity, cause };
    if (kind === 'nan' && !firstNan) firstNan = { kind, index, quantity, cause };
  };
  const isInf = (v: number) => v === Infinity || v === -Infinity;
  z.forEach((v, i) => {
    if (isInf(v)) note('inf', i, 'z', 'scaled-overflow');
  });
  for (let i = 0; i < n; i++) {
    if (Number.isNaN(d[i])) note('nan', i, 'd', isInf(z[i]) && isInf(max) ? 'inf-minus-inf' : 'nan-propagated');
    if (t[i] === Infinity) note('inf', i, 't', 'exp-overflow');
    if (Number.isNaN(t[i])) note('nan', i, 't', 'nan-propagated');
    if (s[i] === Infinity && Number.isFinite(t[i]) && (i === 0 || Number.isFinite(s[i - 1]))) note('inf', i, 's', 'sum-overflow');
    if (Number.isNaN(s[i])) note('nan', i, 's', 'nan-propagated');
  }
  for (let i = 0; i < n; i++)
    if (Number.isNaN(p[i])) note('nan', i, 'p', t[i] === Infinity && denom === Infinity ? 'inf-over-inf' : 'nan-propagated');

  const rows: SoftmaxElement[] = x.map((xi, i) => ({
    x: xi,
    z: z[i],
    d: d[i],
    t: t[i],
    s: s[i],
    p: p[i],
    underflow: t[i] === 0 && Math.exp(d[i]) > 0,
  }));
  const sumP = p.reduce((a, b) => a + b, 0);
  const ok = p.every((v) => Number.isFinite(v)) && Math.abs(sumP - 1) <= sumTolerance(f, n);
  return { f, scale, subtractMax, rows, max, denom, firstInf, firstNan, sumP, ok };
}

/** The explorer's rows. */
export const SOFTMAX_PRESETS: { id: string; label: string; logits: number[] }[] = [
  {
    id: 'typical',
    get label() {
      return tr('a typical row', 'một hàng điển hình');
    },
    logits: [3.2, 1.5, -0.8, 7.1, 2.4, -3, 0.6, 5.5],
  },
  {
    id: 'large',
    get label() {
      return tr('large logits', 'logits lớn');
    },
    logits: [12, 30, 25, -4, 18, 9, 31.5, 2],
  },
  {
    id: 'huge',
    get label() {
      return tr('huge logits', 'logits rất lớn');
    },
    logits: [95, 120, 40, -10, 60, 118, 0, 75],
  },
  {
    id: 'flat',
    get label() {
      return tr('eight equal logits of 10', 'tám logits bằng nhau, đều là 10');
    },
    logits: [10, 10, 10, 10, 10, 10, 10, 10],
  },
  {
    id: 'wide',
    get label() {
      return tr('one very negative', 'một giá trị rất âm');
    },
    logits: [6, 2, -60, 4, 1, -20, 3, 5],
  },
];
