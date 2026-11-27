// One floating-point addition in detail, for appendix A: section A.4's stepper ("Alignment
// and rounding"), its scene ("The disappearing addend") and its FP32-to-BF16 explorer, and
// section A.3's "What comes out?" table and bit-pattern classifier.
//
// Sources: the alignment procedure and the book's worked example 2.0 + 0.25 in the 5-bit
// format (book §A.4, pp. 591–592; notes/appA.md); the four IEEE rounding modes (NVIDIA guide
// 5.6.1.7, appA-facts A9); the special values (book §A.3, pp. 590–591; NVIDIA 5.6.1.3 and
// 5.6.3, appA-facts A7); fmaxf's NaN rule (ch10-facts A10); llama.cpp's FP32-to-BF16
// conversion (ggml-impl.h, appA-facts D3). Formats, rounding and arithmetic come from
// ./minifloat, which every number here is checked against.
//
// What it models: the exact sum of two values of one format, computed with BigInt at the
// smaller operand's scale, shown the way an adder works on it (align, add, normalise,
// round) with the bits that leave the window kept as guard, round and sticky bits. The
// rounded result is tested equal to minifloat.add in all four modes.
//
// What it leaves out: fused multiply-add (one rounding of a·b + c), any particular
// hardware's guard width or pipeline (an adder needs only guard, round and sticky to round
// correctly, which the tests check), flush-to-zero of inputs or results, and NaN payloads.
//
// Erasable TypeScript only.
import {
  BF16,
  E2M1,
  E4M3,
  E5M2,
  FP16,
  FP32,
  TOY5,
  add,
  binade,
  bitsOfValue,
  decode,
  div,
  fields,
  lowExp,
  maxFinite,
  minNormal,
  minSubnormal,
  mul,
  precisionBits,
  round,
  totalBits,
  ulp,
  type Cls,
  type Format,
  type Mode,
} from './minifloat';
import { tr } from '../lib/locale.ts';

export const MODE_LABEL: Record<Mode, string> = {
  get rne() {
    return tr('to nearest, ties to even', 'về gần nhất, hòa thì về chẵn');
  },
  get rtz() {
    return tr('toward zero', 'về phía 0');
  },
  get up() {
    return tr('up (toward +∞)', 'lên (về phía +∞)');
  },
  get down() {
    return tr('down (toward −∞)', 'xuống (về phía −∞)');
  },
};
export const MODES: Mode[] = ['rne', 'rtz', 'up', 'down'];

const big = (n: number) => BigInt(n);
const pow2n = (k: number) => 1n << BigInt(k);
const bin = (v: bigint, width: number) => (width <= 0 ? '' : v.toString(2).padStart(width, '0'));

/** An operand in normalised form: |value| = sig × 2^(exp - (p - 1)), sig a p-bit integer whose top bit is 1. */
export interface Operand {
  value: number;
  sign: 0 | 1;
  /** floor(log2 |value|): the true exponent, below the format's smallest normal one for a subnormal. */
  exp: number;
  /** The p significant bits, leading 1 first. */
  sig: string;
  stored: 'normal' | 'subnormal';
}

function operand(f: Format, v: number): Operand {
  const p = precisionBits(f);
  const a = Math.abs(v);
  const exp = binade(a);
  const sig = big(a / 2 ** (exp - (p - 1)));
  return {
    value: v,
    sign: v < 0 || Object.is(v, -0) ? 1 : 0,
    exp,
    sig: bin(sig, p),
    stored: a < minNormal(f) ? 'subnormal' : 'normal',
  };
}

const orBits = (s: string) => (s.includes('1') ? 1 : 0);
const grs = (tail: string) => ({ g: tail[0] === '1' ? 1 : 0, r: tail[1] === '1' ? 1 : 0, s: orBits(tail.slice(2)) });

export type RoundReason =
  | 'exact' // nothing left the window
  | 'below-half' // nearest: less than half a unit was cut off
  | 'above-half' // nearest: more than half
  | 'tie-even' // nearest: exactly half, the kept last bit is 0, keep it
  | 'tie-odd' // nearest: exactly half, the kept last bit is 1, round up to make it even
  | 'truncate' // toward zero, or a directed mode pointing at zero
  | 'away'; // a directed mode pointing away from zero, with something cut off

export type Step =
  | { kind: 'compare'; big: Operand; small: Operand; diff: number; subtract: boolean; swapped: boolean }
  | {
      kind: 'shift';
      /** Places shifted so far. */
      k: number;
      diff: number;
      /** The smaller operand's exponent after k shifts. */
      exp: number;
      /** The p bits still inside the window. */
      window: string;
      /** All bits that have left the window, most significant first. */
      tail: string;
      g: number;
      r: number;
      s: number;
      /** True for the one step that jumps over shifts which only feed the sticky bit. */
      jump: boolean;
    }
  | {
      kind: 'add';
      subtract: boolean;
      /** The exact sum's bits at the larger operand's scale: p bits, or p + 1 with a carry. */
      window: string;
      /** The exact sum's bits below the window (diff bits). */
      tail: string;
      carry: boolean;
      exact: number;
      /** False when the exact sum has more than 53 significant bits (exact is then the nearest double). */
      exactIsDouble: boolean;
      exactText: string;
    }
  | {
      kind: 'normalise';
      /** The result's exponent. */
      exp: number;
      /** How far the result's exponent moved from the larger operand's: +1 after a carry, negative after cancellation. */
      moved: number;
      /** The result is below the smallest normal value and stays subnormal. */
      subnormal: boolean;
      window: string;
      tail: string;
      g: number;
      r: number;
      s: number;
    }
  | {
      kind: 'round';
      mode: Mode;
      sign: 0 | 1;
      /** Signed candidates: the window as it stands, and one unit further from zero. */
      below: number;
      above: number;
      /** The window's last bit. */
      last: number;
      g: number;
      r: number;
      s: number;
      reason: RoundReason;
      /** Which candidate the mode picks (before any overflow). */
      pick: 'below' | 'above';
      /** above's last bit carried into the exponent (the window was all ones). */
      carryOut: boolean;
      /** The value stored: the pick, or what the format does on overflow. */
      result: number;
      overflow: boolean;
      /** |exact − pick| in units of the result's last place (Infinity after overflow to infinity). */
      errorUlps: number;
      /** Exactly below one unit (errorUlps can round to 1 in binary64 when it is 1 − 2^-100). */
      errorBelowOne: boolean;
      exactText: string;
    }
  | { kind: 'zero-operand'; result: number }
  | { kind: 'exact-zero'; result: number }
  | { kind: 'special'; result: number; why: SpecialWhy };

export type SpecialWhy = 'nan-in' | 'inf-minus-inf' | 'inf-in';

export interface AlignTrace {
  f: Format;
  mode: Mode;
  /** The operands as stored (typed inputs are first rounded to nearest). */
  a: number;
  b: number;
  p: number;
  steps: Step[];
  result: number;
  errorUlps: number;
}

/**
 * The exact decimal expansion of n × 2^k. Longer than `digits` significant digits, a fraction
 * is cut with an ellipsis and a long integer is given approximately in exponent form.
 */
export function exactDecimal(n: bigint, k: number, digits = 26): string {
  const sign = n < 0n ? '−' : '';
  let m = n < 0n ? -n : n;
  let s: string;
  if (m === 0n) s = '0';
  else if (k >= 0) s = (m << BigInt(k)).toString();
  else {
    const q = -k;
    m *= 5n ** BigInt(q); // n × 2^-q = n × 5^q / 10^q
    const t = m.toString().padStart(q + 1, '0');
    s = (t.slice(0, t.length - q) + '.' + t.slice(t.length - q)).replace(/0+$/, '').replace(/\.$/, '');
  }
  const sig = s.replace('.', '').replace(/^0+/, '');
  if (sig.length <= digits) return sign + s;
  const intLen = s.includes('.') ? s.indexOf('.') : s.length;
  if (intLen > digits) return sign + '≈ ' + (Number(n < 0n ? -n : n) * 2 ** k).toExponential(15);
  let count = 0;
  let i = s.search(/[1-9]/);
  for (; i < s.length; i++) {
    if (s[i] === '.') continue;
    if (++count === digits) break;
  }
  return sign + s.slice(0, i + 1) + '…';
}

/**
 * Trace one addition a + b in format f, in rounding mode `mode`. Typed inputs are first
 * rounded to the format (to nearest); the trace then follows the stored operands.
 */
export function alignTrace(f: Format, a0: number, b0: number, mode: Mode = 'rne'): AlignTrace {
  const p = precisionBits(f);
  const a = round(f, a0);
  const b = round(f, b0);
  const result = add(f, a, b, mode);
  const base = { f, mode, a, b, p };
  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    const why: SpecialWhy = Number.isNaN(a) || Number.isNaN(b) ? 'nan-in' : Number.isNaN(a + b) ? 'inf-minus-inf' : 'inf-in';
    return { ...base, steps: [{ kind: 'special', result, why }], result, errorUlps: Number.isNaN(result) ? NaN : 0 };
  }
  if (a === 0 || b === 0) return { ...base, steps: [{ kind: 'zero-operand', result }], result, errorUlps: 0 };

  let A = operand(f, a);
  let B = operand(f, b);
  const swapped = B.exp > A.exp || (B.exp === A.exp && B.sig > A.sig);
  if (swapped) [A, B] = [B, A];
  const d = A.exp - B.exp;
  const subtract = A.sign !== B.sign;
  const Sb = BigInt('0b' + A.sig);
  const Ss = BigInt('0b' + B.sig);
  const steps: Step[] = [{ kind: 'compare', big: A, small: B, diff: d, subtract, swapped }];

  // Shift the smaller significand right one place at a time. After p + 2 places every one of
  // its bits has passed the guard and round positions, and further shifts only feed sticky.
  const cap = Math.min(d, p + 2);
  const shiftStep = (k: number, jump: boolean): Step => {
    const tail = bin(Ss % pow2n(k), k);
    return { kind: 'shift', k, diff: d, exp: B.exp + k, window: bin(Ss >> BigInt(k), p), tail, ...grs(tail), jump };
  };
  for (let k = 1; k <= cap; k++) steps.push(shiftStep(k, false));
  if (d > cap) steps.push(shiftStep(d, true));

  // The exact sum, in units of the smaller operand's last place: u0 = 2^(B.exp - (p - 1)).
  const Abig = Sb << BigInt(d);
  const Sum = subtract ? Abig - Ss : Abig + Ss;
  const u0 = B.exp - (p - 1);
  const sign: 0 | 1 = A.sign;
  const signed = (n: bigint) => (sign ? -n : n);
  const exactNum = (sign ? -1 : 1) * Number(Sum) * 2 ** u0;
  const exactText = exactDecimal(signed(Sum), u0);
  const hi = Sum >> BigInt(d);
  steps.push({
    kind: 'add',
    subtract,
    window: bin(hi, p),
    tail: bin(Sum % pow2n(d), d),
    carry: hi >= pow2n(p),
    exact: exactNum,
    exactIsDouble: Sum < pow2n(53),
    exactText,
  });
  if (Sum === 0n) {
    steps.push({ kind: 'exact-zero', result });
    return { ...base, steps, result, errorUlps: 0 };
  }

  // Normalise: the leading 1 goes to the hidden-bit position, unless the result is subnormal.
  const nb = Sum.toString(2).length;
  let Er = nb - 1 + u0;
  const L = lowExp(f);
  const subnormal = Er < L;
  if (subnormal) Er = L;
  const sh = Er - B.exp; // places the window sits above u0's scale
  const W = sh >= 0 ? Sum >> BigInt(sh) : Sum << BigInt(-sh);
  const tail = sh > 0 ? bin(Sum % pow2n(sh), sh) : '';
  const t = grs(tail);
  steps.push({ kind: 'normalise', exp: Er, moved: Er - A.exp, subnormal, window: bin(W, p), tail, ...t });

  // Round.
  const last = Number(W & 1n);
  const inexact = !!(t.g || t.r || t.s);
  let up: boolean;
  let reason: RoundReason;
  if (!inexact) {
    up = false;
    reason = 'exact';
  } else if (mode === 'rne') {
    if (!t.g) [up, reason] = [false, 'below-half'];
    else if (t.r || t.s) [up, reason] = [true, 'above-half'];
    else if (last) [up, reason] = [true, 'tie-odd'];
    else [up, reason] = [false, 'tie-even'];
  } else {
    const away = (mode === 'up' && sign === 0) || (mode === 'down' && sign === 1);
    [up, reason] = away ? [true, 'away'] : [false, 'truncate'];
  }
  const unit = Er - (p - 1);
  const belowMag = Number(W) * 2 ** unit;
  const aboveMag = Number(W + 1n) * 2 ** unit;
  const pickMag = up ? aboveMag : belowMag;
  // The format decides what an out-of-range pick becomes (infinity, the largest value, NaN).
  const stored = round(f, sign ? -pickMag : pickMag, mode);
  const overflow = pickMag > maxFinite(f);
  const chosenW = up ? W + 1n : W;
  const err = sh >= 0 ? Sum - (chosenW << BigInt(sh)) : 0n;
  const errUnits = err < 0n ? -err : err;
  const errorUlps = overflow && !Number.isFinite(stored) ? Infinity : sh > 0 ? Number(errUnits) / Number(pow2n(sh)) : 0;
  steps.push({
    kind: 'round',
    mode,
    sign,
    below: sign ? -belowMag : belowMag,
    above: sign ? -aboveMag : aboveMag,
    last,
    ...t,
    reason,
    pick: up ? 'above' : 'below',
    carryOut: up && W + 1n === pow2n(p),
    result: stored,
    overflow,
    errorUlps,
    errorBelowOne: sh <= 0 || errUnits < pow2n(sh),
    exactText,
  });
  return { ...base, steps, result: stored, errorUlps };
}

/** The stepper's presets, with the text the operand fields show. */
export const ALIGN_PRESETS: { id: string; label: string; fmt: Format; a: number; b: number; aText: string; bText: string }[] = [
  { id: 'book', get label() { return tr("the book's 2 + 0.25 (5-bit format)", '2 + 0.25 của sách (định dạng 5 bit)'); }, fmt: TOY5, a: 2, b: 0.25, aText: '2', bText: '0.25' },
  { id: 'half-ulp', get label() { return tr('1 + 2⁻¹¹ in FP16', '1 + 2⁻¹¹ trong FP16'); }, fmt: FP16, a: 1, b: 2 ** -11, aText: '1', bText: '2^-11' },
  { id: 'stall', get label() { return tr('2048 + 1 in FP16', '2048 + 1 trong FP16'); }, fmt: FP16, a: 2048, b: 1, aText: '2048', bText: '1' },
  { id: 'probe', get label() { return tr('1 + 0.75 ULP in FP32', '1 + 0.75 ULP trong FP32'); }, fmt: FP32, a: 1, b: 0.75 * 2 ** -23, aText: '1', bText: '0.75×2^-23' },
  { id: 'cancel', get label() { return tr('1.5 − 1.25 in FP16', '1.5 − 1.25 trong FP16'); }, fmt: FP16, a: 1.5, b: -1.25, aText: '1.5', bText: '-1.25' },
  { id: 'addend', get label() { return tr('1000 + 0.3 in FP16', '1000 + 0.3 trong FP16'); }, fmt: FP16, a: 1000, b: 0.3, aText: '1000', bText: '0.3' },
];

/** An operand as typed: a decimal number, a power of two written 2^-11, or a multiple of one, 0.75×2^-23. */
export function parseOperand(t: string): number | null {
  const s = t.trim().replace(/\s+/g, '').replace(/−/g, '-');
  let m = /^(-?)2\^(-?\d+)$/.exec(s);
  if (m) return (m[1] ? -1 : 1) * 2 ** Number(m[2]);
  m = /^(-?[\d.]+)[*×x]2\^(-?\d+)$/.exec(s);
  if (m && Number.isFinite(Number(m[1]))) return Number(m[1]) * 2 ** Number(m[2]);
  if (!/^-?(\d+\.?\d*|\.\d+)(e-?\d+)?$/i.test(s)) return null;
  const v = Number(s);
  return Number.isFinite(v) ? v : null;
}

// ---------------------------------------------------------------- a long loop of one addition
export interface RunningSum {
  f: Format;
  /** The addend as stored in the format. */
  addend: number;
  /** acc[k]: the accumulator after k additions (acc[0] = start, stored). */
  acc: number[];
  /** exact[k] = start + k × addend in real arithmetic (exact in binary64 for these sizes). */
  exact: number[];
  /** The first k at which an addition left the accumulator unchanged, or null. */
  stallAt: number | null;
}

/** acc = start; then n times acc = acc + addend, every addition rounded in format f. */
export function runningSum(f: Format, start: number, addend: number, n: number, mode: Mode = 'rne'): RunningSum {
  const s0 = round(f, start);
  const x = round(f, addend);
  const acc = [s0];
  const exact = [s0];
  let stallAt: number | null = null;
  for (let k = 1; k <= n; k++) {
    const next = add(f, acc[k - 1], x, mode);
    if (stallAt === null && next === acc[k - 1]) stallAt = k;
    acc.push(next);
    exact.push(s0 + k * x);
  }
  return { f, addend: x, acc, exact, stallAt };
}

// ---------------------------------------------------------------- FP32 to BF16 on the bits
const f32buf = new DataView(new ArrayBuffer(4));
export function f32FromBits(u: number): number {
  f32buf.setUint32(0, u >>> 0);
  return f32buf.getFloat32(0);
}
export function bitsFromF32(x: number): number {
  f32buf.setFloat32(0, x);
  return f32buf.getUint32(0);
}
/** The value of a BF16 pattern: the top half of an FP32. */
export const bf16Value = (h: number) => f32FromBits(((h & 0xffff) << 16) >>> 0);

export interface Bf16Conversion {
  input: number;
  /** The FP32 value. */
  x: number;
  nan: boolean;
  /** The 16 bits kept (u >> 16) and the 16 cut off. */
  kept: number;
  cut: number;
  /** The kept half's last bit: what ties are broken on. */
  lsb: number;
  /** 0x7fff + lsb. */
  increment: number;
  /** u + increment, as a 33-bit-safe number (never above 2^32 for a non-NaN input). */
  sum: number;
  /** Did adding the increment change the kept half? */
  roundedUp: boolean;
  /** Did the rounding carry into the exponent field? */
  carryIntoExponent: boolean;
  /** llama.cpp's result (D3): round to nearest even on the bits, NaN forced quiet. */
  rounded: number;
  /** Plain truncation: the top 16 bits. */
  truncated: number;
  roundedValue: number;
  truncatedValue: number;
  /** |x − result| in BF16 ULPs at x (Infinity when the result is infinite and x is not). */
  roundedErr: number;
  truncatedErr: number;
  /** The cut half is exactly 0x8000: a tie. */
  tie: boolean;
}

/** FP32 to BF16 as llama.cpp's ggml_compute_fp32_to_bf16 does it (appA-facts D3), with truncation beside it. */
export function bf16FromBits(u0: number): Bf16Conversion {
  const u = u0 >>> 0;
  const x = f32FromBits(u);
  const nan = (u & 0x7fffffff) > 0x7f800000;
  const kept = u >>> 16;
  const cut = u & 0xffff;
  const lsb = kept & 1;
  const increment = 0x7fff + lsb;
  const sum = u + increment;
  const rounded = nan ? (kept | 64) & 0xffff : Math.floor(sum / 65536) & 0xffff;
  const truncated = kept;
  const roundedValue = bf16Value(rounded);
  const truncatedValue = bf16Value(truncated);
  const errIn = (r: number) => {
    if (nan) return NaN;
    if (!Number.isFinite(x)) return r === x ? 0 : Infinity;
    if (!Number.isFinite(r)) return Infinity;
    return Math.abs(x - r) / ulp(BF16, x);
  };
  const expOf = (h: number) => (h >>> 7) & 0xff;
  return {
    input: u,
    x,
    nan,
    kept,
    cut,
    lsb,
    increment,
    sum,
    roundedUp: !nan && rounded !== kept,
    carryIntoExponent: !nan && expOf(rounded) !== expOf(kept),
    rounded,
    truncated,
    roundedValue,
    truncatedValue,
    roundedErr: errIn(roundedValue),
    truncatedErr: errIn(truncatedValue),
    tie: !nan && cut === 0x8000,
  };
}

export const BF16_PRESETS: { id: string; label: string; bits: number }[] = [
  { id: 'pi', label: 'π', bits: bitsFromF32(Math.PI) },
  { id: 'tie-even', get label() { return tr('a tie, kept bit even: 1 + 2⁻⁸', 'trường hợp hòa, bit giữ lại chẵn: 1 + 2⁻⁸'); }, bits: 0x3f808000 },
  { id: 'tie-odd', get label() { return tr('a tie, kept bit odd: 1 + 3·2⁻⁸', 'trường hợp hòa, bit giữ lại lẻ: 1 + 3·2⁻⁸'); }, bits: 0x3f818000 },
  { id: 'carry', get label() { return tr('a carry into the exponent: just under 2', 'nhớ sang số mũ: ngay dưới 2'); }, bits: 0x3fffffff },
  { id: 'max', get label() { return tr('the largest FP32', 'số FP32 lớn nhất'); }, bits: 0x7f7fffff },
  { id: 'third', label: '−1/3', bits: bitsFromF32(-1 / 3) },
  { id: 'snan', get label() { return tr('a NaN with its payload in the low bits', 'một NaN có payload ở các bit thấp'); }, bits: 0x7f800001 },
];

// ---------------------------------------------------------------- A.3: what comes out?
export type Outcome = 'zero' | 'finite' | '+inf' | '-inf' | 'nan' | 'true' | 'false';
export const OUTCOME_LABEL: Record<Outcome, string> = {
  zero: '0',
  get finite() {
    return tr('a non-zero number', 'một số khác 0');
  },
  '+inf': '+∞',
  '-inf': '−∞',
  nan: 'NaN',
  true: 'true',
  false: 'false',
};
export const OUTCOMES: Outcome[] = ['zero', 'finite', '+inf', '-inf', 'nan', 'true', 'false'];

export function outcomeOf(v: number | boolean): Outcome {
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (Number.isNaN(v)) return 'nan';
  if (v === Infinity) return '+inf';
  if (v === -Infinity) return '-inf';
  return v === 0 ? 'zero' : 'finite';
}

/**
 * CUDA's fmaxf, by its documented rule (ch10-facts A10): "If one argument is a NaN and the
 * other is legitimate numeric value, the numeric value is chosen." "If both arguments are
 * NaN, returns NaN." Metal's fmax states the same rule (ch10-facts C3).
 */
export function fmaxf(a: number, b: number): number {
  if (Number.isNaN(a)) return b;
  if (Number.isNaN(b)) return a;
  return a > b ? a : b;
}

export interface SpecialRow {
  id: string;
  /** The expression as shown. */
  expr: string;
  /** The format it is evaluated in. */
  fmt: Format;
  value: number | boolean;
  outcome: Outcome;
  /** Why, in a sentence. */
  rule: string;
  source: string;
}

/** The A.3 table, every row evaluated through ./minifloat (FP32 unless a row says otherwise). */
export function specialTable(): SpecialRow[] {
  const F = FP32;
  const INF = decode(F, 0x7f800000).value;
  const NAN = decode(F, 0x7fc00000).value;
  const MAX = maxFinite(F);
  const rows: Omit<SpecialRow, 'outcome'>[] = [
    { id: 'zero-zero', expr: '0 / 0', fmt: F, value: div(F, 0, 0), rule: tr('Mathematically undefined, so the result is NaN.', 'Không xác định về mặt toán học, nên kết quả là NaN.'), source: tr('book §A.3, p. 590', 'sách §A.3, tr. 590') },
    { id: 'one-zero', expr: '1 / 0', fmt: F, value: div(F, 1, 0), rule: tr('Division by zero gives infinity with the sign of the quotient; no exception, no trap.', 'Chia cho 0 cho ra vô cực mang dấu của thương; không có ngoại lệ, không có trap.'), source: tr('NVIDIA guide 5.6.1.3 (appA-facts A7)', 'hướng dẫn của NVIDIA 5.6.1.3 (appA-facts A7)') },
    { id: 'minus-one-zero', expr: '−1 / 0', fmt: F, value: div(F, -1, 0), rule: tr('The same, with a negative sign.', 'Tương tự, với dấu âm.'), source: tr('NVIDIA guide 5.6.1.3 (appA-facts A7)', 'hướng dẫn của NVIDIA 5.6.1.3 (appA-facts A7)') },
    { id: 'inf-inf', expr: '∞ − ∞', fmt: F, value: add(F, INF, -INF), rule: tr('Undefined: one of the book\'s four NaN-producing operations.', 'Không xác định: một trong bốn phép toán sinh ra NaN mà sách nêu.'), source: tr('book §A.3, p. 591', 'sách §A.3, tr. 591') },
    { id: 'zero-inf', expr: '0 × ∞', fmt: F, value: mul(F, 0, INF), rule: tr('Undefined: another of the four.', 'Không xác định: một phép khác trong bốn phép đó.'), source: tr('book §A.3, p. 591', 'sách §A.3, tr. 591') },
    { id: 'one-inf', expr: '1 / ∞', fmt: F, value: div(F, 1, INF), rule: tr('Any representable number divided by ±∞ is 0.', 'Mọi số biểu diễn được chia cho ±∞ đều bằng 0.'), source: tr('book §A.3, p. 590', 'sách §A.3, tr. 590') },
    { id: 'one-nan', expr: '1 + NaN', fmt: F, value: add(F, 1, NAN), rule: tr('A quiet NaN goes in, a quiet NaN comes out, silently.', 'Một quiet NaN đi vào, một quiet NaN đi ra, một cách lặng lẽ.'), source: tr('book §A.3, p. 591', 'sách §A.3, tr. 591') }, // i18n-ignore: expression in code notation
    { id: 'nan-eq', expr: 'NaN == NaN', fmt: F, value: NAN === NAN, rule: tr('Every ordered comparison involving a NaN, == included, is false. A NaN does not equal itself.', 'Mọi phép so sánh có thứ tự có mặt NaN, kể cả ==, đều là false. Một NaN không bằng chính nó.'), source: tr('NVIDIA guide 5.6.1.3 (appA-facts A7)', 'hướng dẫn của NVIDIA 5.6.1.3 (appA-facts A7)') },
    { id: 'nan-ne', expr: 'NaN != NaN', fmt: F, value: NAN !== NAN, rule: tr('The unordered comparison is true, so x != x is true exactly for a NaN, unless fast math lets the compiler assume there are none.', 'Phép so sánh không có thứ tự là true, nên x != x là true khi và chỉ khi x là NaN, trừ khi fast math cho phép trình biên dịch giả định rằng không có NaN nào.'), source: tr('NVIDIA guide 5.6.1.3 (appA-facts A7)', 'hướng dẫn của NVIDIA 5.6.1.3 (appA-facts A7)') },
    { id: 'max-two', expr: 'FLT_MAX × 2', fmt: F, value: mul(F, MAX, 2), rule: tr('Overflow saturates to infinity; integer arithmetic would wrap or be undefined instead.', 'Tràn số bão hòa thành vô cực; số học số nguyên thì lại quay vòng hoặc không xác định.'), source: tr('book §A.3, p. 590', 'sách §A.3, tr. 590') }, // i18n-ignore: expression in code notation
    { id: 'fp16-max', expr: tr('65504 + 32 in FP16', '65504 + 32 trong FP16'), fmt: FP16, value: add(FP16, 65504, 32), rule: tr('The exact 65536 lies past 65504 plus half a step (65520), so it rounds to infinity.', 'Giá trị chính xác 65536 nằm quá 65504 cộng nửa bước (65520), nên nó được làm tròn thành vô cực.'), source: tr('FP16 maximum from NVIDIA Table 51 (appA-facts A2); the rounding is ours', 'giá trị lớn nhất của FP16 lấy từ Bảng 51 của NVIDIA (appA-facts A2); phần làm tròn là của khóa học này') },
    { id: 'fmaxf', expr: 'fmaxf(NaN, 1)', fmt: F, value: fmaxf(NAN, 1), rule: tr('fmaxf treats a NaN as missing data and returns the other argument, so a max reduction drops NaNs.', 'fmaxf coi NaN là dữ liệu thiếu và trả về đối số còn lại, nên một phép rút gọn max bỏ qua các NaN.'), source: 'CUDA Math API fmaxf (ch10-facts A10)' }, // i18n-ignore: citation (API name and fact id)
  ];
  return rows.map((r) => ({ ...r, outcome: outcomeOf(r.value) }));
}

// ---------------------------------------------------------------- A.3: the bit-pattern classifier
/** The formats the classifier offers. */
export const CLASSIFIER_FORMATS: Format[] = [FP32, FP16, BF16, E4M3, E5M2, E2M1];

/** Which row of the book's Fig. A.8 a pattern's fields match, by the IEEE rule alone; null for an ordinary exponent. */
export function figA8Row(f: Format, bits: number): 'nan' | 'inf' | 'denorm' | 'zero' | null {
  const { E, M } = fields(f, bits);
  const ones = 2 ** f.e - 1;
  if (E === ones) return M === 0 ? 'inf' : 'nan';
  if (E === 0) return M === 0 ? 'zero' : 'denorm';
  return null;
}

/** Does Fig. A.8's rule agree with what the format says the pattern is? */
export function figA8Agrees(f: Format, bits: number): boolean {
  const row = figA8Row(f, bits);
  const cls = decode(f, bits).cls;
  if (row === null) return cls === 'normal';
  if (row === 'nan') return cls === 'qnan' || cls === 'snan' || cls === 'nan';
  if (row === 'inf') return cls === 'inf';
  if (row === 'denorm') return cls === 'subnormal';
  return cls === 'zero';
}

export interface PatternPreset {
  id: string;
  label: string;
  /** The pattern, or null when the format has none; then `reason` says why. */
  bits: number | null;
  reason?: string;
}

export function classifierPresets(f: Format): PatternPreset[] {
  const has = (v: number) => bitsOfValue(f, v);
  const ieee = f.specials === 'ieee';
  const split = ieee && !!f.quietBit;
  const ones = 2 ** f.e - 1;
  const sub = minSubnormal(f);
  const out: PatternPreset[] = [
    { id: 'pzero', label: '+0', bits: has(0) },
    { id: 'nzero', label: '−0', bits: f.signed && f.specials !== 'fnuz' ? has(-0) : null, reason: tr(`${f.name} has no −0`, `${f.name} không có −0`) },
    { id: 'min-sub', label: tr('smallest denormal', 'số denormal nhỏ nhất'), bits: sub === null ? null : has(sub), reason: tr(`${f.name} has no denormals`, `${f.name} không có số denormal`) },
    { id: 'max-sub', label: tr('largest denormal', 'số denormal lớn nhất'), bits: sub === null ? null : has(minNormal(f) - sub), reason: tr(`${f.name} has no denormals`, `${f.name} không có số denormal`) },
    { id: 'min-normal', label: tr('smallest normal', 'số normal nhỏ nhất'), bits: has(minNormal(f)) },
    { id: 'one', label: '1', bits: has(1) },
    { id: 'max', label: tr('largest finite', 'số hữu hạn lớn nhất'), bits: has(maxFinite(f)) },
    {
      id: 'inf',
      label: '+∞',
      bits: ieee ? has(Infinity) : null,
      reason:
        f.specials === 'fn'
          ? tr(`${f.name} has no infinity: its top exponent code holds numbers`, `${f.name} không có vô cực: mã số mũ cao nhất của nó chứa các số`)
          : tr(`${f.name} has neither infinity nor NaN`, `${f.name} không có cả vô cực lẫn NaN`),
    },
    {
      id: 'nan',
      label: split ? tr('quiet NaN', 'quiet NaN') : 'NaN',
      bits: f.specials === 'none' ? null : has(NaN),
      reason: tr(`${f.name} has neither infinity nor NaN`, `${f.name} không có cả vô cực lẫn NaN`),
    },
    {
      id: 'snan',
      label: tr('signalling NaN', 'signalling NaN'),
      bits: split ? ones * 2 ** f.m + 1 : null,
      reason:
        f.specials === 'none'
          ? tr(`${f.name} has no NaN`, `${f.name} không có NaN`)
          : f.specials === 'fn'
            ? tr(
                `${f.name} has one NaN pattern per sign, so there is nothing to split into quiet and signalling`,
                `${f.name} chỉ có một mẫu NaN cho mỗi dấu, nên không có gì để chia thành quiet và signalling`,
              )
            : tr(
                `OCP's specification does not say which of ${f.name}'s NaN patterns are quiet`,
                `Đặc tả của OCP không nói mẫu NaN nào của ${f.name} là quiet`,
              ),
    },
  ];
  return out.map((p) => (p.bits !== null && p.bits >= 0 ? { ...p, reason: undefined } : { ...p, bits: null }));
}

export type CountKey = 'zero' | 'subnormal' | 'normal' | 'inf' | 'qnan' | 'snan' | 'nan';

/** How many of the format's 2^n bit patterns fall in each class. */
export function patternCounts(f: Format): Record<CountKey, number> & { total: number } {
  const c: Record<CountKey, number> = { zero: 0, subnormal: 0, normal: 0, inf: 0, qnan: 0, snan: 0, nan: 0 };
  const bump = (cls: Cls, n: number) => {
    if (cls !== 'reserved') c[cls] += n;
  };
  const n = totalBits(f);
  if (n <= 16) {
    for (let b = 0; b < 2 ** n; b++) bump(decode(f, b).cls, 1);
  } else {
    // IEEE layouts only: the class depends on the sign, the exponent code, whether M is 0,
    // and M's top bit. One representative pattern stands for each group.
    if (f.specials !== 'ieee') throw new Error('patternCounts: wide formats must be IEEE-like');
    const half = 2 ** (f.m - 1);
    for (let s = 0; s < (f.signed ? 2 : 1); s++)
      for (let E = 0; E < 2 ** f.e; E++) {
        const base = s * 2 ** (f.e + f.m) + E * 2 ** f.m;
        bump(decode(f, base).cls, 1); // M = 0
        bump(decode(f, base + half).cls, half); // top mantissa bit set
        bump(decode(f, base + 1).cls, half - 1); // top bit clear, M not 0
      }
  }
  return { ...c, total: 2 ** n };
}
