// One model of binary floating-point formats for the whole of appendix A. Group 1 owns this
// file; every other section imports the API below and does not edit it.
//
// Sources: the book's formulas (notes/appA.md, §A.1 to §A.3: Eq. A.1, the excess bias
// 2^(e-1) - 1, the denormal rule 0.M · 2^(-2^(n-1) + 2), the reserved all-ones exponent);
// FP32, FP16 and BF16 from notes/today/appA-facts.md A2 (NVIDIA guide 5.6.1.1, Tables 50
// and 51; OFP8 §4.1; Clang; PyTorch); E4M3 and E5M2 from OFP8 r1.1 Tables 1 and 2 (A3);
// E2M1, E2M3, E3M2 and UE8M0 from MX v1.1 Tables 4, 5 and 7 (A4); E4M3FNUZ from AMD (B2).
//
// What it models, exactly:
//  - A format is data: sign bit or not, e exponent bits, m stored mantissa bits, a bias, how
//    the top exponent code is used (`specials`) and what the all-zeros exponent code means
//    (`subnormals`: the book's three conventions of Fig. A.4).
//  - Values are JavaScript numbers (binary64). Every value of every format here is exactly a
//    binary64 number, so decode is exact and nothing is approximated.
//  - Rounding (encode, round) is exact for any binary64 input, in four modes: to nearest with
//    ties to even ('rne'), toward zero ('rtz'), toward +infinity ('up'), toward -infinity
//    ('down'). Ties go to the neighbour whose bit pattern is even, which is the usual rule
//    and also defines ties for UE8M0 (no mantissa bits) and the abrupt-underflow gap (where
//    both neighbours are even, the tie goes to zero: our choice; the book gives no rule).
//  - Overflow follows IEEE 754 §7.4 by default (to nearest or away from zero: infinity;
//    toward zero: the largest finite value). Formats with no infinity give NaN instead where
//    IEEE would give infinity (OFP8's non-saturating E4M3), and formats with neither (FP6,
//    FP4, the book's toy formats) always saturate. overflow = 'saturate' clamps to the
//    largest finite value in every format, infinite inputs included (that last part is our
//    choice; the facts do not pin it).
//  - Arithmetic (add, sub, mul, div) rounds the exact result once. mul and div compute in
//    binary64 and round: for formats with p <= 25 significant bits that is correctly rounded
//    in all four modes (a product of two p-bit significands is exact in 53 bits, and a
//    quotient that is not exact stays further than 2^-(2p+1) relative from every candidate,
//    so 53 >= 2p + 2 settles both). add uses the exact TwoSum error term, so it is correctly
//    rounded in every mode even across a large exponent gap (rounding the binary64 sum alone
//    would not be, for directed modes). The tests check all pairs of TOY5, E2M1 and E4M3
//    against exact rationals.
//  - Flush-to-zero (classify's flushDenormals, and flushDenormals on bits) flushes results:
//    round as if denormals existed, then replace a non-zero subnormal result by zero of the
//    same sign. Real hardware may also flush inputs, and vendors document different modes
//    (appA-facts A6, B3, C2); this is one plausible behaviour, not any vendor's.
//
// What it leaves out: binary64 itself as a format (values of FP32 and narrower only), NaN
// payloads beyond quiet or signaling, signaling exceptions, stochastic rounding, block
// scaling (an MX block is a separate calculation over these element formats), and anything
// about time, hardware or memory.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.
import { tr } from '../lib/locale.ts';

export type Specials = 'ieee' | 'fn' | 'fnuz' | 'none' | 'reserved' | 'e8m0';
export type Subnormals = 'denorm' | 'abrupt' | 'nozero';
export type Mode = 'rne' | 'rtz' | 'up' | 'down';
export type Overflow = 'ieee' | 'saturate';
export type Cls = 'zero' | 'subnormal' | 'normal' | 'inf' | 'qnan' | 'snan' | 'nan' | 'reserved';

export interface Format {
  id: string;
  name: string;
  /** Exponent bits. */
  e: number;
  /** Stored mantissa bits (the hidden bit is not counted). */
  m: number;
  bias: number;
  signed: boolean;
  /**
   * How special values are encoded.
   * ieee: top exponent code is infinity (mantissa 0) or NaN. fn: no infinity, NaN only at
   * S.11..1.11..1 (E4M3). fnuz: no infinity, no -0, NaN is the pattern 1000..0. none: every
   * pattern is a finite number (FP6, FP4). reserved: the book's toy formats, whose top
   * exponent code is set aside (p. 585). e8m0: UE8M0, exponent only, NaN at all ones.
   */
  specials: Specials;
  /** The all-zeros exponent code: the book's three conventions (Fig. A.4). */
  subnormals: Subnormals;
  /** ieee only: the top mantissa bit separates quiet from signaling NaNs. */
  quietBit?: boolean;
  /** True when the book does not describe the format. */
  notInBook?: boolean;
  /** Where the definition comes from. */
  source: string;
}

// ---------------------------------------------------------------- presets
export const FP32: Format = {
  id: 'fp32', name: 'FP32', e: 8, m: 23, bias: 127, signed: true, specials: 'ieee', subnormals: 'denorm', quietBit: true,
  get source() { return tr('IEEE 754 binary32: book §A.3, p. 590; NVIDIA guide 5.6.1.1 and Table 51 (appA-facts A2)', 'IEEE 754 binary32: sách §A.3, p. 590; hướng dẫn của NVIDIA 5.6.1.1 và Table 51 (appA-facts A2)'); },
};
export const FP16: Format = {
  id: 'fp16', name: 'FP16', e: 5, m: 10, bias: 15, signed: true, specials: 'ieee', subnormals: 'denorm', quietBit: true, notInBook: true,
  get source() { return tr('IEEE 754 binary16: NVIDIA Tables 50 and 51, Clang, PyTorch 1-5-10 (appA-facts A2)', 'IEEE 754 binary16: NVIDIA Tables 50 và 51, Clang, PyTorch 1-5-10 (appA-facts A2)'); },
};
export const BF16: Format = {
  id: 'bf16', name: 'BF16', e: 8, m: 7, bias: 127, signed: true, specials: 'ieee', subnormals: 'denorm', quietBit: true, notInBook: true,
  get source() { return tr('bfloat16, not an IEEE format: NVIDIA Tables 50 and 51, OFP8 §4.1, PyTorch 1-8-7 (appA-facts A2)', 'bfloat16, không phải định dạng IEEE: NVIDIA Tables 50 và 51, OFP8 §4.1, PyTorch 1-8-7 (appA-facts A2)'); },
};
export const E4M3: Format = {
  id: 'e4m3', name: 'FP8 E4M3', e: 4, m: 3, bias: 7, signed: true, specials: 'fn', subnormals: 'denorm', notInBook: true, // i18n-ignore: format name
  get source() { return tr('OCP OFP8 r1.1 Table 1 (appA-facts A3)', 'OCP OFP8 r1.1, Table 1 (appA-facts A3)'); },
};
export const E5M2: Format = {
  id: 'e5m2', name: 'FP8 E5M2', e: 5, m: 2, bias: 15, signed: true, specials: 'ieee', subnormals: 'denorm', notInBook: true, // i18n-ignore: format name
  get source() { return tr('OCP OFP8 r1.1 Table 2 (appA-facts A3); NaN mantissa meaning "not defined"', 'OCP OFP8 r1.1, Table 2 (appA-facts A3); mantissa của NaN mang nghĩa "không xác định"'); },
};
export const E2M1: Format = {
  id: 'e2m1', name: 'FP4 E2M1', e: 2, m: 1, bias: 1, signed: true, specials: 'none', subnormals: 'denorm', notInBook: true, // i18n-ignore: format name
  get source() { return tr('OCP MX v1.1 Table 5 (appA-facts A4)', 'OCP MX v1.1, Table 5 (appA-facts A4)'); },
};
export const E2M3: Format = {
  id: 'e2m3', name: 'FP6 E2M3', e: 2, m: 3, bias: 1, signed: true, specials: 'none', subnormals: 'denorm', notInBook: true, // i18n-ignore: format name
  get source() { return tr('OCP MX v1.1 Table 4 (appA-facts A4)', 'OCP MX v1.1, Table 4 (appA-facts A4)'); },
};
export const E3M2: Format = {
  id: 'e3m2', name: 'FP6 E3M2', e: 3, m: 2, bias: 3, signed: true, specials: 'none', subnormals: 'denorm', notInBook: true, // i18n-ignore: format name
  get source() { return tr('OCP MX v1.1 Table 4 (appA-facts A4)', 'OCP MX v1.1, Table 4 (appA-facts A4)'); },
};
export const UE8M0: Format = {
  id: 'ue8m0', name: 'UE8M0', e: 8, m: 0, bias: 127, signed: false, specials: 'e8m0', subnormals: 'nozero', notInBook: true,
  get source() { return tr('OCP MX v1.1 Table 7 (appA-facts A4): bias 127, exponents -127 to 127, NaN 11111111, no zero', 'OCP MX v1.1, Table 7 (appA-facts A4): bias 127, số mũ từ -127 đến 127, NaN 11111111, không có số 0'); },
};
export const E4M3FNUZ: Format = {
  id: 'e4m3fnuz', name: 'FP8 E4M3 FNUZ', e: 4, m: 3, bias: 8, signed: true, specials: 'fnuz', subnormals: 'denorm', notInBook: true, // i18n-ignore: format name
  get source() { return tr("AMD (appA-facts B2): bias 8 from the header amd_hip_fp8.h, not stated in the docs; no infinity, no -0, NaN 0x80 from the docs", 'AMD (appA-facts B2): bias 8 lấy từ header amd_hip_fp8.h, tài liệu không nêu; không có vô cực, không có -0, NaN 0x80 theo tài liệu'); },
};
/** The book's 5-bit format of §A.2 (1-2-2, excess-1), with IEEE's choice of denormals. */
export const TOY5: Format = {
  id: 'toy5', get name() { return tr("The book's 5-bit format (1-2-2)", 'Định dạng 5 bit của sách (1-2-2)'); }, e: 2, m: 2, bias: 1, signed: true, specials: 'reserved', subnormals: 'denorm',
  get source() { return tr('book §A.2, p. 586, Fig. A.4 (denorm columns)', 'sách §A.2, p. 586, Fig. A.4 (các cột denorm)'); },
};
/**
 * The book's 6-bit format of §A.1 (1-3-2, excess-3). §A.1 reads every pattern with Eq. (A.1),
 * E = 000 included, so this is the no-zero convention: denormals arrive only in §A.2, and only
 * for the 5-bit format. Smallest positive value 1/8, as in Exercise 2's answer.
 */
export const TOY6: Format = {
  id: 'toy6', get name() { return tr("The book's 6-bit format (1-3-2)", 'Định dạng 6 bit của sách (1-3-2)'); }, e: 3, m: 2, bias: 3, signed: true, specials: 'reserved', subnormals: 'nozero',
  get source() { return tr('book §A.1, pp. 584–586 (Eq. A.1 for every pattern, so no zero)', 'sách §A.1, pp. 584–586 (Eq. A.1 cho mọi mẫu bit, nên không có số 0)'); },
};

export const PRESETS: Format[] = [TOY6, TOY5, FP32, FP16, BF16, E4M3, E5M2, E2M1, E2M3, E3M2, UE8M0, E4M3FNUZ];
export const preset = (id: string): Format => {
  const f = PRESETS.find((p) => p.id === id);
  if (!f) throw new Error('no format ' + id);
  return f;
};

export const excessBias = (e: number) => 2 ** (e - 1) - 1;

/** A book-style toy format (sign, e exponent bits in excess-(2^(e-1)-1), m mantissa bits, top code reserved). */
export function makeFormat(e: number, m: number, subnormals: Subnormals): Format {
  return {
    id: `toy-${e}-${m}-${subnormals}`,
    name: `1-${e}-${m}`,
    e,
    m,
    bias: excessBias(e),
    signed: true,
    specials: 'reserved',
    subnormals,
    source: tr('book §A.1 and §A.2 (pp. 584–589): excess coding, top code reserved, three conventions near 0', 'sách §A.1 và §A.2 (pp. 584–589): mã hóa excess, mã trên cùng được dành riêng, ba quy ước gần 0'),
  };
}

// ---------------------------------------------------------------- derived quantities
const P2 = (k: number) => 2 ** k;

export const totalBits = (f: Format) => (f.signed ? 1 : 0) + f.e + f.m;
const allOnesE = (f: Format) => P2(f.e) - 1;
/** Largest exponent code that holds finite values. */
export function topCode(f: Format): number {
  return f.specials === 'ieee' || f.specials === 'reserved' || f.specials === 'e8m0' ? allOnesE(f) - 1 : allOnesE(f);
}
/** Largest mantissa code at the top exponent code. */
const topMan = (f: Format) => (f.specials === 'fn' ? P2(f.m) - 2 : P2(f.m) - 1);
/** Unbiased exponent of the top binade. */
export const emax = (f: Format) => topCode(f) - f.bias;
/** Exponent of the lowest binade read with a hidden 1: -bias in the no-zero convention, 1 - bias otherwise. */
export const lowExp = (f: Format) => (f.subnormals === 'nozero' ? -f.bias : 1 - f.bias);

export function maxFinite(f: Format): number {
  return (1 + topMan(f) / P2(f.m)) * P2(emax(f));
}
/** Smallest positive value read with a hidden 1 (for the no-zero convention, the smallest positive value). */
export const minNormal = (f: Format) => P2(lowExp(f));
/** Smallest positive subnormal, or null when the format has none. */
export function minSubnormal(f: Format): number | null {
  return f.subnormals === 'denorm' && f.m > 0 ? P2(lowExp(f) - f.m) : null;
}
export const minPositive = (f: Format) => minSubnormal(f) ?? minNormal(f);
/** The gap between 1 and the next value: 2^-m (NVIDIA Table 51's epsilon column). */
export const epsilon = (f: Format) => P2(-f.m);
/** Significant bits including the hidden one. */
export const precisionBits = (f: Format) => f.m + 1;

/** floor(log2 a), exactly, for finite a > 0. */
export function binade(a: number): number {
  let k = Math.floor(Math.log2(a));
  while (P2(k) > a) k--;
  while (P2(k + 1) <= a) k++;
  return k;
}

// ---------------------------------------------------------------- bits
export interface Fields {
  s: number;
  E: number;
  M: number;
  sBits: string;
  eBits: string;
  mBits: string;
}
export const bitString = (v: number, width: number) => (width === 0 ? '' : (v >>> 0).toString(2).padStart(width, '0').slice(-width));

export function fields(f: Format, bits: number): Fields {
  const b = bits >>> 0;
  const M = b % P2(f.m);
  const E = Math.floor(b / P2(f.m)) % P2(f.e);
  const s = f.signed ? Math.floor(b / P2(f.m + f.e)) % 2 : 0;
  return { s, E, M, sBits: f.signed ? String(s) : '', eBits: bitString(E, f.e), mBits: bitString(M, f.m) };
}
export function fromFields(f: Format, s: number, E: number, M: number): number {
  return (f.signed && s ? P2(f.e + f.m) : 0) + E * P2(f.m) + M;
}
/** Flip bit i (0 = least significant) of a pattern of up to 32 bits. */
export const flipBit = (bits: number, i: number) => ((bits >>> 0) ^ (i === 31 ? 0x80000000 : 1 << i)) >>> 0;
/** "S EEE MM" with spaces between the fields. */
export function showBits(f: Format, bits: number): string {
  const x = fields(f, bits);
  return [x.sBits, x.eBits, x.mBits].filter((t) => t !== '').join(' ');
}

// ---------------------------------------------------------------- decode
export interface Decoded {
  value: number;
  cls: Cls;
}
export interface Anatomy extends Decoded {
  s: number;
  E: number;
  M: number;
  /** The leading bit of the significand: 1 for normal values, 0 for subnormal ones, null for specials. */
  hidden: 0 | 1 | null;
  /** The power of two the significand is scaled by, or null for specials and zeros of the abrupt convention. */
  scale: number | null;
  /** hidden + M / 2^m, or null. */
  significand: number | null;
  /** Which rule produced the value. */
  rule: 'normal' | 'denormal' | 'flushed-pattern' | 'nozero-low' | 'special';
}

const nanCls = (f: Format, M: number): Cls =>
  f.specials === 'ieee' && f.quietBit ? (M >= P2(f.m - 1) ? 'qnan' : 'snan') : 'nan';

export function anatomy(f: Format, bits: number): Anatomy {
  const { s, E, M } = fields(f, bits);
  const sign = s ? -1 : 1;
  const base = { s, E, M };
  const special = (value: number, cls: Cls): Anatomy => ({ ...base, value, cls, hidden: null, scale: null, significand: null, rule: 'special' });
  if (f.specials === 'e8m0' && E === allOnesE(f)) return special(NaN, 'nan');
  if (f.specials === 'ieee' && E === allOnesE(f)) return M === 0 ? special(sign * Infinity, 'inf') : special(NaN, nanCls(f, M));
  if (f.specials === 'reserved' && E === allOnesE(f)) return special(NaN, 'reserved');
  if (f.specials === 'fn' && E === allOnesE(f) && M === P2(f.m) - 1) return special(NaN, 'nan');
  if (f.specials === 'fnuz' && s === 1 && E === 0 && M === 0) return special(NaN, 'nan');
  if (E === 0 && f.subnormals !== 'nozero') {
    if (f.subnormals === 'abrupt')
      return { ...base, value: sign * 0, cls: 'zero', hidden: null, scale: null, significand: null, rule: 'flushed-pattern' };
    const significand = M / P2(f.m);
    const scale = 1 - f.bias;
    const value = sign * significand * P2(scale);
    return { ...base, value, cls: M === 0 ? 'zero' : 'subnormal', hidden: 0, scale, significand, rule: 'denormal' };
  }
  const significand = 1 + M / P2(f.m);
  const scale = E - f.bias;
  return {
    ...base,
    value: sign * significand * P2(scale),
    cls: 'normal',
    hidden: 1,
    scale,
    significand,
    rule: E === 0 ? 'nozero-low' : 'normal',
  };
}

export function decode(f: Format, bits: number): Decoded {
  const a = anatomy(f, bits);
  return { value: a.value, cls: a.cls };
}

// ---------------------------------------------------------------- rounding
/** Code (exponent and mantissa, no sign) of a representable magnitude, or of 2^(emax+1). */
function magCode(f: Format, v: number): number {
  if (v === 0) return 0;
  const L = lowExp(f);
  if (v < P2(L)) return v / P2(L - f.m); // E = 0 in the denormal convention
  const k = binade(v);
  return (k + f.bias) * P2(f.m) + (v / P2(k) - 1) * P2(f.m);
}

/** The grid neighbours lo <= a <= hi of a finite magnitude a, with no upper limit on the exponent. lo is null in the no-zero gap. */
function bracket(f: Format, a: number): { lo: number | null; hi: number } {
  const L = lowExp(f);
  const small = P2(L);
  if (a >= small || f.subnormals === 'denorm') {
    const q = a >= small ? P2(binade(a) - f.m) : P2(L - f.m);
    const lo = Math.floor(a / q) * q;
    return { lo, hi: lo === a ? lo : lo + q };
  }
  if (f.subnormals === 'abrupt') return a === 0 ? { lo: 0, hi: 0 } : { lo: 0, hi: small };
  return { lo: null, hi: small };
}

/** Next representable magnitude above a representable magnitude a (no upper limit). */
function nextMagUp(f: Format, a: number): number {
  const L = lowExp(f);
  const small = P2(L);
  if (a < small) return f.subnormals === 'denorm' ? a + P2(L - f.m) : small;
  return a + P2(binade(a) - f.m);
}
/** Next representable magnitude below a representable magnitude a > 0, or null when there is none (no-zero convention). */
function nextMagDown(f: Format, a: number): number | null {
  const L = lowExp(f);
  const small = P2(L);
  if (a <= small) {
    if (f.subnormals === 'denorm') return a - P2(L - f.m);
    if (f.subnormals === 'abrupt') return 0;
    return null;
  }
  const k = binade(a);
  return a === P2(k) ? a - P2(k - 1 - f.m) : a - P2(k - f.m);
}

type MagMode = 'nearest' | 'zero' | 'away';
const magMode = (mode: Mode, negative: boolean): MagMode =>
  mode === 'rne' ? 'nearest' : mode === 'rtz' ? 'zero' : mode === 'up' ? (negative ? 'zero' : 'away') : negative ? 'away' : 'zero';

/** Round a finite magnitude a, plus an exact tail (|tail| far below one unit of the target), to the grid. */
function roundMag(f: Format, a: number, tail: number, mm: MagMode): number {
  let { lo, hi } = bracket(f, a);
  if (lo === null) return hi; // the no-zero gap: the smallest value is the only candidate
  if (lo === hi) {
    if (tail === 0) return a;
    if (tail > 0) {
      hi = nextMagUp(f, a);
    } else {
      const d = nextMagDown(f, a);
      if (d === null) return a;
      lo = d;
      hi = a;
    }
    if (mm === 'zero') return tail > 0 ? a : lo;
    if (mm === 'away') return tail > 0 ? hi : a;
    return a; // the tail is far smaller than half a unit
  }
  if (mm === 'zero') return lo;
  if (mm === 'away') return hi;
  const d1 = a - lo;
  const d2 = hi - a;
  if (d1 < d2) return lo;
  if (d2 < d1) return hi;
  if (tail > 0) return hi;
  if (tail < 0) return lo;
  const cl = magCode(f, lo) % 2;
  const ch = magCode(f, hi) % 2;
  if (cl === 0 && ch !== 0) return lo;
  if (ch === 0 && cl !== 0) return hi;
  return lo; // both even: only in the abrupt-underflow gap; ties go to zero
}

const hasInf = (f: Format) => f.specials === 'ieee';
const hasNaN = (f: Format) => f.specials === 'ieee' || f.specials === 'fn' || f.specials === 'fnuz' || f.specials === 'e8m0';

/** The value a format stores for hi + lo (lo the exact tail of a TwoSum, or 0). */
function roundPair(f: Format, hi: number, lo: number, mode: Mode, overflow: Overflow): number {
  if (Number.isNaN(hi)) return NaN;
  const negative = hi < 0 || Object.is(hi, -0);
  if (!f.signed && negative && hi !== 0) return NaN;
  const max = maxFinite(f);
  if (!Number.isFinite(hi)) {
    if (hasInf(f) && overflow === 'ieee') return hi;
    if (overflow === 'saturate' || !hasNaN(f)) return negative ? -max : max;
    return NaN;
  }
  const sign = negative && f.signed ? -1 : 1;
  const a = Math.abs(hi);
  const mm = magMode(mode, negative);
  const r = roundMag(f, a, a === 0 ? 0 : negative ? -lo : lo, mm);
  if (r > max) {
    if (overflow === 'saturate' || !hasNaN(f) || mm === 'zero') return sign * max;
    return hasInf(f) ? sign * Infinity : NaN;
  }
  if (r === 0 && f.specials === 'fnuz') return 0;
  return sign * r;
}

/** The value the format stores for x. */
export function round(f: Format, x: number, mode: Mode = 'rne', overflow: Overflow = 'ieee'): number {
  return roundPair(f, x, 0, mode, overflow);
}

/** The bit pattern of a value of the format (no rounding: v must be representable), or -1 when it has none. */
export function bitsOfValue(f: Format, v: number): number {
  if (Number.isNaN(v)) {
    switch (f.specials) {
      case 'ieee':
        return f.m > 0 ? allOnesE(f) * P2(f.m) + P2(f.m - 1) : -1;
      case 'fn':
        return P2(f.e + f.m) - 1;
      case 'fnuz':
        return P2(f.e + f.m);
      case 'e8m0':
        return allOnesE(f);
      default:
        return -1;
    }
  }
  const negative = f.signed && (v < 0 || (Object.is(v, -0) && f.specials !== 'fnuz'));
  const signBit = negative ? P2(f.e + f.m) : 0;
  const a = Math.abs(v);
  if (a === Infinity) return hasInf(f) ? signBit + allOnesE(f) * P2(f.m) : -1;
  return signBit + magCode(f, a);
}

/** The bit pattern the format stores for x (-1 if the result has no encoding: NaN in a format without NaN). */
export function encode(f: Format, x: number, mode: Mode = 'rne', overflow: Overflow = 'ieee'): number {
  return bitsOfValue(f, round(f, x, mode, overflow));
}

// ---------------------------------------------------------------- values, spacing, neighbours
const valueCache = new Map<string, number[]>();
const key = (f: Format) => [f.e, f.m, f.bias, f.signed, f.specials, f.subnormals].join('|');

/** All finite values, sorted ascending, each once (+0 and -0 count as one 0). Formats of up to 16 bits. */
export function values(f: Format): number[] {
  if (totalBits(f) > 16) throw new Error('values(): formats of up to 16 bits only');
  const k = key(f);
  const hit = valueCache.get(k);
  if (hit) return hit;
  const set = new Set<number>();
  const n = P2(totalBits(f));
  for (let b = 0; b < n; b++) {
    const v = decode(f, b).value;
    if (Number.isFinite(v)) set.add(v === 0 ? 0 : v);
  }
  const out = [...set].sort((x, y) => x - y);
  valueCache.set(k, out);
  return out;
}

/**
 * The spacing of the format's values at x: 2^(k - m) for x in the binade [2^k, 2^(k+1)),
 * the subnormal spacing below the smallest normal value, and the width of the gap next to 0
 * in the abrupt and no-zero conventions. Beyond the largest value, the top binade's spacing.
 */
export function ulp(f: Format, x: number): number {
  const a = Math.abs(x);
  if (!Number.isFinite(a)) return NaN;
  const L = lowExp(f);
  if (a < P2(L)) return f.subnormals === 'denorm' ? P2(L - f.m) : P2(L);
  return P2(Math.min(binade(a), emax(f)) - f.m);
}

/** The smallest value of the format above x; Infinity past the top for IEEE formats; null when there is none. */
export function nextUp(f: Format, x: number): number | null {
  if (Number.isNaN(x) || x === Infinity) return null;
  const max = maxFinite(f);
  if (!f.signed && x < minPositive(f)) return minPositive(f);
  if (x === -Infinity) return -max;
  if (x >= 0) {
    const a = Math.abs(x);
    const { lo, hi } = bracket(f, a);
    const next = lo !== null && lo === hi && !(a === 0 && f.subnormals === 'nozero') ? nextMagUp(f, a) : hi;
    if (next > max) return hasInf(f) ? Infinity : null;
    return next;
  }
  const a = -x;
  if (a > max) return -max;
  const { lo, hi } = bracket(f, a);
  const below = lo !== null && lo === hi ? nextMagDown(f, a) : lo;
  if (below === null) return minPositive(f); // no-zero: from the smallest negative to the smallest positive
  if (below === 0) return f.specials === 'fnuz' ? 0 : -0;
  return -below;
}

/** The largest value of the format below x; -Infinity past the bottom for IEEE formats; null when there is none. */
export function nextDown(f: Format, x: number): number | null {
  if (f.signed) {
    const u = nextUp(f, -x);
    return u === null ? null : -u;
  }
  if (Number.isNaN(x) || x <= minPositive(f)) return null;
  if (x > maxFinite(f)) return maxFinite(f);
  const { lo, hi } = bracket(f, x);
  return lo !== null && lo === hi ? nextMagDown(f, x) : lo;
}

// ---------------------------------------------------------------- classify
export type Status = 'zero' | 'normal' | 'subnormal' | 'flushed' | 'inf' | 'nan' | 'saturated';
export interface Classified {
  input: number;
  /** The value stored. */
  value: number;
  /** Its bit pattern (-1 when it has none). */
  bits: number;
  /** The class of the stored pattern. */
  cls: Cls;
  status: Status;
  /** value - input (NaN when either is not finite). */
  error: number;
  /** |error| / |input| (0 for an exact zero input). */
  relError: number;
  /** ulp(f, input). */
  ulp: number;
  exact: boolean;
}
export interface ClassifyOptions {
  mode?: Mode;
  overflow?: Overflow;
  /** Flush a subnormal result to zero of the same sign (flush-to-zero; see the header). */
  flushDenormals?: boolean;
}

export function classify(f: Format, x: number, opts: ClassifyOptions = {}): Classified {
  const mode = opts.mode ?? 'rne';
  const overflow = opts.overflow ?? 'ieee';
  let value = round(f, x, mode, overflow);
  let bits = bitsOfValue(f, value);
  let cls: Cls = bits >= 0 ? decode(f, bits).cls : 'nan';
  let status: Status;
  // Did rounding with no upper limit on the exponent leave the range? Then a finite result is a clamp.
  const negative = x < 0 || Object.is(x, -0);
  const overflowed = Number.isFinite(x)
    ? (f.signed || !negative) && roundMag(f, Math.abs(x), 0, magMode(mode, negative)) > maxFinite(f)
    : !Number.isNaN(x);
  if (Number.isNaN(value)) status = 'nan';
  else if (!Number.isFinite(value)) status = 'inf';
  else if (overflowed) status = 'saturated';
  else if (cls === 'subnormal' && opts.flushDenormals) {
    value = value < 0 && f.specials !== 'fnuz' ? -0 : 0;
    bits = bitsOfValue(f, value);
    cls = 'zero';
    status = 'flushed';
  } else status = cls === 'zero' ? 'zero' : cls === 'subnormal' ? 'subnormal' : 'normal';
  const finite = Number.isFinite(value) && Number.isFinite(x);
  const error = finite ? value - x : NaN;
  return {
    input: x,
    value,
    bits,
    cls,
    status,
    error,
    relError: finite ? (x === 0 ? Math.abs(error) : Math.abs(error) / Math.abs(x)) : NaN,
    ulp: ulp(f, x),
    exact: finite && error === 0,
  };
}

/** Flush-to-zero on a stored pattern: a subnormal becomes zero of the same sign. */
export function flushDenormals(f: Format, bits: number): number {
  const d = decode(f, bits);
  if (d.cls !== 'subnormal') return bits >>> 0;
  return bitsOfValue(f, d.value < 0 && f.specials !== 'fnuz' ? -0 : 0);
}

// ---------------------------------------------------------------- arithmetic
/** a + b, both first rounded to the format, rounded once (exact TwoSum tail, so every mode is correct). */
export function add(f: Format, a: number, b: number, mode: Mode = 'rne', overflow: Overflow = 'ieee'): number {
  const x = round(f, a, mode, overflow);
  const y = round(f, b, mode, overflow);
  const s = x + y;
  if (!Number.isFinite(x) || !Number.isFinite(y)) return round(f, s, mode, overflow);
  if (s === 0) {
    // IEEE 754 §6.3: an exact zero sum keeps the operands' sign when they agree; otherwise it
    // is +0, or -0 when rounding down.
    let z: number;
    if (Object.is(x, -0) && Object.is(y, -0)) z = -0;
    else if (Object.is(x, 0) && Object.is(y, 0)) z = 0;
    else z = mode === 'down' ? -0 : 0;
    return round(f, z, mode, overflow);
  }
  const bb = s - x;
  const tail = x - (s - bb) + (y - bb);
  return roundPair(f, s, tail, mode, overflow);
}
export function sub(f: Format, a: number, b: number, mode: Mode = 'rne', overflow: Overflow = 'ieee'): number {
  return add(f, a, -b, mode, overflow);
}
/** a × b rounded once: exact in binary64 for p <= 26, then one rounding. */
export function mul(f: Format, a: number, b: number, mode: Mode = 'rne', overflow: Overflow = 'ieee'): number {
  return round(f, round(f, a, mode, overflow) * round(f, b, mode, overflow), mode, overflow);
}
/** a / b rounded once: binary64 then one rounding, correctly rounded for p <= 25 (see the header). */
export function div(f: Format, a: number, b: number, mode: Mode = 'rne', overflow: Overflow = 'ieee'): number {
  return round(f, round(f, a, mode, overflow) / round(f, b, mode, overflow), mode, overflow);
}

// ---------------------------------------------------------------- excess coding (Fig. A.1)
/** The excess-(2^(e-1)-1) code of exponent v; v = 2^(e-1) gives the reserved all-ones code. */
export function excessCode(e: number, v: number): number {
  const c = v + excessBias(e);
  if (c < 0 || c >= P2(e)) throw new Error(`exponent ${v} has no ${e}-bit excess code`);
  return c;
}
/** The e-bit two's-complement pattern of v (v mod 2^e). */
export function twosComplement(e: number, v: number): number {
  const n = P2(e);
  return ((v % n) + n) % n;
}
/** The signed value an e-bit two's-complement pattern stands for. */
export const fromTwos = (e: number, c: number) => (c >= P2(e - 1) ? c - P2(e) : c);

export interface ExcessRow {
  excess: number;
  twos: number;
  /** The exponent, or null for the reserved all-ones code. */
  value: number | null;
}
/** Fig. A.1 for any width: every e-bit excess code in increasing order, with the two's-complement pattern beside it. */
export function excessTable(e: number): ExcessRow[] {
  const rows: ExcessRow[] = [];
  for (let c = 0; c < P2(e); c++) {
    const v = c - excessBias(e);
    rows.push({ excess: c, twos: twosComplement(e, v), value: c === P2(e) - 1 ? null : v });
  }
  return rows;
}

// ---------------------------------------------------------------- display
/** A short decimal for a value: integers as they are, others with up to `digits` significant digits. */
export function show(v: number, digits = 8): string {
  if (Number.isNaN(v)) return 'NaN';
  if (v === Infinity) return '+inf';
  if (v === -Infinity) return '−inf';
  if (v === 0) return Object.is(v, -0) ? '−0' : '0';
  const a = Math.abs(v);
  let s: string;
  if (Number.isInteger(v) && a < 1e15) s = String(v);
  else if (a >= 1e-4 && a < 1e9) s = String(Number(v.toPrecision(digits)));
  else s = v.toExponential(Math.max(0, digits - 1)).replace(/\.?0+e/, 'e');
  return s.replace(/^-/, '−');
}
/** A power of two as text, e.g. "2^-14". */
export const pow2 = (k: number) => `2^${k}`;
