// Float-to-ordered-key transform for chapter 14, section 14.4 (sorting logits). Ours, not
// the book's: the book only warns that non-comparison sorts "may not generalize to arbitrary
// types of keys" (p. 330). The transform is the one CUB documents for float keys
// (notes/today/ch14-facts.md A3) and rocPRIM uses (B2); its bit patterns are E1's table.
//
// What it models, exactly:
//  - A float32 is handled as its 32 bits, an unsigned integer, from start to finish. Values
//    are never round-tripped through a JavaScript number once they are bits, because a
//    number cannot carry a NaN's sign or payload.
//  - twiddle: set the sign bit of a non-negative value, invert every bit of a negative one.
//    Read as unsigned integers, the results are in the order of the floats.
//  - The -0.0 step: bits 0x80000000 become 0x00000000 before the transform, so -0.0 and
//    +0.0 get the same key (CUB and rocPRIM do this at digit extraction; E1's "canon").
//  - Descending: invert the key and sort ascending (CUB's documented rule), or, the wrong
//    way, sort ascending and reverse the result, which reverses the order of ties.
//  - Two partial transforms for the explorer: none (raw bits) and sign bit only.
//  - A stable sort of (token, bits) items by key, and two references: std::stable_sort with
//    > on the values, for rows without NaN; CUB's rule (the full transform, the -0.0 step,
//    inverted, stable ascending) for rows with NaN, which puts +NaN above +inf and -NaN
//    below -inf. On rows without NaN the two agree (tested).
//
// What it leaves out: float32 only (no bf16 or fp16 keys); NaN payloads other than the two
// quiet NaNs are not explored; nothing about threads, memory or time.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

export const SIGN = 0x80000000;
const view = new DataView(new ArrayBuffer(4));

/** The 32 bits of x rounded to float32, as an unsigned integer. */
export function bitsOf(x: number): number {
  view.setFloat32(0, x);
  return view.getUint32(0);
}
/** The float32 whose bits are u. A NaN comes back as a plain NaN: keep NaNs as bits. */
export function fromBits(u: number): number {
  view.setUint32(0, u >>> 0);
  return view.getFloat32(0);
}
export const hex = (u: number) => '0x' + (u >>> 0).toString(16).toUpperCase().padStart(8, '0');
export const isNaNBits = (u: number) => ((u >>> 0) & 0x7f800000) === 0x7f800000 && ((u >>> 0) & 0x007fffff) !== 0;
export const isNeg = (u: number) => ((u >>> 0) & SIGN) !== 0;

/** CUB's float transform (TwiddleIn): the result's unsigned order is the floats' order. */
export const twiddle = (u: number) => ((u >>> 0) ^ (isNeg(u) ? 0xffffffff : SIGN)) >>> 0;
/** Flip the sign bit only: the half-right transform the explorer offers. */
export const signFlip = (u: number) => ((u >>> 0) ^ SIGN) >>> 0;
/** The -0.0 step: -0.0 becomes +0.0 before the transform. */
export const zeroStep = (u: number) => ((u >>> 0) === SIGN ? 0 : u >>> 0);
/** Ascending key with the -0.0 step (E1's "Canon key"); what the s144 snippet computes. */
export const keyAsc = (u: number) => twiddle(zeroStep(u));
/** Descending key: the ascending key inverted (E1's "Desc key"). */
export const keyDesc = (u: number) => ~keyAsc(u) >>> 0;

export type Transform = 'none' | 'sign' | 'full';
export const TRANSFORM_LABEL: Record<Transform, string> = {
  get none() {
    return tr('none: raw bits', 'không: bit thô');
  },
  get sign() {
    return tr('flip the sign bit only', 'chỉ lật bit dấu');
  },
  get full() {
    return tr('full: set the sign bit, or invert a negative', 'đầy đủ: đặt bit dấu, hoặc đảo mọi bit của số âm');
  },
};
export const transform = (u: number, t: Transform) => (t === 'none' ? u >>> 0 : t === 'sign' ? signFlip(u) : twiddle(u));

export interface Item {
  token: number;
  bits: number;
}
export const items = (bits: number[]) => bits.map((b, token) => ({ token, bits: b >>> 0 }));

/** Stable ascending sort by an unsigned key: equal keys keep their input order. */
export function stableBy<T>(xs: T[], key: (t: T) => number): T[] {
  return xs
    .map((t, i) => ({ t, i, k: key(t) >>> 0 }))
    .sort((a, b) => (a.k !== b.k ? (a.k < b.k ? -1 : 1) : a.i - b.i))
    .map((x) => x.t);
}

/** std::stable_sort with > on the float values. Only meaningful without NaN (ch14-facts A11). */
export function stableDescByValue(xs: Item[]): Item[] {
  return xs
    .map((t, i) => ({ t, i, v: fromBits(t.bits) }))
    .sort((a, b) => (a.v > b.v ? -1 : a.v < b.v ? 1 : a.i - b.i))
    .map((x) => x.t);
}

/** The reference order this course uses for a descending sort (E1, A3). */
export function referenceDesc(xs: Item[]): { order: Item[]; rule: 'stable_sort with >' | "CUB's bits rule" } { // i18n-ignore: rule ids the chapter compares and words itself
  if (xs.some((x) => isNaNBits(x.bits))) return { order: stableBy(xs, (x) => keyDesc(x.bits)), rule: "CUB's bits rule" }; // i18n-ignore: rule id the chapter compares
  return { order: stableDescByValue(xs), rule: 'stable_sort with >' };
}

export type DescBy = 'invert' | 'reverse';
export interface Options {
  transform: Transform;
  /** Apply the -0.0 step before the transform. */
  zero: boolean;
  /** How to get descending order. */
  descBy: DescBy;
}
export const CUB_OPTIONS: Options = { transform: 'full', zero: true, descBy: 'invert' };

export interface Row {
  token: number;
  bits: number;
  /** Bits after the -0.0 step (the same as bits when the step is off). */
  stepped: number;
  /** The ascending key: transform(stepped). */
  key: number;
  /** What is sorted ascending: ~key when descending by inversion, key when by reversal. */
  sortKey: number;
}
export interface Explored {
  rows: Row[];
  /** Tokens in the order the chosen pipeline produces, largest first (it means to). */
  order: number[];
  /** Tokens in the reference order. */
  reference: number[];
  rule: string;
  /** Per output position: does it hold the reference's token? */
  match: boolean[];
  /** First output position that differs from the reference, or -1. */
  firstMismatch: number;
}

/** Run one descending sort of a row of logits through the chosen pipeline and compare. */
export function explore(xs: Item[], o: Options): Explored {
  const rows: Row[] = xs.map((x) => {
    const stepped = o.zero ? zeroStep(x.bits) : x.bits >>> 0;
    const key = transform(stepped, o.transform);
    return { token: x.token, bits: x.bits >>> 0, stepped, key, sortKey: o.descBy === 'invert' ? ~key >>> 0 : key };
  });
  let sorted = stableBy(rows, (r) => r.sortKey);
  if (o.descBy === 'reverse') sorted = sorted.slice().reverse();
  const order = sorted.map((r) => r.token);
  const ref = referenceDesc(xs);
  const reference = ref.order.map((x) => x.token);
  const match = order.map((t, k) => t === reference[k]);
  return { rows, order, reference, rule: ref.rule, match, firstMismatch: match.indexOf(false) };
}

/** Sort values by their raw bits read as unsigned integers, ascending (the prediction). */
export function rawAscending(values: number[]): number[] {
  return stableBy(values.map((v) => bitsOf(v)), (u) => u).map(fromBits);
}

// ---------------------------------------------------------------- presets
export const FLT_MAX_BITS = 0x7f7fffff;
export const POS_QNAN = 0x7fc00000;
export const NEG_QNAN = 0xffc00000;

/** E1's ten values, in the table's order, with the transform's expected results. */
export const E1: { label: string; bits: number; key: number; canon: number; desc: number }[] = [
  { label: '−inf', bits: 0xff800000, key: 0x007fffff, canon: 0x007fffff, desc: 0xff800000 }, // i18n-ignore: a float value, written as a value
  { label: '−FLT_MAX', bits: 0xff7fffff, key: 0x00800000, canon: 0x00800000, desc: 0xff7fffff },
  { label: '−1.0', bits: 0xbf800000, key: 0x407fffff, canon: 0x407fffff, desc: 0xbf800000 },
  {
    get label() {
      return tr('smallest negative subnormal', 'số subnormal âm nhỏ nhất');
    },
    bits: 0x80000001, key: 0x7ffffffe, canon: 0x7ffffffe, desc: 0x80000001,
  },
  { label: '−0.0', bits: 0x80000000, key: 0x7fffffff, canon: 0x80000000, desc: 0x7fffffff },
  { label: '+0.0', bits: 0x00000000, key: 0x80000000, canon: 0x80000000, desc: 0x7fffffff },
  {
    get label() {
      return tr('smallest positive subnormal', 'số subnormal dương nhỏ nhất');
    },
    bits: 0x00000001, key: 0x80000001, canon: 0x80000001, desc: 0x7ffffffe,
  },
  { label: '1.0', bits: 0x3f800000, key: 0xbf800000, canon: 0xbf800000, desc: 0x407fffff },
  { label: 'FLT_MAX', bits: 0x7f7fffff, key: 0xff7fffff, canon: 0xff7fffff, desc: 0x00800000 },
  { label: '+inf', bits: 0x7f800000, key: 0xff800000, canon: 0xff800000, desc: 0x007fffff }, // i18n-ignore: a float value, written as a value
];

export interface Preset {
  id: string;
  label: string;
  bits: number[];
  /** Display labels, by token, where a number would not say it well. */
  labels?: string[];
}
export const PRESETS: Preset[] = [
  {
    id: 'ties',
    get label() {
      return tr('logits with ties', 'logits có giá trị bằng nhau');
    },
    // Two tied pairs: tokens 1 and 4, tokens 3 and 7.
    bits: [-0.5, 2.25, -1.75, 1.0, 2.25, -3.0, 0.25, 1.0].map(bitsOf),
  },
  {
    id: 'zeros',
    get label() {
      return tr('signs and zeros', 'dấu và số không');
    },
    bits: [-2.0, -1.0, -0, 0, 0.5, 1.0].map(bitsOf),
  },
  {
    id: 'edges',
    get label() {
      return tr('edges (E1 plus two NaNs)', 'các giá trị biên (E1 cộng hai NaN)');
    },
    bits: [...E1.map((e) => e.bits), POS_QNAN, NEG_QNAN],
    get labels() {
      return [...E1.map((e) => e.label), '+NaN', '−NaN'];
    },
  },
];

/** A short label for a float given by its bits, for tables. */
export function show(u: number): string {
  u = u >>> 0;
  if (isNaNBits(u)) return isNeg(u) ? '−NaN' : '+NaN';
  if (u === 0x7f800000) return '+inf';
  if (u === 0xff800000) return '−inf';
  if (u === 0) return '+0.0';
  if (u === SIGN) return '−0.0';
  if (u === FLT_MAX_BITS) return 'FLT_MAX';
  if (u === (FLT_MAX_BITS | SIGN) >>> 0) return '−FLT_MAX';
  const v = fromBits(u);
  const s = Math.abs(v) < 1e-30 || Math.abs(v) > 1e30 ? v.toExponential(1) : Number.isInteger(v) ? v.toFixed(1) : String(v);
  return s.replace(/-/g, '−');
}
