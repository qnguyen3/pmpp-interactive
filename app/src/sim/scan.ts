// Scan semantics (chapter 11, section 11.1, and the cost formulas the chapter's start
// page lists). One model for the "Cut the sausage" explorer, the "Where the nucleus ends"
// explorer and every number section 11.1 states:
//
//   - inclusive and exclusive scans with + (exact), both conversions of p. 253, and the
//     allocation offsets that an exclusive scan of sizes gives (p. 253);
//   - the same scans in float32 (Math.fround after every addition) in three association
//     orders: the left-to-right loop of Fig. 11.1, the Kogge-Stone order of Fig. 11.2
//     (lockstep strides 1, 2, 4, ...) and the Brent-Kung order of Fig. 11.19;
//   - four top-p keep rules over probabilities sorted in descending order, as the real
//     projects compute them (notes/today/ch11-facts.md D2 to D4):
//       'exclusive'  keep i while the mass ranked before i is below p (llama.cpp's host and
//                    backend samplers; vLLM and transformers reach the same set, D2, D4);
//       'inclusive'  keep i while the mass up to and including i is below p: the section
//                    11.1 off-by-one, which drops the token that crosses p;
//       'mlx'        mlx-lm's apply_top_p: the exclusive rule normalised by the computed
//                    total, then every token tied with the boundary token kept (D3);
//       'ascending'  transformers' TopPLogitsWarper (and vLLM's PyTorch path): sort
//                    ascending, inclusive cumsum, remove while it is <= 1 - p, never the
//                    top token (D4);
//   - a rounding band around p inside which a test must accept either decision, and a
//     seeded search for a float32 row on which the loop order and the Kogge-Stone order put
//     one token on opposite sides of p.
//
// Sources: the book, pp. 252-254 (definitions, the sausage example, conversions, the
// loop), p. 266 (Kogge-Stone work), p. 284 (Brent-Kung work); the projects' rules and the
// libraries' non-determinism from notes/today/ch11-facts.md A8, D2, D3, D4; float
// non-associativity from notes/today/ch10-facts.md A9.
//
// What it leaves out (the lesson says so beside the explorers): probabilities are given
// and already sorted, there is no sampling step, float32 is imitated with Math.fround
// after every addition, mlx-lm's comparison in log space is done on the probabilities,
// and transformers' 1 - p is formed in double and rounded to float32 once (our modelling
// choice; the facts file does not say how PyTorch rounds the scalar).
//
// Erasable TypeScript only, and the one import is lib/locale.ts (itself import-free): the browser
// checks import this file directly.

import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- the book's rows
/** The sausage orders of p. 252. */
export const BOOK_ORDERS = [3, 1, 7, 0, 4, 1, 6, 3];
/** Their inclusive scan, as printed on p. 252. */
export const BOOK_INCLUSIVE = [3, 4, 11, 11, 15, 16, 22, 25];
/** Their exclusive scan, as printed on p. 253. */
export const BOOK_EXCLUSIVE = [0, 3, 4, 11, 11, 15, 16, 22];
/** The sausage of p. 252 is 40 inches long. */
export const SAUSAGE_INCHES = 40;
/** Token counts of six sequences to pack into one buffer (ours). */
export const TOKEN_COUNTS = [5, 3, 0, 7, 2, 4];
/** cppreference's example for std::inclusive_scan and std::exclusive_scan (ch11-facts A8). */
export const CPPREF_INPUT = [3, 1, 4, 1, 5, 9, 2, 6];
export const CPPREF_EXCLUSIVE = [0, 3, 4, 8, 9, 14, 23, 25];
export const CPPREF_INCLUSIVE = [3, 4, 8, 9, 14, 23, 25, 31];

// ---------------------------------------------------------------- exact scans
/** Inclusive scan with +: out[i] = x[0] + ... + x[i] (p. 252). */
export function inclusive(xs: number[]): number[] {
  const out: number[] = [];
  let acc = 0;
  xs.forEach((x, i) => {
    acc = i === 0 ? x : acc + x;
    out.push(acc);
  });
  return out;
}

/** Exclusive scan with +: out[0] = identity, out[i] = x[0] + ... + x[i - 1] (p. 253). */
export function exclusive(xs: number[], identity = 0): number[] {
  const out: number[] = [];
  let acc = identity;
  for (const x of xs) {
    out.push(acc);
    acc = acc + x;
  }
  return out;
}

/** Inclusive to exclusive: shift right by one, identity in slot 0 (p. 253). */
export function toExclusive(inc: number[], identity = 0): number[] {
  return inc.length ? [identity, ...inc.slice(0, -1)] : [];
}

/** Exclusive to inclusive: shift left by one, last slot = last exclusive + last input (p. 253). */
export function toInclusive(exc: number[], xs: number[]): number[] {
  if (!exc.length) return [];
  return [...exc.slice(1), exc[exc.length - 1] + xs[xs.length - 1]];
}

/** Where each piece starts and ends when pieces of these sizes are laid out in order. */
export function offsets(sizes: number[]): { start: number[]; end: number[]; total: number } {
  const start = exclusive(sizes);
  const end = inclusive(sizes);
  return { start, end, total: end.length ? end[end.length - 1] : 0 };
}

/** Additions the sequential loop of Fig. 11.1 performs on n elements (p. 254). */
export const sequentialAdditions = (n: number) => Math.max(0, n - 1);
/** Additions of the naive parallel scan, one sequential prefix per thread (p. 254). */
export const naiveAdditions = (n: number) => (n * (n - 1)) / 2;

// ---------------------------------------------------------------- float32 helpers
export const f32 = Math.fround;
const view = new DataView(new ArrayBuffer(4));
/** The 32 bits of a float32, as an unsigned integer. */
export function f32bits(x: number): number {
  view.setFloat32(0, x);
  return view.getUint32(0);
}
/** A float32 written as its bit pattern, e.g. 0x3f400000. */
export function f32hex(x: number): string {
  return '0x' + f32bits(x).toString(16).padStart(8, '0');
}
function ordered(x: number): number {
  const b = f32bits(x);
  return b >= 0x80000000 ? -(b - 0x80000000) : b;
}
/** Signed distance in float32 steps from b to a: 0 when equal, -1 when a is the float just below b. */
export function ulpSigned(a: number, b: number): number {
  return ordered(f32(a)) - ordered(f32(b));
}
/** The shortest decimal that reads back as the same float32. */
export function f32short(x: number): string {
  const v = f32(x);
  if (Number.isNaN(v)) return 'NaN';
  for (let p = 1; p <= 9; p++) {
    const s = v.toPrecision(p);
    if (f32(Number(s)) === v) return tidy(s);
  }
  return tidy(v.toPrecision(9));
}
function tidy(s: string): string {
  if (s.includes('e')) {
    const [m, e] = s.split('e');
    return `${tidy(m)}e${e}`;
  }
  return s.includes('.') ? s.replace(/0+$/, '').replace(/\.$/, '') : s;
}

// ---------------------------------------------------------------- scans in an order
export type Order = 'loop' | 'kogge-stone' | 'brent-kung';
export const ORDER_LABEL: Record<Order, string> = {
  get loop() {
    return tr('the loop of Fig. 11.1', 'vòng lặp của Fig. 11.1');
  },
  get 'kogge-stone'() {
    return tr('Kogge-Stone order', 'thứ tự Kogge-Stone');
  },
  get 'brent-kung'() {
    return tr('Brent-Kung order', 'thứ tự Brent-Kung');
  },
};

/** Inclusive scan with every addition rounded by `round` (Math.fround for float32), in an order. */
export function inclusiveIn(xs: number[], order: Order, round: (v: number) => number = f32): number[] {
  const x = xs.map((v) => round(v));
  const n = x.length;
  if (order === 'loop') {
    for (let i = 1; i < n; i++) x[i] = round(x[i - 1] + x[i]);
    return x;
  }
  if (order === 'kogge-stone') {
    // Fig. 11.2: in step k, every position i >= 2^k adds the value 2^k to its left,
    // all positions reading the values of the step before (lockstep).
    let cur = x;
    for (let stride = 1; stride < n; stride *= 2) {
      cur = cur.map((v, i) => (i >= stride ? round(v + cur[i - stride]) : v));
    }
    return cur;
  }
  // Brent-Kung, Fig. 11.19: a reduction tree, then a reverse tree.
  for (let stride = 1; stride < n; stride *= 2)
    for (let i = 0; i < n; i++) if ((i + 1) % (2 * stride) === 0) x[i] = round(x[i] + x[i - stride]);
  let top = 1;
  while (top * 2 < n) top *= 2; // the largest power of two below n
  for (let stride = top / 2; stride >= 1; stride /= 2)
    for (let t = 0; ; t++) {
      const idx = (t + 1) * 2 * stride - 1;
      if (idx + stride >= n) break;
      x[idx + stride] = round(x[idx + stride] + x[idx]);
    }
  return x;
}

/** Exclusive scan in an order: the inclusive scan in that order, shifted (the book's conversion, p. 253). */
export function exclusiveIn(xs: number[], order: Order, round: (v: number) => number = f32): number[] {
  return toExclusive(inclusiveIn(xs, order, round), 0);
}

/** Additions each order performs on n elements (pp. 254, 266, 284). */
export function orderAdditions(n: number, order: Order): number {
  if (order === 'loop') return sequentialAdditions(n);
  if (order === 'kogge-stone') {
    let s = 0;
    for (let stride = 1; stride < n; stride *= 2) s += n - stride;
    return s;
  }
  let s = 0;
  for (let stride = 1; stride < n; stride *= 2) for (let i = 0; i < n; i++) if ((i + 1) % (2 * stride) === 0) s++;
  let top = 1;
  while (top * 2 < n) top *= 2;
  for (let stride = top / 2; stride >= 1; stride /= 2)
    for (let t = 0; (t + 1) * 2 * stride - 1 + stride < n; t++) s++;
  return s;
}

// ---------------------------------------------------------------- top-p
export type Rule = 'exclusive' | 'inclusive' | 'mlx' | 'ascending';
export const RULES: Rule[] = ['exclusive', 'inclusive', 'mlx', 'ascending'];
export const RULE_LABEL: Record<Rule, string> = {
  get exclusive() {
    return tr('mass before < p', 'khối lượng trước nó < p');
  },
  get inclusive() {
    return tr('mass up to and including < p', 'khối lượng tính cả nó < p');
  },
  get mlx() {
    return tr('mass before < p × total, then keep ties', 'khối lượng trước nó < p × tổng, rồi giữ các token bằng nhau');
  },
  get ascending() {
    return tr('ascending cumsum ≤ 1 − p', 'cumsum tăng dần ≤ 1 − p');
  },
};
export const RULE_WHO: Record<Rule, string> = {
  get exclusive() {
    return tr(
      'llama.cpp (host and backend samplers); vLLM and transformers reach the same set',
      'llama.cpp (bộ lấy mẫu phía host và phía backend); vLLM và transformers ra cùng một tập',
    );
  },
  get inclusive() {
    return tr('nobody: the off-by-one', 'không ai cả: lỗi lệch một (off-by-one)');
  },
  mlx: 'mlx-lm apply_top_p', // i18n-ignore: library and function name, kept in English
  get ascending() {
    return tr('transformers TopPLogitsWarper, vLLM PyTorch path', 'TopPLogitsWarper của transformers, nhánh PyTorch của vLLM');
  },
};
export const RULE_FACT: Record<Rule, string> = {
  exclusive: 'D2, D4',
  inclusive: '§11.1',
  mlx: 'D3',
  ascending: 'D4',
};

/** 'exact': decimal arithmetic on the probabilities as written; 'f32': float32 in an order. */
export type Arith = { kind: 'exact' } | { kind: 'f32'; order: Order };
export const EXACT: Arith = { kind: 'exact' };

export interface TopP {
  rule: Rule;
  keep: boolean[];
  kept: number;
  /** Mass before each token (descending order), as this arithmetic computes it. */
  before: number[];
  /** Mass up to and including each token. */
  upto: number[];
}

/** Units of 10^-6: every probability and p used by the lesson is a decimal with at most 6 places. */
const SCALE = 1_000_000;
function units(x: number): number {
  const u = Math.round(x * SCALE);
  if (Math.abs(u - x * SCALE) > 1e-6) throw new Error(`${x} has more than 6 decimal places`);
  return u;
}

/**
 * Which tokens a top-p rule keeps. `probs` are sorted in descending order, `p` is in (0, 1).
 * Every rule keeps at least the top token except 'inclusive', which can keep none.
 */
export function topP(probs: number[], p: number, rule: Rule, arith: Arith = EXACT): TopP {
  const n = probs.length;
  if (arith.kind === 'exact') {
    const x = probs.map(units);
    const pu = units(p);
    const upto = inclusive(x);
    const before = exclusive(x);
    const total = n ? upto[n - 1] : 0;
    let keep: boolean[];
    if (rule === 'exclusive') keep = before.map((b) => b < pu);
    else if (rule === 'inclusive') keep = upto.map((u) => u < pu);
    else if (rule === 'mlx') {
      const k = before.filter((b) => b * SCALE < pu * total).length;
      keep = x.map((v) => k > 0 && v >= x[k - 1]);
    } else {
      // Ascending inclusive cumsum: the mass at or below token i in descending order.
      const asc = inclusive([...x].reverse());
      keep = x.map((_, i) => i === 0 || !(asc[n - 1 - i] <= SCALE - pu));
    }
    return { rule, keep, kept: keep.filter(Boolean).length, before: before.map((v) => v / SCALE), upto: upto.map((v) => v / SCALE) };
  }
  const x = probs.map(f32);
  const pf = f32(p);
  const upto = inclusiveIn(x, arith.order);
  const before = toExclusive(upto, 0);
  let keep: boolean[];
  if (rule === 'exclusive') keep = before.map((b) => b < pf);
  else if (rule === 'inclusive') keep = upto.map((u) => u < pf);
  else if (rule === 'mlx') {
    const total = n ? f32(before[n - 1] + x[n - 1]) : 0;
    const thr = f32(pf * total);
    const k = before.filter((b) => b < thr).length;
    keep = x.map((v) => k > 0 && v >= x[k - 1]);
  } else {
    const asc = inclusiveIn([...x].reverse(), arith.order);
    const one = f32(1 - p);
    keep = x.map((_, i) => i === 0 || !(asc[n - 1 - i] <= one));
  }
  return { rule, keep, kept: keep.filter(Boolean).length, before, upto };
}

// ---------------------------------------------------------------- rounding band
/** Unit roundoff of float32 (round to nearest): 2^-24. */
export const U32 = 2 ** -24;
/** Higham's gamma_k = k u / (1 - k u): bounds the rounding error of a sum of k + 1 terms in any order. */
export const gamma = (k: number) => (k * U32) / (1 - k * U32);

export type Verdict = 'keep' | 'drop' | 'either';
export interface Band {
  /** gamma_(n-1) times the sum of the row: no order of additions can move a partial sum further. */
  bound: number;
  /** Mass before each token of the float32 row, summed in double. */
  before: number[];
  /** What a test may require of each token under the 'exclusive' rule. */
  verdict: Verdict[];
}

/** The band a test of the 'exclusive' rule must allow, for a row given to the kernel in float32. */
export function band(probs: number[], p: number): Band {
  const x = probs.map(f32);
  const pf = f32(p);
  const before = exclusive(x);
  const bound = gamma(Math.max(0, x.length - 1)) * x.reduce((a, b) => a + Math.abs(b), 0);
  const verdict = before.map((b): Verdict => (b + bound < pf ? 'keep' : b - bound >= pf ? 'drop' : 'either'));
  return { bound, before, verdict };
}

// ---------------------------------------------------------------- presets
export interface Preset {
  id: string;
  label: string;
  probs: number[];
  /** A p that shows what the preset is for. */
  p: number;
}
export const PRESETS: Preset[] = [
  {
    id: 'peaked',
    get label() {
      return tr('peaked', 'nhọn');
    },
    probs: [0.62, 0.18, 0.09, 0.05, 0.03, 0.02, 0.01],
    p: 0.9,
  },
  {
    id: 'flat',
    get label() {
      return tr('flat', 'phẳng');
    },
    probs: [0.14, 0.13, 0.13, 0.12, 0.12, 0.12, 0.11, 0.08, 0.05],
    p: 0.5,
  },
  { id: 'dyadic', label: '0.5 0.25 0.125 0.125', probs: [0.5, 0.25, 0.125, 0.125], p: 0.75 },
  {
    id: 'tied',
    get label() {
      return tr('four tied at 0.25', 'bốn token bằng nhau ở 0.25');
    },
    probs: [0.25, 0.25, 0.25, 0.25],
    p: 0.5,
  },
  {
    id: 'boundary-tie',
    get label() {
      return tr('a tie at the boundary', 'bằng nhau ở ranh giới');
    },
    probs: [0.4, 0.2, 0.2, 0.2],
    p: 0.5,
  },
  {
    id: 'zero-tail',
    get label() {
      return tr('a zero tail', 'đuôi bằng 0');
    },
    probs: [0.5, 0.3, 0.2, 0, 0, 0],
    p: 0.95,
  },
];

// ---------------------------------------------------------------- orders that disagree
export interface Flip {
  probs: number[];
  p: number;
  /** The token whose keep decision differs between the two orders. */
  token: number;
  loop: number;
  ks: number;
}

function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A random row of `n` probabilities in hundredths that sum to exactly 1.00, sorted descending, none zero. */
export function hundredthsRow(n: number, rand: () => number): number[] {
  const cuts = new Set<number>();
  while (cuts.size < n - 1) cuts.add(1 + Math.floor(rand() * 99));
  const c = [0, ...[...cuts].sort((a, b) => a - b), 100];
  return c.slice(1).map((v, i) => v - c[i]).sort((a, b) => b - a).map((h) => h / 100);
}

/**
 * Searches rows of hundredths for a token whose exact mass before it equals p (so the
 * decision rests on the last bits) and on which float32 in loop order and in Kogge-Stone
 * order fall on opposite sides of p. Deterministic for a seed.
 */
export function findOrderFlip(seed = 1, tries = 5000): Flip | null {
  const rand = mulberry(seed);
  for (let t = 0; t < tries; t++) {
    const n = 8 + Math.floor(rand() * 5);
    const probs = hundredthsRow(n, rand);
    const exact = exclusive(probs.map((v) => Math.round(v * 100)));
    const loop = exclusiveIn(probs, 'loop');
    const ks = exclusiveIn(probs, 'kogge-stone');
    for (let k = 1; k < n; k++) {
      const p = exact[k] / 100;
      if (p < 0.05 || p > 0.95) continue;
      const pf = f32(p);
      if (loop[k] < pf !== ks[k] < pf) return { probs, p, token: k, loop: loop[k], ks: ks[k] };
    }
  }
  return null;
}

/**
 * The row findOrderFlip(1) returns; scan.test.ts checks that it still does. The mass
 * before token 5 is exactly 0.85 in decimal. In float32 the loop's sum lands one step
 * below float32(0.85) and the Kogge-Stone sum lands on it, so the 'exclusive' rule keeps
 * token 5 in loop order and drops it in Kogge-Stone order.
 */
export const FLIP: Flip = {
  probs: [0.27, 0.24, 0.15, 0.11, 0.08, 0.07, 0.03, 0.02, 0.01, 0.01, 0.01],
  p: 0.85,
  token: 5,
  loop: 0.8499999642372131,
  ks: 0.8500000238418579,
};

/** The presets plus the row on which the two orders disagree. */
// Array spread keeps the PRESETS objects themselves, so their label getters still read at call time.
export const ALL_PRESETS: Preset[] = [
  ...PRESETS,
  {
    id: 'flip',
    get label() {
      return tr('two orders disagree', 'hai thứ tự cho kết quả khác nhau');
    },
    probs: FLIP.probs,
    p: FLIP.p,
  },
];
