// Reduction trees (chapter 10, sections 10.1 and 10.2). One model for the fold
// explorer, the tree builder, the float32 orders explorer, the tie explorer and the
// "tree-fold" scene:
//
//   - the sequential loop of Figs. 10.1 and 10.2, with a per-iteration trace, in
//     float32 (rounding after every operation, Math.fround) or float64;
//   - reduction trees in three orders: "the loop" (a left fold from element 0, the
//     parenthesisation printed first on p. 225), "neighbours" (Fig. 10.3: adjacent
//     pairs at every level) and "halves" (node j with node j + m/2, the operand order
//     of the convergent kernel, Fig. 10.7);
//   - work, span and parallelism per step (pp. 225-226);
//   - float32 sums of the three orders against a float64 reference, with hex and
//     ULP distances (p. 224, Appendix A section A.6);
//   - (value, index) pair operators with explicit tie and NaN rules (footnote 3,
//     p. 225, and the library policies in notes/today/ch10-facts.md A10, A13, B5, D2-D4);
//   - seeded data generators and a knockout tournament with V venues (pp. 226-227).

import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- float32 helpers
export const f32 = Math.fround;
/** Largest finite float32, C's FLT_MAX: (2 - 2^-23) x 2^127. */
export const FLT_MAX = 3.4028234663852886e38;

const view = new DataView(new ArrayBuffer(4));
/** The 32 bits of a float32, as an unsigned integer. */
export function f32bits(x: number): number {
  view.setFloat32(0, x);
  return view.getUint32(0);
}
/** A float32 written as its bit pattern, e.g. 0x420a6666. */
export function f32hex(x: number): string {
  return '0x' + f32bits(x).toString(16).padStart(8, '0');
}
/** Bit pattern mapped to an integer that orders like the float (sign-magnitude to two's complement). */
function ordered(x: number): number {
  const b = f32bits(x);
  return b >= 0x80000000 ? -(b - 0x80000000) : b;
}
/** How many float32 values lie between a and b, counting one end: 0 when equal, 1 when adjacent. */
export function ulpDistance(a: number, b: number): number {
  return Math.abs(ordered(f32(a)) - ordered(f32(b)));
}
/** The shortest decimal that reads back as the same float32 (what C++17 to_chars prints). */
export function f32short(x: number): string {
  const v = f32(x);
  if (Number.isNaN(v)) return 'NaN';
  if (v === Infinity) return '+∞';
  if (v === -Infinity) return '−∞';
  for (let p = 1; p <= 9; p++) {
    const s = v.toPrecision(p);
    if (f32(Number(s)) === v) return tidy(s);
  }
  return tidy(v.toPrecision(9));
}
/** A float64 value for display: up to 10 significant digits, no trailing zeros. */
export function f64short(x: number): string {
  if (Number.isNaN(x)) return 'NaN';
  if (x === Infinity) return '+∞';
  if (x === -Infinity) return '−∞';
  return tidy(String(Number(x.toPrecision(10))));
}
function tidy(s: string): string {
  // toPrecision may give "7.00" or "1.20e+5": drop trailing zeros of the mantissa, use a real minus sign.
  let [m, e] = s.split('e');
  if (m.includes('.')) m = m.replace(/0+$/, '').replace(/\.$/, '');
  return (m + (e ? 'e' + e : '')).replace(/^-/, '−');
}

// ---------------------------------------------------------------- 10.1: the sequential fold
export type FoldOpId = 'sum' | 'product' | 'max' | 'min' | 'argmax';
export type StartId = 'identity' | 'zero' | 'first' | 'negmax';

export interface FoldOpInfo {
  id: FoldOpId;
  label: string;
  /** The identity value (p. 222); for argmax, the identity of its value part. */
  identity: number;
  identityLabel: string;
}
export const FOLD_OPS: Record<FoldOpId, FoldOpInfo> = {
  sum: {
    id: 'sum',
    get label() {
      return tr('sum', 'tổng');
    },
    identity: 0, identityLabel: '0',
  },
  product: {
    id: 'product',
    get label() {
      return tr('product', 'tích');
    },
    identity: 1, identityLabel: '1',
  },
  max: { id: 'max', label: 'max', identity: -Infinity, identityLabel: '−∞' },
  min: { id: 'min', label: 'min', identity: Infinity, identityLabel: '+∞' },
  argmax: {
    id: 'argmax', label: 'argmax', identity: -Infinity, // i18n-ignore: operator name
    get identityLabel() {
      return tr('(−∞, none)', '(−∞, không có)');
    },
  },
};
export const START_LABEL: Record<StartId, string> = {
  get identity() {
    return tr("the operator's identity", 'phần tử đơn vị của toán tử');
  },
  zero: '0',
  get first() {
    return tr('element 0', 'phần tử 0');
  },
  negmax: '−FLT_MAX',
};

/** The book's example list (p. 222) and the explorer's presets. */
export const BOOK_LIST = [7.0, 2.1, 5.3, 9.0, 11.2];
export const NEGATIVE_ROW = [-4.2, -1.3, -6.8, -0.9, -2.5, -3.7, -1.1, -5.1];
export const HUGE_ROW = [1.2, -0.4, 2.3, 31.7, 0.8, -1.5, 1.9, 0.3];

export interface FoldStep {
  /** Index of the element combined in this iteration. */
  i: number;
  x: number;
  /** The accumulator after the iteration. */
  acc: number;
  /** For argmax: the index held by the accumulator (-1 = none). */
  index: number;
}
export interface FoldTrace {
  op: FoldOpId;
  start: StartId;
  precision: 'f32' | 'f64';
  init: { value: number; index: number };
  /** The first i the loop visits: 0, or 1 when it starts from element 0 (footnote 1, p. 222). */
  from: number;
  steps: FoldStep[];
  result: number;
  resultIndex: number;
  /** The right answer, in float64 (the first maximal index for argmax). */
  truth: number;
  truthIndex: number;
  ok: boolean;
}

/** max and min as CUDA's fmaxf and fminf: a NaN operand counts as missing (ch10 facts A10). */
export const fmax = (a: number, b: number) => (Number.isNaN(a) ? b : Number.isNaN(b) ? a : b > a ? b : a);
export const fmin = (a: number, b: number) => (Number.isNaN(a) ? b : Number.isNaN(b) ? a : b < a ? b : a);

function scalarOp(op: Exclude<FoldOpId, 'argmax'>): (a: number, b: number) => number {
  switch (op) {
    case 'sum':
      return (a, b) => a + b;
    case 'product':
      return (a, b) => a * b;
    case 'max':
      return fmax;
    case 'min':
      return fmin;
  }
}

function startValue(op: FoldOpId, start: StartId, values: number[]): { value: number; index: number; from: number } {
  if (start === 'first') return { value: values[0], index: values.length ? 0 : -1, from: 1 };
  const value = start === 'identity' ? FOLD_OPS[op].identity : start === 'zero' ? 0 : -FLT_MAX;
  return { value, index: -1, from: 0 };
}

/** The right answer for a list, computed in float64 with the operator's own identity. */
export function truthOf(values: number[], op: FoldOpId): { value: number; index: number } {
  if (op === 'argmax') {
    let best = -1;
    values.forEach((x, i) => {
      if (best < 0 || x > values[best]) best = i;
    });
    return { value: best < 0 ? -Infinity : values[best], index: best };
  }
  const f = scalarOp(op);
  return { value: values.reduce(f, FOLD_OPS[op].identity), index: -1 };
}

/** Is a computed result the right answer? Exact for max, min and argmax; within float32 rounding for sum and product. */
export function isRight(op: FoldOpId, values: number[], result: number, index: number, truth: { value: number; index: number }): boolean {
  if (op === 'argmax') return index === truth.index && f32(result) === f32(truth.value);
  if (op === 'max' || op === 'min') return f32(result) === f32(truth.value);
  if (!Number.isFinite(result)) return false;
  const scale = op === 'sum' ? values.reduce((s, x) => s + Math.abs(x), 0) : Math.abs(truth.value);
  return Math.abs(result - truth.value) <= 1e-5 * scale + 1e-30;
}

/**
 * Fig. 10.2: acc = start; for each element, acc = Operator(acc, input[i]). With
 * precision 'f32' the inputs and every intermediate are rounded to float32, as a C
 * float loop does. argmax keeps (value, index) and replaces it only on a strictly
 * larger value, so the lowest index wins a tie.
 */
export function foldSequential(values: number[], op: FoldOpId, start: StartId = 'identity', precision: 'f32' | 'f64' = 'f32'): FoldTrace {
  const r = precision === 'f32' ? f32 : (x: number) => x;
  const xs = values.map(r);
  const s = startValue(op, start, xs);
  let acc = r(s.value);
  let index = s.index;
  const steps: FoldStep[] = [];
  for (let i = s.from; i < xs.length; i++) {
    const x = xs[i];
    if (op === 'argmax') {
      if (x > acc) {
        acc = x;
        index = i;
      }
    } else {
      acc = r(scalarOp(op)(acc, x));
    }
    steps.push({ i, x, acc, index });
  }
  const truth = truthOf(values, op);
  return {
    op,
    start,
    precision,
    init: { value: r(s.value), index: s.index },
    from: s.from,
    steps,
    result: acc,
    resultIndex: index,
    truth: truth.value,
    truthIndex: truth.index,
    ok: isRight(op, values, acc, index, truth),
  };
}

// ---------------------------------------------------------------- seeded data
/** mulberry32: a small, fast, seedable generator. Same seed, same numbers, everywhere. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** Standard normal values (Box-Muller). */
export function normals(n: number, seed: number): number[] {
  const u = rng(seed);
  const out: number[] = [];
  while (out.length < n) {
    const a = Math.max(u(), 1e-12);
    const b = u();
    const m = Math.sqrt(-2 * Math.log(a));
    out.push(m * Math.cos(2 * Math.PI * b));
    if (out.length < n) out.push(m * Math.sin(2 * Math.PI * b));
  }
  return out;
}

/** A row of n logits for the fold explorer: standard normal, rounded to two decimals. */
export function randomRow(n: number, seed: number): number[] {
  return normals(n, seed).map((x) => Math.round(x * 100) / 100);
}

/**
 * The trap of section 10.1, counted: run the loop with a given start value on many
 * random rows (n standard-normal logits each) and count the rows where it is wrong.
 */
export function randomRowTest(op: FoldOpId, start: StartId, rows = 1000, n = 8, seed = 2026): { rows: number; wrong: number } {
  let wrong = 0;
  for (let k = 0; k < rows; k++) if (!foldSequential(randomRow(n, seed + k), op, start).ok) wrong++;
  return { rows, wrong };
}

export type Dist = 'uniform' | 'softmax' | 'wide';
export const DIST_LABEL: Record<Dist, string> = {
  get uniform() {
    return tr('uniform in [−1, 1]', 'phân phối đều trong [−1, 1]');
  },
  get softmax() {
    return tr('softmax terms, one large logit', 'các số hạng softmax, một logit lớn');
  },
  get wide() {
    return tr('wide-range mix', 'hỗn hợp trải rộng');
  },
};
/**
 * n float32 values from a seeded generator.
 * - uniform: uniform in [-1, 1];
 * - softmax: the terms a softmax denominator adds, exp(z_i - max z), for logits z with
 *   standard deviation 2 and one logit 12 above the rest's scale: one term is 1, most are tiny;
 * - wide: random sign, magnitude 10^u with u uniform in [-3, 3], so terms cancel.
 */
export function generate(n: number, dist: Dist, seed: number): number[] {
  const u = rng(seed);
  if (dist === 'uniform') return Array.from({ length: n }, () => f32(2 * u() - 1));
  if (dist === 'wide') return Array.from({ length: n }, () => f32((u() < 0.5 ? -1 : 1) * 10 ** (6 * u() - 3)));
  const z = normals(n, seed ^ 0x9e3779b9).map((x) => 2 * x);
  z[Math.floor(u() * n)] += 12;
  const m = Math.max(...z);
  return z.map((x) => f32(Math.exp(x - m)));
}

// ---------------------------------------------------------------- 10.2: trees
export type Shape = 'loop' | 'neighbours' | 'halves';
export const SHAPES: Shape[] = ['loop', 'neighbours', 'halves'];
export const SHAPE_LABEL: Record<Shape, string> = {
  get loop() {
    return tr('the loop', 'vòng lặp');
  },
  get neighbours() {
    return tr('neighbours', 'lân cận');
  },
  get halves() {
    return tr('halves', 'chia đôi');
  },
};

export interface TreeNode<T> {
  value: T;
  /** 0 for the leaves, then one level per step. */
  level: number;
  /** Position across the drawing, in leaf widths (leaf k sits at k). */
  x: number;
  /** Leaf index, for a leaf. */
  leaf: number | null;
  kids: [TreeNode<T>, TreeNode<T>] | null;
  /** The leaves this node combines, in operand order. */
  leaves: number[];
  /** The parenthesised expression in the style of p. 225. */
  expr: string;
}
export interface Tree<T> {
  shape: Shape;
  n: number;
  /** levels[0] are the leaves; levels[s] the operations of step s. */
  levels: TreeNode<T>[][];
  root: TreeNode<T>;
  expr: string;
  /** Operations performed. */
  work: number;
  /** Steps, if every operation of a step runs at once. */
  span: number;
  /** Operations in each step, step 1 first. */
  opsPerStep: number[];
}

/**
 * A reduction tree over the values in the given order. The combine function receives
 * its operands in order: (left, right). For 'halves' the left operand is the node at
 * the lower position, as in the convergent kernel's input[i] += input[i + stride].
 * n must be a power of two for 'neighbours' and 'halves'.
 */
export function buildTree<T>(
  values: T[],
  combine: (a: T, b: T) => T,
  shape: Shape,
  opts: { sym?: string; show?: (v: T) => string } = {},
): Tree<T> {
  const n = values.length;
  if (n < 1) throw new Error('empty input');
  if (shape !== 'loop' && (n & (n - 1)) !== 0) throw new Error('tree shapes need a power of two');
  const sym = opts.sym ?? '⊕';
  const show = opts.show ?? ((v: T) => String(v));
  const wrap = (t: TreeNode<T>) => (t.leaf !== null ? t.expr : `(${t.expr})`);
  const join = (level: number, a: TreeNode<T>, b: TreeNode<T>, x: number): TreeNode<T> => ({
    value: combine(a.value, b.value),
    level,
    x,
    leaf: null,
    kids: [a, b],
    leaves: [...a.leaves, ...b.leaves],
    expr: `${wrap(a)} ${sym} ${wrap(b)}`,
  });
  const leafExpr = (v: T) => {
    const s = show(v);
    return /^[−-]/.test(s) ? `(${s})` : s;
  };
  const levels: TreeNode<T>[][] = [
    values.map((v, k) => ({ value: v, level: 0, x: k, leaf: k, kids: null, leaves: [k], expr: leafExpr(v) })),
  ];
  if (shape === 'loop') {
    let acc = levels[0][0];
    for (let s = 1; s < n; s++) {
      acc = join(s, acc, levels[0][s], s);
      levels.push([acc]);
    }
  } else {
    let cur = levels[0];
    for (let s = 1; cur.length > 1; s++) {
      const m = cur.length;
      const next: TreeNode<T>[] = [];
      for (let j = 0; j < m / 2; j++) {
        const [a, b] = shape === 'neighbours' ? [cur[2 * j], cur[2 * j + 1]] : [cur[j], cur[j + m / 2]];
        next.push(join(s, a, b, (a.x + b.x) / 2));
      }
      levels.push(next);
      cur = next;
    }
  }
  const root = levels[levels.length - 1][0];
  const opsPerStep = levels.slice(1).map((l) => l.length);
  return {
    shape,
    n,
    levels,
    root,
    expr: root.expr,
    work: opsPerStep.reduce((a, b) => a + b, 0),
    span: opsPerStep.length,
    opsPerStep,
  };
}

/** The same result as buildTree(...).root.value, without building nodes: fast for long inputs. */
export function reduceShape<T>(values: T[], combine: (a: T, b: T) => T, shape: Shape): T {
  const n = values.length;
  if (shape === 'loop') {
    let acc = values[0];
    for (let i = 1; i < n; i++) acc = combine(acc, values[i]);
    return acc;
  }
  if ((n & (n - 1)) !== 0) throw new Error('tree shapes need a power of two');
  let cur = values.slice();
  while (cur.length > 1) {
    const m = cur.length;
    const next = new Array<T>(m / 2);
    for (let j = 0; j < m / 2; j++) next[j] = shape === 'neighbours' ? combine(cur[2 * j], cur[2 * j + 1]) : combine(cur[j], cur[j + m / 2]);
    cur = next;
  }
  return cur[0];
}

/** Work and span of a full tree over n inputs (n a power of two), pp. 225-226. */
export interface TreeStats {
  n: number;
  work: number;
  span: number;
  opsPerStep: number[];
  peak: number;
  average: number;
}
export function treeStats(n: number): TreeStats {
  const opsPerStep: number[] = [];
  for (let m = n / 2; m >= 1; m /= 2) opsPerStep.push(m);
  const work = n - 1;
  const span = opsPerStep.length;
  return { n, work, span, opsPerStep, peak: n / 2, average: work / span };
}

// ---------------------------------------------------------------- integer operators for the tree builder
export type IntOpId = 'sum' | 'max' | 'min' | 'product' | 'sub';
export interface IntOp {
  id: IntOpId;
  label: string;
  /** How the operator is written in an expression (p. 225 writes "max"). */
  sym: string;
  fn: (a: bigint, b: bigint) => bigint;
}
/** Exact integer arithmetic (BigInt), so nothing overflows or rounds. */
export const INT_OPS: Record<IntOpId, IntOp> = {
  sum: {
    id: 'sum',
    get label() {
      return tr('sum', 'tổng');
    },
    sym: '+', fn: (a, b) => a + b,
  },
  max: { id: 'max', label: 'max', sym: 'max', fn: (a, b) => (a > b ? a : b) },
  min: { id: 'min', label: 'min', sym: 'min', fn: (a, b) => (a < b ? a : b) },
  product: {
    id: 'product',
    get label() {
      return tr('product', 'tích');
    },
    sym: '×', fn: (a, b) => a * b,
  },
  sub: {
    id: 'sub',
    get label() {
      return tr('subtraction', 'phép trừ');
    },
    sym: '−', fn: (a, b) => a - b,
  },
};
/** Fig. 10.3: (a) the max tree, (b) the sum tree. */
export const FIG103A = [3, 1, 7, 0, 4, 1, 6, 3];
export const FIG103B = [4, 7, 2, 3, 8, 5, 9, 6];

/** An integer for display: exact up to 9 digits, then in scientific form. */
export function showBig(v: bigint): string {
  const s = (v < 0n ? -v : v).toString();
  const body = s.length <= 9 ? s : `${s[0]}.${s.slice(1, 3)}e${s.length - 1}`;
  return (v < 0n ? '−' : '') + body;
}

export function intTree(values: number[], op: IntOpId, shape: Shape): Tree<bigint> {
  const o = INT_OPS[op];
  return buildTree(values.map((v) => BigInt(v)), o.fn, shape, { sym: o.sym, show: showBig });
}

/**
 * "join": write the operands side by side. It is associative ((ab)c = a(bc)) but not
 * commutative (ab is not ba), so the result spells out the order in which the leaves
 * were combined. Single-digit leaves are joined without a separator, others with commas.
 * Subtraction cannot play this role: in both tree orders a leaf ends up negated once
 * for every 1 bit in its index, so the two trees always agree with each other.
 */
export function joinTree(values: number[], shape: Shape): Tree<string> {
  const short = values.every((v) => Number.isInteger(v) && v >= 0 && v <= 9);
  const sep = short ? '' : ',';
  return buildTree(values.map((v) => (v < 0 ? '−' + -v : String(v))), (a, b) => a + sep + b, shape, { sym: '·' });
}

/** Every operator of the tree builder, numeric or not, as display strings. */
export type BuilderOpId = IntOpId | 'join';
export const BUILDER_OPS: { id: BuilderOpId; label: string }[] = [
  {
    id: 'sum',
    get label() {
      return tr('sum', 'tổng');
    },
  },
  { id: 'max', label: 'max' }, // i18n-ignore: operator name
  { id: 'min', label: 'min' }, // i18n-ignore: operator name
  {
    id: 'product',
    get label() {
      return tr('product', 'tích');
    },
  },
  {
    id: 'sub',
    get label() {
      return tr('subtraction', 'phép trừ');
    },
  },
  {
    id: 'join',
    get label() {
      return tr('join', 'ghép');
    },
  },
];
/** The tree builder's tree, with every node's value as text (exact integers, or the joined string). */
export function builderTree(values: number[], op: BuilderOpId, shape: Shape): Tree<string> {
  if (op === 'join') return joinTree(values, shape);
  return mapTree(intTree(values, op, shape), showBig);
}

/** The same tree with every value mapped; nodes stay shared between levels and kids. */
export function mapTree<A, B>(t: Tree<A>, f: (a: A) => B): Tree<B> {
  const memo = new Map<TreeNode<A>, TreeNode<B>>();
  const m = (nd: TreeNode<A>): TreeNode<B> => {
    let r = memo.get(nd);
    if (!r) {
      r = { ...nd, value: f(nd.value), kids: nd.kids ? [m(nd.kids[0]), m(nd.kids[1])] : null };
      memo.set(nd, r);
    }
    return r;
  };
  return { ...t, levels: t.levels.map((l) => l.map(m)), root: m(t.root) };
}
/** Exact comparison key of a builder tree's result (display strings of exact values are exact up to 9 digits). */
export function builderResult(values: number[], op: BuilderOpId, shape: Shape): string {
  if (op === 'join') return joinTree(values, shape).root.value;
  return intTree(values, op, shape).root.value.toString();
}

// ---------------------------------------------------------------- float32 sums in three orders
export const add32 = (a: number, b: number) => f32(a + b);

/** The float32 sum of float32 values in one order. */
export function evalFloat32(values: number[], shape: Shape): number {
  return reduceShape(values.map(f32), add32, shape);
}

/** Neumaier's compensated sum in float64: a reference far more accurate than any float32 order. */
export function sum64(values: number[]): number {
  let s = 0;
  let c = 0;
  for (const x of values) {
    const t = s + x;
    c += Math.abs(s) >= Math.abs(x) ? s - t + x : x - t + s;
    s = t;
  }
  return s + c;
}

export interface OrdersResult {
  sums: Record<Shape, number>;
  /** float64 reference. */
  ref: number;
  /** ULPs between each float32 sum and the float32 nearest the reference. */
  ulps: Record<Shape, number>;
  /** How many different float32 results the three orders gave. */
  distinct: number;
  /** The orders whose result is nearest the reference (several on a tie). */
  closest: Shape[];
}
export function compareOrders(values: number[]): OrdersResult {
  const sums = { loop: 0, neighbours: 0, halves: 0 } as Record<Shape, number>;
  const ulps = { loop: 0, neighbours: 0, halves: 0 } as Record<Shape, number>;
  const ref = sum64(values.map(f32));
  for (const s of SHAPES) {
    sums[s] = evalFloat32(values, s);
    ulps[s] = ulpDistance(sums[s], f32(ref));
  }
  const best = Math.min(...SHAPES.map((s) => Math.abs(sums[s] - ref)));
  return {
    sums,
    ref,
    ulps,
    distinct: new Set(SHAPES.map((s) => f32bits(sums[s]))).size,
    closest: SHAPES.filter((s) => Math.abs(sums[s] - ref) === best),
  };
}

/** Over many seeds: how often the orders disagree, and how often each is (one of) the closest. */
export interface OrdersSurvey {
  trials: number;
  /** Trials in which at least two orders gave different bits. */
  disagree: number;
  /** Trials in which an order was nearest the reference (alone or tied). */
  closest: Record<Shape, number>;
  /** Trials in which an order was strictly nearer than both others. */
  alone: Record<Shape, number>;
}
export function surveyOrders(n: number, dist: Dist, trials: number, seed0 = 1): OrdersSurvey {
  const closest = { loop: 0, neighbours: 0, halves: 0 } as Record<Shape, number>;
  const alone = { loop: 0, neighbours: 0, halves: 0 } as Record<Shape, number>;
  let disagree = 0;
  for (let k = 0; k < trials; k++) {
    const r = compareOrders(generate(n, dist, seed0 + k));
    if (r.distinct > 1) disagree++;
    for (const s of r.closest) closest[s]++;
    if (r.closest.length === 1) alone[r.closest[0]]++;
  }
  return { trials, disagree, closest, alone };
}

/**
 * The scene's float32 example: the first seed from `seed0` whose n values, from the
 * given distribution, give different float32 sums for the loop and for Fig. 10.3's
 * neighbours tree.
 */
export function disagreeingSet(n = 8, dist: Dist = 'uniform', seed0 = 1): { seed: number; values: number[]; loop: number; tree: number } {
  for (let seed = seed0; ; seed++) {
    const values = generate(n, dist, seed);
    const loop = evalFloat32(values, 'loop');
    const tree = evalFloat32(values, 'neighbours');
    if (f32bits(loop) !== f32bits(tree)) return { seed, values, loop, tree };
  }
}

// ---------------------------------------------------------------- (value, index) pairs
export interface Pair {
  v: number;
  i: number;
}
export type PairRule = 'lower' | 'left' | 'higher' | 'nanfirst' | 'fmax' | 'cmp';
export interface PairRuleInfo {
  id: PairRule;
  label: string;
  /** An argmax rule (the index matters) or a plain max (only the value does). */
  argmax: boolean;
  /** Where the rule comes from, for the explorer's note. */
  note: string;
}
export const PAIR_RULES: Record<PairRule, PairRuleInfo> = {
  lower: {
    id: 'lower',
    get label() {
      return tr('larger value; ties to the lower index', 'giá trị lớn hơn; khi bằng nhau thì lấy chỉ số thấp hơn');
    },
    argmax: true,
    get note() {
      return tr("hipCUB's ArgMax functor, written with comparisons (ch10 facts B5)", 'functor ArgMax của hipCUB, viết bằng các phép so sánh (ch10 facts B5)');
    },
  },
  left: {
    id: 'left',
    get label() {
      return tr('larger value; keep the left operand on ties', 'giá trị lớn hơn; khi bằng nhau thì giữ toán hạng bên trái');
    },
    argmax: true,
    get note() {
      return tr('a "replace only if strictly greater" step, used inside a tree', 'một bước "chỉ thay khi lớn hơn hẳn", dùng bên trong một cây');
    },
  },
  higher: {
    id: 'higher',
    get label() {
      return tr('larger value; ties to the higher index', 'giá trị lớn hơn; khi bằng nhau thì lấy chỉ số cao hơn');
    },
    argmax: true,
    get note() {
      return tr('consistent, but not the index PyTorch documents', 'nhất quán, nhưng không phải chỉ số mà PyTorch ghi trong tài liệu');
    },
  },
  nanfirst: {
    id: 'nanfirst',
    get label() {
      return tr('NaN wins, then larger value; ties to the lower index', 'NaN thắng, rồi đến giá trị lớn hơn; khi bằng nhau thì lấy chỉ số thấp hơn');
    },
    argmax: true,
    get note() {
      return tr("PyTorch's GreaterOrNan and MLX's ArgMax (ch10 facts D3, D4)", 'GreaterOrNan của PyTorch và ArgMax của MLX (ch10 facts D3, D4)');
    },
  },
  fmax: {
    id: 'fmax',
    get label() {
      return tr('fmaxf (NaN as missing)', 'fmaxf (NaN coi như thiếu)');
    },
    argmax: false,
    get note() {
      return tr("CUDA's fmaxf and max(float, float) (A10)", 'fmaxf và max(float, float) của CUDA (A10)');
    },
  },
  cmp: {
    id: 'cmp', label: 'a > b ? a : b', argmax: false,
    get note() {
      return tr('a comparison, as many hand-written kernels do (A10)', 'một phép so sánh, như nhiều kernel viết tay vẫn làm (A10)');
    },
  },
};

const isNaN_ = Number.isNaN;
/** The combine function of a rule, called as combine(left, right). */
export function pairCombine(rule: PairRule): (a: Pair, b: Pair) => Pair {
  switch (rule) {
    case 'lower':
      return (a, b) => (b.v > a.v || (b.v === a.v && b.i < a.i) ? b : a);
    case 'left':
      return (a, b) => (b.v > a.v ? b : a);
    case 'higher':
      return (a, b) => (b.v > a.v || (b.v === a.v && b.i > a.i) ? b : a);
    case 'nanfirst': {
      // PyTorch's GreaterOrNan(a, b, idx_a, idx_b): true means keep a.
      const keepA = (a: Pair, b: Pair) => {
        if (isNaN_(a.v)) return isNaN_(b.v) ? a.i < b.i : true;
        return a.v === b.v ? a.i < b.i : a.v > b.v;
      };
      return (a, b) => (keepA(a, b) ? a : b);
    }
    case 'fmax':
      return (a, b) => (isNaN_(a.v) ? b : isNaN_(b.v) ? a : b.v > a.v ? b : a);
    case 'cmp':
      return (a, b) => (a.v > b.v ? a : b);
  }
}

export const pairs = (row: number[]): Pair[] => row.map((v, i) => ({ v, i }));

/** The winner of a row under a rule, in each of the three orders. */
export function winners(row: number[], rule: PairRule): Record<Shape, Pair> {
  const c = pairCombine(rule);
  const p = pairs(row);
  return { loop: reduceShape(p, c, 'loop'), neighbours: reduceShape(p, c, 'neighbours'), halves: reduceShape(p, c, 'halves') };
}
/** Do all three orders agree? For an argmax rule on value and index; for a plain max on the value only. */
export function orderIndependent(row: number[], rule: PairRule): boolean {
  const w = winners(row, rule);
  const key = (p: Pair) => (PAIR_RULES[rule].argmax ? `${f32bits(p.v)}:${p.i}` : String(f32bits(p.v)));
  return new Set(SHAPES.map((s) => key(w[s]))).size === 1;
}

/** The tie explorer's default row: equal maxima at indices 1 and 4, where 'keep left' depends on the order. */
export const TIE_ROW = [0.6, 4.0, -1.2, 2.5, 4.0, 1.1, 3.3, -0.4];
/** A row of n logits (two decimals, below 4.0) with 4.0 planted at the given indices and an optional NaN. */
export function tieRow(n: number, maxima: number[], nanAt: number | null = null, seed = 7): number[] {
  const base = n === TIE_ROW.length ? TIE_ROW.map((v) => (v === 4.0 ? 1.8 : v)) : randomRow(n, seed).map((v) => Math.min(v, 3.5));
  const row = base.map((v, k) => (maxima.includes(k) ? 4.0 : v));
  if (nanAt !== null && nanAt >= 0 && nanAt < n) row[nanAt] = NaN;
  return row;
}

// ---------------------------------------------------------------- the sidebar: a knockout tournament
export interface Tournament {
  teams: number;
  venues: number;
  rounds: number;
  gamesPerRound: number[];
  daysPerRound: number[];
  days: number;
}
/**
 * A knockout tournament is a max tree (pp. 226-227). Each venue hosts one game a day
 * and a round finishes before the next begins, so a round of g games takes
 * ceil(g / venues) days.
 */
export function tournament(teams: number, venues: number): Tournament {
  const gamesPerRound: number[] = [];
  for (let g = teams / 2; g >= 1; g /= 2) gamesPerRound.push(g);
  const daysPerRound = gamesPerRound.map((g) => Math.ceil(g / venues));
  return { teams, venues, rounds: gamesPerRound.length, gamesPerRound, daysPerRound, days: daysPerRound.reduce((a, b) => a + b, 0) };
}
