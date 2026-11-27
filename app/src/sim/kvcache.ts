// KV caching for chapter 20, section 20.4 (book pp. 488–492; notes/ch20.md §20.4).
//
// Four things, each used by an interactive or by the "decode-stream" scene:
//
//   1. diffStep(n, d, seed): a small single-head attention (book Fig. 20.2 / Eq. 20.1)
//      run on n tokens and again on n + 1 tokens, and every matrix diffed cell by cell.
//      This is the computed answer to "which cells change?" (Fig. 20.5): the new bottom
//      row of X, Q, K, V, S, P and O; a new column of S that is masked (minus infinity)
//      everywhere except (n, n), and of P that is exactly zero there; and every older row
//      bit-identical, because a masked entry adds exp(-inf) = 0 to the softmax
//      denominator ("do not dilute", p. 490).
//   2. Per-token FLOPs of the baseline (recompute the five products of p. 488 over all
//      n rows) and of cached decode (Fig. 20.6: three vector-matrix projections and two
//      length-n vector-matrix products), and the bytes the cache holds.
//   3. Arithmetic intensity of the four products the roofline explorer plots, with
//      compulsory traffic only (read both operands once, write the result once).
//   4. Byte and FLOP counters of one decode step of one layer, phase by phase, for the
//      scene.
//
// Conventions. D is the width of a token's row of Q, K and V summed over the heads,
// the book's "hidden dimension" h_q x d (p. 505). The book draws one head with
// d x d weights (Fig. 20.2); with h heads of width d each, every count below is the same
// as for one head of width D, so the formulas take D. FLOPs count a multiply-add as 2.
// The output projection, the feed-forward layer and the softmax are not counted, as in the
// book's count of five products per layer (p. 488) and its Eq. 20.9 (pp. 506–507).
//
// Erasable TypeScript only and no imports: the browser checks import this file directly.

// ------------------------------------------------------------------ the small problem

import { tr } from '../lib/locale.ts';

export type MatName = 'X' | 'Q' | 'K' | 'V' | 'S' | 'P' | 'O';
export const MAT_NAMES: MatName[] = ['X', 'Q', 'K', 'V', 'S', 'P', 'O'];

export interface Mat {
  rows: number;
  cols: number;
  /** Row-major values. S holds the masked, scaled logits: minus infinity above the diagonal. */
  v: number[];
}

/** How a cell of the (n + 1)-token matrix relates to the n-token one. */
export type CellKind = 'same' | 'changed' | 'new' | 'masked';

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const X_VALUES = [-1, -0.5, 0, 0.5, 1];
const W_VALUES = [-1, -0.5, 0, 0.5, 1];

export interface Problem {
  d: number;
  /** maxTokens x d: the embeddings of every token the problem can hold. */
  X: number[][];
  Wq: number[][];
  Wk: number[][];
  Wv: number[][];
}

/** A deterministic problem with room for maxTokens tokens of width d. */
export function makeProblem(maxTokens: number, d: number, seed = 1): Problem {
  const r = mulberry32(seed * 7919 + d);
  const pick = (vals: number[]) => vals[Math.floor(r() * vals.length)];
  const mat = (rows: number, cols: number, vals: number[]) =>
    Array.from({ length: rows }, () => Array.from({ length: cols }, () => pick(vals)));
  return { d, X: mat(maxTokens, d, X_VALUES), Wq: mat(d, d, W_VALUES), Wk: mat(d, d, W_VALUES), Wv: mat(d, d, W_VALUES) };
}

function matmul(A: number[][], B: number[][]): number[][] {
  const m = A.length;
  const k = B.length;
  const n = B[0].length;
  const C: number[][] = [];
  for (let i = 0; i < m; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let t = 0; t < k; t++) s += A[i][t] * B[t][j];
      row.push(s);
    }
    C.push(row);
  }
  return C;
}

const flat = (A: number[][]): Mat => ({ rows: A.length, cols: A[0]?.length ?? 0, v: A.flat() });

/**
 * Reference causal attention on the first n tokens, every intermediate kept (Eq. 20.1 with
 * the 1/sqrt(d) scale, Eq. 20.2 with the row maximum subtracted). The mask is applied as
 * the book's M: minus infinity added above the diagonal, so exp gives exactly 0 there.
 */
export function attention(prob: Problem, n: number): Record<MatName, Mat> {
  const X = prob.X.slice(0, n);
  const Q = matmul(X, prob.Wq);
  const K = matmul(X, prob.Wk);
  const V = matmul(X, prob.Wv);
  const scale = 1 / Math.sqrt(prob.d);
  const S: number[][] = [];
  const P: number[][] = [];
  for (let r = 0; r < n; r++) {
    const row: number[] = [];
    for (let c = 0; c < n; c++) {
      let s = 0;
      for (let t = 0; t < prob.d; t++) s += Q[r][t] * K[c][t];
      row.push(c > r ? -Infinity : scale * s);
    }
    S.push(row);
    let m = -Infinity;
    for (const x of row) m = Math.max(m, x);
    let sum = 0;
    for (const x of row) sum += Math.exp(x - m);
    P.push(row.map((x) => Math.exp(x - m) / sum));
  }
  const O = matmul(P, V);
  return { X: flat(X), Q: flat(Q), K: flat(K), V: flat(V), S: flat(S), P: flat(P), O: flat(O) };
}

export interface StepDiff {
  n: number;
  d: number;
  before: Record<MatName, Mat>;
  after: Record<MatName, Mat>;
  /** Per matrix, one entry per cell of the (n + 1)-token matrix, row-major. */
  kind: Record<MatName, CellKind[]>;
}

/**
 * Run attention on n tokens and on n + 1 and classify every cell of the larger result.
 * "same" means bit-identical (Object.is), not merely close.
 */
export function diffStep(n: number, d: number, seed = 1): StepDiff {
  const prob = makeProblem(n + 1, d, seed);
  const before = attention(prob, n);
  const after = attention(prob, n + 1);
  const kind = {} as Record<MatName, CellKind[]>;
  for (const name of MAT_NAMES) {
    const a = after[name];
    const b = before[name];
    const out: CellKind[] = [];
    for (let r = 0; r < a.rows; r++)
      for (let c = 0; c < a.cols; c++) {
        const x = a.v[r * a.cols + c];
        if (r < b.rows && c < b.cols) out.push(Object.is(x, b.v[r * b.cols + c]) ? 'same' : 'changed');
        else if (name === 'S' && x === -Infinity) out.push('masked');
        else if (name === 'P' && c > r) out.push('masked');
        else out.push('new');
      }
    kind[name] = out;
  }
  return { n, d, before, after, kind };
}

/** Cells that need arithmetic when token n arrives: everything classified "new". */
export function mustCompute(diff: StepDiff): Record<MatName, Set<number>> {
  const out = {} as Record<MatName, Set<number>>;
  for (const name of MAT_NAMES)
    out[name] = new Set(diff.kind[name].flatMap((k, i) => (k === 'new' || k === 'changed' ? [i] : [])));
  return out;
}

export interface Score {
  /** Cells that must be computed and were marked. */
  hit: number;
  /** Cells that must be computed and were not marked. */
  missed: number;
  /** Unchanged cells that were marked. */
  extraSame: number;
  /** Masked cells that were marked: they exist in the bigger matrix, but their value is known. */
  extraMasked: number;
  /** Cells that must be computed, in all. */
  needed: number;
}

/** Score a reader's marks (per matrix, the indices of marked cells) against the diff. */
export function scoreMarks(diff: StepDiff, marks: Partial<Record<MatName, Set<number>>>): Score {
  const s: Score = { hit: 0, missed: 0, extraSame: 0, extraMasked: 0, needed: 0 };
  for (const name of MAT_NAMES) {
    const m = marks[name] ?? new Set<number>();
    diff.kind[name].forEach((k, i) => {
      const marked = m.has(i);
      if (k === 'new' || k === 'changed') {
        s.needed++;
        if (marked) s.hit++;
        else s.missed++;
      } else if (marked) {
        if (k === 'masked') s.extraMasked++;
        else s.extraSame++;
      }
    });
  }
  return s;
}

// ------------------------------------------------------------------ FLOPs: recompute or cache

export interface StepFlops {
  projections: number;
  scores: number;
  values: number;
  total: number;
}

/**
 * The baseline at context length n (n rows of X, the newest token included): the five
 * products of p. 488 over every row, per layer. X W_Q, X W_K, X W_V: 3 x 2 n D^2.
 * Q K^T: 2 n^2 D. P V: 2 n^2 D.
 */
export function baselineStep(n: number, D: number, layers = 1): StepFlops {
  const projections = layers * 3 * 2 * n * D * D;
  const scores = layers * 2 * n * n * D;
  const values = layers * 2 * n * n * D;
  return { projections, scores, values, total: projections + scores + values };
}

/**
 * Cached decode at context length n (Fig. 20.6): q', k', v' from one row (3 x 2 D^2),
 * q' K^T against n keys (2 n D), p' V over n values (2 n D), per layer.
 */
export function cachedStep(n: number, D: number, layers = 1): StepFlops {
  const projections = layers * 3 * 2 * D * D;
  const scores = layers * 2 * n * D;
  const values = layers * 2 * n * D;
  return { projections, scores, values, total: projections + scores + values };
}

/**
 * Generating T tokens after a prompt of N tokens: generation iteration i runs on N + i
 * rows (the book's toy example: a 7-token prompt, then 8, 9, ... rows, p. 486).
 */
export function generate(N: number, T: number, D: number, layers = 1): { baseline: number; cached: number } {
  let baseline = 0;
  let cached = 0;
  for (let i = 1; i <= T; i++) {
    baseline += baselineStep(N + i, D, layers).total;
    cached += cachedStep(N + i, D, layers).total;
  }
  return { baseline, cached };
}

/** Bytes the K and V caches of every layer hold for n tokens at p bytes per element (Eq. 20.7 x n). */
export function cacheBytes(n: number, D: number, layers: number, p: number): number {
  return 2 * n * D * layers * p;
}

// ------------------------------------------------------------------ arithmetic intensity

/**
 * Intensity of an (m x k) times (k x n) product with compulsory traffic only: read both
 * operands once, write the result once, p bytes per element. FLOP per byte.
 */
export function gemmAI(m: number, k: number, n: number, p: number): number {
  return (2 * m * k * n) / (p * (m * k + k * n + m * n));
}

export interface AIPoint {
  key: 'prefill-proj' | 'prefill-qk' | 'decode-proj' | 'decode-qk';
  label: string;
  /** FLOP per byte. */
  ai: number;
  flops: number;
  bytes: number;
}

function point(key: AIPoint['key'], label: string, m: number, k: number, n: number, p: number): AIPoint {
  return { key, label, ai: gemmAI(m, k, n, p), flops: 2 * m * k * n, bytes: p * (m * k + k * n + m * n) };
}

/**
 * The four products of one head the roofline explorer plots, for a prompt of n tokens and
 * head width d. Prefill: X (n x d) W (d x d), and Q (n x d) K^T (d x n) writing S (n x n).
 * Decode: x' (1 x d) W (d x d), and q' (1 x d) K^T (d x n) writing s' (1 x n).
 */
export function aiPoints(n: number, d: number, p: number): AIPoint[] {
  return [
    point('prefill-proj', 'prefill: X W (GEMM)', n, d, d, p), // i18n-ignore: formula with glossary-kept terms, identical in Vietnamese
    point('prefill-qk', 'prefill: Q Kᵀ (GEMM)', n, d, n, p), // i18n-ignore: formula with glossary-kept terms, identical in Vietnamese
    point('decode-proj', "decode: x′ W (GEMV)", 1, d, d, p), // i18n-ignore: formula with glossary-kept terms, identical in Vietnamese
    point('decode-qk', "decode: q′ Kᵀ (GEMV)", 1, d, n, p), // i18n-ignore: formula with glossary-kept terms, identical in Vietnamese
  ];
}

/** Prefill Q K^T intensity: rises with n towards 2d/p. */
export const aiPrefill = (n: number, d: number, p: number) => gemmAI(n, d, n, p);
/** Decode q' K^T intensity: about 2/p whatever n. */
export const aiDecode = (n: number, d: number, p: number) => gemmAI(1, d, n, p);

/**
 * One decode step's attention (q' K^T then p' V) reading a cache of n rows of width D:
 * 4 n D FLOPs over 2 n D elements read, plus q' read and o' written (Eq. 20.9's
 * accounting, with p bytes per element instead of the book's p = 1). FLOP per byte.
 */
export function decodeAttentionAI(n: number, D: number, p: number): number {
  return (4 * n * D) / (p * (2 * n * D + 2 * D));
}

// ------------------------------------------------------------------ one decode step, phase by phase

export type PhaseKey = 'proj' | 'append' | 'scores' | 'softmax' | 'values';

export interface Phase {
  key: PhaseKey;
  label: string;
  bytesRead: number;
  bytesWritten: number;
  flops: number;
}

/**
 * One decode step of one layer with h heads of width d (D = h d), n tokens in the cache
 * after the append, p bytes per element. The softmax is listed with its size (h n scores)
 * and no FLOPs or traffic, as the book's Eq. 20.9 ignores it and a fused kernel keeps the
 * scores on chip.
 */
export function decodePhases(n: number, h: number, d: number, p: number): Phase[] {
  const D = h * d;
  return [
    { key: 'proj', label: tr('projections', 'các phép chiếu'), bytesRead: p * (D + 3 * D * D), bytesWritten: p * D, flops: 3 * 2 * D * D },
    { key: 'append', label: tr('append k′, v′', 'nối thêm k′, v′'), bytesRead: 0, bytesWritten: p * 2 * D, flops: 0 },
    { key: 'scores', label: 'q′ Kᵀ', bytesRead: p * n * D, bytesWritten: 0, flops: 2 * n * D },
    { key: 'softmax', label: tr('softmax', 'softmax'), bytesRead: 0, bytesWritten: 0, flops: 0 },
    { key: 'values', label: 'p′ V', bytesRead: p * n * D, bytesWritten: p * D, flops: 2 * n * D },
  ];
}

export interface Totals {
  bytes: number;
  flops: number;
  /** FLOP per byte over the phases summed. */
  ai: number;
}

/** Cumulative totals after the first `upto` phases (all of them by default). */
export function phaseTotals(phases: Phase[], upto = phases.length): Totals {
  let bytes = 0;
  let flops = 0;
  for (const ph of phases.slice(0, upto)) {
    bytes += ph.bytesRead + ph.bytesWritten;
    flops += ph.flops;
  }
  return { bytes, flops, ai: bytes > 0 ? flops / bytes : 0 };
}
