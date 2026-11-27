// The online softmax of book §20.5 (pp. 493–495, Eqs. 20.3–20.6) for one row of
// attention scores, merged one tile at a time. Section 20.5's stepper "Online softmax,
// one tile at a time" and the scene "Merging softmax one tile at a time" run on it.
//
// What it models:
//   - One query row: N scores l_j (already scaled) and an N × dv matrix V.
//   - Causality by index: column j is visible to row r only if j <= r (the book's
//     loop-bound trick, Fig. 20.4; the labs mask the same way).
//   - Tiles of width bc, merged in a chosen order (left to right as the book's kernel
//     does, right to left, or a fixed shuffle).
//   - Two forms of the merge:
//       'deferred'   — what the code keeps (notes §20.5 [inference, checked];
//                      FlashAttention-2 §3.1.1, facts A7): the un-normalised
//                      Õ = Σ e^(l − m) v and D, rescaled by e^(m_old − m_new), with one
//                      division at the very end. The rescale of D happens only when the max
//                      rises (Fig. 20.14); the rescale of Õ is unconditional (Fig. 20.13).
//       'normalised' — Eq. 20.4 for D and Eq. 20.6 for o, as printed: each tile's partial
//                      output is normalised by its own D_B and merged with weights. Eq. 20.5
//                      is 0/0 for a tile with no visible column, so this form skips such tiles.
//   - Two starting values for the running max: minus infinity (FlashAttention-2's
//     Algorithm 1, facts A7) or -FLT_MAX/2 (llama.cpp, facts A5, D1).
//   - float32 arithmetic: every operation is rounded with Math.fround.
//
// What it leaves out: exp is JavaScript's Math.exp rounded to float32, not a GPU's expf
// (2 ULP on CUDA, ch10 A11) or Metal's; there are no threads, tiles of S in shared memory
// or warps here (flashkernel.ts has those).

import { tr } from '../lib/locale.ts';

const f = Math.fround;
export const FLT_MAX = 3.4028234663852886e38;
/** llama.cpp's start value for the running max, -FLT_MAX/2 (facts A5, D1), as a float32. */
export const SENTINEL = f(-FLT_MAX / 2);

export type TileOrder = 'ltr' | 'rtl' | 'shuffled';
export type InitMax = 'neg-inf' | 'sentinel';
export type Form = 'deferred' | 'normalised';

export const ORDER_LABEL: Record<TileOrder, string> = {
  get ltr() {
    return tr('left to right (the book)', 'trái sang phải (theo sách)');
  },
  get rtl() {
    return tr('right to left', 'phải sang trái');
  },
  get shuffled() {
    return tr('shuffled', 'xáo trộn');
  },
};
export const INIT_LABEL: Record<InitMax, string> = {
  get 'neg-inf'() {
    return tr('minus infinity (FlashAttention-2)', 'âm vô cực (FlashAttention-2)');
  },
  get sentinel() {
    return tr('-FLT_MAX/2 (llama.cpp)', '-FLT_MAX/2 (llama.cpp)');
  },
};
export const FORM_LABEL: Record<Form, string> = {
  get deferred() {
    return tr('deferred Õ, one division at the end (the code)', 'Õ chưa chia, một phép chia duy nhất ở cuối (như trong mã)');
  },
  get normalised() {
    return tr('normalised o, Eq. 20.6 at every step', 'o đã chuẩn hóa, Eq. 20.6 ở mọi bước');
  },
};

export const initValue = (init: InitMax) => (init === 'neg-inf' ? -Infinity : SENTINEL);

// ---------------------------------------------------------------- data
export type PresetId = 'demo' | 'random' | 'late' | 'offset';
export const PRESET_LABEL: Record<PresetId, string> = {
  get demo() {
    return tr('the scene’s row', 'hàng của cảnh minh họa');
  },
  get random() {
    return tr('random (fixed seed)', 'ngẫu nhiên (seed cố định)');
  },
  get late() {
    return tr('max arrives late', 'giá trị max đến muộn');
  },
  get offset() {
    return tr('large: every score + 100', 'lớn: mọi điểm số + 100');
  },
};

export const ROW_N = 12;
export const DV = 3;

/** A deterministic 12 × 3 V with small values, the same for every preset. */
export const V12: number[][] = Array.from({ length: ROW_N }, (_, j) =>
  Array.from({ length: DV }, (_, c) => Math.round(Math.sin(1.7 * j + 2.3 * c + 0.4) * 100) / 100),
);

/** The scene's row: tile 2 (of width 4) raises the max, tile 3 does not. */
export const DEMO_SCORES = [0.8, -0.4, 1.6, 0.2, 2.9, 0.5, -1.1, 1.7, 1.2, -0.7, 2.1, 0.3];
const LATE_SCORES = [-1.2, 0.3, -0.5, 0.9, 0.1, 1.4, -0.3, 1.9, 0.6, 2.4, 1.1, 4.0];

export function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function presetScores(p: PresetId): number[] {
  switch (p) {
    case 'demo':
      return DEMO_SCORES.slice();
    case 'late':
      return LATE_SCORES.slice();
    case 'offset':
      return DEMO_SCORES.map((s) => s + 100);
    case 'random': {
      const r = mulberry(2005);
      return Array.from({ length: ROW_N }, () => Math.round((r() * 6 - 3) * 10) / 10);
    }
  }
}

// ---------------------------------------------------------------- reference
/** Exact attention (float64) of one row over the given columns: softmax(l) · V. Null if none. */
export function exactOver(scores: number[], V: number[][], cols: number[]): number[] | null {
  if (!cols.length) return null;
  const m = Math.max(...cols.map((j) => scores[j]));
  let D = 0;
  const o = new Array(V[0].length).fill(0);
  for (const j of cols) {
    const e = Math.exp(scores[j] - m);
    D += e;
    for (let c = 0; c < o.length; c++) o[c] += e * V[j][c];
  }
  return o.map((x) => x / D);
}

// ---------------------------------------------------------------- Eq. 20.4 on its own
export interface Merge {
  m: number;
  D: number;
  /** e^(m_A − m_A∪B): multiplies D_A (the A term of Eq. 20.4). */
  aFactor: number;
  /** e^(m_B − m_A∪B): multiplies D_B (the B term). */
  bFactor: number;
}

/** Eq. 20.4 in float64: merge two partial (max, denominator) pairs over disjoint subsets. */
export function mergeD(mA: number, DA: number, mB: number, DB: number): Merge {
  const m = Math.max(mA, mB);
  const aFactor = Math.exp(mA - m);
  const bFactor = Math.exp(mB - m);
  return { m, D: DA * aFactor + DB * bFactor, aFactor, bFactor };
}

/** Eq. 20.3: the partial max and denominator of a set of scores (float64). */
export function partial(scores: number[]): { m: number; D: number } {
  const m = Math.max(...scores);
  return { m, D: scores.reduce((s, l) => s + Math.exp(l - m), 0) };
}

/** A partial result over a set of columns: its max, its denominator and the un-normalised Õ (float64). */
export interface PartialState {
  m: number;
  D: number;
  O: number[];
}

export function partialState(scores: number[], V: number[][], cols: number[]): PartialState {
  const m = Math.max(...cols.map((j) => scores[j]));
  let D = 0;
  const O = new Array(V[0].length).fill(0);
  for (const j of cols) {
    const e = Math.exp(scores[j] - m);
    D += e;
    for (let c = 0; c < O.length; c++) O[c] += e * V[j][c];
  }
  return { m, D, O };
}

/** Eqs. 20.4 and 20.6 (un-normalised) for two partial results, as split-KV decoding merges them (facts A8, D1). */
export function mergeStates(a: PartialState, b: PartialState): PartialState & { aFactor: number; bFactor: number } {
  const r = mergeD(a.m, a.D, b.m, b.D);
  return { m: r.m, D: r.D, O: a.O.map((x, c) => x * r.aFactor + b.O[c] * r.bFactor), aFactor: r.aFactor, bFactor: r.bFactor };
}

// ---------------------------------------------------------------- the recurrence
export interface OnlineConfig {
  scores: number[];
  V: number[][];
  /** Tile width B_c (any width; the last tile may be shorter). */
  bc: number;
  /** Causal row index: columns > row are masked by index. */
  row: number;
  order: TileOrder;
  init: InitMax;
  form: Form;
}

export interface TileStep {
  /** 1-based position in the merge order. */
  k: number;
  /** Index of the tile (0 = leftmost). */
  tile: number;
  cols: number[];
  visible: number[];
  /** Scores of the tile, null where the column is masked. */
  scores: (number | null)[];
  mOld: number;
  /** The tile's own max over its visible columns (the start value if none). */
  mB: number;
  /** Eq. 20.3: Σ e^(l − m_B) over the tile's visible columns. */
  DB: number;
  mNew: number;
  rose: boolean;
  /** e^(m_A − m_A∪B), float32. */
  aFactor: number;
  /** e^(m_B − m_A∪B), float32; null when the tile has no visible column. */
  bFactor: number | null;
  /** The B term as the code builds it: Σ e^(l − m_new) over visible columns (= D_B · bFactor). */
  bTerm: number;
  /** The tile's terms e^(l − m_new), null where masked (what the code writes over S). */
  p: (number | null)[];
  /** Running state after this tile. D, and O as kept by the chosen form. */
  D: number;
  O: number[];
  /** The current estimate of the output: Õ/D (deferred) or o (normalised). */
  out: number[];
  /** Exact attention (float64) over every visible column merged so far; null if none yet. */
  ref: number[] | null;
  /** max |out − ref|, or null without a reference. */
  diff: number | null;
  nan: boolean;
  /** The normalised form skipped this tile (no visible column). */
  skipped: boolean;
}

export interface OnlineTrace {
  config: OnlineConfig;
  start: { m: number; D: number; O: number[] };
  steps: TileStep[];
  final: number[];
  /** Exact attention over every visible column of the row. */
  exact: number[];
  diff: number;
  nan: boolean;
  /** e^(largest visible score) in float32: Infinity shows why the max is subtracted. */
  naiveExpOfMax: number;
}

export const tileCount = (n: number, bc: number) => Math.ceil(n / bc);

/** The order in which the tiles are merged. 'shuffled' is a fixed permutation for each count. */
export function tileOrder(nTiles: number, order: TileOrder): number[] {
  const idx = Array.from({ length: nTiles }, (_, i) => i);
  if (order === 'rtl') return idx.reverse();
  if (order === 'shuffled') {
    const r = mulberry(97 + nTiles);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(r() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    // Never the identity, so "shuffled" always differs from the book's order.
    if (nTiles > 1 && idx.every((v, i) => v === i)) [idx[0], idx[1]] = [idx[1], idx[0]];
  }
  return idx;
}

const fexp = (x: number) => f(Math.exp(f(x)));
const maxDiff = (a: number[], b: number[]) => Math.max(...a.map((x, i) => Math.abs(x - b[i])));
const hasNaN = (a: number[]) => a.some((x) => Number.isNaN(x));

export function runOnline(cfg: OnlineConfig): OnlineTrace {
  const n = cfg.scores.length;
  const dv = cfg.V[0].length;
  const l = cfg.scores.map(f);
  const V = cfg.V.map((r) => r.map(f));
  const start = initValue(cfg.init);
  let m = start;
  let D = 0;
  let O = new Array(dv).fill(0);
  const seen: number[] = [];
  const steps: TileStep[] = [];
  const order = tileOrder(tileCount(n, cfg.bc), cfg.order);

  order.forEach((t, k) => {
    const cols = Array.from({ length: Math.min(cfg.bc, n - t * cfg.bc) }, (_, i) => t * cfg.bc + i);
    const visible = cols.filter((j) => j <= cfg.row);
    const tileScores = cols.map((j) => (j <= cfg.row ? l[j] : null));
    // compute_S_and_max: the tile's max starts from the same value as the running max.
    let mB = start;
    for (const j of visible) mB = Math.max(mB, l[j]);
    let DB = 0;
    for (const j of visible) DB = f(DB + fexp(l[j] - mB));
    const mOld = m;
    const mNew = Math.max(mOld, mB);
    const rose = mOld < mB;
    const aFactor = fexp(mOld - mNew);
    const bFactor = visible.length ? fexp(mB - mNew) : null;
    const p = cols.map((j) => (j <= cfg.row ? fexp(l[j] - mNew) : null));
    let bTerm = 0;
    for (const x of p) if (x != null) bTerm = f(bTerm + x);
    let skipped = false;

    if (cfg.form === 'deferred') {
      // update_m_and_D (Fig. 20.14): D is rescaled only when the max rises.
      if (rose) D = f(D * aFactor);
      m = mNew;
      // compute_P_and_update_D (Fig. 20.15): add the B term, computed against the new max.
      D = f(D + bTerm);
      // compute_O (Fig. 20.13): rescale first, then add P · V_j.
      O = O.map((o, c) => {
        let acc = 0;
        cols.forEach((j, i) => {
          if (p[i] != null) acc = f(acc + f(p[i]! * V[j][c]));
        });
        return f(f(o * aFactor) + acc);
      });
    } else if (!visible.length) {
      skipped = true; // Eq. 20.5 would be 0/0
    } else {
      // Eq. 20.5 for the tile alone, then Eqs. 20.4 and 20.6.
      const oB = new Array(dv).fill(0).map((_, c) => {
        let acc = 0;
        for (const j of visible) acc = f(acc + f(fexp(l[j] - mB) * V[j][c]));
        return f(acc / DB);
      });
      const DA = D;
      const Dn = f(f(DA * aFactor) + f(DB * bFactor!));
      const wA = f(f(DA * aFactor) / Dn);
      const wB = f(f(DB * bFactor!) / Dn);
      O = O.map((o, c) => f(f(wA * o) + f(wB * oB[c])));
      D = Dn;
      m = mNew;
    }

    seen.push(...visible);
    const out = cfg.form === 'deferred' ? O.map((o) => f(o / D)) : O.slice();
    const ref = exactOver(cfg.scores, cfg.V, seen);
    steps.push({
      k: k + 1,
      tile: t,
      cols,
      visible,
      scores: tileScores,
      mOld,
      mB,
      DB,
      mNew,
      rose,
      aFactor,
      bFactor,
      bTerm,
      p,
      D,
      O: O.slice(),
      out,
      ref,
      diff: ref ? maxDiff(out, ref) : null,
      nan: hasNaN(out) || hasNaN(O) || Number.isNaN(D),
      skipped,
    });
  });

  const allVisible = Array.from({ length: n }, (_, j) => j).filter((j) => j <= cfg.row);
  const exact = exactOver(cfg.scores, cfg.V, allVisible) ?? new Array(dv).fill(NaN);
  const final = steps.length ? steps[steps.length - 1].out : new Array(dv).fill(NaN);
  const top = Math.max(...allVisible.map((j) => l[j]));
  return {
    config: cfg,
    start: { m: start, D: 0, O: new Array(dv).fill(0) },
    steps,
    final,
    exact,
    diff: maxDiff(final, exact),
    nan: hasNaN(final),
    naiveExpOfMax: f(Math.exp(top)),
  };
}
