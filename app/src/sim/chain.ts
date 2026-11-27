// The sampler chain of module X1 for one row: the model behind the start page's stage
// map, the "Reorder the chain" explorer of X1.1, the "Temperature" explorer of X1.2 and
// every beat of the "One row through the chain" scene. Nothing here is in the book; the
// book stops at a probability distribution over tokens (ch. 20 §20.1, p. 481).
//
// What it holds:
//   - STAGES: every stage of llama.cpp's default chain in its order (x1-facts D1), which
//     ones are modelled (DRY, top-n-sigma, typical and XTC are listed, not modelled), and
//     for each modelled stage the milestone-M5 kernels that implement it and the lab that
//     supplies each kernel; DRAW and GREEDY for the last step;
//   - ORDERS: the order presets of llama.cpp, transformers, mlx-lm and M5 (D1, D4, D8),
//     with each project's top-k tie rule and repetition rule;
//   - stage functions on one row: penalties (penalties.ts), temperature (divide, or
//     multiply by the reciprocal as llama.cpp's backend does, D2), softmax in double or
//     in float32 in lab 10.3's order, top-k by a stable sort and cut ('lowest' id wins a
//     tie at the cut, or 'keep' every tie as transformers does, D4), top-p through topP of
//     scan.ts with the projects' exclusive rule (ch11 D2), min-p in logit space through
//     minPThreshold of stablescan.ts (ch12 D3), argmax with a 'lowest' or 'highest' tie
//     rule (ch10 D2, D4), and an inverse-CDF draw with the strict rule (first token whose
//     inclusive cumulative probability is greater than u; x1-facts E2);
//   - runChain(row, settings, order): per-stage snapshots (survivors, logits,
//     probabilities among the survivors) and the draw;
//   - the preset rows, and the numbers the lessons state.
//
// What it simplifies (the lessons say so beside each interactive):
//   - one row of a dozen or so tokens; no batch;
//   - double precision unless float32 is requested. In float32 every operation is rounded
//     with Math.fround in lab 10.3's order (scale, subtract the max, exp, sum left to
//     right, one reciprocal, one multiply). Math.exp is not CUDA's expf or Metal's exp
//     (within 2 and 4 ulp, ch10-facts A11, C4), and no fused multiply-add is modelled;
//   - top-p in double compares probabilities rounded to six decimals (the precision of
//     scan.ts's exact mode); in float32 it uses scan.ts's left-to-right float32 scan;
//   - every truncation after another one renormalises over the survivors, as llama.cpp
//     (softmax over the current candidates) and transformers (minus infinity, which
//     softmax turns into 0) both do;
//   - transformers' min-p compares probabilities, the model compares logits (the same
//     set up to rounding); mlx-lm's top-p keeps every token tied with the boundary token,
//     the model uses the exclusive rule for every preset (the same set without ties);
//   - the stages llama.cpp has and M5 does not build (DRY, top-n-sigma, typical, XTC) are
//     shown and skipped.
//
// Erasable TypeScript only.
import { tr } from '../lib/locale.ts';
import { topP } from './scan';
import { minPThreshold } from './stablescan';
import { applyPenalties, countTokens, type PenaltyOrder, type RepRule } from './penalties';

export const f32 = Math.fround;
const same = (x: number) => x;

// ---------------------------------------------------------------- the stages
export type StageId = 'penalties' | 'dry' | 'top-n-sigma' | 'top-k' | 'typical' | 'top-p' | 'min-p' | 'xtc' | 'temperature';
export type ModelledStage = 'penalties' | 'top-k' | 'top-p' | 'min-p' | 'temperature';
export const MODELLED: ModelledStage[] = ['penalties', 'top-k', 'top-p', 'min-p', 'temperature'];

export interface LabRef {
  /** What the reader types after `pmpp.py run`, e.g. "9.3" or "X1.1". */
  num: string;
  /** Chapter id for the link to its labs page. */
  chapter: string;
  title: string;
}
export interface KernelStep {
  what: string;
  lab: LabRef;
}
export interface Stage {
  id: StageId;
  label: string;
  /** reshape: changes logits, never the candidate set. truncate: removes candidates. */
  kind: 'reshape' | 'truncate';
  modelled: boolean;
  /** The M5 kernels that implement the stage, in order. Empty for a stage M5 does not build. */
  kernels: KernelStep[];
}

export const LAB = {
  counts: { num: '9.3', chapter: 'ch09', title: 'token counts for repetition penalties' }, // i18n-ignore: identifier also shown to the reader
  softmax: { num: '10.3', chapter: 'ch10', title: 'softmax and argmax over logits (M4)' }, // i18n-ignore: identifier also shown to the reader
  topp: { num: '11.6', chapter: 'ch11', title: 'top-p keep mask' }, // i18n-ignore: identifier also shown to the reader
  compact: { num: '12.4', chapter: 'ch12', title: 'candidate compaction' }, // i18n-ignore: identifier also shown to the reader
  sort: { num: '14.3', chapter: 'ch14', title: 'sorted candidates, every row descending and stable' }, // i18n-ignore: identifier also shown to the reader
  radix: { num: 'X1.1', chapter: 'x1', title: 'top-k by radix select, measured against a full sort' }, // i18n-ignore: identifier also shown to the reader
  filters: { num: 'X1.2', chapter: 'x1', title: 'top-p and min-p: composed from earlier labs, then fused' }, // i18n-ignore: identifier also shown to the reader
  m5: { num: 'X1.3', chapter: 'x1', title: 'milestone M5: a batched GPU sampler' }, // i18n-ignore: identifier also shown to the reader
} as const satisfies Record<string, LabRef>;

/** llama.cpp's default chain, in its order (x1-facts D1). The draw follows the last stage. */
export const STAGES: Stage[] = [
  {
    id: 'penalties', get label() { return tr('penalties', 'hình phạt'); }, kind: 'reshape', modelled: true,
    kernels: [
      { get what() { return tr('count the tokens of each row', 'đếm token của mỗi hàng'); }, lab: LAB.counts },
      { get what() { return tr('repetition, sign-aware', 'phạt lặp lại, có xét dấu'); }, lab: LAB.m5 },
      { get what() { return tr('frequency and presence', 'tần suất và hiện diện'); }, lab: LAB.counts },
    ],
  },
  { id: 'dry', label: 'DRY', kind: 'reshape', modelled: false, kernels: [] },
  { id: 'top-n-sigma', label: 'top-n-sigma', kind: 'truncate', modelled: false, kernels: [] }, // i18n-ignore: sampler name, kept in English per the glossary
  {
    id: 'top-k', label: 'top-k', kind: 'truncate', modelled: true, // i18n-ignore: sampler name, kept in English per the glossary
    kernels: [
      { get what() { return tr('find the k-th largest logit by radix select', 'tìm logit lớn thứ k bằng radix select'); }, lab: LAB.radix },
      { get what() { return tr('gather the candidates in column order', 'gom các ứng viên theo thứ tự cột'); }, lab: LAB.compact },
      { get what() { return tr('sort the k survivors, descending and stable', 'sắp xếp k ứng viên còn lại, giảm dần và ổn định'); }, lab: LAB.sort },
    ],
  },
  { id: 'typical', label: 'typical', kind: 'truncate', modelled: false, kernels: [] }, // i18n-ignore: sampler name, kept in English per the glossary
  {
    id: 'top-p', label: 'top-p', kind: 'truncate', modelled: true, // i18n-ignore: sampler name, kept in English per the glossary
    kernels: [
      { get what() { return tr('softmax of the survivors at T = 1', 'softmax của các ứng viên còn lại ở T = 1'); }, lab: LAB.softmax },
      { get what() { return tr('scan and keep mask', 'scan và mặt nạ giữ lại'); }, lab: LAB.topp },
    ],
  },
  {
    id: 'min-p', label: 'min-p', kind: 'truncate', modelled: true, // i18n-ignore: sampler name, kept in English per the glossary
    kernels: [{ get what() { return tr('count the prefix at or above max + ln(min-p)', 'đếm phần tiền tố lớn hơn hoặc bằng max + ln(min-p)'); }, lab: LAB.filters }],
  },
  { id: 'xtc', label: 'XTC', kind: 'truncate', modelled: false, kernels: [] },
  {
    id: 'temperature', get label() { return tr('temperature', 'nhiệt độ'); }, kind: 'reshape', modelled: true,
    kernels: [{ get what() { return tr('divide the survivors by T inside the draw', 'chia các ứng viên còn lại cho T bên trong lần rút'); }, lab: LAB.m5 }],
  },
];
export const stage = (id: StageId) => STAGES.find((s) => s.id === id)!;

/** The last step of every chain. */
export const DRAW: KernelStep = { get what() { return tr('Philox u, then an inverse-CDF search over a scan', 'Philox u, rồi tìm kiếm CDF ngược trên một scan'); }, lab: LAB.m5 };
/** Temperature at or below 0: no draw, the largest logit wins. */
export const GREEDY: KernelStep = { get what() { return tr('argmax, ties to the lowest id', 'argmax, hòa thì chọn id nhỏ nhất'); }, lab: LAB.softmax };

/** M5's order, read from STAGES: llama.cpp's default restricted to the stages M5 builds. */
export const M5_ORDER: ModelledStage[] = STAGES.filter((s) => s.modelled).map((s) => s.id as ModelledStage);

// ---------------------------------------------------------------- the order presets
export type TopKTies = 'lowest' | 'keep';
export type OrderId = 'llama' | 'hf' | 'mlx' | 'm5';
export interface OrderPreset {
  id: OrderId;
  label: string;
  /** Every stage the project runs that STAGES lists, in its order; unmodelled ones are skipped. */
  order: StageId[];
  topKTies: TopKTies;
  repRule: RepRule;
  /** transformers has a repetition penalty and no frequency or presence penalty (D4). */
  hasFreqPresence: boolean;
  facts: string;
}
export const ORDERS: Record<OrderId, OrderPreset> = {
  llama: {
    id: 'llama', get label() { return tr('llama.cpp default', 'mặc định của llama.cpp'); },
    order: STAGES.map((s) => s.id),
    topKTies: 'lowest', repRule: 'llama', hasFreqPresence: true, facts: 'x1-facts D1, D2',
  },
  hf: {
    id: 'hf', label: 'transformers', // i18n-ignore: product name
    order: ['penalties', 'temperature', 'top-k', 'top-p', 'min-p', 'typical'],
    topKTies: 'keep', repRule: 'hf', hasFreqPresence: false, facts: 'x1-facts D4',
  },
  mlx: {
    id: 'mlx', label: 'mlx-lm', // i18n-ignore: product name
    order: ['penalties', 'top-p', 'min-p', 'xtc', 'top-k', 'temperature'],
    topKTies: 'lowest', repRule: 'hf', hasFreqPresence: true, facts: 'x1-facts D8; penalties ch09 D5', // i18n-ignore: citation
  },
  m5: {
    id: 'm5', get label() { return tr('M5 (this course)', 'M5 (khóa học này)'); },
    order: [...M5_ORDER],
    topKTies: 'lowest', repRule: 'llama', hasFreqPresence: true, get facts() { return tr('ours: llama.cpp’s order, lowest id on every tie', 'của khóa học này: thứ tự của llama.cpp, mọi trường hợp hòa chọn id nhỏ nhất'); },
  },
};
export const ORDER_IDS: OrderId[] = ['llama', 'hf', 'mlx', 'm5'];
/** The stages of an order that the model runs. */
export const modelledOf = (order: readonly StageId[]): ModelledStage[] =>
  order.filter((s): s is ModelledStage => (MODELLED as StageId[]).includes(s));

// ---------------------------------------------------------------- stage functions on one row
/** Temperature: 'divide' as llama.cpp's host does, 'reciprocal' (multiply by 1/T) as its backend does (D2). */
export function temperature(logits: readonly number[], T: number, float32 = false, how: 'divide' | 'reciprocal' = 'divide'): number[] {
  if (!(T > 0)) throw new Error('temperature needs T > 0; T <= 0 is greedy');
  if (!float32) return how === 'divide' ? logits.map((l) => l / T) : logits.map((l) => l * (1 / T));
  const t = f32(T);
  if (how === 'divide') return logits.map((l) => f32(f32(l) / t));
  const inv = f32(1 / t);
  return logits.map((l) => f32(f32(l) * inv));
}

/** Indices where dividing by T and multiplying by a rounded 1/T give different float32 logits. */
export function reciprocalMismatch(logits: readonly number[], T: number): number[] {
  const a = temperature(logits, T, true, 'divide');
  const b = temperature(logits, T, true, 'reciprocal');
  return a.flatMap((v, i) => (Object.is(v, b[i]) ? [] : [i]));
}

/**
 * softmax(scale * x) as lab 10.3's softmaxRowsKernel computes it: m = max of scale * x,
 * e = exp(scale * x - m), s = their sum, p = e * (1 / s). In float32 each step is rounded;
 * the sum runs left to right. scale = 1 / T folds temperature in.
 */
export function softmax(logits: readonly number[], scale = 1, float32 = false): number[] {
  const r = float32 ? f32 : same;
  const sc = r(scale);
  const x = logits.map((l) => r(sc * r(l)));
  const m = Math.max(...x);
  const e = x.map((v) => r(Math.exp(r(v - m))));
  let s = 0;
  for (const v of e) s = r(s + v);
  const inv = r(1 / s);
  return e.map((v) => r(v * inv));
}

/** Shannon entropy in bits. */
export const entropyBits = (probs: readonly number[]) => -probs.reduce((s, p) => (p > 0 ? s + p * Math.log2(p) : s), 0);

/** Index of the largest logit. 'lowest' is M5's and PyTorch's rule (ch10 D4); 'highest' is llama.cpp's Metal argmax (ch10 D2). */
export function argmax(logits: readonly number[], tie: 'lowest' | 'highest' = 'lowest', alive?: readonly boolean[]): number {
  let best = -1;
  for (let i = 0; i < logits.length; i++) {
    if (alive && !alive[i]) continue;
    if (best < 0 || logits[i] > logits[best] || (tie === 'highest' && logits[i] === logits[best])) best = i;
  }
  return best;
}

/** Survivor ids in descending order of logit, lower id first on a tie (the stable descending sort of lab 14.3). */
export function rankDesc(logits: readonly number[], alive: readonly boolean[]): number[] {
  const ids: number[] = [];
  for (let i = 0; i < logits.length; i++) if (alive[i]) ids.push(i);
  return ids.sort((a, b) => (logits[b] - logits[a]) || a - b);
}

/** Top-k by a stable sort and a cut: exactly k ('lowest' id wins a tie at the cut), or every token tied with the k-th ('keep'). */
export function topK(logits: readonly number[], alive: readonly boolean[], k: number, ties: TopKTies = 'lowest'): boolean[] {
  const ranked = rankDesc(logits, alive);
  if (k <= 0 || k >= ranked.length) return [...alive];
  const out = logits.map(() => false);
  if (ties === 'lowest') for (const i of ranked.slice(0, k)) out[i] = true;
  else {
    const kth = logits[ranked[k - 1]];
    for (const i of ranked) if (logits[i] >= kth) out[i] = true;
  }
  return out;
}

/** Probabilities among the survivors (0 for the others), in double or float32. */
export function survivorProbs(logits: readonly number[], alive: readonly boolean[], float32 = false): number[] {
  const ids = logits.flatMap((_, i) => (alive[i] ? [i] : []));
  const p = softmax(ids.map((i) => logits[i]), 1, float32);
  const out = logits.map(() => 0);
  ids.forEach((id, j) => (out[id] = p[j]));
  return out;
}

const round6 = (x: number) => Math.round(x * 1e6) / 1e6;

/** Top-p over the survivors: renormalise, sort descending, keep while the mass before is below p (scan.ts's exclusive rule). */
export function topPKeep(logits: readonly number[], alive: readonly boolean[], p: number, float32 = false): boolean[] {
  const ranked = rankDesc(logits, alive);
  if (p >= 1 || ranked.length <= 1) return [...alive];
  const probs = survivorProbs(logits, alive, float32);
  const sorted = ranked.map((i) => probs[i]);
  const res = float32
    ? topP(sorted, p, 'exclusive', { kind: 'f32', order: 'loop' })
    : topP(sorted.map(round6), round6(p), 'exclusive');
  const out = logits.map(() => false);
  ranked.forEach((id, j) => (out[id] = res.keep[j]));
  return out;
}

/** Min-p over the survivors in logit space: keep a logit at or above max + ln(min-p) (stablescan.ts, llama.cpp's host rule). */
export function minPKeep(logits: readonly number[], alive: readonly boolean[], minP: number, float32 = false): { keep: boolean[]; threshold: number } {
  const live = logits.filter((_, i) => alive[i]);
  if (minP <= 0 || live.length === 0) return { keep: [...alive], threshold: -Infinity };
  const threshold = float32 ? f32(f32(Math.max(...live)) + f32(Math.log(minP))) : minPThreshold(live, minP);
  return { keep: logits.map((l, i) => alive[i] && l >= threshold), threshold };
}

export interface Draw {
  /** Survivor ids in the order the search walks them: descending probability, lower id first. */
  order: number[];
  /** Probability of each of those survivors. */
  probs: number[];
  /** Inclusive cumulative probability at each of them. */
  cdf: number[];
  u: number;
  /** The id drawn. */
  token: number;
  /** Position of the drawn token in `order`. */
  at: number;
  /** True when no inclusive sum exceeded u (a total that rounds below u) and the last survivor was taken. */
  fallback: boolean;
}

/** Inverse CDF with the strict rule: the first survivor whose inclusive cumulative probability is greater than u. */
export function drawInverseCdf(logits: readonly number[], alive: readonly boolean[], u: number, float32 = false): Draw {
  const order = rankDesc(logits, alive);
  const all = survivorProbs(logits, alive, float32);
  const probs = order.map((i) => all[i]);
  const r = float32 ? f32 : same;
  const cdf: number[] = [];
  let run = 0;
  for (const p of probs) cdf.push((run = r(run + p)));
  let at = cdf.findIndex((c) => c > u);
  const fallback = at < 0;
  if (fallback) at = order.length - 1;
  return { order, probs, cdf, u, token: order[at], at, fallback };
}

// ---------------------------------------------------------------- the chain
export interface Row {
  logits: readonly number[];
  /** Token ids of the context; counts come from them. */
  context: readonly number[];
}

export interface ChainSettings {
  /** Temperature; at or below 0 the chain ends greedy. */
  T: number;
  /** Top-k; at or below 0 (or at least the row's length) it is off. */
  k: number;
  /** Top-p; 1 is off. */
  p: number;
  /** Min-p; 0 is off. */
  minP: number;
  repeat: number;
  freq: number;
  presence: number;
  /** The uniform number of the draw, in [0, 1). */
  u: number;
  float32: boolean;
  penaltyOrder?: PenaltyOrder;
}

export interface Snapshot {
  stage: ModelledStage | 'input';
  alive: boolean[];
  kept: number;
  /** Current logits of every token (a dropped token keeps its last value). */
  logits: number[];
  /** Probabilities among the survivors; 0 for dropped tokens. */
  probs: number[];
  /** Min-p's logit threshold, when this is the min-p stage. */
  threshold?: number;
}

export interface ChainResult {
  snapshots: Snapshot[];
  /** True when T <= 0 made the chain end with argmax instead of a draw. */
  greedy: boolean;
  draw: Draw;
  token: number;
}

export interface ChainOptions {
  topKTies: TopKTies;
  repRule: RepRule;
  /** False for transformers: frequency and presence do not exist there and are ignored. */
  hasFreqPresence: boolean;
}
export const M5_OPTIONS: ChainOptions = { topKTies: 'lowest', repRule: 'llama', hasFreqPresence: true };
export const optionsOf = (o: OrderPreset): ChainOptions => ({ topKTies: o.topKTies, repRule: o.repRule, hasFreqPresence: o.hasFreqPresence });

function snap(stageId: Snapshot['stage'], logits: number[], alive: boolean[], float32: boolean, threshold?: number): Snapshot {
  return { stage: stageId, alive: [...alive], kept: alive.filter(Boolean).length, logits: [...logits], probs: survivorProbs(logits, alive, float32), threshold };
}

/** One row through the stages of `order` that the model runs (others are skipped), then the draw. */
export function runChain(row: Row, s: ChainSettings, order: readonly StageId[], opt: ChainOptions = M5_OPTIONS): ChainResult {
  const n = row.logits.length;
  const counts = countTokens(row.context, n);
  let logits = row.logits.map((l) => (s.float32 ? f32(l) : l));
  let alive = logits.map(() => true);
  const snapshots: Snapshot[] = [snap('input', logits, alive, s.float32)];
  let greedy = false;
  for (const st of modelledOf(order)) {
    let threshold: number | undefined;
    if (st === 'penalties') {
      logits = applyPenalties(logits, counts, {
        repeat: s.repeat,
        freq: opt.hasFreqPresence ? s.freq : 0,
        presence: opt.hasFreqPresence ? s.presence : 0,
        rule: opt.repRule,
        order: s.penaltyOrder ?? 'repeat-first',
        float32: s.float32,
      });
    } else if (st === 'temperature') {
      if (s.T > 0) logits = temperature(logits, s.T, s.float32);
      else {
        // llama.cpp's host temperature sampler at T <= 0 keeps the largest logit only (D2).
        const a = argmax(logits, 'lowest', alive);
        alive = alive.map((_, i) => i === a);
        greedy = true;
      }
    } else if (st === 'top-k') alive = topK(logits, alive, s.k, opt.topKTies);
    else if (st === 'top-p') alive = topPKeep(logits, alive, s.p, s.float32);
    else {
      const r = minPKeep(logits, alive, s.minP, s.float32);
      alive = r.keep;
      threshold = r.threshold;
    }
    snapshots.push(snap(st, logits, alive, s.float32, threshold));
  }
  const draw = drawInverseCdf(logits, alive, s.u, s.float32);
  return { snapshots, greedy, draw, token: draw.token };
}

/** The survivors handed to the draw, ascending id. */
export const finalSet = (r: ChainResult) => r.snapshots[r.snapshots.length - 1].alive.flatMap((a, i) => (a ? [i] : []));

// ---------------------------------------------------------------- preset rows
/** The X1.1 explorer's rows: 16 tokens each, with one shared context of repeated tokens. */
export interface RowPreset {
  id: string;
  label: string;
  row: Row;
}
const CONTEXT16 = [7, 3, 7, 12, 3, 7, 5, 9];
export const CHAIN_ROWS: RowPreset[] = [
  {
    id: 'peaked',
    get label() { return tr('one favourite', 'một token nổi trội'); },
    row: { logits: [0.5, 3.6, 1.2, -0.4, 0.9, 2.1, -1.2, 0.3, 1.5, -0.8, 0.0, 2.6, -2.0, 0.7, -0.5, 1.0], context: CONTEXT16 },
  },
  {
    // From a seeded search (seed 7) for a row on which the two orders of the p-order
    // question keep clearly different numbers of tokens, with every cumulative mass at
    // least 0.008 away from p = 0.9 at T = 0.5, 1 and 2.
    id: 'tail',
    get label() { return tr('three favourites, long tail', 'ba token nổi trội, đuôi dài'); },
    row: { logits: [-0.9, -0.8, -0.9, 2.1, -0.6, -1.3, -0.4, 1.9, 0.8, -1.1, 2.4, -0.5, -0.3, -1.4, -1.1, -0.7], context: CONTEXT16 },
  },
];
export const chainRow = (id: string) => CHAIN_ROWS.find((r) => r.id === id)!.row;

export const DEFAULT_SETTINGS: ChainSettings = { T: 0.8, k: 8, p: 0.9, minP: 0.05, repeat: 1.2, freq: 0, presence: 0, u: 0.5, float32: false };

/** The X1.1 Predict: the 'tail' row, k = 8, p = 0.9, T = 2, min-p and penalties off. */
export const P_ORDER: { rowId: string; settings: ChainSettings } = {
  rowId: 'tail',
  settings: { T: 2, k: 8, p: 0.9, minP: 0, repeat: 1, freq: 0, presence: 0, u: 0.5, float32: false },
};
/** How many tokens each project's order can still draw from in the Predict's case. */
export function pOrderCounts(): Record<'llama' | 'hf', number> {
  const row = chainRow(P_ORDER.rowId);
  const count = (o: OrderPreset) => finalSet(runChain(row, P_ORDER.settings, o.order, optionsOf(o))).length;
  return { llama: count(ORDERS.llama), hf: count(ORDERS.hf) };
}

// ---------------------------------------------------------------- the X1.2 temperature rows
export interface TempPreset {
  id: string;
  label: string;
  logits: readonly number[];
}
export const TEMP_ROWS: TempPreset[] = [
  { id: 'peaked', get label() { return tr('peaked', 'nhọn'); }, logits: [0.5, -0.4, 3.0, 1.2, -1.5, 2.0, 0.1, -0.9, 1.6, -2.2, 0.8, 0.3] },
  { id: 'flat', get label() { return tr('flat', 'phẳng'); }, logits: [0.4, 0.1, 0.6, 0.3, 0.5, 0.0, 0.2, 0.7, 0.35, 0.15, 0.45, 0.25] },
  { id: 'tied', get label() { return tr('two tied maxima', 'hai cực đại bằng nhau'); }, logits: [1.0, 2.5, 0.2, -0.5, 2.5, 1.4, 0.0, -1.0, 0.8, 1.9, -0.3, 0.6] },
];
/** The pair the X1.2 explorer opens on: logits 3.0 and 2.0 of the peaked row, a difference of exactly 1. */
export const OPENING_PAIR: [number, number] = [2, 5];

// ---------------------------------------------------------------- the scene
/**
 * "One row through the chain": twelve tokens, a context that repeats four of them, and
 * the settings of the six beats. Chosen so that every stage visibly does its job: the
 * raw favourite (token 7, used three times) falls to second place; top-k with k = 5 cuts
 * between two tokens of equal logit (3 and 10: the lower id stays); top-p at 0.9 drops one
 * of the five; min-p at 0.1 drops one more; three survivors reach the draw.
 */
export const SCENE_ROW: Row = {
  logits: [0.1, -0.8, 3.2, 0.7, -1.7, 2.5, 0.4, 3.4, -0.4, 0.8, 0.7, -2.0],
  context: [7, 5, 7, 8, 1, 7, 8],
};
export const SCENE_SETTINGS: ChainSettings = { T: 0.8, k: 5, p: 0.9, minP: 0.1, repeat: 1.25, freq: 0.2, presence: 0.3, u: 0, float32: false };

/**
 * The scene's u: the first 32-bit word of Philox4x32-10 with counter 0 and key 0, which is
 * Random123's known-answer vector (x1-facts A4, E1), turned into a float by its top 24 bits
 * (E2). With the counter layout of A3 (our derivation) that is seed 0, row 0, step 0. X1.7
 * builds the generator; the scene only uses this one word.
 */
export const KAT_WORD = 0x6627e8d5;
export const KAT_U = (KAT_WORD >>> 8) / 2 ** 24;

export function sceneChain(): ChainResult {
  return runChain(SCENE_ROW, { ...SCENE_SETTINGS, u: KAT_U }, M5_ORDER);
}

// ---------------------------------------------------------------- one MSD radix-select step
/**
 * One step of an MSD radix select for the k largest keys (ch. 14 §14.9, p. 346; llama.cpp's
 * top-k, ch14-facts D2): given the digit histogram read from the highest bin down, the bin
 * that holds the k-th largest key, how many keys lie above it (all of them are in), and how
 * many must still be picked from inside it.
 */
export function msdSelectStep(countsFromTop: readonly number[], k: number): { bin: number; above: number; need: number; running: number[] } {
  const running: number[] = [];
  let run = 0;
  for (let b = 0; b < countsFromTop.length; b++) {
    running.push((run += countsFromTop[b]));
    if (run >= k) return { bin: b, above: run - countsFromTop[b], need: k - (run - countsFromTop[b]), running };
  }
  throw new Error(`fewer than ${k} keys`);
}

// ---------------------------------------------------------------- sizes for the start page
/** Vocabulary sizes in public model configs (notes/today/ch09-facts.md D7). */
export const VOCAB = { gpt2: 50_257, qwen3: 151_936, gptoss: 201_088 } as const;
/** Minimum bytes one sampling step reads for one row: a float32 logit and a uint32 count per token. */
export const rowBytes = (vocab: number) => vocab * (4 + 4);
