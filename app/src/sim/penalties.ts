// Token counts and the three penalties of section X1.3, and the thread-to-logit
// mappings of its "Who reads which logit" explorer. Also used by chain.ts (the sampler
// chain of X1.1) for its penalties stage.
//
// None of this is in the book. Counting is chapter 9's histogram (pp. 203-217) with token
// ids as bins; the formulas are the projects' (notes/today/x1-facts.md, notes/today/ch09-facts.md):
//
//   - counts: every token of a context, or only the last N (llama.cpp's penalty_last_n,
//     default 64, ch09 D4); appendToken adds one decode step's token, as lab 9.3's
//     appendTokens does;
//   - repetition penalty theta, for a token whose count is above 0, under three rules:
//       'llama'  llama.cpp: multiply a logit that is <= 0, divide a positive one (D2);
//       'hf'     transformers and mlx-lm: multiply a logit that is < 0, else divide (D4, D8);
//       'ctrl'   the CTRL paper: divide regardless of sign (D9), which makes a repeated
//                token with a negative logit more likely;
//   - frequency and presence: logit -= count * freq + (count > 0) * presence, OpenAI's
//     documented formula, which llama.cpp implements after the repetition penalty
//     (D9, ch09 D4). Lab 9.3's applyPenalties is exactly this term.
//
// float32: every operation is rounded with Math.fround in llama.cpp's order (the product
// count * freq, the product (count > 0) * presence, their sum, then the subtraction; the
// repetition product or quotient on its own). No fused multiply-add: CUDA contracts
// a * b + c by default, which can change the last bit (X1.8 and lab X1.3 deal with that).
//
// Mappings: for one load instruction of one warp, which logit each lane reads under
// three ways of assigning threads to a [rows x vocab] tensor, as byte addresses for
// warpAccess (app/src/sim/coalesce.ts, imported read-only). The warp width is always a
// parameter, never a literal.
//
// Erasable TypeScript only (the browser checks import this file through the app's module server).
import { SEGMENT_MODELS, warpAccess, type SegmentModelId, type WarpAccess } from './coalesce';
import { tr } from '../lib/locale.ts';

const id = (x: number) => x;
export const f32 = Math.fround;

// ---------------------------------------------------------------- counts
/**
 * counts[t] = how often token t occurs in `context`, or in its last `lastN` tokens when
 * lastN is given (llama.cpp's window). Token ids outside [0, vocab) are a caller error.
 */
export function countTokens(context: readonly number[], vocab: number, lastN?: number): number[] {
  const counts = new Array<number>(vocab).fill(0);
  const from = lastN == null ? 0 : Math.max(0, context.length - lastN);
  for (let i = from; i < context.length; i++) {
    const t = context[i];
    if (!Number.isInteger(t) || t < 0 || t >= vocab) throw new Error(`token ${t} outside [0, ${vocab})`);
    counts[t] += 1;
  }
  return counts;
}

/** One decode step: the row gains one token. Lab 9.3's appendTokens, for one row. */
export function appendToken(counts: readonly number[], token: number): number[] {
  const out = [...counts];
  out[token] += 1;
  return out;
}

// ---------------------------------------------------------------- the three penalties
export type RepRule = 'llama' | 'hf' | 'ctrl';
export const REP_RULES: RepRule[] = ['llama', 'hf', 'ctrl'];
export const REP_RULE_LABEL: Record<RepRule, string> = {
  get llama() {
    return tr('llama.cpp: multiply when logit ≤ 0', 'llama.cpp: nhân khi logit ≤ 0');
  },
  get hf() {
    return tr('transformers, mlx-lm: multiply when logit < 0', 'transformers, mlx-lm: nhân khi logit < 0');
  },
  get ctrl() {
    return tr('CTRL paper: always divide', 'bài báo CTRL: luôn chia');
  },
};
export const REP_RULE_FACT: Record<RepRule, string> = {
  llama: 'x1-facts D2',
  hf: 'x1-facts D4, D8',
  ctrl: 'x1-facts D9',
};

/** The repetition penalty for one token. A token that has not occurred (count 0) is untouched. */
export function repetition(logit: number, count: number, theta: number, rule: RepRule, float32 = false): number {
  if (count <= 0) return logit;
  const r = float32 ? f32 : id;
  const l = r(logit);
  const th = r(theta);
  if (rule === 'ctrl') return r(l / th);
  const multiply = rule === 'llama' ? l <= 0 : l < 0;
  return multiply ? r(l * th) : r(l / th);
}

/** Frequency and presence: logit - (count * freq + (count > 0) * presence), in llama.cpp's order. */
export function frequencyPresence(logit: number, count: number, freq: number, presence: number, float32 = false): number {
  const r = float32 ? f32 : id;
  const term = r(r(count * r(freq)) + r((count > 0 ? 1 : 0) * r(presence)));
  return r(r(logit) - term);
}

export type PenaltyOrder = 'repeat-first' | 'subtract-first';
export const PENALTY_ORDER_LABEL: Record<PenaltyOrder, string> = {
  get 'repeat-first'() {
    return tr('repetition, then subtract (llama.cpp)', 'phạt lặp lại, rồi trừ (llama.cpp)');
  },
  get 'subtract-first'() {
    return tr('subtract, then repetition', 'trừ, rồi phạt lặp lại');
  },
};

export interface PenaltySettings {
  /** Repetition penalty theta; 1 is off. */
  repeat: number;
  /** Frequency penalty; 0 is off. */
  freq: number;
  /** Presence penalty; 0 is off. */
  presence: number;
  rule: RepRule;
  order: PenaltyOrder;
  float32: boolean;
}
export const PENALTIES_OFF: PenaltySettings = { repeat: 1, freq: 0, presence: 0, rule: 'llama', order: 'repeat-first', float32: false };

/** One logit through all three penalties. */
export function penalise(logit: number, count: number, s: PenaltySettings): number {
  if (s.order === 'repeat-first')
    return frequencyPresence(repetition(logit, count, s.repeat, s.rule, s.float32), count, s.freq, s.presence, s.float32);
  return repetition(frequencyPresence(logit, count, s.freq, s.presence, s.float32), count, s.repeat, s.rule, s.float32);
}

/** A whole row: one pass over the logits and their counts. */
export function applyPenalties(logits: readonly number[], counts: readonly number[], s: PenaltySettings): number[] {
  return logits.map((l, i) => penalise(l, counts[i] ?? 0, s));
}

/** Softmax in double, for the explorer's "probabilities after" row. */
export function softmaxDouble(logits: readonly number[]): number[] {
  const m = Math.max(...logits);
  const e = logits.map((l) => Math.exp(l - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

/**
 * What the penalties explorer reports. `raised`: repeated tokens whose logit went up, which
 * only CTRL's rule does (a negative logit divided by theta moves towards 0). `gained`:
 * repeated tokens whose logit did not go up but whose probability still rose, because the
 * softmax renormalises and other repeated tokens lost more. The two are different effects
 * and the explorer names them separately.
 */
export function penaltyEffects(before: readonly number[], after: readonly number[], counts: readonly number[]): { raised: number[]; gained: number[] } {
  const pB = softmaxDouble(before);
  const pA = softmaxDouble(after);
  const raised: number[] = [];
  const gained: number[] = [];
  for (let i = 0; i < before.length; i++) {
    if (!(counts[i] > 0)) continue;
    if (after[i] > before[i]) raised.push(i);
    else if (pA[i] > pB[i] * (1 + 1e-12)) gained.push(i);
  }
  return { raised, gained };
}

// ---------------------------------------------------------------- the X1.3 explorer's row
/** Twelve logits on both sides of zero; the context repeats some of each sign. */
export const PEN_LOGITS: readonly number[] = [1.6, -2.0, 0.4, 2.3, -0.6, 0.9, -1.3, 1.1, 0.0, -0.2, 1.9, -1.0];
/** About twenty tokens of context (ids into PEN_LOGITS). */
export const PEN_CONTEXT: readonly number[] = [3, 7, 1, 3, 10, 5, 1, 3, 8, 0, 6, 3, 7, 1, 10, 2, 3, 4, 7, 1];
export const PEN_VOCAB = PEN_LOGITS.length;

/** Index of the largest value, lowest index on ties: what "append the most likely token" appends. */
export function argmaxLowest(v: readonly number[]): number {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
  return best;
}

// ---------------------------------------------------------------- bytes
/** Minimum bytes per logit for `passes` separate penalty kernels, each reading the logit and its count and writing the logit. */
export const bytesPerLogit = (passes: number) => passes * (4 + 4 + 4);

// ---------------------------------------------------------------- who reads which logit
export type Mapping = 'element' | 'row' | 'block';
export const MAPPINGS: Mapping[] = ['element', 'block', 'row'];
export const MAPPING_LABEL: Record<Mapping, string> = {
  get element() {
    return tr('one thread per element (flattened)', 'mỗi phần tử một luồng (làm phẳng)');
  },
  get block() {
    return tr('one block per row, strided loop', 'mỗi hàng một khối, vòng lặp theo bước nhảy');
  },
  get row() {
    return tr('one thread per row, looping over the vocabulary', 'mỗi hàng một luồng, lặp qua toàn bộ từ vựng');
  },
};

export interface MappingParams {
  batch: number;
  vocab: number;
  warpWidth: number;
  /** Threads per block, for the 'block' mapping. */
  blockDim: number;
}

export interface LaneRead {
  lane: number;
  row: number;
  col: number;
  /** Byte address of the float32 logit this lane loads. */
  byte: number;
}

export interface WarpLoad {
  mapping: Mapping;
  /** The active lanes of warp 0 in its first load, and what each reads. */
  lanes: LaneRead[];
  /** Threads that have any work at all in the launch. */
  threadsWithWork: number;
  /** Loads of a logit each working thread performs (the longest). */
  loadsPerThread: number;
}

/** What the first warp reads in its first load under each mapping (row-major [batch x vocab], 4-byte logits). */
export function warpLoad(mapping: Mapping, p: MappingParams): WarpLoad {
  const { batch, vocab, warpWidth: W, blockDim } = p;
  const lanes: LaneRead[] = [];
  if (mapping === 'element') {
    const n = batch * vocab;
    for (let lane = 0; lane < W; lane++) {
      const e = lane; // global thread index of warp 0
      if (e < n) lanes.push({ lane, row: Math.floor(e / vocab), col: e % vocab, byte: 4 * e });
    }
    return { mapping, lanes, threadsWithWork: n, loadsPerThread: 1 };
  }
  if (mapping === 'row') {
    for (let lane = 0; lane < W; lane++) {
      if (lane < batch) lanes.push({ lane, row: lane, col: 0, byte: 4 * lane * vocab });
    }
    return { mapping, lanes, threadsWithWork: batch, loadsPerThread: vocab };
  }
  // 'block': block 0 owns row 0; thread t reads columns t, t + blockDim, ...
  const working = Math.min(blockDim, vocab);
  for (let lane = 0; lane < W; lane++) {
    if (lane < working) lanes.push({ lane, row: 0, col: lane, byte: 4 * lane });
  }
  return { mapping, lanes, threadsWithWork: batch * working, loadsPerThread: Math.ceil(vocab / blockDim) };
}

/** The same load, with the segments it touches under a segment size. */
export function mappingAccess(mapping: Mapping, p: MappingParams, segmentBytes: number): WarpLoad & WarpAccess {
  const w = warpLoad(mapping, p);
  return { ...w, ...warpAccess(w.lanes.map((l) => l.byte), 4, segmentBytes) };
}

/** Segment size for a vendor model; Metal documents none, so the caller passes the reader's choice. */
export function segmentFor(model: SegmentModelId, metalChoice: number): number {
  return SEGMENT_MODELS[model].segmentBytes ?? metalChoice;
}

/** Vocabulary of Qwen3-8B (notes/today/ch09-facts.md D7): the explorer's row length. */
export const QWEN3_VOCAB = 151_936;
