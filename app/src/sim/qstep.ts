// The decode step of milestone M8 (lab 22.2) with quantised weights: section X2.7's two
// explorers and its prediction. Group 2 owns this file.
//
// What is the book's: the step itself is lab 22.2's, built on ch. 22 §22.5 (pp. 539–540)
// and ch. 20 §20.6 (pp. 504–506): decode is vector-matrix work, bandwidth-bound in the
// weights. Everything quantised is ours, from notes/today/x2-facts.md: bytes per weight
// from qformats.ts (E1), quantisers from quant.ts (D1, D5), and A1's argument that the
// predicted speed-up is the ratio of bytes moved (which a real kernel only approaches).
//
// stepBytes: the bytes and operations of every piece of the step (lab 22.2's decode.h):
//   embed      read B token ids and B rows of E (in E's format, whole blocks), write H
//   per layer  T = H·W_lᵀ (W_l in the layers' format; read H, write T), T += b_l, H += T
//   LM head    logits = H·Eᵀ (E in its format; read H, write the logits)
//   softmax    read the logits, write the probabilities; argmax: read the logits, write B ids
// Activations, bias, residual and the sampler stay float32, as the project's convention for
// M9 says. Each piece is timed at its roofline bound, max(bytes / bandwidth, operations /
// peak), one after another, as batching.ts does for M8 (ch22-facts E1, E2). The dequantise
// arithmetic is not counted (it is the part a measurement shows), nor are launches or caches.
//
// toyModel and greedyRun: a seeded toy decoder with the structure of lab X2.4's emulator
// case as planned (d 64, vocabulary 256, 2 layers, tied embedding), weights quantised per
// matrix through quant.ts, run greedily from a start token. Ours, random, not a trained model:
// what it shows about tokens flipping says nothing about a real model's perplexity.
//
// Erasable TypeScript only: the browser checks import this file directly.

import { BOOK_DEVICE, M8_DEFAULT, effectivePeak, type Device, type ReportsLike, type StepShape } from './batching';
import { F32, Q4_0, Q8_0, bytesPerWeight, formatById, matrixBytes, type WeightFormat } from './qformats';
import { f32, fakeQuant, rng, gauss, type MatFormat } from './quant';
import { tr } from '../lib/locale.ts';

export { BOOK_DEVICE, M8_DEFAULT };
export type StepFormat = 'f32' | 'q8_0' | 'q4_0';
export const STEP_FORMATS: StepFormat[] = ['f32', 'q8_0', 'q4_0'];
export const STEP_FORMAT_LABEL: Record<StepFormat, string> = { f32: 'F32', q8_0: 'Q8_0', q4_0: 'Q4_0' };
const fmtOf = (f: StepFormat): WeightFormat => (f === 'f32' ? F32 : f === 'q8_0' ? Q8_0 : Q4_0);

export type StepPieceKey = 'embed' | 'linear' | 'bias' | 'residual' | 'head' | 'softmax' | 'argmax';
export interface StepPiece {
  key: StepPieceKey;
  label: string;
  /** Bytes of weights this piece reads (all its repetitions). */
  weightBytes: number;
  /** Every byte it moves, weights included (all its repetitions). */
  bytes: number;
  flops: number;
  /** How many kernels of this piece the step launches. */
  count: number;
  seconds: number;
}
export interface StepFormats {
  /** The layers' d × d weights. */
  layers: StepFormat;
  /** The embedding table, which is also the LM head's weight. */
  embed: StepFormat;
}
export interface StepBytes {
  shape: StepShape;
  batch: number;
  formats: StepFormats;
  pieces: StepPiece[];
  weightBytes: number;
  bytes: number;
  seconds: number;
}

/** The pieces of one decode step at batch B (1 to 8 for the matrix-vector kernels, ch15 D1), timed on dev. */
export function stepBytes(shape: StepShape, formats: StepFormats, B: number, dev: Device = BOOK_DEVICE): StepBytes {
  const { d, layers, vocab } = shape;
  const fl = fmtOf(formats.layers);
  const fe = fmtOf(formats.embed);
  const a = 4; // float32 activations
  const peak = effectivePeak(dev) * 1e9;
  const bw = dev.bw * 1e9;
  const piece = (key: StepPieceKey, label: string, weightBytes: number, other: number, flops: number, count: number): StepPiece => {
    const bytes = weightBytes + other;
    return { key, label, weightBytes, bytes, flops, count, seconds: count * Math.max(bytes / count / bw, flops / count / peak) };
  };
  const linW = matrixBytes(fl, d, d);
  const headW = matrixBytes(fe, vocab, d);
  const embW = matrixBytes(fe, B, d); // B rows of E, whole blocks
  const pieces: StepPiece[] = [
    piece('embed', tr('embedding gather', 'gather embedding'), embW, a * B * d + 4 * B, 0, 1),
    piece('linear', tr('layer linears', 'các lớp tuyến tính'), layers * linW, layers * a * B * (d + d), layers * 2 * B * d * d, layers),
    piece('bias', tr('bias adds', 'cộng bias'), 0, layers * a * (2 * B * d + d), layers * B * d, layers),
    piece('residual', tr('residual adds', 'cộng residual'), 0, layers * 3 * a * B * d, layers * B * d, layers),
    piece('head', tr('LM head', 'LM head'), headW, a * B * (d + vocab), 2 * B * d * vocab, 1),
    piece('softmax', 'softmax', 0, 2 * a * B * vocab, 5 * B * vocab, 1),
    piece('argmax', 'argmax', 0, a * B * vocab + 4 * B, B * vocab, 1),
  ];
  const sum = (k: 'weightBytes' | 'bytes' | 'seconds') => pieces.reduce((s, p) => s + p[k], 0);
  return { shape, batch: B, formats, pieces, weightBytes: sum('weightBytes'), bytes: sum('bytes'), seconds: sum('seconds') };
}

/** Weight bytes of the step's matrices alone (the layers once each, the head once): no gather, no activations. */
export function matrixWeightBytes(shape: StepShape, formats: StepFormats): number {
  return shape.layers * shape.d * shape.d * bytesPerWeight(fmtOf(formats.layers)) + shape.vocab * shape.d * bytesPerWeight(fmtOf(formats.embed));
}

/** The model's predicted speed-up of a quantised step over the float32 step at the same batch. */
export function predictedSpeedup(shape: StepShape, formats: StepFormats, B: number, dev: Device = BOOK_DEVICE): number {
  return stepBytes(shape, { layers: 'f32', embed: 'f32' }, B, dev).seconds / stepBytes(shape, formats, B, dev).seconds;
}

// ---------------------------------------------------------------- lab X2.4's measurements
export const LAB_M9 = 'x2/lab4_m9';
export interface MeasuredStep {
  /** The format pair as 'layers/head', e.g. 'f32/f32', 'q4_0/f32', 'q4_0/q4_0'. */
  format: string;
  batch: number;
  seconds: number;
}
const numOf = (v: unknown): number | undefined => (typeof v === 'number' && isFinite(v) ? v : undefined);
/** 'q4_0' means the same format for layers and head; 'q4_0/f32' is already a pair. */
export const pairKey = (f: string) => (f.includes('/') ? f : `${f}/${f}`);
/**
 * Lab X2.4's rows on one backend: name 'qstep' with format_layers, format_head, batch and
 * seconds (labs/x2/lab4_m9/main.cpp, bench()). Rows named 'step' with a single 'format' are
 * read too. Rows that lack a field are skipped.
 */
export function measuredSteps(reports: ReportsLike, backend: string): MeasuredStep[] {
  const run = reports[backend];
  if (!run) return [];
  return run.results
    .filter((r) => r.lab === LAB_M9 && r.variant === 'solution')
    .flatMap((r) => r.bench)
    .filter((r) => r.name === 'qstep' || r.name === 'step')
    .flatMap((r) => {
      const batch = numOf(r.batch);
      const s = numOf(r.seconds) ?? (numOf(r.ms) != null ? numOf(r.ms)! / 1e3 : undefined);
      const format =
        typeof r.format_layers === 'string' && typeof r.format_head === 'string'
          ? `${r.format_layers}/${r.format_head}`
          : typeof r.format === 'string'
            ? pairKey(r.format)
            : undefined;
      return batch && s && format ? [{ format, batch, seconds: s }] : [];
    });
}
/** Measured speed-up of a format pair over 'f32/f32' at a batch, or null. */
export function measuredSpeedup(rows: MeasuredStep[], format: string, batch: number): number | null {
  const base = rows.find((r) => r.format === 'f32/f32' && r.batch === batch);
  const q = rows.find((r) => r.format === pairKey(format) && r.batch === batch);
  return base && q ? base.seconds / q.seconds : null;
}
/** Model bandwidth utilisation (A1's MBU): weight bytes per step / seconds per step / peak bandwidth. */
export function bandwidthUtilisation(weightBytes: number, seconds: number, bwGBs: number): number {
  return weightBytes / seconds / (bwGBs * 1e9);
}

// ---------------------------------------------------------------- the toy decoder
export interface ToyModel {
  d: number;
  vocab: number;
  layers: number;
  /** vocab × d, row-major; also the LM head's weight. */
  E: number[];
  /** layers of d × d, (out, in) row-major. */
  W: number[][];
  b: number[][];
}
export const TOY = { d: 64, vocab: 256, layers: 2, seed: 30, steps: 32, start: 1, wScale: 3 };

/** A seeded toy decoder: E standard normal, W normal with standard deviation 3 / sqrt(d), small biases (ours). */
export function toyModel(seed = TOY.seed, d = TOY.d, vocab = TOY.vocab, layers = TOY.layers, wScale = TOY.wScale): ToyModel {
  const r = rng(seed);
  const E = Array.from({ length: vocab * d }, () => f32(gauss(r)));
  const W = Array.from({ length: layers }, () => Array.from({ length: d * d }, () => f32((wScale / Math.sqrt(d)) * gauss(r))));
  const b = Array.from({ length: layers }, () => Array.from({ length: d }, () => f32(0.1 * gauss(r))));
  return { d, vocab, layers, E, W, b };
}

export interface ToyFormats {
  /** One format per layer weight. */
  W: MatFormat[];
  E: MatFormat;
}
export const F32_TOY = (m: ToyModel): ToyFormats => ({ W: m.W.map(() => 'f32'), E: 'f32' });

/** The model with each matrix replaced by its dequantised copy. */
export function quantiseModel(m: ToyModel, f: ToyFormats): ToyModel {
  return {
    ...m,
    E: fakeQuant(m.E, m.vocab, m.d, f.E),
    W: m.W.map((w, l) => fakeQuant(w, m.d, m.d, f.W[l])),
  };
}

/** Logits for the next token after `token`, float32 at every operation. */
export function toyLogits(m: ToyModel, token: number): number[] {
  const { d, vocab } = m;
  let h = m.E.slice(token * d, token * d + d);
  for (let l = 0; l < m.layers; l++) {
    const W = m.W[l];
    const t = new Array<number>(d);
    for (let o = 0; o < d; o++) {
      let s = 0;
      for (let k = 0; k < d; k++) s = f32(s + f32(W[o * d + k] * h[k]));
      t[o] = f32(s + m.b[l][o]);
    }
    h = h.map((x, i) => f32(x + t[i]));
  }
  const logits = new Array<number>(vocab);
  for (let v = 0; v < vocab; v++) {
    let s = 0;
    for (let k = 0; k < d; k++) s = f32(s + f32(m.E[v * d + k] * h[k]));
    logits[v] = s;
  }
  return logits;
}

/** Lowest index of the largest value: the project's argmax tie rule (lab 10.3). */
export function argmax(x: number[]): number {
  let best = 0;
  for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
  return best;
}
const top2 = (x: number[]) => {
  const a = argmax(x);
  let b = a === 0 ? 1 : 0;
  for (let i = 0; i < x.length; i++) if (i !== a && x[i] > x[b]) b = i;
  return { a, b, margin: x[a] - x[b] };
};

export interface ToyStep {
  step: number;
  /** The token the float32 run is at, and the one it picks next. */
  input: number;
  f32Next: number;
  /** The quantised model's pick from the same input (teacher-forced on the float32 run). */
  qNext: number;
  /** The float32 logits' top-2 margin. */
  margin: number;
  /** How much the quantisation moved that margin's two logits apart or together: (qa - qb) - (fa - fb). */
  marginShift: number;
  /** RMS of the logits' change, absolute and relative to the float32 logits' RMS. */
  errRms: number;
  rel: number;
  flipped: boolean;
}
export interface ToyRun {
  steps: ToyStep[];
  /** Free-running sequences, start token first. */
  f32Tokens: number[];
  qTokens: number[];
  /** First step at which the free-running sequences differ, or null. */
  firstDivergence: number | null;
  /** Mean of the per-step relative logits error. */
  meanRel: number;
  /** Relative RMS error of each quantised matrix's weights. */
  weightRel: { E: number; W: number[] };
}

const relErr = (a: number[], b: number[]) =>
  Math.sqrt(a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0) / (b.reduce((s, x) => s + x * x, 0) || 1));

export function greedyRun(m: ToyModel, f: ToyFormats, steps = TOY.steps, start = TOY.start): ToyRun {
  const q = quantiseModel(m, f);
  const f32Tokens = [start];
  const qTokens = [start];
  const out: ToyStep[] = [];
  for (let s = 0; s < steps; s++) {
    const input = f32Tokens[s];
    const lf = toyLogits(m, input);
    const lq = toyLogits(q, input);
    const t = top2(lf);
    const qNext = argmax(lq);
    const errRms = Math.sqrt(lf.reduce((acc, x, i) => acc + (x - lq[i]) ** 2, 0) / lf.length);
    out.push({
      step: s, input, f32Next: t.a, qNext, margin: t.margin,
      marginShift: lq[t.a] - lq[t.b] - t.margin, errRms, rel: relErr(lq, lf), flipped: qNext !== t.a,
    });
    f32Tokens.push(t.a);
    qTokens.push(argmax(toyLogits(q, qTokens[s])));
  }
  const div = f32Tokens.findIndex((t, i) => t !== qTokens[i]);
  return {
    steps: out, f32Tokens, qTokens, firstDivergence: div < 0 ? null : div,
    meanRel: out.reduce((s, x) => s + x.rel, 0) / out.length,
    weightRel: { E: relErr(q.E, m.E), W: q.W.map((w, l) => relErr(w, m.W[l])) },
  };
}

/** Bytes of the toy model's weights in its formats (for the explorer's caption). */
export function toyWeightBytes(m: ToyModel, f: ToyFormats): number {
  const one = (fmt: MatFormat, rows: number, cols: number) =>
    fmt === 'affine3' || fmt === 'affine2' ? matrixBytes(formatById(`mlx-${fmt === 'affine3' ? 3 : 2}-g64-f16`), rows, cols) : matrixBytes(fmtOf(fmt), rows, cols); // i18n-ignore: format id
  return one(f.E, m.vocab, m.d) + f.W.reduce((s, w) => s + one(w, m.d, m.d), 0);
}
