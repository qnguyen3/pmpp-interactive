// Argmax tie rules in the projects module X3 reads (section X3.6), and where each sampling
// stage runs. Everything here is our transcription of source read in the facts passes:
//
//   llama.cpp CUDA   ggml-cuda/argmax.cu: a pair is replaced only if (val > maxval), in the
//                    strided loop and in the xor butterfly (notes/today/ch10-facts.md D2)
//   llama.cpp Metal  kernels/misc.metal: simd_max of the values, then simd_max of the indices
//                    of the lanes holding that maximum (ch10 D2)
//   llama.cpp host   src/llama-sampler.cpp greedy: a strict > scan from index 0 (ch10 D2)
//   torch.argmax     documented "the indices of the first maximal value"; comparator: NaN
//                    wins, lower index on equal values (ch10 D4)
//   course M4        lab 10.3 argmaxCombine: otherV > v || (otherV == v && otherI < i), with
//                    the identity pair (-FLT_MAX, INT_MAX)
//
// The model is one warp and one column per lane, so a row has exactly WARP columns. The
// strided loop over longer rows and the second stage across warps are left out.

import { tr } from '../lib/locale.ts';

export const WARP = 32;
/** The butterfly's offsets in the order the CUDA kernel visits them. */
export const OFFSETS = [16, 8, 4, 2, 1] as const;
export const FLT_MAX = 3.4028234663852886e38;
export const INT_MAX = 2147483647;

export interface Pair {
  v: number;
  i: number;
}

export interface ButterflyStage {
  /** null for the state after the per-lane load, else the xor offset just applied. */
  offset: number | null;
  lanes: Pair[];
  /** Lanes whose pair changed at this stage. */
  changed: number[];
}

/** Each lane's pair after its share of the row: one column per lane, kept only if val > -FLT_MAX. */
function cudaLoad(row: number[]): Pair[] {
  return Array.from({ length: WARP }, (_, lane) => {
    const p: Pair = { v: -FLT_MAX, i: -1 };
    for (let c = lane; c < row.length; c += WARP) if (row[c] > p.v) { p.v = row[c]; p.i = c; }
    return p;
  });
}

/** llama.cpp CUDA argmax_f32 for one warp: every lane's (value, index) after the load and after each offset. */
export function cudaButterfly(row: number[]): ButterflyStage[] {
  if (row.length !== WARP) throw new Error(`the model takes a row of ${WARP} columns`);
  let lanes = cudaLoad(row);
  const stages: ButterflyStage[] = [{ offset: null, lanes, changed: [] }];
  for (const off of OFFSETS) {
    const prev = lanes;
    lanes = prev.map((own, lane) => {
      const other = prev[lane ^ off];
      return other.v > own.v ? { ...other } : { ...own };
    });
    const changed = lanes.flatMap((p, k) => (p.i !== prev[k].i || p.v !== prev[k].v ? [k] : []));
    stages.push({ offset: off, lanes, changed });
  }
  return stages;
}

/** The index llama.cpp's CUDA kernel writes: lane 0's after the butterfly (ch10 D2). -1 if nothing beat -FLT_MAX. */
export function cudaArgmax(row: number[]): number {
  const st = cudaButterfly(row);
  return st[st.length - 1].lanes[0].i;
}

/**
 * llama.cpp Metal kernel_argmax_f32 for one SIMD-group of 32, one column per lane:
 * max_val = simd_max(lmax); arg = simd_max(select(-1, larg, lmax == max_val)).
 * The per-lane start (-INFINITY, -1) with a strict > is our assumption; the facts pass quotes only
 * the two simd_max lines. NaN handling in simd_max is undocumented (ch10 C1), so NaN is skipped by
 * the per-lane test before it reaches simd_max.
 */
export function metalLanes(row: number[]): Pair[] {
  if (row.length !== WARP) throw new Error(`the model takes a row of ${WARP} columns`);
  return row.map((x, lane) => (x > -Infinity ? { v: x, i: lane } : { v: -Infinity, i: -1 }));
}
export function metalArgmax(row: number[]): number {
  const lanes = metalLanes(row);
  const max = Math.max(...lanes.map((p) => p.v));
  return Math.max(...lanes.map((p) => (p.v === max ? p.i : -1)));
}
/** The lanes whose value equals the SIMD-group maximum: the ones simd_max over indices looks at. */
export function metalMaxLanes(row: number[]): number[] {
  const lanes = metalLanes(row);
  const max = Math.max(...lanes.map((p) => p.v));
  return lanes.flatMap((p, k) => (p.v === max && p.i >= 0 ? [k] : []));
}

/** llama.cpp host greedy: selected = 0, then i replaces it only if logit[i] > logit[selected]. */
export function hostGreedy(row: number[]): number {
  let s = 0;
  for (let i = 1; i < row.length; i++) if (row[i] > row[s]) s = i;
  return s;
}

/** torch.argmax by its comparator: NaN wins (the lowest NaN index), else the first maximum. */
export function torchArgmax(row: number[]): number {
  let bi = 0;
  for (let i = 1; i < row.length; i++) {
    const a = row[i];
    const b = row[bi];
    const better = Number.isNaN(a) ? !Number.isNaN(b) : Number.isNaN(b) ? false : a > b;
    if (better) bi = i;
  }
  return bi;
}

/** Course M4 (lab 10.3): combine with the identity (-FLT_MAX, INT_MAX); the result does not depend on order. */
export function m4Argmax(row: number[]): number {
  let v = -FLT_MAX;
  let i = INT_MAX;
  row.forEach((x, c) => {
    if (x > v || (x === v && c < i)) { v = x; i = c; }
  });
  return i;
}

export type Impl = 'cuda' | 'metal' | 'host' | 'torch' | 'm4';
export const IMPLS: { id: Impl; label: string; rule: string; cite: string }[] = [
  {
    id: 'cuda',
    get label() { return tr('llama.cpp CUDA (argmax.cu, also the HIP file)', 'llama.cpp CUDA (argmax.cu, cũng là file HIP)'); },
    get rule() { return tr("lane 0's pair after the xor butterfly; a pair is replaced only by a strictly larger value", 'cặp của lane 0 sau butterfly xor; một cặp chỉ bị thay bởi một giá trị lớn hơn hẳn'); },
    cite: 'ch10 D2', // i18n-ignore: citation
  },
  {
    id: 'metal',
    get label() { return tr('llama.cpp Metal (kernel_argmax_f32)', 'llama.cpp Metal (kernel_argmax_f32)'); },
    get rule() { return tr('the largest index among the lanes that hold the maximum', 'chỉ số lớn nhất trong các lane đang giữ cực đại'); },
    cite: 'ch10 D2', // i18n-ignore: citation
  },
  {
    id: 'host',
    get label() { return tr('llama.cpp host greedy sampler', 'sampler greedy trên host của llama.cpp'); },
    get rule() { return tr('the lowest index: a strict > scan from index 0', 'chỉ số nhỏ nhất: quét bằng phép so sánh > chặt từ chỉ số 0'); },
    cite: 'ch10 D2', // i18n-ignore: citation
  },
  {
    id: 'torch',
    get label() { return tr('torch.argmax (transformers greedy)', 'torch.argmax (greedy của transformers)'); },
    get rule() { return tr('documented: the first maximal value', 'theo tài liệu: giá trị cực đại đầu tiên'); },
    cite: 'ch10 D4, x1 D4', // i18n-ignore: citation
  },
  {
    id: 'm4',
    get label() { return tr("the course's M4 (lab 10.3)", 'M4 của khóa học (lab 10.3)'); },
    get rule() { return tr('the lowest index among the maxima, by its combine rule', 'chỉ số nhỏ nhất trong các cực đại, theo quy tắc kết hợp của nó'); },
    cite: 'lab 10.3 argmaxCombine', // i18n-ignore: citation
  },
];
export function argmaxBy(impl: Impl, row: number[]): number {
  switch (impl) {
    case 'cuda': return cudaArgmax(row);
    case 'metal': return metalArgmax(row);
    case 'host': return hostGreedy(row);
    case 'torch': return torchArgmax(row);
    case 'm4': return m4Argmax(row);
  }
}

/**
 * The row the tie stepper shows: 32 distinct values below TIE_VALUE, and TIE_VALUE at every planted
 * column. Values are a fixed permutation so that no two unplanted columns tie.
 */
export const TIE_VALUE = 4;
export function tieRow(planted: number[]): number[] {
  const row = Array.from({ length: WARP }, (_, c) => ((c * 13) % WARP) / 10);
  for (const c of planted) row[c] = TIE_VALUE;
  return row;
}

// ---------------------------------------------------------------- where each stage runs

export type Stage = 'penalties' | 'top-k' | 'top-p' | 'temperature' | 'draw';
export const STAGE_LIST: Stage[] = ['penalties', 'top-k', 'top-p', 'temperature', 'draw'];
export type System = 'llama-default' | 'llama-bs' | 'transformers' | 'mlx-lm';
export const SYSTEMS: { id: System; label: string }[] = [
  { id: 'llama-default', get label() { return tr('llama.cpp, defaults', 'llama.cpp, mặc định'); } },
  { id: 'llama-bs', get label() { return tr('llama.cpp with -bs', 'llama.cpp với -bs'); } },
  { id: 'transformers', get label() { return tr('transformers generate', 'transformers generate'); } },
  { id: 'mlx-lm', get label() { return tr('mlx-lm', 'mlx-lm'); } },
];
export type Bin = 'host' | 'kernels' | 'framework';
export const BINS: { id: Bin; label: string }[] = [
  { id: 'host', get label() { return tr('host CPU code', 'mã CPU trên host'); } },
  { id: 'kernels', get label() { return tr("the project's GPU graph (one backend kernel per op)", 'graph GPU của dự án (mỗi op một kernel backend)'); } },
  { id: 'framework', get label() { return tr("framework tensor ops on the logits' device", 'tensor op của framework trên thiết bị chứa logits'); } },
];

export interface Cell {
  system: System;
  stage: Stage;
  bin: Bin;
  /** Facts item that settles it. */
  cite: string;
  /** What the source shows, in one line. */
  evidence: string;
}

/**
 * Only cells a facts item settles; every one of the 4 × 5 here is settled. A stage or system added
 * later goes in only with the item that settles it, never by analogy.
 */
export const STAGES: Cell[] = [
  { system: 'llama-default', stage: 'penalties', bin: 'host', cite: 'x1 D1, D2; ch09 D4', get evidence() { return tr('backend_sampling = false; llama_sampler_penalties_apply walks the candidate array on the host', 'backend_sampling = false; llama_sampler_penalties_apply duyệt mảng ứng viên trên host'); } }, // i18n-ignore: citation
  { system: 'llama-default', stage: 'top-k', bin: 'host', cite: 'x1 D1; ch14 D3', get evidence() { return tr('host top-k is a partial sort of the candidate array', 'top-k trên host là một phép sắp xếp một phần mảng ứng viên'); } }, // i18n-ignore: citation
  { system: 'llama-default', stage: 'top-p', bin: 'host', cite: 'x1 D1; ch14 D3', get evidence() { return tr('host top-p: softmax, adaptive partial sort, running sum', 'top-p trên host: softmax, sắp xếp một phần thích ứng, tổng tích lũy'); } }, // i18n-ignore: citation
  { system: 'llama-default', stage: 'temperature', bin: 'host', cite: 'x1 D1, D2', get evidence() { return tr('llama_sampler_temp_impl divides each logit by temp', 'llama_sampler_temp_impl chia mỗi logit cho temp'); } }, // i18n-ignore: citation
  { system: 'llama-default', stage: 'draw', bin: 'host', cite: 'x1 D3', get evidence() { return tr('dist: std::mt19937 and a running sum compared with u × total', 'dist: std::mt19937 và một tổng tích lũy so với u × total'); } }, // i18n-ignore: citation
  { system: 'llama-bs', stage: 'penalties', bin: 'kernels', cite: 'ch09 D4', get evidence() { return tr('host counts uploaded, scattered with ggml_set_rows and gathered with ggml_get_rows', 'số đếm từ host được tải lên, rải bằng ggml_set_rows và gom bằng ggml_get_rows'); } }, // i18n-ignore: citation
  { system: 'llama-bs', stage: 'top-k', bin: 'kernels', cite: 'ch14 D3, D2', get evidence() { return tr('backend top-k calls ggml_top_k (top-k.cu, Metal argsort.metal)', 'top-k trên backend gọi ggml_top_k (top-k.cu, Metal argsort.metal)'); } }, // i18n-ignore: citation
  { system: 'llama-bs', stage: 'top-p', bin: 'kernels', cite: 'ch14 D3; ch11 D2', get evidence() { return tr('ggml_argsort, ggml_soft_max, ggml_cumsum, a ggml_step mask', 'ggml_argsort, ggml_soft_max, ggml_cumsum, một mặt nạ ggml_step'); } }, // i18n-ignore: citation
  { system: 'llama-bs', stage: 'temperature', bin: 'kernels', cite: 'x1 D2', evidence: 'ggml_scale(ctx, logits, 1.0f / temp)' }, // i18n-ignore: citation
  { system: 'llama-bs', stage: 'draw', bin: 'kernels', cite: 'x1 D3', get evidence() { return tr('ggml_soft_max, ggml_cumsum, ggml_step, ggml_sum; u still drawn on the host', 'ggml_soft_max, ggml_cumsum, ggml_step, ggml_sum; u vẫn được rút trên host'); } }, // i18n-ignore: citation
  { system: 'transformers', stage: 'penalties', bin: 'framework', cite: 'x1 D4', get evidence() { return tr('RepetitionPenaltyLogitsProcessor: torch.where on the scores', 'RepetitionPenaltyLogitsProcessor: torch.where trên các điểm số'); } }, // i18n-ignore: citation
  { system: 'transformers', stage: 'top-k', bin: 'framework', cite: 'x1 D4', get evidence() { return tr('TopKLogitsWarper: torch.topk and a mask', 'TopKLogitsWarper: torch.topk và một mặt nạ'); } }, // i18n-ignore: citation
  { system: 'transformers', stage: 'top-p', bin: 'framework', cite: 'x1 D4; ch11 D4', get evidence() { return tr('TopPLogitsWarper, tensor ops', 'TopPLogitsWarper, các tensor op'); } }, // i18n-ignore: citation
  { system: 'transformers', stage: 'temperature', bin: 'framework', cite: 'x1 D4', evidence: 'TemperatureLogitsWarper: scores / temperature' }, // i18n-ignore: citation
  { system: 'transformers', stage: 'draw', bin: 'framework', cite: 'x1 D4, D5', get evidence() { return tr('torch.multinomial(probs, 1): argmax(p / q) on the tensor’s device', 'torch.multinomial(probs, 1): argmax(p / q) trên thiết bị chứa tensor'); } }, // i18n-ignore: citation
  { system: 'mlx-lm', stage: 'penalties', bin: 'framework', cite: 'x1 D8', evidence: 'mx.where(selected < 0, selected * penalty, selected / penalty)' }, // i18n-ignore: citation
  { system: 'mlx-lm', stage: 'top-k', bin: 'framework', cite: 'x1 D8; ch12 D6', get evidence() { return tr('mx.argpartition and mx.put_along_axis', 'mx.argpartition và mx.put_along_axis'); } }, // i18n-ignore: citation
  { system: 'mlx-lm', stage: 'top-p', bin: 'framework', cite: 'x1 D8; ch11 D3', evidence: 'mx.where(logprobs < threshold, -inf, logprobs)' }, // i18n-ignore: citation
  { system: 'mlx-lm', stage: 'temperature', bin: 'framework', cite: 'x1 D8', get evidence() { return tr('logits * (1.0 / temp) inside categorical_sampling', 'logits * (1.0 / temp) bên trong categorical_sampling'); } }, // i18n-ignore: citation
  { system: 'mlx-lm', stage: 'draw', bin: 'framework', cite: 'x1 D7, D8', get evidence() { return tr('mx.random.categorical with a Threefry key', 'mx.random.categorical với một khóa Threefry'); } }, // i18n-ignore: citation
];
export const cellOf = (system: System, stage: Stage) => STAGES.find((c) => c.system === system && c.stage === stage);

/** The order each system applies its stages in (x1 D1, D4, D8): the reason the table's columns are not one pipeline. */
export const ORDER: Partial<Record<System, string[]>> = {
  get 'llama-default'() {
    return ['penalties', 'DRY', 'top-n-sigma', 'top-k', 'typical', 'top-p', 'min-p', 'XTC', 'temperature', tr('dist or greedy', 'dist hoặc greedy')];
  },
  get transformers() {
    return [tr('repetition penalty and other processors', 'hình phạt lặp lại và các processor khác'), 'temperature', 'top-k', 'top-p', 'min-p', 'softmax', 'torch.multinomial'];
  },
  get 'mlx-lm'() {
    return [tr('logits processors (penalties)', 'logits processor (hình phạt)'), 'top-p', 'min-p', 'XTC', 'top-k', tr('temperature and categorical', 'nhiệt độ và categorical')];
  },
};
