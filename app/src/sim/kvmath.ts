// KV-cache memory and decode arithmetic intensity, chapter 20 sections 20.6 and 20.7
// (book pp. 504–510; notes/ch20.md; corrections from notes/today/ch20-facts.md).
//
// The book's formulas, exactly as printed (p = bytes per element, l = layers, h_q = query
// heads, d = head width, b = batch, N = tokens per sequence):
//   Eq. 20.7   per token, MHA      2 l h_q d p
//   Eq. 20.8   total, MHA          b N 2 l h_q d p
//   Eq. 20.10  total, MQA          b N 2 l d p
//   Eq. 20.12  total, GQA          b N 2 l (h_q / g_q) d p
//   Eq. 20.14  total, MLA          b N l d p                         (as printed; see below)
//   Eq. 20.9   AI, MHA             N / (1 + N)               ~ 1
//   Eq. 20.11  AI, MQA             N h_q / (h_q + N)         ~ h_q
//   Eq. 20.13  AI, GQA             N h_q / (h_q + (h_q/g_q) N) ~ g_q
//   Eq. 20.15  AI, MLA             2 N h_q / (h_q + N)       ~ 2 h_q   (as printed)
// The intensities count one multiply-add per element read, with p = 1 (book p. 507).
//
// Corrections (facts file):
//   E3  g_q is the number of query heads per K/V head ([13]'s definition), so a model has
//       h_kv = h_q / g_q K/V heads; the GQA paper's G (number of groups) is h_kv.
//   E4  [13] gives MLA's intensity as L / (1 + L/(2 h_q)) = 2 N h_q / (2 h_q + N); the book
//       prints h_q + N in the denominator. The limit 2 h_q is the same.
//   E5  MLA caches one latent per token per layer of width d_c + d_h^R (DeepSeek-V2:
//       512 + 64 = 4.5 d_h), not d; K and V are never formed at inference.
//   E2  The book's "PaLM 2" row is PaLM 62B, which uses multi-query attention with one K/V
//       head of 256; GPT-3's context window is 2,048 tokens, not 4,096.
// Units: the book's "GB" in these examples are binary (2^30 bytes, notes).
//
// Converting the book's intensity (multiply-adds per element) to FLOP per byte at p bytes
// per element: each multiply-add is 2 FLOPs, each element p bytes, so FLOP/B = 2 AI / p.
// That gives 1 FLOP/B for MHA decode in 16-bit, as [13] says (facts E4), and 0.5 in fp32.
//
// Erasable TypeScript only and no imports: the browser checks import this file directly.

import { tr } from '../lib/locale.ts';

export const KiB = 1024;
export const MiB = 1024 ** 2;
export const GiB = 1024 ** 3;

// ------------------------------------------------------------------ the book's equations, as printed

/** Eq. 20.7: KV-cache bytes per token, MHA. */
export const eq207 = (l: number, hq: number, d: number, p: number) => 2 * l * hq * d * p;
/** Eq. 20.8: total KV-cache bytes, MHA. */
export const eq208 = (b: number, N: number, l: number, hq: number, d: number, p: number) => b * N * 2 * l * hq * d * p;
/** Eq. 20.10: total, MQA (one K and one V head). */
export const eq210 = (b: number, N: number, l: number, d: number, p: number) => b * N * 2 * l * d * p;
/** Eq. 20.12: total, GQA with g_q query heads per K/V head. */
export const eq212 = (b: number, N: number, l: number, hq: number, gq: number, d: number, p: number) =>
  b * N * 2 * l * (hq / gq) * d * p;
/** Eq. 20.14 as printed: a latent of width d per token per layer. */
export const eq214 = (b: number, N: number, l: number, d: number, p: number) => b * N * l * d * p;

/** Eq. 20.9: MHA decode intensity, multiply-adds per element. */
export const aiMHA = (N: number) => N / (1 + N);
/** Eq. 20.11: MQA. */
export const aiMQA = (N: number, hq: number) => (N * hq) / (hq + N);
/** Eq. 20.13: GQA, g_q query heads per K/V head. */
export const aiGQA = (N: number, hq: number, gq: number) => (N * hq) / (hq + (hq / gq) * N);
/** MLA per [13] (facts E4): 2 N h_q / (2 h_q + N). */
export const aiMLA = (N: number, hq: number) => (2 * N * hq) / (2 * hq + N);
/** Eq. 20.15 as printed: 2 N h_q / (h_q + N). */
export const aiMLAPrinted = (N: number, hq: number) => (2 * N * hq) / (hq + N);

/** The book's intensity (multiply-adds per element) as FLOP per byte at p bytes per element. */
export const toFlopPerByte = (ai: number, p: number) => (2 * ai) / p;

// ------------------------------------------------------------------ shapes and variants

export type Variant = 'MHA' | 'GQA' | 'MQA' | 'MLA';

export interface AttnShape {
  /** Layers. */
  l: number;
  /** Query heads. */
  hq: number;
  /** K/V heads (h_q for MHA, 1 for MQA). Ignored for MLA. */
  hkv: number;
  /** Head width. */
  d: number;
  /** MLA only: the cached latent's widths, d_c (compressed K and V) and d_h^R (decoupled RoPE key). */
  mla?: { dc: number; dr: number };
}

export function variantOf(s: AttnShape): Variant {
  if (s.mla) return 'MLA';
  if (s.hkv === s.hq) return 'MHA';
  if (s.hkv === 1) return 'MQA';
  return 'GQA';
}

/** Query heads per K/V head ([13]'s g_q, what Eqs. 20.12 and 20.13 use) and the GQA paper's G. */
export function gqaTerms(hq: number, hkv: number): { gq: number; G: number } {
  return { gq: hq / hkv, G: hkv };
}

/**
 * The K/V head query head h reads: consecutive query heads share one, as in FlashAttention's
 * example (6 query heads, 2 K/V heads: heads 0, 1, 2 read 0 and 3, 4, 5 read 1; facts A8)
 * and llama.cpp's gqa_ratio (facts D1).
 */
export function kvHeadOf(h: number, hq: number, hkv: number): number {
  return Math.floor(h / (hq / hkv));
}

/** Cached values per token per layer, corrected: 2 h_kv d, or d_c + d_h^R for MLA. */
export function valuesPerTokenLayer(s: AttnShape): number {
  return s.mla ? s.mla.dc + s.mla.dr : 2 * s.hkv * s.d;
}

/** KV-cache bytes per token for the whole model (all layers), corrected. */
export function bytesPerToken(s: AttnShape, p: number): number {
  return s.l * valuesPerTokenLayer(s) * p;
}

/** KV-cache bytes for b sequences of N tokens, corrected. */
export function cacheTotal(s: AttnShape, b: number, N: number, p: number): number {
  return b * N * bytesPerToken(s, p);
}

/** The same shape as another variant, for the comparison: MHA keeps every head, MQA one, GQA hkv. */
export function asVariant(s: AttnShape, v: Variant, hkv = s.hkv, mla = s.mla ?? { dc: 4 * s.d, dr: s.d / 2 }): AttnShape {
  switch (v) {
    case 'MHA':
      return { l: s.l, hq: s.hq, hkv: s.hq, d: s.d };
    case 'MQA':
      return { l: s.l, hq: s.hq, hkv: 1, d: s.d };
    case 'GQA':
      return { l: s.l, hq: s.hq, hkv, d: s.d };
    case 'MLA':
      return { l: s.l, hq: s.hq, hkv: s.hq, d: s.d, mla };
  }
}

/**
 * Decode intensity of a shape at N cached tokens, multiply-adds per element: [13]'s Table 1
 * (facts E4). The book's printed MLA formula is aiMLAPrinted.
 */
export function decodeAI(s: AttnShape, N: number): number {
  switch (variantOf(s)) {
    case 'MHA':
      return aiMHA(N);
    case 'MQA':
      return aiMQA(N, s.hq);
    case 'GQA':
      return aiGQA(N, s.hq, s.hq / s.hkv);
    case 'MLA':
      return aiMLA(N, s.hq);
  }
}

/** The cache bytes the book's printed equation gives for this shape and variant (per token, all layers). */
export function bytesPerTokenPrinted(s: AttnShape, p: number): number {
  switch (variantOf(s)) {
    case 'MHA':
      return eq208(1, 1, s.l, s.hq, s.d, p);
    case 'MQA':
      return eq210(1, 1, s.l, s.d, p);
    case 'GQA':
      return eq212(1, 1, s.l, s.hq, s.hq / s.hkv, s.d, p);
    case 'MLA':
      return eq214(1, 1, s.l, s.d, p);
  }
}

// ------------------------------------------------------------------ batching (p. 505)

/**
 * A linear layer of inF inputs and outF outputs applied to b rows, p bytes per element,
 * counting the weights once plus every input and output row (facts F1 reproduces the
 * book's 1 and 315 FLOP/B this way). FLOP per byte.
 */
export function linearAI(b: number, inF: number, outF: number, p: number): number {
  return (2 * b * inF * outF) / (p * (inF * outF + b * (inF + outF)));
}

/**
 * Decode attention for a batch of b requests, each with its own cache of N tokens: every
 * request's bytes and FLOPs scale together, so b cancels (Fig. 20.17). FLOP per byte.
 */
export function batchedDecodeAttentionAI(_b: number, N: number, p: number): number {
  return toFlopPerByte(aiMHA(N), p);
}

/** Smallest batch at which the linear layer's intensity reaches `ridge` FLOP/B, or null if never. */
export function batchToReach(ridge: number, inF: number, outF: number, p: number, maxB = 1 << 20): number | null {
  for (let b = 1; b <= maxB; b *= 2) {
    if (linearAI(b, inF, outF, p) >= ridge) {
      let lo = Math.max(1, b / 2);
      let hi = b;
      while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (linearAI(mid, inF, outF, p) >= ridge) hi = mid;
        else lo = mid + 1;
      }
      return lo;
    }
  }
  return null;
}

// ------------------------------------------------------------------ memory budget

/** Largest number of sequences of N tokens whose caches fit beside the weights, or 0. */
export function maxBatch(memBytes: number, weightBytes: number, s: AttnShape, N: number, p: number): number {
  const per = cacheTotal(s, 1, N, p);
  if (per <= 0) return 0;
  return Math.max(0, Math.floor((memBytes - weightBytes) / per));
}

/** Tokens of cache (summed over the batch) that fit in `bytes`. */
export function tokensThatFit(bytes: number, s: AttnShape, p: number): number {
  return Math.max(0, Math.floor(bytes / bytesPerToken(s, p)));
}

/**
 * The floor on one decode step: every weight and every cached byte read once at the
 * measured bandwidth (bytes per second). Ours, not the book's: reads only, nothing cached,
 * no overlap with anything else. Seconds.
 */
export function stepFloorSeconds(weightBytes: number, cacheBytes: number, bytesPerSecond: number): number {
  return (weightBytes + cacheBytes) / bytesPerSecond;
}

// ------------------------------------------------------------------ element sizes and presets

export interface ElementSize {
  key: string;
  label: string;
  bytes: number;
  note: string;
}

/** Bytes per element: the labs' fp32, 16-bit, and llama.cpp's cache types (facts D3). */
export const ELEMENT_SIZES: ElementSize[] = [
  {
    key: 'f32',
    get label() {
      return tr('fp32, 4 bytes', 'fp32, 4 byte');
    },
    bytes: 4,
    get note() {
      return tr("the labs' precision", 'độ chính xác của các lab');
    },
  },
  {
    key: 'f16',
    get label() {
      return tr('fp16 or bf16, 2 bytes', 'fp16 hoặc bf16, 2 byte');
    },
    bytes: 2,
    get note() {
      return tr("the book's examples; llama.cpp's default cache type", 'các ví dụ của sách; kiểu cache mặc định của llama.cpp');
    },
  },
  {
    key: 'f8',
    label: 'fp8 e4m3, 1 byte', // i18n-ignore: reads the same in Vietnamese (type name, number, unit)
    bytes: 1,
    get note() {
      return tr('llama.cpp cache type f8_e4m3', 'kiểu cache f8_e4m3 của llama.cpp');
    },
  },
  {
    key: 'q8_0',
    get label() {
      return tr('q8_0, 34 bytes per 32 values', 'q8_0, 34 byte cho mỗi 32 giá trị');
    },
    bytes: 34 / 32,
    get note() {
      return tr('llama.cpp: a half-precision scale plus 32 int8 values', 'llama.cpp: một scale độ chính xác một nửa (half) cộng 32 giá trị int8');
    },
  },
  {
    key: 'q4_0',
    get label() {
      return tr('q4_0, 18 bytes per 32 values', 'q4_0, 18 byte cho mỗi 32 giá trị');
    },
    bytes: 18 / 32,
    get note() {
      return tr('llama.cpp: a half-precision scale plus 16 bytes of 4-bit values', 'llama.cpp: một scale độ chính xác một nửa (half) cộng 16 byte giá trị 4-bit');
    },
  },
];

export interface Preset {
  key: string;
  label: string;
  shape: AttnShape;
  /** Context window from the facts file, when it gives one. */
  context?: number;
  /** Parameter count, only where the facts file or the book gives one. */
  params?: number;
  paramsNote?: string;
  /** Where the shape comes from. */
  source: string;
  /** A remark shown with the preset. */
  note?: string;
}

/** Shapes from the book (p. 506) and the facts file (E2, E5, E7) only. */
export const PRESETS: Preset[] = [
  {
    key: 'llama2-7b',
    get label() {
      return tr('Llama 2 7B (the book’s example)', 'Llama 2 7B (ví dụ của sách)');
    },
    shape: { l: 32, hq: 32, hkv: 32, d: 128 },
    context: 4096,
    params: 7e9,
    get paramsNote() {
      return tr('the book’s round 7 billion (p. 505)', 'con số tròn 7 tỷ của sách (p. 505)');
    },
    get source() {
      return tr('book p. 506; facts E2 (32 layers, 32 heads, 4,096 wide, 4k context, no GQA)', 'sách p. 506; facts E2 (32 lớp, 32 head, rộng 4,096, ngữ cảnh 4k, không GQA)');
    },
  },
  {
    key: 'gpt3',
    label: 'GPT-3 175B', // i18n-ignore: model name
    shape: { l: 96, hq: 96, hkv: 96, d: 128 },
    context: 2048,
    params: 175.0e9,
    paramsNote: '175.0B (facts E2, GPT-3 Table 2.1)', // i18n-ignore: parameter count and citation
    get source() {
      return tr('facts E2: 96 layers, 96 heads of 128, a 2,048-token window', 'facts E2: 96 lớp, 96 head kích thước 128, cửa sổ 2,048 token');
    },
    get note() {
      return tr('The book’s 18 GB uses 4,096 tokens, twice GPT-3’s window.', 'Con số 18 GB của sách dùng 4,096 token, gấp đôi cửa sổ của GPT-3.');
    },
  },
  {
    key: 'palm2-printed',
    get label() {
      return tr('The book’s “PaLM 2” row, as printed', 'Hàng “PaLM 2” của sách, đúng như in');
    },
    shape: { l: 64, hq: 32, hkv: 32, d: 256 },
    get source() {
      return tr('book p. 506: 64 layers, hidden dimension 8,192 (split here as 32 heads of 256, PaLM 62B’s)', 'sách p. 506: 64 lớp, chiều ẩn 8,192 (ở đây chia thành 32 head kích thước 256, như PaLM 62B)');
    },
    get note() {
      return tr('PaLM 2’s architecture was never published (facts E2). These numbers are PaLM 62B’s, computed as if it used MHA.', 'Kiến trúc của PaLM 2 chưa bao giờ được công bố (facts E2). Các con số này là của PaLM 62B, được tính như thể nó dùng MHA.');
    },
  },
  {
    key: 'palm-62b',
    get label() {
      return tr('PaLM 62B, as it really is (MQA)', 'PaLM 62B, đúng như thực tế (MQA)');
    },
    shape: { l: 64, hq: 32, hkv: 1, d: 256 },
    params: 62e9,
    paramsNote: '62B (facts E2)', // i18n-ignore: parameter count and citation
    get source() {
      return tr('facts E2: 64 layers, 32 heads, d_model 8,192, head size 256, multi-query attention', 'facts E2: 64 lớp, 32 head, d_model 8,192, kích thước head 256, multi-query attention');
    },
  },
  {
    key: 'mistral-7b',
    get label() {
      return tr('Mistral 7B (GQA, 8 K/V heads)', 'Mistral 7B (GQA, 8 head K/V)');
    },
    shape: { l: 32, hq: 32, hkv: 8, d: 128 },
    context: 8192,
    get source() {
      return tr('facts E2: 32 layers, 32 heads of 128, 8 K/V heads, context 8,192', 'facts E2: 32 lớp, 32 head kích thước 128, 8 head K/V, ngữ cảnh 8,192');
    },
    get note() {
      return tr('No parameter count in our facts file: type one if you want the weights counted.', 'Tệp facts của chúng tôi không có số tham số: hãy nhập một số nếu bạn muốn tính cả trọng số.');
    },
  },
  {
    key: 'mixtral',
    get label() {
      return tr('Mixtral (Mistral 7B’s attention, 8 experts per layer)', 'Mixtral (attention của Mistral 7B, 8 expert mỗi lớp)');
    },
    shape: { l: 32, hq: 32, hkv: 8, d: 128 },
    params: 47e9,
    get paramsNote() {
      return tr('47B resident, 13B used per token (facts E7)', '47B thường trú, 13B được dùng cho mỗi token (facts E7)');
    },
    get source() {
      return tr('facts E7: the same architecture as Mistral 7B, 8 feed-forward experts per layer', 'facts E7: cùng kiến trúc với Mistral 7B, 8 expert feed-forward mỗi lớp');
    },
    get note() {
      return tr('Every expert must be in memory, so the weights count all 47B.', 'Mọi expert đều phải nằm trong bộ nhớ, nên phần trọng số tính đủ cả 47B.');
    },
  },
  {
    key: 'deepseek-v2',
    label: 'DeepSeek-V2 (MLA)', // i18n-ignore: model name
    shape: { l: 60, hq: 128, hkv: 128, d: 128, mla: { dc: 512, dr: 64 } },
    source: 'facts E2, E5: 60 layers, 128 heads of 128, d_c = 512, d_h^R = 64',
    get note() {
      return tr('No parameter count in our facts file: type one if you want the weights counted.', 'Tệp facts của chúng tôi không có số tham số: hãy nhập một số nếu bạn muốn tính cả trọng số.');
    },
  },
];

export const presetByKey = (key: string) => PRESETS.find((p) => p.key === key);
