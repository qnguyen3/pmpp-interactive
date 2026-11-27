// Where a decoder layer's arithmetic goes, and which kernel each block of Fig. 20.1 is
// (chapter 20, section 20.1).
//
// FLOP counts are ours, from the book's shapes (pp. 480-483; notes/ch20.md §20.1-20.2),
// with FLOPs = 2 per multiply-add:
//   - Q, K, V and output projections: four d_model x d_model matrices, 4 d_model^2
//     multiply-adds per token (multi-head attention: every head has its own K and V).
//   - Feed-forward: the book's two linear layers with a ReLU (2 d_model d_ff per token), or
//     the three-matrix SwiGLU of today's open models (3 d_model d_ff; facts E8).
//   - Q K^T and P V: over all heads, d_model multiply-adds per (query, key) pair each.
//     Prefill of N tokens has N^2 pairs, or N(N+1)/2 when the masked half is skipped (the
//     book's kernels do not skip it; FlashAttention-2 does, facts A7). One decode token at
//     context N has N pairs (itself included); no mask is needed (Fig. 20.6).
//
// What it leaves out: the softmax, norms, biases, activations and residual adds (all
// O(N d) or O(N^2) with a small constant), the embedding and the un-embedding to the
// vocabulary, and grouped-query attention's smaller K and V projections.
//
// Erasable TypeScript only, no runtime imports: the browser checks import this file in Node.

import { tr } from '../lib/locale.ts';

export type FfnKind = 'relu' | 'swiglu';
export const FFN_LABEL: Record<FfnKind, string> = {
  get relu() {
    return tr("the book's two matrices with a ReLU", 'hai ma trận của sách, với một ReLU');
  },
  get swiglu() {
    return tr("three matrices (SwiGLU, as in today's open models)", 'ba ma trận (SwiGLU, như trong các mô hình mở ngày nay)');
  },
};
export const FFN_MATRICES: Record<FfnKind, number> = { relu: 2, swiglu: 3 };

export type Phase = 'prefill' | 'decode';

export interface LayerShape {
  dModel: number;
  /** d_ff / d_model. The book's original transformer uses 4. */
  ffRatio: number;
  ffn: FfnKind;
  /** Skip the masked half of Q K^T and P V in prefill. */
  causalSkip: boolean;
}

export const BOOK_SHAPE: Omit<LayerShape, 'dModel'> = { ffRatio: 4, ffn: 'relu', causalSkip: false };

export interface LayerFlops {
  /** Q, K, V and output projections. */
  proj: number;
  /** Q K^T over all heads. */
  qk: number;
  /** P V over all heads. */
  pv: number;
  /** Feed-forward. */
  ffn: number;
  total: number;
  /** (qk + pv) / total. */
  attnShare: number;
  /** (qk + pv) / (proj + ffn). */
  attnOverLinear: number;
}

/** Multiply-adds of the linear layers, per token. */
export const linearMacsPerToken = (s: LayerShape) => 4 * s.dModel * s.dModel + FFN_MATRICES[s.ffn] * s.dModel * (s.ffRatio * s.dModel);

/** Query-key pairs the two attention products cover. */
export function attentionPairs(n: number, phase: Phase, causalSkip: boolean): number {
  if (phase === 'decode') return n;
  return causalSkip ? (n * (n + 1)) / 2 : n * n;
}

/** FLOPs of one layer: prefill of n tokens, or one decode token at context n. */
export function layerFlops(n: number, s: LayerShape, phase: Phase): LayerFlops {
  const tokens = phase === 'prefill' ? n : 1;
  const proj = 2 * tokens * 4 * s.dModel * s.dModel;
  const ffn = 2 * tokens * FFN_MATRICES[s.ffn] * s.dModel * (s.ffRatio * s.dModel);
  const pairs = attentionPairs(n, phase, s.causalSkip);
  const qk = 2 * pairs * s.dModel;
  const pv = 2 * pairs * s.dModel;
  const total = proj + ffn + qk + pv;
  return { proj, qk, pv, ffn, total, attnShare: (qk + pv) / total, attnOverLinear: (qk + pv) / (proj + ffn) };
}

/**
 * The context length at which the two attention products cost as much as the linear layers.
 * Without skipping (and always in decode): N = linear MACs per token / (2 d_model), which is
 * 6 d_model for the book's shape. With the masked half skipped in prefill: N(N+1) d_model =
 * N * linear, so N = linear / d_model - 1.
 */
export function crossover(s: LayerShape, phase: Phase): number {
  const lin = linearMacsPerToken(s);
  if (phase === 'prefill' && s.causalSkip) return lin / s.dModel - 1;
  return lin / (2 * s.dModel);
}

/**
 * Compulsory bytes one decode step reads per layer for a batch of `batch` sequences, each at
 * context n: the linear layers' weights once (they are shared by the batch) and every
 * sequence's own K and V cache (2 n d_model elements, multi-head attention). Ours, from the
 * book's shapes. The cache read overtakes the weights once batch * n exceeds
 * linear MACs / (2 d_model), which is 6 d_model for the book's shape: the FLOP crossover.
 */
export function decodeBytes(n: number, s: LayerShape, batch: number, bytesPer: number) {
  const weights = linearMacsPerToken(s) * bytesPer;
  const cache = batch * 2 * n * s.dModel * bytesPer;
  return { weights, cache, cacheOverWeights: cache / weights, crossBatchN: linearMacsPerToken(s) / (2 * s.dModel) };
}

// ---------------------------------------------------------------- Fig. 20.1 as kernels
export type Primitive = 'host' | 'gather' | 'elementwise' | 'gemm' | 'rowreduce' | 'attention';
export const PRIMITIVES: readonly Primitive[] = ['host', 'gather', 'elementwise', 'gemm', 'rowreduce', 'attention'];
export const PRIMITIVE_LABEL: Record<Primitive, string> = {
  get host() {
    return tr('a lookup on the host', 'một phép tra bảng trên host');
  },
  get gather() {
    return tr('a gather (copy rows by index)', 'một phép gather (sao chép các hàng theo chỉ số)');
  },
  get elementwise() {
    return tr('elementwise', 'theo từng phần tử');
  },
  get gemm() {
    return tr('GEMM in prefill, GEMV in decode', 'GEMM trong prefill, GEMV trong decode');
  },
  get rowreduce() {
    return tr('row reductions plus elementwise work', 'các phép rút gọn theo hàng cộng với công việc theo từng phần tử');
  },
  get attention() {
    return tr('fused attention', 'attention gộp (fused)');
  },
};

export type Milestone = 'M0' | 'M1-M3' | 'M4' | 'M5' | 'M6' | 'none';
export const MILESTONES: readonly Milestone[] = ['M0', 'M1-M3', 'M4', 'M5', 'M6', 'none'];
export const MILESTONE_LABEL: Record<Milestone, string> = {
  get M0() {
    return tr('M0 elementwise ops', 'M0 các phép toán theo từng phần tử');
  },
  get 'M1-M3'() {
    return tr('M1 to M3, the GEMMs', 'M1 đến M3, các GEMM');
  },
  get M4() {
    return tr('M4 row softmax, max, argmax', 'M4 softmax, max, argmax theo hàng');
  },
  get M5() {
    return tr('M5 sampling', 'M5 lấy mẫu');
  },
  get M6() {
    return tr('M6 attention (this chapter)', 'M6 attention (chương này)');
  },
  get none() {
    return tr('no milestone', 'không có mốc dự án');
  },
};

export interface Block {
  id: string;
  /** Fig. 20.1's call-out number where it has one. */
  fig?: string;
  name: string;
  /** Accepted answers; the first is the one the feedback leads with. */
  primitive: readonly Primitive[];
  milestone: readonly Milestone[];
  why: string;
}

/** The blocks of Fig. 20.1 (p. 479), top to bottom, as the kernels that run them. */
export const BLOCKS: readonly Block[] = [
  {
    id: 'tokenizer', fig: '1',
    get name() {
      return tr('Tokenizer', 'Tokenizer');
    },
    primitive: ['host'], milestone: ['none'],
    get why() {
      return tr(
        'Text to token IDs through a table built in training (pp. 479–480). It runs on the CPU before any kernel.',
        'Văn bản thành ID token qua một bảng được xây trong lúc huấn luyện (pp. 479–480). Nó chạy trên CPU trước mọi kernel.',
      );
    },
  },
  {
    id: 'embedding', fig: '2',
    get name() {
      return tr('Embedding', 'Embedding');
    },
    primitive: ['gather'], milestone: ['none'],
    get why() {
      return tr(
        'One row of a [vocabulary × d_model] table per token ID: a copy by index, no arithmetic (p. 480).',
        'Mỗi ID token lấy một hàng của bảng [vocabulary × d_model]: một lần sao chép theo chỉ số, không có phép tính nào (p. 480).',
      );
    },
  },
  {
    id: 'position', fig: '3',
    get name() {
      return tr('Positional encoding', 'Mã hóa vị trí');
    },
    primitive: ['elementwise'], milestone: ['M0'],
    get why() {
      return tr(
        "In the book's original transformer, a position vector added to each embedding: an elementwise add (p. 480). Today's open models rotate Q and K inside attention instead (facts E8).",
        'Trong transformer gốc của sách, một vector vị trí được cộng vào mỗi embedding: một phép cộng theo từng phần tử (p. 480). Các mô hình mở ngày nay thay vào đó xoay Q và K bên trong attention (facts E8).',
      );
    },
  },
  {
    id: 'qkv', fig: '5',
    get name() {
      return tr('Q, K, V projections', 'Các phép chiếu Q, K, V');
    },
    primitive: ['gemm'], milestone: ['M1-M3'],
    get why() {
      return tr(
        'X times W_Q, W_K, W_V. N rows in prefill make it a GEMM; one row per decode step makes it a GEMV (p. 483, p. 491).',
        'X nhân W_Q, W_K, W_V. N hàng trong prefill biến nó thành một GEMM; một hàng mỗi bước decode biến nó thành một GEMV (p. 483, p. 491).',
      );
    },
  },
  {
    id: 'qkt', fig: '5',
    get name() {
      return tr('Q K^T', 'Q K^T');
    },
    primitive: ['attention', 'gemm'], milestone: ['M6'],
    get why() {
      return tr(
        'It is a GEMM (section 20.3 runs it with your lab 15.6 kernel), but its output is N × N per head. Section 20.5 fuses it so that output never reaches memory.',
        'Nó là một GEMM (mục 20.3 chạy nó bằng kernel lab 15.6 của bạn), nhưng đầu ra của nó là N × N cho mỗi head. Mục 20.5 gộp nó để đầu ra đó không bao giờ chạm tới bộ nhớ.',
      );
    },
  },
  {
    id: 'softmax', fig: '5',
    get name() {
      return tr('Mask and softmax', 'Mask và softmax');
    },
    primitive: ['attention', 'rowreduce'], milestone: ['M6'],
    get why() {
      return tr(
        "Two row reductions and an elementwise pass, the shape of M4, with the causal mask as a loop bound: lab 20.1 is M6's first part. Section 20.5 folds it into the fused kernel.",
        'Hai phép rút gọn theo hàng và một lượt theo từng phần tử, đúng dạng của M4, với causal mask là một cận vòng lặp: lab 20.1 là phần đầu của M6. Mục 20.5 gộp nó vào kernel gộp (fused).',
      );
    },
  },
  {
    id: 'pv', fig: '5',
    get name() {
      return tr('P V', 'P V');
    },
    primitive: ['attention', 'gemm'], milestone: ['M6'],
    get why() {
      return tr(
        'A GEMM in the unfused pipeline (your lab 5.3 kernel), whose input P is the N × N matrix the fused kernel never writes.',
        'Một GEMM trong pipeline không gộp (kernel lab 5.3 của bạn), có đầu vào P là ma trận N × N mà kernel gộp không bao giờ ghi ra.',
      );
    },
  },
  {
    id: 'oproj', fig: '5',
    get name() {
      return tr('Output projection', 'Phép chiếu đầu ra');
    },
    primitive: ['gemm'], milestone: ['M1-M3'],
    get why() {
      return tr(
        'The heads are concatenated and multiplied by W_O (p. 480): one more linear layer.',
        'Các head được nối lại và nhân với W_O (p. 480): thêm một lớp tuyến tính nữa.',
      );
    },
  },
  {
    id: 'residual', fig: '7',
    get name() {
      return tr('Residual add', 'Cộng residual');
    },
    primitive: ['elementwise'], milestone: ['M0'],
    get why() {
      return tr(
        "The Add of Add & Norm: the sub-layer's input added to its output, element by element (p. 480).",
        'Phần Add của Add & Norm: đầu vào của lớp con được cộng vào đầu ra của nó, từng phần tử một (p. 480).',
      );
    },
  },
  {
    id: 'norm', fig: '7',
    get name() {
      return tr('LayerNorm', 'LayerNorm');
    },
    primitive: ['rowreduce'], milestone: ['M4'],
    get why() {
      return tr(
        'Per token row: a mean and a variance (two reductions), then (x − μ)/σ · γ + β elementwise. One block per row, as M4 does softmax.',
        'Mỗi hàng token: một trung bình và một phương sai (hai phép rút gọn), rồi (x − μ)/σ · γ + β theo từng phần tử. Một khối cho mỗi hàng, như M4 làm với softmax.',
      );
    },
  },
  {
    id: 'ffn', fig: '6',
    get name() {
      return tr('Feed-forward', 'Feed-forward');
    },
    primitive: ['gemm'], milestone: ['M1-M3'],
    get why() {
      return tr(
        'Two linear layers with a ReLU between (p. 481): two GEMMs, the ReLU elementwise or folded into the first.',
        'Hai lớp tuyến tính với một ReLU ở giữa (p. 481): hai GEMM, ReLU chạy theo từng phần tử hoặc được gộp vào GEMM đầu tiên.',
      );
    },
  },
  {
    id: 'unembed', fig: '8',
    get name() {
      return tr('Un-embedding', 'Un-embedding');
    },
    primitive: ['gemm'], milestone: ['M1-M3'],
    get why() {
      return tr(
        'A linear layer from d_model to the vocabulary, run on the last token only in decode (p. 481).',
        'Một lớp tuyến tính từ d_model sang vocabulary, trong decode chỉ chạy trên token cuối cùng (p. 481).',
      );
    },
  },
  {
    id: 'choose',
    get name() {
      return tr('Choosing the next token', 'Chọn token tiếp theo');
    },
    primitive: ['rowreduce'], milestone: ['M4', 'M5'],
    get why() {
      return tr(
        'Greedy decoding is an argmax over the logits (M4); sampling adds softmax, sorting or scanning and a random draw (M5, chapter X1). The book gives it one sentence (p. 481).',
        'Greedy decoding là một argmax trên logits (M4); lấy mẫu thêm softmax, sắp xếp hoặc scan và một lần rút ngẫu nhiên (M5, chương X1). Sách chỉ dành cho nó một câu (p. 481).',
      );
    },
  },
];

export interface Verdict {
  primitive: boolean;
  milestone: boolean;
}
export function judge(b: Block, primitive: Primitive | null, milestone: Milestone | null): Verdict {
  return {
    primitive: primitive != null && b.primitive.includes(primitive),
    milestone: milestone != null && b.milestone.includes(milestone),
  };
}
