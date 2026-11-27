// One decode step of a Llama-style model (milestone M10) as a list of launches of the course's
// kernels (section X3.7, the llama-step scene, and the X3 labs page's launch table).
//
// The order is llama2.c's forward (notes/today/x3-facts.md D1): embed; per layer RMSNorm, the
// q, k and v projections, RoPE on q and k, append k and v to the cache at pos, attention over
// pos + 1 keys with GQA, wo, residual add, RMSNorm, w1 and w3, SwiGLU, w2, residual add; final
// RMSNorm; the classifier tied to the embedding; argmax (or the sampler). Batch 1, float32.
//
// The launch order is lab X3.2's (labs/x3/lab2_tinyllama/llama.h): RoPE is one launch per buffer
// (launchRope rotates one buffer of heads), and the step ends with launchAdvancePositions.
// Which lab supplies each launch is the project notes of x3-facts.md: M8's launchEmbed (lab
// 22.2), the LinearFn of labs 6.4 / 15.6 / 22.1 (at batch 1 lab 22.1's gather GEMV), M0's
// launchAddResidual (lab 2.3), M7's launchKvAppend and decode attention (labs 20.4, 20.5), M4's
// launchArgmaxRows (lab 10.3), M5's launchSampleRows (lab X1.3). RMSNorm, RoPE and SwiGLU are
// lab X3.1's (launchRmsNorm, launchRope, launchSwiGLU, as llama.h declares them).

import { tr } from '../lib/locale.ts';

export const HEAD = 128; // M7's HEAD_DIM (x3-facts E2, project notes)
export const FLOAT = 4;

export interface Config {
  /** hidden size; with head 128 it is heads × 128, as llama2.c's head_size = dim / n_heads */
  d: number;
  hidden: number;
  layers: number;
  heads: number;
  kvHeads: number;
  vocab: number;
  /** head size; 128 for anything M7's kernels run */
  head: number;
  /** the cache length the tables are built for */
  seqLen: number;
}

export type Kind = 'gather' | 'norm' | 'matmul' | 'rope' | 'cache' | 'attention' | 'elementwise' | 'select';
export type Job = 'embed' | 'rmsnorm' | 'q' | 'k' | 'v' | 'rope-q' | 'rope-k' | 'append' | 'attention' | 'wo' | 'residual' | 'w1' | 'w3' | 'swiglu' | 'w2' | 'final-norm' | 'lm-head' | 'argmax' | 'sample' | 'advance';

export interface Launch {
  /** 0-based position in the step */
  n: number;
  /** layer index, or null outside the layers */
  layer: number | null;
  job: Job;
  kind: Kind;
  /** the course's launch function */
  fn: string;
  /** where it comes from */
  lab: string;
  milestone: string;
  /** whether the course has it before lab X3.1 */
  exists: boolean;
  inputs: string;
  outputs: string;
  /** bytes of model weights read */
  weightBytes: number;
  /** bytes of activations read and written (the hidden state, q, k, v, logits, token ids) */
  actBytes: number;
  /** bytes of the KV cache read (attention) or written (append) */
  cacheBytes: number;
  /** bytes of the host-built cos and sin tables read */
  tableBytes: number;
  /** multiply-adds counted as 2 FLOPs, in the matrix products and attention only */
  flops: number;
  /** the job's counterpart in llama.cpp, for the link into section X3.1's map */
  project: string;
}

const kv = (c: Config) => c.kvHeads * c.head;
const qw = (c: Config) => c.heads * c.head;

export function validate(c: Config): string | null {
  if (c.heads < 1 || c.kvHeads < 1) return tr('heads must be positive', 'số head phải dương');
  if (c.heads % c.kvHeads) return tr('the query heads must be a multiple of the KV heads', 'số head truy vấn phải là bội của số head KV');
  if (c.d !== c.heads * c.head) return tr('d must be heads × head size', 'd phải bằng số head × kích thước head');
  if (c.layers < 1 || c.hidden < 1 || c.vocab < 1) return tr('sizes must be positive', 'các kích thước phải dương');
  return null;
}

/** The launches of one step at position pos (0-based: pos tokens are already in the cache). */
export function stepLaunches(c: Config, pos: number, sampler: 'greedy' | 'sample' = 'greedy'): Launch[] {
  const bad = validate(c);
  if (bad) throw new Error(bad);
  const F = FLOAT;
  const out: Launch[] = [];
  const add = (l: Omit<Launch, 'n'>) => out.push({ n: out.length, ...l });
  const linear = (layer: number, job: Job, inW: number, outW: number, inputs: string, outputs: string, project: string) =>
    add({
      layer, job, kind: 'matmul', fn: tr('LinearFn (launchGemvGather at batch 1)', 'LinearFn (launchGemvGather ở batch 1)'), lab: 'lab 22.1 (or 6.4, 15.6)', milestone: 'M8', exists: true,
      inputs, outputs, weightBytes: inW * outW * F, actBytes: (inW + outW) * F, cacheBytes: 0, tableBytes: 0, flops: 2 * inW * outW, project,
    });
  const rms = (layer: number | null, job: Job, inputs: string, outputs: string) =>
    add({
      layer, job, kind: 'norm', fn: 'launchRmsNorm', lab: 'lab X3.1', milestone: 'M10', exists: false,
      inputs, outputs, weightBytes: c.d * F, actBytes: 2 * c.d * F, cacheBytes: 0, tableBytes: 0, flops: 0, project: 'rms_norm_f32 (norm.cu), kernel_rms_norm (norm.metal)', // i18n-ignore: kernel names and files
    });

  add({
    layer: null, job: 'embed', kind: 'gather', fn: 'launchEmbed', lab: 'lab 22.2', milestone: 'M8', exists: true,
    inputs: tr('token id, E [vocab × d]', 'id token, E [vocab × d]'), outputs: `x [${c.d}]`, weightBytes: c.d * F, actBytes: (1 + c.d) * F, cacheBytes: 0, tableBytes: 0, flops: 0,
    project: 'GGML_OP_GET_ROWS: k_get_rows (getrows.cu), kernel_get_rows (Metal)', // i18n-ignore: op and kernel names
  });
  for (let l = 0; l < c.layers; l++) {
    rms(l, 'rmsnorm', `x [${c.d}], w [${c.d}]`, `xb [${c.d}]`); // i18n-ignore: tensor shapes
    linear(l, 'q', c.d, qw(c), `xb [${c.d}], Wq [${qw(c)} × ${c.d}]`, `q [${qw(c)}]`, // i18n-ignore: tensor shapes
      tr('GGML_OP_MUL_MAT: ggml_cuda_mul_mat, which picks a GEMV kernel for one token (x3 B4)', 'GGML_OP_MUL_MAT: ggml_cuda_mul_mat, hàm chọn một kernel GEMV cho một token (x3 B4)'));
    linear(l, 'k', c.d, kv(c), `xb [${c.d}], Wk [${kv(c)} × ${c.d}]`, `k [${kv(c)}]`, 'GGML_OP_MUL_MAT'); // i18n-ignore: tensor shapes
    linear(l, 'v', c.d, kv(c), `xb [${c.d}], Wv [${kv(c)} × ${c.d}]`, `v [${kv(c)}]`, 'GGML_OP_MUL_MAT'); // i18n-ignore: tensor shapes
    for (const [job, buf, width, heads] of [['rope-q', 'q', qw(c), c.heads], ['rope-k', 'k', kv(c), c.kvHeads]] as const)
      add({
        layer: l, job, kind: 'rope', fn: 'launchRope', lab: 'lab X3.1', milestone: 'M10', exists: false,
        inputs: tr(`${buf} [${width}] as ${heads} heads, cos and sin row ${pos}`, `${buf} [${width}] thành ${heads} head, hàng ${pos} của cos và sin`),
        outputs: tr(`${buf} rotated in place`, `${buf} được xoay tại chỗ`),
        weightBytes: 0, actBytes: 2 * width * F, cacheBytes: 0, tableBytes: c.head * F, flops: 0,
        project: 'GGML_OP_ROPE: rope_norm or rope_neox (rope.cu), kernel_rope_norm (rope.metal)', // i18n-ignore: op and kernel names
      });
    add({
      layer: l, job: 'append', kind: 'cache', fn: 'launchKvAppend', lab: 'lab 20.4', milestone: 'M7', exists: true,
      inputs: `k [${kv(c)}], v [${kv(c)}]`, outputs: tr(`K, V cache row ${pos}`, `hàng ${pos} của cache K, V`),
      weightBytes: 0, actBytes: 2 * kv(c) * F, cacheBytes: 2 * kv(c) * F, tableBytes: 0, flops: 0,
      project: tr('the cache write in llama.cpp’s graph (ch20 D3)', 'phép ghi cache trong đồ thị của llama.cpp (ch20 D3)'),
    });
    add({
      layer: l, job: 'attention', kind: 'attention', fn: c.heads === c.kvHeads ? 'launchDecodeAttention' : 'launchDecodeAttentionGrouped',
      lab: c.heads === c.kvHeads ? 'lab 20.4' : 'lab 20.5', milestone: 'M7', exists: true,
      inputs: tr(`q [${qw(c)}], ${pos + 1} cached rows of K and V [${kv(c)}] each`, `q [${qw(c)}], ${pos + 1} hàng K và V trong cache, mỗi hàng [${kv(c)}]`),
      outputs: `o [${qw(c)}]`,
      weightBytes: 0, actBytes: 2 * qw(c) * F, cacheBytes: 2 * (pos + 1) * kv(c) * F, tableBytes: 0, flops: 4 * (pos + 1) * qw(c),
      project: 'GGML_OP_FLASH_ATTN_EXT: fattn-vec.cuh, Metal flash attention (ch20 D1, D2)', // i18n-ignore: op and file names, citation
    });
    linear(l, 'wo', qw(c), c.d, `o [${qw(c)}], Wo [${c.d} × ${qw(c)}]`, `t [${c.d}]`, 'GGML_OP_MUL_MAT'); // i18n-ignore: tensor shapes
    add({
      layer: l, job: 'residual', kind: 'elementwise', fn: 'launchAddResidual', lab: 'lab 2.3', milestone: 'M0', exists: true,
      inputs: `x [${c.d}], t [${c.d}]`, outputs: 'x += t', weightBytes: 0, actBytes: 3 * c.d * F, cacheBytes: 0, tableBytes: 0, flops: 0,
      project: tr('GGML_OP_ADD: k_bin_bcast (binbcast.cu), often fused (ch03 D1, x3 A1)', 'GGML_OP_ADD: k_bin_bcast (binbcast.cu), thường được gộp (ch03 D1, x3 A1)'),
    });
    rms(l, 'rmsnorm', `x [${c.d}], w [${c.d}]`, `xb [${c.d}]`); // i18n-ignore: tensor shapes
    linear(l, 'w1', c.d, c.hidden, `xb [${c.d}], W1 [${c.hidden} × ${c.d}]`, `h1 [${c.hidden}]`, 'GGML_OP_MUL_MAT'); // i18n-ignore: tensor shapes
    linear(l, 'w3', c.d, c.hidden, `xb [${c.d}], W3 [${c.hidden} × ${c.d}]`, `h3 [${c.hidden}]`, 'GGML_OP_MUL_MAT'); // i18n-ignore: tensor shapes
    add({
      layer: l, job: 'swiglu', kind: 'elementwise', fn: 'launchSwiGLU', lab: 'lab X3.1', milestone: 'M10', exists: false,
      inputs: `h1 [${c.hidden}], h3 [${c.hidden}]`, outputs: `h = silu(h1) · h3 [${c.hidden}]`,
      weightBytes: 0, actBytes: 3 * c.hidden * F, cacheBytes: 0, tableBytes: 0, flops: 0,
      project: 'GGML_GLU_OP_SWIGLU: unary_gated_op_kernel (unary.cu), kernel_swiglu (Metal)', // i18n-ignore: op and kernel names
    });
    linear(l, 'w2', c.hidden, c.d, `h [${c.hidden}], W2 [${c.d} × ${c.hidden}]`, `t [${c.d}]`, 'GGML_OP_MUL_MAT');
    add({
      layer: l, job: 'residual', kind: 'elementwise', fn: 'launchAddResidual', lab: 'lab 2.3', milestone: 'M0', exists: true,
      inputs: `x [${c.d}], t [${c.d}]`, outputs: 'x += t', weightBytes: 0, actBytes: 3 * c.d * F, cacheBytes: 0, tableBytes: 0, flops: 0,
      project: 'GGML_OP_ADD',
    });
  }
  rms(null, 'final-norm', `x [${c.d}], w [${c.d}]`, `xb [${c.d}]`); // i18n-ignore: tensor shapes
  linear(-1, 'lm-head', c.d, c.vocab, tr(`xb [${c.d}], E [${c.vocab} × ${c.d}] (tied)`, `xb [${c.d}], E [${c.vocab} × ${c.d}] (dùng chung với embedding)`),
    `logits [${c.vocab}]`, // i18n-ignore: tensor shape
    tr('GGML_OP_MUL_MAT on the output weight', 'GGML_OP_MUL_MAT trên trọng số đầu ra'));
  out[out.length - 1].layer = null;
  if (sampler === 'greedy')
    add({
      layer: null, job: 'argmax', kind: 'select', fn: 'launchArgmaxRows', lab: 'lab 10.3', milestone: 'M4', exists: true,
      inputs: `logits [${c.vocab}]`, // i18n-ignore: tensor shape
      outputs: tr('next token id, written in place', 'id token tiếp theo, ghi tại chỗ'), weightBytes: 0, actBytes: (c.vocab + 1) * F, cacheBytes: 0, tableBytes: 0, flops: 0,
      project: tr('argmax_f32 (argmax.cu), kernel_argmax_f32 (Metal), or the host greedy sampler', 'argmax_f32 (argmax.cu), kernel_argmax_f32 (Metal), hoặc bộ lấy mẫu greedy trên host'),
    });
  else
    add({
      layer: null, job: 'sample', kind: 'select', fn: 'launchSampleRows', lab: 'lab X1.3', milestone: 'M5', exists: true,
      inputs: tr(`logits [${c.vocab}], counts, seed`, `logits [${c.vocab}], số đếm, seed`), outputs: tr('next token id', 'id token tiếp theo'), weightBytes: 0, actBytes: (c.vocab + 1) * F, cacheBytes: 0, tableBytes: 0, flops: 0,
      project: tr('src/llama-sampler.cpp on the host by default (x1 D1)', 'src/llama-sampler.cpp, mặc định chạy trên host (x1 D1)'),
    });
  add({
    layer: null, job: 'advance', kind: 'select', fn: 'launchAdvancePositions', lab: 'lab X3.2 (yours)', milestone: 'M10', exists: false,
    inputs: tr('tokens, pos, the prompt', 'tokens, pos và prompt'), outputs: tr(`out[${pos}], the next input token, pos and len + 1`, `out[${pos}], token đầu vào tiếp theo, pos và len + 1`),
    weightBytes: 0, actBytes: 0, cacheBytes: 0, tableBytes: 0, flops: 0,
    project: tr('no counterpart read in the facts pass', 'chưa đọc phần tương ứng nào trong lượt kiểm tra dữ kiện'),
  });
  return out;
}

export interface Totals {
  launches: number;
  weightBytes: number;
  lmHeadBytes: number;
  /** LM head weight bytes / all weight bytes read in the step */
  lmHeadShare: number;
  actBytes: number;
  cacheBytes: number;
  tableBytes: number;
  flops: number;
  byKind: Record<Kind, number>;
}
export function totals(ls: Launch[]): Totals {
  const byKind = { gather: 0, norm: 0, matmul: 0, rope: 0, cache: 0, attention: 0, elementwise: 0, select: 0 } as Record<Kind, number>;
  let w = 0, a = 0, kvb = 0, tb = 0, fl = 0, lm = 0;
  for (const l of ls) {
    w += l.weightBytes; a += l.actBytes; kvb += l.cacheBytes; tb += l.tableBytes; fl += l.flops;
    if (l.job === 'lm-head') lm += l.weightBytes;
    byKind[l.kind] += l.weightBytes + l.actBytes + l.cacheBytes + l.tableBytes;
  }
  return { launches: ls.length, weightBytes: w, lmHeadBytes: lm, lmHeadShare: lm / w, actBytes: a, cacheBytes: kvb, tableBytes: tb, flops: fl, byKind };
}

/** Launches per token: 1 embed, 16 per layer, final norm, LM head, argmax or sampler, advance. */
export const launchCount = (layers: number) => 1 + 16 * layers + 4;

/**
 * Floats in a llama2.c legacy .bin file with this header (x3-facts E1), skipped RoPE tables included.
 * Our step reads every one of these weights once per token, plus the embedding row, minus the tables.
 */
export function llama2cFloats(c: Config): number {
  const kvDim = (c.d * c.kvHeads) / c.heads;
  const L = c.layers;
  return c.vocab * c.d + L * c.d + L * c.d * c.d + 2 * L * c.d * kvDim + L * c.d * c.d + L * c.d + 3 * L * c.d * c.hidden + c.d + c.seqLen * c.head;
}
export const llama2cBytes = (c: Config) => FLOAT * llama2cFloats(c) + 28;

/** The two tinyllamas headers the facts pass read (E1), and the module's synthetic model. */
export const STORIES15M: Config = { d: 288, hidden: 768, layers: 6, heads: 6, kvHeads: 6, vocab: 32000, head: 48, seqLen: 256 };
export const STORIES260K: Config = { d: 64, hidden: 172, layers: 5, heads: 8, kvHeads: 4, vocab: 512, head: 8, seqLen: 512 };
/** The stepper's starting point: lab X3.2's GPU model (ckptformat's LAB_GPU, tied by a test), head 128 so M7's kernels run it. */
export const SYNTHETIC: Config = { d: 512, hidden: 1376, layers: 4, heads: 4, kvHeads: 2, vocab: 4099, head: HEAD, seqLen: 128 };

/** Short job names for drawing. */
export const JOB_LABEL: Record<Job, string> = {
  embed: 'embed', rmsnorm: 'RMSNorm', q: 'q = Wq x', k: 'k = Wk x', v: 'v = Wv x',
  get 'rope-q'() { return tr('RoPE on q', 'RoPE trên q'); },
  get 'rope-k'() { return tr('RoPE on k', 'RoPE trên k'); },
  get append() { return tr('KV append', 'nối thêm KV'); },
  attention: 'attention', wo: 'Wo',
  get residual() { return tr('residual add', 'cộng residual'); },
  w1: 'W1', w3: 'W3', swiglu: 'SwiGLU', w2: 'W2',
  get 'final-norm'() { return tr('final RMSNorm', 'RMSNorm cuối'); },
  get 'lm-head'() { return tr('LM head', 'LM head'); },
  argmax: 'argmax', sample: 'sample', advance: 'advance',
};
