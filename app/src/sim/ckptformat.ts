// Checkpoint formats for section X3.8: what a loader must read and check before any kernel
// runs. Everything here is ours (module X3 is not in the book); the formats and the
// published numbers come from notes/today/x3-facts.md:
//   D1  llama2.c's "version 0" .bin: a 28-byte header of seven int32, then float32 weights
//       in memory_map_weights order; a negative vocab_size means an unshared classifier.
//   D2  karpathy/tinyllamas: the headers and Hub sizes of stories260K and stories15M.
//   D3  safetensors: 8-byte little-endian N, an N-byte JSON header starting with '{',
//       data_offsets relative to the byte buffer, row-major little-endian.
//   D4  SmolLM2-135M: config.json, the safetensors header (N = 30,528, 272 BF16 tensors).
//   E1  the file-size formula; E2  head sizes against M7's HEAD_DIM 128.
//   appA A2  bfloat16 is the top 16 bits of a float32 (1 sign, 8 exponent, 7 mantissa).

import { tr } from '../lib/locale.ts';

// ------------------------------------------------------------------ llama2.c header (D1)

/** The seven int32 of llama2.c's Config, in file order. vocab_size is as stored (its sign is a flag). */
export interface Llama2cHeader {
  dim: number;
  hidden_dim: number;
  n_layers: number;
  n_heads: number;
  n_kv_heads: number;
  vocab_size: number;
  seq_len: number;
}

export const HEADER_FIELDS = ['dim', 'hidden_dim', 'n_layers', 'n_heads', 'n_kv_heads', 'vocab_size', 'seq_len'] as const;
export type HeaderField = (typeof HEADER_FIELDS)[number];
export const HEADER_BYTES = 28;

/** The 28 header bytes, little-endian int32 each. */
export function encodeHeader(h: Llama2cHeader): Uint8Array {
  const b = new Uint8Array(HEADER_BYTES);
  const v = new DataView(b.buffer);
  HEADER_FIELDS.forEach((f, i) => {
    const x = h[f];
    if (!Number.isInteger(x) || x < -(2 ** 31) || x >= 2 ** 31) throw new Error(`${f} = ${x} is not an int32`);
    v.setInt32(4 * i, x, true);
  });
  return b;
}

/** Reads the header from the first 28 bytes of a file (or of a range request). */
export function parseHeader(bytes: Uint8Array): Llama2cHeader {
  if (bytes.length < HEADER_BYTES) throw new Error(`a llama2.c header is ${HEADER_BYTES} bytes; got ${bytes.length}`);
  const v = new DataView(bytes.buffer, bytes.byteOffset, HEADER_BYTES);
  const out = {} as Llama2cHeader;
  HEADER_FIELDS.forEach((f, i) => (out[f] = v.getInt32(4 * i, true)));
  return out;
}

/** Bytes as two-digit upper-case hex, for showing a dump. */
export const hex = (bytes: ArrayLike<number>) => Array.from(bytes, (x) => x.toString(16).toUpperCase().padStart(2, '0'));

/** The four bytes of field i (0-based) of a header, as hex. */
export const fieldBytes = (h: Llama2cHeader, i: number) => hex(encodeHeader(h).slice(4 * i, 4 * i + 4));

// ------------------------------------------------------------------ layout and size (D1, E1)

export interface Block {
  name: string;
  /** What it holds, for the table. */
  what: string;
  floats: number;
  /** In the file but never read by run.c. */
  skipped?: boolean;
}

export interface Shape {
  headSize: number;
  kvDim: number;
  qPerKv: number;
  /** Problems that make the header unusable (non-integer head size, etc.). */
  problems: string[];
}

/** Head size and grouping implied by a header; problems listed rather than thrown. */
export function shapeOf(h: Pick<Llama2cHeader, 'dim' | 'n_heads' | 'n_kv_heads'>): Shape {
  const problems: string[] = [];
  if (!(h.dim > 0) || !(h.n_heads > 0) || !(h.n_kv_heads > 0)) problems.push(tr('dim and both head counts must be positive', 'dim và cả hai số head phải dương'));
  else {
    if (h.dim % h.n_heads) problems.push(tr(`dim ${h.dim} is not a multiple of n_heads ${h.n_heads}`, `dim ${h.dim} không phải bội của n_heads ${h.n_heads}`));
    if (h.n_heads % h.n_kv_heads) problems.push(tr(`n_heads ${h.n_heads} is not a multiple of n_kv_heads ${h.n_kv_heads}`, `n_heads ${h.n_heads} không phải bội của n_kv_heads ${h.n_kv_heads}`));
  }
  const headSize = h.dim / h.n_heads;
  return { headSize, kvDim: (h.dim * h.n_kv_heads) / h.n_heads, qPerKv: h.n_heads / h.n_kv_heads, problems };
}

/** true when the classifier is the embedding (vocab_size > 0), as run.c decides. */
export const sharedClassifier = (h: Llama2cHeader) => h.vocab_size > 0;

/** The float blocks after the header, in file order (run.c memory_map_weights). */
export function layout(h: Llama2cHeader): Block[] {
  const V = Math.abs(h.vocab_size);
  const d = h.dim;
  const L = h.n_layers;
  const { headSize, kvDim } = shapeOf(h);
  const blocks: Block[] = [
    { name: 'token_embedding_table', what: 'vocab × dim', floats: V * d }, // i18n-ignore: tensor name and shape
    { name: 'rms_att_weight', what: tr('layers × dim', 'số lớp × dim'), floats: L * d }, // i18n-ignore: tensor name and shape
    { name: 'wq', what: tr('layers × dim × dim', 'số lớp × dim × dim'), floats: L * d * d }, // i18n-ignore: tensor name and shape
    { name: 'wk', what: tr('layers × kv_dim × dim', 'số lớp × kv_dim × dim'), floats: L * kvDim * d }, // i18n-ignore: tensor name and shape
    { name: 'wv', what: tr('layers × kv_dim × dim', 'số lớp × kv_dim × dim'), floats: L * kvDim * d }, // i18n-ignore: tensor name and shape
    { name: 'wo', what: tr('layers × dim × dim', 'số lớp × dim × dim'), floats: L * d * d }, // i18n-ignore: tensor name and shape
    { name: 'rms_ffn_weight', what: tr('layers × dim', 'số lớp × dim'), floats: L * d }, // i18n-ignore: tensor name and shape
    { name: 'w1', what: tr('layers × dim × hidden', 'số lớp × dim × hidden'), floats: L * d * h.hidden_dim }, // i18n-ignore: tensor name and shape
    { name: 'w2', what: tr('layers × dim × hidden', 'số lớp × dim × hidden'), floats: L * d * h.hidden_dim }, // i18n-ignore: tensor name and shape
    { name: 'w3', what: tr('layers × dim × hidden', 'số lớp × dim × hidden'), floats: L * d * h.hidden_dim }, // i18n-ignore: tensor name and shape
    { name: 'rms_final_weight', what: 'dim', floats: d }, // i18n-ignore: tensor name and shape
    { name: 'freq_cis_real', what: 'seq_len × head_size / 2', floats: (h.seq_len * headSize) / 2, skipped: true }, // i18n-ignore: tensor name and shape
    { name: 'freq_cis_imag', what: 'seq_len × head_size / 2', floats: (h.seq_len * headSize) / 2, skipped: true }, // i18n-ignore: tensor name and shape
  ];
  if (!sharedClassifier(h)) blocks.push({ name: 'wcls', what: 'vocab × dim', floats: V * d }); // i18n-ignore: tensor name and shape
  return blocks;
}

export const floatCount = (h: Llama2cHeader) => layout(h).reduce((s, b) => s + b.floats, 0);

/** E1: the file size a header predicts, in bytes. */
export const fileBytes = (h: Llama2cHeader) => HEADER_BYTES + 4 * floatCount(h);

export interface SizeCheck {
  expected: number;
  actual: number;
  ok: boolean;
  /** What the mismatch suggests, or null when the sizes agree. */
  hint: string | null;
}

/** Check a download by its size before loading it. */
export function checkSize(h: Llama2cHeader, actual: number): SizeCheck {
  const expected = fileBytes(h);
  if (actual === expected) return { expected, actual, ok: true, hint: null };
  const flipped = fileBytes({ ...h, vocab_size: -h.vocab_size });
  let hint: string;
  if (actual === flipped)
    hint = sharedClassifier(h)
      ? tr(
          'The size fits an unshared classifier: the header should carry a negative vocab_size.',
          'Kích thước khớp với một bộ phân loại riêng: header phải mang vocab_size âm.',
        )
      : tr(
          'The size fits a shared classifier: the header should carry a positive vocab_size.',
          'Kích thước khớp với một bộ phân loại dùng chung: header phải mang vocab_size dương.',
        );
  else if (actual < expected)
    hint = tr(
      `${(expected - actual).toLocaleString('en-US')} bytes short: an interrupted download, or not a version 0 file.`,
      `thiếu ${(expected - actual).toLocaleString('en-US')} byte: tải xuống bị ngắt, hoặc không phải file phiên bản 0.`,
    );
  else
    hint = tr(
      `${(actual - expected).toLocaleString('en-US')} bytes too many: not a version 0 file, or a different header.`,
      `thừa ${(actual - expected).toLocaleString('en-US')} byte: không phải file phiên bản 0, hoặc một header khác.`,
    );
  return { expected, actual, ok: false, hint };
}

// ------------------------------------------------------------------ bfloat16 (appA A2, D4)

/** The float32 bit pattern for a stored bf16 value: the 16 bits become the top half. Exact. */
export const bf16ToF32Bits = (b16: number) => ((b16 & 0xffff) << 16) >>> 0;

/** A float32 bit pattern as a JavaScript number (the float it encodes). */
export function bitsToF32(bits: number): number {
  const v = new DataView(new ArrayBuffer(4));
  v.setUint32(0, bits >>> 0, true);
  return v.getFloat32(0, true);
}

export const bf16ToF32 = (b16: number) => bitsToF32(bf16ToF32Bits(b16));

/** The bit pattern of a float32 (after rounding the number to float32). */
export function f32Bits(x: number): number {
  const v = new DataView(new ArrayBuffer(4));
  v.setFloat32(0, x, true);
  return v.getUint32(0, true);
}

export const hex32 = (bits: number) => '0x' + (bits >>> 0).toString(16).toUpperCase().padStart(8, '0');
export const hex16 = (bits: number) => '0x' + (bits & 0xffff).toString(16).toUpperCase().padStart(4, '0');

// ------------------------------------------------------------------ safetensors (D3)

export type Dtype = 'F32' | 'F16' | 'BF16';
export const DTYPE_BYTES: Record<Dtype, number> = { F32: 4, F16: 2, BF16: 2 };

export interface TensorSpec {
  name: string;
  dtype: Dtype;
  shape: number[];
}

export interface TensorEntry extends TensorSpec {
  /** data_offsets, relative to the start of the byte buffer (not of the file). */
  begin: number;
  end: number;
}

export interface Safetensors {
  /** N, the header length. */
  headerLen: number;
  /** Where the byte buffer starts in the file: 8 + N. */
  dataStart: number;
  tensors: TensorEntry[];
  metadata: Record<string, string> | null;
}

export const numel = (shape: number[]) => shape.reduce((a, b) => a * b, 1);
export const tensorBytes = (t: TensorSpec) => numel(t.shape) * DTYPE_BYTES[t.dtype];

/**
 * Parse a safetensors header. `bytes` may be the whole file or just its first 8 + N bytes
 * (a range request); with `whole` set, every tensor must also lie inside the file.
 */
export function parseSafetensors(bytes: Uint8Array, { whole = false } = {}): Safetensors {
  if (bytes.length < 8) throw new Error('shorter than the 8-byte length');
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const n64 = v.getBigUint64(0, true);
  if (n64 > BigInt(bytes.length - 8)) throw new Error(`header length ${n64} runs past the ${bytes.length} bytes available`);
  const headerLen = Number(n64);
  const head = bytes.subarray(8, 8 + headerLen);
  if (head[0] !== 0x7b) throw new Error("the header does not start with '{'");
  let json: Record<string, unknown>;
  try {
    json = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(head));
  } catch (e) {
    throw new Error('the header is not valid UTF-8 JSON: ' + (e as Error).message);
  }
  let metadata: Record<string, string> | null = null;
  const tensors: TensorEntry[] = [];
  for (const [name, raw] of Object.entries(json)) {
    if (name === '__metadata__') {
      metadata = raw as Record<string, string>;
      continue;
    }
    const t = raw as { dtype: string; shape: number[]; data_offsets: [number, number] };
    if (!(t.dtype in DTYPE_BYTES)) throw new Error(`${name}: dtype ${t.dtype} is not handled here`);
    const [begin, end] = t.data_offsets;
    const spec: TensorSpec = { name, dtype: t.dtype as Dtype, shape: t.shape };
    if (end - begin !== tensorBytes(spec))
      throw new Error(`${name}: data_offsets span ${end - begin} bytes, shape and dtype need ${tensorBytes(spec)}`);
    if (whole && 8 + headerLen + end > bytes.length) throw new Error(`${name}: ends past the end of the file`);
    tensors.push({ ...spec, begin, end });
  }
  return { headerLen, dataStart: 8 + headerLen, tensors, metadata };
}

/** Build a safetensors file (tensors packed in the given order; data as given or zeros). */
export function encodeSafetensors(
  specs: TensorSpec[],
  { metadata, data }: { metadata?: Record<string, string>; data?: Record<string, Uint8Array> } = {},
): Uint8Array {
  const header: Record<string, unknown> = {};
  if (metadata) header.__metadata__ = metadata;
  let off = 0;
  for (const s of specs) {
    const n = tensorBytes(s);
    header[s.name] = { dtype: s.dtype, shape: s.shape, data_offsets: [off, off + n] };
    off += n;
  }
  const head = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(8 + head.length + off);
  new DataView(out.buffer).setBigUint64(0, BigInt(head.length), true);
  out.set(head, 8);
  let at = 8 + head.length;
  for (const s of specs) {
    if (data?.[s.name]) out.set(data[s.name], at);
    at += tensorBytes(s);
  }
  return out;
}

/** The file size for a header of N bytes over these tensors, with no padding. */
export const safetensorsBytes = (headerLen: number, specs: TensorSpec[]) =>
  8 + headerLen + specs.reduce((s, t) => s + tensorBytes(t), 0);

/** The first 8 bytes of a safetensors file whose header is N bytes long. */
export function lengthBytes(headerLen: number): string[] {
  const b = new Uint8Array(8);
  new DataView(b.buffer).setBigUint64(0, BigInt(headerLen), true);
  return hex(b);
}

// ------------------------------------------------------------------ the checkpoints (D2, D4)

export interface Arch {
  dim: number;
  hidden: number;
  layers: number;
  heads: number;
  kvHeads: number;
  vocab: number;
  ropeTheta: number;
  eps: number;
  /** Classifier: the embedding (shared / tied) or its own matrix. */
  classifier: 'shared' | 'unshared' | 'tied';
  dtype: 'F32' | 'BF16';
  /** RoPE pairing: consecutive (0,1),(2,3)... or half-split (i, i + d/2). */
  pairing: 'consecutive' | 'half-split';
  /** Where the constants come from. */
  constantsFrom: string;
}

export interface Checkpoint {
  id: string;
  label: string;
  format: 'llama2c' | 'safetensors';
  /** llama2.c header (format llama2c). */
  header?: Llama2cHeader;
  /** safetensors header length and tensors (format safetensors). */
  st?: { headerLen: number; tensors: TensorSpec[] };
  arch: Arch;
  /** Size of the file on the Hub, in bytes (null for a model the lab generates). */
  hubBytes: number | null;
  file: string;
  licence: string;
  facts: string;
}

/** llama2.c's run.c hard-codes these (D1). */
export const LLAMA2C_EPS = 1e-5;
export const LLAMA2C_THETA = 10000;

export function archFromHeader(h: Llama2cHeader): Arch {
  return {
    dim: h.dim,
    hidden: h.hidden_dim,
    layers: h.n_layers,
    heads: h.n_heads,
    kvHeads: h.n_kv_heads,
    vocab: Math.abs(h.vocab_size),
    ropeTheta: LLAMA2C_THETA,
    eps: LLAMA2C_EPS,
    classifier: sharedClassifier(h) ? 'shared' : 'unshared',
    dtype: 'F32',
    pairing: 'consecutive',
    constantsFrom: 'hard-coded in run.c', // i18n-ignore: identifier compared with ===, not shown
  };
}

export const STORIES_15M: Llama2cHeader = { dim: 288, hidden_dim: 768, n_layers: 6, n_heads: 6, n_kv_heads: 6, vocab_size: 32000, seq_len: 256 };
export const STORIES_260K: Llama2cHeader = { dim: 64, hidden_dim: 172, n_layers: 5, n_heads: 8, n_kv_heads: 4, vocab_size: 512, seq_len: 512 };

/** SmolLM2-135M's config.json values that matter here (D4). */
export const SMOL_CONFIG = {
  hidden_size: 576,
  intermediate_size: 1536,
  num_hidden_layers: 30,
  num_attention_heads: 9,
  num_key_value_heads: 3,
  vocab_size: 49152,
  rope_theta: 100000,
  rms_norm_eps: 1e-5,
  tie_word_embeddings: true,
  torch_dtype: 'bfloat16',
  rope_interleaved: false,
} as const;
/** N for SmolLM2-135M's model.safetensors, read by range request in the facts pass (D4). */
export const SMOL_HEADER_LEN = 30528;

/** SmolLM2-135M's 272 tensors, with the names and shapes of D4 (every one BF16). */
export function smolTensors(): TensorSpec[] {
  const c = SMOL_CONFIG;
  const d = c.hidden_size;
  const kv = (d / c.num_attention_heads) * c.num_key_value_heads;
  const ff = c.intermediate_size;
  const t = (name: string, shape: number[]): TensorSpec => ({ name, dtype: 'BF16', shape });
  const out = [t('model.embed_tokens.weight', [c.vocab_size, d])];
  for (let i = 0; i < c.num_hidden_layers; i++) {
    const p = `model.layers.${i}.`; // i18n-ignore: tensor name
    out.push(
      t(p + 'input_layernorm.weight', [d]),
      t(p + 'self_attn.q_proj.weight', [d, d]),
      t(p + 'self_attn.k_proj.weight', [kv, d]),
      t(p + 'self_attn.v_proj.weight', [kv, d]),
      t(p + 'self_attn.o_proj.weight', [d, d]),
      t(p + 'post_attention_layernorm.weight', [d]),
      t(p + 'mlp.gate_proj.weight', [ff, d]),
      t(p + 'mlp.up_proj.weight', [ff, d]),
      t(p + 'mlp.down_proj.weight', [d, ff]),
    );
  }
  out.push(t('model.norm.weight', [d]));
  return out;
}

/**
 * Lab X3.2's synthetic models (M10), written by provided/make_model.py in the version 0 format.
 * ckptformat.test.ts reads labs/x3/lab2_tinyllama/provided/pieces.cmake and fails if these drift.
 */
export const LAB_EMU: Llama2cHeader = { dim: 256, hidden_dim: 352, n_layers: 2, n_heads: 2, n_kv_heads: 1, vocab_size: 300, seq_len: 16 };
export const LAB_GPU: Llama2cHeader = { dim: 512, hidden_dim: 1376, n_layers: 4, n_heads: 4, n_kv_heads: 2, vocab_size: 4099, seq_len: 128 };

const labModel = (id: string, label: () => string, h: Llama2cHeader): Checkpoint => ({
  id,
  get label() {
    return label();
  },
  format: 'llama2c',
  header: h,
  arch: { ...archFromHeader(h), constantsFrom: 'hard-coded in run.c' }, // i18n-ignore: identifier compared with ===, not shown
  hubBytes: null,
  get file() {
    return tr('generated by labs/x3/lab2_tinyllama/provided/make_model.py', 'sinh bởi labs/x3/lab2_tinyllama/provided/make_model.py');
  },
  get licence() {
    return tr('ours', 'của khóa học này');
  },
  facts: 'D1, E1',
});

export const CHECKPOINTS: Checkpoint[] = [
  labModel('lab-emu', () => tr('M10, emulator model', 'M10, mô hình cho trình giả lập'), LAB_EMU),
  labModel('lab-gpu', () => tr('M10, GPU model', 'M10, mô hình cho GPU'), LAB_GPU),
  {
    id: 'stories260K',
    label: 'stories260K', // i18n-ignore: model name
    format: 'llama2c',
    header: STORIES_260K,
    arch: archFromHeader(STORIES_260K),
    hubBytes: 1056540,
    file: 'karpathy/tinyllamas, stories260K/stories260K.bin',
    licence: 'MIT',
    facts: 'D2, E1, E2',
  },
  {
    id: 'stories15M',
    label: 'stories15M', // i18n-ignore: model name
    format: 'llama2c',
    header: STORIES_15M,
    arch: archFromHeader(STORIES_15M),
    hubBytes: 60816028,
    file: 'karpathy/tinyllamas, stories15M.bin',
    licence: 'MIT',
    facts: 'D2, E1, E2',
  },
  {
    id: 'smollm2',
    label: 'SmolLM2-135M', // i18n-ignore: model name
    format: 'safetensors',
    st: { headerLen: SMOL_HEADER_LEN, tensors: smolTensors() },
    arch: {
      dim: SMOL_CONFIG.hidden_size,
      hidden: SMOL_CONFIG.intermediate_size,
      layers: SMOL_CONFIG.num_hidden_layers,
      heads: SMOL_CONFIG.num_attention_heads,
      kvHeads: SMOL_CONFIG.num_key_value_heads,
      vocab: SMOL_CONFIG.vocab_size,
      ropeTheta: SMOL_CONFIG.rope_theta,
      eps: SMOL_CONFIG.rms_norm_eps,
      classifier: 'tied',
      dtype: 'BF16',
      pairing: 'half-split',
      constantsFrom: 'config.json',
    },
    hubBytes: 269060552,
    file: 'HuggingFaceTB/SmolLM2-135M, model.safetensors',
    licence: 'Apache-2.0',
    facts: 'D3, D4, E2',
  },
];

// ------------------------------------------------------------------ fit and checklist (E2)

/** M7's decode kernels fix the head size at compile time (labs 20.4 and 20.5). */
export const M7_HEAD_DIM = 128;
/** Lab 20.4 spreads a head over a logical warp of 32 lanes: HEAD_DIM / 32 values per lane. */
export const LAB204_LANES = 32;

export interface Fit {
  headSize: number;
  qPerKv: number;
  /** HEAD_DIM 128 as M7 is written. */
  m7: boolean;
  /** Values per lane in lab 20.4's layout, or null when the head size is not a multiple of 32. */
  perLane: number | null;
}

export function fitOf(a: Pick<Arch, 'dim' | 'heads' | 'kvHeads'>): Fit {
  const headSize = a.dim / a.heads;
  return {
    headSize,
    qPerKv: a.heads / a.kvHeads,
    m7: headSize === M7_HEAD_DIM,
    perLane: Number.isInteger(headSize) && headSize % LAB204_LANES === 0 ? headSize / LAB204_LANES : null,
  };
}

export interface CheckItem {
  id: 'dtype' | 'rope' | 'gqa' | 'constants' | 'classifier' | 'head';
  label: string;
  /** Something must be done before the course's kernels can run it. */
  todo: boolean;
  detail: string;
}

/** The conversions and checks between a checkpoint and the course's kernels. */
export function checklist(a: Arch): CheckItem[] {
  const f = fitOf(a);
  const items: CheckItem[] = [];
  items.push(
    a.dtype === 'F32'
      ? { id: 'dtype', label: tr('Weights are float32', 'Trọng số là float32'), todo: false, detail: tr('Read them as they are.', 'Đọc nguyên như vậy.') }
      : {
          id: 'dtype',
          label: tr('bf16 to float32', 'bf16 sang float32'),
          todo: true,
          detail: tr(
            `Put each stored 16 bits in the top half of a 32-bit word: ${hex16(0x3f80)} becomes ${hex32(bf16ToF32Bits(0x3f80))} = ${bf16ToF32(0x3f80)}. Exact, no rounding.`,
            `Đặt mỗi 16 bit đã lưu vào nửa trên của một word 32 bit: ${hex16(0x3f80)} thành ${hex32(bf16ToF32Bits(0x3f80))} = ${bf16ToF32(0x3f80)}. Chính xác, không làm tròn.`,
          ),
        },
  );
  items.push(
    a.pairing === 'consecutive'
      ? {
          id: 'rope',
          label: tr('RoPE on consecutive pairs', 'RoPE trên các cặp liên tiếp'),
          todo: false,
          detail: tr('run.c rotates (0, 1), (2, 3), ...: your RoPE kernel must pair the same way.', 'run.c xoay (0, 1), (2, 3), ...: kernel RoPE của bạn phải ghép cặp theo cùng cách đó.'),
        }
      : {
          id: 'rope',
          label: tr('RoPE on half-split pairs', 'RoPE trên các cặp chia đôi'),
          todo: true,
          detail: tr(
            'Hugging Face rotates element i with i + head_size / 2. Use that pairing, or permute the rows of Q and K as llama.cpp’s converter does, with the KV head count for K.',
            'Hugging Face xoay phần tử i cùng với i + head_size / 2. Hãy dùng cách ghép cặp đó, hoặc hoán vị các hàng của Q và K như bộ chuyển đổi của llama.cpp làm, với số head KV cho K.',
          ),
        },
  );
  items.push(
    f.qPerKv === 1
      ? {
          id: 'gqa',
          label: tr('One KV head per query head', 'Mỗi head query một head KV'),
          todo: false,
          detail: tr(`${a.heads} heads, ${a.kvHeads} KV heads.`, `${a.heads} head, ${a.kvHeads} head KV.`),
        }
      : {
          id: 'gqa',
          label: tr(`Grouped-query attention, ${a.heads} over ${a.kvHeads}`, `Grouped-query attention, ${a.heads} trên ${a.kvHeads}`),
          todo: true,
          detail: tr(
            `Query head h reads KV head h / ${f.qPerKv}. K and V projections have ${a.kvHeads} × ${f.headSize} = ${a.kvHeads * f.headSize} rows, not ${a.dim}.`,
            `Head query h đọc head KV h / ${f.qPerKv}. Các phép chiếu K và V có ${a.kvHeads} × ${f.headSize} = ${a.kvHeads * f.headSize} hàng, không phải ${a.dim}.`,
          ),
        },
  );
  items.push({
    id: 'constants',
    label: tr(
      `RoPE base ${a.ropeTheta.toLocaleString("en-US")}, epsilon ${a.eps.toExponential(0)}`,
      `Cơ số RoPE ${a.ropeTheta.toLocaleString("en-US")}, epsilon ${a.eps.toExponential(0)}`,
    ),
    todo: a.constantsFrom !== 'hard-coded in run.c', // i18n-ignore: identifier compared with ===
    detail:
      a.constantsFrom === 'hard-coded in run.c' // i18n-ignore: identifier compared with ===
        ? tr('Not in the file: run.c hard-codes them.', 'Không có trong file: run.c viết cứng chúng.')
        : tr(
            'From config.json, not llama2.c’s 10,000. transformers moves an old top-level rope_theta into rope_parameters.',
            'Lấy từ config.json, không phải 10,000 của llama2.c. transformers chuyển rope_theta cũ ở cấp cao nhất vào rope_parameters.',
          ),
  });
  items.push({
    id: 'classifier',
    label: a.classifier === 'unshared' ? tr('Separate classifier', 'Bộ phân loại riêng') : tr('Classifier is the embedding', 'Bộ phân loại chính là embedding'),
    todo: a.classifier === 'tied',
    detail:
      a.classifier === 'shared'
        ? tr('vocab_size is positive: the LM head reuses token_embedding_table.', 'vocab_size dương: LM head dùng lại token_embedding_table.')
        : a.classifier === 'unshared'
          ? tr('vocab_size is negative: wcls (vocab × dim) follows the two skipped RoPE tables.', 'vocab_size âm: wcls (vocab × dim) nằm sau hai bảng RoPE bị bỏ qua.')
          : tr(
              'tie_word_embeddings is true and there is no lm_head.weight: use model.embed_tokens.weight.',
              'tie_word_embeddings là true và không có lm_head.weight: hãy dùng model.embed_tokens.weight.',
            ),
  });
  items.push({
    id: 'head',
    label: tr(
      `Head size ${Number.isInteger(f.headSize) ? f.headSize : f.headSize.toFixed(2)}`,
      `Kích thước head ${Number.isInteger(f.headSize) ? f.headSize : f.headSize.toFixed(2)}`,
    ),
    todo: !f.m7,
    detail: f.m7
      ? tr('M7’s HEAD_DIM 128 fits as written.', 'HEAD_DIM 128 của M7 khớp như đã viết.')
      : f.perLane != null
        ? tr(
            `Not M7’s 128. Lab 20.4’s layout works with HEAD_DIM ${f.headSize}: ${f.perLane} value${f.perLane > 1 ? 's' : ''} per lane.`,
            `Không phải 128 của M7. Bố cục của lab 20.4 chạy được với HEAD_DIM ${f.headSize}: ${f.perLane} giá trị mỗi lane.`,
          )
        : tr(
            `Not M7’s 128, and not a multiple of 32: lab 20.4’s lane layout cannot hold it without a change to the kernel.`,
            `Không phải 128 của M7, và không phải bội của 32: bố cục lane của lab 20.4 không chứa được nó nếu không sửa kernel.`,
          ),
  });
  return items;
}
