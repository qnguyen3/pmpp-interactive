// The traffic of a quantised decode GEMV for module X2 (ours: not in the book).
//
// What the book gives (notes/ch05.md §5.1, pp. 93–97; notes/ch20.md §20.6, p. 505;
// notes/appA.md §A.5, p. 593): the roofline and FLOP/B; the 1024-to-4096 layer in 16-bit
// at about 1 FLOP/B at batch 1; and the argument that a smaller format moves fewer bytes per
// operation and so raises intensity. The book applies that argument to floating point only.
// Applying it to integer weight formats with block scales is ours (notes/today/x2-facts.md A1).
//
// What this file models (x2-facts E1, our arithmetic on the layouts in D1, D2, D5 and appA D2):
//   - bytes per weight of each format, its block scales included;
//   - one GEMV Y[B x out] = X[B x in] · Wᵀ: 2·B·in·out FLOPs; every weight read once, every
//     activation read once and every output written once (bytes = w·in·out + a·B·(in + out));
//   - a ceiling time against a Device, time = max(bytes / bandwidth, FLOPs / peak), with the
//     peak taken as threshold × bandwidth exactly as batching.ts does;
//   - the four pipelines of section X2.1: float32 GEMV, dequantise to float32 then a float32
//     GEMV, dequantise to FP16 then an FP16 GEMV, and the fused kernel;
//   - model bandwidth utilisation (MBU), PyTorch's definition (A1): weight bytes times
//     inferences per second, over peak bandwidth.
// What it leaves out: caches, launch overhead, the integer and dequantise arithmetic (which
// matters once a layer nears the ridge), partial use of bandwidth, the activation
// quantisation pass of llama.cpp's CUDA path (D3).
//
// Imports batching.ts (read-only) for the Device type, the book's example device and the
// readers of lab 5.1's measured roofline.
import { BOOK_DEVICE, effectivePeak, measuredDevices, thresholdOf, type Device, type ReportsLike, type RowLike } from './batching';
import { tr } from '../lib/locale.ts';

export { BOOK_DEVICE, measuredDevices, thresholdOf, type Device };

// ---------------------------------------------------------------- formats
export type FormatId =
  | 'f32'
  | 'f16'
  | 'q8_0'
  | 'q4_0'
  | 'q4_1'
  | 'q4_k'
  | 'mxfp4'
  | 'mlx4_g32'
  | 'mlx4_g64'
  | 'mlx4_g128'
  | 'mlx4_g32_f32'
  | 'mlx4_g64_f32'
  | 'mlx4_g128_f32';

export interface WeightFormat {
  id: FormatId;
  /** Short name for menus. */
  label: string;
  /** Whose format it is. */
  owner: string;
  /** Weights per block (1 for an unblocked float format). */
  blockWeights: number;
  /** Bytes per block, scales and offsets included. */
  blockBytes: number;
  /** Of blockBytes, the bytes that are not codes: scales, mins, biases. */
  overheadBytes: number;
  /** Where the layout comes from. */
  facts: string;
}

const MLX = 'MLX affine 4-bit'; // i18n-ignore: format name
export const FORMATS: Record<FormatId, WeightFormat> = {
  f32: { id: 'f32', label: 'FP32', owner: 'IEEE float32', blockWeights: 1, blockBytes: 4, overheadBytes: 0, facts: 'x2-facts E1' }, // i18n-ignore: standard name
  f16: {
    id: 'f16', label: 'FP16 / BF16', // i18n-ignore: format names
    get owner() {
      return tr('16-bit float', 'số thực 16-bit');
    },
    blockWeights: 1, blockBytes: 2, overheadBytes: 0, facts: 'x2-facts E1; appA-facts A2',
  },
  q8_0: { id: 'q8_0', label: 'Q8_0', owner: 'llama.cpp', blockWeights: 32, blockBytes: 34, overheadBytes: 2, facts: 'x2-facts D1, E1' },
  q4_0: { id: 'q4_0', label: 'Q4_0', owner: 'llama.cpp', blockWeights: 32, blockBytes: 18, overheadBytes: 2, facts: 'x2-facts D1, E1' },
  q4_1: { id: 'q4_1', label: 'Q4_1', owner: 'llama.cpp', blockWeights: 32, blockBytes: 20, overheadBytes: 4, facts: 'x2-facts D1, E1' },
  q4_k: { id: 'q4_k', label: 'Q4_K', owner: 'llama.cpp', blockWeights: 256, blockBytes: 144, overheadBytes: 16, facts: 'x2-facts D2, E1' },
  mxfp4: { id: 'mxfp4', label: 'MXFP4', owner: 'OCP MX; llama.cpp block_mxfp4', blockWeights: 32, blockBytes: 17, overheadBytes: 1, facts: 'x2-facts E1; appA-facts D3' }, // i18n-ignore: standard and type names
  mlx4_g32: {
    id: 'mlx4_g32',
    get label() {
      return tr(`${MLX}, g32, FP16 scale and bias`, `${MLX}, g32, scale và bias FP16`);
    },
    owner: 'MLX',
    blockWeights: 32, blockBytes: 20, overheadBytes: 4,
    facts: 'x2-facts D5, E1',
  },
  mlx4_g64: {
    id: 'mlx4_g64',
    get label() {
      return tr(`${MLX}, g64, FP16 scale and bias`, `${MLX}, g64, scale và bias FP16`);
    },
    get owner() {
      return tr('MLX (its default)', 'MLX (mặc định của nó)');
    },
    blockWeights: 64, blockBytes: 36, overheadBytes: 4,
    facts: 'x2-facts D5, E1',
  },
  mlx4_g128: {
    id: 'mlx4_g128',
    get label() {
      return tr(`${MLX}, g128, FP16 scale and bias`, `${MLX}, g128, scale và bias FP16`);
    },
    owner: 'MLX',
    blockWeights: 128, blockBytes: 68, overheadBytes: 4,
    facts: 'x2-facts D5, E1',
  },
  mlx4_g32_f32: {
    id: 'mlx4_g32_f32',
    get label() {
      return tr(`${MLX}, g32, FP32 scale and bias`, `${MLX}, g32, scale và bias FP32`);
    },
    get owner() {
      return tr('MLX, float32 input', 'MLX, đầu vào float32');
    },
    blockWeights: 32, blockBytes: 24, overheadBytes: 8,
    get facts() {
      return tr('x2-facts D5 (scales take the input type); our arithmetic', 'x2-facts D5 (scale lấy kiểu của đầu vào); phép tính của chúng tôi');
    },
  },
  mlx4_g64_f32: {
    id: 'mlx4_g64_f32',
    get label() {
      return tr(`${MLX}, g64, FP32 scale and bias`, `${MLX}, g64, scale và bias FP32`);
    },
    get owner() {
      return tr('MLX, float32 input', 'MLX, đầu vào float32');
    },
    blockWeights: 64, blockBytes: 40, overheadBytes: 8,
    facts: 'x2-facts D5, E1',
  },
  mlx4_g128_f32: {
    id: 'mlx4_g128_f32',
    get label() {
      return tr(`${MLX}, g128, FP32 scale and bias`, `${MLX}, g128, scale và bias FP32`);
    },
    get owner() {
      return tr('MLX, float32 input', 'MLX, đầu vào float32');
    },
    blockWeights: 128, blockBytes: 72, overheadBytes: 8,
    get facts() {
      return tr('x2-facts D5 (scales take the input type); our arithmetic', 'x2-facts D5 (scale lấy kiểu của đầu vào); phép tính của chúng tôi');
    },
  },
};
export const FORMAT_IDS = Object.keys(FORMATS) as FormatId[];

/** Bytes each weight costs in memory, its share of the block's scales included. */
export const bytesPerWeight = (f: FormatId): number => FORMATS[f].blockBytes / FORMATS[f].blockWeights;
export const bitsPerWeight = (f: FormatId): number => 8 * bytesPerWeight(f);
/** Bits per weight that are codes, the rest being scales and offsets. */
export const codeBitsPerWeight = (f: FormatId): number => (8 * (FORMATS[f].blockBytes - FORMATS[f].overheadBytes)) / FORMATS[f].blockWeights;
/** The fraction of a format's bytes that are scales, mins or biases. */
export const overheadShare = (f: FormatId): number => FORMATS[f].overheadBytes / FORMATS[f].blockBytes;
/** How many times fewer bytes than `than` (float32 by default): the speed-up ceiling of a memory-bound GEMV. */
export const byteRatio = (f: FormatId, than: FormatId = 'f32'): number => bytesPerWeight(than) / bytesPerWeight(f);
/** A GEMV's intensity counting weights only, 2 / (bytes per weight). */
export const weightOnlyIntensity = (f: FormatId): number => 2 / bytesPerWeight(f);

// ---------------------------------------------------------------- layers
export interface Layer {
  id: string;
  in: number;
  out: number;
  label: string;
  /** Where the shape comes from. */
  source: string;
}
export const LAYERS: Layer[] = [
  {
    id: '1024x4096', in: 1024, out: 4096,
    get label() {
      return tr("1024 → 4096 (the book's layer)", '1024 → 4096 (lớp của sách)');
    },
    source: 'ch. 20 §20.6, p. 505',
  },
  {
    id: '4096x4096', in: 4096, out: 4096, label: '4096 → 4096',
    get source() {
      return tr('a square projection', 'một phép chiếu vuông');
    },
  },
  {
    id: '4096x14336', in: 4096, out: 14336, label: '4096 → 14336',
    get source() {
      return tr('an up-projection', 'một phép chiếu lên (up-projection)');
    },
  },
  {
    id: '14336x4096', in: 14336, out: 4096, label: '14336 → 4096',
    get source() {
      return tr('a down-projection', 'một phép chiếu xuống (down-projection)');
    },
  },
];
export const layer = (id: string): Layer => LAYERS.find((l) => l.id === id) ?? LAYERS[0];

// ---------------------------------------------------------------- one GEMV
export interface Traffic {
  flops: number;
  weightBytes: number;
  /** Bytes of X read: B · in · a. */
  inputBytes: number;
  /** Bytes of Y written: B · out · a. */
  outputBytes: number;
  bytes: number;
  /** FLOP/B. */
  intensity: number;
}

/** One GEMV of a batch of `batch` rows through `L`, weights in `f`, activations of `actBytes` bytes. */
export function gemv(f: FormatId, L: { in: number; out: number }, batch: number, actBytes: number): Traffic {
  const flops = 2 * batch * L.in * L.out;
  const weightBytes = bytesPerWeight(f) * L.in * L.out;
  const inputBytes = actBytes * batch * L.in;
  const outputBytes = actBytes * batch * L.out;
  const bytes = weightBytes + inputBytes + outputBytes;
  return { flops, weightBytes, inputBytes, outputBytes, bytes, intensity: flops / bytes };
}

export interface Ceiling {
  seconds: number;
  memorySeconds: number;
  computeSeconds: number;
  bound: 'memory' | 'compute';
  /** The roof at this intensity, GFLOP/s. */
  gflops: number;
}

/** The roofline's best time for this traffic on `dev`. */
export function ceiling(t: Traffic, dev: Device): Ceiling {
  const memorySeconds = t.bytes / (dev.bw * 1e9);
  const computeSeconds = t.flops / (effectivePeak(dev) * 1e9);
  const seconds = Math.max(memorySeconds, computeSeconds);
  return { seconds, memorySeconds, computeSeconds, bound: memorySeconds >= computeSeconds ? 'memory' : 'compute', gflops: t.flops / seconds / 1e9 };
}

/** Predicted speed-up of `f` weights over float32 weights, same layer, batch and activations. */
export function speedupOverF32(f: FormatId, L: { in: number; out: number }, batch: number, actBytes: number, dev: Device): number {
  return ceiling(gemv('f32', L, batch, actBytes), dev).seconds / ceiling(gemv(f, L, batch, actBytes), dev).seconds;
}

/** The intensity a layer approaches as the batch grows without bound: weights cost nothing per row. */
export function intensityLimit(L: { in: number; out: number }, actBytes: number): number {
  return (2 * L.in * L.out) / (actBytes * (L.in + L.out));
}

/** The smallest batch at which the layer reaches `threshold` FLOP/B, or null if it never does. */
export function kneeBatch(f: FormatId, L: { in: number; out: number }, actBytes: number, threshold: number, max = 1 << 20): number | null {
  if (intensityLimit(L, actBytes) <= threshold) return null;
  for (let b = 1; b <= max; b++) if (gemv(f, L, b, actBytes).intensity >= threshold) return b;
  return null;
}

// ---------------------------------------------------------------- fused or unfused
export type PipelineId = 'f32' | 'unfused-f32' | 'unfused-f16' | 'fused';
export const PIPELINES: { id: PipelineId; label: string; kernels: number }[] = [
  {
    id: 'f32',
    get label() {
      return tr('float32 weights, float32 GEMV', 'trọng số float32, GEMV float32');
    },
    kernels: 1,
  },
  {
    id: 'unfused-f32',
    get label() {
      return tr('dequantise to float32, then a float32 GEMV', 'giải lượng tử hóa sang float32, rồi một GEMV float32');
    },
    kernels: 2,
  },
  {
    id: 'unfused-f16',
    get label() {
      return tr('dequantise to FP16, then an FP16 GEMV', 'giải lượng tử hóa sang FP16, rồi một GEMV FP16');
    },
    kernels: 2,
  },
  {
    id: 'fused',
    get label() {
      return tr('fused: unpack inside the GEMV', 'gộp (fused): giải nén bên trong GEMV');
    },
    kernels: 1,
  },
];

/**
 * Bytes per weight that cross the memory bus in a pipeline with quantised weights `f`.
 * Unfused: read the codes, write a dequantised copy, read the copy back in the GEMV.
 */
export function pipelineBytesPerWeight(p: PipelineId, f: FormatId): number {
  const q = bytesPerWeight(f);
  if (p === 'f32') return 4;
  if (p === 'unfused-f32') return q + 4 + 4;
  if (p === 'unfused-f16') return q + 2 + 2;
  return q;
}

export interface Pass {
  /** Which kernel of the pipeline, 1 or 2. */
  kernel: number;
  kind: 'read-weights' | 'write-copy' | 'read-copy' | 'read-x' | 'write-y';
  label: string;
  bytes: number;
}

/** Every stream that crosses the memory bus, in order. */
export function pipelinePasses(p: PipelineId, f: FormatId, L: { in: number; out: number }, batch: number, actBytes: number): Pass[] {
  const n = L.in * L.out;
  const x: Pass = { kernel: 0, kind: 'read-x', label: tr('read the activations X', 'đọc các activation X'), bytes: actBytes * batch * L.in };
  const y: Pass = { kernel: 0, kind: 'write-y', label: tr('write the outputs Y', 'ghi các đầu ra Y'), bytes: actBytes * batch * L.out };
  const last = p === 'unfused-f32' || p === 'unfused-f16' ? 2 : 1;
  const io = [{ ...x, kernel: last }, { ...y, kernel: last }];
  if (p === 'f32') return [{ kernel: 1, kind: 'read-weights', label: tr('read float32 weights', 'đọc trọng số float32'), bytes: 4 * n }, ...io];
  if (p === 'fused') return [{ kernel: 1, kind: 'read-weights', label: tr(`read ${FORMATS[f].label} weights`, `đọc trọng số ${FORMATS[f].label}`), bytes: bytesPerWeight(f) * n }, ...io];
  const w = p === 'unfused-f32' ? 4 : 2;
  const name = p === 'unfused-f32' ? 'float32' : 'FP16';
  return [
    { kernel: 1, kind: 'read-weights', label: tr(`read ${FORMATS[f].label} weights`, `đọc trọng số ${FORMATS[f].label}`), bytes: bytesPerWeight(f) * n },
    { kernel: 1, kind: 'write-copy', label: tr(`write a ${name} copy`, `ghi một bản sao ${name}`), bytes: w * n },
    { kernel: 2, kind: 'read-copy', label: tr(`read the ${name} copy back`, `đọc lại bản sao ${name}`), bytes: w * n },
    ...io,
  ];
}

export const passBytes = (ps: Pass[]): number => ps.reduce((s, x) => s + x.bytes, 0);

/** Memory time of a pipeline relative to the float32 GEMV of the same layer (bytes only: it is memory-bound). */
export function pipelineRelative(p: PipelineId, f: FormatId, L: { in: number; out: number }, batch: number, actBytes: number): number {
  return passBytes(pipelinePasses(p, f, L, batch, actBytes)) / passBytes(pipelinePasses('f32', f, L, batch, actBytes));
}

// ---------------------------------------------------------------- the size of a model
/** Bytes of `params` weights in format `f`. */
export const modelBytes = (params: number, f: FormatId): number => params * bytesPerWeight(f);

// ---------------------------------------------------------------- bandwidth utilisation
/**
 * Model bandwidth utilisation (PyTorch's MBU, x2-facts A1): bytes of weights read per
 * pass times passes per second, as a fraction of peak bandwidth (GB/s).
 */
export function mbu(weightBytes: number, seconds: number, bandwidthGBs: number): number {
  return weightBytes / seconds / (bandwidthGBs * 1e9);
}

// ---------------------------------------------------------------- the reader's reports
// The labs that write qgemv rows (labs/x2/lab1_q8gemv and lab2_q4gemv; lab X2.4 writes
// qstep rows, read by qstep.ts). A qgemv row is read tolerantly: name 'qgemv'; numbers in,
// out, batch and seconds (or ms); optional strings kernel and format; optional weight_bytes
// and act_bytes. The format defaults to the lab's own (Q8_0 for X2.1, Q4_0 for X2.2) and
// weight bytes to in · out · bytesPerWeight.
export const LAB_X21 = 'x2/lab1_q8gemv';
export const LAB_X22 = 'x2/lab2_q4gemv';
export const QGEMV_LABS: { id: string; num: string; format: FormatId }[] = [
  { id: LAB_X21, num: 'X2.1', format: 'q8_0' },
  { id: LAB_X22, num: 'X2.2', format: 'q4_0' },
];

export interface QgemvRow {
  backend: string;
  lab: string;
  labNum: string;
  kernel: string;
  format: FormatId;
  batch: number;
  in: number;
  out: number;
  seconds: number;
  weightBytes: number;
  actBytes: number;
  /** Weight bytes per second, GB/s. */
  weightGBs: number;
  /** FLOPs over time, GFLOP/s. */
  gflops: number;
  /** The model's intensity for this row's shape and format. */
  intensity: number;
}

const numOf = (v: unknown): number | undefined => (typeof v === 'number' && isFinite(v) ? v : undefined);
const isFormat = (v: unknown): v is FormatId => typeof v === 'string' && Object.hasOwn(FORMATS, v);

/** Every qgemv benchmark row of labs X2.1 and X2.2's reference solutions on GPU backends. */
export function qgemvRows(reports: ReportsLike): QgemvRow[] {
  const out: QgemvRow[] = [];
  for (const backend of Object.keys(reports)) {
    if (backend === 'emu' || backend === 'race') continue;
    for (const lab of QGEMV_LABS) {
      const rows: RowLike[] = reports[backend].results.filter((r) => r.lab === lab.id && r.variant === 'solution').flatMap((r) => r.bench);
      for (const r of rows) {
        if (r.name !== 'qgemv') continue;
        const inW = numOf(r.in);
        const outW = numOf(r.out);
        const batch = numOf(r.batch);
        const seconds = numOf(r.seconds) ?? (numOf(r.ms) != null ? numOf(r.ms)! / 1e3 : undefined);
        if (!inW || !outW || !batch || !seconds) continue;
        const format = isFormat(r.format) ? r.format : lab.format;
        const weightBytes = numOf(r.weight_bytes) ?? inW * outW * bytesPerWeight(format);
        const actBytes = numOf(r.act_bytes) ?? 4;
        const t = gemv(format, { in: inW, out: outW }, batch, actBytes);
        out.push({
          backend, lab: lab.id, labNum: lab.num, kernel: String(r.kernel ?? 'qgemv'), format, batch, in: inW, out: outW, seconds,
          weightBytes, actBytes, weightGBs: weightBytes / seconds / 1e9, gflops: t.flops / seconds / 1e9, intensity: t.intensity,
        });
      }
    }
  }
  return out;
}

/** Lab 5.1's measured bandwidth on a backend, GB/s, or null. */
export function measuredBandwidth(reports: ReportsLike, backend: string): number | null {
  return measuredDevices(reports).find((d) => d.backend === backend)?.bw ?? null;
}

const isUnfused = (r: QgemvRow) => /unfused/.test(r.kernel);

/**
 * Your runs to put beside the model for one layer, batch and format on one backend: the
 * fused kernel of format `f` and the float32 baseline, at most one row each. Rows from the
 * lab whose own format is `f` come first; unfused kernels are left out (they are not one GEMV).
 */
export function pickRuns(rows: QgemvRow[], backend: string, L: { in: number; out: number }, batch: number, f: FormatId): QgemvRow[] {
  const own = (r: QgemvRow) => (QGEMV_LABS.find((l) => l.id === r.lab)?.format === f ? 0 : 1);
  const cands = rows
    .filter((r) => r.backend === backend && r.in === L.in && r.out === L.out && r.batch === batch && !isUnfused(r) && (r.format === f || r.format === 'f32'))
    .sort((a, b) => own(a) - own(b));
  const out: QgemvRow[] = [];
  for (const r of cands) if (!out.some((x) => x.format === r.format)) out.push(r);
  return out;
}

/**
 * Lab X2.2's measured times for the pipelines of section X2.1 on one backend and shape,
 * relative to its float32 GEMV: float32, unfused to float32 (kernel names containing
 * "unfused") and fused Q4_0. Null where a row is missing.
 */
export function measuredPipelines(rows: QgemvRow[], backend: string, L: { in: number; out: number }, batch: number): Partial<Record<PipelineId, number>> | null {
  const mine = rows.filter((r) => r.lab === LAB_X22 && r.backend === backend && r.in === L.in && r.out === L.out && r.batch === batch);
  const base = mine.find((r) => r.format === 'f32');
  if (!base) return null;
  const res: Partial<Record<PipelineId, number>> = { f32: 1 };
  const unf = mine.find((r) => r.format === 'q4_0' && isUnfused(r));
  const fus = mine.find((r) => r.format === 'q4_0' && !isUnfused(r));
  if (unf) res['unfused-f32'] = unf.seconds / base.seconds;
  if (fus) res.fused = fus.seconds / base.seconds;
  return res;
}
