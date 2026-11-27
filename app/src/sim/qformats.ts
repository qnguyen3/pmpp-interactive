// The weight formats module X2 talks about, as data: how many weights a block holds, how
// many bytes the block takes, who defines it and which fact says so. Group 2 owns this
// file; qtraffic.ts, qlayout.ts, quant.ts, qdot.ts and qstep.ts import it read-only.
//
// Sources (none of this is in the book; notes/today/x2-facts.md):
//   D1  llama.cpp Q8_0 (34 B / 32), Q4_0 (18 B / 32), Q4_1 (20 B / 32), ggml-common.h
//   D2  llama.cpp Q4_K (144 B / 256: d, dmin, 12 bytes of 6-bit scales and mins, 128 of quants)
//   D5  MLX affine b-bit, groups of 32, 64 or 128, one scale and one bias per group in the
//       input's type (FP16 for half weights, FP32 for float32 weights), codes packed LSB-first
//   appA D2 (cited in the facts' table)  llama.cpp block_mxfp4, 17 B / 32
//   E1  the table of bytes and bits per weight and the GEMV's 2 / (bytes per weight) FLOP/B
//
// What it leaves out: how the codes inside a block are ordered (quant.ts), anything about
// accuracy (quant.ts), and the activation vector and output, which E1 also ignores.
//
// Erasable TypeScript only and no imports: the browser checks import this file directly.

import { tr } from '../lib/locale.ts';

export type FormatId =
  | 'f32'
  | 'f16'
  | 'bf16'
  | 'q8_0'
  | 'q4_0'
  | 'q4_1'
  | 'q4_k'
  | 'mxfp4'
  | string; // MLX affine ids: 'mlx-<bits>-g<group>-<f16|f32>'

export interface WeightFormat {
  id: FormatId;
  /** What a reader sees. */
  name: string;
  /** Who defines the layout. */
  owner: string;
  /** Weights per block (1 for a plain float). */
  block: number;
  /** Bytes one block occupies, scales included. */
  blockBytes: number;
  /** Bits of each weight's code (the payload before scales). */
  codeBits: number;
  /** The facts-file items that ground the row. */
  fact: string;
}

export const F32: WeightFormat = { id: 'f32', name: 'FP32', owner: 'IEEE 754 binary32', block: 1, blockBytes: 4, codeBits: 32, fact: 'x2-facts E1' }; // i18n-ignore: standard name
export const F16: WeightFormat = { id: 'f16', name: 'FP16', owner: 'IEEE 754 binary16', block: 1, blockBytes: 2, codeBits: 16, fact: 'x2-facts E1; appA-facts A2' }; // i18n-ignore: standard name
export const BF16: WeightFormat = { id: 'bf16', name: 'BF16', owner: 'bfloat16', block: 1, blockBytes: 2, codeBits: 16, fact: 'x2-facts E1; appA-facts A2' };
export const Q8_0: WeightFormat = { id: 'q8_0', name: 'Q8_0', owner: 'llama.cpp', block: 32, blockBytes: 34, codeBits: 8, fact: 'x2-facts D1, E1' };
export const Q4_0: WeightFormat = { id: 'q4_0', name: 'Q4_0', owner: 'llama.cpp', block: 32, blockBytes: 18, codeBits: 4, fact: 'x2-facts D1, E1' };
export const Q4_1: WeightFormat = { id: 'q4_1', name: 'Q4_1', owner: 'llama.cpp', block: 32, blockBytes: 20, codeBits: 4, fact: 'x2-facts D1, E1' };
export const Q4_K: WeightFormat = { id: 'q4_k', name: 'Q4_K', owner: 'llama.cpp', block: 256, blockBytes: 144, codeBits: 4, fact: 'x2-facts D2, E1' };
export const MXFP4: WeightFormat = {
  id: 'mxfp4', name: 'MXFP4', owner: 'OCP MX; llama.cpp block_mxfp4', block: 32, blockBytes: 17, codeBits: 4, fact: 'x2-facts E1; appA-facts D2', // i18n-ignore: standard and type names
};

export type ScaleType = 'f16' | 'f32';
export const MLX_BITS = [1, 2, 3, 4, 5, 6, 8] as const;
export const MLX_GROUPS = [32, 64, 128] as const;

/**
 * MLX's affine format: `bits`-bit codes in groups of `group`, one scale and one bias per
 * group in the input's type (D5: bits 1 to 6 and 8, groups 32, 64, 128). Bytes per group
 * are group·bits/8 for the codes plus two scalars; the group sizes make that a whole
 * number of bytes for every allowed width. (How MLX packs 3-, 5- and 6-bit codes into
 * words is not modelled: only the count of bits.)
 */
export function mlxAffine(bits: number, group: number, scale: ScaleType = 'f16'): WeightFormat {
  if (!(MLX_BITS as readonly number[]).includes(bits)) throw new Error(`MLX supports bits 1 to 6 and 8, not ${bits}`);
  if (!(MLX_GROUPS as readonly number[]).includes(group)) throw new Error(`MLX supports groups 32, 64, 128, not ${group}`);
  const sb = scale === 'f16' ? 2 : 4;
  return {
    id: `mlx-${bits}-g${group}-${scale}`,
    // A getter: FORMATS calls mlxAffine at module load, and the name must follow the reader's language.
    get name() {
      return tr(
        `MLX affine ${bits}-bit, group ${group}, ${scale === 'f16' ? 'FP16' : 'FP32'} scale and bias`,
        `MLX affine ${bits}-bit, nhóm ${group}, scale và bias dạng ${scale === 'f16' ? 'FP16' : 'FP32'}`,
      );
    },
    owner: 'MLX',
    block: group,
    blockBytes: (group * bits) / 8 + 2 * sb,
    codeBits: bits,
    fact: 'x2-facts D5, E1',
  };
}

/** Every row of x2-facts E1's table, in its order. */
export const FORMATS: WeightFormat[] = [
  F32,
  F16,
  BF16,
  Q8_0,
  Q4_0,
  Q4_1,
  Q4_K,
  MXFP4,
  mlxAffine(4, 32, 'f16'),
  mlxAffine(4, 64, 'f16'),
  mlxAffine(4, 128, 'f16'),
  mlxAffine(4, 64, 'f32'),
];

/** A format by id, including any MLX affine id of the form 'mlx-<bits>-g<group>-<f16|f32>'. */
export function formatById(id: FormatId): WeightFormat {
  const f = FORMATS.find((x) => x.id === id);
  if (f) return f;
  const m = /^mlx-(\d)-g(\d+)-(f16|f32)$/.exec(id);
  if (m) return mlxAffine(Number(m[1]), Number(m[2]), m[3] as ScaleType);
  throw new Error('unknown format ' + id);
}

export const bytesPerBlock = (f: WeightFormat) => f.blockBytes;
export const bytesPerWeight = (f: WeightFormat) => f.blockBytes / f.block;
export const bitsPerWeight = (f: WeightFormat) => (8 * f.blockBytes) / f.block;
/** A GEMV's FLOP per byte of weights: one multiply and one add per weight read once (E1). */
export const gemvIntensity = (f: WeightFormat) => 2 / bytesPerWeight(f);

/**
 * Bytes of a rows × cols weight matrix in the format, quantised along each row (cols must be
 * a whole number of blocks, as every real layout requires).
 */
export function matrixBytes(f: WeightFormat, rows: number, cols: number): number {
  if (cols % f.block !== 0) throw new Error(`a row of ${cols} is not a whole number of ${f.block}-weight blocks`);
  return rows * (cols / f.block) * f.blockBytes;
}
