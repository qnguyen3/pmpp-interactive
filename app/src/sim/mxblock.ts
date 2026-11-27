// Block scaling, for section A.5: the decode-ceiling explorer, the "near the top of E4M3"
// reveal, the MX block explorer and its scene. The element formats, rounding and decoding
// are minifloat.ts's (the one format model for appendix A); this file adds what happens
// when many elements share one scale.
//
// Not in the book: §A.5 (pp. 592-593) names no layout and no block format. Sources, all in
// notes/today/appA-facts.md: OCP MX v1.1 (A4) for the formats, the scale rule and the
// clamp; cuBLAS (A5) for the round-up rule and NVIDIA's 16-element FP4 with E4M3 scales;
// llama.cpp (D3) and MLX (D2) for the names and block sizes. Where the model makes a
// choice no checked source states, the comment says "ours".

import {
  E2M1,
  E2M3,
  E3M2,
  E4M3,
  E5M2,
  binade,
  bitsOfValue,
  classify,
  decode,
  emax,
  maxFinite,
  minSubnormal,
  round,
  totalBits,
  values,
  type Format,
  type Overflow,
} from './minifloat';
import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- element formats
export type ElemId = 'E2M1' | 'E2M3' | 'E3M2' | 'E4M3' | 'E5M2';
/** The element formats an MX block can use (MX v1.1 Table 1). */
export const ELEMENTS: Record<ElemId, Format> = { E2M1, E2M3, E3M2, E4M3, E5M2 };
export const ELEMENT_IDS: ElemId[] = ['E2M1', 'E2M3', 'E3M2', 'E4M3', 'E5M2'];

/** The non-negative values of a format, ascending: the grid an element snaps to. */
export const elemGrid = (f: Format) => values(f).filter((v) => v >= 0);

export interface Converted {
  /** The stored bit pattern (-1 when the result has none). */
  code: number;
  value: number;
  /** The input lay beyond the largest finite value after rounding (clamped, or made NaN or infinite). */
  overflowed: boolean;
}

/**
 * Convert x to the format: round to nearest, ties to even (OFP8 r1.1 §5.2, MX §6.3).
 * 'saturate' clamps to the largest finite value; 'ieee' (OFP8's non-saturating mode) gives
 * NaN for E4M3 and infinity for E5M2. FP6 and FP4 have neither, so they always clamp.
 */
export function convert(x: number, f: Format, overflow: Overflow): Converted {
  const c = classify(f, x, { overflow });
  const overflowed = c.status === 'saturated' || (Number.isFinite(x) && (c.status === 'nan' || c.status === 'inf'));
  return { code: c.bits, value: c.value, overflowed };
}

// ---------------------------------------------------------------- scales
/** UE8M0 (MX v1.1 Table 7): bias 127, exponents -127 to 127, 0xFF is NaN, no zero. */
export const UE8M0_RANGE = { bias: 127, min: -127, max: 127 };

/**
 * ocp-floor: MX v1.1 §6.3: shared exponent floor(log2 max|V|) - emax of the element
 *   format; elements beyond the largest normal are clamped to it.
 * cublas-ceil: our reading of cuBLAS's "round the block exponent up" (A5): the smallest
 *   power of two X with max|V| / X <= the largest normal, so nothing needs clamping.
 * e4m3: NVIDIA's 16-element FP4 recipe as we model it: X = max|V| / largest normal,
 *   rounded to a positive E4M3 value, with the tensor-wide scale taken as 1. Ours: the
 *   exact UE4M3 encoding is not in the checked sources.
 */
export type ScaleRule = 'ocp-floor' | 'cublas-ceil' | 'e4m3';

export const SCALE_RULES: Record<ScaleRule, string> = {
  get 'ocp-floor'() {
    return tr('OCP: round down, then clamp (UE8M0)', 'OCP: làm tròn xuống, rồi kẹp (UE8M0)');
  },
  get 'cublas-ceil'() {
    return tr('cuBLAS: round the exponent up (UE8M0)', 'cuBLAS: làm tròn số mũ lên (UE8M0)');
  },
  get e4m3() {
    return tr('E4M3 scale (NVIDIA 16-element FP4)', 'hệ số tỉ lệ E4M3 (FP4 16 phần tử của NVIDIA)');
  },
};

export interface QuantOpts {
  elem: ElemId;
  /** Elements per scale: 32 as in OCP MX, or 16. */
  k: number;
  scale: ScaleRule;
}

export interface BlockScale {
  /** The shared scale X. */
  scale: number;
  /** log2 X for the UE8M0 rules, else null. */
  scaleExp: number | null;
  /** The stored scale byte: the UE8M0 code (exponent + 127), or the E4M3 pattern. */
  scaleCode: number;
}

export function blockScale(amax: number, f: Format, rule: ScaleRule): BlockScale {
  if (rule === 'e4m3') {
    const tiny = minSubnormal(E4M3)!;
    // Ours: an all-zero block (or one too small for any E4M3 scale) takes the smallest scale.
    const s = amax === 0 ? 0 : round(E4M3, amax / maxFinite(f), 'rne', 'saturate');
    const scale = s === 0 ? tiny : s;
    return { scale, scaleExp: null, scaleCode: bitsOfValue(E4M3, scale) };
  }
  let e: number;
  if (amax === 0) e = UE8M0_RANGE.min; // ours: any scale decodes an all-zero block to zeros
  else if (rule === 'ocp-floor') e = binade(amax) - emax(f);
  else {
    const top = maxFinite(f);
    e = binade(amax / top);
    while (top * 2 ** e < amax) e += 1;
    while (top * 2 ** (e - 1) >= amax) e -= 1;
  }
  e = Math.max(UE8M0_RANGE.min, Math.min(UE8M0_RANGE.max, e));
  return { scale: 2 ** e, scaleExp: e, scaleCode: e + UE8M0_RANGE.bias };
}

export interface BlockResult extends BlockScale {
  start: number;
  size: number;
  /** Largest magnitude in the block: what the max reduction finds. */
  amax: number;
  /** Index (in the whole input) of the element that holds it; the first one on a tie. */
  amaxAt: number;
  clamped: number;
  zeroed: number;
}

export interface Quantized {
  opts: QuantOpts;
  blocks: BlockResult[];
  /** v / X, before rounding: where each element lands on the element grid. */
  scaled: number[];
  /** Element codes P_i as stored bit patterns. */
  codes: number[];
  /** X × P_i: what a kernel reads back. */
  decoded: number[];
  /** decoded - input. */
  err: number[];
  /** Which block each element belongs to. */
  blockOf: number[];
  /** Elements beyond the largest normal after dividing by X, set to it. */
  clamped: number;
  /** Non-zero inputs that came back as 0. */
  zeroed: number;
  /** RMS of the error over RMS of the input. */
  rmsRel: number;
  /** The same, leaving out the single largest-magnitude element of the input. */
  rmsRelRest: number;
  /** Storage, scales included. */
  bitsPerWeight: number;
}

const rms = (xs: number[]) => Math.sqrt(xs.reduce((s, x) => s + x * x, 0) / Math.max(1, xs.length));

/** Quantise values in blocks of k elements, one shared scale per block: v_i ≈ X · P_i (MX v1.1 §5). */
export function quantizeBlock(values: number[], opts: QuantOpts): Quantized {
  const f = ELEMENTS[opts.elem];
  const top = maxFinite(f);
  const blocks: BlockResult[] = [];
  const scaled: number[] = [];
  const codes: number[] = [];
  const decoded: number[] = [];
  const blockOf: number[] = [];
  for (let start = 0; start < values.length; start += opts.k) {
    const part = values.slice(start, start + opts.k);
    let amax = 0;
    let amaxAt = start;
    part.forEach((v, i) => {
      if (Math.abs(v) > amax) {
        amax = Math.abs(v);
        amaxAt = start + i;
      }
    });
    const sc = blockScale(amax, f, opts.scale);
    let clamped = 0;
    let zeroed = 0;
    for (const v of part) {
      const t = v / sc.scale;
      const p = round(f, t, 'rne', 'saturate');
      if (Math.abs(t) > top) clamped += 1;
      const d = p * sc.scale;
      if (v !== 0 && d === 0) zeroed += 1;
      scaled.push(t);
      codes.push(bitsOfValue(f, p));
      decoded.push(d);
      blockOf.push(blocks.length);
    }
    blocks.push({ start, size: part.length, amax, amaxAt, ...sc, clamped, zeroed });
  }
  const err = decoded.map((d, i) => d - values[i]);
  let big = 0;
  values.forEach((v, i) => {
    if (Math.abs(v) > Math.abs(values[big])) big = i;
  });
  const rest = (xs: number[]) => xs.filter((_, i) => i !== big);
  const rIn = rms(values);
  const rRest = rms(rest(values));
  return {
    opts,
    blocks,
    scaled,
    codes,
    decoded,
    err,
    blockOf,
    clamped: blocks.reduce((s, b) => s + b.clamped, 0),
    zeroed: blocks.reduce((s, b) => s + b.zeroed, 0),
    rmsRel: rIn === 0 ? 0 : rms(err) / rIn,
    rmsRelRest: rRest === 0 ? 0 : rms(rest(err)) / rRest,
    bitsPerWeight: totalBits(f) + 8 / opts.k,
  };
}

/**
 * The max reduction that finds a block's largest magnitude, as a tree (chapter 10's
 * pairing of neighbours): the indices still in the running after each round. A tie keeps
 * the left one. The last level holds blocks' amaxAt.
 */
export function maxTree(values: number[]): number[][] {
  let cur = values.map((_, i) => i);
  const levels = [cur];
  while (cur.length > 1) {
    const next: number[] = [];
    for (let j = 0; j < cur.length; j += 2)
      next.push(j + 1 < cur.length && Math.abs(values[cur[j + 1]]) > Math.abs(values[cur[j]]) ? cur[j + 1] : cur[j]);
    levels.push(next);
    cur = next;
  }
  return levels;
}

/** Decode one stored element: X × the element's value. */
export const decodeElement = (code: number, f: Format, scale: number) => decode(f, code).value * scale;

/** The documented format an option set amounts to, or null. */
export function namedFormat(o: QuantOpts): string | null {
  const ue8 = o.scale !== 'e4m3';
  if (o.k === 32 && ue8) {
    if (o.elem === 'E2M1') return 'MXFP4';
    if (o.elem === 'E2M3' || o.elem === 'E3M2') return 'MXFP6';
    if (o.elem === 'E4M3' || o.elem === 'E5M2') return 'MXFP8';
  }
  if (o.k === 16 && o.scale === 'e4m3' && o.elem === 'E2M1') return tr("NVIDIA's 16-element FP4 (NVFP4 in MLX and llama.cpp)", 'FP4 16 phần tử của NVIDIA (NVFP4 trong MLX và llama.cpp)');
  return null;
}

// ---------------------------------------------------------------- storage and the decode ceiling
export type WeightFormat = 'fp32' | 'bf16' | 'fp8' | 'mxfp8' | 'mxfp4' | 'nvfp4';

export const WEIGHT_FORMATS: Record<WeightFormat, { label: string; elemBits: number; block: number | null; scaleBits: number; extra: string }> = {
  fp32: { label: 'FP32', elemBits: 32, block: null, scaleBits: 0, extra: '' },
  bf16: {
    get label() {
      return tr('BF16 or FP16', 'BF16 hoặc FP16');
    },
    elemBits: 16, block: null, scaleBits: 0, extra: '',
  },
  fp8: {
    get label() {
      return tr('FP8 with one scale per tensor', 'FP8 với một hệ số tỉ lệ cho mỗi tensor');
    },
    elemBits: 8, block: null, scaleBits: 0,
    get extra() {
      return tr('plus one scale per tensor', 'cộng một hệ số tỉ lệ cho mỗi tensor');
    },
  },
  mxfp8: {
    get label() {
      return tr('MXFP8 (32 per UE8M0 scale)', 'MXFP8 (32 phần tử mỗi hệ số tỉ lệ UE8M0)');
    },
    elemBits: 8, block: 32, scaleBits: 8, extra: '',
  },
  mxfp4: {
    get label() {
      return tr('MXFP4 (32 per UE8M0 scale)', 'MXFP4 (32 phần tử mỗi hệ số tỉ lệ UE8M0)');
    },
    elemBits: 4, block: 32, scaleBits: 8, extra: '',
  },
  nvfp4: {
    get label() {
      return tr("NVIDIA's FP4 (16 per E4M3 scale)", 'FP4 của NVIDIA (16 phần tử mỗi hệ số tỉ lệ E4M3)');
    },
    elemBits: 4, block: 16, scaleBits: 8,
    get extra() {
      return tr('plus one scale per tensor', 'cộng một hệ số tỉ lệ cho mỗi tensor');
    },
  },
};
export const WEIGHT_FORMAT_IDS = Object.keys(WEIGHT_FORMATS) as WeightFormat[];

/** Bits per weight with block scales included; a per-tensor scale is left out (it rounds to nothing per weight). */
export function bitsPerWeight(f: WeightFormat): number {
  const w = WEIGHT_FORMATS[f];
  return w.elemBits + (w.block ? w.scaleBits / w.block : 0);
}

/** Bytes of one stored block: n elements of elemBits each, plus one byte per scale. */
export const blockBytes = (elemBits: number, n: number, scales: number) => (n * elemBits) / 8 + scales;

/** At batch 1 each weight is read once for one multiply and one add: 2 operations per weight. */
export const opsPerByte = (bits: number) => 2 / (bits / 8);

/** Bytes of weights a decode step reads. */
export const weightBytes = (params: number, bits: number) => (params * bits) / 8;

/**
 * The bandwidth-bound ceiling on tokens per second for one decode step at batch 1: every
 * weight read once per token, nothing else read, the bandwidth fully used.
 */
export function decodeCeiling(params: number, bits: number, bandwidthGBs: number): number {
  return (bandwidthGBs * 1e9) / weightBytes(params, bits);
}

// ---------------------------------------------------------------- sample data
/** mulberry32: small, seeded, well mixed. */
export function rng(seed: number): () => number {
  let h = seed >>> 0;
  return () => {
    h = (h + 0x6d2b79f5) >>> 0;
    let t = Math.imul(h ^ (h >>> 15), h | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * n weight-like values: normal, mean 0, standard deviation sigma, rounded to float32 so a
 * C++ program given the same numbers sees the same values. When outlier > 0, element `at`
 * is replaced by one value of that magnitude (keeping the sample's sign).
 */
export function sampleBlock(seed: number, n = 32, sigma = 0.25, outlier = 0, at = 7): number[] {
  const r = rng(seed);
  const out: number[] = [];
  while (out.length < n) {
    const u = Math.max(r(), 1e-12);
    const v = r();
    out.push(Math.fround(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * sigma));
  }
  if (outlier > 0) out[at] = Math.fround(out[at] < 0 ? -outlier : outlier);
  return out;
}

/** The block the "Quantising one block" scene draws; fixed, so its script can quote the numbers. */
export const SCENE_BLOCK = { seed: 11, sigma: 0.25, outlier: 12, at: 7 };
export const sceneValues = () => sampleBlock(SCENE_BLOCK.seed, 32, SCENE_BLOCK.sigma, SCENE_BLOCK.outlier, SCENE_BLOCK.at);
