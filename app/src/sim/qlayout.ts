// One warp walking a row of quantised weights, for section X2.4 (ours: not in the book).
//
// The kernel shape is lab 22.1's gather (notes/ch22.md §22.2, pp. 532–536): one warp owns
// output row j of W, stored (out, in) row-major, and its lanes walk along `in` together. What
// changes with quantised weights is the unit of work: a lane takes a piece of a block, and the
// block's scale is applied once per piece. This file says which bytes each lane loads.
//
// Layouts (notes/today/x2-facts.md):
//   q8_0-split   Q8_0 codes (int8, 32 per block) in one array, one FP16 scale per block (as
//                uint16_t bits) in a second array: lab X2.1's layout
//                (labs/x2/lab1_q8gemv/solution/kernel.cu). Separate arrays keep the codes
//                4-byte aligned (A5).
//   q4_0-blocks  llama.cpp's Q4_0: 18-byte blocks, a 2-byte FP16 d then 16 code bytes (D1).
//                Blocks are only 2-byte aligned, so every other block's codes start 2 mod 4
//                (A5, C5).
//   mlx4-split   MLX-style 4-bit: codes packed in 32-bit words, 32 code bytes per group of
//                64, scales and biases in their own arrays (D5); here float32 scales, the
//                type MLX gives float32 input.
// Mapping: a block is worked on by `lanesPerBlock` neighbouring lanes, so one iteration of
// the loop covers warpWidth / lanesPerBlock blocks. Each lane issues codeBytes /
// (lanesPerBlock · loadBytes) load instructions per iteration. Two orders for those loads:
//   interleaved  in load k, lane `sub` of its block reads at code offset
//                (k · lanesPerBlock + sub) · loadBytes: neighbouring lanes read neighbouring
//                bytes in every load;
//   piece        lane `sub` owns the contiguous piece starting at sub · codeBytes /
//                lanesPerBlock and reads it front to back: lab X2.2's mapping (lane s of 4
//                reads qs bytes 4s to 4s + 3 as two 16-bit loads).
// One scale load per iteration: every lane reads its block's scale (lanes of one block read
// the same address).
// Coalescing is chapter 6's segment model, coalesce.ts warpAccess (read-only), with
// SEGMENT_MODELS for the sizes (ch06-facts A2 to A4, B1; Apple documents none, C1 and
// infra-facts C13).
// What it leaves out: caches (a segment fetched by one instruction is not remembered by the
// next), timing, the activation loads, and anything a compiler might merge or split.
//
// The warp width is always a parameter (32 on NVIDIA, 64 or 32 on AMD, read from the
// pipeline on Metal: ch05-facts B5, ch03-facts C9); nothing here assumes 32.
import { SEGMENT_MODELS, warpAccess, type SegmentModelId, type WarpAccess } from './coalesce';
import { tr } from '../lib/locale.ts';

export { SEGMENT_MODELS, type SegmentModelId };

export type LayoutId = 'q8_0-split' | 'q4_0-blocks' | 'mlx4-split';

export interface QLayout {
  id: LayoutId;
  label: string;
  owner: string;
  /** Weights per block (or group). */
  blockWeights: number;
  /** Code bytes per block. */
  codeBytes: number;
  /** Bytes of one scale. */
  scaleBytes: number;
  /** Scales and codes interleaved in one array of blocks (Q4_0), or in separate arrays. */
  interleaved: boolean;
  facts: string;
}

export const LAYOUTS: Record<LayoutId, QLayout> = {
  'q8_0-split': {
    id: 'q8_0-split',
    get label() {
      return tr('Q8_0, codes and scales in separate arrays (lab X2.1)', 'Q8_0, mã và hệ số tỷ lệ ở các mảng riêng (lab X2.1)');
    },
    get owner() {
      return tr("llama.cpp's Q8_0 values, split into two arrays", 'giá trị Q8_0 của llama.cpp, tách thành hai mảng');
    },
    blockWeights: 32, codeBytes: 32, scaleBytes: 2, interleaved: false,
    facts: 'x2-facts D1, A5; lab X2.1 solution', // i18n-ignore: facts-file citation
  },
  'q4_0-blocks': {
    id: 'q4_0-blocks',
    get label() {
      return tr("Q4_0 18-byte blocks (llama.cpp's layout)", 'khối Q4_0 18 byte (bố cục của llama.cpp)');
    },
    owner: 'llama.cpp',
    blockWeights: 32, codeBytes: 16, scaleBytes: 2, interleaved: true, facts: 'x2-facts D1, A5, C5',
  },
  'mlx4-split': {
    id: 'mlx4-split',
    get label() {
      return tr('MLX-style 4-bit, group 64, scales in their own array', '4-bit kiểu MLX, nhóm 64, hệ số tỷ lệ ở mảng riêng');
    },
    owner: 'MLX',
    blockWeights: 64, codeBytes: 32, scaleBytes: 4, interleaved: false, facts: 'x2-facts D5',
  },
};
export const LAYOUT_IDS = Object.keys(LAYOUTS) as LayoutId[];

/** Bytes from one block's start to the next block's in the code array. */
export const blockStride = (l: QLayout): number => (l.interleaved ? l.scaleBytes + l.codeBytes : l.codeBytes);

export type LoadOrder = 'interleaved' | 'piece';

export interface Mapping {
  layout: LayoutId;
  /** Which bytes each load of a lane reads (see the header); interleaved when left out. */
  order?: LoadOrder;
  /** Lanes that share one block: 1, 2, 4 or 8. */
  lanesPerBlock: number;
  /** Bytes per load instruction per lane: 1, 2 or 4. */
  loadBytes: number;
  /** Lanes per warp: 32 or 64 (Metal reads it from the pipeline). */
  warpWidth: number;
  /** Row length, in weights. */
  inW: number;
  /** Which row of W the warp owns. */
  row?: number;
}

/** Why a mapping cannot run, or null. */
export function problem(m: Mapping): string | null {
  const l = LAYOUTS[m.layout];
  if (m.inW <= 0 || m.inW % l.blockWeights !== 0) return tr(`the row length must be a whole number of blocks of ${l.blockWeights} weights`, `độ dài hàng phải là một số nguyên lần khối ${l.blockWeights} trọng số`);
  if (m.warpWidth % m.lanesPerBlock !== 0) return tr('the lanes of a warp must split evenly into blocks', 'các lane của một warp phải chia đều vào các khối');
  if (l.codeBytes % (m.lanesPerBlock * m.loadBytes) !== 0)
    return tr(
        `a block has ${l.codeBytes} code bytes, which ${m.lanesPerBlock} lanes cannot split into ${m.loadBytes}-byte loads`,
        `một khối có ${l.codeBytes} byte mã, ${m.lanesPerBlock} lane không thể chia chúng thành các lần tải ${m.loadBytes} byte`,
      );
  return null;
}

export const blocksPerRow = (m: Mapping): number => m.inW / LAYOUTS[m.layout].blockWeights;
export const blocksPerIteration = (m: { warpWidth: number; lanesPerBlock: number }): number => m.warpWidth / m.lanesPerBlock;
/** Load instructions each lane issues per iteration for its piece of a block. */
export const loadsPerLane = (m: Mapping): number => LAYOUTS[m.layout].codeBytes / (m.lanesPerBlock * m.loadBytes);

/** Byte address of code byte `offset` of block `block` of the warp's row. */
export function codeAddress(m: Mapping, block: number, offset: number): number {
  const l = LAYOUTS[m.layout];
  const b = (m.row ?? 0) * blocksPerRow(m) + block;
  return b * blockStride(l) + (l.interleaved ? l.scaleBytes : 0) + offset;
}

/** Byte address of block `block`'s scale: inside the block for Q4_0, in the scale array otherwise. */
export function scaleAddress(m: Mapping, block: number): number {
  const l = LAYOUTS[m.layout];
  const b = (m.row ?? 0) * blocksPerRow(m) + block;
  return l.interleaved ? b * blockStride(l) : b * l.scaleBytes;
}

export interface Plan {
  blocks: number;
  blocksPerIteration: number;
  iterations: number;
  /** Lanes that pass the bounds test in the last iteration. */
  activeLast: number;
  /** Lanes that take the false side of the bounds test in the last iteration. */
  idleLast: number;
  /** Idle lane-iterations over all lane-iterations of the row. */
  idleFraction: number;
}

/** How a warp covers a row of `inW` weights. */
export function iterationPlan(inW: number, layout: LayoutId, lanesPerBlock: number, warpWidth: number): Plan {
  const blocks = inW / LAYOUTS[layout].blockWeights;
  const per = blocksPerIteration({ warpWidth, lanesPerBlock });
  const iterations = Math.ceil(blocks / per);
  const activeLast = (blocks - (iterations - 1) * per) * lanesPerBlock;
  const idleLast = warpWidth - activeLast;
  return { blocks, blocksPerIteration: per, iterations, activeLast, idleLast, idleFraction: idleLast / (iterations * warpWidth) };
}

export interface LaneLoad {
  lane: number;
  /** False on the false side of the bounds test. */
  active: boolean;
  block: number;
  /** Position of the lane among its block's lanes. */
  sub: number;
  addr: number;
  bytes: number;
  /** The address is not a multiple of the load's size. */
  misaligned: boolean;
}

export interface Step {
  iteration: number;
  load: number;
  codeLanes: LaneLoad[];
  code: WarpAccess;
  scaleLanes: LaneLoad[];
  scale: WarpAccess;
  /** Active lanes whose code load is misaligned. */
  misaligned: number;
  idle: number;
}

/** The warp's code load `load` and its scale load in `iteration`, against segments of `segmentBytes`. */
export function loadStep(m: Mapping, iteration: number, load: number, segmentBytes: number): Step {
  const l = LAYOUTS[m.layout];
  const per = blocksPerIteration(m);
  const nBlocks = blocksPerRow(m);
  const codeLanes: LaneLoad[] = [];
  const scaleLanes: LaneLoad[] = [];
  for (let lane = 0; lane < m.warpWidth; lane++) {
    const block = iteration * per + Math.floor(lane / m.lanesPerBlock);
    const sub = lane % m.lanesPerBlock;
    const active = block < nBlocks;
    const offset = m.order === 'piece' ? sub * (l.codeBytes / m.lanesPerBlock) + load * m.loadBytes : (load * m.lanesPerBlock + sub) * m.loadBytes;
    const addr = codeAddress(m, block, offset);
    codeLanes.push({ lane, active, block, sub, addr, bytes: m.loadBytes, misaligned: addr % m.loadBytes !== 0 });
    const sa = scaleAddress(m, block);
    scaleLanes.push({ lane, active, block, sub, addr: sa, bytes: l.scaleBytes, misaligned: sa % l.scaleBytes !== 0 });
  }
  const on = (xs: LaneLoad[]) => xs.filter((x) => x.active).map((x) => x.addr);
  return {
    iteration,
    load,
    codeLanes,
    code: warpAccess(on(codeLanes), m.loadBytes, segmentBytes),
    scaleLanes,
    scale: warpAccess(on(scaleLanes), l.scaleBytes, segmentBytes),
    misaligned: codeLanes.filter((x) => x.active && x.misaligned).length,
    idle: codeLanes.filter((x) => !x.active).length,
  };
}

/** Byte range [lo, hi) of the row's array covered by the blocks of one iteration (scales included for Q4_0). */
export function iterationSpan(m: Mapping, iteration: number): [number, number] {
  const l = LAYOUTS[m.layout];
  const per = blocksPerIteration(m);
  const first = iteration * per;
  const last = Math.min(blocksPerRow(m), first + per) - 1;
  return [codeAddress(m, first, 0) - (l.interleaved ? l.scaleBytes : 0), codeAddress(m, last, l.codeBytes)];
}

/** Byte range [lo, hi) of the scales of one iteration's blocks. */
export function scaleSpan(m: Mapping, iteration: number): [number, number] {
  const l = LAYOUTS[m.layout];
  const per = blocksPerIteration(m);
  const first = iteration * per;
  const last = Math.min(blocksPerRow(m), first + per) - 1;
  return [scaleAddress(m, first), scaleAddress(m, last) + l.scaleBytes];
}

/** The steps of a whole row, in the order the stepper walks them. */
export function rowSteps(m: Mapping): { iteration: number; load: number }[] {
  const out: { iteration: number; load: number }[] = [];
  const its = Math.ceil(blocksPerRow(m) / blocksPerIteration(m));
  for (let it = 0; it < its; it++) for (let k = 0; k < loadsPerLane(m); k++) out.push({ iteration: it, load: k });
  return out;
}

/** The segment size of a model, with Apple's undocumented size taken as `assumed`. */
export function segmentBytesOf(id: SegmentModelId, assumed = 32): number {
  return SEGMENT_MODELS[id].segmentBytes ?? assumed;
}
