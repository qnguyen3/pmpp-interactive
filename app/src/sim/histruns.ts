// Thread-level privatisation, chapter 9 section 9.6 (book pp. 217–219, Fig. 9.15).
// Runs the "Run merging, one warp" stepper, the "Atomics per element" table and the
// numbers in the section's questions.
//
// What Fig. 9.15 does, per thread (notes/ch09.md, listing of Fig. 9.15):
//   if (first element < n) {                         // line 11: bounds check on the first pixel
//     b = image[first]; bin_r = 1;                     // lines 12–13
//     for (c = 1; c < COARSE_FACTOR; ++c) {            // line 14
//       i = line 15 index; if (i < n) {                 // lines 15–16
//         bNext = image[i];
//         if (bNext == b) ++bin_r;                      // extend the run
//         else { bins_s[b] += bin_r (one atomic); b = bNext; bin_r = 1; }   // flush and restart
//     } }
//     bins_s[b] += bin_r (one atomic);                  // lines 29–31: the final flush, always
//   }
// The book's line 15 is the interleaved index of Fig. 9.14. For comparison the model also
// runs the same code with the contiguous index of Fig. 9.12 (the book does not print
// that combination; it is ours).
//
// The data presets are ours, generated from a fixed seed: uniform noise, a constant
// image, "sky" (each 4096-pixel row one value, neighbouring rows different), "stripes"
// (runs of 8 equal bytes, each run a new value), and token ids drawn from a Zipf-like
// distribution over 256 ids (a stand-in: real vocabularies are far larger).
//
// What this leaves out (the lesson says so beside the stepper):
//  - No time. It counts atomics and iterations in which a warp's lanes take different
//    sides of the new branch; it does not say what divergence or an atomic costs.
//  - A warp is warpWidth consecutive threads of a block, and its lanes run each
//    iteration together.
//
// Erasable TypeScript and no imports: the browser check imports this file directly.
// elementIndex() repeats line 12 of histcoarsen.ts; a unit test ties the two together.

import { tr } from '../lib/locale.ts';

export type Partitioning = 'contiguous' | 'interleaved';

/** Line 12 of Figs. 9.12 and 9.14 (line 15 of Fig. 9.15). */
export function elementIndex(p: Partitioning, blockIdx: number, threadIdx: number, c: number, blockDim: number, coarse: number): number {
  const segment = coarse * blockIdx * blockDim;
  return p === 'contiguous' ? segment + threadIdx * coarse + c : segment + c * blockDim + threadIdx;
}

// ---------------------------------------------------------------- data

export type PresetId = 'noise' | 'constant' | 'sky' | 'stripes' | 'tokens';

export const IMAGE_PIXELS = 65_536;
export const SKY_WIDTH = 4096;
export const STRIPE = 8;
export const CONSTANT_VALUE = 200;
export const BINS = 256;

export const PRESETS: Record<PresetId, { id: PresetId; label: string; note: string }> = {
  noise: {
    id: 'noise',
    get label() {
      return tr('noise', 'nhiễu');
    },
    get note() {
      return tr('every byte drawn uniformly from 0 to 255', 'mỗi byte được rút ngẫu nhiên đều trong khoảng 0 đến 255');
    },
  },
  constant: {
    id: 'constant',
    get label() {
      return tr('constant', 'hằng');
    },
    get note() {
      return tr(`every byte ${CONSTANT_VALUE}`, `mọi byte đều bằng ${CONSTANT_VALUE}`);
    },
  },
  sky: {
    id: 'sky',
    get label() {
      return tr('sky', 'bầu trời');
    },
    get note() {
      return tr(`rows ${SKY_WIDTH} pixels wide, each row one value, neighbouring rows different`, `các hàng rộng ${SKY_WIDTH} pixel, mỗi hàng một giá trị, các hàng kề nhau khác nhau`);
    },
  },
  stripes: {
    id: 'stripes',
    get label() {
      return tr('short stripes', 'sọc ngắn');
    },
    get note() {
      return tr(`runs of ${STRIPE} equal bytes, each run a new value`, `các dãy ${STRIPE} byte bằng nhau, mỗi dãy một giá trị mới`);
    },
  },
  tokens: {
    id: 'tokens',
    get label() {
      return tr('token ids', 'id token');
    },
    get note() {
      return tr('ids 0 to 255 drawn from a Zipf-like distribution, neighbours independent', 'các id từ 0 đến 255 rút từ một phân phối kiểu Zipf, các phần tử kề nhau độc lập');
    },
  },
};
export const PRESET_ORDER: PresetId[] = ['noise', 'constant', 'sky', 'stripes', 'tokens'];

/** A small seeded generator (mulberry32), so every run and every test sees the same data. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Sky row value: a slow gradient in which neighbouring rows always differ. */
export const skyValue = (row: number): number => 230 - (row % 64);

export function makeImage(preset: PresetId, n = IMAGE_PIXELS, seed = 9): Uint8Array {
  const img = new Uint8Array(n);
  const r = rng(seed);
  switch (preset) {
    case 'noise':
      for (let i = 0; i < n; i++) img[i] = Math.floor(r() * 256);
      break;
    case 'constant':
      img.fill(CONSTANT_VALUE);
      break;
    case 'sky':
      for (let i = 0; i < n; i++) img[i] = skyValue(Math.floor(i / SKY_WIDTH));
      break;
    case 'stripes': {
      let v = -1;
      for (let i = 0; i < n; i++) {
        if (i % STRIPE === 0) {
          let w = Math.floor(r() * 255);
          if (w >= v && v >= 0) w++; // never the previous run's value: runs are exactly STRIPE long
          v = w;
        }
        img[i] = v;
      }
      break;
    }
    case 'tokens': {
      // P(id k) proportional to 1 / (k + 1): a few ids are common, most are rare.
      const cdf: number[] = [];
      let s = 0;
      for (let k = 0; k < BINS; k++) cdf.push((s += 1 / (k + 1)));
      for (let i = 0; i < n; i++) {
        const u = r() * s;
        let lo = 0;
        let hi = BINS - 1;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if (cdf[mid] < u) lo = mid + 1;
          else hi = mid;
        }
        img[i] = lo;
      }
      break;
    }
  }
  return img;
}

/** The worked example: eight lanes, eight elements each, in each lane's own order. Lane 0 is q-runs-count. */
export const WORKED: readonly (readonly number[])[] = [
  [7, 7, 7, 3, 3, 7, 7, 9],
  [7, 7, 7, 7, 7, 7, 7, 7],
  [5, 6, 7, 8, 9, 10, 11, 12],
  [7, 7, 7, 3, 3, 7, 7, 9],
  [2, 2, 2, 2, 9, 9, 9, 9],
  [0, 0, 0, 0, 0, 0, 0, 0],
  [1, 2, 1, 2, 1, 2, 1, 2],
  [7, 7, 3, 3, 7, 7, 9, 9],
];
export const WORKED_COARSE = 8;

/**
 * One block's image that hands thread t the sequence WORKED[t mod 8] under the chosen
 * partitioning, so the lanes see the same values whichever way elements are handed out.
 */
export function workedImage(p: Partitioning, blockDim: number): Uint8Array {
  const img = new Uint8Array(blockDim * WORKED_COARSE);
  for (let t = 0; t < blockDim; t++)
    for (let c = 0; c < WORKED_COARSE; c++) img[elementIndex(p, 0, t, c, blockDim, WORKED_COARSE)] = WORKED[t % 8][c];
  return img;
}

// ---------------------------------------------------------------- one warp, one step at a time

/**
 * What a lane did in one step.
 *  first: loaded its first element (b, bin_r = 1);   extend: same value, ++bin_r;
 *  flush: different value, one atomic of bin_r into bins_s[b], then restart;
 *  inactive: its element is past the end (bounds check fails), or it had no first element;
 *  final: the flush after the loop;   wait: nothing yet.
 */
export type Branch = 'wait' | 'first' | 'extend' | 'flush' | 'inactive' | 'final';

export interface LaneState {
  lane: number;
  thread: number;
  /** Element read in this step, or null. */
  element: number | null;
  /** Its value (bNext, or the first b). */
  value: number | null;
  branch: Branch;
  /** Amount added by this step's atomic (0 if none) and the bin it went to. */
  amount: number;
  bin: number | null;
  /** Registers after the step. */
  b: number | null;
  binR: number;
  /** Running totals for this lane. */
  atomics: number;
  elements: number;
  /** False once the lane is known to have no first element. */
  live: boolean;
}

export interface WarpConfig {
  p: Partitioning;
  n: number;
  blockDim: number;
  coarse: number;
  block: number;
  /** First thread of the warp shown, and how many lanes it has. */
  firstThread: number;
  lanes: number;
}

export interface WarpState {
  cfg: WarpConfig;
  /** 0: nothing run; 1: first loads; 2 … coarse: loop iterations c = 1 … coarse − 1; coarse + 1: final flush. */
  step: number;
  /** Loop iteration c of the step just run, or null for the first loads and the final flush. */
  c: number | null;
  lanes: LaneState[];
  /** Active lanes took both sides of the new branch in this step. */
  divergent: boolean;
  /** Totals over the steps run so far. */
  atomics: number;
  elements: number;
  divergentSteps: number;
  /** What these lanes have added to the block's private copy, by bin. */
  counts: number[];
  done: boolean;
}

export function initial(cfg: WarpConfig): WarpState {
  return {
    cfg,
    step: 0,
    c: null,
    lanes: Array.from({ length: cfg.lanes }, (_, l) => ({
      lane: l, thread: cfg.firstThread + l, element: null, value: null, branch: 'wait' as Branch, amount: 0, bin: null,
      b: null, binR: 0, atomics: 0, elements: 0, live: true,
    })),
    divergent: false,
    atomics: 0,
    elements: 0,
    divergentSteps: 0,
    counts: new Array(BINS).fill(0),
    done: false,
  };
}

/** Runs one more step of Fig. 9.15 for every lane of the warp. */
export function step(img: ArrayLike<number>, s: WarpState): WarpState {
  if (s.done) return s;
  const { p, n, blockDim, coarse, block } = s.cfg;
  const k = s.step + 1;
  const counts = s.counts.slice();
  let atomics = s.atomics;
  let elements = s.elements;
  let sawExtend = false;
  let sawFlush = false;
  const c = k === 1 ? 0 : k <= coarse ? k - 1 : null;
  const lanes = s.lanes.map((L): LaneState => {
    const x: LaneState = { ...L, element: null, value: null, amount: 0, bin: null };
    if (k === 1) {
      const e = elementIndex(p, block, L.thread, 0, blockDim, coarse);
      if (e >= n) return { ...x, branch: 'inactive', live: false };
      const v = img[e];
      elements++;
      return { ...x, element: e, value: v, branch: 'first', b: v, binR: 1, elements: L.elements + 1 };
    }
    if (!L.live) return { ...x, branch: 'inactive' };
    if (c != null) {
      const e = elementIndex(p, block, L.thread, c, blockDim, coarse);
      if (e >= n) return { ...x, branch: 'inactive' };
      const v = img[e];
      elements++;
      if (v === L.b) {
        sawExtend = true;
        return { ...x, element: e, value: v, branch: 'extend', binR: L.binR + 1, elements: L.elements + 1 };
      }
      sawFlush = true;
      counts[L.b!] += L.binR;
      atomics++;
      return { ...x, element: e, value: v, branch: 'flush', amount: L.binR, bin: L.b, b: v, binR: 1, atomics: L.atomics + 1, elements: L.elements + 1 };
    }
    // After the loop: flush whatever run is in the registers.
    counts[L.b!] += L.binR;
    atomics++;
    return { ...x, branch: 'final', amount: L.binR, bin: L.b, atomics: L.atomics + 1 };
  });
  const divergent = sawExtend && sawFlush;
  return {
    cfg: s.cfg,
    step: k,
    c,
    lanes,
    divergent,
    atomics,
    elements,
    divergentSteps: s.divergentSteps + (divergent ? 1 : 0),
    counts,
    done: k === coarse + 1,
  };
}

/** Every state from nothing run to the final flush: trace[k] is the state after step k. */
export function trace(img: ArrayLike<number>, cfg: WarpConfig): WarpState[] {
  const out = [initial(cfg)];
  while (!out[out.length - 1].done) out.push(step(img, out[out.length - 1]));
  return out;
}

/** What the lanes of a warp read, bin by bin: the reference their flushes must add up to. */
export function warpReference(img: ArrayLike<number>, cfg: WarpConfig): number[] {
  const ref = new Array(BINS).fill(0);
  for (let l = 0; l < cfg.lanes; l++)
    for (let c = 0; c < cfg.coarse; c++) {
      const e = elementIndex(cfg.p, cfg.block, cfg.firstThread + l, c, cfg.blockDim, cfg.coarse);
      if (e < cfg.n) ref[img[e]]++;
    }
  return ref;
}

// ---------------------------------------------------------------- whole blocks and whole images

export interface RunTotals {
  /** Elements read; Fig. 9.14 issues one block-scope atomic for each. */
  elements: number;
  /** Fig. 9.15: one per run per thread, final flushes included. */
  atomics: number;
  /** (warp, loop iteration) pairs with at least one active lane, and those whose active lanes disagreed. */
  loopSteps: number;
  divergentSteps: number;
  /** The private copies' counts equal a plain count of the same elements. */
  match: boolean;
}

/** Fig. 9.15 over blocks [b0, b1) of an image, warps of `warpWidth` lanes. */
function runBlocks(img: ArrayLike<number>, p: Partitioning, blockDim: number, coarse: number, warpWidth: number, b0: number, b1: number): RunTotals {
  const n = img.length;
  const got = new Float64Array(BINS);
  const ref = new Float64Array(BINS);
  let elements = 0;
  let atomics = 0;
  let loopSteps = 0;
  let divergentSteps = 0;
  const W = Math.max(1, warpWidth);
  const bReg = new Int32Array(W);
  const rReg = new Int32Array(W);
  const live = new Uint8Array(W);
  for (let blk = b0; blk < b1; blk++)
    for (let w0 = 0; w0 < blockDim; w0 += W) {
      const lanes = Math.min(W, blockDim - w0);
      for (let l = 0; l < lanes; l++) {
        const e = elementIndex(p, blk, w0 + l, 0, blockDim, coarse);
        live[l] = e < n ? 1 : 0;
        if (live[l]) {
          bReg[l] = img[e];
          rReg[l] = 1;
          ref[img[e]]++;
          elements++;
        }
      }
      for (let c = 1; c < coarse; c++) {
        let any = false;
        let ext = false;
        let fl = false;
        for (let l = 0; l < lanes; l++) {
          if (!live[l]) continue;
          const e = elementIndex(p, blk, w0 + l, c, blockDim, coarse);
          if (e >= n) continue;
          any = true;
          const v = img[e];
          ref[v]++;
          elements++;
          if (v === bReg[l]) {
            rReg[l]++;
            ext = true;
          } else {
            got[bReg[l]] += rReg[l];
            atomics++;
            bReg[l] = v;
            rReg[l] = 1;
            fl = true;
          }
        }
        if (any) loopSteps++;
        if (ext && fl) divergentSteps++;
      }
      for (let l = 0; l < lanes; l++)
        if (live[l]) {
          got[bReg[l]] += rReg[l];
          atomics++;
        }
    }
  let match = true;
  for (let k = 0; k < BINS; k++) if (got[k] !== ref[k]) match = false;
  return { elements, atomics, loopSteps, divergentSteps, match };
}

/** Fig. 9.15 on one block. */
export function blockRun(img: ArrayLike<number>, p: Partitioning, blockDim: number, coarse: number, block: number, warpWidth: number): RunTotals {
  return runBlocks(img, p, blockDim, coarse, warpWidth, block, block + 1);
}

export interface Summary extends RunTotals {
  /** Fig. 9.15's atomics per element (Fig. 9.14's is 1). */
  perElement: number;
  /** Share of loop steps in which a warp split at the new branch; null when there is no loop (factor 1). */
  divergentShare: number | null;
}

/** Fig. 9.15 over a whole image: the "Atomics per element" table. */
export function summary(img: ArrayLike<number>, p: Partitioning, blockDim: number, coarse: number, warpWidth: number): Summary {
  const blocks = Math.ceil(img.length / (blockDim * coarse));
  const r = runBlocks(img, p, blockDim, coarse, warpWidth, 0, blocks);
  return {
    ...r,
    perElement: r.elements ? r.atomics / r.elements : 0,
    divergentShare: r.loopSteps ? r.divergentSteps / r.loopSteps : null,
  };
}
