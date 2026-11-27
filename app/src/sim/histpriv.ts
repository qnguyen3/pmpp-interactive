// Privatisation (section 9.4, pp. 211-214): how many threads aim at one counter, how
// much the commit costs, whether one private copy fits on chip, and a tiny privatised
// kernel run the way the course's emulator runs it, for the bug hunt.
//
// Three parts:
//   1. plan(): counts for the explorer "Copies and contenders". Work only, no time.
//   2. simulate() and races(): the bug hunt's kernels (labs/ch09/snippets/
//      privatised_variants.cu) on two blocks of four threads. simulate() hands a baton
//      from thread to thread exactly as the emu backend does (labs/common/include/pmpp/
//      emu/emu_runtime.h: blocks one after another, a thread runs until a barrier or its
//      end, the forward, reverse and seeded shuffle orders, shared memory filled with
//      0xCD bytes). races() is a happens-before check of the kind ThreadSanitizer runs
//      in the race backend: threads of one block are ordered only across a barrier,
//      blocks are never ordered, two atomics never race, and scopes are ignored as the
//      emulator ignores them. histpriv.test.ts runs the snippet file on both backends
//      and compares.
//   3. sceneData(): the numbers the narrated animation "Privatise, then commit" draws.
import { tr } from '../lib/locale.ts';
import { generateImage, rng, type ImageKind } from './histogram';

// ------------------------------------------------------------------ 1. the explorer
export type Scheme = 'public' | 'two' | 'four' | 'block';
export const SCHEMES: Scheme[] = ['public', 'two', 'four', 'block'];
export const SCHEME_LABEL: Record<Scheme, string> = {
  get public() { return tr('one public copy (Fig. 9.6)', 'một bản sao chung (Fig. 9.6)'); },
  get two() { return tr('two copies: even and odd blocks', 'hai bản sao: khối chẵn và khối lẻ'); },
  get four() { return tr('four copies: block i uses copy i mod 4', 'bốn bản sao: khối i dùng bản sao i mod 4'); },
  get block() { return tr('one copy per block (Figs. 9.9, 9.10)', 'mỗi khối một bản sao (Figs. 9.9, 9.10)'); },
};

/** Pixel levels of an 8-bit image (p. 202). */
export const LEVELS = 256;

/**
 * Seeded input values in 0..bins-1. Up to 256 bins they are pixels of section 9.1's
 * images (histogram.ts) binned by intervals of 256/bins values; above that they are
 * token ids: 'uniform' over the vocabulary, 'bright' log-uniformly skewed towards the
 * low ids (a few very frequent tokens), 'constant' one id repeated. Ours, not the book's.
 */
export function makeValues(kind: ImageKind, n: number, bins: number, seed = 1): number[] {
  if (!Number.isInteger(bins) || bins < 1) throw new Error(`bins ${bins}`);
  if (bins <= LEVELS) return generateImage(n, kind, seed).map((v) => Math.floor((v * bins) / LEVELS));
  if (kind === 'constant') return new Array<number>(n).fill(bins - 1);
  const r = rng(seed);
  if (kind === 'uniform') return Array.from({ length: n }, () => Math.floor(r() * bins));
  const lnB = Math.log(bins + 1);
  return Array.from({ length: n }, () => Math.min(bins - 1, Math.floor(Math.exp(r() * lnB)) - 1));
}

export interface Plan {
  n: number;
  bins: number;
  threadsPerBlock: number;
  coarse: number;
  blocks: number;
  scheme: Scheme;
  /** Copies of the histogram the updates go to (1 for the public scheme: the public copy itself). */
  copies: number;
  /** Bytes of one copy. */
  copyBytes: number;
  /** Bytes of all private copies (0 for the public scheme). */
  privateBytes: number;
  /** Atomics in the update phase: one per input element, whatever the scheme. */
  updates: number;
  /** The most updates aimed at one location of one copy: the longest queue of atomics. */
  updateContention: number;
  hottest: { copy: number; bin: number };
  /** Global atomics in the commit with the book's zero test: (copy, bin) pairs that are not zero. */
  commitAtomics: number;
  /** Without the zero test: every bin of every copy. */
  commitBound: number;
  /** The most copies that commit to one public bin. */
  commitContention: number;
  /** Global-memory atomics over the whole kernel when a per-block copy sits in shared memory. */
  globalAtomics: number;
  /** One copy fits the per-block on-chip budget. */
  fits: boolean;
  /** Where the copies live in this plan. */
  where: 'global' | 'shared';
}

export const blocksFor = (n: number, threadsPerBlock: number, coarse = 1) => Math.ceil(n / (threadsPerBlock * coarse));

export function copiesFor(scheme: Scheme, blocks: number): number {
  return scheme === 'public' ? 1 : scheme === 'two' ? Math.min(2, blocks) : scheme === 'four' ? Math.min(4, blocks) : blocks;
}
export function copyOf(scheme: Scheme, block: number): number {
  return scheme === 'public' ? 0 : scheme === 'two' ? block % 2 : scheme === 'four' ? block % 4 : block;
}

/**
 * Counts for one configuration. Element i belongs to block floor(i / (threadsPerBlock *
 * coarse)), which holds for both of section 9.5's partitionings. A thread is counted as a
 * contender for a location if it updates it at all, even if its block runs at another time.
 */
export function plan(
  values: readonly number[],
  threadsPerBlock: number,
  scheme: Scheme,
  coarse: number,
  bins: number,
  bytesPerBin: number,
  budget: number,
): Plan {
  const n = values.length;
  const per = threadsPerBlock * coarse;
  const blocks = Math.max(1, blocksFor(n, threadsPerBlock, coarse));
  const copies = copiesFor(scheme, blocks);
  // Count per (copy, bin) in one pass. With a copy per block, one block's counts at a time,
  // so no table of blocks x bins is ever built (a vocabulary would make it huge).
  const perBlock = scheme === 'block';
  const counts = new Uint32Array(perBlock ? bins : copies * bins);
  const copiesPerBin = new Uint32Array(bins);
  let touched: number[] = [];
  let updateContention = 0;
  let hottest = { copy: 0, bin: 0 };
  let commitAtomics = 0;
  const settle = (copy: number) => {
    for (const k of touched) {
      const bin = k % bins;
      const c = counts[k];
      if (c > updateContention) {
        updateContention = c;
        hottest = { copy: perBlock ? copy : Math.floor(k / bins), bin };
      }
      commitAtomics++;
      copiesPerBin[bin]++;
      if (perBlock) counts[k] = 0;
    }
    touched = [];
  };
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (!(v >= 0 && v < bins)) throw new Error(`value ${v} is not a bin of ${bins}`);
    const block = Math.floor(i / per);
    if (perBlock && i > 0 && i % per === 0) settle(block - 1);
    const k = perBlock ? v : copyOf(scheme, block) * bins + v;
    if (counts[k]++ === 0) touched.push(k);
  }
  settle(Math.max(0, blocks - 1));
  let commitContention = 0;
  for (let b = 0; b < bins; b++) commitContention = Math.max(commitContention, copiesPerBin[b]);
  const copyBytes = bins * bytesPerBin;
  const fits = copyBytes <= budget;
  const isPublic = scheme === 'public';
  return {
    n, bins, threadsPerBlock, coarse, blocks, scheme, copies, copyBytes,
    privateBytes: isPublic ? 0 : copies * copyBytes,
    updates: n,
    updateContention,
    hottest,
    commitAtomics: isPublic ? 0 : commitAtomics,
    commitBound: isPublic ? 0 : copies * bins,
    commitContention: isPublic ? 0 : commitContention,
    globalAtomics: isPublic ? n : scheme === 'block' && fits ? commitAtomics : n + commitAtomics,
    fits,
    where: scheme === 'block' && fits ? 'shared' : 'global',
  };
}

// Exercise 5 (p. 220). Our answers: the book prints none.
/** 5a: the basic kernel (Fig. 9.6) does one global atomic per element. */
export const ex5a = (n: number) => n;
/** 5b and 5c: at most one global atomic per bin per block in the commit. */
export const ex5b = (n: number, bins: number, threadsPerBlock: number, coarse = 1) =>
  blocksFor(n, threadsPerBlock, coarse) * bins;

/** Vocabulary sizes from the models' public configs, read 2026-11-20 (notes/today/ch09-facts.md, D7). */
export const VOCABS = [
  { id: 'gpt2', name: 'GPT-2', config: 'openai-community/gpt2', bins: 50_257 },
  { id: 'qwen3', name: 'Qwen3-8B', config: 'Qwen/Qwen3-8B', bins: 151_936 }, // i18n-ignore: model name
  { id: 'gptoss', name: 'gpt-oss-20b', config: 'openai/gpt-oss-20b', bins: 201_088 }, // i18n-ignore: model name
] as const;
export const KIB = 1024;
export const kib = (bytes: number) => bytes / KIB;

// ------------------------------------------------------------------ 2. the bug hunt
export const HUNT_BINS = 8;
export const HUNT_BLOCKS = 2;
export const HUNT_THREADS = 4;
/** The eight pixels of the hunt, already bin numbers. Block 0 gets the first four. */
export const HUNT_PIXELS: readonly number[] = [1, 1, 6, 3, 1, 5, 6, 2];
/** What the emulator leaves in shared memory that nobody wrote: four 0xCD bytes. */
export const GARBAGE = 0xcdcdcdcd;
export const HUNT_FILE = 'ch09/snippets/privatised_variants.cu';

export type Variant =
  | 'correct'
  | 'no-zeroing'
  | 'no-barrier-after-zeroing'
  | 'no-barrier-before-commit'
  | 'plain-shared-update'
  | 'plain-commit'
  | 'block-scope-commit'
  | 'commit-bound-blockDim';
export const VARIANTS: Variant[] = [
  'correct', 'no-zeroing', 'no-barrier-after-zeroing', 'no-barrier-before-commit',
  'plain-shared-update', 'plain-commit', 'block-scope-commit', 'commit-bound-blockDim',
];
/** The broken ones, in the order the hunt shows them. */
export const BROKEN: Variant[] = VARIANTS.slice(1);

export interface Shape {
  zero: boolean;
  barrier1: boolean;
  sharedAtomic: boolean;
  barrier2: boolean;
  commitAtomic: boolean;
  commitScope: 'device' | 'block';
  commitBound: 'bins' | 'blockDim';
}
const CORRECT: Shape = {
  zero: true, barrier1: true, sharedAtomic: true, barrier2: true, commitAtomic: true, commitScope: 'device', commitBound: 'bins',
};

/**
 * Each variant: its kernel in the snippet file, how it differs from the correct one, and
 * what a run on a GPU can be expected to show. `gpu` is our reasoning from the vendor
 * documents in notes/today/ch09-facts.md (A2, A3, A10), not a measurement.
 */
export const VARIANT: Record<Variant, { kernel: string; label: string; shape: Shape; gpu: string }> = {
  correct: {
    kernel: 'histPrivatised', get label() { return tr('correct (the reference)', 'đúng (bản tham chiếu)'); }, shape: CORRECT,
    get gpu() { return tr('Correct.', 'Đúng.'); },
  },
  'no-zeroing': {
    kernel: 'histNoZeroing', label: 'A', shape: { ...CORRECT, zero: false },
    get gpu() {
      return tr('Shared memory holds whatever was there before, so the counts start from leftovers. Usually visible, but a run that happens to find zeros passes.', 'Bộ nhớ chia sẻ giữ bất cứ thứ gì có sẵn ở đó, nên các số đếm bắt đầu từ giá trị còn sót lại. Thường thấy được, nhưng một lần chạy tình cờ gặp toàn số 0 thì vẫn qua.');
    },
  },
  'no-barrier-after-zeroing': {
    kernel: 'histNoBarrierAfterZeroing', label: 'B', shape: { ...CORRECT, barrier1: false },
    get gpu() {
      return tr('Wrong only when a thread counts before another thread has zeroed that bin, which depends on how its warps happen to run. Intermittent.', 'Chỉ sai khi một luồng đếm trước khi một luồng khác kịp đặt bin đó về 0, điều này phụ thuộc vào cách các warp của nó tình cờ chạy. Lúc có lúc không.');
    },
  },
  'no-barrier-before-commit': {
    kernel: 'histNoBarrierBeforeCommit', label: 'C', shape: { ...CORRECT, barrier2: false },
    get gpu() {
      return tr('Wrong only when a bin is committed before another warp has counted into it. Intermittent.', 'Chỉ sai khi một bin được commit trước khi một warp khác kịp đếm vào nó. Lúc có lúc không.');
    },
  },
  'plain-shared-update': {
    kernel: 'histPlainSharedUpdate', label: 'D', shape: { ...CORRECT, sharedAtomic: false },
    get gpu() {
      return tr('Loses counts when threads of a block hit the same bin at once, so skewed images show it most. On CUDA, compute-sanitizer --tool racecheck examines shared memory and is the GPU-side tool for this one.', 'Mất số đếm khi các luồng của một khối cùng lúc trúng một bin, nên ảnh bị lệch thấy rõ nhất. Trên CUDA, compute-sanitizer --tool racecheck kiểm tra bộ nhớ chia sẻ và là công cụ phía GPU cho trường hợp này.');
    },
  },
  'plain-commit': {
    kernel: 'histPlainCommit', label: 'E', shape: { ...CORRECT, commitAtomic: false },
    get gpu() {
      return tr('Loses a block\'s whole count for a bin when two blocks commit to it at once. Intermittent, and racecheck cannot see it: it examines shared memory only.', 'Mất toàn bộ số đếm của một khối cho một bin khi hai khối cùng lúc commit vào bin đó. Lúc có lúc không, và racecheck không thấy được: nó chỉ kiểm tra bộ nhớ chia sẻ.');
    },
  },
  'block-scope-commit': {
    kernel: 'histBlockScopeCommit', label: 'F', shape: { ...CORRECT, commitScope: 'block' },
    get gpu() {
      return tr('A data race and undefined behaviour under CUDA\'s memory model, whatever numbers it prints. No document says whether a GPU shows it; it may pass every run you ever make.', 'Một race condition (tranh chấp dữ liệu) và hành vi không xác định theo mô hình bộ nhớ của CUDA, dù nó in ra số nào. Không tài liệu nào nói GPU có để lộ lỗi này hay không; nó có thể qua mọi lần chạy bạn từng thực hiện.');
    },
  },
  'commit-bound-blockDim': {
    kernel: 'histCommitBoundBlockDim', label: 'G', shape: { ...CORRECT, commitBound: 'blockDim' },
    get gpu() {
      return tr('Wrong on every run on every device whenever there are more bins than threads in a block: the high bins are never committed.', 'Sai ở mọi lần chạy trên mọi thiết bị mỗi khi số bin nhiều hơn số luồng trong một khối: các bin cao không bao giờ được commit.');
    },
  },
};

export type Schedule = 'forward' | 'reverse' | 'shuffle';
export const SCHEDULES: Schedule[] = ['forward', 'reverse', 'shuffle'];
export const SCHEDULE_LABEL: Record<Schedule, string> = {
  get forward() { return tr('in order', 'theo thứ tự'); },
  get reverse() { return tr('reversed', 'đảo ngược'); },
  get shuffle() { return tr('shuffled', 'xáo trộn'); },
};

/**
 * The emulator's thread order for each block of one launch: the identity, its reverse,
 * or a Fisher-Yates shuffle that continues from one block to the next, with the seed the
 * labs use (7, labs/common/src/backend_cudalike.cpp) and the generator of emu_runtime.h.
 */
export function emuOrders(schedule: Schedule, blocks: number, threads: number, seed = 7): number[][] {
  const order = Array.from({ length: threads }, (_, i) => i);
  let r = (Math.imul(seed, 2654435761) + 12345) >>> 0;
  const out: number[][] = [];
  for (let blk = 0; blk < blocks; blk++) {
    if (schedule === 'reverse') for (let i = 0; i < threads; i++) order[i] = threads - 1 - i;
    else if (schedule === 'shuffle')
      for (let i = threads - 1; i > 0; i--) {
        r = (Math.imul(r, 1664525) + 1013904223) >>> 0;
        const j = (r >>> 8) % (i + 1);
        [order[i], order[j]] = [order[j], order[i]];
      }
    out.push([...order]);
  }
  return out;
}

export type Site = 'zero' | 'update' | 'commit-read' | 'commit-add';
export const SITE_TEXT: Record<Site, string> = {
  get zero() { return tr('the zeroing store', 'lệnh ghi đặt về 0'); },
  get update() { return tr('the update of the private copy', 'lần cập nhật bản sao riêng'); },
  get 'commit-read'() { return tr('the read of the private count', 'lần đọc số đếm riêng'); },
  get 'commit-add'() { return tr('the add to the public bin', 'phép cộng vào bin chung'); },
};

type Op =
  | { k: 'zero'; b: number }
  | { k: 'barrier' }
  | { k: 'update'; b: number }
  | { k: 'commit'; b: number };

function program(shape: Shape, block: number, tid: number, pixels: readonly number[]): Op[] {
  const ops: Op[] = [];
  if (shape.zero) for (let b = tid; b < HUNT_BINS; b += HUNT_THREADS) ops.push({ k: 'zero', b });
  if (shape.barrier1) ops.push({ k: 'barrier' });
  const i = block * HUNT_THREADS + tid;
  if (i < pixels.length) ops.push({ k: 'update', b: pixels[i] });
  if (shape.barrier2) ops.push({ k: 'barrier' });
  const bound = shape.commitBound === 'bins' ? HUNT_BINS : HUNT_THREADS;
  for (let b = tid; b < bound; b += HUNT_THREADS) ops.push({ k: 'commit', b });
  return ops;
}

export interface Access {
  block: number;
  thread: number;
  /** Barriers this thread had passed before the access. */
  epoch: number;
  space: 'shared' | 'global';
  addr: number;
  write: boolean;
  atomic: boolean;
  site: Site;
}

export interface Run {
  bins: number[];
  expected: number[];
  /** Bins that differ from the expected histogram. */
  wrong: number[];
  correct: boolean;
  /** Every access in the order it happened. */
  accesses: Access[];
}

export function expectedBins(pixels: readonly number[] = HUNT_PIXELS): number[] {
  const out = new Array<number>(HUNT_BINS).fill(0);
  for (const p of pixels) out[p] += 1;
  return out;
}

/** One launch of a variant on the emu backend's baton scheduler. */
export function simulate(variant: Variant, schedule: Schedule, pixels: readonly number[] = HUNT_PIXELS): Run {
  if (pixels.length !== HUNT_BLOCKS * HUNT_THREADS) throw new Error(`the hunt runs ${HUNT_BLOCKS * HUNT_THREADS} pixels`);
  const shape = VARIANT[variant].shape;
  const bins = new Array<number>(HUNT_BINS).fill(0); // the host zeroes the public bins
  const orders = emuOrders(schedule, HUNT_BLOCKS, HUNT_THREADS);
  const accesses: Access[] = [];
  for (let blk = 0; blk < HUNT_BLOCKS; blk++) {
    const sh = new Array<number>(HUNT_BINS).fill(GARBAGE);
    const progs = Array.from({ length: HUNT_THREADS }, (_, t) => program(shape, blk, t, pixels));
    const pc = new Array<number>(HUNT_THREADS).fill(0);
    const epoch = new Array<number>(HUNT_THREADS).fill(0);
    const waiting = new Array<boolean>(HUNT_THREADS).fill(false);
    const done = (t: number) => pc[t] >= progs[t].length;
    for (;;) {
      for (const t of orders[blk]) {
        if (done(t) || waiting[t]) continue;
        // Run t until it reaches a barrier or returns.
        while (!done(t)) {
          const op = progs[t][pc[t]];
          if (op.k === 'barrier') {
            waiting[t] = true;
            break;
          }
          const at = { block: blk, thread: t, epoch: epoch[t] };
          if (op.k === 'zero') {
            sh[op.b] = 0;
            accesses.push({ ...at, space: 'shared', addr: op.b, write: true, atomic: false, site: 'zero' });
          } else if (op.k === 'update') {
            sh[op.b] = (sh[op.b] + 1) >>> 0;
            accesses.push({ ...at, space: 'shared', addr: op.b, write: true, atomic: shape.sharedAtomic, site: 'update' });
          } else {
            const count = sh[op.b];
            accesses.push({ ...at, space: 'shared', addr: op.b, write: false, atomic: false, site: 'commit-read' });
            if (count > 0) {
              bins[op.b] = (bins[op.b] + count) >>> 0;
              accesses.push({ ...at, space: 'global', addr: op.b, write: true, atomic: shape.commitAtomic, site: 'commit-add' });
            }
          }
          pc[t]++;
        }
      }
      if (pc.every((_, t) => done(t))) break;
      // Every thread still alive is waiting at the barrier: release it.
      for (let t = 0; t < HUNT_THREADS; t++)
        if (waiting[t]) {
          waiting[t] = false;
          epoch[t]++;
          pc[t]++;
        }
    }
  }
  const expected = expectedBins(pixels);
  const wrong = bins.flatMap((v, b) => (v === expected[b] ? [] : [b]));
  return { bins, expected, wrong, correct: wrong.length === 0, accesses };
}

export interface Race {
  /** The two racing sites, in a fixed order. */
  sites: [Site, Site];
  space: 'shared' | 'global';
  /** One concrete instance. */
  example: { addr: number; a: { block: number; thread: number }; b: { block: number; thread: number } };
}

const SITE_ORDER: Site[] = ['zero', 'update', 'commit-read', 'commit-add'];

/**
 * Pairs of sites that race: same location, at least one write, not both atomic, and not
 * ordered by happens-before (same thread, or same block with a barrier between them).
 * Thread scopes are ignored, as the emulator ignores them. The accesses come from the
 * in-order run; which commits happen does not depend on the order wherever it matters here.
 */
export function races(variant: Variant, pixels: readonly number[] = HUNT_PIXELS): Race[] {
  const acc = simulate(variant, 'forward', pixels).accesses;
  const found = new Map<string, Race>();
  for (let x = 0; x < acc.length; x++)
    for (let y = x + 1; y < acc.length; y++) {
      const a = acc[x];
      const b = acc[y];
      if (a.space !== b.space || a.addr !== b.addr) continue;
      if (a.space === 'shared' && a.block !== b.block) continue; // each block has its own copy
      if (!a.write && !b.write) continue;
      if (a.atomic && b.atomic) continue;
      const ordered = a.block === b.block && (a.thread === b.thread || a.epoch !== b.epoch);
      if (ordered) continue;
      const sites = [a.site, b.site].sort((p, q) => SITE_ORDER.indexOf(p) - SITE_ORDER.indexOf(q)) as [Site, Site];
      const key = sites.join('|');
      if (!found.has(key))
        found.set(key, {
          sites, space: a.space,
          example: { addr: a.addr, a: { block: a.block, thread: a.thread }, b: { block: b.block, thread: b.thread } },
        });
    }
  return [...found.values()];
}

/** Which tools reveal a variant: the emu column (any order wrong) and the race column (any race). */
export function caughtBy(variant: Variant, pixels: readonly number[] = HUNT_PIXELS): { emu: boolean; race: boolean } {
  return {
    emu: SCHEDULES.some((s) => !simulate(variant, s, pixels).correct),
    race: races(variant, pixels).length > 0,
  };
}

/** The line of a kernel's source that holds a site (0-based within the kernel text), or -1. */
export function siteLine(kernelText: string, site: Site): number {
  const pattern: Record<Site, RegExp> = {
    zero: /bins_s\[b\]\s*=\s*0;/,
    update: /fetch_add\(1,|\+\+bins_s\[/,
    'commit-read': /=\s*bins_s\[b\];/,
    'commit-add': /fetch_add\(count,|bins\[b\]\s*\+=/,
  };
  return kernelText.split('\n').findIndex((l) => pattern[site].test(l));
}

export interface Diff {
  removed: string[];
  added: string[];
  /** 1-based line of each removed line in `before` (the header is line 1). */
  removedAt: number[];
  /** 1-based line of each added line in `after`. */
  addedAt: number[];
}

/** A line diff (longest common subsequence) between two kernels, ignoring the header line. */
export function diffLines(before: string, after: string): Diff {
  const a = before.split('\n').slice(1);
  const b = after.split('\n').slice(1);
  const L = Array.from({ length: a.length + 1 }, () => new Array<number>(b.length + 1).fill(0));
  for (let i = a.length - 1; i >= 0; i--)
    for (let j = b.length - 1; j >= 0; j--) L[i][j] = a[i] === b[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const d: Diff = { removed: [], added: [], removedAt: [], addedAt: [] };
  let i = 0;
  let j = 0;
  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) {
      i++;
      j++;
    } else if (j < b.length && (i === a.length || L[i][j + 1] >= L[i + 1][j])) {
      d.added.push(b[j].trim());
      d.addedAt.push(j + 2);
      j++;
    } else {
      d.removed.push(a[i].trim());
      d.removedAt.push(i + 2);
      i++;
    }
  }
  return d;
}

/**
 * A marked region of a lab file: the lines strictly between a line naming the region
 * (`// region: pool`, `// region pool`, `// #region pool`, `// [pool]`, `// BEGIN pool`) and
 * the next end marker (`// endregion`, `// end region`, `// #endregion`, `// [/pool]`,
 * `// END pool`). Returns null when there is no such region.
 */
export function region(source: string, name: string): { text: string; firstLine: number; lastLine: number } | null {
  const lines = source.replace(/\r\n/g, '\n').split('\n');
  const esc = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const start = new RegExp(`^\\s*//\\s*(?:#?region\\s*:?\\s*${esc}\\b|\\[${esc}\\]|BEGIN\\s*:?\\s*${esc}\\b)`, 'i');
  const end = new RegExp(`^\\s*//\\s*(?:#?end\\s*region\\b|\\[/${esc}\\]|END\\s*:?\\s*${esc}\\b)`, 'i');
  const s = lines.findIndex((l) => start.test(l));
  if (s < 0) return null;
  let e = s + 1;
  while (e < lines.length && !end.test(lines[e])) e++;
  if (e >= lines.length) return null;
  const body = lines.slice(s + 1, e);
  while (body.length && body[body.length - 1].trim() === '') body.pop();
  return { text: body.join('\n'), firstLine: s + 2, lastLine: s + 1 + body.length };
}

// ------------------------------------------------------------------ 3. the narrated scene
/** 24 pixels of our own, already bin numbers 0 to 7: a bright strip, mostly sky (bin 7). */
export const SCENE_PIXELS: readonly number[] = [
  7, 7, 3, 7, 7, 1, 7, 6,
  7, 6, 7, 3, 3, 7, 6, 7,
  2, 7, 7, 6, 7, 7, 0, 7,
];
export const SCENE_BINS = 8;
export const SCENE_TPB = 8;

export function sceneData() {
  const blocks = SCENE_PIXELS.length / SCENE_TPB;
  const before = plan(SCENE_PIXELS, SCENE_TPB, 'public', 1, SCENE_BINS, 4, Infinity);
  const after = plan(SCENE_PIXELS, SCENE_TPB, 'block', 1, SCENE_BINS, 4, Infinity);
  const perBlock = Array.from({ length: blocks }, (_, k) => {
    const c = new Array<number>(SCENE_BINS).fill(0);
    for (let t = 0; t < SCENE_TPB; t++) c[SCENE_PIXELS[k * SCENE_TPB + t]] += 1;
    return c;
  });
  const total = new Array<number>(SCENE_BINS).fill(0);
  for (const p of SCENE_PIXELS) total[p] += 1;
  const busiest = before.hottest.bin;
  // Commits in the order the scene draws them: block by block, bin by bin, zero bins skipped.
  const commits = perBlock.flatMap((c, block) => c.flatMap((count, bin) => (count > 0 ? [{ block, bin, count }] : [])));
  const committers = total.map((_, bin) => perBlock.filter((c) => c[bin] > 0).length);
  return {
    pixels: SCENE_PIXELS, blocks, tpb: SCENE_TPB, bins: SCENE_BINS,
    total, perBlock, busiest,
    /** Threads from all blocks aiming at the busiest public bin in the basic kernel. */
    publicContenders: before.updateContention,
    /** The busiest location of any block's private copy. */
    privateContenders: after.updateContention,
    perBlockBusiest: perBlock.map((c) => Math.max(...c)),
    commits,
    committers,
    commitContention: after.commitContention,
    globalBefore: before.globalAtomics,
    globalAfter: after.commitAtomics,
    zeroingStores: blocks * SCENE_BINS,
  };
}
