// A thread-level model of one block of the z-marching stencil kernels of chapter 8:
// Fig. 8.10 (three planes in shared memory) and Fig. 8.12 (registers for the planes
// before and after, one plane in shared memory), both corrected (see notes/ch08.md,
// "Unclear in the PDF"). Small enough to step through by hand.
//
// What it models: every thread of one t x t block as its own sequence of source-level
// operations (load the plane before, load the current plane, then per output plane:
// load the next plane, barrier 1, compute and store if active, barrier 2, shift the
// window), shared-memory slots with the global plane z each one holds and the
// iteration that wrote it, each thread's registers, global arrays in (z, y, x)
// row-major order, and several ways of interleaving the threads (as sim/tiled.ts).
//
// What it leaves out, and what the app says wherever this model is shown:
//   - Real hardware interleaves at the instruction level and runs threads in warps;
//     here a thread's unit of progress is one whole source-level operation.
//   - "Lockstep" means every thread finishes an operation before any thread starts the
//     next. No real block guarantees that: not across warps, and on NVIDIA since
//     compute capability 7.0 not within one either (notes/today/ch04-facts.md A10).
//   - Uninitialised shared memory is NaN (the worst case). Registers start at 0, as
//     in the corrected kernel, which initialises them.
//   - No timing, no caches, no memory bandwidth.

import { tr as trText } from '../lib/locale.ts';

export type Kernel = 'planes' | 'registers';
/** Which coefficient multiplies which y neighbour: Figs. 8.6 and 8.8, or as printed in Figs. 8.10 and 8.12. */
export type Pairing = 'book-8.6' | 'as-printed-8.10';
export type Schedule = 'lockstep' | 'in-order' | 'reversed' | 'shuffled';

export const SCHEDULE_LABEL: Record<Schedule, string> = {
  get lockstep() {
    return trText('all threads in lockstep', 'mọi luồng đi đồng bộ từng bước (lockstep)');
  },
  get 'in-order'() {
    return trText('each thread runs as far as it can, thread 0 first', 'mỗi luồng chạy xa nhất có thể, luồng 0 trước');
  },
  get reversed() {
    return trText('each thread runs as far as it can, last thread first', 'mỗi luồng chạy xa nhất có thể, luồng cuối trước');
  },
  get shuffled() {
    return trText('random interleaving', 'xen kẽ ngẫu nhiên');
  },
};

export interface ZConfig {
  /** The grid is N x N x N. */
  N: number;
  /** IN_TILE_DIM: the block is inTile x inTile threads; outputs per plane are (inTile - 2)^2. */
  inTile: number;
  /** Output planes per block (OUT_TILE_DIM in z in the book). Defaults to inTile - 2, as coded. */
  depth?: number;
  kernel: Kernel;
  /** Barrier after loading the next plane (protects the read-after-write dependence). */
  barrier1: boolean;
  /** Barrier before shifting the window (protects the write-after-read dependence). */
  barrier2: boolean;
  pairing: Pairing;
}

export const CORRECT = { barrier1: true, barrier2: true, pairing: 'book-8.6' as Pairing };

/** c0..c6: small, distinct where it matters (c1 != c2, c3 != c4, c5 != c6), so a swapped pair shows. */
export const COEFFS = [2, 1, 3, 4, 1, 3, 2] as const;

export const outWidth = (cfg: ZConfig) => cfg.inTile - 2;
export const depthOf = (cfg: ZConfig) => cfg.depth ?? cfg.inTile - 2;
/** Blocks per dimension: x and y step by the output width, z by the slab depth. */
export const gridBlocks = (cfg: ZConfig) => ({
  xy: Math.ceil(cfg.N / outWidth(cfg)),
  z: Math.ceil(cfg.N / depthOf(cfg)),
});

/** A block's input planes numbered from 0 at the plane before its first output plane, as Fig. 8.9 draws the tile. */
export const localZ = (iStart: number, z: number) => z - (iStart - 1);

/** Linear index of (z, y, x) in an N^3 array: the book's i*N*N + j*N + k. */
export const idx3 = (N: number, z: number, j: number, k: number) => (z * N + j) * N + k;

/** Small integers whose neighbours always differ in every direction. */
export const inputValue = (z: number, j: number, k: number) => ((2 * z + 5 * j + 3 * k + 1) % 9) + 1;

export interface ZProblem {
  N: number;
  input: number[];
  /** The CPU 7-point sweep (Fig. 8.6's pairing) at interior points; NaN on the boundary. */
  ref: number[];
  /** Linear indices of the (N - 2)^3 interior points. */
  interior: number[];
}

/** The CPU reference: one 7-point sweep over the interior, coefficients as in Figs. 8.6 and 8.8. */
export function reference(N: number, input: number[]): number[] {
  const [c0, c1, c2, c3, c4, c5, c6] = COEFFS;
  const out = new Array(N * N * N).fill(NaN);
  const at = (z: number, j: number, k: number) => input[idx3(N, z, j, k)];
  for (let z = 1; z < N - 1; z++)
    for (let j = 1; j < N - 1; j++)
      for (let k = 1; k < N - 1; k++)
        out[idx3(N, z, j, k)] =
          c0 * at(z, j, k) +
          c1 * at(z, j, k - 1) + c2 * at(z, j, k + 1) +
          c3 * at(z, j - 1, k) + c4 * at(z, j + 1, k) +
          c5 * at(z - 1, j, k) + c6 * at(z + 1, j, k);
  return out;
}

export function makeProblem(N: number): ZProblem {
  const input: number[] = [];
  for (let z = 0; z < N; z++) for (let j = 0; j < N; j++) for (let k = 0; k < N; k++) input.push(inputValue(z, j, k));
  const interior: number[] = [];
  for (let z = 1; z < N - 1; z++) for (let j = 1; j < N - 1; j++) for (let k = 1; k < N - 1; k++) interior.push(idx3(N, z, j, k));
  return { N, input, ref: reference(N, input), interior };
}

// ---------------------------------------------------------------- operations and steps

export type SharedBuf = 'inPrev_s' | 'inCurr_s' | 'inNext_s';
export type RegBuf = 'inPrev' | 'inCurr' | 'inNext';
export type Buf = SharedBuf | RegBuf;
export const isShared = (b: Buf): b is SharedBuf => b.endsWith('_s');

export type ZOp =
  | 'loadPrev'
  | 'loadCurr'
  | 'loadNext'
  | 'barrier1'
  | 'barrier2'
  | 'release1'
  | 'release2'
  | 'compute'
  | 'shiftPrev' // planes: inPrev_s <- inCurr_s
  | 'shiftCurr' // planes: inCurr_s <- inNext_s
  | 'rotate' // registers: inPrev = inCurr; inCurr = inNext
  | 'storeCurr'; // registers: inCurr_s <- inNext

/** What one operand of the stencil is, in the book's order c0..c6. */
export type Role = 'centre' | 'x-1' | 'x+1' | 'y-1' | 'y+1' | 'z-1' | 'z+1';
export const ROLES: Role[] = ['centre', 'x-1', 'x+1', 'y-1', 'y+1', 'z-1', 'z+1'];

export interface Slot {
  value: number;
  /** The global plane whose value this holds, or null if never written. */
  z: number | null;
  /** The loop iteration in which it was written (-1 before the loop), or null. */
  iter: number | null;
}

export interface ZRead {
  buf: Buf;
  r: number;
  c: number;
  value: number;
  z: number | null;
  /** The plane this read needs. */
  need: number;
  ok: boolean;
  role: Role | 'copy';
}
export interface ZWrite {
  buf: Buf;
  r: number;
  c: number;
  value: number;
  z: number | null;
}
export interface Term {
  /** Coefficient index 0..6 used for this operand. */
  coeff: number;
  read: ZRead;
}

export type ViolationKind = 'read-before-write' | 'overwritten-before-read';
export interface ZViolation {
  kind: ViolationKind;
  step: number;
  thread: number;
  message: string;
}

export interface ZStep {
  index: number;
  /** Thread index ty*inTile + tx, or -1 for a barrier opening (which involves all of them). */
  thread: number;
  tx: number;
  ty: number;
  /** Loop iteration, -1 before the loop. */
  iter: number;
  /** The output plane z this iteration computes (iStart + iter); for the loads before the loop, iStart. */
  i: number;
  op: ZOp;
  global?: { array: 'in' | 'out'; z: number; j: number; k: number; index: number; access: 'ok' | 'skipped' };
  reads?: ZRead[];
  writes?: ZWrite[];
  /** compute: the thread passed both guards. */
  active?: boolean;
  /** compute: which guard stopped it. */
  why?: 'halo' | 'boundary';
  terms?: Term[];
  value?: number;
  /** The thread's last operation. */
  final?: boolean;
  violation?: ZViolation;
}

export interface Counters {
  /** Elements read from global memory. */
  globalLoads: number;
  /** Outputs written to global memory. */
  globalStores: number;
  /** Shared-memory reads made by the stencil sum. */
  stencilSharedReads: number;
  /** Shared-memory reads made by shifting the window (planes kernel only). */
  copySharedReads: number;
  /** Shared-memory writes made by shifting the window (both kernels). */
  copySharedWrites: number;
  /** Shared-memory writes made while loading from global memory. */
  loadSharedWrites: number;
  /** Register-to-register moves of window values (registers kernel only). */
  registerMoves: number;
}
export const zeroCounters = (): Counters => ({
  globalLoads: 0,
  globalStores: 0,
  stencilSharedReads: 0,
  copySharedReads: 0,
  copySharedWrites: 0,
  loadSharedWrites: 0,
  registerMoves: 0,
});

export interface ZTrace {
  block: { bx: number; by: number; bz: number };
  iStart: number;
  steps: ZStep[];
  violations: ZViolation[];
  stores: { index: number; value: number; thread: number }[];
  counters: Counters;
}

type PlanOp = { op: Exclude<ZOp, 'release1' | 'release2'>; iter: number };

/** The operation list, identical for every thread of the block. */
export function plan(cfg: ZConfig): PlanOp[] {
  const p: PlanOp[] = [
    { op: 'loadPrev', iter: -1 },
    { op: 'loadCurr', iter: -1 },
  ];
  for (let n = 0; n < depthOf(cfg); n++) {
    p.push({ op: 'loadNext', iter: n });
    if (cfg.barrier1) p.push({ op: 'barrier1', iter: n });
    p.push({ op: 'compute', iter: n });
    if (cfg.barrier2) p.push({ op: 'barrier2', iter: n });
    if (cfg.kernel === 'planes') p.push({ op: 'shiftPrev', iter: n }, { op: 'shiftCurr', iter: n });
    else p.push({ op: 'rotate', iter: n }, { op: 'storeCurr', iter: n });
  }
  return p;
}

function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const threadName = (cfg: ZConfig, t: number) =>
  trText(`thread (tx ${t % cfg.inTile}, ty ${Math.floor(t / cfg.inTile)})`, `luồng (tx ${t % cfg.inTile}, ty ${Math.floor(t / cfg.inTile)})`);

/** Where thread (tx, ty) of block (bx, by) sits in the grid's x-y plane: the book's j and k. */
export const position = (cfg: ZConfig, bx: number, by: number, tx: number, ty: number) => ({
  j: by * outWidth(cfg) + ty - 1,
  k: bx * outWidth(cfg) + tx - 1,
});

/** Runs one block under one schedule and records every operation. */
export function runBlock(cfg: ZConfig, prob: ZProblem, bx: number, by: number, bz: number, schedule: Schedule, seed = 1): ZTrace {
  const T = cfg.inTile;
  const N = cfg.N;
  const nT = T * T;
  const iStart = bz * depthOf(cfg);
  const ops = plan(cfg);
  const fresh = (): Slot[] => Array.from({ length: nT }, () => ({ value: NaN, z: null, iter: null }));
  const shared: Record<SharedBuf, Slot[]> = { inPrev_s: fresh(), inCurr_s: fresh(), inNext_s: fresh() };
  const regZero = (): Slot => ({ value: 0, z: null, iter: null });
  const regs = Array.from({ length: nT }, () => ({ inPrev: regZero(), inCurr: regZero(), inNext: regZero() }));
  const pc = new Array(nT).fill(0);
  const waiting = new Array(nT).fill(false);
  const steps: ZStep[] = [];
  const violations: ZViolation[] = [];
  const stores: ZTrace['stores'] = [];
  const counters = zeroCounters();
  const rand = mulberry(seed);
  const name = (t: number) => threadName(cfg, t);

  const flag = (s: ZStep, kind: ViolationKind, message: string) => {
    const v: ZViolation = { kind, step: s.index, thread: s.thread, message };
    if (!s.violation) s.violation = v;
    violations.push(v);
  };

  const exec = (t: number) => {
    const tx = t % T;
    const ty = Math.floor(t / T);
    const { j, k } = position(cfg, bx, by, tx, ty);
    const jk = j >= 0 && j < N && k >= 0 && k < N;
    const p = ops[pc[t]];
    const i = p.iter < 0 ? iStart : iStart + p.iter;
    const s: ZStep = { index: steps.length, thread: t, tx, ty, iter: p.iter, i, op: p.op };
    steps.push(s);
    const slot = (b: SharedBuf, r: number, c: number) => shared[b][r * T + c];
    const write = (b: Buf, value: number, z: number | null) => {
      const target = isShared(b) ? slot(b, ty, tx) : regs[t][b];
      target.value = value;
      target.z = z;
      target.iter = p.iter;
      (s.writes ??= []).push({ buf: b, r: ty, c: tx, value, z });
    };
    const load = (z: number, into: Buf, mirror?: SharedBuf) => {
      const ok = jk && z >= 0 && z < N;
      const index = idx3(N, z, j, k);
      s.global = { array: 'in', z, j, k, index, access: ok ? 'ok' : 'skipped' };
      if (!ok) return;
      const v = prob.input[index];
      counters.globalLoads++;
      write(into, v, z);
      if (isShared(into)) counters.loadSharedWrites++;
      if (mirror) {
        write(mirror, v, z);
        counters.loadSharedWrites++;
      }
    };

    switch (p.op) {
      case 'barrier1':
      case 'barrier2':
        waiting[t] = true;
        return;
      case 'loadPrev':
        load(iStart - 1, cfg.kernel === 'planes' ? 'inPrev_s' : 'inPrev');
        break;
      case 'loadCurr':
        if (cfg.kernel === 'planes') load(iStart, 'inCurr_s');
        else load(iStart, 'inCurr', 'inCurr_s');
        break;
      case 'loadNext':
        load(i + 1, cfg.kernel === 'planes' ? 'inNext_s' : 'inNext');
        break;
      case 'compute': {
        const inGrid = i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1;
        const inBlock = ty >= 1 && ty < T - 1 && tx >= 1 && tx < T - 1;
        s.active = inGrid && inBlock;
        if (!s.active) {
          s.why = inBlock ? 'boundary' : 'halo';
          break;
        }
        // Operands in the book's order c0..c6, with where each one lives in this kernel.
        const P = cfg.kernel === 'planes';
        const where: [Role, Buf, number, number, number][] = [
          ['centre', P ? 'inCurr_s' : 'inCurr', ty, tx, i],
          ['x-1', 'inCurr_s', ty, tx - 1, i],
          ['x+1', 'inCurr_s', ty, tx + 1, i],
          ['y-1', 'inCurr_s', ty - 1, tx, i],
          ['y+1', 'inCurr_s', ty + 1, tx, i],
          ['z-1', P ? 'inPrev_s' : 'inPrev', ty, tx, i - 1],
          ['z+1', P ? 'inNext_s' : 'inNext', ty, tx, i + 1],
        ];
        const reads: ZRead[] = where.map(([role, buf, r, c, need]) => {
          const src = isShared(buf) ? slot(buf, r, c) : regs[t][buf];
          if (isShared(buf)) counters.stencilSharedReads++;
          return { buf, r, c, value: src.value, z: src.z, need, ok: src.z === need, role };
        });
        s.reads = reads;
        for (const rd of reads) {
          if (!isShared(rd.buf) || rd.ok) continue;
          const at = `${rd.buf}[${rd.r}][${rd.c}]`;
          const owner = name(rd.r * T + rd.c);
          if (rd.z === null || rd.z < rd.need)
            flag(s, 'read-before-write',
              trText(`${name(t)} read ${at} for plane ${rd.need} before ${owner} had put plane ${rd.need} there`, `${name(t)} đọc ${at} để lấy mặt phẳng ${rd.need} trước khi ${owner} đặt mặt phẳng ${rd.need} vào đó`) +
                (rd.z === null ? trText(' (the slot was never written)', ' (ô này chưa từng được ghi)') : trText(` (it still held plane ${rd.z})`, ` (ô vẫn còn giữ mặt phẳng ${rd.z})`)));
          else
            flag(s, 'overwritten-before-read',
              trText(`${name(t)} still needed plane ${rd.need} from ${at}, but ${owner} had already replaced it with plane ${rd.z}`, `${name(t)} vẫn cần mặt phẳng ${rd.need} từ ${at}, nhưng ${owner} đã thay nó bằng mặt phẳng ${rd.z}`));
        }
        // Figs. 8.6 and 8.8 pair c3 with y-1 and c4 with y+1; Figs. 8.10 and 8.12 print the reverse.
        const coeffOf = (role: Role) => {
          const n = ROLES.indexOf(role);
          if (cfg.pairing === 'as-printed-8.10' && (n === 3 || n === 4)) return 7 - n;
          return n;
        };
        s.terms = reads.map((read) => ({ coeff: coeffOf(read.role as Role), read }));
        s.value = s.terms.reduce((acc, x) => acc + COEFFS[x.coeff] * x.read.value, 0);
        const index = idx3(N, i, j, k);
        s.global = { array: 'out', z: i, j, k, index, access: 'ok' };
        stores.push({ index, value: s.value, thread: t });
        counters.globalStores++;
        break;
      }
      case 'shiftPrev':
      case 'shiftCurr': {
        const from: SharedBuf = p.op === 'shiftPrev' ? 'inCurr_s' : 'inNext_s';
        const to: SharedBuf = p.op === 'shiftPrev' ? 'inPrev_s' : 'inCurr_s';
        const src = slot(from, ty, tx);
        const need = p.op === 'shiftPrev' ? i : i + 1;
        s.reads = [{ buf: from, r: ty, c: tx, value: src.value, z: src.z, need, ok: src.z === need, role: 'copy' }];
        counters.copySharedReads++;
        write(to, src.value, src.z);
        counters.copySharedWrites++;
        break;
      }
      case 'rotate': {
        const r = regs[t];
        s.reads = [
          { buf: 'inCurr', r: ty, c: tx, value: r.inCurr.value, z: r.inCurr.z, need: i, ok: r.inCurr.z === i, role: 'copy' },
          { buf: 'inNext', r: ty, c: tx, value: r.inNext.value, z: r.inNext.z, need: i + 1, ok: r.inNext.z === i + 1, role: 'copy' },
        ];
        const cur = { ...r.inCurr };
        const nxt = { ...r.inNext };
        write('inPrev', cur.value, cur.z);
        write('inCurr', nxt.value, nxt.z);
        counters.registerMoves += 2;
        break;
      }
      case 'storeCurr': {
        const r = regs[t];
        s.reads = [{ buf: 'inNext', r: ty, c: tx, value: r.inNext.value, z: r.inNext.z, need: i + 1, ok: r.inNext.z === i + 1, role: 'copy' }];
        write('inCurr_s', r.inNext.value, r.inNext.z);
        counters.copySharedWrites++;
        break;
      }
    }
    pc[t]++;
    if (pc[t] === ops.length) s.final = true;
  };

  const live = (t: number) => pc[t] < ops.length;
  const runnable = (t: number) => live(t) && !waiting[t];
  const release = () => {
    const anyWaiting = waiting.some(Boolean);
    const allThere = waiting.every((w, t) => w || !live(t));
    if (!anyWaiting || !allThere) return false;
    const t0 = waiting.findIndex(Boolean);
    const p = ops[pc[t0]];
    steps.push({
      index: steps.length, thread: -1, tx: -1, ty: -1, iter: p.iter, i: iStart + p.iter,
      op: p.op === 'barrier1' ? 'release1' : 'release2',
    });
    for (let t = 0; t < nT; t++)
      if (waiting[t]) {
        waiting[t] = false;
        pc[t]++;
        if (pc[t] === ops.length) steps[steps.length - 1].final = true;
      }
    return true;
  };

  const order = Array.from({ length: nT }, (_, n) => n);
  if (schedule === 'reversed') order.reverse();
  let guard = 0;
  while (order.some(live)) {
    if (++guard > 1e6) throw new Error('simulation did not terminate');
    if (schedule === 'lockstep') {
      for (const t of order) if (runnable(t)) exec(t);
    } else if (schedule === 'shuffled') {
      const cands = order.filter(runnable);
      if (cands.length) exec(cands[Math.floor(rand() * cands.length)]);
    } else {
      for (const t of order) while (runnable(t)) exec(t);
    }
    release();
  }
  return { block: { bx, by, bz }, iStart, steps, violations, stores, counters };
}

// ---------------------------------------------------------------- the whole sweep

export interface ZRunResult {
  out: number[];
  /** Interior indices whose output differs from the reference. */
  wrong: number[];
  violations: (ZViolation & { block: { bx: number; by: number; bz: number } })[];
  counters: Counters;
  correct: boolean;
}

/** Runs every block (one after another) and compares the interior of the output with the CPU sweep. */
export function runAll(cfg: ZConfig, prob: ZProblem, schedule: Schedule, seed = 1): ZRunResult {
  const g = gridBlocks(cfg);
  const out = new Array(cfg.N ** 3).fill(NaN);
  const violations: ZRunResult['violations'] = [];
  const counters = zeroCounters();
  let n = 0;
  for (let bz = 0; bz < g.z; bz++)
    for (let by = 0; by < g.xy; by++)
      for (let bx = 0; bx < g.xy; bx++) {
        const tr = runBlock(cfg, prob, bx, by, bz, schedule, seed + n++);
        for (const s of tr.stores) out[s.index] = s.value;
        for (const v of tr.violations) violations.push({ ...v, block: tr.block });
        for (const key of Object.keys(counters) as (keyof Counters)[]) counters[key] += tr.counters[key];
      }
  const wrong = prob.interior.filter((ix) => !(out[ix] === prob.ref[ix]));
  return { out, wrong, violations, counters, correct: wrong.length === 0 };
}

// ---------------------------------------------------------------- state after n steps

export interface ZState {
  shared: Record<SharedBuf, Slot[]>;
  regs: Record<RegBuf, Slot>[];
  /** Per thread: index of the last step it executed, or -1. */
  last: number[];
  waiting: boolean[];
  done: boolean[];
  stores: ZTrace['stores'];
  counters: Counters;
  /** Distinct input planes loaded so far. */
  planes: number[];
}

/** The state of a block after the first n steps of its trace, for the stepper. */
export function stateAt(cfg: ZConfig, trace: ZTrace, n: number): ZState {
  const nT = cfg.inTile * cfg.inTile;
  const fresh = (): Slot[] => Array.from({ length: nT }, () => ({ value: NaN, z: null, iter: null }));
  const reg = (): Slot => ({ value: 0, z: null, iter: null });
  const st: ZState = {
    shared: { inPrev_s: fresh(), inCurr_s: fresh(), inNext_s: fresh() },
    regs: Array.from({ length: nT }, () => ({ inPrev: reg(), inCurr: reg(), inNext: reg() })),
    last: new Array(nT).fill(-1),
    waiting: new Array(nT).fill(false),
    done: new Array(nT).fill(false),
    stores: [],
    counters: zeroCounters(),
    planes: [],
  };
  const planes = new Set<number>();
  for (let x = 0; x < Math.min(n, trace.steps.length); x++) {
    const s = trace.steps[x];
    if (s.thread < 0) {
      st.waiting.fill(false);
      if (s.final) st.done.fill(true);
      continue;
    }
    st.last[s.thread] = x;
    for (const w of s.writes ?? []) {
      const v: Slot = { value: w.value, z: w.z, iter: s.iter };
      if (isShared(w.buf)) st.shared[w.buf][w.r * cfg.inTile + w.c] = v;
      else st.regs[s.thread][w.buf] = v;
    }
    if (s.op === 'barrier1' || s.op === 'barrier2') st.waiting[s.thread] = true;
    if (s.final) st.done[s.thread] = true;
    const c = st.counters;
    if (s.global?.array === 'in' && s.global.access === 'ok') {
      c.globalLoads++;
      planes.add(s.global.z);
      c.loadSharedWrites += (s.writes ?? []).filter((w) => isShared(w.buf)).length;
    }
    if (s.op === 'compute' && s.active) {
      c.globalStores++;
      c.stencilSharedReads += (s.reads ?? []).filter((r) => isShared(r.buf)).length;
      st.stores.push({ index: s.global!.index, value: s.value!, thread: s.thread });
    }
    if (s.op === 'shiftPrev' || s.op === 'shiftCurr') {
      c.copySharedReads++;
      c.copySharedWrites++;
    }
    if (s.op === 'storeCurr') c.copySharedWrites++;
    if (s.op === 'rotate') c.registerMoves += 2;
  }
  st.planes = [...planes].sort((a, b) => a - b);
  return st;
}

// ---------------------------------------------------------------- a block that touches no edge

/** The first block whose loads and outputs are all inside the grid's interior, if there is one. */
export function interiorBlock(cfg: ZConfig): { bx: number; by: number; bz: number } {
  const T = cfg.inTile;
  const O = outWidth(cfg);
  const D = depthOf(cfg);
  const g = gridBlocks(cfg);
  const okXY = (b: number) => b * O - 1 >= 0 && b * O + T - 2 <= cfg.N - 1 && b * O >= 1 && b * O + O - 1 <= cfg.N - 2;
  const okZ = (b: number) => b * D - 1 >= 0 && b * D + D <= cfg.N - 1 && b * D >= 1 && b * D + D - 1 <= cfg.N - 2;
  for (let bz = 0; bz < g.z; bz++)
    for (let by = 0; by < g.xy; by++)
      for (let bx = 0; bx < g.xy; bx++) if (okXY(bx) && okXY(by) && okZ(bz)) return { bx, by, bz };
  return { bx: 0, by: 0, bz: 0 };
}

// ---------------------------------------------------------------- the march, for the scene

export interface PlaneFrame {
  /** start: the two planes loaded before the loop; then, per iteration, after barrier 1, after the compute, after the shift. */
  label: 'start' | 'loaded' | 'computed' | 'shifted';
  iter: number;
  /** Output plane of this iteration (global z). */
  i: number;
  /** Global z of the plane each part of the window holds, read from the block's centre thread. */
  hold: { prev: number | null; curr: number | null; next: number | null };
  /** Global z the shared copy of the current plane holds (the registers kernel keeps one). */
  sharedCurr: number | null;
  /** Threads (ty * inTile + tx) that computed an output in this iteration. */
  active: boolean[];
  /** Distinct input planes loaded so far. */
  planesLoaded: number;
  /** Elements loaded from global memory so far. */
  loads: number;
  /** Outputs stored so far. */
  stores: number;
  /** Output planes (global z) with at least one stored output. */
  outputPlanes: number[];
}

/**
 * Which plane each buffer holds, frame by frame, for one block marching in lockstep.
 * Every number the "Marching through z" scene draws comes from here.
 */
export function planeTimeline(cfg: ZConfig, prob: ZProblem, block = interiorBlock(cfg)) {
  const trace = runBlock(cfg, prob, block.bx, block.by, block.bz, 'lockstep');
  const T = cfg.inTile;
  const centre = Math.floor(T / 2) * T + Math.floor(T / 2);
  const after = (pred: (s: ZStep) => boolean) => {
    let last = -1;
    trace.steps.forEach((s, n) => { if (pred(s)) last = n; });
    return last + 1;
  };
  const frame = (label: PlaneFrame['label'], iter: number, n: number): PlaneFrame => {
    const st = stateAt(cfg, trace, n);
    const P = cfg.kernel === 'planes';
    const active = new Array(T * T).fill(false);
    if (label !== 'start' && label !== 'loaded')
      for (const s of trace.steps.slice(0, n)) if (s.op === 'compute' && s.active && s.iter === iter) active[s.thread] = true;
    return {
      label,
      iter,
      i: trace.iStart + Math.max(iter, 0),
      hold: {
        prev: P ? st.shared.inPrev_s[centre].z : st.regs[centre].inPrev.z,
        curr: P ? st.shared.inCurr_s[centre].z : st.regs[centre].inCurr.z,
        next: P ? st.shared.inNext_s[centre].z : st.regs[centre].inNext.z,
      },
      sharedCurr: st.shared.inCurr_s[centre].z,
      active,
      planesLoaded: st.planes.length,
      loads: st.counters.globalLoads,
      stores: st.counters.globalStores,
      outputPlanes: [...new Set(st.stores.map((x) => Math.floor(x.index / (cfg.N * cfg.N))))].sort((a, b) => a - b),
    };
  };
  const frames: PlaneFrame[] = [frame('start', -1, after((s) => s.op === 'loadCurr'))];
  const shiftOps: ZOp[] = cfg.kernel === 'planes' ? ['shiftPrev', 'shiftCurr'] : ['rotate', 'storeCurr'];
  for (let n = 0; n < depthOf(cfg); n++) {
    frames.push(frame('loaded', n, after((s) => s.iter === n && (s.op === 'loadNext' || s.op === 'release1'))));
    frames.push(frame('computed', n, after((s) => s.iter === n && s.op === 'compute')));
    frames.push(frame('shifted', n, after((s) => s.iter === n && shiftOps.includes(s.op))));
  }
  return { block, iStart: trace.iStart, frames, trace };
}
