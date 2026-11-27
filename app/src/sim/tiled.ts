// A thread-level model of the tiled matrix multiplication kernel of chapter 5
// (book Figs. 5.9 and 5.13), small enough to step through by hand.
//
// What it models: every thread of one block as its own sequence of operations
// (load an M element, load an N element, wait at a barrier, one multiply-add per
// k, wait again, finally store), shared-memory tiles with one slot per thread,
// global arrays in row-major order, and several ways of interleaving the threads.
//
// What it leaves out, and what the app says wherever this model is shown:
//   - Real hardware interleaves at the instruction level and runs threads in
//     warps; here a thread's unit of progress is one whole source-level operation.
//   - "Lockstep" is an idealised picture of a single warp. A real block has
//     several warps and no guarantee about their relative order.
//   - Uninitialised shared memory is treated as NaN (the worst case). On a real
//     device it holds whatever was there before, which is often harmless.
//   - No timing, no caches, no memory bandwidth.

import { tr as trText } from '../lib/locale.ts';

export interface Config {
  /** Square matrices of this width, row-major. */
  width: number;
  /** TILE_WIDTH. Blocks are tile x tile threads. */
  tile: number;
  /** Barrier after the two loads (protects the read-after-write dependence). */
  barrier1: boolean;
  /** Barrier after the partial dot product (protects the write-after-read dependence). */
  barrier2: boolean;
  /** Bounds check on the load from M. */
  guardM: boolean;
  /** Bounds check on the load from N. */
  guardN: boolean;
  /** Bounds check on the store to P. */
  guardP: boolean;
  /** When a guarded M load is out of range, write 0 into the slot. */
  zeroFillM: boolean;
  /** When a guarded N load is out of range, write 0 into the slot. */
  zeroFillN: boolean;
}

export const CORRECT: Omit<Config, 'width' | 'tile'> = {
  barrier1: true,
  barrier2: true,
  guardM: true,
  guardN: true,
  guardP: true,
  zeroFillM: true,
  zeroFillN: true,
};

/** How the threads of a block are interleaved. */
export type Schedule = 'lockstep' | 'in-order' | 'reversed' | 'shuffled';

export const SCHEDULE_LABEL: Record<Schedule, string> = {
  get lockstep() {
    return trText('all threads in lockstep', 'mọi luồng chạy đồng bộ (lockstep)');
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

export interface Problem {
  width: number;
  M: number[];
  N: number[];
  /** M * N computed directly. Inputs are small integers, so this is exact. */
  ref: number[];
}

export function makeProblem(width: number): Problem {
  const M: number[] = [];
  const N: number[] = [];
  for (let r = 0; r < width; r++)
    for (let c = 0; c < width; c++) {
      M.push(((r * 3 + c * 5 + 1) % 9) + 1);
      N.push(((r * 7 + c * 2 + 3) % 9) + 1);
    }
  const ref = new Array(width * width).fill(0);
  for (let r = 0; r < width; r++)
    for (let c = 0; c < width; c++)
      for (let k = 0; k < width; k++) ref[r * width + c] += M[r * width + k] * N[k * width + c];
  return { width, M, N, ref };
}

/** What a global-memory access actually touched. */
export type Access =
  | 'ok' // the intended element
  | 'wrap' // past the end of a row: a valid address holding the WRONG element
  | 'outside' // beyond the allocation: garbage, or a fault
  | 'skipped'; // a bounds check prevented the access

export type ViolationKind =
  | 'read-before-write' // a tile slot was read before its owner wrote this phase's value
  | 'overwritten-before-read' // a tile slot already holds a later phase's value
  | 'unfilled' // a guarded load left the slot untouched and it was then used
  | 'wrong-element' // an unchecked load fetched a different element than intended
  | 'outside-read' // an unchecked load reached beyond the array
  | 'bad-store'; // an unchecked store hit another element or left the array

export interface Violation {
  kind: ViolationKind;
  step: number;
  thread: number;
  message: string;
}

export type Op = 'loadM' | 'loadN' | 'barrier1' | 'barrier2' | 'release1' | 'release2' | 'mul' | 'store' | 'done';

export interface SlotRef {
  array: 'Mds' | 'Nds';
  r: number;
  c: number;
}

export interface Step {
  index: number;
  /** Thread index ty*tile+tx, or -1 for a barrier release (which involves all of them). */
  thread: number;
  tx: number;
  ty: number;
  ph: number;
  op: Op;
  /** k for a multiply-add. */
  k?: number;
  global?: { array: 'M' | 'N' | 'P'; row: number; col: number; index: number; access: Access };
  /** Slot written by a load. */
  write?: SlotRef & { value: number; phase: number; unfilled?: boolean };
  /** Slots read by a multiply-add, with the phase of the value found there. */
  reads?: (SlotRef & { value: number; phase: number; ok: boolean })[];
  /** The thread's running sum after this step. */
  pvalue?: number;
  violation?: Violation;
}

export interface BlockTrace {
  bx: number;
  by: number;
  steps: Step[];
  violations: Violation[];
  /** Stores this block made to P: linear index -> value. */
  stores: { index: number; value: number; thread: number }[];
}

export const phases = (cfg: Config) => Math.ceil(cfg.width / cfg.tile);
export const gridSide = (cfg: Config) => Math.ceil(cfg.width / cfg.tile);

interface Slot {
  value: number;
  phase: number; // -1 = never written
  unfilled: boolean;
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

/** Runs one block under one schedule and records every operation. */
export function runBlock(cfg: Config, prob: Problem, bx: number, by: number, schedule: Schedule, seed = 1): BlockTrace {
  const T = cfg.tile;
  const W = cfg.width;
  const nT = T * T;
  const nPh = phases(cfg);

  // The operation list is identical for every thread.
  type PlanOp = { op: 'loadM' | 'loadN' | 'barrier1' | 'barrier2' | 'mul' | 'store'; ph: number; k?: number };
  const plan: PlanOp[] = [];
  for (let ph = 0; ph < nPh; ph++) {
    plan.push({ op: 'loadM', ph }, { op: 'loadN', ph });
    if (cfg.barrier1) plan.push({ op: 'barrier1', ph });
    for (let k = 0; k < T; k++) plan.push({ op: 'mul', ph, k });
    if (cfg.barrier2) plan.push({ op: 'barrier2', ph });
  }
  plan.push({ op: 'store', ph: nPh - 1 });

  const fresh = (): Slot[] => Array.from({ length: nT }, () => ({ value: NaN, phase: -1, unfilled: false }));
  const Mds = fresh();
  const Nds = fresh();
  const pc = new Array(nT).fill(0);
  const waiting = new Array(nT).fill(false);
  const pvalue = new Array(nT).fill(0);
  const steps: Step[] = [];
  const violations: Violation[] = [];
  const stores: BlockTrace['stores'] = [];
  const rand = mulberry(seed);
  // The book writes threads as (row, column); spell out tx and ty so there is no doubt.
  const name = (t: number) => trText(`thread (tx ${t % T}, ty ${Math.floor(t / T)})`, `luồng (tx ${t % T}, ty ${Math.floor(t / T)})`);

  const flag = (s: Step, kind: ViolationKind, message: string) => {
    const v: Violation = { kind, step: s.index, thread: s.thread, message };
    // Keep the first violation on a step for display, but record them all.
    if (!s.violation) s.violation = v;
    violations.push(v);
  };

  const exec = (t: number) => {
    const tx = t % T;
    const ty = Math.floor(t / T);
    const Row = by * T + ty;
    const Col = bx * T + tx;
    const p = plan[pc[t]];
    const s: Step = { index: steps.length, thread: t, tx, ty, ph: p.ph, op: p.op };
    steps.push(s);
    if (p.op === 'barrier1' || p.op === 'barrier2') {
      waiting[t] = true;
      return;
    }
    if (p.op === 'loadM' || p.op === 'loadN') {
      const isM = p.op === 'loadM';
      const row = isM ? Row : p.ph * T + ty;
      const col = isM ? p.ph * T + tx : Col;
      const inRange = row < W && col < W;
      const guard = isM ? cfg.guardM : cfg.guardN;
      const zero = isM ? cfg.zeroFillM : cfg.zeroFillN;
      const src = isM ? prob.M : prob.N;
      const slots = isM ? Mds : Nds;
      const li = row * W + col;
      const slot = slots[ty * T + tx];
      const ref: SlotRef = { array: isM ? 'Mds' : 'Nds', r: ty, c: tx };
      if (inRange) {
        slot.value = src[li];
        slot.phase = p.ph;
        slot.unfilled = false;
        s.global = { array: isM ? 'M' : 'N', row, col, index: li, access: 'ok' };
        s.write = { ...ref, value: slot.value, phase: p.ph };
      } else if (guard) {
        s.global = { array: isM ? 'M' : 'N', row, col, index: li, access: 'skipped' };
        if (zero) {
          slot.value = 0;
          slot.phase = p.ph;
          slot.unfilled = false;
          s.write = { ...ref, value: 0, phase: p.ph };
        } else {
          // The slot keeps whatever it held: a value from an earlier phase, or garbage.
          slot.phase = p.ph;
          slot.unfilled = true;
          s.write = { ...ref, value: slot.value, phase: p.ph, unfilled: true };
        }
      } else if (li < W * W) {
        // Past the end of a row but still inside the array: the next row's element.
        slot.value = src[li];
        slot.phase = p.ph;
        slot.unfilled = false;
        s.global = { array: isM ? 'M' : 'N', row, col, index: li, access: 'wrap' };
        s.write = { ...ref, value: slot.value, phase: p.ph };
        flag(s, 'wrong-element',
          trText(
            `${name(t)} asked for ${isM ? 'M' : 'N'}[${row}][${col}], which does not exist, and got ` +
              `${isM ? 'M' : 'N'}[${Math.floor(li / W)}][${li % W}] instead`,
            `${name(t)} yêu cầu ${isM ? 'M' : 'N'}[${row}][${col}], phần tử không tồn tại, và lại nhận về ` +
              `${isM ? 'M' : 'N'}[${Math.floor(li / W)}][${li % W}]`,
          ));
      } else {
        slot.value = NaN;
        slot.phase = p.ph;
        slot.unfilled = false;
        s.global = { array: isM ? 'M' : 'N', row, col, index: li, access: 'outside' };
        s.write = { ...ref, value: NaN, phase: p.ph };
        flag(s, 'outside-read',
          trText(
            `${name(t)} read ${isM ? 'M' : 'N'}[${row}][${col}], which is beyond the end of the array`,
            `${name(t)} đọc ${isM ? 'M' : 'N'}[${row}][${col}], nằm ngoài cuối mảng`,
          ));
      }
    } else if (p.op === 'mul') {
      const k = p.k!;
      const a = Mds[ty * T + k];
      const b = Nds[k * T + tx];
      s.k = k;
      s.reads = [
        { array: 'Mds', r: ty, c: k, value: a.value, phase: a.phase, ok: a.phase === p.ph && !a.unfilled },
        { array: 'Nds', r: k, c: tx, value: b.value, phase: b.phase, ok: b.phase === p.ph && !b.unfilled },
      ];
      for (const [slot, r] of [[a, s.reads[0]], [b, s.reads[1]]] as const) {
        const at = `${r.array}[${r.r}][${r.c}]`;
        const owner = name(r.r * T + r.c);
        if (slot.phase < p.ph)
          flag(s, 'read-before-write',
            trText(
              `${name(t)} read ${at} in phase ${p.ph} before ${owner} had loaded this phase's value into it`,
              `${name(t)} đọc ${at} ở pha ${p.ph} trước khi ${owner} nạp giá trị của pha này vào đó`,
            ));
        else if (slot.phase > p.ph)
          flag(s, 'overwritten-before-read',
            trText(
              `${name(t)} still needed phase ${p.ph}'s value in ${at}, but ${owner} had already overwritten it for phase ${slot.phase}`,
              `${name(t)} vẫn cần giá trị của pha ${p.ph} trong ${at}, nhưng ${owner} đã ghi đè nó cho pha ${slot.phase}`,
            ));
        else if (slot.unfilled)
          flag(s, 'unfilled',
            trText(
              `${name(t)} used ${at}, which no load filled this phase (its element is out of range and nothing wrote a 0)`,
              `${name(t)} dùng ${at}, ô mà không lần nạp nào điền trong pha này (phần tử của nó nằm ngoài phạm vi và không có gì ghi số 0)`,
            ));
      }
      pvalue[t] += a.value * b.value;
    } else if (p.op === 'store') {
      const inRange = Row < W && Col < W;
      const li = Row * W + Col;
      if (inRange) {
        stores.push({ index: li, value: pvalue[t], thread: t });
        s.global = { array: 'P', row: Row, col: Col, index: li, access: 'ok' };
      } else if (cfg.guardP) {
        s.global = { array: 'P', row: Row, col: Col, index: li, access: 'skipped' };
      } else if (li < W * W) {
        stores.push({ index: li, value: pvalue[t], thread: t });
        s.global = { array: 'P', row: Row, col: Col, index: li, access: 'wrap' };
        flag(s, 'bad-store',
          trText(
            `${name(t)} owns no element of P but stored to P[${Row}][${Col}], which lands on P[${Math.floor(li / W)}][${li % W}]`,
            `${name(t)} không sở hữu phần tử nào của P nhưng đã ghi vào P[${Row}][${Col}], vị trí rơi vào P[${Math.floor(li / W)}][${li % W}]`,
          ));
      } else {
        s.global = { array: 'P', row: Row, col: Col, index: li, access: 'outside' };
        flag(s, 'bad-store', trText(`${name(t)} stored to P[${Row}][${Col}], beyond the end of the array`, `${name(t)} đã ghi vào P[${Row}][${Col}], ngoài cuối mảng`));
      }
    }
    s.pvalue = pvalue[t];
    pc[t]++;
  };

  const live = (t: number) => pc[t] < plan.length;
  const runnable = (t: number) => live(t) && !waiting[t];
  const release = () => {
    const anyWaiting = waiting.some(Boolean);
    const allThere = waiting.every((w, t) => w || !live(t));
    if (!anyWaiting || !allThere) return false;
    const t0 = waiting.findIndex(Boolean);
    const p = plan[pc[t0]];
    steps.push({ index: steps.length, thread: -1, tx: -1, ty: -1, ph: p.ph, op: p.op === 'barrier1' ? 'release1' : 'release2' });
    for (let t = 0; t < nT; t++)
      if (waiting[t]) {
        waiting[t] = false;
        pc[t]++;
      }
    return true;
  };

  const order = Array.from({ length: nT }, (_, i) => i);
  if (schedule === 'reversed') order.reverse();
  let guardCount = 0;
  while (order.some(live)) {
    if (++guardCount > 1e6) throw new Error('simulation did not terminate');
    if (schedule === 'lockstep') {
      for (const t of order) if (runnable(t)) exec(t);
    } else if (schedule === 'shuffled') {
      const cands = order.filter(runnable);
      if (cands.length) exec(cands[Math.floor(rand() * cands.length)]);
    } else {
      // Run each thread until it reaches a barrier or finishes, in turn.
      for (const t of order) while (runnable(t)) exec(t);
    }
    release();
  }
  return { bx, by, steps, violations, stores };
}

export interface RunResult {
  P: number[];
  /** Indices of P that differ from the reference. */
  wrong: number[];
  traces: BlockTrace[];
  violations: (Violation & { bx: number; by: number })[];
  correct: boolean;
}

/** Runs every block (one after another) and compares P with the reference. */
export function runAll(cfg: Config, prob: Problem, schedule: Schedule, seed = 1): RunResult {
  const g = gridSide(cfg);
  const P = new Array(cfg.width * cfg.width).fill(NaN);
  const traces: BlockTrace[] = [];
  const violations: RunResult['violations'] = [];
  for (let by = 0; by < g; by++)
    for (let bx = 0; bx < g; bx++) {
      const tr = runBlock(cfg, prob, bx, by, schedule, seed + by * g + bx);
      traces.push(tr);
      for (const s of tr.stores) P[s.index] = s.value;
      for (const v of tr.violations) violations.push({ ...v, bx, by });
    }
  const wrong: number[] = [];
  for (let i = 0; i < P.length; i++) if (!(P[i] === prob.ref[i])) wrong.push(i);
  return { P, wrong, traces, violations, correct: wrong.length === 0 };
}

export interface BlockState {
  Mds: Slot[];
  Nds: Slot[];
  pvalue: number[];
  /** Per thread: the last step it executed (index into trace.steps), or -1. */
  last: number[];
  waiting: boolean[];
  done: boolean[];
  stores: BlockTrace['stores'];
}

/** The state of a block after its first `n` steps, for the stepper. */
export function stateAt(cfg: Config, trace: BlockTrace, n: number): BlockState {
  const nT = cfg.tile * cfg.tile;
  const mk = (): Slot[] => Array.from({ length: nT }, () => ({ value: NaN, phase: -1, unfilled: false }));
  const st: BlockState = {
    Mds: mk(),
    Nds: mk(),
    pvalue: new Array(nT).fill(0),
    last: new Array(nT).fill(-1),
    waiting: new Array(nT).fill(false),
    done: new Array(nT).fill(false),
    stores: [],
  };
  for (let i = 0; i < Math.min(n, trace.steps.length); i++) {
    const s = trace.steps[i];
    if (s.thread < 0) {
      st.waiting.fill(false);
      continue;
    }
    st.last[s.thread] = i;
    if (s.write) {
      const slots = s.write.array === 'Mds' ? st.Mds : st.Nds;
      slots[s.write.r * cfg.tile + s.write.c] = { value: s.write.value, phase: s.write.phase, unfilled: !!s.write.unfilled };
    }
    if (s.op === 'barrier1' || s.op === 'barrier2') st.waiting[s.thread] = true;
    if (s.pvalue !== undefined) st.pvalue[s.thread] = s.pvalue;
    if (s.op === 'store') {
      st.done[s.thread] = true;
      if (s.global && (s.global.access === 'ok' || s.global.access === 'wrap'))
        st.stores.push({ index: s.global.index, value: s.pvalue ?? 0, thread: s.thread });
    }
  }
  return st;
}

// ---------- counting global-memory traffic (sections 5.1 and 5.3) ----------

/** How many threads of one block read each element of M and of N in the NAIVE kernel. */
export function naiveBlockReads(width: number, block: number, bx: number, by: number) {
  const M = new Array(width * width).fill(0);
  const N = new Array(width * width).fill(0);
  for (let ty = 0; ty < block; ty++)
    for (let tx = 0; tx < block; tx++) {
      const row = by * block + ty;
      const col = bx * block + tx;
      if (row >= width || col >= width) continue;
      for (let k = 0; k < width; k++) {
        M[row * width + k]++;
        N[k * width + col]++;
      }
    }
  return { M, N };
}

export interface Traffic {
  /** Global loads of input elements for the whole multiplication. */
  loads: number;
  /** Times each input element is fetched from global memory. */
  perElement: number;
  /** FLOP per byte of input loaded from global memory (the book's headline figure). */
  flopPerByte: number;
  /** The same, also counting the one store of each output element. */
  flopPerByteWithStores: number;
}

/** The book's simple model (section 5.1 and 5.4): width a multiple of the tile, no caches. */
export function traffic(width: number, tile: number | null): Traffic {
  const flop = 2 * width ** 3;
  const perElement = tile ? width / tile : width;
  const loads = 2 * width * width * perElement;
  const stores = width * width;
  return { loads, perElement, flopPerByte: flop / (4 * loads), flopPerByteWithStores: flop / (4 * (loads + stores)) };
}
