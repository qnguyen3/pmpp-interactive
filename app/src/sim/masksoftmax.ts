// A thread-level model of the book's causal mask-and-softmax kernel (Fig. 20.4, p. 487),
// as corrected in labs/ch20/snippets/s203_softmax_corrected.cu, and the global-memory
// traffic of the unfused attention pipeline around it (chapter 20, section 20.3).
//
// What it models, for one block (one row r of the N x N matrix S):
//   - every thread t as its own sequence of source-level operations, numbered by the
//     kernel's lines (the book's numbering, 1 to 28): the max loop over columns
//     t, t + BLOCK_SIZE, ... while idx <= r (line 8), the barrier at line 10, the max
//     BlockReduce (11), thread 0 publishing it through the shared scalar max_or_sum (12),
//     the barrier at 13, every thread reading it back (14), the sum loop (17), the barriers
//     at 19 and 22 around the sum reduction (20) and its publication (21, 23), the output
//     loop over all N columns writing explicit zeros above the diagonal (25), and thread 0
//     writing the row's log-sum-exp (27; the book writes the denominator D, see A7).
//   - each barrier can be removed; the loop bound can be the defect idx < N (the mask is
//     lost); the max subtraction can be left out (lines 17, 25 and 27 then use exp(S) and
//     log(sum)).
//   - four thread orders, as tiled.ts: lockstep, in order, reversed, seeded shuffle.
//   - float32 arithmetic through Math.fround, so exp of a logit above about 88.72 is Infinity.
//
// What it leaves out, and what the lesson says where the model is shown:
//   - BlockReduce is one collective step: every thread arrives, then the result appears in
//     thread 0 only (facts A2: "valid in thread 0"). Its temporary storage is not modelled;
//     CUB asks for a barrier between two uses of it (facts A2), and the model only checks
//     that one of the barriers at lines 13 or 19 is there.
//   - A thread advances one source-level operation at a time; no warps, no timing.
//   - Shared memory that nobody wrote reads as NaN (the worst case; on a device it holds
//     whatever was there before).
//
// Erasable TypeScript only, no runtime imports: the browser checks import this file in Node.
import { tr as trText } from '../lib/locale.ts';

export type Schedule = 'lockstep' | 'in-order' | 'reversed' | 'shuffled';
export const SCHEDULES: readonly Schedule[] = ['lockstep', 'in-order', 'reversed', 'shuffled'];
export const SCHEDULE_LABEL: Record<Schedule, string> = {
  get lockstep() { return trText('all threads in lockstep', 'mọi luồng chạy đồng bộ từng bước (lockstep)'); },
  get 'in-order'() { return trText('each thread runs as far as it can, thread 0 first', 'mỗi luồng chạy xa nhất có thể, luồng 0 trước'); },
  get reversed() { return trText('each thread runs as far as it can, last thread first', 'mỗi luồng chạy xa nhất có thể, luồng cuối trước'); },
  get shuffled() { return trText('random interleaving', 'xen kẽ ngẫu nhiên'); },
};

/** The corrected kernel's identity for max: the most negative finite float (not -infinity). */
export const NEG_IDENTITY = -3.4028234663852886e38;

export interface SoftmaxConfig {
  /** Rows and columns of S. */
  n: number;
  /** BLOCK_SIZE: threads per block. */
  block: number;
  barrier10: boolean;
  barrier13: boolean;
  barrier19: boolean;
  barrier22: boolean;
  /** Bound of the max and sum loops: 'row' is idx <= blockIdx.x (the mask); 'n' is the defect idx < N. */
  bound: 'row' | 'n';
  /** Subtract the row max before exp (Eq. 20.2). */
  subtractMax: boolean;
}

export const CORRECT_SOFTMAX: Omit<SoftmaxConfig, 'n' | 'block'> = {
  barrier10: true,
  barrier13: true,
  barrier19: true,
  barrier22: true,
  bound: 'row',
  subtractMax: true,
};

/** The kernel's input and the answer it must give; attnref.causalSoftmaxProblem builds one. */
export interface SoftmaxInput {
  n: number;
  /** The full N x N matrix of scaled logits, in float32. */
  S: number[][];
  refP: number[][];
  refL: number[];
}

export type ThreadOp = 'max' | 'barrier' | 'reduceMax' | 'writeMax' | 'readMax' | 'sum' | 'reduceSum' | 'writeSum' | 'readSum' | 'out' | 'lse';
export type BlockOp = 'release' | 'reduced';

/** The kernel line each operation is on (the book's numbering). */
export const LINE: Record<Exclude<ThreadOp, 'barrier'>, number> = {
  max: 8, reduceMax: 11, writeMax: 12, readMax: 14, sum: 17, reduceSum: 20, writeSum: 21, readSum: 23, out: 25, lse: 27,
};
export const BARRIER_LINES = [10, 13, 19, 22] as const;

export type ViolationKind =
  | 'read-before-write' // a thread read max_or_sum before thread 0 put the value it needs there
  | 'overwritten-before-read' // thread 0 had already replaced the max with the sum
  | 'masked-read' // a thread read a column in the row's future
  | 'overflow' // exp gave infinity
  | 'temp-reuse'; // the second BlockReduce reused temp_store with no barrier after the first

export interface Violation {
  kind: ViolationKind;
  step: number;
  thread: number;
  message: string;
}

export interface ThreadState {
  /** Line 6/8: the thread's private max. */
  privMax: number;
  /** max_val_row: NaN until this thread has a value for it. */
  maxRow: number;
  /** Line 15/17: the thread's private sum. */
  privSum: number;
  /** sum_row: NaN until this thread has a value for it. */
  sumRow: number;
  /** Line of the barrier or reduction it waits at, or null. */
  waitingAt: number | null;
  done: boolean;
}
export interface SharedState {
  value: number;
  holds: 'unset' | 'max' | 'sum';
}
export interface Snapshot {
  threads: ThreadState[];
  shared: SharedState;
  /** The block's row of P: null where not yet written. */
  P: (number | null)[];
  L: number | null;
}

export interface Step {
  index: number;
  /** The thread, or -1 for a block-wide event (a barrier opening, a reduction completing). */
  thread: number;
  op: ThreadOp | BlockOp;
  line: number;
  /** Column, for max, sum and out. */
  idx?: number;
  /** What was read, computed or written. */
  value?: number;
  /** The column is in the row's future (idx > row). */
  masked?: boolean;
  violation?: Violation;
  after: Snapshot;
}

export interface RowTrace {
  row: number;
  steps: Step[];
  violations: Violation[];
  P: number[];
  L: number;
  /** Columns of P that differ from the reference. */
  wrongCols: number[];
  lOk: boolean;
  correct: boolean;
}

/** The columns thread t reads in the max and sum loops of block `row`. */
export function columnsOf(cfg: SoftmaxConfig, row: number, t: number): number[] {
  const bound = cfg.bound === 'row' ? row : cfg.n - 1;
  const out: number[] = [];
  for (let idx = t; idx <= bound; idx += cfg.block) out.push(idx);
  return out;
}
/** The columns thread t writes in the output loop (all N, as the book's line 24 does). */
export function outputColumnsOf(cfg: SoftmaxConfig, t: number): number[] {
  const out: number[] = [];
  for (let idx = t; idx < cfg.n; idx += cfg.block) out.push(idx);
  return out;
}

function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const f = Math.fround;
const close = (a: number, b: number, rel: number) => Math.abs(a - b) <= rel * Math.max(1, Math.abs(b));

interface PlanOp {
  op: ThreadOp;
  line: number;
  idx?: number;
}

function planFor(cfg: SoftmaxConfig, row: number, t: number): PlanOp[] {
  const plan: PlanOp[] = [];
  const cols = columnsOf(cfg, row, t);
  for (const idx of cols) plan.push({ op: 'max', line: LINE.max, idx });
  if (cfg.barrier10) plan.push({ op: 'barrier', line: 10 });
  plan.push({ op: 'reduceMax', line: LINE.reduceMax });
  if (t === 0) plan.push({ op: 'writeMax', line: LINE.writeMax });
  if (cfg.barrier13) plan.push({ op: 'barrier', line: 13 });
  plan.push({ op: 'readMax', line: LINE.readMax });
  for (const idx of cols) plan.push({ op: 'sum', line: LINE.sum, idx });
  if (cfg.barrier19) plan.push({ op: 'barrier', line: 19 });
  plan.push({ op: 'reduceSum', line: LINE.reduceSum });
  if (t === 0) plan.push({ op: 'writeSum', line: LINE.writeSum });
  if (cfg.barrier22) plan.push({ op: 'barrier', line: 22 });
  plan.push({ op: 'readSum', line: LINE.readSum });
  for (const idx of outputColumnsOf(cfg, t)) plan.push({ op: 'out', line: LINE.out, idx });
  if (t === 0) plan.push({ op: 'lse', line: LINE.lse });
  return plan;
}

const isSync = (op: ThreadOp) => op === 'barrier' || op === 'reduceMax' || op === 'reduceSum';

/** Runs block `row` under one thread order and records every operation with the state after it. */
export function runRow(cfg: SoftmaxConfig, input: SoftmaxInput, row: number, schedule: Schedule, seed = 1): RowTrace {
  const B = cfg.block;
  const S = input.S[row];
  const plans = Array.from({ length: B }, (_, t) => planFor(cfg, row, t));
  const pc = new Array(B).fill(0);
  const threads: ThreadState[] = Array.from({ length: B }, () => ({
    privMax: NEG_IDENTITY,
    maxRow: NaN,
    privSum: 0,
    sumRow: NaN,
    waitingAt: null,
    done: false,
  }));
  const shared: SharedState = { value: NaN, holds: 'unset' };
  const P: (number | null)[] = new Array(cfg.n).fill(null);
  let L: number | null = null;
  const steps: Step[] = [];
  const violations: Violation[] = [];
  const rand = mulberry(seed * 9973 + row);
  let reductions = 0;

  const snap = (): Snapshot => ({ threads: threads.map((x) => ({ ...x })), shared: { ...shared }, P: [...P], L });
  const flag = (s: Omit<Step, 'after'>, kind: ViolationKind, message: string) => {
    const v: Violation = { kind, step: s.index, thread: s.thread, message };
    if (!s.violation) s.violation = v;
    violations.push(v);
  };
  const live = (t: number) => pc[t] < plans[t].length;
  const runnable = (t: number) => live(t) && threads[t].waitingAt === null;

  const exec = (t: number) => {
    const p = plans[t][pc[t]];
    const th = threads[t];
    const s: Omit<Step, 'after'> = { index: steps.length, thread: t, op: p.op, line: p.line, idx: p.idx };
    if (isSync(p.op)) {
      th.waitingAt = p.line;
      steps.push({ ...s, after: snap() });
      return; // pc advances when the block-wide event happens
    }
    switch (p.op) {
      case 'max': {
        const v = S[p.idx!];
        s.value = v;
        s.masked = p.idx! > row;
        if (s.masked) flag(s, 'masked-read', trText(`thread ${t} read S[${row}][${p.idx}], a column in row ${row}'s future`, `luồng ${t} đọc S[${row}][${p.idx}], một cột nằm trong tương lai của hàng ${row}`));
        if (v > th.privMax) th.privMax = v;
        break;
      }
      case 'writeMax':
        shared.value = th.maxRow;
        shared.holds = 'max';
        s.value = th.maxRow;
        break;
      case 'readMax':
        s.value = shared.value;
        if (shared.holds === 'unset')
          flag(s, 'read-before-write', trText(`thread ${t} read max_or_sum before thread 0 had stored the row max in it`, `luồng ${t} đọc max_or_sum trước khi luồng 0 lưu max của hàng vào đó`));
        else if (shared.holds === 'sum')
          flag(s, 'overwritten-before-read', trText(`thread ${t} wanted the max, but thread 0 had already overwritten max_or_sum with the sum`, `luồng ${t} cần max, nhưng luồng 0 đã ghi đè max_or_sum bằng sum`));
        th.maxRow = shared.value;
        break;
      case 'sum': {
        const x = cfg.subtractMax ? f(S[p.idx!] - th.maxRow) : S[p.idx!];
        const e = f(Math.exp(x));
        s.value = e;
        s.masked = p.idx! > row;
        if (s.masked) flag(s, 'masked-read', trText(`thread ${t} read S[${row}][${p.idx}], a column in row ${row}'s future`, `luồng ${t} đọc S[${row}][${p.idx}], một cột nằm trong tương lai của hàng ${row}`));
        if (e === Infinity) flag(s, 'overflow', trText(`exp(${fmtNum(x)}) is beyond float32's range (above about 88.72), so it is infinity`, `exp(${fmtNum(x)}) vượt quá phạm vi của float32 (trên khoảng 88.72), nên nó là vô cực`));
        th.privSum = f(th.privSum + e);
        break;
      }
      case 'writeSum':
        shared.value = th.sumRow;
        shared.holds = 'sum';
        s.value = th.sumRow;
        break;
      case 'readSum':
        s.value = shared.value;
        if (shared.holds !== 'sum')
          flag(s, 'read-before-write', trText(`thread ${t} read max_or_sum before thread 0 had stored the row sum in it (it still held ${shared.holds === 'max' ? 'the max' : 'nothing'})`, `luồng ${t} đọc max_or_sum trước khi luồng 0 lưu sum của hàng vào đó (nó vẫn đang giữ ${shared.holds === 'max' ? 'max' : 'không gì cả'})`));
        th.sumRow = shared.value;
        break;
      case 'out': {
        const idx = p.idx!;
        const v = idx <= row ? f(f(Math.exp(cfg.subtractMax ? f(S[idx] - th.maxRow) : S[idx])) / th.sumRow) : 0;
        P[idx] = v;
        s.value = v;
        s.masked = idx > row;
        break;
      }
      case 'lse':
        L = cfg.subtractMax ? f(th.maxRow + f(Math.log(th.sumRow))) : f(Math.log(th.sumRow));
        s.value = L;
        break;
    }
    pc[t]++;
    if (!live(t)) th.done = true;
    steps.push({ ...s, after: snap() });
  };

  const release = () => {
    const waiting = threads.map((x) => x.waitingAt);
    if (!waiting.some((w) => w !== null)) return;
    if (!threads.every((x, t) => x.waitingAt !== null || !live(t))) return;
    const lines = new Set(waiting.filter((w) => w !== null));
    if (lines.size !== 1) throw new Error('threads wait at different points: ' + [...lines].join(', '));
    const line = [...lines][0]!;
    const s: Omit<Step, 'after'> = { index: steps.length, thread: -1, op: 'release', line };
    if (line === LINE.reduceMax || line === LINE.reduceSum) {
      s.op = 'reduced';
      reductions++;
      if (line === LINE.reduceMax) {
        let m = NEG_IDENTITY;
        for (const x of threads) if (x.privMax > m) m = x.privMax;
        threads[0].maxRow = m; // valid in thread 0 only
        s.value = m;
      } else {
        let sum = 0;
        for (const x of threads) sum = f(sum + x.privSum);
        threads[0].sumRow = sum;
        s.value = sum;
        if (reductions === 2 && !cfg.barrier13 && !cfg.barrier19)
          flag(s, 'temp-reuse', trText('the sum reduction reused temp_store with no barrier since the max reduction finished; CUB asks for one', 'phép rút gọn sum dùng lại temp_store mà không có rào chắn nào kể từ khi phép rút gọn max kết thúc; CUB yêu cầu có một rào chắn'));
      }
    }
    threads.forEach((x, t) => {
      if (x.waitingAt !== null) {
        x.waitingAt = null;
        pc[t]++;
        if (!live(t)) x.done = true;
      }
    });
    steps.push({ ...s, after: snap() });
  };

  const order = Array.from({ length: B }, (_, i) => i);
  if (schedule === 'reversed') order.reverse();
  let guard = 0;
  while (order.some(live)) {
    if (++guard > 1e5) throw new Error('simulation did not terminate');
    if (schedule === 'lockstep') {
      for (const t of order) if (runnable(t)) exec(t);
    } else if (schedule === 'shuffled') {
      const c = order.filter(runnable);
      if (c.length) exec(c[Math.floor(rand() * c.length)]);
    } else {
      for (const t of order) while (runnable(t)) exec(t);
    }
    release();
  }

  const finalP = P.map((x) => (x === null ? NaN : x));
  const ref = input.refP[row];
  const wrongCols = finalP.flatMap((v, j) => (close(v, ref[j], 2e-6) ? [] : [j]));
  const finalL = L ?? NaN;
  const lOk = close(finalL, input.refL[row], 1e-5);
  return { row, steps, violations, P: finalP, L: finalL, wrongCols, lOk, correct: wrongCols.length === 0 && lOk };
}

/** The state after the first n steps (n = 0: before anything ran). */
export function stateAt(cfg: SoftmaxConfig, trace: RowTrace, n: number): Snapshot {
  if (n > 0) return trace.steps[Math.min(n, trace.steps.length) - 1].after;
  return {
    threads: Array.from({ length: cfg.block }, () => ({ privMax: NEG_IDENTITY, maxRow: NaN, privSum: 0, sumRow: NaN, waitingAt: null, done: false })),
    shared: { value: NaN, holds: 'unset' },
    P: new Array(cfg.n).fill(null),
    L: null,
  };
}

export interface AllRows {
  correct: boolean;
  wrongRows: number[];
  kinds: ViolationKind[];
}

/** Every block of the grid (every row), one after another, compared with the reference. */
export function runAllRows(cfg: SoftmaxConfig, input: SoftmaxInput, schedule: Schedule, seed = 1): AllRows {
  const wrongRows: number[] = [];
  const kinds = new Set<ViolationKind>();
  for (let r = 0; r < cfg.n; r++) {
    const tr = runRow(cfg, input, r, schedule, seed);
    if (!tr.correct) wrongRows.push(r);
    for (const v of tr.violations) kinds.add(v.kind);
  }
  return { correct: wrongRows.length === 0, wrongRows, kinds: [...kinds] };
}

/** The kernel variants the lesson's outcome table runs. */
export const VARIANTS: readonly { id: string; label: string; over: Partial<SoftmaxConfig>; offset?: number }[] = [
  { id: 'as-corrected', get label() { return trText('all four barriers', 'cả bốn rào chắn'); }, over: {} },
  { id: 'no10', get label() { return trText('no barrier at line 10', 'không có rào chắn ở dòng 10'); }, over: { barrier10: false } },
  { id: 'no13', get label() { return trText('no barrier at line 13', 'không có rào chắn ở dòng 13'); }, over: { barrier13: false } },
  { id: 'no19', get label() { return trText('no barrier at line 19', 'không có rào chắn ở dòng 19'); }, over: { barrier19: false } },
  { id: 'no22', get label() { return trText('no barrier at line 22', 'không có rào chắn ở dòng 22'); }, over: { barrier22: false } },
  { id: 'bound', get label() { return trText('loop bound idx < N (mask lost)', 'cận vòng lặp idx < N (mất mặt nạ)'); }, over: { bound: 'n' } },
  { id: 'nomax', get label() { return trText('no max subtraction', 'không trừ max'); }, over: { subtractMax: false } },
  { id: 'nomax90', get label() { return trText('no max subtraction, logits + 90', 'không trừ max, logits + 90'); }, over: { subtractMax: false }, offset: 90 },
];

function fmtNum(x: number): string {
  return Number.isFinite(x) ? String(Number(x.toPrecision(4))) : String(x);
}

// ---------------------------------------------------------------- traffic of the unfused pipeline
export interface KernelBytes {
  read: number;
  write: number;
}
export interface UnfusedTraffic {
  /** S = scale * Q K^T: read Q and K, write all of S (a library GEMM knows no mask). */
  gemm1: KernelBytes;
  /** Fig. 20.4: three passes over the visible half of S (max, sum, output), write all of P and L. */
  softmax: KernelBytes;
  /** O = P V: read P and V, write O. */
  gemm2: KernelBytes;
  total: number;
  /** Read Q, K, V once and write O once: what any implementation must move. */
  irreducible: number;
  /** Bytes S and P occupy between the kernels. */
  sp: number;
}

/**
 * Minimum global-memory bytes of each kernel as written, for `heads` independent heads of an
 * n-token causal problem with head width d, at `bytes` per element of Q, K, V, S, P and O.
 * L (one log-sum-exp per row) is kept in float32 whatever `bytes` is. No caches; each GEMM is
 * counted at its compulsory minimum (every input read once, every output written once).
 */
export function unfusedTraffic(n: number, d: number, heads: number, bytes: number): UnfusedTraffic {
  const visibleEntries = (n * (n + 1)) / 2;
  const gemm1 = { read: 2 * n * d * bytes * heads, write: n * n * bytes * heads };
  const softmax = { read: 3 * visibleEntries * bytes * heads, write: (n * n * bytes + n * 4) * heads };
  const gemm2 = { read: (n * n + n * d) * bytes * heads, write: n * d * bytes * heads };
  const total = gemm1.read + gemm1.write + softmax.read + softmax.write + gemm2.read + gemm2.write;
  return { gemm1, softmax, gemm2, total, irreducible: 4 * n * d * bytes * heads, sp: 2 * n * n * bytes * heads };
}
