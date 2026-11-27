// Chapter 11, sections 11.2 and 11.3: a thread-level model of the Kogge-Stone block
// kernels. It runs the "stride by stride" stepper, the warp explorer, the race hunt and
// its outcome table of section 11.2, the two-buffer stepper and the buffer prediction of
// section 11.3, and every frame of the narrated scene "Doubling strides".
//
// The kernels, as labs/ch11/snippets/s112_kogge_stone.cu and s112_race_variants.cu hold
// them (kernelSource below prints the same text; ksblock.test.ts checks):
//   - Fig. 11.3 (p. 256): one shared buffer; each iteration a barrier (line 10), the active
//     threads add into a private temp (line 13), a second barrier (line 14), then store
//     (line 16). Either barrier can be switched off, and the add can go straight into
//     buffer_s (no temp), in which case the second barrier can only follow the store.
//   - Fig. 11.5 (p. 260): two shared buffers whose roles swap every iteration, one barrier
//     (line 09); the threads that are not adding copy their value across (line 14). The
//     copy can be switched off.
//
// What it records: every load, read, write, copy, pointer swap and barrier of every
// thread, with the value each access carries and the inputs that value counts (a
// coefficient per input: "x0 + 2x1 + 2x2 + x3" when x1 and x2 are counted twice); which
// write each read should have seen and which it saw; the violations (read before write,
// overwritten before read, a copy that never happened); the block's output.
//
// Sources for the rules: the book, pp. 255-261 (the algorithm, both kernels, the race
// of pp. 257-258 with thread 4 and thread 6 in the stride-2 iteration, the copy branch of
// pp. 259-260), p. 266 (additions per step), exercise 1 and 2 (p. 286); warp widths from
// notes/today/ch05-facts.md B5 and ch10-facts B2 (32 on NVIDIA, 64 on AMD gfx9). The
// thread orders and the step granularity follow sim/tiled.ts and sim/doublebuf.ts, so the
// tables of chapters 5, 6 and 11 read the same way.
//
// What it leaves out (the lessons say so beside every interactive):
//   - A thread's unit of progress is one source-level operation; real threads interleave
//     at the instruction level and run in warps. The add straight into buffer_s is split
//     into its load and its store, as compiled code does.
//   - "Lockstep" is an idealised single warp; a real block has several warps and no
//     promised order between them.
//   - No timing, and nothing about what divergence costs: warpStates says which lanes are
//     active, not how long the warp takes.
//   - Uninitialised shared memory is NaN (the worst case). The course's emulator fills it
//     with the bytes 0xCD instead.
//
// Erasable TypeScript only, and no imports: the browser checks import this file directly.
import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- configuration
export type Kind = 'single' | 'double';
export interface Config {
  /** Threads in the block, which is also the segment size (one thread per element). */
  threads: number;
  kind: Kind;
  /** The barrier at the top of the loop: line 10 of Fig. 11.3, line 09 of Fig. 11.5. */
  barrier1: boolean;
  /** One buffer only: line 14 of Fig. 11.3 (or after the store when there is no temp). */
  barrier2: boolean;
  /** One buffer only: add into a private temp (Fig. 11.3) rather than straight into buffer_s. */
  temp: boolean;
  /** Two buffers only: the idle threads copy their value across (Fig. 11.5, line 14). */
  copy: boolean;
}

export const FIG_11_3: Omit<Config, 'threads'> = { kind: 'single', barrier1: true, barrier2: true, temp: true, copy: true };
export const FIG_11_5: Omit<Config, 'threads'> = { kind: 'double', barrier1: true, barrier2: false, temp: true, copy: true };

export type Schedule = 'lockstep' | 'in-order' | 'reversed' | 'shuffled' | 'book-race';
export const SCHEDULES: Schedule[] = ['lockstep', 'in-order', 'reversed', 'shuffled', 'book-race'];
export const SCHEDULE_LABEL: Record<Schedule, string> = {
  get lockstep() { return tr('all threads in lockstep', 'mọi luồng chạy đồng bộ từng bước (lockstep)'); },
  get 'in-order'() { return tr('each thread runs as far as it can, thread 0 first', 'mỗi luồng chạy xa nhất có thể, luồng 0 trước'); },
  get reversed() { return tr('each thread runs as far as it can, last thread first', 'mỗi luồng chạy xa nhất có thể, luồng cuối trước'); },
  get shuffled() { return tr('random interleaving', 'xen kẽ ngẫu nhiên'); },
  get 'book-race'() { return tr("the book's race: thread 4 runs ahead in the stride-2 step", 'race condition của sách: luồng 4 chạy trước ở bước stride 2'); },
};
/** Pages 257-258: in the stride-2 iteration (iteration 1, counting from 0) thread 4 stores before thread 6 reads. */
export const RACE = { thread: 4, iter: 1 };

/** Exercise 1 (p. 286). */
export const EXERCISE_1 = [4, 6, 7, 1, 2, 8, 5, 2];
/** Exercise 1's arrays after the strides 1, 2 and 4 (notes, exercise 1). */
export const EXERCISE_1_STEPS = [
  [4, 10, 13, 8, 3, 10, 13, 7],
  [4, 10, 17, 18, 16, 18, 16, 17],
  [4, 10, 17, 18, 20, 28, 33, 35],
];

/** The strides of the loop: 1, 2, 4, ... while below the block size. */
export function strides(B: number): number[] {
  const s: number[] = [];
  for (let stride = 1; stride < B; stride *= 2) s.push(stride);
  return s;
}
export const iterations = (B: number) => strides(B).length;

// ---------------------------------------------------------------- values that know what they count
/** Coefficient of every input in a value; null when the value is garbage (uninitialised). */
export type Coef = number[] | null;
export const unit = (i: number, B: number): number[] => Array.from({ length: B }, (_, k) => (k === i ? 1 : 0));
export const zeros = (B: number): number[] => new Array(B).fill(0);
const addCoef = (a: Coef, b: Coef): Coef => (a && b ? a.map((v, k) => v + b[k]) : null);
const sameCoef = (a: Coef, b: Coef) => !!a && !!b && a.length === b.length && a.every((v, k) => v === b[k]);
/** The inputs position s should hold after `done` iterations: the last min(2^done, s + 1) of x0..xs. */
export function windowCoef(s: number, done: number, B: number): number[] {
  const from = Math.max(0, s - 2 ** done + 1);
  return Array.from({ length: B }, (_, k) => (k >= from && k <= s ? 1 : 0));
}

/** "x3", "x0..x3", "0", "garbage", or "x0 + 2x1 + 2x2 + x3" for anything that is not a window. */
export function rangeText(c: Coef, sigma = false): string {
  if (!c) return tr('garbage', 'rác');
  const on = c.flatMap((v, k) => (v ? [k] : []));
  if (!on.length) return '0';
  const contiguous = on.every((k, j) => j === 0 || k === on[j - 1] + 1) && on.every((k) => c[k] === 1);
  if (contiguous) {
    const a = on[0];
    const b = on[on.length - 1];
    return a === b ? `x${a}` : `${sigma ? 'Σ' : ''}x${a}..x${b}`;
  }
  return on.map((k) => (c[k] === 1 ? `x${k}` : `${c[k]}x${k}`)).join(' + ');
}

/**
 * The same, short enough for a cell: a window "x0..x6", plus the inputs counted more than
 * once ("x0..x6 +x1+x2" for x0 + 2x1 + 2x2 + x3 + x4 + x5 + x6), "?" for garbage.
 */
export function rangeShort(c: Coef): string {
  if (!c) return '?';
  const on = c.flatMap((v, k) => (v ? [k] : []));
  if (!on.length) return '0';
  const contiguous = on.every((k, j) => j === 0 || k === on[j - 1] + 1);
  if (!contiguous) return rangeText(c);
  const a = on[0];
  const b = on[on.length - 1];
  const base = a === b ? `x${a}` : `x${a}..x${b}`;
  const extra = on.flatMap((k) => Array.from({ length: c[k] - 1 }, () => `+x${k}`)).join('');
  return base + (extra ? ' ' + extra : '');
}

// ---------------------------------------------------------------- the programs
export type Op = 'load' | 'barrier1' | 'read' | 'barrier2' | 'write' | 'add' | 'copy' | 'swap' | 'store';
interface Instr {
  op: Op;
  /** Loop iteration (0 for stride 1), -1 for the load, the number of iterations for the store. */
  iter: number;
}

export const active = (t: number, iter: number) => t >= 2 ** iter;

/** What thread t executes, in source order, under a configuration. */
export function program(cfg: Config, t: number): Instr[] {
  const L = iterations(cfg.threads);
  const out: Instr[] = [{ op: 'load', iter: -1 }];
  for (let k = 0; k < L; k++) {
    if (cfg.barrier1) out.push({ op: 'barrier1', iter: k });
    if (cfg.kind === 'single') {
      if (active(t, k)) out.push({ op: 'read', iter: k });
      if (cfg.barrier2 && cfg.temp) out.push({ op: 'barrier2', iter: k });
      if (active(t, k)) out.push({ op: 'write', iter: k });
      if (cfg.barrier2 && !cfg.temp) out.push({ op: 'barrier2', iter: k });
    } else {
      if (active(t, k)) out.push({ op: 'add', iter: k });
      else if (cfg.copy) out.push({ op: 'copy', iter: k });
      out.push({ op: 'swap', iter: k });
    }
  }
  out.push({ op: 'store', iter: L });
  return out;
}

// ---------------------------------------------------------------- the run
export interface Read {
  buf: 0 | 1;
  slot: number;
  value: number;
  coef: Coef;
  /** Iteration of the write found: -1 the load, null never written. */
  found: number | null;
  /** Iteration of the write this read should have found. */
  expected: number;
  /** It found the intended write. */
  ok: boolean;
  /** It found the intended write, but that value was already wrong. */
  tainted: boolean;
}

export type ViolationKind = 'read-before-write' | 'overwritten-before-read' | 'missing-copy';
export const KIND_LABEL: Record<ViolationKind, string> = {
  get 'read-before-write'() { return tr('read before write', 'đọc trước khi ghi'); },
  get 'overwritten-before-read'() { return tr('overwritten before read', 'bị ghi đè trước khi đọc'); },
  get 'missing-copy'() { return tr('copy missing', 'thiếu lần sao chép'); },
};

export interface Violation {
  kind: ViolationKind;
  step: number;
  thread: number;
  iter: number;
  buf: 0 | 1;
  slot: number;
  message: string;
}

export interface Step {
  index: number;
  /** The thread, or -1 when a barrier opens (which involves all of them). */
  thread: number;
  iter: number;
  op: Op | 'release';
  /** For a release: which barrier opened. */
  barrier?: 1 | 2;
  reads?: Read[];
  write?: { buf: 0 | 1; slot: number; value: number; coef: Coef; iter: number };
  pointers?: { inBuf: 0 | 1; outBuf: 0 | 1 };
  violation?: Violation;
}

export interface Slot {
  value: number;
  coef: Coef;
  /** Iteration of the write that put it there: -1 the load, null never written. */
  iter: number | null;
}

export interface Trace {
  cfg: Config;
  schedule: Schedule;
  values: number[];
  steps: Step[];
  violations: Violation[];
  output: number[];
  outputCoef: Coef[];
  reference: number[];
  /** Positions whose output differs from the reference (NaN counts as different). */
  wrong: number[];
  correct: boolean;
  /** The physical buffer every thread's inBuffer_s names at the end (two buffers): 0 = buffer1_s. */
  resultBuf: 0 | 1;
  /** Additions performed in each iteration. */
  additions: number[];
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

const bufName = (cfg: Config, b: number) => (cfg.kind === 'double' ? `buffer${b + 1}_s` : 'buffer_s'); // i18n-ignore: variable names in the kernel

/** Iteration of the write a read of `slot` in iteration k should find (k = iterations for the final store). */
export function expectedWrite(cfg: Config, slot: number, k: number): number {
  if (cfg.kind === 'double') return k - 1;
  if (slot === 0 || k === 0) return -1;
  return Math.min(k - 1, Math.floor(Math.log2(slot)));
}

const writeName = (s: number, it: number) => (it < 0 ? tr(`x${s} (the load)`, `x${s} (lần nạp)`) : tr(`its iteration-${it} value`, `giá trị vòng lặp ${it} của nó`));

/** Runs one block under one thread order and records every operation. */
export function run(cfg: Config, values: number[], schedule: Schedule, seed = 1): Trace {
  const B = cfg.threads;
  const L = iterations(B);
  const progs = Array.from({ length: B }, (_, t) => program(cfg, t));
  const fresh = (): Slot[] => Array.from({ length: B }, () => ({ value: NaN, coef: null, iter: null }));
  const buffers: Slot[][] = cfg.kind === 'double' ? [fresh(), fresh()] : [fresh()];
  const pc = new Array(B).fill(0);
  const waiting: (0 | 1 | 2)[] = new Array(B).fill(0);
  const temp: { value: number; coef: Coef }[] = Array.from({ length: B }, () => ({ value: NaN, coef: null }));
  const inBuf: (0 | 1)[] = new Array(B).fill(0);
  const outBuf: (0 | 1)[] = new Array(B).fill(cfg.kind === 'double' ? 1 : 0);
  const output: number[] = new Array(B).fill(NaN);
  const outputCoef: Coef[] = new Array(B).fill(null);
  const steps: Step[] = [];
  const violations: Violation[] = [];
  const additions = new Array(L).fill(0);
  const rand = mulberry(seed);

  const read = (s: Step, t: number, buf: 0 | 1, slot: number, k: number): Read => {
    const v = buffers[buf][slot];
    const expected = expectedWrite(cfg, slot, k);
    const r: Read = { buf, slot, value: v.value, coef: v.coef, found: v.iter, expected, ok: v.iter === expected, tainted: false };
    if (r.ok) {
      r.tainted = !sameCoef(v.coef, windowCoef(slot, expected + 1, B));
      return r;
    }
    let kind: ViolationKind;
    const noCopy = cfg.kind === 'double' && !cfg.copy;
    if (noCopy && expected >= 0 && !active(slot, expected)) kind = 'missing-copy';
    else if (v.iter === null) kind = noCopy ? 'missing-copy' : 'read-before-write';
    else if (v.iter > expected) kind = 'overwritten-before-read';
    else kind = 'read-before-write';
    const where = tr(`slot ${slot} of ${bufName(cfg, buf)}`, `vị trí ${slot} của ${bufName(cfg, buf)}`);
    const foundText = v.iter === null ? tr('a slot nobody had written yet', 'một vị trí chưa ai ghi') : rangeText(v.coef, true);
    const message =
      kind === 'overwritten-before-read'
        ? tr(`thread ${t} needed ${rangeText(windowCoef(slot, expected + 1, B), true)} from ${where}, but thread ${slot} had already overwritten it with ${foundText}`, `luồng ${t} cần ${rangeText(windowCoef(slot, expected + 1, B), true)} từ ${where}, nhưng luồng ${slot} đã ghi đè nó bằng ${foundText}`)
        : kind === 'missing-copy'
          ? tr(`thread ${t} read ${where}: thread ${slot} was idle in iteration ${expected} and did not copy its value across, so it found ${v.iter === null ? 'a slot nobody ever wrote' : `${foundText}, left from iteration ${v.iter}`}`, `luồng ${t} đọc ${where}: luồng ${slot} rảnh ở vòng lặp ${expected} và không sao chép giá trị của nó sang, nên nó thấy ${v.iter === null ? 'một vị trí chưa ai từng ghi' : `${foundText}, còn lại từ vòng lặp ${v.iter}`}`)
          : tr(`thread ${t} read ${where} before thread ${slot} had written ${writeName(slot, expected)}: it found ${foundText}`, `luồng ${t} đọc ${where} trước khi luồng ${slot} ghi ${writeName(slot, expected)}: nó thấy ${foundText}`);
    const viol: Violation = { kind, step: s.index, thread: t, iter: k, buf, slot, message };
    if (!s.violation) {
      s.violation = viol;
      violations.push(viol);
    }
    return r;
  };

  const exec = (t: number) => {
    const ins = progs[t][pc[t]];
    const s: Step = { index: steps.length, thread: t, iter: ins.iter, op: ins.op };
    steps.push(s);
    const k = ins.iter;
    const stride = 2 ** k;
    switch (ins.op) {
      case 'barrier1':
      case 'barrier2':
        waiting[t] = ins.op === 'barrier1' ? 1 : 2;
        return; // the thread moves on when the barrier opens
      case 'load': {
        const value = t < values.length ? values[t] : 0;
        const coef = t < values.length ? unit(t, B) : zeros(B);
        buffers[0][t] = { value, coef, iter: -1 };
        s.write = { buf: 0, slot: t, value, coef, iter: -1 };
        break;
      }
      case 'read': {
        const a = read(s, t, 0, t, k);
        const b = read(s, t, 0, t - stride, k);
        s.reads = [a, b];
        temp[t] = { value: a.value + b.value, coef: addCoef(a.coef, b.coef) };
        additions[k]++;
        break;
      }
      case 'write': {
        buffers[0][t] = { value: temp[t].value, coef: temp[t].coef, iter: k };
        s.write = { buf: 0, slot: t, value: temp[t].value, coef: temp[t].coef, iter: k };
        break;
      }
      case 'add':
      case 'copy': {
        const a = read(s, t, inBuf[t], t, k);
        const rs = [a];
        let value = a.value;
        let coef = a.coef;
        if (ins.op === 'add') {
          const b = read(s, t, inBuf[t], t - stride, k);
          rs.push(b);
          value = a.value + b.value;
          coef = addCoef(a.coef, b.coef);
          additions[k]++;
        }
        s.reads = rs;
        buffers[outBuf[t]][t] = { value, coef, iter: k };
        s.write = { buf: outBuf[t], slot: t, value, coef, iter: k };
        break;
      }
      case 'swap': {
        const x = inBuf[t];
        inBuf[t] = outBuf[t];
        outBuf[t] = x;
        s.pointers = { inBuf: inBuf[t], outBuf: outBuf[t] };
        break;
      }
      case 'store': {
        const a = read(s, t, cfg.kind === 'double' ? inBuf[t] : 0, t, L);
        s.reads = [a];
        output[t] = a.value;
        outputCoef[t] = a.coef;
        break;
      }
    }
    pc[t]++;
  };

  const live = (t: number) => pc[t] < progs[t].length;
  const runnable = (t: number) => live(t) && !waiting[t];
  const nextIter = (t: number) => progs[t][pc[t]]?.iter;
  const release = () => {
    if (!waiting.some(Boolean)) return;
    if (!waiting.every((w, t) => w || !live(t))) return;
    const t0 = waiting.findIndex(Boolean);
    steps.push({ index: steps.length, thread: -1, iter: progs[t0][pc[t0]].iter, op: 'release', barrier: waiting[t0] as 1 | 2 });
    for (let t = 0; t < B; t++)
      if (waiting[t]) {
        waiting[t] = 0;
        pc[t]++;
      }
  };

  const order = Array.from({ length: B }, (_, i) => i);
  if (schedule === 'reversed') order.reverse();
  let guard = 0;
  while (order.some(live)) {
    if (++guard > 1e5) throw new Error('simulation did not terminate');
    if (schedule === 'lockstep') {
      for (const t of order) if (runnable(t)) exec(t);
    } else if (schedule === 'book-race') {
      const a = RACE.thread;
      if (a < B) while (runnable(a) && nextIter(a) === RACE.iter) exec(a);
      for (const t of order) if (runnable(t)) exec(t);
    } else if (schedule === 'shuffled') {
      const cands = order.filter(runnable);
      if (cands.length) exec(cands[Math.floor(rand() * cands.length)]);
    } else {
      for (const t of order) while (runnable(t)) exec(t);
    }
    release();
  }

  const reference: number[] = [];
  let acc = 0;
  for (let t = 0; t < B; t++) {
    acc += t < values.length ? values[t] : 0;
    reference.push(acc);
  }
  const wrong = reference.flatMap((r, t) => (output[t] === r ? [] : [t]));
  return {
    cfg, schedule, values, steps, violations, output, outputCoef, reference, wrong,
    correct: wrong.length === 0, resultBuf: inBuf[0], additions,
  };
}

// ---------------------------------------------------------------- replay, for the steppers
export interface ThreadState {
  /** The last operation the thread executed, or null before it starts. */
  last: Op | null;
  iter: number;
  waiting: 0 | 1 | 2;
  done: boolean;
  temp: { value: number; coef: Coef } | null;
  inBuf: 0 | 1;
  outBuf: 0 | 1;
}
export interface State {
  buffers: Slot[][];
  threads: ThreadState[];
  output: (number | null)[];
  /** For the history drawing: what each iteration wrote, slot by slot, so far (index 0 = the load). */
  rows: (Slot | null)[][];
}

/** The block after its first `n` steps. */
export function stateAt(trace: Trace, n: number): State {
  const { cfg } = trace;
  const B = cfg.threads;
  const L = iterations(B);
  const fresh = (): Slot[] => Array.from({ length: B }, () => ({ value: NaN, coef: null, iter: null }));
  const buffers: Slot[][] = cfg.kind === 'double' ? [fresh(), fresh()] : [fresh()];
  const threads: ThreadState[] = Array.from({ length: B }, () => ({
    last: null, iter: -1, waiting: 0, done: false, temp: null, inBuf: 0, outBuf: cfg.kind === 'double' ? 1 : 0,
  }));
  const output: (number | null)[] = new Array(B).fill(null);
  const rows: (Slot | null)[][] = Array.from({ length: L + 1 }, () => new Array(B).fill(null));
  for (let i = 0; i < Math.min(n, trace.steps.length); i++) {
    const s = trace.steps[i];
    if (s.thread < 0) {
      for (const th of threads) th.waiting = 0;
      continue;
    }
    const th = threads[s.thread];
    th.last = s.op as Op;
    th.iter = s.iter;
    if (s.op === 'barrier1' || s.op === 'barrier2') th.waiting = s.op === 'barrier1' ? 1 : 2;
    if (s.op === 'read' && s.reads) th.temp = { value: s.reads[0].value + s.reads[1].value, coef: addCoef(s.reads[0].coef, s.reads[1].coef) };
    if (s.write) {
      const slot = { value: s.write.value, coef: s.write.coef, iter: s.write.iter };
      buffers[s.write.buf][s.write.slot] = slot;
      rows[s.write.iter + 1][s.write.slot] = slot;
    }
    if (s.pointers) {
      th.inBuf = s.pointers.inBuf;
      th.outBuf = s.pointers.outBuf;
    }
    if (s.op === 'store') {
      output[s.thread] = s.reads![0].value;
      th.done = true;
    }
  }
  return { buffers, threads, output, rows };
}

/** Index just after the first violation, or null. */
export const firstFailure = (trace: Trace): number | null => (trace.violations.length ? trace.violations[0].step + 1 : null);

/** Index just after the next barrier release at or after step `from`, or the end. */
export function nextRelease(trace: Trace, from: number): number {
  for (let i = from; i < trace.steps.length; i++) if (trace.steps[i].op === 'release') return i + 1;
  return trace.steps.length;
}

// ---------------------------------------------------------------- Fig. 11.2, level by level
export interface Level {
  /** Null for the inputs. */
  stride: number | null;
  values: number[];
  coef: number[][];
  /** Positions that performed an addition to reach this level. */
  added: boolean[];
}

/** The array after each stride, all positions in lockstep (Fig. 11.2; exercise 1). */
export function levels(values: number[]): Level[] {
  const B = values.length;
  const out: Level[] = [{ stride: null, values: [...values], coef: values.map((_, i) => unit(i, B)), added: new Array(B).fill(false) }];
  for (const stride of strides(B)) {
    const prev = out[out.length - 1];
    out.push({
      stride,
      values: prev.values.map((v, i) => (i >= stride ? v + prev.values[i - stride] : v)),
      coef: prev.coef.map((c, i) => (i >= stride ? c.map((x, k) => x + prev.coef[i - stride][k]) : c)),
      added: prev.values.map((_, i) => i >= stride),
    });
  }
  return out;
}

/** Positions whose value is final after `done` iterations: the first 2^done. */
export const finalCount = (B: number, done: number) => Math.min(B, 2 ** done);

/** Additions in the iteration with this stride (B - stride, p. 266), and in all. */
export const additionsAt = (B: number, stride: number) => Math.max(0, B - stride);
export const totalAdditions = (B: number) => strides(B).reduce((s, st) => s + additionsAt(B, st), 0);
/** The naive scan of p. 254: N(N-1)/2. */
export const naiveAdditions = (N: number) => (N * (N - 1)) / 2;

export interface TreeNode {
  /** 0 = the inputs, k = after iteration k - 1 (stride 2^(k-1)). */
  level: number;
  pos: number;
}
/** The reduction tree behind output `i` (heavy lines of Fig. 11.2 for y15). */
export function treeOf(i: number, B: number): { nodes: TreeNode[]; edges: [TreeNode, TreeNode][] } {
  const L = iterations(B);
  const nodes: TreeNode[] = [];
  const edges: [TreeNode, TreeNode][] = [];
  const seen = new Set<string>();
  const visit = (level: number, pos: number) => {
    const key = `${level}:${pos}`;
    if (seen.has(key)) return;
    seen.add(key);
    nodes.push({ level, pos });
    if (level === 0) return;
    const stride = 2 ** (level - 1);
    edges.push([{ level: level - 1, pos }, { level, pos }]);
    visit(level - 1, pos);
    if (pos >= stride) {
      edges.push([{ level: level - 1, pos: pos - stride }, { level, pos }]);
      visit(level - 1, pos - stride);
    }
  };
  visit(L, i);
  return { nodes, edges };
}
/** The adders in that tree that actually add (a node with two parents). */
export const treeAdders = (i: number, B: number) => treeOf(i, B).nodes.filter((n) => n.level > 0 && n.pos >= 2 ** (n.level - 1)).length;

// ---------------------------------------------------------------- warps (exercise 2)
export type WarpState = 'full' | 'divergent' | 'idle';
export interface Warp {
  warp: number;
  /** Lanes that pass threadIdx.x >= stride. */
  activeLanes: number;
  width: number;
  state: WarpState;
}
/** Which warps are full, divergent or idle for the test threadIdx.x >= stride. */
export function warpStates(B: number, width: number, stride: number): Warp[] {
  const n = Math.ceil(B / width);
  return Array.from({ length: n }, (_, w) => {
    const lanes = Math.min(width, B - w * width);
    let on = 0;
    for (let l = 0; l < lanes; l++) if (w * width + l >= stride) on++;
    return { warp: w, activeLanes: on, width: lanes, state: on === 0 ? 'idle' : on === lanes ? 'full' : 'divergent' };
  });
}
/** Over the whole loop: iterations with a divergent warp, and (warp, iteration) pairs that diverge. */
export function divergence(B: number, width: number): { iterations: number; pairs: number; strides: number[] } {
  const ss = strides(B).filter((s) => warpStates(B, width, s).some((w) => w.state === 'divergent'));
  return {
    iterations: ss.length,
    pairs: strides(B).reduce((n, s) => n + warpStates(B, width, s).filter((w) => w.state === 'divergent').length, 0),
    strides: ss,
  };
}

// ---------------------------------------------------------------- the two buffers (section 11.3)
/** Pointer swaps of Fig. 11.5 for a block of B threads: which physical buffer is in and out per iteration. */
export function swaps(B: number): { iter: number; stride: number; inBuf: 0 | 1; outBuf: 0 | 1 }[] {
  return strides(B).map((stride, k) => ({ iter: k, stride, inBuf: (k % 2) as 0 | 1, outBuf: ((k + 1) % 2) as 0 | 1 }));
}
/** The physical buffer inBuffer_s names after the loop: 1 for buffer1_s, 2 for buffer2_s. */
export const resultBuffer = (B: number): 1 | 2 => (iterations(B) % 2 === 0 ? 1 : 2);
/** Shared memory per block for SEG_SIZE floats: Fig. 11.3 one buffer, Fig. 11.5 two. */
export const sharedBytes = (seg: number, kind: Kind) => (kind === 'double' ? 2 : 1) * seg * 4;
/** Barriers one block executes: two per iteration in Fig. 11.3, one in Fig. 11.5 (pp. 257, 261). */
export const barriersPerScan = (B: number, kind: Kind) => (kind === 'double' ? 1 : 2) * iterations(B);

// ---------------------------------------------------------------- the source the steppers print
export interface Source {
  name: string;
  code: string;
  /** 1-based lines of each operation, counting the header as line 1 (the book's numbering). */
  line: Partial<Record<Op, number[]>>;
}

/** The kernel for a configuration, as the snippet files hold it (named variants) or would hold it. */
export function kernelSource(cfg: Omit<Config, 'threads'>): Source {
  const rows: string[] = [];
  const line: Partial<Record<Op, number[]>> = {};
  const mark = (op: Op) => (line[op] = [...(line[op] ?? []), rows.length]);
  const add = (text: string, op?: Op) => {
    rows.push(text);
    if (op) mark(op);
  };
  const name = kernelName(cfg);
  if (cfg.kind === 'single') {
    add(`__global__ void ${name}(float *input, float *output, unsigned int N) {`); // i18n-ignore: kernel source
    add('    __shared__ float buffer_s[SEG_SIZE];'); // i18n-ignore: kernel source
    add('    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;');
    add('    if(i < N) {');
    add('        buffer_s[threadIdx.x] = input[i];', 'load');
    add('    } else {'); // i18n-ignore: kernel source
    add('        buffer_s[threadIdx.x] = 0.0f;', 'load');
    add('    }');
    add('    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2) {'); // i18n-ignore: kernel source
    if (cfg.barrier1) add('        __syncthreads();', 'barrier1');
    if (cfg.temp) {
      add('        float temp;'); // i18n-ignore: kernel source
      add('        if(threadIdx.x >= stride)');
      add('            temp = buffer_s[threadIdx.x] + buffer_s[threadIdx.x-stride];', 'read');
      if (cfg.barrier2) add('        __syncthreads();', 'barrier2');
      add('        if(threadIdx.x >= stride)');
      add('            buffer_s[threadIdx.x] = temp;', 'write');
    } else {
      add('        if(threadIdx.x >= stride)');
      add('            buffer_s[threadIdx.x] += buffer_s[threadIdx.x-stride];', 'read');
      mark('write');
      if (cfg.barrier2) add('        __syncthreads();', 'barrier2');
    }
    add('    }');
    add('    if(i < N) {');
    add('        output[i] = buffer_s[threadIdx.x];', 'store');
    add('    }');
    add('}');
  } else {
    add(`__global__ void ${name}(float* input, float* output, unsigned int N) {`); // i18n-ignore: kernel source
    add('    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;');
    add('    __shared__ float buffer1_s[SEG_SIZE];'); // i18n-ignore: kernel source
    add('    __shared__ float buffer2_s[SEG_SIZE];'); // i18n-ignore: kernel source
    add('    float* inBuffer_s = buffer1_s;');
    add('    float* outBuffer_s = buffer2_s;');
    add('    inBuffer_s[threadIdx.x] = (i < N)?input[i]:0.0f;', 'load');
    add('    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2) {'); // i18n-ignore: kernel source
    if (cfg.barrier1) add('        __syncthreads();', 'barrier1');
    add('        if(threadIdx.x >= stride) {');
    add('            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x]', 'add');
    add('                                       + inBuffer_s[threadIdx.x - stride];', 'add'); // i18n-ignore: kernel source
    if (cfg.copy) {
      add('        } else {'); // i18n-ignore: kernel source
      add('            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];', 'copy');
    }
    add('        }');
    add('        float* temp = inBuffer_s;', 'swap');
    add('        inBuffer_s = outBuffer_s;', 'swap');
    add('        outBuffer_s = temp;', 'swap');
    add('    }');
    add('    if(i < N) {');
    add('        output[i] = inBuffer_s[threadIdx.x];', 'store');
    add('    }');
    add('}');
  }
  return { name, code: rows.join('\n'), line };
}

/** The function name each variant has in the snippet files; others are only generated. */
export function kernelName(cfg: Omit<Config, 'threads'>): string {
  if (cfg.kind === 'double') {
    if (!cfg.barrier1) return 'scan_kernel';
    return cfg.copy ? 'double_buffer_scan_kernel' : 'no_copy_double_buffer_scan_kernel';
  }
  if (cfg.temp) {
    if (cfg.barrier1 && cfg.barrier2) return 'kogge_stone_scan_kernel';
    if (cfg.barrier1) return 'no_barrier2_scan_kernel';
    if (cfg.barrier2) return 'no_barrier1_scan_kernel';
    return 'scan_kernel';
  }
  return cfg.barrier1 && cfg.barrier2 ? 'direct_add_scan_kernel' : 'scan_kernel';
}

/** The variants the snippet files hold, by name. */
export const FILE_VARIANTS: { file: string; cfg: Omit<Config, 'threads'> }[] = [
  { file: 's112_kogge_stone.cu', cfg: FIG_11_3 },
  { file: 's112_kogge_stone.cu', cfg: FIG_11_5 },
  { file: 's112_race_variants.cu', cfg: { ...FIG_11_3, barrier2: false } },
  { file: 's112_race_variants.cu', cfg: { ...FIG_11_3, barrier1: false } },
  { file: 's112_race_variants.cu', cfg: { ...FIG_11_3, temp: false } },
  { file: 's112_race_variants.cu', cfg: { ...FIG_11_5, copy: false } },
];
