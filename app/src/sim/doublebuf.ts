// Chapter 6, section 6.7: double buffering. A thread-level model of the book's
// exchange loop (p. 146: read a neighbour's slot, barrier, write your own slot,
// barrier; p. 147: two buffers, one barrier, swap). It runs the stepper, the
// outcome table and every frame of the narrated scene "Two buffers, one barrier".
//
// The loop, made concrete (labs/ch06/snippets/s67_exchange.cu holds it as CUDA):
// P threads in a ring; thread t starts with the value t in its slot; in every
// iteration it reads the slot of thread t + 1 (mod P) and writes that value plus
// 10 into its own slot. After n iterations thread t should hold 10n + (t + n) mod P:
// the tens digit counts the steps, the units digit names the thread the value
// started at. The book leaves the computation and the declarations out (notes
// item 15); they are ours.
//
// Sources for the rules:
//  - the book, section 6.7, pp. 146 to 147: the barrier after the write guards a
//    read-after-write (true) dependence, the one after the read a write-after-read
//    (false) dependence that exists only because the slot is reused; two buffers
//    remove the false one; the swap at the end of the body. Page 147 says the
//    "second" barrier can go; the analysis and the code mean the first, the one
//    after the read (notes item 2, ch06-facts E1).
//  - chapter 5's model of the tiled kernel (sim/tiled.ts), whose thread orders and
//    step granularity this copies so that the two tables read the same way.
//
// What it leaves out (the lesson says so beside the stepper):
//  - A thread's unit of progress is one whole source-level operation (read, write,
//    barrier, swap). Real threads interleave at the instruction level, in warps.
//  - "Lockstep" is an idealised single warp; a real block has several warps and no
//    promised order between them.
//  - No timing: the model says which values a thread can see, not how long anything
//    takes.
//
// Erasable TypeScript only, and no imports: the browser checks import this file
// directly in Node.

import { tr } from '../lib/locale.ts';

export type Variant =
  | 'single-both'
  | 'single-no-read-barrier'
  | 'single-no-write-barrier'
  | 'double'
  | 'double-no-swap'
  | 'double-no-barrier';

export const VARIANTS: Variant[] = [
  'single-both',
  'single-no-read-barrier',
  'single-no-write-barrier',
  'double',
  'double-no-swap',
  'double-no-barrier',
];

export const VARIANT_LABEL: Record<Variant, string> = {
  get 'single-both'() { return tr('one buffer, both barriers', 'một buffer, cả hai rào chắn'); },
  get 'single-no-read-barrier'() { return tr('one buffer, no barrier after the read', 'một buffer, không có rào chắn sau lần đọc'); },
  get 'single-no-write-barrier'() { return tr('one buffer, no barrier after the write', 'một buffer, không có rào chắn sau lần ghi'); },
  get double() { return tr('two buffers, one barrier', 'hai buffer, một rào chắn'); },
  get 'double-no-swap'() { return tr('two buffers, swap forgotten', 'hai buffer, quên hoán đổi'); },
  get 'double-no-barrier'() { return tr('two buffers, no barrier', 'hai buffer, không có rào chắn'); },
};

export type Schedule = 'lockstep' | 'in-order' | 'reversed' | 'shuffled';
export const SCHEDULES: Schedule[] = ['lockstep', 'in-order', 'reversed', 'shuffled'];
export const SCHEDULE_LABEL: Record<Schedule, string> = {
  get lockstep() { return tr('all threads in lockstep', 'mọi luồng chạy đồng bộ từng bước (lockstep)'); },
  get 'in-order'() { return tr('each thread runs as far as it can, thread 0 first', 'mỗi luồng chạy xa nhất có thể, luồng 0 trước'); },
  get reversed() { return tr('each thread runs as far as it can, last thread first', 'mỗi luồng chạy xa nhất có thể, luồng cuối trước'); },
  get shuffled() { return tr('random interleaving', 'xen kẽ ngẫu nhiên'); },
};

export interface Config {
  /** Threads in the block (the stepper offers 4 to 8). */
  threads: number;
  /** Trips round the loop. */
  iterations: number;
  variant: Variant;
}

export type Op = 'read' | 'barrierRead' | 'write' | 'barrierWrite' | 'swap';
export type ViolationKind = 'read-after-write' | 'write-after-read' | 'wrong-buffer';

export const KIND_LABEL: Record<ViolationKind, string> = {
  'read-after-write': 'read-after-write',
  'write-after-read': 'write-after-read',
  get 'wrong-buffer'() { return tr('wrong buffer (no race)', 'sai buffer (không có race)'); },
};

export const isDouble = (v: Variant) => v.startsWith('double');

/** The operations of one trip round the loop, in source order. */
export function plan(v: Variant): Op[] {
  switch (v) {
    case 'single-both':
      return ['read', 'barrierRead', 'write', 'barrierWrite'];
    case 'single-no-read-barrier':
      return ['read', 'write', 'barrierWrite'];
    case 'single-no-write-barrier':
      return ['read', 'barrierRead', 'write'];
    case 'double':
      return ['read', 'write', 'barrierWrite', 'swap'];
    case 'double-no-swap':
      return ['read', 'write', 'barrierWrite'];
    case 'double-no-barrier':
      return ['read', 'write', 'swap'];
  }
}

// The source lines the stepper shows. For 'single-both' and 'double' they are the
// loop of labs/ch06/snippets/s67_exchange.cu (a unit test checks); the other
// variants drop one line.
const LINES: Record<Op | 'head' | 'tail', { single: string; double: string }> = {
  head: { single: 'for (unsigned int i = 0; i < n; ++i) {', double: 'for (unsigned int i = 0; i < n; ++i) {' },
  read: { single: '    float v = buffer[other];', double: '    float v = inBuffer[other];' },
  barrierRead: { single: '    __syncthreads();            // after the read', double: '    __syncthreads();            // after the read' }, // i18n-ignore: kernel source shown as code
  write: { single: '    buffer[me] = v + 10.0f;', double: '    outBuffer[me] = v + 10.0f;' },
  barrierWrite: { single: '    __syncthreads();            // after the write', double: '    __syncthreads();            // after the write' }, // i18n-ignore: kernel source shown as code
  swap: { single: '    swapPointers(inBuffer, outBuffer);', double: '    swapPointers(inBuffer, outBuffer);' }, // i18n-ignore: kernel source shown as code
  tail: { single: '}', double: '}' },
};

/** The loop's source for a variant, and the 1-based line each operation sits on. */
export function loopSource(v: Variant): { code: string; line: Record<Op, number> } {
  const k = isDouble(v) ? 'double' : 'single';
  const ops = plan(v);
  const rows = [LINES.head[k], ...ops.map((o) => LINES[o][k]), LINES.tail[k]];
  const line = {} as Record<Op, number>;
  ops.forEach((o, i) => (line[o] = i + 2));
  return { code: rows.join('\n'), line };
}

export interface Read {
  buf: 0 | 1;
  slot: number;
  value: number;
  /** Iteration whose write produced the value found (0 = the initial value). */
  version: number;
  /** The version this read should have found. */
  expected: number;
  ok: boolean;
}

export interface Step {
  index: number;
  /** The thread, or -1 when a barrier opens (which involves all of them). */
  thread: number;
  iter: number;
  op: Op | 'release';
  read?: Read;
  write?: { buf: 0 | 1; slot: number; value: number; version: number };
  /** After a swap: the thread's pointers. */
  pointers?: { inBuf: 0 | 1; outBuf: 0 | 1 };
  violation?: Violation;
}

export interface Violation {
  kind: ViolationKind;
  step: number;
  thread: number;
  iter: number;
  message: string;
}

export interface Slot {
  value: number;
  /** -1 = never written. */
  version: number;
}

export interface Trace {
  cfg: Config;
  schedule: Schedule;
  steps: Step[];
  violations: Violation[];
  /** What each thread hands back at the end: buffer[me], or inBuffer[me] after the last swap. */
  output: number[];
  reference: number[];
  /** Threads whose output differs from the reference. */
  wrong: number[];
  correct: boolean;
}

export const initialValue = (t: number) => t;
export const neighbour = (t: number, P: number) => (t + 1) % P;
/** Thread t's value after n iterations if nothing goes wrong. */
export const expectedValue = (t: number, n: number, P: number) => 10 * n + initialValue((t + n) % P);

/** Where a thread's pointers point in iteration k under a variant's code (0 = buffer A, 1 = buffer B). */
export function pointersAt(v: Variant, k: number): { inBuf: 0 | 1; outBuf: 0 | 1 } {
  if (!isDouble(v)) return { inBuf: 0, outBuf: 0 };
  if (v === 'double-no-swap') return { inBuf: 0, outBuf: 1 };
  return k % 2 === 0 ? { inBuf: 0, outBuf: 1 } : { inBuf: 1, outBuf: 0 };
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

/** Runs one block of the loop under one thread order and records every operation. */
export function run(cfg: Config, schedule: Schedule, seed = 1): Trace {
  const P = cfg.threads;
  const N = cfg.iterations;
  const ops = plan(cfg.variant);
  const total = ops.length * N;
  const buffers: Slot[][] = [
    Array.from({ length: P }, (_, t) => ({ value: initialValue(t), version: 0 })),
    Array.from({ length: P }, () => ({ value: NaN, version: -1 })),
  ];
  const pc = new Array(P).fill(0);
  const waiting = new Array(P).fill(false);
  const reg = new Array(P).fill(NaN);
  const inBuf: (0 | 1)[] = new Array(P).fill(0);
  const outBuf: (0 | 1)[] = new Array(P).fill(isDouble(cfg.variant) ? 1 : 0);
  const steps: Step[] = [];
  const violations: Violation[] = [];
  const rand = mulberry(seed);
  const name = (t: number) => tr(`thread ${t}`, `luồng ${t}`);
  const bufName = (b: number) => (isDouble(cfg.variant) ? (b === 0 ? 'A' : 'B') : '');

  const exec = (t: number) => {
    const iter = Math.floor(pc[t] / ops.length);
    const op = ops[pc[t] % ops.length];
    const s: Step = { index: steps.length, thread: t, iter, op };
    steps.push(s);
    if (op === 'barrierRead' || op === 'barrierWrite') {
      waiting[t] = true;
      return;
    }
    if (op === 'read') {
      const other = neighbour(t, P);
      const buf = isDouble(cfg.variant) ? inBuf[t] : 0;
      const slot = buffers[buf][other];
      // The value this read needs was written by `other` in the previous iteration
      // (or is the initial value), into that thread's out buffer at the time.
      const producerBuf = iter === 0 ? 0 : pointersAt(cfg.variant, iter - 1).outBuf;
      const r: Read = { buf, slot: other, value: slot.value, version: slot.version, expected: iter, ok: true };
      const at = tr(`${bufName(buf) ? 'buffer ' + bufName(buf) + ', ' : ''}slot ${other}`, `${bufName(buf) ? 'buffer ' + bufName(buf) + ', ' : ''}vị trí ${other}`);
      let v: Violation | undefined;
      if (buf !== producerBuf) {
        v = {
          kind: 'wrong-buffer', step: s.index, thread: t, iter,
          message: tr(`${name(t)} read ${at} in iteration ${iter}, but iteration ${iter - 1} wrote its results to buffer ${bufName(producerBuf)}: this value will never arrive here`, `${name(t)} đọc ${at} ở vòng lặp ${iter}, nhưng vòng lặp ${iter - 1} đã ghi kết quả vào buffer ${bufName(producerBuf)}: giá trị này sẽ không bao giờ tới đây`),
        };
      } else if (slot.version < iter) {
        v = {
          kind: 'read-after-write', step: s.index, thread: t, iter,
          message: tr(`${name(t)} read ${at} in iteration ${iter} before ${name(other)} had written that iteration's value into it`, `${name(t)} đọc ${at} ở vòng lặp ${iter} trước khi ${name(other)} ghi giá trị của vòng lặp đó vào`),
        };
      } else if (slot.version > iter) {
        v = {
          kind: 'write-after-read', step: s.index, thread: t, iter,
          message: tr(`${name(t)} still needed ${iter === 0 ? 'the initial value' : `iteration ${iter - 1}'s value`} in ${at}, but ${name(other)} had already overwritten it in iteration ${slot.version - 1}`, `${name(t)} vẫn cần ${iter === 0 ? 'giá trị ban đầu' : `giá trị của vòng lặp ${iter - 1}`} trong ${at}, nhưng ${name(other)} đã ghi đè nó ở vòng lặp ${slot.version - 1}`),
        };
      }
      if (v) {
        r.ok = false;
        s.violation = v;
        violations.push(v);
      }
      s.read = r;
      reg[t] = slot.value;
    } else if (op === 'write') {
      const buf = isDouble(cfg.variant) ? outBuf[t] : 0;
      const value = reg[t] + 10;
      buffers[buf][t] = { value, version: iter + 1 };
      s.write = { buf, slot: t, value, version: iter + 1 };
    } else if (op === 'swap') {
      const x = inBuf[t];
      inBuf[t] = outBuf[t];
      outBuf[t] = x;
      s.pointers = { inBuf: inBuf[t], outBuf: outBuf[t] };
    }
    pc[t]++;
  };

  const live = (t: number) => pc[t] < total;
  const runnable = (t: number) => live(t) && !waiting[t];
  const release = () => {
    if (!waiting.some(Boolean)) return;
    if (!waiting.every((w, t) => w || !live(t))) return;
    const t0 = waiting.findIndex(Boolean);
    steps.push({ index: steps.length, thread: -1, iter: Math.floor(pc[t0] / ops.length), op: 'release' });
    for (let t = 0; t < P; t++)
      if (waiting[t]) {
        waiting[t] = false;
        pc[t]++;
      }
  };

  const order = Array.from({ length: P }, (_, i) => i);
  if (schedule === 'reversed') order.reverse();
  let guard = 0;
  while (order.some(live)) {
    if (++guard > 1e5) throw new Error('simulation did not terminate');
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

  const output = order
    .slice()
    .sort((a, b) => a - b)
    .map((t) => buffers[isDouble(cfg.variant) ? inBuf[t] : 0][t].value);
  const reference = Array.from({ length: P }, (_, t) => expectedValue(t, N, P));
  const wrong = reference.flatMap((r, t) => (output[t] === r ? [] : [t]));
  return { cfg, schedule, steps, violations, output, reference, wrong, correct: wrong.length === 0 };
}

export interface ThreadState {
  iter: number;
  /** The operation the thread executed last, or null before it starts. */
  last: Op | null;
  waiting: boolean;
  done: boolean;
  /** The value in its register v, NaN before the first read. */
  v: number;
  inBuf: 0 | 1;
  outBuf: 0 | 1;
}

export interface State {
  buffers: Slot[][];
  threads: ThreadState[];
}

/** The block after its first `n` steps, for the stepper and the scene. */
export function stateAt(trace: Trace, n: number): State {
  const P = trace.cfg.threads;
  const total = plan(trace.cfg.variant).length * trace.cfg.iterations;
  const dbl = isDouble(trace.cfg.variant);
  const buffers: Slot[][] = [
    Array.from({ length: P }, (_, t) => ({ value: initialValue(t), version: 0 })),
    Array.from({ length: P }, () => ({ value: NaN, version: -1 })),
  ];
  const threads: ThreadState[] = Array.from({ length: P }, () => ({
    iter: 0, last: null, waiting: false, done: false, v: NaN, inBuf: 0, outBuf: dbl ? 1 : 0,
  }));
  const executed = new Array(P).fill(0);
  for (let i = 0; i < Math.min(n, trace.steps.length); i++) {
    const s = trace.steps[i];
    if (s.thread < 0) {
      for (let t = 0; t < P; t++)
        if (threads[t].waiting) {
          threads[t].waiting = false;
          executed[t]++;
          threads[t].done = executed[t] >= total;
        }
      continue;
    }
    const th = threads[s.thread];
    th.iter = s.iter;
    th.last = s.op as Op;
    if (s.op === 'barrierRead' || s.op === 'barrierWrite') {
      th.waiting = true;
      continue;
    }
    if (s.read) th.v = s.read.value;
    if (s.write) buffers[s.write.buf][s.write.slot] = { value: s.write.value, version: s.write.version };
    if (s.pointers) {
      th.inBuf = s.pointers.inBuf;
      th.outBuf = s.pointers.outBuf;
    }
    executed[s.thread]++;
    th.done = executed[s.thread] >= total;
  }
  return { buffers, threads };
}

/** Index of the step just after the first violation (the stepper's "go to the first failure"), or null. */
export function firstFailure(trace: Trace): number | null {
  return trace.violations.length ? trace.violations[0].step + 1 : null;
}

/** Index just after the first barrier release at or after step `from`, or the end. */
export function nextRelease(trace: Trace, from: number): number {
  for (let i = from; i < trace.steps.length; i++) if (trace.steps[i].op === 'release') return i + 1;
  return trace.steps.length;
}

// ---------------------------------------------------------------- the tiled kernel's cost
/** Shared memory for the tiles of a tiled matmul: one M and one N tile of T x T floats per buffer. */
export const tileBytes = (T: number, buffers: 1 | 2) => buffers * 2 * T * T * 4;
