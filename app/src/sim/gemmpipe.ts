// Chapter 15, section 15.8: double buffering and software pipelining of the tile
// loop. One model for every interactive of the section and every frame of the
// narrated scene "Loading the next tile while computing this one":
//  1. the instruction stream of Fig. 15.12(a) with each barrier's kind (true or
//     false dependence), for the classification exercise;
//  2. two-lane schedules (memory hardware, compute hardware) for the three loops of
//     Figs. 15.12 and 15.13, for the Gantt explorer and the scene;
//  3. which physical buffer is "curr" and which is "next" in each iteration of
//     Fig. 15.14;
//  4. a thread-level model of the pipelined loop, for the stepper and the outcome
//     table: P threads, each loads one slot of a tile per iteration and adds up every
//     slot of the current buffer.
//
// From the book (notes/ch15.md, section 15.8, pp. 364 to 368):
//  - the barrier between loading a pair and computing with it guards a true
//    dependence; the one between computing and loading the next pair a false one,
//    which exists only because the next pair goes into the same memory (p. 366);
//  - two buffers remove the false one; even then the load of pair i + 1 and the
//    compute of pair i sit in different iterations and cannot be interleaved, so the
//    loop is rotated: iteration i computes pair i and loads pair i + 1 (p. 366);
//  - Fig. 15.13 draws the two loops as a timing diagram, about 8 phase widths against
//    5 for four pairs (schematic, not measured);
//  - Fig. 15.14: prologue load of pair 0 and a barrier; the loop runs to numTiles - 1;
//    no barrier between mm and loadTile; one barrier at the end of the iteration;
//    one final mm after the loop.
// Ours (labelled in the lesson): the single barrier also stops a thread from loading
// into the buffer a slower thread is still reading (notes, [inferred]); the toy
// computation (each thread adds up the slots); the load and compute times as
// sliders; the variants and thread orders.
//
// What the schedules leave out: fixed per-tile times, perfect overlap of the two
// lanes within an iteration, no caches, one block per SM, and the fact that the real
// interleaving is the compiler's instruction schedule (p. 367).
// What the thread model leaves out: a thread advances one source-level operation at
// a time (a whole mm is one step); "lockstep" is an idealised single warp; no timing.
//
// Erasable TypeScript only, and no imports: the browser checks import this file
// directly in Node.

// ---------------------------------------------------------------- 1. Fig. 15.12(a)
import { tr } from '../lib/locale.ts';

export type DepKind = 'true' | 'false';

export interface Instr {
  kind: 'load' | 'compute' | 'sync';
  /** The tile pair a load or compute works on; for a barrier, the pair before it. */
  tile: number;
  label: string;
  /** Barriers only: index among the barriers, kind, and the reason in one sentence. */
  barrier?: number;
  dep?: DepKind;
  why?: string;
}

/** Fig. 15.12(a): load, barrier, compute, barrier, for n pairs (no barrier after the last compute). */
export function baselineStream(n: number): Instr[] {
  const out: Instr[] = [];
  let b = 0;
  for (let i = 0; i < n; i++) {
    out.push({ kind: 'load', tile: i, label: tr(`Load A${i} and B${i}`, `Nạp A${i} và B${i}`) });
    out.push({
      kind: 'sync', tile: i, label: '__syncthreads()', barrier: b++, dep: 'true',
      why: tr(
        `Compute C${i} reads elements of A${i} and B${i} that other threads loaded: the data has to arrive before anyone uses it.`,
        `Tính C${i} đọc các phần tử của A${i} và B${i} mà các luồng khác đã nạp: dữ liệu phải đến nơi trước khi ai đó dùng nó.`,
      ),
    });
    out.push({ kind: 'compute', tile: i, label: tr(`Compute C${i}`, `Tính C${i}`) });
    if (i < n - 1)
      out.push({
        kind: 'sync', tile: i, label: '__syncthreads()', barrier: b++, dep: 'false',
        why: tr(
          `Loading A${i + 1} and B${i + 1} needs nothing from computing C${i}. It waits only because it would overwrite A${i} and B${i}, in the same shared arrays, while other threads still read them.`,
          `Việc nạp A${i + 1} và B${i + 1} không cần gì từ việc tính C${i}. Nó chờ chỉ vì nó sẽ ghi đè A${i} và B${i}, trong cùng các mảng chia sẻ, khi các luồng khác vẫn đang đọc chúng.`,
        ),
      });
  }
  return out;
}

/** The barriers of Fig. 15.12(a) alone. For n = 4: seven, four true and three false. */
export const baselineBarriers = (n: number) => baselineStream(n).filter((x) => x.kind === 'sync');

// ---------------------------------------------------------------- 2. Figs. 15.12 and 15.13 as schedules
export type Scheme = 'two-barriers' | 'double-buffered' | 'pipelined';
export const SCHEMES: Scheme[] = ['two-barriers', 'double-buffered', 'pipelined'];
export const SCHEME_LABEL: Record<Scheme, string> = {
  get 'two-barriers'() { return tr('One buffer, two barriers (Fig. 15.12a)', 'Một buffer, hai rào chắn (Fig. 15.12a)'); },
  get 'double-buffered'() { return tr('Two buffers, not rotated (Fig. 15.12b)', 'Hai buffer, không xoay vòng (Fig. 15.12b)'); },
  get pipelined() { return tr('Two buffers, pipelined (Fig. 15.12c)', 'Hai buffer, chạy đường ống (Fig. 15.12c)'); },
};

export type Lane = 'mem' | 'comp';
export interface Box {
  lane: Lane;
  /** Tile pair. */
  tile: number;
  start: number;
  end: number;
  /** The buffer the pair lives in: 0 or 1; always 0 with one buffer. */
  buf: 0 | 1;
}
export interface Bar {
  start: number;
  end: number;
  /** Guards the arrival of loaded data (true dependence). */
  arrival: boolean;
  /** Guards reuse: a later load writes the buffer read before this barrier (false dependence). */
  reuse: boolean;
}
export interface Span {
  label: string;
  start: number;
  end: number;
}
export interface Timeline {
  scheme: Scheme;
  n: number;
  L: number;
  C: number;
  s: number;
  boxes: Box[];
  barriers: Bar[];
  spans: Span[];
  total: number;
  /** Time each lane is busy, and that as a fraction of the total. */
  memBusy: number;
  compBusy: number;
  memUtil: number;
  compUtil: number;
}

/**
 * The schedule of `n` tile pairs with load time L and compute time C per pair and
 * barrier cost s. In the pipelined loop the load of pair i + 1 and the compute of pair
 * i start together at the top of iteration i and the iteration lasts max(L, C).
 */
export function timeline(scheme: Scheme, n: number, L: number, C: number, s = 0): Timeline {
  const boxes: Box[] = [];
  const barriers: Bar[] = [];
  const spans: Span[] = [];
  const dbl = scheme !== 'two-barriers';
  const buf = (k: number): 0 | 1 => (dbl ? ((k & 1) as 0 | 1) : 0);
  let t = 0;
  const bar = (arrival: boolean, reuse: boolean) => {
    barriers.push({ start: t, end: t + s, arrival, reuse });
    t += s;
  };
  if (scheme === 'pipelined') {
    boxes.push({ lane: 'mem', tile: 0, start: 0, end: L, buf: 0 });
    t = L;
    bar(true, false);
    spans.push({ label: 'prologue', start: 0, end: t }); // i18n-ignore: span label parsed by PipelineScene (startsWith/split), which shows its own text
    for (let i = 0; i < n - 1; i++) {
      const t0 = t;
      boxes.push({ lane: 'comp', tile: i, start: t, end: t + C, buf: buf(i) });
      boxes.push({ lane: 'mem', tile: i + 1, start: t, end: t + L, buf: buf(i + 1) });
      t += Math.max(L, C);
      // A later load (pair i + 2, in iteration i + 1) writes the buffer computed on here.
      bar(true, i + 2 < n);
      spans.push({ label: `iteration ${i}`, start: t0, end: t }); // i18n-ignore: span label parsed by PipelineScene (startsWith/split), which shows its own text
    }
    const t0 = t;
    boxes.push({ lane: 'comp', tile: n - 1, start: t, end: t + C, buf: buf(n - 1) });
    t += C;
    spans.push({ label: n > 1 ? 'after the loop' : 'compute', start: t0, end: t }); // i18n-ignore: span label parsed by PipelineScene (startsWith/split), which shows its own text
  } else {
    for (let i = 0; i < n; i++) {
      const t0 = t;
      boxes.push({ lane: 'mem', tile: i, start: t, end: t + L, buf: buf(i) });
      t += L;
      // With two buffers, a thread past this barrier may go on to load pair i + 1 into the
      // buffer pair i - 1 used: this barrier is the one that keeps that safe.
      bar(true, dbl && i >= 1 && i + 1 < n);
      boxes.push({ lane: 'comp', tile: i, start: t, end: t + C, buf: buf(i) });
      t += C;
      if (!dbl && i < n - 1) bar(false, true);
      spans.push({ label: `iteration ${i}`, start: t0, end: t }); // i18n-ignore: span label parsed by PipelineScene (startsWith/split), which shows its own text
    }
  }
  const memBusy = n * L;
  const compBusy = n * C;
  return {
    scheme, n, L, C, s, boxes, barriers, spans, total: t,
    memBusy, compBusy, memUtil: t > 0 ? memBusy / t : 0, compUtil: t > 0 ? compBusy / t : 0,
  };
}

/** Two-barrier time over pipelined time. */
export const speedup = (n: number, L: number, C: number, s = 0) =>
  timeline('two-barriers', n, L, C, s).total / timeline('pipelined', n, L, C, s).total;

/** The limit of the speed-up for many pairs and free barriers: (L + C) / max(L, C), at most 2. */
export const speedupLimit = (L: number, C: number) => (L + C) / Math.max(L, C);

// ---------------------------------------------------------------- 3. Fig. 15.14: curr and next
/** Physical buffers in iteration `tile` of Fig. 15.14: curr is read, next is filled. */
export const roles = (tile: number): { curr: 0 | 1; next: 0 | 1 } => ({ curr: (tile & 1) as 0 | 1, next: ((tile + 1) & 1) as 0 | 1 });

// ---------------------------------------------------------------- 4. the pipelined loop, thread by thread
export type Variant = 'correct' | 'no-loop-barrier' | 'barrier-before-load' | 'no-epilogue' | 'wrong-tile' | 'single-buffer' | 'load-first';
/** The variants section 15.8's stepper and outcome table offer. 'load-first' (the loads of pair
 * tile + 1 before mm on pair tile, lab 15.5's README experiment) is used by the chapter's summary
 * page only, and is kept out of this list on purpose. */
export const VARIANTS: Variant[] = ['correct', 'no-loop-barrier', 'barrier-before-load', 'no-epilogue', 'wrong-tile', 'single-buffer'];
export const VARIANT_LABEL: Record<Variant, string> = {
  get correct() { return tr('as in Fig. 15.14', 'như Fig. 15.14'); },
  get 'no-loop-barrier'() { return tr('barrier in the loop removed', 'bỏ rào chắn trong vòng lặp'); },
  get 'barrier-before-load'() { return tr('barrier moved before loadTile', 'rào chắn dời lên trước loadTile'); },
  get 'no-epilogue'() { return tr('mm after the loop missing', 'thiếu mm sau vòng lặp'); },
  get 'wrong-tile'() { return tr('loads tile instead of tile + 1', 'nạp tile thay vì tile + 1'); },
  get 'single-buffer'() { return tr('one buffer, one barrier', 'một buffer, một rào chắn'); },
  get 'load-first'() { return tr('loads first, then mm', 'nạp trước, rồi mm'); },
};

export type Schedule = 'lockstep' | 'in-order' | 'reversed' | 'shuffled';
export const SCHEDULES: Schedule[] = ['lockstep', 'in-order', 'reversed', 'shuffled'];
export const SCHEDULE_LABEL: Record<Schedule, string> = {
  get lockstep() { return tr('all threads in lockstep', 'mọi luồng chạy đồng bộ từng bước'); },
  get 'in-order'() { return tr('each thread runs as far as it can, thread 0 first', 'mỗi luồng chạy xa nhất có thể, luồng 0 trước'); },
  get reversed() { return tr('each thread runs as far as it can, last thread first', 'mỗi luồng chạy xa nhất có thể, luồng cuối trước'); },
  get shuffled() { return tr('random interleaving', 'đan xen ngẫu nhiên'); },
};

export type Hazard = 'read-before-write' | 'overwritten-before-read';
export const HAZARD_LABEL: Record<Hazard, string> = {
  get 'read-before-write'() { return tr('read before write', 'đọc trước khi ghi'); },
  get 'overwritten-before-read'() { return tr('overwritten before read', 'bị ghi đè trước khi đọc'); },
};

export type Phase = 'prologue' | 'loop' | 'epilogue';
export interface Op {
  kind: 'load' | 'barrier' | 'mm';
  phase: Phase;
  /** Loop iteration (the `tile` variable); -1 in the prologue, the last tile after the loop. */
  iter: number;
  /** load and mm: the physical buffer. */
  buf?: 0 | 1;
  /** load: the tile it copies. */
  tile?: number;
  /** 1-based line in loopSource(variant).code. */
  line: number;
}

// The loop as the stepper shows it. For 'correct' these are, line for line, the loop of
// labs/ch15/snippets/s158_pipeline_toy.cu (gemmpipe.test.ts checks).
const SRC = {
  prologueLoad: 'loadTile(buf[0], tiles, 0);', // i18n-ignore: CUDA source shown as code
  prologueSync: '__syncthreads();',
  head: 'for (unsigned int tile = 0; tile < numTiles - 1; ++tile) {', // i18n-ignore: CUDA source shown as code
  mm: '    mm(buf[tile & 1], sum);', // i18n-ignore: CUDA source shown as code
  load: '    loadTile(buf[(tile + 1) & 1], tiles, tile + 1);', // i18n-ignore: CUDA source shown as code
  sync: '    __syncthreads();',
  tail: '}',
  epilogue: 'mm(buf[(numTiles - 1) & 1], sum);', // i18n-ignore: CUDA source shown as code
};

type Row = 'prologueLoad' | 'prologueSync' | 'head' | 'mm' | 'load' | 'sync' | 'tail' | 'epilogue';

function rows(v: Variant): { row: Row; text: string }[] {
  const text = (r: Row): string => {
    if (v === 'single-buffer') {
      if (r === 'mm') return '    mm(buf[0], sum);'; // i18n-ignore: CUDA source shown as code
      if (r === 'load') return '    loadTile(buf[0], tiles, tile + 1);'; // i18n-ignore: CUDA source shown as code
      if (r === 'epilogue') return 'mm(buf[0], sum);'; // i18n-ignore: CUDA source shown as code
    }
    if (v === 'wrong-tile' && r === 'load') return '    loadTile(buf[(tile + 1) & 1], tiles, tile);'; // i18n-ignore: CUDA source shown as code
    return SRC[r];
  };
  const body: Row[] =
    v === 'no-loop-barrier'
      ? ['mm', 'load']
      : v === 'barrier-before-load'
        ? ['mm', 'sync', 'load']
        : v === 'load-first'
          ? ['load', 'mm', 'sync']
          : ['mm', 'load', 'sync'];
  const order: Row[] = ['prologueLoad', 'prologueSync', 'head', ...body, 'tail', ...(v === 'no-epilogue' ? [] : (['epilogue'] as Row[]))];
  return order.map((row) => ({ row, text: text(row) }));
}

/** The loop's source for a variant, and the 1-based line of each kind of operation. */
export function loopSource(v: Variant): { code: string; line: Partial<Record<Row, number>> } {
  const rs = rows(v);
  const line: Partial<Record<Row, number>> = {};
  rs.forEach((r, i) => (line[r.row] = i + 1));
  return { code: rs.map((r) => r.text).join('\n'), line };
}

/** Every operation one thread executes, in order, for `n` tiles. */
export function program(v: Variant, n: number): Op[] {
  const ln = loopSource(v).line;
  const single = v === 'single-buffer';
  const b = (k: number): 0 | 1 => (single ? 0 : ((k & 1) as 0 | 1));
  const ops: Op[] = [];
  if (n === 0) return ops;
  ops.push({ kind: 'load', phase: 'prologue', iter: -1, buf: 0, tile: 0, line: ln.prologueLoad! });
  ops.push({ kind: 'barrier', phase: 'prologue', iter: -1, line: ln.prologueSync! });
  for (let i = 0; i < n - 1; i++) {
    const mm: Op = { kind: 'mm', phase: 'loop', iter: i, buf: b(i), line: ln.mm! };
    const load: Op = { kind: 'load', phase: 'loop', iter: i, buf: b(i + 1), tile: v === 'wrong-tile' ? i : i + 1, line: ln.load! };
    const sync: Op = { kind: 'barrier', phase: 'loop', iter: i, line: ln.sync ?? 0 };
    if (v === 'no-loop-barrier') ops.push(mm, load);
    else if (v === 'barrier-before-load') ops.push(mm, sync, load);
    else if (v === 'load-first') ops.push(load, mm, sync);
    else ops.push(mm, load, sync);
  }
  if (v !== 'no-epilogue') ops.push({ kind: 'mm', phase: 'epilogue', iter: n - 1, buf: b(n - 1), line: ln.epilogue! });
  return ops;
}

/** Slot p of tile k holds 10^k: a thread's sum shows, digit by digit, how many slots of each tile it added. */
export const tileValue = (k: number) => 10 ** k;
/** What every thread's sum should be: P slots of each of the n tiles. */
export const referenceSum = (P: number, n: number) => {
  let s = 0;
  for (let k = 0; k < n; k++) s += P * tileValue(k);
  return s;
};

export interface Slot {
  /** Tile it holds, -1 if never written (it then counts as 0). */
  tile: number;
  value: number;
  /** Index in the writer's program of the load that wrote it, -1 if never written. */
  pos: number;
}
export interface Read {
  buf: 0 | 1;
  slot: number;
  /** Tile found, -1 for an empty slot. */
  tile: number;
  /** Tile the program meant this read to find: the last load into this slot before this point of the program. */
  meant: number;
  ok: boolean;
  hazard?: Hazard;
}
export interface Violation {
  kind: Hazard;
  step: number;
  thread: number;
  message: string;
}
export interface Step {
  index: number;
  /** The thread, or -1 when a barrier opens. */
  thread: number;
  /** Index of the operation in the thread's program (the released barrier's for -1). */
  pos: number;
  op: Op;
  release?: boolean;
  write?: { buf: 0 | 1; slot: number; tile: number; value: number };
  reads?: Read[];
  /** mm: the thread's running sum after this step. */
  sum?: number;
  violations?: Violation[];
}
export interface Config {
  threads: number;
  tiles: number;
  variant: Variant;
}
export interface Trace {
  cfg: Config;
  schedule: Schedule;
  ops: Op[];
  steps: Step[];
  violations: Violation[];
  sums: number[];
  reference: number;
  wrong: number[];
  correct: boolean;
}

/** Where an operation sits, in words: "the prologue", "iteration 2", "after the loop". */
export const where = (op: Op) =>
  op.phase === 'prologue' ? tr('the prologue', 'phần mở đầu') : op.phase === 'epilogue' ? tr('after the loop', 'sau vòng lặp') : tr(`iteration ${op.iter}`, `vòng lặp ${op.iter}`);

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

const emptySlots = (P: number): Slot[][] => [0, 1].map(() => Array.from({ length: P }, () => ({ tile: -1, value: 0, pos: -1 })));

/** Runs one block of the loop under one thread order and records every operation. */
export function run(cfg: Config, schedule: Schedule, seed = 1): Trace {
  const P = cfg.threads;
  const ops = program(cfg.variant, cfg.tiles);
  const total = ops.length;
  const buffers = emptySlots(P);
  const pc = new Array(P).fill(0);
  const waiting = new Array(P).fill(false);
  const sums = new Array(P).fill(0);
  const steps: Step[] = [];
  const violations: Violation[] = [];
  const rand = mulberry(seed);
  const meantPos = (pos: number, buf: number) => {
    for (let a = pos - 1; a >= 0; a--) if (ops[a].kind === 'load' && ops[a].buf === buf) return a;
    return -1;
  };

  const exec = (t: number) => {
    const pos = pc[t];
    const op = ops[pos];
    const s: Step = { index: steps.length, thread: t, pos, op };
    steps.push(s);
    if (op.kind === 'barrier') {
      waiting[t] = true;
      return;
    }
    if (op.kind === 'load') {
      const tile = op.tile!;
      buffers[op.buf!][t] = { tile, value: tileValue(tile), pos };
      s.write = { buf: op.buf!, slot: t, tile, value: tileValue(tile) };
    } else {
      const want = meantPos(pos, op.buf!);
      const meant = want >= 0 ? ops[want].tile! : -1;
      s.reads = [];
      for (let j = 0; j < P; j++) {
        const sl = buffers[op.buf!][j];
        const r: Read = { buf: op.buf!, slot: j, tile: sl.tile, meant, ok: sl.pos === want };
        if (!r.ok) {
          r.hazard = sl.pos < want ? 'read-before-write' : 'overwritten-before-read';
          const found = sl.tile < 0 ? tr('it was still empty', 'ô vẫn còn trống') : tr(`it still held tile ${sl.tile}`, `ô vẫn còn giữ tile ${sl.tile}`);
          const msg =
            r.hazard === 'read-before-write'
              ? tr(
                  `thread ${t} added up buffer ${op.buf} in ${where(op)} before thread ${j} had loaded tile ${meant} into slot ${j}: ${found}`,
                  `luồng ${t} cộng buffer ${op.buf} trong ${where(op)} trước khi luồng ${j} nạp tile ${meant} vào ô ${j}: ${found}`,
                )
              : tr(
                  `thread ${t} needed tile ${meant} in slot ${j} of buffer ${op.buf} in ${where(op)}, but thread ${j} had already loaded tile ${sl.tile} over it`,
                  `luồng ${t} cần tile ${meant} ở ô ${j} của buffer ${op.buf} trong ${where(op)}, nhưng luồng ${j} đã nạp tile ${sl.tile} đè lên nó`,
                );
          const v: Violation = { kind: r.hazard, step: s.index, thread: t, message: msg };
          (s.violations ??= []).push(v);
          violations.push(v);
        }
        sums[t] += sl.value;
        s.reads.push(r);
      }
      s.sum = sums[t];
    }
    pc[t]++;
  };

  const live = (t: number) => pc[t] < total;
  const runnable = (t: number) => live(t) && !waiting[t];
  const release = () => {
    if (!waiting.some(Boolean)) return;
    if (!waiting.every((w, t) => w || !live(t))) return;
    const t0 = waiting.findIndex(Boolean);
    steps.push({ index: steps.length, thread: -1, pos: pc[t0], op: ops[pc[t0]], release: true });
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

  const reference = referenceSum(P, cfg.tiles);
  const wrong = sums.flatMap((v, t) => (v === reference ? [] : [t]));
  return { cfg, schedule, ops, steps, violations, sums, reference, wrong, correct: wrong.length === 0 };
}

export interface ThreadState {
  /** Operations finished. */
  pc: number;
  /** The operation it executed last, or null before it starts. */
  last: Op | null;
  waiting: boolean;
  done: boolean;
  sum: number;
}
export interface State {
  buffers: Slot[][];
  threads: ThreadState[];
}

/** The block after its first `n` steps, for the stepper and the scene. */
export function stateAt(trace: Trace, n: number): State {
  const P = trace.cfg.threads;
  const total = trace.ops.length;
  const buffers = emptySlots(P);
  const threads: ThreadState[] = Array.from({ length: P }, () => ({ pc: 0, last: null, waiting: false, done: total === 0, sum: 0 }));
  for (let i = 0; i < Math.min(n, trace.steps.length); i++) {
    const s = trace.steps[i];
    if (s.release) {
      for (const th of threads)
        if (th.waiting) {
          th.waiting = false;
          th.pc++;
          th.done = th.pc >= total;
        }
      continue;
    }
    const th = threads[s.thread];
    th.last = s.op;
    if (s.op.kind === 'barrier') {
      th.waiting = true;
      continue;
    }
    if (s.write) buffers[s.write.buf][s.write.slot] = { tile: s.write.tile, value: s.write.value, pos: s.pos };
    if (s.sum != null) th.sum = s.sum;
    th.pc++;
    th.done = th.pc >= total;
  }
  return { buffers, threads };
}

/** Index just after the first step with a hazard (the stepper's "go to the first failure"), or null. */
export function firstFailure(trace: Trace): number | null {
  return trace.violations.length ? trace.violations[0].step + 1 : null;
}

/** Index just after the first barrier release at or after step `from`, or the end. */
export function nextRelease(trace: Trace, from: number): number {
  for (let i = from; i < trace.steps.length; i++) if (trace.steps[i].release) return i + 1;
  return trace.steps.length;
}

/** The hazards a run hit, without repeats, in a fixed order. */
export const hazardsOf = (trace: Trace): Hazard[] =>
  (['read-before-write', 'overwritten-before-read'] as Hazard[]).filter((h) => trace.violations.some((v) => v.kind === h));

/** Unsigned 32-bit numTiles - 1, as the loop condition of Fig. 15.14 computes it (ours, for K = 0). */
export const loopBound = (numTiles: number) => (numTiles - 1) >>> 0;
/** numTiles = (K + bK - 1) / bK in unsigned arithmetic. */
export const numTilesFor = (K: number, bK: number) => Math.floor((K + bK - 1) / bK) >>> 0;
