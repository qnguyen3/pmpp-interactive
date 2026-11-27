// The privatised unstable filter of Fig. 12.6 (book §12.4, pp. 295-297), run the way the
// course's emulator runs it, for section 12.4's stepper and outcome table.
//
// The kernels are labs/ch12/snippets/s124_privatised_variants.cu: the book's kernel at
// BLOCK_DIM = 4, and four copies with one defect each (a barrier dropped, or the private
// counter bumped with a plain ++). kernelBody() writes the body of any combination of
// those defects; privfilter.test.ts checks that the body of each single defect is exactly
// the snippet file's kernel, and runs the file on the emulator and on its race flavour.
//
// simulate() hands a baton from thread to thread as the emu backend does
// (labs/common/include/pmpp/emu/emu_runtime.h): blocks one after another, a thread runs
// until it reaches a barrier or its end, then the next thread in the order; the forward,
// reverse and seeded shuffle orders (emuOrders from histpriv.ts, which reproduces the
// emulator's generator); shared memory filled with 0xCD bytes at the start of each block.
// A write outside an array stops the run, as AddressSanitizer stops the emulator. The
// public output starts as 0xFF bytes, as the snippet's driver fills it.
//
// One addition the emulator does not have: the order in which blocks run, and so the order
// in which they reserve their public ranges, can be chosen (the emulator runs block 0, 1, 2,
// ...; Fig. 12.5 shows another order). Only the emulator's order is compared with it.
//
// races() is a happens-before check of the kind ThreadSanitizer runs in the race flavour:
// two accesses to one location race when at least one writes, they are not both atomic, and
// neither is ordered before the other (same thread, or same block with a barrier between).
// Blocks are never ordered. Scopes are ignored, as the emulator ignores them.
//
// What it leaves out: time, warps, caches, and any interleaving finer than one source
// statement. It is faithful about one thing: which value each read sees.
//
// Erasable TypeScript only.
import { emuOrders, SCHEDULES, SCHEDULE_LABEL, type Schedule } from './histpriv';
import { tr } from '../lib/locale.ts';

export { SCHEDULES, SCHEDULE_LABEL, type Schedule };

export const BLOCKS = 4;
export const THREADS = 4;
export const BLOCK_DIM = THREADS;
export const N = BLOCKS * THREADS;
/** What the emulator leaves in shared memory that nobody wrote: four 0xCD bytes. */
export const GARBAGE = 0xcdcdcdcd;
/** What the driver fills the public output with: four 0xFF bytes. */
export const UNWRITTEN = 0xffffffff;
export const SNIPPET = 'ch12/snippets/s124_privatised_variants.cu';

/** Fig. 12.5's kept keys: k1, k3 (block 0), k4, k6 (block 1), k8, k10, k11 (block 2), k14 (block 3). */
export const BOOK_KEEP: readonly boolean[] = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0].map((x) => x === 1);
/** Fig. 12.5 appends the blocks' lists in the order green, orange, yellow, blue: blocks 1, 3, 2, 0. */
export const BOOK_BLOCK_ORDER: readonly number[] = [1, 3, 2, 0];
export const BOOK_COLOUR = ['blue', 'green', 'yellow', 'orange'] as const;

/** The driver stores key i as 2i + 1 when it is kept and as 2i when not; cond() keeps odd values. */
export const keyValue = (i: number, keep: boolean) => 2 * i + (keep ? 1 : 0);
export const cond = (v: number) => (v & 1) === 1;
/** "k5" for the value of key 5, "0xCDCDCDCD" for shared-memory garbage, "" for an unwritten public slot. */
export function keyName(v: number): string {
  if (v === UNWRITTEN) return '';
  if (v === GARBAGE) return '0xCDCDCDCD';
  if (v < 2 * N) return `k${v >>> 1}`;
  return String(v);
}

// ---------------------------------------------------------------- shapes and kernels
export interface Shape {
  barrier1: boolean;
  barrier2: boolean;
  barrier3: boolean;
  atomicCounter: boolean;
}
export const CORRECT: Shape = { barrier1: true, barrier2: true, barrier3: true, atomicCounter: true };

export type Variant = 'correct' | 'no-barrier-1' | 'no-barrier-2' | 'no-barrier-3' | 'plain-counter';
export const VARIANTS: Variant[] = ['correct', 'no-barrier-1', 'no-barrier-2', 'no-barrier-3', 'plain-counter'];
export const VARIANT: Record<Variant, { kernel: string; label: string; shape: Shape }> = {
  correct: {
    kernel: 'filterPrivatised',
    get label() {
      return tr('all three barriers', 'cả ba rào chắn');
    },
    shape: CORRECT,
  },
  'no-barrier-1': {
    kernel: 'filterNoBarrier1',
    get label() {
      return tr('barrier 1 removed', 'bỏ rào chắn 1');
    },
    shape: { ...CORRECT, barrier1: false },
  },
  'no-barrier-2': {
    kernel: 'filterNoBarrier2',
    get label() {
      return tr('barrier 2 removed', 'bỏ rào chắn 2');
    },
    shape: { ...CORRECT, barrier2: false },
  },
  'no-barrier-3': {
    kernel: 'filterNoBarrier3',
    get label() {
      return tr('barrier 3 removed', 'bỏ rào chắn 3');
    },
    shape: { ...CORRECT, barrier3: false },
  },
  'plain-counter': {
    kernel: 'filterPlainCounter',
    get label() {
      return tr('private counter not atomic', 'bộ đếm riêng không nguyên tử (atomic)');
    },
    shape: { ...CORRECT, atomicCounter: false },
  },
};

type LineKey =
  | 'd1' | 'd2' | 'd3' | 'zero' | 'b1' | 'idx' | 'guard' | 'load' | 'cond' | 'ref' | 'count' | 'store'
  | 'c2' | 'c1' | 'b2' | 't0' | 'pref' | 'reserve' | 'c3' | 'b3' | 'test' | 'commit';
const TEXT: Record<LineKey, string> = {
  d1: '__shared__ unsigned int output_s[BLOCK_DIM];', // i18n-ignore: kernel source
  d2: '__shared__ unsigned int outputSize_s;', // i18n-ignore: kernel source
  d3: '__shared__ unsigned int blockStart;', // i18n-ignore: kernel source
  zero: 'if (threadIdx.x == 0) outputSize_s = 0;',
  b1: '__syncthreads();  // barrier 1', // i18n-ignore: kernel source
  idx: 'unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;',
  guard: 'if (i < N) {',
  load: '    unsigned int val = input[i];', // i18n-ignore: kernel source
  cond: '    if (cond(val)) {', // i18n-ignore: kernel source
  // Four statements take two lines each, so that the stepper's listing fits its column.
  ref: '        cuda::atomic_ref<unsigned int, cuda::thread_scope_block>\n            outputSize_s_ref(outputSize_s);',
  count: '        unsigned int slot = outputSize_s_ref.fetch_add(1,\n            cuda::memory_order_relaxed);',
  store: '        output_s[slot] = val;',
  c2: '    }',
  c1: '}',
  b2: '__syncthreads();  // barrier 2', // i18n-ignore: kernel source
  t0: 'if (threadIdx.x == 0) {',
  pref: '    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>\n        outputSize_ref(*outputSize);',
  reserve: '    blockStart = outputSize_ref.fetch_add(outputSize_s,\n        cuda::memory_order_relaxed);',
  c3: '}',
  b3: '__syncthreads();  // barrier 3', // i18n-ignore: kernel source
  test: 'if (threadIdx.x < outputSize_s)',
  commit: '    output[blockStart + threadIdx.x] = output_s[threadIdx.x];',
};
const PLAIN_COUNT = '        unsigned int slot = outputSize_s++;'; // i18n-ignore: kernel source

function bodyLines(shape: Shape, placeholders: boolean): { key: LineKey | null; text: string }[] {
  const out: { key: LineKey | null; text: string }[] = [];
  const add = (k: LineKey) => out.push({ key: k, text: TEXT[k] });
  const gone = (n: number) => placeholders && out.push({ key: null, text: `// barrier ${n} removed` });
  add('d1'); add('d2'); add('d3'); add('zero');
  if (shape.barrier1) add('b1'); else gone(1);
  add('idx'); add('guard'); add('load'); add('cond');
  if (shape.atomicCounter) {
    add('ref');
    add('count');
  } else out.push({ key: 'count', text: PLAIN_COUNT });
  add('store'); add('c2'); add('c1');
  if (shape.barrier2) add('b2'); else gone(2);
  add('t0'); add('pref'); add('reserve'); add('c3');
  if (shape.barrier3) add('b3'); else gone(3);
  add('test'); add('commit');
  return out;
}

/**
 * The kernel body (everything between the braces, unindented by one level). With
 * placeholders, a removed barrier leaves a comment line in its place, so the stepper's
 * listing shows where it was; without, it is exactly the snippet file's kernel body.
 */
export function kernelBody(shape: Shape, placeholders = false): string {
  return bodyLines(shape, placeholders).map((l) => l.text).join('\n');
}
/** Each statement of kernelBody(shape, placeholders) with its first 1-based line and its line count. */
function statements(shape: Shape, placeholders: boolean) {
  let line = 1;
  return bodyLines(shape, placeholders).map((l) => {
    const rows = l.text.split('\n').length;
    const s = { key: l.key, first: line, rows };
    line += rows;
    return s;
  });
}
/** First 1-based line of a statement in kernelBody(shape, placeholders), or 0 when it is not there. */
function lineOf(shape: Shape, placeholders: boolean, key: LineKey): number {
  return statements(shape, placeholders).find((s) => s.key === key)?.first ?? 0;
}
/** Every line of the statement that starts at `line` in kernelBody(shape, placeholders): what a listing marks. */
export function statementLines(shape: Shape, placeholders: boolean, line: number): number[] {
  const s = statements(shape, placeholders).find((x) => x.first === line);
  return s ? Array.from({ length: s.rows }, (_, k) => line + k) : [line];
}

// ---------------------------------------------------------------- block orders
export type BlockOrder = 'in-order' | 'book' | 'random';
export const BLOCK_ORDER_LABEL: Record<BlockOrder, string> = {
  get 'in-order'() {
    return tr('0, 1, 2, 3 (as the emulator)', '0, 1, 2, 3 (như trình giả lập)');
  },
  book: '1, 3, 2, 0 (Fig. 12.5)',
  get random() {
    return tr('random', 'ngẫu nhiên');
  },
};
export function blockOrderOf(kind: BlockOrder, seed = 1): number[] {
  if (kind === 'in-order') return [0, 1, 2, 3];
  if (kind === 'book') return [...BOOK_BLOCK_ORDER];
  const o = [0, 1, 2, 3];
  let r = (Math.imul(seed, 2654435761) + 977) >>> 0;
  for (let i = o.length - 1; i > 0; i--) {
    r = (Math.imul(r, 1664525) + 1013904223) >>> 0;
    const j = (r >>> 8) % (i + 1);
    [o[i], o[j]] = [o[j], o[i]];
  }
  return o;
}

// ---------------------------------------------------------------- the run
export type Site =
  | 'zero' | 'count' | 'store' | 'reserve-read' | 'reserve-write' | 'public-add'
  | 'commit-test' | 'commit-read-start' | 'commit-read-private' | 'commit-write';
export const SITE_TEXT: Record<Site, string> = {
  get zero() {
    return tr('the zeroing of outputSize_s', 'việc đặt outputSize_s về 0');
  },
  get count() {
    return tr('the increment of outputSize_s', 'việc tăng outputSize_s');
  },
  get store() {
    return tr('the store into output_s', 'việc ghi vào output_s');
  },
  get 'reserve-read'() {
    return tr('thread 0\'s read of outputSize_s for the reservation', 'lần đọc outputSize_s của luồng 0 để giữ chỗ');
  },
  get 'reserve-write'() {
    return tr('thread 0\'s write of blockStart', 'lần ghi blockStart của luồng 0');
  },
  get 'public-add'() {
    return tr('the add to the public counter', 'phép cộng vào bộ đếm công khai');
  },
  get 'commit-test'() {
    return tr('the read of outputSize_s in the commit test', 'lần đọc outputSize_s trong phép kiểm tra của bước commit');
  },
  get 'commit-read-start'() {
    return tr('the read of blockStart in the commit', 'lần đọc blockStart trong bước commit');
  },
  get 'commit-read-private'() {
    return tr('the read of output_s in the commit', 'lần đọc output_s trong bước commit');
  },
  get 'commit-write'() {
    return tr('the store to the public output', 'việc ghi vào output công khai');
  },
};

export interface Access {
  block: number;
  thread: number;
  epoch: number;
  space: 'shared' | 'global';
  addr: string;
  write: boolean;
  atomic: boolean;
  site: Site;
}

export type Op = 'zero' | 'arrive' | 'release' | 'load' | 'count' | 'store' | 'reserve' | 'commit' | 'skip';
export type ViolationKind =
  | 'garbage-count' | 'late-zero' | 'short-reservation' | 'garbage-reservation'
  | 'garbage-start' | 'garbage-copy' | 'overwrite' | 'crash';
export interface Violation {
  kind: ViolationKind;
  message: string;
}

export interface BlockState {
  output_s: number[];
  outputSize_s: number;
  blockStart: number;
  /** Which shared words have been written in this block (the rest still hold 0xCD bytes). */
  written: { output_s: boolean[]; outputSize_s: boolean; blockStart: boolean };
  /** Per thread: 'not started', 'running', 'at barrier N', 'finished'. */
  status: string[];
}
export interface Snapshot {
  blocks: BlockState[];
  output: number[];
  /** The block that last wrote each public slot, or -1. */
  writer: number[];
  outputSize: number;
}

export interface Step {
  block: number;
  /** -1 for a barrier opening. */
  thread: number;
  op: Op;
  barrier?: 1 | 2 | 3;
  /** 1-based line in kernelBody(shape, true). */
  line: number;
  text: string;
  violation?: Violation;
  /** A remark that is not a failure on this schedule (the plain ++). */
  note?: string;
  after: Snapshot;
}

export interface Outcome {
  kind: 'correct' | 'wrong' | 'crash';
  size: number;
  output: number[];
  expectedSize: number;
  /** Kept keys (by index) that appear nowhere in the output buffer. */
  lost: number[];
  /** Kept keys (by index) that appear more than once. */
  duplicated: number[];
  /** Slots below the reported size that hold no kept key. */
  foreign: number;
  /** Slots at or above the expected size that were written. */
  extra: number;
  crash?: string;
}

export interface Trace {
  shape: Shape;
  schedule: Schedule;
  blockOrder: number[];
  keep: boolean[];
  /** The state before the first step: shared memory all 0xCD bytes, the public output all 0xFF. */
  initial: Snapshot;
  steps: Step[];
  accesses: Access[];
  outcome: Outcome;
  /** Index of the first step with a violation, or -1. */
  firstFailure: number;
}

const hex = (v: number) => '0x' + (v >>> 0).toString(16).toUpperCase().padStart(8, '0');
const show = (v: number) => (v === GARBAGE ? '0xCDCDCDCD' : String(v));

function fresh(): BlockState {
  return {
    output_s: new Array<number>(BLOCK_DIM).fill(GARBAGE),
    outputSize_s: GARBAGE,
    blockStart: GARBAGE,
    written: { output_s: new Array<boolean>(BLOCK_DIM).fill(false), outputSize_s: false, blockStart: false },
    status: new Array<string>(THREADS).fill(tr('not started', 'chưa bắt đầu')),
  };
}
function clone(s: Snapshot): Snapshot {
  return {
    blocks: s.blocks.map((b) => ({
      output_s: [...b.output_s],
      outputSize_s: b.outputSize_s,
      blockStart: b.blockStart,
      written: { output_s: [...b.written.output_s], outputSize_s: b.written.outputSize_s, blockStart: b.written.blockStart },
      status: [...b.status],
    })),
    output: [...s.output],
    writer: [...s.writer],
    outputSize: s.outputSize,
  };
}

type POp = { k: 'zero' } | { k: 'barrier'; n: 1 | 2 | 3 } | { k: 'load' } | { k: 'count' } | { k: 'store' } | { k: 'reserve' } | { k: 'commit' };
function program(shape: Shape, t: number): POp[] {
  const p: POp[] = [];
  if (t === 0) p.push({ k: 'zero' });
  if (shape.barrier1) p.push({ k: 'barrier', n: 1 });
  p.push({ k: 'load' }, { k: 'count' }, { k: 'store' });
  if (shape.barrier2) p.push({ k: 'barrier', n: 2 });
  if (t === 0) p.push({ k: 'reserve' });
  if (shape.barrier3) p.push({ k: 'barrier', n: 3 });
  p.push({ k: 'commit' });
  return p;
}

/** One launch: four blocks of four threads over sixteen keys. */
export function simulate(
  shape: Shape,
  schedule: Schedule,
  keep: readonly boolean[] = BOOK_KEEP,
  blockOrder: readonly number[] = [0, 1, 2, 3],
): Trace {
  if (keep.length !== N) throw new Error(`the stepper runs ${N} keys`);
  const input = keep.map((k, i) => keyValue(i, k));
  const orders = emuOrders(schedule, BLOCKS, THREADS);
  const st: Snapshot = {
    blocks: Array.from({ length: BLOCKS }, fresh),
    output: new Array<number>(N).fill(UNWRITTEN),
    writer: new Array<number>(N).fill(-1),
    outputSize: 0,
  };
  const initial = clone(st);
  const steps: Step[] = [];
  const accesses: Access[] = [];
  const line = (k: LineKey) => lineOf(shape, true, k);
  let crashed: string | undefined;

  for (let pos = 0; pos < BLOCKS && !crashed; pos++) {
    const b = blockOrder[pos];
    const sh = st.blocks[b];
    const keptHere = keep.slice(b * THREADS, (b + 1) * THREADS).filter(Boolean).length;
    const progs = Array.from({ length: THREADS }, (_, t) => program(shape, t));
    const pc = new Array<number>(THREADS).fill(0);
    const epoch = new Array<number>(THREADS).fill(0);
    const waiting = new Array<boolean>(THREADS).fill(false);
    const slot = new Array<number>(THREADS).fill(-1);
    const val = new Array<number>(THREADS).fill(0);
    let increments = 0;
    const done = (t: number) => pc[t] >= progs[t].length;
    const push = (s: Omit<Step, 'after' | 'block'>) => steps.push({ ...s, block: b, after: clone(st) });
    const acc = (t: number, a: Omit<Access, 'block' | 'thread' | 'epoch'>) => accesses.push({ ...a, block: b, thread: t, epoch: epoch[t] });
    const who = (t: number) => tr(`Block ${b}, thread ${t}`, `Khối ${b}, luồng ${t}`);

    while (!crashed) {
      for (const t of orders[pos]) {
        if (done(t) || waiting[t] || crashed) continue;
        while (!done(t) && !crashed) {
          const op = progs[t][pc[t]];
          sh.status[t] = tr('running', 'đang chạy');
          const i = b * THREADS + t;
          if (op.k === 'barrier') {
            waiting[t] = true;
            sh.status[t] = tr(`at barrier ${op.n}`, `tại rào chắn ${op.n}`);
            push({
              thread: t,
              op: 'arrive',
              barrier: op.n,
              line: line(`b${op.n}` as LineKey),
              text: tr(
                `${who(t)} reaches barrier ${op.n} and waits for the other three threads of its block.`,
                `${who(t)} tới rào chắn ${op.n} và chờ ba luồng còn lại của khối.`,
              ),
            });
            break;
          }
          pc[t]++;
          const last = done(t);
          const fin = () => {
            if (done(t)) sh.status[t] = tr('finished', 'đã xong');
          };
          if (op.k === 'zero') {
            const v: Violation | undefined = increments
              ? {
                  kind: 'late-zero',
                  message: tr(
                    `thread 0 zeroes the counter after ${increments} key${increments === 1 ? ' was' : 's were'} already counted, so those slots will be handed out again`,
                    `luồng 0 đặt bộ đếm về 0 sau khi ${increments} khóa đã được đếm, nên các ô đó sẽ bị phát ra lần nữa`,
                  ),
                }
              : undefined;
            sh.outputSize_s = 0;
            sh.written.outputSize_s = true;
            acc(t, { space: 'shared', addr: 'outputSize_s', write: true, atomic: false, site: 'zero' });
            fin();
            push({
              thread: t,
              op: 'zero',
              line: line('zero'),
              text: tr(`${who(t)} sets the private counter outputSize_s to 0.`, `${who(t)} đặt bộ đếm riêng outputSize_s về 0.`),
              violation: v,
            });
          } else if (op.k === 'load') {
            val[t] = input[i];
            fin();
            const kept = cond(val[t]);
            if (!kept) pc[t] += 2; // skip the count and the store
            fin();
            push({
              thread: t, op: 'load', line: line(kept ? 'cond' : 'load'),
              text: kept
                ? tr(
                    `${who(t)}: i = ${b}·4 + ${t} = ${i}. It reads k${i}, which passes cond(), so it will take a slot in the private list.`,
                    `${who(t)}: i = ${b}·4 + ${t} = ${i}. Nó đọc k${i}, khóa này qua được cond(), nên nó sẽ lấy một ô trong danh sách riêng.`,
                  )
                : tr(
                    `${who(t)}: i = ${b}·4 + ${t} = ${i}. It reads k${i}, which fails cond(): nothing to count or store.`,
                    `${who(t)}: i = ${b}·4 + ${t} = ${i}. Nó đọc k${i}, khóa này không qua cond(): không có gì để đếm hay ghi.`,
                  ),
            });
          } else if (op.k === 'count') {
            const old = sh.outputSize_s;
            const garbage = !sh.written.outputSize_s;
            sh.outputSize_s = (old + 1) >>> 0;
            increments++;
            slot[t] = old;
            if (shape.atomicCounter) acc(t, { space: 'shared', addr: 'outputSize_s', write: true, atomic: true, site: 'count' });
            else {
              acc(t, { space: 'shared', addr: 'outputSize_s', write: false, atomic: false, site: 'count' });
              acc(t, { space: 'shared', addr: 'outputSize_s', write: true, atomic: false, site: 'count' });
            }
            push({
              thread: t, op: 'count', line: line('count'),
              text: tr(
                `${who(t)} ${shape.atomicCounter ? 'atomically adds 1 to' : 'reads and increments'} outputSize_s: it gets the old value, ${show(old)}, as its slot, and the counter becomes ${show(sh.outputSize_s)}.`,
                `${who(t)} ${shape.atomicCounter ? 'cộng 1 một cách nguyên tử vào' : 'đọc rồi tăng'} outputSize_s: nó nhận giá trị cũ, ${show(old)}, làm ô của mình, và bộ đếm thành ${show(sh.outputSize_s)}.`,
              ),
              violation: garbage
                ? {
                    kind: 'garbage-count',
                    message: tr(
                      `outputSize_s has not been zeroed yet: it still holds the 0xCD bytes shared memory started with, so the slot is ${show(old)}`,
                      `outputSize_s chưa được đặt về 0: nó vẫn giữ các byte 0xCD mà bộ nhớ chia sẻ có từ đầu, nên ô là ${show(old)}`,
                    ),
                  }
                : undefined,
              note: shape.atomicCounter
                ? undefined
                : tr(
                    'Not atomic. Here threads take turns, so it gives the right slot; threads running at the same time can both read the same value. The race check below reports it.',
                    'Không nguyên tử. Ở đây các luồng lần lượt chạy, nên nó cho đúng ô; các luồng chạy cùng lúc có thể cùng đọc một giá trị. Phép kiểm tra race bên dưới báo điều đó.',
                  ),
            });
          } else if (op.k === 'store') {
            const s = slot[t];
            if (s >= BLOCK_DIM) {
              crashed = tr(
                `block ${b}, thread ${t} writes output_s[${s}], outside the ${BLOCK_DIM}-entry list`,
                `khối ${b}, luồng ${t} ghi output_s[${s}], ngoài danh sách ${BLOCK_DIM} phần tử`,
              );
              push({
                thread: t,
                op: 'store',
                line: line('store'),
                text: tr(`${who(t)} stores k${i} into output_s[${show(s)}].`, `${who(t)} ghi k${i} vào output_s[${show(s)}].`),
                violation: {
                  kind: 'crash',
                  message: tr(
                    `output_s[${show(s)}] is outside the ${BLOCK_DIM}-entry list: the emulator stops here with an AddressSanitizer report`,
                    `output_s[${show(s)}] nằm ngoài danh sách ${BLOCK_DIM} phần tử: trình giả lập dừng ở đây với một báo cáo AddressSanitizer`,
                  ),
                },
              });
              break;
            }
            sh.output_s[s] = val[t];
            sh.written.output_s[s] = true;
            acc(t, { space: 'shared', addr: `output_s[${s}]`, write: true, atomic: false, site: 'store' });
            fin();
            push({ thread: t, op: 'store', line: line('store'), text: tr(`${who(t)} stores k${i} into output_s[${s}].`, `${who(t)} ghi k${i} vào output_s[${s}].`) });
          } else if (op.k === 'reserve') {
            const size = sh.outputSize_s;
            const start = st.outputSize;
            st.outputSize = (start + size) >>> 0;
            sh.blockStart = start;
            sh.written.blockStart = true;
            acc(t, { space: 'shared', addr: 'outputSize_s', write: false, atomic: false, site: 'reserve-read' });
            accesses.push({ block: b, thread: t, epoch: epoch[t], space: 'global', addr: 'outputSize', write: true, atomic: true, site: 'public-add' });
            acc(t, { space: 'shared', addr: 'blockStart', write: true, atomic: false, site: 'reserve-write' });
            fin();
            const v: Violation | undefined = !sh.written.outputSize_s
              ? {
                  kind: 'garbage-reservation',
                  message: tr(`outputSize_s was never zeroed: thread 0 reserves ${show(size)} slots`, `outputSize_s chưa từng được đặt về 0: luồng 0 giữ chỗ ${show(size)} ô`),
                }
              : size !== keptHere
                ? {
                    kind: 'short-reservation',
                    message: tr(
                      `thread 0 reserves ${size} public slot${size === 1 ? '' : 's'}, but block ${b} keeps ${keptHere}: it read the counter before every thread had counted`,
                      `luồng 0 giữ chỗ ${size} ô công khai, nhưng khối ${b} giữ lại ${keptHere}: nó đọc bộ đếm trước khi mọi luồng đếm xong`,
                    ),
                  }
                : undefined;
            push({
              thread: t, op: 'reserve', line: line('reserve'),
              text: tr(
                `${who(t)} reserves outputSize_s = ${show(size)} public slots: the public counter goes from ${start} to ${st.outputSize}, so blockStart = ${start}.`,
                `${who(t)} giữ chỗ outputSize_s = ${show(size)} ô công khai: bộ đếm công khai đi từ ${start} lên ${st.outputSize}, nên blockStart = ${start}.`,
              ),
              violation: v,
            });
          } else {
            const size = sh.outputSize_s;
            acc(t, { space: 'shared', addr: 'outputSize_s', write: false, atomic: false, site: 'commit-test' });
            if (!(t < size)) {
              fin();
              push({
                thread: t,
                op: 'skip',
                line: line('test'),
                text: tr(
                  `${who(t)}: ${t} < outputSize_s (${show(size)}) is false, so it has nothing to copy${last ? ' and finishes' : ''}.`,
                  `${who(t)}: ${t} < outputSize_s (${show(size)}) là sai, nên nó không có gì để chép${last ? ' và kết thúc' : ''}.`,
                ),
              });
              continue;
            }
            const start = sh.blockStart;
            const startGarbage = !sh.written.blockStart;
            const v = sh.output_s[t];
            const copyGarbage = !sh.written.output_s[t];
            acc(t, { space: 'shared', addr: 'blockStart', write: false, atomic: false, site: 'commit-read-start' });
            acc(t, { space: 'shared', addr: `output_s[${t}]`, write: false, atomic: false, site: 'commit-read-private' });
            const j = (start + t) >>> 0;
            if (j >= N) {
              crashed = tr(
                `block ${b}, thread ${t} writes output[${j}], outside the ${N}-slot output`,
                `khối ${b}, luồng ${t} ghi output[${j}], ngoài output ${N} ô`,
              );
              push({
                thread: t, op: 'commit', line: line('commit'),
                text: tr(
                  `${who(t)} copies output_s[${t}] to output[blockStart + ${t}] = output[${show(start)} + ${t}].`,
                  `${who(t)} chép output_s[${t}] sang output[blockStart + ${t}] = output[${show(start)} + ${t}].`,
                ),
                violation: {
                  kind: 'crash',
                  message: tr(
                    `${startGarbage ? 'blockStart has not been written yet (0xCD bytes), so ' : ''}output[${j}] is outside the ${N}-slot output: the emulator stops here with an AddressSanitizer report`,
                    `${startGarbage ? 'blockStart chưa được ghi (các byte 0xCD), nên ' : ''}output[${j}] nằm ngoài output ${N} ô: trình giả lập dừng ở đây với một báo cáo AddressSanitizer`,
                  ),
                },
              });
              break;
            }
            const prev = st.writer[j];
            st.output[j] = v;
            st.writer[j] = b;
            accesses.push({ block: b, thread: t, epoch: epoch[t], space: 'global', addr: `output[${j}]`, write: true, atomic: false, site: 'commit-write' });
            fin();
            const viol: Violation | undefined = startGarbage
              ? { kind: 'garbage-start', message: tr(`blockStart has not been written yet; it holds ${hex(start)}`, `blockStart chưa được ghi; nó đang giữ ${hex(start)}`) }
              : copyGarbage
                ? {
                    kind: 'garbage-copy',
                    message: tr(
                      `output_s[${t}] was never written: thread ${t} copies the 0xCD bytes into the public output`,
                      `output_s[${t}] chưa từng được ghi: luồng ${t} chép các byte 0xCD vào output công khai`,
                    ),
                  }
                : prev >= 0 && prev !== b
                  ? {
                      kind: 'overwrite',
                      message: tr(
                        `output[${j}] was already written by block ${prev}: this overwrites one of its keys`,
                        `output[${j}] đã được khối ${prev} ghi: lần ghi này đè lên một khóa của khối đó`,
                      ),
                    }
                  : undefined;
            push({
              thread: t, op: 'commit', line: line('commit'),
              text: tr(
                `${who(t)} copies output_s[${t}] (${keyName(v) || show(v)}) to output[blockStart + ${t}] = output[${j}]${last ? ' and finishes' : ''}.`,
                `${who(t)} chép output_s[${t}] (${keyName(v) || show(v)}) sang output[blockStart + ${t}] = output[${j}]${last ? ' và kết thúc' : ''}.`,
              ),
              violation: viol,
            });
          }
        }
      }
      if (crashed) break;
      if (progs.every((_, t) => done(t))) break;
      // Every live thread waits at a barrier: open it.
      const n = (progs[0][pc[0]] as { n: 1 | 2 | 3 }).n;
      for (let t = 0; t < THREADS; t++)
        if (waiting[t]) {
          waiting[t] = false;
          epoch[t]++;
          pc[t]++;
          sh.status[t] = tr(`past barrier ${(progs[t][pc[t] - 1] as { n: number }).n}`, `đã qua rào chắn ${(progs[t][pc[t] - 1] as { n: number }).n}`);
        }
      for (let t = 0; t < THREADS; t++) if (done(t)) sh.status[t] = tr('finished', 'đã xong');
      const what =
        n === 1 ? tr('the counter is zero before anyone counts', 'bộ đếm bằng 0 trước khi có luồng nào đếm')
          : n === 2 ? tr('every kept key is in the private list, so outputSize_s is the block\'s full count', 'mọi khóa được giữ lại đều đã ở trong danh sách riêng, nên outputSize_s là số đếm đầy đủ của khối')
            : tr('blockStart is written, so every thread can read it', 'blockStart đã được ghi, nên mọi luồng đều đọc được nó');
      push({ thread: -1, op: 'release', barrier: n, line: line(`b${n}` as LineKey), text: tr(`Barrier ${n} opens: ${what}.`, `Rào chắn ${n} mở: ${what}.`) });
    }
  }

  const outcome = classify(st, keep, crashed);
  return {
    shape, schedule, blockOrder: [...blockOrder], keep: [...keep], initial, steps, accesses, outcome,
    firstFailure: steps.findIndex((s) => !!s.violation),
  };
}

function classify(st: Snapshot, keep: readonly boolean[], crash?: string): Outcome {
  const kept = keep.flatMap((k, i) => (k ? [keyValue(i, true)] : []));
  const count = new Map<number, number>();
  for (const v of st.output) count.set(v, (count.get(v) ?? 0) + 1);
  const lost = kept.filter((v) => !count.has(v)).map((v) => v >>> 1);
  const duplicated = kept.filter((v) => (count.get(v) ?? 0) > 1).map((v) => v >>> 1);
  const keptSet = new Set(kept);
  let foreign = 0;
  for (let j = 0; j < Math.min(st.outputSize, N); j++) if (!keptSet.has(st.output[j])) foreign++;
  let extra = 0;
  for (let j = kept.length; j < N; j++) if (st.output[j] !== UNWRITTEN) extra++;
  const correct = !crash && st.outputSize === kept.length && !lost.length && !duplicated.length && !foreign && !extra;
  return {
    kind: crash ? 'crash' : correct ? 'correct' : 'wrong',
    size: st.outputSize,
    output: [...st.output],
    expectedSize: kept.length,
    lost, duplicated, foreign, extra, crash,
  };
}

/** One line for a table cell. */
export function summary(o: Outcome): string {
  if (o.kind === 'crash') return tr('an out-of-range write', 'một lần ghi ngoài phạm vi');
  if (o.kind === 'correct') return tr('correct', 'đúng');
  const parts: string[] = [];
  if (o.size !== o.expectedSize) parts.push(tr(`size ${o.size > N ? show(o.size) : o.size} (not ${o.expectedSize})`, `kích thước ${o.size > N ? show(o.size) : o.size} (không phải ${o.expectedSize})`));
  if (o.lost.length) parts.push(tr(`${o.lost.length} key${o.lost.length === 1 ? '' : 's'} lost`, `mất ${o.lost.length} khóa`));
  if (o.duplicated.length) parts.push(tr(`${o.duplicated.length} duplicated`, `${o.duplicated.length} bị trùng`));
  if (o.foreign) parts.push(tr(`${o.foreign} slot${o.foreign === 1 ? '' : 's'} of garbage`, `${o.foreign} ô chứa rác`));
  return parts.join(', ') || tr('wrong', 'sai');
}

// ---------------------------------------------------------------- the race check
export interface Race {
  sites: [Site, Site];
  space: 'shared' | 'global';
  example: { addr: string; a: { block: number; thread: number }; b: { block: number; thread: number } };
}
const SITE_ORDER: Site[] = [
  'zero', 'count', 'store', 'reserve-read', 'reserve-write', 'public-add',
  'commit-test', 'commit-read-start', 'commit-read-private', 'commit-write',
];
export function racesIn(acc: readonly Access[]): Race[] {
  const found = new Map<string, Race>();
  for (let x = 0; x < acc.length; x++)
    for (let y = x + 1; y < acc.length; y++) {
      const a = acc[x];
      const b = acc[y];
      if (a.space !== b.space || a.addr !== b.addr) continue;
      if (a.space === 'shared' && a.block !== b.block) continue; // each block has its own shared memory
      if (!a.write && !b.write) continue;
      if (a.atomic && b.atomic) continue;
      if (a.block === b.block && (a.thread === b.thread || a.epoch !== b.epoch)) continue;
      const sites = [a.site, b.site].sort((p, q) => SITE_ORDER.indexOf(p) - SITE_ORDER.indexOf(q)) as [Site, Site];
      const key = sites.join('|');
      if (!found.has(key))
        found.set(key, { sites, space: a.space, example: { addr: a.addr, a: { block: a.block, thread: a.thread }, b: { block: b.block, thread: b.thread } } });
    }
  return [...found.values()].sort((p, q) => SITE_ORDER.indexOf(p.sites[0]) - SITE_ORDER.indexOf(q.sites[0]) || SITE_ORDER.indexOf(p.sites[1]) - SITE_ORDER.indexOf(q.sites[1]));
}
/** Races found in any of the emulator's three runs (each run checked on its own, up to where it stops). */
export function races(shape: Shape, keep: readonly boolean[] = BOOK_KEEP): Race[] {
  const found = new Map<string, Race>();
  for (const s of SCHEDULES)
    for (const r of racesIn(simulate(shape, s, keep).accesses)) if (!found.has(r.sites.join('|'))) found.set(r.sites.join('|'), r);
  return [...found.values()].sort((p, q) => SITE_ORDER.indexOf(p.sites[0]) - SITE_ORDER.indexOf(q.sites[0]) || SITE_ORDER.indexOf(p.sites[1]) - SITE_ORDER.indexOf(q.sites[1]));
}

/** The source line of a site within a kernel's text (0-based), or -1. */
export function siteLine(kernelText: string, site: Site): number {
  const pattern: Record<Site, RegExp> = {
    zero: /outputSize_s = 0;/,
    count: /fetch_add\(1,|outputSize_s\+\+/,
    store: /output_s\[slot\] = val;/,
    'reserve-read': /blockStart = outputSize_ref\.fetch_add/,
    'reserve-write': /blockStart = outputSize_ref\.fetch_add/,
    'public-add': /blockStart = outputSize_ref\.fetch_add/,
    'commit-test': /threadIdx\.x < outputSize_s/,
    'commit-read-start': /output\[blockStart \+ threadIdx\.x\]/,
    'commit-read-private': /output\[blockStart \+ threadIdx\.x\]/,
    'commit-write': /output\[blockStart \+ threadIdx\.x\]/,
  };
  return kernelText.split('\n').findIndex((l) => pattern[site].test(l));
}

// ---------------------------------------------------------------- atomics on the public counter
/**
 * Atomics on the public counter for n keys, one per thread, a fraction f kept at random:
 * Fig. 12.2 issues one per kept key, a warp-aggregated filter (Fig. 12.3) one per warp that
 * keeps anything (expected value, keys kept independently), Fig. 12.6 one per block, kept
 * keys or not (p. 296). Our arithmetic, for section 12.4's question.
 */
export function publicAtomics(n: number, blockDim: number, f: number, warp = 32) {
  const warps = Math.ceil(n / warp);
  return {
    perKey: Math.round(n * f),
    perWarp: Math.round(warps * (1 - (1 - f) ** warp)),
    perBlock: Math.ceil(n / blockDim),
  };
}

// ---------------------------------------------------------------- the outcome table
export interface Row {
  variant: Variant;
  cells: Record<Schedule, Outcome>;
  races: Race[];
}
export function outcomeTable(keep: readonly boolean[] = BOOK_KEEP): Row[] {
  return VARIANTS.map((v) => ({
    variant: v,
    cells: Object.fromEntries(SCHEDULES.map((s) => [s, simulate(VARIANT[v].shape, s, keep).outcome])) as Record<Schedule, Outcome>,
    races: races(VARIANT[v].shape, keep),
  }));
}
