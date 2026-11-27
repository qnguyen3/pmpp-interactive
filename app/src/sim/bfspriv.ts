// One level of the privatised-frontier BFS of Fig. 18.15 (book §18.6, pp. 442-444), corrected,
// run the way the course's emulator runs it, for section 18.6's stepper "One block's private
// frontier".
//
// The kernels are labs/ch18/snippets/s186_privatised_variants.cu: the corrected kernel (line 29
// as an atomic store), the clamp correction, the book's plain store, and copies with one barrier
// dropped, an `if` commit, or no overflow check. kernelBody() writes the body of any
// combination; bfspriv.test.ts checks that each single change is exactly the snippet's kernel
// and runs the file on the emulator and its race flavour.
//
// simulate() hands a baton from thread to thread as the emu backend does
// (labs/common/include/pmpp/emu/emu_runtime.h): blocks one after another; inside a block a
// thread runs until it reaches a barrier or its end, then the next thread in the order; the
// forward, reverse and seeded shuffle orders (emuOrders from histpriv.ts, as privfilter.ts);
// shared memory filled with 0xCD bytes. A read or write outside an array stops the run, as
// AddressSanitizer stops the emulator. Two additions the emulator does not have: the order in
// which blocks run (and so reserve) can be chosen, and shared memory can start with a small
// leftover value instead of 0xCD bytes, as it may on a GPU.
//
// races() is a happens-before check of the kind ThreadSanitizer runs in the race flavour: two
// accesses to one location race when at least one writes, they are not both atomic, and
// neither is ordered before the other (same thread, or same block with a barrier between).
// Blocks are never ordered. Scopes are ignored, as the emulator ignores them.
//
// counterInterleavings() enumerates every interleaving of the appends to one private counter
// (ch18-facts E1, our check, not a vendor fact).
//
// What it leaves out: time, warps, caches, and any interleaving finer than one source statement.
//
// Erasable TypeScript only.
import { tr } from '../lib/locale.ts';
import { emuOrders, SCHEDULES, SCHEDULE_LABEL, type Schedule } from './histpriv';

export { SCHEDULES, SCHEDULE_LABEL, type Schedule };

/** What the emulator leaves in shared memory that nobody wrote: four 0xCD bytes. */
export const GARBAGE = 0xcdcdcdcd;
/** What the driver fills the public frontier with: four 0xFF bytes. */
export const UNWRITTEN = 0xffffffff;
export const UNVISITED = 0xffffffff;
/** The level the stepper builds; the previous frontier is at level 1. */
export const CURR_LEVEL = 2;
export const HUB_DEGREE = 10;
export const SNIPPET = 'ch18/snippets/s186_privatised_variants.cu';
export const MAX_CAPACITY = 8;

// ---------------------------------------------------------------- level presets
export type Preset = 'fanout' | 'shared' | 'hub';
export const PRESETS: Preset[] = ['fanout', 'shared', 'hub'];
export const PRESET_LABEL: Record<Preset, string> = {
  get fanout() {
    return tr('fan-out: every vertex finds 2 new ones', 'fan-out: mỗi đỉnh tìm thấy 2 đỉnh mới');
  },
  get shared() {
    return tr('shared: neighbouring vertices share a new one', 'shared: các đỉnh liền kề dùng chung một đỉnh mới');
  },
  get hub() {
    return tr(`hub: one vertex finds ${HUB_DEGREE}, the others 1`, `hub: một đỉnh tìm thấy ${HUB_DEGREE}, các đỉnh khác 1`);
  },
};
export interface Graph {
  preset: Preset;
  /** Previous-frontier vertices: 0 .. F-1, one per thread. */
  F: number;
  numVertices: number;
  srcPtrs: number[];
  dst: number[];
  /** The vertices this level must find, in id order. */
  expected: number[];
}
/** The graph the check driver builds for this preset (check/s186_privatised_check.cu). */
export function makeGraph(preset: Preset, blocks: number, threads: number): Graph {
  const F = blocks * threads;
  const out: number[][] = Array.from({ length: F }, () => []);
  let numVertices: number;
  if (preset === 'fanout') {
    for (let i = 0; i < F; i++) out[i] = [F + 2 * i, F + 2 * i + 1];
    numVertices = 3 * F;
  } else if (preset === 'shared') {
    for (let i = 0; i < F; i++) out[i] = [F + i, F + ((i + 1) % F)];
    numVertices = 2 * F;
  } else {
    for (let k = 0; k < HUB_DEGREE; k++) out[0].push(F + k);
    for (let i = 1; i < F; i++) out[i] = [F + HUB_DEGREE - 1 + i];
    numVertices = 2 * F + HUB_DEGREE - 1;
  }
  const srcPtrs = [0];
  const dst: number[] = [];
  for (let v = 0; v < numVertices; v++) {
    if (v < F) dst.push(...out[v]);
    srcPtrs.push(dst.length);
  }
  const expected = Array.from({ length: numVertices - F }, (_, k) => F + k);
  return { preset, F, numVertices, srcPtrs, dst, expected };
}

// ---------------------------------------------------------------- shapes and kernels
export type Reset = 'atomic' | 'clamp' | 'plain';
export const RESET_LABEL: Record<Reset, string> = {
  get atomic() {
    return tr('atomic store (corrected)', 'store nguyên tử (đã sửa)');
  },
  get clamp() {
    return tr('clamp after barrier 2 (corrected)', 'kẹp sau rào chắn 2 (đã sửa)');
  },
  get plain() {
    return tr('plain store (the book\'s line 29)', 'store thường (dòng 29 của sách)');
  },
};
export interface Shape {
  reset: Reset;
  barrier1: boolean;
  barrier2: boolean;
  barrier3: boolean;
  commit: 'loop' | 'if';
  overflowCheck: boolean;
}
export const CORRECT: Shape = { reset: 'atomic', barrier1: true, barrier2: true, barrier3: true, commit: 'loop', overflowCheck: true };

export type Variant =
  | 'correct' | 'clamp' | 'plain-store' | 'no-barrier-1' | 'no-barrier-2' | 'no-barrier-3' | 'if-commit' | 'no-overflow-check';
export const VARIANTS: Variant[] = ['correct', 'clamp', 'plain-store', 'no-barrier-1', 'no-barrier-2', 'no-barrier-3', 'if-commit', 'no-overflow-check'];
export const VARIANT: Record<Variant, { kernel: string; label: string; shape: Shape }> = {
  correct: {
    kernel: 'bfsPrivatised',
    get label() {
      return tr('corrected: atomic store', 'đã sửa: store nguyên tử');
    },
    shape: CORRECT,
  },
  clamp: {
    kernel: 'bfsPrivatisedClamp',
    get label() {
      return tr('corrected: clamp after barrier 2', 'đã sửa: kẹp sau rào chắn 2');
    },
    shape: { ...CORRECT, reset: 'clamp' },
  },
  'plain-store': {
    kernel: 'bfsPlainStore',
    get label() {
      return tr('the book\'s plain store', 'store thường của sách');
    },
    shape: { ...CORRECT, reset: 'plain' },
  },
  'no-barrier-1': {
    kernel: 'bfsNoBarrier1',
    get label() {
      return tr('barrier 1 removed', 'bỏ rào chắn 1');
    },
    shape: { ...CORRECT, barrier1: false },
  },
  'no-barrier-2': {
    kernel: 'bfsNoBarrier2',
    get label() {
      return tr('barrier 2 removed', 'bỏ rào chắn 2');
    },
    shape: { ...CORRECT, barrier2: false },
  },
  'no-barrier-3': {
    kernel: 'bfsNoBarrier3',
    get label() {
      return tr('barrier 3 removed', 'bỏ rào chắn 3');
    },
    shape: { ...CORRECT, barrier3: false },
  },
  'if-commit': {
    kernel: 'bfsIfCommit',
    get label() {
      return tr('commit with an if', 'commit bằng một lệnh if');
    },
    shape: { ...CORRECT, commit: 'if' },
  },
  'no-overflow-check': {
    kernel: 'bfsNoOverflowCheck',
    get label() {
      return tr('no overflow check', 'không kiểm tra tràn');
    },
    shape: { ...CORRECT, overflowCheck: false },
  },
};

type LineKey =
  | 'd1' | 'd2' | 'zero' | 'b1' | 'idx' | 'guard' | 'load' | 'loop' | 'nbr' | 'visit' | 'ref' | 'count'
  | 'test' | 'store' | 'else' | 'reset' | 'oref' | 'oadd' | 'ostore' | 'c4' | 'c3' | 'c2' | 'c1'
  | 'b2' | 'd3' | 't0' | 'clamp' | 'pref' | 'reserve' | 'c5' | 'b3' | 'cfor' | 'cidx' | 'cwrite' | 'c6';
const TEXT: Partial<Record<LineKey, string>> = {
  d1: '__shared__ unsigned int currFrontier_s[PRIVATE_FRONTIER_CAPACITY];', // i18n-ignore: kernel source
  d2: '__shared__ unsigned int numCurrFrontier_s;', // i18n-ignore: kernel source
  zero: 'if (threadIdx.x == 0) numCurrFrontier_s = 0;', // i18n-ignore: kernel source
  b1: '__syncthreads();  // barrier 1', // i18n-ignore: kernel source
  idx: 'unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;', // i18n-ignore: kernel source
  guard: 'if (i < numPrevFrontier) {', // i18n-ignore: kernel source
  load: '    unsigned int vertex = prevFrontier[i];', // i18n-ignore: kernel source
  loop: '    for (unsigned int edge = csrGraph.srcPtrs[vertex];\n            edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {', // i18n-ignore: kernel source
  nbr: '        unsigned int neighbor = csrGraph.dst[edge];', // i18n-ignore: kernel source
  visit: '        if (visitVertexAtomically(neighbor, level, currLevel)) {', // i18n-ignore: kernel source
  ref: '            cuda::atomic_ref<unsigned int, cuda::thread_scope_block>\n                numCurrFrontier_s_ref(numCurrFrontier_s);', // i18n-ignore: kernel source
  count: '            unsigned int currFrontierIdx_s = numCurrFrontier_s_ref\n                .fetch_add(1, cuda::memory_order_relaxed);', // i18n-ignore: kernel source
  test: '            if (currFrontierIdx_s < PRIVATE_FRONTIER_CAPACITY) {', // i18n-ignore: kernel source
  else: '            } else {', // i18n-ignore: kernel source
  oref: '                cuda::atomic_ref<unsigned int, cuda::thread_scope_device>\n                    numCurrFrontier_ref(*numCurrFrontier);', // i18n-ignore: kernel source
  oadd: '                unsigned int currFrontierIdx = numCurrFrontier_ref\n                    .fetch_add(1, cuda::memory_order_relaxed);', // i18n-ignore: kernel source
  ostore: '                currFrontier[currFrontierIdx] = neighbor;', // i18n-ignore: kernel source
  c4: '            }', // i18n-ignore: kernel source
  c3: '        }', // i18n-ignore: kernel source
  c2: '    }', // i18n-ignore: kernel source
  c1: '}', // i18n-ignore: kernel source
  b2: '__syncthreads();  // barrier 2', // i18n-ignore: kernel source
  d3: '__shared__ unsigned int currFrontierStartIdx;', // i18n-ignore: kernel source
  t0: 'if (threadIdx.x == 0) {', // i18n-ignore: kernel source
  clamp: '    if (numCurrFrontier_s > PRIVATE_FRONTIER_CAPACITY)\n        numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;', // i18n-ignore: kernel source
  pref: '    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>\n        numCurrFrontier_ref(*numCurrFrontier);', // i18n-ignore: kernel source
  reserve: '    currFrontierStartIdx = numCurrFrontier_ref\n        .fetch_add(numCurrFrontier_s, cuda::memory_order_relaxed);', // i18n-ignore: kernel source
  c5: '}', // i18n-ignore: kernel source
  b3: '__syncthreads();  // barrier 3', // i18n-ignore: kernel source
};
const RESET_TEXT: Record<'atomic' | 'plain', string> = {
  atomic: '                numCurrFrontier_s_ref.store(PRIVATE_FRONTIER_CAPACITY,\n                    cuda::memory_order_relaxed);',
  plain: '                numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;',
};

function bodyLines(shape: Shape, placeholders: boolean): { key: LineKey | null; text: string }[] {
  const out: { key: LineKey | null; text: string }[] = [];
  const add = (k: LineKey, text = TEXT[k]!) => out.push({ key: k, text });
  const gone = (n: number) => placeholders && out.push({ key: null, text: `// barrier ${n} removed` });
  add('d1'); add('d2'); add('zero');
  if (shape.barrier1) add('b1'); else gone(1);
  add('idx'); add('guard'); add('load'); add('loop'); add('nbr'); add('visit'); add('ref'); add('count');
  if (shape.overflowCheck) {
    add('test');
    add('store', '                currFrontier_s[currFrontierIdx_s] = neighbor;');
    add('else');
    if (shape.reset !== 'clamp') add('reset', RESET_TEXT[shape.reset]);
    add('oref'); add('oadd'); add('ostore'); add('c4');
  } else add('store', '            currFrontier_s[currFrontierIdx_s] = neighbor;');
  add('c3'); add('c2'); add('c1');
  if (shape.barrier2) add('b2'); else gone(2);
  add('d3'); add('t0');
  if (shape.reset === 'clamp') add('clamp');
  add('pref'); add('reserve'); add('c5');
  if (shape.barrier3) add('b3'); else gone(3);
  if (shape.commit === 'loop') {
    add('cfor', 'for (unsigned int currFrontierIdx_s = threadIdx.x;\n        currFrontierIdx_s < numCurrFrontier_s;\n        currFrontierIdx_s += blockDim.x) {');
    add('cidx', '    unsigned int currFrontierIdx =\n        currFrontierStartIdx + currFrontierIdx_s;');
    add('cwrite', '    currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];');
    add('c6', '}');
  } else {
    add('cfor', 'if (threadIdx.x < numCurrFrontier_s)');
    add('cwrite', '    currFrontier[currFrontierStartIdx + threadIdx.x] =\n        currFrontier_s[threadIdx.x];');
  }
  return out;
}
/**
 * The kernel body (between the braces, one indent level removed). With placeholders a removed
 * barrier leaves a comment line, so the stepper's listing shows where it was; without, it is
 * exactly the snippet file's kernel body.
 */
export function kernelBody(shape: Shape, placeholders = false): string {
  return bodyLines(shape, placeholders).map((l) => l.text).join('\n');
}
/** The 1-based lines of a statement in kernelBody(shape, placeholders), or [] when it is not there. */
export function linesOf(shape: Shape, placeholders: boolean, key: LineKey): number[] {
  let line = 1;
  for (const l of bodyLines(shape, placeholders)) {
    const rows = l.text.split('\n').length;
    if (l.key === key) return Array.from({ length: rows }, (_, k) => line + k);
    line += rows;
  }
  return [];
}
/** The snippet kernel with this shape, if there is one. */
export function variantOf(s: Shape): Variant | null {
  for (const v of VARIANTS) {
    const t = VARIANT[v].shape;
    if (t.reset === s.reset && t.barrier1 === s.barrier1 && t.barrier2 === s.barrier2 && t.barrier3 === s.barrier3 &&
      t.commit === s.commit && t.overflowCheck === s.overflowCheck) return v;
  }
  return null;
}

// ---------------------------------------------------------------- block orders
export type BlockOrder = 'in-order' | 'reversed' | 'random';
export const BLOCK_ORDER_LABEL: Record<BlockOrder, string> = {
  get 'in-order'() {
    return tr('0, 1, 2, ... (as the emulator)', '0, 1, 2, ... (như trình giả lập)');
  },
  get reversed() {
    return tr('last block first', 'khối cuối chạy trước');
  },
  get random() {
    return tr('random', 'ngẫu nhiên');
  },
};
export function blockOrderOf(kind: BlockOrder, blocks: number, seed = 1): number[] {
  const o = Array.from({ length: blocks }, (_, b) => b);
  if (kind === 'in-order') return o;
  if (kind === 'reversed') return o.reverse();
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
  | 'zero' | 'count' | 'store' | 'reset' | 'overflow-add' | 'overflow-store' | 'clamp'
  | 'reserve-read' | 'reserve-add' | 'reserve-write'
  | 'commit-test' | 'commit-read-start' | 'commit-read-private' | 'commit-write';
export const SITE_TEXT: Record<Site, string> = {
  get zero() {
    return tr('the zeroing of numCurrFrontier_s', 'lệnh đặt numCurrFrontier_s về 0');
  },
  get count() {
    return tr('the fetch_add on numCurrFrontier_s', 'fetch_add trên numCurrFrontier_s');
  },
  get store() {
    return tr('the store into currFrontier_s', 'lệnh store vào currFrontier_s');
  },
  get reset() {
    return tr('the store that puts numCurrFrontier_s back to capacity', 'lệnh store đặt numCurrFrontier_s về lại dung lượng');
  },
  get 'overflow-add'() {
    return tr('the overflow fetch_add on the public counter', 'fetch_add tràn trên bộ đếm chung');
  },
  get 'overflow-store'() {
    return tr('the overflow store into currFrontier', 'lệnh store tràn vào currFrontier');
  },
  get clamp() {
    return tr('the clamp of numCurrFrontier_s', 'lệnh kẹp numCurrFrontier_s');
  },
  get 'reserve-read'() {
    return tr('thread 0\'s read of numCurrFrontier_s for the reservation', 'lần đọc numCurrFrontier_s của luồng 0 để đặt chỗ');
  },
  get 'reserve-add'() {
    return tr('the reservation on the public counter', 'lần đặt chỗ trên bộ đếm chung');
  },
  get 'reserve-write'() {
    return tr('thread 0\'s write of currFrontierStartIdx', 'lần ghi currFrontierStartIdx của luồng 0');
  },
  get 'commit-test'() {
    return tr('the read of numCurrFrontier_s in the commit', 'lần đọc numCurrFrontier_s trong bước commit');
  },
  get 'commit-read-start'() {
    return tr('the read of currFrontierStartIdx in the commit', 'lần đọc currFrontierStartIdx trong bước commit');
  },
  get 'commit-read-private'() {
    return tr('the read of currFrontier_s in the commit', 'lần đọc currFrontier_s trong bước commit');
  },
  get 'commit-write'() {
    return tr('the commit store to currFrontier', 'lệnh store của bước commit vào currFrontier');
  },
};
const SITE_ORDER = Object.keys(SITE_TEXT) as Site[];

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
export type Op = 'zero' | 'load' | 'visit' | 'count' | 'store' | 'reset' | 'append' | 'reserve' | 'commit' | 'skip' | 'arrive' | 'release';
export type ViolationKind =
  | 'late-zero' | 'private-overwrite' | 'short-reservation' | 'garbage-start' | 'garbage-copy'
  | 'public-overwrite' | 'uncopied' | 'crash';
export interface Violation {
  kind: ViolationKind;
  message: string;
}
export interface BlockState {
  private_: number[];
  counter: number;
  start: number;
  written: { private_: boolean[]; counter: boolean; start: boolean };
  status: string[];
}
export interface Snapshot {
  blocks: BlockState[];
  /** The public frontier, currFrontier. */
  pub: number[];
  /** Per public slot: the block that wrote it and how ('overflow' or 'commit'), or null. */
  how: ({ block: number; via: 'overflow' | 'commit' } | null)[];
  /** The public counter, *numCurrFrontier. */
  count: number;
  /** Atomic operations on the public counter so far. */
  globalAtomics: number;
  overflows: number;
}
export interface Step {
  block: number;
  /** -1 for a barrier opening. */
  thread: number;
  op: Op;
  barrier?: 1 | 2 | 3;
  /** 1-based lines of kernelBody(shape, true) to mark. */
  lines: number[];
  text: string;
  violation?: Violation;
  note?: string;
  after: Snapshot;
}
export interface Reservation {
  block: number;
  start: number;
  count: number;
  step: number;
}
export interface Outcome {
  kind: 'correct' | 'wrong' | 'crash';
  size: number;
  expectedSize: number;
  lost: number[];
  duplicated: number[];
  /** Slots below the public counter that hold no expected vertex. */
  foreign: number;
  /** Slots at or above the public counter that were written. */
  extra: number;
  crash?: string;
  /** Each block's committed entries form one contiguous range. */
  grouped: boolean;
}
export interface Tally {
  /** Compare-and-swaps on level[]: one per edge looked at, in both kernels. */
  cas: number;
  newVertices: number;
  /** Fig. 18.12: one atomic on the public counter per new vertex. */
  fig1812: number;
  /** Fig. 18.15: one per block (the reservation, even of 0) plus one per overflow. */
  fig1815: number;
  reservations: number;
  overflows: number;
}
export interface Trace {
  shape: Shape;
  schedule: Schedule;
  capacity: number;
  blocks: number;
  threads: number;
  graph: Graph;
  blockOrder: number[];
  leftover: number;
  initial: Snapshot;
  steps: Step[];
  accesses: Access[];
  reservations: Reservation[];
  /** Per block: the public addresses each pass of the commit wrote, pass by pass. */
  commitPasses: Record<number, number[][]>;
  outcome: Outcome;
  tally: Tally;
  firstFailure: number;
}

const show = (v: number) => (v === GARBAGE ? '0xCDCDCDCD' : v === UNWRITTEN ? '0xFFFFFFFF' : String(v));
const plural = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;

function cloneSnap(s: Snapshot): Snapshot {
  return {
    blocks: s.blocks.map((b) => ({
      private_: [...b.private_], counter: b.counter, start: b.start,
      written: { private_: [...b.written.private_], counter: b.written.counter, start: b.written.start },
      status: [...b.status],
    })),
    pub: [...s.pub],
    how: s.how.map((h) => (h ? { ...h } : null)),
    count: s.count,
    globalAtomics: s.globalAtomics,
    overflows: s.overflows,
  };
}

type Yield = { barrier: 1 | 2 | 3 } | null;

export interface Options {
  capacity: number;
  blocks: number;
  threads: number;
  preset: Preset;
  blockOrder?: number[];
  /** What shared memory holds before anyone writes it: the emulator's 0xCD bytes, or a small leftover. */
  leftover?: number;
}

/** One launch of one level: `blocks` blocks of `threads` threads, one previous-frontier vertex per thread. */
export function simulate(shape: Shape, schedule: Schedule, o: Options): Trace {
  const { capacity: CAP, blocks: B, threads: T, preset } = o;
  if (!(CAP >= 1 && CAP <= MAX_CAPACITY)) throw new Error(`capacity ${CAP}`);
  const graph = makeGraph(preset, B, T);
  const blockOrder = o.blockOrder ?? blockOrderOf('in-order', B);
  const leftover = o.leftover ?? GARBAGE;
  const n = graph.numVertices;
  const level = Array.from({ length: n }, (_, v) => (v < graph.F ? CURR_LEVEL - 1 : UNVISITED));
  const fresh = (): BlockState => ({
    private_: new Array<number>(CAP).fill(leftover),
    counter: leftover,
    start: leftover,
    written: { private_: new Array<boolean>(CAP).fill(false), counter: false, start: false },
    status: new Array<string>(T).fill(tr('not started', 'chưa bắt đầu')),
  });
  const st: Snapshot = {
    blocks: Array.from({ length: B }, fresh),
    pub: new Array<number>(n).fill(UNWRITTEN),
    how: new Array(n).fill(null),
    count: 0,
    globalAtomics: 0,
    overflows: 0,
  };
  const initial = cloneSnap(st);
  const steps: Step[] = [];
  const accesses: Access[] = [];
  const reservations: Reservation[] = [];
  const commitPasses: Record<number, number[][]> = {};
  const orders = emuOrders(schedule, B, T);
  const L = (k: LineKey) => linesOf(shape, true, k);
  let crashed: string | undefined;
  let cas = 0;
  let wins = 0;

  for (let pos = 0; pos < B && !crashed; pos++) {
    const b = blockOrder[pos];
    const sh = st.blocks[b];
    const epoch = new Array<number>(T).fill(0);
    let privateStores = 0;
    let increments = 0;
    let reservation: Reservation | null = null;
    const push = (t: number, s: Omit<Step, 'after' | 'block' | 'thread'>) => steps.push({ ...s, block: b, thread: t, after: cloneSnap(st) });
    const acc = (t: number, a: Omit<Access, 'block' | 'thread' | 'epoch'>) => accesses.push({ ...a, block: b, thread: t, epoch: epoch[t] });
    const who = (t: number) => tr(`Block ${b}, thread ${t}`, `Khối ${b}, luồng ${t}`);
    const crash = (t: number, op: Op, lines: number[], text: string, msg: string) => {
      crashed = msg;
      push(t, { op, lines, text, violation: { kind: 'crash', message: tr(`${msg}: the emulator stops here with an AddressSanitizer report`, `${msg}: trình giả lập dừng ở đây với một báo cáo AddressSanitizer`) } });
    };

    function* thread(t: number): Generator<Yield, void, void> {
      if (t === 0) {
        sh.counter = 0;
        sh.written.counter = true;
        acc(t, { space: 'shared', addr: 'numCurrFrontier_s', write: true, atomic: false, site: 'zero' });
        push(t, {
          op: 'zero', lines: L('zero'), text: tr(`${who(t)} sets the private counter numCurrFrontier_s to 0.`, `${who(t)} đặt bộ đếm riêng numCurrFrontier_s về 0.`),
          violation: privateStores
            ? { kind: 'late-zero', message: tr(`${plural(privateStores, 'vertex has', 'vertices have')} already been stored in the private queue from a count that was never zeroed; zeroing now hands those slots out again`, `${privateStores} đỉnh đã được lưu vào hàng đợi riêng từ một bộ đếm chưa từng được đặt về 0; đặt về 0 lúc này sẽ cấp lại những ô đó lần nữa`) }
            : undefined,
          note: !privateStores && increments
            ? tr(`Late: ${plural(increments, 'vertex was', 'vertices were')} already counted from the leftover value. Above capacity, they took the overflow path, so nothing is lost this time.`, `Muộn: ${increments} đỉnh đã được đếm từ giá trị còn sót lại. Vượt quá dung lượng, chúng đã đi theo nhánh tràn, nên lần này không mất gì.`)
            : undefined,
        });
        yield null;
      }
      if (shape.barrier1) yield { barrier: 1 };
      const i = b * T + t;
      const vertex = i; // prevFrontier[i] = i
      const first = graph.srcPtrs[vertex];
      const last = graph.srcPtrs[vertex + 1];
      push(t, {
        op: 'load', lines: L('load'),
        text: tr(`${who(t)}: i = ${b}·${T} + ${t} = ${i}, so its vertex is prevFrontier[${i}] = ${vertex}, with ${plural(last - first, 'out-edge')}.`, `${who(t)}: i = ${b}·${T} + ${t} = ${i}, nên đỉnh của nó là prevFrontier[${i}] = ${vertex}, với ${last - first} cạnh đi ra.`),
      });
      yield null;
      for (let e = first; e < last; e++) {
        const nb = graph.dst[e];
        cas++;
        const won = level[nb] === UNVISITED;
        if (won) {
          level[nb] = CURR_LEVEL;
          wins++;
        }
        push(t, {
          op: 'visit', lines: L('visit'),
          text: won
            ? tr(`${who(t)} looks at neighbour ${nb}: the compare-and-swap finds it unvisited and labels it level ${CURR_LEVEL}. It is new, so it must be appended.`, `${who(t)} xét đỉnh kề ${nb}: compare-and-swap thấy nó chưa được thăm và gán cho nó mức ${CURR_LEVEL}. Đỉnh này mới, nên phải được thêm vào.`)
            : tr(`${who(t)} looks at neighbour ${nb}: the compare-and-swap fails, another thread labelled it first. Nothing to append.`, `${who(t)} xét đỉnh kề ${nb}: compare-and-swap thất bại, một luồng khác đã gán mức cho nó trước. Không có gì để thêm.`),
        });
        yield null;
        if (!won) continue;
        const idx = sh.counter;
        const fromLeftover = !sh.written.counter;
        sh.counter = (idx + 1) >>> 0;
        increments++;
        acc(t, { space: 'shared', addr: 'numCurrFrontier_s', write: true, atomic: true, site: 'count' });
        const over = idx >= CAP;
        push(t, {
          op: 'count', lines: L('count'),
          text: tr(
            `${who(t)} does fetch_add(1) on numCurrFrontier_s: it gets ${show(idx)}, and the counter becomes ${show(sh.counter)}.`,
            `${who(t)} thực hiện fetch_add(1) trên numCurrFrontier_s: nó nhận ${show(idx)}, và bộ đếm trở thành ${show(sh.counter)}.`,
          ) +
            (shape.overflowCheck ? (over ? tr(` That is not below the capacity of ${CAP}: overflow.`, ` Giá trị đó không nhỏ hơn dung lượng ${CAP}: tràn.`) : tr(` Below the capacity of ${CAP}: a private slot.`, ` Nhỏ hơn dung lượng ${CAP}: một ô riêng.`)) : ''),
          note: fromLeftover
            ? tr(
                `The counter has not been zeroed yet: it still holds the leftover ${show(idx)}.${over && shape.overflowCheck ? ' Far above capacity, so this vertex takes the overflow path and is not lost.' : ''}`,
                `Bộ đếm chưa được đặt về 0: nó vẫn giữ giá trị còn sót lại ${show(idx)}.${over && shape.overflowCheck ? ' Vượt xa dung lượng, nên đỉnh này đi theo nhánh tràn và không bị mất.' : ''}`,
              )
            : undefined,
        });
        yield null;
        if (shape.overflowCheck && over) {
          if (shape.reset !== 'clamp') {
            sh.counter = CAP;
            acc(t, { space: 'shared', addr: 'numCurrFrontier_s', write: true, atomic: shape.reset === 'atomic', site: 'reset' });
            push(t, {
              op: 'reset', lines: L('reset'),
              text: tr(
                `${who(t)} puts numCurrFrontier_s back to ${CAP}${shape.reset === 'atomic' ? ' with an atomic store through the same atomic_ref' : ' with a plain store'}.`,
                `${who(t)} đặt numCurrFrontier_s về lại ${CAP}${shape.reset === 'atomic' ? ' bằng một store nguyên tử qua cùng atomic_ref đó' : ' bằng một store thường'}.`,
              ),
              note: shape.reset === 'plain'
                ? tr(
                    'A plain store to a counter other threads of the block may be adding to at this moment: a data race. Taking turns, as here, it gives the right number. The race check below reports it.',
                    'Một store thường vào bộ đếm mà các luồng khác của khối có thể đang cộng vào ngay lúc này: một race condition (tranh chấp dữ liệu). Khi các luồng lần lượt chạy, như ở đây, nó cho đúng con số. Phép kiểm tra race bên dưới báo cáo nó.',
                  )
                : undefined,
            });
            yield null;
          }
          const j = st.count;
          st.count = (j + 1) >>> 0;
          st.globalAtomics++;
          st.overflows++;
          accesses.push({ block: b, thread: t, epoch: epoch[t], space: 'global', addr: 'numCurrFrontier', write: true, atomic: true, site: 'overflow-add' });
          if (j >= n) {
            crash(t, 'append', [...L('oadd'), ...L('ostore')], tr(`${who(t)} appends ${nb} straight to the public frontier at slot ${j}.`, `${who(t)} thêm ${nb} thẳng vào frontier chung tại ô ${j}.`), tr(`currFrontier[${j}] is past the end of the ${n}-slot public frontier`, `currFrontier[${j}] nằm quá cuối frontier chung có ${n} ô`));
            return;
          }
          const prevHow = st.how[j];
          st.pub[j] = nb;
          st.how[j] = { block: b, via: 'overflow' };
          accesses.push({ block: b, thread: t, epoch: epoch[t], space: 'global', addr: `currFrontier[${j}]`, write: true, atomic: false, site: 'overflow-store' });
          push(t, {
            op: 'append', lines: [...L('oadd'), ...L('ostore')],
            text: tr(`${who(t)} appends ${nb} straight to the public frontier: fetch_add(1) on the public counter gives slot ${j}, a global atomic.`, `${who(t)} thêm ${nb} thẳng vào frontier chung: fetch_add(1) trên bộ đếm chung cho ô ${j}, một atomic trên bộ nhớ toàn cục.`),
            violation: prevHow ? { kind: 'public-overwrite', message: tr(`currFrontier[${j}] already held ${st.pub[j]} from block ${prevHow.block}`, `currFrontier[${j}] đã chứa ${st.pub[j]} từ khối ${prevHow.block}`) } : undefined,
          });
          yield null;
        } else {
          if (idx >= CAP) {
            crash(t, 'store', L('store'), tr(`${who(t)} stores ${nb} into currFrontier_s[${show(idx)}].`, `${who(t)} lưu ${nb} vào currFrontier_s[${show(idx)}].`), tr(`currFrontier_s[${show(idx)}] is outside the ${CAP}-entry private queue`, `currFrontier_s[${show(idx)}] nằm ngoài hàng đợi riêng có ${CAP} phần tử`));
            return;
          }
          const had = sh.written.private_[idx] ? sh.private_[idx] : null;
          sh.private_[idx] = nb;
          sh.written.private_[idx] = true;
          privateStores++;
          acc(t, { space: 'shared', addr: `currFrontier_s[${idx}]`, write: true, atomic: false, site: 'store' });
          push(t, {
            op: 'store', lines: L('store'), text: tr(`${who(t)} stores ${nb} into currFrontier_s[${idx}].`, `${who(t)} lưu ${nb} vào currFrontier_s[${idx}].`),
            violation: had !== null
              ? { kind: 'private-overwrite', message: tr(`currFrontier_s[${idx}] already held ${had}, which is now lost`, `currFrontier_s[${idx}] đã chứa ${had}, giá trị này giờ bị mất`) }
              : undefined,
          });
          yield null;
        }
      }
      if (shape.barrier2) yield { barrier: 2 };
      if (t === 0) {
        if (shape.reset === 'clamp') {
          const before = sh.counter;
          acc(t, { space: 'shared', addr: 'numCurrFrontier_s', write: false, atomic: false, site: 'clamp' });
          if (before > CAP) {
            sh.counter = CAP;
            acc(t, { space: 'shared', addr: 'numCurrFrontier_s', write: true, atomic: false, site: 'clamp' });
          }
          push(t, {
            op: 'reserve', lines: L('clamp'),
            text: before > CAP
              ? tr(`${who(t)} clamps numCurrFrontier_s from ${show(before)} to the capacity, ${CAP}: the first ${CAP} new vertices are in the private queue, the rest went to the public frontier.`, `${who(t)} kẹp numCurrFrontier_s từ ${show(before)} xuống dung lượng, ${CAP}: ${CAP} đỉnh mới đầu tiên nằm trong hàng đợi riêng, phần còn lại đã vào frontier chung.`)
              : tr(`${who(t)} checks numCurrFrontier_s = ${show(before)} against the capacity, ${CAP}: nothing to clamp.`, `${who(t)} so numCurrFrontier_s = ${show(before)} với dung lượng, ${CAP}: không có gì để kẹp.`),
          });
          yield null;
        }
        const size = sh.counter;
        const start = st.count;
        st.count = (start + size) >>> 0;
        st.globalAtomics++;
        sh.start = start;
        sh.written.start = true;
        acc(t, { space: 'shared', addr: 'numCurrFrontier_s', write: false, atomic: false, site: 'reserve-read' });
        accesses.push({ block: b, thread: t, epoch: epoch[t], space: 'global', addr: 'numCurrFrontier', write: true, atomic: true, site: 'reserve-add' });
        acc(t, { space: 'shared', addr: 'currFrontierStartIdx', write: true, atomic: false, site: 'reserve-write' });
        reservation = { block: b, start, count: size, step: steps.length };
        reservations.push(reservation);
        push(t, {
          op: 'reserve', lines: L('reserve'),
          text: tr(`${who(t)} reserves the block's range with one fetch_add(${show(size)}) on the public counter: it goes from ${start} to ${show(st.count)}, so currFrontierStartIdx = ${start}. One global atomic for the whole block.`, `${who(t)} đặt chỗ cho dải của khối bằng một fetch_add(${show(size)}) trên bộ đếm chung: bộ đếm đi từ ${start} lên ${show(st.count)}, nên currFrontierStartIdx = ${start}. Một atomic toàn cục cho cả khối.`),
        });
        yield null;
      }
      if (shape.barrier3) yield { barrier: 3 };
      // The commit.
      const cmp = (k: number) => {
        acc(t, { space: 'shared', addr: 'numCurrFrontier_s', write: false, atomic: false, site: 'commit-test' });
        return k < sh.counter;
      };
      const copy = (k: number): boolean => {
        const start = sh.start;
        const startGarbage = !sh.written.start;
        acc(t, { space: 'shared', addr: 'currFrontierStartIdx', write: false, atomic: false, site: 'commit-read-start' });
        if (k >= CAP) {
          crash(t, 'commit', L('cwrite'), tr(`${who(t)} copies currFrontier_s[${k}].`, `${who(t)} sao chép currFrontier_s[${k}].`), tr(`currFrontier_s[${k}] is outside the ${CAP}-entry private queue`, `currFrontier_s[${k}] nằm ngoài hàng đợi riêng có ${CAP} phần tử`));
          return false;
        }
        const v = sh.private_[k];
        const garbage = !sh.written.private_[k];
        acc(t, { space: 'shared', addr: `currFrontier_s[${k}]`, write: false, atomic: false, site: 'commit-read-private' });
        const j = (start + k) >>> 0;
        if (j >= n) {
          crash(t, 'commit', L('cwrite'), tr(`${who(t)} copies currFrontier_s[${k}] to currFrontier[${show(start)} + ${k}].`, `${who(t)} sao chép currFrontier_s[${k}] sang currFrontier[${show(start)} + ${k}].`),
            tr(
              `${startGarbage ? 'currFrontierStartIdx has not been written yet, so ' : ''}currFrontier[${j}] is past the end of the ${n}-slot public frontier`,
              `${startGarbage ? 'currFrontierStartIdx chưa được ghi, nên ' : ''}currFrontier[${j}] nằm quá cuối frontier chung có ${n} ô`,
            ));
          return false;
        }
        const prev = st.how[j];
        st.pub[j] = v;
        st.how[j] = { block: b, via: 'commit' };
        accesses.push({ block: b, thread: t, epoch: epoch[t], space: 'global', addr: `currFrontier[${j}]`, write: true, atomic: false, site: 'commit-write' });
        (commitPasses[b] ??= []);
        const pass = Math.floor(k / T);
        (commitPasses[b][pass] ??= []).push(j);
        push(t, {
          op: 'commit', lines: L('cwrite'),
          text: tr(`${who(t)} copies currFrontier_s[${k}] (${show(v)}) to currFrontier[${start} + ${k}] = currFrontier[${j}].`, `${who(t)} sao chép currFrontier_s[${k}] (${show(v)}) sang currFrontier[${start} + ${k}] = currFrontier[${j}].`),
          violation: startGarbage
            ? { kind: 'garbage-start', message: tr(`currFrontierStartIdx has not been written yet: it holds the leftover ${show(start)}`, `currFrontierStartIdx chưa được ghi: nó giữ giá trị còn sót lại ${show(start)}`) }
            : garbage
              ? { kind: 'garbage-copy', message: tr(`currFrontier_s[${k}] was never written: the leftover ${show(v)} goes into the public frontier`, `currFrontier_s[${k}] chưa từng được ghi: giá trị còn sót lại ${show(v)} đi vào frontier chung`) }
              : prev && prev.block !== b
                ? { kind: 'public-overwrite', message: tr(`currFrontier[${j}] already held a vertex from block ${prev.block}, which is now lost`, `currFrontier[${j}] đã chứa một đỉnh từ khối ${prev.block}, đỉnh đó giờ bị mất`) }
                : prev && prev.via === 'overflow'
                  ? { kind: 'public-overwrite', message: tr(`currFrontier[${j}] held an overflow vertex, which is now lost`, `currFrontier[${j}] đã chứa một đỉnh tràn, đỉnh đó giờ bị mất`) }
                  : undefined,
        });
        return true;
      };
      if (shape.commit === 'loop') {
        for (let k = t; ; k += T) {
          if (!cmp(k)) {
            push(t, { op: 'skip', lines: L('cfor'), text: tr(`${who(t)}: ${k} < numCurrFrontier_s (${show(sh.counter)}) is false, so its commit loop ends.`, `${who(t)}: ${k} < numCurrFrontier_s (${show(sh.counter)}) sai, nên vòng lặp commit của nó kết thúc.`) });
            yield null;
            break;
          }
          if (!copy(k)) return;
          yield null;
        }
      } else if (cmp(t)) {
        if (!copy(t)) return;
        yield null;
      } else {
        push(t, { op: 'skip', lines: L('cfor'), text: tr(`${who(t)}: ${t} < numCurrFrontier_s (${show(sh.counter)}) is false, so it copies nothing.`, `${who(t)}: ${t} < numCurrFrontier_s (${show(sh.counter)}) sai, nên nó không sao chép gì.`) });
        yield null;
      }
    }

    const gens = Array.from({ length: T }, (_, t) => thread(t));
    const done = new Array<boolean>(T).fill(false);
    const waiting = new Array<0 | 1 | 2 | 3>(T).fill(0);
    while (!crashed) {
      for (const t of orders[pos]) {
        if (done[t] || waiting[t] || crashed) continue;
        sh.status[t] = tr('running', 'đang chạy');
        for (;;) {
          const r = gens[t].next();
          if (crashed) break;
          if (r.done) {
            done[t] = true;
            sh.status[t] = tr('finished', 'đã xong');
            break;
          }
          if (r.value) {
            waiting[t] = r.value.barrier;
            sh.status[t] = tr(`at barrier ${r.value.barrier}`, `tại rào chắn ${r.value.barrier}`);
            push(t, { op: 'arrive', barrier: r.value.barrier, lines: L(`b${r.value.barrier}` as LineKey), text: tr(`${who(t)} reaches barrier ${r.value.barrier} and waits for the rest of its block.`, `${who(t)} tới rào chắn ${r.value.barrier} và chờ phần còn lại của khối.`) });
            break;
          }
        }
      }
      if (crashed || done.every(Boolean)) break;
      const n0 = waiting.find((w) => w) as 1 | 2 | 3;
      for (let t = 0; t < T; t++)
        if (waiting[t]) {
          waiting[t] = 0;
          epoch[t]++;
          sh.status[t] = tr(`past barrier ${n0}`, `đã qua rào chắn ${n0}`);
        }
      const why =
        n0 === 1 ? tr('the private counter is 0 before anyone counts', 'bộ đếm riêng bằng 0 trước khi ai bắt đầu đếm')
          : n0 === 2 ? tr('every new vertex is in the private queue or the public frontier, so numCurrFrontier_s is final', 'mọi đỉnh mới đều đã nằm trong hàng đợi riêng hoặc frontier chung, nên numCurrFrontier_s là giá trị cuối cùng')
            : tr('currFrontierStartIdx is written, so every thread can read it', 'currFrontierStartIdx đã được ghi, nên mọi luồng đều đọc được');
      push(-1, { op: 'release', barrier: n0, lines: L(`b${n0}` as LineKey), text: tr(`Barrier ${n0} opens for block ${b}: ${why}.`, `Rào chắn ${n0} mở cho khối ${b}: ${why}.`) });
    }
    if (crashed) break;
    // After the block: was its reservation right, and did the commit copy everything?
    if (reservation) {
      const r: Reservation = reservation;
      if (r.count !== sh.counter && !steps[r.step].violation)
        steps[r.step].violation = {
          kind: 'short-reservation',
          message: tr(`thread 0 reserves ${show(r.count)} public slot${r.count === 1 ? '' : 's'}, but the private queue ends with ${show(sh.counter)}: it read the counter before every thread had counted`, `luồng 0 đặt chỗ ${show(r.count)} ô chung, nhưng hàng đợi riêng kết thúc với ${show(sh.counter)}: nó đã đọc bộ đếm trước khi mọi luồng đếm xong`),
        };
    }
    if (shape.commit === 'if' && sh.counter > T) {
      let last = steps.length - 1;
      while (last >= 0 && (steps[last].block !== b || steps[last].thread < 0)) last--;
      if (last >= 0 && !steps[last].violation)
        steps[last].violation = {
          kind: 'uncopied',
          message: tr(`the private queue holds ${sh.counter} entries but the block has ${T} threads: entries ${T} to ${sh.counter - 1} are never copied, and their reserved public slots stay empty`, `hàng đợi riêng chứa ${sh.counter} phần tử nhưng khối có ${T} luồng: các phần tử ${T} đến ${sh.counter - 1} không bao giờ được sao chép, và các ô chung đã đặt cho chúng vẫn trống`),
        };
    }
  }

  const outcome = classify(st, graph, crashed, reservations);
  return {
    shape, schedule, capacity: CAP, blocks: B, threads: T, graph, blockOrder: [...blockOrder], leftover,
    initial, steps, accesses, reservations, commitPasses, outcome,
    tally: { cas, newVertices: wins, fig1812: wins, fig1815: st.globalAtomics, reservations: reservations.length, overflows: st.overflows },
    firstFailure: steps.findIndex((s) => !!s.violation),
  };
}

function classify(st: Snapshot, g: Graph, crash: string | undefined, res: Reservation[]): Outcome {
  const expected = new Set(g.expected);
  const seen = new Map<number, number>();
  const size = Math.min(st.count, g.numVertices);
  for (let j = 0; j < size; j++) seen.set(st.pub[j], (seen.get(st.pub[j]) ?? 0) + 1);
  const lost = g.expected.filter((v) => !seen.has(v));
  const duplicated = g.expected.filter((v) => (seen.get(v) ?? 0) > 1);
  let foreign = 0;
  for (let j = 0; j < size; j++) if (!expected.has(st.pub[j])) foreign++;
  let extra = 0;
  for (let j = size; j < g.numVertices; j++) if (st.pub[j] !== UNWRITTEN) extra++;
  // Every block's committed slots form its reserved range.
  const grouped = res.every((r) => {
    for (let j = r.start; j < r.start + r.count && j < g.numVertices; j++) {
      const h = st.how[j];
      if (!h || h.block !== r.block || h.via !== 'commit') return false;
    }
    return true;
  });
  const correct = !crash && st.count === g.expected.length && !lost.length && !duplicated.length && !foreign && !extra;
  return {
    kind: crash ? 'crash' : correct ? 'correct' : 'wrong',
    size: st.count, expectedSize: g.expected.length, lost, duplicated, foreign, extra, crash, grouped,
  };
}

/** One line for a table cell or a chip. */
export function summary(o: Outcome): string {
  if (o.kind === 'crash') return tr('stopped: an out-of-range access', 'đã dừng: một truy cập ngoài phạm vi');
  if (o.kind === 'correct') return tr('correct', 'đúng');
  const parts: string[] = [];
  if (o.size !== o.expectedSize) parts.push(tr(`size ${show(o.size)} (not ${o.expectedSize})`, `kích thước ${show(o.size)} (không phải ${o.expectedSize})`));
  if (o.lost.length) parts.push(tr(`${plural(o.lost.length, 'vertex', 'vertices')} lost`, `mất ${o.lost.length} đỉnh`));
  if (o.duplicated.length) parts.push(tr(`${o.duplicated.length} duplicated`, `${o.duplicated.length} bị trùng`));
  if (o.foreign) parts.push(tr(`${plural(o.foreign, 'slot')} of garbage`, `${o.foreign} ô chứa rác`));
  return parts.join(', ') || tr('wrong', 'sai');
}

// ---------------------------------------------------------------- the race check
export interface Race {
  sites: [Site, Site];
  space: 'shared' | 'global';
  example: { addr: string; a: { block: number; thread: number }; b: { block: number; thread: number } };
}
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
/** Races found in any of the emulator's three orders (each run checked up to where it stops). */
export function races(shape: Shape, o: Options): Race[] {
  const found = new Map<string, Race>();
  for (const s of SCHEDULES)
    for (const r of racesIn(simulate(shape, s, o).accesses)) if (!found.has(r.sites.join('|'))) found.set(r.sites.join('|'), r);
  return [...found.values()].sort((p, q) => SITE_ORDER.indexOf(p.sites[0]) - SITE_ORDER.indexOf(q.sites[0]) || SITE_ORDER.indexOf(p.sites[1]) - SITE_ORDER.indexOf(q.sites[1]));
}
const SITE_PATTERN: Record<Site, RegExp> = {
  zero: /numCurrFrontier_s = 0;/,
  count: /currFrontierIdx_s = numCurrFrontier_s_ref/,
  store: /currFrontier_s\[currFrontierIdx_s\] = neighbor;/,
  reset: /numCurrFrontier_s_ref\.store\(|^\s{16,}numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;/,
  'overflow-add': /unsigned int currFrontierIdx = numCurrFrontier_ref/,
  'overflow-store': /currFrontier\[currFrontierIdx\] = neighbor;/,
  clamp: /if \(numCurrFrontier_s > PRIVATE_FRONTIER_CAPACITY\)/,
  'reserve-read': /currFrontierStartIdx = numCurrFrontier_ref/,
  'reserve-add': /currFrontierStartIdx = numCurrFrontier_ref/,
  'reserve-write': /currFrontierStartIdx = numCurrFrontier_ref/,
  'commit-test': /currFrontierIdx_s < numCurrFrontier_s;|threadIdx\.x < numCurrFrontier_s/,
  'commit-read-start': /currFrontierStartIdx \+ currFrontierIdx_s|currFrontier\[currFrontierStartIdx \+ threadIdx\.x\]/,
  'commit-read-private': /= currFrontier_s\[currFrontierIdx_s\];|^\s*currFrontier_s\[threadIdx\.x\];/,
  'commit-write': /currFrontier\[currFrontierIdx\] = currFrontier_s|currFrontier\[currFrontierStartIdx \+ threadIdx\.x\] =/,
};
/** The 0-based line of a site within a kernel's text, or -1. */
export function siteLine(kernelText: string, site: Site): number {
  return kernelText.split('\n').findIndex((l) => SITE_PATTERN[site].test(l));
}
/** The 1-based line of a site in the stepper's listing, kernelBody(shape, true), or 0. */
export function listingLine(shape: Shape, site: Site): number {
  return siteLine(kernelBody(shape, true), site) + 1;
}

// ---------------------------------------------------------------- E1: every interleaving
/**
 * Every interleaving of n appends to one private counter of capacity cap: each thread does
 * fetch_add(1); below capacity it stores privately; at or above, the atomic or plain reset
 * stores cap (the clamp form stores nothing), then it appends publicly. Each step is one
 * sequentially consistent action. Returns the counter values the reservation can read after
 * the barrier (after the clamp, for the clamp form), how many interleavings there are, and
 * whether every thread landed exactly once with distinct private slots below cap.
 */
export function counterInterleavings(n: number, cap: number, reset: Reset) {
  type State = { pc: number[]; idx: number[]; counter: number };
  const finals = new Set<number>();
  let ok = true;
  const memo = new Map<string, number>();
  // pc: 0 = before fetch_add; 1 = after it (next: private store, or reset); 2 = after reset (next: public append); 3 = done.
  const next = (s: State, t: number): State | null => {
    const pc = s.pc[t];
    if (pc === 3) return null;
    const ns: State = { pc: [...s.pc], idx: [...s.idx], counter: s.counter };
    if (pc === 0) {
      ns.idx[t] = s.counter;
      ns.counter = s.counter + 1;
      ns.pc[t] = 1;
    } else if (pc === 1) {
      if (s.idx[t] < cap) ns.pc[t] = 3;
      else if (reset === 'clamp') ns.pc[t] = 2;
      else {
        ns.counter = cap;
        ns.pc[t] = 2;
      }
    } else ns.pc[t] = 3; // the public append does not touch the counter
    return ns;
  };
  const walk = (s: State): number => {
    const key = s.pc.join(',') + '|' + s.idx.join(',') + '|' + s.counter;
    const m = memo.get(key);
    if (m !== undefined) return m;
    let count = 0;
    let any = false;
    for (let t = 0; t < n; t++) {
      const ns = next(s, t);
      if (!ns) continue;
      any = true;
      count += walk(ns);
    }
    if (!any) {
      const final = reset === 'clamp' ? Math.min(s.counter, cap) : s.counter;
      finals.add(final);
      const priv = s.idx.filter((i) => i < cap);
      if (new Set(priv).size !== priv.length || priv.some((i) => i >= cap)) ok = false;
      count = 1;
    }
    memo.set(key, count);
    return count;
  };
  const interleavings = walk({ pc: new Array<number>(n).fill(0), idx: new Array<number>(n).fill(-1), counter: 0 });
  return { finals: [...finals].sort((a, b) => a - b), interleavings, ok };
}

/** Section 18.6's numbers for q-atomics: atomics on the global counter for one level. */
export function levelAtomics(appended: number, blocks: number, overflows = 0) {
  return { fig1812: appended, fig1815: blocks + overflows };
}
