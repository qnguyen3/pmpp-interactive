// Every BFS level in one cooperative launch (book §18.7, pp. 445-448), at the grain of the
// "Three grid barriers" stepper, the "Does it fit?" tab, the levels-and-launches prediction
// and the narrated scene "One launch for every level" of section 18.6.
//
// Five parts:
//   1. Frontier sizes from formulas: a W x W 4-neighbour grid searched from a corner has
//      min(k + 1, 2W - 1 - k) vertices at level k and 2(W - 1) levels; a b-ary tree has b^k.
//      The per-level host loop launches once per non-empty frontier, the last launch finding
//      nothing. (Our arithmetic; gridbfs.test.ts checks it against a BFS written in the test.)
//   2. The stride assignment of Fig. 18.17 lines 21-22: frontier element i goes to grid
//      thread i mod G.
//   3. Residency (pp. 445-446): a grid of G blocks on R resident slots. Under an ordinary
//      launch the resident blocks wait at a grid barrier for blocks that cannot start; under
//      a cooperative launch CUDA and HIP refuse the grid at launch (ch18-facts A3, B3).
//   4. simulate(): the kernels of labs/ch18/snippets/s186_coop_swap.cu (the swap of Fig.
//      18.17 lines 65-74 with each grid.sync() optional, and Exercise 3's counter per level)
//      on a few blocks of a few threads, run the way the emu backend runs a cooperative
//      launch (labs/common/include/pmpp/emu/emu_runtime.h): every block alive at once; in
//      each pass over the order every ready thread runs until it reaches a grid.sync() or
//      leaves the kernel; then a grid barrier that every thread has reached opens. The orders
//      are the emulator's (forward, reverse, seeded shuffle; emuOrders from histpriv.ts).
//      Two rules for a barrier that some threads never reach because they have left the
//      kernel: 'gpu' (the documented contract of sync(): every thread of the group reaches it
//      before any passes, ch18-facts A5, so it never opens: a deadlock) and 'emu' (the
//      emulator counts only threads that have not exited, so the others pass). gridbfs.test.ts
//      runs the snippet on the emulator and compares with the 'emu' rule.
//   5. Timelines for the scene, in levels and launches, never in seconds.
//
// What it leaves out: time, warps, caches, the private queue of Fig. 18.17 (the snippet
// appends straight to the public counter, as Fig. 18.12 does), and any interleaving finer
// than one grid barrier to the next.
//
// Erasable TypeScript only. Does not import bfs.ts.
import { emuOrders, SCHEDULES, SCHEDULE_LABEL as EMU_LABEL, type Schedule } from './histpriv';
import { tr } from '../lib/locale.ts';

export { SCHEDULES, type Schedule };
export const SCHEDULE_LABEL: Record<Schedule, string> = {
  get forward() {
    return tr(`${EMU_LABEL.forward} (thread 0 first)`, `${EMU_LABEL.forward} (luồng 0 trước)`);
  },
  get reverse() {
    return tr(`${EMU_LABEL.reverse} (thread 0 last)`, `${EMU_LABEL.reverse} (luồng 0 sau cùng)`);
  },
  get shuffle() {
    return EMU_LABEL.shuffle;
  },
};

export const UNVISITED = 0xffffffff;
export const UNWRITTEN = 0xffffffff;
export const SNIPPET = 'ch18/snippets/s186_coop_swap.cu';
/** The emulator's own cap on a cooperative grid (labs/README.md); it describes the emulator, not a GPU. */
export const EMU_COOP_LIMIT = 1024;

// ---------------------------------------------------------------- 1. frontier sizes
/** Vertices at level k of a W x W 4-neighbour grid searched from a corner: the k-th anti-diagonal. */
export function gridFrontier(W: number, k: number): number {
  if (!Number.isInteger(W) || W < 1) throw new Error(`W ${W}`);
  if (k < 0 || k > 2 * (W - 1)) return 0;
  return Math.min(k + 1, 2 * W - 1 - k);
}
/** Vertices at level k of a b-ary tree of the given depth searched from its root. */
export const treeFrontier = (b: number, depth: number, k: number) => (k < 0 || k > depth ? 0 : b ** k);

export interface LevelSummary {
  /** Frontier sizes from level 0 (the root) to the last non-empty level. */
  sizes: number[];
  /** Levels after the root: the depth of the search. */
  levels: number;
  /** Launches of a per-level kernel: one per non-empty frontier; the last finds nothing new. */
  launches: number;
  maxFrontier: number;
  vertices: number;
}
function summarise(sizes: number[]): LevelSummary {
  return {
    sizes,
    levels: sizes.length - 1,
    launches: sizes.length,
    maxFrontier: Math.max(...sizes),
    vertices: sizes.reduce((a, b) => a + b, 0),
  };
}
/** A W x W grid from a corner. For W = 1000: 1,998 levels, 1,999 launches, frontiers never above 1,000. */
export function gridSummary(W: number): LevelSummary {
  return summarise(Array.from({ length: 2 * W - 1 }, (_, k) => gridFrontier(W, k)));
}
/** Levels are counted without materialising sizes; for W in the millions. */
export const gridLevels = (W: number) => 2 * (W - 1);
export const gridLaunches = (W: number) => 2 * (W - 1) + 1;
export function treeSummary(b: number, depth: number): LevelSummary {
  return summarise(Array.from({ length: depth + 1 }, (_, k) => treeFrontier(b, depth, k)));
}

// ---------------------------------------------------------------- 2. the stride assignment
/** Fig. 18.17 lines 21-22: thread r takes elements r, r + G, r + 2G, ... of an n-element frontier. */
export function strideAssignment(n: number, G: number): number[][] {
  const out = Array.from({ length: G }, () => [] as number[]);
  for (let i = 0; i < n; i++) out[i % G].push(i);
  return out;
}

// ---------------------------------------------------------------- 3. residency
export const TOO_LARGE = {
  cuda: 'cudaErrorCooperativeLaunchTooLarge',
  hip: 'hipErrorCooperativeLaunchTooLarge',
  metal: null,
} as const;

export interface Residency {
  /** Resident slots: SMs x blocks per SM (the book's recipe, p. 446). */
  slots: number;
  grid: number;
  fits: boolean;
  /** Under an ordinary launch: blocks that start and reach the grid barrier. */
  resident: number;
  /** Blocks that wait for a slot that is never freed (0 when everything fits). */
  pending: number;
  /** Under an ordinary launch, the book's argument (p. 445): the barrier completes only if every block is resident. */
  ordinary: 'completes' | 'never completes'; // i18n-ignore: union member, used as a data attribute
  /** Under a cooperative launch: CUDA and HIP check the same product at launch (A2, A3, B3). */
  cooperative: 'launches' | 'refused';
}
export function residency(sms: number, blocksPerSM: number, grid: number): Residency {
  const slots = sms * blocksPerSM;
  const fits = grid <= slots;
  return {
    slots, grid, fits,
    resident: Math.min(slots, grid),
    pending: Math.max(0, grid - slots),
    ordinary: fits ? 'completes' : 'never completes', // i18n-ignore: union member, used as a data attribute
    cooperative: fits ? 'launches' : 'refused',
  };
}

// ---------------------------------------------------------------- 4. the kernels
export type Kernel = 'swap' | 'ex3';
export interface Shape {
  kernel: Kernel;
  /** The swap's three barriers; Exercise 3 uses sync1 only, its single barrier. */
  sync1: boolean;
  sync2: boolean;
  sync3: boolean;
}
export const CORRECT: Shape = { kernel: 'swap', sync1: true, sync2: true, sync3: true };
export const EX3: Shape = { kernel: 'ex3', sync1: true, sync2: true, sync3: true };

export type Variant = 'swap' | 'no-sync-1' | 'no-sync-2' | 'no-sync-3' | 'ex3' | 'ex3-no-sync';
export const VARIANTS: Variant[] = ['swap', 'no-sync-1', 'no-sync-2', 'no-sync-3', 'ex3', 'ex3-no-sync'];
export const VARIANT: Record<Variant, { kernel: string; label: string; shape: Shape }> = {
  swap: {
    kernel: 'bfsCoop',
    get label() {
      return tr('all three barriers', 'cả ba rào chắn');
    },
    shape: CORRECT,
  },
  'no-sync-1': {
    kernel: 'bfsCoopNoSync1',
    get label() {
      return tr('grid.sync() 1 removed', 'bỏ grid.sync() 1');
    },
    shape: { ...CORRECT, sync1: false },
  },
  'no-sync-2': {
    kernel: 'bfsCoopNoSync2',
    get label() {
      return tr('grid.sync() 2 removed', 'bỏ grid.sync() 2');
    },
    shape: { ...CORRECT, sync2: false },
  },
  'no-sync-3': {
    kernel: 'bfsCoopNoSync3',
    get label() {
      return tr('grid.sync() 3 removed', 'bỏ grid.sync() 3');
    },
    shape: { ...CORRECT, sync3: false },
  },
  ex3: {
    kernel: 'bfsCoopEx3',
    get label() {
      return tr('Exercise 3: a counter per level', 'Bài tập 3: một bộ đếm cho mỗi mức');
    },
    shape: EX3,
  },
  'ex3-no-sync': {
    kernel: 'bfsCoopEx3NoSync',
    get label() {
      return tr('Exercise 3 without its barrier', 'Bài tập 3 không có rào chắn của nó');
    },
    shape: { ...EX3, sync1: false },
  },
};
export function variantOf(s: Shape): Variant | null {
  for (const v of VARIANTS) {
    const t = VARIANT[v].shape;
    if (t.kernel !== s.kernel || t.sync1 !== s.sync1) continue;
    if (s.kernel === 'ex3' || (t.sync2 === s.sync2 && t.sync3 === s.sync3)) return v;
  }
  return null;
}

type LineKey =
  | 'grid' | 'loop' | 'work' | 's1' | 'read' | 's2' | 'reset' | 's3' | 'swap' | 'end';
const WORK_HEAD = [
  'for (unsigned int i = grid.thread_rank(); i < numPrevFrontier;',
  '        i += grid.num_threads()) {',
  '    unsigned int vertex = prevFrontier[i];', // i18n-ignore: kernel source
  '    for (unsigned int edge = csrGraph.srcPtrs[vertex];', // i18n-ignore: kernel source
  '            edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {',
  '        unsigned int neighbor = csrGraph.dst[edge];', // i18n-ignore: kernel source
  '        if (visitVertexAtomically(neighbor, level, currLevel)) {', // i18n-ignore: kernel source
  '            cuda::atomic_ref<unsigned int, cuda::thread_scope_device>',
];
const WORK_TAIL = ['            currFrontier[currFrontierIdx] = neighbor;', '        }', '    }', '}'];
function workText(kernel: Kernel): string {
  const ref = kernel === 'swap' ? 'numCurrFrontier_ref' : 'numFrontier_ref';
  const arg = kernel === 'swap' ? '(*numCurrFrontier);' : '(numFrontier[currLevel]);';
  return [
    ...WORK_HEAD,
    `                ${ref}${arg}`,
    `            unsigned int currFrontierIdx = ${ref}`,
    '                .fetch_add(1, cuda::memory_order_relaxed);',
    ...WORK_TAIL,
  ].map((l) => '    ' + l).join('\n');
}
function bodyLines(shape: Shape, placeholders: boolean): { key: LineKey | null; text: string }[] {
  const out: { key: LineKey | null; text: string }[] = [];
  const add = (key: LineKey, text: string) => out.push({ key, text });
  const gone = (text: string) => placeholders && out.push({ key: null, text });
  add('grid', 'cg::grid_group grid = cg::this_grid();');
  add('loop', 'for (unsigned int currLevel = 1; numPrevFrontier > 0; ++currLevel) {');
  add('work', workText(shape.kernel));
  if (shape.kernel === 'ex3') {
    if (shape.sync1) add('s1', '    grid.sync();  // the only barrier: every append is done'); // i18n-ignore: kernel source, must match the snippet file
    else gone('    // the barrier removed');
    add('read', '    numPrevFrontier = numFrontier[currLevel];');
  } else {
    if (shape.sync1) add('s1', '    grid.sync();  // 1: every append is done'); // i18n-ignore: kernel source, must match the snippet file
    else gone('    // grid.sync() 1 removed');
    add('read', '    numPrevFrontier = *numCurrFrontier;');
    if (shape.sync2) add('s2', '    grid.sync();  // 2: everyone has read the size'); // i18n-ignore: kernel source, must match the snippet file
    else gone('    // grid.sync() 2 removed');
    add('reset', '    if (grid.thread_rank() == 0) {\n        *numCurrFrontier = 0;\n    }');
    if (shape.sync3) add('s3', '    grid.sync();  // 3: the reset is visible'); // i18n-ignore: kernel source, must match the snippet file
    else gone('    // grid.sync() 3 removed');
  }
  add('swap', '    unsigned int* tmp = prevFrontier;\n    prevFrontier = currFrontier;\n    currFrontier = tmp;');
  add('end', '}');
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
/** The 1-based lines of a statement in kernelBody(shape, placeholders). */
export function linesOf(shape: Shape, placeholders: boolean, key: LineKey): number[] {
  let line = 1;
  for (const l of bodyLines(shape, placeholders)) {
    const rows = l.text.split('\n').length;
    if (l.key === key) return Array.from({ length: rows }, (_, k) => line + k);
    line += rows;
  }
  return [];
}

// ---------------------------------------------------------------- graphs
export interface Graph {
  label: string;
  numVertices: number;
  srcPtrs: number[];
  dst: number[];
  /** For drawing: the grid's width, or the tree's branching factor and depth. */
  kind: 'grid' | 'tree';
  W?: number;
  b?: number;
  depth?: number;
}
function csr(out: number[][]): { srcPtrs: number[]; dst: number[] } {
  const srcPtrs = [0];
  const dst: number[] = [];
  for (const o of out) {
    dst.push(...o);
    srcPtrs.push(dst.length);
  }
  return { srcPtrs, dst };
}
/** A W x W grid, vertex y*W + x, neighbours up, left, right, down (ascending id), as the driver builds it. */
export function gridGraph(W: number): Graph {
  const out: number[][] = [];
  for (let y = 0; y < W; y++)
    for (let x = 0; x < W; x++) {
      const o: number[] = [];
      if (y > 0) o.push((y - 1) * W + x);
      if (x > 0) o.push(y * W + x - 1);
      if (x + 1 < W) o.push(y * W + x + 1);
      if (y + 1 < W) o.push((y + 1) * W + x);
      out.push(o);
    }
  return { label: tr(`a ${W} × ${W} grid from a corner`, `lưới ${W} × ${W} tính từ một góc`), numVertices: W * W, ...csr(out), kind: 'grid', W };
}
/** A b-ary tree of the given depth, vertex v linked to b*v + 1 ... b*v + b. */
export function treeGraph(b: number, depth: number): Graph {
  let n = 0;
  for (let d = 0, w = 1; d <= depth; d++, w *= b) n += w;
  const out: number[][] = Array.from({ length: n }, (_, v) => {
    const o: number[] = [];
    for (let c = 1; c <= b; c++) if (b * v + c < n) o.push(b * v + c);
    return o;
  });
  return { label: tr(`a ${b}-ary tree of depth ${depth}`, `cây ${b}-phân có độ sâu ${depth}`), numVertices: n, ...csr(out), kind: 'tree', b, depth };
}
export type Preset = 'grid' | 'tree';
export const PRESET_GRAPH: Record<Preset, () => Graph> = {
  grid: () => gridGraph(5),
  tree: () => treeGraph(4, 2),
};
/** Reference BFS levels from vertex 0 (a plain queue). */
export function referenceLevels(g: Graph): number[] {
  const lv = new Array<number>(g.numVertices).fill(UNVISITED);
  lv[0] = 0;
  const q = [0];
  for (let h = 0; h < q.length; h++) {
    const v = q[h];
    for (let e = g.srcPtrs[v]; e < g.srcPtrs[v + 1]; e++) {
      const u = g.dst[e];
      if (lv[u] === UNVISITED) {
        lv[u] = lv[v] + 1;
        q.push(u);
      }
    }
  }
  return lv;
}

// ---------------------------------------------------------------- the run
export type Rule = 'gpu' | 'emu';
export type Buf = 'A' | 'B';
export interface ThreadState {
  rank: number;
  block: number;
  currLevel: number;
  numPrev: number;
  prev: Buf;
  curr: Buf;
  status: 'ready' | 'waiting' | 'exited';
  /** The grid.sync() it waits at. */
  at?: 1 | 2 | 3;
  /** The last level whose traversal it has finished (0 before any). */
  worked: number;
}
export interface Snapshot {
  threads: ThreadState[];
  level: number[];
  A: number[];
  B: number[];
  /** numCurrFrontier, or numFrontier[0..] for Exercise 3. */
  counters: number[];
}
/** `step` is the index in Trace.steps of the step that makes the event. */
export type CounterEvent =
  | { kind: 'append'; step: number; rank: number; level: number; slot: number; vertex: number; after: number }
  | { kind: 'read'; step: number; rank: number; level: number; value: number }
  | { kind: 'reset'; step: number; rank: number; level: number; before: number };
export type ViolationKind = 'partial-count' | 'read-after-reset' | 'early-exit' | 'late-reset' | 'deadlock' | 'crash';
export interface Violation {
  kind: ViolationKind;
  message: string;
}
export type Op = 'work' | 'read' | 'reset' | 'swap' | 'release' | 'deadlock';
export interface Element {
  i: number;
  vertex: number;
  appended: { vertex: number; slot: number }[];
  /** Neighbours someone else had already labelled. */
  taken: number[];
}
export interface Step {
  /** -1 for a barrier opening or the deadlock. */
  rank: number;
  op: Op;
  level: number;
  /** After this step the thread waits at this grid.sync(). */
  waits?: 1 | 2 | 3;
  /** For a release: the barrier that opens. */
  barrier?: 1 | 2 | 3;
  elements?: Element[];
  value?: number;
  exits?: boolean;
  /** 1-based lines of kernelBody(shape, true) to mark. */
  lines: number[];
  text: string;
  violation?: Violation;
  note?: string;
  after: Snapshot;
}
export interface Outcome {
  kind: 'correct' | 'partial-count' | 'early-exit' | 'lost-appends' | 'wrong' | 'crash';
  /** One line for a chip: the root cause, the first thing that went wrong. */
  label: string;
  /** What the root cause led to, when it led to an early exit. */
  consequence?: string;
  deadlock: boolean;
  /** The levels in memory at the end match a BFS (false after a deadlock: the search never finished). */
  levelsCorrect: boolean;
  /** Vertices still unvisited, or labelled with the wrong level, at the end. */
  wrongVertices: number;
  crash?: string;
}
export interface Trace {
  shape: Shape;
  schedule: Schedule;
  rule: Rule;
  graph: Graph;
  blocks: number;
  threads: number;
  initial: Snapshot;
  steps: Step[];
  history: CounterEvent[];
  outcome: Outcome;
  firstFailure: number;
  reference: number[];
}

const cloneSnap = (s: Snapshot): Snapshot => ({
  threads: s.threads.map((t) => ({ ...t })),
  level: [...s.level],
  A: [...s.A],
  B: [...s.B],
  counters: [...s.counters],
});
type Phase = 'work' | 'read' | 'reset' | 'swap';
const plural = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;
const list = (xs: number[]) => (xs.length <= 1 ? xs.join('') : xs.slice(0, -1).join(', ') + tr(' and ', ' và ') + xs[xs.length - 1]);

/**
 * The whole search in one cooperative launch of `blocks` blocks of `threads` threads. The
 * per-thread loop is the snippet's kernel; the order is the emulator's; `rule` says what a
 * grid barrier does when some threads have left the kernel.
 */
export function simulate(
  shape: Shape,
  schedule: Schedule,
  graph: Graph,
  blocks: number,
  threads: number,
  rule: Rule = 'gpu',
): Trace {
  const G = blocks * threads;
  if (G > EMU_COOP_LIMIT) throw new Error('a grid the emulator would refuse'); // i18n-ignore: developer error
  const n = graph.numVertices;
  const ex3 = shape.kernel === 'ex3';
  const st: Snapshot = {
    threads: Array.from({ length: G }, (_, r) => ({
      rank: r, block: Math.floor(r / threads), currLevel: 1, numPrev: 1, prev: 'A', curr: 'B', status: 'ready', worked: 0,
    })),
    level: Array.from({ length: n }, (_, v) => (v === 0 ? 0 : UNVISITED)),
    A: Array.from({ length: n }, (_, v) => (v === 0 ? 0 : UNWRITTEN)),
    B: new Array<number>(n).fill(UNWRITTEN),
    counters: new Array<number>(ex3 ? n + 1 : 1).fill(0),
  };
  const initial = cloneSnap(st);
  const steps: Step[] = [];
  const history: CounterEvent[] = [];
  const phase: Phase[] = new Array<Phase>(G).fill('work');
  const order = emuOrders(schedule, 1, G)[0];
  const appendsAt = new Map<number, number>(); // level -> appends made for it
  const resetDone = new Set<number>(); // levels whose counter thread 0 has reset
  const L = (key: LineKey) => linesOf(shape, true, key);
  const who = (r: number) => tr(`Thread ${r} (block ${Math.floor(r / threads)})`, `Luồng ${r} (khối ${Math.floor(r / threads)})`);
  const reference = referenceLevels(graph);
  /** The true size of each level's frontier. */
  const refSize: number[] = [];
  for (const l of reference) if (l !== UNVISITED) refSize[l] = (refSize[l] ?? 0) + 1;
  let crashed: string | undefined;
  let deadlocked = false;

  const push = (s: Omit<Step, 'after'>) => steps.push({ ...s, after: cloneSnap(st) });
  const counterFor = (lvl: number) => (ex3 ? lvl : 0);
  const waitAfter = (r: number, n: 1 | 2 | 3) => {
    st.threads[r].status = 'waiting';
    st.threads[r].at = n;
    return n;
  };

  // Run thread r until it waits at a grid barrier or leaves the kernel.
  const runSegment = (r: number) => {
    const t = st.threads[r];
    for (;;) {
      if (crashed) return;
      const p = phase[r];
      if (p === 'work') {
        const lvl = t.currLevel;
        const bufP = t.prev === 'A' ? st.A : st.B;
        const bufC = t.curr === 'A' ? st.A : st.B;
        const els: Element[] = [];
        for (let i = r; i < t.numPrev; i += G) {
          const vertex = bufP[i];
          if (vertex >= n) {
            crashed = tr(
              `thread ${r} reads prevFrontier[${i}] = ${vertex === UNWRITTEN ? '0xFFFFFFFF (never written)' : vertex}, which is not a vertex, and indexes srcPtrs with it`,
              `luồng ${r} đọc prevFrontier[${i}] = ${vertex === UNWRITTEN ? '0xFFFFFFFF (chưa từng được ghi)' : vertex}, không phải một đỉnh, và dùng nó làm chỉ số vào srcPtrs`,
            );
            break;
          }
          const el: Element = { i, vertex, appended: [], taken: [] };
          for (let e = graph.srcPtrs[vertex]; e < graph.srcPtrs[vertex + 1]; e++) {
            const nb = graph.dst[e];
            if (st.level[nb] !== UNVISITED) {
              el.taken.push(nb);
              continue;
            }
            st.level[nb] = lvl;
            const c = counterFor(lvl);
            if (c >= st.counters.length) {
              crashed = tr(`numFrontier[${lvl}] is past the end of the counters`, `numFrontier[${lvl}] nằm quá phần cuối của dãy bộ đếm`);
              break;
            }
            const slot = st.counters[c];
            st.counters[c] = slot + 1;
            if (slot >= n) {
              crashed = tr(
                `thread ${r} writes currFrontier[${slot}], past the end of the ${n}-slot buffer`,
                `luồng ${r} ghi currFrontier[${slot}], quá phần cuối của buffer ${n} vị trí`,
              );
              break;
            }
            bufC[slot] = nb;
            el.appended.push({ vertex: nb, slot });
            appendsAt.set(lvl, (appendsAt.get(lvl) ?? 0) + 1);
            history.push({ kind: 'append', step: steps.length, rank: r, level: lvl, slot, vertex: nb, after: slot + 1 });
          }
          els.push(el);
          if (crashed) break;
        }
        t.worked = lvl;
        phase[r] = 'read';
        const got = els.flatMap((x) => x.appended);
        const mine = els.length
          ? tr(
              `It takes ${els.map((x) => `element ${x.i} (vertex ${x.vertex})`).join(', ')}` +
                (got.length ? ` and appends ${plural(got.length, 'new vertex', 'new vertices')}: ${got.map((a) => `${a.vertex} at slot ${a.slot}`).join(', ')}.` : ', but every neighbour is already labelled.'),
              `Nó nhận ${els.map((x) => `phần tử ${x.i} (đỉnh ${x.vertex})`).join(', ')}` +
                (got.length ? ` và nối thêm ${got.length} đỉnh mới: ${got.map((a) => `${a.vertex} ở vị trí ${a.slot}`).join(', ')}.` : ', nhưng mọi đỉnh kề đều đã được gán nhãn.'),
            )
          : tr(
              `Its rank is not below numPrevFrontier = ${t.numPrev}, so it has no element to take.`,
              `Rank của nó không nhỏ hơn numPrevFrontier = ${t.numPrev}, nên nó không có phần tử nào để nhận.`,
            );
        const waits = shape.sync1 ? waitAfter(r, 1) : undefined;
        push({
          rank: r, op: 'work', level: lvl, elements: els, lines: L('work'), waits,
          text: tr(
            `${who(r)}, level ${lvl}: ${mine}${waits ? ' Then it waits at grid.sync() 1.' : ''}`,
            `${who(r)}, mức ${lvl}: ${mine}${waits ? ' Rồi nó chờ tại grid.sync() 1.' : ''}`,
          ),
          violation: crashed
            ? {
                kind: 'crash',
                message: tr(
                  `${crashed}: the emulator stops here with an AddressSanitizer report`,
                  `${crashed}: trình giả lập dừng ở đây với một báo cáo AddressSanitizer`,
                ),
              }
            : undefined,
        });
        if (crashed || waits) return;
        continue;
      }
      if (p === 'read') {
        const lvl = t.currLevel;
        const c = counterFor(lvl);
        const value = st.counters[c];
        t.numPrev = value;
        history.push({ kind: 'read', step: steps.length, rank: r, level: lvl, value });
        const unfinished = st.threads.filter((u) => u.rank !== r && u.status !== 'exited' && u.worked < lvl).map((u) => u.rank);
        let violation: Violation | undefined;
        let note: string | undefined;
        if (!ex3 && resetDone.has(lvl)) {
          violation = {
            kind: 'read-after-reset',
            message: tr(
              `grid thread 0 has already reset the counter for this level, so thread ${r} reads ${value} instead of the size of level ${lvl}`,
              `luồng 0 của lưới đã đặt lại bộ đếm của mức này, nên luồng ${r} đọc được ${value} thay vì kích thước của mức ${lvl}`,
            ),
          };
        } else if (unfinished.length) {
          // Judged after the run, when the level's full size is known.
          note = `unfinished:${unfinished.join(',')}`; // i18n-ignore: internal marker, replaced before the run returns
        }
        if (ex3) phase[r] = 'swap';
        else phase[r] = 'reset';
        const waits = !ex3 && shape.sync2 ? waitAfter(r, 2) : undefined;
        push({
          rank: r, op: 'read', level: lvl, value, lines: L('read'), waits, violation, note,
          text: tr(
            `${who(r)} reads ${ex3 ? `numFrontier[${lvl}]` : '*numCurrFrontier'} = ${value} into its own numPrevFrontier.${waits ? ' It waits at grid.sync() 2.' : ''}`,
            `${who(r)} đọc ${ex3 ? `numFrontier[${lvl}]` : '*numCurrFrontier'} = ${value} vào numPrevFrontier của riêng nó.${waits ? ' Nó chờ tại grid.sync() 2.' : ''}`,
          ),
        });
        if (waits) return;
        continue;
      }
      if (p === 'reset') {
        const lvl = t.currLevel;
        phase[r] = 'swap';
        const waits = shape.sync3 ? waitAfter(r, 3) : undefined;
        if (r === 0) {
          const before = st.counters[0];
          const early = Math.max(0, (appendsAt.get(lvl + 1) ?? 0));
          st.counters[0] = 0;
          resetDone.add(lvl);
          history.push({ kind: 'reset', step: steps.length, rank: r, level: lvl, before });
          push({
            rank: r, op: 'reset', level: lvl, lines: L('reset'), waits,
            text: tr(
              `${who(r)} is grid thread 0: it sets *numCurrFrontier from ${before} to 0.${waits ? ' It waits at grid.sync() 3.' : ''}`,
              `${who(r)} là luồng 0 của lưới: nó đặt *numCurrFrontier từ ${before} về 0.${waits ? ' Nó chờ tại grid.sync() 3.' : ''}`,
            ),
            violation: early
              ? {
                  kind: 'late-reset',
                  message: tr(
                    `${plural(early, 'append')} of level ${lvl + 1} ${early === 1 ? 'was' : 'were'} already counted; the reset throws ${early === 1 ? 'it' : 'them'} away, and the next appends overwrite their slots`,
                    `${early} lần nối thêm của mức ${lvl + 1} đã được đếm; việc đặt lại vứt bỏ chúng, và các lần nối thêm tiếp theo ghi đè lên vị trí của chúng`,
                  ),
                }
              : undefined,
          });
        } else {
          push({
            rank: r, op: 'reset', level: lvl, lines: L('reset'), waits,
            text: tr(
              `${who(r)} is not grid thread 0, so it skips the reset.${waits ? ' It waits at grid.sync() 3.' : ''}`,
              `${who(r)} không phải luồng 0 của lưới, nên nó bỏ qua bước đặt lại.${waits ? ' Nó chờ tại grid.sync() 3.' : ''}`,
            ),
          });
        }
        if (waits) return;
        continue;
      }
      // swap and loop test
      const lvl = t.currLevel;
      [t.prev, t.curr] = [t.curr, t.prev];
      t.currLevel = lvl + 1;
      const exits = t.numPrev === 0;
      if (exits) t.status = 'exited';
      phase[r] = 'work';
      const size = refSize[lvl] ?? 0;
      push({
        rank: r, op: 'swap', level: lvl, exits, lines: [...L('swap'), ...L('loop')],
        text: exits
          ? tr(
              `${who(r)} swaps its pointers and tests numPrevFrontier = 0 > 0: false, so it leaves the loop and the kernel.`,
              `${who(r)} hoán đổi các con trỏ và kiểm tra numPrevFrontier = 0 > 0: sai, nên nó rời vòng lặp và kernel.`,
            )
          : tr(
              `${who(r)} swaps its pointers (prevFrontier is now ${t.prev}) and tests numPrevFrontier = ${t.numPrev} > 0: it goes on to level ${lvl + 1}.`,
              `${who(r)} hoán đổi các con trỏ (prevFrontier giờ là ${t.prev}) và kiểm tra numPrevFrontier = ${t.numPrev} > 0: nó đi tiếp sang mức ${lvl + 1}.`,
            ),
        violation: exits && size > 0
          ? {
              kind: 'early-exit',
              message: tr(
                `level ${lvl} has ${plural(size, 'vertex', 'vertices')}, but this thread read 0 and leaves while the search is not finished`,
                `mức ${lvl} có ${size} đỉnh, nhưng luồng này đọc được 0 và rời đi khi việc tìm kiếm chưa xong`,
              ),
            }
          : undefined,
      });
      if (exits) return;
    }
  };

  for (let guard = 0; guard < 100000; guard++) {
    for (const r of order) {
      if (crashed) break;
      if (st.threads[r].status === 'ready') runSegment(r);
    }
    if (crashed) break;
    const live = st.threads.filter((t) => t.status !== 'exited');
    if (!live.length) break;
    const waiting = live.filter((t) => t.status === 'waiting');
    if (waiting.length !== live.length) continue; // not reachable: every pass runs each ready thread to a barrier
    const exited = G - live.length;
    const at = waiting[0].at!;
    if (rule === 'gpu' && exited > 0) {
      deadlocked = true;
      const gone = st.threads.filter((t) => t.status === 'exited').map((t) => t.rank);
      push({
        rank: -1, op: 'deadlock', level: waiting[0].currLevel, lines: L(at === 1 ? 's1' : at === 2 ? 's2' : 's3'),
        text: tr(
          `${waiting.length === 1 ? `Thread ${waiting[0].rank} waits` : `Threads ${list(waiting.map((t) => t.rank))} wait`} at grid.sync() ${at} for ${gone.length === 1 ? `thread ${gone[0]}, which has` : `threads ${list(gone)}, which have`} left the kernel. sync() lets no thread past until every thread of the grid arrives, so it never opens.`,
          `${waiting.length === 1 ? `Luồng ${waiting[0].rank} chờ` : `Các luồng ${list(waiting.map((t) => t.rank))} chờ`} tại grid.sync() ${at}, đợi ${gone.length === 1 ? `luồng ${gone[0]}, vốn đã` : `các luồng ${list(gone)}, vốn đã`} rời kernel. sync() không cho luồng nào đi qua cho đến khi mọi luồng của lưới tới nơi, nên nó không bao giờ mở.`,
        ),
        violation: { kind: 'deadlock', message: tr('a deadlock: the kernel never finishes', 'một deadlock: kernel không bao giờ kết thúc') },
      });
      break;
    }
    for (const t of waiting) {
      t.status = 'ready';
      t.at = undefined;
    }
    const why =
      shape.kernel === 'ex3'
        ? tr(
            `every append to numFrontier[${waiting[0].currLevel}] is done, so its value is the level's size`,
            `mọi lần nối thêm vào numFrontier[${waiting[0].currLevel}] đã xong, nên giá trị của nó là kích thước của mức`,
          )
        : at === 1 ? tr('every append of this level is done, so *numCurrFrontier is the size of the new frontier', 'mọi lần nối thêm của mức này đã xong, nên *numCurrFrontier là kích thước của frontier mới')
          : at === 2 ? tr('every thread has read the size, so it is safe to reset it', 'mọi luồng đã đọc kích thước, nên có thể an toàn đặt lại nó')
            : tr('the reset is visible to everyone, so the next level\'s appends start from 0', 'mọi luồng đều thấy việc đặt lại, nên các lần nối thêm của mức tiếp theo bắt đầu từ 0');
    push({
      rank: -1, op: 'release', level: waiting[0].currLevel, barrier: at, lines: L(at === 1 ? 's1' : at === 2 ? 's2' : 's3'),
      text: tr(
        `grid.sync() ${at} opens${exited ? ` for the ${plural(live.length, 'thread')} still in the kernel (the emulator does not wait for threads that have left)` : ''}: ${why}.`,
        `grid.sync() ${at} mở${exited ? ` cho ${live.length} luồng còn trong kernel (trình giả lập không chờ các luồng đã rời đi)` : ''}: ${why}.`,
      ),
    });
  }

  // Judge the reads made before every thread had appended against the true size of the level.
  for (const s of steps) {
    if (s.op !== 'read' || !s.note?.startsWith('unfinished:')) continue;
    const pending = s.note.slice('unfinished:'.length).split(',').map(Number);
    const full = refSize[s.level] ?? 0;
    const them = pending.length === 1 ? tr(`thread ${pending[0]} has`, `luồng ${pending[0]}`) : tr(`threads ${list(pending)} have`, `các luồng ${list(pending)}`);
    if (s.value !== full && !s.violation)
      s.violation = {
        kind: 'partial-count',
        message: tr(
          `${them} not finished appending, so ${s.value} is not the size of level ${s.level}, which is ${full}`,
          `${them} chưa nối thêm xong, nên ${s.value} không phải kích thước của mức ${s.level}, vốn là ${full}`,
        ),
      };
    s.note = s.value === full
      ? tr(
          `${them[0].toUpperCase() + them.slice(1)} not finished appending, but ${pending.length === 1 ? 'it adds' : 'they add'} nothing to this level, so the number happens to be right.`,
          `${them[0].toUpperCase() + them.slice(1)} chưa nối thêm xong, nhưng không thêm gì vào mức này, nên con số tình cờ lại đúng.`,
        )
      : undefined;
  }

  const wrongVertices = st.level.filter((v, i) => v !== reference[i]).length;
  const levelsCorrect = !deadlocked && !crashed && wrongVertices === 0;
  const first = steps.find((s) => s.violation)?.violation?.kind;
  const exitedEarly = steps.some((s) => s.violation?.kind === 'early-exit');
  const kind: Outcome['kind'] = crashed ? 'crash'
    : first === 'partial-count' ? 'partial-count'
      : first === 'late-reset' ? 'lost-appends'
        : first === 'read-after-reset' || first === 'early-exit' || first === 'deadlock' ? 'early-exit'
          : wrongVertices ? 'wrong' : 'correct';
  const label =
    kind === 'crash' ? 'an out-of-range access' // i18n-ignore: identifier also shown to the reader (data-outcome compared by e2e/ch18-s186.mjs)
      : kind === 'early-exit' ? (deadlocked ? 'early exit, deadlock' : 'early exit') // i18n-ignore: identifier also shown to the reader (data-outcome compared by e2e/ch18-s186.mjs)
        : kind === 'lost-appends' ? 'appends lost to a late reset' // i18n-ignore: identifier also shown to the reader (data-outcome compared by e2e/ch18-s186.mjs)
          : kind === 'partial-count' ? 'a thread read a partial count' // i18n-ignore: identifier also shown to the reader (data-outcome compared by e2e/ch18-s186.mjs)
            : kind === 'wrong' ? 'levels wrong' : 'correct'; // i18n-ignore: identifier also shown to the reader (data-outcome compared by e2e/ch18-s186.mjs)
  const consequence = kind !== 'early-exit' && exitedEarly
    ? (deadlocked ? tr('then a thread leaves early: early exit, deadlock', 'rồi một luồng rời đi sớm: thoát sớm, deadlock') : tr('then a thread leaves early', 'rồi một luồng rời đi sớm'))
    : undefined;
  return {
    shape, schedule, rule, graph, blocks, threads, initial, steps, history, reference,
    outcome: { kind, label, consequence, deadlock: deadlocked, levelsCorrect, wrongVertices, crash: crashed },
    firstFailure: steps.findIndex((s) => !!s.violation),
  };
}

/** The emulator's run of the same kernel: whether the levels it leaves are right. */
export function emuResult(shape: Shape, schedule: Schedule, graph: Graph, blocks: number, threads: number) {
  const t = simulate(shape, schedule, graph, blocks, threads, 'emu');
  return { levelsCorrect: t.outcome.levelsCorrect, wrongVertices: t.outcome.wrongVertices, crash: t.outcome.crash };
}

/** The final memory of a run, in the order the check driver prints it. */
export function finalMemory(t: Trace): { levels: number[]; A: number[]; B: number[]; counters: number[] } {
  const s = t.steps.length ? t.steps[t.steps.length - 1].after : t.initial;
  return { levels: s.level, A: s.A, B: s.B, counters: s.counters };
}

// ---------------------------------------------------------------- 5. timelines (counts, not seconds)
export type HostEvent = { kind: 'launch'; level: number; frontier: number } | { kind: 'readback'; level: number; next: number };
/**
 * The per-level host loop: launch the kernel for level k + 1 from a frontier of sizes[k],
 * copy the new size back, decide. One launch and one read-back per non-empty frontier.
 */
export function hostLoop(sizes: number[]): HostEvent[] {
  const out: HostEvent[] = [];
  for (let k = 0; k < sizes.length; k++) {
    out.push({ kind: 'launch', level: k + 1, frontier: sizes[k] });
    out.push({ kind: 'readback', level: k + 1, next: sizes[k + 1] ?? 0 });
  }
  return out;
}
/**
 * For the scene: the host loop as a strip in units of vertex work. Each launch's bar is its
 * frontier's size; each round trip after it is drawn as the work of `gap` vertices. The gap
 * is an assumption for the picture, the same in every strip, not a measurement: the lab's
 * floor row measures it on the reader's device.
 */
export function hostStrip(sizes: number[], gap: number) {
  const parts: { kind: 'kernel' | 'gap'; level: number; from: number; len: number }[] = [];
  let x = 0;
  sizes.forEach((s, k) => {
    parts.push({ kind: 'kernel', level: k + 1, from: x, len: s });
    x += s;
    parts.push({ kind: 'gap', level: k + 1, from: x, len: gap });
    x += gap;
  });
  return { parts, total: x, gapShare: x ? (sizes.length * gap) / x : 0 };
}
/** The same search in one cooperative launch: one launch, then per level its work and its grid barriers. */
export function coopLoop(sizes: number[], barriersPerLevel = 3) {
  return { launches: 1, readbacks: 0, gridBarriers: sizes.length * barriersPerLevel, levels: sizes.length };
}

/** The swap's event order for one level, for the scene: what each barrier separates. */
export const SWAP_ORDER = [
  {
    get what() {
      return tr('every thread appends its new vertices', 'mỗi luồng nối thêm các đỉnh mới của nó');
    },
    guard: null,
  },
  {
    what: 'grid.sync() 1', // i18n-ignore: code
    get guard() {
      return tr('no one reads the size before every append is in', 'không ai đọc kích thước trước khi mọi lần nối thêm xong');
    },
  },
  {
    get what() {
      return tr('every thread reads *numCurrFrontier', 'mỗi luồng đọc *numCurrFrontier');
    },
    guard: null,
  },
  {
    what: 'grid.sync() 2', // i18n-ignore: code
    get guard() {
      return tr('no one reads after the reset', 'không ai đọc sau khi đặt lại');
    },
  },
  {
    get what() {
      return tr('grid thread 0 sets it to 0', 'luồng 0 của lưới đặt nó về 0');
    },
    guard: null,
  },
  {
    what: 'grid.sync() 3', // i18n-ignore: code
    get guard() {
      return tr('no append starts before the reset', 'không lần nối thêm nào bắt đầu trước khi đặt lại');
    },
  },
  {
    get what() {
      return tr('every thread swaps its two pointers', 'mỗi luồng hoán đổi hai con trỏ của nó');
    },
    guard: null,
  },
] as const;
