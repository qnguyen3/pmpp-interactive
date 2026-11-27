// Graphs and level-synchronous breadth-first search for chapter 18: the "Before you start"
// page (format lens, move the root), section 18.3's explorer ("One graph, five ways to run
// a level") and the wavefront scene.
//
// From the book (notes/ch18.md):
//  - Fig. 18.1's graph (9 vertices, 15 directed edges, p. 426) and its CSR, CSC and COO
//    arrays under the book's names: srcPtrs + dst, dstPtrs + src, src + dst (Fig. 18.3).
//  - BFS levels from a root (Fig. 18.4) and path recovery by stepping to an in-neighbour
//    one level lower (p. 430).
//  - The per-level kernels: vertex-centric push (Fig. 18.6), pull with an early break
//    (Fig. 18.8), edge-centric (Fig. 18.10), push with frontiers and compare-and-swap
//    (Figs. 18.12 and 18.14), and direction-optimised BFS, push early and pull late
//    (p. 436). One launch per level; the host stops when a launch labels nothing.
//  - The work table of pp. 438-439: push O(d·n + m), pull O(d·n + d·m), edge-centric
//    O(d·m), frontier O(n + m).
// Ours, not the book's:
//  - The Exercise 1 graph is our reading of the drawing on p. 450 (notes, "Unclear").
//  - The chain, star and grid presets, and the 15 × 15 maze, which is our own layout in
//    the spirit of Fig. 18.5, not the book's figure.
//  - The back-trace tie rule (the lowest-numbered in-neighbour one level lower).
//  - m_f and m_u, edges out of the frontier and out of unvisited vertices: the quantities
//    cuGraph's direction-optimising switch compares (notes/today/ch18-facts.md D1).
//  - The order in which the frontier kernel appends: threads in index order, each walking
//    its edges in CSR order, the first visitor of a vertex winning. That is one possible
//    order out of many; the next level's thread-to-vertex mapping follows from it.
//
// What this model simplifies:
//  - No time: a "step" is one kernel call, and nothing says how long a call takes.
//  - Every thread of a level sees the state as it was before the level. On a GPU a thread
//    may already see a neighbour labelled by another thread of the same launch and skip
//    it, so the labels written by push and edge-centric threads are given as a range:
//    at least one write per newly labelled vertex, at most one per candidate.
//  - Every launched thread is counted, whatever the hardware's limits on resident blocks.
//  - Per-warp passes count loop iterations only: a warp makes as many passes as its
//    longest-looping lane. Latency, memory traffic and caches are not modelled.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

export const UNVISITED = 4294967295; // UINT_MAX, as the course's .cu files define it

export type Edge = readonly [number, number];
export interface Pt {
  x: number;
  y: number;
}
export interface Graph {
  id: string;
  name: string;
  n: number;
  /** Directed edges [source, destination], sorted, no duplicates, no self-edges. */
  edges: Edge[];
  /** Drawing positions in the unit square (x right, y down). */
  pos: Pt[];
  /** Grid graphs: the cell of each vertex. */
  cell?: { r: number; c: number }[];
  w?: number;
  h?: number;
  /** Default root for this graph. */
  root: number;
}

const sortEdges = (edges: readonly Edge[]): Edge[] => {
  const seen = new Set<string>();
  const out: Edge[] = [];
  for (const [s, d] of edges) {
    if (s === d) continue;
    const k = s + ',' + d;
    if (seen.has(k)) continue;
    seen.add(k);
    out.push([s, d]);
  }
  return out.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
};

export function makeGraph(id: string, name: string, n: number, edges: readonly Edge[], pos: Pt[], root = 0, extra: Partial<Graph> = {}): Graph {
  return { id, name, n, edges: sortEdges(edges), pos, root, ...extra };
}

/** The same graph with edge s→d added if it was missing, removed if it was there. */
export function toggleEdge(g: Graph, s: number, d: number): Graph {
  if (s === d) return g;
  const has = g.edges.some(([a, b]) => a === s && b === d);
  const edges = has ? g.edges.filter(([a, b]) => !(a === s && b === d)) : [...g.edges, [s, d] as Edge];
  return { ...g, id: g.id, edges: sortEdges(edges) };
}
export const hasEdge = (g: Graph, s: number, d: number) => g.edges.some(([a, b]) => a === s && b === d);

// ---------------------------------------------------------------- presets
/** Fig. 18.1 (p. 426): 9 vertices, 15 edges. Positions are ours, laid out like the figure. */
export const BOOK: Graph = makeGraph(
  'book',
  "the book's graph (Fig. 18.1)", // i18n-ignore: graph name built at module level and copied by toggleEdge; shown untranslated by ch18 LevelExplorer (consumer fix)
  9,
  [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6], [2, 7], [3, 4], [3, 8], [4, 5], [4, 8], [5, 6], [6, 8], [7, 0], [7, 6]],
  [
    { x: 0.04, y: 0.55 }, { x: 0.27, y: 0.33 }, { x: 0.33, y: 0.62 }, { x: 0.42, y: 0.06 }, { x: 0.57, y: 0.33 },
    { x: 0.6, y: 0.6 }, { x: 0.74, y: 0.86 }, { x: 0.33, y: 0.95 }, { x: 0.94, y: 0.33 },
  ],
);

/** Exercise 1 (p. 450): 8 vertices, 15 edges. The edge directions are our reading of the drawing. */
export const EX1: Graph = makeGraph(
  'ex1',
  'the Exercise 1 graph (our reading of p. 450)', // i18n-ignore: graph name built at module level and copied by toggleEdge; shown untranslated by ch18 LevelExplorer (consumer fix)
  8,
  [[0, 5], [0, 2], [1, 0], [1, 4], [1, 7], [2, 3], [3, 0], [3, 6], [4, 3], [5, 1], [5, 7], [6, 4], [7, 2], [7, 4], [7, 6]],
  [
    { x: 0.5, y: 0.04 }, { x: 0.12, y: 0.5 }, { x: 0.86, y: 0.3 }, { x: 0.84, y: 0.72 }, { x: 0.44, y: 0.96 },
    { x: 0.16, y: 0.16 }, { x: 0.66, y: 0.98 }, { x: 0.46, y: 0.55 },
  ],
);

const both = (pairs: [number, number][]): Edge[] => pairs.flatMap(([a, b]) => [[a, b] as Edge, [b, a] as Edge]);

/** A road: n vertices in a line, each joined both ways to the next. Ours. */
export function chain(n = 10): Graph {
  const pairs: [number, number][] = [];
  for (let i = 0; i + 1 < n; i++) pairs.push([i, i + 1]);
  // Two rows that snake, so ten vertices fit a narrow drawing.
  const half = Math.ceil(n / 2);
  const pos = Array.from({ length: n }, (_, i) =>
    i < half ? { x: 0.04 + (0.92 * i) / Math.max(1, half - 1), y: 0.25 } : { x: 0.96 - (0.92 * (i - half)) / Math.max(1, half - 1), y: 0.78 },
  );
  return makeGraph('chain', `a chain of ${n} (a road)`, n, both(pairs), pos, 0); // i18n-ignore: graph name built at module level and copied by toggleEdge; shown untranslated by ch18 LevelExplorer (consumer fix)
}

/** A celebrity: hub 0 joined both ways to every leaf. Ours. */
export function star(leaves = 10): Graph {
  const pairs: [number, number][] = [];
  for (let i = 1; i <= leaves; i++) pairs.push([0, i]);
  const pos = [{ x: 0.5, y: 0.5 }];
  for (let i = 0; i < leaves; i++) {
    const a = (2 * Math.PI * i) / leaves - Math.PI / 2;
    pos.push({ x: 0.5 + 0.44 * Math.cos(a), y: 0.5 + 0.44 * Math.sin(a) });
  }
  return makeGraph('star', `a star with one hub and ${leaves} leaves`, leaves + 1, both(pairs), pos, 1); // i18n-ignore: graph name built at module level and copied by toggleEdge; shown untranslated by ch18 LevelExplorer (consumer fix)
}

/** A w × h grid, each cell joined both ways to its four neighbours. Vertex r·w + c. Ours. */
export function grid(w = 4, h = w, blocked: (r: number, c: number) => boolean = () => false, id = 'grid', name = `a ${w} × ${h} grid`): Graph { // i18n-ignore: graph name built at module level and copied by toggleEdge; shown untranslated by ch18 LevelExplorer (consumer fix)
  const vertexOf = new Map<number, number>();
  const cell: { r: number; c: number }[] = [];
  for (let r = 0; r < h; r++)
    for (let c = 0; c < w; c++)
      if (!blocked(r, c)) {
        vertexOf.set(r * w + c, cell.length);
        cell.push({ r, c });
      }
  const pairs: [number, number][] = [];
  for (const { r, c } of cell) {
    const v = vertexOf.get(r * w + c)!;
    if (c + 1 < w && vertexOf.has(r * w + c + 1)) pairs.push([v, vertexOf.get(r * w + c + 1)!]);
    if (r + 1 < h && vertexOf.has((r + 1) * w + c)) pairs.push([v, vertexOf.get((r + 1) * w + c)!]);
  }
  const pos = cell.map(({ r, c }) => ({ x: w === 1 ? 0.5 : 0.04 + (0.92 * c) / (w - 1), y: h === 1 ? 0.5 : 0.04 + (0.92 * r) / (h - 1) }));
  return makeGraph(id, name, cell.length, both(pairs), pos, 0, { cell, w, h });
}

/** The presets the explorer offers, in order. */
export const PRESETS: Graph[] = [BOOK, EX1, chain(10), star(10), grid(4)];

// ---------------------------------------------------------------- the maze (scene)
/**
 * A 15 × 15 wiring grid with two blockages, our own layout in the spirit of Fig. 18.5:
 * an L in the upper right and a J across the lower middle (a long bar, a stem hanging from
 * it and a foot at the stem's bottom, 14 + 3 + 4 = 21 blocks, as the book counts its J).
 * Blocked cells are removed from the graph (p. 430 allows either). The source terminal is
 * in the upper left, the target below the bar on the right, so a wire must go round.
 */
export const MAZE_W = 15;
export const MAZE_H = 15;
export const MAZE_BLOCKED: readonly (readonly [number, number])[] = [
  // the L: down from the top edge, then right
  [0, 10], [1, 10], [2, 10], [3, 10], [3, 11], [3, 12], [3, 13],
  // the J: the bar, the stem, the foot
  ...Array.from({ length: 14 }, (_, i) => [8, i + 1] as const),
  [9, 8], [10, 8], [11, 8],
  [11, 4], [11, 5], [11, 6], [11, 7],
];
export const MAZE_SOURCE = { r: 3, c: 4 } as const;
export const MAZE_TARGET = { r: 10, c: 11 } as const;
const blockedSet = new Set(MAZE_BLOCKED.map(([r, c]) => r * MAZE_W + c));
export const MAZE: Graph = (() => {
  const g = grid(MAZE_W, MAZE_H, (r, c) => blockedSet.has(r * MAZE_W + c), 'maze', 'a 15 × 15 wiring grid with blockages (ours)'); // i18n-ignore: name of the scene's graph, never shown
  return { ...g, root: vertexAt(g, MAZE_SOURCE.r, MAZE_SOURCE.c) };
})();
/** The vertex of a grid graph at (r, c), or -1 for a blocked or missing cell. */
export function vertexAt(g: Graph, r: number, c: number): number {
  if (!g.cell) return -1;
  return g.cell.findIndex((x) => x.r === r && x.c === c);
}
export const MAZE_ROOT = vertexAt(MAZE, MAZE_SOURCE.r, MAZE_SOURCE.c);
export const MAZE_GOAL = vertexAt(MAZE, MAZE_TARGET.r, MAZE_TARGET.c);

// ---------------------------------------------------------------- formats (Fig. 18.3)
export interface CSR {
  srcPtrs: number[];
  dst: number[];
}
export interface CSC {
  dstPtrs: number[];
  src: number[];
}
export interface COO {
  src: number[];
  dst: number[];
}
/** Outgoing edges, row by row, neighbours sorted. */
export function toCSR(g: Graph): CSR {
  const srcPtrs = new Array(g.n + 1).fill(0);
  for (const [s] of g.edges) srcPtrs[s + 1]++;
  for (let i = 0; i < g.n; i++) srcPtrs[i + 1] += srcPtrs[i];
  const dst = g.edges.map(([, d]) => d); // edges are sorted by (source, destination)
  return { srcPtrs, dst };
}
/** Incoming edges, column by column, sources sorted. */
export function toCSC(g: Graph): CSC {
  const byDst = [...g.edges].sort((a, b) => a[1] - b[1] || a[0] - b[0]);
  const dstPtrs = new Array(g.n + 1).fill(0);
  for (const [, d] of byDst) dstPtrs[d + 1]++;
  for (let i = 0; i < g.n; i++) dstPtrs[i + 1] += dstPtrs[i];
  return { dstPtrs, src: byDst.map(([s]) => s) };
}
/** One entry per edge, sorted by source then destination, as Fig. 18.3(c) lists them. */
export function toCOO(g: Graph): COO {
  return { src: g.edges.map(([s]) => s), dst: g.edges.map(([, d]) => d) };
}
export const outNeighbours = (csr: CSR, v: number) => csr.dst.slice(csr.srcPtrs[v], csr.srcPtrs[v + 1]);
export const inNeighbours = (csc: CSC, v: number) => csc.src.slice(csc.dstPtrs[v], csc.dstPtrs[v + 1]);

/** Storage in array elements once the data array is dropped (p. 428). */
export function storage(g: Graph) {
  const m = g.edges.length;
  return { dense: g.n * g.n, csr: g.n + 1 + m, csc: g.n + 1 + m, coo: 2 * m };
}

// ---------------------------------------------------------------- BFS on the CPU (§18.2)
/** BFS levels from the root; UNVISITED for vertices it cannot reach. */
export function levels(g: Graph, root: number): number[] {
  const csr = toCSR(g);
  const lv = new Array(g.n).fill(UNVISITED);
  lv[root] = 0;
  let cur = [root];
  for (let L = 1; cur.length; L++) {
    const next: number[] = [];
    for (const v of cur)
      for (const u of outNeighbours(csr, v))
        if (lv[u] === UNVISITED) {
          lv[u] = L;
          next.push(u);
        }
    cur = next;
  }
  return lv;
}
/** d: the number of levels excluding level 0 (p. 438), over the reachable vertices. */
export const depth = (lv: number[]) => Math.max(0, ...lv.filter((x) => x !== UNVISITED));

/**
 * Each vertex's parent in the BFS tree: an in-neighbour exactly one level lower. Where
 * several qualify, the fixed rule takes the lowest-numbered (ours; the book allows any).
 * -1 for the root and for unreached vertices.
 */
export function predecessors(g: Graph, root: number, lv = levels(g, root)): number[] {
  const csc = toCSC(g);
  return lv.map((L, v) => {
    if (v === root || L === UNVISITED) return -1;
    return inNeighbours(csc, v).find((u) => lv[u] === L - 1) ?? -1;
  });
}
/** How many in-neighbours one level lower each vertex has: more than one is a tie. */
export function parentChoices(g: Graph, root: number, lv = levels(g, root)): number[] {
  const csc = toCSC(g);
  return lv.map((L, v) => (v === root || L === UNVISITED ? 0 : inNeighbours(csc, v).filter((u) => lv[u] === L - 1).length));
}
/** The path root … target recovered backwards through predecessors, or null if unreachable. */
export function backtrace(g: Graph, root: number, target: number): number[] | null {
  const lv = levels(g, root);
  if (lv[target] === UNVISITED) return null;
  const pred = predecessors(g, root, lv);
  const path = [target];
  while (path[path.length - 1] !== root) path.push(pred[path[path.length - 1]]);
  return path.reverse();
}
/** The BFS tree: the edges from each vertex's predecessor to it. */
export function treeEdges(g: Graph, root: number): Edge[] {
  const pred = predecessors(g, root);
  return pred.flatMap((p, v) => (p >= 0 ? [[p, v] as Edge] : []));
}

// ---------------------------------------------------------------- one launch per level (§18.3 to §18.5)
export type Variant = 'push' | 'pull' | 'edge' | 'frontier' | 'diropt';
export type Kernel = 'push' | 'pull' | 'edge' | 'frontier';
export const VARIANT_NAME: Record<Variant, string> = {
  get push() {
    return tr('vertex-centric push (Fig. 18.6)', 'push lấy đỉnh làm trung tâm (Fig. 18.6)');
  },
  get pull() {
    return tr('vertex-centric pull (Fig. 18.8)', 'pull lấy đỉnh làm trung tâm (Fig. 18.8)');
  },
  get edge() {
    return tr('edge-centric (Fig. 18.10)', 'lấy cạnh làm trung tâm (Fig. 18.10)');
  },
  get frontier() {
    return tr('push with frontiers (Figs. 18.12, 18.14)', 'push với frontier (Figs. 18.12, 18.14)');
  },
  get diropt() {
    return tr('direction-optimised (push, then pull)', 'tối ưu theo hướng (push, rồi pull)');
  },
};
export type Role = 'surplus' | 'idle' | 'iterates' | 'labels';
export type ArrayName = 'level' | 'srcPtrs' | 'dst' | 'dstPtrs' | 'src' | 'prevFrontier';
export interface Read {
  array: ArrayName;
  index: number;
}
export interface ThreadInfo {
  /** blockIdx.x·blockDim.x + threadIdx.x */
  t: number;
  block: number;
  lane: number;
  /** The vertex, edge (COO index) or frontier slot this thread owns; null past the end. */
  item: number | null;
  /** The vertex the thread works from: its own (push, pull), its edge's source, or prevFrontier[i]. */
  vertex: number | null;
  role: Role;
  /** Loop iterations: edges walked (push, frontier), incoming edges scanned (pull); 1 for an edge-centric thread in range. */
  trips: number;
  /** Vertices this thread may label: unvisited out-neighbours (push, edge, frontier) or its own vertex (pull). */
  labels: number[];
  /** Frontier only: the vertices this thread wins in the model's order. */
  wins: number[];
  /** Array slots the thread reads, in program order. */
  reads: Read[];
}
export interface WarpInfo {
  block: number;
  first: number;
  size: number;
  passes: number;
  useful: number;
}
export interface LevelRun {
  /** Kernel call k labels level k. */
  call: number;
  kernel: Kernel;
  /** Threads needed: n, m or the previous frontier's size. */
  items: number;
  blocks: number;
  launched: number;
  surplus: number;
  threads: ThreadInfo[];
  pastCheck: number;
  warps: WarpInfo[];
  warpPasses: number;
  laneSlots: number;
  useful: number;
  /** Threads that each read one vertex's data: every in-range thread of push, pull and frontier. */
  vertexWork: number;
  /** Edges examined. */
  edgeWork: number;
  labelsMin: number;
  labelsMax: number;
  newVertices: number[];
  /** Vertices of level call − 1: for the frontier kernel, in the order the previous call appended them. */
  prevFrontier: number[];
  /** The frontier this call builds, in the model's order. */
  nextFrontier: number[];
  /** Edges out of the frontier, and out of vertices still unvisited, before this call. */
  mf: number;
  mu: number;
  before: number[];
  after: number[];
}
export interface RunOptions {
  variant: Variant;
  root: number;
  blockSize: number;
  warpWidth: number;
  /** Direction-optimised: the first call that pulls. */
  switchCall?: number;
}
export interface Totals {
  calls: number;
  launched: number;
  items: number;
  pastCheck: number;
  vertexWork: number;
  edgeWork: number;
  work: number;
  labelsMin: number;
  labelsMax: number;
  warpPasses: number;
  laneSlots: number;
  useful: number;
}
export interface Run {
  graph: Graph;
  options: RunOptions;
  calls: LevelRun[];
  /** Levels after the last call. */
  final: number[];
  /** Levels excluding level 0. */
  d: number;
  totals: Totals;
}

/** The kernel a call runs: direction-optimised pushes before switchCall and pulls from it on. */
export function kernelFor(o: RunOptions, call: number): Kernel {
  if (o.variant !== 'diropt') return o.variant;
  return call < (o.switchCall ?? 3) ? 'push' : 'pull';
}

function warpsOf(threads: ThreadInfo[], blockSize: number, warpWidth: number): WarpInfo[] {
  const width = Math.max(1, Math.min(warpWidth, blockSize));
  const out: WarpInfo[] = [];
  let first = 0;
  while (first < threads.length) {
    // A warp never spans two blocks: a block narrower than the warp width makes one partial warp.
    const block = Math.floor(first / blockSize);
    const end = Math.min((block + 1) * blockSize, first + width);
    const lanes = threads.slice(first, end);
    out.push({ block, first, size: lanes.length, passes: Math.max(0, ...lanes.map((t) => t.trips)), useful: lanes.reduce((s, t) => s + t.trips, 0) });
    first = end;
  }
  return out;
}

/** Run a whole search, one kernel call per level, and account for every thread of every call. */
export function runBFS(g: Graph, o: RunOptions): Run {
  const csr = toCSR(g);
  const csc = toCSC(g);
  const coo = toCOO(g);
  const m = g.edges.length;
  const outDeg = (v: number) => csr.srcPtrs[v + 1] - csr.srcPtrs[v];
  const lv = new Array(g.n).fill(UNVISITED);
  lv[o.root] = 0;
  let frontier = [o.root];
  const calls: LevelRun[] = [];
  for (let call = 1; call <= g.n + 1; call++) {
    const kernel = kernelFor(o, call);
    const before = lv.slice();
    const prevLevel = call - 1;
    const prevSorted = before.flatMap((L, v) => (L === prevLevel ? [v] : []));
    const prev = o.variant === 'frontier' ? frontier : prevSorted;
    const mf = prevSorted.reduce((s, v) => s + outDeg(v), 0);
    const mu = before.reduce((s, L, v) => (L === UNVISITED ? s + outDeg(v) : s), 0);
    const items = kernel === 'edge' ? m : kernel === 'frontier' ? prev.length : g.n;
    const blocks = Math.ceil(items / o.blockSize);
    const launched = blocks * o.blockSize;
    const threads: ThreadInfo[] = [];
    const claimed = new Set<number>();
    const next: number[] = [];
    for (let t = 0; t < launched; t++) {
      const base = { t, block: Math.floor(t / o.blockSize), lane: t % o.blockSize, wins: [] as number[] };
      if (t >= items) {
        threads.push({ ...base, item: null, vertex: null, role: 'surplus', trips: 0, labels: [], reads: [] });
        continue;
      }
      const reads: Read[] = [];
      let trips = 0;
      let past = false;
      let vertex = t;
      const labels: number[] = [];
      const wins: number[] = [];
      if (kernel === 'push') {
        reads.push({ array: 'level', index: t });
        past = before[t] === prevLevel;
        if (past) {
          reads.push({ array: 'srcPtrs', index: t }, { array: 'srcPtrs', index: t + 1 });
          for (let e = csr.srcPtrs[t]; e < csr.srcPtrs[t + 1]; e++) {
            const u = csr.dst[e];
            reads.push({ array: 'dst', index: e }, { array: 'level', index: u });
            trips++;
            if (before[u] === UNVISITED) labels.push(u);
          }
        }
      } else if (kernel === 'pull') {
        reads.push({ array: 'level', index: t });
        past = before[t] === UNVISITED;
        if (past) {
          reads.push({ array: 'dstPtrs', index: t }, { array: 'dstPtrs', index: t + 1 });
          for (let e = csc.dstPtrs[t]; e < csc.dstPtrs[t + 1]; e++) {
            const u = csc.src[e];
            reads.push({ array: 'src', index: e }, { array: 'level', index: u });
            trips++;
            if (before[u] === prevLevel) {
              labels.push(t);
              break; // Fig. 18.8 line 12
            }
          }
        }
      } else if (kernel === 'edge') {
        vertex = coo.src[t];
        reads.push({ array: 'src', index: t }, { array: 'level', index: vertex });
        trips = 1;
        past = before[vertex] === prevLevel;
        if (past) {
          const u = coo.dst[t];
          reads.push({ array: 'dst', index: t }, { array: 'level', index: u });
          if (before[u] === UNVISITED) labels.push(u);
        }
      } else {
        vertex = prev[t];
        past = true;
        reads.push({ array: 'prevFrontier', index: t }, { array: 'srcPtrs', index: vertex }, { array: 'srcPtrs', index: vertex + 1 });
        for (let e = csr.srcPtrs[vertex]; e < csr.srcPtrs[vertex + 1]; e++) {
          const u = csr.dst[e];
          reads.push({ array: 'dst', index: e }, { array: 'level', index: u });
          trips++;
          if (before[u] === UNVISITED) {
            labels.push(u);
            if (!claimed.has(u)) {
              claimed.add(u);
              wins.push(u);
              next.push(u);
            }
          }
        }
      }
      const role: Role = !past ? 'idle' : labels.length ? 'labels' : 'iterates';
      threads.push({ ...base, item: t, vertex, role, trips, labels, wins, reads });
    }
    const candidates = threads.reduce((s, t) => s + t.labels.length, 0);
    const fresh = [...new Set(threads.flatMap((t) => t.labels))].sort((a, b) => a - b);
    for (const v of fresh) lv[v] = call;
    const warps = warpsOf(threads, o.blockSize, o.warpWidth);
    const exact = kernel === 'pull' || kernel === 'frontier';
    const inRange = threads.filter((t) => t.role !== 'surplus');
    calls.push({
      call,
      kernel,
      items,
      blocks,
      launched,
      surplus: launched - items,
      threads,
      pastCheck: inRange.filter((t) => t.role !== 'idle').length,
      warps,
      warpPasses: warps.reduce((s, w) => s + w.passes, 0),
      laneSlots: warps.reduce((s, w) => s + w.passes * w.size, 0),
      useful: warps.reduce((s, w) => s + w.useful, 0),
      vertexWork: kernel === 'edge' ? 0 : items,
      edgeWork: kernel === 'edge' ? items : inRange.reduce((s, t) => s + t.trips, 0),
      labelsMin: fresh.length,
      labelsMax: exact ? fresh.length : candidates,
      newVertices: fresh,
      prevFrontier: prev,
      nextFrontier: o.variant === 'frontier' ? next : fresh,
      mf,
      mu,
      before,
      after: lv.slice(),
    });
    frontier = o.variant === 'frontier' ? next : fresh;
    if (!fresh.length) break;
  }
  const sum = (k: keyof LevelRun) => calls.reduce((s, c) => s + (c[k] as number), 0);
  const totals: Totals = {
    calls: calls.length,
    launched: sum('launched'),
    items: sum('items'),
    pastCheck: sum('pastCheck'),
    vertexWork: sum('vertexWork'),
    edgeWork: sum('edgeWork'),
    work: sum('vertexWork') + sum('edgeWork'),
    labelsMin: sum('labelsMin'),
    labelsMax: sum('labelsMax'),
    warpPasses: sum('warpPasses'),
    laneSlots: sum('laneSlots'),
    useful: sum('useful'),
  };
  return { graph: g, options: o, calls, final: lv, d: depth(lv), totals };
}

/** The bounds of the work table (pp. 438-439) for this graph and search: n, m, d. */
export function workBounds(g: Graph, d: number) {
  const n = g.n;
  const m = g.edges.length;
  const calls = d + 1; // the last call finds nothing
  return {
    n,
    m,
    d,
    calls,
    push: calls * n + m,
    pull: calls * (n + m),
    edge: calls * m,
    frontier: n + m,
  };
}

/** Shown for a level array entry. */
export const showLevel = (L: number) => (L === UNVISITED ? '·' : String(L));
