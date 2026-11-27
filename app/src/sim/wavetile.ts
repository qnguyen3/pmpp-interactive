// Tiled Smith-Waterman, square tiles and sheared tiles (book §16.6–16.7, pp. 384–396,
// Figs. 16.7–16.19). The index formulas are those of lab 16.1's reference solution as
// the course corrects the book's listings: integer arithmetic for tile_col and q_tile
// (ch16-facts A3), exactly the tiles of each front launched (the in-text exercise of
// p. 386: tile_row = firstRow + blockIdx.x), the sheared last tile column inclusive
// (tile_col <= T, Fig. 16.16 line 11) and q >= 1 tested in the sheared store too.
//
// Table: (L_seq + 1) x (L_seq + 1) cells, row 0 and column 0 the boundary (all zero).
// Interior cell (r, q), 1 <= r, q <= L_seq, is
//     H[r][q] = max(0, H[r-1][q-1] + s(r, q), H[r][q-1] + GAP, H[r-1][q] + GAP)
// with s = MATCH or MISMATCH (Eq. 16.2 with the kernel's signs: the penalties are added,
// so they are negative). The constants are the book's example values (p. 382).
//
// What it models: which tiles each launch runs, which cell each thread computes at each
// intra-tile step, where each of its three inputs is read from (a tile slot at its
// stored position, or global memory), the store, and, for one tile, what any order of
// threads with or without the barrier between steps reads and writes.
// What it leaves out: time. A step is one unit, every tile of a launch runs at once
// (no SM count, no occupancy), memory costs nothing, and "lockstep" is an idealised
// single warp in which every lane reads before any lane writes.

import { tr } from '../lib/locale.ts';

export const MATCH = 3;
export const MISMATCH = -3;
/** The gap penalty as the kernel adds it: both INSERTION and DELETION. */
export const GAP = -2;

export type Scheme = 'square' | 'sheared';

export interface Geometry {
  /** Length of each sequence (both equal, as in the book, p. 384). */
  Lseq: number;
  /** Tile width = threads per block. */
  W: number;
  scheme: Scheme;
  /** Sheared only: the defect of testing tile_col < T instead of tile_col <= T. */
  lastColLess?: boolean;
}

export interface Tile {
  row: number;
  col: number;
}

export const tilesPerSide = (Lseq: number, W: number) => Math.ceil(Lseq / W);
export const tableSize = (g: Geometry) => g.Lseq + 1;

/** Kernel launches (tile fronts): 2T - 1 for square tiles, 3T - 1 for sheared tiles. */
export function launches(g: Geometry): number {
  const T = tilesPerSide(g.Lseq, g.W);
  return g.scheme === 'square' ? 2 * T - 1 : 3 * T - 1;
}

/** Intra-tile steps, each ended by a barrier: 2W - 1 square, W sheared. */
export const stepsPerTile = (g: Pick<Geometry, 'W' | 'scheme'>) => (g.scheme === 'square' ? 2 * g.W - 1 : g.W);

export interface LaunchPlan {
  d: number;
  /** Tile row of block 0. */
  firstRow: number;
  /** tiles[b] is block b's tile. */
  tiles: Tile[];
}

/** The tiles of front d, in block order: tile_row = firstRow + blockIdx.x. */
export function launchPlan(g: Geometry, d: number): LaunchPlan {
  const T = tilesPerSide(g.Lseq, g.W);
  let lo: number;
  let hi: number;
  if (g.scheme === 'square') {
    // tile_col = d - tile_row must lie in [0, T - 1].
    lo = Math.max(0, d - T + 1);
    hi = Math.min(d, T - 1);
  } else {
    // tile_col = d - 2 tile_row must lie in [0, T] (or [0, T - 1] with the '<' defect).
    const lastCol = g.lastColLess ? T - 1 : T;
    lo = Math.max(0, Math.ceil((d - lastCol) / 2));
    hi = Math.min(T - 1, Math.floor(d / 2));
  }
  const tiles: Tile[] = [];
  for (let row = lo; row <= hi; row++) tiles.push({ row, col: g.scheme === 'square' ? d - row : d - 2 * row });
  return { d, firstRow: tiles.length ? lo : 0, tiles };
}

/** Number of blocks of front d: min(d, T - 1) - max(0, d - T + 1) + 1 for square tiles. */
export const blocksAt = (g: Geometry, d: number) => launchPlan(g, d).tiles.length;

export function schedule(g: Geometry): LaunchPlan[] {
  return Array.from({ length: launches(g) }, (_, d) => launchPlan(g, d));
}

// ---------------------------------------------------------------- thread to cell

export interface Cell {
  thread: number;
  rTile: number;
  /** Square: the tile column d_tile - r_tile (may be out of range). Sheared: d_tile, the stored column. */
  qTile: number;
  r: number;
  q: number;
  /** The tile test: 0 <= q_tile < W (always true for sheared tiles). */
  inTile: boolean;
  /** The table test: r < L, 1 <= q < L. */
  inTable: boolean;
  active: boolean;
}

/** The cell thread t of `tile` computes at intra-tile step dTile. */
export function cellAt(g: Geometry, tile: Tile, dTile: number, t: number): Cell {
  const { W } = g;
  const L = tableSize(g);
  const rTile = t;
  const r = W * tile.row + rTile + 1;
  let qTile: number;
  let q: number;
  if (g.scheme === 'square') {
    qTile = dTile - t;
    q = W * tile.col + qTile + 1;
  } else {
    qTile = dTile;
    q = W * tile.col + qTile - rTile + 1; // the shear, factor -1
  }
  const inTile = qTile >= 0 && qTile < W;
  const inTable = r < L && q >= 1 && q < L;
  return { thread: t, rTile, qTile, r, q, inTile, inTable, active: inTile && inTable };
}

/** Table cell held by stored slot (row, col) of a tile. */
export function slotCell(g: Pick<Geometry, 'W' | 'scheme'>, tile: Tile, row: number, col: number): { r: number; q: number } {
  const r = g.W * tile.row + row + 1;
  const q = g.scheme === 'square' ? g.W * tile.col + col + 1 : g.W * tile.col + col - row + 1;
  return { r, q };
}

/** The stored slot of `tile` that stands for table position (r, q), or null if none does (inverse of slotCell). */
export function slotAt(g: Pick<Geometry, 'W' | 'scheme'>, tile: Tile, r: number, q: number): [number, number] | null {
  const rt = r - (g.W * tile.row + 1);
  if (rt < 0 || rt >= g.W) return null;
  const qt = q - (g.W * tile.col + 1) + (g.scheme === 'square' ? 0 : rt);
  return qt >= 0 && qt < g.W ? [rt, qt] : null;
}

/**
 * The outline of the cells a tile covers, unclipped, as a polygon in cell units
 * (x = table column, y = table row; cell (r, q) spans [q, q + 1) x [r, r + 1)). Row
 * r_tile starts at the column of slot (r_tile, 0), so square tiles give a square and
 * sheared tiles a staircase parallelogram leaning left as it goes down.
 */
export function tileOutline(g: Pick<Geometry, 'W' | 'scheme'>, tile: Tile): [number, number][] {
  const y0 = g.W * tile.row + 1;
  const left = (rt: number) => slotCell(g, tile, rt, 0).q;
  const pts: [number, number][] = [];
  for (let rt = 0; rt < g.W; rt++) pts.push([left(rt) + g.W, y0 + rt], [left(rt) + g.W, y0 + rt + 1]);
  for (let rt = g.W - 1; rt >= 0; rt--) pts.push([left(rt), y0 + rt + 1], [left(rt), y0 + rt]);
  return pts;
}

/** Threads that compute a cell at step dTile. */
export function activeThreads(g: Geometry, tile: Tile, dTile: number): number[] {
  return Array.from({ length: g.W }, (_, t) => t).filter((t) => cellAt(g, tile, dTile, t).active);
}

/** Busy threads per intra-tile step in a full tile: 1, 2, ..., W, ..., 2, 1 square; W at every step sheared. */
export function busyPerStep(W: number, scheme: Scheme): number[] {
  return Array.from({ length: stepsPerTile({ W, scheme }) }, (_, k) =>
    scheme === 'square' ? Math.min(k, W - 1) - Math.max(0, k - W + 1) + 1 : W,
  );
}

/** Eqs. 16.3 and 16.4: cells of a full tile, steps, mean cells per step. */
export function tileWork(W: number, scheme: Scheme): { cells: number; steps: number; perStep: number } {
  const b = busyPerStep(W, scheme);
  const cells = b.reduce((s, x) => s + x, 0);
  return { cells, steps: b.length, perStep: cells / b.length };
}

// ---------------------------------------------------------------- inputs

export type LoadName = 'n' | 'w' | 'nw';
export const LOADS: LoadName[] = ['n', 'w', 'nw'];

export interface Load {
  name: LoadName;
  /** The table cell this input is. */
  r: number;
  q: number;
  from: 'tile' | 'global';
  /** Stored slot read, when from the tile. */
  slot: [number, number] | null;
}

/**
 * The three loads of an active thread, as the load functions write them.
 * Square (Fig. 16.10): an input on the tile's top edge (r_tile == 0) or left edge
 * (q_tile == 0) comes from global memory; otherwise from the slot one row up and/or one
 * column left. Sheared (Fig. 16.17): the tile is stored as a square, cell
 * (r_tile, q_tile) at table column base + q_tile - r_tile, so north and west are one
 * stored column left and north-west two.
 */
export function loadsOf(g: Pick<Geometry, 'scheme'>, c: Cell): Load[] {
  const { rTile: rt, qTile: qt, r, q } = c;
  const tile = (name: LoadName, rr: number, qq: number, slot: [number, number]): Load => ({ name, r: rr, q: qq, from: 'tile', slot });
  const glob = (name: LoadName, rr: number, qq: number): Load => ({ name, r: rr, q: qq, from: 'global', slot: null });
  if (g.scheme === 'square')
    return [
      rt === 0 ? glob('n', r - 1, q) : tile('n', r - 1, q, [rt - 1, qt]),
      qt === 0 ? glob('w', r, q - 1) : tile('w', r, q - 1, [rt, qt - 1]),
      rt === 0 || qt === 0 ? glob('nw', r - 1, q - 1) : tile('nw', r - 1, q - 1, [rt - 1, qt - 1]),
    ];
  return [
    rt === 0 || qt === 0 ? glob('n', r - 1, q) : tile('n', r - 1, q, [rt - 1, qt - 1]),
    qt === 0 ? glob('w', r, q - 1) : tile('w', r, q - 1, [rt, qt - 1]),
    rt === 0 || qt <= 1 ? glob('nw', r - 1, q - 1) : tile('nw', r - 1, q - 1, [rt - 1, qt - 2]),
  ];
}

export interface Producer {
  tile: Tile;
  /** The launch (tile front) that computes it. */
  d: number;
  dTile: number;
  thread: number;
  slot: [number, number];
}

/** Who computes interior cell (r, q); null for a boundary cell (row 0 or column 0). */
export function producerOf(g: Pick<Geometry, 'W' | 'scheme'>, r: number, q: number): Producer | null {
  if (r <= 0 || q <= 0) return null;
  const { W } = g;
  const row = Math.floor((r - 1) / W);
  const rTile = (r - 1) % W;
  if (g.scheme === 'square') {
    const col = Math.floor((q - 1) / W);
    const qTile = (q - 1) % W;
    return { tile: { row, col }, d: row + col, dTile: rTile + qTile, thread: rTile, slot: [rTile, qTile] };
  }
  const x = q - 1 + rTile;
  const col = Math.floor(x / W);
  const qTile = x % W;
  return { tile: { row, col }, d: col + 2 * row, dTile: qTile, thread: rTile, slot: [rTile, qTile] };
}

/** Whether a tile is launched at all (the '<' defect skips the sheared tile column T). */
export function tileRuns(g: Geometry, tile: Tile): boolean {
  const T = tilesPerSide(g.Lseq, g.W);
  if (tile.row < 0 || tile.row >= T || tile.col < 0) return false;
  if (g.scheme === 'square') return tile.col < T;
  return g.lastColLess ? tile.col < T : tile.col <= T;
}

// ---------------------------------------------------------------- store

export interface StoreOp {
  row: number;
  thread: number;
  slot: [number, number];
  r: number;
  q: number;
  /** The store's bounds test: r < L, 1 <= q < L. */
  writes: boolean;
}

/**
 * store_tile (Figs. 16.12, 16.18): in loop iteration `row`, thread t writes slot
 * (row, t) to the table, so the threads of a block write consecutive columns of one
 * table row: thread index means "column" here, "row" in the compute loop.
 */
export function storeOps(g: Geometry, tile: Tile): StoreOp[][] {
  const L = tableSize(g);
  return Array.from({ length: g.W }, (_, row) =>
    Array.from({ length: g.W }, (_, t) => {
      const { r, q } = slotCell(g, tile, row, t);
      return { row, thread: t, slot: [row, t] as [number, number], r, q, writes: r < L && q >= 1 && q < L };
    }),
  );
}

// ---------------------------------------------------------------- values

/** Deterministic sequences over A, C, G, T (a small linear congruential generator). */
export function makeSequences(Lseq: number, seed = 7): { a: string; b: string } {
  let s = seed >>> 0;
  const next = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return 'ACGT'[s >>> 30];
  };
  const a = Array.from({ length: Lseq }, next).join('');
  const b = Array.from({ length: Lseq }, next).join('');
  return { a, b };
}

/** s(r, q): the row sequence is a, the column sequence b (rea and ref in the kernel). */
export const subst = (a: string, b: string, r: number, q: number) => (a[r - 1] === b[q - 1] ? MATCH : MISMATCH);

export const max4 = (x: number, y: number, z: number, w: number) => Math.max(x, y, z, w);

/** The kernel's cell update from its three inputs. */
export const update = (a: string, b: string, r: number, q: number, n: number, w: number, nw: number) =>
  max4(0, nw + subst(a, b, r, q), w + GAP, n + GAP);

/** A direct CPU fill, row by row: the reference. Flat, L x L. */
export function cpuTable(a: string, b: string): number[] {
  const L = a.length + 1;
  if (b.length + 1 !== L) throw new Error('sequences must have equal length');
  const H = new Array<number>(L * L).fill(0);
  for (let r = 1; r < L; r++)
    for (let q = 1; q < L; q++) H[r * L + q] = update(a, b, r, q, H[(r - 1) * L + q], H[r * L + q - 1], H[(r - 1) * L + q - 1]);
  return H;
}

// ---------------------------------------------------------------- the whole table

export interface Fill {
  g: Geometry;
  L: number;
  T: number;
  /** The table after every launch; NaN where nothing was ever written. */
  H: number[];
  /** Launch that computed each cell: -1 boundary, -2 never computed. */
  launchOf: number[];
  /** How many times each cell was computed (should be 1 for every interior cell). */
  computed: number[];
  /** How many times each cell was stored to the table. */
  stored: number[];
  /** Blocks per launch. */
  perLaunch: number[];
  /** Tile computations (blocks launched over all launches), incomplete tiles included. */
  blocks: number;
  /** Interior cells no launch computes. */
  neverComputed: number;
  /** Interior cells whose value is missing: never computed, or computed from a missing input. */
  missing: number;
  /** Any departure from the dependency rules, in words (empty for a correct schedule). */
  problems: string[];
}

/**
 * Runs every launch in order, every tile of a launch in lockstep with a barrier between
 * steps, then the store. Each read is checked: from the tile only when the producer is
 * the same tile at an earlier step (or, sheared, a boundary cell whose slot was zeroed),
 * from global memory only when an earlier launch wrote it or it is a boundary cell.
 */
export function fill(g: Geometry, a: string, b: string): Fill {
  const L = tableSize(g);
  const T = tilesPerSide(g.Lseq, g.W);
  const ref = cpuTable(a, b);
  const H = new Array<number>(L * L).fill(NaN);
  const launchOf = new Array<number>(L * L).fill(-2);
  for (let i = 0; i < L; i++) {
    H[i] = 0;
    H[i * L] = 0;
    launchOf[i] = -1;
    launchOf[i * L] = -1;
  }
  const computed = new Array<number>(L * L).fill(0);
  const stored = new Array<number>(L * L).fill(0);
  const problems: string[] = [];
  const perLaunch: number[] = [];
  const S = stepsPerTile(g);
  const W = g.W;
  const nL = launches(g);
  for (let d = 0; d < nL; d++) {
    const plan = launchPlan(g, d);
    perLaunch.push(plan.tiles.length);
    for (const tile of plan.tiles) {
      // Sheared tiles are zeroed first (Fig. 16.19); a square tile starts unknown.
      const slots = new Array<number>(W * W).fill(g.scheme === 'sheared' ? 0 : NaN);
      const writtenAt = new Array<number>(W * W).fill(-1);
      for (let k = 0; k < S; k++) {
        const writes: [number, number][] = [];
        for (let t = 0; t < W; t++) {
          const c = cellAt(g, tile, k, t);
          if (!c.active) continue;
          const vals: Record<LoadName, number> = { n: NaN, w: NaN, nw: NaN };
          for (const ld of loadsOf(g, c)) {
            const p = producerOf(g, ld.r, ld.q);
            const where = `${g.scheme} L_seq=${g.Lseq} W=${W} d=${d} tile (${tile.row},${tile.col}) step ${k} thread ${t} ${ld.name} of (${c.r},${c.q})`;
            if (ld.from === 'tile') {
              const [sr, sc] = ld.slot!;
              const at = sr * W + sc;
              const same = p && p.tile.row === tile.row && p.tile.col === tile.col && p.slot[0] === sr && p.slot[1] === sc;
              if (same) {
                if (!(p.dTile < k) || writtenAt[at] < 0) problems.push(`${where}: slot read before its producer ran`); // i18n-ignore: developer diagnostic, checked by tests, not shown to the reader
              } else if (!(p === null && g.scheme === 'sheared' && writtenAt[at] < 0)) {
                problems.push(`${where}: tile slot does not hold this cell`); // i18n-ignore: developer diagnostic, checked by tests, not shown to the reader
              }
              vals[ld.name] = slots[at];
            } else {
              if (p && (p.d >= d || launchOf[ld.r * L + ld.q] === -2) && tileRuns(g, p.tile))
                problems.push(`${where}: global read of a cell not yet written`); // i18n-ignore: developer diagnostic, checked by tests, not shown to the reader
              if (p && p.tile.row === tile.row && p.tile.col === tile.col) problems.push(`${where}: own-tile cell read from global memory`); // i18n-ignore: developer diagnostic, checked by tests, not shown to the reader
              vals[ld.name] = H[ld.r * L + ld.q];
            }
          }
          const v = update(a, b, c.r, c.q, vals.n, vals.w, vals.nw);
          computed[c.r * L + c.q]++;
          writes.push([c.rTile * W + c.qTile, v]);
        }
        for (const [at, v] of writes) {
          slots[at] = v;
          writtenAt[at] = k;
        }
      }
      for (const row of storeOps(g, tile))
        for (const op of row) {
          if (!op.writes) continue;
          const i = op.r * L + op.q;
          H[i] = slots[op.slot[0] * W + op.slot[1]];
          launchOf[i] = d;
          stored[i]++;
        }
    }
  }
  let neverComputed = 0;
  let missing = 0;
  for (let r = 1; r < L; r++)
    for (let q = 1; q < L; q++) {
      const i = r * L + q;
      if (computed[i] === 0) neverComputed++;
      if (Number.isNaN(H[i])) missing++;
      else if (H[i] !== ref[i]) problems.push(`cell (${r},${q}) = ${H[i]}, CPU table ${ref[i]}`);
    }
  return { g, L, T, H, launchOf, computed, stored, perLaunch, blocks: perLaunch.reduce((s, x) => s + x, 0), neverComputed, missing, problems };
}

/** The readouts of the fronts explorer. */
export function summary(g: Geometry) {
  const T = tilesPerSide(g.Lseq, g.W);
  const plans = schedule(g);
  const perLaunch = plans.map((p) => p.tiles.length);
  const work = tileWork(g.W, g.scheme);
  return {
    T,
    launches: plans.length,
    mostTiles: Math.max(0, ...perLaunch),
    perLaunch,
    blocks: perLaunch.reduce((s, x) => s + x, 0),
    stepsPerTile: work.steps,
    /** Barrier-separated steps one after another: launches x steps per tile. */
    criticalSteps: plans.length * work.steps,
    busyMean: work.perStep,
  };
}

// ---------------------------------------------------------------- one tile, any thread order

export type Order = 'lockstep' | 'first' | 'last';
export const ORDER_LABEL: Record<Order, string> = {
  get lockstep() {
    return tr('lockstep (one idealised warp)', 'lockstep (một warp lý tưởng hóa)');
  },
  get first() {
    return tr('thread 0 first', 'luồng 0 trước');
  },
  get last() {
    return tr('last thread first', 'luồng cuối trước');
  },
};

export interface RunOptions {
  order: Order;
  /** The barrier at the end of each intra-tile step. */
  barrier: boolean;
  /** Sheared only: initialize_tile (Fig. 16.19). Square tiles are never zeroed. */
  zeroInit?: boolean;
}

export type ReadProblem = 'read-before-write' | 'uninitialised' | 'missing-global';

export interface Read extends Load {
  value: number;
  /** The CPU table's value of the cell this input stands for. */
  expected: number;
  problem: ReadProblem | null;
}

export interface Part {
  thread: number;
  kind: 'step' | 'store';
  /** Intra-tile step (kind 'step'). */
  k: number;
  cell: Cell | null;
  reads: Read[];
  /** Value written to the slot (active compute only). */
  value?: number;
  expected?: number;
  stores?: { row: number; r: number; q: number; value: number; expected: number; problem: ReadProblem | null }[];
}

export interface Tick {
  kind: 'step' | 'store';
  /** Intra-tile step, or -1 for the store. */
  k: number;
  parts: Part[];
  /** Ends with the block barrier (all threads have finished this step). */
  barrierAfter: boolean;
}

export interface SlotState {
  value: number;
  kind: 'unset' | 'zero' | 'written';
  /** Tick that wrote it, -1 if not written. */
  tick: number;
}

export interface TileRun {
  g: Geometry;
  tile: Tile;
  d: number;
  a: string;
  b: string;
  ref: number[];
  ticks: Tick[];
  /** snapshots[n]: the slots after n ticks (snapshots[0] is the start). */
  snapshots: SlotState[][];
  /** Index of the first tick with a bad read, or null. */
  firstBad: number | null;
  badReads: number;
  /** Table cells the tile stores with a wrong value. */
  wrongCells: number;
  /** Table cells the tile is responsible for. */
  cells: number;
  correct: boolean;
}

/** The table as earlier launches leave it for this tile: earlier fronts written, the rest missing. */
function globalBefore(g: Geometry, d: number, ref: number[]): number[] {
  const L = tableSize(g);
  const H = new Array<number>(L * L).fill(NaN);
  for (let r = 0; r < L; r++)
    for (let q = 0; q < L; q++) {
      const p = producerOf(g, r, q);
      if (!p || (p.d < d && tileRuns(g, p.tile))) H[r * L + q] = ref[r * L + q];
    }
  return H;
}

/**
 * One tile of a full table, with its threads interleaved in a chosen order. A thread
 * advances one whole step at a time (three reads, then its write). With the barrier,
 * every thread finishes step k before any starts step k + 1; without it, each thread
 * runs all its steps and its part of the store before the next thread starts (except in
 * lockstep, where all lanes always move together, reads before writes).
 */
export function runTile(g: Geometry, tile: Tile, opts: RunOptions, seqs = makeSequences(g.Lseq)): TileRun {
  const { a, b } = seqs;
  const ref = cpuTable(a, b);
  const L = tableSize(g);
  const W = g.W;
  const S = stepsPerTile(g);
  const d = g.scheme === 'square' ? tile.row + tile.col : tile.col + 2 * tile.row;
  const H = globalBefore(g, d, ref);
  const zero = g.scheme === 'sheared' && opts.zeroInit !== false;
  const slots: SlotState[] = Array.from({ length: W * W }, () => (zero ? { value: 0, kind: 'zero', tick: -1 } : { value: NaN, kind: 'unset', tick: -1 }));
  const snapshots: SlotState[][] = [slots.map((s) => ({ ...s }))];
  const ticks: Tick[] = [];
  const ids = Array.from({ length: W }, (_, t) => t);
  const threadOrder = opts.order === 'last' ? [...ids].reverse() : ids;

  const read = (ld: Load): Read => {
    const expected = ld.r >= 0 && ld.q >= 0 ? ref[ld.r * L + ld.q] : 0;
    if (ld.from === 'global') {
      const value = H[ld.r * L + ld.q];
      return { ...ld, value, expected, problem: Number.isNaN(value) ? 'missing-global' : null };
    }
    const s = slots[ld.slot![0] * W + ld.slot![1]];
    const holds = slotCell(g, tile, ld.slot![0], ld.slot![1]);
    const inTable = holds.r < L && holds.q >= 1 && holds.q < L;
    let problem: ReadProblem | null = null;
    if (s.kind !== 'written') problem = inTable ? 'read-before-write' : s.kind === 'unset' ? 'uninitialised' : null;
    return { ...ld, value: s.value, expected, problem };
  };

  const stepParts = (k: number, threads: number[]): Part[] =>
    threads.map((t) => {
      const cell = cellAt(g, tile, k, t);
      if (!cell.active) return { thread: t, kind: 'step', k, cell, reads: [] };
      const reads = loadsOf(g, cell).map(read);
      const v = (n: LoadName) => reads.find((x) => x.name === n)!.value;
      return {
        thread: t, kind: 'step', k, cell, reads,
        value: update(a, b, cell.r, cell.q, v('n'), v('w'), v('nw')),
        expected: ref[cell.r * L + cell.q],
      };
    });
  const writeParts = (parts: Part[], tickNo: number) => {
    for (const p of parts)
      if (p.cell?.active && p.value !== undefined) slots[p.cell.rTile * W + p.cell.qTile] = { value: p.value, kind: 'written', tick: tickNo };
  };
  const storeParts = (threads: number[]): Part[] => {
    const ops = storeOps(g, tile);
    return threads.map((t) => {
      const stores = ops
        .map((row) => row[t])
        .filter((op) => op.writes)
        .map((op) => {
          const s = slots[op.slot[0] * W + op.slot[1]];
          const value = s.value;
          H[op.r * L + op.q] = value;
          return { row: op.row, r: op.r, q: op.q, value, expected: ref[op.r * L + op.q], problem: s.kind === 'written' ? null : ('read-before-write' as ReadProblem) };
        });
      return { thread: t, kind: 'store', k: -1, cell: null, reads: [], stores };
    });
  };
  const push = (tick: Tick) => {
    ticks.push(tick);
    snapshots.push(slots.map((s) => ({ ...s })));
  };

  if (opts.order === 'lockstep') {
    for (let k = 0; k < S; k++) {
      const parts = stepParts(k, ids); // every lane reads ...
      writeParts(parts, ticks.length); // ... then every lane writes
      push({ kind: 'step', k, parts, barrierAfter: opts.barrier });
    }
    push({ kind: 'store', k: -1, parts: storeParts(ids), barrierAfter: false });
  } else if (opts.barrier) {
    for (let k = 0; k < S; k++)
      threadOrder.forEach((t, i) => {
        const parts = stepParts(k, [t]);
        writeParts(parts, ticks.length);
        push({ kind: 'step', k, parts, barrierAfter: i === W - 1 });
      });
    for (const t of threadOrder) push({ kind: 'store', k: -1, parts: storeParts([t]), barrierAfter: false });
  } else {
    for (const t of threadOrder) {
      for (let k = 0; k < S; k++) {
        const parts = stepParts(k, [t]);
        writeParts(parts, ticks.length);
        push({ kind: 'step', k, parts, barrierAfter: false });
      }
      push({ kind: 'store', k: -1, parts: storeParts([t]), barrierAfter: false });
    }
  }

  let firstBad: number | null = null;
  let badReads = 0;
  ticks.forEach((tk, n) =>
    tk.parts.forEach((p) => {
      const bad = p.reads.filter((r) => r.problem).length + (p.stores ?? []).filter((s) => s.problem).length;
      badReads += bad;
      if (bad && firstBad === null) firstBad = n;
    }),
  );
  const finalStores = ticks.flatMap((tk) => tk.parts.flatMap((p) => p.stores ?? []));
  const wrongCells = finalStores.filter((s) => !(s.value === s.expected)).length;
  return { g, tile, d, a, b, ref, ticks, snapshots, firstBad, badReads, wrongCells, cells: finalStores.length, correct: wrongCells === 0 && badReads === 0 };
}
