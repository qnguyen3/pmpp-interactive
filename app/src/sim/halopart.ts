// Row-split arithmetic for chapter 23's start page: the book's running example
// (§23.1, pp. 542–545), the rows its first loop exchanges (Fig. 23.7, pp. 548–549),
// the three launches of §23.3 (Fig. 23.14, pp. 557–558) and exercise 1 (p. 576).
//
// What is the book's, and what is ours (notes/ch23.md):
//   - The book's: a row-major grid with x fastest, split along y so that each GPU
//     owns a run of whole rows (pp. 543–544); a GPU's local arrays have ny rows, its
//     own plus one halo row above and one below (p. 544); the kernel of Fig. 23.1
//     computes nothing in row 0, row ny − 1, column 0 or column nx − 1 (p. 542);
//     every step a GPU sends its top and bottom boundary rows, nx floats each, to
//     its two neighbours, which wrap around at the ends (Fig. 23.7 lines 11–18,
//     pp. 549, 551–552); the three launches of Fig. 23.14 with the pointer and row
//     count each receives (pp. 557–558).
//   - Ours: which rows each launch writes (read off Fig. 23.1's guard, as the notes
//     do); how a row count that does not divide evenly is shared out (the book's
//     examples divide evenly); every number about a 2D tile split (the book states
//     only the trade-off, p. 545).
//   - Assumed: 4-byte floats, as exercise 1e's answer does.
//
// No imports and no syntax that needs compiling away beyond types, so the browser
// check can load this file directly.

/** Bytes per grid value: a 4-byte float. */
export const FLOAT_BYTES = 4;

/** Exercise 1 (p. 576): 64 points in x, 512 in y, 16 MPI ranks (one GPU each). */
export const EXERCISE1 = { nx: 64, rows: 512, gpus: 16 } as const;

const whole = (v: number, min: number, what: string) => {
  if (!Number.isInteger(v) || v < min) throw new RangeError(`${what} must be a whole number of at least ${min}, got ${v}`);
};

// ---------------------------------------------------------------- the row split

/**
 * Rows each GPU owns: an even share, rounded down, with whatever is left over on
 * the last GPU. Every GPU owns at least one row, so `rows` must be at least `gpus`.
 */
export function rowShares(rows: number, gpus: number): number[] {
  whole(gpus, 1, 'gpus');
  whole(rows, gpus, 'rows');
  const base = Math.floor(rows / gpus);
  return Array.from({ length: gpus }, (_, g) => (g === gpus - 1 ? rows - base * (gpus - 1) : base));
}

/** The first global row each GPU owns, for the shares of rowShares(). */
export function firstRows(shares: number[]): number[] {
  const out: number[] = [];
  let at = 0;
  for (const s of shares) {
    out.push(at);
    at += s;
  }
  return out;
}

/**
 * A GPU's two neighbours as Fig. 23.7 (lines 11–12) computes them: the GPU above,
 * and the GPU below, wrapping around at both ends. With one GPU both are itself.
 */
export function neighbours(gpu: number, gpus: number): { top: number; bottom: number } {
  whole(gpus, 1, 'gpus');
  if (!Number.isInteger(gpu) || gpu < 0 || gpu >= gpus) throw new RangeError(`gpu must be 0 to ${gpus - 1}, got ${gpu}`);
  return { top: gpu > 0 ? gpu - 1 : gpus - 1, bottom: (gpu + 1) % gpus };
}

/**
 * A count of grid points two ways: over all nx columns, and over the nx − 2 columns
 * the kernel of Fig. 23.1 computes. Exercise 1 does not say which it means (notes,
 * "Unclear in the PDF"), so the explorer shows both.
 */
export interface Counted {
  all: number;
  kernel: number;
}

export interface Partition {
  nx: number;
  rows: number;
  gpus: number;
  gpu: number;
  /** Rows this GPU owns. */
  owned: number;
  /** Its first and last global rows. */
  first: number;
  last: number;
  /** Height of its local arrays: the owned rows plus two halo rows (the book's ny, p. 544). */
  ny: number;
  /** The neighbours it exchanges with. */
  top: number;
  bottom: number;
  /** One GPU: the wrap-around makes it its own neighbour, so nothing crosses a link. */
  self: boolean;
  /** Points it computes: every owned row (exercise 1a). */
  output: Counted;
  /** Halo points it receives: two whole rows (exercise 1b). */
  halo: number;
  /** Rows 1 and ny − 2, computed first in §23.3's stage 1; one row when it owns one. */
  boundaryRows: number;
  boundary: Counted;
  /** Rows 2 to ny − 3, computed in stage 2 while the boundary rows travel. */
  internalRows: number;
  internal: Counted;
  /** Bytes it sends each step: its two boundary rows, nx floats each (exercise 1e). */
  bytesSent: number;
  /** Bytes to each neighbour. */
  bytesPerNeighbour: number;
  /** Rows it sends per row it computes: 2 / owned (both edge rows go out, even when they are the same row). */
  exchangeShare: number;
}

/** Everything the start page's explorer reads out for one GPU of a row split. */
export function partition(nx: number, rows: number, gpus: number, gpu: number): Partition {
  whole(nx, 1, 'nx');
  const shares = rowShares(rows, gpus);
  const { top, bottom } = neighbours(gpu, gpus);
  const owned = shares[gpu];
  const first = firstRows(shares)[gpu];
  const kx = Math.max(nx - 2, 0);
  const boundaryRows = Math.min(owned, 2);
  const internalRows = Math.max(owned - 2, 0);
  return {
    nx,
    rows,
    gpus,
    gpu,
    owned,
    first,
    last: first + owned - 1,
    ny: owned + 2,
    top,
    bottom,
    self: gpus === 1,
    output: { all: owned * nx, kernel: owned * kx },
    halo: 2 * nx,
    boundaryRows,
    boundary: { all: boundaryRows * nx, kernel: boundaryRows * kx },
    internalRows,
    internal: { all: internalRows * nx, kernel: internalRows * kx },
    bytesSent: 2 * nx * FLOAT_BYTES,
    bytesPerNeighbour: nx * FLOAT_BYTES,
    exchangeShare: 2 / owned,
  };
}

// ---------------------------------------------------------------- rows of one local array

export type RowKind = 'halo' | 'boundary' | 'internal';

/** What row r of a local array of ny rows is: halo (0, ny − 1), boundary (1, ny − 2) or internal. */
export function rowKind(ny: number, r: number): RowKind {
  whole(ny, 3, 'ny');
  if (!Number.isInteger(r) || r < 0 || r >= ny) throw new RangeError(`row must be 0 to ${ny - 1}, got ${r}`);
  if (r === 0 || r === ny - 1) return 'halo';
  if (r === 1 || r === ny - 2) return 'boundary';
  return 'internal';
}

/**
 * How the book writes the pointer to row r of `output` (Figs. 23.7 and 23.14), or
 * the plain form output + r·nx for rows it never names. Rows the book names by
 * more than one expression (a short array) take the first of: output, output + nx,
 * output + (ny - 1)*nx, output + (ny - 2)*nx, output + (ny - 3)*nx.
 */
export function pointerExpr(ny: number, r: number): string {
  rowKind(ny, r); // validates
  if (r === 0) return 'output';
  if (r === 1) return 'output + nx'; // i18n-ignore: C pointer expression
  if (r === ny - 1) return 'output + (ny - 1)*nx'; // i18n-ignore: C pointer expression
  if (r === ny - 2) return 'output + (ny - 2)*nx'; // i18n-ignore: C pointer expression
  if (r === ny - 3) return 'output + (ny - 3)*nx'; // i18n-ignore: C pointer expression
  return `output + ${r}*nx`; // i18n-ignore: C pointer expression
}

export interface RowPointer {
  row: number;
  kind: RowKind;
  /** The expression, as the book writes it where it names the row. */
  expr: string;
  /** Elements from the start of the array: r·nx. */
  offset: number;
}

/** output + k·nx for every row k of a local array, each with its class. */
export function rowPointers(ny: number, nx: number): RowPointer[] {
  whole(nx, 1, 'nx');
  return Array.from({ length: ny }, (_, r) => ({ row: r, kind: rowKind(ny, r), expr: pointerExpr(ny, r), offset: r * nx }));
}

/** One item of a compressed list of rows: a row, or a run of internal rows left out. */
export type RowItem = { kind: 'row'; row: number } | { kind: 'gap'; from: number; to: number };

/**
 * The rows worth drawing: `edge` rows at each end and one gap item for the middle.
 * Short arrays (no more than 2·edge + 1 rows) are listed whole.
 */
export function visibleRows(ny: number, edge = 3): RowItem[] {
  whole(ny, 1, 'ny');
  whole(edge, 1, 'edge');
  if (ny <= 2 * edge + 1) return Array.from({ length: ny }, (_, r) => ({ kind: 'row', row: r }));
  const head = Array.from({ length: edge }, (_, r): RowItem => ({ kind: 'row', row: r }));
  const tail = Array.from({ length: edge }, (_, i): RowItem => ({ kind: 'row', row: ny - edge + i }));
  return [...head, { kind: 'gap', from: edge, to: ny - edge - 1 }, ...tail];
}

// ---------------------------------------------------------------- the three launches of §23.3

/**
 * Rows of the local array that one launch of Fig. 23.1's kernel writes when it is
 * handed `base` rows down (output + base·nx) and told the array has `rows` rows.
 * Thread row t works on local row t + 1 and passes the guard while t + 1 < rows − 1,
 * so the launch writes local rows 1 to rows − 2, which are rows base + 1 to
 * base + rows − 2 of the whole array.
 */
export function writtenRows(base: number, rows: number): number[] {
  const out: number[] = [];
  for (let y = 1; y < rows - 1; y++) out.push(base + y);
  return out;
}

/** The array row that thread row t of a launch at `base` with `rows` rows writes, or null if it fails the guard. */
export function threadRowToDataRow(base: number, rows: number, t: number): number | null {
  const y = t + 1;
  return t >= 0 && y < rows - 1 ? base + y : null;
}

export type LaunchName = 'top' | 'bottom' | 'internal';

export interface Launch {
  name: LaunchName;
  /** The stream Fig. 23.14 inserts it into. */
  stream: string;
  /** The pointer it receives, as the book writes it, and that pointer in elements. */
  pointer: string;
  base: number;
  offset: number;
  /** The row count it is passed, as the book writes it, and its value. */
  rowsExpr: string;
  rows: number;
  /** The rows of the array it reads (base to base + rows − 1) and the rows it writes. */
  reads: number[];
  writes: number[];
}

/** The three launches of Fig. 23.14 (lines 10, 12–13, 15–16; pp. 557–558) for an array of ny rows. */
export function launches(ny: number, nx: number): Launch[] {
  whole(ny, 3, 'ny');
  whole(nx, 1, 'nx');
  const make = (name: LaunchName, stream: string, pointer: string, base: number, rowsExpr: string, rows: number): Launch => ({
    name,
    stream,
    pointer,
    base,
    offset: base * nx,
    rowsExpr,
    rows,
    reads: Array.from({ length: rows }, (_, i) => base + i),
    writes: writtenRows(base, rows),
  });
  return [
    make('top', 'topStream', 'output', 0, '3', 3),
    make('bottom', 'bottomStream', 'output + (ny - 3)*nx', ny - 3, '3', 3), // i18n-ignore: C pointer expression
    make('internal', 'internalStream', 'output + nx', 1, 'ny - 2', ny - 2), // i18n-ignore: C pointer expression
  ];
}

export interface Coverage {
  /** How many of the three launches write each row 0 to ny − 1. */
  counts: number[];
  /** Rows 1 to ny − 2 written exactly once each, and the two halo rows not at all. */
  exact: boolean;
  /** Rows written more than once, and rows 1 to ny − 2 written by none. */
  twice: number[];
  missed: number[];
}

/**
 * Whether the three launches together write rows 1 to ny − 2 exactly once. They do
 * for ny ≥ 4. With ny = 3 (one owned row) the top and bottom launches both write
 * row 1, so its residue is added to the norm twice: our reading, not the book's.
 */
export function coverage(ny: number): Coverage {
  const counts = new Array<number>(ny).fill(0);
  for (const l of launches(ny, 1)) for (const r of l.writes) counts[r]++;
  const twice = counts.flatMap((c, r) => (c > 1 ? [r] : []));
  const missed = counts.flatMap((c, r) => (r >= 1 && r <= ny - 2 && c === 0 ? [r] : []));
  const exact = twice.length === 0 && missed.length === 0 && counts[0] === 0 && counts[ny - 1] === 0;
  return { counts, exact, twice, missed };
}

// ---------------------------------------------------------------- the 2D alternative (ours)

/** Ways to arrange g GPUs as P tile rows by Q tile columns, P from g down to 1. */
export function factorisations(g: number): { p: number; q: number }[] {
  whole(g, 1, 'gpus');
  const out: { p: number; q: number }[] = [];
  for (let p = g; p >= 1; p--) if (g % p === 0) out.push({ p, q: g / p });
  return out;
}

export interface TileSplit {
  p: number;
  q: number;
  /** The largest tile: h rows by w columns, both shared out as rowShares() shares rows. */
  h: number;
  w: number;
  /** Cut sides, one neighbour each: top and bottom when P > 1, left and right when Q > 1. */
  neighbours: number;
  /** Values it sends per step: a row piece of w for each cut top or bottom, a column of h for each cut side. */
  points: number;
  bytes: number;
  /** Messages that are one contiguous run (row pieces) and messages of h single values nx apart (columns). */
  rowMessages: number;
  columnMessages: number;
}

/**
 * One GPU's exchange when the grid is cut into P × Q tiles (ours; the book stops
 * at the trade-off, p. 545). Every tile is counted as if it had a neighbour on each
 * side that is cut, as the book's wrap-around gives in y; a real grid's edge tiles
 * have fewer. With Q = 1 this is the row split: 2 neighbours and 2·nx values when P > 1.
 */
export function tileSplit(nx: number, rows: number, p: number, q: number): TileSplit {
  whole(nx, 1, 'nx');
  whole(p, 1, 'p');
  whole(q, 1, 'q');
  whole(rows, p, 'rows');
  if (nx < q) throw new RangeError(`nx must be at least q, got ${nx} < ${q}`);
  // The last share is the largest (rowShares puts the remainder there).
  const h = rowShares(rows, p)[p - 1];
  const w = rowShares(nx, q)[q - 1];
  const rowMessages = p > 1 ? 2 : 0;
  const columnMessages = q > 1 ? 2 : 0;
  const points = rowMessages * w + columnMessages * h;
  return { p, q, h, w, neighbours: rowMessages + columnMessages, points, bytes: points * FLOAT_BYTES, rowMessages, columnMessages };
}

/**
 * The arrangement that sends the least, of those that fit the grid (no tile narrower
 * than one column). Ties go to fewer neighbours, then to more tile rows (closer to
 * the book's row split).
 */
export function bestTiles(nx: number, rows: number, g: number): TileSplit {
  const all = factorisations(g)
    .filter(({ p, q }) => rows >= p && nx >= q)
    .map(({ p, q }) => tileSplit(nx, rows, p, q));
  if (all.length === 0) throw new RangeError('no arrangement fits');
  return all.reduce((best, t) =>
    t.points < best.points || (t.points === best.points && (t.neighbours < best.neighbours || (t.neighbours === best.neighbours && t.p > best.p)))
      ? t
      : best,
  );
}

/** Which tile GPU g gets: tile row g / Q, tile column g mod Q. */
export function tileOf(g: number, q: number): { row: number; col: number } {
  return { row: Math.floor(g / q), col: g % q };
}
