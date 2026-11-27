// Wavefronts of a recurrence on a 2D table. Chapter 16, section 16.3 (Fig. 16.2,
// pp. 376–377; the Smith-Waterman fronts of p. 384).
//
// A stencil lists the earlier cells that each cell reads, as offsets (dr, dc). "Earlier"
// means above (dr < 0), or to the left in the same row (dr = 0, dc < 0): the order a
// sequential loop nest fills the table in, so every stencil accepted here is a recurrence
// that can be computed. A cell's step is one more than the latest step among its inputs
// that lie inside the table; inputs outside it are boundary values, known before step 0.
// The cells that share a step form one wavefront: they need nothing from each other.
//
// What it leaves out: every cell costs one unit of work, there are as many threads as
// cells in a step, and memory costs nothing. It says which cells can run together and
// how many barrier-separated steps a table needs, nothing about time on a device.
//
// Fig. 16.2(b), "Financial", is drawn small in the book; its stencil here (the cell
// above, above-left and two to the left, above) is our best reading of the figure
// (notes/ch16.md, "Unclear in the PDF"). PyTorch's CTC alpha recurrence reads the same
// three cells of the previous time step (ch16-facts D2).
//
// The one import is lib/locale.ts (itself import-free), so the browser checks can still import
// this file directly in Node.

import { tr } from '../lib/locale.ts';

export interface Offset {
  dr: number;
  dc: number;
}

export type NeighbourId = 'N' | 'W' | 'NW' | 'NE' | 'NWW' | 'WW';

/** The six earlier neighbours the explorer offers. */
export const NEIGHBOURS: { id: NeighbourId; label: string; dr: number; dc: number }[] = [
  {
    id: 'N',
    get label() {
      return tr('above', 'trên');
    },
    dr: -1,
    dc: 0,
  },
  {
    id: 'W',
    get label() {
      return tr('left', 'trái');
    },
    dr: 0,
    dc: -1,
  },
  {
    id: 'NW',
    get label() {
      return tr('above-left', 'chéo trên trái');
    },
    dr: -1,
    dc: -1,
  },
  {
    id: 'NE',
    get label() {
      return tr('above-right', 'chéo trên phải');
    },
    dr: -1,
    dc: 1,
  },
  {
    id: 'NWW',
    get label() {
      return tr('above, two left', 'trên, lùi hai cột trái');
    },
    dr: -1,
    dc: -2,
  },
  {
    id: 'WW',
    get label() {
      return tr('two left', 'lùi hai cột trái');
    },
    dr: 0,
    dc: -2,
  },
];

export const isEarlier = (o: Offset) => o.dr < 0 || (o.dr === 0 && o.dc < 0);

export function stencil(ids: readonly NeighbourId[]): Offset[] {
  return NEIGHBOURS.filter((n) => ids.includes(n.id)).map(({ dr, dc }) => ({ dr, dc }));
}

export type PresetId = 'smith-waterman' | 'checkerboard' | 'financial' | 'h264' | 'fibonacci';

export interface Preset {
  id: PresetId;
  label: string;
  /** Where the book draws it. */
  figure: string;
  neighbours: NeighbourId[];
  rows: number;
  cols: number;
}

export const PRESETS: Preset[] = [
  {
    id: 'smith-waterman',
    get label() {
      return tr('Smith-Waterman', 'Smith-Waterman');
    },
    get figure() {
      return tr('Fig. 16.2(c)', 'Hình 16.2(c)');
    },
    neighbours: ['N', 'W', 'NW'],
    rows: 8,
    cols: 8,
  },
  {
    id: 'checkerboard',
    get label() {
      return tr('Checkerboard', 'Bàn cờ');
    },
    get figure() {
      return tr('Fig. 16.2(a)', 'Hình 16.2(a)');
    },
    neighbours: ['NW', 'N', 'NE'],
    rows: 8,
    cols: 8,
  },
  {
    id: 'financial',
    get label() {
      return tr('Financial; also CTC', 'Tài chính; cả CTC');
    },
    get figure() {
      return tr('Fig. 16.2(b), our reading', 'Hình 16.2(b), theo cách đọc của chúng tôi');
    },
    neighbours: ['N', 'NW', 'NWW'],
    rows: 8,
    cols: 8,
  },
  {
    id: 'h264',
    get label() {
      return tr('H.264', 'H.264');
    },
    get figure() {
      return tr('Fig. 16.2(d)', 'Hình 16.2(d)');
    },
    neighbours: ['W', 'NE'],
    rows: 8,
    cols: 8,
  },
  {
    id: 'fibonacci',
    get label() {
      return tr('Fibonacci', 'Fibonacci');
    },
    get figure() {
      return tr('Eq. 16.1, p. 374', 'Phương trình 16.1, p. 374');
    },
    neighbours: ['W', 'WW'],
    rows: 1,
    cols: 12,
  },
];

export interface Fronts {
  rows: number;
  cols: number;
  offsets: Offset[];
  /** step[r * cols + c]: the step at which cell (r, c) can run, from 0. */
  step: number[];
  /** sizes[s]: how many cells run at step s. */
  sizes: number[];
  /** Number of barrier-separated steps. */
  steps: number;
  /** The largest number of cells in one step. */
  widest: number;
  /** Cells per step on average. */
  mean: number;
  /** How many fronts back the farthest input lies (0 when no cell reads another). */
  reach: number;
  /** Steps between the first cells of rows 0 and 1, or null for a one-row table. */
  rowLag: number | null;
}

/** Numbers every cell of a rows x cols table by the step at which it can run. */
export function wavefronts(rows: number, cols: number, offsets: readonly Offset[]): Fronts {
  if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows < 1 || cols < 1)
    throw new Error(`table must be at least 1 x 1, got ${rows} x ${cols}`);
  for (const o of offsets)
    if (!isEarlier(o)) throw new Error(`offset (${o.dr}, ${o.dc}) is not an earlier cell: a recurrence can only read cells above, or to the left in the same row`);
  const step = new Array<number>(rows * cols).fill(0);
  let reach = 0;
  // Row-major order visits every input before the cell that reads it.
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      let s = 0;
      for (const o of offsets) {
        const rr = r + o.dr;
        const cc = c + o.dc;
        if (rr >= 0 && rr < rows && cc >= 0 && cc < cols) s = Math.max(s, step[rr * cols + cc] + 1);
      }
      step[r * cols + c] = s;
    }
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      for (const { rr, cc } of inputsAt(rows, cols, offsets, r, c)) reach = Math.max(reach, step[r * cols + c] - step[rr * cols + cc]);
  // A loop, not Math.max(...step): a large table would overflow the argument list.
  const steps = step.reduce((m, s) => Math.max(m, s), 0) + 1;
  const sizes = new Array<number>(steps).fill(0);
  for (const s of step) sizes[s]++;
  return {
    rows,
    cols,
    offsets: [...offsets],
    step,
    sizes,
    steps,
    widest: sizes.reduce((m, s) => Math.max(m, s), 0),
    mean: (rows * cols) / steps,
    reach,
    rowLag: rows > 1 ? step[cols] - step[0] : null,
  };
}

function inputsAt(rows: number, cols: number, offsets: readonly Offset[], r: number, c: number) {
  const out: { rr: number; cc: number }[] = [];
  for (const o of offsets) {
    const rr = r + o.dr;
    const cc = c + o.dc;
    if (rr >= 0 && rr < rows && cc >= 0 && cc < cols) out.push({ rr, cc });
  }
  return out;
}

/** The inputs of cell (r, c) that lie inside the table, as [row, col] pairs. */
export function inputsOf(f: Fronts, r: number, c: number): [number, number][] {
  return inputsAt(f.rows, f.cols, f.offsets, r, c).map(({ rr, cc }) => [rr, cc]);
}

export const stepAt = (f: Fronts, r: number, c: number) => f.step[r * f.cols + c];

/** A preset's fronts at a given size. */
export function presetFronts(id: PresetId, rows?: number, cols?: number): Fronts {
  const p = PRESETS.find((x) => x.id === id)!;
  return wavefronts(rows ?? p.rows, cols ?? p.cols, stencil(p.neighbours));
}
