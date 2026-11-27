// Cutoff binning with fixed-capacity bins and an overflow list (chapter 22, section 22.1;
// chapter 21, section 21.5). It runs the explorer "Bin capacity and the overflow list" and
// is meant to be reused by chapter 21.
//
// From the book:
//  - Atoms are sorted into bins by coordinate; each grid point has a neighbourhood of bins,
//    and the threads of a block scan the atoms of their combined neighbourhood together,
//    each testing every atom against its own cutoff (p. 531; ch. 21 pp. 524-526).
//  - Bins hold different numbers of atoms. Sizing every bin for the largest count is
//    prohibitive, so the capacity is fixed well below the worst case and an atom whose bin
//    is full goes on an overflow list, processed after the kernel (p. 531).
//  - All bins have the same size and are padded with dummy atoms of charge 0, which cost
//    memory and lengthen the scans of blocks whose bins are not full (ch. 21 p. 527).
//  - The host processes the overflow list with a sequential cutoff pass; with a small share
//    of overflow atoms ("e.g. under 3%") that pass typically takes less time than the device
//    (ch. 21 p. 527).
//
// Ours, not the book's (constants stated on screen):
//  - 2-D, a 16 x 16 grid of bins, one block per bin with 8 x 8 grid points (the book's
//    worked block is 8 x 8 x 8 in 3-D, ch. 21 p. 525), and a fixed 3 x 3 neighbourhood of
//    bins, clipped at the edge of the domain.
//  - A seeded generator for three atom distributions: uniform, clustered (8 clusters plus a
//    uniform background) and one dense pocket (4% of the atoms in one small region).
//  - The work model. One unit is one test of one atom slot for one grid point. The device
//    tests, for every grid point, every slot of every bin in its neighbourhood, dummies
//    included: 64 x c units per neighbour pair of bins. The host tests each overflow atom
//    against every grid point of its home bin's neighbourhood, one at a time. The device is
//    S times faster than the host at this test (a free parameter, not a device figure).
//    Serial time = device units / S + host units; overlapped (the book's sub-volume
//    refinement, ch. 21 p. 527) = the larger of the two.
//
// What it leaves out: the third dimension, the cutoff test's result (every scanned atom
// costs the same whether or not it is inside the cutoff), divergence, memory traffic and
// coalescing, the binning pass itself, and the copies between host and device.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

export const GRID = 16;
export const BINS = GRID * GRID;
/** Grid points per bin: one block of 8 x 8 threads per bin. */
export const POINTS_PER_BIN = 64;
export const SEED = 22;

export type Distribution = 'uniform' | 'clustered' | 'pocket';
export const DISTRIBUTIONS: Distribution[] = ['uniform', 'clustered', 'pocket'];
export const DIST_LABEL: Record<Distribution, string> = {
  get uniform() {
    return tr('uniform', 'đều');
  },
  get clustered() {
    return tr('clustered', 'theo cụm');
  },
  get pocket() {
    return tr('one dense pocket', 'một túi dày đặc');
  },
};

// ---------------------------------------------------------------- the generator
/** mulberry32: a small, well-mixed 32-bit generator. Returns numbers in [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A standard normal sample (Box-Muller). */
function gauss(rand: () => number): number {
  const u = Math.max(rand(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

export interface Atom {
  x: number;
  y: number;
}

const CLUSTERS = 8;
const CLUSTER_SHARE = 0.75;
const CLUSTER_SIGMA = 1.2;
const POCKET_SHARE = 0.04;
const POCKET_SIGMA = 0.45;

/** n atoms in [0, grid)^2, the same for the same arguments. */
export function makeAtoms(dist: Distribution, n: number, grid = GRID, seed = SEED): Atom[] {
  const rand = mulberry32(seed * 7919 + (dist === 'uniform' ? 1 : dist === 'clustered' ? 2 : 3));
  const uniform = (): Atom => ({ x: rand() * grid, y: rand() * grid });
  const around = (cx: number, cy: number, sigma: number): Atom => {
    for (let tries = 0; tries < 32; tries++) {
      const x = cx + sigma * gauss(rand);
      const y = cy + sigma * gauss(rand);
      if (x >= 0 && x < grid && y >= 0 && y < grid) return { x, y };
    }
    return uniform();
  };
  const out: Atom[] = [];
  if (dist === 'uniform') {
    for (let i = 0; i < n; i++) out.push(uniform());
    return out;
  }
  if (dist === 'clustered') {
    const centres = Array.from({ length: CLUSTERS }, () => ({ x: 1 + rand() * (grid - 2), y: 1 + rand() * (grid - 2) }));
    for (let i = 0; i < n; i++) {
      if (rand() < CLUSTER_SHARE) {
        const c = centres[Math.floor(rand() * CLUSTERS)];
        out.push(around(c.x, c.y, CLUSTER_SIGMA));
      } else out.push(uniform());
    }
    return out;
  }
  const px = grid * (0.25 + 0.5 * rand());
  const py = grid * (0.25 + 0.5 * rand());
  for (let i = 0; i < n; i++) out.push(rand() < POCKET_SHARE ? around(px, py, POCKET_SIGMA) : uniform());
  return out;
}

/** Atoms per bin, row-major (bin index = by * grid + bx). */
export function binCounts(atoms: readonly Atom[], grid = GRID): number[] {
  const counts = new Array<number>(grid * grid).fill(0);
  for (const a of atoms) {
    const bx = Math.min(grid - 1, Math.max(0, Math.floor(a.x)));
    const by = Math.min(grid - 1, Math.max(0, Math.floor(a.y)));
    counts[by * grid + bx]++;
  }
  return counts;
}

/** The counts for one setting of the explorer. */
export const countsFor = (dist: Distribution, n: number, grid = GRID, seed = SEED) => binCounts(makeAtoms(dist, n, grid, seed), grid);

/** h[k] = number of bins holding exactly k atoms, for k = 0 .. the fullest bin. */
export function histogram(counts: readonly number[]): number[] {
  const h = new Array<number>(Math.max(0, ...counts) + 1).fill(0);
  for (const c of counts) h[c]++;
  return h;
}

// ---------------------------------------------------------------- fixed capacity
export interface Binning {
  bins: number;
  capacity: number;
  atoms: number;
  /** Real atoms stored in the bins. */
  real: number;
  /** Dummy atoms (charge 0) that pad the bins to the capacity. */
  dummies: number;
  /** Atoms on the overflow list. */
  overflow: number;
  overflowShare: number;
  /** Slots in all bins: bins x capacity. */
  slots: number;
  /** The fullest bin's count. */
  fullest: number;
  /** Bins that hold more atoms than the capacity. */
  overflowingBins: number;
}

export function binWith(counts: readonly number[], capacity: number): Binning {
  const c = Math.max(0, Math.floor(capacity));
  let real = 0;
  let overflow = 0;
  let overflowingBins = 0;
  let atoms = 0;
  let fullest = 0;
  for (const k of counts) {
    atoms += k;
    fullest = Math.max(fullest, k);
    real += Math.min(k, c);
    if (k > c) {
      overflow += k - c;
      overflowingBins++;
    }
  }
  const slots = counts.length * c;
  return {
    bins: counts.length,
    capacity: c,
    atoms,
    real,
    dummies: slots - real,
    overflow,
    overflowShare: atoms ? overflow / atoms : 0,
    slots,
    fullest,
    overflowingBins,
  };
}

/** The smallest capacity whose overflow share is at most `share`. */
export function capacityForShare(counts: readonly number[], share: number): number {
  const fullest = Math.max(0, ...counts);
  for (let c = 0; c <= fullest; c++) if (binWith(counts, c).overflowShare <= share) return c;
  return fullest;
}

// ---------------------------------------------------------------- work
/** Bins in the clipped 3 x 3 neighbourhood of every bin, row-major. */
export function neighbourCounts(grid = GRID): number[] {
  const out: number[] = [];
  for (let by = 0; by < grid; by++)
    for (let bx = 0; bx < grid; bx++) {
      const nx = 1 + (bx > 0 ? 1 : 0) + (bx < grid - 1 ? 1 : 0);
      const ny = 1 + (by > 0 ? 1 : 0) + (by < grid - 1 ? 1 : 0);
      out.push(nx * ny);
    }
  return out;
}

/** Ordered pairs (block, neighbour bin) over the whole grid: (3G - 2)^2. */
export const neighbourPairs = (grid = GRID) => (3 * grid - 2) ** 2;

export interface Work {
  /** Slot tests on the device: every grid point, every slot of every neighbour bin. */
  device: number;
  /** Atom tests on the host: every overflow atom against every grid point of its neighbourhood. */
  host: number;
  /** Device units / S, in host test times. */
  deviceTime: number;
  hostTime: number;
  /** Kernel, then the host pass (ch. 21 p. 527). */
  serial: number;
  /** Host pass hidden behind the next kernel (the book's refinement, ch. 21 p. 527). */
  overlapped: number;
  /** Device units spent on dummy slots. */
  wasted: number;
}

export function work(counts: readonly number[], capacity: number, speedup: number, grid = GRID, pointsPerBin = POINTS_PER_BIN): Work {
  if (counts.length !== grid * grid) throw new Error('counts must have grid x grid entries');
  const c = Math.max(0, Math.floor(capacity));
  const nb = neighbourCounts(grid);
  let device = 0;
  let host = 0;
  let wasted = 0;
  counts.forEach((k, b) => {
    device += pointsPerBin * nb[b] * c;
    wasted += pointsPerBin * nb[b] * Math.max(0, c - k);
    host += pointsPerBin * nb[b] * Math.max(0, k - c);
  });
  const deviceTime = device / speedup;
  return { device, host, deviceTime, hostTime: host, serial: deviceTime + host, overlapped: Math.max(deviceTime, host), wasted };
}

/** The capacity (0 .. the fullest bin) with the lowest serial time, the smallest on a tie. */
export function bestCapacity(counts: readonly number[], speedup: number, grid = GRID, pointsPerBin = POINTS_PER_BIN): number {
  const fullest = Math.max(0, ...counts);
  let best = 0;
  let bestT = Infinity;
  for (let c = 0; c <= fullest; c++) {
    const t = work(counts, c, speedup, grid, pointsPerBin).serial;
    if (t < bestT - 1e-9) {
      best = c;
      bestT = t;
    }
  }
  return best;
}

/** The book's "e.g. under 3%" (ch. 21 p. 527). */
export const BOOK_SHARE = 0.03;
