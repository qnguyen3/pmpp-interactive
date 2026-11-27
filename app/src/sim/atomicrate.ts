// Throughput of atomic operations, for section 9.3 and its narrated animation.
//
// The book's argument (pp. 209-210, Fig. 9.7): atomics on one location cannot overlap,
// and each lasts about a load latency plus a store latency, so one location takes at
// most clock / (load + store) atomics per second. Different locations proceed in
// parallel, so a whole histogram is limited by its busiest bin.
//
// What it simplifies: one atomic per location in flight; different locations fully
// parallel; the transfer delay between load and store ignored, as in the book's own
// arithmetic; no caches, no compiler aggregation of same-address atomics. The latencies
// for the last-level cache and for shared memory are illustrative: the book says only
// "tens of cycles" and "a few cycles", and no vendor documents a figure
// (notes/today/ch09-facts.md A4, A5, B2, C4).

// ------------------------------------------------------------------ the book's example
export interface MemorySystem {
  /** Width of one channel's interface, in bytes per transfer. */
  bytesPerTransfer: number;
  /** Double data rate: two transfers per clock. */
  transfersPerClock: number;
  channels: number;
  clockHz: number;
  /** Typical access latency, in clock cycles. */
  latencyCycles: number;
  /** Size of one data element, in bytes. */
  elementBytes: number;
}

/** The book's hypothetical memory system (pp. 209-210). Not a product. */
export const BOOK_MEMORY: MemorySystem = {
  bytesPerTransfer: 8,
  transfersPerClock: 2,
  channels: 8,
  clockHz: 1e9,
  latencyCycles: 200,
  elementBytes: 4,
};

/** Bytes per second the memory system can deliver: 128 GB/s for the book's example. */
export const peakBandwidth = (m: MemorySystem) => m.bytesPerTransfer * m.transfersPerClock * m.clockHz * m.channels;
/** Elements per second: 32 G four-byte elements for the book's example. */
export const elementRate = (m: MemorySystem) => peakBandwidth(m) / m.elementBytes;

/** Cycles one atomic occupies its location: a load latency plus a store latency (Fig. 9.7). */
export const atomicCycles = (loadCycles: number, storeCycles = loadCycles) => loadCycles + storeCycles;

/** Atomics per second on one location: 1 GHz / 400 cycles = 2.5 M for the book's example. */
export const perLocationRate = (clockHz: number, loadCycles: number, storeCycles = loadCycles) =>
  clockHz / atomicCycles(loadCycles, storeCycles);

export interface Bound {
  /** Atomics per second for the whole histogram. */
  rate: number;
  /** True when the memory system's element rate, not the busiest bin, is the limit. */
  capped: boolean;
}

/**
 * Atomics per second for a whole histogram. The busiest bin receives `hottestShare` of
 * all updates and takes them one at a time; every other bin finishes earlier. So the
 * whole input goes through at perLocation / hottestShare, but never faster than the
 * memory system can deliver the input (`cap`).
 */
export function histogramBound(perLocation: number, hottestShare: number, cap = Infinity): Bound {
  if (!(hottestShare > 0 && hottestShare <= 1)) throw new Error(`hottest-bin share ${hottestShare} is not in (0, 1]`);
  const raw = perLocation / hottestShare;
  return raw > cap ? { rate: cap, capped: true } : { rate: raw, capped: false };
}

/** Uniform data over k bins puts 1/k of the updates in each. */
export const uniformShare = (k: number) => 1 / k;

/** How many times slower an all-one-value input is than a uniform one over `bins` bins. */
export const skewSlowdown = (perLocation: number, bins: number, cap = Infinity) =>
  histogramBound(perLocation, uniformShare(bins), cap).rate / histogramBound(perLocation, 1, cap).rate;

// ------------------------------------------------------------------ exercises (p. 220)
/** Operations per second when each one takes `ns` nanoseconds. */
export const perSecond = (ns: number) => 1e9 / ns;
/** Average time of an atomic with a cache: hitRate * hit + (1 - hitRate) * miss. */
export const meanTime = (hitRate: number, hitNs: number, missNs: number) => hitRate * hitNs + (1 - hitRate) * missNs;

/** Exercise 1: 100 ns per atomic. */
export const ex1 = () => perSecond(100);
/** Exercise 2: 4 ns on an L2 hit, 100 ns in DRAM, 90% hits. */
export const ex2 = () => ({ meanNs: meanTime(0.9, 4, 100), rate: perSecond(meanTime(0.9, 4, 100)) });
/** Exercise 3: 5 floating-point operations per atomic, at exercise 1's rate. */
export const ex3 = () => 5 * ex1();
/**
 * Exercise 4: the variable privatised in shared memory; one shared-memory atomic takes
 * `nsPerAtomic` (1 ns in our wording; 2 ns if the book's "access latency is 1 ns" is
 * counted twice, read plus write); the global commits add 10% to the run time.
 */
export function ex4(nsPerAtomic = 1, overhead = 0.1, flopPerAtomic = 5) {
  const ns = nsPerAtomic * (1 + overhead);
  return { nsPerAtomic: ns, rate: perSecond(ns), flops: flopPerAtomic * perSecond(ns) };
}

// ------------------------------------------------------------------ timelines
export type ReqKind = 'load' | 'atomic' | 'plain';

export interface Req {
  /** Row label for drawing, e.g. "thread 3". */
  row: string;
  address: string;
  kind: ReqKind;
  /** Cycle at which the request is issued; default: its position times the issue interval. */
  issueAt?: number;
  /** Latency overrides for this request (cycles). */
  load?: number;
  store?: number;
  /** Amount an atomic adds (default 1). */
  amount?: number;
}

export interface Latencies {
  load: number;
  store: number;
  /** Cycles between two successive issues. */
  issue: number;
}

export interface Bar {
  row: string;
  address: string;
  kind: ReqKind;
  issue: number;
  /** When the read starts. */
  start: number;
  /** When the read completes. */
  readEnd: number;
  /** When the write starts (atomic: at readEnd; plain: after every lane has read); null for a load. */
  storeStart: number | null;
  end: number;
  /** Cycles spent waiting for the location to be free (atomics). */
  waited: number;
  read: number;
  wrote: number | null;
}

export interface Timeline {
  bars: Bar[];
  /** When the last request completes. */
  end: number;
  /** Final value at every address touched by an atomic or plain update. */
  memory: Record<string, number>;
  /** Increments requested by atomic and plain updates. */
  increments: number;
  /** Increments missing from the final values. */
  lost: number;
}

/**
 * Start and end times for a list of requests:
 * - load: issued, completes one load latency later; loads overlap freely.
 * - atomic: starts when issued AND its address is free; holds the address for a load
 *   plus a store latency; atomics on different addresses run side by side.
 * - plain: a read-modify-write without an atomic, executed by lanes in lockstep: every
 *   plain read is issued first, then, once all reads are back, every write. Each lane
 *   writes what it read plus one, so the last write wins and the others are lost.
 */
export function timeline(reqs: readonly Req[], lat: Latencies): Timeline {
  const mem: Record<string, number> = {};
  const free: Record<string, number> = {};
  const bars: Bar[] = [];
  let increments = 0;
  const plain: number[] = [];
  reqs.forEach((r, i) => {
    const issue = r.issueAt ?? i * lat.issue;
    const load = r.load ?? lat.load;
    const store = r.store ?? lat.store;
    const now = mem[r.address] ?? 0;
    if (r.kind === 'load') {
      bars.push({ row: r.row, address: r.address, kind: 'load', issue, start: issue, readEnd: issue + load, storeStart: null, end: issue + load, waited: 0, read: now, wrote: null });
    } else if (r.kind === 'atomic') {
      const amount = r.amount ?? 1;
      const start = Math.max(issue, free[r.address] ?? 0);
      const end = start + load + store;
      free[r.address] = end;
      mem[r.address] = now + amount;
      increments += amount;
      bars.push({ row: r.row, address: r.address, kind: 'atomic', issue, start, readEnd: start + load, storeStart: start + load, end, waited: start - issue, read: now, wrote: now + amount });
    } else {
      // Reads now (they all see the value from before the group); writes below.
      plain.push(bars.length);
      increments += 1;
      bars.push({ row: r.row, address: r.address, kind: 'plain', issue, start: issue, readEnd: issue + load, storeStart: null, end: issue + load, waited: 0, read: now, wrote: now + 1 });
    }
  });
  if (plain.length) {
    const allRead = Math.max(...plain.map((k) => bars[k].readEnd));
    plain.forEach((k, j) => {
      const b = bars[k];
      const store = reqs[k].store ?? lat.store;
      b.storeStart = allRead + j * lat.issue;
      b.end = b.storeStart + store;
      mem[b.address] = b.wrote!; // in write order: the last write wins
    });
  }
  const end = bars.reduce((m, b) => Math.max(m, b.end), 0);
  const kept = Object.values(mem).reduce((s, v) => s + v, 0);
  return { bars, end, memory: mem, increments, lost: increments - kept };
}

// ------------------------------------------------------------------ the scene's cases
/**
 * Cycle figures for the animation. DRAM is the book's 200-cycle example. The cache,
 * shared-memory and issue figures are ours and illustrative: the book says "tens of
 * cycles" and "a few cycles"; no vendor documents a number.
 */
export const SCENE = { dram: 200, cache: 20, shared: 4, issue: 10 } as const;
export const LANES = 8;
const dramLat: Latencies = { load: SCENE.dram, store: SCENE.dram, issue: SCENE.issue };
const rowName = (t: number) => `thread ${t}`; // i18n-ignore: identifier also shown to the reader

/** Eight threads read eight different addresses. */
export const sceneLoads = (n = LANES) =>
  timeline(Array.from({ length: n }, (_, t) => ({ row: rowName(t), address: `a${t}`, kind: 'load' as const })), dramLat);

/** Eight threads atomically add one to the same bin. */
export const sceneOneBin = (n = LANES, perAccess: number = SCENE.dram) =>
  timeline(Array.from({ length: n }, (_, t) => ({ row: rowName(t), address: 'bin', kind: 'atomic' as const })), { load: perAccess, store: perAccess, issue: SCENE.issue });

/** A warp of eight lanes does a plain read, add and write on one bin, in step. */
export const sceneLockstep = (n = LANES) =>
  timeline(Array.from({ length: n }, (_, t) => ({ row: rowName(t), address: 'bin', kind: 'plain' as const })), dramLat);

/** Bin hit by each of the eight threads: uniform over four bins, and skewed. */
export const UNIFORM4 = [0, 1, 2, 3, 0, 1, 2, 3];
export const SKEWED4 = [0, 0, 0, 0, 0, 1, 2, 3];

/** Each thread atomically adds one to the bin it is given; chains form per bin. */
export const sceneBins = (binOfThread: readonly number[], perAccess: number = SCENE.dram) =>
  timeline(
    binOfThread.map((b, t) => ({ row: rowName(t), address: `bin ${b}`, kind: 'atomic' as const })), // i18n-ignore: address key, compared with ===
    { load: perAccess, store: perAccess, issue: SCENE.issue },
  );

/** Length of the longest chain: the number of updates to the busiest bin. */
export const longestChain = (binOfThread: readonly number[]) => {
  const c = new Map<number, number>();
  for (const b of binOfThread) c.set(b, (c.get(b) ?? 0) + 1);
  return Math.max(...c.values());
};

/**
 * Privatisation preview: `blocks` blocks of `perBlock` threads, every thread updating
 * the same bin. Each block first adds into its own on-chip copy (shared-memory
 * latency), then, after its barrier, one thread commits the block's count to the
 * global bin with one atomic (DRAM latency). The commits of different blocks queue.
 */
export function scenePrivatised(blocks = 2, perBlock = LANES / 2, shared: number = SCENE.shared, global: number = SCENE.dram): Timeline {
  const lat: Latencies = { load: shared, store: shared, issue: SCENE.issue };
  const priv: Req[] = [];
  for (let b = 0; b < blocks; b++)
    for (let j = 0; j < perBlock; j++)
      priv.push({ row: rowName(b * perBlock + j), address: `private ${b}`, kind: 'atomic', issueAt: j * SCENE.issue }); // i18n-ignore: address key, compared with ===
  const first = timeline(priv, lat);
  const blockEnd = (b: number) => Math.max(...first.bars.filter((x) => x.address === `private ${b}`).map((x) => x.end));
  const commits: Req[] = Array.from({ length: blocks }, (_, b) => ({
    row: `block ${b} commit`, address: 'bin', kind: 'atomic', issueAt: blockEnd(b), load: global, store: global, amount: perBlock, // i18n-ignore: identifier also shown to the reader
  }));
  return timeline([...priv, ...commits], lat);
}
