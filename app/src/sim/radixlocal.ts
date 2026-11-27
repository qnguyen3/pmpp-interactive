// One radix-sort pass, direct or with a local sort first, for chapter 14, sections 14.6
// to 14.8 (pp. 339-345), and the drawings of the "Sort locally, then write in runs" scene.
//
// What it models. One pass of a least-significant-digit radix sort over unsigned keys,
// by one r-bit digit, in two designs:
//   direct  Figs. 14.5 to 14.7 with an r-bit digit: thread i writes key i straight to
//           its place in the output, (keys with a smaller digit) + (earlier keys with
//           the same digit). For r = 1 that is the book's two formulas (pp. 337-338).
//   local   Figs. 14.8 to 14.13: each block sorts its own keys by the digit (as r
//           one-bit splits, p. 342), writes its bucket sizes into a table laid out
//           digit by digit (every block's count of digit 0, then of digit 1, ...: Figs.
//           14.9 and 14.12), one exclusive scan of the table gives every local bucket
//           its global start, and the block copies its buckets out. In store
//           instruction k, thread t copies local slot k*B + t, so a warp's lanes always
//           copy consecutive slots. That assignment is ours: the book does not say which
//           thread writes which slot when a thread owns several keys (notes, erratum 8).
// The memory side counts, for every warp-wide store instruction, the runs of
// consecutive addresses and the aligned memory units the lanes touch: NVIDIA's 32-byte
// sectors or 128-byte lines, AMD's 64- or 128-byte lines, or a size the reader picks for
// Metal, which documents none (notes/today/ch06-facts.md A2, A4, B1, C1; the arithmetic
// of ch14-facts A10). The rule is counted in this file's own code. The output array
// starts on a unit boundary.
//
// What it leaves out. One store instruction at a time, no cache (a unit two stores touch
// is counted twice), no timing, no DRAM. Reads are not counted: every design reads its
// keys in order. Values that travel with the keys follow the same addresses in a second
// array: they double the units and change no ratio.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

/** The sixteen 4-bit keys of Figs. 14.4 to 14.13. */
export const BOOK_KEYS: readonly number[] = [
  0b1100, 0b0011, 0b0110, 0b1001, 0b1111, 0b1000, 0b0101, 0b1010, 0b1001, 0b0110, 0b1011, 0b1101, 0b0100, 0b1010,
  0b0111, 0b0000,
];
export const BOOK_KEY_BITS = 4;

/** A key in binary, padded to `bits` digits. */
export const bin = (k: number, bits: number) => k.toString(2).padStart(bits, '0');

/** The r-bit digit of an unsigned key starting at bit `shift` (r up to 16, shift below 32). */
export function digitOf(key: number, shift: number, r: number): number {
  return (key >>> shift) & (2 ** r - 1);
}

/** Passes an LSD radix sort makes over `keyBits`-bit keys with an r-bit digit. */
export const passCount = (keyBits: number, r: number) => Math.ceil(keyBits / r);

export function exclusiveScan(xs: readonly number[]): number[] {
  const out = new Array<number>(xs.length);
  let s = 0;
  for (let i = 0; i < xs.length; i++) {
    out[i] = s;
    s += xs[i];
  }
  return out;
}

/** One whole pass done right: the stable partition by the digit, and every key's destination. */
export function stablePass(keys: readonly number[], shift: number, r: number): { out: number[]; dest: number[] } {
  const nb = 2 ** r;
  const counts = new Array<number>(nb).fill(0);
  for (const k of keys) counts[digitOf(k, shift, r)]++;
  const next = exclusiveScan(counts);
  const dest = keys.map((k) => next[digitOf(k, shift, r)]++);
  const out = new Array<number>(keys.length);
  keys.forEach((k, i) => (out[dest[i]] = k));
  return { out, dest };
}

/** The keys as pass `pass` sees them: the input after the earlier passes. */
export function passInput(keys: readonly number[], r: number, pass: number): number[] {
  let a = keys.slice();
  for (let p = 0; p < pass; p++) a = stablePass(a, p * r, r).out;
  return a;
}

/** The book's way to sort a block by an r-bit digit (p. 342): r one-bit splits, each stable. Returns the keys after each split. */
export function splitSteps(keys: readonly number[], shift: number, r: number): number[][] {
  const steps: number[][] = [];
  let a = keys.slice();
  for (let b = 0; b < r; b++) {
    // Fig. 14.6's formulas inside the block: a zero goes to (index - ones before), a one to (zeros in total + ones before).
    const bits = a.map((k) => digitOf(k, shift + b, 1));
    const onesBefore = exclusiveScan(bits);
    const ones = bits.reduce((x, y) => x + y, 0);
    const zeros = a.length - ones;
    const next = new Array<number>(a.length);
    a.forEach((k, i) => (next[bits[i] === 0 ? i - onesBefore[i] : zeros + onesBefore[i]] = k));
    a = next;
    steps.push(a);
  }
  return steps;
}

export interface LocalBlock {
  block: number;
  /** Index of the block's first key in the pass input. */
  first: number;
  /** The block's keys sorted by the digit, stably: what sits in its shared memory. */
  sorted: number[];
  /** For each local slot, the input index of the key in it. */
  from: number[];
  /** Keys per digit value (the local bucket sizes), length 2^r. */
  counts: number[];
  /** Where each local bucket starts inside the block: the exclusive scan of counts. */
  offsets: number[];
}

/** Phase one of the local design: every block sorts its own keys by the digit (Figs. 14.8, 14.11, 14.13). */
export function localSortBlocks(keys: readonly number[], shift: number, r: number, keysPerBlock: number): LocalBlock[] {
  const nb = 2 ** r;
  const blocks: LocalBlock[] = [];
  for (let first = 0, b = 0; first < keys.length; first += keysPerBlock, b++) {
    const end = Math.min(keys.length, first + keysPerBlock);
    const counts = new Array<number>(nb).fill(0);
    for (let i = first; i < end; i++) counts[digitOf(keys[i], shift, r)]++;
    const offsets = exclusiveScan(counts);
    const next = offsets.slice();
    const sorted = new Array<number>(end - first);
    const from = new Array<number>(end - first);
    for (let i = first; i < end; i++) {
      const j = next[digitOf(keys[i], shift, r)]++;
      sorted[j] = keys[i];
      from[j] = i;
    }
    blocks.push({ block: b, first, sorted, from, counts, offsets });
  }
  return blocks;
}

/** How the bucket-size table is laid out before its scan. Digit-major is the book's (Figs. 14.9, 14.12). */
export type Layout = 'digit-major' | 'block-major';

export function tableIndex(block: number, d: number, nBlocks: number, nBuckets: number, layout: Layout): number {
  return layout === 'digit-major' ? d * nBlocks + block : block * nBuckets + d;
}

/** The table of local bucket sizes, linearised; counts[b][d] is block b's count of digit d. */
export function bucketTable(counts: readonly (readonly number[])[], layout: Layout): number[] {
  const nBlocks = counts.length;
  const nBuckets = nBlocks ? counts[0].length : 0;
  const t = new Array<number>(nBlocks * nBuckets);
  counts.forEach((row, b) => row.forEach((c, d) => (t[tableIndex(b, d, nBlocks, nBuckets, layout)] = c)));
  return t;
}

/** Global start of every local bucket, starts[b][d], read back from the scanned table. */
export function starts(counts: readonly (readonly number[])[], layout: Layout): number[][] {
  const nBlocks = counts.length;
  const nBuckets = nBlocks ? counts[0].length : 0;
  const scanned = exclusiveScan(bucketTable(counts, layout));
  return counts.map((row, b) => row.map((_, d) => scanned[tableIndex(b, d, nBlocks, nBuckets, layout)]));
}

/** One store by one thread. `addr` is an element index into the output array. */
export interface Write {
  block: number;
  thread: number;
  /** Which store instruction of the thread (0 unless a thread owns several keys). */
  store: number;
  /** Local slot (local design) or input index (direct design). */
  slot: number;
  key: number;
  digit: number;
  addr: number;
}

export type Design = 'direct' | 'local';

export interface PassPlan {
  design: Design;
  n: number;
  shift: number;
  r: number;
  threadsPerBlock: number;
  keysPerThread: number;
  /** Empty for the direct design. */
  blocks: LocalBlock[];
  /** The linearised table and its exclusive scan (empty for the direct design). */
  table: number[];
  scanned: number[];
  starts: number[][];
  writes: Write[];
  /** The output array after the pass. */
  out: number[];
}

/** The direct scatter: thread i stores key i at its destination (one key per thread). */
export function directPass(keys: readonly number[], shift: number, r: number, threadsPerBlock: number): PassPlan {
  const { out, dest } = stablePass(keys, shift, r);
  const writes: Write[] = keys.map((key, i) => ({
    block: Math.floor(i / threadsPerBlock),
    thread: i % threadsPerBlock,
    store: 0,
    slot: i,
    key,
    digit: digitOf(key, shift, r),
    addr: dest[i],
  }));
  return { design: 'direct', n: keys.length, shift, r, threadsPerBlock, keysPerThread: 1, blocks: [], table: [], scanned: [], starts: [], writes, out };
}

/** The local design: local sort, table, scan, then each block writes its buckets as runs. */
export function localPass(
  keys: readonly number[],
  shift: number,
  r: number,
  threadsPerBlock: number,
  keysPerThread = 1,
  layout: Layout = 'digit-major',
): PassPlan {
  const per = threadsPerBlock * keysPerThread;
  const blocks = localSortBlocks(keys, shift, r, per);
  const counts = blocks.map((b) => b.counts);
  const table = bucketTable(counts, layout);
  const scanned = exclusiveScan(table);
  const st = starts(counts, layout);
  const writes: Write[] = [];
  const out = new Array<number>(keys.length);
  for (const b of blocks) {
    for (let slot = 0; slot < b.sorted.length; slot++) {
      const key = b.sorted[slot];
      const d = digitOf(key, shift, r);
      const addr = st[b.block][d] + (slot - b.offsets[d]);
      writes.push({ block: b.block, thread: slot % threadsPerBlock, store: Math.floor(slot / threadsPerBlock), slot, key, digit: d, addr });
      out[addr] = key;
    }
  }
  return { design: 'local', n: keys.length, shift, r, threadsPerBlock, keysPerThread, blocks, table, scanned, starts: st, writes, out };
}

export function plan(design: Design, keys: readonly number[], shift: number, r: number, threadsPerBlock: number, keysPerThread = 1): PassPlan {
  return design === 'direct' ? directPass(keys, shift, r, threadsPerBlock) : localPass(keys, shift, r, threadsPerBlock, keysPerThread);
}

/** True when the digits are in nondecreasing order: the pass partitioned the keys. */
export function isPartitioned(keys: readonly number[], shift: number, r: number): boolean {
  for (let i = 1; i < keys.length; i++) if (digitOf(keys[i - 1], shift, r) > digitOf(keys[i], shift, r)) return false;
  return true;
}

/**
 * Where every local slot lands when only the bucket sizes are known (the bucket-table
 * explorer): block b's slots hold counts[b][0] keys of digit 0, then counts[b][1] of
 * digit 1, and so on.
 */
export function placeFromCounts(counts: readonly (readonly number[])[], layout: Layout): { block: number; slot: number; digit: number; addr: number }[] {
  const st = starts(counts, layout);
  const out: { block: number; slot: number; digit: number; addr: number }[] = [];
  counts.forEach((row, b) => {
    let slot = 0;
    row.forEach((c, d) => {
      for (let j = 0; j < c; j++) out.push({ block: b, slot: slot++, digit: d, addr: st[b][d] + j });
    });
  });
  return out;
}

// ---------------------------------------------------------------- memory units

export interface UnitChoice {
  id: string;
  label: string;
  /** Bytes per unit, or null where the vendor documents none and the reader picks one. */
  bytes: number | null;
  /** What the unit is called. */
  noun: string;
  source: string;
}

export const UNITS: UnitChoice[] = [
  {
    id: 'nv-sector',
    get label() {
      return tr('NVIDIA: 32-byte sectors', 'NVIDIA: sector 32 byte');
    },
    bytes: 32,
    noun: 'sector',
    source: 'ch06-facts A2',
  },
  {
    id: 'nv-line',
    get label() {
      return tr("NVIDIA: 128-byte lines (the book's cache lines)", 'NVIDIA: line 128 byte (cache line của sách)');
    },
    bytes: 128,
    noun: 'line',
    source: 'ch06-facts A4',
  },
  {
    id: 'amd-64',
    get label() {
      return tr('AMD: 64-byte lines', 'AMD: line 64 byte');
    },
    bytes: 64,
    noun: 'line',
    source: 'ch06-facts B1',
  },
  {
    id: 'amd-128',
    get label() {
      return tr('AMD: 128-byte lines', 'AMD: line 128 byte');
    },
    bytes: 128,
    noun: 'line',
    source: 'ch06-facts B1',
  },
  {
    id: 'metal',
    get label() {
      return tr('Metal: not documented, pick a size', 'Metal: không có tài liệu, hãy chọn một kích thước');
    },
    bytes: null,
    noun: 'segment',
    source: 'ch06-facts C1',
  },
];

/** Separate runs of consecutive element addresses (lane order does not matter). */
export function runsOf(addrs: readonly number[]): number {
  if (!addrs.length) return 0;
  const s = [...new Set(addrs)].sort((a, b) => a - b);
  let runs = 1;
  for (let i = 1; i < s.length; i++) if (s[i] !== s[i - 1] + 1) runs++;
  return runs;
}

/** The aligned units the lanes' elements touch, ascending (an element straddling a boundary touches both). */
export function unitsOf(addrs: readonly number[], elemBytes: number, unitBytes: number): number[] {
  const u = new Set<number>();
  for (const a of addrs) {
    const lo = Math.floor((a * elemBytes) / unitBytes);
    const hi = Math.floor((a * elemBytes + elemBytes - 1) / unitBytes);
    for (let x = lo; x <= hi; x++) u.add(x);
  }
  return [...u].sort((a, b) => a - b);
}

/** Units one aligned, contiguous store of `lanes` elements needs: the coalesced minimum. */
export const minUnits = (lanes: number, elemBytes: number, unitBytes: number) => Math.ceil((lanes * elemBytes) / unitBytes);

export interface WarpStore {
  block: number;
  warp: number;
  store: number;
  /** Lane l's element address, in lane order. */
  addrs: number[];
  /** Lane l's key and digit. */
  keys: number[];
  digits: number[];
  runs: number;
  units: number[];
  minUnits: number;
}

/** Group a pass's writes into warp-wide store instructions and count each one. */
export function warpStores(p: PassPlan, warpWidth: number, elemBytes: number, unitBytes: number): WarpStore[] {
  const map = new Map<string, Write[]>();
  for (const w of p.writes) {
    const warp = Math.floor(w.thread / warpWidth);
    const k = `${w.block}/${w.store}/${warp}`;
    let g = map.get(k);
    if (!g) map.set(k, (g = []));
    g.push(w);
  }
  const out: WarpStore[] = [];
  for (const g of map.values()) {
    g.sort((a, b) => a.thread - b.thread);
    const addrs = g.map((w) => w.addr);
    out.push({
      block: g[0].block,
      warp: Math.floor(g[0].thread / warpWidth),
      store: g[0].store,
      addrs,
      keys: g.map((w) => w.key),
      digits: g.map((w) => w.digit),
      runs: runsOf(addrs),
      units: unitsOf(addrs, elemBytes, unitBytes),
      minUnits: minUnits(g.length, elemBytes, unitBytes),
    });
  }
  // Program order: block by block, store instruction by store instruction, warp by warp.
  return out.sort((a, b) => a.block - b.block || a.store - b.store || a.warp - b.warp);
}

export interface Summary {
  stores: number;
  meanRuns: number;
  meanUnits: number;
  meanMin: number;
  maxRuns: number;
  maxUnits: number;
  /** meanUnits / meanMin: 1 is perfectly coalesced. */
  ratio: number;
}

export function summarise(ws: readonly WarpStore[]): Summary {
  const n = ws.length;
  if (!n) return { stores: 0, meanRuns: 0, meanUnits: 0, meanMin: 0, maxRuns: 0, maxUnits: 0, ratio: 0 };
  let r = 0, u = 0, m = 0, mr = 0, mu = 0;
  for (const w of ws) {
    r += w.runs;
    u += w.units.length;
    m += w.minUnits;
    mr = Math.max(mr, w.runs);
    mu = Math.max(mu, w.units.length);
  }
  return { stores: n, meanRuns: r / n, meanUnits: u / n, meanMin: m / n, maxRuns: mr, maxUnits: mu, ratio: u / m };
}

/**
 * The same counts as summarise(warpStores(plan(...))) without building a write per key:
 * the sampled sweeps of sections 14.7 and 14.8 run it over every pass of a sort, many
 * times per control change. Returns the totals over all warp stores and the pass output.
 */
export function storeTotals(
  design: Design,
  keys: readonly number[],
  shift: number,
  r: number,
  threadsPerBlock: number,
  keysPerThread: number,
  warpWidth: number,
  elemBytes: number,
  unitBytes: number,
): { stores: number; runs: number; units: number; min: number; out: number[] } {
  const n = keys.length;
  const nb = 2 ** r;
  const c = design === 'direct' ? 1 : keysPerThread;
  const per = threadsPerBlock * c;
  const nBlocks = Math.ceil(n / per);
  const dig = new Int32Array(n);
  for (let i = 0; i < n; i++) dig[i] = (keys[i] >>> shift) & (nb - 1);
  // Global destination of input key i (direct), or of block-local slot j (local), in a flat array.
  const addr = new Int32Array(n);
  const slotKey = new Float64Array(n);
  if (design === 'direct') {
    const cnt = new Int32Array(nb);
    for (let i = 0; i < n; i++) cnt[dig[i]]++;
    for (let d = 0, s = 0; d < nb; d++) { const x = cnt[d]; cnt[d] = s; s += x; }
    for (let i = 0; i < n; i++) { addr[i] = cnt[dig[i]]++; slotKey[i] = keys[i]; }
  } else {
    // Per-block counts, laid out digit-major (Fig. 14.9), scanned.
    const tbl = new Float64Array(nb * nBlocks);
    for (let i = 0; i < n; i++) tbl[dig[i] * nBlocks + Math.floor(i / per)]++;
    for (let k = 0, s = 0; k < tbl.length; k++) { const x = tbl[k]; tbl[k] = s; s += x; }
    const off = new Int32Array(nb);
    for (let b = 0; b < nBlocks; b++) {
      const first = b * per;
      const end = Math.min(n, first + per);
      off.fill(0);
      for (let i = first; i < end; i++) off[dig[i]]++;
      for (let d = 0, s = 0; d < nb; d++) { const x = off[d]; off[d] = s; s += x; }
      for (let i = first; i < end; i++) {
        const d = dig[i];
        const j = first + off[d]++;
        addr[j] = tbl[d * nBlocks + b]++;
        slotKey[j] = keys[i];
      }
    }
  }
  const out = new Array<number>(n);
  for (let j = 0; j < n; j++) out[addr[j]] = slotKey[j];
  // Warp stores: in block b, store k, warp w, the lanes hold flat positions b*per + k*B + w*W + lane.
  let stores = 0, runs = 0, units = 0, min = 0;
  const lanes = Math.min(warpWidth, threadsPerBlock);
  const buf = new Int32Array(lanes);
  for (let b = 0; b < nBlocks; b++) {
    const first = b * per;
    const end = Math.min(n, first + per);
    for (let k = 0; k * threadsPerBlock < end - first; k++) {
      for (let w = 0; w * warpWidth < threadsPerBlock; w++) {
        const lo = first + k * threadsPerBlock + w * warpWidth;
        const hi = Math.min(end, lo + lanes, first + k * threadsPerBlock + threadsPerBlock);
        if (lo >= hi) continue;
        const m = hi - lo;
        for (let t = 0; t < m; t++) buf[t] = addr[lo + t];
        const a = buf.subarray(0, m).sort();
        let rr = 1;
        let uu = 0;
        let lastUnit = -1;
        for (let t = 0; t < m; t++) {
          if (t > 0 && a[t] !== a[t - 1] + 1) rr++;
          const u0 = Math.floor((a[t] * elemBytes) / unitBytes);
          const u1 = Math.floor((a[t] * elemBytes + elemBytes - 1) / unitBytes);
          for (let x = Math.max(u0, lastUnit + 1); x <= u1; x++) uu++;
          if (u1 > lastUnit) lastUnit = u1;
        }
        stores++;
        runs += rr;
        units += uu;
        min += Math.ceil((m * elemBytes) / unitBytes);
      }
    }
  }
  return { stores, runs, units, min, out };
}

// ---------------------------------------------------------------- keys

/** A small seeded generator (mulberry32), so every figure on the page is reproducible. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** n uniformly random unsigned keys of `bits` bits. */
export function uniformKeys(n: number, seed: number, bits = 32): number[] {
  const g = rng(seed);
  return Array.from({ length: n }, () => Math.floor(g() * 2 ** bits));
}

const F32 = new Float32Array(1);
const U32 = new Uint32Array(F32.buffer);
/** The bits of x as a 32-bit float. */
export function f32bits(x: number): number {
  F32[0] = x;
  return U32[0];
}
/** The bits of x as bf16: float32 rounded to nearest even in its top 16 bits. */
export function bf16bits(x: number): number {
  const u = f32bits(x);
  return ((u + 0x7fff + ((u >>> 16) & 1)) >>> 16) & 0xffff;
}

/**
 * A float's bits as an unsigned key whose order is the float's order (ch14-facts A3, E1):
 * flip the sign bit of a positive value, every bit of a negative one, map -0.0 onto +0.0
 * first, and invert the result for a descending sort.
 */
export function orderedKey(bits: number, width: 32 | 16, dir: 'asc' | 'desc'): number {
  const sign = width === 32 ? 0x80000000 : 0x8000;
  const all = width === 32 ? 0xffffffff : 0xffff;
  const u = bits === sign ? 0 : bits;
  const k = (u & sign ? (u ^ all) : (u | sign)) >>> 0;
  return (dir === 'desc' ? ~k & all : k) >>> 0;
}

/**
 * A synthetic softmax row (ours): logits drawn from a normal distribution with standard
 * deviation `sigma`, exponentiated and normalised. Not a measured model output.
 */
export function softmaxRow(n: number, seed: number, sigma = 2): number[] {
  const g = rng(seed);
  const z = new Array<number>(n);
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.max(g(), 1e-12);
    const u2 = g();
    const m = Math.sqrt(-2 * Math.log(u1));
    z[i] = sigma * m * Math.cos(2 * Math.PI * u2);
    if (i + 1 < n) z[i + 1] = sigma * m * Math.sin(2 * Math.PI * u2);
  }
  let mx = -Infinity;
  for (const v of z) if (v > mx) mx = v;
  let s = 0;
  const e = z.map((v) => {
    const x = Math.exp(v - mx);
    s += x;
    return x;
  });
  return e.map((x) => x / s);
}

/** The first `n` probabilities of a `rowLength`-token row, as keys for a descending sort. */
export function softmaxKeys(n: number, seed: number, width: 32 | 16 = 32, rowLength = n): number[] {
  const p = softmaxRow(Math.max(n, rowLength), seed).slice(0, n);
  return p.map((x) => orderedKey(width === 32 ? f32bits(x) : bf16bits(x), width, 'desc'));
}
