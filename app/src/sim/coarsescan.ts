// A model of the coarsened block scan of Figs. 11.11 to 11.13 (book §11.6 and §11.7,
// pp. 267–272).
//
// What it models:
//   - the three phases on values: each thread scans its own run of F elements one after
//     another, the run totals go through one block scan, each thread adds its
//     predecessor's scanned total (scan-scan-add once more, Fig. 11.11);
//   - the book's work and step counts with and without coarsening (pp. 266, 269–270);
//   - the two index mappings: chunk-major for global ↔ shared (in load iteration c, thread
//     t moves element c·BLOCK_DIM + t) and thread-major for the scan (thread t owns elements
//     t·F to t·F + F − 1), and the addresses one warp touches in one load instruction when
//     each thread loads its own run directly instead; segments are counted by chapter 6's
//     warpAccess (coalesce.ts, a fixed contract, imported read-only);
//   - the shared-memory, register and global accesses of every line of Figs. 11.12 and
//     11.13 that touches the run (coarsescan.test.ts ties each entry to the line it names
//     in labs/ch11/snippets/s116_coarsened_as_printed.cu and s117_regtiled_as_printed.cu);
//   - the bank of every lane's word for buffer_s[t·F + c] (and for the load side,
//     buffer_s[c·BLOCK_DIM + t]) with 32 or 64 banks of 4 bytes, unpadded or padded by one
//     word every bank-count words (infra-facts A10, B10; exercise 4, p. 286).
//
// What it leaves out: time, caches and the hardware's scheduling. A load instruction is
// counted on its own: a segment fetched by one instruction is not remembered by the next,
// although a real cache may serve it. A bank map counts the distinct words that land in
// each bank for one access; what that costs in cycles is vendor-specific (NVIDIA serialises
// the conflicting accesses, A10; AMD serves a wavefront 16 lanes per cycle, B10), and
// Apple documents no banks at all (C13).
//
// Used by sections 11.6 and 11.7 and by the "Three phases of a coarsened scan" scene.

import { warpAccess, runs as byteRuns, type WarpAccess } from './coalesce';
import { FIG_11_6, inclusiveScan } from './warpscan';
import { tr } from '../lib/locale.ts';

export const log2 = (n: number) => Math.log2(n);
const isPow2 = (n: number) => n >= 1 && (n & (n - 1)) === 0;

/** Small integers to scan: Fig. 11.6's sixteen for 16 elements, a fixed pattern otherwise. */
export function demoValues(n: number): number[] {
  if (n === 16) return [...FIG_11_6];
  return Array.from({ length: n }, (_, i) => (i * 7 + 2) % 6);
}

// ---------------------------------------------------------------- the three phases

export interface ThreePhases {
  /** Threads (BLOCK_DIM). */
  P: number;
  /** Elements per thread (COARSE_FACTOR). */
  F: number;
  N: number;
  input: number[];
  /** After phase 1: each thread's run scanned sequentially, run t at index t. */
  runs: number[][];
  /** Each run's total: its last value after phase 1. */
  totals: number[];
  /** Phase 2: the totals scanned (inclusive), one block scan. */
  scannedTotals: number[];
  /** What phase 3 adds to each run: 0 for thread 0, scannedTotals[t − 1] otherwise. */
  added: number[];
  /** After phase 3, in element order. */
  final: number[];
  expected: number[];
  correct: boolean;
}

export function threePhases(input: readonly number[], P: number): ThreePhases {
  const N = input.length;
  if (P < 1 || N % P !== 0) throw new Error('N must be a whole number of runs');
  const F = N / P;
  const runs = Array.from({ length: P }, (_, t) => inclusiveScan(input.slice(t * F, t * F + F)));
  const totals = runs.map((r) => r[F - 1]);
  const scannedTotals = inclusiveScan(totals);
  const added = runs.map((_, t) => (t === 0 ? 0 : scannedTotals[t - 1]));
  const final = runs.flatMap((r, t) => r.map((v) => v + added[t]));
  const expected = inclusiveScan(input);
  return { P, F, N, input: [...input], runs, totals, scannedTotals, added, final, expected, correct: final.every((v, i) => v === expected[i]) };
}

// ---------------------------------------------------------------- work and steps (pp. 266, 269–270)

export interface Count {
  total: number;
  parts: { label: string; value: number }[];
}

/** Additions of the coarsened scan of N elements by P threads: 2N + P·log2P − P − N/P. */
export function coarsenedWork(N: number, P: number): Count {
  const seq = N - P;
  const scan = P * log2(P);
  const add = N - N / P;
  return {
    total: seq + scan + add,
    parts: [
      { label: tr('sequential run scans, P·(N/P − 1)', 'scan tuần tự từng đoạn, P·(N/P − 1)'), value: seq },
      { label: tr('block scan of the run totals, P·log2P', 'scan cấp khối trên tổng của các đoạn, P·log2P'), value: scan },
      { label: tr('adding the predecessor\'s total, (P − 1)·N/P', 'cộng tổng của đoạn đứng trước, (P − 1)·N/P'), value: add },
    ],
  };
}

/** Steps of the coarsened scan: (N/P − 1) + log2P + N/P = 2N/P + log2P − 1. */
export function coarsenedSteps(N: number, P: number): Count {
  const seq = N / P - 1;
  const scan = log2(P);
  const add = N / P;
  return {
    total: seq + scan + add,
    parts: [
      { label: tr('sequential run scan', 'scan tuần tự từng đoạn'), value: seq },
      { label: tr('block scan of the totals', 'scan cấp khối trên các tổng'), value: scan },
      { label: tr('add', 'cộng'), value: add },
    ],
  };
}

/** Kogge-Stone on N elements, one per thread: N·log2N − (N − 1) additions (p. 266). */
export const koggeStoneWork = (N: number) => N * log2(N) - (N - 1);
/** Kogge-Stone's steps with P execution units, (N·log2N)/P (p. 266). */
export const koggeStoneSteps = (N: number, P: number) => (N * log2(N)) / P;
/** The sequential loop: N − 1 additions in about N steps (pp. 265–266). */
export const sequentialWork = (N: number) => N - 1;

// ---------------------------------------------------------------- index mappings (p. 269)

export type LoadMode = 'staged' | 'direct';
export const LOAD_MODE_LABEL: Record<LoadMode, string> = {
  get staged() {
    return tr('chunk at a time, through shared memory (Fig. 11.12)', 'từng chunk một, qua bộ nhớ chia sẻ (Hình 11.12)');
  },
  get direct() {
    return tr('each thread loads its own run', 'mỗi luồng tự tải đoạn của mình');
  },
};

/** The element thread t moves in load iteration c. */
export const elementOf = (mode: LoadMode, t: number, c: number, B: number, F: number) => (mode === 'staged' ? c * B + t : t * F + c);
/** The thread whose run holds element e, i.e. which thread scans it. */
export const ownerOf = (e: number, F: number) => Math.floor(e / F);
/** Which thread loads element e, and in which iteration. */
export function loaderOf(mode: LoadMode, e: number, B: number, F: number): { thread: number; iter: number } {
  return mode === 'staged' ? { thread: e % B, iter: Math.floor(e / B) } : { thread: Math.floor(e / F), iter: e % F };
}

export interface WarpLoad {
  mode: LoadMode;
  c: number;
  /** Element each lane loads, lane order. */
  elements: number[];
  access: WarpAccess;
  /** Separate runs of consecutive bytes: 1 is the book's picture of a coalesced access (p. 126). */
  runs: number;
  /** Segment index of each lane's element. */
  segmentOf: number[];
}

/**
 * The global addresses one warp touches in load iteration c of a block segment that starts
 * on a segment boundary (cudaMalloc returns 256-byte-aligned memory, ch06-facts A3).
 */
export function warpLoad(mode: LoadMode, c: number, F: number, B: number, warpWidth: number, segBytes: number, warp = 0, elemBytes = 4): WarpLoad {
  const elements: number[] = [];
  for (let l = 0; l < warpWidth; l++) {
    const t = warp * warpWidth + l;
    if (t < B) elements.push(elementOf(mode, t, c, B, F));
  }
  const bytes = elements.map((e) => e * elemBytes);
  return {
    mode,
    c,
    elements,
    access: warpAccess(bytes, elemBytes, segBytes),
    runs: byteRuns(bytes, elemBytes),
    segmentOf: bytes.map((b) => Math.floor(b / segBytes)),
  };
}

/** Segments for a whole load (all F iterations) of one warp, each instruction counted on its own. */
export function loadTransactions(mode: LoadMode, F: number, B: number, warpWidth: number, segBytes: number): number {
  let n = 0;
  for (let c = 0; c < F; c++) n += warpLoad(mode, c, F, B, warpWidth, segBytes).access.transactions;
  return n;
}

// ---------------------------------------------------------------- accesses per line (Figs. 11.12, 11.13)

export type CoarseKernel = 'fig11.12' | 'fig11.13';
export type Phase = 'load' | 'scan' | 'total' | 'hand-off' | 'add' | 'store';

export interface AccessLine {
  phase: Phase;
  /** The book's line number in the figure (the first line of a statement that spans two). */
  line: number;
  /** The text on that line, as printed. */
  code: string;
  /** How many times one thread executes it, as a function of F. */
  trip: 'F' | 'F − 1' | '1';
  /** Does thread 0 execute it? */
  thread0: boolean;
  /** Accesses per execution. buffer_s and threadSums are shared memory; buffer_r is registers. */
  per: { sharedRead: number; sharedWrite: number; regRead: number; regWrite: number; globalRead: number; globalWrite: number };
  /** Which shared array the shared accesses go to. */
  array?: 'buffer_s' | 'threadSums';
}

const z = { sharedRead: 0, sharedWrite: 0, regRead: 0, regWrite: 0, globalRead: 0, globalWrite: 0 };

export const ACCESS_LINES: Record<CoarseKernel, AccessLine[]> = {
  'fig11.12': [
    { phase: 'load', line: 6, code: 'buffer_s[c*BLOCK_DIM + threadIdx.x] =', trip: 'F', thread0: true, per: { ...z, sharedWrite: 1, globalRead: 1 }, array: 'buffer_s' },
    { phase: 'scan', line: 13, code: 'buffer_s[threadSegment + c] += buffer_s[threadSegment + c - 1];', trip: 'F − 1', thread0: true, per: { ...z, sharedRead: 2, sharedWrite: 1 }, array: 'buffer_s' },
    { phase: 'total', line: 16, code: 'float threadSum = buffer_s[threadSegment + COARSE_FACTOR - 1];', trip: '1', thread0: true, per: { ...z, sharedRead: 1 }, array: 'buffer_s' },
    { phase: 'hand-off', line: 20, code: 'threadSums[threadIdx.x] = threadSum;', trip: '1', thread0: true, per: { ...z, sharedWrite: 1 }, array: 'threadSums' },
    { phase: 'hand-off', line: 24, code: 'float prevPartialSum = threadSums[threadIdx.x - 1];', trip: '1', thread0: false, per: { ...z, sharedRead: 1 }, array: 'threadSums' },
    { phase: 'add', line: 26, code: 'buffer_s[threadSegment + c] += prevPartialSum;', trip: 'F', thread0: false, per: { ...z, sharedRead: 1, sharedWrite: 1 }, array: 'buffer_s' },
    { phase: 'store', line: 32, code: 'output[blockSegment + c*BLOCK_DIM + threadIdx.x] =', trip: 'F', thread0: true, per: { ...z, sharedRead: 1, globalWrite: 1 }, array: 'buffer_s' },
  ],
  'fig11.13': [
    { phase: 'load', line: 6, code: 'buffer_s[c*BLOCK_DIM + threadIdx.x] =', trip: 'F', thread0: true, per: { ...z, sharedWrite: 1, globalRead: 1 }, array: 'buffer_s' },
    { phase: 'scan', line: 13, code: 'buffer_r[0] = buffer_s[threadSegment];', trip: '1', thread0: true, per: { ...z, sharedRead: 1, regWrite: 1 }, array: 'buffer_s' },
    { phase: 'scan', line: 16, code: 'buffer_r[c] = buffer_s[threadSegment + c] + buffer_r[c - 1];', trip: 'F − 1', thread0: true, per: { ...z, sharedRead: 1, regRead: 1, regWrite: 1 }, array: 'buffer_s' },
    { phase: 'total', line: 19, code: 'float threadSum = buffer_r[COARSE_FACTOR - 1];', trip: '1', thread0: true, per: { ...z, regRead: 1 } },
    { phase: 'hand-off', line: 23, code: 'threadSums[threadIdx.x] = threadSum;', trip: '1', thread0: true, per: { ...z, sharedWrite: 1 }, array: 'threadSums' },
    { phase: 'hand-off', line: 26, code: 'float prevPartialSum = (threadIdx.x>0)?threadSums[threadIdx.x - 1]:0.0f;', trip: '1', thread0: false, per: { ...z, sharedRead: 1 }, array: 'threadSums' },
    { phase: 'add', line: 29, code: 'buffer_s[threadSegment + c] = buffer_r[c] + prevPartialSum;', trip: 'F', thread0: true, per: { ...z, sharedWrite: 1, regRead: 1 }, array: 'buffer_s' },
    { phase: 'store', line: 34, code: 'output[blockSegment + c*BLOCK_DIM + threadIdx.x] =', trip: 'F', thread0: true, per: { ...z, sharedRead: 1, globalWrite: 1 }, array: 'buffer_s' },
  ],
};

export const tripCount = (trip: AccessLine['trip'], F: number) => (trip === 'F' ? F : trip === 'F − 1' ? F - 1 : 1);

export interface AccessTotals {
  /** Every access counted once per execution, for one thread. */
  lines: (AccessLine & { times: number; sharedRead: number; sharedWrite: number; regRead: number; regWrite: number; globalRead: number; globalWrite: number })[];
  /** Accesses to buffer_s, the run itself, for one thread. */
  bufferS: { read: number; write: number; total: number };
  /** Accesses to threadSums, the per-thread hand-off. */
  threadSums: { read: number; write: number };
  registers: { read: number; write: number };
  global: { read: number; write: number };
  /** buffer_s accesses per element of the thread's run. */
  perElement: number;
}

/** One thread's accesses in Fig. 11.12 or 11.13; thread 0 skips the lines it does not execute. */
export function sharedAccesses(kernel: CoarseKernel, F: number, thread: 'first' | 'other' = 'other'): AccessTotals {
  const lines = ACCESS_LINES[kernel].map((l) => {
    const times = l.thread0 || thread === 'other' ? tripCount(l.trip, F) : 0;
    return {
      ...l,
      times,
      sharedRead: l.per.sharedRead * times,
      sharedWrite: l.per.sharedWrite * times,
      regRead: l.per.regRead * times,
      regWrite: l.per.regWrite * times,
      globalRead: l.per.globalRead * times,
      globalWrite: l.per.globalWrite * times,
    };
  });
  const sum = (f: (x: (typeof lines)[number]) => number, arr?: AccessLine['array']) =>
    lines.filter((l) => arr == null || l.array === arr).reduce((n, l) => n + f(l), 0);
  const bufferS = { read: sum((l) => l.sharedRead, 'buffer_s'), write: sum((l) => l.sharedWrite, 'buffer_s'), total: 0 };
  bufferS.total = bufferS.read + bufferS.write;
  return {
    lines,
    bufferS,
    threadSums: { read: sum((l) => l.sharedRead, 'threadSums'), write: sum((l) => l.sharedWrite, 'threadSums') },
    registers: { read: sum((l) => l.regRead), write: sum((l) => l.regWrite) },
    global: { read: sum((l) => l.globalRead), write: sum((l) => l.globalWrite) },
    perElement: bufferS.total / F,
  };
}

// ---------------------------------------------------------------- banks (exercise 4, p. 286)

export const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));
/** One extra word every `banks` words: index i lands at i + floor(i / banks). */
export const paddedIndex = (i: number, banks: number) => i + Math.floor(i / banks);

export type BankSide = 'scan' | 'load';
export interface BankLane {
  lane: number;
  /** Index into buffer_s as the code writes it. */
  word: number;
  /** Where it lands with padding (= word without). */
  slot: number;
  bank: number;
}
export interface BankMap {
  lanes: BankLane[];
  /** Distinct words that land in each bank. */
  perBank: number[];
  /** The most distinct words any one bank serves: 1 means no conflict. */
  degree: number;
}

/**
 * The bank of every lane's word for one access of one warp. Scan side: buffer_s[t·F + c]
 * (Fig. 11.13, lines 13, 16 and 29). Load side: buffer_s[c·BLOCK_DIM + t] (lines 06 and 35).
 */
export function bankMap(F: number, banks: number, pad: boolean, opts: { c?: number; lanes?: number; side?: BankSide; blockDim?: number } = {}): BankMap {
  const c = opts.c ?? 0;
  const n = opts.lanes ?? 32;
  const side = opts.side ?? 'scan';
  const B = opts.blockDim ?? 256;
  const lanes: BankLane[] = [];
  for (let t = 0; t < n; t++) {
    const word = side === 'scan' ? t * F + c : c * B + t;
    const slot = pad ? paddedIndex(word, banks) : word;
    lanes.push({ lane: t, word, slot, bank: slot % banks });
  }
  const words: Set<number>[] = Array.from({ length: banks }, () => new Set<number>());
  for (const l of lanes) words[l.bank].add(l.slot);
  const perBank = words.map((s) => s.size);
  return { lanes, perBank, degree: Math.max(...perBank) };
}

/**
 * Unpadded scan-side degree for `lanes` lanes: the lanes spread over banks / gcd(F, banks)
 * banks, so max(1, lanes·gcd(F, banks) / banks). With 32 lanes on 32 banks that is gcd(F, 32).
 */
export const conflictDegree = (F: number, banks: number, lanes = 32) => Math.max(1, (lanes * gcd(F, banks)) / banks);

/** The worst degree over all F iterations c of the scan-side access. */
export function worstDegree(F: number, banks: number, pad: boolean, lanes = 32): number {
  let d = 0;
  for (let c = 0; c < F; c++) d = Math.max(d, bankMap(F, banks, pad, { c, lanes }).degree);
  return d;
}

// ---------------------------------------------------------------- the scene

/** What the "Three phases of a coarsened scan" scene draws: one block of 4 threads, factor 4. */
export function sceneData() {
  const P = 4;
  const F = 4;
  const values = demoValues(P * F);
  const phases = threePhases(values, P);
  // A four-lane toy warp: the whole block is one warp here.
  const staged = Array.from({ length: F }, (_, c) => warpLoad('staged', c, F, P, P, 32));
  const direct = Array.from({ length: F }, (_, c) => warpLoad('direct', c, F, P, P, 32));
  return {
    P,
    F,
    N: P * F,
    values,
    phases,
    staged,
    direct,
    work: coarsenedWork(P * F, P).total,
    perElement: sharedAccesses('fig11.13', F).perElement,
  };
}

export const isPowerOfTwo = isPow2;
