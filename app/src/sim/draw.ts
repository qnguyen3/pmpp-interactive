// The final draw of milestone M5: the model behind section X1.8 (the "Find u" stepper and
// its barrier hunt, "Near a boundary", "Same bits, three samplers") and every beat of the
// "Finding u in the scan" scene. Nothing here is in the book: it is built from the scan of
// ch. 11 (§11.1, §11.4), the two-stage reduction of ch. 10 (§10.8) and the warp vote of
// ch. 12 (§12.3).
//
// What it holds:
//   - inverseCdf: the rule on one row. The token is the first whose inclusive sum is
//     strictly greater than the target u x total ('gt', M5's rule), or greater or equal
//     ('ge', llama.cpp's host rule, x1-facts D3); with or without the fallback to the last
//     token of positive weight when nothing qualifies (FlashInfer's, A6);
//   - drawKernel / chunkedSearch: labs/x1/snippets/draw_search.cu step by step, in float32
//     in the kernel's own order of additions, for one block: the row total (a strided sum
//     per thread, a shuffle-down tree per warp, warp 0 over the warp sums), then chunks of
//     threads x VEC survivors: each thread's inclusive sums over its VEC survivors, a
//     Kogge-Stone scan of the thread totals inside each warp (shuffle up), the carries from
//     the warp totals, the chunk's sum, the skip test, each survivor's flag, one ballot per
//     warp, the lowest flagged lane and the lowest flagged warp;
//   - barrierHunt: the kernel's first phase as segments between synchronisation points,
//     run under the CPU emulator's two fixed orders (each thread runs until its next
//     barrier or warp call, then the next thread; labs/README.md) and under one order a GPU
//     may also produce (one warp runs ahead to the next barrier), with either barrier around
//     the read of the total removed: which threads read the total and which read something
//     else;
//   - CDFs in three orders (double; float32 left to right; float32 in the kernel's order),
//     an optional relative perturbation of the weights, the flip points of each boundary,
//     the bands of u where the orders choose differently, a draw's margin and the bound
//     2 (n - 1) m on the share of draws within a margin m of a boundary;
//   - three exact samplers fed the same uniforms from philox.ts: the inverse CDF, the
//     exponential race argmax(p / q) (PyTorch's multinomial for one sample, D5) and
//     Gumbel-max argmax(l + g) (MLX for more than one row, D7).
//
// What it simplifies (the lesson says so beside each interactive):
//   - one block; the thread count and the warp width are display choices (the emulator's
//     warp is 32 and real devices differ), and the barrier hunt models only the emulator's
//     scheduling, not GPU timing;
//   - float32 is imitated with Math.fround after every addition and multiplication; exp is
//     Math.exp rounded to float32, not any backend's exp (ch10 facts A11, B4, C4), which the
//     perturbation stands in for; no fused multiply-add (there is none to form here: the
//     kernel's only multiplication is u x total);
//   - the three-sampler explorer works in double precision.
//
// Value imports from scan.ts and philox.ts: browser checks load this file through a resolve
// hook that adds the .ts extension (see app/e2e/x1-sx18.mjs).
import { inclusiveIn } from './scan';
import { m5Uniform, philox } from './philox';
import { tr } from '../lib/locale.ts';

const f32 = Math.fround;

export type DrawRule = 'gt' | 'ge';
export const RULE_LABEL: Record<DrawRule, string> = {
  get gt() { return tr('first inclusive sum > target', 'tổng bao gồm đầu tiên > target'); },
  get ge() { return tr('first inclusive sum ≥ target', 'tổng bao gồm đầu tiên ≥ target'); },
};

/** The last index whose weight is positive, or -1. */
export function lastPositive(w: number[]): number {
  for (let i = w.length - 1; i >= 0; i--) if (w[i] > 0) return i;
  return -1;
}

export interface Draw {
  index: number;
  target: number;
  total: number;
  inclusive: number[];
  /** Nothing qualified and the fallback chose. */
  fallback: boolean;
  /** Nothing qualified and there was no fallback: the index is n, one past the row. */
  invalid: boolean;
}
export interface DrawOpts {
  rule?: DrawRule;
  fallback?: boolean;
  /** Compare with u x total (true, M5) or with u itself, for rows of probabilities (FlashInfer, cuRAND's u). */
  scale?: boolean;
  /** Rounding after every operation: identity for double, Math.fround for float32. */
  round?: (v: number) => number;
}

/** The inverse CDF on one row, the inclusive sums formed left to right. */
export function inverseCdf(weights: number[], u: number, opts: DrawOpts = {}): Draw {
  const { rule = 'gt', fallback = true, scale = true, round = (v: number) => v } = opts;
  const inclusive: number[] = [];
  let acc = 0;
  weights.forEach((w, i) => {
    acc = i === 0 ? round(w) : round(acc + w);
    inclusive.push(acc);
  });
  const total = inclusive.length ? inclusive[inclusive.length - 1] : 0;
  const target = scale ? round(u * total) : u;
  const hit = inclusive.findIndex((c) => (rule === 'gt' ? c > target : c >= target));
  if (hit >= 0) return { index: hit, target, total, inclusive, fallback: false, invalid: false };
  if (fallback) return { index: lastPositive(weights), target, total, inclusive, fallback: true, invalid: false };
  return { index: weights.length, target, total, inclusive, fallback: false, invalid: true };
}

// ---------------------------------------------------------------- the kernel, one block
export interface KernelCfg {
  /** Threads in the block. */
  threads: number;
  /** Display warp width (the emulator's is 32; NVIDIA 32, AMD 64 or 32, Metal read from the pipeline). */
  warp: number;
  /** Survivors per thread per chunk (VEC in the snippet). */
  vec: number;
}
export const STEPPER_CFG: KernelCfg = { threads: 16, warp: 8, vec: 4 };
export const SNIPPET_CFG: KernelCfg = { threads: 256, warp: 32, vec: 4 };

/** A warp's shuffle-down tree (Fig. 10.13's order): lane 0's result. Lanes past the end keep their own value. */
export function warpSumDown(v: number[]): number {
  let cur = v.map((x) => f32(x));
  for (let off = cur.length / 2; off >= 1; off /= 2) cur = cur.map((x, l) => f32(x + (l + off < cur.length ? cur[l + off] : x)));
  return cur[0];
}

export interface Lane {
  thread: number;
  warp: number;
  lane: number;
  /** Index of this thread's first survivor in the chunk. */
  first: number;
  /** Its VEC weights (0 past the end of the row). */
  w: number[];
  /** Inclusive sums over its own VEC survivors. */
  local: number[];
  /** Its total (= local[VEC - 1]). */
  run: number;
  /** Inclusive and exclusive scan of the thread totals inside the warp. */
  incl: number;
  excl: number;
  /** Sum of the totals of the warps before this one, this chunk. */
  carry: number;
  /** aggregate + (carry + excl): everything before this thread's first survivor. */
  before: number;
  /** before + local[j]; empty when the chunk was skipped. */
  cum: number[];
  flags: boolean[];
  /** The first flagged survivor among its VEC, or -1. */
  hit: number;
}
export interface Chunk {
  c: number;
  base: number;
  /** One past the last survivor in the chunk. */
  end: number;
  warpTotals: number[];
  sum: number;
  aggBefore: number;
  aggAfter: number;
  /** aggregate + sum > target: the chunk is scanned and voted on. */
  crosses: boolean;
  lanes: Lane[];
  /** One ballot per warp (bit l = lane l found a survivor), when the chunk crosses. */
  ballots: number[];
  /** Lowest flagged lane per warp, or -1. */
  warpFirst: number[];
  /** The block's winner in this chunk (an index into the row), or null. */
  winner: number | null;
}
export interface KernelTrace {
  cfg: KernelCfg;
  n: number;
  u: number;
  /** Phase 1: per-thread strided sums, per-warp sums, the total. */
  partials: number[];
  warpSums: number[];
  total: number;
  target: number;
  chunks: Chunk[];
  index: number;
  fallback: boolean;
  winner: { chunk: number; warp: number; lane: number; thread: number; j: number } | null;
}

/** Phase 1 of the kernel: the row's total in float32, in the kernel's order. */
export function kernelTotal(weights: number[], cfg: KernelCfg): { partials: number[]; warpSums: number[]; total: number } {
  const n = weights.length;
  const partials = Array.from({ length: cfg.threads }, (_, t) => {
    let p = 0;
    for (let i = t; i < n; i += cfg.threads) p = f32(p + weights[i]);
    return p;
  });
  const nW = Math.ceil(cfg.threads / cfg.warp);
  const warpSums = Array.from({ length: nW }, (_, w) => warpSumDown(partials.slice(w * cfg.warp, (w + 1) * cfg.warp)));
  const total = warpSumDown(Array.from({ length: cfg.warp }, (_, l) => (l < nW ? warpSums[l] : 0)));
  return { partials, warpSums, total };
}

/**
 * One chunk scanned as the kernel scans it (every survivor's cumulative sum, flags against
 * the target). `target` is one number, or one per thread (what each thread read as the total
 * after a missing barrier); `flagIt` says which threads take the crossing branch.
 */
function scanChunk(weights: number[], base: number, aggregate: number, target: number | number[], cfg: KernelCfg, flagIt: boolean | boolean[]): Omit<Chunk, 'c' | 'aggAfter' | 'crosses' | 'winner'> {
  const n = weights.length;
  const { threads, warp, vec } = cfg;
  const nW = Math.ceil(threads / warp);
  const raw = Array.from({ length: threads }, (_, t) => {
    const first = base + t * vec;
    const w = Array.from({ length: vec }, (_, j) => (first + j < n ? weights[first + j] : 0));
    const local = inclusiveIn(w, 'loop');
    return { first, w, local, run: local[vec - 1] };
  });
  const inclAll: number[] = [];
  for (let k = 0; k < nW; k++) inclAll.push(...inclusiveIn(raw.slice(k * warp, (k + 1) * warp).map((r) => r.run), 'kogge-stone'));
  const warpTotals = Array.from({ length: nW }, (_, k) => inclAll[Math.min(threads, (k + 1) * warp) - 1]);
  let sum = 0;
  for (let k = 0; k < nW; k++) sum = f32(sum + warpTotals[k]);
  const lanes: Lane[] = raw.map((r, t) => {
    const wi = Math.floor(t / warp);
    const lane = t % warp;
    let carry = 0;
    for (let k = 0; k < wi; k++) carry = f32(carry + warpTotals[k]);
    const incl = inclAll[t];
    const excl = lane === 0 ? 0 : inclAll[t - 1];
    const before = f32(aggregate + f32(carry + excl));
    const on = Array.isArray(flagIt) ? flagIt[t] : flagIt;
    const tt = Array.isArray(target) ? target[t] : target;
    const cum = on ? r.local.map((x) => f32(before + x)) : [];
    const flags = on ? cum.map((c, j) => r.first + j < n && c > tt) : [];
    const hit = flags.findIndex(Boolean);
    return { thread: t, warp: wi, lane, first: r.first, w: r.w, local: r.local, run: r.run, incl, excl, carry, before, cum, flags, hit };
  });
  return { base, end: Math.min(n, base + threads * vec), warpTotals, sum, aggBefore: aggregate, lanes, ballots: [], warpFirst: [] };
}

export interface Search {
  chunks: Chunk[];
  index: number;
  fallback: boolean;
  winner: KernelTrace['winner'];
  /** Some warp had lanes on both sides of the crossing test, so its full-mask ballot was not reached by every lane (only possible with per-thread targets). */
  voteSplit: boolean;
}

/**
 * The chunk loop of the kernel against a given target (float32, the kernel's order). The
 * target is one number, or one per thread: what each thread computed from the value it read
 * as the total (barrierHunt), so a missing barrier can be followed to the token it causes.
 */
export function chunkedSearch(weights: number[], target: number | number[], cfg: KernelCfg = STEPPER_CFG): Search {
  const n = weights.length;
  const nW = Math.ceil(cfg.threads / cfg.warp);
  const chunks: Chunk[] = [];
  const tOf = (t: number) => (Array.isArray(target) ? target[t] : target);
  let aggregate = 0;
  let winner: KernelTrace['winner'] = null;
  let voteSplit = false;
  for (let base = 0, c = 0; base < n; base += cfg.threads * cfg.vec, c++) {
    const probe = scanChunk(weights, base, aggregate, target, cfg, false);
    const each = Array.from({ length: cfg.threads }, (_, t) => f32(aggregate + probe.sum) > tOf(t));
    const crosses = each.some(Boolean);
    for (let k = 0; k < nW; k++) {
      const ws = each.slice(k * cfg.warp, (k + 1) * cfg.warp);
      if (ws.some(Boolean) && !ws.every(Boolean)) voteSplit = true;
    }
    const s = crosses ? scanChunk(weights, base, aggregate, target, cfg, each) : probe;
    const ballots: number[] = [];
    const warpFirst: number[] = [];
    let best: number | null = null;
    if (crosses) {
      for (let k = 0; k < nW; k++) {
        const ls = s.lanes.filter((l) => l.warp === k);
        const mask = ls.reduce((m, l) => (l.hit >= 0 ? m + 2 ** l.lane : m), 0);
        ballots.push(mask);
        const firstLane = ls.find((l) => l.hit >= 0);
        warpFirst.push(firstLane ? firstLane.lane : -1);
        if (firstLane) {
          const cand = firstLane.first + firstLane.hit;
          if (best === null || cand < best) {
            best = cand;
            winner = { chunk: c, warp: k, lane: firstLane.lane, thread: firstLane.thread, j: firstLane.hit };
          }
        }
      }
    }
    const aggAfter = f32(aggregate + s.sum);
    chunks.push({ ...s, c, crosses, ballots, warpFirst, winner: best, aggAfter });
    if (best !== null) return { chunks, index: best, fallback: false, winner, voteSplit };
    aggregate = aggAfter;
  }
  return { chunks, index: lastPositive(weights), fallback: true, winner: null, voteSplit };
}

/** The whole kernel for one row: total, target = u x total, the chunk loop, the fallback. */
export function drawKernel(weights: number[], u: number, cfg: KernelCfg = STEPPER_CFG): KernelTrace {
  const { partials, warpSums, total } = kernelTotal(weights, cfg);
  const target = f32(f32(u) * total);
  const s = chunkedSearch(weights, target, cfg);
  return { cfg, n: weights.length, u, partials, warpSums, total, target, ...s };
}

// ---------------------------------------------------------------- the barrier hunt
export type Missing = 'none' | 'written' | 'read';
export const MISSING_LABEL: Record<Missing, string> = {
  get none() { return tr('both barriers in place', 'cả hai rào chắn còn nguyên'); },
  get written() { return tr('barrier A removed (after the total is written)', 'bỏ rào chắn A (sau khi ghi tổng)'); },
  get read() { return tr('barrier B removed (after the total is read)', 'bỏ rào chắn B (sau khi đọc tổng)'); },
};
export type HuntOrder = 'in-order' | 'reversed' | 'warp-ahead';
export const HUNT_ORDER_LABEL: Record<HuntOrder, string> = {
  get 'in-order'() { return tr('emulator, in order', 'trình giả lập, theo thứ tự'); },
  get reversed() { return tr('emulator, reversed', 'trình giả lập, đảo ngược'); },
  get 'warp-ahead'() { return tr('warp 0 runs ahead (a GPU may)', 'warp 0 chạy trước (GPU có thể làm vậy)'); },
};
type Sync = { kind: 'warp' | 'block' | 'end'; id: string };
type Cell = { v: number; what: string } | null;
interface Seg {
  /** What the segment does to scratch_s before it stops at `sync`. */
  run: (scratch: Cell[]) => void;
  sync: Sync;
}
export interface HuntRead {
  thread: number;
  warp: number;
  /** The value read from scratch_s[0] as the total (NaN if nothing was ever written there). */
  value: number;
  /** The value is the row's total. */
  ok: boolean;
  /** What had been written there: 'the total', or what overwrote it, or 'nothing yet'. */
  what: string;
}
export interface Hunt {
  order: HuntOrder;
  missing: Missing;
  reads: HuntRead[];
  /** Threads that read a value other than the total. */
  wrong: number;
  /** Threads that read something other than the total's write, whatever its value. */
  stale: number;
  /** One line per pass (emulator orders) or per warp turn (warp ahead). */
  log: string[];
  total: number;
}

/**
 * The kernel from its start to the first chunk's warp totals, for one block, under one
 * order, with barrier A (after thread 0 writes the total) or barrier B (after every thread
 * reads it) removed. Each thread is a list of segments, each ending at a barrier, a warp
 * call (a shuffle) or the end. The emulator's orders run every waiting-free thread to its
 * next stop, thread by thread, then release every rendezvous that is complete
 * (labs/common/include/pmpp/emu/emu_runtime.h, run_wave). "Warp 0 runs ahead" runs one
 * warp at a time in lockstep until it waits at a barrier, which a GPU is free to do.
 */
export function barrierHunt(weights: number[], u: number, cfg: KernelCfg, missing: Missing, order: HuntOrder): Hunt {
  void u;
  const { threads, warp } = cfg;
  const nW = Math.ceil(threads / warp);
  const { warpSums, total } = kernelTotal(weights, cfg);
  const chunk0 = scanChunk(weights, 0, 0, Infinity, cfg, false);
  const levels = Math.round(Math.log2(warp)); // shuffles in a tree, or in a Kogge-Stone scan, over one warp
  const reads: HuntRead[] = [];

  const prog = (t: number): Seg[] => {
    const w = Math.floor(t / warp);
    const l = t % warp;
    const segs: Seg[] = [];
    let pending: Seg['run'][] = [];
    const act = (f: Seg['run']) => pending.push(f);
    const stop = (kind: Sync['kind'], id: string) => {
      const fs = pending;
      pending = [];
      segs.push({ run: (s) => fs.forEach((f) => f(s)), sync: { kind, id } });
    };
    // 1. the total: a strided sum, a shuffle-down tree, lane 0 writes the warp's sum
    for (let i = 0; i < levels; i++) stop('warp', 'reduce' + i);
    act((s) => { if (l === 0) s[w] = { v: warpSums[w], what: tr(`warp ${w}'s partial sum`, `tổng riêng phần của warp ${w}`) }; });
    stop('block', 'after the warp sums'); // i18n-ignore: barrier id, compared; translated where the log prints it
    // warp 0 adds the warp sums with one more tree; thread 0 writes the total
    if (w === 0) {
      for (let i = 0; i < levels; i++) stop('warp', 'final' + i);
      act((s) => { if (t === 0) s[0] = { v: total, what: tr('the total', 'tổng') }; });
    }
    if (missing !== 'written') stop('block', 'A');
    act((s) => {
      const c = s[0];
      reads.push({ thread: t, warp: w, value: c ? c.v : NaN, ok: !!c && c.v === total, what: c ? c.what : tr('nothing yet', 'chưa có gì') });
    });
    if (missing !== 'read') stop('block', 'B');
    // 2. the first chunk: a Kogge-Stone scan by shuffle up, one more shuffle for the exclusive
    // value, then the last lane writes the warp's chunk total into the same array
    for (let i = 0; i < levels; i++) stop('warp', 'scan' + i);
    stop('warp', 'exclusive');
    act((s) => { if (l === warp - 1) s[w] = { v: chunk0.warpTotals[w], what: tr(`warp ${w}'s chunk-0 total`, `tổng chunk 0 của warp ${w}`) }; });
    stop('block', 'after the chunk totals'); // i18n-ignore: barrier id, compared; translated where the log prints it
    stop('end', 'end');
    return segs;
  };
  const progs = Array.from({ length: threads }, (_, t) => prog(t));
  const scratch: Cell[] = Array.from({ length: Math.max(nW, 1) }, () => null);
  const pc = new Array<number>(threads).fill(0);
  const waiting = new Array<Sync | null>(threads).fill(null);
  const log: string[] = [];
  const step = (t: number) => {
    const seg = progs[t][pc[t]++];
    seg.run(scratch);
    waiting[t] = seg.sync;
  };
  const lanesOf = (k: number) => Array.from({ length: warp }, (_, l) => k * warp + l).filter((t) => t < threads);
  const releaseWarp = (k: number) => {
    const ts = lanesOf(k);
    const s0 = waiting[ts[0]];
    if (s0 && s0.kind === 'warp' && ts.every((t) => waiting[t]?.kind === 'warp' && waiting[t]!.id === s0.id)) {
      ts.forEach((t) => (waiting[t] = null));
      return true;
    }
    return false;
  };
  const releaseBlock = () => {
    const s0 = waiting[0];
    if (s0 && s0.kind === 'block' && waiting.every((s) => s?.kind === 'block' && s.id === s0.id)) {
      log.push(tr(`barrier "${s0.id}" releases the block`, `rào chắn "${({ 'after the warp sums': 'sau các tổng warp', 'after the chunk totals': 'sau các tổng chunk' } as Record<string, string>)[s0.id] ?? s0.id}" giải phóng khối`));
      waiting.fill(null);
      return true;
    }
    return false;
  };
  const done = () => waiting.every((s) => s?.kind === 'end');
  const readsSoFar = () => reads.length;
  for (let guard = 0; !done() && guard < 500; guard++) {
    if (order === 'warp-ahead') {
      for (let k = 0; k < nW; k++) {
        const ts = lanesOf(k);
        const r0 = readsSoFar();
        do {
          ts.forEach((t) => { if (waiting[t] === null) step(t); });
        } while (releaseWarp(k));
        const r = readsSoFar() - r0;
        log.push(tr(`warp ${k} runs until it waits at "${waiting[ts[0]]?.id}"${r ? `; ${r} of its threads read scratch_s[0]` : ''}`, `warp ${k} chạy cho tới khi chờ ở "${({ 'after the warp sums': 'sau các tổng warp', 'after the chunk totals': 'sau các tổng chunk' } as Record<string, string>)[waiting[ts[0]]?.id ?? ''] ?? waiting[ts[0]]?.id}"${r ? `; ${r} luồng của nó đọc scratch_s[0]` : ''}`));
      }
      if (!releaseBlock() && !done()) { log.push(tr('stuck', 'kẹt')); break; }
    } else {
      const seq = Array.from({ length: threads }, (_, i) => (order === 'in-order' ? i : threads - 1 - i));
      const r0 = readsSoFar();
      const ran = seq.filter((t) => waiting[t] === null);
      ran.forEach(step);
      const r = readsSoFar() - r0;
      if (r) log.push(tr(`one pass: threads ${ran[0]} to ${ran[ran.length - 1]} each run to their next stop; ${r} of them read scratch_s[0]`, `một lượt: các luồng ${ran[0]} đến ${ran[ran.length - 1]} mỗi luồng chạy tới điểm dừng kế tiếp; ${r} luồng trong số đó đọc scratch_s[0]`));
      let any = releaseBlock();
      for (let k = 0; k < nW; k++) any = releaseWarp(k) || any;
      if (!any && !done()) { log.push(tr('stuck', 'kẹt')); break; }
    }
  }
  reads.sort((a, b) => a.thread - b.thread);
  return {
    order, missing, reads, total, log,
    wrong: reads.filter((r) => !r.ok).length,
    stale: reads.filter((r) => r.what !== 'the total').length,
  };
}

export interface HuntDraw {
  hunt: Hunt;
  /** Each thread's target: u times what it read as the total. */
  targets: number[];
  /** The token the kernel writes under this order with this barrier missing. */
  index: number;
  /** The token with both barriers in place. */
  correct: number;
  search: Search;
}
/** Follow a missing barrier to the token: barrierHunt's reads become per-thread targets for the chunk loop. */
export function huntDraw(weights: number[], u: number, cfg: KernelCfg, missing: Missing, order: HuntOrder): HuntDraw {
  const hunt = barrierHunt(weights, u, cfg, missing, order);
  const targets = hunt.reads.map((r) => f32(f32(u) * r.value));
  const search = chunkedSearch(weights, targets, cfg);
  return { hunt, targets, index: search.index, correct: drawKernel(weights, u, cfg).index, search };
}

/**
 * The stepper's rows (ours). Each is a sorted survivor row at a temperature with a u; 'hunt'
 * is chosen so that removing barrier A changes the token under the emulator's in-order run
 * at the stepper's display width (pinned in draw.test.ts).
 */
export const STEP_ROWS: { id: string; label: string; n: number; T: number; u: number }[] = [
  { id: 'short', get label() { return tr('40 survivors', '40 phần tử còn lại'); }, n: 40, T: 1, u: 0.6 },
  { id: 'long', get label() { return tr('200 survivors, T = 2.5', '200 phần tử còn lại, T = 2.5'); }, n: 200, T: 2.5, u: 0.85 },
  { id: 'hunt', get label() { return tr('120 survivors, T = 2', '120 phần tử còn lại, T = 2'); }, n: 120, T: 2, u: 0.7 },
];

// ---------------------------------------------------------------- near a boundary
export type CdfOrder = 'double' | 'f32-loop' | 'f32-kernel';
export const CDF_LABEL: Record<CdfOrder, string> = {
  get double() { return tr('double, left to right (the CPU reference)', 'double, từ trái sang phải (bản tham chiếu trên CPU)'); },
  get 'f32-loop'() { return tr('float32, left to right', 'float32, từ trái sang phải'); },
  get 'f32-kernel'() { return tr("float32, the kernel's order (thread, warp, block)", 'float32, thứ tự của kernel (luồng, warp, khối)'); },
};
/** Cumulative sums and the total in one order. The kernel's total comes from its own reduction. */
export function cdfIn(weights: number[], order: CdfOrder, cfg: KernelCfg = STEPPER_CFG): { cum: number[]; total: number } {
  if (order === 'double') {
    const cum = inclusiveIn(weights, 'loop', (v) => v);
    return { cum, total: cum[cum.length - 1] ?? 0 };
  }
  if (order === 'f32-loop') {
    const cum = inclusiveIn(weights, 'loop');
    return { cum, total: cum[cum.length - 1] ?? 0 };
  }
  const cum: number[] = [];
  let aggregate = 0;
  for (let base = 0; base < weights.length; base += cfg.threads * cfg.vec) {
    const s = scanChunk(weights, base, aggregate, Infinity, cfg, true);
    for (const l of s.lanes) l.cum.forEach((c, j) => { if (l.first + j < weights.length) cum[l.first + j] = c; });
    aggregate = f32(aggregate + s.sum);
  }
  return { cum, total: kernelTotal(weights, cfg).total };
}
/** The token one order picks for u: first cum > round(u x total), else the last positive weight. */
export function chooseIn(weights: number[], u: number, order: CdfOrder, cfg: KernelCfg = STEPPER_CFG): number {
  const { cum, total } = cdfIn(weights, order, cfg);
  const target = order === 'double' ? u * total : f32(f32(u) * total);
  const i = cum.findIndex((c) => c > target);
  return i >= 0 ? i : lastPositive(weights);
}
/** u at which the choice moves past boundary i (i = 0 .. n-2): cum_i / total, to within a float step. */
export function flipPoints(weights: number[], order: CdfOrder, cfg: KernelCfg = STEPPER_CFG): number[] {
  const { cum, total } = cdfIn(weights, order, cfg);
  return cum.slice(0, -1).map((c) => c / total);
}
export interface Band {
  boundary: number;
  lo: number;
  hi: number;
}
/** Bands of u where the given weight rows (one per order) choose differently, and their total width. */
export function disagreement(rows: { weights: number[]; order: CdfOrder }[], cfg: KernelCfg = STEPPER_CFG): { bands: Band[]; measure: number } {
  const flips = rows.map((r) => flipPoints(r.weights, r.order, cfg));
  const bands: Band[] = [];
  for (let i = 0; i < flips[0].length; i++) {
    const xs = flips.map((f) => f[i]);
    const lo = Math.min(...xs);
    const hi = Math.max(...xs);
    if (hi > lo) bands.push({ boundary: i, lo, hi });
  }
  return { bands, measure: bands.reduce((s, b) => s + (b.hi - b.lo), 0) };
}
/** A draw's margin: the distance of u x total from the nearest interior boundary, relative to the total (double). */
export function marginOf(weights: number[], u: number): number {
  const { cum, total } = cdfIn(weights, 'double');
  const t = u * total;
  let m = Infinity;
  for (let i = 0; i < cum.length - 1; i++) m = Math.min(m, Math.abs(t - cum[i]) / total);
  return m;
}
/** Upper bound on the share of uniform u within a relative margin m of one of the n - 1 interior boundaries. */
export const atRiskBound = (n: number, m: number) => Math.min(1, 2 * (n - 1) * m);

/** Multiply each weight by (1 + eps r_i), r_i in [-1, 1) from Philox (seed 7): a stand-in for another backend's exp. */
export function perturb(weights: number[], eps: number, seed = 7): number[] {
  return weights.map((w, i) => {
    const r = m5Uniform(philox([i, 0, 0, 0], [seed, 0])[0]) * 2 - 1;
    return f32(w * (1 + eps * r));
  });
}

// ---------------------------------------------------------------- rows
/**
 * A sorted row of survivor logits (ours): 12 - 1.6 ln(1 + i) plus a little Philox noise,
 * sorted in descending order, so that l_0 is the largest.
 */
export function sampleLogits(n: number, seed = 3): number[] {
  const out = Array.from({ length: n }, (_, i) => {
    const r = m5Uniform(philox([i, 0, 1, 0], [seed, 0])[0]);
    return f32(12 - 1.6 * Math.log(1 + i) + 0.8 * (r - 0.5));
  });
  return out.sort((a, b) => b - a);
}
/** Survivor weights w_i = exp((l_i - l_0) / T), rounded to float32 (temperature folded in, X1.2). */
export function weightsFrom(logits: number[], T: number): number[] {
  const l0 = logits[0];
  return logits.map((l) => f32(Math.exp((l - l0) / T)));
}

// ---------------------------------------------------------------- three samplers
export type Sampler = 'inverse-cdf' | 'race' | 'gumbel';
export const SAMPLER_LABEL: Record<Sampler, string> = {
  get 'inverse-cdf'() { return tr('inverse CDF (llama.cpp, FlashInfer, M5; MLX at batch 1)', 'CDF nghịch đảo (llama.cpp, FlashInfer, M5; MLX ở batch 1)'); },
  get race() { return tr('exponential race argmax(p / q) (torch.multinomial, one sample)', 'cuộc đua hàm mũ (exponential race) argmax(p / q) (torch.multinomial, một mẫu)'); },
  get gumbel() { return tr('Gumbel-max argmax(l + g) (MLX at batch 2 or more)', 'Gumbel-max argmax(l + g) (MLX ở batch 2 trở lên)'); },
};
export type RaceQ = 'one-minus' | 'log-u';
/**
 * n uniforms for one draw (ours): uniform i is word i mod 4 of Philox4x32-10 with counter
 * (draw, i / 4, stream, 0) and key (seed, 0), through M5's 24-bit conversion. Uniform 0 is
 * M5's u for (seed, stream, step = draw).
 */
export function uniformsFor(seed: number, stream: number, draw: number, n: number): number[] {
  const out: number[] = [];
  let words = philox([draw >>> 0, 0, stream >>> 0, 0], [seed >>> 0, 0]);
  for (let i = 0; i < n; i++) {
    if (i > 0 && i % 4 === 0) words = philox([draw >>> 0, i / 4, stream >>> 0, 0], [seed >>> 0, 0]);
    out.push(m5Uniform(words[i % 4]));
  }
  return out;
}
const argmax = (xs: number[]) => xs.reduce((b, x, i) => (x > xs[b] ? i : b), 0);
/** One draw by one method from the same uniforms (double precision). */
export function sampleWith(probs: number[], us: number[], method: Sampler, q: RaceQ = 'one-minus'): number {
  if (method === 'inverse-cdf') return inverseCdf(probs, us[0]).index;
  if (method === 'race') return argmax(probs.map((p, i) => p / (q === 'one-minus' ? -Math.log(1 - us[i]) : -Math.log(us[i]))));
  return argmax(probs.map((p, i) => Math.log(p) - Math.log(-Math.log(us[i]))));
}
export interface Comparison {
  draws: number;
  counts: Record<Sampler, number[]>;
  /** Share of draws on which two methods chose the same token. */
  agree: { invRace: number; invGumbel: number; raceGumbel: number };
}
export function compareSamplers(probs: number[], draws: number, seed = 5, stream = 0, q: RaceQ = 'one-minus'): Comparison {
  const n = probs.length;
  const counts: Record<Sampler, number[]> = { 'inverse-cdf': new Array(n).fill(0), race: new Array(n).fill(0), gumbel: new Array(n).fill(0) };
  let ir = 0, ig = 0, rg = 0;
  for (let d = 0; d < draws; d++) {
    const us = uniformsFor(seed, stream, d, n);
    const a = sampleWith(probs, us, 'inverse-cdf');
    const b = sampleWith(probs, us, 'race', q);
    const c = sampleWith(probs, us, 'gumbel');
    counts['inverse-cdf'][a]++;
    counts.race[b]++;
    counts.gumbel[c]++;
    if (a === b) ir++;
    if (a === c) ig++;
    if (b === c) rg++;
  }
  return { draws, counts, agree: { invRace: ir / draws, invGumbel: ig / draws, raceGumbel: rg / draws } };
}
/** The row the three-sampler explorer uses: six tokens. */
export const SAMPLER_PROBS = [0.4, 0.25, 0.15, 0.1, 0.06, 0.04];

// ---------------------------------------------------------------- worked numbers the lesson states
/** Prediction x1/X1.8/p-draw: four equal weights, u = 0.6, the strict rule. */
export const P_DRAW = { weights: [1, 1, 1, 1], u: 0.6 };
/** Question x1/X1.8/q-ge-gt: probabilities in vocabulary order, u = 0. */
export const GE_GT = { weights: [0, 0.5, 0.5], u: 0 };
/** Question x1/X1.8/q-risk: 40 survivors, a margin of 2^-20. */
export const RISK = { n: 40, margin: 2 ** -20 };

// ---------------------------------------------------------------- the find-u scene
/** The scene's block: 8 threads, a display warp of 4, 2 survivors per thread, so chunks of 16. */
export const SCENE_CFG: KernelCfg = { threads: 8, warp: 4, vec: 2 };
/** The scene's row: 40 sorted survivors at T = 3 (sampleLogits seed 3), flat enough to need a second chunk. */
export const SCENE_T = 3;
export const SCENE_WEIGHTS = weightsFrom(sampleLogits(40, 3), SCENE_T);
/** The scene's u: chunk 0 is skipped and chunk 1 crosses (pinned in draw.test.ts). */
export const SCENE_U = 0.8;
/** The scene's draw through the kernel. */
export const sceneDraw = () => drawKernel(SCENE_WEIGHTS, SCENE_U, SCENE_CFG);

export interface BoundaryCase {
  u: number;
  boundary: number;
  /** Per order: the inclusive sum at the boundary, the target, the token chosen. */
  loop: { cum: number; target: number; index: number };
  kernel: { cum: number; target: number; index: number };
}
/**
 * The first u (a multiple of 2^-24, so one M5 can produce) at which the scene row's two
 * float32 orders pick neighbouring tokens: the left-to-right sums and the kernel's.
 */
export function sceneBoundary(weights = SCENE_WEIGHTS, cfg = SCENE_CFG): BoundaryCase | null {
  const { bands } = disagreement([{ weights, order: 'f32-loop' }, { weights, order: 'f32-kernel' }], cfg);
  const L = cdfIn(weights, 'f32-loop', cfg);
  const K = cdfIn(weights, 'f32-kernel', cfg);
  for (const b of bands) {
    for (let q = Math.floor(b.lo * 2 ** 24) - 2; q <= Math.ceil(b.hi * 2 ** 24) + 2; q++) {
      const u = q / 2 ** 24;
      const a = chooseIn(weights, u, 'f32-loop', cfg);
      const k = chooseIn(weights, u, 'f32-kernel', cfg);
      if (a !== k) {
        const i = b.boundary;
        return {
          u,
          boundary: i,
          loop: { cum: L.cum[i], target: f32(f32(u) * L.total), index: a },
          kernel: { cum: K.cum[i], target: f32(f32(u) * K.total), index: k },
        };
      }
    }
  }
  return null;
}

/** M5's largest u, 1 - 2^-24. */
export const TOP_U = 1 - 2 ** -24;
/**
 * The scene's fallback row: 8 survivors (sampleLogits(8, 2), T = 1) where, at M5's largest u,
 * the kernel's last scanned sum ends at or below u x total, because the total (a strided sum and
 * two warp trees) and the scan add the same weights in different orders (pinned in draw.test.ts).
 */
export const FALLBACK_WEIGHTS = weightsFrom(sampleLogits(8, 2), 1);
export const fallbackDraw = () => drawKernel(FALLBACK_WEIGHTS, TOP_U, SCENE_CFG);
