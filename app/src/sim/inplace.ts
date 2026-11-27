// In-place filtering (book §12.7, p. 300) and the related one-directional data movements
// (§12.8, p. 301). Used by chapter 12's sections 12.7 and 12.8 and by ScanChainScene.
//
// Four models, each small enough to check by hand:
//   overlap()          which slot each write lands on, whose key lived there, and which
//                      block must have read it first (the book's Fig. 12.10 example);
//   dependencePath()   the path read(i) -> scan(i) -> ... -> scan(j) -> write(j) of Fig. 12.11;
//   orderingVerdict()  the ordering puzzle: is block i's read ordered before block j's write
//                      under the C++ happens-before rules, given barriers and memory orders?
//   chainTimeline()    blocks that start in some order, with a limited number of resident
//                      slots, numbered by a counter or by blockIdx: progress or a starving spin;
// plus the related patterns of §12.8 (dedupFlags, moveByIndex, blockDependencies).

import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- shared helpers

/** Exclusive prefix sum: out[i] = xs[0] + ... + xs[i-1], out[0] = 0. */
export function exclusiveScan(xs: number[]): number[] {
  const out: number[] = [];
  let s = 0;
  for (const x of xs) {
    out.push(s);
    s += x;
  }
  return out;
}

/** The block whose range of `perBlock` consecutive slots holds `slot`. */
export const blockOf = (slot: number, perBlock: number) => Math.floor(slot / perBlock);
export const blockCount = (n: number, perBlock: number) => Math.ceil(n / perBlock);

/** The keep pattern of the book's Figs. 12.1, 12.7, 12.9 and 12.10: k1, k3, k4, k6, k8, k10, k11, k14 of k0..k15. */
export const FIG_KEPT = [1, 3, 4, 6, 8, 10, 11, 14];
export const keepOf = (n: number, kept: number[]) => Array.from({ length: n }, (_, i) => kept.includes(i));
export const FIG_KEEP: boolean[] = keepOf(16, FIG_KEPT);

// ---------------------------------------------------------------- §12.7: what each write overwrites

/** remove: a stable filter, keys move to lower slots. insert: the reverse (§12.8), keys move back out. */
export type Direction = 'remove' | 'insert';
/**
 * own: the key stays in its own slot.
 * same: the slot belongs to the writer's own block: a barrier between "all read" and "all write" orders it.
 * earlier: the slot is read by an earlier block: the chain of scans must order it (removal).
 * later: the slot is read by a later block: the chain must run from the last block to the first (insertion).
 * fresh: no key lived in the slot, so there is nothing to protect.
 */
export type WriteKind = 'own' | 'same' | 'earlier' | 'later' | 'fresh';
export const KIND_LABEL: Record<WriteKind, string> = {
  get own() { return tr('own slot', 'ô của chính nó'); },
  get same() { return tr('same block: barrier', 'cùng khối: rào chắn'); },
  get earlier() { return tr('earlier block: scan chain', 'khối trước: chuỗi scan'); },
  get later() { return tr('later block: chain runs backwards', 'khối sau: chuỗi chạy ngược'); },
  get fresh() { return tr('empty slot: nothing to protect', 'ô trống: không có gì cần bảo vệ'); },
};

export interface Write {
  /** The key's label: k<key>, its index in the original (unfiltered) list. */
  key: number;
  /** Slot it is read from, and slot it is written to. */
  from: number;
  to: number;
  /** The block whose range holds `from`: it reads the key and writes it. */
  writer: number;
  /** Label of the key that lived in slot `to` before any write, or null if none did. */
  victim: number | null;
  /** The block whose range holds `to` and which therefore reads the victim; null if none. */
  reader: number | null;
  kind: WriteKind;
}

export interface Overlap {
  n: number;
  perBlock: number;
  blocks: number;
  direction: Direction;
  /** Kept keys. */
  count: number;
  /** Labels in each slot before and after the move (null: nothing meaningful there). */
  before: (number | null)[];
  after: (number | null)[];
  writes: Write[];
  counts: Record<WriteKind, number>;
  /** Distinct cross-block requirements: `reader` must finish reading before `writer` writes. */
  pairs: { reader: number; writer: number; writes: number }[];
}

/**
 * Overlay the output slots on the input slots for an in-place stable filter (`remove`) or
 * its reverse (`insert`), with blocks of `perBlock` consecutive slots.
 */
export function overlap(keep: boolean[], perBlock: number, direction: Direction = 'remove'): Overlap {
  const n = keep.length;
  const offset = exclusiveScan(keep.map((k) => (k ? 1 : 0)));
  const keptIdx = keep.flatMap((k, i) => (k ? [i] : []));
  const count = keptIdx.length;
  const writes: Write[] = [];
  let before: (number | null)[];
  let after: (number | null)[];
  if (direction === 'remove') {
    before = Array.from({ length: n }, (_, i) => i);
    after = Array.from({ length: n }, (_, s) => (s < count ? keptIdx[s] : null));
    for (const i of keptIdx) {
      const to = offset[i];
      writes.push(classify({ key: i, from: i, to, victim: to, perBlock }));
    }
  } else {
    before = Array.from({ length: n }, (_, s) => (s < count ? keptIdx[s] : null));
    after = Array.from({ length: n }, (_, i) => (keep[i] ? i : null));
    for (const i of keptIdx) {
      const from = offset[i];
      writes.push(classify({ key: i, from, to: i, victim: before[i], perBlock }));
    }
  }
  const counts: Record<WriteKind, number> = { own: 0, same: 0, earlier: 0, later: 0, fresh: 0 };
  const pairMap = new Map<string, { reader: number; writer: number; writes: number }>();
  for (const w of writes) {
    counts[w.kind]++;
    if ((w.kind === 'earlier' || w.kind === 'later') && w.reader != null) {
      const k = `${w.reader}>${w.writer}`;
      const e = pairMap.get(k) ?? { reader: w.reader, writer: w.writer, writes: 0 };
      e.writes++;
      pairMap.set(k, e);
    }
  }
  const pairs = [...pairMap.values()].sort((a, b) => a.writer - b.writer || a.reader - b.reader);
  return { n, perBlock, blocks: blockCount(n, perBlock), direction, count, before, after, writes, counts, pairs };
}

function classify(w: { key: number; from: number; to: number; victim: number | null; perBlock: number }): Write {
  const writer = blockOf(w.from, w.perBlock);
  if (w.from === w.to) return { key: w.key, from: w.from, to: w.to, writer, victim: w.victim, reader: writer, kind: 'own' };
  if (w.victim == null) return { key: w.key, from: w.from, to: w.to, writer, victim: null, reader: null, kind: 'fresh' };
  const reader = blockOf(w.to, w.perBlock);
  const kind: WriteKind = reader === writer ? 'same' : reader < writer ? 'earlier' : 'later';
  return { key: w.key, from: w.from, to: w.to, writer, victim: w.victim, reader, kind };
}

// ---------------------------------------------------------------- Fig. 12.11: the dependence path

export type PathNode = { op: 'read' | 'scan' | 'write' | 'barrier'; block: number };

/**
 * The chain of events that orders block `reader`'s read before block `writer`'s write, with
 * the scans chained first to last (`remove`) or last to first (`insert`). Null when the
 * chain does not run that way, so nothing orders the pair.
 */
export function dependencePath(reader: number, writer: number, direction: Direction = 'remove'): PathNode[] | null {
  if (reader === writer)
    return [{ op: 'read', block: reader }, { op: 'barrier', block: reader }, { op: 'write', block: writer }];
  const step = direction === 'remove' ? 1 : -1;
  if ((writer - reader) * step < 0) return null;
  const out: PathNode[] = [{ op: 'read', block: reader }];
  for (let b = reader; b !== writer + step; b += step) out.push({ op: 'scan', block: b });
  out.push({ op: 'write', block: writer });
  return out;
}

// ---------------------------------------------------------------- the ordering puzzle

/**
 * Block i reads its keys, then (optionally) a barrier, then its leader stores the flag.
 * Block j's leader loads the flag (it spins until it sees it set), then (optionally) a
 * barrier, then block j's threads write. The leader is one thread per block (the book's
 * Fig. 11.17 uses the block's last thread).
 */
export interface PuzzleConfig {
  barrierI: boolean;
  publish: 'relaxed' | 'release';
  observe: 'relaxed' | 'acquire';
  barrierJ: boolean;
}
export const BOOK_PUZZLE: PuzzleConfig = { barrierI: true, publish: 'release', observe: 'acquire', barrierJ: true };
/** Which thread reads slot s in block i, or writes it in block j. */
export type Who = 'leader' | 'other';
export interface Link {
  from: string;
  to: string;
  /** The rule that orders `from` before `to`, or why nothing does. */
  why: string;
  ok: boolean;
}
export interface Verdict {
  ordered: boolean;
  links: Link[];
  /** The first missing link, in words; null when ordered. */
  missing: string | null;
}

/**
 * Is block i's read of slot s ordered before block j's write of slot s? Happens-before in
 * the C++ model: sequenced-before inside a thread, a barrier between the threads of one
 * block, and a release store read by an acquire load between blocks. Scopes are ignored.
 */
export function orderingVerdict(cfg: PuzzleConfig, reader: Who, writer: Who): Verdict {
  const links: Link[] = [];
  const readName = reader === 'leader' ? tr("block i's leader reads slot s", 'luồng dẫn đầu của khối i đọc ô s') : tr('another thread of block i reads slot s', 'một luồng khác của khối i đọc ô s');
  if (reader === 'leader')
    links.push({ from: readName, to: tr("block i's leader stores the flag", 'luồng dẫn đầu của khối i store cờ'), why: tr('sequenced before: same thread, program order', 'sequenced before: cùng một luồng, theo thứ tự chương trình'), ok: true });
  else
    links.push(
      cfg.barrierI
        ? { from: readName, to: tr("block i's leader stores the flag", 'luồng dẫn đầu của khối i store cờ'), why: tr('the barrier in block i: every read before it happens before the store after it', 'rào chắn trong khối i: mọi lần đọc trước nó happens-before lần store sau nó'), ok: true }
        : { from: readName, to: tr("block i's leader stores the flag", 'luồng dẫn đầu của khối i store cờ'), why: tr('no barrier in block i: nothing orders another thread\'s read before the leader\'s store', 'khối i không có rào chắn: không gì xếp lần đọc của luồng khác trước lần store của luồng dẫn đầu'), ok: false },
    );
  const sync = cfg.publish === 'release' && cfg.observe === 'acquire';
  links.push({
    from: tr("block i's leader stores the flag", 'luồng dẫn đầu của khối i store cờ'),
    to: tr("block j's leader loads the flag and sees it set", 'luồng dẫn đầu của khối j load cờ và thấy cờ đã được đặt'),
    why: sync
      ? tr('synchronizes with: a release store read by an acquire load', 'synchronizes with: một store release được một load acquire đọc')
      : cfg.publish === 'relaxed' && cfg.observe === 'relaxed'
        ? tr('no synchronization: both the store and the load are relaxed', 'không đồng bộ hóa: cả store lẫn load đều relaxed')
        : cfg.publish === 'relaxed'
          ? tr('no synchronization: a relaxed store releases nothing, even to an acquire load', 'không đồng bộ hóa: store relaxed không release gì cả, kể cả cho một load acquire')
          : tr('no synchronization: a relaxed load acquires nothing, even from a release store', 'không đồng bộ hóa: load relaxed không acquire gì cả, kể cả từ một store release'),
    ok: sync,
  });
  const writeName = writer === 'leader' ? tr("block j's leader writes slot s", 'luồng dẫn đầu của khối j ghi ô s') : tr('another thread of block j writes slot s', 'một luồng khác của khối j ghi ô s');
  if (writer === 'leader')
    links.push({ from: tr("block j's leader loads the flag and sees it set", 'luồng dẫn đầu của khối j load cờ và thấy cờ đã được đặt'), to: writeName, why: tr('sequenced before: same thread, program order', 'sequenced before: cùng một luồng, theo thứ tự chương trình'), ok: true });
  else
    links.push(
      cfg.barrierJ
        ? { from: tr("block j's leader loads the flag and sees it set", 'luồng dẫn đầu của khối j load cờ và thấy cờ đã được đặt'), to: writeName, why: tr('the barrier in block j: the load before it happens before every write after it', 'rào chắn trong khối j: lần load trước nó happens-before mọi lần ghi sau nó'), ok: true }
        : { from: tr("block j's leader loads the flag and sees it set", 'luồng dẫn đầu của khối j load cờ và thấy cờ đã được đặt'), to: writeName, why: tr('no barrier in block j: the other threads may write before the leader has seen the flag', 'khối j không có rào chắn: các luồng khác có thể ghi trước khi luồng dẫn đầu thấy cờ'), ok: false },
    );
  const bad = links.find((l) => !l.ok);
  return { ordered: !bad, links, missing: bad ? bad.why : null };
}

/** All four reader/writer combinations. The filter is safe only if every one is ordered. */
export function orderingTable(cfg: PuzzleConfig): { reader: Who; writer: Who; verdict: Verdict }[] {
  const out: { reader: Who; writer: Who; verdict: Verdict }[] = [];
  for (const reader of ['leader', 'other'] as Who[])
    for (const writer of ['leader', 'other'] as Who[]) out.push({ reader, writer, verdict: orderingVerdict(cfg, reader, writer) });
  return out;
}

// ---------------------------------------------------------------- the scan chain over time

export type Numbering = 'dynamic' | 'blockIdx';
/**
 * What a block does in one step. queued: not started (no resident slot yet). index: takes
 * its number. read: reads the keys of its range. scan: counts its kept keys. wait: spins on
 * the previous number's flag (acquire). publish: sees the flag and stores its own (release).
 * write: writes its kept keys at their final slots. done: finished, slot free.
 */
export type ChainPhase = 'queued' | 'index' | 'read' | 'scan' | 'wait' | 'publish' | 'write' | 'done';
export interface ChainBlock {
  /** blockIdx.x */
  hw: number;
  /** The number it works under (decides its range of keys): from the counter, or blockIdx. */
  num: number | null;
  slot: number | null;
  phase: ChainPhase;
  /** For a waiting block: the number it waits on. */
  waitsOn: number | null;
}
export interface ChainFrame {
  step: number;
  blocks: ChainBlock[];
  counter: number;
  /** Step at which number b published its flag, or null. */
  flagAt: (number | null)[];
  /** Labels in the array after this step. */
  array: (number | null)[];
  /** What happened in this step, in words. */
  events: string[];
}
export interface ChainRun {
  frames: ChainFrame[];
  outcome: 'finished' | 'starved';
  /** Writes that landed on a slot whose key had not been read yet. */
  hazards: number;
  /** The array is the stable filter of the input. */
  correct: boolean;
  /** For a starved run: who waits on whom, and whether that one ever started. */
  stuck: { num: number; waitsOn: number; started: boolean }[];
}

export interface ChainOptions {
  keep: boolean[];
  perBlock: number;
  /** How many blocks can be resident at once. */
  slots: number;
  /** blockIdx values in the order the hardware starts them. */
  order: number[];
  numbering: Numbering;
  /** false: no flags at all; each block's offset is known in advance and it writes right after its scan. */
  chain?: boolean;
  maxSteps?: number;
}

/** Step-by-step run of a single-pass in-place stable filter. Deterministic. */
export function chainTimeline(o: ChainOptions): ChainRun {
  const chain = o.chain ?? true;
  const n = o.keep.length;
  const nb = blockCount(n, o.perBlock);
  if (o.order.length !== nb || [...o.order].sort((a, b) => a - b).some((v, i) => v !== i))
    throw new Error('order must be a permutation of the block indices');
  const offset = exclusiveScan(o.keep.map((k) => (k ? 1 : 0)));
  const array: (number | null)[] = Array.from({ length: n }, (_, i) => i);
  const readDone: boolean[] = Array(n).fill(false);
  const held = new Map<number, (number | null)[]>(); // num -> values it read
  const flagAt: (number | null)[] = Array(nb).fill(null);
  const blocks: ChainBlock[] = Array.from({ length: nb }, (_, hw) => ({ hw, num: null, slot: null, phase: 'queued', waitsOn: null }));
  const queue = [...o.order];
  let counter = 0;
  let hazards = 0;
  const frames: ChainFrame[] = [];
  const snap = (step: number, events: string[]) =>
    frames.push({ step, blocks: blocks.map((b) => ({ ...b })), counter, flagAt: [...flagAt], array: [...array], events });
  snap(0, [tr('nothing has started', 'chưa có gì bắt đầu')]);
  const range = (num: number) => {
    const out: number[] = [];
    for (let s = num * o.perBlock; s < Math.min(n, (num + 1) * o.perBlock); s++) out.push(s);
    return out;
  };
  const max = o.maxSteps ?? 200;
  let outcome: ChainRun['outcome'] = 'finished';
  for (let step = 1; step <= max; step++) {
    const events: string[] = [];
    const prev = JSON.stringify(blocks);
    // 1. blocks that wrote last step are done; their slots free up
    const free: number[] = [];
    for (const b of blocks)
      if (b.phase === 'write') {
        b.phase = 'done';
        free.push(b.slot!);
        events.push(tr(`number ${b.num} is done`, `số ${b.num} xong`));
        b.slot = null;
      }
    const used = new Set(blocks.filter((b) => b.slot != null).map((b) => b.slot!));
    for (let s = 0; s < o.slots; s++) if (!used.has(s) && !free.includes(s)) free.push(s);
    free.sort((a, b) => a - b);
    // 2. resident blocks advance on what was visible at the end of the previous step
    for (const b of blocks) {
      if (b.slot == null || b.phase === 'queued' || b.phase === 'done') continue;
      const num = b.num!;
      switch (b.phase) {
        case 'index':
          b.phase = 'read';
          held.set(num, range(num).map((s) => array[s]));
          for (const s of range(num)) readDone[s] = true;
          events.push(tr(`number ${num} reads slots ${range(num)[0]} to ${range(num)[range(num).length - 1]}`, `số ${num} đọc các ô ${range(num)[0]} đến ${range(num)[range(num).length - 1]}`));
          break;
        case 'read':
          b.phase = 'scan';
          events.push(tr(`number ${num} counts its kept keys`, `số ${num} đếm các khóa nó giữ lại`));
          break;
        case 'scan':
        case 'wait': {
          if (!chain) {
            doWrite(b, events);
            break;
          }
          const ready = num === 0 || (flagAt[num - 1] != null && flagAt[num - 1]! < step);
          if (ready) {
            b.phase = 'publish';
            b.waitsOn = null;
            flagAt[num] = step;
            events.push(num === 0 ? tr('number 0 publishes its total (release)', 'số 0 công bố tổng của nó (release)') : tr(`number ${num} sees number ${num - 1}'s flag (acquire) and publishes its own (release)`, `số ${num} thấy cờ của số ${num - 1} (acquire) và công bố cờ của nó (release)`));
          } else {
            b.phase = 'wait';
            b.waitsOn = num - 1;
            events.push(tr(`number ${num} waits for number ${num - 1}'s flag`, `số ${num} chờ cờ của số ${num - 1}`));
          }
          break;
        }
        case 'publish':
          doWrite(b, events);
          break;
      }
    }
    // 3. queued blocks take free slots, in the hardware's order
    for (const s of free) {
      const hw = queue.shift();
      if (hw == null) break;
      const b = blocks[hw];
      b.slot = s;
      b.phase = 'index';
      b.num = o.numbering === 'dynamic' ? counter++ : hw;
      events.push(o.numbering === 'dynamic' ? tr(`blockIdx ${hw} starts and takes number ${b.num}`, `blockIdx ${hw} bắt đầu và lấy số ${b.num}`) : tr(`blockIdx ${hw} starts as number ${b.num}`, `blockIdx ${hw} bắt đầu với số ${b.num}`));
    }
    if (blocks.every((b) => b.phase === 'done')) {
      snap(step, events);
      outcome = 'finished';
      break;
    }
    if (JSON.stringify(blocks) === prev) {
      outcome = 'starved';
      break;
    }
    snap(step, events);
    if (step === max) outcome = 'starved';
  }
  function doWrite(b: ChainBlock, events: string[]) {
    const num = b.num!;
    b.phase = 'write';
    const vals = held.get(num)!;
    // Slots past the output keep whatever was there: the filter defines only the first `count`.
    range(num).forEach((s, k) => {
      if (!o.keep[s]) return;
      const to = offset[s];
      if (!readDone[to]) hazards++;
      array[to] = vals[k];
    });
    events.push(tr(`number ${num} writes its kept keys`, `số ${num} ghi các khóa nó giữ lại`));
  }
  const count = offset[n - 1] + (o.keep[n - 1] ? 1 : 0);
  const last = frames[frames.length - 1];
  const expected = o.keep.flatMap((k, i) => (k ? [i] : []));
  const correct = outcome === 'finished' && expected.every((v, s) => last.array[s] === v) && count === expected.length;
  const stuck =
    outcome === 'starved'
      ? last.blocks
          .filter((b) => b.phase === 'wait')
          .map((b) => ({ num: b.num!, waitsOn: b.waitsOn!, started: last.blocks.some((x) => x.num === b.waitsOn && x.phase !== 'queued') }))
      : [];
  return { frames, outcome, hazards, correct, stuck };
}

/** The scene's problem: 12 keys of Fig. 12.10's pattern, three blocks of four, two resident slots, started as blockIdx 2, 1, 0. */
export const SCENE = {
  keep: keepOf(12, FIG_KEPT),
  perBlock: 4,
  slots: 2,
  order: [2, 1, 0],
};
export function sceneRuns() {
  return {
    overlap: overlap(SCENE.keep, SCENE.perBlock),
    dynamic: chainTimeline({ ...SCENE, numbering: 'dynamic' }),
    blockIdx: chainTimeline({ ...SCENE, numbering: 'blockIdx' }),
  };
}

// ---------------------------------------------------------------- §12.8: related patterns

/** Keep a key when it differs from the one before it; the first key is always kept. Then compact. */
export function dedupFlags(sorted: number[]): { keep: number[]; offset: number[]; out: number[]; count: number } {
  const keep = sorted.map((v, i) => (i === 0 || v !== sorted[i - 1] ? 1 : 0));
  const offset = exclusiveScan(keep);
  const out = sorted.filter((_, i) => keep[i] === 1);
  return { keep, offset, out, count: out.length };
}

export type MatrixPattern =
  | { kind: 'removeRows'; rows: number; cols: number; remove: number[] }
  | { kind: 'removeCols'; rows: number; cols: number; remove: number[] }
  | { kind: 'padCols'; rows: number; cols: number; pad: number };

export interface Move {
  src: number;
  /** Null: the element is removed. */
  dest: number | null;
  r: number;
  c: number;
  /** The destination worked out from the index, with its numbers. */
  formula: string;
}
export interface Moved {
  moves: Move[];
  newRows: number;
  newCols: number;
  /** Elements in the result. */
  size: number;
}

const below = (xs: number[], v: number) => xs.filter((x) => x < v).length;

/** Each element's destination for removing rows, removing columns or padding columns of a row-major matrix. No scan. */
export function moveByIndex(p: MatrixPattern): Moved {
  const moves: Move[] = [];
  const R = p.rows;
  const C = p.cols;
  if (p.kind === 'removeRows') {
    const rm = [...new Set(p.remove)].filter((r) => r >= 0 && r < R);
    for (let r = 0; r < R; r++)
      for (let c = 0; c < C; c++) {
        const src = r * C + c;
        if (rm.includes(r)) moves.push({ src, dest: null, r, c, formula: tr(`row ${r} is removed`, `hàng ${r} bị xóa`) });
        else {
          const nr = r - below(rm, r);
          moves.push({ src, dest: nr * C + c, r, c, formula: `(${r} − ${below(rm, r)}) × ${C} + ${c} = ${nr * C + c}` });
        }
      }
    return { moves, newRows: R - rm.length, newCols: C, size: (R - rm.length) * C };
  }
  if (p.kind === 'removeCols') {
    const rm = [...new Set(p.remove)].filter((c) => c >= 0 && c < C);
    const W = C - rm.length;
    for (let r = 0; r < R; r++)
      for (let c = 0; c < C; c++) {
        const src = r * C + c;
        if (rm.includes(c)) moves.push({ src, dest: null, r, c, formula: tr(`column ${c} is removed`, `cột ${c} bị xóa`) });
        else {
          const nc = c - below(rm, c);
          moves.push({ src, dest: r * W + nc, r, c, formula: `${r} × ${W} + (${c} − ${below(rm, c)}) = ${r * W + nc}` });
        }
      }
    return { moves, newRows: R, newCols: W, size: R * W };
  }
  const W = C + p.pad;
  for (let r = 0; r < R; r++)
    for (let c = 0; c < C; c++) {
      const src = r * C + c;
      moves.push({ src, dest: r * W + c, r, c, formula: `${r} × ${W} + ${c} = ${r * W + c}` });
    }
  return { moves, newRows: R, newCols: W, size: R * W };
}

/** Moves for a deduplicated list (positions come from a scan of the keep flags). */
export function dedupMoves(sorted: number[]): Move[] {
  const d = dedupFlags(sorted);
  return sorted.map((_, i) => ({
    src: i,
    dest: d.keep[i] ? d.offset[i] : null,
    r: 0,
    c: i,
    formula: d.keep[i] ? tr(`offset[${i}] = ${d.offset[i]} (exclusive scan of the keep flags)`, `offset[${i}] = ${d.offset[i]} (scan loại trừ của các cờ giữ lại)`) : tr(`same as key ${i - 1}: dropped`, `giống khóa ${i - 1}: bị bỏ`),
  }));
}

export type ChainDirection = 'forward' | 'backward' | 'none' | 'mixed';
export interface Dependencies {
  /** Block `reader` must finish reading before block `writer` writes. */
  edges: { reader: number; writer: number; writes: number }[];
  /** Writes that land on a slot read by the writer's own block (a barrier orders them). */
  within: number;
  /** Writes into memory past the original array: nothing lived there. */
  fresh: number;
  /** forward: earlier blocks read before later ones write (first to last); backward: the reverse. */
  direction: ChainDirection;
  blocks: number;
}

/**
 * In place, which blocks must read before which others write. Blocks own consecutive
 * ranges of `perBlock` source slots (for a matrix, `rowsPerBlock × cols`).
 */
export function blockDependencies(moves: Move[], srcSize: number, perBlock: number): Dependencies {
  const map = new Map<string, { reader: number; writer: number; writes: number }>();
  let within = 0;
  let fresh = 0;
  for (const m of moves) {
    if (m.dest == null || m.dest === m.src) continue;
    if (m.dest >= srcSize) {
      fresh++;
      continue;
    }
    const writer = blockOf(m.src, perBlock);
    const reader = blockOf(m.dest, perBlock);
    if (reader === writer) {
      within++;
      continue;
    }
    const k = `${reader}>${writer}`;
    const e = map.get(k) ?? { reader, writer, writes: 0 };
    e.writes++;
    map.set(k, e);
  }
  const edges = [...map.values()].sort((a, b) => a.writer - b.writer || a.reader - b.reader);
  const fwd = edges.some((e) => e.reader < e.writer);
  const bwd = edges.some((e) => e.reader > e.writer);
  const direction: ChainDirection = fwd && bwd ? 'mixed' : fwd ? 'forward' : bwd ? 'backward' : 'none';
  return { edges, within, fresh, direction, blocks: blockCount(srcSize, perBlock) };
}
