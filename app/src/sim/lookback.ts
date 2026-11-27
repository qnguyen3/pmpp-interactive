// Blocks waiting for blocks (chapter 11, section 11.9). One model for the "who waits for
// whom" simulator, the "make the flag safe" classification, the "Waiting for the block
// before you" scene and the section's questions.
//
// A discrete-event model of an inter-block scan: G blocks share R resident slots. Blocks
// start in a given order whenever a slot is free. Each block scans its own segment, then
// needs the total of every segment before it, then adds and stores. Time runs in rounds.
//
// From the book (notes/ch11.md, §11.9, pp. 276-282):
//   - (a) grid-wide barrier: every block must be running at once (a cooperative launch),
//     two barriers, one block scans the block sums in between (pp. 276-277);
//   - (b) single lookback: block b waits for block b - 1's scanned sum, adds its own and
//     publishes; only earlier blocks are waited for, a chain through all blocks (pp. 277, 279);
//   - decoupled lookback: a waiting block adds the plain sums of the blocks in between until
//     it reaches a block whose scanned sum is ready; less waiting, some additions repeated
//     (pp. 279-280); at its extreme every block adds all predecessors, Fig. 11.16(c) (p. 280);
//   - the block index is claimed from an atomic counter as a block starts, so a block never
//     waits for one that has not been scheduled (p. 280; "deadlocks", p. 285);
//   - the flag protocol of Fig. 11.17: device-scope atomics, release on the write, acquire on
//     the read (pp. 278-279).
// From the facts (notes/today/ch11-facts.md):
//   - A6: CUDA documents parallel forward progress: a thread that has executed a step keeps
//     making progress, so waiting on a block that claimed its index terminates; waiting on
//     blockIdx.x - 1 has no such backing. That is the rule "started".
//   - C5: Metal documents no forward progress between threadgroups; an Apple engineer named
//     fair scheduling of concurrent threadgroups as an invalid assumption. The rule "none" is
//     one adversarial schedule that the absence of a guarantee allows: whenever a resident
//     block is spinning, only spinning blocks are run. It is not a model of any GPU.
//   - A5: NVIDIA's message-passing example; the variant with block scope on the store is
//     marked "UB: data race" (the classification puts block scope on both sides, no better).
//   - infra A4 / ch04 C3: a grid barrier needs a cooperative launch with every block resident
//     (CUDA, HIP); Metal has none.
//
// Ours (the lessons say so): the time units and costs below; a look-back round reads every
// predecessor's state at once; repeated additions counted against the G - 1 a chain needs.
//
// What this leaves out: the memory system (a published value is visible from the next
// round), real segment times (equal segments with seeded jitter), the cost of a barrier
// (a fixed charge; the book calls it very expensive and gives no figure), back-off in the
// spin, and anything a real scheduler does beyond the two rules.
//
// Erasable TypeScript only, and no value imports: the browser checks import this file
// directly in Node.

import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- configuration
export type StartOrder = 'index' | 'reversed' | 'shuffle';
export type BlockIndex = 'hardware' | 'counter';
export type Scheme = 'barrier' | 'single' | 'decoupled';
export type Progress = 'started' | 'none';

export const ORDER_LABEL: Record<StartOrder, string> = {
  get index() {
    return tr('index order', 'thứ tự chỉ số');
  },
  get reversed() {
    return tr('reversed', 'đảo ngược');
  },
  get shuffle() {
    return tr('seeded shuffle', 'xáo trộn có seed');
  },
};
export const INDEX_LABEL: Record<BlockIndex, string> = {
  hardware: 'blockIdx.x',
  get counter() {
    return tr('claimed from a counter', 'lấy từ một bộ đếm');
  },
};
export const SCHEME_LABEL: Record<Scheme, string> = {
  get barrier() {
    return tr('grid-wide barrier', 'rào chắn toàn lưới');
  },
  get single() {
    return tr('single lookback', 'lookback đơn');
  },
  get decoupled() {
    return tr('decoupled lookback', 'lookback tách rời (decoupled)');
  },
};
export const PROGRESS_LABEL: Record<Progress, string> = {
  get started() {
    return tr('a started block keeps progressing (CUDA)', 'một khối đã bắt đầu sẽ tiếp tục tiến triển (CUDA)');
  },
  get none() {
    return tr('no guarantee (Metal)', 'không có bảo đảm (Metal)');
  },
};

export interface LookbackConfig {
  /** Blocks in the grid. */
  G: number;
  /** Blocks the device can hold at once. */
  R: number;
  order: StartOrder;
  /** Seed for the shuffle and for each block's jitter. */
  seed: number;
  index: BlockIndex;
  scheme: Scheme;
  progress: Progress;
  /** An explicit start order (hardware indices), overriding `order`. */
  custom?: number[];
  /** Turn the jitter off: every segment takes LOCAL rounds. */
  even?: boolean;
}

/** Rounds to load and scan a segment, before jitter of 0 to JITTER - 1 more. */
export const LOCAL = 6;
export const JITTER = 4;
/** Rounds to add the prefix to the segment and store it. */
export const FINISH = 2;
/** Rounds a grid barrier is charged after its last arrival (ours; no figure in the book or the facts). */
export const BARRIER = 3;
/** Rounds block 0 needs to scan the block sums between the two barriers. */
export const SCAN_SUMS = 2;

// ---------------------------------------------------------------- seeded helpers
/** A small seeded generator (mulberry32). */
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
/** Hardware block indices in the order the scheduler starts them. */
export function startOrder(G: number, order: StartOrder, seed: number): number[] {
  const ids = Array.from({ length: G }, (_, i) => i);
  if (order === 'reversed') return ids.reverse();
  if (order === 'shuffle') {
    const r = rng(seed);
    for (let i = G - 1; i > 0; i--) {
      const j = Math.floor(r() * (i + 1));
      [ids[i], ids[j]] = [ids[j], ids[i]];
    }
  }
  return ids;
}
/** Rounds each hardware block needs for its local scan: LOCAL plus seeded jitter. */
export function localRounds(G: number, seed: number, even = false): number[] {
  const r = rng((seed ^ 0x9e3779b9) >>> 0);
  return Array.from({ length: G }, () => (even ? LOCAL : LOCAL + Math.floor(r() * JITTER)));
}

// ---------------------------------------------------------------- trace
/** What a block does in a round. "spin" is waiting on a flag or at a barrier; "starved" is resident but not run. */
export type Phase = 'local' | 'look' | 'spin' | 'barrier' | 'scan' | 'finish' | 'starved';
export const PHASE_LABEL: Record<Phase, string> = {
  get local() {
    return tr('scanning its segment', 'đang scan đoạn của nó');
  },
  get look() {
    return tr('looking back, adding', 'đang nhìn lại, cộng dồn');
  },
  get spin() {
    return tr('spinning on a flag', 'đang quay chờ một cờ');
  },
  get barrier() {
    return tr('waiting at the barrier', 'đang chờ ở rào chắn');
  },
  get scan() {
    return tr('scanning the block sums', 'đang scan các tổng của khối');
  },
  get finish() {
    return tr('adding and storing', 'đang cộng và lưu');
  },
  get starved() {
    return tr('resident, not run', 'thường trú, không được chạy');
  },
};
export interface Span {
  from: number;
  to: number;
  phase: Phase;
}
export type Flag = 'none' | 'own' | 'inclusive';
export interface Event {
  t: number;
  hw: number;
  idx: number;
  kind: 'start' | 'own' | 'inclusive' | 'done';
}
export interface BlockTrace {
  /** blockIdx.x */
  hw: number;
  /** The segment it scans: blockIdx.x, or the index it claimed; null if it never started. */
  idx: number | null;
  start: number | null;
  end: number | null;
  spans: Span[];
  /** Logical indices it spun on. */
  waitedOn: number[];
  /** Rounds spent spinning or waiting at a barrier. */
  wait: number;
  additions: number;
  /** Logical index whose scanned sum (inclusive prefix) it used. */
  dep: number | null;
  /** Hops from block 0 along the chain of scanned sums it depends on. */
  chain: number | null;
}
export type Outcome = 'done' | 'deadlock' | 'stall' | 'refused';
export interface LookbackRun {
  config: LookbackConfig;
  outcome: Outcome;
  /** Why the launch was refused, if it was. */
  refused: string | null;
  /** Hardware indices in start order. */
  dispatch: number[];
  local: number[];
  /** By hardware index. */
  blocks: BlockTrace[];
  events: Event[];
  /** Rounds until every block finished, or until nothing could move. */
  time: number;
  /** Longest chain of scanned sums (lookback), or 2 barriers. */
  criticalPath: number;
  additions: number;
  /** Additions beyond the G - 1 a chain of single lookbacks needs. */
  repeated: number;
  waitTotal: number;
  /** When stuck: each spinning block and what it waits for. */
  stuck: { hw: number; idx: number; on: number; onStarted: boolean }[];
}

// ---------------------------------------------------------------- the simulation
interface State {
  status: 'queued' | 'running' | 'done';
  idx: number | null;
  phase: 'local' | 'look' | 'barrier1' | 'scan' | 'barrier2' | 'finish';
  remaining: number;
  lookPos: number;
}

export function simulate(config: LookbackConfig): LookbackRun {
  const { G, R, scheme, progress } = config;
  const dispatch = config.custom ? [...config.custom] : startOrder(G, config.order, config.seed);
  const local = localRounds(G, config.seed, config.even);
  const blocks: BlockTrace[] = Array.from({ length: G }, (_, hw) => ({
    hw, idx: null, start: null, end: null, spans: [], waitedOn: [], wait: 0, additions: 0, dep: null, chain: null,
  }));
  const base = { config, dispatch, local, blocks, events: [] as Event[], time: 0, criticalPath: 0, additions: 0, repeated: 0, waitTotal: 0, stuck: [] };

  if (scheme === 'barrier' && progress === 'none')
    return { ...base, outcome: 'refused', refused: tr('A grid-wide barrier needs a cooperative launch. Under this rule there is none: Metal has no grid barrier.', 'Một rào chắn toàn lưới cần một lần khởi chạy cooperative. Theo quy tắc này thì không có: Metal không có rào chắn lưới.') };
  if (scheme === 'barrier' && G > R)
    return {
      ...base,
      outcome: 'refused',
      refused: tr(
        `A grid-wide barrier needs every block resident at once: ${G} blocks do not fit in ${R} slots, so the cooperative launch is refused.`,
        `Một rào chắn toàn lưới cần mọi khối cùng thường trú một lúc: ${G} khối không vừa ${R} chỗ, nên lần khởi chạy cooperative bị từ chối.`,
      ),
    };

  const st: State[] = Array.from({ length: G }, () => ({ status: 'queued', idx: null, phase: 'local', remaining: 0, lookPos: -1 }));
  const flags: Flag[] = Array.from({ length: G }, () => 'none'); // by logical index
  const events: Event[] = [];
  let next = 0;
  let counter = 0;
  let arrivals1 = 0;
  let release1: number | null = null;
  let release2: number | null = null;
  let t = 0;
  let outcome: Outcome = 'done';

  const mark = (hw: number, phase: Phase) => {
    const s = blocks[hw].spans;
    const last = s[s.length - 1];
    if (last && last.phase === phase && last.to === t) last.to = t + 1;
    else s.push({ from: t, to: t + 1, phase });
  };
  const byIdx = (i: number) => st.findIndex((s) => s.idx === i);

  for (; t < 100000; t++) {
    let changed = false;
    // Start blocks while slots are free.
    while (next < G && st.filter((s) => s.status === 'running').length < R) {
      const hw = dispatch[next++];
      const idx = config.index === 'hardware' ? hw : counter++;
      st[hw] = { status: 'running', idx, phase: 'local', remaining: local[hw], lookPos: idx - 1 };
      blocks[hw].idx = idx;
      blocks[hw].start = t;
      events.push({ t, hw, idx, kind: 'start' });
      changed = true;
    }
    if (st.every((s) => s.status === 'done')) break;
    // With the hardware index, a block that never starts still has its index.
    if (config.index === 'hardware') for (let hw = 0; hw < G; hw++) if (blocks[hw].idx == null) blocks[hw].idx = hw;

    const snap = [...flags];
    const running = st.map((s, hw) => ({ s, hw })).filter((x) => x.s.status === 'running');
    // Would this block spin this round, reading the flags as they stand?
    const spins = (s: State) => {
      if (s.phase !== 'look' || s.idx == null) return false;
      if (scheme === 'single') return snap[s.idx - 1] !== 'inclusive';
      return snap[s.lookPos] === 'none';
    };
    const atBarrier = (s: State) =>
      (s.phase === 'barrier1' && (release1 == null || t < release1)) || (s.phase === 'barrier2' && (release2 == null || t < release2));
    const spinners = running.filter((x) => spins(x.s));
    const scheduled =
      progress === 'none' && spinners.length ? new Set(spinners.map((x) => x.hw)) : new Set(running.map((x) => x.hw));

    const pending: { idx: number; flag: Flag; hw: number }[] = [];
    for (const { s, hw } of running) {
      const b = blocks[hw];
      const idx = s.idx!;
      if (!scheduled.has(hw)) {
        mark(hw, 'starved');
        continue;
      }
      // The phase this round runs. A block that passes a barrier runs its next phase in the same round.
      let now = s.phase;
      if (now === 'local') {
        mark(hw, 'local');
        changed = true;
        if (--s.remaining > 0) continue;
        if (scheme === 'barrier') {
          s.phase = 'barrier1';
          arrivals1++;
        } else if (idx === 0) {
          pending.push({ idx, flag: 'inclusive', hw });
          b.dep = null;
          b.chain = 0;
          s.phase = 'finish';
          s.remaining = FINISH;
        } else {
          if (scheme === 'decoupled') pending.push({ idx, flag: 'own', hw });
          s.phase = 'look';
          s.lookPos = idx - 1;
        }
      } else if (now === 'look') {
        if (scheme === 'single') {
          if (snap[idx - 1] === 'inclusive') {
            b.additions++;
            b.dep = idx - 1;
            pending.push({ idx, flag: 'inclusive', hw });
            s.phase = 'finish';
            s.remaining = FINISH;
            mark(hw, 'look');
            changed = true;
          } else {
            b.wait++;
            if (!b.waitedOn.includes(idx - 1)) b.waitedOn.push(idx - 1);
            mark(hw, 'spin');
          }
        } else {
          let j = s.lookPos;
          let moved = false;
          while (j >= 0) {
            const f = snap[j];
            if (f === 'inclusive') {
              b.additions++;
              b.dep = j;
              pending.push({ idx, flag: 'inclusive', hw });
              s.phase = 'finish';
              s.remaining = FINISH;
              moved = true;
              break;
            }
            if (f === 'own') {
              b.additions++;
              j--;
              moved = true;
              continue;
            }
            break;
          }
          s.lookPos = j;
          if (moved) {
            mark(hw, 'look');
            changed = true;
          } else {
            b.wait++;
            if (!b.waitedOn.includes(j)) b.waitedOn.push(j);
            mark(hw, 'spin');
          }
        }
      } else if (now === 'barrier1' || now === 'barrier2') {
        if (atBarrier(s)) {
          b.wait++;
          mark(hw, 'barrier');
          if ((s.phase === 'barrier1' ? release1 : release2) != null) changed = true; // the barrier is opening
          continue;
        }
        changed = true;
        // Through the barrier: block 0 scans the block sums, the others go on to wait at the second
        // barrier; after the second barrier everyone adds and stores. The new phase runs this round.
        if (s.phase === 'barrier1' && idx !== 0) {
          s.phase = 'barrier2';
          b.wait++;
          mark(hw, 'barrier');
          continue;
        }
        s.phase = s.phase === 'barrier1' ? 'scan' : 'finish';
        s.remaining = s.phase === 'scan' ? SCAN_SUMS : FINISH;
        now = s.phase;
      }
      if (now === 'scan') {
        mark(hw, 'scan');
        changed = true;
        if (--s.remaining === 0) {
          b.additions += G - 1;
          s.phase = 'barrier2';
          release2 = t + 1 + BARRIER;
        }
      } else if (now === 'finish') {
        mark(hw, 'finish');
        changed = true;
        if (--s.remaining === 0) {
          s.status = 'done';
          b.end = t + 1;
          events.push({ t: t + 1, hw, idx, kind: 'done' });
        }
      }
    }
    for (const p of pending) {
      flags[p.idx] = p.flag;
      events.push({ t: t + 1, hw: p.hw, idx: p.idx, kind: p.flag === 'own' ? 'own' : 'inclusive' });
      changed = true;
    }
    if (scheme === 'barrier' && arrivals1 === G && release1 == null) release1 = t + 1 + BARRIER;
    if (!changed) {
      outcome = progress === 'started' ? 'deadlock' : 'stall';
      break;
    }
  }

  // Chains of scanned sums, in index order (a block's dependency always has a smaller index).
  const chainOf = new Map<number, number>();
  if (scheme !== 'barrier') {
    const order = [...blocks].filter((b) => b.idx != null && (b.dep != null || b.chain === 0)).sort((a, b) => a.idx! - b.idx!);
    for (const b of order) {
      const c = b.dep == null ? 0 : (chainOf.get(b.dep) ?? 0) + 1;
      b.chain = c;
      chainOf.set(b.idx!, c);
    }
  }
  const additions = blocks.reduce((s, b) => s + b.additions, 0);
  const stuck =
    outcome === 'deadlock' || outcome === 'stall'
      ? st.flatMap((s, hw) => {
          if (s.status !== 'running' || s.phase !== 'look' || s.idx == null) return [];
          const on = scheme === 'single' ? s.idx - 1 : s.lookPos;
          const target = byIdx(on);
          return [{ hw, idx: s.idx, on, onStarted: target >= 0 && st[target].status !== 'queued' }];
        })
      : [];
  return {
    ...base,
    events,
    outcome,
    refused: null,
    time: t,
    criticalPath: scheme === 'barrier' ? 2 : Math.max(0, ...chainOf.values()),
    additions,
    repeated: scheme === 'barrier' ? 0 : Math.max(0, additions - (G - 1)),
    waitTotal: blocks.reduce((s, b) => s + b.wait, 0),
    stuck,
  };
}

// ---------------------------------------------------------------- reading a run at a moment
/** What a block is doing in round t: a phase, "queued" before it starts, "done" after it ends. */
export function phaseAt(b: BlockTrace, t: number): Phase | 'queued' | 'done' {
  if (b.start == null || t < b.start) return 'queued';
  if (b.end != null && t >= b.end) return 'done';
  const s = b.spans.find((x) => x.from <= t && t < x.to);
  return s ? s.phase : b.spans.length ? b.spans[b.spans.length - 1].phase : 'queued';
}
/** Each logical index's flag after round t - 1 (events up to and including time t). */
export function flagsAt(run: LookbackRun, t: number): Flag[] {
  const f: Flag[] = Array.from({ length: run.config.G }, () => 'none');
  for (const e of run.events) if (e.t <= t && (e.kind === 'own' || e.kind === 'inclusive')) f[e.idx] = e.kind;
  return f;
}
/** Blocks ordered for drawing: by the segment they scan, then those that never started, by blockIdx. */
export function rows(run: LookbackRun): BlockTrace[] {
  return [...run.blocks].sort((a, b) => {
    const ai = a.start == null && run.config.index === 'counter' ? Infinity : (a.idx ?? Infinity);
    const bi = b.start == null && run.config.index === 'counter' ? Infinity : (b.idx ?? Infinity);
    return ai - bi || a.hw - b.hw;
  });
}

// ---------------------------------------------------------------- the scene's runs
/** Six blocks on three slots, fixed seeds. Every frame of the "lookback" scene is drawn from these. */
export const SCENE_G = 6;
export const SCENE_R = 3;
export const SCENE_SEED = 6;
export function sceneRuns() {
  const base = { G: SCENE_G, R: SCENE_R, seed: SCENE_SEED, progress: 'started' as Progress };
  return {
    single: simulate({ ...base, order: 'index', index: 'counter', scheme: 'single' }),
    deadlock: simulate({ ...base, order: 'index', custom: [3, 4, 5, 0, 1, 2], index: 'hardware', scheme: 'single' }),
    decoupled: simulate({ ...base, order: 'index', index: 'counter', scheme: 'decoupled' }),
    metal: simulate({ ...base, order: 'index', index: 'counter', scheme: 'single', progress: 'none' }),
  };
}

// ---------------------------------------------------------------- the flag protocol (Fig. 11.17, facts A5)
export type StoreOrder = 'relaxed' | 'release';
export type LoadOrder = 'relaxed' | 'acquire';
export type Scope = 'block' | 'device';
export type Verdict = 'guaranteed' | 'race' | 'scope';
export interface MessagePassing {
  verdict: Verdict;
  title: string;
  why: string;
  /** Where the verdict comes from. */
  source: string;
  /**
   * The two blocks' code in CUDA, with the chosen orders and scope: the bodies of publish() and
   * waitFor() in labs/ch11/snippets/s119_flag_protocol.cu with the names substituted
   * (app/src/chapters/ch11/S119.snippets.test.ts checks it and front-end-checks every combination).
   */
  writer: string[];
  reader: string[];
}

const ORDER_NAME = (o: string) => `cuda::memory_order_${o}`; // i18n-ignore: CUDA code
export function messagePassing(store: StoreOrder, load: LoadOrder, scope: Scope): MessagePassing {
  const ref = `cuda::atomic_ref<unsigned int, cuda::thread_scope_${scope}>`;
  const writer = [
    `partialSums[bid] = prev + blockSum;          // plain store`,
    `${ref} f(flags[bid]);`, // i18n-ignore: CUDA code
    `f.store(1, ${ORDER_NAME(store)});`, // i18n-ignore: CUDA code
  ];
  const reader = [
    `${ref} f(flags[bid - 1]);`, // i18n-ignore: CUDA code
    `while (f.load(${ORDER_NAME(load)}) == 0) {}`,
    `prev = partialSums[bid - 1];                 // plain load`,
  ];
  if (scope === 'block')
    return {
      verdict: 'scope', // i18n-ignore: verdict identifier
      title: tr('Undefined: a block-scope atomic used across blocks', 'Không xác định: một atomic phạm vi khối được dùng giữa các khối'),
      why:
        tr(
          'A block-scope atomic orders accesses only among the threads of one block. The flag is written by one block and read by another, so even the flag accesses race. NVIDIA marks the variant with block scope on the store "UB: data race"; block scope on the load as well is no better.',
          'Một atomic phạm vi khối chỉ sắp thứ tự các truy cập giữa các luồng của một khối. Cờ được một khối ghi và một khối khác đọc, nên ngay cả các truy cập cờ cũng tranh chấp. NVIDIA đánh dấu biến thể có phạm vi khối ở phép store là "UB: data race"; thêm phạm vi khối ở phép load cũng không khá hơn.',
        ),
      source: tr('NVIDIA memory model, message-passing example (facts A5); ch09 A2', 'Mô hình bộ nhớ NVIDIA, ví dụ message-passing (facts A5); ch09 A2'),
      writer,
      reader,
    };
  if (store === 'release' && load === 'acquire')
    return {
      verdict: 'guaranteed', // i18n-ignore: verdict identifier
      title: tr('Guaranteed to see the sum', 'Bảo đảm thấy được tổng'),
      why:
        tr(
          'The release store cannot be moved before the plain store of the sum, and the acquire load cannot be moved after the plain load. A reader that sees the flag set therefore sees the sum. This is NVIDIA\'s message-passing example with partialSums in the role of x.',
          'Phép store release không thể bị dời lên trước phép store thường của tổng, và phép load acquire không thể bị dời xuống sau phép load thường. Vì vậy một bên đọc thấy cờ đã bật thì cũng thấy tổng. Đây là ví dụ message-passing của NVIDIA với partialSums đóng vai x.',
        ),
      source: tr('NVIDIA memory model, message-passing example (facts A5); book pp. 278-279', 'Mô hình bộ nhớ NVIDIA, ví dụ message-passing (facts A5); sách pp. 278-279'),
      writer,
      reader,
    };
  const which =
    store === 'relaxed' && load === 'relaxed'
      ? tr('Neither side orders the plain accesses', 'Không bên nào sắp thứ tự các truy cập thường')
      : store === 'relaxed'
        ? tr('The writer\'s relaxed store lets the flag become visible before the sum', 'Phép store relaxed của bên ghi để cờ hiện ra trước tổng')
        : tr('The reader\'s relaxed load lets the load of the sum be moved before the flag check', 'Phép load relaxed của bên đọc để phép load tổng bị dời lên trước lần kiểm tra cờ');
  return {
    verdict: 'race',
    title: tr('Data race on partialSums: undefined, in practice possibly a stale sum', 'Race condition (tranh chấp dữ liệu) trên partialSums: không xác định, trong thực tế có thể là một tổng cũ'),
    why: tr(
      `${which}. The two blocks' accesses to partialSums are then not ordered, which is a data race: undefined behaviour, and on real hardware the reader can get the old value.`,
      `${which}. Khi đó các truy cập của hai khối vào partialSums không được sắp thứ tự, tức là một race condition: hành vi không xác định, và trên phần cứng thật bên đọc có thể nhận giá trị cũ.`,
    ),
    source: tr('book p. 279 (acquire on the read, release on the write); NVIDIA memory model (facts A5)', 'sách p. 279 (acquire khi đọc, release khi ghi); mô hình bộ nhớ NVIDIA (facts A5)'),
    writer,
    reader,
  };
}
export const MP_TABLE: { store: StoreOrder; load: LoadOrder; scope: Scope; verdict: Verdict }[] = (['device', 'block'] as Scope[]).flatMap((scope) =>
  (['release', 'relaxed'] as StoreOrder[]).flatMap((store) =>
    (['acquire', 'relaxed'] as LoadOrder[]).map((load) => ({ store, load, scope, verdict: messagePassing(store, load, scope).verdict })),
  ),
);

// The safe protocol spelled in MSL (facts C3, C4) is not generated here: the page loads the two
// files that the snippet test compiles, labs/ch11/snippets/s119_flag_msl41.metal (acquire and
// release, MSL 4.1) and s119_flag_msl32.metal (relaxed atomics with device-scope fences, MSL 3.2).

/** What the MSL spelling still needs, and why it is not enough (facts C3, C4, C5). */
export const METAL_NOTES = [
  'partialSums must be declared coherent(device): by default device memory is only threadgroup-coherent (MSL 3.2 and later).', // i18n-ignore: a module-level array cannot hold getters without changing its English tokens; consumer fix needed in S119
  'There is no sleep for the spin, and no documented forward progress between threadgroups: the single-pass scan is still unsafe on Metal.', // i18n-ignore: a module-level array cannot hold getters without changing its English tokens; consumer fix needed in S119
];
