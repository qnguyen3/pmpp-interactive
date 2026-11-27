// Problem decomposition for chapter 22, section 22.2 (book pp. 532–536): who owns what, who
// writes, who conflicts. Chapter 21 (atom-centric against grid-centric) can reuse it.
//
// What it models:
//  - A computation is a contribution graph: every contribution says "input i adds into
//    output o". Four computations: a 1-D convolution (zero-padded "same", or "valid"), a
//    decode-time matrix-vector product y = W x with W row-major, a histogram with a skew
//    control, and a sparse matrix-vector product whose first row can be made heavy.
//  - A decomposition hands the contributions to workers (threads, or warps for the
//    matrix-vector mappings). Output-centric: a worker owns outputs and gathers. Input-centric:
//    a worker owns inputs and scatters. For the matrix-vector product four mappings: a thread
//    per input column (input-centric), a thread per output, a warp per output, and split-K
//    (S warps per output, each over one chunk of the input dimension).
//  - The book's criteria (p. 533) as numbers: threads (parallelism exposed); outputs that
//    more than one worker writes, which need atomics, and the atomic updates they take; the
//    worst number of writers on one output; additions done in registers; inputs a worker must
//    inspect to find its own (ease of mapping); useful work per thread, maximum over mean (load
//    balance). A worker adds all its contributions to one output in a register and writes once.
//  - Warp 0's first load of W for the matrix-vector mappings, as byte addresses. The caller
//    applies coalesce.ts warpAccess to them (NVIDIA's 32-byte sector, ch06-facts A2), as
//    coarsen.ts does, because this file must have no value imports.
//  - A serialised-atomics schedule for the gather-scatter scene: every update is issued at
//    once; one address serves one update per tick; different addresses in parallel.
//  - A split-K trace for the stepper: a batch of rows times W, warp by warp, with lanes, a
//    shuffle reduction, and either a direct store (S = 1), an atomic add, or a plain += split
//    into its read and its write so that an interleaving can lose an update.
//  - TOUR: the book's classification of every pattern (pp. 533–536), with four rows of ours.
//
// What it leaves out: time, caches, coalescing beyond one instruction's segment count, the
// cost of an atomic relative to a register add (documented nowhere, ch22-facts A1), banks,
// occupancy, divergence. A lost update is the typical outcome of a race, not a guaranteed
// one (ch09-facts A3): the trace shows the interleaving that loses it.
//
// Erasable TypeScript only, and no value imports: the browser checks import this file
// directly in Node.

import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- problems

export type Kind = 'conv' | 'matvec' | 'hist' | 'spmv';

/** One contribution: input `i` adds into output `o`. */
export interface Edge {
  i: number;
  o: number;
}

interface Base {
  nIn: number;
  nOut: number;
  /** Every contribution, in the order a sequential loop over the outputs adds them. */
  edges: Edge[];
}
export interface ConvProblem extends Base {
  kind: 'conv';
  n: number;
  /** Filter width, odd. */
  w: number;
  /** "valid": n − w + 1 outputs, out[o] = Σ in[o + t]. Otherwise "same" with zero padding. */
  valid: boolean;
}
export interface MatvecProblem extends Base {
  kind: 'matvec';
  /** Outputs (rows of W). */
  N: number;
  /** Inputs (columns of W, length of x). Input index i of an edge is the weight W[o][i]'s column. */
  K: number;
}
export interface HistProblem extends Base {
  kind: 'hist';
  bins: number;
  values: number[];
}
export interface SpmvProblem extends Base {
  kind: 'spmv';
  rows: number;
  cols: number;
  /** Non-zero k is in row rowOf[k], column colOf[k]; non-zeros are stored row by row. */
  rowOf: number[];
  colOf: number[];
}
export type Problem = ConvProblem | MatvecProblem | HistProblem | SpmvProblem;

export function conv(n: number, w: number, valid = false): ConvProblem {
  const r = (w - 1) / 2;
  const nOut = valid ? n - w + 1 : n;
  const edges: Edge[] = [];
  for (let o = 0; o < nOut; o++)
    for (let t = 0; t < w; t++) {
      const i = valid ? o + t : o + t - r;
      if (i >= 0 && i < n) edges.push({ i, o });
    }
  return { kind: 'conv', n, w, valid, nIn: n, nOut, edges };
}

export function matvec(N: number, K: number): MatvecProblem {
  const edges: Edge[] = [];
  for (let o = 0; o < N; o++) for (let i = 0; i < K; i++) edges.push({ i, o });
  return { kind: 'matvec', N, K, nIn: K, nOut: N, edges };
}

/**
 * A deterministic histogram input: a fraction `skew` of the inputs (picked by a golden-ratio
 * sequence, so they are spread along the array) fall in bin 0, the rest go round the bins.
 */
export function histValues(n: number, bins: number, skew: number): number[] {
  const g = 0.6180339887498949;
  return Array.from({ length: n }, (_, i) => (((i + 1) * g) % 1 < skew ? 0 : i % bins));
}

export function hist(values: number[], bins: number): HistProblem {
  return {
    kind: 'hist', bins, values, nIn: values.length, nOut: bins,
    edges: values.map((v, i) => ({ i, o: v })),
  };
}

/**
 * A sparse matrix with `rows` rows of 2 non-zeros each, plus `heavy` extra non-zeros in row 0.
 * Columns are spread so that no row repeats one.
 */
export function spmv(rows: number, heavy: number): SpmvProblem {
  const cols = Math.max(8, 2 + heavy);
  const rowOf: number[] = [];
  const colOf: number[] = [];
  for (let r = 0; r < rows; r++) {
    const len = r === 0 ? 2 + heavy : 2;
    for (let k = 0; k < len; k++) {
      rowOf.push(r);
      colOf.push((r * 3 + k) % cols);
    }
  }
  return {
    kind: 'spmv', rows, cols, rowOf, colOf, nIn: rowOf.length, nOut: rows,
    edges: rowOf.map((o, i) => ({ i, o })),
  };
}

// ---------------------------------------------------------------- decompositions

/** conv, hist, spmv: 'output' or 'input'. matvec: all four. */
export type Mapping = 'output' | 'input' | 'warp' | 'splitk';

export const MAPPING_LABEL: Record<Kind, Partial<Record<Mapping, string>>> = {
  conv: {
    get output() {
      return tr('output-centric (gather)', 'hướng đầu ra (gather)');
    },
    get input() {
      return tr('input-centric (scatter)', 'hướng đầu vào (scatter)');
    },
  },
  hist: {
    get output() {
      return tr('output-centric: a thread per bin', 'hướng đầu ra: một luồng cho mỗi bin');
    },
    get input() {
      return tr('input-centric: a thread per input', 'hướng đầu vào: một luồng cho mỗi đầu vào');
    },
  },
  spmv: {
    get output() {
      return tr('output-centric: a thread per row (CSR)', 'hướng đầu ra: một luồng cho mỗi hàng (CSR)');
    },
    get input() {
      return tr('input-centric: a thread per non-zero (COO)', 'hướng đầu vào: một luồng cho mỗi phần tử khác 0 (COO)');
    },
  },
  matvec: {
    get input() {
      return tr('input-centric: a thread per input', 'hướng đầu vào: một luồng cho mỗi đầu vào');
    },
    get output() {
      return tr('a thread per output', 'một luồng cho mỗi đầu ra');
    },
    get warp() {
      return tr('a warp per output', 'một warp cho mỗi đầu ra');
    },
    get splitk() {
      return tr('split the input dimension: S warps per output', 'chia chiều đầu vào: S warp cho mỗi đầu ra');
    },
  },
};

export const mappingsOf = (kind: Kind): Mapping[] => (kind === 'matvec' ? ['input', 'output', 'warp', 'splitk'] : ['output', 'input']);

export interface Write {
  o: number;
  /** Contributions this worker adds together before this one write. */
  parts: number;
}

export interface Worker {
  id: number;
  /** 1 for a thread; the warp width for a warp. */
  lanes: number;
  /** Inputs whose contributions it computes (its useful reads), in order. */
  useful: number[];
  /** How many inputs it must look at to find them. */
  inspects: number;
  /** True when it must look at every input (an output-centric histogram bin). */
  inspectsAll: boolean;
  writes: Write[];
  /** Useful work of each of its threads (one entry for a thread, one per lane for a warp). */
  laneWork: number[];
}

export interface Metrics {
  /** Threads launched (lanes, for warp mappings): the parallelism exposed. */
  threads: number;
  workers: number;
  contributions: number;
  /** Outputs written by more than one worker: each of their updates needs an atomic. */
  outputsNeedingAtomics: number;
  atomicUpdates: number;
  /** Writes to outputs with a single writer: plain stores. */
  plainStores: number;
  worstWriters: number;
  /** Contributions added in a register rather than in memory: contributions − writes. */
  registerAccumulations: number;
  inspectedTotal: number;
  inspectedMax: number;
  /** Useful work per thread. */
  workMax: number;
  workMean: number;
  /** workMax ÷ workMean: 1 is perfect balance. */
  imbalance: number;
}

export interface Options {
  /** Lanes per warp for the warp mappings. */
  warpWidth: number;
  /** Chunks of the input dimension for split-K. */
  S: number;
}

export interface Decomposition {
  problem: Problem;
  mapping: Mapping;
  opts: Options;
  workers: Worker[];
  /** Distinct workers writing each output. */
  writers: number[];
  metrics: Metrics;
}

/** Split `count` consecutive items over `lanes` lanes the way a strided loop does: lane l gets l, l + lanes, ... */
function stride(count: number, lanes: number): number[] {
  return Array.from({ length: lanes }, (_, l) => (l < count ? Math.floor((count - 1 - l) / lanes) + 1 : 0));
}

function thread(id: number, useful: number[], writes: Write[], inspects = useful.length, inspectsAll = false): Worker {
  return { id, lanes: 1, useful, inspects, inspectsAll, writes, laneWork: [useful.length] };
}

function workersOf(p: Problem, mapping: Mapping, opts: Options): Worker[] {
  if (p.kind === 'matvec') {
    const { N, K } = p;
    const all = Array.from({ length: K }, (_, i) => i);
    switch (mapping) {
      case 'input':
        return all.map((i) => thread(i, [i], Array.from({ length: N }, (_, o) => ({ o, parts: 1 })), 1));
      case 'output':
        return Array.from({ length: N }, (_, o) => thread(o, all, [{ o, parts: K }]));
      case 'warp':
        return Array.from({ length: N }, (_, o) => ({
          id: o, lanes: opts.warpWidth, useful: all, inspects: K, inspectsAll: false,
          writes: [{ o, parts: K }], laneWork: stride(K, opts.warpWidth),
        }));
      case 'splitk': {
        const { S, warpWidth } = opts;
        const out: Worker[] = [];
        for (let w = 0; w < N * S; w++) {
          const o = Math.floor(w / S);
          const { start, len } = chunkOf(K, S, w % S);
          const useful = Array.from({ length: len }, (_, t) => start + t);
          out.push({ id: w, lanes: warpWidth, useful, inspects: len, inspectsAll: false, writes: [{ o, parts: len }], laneWork: stride(len, warpWidth) });
        }
        return out;
      }
    }
  }
  if (mapping === 'output') {
    const by: number[][] = Array.from({ length: p.nOut }, () => []);
    for (const e of p.edges) by[e.o].push(e.i);
    // An output-centric histogram bin cannot tell which inputs are its own without looking (p. 534).
    const all = p.kind === 'hist';
    return by.map((ins, o) => thread(o, ins, [{ o, parts: ins.length }], all ? p.nIn : ins.length, all));
  }
  // input-centric: one input per thread; its contributions to one output are added before the write.
  const by: Map<number, number>[] = Array.from({ length: p.nIn }, () => new Map());
  for (const e of p.edges) by[e.i].set(e.o, (by[e.i].get(e.o) ?? 0) + 1);
  return by.map((m, i) => {
    const writes = [...m.entries()].map(([o, parts]) => ({ o, parts }));
    const w = thread(i, [i], writes, 1);
    w.laneWork = [writes.reduce((s, x) => s + x.parts, 0)];
    return w;
  });
}

/**
 * Chunk `c` of `S` over K inputs, cut as lab 22.1's gemvSplitKKernel cuts them: chunks of
 * ceil(K / S), the last one shorter (or empty) when S does not divide K.
 */
export function chunkOf(K: number, S: number, c: number): { start: number; len: number } {
  const size = Math.ceil(K / S);
  const start = Math.min(K, c * size);
  return { start, len: Math.max(0, Math.min(K, start + size) - start) };
}

export function decompose(problem: Problem, mapping: Mapping, opts: Options): Decomposition {
  if (problem.kind !== 'matvec' && (mapping === 'warp' || mapping === 'splitk'))
    throw new Error(`mapping ${mapping} is for the matrix-vector product only`);
  const workers = workersOf(problem, mapping, opts);
  const writerSets: Set<number>[] = Array.from({ length: problem.nOut }, () => new Set());
  for (const w of workers) for (const x of w.writes) writerSets[x.o].add(w.id);
  const writers = writerSets.map((s) => s.size);
  let atomicUpdates = 0;
  let plainStores = 0;
  for (const w of workers)
    for (const x of w.writes) {
      if (writers[x.o] > 1) atomicUpdates++;
      else plainStores++;
    }
  const contributions = problem.edges.length;
  const laneWork = workers.flatMap((w) => w.laneWork);
  const threads = laneWork.length;
  const max = (xs: number[]) => xs.reduce((m, v) => (v > m ? v : m), 0);
  const workMax = max(laneWork);
  const workMean = threads ? laneWork.reduce((s, v) => s + v, 0) / threads : 0;
  const metrics: Metrics = {
    threads,
    workers: workers.length,
    contributions,
    outputsNeedingAtomics: writers.filter((v) => v > 1).length,
    atomicUpdates,
    plainStores,
    worstWriters: max(writers),
    registerAccumulations: contributions - atomicUpdates - plainStores,
    inspectedTotal: workers.reduce((s, w) => s + w.inspects, 0),
    inspectedMax: max(workers.map((w) => w.inspects)),
    workMax,
    workMean,
    imbalance: workMean ? workMax / workMean : 1,
  };
  return { problem, mapping, opts, workers, writers, metrics };
}

/** How many workers read each input (the reuse a block could keep in shared memory). */
export function readsPerInput(d: Decomposition): number[] {
  const n = new Array(d.problem.nIn).fill(0);
  for (const w of d.workers) for (const i of w.useful) n[i]++;
  return n;
}

// ---------------------------------------------------------------- warp 0's first load of W

/**
 * Byte addresses (4-byte floats, W row-major N × K) of warp 0's first load of W under a
 * matrix-vector mapping. Lane l of warp 0 is thread l for the thread mappings. Apply
 * coalesce.ts warpAccess(addrs, 4, 32) for NVIDIA's sector count.
 */
export function warp0Load(p: MatvecProblem, mapping: Mapping, opts: Options): number[] {
  const { N, K } = p;
  const lanes = (n: number) => Array.from({ length: Math.min(opts.warpWidth, n) }, (_, l) => l);
  switch (mapping) {
    case 'output':
      return lanes(N).map((l) => (l * K + 0) * 4); // thread l walks row l: W[l][0], K floats apart
    case 'input':
      return lanes(K).map((l) => (0 * K + l) * 4); // thread l walks column l: W[0][l]
    case 'warp':
      return lanes(K).map((l) => (0 * K + l) * 4); // warp 0 owns row 0: W[0][l]
    case 'splitk': {
      const { start, len } = chunkOf(K, opts.S, 0);
      return lanes(len).map((l) => (0 * K + start + l) * 4); // warp 0: row 0, chunk 0
    }
  }
}

// ---------------------------------------------------------------- one sentence per worker

const fmtList = (xs: number[]) => (xs.length <= 6 ? xs.join(', ') : `${xs[0]}, ${xs[1]}, … ${xs[xs.length - 1]}`);

/** The index mapping of one worker, in words, with its own numbers. */
export function describeWorker(d: Decomposition, id: number): string {
  const p = d.problem;
  const w = d.workers[id];
  const shared = (o: number) => d.writers[o] > 1;
  const outs = w.writes.map((x) => x.o);
  const nShared = outs.filter(shared).length;
  const tail = nShared
    ? tr(
        ` ${nShared === outs.length ? (outs.length === 1 ? 'That output has' : 'Every one of them has') : `${nShared} of them have`} other writers too, so each update is an atomic.`,
        ` ${nShared === outs.length ? (outs.length === 1 ? 'Đầu ra đó' : 'Tất cả các đầu ra đó') : `${nShared} trong số đó`} còn có worker khác cùng ghi, nên mỗi lần cập nhật là một phép toán nguyên tử.`,
      )
    : tr(' No other worker writes there, so a plain store is safe.', ' Không worker nào khác ghi vào đó, nên một lệnh store thường là an toàn.');
  if (p.kind === 'conv') {
    const r = (p.w - 1) / 2;
    if (d.mapping === 'output') {
      const lo = p.valid ? 'o' : 'o − ' + r;
      const hi = p.valid ? `o + ${p.w - 1}` : 'o + ' + r;
      return tr(`Thread ${id} owns out[${id}]. It reads in[${lo}] to in[${hi}] that exist (in[${fmtList(w.useful)}]), adds ${w.useful.length} products in a register and stores once.`, `Luồng ${id} sở hữu out[${id}]. Nó đọc những phần tử từ in[${lo}] đến in[${hi}] có tồn tại (in[${fmtList(w.useful)}]), cộng ${w.useful.length} tích trong một thanh ghi và store một lần.`) + tail;
    }
    return tr(`Thread ${id} owns in[${id}]. It multiplies it by each filter tap and adds the product into every output that uses it: out[${fmtList(outs)}].`, `Luồng ${id} sở hữu in[${id}]. Nó nhân phần tử này với từng hệ số của bộ lọc và cộng tích vào mọi đầu ra dùng nó: out[${fmtList(outs)}].`) + tail;
  }
  if (p.kind === 'hist') {
    if (d.mapping === 'output')
      return tr(`Thread ${id} owns bin ${id}. It cannot know which inputs fall in its bin without looking, so it reads all ${p.nIn} inputs and counts the ${w.useful.length} that do.`, `Luồng ${id} sở hữu bin ${id}. Nó không thể biết đầu vào nào rơi vào bin của mình nếu không xem, nên nó đọc cả ${p.nIn} đầu vào và đếm ${w.useful.length} đầu vào rơi vào đó.`) + tail;
    return tr(`Thread ${id} owns input ${id}, whose value is ${p.values[id]}. It adds 1 to bin ${p.values[id]}.`, `Luồng ${id} sở hữu đầu vào ${id}, có giá trị ${p.values[id]}. Nó cộng 1 vào bin ${p.values[id]}.`) + tail;
  }
  if (p.kind === 'spmv') {
    if (d.mapping === 'output') {
      const first = w.useful[0] ?? 0;
      return tr(`Thread ${id} owns y[${id}]. The row pointers give its ${w.useful.length} non-zeros directly (k = ${first} to ${first + w.useful.length - 1}); it sums A·x over them in a register and stores once.`, `Luồng ${id} sở hữu y[${id}]. Các con trỏ hàng cho nó trực tiếp ${w.useful.length} phần tử khác 0 của mình (k = ${first} đến ${first + w.useful.length - 1}); nó cộng A·x trên các phần tử đó trong một thanh ghi và store một lần.`) + tail;
    }
    return tr(`Thread ${id} owns non-zero ${id}, in row ${p.rowOf[id]}, column ${p.colOf[id]}. It adds A[${p.rowOf[id]}][${p.colOf[id]}]·x[${p.colOf[id]}] into y[${p.rowOf[id]}].`, `Luồng ${id} sở hữu phần tử khác 0 thứ ${id}, ở hàng ${p.rowOf[id]}, cột ${p.colOf[id]}. Nó cộng A[${p.rowOf[id]}][${p.colOf[id]}]·x[${p.colOf[id]}] vào y[${p.rowOf[id]}].`) + tail;
  }
  const { K } = p;
  const ww = d.opts.warpWidth;
  switch (d.mapping) {
    case 'input':
      return tr(`Thread ${id} owns x[${id}], column ${id} of W. For every output j it adds W[j][${id}]·x[${id}] into y[j].`, `Luồng ${id} sở hữu x[${id}], cột ${id} của W. Với mọi đầu ra j, nó cộng W[j][${id}]·x[${id}] vào y[j].`) + tail;
    case 'output':
      return tr(`Thread ${id} owns y[${id}] and walks row ${id} of W alone, W[${id}][0] to W[${id}][${K - 1}]. Lane l of a warp reads W[l][k]: neighbouring lanes are ${K} floats apart.`, `Luồng ${id} sở hữu y[${id}] và một mình duyệt hàng ${id} của W, từ W[${id}][0] đến W[${id}][${K - 1}]. Lane l của một warp đọc W[l][k]: các lane cạnh nhau cách nhau ${K} float.`) + tail;
    case 'warp': {
      const it = Math.ceil(K / ww);
      return tr(`Warp ${id} owns y[${id}]. Lane l reads W[${id}][l + i × ${ww}] for i = 0 to ${it - 1}, so the warp's lanes read one contiguous run at each step; the lanes combine their sums by shuffles and lane 0 stores.`, `Warp ${id} sở hữu y[${id}]. Lane l đọc W[${id}][l + i × ${ww}] với i = 0 đến ${it - 1}, nên ở mỗi bước các lane của warp đọc một đoạn liên tiếp; các lane gộp tổng của chúng bằng shuffle và lane 0 store.`) + tail;
    }
    case 'splitk': {
      const S = d.opts.S;
      const o = Math.floor(id / S);
      const c = id % S;
      const { start, len } = chunkOf(K, S, c);
      const it = Math.ceil(len / ww);
      return tr(`Warp ${id} owns output ${id} ÷ ${S} = ${o} and chunk ${id} mod ${S} = ${c}, which starts at ${start}. Lane l reads W[${o}][${start} + l + i × ${ww}] for i = 0 to ${it - 1}; the lanes combine their sums by shuffles and lane 0 adds the warp's sum into y[${o}].`, `Warp ${id} sở hữu đầu ra ${id} ÷ ${S} = ${o} và chunk ${id} mod ${S} = ${c}, bắt đầu tại ${start}. Lane l đọc W[${o}][${start} + l + i × ${ww}] với i = 0 đến ${it - 1}; các lane gộp tổng của chúng bằng shuffle và lane 0 cộng tổng của warp vào y[${o}].`) + tail;
    }
  }
  return '';
}

/** Which worker owns W[row][col] under a matrix-vector mapping (for clicking on the drawing). */
export function ownerOfWeight(p: MatvecProblem, mapping: Mapping, opts: Options, row: number, col: number): number {
  switch (mapping) {
    case 'input':
      return col;
    case 'output':
    case 'warp':
      return row;
    case 'splitk': {
      for (let c = 0; c < opts.S; c++) {
        const { start, len } = chunkOf(p.K, opts.S, c);
        if (col >= start && col < start + len) return row * opts.S + c;
      }
      return row * opts.S;
    }
  }
}

// ---------------------------------------------------------------- explorer state and presets

export interface ExplorerState {
  kind: Kind;
  mapping: Mapping;
  conv: { n: number; w: number };
  matvec: { N: number; K: number; S: number; warpWidth: number };
  hist: { n: number; bins: number; skew: number };
  spmv: { rows: number; heavy: number };
}

/** NVIDIA's warp width on every listed compute capability (infra-facts, "Where the backends really differ"). */
export const NVIDIA_WARP = 32;

export const DEFAULT_STATE: ExplorerState = {
  kind: 'conv',
  mapping: 'output',
  conv: { n: 12, w: 5 },
  matvec: { N: 8, K: 256, S: 4, warpWidth: NVIDIA_WARP },
  hist: { n: 16, bins: 4, skew: 0.5 },
  spmv: { rows: 6, heavy: 8 },
};

export interface Preset {
  id: string;
  label: string;
  state: ExplorerState;
}

export const PRESETS: Preset[] = [
  { id: 'conv-scatter', get label() { return tr('Convolution, scattered', 'Tích chập, kiểu scatter'); }, state: { ...DEFAULT_STATE, kind: 'conv', mapping: 'input' } },
  { id: 'hist-bins', get label() { return tr('Histogram, a thread per bin', 'Histogram, một luồng cho mỗi bin'); }, state: { ...DEFAULT_STATE, kind: 'hist', mapping: 'output' } },
  { id: 'spmv-coo', get label() { return tr('Skewed sparse matrix, COO', 'Ma trận thưa bị lệch, COO'); }, state: { ...DEFAULT_STATE, kind: 'spmv', mapping: 'input' } },
  {
    id: 'gemv-narrow', get label() { return tr('Narrow decode GEMV, split-K', 'GEMV hẹp khi decode, split-K'); },
    state: { ...DEFAULT_STATE, kind: 'matvec', mapping: 'splitk', matvec: { N: 2, K: 256, S: 4, warpWidth: NVIDIA_WARP } },
  },
];

export function problemOf(s: ExplorerState): Problem {
  switch (s.kind) {
    case 'conv':
      return conv(s.conv.n, s.conv.w);
    case 'matvec':
      return matvec(s.matvec.N, s.matvec.K);
    case 'hist':
      return hist(histValues(s.hist.n, s.hist.bins, s.hist.skew), s.hist.bins);
    case 'spmv':
      return spmv(s.spmv.rows, s.spmv.heavy);
  }
}

export function decomposeState(s: ExplorerState): Decomposition {
  const mapping = mappingsOf(s.kind).includes(s.mapping) ? s.mapping : 'output';
  return decompose(problemOf(s), mapping, { warpWidth: s.matvec.warpWidth, S: s.matvec.S });
}

/** S values that give every lane of a split-K warp at least one element. */
export const splitChoices = (K: number, warpWidth: number): number[] => [1, 2, 4, 8, 16].filter((S) => K / S >= warpWidth);

// ---------------------------------------------------------------- serialised atomics (the scene)

export interface Tick {
  /** Updates completed on this tick: (output, worker). */
  served: { o: number; worker: number }[];
  /** Updates still waiting at each output after this tick. */
  waiting: number[];
}

/**
 * Every write is issued at tick 0. Each output serves one update per tick, in worker order;
 * different outputs are served in parallel. Returns one entry per tick until all are served.
 */
export function serialise(d: Decomposition): { ticks: Tick[]; issued: number[] } {
  const queues: number[][] = Array.from({ length: d.problem.nOut }, () => []);
  for (const w of d.workers) for (const x of w.writes) queues[x.o].push(w.id);
  const issued = queues.map((q) => q.length);
  const ticks: Tick[] = [];
  for (let t = 0; queues.some((q) => q.length > t); t++)
    ticks.push({
      served: queues.flatMap((q, o) => (t < q.length ? [{ o, worker: q[t] }] : [])),
      waiting: queues.map((q) => Math.max(0, q.length - t - 1)),
    });
  return { ticks, issued };
}

// ---------------------------------------------------------------- the scene's numbers

/** The scene's example: eight inputs, six outputs, each the sum of three neighbours. */
export const SCENE = { n: 8, w: 3, inputs: [3, 1, 4, 1, 5, 9, 2, 6], histBins: 3 };

export function sceneModel() {
  const p = conv(SCENE.n, SCENE.w, true);
  const gather = decompose(p, 'output', { warpWidth: 1, S: 1 });
  const scatter = decompose(p, 'input', { warpWidth: 1, S: 1 });
  const sums = gather.workers.map((w) => w.useful.reduce((s, i) => s + SCENE.inputs[i], 0));
  const fanOut = Array.from({ length: p.nIn }, (_, i) => p.edges.filter((e) => e.i === i).length);
  const hp = hist(SCENE.inputs.map((v) => v % SCENE.histBins), SCENE.histBins);
  const histGather = decompose(hp, 'output', { warpWidth: 1, S: 1 });
  return { p, gather, scatter, sums, fanOut, reads: readsPerInput(gather), gatherTicks: serialise(gather), scatterTicks: serialise(scatter), hp, histGather };
}

// ---------------------------------------------------------------- split-K, warp by warp

export interface SplitKConfig {
  batch: number;
  /** Inputs (length of each x row, columns of W). */
  K: number;
  /** Outputs (rows of W). */
  N: number;
  warpWidth: number;
  /** Chunks of the input dimension; N·S warps. */
  S: number;
  /** Lane 0 commits with an atomic add (true) or a plain += (false). Ignored when S = 1. */
  atomics: boolean;
}

/** The stepper's tiny case: batch 2, 16 inputs, 3 outputs, a toy warp of 4 so every lane is visible. */
export const SPLITK_TINY = { batch: 2, K: 16, N: 3, warpWidth: 4 };

/** Small integers so every sum is exact in floating point, whatever the order. */
export function splitKData(cfg: Pick<SplitKConfig, 'batch' | 'K' | 'N'>): { W: number[]; X: number[]; ref: number[] } {
  const { batch, K, N } = cfg;
  const W = Array.from({ length: N * K }, (_, t) => ((Math.floor(t / K) * 5 + (t % K) * 3) % 7) - 3);
  const X = Array.from({ length: batch * K }, (_, t) => ((Math.floor(t / K) * 3 + (t % K)) % 5) - 2);
  const ref: number[] = [];
  for (let b = 0; b < batch; b++)
    for (let j = 0; j < N; j++) {
      let s = 0;
      for (let k = 0; k < K; k++) s += W[j * K + k] * X[b * K + k];
      ref.push(s);
    }
  return { W, X, ref };
}

export type OpKind = 'load' | 'fma' | 'shfl' | 'store' | 'atomic' | 'read' | 'write';
export interface SKOp {
  kind: OpKind;
  /** Strip index for load and fma. */
  strip?: number;
  /** Shuffle distance for shfl. */
  offset?: number;
}

export const splitKWarps = (cfg: SplitKConfig) => cfg.N * cfg.S;

export function warpPlace(cfg: SplitKConfig, w: number) {
  const j = Math.floor(w / cfg.S);
  const c = w % cfg.S;
  const { start, len } = chunkOf(cfg.K, cfg.S, c);
  return { j, c, start, len, strips: Math.ceil(len / cfg.warpWidth) };
}

export function warpOps(cfg: SplitKConfig, w: number): SKOp[] {
  const { strips } = warpPlace(cfg, w);
  const ops: SKOp[] = [];
  for (let s = 0; s < strips; s++) ops.push({ kind: 'load', strip: s }, { kind: 'fma', strip: s });
  for (let off = cfg.warpWidth / 2; off >= 1; off /= 2) ops.push({ kind: 'shfl', offset: off });
  if (cfg.S === 1) ops.push({ kind: 'store' });
  else if (cfg.atomics) ops.push({ kind: 'atomic' });
  else ops.push({ kind: 'read' }, { kind: 'write' });
  return ops;
}

export type Interleave = 'sequential' | 'round-robin' | 'reversed';
export const INTERLEAVE_LABEL: Record<Interleave, string> = {
  get sequential() {
    return tr('one warp at a time', 'lần lượt từng warp');
  },
  get 'round-robin'() {
    return tr('all warps together, in turn', 'tất cả các warp cùng lúc, luân phiên');
  },
  get reversed() {
    return tr('one warp at a time, last warp first', 'lần lượt từng warp, warp cuối trước');
  },
};

/** A schedule: the warp that performs each successive operation. */
export function splitKOrder(cfg: SplitKConfig, how: Interleave): number[] {
  const W = splitKWarps(cfg);
  const counts = Array.from({ length: W }, (_, w) => warpOps(cfg, w).length);
  const order: number[] = [];
  if (how === 'sequential') counts.forEach((c, w) => { for (let k = 0; k < c; k++) order.push(w); });
  else if (how === 'reversed') for (let w = W - 1; w >= 0; w--) for (let k = 0; k < counts[w]; k++) order.push(w);
  else {
    const left = [...counts];
    while (left.some((c) => c > 0))
      for (let w = 0; w < W; w++)
        if (left[w] > 0) {
          order.push(w);
          left[w]--;
        }
  }
  return order;
}

export interface SKStep {
  warp: number;
  op: SKOp;
  j: number;
  c: number;
  /** For load: the K index each lane reads (−1 for a lane with nothing to read). */
  ks?: number[];
  /** Y cells touched (b·N + j) and what they hold afterwards. */
  touched?: number[];
  /** For write: chunks whose contribution this write destroyed. */
  lost?: number[];
}

export interface SKSnapshot {
  /** Y, batch × N, row-major. */
  Y: number[];
  /** Chunks each Y cell holds. */
  parts: number[][];
  /** Each warp's running sums, [b][lane]. */
  acc: number[][][];
  /** Operations each warp has done. */
  done: number[];
}

export interface SKRun {
  cfg: SplitKConfig;
  data: { W: number[]; X: number[]; ref: number[] };
  steps: SKStep[];
  /** Snapshot after each step; snaps[0] is before the first. */
  snaps: SKSnapshot[];
  final: number[];
  correct: boolean;
  /** Contributions (b, j, chunk) lost to plain += interleavings. */
  lostCount: number;
}

const clone = (s: SKSnapshot): SKSnapshot => ({
  Y: [...s.Y], parts: s.parts.map((p) => [...p]), acc: s.acc.map((a) => a.map((r) => [...r])), done: [...s.done],
});

/**
 * Runs the warps in `order` (each entry: the warp that performs its next operation; entries
 * for a finished warp are skipped). Every operation of a warp is one step.
 */
export function splitKRun(cfg: SplitKConfig, order: number[]): SKRun {
  const data = splitKData(cfg);
  const { W, X } = data;
  const { batch, N, K, warpWidth: ww } = cfg;
  const nW = splitKWarps(cfg);
  const ops = Array.from({ length: nW }, (_, w) => warpOps(cfg, w));
  let s: SKSnapshot = {
    Y: new Array(batch * N).fill(0),
    parts: Array.from({ length: batch * N }, () => []),
    acc: Array.from({ length: nW }, () => Array.from({ length: batch }, () => new Array(ww).fill(0))),
    done: new Array(nW).fill(0),
  };
  // Registers not in the snapshot: loaded weights and the copy plain += reads.
  const wreg: number[][] = Array.from({ length: nW }, () => new Array(ww).fill(0));
  const kreg: number[][] = Array.from({ length: nW }, () => new Array(ww).fill(-1));
  const tmp: { y: number[]; parts: number[][] }[] = Array.from({ length: nW }, () => ({ y: [], parts: [] }));
  const steps: SKStep[] = [];
  const snaps: SKSnapshot[] = [clone(s)];
  let lostCount = 0;
  for (const w of order) {
    if (s.done[w] >= ops[w].length) continue;
    const op = ops[w][s.done[w]];
    const { j, c, start, len } = warpPlace(cfg, w);
    const cells = Array.from({ length: batch }, (_, b) => b * N + j);
    const step: SKStep = { warp: w, op, j, c };
    s = clone(s);
    switch (op.kind) {
      case 'load': {
        const ks = Array.from({ length: ww }, (_, l) => {
          const t = l + op.strip! * ww;
          return t < len ? start + t : -1;
        });
        ks.forEach((k, l) => {
          kreg[w][l] = k;
          wreg[w][l] = k >= 0 ? W[j * K + k] : 0;
        });
        step.ks = ks;
        break;
      }
      case 'fma':
        for (let b = 0; b < batch; b++)
          for (let l = 0; l < ww; l++) if (kreg[w][l] >= 0) s.acc[w][b][l] += wreg[w][l] * X[b * K + kreg[w][l]];
        step.ks = [...kreg[w]];
        break;
      case 'shfl':
        for (let b = 0; b < batch; b++) {
          const old = [...s.acc[w][b]];
          for (let l = 0; l < op.offset!; l++) s.acc[w][b][l] = old[l] + old[l + op.offset!];
        }
        break;
      case 'store':
        cells.forEach((cell, b) => {
          s.Y[cell] = s.acc[w][b][0];
          s.parts[cell] = [c];
        });
        step.touched = cells;
        break;
      case 'atomic':
        cells.forEach((cell, b) => {
          s.Y[cell] += s.acc[w][b][0];
          s.parts[cell] = [...s.parts[cell], c].sort((x, y) => x - y);
        });
        step.touched = cells;
        break;
      case 'read':
        tmp[w] = { y: cells.map((cell) => s.Y[cell]), parts: cells.map((cell) => [...s.parts[cell]]) };
        step.touched = cells;
        break;
      case 'write': {
        const lost = new Set<number>();
        cells.forEach((cell, b) => {
          for (const q of s.parts[cell]) if (!tmp[w].parts[b].includes(q)) lost.add(q);
          lostCount += s.parts[cell].filter((q) => !tmp[w].parts[b].includes(q)).length;
          s.Y[cell] = tmp[w].y[b] + s.acc[w][b][0];
          s.parts[cell] = [...tmp[w].parts[b], c].sort((x, y) => x - y);
        });
        step.touched = cells;
        if (lost.size) step.lost = [...lost].sort((x, y) => x - y);
        break;
      }
    }
    s.done[w]++;
    steps.push(step);
    snaps.push(s);
  }
  const final = s.Y;
  return { cfg, data, steps, snaps, final, correct: final.every((v, t) => v === data.ref[t]), lostCount };
}

/** Bytes of W, bytes of x, threads and atomic updates for a split-K configuration. */
export function splitKTraffic(cfg: SplitKConfig) {
  return {
    wBytes: cfg.N * cfg.K * 4,
    xBytes: cfg.N * cfg.K * cfg.batch * 4,
    threads: cfg.N * cfg.S * cfg.warpWidth,
    atomics: cfg.S > 1 ? cfg.N * cfg.S * cfg.batch : 0,
  };
}

/** One sentence for a step, with the index arithmetic for one chosen lane. */
export function describeSKStep(run: SKRun, n: number, lane: number): string {
  const st = run.steps[n];
  const { cfg } = run;
  const { j, c, start } = warpPlace(cfg, st.warp);
  const ww = cfg.warpWidth;
  const who = tr(`Warp ${st.warp}`, `Warp ${st.warp}`);
  const where = cfg.S === 1
    ? tr(`${who} owns output j = ${st.warp}`, `${who} sở hữu đầu ra j = ${st.warp}`)
    : tr(
        `${who} owns output j = ${st.warp} ÷ ${cfg.S} = ${j} and chunk c = ${st.warp} mod ${cfg.S} = ${c}, which starts at c × ${cfg.K} ÷ ${cfg.S} = ${start}`,
        `${who} sở hữu đầu ra j = ${st.warp} ÷ ${cfg.S} = ${j} và chunk c = ${st.warp} mod ${cfg.S} = ${c}, bắt đầu tại c × ${cfg.K} ÷ ${cfg.S} = ${start}`,
      );
  const after = run.snaps[n + 1];
  const B = cfg.batch;
  const ys = (cells: number[]) => cells.map((cell) => `Y[${Math.floor(cell / cfg.N)}][${cell % cfg.N}] = ${after.Y[cell]}`).join(', ');
  switch (st.op.kind) {
    case 'load': {
      const live = st.ks!.filter((k) => k >= 0);
      const k = st.ks![lane];
      const laneTxt = k >= 0
        ? ` Lane ${lane}: W[${j}][${start} + ${lane} + ${st.op.strip} × ${ww}] = W[${j}][${k}] = ${run.data.W[j * cfg.K + k]}.`
        : tr(` Lane ${lane} has nothing to read in this strip.`, ` Lane ${lane} không có gì để đọc trong dải này.`);
      return tr(
        `${where}. Strip ${st.op.strip}: the lanes load W[${j}][${live[0]}] to W[${j}][${live[live.length - 1]}], one contiguous run, and the same columns of x for all ${B} batch rows.`,
        `${where}. Dải ${st.op.strip}: các lane nạp W[${j}][${live[0]}] đến W[${j}][${live[live.length - 1]}], một đoạn liên tiếp, và cùng các cột đó của x cho cả ${B} hàng của batch.`,
      ) + laneTxt;
    }
    case 'fma': {
      const k = st.ks![lane];
      const before = run.snaps[n].acc[st.warp];
      const laneTxt = k >= 0
        ? ` Lane ${lane}: ${Array.from({ length: B }, (_, b) => tr(`sum ${b} = ${before[b][lane]} + ${run.data.W[j * cfg.K + k]} × ${run.data.X[b * cfg.K + k]} = ${after.acc[st.warp][b][lane]}`, `tổng ${b} = ${before[b][lane]} + ${run.data.W[j * cfg.K + k]} × ${run.data.X[b * cfg.K + k]} = ${after.acc[st.warp][b][lane]}`)).join('; ')}.` // i18n-ignore: lane label and code notation; the sums inside are translated
        : '';
      return tr(
        `${where}. Each lane multiplies its weight by x for every batch row and adds the products to its ${B} running sums, in registers.`,
        `${where}. Mỗi lane nhân trọng số của mình với x cho mọi hàng của batch và cộng các tích vào ${B} tổng đang tích lũy của nó, trong thanh ghi.`,
      ) + laneTxt;
    }
    case 'shfl':
      return tr(
        `${who}: shuffle down by ${st.op.offset}. Lane l adds lane l + ${st.op.offset}'s sums to its own, through registers, not memory. After this step lanes 0 to ${st.op.offset! - 1} hold the partial totals.`,
        `${who}: shuffle xuống ${st.op.offset}. Lane l cộng các tổng của lane l + ${st.op.offset} vào tổng của mình, qua thanh ghi, không qua bộ nhớ. Sau bước này các lane 0 đến ${st.op.offset! - 1} giữ các tổng một phần.`,
      );
    case 'store':
      return tr(
        `${who}: lane 0 stores the warp's totals, ${ys(st.touched!)}. With S = 1 no other warp writes these cells, so a plain store is safe.`,
        `${who}: lane 0 store các tổng của warp, ${ys(st.touched!)}. Với S = 1, không warp nào khác ghi vào các ô này, nên một lệnh store thường là an toàn.`,
      );
    case 'atomic':
      return tr(
        `${who}: lane 0 adds the warp's totals with an atomic, which reads, adds and writes in one indivisible step: ${ys(st.touched!)}. These cells now hold chunk${after.parts[st.touched![0]].length > 1 ? 's' : ''} ${after.parts[st.touched![0]].join(', ')}.`,
        `${who}: lane 0 cộng các tổng của warp bằng một phép toán nguyên tử, vốn đọc, cộng và ghi trong một bước không thể chia cắt: ${ys(st.touched!)}. Các ô này giờ chứa chunk ${after.parts[st.touched![0]].join(', ')}.`,
      );
    case 'read':
      return tr(
        `${who}: plain +=, part 1 of 2. Lane 0 reads ${st.touched!.map((cell) => `Y[${Math.floor(cell / cfg.N)}][${cell % cfg.N}] = ${run.snaps[n].Y[cell]}`).join(', ')} into registers.`,
        `${who}: += thường, phần 1 trên 2. Lane 0 đọc ${st.touched!.map((cell) => `Y[${Math.floor(cell / cfg.N)}][${cell % cfg.N}] = ${run.snaps[n].Y[cell]}`).join(', ')} vào thanh ghi.`,
      );
    case 'write':
      return st.lost
        ? tr(
            `${who}: plain +=, part 2 of 2. Lane 0 writes its copy plus its totals: ${ys(st.touched!)}. Another warp wrote these cells after this warp read them, so chunk ${st.lost.join(', ')}'s contribution is overwritten and lost.`,
            `${who}: += thường, phần 2 trên 2. Lane 0 ghi bản sao của nó cộng với các tổng của nó: ${ys(st.touched!)}. Một warp khác đã ghi vào các ô này sau khi warp này đọc chúng, nên phần đóng góp của chunk ${st.lost.join(', ')} bị ghi đè và mất.`,
          )
        : tr(
            `${who}: plain +=, part 2 of 2. Lane 0 writes its copy plus its totals: ${ys(st.touched!)}. Nobody wrote in between, so nothing is lost this time.`,
            `${who}: += thường, phần 2 trên 2. Lane 0 ghi bản sao của nó cộng với các tổng của nó: ${ys(st.touched!)}. Không ai ghi xen vào giữa, nên lần này không có gì bị mất.`,
          );
  }
}

// ---------------------------------------------------------------- the book's tour (pp. 533–536)

export type Decomp = 'output' | 'input' | 'either';
export type Criterion = 'atomics' | 'parallelism' | 'mapping' | 'balance' | 'none';

export const DECOMP_LABEL: Record<Decomp, string> = {
  get output() {
    return tr('output-centric', 'hướng đầu ra');
  },
  get input() {
    return tr('input-centric', 'hướng đầu vào');
  },
  get either() {
    return tr('the same either way', 'cách nào cũng như nhau');
  },
};
export const CRITERION_LABEL: Record<Criterion, string> = {
  get atomics() {
    return tr('avoids atomics', 'tránh phép toán nguyên tử');
  },
  get parallelism() {
    return tr('exposes parallelism', 'bộc lộ tính song song');
  },
  get mapping() {
    return tr('inputs easy to find', 'dễ tìm đầu vào');
  },
  get balance() {
    return tr('load balance', 'cân bằng tải');
  },
  get none() {
    return tr('atomics not needed either way', 'cách nào cũng không cần phép toán nguyên tử');
  },
};

export interface TourRow {
  id: string;
  name: string;
  /** Chapter as this course numbers it. */
  chapter: string;
  decomp: Decomp;
  /** Criteria accepted as "what decided it". */
  accept: Criterion[];
  /** Would an input-centric version need atomics? 'idempotent': several writers, same value. */
  inputCentricAtomics: 'needed' | 'one-writer' | 'idempotent';
  pages: string;
  /** The reason, in our words. */
  why: string;
  /** A row of ours, not in the book. */
  ours?: boolean;
  /** The words that identify the row in notes/ch22.md's tour table (book rows only). */
  notesKey?: string;
}

export const TOUR: TourRow[] = [
  {
    id: 'matmul', get name() { return tr('Matrix multiplication', 'Phép nhân ma trận'); }, chapter: '3, 5, 15', decomp: 'output', accept: ['atomics'], inputCentricAtomics: 'needed', pages: '533–534',
    get why() {
      return tr(
        'A thread per output element, gathering in a register. Every other criterion agrees: plenty of outputs, each output\'s inputs are known (a row and a column), and every output costs the same.',
        'Một luồng cho mỗi phần tử đầu ra, gom (gather) trong một thanh ghi. Mọi tiêu chí khác đều đồng ý: rất nhiều đầu ra, đầu vào của mỗi đầu ra đã biết (một hàng và một cột), và mọi đầu ra tốn như nhau.',
      );
    },
    notesKey: 'matrix multiplication (Ch. 3, 5, 15)', // i18n-ignore: lookup key matched against notes/ch22.md
  },
  {
    id: 'conv', get name() { return tr('Convolution and stencil', 'Tích chập và stencil'); }, chapter: '7, 8', decomp: 'output', accept: ['atomics'], inputCentricAtomics: 'needed', pages: '533–534',
    get why() {
      return tr(
        'As for matrix multiplication: the input-centric version would loop over the outputs each input feeds and update them atomically, and nothing else favours it.',
        'Như với phép nhân ma trận: phiên bản hướng đầu vào sẽ lặp qua các đầu ra mà mỗi đầu vào đóng góp và cập nhật chúng bằng phép toán nguyên tử, và không có gì khác ủng hộ nó.',
      );
    },
    notesKey: 'convolution (Ch. 7), stencil (Ch. 8)', // i18n-ignore: lookup key matched against notes/ch22.md
  },
  {
    id: 'hist', get name() { return tr('Histogram', 'Histogram'); }, chapter: '9', decomp: 'input', accept: ['parallelism', 'mapping', 'balance'], inputCentricAtomics: 'needed', pages: '534',
    get why() {
      return tr(
        'Input-centric, atomics and all. A thread per bin would avoid them but lose on three counts: far fewer bins than inputs (parallelism), every bin thread must inspect every input (mapping), and bins get different counts (balance).',
        'Hướng đầu vào, chấp nhận cả phép toán nguyên tử. Một luồng cho mỗi bin sẽ tránh được chúng nhưng thua ở ba điểm: số bin ít hơn nhiều so với số đầu vào (tính song song), mỗi luồng của bin phải xem mọi đầu vào (ánh xạ), và các bin nhận số lượng khác nhau (cân bằng tải).',
      );
    },
    notesKey: 'Histogram (Ch. 9)', // i18n-ignore: lookup key matched against notes/ch22.md
  },
  {
    id: 'reduction', get name() { return tr('Reduction', 'Phép rút gọn'); }, chapter: '10', decomp: 'output', accept: ['atomics'], inputCentricAtomics: 'needed', pages: '534',
    get why() {
      return tr(
        'Judged one iteration at a time: each thread produces one element of the iteration\'s output by adding two of its inputs. Scattering instead would make several inputs update one output, which needs atomics.',
        'Xét từng vòng lặp một: mỗi luồng tạo ra một phần tử đầu ra của vòng lặp bằng cách cộng hai đầu vào của nó. Nếu scatter thay vào đó thì nhiều đầu vào sẽ cập nhật một đầu ra, việc này cần phép toán nguyên tử.',
      );
    },
    notesKey: 'Reduction (Ch. 10)', // i18n-ignore: lookup key matched against notes/ch22.md
  },
  {
    id: 'scan', get name() { return tr('Scan', 'Scan'); }, chapter: '11', decomp: 'output', accept: ['atomics'], inputCentricAtomics: 'needed', pages: '534',
    get why() {
      return tr(
        'Like reduction, output-centric per iteration to avoid atomics. (The book prints "Chapter 12" here; scan is chapter 11.)',
        'Giống phép rút gọn, hướng đầu ra theo từng vòng lặp để tránh phép toán nguyên tử. (Sách in "Chapter 12" ở đây; scan là chương 11.)',
      );
    },
    notesKey: 'scan (book says "Chapter 12"', // i18n-ignore: lookup key matched against notes/ch22.md
  },
  {
    id: 'filter', get name() { return tr('Filter (unstable and stable)', 'Lọc (không ổn định và ổn định)'); }, chapter: '12', decomp: 'input', accept: ['mapping'], inputCentricAtomics: 'one-writer', pages: '534–535',
    get why() {
      return tr(
        'A thread per input decides whether to keep it and where it goes. A thread per output slot would have to inspect many elements to learn which input lands in its slot. And the stable filter mixes: its scan is itself output-centric.',
        'Một luồng cho mỗi đầu vào quyết định có giữ nó không và nó đi đâu. Một luồng cho mỗi ô đầu ra sẽ phải xem nhiều phần tử để biết đầu vào nào rơi vào ô của mình. Và phép lọc ổn định thì pha trộn: bước scan của nó tự nó là hướng đầu ra.',
      );
    },
    notesKey: 'Unstable and stable filter (Ch. 12)', // i18n-ignore: lookup key matched against notes/ch22.md
  },
  {
    id: 'merge', get name() { return tr('Merge', 'Trộn'); }, chapter: '13', decomp: 'output', accept: ['balance'], inputCentricAtomics: 'one-writer', pages: '535',
    get why() {
      return tr(
        'Not atomics: every output receives exactly one input, so neither version needs them. Chunks of the output give every thread the same number of elements; chunks of one input would leave threads merging very different amounts (imbalance and divergence).',
        'Không phải vì phép toán nguyên tử: mỗi đầu ra nhận đúng một đầu vào, nên phiên bản nào cũng không cần chúng. Chia đầu ra thành các đoạn cho mọi luồng cùng số phần tử; chia theo một đầu vào sẽ khiến các luồng trộn những lượng rất khác nhau (mất cân bằng và phân kỳ).',
      );
    },
    notesKey: 'Merge (Ch. 13)', // i18n-ignore: lookup key matched against notes/ch22.md
  },
  {
    id: 'oddeven', get name() { return tr('Odd-even sort', 'Sắp xếp chẵn-lẻ'); }, chapter: '14', decomp: 'either', accept: ['none'], inputCentricAtomics: 'one-writer', pages: '533, 535',
    get why() {
      return tr(
        'Each thread reads a pair and writes the same pair. Inputs and outputs correspond so directly that the label does not matter: no gather, no scatter.',
        'Mỗi luồng đọc một cặp và ghi chính cặp đó. Đầu vào và đầu ra tương ứng trực tiếp đến mức cái nhãn không quan trọng: không gather, không scatter.',
      );
    },
    notesKey: 'Sorting (Ch. 14)', // i18n-ignore: lookup key matched against notes/ch22.md
  },
  {
    id: 'radix', get name() { return tr('One radix-sort iteration', 'Một vòng lặp radix sort'); }, chapter: '14', decomp: 'input', accept: ['mapping'], inputCentricAtomics: 'one-writer', pages: '535',
    get why() {
      return tr(
        'Each thread takes an input element and works out where it goes, like the filter it generalises. A thread per output position could not tell which key lands there without looking.',
        'Mỗi luồng lấy một phần tử đầu vào và tính xem nó đi đâu, giống phép lọc mà nó tổng quát hóa. Một luồng cho mỗi vị trí đầu ra không thể biết khóa nào rơi vào đó nếu không xem.',
      );
    },
    notesKey: 'Sorting (Ch. 14)', // i18n-ignore: lookup key matched against notes/ch22.md
  },
  {
    id: 'mergesort', get name() { return tr('One merge-sort level', 'Một mức merge sort'); }, chapter: '14', decomp: 'output', accept: ['balance'], inputCentricAtomics: 'one-writer', pages: '535',
    get why() {
      return tr(
        'Each level is mainly a merge, so it takes merge\'s decomposition, for merge\'s reason.',
        'Mỗi mức chủ yếu là một phép trộn, nên nó dùng cách phân rã của phép trộn, vì cùng lý do.',
      );
    },
    notesKey: 'Sorting (Ch. 14)', // i18n-ignore: lookup key matched against notes/ch22.md
  },
  {
    id: 'wavefront', get name() { return tr('Wavefront dynamic programming', 'Quy hoạch động theo wavefront'); }, chapter: '16', decomp: 'output', accept: ['atomics'], inputCentricAtomics: 'needed', pages: '535',
    get why() {
      return tr(
        'The thread for one cell gathers the earlier cells it depends on. Scattering each finished cell into its successors would need atomics, because several cells feed one successor.',
        'Luồng của một ô gom các ô trước đó mà nó phụ thuộc. Scatter mỗi ô đã xong vào các ô kế tiếp sẽ cần phép toán nguyên tử, vì nhiều ô cùng đóng góp vào một ô kế tiếp.',
      );
    },
    notesKey: 'Wavefront / dynamic programming (Ch. 16)', // i18n-ignore: lookup key matched against notes/ch22.md
  },
  {
    id: 'csr', get name() { return tr('SpMV with CSR, ELL or JDS', 'SpMV với CSR, ELL hoặc JDS'); }, chapter: '17', decomp: 'output', accept: ['atomics'], inputCentricAtomics: 'needed', pages: '535',
    get why() {
      return tr(
        'A thread per output element walks its row\'s non-zeros and gathers.',
        'Một luồng cho mỗi phần tử đầu ra duyệt các phần tử khác 0 trong hàng của nó và gom lại.',
      );
    },
    notesKey: 'SpMV/CSR, SpMV/ELL, SpMV/JDS (Ch. 17)', // i18n-ignore: lookup key matched against notes/ch22.md
  },
  {
    id: 'coo', get name() { return tr('SpMV with COO', 'SpMV với COO'); }, chapter: '17', decomp: 'input', accept: ['balance', 'parallelism', 'mapping'], inputCentricAtomics: 'needed', pages: '535–536',
    get why() {
      return tr(
        'A thread per non-zero, with an atomic per update: the atomics are the price of more parallelism and even load. A thread per row would have to search all the non-zeros for its own.',
        'Một luồng cho mỗi phần tử khác 0, với một phép toán nguyên tử cho mỗi lần cập nhật: phép toán nguyên tử là cái giá cho tính song song cao hơn và tải đều. Một luồng cho mỗi hàng sẽ phải tìm trong tất cả các phần tử khác 0 để ra phần của mình.',
      );
    },
    notesKey: 'SpMV/COO (Ch. 17)', // i18n-ignore: lookup key matched against notes/ch22.md
  },
  {
    id: 'csc', get name() { return tr('SpMV with CSC', 'SpMV với CSC'); }, chapter: '17', decomp: 'input', accept: ['mapping'], inputCentricAtomics: 'needed', pages: '536',
    get why() {
      return tr(
        'CSC stores columns, so a thread per input-vector element walks its column and scatters with atomics. A thread per row would have to scan every column to find its row\'s non-zeros.',
        'CSC lưu theo cột, nên một luồng cho mỗi phần tử của vector đầu vào duyệt cột của nó và scatter bằng phép toán nguyên tử. Một luồng cho mỗi hàng sẽ phải quét mọi cột để tìm các phần tử khác 0 của hàng mình.',
      );
    },
    notesKey: 'SpMV/CSC (Ch. 17)', // i18n-ignore: lookup key matched against notes/ch22.md
  },
  {
    id: 'bfs-push', get name() { return tr('BFS, vertex-centric push or edge-centric', 'BFS, push hướng đỉnh hoặc hướng cạnh'); }, chapter: '18', decomp: 'input', accept: ['parallelism', 'balance'], inputCentricAtomics: 'idempotent', pages: '536',
    get why() {
      return tr(
        'Threads own vertices or edges and write their neighbours\' levels. No atomics: every writer writes the same value, so the update is idempotent. What decides is parallelism and load balance, and those depend on the graph.',
        'Các luồng sở hữu đỉnh hoặc cạnh và ghi mức (level) của các đỉnh kề. Không cần phép toán nguyên tử: mọi luồng ghi đều ghi cùng một giá trị, nên phép cập nhật là lũy đẳng. Điều quyết định là tính song song và cân bằng tải, và những thứ đó phụ thuộc vào đồ thị.',
      );
    },
    notesKey: 'vertex-centric push and edge-centric', // i18n-ignore: lookup key matched against notes/ch22.md
  },
  {
    id: 'bfs-pull', get name() { return tr('BFS, vertex-centric pull', 'BFS, pull hướng đỉnh'); }, chapter: '18', decomp: 'output', accept: ['parallelism', 'balance'], inputCentricAtomics: 'idempotent', pages: '536',
    get why() {
      return tr(
        'Each thread updates only its own vertex\'s level. As for push, atomics are not the question; parallelism and balance are, and they depend on the graph.',
        'Mỗi luồng chỉ cập nhật mức của đỉnh của chính nó. Như với push, phép toán nguyên tử không phải là vấn đề; tính song song và cân bằng tải mới là vấn đề, và chúng phụ thuộc vào đồ thị.',
      );
    },
    notesKey: 'vertex-centric pull', // i18n-ignore: lookup key matched against notes/ch22.md
  },
  {
    id: 'epm', get name() { return tr('Electrostatic potential map', 'Bản đồ thế tĩnh điện'); }, chapter: '21', decomp: 'output', accept: ['atomics'], inputCentricAtomics: 'needed', pages: '536',
    get why() {
      return tr(
        'Grid-centric (a thread per grid point) gathers; atom-centric scatters into every grid point with atomics. Both expose plenty of parallelism, so atomics decide. For the cutoff version, binning solves the problem of finding each point\'s atoms.',
        'Hướng lưới (một luồng cho mỗi điểm lưới) thì gom; hướng nguyên tử thì scatter vào mọi điểm lưới bằng phép toán nguyên tử. Cả hai đều bộc lộ nhiều tính song song, nên phép toán nguyên tử là yếu tố quyết định. Với phiên bản có bán kính cắt (cutoff), việc chia bin giải quyết bài toán tìm các nguyên tử của mỗi điểm.',
      );
    },
    notesKey: 'Electrostatic potential map (Ch. 21)', // i18n-ignore: lookup key matched against notes/ch22.md
  },
  {
    id: 'embedding', get name() { return tr('Embedding lookup', 'Tra cứu embedding'); }, chapter: 'ours', decomp: 'output', accept: ['mapping'], inputCentricAtomics: 'one-writer', pages: '533 (criteria)', ours: true,
    get why() {
      return tr(
        'out[t][d] = table[token[t]][d]: the owner of an output knows its input from the token id. A thread per table row could scatter without atomics (each output has one source) but would have to search the tokens for its own.',
        'out[t][d] = table[token[t]][d]: chủ sở hữu của một đầu ra biết đầu vào của nó từ token id. Một luồng cho mỗi hàng của bảng có thể scatter mà không cần phép toán nguyên tử (mỗi đầu ra có một nguồn) nhưng sẽ phải tìm trong các token để ra phần của mình.',
      );
    },
  },
  {
    id: 'gemv', get name() { return tr('Decode GEMV, y = W x', 'GEMV khi decode, y = W x'); }, chapter: 'ours', decomp: 'output', accept: ['atomics'], inputCentricAtomics: 'needed', pages: '533–534 (as matrix multiplication)', ours: true,
    get why() {
      return tr(
        'A warp per output gathers in registers and reads W in contiguous runs. When there are too few outputs to keep the device busy, split-K trades some atomics for parallelism: lab 22.1.',
        'Một warp cho mỗi đầu ra gom trong thanh ghi và đọc W theo các đoạn liên tiếp. Khi có quá ít đầu ra để giữ thiết bị bận rộn, split-K đổi một ít phép toán nguyên tử lấy tính song song: lab 22.1.',
      );
    },
  },
  {
    id: 'softmax', get name() { return tr('Softmax normaliser (sum over a row)', 'Mẫu số chuẩn hóa softmax (tổng trên một hàng)'); }, chapter: 'ours', decomp: 'output', accept: ['atomics'], inputCentricAtomics: 'needed', pages: '534 (as reduction)', ours: true,
    get why() {
      return tr(
        'The sum of exponentials over a row is a reduction, so by the book\'s reasoning it is output-centric per iteration. Adding every element into one total with atomics would serialise on one address.',
        'Tổng các hàm mũ trên một hàng là một phép rút gọn, nên theo lập luận của sách, nó hướng đầu ra theo từng vòng lặp. Cộng mọi phần tử vào một tổng duy nhất bằng phép toán nguyên tử sẽ tuần tự hóa trên một địa chỉ.',
      );
    },
  },
  {
    id: 'tokencounts', get name() { return tr('Repetition-penalty token counts (lab 9.3)', 'Đếm token cho repetition penalty (lab 9.3)'); }, chapter: 'ours', decomp: 'input', accept: ['mapping'], inputCentricAtomics: 'needed', pages: '534 (as histogram)', ours: true,
    get why() {
      return tr(
        'A histogram over a vocabulary. Here the bins outnumber the tokens, so the parallelism argument flips; but a thread per vocabulary id would still read every token to find its own. Input-centric, atomics and all.',
        'Một histogram trên một bộ từ vựng. Ở đây số bin nhiều hơn số token, nên lập luận về tính song song bị đảo ngược; nhưng một luồng cho mỗi id trong bộ từ vựng vẫn sẽ phải đọc mọi token để tìm phần của mình. Hướng đầu vào, chấp nhận cả phép toán nguyên tử.',
      );
    },
  },
];

/** Grade one card: decomposition right, and the criterion one of those accepted. */
export function gradeCard(row: TourRow, decomp: Decomp, criterion: Criterion): { decomp: boolean; criterion: boolean } {
  return { decomp: decomp === row.decomp, criterion: row.accept.includes(criterion) };
}
