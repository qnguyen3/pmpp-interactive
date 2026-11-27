// An executable model of the flash-attention forward kernel of book §20.5 (Figs. 20.8–20.16,
// pp. 495–504) as lab 20.2 corrects it, at a size that fits on a screen. Section 20.5's
// stepper "Step one block" and the scene "One block's journey through the flash kernel"
// run on it.
//
// What it models, per thread:
//   - Fig. 20.9's two loops: a block takes a panel of B_r rows of Q (block index i), walks
//     the K/V tiles j, and inside each tile every warp walks its B_r_warp rows (ii).
//   - Fig. 20.10's layout: lane L of a warp holds elements L, L + W, L + 2W, ... of each of
//     its Q rows and O rows (W = lanes per warp); m and D are per-lane copies.
//   - Fig. 20.11: thread t loads element dd = t (stride = threads) of each K and V row of
//     the tile; K is stored transposed in KT_j (no padding at this size).
//   - Fig. 20.12: each lane computes the scores of its columns of the tile (jj = L, L + W,
//     ...), receiving Q element dd from its owner lane; masking by index (columns past the
//     row, and columns ≥ N); the warp max reaches every lane (the labs' butterfly, facts A2).
//   - Fig. 20.14: update_m_and_D, which returns the old max.
//   - Fig. 20.15: P overwrites S in place; the warp sum reaches every lane and joins D.
//   - The __syncwarp() the book leaves out (facts A3), then Fig. 20.13's compute_O: rescale,
//     then add the P segment times the V tile, reading P elements other lanes wrote.
//   - Fig. 20.16's store_O with L = m + log D (FlashAttention-2's logsumexp, facts A7), and
//     only rows < N stored; panels and tiles use the ceiling, zero-filled past N.
//   - The two __syncthreads() of Fig. 20.9.
//   - Defects, each switchable: no __syncwarp; update_m_and_D returning the new max; no
//     barrier after the warp loop; the running max starting at minus infinity; tiles right
//     to left.
//
// Schedules. 'lockstep' lanes execute each function together (one event per warp);
// 'apart' lanes run one at a time, lane 0 first, each as far as it can before its next
// warp or block barrier. 'together' warps take turns event by event; 'ahead' lets warp 0
// run as far as it can before warp 1. A shuffle or warp reduction hands every lane the
// values the other lanes compute for that same call, whenever they get there: register
// exchange through *_sync intrinsics is defined, but it "does not guarantee any memory
// ordering" (facts A3), so in 'apart' it does not hold lanes back. That is a deliberately
// pessimistic reading of what the rules permit, not a claim about what a GPU does.
//
// Left out: time, caches, banks, warp width beyond the toy's, float32 (arithmetic is
// double precision, so only structural errors show), and uninitialised shared memory is
// NaN (the worst case).

import { tr } from '../lib/locale.ts';

export type InitMax = 'sentinel' | 'neg-inf';
export type TileOrder = 'ltr' | 'rtl';

export interface KConfig {
  n: number;
  d: number;
  /** Lanes per warp (the toy uses 4; the book 32). */
  lanes: number;
  warps: number;
  brWarp: number;
  bc: number;
  init: InitMax;
  order: TileOrder;
  /** __syncwarp() between compute_P_and_update_D and compute_O (missing in the book). */
  syncwarp: boolean;
  /** __syncthreads() after the warp loop (Fig. 20.9 line 39). */
  barrierAfter: boolean;
  /** Defect: update_m_and_D returns the new max instead of the old one. */
  returnsNewMax: boolean;
}

export type LaneSchedule = 'lockstep' | 'apart';
export type WarpSchedule = 'together' | 'ahead';

export const LANE_LABEL: Record<LaneSchedule, string> = {
  get lockstep() {
    return tr('lanes in lockstep', 'các lane chạy đồng bộ từng bước (lockstep)');
  },
  get apart() {
    return tr('lanes run apart between warp barriers', 'các lane chạy lệch nhau giữa các rào chắn warp');
  },
};
export const WARP_LABEL: Record<WarpSchedule, string> = {
  get together() {
    return tr('warps take turns', 'các warp chạy lần lượt');
  },
  get ahead() {
    return tr('warp 0 runs ahead', 'warp 0 chạy trước');
  },
};

/** The corrected kernel at the toy size: 12 rows, d = 8, 2 warps of 4 lanes, 2 rows each, B_c = 4. */
export const TOY: KConfig = {
  n: 12,
  d: 8,
  lanes: 4,
  warps: 2,
  brWarp: 2,
  bc: 4,
  init: 'sentinel',
  order: 'ltr',
  syncwarp: true,
  barrierAfter: true,
  returnsNewMax: false,
};

export type DefectId = 'noSyncwarp' | 'newMax' | 'noBarrierAfter' | 'negInfRtl';
export const DEFECTS: { id: DefectId; label: string; apply: Partial<KConfig> }[] = [
  { id: 'noSyncwarp', get label() { return tr('no __syncwarp before compute_O', 'không có __syncwarp trước compute_O'); }, apply: { syncwarp: false } },
  { id: 'newMax', get label() { return tr('update_m_and_D returns the new max', 'update_m_and_D trả về max mới'); }, apply: { returnsNewMax: true } },
  { id: 'noBarrierAfter', get label() { return tr('no barrier after the warp loop', 'không có rào chắn sau vòng lặp warp'); }, apply: { barrierAfter: false } },
  { id: 'negInfRtl', get label() { return tr('start m at minus infinity and go right to left', 'khởi đầu m bằng âm vô cùng và đi từ phải sang trái'); }, apply: { init: 'neg-inf', order: 'rtl' } },
];
export function withDefects(base: KConfig, on: DefectId[]): KConfig {
  return DEFECTS.filter((x) => on.includes(x.id)).reduce((c, x) => ({ ...c, ...x.apply }), base);
}

const SENTINEL = Math.fround(-3.4028234663852886e38 / 2);
export const startMax = (init: InitMax) => (init === 'neg-inf' ? -Infinity : SENTINEL);

// ---------------------------------------------------------------- problem and reference
export interface KProblem {
  n: number;
  d: number;
  Q: number[];
  K: number[];
  V: number[];
  scaling: number;
  /** Exact causal attention, row-major n × d. */
  O: number[];
  /** Exact logsumexp per row. */
  L: number[];
}

function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeProblem(n: number, d: number, seed = 20): KProblem {
  const r = mulberry(seed);
  const gen = () => Array.from({ length: n * d }, () => Math.round((r() * 3 - 1.5) * 10) / 10);
  const Q = gen();
  const K = gen();
  const V = gen().map((x) => Math.round(x * 6.67) / 10);
  const scaling = 1 / Math.sqrt(d);
  const ref = reference(Q, K, V, n, d, scaling);
  return { n, d, Q, K, V, scaling, ...ref };
}

export function rowScores(Q: number[], K: number[], n: number, d: number, scaling: number, row: number): number[] {
  return Array.from({ length: n }, (_, j) => {
    let s = 0;
    for (let k = 0; k < d; k++) s += Q[row * d + k] * K[j * d + k];
    return s * scaling;
  });
}

export function reference(Q: number[], K: number[], V: number[], n: number, d: number, scaling: number, upto?: (row: number) => number[]) {
  const O = new Array(n * d).fill(0);
  const L = new Array(n).fill(0);
  for (let r = 0; r < n; r++) {
    const s = rowScores(Q, K, n, d, scaling, r);
    const cols = upto ? upto(r) : Array.from({ length: r + 1 }, (_, j) => j);
    const m = Math.max(...cols.map((j) => s[j]));
    let D = 0;
    for (const j of cols) D += Math.exp(s[j] - m);
    for (const j of cols) for (let c = 0; c < d; c++) O[r * d + c] += (Math.exp(s[j] - m) * V[j * d + c]) / D;
    L[r] = m + Math.log(D);
  }
  return { O, L };
}

/** Exact attention of one row over a subset of its visible columns (null if the subset is empty). */
export function exactRow(prob: KProblem, row: number, cols: number[]): number[] | null {
  const vis = cols.filter((j) => j <= row && j < prob.n);
  if (!vis.length) return null;
  const s = rowScores(prob.Q, prob.K, prob.n, prob.d, prob.scaling, row);
  const m = Math.max(...vis.map((j) => s[j]));
  let D = 0;
  const o = new Array(prob.d).fill(0);
  for (const j of vis) {
    const e = Math.exp(s[j] - m);
    D += e;
    for (let c = 0; c < prob.d; c++) o[c] += e * prob.V[j * prob.d + c];
  }
  return o.map((x) => x / D);
}

// ---------------------------------------------------------------- derived sizes
export const br = (c: KConfig) => c.warps * c.brWarp;
export const threadsOf = (c: KConfig) => c.warps * c.lanes;
export const dSize = (c: KConfig) => c.d / c.lanes;
export const panels = (c: KConfig) => Math.ceil(c.n / br(c));
export const tiles = (c: KConfig) => Math.ceil(c.n / c.bc);
export const tileSeq = (c: KConfig) => {
  const t = Array.from({ length: tiles(c) }, (_, j) => j);
  return c.order === 'rtl' ? t.reverse() : t;
};
/** The columns jj of a tile that lane L owns: L, L + W, ... (none when B_c ≤ L). */
export const laneCols = (c: KConfig, lane: number) => {
  const out: number[] = [];
  for (let jj = lane; jj < c.bc; jj += c.lanes) out.push(jj);
  return out;
};

// ---------------------------------------------------------------- the trace
export type Fn =
  | 'load_Q'
  | 'load_KT_and_V'
  | 'compute_S_and_max'
  | 'update_m_and_D'
  | 'compute_P_and_update_D'
  | 'compute_O'
  | 'store_O';
export type Op = Fn | 'syncthreads' | 'syncwarp' | 'release_syncthreads' | 'release_syncwarp';

export type SlotKind = 'none' | 'S' | 'P';

export interface Touch {
  /** Q, K, V, O: global; KT, S, Vj: shared; reg: the executing lanes' registers. */
  array: 'Q' | 'K' | 'V' | 'O' | 'KT' | 'S' | 'Vj';
  cells: number[];
  write: boolean;
}

export interface KViolation {
  kind: 'stale-P' | 'kt-tile' | 'v-tile';
  message: string;
}

export interface KStep {
  index: number;
  /** -1 for a block barrier opening. */
  warp: number;
  lanes: number[];
  op: Op;
  /** For barriers, which one: after the loads, or after the warp loop. */
  barrier?: 'load' | 'end';
  tile: number | null;
  ii: number | null;
  text: string;
  touch: Touch[];
  violation?: KViolation;
}

export interface LaneRegs {
  Q: number[][];
  O: number[][];
  m: number[];
  D: number[];
}

export interface Snapshot {
  KT: number[];
  KTtile: number[];
  S: number[];
  Skind: SlotKind[];
  Stile: number[];
  Vj: number[];
  Vtile: number[];
  regs: LaneRegs[];
  /** Per thread: next plan position, and what it waits at (or null). */
  pc: number[];
  waiting: ('syncthreads' | 'syncwarp' | null)[];
  done: boolean[];
  /** Per local row, per lane: the tiles that lane has merged into its part of O. */
  merged: number[][][];
  /** Rows this block has stored: global row -> O row and L. */
  stored: Record<number, { O: number[]; L: number }>;
}

export interface BlockTrace {
  config: KConfig;
  panel: number;
  lanes: LaneSchedule;
  warpsOrder: WarpSchedule;
  steps: KStep[];
  /** snaps[k] is the state after the first k steps (snaps[0] = before anything ran). */
  snaps: Snapshot[];
  violations: (KViolation & { step: number })[];
  stored: Record<number, { O: number[]; L: number }>;
}

type PlanOp =
  | { op: 'load_Q' }
  | { op: 'load_KT_and_V'; tile: number }
  | { op: 'syncthreads'; tile: number; barrier: 'load' | 'end' }
  | { op: 'compute_S_and_max' | 'update_m_and_D' | 'compute_P_and_update_D' | 'syncwarp' | 'compute_O'; tile: number; ii: number }
  | { op: 'store_O' };

function buildPlan(c: KConfig): PlanOp[] {
  const p: PlanOp[] = [{ op: 'load_Q' }];
  for (const tile of tileSeq(c)) {
    p.push({ op: 'load_KT_and_V', tile }, { op: 'syncthreads', tile, barrier: 'load' });
    for (let ii = 0; ii < c.brWarp; ii++) {
      p.push({ op: 'compute_S_and_max', tile, ii }, { op: 'update_m_and_D', tile, ii }, { op: 'compute_P_and_update_D', tile, ii });
      if (c.syncwarp) p.push({ op: 'syncwarp', tile, ii });
      p.push({ op: 'compute_O', tile, ii });
    }
    if (c.barrierAfter) p.push({ op: 'syncthreads', tile, barrier: 'end' });
  }
  p.push({ op: 'store_O' });
  return p;
}

/** A number as the stepper prints it. */
export function fmtv(x: number): string {
  if (Number.isNaN(x)) return 'NaN';
  if (x === -Infinity) return '−∞';
  if (x === Infinity) return '∞';
  if (x <= -1e37) return '−FLT_MAX/2';
  const r = Math.round(x * 1000) / 1000;
  return (Object.is(r, -0) ? 0 : r).toString().replace('-', '−');
}
const span = (xs: number[]) => (xs.length === 1 ? String(xs[0]) : xs.length === 2 ? tr(`${xs[0]} and ${xs[1]}`, `${xs[0]} và ${xs[1]}`) : tr(`${xs[0]} to ${xs[xs.length - 1]}`, `${xs[0]} đến ${xs[xs.length - 1]}`));

/** Runs one block (Q panel `panel`) under the given schedules and records every event. */
export function runBlock(
  c: KConfig,
  prob: KProblem,
  panel: number,
  laneSched: LaneSchedule,
  warpSched: WarpSchedule,
  opts: { record?: boolean } = {},
): BlockTrace {
  const record = opts.record !== false;
  const W = c.lanes;
  const BR = br(c);
  const T = threadsOf(c);
  const DS = dSize(c);
  const start = startMax(c.init);
  const plan = buildPlan(c);

  const KT = new Array(c.d * c.bc).fill(NaN);
  const KTtile = new Array(c.d * c.bc).fill(-1);
  const S = new Array(BR * c.bc).fill(NaN);
  const Skind: SlotKind[] = new Array(BR * c.bc).fill('none');
  const Stile = new Array(BR * c.bc).fill(-1);
  const Vj = new Array(c.bc * c.d).fill(NaN);
  const Vtile = new Array(c.bc * c.d).fill(-1);
  const regs: LaneRegs[] = Array.from({ length: T }, () => ({
    Q: Array.from({ length: c.brWarp }, () => new Array(DS).fill(0)),
    O: Array.from({ length: c.brWarp }, () => new Array(DS).fill(0)),
    m: new Array(c.brWarp).fill(start),
    D: new Array(c.brWarp).fill(0),
  }));
  const curMax: number[][] = Array.from({ length: T }, () => new Array(c.brWarp).fill(start));
  const lastM: number[][] = Array.from({ length: T }, () => new Array(c.brWarp).fill(start));
  const pc = new Array(T).fill(0);
  const waiting: ('syncthreads' | 'syncwarp' | null)[] = new Array(T).fill(null);
  const merged: number[][][] = Array.from({ length: BR }, () => Array.from({ length: W }, () => [] as number[]));
  const stored: Record<number, { O: number[]; L: number }> = {};
  const steps: KStep[] = [];
  const snaps: Snapshot[] = [];
  const violations: BlockTrace['violations'] = [];

  // Shuffle collectives: the values every lane contributes to one call, fixed at the first
  // lane that reaches it (see the header: register exchange is defined; memory order is not).
  const colls = new Map<string, { s: (number | null)[]; max: number; sum?: number }>();

  const rowOf = (w: number, ii: number) => panel * BR + w * c.brWarp + ii;
  const masked = (row: number, col: number) => col > row || col >= c.n;
  const live = (t: number) => pc[t] < plan.length;

  const snap = (): Snapshot => ({
    KT: KT.slice(),
    KTtile: KTtile.slice(),
    S: S.slice(),
    Skind: Skind.slice(),
    Stile: Stile.slice(),
    Vj: Vj.slice(),
    Vtile: Vtile.slice(),
    regs: regs.map((r) => ({ Q: r.Q.map((x) => x.slice()), O: r.O.map((x) => x.slice()), m: r.m.slice(), D: r.D.slice() })),
    pc: pc.slice(),
    waiting: waiting.slice(),
    done: pc.map((p) => p >= plan.length),
    merged: merged.map((x) => x.map((y) => y.slice())),
    stored: Object.fromEntries(Object.entries(stored).map(([k, v]) => [k, { O: v.O.slice(), L: v.L }])),
  });
  if (record) snaps.push(snap());

  /** Scores of one row against the KT tile as it is in shared memory now; null where masked. */
  const scoresNow = (w: number, ii: number, tile: number): (number | null)[] =>
    Array.from({ length: c.bc }, (_, jj) => {
      const row = rowOf(w, ii);
      const col = tile * c.bc + jj;
      if (masked(row, col)) return null;
      let acc = 0;
      for (let dd = 0; dd < c.d; dd++) acc += regs[w * W + (dd % W)].Q[ii][Math.floor(dd / W)] * KT[dd * c.bc + jj];
      return prob.scaling * acc;
    });

  const flag = (s: KStep, v: KViolation) => {
    if (!s.violation) s.violation = v;
    violations.push({ ...v, step: s.index });
  };

  /** One function by a set of lanes of one warp (all lanes in lockstep, one lane otherwise). */
  const exec = (w: number, lanes: number[]) => {
    const t0 = w * W + lanes[0];
    const p = plan[pc[t0]];
    const tile = 'tile' in p ? p.tile : null;
    const ii = 'ii' in p ? p.ii : null;
    const s: KStep = { index: steps.length, warp: w, lanes: lanes.slice(), op: p.op, tile, ii, text: '', touch: [] };
    if (p.op === 'syncthreads') s.barrier = p.barrier;
    steps.push(s);
    const who = lanes.length === W ? tr(`Warp ${w}`, `Warp ${w}`) : tr(`Lane ${lanes[0]} of warp ${w}`, `Lane ${lanes[0]} của warp ${w}`);
    const lr = ii == null ? 0 : w * c.brWarp + ii;
    const row = ii == null ? 0 : rowOf(w, ii);

    switch (p.op) {
      case 'load_Q': {
        const cells: number[] = [];
        for (const L of lanes) {
          const t = w * W + L;
          for (let i2 = 0; i2 < c.brWarp; i2++)
            for (let sl = 0; sl < DS; sl++) {
              const r = rowOf(w, i2);
              const dd = L + sl * W;
              regs[t].Q[i2][sl] = r < c.n ? prob.Q[r * c.d + dd] : 0;
              if (r < c.n) cells.push(r * c.d + dd);
            }
        }
        const rows = Array.from({ length: c.brWarp }, (_, i2) => rowOf(w, i2)).filter((r) => r < c.n);
        s.touch.push({ array: 'Q', cells, write: false });
        s.text =
          lanes.length === W
            ? tr(
                `${who} loads its rows of Q (${rows.length ? 'rows ' + span(rows) : 'none: past N, zero-filled'}) into registers. Lane L takes columns L and L + ${W}${DS > 2 ? ', ...' : ''} of each row, so the ${W} lanes read ${W} neighbouring elements at a time.`,
                `${who} nạp các hàng Q của nó (${rows.length ? 'hàng ' + span(rows) : 'không có: đã quá N, điền số 0'}) vào thanh ghi. Lane L lấy các cột L và L + ${W}${DS > 2 ? ', ...' : ''} của mỗi hàng, nên ${W} lane đọc ${W} phần tử liền kề mỗi lần.`,
              )
            : tr(
                `${who} loads Q columns ${laneElementsList(lanes[0], c)} of rows ${span(Array.from({ length: c.brWarp }, (_, i2) => rowOf(w, i2)))} into its registers${rows.length < c.brWarp ? ' (rows past N are zero-filled)' : ''}.`,
                `${who} nạp các cột Q ${laneElementsList(lanes[0], c)} của các hàng ${span(Array.from({ length: c.brWarp }, (_, i2) => rowOf(w, i2)))} vào thanh ghi của nó${rows.length < c.brWarp ? ' (các hàng quá N được điền số 0)' : ''}.`,
              );
        break;
      }
      case 'load_KT_and_V': {
        const kt: number[] = [];
        const vj: number[] = [];
        const gk: number[] = [];
        for (const L of lanes) {
          const t = w * W + L;
          for (let jj = 0; jj < c.bc; jj++)
            for (let dd = t; dd < c.d; dd += T) {
              const kr = p.tile * c.bc + jj;
              const ok = kr < c.n;
              KT[dd * c.bc + jj] = ok ? prob.K[kr * c.d + dd] : 0;
              KTtile[dd * c.bc + jj] = p.tile;
              Vj[jj * c.d + dd] = ok ? prob.V[kr * c.d + dd] : 0;
              Vtile[jj * c.d + dd] = p.tile;
              kt.push(dd * c.bc + jj);
              vj.push(jj * c.d + dd);
              if (ok) gk.push(kr * c.d + dd);
            }
        }
        s.touch.push({ array: 'K', cells: gk, write: false }, { array: 'V', cells: gk, write: false });
        s.touch.push({ array: 'KT', cells: kt, write: true }, { array: 'Vj', cells: vj, write: true });
        const ts = lanes.map((L) => w * W + L);
        const dds = ts.filter((t) => t < c.d);
        const k0 = p.tile * c.bc;
        s.text = dds.length
          ? tr(
              `${lanes.length === W ? `${who} (threads ${span(ts)})` : `Thread ${ts[0]}`} copies column${dds.length > 1 ? 's' : ''} ${span(dds)} of K rows ${span(range(k0, Math.min(k0 + c.bc, c.n)))} into KT_j, transposed (K row jj becomes column jj), and the same part of V into V_j: tile ${p.tile}.${k0 + c.bc > c.n ? ' Rows past N are zero-filled.' : ''}`,
              `${lanes.length === W ? `${who} (các luồng ${span(ts)})` : `Luồng ${ts[0]}`} sao chép ${dds.length > 1 ? 'các cột' : 'cột'} ${span(dds)} của các hàng K ${span(range(k0, Math.min(k0 + c.bc, c.n)))} vào KT_j, có chuyển vị (hàng jj của K thành cột jj), và phần tương ứng của V vào V_j: ô ${p.tile}.${k0 + c.bc > c.n ? ' Các hàng quá N được điền số 0.' : ''}`,
            )
          : tr(`${who}: no thread here has a column to load (d = ${c.d}).`, `${who}: không luồng nào ở đây có cột để nạp (d = ${c.d}).`);
        break;
      }
      case 'syncthreads':
      case 'syncwarp': {
        for (const L of lanes) waiting[w * W + L] = p.op;
        s.text =
          p.op === 'syncwarp'
            ? tr(`${who} reaches __syncwarp() and waits for the other lanes of its warp.`, `${who} đến __syncwarp() và chờ các lane khác của warp mình.`)
            : tr(
                `${who} reaches the block barrier ${s.barrier === 'load' ? 'after the loads' : 'after the warp loop'} and waits.`,
                `${who} đến rào chắn của khối ${s.barrier === 'load' ? 'sau các lần nạp' : 'sau vòng lặp warp'} và chờ.`,
              );
        return; // the pc moves when the barrier opens
      }
      case 'compute_S_and_max': {
        const key = `${w}:${p.tile}:${p.ii}`;
        let co = colls.get(key);
        if (!co) {
          const sc = scoresNow(w, p.ii, p.tile);
          co = { s: sc, max: sc.reduce<number>((a, x) => (x == null ? a : Math.max(a, x)), start) };
          colls.set(key, co);
        }
        const written: number[] = [];
        const ktCells: number[] = [];
        const shown: string[] = [];
        let maskedCols: number[] = [];
        for (const L of lanes) {
          const t = w * W + L;
          for (const jj of laneCols(c, L)) {
            const col = p.tile * c.bc + jj;
            let v: number;
            if (masked(row, col)) {
              v = start;
              maskedCols.push(col);
            } else {
              let acc = 0;
              for (let dd = 0; dd < c.d; dd++) {
                const at = dd * c.bc + jj;
                ktCells.push(at);
                if (KTtile[at] !== p.tile)
                  flag(s, {
                    kind: 'kt-tile',
                    message: tr(
                      `lane ${L} read KT_j[${dd}][${jj}] for tile ${p.tile}, but it holds ${KTtile[at] < 0 ? 'nothing yet' : `tile ${KTtile[at]}'s value: another warp had already loaded the next tile over it`}`,
                      `lane ${L} đọc KT_j[${dd}][${jj}] cho ô ${p.tile}, nhưng nó đang chứa ${KTtile[at] < 0 ? 'chưa có gì' : `giá trị của ô ${KTtile[at]}: một warp khác đã nạp ô kế tiếp đè lên nó`}`,
                    ),
                  });
                acc += regs[w * W + (dd % W)].Q[p.ii][Math.floor(dd / W)] * KT[at];
              }
              v = prob.scaling * acc;
              shown.push(tr(`col ${col}: ${fmtv(v)}`, `cột ${col}: ${fmtv(v)}`));
            }
            S[lr * c.bc + jj] = v;
            Skind[lr * c.bc + jj] = 'S';
            Stile[lr * c.bc + jj] = p.tile;
            written.push(lr * c.bc + jj);
          }
          curMax[t][p.ii] = co.max;
        }
        maskedCols = [...new Set(maskedCols)].sort((a, b) => a - b);
        s.touch.push({ array: 'KT', cells: ktCells, write: false }, { array: 'S', cells: written, write: true });
        const noCol = lanes.filter((L) => !laneCols(c, L).length);
        s.text =
          tr(`${who}, row ${row}, tile ${p.tile}: `, `${who}, hàng ${row}, ô ${p.tile}: `) +
          (written.length
            ? tr(
                `${lanes.length === W ? 'each lane' : 'it'} builds ${lanes.length === W ? 'its column’s' : 'its'} score from ${c.d} Q elements, each broadcast by its owner lane with a shuffle, then scales it. `,
                `${lanes.length === W ? 'mỗi lane' : 'nó'} tạo điểm số ${lanes.length === W ? 'của cột mình' : 'của nó'} từ ${c.d} phần tử Q, mỗi phần tử do lane sở hữu nó broadcast bằng một shuffle, rồi nhân tỉ lệ. `,
              ) +
              (shown.length ? tr(`Scores ${shown.join(', ')}. `, `Điểm số ${shown.join(', ')}. `) : '') +
              (maskedCols.length
                ? tr(
                    `Column${maskedCols.length > 1 ? 's' : ''} ${span(maskedCols)} ${maskedCols.length > 1 ? 'are' : 'is'} masked by index. `,
                    `${maskedCols.length > 1 ? 'Các cột' : 'Cột'} ${span(maskedCols)} bị che (mask) theo chỉ số. `,
                  )
                : '')
            : `${noCol.length ? tr('no column of this tile belongs to it; it brings the identity to the reduction. ', 'không cột nào của ô này thuộc về nó; nó mang phần tử đơn vị (identity) vào phép rút gọn. ') : ''}`) +
          tr(`Warp max ${fmtv(co.max)}, now in every lane.`, `Giá trị max của warp ${fmtv(co.max)}, nay có ở mọi lane.`);
        break;
      }
      case 'update_m_and_D': {
        let rose = false;
        let before = 0;
        let factor = 1;
        let mOld = 0;
        for (const L of lanes) {
          const t = w * W + L;
          const r = regs[t];
          mOld = r.m[p.ii];
          before = r.D[p.ii];
          if (r.m[p.ii] < curMax[t][p.ii]) {
            r.m[p.ii] = curMax[t][p.ii];
            factor = Math.exp(mOld - r.m[p.ii]);
            r.D[p.ii] *= factor;
            rose = true;
          }
          lastM[t][p.ii] = c.returnsNewMax ? r.m[p.ii] : mOld;
        }
        const r0 = regs[w * W + lanes[0]];
        s.text =
          tr(`${who}, row ${row}: `, `${who}, hàng ${row}: `) +
          (rose
            ? tr(
                `the max rises from ${fmtv(mOld)} to ${fmtv(r0.m[p.ii])}, so D is multiplied by e^(${fmtv(mOld)} − ${fmtv(r0.m[p.ii])}) = ${fmtv(factor)}: ${fmtv(before)} → ${fmtv(r0.D[p.ii])}. `,
                `max tăng từ ${fmtv(mOld)} lên ${fmtv(r0.m[p.ii])}, nên D được nhân với e^(${fmtv(mOld)} − ${fmtv(r0.m[p.ii])}) = ${fmtv(factor)}: ${fmtv(before)} → ${fmtv(r0.D[p.ii])}. `,
              )
            : tr(`the max stays ${fmtv(r0.m[p.ii])}; D stays ${fmtv(r0.D[p.ii])}. `, `max giữ nguyên ${fmtv(r0.m[p.ii])}; D giữ nguyên ${fmtv(r0.D[p.ii])}. `)) +
          (c.returnsNewMax
            ? tr(`It returns the new max ${fmtv(r0.m[p.ii])} (the defect).`, `Nó trả về max mới ${fmtv(r0.m[p.ii])} (lỗi cài sẵn).`)
            : tr(`It returns the old max ${fmtv(mOld)} for compute_O.`, `Nó trả về max cũ ${fmtv(mOld)} cho compute_O.`));
        break;
      }
      case 'compute_P_and_update_D': {
        const key = `${w}:${p.tile}:${p.ii}`;
        const co = colls.get(key)!;
        const m0 = regs[w * W + lanes[0]].m[p.ii];
        if (co.sum === undefined)
          co.sum = co.s.reduce<number>((a, x) => (x == null ? a : a + Math.exp(x - m0)), 0);
        const written: number[] = [];
        const ps: string[] = [];
        for (const L of lanes) {
          const t = w * W + L;
          for (const jj of laneCols(c, L)) {
            const at = lr * c.bc + jj;
            const col = p.tile * c.bc + jj;
            const v = masked(row, col) ? 0 : Math.exp(S[at] - regs[t].m[p.ii]);
            S[at] = v;
            Skind[at] = 'P';
            Stile[at] = p.tile;
            written.push(at);
            ps.push(fmtv(v));
          }
          regs[t].D[p.ii] += co.sum;
        }
        s.touch.push({ array: 'S', cells: written, write: true });
        s.text =
          tr(`${who}, row ${row}: `, `${who}, hàng ${row}: `) +
          (written.length
            ? tr(
                `overwrites ${written.length > 1 ? 'the scores' : 'its score'} in place with e^(s − m) = ${ps.length > 1 ? '[' + ps.join(', ') + ']' : ps[0]}; masked columns get 0. `,
                `ghi đè ${written.length > 1 ? 'các điểm số' : 'điểm số của nó'} tại chỗ bằng e^(s − m) = ${ps.length > 1 ? '[' + ps.join(', ') + ']' : ps[0]}; các cột bị che nhận 0. `,
              )
            : '') +
          tr(`Warp sum ${fmtv(co.sum)} joins D: ${fmtv(regs[w * W + lanes[0]].D[p.ii])}.`, `Tổng của warp ${fmtv(co.sum)} được cộng vào D: ${fmtv(regs[w * W + lanes[0]].D[p.ii])}.`);
        break;
      }
      case 'compute_O': {
        const sCells: number[] = [];
        const vCells: number[] = [];
        let factor = 1;
        for (const L of lanes) {
          const t = w * W + L;
          const r = regs[t];
          factor = Math.exp(lastM[t][p.ii] - r.m[p.ii]);
          for (let sl = 0; sl < DS; sl++) {
            const dd = L + sl * W;
            let acc = 0;
            for (let jj = 0; jj < c.bc; jj++) {
              const at = lr * c.bc + jj;
              const va = jj * c.d + dd;
              sCells.push(at);
              vCells.push(va);
              if (!(Skind[at] === 'P' && Stile[at] === p.tile)) {
                const owner = jj % W;
                flag(s, {
                  kind: 'stale-P',
                  message:
                    Skind[at] === 'none'
                      ? tr(
                          `lane ${L} read P[${row}][col ${p.tile * c.bc + jj}] before lane ${owner} had written anything there: the slot is uninitialised`,
                          `lane ${L} đọc P[${row}][cột ${p.tile * c.bc + jj}] trước khi lane ${owner} ghi bất cứ gì vào đó: vị trí này chưa được khởi tạo`,
                        )
                      : Skind[at] === 'S' && Stile[at] === p.tile
                        ? tr(
                            `lane ${L} read slot ${jj} of row ${row} while it still held lane ${owner}'s raw score, not yet exponentiated`,
                            `lane ${L} đọc vị trí ${jj} của hàng ${row} khi nó vẫn chứa điểm số thô của lane ${owner}, chưa lấy mũ`,
                          )
                        : tr(
                            `lane ${L} read slot ${jj} of row ${row} before lane ${owner} had written this tile's P: it found ${Stile[at] === p.tile ? 'a score' : `tile ${Stile[at]}'s leftover`}`,
                            `lane ${L} đọc vị trí ${jj} của hàng ${row} trước khi lane ${owner} ghi P của ô này: nó gặp ${Stile[at] === p.tile ? 'một điểm số' : `phần còn sót của ô ${Stile[at]}`}`,
                          ),
                });
              }
              if (Vtile[va] !== p.tile)
                flag(s, {
                  kind: 'v-tile',
                  message: tr(
                    `lane ${L} read V_j[${jj}][${dd}] for tile ${p.tile}, but it holds ${Vtile[va] < 0 ? 'nothing yet' : `tile ${Vtile[va]}'s value: another warp had already loaded the next tile over it`}`,
                    `lane ${L} đọc V_j[${jj}][${dd}] cho ô ${p.tile}, nhưng nó đang chứa ${Vtile[va] < 0 ? 'chưa có gì' : `giá trị của ô ${Vtile[va]}: một warp khác đã nạp ô kế tiếp đè lên nó`}`,
                  ),
                });
              acc += S[at] * Vj[va];
            }
            r.O[p.ii][sl] = r.O[p.ii][sl] * factor + acc;
          }
          if (!merged[lr][L].includes(p.tile)) merged[lr][L].push(p.tile);
        }
        s.touch.push({ array: 'S', cells: sCells, write: false }, { array: 'Vj', cells: vCells, write: false });
        s.text =
          tr(`${who}, row ${row}: `, `${who}, hàng ${row}: `) +
          tr(
            `${lanes.length === W ? 'each lane multiplies its O elements' : 'it multiplies its O elements'} by e^(old max − new max) = ${fmtv(factor)}${factor === 1 ? ' (the max did not rise' + (c.returnsNewMax ? ', or so the defect says' : '') + ')' : ''}, then adds the P row times its column${DS > 1 ? 's' : ''} of V_j, reading P elements ${lanes.length === W ? 'other lanes wrote' : 'written by every lane'}.`,
            `${lanes.length === W ? 'mỗi lane nhân các phần tử O của mình' : 'nó nhân các phần tử O của mình'} với e^(max cũ − max mới) = ${fmtv(factor)}${factor === 1 ? ' (max không tăng' + (c.returnsNewMax ? ', hoặc lỗi cài sẵn báo như vậy' : '') + ')' : ''}, rồi cộng thêm hàng P nhân với ${DS > 1 ? 'các cột' : 'cột'} V_j của mình, đọc các phần tử P ${lanes.length === W ? 'do các lane khác ghi' : 'do mọi lane ghi'}.`,
          );
        break;
      }
      case 'store_O': {
        const cells: number[] = [];
        const rows: number[] = [];
        for (const L of lanes) {
          const t = w * W + L;
          const r = regs[t];
          for (let i2 = 0; i2 < c.brWarp; i2++) {
            const gr = rowOf(w, i2);
            if (gr >= c.n) continue;
            if (!stored[gr]) stored[gr] = { O: new Array(c.d).fill(NaN), L: NaN };
            for (let sl = 0; sl < DS; sl++) {
              const dd = L + sl * W;
              stored[gr].O[dd] = r.O[i2][sl] / r.D[i2];
              cells.push(gr * c.d + dd);
            }
            if (L === 0) stored[gr].L = r.m[i2] + Math.log(r.D[i2]);
            if (!rows.includes(gr)) rows.push(gr);
          }
        }
        s.touch.push({ array: 'O', cells, write: true });
        s.text = rows.length
          ? tr(
              `${who} divides O by D once and writes ${rows.length > 1 ? 'rows' : 'row'} ${span(rows)} of O${lanes.includes(0) ? `, and L = m + log D for each (${rows.map((g) => fmtv(stored[g].L)).join(', ')})` : ''}.`,
              `${who} chia O cho D một lần và ghi ${rows.length > 1 ? 'các hàng' : 'hàng'} ${span(rows)} của O${lanes.includes(0) ? `, cùng L = m + log D cho mỗi hàng (${rows.map((g) => fmtv(stored[g].L)).join(', ')})` : ''}.`,
            )
          : tr(`${who}: its rows are past N, so nothing is stored.`, `${who}: các hàng của nó đã quá N, nên không lưu gì.`);
        break;
      }
    }
    for (const L of lanes) pc[w * W + L]++;
  };

  const runnable = (t: number) => live(t) && waiting[t] === null;
  const warpRunnable = (w: number) => range(0, W).some((L) => runnable(w * W + L));

  /** The next event of warp w under the lane schedule. */
  const warpEvent = (w: number) => {
    if (laneSched === 'lockstep') {
      const lanes = range(0, W).filter((L) => runnable(w * W + L));
      exec(w, lanes);
    } else {
      const L = range(0, W).find((x) => runnable(w * W + x))!;
      exec(w, [L]);
    }
  };

  const release = (): boolean => {
    // A warp barrier opens when every live lane of the warp waits at it.
    for (let w = 0; w < c.warps; w++) {
      const ts = range(0, W).map((L) => w * W + L);
      if (ts.some((t) => waiting[t] === 'syncwarp') && ts.every((t) => waiting[t] === 'syncwarp' || !live(t))) {
        const p = plan[pc[ts[0]]];
        steps.push({
          index: steps.length,
          warp: w,
          lanes: range(0, W),
          op: 'release_syncwarp',
          tile: 'tile' in p ? p.tile : null,
          ii: 'ii' in p ? p.ii : null,
          text: tr(
            `Warp ${w}'s __syncwarp() opens: every lane's P element is now visible to the whole warp.`,
            `__syncwarp() của warp ${w} mở: phần tử P của mọi lane nay đã hiển thị với cả warp.`,
          ),
          touch: [],
        });
        for (const t of ts) {
          waiting[t] = null;
          pc[t]++;
        }
        if (record) snaps.push(snap());
        return true;
      }
    }
    const all = range(0, T);
    if (all.some((t) => waiting[t] === 'syncthreads') && all.every((t) => waiting[t] === 'syncthreads' || !live(t))) {
      const p = plan[pc[all.find((t) => waiting[t] === 'syncthreads')!]] as Extract<PlanOp, { op: 'syncthreads' }>;
      steps.push({
        index: steps.length,
        warp: -1,
        lanes: [],
        op: 'release_syncthreads',
        barrier: p.barrier,
        tile: p.tile,
        ii: null,
        text:
          p.barrier === 'load'
            ? tr(
                `The block barrier opens: tile ${p.tile} of K^T and V is complete in shared memory, so every warp may read it.`,
                `Rào chắn của khối mở: ô ${p.tile} của K^T và V đã đầy đủ trong bộ nhớ chia sẻ, nên mọi warp có thể đọc nó.`,
              )
            : tr(
                `The block barrier opens: every warp has finished with tile ${p.tile}, so the next tile may overwrite it.`,
                `Rào chắn của khối mở: mọi warp đã dùng xong ô ${p.tile}, nên ô kế tiếp có thể ghi đè lên nó.`,
              ),
        touch: [],
      });
      for (const t of all)
        if (waiting[t]) {
          waiting[t] = null;
          pc[t]++;
        }
      if (record) snaps.push(snap());
      return true;
    }
    return false;
  };

  let guard = 0;
  const after = () => {
    if (record) snaps.push(snap());
    while (release()) {
      /* open every barrier that can open */
    }
  };
  while (range(0, T).some(live)) {
    if (++guard > 1e6) throw new Error('flash kernel model did not terminate');
    let progressed = false;
    if (warpSched === 'together') {
      for (let w = 0; w < c.warps; w++)
        if (warpRunnable(w)) {
          warpEvent(w);
          after();
          progressed = true;
        }
    } else {
      for (let w = 0; w < c.warps; w++)
        while (warpRunnable(w)) {
          warpEvent(w);
          after();
          progressed = true;
        }
    }
    if (!progressed && !release()) throw new Error('flash kernel model deadlocked');
  }
  return { config: c, panel, lanes: laneSched, warpsOrder: warpSched, steps, snaps, violations, stored };
}

function range(a: number, b: number): number[] {
  return Array.from({ length: Math.max(0, b - a) }, (_, i) => a + i);
}
function laneElementsList(L: number, c: KConfig): string {
  const out: number[] = [];
  for (let dd = L; dd < c.d; dd += c.lanes) out.push(dd);
  return out.length === 2 ? tr(`${out[0]} and ${out[1]}`, `${out[0]} và ${out[1]}`) : out.join(', ');
}

// ---------------------------------------------------------------- whole kernel
export interface KernelResult {
  O: number[];
  L: number[];
  /** Rows whose O or L differs from the reference (NaN counts as different). */
  wrongRows: number[];
  correct: boolean;
  violations: (KViolation & { step: number; panel: number })[];
  maxErr: number;
}

const TOL = 1e-9;

/** Runs every block, one after another, and compares O and L with exact attention. */
export function runKernel(c: KConfig, prob: KProblem, laneSched: LaneSchedule, warpSched: WarpSchedule): KernelResult {
  const O = new Array(c.n * c.d).fill(NaN);
  const L = new Array(c.n).fill(NaN);
  const violations: KernelResult['violations'] = [];
  for (let i = 0; i < panels(c); i++) {
    const tr = runBlock(c, prob, i, laneSched, warpSched, { record: false });
    for (const [r, v] of Object.entries(tr.stored)) {
      const row = Number(r);
      v.O.forEach((x, dd) => (O[row * c.d + dd] = x));
      L[row] = v.L;
    }
    for (const v of tr.violations) violations.push({ ...v, panel: i });
  }
  const wrongRows: number[] = [];
  let maxErr = 0;
  for (let r = 0; r < c.n; r++) {
    let bad = !(Math.abs(L[r] - prob.L[r]) <= TOL);
    for (let dd = 0; dd < c.d; dd++) {
      const e = Math.abs(O[r * c.d + dd] - prob.O[r * c.d + dd]);
      if (!(e <= TOL)) bad = true;
      maxErr = Number.isNaN(e) ? NaN : Math.max(maxErr, e);
    }
    if (bad) wrongRows.push(r);
  }
  return { O, L, wrongRows, correct: wrongRows.length === 0, violations, maxErr };
}

/** True when the stored row matches the reference. */
export function rowOk(prob: KProblem, row: number, v: { O: number[]; L: number }): boolean {
  return Math.abs(v.L - prob.L[row]) <= TOL && v.O.every((x, dd) => Math.abs(x - prob.O[row * prob.d + dd]) <= TOL);
}

/** The functions of the kernel in the order the stepper's outline lists them. */
export const OUTLINE: { op: Op; text: string; depth: number }[] = [
  { op: 'load_Q', text: 'load_Q(Q, Q_i, i)', depth: 0 },
  { op: 'load_KT_and_V', text: 'load_KT_and_V(K, V, KT_j, V_j, j)', depth: 1 },
  { op: 'syncthreads', text: '__syncthreads()   // tiles complete', depth: 1 }, // i18n-ignore: code shown as written
  { op: 'compute_S_and_max', text: 'compute_S_and_max(...)', depth: 2 },
  { op: 'update_m_and_D', text: 'last_m = update_m_and_D(...)', depth: 2 },
  { op: 'compute_P_and_update_D', text: 'compute_P_and_update_D(...)', depth: 2 },
  { op: 'syncwarp', text: '__syncwarp()      // P row visible to the warp', depth: 2 }, // i18n-ignore: code shown as written
  { op: 'compute_O', text: 'compute_O(..., last_m, ...)', depth: 2 },
  { op: 'syncthreads', text: '__syncthreads()   // tiles free again', depth: 1 }, // i18n-ignore: code shown as written
  { op: 'store_O', text: 'store_O(O, L, O_i, m_i, D_i, i, N)', depth: 0 },
];
