import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { defineLabs, solutionTabStatus } from '../../components/LabKit';
import { Aside, Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BANK_MODELS,
  BANK_WIDTHS,
  COMPUTE_ACCESSES,
  LAYOUT_LABEL,
  computeRange,
  stepsFor,
  warpUse,
  warpsFor,
  worstOf,
  type BankAccess,
  type BankKernel,
  type BankLayout,
  type BankModel,
} from '../../sim/wavebanks';
import {
  GAP,
  MATCH,
  MISMATCH,
  ORDER_LABEL,
  busyPerStep,
  cellAt,
  fill,
  launchPlan,
  loadsOf,
  makeSequences,
  producerOf,
  runTile,
  schedule,
  slotAt,
  slotCell,
  stepsPerTile,
  subst,
  summary,
  tableSize,
  tileOutline,
  type Geometry,
  type Order,
  type Part,
  type Read,
  type Scheme,
  type Tick,
  type Tile,
} from '../../sim/wavetile';
import { ShearScene } from './ShearScene';

// Section 16.6 of the course: book §§16.5–16.8 (pp. 382–398) for the tiled Smith-Waterman
// kernels. Everything the interactives show is computed by src/sim/wavetile.ts and
// src/sim/wavebanks.ts; the listing is lab 16.1's reference solution, loaded from disk.

const [LAB] = defineLabs('ch16', [{ num: '16.1', dir: 'lab1_wavefront', name: 'wavefront', vi: { name: 'wavefront' } }]);
const LAB_DIR = 'ch16/lab1_wavefront/solution/';

// Identifiers coloured by where they live. The sequences a and b are global too, but those
// names are also max4's parameters, so they are left plain.
const SW_TIERS: Record<string, Tier> = {
  H: 'global', swTile: 'shared',
  r: 'reg', q: 'reg', r_tile: 'reg', q_tile: 'reg', d_tile: 'reg', tile_row: 'reg', tile_col: 'reg',
  n: 'reg', w: 'reg', nw: 'reg', s: 'reg', row: 'reg', firstRow: 'reg', stride: 'reg',
};

/** Status lines of the lab's files, in the labs page's words (LabKit's solutionTabStatus). */
function labStatuses(reports: Reports): { cu: string; metal: string } {
  return { cu: solutionTabStatus(reports, LAB, 'kernel.cu'), metal: solutionTabStatus(reports, LAB, 'kernel.metal') };
}

// ---------------------------------------------------------------- the load functions the stepper highlights

// The three load helpers of each kernel, verbatim from lab 16.1's solution
// (labs/ch16/lab1_wavefront/solution/kernel.cu; snippets-s166.test.ts ties them to the
// file). LOAD_LINE gives the line the stepper highlights when a load is served by the tile.
const SQUARE_LOADS = `__device__ inline int sqLoadN(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                              int stride) {
    return (r_tile == 0) ? H[(r - 1) * L + q] : swTile[(r_tile - 1) * stride + q_tile];
}
__device__ inline int sqLoadW(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                              int stride) {
    return (q_tile == 0) ? H[r * L + (q - 1)] : swTile[r_tile * stride + (q_tile - 1)];
}
__device__ inline int sqLoadNW(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                               int stride) {
    return (r_tile == 0 || q_tile == 0) ? H[(r - 1) * L + (q - 1)] : swTile[(r_tile - 1) * stride + (q_tile - 1)];
}`;
const SHEARED_LOADS = `__device__ inline int hyLoadN(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                              int stride) {
    return (r_tile == 0 || q_tile == 0) ? H[(r - 1) * L + q] : swTile[(r_tile - 1) * stride + (q_tile - 1)];
}
__device__ inline int hyLoadW(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                              int stride) {
    return (q_tile == 0) ? H[r * L + (q - 1)] : swTile[r_tile * stride + (q_tile - 1)];
}
__device__ inline int hyLoadNW(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                               int stride) {
    return (r_tile == 0 || q_tile == 0 || q_tile == 1) ? H[(r - 1) * L + (q - 1)]
                                                       : swTile[(r_tile - 1) * stride + (q_tile - 2)];
}`;
const LOAD_LINE: Record<Scheme, Record<'n' | 'w' | 'nw', number>> = {
  square: { n: 3, w: 7, nw: 11 },
  sheared: { n: 3, w: 7, nw: 12 },
};
const LOAD_NAME = {
  get n() {
    return tr('north', 'bắc');
  },
  get w() {
    return tr('west', 'tây');
  },
  get nw() {
    return tr('north-west', 'tây bắc');
  },
} as const;

// ---------------------------------------------------------------- explorer: fronts of tiles

type CellState = 'boundary' | 'done' | 'now' | 'wait' | 'never';
const CELL_FILL: Record<CellState, string> = {
  boundary: 'none',
  done: 'var(--global-wash)',
  now: 'var(--shared)',
  wait: 'var(--cell)',
  never: 'var(--fail-wash)',
};
const CELL_STROKE: Record<CellState, string> = {
  boundary: 'var(--rule)',
  done: 'var(--rule-soft)',
  now: 'var(--shared)',
  wait: 'var(--rule-soft)',
  never: 'var(--fail)',
};

function Swatch({ fill, stroke, dash, children }: { fill: string; stroke: string; dash?: boolean; children: ReactNode }) {
  return (
    <span>
      <i style={{ background: fill === 'none' ? 'transparent' : fill, border: `1px ${dash ? 'dashed' : 'solid'} ${stroke}` }} />
      {children}
    </span>
  );
}

function Readout({ id, label, value, note }: { id: string; label: ReactNode; value: ReactNode; note?: ReactNode }) {
  return (
    <tr>
      <td>{label}</td>
      <td className="num" data-readout={id}>{value}</td>
      <td className="muted">{note}</td>
    </tr>
  );
}

function FrontsExplorer() {
  const [Lseq, setLseq] = useState(24);
  const [W, setW] = useState(4);
  const [scheme, setScheme] = useState<Scheme>('square');
  const [less, setLess] = useState(false);
  const [d, setD] = useState(0);
  const g: Geometry = useMemo(() => ({ Lseq, W, scheme, lastColLess: scheme === 'sheared' && less }), [Lseq, W, scheme, less]);
  const seqs = useMemo(() => makeSequences(Lseq), [Lseq]);
  const f = useMemo(() => fill(g, seqs.a, seqs.b), [g, seqs]);
  const sum = useMemo(() => summary(g), [g]);
  const plans = useMemo(() => schedule(g), [g]);
  const n = sum.launches;
  const dd = Math.min(d, n - 1);
  const plan = plans[dd];
  const T = sum.T;
  const L = tableSize(g);
  const cs = Math.max(6, Math.min(20, Math.floor(500 / L)));
  const pad = 4;
  const size = L * cs + 2 * pad;
  const busy = busyPerStep(W, scheme);
  const skipped: Tile[] = g.lastColLess ? Array.from({ length: T }, (_, row) => ({ row, col: T })) : [];
  const state = (r: number, q: number): CellState => {
    if (r === 0 || q === 0) return 'boundary';
    const at = f.launchOf[r * L + q];
    if (at === -2) return 'never';
    return at < dd ? 'done' : at === dd ? 'now' : 'wait';
  };
  const poly = (t: Tile) => tileOutline(g, t).map(([x, y]) => `${pad + x * cs},${pad + y * cs}`).join(' ');
  const barW = Math.max(3, Math.min(16, Math.floor(420 / n)));
  const busyW = Math.max(4, Math.min(12, Math.floor(260 / busy.length)));
  const mostBar = Math.max(1, sum.mostTiles);

  return (
    <Bench
      title={tr('Fronts of tiles', 'Các mặt sóng của ô')}
      task={tr('At L_seq = 48, compare W = 4 with W = 12: launches against tiles per launch. Then shear, and compare again.', 'Ở L_seq = 48, so sánh W = 4 với W = 12: số lần khởi chạy so với số ô mỗi lần khởi chạy. Rồi trượt, và so sánh lại.')}
      foot={
        <>
          <Translated
            vi={
              <>
                Bảng có kích thước (L_seq + 1) × (L_seq + 1); hàng 0 và cột 0 là biên. Mỗi lần khởi chạy chạy đúng các ô của mặt sóng
                của nó (khối b nhận tile_row = firstRow + b). Được tính bởi <code>src/sim/wavetile.ts</code>, mô hình này cũng lấp bảng
                bằng điểm số thật và kiểm tra mọi giá trị so với một lần lấp trực tiếp trên CPU.
              </>
            }
          >
          The table is (L_seq + 1) × (L_seq + 1); row 0 and column 0 are the boundary. Each launch runs exactly the tiles of
          its front (block b takes tile_row = firstRow + b). Computed by <code>src/sim/wavetile.ts</code>, which also fills
          the table with real scores and checks every value against a direct CPU fill.
          </Translated>
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated
            vi={
              <>L_seq = {Lseq}</>
            }
          >
          L_seq = {Lseq}
          </Translated>
          <input type="range" min={1} max={48} value={Lseq} aria-label={tr('L_seq', 'L_seq')} onChange={(e) => setLseq(+e.target.value)} />
        </label>
        <label className="field">
          <Translated
            vi={
              <>độ rộng ô W = {W}</>
            }
          >
          tile width W = {W}
          </Translated>
          <input type="range" min={1} max={12} value={W} aria-label={tr('tile width W', 'độ rộng ô W')} onChange={(e) => setW(+e.target.value)} />
        </label>
        <span className="seg" role="group" aria-label={tr('Tiles', 'Ô')}>
          {(['square', 'sheared'] as Scheme[]).map((s) => (
            <button key={s} aria-pressed={scheme === s} onClick={() => setScheme(s)}><Translated vi={s === 'square' ? 'ô vuông' : 'ô đã trượt'}>{s} tiles</Translated></button>
          ))}
        </span>
        {scheme === 'sheared' ? (
          <label className="check">
            <input type="checkbox" checked={less} onChange={(e) => setLess(e.target.checked)} /> <Translated vi={<>kiểm tra cột cuối bằng <code>tile_col &lt; T</code></>}>test the last column with <code>tile_col &lt; T</code></Translated>
          </label>
        ) : null}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setD(Math.max(0, dd - 1))} disabled={dd === 0}><Translated vi="Lần khởi chạy trước">Previous launch</Translated></button>
        <button className="primary" onClick={() => setD(Math.min(n - 1, dd + 1))} disabled={dd >= n - 1}><Translated vi="Lần khởi chạy sau">Next launch</Translated></button>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          <Translated
            vi={
              <>lần khởi chạy d = {dd} trong khoảng 0 đến {n - 1}</>
            }
          >
          launch d = {dd} of 0 to {n - 1}
          </Translated>
          <input type="range" min={0} max={n - 1} value={dd} aria-label={tr('launch d', 'lần khởi chạy d')} onChange={(e) => setD(+e.target.value)} />
        </label>
      </div>
      <p className="narrate" data-launch-text style={{ marginBottom: '0.8rem' }}>
        <Translated
          vi={
            <>
              Lần khởi chạy d = {dd}: {plan.tiles.length} khối
              {plan.tiles.length ? <>, firstRow = {plan.firstRow}; các ô (tile_row, tile_col) {plan.tiles.length <= 8 ? plan.tiles.map((t) => `(${t.row}, ${t.col})`).join(', ') : `(${plan.tiles[0].row}, ${plan.tiles[0].col}) đến (${plan.tiles[plan.tiles.length - 1].row}, ${plan.tiles[plan.tiles.length - 1].col})`}</> : ': không ô nào của nó tồn tại'}
              . {scheme === 'square' ? 'tile_col = d − tile_row.' : 'tile_col = d − 2·tile_row.'}
            </>
          }
        >
        Launch d = {dd}: {plan.tiles.length} block{plan.tiles.length === 1 ? '' : 's'}
        {plan.tiles.length ? <>, firstRow = {plan.firstRow}; tiles (tile_row, tile_col) {plan.tiles.length <= 8 ? plan.tiles.map((t) => `(${t.row}, ${t.col})`).join(', ') : `(${plan.tiles[0].row}, ${plan.tiles[0].col}) to (${plan.tiles[plan.tiles.length - 1].row}, ${plan.tiles[plan.tiles.length - 1].col})`}</> : ': none of its tiles exist'}
        . {scheme === 'square' ? 'tile_col = d − tile_row.' : 'tile_col = d − 2·tile_row.'}
        </Translated>
      </p>
      <div className="cols plot">
        <div>
          <svg viewBox={`0 0 ${size} ${size}`} style={{ width: '100%', maxWidth: size, display: 'block' }} role="img" aria-label={tr('The table with its tiles and the current launch', 'Bảng với các ô và lần khởi chạy hiện tại')} data-table-svg>
            <defs>
              <clipPath id="fronts-clip">
                <rect x={pad + cs} y={pad + cs} width={(L - 1) * cs} height={(L - 1) * cs} />
              </clipPath>
            </defs>
            {Array.from({ length: L * L }, (_, i) => {
              const r = Math.floor(i / L);
              const q = i % L;
              const s = state(r, q);
              return (
                <rect key={i} x={pad + q * cs} y={pad + r * cs} width={cs - 1} height={cs - 1} data-cell={`${r},${q}`} data-state={s}
                  fill={CELL_FILL[s]} stroke={CELL_STROKE[s]} strokeDasharray={s === 'boundary' ? '2 2' : undefined} />
              );
            })}
            <g clipPath="url(#fronts-clip)">
              {plans.flatMap((p) =>
                p.tiles.map((t) => (
                  <polygon key={`${t.row},${t.col}`} points={poly(t)} fill="none" stroke="var(--ink)" strokeWidth={p.d === dd ? 2.4 : 0.9} opacity={p.d === dd ? 1 : 0.55} data-tile={`${t.row},${t.col}`} />
                )),
              )}
              {skipped.map((t) => (
                <polygon key={'skip' + t.row} points={poly(t)} fill="none" stroke="var(--fail)" strokeWidth="1.6" strokeDasharray="4 3" data-skipped={`${t.row},${t.col}`} />
              ))}
            </g>
          </svg>
          <div className="legend" style={{ marginTop: '0.5rem' }}>
            <Swatch fill="none" stroke="var(--rule)" dash><Translated vi="biên (0)">boundary (0)</Translated></Swatch>
            <Swatch fill={CELL_FILL.done} stroke="var(--global)"><Translated vi="xong, trong bộ nhớ toàn cục">done, in global memory</Translated></Swatch>
            <Swatch fill={CELL_FILL.now} stroke="var(--shared)"><Translated vi="lần khởi chạy này">this launch</Translated></Swatch>
            <Swatch fill={CELL_FILL.wait} stroke="var(--rule)"><Translated vi="đang chờ">waiting</Translated></Swatch>
            {g.lastColLess ? <Swatch fill={CELL_FILL.never} stroke="var(--fail)"><Translated vi="không bao giờ được tính">never computed</Translated></Swatch> : null}
          </div>
        </div>
        <div>
          <div className="mx-label"><Translated vi="Số ô mỗi lần khởi chạy, mọi d">Tiles per launch, every d</Translated></div>
          <svg viewBox={`0 0 ${n * barW + 10} 92`} style={{ width: '100%', maxWidth: n * barW + 10, display: 'block' }} role="img" aria-label={tr('Tiles per launch', 'Số ô mỗi lần khởi chạy')} data-bars={sum.perLaunch.join(',')}>
            {sum.perLaunch.map((k, i) => (
              <rect key={i} x={2 + i * barW} y={84 - (k / mostBar) * 76} width={barW - 1} height={(k / mostBar) * 76}
                fill={i === dd ? 'var(--ink)' : 'var(--rule-soft)'} stroke="var(--ink-3)" strokeWidth={0.6} />
            ))}
            <line x1={0} y1={84.5} x2={n * barW + 6} y2={84.5} stroke="var(--rule)" />
          </svg>
          <div className="muted" style={{ fontSize: '0.75rem', marginBottom: '0.8rem' }}><Translated vi={<>cột cao nhất: {sum.mostTiles} ô; lần khởi chạy hiện tại được tô đặc</>}>highest bar: {sum.mostTiles} tile{sum.mostTiles === 1 ? '' : 's'}; the current launch is solid</Translated></div>
          <div className="mx-label"><Translated vi="Số luồng bận mỗi bước bên trong một ô đầy đủ">Busy threads per step inside a full tile</Translated></div>
          <svg viewBox={`0 0 ${busy.length * busyW + 6} 46`} style={{ width: '100%', maxWidth: (busy.length * busyW + 6) * 1.5, display: 'block' }} role="img" aria-label={tr('Busy threads per step', 'Số luồng bận mỗi bước')} data-busy={busy.join(',')}>
            {busy.map((b, i) => (
              <rect key={i} x={2 + i * busyW} y={42 - (b / W) * 38} width={busyW - 1} height={(b / W) * 38} fill="var(--rule-soft)" stroke="var(--ink-3)" strokeWidth="0.6" />
            ))}
            <line x1={0} y1={42.5} x2={busy.length * busyW + 4} y2={42.5} stroke="var(--rule)" />
          </svg>
          <div className="muted" style={{ fontSize: '0.75rem', marginBottom: '0.8rem' }}><Translated vi={<>{busy.length} bước, mỗi bước tối đa {W} luồng</>}>{busy.length} steps of at most {W} threads</Translated></div>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <Readout id="T" label={<Translated vi="số ô mỗi cạnh, T">tiles per side, T</Translated>} value={T} note={`⌈${Lseq} / ${W}⌉`} />
              <Readout id="launches" label={<Translated vi="số lần khởi chạy">launches</Translated>} value={n}
                note={scheme === 'square' ? (W === 1 ? tr(`2T − 1 = 2·L_seq − 1: the baseline`, `2T − 1 = 2·L_seq − 1: phương án cơ sở`) : '2T − 1') : '3T − 1'} />
              <Readout id="most" label={<Translated vi="nhiều ô nhất trong một lần khởi chạy">most tiles in one launch</Translated>} value={sum.mostTiles} />
              <Readout id="blocks" label={<Translated vi="số lần tính ô">tile computations</Translated>} value={sum.blocks} note={scheme === 'sheared' ? tr('incomplete edge tiles included', 'gồm cả các ô rìa không trọn') : undefined} />
              <Readout id="steps" label={<Translated vi="số bước mỗi ô">steps per tile</Translated>} value={sum.stepsPerTile} note={scheme === 'square' ? '2W − 1' : 'W'} />
              <Readout id="critical" label={<Translated vi="số bước rào chắn trên đường găng">barrier steps on the critical path</Translated>} value={sum.criticalSteps} note={tr('launches × steps per tile', 'số lần khởi chạy × số bước mỗi ô')} />
              <Readout id="busy" label={<Translated vi="số luồng bận trung bình mỗi bước">mean busy threads per step</Translated>} value={Number(sum.busyMean.toFixed(2))} note={tr(`of ${W}, in a full tile`, `trên ${W}, trong một ô đầy đủ`)} />
              {g.lastColLess ? (
                <>
                  <Readout id="never" label={<span className="t-fail"><Translated vi="số ô con không bao giờ được tính">cells never computed</Translated></span>} value={f.neverComputed} />
                  <Readout id="missing" label={<span className="t-fail"><Translated vi="số ô con thiếu giá trị">cells with a missing value</Translated></span>} value={f.missing} note={tr('never computed, or computed from one that was', 'không bao giờ được tính, hoặc được tính từ một ô như vậy')} />
                </>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- stepper: inside one tile

const TILES: Record<'interior' | 'edge', { label: string; tile: Tile }> = {
  interior: {
    get label() {
      return tr('an interior tile, (1, 1)', 'một ô ở phần trong, (1, 1)');
    },
    tile: { row: 1, col: 1 },
  },
  edge: {
    get label() {
      return tr('a left-edge tile, (1, 0)', 'một ô ở rìa trái, (1, 0)');
    },
    tile: { row: 1, col: 0 },
  },
};
const ORDERS: Order[] = ['lockstep', 'first', 'last'];

const fmtV = (v: number) => (Number.isNaN(v) ? '?' : String(v));
const list = (xs: number[]) => (xs.length <= 1 ? xs.join('') : tr(`${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`, `${xs.slice(0, -1).join(', ')} và ${xs[xs.length - 1]}`));
const problemText = (r: Read) =>
  r.problem === 'read-before-write'
    ? tr(`read slot (${r.slot![0]}, ${r.slot![1]}) before thread ${r.slot![0]} wrote it (read before write)`, `đọc ô nhớ (${r.slot![0]}, ${r.slot![1]}) trước khi luồng ${r.slot![0]} ghi nó (đọc trước khi ghi)`)
    : r.problem === 'uninitialised'
      ? tr(`read slot (${r.slot![0]}, ${r.slot![1]}), which nothing wrote or zeroed: garbage`, `đọc ô nhớ (${r.slot![0]}, ${r.slot![1]}), vốn chưa được ghi hay đặt về 0: rác`)
      : tr(`read table cell (${r.r}, ${r.q}) before any launch wrote it`, `đọc ô (${r.r}, ${r.q}) của bảng trước khi lần khởi chạy nào ghi nó`);

function skipReason(g: Geometry, p: Part): string {
  const c = p.cell!;
  if (!c.inTile) return tr(`q_tile = ${c.qTile} is outside 0 to ${g.W - 1}`, `q_tile = ${c.qTile} nằm ngoài 0 đến ${g.W - 1}`);
  if (c.q < 1) return tr(`q = ${c.q} is left of the table`, `q = ${c.q} nằm bên trái bảng`);
  return tr(`(${c.r}, ${c.q}) is outside the table`, `(${c.r}, ${c.q}) nằm ngoài bảng`);
}

function describeTick(g: Geometry, tk: Tick, seqs: { a: string; b: string }): ReactNode {
  const bar = tk.barrierAfter ? <Translated vi={<> Rào chắn: mọi luồng đã xong bước {tk.k}.</>}><> Barrier: every thread has finished step {tk.k}.</></Translated> : null;
  if (tk.kind === 'store') {
    const t = tk.parts.map((p) => p.thread);
    const n = tk.parts.reduce((s, p) => s + (p.stores?.length ?? 0), 0);
    return tk.parts.length > 1 ? (
      <Translated
        vi={
          <>store_tile: trong mỗi vòng lặp của W vòng lặp, luồng t chép ô nhớ (row, t), nên các luồng ghi các cột liên tiếp của một hàng trong bảng. {n} ô đến được bảng.</>
        }
      >
      <>store_tile: in each of the W iterations, thread t copies slot (row, t), so the threads write consecutive columns of one table row. {n} cells reach the table.</>
      </Translated>
    ) : (
      <Translated
        vi={
          <>Luồng {t[0]} chạy phần store_tile của nó: cột {t[0]} của mọi hàng trong ô, {n} ô.</>
        }
      >
      <>Thread {t[0]} runs its part of store_tile: column {t[0]} of every tile row, {n} cell{n === 1 ? '' : 's'}.</>
      </Translated>
    );
  }
  const busy = tk.parts.filter((p) => p.cell?.active);
  const idle = tk.parts.filter((p) => !p.cell?.active);
  if (tk.parts.length > 1)
    return (
      <>
        <Translated
          vi={
            <>
              Bước {tk.k}: {busy.length ? <>luồng {list(busy.map((p) => p.thread))} mỗi luồng tính một ô</> : 'không luồng nào tính'}.{' '}
              {idle.length ? (
                <>Luồng {list(idle.map((p) => p.thread))} bỏ qua nó ({skipReason(g, idle[0])}{idle.length > 1 ? ', v.v.' : ''}): warp tách nhánh ở phép kiểm tra biên.</>
              ) : (
                <>Mọi luồng đều bận.</>
              )}
            </>
          }
        >
        Step {tk.k}: {busy.length ? <>thread{busy.length === 1 ? '' : 's'} {list(busy.map((p) => p.thread))} each compute one cell</> : 'no thread computes'}.{' '}
        {idle.length ? (
          <>Thread{idle.length === 1 ? '' : 's'} {list(idle.map((p) => p.thread))} skip{idle.length === 1 ? 's' : ''} it ({skipReason(g, idle[0])}{idle.length > 1 ? ', and so on' : ''}): the warp splits at the bounds test.</>
        ) : (
          <>Every thread is busy.</>
        )}
        </Translated>
        {bar}
      </>
    );
  const p = tk.parts[0];
  if (!p.cell?.active) return <><Translated vi={<>Luồng {p.thread}, bước {tk.k}: không có gì để tính ({skipReason(g, p)}).</>}>Thread {p.thread}, step {tk.k}: nothing to compute ({skipReason(g, p)}).</Translated>{bar}</>;
  const v = (k: 'n' | 'w' | 'nw') => fmtV(p.reads.find((x) => x.name === k)!.value);
  const s = subst(seqs.a, seqs.b, p.cell.r, p.cell.q);
  return (
    <>
      <Translated
        vi={
          <>
            Luồng {p.thread}, bước {tk.k}: ô ({p.cell.r}, {p.cell.q}) = max(0, nw {s > 0 ? '+' : '−'} {Math.abs(s)}, w − {-GAP}, n − {-GAP}) = max(0,{' '}
            {v('nw')} {s > 0 ? '+' : '−'} {Math.abs(s)}, {v('w')} − {-GAP}, {v('n')} − {-GAP}) = {fmtV(p.value!)}.
          </>
        }
      >
      Thread {p.thread}, step {tk.k}: cell ({p.cell.r}, {p.cell.q}) = max(0, nw {s > 0 ? '+' : '−'} {Math.abs(s)}, w − {-GAP}, n − {-GAP}) = max(0,{' '}
      {v('nw')} {s > 0 ? '+' : '−'} {Math.abs(s)}, {v('w')} − {-GAP}, {v('n')} − {-GAP}) = {fmtV(p.value!)}.{bar}
      </Translated>
    </>
  );
}

function TileStepper({ title, task, scheme }: { title: string; task: string; scheme: Scheme }) {
  const [W, setW] = useState(4);
  const [where, setWhere] = useState<'interior' | 'edge'>('interior');
  const [order, setOrder] = useState<Order>('lockstep');
  const [barrier, setBarrier] = useState(true);
  const [zeroInit, setZeroInit] = useState(true);
  const [n, setN] = useState(0);
  const [sel, setSel] = useState(1);
  const [playing, setPlaying] = useState(false);

  const kind = scheme === 'square' ? 'interior' : where;
  const tile = TILES[kind].tile;
  const g: Geometry = useMemo(() => ({ Lseq: 3 * W, W, scheme }), [W, scheme]);
  const seqs = useMemo(() => makeSequences(3 * W), [W]);
  const run = useMemo(() => runTile(g, tile, { order, barrier, zeroInit }, seqs), [g, tile, order, barrier, zeroInit, seqs]);
  const total = run.ticks.length;
  const at = Math.min(n, total);
  const cur: Tick | undefined = at > 0 ? run.ticks[at - 1] : undefined;
  const slots = run.snapshots[at];
  const L = tableSize(g);
  const d = run.d;
  const firstRow = launchPlan(g, d).firstRow;

  useEffect(() => setN(0), [scheme, W, kind, order, barrier, zeroInit]);
  useEffect(() => setSel((s) => Math.min(s, W - 1)), [W]);
  // In an order that runs one thread at a time, follow the thread that just ran.
  useEffect(() => {
    if (cur && cur.parts.length === 1) setSel(cur.parts[0].thread);
  }, [cur]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) return setPlaying(false);
    const id = window.setTimeout(() => setN((x) => x + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  // The selected thread at the current step (or step 0 before anything ran, the last step after the store).
  const k = cur?.kind === 'step' ? cur.k : at === 0 ? 0 : stepsPerTile(g) - 1;
  const selPart = cur?.parts.find((p) => p.thread === sel && p.kind === 'step');
  const cell = cellAt(g, tile, k, sel);
  const loads = cell.active ? loadsOf(g, cell) : [];
  const reads: (Read | undefined)[] = loads.map((ld) => selPart?.reads.find((r) => r.name === ld.name));
  const failPart = cur?.parts.find((p) => p.reads.some((r) => r.problem) || p.stores?.some((s) => s.problem));
  const failRead = failPart?.reads.find((r) => r.problem);
  const failStore = failPart?.stores?.find((s) => s.problem);
  const failText = !failPart
    ? null
    : failRead
      ? tr(`thread ${failPart.thread} ${problemText(failRead)}`, `luồng ${failPart.thread} ${problemText(failRead)}`)
      : tr(`thread ${failPart.thread} stored slot (${failStore!.row}, ${failPart.thread}) to the table before thread ${failStore!.row} had written it`, `luồng ${failPart.thread} lưu ô nhớ (${failStore!.row}, ${failPart.thread}) vào bảng trước khi luồng ${failStore!.row} ghi nó`);

  const nextBarrier = () => {
    for (let i = at; i < total; i++) if (run.ticks[i].barrierAfter || run.ticks[i].kind === 'store') return setN(i + 1);
    setN(total);
  };

  // The window of the table drawn: the row above the tile, the tile, and the column(s) to its left.
  const W0 = W * tile.row;
  const rows = Array.from({ length: W + 1 }, (_, i) => W0 + i);
  const q0 = scheme === 'square' ? W * tile.col : W * tile.col - W + 1;
  const cols = Array.from({ length: scheme === 'square' ? W + 1 : 2 * W }, (_, i) => q0 + i);
  const cs = 34;
  const ox = 26;
  const oy = 18;
  const winW = ox + cols.length * cs + 4;
  const hotSlots = new Set((cur?.kind === 'step' ? cur.parts.filter((p) => p.cell?.active) : []).map((p) => `${p.cell!.rTile},${p.cell!.qTile}`));
  const readMark = (r: number, q: number) => {
    const i = loads.findIndex((ld) => ld.r === r && ld.q === q);
    if (i < 0) return null;
    const rd = reads[i];
    return { ld: loads[i], bad: !!rd?.problem };
  };
  const slotRead = (rt: number, qt: number) => {
    const i = loads.findIndex((ld) => ld.slot && ld.slot[0] === rt && ld.slot[1] === qt);
    return i < 0 ? null : { ld: loads[i], bad: !!reads[i]?.problem };
  };

  return (
    <Bench
      title={title}
      task={task}
      foot={
        <>
          <Translated
            vi={
              <>
                Bảng L_seq = 3W, nên ô (1, 1) nằm ở phần trong. Một ô nhớ của ô hiện giá trị khi đã được ghi. Các đầu vào được viền
                bằng màu của bộ nhớ mà chúng được đọc từ đó. <TierLegend tiers={['shared', 'global']} />
              </>
            }
          >
          Table L_seq = 3W, so tile (1, 1) is interior. A tile slot shows its value once written. Inputs are outlined in the
          colour of the memory they are read from. <TierLegend tiers={['shared', 'global']} />
          </Translated>
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          W
          <select value={W} aria-label={tr('tile width', 'độ rộng ô')} onChange={(e) => setW(+e.target.value)}>
            {[2, 3, 4, 5, 6, 7, 8].map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        {scheme === 'sheared' ? (
          <label className="field">
            <Translated vi={"\"Ô\""}>
            Tile
            </Translated>
            <select value={where} aria-label={tr('which tile', 'ô nào')} onChange={(e) => setWhere(e.target.value as 'interior' | 'edge')}>
              {(['interior', 'edge'] as const).map((x) => <option key={x} value={x}>{TILES[x].label}</option>)}
            </select>
          </label>
        ) : null}
        <label className="field">
          <Translated vi={"\"Thứ tự luồng\""}>
          Thread order
          </Translated>
          <select value={order} aria-label={tr('thread order', 'thứ tự luồng')} onChange={(e) => setOrder(e.target.value as Order)}>
            {ORDERS.map((o) => <option key={o} value={o}>{tr(ORDER_LABEL[o], ({ lockstep: 'lockstep (một warp lý tưởng hóa)', first: 'luồng 0 trước', last: 'luồng cuối trước' } as Record<Order, string>)[o])}</option>)}
          </select>
        </label>
        <label className="check"><input type="checkbox" checked={barrier} onChange={(e) => setBarrier(e.target.checked)} /> <Translated vi="rào chắn giữa các bước">barrier between steps</Translated></label>
        {scheme === 'sheared' ? (
          <label className="check"><input type="checkbox" checked={zeroInit} onChange={(e) => setZeroInit(e.target.checked)} /> <Translated vi="đặt ô về 0 trước">zero the tile first</Translated></label>
        ) : null}
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={() => { setPlaying(false); setN(0); }} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => { setPlaying(false); setN(at - 1); }} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => { setPlaying(false); setN(at + 1); }} disabled={at >= total}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => { setPlaying(false); nextBarrier(); }} disabled={at >= total}><Translated vi="Chạy đến hết bước">Run to the end of the step</Translated></button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Phát')}</button>
        {run.firstBad !== null ? <button onClick={() => { setPlaying(false); setN(run.firstBad! + 1); }}><Translated vi="Đến lần đọc sai đầu tiên">Go to the first bad read</Translated></button> : null}
        <span className="muted" aria-live="polite" data-progress><Translated vi={<>{at} trên {total}</>}>{at} of {total}</Translated></span>
      </div>
      <div className={'narrate' + (failText ? ' bad' : '')} role="status" style={{ marginBottom: '0.8rem' }} data-narrate>
        {cur ? describeTick(g, cur, seqs) : (
          <Translated
            vi={
              <>Lần khởi chạy d = {d} chạy ô ({tile.row}, {tile.col}) như khối {tile.row - firstRow}. {W} luồng của nó sẽ mất {stepsPerTile(g)} bước{scheme === 'sheared' ? (zeroInit ? ', sau khi đặt ô về 0' : ', với ô chưa được đặt về 0') : ''}, rồi lưu. Bấm Bước.</>
            }
          >
          <>Launch d = {d} runs tile ({tile.row}, {tile.col}) as block {tile.row - firstRow}. Its {W} threads will take {stepsPerTile(g)} steps{scheme === 'sheared' ? (zeroInit ? ', after zeroing the tile' : ', with the tile not zeroed') : ''}, then store. Press Step.</>
          </Translated>
        )}
        {failText ? <><br /><b className="t-fail" data-failure><Translated vi={<>Lỗi: {failText}.</>}>Failure: {failText}.</Translated></b></> : null}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 7fr) minmax(0, 5fr)' }}>
        <div>
          <div className="mx-label"><Translated vi={<>Bảng quanh ô ({tile.row}, {tile.col}); mỗi ô con của ô ghi luồng của nó</>}>The table around tile ({tile.row}, {tile.col}); each tile cell shows its thread</Translated></div>
          <svg viewBox={`0 0 ${winW} ${oy + rows.length * cs + 4}`} style={{ width: '100%', maxWidth: Math.min(560, winW * 1.6), display: 'block' }} role="img" aria-label={tr('Table window', 'Cửa sổ bảng')} data-window>
            {cols.map((q, j) => (
              <text key={'h' + q} x={ox + j * cs + cs / 2} y={12} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{q}</text>
            ))}
            {rows.map((r, i) => (
              <text key={'v' + r} x={ox - 6} y={oy + i * cs + cs / 2 + 3} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>{r}</text>
            ))}
            {rows.flatMap((r, i) =>
              cols.map((q, j) => {
                const x = ox + j * cs;
                const y = oy + i * cs;
                const s = slotAt(g, tile, r, q);
                const mark = readMark(r, q);
                const inTable = r < L && q >= 0 && q < L;
                let fillC = 'var(--cell)';
                let stroke = 'var(--rule-soft)';
                let label = '';
                let dash: string | undefined;
                let sub = '';
                let state = 'none';
                if (s) {
                  const st = slots[s[0] * W + s[1]];
                  const outside = !(r < L && q >= 1 && q < L);
                  sub = String(s[0]);
                  state = hotSlots.has(`${s[0]},${s[1]}`) ? 'hot' : st.kind === 'written' ? 'written' : 'empty';
                  if (outside) { dash = '3 2'; stroke = 'var(--ink-3)'; fillC = 'none'; state = 'outside'; }
                  else if (state === 'hot') { fillC = 'var(--shared)'; stroke = 'var(--shared)'; label = fmtV(st.value); }
                  else if (state === 'written') { fillC = 'var(--shared-wash)'; stroke = 'var(--shared)'; label = fmtV(st.value); }
                  else { stroke = 'var(--shared)'; }
                } else if (!inTable || q < 0) {
                  state = 'outside';
                  fillC = 'none';
                  stroke = 'none';
                } else if (r === 0 || q === 0) {
                  state = 'boundary';
                  fillC = 'none';
                  dash = '2 2';
                  stroke = 'var(--rule)';
                  label = '0';
                } else {
                  if (producerOf(g, r, q)!.d < d) { state = 'global'; fillC = 'var(--global-wash)'; label = String(run.ref[r * L + q]); }
                  else state = 'later';
                }
                if (mark) { stroke = mark.bad ? 'var(--fail)' : mark.ld.from === 'tile' ? 'var(--shared)' : 'var(--global)'; }
                return (
                  <g key={`${r},${q}`} data-wcell={`${r},${q}`} data-state={state} data-input={mark ? mark.ld.name : undefined}>
                    {stroke !== 'none' || fillC !== 'none' ? (
                      <rect x={x + 1} y={y + 1} width={cs - 3} height={cs - 3} rx="2" fill={fillC === 'none' ? 'transparent' : fillC} stroke={stroke}
                        strokeWidth={mark ? 2.6 : 1} strokeDasharray={dash} />
                    ) : null}
                    {label ? <text x={x + cs / 2} y={y + cs / 2 + 4} fontSize="11" textAnchor="middle" fontWeight={state === 'hot' ? 650 : 450} style={{ fill: state === 'hot' ? 'var(--page)' : 'var(--ink)' }}>{label}</text> : null}
                    {sub ? <text x={x + cs - 5} y={y + cs - 5} fontSize="7.5" textAnchor="end" style={{ fill: state === 'hot' ? 'var(--page)' : 'var(--ink-3)' }}>{sub}</text> : null}
                    {mark ? <text x={x + 4} y={y + 10} fontSize="7.5" fontWeight="650" style={{ fill: mark.bad ? 'var(--fail)' : mark.ld.from === 'tile' ? 'var(--shared)' : 'var(--global)' }}>{mark.ld.name}</text> : null}
                  </g>
                );
              }),
            )}
          </svg>
        </div>
        <div>
          <div className="mx-label t-shared"><Translated vi={<>swTile trong bộ nhớ chia sẻ{scheme === 'sheared' ? ': cột lưu trữ = bước' : ''}</>}>swTile in shared memory{scheme === 'sheared' ? ': stored column = step' : ''}</Translated></div>
          <div className="mx shared" style={{ gridTemplateColumns: `repeat(${W}, auto)` }} data-slots>
            {slots.map((st, i) => {
              const rt = Math.floor(i / W);
              const qt = i % W;
              const rd = slotRead(rt, qt);
              const hot = hotSlots.has(`${rt},${qt}`);
              const holds = slotCell(g, tile, rt, qt);
              const out = !(holds.r < L && holds.q >= 1 && holds.q < L);
              const cls = rd?.bad ? 'bad' : hot ? 'hot' : rd ? 'on' : out ? 'void' : st.kind === 'written' ? 'on' : '';
              return (
                <span key={i} className={cls} data-slot={`${rt},${qt}`} data-kind={st.kind} title={tr(`slot (${rt}, ${qt}) holds table cell (${holds.r}, ${holds.q})`, `ô nhớ (${rt}, ${qt}) giữ ô (${holds.r}, ${holds.q}) của bảng`)}
                  style={rd && !rd.bad ? { outline: '2px solid var(--shared)', outlineOffset: '-2px' } : undefined}>
                  {st.kind === 'unset' ? '·' : fmtV(st.value)}
                  {rd ? <sub>{rd.ld.name}</sub> : null}
                </span>
              );
            })}
          </div>
          <p className="muted" style={{ fontSize: '0.75rem', margin: '0.4rem 0 0' }}>
            <Translated
              vi={
                <>Các hàng là các luồng. · : chưa bao giờ được ghi{scheme === 'sheared' ? '; 0 trong một ô nét đứt của bảng là một ô nhớ đã đặt về 0 đại diện cho một ô nằm ngoài bảng' : ''}.</>
              }
            >
            Rows are threads. · : never written{scheme === 'sheared' ? '; 0 in a dashed cell of the table is a zeroed slot standing for a cell outside it' : ''}.
            </Translated>
          </p>
        </div>
      </div>

      <div className="cols" style={{ marginTop: '1rem', gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 6fr)' }}>
        <div>
          <div className="row" style={{ marginBottom: '0.5rem' }}>
            <label className="field">
              <Translated vi={"\"Luồng\""}>
              Thread
              </Translated>
              <select value={sel} aria-label={tr('selected thread', 'luồng đang chọn')} onChange={(e) => setSel(+e.target.value)}>
                {Array.from({ length: W }, (_, t) => <option key={t} value={t}>{t}</option>)}
              </select>
            </label>
            <span className="muted"><Translated vi={<>ở bước d_tile = {k}</>}>at step d_tile = {k}</Translated></span>
          </div>
          <table className="threads" aria-label={tr('Index arithmetic of the selected thread', 'Phép tính chỉ số của luồng đang chọn')} data-arith>
            <tbody>
              <tr><td>tile_row = firstRow + blockIdx.x</td><td>{firstRow} + {tile.row - firstRow} = {tile.row}</td></tr>
              <tr><td>{scheme === 'square' ? 'tile_col = d − tile_row' : 'tile_col = d − 2·tile_row'}</td><td>{scheme === 'square' ? `${d} − ${tile.row}` : `${d} − 2·${tile.row}`} = {tile.col}</td></tr>
              <tr><td>r_tile = threadIdx.x</td><td>{cell.rTile}</td></tr>
              <tr><td>{scheme === 'square' ? 'q_tile = d_tile − r_tile' : 'q_tile = d_tile'}</td><td>{scheme === 'square' ? `${k} − ${sel} = ${cell.qTile}` : cell.qTile}</td></tr>
              <tr><td>r = W·tile_row + r_tile + 1</td><td>{W}·{tile.row} + {cell.rTile} + 1 = {cell.r}</td></tr>
              <tr>
                <td>{scheme === 'square' ? 'q = W·tile_col + q_tile + 1' : 'q = W·tile_col + q_tile − r_tile + 1'}</td>
                <td>{scheme === 'square' ? `${W}·${tile.col} + ${cell.qTile} + 1` : `${W}·${tile.col} + ${cell.qTile} − ${cell.rTile} + 1`} = {cell.q}</td>
              </tr>
              <tr>
                <td>{scheme === 'square' ? '0 ≤ q_tile < W, r < L, q < L' : 'r < L, 1 ≤ q < L'}</td>
                <td data-active={cell.active ? 'yes' : 'no'}>{cell.active ? <span className="chip ok"><Translated vi={<>tính ({cell.r}, {cell.q})</>}>computes ({cell.r}, {cell.q})</Translated></span> : <span className="chip fail"><Translated vi={<>bỏ qua: {cell.inTile ? (cell.q < 1 ? `q = ${cell.q} < 1` : 'ngoài bảng') : `q_tile = ${cell.qTile}`}</>}>skips: {cell.inTile ? (cell.q < 1 ? `q = ${cell.q} < 1` : 'outside the table') : `q_tile = ${cell.qTile}`}</Translated></span>}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div>
          {cell.active ? (
            <table className="threads" aria-label={tr('The three inputs of the selected thread', 'Ba đầu vào của luồng đang chọn')} data-inputs>
              <thead><tr><th><Translated vi="đầu vào">input</Translated></th><th><Translated vi="ô">cell</Translated></th><th><Translated vi="đọc từ">read from</Translated></th><th><Translated vi="giá trị">value</Translated></th></tr></thead>
              <tbody>
                {loads.map((ld, i) => {
                  const rd = reads[i];
                  return (
                    <tr key={ld.name} data-load={ld.name} data-from={ld.from}>
                      <td>{LOAD_NAME[ld.name]}</td>
                      <td>({ld.r}, {ld.q})</td>
                      <td>{ld.from === 'tile' ? <span className="chip shared"><Translated vi={<>ô nhớ swTile ({ld.slot![0]}, {ld.slot![1]})</>}>swTile slot ({ld.slot![0]}, {ld.slot![1]})</Translated></span> : <span className="chip global"><Translated vi="H, bộ nhớ toàn cục">H, global memory</Translated></span>}</td>
                      <td>{rd ? (rd.problem ? <span className="chip fail">{fmtV(rd.value)}: {rd.problem === 'read-before-write' ? tr('not yet written', 'chưa được ghi') : rd.problem === 'uninitialised' ? tr('garbage', 'rác') : tr('missing', 'thiếu')}</span> : fmtV(rd.value)) : <span className="muted"><Translated vi="đọc ở bước này">read at this step</Translated></span>}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : (
            <p className="muted"><Translated vi={<>Luồng {sel} không tính gì ở bước này, nên nó không đọc gì.</>}>Thread {sel} computes nothing at this step, so it reads nothing.</Translated></p>
          )}
        </div>
      </div>
      <Code tabs={[{ label: scheme === 'square' ? tr('square loads', 'các lần nạp ô vuông') : tr('sheared loads', 'các lần nạp ô đã trượt'), code: scheme === 'square' ? SQUARE_LOADS : SHEARED_LOADS, tiers: SW_TIERS }]}
        highlight={cell.active ? loads.filter((ld) => ld.from === 'tile').map((ld) => LOAD_LINE[scheme][ld.name]) : []}
        caption={
          scheme === 'sheared' ? (
            <Translated
              vi={
                <>
                  Lời giải của lab 16.1 cho ba hàm phụ này: hãy trả lời câu hỏi ở trên và thử bài thực hành trước khi đọc nó. Được tô
                  sáng: các lần nạp mà luồng {sel} lấy từ ô ở bước này; các lần còn lại đi ra bộ nhớ toàn cục.
                </>
              }
            >
            <>
              Lab 16.1's solution for these three helpers: answer the question above and try the lab before you read it.
              Highlighted: the loads thread {sel} serves from the tile at this step; the others go to global memory.
            </>
            </Translated>
          ) : (
            <Translated vi={<>Được tô sáng: các lần nạp mà luồng {sel} lấy từ ô ở bước này; các lần còn lại đi ra bộ nhớ toàn cục.</>}><>Highlighted: the loads thread {sel} serves from the tile at this step; the others go to global memory.</></Translated>
          )
        } />
      {at >= total ? (
        <p style={{ margin: '0.6rem 0 0' }} data-verdict={run.correct ? 'correct' : 'wrong'}>
          {run.correct ? (
            <span className="chip ok"><Translated vi={<>Cả {run.cells} ô con của ô này khớp với bảng CPU.</>}>All {run.cells} cells of this tile match the CPU table.</Translated></span>
          ) : (
            <span className="chip fail"><Translated vi={<>{run.wrongCells} trên {run.cells} ô con của ô này sai; {run.badReads} lần đọc sai.</>}>{run.wrongCells} of {run.cells} cells of this tile are wrong; {run.badReads} bad read{run.badReads === 1 ? '' : 's'}.</Translated></span>
          )}
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: banks

const KERNELS: BankKernel[] = ['square', 'sheared'];
const LAYOUTS: BankLayout[] = ['unpadded', 'rowgap', 'pad'];
const ACCESS_SHORT: Record<BankAccess, string> = {
  get write() {
    return tr('write', 'ghi');
  },
  n: 'load_n',
  w: 'load_w',
  nw: 'load_nw',
  store: 'store_tile',
  init: 'initialize_tile',
};

function Degree({ d }: { d: number }) {
  return d <= 1 ? <span className="chip ok">{d === 0 ? tr('no lane', 'không lane nào') : tr('none', 'không')}</span> : <span className="chip fail"><Translated vi={<>{d} chiều</>}>{d}-way</Translated></span>;
}

function BankExplorer() {
  const [kernel, setKernel] = useState<BankKernel>('sheared');
  const [W, setW] = useState<number>(16);
  const [layout, setLayout] = useState<BankLayout>('unpadded');
  const [modelId, setModelId] = useState<BankModel['id']>('nvidia');
  const [access, setAccess] = useState<BankAccess>('write');
  const [k, setK] = useState(5);
  const [warp, setWarp] = useState(0);
  const m = BANK_MODELS[modelId];
  const steps = stepsFor(kernel, access, W);
  const kk = Math.min(k, steps - 1);
  const nw = warpsFor(W, m);
  const ww = Math.min(warp, nw - 1);
  const use = warpUse(kernel, access, layout, W, m, kk, ww);
  const worst = COMPUTE_ACCESSES.map((a) => ({ a, w: worstOf(kernel, a, layout, W, m) }));
  const part = use.lanes.filter((l) => l.bank !== null);
  const B = m.banks;
  const colW = 640 / B;
  const maxStack = Math.max(1, ...use.result.perBank.map((b) => b.lanes.length));
  const chipH = Math.max(5, Math.min(14, Math.floor(180 / maxStack)));

  return (
    <Bench
      title={tr('Which bank does each thread hit?', 'Mỗi luồng chạm bank nào?')}
      task={tr('Is there a width where the sheared tile is conflict-free unpadded on 32 banks? Then pad it, and pad the square kernel, and watch what happens.', 'Có độ rộng nào mà ô đã trượt không đệm vẫn không xung đột trên 32 bank không? Rồi đệm nó, đệm cả kernel vuông, và xem điều gì xảy ra.')}
      foot={
        <>
          <Translated
            vi={
              <>
                Một ô đầy đủ ở phần trong; các lane là các luồng liên tiếp của một khối W luồng. Bank = word mod {B}; các word khác
                nhau trong cùng một bank được phục vụ lần lượt (quy tắc được NVIDIA ghi nhận). Chế độ xem 64 bank áp dụng quy tắc đó
                cho số bank của AMD: AMD không ghi nhận một truy cập 64 lane được chia như thế nào, nên đó là một mô hình. Được tính
                bởi <code>src/sim/wavebanks.ts</code>.
              </>
            }
          >
          One full interior tile; lanes are consecutive threads of a block of W. Bank = word mod {B}; distinct words in one bank are
          served one after another (NVIDIA's documented rule). The 64-bank view applies that rule to an AMD bank count: AMD does not
          document how a 64-lane access is split, so it is a model. Computed by <code>src/sim/wavebanks.ts</code>.
          </Translated>
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label={tr('Kernel', 'Kernel')}>
          {KERNELS.map((x) => <button key={x} aria-pressed={kernel === x} onClick={() => setKernel(x)}><Translated vi={x === 'square' ? 'ô vuông' : 'ô đã trượt'}>{x} tiles</Translated></button>)}
        </span>
        <span className="seg" role="group" aria-label={tr('Tile width', 'Độ rộng ô')}>
          {BANK_WIDTHS.map((x) => <button key={x} aria-pressed={W === x} onClick={() => setW(x)}>W = {x}</button>)}
        </span>
        <label className="field">
          <Translated vi={"\"Cách bố trí\""}>
          Layout
          </Translated>
          <select value={layout} aria-label={tr('layout', 'cách bố trí')} onChange={(e) => setLayout(e.target.value as BankLayout)}>
            {LAYOUTS.map((x) => <option key={x} value={x}>{tr(LAYOUT_LABEL[x], ({ unpadded: 'không đệm', rowgap: 'một khoảng trống mỗi hàng', pad: 'pad() của sách' } as Record<BankLayout, string>)[x])}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi={"\"Bank\""}>
          Banks
          </Translated>
          <select value={modelId} aria-label={tr('banks', 'bank')} onChange={(e) => setModelId(e.target.value as BankModel['id'])}>
            {Object.values(BANK_MODELS).map((x) => <option key={x.id} value={x.id}>{tr(x.label, ({ nvidia: '32 bank, warp 32 lane (NVIDIA)', amd64: '64 bank, wavefront 64 lane (một mô hình AMD)' } as Record<BankModel['id'], string>)[x.id])}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi={"\"Truy cập\""}>
          Access
          </Translated>
          <select value={access} aria-label={tr('access', 'truy cập')} onChange={(e) => setAccess(e.target.value as BankAccess)}>
            {(['write', 'n', 'w', 'nw', 'store', 'init'] as BankAccess[]).map((x) => <option key={x} value={x}>{ACCESS_SHORT[x]}</option>)}
          </select>
        </label>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          {access === 'store' || access === 'init' ? tr('tile row', 'hàng của ô') : tr('step d_tile', 'bước d_tile')} = {kk}
          <input type="range" min={0} max={steps - 1} value={kk} aria-label={tr('step', 'bước')} onChange={(e) => setK(+e.target.value)} />
        </label>
        {nw > 1 ? (
          <label className="field">
            <Translated vi={"\"Warp\""}>
            Warp
            </Translated>
            <select value={ww} aria-label={tr('warp', 'warp')} onChange={(e) => setWarp(+e.target.value)}>
              {Array.from({ length: nw }, (_, i) => <option key={i} value={i}>{i} {tr('(threads', '(luồng')} {i * m.lanes} {tr('to', 'đến')} {Math.min(W, (i + 1) * m.lanes) - 1})</option>)}
            </select>
          </label>
        ) : null}
        <span data-degree={use.degree}><Translated vi="Truy cập này:">This access:</Translated> <Degree d={use.degree} /></span>
      </div>

      <div className="mx-label"><Translated vi={<>Các lane theo bank ({part.length} trên {use.lanes.length} lane dùng ô; một cột có nhiều hơn một word là một xung đột)</>}>Lanes by bank ({part.length} of {use.lanes.length} lanes use the tile; a column with more than one word is a conflict)</Translated></div>
      <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 660 ${chipH * maxStack + 34}`} style={{ width: '100%', minWidth: 660, maxWidth: 660, display: 'block' }} role="img" aria-label={tr('Lanes grouped by bank', 'Các lane nhóm theo bank')} data-banks>
        {use.result.perBank.map((b) => {
          const bad = b.words.length > 1;
          return (
            <g key={b.bank} data-bank={b.bank} data-lanes={b.lanes.length} data-words={b.words.length}>
              {b.bank % (B === 32 ? 4 : 8) === 0 ? <text x={10 + b.bank * colW + colW / 2} y={10} fontSize="8.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{b.bank}</text> : null}
              <rect x={10 + b.bank * colW} y={14} width={colW - 1} height={chipH * maxStack + 4} fill={bad ? 'var(--fail-wash)' : 'none'} stroke="var(--rule-soft)" />
              {b.lanes.map((lane, j) => {
                const l = use.lanes[lane];
                const wordIdx = b.words.indexOf(l.word!);
                const dark = !bad || wordIdx % 2 === 1;
                return (
                  <g key={lane} data-thread={l.thread} data-word={l.word ?? ''}>
                    <rect x={11 + b.bank * colW} y={16 + j * chipH} width={colW - 3} height={chipH - 1} rx="1"
                      fill={bad ? (wordIdx % 2 ? 'var(--fail)' : 'var(--fail-wash)') : 'var(--shared)'} stroke={bad ? 'var(--fail)' : 'var(--shared)'} strokeWidth="0.5">
                      <title>{tr(`thread ${l.thread}: slot (${l.slot![0]}, ${l.slot![1]}), word ${l.word}, bank ${l.bank}`, `luồng ${l.thread}: ô nhớ (${l.slot![0]}, ${l.slot![1]}), word ${l.word}, bank ${l.bank}`)}</title>
                    </rect>
                    {chipH >= 9 && colW >= 16 ? (
                      <text x={11 + b.bank * colW + (colW - 3) / 2} y={16 + j * chipH + chipH - 3} fontSize={Math.min(9, chipH - 2)} textAnchor="middle"
                        style={{ fill: dark ? 'var(--page)' : 'var(--fail)', pointerEvents: 'none' }}>{l.thread}</text>
                    ) : null}
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      </div>
      <p className="muted" style={{ fontSize: '0.75rem', margin: '0.3rem 0 0' }}><Translated vi="Mỗi mảnh là một luồng, ghi số của nó; rê chuột lên một mảnh để xem ô nhớ, word và bank của nó. Trên màn hình hẹp, biểu đồ cuộn ngang.">Each chip is a thread, labelled with its number; hover one for its slot, word and bank. On a narrow screen the chart scrolls sideways.</Translated></p>
      <div style={{ marginTop: '0.8rem', maxWidth: '30rem' }}>
        <div>
          <div className="mx-label"><Translated vi="Tệ nhất qua mọi bước và warp, với kernel, cách bố trí và W này">Worst over every step and warp, this kernel, layout and W</Translated></div>
          <table className="data" style={{ margin: '0 0 0.8rem' }} data-worst>
            <tbody>
              {worst.map(({ a, w }) => (
                <tr key={a} data-access={a} data-worst-degree={w.degree}>
                  <td>{ACCESS_SHORT[a]}</td>
                  <td><Degree d={w.degree} /></td>
                  <td className="muted">{w.degree > 1 ? tr(`first at step ${w.k}${nw > 1 ? `, warp ${w.warp}` : ''}`, `lần đầu ở bước ${w.k}${nw > 1 ? `, warp ${w.warp}` : ''}`) : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="mx-label" style={{ marginTop: '1rem' }}><Translated vi={<>Mọi độ rộng ({({ nvidia: '32 bank, warp 32 lane (NVIDIA)', amd64: '64 bank, wavefront 64 lane (một mô hình AMD)' } as Record<BankModel['id'], string>)[m.id]}): tệ nhất trong lần ghi và ba lần nạp</>}>Every width ({m.label}): worst of the write and the three loads</Translated></div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }} data-summary>
          <thead>
            <tr><th><Translated vi="kernel">kernel</Translated></th><th><Translated vi="cách bố trí">layout</Translated></th>{BANK_WIDTHS.map((x) => <th key={x} className="num">W = {x}</th>)}</tr>
          </thead>
          <tbody>
            {KERNELS.flatMap((kn) =>
              LAYOUTS.map((ly) => (
                <tr key={kn + ly} style={kn === kernel && ly === layout ? { background: 'var(--rule-soft)' } : undefined}>
                  <td><Translated vi={kn === 'square' ? 'vuông' : 'đã trượt'}>{kn}</Translated></td>
                  <td>{tr(LAYOUT_LABEL[ly], ({ unpadded: 'không đệm', rowgap: 'một khoảng trống mỗi hàng', pad: 'pad() của sách' } as Record<BankLayout, string>)[ly])}</td>
                  {BANK_WIDTHS.map((x) => {
                    const [lo, hi] = computeRange(kn, ly, x, m);
                    return (
                      <td key={x} className="num" data-cell={`${kn}/${ly}/${x}`} data-value={hi}>
                        {hi <= 1 ? <span className="t-ok">1</span> : <span className="t-fail">{lo === hi ? hi : `${lo}–${hi}`}</span>}
                      </td>
                    );
                  })}
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions

export const questionsS166: Record<string, Question> = {
  'ch16/16.6/q-launches': {
    kind: 'numeric',
    id: 'ch16/16.6/q-launches',
    prompt: (
      <Translated
        vi={
          <>
            <code>L_seq = 1000</code>, <code>W = 32</code>, ô vuông. Bảng cần bao nhiêu lần khởi chạy kernel?
          </>
        }
      >
      <>
        <code>L_seq = 1000</code>, <code>W = 32</code>, square tiles. How many kernel launches does the table take?
      </>
      </Translated>
    ),
    answer: 63,
    unit: 'launches',
    vi: { unit: 'lần khởi chạy' },
    explain: (
      <Translated
        vi={
          <p>
            T = ⌈1000 / 32⌉ = 32 ô mỗi cạnh (hàng và cột ô cuối cùng không trọn), và có 2T − 1 = 63 mặt sóng của ô. Phương án cơ sở
            một lần khởi chạy mỗi đường chéo phụ cần 2·1000 − 1 = 1,999. Ô đã trượt cùng độ rộng cần 3T − 1 = 95.
          </p>
        }
      >
      <p>
        T = ⌈1000 / 32⌉ = 32 tiles per side (the last row and column of tiles are partial), and there are 2T − 1 = 63 fronts
        of tiles. The one-launch-per-anti-diagonal baseline needs 2·1000 − 1 = 1,999. Sheared tiles of the same width need
        3T − 1 = 95.
      </p>
      </Translated>
    ),
    source: 'ch. 16 §16.6, pp. 384–386; §16.7, p. 392', // i18n-ignore: citation
  },
  'ch16/16.6/q-exact-blocks': {
    kind: 'numeric',
    id: 'ch16/16.6/q-exact-blocks',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập trong phần chữ của sách: khởi chạy đúng số khối mà một mặt sóng cần. Với <code>T = 5</code> ô mỗi cạnh, ô
            vuông, mặt sóng <code>d = 6</code>: bao nhiêu khối?
          </>
        }
      >
      <>
        The book's in-text exercise: launch exactly the blocks a front needs. With <code>T = 5</code> tiles per side, square
        tiles, front <code>d = 6</code>: how many blocks?
      </>
      </Translated>
    ),
    answer: 3,
    unit: 'blocks',
    vi: { unit: 'khối' },
    explain: (
      <Translated
        vi={
          <p>
            Mặt sóng gồm các ô có tile_row + tile_col = 6 và cả hai trong khoảng 0 đến 4: (2, 4), (3, 3) và (4, 2). Nói chung số
            ô là min(d, T − 1) − max(0, d − T + 1) + 1 = 4 − 2 + 1 = 3, và khối b nhận tile_row = max(0, d − T + 1) + b, nên khối 0
            bắt đầu ở hàng 2. Phiên bản của sách khởi chạy cả 5 khối và để hai khối thấy tile_col nằm ngoài phạm vi rồi thoát.
          </p>
        }
      >
      <p>
        The front holds the tiles with tile_row + tile_col = 6 and both in 0 to 4: (2, 4), (3, 3) and (4, 2). In general the
        count is min(d, T − 1) − max(0, d − T + 1) + 1 = 4 − 2 + 1 = 3, and block b takes tile_row = max(0, d − T + 1) + b, so
        block 0 starts at row 2. The book's version launches all 5 blocks and lets two of them find tile_col out of range and
        exit.
      </p>
      </Translated>
    ),
    source: 'ch. 16 §16.6, p. 386', // i18n-ignore: citation
  },
  'ch16/16.6/q-idle': {
    kind: 'numeric',
    id: 'ch16/16.6/q-idle',
    prompt: (
      <Translated vi={<>Một ô vuông 16 × 16 đầy đủ, một khối 16 luồng. Tính trung bình qua các bước trong ô, có bao nhiêu luồng bận?</>}>
        <>A full 16 × 16 square tile, a block of 16 threads. On average over its intra-tile steps, how many threads are busy?</>
      </Translated>
    ),
    answer: 8.26,
    tol: 0.05,
    unit: 'threads',
    vi: { unit: 'luồng' },
    explain: (
      <Translated
        vi={
          <p>
            256 ô con qua 2·16 − 1 = 31 bước: 8.26 mỗi bước (Eq. 16.3). Các bước mang 1, 2, …, 16, …, 2, 1 ô con, nên trung bình
            khoảng một nửa số luồng rảnh. Một ô đã trượt tính cùng 256 ô con trong 16 bước với cả 16 luồng đều bận (Eq. 16.4).
          </p>
        }
      >
      <p>
        256 cells over 2·16 − 1 = 31 steps: 8.26 per step (Eq. 16.3). The steps carry 1, 2, …, 16, …, 2, 1 cells, so about half
        the threads idle on average. A sheared tile computes the same 256 cells in 16 steps with all 16 threads busy (Eq. 16.4).
      </p>
      </Translated>
    ),
    source: 'ch. 16 §16.7, Eqs. 16.3–16.4, p. 396', // i18n-ignore: citation
  },
  'ch16/16.6/q-store': {
    kind: 'choice',
    id: 'ch16/16.6/q-store',
    prompt: (
      <Translated
        vi={
          <>
            Trong <code>store_tile</code>, luồng <code>t</code> ghi cột <code>t</code> của mọi hàng trong ô, dù trong vòng lặp tính
            toán nó sở hữu hàng <code>t</code>. Vì sao đổi vai trò?
          </>
        }
      >
      <>
        In <code>store_tile</code>, thread <code>t</code> writes column <code>t</code> of every tile row, although in the compute
        loop it owned row <code>t</code>. Why switch roles?
      </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Để các luồng liên tiếp ghi các địa chỉ liên tiếp của một hàng trong bảng, và mỗi hàng của ô đi ra được hợp nhất."}>{'So consecutive threads write consecutive addresses of one table row, and each tile row leaves coalesced.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Để các lần ghi tránh xung đột bank, thứ mà bộ nhớ toàn cục cũng có trên một warp giống như bộ nhớ chia sẻ."}>{'So the writes avoid bank conflicts, which global memory has across a warp in the same way as shared memory does.'}</Translated>,
        why: <Translated vi={"Bank là một cấu trúc của bộ nhớ chia sẻ. Điều quan trọng với bộ nhớ toàn cục là các địa chỉ của một warp có liền nhau hay không."}>{'Banks are a shared-memory structure. What matters for global memory is whether a warp\'s addresses are contiguous.'}</Translated>,
      },
      {
        label: <Translated vi={"Vì rào chắn ở cuối vòng lặp tính toán đòi hỏi mọi luồng đổi vai trò."}>{'Because the barrier at the end of the compute loop requires every thread to change its role.'}</Translated>,
        why: <Translated vi={"Rào chắn chỉ nói rằng mọi luồng đã đến; nó không áp đặt cách ánh xạ luồng với dữ liệu."}>{'A barrier only says that every thread has arrived; it imposes no mapping of threads to data.'}</Translated>,
      },
      {
        label: <Translated vi={"Để mỗi luồng phát ra số lệnh ghi bằng một nửa so với khi ghi theo hàng."}>{'So that each thread issues half as many store instructions as it would by rows.'}</Translated>,
        why: <Translated vi={"Dù cách nào, mỗi luồng vẫn ghi W giá trị, mỗi hàng của ô một giá trị."}>{'Each thread still writes W values, one per tile row, either way.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Nếu giữ theo hàng của ô, luồng t sẽ ghi hàng t: ở mỗi lệnh ghi, các địa chỉ của warp sẽ cách nhau một hàng của bảng,
            L word, và không gì được hợp nhất. Đổi sang cột khiến warp ghi W word kề nhau của một hàng. Đó là corner turning của
            chương 6: thứ tự khó chịu diễn ra trong bộ nhớ chia sẻ, và bộ nhớ toàn cục thấy thứ tự thuận lợi.
          </p>
        }
      >
      <p>
        Kept as tile rows, thread t would write row t: at each store instruction the warp's addresses would be one table row
        apart, L words, and nothing would coalesce. Switching to columns makes the warp write W neighbouring words of one row.
        It is chapter 6's corner turning: the awkward order happens in shared memory, and global memory sees the friendly one.
      </p>
      </Translated>
    ),
    source: 'ch. 16 §16.6, p. 389, Fig. 16.12; ch. 6 §6.1', // i18n-ignore: citation
  },
  'ch16/16.6/q-nw-offset': {
    kind: 'choice',
    id: 'ch16/16.6/q-nw-offset',
    prompt: (
      <Translated
        vi={
          <>
            Một ô đã trượt được lưu dưới dạng hình vuông: ô con (r_tile, q_tile) nằm ở cột lưu trữ q_tile = d_tile, và cột của nó
            trong bảng là base + q_tile − r_tile. Láng giềng tây bắc của nó ở cột lưu trữ nào của hàng r_tile − 1?
          </>
        }
      >
      <>
        A sheared tile is stored as a square: cell (r_tile, q_tile) sits in stored column q_tile = d_tile, and its table column is
        base + q_tile − r_tile. In which stored column of row r_tile − 1 is its north-west neighbour?
      </>
      </Translated>
    ),
    options: [
      { label: <code>q_tile − 2</code>, correct: true }, // i18n-ignore: code
      { label: <code>q_tile − 1</code>, why: <Translated vi="Đó là chỗ của láng giềng bắc: cùng cột trong bảng, lên một hàng.">{'That is where the north neighbour is: same table column, one row up.'}</Translated> },
      { label: <code>q_tile</code>, why: <Translated vi="Ở hàng r_tile − 1, cột lưu trữ q_tile giữ cột base + q_tile − r_tile + 1 của bảng, bên phải chính ô con đó một cột.">{'In row r_tile − 1, stored column q_tile holds table column base + q_tile − r_tile + 1, one to the right of the cell itself.'}</Translated> },
      { label: <code>q_tile + 1</code>, why: <Translated vi="Đó là cách láng giềng tây bắc hai cột về bên phải trong bảng.">{'That is two table columns to the right of the north-west neighbour.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Láng giềng tây bắc ở trên một hàng và lùi một cột của bảng: cột base + q_tile − r_tile − 1. Ở hàng r_tile − 1, cột
            lưu trữ q′ giữ cột base + q′ − (r_tile − 1) của bảng. Hai giá trị bằng nhau khi q′ = q_tile − 2. Cùng lập luận đặt láng
            giềng bắc ở q_tile − 1 (lên một hàng, cùng cột bảng) và láng giềng tây ở q_tile − 1 (cùng hàng). Vậy{' '}
            <code>load_nw</code> chỉ nằm trong ô từ q_tile = 2: đó là phép kiểm tra <code>q_tile == 1</code> trong Fig. 16.17.
          </p>
        }
      >
      <p>
        The north-west neighbour is one row up and one table column left: table column base + q_tile − r_tile − 1. In row
        r_tile − 1, stored column q′ holds table column base + q′ − (r_tile − 1). The two are equal when q′ = q_tile − 2. The same
        reasoning puts the north neighbour at q_tile − 1 (row up, same table column) and the west neighbour at q_tile − 1 (same
        row). So <code>load_nw</code> stays inside the tile only from q_tile = 2: that is the <code>q_tile == 1</code> test in
        Fig. 16.17.
      </p>
      </Translated>
    ),
    source: 'ch. 16 §16.7, Fig. 16.17, p. 394', // i18n-ignore: citation
  },
  'ch16/16.6/q-zero': {
    kind: 'choice',
    id: 'ch16/16.6/q-zero',
    prompt: (
      <Translated
        vi={
          <>
            Kernel trượt đặt ô của nó về 0 trước khi tính (Fig. 16.19). Vì sao điều đó cho giá trị đúng ở rìa trái của bảng, và
            khi nào thì không?
          </>
        }
      >
      <>
        The sheared kernel zeroes its tile before computing (Fig. 16.19). Why does that give the right values at the table's
        left edge, and when would it not?
      </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Các ô nhớ cho cột 0 không bao giờ được tính nhưng lại được cột 1 đọc, và 0 là giá trị biên của hệ thức truy hồi này; một biên khác sẽ cần giá trị khác."}>{'Slots for column 0 are never computed yet are read by column 1, and 0 is this recurrence\'s boundary value; another boundary would need other values.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Dù sao bộ nhớ chia sẻ trên GPU cũng khởi đầu bằng 0; lời gọi chỉ làm giả định đó tường minh, giúp người đọc mã và không đổi giá trị nào."}>{'Shared memory starts at zero on GPUs anyway; the call only makes that assumption explicit, which helps readers of the code and changes no value.'}</Translated>,
        why: <Translated vi={"Sách đưa ra lý do ngược lại: không đặt về 0, các ô nhớ chưa ghi chứa rác mà load_n, load_w và load_nw có thể đọc (p. 395)."}>{'The book gives the opposite reason: without the zeroing, the unwritten slots hold garbage that load_n, load_w and load_nw can read (p. 395).'}</Translated>,
      },
      {
        label: <Translated vi={"Nó bảo vệ các ô con ở các hàng dưới cùng của ô, có đầu vào nằm dưới hàng cuối của bảng, nơi không lần khởi chạy nào ghi gì."}>{'It protects the cells in the tile\'s bottom rows, whose inputs lie below the last row of the table, where no launch writes anything.'}</Translated>,
        why: <Translated vi={"Không đầu vào nào nằm dưới một ô con: mọi đầu vào đều ở phía bắc, tây hoặc tây bắc của nó."}>{'No input lies below a cell: every input is north, west or north-west of it.'}</Translated>,
      },
      {
        label: <Translated vi={"Nó loại bỏ xung đột bank của bước đầu tiên, bằng cách chạm vào mọi bank một lần trước khi vòng lặp tính toán bắt đầu."}>{'It removes the bank conflicts of the first step, by touching every bank once before the compute loop begins.'}</Translated>,
        why: <Translated vi={"Chạm vào bank sớm hơn không thay đổi gì với các truy cập sau; xung đột bank phụ thuộc vào địa chỉ của một lần truy cập."}>{'Touching banks earlier changes nothing about later accesses; bank conflicts depend on the addresses of one access.'}</Translated>,
      },
    ],
    explain: (
      <>
        <Translated
          vi={
            <p>
              Trong một ô ở rìa trái, một luồng có cột trượt q nhỏ hơn 1 không có ô con nào để tính, nên ô nhớ của nó không bao giờ
              được ghi. Một ô con ở cột 1 vẫn đọc đầu vào tây và tây bắc từ các ô nhớ đó (chúng nằm trong cùng ô ở một bước trước).
              Khi đã đặt về 0, chúng đọc ra 0, đúng như hàng 0 và cột 0 chứa. Hãy chạy trình chạy từng bước trên ô rìa trái có và
              không có "đặt ô về 0 trước".
            </p>
          }
        >
        <p>
          In a tile at the left edge, a thread whose sheared column q is below 1 has no cell to compute, so its slot is never
          written. A cell in column 1 still reads its west and north-west inputs from those slots (they are in the same tile at
          an earlier step). Zeroed, they read 0, which is what row 0 and column 0 hold. Run the stepper on the left-edge tile with
          and without "zero the tile first".
        </p>
        </Translated>
        <Translated
          vi={
            <p>
              Ví dụ của chúng tôi, không phải của sách: edit distance có biên D[i][0] = i. Các ô nhớ đặt về 0 sẽ cho nó giá trị sai;
              nó sẽ phải ghi các giá trị biên vào những ô nhớ đó, hoặc đọc các đầu vào ở rìa từ bộ nhớ toàn cục.
            </p>
          }
        >
        <p>
          Our example, not the book's: edit distance has the boundary D[i][0] = i. Zeroed slots would give it wrong values; it
          would have to write the boundary values into those slots, or read edge inputs from global memory.
        </p>
        </Translated>
      </>
    ),
    source: 'ch. 16 §16.7, p. 395, Fig. 16.19; the edit-distance example is ours',
    vi: { source: 'ch. 16 §16.7, p. 395, Fig. 16.19; ví dụ edit distance là của khóa học này' },
  },
  'ch16/16.6/q-hyper-cost': {
    kind: 'choice',
    id: 'ch16/16.6/q-hyper-cost',
    prompt: (
      <Translated vi={<>Phép trượt giảm một nửa số bước bên trong mỗi ô. Khi nào nó có khả năng thua ô vuông?</>}>
        <>Shearing halves the steps inside each tile. When is it likely to lose to square tiles?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Khi T nhỏ: số lần khởi chạy gấp khoảng 1.5 lần, mỗi lần chứa tối đa khoảng T/2 + 1 ô, nên SM rảnh và chi phí khởi chạy cộng dồn."}>{'When T is small: about 1.5 times as many launches, each holding at most about T/2 + 1 tiles, so SMs idle and launch overhead adds up.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Hầu như không bao giờ: giảm một nửa số bước bên trong mỗi ô có lợi hơn cái giá của lịch chạy các ô bên ngoài, bất kể kích thước bảng."}>{'Hardly ever: halving the steps inside every tile outweighs what the schedule of tiles costs outside it, whatever the size of the table.'}</Translated>,
        why: <Translated vi={"Số bước bên trong một ô chỉ là một thừa số; số lần khởi chạy nhân với nó, và số này tăng lên."}>{'The steps inside a tile are only one factor; the number of launches multiplies them, and it rises.'}</Translated>,
      },
      {
        label: <Translated vi={"Khi W là lũy thừa của hai, vì khi đó ô đã trượt có xung đột bank mà ô vuông không có và đệm không loại bỏ được."}>{'When W is a power of two, because the sheared tile then has bank conflicts that the square tile does not have and that padding cannot remove.'}</Translated>,
        why: <Translated vi={"Các xung đột đó có thật (ở phần sau của trang này), nhưng một khoảng trống mỗi hàng loại bỏ chúng ở mọi độ rộng; chúng không quyết định phép trượt có đáng hay không."}>{'Those conflicts are real (later on this page), but one gap per row removes them at every width; they do not decide whether shearing pays.'}</Translated>,
      },
      {
        label: <Translated vi={"Khi hai chuỗi khác nhau nhiều, vì phép trượt chỉ có lợi khi phần lớn ô con được điểm khớp và lấy đầu vào chéo."}>{'When the two sequences differ a lot, because the shear pays off only when most cells score a match and take their diagonal input.'}</Translated>,
        why: <Translated vi={"Các chuỗi thay đổi giá trị trong bảng, không thay đổi ô nào phụ thuộc ô nào hay lịch chạy."}>{'The sequences change the values in the table, not which cells depend on which or the schedule.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bên trong một ô, phép trượt là lợi ích rõ ràng. Bên ngoài, mặt sóng của ô trở nên nông hơn: 3T − 1 lần khởi chạy thay vì
            2T − 1, và tối đa ⌊T/2⌋ + 1 ô mỗi lần khởi chạy thay vì T. Khi T lớn, mỗi lần khởi chạy vẫn lấp đầy thiết bị và các lần
            khởi chạy thêm là cái giá nhỏ. Khi T nhỏ, thiết bị chạy một nhúm khối mỗi lần khởi chạy dù theo cách nào và các lần
            khởi chạy thêm là chi phí thuần. Sách gọi chi phí khởi chạy là hình phạt còn lại, thứ mà sự đồng bộ hóa không cần khởi
            chạy lại của §16.8 loại bỏ.
          </p>
        }
      >
      <p>
        Inside a tile the shear is a clear gain. Outside, the front of tiles gets shallower: 3T − 1 launches instead of 2T − 1,
        and at most ⌊T/2⌋ + 1 tiles per launch instead of T. When T is large, each launch still fills the device and the extra
        launches are a small price. When T is small, the device runs a handful of blocks per launch either way and the extra
        launches are pure cost. The book names launch overhead as the remaining penalty, which §16.8's synchronisation without
        relaunching removes.
      </p>
      </Translated>
    ),
    source: 'ch. 16 §16.7, pp. 391–392, 396; ch16-facts E2', // i18n-ignore: citation
  },
  'ch16/16.6/q-banks16': {
    kind: 'numeric',
    id: 'ch16/16.6/q-banks16',
    prompt: (
      <Translated
        vi={
          <>
            Ô đã trượt, <code>W = 16</code> (một khối 16 luồng), lưu không đệm, 32 bank. Ở bước d_tile, luồng t ghi ô nhớ
            (t, d_tile). Phần cứng bank cần bao nhiêu lượt cho lần ghi đó (1 nghĩa là không xung đột)?
          </>
        }
      >
      <>
        Sheared tile, <code>W = 16</code> (a block of 16 threads), stored unpadded, 32 banks. At step d_tile, thread t writes slot
        (t, d_tile). How many passes does the bank hardware need for that write (1 means no conflict)?
      </>
      </Translated>
    ),
    answer: 8,
    unit: 'passes',
    explain: (
      <Translated
        vi={
          <p>
            Luồng t ghi word 16t + d_tile. Bank = word mod 32, nên các luồng chẵn đều rơi vào bank d_tile và các luồng lẻ vào bank
            d_tile + 16: hai bank, mỗi bank 8 word khác nhau, 8 lượt. Với một khoảng trống mỗi hàng, word là 17t + d_tile; 17 và 32
            không có ước chung, nên 16 luồng chạm 16 bank khác nhau.
          </p>
        }
      >
      <p>
        Thread t writes word 16t + d_tile. Bank = word mod 32, so even threads all land in bank d_tile and odd threads in bank
        d_tile + 16: two banks, 8 distinct words each, 8 passes. With one gap per row the word is 17t + d_tile; 17 and 32 share no
        factor, so the 16 threads hit 16 different banks.
      </p>
      </Translated>
    ),
    source: 'ch. 16 §16.7, pp. 395–396, Fig. 16.20; ch16-facts E1 (our arithmetic on NVIDIA\'s rule)',
    vi: { unit: 'lượt', source: 'ch. 16 §16.7, pp. 395–396, Fig. 16.20; ch16-facts E1 (phép tính của chúng tôi dựa trên quy tắc của NVIDIA)' },
  },
  'ch16/16.6/q-pad-square': {
    kind: 'choice',
    id: 'ch16/16.6/q-pad-square',
    prompt: (
      <Translated vi={<>Kernel ô vuông có nên được đệm một khoảng trống mỗi hàng giống kernel đã trượt không?</>}>
        <>Should the square-tile kernel get the same one-gap-per-row padding as the sheared one?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Không: phép duyệt đường chéo phụ của nó có bước nhảy W − 1 word, vốn đã không xung đột với W = 8 đến 64 trên 32 bank; khi có đệm, bước nhảy thành W."}>{'No: its anti-diagonal walk strides W − 1 words, already conflict-free for W = 8 to 64 on 32 banks; padded, it strides W.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Có: một khoảng trống mỗi hàng trải bất kỳ bước nhảy nào ra nhiều bank hơn, và chỉ tốn một word bộ nhớ chia sẻ mỗi hàng của ô."}>{'Yes: a gap per row spreads any stride over more banks, and it costs only one word of shared memory per tile row.'}</Translated>,
        why: <Translated vi={"Đệm thay đổi bước nhảy. Với kernel vuông, nó biến W − 1, ở đây không có ước chung với 32, thành W, thứ có ước chung: 32 chiều khi W = 32."}>{'Padding changes the stride. For the square kernel it turns W − 1, which shares no factor with 32 here, into W, which does: 32-way at W = 32.'}</Translated>,
      },
      {
        label: <Translated vi={"Có, nhưng chỉ trên GPU AMD: chúng có 64 bank trong khi NVIDIA có 32, nên bước nhảy W − 1 không đệm bị xung đột ở đó."}>{'Yes, but only on AMD GPUs: they have 64 banks where NVIDIA\'s have 32, so the unpadded W − 1 stride conflicts there.'}</Translated>,
        why: <Translated vi={"Một số GPU AMD có 32 bank, một số có 64; trên loại nào cũng vậy, đệm kernel vuông biến một bước nhảy nguyên tố cùng nhau thành lũy thừa của hai."}>{'Some AMD GPUs have 32 banks, some 64; on either, padding the square kernel turns a coprime stride into a power of two.'}</Translated>,
      },
      {
        label: <Translated vi={"Không có khác biệt gì, vì kernel vuông chạm mỗi bank tối đa một lần mỗi truy cập, dù có đệm hay không."}>{'It makes no difference either way, because the square kernel touches each bank at most once per access, whether it is padded or not.'}</Translated>,
        why: <Translated vi={"Không đệm thì đúng; có đệm thì không."}>{'Unpadded that is true; padded it is not.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong kernel vuông, luồng t ở bước k chạm ô nhớ (t, k − t): word t(W − 1) + k. Bước nhảy W − 1 (7, 15, 31, 63) không có
            ước chung với 32, nên một warp chạm các bank khác nhau. Với một khoảng trống mỗi hàng, word thành t·(W + 1) + (k − t) =
            t·W + k: bước nhảy W, một bội hoặc ước của 32, nên xuất hiện xung đột 2, 8 và 32 chiều. Đệm chỉ sửa được cách bố trí
            của ô đã trượt. Một bài thực hành đệm cả hai kernel sẽ làm kernel vuông chậm đi.
          </p>
        }
      >
      <p>
        In the square kernel, thread t at step k touches slot (t, k − t): word t(W − 1) + k. A stride of W − 1 (7, 15, 31, 63)
        shares no factor with 32, so a warp hits distinct banks. With one gap per row the word becomes t·(W + 1) + (k − t) =
        t·W + k: stride W, a multiple or divisor of 32, so 2-, 8- and 32-way conflicts appear. Padding fixes the sheared
        layout only. A lab that pads both kernels makes the square one slower.
      </p>
      </Translated>
    ),
    source: 'ch16-facts E1 (our arithmetic on NVIDIA\'s documented rule)',
    vi: { source: 'ch16-facts E1 (phép tính của chúng tôi dựa trên quy tắc được NVIDIA ghi nhận)' },
  },
  'ch16/16.6/q-dpx': {
    kind: 'choice',
    id: 'ch16/16.6/q-dpx',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 6 của sách: bạn thay <code>max4(0, nw + s, w + g, n + g)</code> bằng{' '}
            <code>__vimax3_s32_relu(nw + s, w + g, n + g)</code> trong tệp <code>.cu</code> của khóa học. Cần kèm theo điều gì?
          </>
        }
      >
      <>
        Book exercise 6: you replace <code>max4(0, nw + s, w + g, n + g)</code> with{' '}
        <code>__vimax3_s32_relu(nw + s, w + g, n + g)</code> in the course's <code>.cu</code> file. What has to go with it?
      </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Một điều kiện chỉ dành cho CUDA với max4 làm nhánh còn lại, vì HIP không khai báo hàm DPX nào, và một phép kiểm tra so với max4 vì một lỗi biên dịch sai đã được báo cáo."}>{'A CUDA-only guard with max4 as the other path, since HIP declares no DPX function, and a test against max4 for a reported miscompile.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Không cần gì: pmpp/vendors/hip.h đổi tên nó thành intrinsic HIP tương ứng, như với các lời gọi runtime."}>{'Nothing: pmpp/vendors/hip.h renames it to the matching HIP intrinsic, as it does for the runtime calls.'}</Translated>,
        why: <Translated vi={"Không có gì để đổi tên thành: không header HIP nào và không hàng nào trong bảng của HIPIFY khai báo một hàm DPX."}>{'There is nothing to rename it to: no HIP header and no row of HIPIFY\'s table declares a DPX function.'}</Translated>,
      },
      {
        label: <Translated vi={"Điểm số mới, vì nó cho kết quả khác max4 bất cứ khi nào có đầu vào âm, nên bản tham chiếu CPU thay đổi."}>{'New scores, because it gives a different result from max4 whenever some of its inputs are negative, so the CPU reference changes.'}</Translated>,
        why: <Translated vi={"Nó tính max(max(max(a, b), c), 0), bằng max4(0, a, b, c) với mọi int: max số nguyên có tính kết hợp và giao hoán."}>{'It computes max(max(max(a, b), c), 0), which equals max4(0, a, b, c) for every int: integer max is associative and commutative.'}</Translated>,
      },
      {
        label: <Translated vi={"Một GPU Hopper trở lên, vì hàm này không biên dịch cho kiến trúc NVIDIA cũ hơn, vốn phải giữ max4."}>{'A Hopper GPU or later, because the function does not compile for an older NVIDIA architecture, which must keep max4.'}</Translated>,
        why: <Translated vi={"NVIDIA cũng liệt kê nó cho các compute capability cũ hơn, chạy bằng vài lệnh thay vì một."}>{'NVIDIA lists it for older compute capabilities too, run as several instructions instead of one.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Giá trị vẫn như cũ; điều thay đổi là tính khả chuyển và độ tin cậy. Tệp <code>.cu</code> cũng được dựng cho HIP và các
            trình giả lập, nên lời gọi DPX nằm sau một điều kiện chỉ dành cho CUDA với <code>max4</code> là nhánh ở mọi nơi khác;
            Metal viết <code>max(max3(x, y, z), 0)</code>. Mã nguồn CCCL của chính NVIDIA nói rằng ptxas trước CUDA 13.4 có thể
            biên dịch sai các chuỗi phụ thuộc của lệnh này trên SM90 và SM103, và một ô Smith-Waterman cấp giá trị cho phép lấy
            max của ô kế tiếp chính là một chuỗi như vậy. Ghi chú đó không có trong release notes của NVIDIA, nên hãy xem nó là
            unverified và kiểm tra so với <code>max4</code>.
          </p>
        }
      >
      <p>
        The value is the same; what changes is portability and trust. The <code>.cu</code> file also builds for HIP and the
        emulators, so the DPX call goes behind a CUDA-only guard with <code>max4</code> as the path everywhere else; Metal writes{' '}
        <code>max(max3(x, y, z), 0)</code>. NVIDIA's own CCCL source says ptxas before CUDA 13.4 could miscompile dependent chains
        of this instruction on SM90 and SM103, and a Smith-Waterman cell feeding the next cell's maximum is such a chain. That note
        is not in NVIDIA's release notes, so treat it as unverified and test against <code>max4</code>.
      </p>
      </Translated>
    ),
    source: 'ch. 16 §16.8, p. 398; exercise 6, p. 400; ch16-facts A2, B2, C2',
    vi: { source: 'ch. 16 §16.8, p. 398; bài tập 6, p. 400; ch16-facts A2, B2, C2' },
  },
  'ch16/16.6/q-ssd': {
    kind: 'choice',
    id: 'ch16/16.6/q-ssd',
    prompt: (
      <Translated
        vi={
          <>
            Prefill Mamba-2 của llama.cpp trên NVIDIA chia chuỗi thành các đoạn 256 token, thực hiện các phép nhân ma trận dày đặc
            bên trong mỗi đoạn, và mang trạng thái hồi quy từ đoạn này sang đoạn khác bằng các lần khởi chạy kernel. Theo thuật
            ngữ của trang này, nó là gì?
          </>
        }
      >
      <>
        llama.cpp's Mamba-2 prefill on NVIDIA splits the sequence into 256-token chunks, does dense matrix products inside each
        chunk, and carries the recurrent state from chunk to chunk with kernel launches. In this page's terms, what is it?
      </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Các ô nối với nhau bằng các lần khởi chạy: mỗi đoạn là một ô, và các lần khởi chạy là rào chắn giữa ô này và ô sau.">{'Tiles joined by launches: each chunk is a tile, and the launches are the barrier between one tile and the next.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Một ô đã trượt: mỗi đoạn được làm nghiêng để mọi luồng có một ô con để tính ở mọi bước."}>{'A sheared tile: each chunk is tilted so that every thread has a cell to compute at every step.'}</Translated>,
        why: <Translated vi={"Không có gì bị trượt: một đoạn là một dãy token liên tiếp, và công việc dày đặc bên trong không có luồng rảnh nào cần tận dụng lại."}>{'Nothing is sheared: a chunk is a run of consecutive tokens, and the dense work inside it has no idle threads to recover.'}</Translated>,
      },
      {
        label: <Translated vi={"Sơ đồ cờ của §16.8: mỗi đoạn quay vòng chờ một cờ mà đoạn trước bật khi đã xong trạng thái của nó."}>{'The flag scheme of §16.8: each chunk spins on a flag that the previous chunk sets once it has finished its state.'}</Translated>,
        why: <Translated vi={"Trạng thái truyền giữa các đoạn qua các lần khởi chạy của vòng lặp host, không qua cờ giữa các khối đang chạy."}>{'The state passes between chunks through the host loop\'s launches, not through flags between running blocks.'}</Translated>,
      },
      {
        label: <Translated vi={"Một cách loại bỏ phụ thuộc dọc chuỗi, để mọi đoạn có thể chạy cùng lúc."}>{'A way to remove the dependency along the sequence, so all the chunks can run at the same time.'}</Translated>,
        why: <Translated vi={"Trạng thái vẫn chảy từ đoạn này sang đoạn khác theo thứ tự. Chỉ công việc bên trong một đoạn trở nên song song."}>{'The state still flows from chunk to chunk in order. Only the work inside a chunk becomes parallel.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hệ thức truy hồi dọc chuỗi không biến mất; nó được cắt thành các ô. Bên trong một đoạn 256 token, công việc là các
            phép nhân ma trận dày đặc, song song như các ô con của một ô; giữa các đoạn, việc cập nhật trạng thái là phụ thuộc, và
            một ranh giới kernel bảo đảm nó, như giữa các mặt sóng của ô ở đây. Kích thước đoạn đánh đổi chi phí bên trong một
            đoạn (mã giới hạn chi phí matmul ở O(chunk²) mỗi đoạn) với số bước tuần tự. Trên HIP, đường này bị loại khi biên dịch.
          </p>
        }
      >
      <p>
        The recurrence along the sequence does not go away; it is cut into tiles. Inside a 256-token chunk the work is dense
        matrix products, as parallel as a tile's cells; between chunks the state update is the dependency, and a kernel boundary
        enforces it, as between fronts of tiles here. The chunk size trades the cost inside a chunk (the code caps the matmul cost
        at O(chunk²) per chunk) against the number of sequential steps. On HIP this path is compiled out.
      </p>
      </Translated>
    ),
    source: 'ch16-facts D1 (llama.cpp ggml/src/ggml-cuda/ssm-scan.cu, read 2026-11-20); the mapping is ours',
    vi: { source: 'ch16-facts D1 (llama.cpp ggml/src/ggml-cuda/ssm-scan.cu, đọc ngày 2026-11-20); mối liên hệ là của khóa học này' },
  },
};

// ---------------------------------------------------------------- the page

function LabListing() {
  const { reports } = useReports();
  const st = labStatuses(reports);
  const cu = useLabFile(LAB_DIR + 'kernel.cu');
  const metal = useLabFile(LAB_DIR + 'kernel.metal');
  const mm = useLabFile(LAB_DIR + 'launch.mm');
  if (!cu || !metal || !mm) return null;
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu, tiers: SW_TIERS, status: st.cu },
        { label: tr('Metal kernel', 'kernel Metal'), code: metal, tiers: SW_TIERS, status: st.metal },
        { label: tr('Metal host', 'host Metal'), code: mm, tiers: {}, status: st.metal },
      ]}
      caption={
        <Translated
          vi={
            <>
              Lời giải tham chiếu của lab 16.1, cả hai kernel, được nạp từ <code>labs/ch16/lab1_wavefront/solution/</code>. Các
              trạng thái lấy từ báo cáo của máy này. Hãy thử bài thực hành trước khi nghiên cứu nó. Bản dựng HIP được dự định biên
              dịch chính tệp CUDA này qua <code>pmpp/vendors/hip.h</code> (unverified: chưa máy ROCm nào chạy nó). Điều đó khả thi ở
              đây vì các kernel chỉ dùng luồng, khối, bộ nhớ chia sẻ và rào chắn, không dùng phần nào khác nhau giữa các backend,
              như intrinsic của warp hay lời gọi DPX.
            </>
          }
        >
        <>
          The reference solution of lab 16.1, both kernels, loaded from <code>labs/ch16/lab1_wavefront/solution/</code>. The
          statuses come from this machine's reports. Try the lab before studying it. The HIP build is meant to compile this same CUDA
          file through <code>pmpp/vendors/hip.h</code> (unverified: no ROCm machine has run it). That works here because the
          kernels use only threads, blocks, shared memory and barriers, and none of the parts that differ, such as warp
          intrinsics or a DPX call.
        </>
        </Translated>
      }
    />
  );
}

export function S166() {
  return (
    <>
      <Translated
        vi={
          <p>
            Ở trang <a href={href('ch16', '16.3')}>16.3</a>, một bảng Smith-Waterman được lấp từng đường chéo phụ một. Trang này
            làm việc đó nhanh lên. Các khối nhận các ô (tile), một lần khởi chạy kernel ngăn cách các mặt sóng của ô, và một rào
            chắn (barrier) ngăn cách các bước bên trong một ô. Rồi sách trượt (shear) các ô để không luồng nào rảnh, và bạn trả
            giá cho điều đó bằng số lần khởi chạy và bằng các bank của bộ nhớ chia sẻ (shared memory). Lab 16.1 dựng cả hai kernel.
          </p>
        }
      >
      <p>
        On <a href={href('ch16', '16.3')}>16.3</a> a Smith-Waterman table filled one anti-diagonal at a time. This page makes
        that fast. Blocks take tiles, a kernel launch separates fronts of tiles, and a barrier separates steps inside a tile.
        Then the book shears the tiles so that no thread idles, and you pay for it in launches and in shared-memory banks. Lab
        16.1 builds both kernels.
      </p>
      </Translated>

      <h2><Translated vi="Phụ thuộc, không kèm sinh học hệ gen">The dependency, without the genomics</Translated></h2>
      <Translated
        vi={
          <p>
            Hai chuỗi độ dài L_seq cho một bảng H gồm L_seq + 1 hàng và cột. Hàng 0 và cột 0 chứa số 0. Mỗi ô còn lại là giá trị
            lớn nhất trong bốn giá trị: số 0; láng giềng tây bắc cộng một điểm số dương khi khớp và âm khi không khớp; láng giềng
            tây trừ một điểm phạt khoảng trống (gap); và láng giềng bắc trừ một điểm phạt khoảng trống.
          </p>
        }
      >
      <p>
        Two sequences of length L_seq give a table H of L_seq + 1 rows and columns. Row 0 and column 0 hold zeros. Every other
        cell is the largest of four values: zero; its north-west neighbour plus a score that is positive on a match and negative
        on a mismatch; its west neighbour minus a gap penalty; and its north neighbour minus a gap penalty.
      </p>
      </Translated>
      <Source section="16.5" pages="382–383" figs={tr('Eq. 16.2, Fig. 16.6', 'Phương trình 16.2, Hình 16.6')} />
      <pre className="wide" style={{ margin: '0 0 1.2rem', fontSize: '0.85rem' }}>
        <code>H[r][q] = max(0, H[r−1][q−1] + s(r, q), H[r][q−1] − gap, H[r−1][q] − gap)</code>
      </pre>
      <Translated
        vi={
          <p>
            Vậy một ô cần các láng giềng bắc, tây và tây bắc và không gì khác. Các ô trên một đường chéo phụ không cần gì từ nhau:
            bắc và tây nằm trên đường chéo phụ trước đó, tây bắc nằm trên đường trước nữa. Các đường chéo phụ chính là các mặt
            sóng. Chúng dài thêm một ô mỗi bước cho đến khi băng qua cả bảng, rồi ngắn dần.
          </p>
        }
      >
      <p>
        So a cell needs its north, west and north-west neighbours and nothing else. Cells on one anti-diagonal need nothing from
        each other: north and west lie on the previous anti-diagonal, north-west on the one before. The anti-diagonals are the
        fronts. They grow by one cell per step until they cross the whole table, then shrink.
      </p>
      </Translated>
      <Source section="16.5" pages="383–384" />
      <Translated
        vi={
          <p>
            Ý nghĩa của điểm số, cách chọn sơ đồ chấm điểm, và bước truy vết ngược để khôi phục một phép căn chỉnh từ bảng đã xong
            đều bị bỏ ra: chúng là sinh học và một phép duyệt tuần tự, và không phần nào thay đổi cấu trúc song song. Bài thực
            hành cố định các giá trị ví dụ của sách thành hằng số: +{MATCH} khi khớp, {MISMATCH} khi không khớp, {-GAP} cho một
            khoảng trống.
          </p>
        }
      >
      <p>
        What the score means, how a scoring scheme is chosen, and the traceback that recovers an alignment from the finished
        table are left out: they are biology and a sequential walk, and none of it changes the parallel structure. The lab fixes
        the book's example values as constants: +{MATCH} for a match, {MISMATCH} for a mismatch, {-GAP} for a gap.
      </p>
      </Translated>
      <Source section="16.5" pages="382–384" />
      <Aside>
        <Translated
          vi={
            <p>
              Sách đặt tên hai điểm phạt khoảng trống theo một cách ở Eq. 16.2 và theo cách ngược lại ở Fig. 16.6 và trong kernel.
              Với các điểm phạt bằng nhau thì không gì thay đổi; trang tổng kết liệt kê nó cùng các chỗ sơ suất khác của chương.
            </p>
          }
        >
        <p>
          The book names the two gap penalties one way in Eq. 16.2 and the other way in Fig. 16.6 and the kernel. With equal
          penalties nothing changes; the summary page lists it with the chapter's other slips.
        </p>
        </Translated>
      </Aside>

      <h2><Translated vi="Một lần khởi chạy mỗi mặt sóng, rồi một khối mỗi ô">One launch per front, then one block per tile</Translated></h2>
      <Translated
        vi={
          <p>
            Bản chuyển thẳng khởi chạy một kernel cho mỗi đường chéo phụ với một luồng mỗi ô, và mọi luồng đọc ba đầu vào của nó
            từ bộ nhớ toàn cục (global memory), nơi hai lần khởi chạy trước để chúng lại. Việc kernel kết thúc là rào chắn giữa
            các mặt sóng. Như vậy có 2·L_seq − 1 lần khởi chạy, lần đầu và lần cuối chỉ có vài luồng, và không gì được tái sử dụng
            từ bộ nhớ nhanh.
          </p>
        }
      >
      <p>
        The direct port launches a kernel per anti-diagonal with one thread per cell, and every thread reads its three inputs from
        global memory, where the previous two launches left them. Kernel termination is the barrier between fronts. That is 2·L_seq
        − 1 launches, the first and last of them with only a few threads, and nothing is reused from fast memory.
      </p>
      </Translated>
      <Source section="16.6" pages="384" figs={tr('Fig. 16.7(a)', 'Hình 16.7(a)')} />
      <Translated
        vi={
          <p>
            Phân ô (tiling) mức khối cắt phần trong thành các ô W × W và giao mỗi ô cho một khối W luồng. Bên trong một ô, thứ tự
            đường chéo phụ lại chạy, ở một mức thấp hơn: luồng t sở hữu hàng t của ô, các bước chạy dọc các đường chéo phụ của ô,
            và một rào chắn ngăn cách chúng. Giữa các ô, mặt sóng là các đường chéo phụ của ô, và việc kernel kết thúc vẫn ngăn
            cách chúng. Vậy có hai mức mặt sóng: với T = ⌈L_seq / W⌉ ô mỗi cạnh, có 2T − 1 lần khởi chạy, và bên trong mỗi ô có
            2W − 1 bước được ngăn cách bởi rào chắn.
          </p>
        }
      >
      <p>
        Block-level tiling cuts the interior into W × W tiles and gives each tile to a block of W threads. Inside a tile the
        anti-diagonal order runs again, one level down: thread t owns tile row t, the steps run along the tile's anti-diagonals,
        and a barrier separates them. Between tiles the fronts are anti-diagonals of tiles, and kernel termination still separates
        those. So there are two levels of fronts: with T = ⌈L_seq / W⌉ tiles per side, 2T − 1 launches, and inside each tile 2W −
        1 barrier-separated steps.
      </p>
      </Translated>
      <Source section="16.6" pages="384–386" figs={tr('Figs. 16.7(b), (c), 16.8', 'Hình 16.7(b), (c), 16.8')} />
      <Translated
        vi={
          <p>
            <b>Khối với ô.</b> Lần khởi chạy d chạy các ô có tile_row + tile_col = d. Sách khởi chạy T khối mỗi lần và để những
            khối có tile_col nằm ngoài 0 đến T − 1 thoát ra. Bài thực hành khởi chạy đúng các ô của mặt sóng, chính là bài tập sách
            đặt ra ở p. 386: khối b nhận <code>tile_row = firstRow + b</code> và <code>tile_col = d − tile_row</code>, với
            firstRow = max(0, d − T + 1).
          </p>
        }
      >
      <p>
        <b>Block to tile.</b> Launch d runs the tiles with tile_row + tile_col = d. The book launches T blocks every time and lets
        those whose tile_col falls outside 0 to T − 1 exit. The lab launches exactly the tiles of the front, which is the exercise
        the book sets on p. 386: block b takes <code>tile_row = firstRow + b</code> and <code>tile_col = d − tile_row</code>, with
        firstRow = max(0, d − T + 1).
      </p>
      </Translated>
      <Source section="16.6" pages="386–387" figs={tr('Fig. 16.9', 'Hình 16.9')} />
      <Translated
        vi={
          <p>
            <b>Luồng với ô con.</b> Ở bước trong ô d_tile, luồng r_tile làm việc trên cột <code>q_tile = d_tile − r_tile</code>{' '}
            của ô, nếu nó nằm trong ô. Ô con tương ứng trong bảng là <code>r = W·tile_row + r_tile + 1</code>,{' '}
            <code>q = W·tile_col + q_tile + 1</code>; phần + 1 bỏ qua hàng và cột biên, vốn không bao giờ được tính.
          </p>
        }
      >
      <p>
        <b>Thread to cell.</b> At intra-tile step d_tile, thread r_tile works on tile column <code>q_tile = d_tile − r_tile</code>,
        if that lies inside the tile. Its table cell is <code>r = W·tile_row + r_tile + 1</code>,{' '}
        <code>q = W·tile_col + q_tile + 1</code>; the + 1 skips the boundary row and column, which are never computed.
      </p>
      </Translated>
      <Source section="16.6" pages="387–388" />
      <Translated
        vi={
          <p>
            <b>Đầu vào đến từ đâu.</b> Ba láng giềng của một ô con nằm trong chính ô của nó, trừ khi ô con nằm ở hàng trên cùng
            hoặc cột trái của ô. Những đầu vào ở rìa đó được một lần khởi chạy trước ghi, nên chúng được đọc từ bộ nhớ toàn cục.
            Những đầu vào còn lại được ghi vào ô trong bộ nhớ chia sẻ ở một bước trước, và rào chắn đã làm chúng hiện ra.
          </p>
        }
      >
      <p>
        <b>Where the inputs come from.</b> A cell's three neighbours are in its own tile unless the cell sits on the tile's top row
        or left column. Those edge inputs were written by an earlier launch, so they are read from global memory. The others were
        written into the tile in shared memory at an earlier step, and the barrier made them visible.
      </p>
      </Translated>
      <Source section="16.6" pages="388" figs={tr('Fig. 16.10', 'Hình 16.10')} />
      <Translated
        vi={
          <p>
            <b>Lưu trữ.</b> Khi các bước xong, <code>store_tile</code> sao chép ô vào bảng. Trong vòng lặp tính toán, luồng t sở
            hữu một hàng của ô; khi lưu nó nhận một cột, nên với mỗi hàng của ô, W luồng ghi W địa chỉ kề nhau.
          </p>
        }
      >
      <p>
        <b>Storing.</b> When the steps are done, <code>store_tile</code> copies the tile into the table. In the compute loop thread t
        owned a tile row; in the store it takes a column, so for each tile row the W threads write W neighbouring addresses.
      </p>
      </Translated>
      <Source section="16.6" pages="389" figs={tr('Figs. 16.11, 16.12', 'Hình 16.11, 16.12')} />
      <LabListing />
      <Translated vi={<p>Những chỗ mã của bài thực hành khác Figs. 16.8 đến 16.19 của sách, và vì sao:</p>}><p>Where the lab's code differs from the book's Figs. 16.8 to 16.19, and why:</p></Translated>
      <ul>
        <li>
          <Translated
            vi={
              <>
                <code>tile_col</code> và <code>q_tile</code> được tính bằng <code>int</code>. Sách trừ các biến dựng sẵn unsigned
                (<code>d − blockIdx.x</code>) và lưu kết quả vào một <code>int</code>; khi đó một kết quả âm là một phép chuyển đổi
                do cài đặt quyết định (implementation-defined) trong C++17 mà các bài thực hành biên dịch theo.
              </>
            }
          >
          <code>tile_col</code> and <code>q_tile</code> are computed in <code>int</code>. The book subtracts unsigned built-ins
          (<code>d − blockIdx.x</code>) and stores the result in an <code>int</code>; a negative result is then an
          implementation-defined conversion in the C++17 the labs compile as.
          </Translated>
        </li>
        <li><Translated vi="Mỗi lần khởi chạy có đúng các khối của mặt sóng của nó, thay vì T khối trong đó vài khối thoát ra.">Each launch has exactly the blocks of its front, instead of T blocks of which some exit.</Translated></li>
        <li><Translated vi="Các hàm phụ nhận bước nhảy hàng của ô, nên cùng mã phục vụ được ô có đệm của phần về bank bên dưới.">The helpers take the tile's row stride, so the same code serves the padded tile of the bank section below.</Translated></li>
        <li>
          <Translated
            vi={
              <>
                <code>store_tile</code> đã trượt kiểm tra <code>q &gt;= 1</code>. Fig. 16.18 để q âm cho một phép so sánh unsigned
                với L, và q = 0 sẽ chép một ô nhớ đã đặt về 0 đè lên biên.
              </>
            }
          >
          The sheared <code>store_tile</code> tests <code>q &gt;= 1</code>. Fig. 16.18 leaves negative q to an unsigned comparison
          with L, and q = 0 would copy a zeroed slot onto the boundary.
          </Translated>
        </li>
        <li><Translated vi={<>Các điểm số đến từ <code>sw.h</code>, trong khi sách để MATCH, MISMATCH, INSERTION và DELETION không được định nghĩa.</>}>The scores come from <code>sw.h</code>, where the book leaves MATCH, MISMATCH, INSERTION and DELETION undefined.</Translated></li>
      </ul>
      <Beyond title={tr("Corrected, not the book's", 'Đã sửa, không phải của sách')}>
        <Translated vi={<p>Điểm về chuyển đổi unsigned là ch16-facts A3; phần còn lại là lựa chọn của khóa học này cho bài thực hành.</p>}><p>The unsigned-conversion point is ch16-facts A3; the rest are the course's choices for the lab.</p></Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Trên Metal, ô được định kích thước lúc khởi chạy: một con trỏ <code>[[threadgroup(0)]]</code> trong kernel, và{' '}
            <code>setThreadgroupMemoryLength</code> trên encoder, theo bội số của 16 byte. Hàm phụ không thể khai báo bộ nhớ
            threadgroup, nên kernel khai báo ô và truyền nó vào. Rào chắn của nó chỉ cần <code>mem_threadgroup</code>: các đầu vào
            ở rìa trong bộ nhớ device được các dispatch trước ghi, và một dispatch mỗi mặt sóng trong một compute encoder tuần tự
            chính là rào chắn toàn cục. Một ô W × W kiểu <code>int</code> là 4W² byte, nên W quyết định một khối xin bao nhiêu bộ
            nhớ chia sẻ.
          </p>
        }
      >
      <p>
        On Metal the tile is sized at launch: a <code>[[threadgroup(0)]]</code> pointer in the kernel, and{' '}
        <code>setThreadgroupMemoryLength</code> on the encoder, in multiples of 16 bytes. Helpers cannot declare threadgroup
        memory, so the kernel declares the tile and passes it in. Its barrier needs only <code>mem_threadgroup</code>: the
        edge inputs in device memory were written by earlier dispatches, and one dispatch per front in one serial compute
        encoder is the global barrier. A W × W tile of <code>int</code> is 4W² bytes, so W sets how much shared memory a
        block asks for.
      </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch16-facts E3, C3, C4; ch05-facts A3, C1, C8)', 'ngày 2026-11-20 (ch16-facts E3, C3, C4; ch05-facts A3, C1, C8)')}>
        <Translated
          vi={
            <p>
              CUDA cần bật tường minh khi vượt quá 48 KB mỗi khối (W = 128 là 64 KiB). Trần threadgroup được tài liệu của Metal ghi
              nhận là 32 KB trên Apple4 trở đi.
            </p>
          }
        >
        <p>
          CUDA needs an explicit opt-in above 48 KB per block (W = 128 is 64 KiB). Metal's documented threadgroup ceiling is
          32 KB on Apple4 and later.
        </p>
        </Translated>
      </Today>

      <h3><Translated vi="Các mặt sóng của ô">Fronts of tiles</Translated></h3>
      <Translated
        vi={
          <p>
            Trình khám phá cho thấy cả hai mức cùng lúc: bảng với các ô của nó, các ô của mỗi lần khởi chạy, và các luồng bận ở
            mỗi bước bên trong một ô đầy đủ. Với W = 1, một ô là một ô con, và sơ đồ vuông chính là phương án cơ sở.
          </p>
        }
      >
      <p>
        The explorer shows both levels at once: the table with its tiles, the tiles of each launch, and the threads busy at each
        step inside a full tile. With W = 1 a tile is a cell, and the square scheme is the baseline.
      </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một bước tốn một đơn vị và mọi ô của một lần khởi chạy chạy cùng lúc: không có số SM, không có occupancy (mức chiếm
              dụng), và bộ nhớ không tốn gì. Nó cho thấy mỗi lần khởi chạy và mỗi bước cung cấp bao nhiêu công việc song song,
              không phải thời gian nó mất.
            </p>
          }
        >
        <p>
          A step costs one unit and every tile of a launch runs at the same time: there is no SM count and no occupancy, and
          memory costs nothing. It shows how much parallel work each launch and step offers, not the time it takes.
        </p>
        </Translated>
      </Simplification>
      <FrontsExplorer />
      <Ask q={questionsS166['ch16/16.6/q-launches']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS166['ch16/16.6/q-exact-blocks']} label={tr('Work it out', 'Tự tính')} />

      <h3><Translated vi="Bên trong một ô">Inside one tile</Translated></h3>
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy một ô của một lần khởi chạy với điểm số thật. Chọn một luồng và bấm bước: bảng điều khiển
            trình bày phép tính chỉ số của nó và cho thấy mỗi đầu vào trong ba đầu vào đến từ đâu. Rồi đặt thứ tự thành "luồng
            cuối trước", bỏ đánh dấu rào chắn, và bấm <i>Đến lần đọc sai đầu tiên</i>.
          </p>
        }
      >
      <p>
        The stepper runs one tile of one launch with real scores. Pick a thread and step: the panel spells out its index
        arithmetic and shows where each of its three inputs comes from. Then set the order to "last thread first", untick the
        barrier, and press <i>Go to the first bad read</i>.
      </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một luồng tiến trọn một bước mỗi lần (ba lần đọc, rồi lần ghi của nó). "Lockstep" là một warp đơn lý tưởng hóa trong
              đó mọi lane đọc trước khi bất kỳ lane nào ghi. Không có rào chắn, một luồng chạy hết mọi bước của nó trước khi luồng
              sau bắt đầu: một trong các thứ tự GPU có thể tạo ra, được chọn để làm lộ ra rào chắn bị thiếu. Không có thời gian.
            </p>
          }
        >
        <p>
          A thread moves one whole step at a time (three reads, then its write). "Lockstep" is an idealised single warp in which
          every lane reads before any lane writes. Without the barrier, one thread runs all its steps before the next starts:
          one of the orders a GPU may produce, chosen to make the missing barrier visible. There is no timing.
        </p>
        </Translated>
      </Simplification>
      <TileStepper title={tr('Inside one tile', 'Bên trong một ô')} task={tr('Find a step where most threads skip; then break the barrier and find the first bad read.', 'Tìm một bước mà phần lớn luồng bỏ qua; rồi phá rào chắn và tìm lần đọc sai đầu tiên.')} scheme="square" /* i18n-ignore: scheme id */ />
      <Ask q={questionsS166['ch16/16.6/q-idle']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS166['ch16/16.6/q-store']} />

      <h2><Translated vi="Trượt ô">Shear the tile</Translated></h2>
      <Translated
        vi={
          <p>
            Ô vuông có hai nhược điểm. Bên trong một ô, các bước mang 1, 2, …, W, …, 2, 1 ô con, nên ở hầu hết các bước một số
            luồng rảnh và các warp chạy vơi; một ô W × W cần 2W − 1 bước.
          </p>
        }
      >
      <p>
        Square tiles have two drawbacks. Inside a tile, the steps carry 1, 2, …, W, …, 2, 1 cells, so on most steps some threads
        idle and warps run part-empty; a W × W tile takes 2W − 1 steps.
      </p>
      </Translated>
      <Source section="16.7" pages="389–390" />
      <Translated
        vi={
          <p>
            Nhược điểm thứ hai liên quan đến bộ nhớ đệm (cache), và sách lập luận mà không đo: các ô con trên cùng bên phải của
            một ô được tính sớm, nhưng ô kế tiếp lại cần chúng trước tiên, lúc đó chúng có thể đã rời khỏi cache; những ô con còn
            trong cache lại là những ô nó cần sau cùng. Không gì trên trang này mô hình hóa cache.
          </p>
        }
      >
      <p>
        The second is about caches, and the book argues it without measuring it: a tile's top-right cells are computed early, but
        the next tile along needs them first, by which time they may have left the cache; the cells still in the cache are the
        ones it needs last. Nothing on this page models a cache.
      </p>
      </Translated>
      <Source section="16.7" pages="390" figs={tr('Fig. 16.13', 'Hình 16.13')} />
      <Translated
        vi={
          <p>
            Cách khắc phục là làm nghiêng ô. Trượt hàng r_tile của mọi ô sang trái r_tile ô con: một phép trượt (shear). Hình vuông
            trở thành hình bình hành, và mỗi cột của ô giờ nằm dọc một đường chéo phụ của bảng, nên cả một cột của ô có thể được
            tính trong một bước.
          </p>
        }
      >
      <p>
        The fix is to tilt the tile. Slide row r_tile of every tile r_tile cells to the left: a shear. The square becomes a
        parallelogram, and each column of the tile now lies along an anti-diagonal of the table, so a whole tile column can be
        computed in one step.
      </p>
      </Translated>
      <Source section="16.7" pages="390–391" figs={tr('Fig. 16.14', 'Hình 16.14')} />
      <Predict
        id="ch16/16.6/p-hyper-launches"
        prompt={<Translated vi={<>Một phần trong 9 × 9 cắt thành các ô 3 × 3 cần 5 lần khởi chạy với ô vuông. Khi trượt, cần bao nhiêu lần khởi chạy?</>}><>A 9 × 9 interior cut into 3 × 3 tiles takes 5 launches with square tiles. Sheared, how many launches?</></Translated>}
        unit={tr('launches', 'lần khởi chạy')}
      >
        {(v) => (
          <Translated
            vi={
              <p data-reveal>
                {v === 8 ? 'Đúng: ' : ''}8 lần khởi chạy, 3T − 1 với T = 3, so với 2T − 1 = 5 cho ô vuông.{' '}
                {v < 8 ? 'Ít bước hơn bên trong một ô không có nghĩa là ít lần khởi chạy hơn: phần cuối của hoạt hình cho thấy vì sao.' : 'Phần cuối của hoạt hình cho thấy chúng đến từ đâu.'}
              </p>
            }
          >
          <p data-reveal>
            {v === 8 ? 'Right: ' : ''}8 launches, 3T − 1 with T = 3, against 2T − 1 = 5 for square tiles.{' '}
            {v < 8 ? 'Fewer steps inside a tile does not mean fewer launches: the last part of the animation shows why.' : 'The last part of the animation shows where they come from.'}
          </p>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Hoạt hình lấy ô (1, 1) của phần trong 9 × 9 đó, vuông rồi trượt, và kết thúc ở toàn bộ bảng.
          </p>
        }
      >
      <p>
        The animation takes tile (1, 1) of that 9 × 9 interior, square and then sheared, and ends on the whole table.
      </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một bước tốn một đơn vị, không có cache, và mọi ô của một lần khởi chạy chạy cùng lúc. Các ô con, bước và đầu vào được
              vẽ đều do cùng mô hình với trình khám phá tính ra.
            </p>
          }
        >
        <p>
          One step costs one unit, there is no cache, and every tile of a launch runs at the same time. The cells, steps and
          inputs drawn are computed by the same model as the explorer.
        </p>
        </Translated>
      </Simplification>
      <ShearScene />
      <Translated
        vi={
          <p>
            Trong mã, phép trượt chỉ là một số hạng. Với hệ số trượt −1, cột trong bảng trở thành{' '}
            <code>q = W·tile_col + q_tile − r_tile + 1</code>, và mọi luồng dùng <code>q_tile = d_tile</code>. Một ô cần W bước thay
            vì 2W − 1, và trong một ô nằm trọn trong bảng, mọi luồng đều bận ở mọi bước: cùng W² ô con với số ô con mỗi bước gấp
            khoảng hai lần (Eqs. 16.3 và 16.4).
          </p>
        }
      >
      <p>
        In the code the shear is one term. With shear factor −1 the table column becomes{' '}
        <code>q = W·tile_col + q_tile − r_tile + 1</code>, and every thread uses <code>q_tile = d_tile</code>. A tile takes W
        steps instead of 2W − 1, and in a tile that lies wholly inside the table every thread is busy at every step: the same W²
        cells at about twice the cells per step (Eqs. 16.3 and 16.4).
      </p>
      </Translated>
      <Source section="16.7" pages="391–394, 396" figs={tr('Figs. 16.15, 16.16; Eqs. 16.3–16.4', 'Hình 16.15, 16.16; Phương trình 16.3–16.4')} />
      <Translated
        vi={
          <p>
            Cái giá nằm bên ngoài ô. Vì các hàng của nó nghiêng sang trái, một ô giờ đọc từ ô phía trên bên phải,
            (tile_row − 1, tile_col + 1), cũng như từ ô bên trái, nên mỗi hàng ô phải đi sau hàng phía trên hai cột ô:{' '}
            <code>tile_col = d − 2·tile_row</code>. Các ô nghiêng để lại các ô không trọn ở cả hai rìa bảng, nên mỗi hàng ô có
            T + 1 ô và cột ô cuối được kiểm tra bằng <code>tile_col &lt;= T</code>. Số lần khởi chạy tăng lên 3T − 1, và một lần
            khởi chạy chứa tối đa ⌊T/2⌋ + 1 ô trong khi ô vuông có T. Quay lại trình khám phá, chuyển sang ô đã trượt và so sánh;
            rồi đổi phép kiểm tra cột cuối thành &lt; và nhìn vào rìa phải.
          </p>
        }
      >
      <p>
        The cost is outside the tile. Because its rows lean left, a tile now reads from the tile above and to the right,
        (tile_row − 1, tile_col + 1), as well as from the one on its left, so each tile row must stay two tile columns behind
        the row above: <code>tile_col = d − 2·tile_row</code>. Tilted tiles leave part tiles at both
        edges of the table, so each tile row has T + 1 tiles and the last tile column is tested with <code>tile_col &lt;= T</code>.
        The launches rise to 3T − 1, and a launch holds at most ⌊T/2⌋ + 1 tiles where square tiles had T. Back in the explorer,
        switch to sheared tiles and compare; then switch the last-column test to &lt; and look at the right edge.
      </p>
      </Translated>
      <Source section="16.7" pages="391–392" figs={tr('Fig. 16.14(b)', 'Hình 16.14(b)')} />
      <Translated
        vi={
          <p>
            Ô được lưu dưới dạng hình vuông trong bộ nhớ chia sẻ, nên cách đánh chỉ số vẫn đơn giản: cột lưu trữ q_tile = d_tile
            giữ những gì bước d_tile đã tính. Điều đó làm dịch chỗ các láng giềng. Hãy tự tính một láng giềng trước khi trình chạy
            từng bước cho thấy cả ba.
          </p>
        }
      >
      <p>
        The tile is stored as a square in shared memory, so its indexing stays simple: stored column q_tile = d_tile holds what
        step d_tile computed. That moves the neighbours. Work out one before the stepper shows all three.
      </p>
      </Translated>
      <Source section="16.7" pages="394" figs={tr('Fig. 16.17', 'Hình 16.17')} />
      <Ask q={questionsS166['ch16/16.6/q-nw-offset']} label={tr('Before you look', 'Trước khi xem')} />
      <TileStepper title={tr('Inside one sheared tile', 'Bên trong một ô đã trượt')} task={tr('Step the left-edge tile; find the reads that land outside the table, then untick the zeroing.', 'Chạy từng bước ô ở rìa trái; tìm các lần đọc rơi ra ngoài bảng, rồi bỏ đánh dấu việc đặt về 0.')} scheme="sheared" /* i18n-ignore: scheme id */ />
      <Translated
        vi={
          <p>
            Chọn ô ở rìa trái. Các hàng dưới của nó trượt ra khỏi bảng, nên một số luồng không có ô con ở một số bước. Thế nhưng
            một ô con ở cột 1 vẫn đọc đầu vào tây và tây bắc từ các ô nhớ của ô, và các ô nhớ đó đại diện cho cột 0, cột mà không
            luồng nào tính. Kernel trượt đặt ô về 0 trước khi bắt đầu, và chỉ kernel trượt mới cần làm vậy.
          </p>
        }
      >
      <p>
        Pick the left-edge tile. Its lower rows slide out of the table, so some threads have no cell at some steps. Yet a cell in
        column 1 still reads its west and north-west inputs from tile slots, and those slots stand for column 0, which no thread
        computes. The sheared kernel zeroes its tile before it starts, and only the sheared kernel needs to.
      </p>
      </Translated>
      <Source section="16.7" pages="395" figs={tr('Fig. 16.19', 'Hình 16.19')} />
      <Ask q={questionsS166['ch16/16.6/q-zero']} />
      <Ask q={questionsS166['ch16/16.6/q-hyper-cost']} />
      <Translated
        vi={
          <p>
            Sách kết thúc phần này bằng một khung bên lề áp dụng cùng phép trượt cho stencil: với một vòng lặp thời gian bao quanh
            một cập nhật kiểu Jacobi, có các mặt sóng theo thời gian và theo không gian, và các ô đã trượt trong mặt phẳng thời
            gian - không gian cần ít lần đồng bộ hóa toàn cục hơn. Stencil là <a href={href('ch08', 'start')}>chương 8</a>; khung
            bên lề không được mô hình hóa ở đây.
          </p>
        }
      >
      <p>
        The book ends the section with a sidebar applying the same shear to stencils: with a time loop around a Jacobi-style
        update there are fronts in time and in space, and sheared tiles in the time-space plane need fewer global
        synchronisations. Stencils are <a href={href('ch08', 'start')}>chapter 8</a>; the sidebar is not modelled here.
      </p>
      </Translated>
      <Source pages="397" figs={tr('sidebar', 'khung bên lề')} />
      <Aside>
        <Translated
          vi={
            <p>
              Sách trích Irigoin và Triolet, "Supernode partitioning" (POPL 1988), tài liệu tham khảo [2] của sách, cho phép biến đổi
              mà sách gọi là hyperplane partitioning. (ch16-facts A5.)
            </p>
          }
        >
        <p>
          The book cites Irigoin and Triolet, "Supernode partitioning" (POPL 1988), its reference [2], for the transformation it
          calls hyperplane partitioning. (ch16-facts A5.)
        </p>
        </Translated>
      </Aside>

      <h2><Translated vi="Bank">Banks</Translated></h2>
      <Translated
        vi={
          <p>
            Lưu một ô đã trượt không có đệm thì mọi truy cập của một bước đi dọc một cột lưu trữ: luồng t chạm word t·W + d_tile.
            Các luồng liên tiếp cách nhau W word, và khi W là bội hay ước của số bank, chúng dồn vào vài bank. Một word trống sau
            mỗi hàng của ô làm bước nhảy (stride) thành W + 1. Với các độ rộng lũy thừa của hai bị xung đột, W + 1 là số lẻ và
            không có ước chung với số bank, và các xung đột bank (bank conflict) biến mất.
          </p>
        }
      >
      <p>
        Store a sheared tile unpadded and every access of a step walks down one stored column: thread t touches word t·W + d_tile.
        Consecutive threads are W words apart, and when W is a multiple or divisor of the bank count they pile into a few banks.
        One empty word after each tile row makes the stride W + 1. For the power-of-two widths that conflict, W + 1 is odd and
        shares no factor with the bank count, and the conflicts are gone.
      </p>
      </Translated>
      <Source section="16.7" pages="395–396" figs={tr('Fig. 16.20', 'Hình 16.20')} />
      <Ask q={questionsS166['ch16/16.6/q-banks16']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Kernel vuông thì ngược lại. Phép duyệt đường chéo phụ của nó chạm word t·(W − 1) + d_tile: bước nhảy W − 1, không xung
            đột khi không đệm. Thêm đệm thì bước nhảy thành W. Và macro <code>pad()</code> của sách không phải là một khoảng trống
            mỗi hàng: nó thêm một khoảng trống sau mỗi số word bằng số bank, điều này để lại xung đột 2 chiều cho một ô đã trượt
            rộng 64 trên 32 bank. Các mã in trong sách không bao giờ áp dụng nó.
          </p>
        }
      >
      <p>
        The square kernel is the other way round. Its anti-diagonal walk touches word t·(W − 1) + d_tile: stride W − 1, conflict-free
        unpadded. Pad it and the stride becomes W. And the book's <code>pad()</code> macro is not one gap per row: it adds one gap
        per bank-count words, which leaves a 2-way conflict for a 64-wide sheared tile on 32 banks. The printed listings never
        apply it.
      </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch16-facts E1, B3; infra-facts A10, B10, C13)', 'ngày 2026-11-20 (ch16-facts E1, B3; infra-facts A10, B10, C13)')}>
        <Translated
          vi={
            <p>
              NVIDIA: 32 bank, mỗi bank 4 byte. AMD: 32 hoặc 64 tùy kiến trúc; cách một truy cập 64 lane được chia không được tài
              liệu ghi nhận. Apple không ghi nhận bank nào, nên lab 16.1 đo thay vào đó.
            </p>
          }
        >
        <p>
          NVIDIA: 32 banks of 4 bytes. AMD: 32 or 64 by architecture; how a 64-lane access splits is undocumented. Apple
          documents no banks, so lab 16.1 measures instead.
        </p>
        </Translated>
      </Today>
      <Ask q={questionsS166['ch16/16.6/q-pad-square']} label={tr('Before you look', 'Trước khi xem')} />
      <Translated
        vi={
          <p>
            Trình khám phá tính mọi truy cập của một ô cho cả hai kernel, ba cách bố trí và bốn độ rộng, để bạn kiểm tra cả hai
            đáp án. Số xung đột là phép tính của chúng tôi dựa trên quy tắc bank được NVIDIA ghi nhận (ch16-facts E1); sách không
            đưa ra con số nào.
          </p>
        }
      >
      <p>
        The explorer computes every access of one tile for both kernels, three layouts and four widths, so you can check both
        answers. The conflict counts are our arithmetic on NVIDIA's documented bank rule (ch16-facts E1); the book gives none.
      </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một ô đầy đủ ở phần trong. Quy tắc của NVIDIA được áp dụng cho mỗi warp gồm các luồng liên tiếp; các lane có đầu vào
              nằm ở rìa ô đi ra bộ nhớ toàn cục và được loại ra. Không có thời gian. Chế độ xem 64 bank là một mô hình.
            </p>
          }
        >
        <p>
          One full interior tile. NVIDIA's rule is applied per warp of consecutive threads; lanes whose input lies on the
          tile's edge go to global memory and are left out. No timing. The 64-bank view is a model.
        </p>
        </Translated>
      </Simplification>
      <BankExplorer />

      <h2><Translated vi="Một lệnh cho max4">One instruction for max4</Translated></h2>
      <Translated
        vi={
          <p>
            Mẹo cuối của sách cho kernel này: từ Hopper, CUDA có các lệnh DPX cho quy hoạch động, và{' '}
            <code>__vimax3_s32_relu(x, y, z)</code> tính giá trị lớn nhất của ba số nguyên và 0, nên nó có thể thay cho{' '}
            <code>max4(0, x, y, z)</code>. Các bảng nhỏ vừa một khối hoặc một warp, mẹo §16.8 còn lại của sách, đã ở trang{' '}
            <a href={href('ch16', '16.3')}>16.3</a>.
          </p>
        }
      >
      <p>
        The book's last tip for this kernel: since Hopper, CUDA has DPX instructions for dynamic programming, and{' '}
        <code>__vimax3_s32_relu(x, y, z)</code> computes the maximum of three integers and 0, so it can replace{' '}
        <code>max4(0, x, y, z)</code>. Small tables that fit one block or one warp, the book's other §16.8 tip, were on{' '}
        <a href={href('ch16', '16.3')}>16.3</a>.
      </p>
      </Translated>
      <Source section="16.8" pages="398" />
      <Today checked={tr('2026-11-20 (notes/today/ch16-facts.md A2, B2, C2)', 'ngày 2026-11-20 (notes/today/ch16-facts.md A2, B2, C2)')}>
        <Translated
          vi={
            <p>
              Bảng tính năng của NVIDIA liệt kê DPX là native trên compute capability 9.0 và 10.x và là vài lệnh trên các loại khác;
              bảng thông lượng của nó lại nói khác về 12.x. HIP hoàn toàn không khai báo hàm DPX nào, nên tệp <code>.cu</code> giữ{' '}
              <code>max4</code> và mọi lời gọi DPX nằm sau một điều kiện chỉ dành cho CUDA. Metal viết{' '}
              <code>max(max3(x, y, z), 0)</code>. Mã nguồn CCCL của chính NVIDIA báo cáo một lỗi biên dịch sai của ptxas đúng với
              loại chuỗi phụ thuộc này trước CUDA 13.4; nó không có trong release notes, nên nó là unverified.
            </p>
          }
        >
        <p>
          NVIDIA's feature table lists DPX as native on compute capability 9.0 and 10.x and as several instructions on the
          others; its throughput table disagrees about 12.x. HIP declares no DPX function at all, so the <code>.cu</code> keeps{' '}
          <code>max4</code> and any DPX call goes behind a CUDA-only guard. Metal writes <code>max(max3(x, y, z), 0)</code>.
          NVIDIA's CCCL source reports a ptxas miscompile of exactly this kind of dependent chain before CUDA 13.4; it is not in
          the release notes, so it is unverified.
        </p>
        </Translated>
      </Today>
      <Ask q={questionsS166['ch16/16.6/q-dpx']} />

      <h2><Translated vi="Điều này chạy ở đâu trong một hệ LLM">Where this runs in an LLM stack</Translated></h2>
      <Translated
        vi={
          <p>
            Backend CUDA của llama.cpp chạy prefill Mamba-2 dài hơn 128 token (đến một giới hạn) theo kiểu chunked state-space
            duality, trên GPU NVIDIA từ Turing trở đi: chuỗi được cắt thành các đoạn 256 token, mỗi đoạn là các phép nhân ma trận
            dày đặc, và một vòng lặp host qua các đoạn khởi chạy các kernel mang trạng thái hồi quy từ đoạn này sang đoạn sau. Đó
            chính là cấu trúc của trang này: công việc song song bên trong một ô, và các lần khởi chạy là rào chắn giữa các ô.
            Đường này chỉ được chọn trên NVIDIA và bị loại khi biên dịch cho HIP.
          </p>
        }
      >
      <p>
        llama.cpp's CUDA backend runs Mamba-2 prefill of more than 128 tokens (up to a cap) as chunked state-space duality, on
        NVIDIA GPUs from Turing on: the sequence is cut into 256-token chunks, each chunk is dense matrix products, and a host loop over the chunks launches the kernels that carry
        the recurrent state from one chunk to the next. It is this page's structure: parallel work inside a tile, and launches as
        the barrier between tiles. The path is selected only on NVIDIA and is compiled out for HIP.
      </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Từ <code>ggml/src/ggml-cuda/ssm-scan.cu</code> của llama.cpp như được đọc ngày 2026-11-20 (ch16-facts D1). Mối liên
              hệ với các ô là của khóa học này.
            </p>
          }
        >
        <p>
          From llama.cpp's <code>ggml/src/ggml-cuda/ssm-scan.cu</code> as read on 2026-11-20 (ch16-facts D1). The mapping to
          tiles is ours.
        </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS166['ch16/16.6/q-ssd']} />
      <Translated
        vi={
          <p>
            Lab 16.1 (<a href={href('ch16', 'labs')}>trang bài thực hành</a>) cho bạn viết cả hai kernel và đo ô vuông so với ô
            đã trượt, và ô đã trượt có và không có đệm, trên GPU của chính bạn.
          </p>
        }
      >
      <p>
        Lab 16.1 (<a href={href('ch16', 'labs')}>labs page</a>) has you write both kernels and measure square against sheared
        tiles, and the sheared tile with and without padding, on your own GPU.
      </p>
      </Translated>
    </>
  );
}
