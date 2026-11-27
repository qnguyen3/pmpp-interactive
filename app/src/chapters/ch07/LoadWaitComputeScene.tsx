import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import {
  CONV_CORRECT,
  constMemIntensity,
  makeProblem,
  runTiledBlock,
  stateAt,
  threadInfo,
  tiledIntensity,
  type ConvConfig,
  type ConvStep,
} from '../../sim/convtile';

// "Load, wait, compute": block (0,0) of the Fig. 7.12 kernel at Fig. 7.13's sizes
// (8x8 block and input tile, r = 1, 6x6 output tile) on a 10x10 image. Every frame
// is the model's state (sim/convtile.ts, in-order schedule) at some step of the
// trace; the tally comes from the same file's formula. Six beats, one narration
// clip each (content/ch07/narration/load-wait-compute-N.txt).
const cfg: ConvConfig = { ...CONV_CORRECT, inTile: 8, r: 1 };
const W = 10;
const IN = 8;
const OUT = 6;
const prob = makeProblem(W, W, 1);
const trace = runTiledBlock(cfg, prob, 0, 0, 'in-order');
const steps = trace.steps;
const loadIdx = steps.flatMap((s, i) => (s.op === 'load' ? [i] : []));
const releaseIdx = steps.findIndex((s) => s.op === 'release');
const info = Array.from({ length: IN * IN }, (_, t) => threadInfo(cfg, prob, 0, 0, t % IN, Math.floor(t / IN)));
const ghostCount = steps.filter((s) => s.op === 'load' && s.write?.ghost).length;
const idleCount = info.filter((t) => !t.active).length;
const taps = steps.filter((s) => s.op === 'tap').length;
const stores = trace.stores.length;
const aiToy = tiledIntensity(OUT, 3);
const aiBook = tiledIntensity(28, 5);
const aiConst = constMemIntensity();

const C = 22; // cell size
const NX = 12, NY = 46; // N panel origin, including a ghost margin of one cell
const SX = 300, SY = 68; // shared tile
const PX = 500, PY = 68; // output P
const nCell = (y: number, x: number) => ({ x: NX + (x + 1) * C, y: NY + (y + 1) * C });
const sCell = (r: number, c: number) => ({ x: SX + c * C, y: SY + r * C });
const pCell = (y: number, x: number) => ({ x: PX + x * C, y: PY + y * C });

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const f2 = (v: number) => v.toFixed(2);

interface Frame {
  /** Steps of the trace that have run. */
  n: number;
  /** Opacity of the block's input-tile outline over N. */
  tile?: number;
  /** A thread whose element is linked to its slot by a line. */
  pin?: number;
  /** Grey out the threads that do not compute. */
  ring?: boolean;
  barrier?: 'down' | 'open' | null;
  /** Opacity of the next block's input-tile outline. */
  next?: number;
  headline?: ReactNode;
  sub?: ReactNode;
  tally?: number;
}

function Stage(f: Frame) {
  const st = stateAt(cfg, trace, f.n);
  const cur: ConvStep | undefined = f.n > 0 ? steps[f.n - 1] : undefined;
  const curT = cur && cur.thread >= 0 ? info[cur.thread] : undefined;
  const stored = new Map(st.stores.map((s) => [s.index, s.value]));
  const hotN = cur?.op === 'load' && cur.global?.inside ? cur.global.index : -1;
  const pinT = f.pin != null ? info[f.pin] : undefined;
  return (
    <svg viewBox="0 0 732 404" role="img" aria-label={tr('Animation of one block of the tiled convolution kernel', 'Hoạt ảnh một khối của kernel tích chập phân ô')} style={{ width: '100%', maxWidth: 732, display: 'block' }}>
      {/* N with a ring of ghost cells around it */}
      <text x={NX} y={NY - 12} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}><Translated vi="N (bộ nhớ toàn cục), 10 × 10">N (global memory), 10 × 10</Translated></text>
      {Array.from({ length: (W + 2) * (W + 2) }, (_, i) => {
        const y = Math.floor(i / (W + 2)) - 1;
        const x = (i % (W + 2)) - 1;
        const inside = y >= 0 && y < W && x >= 0 && x < W;
        const { x: px, y: py } = nCell(y, x);
        const hot = inside && y * W + x === hotN;
        return (
          <g key={i}>
            <rect x={px} y={py} width={C - 2} height={C - 2} rx="2"
              fill={!inside ? 'none' : hot ? 'var(--global)' : 'var(--cell)'}
              stroke={inside ? 'var(--rule)' : 'var(--ink-3)'} strokeDasharray={inside ? undefined : '2 2'} opacity={inside ? 1 : 0.6} />
            {inside ? (
              <text x={px + C / 2 - 1} y={py + C / 2 + 3} fontSize="10" textAnchor="middle" style={{ fill: hot ? 'var(--page)' : 'var(--ink)' }}>
                {prob.N[y * W + x]}
              </text>
            ) : null}
          </g>
        );
      })}
      {f.tile ? (
        <rect x={nCell(-1, -1).x - 2} y={nCell(-1, -1).y - 2} width={IN * C + 2} height={IN * C + 2} fill="none"
          stroke="var(--shared)" strokeWidth="2.4" opacity={f.tile} />
      ) : null}
      {f.next ? (
        <g opacity={f.next}>
          {/* block (1, 0)'s input tile runs from column 5 to 12; the panel ends at column 10 */}
          <rect x={nCell(-1, OUT - 1).x - 1} y={nCell(-1, -1).y - 1} width={2 * C} height={IN * C} fill="var(--shared-wash)" />
          <path d={`M ${nCell(-1, W).x + C} ${nCell(-1, -1).y + 2} H ${nCell(-1, OUT - 1).x - 2} V ${nCell(-1, -1).y + IN * C + 2} H ${nCell(-1, W).x + C}`}
            fill="none" stroke="var(--ink)" strokeWidth="1.6" strokeDasharray="5 3" />
        </g>
      ) : null}

      {/* the shared tile, one slot per thread */}
      <text x={SX} y={SY - 34} fontSize="12" fontWeight="600" style={{ fill: 'var(--shared)' }}><Translated vi="N_s (bộ nhớ chia sẻ)">N_s (shared memory)</Translated></text>
      <text x={SX} y={SY - 20} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="mỗi luồng một ô nhớ (8 × 8)">one slot per thread (8 × 8)</Translated></text>
      {st.slots.map((s, t) => {
        const ti = info[t];
        const { x, y } = sCell(ti.ty, ti.tx);
        const grey = f.ring && !ti.active;
        const isCur = curT?.t === t;
        const inPatch =
          (cur?.op === 'tap' || cur?.op === 'store') && curT && ti.ty >= curT.tileRow && ti.ty < curT.tileRow + 3 && ti.tx >= curT.tileCol && ti.tx < curT.tileCol + 3;
        const tapHere = cur?.op === 'tap' && cur.tap!.slotR === ti.ty && cur.tap!.slotC === ti.tx;
        return (
          <g key={t} opacity={grey ? 0.32 : 1}>
            <rect x={x} y={y} width={C - 2} height={C - 2} rx="2"
              fill={tapHere || (isCur && cur?.op === 'load') ? 'var(--shared)' : s.written ? 'var(--shared-wash)' : 'var(--cell)'}
              stroke={inPatch ? 'var(--ink)' : 'var(--shared)'} strokeWidth={inPatch ? 1.8 : 1}
              strokeDasharray={s.ghost ? '3 2' : undefined} />
            {s.written ? (
              <text x={x + C / 2 - 1} y={y + C / 2 + 3} fontSize="10" textAnchor="middle" fontStyle={s.ghost ? 'italic' : undefined}
                style={{ fill: tapHere || (isCur && cur?.op === 'load') ? 'var(--page)' : s.ghost ? 'var(--ink-3)' : 'var(--ink)' }}>
                {s.value}
              </text>
            ) : null}
            {st.waiting[t] ? <circle cx={x + C - 6} cy={y + 4} r="2.4" fill="var(--ink-2)" /> : null}
          </g>
        );
      })}
      {f.ring ? (
        <rect x={sCell(1, 1).x - 2} y={sCell(1, 1).y - 2} width={OUT * C + 2} height={OUT * C + 2} fill="none" stroke="var(--ink)" strokeWidth="2" />
      ) : null}
      {(cur?.op === 'tap' || cur?.op === 'store') && curT ? (
        <rect x={sCell(curT.tileRow, curT.tileCol).x - 3} y={sCell(curT.tileRow, curT.tileCol).y - 3} width={3 * C + 4} height={3 * C + 4}
          fill="none" stroke="var(--reg)" strokeWidth="2.4" rx="3" />
      ) : null}
      {pinT ? (
        <g>
          <line x1={sCell(pinT.ty, pinT.tx).x + C / 2} y1={sCell(pinT.ty, pinT.tx).y + C / 2}
            x2={nCell(pinT.row, pinT.col).x + C / 2} y2={nCell(pinT.row, pinT.col).y + C / 2} stroke="var(--ink)" strokeWidth="1.4" />
          <circle cx={nCell(pinT.row, pinT.col).x + C / 2 - 1} cy={nCell(pinT.row, pinT.col).y + C / 2 - 1} r="7" fill="none" stroke="var(--ink)" strokeWidth="1.6" />
          <rect x={sCell(pinT.ty, pinT.tx).x - 1} y={sCell(pinT.ty, pinT.tx).y - 1} width={C} height={C} fill="none" stroke="var(--ink)" strokeWidth="2" />
          <text x={SX} y={SY + IN * C + 18} fontSize="11" style={{ fill: 'var(--ink)' }}>
            <Translated vi={<>luồng ({pinT.tx}, {pinT.ty}) → {pinT.inImage ? `N[${pinT.row}][${pinT.col}]` : `ô ma (${pinT.row}, ${pinT.col})`}</>}>
              thread ({pinT.tx}, {pinT.ty}) → {pinT.inImage ? `N[${pinT.row}][${pinT.col}]` : `ghost (${pinT.row}, ${pinT.col})`}
            </Translated>
          </text>
        </g>
      ) : null}
      {f.barrier ? (
        <g>
          <rect x={SX} y={SY + IN * C + 6} width={IN * C} height={24} rx="3" fill={f.barrier === 'down' ? 'var(--ink)' : 'none'} stroke="var(--ink)" strokeDasharray={f.barrier === 'open' ? '4 3' : undefined} />
          <text x={SX + (IN * C) / 2} y={SY + IN * C + 22} fontSize="11" textAnchor="middle" style={{ fill: f.barrier === 'down' ? 'var(--page)' : 'var(--ink)' }}>
            {f.barrier === 'down' ? tr('__syncthreads(): waiting', '__syncthreads(): đang chờ') : tr('barrier open', 'rào chắn mở')}
          </text>
        </g>
      ) : null}

      {/* the output */}
      <text x={PX} y={PY - 34} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}><Translated vi="P (bộ nhớ toàn cục)">P (global memory)</Translated></text>
      <text x={PX} y={PY - 20} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="ô đầu ra của khối này: 6 × 6">this block's output tile: 6 × 6</Translated></text>
      {prob.ref.map((ref, i) => {
        const y = Math.floor(i / W);
        const x = i % W;
        const { x: px, y: py } = pCell(y, x);
        const mine = y < OUT && x < OUT;
        const v = stored.get(i);
        const isCur = cur?.op === 'store' && cur.global?.index === i;
        const target = (cur?.op === 'tap' || cur?.op === 'store') && curT && curT.row === y && curT.col === x;
        return (
          <g key={i}>
            <rect x={px} y={py} width={C - 2} height={C - 2} rx="2"
              fill={isCur ? 'var(--global)' : v !== undefined ? (v === ref ? 'var(--ok-wash)' : 'var(--fail-wash)') : mine ? 'var(--global-wash)' : 'var(--cell)'}
              stroke={target ? 'var(--ink)' : mine ? 'var(--global)' : 'var(--rule)'} strokeWidth={target ? 1.8 : 1} />
            {v !== undefined ? (
              <text x={px + C / 2 - 1} y={py + C / 2 + 3} fontSize="8.5" textAnchor="middle" style={{ fill: isCur ? 'var(--page)' : 'var(--ink)' }}>
                {v}
              </text>
            ) : null}
          </g>
        );
      })}
      {(cur?.op === 'tap' || cur?.op === 'store') && curT ? (
        <text x={PX} y={PY + W * C + 16} fontSize="11" style={{ fill: 'var(--ink)' }}>
          <Translated vi={<>luồng ({curT.tx}, {curT.ty}): vùng tại N_s[{curT.tileRow}][{curT.tileCol}] → P[{curT.row}][{curT.col}]</>}>
            thread ({curT.tx}, {curT.ty}): patch at N_s[{curT.tileRow}][{curT.tileCol}] → P[{curT.row}][{curT.col}]
          </Translated>
        </text>
      ) : null}

      {f.tally != null ? <Tally p={f.tally} /> : null}
      {f.headline ? (
        <text x={NX} y={f.tally != null ? 330 : 340} fontSize="13" fontWeight="600" style={{ fill: 'var(--ink)' }}>{f.headline}</text>
      ) : null}
      {f.sub && f.tally == null ? (
        <text x={NX} y={360} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>{f.sub}</text>
      ) : null}
    </svg>
  );
}

/** Bytes of global traffic for one interior block: tiled against one load per tap. */
function Tally({ p }: { p: number }) {
  const tiled = (IN * IN + OUT * OUT) * 4; // 64 loads + 36 stores
  const perTap = OUT * OUT * 9 * 4; // 324 loads
  const scale = 380 / perTap;
  const g = ramp(p, 0.08, 0.4);
  return (
    <g>
      <rect x={NX - 6} y={312} width={724} height={88} fill="var(--page)" />
      <text x={NX} y={348} fontSize="11.5"><Translated vi="phân ô: 64 lần nạp + 36 lần ghi">tiled: 64 loads + 36 stores</Translated></text>
      <rect x={210} y={337} width={tiled * scale * g} height={14} fill="var(--shared)" />
      <text x={210 + tiled * scale * g + 6} y={348} fontSize="11.5">{Math.round(tiled * g)} B</text>
      <text x={NX} y={372} fontSize="11.5"><Translated vi="bộ nhớ hằng: 324 lần nạp">constant memory: 324 loads</Translated></text>
      <rect x={210} y={361} width={perTap * scale * g} height={14} fill="var(--global)" />
      <text x={210 + perTap * scale * g + 6} y={372} fontSize="11.5">{Math.round(perTap * g)} B</text>
      <text x={NX} y={396} fontSize="11.5" opacity={ramp(p, 0.45, 0.55)}>
        <Translated vi={<>648 FLOP: {f2(aiToy)} FLOP/B khi phân ô so với {f2(aiConst)}</>}>
          648 FLOP: {f2(aiToy)} FLOP/B tiled against {f2(aiConst)}
        </Translated>
        <tspan opacity={ramp(p, 0.68, 0.78)} dx="14" fontWeight="600"><Translated vi={<>một khối 32 × 32 với bộ lọc 5 × 5: {f2(aiBook)} FLOP/B</>}>a 32 × 32 block with a 5 × 5 filter: {f2(aiBook)} FLOP/B</Translated></tspan>
      </text>
    </g>
  );
}

/** Steps run up to and including the k-th load; after the last one, every thread is at the barrier. */
const afterLoads = (k: number) => (k <= 0 ? 0 : k >= loadIdx.length ? releaseIdx : loadIdx[k - 1] + 1);
const PINS = [9, 0, 7, 63, 54];

const beats: Beat[] = [
  {
    clip: 'load-wait-compute-1',
    get label() {
      return tr('The block', 'Khối');
    },
    draw: (p) => (
      <Stage n={0} tile={ramp(p, 0.05, 0.3)} pin={p > 0.3 ? PINS[Math.min(PINS.length - 1, Math.floor(ramp(p, 0.3, 0.98) * PINS.length))] : undefined}
        headline={tr('block (0, 0): 64 threads, one per input-tile element', 'khối (0, 0): 64 luồng, mỗi luồng một phần tử của ô đầu vào')}
        sub={tr('row = 0·6 + ty − 1, col = 0·6 + tx − 1: the tile starts one row and one column outside the image', 'row = 0·6 + ty − 1, col = 0·6 + tx − 1: ô bắt đầu ở một hàng và một cột bên ngoài ảnh')} />
    ),
  },
  {
    clip: 'load-wait-compute-2',
    get label() {
      return tr('Load', 'Nạp');
    },
    draw: (p) => {
      const k = Math.round(ramp(p, 0.04, 0.9) * loadIdx.length);
      const n = afterLoads(k);
      const ghosts = steps.slice(0, n).filter((s) => s.op === 'load' && s.write?.ghost).length;
      return <Stage n={n} tile={1} headline={tr(`load phase: ${k} of 64 slots filled`, `pha nạp: đã điền ${k} trên 64 ô nhớ`)} sub={tr(`${ghosts} of them ghost cells, stored as 0 (dashed)`, `${ghosts} trong số đó là ô ma (ghost cell), lưu là 0 (nét đứt)`)} />;
    },
  },
  {
    clip: 'load-wait-compute-3',
    get label() {
      return tr('Wait', 'Chờ');
    },
    draw: (p) => (
      <Stage n={p < 0.72 ? releaseIdx : releaseIdx + 1} tile={1} barrier={p < 0.12 ? null : p < 0.72 ? 'down' : 'open'}
        headline={p < 0.72 ? tr('every thread waits at the one barrier', 'mọi luồng chờ ở rào chắn duy nhất') : tr('all 64 slots are filled: the barrier opens', 'cả 64 ô nhớ đã được điền: rào chắn mở')}
        sub={tr('the tile is loaded once and never overwritten, so no second barrier', 'ô được nạp một lần và không bao giờ bị ghi đè, nên không cần rào chắn thứ hai')} />
    ),
  },
  {
    clip: 'load-wait-compute-4',
    get label() {
      return tr('Compute', 'Tính');
    },
    draw: (p) => {
      const span = steps.length - (releaseIdx + 1);
      const n = releaseIdx + 1 + Math.round(ramp(p, 0.04, 0.95) * span);
      const done = stateAt(cfg, trace, n).stores.length;
      return (
        <Stage n={n} tile={1} ring headline={tr(`compute phase: the ring of ${idleCount} threads is idle`, `pha tính: vòng ngoài gồm ${idleCount} luồng ngồi không`)}
          sub={tr(`${done} of 36 outputs stored; ${taps} multiply-adds in all, every operand from N_s`, `đã ghi ${done} trên 36 đầu ra; tổng cộng ${taps} phép nhân-cộng, mọi toán hạng đều lấy từ N_s`)} />
      );
    },
  },
  {
    clip: 'load-wait-compute-5',
    get label() {
      return tr('Store', 'Ghi');
    },
    draw: (p) => (
      <Stage n={steps.length} tile={1} ring next={ramp(p, 0.3, 0.55)}
        headline={tr(`${stores} outputs: one 6 × 6 tile of P`, `${stores} đầu ra: một ô 6 × 6 của P`)}
        sub={p < 0.3 ? tr('each active thread stored one result', 'mỗi luồng hoạt động đã ghi một kết quả') : tr('the next block re-loads 2 columns of this block’s inputs: the halo', 'khối kế tiếp nạp lại 2 cột đầu vào của khối này: vùng biên (halo)')} />
    ),
  },
  {
    clip: 'load-wait-compute-6',
    get label() {
      return tr('Count', 'Đếm');
    },
    draw: (p) => <Stage n={steps.length} tile={1} ring tally={p} headline={tr('global traffic for one block away from the edges', 'lưu lượng bộ nhớ toàn cục của một khối nằm xa các cạnh')} />,
  },
];

export function LoadWaitComputeScene() {
  return <Scene chapter="ch07" title={tr('Load, wait, compute', 'Nạp, chờ, tính')} beats={beats} />;
}

// For the browser check and the lesson text: the counts the scene shows.
export const SCENE_COUNTS = { ghostCount, idleCount, taps, stores, aiToy, aiBook, aiConst };
