import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import {
  additions,
  brentKung,
  finalAfter,
  koggeStone,
  needs,
  ranges,
  upSteps,
  warpView,
  type Network,
  type WarpView,
} from '../../sim/prefixnet';

// "Up-sweep and down-sweep" (section 10 of chapter 11): Brent-Kung on sixteen positions, step
// by step, then one warp of 32 lanes under both networks. Every adder, range, count and lane
// drawn comes from src/sim/prefixnet.ts. Six beats, one narration clip each
// (content/ch11/narration/bk-sweeps-N.txt).

const N = 16;
const BK: Network = brentKung(N);
const KS: Network = koggeStone(N);
const RANGES = ranges(BK);
const UP = upSteps(BK);
const W32: { ks: WarpView; bk: WarpView } = { ks: warpView('kogge-stone', 32), bk: warpView('brent-kung', 32) };

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

// ---------------------------------------------------------------- the network drawing
const X0 = 70;
const DX = 39;
const x = (i: number) => X0 + i * DX;
const ROW0 = 64;
const DY = 27;
const y = (k: number) => ROW0 + k * DY; // row of step k (0-based)
const CELL_Y = ROW0 + BK.steps.length * DY + 6;
const NEED_Y = CELL_Y + 40;
const SHADE = [0, 0.2, 0.45, 0.75];

interface NetFrame {
  /** Steps drawn (and applied to the cells). */
  shown: number;
  /** Opacity of the newest step drawn, for its fade-in. */
  fresh?: number;
  /** Show the partial sums each position still needs. */
  needs?: boolean;
  /** Mark the positions that are final. */
  finals?: boolean;
  headline: string;
  extra?: ReactNode;
}

function NetStage({ shown, fresh = 1, needs: showNeeds, finals, headline, extra }: NetFrame) {
  const state = RANGES[shown];
  const fin = new Set(finalAfter(BK, shown));
  const changed = new Set(shown > 0 ? BK.steps[shown - 1].adders.map((a) => a.dst) : []);
  const need = needs(state);
  const adds = BK.steps.slice(0, shown).reduce((s, st) => s + st.adders.length, 0);
  return (
    <svg viewBox="0 0 720 400" role="img" aria-label={tr('Brent-Kung on sixteen positions', 'Brent-Kung trên mười sáu vị trí')} style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      <text x={8} y={18} fontSize="13" fontWeight="600" data-part="headline" style={{ fill: 'var(--ink)' }}>{headline}</text>
      {Array.from({ length: N }, (_, i) => (
        <g key={i}>
          <text x={x(i)} y={44} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{i}</text>
          <line x1={x(i)} y1={50} x2={x(i)} y2={CELL_Y - 4} stroke="var(--rule)" />
        </g>
      ))}
      <text x={8} y={44} fontSize="10" style={{ fill: 'var(--ink-3)' }}><Translated vi="vị trí">position</Translated></text>
      {BK.steps.map((st, k) => {
        if (k >= shown) return null;
        const op = k === shown - 1 ? fresh : 0.5;
        return (
          <g key={k} opacity={op} data-part="step" data-step={k + 1}>
            <text x={8} y={y(k) + 4} fontSize="10" style={{ fill: 'var(--ink-2)' }}>
              {st.phase === 'up' ? tr(`up ${k + 1}`, `lên ${k + 1}`) : tr(`down ${st.level}`, `xuống ${st.level}`)}
            </text>
            {st.adders.map((a) => (
              <g key={a.dst} data-part="adder" data-src={a.src} data-dst={a.dst}>
                <line x1={x(a.src)} y1={y(k) - 11} x2={x(a.dst) - 5} y2={y(k) - 3} stroke="var(--ink)" strokeWidth={k === shown - 1 ? 1.6 : 1} />
                <circle cx={x(a.src)} cy={y(k) - 11} r={2} fill="var(--ink)" />
                <circle cx={x(a.dst)} cy={y(k)} r={7} fill="var(--cell)" stroke="var(--ink)" strokeWidth={k === shown - 1 ? 1.8 : 1} />
                <text x={x(a.dst)} y={y(k) + 3.5} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink)' }}>+</text>
              </g>
            ))}
          </g>
        );
      })}
      {UP < shown ? <line x1={X0 - 16} x2={x(N - 1) + 16} y1={y(UP) - 15} y2={y(UP) - 15} stroke="var(--ink-3)" strokeDasharray="4 4" /> : null}
      {state.map((r, i) => {
        const isFinal = finals && fin.has(i);
        return (
          <g key={i} data-part="cell" data-pos={i} data-range={`${r.lo}..${r.hi}`} data-final={isFinal ? 1 : 0}>
            <rect x={x(i) - 18} y={CELL_Y} width={36} height={30} rx={3}
              fill={isFinal ? 'var(--ok-wash)' : 'var(--cell)'}
              stroke={isFinal ? 'var(--ok)' : changed.has(i) ? 'var(--ink)' : 'var(--rule)'}
              strokeWidth={changed.has(i) ? 1.8 : 1} />
            <text x={x(i)} y={CELL_Y + 13} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink)' }}>{r.lo === r.hi ? `x${r.lo}` : `x${r.lo}..`}</text>
            {r.lo !== r.hi ? <text x={x(i)} y={CELL_Y + 25} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink)' }}>{`x${r.hi}`}</text> : null}
          </g>
        );
      })}
      <text x={8} y={CELL_Y + 19} fontSize="10" style={{ fill: 'var(--ink-3)' }}><Translated vi="chứa">holds</Translated></text>
      {showNeeds ? (
        <g>
          <text x={8} y={NEED_Y + 13} fontSize="10" style={{ fill: 'var(--ink-3)' }}><Translated vi="cần">needs</Translated></text>
          {need.map((v, i) => (
            <g key={i} data-part="need" data-pos={i} data-needs={v}>
              <rect x={x(i) - 13} y={NEED_Y} width={26} height={20} rx={2} fill="var(--ink)" fillOpacity={SHADE[Math.min(3, v)]} stroke="var(--rule)" />
              <text x={x(i)} y={NEED_Y + 14} fontSize="10.5" textAnchor="middle" style={{ fill: v >= 2 ? 'var(--page)' : 'var(--ink)' }}>{v}</text>
            </g>
          ))}
        </g>
      ) : null}
      <text x={8} y={392} fontSize="12" data-part="counter" data-adds={adds} data-steps={shown} style={{ fill: 'var(--ink)' }}>
        <Translated vi={<>Brent-Kung đến giờ: {adds} phép cộng trong {shown} bước</>}>
          Brent-Kung so far: {adds} additions in {shown} steps
        </Translated>
      </text>
      {extra}
    </svg>
  );
}

/** Bars comparing the two networks on sixteen positions. */
function Compare({ p }: { p: number }) {
  const rows = [
    { name: 'Brent-Kung', adds: additions(BK), steps: BK.steps.length }, // i18n-ignore: algorithm name
    { name: 'Kogge-Stone', adds: additions(KS), steps: KS.steps.length }, // i18n-ignore: algorithm name
  ];
  const ax = 420;
  return (
    <g opacity={p} data-part="compare">
      <rect x={ax - 10} y={340} width={300} height={58} fill="var(--page)" />
      {rows.map((r, k) => (
        <g key={r.name} data-part="compare-row" data-name={r.name} data-adds={r.adds} data-steps={r.steps}>
          <text x={ax} y={356 + k * 22} fontSize="11" style={{ fill: 'var(--ink)' }}>{r.name}</text>
          <rect x={ax + 78} y={347 + k * 22} width={r.adds * 1.8} height={11} fill="var(--ink-2)" />
          <text x={ax + 82 + r.adds * 1.8} y={356 + k * 22} fontSize="11" style={{ fill: 'var(--ink)' }}><Translated vi={<>{r.adds} phép cộng, {r.steps} bước</>}>{r.adds} additions, {r.steps} steps</Translated></text>
        </g>
      ))}
    </g>
  );
}

// ---------------------------------------------------------------- one warp
function WarpStage({ upto }: { upto: number }) {
  const L = 13;
  const G = 2;
  const LX = 150;
  const grid = (v: WarpView, top: number, label: string) => {
    const done = upto >= v.finishesAt;
    return (
      <g data-part="warp" data-net={v.id} data-shown={Math.min(upto, v.finishesAt)} data-finishes={v.finishesAt}>
        <text x={8} y={top - 8} fontSize="12" fontWeight="600" style={{ fill: 'var(--ink)' }}>{label}</text>
        <text x={LX} y={top - 8} fontSize="11" style={{ fill: done ? 'var(--ok)' : 'var(--ink-2)' }} data-part="warp-status">
          {done ? tr(`done after ${v.finishesAt} steps`, `xong sau ${v.finishesAt} bước`) : tr(`step ${Math.min(upto, v.finishesAt)} of ${v.finishesAt}`, `bước ${Math.min(upto, v.finishesAt)} trên ${v.finishesAt}`)}
        </text>
        {v.steps.map((s, k) => {
          if (k >= upto) return null;
          const on = new Set(s.lanes);
          return (
            <g key={k} data-part="warp-step" data-step={k + 1} data-adds={s.adds} data-idle={s.idle}>
              <text x={LX - 8} y={top + k * (L + G) + 10} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
                <Translated vi={<>{s.adds} cộng, {s.idle} rảnh</>}>
                  {s.adds} add, {s.idle} idle
                </Translated>
              </text>
              {Array.from({ length: v.width }, (_, l) => (
                <rect key={l} x={LX + l * (L + G)} y={top + k * (L + G)} width={L} height={L} rx={1.5}
                  fill={on.has(l) ? 'var(--ink-2)' : 'none'} stroke="var(--rule)" />
              ))}
            </g>
          );
        })}
      </g>
    );
  };
  return (
    <svg viewBox="0 0 720 400" role="img" aria-label={tr('One warp of 32 lanes under both networks', 'Một warp 32 lane dưới cả hai mạng')} style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      <text x={8} y={18} fontSize="13" fontWeight="600" data-part="headline" style={{ fill: 'var(--ink)' }}>
        <Translated vi="Một warp 32 lane: ô được tô đậm thực hiện phép cộng, ô trống thì rảnh nhưng vẫn chiếm chỗ trong warp">
          One warp of 32 lanes: a filled square adds, an empty one idles but still occupies the warp
        </Translated>
      </text>
      {grid(W32.ks, 56, 'Kogge-Stone')}
      {grid(W32.bk, 56 + W32.ks.finishesAt * 15 + 36, 'Brent-Kung')}
      <text x={8} y={392} fontSize="12" data-part="counter" style={{ fill: 'var(--ink)' }}>
        <Translated vi={<>Số lượt lane: Kogge-Stone {W32.ks.slots} ({W32.ks.additions} phép cộng), Brent-Kung {W32.bk.slots} ({W32.bk.additions} phép cộng)</>}>
          Lane slots: Kogge-Stone {W32.ks.slots} ({W32.ks.additions} additions), Brent-Kung {W32.bk.slots} ({W32.bk.additions} additions)
        </Translated>
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------- beats
const beats: Beat[] = [
  {
    clip: 'bk-sweeps-1',
    get label() {
      return tr('Up-sweep', 'Quét lên');
    },
    draw: (p) =>
      p < 0.3 ? (
        <NetStage shown={0} headline={tr('16 positions, one input each', '16 vị trí, mỗi vị trí một đầu vào')} />
      ) : (
        <NetStage shown={1} fresh={ramp(p, 0.3, 0.55)} headline={tr(`Up-sweep, step 1: ${BK.steps[0].adders.length} adders, each odd position adds its left neighbour`, `Quét lên, bước 1: ${BK.steps[0].adders.length} bộ cộng, mỗi vị trí lẻ cộng thêm phần tử láng giềng bên trái`)} />
      ),
  },
  {
    clip: 'bk-sweeps-2',
    get label() {
      return tr('The tree', 'Cây');
    },
    draw: (p) => {
      const shown = p < 0.25 ? 2 : p < 0.5 ? 3 : 4;
      const start = shown === 2 ? 0 : shown === 3 ? 0.25 : 0.5;
      return (
        <NetStage shown={shown} fresh={ramp(p, start, start + 0.15)} finals={p >= 0.7}
          headline={p >= 0.7 ? tr(`Reduction tree done: positions ${finalAfter(BK, UP).join(', ')} are final`, `Cây rút gọn xong: các vị trí ${finalAfter(BK, UP).join(', ')} đã có kết quả cuối`) : tr(`Up-sweep, step ${shown}`, `Quét lên, bước ${shown}`)} />
      );
    },
  },
  {
    clip: 'bk-sweeps-3',
    get label() {
      return tr('Level 1', 'Mức 1');
    },
    draw: (p) =>
      p < 0.45 ? (
        <NetStage shown={UP} needs finals headline={tr('Down-sweep: how many partial sums each position still needs', 'Quét xuống: mỗi vị trí còn cần bao nhiêu tổng riêng phần')} />
      ) : (
        <NetStage shown={UP + 1} fresh={ramp(p, 0.45, 0.65)} needs finals headline={tr('Level 1: position 7 into position 11, 4 places apart', 'Mức 1: vị trí 7 vào vị trí 11, cách nhau 4 chỗ')} />
      ),
  },
  {
    clip: 'bk-sweeps-4',
    get label() {
      return tr('Level 2', 'Mức 2');
    },
    draw: (p) => (
      <NetStage shown={p < 0.3 ? UP + 1 : UP + 2} fresh={ramp(p, 0.3, 0.5)} needs finals
        headline={p < 0.3 ? tr('Level 2: the sums 2 places apart are final now', 'Mức 2: các tổng cách nhau 2 chỗ giờ đã là kết quả cuối') : tr('Level 2: 3 into 5, 7 into 9, 11 into 13', 'Mức 2: 3 vào 5, 7 vào 9, 11 vào 13')} />
    ),
  },
  {
    clip: 'bk-sweeps-5',
    get label() {
      return tr('Level 3', 'Mức 3');
    },
    draw: (p) => (
      <NetStage shown={p < 0.2 ? UP + 2 : BK.steps.length} fresh={ramp(p, 0.2, 0.4)} needs finals
        headline={p < 0.55 ? tr('Level 3: every even position adds its left neighbour', 'Mức 3: mỗi vị trí chẵn cộng thêm phần tử láng giềng bên trái') : tr('All 16 final. Fewer additions, more steps', 'Cả 16 đã là kết quả cuối. Ít phép cộng hơn, nhiều bước hơn')}
        extra={p >= 0.55 ? <Compare p={ramp(p, 0.55, 0.7)} /> : null} />
    ),
  },
  {
    clip: 'bk-sweeps-6',
    get label() {
      return tr('One warp', 'Một warp');
    },
    draw: (p) => <WarpStage upto={Math.min(W32.bk.finishesAt, Math.floor(ramp(p, 0.05, 0.85) * W32.bk.finishesAt) + 1)} />,
  },
];

export function BkSweepsScene() {
  return <Scene chapter="ch11" title={tr('Up-sweep and down-sweep', 'Quét lên và quét xuống')} beats={beats} />;
}
