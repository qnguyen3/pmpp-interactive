import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { SCENE, f32text, gatherTimeline, sceneTally, scatterTimeline, scatterTotal, sumInOrder } from '../../sim/dcs';
import { Translated, tr } from '../../lib/i18n';

// "Scatter and gather, step by step": a row of six points and four atoms (SCENE in
// src/sim/dcs.ts). Groups of four and six threads stand in for a warp. Every number drawn
// comes from the model: the lockstep timelines, the float32 sums in two orders, the tally.
// Six beats, one narration clip each (content/ch21/narration/scatter-gather-N.txt).
const { W, atoms, runA, runB } = SCENE;
const SC = scatterTimeline(atoms, W);
const GA = gatherTimeline(atoms, W);
const TALLY = sceneTally(atoms, W);
/** The point the scene uses to show order dependence: the first whose two run totals differ (point 3, pinned by dcs.test.ts). */
const DIFF_POINT = 3;
const TOT_A = scatterTotal(atoms, DIFF_POINT, runA);
const TOT_B = scatterTotal(atoms, DIFF_POINT, runB);

const CW = 74; // cell width
const X0 = 214; // first cell
const ROW_Y = 214; // top of the row of cells
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const ax = (k: number) => X0 + (atoms[k].x + 0.5) * CW;
const ay = (k: number) => 96 + atoms[k].y * 34;
const cellX = (i: number) => X0 + i * CW;
const val = (v: number) => (v >= 0 ? ' ' : '') + v.toFixed(4);

/** The common index of the digits where two numbers' strings first differ. */
function splitAt(a: string, b: string) {
  let k = 0;
  while (k < a.length && a[k] === b[k]) k++;
  return k;
}

interface Frame {
  mode: 'setup' | 'scatter' | 'gather' | 'tally';
  headline: ReactNode;
  /** Cell texts for the row. */
  cells?: (string | null)[];
  /** Highlighted cell. */
  hot?: number | null;
  /** Lines from atoms to points: [atom, point, opacity]. */
  lines?: [number, number, number][];
  /** Queue at the hot cell: the values still waiting. */
  queue?: number[];
  /** Gather: registers per point, and the atom being read. */
  registers?: (number | null)[];
  reading?: number | null;
  stored?: boolean;
  /** Two runs' totals at DIFF_POINT. */
  runs?: boolean;
  tally?: number;
  step?: number;
}

function Stage(f: Frame) {
  return (
    <svg viewBox="0 0 720 345" role="img" aria-label={tr('Scatter and gather on a row of six points', 'Scatter và gather trên một hàng sáu điểm')}
      style={{ width: '100%', maxWidth: 720, display: 'block' }}
      data-mode={f.mode} data-step={f.step ?? ''} data-hot={f.hot ?? ''}
      data-registers={f.registers ? JSON.stringify(f.registers.map((r) => (r == null ? null : f32text(r)))) : ''}
      data-cells={f.cells ? JSON.stringify(f.cells) : ''}>
      <text x={16} y={24} fontSize="14" fontWeight="600" data-headline="">{f.headline}</text>
      {/* atoms, in constant memory */}
      <text x={16} y={70} fontSize="11.5" style={{ fill: 'var(--const)' }} fontWeight="600"><Translated vi="các nguyên tử (bộ nhớ hằng)">atoms (constant memory)</Translated></text>
      {atoms.map((a, k) => {
        const reading = f.reading === k;
        return (
          <g key={k} data-atom={k}>
            <rect x={16 + k * 44} y={78} width={40} height={24} rx="3" fill={reading ? 'var(--const)' : 'var(--const-wash)'} stroke="var(--const)" />
            <text x={36 + k * 44} y={94} fontSize="11" textAnchor="middle" style={{ fill: reading ? 'var(--page)' : 'var(--ink)' }}><Translated vi={<>nguyên tử {k}</>}>atom {k}</Translated></text>
            <circle cx={ax(k)} cy={ay(k)} r={13} fill="var(--const)" fillOpacity={f.mode === 'gather' && !reading ? 0.45 : 0.9} />
            <text x={ax(k)} y={ay(k) + 4} fontSize="11" textAnchor="middle" style={{ fill: 'var(--page)' }}>{k}</text>
            {f.mode === 'scatter' ? (
              <text x={ax(k)} y={ay(k) - 18} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>luồng {k}</>}>thread {k}</Translated></text>
            ) : null}
            <text x={ax(k) + 16} y={ay(k) + 4} fontSize="9.5" style={{ fill: 'var(--ink-3)' }}>q {a.q > 0 ? '+' : ''}{a.q}</text>
          </g>
        );
      })}
      {/* lines from atoms to points */}
      {f.lines?.map(([k, i, o], n) => (
        <line key={n} x1={ax(k)} y1={ay(k) + 13} x2={cellX(i) + CW / 2} y2={ROW_Y - 2} stroke={f.mode === 'scatter' ? 'var(--global)' : 'var(--ink-3)'} strokeOpacity={o} strokeWidth={f.mode === 'scatter' ? 1.8 : 0.9} />
      ))}
      {/* broadcast of one constant address to every gather lane */}
      {f.mode === 'gather' && f.reading != null ? (
        <g data-broadcast={f.reading}>
          {Array.from({ length: W }, (_, i) => (
            <line key={i} x1={36 + f.reading! * 44} y1={102} x2={cellX(i) + CW / 2} y2={ROW_Y + 50} stroke="var(--const)" strokeOpacity="0.55" strokeWidth="1.2" />
          ))}
        </g>
      ) : null}
      {/* the row of points (global memory) */}
      <text x={16} y={ROW_Y + 24} fontSize="11.5" style={{ fill: 'var(--global)' }} fontWeight="600"><Translated vi="các điểm lưới (toàn cục)">grid points (global)</Translated></text>
      {Array.from({ length: W }, (_, i) => (
        <g key={i} data-cell={i}>
          <rect x={cellX(i) + 2} y={ROW_Y} width={CW - 4} height={38} rx="3"
            fill={f.hot === i ? 'var(--global-wash)' : 'var(--cell)'} stroke={f.hot === i ? 'var(--global)' : 'var(--rule)'} strokeWidth={f.hot === i ? 2 : 1} />
          <text x={cellX(i) + CW / 2} y={ROW_Y - 6} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}><Translated vi={<>điểm {i}</>}>point {i}</Translated></text>
          <text x={cellX(i) + CW / 2} y={ROW_Y + 23} fontSize="11.5" textAnchor="middle" className="mono">{f.cells?.[i] ?? ''}</text>
        </g>
      ))}
      {/* the queue of atomic updates at the hot point */}
      {f.queue && f.hot != null ? (
        <g data-queue={f.queue.length}>
          {f.queue.map((v, n) => (
            <g key={n}>
              <rect x={cellX(f.hot!) + 6} y={ROW_Y - 36 - n * 17} width={CW - 12} height={15} rx="2" fill="var(--global)" fillOpacity="0.85" />
              <text x={cellX(f.hot!) + CW / 2} y={ROW_Y - 25 - n * 17} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--page)' }}>{v < 0 ? '−' : '+'}{Math.abs(v).toFixed(3)}</text>
            </g>
          ))}
        </g>
      ) : null}
      {/* gather registers */}
      {f.registers ? (
        <g>
          <text x={16} y={ROW_Y + 72} fontSize="11.5" style={{ fill: 'var(--reg)' }} fontWeight="600"><Translated vi="thanh ghi, mỗi luồng một cái">registers, one per thread</Translated></text>
          {f.registers.map((r, i) => (
            <g key={i}>
              <rect x={cellX(i) + 6} y={ROW_Y + 52} width={CW - 12} height={30} rx="3" fill="var(--reg-wash)" stroke="var(--reg)" />
              <text x={cellX(i) + CW / 2} y={ROW_Y + 71} fontSize="11" textAnchor="middle" className="mono">{r == null ? '0' : val(r)}</text>
              <text x={cellX(i) + CW / 2} y={ROW_Y + 96} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}><Translated vi={<>luồng {i}</>}>thread {i}</Translated></text>
            </g>
          ))}
          {f.stored ? <text x={X0} y={ROW_Y + 120} fontSize="11.5" style={{ fill: 'var(--ok)' }}><Translated vi="mỗi luồng lưu thanh ghi của nó vào điểm của riêng nó: 6 lần ghi thường">each thread stores its register to its own point: 6 plain writes</Translated></text> : null}
        </g>
      ) : null}
      {/* two runs, two orders */}
      {f.runs ? (() => {
        const a = f32text(TOT_A);
        const b = f32text(TOT_B);
        const k = splitAt(a, b);
        return (
          <g data-runs={`${a}|${b}`}>
            <text x={16} y={ROW_Y + 70} fontSize="12"><Translated vi={<>điểm {DIFF_POINT}, lần chạy 1, các cập nhật áp dụng theo thứ tự luồng {runA.join(', ')}:</>}>point {DIFF_POINT}, run 1, updates applied in thread order {runA.join(', ')}:</Translated></text>
            <text x={470} y={ROW_Y + 70} fontSize="13" className="mono">{a.slice(0, k)}<tspan style={{ fill: 'var(--fail)', fontWeight: 700 }}>{a.slice(k)}</tspan></text>
            <text x={16} y={ROW_Y + 92} fontSize="12"><Translated vi={<>điểm {DIFF_POINT}, lần chạy 2, các cập nhật áp dụng theo thứ tự luồng {runB.join(', ')}:</>}>point {DIFF_POINT}, run 2, updates applied in thread order {runB.join(', ')}:</Translated></text>
            <text x={470} y={ROW_Y + 92} fontSize="13" className="mono">{b.slice(0, k)}<tspan style={{ fill: 'var(--fail)', fontWeight: 700 }}>{b.slice(k)}</tspan></text>
            <text x={16} y={ROW_Y + 112} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="cùng bốn số float32 đó, cộng theo một thứ tự khác">the same four float32 numbers, added in a different order</Translated></text>
          </g>
        );
      })() : null}
      {/* the tally */}
      {f.tally != null ? (
        <g opacity={f.tally} data-tally="">
          <rect x={16} y={140} width={688} height={62} fill="var(--page)" />
          {[
            [tr('scatter', 'scatter'), tr(`${TALLY.scatterUpdates} atomic updates to ${TALLY.scatterAddresses} addresses (${TALLY.updatesPerAddress} each)`, `${TALLY.scatterUpdates} cập nhật nguyên tử vào ${TALLY.scatterAddresses} địa chỉ (mỗi địa chỉ ${TALLY.updatesPerAddress})`), tr(`${TALLY.scatterConstPerStep} constant addresses per step`, `${TALLY.scatterConstPerStep} địa chỉ hằng mỗi bước`)],
            [tr('gather', 'gather'), tr(`${TALLY.gatherAdds} register additions, ${TALLY.gatherWrites} plain writes`, `${TALLY.gatherAdds} phép cộng vào thanh ghi, ${TALLY.gatherWrites} lần ghi thường`), tr(`${TALLY.gatherConstPerStep} constant address per step`, `${TALLY.gatherConstPerStep} địa chỉ hằng mỗi bước`)],
          ].map(([name, a, b], n) => (
            <g key={name} data-row={name}>
              <text x={20} y={162 + n * 26} fontSize="13" fontWeight="650" style={{ fill: n ? 'var(--ok)' : 'var(--fail)' }}>{name}</text>
              <text x={100} y={162 + n * 26} fontSize="12.5">{a}</text>
              <text x={480} y={162 + n * 26} fontSize="12.5">{b}</text>
            </g>
          ))}
        </g>
      ) : null}
    </svg>
  );
}

/** Scatter at step s, with `applied` of the four updates already added to point s (run 1's order). */
function scatterFrame(s: number, applied: number, headline: ReactNode): Frame {
  const cells: (string | null)[] = Array.from({ length: W }, (_, i) => {
    if (i < s) return val(scatterTotal(atoms, i, runA));
    if (i === s) return applied ? val(sumInOrder(SC[s].values, runA.slice(0, applied))) : val(0);
    return val(0);
  });
  return {
    mode: 'scatter',
    headline,
    cells,
    hot: s,
    step: s,
    lines: atoms.map((_, k) => [k, s, 0.9] as [number, number, number]),
    queue: runA.slice(applied).map((k) => SC[s].values[k]).reverse(),
  };
}

const beats: Beat[] = [
  {
    clip: 'scatter-gather-1',
    get label() { return tr('The pairs', 'Các cặp'); },
    draw: (p) => {
      const shown = Math.round(ramp(p, 0.05, 0.6) * atoms.length * W);
      const lines: [number, number, number][] = [];
      for (let n = 0; n < shown; n++) lines.push([Math.floor(n / W), n % W, 0.7]);
      return (
        <Stage mode="setup" lines={lines} cells={Array(W).fill('')}
          headline={p < 0.62 ? tr(`${shown} of ${atoms.length * W} atom-point pairs`, `${shown} trên ${atoms.length * W} cặp nguyên tử–điểm`) : tr(`${atoms.length * W} pairs: split them by atom (scatter, Fig. 21.5) or by point (gather, Fig. 21.6)`, `${atoms.length * W} cặp: chia theo nguyên tử (scatter, Fig. 21.5) hoặc theo điểm (gather, Fig. 21.6)`)} />
      );
    },
  },
  {
    clip: 'scatter-gather-2',
    get label() { return tr('Scatter: one point', 'Scatter: một điểm'); },
    draw: (p) => {
      const applied = Math.min(4, Math.floor(ramp(p, 0.3, 0.95) * 5));
      return <Stage {...scatterFrame(0, applied, applied < 4 ? tr(`step 0: four threads, one address, ${4 - applied} updates waiting`, `bước 0: bốn luồng, một địa chỉ, ${4 - applied} cập nhật đang chờ`) : tr('step 0: the four atomic updates applied one at a time', 'bước 0: bốn cập nhật nguyên tử được áp dụng lần lượt từng cái'))} />;
    },
  },
  {
    clip: 'scatter-gather-3',
    get label() { return tr('Scatter: every point', 'Scatter: mọi điểm'); },
    draw: (p) => {
      if (p < 0.62) {
        const s = Math.min(W - 1, 1 + Math.floor(ramp(p, 0, 0.62) * (W - 1)));
        const local = (ramp(p, 0, 0.62) * (W - 1)) % 1;
        const applied = Math.min(4, Math.floor(local * 5));
        return <Stage {...scatterFrame(s, applied, tr(`step ${s}: another four-way collision, on point ${s}`, `bước ${s}: thêm một va chạm bốn chiều, trên điểm ${s}`))} />;
      }
      const cells = Array.from({ length: W }, (_, i) => val(scatterTotal(atoms, i, runA)));
      return <Stage mode="scatter" cells={cells} hot={DIFF_POINT} runs step={W} headline={tr('two runs, two orders of the same updates', 'hai lần chạy, hai thứ tự của cùng các cập nhật')} />;
    },
  },
  {
    clip: 'scatter-gather-4',
    get label() { return tr('Gather: one atom', 'Gather: một nguyên tử'); },
    draw: (p) => {
      const r = p < 0.35 ? Array(W).fill(null) : GA[0].registers;
      return <Stage mode="gather" reading={0} registers={r} cells={Array(W).fill('')} step={0}
        headline={p < 0.35 ? tr('step 0: six threads, every lane reads atom 0', 'bước 0: sáu luồng, mọi lane đọc nguyên tử 0') : tr('step 0: one address, broadcast; each thread adds into its own register', 'bước 0: một địa chỉ, phát rộng; mỗi luồng cộng vào thanh ghi của riêng nó')} />;
    },
  },
  {
    clip: 'scatter-gather-5',
    get label() { return tr('Gather: one write', 'Gather: một lần ghi'); },
    draw: (p) => {
      const s = Math.min(atoms.length - 1, 1 + Math.floor(ramp(p, 0, 0.6) * (atoms.length - 1)));
      const stored = p >= 0.7;
      const regs = GA[stored ? atoms.length - 1 : s].registers;
      return (
        <Stage mode="gather" reading={stored ? null : s} registers={regs} stored={stored} step={stored ? atoms.length : s}
          cells={stored ? regs.map((v) => val(v)) : Array(W).fill('')}
          headline={stored ? tr('every point written once: the same bits on every run', 'mỗi điểm được ghi một lần: cùng các bit ở mọi lần chạy') : tr(`step ${s}: every lane reads atom ${s}`, `bước ${s}: mọi lane đọc nguyên tử ${s}`)} />
      );
    },
  },
  {
    clip: 'scatter-gather-6',
    get label() { return tr('Tally', 'Tổng kết'); },
    draw: (p) => (
      <Stage mode="tally" tally={ramp(p, 0.05, 0.3)} cells={GA[atoms.length - 1].registers.map((v) => val(v))}
        headline={tr('the same 24 pairs, two ways to share them out', 'cùng 24 cặp, hai cách chia chúng ra')} />
    ),
  },
];

export function ScatterGatherScene() {
  return <Scene chapter="ch21" title={tr('Scatter and gather, step by step', 'Scatter và gather, từng bước một')} beats={beats} />;
}
