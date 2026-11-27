import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  design,
  race,
  regions,
  sweep,
  timeline,
  waves,
  type Design,
  type Finish,
  type Region,
  type SweepPoint,
} from '../../sim/designs';
import { BUDGET, Plan, TwoDesignsScene } from './TwoDesignsScene';
import { useWidth } from './useWidth';

// ---------------------------------------------------------------- helpers
const nf = (v: number) => v.toLocaleString('en-US', { maximumFractionDigits: 3 });
const plural = (n: number, one: string) => `${nf(n)} ${one}${n === 1 ? '' : 's'}`;
const MAX_CHAINS = 4096;
const MAX_LENGTH = 1000;
const HALVINGS = [0, 1, 2, 3];
/** A caption above a group of buttons (a <label> would forward clicks on the caption to the first button). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const describe = (d: Design) => tr(`${plural(d.units, 'unit')}, speed ${d.speed}`, `${nf(d.units)} đơn vị, tốc độ ${d.speed}`);

// The pair the prediction and the default explorer use: the animation's two designs.
const FAST = design(BUDGET, 2);
const MANY = design(BUDGET, 0);

// q-chainlen: two jobs where the one with more arithmetic has fewer independent pieces.
// Both fit the explorer's ranges, so the reader can check them there.
export const LONG_FEW = { chains: 8, length: 1000 };
export const SHORT_MANY = { chains: 1024, length: 5 };
const longFew = race(FAST, MANY, LONG_FEW.chains, LONG_FEW.length);
const shortMany = race(FAST, MANY, SHORT_MANY.chains, SHORT_MANY.length);
const ops = (j: { chains: number; length: number }) => nf(j.chains * j.length);

// ---------------------------------------------------------------- Fig. 1.1, redrawn
/**
 * Our own drawing of the idea in the book's Fig. 1.1: the same chip area divided
 * two ways. Schematic only. The unit counts are the two designs used on this page.
 */
function AreaFigure() {
  const W = 640;
  const chipW = 286;
  const H = 262;
  // Below this width the two chips are drawn one above the other, each at its own size,
  // so that the labels stay legible on a phone.
  const [ref, measured] = useWidth(W);
  const stacked = measured < 600;
  const top = 34;
  const chip = (x: number, title: string, bands: [string, number][], d: Design, channels: number, channelNote: string) => {
    let y = top;
    const rows = d.units > 8 ? 4 : 1;
    const cols = d.units / rows;
    const out = bands.map(([label, h], i) => {
      const y0 = y;
      y += h;
      const arithmetic = i === bands.length - 1;
      return (
        <g key={label}>
          <rect x={x} y={y0} width={chipW} height={h} fill={arithmetic ? 'var(--bench)' : 'var(--cell)'} stroke="var(--ink)" strokeWidth="1" />
          {arithmetic ? (
            Array.from({ length: d.units }, (_, k) => (
              <rect key={k} x={x + 5 + (k % cols) * ((chipW - 10) / cols) + 1.5} y={y0 + 5 + Math.floor(k / cols) * ((h - 10) / rows) + 1.5}
                width={(chipW - 10) / cols - 3} height={(h - 10) / rows - 3} rx="1.5" fill="var(--page)" stroke="var(--ink)" strokeWidth={rows > 1 ? 0.9 : 1.5} />
            ))
          ) : (
            <text x={x + chipW / 2} y={y0 + h / 2 + 4} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--ink)' }}>{label}</text>
          )}
        </g>
      );
    });
    const bottom = y;
    return (
      <g>
        <text x={x} y={14} fontSize="13" fontWeight="600" style={{ fill: 'var(--ink)' }}>{title}</text>
        <text x={x} y={28} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{bands[bands.length - 1][0]}</text>
        {out}
        {Array.from({ length: channels }, (_, k) => {
          const cx = x + ((k + 0.5) * chipW) / channels;
          return <line key={k} x1={cx} x2={cx} y1={bottom} y2={bottom + 22} stroke="var(--ink)" strokeWidth={channels > 4 ? 2 : 5} />;
        })}
        <rect x={x} y={bottom + 22} width={chipW} height={22} fill="var(--cell)" stroke="var(--ink)" />
        <text x={x + chipW / 2} y={bottom + 37} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--ink)' }}><Translated vi="bộ nhớ">memory</Translated></text>
        <text x={x} y={bottom + 60} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{channelNote}</text>
      </g>
    );
  };
  const cpu = (x: number) =>
    chip(x, tr('Latency-oriented (a CPU)', 'Hướng độ trễ (một CPU)'),
      [[tr('control logic and branch prediction', 'logic điều khiển và dự đoán rẽ nhánh'), 48], [tr('large caches', 'bộ nhớ đệm lớn'), 58], [tr(`arithmetic: ${describe(FAST).replace(', speed', ' of speed')}`, `số học: ${describe(FAST)}`), 54]], FAST, 2, tr('few channels to memory', 'ít kênh tới bộ nhớ'));
  const gpu = (x: number) =>
    chip(x, tr('Throughput-oriented (a GPU)', 'Hướng thông lượng (một GPU)'),
      [[tr('control', 'điều khiển'), 18], [tr('small caches', 'bộ nhớ đệm nhỏ'), 18], [tr(`arithmetic: ${describe(MANY).replace(', speed', ' of speed')}`, `số học: ${describe(MANY)}`), 124]], MANY, 8, tr('many channels to memory, each allowed to be slow', 'nhiều kênh tới bộ nhớ, mỗi kênh được phép chậm'));
  const svgStyle = { width: '100%', display: 'block', fontFamily: 'var(--sans)' } as const;
  return (
    <figure style={{ margin: '0.2rem 0 1.4rem' }} data-stacked={stacked ? '1' : '0'}>
      <div ref={ref} style={{ width: '100%' }}>
        {stacked ? (
          <div style={{ display: 'grid', gap: '1rem' }}>
            <svg viewBox={`-2 0 ${chipW + 4} ${H}`} role="img" aria-label={tr('Chip area of a latency-oriented design', 'Diện tích chip của một thiết kế hướng độ trễ')} style={{ ...svgStyle, maxWidth: 360 }}>
              {cpu(0)}
            </svg>
            <svg viewBox={`-2 0 ${chipW + 4} ${H}`} role="img" aria-label={tr('Chip area of a throughput-oriented design', 'Diện tích chip của một thiết kế hướng thông lượng')} style={{ ...svgStyle, maxWidth: 360 }}>
              {gpu(0)}
            </svg>
          </div>
        ) : (
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('The same chip area divided two ways', 'Cùng một diện tích chip được chia theo hai cách')} style={{ ...svgStyle, maxWidth: W }}>
            {cpu(0)}
            {gpu(W - chipW)}
          </svg>
        )}
      </div>
      <figcaption className="ui muted" style={{ fontSize: '0.8rem', marginTop: '0.4rem' }}>
        <Translated
          vi={
            <>
              Diện tích đi về đâu, vẽ lại từ ý tưởng của Hình 1.1 trong sách. Đây là sơ đồ: tỷ lệ không phải số đo,
              và số đơn vị là của hai thiết kế trong hoạt hình bên dưới.
            </>
          }
        >
          Where the area goes, redrawn from the idea of the book's Fig. 1.1. A schematic: the proportions are not
          measurements, and the unit counts are the two designs of the animation below.
        </Translated>
      </figcaption>
    </figure>
  );
}

// ---------------------------------------------------------------- explorer drawings
/** Which unit runs which chain: one row per unit, time across, on a scale shared by both designs. */
function Strip({ d, chains, length, tMax, f }: { d: Design; chains: number; length: number; tMax: number; f: Finish }) {
  const W = 300;
  const top = 8;
  const hgt = 100;
  const x0 = 2;
  const w = W - 4;
  const X = (t: number) => x0 + (t / tMax) * w;
  const row = hgt / d.units;
  const shade = (wave: number) => (wave % 2 ? 'var(--ink-3)' : 'var(--ink-2)');
  const ws = waves(d, chains, length);
  let marks;
  if (chains <= 512) {
    // few enough to draw every chain
    marks = timeline(d, chains, length).map((s) => (
      <rect key={s.chain} x={X(s.start)} y={top + s.unit * row} width={Math.max(0.8, X(s.end) - X(s.start) - (ws.length > 1 ? 0.8 : 0))}
        height={Math.max(0.8, row - (row > 3 ? 1 : 0.4))} fill={shade(s.wave)} />
    ));
  } else if (ws.length <= 400) {
    // one block per wave
    marks = ws.map((v) => (
      <rect key={v.wave} x={X(v.start)} y={top} width={Math.max(0.6, X(v.end) - X(v.start) - 0.5)} height={(v.units / d.units) * hgt} fill={shade(v.wave)} />
    ));
  } else {
    // too many waves to tell apart: the full ones as one block, then the last
    const lastWave = ws[ws.length - 1];
    marks = [
      <rect key="full" x={X(0)} y={top} width={X(lastWave.start) - X(0)} height={hgt} fill="var(--ink-2)" />,
      <rect key="last" x={X(lastWave.start)} y={top} width={Math.max(0.6, X(lastWave.end) - X(lastWave.start))} height={(lastWave.units / d.units) * hgt} fill="var(--ink-3)" />,
    ];
  }
  const endRight = X(f.last) > W - 130;
  return (
    <svg viewBox={`0 0 ${W} 132`} role="img" aria-label={tr(`Which of the ${d.units} units runs which chain, over time`, `Đơn vị nào trong ${d.units} đơn vị chạy chuỗi nào, theo thời gian`)} style={{ width: '100%', maxWidth: 440, display: 'block' }}>
      <rect x={x0} y={top} width={w} height={hgt} fill="var(--cell)" stroke="var(--rule)" />
      {marks}
      <line x1={X(f.first)} x2={X(f.first)} y1={top - 5} y2={top + hgt + 5} stroke="var(--ink)" strokeWidth="1" strokeDasharray="3 3" />
      <line x1={X(f.last)} x2={X(f.last)} y1={top - 5} y2={top + hgt + 5} stroke="var(--ink)" strokeWidth="1.8" />
      <text x={x0} y={top + hgt + 20} fontSize="11" style={{ fill: 'var(--ink-2)' }}>0</text>
      <text x={endRight ? X(f.last) : X(f.last) + 5} y={top + hgt + 20} fontSize="11" textAnchor={endRight ? 'end' : 'start'}>
        <Translated vi={<>kết quả cuối cùng lúc {nf(f.last)}</>}>last result at {nf(f.last)}</Translated>
      </text>
    </svg>
  );
}

/** Finishing time of both designs against the number of chains, log scales, with the current point marked. */
function CrossPlot({ pts, regs, chains, a, b }: { pts: SweepPoint[]; regs: Region[]; chains: number; a: Design; b: Design }) {
  // Drawn at its measured width, so that its text keeps its size on a phone.
  const [ref, W] = useWidth(560);
  const narrow = W < 440;
  // On a wide plot, room on the right for the label of the last point and the legend inside the frame.
  // On a narrow one the label turns left and the legend moves under the axis, where no line runs.
  const H = Math.round(Math.min(330, Math.max(250, W * 0.59))) + (narrow ? 40 : 0);
  const pad = { l: 58, r: narrow ? 26 : 70, t: 14, b: narrow ? 84 : 44 };
  const lo = Math.min(pts[0].a, pts[0].b) / 1.7;
  const hi = Math.max(pts[pts.length - 1].a, pts[pts.length - 1].b) * 1.7;
  const lx = (v: number) => Math.log2(v) / Math.log2(MAX_CHAINS);
  const X = (v: number) => pad.l + lx(v) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (H - pad.t - pad.b);
  // A staircase has long flat runs: keep only the points where a value changes.
  const line = (key: 'a' | 'b') =>
    pts
      .filter((q, i) => i === 0 || i === pts.length - 1 || pts[i - 1][key] !== q[key] || pts[i + 1][key] !== q[key])
      .map((q) => `${X(q.chains).toFixed(1)},${Y(q[key]).toFixed(1)}`)
      .join(' ');
  const xt = [1, 4, 16, 64, 256, 1024, 4096];
  const yt: number[] = [];
  for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++) yt.push(10 ** e);
  const cur = pts[chains - 1];
  const same = cur.a === cur.b;
  const aAbove = cur.a > cur.b;
  // The slower design's label goes above its line and to the left, the faster one's below and to the right,
  // where neither staircase runs.
  const cx = X(chains);
  const upper = cx < pad.l + 84 ? { dx: 9, anchor: 'start' as const } : { dx: -9, anchor: 'end' as const };
  const lower = cx > W - 90 ? { dx: -9, dy: 18, anchor: 'end' as const } : { dx: 9, dy: 18, anchor: 'start' as const };
  const lowest = Math.max(Y(cur.a), Y(cur.b));
  const lg = narrow ? { x: 20, y: H - pad.b + 56 } : { x: W - pad.r - 176, y: H - pad.b - 34 };
  return (
    <div ref={ref} style={{ width: '100%', overflow: 'hidden' }}>
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Time to the last result against the number of independent chains', 'Thời gian đến kết quả cuối theo số chuỗi độc lập')} style={{ display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {regs.filter((g) => g.winner === 'tie' && regs.length > 1).map((g) => (
        <rect key={g.from} x={X(g.from) - 1} y={pad.t} width={X(g.to) - X(g.from) + 2} height={H - pad.t - pad.b} fill="var(--rule-soft)" stroke="var(--rule)" />
      ))}
      {xt.map((v) => (
        <g key={'x' + v}>
          <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(v)} y={H - pad.b + 15} fontSize="11" textAnchor="middle">{nf(v)}</text>
        </g>
      ))}
      {yt.map((v) => (
        <g key={'y' + v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end">{nf(v)}</text>
        </g>
      ))}
      <text x={(W - pad.r + pad.l) / 2} y={H - pad.b + 36} fontSize="11.5" textAnchor="middle"><Translated vi="chuỗi độc lập">independent chains</Translated></text>
      <text x={14} y={(H - pad.b + pad.t) / 2} fontSize="11.5" textAnchor="middle" transform={`rotate(-90 14 ${(H - pad.b + pad.t) / 2})`}><Translated vi="thời gian đến kết quả cuối">time to the last result</Translated></text>
      <polyline points={line('a')} fill="none" stroke="var(--ink)" strokeWidth="2.2" strokeLinejoin="round" /* i18n-ignore: SVG attribute value */ />
      <polyline points={line('b')} fill="none" stroke="var(--ink)" strokeWidth="2.2" strokeDasharray="6 4" strokeLinejoin="round" /* i18n-ignore: SVG attribute value */ />
      {/* legend */}
      <line x1={lg.x} x2={lg.x + 28} y1={lg.y} y2={lg.y} stroke="var(--ink)" strokeWidth="2.2" />
      <text x={lg.x + 34} y={lg.y + 4} fontSize="11.5">A: {describe(a)}</text>
      <line x1={lg.x} x2={lg.x + 28} y1={lg.y + 18} y2={lg.y + 18} stroke="var(--ink)" strokeWidth="2.2" strokeDasharray="6 4" />
      <text x={lg.x + 34} y={lg.y + 22} fontSize="11.5">B: {describe(b)}</text>
      {/* the current point */}
      <line x1={cx} x2={cx} y1={pad.t} y2={lowest} stroke="var(--ink)" strokeDasharray="3 3" />
      <circle cx={cx} cy={Y(cur.a)} r="4.5" fill="var(--ink)" />
      <circle cx={cx} cy={Y(cur.b)} r="4.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.8" />
      {same ? (
        <text x={cx + upper.dx} y={Y(cur.a) - 9} fontSize="11.5" fontWeight="600" textAnchor={upper.anchor}><Translated vi={<>A và B: {nf(cur.a)}</>}>A and B: {nf(cur.a)}</Translated></text>
      ) : (
        <>
          <text x={cx + (aAbove ? upper.dx : lower.dx)} y={Y(cur.a) + (aAbove ? -9 : lower.dy)} fontSize="11.5" fontWeight="600"
            textAnchor={aAbove ? upper.anchor : lower.anchor}>A: {nf(cur.a)}</text>
          <text x={cx + (aAbove ? lower.dx : upper.dx)} y={Y(cur.b) + (aAbove ? lower.dy : -9)} fontSize="11.5" fontWeight="600"
            textAnchor={aAbove ? lower.anchor : upper.anchor}>B: {nf(cur.b)}</text>
        </>
      )}
    </svg>
    </div>
  );
}

const leads = (regs: Region[]) =>
  regs
    .map((g) => {
      const span = g.from === g.to ? tr(`at ${nf(g.from)}`, `tại ${nf(g.from)}`) : tr(`from ${nf(g.from)} to ${nf(g.to)}`, `từ ${nf(g.from)} đến ${nf(g.to)}`);
      return g.winner === 'tie' ? tr(`a tie ${span}`, `hòa ${span}`) : tr(`${g.winner === 'a' ? 'A' : 'B'} leads ${span}`, `${g.winner === 'a' ? 'A' : 'B'} dẫn trước ${span}`);
    })
    .join(', ');

function BudgetBench() {
  const [ha, setHa] = useState(2);
  const [hb, setHb] = useState(0);
  const [chains, setChains] = useState(1);
  const [length, setLength] = useState(100);
  const a = design(BUDGET, ha);
  const b = design(BUDGET, hb);
  const r = race(a, b, chains, length);
  const pts = useMemo(() => sweep(design(BUDGET, ha), design(BUDGET, hb), MAX_CHAINS, length), [ha, hb, length]);
  const regs = useMemo(() => regions(pts), [pts]);
  const tMax = Math.max(r.a.last, r.b.last);
  const clampChains = (v: number) => Math.max(1, Math.min(MAX_CHAINS, Math.round(v) || 1));

  const pick = (label: string, h: number, set: (h: number) => void) => (
    <span style={FIELD}>
      <Translated vi={<>{label}: giảm một nửa độ trễ của một đơn vị {h === 1 ? 'một lần' : `${h} lần`}</>}>
        {label}: halve the latency of a unit {h === 1 ? 'once' : `${h} times`}
      </Translated>
      <span className="seg" role="group" aria-label={label}>
        {HALVINGS.map((k) => (
          <button key={k} aria-pressed={h === k} onClick={() => set(k)}>
            {describe(design(BUDGET, k))}
          </button>
        ))}
      </span>
    </span>
  );
  const panel = (key: 'a' | 'b', d: Design, f: Finish) => (
    <div className="panel" data-design={key}>
      <Translated
        vi={
          <p style={{ margin: '0 0 0.5rem' }}>
            <b>Thiết kế {key.toUpperCase()}</b>: {describe(d)}. Mỗi đơn vị dùng {d.unitPower} trong số {BUDGET}.
          </p>
        }
      >
        <p style={{ margin: '0 0 0.5rem' }}>
          <b>Design {key.toUpperCase()}</b>: {describe(d)}. {d.unitPower === 1 ? 'Each unit draws 1 of' : `Each unit draws ${d.unitPower} of`} the{' '}
          {BUDGET}.
        </p>
      </Translated>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem 0.9rem', alignItems: 'flex-start' }}>
        <svg viewBox="0 0 120 120" role="img" aria-label={tr(`Floor plan: ${describe(d)}`, `Sơ đồ mặt bằng: ${describe(d)}`)} style={{ width: 120, flex: '0 0 auto', display: 'block' }}>
          <Plan x={0} y={0} size={120} d={d} busy={new Set(Array.from({ length: d.units - f.idle }, (_, i) => i))} />
        </svg>
        <div style={{ flex: '1 1 220px', minWidth: 0 }}>
          <Strip d={d} chains={chains} length={length} tMax={tMax} f={f} />
        </div>
      </div>
      <p className="muted" style={{ margin: '0.4rem 0 0' }} data-out={key + '-used'}>
        {f.idle === 0
          ? tr(`Every unit gets a chain. ${plural(f.waves, 'wave')}.`, `Mọi đơn vị đều nhận một chuỗi. ${nf(f.waves)} đợt.`)
          : tr(`${plural(d.units - f.idle, 'unit')} working, ${nf(f.idle)} never used. ${plural(f.waves, 'wave')}.`, `${nf(d.units - f.idle)} đơn vị làm việc, ${nf(f.idle)} không bao giờ dùng đến. ${nf(f.waves)} đợt.`)}
      </p>
    </div>
  );
  const stat = (label: string, key: string, va: number | string, vb: number | string) => (
    <tr>
      <td>{label}</td>
      <td className="num" data-stat={'a-' + key}>{typeof va === 'number' ? nf(va) : va}</td>
      <td className="num" data-stat={'b-' + key}>{typeof vb === 'number' ? nf(vb) : vb}</td>
    </tr>
  );
  return (
    <Bench
      title={tr('Spend a power budget', 'Tiêu một ngân sách công suất')}
      task={tr('Find the number of chains at which the lead changes hands. Then change the chain length and see what moves.', 'Tìm số chuỗi mà tại đó bên dẫn trước đổi chủ. Rồi đổi độ dài chuỗi và xem điều gì dịch chuyển.')}
      foot={
        <Translated
          vi={
            <>
              Thời gian được đếm bằng số phép toán của một đơn vị thường (tốc độ 1). Trong các dải, mỗi hàng là một đơn vị,
              các đợt xen kẽ có sắc độ xen kẽ, vạch đứt là kết quả đầu tiên và vạch liền là kết quả cuối. Hai dải dùng chung
              một thang thời gian. Ngân sách {BUDGET} là một con số trừu tượng, không phải một thiết bị.
            </>
          }
        >
          Time is counted in operations of a plain unit (speed 1). In the strips, one row is one unit, alternate waves
          alternate in shade, the dashed line is the first result and the solid line the last. Both strips share one
          time scale. The budget of {BUDGET} is an abstract number, not a device.
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        {pick(tr('Design A', 'Thiết kế A'), ha, setHa)}
        {pick(tr('Design B', 'Thiết kế B'), hb, setHb)}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi={<>Số chuỗi độc lập = {nf(chains)}</>}>Independent chains = {nf(chains)}</Translated>
          <input type="range" min="0" max="120" value={Math.round(10 * Math.log2(chains))} aria-label={tr('Independent chains (log scale)', 'Số chuỗi độc lập (thang log)')}
            onChange={(e) => setChains(clampChains(2 ** (+e.target.value / 10)))} />
        </label>
        <label className="field">
          <Translated vi="số chính xác">exact number</Translated>
          <input type="number" min="1" max={MAX_CHAINS} step="1" value={chains} aria-label={tr('Independent chains (exact)', 'Số chuỗi độc lập (chính xác)')}
            onChange={(e) => setChains(clampChains(+e.target.value))} />
        </label>
        <label className="field">
          <Translated vi={<>Số phép toán trong một chuỗi = {nf(length)}</>}>Operations in a chain = {nf(length)}</Translated>
          <input type="range" min="1" max={MAX_LENGTH} value={length} aria-label={tr('Operations in a chain', 'Số phép toán trong một chuỗi')} onChange={(e) => setLength(+e.target.value)} />
        </label>
      </div>
      <div className="cols" style={{ marginBottom: '1.2rem' }}>
        {panel('a', a, r.a)}
        {panel('b', b, r.b)}
      </div>
      <div className="cols plot">
        <CrossPlot pts={pts} regs={regs} chains={chains} a={a} b={b} />
        <div className="panel">
          <p style={{ fontSize: '1.15rem', fontWeight: 600, margin: '0 0 0.2rem' }} data-out="verdict">
            {r.winner === 'tie'
              ? tr(`A tie: both finish at ${nf(r.a.last)}.`, `Hòa: cả hai xong lúc ${nf(r.a.last)}.`)
              : tr(`Design ${r.winner.toUpperCase()} finishes first, ${nf(r.factor)} times sooner.`, `Thiết kế ${r.winner.toUpperCase()} xong trước, sớm hơn ${nf(r.factor)} lần.`)}
          </p>
          <p className="muted" style={{ margin: '0 0 0.7rem' }}>
            <Translated vi={<>{nf(chains)} chuỗi, mỗi chuỗi {nf(length)} phép toán.</>}>
              {plural(chains, 'chain')} of {plural(length, 'operation')}.
            </Translated>
          </p>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th />
                <th className="num">A</th>
                <th className="num">B</th>
              </tr>
            </thead>
            <tbody>
              {stat(tr('Units × speed of each', 'Số đơn vị × tốc độ mỗi đơn vị'), 'shape', `${a.units} × ${a.speed}`, `${b.units} × ${b.speed}`)}
              {stat(tr('Peak rate, every unit busy', 'Tốc độ đỉnh, mọi đơn vị đều bận'), 'peak', a.throughput, b.throughput)}
              {stat(tr('First result at', 'Kết quả đầu tiên lúc'), 'first', r.a.first, r.b.first)}
              {stat(tr('Last result at', 'Kết quả cuối cùng lúc'), 'last', r.a.last, r.b.last)}
              {stat(tr('Rate delivered on this work', 'Tốc độ đạt được trên công việc này'), 'delivered', r.a.delivered, r.b.delivered)}
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.5rem 0 0' }}><Translated vi="Tốc độ tính bằng số phép toán trên mỗi đơn vị thời gian.">Rates are operations per time unit.</Translated></p>
          <p style={{ margin: '0.7rem 0 0' }} data-out="regions">
            {regs.length === 1
              ? tr('These two designs are the same, so every number of chains is a tie.', 'Hai thiết kế này giống nhau, nên mọi số chuỗi đều hòa.')
              : tr(`For these two designs, at any chain length: ${leads(regs)} chains, the most this explorer runs.`, `Với hai thiết kế này, ở mọi độ dài chuỗi: ${leads(regs)} chuỗi, mức tối đa trình khám phá này chạy.`)}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questions10: Record<string, Question> = {
  'ch01/start/q-batch': {
    kind: 'choice',
    id: 'ch01/start/q-batch',
    prompt: (
      <Translated
        vi={
          <>
            Một máy chủ GPU chạy mô hình cho từng yêu cầu một (kích thước batch 1). Bạn chuyển sang batch 32, và luôn có
            đủ yêu cầu đang chờ để lấp đầy một batch. Điều gì xảy ra với thời gian một yêu cầu cần khi nó bắt đầu chạy,
            và với số yêu cầu hoàn thành mỗi giây?
          </>
        }
      >
        A GPU server runs a model on one request at a time (batch size 1). You switch it to batches of 32, and there are
        always enough requests waiting to fill a batch. What happens to the time one request takes once it starts
        running, and to the number of requests finished per second?
      </Translated>
    ),
    options: [
      {
        get label() {
          return tr('Each request takes longer; requests per second go up.', 'Mỗi yêu cầu lâu hơn; số yêu cầu mỗi giây tăng.');
        },
        correct: true,
      },
      {
        get label() {
          return tr('Each request is quicker; requests per second go up.', 'Mỗi yêu cầu nhanh hơn; số yêu cầu mỗi giây tăng.');
        },
        get why() {
          return tr(
            'A pass over 32 requests does more work than a pass over one. No single request gets its answer sooner than it would alone.',
            'Một lượt qua 32 yêu cầu làm nhiều việc hơn một lượt qua một yêu cầu. Không yêu cầu nào nhận được câu trả lời sớm hơn so với khi chạy một mình.',
          );
        },
      },
      {
        get label() {
          return tr('Each request takes the same time; requests per second go up 32 times.', 'Mỗi yêu cầu mất cùng thời gian; số yêu cầu mỗi giây tăng 32 lần.');
        },
        get why() {
          return tr(
            'The batched pass is not free. It takes longer than one request alone, though far less than 32 of them in a row.',
            'Lượt chạy theo batch không miễn phí. Nó lâu hơn một yêu cầu chạy một mình, dù ngắn hơn nhiều so với 32 yêu cầu chạy nối tiếp.',
          );
        },
      },
      {
        get label() {
          return tr('Each request takes longer; requests per second stay the same.', 'Mỗi yêu cầu lâu hơn; số yêu cầu mỗi giây giữ nguyên.');
        },
        get why() {
          return tr(
            'Then nobody would batch. The batched pass takes much less time than its 32 requests run one after another.',
            'Vậy thì chẳng ai dùng batch. Lượt chạy theo batch tốn ít thời gian hơn nhiều so với 32 yêu cầu của nó chạy lần lượt.',
          );
        },
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một lượt qua 32 yêu cầu lâu hơn một lượt qua một yêu cầu, và ngắn hơn nhiều so với 32 lượt. Vì thế mỗi yêu cầu
            chờ lâu hơn còn cả nhóm xong sớm hơn. Sách trình bày như vậy trong mục về batching, rồi nói rằng đó cũng là sự
            đánh đổi giữa hai loại bộ xử lý: một thiết kế cho mỗi phần việc độ trễ ngắn hơn, thiết kế kia cho nhiều phần
            việc thông lượng cao hơn. Trang này nói về sự đánh đổi đó, được đúc vào phần cứng.
          </p>
        }
      >
        <p>
          One pass over 32 requests takes longer than a pass over one, and much less than 32 passes. So each request
          waits longer and the group is done sooner. The book puts it this way in its section on batching, and then says
          it is the same trade as the one between the two kinds of processor: one design gives each piece of work a
          shorter latency, the other gives many pieces a higher throughput. This page is about that trade, built into
          hardware.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.5, pp. 539–540',
  },
  'ch01/start/q-sixteenbit': {
    kind: 'choice',
    id: 'ch01/start/q-sixteenbit',
    prompt: (
      <Translated
        vi={
          <>
            Sách liệt kê 34, 67 và 1,979 TFLOPS cho số học 64, 32 và 16 bit trên GPU ví dụ của nó. Đi từ 64 xuống 32 bit
            thì tốc độ gấp đôi. Điều gì giải thích bước nhảy khoảng 30 lần từ 32 xuống 16 bit?
          </>
        }
      >
        The book lists 34, 67 and 1,979 TFLOPS for 64-, 32- and 16-bit arithmetic on its example GPU. Going from 64 to
        32 bits doubles the rate. What explains the step of about 30 times from 32 to 16 bits?
      </Translated>
    ),
    options: [
      {
        get label() {
          return tr(
            'Halving the precision again: a 16-bit unit is much smaller than a 32-bit one, so far more of them fit on the chip.',
            'Lại giảm một nửa độ chính xác: một đơn vị 16 bit nhỏ hơn nhiều so với đơn vị 32 bit, nên chip chứa được nhiều hơn hẳn.',
          );
        },
        get why() {
          return tr(
            'The step from 64 to 32 bits shows what narrower values buy: a factor of 2. The same step again would give about 134, not 1,979.',
            'Bước từ 64 xuống 32 bit cho thấy giá trị hẹp hơn mua được gì: hệ số 2. Lặp lại bước đó sẽ cho khoảng 134, không phải 1,979.',
          );
        },
      },
      {
        get label() {
          return tr(
            'Another kind of unit, and an assumption: it is the tensor-core (matrix-multiply) rate, quoted with sparsity.',
            'Một loại đơn vị khác, và một giả định: đó là tốc độ của tensor core (nhân ma trận), công bố kèm tính thưa (sparsity).',
          );
        },
        correct: true,
      },
      {
        get label() {
          return tr(
            'Memory: 16-bit values need half the bandwidth, so twice as many of them arrive per second.',
            'Bộ nhớ: giá trị 16 bit cần một nửa băng thông, nên số giá trị đến mỗi giây gấp đôi.',
          );
        },
        get why() {
          return tr(
            'These are peak rates of the arithmetic units. Memory does not enter into them.',
            'Đây là tốc độ đỉnh của các đơn vị số học. Bộ nhớ không tham gia vào chúng.',
          );
        },
      },
      {
        get label() {
          return tr(
            'A misprint: by the pattern of the first two figures, the 16-bit one should read about 134.',
            'Lỗi in: theo quy luật của hai con số đầu, con số 16 bit lẽ ra phải khoảng 134.',
          );
        },
        get why() {
          return tr('The vendor\'s own table prints the same three numbers.', 'Chính bảng của nhà sản xuất in đúng ba con số đó.');
        },
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Giảm một nửa độ rộng của giá trị chỉ làm tốc độ gấp đôi một lần, nên giảm một nửa nữa không thể giải thích hệ
            số 30. Con số thứ ba phải đến từ một hàng khác trong bảng của nhà sản xuất, và đúng vậy: hàng đó là "FP16
            Tensor Core", kèm chú thích "With sparsity". Tensor core là các đơn vị được xây dựng cho phép nhân ma trận. Vì
            thế với suy luận LLM, phép so sánh quan trọng không phải 16 bit với 32 bit. Đó là đơn vị ma trận với đơn vị
            thông thường, và những gì một kernel phải làm để dùng được chúng, tức chương 15.
          </p>
        }
      >
        <p>
          Halving the width of the values doubled the rate once, so a further halving cannot account for a factor of 30.
          The third number has to come from a different row of the vendor's table, and it does: the row is "FP16 Tensor
          Core", and it carries the footnote "With sparsity". Tensor cores are units built for matrix multiplication.
          For LLM inference the comparison that matters is therefore not 16 bits against 32. It is matrix units against
          ordinary ones, and what a kernel has to do to use them, which is chapter 15.
        </p>
      </Translated>
    ),
    source: 'book p. 3; NVIDIA H100 specification table, checked 2026-11-20 (notes/today/ch01-facts.md, A1)',
    vi: { source: 'sách tr. 3; bảng thông số NVIDIA H100, kiểm tra ngày 2026-11-20 (notes/today/ch01-facts.md, A1)' },
  },
  'ch01/start/q-chainlen': {
    kind: 'choice',
    id: 'ch01/start/q-chainlen',
    prompt: (
      <Translated
        vi={
          <>
            Hai công việc cho các thiết kế của hoạt hình, A ({describe(FAST)}) và B ({describe(MANY)}). Công việc X gồm{' '}
            {nf(LONG_FEW.chains)} chuỗi độc lập, mỗi chuỗi {nf(LONG_FEW.length)} phép toán: tổng cộng {ops(LONG_FEW)}{' '}
            phép toán. Công việc Y gồm {nf(SHORT_MANY.chains)} chuỗi độc lập, mỗi chuỗi {nf(SHORT_MANY.length)} phép
            toán: tổng cộng {ops(SHORT_MANY)}. Trong mô hình của trình khám phá, thiết kế nào xong mỗi công việc trước?
          </>
        }
      >
        Two jobs for the animation's designs, A ({describe(FAST)}) and B ({describe(MANY)}). Job X is{' '}
        {nf(LONG_FEW.chains)} independent chains of {nf(LONG_FEW.length)} operations: {ops(LONG_FEW)} operations in
        all. Job Y is {nf(SHORT_MANY.chains)} independent chains of {nf(SHORT_MANY.length)} operations:{' '}
        {ops(SHORT_MANY)} in all. In the explorer's model, which design finishes each job first?
      </Translated>
    ),
    options: [
      {
        get label() {
          return tr(`Both on A: its units are ${FAST.speed} times faster.`, `Cả hai trên A: các đơn vị của nó nhanh hơn ${FAST.speed} lần.`);
        },
        get why() {
          return tr(
            `For Y, A needs ${nf(shortMany.a.waves)} waves of ${nf(SHORT_MANY.length / FAST.speed)}, done at ${nf(shortMany.a.last)}. B needs ${nf(shortMany.b.waves)} waves of ${nf(SHORT_MANY.length)}, done at ${nf(shortMany.b.last)}.`,
            `Với Y, A cần ${nf(shortMany.a.waves)} đợt, mỗi đợt ${nf(SHORT_MANY.length / FAST.speed)}, xong lúc ${nf(shortMany.a.last)}. B cần ${nf(shortMany.b.waves)} đợt, mỗi đợt ${nf(SHORT_MANY.length)}, xong lúc ${nf(shortMany.b.last)}.`,
          );
        },
      },
      {
        get label() {
          return tr(`Both on B: it has ${MANY.throughput / FAST.throughput} times the peak rate.`, `Cả hai trên B: tốc độ đỉnh của nó gấp ${MANY.throughput / FAST.throughput} lần.`);
        },
        get why() {
          return tr(
            `A peak rate needs every unit busy. For X, B can use only ${nf(LONG_FEW.chains)} of its ${MANY.units} units: one wave of ${nf(LONG_FEW.length)}, done at ${nf(longFew.b.last)}. A runs ${nf(longFew.a.waves)} waves of ${nf(LONG_FEW.length / FAST.speed)} and is done at ${nf(longFew.a.last)}.`,
            `Tốc độ đỉnh cần mọi đơn vị đều bận. Với X, B chỉ dùng được ${nf(LONG_FEW.chains)} trong ${MANY.units} đơn vị của nó: một đợt dài ${nf(LONG_FEW.length)}, xong lúc ${nf(longFew.b.last)}. A chạy ${nf(longFew.a.waves)} đợt, mỗi đợt ${nf(LONG_FEW.length / FAST.speed)}, và xong lúc ${nf(longFew.a.last)}.`,
          );
        },
      },
      {
        get label() {
          return tr('X on A, Y on B: A for the few chains, B for the many.', 'X trên A, Y trên B: A cho ít chuỗi, B cho nhiều chuỗi.');
        },
        correct: true,
      },
      {
        get label() {
          return tr(
            'X on B, Y on A: the bigger job goes to the design with the higher peak rate.',
            'X trên B, Y trên A: công việc lớn hơn giao cho thiết kế có tốc độ đỉnh cao hơn.',
          );
        },
        get why() {
          return tr(
            `That is each job on the design that suits it least: X takes ${nf(longFew.b.last)} on B against ${nf(longFew.a.last)} on A, and Y takes ${nf(shortMany.a.last)} on A against ${nf(shortMany.b.last)} on B.`,
            `Đó là mỗi công việc trên thiết kế ít hợp với nó nhất: X mất ${nf(longFew.b.last)} trên B so với ${nf(longFew.a.last)} trên A, còn Y mất ${nf(shortMany.a.last)} trên A so với ${nf(shortMany.b.last)} trên B.`,
          );
        },
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đếm số phần việc độc lập trước khi đếm phép toán. X có {nf(LONG_FEW.chains)}: nhiều hơn {FAST.units} đơn vị
            của A và ít hơn hẳn {MANY.units} đơn vị của B. A xong nó trong {nf(longFew.a.waves)} đợt, lúc{' '}
            {nf(longFew.a.last)}; B trong một đợt, lúc {nf(longFew.b.last)}. Y có {nf(SHORT_MANY.chains)}: B xong trong{' '}
            {nf(shortMany.b.waves)} đợt, lúc {nf(shortMany.b.last)}, còn A cần {nf(shortMany.a.waves)} đợt, đến{' '}
            {nf(shortMany.a.last)}. Vậy công việc có nhiều phép tính hơn lại thuộc về thiết kế hướng độ trễ. Trên một
            công việc, độ dài chuỗi co giãn thời gian của cả hai thiết kế như nhau: nó quyết định công việc mất bao lâu,
            không quyết định thiết kế nào nên nhận nó. Cả hai công việc đều vừa với trình khám phá ở trên, nếu bạn muốn
            kiểm tra.
          </p>
        }
      >
        <p>
          Count the independent pieces before the operations. X has {nf(LONG_FEW.chains)}: more than A's {FAST.units}{' '}
          units and far fewer than B's {MANY.units}. A finishes it in {nf(longFew.a.waves)} waves, at{' '}
          {nf(longFew.a.last)}; B in one wave, at {nf(longFew.b.last)}. Y has {nf(SHORT_MANY.chains)}: B finishes in{' '}
          {nf(shortMany.b.waves)} waves, at {nf(shortMany.b.last)}, while A needs {nf(shortMany.a.waves)}, until{' '}
          {nf(shortMany.a.last)}. So the job with more arithmetic goes to the latency design. On one job the length of
          a chain scales both designs' times alike: it sets how long the job takes, not which design should take it.
          Both jobs fit the explorer above, if you want to check.
        </p>
      </Translated>
    ),
    source: 'our model of the book\'s argument, ch. 1 §1.1, pp. 4–5',
    vi: { source: 'mô hình của chúng tôi cho lập luận của sách, ch. 1 §1.1, pp. 4–5' },
  },
  'ch01/start/q-tokens': {
    kind: 'choice',
    id: 'ch01/start/q-tokens',
    prompt: (
      <Translated
        vi={
          <>
            Một yêu cầu sinh văn bản từng token một, và tính token t+1 cần token t. Tính độc lập mà một bộ xử lý hướng
            thông lượng cần nằm ở đâu?
          </>
        }
      >
        One request generates text token by token, and computing token t+1 needs token t. Where is the independence
        that a throughput-oriented processor needs?
      </Translated>
    ),
    options: [
      {
        get label() {
          return tr(
            'Across tokens: the GPU works on many future tokens of the request at the same time.',
            'Giữa các token: GPU xử lý cùng lúc nhiều token tương lai của yêu cầu.',
          );
        },
        get why() {
          return tr(
            'Token t+1 needs token t. Along the sequence, the work is one dependent chain.',
            'Token t+1 cần token t. Dọc theo chuỗi token, công việc là một chuỗi phụ thuộc.',
          );
        },
      },
      {
        get label() {
          return tr(
            'Inside each step: every matrix product is thousands of dot products that do not depend on each other.',
            'Bên trong mỗi bước: mỗi phép nhân ma trận là hàng nghìn tích vô hướng không phụ thuộc vào nhau.',
          );
        },
        correct: true,
      },
      {
        get label() {
          return tr(
            'Nowhere in one request: generation is sequential, so a GPU only helps once requests are batched.',
            'Không có trong một yêu cầu: việc sinh là tuần tự, nên GPU chỉ giúp được khi các yêu cầu được gom batch.',
          );
        },
        get why() {
          return tr(
            'The loop over tokens is sequential. What happens inside one turn of the loop is not.',
            'Vòng lặp qua các token là tuần tự. Những gì diễn ra bên trong một vòng lặp thì không.',
          );
        },
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Vòng lặp qua các token là chuỗi. Các phép nhân ma trận bên trong một bước là đám đông: mỗi phần tử đầu ra là
            một tích vô hướng không cần phần tử đầu ra nào khác, đúng như hình ở phần cuối của hoạt hình. Batching thêm
            một nguồn độc lập thứ hai, giữa các yêu cầu. Vì thế một luồng sinh đơn lẻ bị giới hạn bởi chuỗi (bước này
            nối bước kia) và được đám đông bên trong mỗi bước trợ giúp. Cách ánh xạ này là của chúng tôi.
          </p>
        }
      >
        <p>
          The loop over tokens is the chain. The matrix products inside one step are the crowd: each output element is a
          dot product that needs no other output element, which is the picture in the last part of the animation.
          Batching adds a second source of independence, across requests. So a single stream is limited by the chain
          (one step after another) and helped by the crowd inside each step. This mapping is ours.
        </p>
      </Translated>
    ),
    source: 'our analysis, using ch. 1 §1.1, pp. 4–6; the book treats generation in ch. 20',
    vi: { source: 'phân tích của chúng tôi, dựa trên ch. 1 §1.1, pp. 4–6; sách bàn về việc sinh văn bản ở ch. 20' },
  },
  'ch01/start/q-oneunitmore': {
    kind: 'numeric',
    id: 'ch01/start/q-oneunitmore',
    prompt: (
      <Translated
        vi={
          <>
            Thiết kế B có 64 đơn vị tốc độ 1. Công việc là 80 chuỗi độc lập, mỗi chuỗi 100 phép toán. Trong mô hình của
            trình khám phá, bao nhiêu đơn vị thời gian trôi qua trước khi chuỗi cuối cùng xong? Hãy tính ra, rồi đặt trình
            khám phá theo đó.
          </>
        }
      >
        Design B has 64 units of speed 1. The work is 80 independent chains of 100 operations each. In the explorer's
        model, how many time units pass before the last chain finishes? Work it out, then set the explorer to it.
      </Translated>
    ),
    answer: 200,
    unit: 'time units',
    explain: (
      <Translated
        vi={
          <p>
            64 chuỗi bắt đầu cùng lúc và xong lúc 100. 16 chuỗi còn lại bắt đầu khi đó và xong lúc 200: hai đợt. Một đơn
            vị thứ 65 sẽ không thay đổi gì, vì 80 chuỗi vẫn cần hai đợt. Bớt đi 16 chuỗi sẽ giảm thời gian một nửa. Thông
            lượng đến theo từng bậc có kích thước bằng phần cứng. Bạn gặp lại phép làm tròn lên này ở chương 2, nơi nó quyết
            định một lần khởi chạy kernel cần bao nhiêu khối luồng.
          </p>
        }
      >
        <p>
          64 chains start at once and finish at 100. The other 16 start then and finish at 200: two waves. A 65th unit
          would change nothing, because 80 chains would still need two waves. Taking 16 chains away would halve the
          time. Throughput arrives in steps the size of the hardware. You meet the same rounding up in chapter 2, where
          it sets how many blocks of threads a kernel launch needs.
        </p>
      </Translated>
    ),
    source: 'our model of the book\'s argument, ch. 1 §1.1, pp. 4–5',
    vi: { unit: 'đơn vị thời gian', source: 'mô hình của chúng tôi cho lập luận của sách, ch. 1 §1.1, pp. 4–5' },
  },
};
/** The same record under the name the chapter index convention expects. */
export const questionsS10 = questions10;

// ---------------------------------------------------------------- page
export function S10() {
  const crowd = race(FAST, MANY, 1024, 100);
  return (
    <>
      <Translated
        vi={
          <p>
            Chương đầu của sách không có mã và không có bài tập. Nó lập luận rằng lập trình song song đáng học và phác ra phần
            còn lại của sách. Khóa học này giữ lại bốn điều từ chương đó: vì sao hai loại bộ xử lý được xây dựng khác nhau (trang
            này); cách đặt trần cho một mức tăng tốc, và bốn khó khăn mà sách được tổ chức xoay quanh (
            <a href={href('ch01', '1.3')}>mục 1.3</a>); và một bản đồ của sách theo mục tiêu của bạn (
            <a href={href('ch01', '1.7')}>mục 1.7</a>).
          </p>
        }
      >
        <p>
          The book's first chapter has no code and no exercises. It argues that parallel programming is worth learning
          and lays out the rest of the book. This course keeps four things from it: why the two kinds of processor are
          built differently (this page); how to put a ceiling on a speed-up, and the four difficulties the book is
          organised around (<a href={href('ch01', '1.3')}>section 1.3</a>); and a map of the book for your goals (
          <a href={href('ch01', '1.7')}>section 1.7</a>).
        </p>
      </Translated>
      <Source pages="1–20" />
      <Translated
        vi={
          <p>
            Mỗi chương sau đều mở đầu bằng một bài kiểm tra những gì các chương trước đã dạy. Trước chương này không có gì, nên
            ở đây không có bài kiểm tra như vậy. Một câu hỏi từ phía framework thay thế cho nó. Hãy trả lời từ những gì bạn biết
            về việc phục vụ mô hình. Phần còn lại của trang giải thích vì sao câu trả lời cũng là một nhận định về phần cứng.
          </p>
        }
      >
        <p>
          Every later chapter opens with a check on what the chapters before it taught. Nothing comes before this one,
          so there is no such check here. One question from the framework side stands in for it. Answer it from what
          you know about serving models. The rest of the page explains why the answer is also a statement about
          hardware.
        </p>
      </Translated>
      <Ask q={questions10['ch01/start/q-batch']} label={tr('From what you already know', 'Từ những gì bạn đã biết')} />

      <h2><Translated vi="Vì sao có hai loại bộ xử lý">Why there are two kinds of processor</Translated></h2>
      <Translated
        vi={
          <p>
            Cho đến khoảng năm 2003, một chương trình nhanh lên theo mỗi bộ xử lý mới mà không cần sửa gì, vì mỗi bộ xử lý mới
            chạy một dãy lệnh đơn nhanh hơn bộ trước. Mức tiêu thụ năng lượng và nhiệt đã chấm dứt điều đó: chúng giới hạn cả tần
            số xung nhịp lẫn lượng việc vừa trong một chu kỳ xung nhịp. Từ đó, tốc độ cao hơn đến từ việc chạy nhiều dãy lệnh
            (luồng, thread) cùng lúc, và một chương trình chỉ được lợi nếu công việc của nó được chia theo cách đó. Hai hướng
            thiết kế ra đời. Bộ xử lý <b>đa lõi</b> (multi-core) giữ mỗi luồng chạy nhanh và thêm lõi. Bộ xử lý{' '}
            <b>nhiều luồng</b> (many-thread), tức GPU, hy sinh tốc độ mỗi luồng để tối đa hóa tổng lượng việc do một số rất lớn
            luồng thực hiện.
          </p>
        }
      >
        <p>
          Until about 2003 a program got faster with each new processor without being touched, because each new
          processor ran a single sequence of instructions faster than the last. Energy use and heat ended that: they
          capped both the clock rate and the work that fits in one clock period. Since then, more speed has come from
          running several instruction sequences (threads) at the same time, and a program benefits only if its work is
          split that way. Two lines of design followed. <b>Multi-core</b> processors keep each thread fast and add
          cores. <b>Many-thread</b> processors, the GPUs, give up speed per thread to maximise the total work done by a
          very large number of them.
        </p>
      </Translated>
      <Source pages="1–3" />
      <Translated
        vi={
          <p>
            Ví dụ của sách cho loại thứ hai là Hopper H100 của NVIDIA, với hàng trăm nghìn luồng trên nhiều pipeline đơn giản.
            Sách dẫn tốc độ đỉnh, tính đến năm 2022, là 34, 67 và 1,979 TFLOPS cho số học 64, 32 và 16 bit (một TFLOPS là
            10<sup>12</sup> phép toán dấu phẩy động mỗi giây), so với "vài TFLOPS" của một CPU máy chủ cùng năm. Sách tự thêm
            một lưu ý: đây là tốc độ đỉnh của phần cứng, không phải tốc độ của bất kỳ ứng dụng nào.
          </p>
        }
      >
        <p>
          The book's example of the second kind is NVIDIA's Hopper H100, with hundreds of thousands of threads on many
          simple pipelines. It quotes peak rates, as of 2022, of 34, 67 and
          1,979 TFLOPS for 64-, 32- and 16-bit arithmetic (one TFLOPS is 10<sup>12</sup> floating-point operations per
          second), against "a few TFLOPS" for a server CPU of the same year. It adds its own caveat: these are peak
          rates of the hardware, not the speed of any application.
        </p>
      </Translated>
      <Source section="1.1" pages="3" />
      <Ask q={questions10['ch01/start/q-sixteenbit']} />
      <Translated
        vi={
          <p>
            Hãy đọc mọi con số kiểu này với cả hai điểm trong đầu. Tốc độ đỉnh là những gì các đơn vị số học có thể làm nếu
            chúng không bao giờ phải chờ thứ gì, và những con số lớn nhất trên bảng thông số thuộc về các đơn vị ma trận. Một
            kernel đến gần điểm thứ nhất đến mức nào là chủ đề của chương 5. Cần gì để dùng được điểm thứ hai thì ở chương 15.
          </p>
        }
      >
        <p>
          Read every figure of this kind with both points in mind. A peak rate is what the arithmetic units could do if
          they never had to wait for anything, and the largest figures on a specification sheet belong to the matrix
          units. How close a kernel gets to the first is the subject of chapter 5. What it takes to use the second is
          in chapter 15.
        </p>
      </Translated>
      <Today checked="2026-11-20">
        <Translated
          vi={
            <p>
              Trang H100 của NVIDIA vẫn in cả ba con số, cho biến thể SXM. 34 và 67 là các hàng FP64 và FP32 thông thường. 1,979
              là hàng "FP16 Tensor Core", có chú thích "With sparsity".
            </p>
          }
        >
          <p>
            NVIDIA's H100 page still prints all three figures, for the SXM variant. 34 and 67 are its plain FP64 and
            FP32 rows. 1,979 is the row "FP16 Tensor Core", footnoted "With sparsity".
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Diện tích chip đi về đâu">Where the chip area goes</Translated></h2>
      <Translated
        vi={
          <p>
            Khoảng cách đến từ việc mỗi thiết kế được yêu cầu giỏi điều gì. CPU là một thiết kế <b>hướng độ trễ</b>: nó được xây
            dựng để mỗi phép toán của một luồng đơn xong càng sớm càng tốt. Điều đó cần các đơn vị số học được thiết kế cho độ
            trễ ngắn, bộ nhớ đệm (cache) lớn biến các truy cập bộ nhớ chậm thành nhanh, và logic điều khiển có dự đoán rẽ nhánh
            để giữ một dòng lệnh luôn chạy. Tất cả đều dùng diện tích chip và công suất mà lẽ ra có thể dành cho thêm đơn vị số
            học và thêm kênh tới bộ nhớ.
          </p>
        }
      >
        <p>
          The gap comes from what each design is asked to be good at. A CPU is a <b>latency-oriented</b> design: it is
          built so that each operation of a single thread finishes as soon as possible. That takes arithmetic units
          designed for short latency, large caches that turn slow memory accesses into fast ones, and control logic
          with branch prediction to keep one stream of instructions moving. All of it uses chip area and power that
          could have gone to more arithmetic units and more channels to memory.
        </p>
      </Translated>
      <Source section="1.1" pages="4–5" figs="Fig. 1.1" />
      <AreaFigure />
      <Translated
        vi={
          <p>
            GPU là một thiết kế <b>hướng thông lượng</b>. Hình hài của nó đến từ trò chơi điện tử, vốn cần một số rất lớn phép
            tính dấu phẩy động và truy cập bộ nhớ cho mỗi khung hình. Vì thế diện tích và công suất dành cho nhiều đơn vị số học
            đơn giản và nhiều kênh bộ nhớ, với ít logic điều khiển và bộ nhớ đệm nhỏ, và mỗi phép toán đơn lẻ được phép mất
            nhiều thời gian.
          </p>
        }
      >
        <p>
          A GPU is a <b>throughput-oriented</b> design. Its shape comes from video games, which need a very large number
          of floating-point calculations and memory accesses for every frame. So the area and the power go to many
          simple arithmetic units and many memory channels, with little control logic and small caches, and each single
          operation is allowed to take a long time.
        </p>
      </Translated>
      <Today checked="2026-11-20">
        <Translated
          vi={
            <p>
              Hướng dẫn lập trình CUDA hiện hành của NVIDIA nêu đúng sự phân chia này: GPU đổi "lower single-thread performance"
              lấy "much greater total throughput", và GPU "devote more transistors to data processing units".
            </p>
          }
        >
          <p>
            NVIDIA's current CUDA programming guide states the same split: a GPU trades "lower single-thread
            performance" for "much greater total throughput", and GPUs "devote more transistors to data processing
            units".
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Sao không xây một chip giỏi cả hai? Câu trả lời của sách là một quy tắc chi phí. Để gấp đôi thông lượng, hãy gấp đôi
            số đơn vị: gấp đôi diện tích và gấp đôi công suất. Để giảm một nửa độ trễ của một đơn vị, bạn có thể cần gấp đôi
            dòng điện, hơn gấp đôi diện tích và gấp bốn lần công suất. Độ trễ đắt hơn thông lượng rất nhiều, nên một thiết kế
            phải chọn. Hãy coi quy tắc này là lập luận của sách: không tài liệu nào của nhà sản xuất nêu nó, và nó là unverified.
            Hoạt hình và trình khám phá bên dưới lấy đúng quy tắc đó và đi theo nơi nó dẫn tới.
          </p>
        }
      >
        <p>
          Why not build one chip that is good at both? The book's answer is a cost rule. To double the throughput,
          double the number of units: twice the area and twice the power. To halve the latency of a unit, you may need
          twice the current, more than twice the area and four times the power. Latency is far more expensive to buy
          than throughput, so a design has to choose. Treat this rule as the book's argument: no vendor document states
          it, and it is unverified. The animation and the explorer below take that one rule and follow where it leads.
        </p>
      </Translated>
      <Source section="1.1" pages="5" />

      <h2><Translated vi="Một ngân sách, hai thiết kế">One budget, two designs</Translated></h2>
      <Translated
        vi={
          <p>
            Trước hết là hai thuật ngữ. <b>Độ trễ</b> là thời gian một phần việc mất từ đầu đến cuối. <b>Thông lượng</b> là lượng
            việc hoàn thành trên mỗi đơn vị thời gian, tính trên mọi phần việc. Hoạt hình cho hai thiết kế cùng một ngân sách
            công suất và cùng một công việc, rồi chạy chúng theo cùng một đồng hồ. Hãy xem nó một lần trước khi dùng các nút
            điều khiển bên dưới.
          </p>
        }
      >
        <p>
          Two terms first. <b>Latency</b> is how long one piece of work takes from start to finish.{' '}
          <b>Throughput</b> is how much work is finished per unit of time, over all the pieces. The animation gives two
          designs the same power budget and the same work, and runs them against one clock. Watch it once before you
          use the controls further down.
        </p>
      </Translated>
      <TwoDesignsScene />
      <Predict
        id="ch01/start/p-crowd"
        prompt={
          <Translated
            vi={
              <>
                Thiết kế A có {FAST.units} đơn vị, mỗi đơn vị nhanh hơn đơn vị thường {FAST.speed} lần. Thiết kế B có{' '}
                {MANY.units} đơn vị thường. Theo quy tắc của sách, cả hai dùng cùng công suất. Công việc là 1,024 chuỗi độc
                lập có độ dài bằng nhau. B xong sớm hơn bao nhiêu lần?
              </>
            }
          >
            Design A has {FAST.units} units, each {FAST.speed} times faster than a plain one. Design B has {MANY.units}{' '}
            plain units. By the book's rule both draw the same power. The work is 1,024 independent chains of equal
            length. How many times sooner does B finish?
          </Translated>
        }
        unit={tr('times sooner', 'lần sớm hơn')}
      >
        {(v) => (
          <Translated
            vi={
              <p>
                <b>{nf(crowd.factor)} lần.</b> B chạy {MANY.units} chuỗi cùng lúc, nên 1,024 chuỗi cần {nf(crowd.b.waves)} đợt. A
                chạy {FAST.units} chuỗi cùng lúc và cần {nf(crowd.a.waves)} đợt, mỗi đợt ngắn hơn {FAST.speed} lần, tức bằng thời
                gian của {nf(crowd.a.waves / FAST.speed)} đợt của B. {nf(crowd.a.waves / FAST.speed)} so với {nf(crowd.b.waves)} là{' '}
                {nf(crowd.factor)}.{' '}
                {v === crowd.factor
                  ? 'Đúng như bạn đã dự đoán.'
                  : v === MANY.units / FAST.units
                    ? `${nf(v)} là tỷ lệ số đơn vị, hay tỷ lệ số đợt. Nó quên rằng mỗi đơn vị của A nhanh hơn ${FAST.speed} lần.`
                    : v > 0 && v < 1
                      ? 'Một con số dưới 1 nghiêng về các đơn vị nhanh hơn. Chúng chỉ thắng khi có quá ít chuỗi để giữ các đơn vị của B bận rộn.'
                      : v === MANY.units
                        ? `${nf(v)} chỉ đếm các đơn vị của B và bỏ A ra khỏi phép so sánh.`
                        : 'Hãy đếm số đợt mỗi thiết kế cần, rồi một đợt mất bao lâu.'}{' '}
                Đặt trình khám phá bên dưới về 1,024 chuỗi và kiểm tra. Rồi đưa số chuỗi về lại 1 và xem hệ số đảo chiều.
              </p>
            }
          >
            <p>
              <b>{nf(crowd.factor)} times.</b> B runs {MANY.units} chains at a time, so 1,024 chains take{' '}
              {nf(crowd.b.waves)} waves. A runs {FAST.units} at a time and needs {nf(crowd.a.waves)} waves, each{' '}
              {FAST.speed} times shorter, which is the time of {nf(crowd.a.waves / FAST.speed)} of B's waves.{' '}
              {nf(crowd.a.waves / FAST.speed)} against {nf(crowd.b.waves)} is {nf(crowd.factor)}.{' '}
              {v === crowd.factor
                ? 'That is what you had.'
                : v === MANY.units / FAST.units
                  ? `${nf(v)} is the ratio of the unit counts, or of the wave counts. It forgets that each of A's units is ${FAST.speed} times faster.`
                  : v > 0 && v < 1
                    ? 'A number below 1 backs the faster units. They win only while there are too few chains to keep B\'s units busy.'
                    : v === MANY.units
                      ? `${nf(v)} counts B's units and leaves A out of the comparison.`
                      : 'Count the waves each design needs, then how long one wave takes.'}{' '}
              Set the explorer below to 1,024 chains and check it. Then bring the chains back down to 1 and watch the
              factor turn round.
            </p>
          </Translated>
        )}
      </Predict>
      <BudgetBench />
      <Translated
        vi={
          <p>
            Ba điều rút ra từ đó. Trên một ngân sách, hai thiết kế chênh nhau cùng một hệ số theo hai hướng ngược nhau: các đơn
            vị nhanh thắng một chuỗi đơn đúng bằng hệ số mà nhiều đơn vị thắng một đám đông. Độ dài chuỗi triệt tiêu, nên nó
            không bao giờ thay đổi bên thắng hay hệ số. Điều quyết định bên thắng là bao nhiêu phần của công việc độc lập với
            nhau. Và độ trễ mua bằng một ngân sách cố định phải trả bằng tốc độ đỉnh: mỗi lần giảm một nửa đưa tốc độ khi mọi
            đơn vị đều bận từ {HALVINGS.map((h) => design(BUDGET, h).throughput).join(' xuống ')}.
          </p>
        }
      >
        <p>
          Three things to take from it. On one budget the two designs differ by the same factor in opposite directions:
          the fast units win a single chain by exactly the factor by which the many units win a crowd. The length of a
          chain cancels out, so it never changes the winner or the factor. What decides the winner is how many pieces
          of the work are independent of each other. And latency bought on a fixed budget is paid for in peak rate:
          each halving takes the rate with every unit busy from {HALVINGS.map((h) => design(BUDGET, h).throughput).join(' to ')}.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một bước, một nửa độ trễ đổi lấy gấp bốn công suất, là quy tắc của sách (p. 5). Lặp lại nó là phép ngoại suy của
              chúng tôi; mạch thật không tuân theo một số mũ duy nhất.
            </p>
          }
        >
          <p>
            One step, half the latency for four times the power, is the book's rule (p. 5). Repeating it is our
            extrapolation; real circuits do not follow one exponent.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Bị bỏ qua: chi phí diện tích, bộ nhớ đệm, kênh bộ nhớ, pipelining. Một đơn vị chạy một chuỗi mỗi lúc; một GPU thật
              giữ thường trú nhiều luồng hơn số đơn vị rất nhiều (chương 4). Ngân sách {BUDGET} là một con số trừu tượng, không
              phải một thiết bị.
            </p>
          }
        >
          <p>
            Left out: the area cost, caches, memory channels, pipelining. A unit runs one chain at a time; a real GPU
            keeps many more threads resident than units (chapter 4). The budget of {BUDGET} is an abstract number, not
            a device.
          </p>
        </Translated>
      </Simplification>
      <Translated
        vi={
          <p>
            Hãy nhìn bảng thêm một lần trước khi rời nó. Tốc độ đỉnh của một thiết kế không bao giờ đổi, dù bạn giao cho nó
            công việc gì. Tốc độ đạt được thì đổi, và với một chuỗi đơn nó chỉ là một phần nhỏ của tốc độ đỉnh. Đó chính là lưu
            ý của sách về các con số của H100, trong một mô hình đủ nhỏ để kiểm tra bằng tay.
          </p>
        }
      >
        <p>
          Look at the table once more before you leave it. The peak rate of a design never moves, whatever work you
          give it. The rate delivered does, and with a single chain it is a small fraction of the peak. That is the
          book's caveat about the H100 figures, in a model small enough to check by hand.
        </p>
      </Translated>
      <Ask q={questions10['ch01/start/q-chainlen']} />
      <Ask q={questions10['ch01/start/q-oneunitmore']} />

      <h2><Translated vi="Thiết kế này đòi hỏi gì ở mã của bạn">What the design asks of your code</Translated></h2>
      <Translated
        vi={
          <p>
            Trong trình khám phá, một đơn vị ngồi không mỗi khi số chuỗi ít hơn số đơn vị. Trên một GPU thật còn có lý do thứ hai
            khiến một đơn vị không có việc. Thiết kế cho phép truy cập bộ nhớ và phép tính số học mất nhiều thời gian, và một
            luồng đang chờ một trong hai thứ đó thì không có gì để chạy. Sách rút ra kết luận cho phần mềm: mã GPU được kỳ vọng
            cung cấp một số rất lớn luồng, để phần cứng luôn có vài luồng sẵn sàng trong khi những luồng khác chờ.
          </p>
        }
      >
        <p>
          In the explorer a unit sits idle whenever there are fewer chains than units. On a real GPU there is a second
          reason for a unit to have nothing to do. The design lets memory accesses and arithmetic take a long time, and
          a thread that is waiting for one of them has nothing to run. The book draws the conclusion for software: GPU
          code is expected to supply a very large number of threads, so that the hardware always has some that are
          ready while others wait.
        </p>
      </Translated>
      <Source section="1.1" pages="5" />
      <Translated
        vi={
          <p>
            Chương 4 trình bày cơ chế (<a href={href('ch04', '4.6')}>§4.6, lập lịch warp và khả năng chịu độ trễ</a>, pp.
            82–85): phần cứng giữ thường trú nhiều luồng hơn nhiều so với số nó có thể thực thi trong một khoảnh khắc, và ở mỗi
            bước nó chọn một luồng không phải đang chờ. Hàng trăm nghìn luồng mà sách gán cho H100 là luồng kiểu này: thường
            trú, với số luồng đang thực thi ở bất kỳ thời điểm nào ít hơn nhiều.
          </p>
        }
      >
        <p>
          Chapter 4 shows the mechanism (<a href={href('ch04', '4.6')}>§4.6, warp scheduling and latency tolerance</a>,
          pp. 82–85): the hardware keeps many more threads resident than it can
          execute at one instant, and at each step it picks one that is not waiting. The hundreds of thousands of
          threads the book gives the H100 are threads of this kind: resident, with far fewer executing at any moment.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch01-facts.md, A2)">{/* i18n-ignore: date and notes reference */}
        <Translated
          vi={
            <p>
              Nhân hai con số H100 của chính NVIDIA với nhau cho ra số luồng thường trú cỡ đó. Phép nhân là của chúng tôi.
            </p>
          }
        >
          <p>
            Two of NVIDIA's own H100 figures, multiplied together, give a count of that size for threads held resident.
            The multiplication is ours.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Không thiết kế nào thay thế được thiết kế kia. Các phần tuần tự của chương trình ở lại trên CPU và các phần nặng về
            tính toán số đi sang GPU, đó là lý do mô hình lập trình bạn gặp ở chương 2 có host và device (thiết bị).
          </p>
        }
      >
        <p>
          Neither design replaces the other. The sequential parts of a program stay on the CPU and the numerically
          heavy parts go to the GPU, which is why the programming model you meet in chapter 2 has a host and a device.
        </p>
      </Translated>
      <Source section="1.1 and 1.3" pages="5–6, 10" />

      <h2><Translated vi="Sự đánh đổi bạn vẫn đang làm">The trade you already make</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn thực hiện sự đánh đổi này mỗi lần chọn kích thước batch. Với batch 1, một yêu cầu không chờ gì ngoài phép tính
            của chính nó, đó là thời gian chờ ngắn nhất một yêu cầu có thể có. Với batch lớn, mỗi yêu cầu lâu hơn, còn GPU hoàn
            thành tổng số token mỗi giây nhiều hơn. Phục vụ với batch 1 là bắt một bộ xử lý hướng thông lượng cư xử như một bộ
            xử lý hướng độ trễ. Chính sách cũng vẽ ra sự tương đồng này: mục về batching của nó,{' '}
            <a href={href('ch22', '22.5')}>§22.5</a>, quay về hai thiết kế của chương này.
          </p>
        }
      >
        <p>
          You make this trade every time you choose a batch size. At batch size 1 a request waits for nothing except
          its own arithmetic, which is the shortest wait one request can have. With a large batch every request takes
          longer, and the GPU finishes more tokens per second in total. Serving at batch size 1 asks a
          throughput-oriented processor to behave like a latency-oriented one. The book draws the same parallel
          itself: its section on batching, <a href={href('ch22', '22.5')}>§22.5</a>, points back to the two designs of
          this chapter.
        </p>
      </Translated>
      <Source section="22.5" pages="539–540" />
      <Ask q={questions10['ch01/start/q-tokens']} />

      <h2><Translated vi="Các backend khác nằm ở đâu">Where the other backends fit</Translated></h2>
      <Translated
        vi={
          <p>
            Sách đưa ra lập luận này với phần cứng NVIDIA, và không chỗ nào trong chương nhắc đến HIP hay Metal. Theo cách chúng
            tôi đọc tài liệu của chính các nhà sản xuất, sự phân chia giữa hai thiết kế không chỉ riêng NVIDIA. Nó áp dụng cho
            hai backend còn lại của khóa học này: GPU AMD, lập trình qua HIP, và GPU Apple, lập trình qua Metal.
          </p>
        }
      >
        <p>
          The book makes this argument with NVIDIA hardware, and nowhere in the chapter mentions HIP or Metal. By our
          reading of the vendors' own documents, the split between the two designs is not NVIDIA's alone. It carries
          over to the other two backends of this course: AMD GPUs, programmed through HIP, and Apple GPUs, programmed
          through Metal.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              HIP là mô hình lập trình của CUDA trên GPU AMD. AMD viết tài liệu cho nó trên GPU AMD và nói nó "not intended to be a
              drop-in replacement" cho CUDA. Metal là API đồ họa và tính toán của Apple, với kernel viết bằng Metal Shading
              Language (MSL): một hàm tính toán "also known as a compute kernel" và chạy trên "a grid of threads".
            </p>
          }
        >
          <p>
            HIP is CUDA's programming model on AMD GPUs. AMD documents it for AMD GPUs and says it is "not intended to
            be a drop-in replacement" for CUDA. Metal is Apple's graphics and compute API, with kernels written in the
            Metal Shading Language (MSL): a compute function is "also known as a compute kernel" and runs on "a grid of
            threads".
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Một khác biệt cần nhớ. Hình 1.1 vẽ GPU có DRAM riêng. Điều đó đúng với thiết bị CUDA và HIP. Nó không đúng với Apple
              silicon, nơi CPU và GPU dùng chung bộ nhớ hệ thống. Chương 2 bàn về nó trong{' '}
              <a href={href('ch02', '2.4')}>§2.4, bộ nhớ thiết bị và truyền dữ liệu</a>.
            </p>
          }
        >
          <p>
            One difference to hold on to. Fig. 1.1 draws the GPU with its own DRAM. That fits CUDA and HIP devices. It
            does not fit Apple silicon, where CPU and GPU share system memory. Chapter 2 takes it up in{' '}
            <a href={href('ch02', '2.4')}>§2.4, device memory and data transfer</a>.
          </p>
        </Translated>
        <p className="muted"><Translated vi="Kiểm tra ngày 2026-11-20.">Checked 2026-11-20.</Translated></p>
      </Beyond>
    </>
  );
}
