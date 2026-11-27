import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { design, finish, floorplan, race, regions, snapshot, sweep, timeline, type Design } from '../../sim/designs';

// "One budget, two designs": the book's cost argument for Fig. 1.1 (p. 5) run as a
// race. Five beats, one narration clip each (content/ch01/narration/two-designs-N.txt).
// Every number drawn comes from sim/designs.ts; the only inputs are the ones below.
export const BUDGET = 64;
const A = design(BUDGET, 2); // the fast units, drawn on the left
const B = design(BUDGET, 0); // the many units, drawn on the right
const CHAIN = 16; // operations in a chain, beats 2 to 4
const CROWD = 64; // independent chains, beats 3 and 4
const MAT = 8; // beat 5: an 8 by 8 matrix product, 64 chains of 8
const W = 720;
const H = 348;
const LX = 14; // left panel origin
const RX = 374; // right panel origin

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
const plural = (n: number, one: string) => `${n} ${one}${n === 1 ? '' : 's'}`;
const name = (d: Design) => tr(`${plural(d.units, 'unit')}, speed ${d.speed}`, `${d.units} đơn vị, tốc độ ${d.speed}`);

/**
 * A design as a floor plan: the budget as equal squares of power, grouped into
 * units. A unit's side is its speed and its area is its power.
 */
export function Plan({ x, y, size, d, busy, grid = 1, units = 1, dash = d.unitPower > 1 }: {
  x: number; y: number; size: number; d: Design; busy?: Set<number>; grid?: number; units?: number; dash?: boolean;
}) {
  const p = floorplan(d);
  const c = size / p.side;
  const squares: ReactNode[] = [];
  for (let i = 0; i < d.budget; i++)
    squares.push(
      <rect key={i} x={x + (i % p.side) * c + 0.5} y={y + Math.floor(i / p.side) * c + 0.5} width={c - 1} height={c - 1}
        fill="none" stroke="var(--ink-3)" strokeWidth="0.6" strokeDasharray={dash ? '2 2' : undefined} opacity={grid} />,
    );
  return (
    <g>
      <g opacity={units}>
        {p.cells.map((u) => (
          <rect key={u.unit} x={x + u.x * c + 1} y={y + u.y * c + 1} width={u.side * c - 2} height={u.side * c - 2} rx="1.5"
            fill={busy?.has(u.unit) ? 'var(--ink-2)' : 'var(--cell)'} stroke="var(--ink)" strokeWidth={d.unitPower > 1 ? 1.6 : 1} />
        ))}
      </g>
      {d.unitPower > 1 || units < 1 ? squares : null}
    </g>
  );
}

/** Units that have a chain at time t. */
const busyAt = (d: Design, chains: number, length: number, t: number) =>
  new Set(timeline(d, chains, length).filter((s) => t >= s.start && t < s.end).map((s) => s.unit));

/** One chain as linked boxes: the first `done` are finished. */
function Chain({ x, y, n, done, w }: { x: number; y: number; n: number; done: number; w: number }) {
  const cw = w / n;
  return (
    <g>
      <line x1={x + cw / 2} x2={x + w - cw / 2} y1={y + 8} y2={y + 8} stroke="var(--ink-3)" strokeWidth="1.2" />
      {Array.from({ length: n }, (_, i) => (
        <rect key={i} x={x + i * cw + 1.5} y={y} width={cw - 3} height={16} rx="2"
          fill={i < done ? 'var(--ink)' : 'var(--cell)'} stroke="var(--ink)" strokeWidth="1" />
      ))}
    </g>
  );
}

/** Many chains as a grid of cells: empty, partly done, finished. */
function Chains({ x, y, size, ops, length, cols }: { x: number; y: number; size: number; ops: number[]; length: number; cols: number }) {
  const c = size / cols;
  return (
    <g>
      {ops.map((o, i) => (
        <g key={i}>
          <rect x={x + (i % cols) * c + 1} y={y + Math.floor(i / cols) * c + 1} width={c - 2} height={c - 2} rx="1.5" fill="var(--cell)" stroke="var(--rule)" />
          {o > 0 ? (
            <rect x={x + (i % cols) * c + 1} y={y + Math.floor(i / cols) * c + 1} width={c - 2} height={c - 2} rx="1.5"
              fill={o >= length ? 'var(--ink)' : 'var(--ink-2)'} opacity={o >= length ? 1 : 0.25 + (0.6 * o) / length} />
          ) : null}
        </g>
      ))}
    </g>
  );
}

/** A time axis from 0 to tMax with the elapsed part filled and a mark where this design finished. */
function Clock({ x, y, w, t, tMax, end }: { x: number; y: number; w: number; t: number; tMax: number; end: number }) {
  const X = (v: number) => x + (v / tMax) * w;
  return (
    <g>
      <rect x={x} y={y} width={w} height={8} fill="var(--cell)" stroke="var(--rule)" />
      <rect x={x} y={y} width={X(Math.min(t, end)) - x} height={8} fill="var(--ink-2)" />
      <line x1={X(end)} x2={X(end)} y1={y - 4} y2={y + 12} stroke="var(--ink)" strokeWidth="1.6" opacity={t >= end ? 1 : 0.25} />
      <text x={x} y={y + 24} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>0</text>
      <text x={x + w} y={y + 24} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{tMax}</text>
      {t >= end ? (
        <text x={Math.min(X(end), x + w - 40)} y={y - 8} fontSize="11.5" fontWeight="600" textAnchor={end >= tMax ? 'end' : 'start'}
          transform={end >= tMax ? `translate(40 0)` : undefined}><Translated vi={<>xong lúc {end}</>}>done at {end}</Translated></text>
      ) : null}
    </g>
  );
}

function Stage({ children, label }: { children: ReactNode; label: string }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} style={{ width: '100%', maxWidth: 900, display: 'block', margin: '0 auto' }}>
      {children}
    </svg>
  );
}
const Divider = () => <line x1={W / 2} x2={W / 2} y1={8} y2={292} stroke="var(--rule)" />;
const Title = ({ x, children }: { x: number; children: ReactNode }) => (
  <text x={x} y={18} fontSize="13" fontWeight="600">{children}</text>
);
const Headline = ({ show, children }: { show: number; children: ReactNode }) => (
  <text x={LX} y={336} fontSize="13.5" fontWeight="600" opacity={show}>{children}</text>
);
const Now = ({ t }: { t: number }) => (
  <text x={W - LX} y={336} fontSize="13" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>thời gian {Math.floor(t)}</>}>time {Math.floor(t)}</Translated></text>
);

// ---------------------------------------------------------------- beat 1
function Budget(p: number) {
  const rule = ramp(p, 0.16, 0.28); // the rule appears as it is spoken
  const split = ramp(p, 0.42, 0.56); // one grid becomes two
  const group = ramp(p, 0.56, 0.72); // squares are grouped into units
  const size = 126;
  const cx = (W - size) / 2;
  const lx = lerp(cx, LX + 30, split);
  const rx = lerp(cx, RX + 30, split);
  const steps = [0, 1, 2, 3].map((h) => design(BUDGET, h));
  const gain = steps[1].speed / steps[0].speed;
  const cost = steps[1].unitPower / steps[0].unitPower;
  return (
    <Stage label={tr('One power budget grouped into units in two ways', 'Một ngân sách công suất được gom thành các đơn vị theo hai cách')}>
      {split < 1 ? <text x={W / 2} y={18} fontSize="13" fontWeight="600" textAnchor="middle" opacity={1 - split}><Translated vi={<>một ngân sách {BUDGET} ô công suất</>}>a budget of {BUDGET} squares of power</Translated></text> : null}
      <Plan x={lx} y={30} size={size} d={A} grid={1} units={group} dash={group > 0.5} />
      <Plan x={rx} y={30} size={size} d={B} grid={1 - group} units={group} />
      <g opacity={group}>
        <text x={LX + 174} y={62} fontSize="13" fontWeight="600">{tr(plural(A.units, 'unit'), `${A.units} đơn vị`)}</text>
        <text x={LX + 174} y={82} fontSize="12"><Translated vi={<>mỗi đơn vị {A.unitPower} ô</>}>{A.unitPower} squares each</Translated></text>
        <text x={LX + 174} y={100} fontSize="12"><Translated vi={<>tốc độ {A.speed}</>}>speed {A.speed}</Translated></text>
        <text x={RX + 174} y={62} fontSize="13" fontWeight="600">{tr(plural(B.units, 'unit'), `${B.units} đơn vị`)}</text>
        <text x={RX + 174} y={82} fontSize="12"><Translated vi={<>mỗi đơn vị {B.unitPower} ô</>}>{plural(B.unitPower, 'square')} each</Translated></text>
        <text x={RX + 174} y={100} fontSize="12"><Translated vi={<>tốc độ {B.speed}</>}>speed {B.speed}</Translated></text>
      </g>
      <g opacity={rule}>
        <line x1={LX} x2={W - LX} y1={182} y2={182} stroke="var(--rule)" />
        <text x={LX} y={204} fontSize="12.5" fontWeight="600">
          <Translated vi={<>Quy tắc của sách: tốc độ một đơn vị gấp {gain} lần thì tốn công suất gấp {cost} lần</>}>
            The book's rule: {gain} times the speed of a unit costs {cost} times the power
          </Translated>
        </text>
        {steps.map((d, i) => {
          const s = d.speed * 8;
          const x = LX + i * 176;
          return (
            <g key={i}>
              <rect x={x} y={296 - s} width={s} height={s} rx="1.5" fill="var(--cell)" stroke="var(--ink)" strokeWidth="1.3" />
              <text x={x + s + 8} y={280} fontSize="11.5"><Translated vi={<>tốc độ {d.speed}, công suất {d.unitPower}</>}>speed {d.speed}, power {d.unitPower}</Translated></text>
              <text x={x + s + 8} y={295} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>ngân sách mua được {d.units}</>}>the budget buys {d.units}</Translated></text>
            </g>
          );
        })}
      </g>
      <Headline show={ramp(p, 0.78, 0.88)}>
        <Translated vi={<>một ngân sách: {A.units} đơn vị tốc độ {A.speed}, hoặc {B.units} đơn vị tốc độ {B.speed}</>}>
          one budget: {plural(A.units, 'unit')} of speed {A.speed}, or {plural(B.units, 'unit')} of speed {B.speed}
        </Translated>
      </Headline>
    </Stage>
  );
}

// ---------------------------------------------------------------- beat 2
function OneChain(p: number) {
  const r = race(A, B, 1, CHAIN);
  const tMax = Math.max(r.a.last, r.b.last);
  const t = ramp(p, 0.14, 0.82) * tMax;
  const side = (d: Design, x: number, f: typeof r.a) => {
    const s = snapshot(d, 1, CHAIN, t);
    return (
      <g>
        <Title x={x}>{name(d)}</Title>
        <Plan x={x} y={30} size={112} d={d} busy={busyAt(d, 1, CHAIN, t)} />
        <text x={x + 130} y={62} fontSize="12"><Translated vi={<>đang làm việc: {s.running} đơn vị</>}>working: {plural(s.running, 'unit')}</Translated></text>
        <text x={x + 130} y={80} fontSize="12"><Translated vi={<>không bao giờ dùng đến: {f.idle} đơn vị</>}>never used: {plural(f.idle, 'unit')}</Translated></text>
        <text x={x} y={176} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>
          <Translated vi={<>một chuỗi {CHAIN} phép toán phụ thuộc: xong {s.ops[0]}</>}>
            a chain of {CHAIN} dependent operations: {s.ops[0]} done
          </Translated>
        </text>
        <Chain x={x} y={186} n={CHAIN} done={s.ops[0]} w={320} />
        <Clock x={x} y={248} w={320} t={t} tMax={tMax} end={f.last} />
      </g>
    );
  };
  return (
    <Stage label={tr('One chain of dependent operations run on both designs', 'Một chuỗi phép toán phụ thuộc chạy trên cả hai thiết kế')}>
      {side(A, LX, r.a)}
      {side(B, RX, r.b)}
      <Divider />
      <Headline show={ramp(p, 0.84, 0.92)}>
        <Translated vi={<>một chuỗi: đơn vị nhanh xong sớm hơn {r.factor} lần</>}>
          one chain: the fast unit finishes {r.factor} times sooner
        </Translated>
      </Headline>
      <Now t={t} />
    </Stage>
  );
}

// ---------------------------------------------------------------- beat 3
function Crowd(p: number) {
  const r = race(A, B, CROWD, CHAIN);
  const tMax = Math.max(r.a.last, r.b.last);
  const t = ramp(p, 0.1, 0.86) * tMax;
  const side = (d: Design, x: number, f: typeof r.a) => {
    const s = snapshot(d, CROWD, CHAIN, t);
    return (
      <g>
        <Title x={x}>{name(d)}</Title>
        <text x={x} y={36} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>các đơn vị ({s.running} đơn vị đang làm việc)</>}>units ({plural(s.running, 'unit')} working)</Translated></text>
        <Plan x={x} y={42} size={112} d={d} busy={busyAt(d, CROWD, CHAIN, t)} />
        <text x={x + 150} y={36} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>{CROWD} chuỗi dài {CHAIN}</>}>{CROWD} chains of {CHAIN}</Translated></text>
        <Chains x={x + 150} y={42} size={112} ops={s.ops} length={CHAIN} cols={8} />
        <text x={x} y={180} fontSize="12"><Translated vi={<>chuỗi đã xong: {s.done} trên {CROWD}, trong {f.waves} đợt</>}>chains finished: {s.done} of {CROWD}, in {plural(f.waves, 'wave')}</Translated></text>
        <text x={x} y={198} fontSize="12" opacity={t >= f.first ? 1 : 0.25}><Translated vi={<>kết quả đầu tiên lúc {f.first}</>}>first result at {f.first}</Translated></text>
        <text x={x + 150} y={198} fontSize="12" opacity={t >= f.last ? 1 : 0.25}><Translated vi={<>kết quả cuối cùng lúc {f.last}</>}>last result at {f.last}</Translated></text>
        <Clock x={x} y={248} w={320} t={t} tMax={tMax} end={f.last} />
      </g>
    );
  };
  return (
    <Stage label={tr('Sixty-four independent chains run on both designs', 'Sáu mươi bốn chuỗi độc lập chạy trên cả hai thiết kế')}>
      {side(A, LX, r.a)}
      {side(B, RX, r.b)}
      <Divider />
      <Headline show={ramp(p, 0.88, 0.95)}>
        <Translated vi={<>{CROWD} chuỗi: nhiều đơn vị xong sớm hơn {r.factor} lần</>}>
          {CROWD} chains: the many units finish {r.factor} times sooner
        </Translated>
      </Headline>
      <Now t={t} />
    </Stage>
  );
}

// ---------------------------------------------------------------- beat 4
function Crossing(p: number) {
  const n = 1 + Math.round(ramp(p, 0.06, 0.52) * (CROWD - 1)); // chains swept so far
  const stretch = ramp(p, 0.62, 0.84);
  const length = CHAIN + Math.round(stretch * CHAIN); // then every chain grows to twice its length
  const base = sweep(A, B, CROWD, CHAIN);
  const pts = sweep(A, B, CROWD, length);
  const yMax = Math.max(...sweep(A, B, CROWD, 2 * CHAIN).map((q) => Math.max(q.a, q.b)));
  const pad = { l: 56, r: 228, t: 16, b: 46 };
  const X = (v: number) => pad.l + ((v - 1) / (CROWD - 1)) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - (v / yMax) * (H - pad.t - pad.b);
  const line = (q: typeof pts, key: 'a' | 'b', upto: number) => q.slice(0, upto).map((s) => `${X(s.chains)},${Y(s[key])}`).join(' ');
  const tie = regions(pts).find((g) => g.winner === 'tie');
  const lead = regions(pts);
  const cur = pts[n - 1];
  const tx = W - pad.r + 18;
  return (
    <Stage label={tr('Finishing time of both designs against the number of independent chains', 'Thời gian hoàn thành của hai thiết kế theo số chuỗi độc lập')}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--cell)" stroke="var(--rule)" />
      {tie && n >= tie.from ? (
        <g>
          <rect x={X(tie.from)} y={pad.t} width={X(Math.min(n, tie.to)) - X(tie.from)} height={H - pad.t - pad.b} fill="var(--rule-soft)" stroke="var(--rule)" />
          {n >= tie.to ? <text x={X(tie.to) + 6} y={pad.t + 16} fontSize="11.5"><Translated vi={<>hòa từ {tie.from} đến {tie.to} chuỗi</>}>tie from {tie.from} to {tie.to} chains</Translated></text> : null}
        </g>
      ) : null}
      {[0, yMax / 4, yMax / 2, (3 * yMax) / 4, yMax].map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end">{v}</text>
        </g>
      ))}
      {[1, 16, 32, 48, 64].map((v) => (
        <text key={v} x={X(v)} y={H - pad.b + 15} fontSize="11" textAnchor="middle">{v}</text>
      ))}
      <text x={(pad.l + W - pad.r) / 2} y={H - 10} fontSize="11.5" textAnchor="middle"><Translated vi="chuỗi độc lập">independent chains</Translated></text>
      <text x={14} y={(H - pad.b + pad.t) / 2} fontSize="11.5" textAnchor="middle" transform={`rotate(-90 14 ${(H - pad.b + pad.t) / 2})`}><Translated vi="thời gian đến kết quả cuối">time to the last result</Translated></text>
      {stretch > 0 ? (
        <g opacity="0.35">
          <polyline points={line(base, 'a', CROWD)} fill="none" stroke="var(--ink)" strokeWidth="1.4" />
          <polyline points={line(base, 'b', CROWD)} fill="none" stroke="var(--ink)" strokeWidth="1.4" strokeDasharray="6 4" />
        </g>
      ) : null}
      <polyline points={line(pts, 'a', n)} fill="none" stroke="var(--ink)" strokeWidth="2.2" />
      <polyline points={line(pts, 'b', n)} fill="none" stroke="var(--ink)" strokeWidth="2.2" strokeDasharray="6 4" />
      <circle cx={X(cur.chains)} cy={Y(cur.a)} r="4" fill="var(--ink)" />
      <circle cx={X(cur.chains)} cy={Y(cur.b)} r="4" fill="var(--cell)" stroke="var(--ink)" strokeWidth="1.8" />

      <text x={tx} y={30} fontSize="12.5" fontWeight="600"><Translated vi={<>chuỗi dài {length} phép toán</>}>chains of {length} operations</Translated></text>
      <line x1={tx} x2={tx + 26} y1={52} y2={52} stroke="var(--ink)" strokeWidth="2.2" />
      <text x={tx + 32} y={56} fontSize="12">{name(A)}</text>
      <line x1={tx} x2={tx + 26} y1={72} y2={72} stroke="var(--ink)" strokeWidth="2.2" strokeDasharray="6 4" />
      <text x={tx + 32} y={76} fontSize="12">{name(B)}</text>
      <text x={tx} y={112} fontSize="12">{tr(plural(cur.chains, 'chain'), `${cur.chains} chuỗi`)}</text>
      <text x={tx} y={130} fontSize="12"><Translated vi={<>đơn vị nhanh xong lúc {cur.a}</>}>fast units done at {cur.a}</Translated></text>
      <text x={tx} y={148} fontSize="12"><Translated vi={<>nhiều đơn vị xong lúc {cur.b}</>}>many units done at {cur.b}</Translated></text>
      <text x={tx} y={170} fontSize="12" fontWeight="600">
        {cur.winner === 'tie' ? tr('a tie', 'hòa') : tr(`${cur.winner === 'a' ? 'fast units' : 'many units'} lead, ${cur.factor}×`, `${cur.winner === 'a' ? 'đơn vị nhanh' : 'nhiều đơn vị'} dẫn trước, ${cur.factor}×`)}
      </text>
      {stretch > 0 ? (
        <g>
          {lead.map((g, i) => (
            <text key={i} x={tx} y={206 + i * 18} fontSize="12">
              {g.winner === 'tie' ? tr('tie', 'hòa') : g.winner === 'a' ? tr('fast units lead', 'đơn vị nhanh dẫn trước') : tr('many units lead', 'nhiều đơn vị dẫn trước')}: <Translated vi={<>{g.from} đến {g.to}</>}>{g.from} to {g.to}</Translated>
            </text>
          ))}
          <text x={tx} y={206 + lead.length * 18 + 6} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="như nhau ở mọi độ dài">the same at every length</Translated></text>
        </g>
      ) : null}
    </Stage>
  );
}

// ---------------------------------------------------------------- beat 5
function Matmul(p: number) {
  const n = MAT * MAT;
  const fb = finish(B, n, MAT);
  const fa = finish(A, n, MAT);
  const t = ramp(p, 0.3, 0.72) * fb.last;
  const s = snapshot(B, n, MAT, t);
  const map = ramp(p, 0.06, 0.24);
  return (
    <Stage label={tr('The many units each compute one element of a matrix product', 'Mỗi đơn vị trong nhiều đơn vị tính một phần tử của tích ma trận')}>
      <Title x={LX}><Translated vi={<>P = M × N, {MAT} x {MAT}: {n} phần tử</>}>P = M × N, {MAT} by {MAT}: {n} elements</Translated></Title>
      <Chains x={LX} y={32} size={176} ops={s.ops} length={MAT} cols={MAT} />
      <g opacity={map}>
        <line x1={LX + 190} x2={RX - 86} y1={120} y2={120} stroke="var(--ink)" strokeWidth="1.4" />
        <path d={`M ${RX - 86} 120 l -8 -5 v 10 z`} fill="var(--ink)" />
        <text x={(LX + 190 + RX - 86) / 2} y={110} fontSize="11.5" textAnchor="middle"><Translated vi="một phần tử">one element</Translated></text>
        <text x={(LX + 190 + RX - 86) / 2} y={138} fontSize="11.5" textAnchor="middle"><Translated vi="cho mỗi đơn vị">per unit</Translated></text>
      </g>
      <text x={RX - 74} y={18} fontSize="13" fontWeight="600">{name(B)}</text>
      <Plan x={RX - 74} y={32} size={176} d={B} busy={busyAt(B, n, MAT, t)} />
      <text x={RX + 124} y={62} fontSize="12"><Translated vi={<>{s.running} đơn vị đang làm việc</>}>{plural(s.running, 'unit')} working</Translated></text>
      <text x={RX + 124} y={80} fontSize="12"><Translated vi={<>xong {s.done} trên {n} phần tử</>}>{s.done} of {n} elements done</Translated></text>
      <text x={RX + 124} y={112} fontSize="12" fontWeight="600" opacity={t >= fb.last ? 1 : 0.25}><Translated vi={<>tất cả xong lúc {fb.last}</>}>all done at {fb.last}</Translated></text>
      <text x={RX + 124} y={130} fontSize="11.5" opacity={t >= fb.last ? 1 : 0.25} style={{ fill: 'var(--ink-2)' }}>
        <Translated vi={<>{A.units} đơn vị nhanh: lúc {fa.last}</>}>the {A.units} fast units: at {fa.last}</Translated>
      </text>
      <text x={LX} y={238} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi={<>bên trong một phần tử: một tích vô hướng {MAT} số hạng, ở đây chạy như một chuỗi {MAT} phép toán</>}>
          inside one element: a dot product of {MAT} terms, run here as one chain of {MAT} operations
        </Translated>
      </text>
      <Chain x={LX} y={248} n={MAT} done={s.ops[0]} w={208} />
      <Headline show={ramp(p, 0.76, 0.86)}>
        <Translated vi={<>{n} phần việc độc lập cho {B.units} đơn vị: không đơn vị nào phải chờ việc</>}>
          {n} independent pieces for {plural(B.units, 'unit')}: no unit waits for work
        </Translated>
      </Headline>
      <Now t={t} />
    </Stage>
  );
}

const beats: Beat[] = [
  { clip: 'two-designs-1', get label() { return tr('One budget', 'Một ngân sách'); }, draw: Budget },
  { clip: 'two-designs-2', get label() { return tr('One chain', 'Một chuỗi'); }, draw: OneChain },
  { clip: 'two-designs-3', get label() { return tr('Sixty-four chains', 'Sáu mươi bốn chuỗi'); }, draw: Crowd },
  { clip: 'two-designs-4', get label() { return tr('Where they cross', 'Nơi chúng giao nhau'); }, draw: Crossing },
  { clip: 'two-designs-5', get label() { return tr('What it asks of you', 'Điều nó đòi hỏi ở bạn'); }, draw: Matmul },
];

export function TwoDesignsScene() {
  return <Scene chapter="ch01" title={tr('One budget, two designs', 'Một ngân sách, hai thiết kế')} beats={beats} />;
}
