import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { ones, smallValues, warpReduceTrace, type Trace } from '../../sim/shuffle';
import { tr } from '../../lib/i18n';

// "Lanes exchanging values": one warp reducing in registers, one row per step, time
// running down the page. Every row is a step of a Trace from src/sim/shuffle.ts, the
// model the section's stepper runs on. Six beats, one narration clip each
// (content/ch10/narration/shuffle-N.txt). WarpGrid is also the stepper's drawing.

export const VIEW_W = 760;
/** Narrower than this, a lane drawing scrolls inside its bench instead of shrinking past legibility. */
export const MIN_W = 560;
const LABEL_W = 96;
const LANES_W = VIEW_W - LABEL_W - 8;
const ROW_H = 24;
const GAP = 14;
/** The scene has room to spare: wider gaps make each step's arrows readable. */
const SCENE_GAP = 26;
const TOP = 26;

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const show = (v: number | null) => (v == null ? '?' : String(v));

export interface GridProps {
  trace: Trace;
  /** Steps shown below the starting row: 0 to trace.steps.length. */
  upto: number;
  /** The step whose arrows are drawn (0-based); default: the last one shown. Null: none. */
  arrows?: number | null;
  /** Colour the lane-steps that reach the result and dim the rest. */
  cone?: boolean;
  /** With cone: show it only in the last this-many rows (it is revealed from the bottom up). */
  coneRows?: number;
  /** Wash the lanes below each row's stride (the part of the warp whose sums still matter). */
  live?: boolean;
  /** Lanes to outline in the last row shown. */
  ring?: number[];
  /** Label of the starting row. */
  startLabel?: string;
  /** Vertical space between rows, where the arrows go. */
  gap?: number;
}

const opName = (t: Trace) => (t.pattern === 'down' ? 'down' : 'xor');

/** Height of a grid with n rows. */
export const gridHeight = (rows: number, gap = GAP) => TOP + rows * ROW_H + (rows - 1) * gap;

export function WarpGrid({ trace, upto, arrows, cone = false, coneRows, live = false, ring = [], startLabel = tr('start', 'bắt đầu'), gap = GAP }: GridProps) {
  const hw = trace.hardwareWidth;
  const cw = LANES_W / hw;
  const x = (lane: number) => LABEL_W + lane * cw;
  const y = (row: number) => TOP + row * (ROW_H + gap);
  const shown = Math.max(0, Math.min(upto, trace.steps.length));
  const arrowStep = arrows === undefined ? shown - 1 : arrows;
  const rowsShown = shown + 1;
  const font = hw > 32 ? 7.5 : 9.5;
  const totalRows = rowsShown;
  // Which rows show the cone: the last `coneRows` of those drawn.
  const coneFrom = cone ? totalRows - (coneRows ?? totalRows) : Infinity;

  const cell = (row: number, lane: number, value: number | null, isLive: boolean, inCone: boolean, useful: boolean, stride: number | null) => {
    const coneOn = row >= coneFrom;
    let fill = 'var(--reg-wash)';
    let stroke = 'var(--reg)';
    let ink = 'var(--ink)';
    let opacity = 1;
    // Outside the live region: the lanes whose sums no longer matter, in every group the loop treats as a warp.
    const outside = live && stride != null && trace.pattern === 'down' && lane % trace.loopWidth >= stride;
    if (!isLive) {
      fill = 'none';
      stroke = 'var(--ink-3)';
      ink = 'var(--ink-3)';
    } else if (value == null) {
      fill = 'var(--fail-wash)';
      stroke = 'var(--fail)';
      ink = 'var(--fail)';
    } else if (coneOn) {
      if (inCone && (useful || row === 0)) {
        fill = 'var(--reg)';
        ink = 'var(--page)';
      } else if (inCone) {
        fill = 'var(--reg-wash)';
      } else {
        fill = 'var(--cell)';
        stroke = 'var(--rule)';
        ink = 'var(--ink-3)';
        opacity = 0.75;
      }
    } else if (outside) {
      fill = 'var(--cell)';
      stroke = 'var(--rule)';
    }
    const ringed = row === rowsShown - 1 && ring.includes(lane);
    // 64 lanes leave no room for every number: keep those that carry the story.
    const quiet = hw > 32 && !ringed && value != null && ((coneOn && !inCone) || outside);
    return (
      <g key={lane} data-lane={lane} data-cone={coneOn && inCone ? 1 : 0} data-value={show(value)} data-live={isLive ? 1 : 0} opacity={opacity}>
        <rect x={x(lane) + 0.6} y={y(row)} width={cw - 1.2} height={ROW_H} rx="2" fill={fill} stroke={ringed ? 'var(--ink)' : stroke}
          strokeWidth={ringed ? 2.2 : 1} strokeDasharray={isLive ? undefined : '2 2'} />
        {isLive && !quiet ? (
          <text x={x(lane) + cw / 2} y={y(row) + ROW_H / 2 + font / 2.6} fontSize={font} textAnchor="middle" style={{ fill: ink }}>
            {show(value)}
          </text>
        ) : null}
      </g>
    );
  };

  const rows: ReactNode[] = [];
  rows.push(
    <g key="r0" data-row={0} data-values={trace.initial.map((v, i) => (trace.live[i] ? show(v) : '-')).join(' ')}>
      <text x={LABEL_W - 8} y={y(0) + ROW_H / 2 + 4} fontSize="11" textAnchor="end">{startLabel}</text>
      {trace.initial.map((v, i) => cell(0, i, v, trace.live[i], trace.initialCone[i], true, trace.loopWidth))}
    </g>,
  );
  for (let k = 0; k < shown; k++) {
    const st = trace.steps[k];
    rows.push(
      <g key={'r' + (k + 1)} data-row={k + 1} data-stride={st.stride}
        data-values={st.lanes.map((l) => (l.live ? show(l.after) : '-')).join(' ')}>
        <text x={LABEL_W - 8} y={y(k + 1) + ROW_H / 2 + 4} fontSize="11" textAnchor="end">{opName(trace)} {st.stride}</text>
        {st.lanes.map((l) => cell(k + 1, l.lane, l.after, l.live, l.inCone, l.useful, st.stride))}
      </g>,
    );
  }

  let arrowEls: ReactNode = null;
  if (arrowStep != null && arrowStep >= 0 && arrowStep <= shown && arrowStep < trace.steps.length) {
    const st = trace.steps[arrowStep];
    const y0 = y(arrowStep) + ROW_H;
    const y1 = y(arrowStep + 1);
    arrowEls = (
      <g data-arrows={arrowStep} data-count={st.lanes.filter((l) => l.live && l.outcome === 'moved' && l.src !== l.lane).length}>
        {st.lanes.map((l) => {
          if (!l.live) return null;
          const xd = x(l.lane) + cw / 2;
          if (l.outcome === 'own')
            return <line key={l.lane} x1={xd} y1={y0 + 1} x2={xd} y2={y1 - 1} stroke="var(--ink-3)" strokeWidth="1" strokeDasharray="2 2" />;
          if (l.src == null || l.src === l.lane || l.src >= hw) return null;
          const xs = x(l.src) + cw / 2;
          const strong = !cone || l.inCone;
          return (
            <line key={l.lane} x1={xs} y1={y0 + 1} x2={xd} y2={y1 - 1}
              stroke={l.outcome === 'undefined' ? 'var(--fail)' : strong ? 'var(--reg)' : 'var(--ink-3)'}
              strokeWidth={strong ? 1.2 : 0.8} strokeDasharray={l.outcome === 'undefined' ? '3 2' : undefined} opacity={strong ? 0.9 : 0.5} />
          );
        })}
      </g>
    );
  }

  const every = hw > 32 ? 8 : hw > 16 ? 4 : 1;
  return (
    <g>
      {Array.from({ length: hw }, (_, i) =>
        i % every === 0 || i === hw - 1 ? (
          <text key={i} x={x(i) + cw / 2} y={TOP - 8} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{i}</text>
        ) : null,
      )}
      <text x={LABEL_W - 8} y={TOP - 8} fontSize="9" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{tr('lane', 'lane')}</text>
      {arrowEls}
      {rows}
    </g>
  );
}

// ---------------------------------------------------------------- the scene
const V32 = smallValues(32);
export const SCENE = {
  down: warpReduceTrace(V32, { pattern: 'down' }),
  xor: warpReduceTrace(V32, { pattern: 'xor' }),
  tailFrom: 20,
  gone: warpReduceTrace(V32, { pattern: 'down', tail: { from: 20, mode: 'return' } }),
  zero: warpReduceTrace(V32, { pattern: 'down', tail: { from: 20, mode: 'zero' } }),
  wave64: warpReduceTrace(ones(64), { pattern: 'down', loopWidth: 32 }),
  fixed64: warpReduceTrace(ones(64), { pattern: 'down' }),
};

const VIEW_H = gridHeight(7, SCENE_GAP) + 58;

function Stage({ children, headline, counters = [] }: { children: ReactNode; headline: ReactNode; counters?: { label: string; value: string }[] }) {
  const yb = VIEW_H - 14;
  return (
    <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} role="img" aria-label={tr('Animation of one warp reducing values in registers, one row per step', 'Hoạt hình một warp rút gọn các giá trị trong thanh ghi, mỗi bước một hàng')}
      style={{ width: '100%', maxWidth: VIEW_W, minWidth: MIN_W, display: 'block' }}>
      {children}
      <text x={8} y={yb} fontSize="13" fontWeight="600" data-headline>{headline}</text>
      {counters.map((c, i) => (
        <g key={c.label} data-counter={c.label}>
          <text x={VIEW_W - 8} y={yb - 18 * (counters.length - 1 - i) - 22} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
            {c.label}: <tspan data-value fontWeight="600" style={{ fill: 'var(--ink)' }}>{c.value}</tspan>
          </text>
        </g>
      ))}
    </svg>
  );
}

const steps = (t: Trace, p: number, a: number, b: number) => Math.min(t.steps.length, Math.floor(ramp(p, a, b) * t.steps.length + 1e-9));

const beats: Beat[] = [
  {
    clip: 'shuffle-1',
    get label() {
      return tr('Values in registers', 'Giá trị trong thanh ghi');
    },
    draw: (p) => {
      const t = SCENE.down;
      const k = p < 0.55 ? 0 : 1;
      return (
        <Stage headline={p < 0.3 ? tr('32 lanes, one value each, all in registers', '32 lane, mỗi lane một giá trị, tất cả trong thanh ghi') : p < 0.55 ? tr('every lane asks for the value 16 lanes above it', 'mọi lane đều xin giá trị của lane cao hơn nó 16') : tr('lanes 0 to 15 receive; lanes 16 to 31 keep their own', 'lane 0 đến 15 nhận; lane 16 đến 31 giữ giá trị của mình')}>
          <WarpGrid gap={SCENE_GAP} trace={t} upto={k} arrows={p >= 0.3 ? 0 : null} />
          {p >= 0.3 && k === 0 ? <PendingArrows t={t} /> : null}
        </Stage>
      );
    },
  },
  {
    clip: 'shuffle-2',
    get label() {
      return tr('Halve, add, repeat', 'Chia đôi, cộng, lặp lại');
    },
    draw: (p) => {
      const t = SCENE.down;
      const k = Math.max(1, steps(t, p, 0, 0.85));
      const done = k === t.steps.length;
      return (
        <Stage headline={done ? tr(`after 5 steps lane 0 holds ${t.lane0}, the sum of all 32`, `sau 5 bước lane 0 giữ ${t.lane0}, tổng của cả 32`) : tr(`stride ${t.steps[k - 1].stride}: the part that still matters halves`, `bước nhảy ${t.steps[k - 1].stride}: phần còn có ý nghĩa giảm một nửa`)}>
          <WarpGrid gap={SCENE_GAP} trace={t} upto={k} live ring={done ? [0] : []} />
        </Stage>
      );
    },
  },
  {
    clip: 'shuffle-3',
    get label() {
      return tr('Everyone computes', 'Ai cũng tính');
    },
    draw: (p) => {
      const t = SCENE.down;
      const rows = t.steps.length + 1;
      const coneRows = Math.max(1, Math.ceil(ramp(p, 0.2, 0.7) * rows));
      return (
        <Stage headline={p < 0.2 ? tr('every lane ran every shuffle and every add', 'mọi lane đều chạy mọi shuffle và mọi phép cộng') : p < 0.75 ? tr('the cone: what reaches lane 0', 'hình nón: những gì đến được lane 0') : tr('the rest is work nobody reads', 'phần còn lại là việc không ai đọc')}
          counters={[
            { label: tr('lane-operations', 'thao tác lane'), value: String(t.laneOps) },
            { label: tr('reach lane 0', 'đến được lane 0'), value: p >= 0.45 ? String(t.useful) : '…' },
            { label: tr('wasted', 'lãng phí'), value: p >= 0.75 ? String(t.wasted) : '…' },
          ]}>
          <WarpGrid gap={SCENE_GAP} trace={t} upto={t.steps.length} arrows={null} cone={p >= 0.2} coneRows={coneRows} />
        </Stage>
      );
    },
  },
  {
    clip: 'shuffle-4',
    get label() {
      return tr('Butterfly', 'Butterfly');
    },
    draw: (p) => {
      const t = SCENE.xor;
      const k = Math.max(1, steps(t, p, 0.05, 0.8));
      const done = k === t.steps.length;
      return (
        <Stage headline={done ? tr(`every lane holds ${t.total}`, `mọi lane đều giữ ${t.total}`) : tr(`xor ${t.steps[k - 1].stride}: partners trade in both directions`, `xor ${t.steps[k - 1].stride}: các cặp trao đổi theo cả hai chiều`)}
          counters={[{ label: tr('lane-operations', 'thao tác lane'), value: String(t.laneOps) }]}>
          <WarpGrid gap={SCENE_GAP} trace={t} upto={k} ring={done ? t.holders : []} />
        </Stage>
      );
    },
  },
  {
    clip: 'shuffle-5',
    get label() {
      return tr('A lane that left early', 'Một lane rời đi sớm');
    },
    draw: (p) => {
      if (p < 0.72) {
        const t = SCENE.gone;
        const k = steps(t, p, 0.1, 0.62);
        const firstQ = t.steps.findIndex((s) => s.lanes[0].after == null);
        return (
          <Stage headline={k === 0 ? tr(`lanes ${SCENE.tailFrom} and up have returned`, `các lane từ ${SCENE.tailFrom} trở lên đã return`) : k > firstQ ? tr('the question mark has reached lane 0: the total is undefined', 'dấu hỏi đã đến lane 0: tổng là không xác định') : tr('reading a lane that left gives an undefined value', 'đọc một lane đã rời đi cho ra một giá trị không xác định')}>
            <WarpGrid gap={SCENE_GAP} trace={t} upto={k} startLabel={tr('start', 'bắt đầu')} />
          </Stage>
        );
      }
      const z = SCENE.zero;
      return (
        <Stage headline={tr(`stay and add 0: lane 0 holds ${z.lane0}, the sum of lanes 0 to ${SCENE.tailFrom - 1}`, `ở lại và cộng 0: lane 0 giữ ${z.lane0}, tổng của lane 0 đến ${SCENE.tailFrom - 1}`)}>
          <WarpGrid gap={SCENE_GAP} trace={z} upto={z.steps.length} ring={[0]} startLabel={tr('0 past the end', '0 khi vượt quá cuối')} />
        </Stage>
      );
    },
  },
  {
    clip: 'shuffle-6',
    get label() {
      return tr('Sixty-four lanes', 'Sáu mươi tư lane');
    },
    draw: (p) => {
      if (p < 0.7) {
        const t = SCENE.wave64;
        const k = steps(t, p, 0.05, 0.55);
        const done = k === t.steps.length;
        return (
          <Stage headline={done ? tr(`lane 0 holds ${t.final[0]}, lane 32 holds ${t.final[32]}, nobody holds ${t.total}`, `lane 0 giữ ${t.final[0]}, lane 32 giữ ${t.final[32]}, không ai giữ ${t.total}`) : tr('a 64-lane wavefront, a loop written for 32', 'một wavefront 64 lane, một vòng lặp viết cho 32')}>
            <WarpGrid gap={SCENE_GAP} trace={t} upto={k} live ring={done ? [0, 32] : []} />
          </Stage>
        );
      }
      const f = SCENE.fixed64;
      return (
        <Stage headline={tr(`loop from warpSize / 2: 6 steps, lane 0 holds ${f.lane0}`, `lặp từ warpSize / 2: 6 bước, lane 0 giữ ${f.lane0}`)}>
          <WarpGrid gap={SCENE_GAP} trace={f} upto={f.steps.length} live ring={[0]} />
        </Stage>
      );
    },
  },
];

/** Beat 1, before the first row exists: the arrows of the first shuffle, ending in empty cells. */
function PendingArrows({ t }: { t: Trace }) {
  const hw = t.hardwareWidth;
  const cw = LANES_W / hw;
  const y1 = TOP + ROW_H + SCENE_GAP;
  return (
    <g>
      {t.steps[0].lanes.map((l) => (
        <rect key={l.lane} x={LABEL_W + l.lane * cw + 0.6} y={y1} width={cw - 1.2} height={ROW_H} rx="2" fill="none" stroke="var(--rule)" strokeDasharray="2 2" />
      ))}
    </g>
  );
}

export function ShuffleScene() {
  return <Scene chapter="ch10" title={tr('Lanes exchanging values', 'Các lane trao đổi giá trị')} beats={beats} />;
}
