import { useState, type KeyboardEvent } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  CHAPTERS,
  EDGES,
  EXTRAS,
  FOURTH_TO_FIFTH,
  GOALS,
  NEW_IN_FIFTH,
  chapter,
  chaptersBefore,
  courseDepartures,
  courseOrder,
  coursePath,
  extra,
  fifthFor,
  firstRenumbered,
  leadsTo,
  optionalPrerequisites,
  readingOrder,
  stageOf,
  type Edge,
  type ExtraId,
  type GoalId,
  type Item,
  type Tier,
} from '../../sim/bookmap';

// ---------------------------------------------------------------- small helpers
/** "2, 3 and 4" */
const list = (xs: (number | string)[]) =>
  xs.length <= 1 ? xs.join('') : xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1];
const edgeKey = (x: { from: number; to: number }) => `${x.from}>${x.to}`;
const isChapter = (i: Item): i is number => typeof i === 'number';

/** Break a short label into at most two lines for a small box. */
function twoLines(label: string): string[] {
  if (label.length <= 14) return [label];
  const words = label.split(' ');
  let best = 1;
  for (let i = 1; i < words.length; i++) {
    const a = words.slice(0, i).join(' ').length;
    const b = words.slice(0, best).join(' ').length;
    if (Math.abs(a - label.length / 2) < Math.abs(b - label.length / 2)) best = i;
  }
  return [words.slice(0, best).join(' '), words.slice(best).join(' ')];
}

// ---------------------------------------------------------------- the chart, in our own layout
// Positions are presentation only; which boxes and arrows exist comes from sim/bookmap.ts.
// Part I runs left to right, Parts II and III top to bottom, so that no arrow has to pass
// under a box it does not belong to.
const BW = 100;
const BH = 34;
const HW = BW / 2;
const HH = BH / 2;
const col = (c: number) => 86 + c * 110;
const AT: Record<number, [number, number]> = {
  1: [col(0), 26],
  // Part I has four columns of its own, spaced wider than the grid below it.
  2: [col(0), 116],
  3: [col(0) + 140, 92],
  4: [col(0) + 140, 140],
  5: [col(0) + 280, 116],
  6: [col(0) + 420, 116],
  7: [col(1), 240],
  15: [col(2), 240],
  9: [col(3), 240],
  10: [col(4), 240],
  13: [col(5), 240],
  8: [col(1), 292],
  11: [col(4), 292],
  12: [col(4), 344],
  14: [col(5), 396],
  19: [col(0), 490],
  23: [col(1), 490],
  16: [col(2), 490],
  21: [col(3), 490],
  17: [col(4), 490],
  20: [col(0), 542],
  18: [col(4), 542],
  22: [col(3), 594],
  24: [col(5), 650],
};
const EXTRA_AT: Record<ExtraId, [number, number]> = {
  appA: [col(0), 732],
  appC: [col(1), 732],
  x1: [col(2), 732],
  x2: [col(3), 732],
  x3: [col(4), 732],
};
const BANDS = [
  { get part() { return tr('Part I', 'Phần I'); }, top: 64, bottom: 168 },
  { get part() { return tr('Part II', 'Phần II'); }, top: 200, bottom: 424 },
  { get part() { return tr('Part III', 'Phần III'); }, top: 446, bottom: 622 },
];
const VIEW_W = 694;
const VIEW_H = 762;

/** From the bottom of one box to the top of another. */
const down = (a: number, b: number, dxa = 0, dxb = 0) => {
  const [xa, ya] = AT[a];
  const [xb, yb] = AT[b];
  const x1 = xa + dxa;
  const y1 = ya + HH;
  const x2 = xb + dxb;
  const y2 = yb - HH;
  const ym = (y1 + y2) / 2;
  return x1 === x2 ? `M${x1} ${y1}V${y2}` : `M${x1} ${y1}C${x1} ${ym} ${x2} ${ym} ${x2} ${y2}`;
};
/** From the right edge of one box to the left edge of another. */
const across = (a: number, b: number) => {
  const [xa, ya] = AT[a];
  const [xb, yb] = AT[b];
  const x1 = xa + HW;
  const x2 = xb - HW;
  const xm = (x1 + x2) / 2;
  return ya === yb ? `M${x1} ${ya}H${x2}` : `M${x1} ${ya}C${xm} ${ya} ${xm} ${yb} ${x2} ${yb}`;
};
const X = (n: number) => AT[n][0];
const Y = (n: number) => AT[n][1];
const ROUTE: Record<string, string> = {
  '1>2': down(1, 2),
  '2>3': across(2, 3),
  '2>4': across(2, 4),
  '3>5': across(3, 5),
  '4>5': across(4, 5),
  '5>6': across(5, 6),
  '6>7': down(6, 7, -36),
  '6>15': down(6, 15, -18),
  '6>9': down(6, 9),
  '6>10': down(6, 10, 18),
  '6>13': down(6, 13, 36),
  // The dashed arrow arches over the gap between the two boxes.
  '9>10': `M${X(9) + 34} ${Y(9) - HH}C${X(9) + 34} ${Y(9) - 42} ${X(10) - 34} ${Y(10) - 42} ${X(10) - 34} ${Y(10) - HH}`,
  '7>8': down(7, 8),
  '7>19': `M${X(7) - HW} ${Y(7)}C${X(19)} ${Y(7)} ${X(19)} ${Y(7) + 10} ${X(19)} ${Y(7) + 50}V${Y(19) - HH}`,
  '9>11': `M${X(9) + 26} ${Y(9) + HH}C${X(9) + 26} ${Y(11)} ${X(11) - 80} ${Y(11)} ${X(11) - HW} ${Y(11)}`,
  '10>11': down(10, 11),
  '11>12': down(11, 12),
  '12>14': `M${X(12) + HW} ${Y(12)}C${X(14) - 18} ${Y(12)} ${X(14) - 18} ${Y(12) + 10} ${X(14) - 18} ${Y(14) - HH}`,
  '13>14': `M${X(13) + 18} ${Y(13) + HH}V${Y(14) - HH}`,
  // Both arrows out of chapter 9 share a trunk, then fork in the gap between the parts.
  '9>21': `M${X(9)} ${Y(9) + HH}V${Y(21) - HH}`,
  '9>17': `M${X(9)} ${Y(9) + HH}V428C${X(9)} 456 ${X(17)} 440 ${X(17)} ${Y(17) - HH}`,
  '8>23': down(8, 23),
  '8>16': `M${X(8) + HW} ${Y(8)}C${X(16)} ${Y(8)} ${X(16)} ${Y(8) + 10} ${X(16)} ${Y(8) + 50}V${Y(16) - HH}`,
  '19>20': down(19, 20),
  '17>18': down(17, 18),
  '21>22': down(21, 22),
  '16>22': `M${X(16)} ${Y(16) + HH}V${Y(22) - 40}C${X(16)} ${Y(22)} ${X(22) - 90} ${Y(22)} ${X(22) - HW} ${Y(22)}`,
  '18>22': `M${X(18)} ${Y(18) + HH}C${X(18)} ${Y(22)} ${X(22) + 90} ${Y(22)} ${X(22) + HW} ${Y(22)}`,
};

type BoxState = 'target' | 'pre' | 'added' | 'later' | 'dim';

function Box({
  x,
  y,
  num,
  label,
  tier,
  state,
  shade,
  ordinal,
  attrs,
  onPick,
  aria,
}: {
  x: number;
  y: number;
  num: string;
  label: string;
  tier: Tier;
  state: BoxState;
  shade: boolean;
  ordinal?: number;
  attrs: Record<string, string>;
  onPick?: () => void;
  aria: string;
}) {
  const x0 = x - HW;
  const y0 = y - HH;
  const tab = 24;
  const filled = shade && tier === 'Deep';
  const tabFilled = shade && tier !== 'Condensed';
  const lines = twoLines(label);
  const onKey = (ev: KeyboardEvent) => {
    if (ev.key === 'Enter' || ev.key === ' ') {
      ev.preventDefault();
      onPick?.();
    }
  };
  return (
    <g
      {...attrs}
      data-state={state}
      data-tier={tier}
      role={onPick ? 'button' : undefined}
      tabIndex={onPick ? 0 : undefined}
      aria-pressed={onPick ? state === 'target' : undefined}
      aria-label={aria}
      onClick={onPick}
      onKeyDown={onPick ? onKey : undefined}
      style={{ cursor: onPick ? 'pointer' : 'default', opacity: state === 'dim' ? 0.42 : 1 }}
    >
      {state === 'added' ? (
        <rect x={x0 - 3.5} y={y0 - 3.5} width={BW + 7} height={BH + 7} rx="5" fill="none" stroke="var(--ink)" strokeWidth="1" />
      ) : null}
      <rect
        x={x0}
        y={y0}
        width={BW}
        height={BH}
        rx="3"
        fill={filled ? 'var(--ink)' : 'var(--cell)'}
        stroke="var(--ink)"
        strokeWidth={state === 'target' ? 3 : 1.2}
        strokeDasharray={state === 'later' ? '5 3' : undefined}
      />
      {tabFilled && !filled ? (
        <path d={`M${x0 + 3} ${y0}H${x0 + tab}V${y0 + BH}H${x0 + 3}a3 3 0 0 1 -3 -3V${y0 + 3}a3 3 0 0 1 3 -3z`} fill="var(--ink)" />
      ) : (
        <line x1={x0 + tab} x2={x0 + tab} y1={y0} y2={y0 + BH} stroke={filled ? 'var(--page)' : 'var(--ink)'} strokeOpacity="0.3" />
      )}
      <text x={x0 + tab / 2} y={y + 4.5} fontSize="12.5" fontWeight="700" textAnchor="middle" style={tabFilled ? { fill: 'var(--page)' } : undefined}>
        {num}
      </text>
      {lines.map((l, i) => (
        <text
          key={i}
          x={x0 + tab + 5}
          y={lines.length === 1 ? y + 3.5 : y - 2 + i * 11}
          fontSize="9.3"
          style={filled ? { fill: 'var(--page)' } : undefined}
        >
          {l}
        </text>
      ))}
      {ordinal != null ? (
        <g data-ordinal={ordinal}>
          <circle cx={x + HW - 1} cy={y0 + 1} r="8.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1" />
          <text x={x + HW - 1} y={y0 + 4.2} fontSize="8.6" fontWeight="650" textAnchor="middle">
            {ordinal}
          </text>
        </g>
      ) : null}
    </g>
  );
}

type Selection = { kind: 'goal'; id: GoalId } | { kind: 'chapter'; n: number };
type Mode = 'book' | 'course';

/** Why this course reads a chapter after something the chart puts it before. Ours. */
function whyLater(n: number): string {
  if (n === 13) return tr('condensed here: the radix sort this course needs builds on scan and filter', 'rút gọn ở đây: radix sort mà khóa học này cần xây trên scan và lọc');
  if (n === 7 || n === 19) return tr('read after attention: chapter 20 leans on chapters 5, 10 and 15', 'đọc sau attention: chương 20 dựa vào các chương 5, 10 và 15');
  return chapter(n).tier === 'Condensed'
    ? tr('condensed here, in the last stage', 'rút gọn ở đây, trong giai đoạn cuối')
    : tr(`in a later stage (${stageOf(n).toLowerCase()})`, `ở một giai đoạn sau (${stageOf(n).toLowerCase()})`);
}

function Chart({
  on,
  targets,
  added,
  later,
  extrasOn,
  extrasGoal,
  mode,
  shade,
  onPick,
}: {
  on: Set<number>;
  targets: number[];
  added: number[];
  later: number[];
  extrasOn: Set<ExtraId>;
  extrasGoal: ExtraId[];
  mode: Mode;
  shade: boolean;
  onPick: (n: number) => void;
}) {
  const departures = new Set(mode === 'course' ? courseDepartures().map(edgeKey) : []);
  const order = courseOrder();
  const ordinal = (i: Item) => (mode === 'course' ? order.indexOf(i) + 1 : undefined);
  const stateOf = (n: number): BoxState =>
    targets.includes(n) ? 'target' : later.includes(n) ? 'later' : added.includes(n) ? 'added' : on.has(n) ? 'pre' : 'dim';
  const edgeOn = (x: Edge) => on.has(x.from) && on.has(x.to);
  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      role="group"
      aria-label={tr("The book's chapters and their prerequisite arrows", 'Các chương của sách và các mũi tên tiên quyết của chúng')}
      style={{ width: '100%', minWidth: 600, maxWidth: VIEW_W, display: 'block' }}
    >
      <defs>
        <marker id="s17-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto" /* i18n-ignore: SVG attribute value */>
          <path d="M0 0.6L8 4L0 7.4z" style={{ fill: 'var(--ink)' }} />
        </marker>
      </defs>
      {BANDS.map((b) => (
        <g key={b.part}>
          <rect x="28" y={b.top} width={VIEW_W - 30} height={b.bottom - b.top} rx="4" fill="var(--page)" stroke="var(--rule)" />
          <text
            x="17"
            y={(b.top + b.bottom) / 2}
            fontSize="11"
            fontWeight="620"
            textAnchor="middle"
            transform={`rotate(-90 17 ${(b.top + b.bottom) / 2})`}
            style={{ fill: 'var(--ink-2)' }}
          >
            {b.part}
          </text>
        </g>
      ))}
      <rect x="28" y="686" width={VIEW_W - 30} height="72" rx="4" fill="none" stroke="var(--rule)" strokeDasharray="3 3" />
      <text x="36" y="702" fontSize="10" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi="Không phải các ô của sơ đồ trong sách. A và C là phụ lục của sách; X1 đến X3 là các module của khóa học này.">
          Not boxes of the book's chart. A and C are the book's appendices; X1 to X3 are this course's modules.
        </Translated>
      </text>
      {EDGES.map((x) => {
        const k = edgeKey(x);
        const active = edgeOn(x);
        const ahead = departures.has(k);
        return (
          <path
            key={k}
            data-edge={k}
            data-on={active ? '1' : '0'}
            data-kind={x.dashed ? 'dashed' : x.uncertain ? 'uncertain' : 'solid'}
            data-ahead={ahead ? '1' : undefined}
            d={ROUTE[k] ?? down(x.from, x.to)}
            fill="none"
            stroke="var(--ink)"
            strokeWidth={ahead ? 3 : 1.3}
            strokeDasharray={x.dashed ? '6 4' : x.uncertain ? '1.5 3.5' : undefined}
            strokeLinecap={x.uncertain ? 'round' : undefined}
            markerEnd="url(#s17-arrow)"
            opacity={active ? 1 : ahead ? 0.55 : 0.25}
          />
        );
      })}
      {CHAPTERS.map((c) => (
        <Box
          key={c.n}
          x={AT[c.n][0]}
          y={AT[c.n][1]}
          num={String(c.n)}
          label={c.short}
          tier={c.tier}
          state={stateOf(c.n)}
          shade={shade}
          ordinal={ordinal(c.n)}
          attrs={{ 'data-ch': String(c.n) }}
          onPick={() => onPick(c.n)}
          aria={tr(`Chapter ${c.n}: ${c.title}`, `Chương ${c.n}: ${c.title}`)}
        />
      ))}
      {EXTRAS.map((x) => (
        <Box
          key={x.id}
          x={EXTRA_AT[x.id][0]}
          y={EXTRA_AT[x.id][1]}
          num={x.num}
          label={x.short}
          tier={x.tier}
          state={extrasGoal.includes(x.id) ? 'target' : extrasOn.has(x.id) ? 'pre' : 'dim'}
          shade={shade}
          ordinal={ordinal(x.id)}
          attrs={{ 'data-extra': x.id }}
          aria={tr(`${x.pages ? 'Appendix' : 'Module'} ${x.num}: ${x.title}`, `${x.pages ? 'Phụ lục' : 'Module'} ${x.num}: ${x.title}`)}
        />
      ))}
    </svg>
  );
}

/** A short line sample for the legend. */
function LineSample({ dash, width = 1.3, round }: { dash?: string; width?: number; round?: boolean }) {
  return (
    <svg width="34" height="10" viewBox="0 0 34 10" aria-hidden="true" style={{ verticalAlign: 'middle', marginRight: '0.35em' }}>
      <path d="M1 5H27" fill="none" stroke="var(--ink)" strokeWidth={width} strokeDasharray={dash} strokeLinecap={round ? 'round' : undefined} />
      <path d="M26 1.6L33 5L26 8.4z" style={{ fill: 'var(--ink)' }} />
    </svg>
  );
}
function BoxSample({ kind }: { kind: 'Deep' | 'Standard' | 'Condensed' | 'target' | 'later' | 'added' }) {
  return (
    <svg width="26" height="14" viewBox="0 0 26 14" aria-hidden="true" style={{ verticalAlign: 'middle', marginRight: '0.35em' }}>
      {kind === 'added' ? <rect x="0.5" y="0.5" width="25" height="13" rx="3" fill="none" stroke="var(--ink)" strokeWidth="0.8" /> : null}
      <rect
        x={kind === 'added' ? 3 : 1.5}
        y={kind === 'added' ? 3 : 1.5}
        width={kind === 'added' ? 20 : 23}
        height={kind === 'added' ? 8 : 11}
        rx="2"
        fill={kind === 'Deep' ? 'var(--ink)' : 'var(--cell)'}
        stroke="var(--ink)"
        strokeWidth={kind === 'target' ? 2.6 : 1.1}
        strokeDasharray={kind === 'later' ? '4 2.5' : undefined}
      />
      {kind === 'Standard' ? <rect x="1.5" y="1.5" width="8" height="11" fill="var(--ink)" /> : null}
    </svg>
  );
}

function ReadingPath() {
  const [sel, setSel] = useState<Selection>({ kind: 'goal', id: 'matmul' });
  const [mode, setMode] = useState<Mode>('book');
  const [shade, setShade] = useState(false);

  const g = sel.kind === 'goal' ? GOALS.find((x) => x.id === sel.id)! : null;
  const targets = g ? g.chapters : [(sel as { n: number }).n];
  const extrasGoal: ExtraId[] = g ? g.extras : [];
  const single = targets.length === 1 ? targets[0] : null;

  // Everything below comes from the model.
  const path = coursePath(targets, extrasGoal);
  const order: Item[] = mode === 'book' ? [...readingOrder(targets), ...extrasGoal] : path.order;
  const later = mode === 'course' ? path.later : [];
  const added = mode === 'course' ? path.added : [];
  const chaptersInOrder = order.filter(isChapter);
  const on = new Set<number>([...chaptersInOrder, ...later]);
  const extrasOn = new Set<ExtraId>(order.filter((i): i is ExtraId => !isChapter(i)));
  const shown = order.filter((i) => i !== 1);
  const before = shown.filter((i) => isChapter(i) && !targets.includes(i)).length;
  const total = shown.filter(isChapter).length;
  const optional = optionalPrerequisites([...chaptersInOrder, ...later]).filter((n) => !on.has(n));
  const builtOn = single != null ? leadsTo([single]) : [];
  const offChart = mode === 'book' && targets.length === 0;

  const label = g
    ? tr(
        `${g.label}: ${[
          g.chapters.length ? (g.chapters.length === 1 ? 'chapter ' : 'chapters ') + list(g.chapters) : '',
          ...g.extras.map((x) => (extra(x).pages ? 'Appendix ' : 'module ') + extra(x).num),
        ]
          .filter(Boolean)
          .join(', ')}`,
        `${g.label}: ${[
          g.chapters.length ? (g.chapters.length === 1 ? 'chương ' : 'các chương ') + g.chapters.join(', ') : '',
          ...g.extras.map((x) => (extra(x).pages ? 'Phụ lục ' : 'module ') + extra(x).num),
        ]
          .filter(Boolean)
          .join(', ')}`,
      )
    : tr(`Chapter ${single}: ${chapter(single!).title}`, `Chương ${single}: ${chapter(single!).title}`);

  const row = (i: Item) => {
    if (isChapter(i)) {
      const c = chapter(i);
      const isTarget = targets.includes(i);
      return (
        <li key={i} data-item={i} style={{ marginBottom: '0.3em' }}>
          <span style={{ fontWeight: isTarget ? 700 : 500 }}>
            {c.n}. <a href={href(c.id)}>{c.title}</a>
          </span>{' '}
          <span className="muted">
            pp. {c.pages}, {tr(c.tier.toLowerCase(), ({ Deep: 'sâu', Standard: 'chuẩn', Condensed: 'rút gọn' } as Record<string, string>)[c.tier] ?? c.tier)}
            {isTarget ? tr(', your selection', ', lựa chọn của bạn') : ''}
            {added.includes(i) ? tr(', added by this course', ', do khóa học này thêm vào') : ''}
          </span>
        </li>
      );
    }
    const x = extra(i);
    return (
      <li key={i} data-item={i} style={{ marginBottom: '0.3em' }}>
        <span style={{ fontWeight: extrasGoal.includes(i) ? 700 : 500 }}>
          {x.num}. <a href={href(x.id)}>{x.title}</a>
        </span>{' '}
        <span className="muted">{x.pages ? tr(`appendix, pp. ${x.pages}`, `phụ lục, pp. ${x.pages}`) : tr("this course's module, not in the book", 'module của khóa học này, không có trong sách')}</span>
      </li>
    );
  };

  return (
    <Bench
      title={tr('Reading path', 'Lộ trình đọc')}
      task={tr(
        "Pick a goal or click a chapter. Then switch to this course's order and see what moves.",
        'Chọn một mục tiêu hoặc bấm vào một chương. Rồi chuyển sang thứ tự của khóa học này và xem điều gì dịch chuyển.',
      )}
      foot={
        <Translated
          vi={
            <>
              Các mũi tên được đọc từ một hình ảnh của Fig. P.1 trong sách (p. xxiii), vẽ lại theo bố cục của chúng tôi. Hai
              mũi tên chấm đến từ một điểm giao không rõ ràng trong bản PDF. Mục tiêu, mức độ sâu và thứ tự của khóa học này
              là của chúng tôi (PLAN.md).
            </>
          }
        >
          Arrows as read from an image of the book's Fig. P.1 (p. xxiii), redrawn in our own layout. The two dotted
          arrows come from a junction that is ambiguous in the PDF. Goals, depth tiers and this course's order are
          ours (PLAN.md).
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted"><Translated vi="Mục tiêu">Goal</Translated></span>
        <span className="seg" data-role="goals" style={{ flexWrap: 'wrap' }}>
          {GOALS.map((x) => (
            <button
              key={x.id}
              aria-pressed={sel.kind === 'goal' && sel.id === x.id}
              onClick={() => {
                setSel({ kind: 'goal', id: x.id });
                // Module X3 is not on the chart: only this course's order can place it.
                if (!x.chapters.length) setMode('course');
              }}
            >
              {x.label}
            </button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span className="muted"><Translated vi="Thứ tự">Order</Translated></span>
        <span className="seg" data-role="mode">
          <button aria-pressed={mode === 'book'} onClick={() => setMode('book')}>
            <Translated vi="Các mũi tên của sách">The book's arrows</Translated>
          </button>
          <button aria-pressed={mode === 'course'} onClick={() => setMode('course')}>
            <Translated vi="Thứ tự của khóa học này">This course's order</Translated>
          </button>
        </span>
        <label className="check">
          <input type="checkbox" checked={shade} onChange={(ev) => setShade(ev.target.checked)} />
          <Translated vi="tô các ô theo mức độ sâu của khóa học này">shade boxes by this course's depth</Translated>
        </label>
      </div>
      <div className="row" style={{ alignItems: 'flex-start', gap: '1rem' }}>
        <div style={{ flex: '1 1 38rem', minWidth: 0, overflowX: 'auto' }}>
          <Chart
            on={on}
            targets={targets}
            added={added}
            later={later}
            extrasOn={extrasOn}
            extrasGoal={extrasGoal}
            mode={mode}
            shade={shade}
            onPick={(n) => setSel({ kind: 'chapter', n })}
          />
        </div>
        <div className="panel" data-role="path" style={{ flex: '1 1 13.5rem', minWidth: 0 }}>
          <p style={{ margin: '0 0 0.4rem' }}>
            <b data-role="selection">{label}</b>
          </p>
          {offChart ? (
            <p data-role="off-chart" style={{ margin: 0 }}>
              <Translated vi="Module X3 không có trên sơ đồ của sách, nên các mũi tên không nói gì về nó. Hãy chuyển sang thứ tự của khóa học này.">
                Module X3 is not on the book's chart, so the arrows say nothing about it. Switch to this course's order.
              </Translated>
            </p>
          ) : (
            <>
              <Translated
                vi={
                  <p style={{ margin: '0 0 0.6rem' }}>
                    Trước lựa chọn: <b data-stat="before">{before}</b> chương. Tính cả nó: <b data-stat="total">{total}</b>.{' '}
                    <span className="muted">
                      {mode === 'book' ? 'Theo các mũi tên của sách.' : 'Theo thứ tự của khóa học này.'} Chương 1 không được tính: bạn
                      đang ở trong nó.
                    </span>
                  </p>
                }
              >
                <p style={{ margin: '0 0 0.6rem' }}>
                  Before the selection: <b data-stat="before">{before}</b> {before === 1 ? 'chapter' : 'chapters'}. With
                  it: <b data-stat="total">{total}</b>.{' '}
                  <span className="muted">
                    {mode === 'book' ? "By the book's arrows." : "In this course's order."} Chapter 1 is not counted: you
                    are in it.
                  </span>
                </p>
              </Translated>
              <ol data-role="order" style={{ margin: 0, paddingLeft: '1.2em', listStyle: 'none' }}>
                {shown.map(row)}
              </ol>
              {later.length ? (
                <>
                  <Translated
                    vi={
                      <p style={{ margin: '0.7rem 0 0.3rem' }}>
                        Các mũi tên của sách cũng đặt {later.length === 1 ? 'chương này' : 'các chương này'} trước. Khóa học này đọc{' '}
                        {later.length === 1 ? 'nó' : 'chúng'} sau:
                      </p>
                    }
                  >
                    <p style={{ margin: '0.7rem 0 0.3rem' }}>
                      The book's arrows also put {later.length === 1 ? 'this' : 'these'} first. This course reads{' '}
                      {later.length === 1 ? 'it' : 'them'} afterwards:
                    </p>
                  </Translated>
                  <ul data-role="later" style={{ margin: 0, paddingLeft: '1.2em' }}>
                    {later.map((n) => (
                      <li key={n} data-item={n} style={{ marginBottom: '0.3em' }}>
                        {n}. <a href={href(chapter(n).id)}>{chapter(n).title}</a>{' '}
                        <span className="muted">({whyLater(n)})</span>
                      </li>
                    ))}
                  </ul>
                </>
              ) : null}
              {optional.length ? (
                <p data-role="optional" className="muted" style={{ margin: '0.7rem 0 0' }}>
                  <Translated vi={<>Tùy chọn theo mũi tên nét đứt của sơ đồ: chương {optional.join(', ')} trước chương 10.</>}>
                    Optional by the chart's dashed arrow: chapter {list(optional)} before chapter 10.
                  </Translated>
                </p>
              ) : null}
              {single != null ? (
                <p data-role="leads" className="muted" style={{ margin: '0.7rem 0 0' }}>
                  {builtOn.length
                    ? tr(`Chapters that build on chapter ${single}: ${list(builtOn)}.`, `Các chương xây trên chương ${single}: ${builtOn.join(', ')}.`)
                    : single === 24
                      ? tr('Chapter 24 has no arrows in the chart.', 'Chương 24 không có mũi tên nào trong sơ đồ.')
                      : tr(
                          `Nothing in the chart builds on chapter ${single}. It leads nowhere further${
                            single === 14 ? ', reading the junction under it as starting at chapter 9' : ''
                          }.`,
                          `Không gì trong sơ đồ xây trên chương ${single}. Nó không dẫn tới đâu nữa${
                            single === 14 ? ', khi đọc điểm giao bên dưới nó là bắt đầu từ chương 9' : ''
                          }.`,
                        )}
                </p>
              ) : null}
            </>
          )}
        </div>
      </div>
      <div className="legend" data-role="legend" style={{ marginTop: '0.8rem' }}>
        <span>
          <BoxSample kind="target" />
          <Translated vi="lựa chọn của bạn">your selection</Translated>
        </span>
        <span>
          <LineSample />
          <Translated vi="tiên quyết">prerequisite</Translated>
        </span>
        <span>
          <LineSample dash="6 4" />
          <Translated vi="vẽ nét đứt trong sách (9 đến 10)">drawn dashed in the book (9 to 10)</Translated>
        </span>
        <span>
          <LineSample dash="1.5 3.5" round />
          <Translated vi="đọc từ điểm giao không rõ ràng (9 đến 17, 9 đến 21)">read from the ambiguous junction (9 to 17, 9 to 21)</Translated>
        </span>
        {mode === 'course' ? (
          <>
            <span data-legend="ahead">
              <LineSample width={3} />
              <Translated vi="một mũi tên mà khóa học này đi trước">an arrow this course runs ahead of</Translated>
            </span>
            <span>
              <BoxSample kind="later" />
              <Translated vi="đọc sau lựa chọn của bạn">read after your selection</Translated>
            </span>
            <span>
              <BoxSample kind="added" />
              <Translated vi="do khóa học này thêm vào">added by this course</Translated>
            </span>
            <span><Translated vi="số trong vòng tròn: vị trí trong thứ tự của khóa học này">numbers in circles: position in this course's order</Translated></span>
          </>
        ) : null}
        {shade ? (
          <>
            <span data-legend="tier">
              <BoxSample kind="Deep" />
              <Translated vi="sâu">deep</Translated>
            </span>
            <span>
              <BoxSample kind="Standard" />
              <Translated vi="chuẩn">standard</Translated>
            </span>
            <span>
              <BoxSample kind="Condensed" />
              <Translated vi="rút gọn">condensed</Translated>
            </span>
          </>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- edition map
/** Where each 4th-edition chapter is now. A static table: every row is visible at once. */
function EditionMap() {
  const from = firstRenumbered();
  const kept = FOURTH_TO_FIFTH.filter((c) => c.fate.kind === 'same');
  const changed = FOURTH_TO_FIFTH.filter((c) => c.fate.kind !== 'same');
  const link = (n: number) => <a href={href(chapter(n).id)}>{n}. {chapter(n).title}</a>;
  return (
    <table className="data wide" data-role="edition-map" style={{ maxWidth: '46rem' }}>
      <thead>
        <tr>
          <th><Translated vi="Ấn bản thứ 4 (2022)">4th edition (2022)</Translated></th>
          <th><Translated vi="Trong ấn bản thứ 5">In the 5th edition</Translated></th>
        </tr>
      </thead>
      <tbody>
        <tr data-n={`1-${from - 1}`} data-fate="same">
          <td>
            <Translated vi={<>1 đến {from - 1}: {kept[0].title} đến {kept[kept.length - 1].title}</>}>
              1 to {from - 1}: {kept[0].title} to {kept[kept.length - 1].title}
            </Translated>
          </td>
          <td><Translated vi="cùng số chương">the same numbers</Translated></td>
        </tr>
        {changed.map((c) => (
          <tr key={c.n} data-n={c.n} data-fate={c.fate.kind}>
            <td>
              {c.n}. {c.title}
            </td>
            <td>
              {c.fate.kind === 'moved' ? (
                link(c.fate.to)
              ) : c.fate.kind === 'removed' ? (
                <>
                  <Translated vi="đã bị bỏ">removed</Translated>
                  {c.fate.foldedInto ? (
                    <Translated vi={<>; một phần của nó giờ nằm ở {link(c.fate.foldedInto)}</>}>
                      ; some of it is now in {link(c.fate.foldedInto)}
                    </Translated>
                  ) : null}
                  {c.fate.appendix ? (
                    <Translated
                      vi={
                        <>
                          ; phần lịch sử mô hình bộ nhớ CUDA của nó giờ là{' '}
                          <a href={href('app' + c.fate.appendix)}>Phụ lục {c.fate.appendix}</a>
                        </>
                      }
                    >
                      ; its history of the CUDA memory model is now{' '}
                      <a href={href('app' + c.fate.appendix)}>Appendix {c.fate.appendix}</a>
                    </Translated>
                  ) : null}
                </>
              ) : null}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------- questions
const BEFORE_20 = chaptersBefore([20]);
/** If the chart were a line: every chapter from 2 to 19. */
const LINE_BEFORE_20 = CHAPTERS.filter((c) => c.n > 1 && c.n < 20).length;
const PART2_BEFORE_14 = chaptersBefore([14]).filter((n) => chapter(n).part === 'II');

export const questions17: Record<string, Question> = {
  'ch01/1.7/q-postpone': {
    kind: 'choice',
    id: 'ch01/1.7/q-postpone',
    prompt: (
      <Translated
        vi={
          <>
            Bạn muốn có một GEMM nhanh càng sớm càng tốt. Theo các mũi tên của sách, chương nào dưới đây có thể chờ đến sau mà
            không phá vỡ điều kiện tiên quyết của chương 15?
          </>
        }
      >
        <>
          You want a fast GEMM as soon as possible. By the book's arrows, which of these can wait until afterwards without
          breaking a prerequisite of chapter 15?
        </>
      </Translated>
    ),
    options: [
      {
        get label() {
          return tr(
            'Chapter 4, compute architecture and scheduling.',
            'Chương 4, kiến trúc tính toán và lập lịch.',
          );
        },
        get why() {
          return tr(
            'Chapter 4 feeds chapter 5, and 5 feeds 6.',
            'Chương 4 dẫn vào chương 5, và 5 dẫn vào 6.',
          );
        },
      },
      {
        get label() {
          return tr(
            'Chapter 5, memory architecture and data locality.',
            'Chương 5, kiến trúc bộ nhớ và tính cục bộ dữ liệu.',
          );
        },
        get why() {
          return tr(
            'Chapter 6 has one arrow into it, and it comes from 5.',
            'Chương 6 có một mũi tên đi vào, và nó đến từ 5.',
          );
        },
      },
      {
        get label() {
          return tr(
            'Chapter 6, performance considerations.',
            'Chương 6, các cân nhắc về hiệu năng.',
          );
        },
        get why() {
          return tr(
            'The only arrow into chapter 15 comes from chapter 6.',
            'Mũi tên duy nhất vào chương 15 đến từ chương 6.',
          );
        },
      },
      {
        get label() {
          return tr(
            'Chapter 7, convolution.',
            'Chương 7, tích chập.',
          );
        },
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Chương 15 treo trực tiếp vào chương 6, và 6 cần toàn bộ Phần I: 5 trước nó, 3 và 4 trước 5, 2 trước những chương đó.
            Chương 7 mở ra một nhánh khác (8, 19 và những gì theo sau chúng). Không gì trên đường tới 15 đi qua nó.
          </p>
        }
      >
        <p>
          Chapter 15 hangs directly off chapter 6, and 6 needs all of Part I: 5 before it, 3 and 4 before 5, 2 before
          those. Chapter 7 opens a different branch (8, 19 and what follows them). Nothing on the way to 15 passes
          through it.
        </p>
      </Translated>
    ),
    source: 'Fig. P.1, p. xxiii', // i18n-ignore: citation only
  },
  'ch01/1.7/q-edition16': {
    kind: 'numeric',
    id: 'ch01/1.7/q-edition16',
    prompt: (
      <Translated
        vi={
          <>
            Một bài viết năm 2023 nói "PMPP chương 16 cho thấy cách các lớp tích chập được hạ xuống phép nhân ma trận". Bạn mở
            chương nào của ấn bản thứ 5?
          </>
        }
      >
        <>
          A 2023 post says "PMPP chapter 16 shows how convolution layers are lowered to matrix multiplication". Which
          chapter of the 5th edition do you open?
        </>
      </Translated>
    ),
    answer: fifthFor(16)!,
    explain: (
      <Translated
        vi={
          <p>
            Năm 2023, ấn bản mới nhất là ấn bản thứ 4 (2022), nên hãy bắt đầu từ đó, nhưng kiểm tra xem bài viết nêu ấn bản nào:
            các ấn bản cũ hơn đánh số khác, và bản đồ trên trang này chỉ phủ ấn bản thứ 4. Chương 16 của ấn bản thứ 4 là Deep
            learning, và chủ đề của bài viết khớp với nó. Nội dung đó giờ là chương {fifthFor(16)}, Convolutional neural networks.
            Chương 16 của ấn bản thứ 5 là song song wavefront, chẳng liên quan gì đến bài viết.
          </p>
        }
      >
        <p>
          In 2023 the newest edition was the 4th (2022), so start there, but check which edition the post names: older
          editions number differently, and the map on this page covers the 4th only. The 4th edition's chapter 16 is
          Deep learning, and the post's topic fits it. That material is chapter {fifthFor(16)} now, Convolutional neural
          networks. Chapter 16 of the 5th edition is wavefront parallelism, which has nothing to do with the post.
        </p>
      </Translated>
    ),
    source: 'Preface pp. xviii–xxii; notes/today/ch01-facts.md, A15',
    vi: { source: 'Lời nói đầu pp. xviii–xxii; notes/today/ch01-facts.md, A15' },
  },
  'ch01/1.7/q-sortneeds': {
    kind: 'numeric',
    id: 'ch01/1.7/q-sortneeds',
    prompt: (
      <Translated
        vi={
          <>
            Lấy mẫu top-p cần các ứng viên đã sắp xếp. Theo các mũi tên của sách, có bao nhiêu chương của Phần II đứng trước
            chương 14?
          </>
        }
      >
        <>
          Top-p sampling wants sorted candidates. By the book's arrows, how many Part II chapters come before chapter 14?
        </>
      </Translated>
    ),
    answer: PART2_BEFORE_14.length,
    explain: (
      <Translated
        vi={
          <p>
            {PART2_BEFORE_14.join(', ')}: histogram, phép rút gọn (reduction), scan, lọc và trộn (merge). Khóa học này đi sâu vào
            10, 11 và 12, vì bộ chuẩn hóa của softmax là một phép rút gọn, xác suất tích lũy là một scan và giữ lại các ứng viên
            hàng đầu là một phép lọc. Nó rút gọn 13: trộn nằm trên đường tới merge sort, còn phép sắp xếp khóa học này cần là radix
            sort, mà Lời nói đầu mô tả là xây trên phép lọc ổn định của chương 12.
          </p>
        }
      >
        <p>
          {list(PART2_BEFORE_14)}: histogram, reduction, scan, filter and merge. This course goes deep on 10, 11 and 12,
          because a softmax normaliser is a reduction, cumulative probability is a scan and keeping the top candidates is
          a filter. It condenses 13: merge is on the path for merge sort, and the sort this course needs is radix sort,
          which the Preface describes as built on the stable filter of chapter 12.
        </p>
      </Translated>
    ),
    source: 'Fig. P.1, p. xxiii; Preface p. xx',
    vi: { source: 'Fig. P.1, p. xxiii; Lời nói đầu p. xx' },
  },
  'ch01/1.7/q-portable': {
    kind: 'choice',
    id: 'ch01/1.7/q-portable',
    prompt: (
      <Translated
        vi={
          <>
            Bạn chuyển nguyên một kernel từ GPU của nhà sản xuất này sang GPU của nhà sản xuất khác. Hai bảng thông số cho cùng
            tốc độ số học đỉnh và cùng băng thông bộ nhớ. Kernel qua các bài kiểm tra trên cả hai, và trên máy thứ hai thời gian
            chạy của riêng nó dài gấp ba. Lời giải thích khả dĩ nhất là gì?
          </>
        }
      >
        <>
          You move a kernel, unchanged, from one vendor's GPU to another's. The two specification sheets give the same
          peak arithmetic rate and the same memory bandwidth. The kernel passes its tests on both, and on the second its
          own run time is three times longer. What is the most likely explanation?
        </>
      </Translated>
    ),
    options: [
      {
        get label() {
          return tr(
            'The second GPU is the slower device, so a third of the speed is what its hardware allows.',
            'GPU thứ hai là thiết bị chậm hơn, nên một phần ba tốc độ là mức phần cứng của nó cho phép.',
          );
        },
        get why() {
          return tr(
            'The specification sheets rule that out: the peaks are the same. The second device could do this work as fast; the kernel does not ask for it in a way that device does well.',
            'Các bảng thông số loại trừ điều đó: tốc độ đỉnh như nhau. Thiết bị thứ hai có thể làm việc này nhanh như vậy; chỉ là kernel không yêu cầu nó theo cách mà thiết bị đó làm tốt.',
          );
        },
      },
      {
        get label() {
          return tr(
            "Amdahl's law: on the new machine the work left on the host is a larger share.",
            'Định luật Amdahl: trên máy mới, công việc để lại trên host chiếm phần lớn hơn.',
          );
        },
        get why() {
          return tr(
            "Amdahl's law is about the whole run. The figure here is the kernel's own run time, and host work does not enter it.",
            'Định luật Amdahl nói về toàn bộ lần chạy. Con số ở đây là thời gian chạy của riêng kernel, và công việc trên host không nằm trong đó.',
          );
        },
      },
      {
        get label() {
          return tr(
            'Choices that suited the first device, such as how the work is divided among threads, do not suit the second.',
            'Những lựa chọn hợp với thiết bị thứ nhất, chẳng hạn cách chia việc cho các luồng, không hợp với thiết bị thứ hai.',
          );
        },
        correct: true,
      },
      {
        get label() {
          return tr(
            'A bug the tests miss: correct source on hardware with equal peaks should run in about the same time.',
            'Một lỗi mà các bài kiểm tra bỏ sót: mã nguồn đúng trên phần cứng có tốc độ đỉnh bằng nhau lẽ ra phải chạy trong khoảng cùng thời gian.',
          );
        },
        get why() {
          return tr(
            'Nothing here points to a bug. Equal peaks do not make two devices the same design, so correct source has no reason to run equally fast on both. A slow kernel with correct output is a tuning problem until shown otherwise.',
            'Không có gì ở đây chỉ ra một lỗi. Tốc độ đỉnh bằng nhau không làm hai thiết bị có cùng thiết kế, nên mã nguồn đúng chẳng có lý do gì để chạy nhanh như nhau trên cả hai. Một kernel chậm cho kết quả đúng là vấn đề tinh chỉnh cho đến khi chứng minh được điều ngược lại.',
          );
        },
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sách đưa ra nhận xét này về OpenCL (p. 13): một chương trình chạy đúng trên mọi thiết bị hỗ trợ nó và thường cần sửa
            đổi để chạy tốt trên một thiết bị mới. AMD nói điều tương tự về việc đưa mã CUDA sang HIP: hãy chuẩn bị chuyển đổi
            thủ công và tinh chỉnh hiệu năng. Một lý do cụ thể, từ tài liệu của AMD: GPU trung tâm dữ liệu của AMD lập lịch luồng
            theo nhóm 64 trong khi của NVIDIA dùng 32, và mã giả định 32 chạy trên chúng chỉ dùng một nửa tài nguyên tính toán.
            Chương 4 giải thích các nhóm này. Tính đúng đắn đi theo mã nguồn. Hiệu năng phải được giành lại trên từng thiết bị,
            bằng cách đo ở đó.
          </p>
        }
      >
        <p>
          The book makes the remark about OpenCL (p. 13): a program runs correctly on any device that supports it and
          usually needs modification to perform well on a new one. AMD says the same about bringing CUDA code to HIP:
          expect manual porting and performance tuning. One concrete reason, from AMD's documentation: AMD's data-centre
          GPUs schedule threads in groups of 64 where NVIDIA's use 32, and code that assumes 32 runs on them using half
          of the compute resources. Chapter 4 explains these groups. Correctness travels with the source. Performance
          has to be earned again on each device, by measuring there.
        </p>
      </Translated>
    ),
    source: 'book p. 13; notes/today/ch01-facts.md, B2; ch05-facts.md, B5',
    vi: { source: 'sách p. 13; notes/today/ch01-facts.md, B2; ch05-facts.md, B5' },
  },
};
/** The same record under the name the chapter-building rules use. */
export const questionsS17 = questions17;

// ---------------------------------------------------------------- the section
export function S17() {
  return (
    <>
      <Translated
        vi={
          <p>
            Trang này nói về cách dùng sách, không nói về GPU. Hai điều giúp bạn tiết kiệm thời gian về sau: biết mục tiêu của
            mình phụ thuộc vào những chương nào, và biết một trang bạn tìm thấy trên mạng đang nói về ấn bản nào.
          </p>
        }
      >
        <p>
          This page is about using the book, not about GPUs. Two things save time later: knowing which chapters your
          goals depend on, and knowing which edition a page you found online is talking about.
        </p>
      </Translated>

      <h2 style={{ marginTop: '1.6rem' }}><Translated vi="Hình hài của cuốn sách">The shape of the book</Translated></h2>
      <Translated
        vi={
          <p>
            Ba phần nằm giữa hai chương mở và đóng. <b>Phần I</b>, chương 2 đến 6, là nền tảng: cách viết và khởi chạy một
            kernel, cách ánh xạ luồng tới dữ liệu, cách phần cứng lập lịch cho chúng, dữ liệu nằm ở đâu, và một danh sách tối ưu
            hóa mà phần còn lại của sách dùng lại. <b>Phần II</b>, chương 7 đến 15, mỗi chương một mẫu hình song song. Mỗi chương
            có mặt vì một kỹ thuật nó buộc bạn phải học: histogram cho phép toán nguyên tử, phép rút gọn cho cái giá của rào chắn,
            scan cho hiệu quả công việc. Phần này kết thúc bằng phép nhân ma trận chuyên sâu. <b>Phần III</b>, chương 16 đến 23,
            là các mẫu hình và ứng dụng nâng cao, nơi chủ đề là chọn giữa các cách phân rã một bài toán. Chương 1 đứng trên các
            phần và chương 24 khép lại cuốn sách. (Lời nói đầu có một lần nói "four major parts", ở p. xxii, rồi mô tả ba phần.
            Sơ đồ và §1.7 đều nói ba.)
          </p>
        }
      >
        <p>
          Three parts between two bookends. <b>Part I</b>, chapters 2 to 6, is the fundamentals: how a kernel is written
          and launched, how threads map to data, how the hardware schedules them, where data lives, and a checklist of
          optimisations that the rest of the book reuses. <b>Part II</b>, chapters 7 to 15, takes one parallel pattern per
          chapter. Each is there for a technique it forces on you: histogram for atomic operations, reduction for the
          cost of barriers, scan for work efficiency. It ends with matrix multiplication in depth. <b>Part III</b>,
          chapters 16 to 23, is advanced patterns and applications, where the subject is choosing between ways to
          decompose a problem. Chapter 1 sits above the parts and chapter 24 closes the book. (The Preface says "four
          major parts" once, on p. xxii, and then describes three. The chart and §1.7 both say three.)
        </p>
      </Translated>
      <Source section="1.7" pages="15–18" figs={tr('Preface pp. xxii–xxiv', 'Lời nói đầu pp. xxii–xxiv')} />

      <h2><Translated vi="Mục tiêu của bạn phụ thuộc vào đâu">What your goals depend on</Translated></h2>
      <Translated
        vi={
          <p>
            Lời nói đầu vẽ các chương thành một sơ đồ mũi tên tiên quyết, để giảng viên có thể bỏ qua hoặc sắp xếp lại chương mà
            không để lại lỗ hổng. Bạn có thể dùng nó theo cách tương tự. Trước khi nhìn vào nó, hãy chốt một con số.
          </p>
        }
      >
        <p>
          The Preface draws the chapters as a chart of prerequisite arrows, so that an instructor can skip or reorder
          chapters without leaving a gap. You can use it the same way. Before you look at it, commit to a number.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              <b>Sách</b> Fig. P.1, p. xxiii. Lời nói đầu trình bày các mũi tên như điều kiện tiên quyết, vẽ ra để giúp giảng viên
              bỏ qua hoặc sắp xếp lại chương (p. xxv). Một điểm giao của sơ đồ không rõ ràng trong bản PDF: đường từ chương 9 đến
              chương 17 và 21 đi ngay dưới ô của chương 14. Mũi tên từ 9 đến 10 được vẽ nét đứt.
            </p>
          }
        >
          <p>
            <b>Book</b> Fig. P.1, p. xxiii. The Preface presents the arrows as prerequisites, drawn to help instructors
            skip or reorder chapters (p. xxv). One junction of the chart is ambiguous in the PDF: the line from chapter 9
            to chapters 17 and 21 passes directly under the box of chapter 14. The arrow from 9 to 10 is drawn dashed.
          </p>
        </Translated>
      </Aside>
      <Predict
        id="ch01/1.7/p-before20"
        prompt={
          <Translated
            vi={
              <>
                Chương 20 là chương về attention. Theo các mũi tên tiên quyết của sách, bạn phải đọc bao nhiêu chương trước nó, không
                tính chương 1 và chính chương 20?
              </>
            }
          >
            <>
              Chapter 20 is the one on attention. By the book's prerequisite arrows, how many chapters must you have read
              before it, not counting chapter 1 or chapter 20 itself?
            </>
          </Translated>
        }
        unit={tr('chapters', 'chương')}
      >
        {(v) => (
          <Translated
            vi={
              <p>
                <b>{BEFORE_20.length}</b>: các chương {BEFORE_20.join(', ')}.{' '}
                {v === BEFORE_20.length
                  ? 'Đúng như bạn đã dự đoán.'
                  : v === LINE_BEFORE_20
                    ? 'Mười tám sẽ đúng nếu sơ đồ là một đường thẳng: chương 2 đến 19. Nhưng không phải: phần lớn Phần II và Phần III nằm trên các nhánh mà chương 20 không bao giờ chạm tới.'
                    : v === chaptersBefore([15]).length
                      ? 'Năm là đáp án cho chương 15, chương cần Phần I và không gì khác. Chương 20 thêm tích chập và CNN.'
                      : v > BEFORE_20.length
                        ? 'Ít hơn bạn nghĩ: sơ đồ rẽ nhánh sau chương 6, và chương 20 nằm trên một nhánh ngắn.'
                        : 'Toàn bộ Phần I nằm trên đường đi, rồi thêm hai chương: tích chập và CNN.'}{' '}
                Kiểm tra nó trong trình khám phá bên dưới.
              </p>
            }
          >
            <p>
              <b>{BEFORE_20.length}</b>: chapters {list(BEFORE_20)}.{' '}
              {v === BEFORE_20.length
                ? 'That is what you had.'
                : v === LINE_BEFORE_20
                  ? 'Eighteen would be right if the chart were a line: chapters 2 to 19. It is not: most of Part II and Part III sits on branches that chapter 20 never touches.'
                  : v === chaptersBefore([15]).length
                    ? 'Five is the answer for chapter 15, which needs Part I and nothing else. Chapter 20 adds convolution and CNNs.'
                    : v > BEFORE_20.length
                      ? 'Fewer than you thought: the chart branches after chapter 6, and chapter 20 sits on one short branch.'
                      : 'All of Part I is on the way, and then two more: convolution and CNNs.'}{' '}
              Check it in the explorer below.
            </p>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Trình khám phá là chính sơ đồ đó, vẽ lại. Chọn một mục tiêu, hoặc bấm vào bất kỳ chương nào, và nó giữ lại những gì
            các mũi tên đặt trước lựa chọn của bạn và làm mờ phần còn lại.
          </p>
        }
      >
        <p>
          The explorer is the same chart, redrawn. Pick a goal, or click any chapter, and it keeps what the arrows put
          before your choice and dims the rest.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một chương là một ô. Trong sách, một chương thường chỉ cần một hoặc hai mục của một chương trước, điều mà sơ đồ không
              thể hiện được.
            </p>
          }
        >
          <p>
            A chapter is one box. In the book a chapter often needs only a section or two of an earlier one, which the
            chart cannot show.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Các mũi tên được đọc từ một hình ảnh và một điểm giao không rõ ràng (xem ghi chú ở trên). Một mục tiêu kết thúc ở
              những chương nào, các mức độ sâu và "thứ tự của khóa học này" là của chúng tôi, từ PLAN.md, không phải của sách.
            </p>
          }
        >
          <p>
            The arrows were read from an image and one junction is ambiguous (see the note above). Which chapters a goal
            ends at, the depth tiers and "this course's order" are ours, from PLAN.md, not the book's.
          </p>
        </Translated>
      </Simplification>
      <ReadingPath />
      <Translated
        vi={
          <p>
            Bốn điều cần tìm. Chương 15, matmul nhanh, cần năm chương và không gì từ Phần II: bạn có thể đi thẳng từ chương 6 tới
            nó. Chương 20 cần bảy. Sắp xếp, chương 14, kéo theo mười, đó là lý do lấy mẫu là con đường dài nhất của bạn. Và năm
            chương không dẫn tới đâu nữa (14, 15, 20, 22 và 23, khi đọc điểm giao dưới chương 14 là bắt đầu từ 9), nên không có
            gì phía sau chờ chúng.
          </p>
        }
      >
        <p>
          Four things to find. Chapter 15, the fast matmul, needs five chapters and nothing from Part II: you can go from
          chapter 6 straight to it. Chapter 20 needs seven. Sorting, chapter 14, pulls in ten, which is why sampling is
          the longest of your roads. And five chapters lead nowhere further (14, 15, 20, 22 and 23, reading the junction
          under chapter 14 as starting at 9), so nothing downstream waits on them.
        </p>
      </Translated>
      <Ask q={questions17['ch01/1.7/q-postpone']} />

      <h3><Translated vi="Những chỗ khóa học này không theo các mũi tên">Where this course does not follow the arrows</Translated></h3>
      <Translated
        vi={
          <p>
            Chuyển trình khám phá sang thứ tự của khóa học này. Nó đi qua sách theo những gì mục tiêu của bạn cần trước: nền tảng,
            matmul, chuỗi lấy mẫu, attention và batching, lượng tử hóa, rồi phần còn lại. Ở ba chỗ, nó đọc một chương trước một
            thứ mà sơ đồ đặt trước, và nó thêm một điều kiện tiên quyết mà sơ đồ không có.
          </p>
        }
      >
        <p>
          Switch the explorer to this course's order. It takes the book by what your goals need first: foundations,
          matmul, the sampling chain, attention and batching, quantisation, then the rest. In three places it reads a
          chapter before something the chart puts first, and it adds one prerequisite the chart does not have.
        </p>
      </Translated>
      <Beyond title={tr("Ours, not the book's", 'Của chúng tôi, không phải của sách')}>
        <Translated
          vi={
            <p>
              Sách vẽ các mũi tên như điều kiện tiên quyết. Coi chúng là lời khuyên, mà khóa học này có thể gạt đi, là cách đọc
              của chúng tôi. Thứ tự là của PLAN.md. Trong một giai đoạn, trình khám phá sắp xếp các chương theo mũi tên của sách,
              và mọi mũi tên mà thứ tự này đi trước đều được vẽ đậm, để đến lượt bạn có thể gạt nó đi. Các lý do trong danh sách là
              của chúng tôi.
            </p>
          }
        >
          <p>
            The book draws the arrows as prerequisites. Treating them as advice, which this course may overrule, is our
            reading. The order is PLAN.md's. Inside a stage the explorer arranges chapters by the book's arrows, and
            every arrow the order runs ahead of is drawn heavy, so you can overrule it in turn. The reasons in the list
            are ours.
          </p>
        </Translated>
      </Beyond>
      <ul>
        <li>
          <Translated
            vi={
              <>
                <b>Thêm vào: phép rút gọn trước attention.</b> Softmax là một phép rút gọn trên một hàng, và trong kernel attention
                của chương 20, các luồng của một warp truyền giá trị cho nhau bằng warp shuffle (pp. 497–501), thứ mà chương 10 giới
                thiệu (<a href={href('ch10', '10.7')}>§10.7</a>). Vì thế ở đây chương 10 đứng trước 20.
              </>
            }
          >
            <b>Added: reduction before attention.</b> A softmax is a reduction over a row, and in the attention kernel of
            chapter 20 the threads of a warp pass values to each other with warp shuffles (pp. 497–501), which chapter 10
            introduces (<a href={href('ch10', '10.7')}>§10.7</a>). So chapter 10 comes before 20 here.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Attention trước tích chập và CNN.</b> Sơ đồ dẫn chương 20 qua 7 và 19. Các tham chiếu chéo mà chúng tôi ghi lại
                trong chương 20 chỉ về các chương 5, 10, 15 và 16 và Phụ lục B, phần nền tảng học sâu mà bạn đã có. Không tham chiếu
                nào chỉ về 7 hay 19, nên khóa học này đọc hai chương đó sau.
              </>
            }
          >
            <b>Attention before convolution and CNNs.</b> The chart routes chapter 20 through 7 and 19. The
            cross-references we recorded in chapter 20 point back to chapters 5, 10, 15 and 16 and to Appendix B, the
            deep-learning background you already have. None points to 7 or 19, so this course reads those two afterwards.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Trộn sau sắp xếp, và rút gọn.</b> Sơ đồ đặt trộn (13) trước sắp xếp (14) vì merge sort xây trên nó. Phép sắp xếp
                bạn cần cho lấy mẫu là radix sort, mà Lời nói đầu mô tả là xây trên phép lọc ổn định của chương 12 (p. xx).
              </>
            }
          >
            <b>Merge after sort, and condensed.</b> The chart puts merge (13) before sort (14) because merge sort is
            built on it. The sort you need for sampling is radix sort, which the Preface describes as built on the stable
            filter of chapter 12 (p. xx).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Chương 22 trước ba chương mà nó dựa vào.</b> Sơ đồ đặt 16, 18 và 21 trước 22. Khóa học này đọc 22 ngay sau
                attention, vì mục về batching như sự đánh đổi giữa độ trễ và thông lượng (<a href={href('ch22', '22.5')}>§22.5</a>),
                và giữ 16, 18 và 21 ở dạng trang rút gọn.
              </>
            }
          >
            <b>Chapter 22 before three chapters it draws on.</b> The chart puts 16, 18 and 21 before 22. This course
            reads 22 straight after attention, for its section on batching as latency against throughput (<a href={href('ch22', '22.5')}>§22.5</a>), and
            keeps 16, 18 and 21 as condensed pages.
          </Translated>
        </li>
      </ul>
      <Ask q={questions17['ch01/1.7/q-sortneeds']} />

      <h2><Translated vi="Ấn bản thứ 5 đã thay đổi gì">What the 5th edition changed</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn đang đọc ấn bản thứ 5. Ấn bản thứ 4 đề năm 2022, và phần lớn slide, bài blog và câu trả lời trên diễn đàn mà bạn
            sẽ tìm thấy được viết theo nó hoặc một ấn bản cũ hơn. Lời nói đầu nêu lý do cho một ấn bản mới: thêm ba thế hệ GPU
            (Ada Lovelace, Hopper, Blackwell), CUDA 11 đến 13, và các cụm GPU rất lớn giờ đang phục vụ truy vấn LLM.
          </p>
        }
      >
        <p>
          You are reading the 5th edition. The 4th is dated 2022, and most slides, blog posts and forum answers you will
          find were written against it or an earlier one. The Preface gives the reasons for a new edition: three more GPU
          generations (Ada Lovelace, Hopper, Blackwell), CUDA 11 through 13, and very large GPU clusters now serving LLM
          queries.
        </p>
      </Translated>
      <Today checked={tr("2026-11-20 against NVIDIA's pages (notes/today/ch01-facts.md, A5)", '2026-11-20, đối chiếu với các trang của NVIDIA (notes/today/ch01-facts.md, A5)')}>
        <Translated
          vi={
            <p>
              Một thế hệ sau Blackwell đang được giao hàng (Vera Rubin, theo thông cáo kết quả kinh doanh ngày 17 tháng 11 năm 2026
              của NVIDIA) và bộ toolkit hiện hành là CUDA 13.5. Bảng compute capability của NVIDIA chưa liệt kê thế hệ mới.
            </p>
          }
        >
          <p>
            A generation after Blackwell is shipping (Vera Rubin, per NVIDIA's results release of November 17, 2026) and
            the current toolkit is CUDA 13.5. NVIDIA's compute capability table did not list the new generation yet.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p data-role="changes">
            Những gì đã thay đổi, theo danh sách của Lời nói đầu, cùng phần liên quan đến bạn. Bốn chương mới:{' '}
            {NEW_IN_FIFTH.join(', ')}. Chương 12, lọc, là cách bạn giữ lại các ứng viên top-k hoặc top-p. Chương 15 đi sâu vào
            phép nhân ma trận, mà Lời nói đầu gọi là phép tính quan trọng nhất mà GPU được dùng ngày nay. Chương 16 là song song
            wavefront. Chương 20 bàn về mô hình ngôn ngữ lớn: attention, KV cache, flash attention và batching, những kernel
            attention duy nhất trong sách. Ba chương được viết lại: 6, với danh sách tối ưu hóa dài hơn mà các chương sau áp dụng;
            11, scan, thứ top-p cần; và 23, lập trình đa GPU.
          </p>
        }
      >
        <p data-role="changes">
          What changed, as the Preface lists it, with the part that concerns you. Four chapters are new: {list(NEW_IN_FIFTH)}.
          Chapter 12, filtering, is how you keep the top-k or top-p candidates. Chapter 15 takes matrix multiplication
          in depth, which the Preface calls the most important computation GPUs are used for today. Chapter 16 is
          wavefront parallelism. Chapter 20 covers large language models: attention, KV caching, flash attention and
          batching, the only attention kernels in the book. Three chapters were rewritten: 6, whose longer optimisation
          checklist the later chapters apply; 11, scan, which top-p needs; and 23, multi-GPU programming.
        </p>
      </Translated>
      <Source pages="xviii–xxii" figs={tr('Preface', 'Lời nói đầu')} />
      <Translated
        vi={
          <p>
            Trong các cập nhật, bốn điều quan trọng với bạn. Chương 9 giới thiệu API atomic của CUDA C++ mà các đoạn mã sau dùng.
            Chương 10 thêm warp shuffle, thứ kernel attention dựa vào. Chương 19, nơi tích chập được hạ xuống phép nhân ma trận,
            chuyển phần nền tảng học sâu sang Phụ lục B mới. Và Phụ lục A giờ bàn về các định dạng độ chính xác thấp, xuất phát
            từ trọng số LLM được lượng tử hóa. Chương 4 và 5 cũng lấy ví dụ từ H100, vẫn là thiết bị ví dụ của sách, không phải
            của bạn. Ba chương bị bỏ: tái tạo ảnh MRI lặp, song song động (dynamic parallelism) và "other advanced practices".
          </p>
        }
      >
        <p>
          Among the updates, four matter to you. Chapter 9 introduces the CUDA C++ atomics API that later listings use.
          Chapter 10 adds warp shuffles, which the attention kernel relies on. Chapter 19, where convolution is lowered
          to matrix multiplication, moved its deep-learning background to a new Appendix B. And Appendix A now covers
          low-precision formats, motivated by quantised LLM weights. Chapters 4 and 5 also take their examples from the
          H100, which stays the book's example device, not yours. Three chapters were removed: iterative MRI
          reconstruction, dynamic parallelism and "other advanced practices".
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Hệ quả với bất cứ thứ gì viết theo một ấn bản cũ hơn: nó không có chương về attention, không có chương chuyên sâu về
            GEMM, và từ chương 12 trở đi số chương của nó mang nghĩa khác.
          </p>
        }
      >
        <p>
          The consequence for anything written against an earlier edition: it has no chapter on attention and no chapter
          on GEMM in depth, and from chapter 12 onward its chapter numbers mean something else.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Trong những phần bị bỏ, song song động là thứ bạn dễ gặp nhất ngoài thực tế: mã chạy trên GPU khởi chạy thêm công
            việc GPU. Lời nói đầu nói tính năng này ít được áp dụng, một phần vì chi phí của nó, và chỉ bạn tới hướng dẫn lập
            trình CUDA và các ấn bản cũ hơn để tìm hiểu.
          </p>
        }
      >
        <p>
          Of the removals, dynamic parallelism is the one you are most likely to meet in the wild: code running on the
          GPU that launches more GPU work. The Preface says the feature saw little adoption, partly because of its
          overhead, and points to the CUDA programming guide and to older editions for it.
        </p>
      </Translated>
      <Today checked={tr("2026-11-20 against the CUDA 13.5 guide and AMD's HIP docs (notes/today/ch01-facts.md, A14, C4; ch05-facts.md, B4)", '2026-11-20, đối chiếu với hướng dẫn CUDA 13.5 và tài liệu HIP của AMD (notes/today/ch01-facts.md, A14, C4; ch05-facts.md, B4)')}>
        <Translated
          vi={
            <p>
              Nó rời khỏi sách, không rời khỏi CUDA: hướng dẫn vẫn ghi tài liệu cho nó (phiên bản gọi là CDP2 là mặc định từ CUDA
              12.0), không có thông báo ngừng hỗ trợ. HIP không hỗ trợ nó. Với Metal, chúng tôi không thấy tài liệu nào; chúng tôi
              cũng không tìm kiếm riêng cho nó.
            </p>
          }
        >
          <p>
            It left the book, not CUDA: the guide still documents it (the version called CDP2 is the default from CUDA
            12.0), with no deprecation notice. HIP does not support it. For Metal, none documented that we found; we did
            not search for it specifically.
          </p>
        </Translated>
      </Today>

      <h3><Translated vi="Trang đó nói về ấn bản nào?">Which edition is that page about?</Translated></h3>
      <Translated
        vi={
          <p>
            Một số chương không kèm ấn bản là mơ hồ từ 12 trở lên. Bản đồ dưới đây chuyển số chương của ấn bản thứ 4 sang ấn bản
            này. Trang web của nhà xuất bản sẽ không giải quyết câu hỏi giúp bạn; ghi chú nói vì sao.
          </p>
        }
      >
        <p>
          A chapter number with no edition attached is ambiguous from 12 upward. The map below takes the 4th edition's
          numbers to this one's. The publisher's web page will not settle the question for you; the note says why.
        </p>
      </Translated>
      <Today checked={tr("2026-11-20 against the publisher's pages (notes/today/ch01-facts.md, A15)", '2026-11-20, đối chiếu với các trang của nhà xuất bản (notes/today/ch01-facts.md, A15)')}>
        <Translated
          vi={
            <p>
              Các trang của nhà xuất bản cho ấn bản thứ 5 hiển thị một mục lục trước khi xuất bản: 25 chương trong bốn phần, không
              có chương LLM. Sách in có 24 chương trong ba phần. Hãy tin cuốn sách.
            </p>
          }
        >
          <p>
            The publisher's pages for the 5th edition showed a pre-publication contents list: 25 chapters in four parts,
            no LLM chapter. The printed book has 24 in three parts. Trust the book.
          </p>
        </Translated>
      </Today>
      <EditionMap />
      <Translated
        vi={
          <p className="muted" style={{ fontSize: '0.85rem' }}>
            Tên chương của ấn bản thứ 4 lấy từ trang của nhà xuất bản. Việc ghép hai ấn bản theo tên chương là suy luận của chúng
            tôi (notes/today/ch01-facts.md, A15). Phía ấn bản thứ 5 là sách in.
          </p>
        }
      >
        <p className="muted" style={{ fontSize: '0.85rem' }}>
          The 4th-edition titles are from the publisher's page. Pairing the two editions by title is our inference
          (notes/today/ch01-facts.md, A15). The 5th-edition side is the printed book.
        </p>
      </Translated>
      <Ask q={questions17['ch01/1.7/q-edition16']} />

      <h2><Translated vi="CUDA đứng ở đâu giữa các giao diện khác">Where CUDA sits among other interfaces</Translated></h2>
      <Translated
        vi={
          <p>
            Sách đặt CUDA cạnh ba giao diện khác. <b>OpenMP</b>: bạn chú thích vòng lặp bằng các chỉ thị và trình biên dịch cùng
            runtime tạo ra mã song song; nó sinh ra trên CPU rồi mở rộng sang GPU, và nó mua sự tiện lợi bằng cái giá là quyền
            kiểm soát. <b>MPI</b>: cho các cụm máy mà các nút không chia sẻ bộ nhớ, nên mọi trao đổi đều là một thông điệp tường
            minh; đó là tầng phía trên một GPU, mà khóa học này chỉ bàn ngắn gọn (<a href={href('ch23')}>chương 23</a> của nó giữ
            lại stream, sao chép bất đồng bộ và bộ nhớ ghim, và chỉ đọc vòng lặp MPI của sách để xem ai chờ). <b>OpenCL</b>: gần CUDA
            về khái niệm, dựa nhiều hơn vào lời gọi API và ít hơn vào phần mở rộng ngôn ngữ; một chương trình OpenCL chạy không cần
            sửa trên mọi thiết bị hỗ trợ nó, và thường cần sửa đổi để chạy tốt trên một thiết bị mới.
          </p>
        }
      >
        <p>
          The book places CUDA against three others. <b>OpenMP</b>: you annotate loops with directives and a compiler and
          runtime produce the parallel code; it was born on CPUs and extended to GPUs, and it buys convenience at the
          price of control. <b>MPI</b>: for clusters whose nodes share no memory, so every exchange is an explicit
          message; that is the level above one GPU, which this course treats only briefly (its{' '}
          <a href={href('ch23')}>chapter 23</a> keeps streams, asynchronous copies and pinned memory, and reads the
          book's MPI loop only to see who waits). <b>OpenCL</b>: close to
          CUDA in concepts, built more on API calls and less on language extensions; an OpenCL program runs unchanged on
          any device that supports it, and usually needs modification to run well on a new one.
        </p>
      </Translated>
      <Source section="1.5" pages="12–13" />
      <Translated
        vi={
          <p>
            Hãy giữ lấy nhận xét cuối đó. Nó nói về OpenCL, và nó mô tả mọi lần chuyển đổi bạn sẽ làm: tính đúng đắn đi theo mã
            nguồn, hiệu năng thì không. Sách không bao giờ nhắc đến HIP hay Metal, hai giao diện mà khóa học này thêm vào, nhưng
            AMD nói về HIP đúng điều sách nói về OpenCL: một lập trình viên CUDA sẽ thấy API dễ hiểu, và HIP "not intended to be
            a drop-in replacement", nên hãy chuẩn bị chuyển đổi thủ công và tinh chỉnh hiệu năng. Còn bản thân OpenCL, Apple đã
            ngừng hỗ trợ nó từ macOS 10.14 và chỉ định Metal là sự thay thế.
          </p>
        }
      >
        <p>
          Hold on to that last remark. It is about OpenCL, and it describes every port you will do: correctness travels
          with the source, performance does not. The book never mentions HIP or Metal, the two interfaces this course
          adds, but AMD says of HIP what the book says of OpenCL: a CUDA programmer will find the API straightforward,
          and HIP is "not intended to be a drop-in replacement", so expect manual porting and performance tuning. As for
          OpenCL itself, Apple deprecated it in macOS 10.14 and names Metal as the replacement.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>HIP và Metal là phần khóa học này thêm vào. Các phát biểu về chúng là của chính các nhà sản xuất.</p>}>
          <p>HIP and Metal are this course's additions. The statements about them are the vendors' own.</p>
        </Translated>
        <Translated
          vi={
            <p className="muted">
              Kiểm tra ngày 2026-11-20 với tài liệu của AMD và Apple (notes/today/ch01-facts.md, B1, B2, A11).
            </p>
          }
        >
          <p className="muted">Checked 2026-11-20 against AMD's and Apple's documentation (notes/today/ch01-facts.md, B1, B2, A11).</p>
        </Translated>
      </Beyond>
      <Ask q={questions17['ch01/1.7/q-portable']} />
      <Translated
        vi={
          <p>
            Với mục tiêu thứ ba của bạn, đây là cách ba code base bạn muốn đọc sắp xếp các backend GPU của chúng. Đọc mỗi hàng
            để tìm một điều: cột AMD và Metal chứa mã mới hay lại là mã CUDA. Tóm tắt là ba backend, hai code base.
          </p>
        }
      >
        <p>
          For your third goal, here is how the three code bases you want to read arrange their GPU backends. Read each
          row for one thing: whether the AMD and Metal columns hold new code or the CUDA code again. The short version
          is three backends, two code bases.
        </p>
      </Translated>
      <Today checked={tr("2026-11-20 against each project's repository and docs (notes/today/ch01-facts.md, D1, D3, D4; ch05-facts.md, B9)", '2026-11-20, đối chiếu với repository và tài liệu của từng dự án (notes/today/ch01-facts.md, D1, D3, D4; ch05-facts.md, B9)')}>
        <Translated
          vi={
            <p>
              Bố cục thay đổi. <a href={href('x3')}>Module X3</a> kiểm tra lại chúng khi mở các tệp.
            </p>
          }
        >
          <p>
            Layouts move. <a href={href('x3')}>Module X3</a> checks them again when it opens the files.
          </p>
        </Translated>
      </Today>
      <table className="data wide" data-role="backends">
        <thead>
          <tr>
            <th style={{ width: '7rem' }}><Translated vi="Dự án">Project</Translated></th>
            <th>CUDA</th>
            <th><Translated vi="HIP (AMD)">HIP (AMD)</Translated></th>
            <th>Metal</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <b>PyTorch</b>
            </td>
            <td>
              <Translated
                vi={
                  <>
                    Tên device <code>'cuda'</code>.
                  </>
                }
              >
                Device name <code>'cuda'</code>.
              </Translated>
            </td>
            <td>
              <Translated
                vi={
                  <>
                    Tên device cũng là <code>'cuda'</code>. Chủ yếu là mã nguồn CUDA chạy qua một script HIPify, cộng vài tệp chỉ dành
                    cho AMD.
                  </>
                }
              >
                Device name <code>'cuda'</code> as well. Mostly the CUDA sources run through a HIPify script, plus a few
                AMD-only files.
              </Translated>
            </td>
            <td>
              <Translated
                vi={
                  <>
                    Một device <code>'mps'</code> riêng, xây trên các framework Metal Performance Shaders và MPSGraph của Apple và trên
                    các kernel <code>.metal</code> viết tay của riêng nó.
                  </>
                }
              >
                A separate <code>'mps'</code> device, built on Apple's Metal Performance Shaders and MPSGraph frameworks
                and on hand-written <code>.metal</code> kernels of its own.
              </Translated>
            </td>
          </tr>
          <tr>
            <td>
              <b>llama.cpp</b>
            </td>
            <td>
              <code>ggml-cuda</code>
            </td>
            <td>
              <Translated
                vi={
                  <>
                    <code>ggml-hip</code> chỉ chứa tệp build và biên dịch mã nguồn <code>ggml-cuda</code>.
                  </>
                }
              >
                <code>ggml-hip</code> holds build files only and compiles the <code>ggml-cuda</code> sources.
              </Translated>
            </td>
            <td>
              <Translated
                vi={
                  <>
                    <code>ggml-metal</code>, mã riêng, bật mặc định trên macOS.
                  </>
                }
              >
                <code>ggml-metal</code>, separate code, on by default on macOS.
              </Translated>
            </td>
          </tr>
          <tr>
            <td>
              <b>MLX</b>
            </td>
            <td><Translated vi="Một backend CUDA, trên Linux.">A CUDA backend, on Linux.</Translated></td>
            <td><Translated vi="Không có.">None.</Translated></td>
            <td><Translated vi="Backend chính. MLX tự mô tả là một framework cho Apple silicon.">The primary backend. MLX describes itself as a framework for Apple silicon.</Translated></td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Khi có backend AMD, nó chủ yếu là mã CUDA dưới tên khác, còn backend Metal là một code base thứ hai. Vì thế học CUDA
            cho tốt là phủ được hai trong ba, và Metal là nơi chuyển đổi đồng nghĩa với viết lại:{' '}
            <a href={href('x3')}>module X3</a> đọc các thư mục này.
          </p>
        }
      >
        <p>
          Where an AMD backend exists it is mostly the CUDA code under other names, and the Metal backend is a second
          code base. So CUDA learned well covers two of the three, and Metal is where a port means a rewrite:{' '}
          <a href={href('x3')}>module X3</a> reads these directories.
        </p>
      </Translated>

      <h2><Translated vi="Ba mục tiêu, ba thói quen">Three goals, three habits</Translated></h2>
      <Translated vi={<p>Sách nêu ba mục tiêu cho bạn. Khóa học này gắn một thói quen với mỗi mục tiêu.</p>}>
        <p>The book states three goals for you. This course attaches a habit to each.</p>
      </Translated>
      <Source section="1.6" pages="14" />
      <ul>
        <li>
          <Translated
            vi={
              <>
                <b>Hiệu năng.</b> Lập trình cho hiệu năng cao từ sự hiểu biết khái niệm về phần cứng, mà không cần là chuyên gia phần
                cứng. Thói quen: đo trên thiết bị của chính bạn. Mọi lab đều đo hiệu năng, và không bài học nào hứa với bạn một con
                số.
              </>
            }
          >
            <b>Performance.</b> Program for high performance from a conceptual understanding of the hardware, without
            having to be a hardware expert. The habit: measure on your own device. Every lab benchmarks, and no lesson
            promises you a number.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Chức năng đúng và độ tin cậy.</b> Thói quen: chạy các kernel CUDA của bạn qua các trình giả lập (emulator) CPU của
                khóa học này. Backend <code>emu</code> chạy lại một kernel theo ba thứ tự luồng cố định, nên một chỉ số sai, một rào
                chắn bị thiếu hay một truy cập ngoài biên đều thất bại mọi lần, không phải mỗi tuần một lần. Các cập nhật không đồng
                bộ, như một atomic bị thiếu, cần backend <code>race</code>, backend này báo cáo chúng kèm dòng mã nguồn, nhưng chỉ
                với những lịch chạy thực sự xảy ra.
              </>
            }
          >
            <b>Correct functionality and reliability.</b> The habit: run your CUDA kernels through this course's CPU
            emulators. The <code>emu</code> backend replays a kernel under three fixed thread orders, so a wrong index, a
            missing barrier or an out-of-bounds access fails every time, not once a week. Unsynchronised updates, such as
            a missing atomic, need the <code>race</code> backend, which reports them with source lines, but only for
            schedules that actually occur.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Khả năng mở rộng qua các thế hệ phần cứng.</b> Chìa khóa của sách cho điều này là truy cập bộ nhớ đều đặn và cục
                bộ. Thói quen đi kèm: truy vấn thiết bị, đừng bao giờ giả định.
              </>
            }
          >
            <b>Scalability across hardware generations.</b> The book's key to it is regular, local memory access. The
            habit that goes with it: query the device, never assume it.
          </Translated>
        </li>
      </ul>
      <Translated
        vi={
          <p>
            Phần giới thiệu chương 2 của sách nêu một công thức sáu bước để chuyển một phép tính sang GPU (§1.7, p. 15): chọn
            phần cần song song hóa, cấp phát bộ nhớ device, chép dữ liệu sang, viết kernel, khởi chạy nó, chép kết quả về. Chương
            2 đi qua nó, nên ở đây chỉ nêu tên. Trình tự phía host của Apple cho một kernel tính toán Metal phủ cùng nội dung dưới
            tên khác, không có hai lần sao chép: trên Apple silicon, một buffer ở chế độ lưu trữ shared hiện ra với cả hai bộ xử lý
            (xem câu hỏi cuối của <a href={href('ch01', 'wrap')}>phần tóm tắt</a>).
          </p>
        }
      >
        <p>
          The book's preview of chapter 2 names a six-step recipe for moving a computation to a GPU (§1.7, p. 15): pick
          the part to parallelise, allocate device memory, copy the data over, write the kernel, launch it, copy the
          results back. Chapter 2 works through it, so it is only named here. Apple's host-side sequence for a Metal
          compute kernel covers the same ground under other names, without the two copies: on Apple silicon a buffer
          in shared storage is visible to both processors (see the last question of the{' '}
          <a href={href('ch01', 'wrap')}>summary</a>).
        </p>
      </Translated>
      <Today checked={tr("2026-11-20 against Apple's Metal documentation (notes/today/ch01-facts.md, C1; ch05-facts.md, C6)", '2026-11-20, đối chiếu với tài liệu Metal của Apple (notes/today/ch01-facts.md, C1; ch05-facts.md, C6)')}>
        <Translated
          vi={
            <p>
              Cách diễn đạt của Apple: tìm một GPU, tạo một pipeline cho hàm, tạo các đối tượng dữ liệu mà GPU truy cập được, rồi
              tạo một command buffer, ghi lệnh vào đó và commit nó vào một command queue. Việc khớp điều đó với sáu bước của sách
              là cách đọc của chúng tôi.
            </p>
          }
        >
          <p>
            Apple's wording: find a GPU, create a pipeline for the function, create data objects the GPU can access, then
            create a command buffer, write commands into it and commit it to a command queue. Matching that to the book's
            six steps is our reading.
          </p>
        </Translated>
      </Today>
    </>
  );
}
