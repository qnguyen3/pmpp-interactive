import { useEffect, useId, useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { benchRows, fmt, num, realBackends, useReports, type BenchRow, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  EX1,
  FIG49,
  countTaken,
  discardedBySelect,
  exercise1,
  fig410Trips,
  ifElse,
  loop,
  loopBatch,
  loopRows,
  maskBelow,
  maskModulo,
  run,
  select,
  shuffledTrips,
  sortedTrips,
  type Issue,
  type LoopBatch,
  type Order,
} from '../../sim/divergence';
import {
  BOOK_CASES,
  BOOK_WIDTH,
  boundary1D,
  boundary2D,
  causal,
  classifyGrid,
  dims,
  gridFor,
  regions,
  splitWarp1D,
  warpsInRightEdgeBlocks,
  widthChoices,
  type Picture,
  type WarpState,
  type WidthChoice,
} from '../../sim/warps';
import { WarpBranchScene } from './WarpBranchScene';

// The kernel of the book's exercise 1 (pp. 90 to 91), exactly as stored in
// labs/ch04/snippets/foo_kernel.cu, where the CUDA front-end check parses it.
// app/src/sim/divergence.test.ts fails if this copy drifts from the file.
const FOO = `__global__ void foo_kernel(int* a, int* b) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(threadIdx.x < 40 || threadIdx.x >= 104) {
        b[i] = a[i] + 1;
    }
    if(i%2 == 0) {
        a[i] = b[i]*2;
    }
    for(unsigned int j = 0; j < 5 - (i%3); ++j) {
        b[i] += j;
    }
}
void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel<<<(N + 128 - 1)/128, 128>>>(a_d, b_d);
}`;

// ---------------------------------------------------------------- small helpers
const pct = (v: number) => `${Math.round(v * 1000) / 10}%`;
/** A count of warp-steps that may be fractional (work divided by the width). */
const stepsText = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2));
const plural = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;
const big = (v: number) => v.toLocaleString('en-US');
/** The most lanes any control on this page stores a value for. */
const LANES_MAX = 128;
const cellFor = (width: number) => (width <= 32 ? 14 : width <= 64 ? 9 : 5);
/** A label above a group of buttons, laid out like label.field. */
const FIELD = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' } as const;
/** Lets a label.field holding a long select shrink to the width of a narrow screen. */
const NARROW_OK = { maxWidth: '100%', minWidth: 0 } as const;
/** A table that scrolls sideways on a narrow screen instead of running out of its box. */
const Wide = ({ children, style }: { children: ReactNode; style?: CSSProperties }) => (
  <div style={{ overflowX: 'auto', maxWidth: '100%', ...style }}>{children}</div>
);
/** Where the chosen width comes from, in one sentence. */
const widthSays = (c: WidthChoice): string =>
  c.source === 'book'
    ? tr('The width is the 32 the book reports for CUDA devices to date (pp. 74, 77).', 'Độ rộng là 32, con số sách nêu cho các thiết bị CUDA tính đến nay (pp. 74, 77).')
    : c.source === 'doc'
      ? tr("The width is the 64 that AMD's HIP documentation gives for gfx9 devices.", 'Độ rộng là 64, con số tài liệu HIP của AMD đưa ra cho các thiết bị gfx9.')
      : tr(`The width is the warpSize your device reported (${c.width}).`, `Độ rộng là warpSize mà thiết bị của bạn báo cáo (${c.width}).`);
const LEFT = 150; // room for a row's label, left of lane 0
const RIGHT = 150; // room for a row's note, right of the last lane

// ---------------------------------------------------------------- lanes, one row per instruction
// Lanes that execute are solid ink (path B a lighter ink); lanes that wait are
// hatched. No hue: colour in this course means a memory tier or a failure.
interface LaneRow {
  label: string;
  active: boolean[];
  tone?: 'a' | 'b';
  note?: string;
  cur?: boolean;
  part?: string;
}

function LaneRows({
  name, width, rows, head, headLabel, onHead, mark,
}: {
  /** What the drawing is, for a screen reader. */
  name: string;
  width: number;
  rows: LaneRow[];
  /** One entry per lane, written above the rows: what each thread wants. */
  head?: (string | number)[];
  headLabel?: string;
  /** Makes the head cells buttons. */
  onHead?: (lane: number) => void;
  /** Draws a dashed line before this lane. */
  mark?: number;
}) {
  const hatch = useId();
  const c = cellFor(width);
  const top = head ? 48 : 26;
  const rh = 17;
  const W = LEFT + width * c + RIGHT;
  const H = top + Math.max(1, rows.length) * rh + 4;
  const ticks = [...new Set([...Array.from({ length: Math.ceil(width / 8) }, (_, i) => i * 8), width - 1])];
  const laneX = (l: number) => LEFT + l * c;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={name} data-view="lanes" style={{ width: W, maxWidth: '100%', display: 'block' }}>
      <defs>
        <pattern id={hatch} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.2" />
        </pattern>
      </defs>
      <text x={LEFT - 8} y={12} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi="lane">lane</Translated></text>
      {ticks.map((i) => (
        <text key={i} x={laneX(i) + c / 2 - 0.5} y={12} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{i}</text>
      ))}
      {head ? (
        <g data-head="">
          <text x={LEFT - 8} y={33} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{headLabel}</text>
          {head.map((h, l) => (
            <g key={l}>
              {onHead ? (
                <rect x={laneX(l)} y={21} width={c - 1} height={17} rx="1.5" data-head-lane={l} fill="var(--page)" stroke="var(--ink-3)" strokeWidth="0.8"
                  onClick={() => onHead(l)} style={{ cursor: 'pointer' }} />
              ) : null}
              <text x={laneX(l) + c / 2 - 0.5} y={33} fontSize={c >= 14 ? 10 : 7.5} textAnchor="middle" fontWeight="600" pointerEvents="none" /* i18n-ignore: SVG attribute value */ style={{ fill: 'var(--ink)' }}>{h}</text>
            </g>
          ))}
        </g>
      ) : null}
      {rows.map((r, i) => {
        const y = top + i * rh;
        const count = r.active.filter(Boolean).length;
        return (
          <g key={i} data-row={i} data-count={count} data-part={r.part}>
            <text x={LEFT - 8} y={y + 11} fontSize="11" textAnchor="end" fontWeight={r.cur ? 650 : 400} style={{ fill: r.cur ? 'var(--ink)' : 'var(--ink-2)' }}>
              {r.cur ? '▸ ' : ''}{r.label}
            </text>
            {r.active.map((on, l) => (
              <g key={l}>
                <rect x={laneX(l)} y={y} width={c - 1} height={rh - 4} rx="1.5" data-lane={l} data-on={on ? '' : undefined}
                  fill={on ? (r.tone === 'b' ? 'var(--ink-2)' : 'var(--ink)') : 'var(--cell)'} stroke={on ? 'none' : 'var(--rule)'} strokeWidth="0.6" />
                {on ? null : <rect x={laneX(l)} y={y} width={c - 1} height={rh - 4} rx="1.5" fill={`url(#${hatch})`} pointerEvents="none" /* i18n-ignore: SVG attribute value */ />}
              </g>
            ))}
            {r.note ? <text x={laneX(width) + 8} y={y + 11} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{r.note}</text> : null}
          </g>
        );
      })}
      {mark != null && mark > 0 && mark < width ? (
        <line x1={laneX(mark) - 0.5} x2={laneX(mark) - 0.5} y1={16} y2={H - 2} stroke="var(--ink)" strokeWidth="1.5" strokeDasharray="4 3" pointerEvents="none" /* i18n-ignore: SVG attribute value */ />
      ) : null}
    </svg>
  );
}

const MAX_TRIPS = 10;
/** One column per lane; click a column at a height to set that lane's trip count. */
function TripEditor({ width, trips, onSet }: { width: number; trips: number[]; onSet: (lane: number, trips: number) => void }) {
  const c = cellFor(width);
  const ch = 9;
  const W = LEFT + width * c + RIGHT;
  return (
    <svg viewBox={`0 0 ${W} ${MAX_TRIPS * ch + 6}`} role="group" aria-label={tr('Trip count of every lane: click a column at a height to set it', 'Số vòng lặp của mọi lane: nhấp vào một cột ở một độ cao để đặt giá trị')} data-view="trips"
      style={{ width: W, maxWidth: '100%', display: 'block', marginBottom: '0.3rem' }}>
      <text x={LEFT - 8} y={11} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>{MAX_TRIPS} lần lặp</>}>{MAX_TRIPS} iterations</Translated></text>
      <text x={LEFT - 8} y={MAX_TRIPS * ch} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi="1 lần lặp">1 iteration</Translated></text>
      <text x={LEFT + width * c + 8} y={MAX_TRIPS * ch} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="nhấp để đặt một lane">click to set a lane</Translated></text>
      {Array.from({ length: width }, (_, l) =>
        Array.from({ length: MAX_TRIPS }, (_, r) => {
          const t = MAX_TRIPS - r;
          const on = trips[l] >= t;
          return (
            <rect key={`${l}-${t}`} x={LEFT + l * c} y={2 + r * ch} width={c - 1} height={ch - 1} rx="1" data-lane={l} data-trip={t} data-on={on ? '' : undefined}
              fill={on ? 'color-mix(in srgb, var(--ink) 55%, var(--cell))' : 'var(--cell)'} stroke="var(--rule)" strokeWidth="0.5"
              onClick={() => onSet(l, trips[l] === t ? t - 1 : t)} style={{ cursor: 'pointer' }} />
          );
        }),
      )}
    </svg>
  );
}

// ---------------------------------------------------------------- the divergence simulator
type Kind = 'below' | 'modulo' | 'paint' | 'loop';
const KINDS: { v: Kind; label: string }[] = [
  { v: 'below', label: 'if (threadIdx.x < T) A else B' },
  { v: 'modulo', label: 'if (i % m == 0) A else B' },
  { v: 'paint', get label() { return tr('if (mask[i]) A else B, with a mask you paint', 'if (mask[i]) A else B, với mặt nạ do bạn tô'); } },
  { v: 'loop', get label() { return tr('for (k = 0; k < N[i]; ++k) A, with a trip count per lane', 'for (k = 0; k < N[i]; ++k) A, mỗi lane một số vòng lặp'); } },
];
const ORDERS: { v: Order; label: string }[] = [
  { v: 'serialised', get label() { return tr('one path, then the other', 'nhánh này rồi đến nhánh kia'); } },
  { v: 'interleaved', get label() { return tr('interleaved', 'xen kẽ'); } },
];

function DivergenceSim() {
  const { reports } = useReports();
  const choices = useMemo(() => widthChoices(reports), [reports]);
  const [key, setKey] = useState('book');
  const chosen = choices.find((c) => c.key === key) ?? choices[0];
  const width = chosen.width;
  const [kind, setKind] = useState<Kind>('below');
  const [T, setT] = useState<number>(FIG49.threshold);
  const [m, setM] = useState(2);
  const [warp, setWarp] = useState(0);
  const [paint, setPaint] = useState<boolean[]>(() => maskBelow(LANES_MAX, 16));
  const [trips, setTrips] = useState<number[]>(() => fig410Trips(LANES_MAX));
  const [costA, setCostA] = useState(3);
  const [costB, setCostB] = useState(3);
  const [body, setBody] = useState(1);
  const [order, setOrder] = useState<Order>('serialised');
  /** Instructions issued so far; null means all of them. */
  const [n, setN] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);

  // The warp belongs to a block the size of exercise 1's (128 threads at the usual widths).
  const warpsInBlock = Math.max(1, Math.round(EX1.blockDim / width));
  const blockThreads = warpsInBlock * width;
  const w = Math.min(warp, warpsInBlock - 1);
  const first = w * width;
  const isLoop = kind === 'loop';
  const maskOf = (k: number): boolean[] =>
    kind === 'below' ? maskBelow(width, T, k * width) : kind === 'modulo' ? maskModulo(width, m, k * width) : Array.from({ length: width }, (_, i) => !!paint[i]);
  const taken = maskOf(w);
  const lanesTrips = Array.from({ length: width }, (_, i) => trips[i] ?? 0);
  const cond = kind === 'below' ? `threadIdx.x < ${T}` : kind === 'modulo' ? `i % ${m} == 0` : 'mask[i]';
  const program = isLoop ? loop(width, lanesTrips, body) : ifElse(width, taken, costA, costB, cond);
  const trace = run(program, order);
  const shown = n == null ? trace.spent : Math.min(n, trace.spent);
  const sel = isLoop ? null : run(select(width, costA, costB, cond));
  const st = trace.stmts[1];

  useEffect(() => {
    setN(null);
    setPlaying(false);
  }, [key, kind, T, m, warp, paint, trips, costA, costB, body, order]);
  useEffect(() => {
    if (!playing) return;
    if (shown >= trace.spent) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setN(shown + 1), 450);
    return () => window.clearTimeout(id);
  }, [playing, shown, trace.spent]);

  const label = (x: Issue): string =>
    x.part === 'A' ? tr(`A, instruction ${x.k + 1}`, `A, lệnh ${x.k + 1}`)
      : x.part === 'B' ? tr(`B, instruction ${x.k + 1}`, `B, lệnh ${x.k + 1}`)
        : x.part === 'body' ? tr(`iteration ${(x.iter ?? 0) + 1}${body > 1 ? `, instr. ${x.k + 1}` : ''}`, `lần lặp ${(x.iter ?? 0) + 1}${body > 1 ? `, lệnh ${x.k + 1}` : ''}`)
          : x.stmt === 0 ? (isLoop ? x.label : tr(`test: ${x.label}`, `phép thử: ${x.label}`))
            : x.label;
  const rows: LaneRow[] = trace.issues.slice(0, shown).map((x, i) => ({
    label: label(x),
    active: x.active,
    tone: x.part === 'B' ? 'b' : 'a',
    part: x.part,
    cur: i === shown - 1,
    note: x.count === width ? tr(`all ${width}: 100%`, `cả ${width}: 100%`) : tr(`${x.count} run, ${width - x.count} wait: ${pct(x.efficiency)}`, `${x.count} chạy, ${width - x.count} chờ: ${pct(x.efficiency)}`),
  }));
  const said = (): string => {
    if (shown === 0)
      return isLoop
        ? tr('Nothing issued yet. The first instruction reads each lane’s trip count; every lane executes it.', 'Chưa phát lệnh nào. Lệnh đầu tiên đọc số vòng lặp của từng lane; mọi lane đều thực thi nó.')
        : tr('Nothing issued yet. The first instruction is the test; every lane executes it, each on its own index.', 'Chưa phát lệnh nào. Lệnh đầu tiên là phép thử; mọi lane đều thực thi nó, mỗi lane trên chỉ số của riêng mình.');
    const x = trace.issues[shown - 1];
    const lanes = x.count === width
      ? tr(`All ${width} lanes execute it.`, `Cả ${width} lane đều thực thi nó.`)
      : tr(`${x.count} of ${width} lanes execute it and ${width - x.count} wait.`, `${x.count} trên ${width} lane thực thi nó và ${width - x.count} lane chờ.`);
    const end = shown === trace.spent
      ? tr(` That is the whole program: ${plural(trace.spent, 'warp-step')}, where the same work with no lane waiting would take ${stepsText(trace.uniform)}.`, ` Đó là toàn bộ chương trình: ${trace.spent} bước warp, trong khi cùng khối lượng việc mà không lane nào phải chờ sẽ mất ${stepsText(trace.uniform)}.`)
      : '';
    return tr(`Step ${shown} of ${trace.spent}: ${label(x)}. ${lanes}${end}`, `Bước ${shown} trên ${trace.spent}: ${label(x)}. ${lanes}${end}`);
  };
  const setLaneTrips = (lane: number, t: number) => setTrips((old) => old.map((v, i) => (i === lane ? t : v)));
  const numberField = (name: string, value: number, set: (v: number) => void, min: number, max: number) => (
    <label className="field">
      {name}
      <input type="number" min={min} max={max} step={1} value={value} aria-label={name} style={{ width: '4.6rem' }}
        onChange={(e) => {
          const v = Math.floor(Number(e.target.value));
          if (e.target.value.trim() !== '' && Number.isFinite(v)) set(Math.max(min, Math.min(max, v)));
        }} />
    </label>
  );

  return (
    <Bench
      title={tr('Divergence simulator', 'Trình mô phỏng phân kỳ')}
      task={tr('Choose a statement and predict the passes before you look at them.', 'Chọn một câu lệnh và dự đoán các lượt trước khi xem chúng.')}
      foot={
        <Translated
          vi={
            <>
              Một hàng là một lệnh được phát cho warp. Các lane tô đặc thực thi nó; các lane gạch chéo phải chờ. Màu đặc nhạt hơn là nhánh B.
              Warp này là một trong {warpsInBlock} warp của một khối {blockThreads} luồng. {widthSays(chosen)}
            </>
          }
        >
          <>
            One row is one instruction issued to the warp. Solid lanes execute it; hatched lanes wait. A lighter solid is path B. The
            warp is one of the {warpsInBlock} warps of a {blockThreads}-thread block. {widthSays(chosen)}
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field" style={NARROW_OK}>
          <Translated vi="Câu lệnh">Statement</Translated>
          <select value={kind} aria-label={tr('Statement', 'Câu lệnh')} onChange={(e) => setKind(e.target.value as Kind)}>
            {KINDS.map((k) => <option key={k.v} value={k.v}>{k.label}</option>)}
          </select>
        </label>
        <label className="field" style={NARROW_OK}>
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={key} aria-label={tr('Warp width', 'Độ rộng warp')} onChange={(e) => setKey(e.target.value)}>
            {choices.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        {kind === 'below' ? (
          <label className="field">
            <span>T = <b data-out="T">{T}</b></span>
            <input type="range" min={0} max={blockThreads} step={1} value={Math.min(T, blockThreads)} aria-label={tr('Threshold T', 'Ngưỡng T')} style={{ width: '16rem' }}
              onChange={(e) => setT(Number(e.target.value))} />
          </label>
        ) : null}
        {kind === 'modulo' ? numberField('m', m, setM, 2, 16) : null}
        {isLoop ? numberField(tr('Instructions in the loop body', 'Số lệnh trong thân vòng lặp'), body, setBody, 1, 3) : numberField(tr('Instructions in A', 'Số lệnh trong A'), costA, setCostA, 1, 8)}
        {isLoop ? null : numberField(tr('Instructions in B', 'Số lệnh trong B'), costB, setCostB, 0, 8)}
        {isLoop ? null : (
          <div style={FIELD}>
            <Translated vi="Thứ tự các lượt">Order of the passes</Translated>
            <span className="seg" role="group" aria-label={tr('Order of the passes', 'Thứ tự các lượt')}>
              {ORDERS.map((o) => <button key={o.v} aria-pressed={order === o.v} onClick={() => setOrder(o.v)}>{o.label}</button>)}
            </span>
          </div>
        )}
      </div>
      {kind === 'below' || kind === 'modulo' ? (
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <span className="muted"><Translated vi="Warp của khối">Warp of the block</Translated></span>
          <span className="seg" role="group" aria-label={tr('Warp of the block', 'Warp của khối')}>
            {Array.from({ length: warpsInBlock }, (_, k) => {
              const a = countTaken(maskOf(k));
              return (
                <button key={k} aria-pressed={k === w} data-warp-summary={k} data-a={a} onClick={() => setWarp(k)}>
                  <Translated vi={<>warp {k}: {a === width ? 'toàn A' : a === 0 ? 'toàn B' : `${a} A, ${width - a} B`}</>}>
                    warp {k}: {a === width ? 'all A' : a === 0 ? 'all B' : `${a} A, ${width - a} B`}
                  </Translated>
                </button>
              );
            })}
          </span>
        </div>
      ) : null}
      {kind === 'paint' ? (
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <span className="muted"><Translated vi="Nhấp vào chữ cái của một lane để đảo nó, hoặc bắt đầu từ">Click a lane's letter to flip it, or start from</Translated></span>
          <span className="seg" role="group" aria-label={tr('Mask presets', 'Mặt nạ có sẵn')}>
            <button onClick={() => setPaint(maskBelow(LANES_MAX, width / 2))}><Translated vi="nửa đầu là A">first half A</Translated></button>
            <button onClick={() => setPaint(maskModulo(LANES_MAX, 2))}><Translated vi="cách một lane">every other lane</Translated></button>
            <button onClick={() => setPaint(maskBelow(LANES_MAX, 1))}><Translated vi="một lane A">one lane A</Translated></button>
            <button onClick={() => setPaint(maskBelow(LANES_MAX, LANES_MAX))}><Translated vi="toàn A">all A</Translated></button>
          </span>
        </div>
      ) : null}
      {isLoop ? (
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <span className="muted"><Translated vi="Số vòng lặp">Trip counts</Translated></span>
          <span className="seg" role="group" aria-label={tr('Trip count presets', 'Số vòng lặp có sẵn')}>
            <button onClick={() => setTrips(fig410Trips(LANES_MAX))}><Translated vi="giá trị in trong Hình 4.10, lặp lại">Fig. 4.10's printed values, repeated</Translated></button>
            <button onClick={() => setTrips([...sortedTrips(lanesTrips), ...Array<number>(LANES_MAX - width).fill(0)])}><Translated vi="cùng các số đó, đã sắp xếp">the same counts, sorted</Translated></button>
            <button onClick={() => setTrips(Array<number>(LANES_MAX).fill(6))}><Translated vi="tất cả là 6">all 6</Translated></button>
            <button onClick={() => setTrips(Array.from({ length: LANES_MAX }, (_, i) => (i === 0 ? MAX_TRIPS : 2)))}><Translated vi={<>một lane {MAX_TRIPS}, còn lại 2</>}>one lane of {MAX_TRIPS}, the rest 2</Translated></button>
          </span>
        </div>
      ) : null}

      <div className="row" style={{ marginBottom: '0.8rem' }} data-readout="trace">
        {isLoop ? (
          <Translated
            vi={
              <span>
                <b data-out="passes">{st.passes}</b> lần lặp: <b data-out="uniform-iterations">{st.uniformIterations}</b> với đủ mọi lane,{' '}
                <b data-out="divergent-iterations">{st.divergentIterations}</b> có lane phải chờ
              </span>
            }
          >
            <span>
              <b data-out="passes">{st.passes}</b> iterations: <b data-out="uniform-iterations">{st.uniformIterations}</b> with every lane,{' '}
              <b data-out="divergent-iterations">{st.divergentIterations}</b> with lanes waiting
            </span>
          </Translated>
        ) : (
          <Translated vi={<span><b data-out="passes">{st.passes}</b> lượt qua nhánh</span>}>
            <span><b data-out="passes">{st.passes}</b> {st.passes === 1 ? 'pass' : 'passes'} over the branch</span>
          </Translated>
        )}
        <Translated vi={<span><b data-out="spent">{trace.spent}</b> bước warp đã tiêu</span>}><span><b data-out="spent">{trace.spent}</b> warp-steps spent</span></Translated>
        <Translated vi={<span><b data-out="uniform">{stepsText(trace.uniform)}</b> nếu không lane nào phải chờ</span>}><span><b data-out="uniform">{stepsText(trace.uniform)}</b> if no lane ever waited</span></Translated>
        <Translated vi={<span><b data-out="overhead">{fmt(trace.overhead)}</b>× chi phí đó</span>}><span><b data-out="overhead">{fmt(trace.overhead)}</b>× that cost</span></Translated>
        <Translated vi={<span><b data-out="idle">{trace.idle}</b> bước lane rảnh</span>}><span><b data-out="idle">{trace.idle}</b> idle lane-steps</span></Translated>
        <Translated vi={<span><b data-out="efficiency">{pct(trace.efficiency)}</b> số bước lane có làm việc</span>}><span><b data-out="efficiency">{pct(trace.efficiency)}</b> of lane-steps did work</span></Translated>
      </div>
      {sel ? (
        <p className="muted" data-readout="select" style={{ margin: '0 0 0.8rem' }}>
          <Translated
            vi={
              <>
                Tính cả hai rồi chọn, để so sánh: <b data-out="select-spent">{sel.spent}</b> bước warp, không lane nào chờ,{' '}
                <b data-out="select-discarded">{discardedBySelect(taken, costA, costB)}</b> bước lane tính toán bị tính ra rồi bỏ đi.
              </>
            }
          >
            Compute both and select, for comparison: <b data-out="select-spent">{sel.spent}</b> warp-steps, no lane waits,{' '}
            <b data-out="select-discarded">{discardedBySelect(taken, costA, costB)}</b> lane-steps of arithmetic computed and thrown away.
          </Translated>
        </p>
      ) : null}

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setN(0); }} disabled={shown === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => { setPlaying(false); setN(shown - 1); }} disabled={shown === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => { setPlaying(false); setN(shown + 1); }} disabled={shown >= trace.spent}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => { if (shown >= trace.spent) setN(0); setPlaying(!playing); }}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Chạy')}</button>
        <button onClick={() => { setPlaying(false); setN(null); }} disabled={shown >= trace.spent}><Translated vi="Hiện tất cả">Show all</Translated></button>
        <span className="muted" aria-live="polite"><Translated vi={<>đã phát {shown} trên {trace.spent} lệnh</>}>{shown} of {trace.spent} instructions issued</Translated></span>
      </div>
      <div className="narrate" role="status" data-readout="step" style={{ marginBottom: '0.9rem' }}>{said()}</div>
      {isLoop ? <TripEditor width={width} trips={lanesTrips} onSet={setLaneTrips} /> : null}
      <LaneRows
        name={tr('The instructions issued to the warp, one row each, one cell per lane', 'Các lệnh được phát cho warp, mỗi lệnh một hàng, mỗi lane một ô')}
        width={width}
        rows={rows}
        head={isLoop ? lanesTrips : taken.map((a) => (a ? 'A' : 'B'))}
        headLabel={isLoop ? tr('trip count N', 'số vòng lặp N') : tr('wants', 'muốn')}
        onHead={kind === 'paint' ? (lane) => setPaint((old) => old.map((v, i) => (i === lane ? !v : v))) : undefined}
        mark={kind === 'below' ? T - first : undefined}
      />
    </Bench>
  );
}

// ---------------------------------------------------------------- many warps of loops
const TRIP_LO = 1;
const TRIP_HI = 64;

function BatchView({ batch, name }: { batch: LoopBatch; name: string }) {
  const hatch = useId();
  const VW = 760;
  const H = 84;
  const top = 4;
  const gap = 6;
  const count = batch.warps.length;
  const ww = (VW - gap * (count - 1)) / count;
  const lw = ww / batch.width;
  const s = H / TRIP_HI;
  return (
    <svg viewBox={`0 0 ${VW} ${H + top + 16}`} role="img" aria-label={name} style={{ width: VW, maxWidth: '100%', display: 'block' }}>
      <defs>
        <pattern id={hatch} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.2" />
        </pattern>
      </defs>
      {batch.warps.map((wp, k) => {
        const x = k * (ww + gap);
        return (
          <g key={k} data-batch-warp={k} data-longest={wp.longest}>
            <rect x={x} y={top + H - wp.longest * s} width={ww} height={wp.longest * s} fill="var(--cell)" stroke="var(--rule)" strokeWidth="0.6" />
            <rect x={x} y={top + H - wp.longest * s} width={ww} height={wp.longest * s} fill={`url(#${hatch})`} />
            {wp.trips.map((t, l) => (
              <rect key={l} x={x + l * lw} y={top + H - t * s} width={Math.max(0.5, lw - (lw > 3 ? 0.6 : 0))} height={t * s} fill="var(--ink)" />
            ))}
            <text x={x + ww / 2} y={top + H + 12} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
              {count <= 8 ? tr(`warp ${k}: `, `warp ${k}: `) : ''}{wp.longest}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function LengthsBatch() {
  const { reports } = useReports();
  const choices = useMemo(() => widthChoices(reports), [reports]);
  const [key, setKey] = useState('book');
  const chosen = choices.find((c) => c.key === key) ?? choices[0];
  const width = chosen.width;
  const [warps, setWarps] = useState(8);
  const [seed, setSeed] = useState(7);
  const trips = useMemo(() => shuffledTrips(warps * width, TRIP_LO, TRIP_HI, seed), [warps, width, seed]);
  const shuffled = loopBatch(trips, width);
  const sorted = loopBatch(sortedTrips(trips), width);
  const row = (id: string, name: string, b: LoopBatch) => (
    <tr data-batch={id}>
      <td>{name}</td>
      <td data-out="spent">{b.spent}</td>
      <td data-out="useful">{big(b.useful)}</td>
      <td data-out="idle">{big(b.idle)}</td>
      <td data-out="efficiency">{pct(b.efficiency)}</td>
    </tr>
  );
  return (
    <Bench
      title={tr('One loop per thread, lengths from the data', 'Mỗi luồng một vòng lặp, độ dài lấy từ dữ liệu')}
      task={tr('The same lengths, in two orders. Which batch finishes first, and by how much?', 'Cùng các độ dài, theo hai thứ tự. Batch nào xong trước, và sớm hơn bao nhiêu?')}
      foot={
        <Translated
          vi={
            <>
              Mỗi nhóm cột là một warp; một cột là số vòng lặp của một luồng; vùng gạch chéo phía trên các cột là thời gian lane chờ luồng
              dài nhất của warp. Con số dưới mỗi warp là số lần lặp nó chạy. Các độ dài được rải đều từ {TRIP_LO} đến{' '}
              {TRIP_HI}. {widthSays(chosen)}
            </>
          }
        >
          <>
            Each group of bars is one warp; a bar is one thread's trip count; the hatched area above the bars is lane time spent waiting for
            the warp's longest thread. The number under a warp is how many iterations it runs. Lengths are spread evenly from {TRIP_LO} to{' '}
            {TRIP_HI}. {widthSays(chosen)}
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <div style={FIELD}>
          <Translated vi="Số warp">Warps</Translated>
          <span className="seg" role="group" aria-label={tr('Number of warps', 'Số warp')}>
            {[4, 8, 16].map((v) => <button key={v} aria-pressed={warps === v} onClick={() => setWarps(v)}>{v}</button>)}
          </span>
        </div>
        <label className="field" style={NARROW_OK}>
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={key} aria-label={tr('Warp width of the batch', 'Độ rộng warp của batch')} onChange={(e) => setKey(e.target.value)}>
            {choices.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
        <button onClick={() => setSeed(seed + 1)}><Translated vi="Xáo lại">Shuffle again</Translated></button>
        <span className="muted"><Translated vi={<>{warps * width} luồng</>}>{warps * width} threads</Translated></span>
      </div>
      <div className="mx-label"><Translated vi="Đã xáo: độ dài theo thứ tự chúng đến">Shuffled: lengths in the order they arrived</Translated></div>
      <BatchView batch={shuffled} name={tr('Trip counts of every thread, in shuffled order, grouped by warp', 'Số vòng lặp của mọi luồng, theo thứ tự đã xáo, nhóm theo warp')} />
      <div className="mx-label" style={{ marginTop: '0.8rem' }}><Translated vi="Sắp xếp theo độ dài">Sorted by length</Translated></div>
      <BatchView batch={sorted} name={tr('The same trip counts sorted by length, grouped by warp', 'Cùng các số vòng lặp đó, sắp xếp theo độ dài, nhóm theo warp')} />
      <Wide style={{ marginTop: '0.9rem' }}><table className="threads" style={{ width: 'auto' }} aria-label={tr('What each order costs', 'Chi phí của mỗi thứ tự')}>
        <thead>
          <tr>
            <th><Translated vi="Thứ tự">Order</Translated></th>
            <th><Translated vi="lần lặp warp đã tiêu">warp-iterations spent</Translated></th>
            <th><Translated vi="lần lặp lane có việc">lane-iterations of work</Translated></th>
            <th><Translated vi="lần lặp lane phải chờ">lane-iterations waiting</Translated></th>
            <th><Translated vi="tỉ lệ thời gian lane có việc">lane time that is work</Translated></th>
          </tr>
        </thead>
        <tbody>
          {row('shuffled', tr('shuffled', 'đã xáo'), shuffled)}
          {row('sorted', tr('sorted', 'đã sắp xếp'), sorted)}
        </tbody>
      </table></Wide>
      <p style={{ margin: '0.7rem 0 0' }} data-readout="batch">
        <Translated
          vi={
            <>
              Khối lượng việc như nhau, <b>{big(sorted.useful)}</b> lần lặp lane. Khi xáo, các warp chạy số lần lặp gấp{' '}
              <b data-out="ratio">{fmt(shuffled.spent / sorted.spent)}</b> lần so với khi sắp xếp. Nếu không phải chờ chút nào, sẽ chỉ mất{' '}
              <b data-out="uniform">{stepsText(sorted.uniform)}</b>.
            </>
          }
        >
          The work is the same, <b>{big(sorted.useful)}</b> lane-iterations. Shuffled, the warps run for{' '}
          <b data-out="ratio">{fmt(shuffled.spent / sorted.spent)}</b> times as many iterations as sorted. With no waiting at all it would take{' '}
          <b data-out="uniform">{stepsText(sorted.uniform)}</b>.
        </Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- exercise 1, predicted first
const E1 = exercise1(BOOK_WIDTH);
const E1_WIDE = exercise1(64);
const block0 = (k: number) => E1.detail.find((x) => x.block === 0 && x.warp === k)!;
const EX1_ID = 'ch04/4.5/p-ex1';
const EX1_GROUPS: { title: string; fields: { key: string; label: string; unit: string; model: number }[] }[] = [
  {
    get title() { return tr('Line 4, guarded by the test on line 3', 'Dòng 4, được canh bởi phép thử ở dòng 3'); },
    fields: [
      { key: 'l4-active', get label() { return tr('active warps in the grid', 'số warp hoạt động trong lưới'); }, get unit() { return tr('warps', 'warp'); }, model: E1.line4.active },
      { key: 'l4-divergent', get label() { return tr('divergent warps in the grid', 'số warp bị phân kỳ trong lưới'); }, get unit() { return tr('warps', 'warp'); }, model: E1.line4.divergent },
      { key: 'l4-eff0', get label() { return tr('SIMD efficiency of warp 0 of block 0', 'hiệu suất SIMD của warp 0 thuộc khối 0'); }, unit: '%', model: 100 * block0(0).line4.efficiency },
      { key: 'l4-eff1', get label() { return tr('SIMD efficiency of warp 1 of block 0', 'hiệu suất SIMD của warp 1 thuộc khối 0'); }, unit: '%', model: 100 * block0(1).line4.efficiency },
      { key: 'l4-eff3', get label() { return tr('SIMD efficiency of warp 3 of block 0', 'hiệu suất SIMD của warp 3 thuộc khối 0'); }, unit: '%', model: 100 * block0(3).line4.efficiency },
    ],
  },
  {
    get title() { return tr('Line 7, guarded by the test on line 6', 'Dòng 7, được canh bởi phép thử ở dòng 6'); },
    fields: [
      { key: 'l7-active', get label() { return tr('active warps in the grid', 'số warp hoạt động trong lưới'); }, get unit() { return tr('warps', 'warp'); }, model: E1.line7.active },
      { key: 'l7-divergent', get label() { return tr('divergent warps in the grid', 'số warp bị phân kỳ trong lưới'); }, get unit() { return tr('warps', 'warp'); }, model: E1.line7.divergent },
      { key: 'l7-eff0', get label() { return tr('SIMD efficiency of warp 0 of block 0', 'hiệu suất SIMD của warp 0 thuộc khối 0'); }, unit: '%', model: 100 * block0(0).line7.efficiency },
    ],
  },
  {
    get title() { return tr('The loop on line 9', 'Vòng lặp ở dòng 9'); },
    fields: [
      { key: 'loop-uniform', get label() { return tr('iterations with no divergence', 'số lần lặp không phân kỳ'); }, get unit() { return tr('iterations', 'lần lặp'); }, model: E1.loop.uniform },
      { key: 'loop-divergent', get label() { return tr('iterations with divergence', 'số lần lặp có phân kỳ'); }, get unit() { return tr('iterations', 'lần lặp'); }, model: E1.loop.divergent },
    ],
  },
];
const EX1_FIELDS = EX1_GROUPS.flatMap((g) => g.fields);
const near = (a: number, b: number) => Math.abs(a - b) < 0.5;

function Ex1Reveal({ mine }: { mine: number[] }) {
  const right = EX1_FIELDS.filter((f, i) => near(mine[i], f.model)).length;
  const b0 = [0, 1, 2, 3].map(block0);
  const statementRows = (pick: 'line4' | 'line7'): LaneRow[] =>
    b0.map((x) => {
      const s = x[pick];
      return {
        label: tr(`warp ${x.warp}: threads ${x.first} to ${x.first + x.real - 1}`, `warp ${x.warp}: luồng ${x.first} đến ${x.first + x.real - 1}`),
        active: s.mask,
        note: tr(
          `${s.count} of ${E1.width}: ${pct(s.efficiency)}${s.divergent ? ', divergent' : s.active ? '' : ', not active'}`,
          `${s.count} trên ${E1.width}: ${pct(s.efficiency)}${s.divergent ? ', bị phân kỳ' : s.active ? '' : ', không hoạt động'}`,
        ),
      };
    });
  const w0 = block0(0);
  return (
    <div data-reveal={EX1_ID}>
      <Translated
        vi={
          <p>
            {right === EX1_FIELDS.length ? 'Cả mười số đều khớp với mô hình.' : `${right} trên ${EX1_FIELDS.length} số khớp với mô hình.`} Sách không in
            đáp án; đây là các số đếm của mô hình mà trang này chạy, theo các định nghĩa đã nêu ở trên.
          </p>
        }
      >
        <p>
          {right === EX1_FIELDS.length ? 'All ten match the model.' : `${right} of ${EX1_FIELDS.length} match the model.`} The book prints no
          answers; these are the counts of the model this page runs on, with the definitions given above.
        </p>
      </Translated>
      <Wide style={{ marginBottom: '0.9rem' }}><table className="threads" style={{ width: 'auto' }} aria-label={tr("Your entries beside the model's counts", 'Số bạn nhập bên cạnh số đếm của mô hình')}>
        <thead>
          <tr>
            <th><Translated vi="Câu lệnh">Statement</Translated></th>
            <th><Translated vi="Đại lượng">Quantity</Translated></th>
            <th><Translated vi="Bạn">You</Translated></th>
            <th><Translated vi="Mô hình">Model</Translated></th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {EX1_GROUPS.flatMap((g) =>
            g.fields.map((f) => {
              const i = EX1_FIELDS.indexOf(f);
              const ok = near(mine[i], f.model);
              return (
                <tr key={f.key} data-field={f.key} data-ok={ok ? 'yes' : 'no'}>
                  <td>{g.title.split(',')[0]}</td>
                  <td>{f.label}</td>
                  <td>{mine[i]}{f.unit === '%' ? '%' : ''}</td>
                  <td data-out="model">{f.model}{f.unit === '%' ? '%' : ''}</td>
                  <td className={ok ? 't-ok' : 't-fail'}>{ok ? tr('matches', 'khớp') : tr('differs', 'khác')}</td>
                </tr>
              );
            }),
          )}
        </tbody>
      </table></Wide>
      <div className="mx-label">
        <Translated vi={<>Dòng 4 trong khối 0: những lane nào thực thi <code>b[i] = a[i] + 1</code></>}>
          Line 4 in block 0: which lanes execute <code>b[i] = a[i] + 1</code>
        </Translated>
      </div>
      <LaneRows name={tr('Line 4 in the four warps of block 0', 'Dòng 4 trong bốn warp của khối 0')} width={E1.width} rows={statementRows('line4')} />
      <Translated
        vi={
          <p>
            Phép thử đúng với các luồng 0 đến 39 và 104 đến 127 của mọi khối. Warp 0 nằm trọn trong khoảng thứ nhất, warp 2 nằm trọn ngoài
            cả hai, còn warp 1 và 3 bị cắt. Vậy mỗi khối có ba warp hoạt động, 3 · 8 = {E1.line4.active}, và hai warp bị phân kỳ, 2
            · 8 = {E1.line4.divergent}. Mọi khối đều giống nhau vì phép thử dùng <code>threadIdx.x</code>, không phải <code>i</code>.
          </p>
        }
      >
        <p>
          The test passes for threads 0 to 39 and 104 to 127 of every block. Warp 0 lies wholly inside the first range, warp 2 wholly outside
          both, and warps 1 and 3 are cut. So three warps of each block are active, 3 · 8 = {E1.line4.active}, and two of each are divergent, 2
          · 8 = {E1.line4.divergent}. Every block is the same because the test uses <code>threadIdx.x</code>, not <code>i</code>.
        </p>
      </Translated>
      <div className="mx-label">
        <Translated vi={<>Dòng 7 trong khối 0: những lane nào thực thi <code>a[i] = b[i]*2</code></>}>
          Line 7 in block 0: which lanes execute <code>a[i] = b[i]*2</code>
        </Translated>
      </div>
      <LaneRows name={tr('Line 7 in the four warps of block 0', 'Dòng 7 trong bốn warp của khối 0')} width={E1.width} rows={statementRows('line7')} />
      <Translated
        vi={
          <p>
            <code>i</code> chẵn và lẻ xen kẽ nhau từng lane, nên mỗi warp trong {E1.warps} warp đều chứa cả hai nhánh và chạy ở mức{' '}
            {pct(w0.line7.efficiency)}. Một nửa số lane đi vào nhánh, và không cách bố trí lần khởi chạy nào làm warp đồng nhất được: đây là
            mẫu tệ nhất có thể, và nó tốn đúng bằng một lần chia 31 đối 1.
          </p>
        }
      >
        <p>
          Even and odd <code>i</code> alternate lane by lane, so every one of the {E1.warps} warps has both paths in it and runs at{' '}
          {pct(w0.line7.efficiency)}. Half the lanes take the branch, and no arrangement of the launch can make a warp uniform: this is the worst
          pattern there is, and it costs exactly what a 31 to 1 split costs.
        </p>
      </Translated>
      <div className="mx-label"><Translated vi="Vòng lặp trong warp 0 của khối 0: mỗi lần lặp một hàng">The loop in warp 0 of block 0: one row per iteration</Translated></div>
      <LaneRows
        name={tr('The loop of line 9 in warp 0 of block 0, one row per iteration', 'Vòng lặp ở dòng 9 trong warp 0 của khối 0, mỗi lần lặp một hàng')}
        width={E1.width}
        head={w0.trips}
        headLabel={tr('trip count', 'số vòng lặp')}
        rows={loopRows(w0.trips).map((active, j) => {
          const c = countTaken(active);
          return { label: `j = ${j}`, active, note: c === E1.width ? tr(`all ${E1.width}`, `cả ${E1.width}`) : tr(`${c} run, ${E1.width - c} wait`, `${c} chạy, ${E1.width - c} chờ`) };
        })}
      />
      <Translated
        vi={
          <p>
            Số vòng lặp là 5, 4 và 3 khi <code>i % 3</code> bằng 0, 1 và 2, và warp nào cũng chứa cả ba. Các lần lặp <code>j</code> = 0, 1 và 2
            được mọi lane thực thi; ở <code>j</code> = 3 một phần ba số lane đã xong, và ở <code>j</code> = 4 là hai phần ba. Tức là
            {' '}{E1.loop.uniform} lần lặp không phân kỳ và {E1.loop.divergent} lần có phân kỳ.
          </p>
        }
      >
        <p>
          Trip counts are 5, 4 and 3 as <code>i % 3</code> is 0, 1 and 2, and every warp holds all three. Iterations <code>j</code> = 0, 1 and 2
          are executed by every lane; in <code>j</code> = 3 a third of the lanes have finished, and in <code>j</code> = 4 two thirds have. That
          is {E1.loop.uniform} iterations without divergence and {E1.loop.divergent} with.
        </p>
      </Translated>
      <Translated
        vi={
          <p className="muted">
            Bài tập giả định warp rộng 32. Ở độ rộng 64, cùng lần khởi chạy đó có {E1_WIDE.warps} warp, và dòng 4 chia tách cả {E1_WIDE.line4.divergent}
            {' '}warp: {E1_WIDE.detail[0].line4.count} trên 64 lane thực thi nó ở warp đầu của một khối và {E1_WIDE.detail[1].line4.count} trên 64 ở
            warp thứ hai.
          </p>
        }
      >
        <p className="muted">
          The exercise assumes 32-wide warps. At 64 the same launch has {E1_WIDE.warps} warps, and line 4 splits all {E1_WIDE.line4.divergent} of
          them: {E1_WIDE.detail[0].line4.count} of 64 lanes execute it in the first warp of a block and {E1_WIDE.detail[1].line4.count} of 64 in
          the second.
        </p>
      </Translated>
    </div>
  );
}

function Ex1Predict() {
  const { progress, answer } = useProgress();
  const prior = progress.answers[EX1_ID]?.value;
  const committed =
    Array.isArray(prior) && prior.length === EX1_FIELDS.length && prior.every((v) => typeof v === 'number') ? (prior as number[]) : null;
  const [text, setText] = useState<Record<string, string>>({});
  const values = EX1_FIELDS.map((f) => (text[f.key]?.trim() ? Number(text[f.key]) : NaN));
  const ready = values.every((v) => Number.isFinite(v));
  return (
    <div className="q" data-predict={EX1_ID}>
      <div className="kind"><Translated vi="Dự đoán trước">Predict first</Translated></div>
      <div className="prompt">
        <Translated
          vi={
            <>
              Bài tập 1 của sách, phần c, d và e, cho kernel và lần khởi chạy ở trên trên một thiết bị có warp rộng 32. Điền đủ cả mười số trước
              khi xem: phần đáp án vẽ bốn warp của khối 0 cho từng câu lệnh.
            </>
          }
        >
          The book's exercise 1, parts c, d and e, for the kernel and launch above on a device with 32-wide warps. Fill in all ten before you
          look: the reveal draws block 0's four warps for each statement.
        </Translated>
      </div>
      {committed == null ? (
        <>
          {EX1_GROUPS.map((g) => (
            <div key={g.title} style={{ marginBottom: '0.8rem' }}>
              <div className="mx-label">{g.title}</div>
              {g.fields.map((f) => (
                <div className="row" key={f.key} style={{ marginBottom: '0.3rem', gap: '0.2rem 0.6rem' }}>
                  <span style={{ minWidth: '17rem' }}>{f.label}</span>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
                    <input type="number" step="any" value={text[f.key] ?? ''} aria-label={`${g.title}: ${f.label}`} data-field={f.key} style={{ width: '5.5rem' }}
                      onChange={(e) => setText({ ...text, [f.key]: e.target.value })} />
                    <span className="ui muted">{f.unit === '%' ? '%' : f.unit}</span>
                  </span>
                </div>
              ))}
            </div>
          ))}
          <button className="primary" disabled={!ready} onClick={() => ready && answer(EX1_ID, values)}><Translated vi="Chốt dự đoán">Commit prediction</Translated></button>
        </>
      ) : (
        <div className="verdict info">
          <p><b><Translated vi="Bạn đã chốt mười số.">You committed ten numbers.</Translated></b></p>
          <Ex1Reveal mine={committed} />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- the boundary-check counter
const STATE_FILL: Record<WarpState, string> = {
  full: 'color-mix(in srgb, var(--ink) 24%, var(--cell))', // i18n-ignore: CSS colour
  divergent: 'var(--ink)',
  idle: 'var(--cell)',
};
const STATE_WORDS: Record<WarpState, string> = {
  get full() { return tr('every thread passes the test', 'mọi luồng đều qua phép thử'); },
  get divergent() { return tr('split: some threads pass and some do not', 'bị tách: một số luồng qua, một số không'); },
  get idle() { return tr('no thread passes: launched, and idle', 'không luồng nào qua: được khởi chạy, và rảnh'); },
};
const MAX_DRAWN_BLOCKS = 400;
const MAX_DRAWN_WARPS_1D = 640;

/** Every warp of a launch, drawn in place: bands inside each block (2D), or cells in a line (1D). */
function LaunchMap({ p, states, oneD }: { p: Picture; states: WarpState[][][]; oneD: boolean }) {
  const hatch = useId();
  const gridY = states.length;
  const gridX = states[0]?.length ?? 0;
  const perBlock = states[0]?.[0]?.length ?? 0;
  const defs = (
    <defs>
      <pattern id={hatch} width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <line x1="0" y1="0" x2="0" y2="4" stroke="var(--ink-3)" strokeWidth="1" />
      </pattern>
    </defs>
  );
  const cell = (key: string, x: number, y: number, w: number, h: number, s: WarpState, attrs: Record<string, number>) => (
    <g key={key}>
      <rect x={x} y={y} width={w} height={h} data-state={s} {...Object.fromEntries(Object.entries(attrs).map(([k, v]) => [`data-${k}`, v]))} /* i18n-ignore: data attribute name */
        fill={STATE_FILL[s]} stroke="var(--rule)" strokeWidth="0.5" />
      {s === 'idle' ? <rect x={x} y={y} width={w} height={h} fill={`url(#${hatch})`} pointerEvents="none" /* i18n-ignore: SVG attribute value */ /> : null}
    </g>
  );
  if (oneD) {
    const roomy = gridX * perBlock <= 64;
    const c = roomy ? 18 : 11;
    const h = roomy ? 22 : 16;
    const gap = 5;
    const perLine = Math.max(1, Math.floor(760 / (perBlock * c + gap)));
    const lines = Math.ceil(gridX / perLine);
    const W = Math.min(gridX, perLine) * (perBlock * c + gap);
    return (
      <svg viewBox={`0 0 ${W} ${lines * (h + 6) + 2}`} role="img" aria-label={tr('Every warp of the launch in order, block by block', 'Mọi warp của lần khởi chạy theo thứ tự, từng khối một')} data-view="launch"
        style={{ width: W, maxWidth: '100%', display: 'block' }}>
        {defs}
        {states[0].map((block, bx) => {
          const x0 = (bx % perLine) * (perBlock * c + gap);
          const y0 = Math.floor(bx / perLine) * (h + 6) + 1;
          return block.map((s, k) => cell(`${bx}-${k}`, x0 + k * c, y0, c - 1, h, s, { block: bx, warp: k }));
        })}
      </svg>
    );
  }
  const S = Math.max(12, Math.min(84, Math.floor(700 / gridX)));
  const band = (S - 2) / perBlock;
  const ex = (p.width / p.blockX) * S;
  const ey = (p.height / p.blockY) * S;
  const W = gridX * S + 2;
  const H = gridY * S + 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Every block of the launch, each drawn as its warps from top to bottom, with the edge of the picture', 'Mọi khối của lần khởi chạy, mỗi khối vẽ thành các warp của nó từ trên xuống, cùng với mép của bức ảnh')} data-view="launch"
      style={{ width: W, maxWidth: '100%', display: 'block' }}>
      {defs}
      {states.map((line, by) =>
        line.map((block, bx) => (
          <g key={`${bx}-${by}`}>
            {block.map((s, k) => cell(`${k}`, 1 + bx * S + 0.5, 1 + by * S + 0.5 + k * band, S - 2, band, s, { bx, by, warp: k }))}
            <rect x={1 + bx * S} y={1 + by * S} width={S - 1} height={S - 1} fill="none" stroke="var(--ink-3)" strokeWidth="0.8" />
          </g>
        )),
      )}
      <path d={`M ${1 + ex} 0 V ${Math.min(H, 1 + ey)} M 0 ${1 + ey} H ${Math.min(W, 1 + ex)}`} fill="none" stroke="var(--page)" strokeWidth="3.5" />
      <path d={`M ${1 + ex} 0 V ${Math.min(H, 1 + ey)} M 0 ${1 + ey} H ${Math.min(W, 1 + ex)}`} fill="none" stroke="var(--ink)" strokeWidth="1.5" strokeDasharray="5 3" />
    </svg>
  );
}

const WIDE = BOOK_CASES.find((c) => c.id === 'p200x150')!.picture;
const TALL = BOOK_CASES.find((c) => c.id === 'p150x200')!.picture;
const WIDE_R = regions(WIDE, BOOK_WIDTH);
const HUNT_PRINTED = 80;
const HUNT: { label: string; count: number }[] = [
  { get label() { return tr('Warps the test really splits, with 200 as the width and 150 as the height.', 'Số warp phép thử thực sự chia tách, với chiều rộng 200 và chiều cao 150.'); }, count: boundary2D(WIDE, BOOK_WIDTH).divergent },
  { get label() { return tr('Warps the test really splits, with 150 as the width and 200 as the height.', 'Số warp phép thử thực sự chia tách, với chiều rộng 150 và chiều cao 200.'); }, count: boundary2D(TALL, BOOK_WIDTH).divergent },
  { get label() { return tr('Every warp of every block the right edge crosses, split or not, with 200 as the width.', 'Mọi warp của mọi khối mà mép phải cắt qua, dù bị tách hay không, với chiều rộng 200.'); }, count: warpsInRightEdgeBlocks(WIDE, BOOK_WIDTH) },
  { get label() { return tr('Every warp of every block the right edge crosses, split or not, with 150 as the width.', 'Mọi warp của mọi khối mà mép phải cắt qua, dù bị tách hay không, với chiều rộng 150.'); }, count: warpsInRightEdgeBlocks(TALL, BOOK_WIDTH) },
  { get label() { return tr('Warps the test really splits on a device with 64-wide warps, with 200 as the width.', 'Số warp phép thử thực sự chia tách trên thiết bị có warp rộng 64, với chiều rộng 200.'); }, count: boundary2D(WIDE, 64).divergent },
];
const isInt = (v: number) => Number.isInteger(v) && v >= 1;

function BoundaryCounter() {
  const { reports } = useReports();
  const choices = useMemo(() => widthChoices(reports), [reports]);
  const [key, setKey] = useState('book');
  const chosen = choices.find((c) => c.key === key) ?? choices[0];
  const width = chosen.width;
  const [tab, setTab] = useState<'1d' | '2d'>('1d');
  const [v1, setV1] = useState({ n: 1003, block: 64 });
  const [v2, setV2] = useState<Picture>({ width: 76, height: 62, blockX: 16, blockY: 16 });
  const [huntPick, setHuntPick] = useState<number | null>(null);
  const [huntChecked, setHuntChecked] = useState(false);

  const oneD = tab === '1d';
  const p: Picture = oneD ? { width: v1.n, height: 1, blockX: v1.block, blockY: 1 } : v2;
  const valid = [p.width, p.height, p.blockX, p.blockY].every(isInt);
  const over = valid && p.blockX * p.blockY > chosen.maxThreads;
  const b = valid && !over ? boundary2D(p, width) : null;
  const drawable = b != null && (oneD ? b.warps <= MAX_DRAWN_WARPS_1D && b.warpsPerBlock <= 60 : b.blocks <= MAX_DRAWN_BLOCKS && b.warpsPerBlock <= 32);
  const states = useMemo(() => {
    if (!b || !drawable) return null;
    const g = gridFor(p);
    return classifyGrid(g.gridX, g.gridY, dims(p.blockX, p.blockY), width, (col, row) => col < p.width && row < p.height, true).states;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [p.width, p.height, p.blockX, p.blockY, width, drawable]);
  const split = oneD && b ? splitWarp1D(v1.n, v1.block, width) : null;

  const field = (name: string, value: number, set: (v: number) => void) => (
    <label className="field">
      {name}
      <input type="number" min={1} step={1} value={value || ''} aria-label={name} style={{ width: '6.4rem' }}
        onChange={(e) => set(e.target.value.trim() === '' ? 0 : Math.floor(Number(e.target.value)))} />
    </label>
  );
  const load = (c: Picture) => {
    if (c.height === 1 && c.blockY === 1) {
      setTab('1d');
      setV1({ n: c.width, block: c.blockX });
    } else {
      setTab('2d');
      setV2(c);
    }
  };

  return (
    <Bench
      title={tr('Boundary-check counter', 'Bộ đếm kiểm tra biên')}
      task={tr('Set a size and a block. How many warps does the test split, and which ones?', 'Đặt một kích thước và một khối. Phép thử chia tách bao nhiêu warp, và những warp nào?')}
      foot={
        <Translated
          vi={
            <>
              Mỗi phần tử một luồng, với điều kiện canh thông thường: <code>i &lt; n</code> ở một chiều, <code>col &lt; width &amp;&amp; row &lt; height</code>{' '}
              ở hai chiều. Một warp bị tách khi một số luồng của nó qua điều kiện canh và một số thì không. {chosen.note}
            </>
          }
        >
          <>
            One thread per element, and the usual guard: <code>i &lt; n</code> in one dimension, <code>col &lt; width &amp;&amp; row &lt; height</code>{' '}
            in two. A warp is split when some of its threads pass the guard and some do not. {chosen.note}
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span className="seg" role="group" aria-label={tr('Dimensions', 'Số chiều')}>
          <button aria-pressed={oneD} onClick={() => setTab('1d')}><Translated vi="1D: một vector">1D: a vector</Translated></button>
          <button aria-pressed={!oneD} onClick={() => setTab('2d')}><Translated vi="2D: một bức ảnh hoặc một ma trận">2D: a picture or a matrix</Translated></button>
        </span>
        {oneD ? (
          <>
            {field(tr('Elements n', 'Số phần tử n'), v1.n, (v) => setV1({ ...v1, n: v }))}
            {field(tr('Threads in a block', 'Số luồng trong một khối'), v1.block, (v) => setV1({ ...v1, block: v }))}
          </>
        ) : (
          <>
            {field(tr('Width (columns)', 'Chiều rộng (cột)'), v2.width, (v) => setV2({ ...v2, width: v }))}
            {field(tr('Height (rows)', 'Chiều cao (hàng)'), v2.height, (v) => setV2({ ...v2, height: v }))}
            {field('blockDim.x', v2.blockX, (v) => setV2({ ...v2, blockX: v }))}
            {field('blockDim.y', v2.blockY, (v) => setV2({ ...v2, blockY: v }))}
          </>
        )}
        <label className="field" style={NARROW_OK}>
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={key} aria-label={tr('Warp width of the launch', 'Độ rộng warp của lần khởi chạy')} onChange={(e) => setKey(e.target.value)}>
            {choices.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
      </div>

      {!valid ? (
        <p><span className="chip fail"><Translated vi="Mọi kích thước đều là số nguyên, từ 1 trở lên.">Every size is a whole number, 1 or more.</Translated></span></p>
      ) : over ? (
        <Translated
          vi={
            <p data-over="">
              <span className="chip fail">{p.blockX * p.blockY} luồng trong một khối</span> là nhiều hơn mức {chosen.maxThreads} mà lựa chọn này cho phép,
              nên lần khởi chạy sẽ bị từ chối.
            </p>
          }
        >
          <p data-over="">
            <span className="chip fail">{p.blockX * p.blockY} threads in a block</span> is more than the {chosen.maxThreads} this choice allows, so the
            launch would be refused.
          </p>
        </Translated>
      ) : b ? (
        <>
          <div className="row" style={{ marginBottom: '0.7rem' }} data-readout="boundary">
            <Translated vi={<span><b data-out="blocks">{big(b.blocks)}</b> khối{oneD ? '' : ` (${b.gridX} theo chiều ngang, ${b.gridY} theo chiều dọc)`}</span>}>
              <span><b data-out="blocks">{big(b.blocks)}</b> {b.blocks === 1 ? 'block' : 'blocks'}{oneD ? '' : ` (${b.gridX} across, ${b.gridY} down)`}</span>
            </Translated>
            <Translated vi={<span><b data-out="threads">{big(b.threads)}</b> luồng được khởi chạy</span>}><span><b data-out="threads">{big(b.threads)}</b> threads launched</span></Translated>
            <Translated vi={<span><b data-out="warps">{big(b.warps)}</b> warp</span>}><span><b data-out="warps">{big(b.warps)}</b> warps</span></Translated>
            <Translated vi={<span><b data-out="divergent">{big(b.divergent)}</b> bị tách</span>}><span><b data-out="divergent">{big(b.divergent)}</b> split</span></Translated>
            <Translated vi={<span><b data-out="idle">{big(b.idle)}</b> rảnh</span>}><span><b data-out="idle">{big(b.idle)}</b> idle</span></Translated>
            <Translated vi={<span><b data-out="full">{big(b.full)}</b> đầy đủ</span>}><span><b data-out="full">{big(b.full)}</b> full</span></Translated>
            <Translated vi={<span><b data-out="share">{pct(b.share)}</b> số warp bị tách</span>}><span><b data-out="share">{pct(b.share)}</b> of warps split</span></Translated>
          </div>
          {oneD ? (
            <div className="narrate" role="status" data-readout="split" style={{ marginBottom: '0.8rem', minHeight: '2.6em' }}>
              {split
                ? tr(
                    `The one warp that splits is warp ${split.warp} of block ${split.block}: ${plural(split.valid, 'thread passes', 'threads pass')} the test and ${split.invalid} ${split.invalid === 1 ? 'does' : 'do'} not. ${big(b.threads - v1.n)} of the ${big(b.threads)} threads launched have no element.`,
                    `Warp duy nhất bị tách là warp ${split.warp} của khối ${split.block}: ${split.valid} luồng qua phép thử và ${split.invalid} luồng thì không. ${big(b.threads - v1.n)} trên ${big(b.threads)} luồng được khởi chạy không có phần tử nào.`,
                  )
                : tr(
                    `The last element is the last lane of a warp, so nothing splits. ${b.threads - v1.n > 0 ? `The ${big(b.threads - v1.n)} threads with no element fill whole warps, which stay idle.` : 'Every thread has an element.'}`,
                    `Phần tử cuối cùng rơi vào lane cuối của một warp, nên không có gì bị tách. ${b.threads - v1.n > 0 ? `${big(b.threads - v1.n)} luồng không có phần tử lấp đầy trọn các warp, và các warp đó rảnh.` : 'Luồng nào cũng có một phần tử.'}`,
                  )}
            </div>
          ) : (
            <Wide style={{ marginBottom: '0.8rem' }}><table className="threads" style={{ width: 'auto' }} aria-label={tr("The book's four regions", 'Bốn vùng của sách')}>
              <thead>
                <tr>
                  <th><Translated vi="Vùng">Region</Translated></th>
                  <th><Translated vi="khối">blocks</Translated></th>
                  <th><Translated vi="số cột × hàng hợp lệ trong một khối">valid columns × rows in a block</Translated></th>
                  <th><Translated vi="warp">warps</Translated></th>
                  <th><Translated vi="bị tách">split</Translated></th>
                  <th><Translated vi="rảnh">idle</Translated></th>
                </tr>
              </thead>
              <tbody>
                {b.regions.map((r) => (
                  <tr key={r.id} data-region={r.id}>
                    <td>{r.id}: {r.where}</td>
                    <td data-out="blocks">{r.blocks}</td>
                    <td>{r.blocks ? `${r.validX} × ${r.validY}` : tr('none', 'không có')}</td>
                    <td data-out="warps">{r.warps}</td>
                    <td data-out="divergent">{r.divergent}</td>
                    <td data-out="idle">{r.idle}</td>
                  </tr>
                ))}
              </tbody>
            </table></Wide>
          )}
          {states ? (
            <>
              <LaunchMap p={p} states={states} oneD={oneD} />
              <div className="row muted" style={{ margin: '0.5rem 0 0', gap: '0.3rem 1.1rem' }}>
                {(['full', 'divergent', 'idle'] as WarpState[]).map((s) => (
                  <span key={s} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                    <i style={{ display: 'inline-block', width: '0.9em', height: '0.9em', background: STATE_FILL[s], border: '1px solid var(--ink-3)' }} />
                    {STATE_WORDS[s]}
                  </span>
                ))}
                {oneD
                  ? <span><Translated vi="mỗi warp một ô, nhóm theo khối">one cell per warp, grouped by block</Translated></span>
                  : <span><Translated vi="mỗi khối được vẽ thành các warp của nó, từ trên xuống; đường nét đứt là mép của dữ liệu">each block is drawn as its warps, top to bottom; the dashed line is the edge of the data</Translated></span>}
              </div>
            </>
          ) : (
            <p className="muted" data-undrawn=""><Translated vi="Quá nhiều warp để vẽ từng cái. Các số đếm ở trên là chính xác.">Too many warps to draw one by one. The counts above are exact.</Translated></p>
          )}
        </>
      ) : null}

      <div className="mx-label" style={{ marginTop: '1.1rem' }}><Translated vi="Các ví dụ của sách, ở độ rộng đã chọn phía trên">The book's examples, at the width chosen above</Translated></div>
      <Wide><table className="threads" style={{ width: 'auto' }} aria-label={tr("The book's examples: the printed figure beside the model's", 'Các ví dụ của sách: con số in trong sách bên cạnh con số của mô hình')}>
        <thead>
          <tr>
            <th><Translated vi="Trường hợp">Case</Translated></th>
            <th><Translated vi="Trang sách">Book p.</Translated></th>
            <th><Translated vi="Sách in">The book prints</Translated></th>
            <th><Translated vi="Mô hình đếm được">The model counts</Translated></th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {BOOK_CASES.map((c) => {
            const m = boundary2D(c.picture, width);
            const model = tr(`${m.divergent} of ${big(m.warps)}`, `${m.divergent} trên ${big(m.warps)}`);
            const differs = c.printed != null && width === BOOK_WIDTH && c.printed !== `${m.divergent} of ${m.warps}`;
            return (
              <tr key={c.id} data-case={c.id} data-differs={differs ? '' : undefined}>
                <td>{c.label}</td>
                <td>{c.page}</td>
                <td>{tr(c.printed ?? tr('no count (an exercise)', 'không có số (một bài tập)'), c.printed != null ? c.printed.replace(' of ', ' trên ') : 'không có số (một bài tập)')}</td>
                <td data-out="model">{model}{differs ? tr(' (the page differs)', ' (trang sách khác)') : ''}</td>
                <td><button className="quiet" onClick={() => load(c.picture)}><Translated vi="Nạp">Load</Translated></button></td>
              </tr>
            );
          })}
        </tbody>
      </table></Wide>

      <div className="q" style={{ marginTop: '1.1rem' }} data-hunt="">
        <div className="kind"><Translated vi="Truy tìm lỗi in">Erratum hunt</Translated></div>
        <div className="prompt">
          <Translated
            vi={
              <>
                Sách nói rằng với một bức ảnh “200 × 150” chỉ 80 trong 1040 warp bị phân kỳ. Hãy nạp cả hai cách đọc ở trên: không cách nào cho ra 80.
                Cách đếm nào mới cho ra con số in {HUNT_PRINTED}?
              </>
            }
          >
            The book says that for a “200 × 150” picture only 80 of the 1040 warps diverge. Load both readings above: neither count is 80. Which
            way of counting does give the printed {HUNT_PRINTED}?
          </Translated>
        </div>
        <div className="opts" role="radiogroup">
          {HUNT.map((h, i) => (
            <label key={i} className={huntChecked && h.count === HUNT_PRINTED ? 'right' : huntChecked && huntPick === i ? 'wrong' : ''}>
              <input type="radio" name="ch04-hunt-80" /* i18n-ignore: radio group name */ checked={huntPick === i} onChange={() => { setHuntPick(i); setHuntChecked(false); }} />
              <span>
                {h.label}
                {huntChecked ? (
                  <Translated vi={<span className="muted" data-hunt-count={h.count}> Số đếm đó là {h.count}.</span>}>
                    <span className="muted" data-hunt-count={h.count}> That count is {h.count}.</span>
                  </Translated>
                ) : null}
              </span>
            </label>
          ))}
        </div>
        <button className="primary" disabled={huntPick == null} onClick={() => setHuntChecked(true)}><Translated vi="Kiểm tra">Check</Translated></button>
        {huntChecked && huntPick != null ? (
          <div className={'verdict ' + (HUNT[huntPick].count === HUNT_PRINTED ? 'right' : 'wrong')}>
            <p><b>{HUNT[huntPick].count === HUNT_PRINTED ? tr('That gives 80.', 'Cách đó cho ra 80.') : tr(`That gives ${HUNT[huntPick].count}, not 80.`, `Cách đó cho ra ${HUNT[huntPick].count}, không phải 80.`)}</b></p>
            <Translated
              vi={
                <p>
                  Kích thước các vùng in trong sách (864, 72, 96 và 8 warp) là của một bức ảnh rộng 200 và cao 150. Trong bức ảnh đó, mép phải
                  cắt qua {WIDE_R[1].blocks} khối cao trọn, cả {WIDE_R[1].divergent} warp của chúng đều bị tách, và khối ở góc. Khối góc
                  có {WIDE_R[3].validY} hàng hợp lệ: {WIDE_R[3].divergent} warp của nó bị tách và {WIDE_R[3].idle} warp nằm dưới bức ảnh, rảnh.{' '}
                  {WIDE_R[1].divergent} + {WIDE_R[3].divergent} = {HUNT[0].count}. Đếm cả {WIDE_R[3].warps} warp của khối góc thì được{' '}
                  {WIDE_R[1].divergent} + {WIDE_R[3].warps} = {warpsInRightEdgeBlocks(WIDE, BOOK_WIDTH)}. Chúng tôi không thể biết các tác giả đã đếm gì; đây là cách đếm đơn giản tái tạo được con số của họ. Dù thế nào,
                  kết luận của sách vẫn đúng với cách đọc này: {HUNT[0].count} hay {HUNT_PRINTED} trên 1040 đều dưới 8%.
                </p>
              }
            >
              <p>
                The printed region sizes (864, 72, 96 and 8 warps) are those of a picture 200 wide and 150 high. In that picture the right edge
                crosses {WIDE_R[1].blocks} full-height blocks, all of whose {WIDE_R[1].divergent} warps are split, and the corner block. The corner
                has {WIDE_R[3].validY} valid rows: {WIDE_R[3].divergent} of its warps are split and {WIDE_R[3].idle} lie below the picture, idle.{' '}
                {WIDE_R[1].divergent} + {WIDE_R[3].divergent} = {HUNT[0].count}. Counting all {WIDE_R[3].warps} warps of the corner gives{' '}
                {WIDE_R[1].divergent} + {WIDE_R[3].warps} = {warpsInRightEdgeBlocks(WIDE, BOOK_WIDTH)}. We cannot know what the authors counted; this is the simple count that reproduces their number. Either way the
                book's conclusion holds for this reading: {HUNT[0].count} or {HUNT_PRINTED} of 1040 is under 8%.
              </p>
            </Translated>
          </div>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- lab 4.2's measurement
const LAB2 = 'ch04/lab2_divergence';
const DIVCOST_ID = 'ch04/4.5/p-divcost';
interface Lab2Result {
  backend: string;
  device: string;
  whose: string;
  width: number | undefined;
  /** branchKernel on "half of each warp's lanes", over the same mix sorted. */
  mixed: number;
  patterns: { pattern: string; ratio: number }[];
  /** selectKernel over branchKernel on the mixed pattern. */
  selectOverBranch: number | undefined;
  /** loopKernel with shuffled trip counts over the same counts sorted. */
  loopShuffled: number | undefined;
}
/** The rows lab 4.2's benchmark wrote on each real device, the reader's own kernels first. */
function lab2Results(reports: Reports): Lab2Result[] {
  return realBackends(reports)
    .filter((b) => b !== 'race')
    .flatMap((backend): Lab2Result[] => {
      const run = reports[backend];
      for (const variant of ['work', 'solution']) {
        const rows = benchRows(run, LAB2, variant);
        const isHalf = (r: BenchRow) => String(r.pattern ?? '').startsWith('half of each');
        const mixed = num(rows.find((r) => r.name === 'branch' && isHalf(r))?.ratio_to_sorted);
        if (mixed == null) continue;
        return [{
          backend,
          device: String(run.device.name ?? backend),
          whose: variant === 'work' ? tr('your kernels', 'các kernel của bạn') : tr('the reference solution', 'lời giải tham chiếu'),
          width: num(rows.find((r) => r.name === 'branch')?.width),
          mixed,
          patterns: rows.filter((r) => r.name === 'branch').flatMap((r) => {
            const ratio = num(r.ratio_to_sorted);
            return ratio == null ? [] : [{ pattern: String(r.pattern), ratio }];
          }),
          selectOverBranch: num(rows.find((r) => r.name === 'select' && isHalf(r))?.ratio_to_branch),
          loopShuffled: num(rows.find((r) => r.name === 'loop' && r.order === 'shuffled')?.ratio_to_sorted),
        }];
      }
      return [];
    });
}
// What the pass count of this page says for long, equal paths mixed half and half.
const MODEL_MIXED = run(ifElse(BOOK_WIDTH, maskBelow(BOOK_WIDTH, BOOK_WIDTH / 2), 1000, 1000)).overhead;

function MeasuredDivergence({ predicted }: { predicted: number }) {
  const { reports, loading, reload } = useReports();
  const found = lab2Results(reports);
  if (!found.length)
    return loading ? (
      <p data-measured="loading"><Translated vi="Đang đọc báo cáo.">Reading reports.</Translated></p>
    ) : (
      <Translated
        vi={
          <p data-measured="none">
            Máy này chưa có số đo nào. Hãy làm lab 4.2 trên <a href={href('ch04', 'labs')}>trang lab</a>. Khi các kernel của bạn đã qua, chạy{' '}
            <code>python3 labs/pmpp.py run 4.2 --bench</code>: lệnh này in bảng kết quả và ghi lại ở nơi trang này đọc. Sau đó quay lại:
            ô này sẽ đặt dự đoán của bạn bên cạnh kết quả.{' '}
            <button className="quiet" onClick={reload}>Tải lại kết quả</button>
          </p>
        }
      >
        <p data-measured="none">
          No measurement on this machine yet. Do lab 4.2 on the <a href={href('ch04', 'labs')}>labs page</a>. Once your kernels pass, run{' '}
          <code>python3 labs/pmpp.py run 4.2 --bench</code>: it prints the table and records it where this page reads it. Then come back:
          this box will set your prediction beside the result.{' '}
          <button className="quiet" onClick={reload}>Reload results</button>
        </p>
      </Translated>
    );
  return (
    <>
      {found.map((f) => (
        <div key={f.backend} data-measured={f.backend} data-mixed={fmt(f.mixed)}>
          <Translated
            vi={
              <p>
                Trên <b>{f.device}</b> ({f.backend}), với {f.whose}{f.width ? ` và độ rộng ${f.width}` : ''}: mẫu có cả hai nhánh trong mọi
                warp mất thời gian gấp <b>{fmt(f.mixed)}</b> lần so với cùng hỗn hợp đó khi đã sắp xếp.{' '}
                {predicted > f.mixed * 1.25 ? 'Bạn đã dự đoán nhiều hơn.' : predicted < f.mixed / 1.25 ? 'Bạn đã dự đoán ít hơn.' : 'Kết quả gần với điều bạn dự đoán.'}
              </p>
            }
          >
            <p>
              On <b>{f.device}</b> ({f.backend}), with {f.whose}{f.width ? ` and a width of ${f.width}` : ''}: the pattern with both paths in every
              warp took <b>{fmt(f.mixed)}</b> times as long as the same mix sorted.{' '}
              {predicted > f.mixed * 1.25 ? 'You expected more.' : predicted < f.mixed / 1.25 ? 'You expected less.' : 'That is close to what you predicted.'}
            </p>
          </Translated>
          <Wide style={{ marginBottom: '0.6rem' }}><table className="threads" style={{ width: 'auto' }} aria-label={tr(`Lab 4.2 on ${f.backend}: every pattern`, `Lab 4.2 trên ${f.backend}: mọi mẫu`)}>
            <thead>
              <tr>
                <th><Translated vi="Mẫu">Pattern</Translated></th>
                <th><Translated vi="thời gian ÷ thời gian của cùng hỗn hợp, đã sắp xếp">time ÷ time of the same mix, sorted</Translated></th>
              </tr>
            </thead>
            <tbody>
              {f.patterns.map((r) => <tr key={r.pattern} data-pattern={r.pattern}><td>{r.pattern}</td><td>{fmt(r.ratio)}</td></tr>)}
            </tbody>
          </table></Wide>
          <p>
            {f.selectOverBranch != null ? (
              <Translated vi={<>Tính cả hai nhánh rồi chọn mất thời gian gấp <b data-out="select">{fmt(f.selectOverBranch)}</b> lần so với rẽ nhánh trên mẫu hỗn hợp đó. </>}>
                <>Computing both paths and selecting took <b data-out="select">{fmt(f.selectOverBranch)}</b> times as long as the branch on that mixed pattern. </>
              </Translated>
            ) : null}
            {f.loopShuffled != null ? (
              <Translated vi={<>Kernel vòng lặp với số vòng lặp đã xáo mất thời gian gấp <b data-out="loop">{fmt(f.loopShuffled)}</b> lần so với khi cùng các số đó đã sắp xếp.</>}>
                <>The loop kernel with shuffled trip counts took <b data-out="loop">{fmt(f.loopShuffled)}</b> times as long as with the same counts sorted.</>
              </Translated>
            ) : null}
          </p>
          {f.backend === 'metal' ? (
            <Translated
              vi={
                <p className="muted">
                  Trên Metal, các mẫu được bố trí dựa trên những gì một kernel thăm dò báo cáo về SIMD group, và Apple không công bố mô hình
                  phân kỳ nào để so sánh.
                </p>
              }
            >
              <p className="muted">
                On Metal the patterns were laid out from what a probe kernel reported about SIMD-groups, and Apple documents no model of
                divergence to compare this with.
              </p>
            </Translated>
          ) : null}
        </div>
      ))}
      <Translated
        vi={
          <p className="muted">
            Đây là số đo từ máy của bạn, không phải giá trị kỳ vọng. Cách đếm lượt của trang này cho ra {fmt(MODEL_MIXED, 2)} với các nhánh dài
            bằng nhau: hai lượt ở chỗ một warp đồng nhất chỉ cần một. Một thiết bị có thể cho kết quả thấp hơn hoặc cao hơn số lượt của nó.
          </p>
        }
      >
        <p className="muted">
          These are measurements from your machine, not expected values. The pass count of this page says {fmt(MODEL_MIXED, 2)} for long, equal
          paths: two passes where a uniform warp makes one. A device can come out lower or higher than its pass count.
        </p>
      </Translated>
    </>
  );
}

// ---------------------------------------------------------------- questions
const EX3 = boundary1D(2000, 512, BOOK_WIDTH);
const EX3_SPLIT = splitWarp1D(2000, 512, BOOK_WIDTH)!;
const TALL_B = boundary2D(TALL, BOOK_WIDTH);
const WIDE_B = boundary2D(WIDE, BOOK_WIDTH);
const CAUSAL = classifyGrid(1024 / 32, 1024 / 8, dims(32, 8), BOOK_WIDTH, causal);
const Q_TRIPS = shuffledTrips(8 * BOOK_WIDTH, TRIP_LO, TRIP_HI, 7);
const Q_SHUFFLED = loopBatch(Q_TRIPS, BOOK_WIDTH);
const Q_SORTED = loopBatch(sortedTrips(Q_TRIPS), BOOK_WIDTH);
const Q_BRANCH = run(ifElse(BOOK_WIDTH, maskModulo(BOOK_WIDTH, 2), 100, 100));
const Q_SELECT = run(select(BOOK_WIDTH, 100, 100));
const Q_UNIFORM = run(ifElse(BOOK_WIDTH, maskBelow(BOOK_WIDTH, BOOK_WIDTH), 100, 100));
const AMD64 = [0, 1].map((k) => countTaken(maskBelow(64, 32, k * 64)));

export const questionsS45: Record<string, Question> = {
  'ch04/4.5/q-ex2': {
    kind: 'numeric',
    id: 'ch04/4.5/q-ex2',
    prompt: (
      <Translated vi={<>Bài tập 2 của sách. Với phép cộng vector, vector có 2000 phần tử và mỗi khối có 512 luồng. Lưới có bao nhiêu luồng?</>}>
        <>The book's exercise 2. For vector addition, the vector has 2000 elements and each block has 512 threads. How many threads are in the grid?</>
      </Translated>
    ),
    answer: EX3.threads,
    unit: 'threads',
    explain: (
      <Translated
        vi={
          <p>
            Một lưới gồm trọn các khối: ⌈2000 / 512⌉ = {EX3.blocks} khối 512 luồng, vậy là {EX3.threads} luồng. {EX3.threads - 2000} luồng trong số đó không có phần tử
            nào và bị chặn lại bởi phép kiểm tra biên. Sách không in đáp án; đáp án này là của chúng tôi.
          </p>
        }
      >
        <p>
          A grid is whole blocks: ⌈2000 / 512⌉ = {EX3.blocks} blocks of 512, so {EX3.threads} threads. {EX3.threads - 2000} of them have no element
          and are stopped by the boundary check. The book prints no answers; this one is ours.
        </p>
      </Translated>
    ),
    source: 'ch. 4 exercise 2, p. 91',
    vi: { unit: 'luồng', source: 'ch. 4, bài tập 2, p. 91' },
  },
  'ch04/4.5/q-ex3': {
    kind: 'choice',
    id: 'ch04/4.5/q-ex3',
    prompt: (
      <Translated vi={<>Bài tập 3 của sách. Trong lần khởi chạy đó (2000 phần tử, khối 512 luồng, warp rộng 32), bao nhiêu warp bị phân kỳ ở phép kiểm tra biên?</>}>
        <>The book's exercise 3. In that launch (2000 elements, 512-thread blocks, warps 32 wide), how many warps diverge at the boundary check?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="2: hai warp cuối đều chứa những luồng không có phần tử nào để làm.">2: the last two warps both contain threads with no element to work on.</Translated>,
        why: <Translated vi="Warp cuối chỉ chứa toàn những luồng như vậy. Tất cả đều trượt phép thử, nên cùng đi một nhánh.">The last warp contains only such threads. All of them fail the test, so they all take the same path.</Translated>,
      },
      {
        label: (
          <Translated vi={`${EX3.divergent}: warp nơi dữ liệu kết thúc, ${EX3_SPLIT.valid} luồng trong phạm vi và ${EX3_SPLIT.invalid} luồng ngoài.`}>
            {`${EX3.divergent}: the warp where the data ends, ${EX3_SPLIT.valid} threads in range and ${EX3_SPLIT.invalid} out.`}
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="0: mọi luồng chạy cùng một phép kiểm tra, nên mọi warp đi theo một nhánh.">0: every thread runs the same check, so every warp follows one path.</Translated>,
        why: <Translated vi="Câu lệnh thì giống nhau; kết quả của nó thì không. Có một warp chứa luồng ở cả hai phía của 2000.">The statement is the same; its outcome is not. One warp has threads on both sides of 2000.</Translated>,
      },
      {
        label: <Translated vi="48: mỗi luồng không có phần tử tự đi nhánh kia một mình.">48: each thread with no element takes the other path on its own.</Translated>,
        why: <Translated vi="Phân kỳ được đếm theo warp: một warp hoặc có nhiều hơn một nhánh bên trong, hoặc không.">Divergence is counted in warps: a warp either has more than one path in it or it does not.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            2000 = 62 · 32 + 16. Warp 0 đến 61 của lưới chứa luồng 0 đến 1983, tất cả đều trong phạm vi. Warp 62 chứa luồng 1984 đến 2015: 16 luồng qua
            phép thử và 16 luồng trượt, nên nó bị phân kỳ. Warp 63 chứa 2016 đến 2047, tất cả đều ngoài phạm vi: một nhánh, không phân kỳ, và {EX3.idle} warp
            phần cứng không làm gì cả. Một biên một chiều có thể tách nhiều nhất một warp, dù kích thước là bao nhiêu. Đáp án là của chúng tôi.
          </p>
        }
      >
        <p>
          2000 = 62 · 32 + 16. Warps 0 to 61 of the grid hold threads 0 to 1983, all in range. Warp 62 holds threads 1984 to 2015: 16 pass the
          test and 16 fail, so it diverges. Warp 63 holds 2016 to 2047, every one out of range: one path, no divergence, and {EX3.idle} warp of
          hardware that does nothing. A one-dimensional boundary can split at most one warp, whatever the size. The answer is ours.
        </p>
      </Translated>
    ),
    source: 'ch. 4 exercise 3, p. 91; notes/today/ch04-facts.md E1',
    vi: { source: 'ch. 4, bài tập 3, p. 91; notes/today/ch04-facts.md E1' },
  },
  'ch04/4.5/q-200x150': {
    kind: 'numeric',
    id: 'ch04/4.5/q-200x150',
    prompt: (
      <Translated
        vi={
          <>
            Một bức ảnh rộng 150 pixel và cao 200 pixel. Nó được xử lý bằng các khối 16 × 16, mỗi pixel một luồng, trên một thiết bị có warp rộng 32.
            Bao nhiêu warp bị phân kỳ ở phép kiểm tra biên?
          </>
        }
      >
        <>
          A picture is 150 pixels wide and 200 high. It is processed with 16 × 16 blocks, one thread per pixel, on a device with 32-wide warps.
          How many warps diverge at the boundary check?
        </>
      </Translated>
    ),
    answer: TALL_B.divergent,
    unit: 'warps',
    explain: (
      <Translated
        vi={
          <p>
            150 = 9 · 16 + 6 và 200 = 12 · 16 + 8, nên lưới rộng 10 khối và cao 13 khối: {TALL_B.blocks} khối, {big(TALL_B.warps)} warp. Một
            warp là hai hàng của một khối. Mép phải chừa lại 6 cột hợp lệ trên 16, nên nó tách mọi warp của 12 khối cao trọn mà nó
            cắt qua: 12 · 8 = {TALL_B.regions[1].divergent}. Mép dưới chừa lại 8 hàng hợp lệ, tức 4 warp trọn vẹn trong phạm vi và 4 warp ngoài phạm vi:
            không warp nào bị tách. Khối góc có 4 warp bị tách. {TALL_B.regions[1].divergent} + {TALL_B.regions[3].divergent} = {TALL_B.divergent} trên{' '}
            {big(TALL_B.warps)}, tức {pct(TALL_B.share)}. Đó là lý do “dưới 8%” của sách cần cách đọc còn lại: rộng 200 và cao 150 cho ra{' '}
            {WIDE_B.divergent}, tức {pct(WIDE_B.share)}.
          </p>
        }
      >
        <p>
          150 = 9 · 16 + 6 and 200 = 12 · 16 + 8, so the grid is 10 blocks across and 13 down: {TALL_B.blocks} blocks, {big(TALL_B.warps)} warps. A
          warp is two rows of a block. The right edge leaves 6 valid columns of 16, so it splits every warp of the 12 full-height blocks it
          crosses: 12 · 8 = {TALL_B.regions[1].divergent}. The bottom edge leaves 8 valid rows, which is 4 whole warps in range and 4 out of range:
          none split. The corner block has 4 split warps. {TALL_B.regions[1].divergent} + {TALL_B.regions[3].divergent} = {TALL_B.divergent} of{' '}
          {big(TALL_B.warps)}, or {pct(TALL_B.share)}. That is why the book's “under 8%” needs the other reading: 200 wide and 150 high gives{' '}
          {WIDE_B.divergent}, or {pct(WIDE_B.share)}.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.5, p. 82; notes/today/ch04-facts.md E1', // i18n-ignore: citation
    vi: { unit: 'warp' },
  },
  'ch04/4.5/q-amd32': {
    kind: 'choice',
    id: 'ch04/4.5/q-amd32',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel chứa <code>if (threadIdx.x &lt; 32) {'{ ... }'} else {'{ ... }'}</code> và chạy với khối 128 luồng trên một thiết bị có warp
            rộng 64. Những warp nào bị phân kỳ tại đó?
          </>
        }
      >
        <>
          A kernel contains <code>if (threadIdx.x &lt; 32) {'{ ... }'} else {'{ ... }'}</code> and runs in 128-thread blocks on a device whose warps
          are 64 wide. Which warps diverge at it?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Không warp nào: 32 là một ranh giới warp, nên mỗi warp nằm về một phía của phép thử.">None: 32 is a warp boundary, so each warp falls on one side of the test.</Translated>,
        why: <Translated vi="Nó là ranh giới warp ở nơi warp rộng 32. Ở đây ranh giới đầu tiên nằm ở 64.">It is a warp boundary where warps are 32 wide. Here the first boundary is at 64.</Translated>,
      },
      {
        label: <Translated vi="Cả hai warp của mọi khối: ngưỡng nằm bên trong khối, nên cả hai đều thấy nó.">Both warps of every block: the threshold sits inside the block, so both see it.</Translated>,
        why: <Translated vi="Warp 1 chứa luồng 64 đến 127. Không luồng nào dưới 32, nên tất cả cùng đi nhánh else.">Warp 1 holds threads 64 to 127. None is below 32, so they all take the else path together.</Translated>,
      },
      {
        label: (
          <Translated vi={`Warp 0 của mọi khối, mỗi phía ${AMD64[0]} lane; warp 1 trượt phép thử toàn bộ.`}>
            {`Warp 0 of every block, ${AMD64[0]} lanes each way; warp 1 fails the test whole.`}
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Warp 1 của mọi khối: nó chứa các luồng đứng sau ngưỡng.">Warp 1 of every block: it holds the threads that come after the threshold.</Translated>,
        why: <Translated vi="Luồng 0 đến 31 và 32 đến 63 mới là những luồng bất đồng, và chúng chung warp 0.">Threads 0 to 31 and 32 to 63 are the ones that disagree, and they share warp 0.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Ở độ rộng 64, một khối 128 luồng là hai warp: luồng 0 đến 63 và 64 đến 127. Ngưỡng 32 rơi vào giữa warp thứ nhất. Câu
            lệnh này chính là ví dụ của AMD về một nhánh không phân kỳ, và nó chỉ không phân kỳ trên các thiết bị có warp rộng 32. Một
            ngưỡng viết dưới dạng bội số của <code>warpSize</code> sẽ đúng trên cả hai.
          </p>
        }
      >
        <p>
          At 64 wide, a 128-thread block is two warps: threads 0 to 63 and 64 to 127. The threshold 32 falls in the middle of the first. This
          statement is AMD's own example of a branch with no divergence, and it is divergence-free only on devices with 32-wide warps. A
          threshold written as a multiple of <code>warpSize</code> would hold on both.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch04-facts.md B7', // i18n-ignore: citation
  },
  'ch04/4.5/q-buckets': {
    kind: 'choice',
    id: 'ch04/4.5/q-buckets',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel giao cho mỗi chuỗi của một batch một luồng, và mỗi luồng lặp tới độ dài chuỗi của chính nó. Bạn có thể khởi chạy với
            batch theo thứ tự đến hoặc đã sắp xếp theo độ dài. Tổng số lần lặp là như nhau. Cách nào nhanh hơn, và vì sao?
          </>
        }
      >
        <>
          A kernel gives each sequence of a batch one thread, and each thread loops up to its own sequence's length. You can launch it with the
          batch in arrival order or sorted by length. The total number of loop iterations is the same. Which is faster, and why?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Không cách nào: tổng số lần lặp như nhau, nên khối lượng việc và thời gian như nhau.">Neither: the total number of iterations is the same, so the work and the time are the same.</Translated>,
        why: <Translated vi="Số lần lặp hữu ích như nhau, đúng vậy. Điều thay đổi là các lane đã xong phải ngồi bao lâu trong một warp vẫn đang chạy.">The same useful iterations, yes. What changes is how long finished lanes sit in a warp that is still running.</Translated>,
      },
      {
        label: <Translated vi="Thứ tự đến: trộn luồng dài và ngắn trong mọi warp giúp cân bằng tải giữa các warp.">Arrival order: mixing long and short threads in every warp balances the load across warps.</Translated>,
        why: <Translated vi="Cân bằng giữa các warp không giúp gì cho một lane đã xong. Nó vẫn chờ warp của chính nó.">Balance between warps does not help a lane that has finished. It waits for its own warp.</Translated>,
      },
      {
        label: <Translated vi="Đã sắp xếp: gom các độ dài bằng nhau lại giúp lần khởi chạy dùng ít warp hơn cho batch.">Sorted: grouping equal lengths together lets the launch use fewer warps for the batch.</Translated>,
        why: <Translated vi="Số warp bằng số luồng chia cho độ rộng, theo thứ tự nào cũng vậy.">The number of warps is threads divided by the width, in either order.</Translated>,
      },
      {
        label: (
          <Translated vi="Đã sắp xếp: một warp chạy lâu bằng luồng dài nhất của nó, và sắp xếp làm các độ dài trong một warp gần bằng nhau.">
            Sorted: a warp runs as long as its longest thread, and sorting makes a warp's lengths nearly equal.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đây là vòng lặp trong Hình 4.10 của sách ở quy mô một batch. Trong batch vẽ trên trang này ({Q_TRIPS.length} luồng, độ dài{' '}
            {TRIP_LO} đến {TRIP_HI}, warp rộng 32) cả hai thứ tự đều làm {big(Q_SORTED.useful)} lần lặp lane có việc; khi sắp xếp, các warp chạy tổng cộng{' '}
            {Q_SORTED.spent} lần lặp, còn khi xáo là {Q_SHUFFLED.spent}. Gom nhóm theo độ dài là việc bạn vẫn làm để giảm
            padding; ở đây nó còn làm các warp đồng nhất. Mối liên hệ với batching là của chúng tôi, con số đếm cũng vậy; lab 4.2 đo cùng điều này trên một
            thiết bị.
          </p>
        }
      >
        <p>
          This is the loop of the book's Fig. 4.10 at the scale of a batch. In the batch drawn on this page ({Q_TRIPS.length} threads, lengths{' '}
          {TRIP_LO} to {TRIP_HI}, 32-wide warps) both orders do {big(Q_SORTED.useful)} lane-iterations of work; sorted, the warps run for{' '}
          {Q_SORTED.spent} iterations in total, and shuffled for {Q_SHUFFLED.spent}. Bucketing by length is something you already do to cut
          padding; here it also makes warps uniform. The link to batching is ours, and so is the count; lab 4.2 measures the same thing on a
          device.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.5, p. 81 (Fig. 4.10); the batching example is ours',
    vi: { source: 'ch. 4 §4.5, p. 81 (Hình 4.10); ví dụ về batching là của chúng tôi' },
  },
  'ch04/4.5/q-causal': {
    kind: 'choice',
    id: 'ch04/4.5/q-causal',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel điền một ma trận điểm attention kích thước T × T, mỗi điểm một luồng, với T = 1024. Mỗi luồng thử <code>col &lt;= row</code>{' '}
            và đi một nhánh cho các điểm được giữ, một nhánh khác cho các điểm bị che. Khối rộng 32 và cao 8, warp rộng 32. Xấp xỉ bao nhiêu
            phần trăm số warp bị phân kỳ ở phép thử?
          </>
        }
      >
        <>
          A kernel fills a T × T matrix of attention scores, one thread per score, with T = 1024. Each thread tests <code>col &lt;= row</code>{' '}
          and takes one path for kept scores and another for masked ones. Blocks are 32 wide and 8 high and warps are 32 wide. Roughly what
          share of the warps diverge at the test?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Khoảng 50%: một nửa số điểm bị che, nên một nửa số warp đi nhánh kia.">About 50%: half of the scores are masked, so half of the warps take the other path.</Translated>,
        why: <Translated vi="Một nửa số luồng đi mỗi nhánh. Nhưng phần lớn warp nằm trọn về một phía của đường chéo.">Half of the threads take each path. But most warps lie wholly on one side of the diagonal.</Translated>,
      },
      {
        label: (
          <Translated vi={`Khoảng ${Math.round(CAUSAL.share * 100)}%: ở mỗi hàng, chỉ warp nằm vắt qua đường chéo.`}>
            {`About ${Math.round(CAUSAL.share * 100)}%: in each row, only the warp that straddles the diagonal.`}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Tất cả: phép thử phụ thuộc vào hàng và cột của từng luồng, nên nó thay đổi trong mọi warp.">
            {'All of them: the test depends on each thread\'s row and column, so it varies in every warp.'}
          </Translated>
        ),
        why: <Translated vi="Phụ thuộc vào chỉ số khiến phân kỳ có thể xảy ra. Nó chỉ xảy ra trong warp có các luồng bất đồng.">Depending on the index makes divergence possible. It happens only in a warp whose threads disagree.</Translated>,
      },
      {
        label: <Translated vi="Không warp nào: khối rộng 32, nên đường chéo rơi đúng mép khối và không cắt warp nào.">None: blocks are 32 wide, so the diagonal falls on block edges and cuts no warp.</Translated>,
        why: <Translated vi="Nó cắt mỗi hàng một lần, bên trong một warp nào đó, trừ khi nó kết thúc đúng tại ranh giới warp.">It crosses each row once, inside some warp, except where it ends exactly on a warp boundary.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Với khối rộng 32, một warp là 32 cột liên tiếp của một hàng. Ở hàng <i>r</i> phép thử đổi từ đúng sang sai sau cột{' '}
            <i>r</i>, nên đúng một warp của hàng có thể bị tách: warp chứa cột <i>r</i>. Nó bị tách trừ khi <i>r</i> là cột cuối của warp
            đó, điều xảy ra ở 1 trên 32 hàng. Còn lại 1024 − 32 = {CAUSAL.divergent} warp bị tách trên tổng số {big(CAUSAL.warps)}, tức{' '}
            {pct(CAUSAL.share)}. Phương pháp là cách sách đếm cho mép phải của một bức ảnh, áp dụng cho một đường chéo; ví dụ là của chúng tôi.
          </p>
        }
      >
        <p>
          With blocks 32 wide, a warp is 32 consecutive columns of one row. In row <i>r</i> the test changes from true to false after column{' '}
          <i>r</i>, so exactly one warp of the row can be split: the one holding column <i>r</i>. It is split unless <i>r</i> is that warp's last
          column, which happens in 1 row of 32. That leaves 1024 − 32 = {CAUSAL.divergent} split warps out of {big(CAUSAL.warps)}, or{' '}
          {pct(CAUSAL.share)}. The method is the book's count for the right edge of a picture, applied to a diagonal; the example is ours.
        </p>
      </Translated>
    ),
    source: 'method: ch. 4 §4.5, p. 82; the example is ours',
    vi: { source: 'phương pháp: ch. 4 §4.5, p. 82; ví dụ là của chúng tôi' },
  },
  'ch04/4.5/q-ex5': {
    kind: 'choice',
    id: 'ch04/4.5/q-ex5',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 5 của sách. Một lập trình viên nói: “Nếu tôi khởi chạy với chỉ 32 luồng mỗi khối, tôi có thể bỏ các lời gọi{' '}
            <code>__syncthreads()</code>.” Đó có phải ý hay không?
          </>
        }
      >
        <>
          The book's exercise 5. A programmer says: “If I launch with only 32 threads in each block, I can leave out the{' '}
          <code>__syncthreads()</code> calls.” Is that a good idea?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có: một khối 32 luồng là một warp, và một warp phát một lệnh cho mọi luồng của nó, nên chúng không thể lệch nhau.">
            Yes: a 32-thread block is one warp, and a warp issues one instruction for all its threads, so they cannot drift apart.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là mô tả phần cứng NVIDIA trước Volta, ở một độ rộng cụ thể. Mô hình lập trình không hề hứa điều đó.">
            That was a description of NVIDIA hardware before Volta, at one particular width. It is not something the programming model promises.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Có: với một warp trong khối, rào chắn (barrier) toàn khối chẳng có gì để chờ, nên dù sao nó cũng được biên dịch thành không có gì.">
            Yes: with one warp in a block a block-wide barrier has nothing to wait for, so it compiles to nothing anyway.
          </Translated>
        ),
        why: <Translated vi="Không tài liệu nào nói vậy, và lập luận này vẫn phụ thuộc vào việc độ rộng là 32.">Nothing documents that, and the argument would still depend on the width being 32.</Translated>,
      },
      {
        label: (
          <Translated vi="Không: nó giả định rằng một khối đúng bằng một warp và một warp luôn đi đồng bộ, và cả hai đều không đúng trên mọi thiết bị.">
            No: it assumes that a block is exactly one warp and that a warp stays in step, and neither holds on every device.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không: một khối 32 luồng được trải trên hai nửa warp, nên vẫn cần rào chắn giữa chúng.">
            No: a block of 32 threads is spread over two half-warps, so the barrier is still needed between them.
          </Translated>
        ),
        why: <Translated vi="Ở độ rộng 32 nó là một warp. Vấn đề là dựa vào điều đó.">At a width of 32 it is one warp. The trouble is relying on that.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Có ba điều sai. Mục 4.4 mở đầu bằng quy tắc rằng trong một khối bạn giả định mọi thứ tự đều có thể xảy ra và nêu rõ mọi
              phụ thuộc bằng một rào chắn. Độ rộng là một chi tiết hiện thực mà chính sách cũng cho là sẽ thay đổi, và nó đã là 64 trên
              một số thiết bị AMD. Và kể từ Volta, hướng dẫn của NVIDIA nói rằng mã giả định các luồng trong warp chạy đồng bộ là không hợp lệ và phải được
              đồng bộ hóa tường minh.
            </p>
            <p>
              Còn một cái giá nữa: trên thiết bị ví dụ của sách, khối 32 luồng giới hạn occupancy (mức chiếm dụng) ở 50%, như{' '}
              <a href={href('ch04', '4.7')}>mục 4.7</a> tính ra. Đáp án là của chúng tôi; sách không in đáp án.
            </p>
          </>
        }
      >
        <>
          <p>
            Three things are wrong with it. Section 4.4 began with the rule that inside a block you assume any order and state every
            dependence with a barrier. The width is an implementation detail the book itself expects to change, and it is already 64 on
            some AMD devices. And since Volta, NVIDIA's guide says code that assumes lockstep within a warp is invalid and should be
            synchronised explicitly.
          </p>
          <p>
            There is a cost on top: on the book's example device, 32-thread blocks cap occupancy at 50%, as{' '}
            <a href={href('ch04', '4.7')}>section 4.7</a> works out. The answer is ours; the book prints none.
          </p>
        </>
      </Translated>
    ),
    source: 'ch. 4 exercise 5, p. 91; §4.4 p. 74, §4.5 p. 81, §4.7 pp. 85–86; notes/today/ch04-facts.md A10',
    vi: { source: 'ch. 4, bài tập 5, p. 91; §4.4 p. 74, §4.5 p. 81, §4.7 pp. 85–86; notes/today/ch04-facts.md A10' },
  },
  'ch04/4.5/q-select': {
    kind: 'choice',
    id: 'ch04/4.5/q-select',
    prompt: (
      <Translated
        vi={
          <>
            Cả hai nhánh của một câu lệnh rẽ nhánh đều là vòng lặp khoảng 100 lệnh, và trong mọi warp một nửa số lane đi mỗi nhánh. Bạn thay
            câu lệnh rẽ nhánh bằng “tính cả hai kết quả, rồi chọn một”. Bạn được gì?
          </>
        }
      >
        <>
          Both paths of a branch are loops of about 100 instructions, and in every warp half of the lanes take each path. You replace the
          branch with “compute both results, then select one”. What do you gain?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Chẳng đáng kể: giờ mọi lane chạy cả hai nhánh, xấp xỉ những gì warp bị tách vốn đã tiêu.">Nothing to speak of: every lane now runs both paths, about what the split warp already spent.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Gấp đôi: lượt thứ hai biến mất, nên warp chỉ chạy một lượt thay vì hai.">A factor of two: the second pass disappears, so the warp makes one pass instead of two.</Translated>,
        why: <Translated vi="Lượt thứ hai biến mất rồi quay lại thành nửa sau của một lượt dài gấp đôi.">The second pass disappears and comes back as the second half of a pass twice as long.</Translated>,
      },
      {
        label: <Translated vi="Rất nhiều lần: một nhánh phân kỳ rất đắt trên GPU, còn phép chọn chỉ tốn một lệnh.">A large factor: a divergent branch is expensive on a GPU, and a select costs one instruction.</Translated>,
        why: <Translated vi="Phép chọn thì rẻ. Tính cả hai đầu vào của nó mới tốn kém.">The select is cheap. Computing both of its inputs is what costs.</Translated>,
      },
      {
        label: <Translated vi="Chậm đi gấp đôi: giờ mọi lane làm việc của cả hai nhánh thay vì một.">It gets twice as slow: every lane now does the work of both paths instead of one.</Translated>,
        why: <Translated vi="Chậm gấp đôi so với một warp đồng nhất, đúng. So với warp phân kỳ mà nó thay thế thì xấp xỉ như nhau.">Twice as slow as a uniform warp, yes. Compared with the divergent warp it replaces, about the same.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đếm bước warp trong mô hình của trang này: warp phân kỳ tiêu {Q_BRANCH.spent} (phép thử, 100 cho A, 100 cho B, rồi C); phiên bản
            phép chọn tiêu {Q_SELECT.spent} và không lane nào chờ, nhưng một nửa phép tính của mọi lane bị bỏ đi. Một warp đồng nhất tiêu{' '}
            {Q_UNIFORM.spent}. Trong mô hình này, phép chọn không bao giờ rẻ hơn nhánh bị tách mà nó thay thế, với bất kỳ độ dài nhánh nào: nó cộng thêm chính
            phép chọn. Nó có thắng với các nhánh vài lệnh hay không phụ thuộc vào chi phí của một lệnh rẽ nhánh ngoài các lượt, điều mà mô hình bỏ qua
            và không nhà cung cấp nào công bố; đó sẽ là nhận định của chúng tôi, và nó chưa được kiểm chứng. Với các nhánh dài, cách chữa là thôi trộn chúng trong một warp.
            Hướng dẫn của AMD liệt kê cả hai cách chữa, và lab 4.2 đo thời gian phép chọn so với rẽ nhánh trên thiết bị của bạn.
          </p>
        }
      >
        <p>
          Count warp-steps in this page's model: the divergent warp spends {Q_BRANCH.spent} (the test, 100 for A, 100 for B, then C); the
          select version spends {Q_SELECT.spent} and no lane waits, but half of every lane's arithmetic is thrown away. A uniform warp spends{' '}
          {Q_UNIFORM.spent}. In this model a select never costs less than the split branch it replaces, at any length of path: it adds the
          select. Whether it wins for paths of a few instructions depends on what a branch costs beyond its passes, which the model leaves out
          and no vendor documents; that would be our claim, and it is untested. For long paths the remedy is to stop mixing them inside a warp.
          AMD's guide lists both remedies, and lab 4.2 times the select against the branch on your device.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch04-facts.md A9, B7; the comparison is ours',
    vi: { source: 'notes/today/ch04-facts.md A9, B7; phép so sánh là của chúng tôi' },
  },
};

// ---------------------------------------------------------------- the page
const V1003 = boundary1D(1003, 64, BOOK_WIDTH);
const V1003_SPLIT = splitWarp1D(1003, 64, BOOK_WIDTH)!;
const V10000 = boundary1D(10000, 64, BOOK_WIDTH);
const P76: Picture = { width: 76, height: 62, blockX: 16, blockY: 16 };
const P76_32 = boundary2D(P76, BOOK_WIDTH);
const P76_64 = boundary2D(P76, 64);
const P76_R = regions(P76, BOOK_WIDTH);

export function S45() {
  return (
    <>
      <Translated
        vi={
          <p>
            Một warp có một dòng lệnh duy nhất. Đó là toàn bộ cái lợi của <a href={href('ch04', '4.4')}>mục 4.4</a>: một lần nạp lệnh, cho mọi
            lane. Điều đó hiệu quả chừng nào mọi luồng của warp đều muốn cùng một lệnh tiếp theo. Mục này nói về chuyện xảy ra khi
            chúng không muốn vậy, cách đếm nó, và khi nào con số đó đủ lớn để bạn phải bận tâm.
          </p>
        }
      >
        <p>
          A warp has one instruction stream. That is the whole economy of <a href={href('ch04', '4.4')}>section 4.4</a>: one fetch, every
          lane. It works for as long as every thread of the warp wants the same next instruction. This section is about what happens when
          they do not, how to count it, and when the count is large enough to care about.
        </p>
      </Translated>
      <Source section="4.5" pages="79" />

      <h2><Translated vi="Một dòng lệnh, nhiều nhánh">One stream, several paths</Translated></h2>
      <Translated
        vi={
          <p>
            Đặt một <code>if</code> vào kernel, và một số luồng của warp có thể đi vào nó trong khi các luồng khác đi vào <code>else</code>. Phần cứng
            vẫn phát từng lệnh một cho warp, nên nó đi qua các nhánh lần lượt. Nó chạy một lượt qua nhánh thứ nhất, trong đó chỉ các luồng
            đã chọn nhánh ấy có hiệu lực, rồi một lượt qua nhánh kia cho các luồng còn lại. Các luồng của cùng một warp đi theo những nhánh
            khác nhau chính là điều sách gọi là <b>phân kỳ điều khiển (control divergence)</b>.
          </p>
        }
      >
        <p>
          Put an <code>if</code> in a kernel and some threads of a warp may take it while others take the <code>else</code>. The hardware
          still issues one instruction at a time to the warp, so it visits the paths one after another. It makes a pass over the first path
          in which only the threads that chose it take effect, then a pass over the other path for the rest. Threads of one warp following
          different paths is what the book calls <b>control divergence</b>.
        </p>
      </Translated>
      <Source section="4.5" pages="79–80" />
      <Translated
        vi={
          <p>
            Từ đó suy ra hai điều, kéo về hai hướng ngược nhau. Mọi luồng vẫn làm đúng những gì mã nguồn viết cho nó: các lượt chính là
            cách phần cứng SIMD giữ lời hứa rằng bạn chỉ viết một luồng vô hướng. Và warp phải trả giá hai lần: nó chạy nhiều lượt hơn
            một warp đồng nhất, và trong mỗi lượt, lane của các luồng ở nhánh kia bị chiếm chỗ mà vẫn rảnh.
          </p>
        }
      >
        <p>
          Two things follow, and they pull in opposite directions. Every thread still does exactly what the source says for it: the passes
          are how SIMD hardware keeps the promise that you write one scalar thread. And the warp pays for it twice: it makes more passes
          than a uniform warp would, and in each pass the lanes of the threads on the other path are occupied and idle.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A9)" /* i18n-ignore: date and citation */>
        <Translated
          vi={
            <p>
              Hướng dẫn của NVIDIA gọi nó là warp divergence hay branch divergence và mô tả cùng một cơ chế: warp “executes each branch
              path taken, disabling threads that are not on that path”. Hướng dẫn không đưa ra con số nào cho chi phí của một lượt.
            </p>
          }
        >
          <p>
            NVIDIA's guide calls it warp divergence or branch divergence and describes the same mechanism: the warp “executes each branch
            path taken, disabling threads that are not on that path”. It gives no figure for what a pass costs.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Nếu bạn từng tối ưu mã CPU, hãy đảo ngược một thói quen. Một lệnh rẽ nhánh trên CPU rẻ khi kết quả của nó đều đặn theo thời gian, để
            bộ dự đoán nhánh đoán trúng. Một lệnh rẽ nhánh trên GPU rẻ khi nó đi cùng một hướng ở mọi lane của warp vào lúc được gặp. Điều
            quan trọng là sự đồng thuận giữa các lane lân cận, không phải sự đều đặn theo thời gian.
          </p>
        }
      >
        <p>
          If you have tuned CPU code, turn one habit around. A CPU branch is cheap when its outcome is regular over time, so that the
          predictor guesses it. A GPU branch is cheap when it goes the same way in every lane of the warp at the moment it is reached. What
          matters is agreement across neighbours, not regularity over time.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Hướng dẫn HIP của AMD diễn đạt sự tương phản này là tính đồng nhất theo thời gian đối với theo không gian (notes/today/ch04-facts.md B7). Cách
              diễn đạt ở đây là của chúng tôi.
            </p>
          }
        >
          <p>
            AMD's HIP guide draws this contrast as temporal against spatial uniformity (notes/today/ch04-facts.md B7). The wording here is
            ours.
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Phân kỳ là một chuỗi sự kiện, nên hãy xem nó diễn ra một lần trước khi đếm bất cứ thứ gì. Hoạt hình theo một warp 32 lane
            của sách qua một câu lệnh rẽ nhánh, một vòng lặp, và câu hỏi các lượt diễn ra theo thứ tự nào.
          </p>
        }
      >
        <p>
          Divergence is a sequence of events, so watch it happen once before you count anything. The animation follows one warp of the
          book's 32 lanes through a branch, a loop, and the question of what order the passes come in.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi hàng của hoạt hình là một lệnh được phát cho cả warp, và được tính một bước dù 32 lane thực thi nó hay chỉ
              một. Đó là bức tranh của sách về cơ chế này. Nó không phải chi phí đo được trên thiết bị nào.
            </p>
          }
        >
          <p>
            Every row of the animation is one instruction issued to the whole warp, and it counts one step whether 32 lanes execute it or
            one. That is the book's picture of the mechanism. It is not a measured cost on any device.
          </p>
        </Translated>
      </Simplification>
      <WarpBranchScene />

      <h2><Translated vi="Một if-else, từng lượt một">An if-else, pass by pass</Translated></h2>
      <Translated
        vi={
          <p>
            Ví dụ của sách là một warp gồm luồng 0 đến 31 gặp <code>if (threadIdx.x &lt; 24)</code>. Trong một lượt, luồng 0 đến 23 thực thi
            A trong khi 24 đến 31 không hoạt động; trong lượt kia, luồng 24 đến 31 thực thi B trong khi 0 đến 23 không hoạt động. Sau đó các luồng có thể
            hợp lại và thực thi C.
          </p>
        }
      >
        <p>
          The book's example is a warp of threads 0 to 31 reaching <code>if (threadIdx.x &lt; 24)</code>. In one pass threads 0 to 23 execute
          A while 24 to 31 are inactive; in another, threads 24 to 31 execute B while 0 to 23 are inactive. After that the threads may come
          back together and execute C.
        </p>
      </Translated>
      <Source section="4.5" pages="80" figs={tr('Fig. 4.9', 'Hình 4.9')} />
      <Translated
        vi={
          <p>
            Trình mô phỏng dưới đây mở ra đúng trường hợp đó. Nó đếm theo <b>bước warp</b>: một lệnh được phát cho warp là một bước, bất kể
            bao nhiêu lane thực thi nó. Bên cạnh những gì warp đã tiêu, nó cho thấy cùng khối lượng việc sẽ mất bao nhiêu nếu không lane nào phải chờ, tức là
            khối lượng việc tính bằng bước lane chia cho độ rộng.
          </p>
        }
      >
        <p>
          The simulator below opens on that case. It counts in <b>warp-steps</b>: one instruction issued to the warp is one step, whatever the
          number of lanes that execute it. Beside what the warp spent it shows what the same work would take if no lane ever waited, which is
          the work in lane-steps divided by the width.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi lệnh tốn một bước warp, dù ít lane chạy nó đến đâu: đó là bức tranh của sách, không phải chi phí đo được (facts A9, B7, C4). Phần
              việc của chính lệnh rẽ nhánh, cũng như phép thử và bước nhảy của vòng lặp, ở đây không tốn gì.
            </p>
          }
        >
          <p>
            Every instruction costs one warp-step, however few lanes run it: the book's picture, not a measured cost (facts A9, B7, C4). A
            branch's own work, and a loop's test and jump, cost nothing here.
          </p>
        </Translated>
      </Simplification>
      <DivergenceSim />
      <Translated vi={<p>Bốn điều cần tìm trước khi đọc tiếp. Dự đoán từng điều, rồi thử.</p>}>
        <p>Four things to find before you read on. Predict each, then try it.</p>
      </Translated>
      <Translated
        vi={
          <ol>
            <li>
              Kéo T từ 24 xuống 16, rồi xuống 1. Cách chia đi từ 24 đối 8 sang 1 đối 31. Số lượt và số bước warp đã tiêu
              thay đổi thế nào?
            </li>
            <li>
              Kéo T lên 32, rồi xem các warp khác của khối. Sau đó đặt độ rộng là 64 và xem lại.
            </li>
            <li>Đổi thứ tự các lượt sang xen kẽ. Những số đọc nào thay đổi?</li>
            <li>
              Đặt A và B mỗi nhánh 1 lệnh, rồi mỗi nhánh 8 lệnh, và so sánh rẽ nhánh với “tính cả hai rồi chọn” ở cả hai kích thước: trước
              tiên ở T = 24, rồi ở T = 32.
            </li>
          </ol>
        }
      >
        <ol>
          <li>
            Slide T from 24 to 16, then to 1. The split goes from 24 against 8 to 1 against 31. What happens to the number of passes and to
            the warp-steps spent?
          </li>
          <li>
            Slide T to 32, then look at the other warps of the block. Then set the width to 64 and look again.
          </li>
          <li>Switch the order of the passes to interleaved. Which readouts change?</li>
          <li>
            Set A and B to 1 instruction each, then to 8 each, and compare the branch with “compute both and select” at both sizes: first
            at T = 24, then at T = 32.
          </li>
        </ol>
      </Translated>
      <Translated
        vi={
          <p>
            Điều bạn lẽ ra đã thấy: chi phí đi theo số nhánh khác nhau bên trong một warp, không phải số lane đi mỗi nhánh. Một lane
            lệch nhịp khiến warp tốn trọn một lượt thêm. Một ngưỡng rơi đúng ranh giới warp không tốn gì, vì khi đó mỗi warp đều
            đồng nhất; và ranh giới nằm ở đâu thì tùy vào độ rộng. Phép chọn không bao giờ thắng trong mô hình này: ở một warp bị tách nó tốn
            nhiều hơn rẽ nhánh một bước, chính là phép chọn, với bất kỳ độ dài nhánh nào, và ở một warp đồng nhất nó còn trả giá cho nhánh mà lệnh rẽ nhánh
            bỏ qua.
          </p>
        }
      >
        <p>
          What you should have found: the cost follows the number of different paths inside one warp, not how many lanes take each. One lane
          out of step costs the warp a whole extra pass. A threshold that falls on a warp boundary costs nothing, because each warp is then
          uniform; and where the boundaries are depends on the width. Selecting never comes out ahead in this model: in a split warp it costs
          one step more than the branch, the select itself, at any length of path, and in a uniform warp it also pays for the path the branch
          skips.
        </p>
      </Translated>

      <h2><Translated vi="Vòng lặp: các lane rời cuộc chơi">Loops: lanes drop out</Translated></h2>
      <Translated
        vi={
          <p>
            Một vòng lặp phân kỳ khi số vòng lặp khác nhau giữa các luồng của một warp. Trong hình của sách, mỗi luồng đọc số vòng lặp của riêng nó
            từ dữ liệu, <code>N = a[threadIdx.x]</code>, và các số này từ bốn đến tám. Trong bốn lần lặp đầu, mọi luồng đều
            hoạt động. Từ lần thứ năm trở đi, các luồng đã xong chuyển sang không hoạt động trong khi số còn lại tiếp tục, cho đến khi chỉ còn những luồng dài nhất.
          </p>
        }
      >
        <p>
          A loop diverges when its trip count differs between the threads of a warp. In the book's figure each thread reads its own count
          from data, <code>N = a[threadIdx.x]</code>, and the counts range from four to eight. For the first four iterations every thread is
          active. From the fifth on, threads that have finished go inactive while the rest continue, until only the longest are left.
        </p>
      </Translated>
      <Source section="4.5" pages="81" figs={tr('Fig. 4.10', 'Hình 4.10')} />
      <Translated
        vi={
          <p>
            Chọn vòng lặp trong danh sách câu lệnh của trình mô phỏng. Số vòng lặp ban đầu là tám giá trị in trong hình, lặp lại khắp
            warp; nhấp vào một cột để đổi một lane. Warp chạy số lần lặp bằng lane dài nhất của nó, nên chỉ một luồng dài cũng quyết định
            chi phí cho cả 32. Hãy thử mẫu có sẵn với một lane 10 và còn lại 2, rồi đọc tỉ lệ bước lane có làm việc.
          </p>
        }
      >
        <p>
          Choose the loop in the simulator's statement list. Its trip counts start as the figure's eight printed values, repeated across the
          warp; click a column to change a lane. The warp runs for as many iterations as its longest lane, so a single long thread sets the
          cost for all 32. Try the preset with one lane of 10 and the rest 2, and read the share of lane-steps that did work.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Hình và lời văn của nó không khớp nhau. Các cột được đánh nhãn 0, 1, 2, 4, 5, 6, 7 (không có 3), và lời văn nói rằng ở
              lần lặp thứ tám chỉ luồng 1 và 7 chạy, trong khi các nhãn in đặt hai số 8 dưới 0 và 7. Trình mô phỏng dùng
              các số in, 8, 6, 7, 4, 5, 6, 8, 7, và tự đánh số các lane.
            </p>
          }
        >
          <p>
            The figure and its text disagree. The columns are labelled 0, 1, 2, 4, 5, 6, 7 (there is no 3), and the text says that in the
            eighth iteration only threads 1 and 7 run, where the printed labels put the two counts of 8 under 0 and 7. The simulator uses
            the printed counts, 8, 6, 7, 4, 5, 6, 8, 7, and numbers the lanes itself.
          </p>
        </Translated>
      </Aside>

      <h2><Translated vi="Các lượt chạy theo thứ tự nào">In what order the passes run</Translated></h2>
      <Translated
        vi={
          <p>
            Đến giờ các lượt diễn ra nối tiếp nhau. Sách ghi rõ mốc thời gian của điều đó: trên kiến trúc Pascal của NVIDIA và trước đó, một nhánh chạy đến
            hết rồi mới đến nhánh kia. Từ Volta trở đi, các lượt có thể xen kẽ, một tính năng gọi là independent thread scheduling. Công tắc
            thứ tự của trình mô phỏng cho thấy hệ quả đối với việc đếm, tức là không có gì: cùng những lệnh đó được phát theo một thứ tự khác.
          </p>
        }
      >
        <p>
          So far the passes came one after the other. The book dates that: on NVIDIA's Pascal architecture and earlier, one path runs to
          completion and then the other. From Volta on, the passes may be interleaved, a feature called independent thread scheduling. The
          simulator's order switch shows the consequence for counting, which is none: the same instructions are issued in a different order.
        </p>
      </Translated>
      <Source section="4.5" pages="80–81" />
      <Translated
        vi={
          <p>
            Hệ quả đối với tính đúng đắn thì có thật. Bạn không thể giả định các luồng của một warp đã hợp lại sau hai nhánh.
            Nếu mọi luồng của warp phải xong một giai đoạn trước khi bất kỳ luồng nào đi tiếp, hãy nói rõ điều đó bằng một rào chắn (barrier) cấp warp,{' '}
            <code>__syncwarp()</code>. Hình 4.9 vẽ C thành một lượt với mọi lane hoạt động; lời văn chỉ nói các luồng “có thể” hợp lại, nên
            hãy đọc hình vẽ như một điều có thể xảy ra, không phải một sự đảm bảo.
          </p>
        }
      >
        <p>
          The consequence for correctness is real. You cannot assume that the threads of a warp have come back together after the two paths.
          If every thread of the warp must finish a phase before any goes on, say so with a warp-level barrier,{' '}
          <code>__syncwarp()</code>. Fig. 4.9 draws C as one pass with all lanes active; the text says only that the threads “may” reconverge, so
          read the drawing as something that can happen, not as a guarantee.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A10, B5, C2)" /* i18n-ignore: date and citation */>
        <Translated
          vi={
            <p>
              NVIDIA: các lượt có thể xen kẽ từ compute capability 7.0, và hướng dẫn hiện tại của hãng bắt đầu từ 7.5. AMD ghi rằng các lane chạy đồng bộ,
              hai nhánh được tuần tự hóa. Apple không nói gì.
            </p>
          }
        >
          <p>
            NVIDIA: the passes may interleave from compute capability 7.0, and its current guide starts at 7.5. AMD documents lockstep,
            both branches serialised. Apple does not say.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Rào chắn cũng là chỗ mà một tệp nguồn thôi phục vụ được hai backend. <code>__syncwarp()</code> không đối số biên dịch được
            trên CUDA và trên HIP. Dạng có mặt nạ của CUDA nhận một <code>unsigned</code> 32-bit; dạng của HIP chỉ nhận số nguyên 64-bit và từ chối{' '}
            <code>__syncwarp(0xFFFFFFFF)</code> lúc biên dịch. Bản tương ứng của Metal là <code>simdgroup_barrier</code>, nhận các cờ bộ nhớ
            và không nhận mặt nạ.
          </p>
        }
      >
        <p>
          The barrier is also a place where one source file stops serving two backends. <code>__syncwarp()</code> with no argument compiles
          on CUDA and on HIP. CUDA's form with a mask takes a 32-bit <code>unsigned</code>; HIP's takes only a 64-bit integer and rejects{' '}
          <code>__syncwarp(0xFFFFFFFF)</code> at compile time. Metal's counterpart is <code>simdgroup_barrier</code>, which takes memory flags
          and no mask.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách nêu tên <code>__syncwarp()</code> và chỉ dùng CUDA. Header của HIP và hàm của Metal lấy từ
              notes/today/ch04-facts.md B4 và C3. <code>__syncwarp</code> của HIP có trong header của nó nhưng không có trên trang mở rộng ngôn ngữ.
            </p>
          }
        >
          <p>
            The book names <code>__syncwarp()</code> and uses CUDA only. The HIP header and the Metal function are from
            notes/today/ch04-facts.md B4 and C3. HIP's <code>__syncwarp</code> is in its header and not on its language-extensions page.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS45['ch04/4.5/q-ex5']} />

      <h2><Translated vi="Nhận ra phân kỳ trong mã nguồn">Spotting divergence in source</Translated></h2>
      <Translated
        vi={
          <p>
            Hãy nhìn vào điều kiện. Nếu giá trị của nó có thể khác nhau giữa các luồng của một warp, câu lệnh có thể phân kỳ. Quy tắc của sách nêu
            các điều kiện phụ thuộc vào <code>threadIdx</code>: <code>if (threadIdx.x &gt; 2)</code> tách warp đầu của mọi khối thành
            luồng 0, 1 và 2 đối với phần còn lại, và một vòng lặp bị chặn bởi chỉ số luồng cũng có thể phân kỳ như vậy. Hình 4.10 cho thấy dữ liệu riêng của từng luồng
            cũng gây ra điều đó.
          </p>
        }
      >
        <p>
          Look at the condition. If its value can differ between the threads of a warp, the statement can diverge. The book's rule names
          conditions that depend on <code>threadIdx</code>: <code>if (threadIdx.x &gt; 2)</code> splits the first warp of every block into
          threads 0, 1 and 2 against the rest, and a loop bounded by a thread index can diverge the same way. Fig. 4.10 shows that per-thread
          data does it too.
        </p>
      </Translated>
      <Source section="4.5" pages="81" />
      <Translated
        vi={
          <p>
            Bài tập đầu tiên của sách yêu cầu bạn đếm điều này cho một kernel ngắn, dùng ba thuật ngữ mà chương không hề định nghĩa. Trang này dùng
            chúng như sau. Một warp <b>hoạt động</b> tại một câu lệnh khi có ít nhất một luồng của nó thực thi câu lệnh đó. Nó <b>bị phân kỳ</b> ở đó
            khi một số luồng của nó thực thi và một số thì không. <b>Hiệu suất SIMD</b> của nó tại câu lệnh là số luồng
            thực thi câu lệnh chia cho độ rộng của warp.
          </p>
        }
      >
        <p>
          The book's first exercise asks you to count this for a short kernel, using three terms the chapter never defines. This page uses
          them as follows. A warp is <b>active</b> at a statement when at least one of its threads executes it. It is <b>divergent</b> there
          when some of its threads execute it and some do not. Its <b>SIMD efficiency</b> at the statement is the number of threads that
          execute it divided by the width of the warp.
        </p>
      </Translated>
      <Beyond title={tr('Our definitions', 'Định nghĩa của chúng tôi')}>
        <Translated
          vi={
            <p>
              “SIMD efficiency” và “active” chỉ xuất hiện trong bài tập 1 (p. 91). Hướng dẫn của AMD dùng thuật ngữ thứ nhất một lần mà không định nghĩa và
              định nghĩa một thước đo khác, branch efficiency (facts B7). Các định nghĩa ở đây là của chúng tôi.
            </p>
          }
        >
          <p>
            “SIMD efficiency” and “active” appear only in exercise 1 (p. 91). AMD's guide uses the first term once without defining it and
            defines a different metric, branch efficiency (facts B7). The definitions here are ours.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Đây là kernel của bài tập và lần khởi chạy của nó: <code>N = 1024</code> phần tử trong các khối 128 luồng, vậy là {E1.blocks} khối, mỗi khối{' '}
            {E1.warpsPerBlock} warp ở độ rộng 32, như bạn đã đếm ở mục 4.4. Dòng 4, dòng 7 và vòng lặp ở dòng 9 là ba loại
            điều kiện khác nhau.
          </p>
        }
      >
        <p>
          Here is the exercise's kernel and its launch: <code>N = 1024</code> elements in blocks of 128 threads, so {E1.blocks} blocks of{' '}
          {E1.warpsPerBlock} warps at a width of 32, as you counted in section 4.4. Line 4, line 7 and the loop on line 9 are three different
          kinds of condition.
        </p>
      </Translated>
      <Code
        tabs={[{ label: 'foo_kernel.cu', code: FOO, status: tr('front-end-checked; never run', 'front-end-checked; chưa từng chạy') }]}
        highlight={[3, 4, 6, 7, 9, 10]}
        caption={
          <Translated
            vi={
              <>
                Bài tập 1 của sách (pp. 90–91), lưu ở <code>labs/ch04/snippets/foo_kernel.cu</code>. Nó không có kiểm tra biên vì N là
                bội số của kích thước khối.
              </>
            }
          >
            <>
              The book's exercise 1 (pp. 90–91), stored as <code>labs/ch04/snippets/foo_kernel.cu</code>. It has no boundary check because N is
              a multiple of the block size.
            </>
          </Translated>
        }
      />
      <Ex1Predict />

      <h2><Translated vi="Kiểm tra biên, trường hợp thường gặp">Boundary checks, the common case</Translated></h2>
      <Translated
        vi={
          <p>
            Phần lớn phân kỳ trong kernel thực tế đến từ một nguồn. Lưới là một số nguyên khối, dữ liệu thì có kích thước bất kỳ, và
            vì vậy kernel kiểm tra xem luồng của nó có phần tử nào không: chính là <code>if (i &lt; n)</code> của phép cộng vector. Warp
            chứa điểm kết thúc của dữ liệu có luồng ở cả hai phía của phép thử.
          </p>
        }
      >
        <p>
          Most divergence in real kernels comes from one source. A grid is a whole number of blocks, the data is whatever size it is, and
          so the kernel tests whether its thread has an element at all: the <code>if (i &lt; n)</code> of vector addition. The warp that
          contains the end of the data has threads on both sides of the test.
        </p>
      </Translated>
      <Source section="4.5" pages="81" />
      <Translated
        vi={
          <p>
            Lấy các con số của sách: một vector 1003 phần tử trong các khối 64 luồng. Tức là {V1003.blocks} khối và {big(V1003.threads)}{' '}
            luồng, trong đó {V1003_SPLIT.invalid} luồng cuối không được làm gì. {V1003.blocks} khối này là {V1003.warps} warp, và chỉ một
            trong số đó, warp thứ hai của khối cuối, có luồng ở cả hai phía. {V1003.warps - 1} warp còn lại đều đồng nhất.
          </p>
        }
      >
        <p>
          Take the book's numbers: a vector of 1003 elements in blocks of 64 threads. That is {V1003.blocks} blocks and {big(V1003.threads)}{' '}
          threads, of which the last {V1003_SPLIT.invalid} must do nothing. The {V1003.blocks} blocks are {V1003.warps} warps, and only one of
          them, the second warp of the last block, has threads on both sides. The other {V1003.warps - 1} are uniform.
        </p>
      </Translated>
      <Source section="4.5" pages="81–82" />
      <Translated
        vi={
          <p>
            Vậy thiệt hại là một warp, và nó đáng kể đến đâu tùy vào có bao nhiêu warp. Với 100 phần tử, một trên bốn warp
            bị phân kỳ, điều có thể đáng kể. Với 1000 thì là một trên 32, khoảng 3%: kể cả nếu warp đó mất gấp đôi thời gian, cả lần khởi chạy cũng chỉ
            lâu hơn khoảng 3%. Với 10,000 thì là một trên khoảng ba trăm. Phân kỳ ở biên là một chi phí cố định mà dữ liệu lớn làm loãng đi.
          </p>
        }
      >
        <p>
          So the damage is one warp, and what it amounts to depends on how many warps there are. With 100 elements one warp of four
          diverges, which can matter. With 1000 it is one of 32, about 3%: even if that warp took twice as long, the whole launch would take
          about 3% longer. With 10,000 it is one of some three hundred. Boundary divergence is a fixed cost that large data dilutes.
        </p>
      </Translated>
      <Source section="4.5" pages="82" />
      <Translated
        vi={
          <p>
            Ở hai chiều, mép dài hơn. Sách quay lại bức ảnh của chương 3, rộng 76 pixel và cao 62, được phủ bằng các khối 16 × 16:
            {' '}{P76_32.blocks} khối, mỗi khối 8 warp, mỗi warp là hai hàng của khối. Sách chia các khối thành bốn vùng.{' '}
            {P76_R[0].blocks} khối nằm trọn trong bức ảnh không phân kỳ. {P76_R[1].blocks} khối bị mép phải cắt có 12 cột hợp lệ
            và 4 cột không hợp lệ ở mỗi hàng, nên cả {P76_R[1].divergent} warp của chúng đều phân kỳ. {P76_R[2].blocks} khối bị mép dưới
            cắt hoàn toàn không phân kỳ: mép rơi vào giữa hai warp, và warp bên dưới nằm trọn bên ngoài. Khối góc có bảy
            warp phân kỳ và một warp nằm ngoài. Tổng cộng {P76_32.divergent} trên {P76_32.warps}.
          </p>
        }
      >
        <p>
          In two dimensions the edge is longer. The book returns to chapter 3's picture, 76 pixels wide and 62 high, covered by 16 × 16
          blocks: {P76_32.blocks} blocks, each of 8 warps, each warp two rows of its block. It sorts the blocks into four regions. The{' '}
          {P76_R[0].blocks} blocks wholly inside the picture do not diverge. The {P76_R[1].blocks} blocks the right edge cuts have 12 valid
          and 4 invalid columns in every row, so all {P76_R[1].divergent} of their warps diverge. The {P76_R[2].blocks} blocks the bottom edge
          cuts do not diverge at all: the edge falls between two warps, and the warp below it is wholly outside. The corner block has seven
          divergent warps and one outside. That is {P76_32.divergent} of {P76_32.warps}.
        </p>
      </Translated>
      <Source section="4.5" pages="82" />
      <Translated
        vi={
          <p>
            Mép dưới gặp may: 62 là số chẵn, và một warp là hai hàng. Sách lưu ý rằng một chiều cao lẻ cũng sẽ tách các warp đó.
            Hãy tự đếm ở dưới, rồi mở rộng bức ảnh và xem tỉ lệ giảm xuống.
          </p>
        }
      >
        <p>
          The bottom edge got lucky: 62 is even, and a warp is two rows. The book notes that an odd height would split those warps too.
          Count it yourself below, then widen the picture and watch the share fall.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Bộ đếm áp dụng cách cắt mà NVIDIA công bố, với độ rộng bạn chọn. AMD công bố thứ tự tuyến tính và phép tính lane
              nhưng không nêu cách cắt trong một câu, còn cách cắt của Metal do hiện thực quyết định (facts B6, C2). Nó đếm số warp bị tách. Nó không nói gì
              về chi phí của một warp bị tách, điều phụ thuộc vào việc kernel làm gì sau phép thử.
            </p>
          }
        >
          <p>
            The counter applies the cut NVIDIA documents, with the width you choose. AMD documents the linear order and the lane arithmetic
            but not the cut in one sentence, and Metal's cut is implementation-defined (facts B6, C2). It counts split warps. It says nothing
            about what a split warp costs, which depends on what the kernel does after the test.
          </p>
        </Translated>
      </Simplification>
      <BoundaryCounter />
      <Translated vi={<p>Những gì bộ đếm cho thấy, theo thứ tự bạn có khả năng gặp:</p>}>
        <p>What the counter shows, in the order you are likely to meet it:</p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>Ở một chiều, nhiều nhất một warp bị tách, với mọi kích thước, và không warp nào khi dữ liệu kết thúc đúng ranh giới warp.</li>
            <li>
              Ở hai chiều với warp là trọn các hàng, mép phải tách mọi warp nó cắt qua, còn mép dưới chỉ tách một warp
              nếu nó rơi vào bên trong warp đó. Các warp bị tách xấp xỉ một cột khối trên tổng số cột, nên tỉ lệ giảm khi
              bức ảnh rộng ra. Quy tắc kinh nghiệm của sách là dưới 2% với ảnh rộng hơn 1000 pixel.
            </li>
            <li>
              Hai con số in trong sách không đứng vững khi đếm lại. “Một trên 313” warp cho 10,000 phần tử thực ra là một trên{' '}
              {V10000.warps} warp được khởi chạy: 313 warp có phần tử và thêm một warp được khởi chạy mà rảnh. Còn 80 trên 1040 cho bức ảnh
              “200 × 150” là {WIDE_B.divergent} nếu 200 là chiều rộng và {TALL_B.divergent} nếu 150 là chiều rộng. Cả hai đều không thay đổi kết luận của sách với
              dữ liệu lớn.
            </li>
          </ul>
        }
      >
        <ul>
          <li>In one dimension at most one warp is split, at any size, and none when the data ends on a warp boundary.</li>
          <li>
            In two dimensions with warps that are whole rows, a right edge splits every warp it crosses, while a bottom edge splits a warp
            only if it falls inside one. The split warps are roughly one block-column out of all the columns, so the share falls as the
            picture gets wider. The book's rule of thumb is under 2% for pictures more than 1000 pixels wide.
          </li>
          <li>
            Two of the book's printed figures do not survive a recount. Its “one of 313” warps for 10,000 elements is one of{' '}
            {V10000.warps} launched: 313 warps hold an element and one more is launched and idle. And its 80 of 1040 for the “200 × 150”
            picture is {WIDE_B.divergent} if 200 is the width and {TALL_B.divergent} if 150 is. Neither changes the book's conclusion for
            large data.
          </li>
        </ul>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Các lần đếm lại này là của chúng tôi (notes/ch04.md, mục 4 của “Unclear in the PDF”; notes/today/ch04-facts.md E1). Sách viết kích thước theo dạng
              dọc × ngang, nên “200 × 150” sẽ rộng 150; nhưng kích thước các vùng in trong sách khớp với chiều rộng 200.
            </p>
          }
        >
          <p>
            These recounts are ours (notes/ch04.md, “Unclear in the PDF” item 4; notes/today/ch04-facts.md E1). The book writes sizes as
            vertical × horizontal, which would make “200 × 150” 150 wide; its printed region sizes match 200 wide.
          </p>
        </Translated>
      </Aside>
      <Ask q={questionsS45['ch04/4.5/q-ex2']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS45['ch04/4.5/q-ex3']} />
      <Ask q={questionsS45['ch04/4.5/q-200x150']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Độ rộng làm thay đổi số đếm hai chiều">The width changes the two-dimensional counts</Translated></h2>
      <Translated
        vi={
          <p>
            Mọi số đếm ở trên đều giả định warp rộng 32. Chuyển bộ đếm sang 64 thì các trường hợp một chiều vẫn chỉ có một warp bị tách, nhưng
            bức ảnh thì khác: một khối 16 × 16 giờ là bốn warp, mỗi warp bốn hàng, mép dưới ở hàng 62 rơi vào bên trong một warp, và
            cùng bức ảnh có {P76_64.divergent} warp bị tách trên {P76_64.warps}, trong khi trước đó là {P76_32.divergent} trên {P76_32.warps}. Một ranh giới
            miễn phí ở độ rộng này lại không miễn phí ở độ rộng kia.
          </p>
        }
      >
        <p>
          Every count above assumed 32-wide warps. Switch the counter to 64 and the one-dimensional cases keep their single split warp, but
          the picture changes: a 16 × 16 block is now four warps of four rows each, the bottom edge at row 62 falls inside a warp, and the
          same picture has {P76_64.divergent} split warps of {P76_64.warps} where it had {P76_32.divergent} of {P76_32.warps}. A boundary
          that was free at one width is not at the other.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ đếm ở 32. Các số đếm ở độ rộng 64 là mô hình của trang này áp dụng với độ rộng mà HIP công bố cho các thiết bị gfx9
              (notes/today/ch04-facts.md E1, B6).
            </p>
          }
        >
          <p>
            The book counts at 32 only. The 64-wide counts are this page's model applied with the width HIP documents for gfx9 devices
            (notes/today/ch04-facts.md E1, B6).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Cái bẫy tương tự có trong tài liệu của nhà cung cấp. Hướng dẫn hiệu năng của AMD đưa <code>if (threadIdx.x &lt; 32)</code> làm ví dụ về nhánh không
            phân kỳ. Ở nơi warp gồm 32 luồng, điều đó đúng: phép thử rơi đúng ranh giới warp. Nhưng chính các thiết bị Instinct của AMD có warp
            rộng 64. Hãy đặt trình mô phỏng ở 64 lane và T = 32 rồi xem warp 0.
          </p>
        }
      >
        <p>
          The same trap is in vendor documentation. AMD's performance guide shows <code>if (threadIdx.x &lt; 32)</code> as a branch with no
          divergence. Where a warp is 32 threads, that holds: the test falls on a warp boundary. AMD's own Instinct devices have 64-wide
          warps. Set the simulator to 64 lanes and T = 32 and look at warp 0.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md B7)" /* i18n-ignore: date and citation */>
        <Translated
          vi={
            <p>
              Ví dụ của AMD được đánh dấu “Good: no divergence”, kèm chú thích “All threads in first half-warp execute”. Việc nó phân kỳ trên một
              thiết bị rộng 64 là cách đọc của chúng tôi, theo phép tính của cách cắt.
            </p>
          }
        >
          <p>
            AMD's example is marked “Good: no divergence”, with the comment “All threads in first half-warp execute”. That it diverges on a
            64-wide device is our reading, by the arithmetic of the cut.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS45['ch04/4.5/q-amd32']} />

      <h2><Translated vi="Trên Metal có thể không có biên nào để kiểm tra">On Metal there may be no boundary to check</Translated></h2>
      <Translated
        vi={
          <p>
            Metal có hai cách khởi chạy. <code>dispatchThreadgroups</code> khởi chạy trọn các threadgroup, giống một lưới CUDA, và kernel
            cần cùng phép kiểm tra biên. <code>dispatchThreads</code> nhận đúng kích thước của dữ liệu và không khởi chạy luồng nào nằm ngoài:
            các threadgroup ở mép chỉ đơn giản là nhỏ hơn. Với lời gọi đó thì không có kiểm tra biên, nên mép không có gì để
            gây phân kỳ.
          </p>
        }
      >
        <p>
          Metal offers two ways to launch. <code>dispatchThreadgroups</code> launches whole threadgroups, like a CUDA grid, and the kernel
          needs the same boundary check. <code>dispatchThreads</code> takes the exact size of the data and launches no thread outside it:
          the threadgroups on the edge are simply smaller. With that call there is no boundary check, so there is nothing for an edge to
          diverge on.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Apple: “You don't need to check a thread's position in a grid if you use the dispatchThreads(_:threadsPerThreadgroup:)
              technique.” (notes/today/ch05-facts.md C5; ch04-facts.md C4). Backend MPS của PyTorch khởi chạy theo cách này (facts D4).
            </p>
          }
        >
          <p>
            Apple: “You don't need to check a thread's position in a grid if you use the dispatchThreads(_:threadsPerThreadgroup:)
            technique.” (notes/today/ch05-facts.md C5; ch04-facts.md C4). PyTorch's MPS backend launches this way (facts D4).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Điều đó loại bỏ nguyên nhân phổ biến nhất, chứ không loại bỏ cơ chế. Một kernel Metal có nhánh phụ thuộc dữ liệu vẫn có các luồng của cùng một
            SIMD group muốn những lệnh khác nhau, và đến đây tài liệu dừng lại: đặc tả của Apple nói một luồng có thể không hoạt động
            “due to flow control” và không mô tả lượt nào cũng như chi phí của chúng. Trên Metal, bức tranh của mục này là một giả thuyết, và lab
            4.2 là cách bạn kiểm chứng nó trên thiết bị của chính mình.
          </p>
        }
      >
        <p>
          That removes the commonest cause, not the mechanism. A Metal kernel with a data-dependent branch still has threads of one
          SIMD-group wanting different instructions, and here the documentation stops: Apple's specification says a thread can be inactive
          “due to flow control” and describes neither passes nor their cost. On Metal the picture of this section is a hypothesis, and lab
          4.2 is how you test it on your own device.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md C4)" /* i18n-ignore: date and citation */>
        <Translated
          vi={
            <p>
              Mô hình nhiều lượt được NVIDIA và AMD công bố. Không nguồn nào của Apple mà khóa học này đã đọc mô tả các nhánh phân kỳ trong một
              SIMD group thực thi ra sao.
            </p>
          }
        >
          <p>
            The multi-pass model is documented by NVIDIA and by AMD. No Apple source read for this course describes how divergent paths in a
            SIMD-group execute.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Nơi bạn đã gặp cả hai dạng">Where you have met both shapes</Translated></h2>
      <Translated
        vi={
          <p>
            Cả hai hình của mục này đã có sẵn trong các mô hình bạn chạy. Một kernel giao cho mỗi chuỗi của batch một luồng và lặp
            tới độ dài của chuỗi đó chính là Hình 4.10: số vòng lặp lấy từ dữ liệu. Một causal mask, <code>col &lt;= row</code>, là Hình 4.9: một{' '}
            <code>if</code> theo vị trí của luồng. Dạng thứ hai rẻ vì cùng lý do kiểm tra biên rẻ, điều mà câu hỏi bên dưới
            bắt bạn đếm. Dạng thứ nhất mặc định không rẻ, và cách sửa là cách bạn vốn đã áp dụng vì một lý do khác.
          </p>
        }
      >
        <p>
          Both figures of this section are already in the models you run. A kernel that gives each sequence of a batch one thread and loops
          to that sequence's length is Fig. 4.10: trip counts from data. A causal mask, <code>col &lt;= row</code>, is Fig. 4.9: an{' '}
          <code>if</code> on the thread's position. The second is cheap for the reason boundary checks are cheap, which the question below
          has you count. The first is not cheap by default, and the fix is one you already apply for another reason.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Sách không bàn về batch hay attention. Cầu nối này, batch bên dưới và cả hai câu hỏi là của chúng tôi.</p>}>
          <p>The book does not discuss batches or attention. This bridge, the batch below and both questions are ours.</p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Bên dưới, mỗi luồng chạy một vòng lặp có độ dài lấy từ 1 đến 64, như độ dài trong một batch có thể có. Hình trên xếp các
            luồng vào warp theo thứ tự đến, hình dưới theo độ dài đã sắp xếp.
          </p>
        }
      >
        <p>
          Below, every thread runs a loop whose length is drawn from 1 to 64, as lengths in a batch might be. The top drawing puts the
          threads in warps in arrival order, the bottom one sorted by length.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Ở đây một lần lặp warp tốn như nhau dù chỉ một lane hay tất cả các lane còn đang lặp, và ngoài ra hai thứ tự không khác gì
              nhau. Trên thiết bị thật, việc sắp xếp còn thay đổi vùng bộ nhớ mà mỗi warp chạm tới, điều mô hình này bỏ qua.
            </p>
          }
        >
          <p>
            A warp-iteration costs the same here whether one lane or all of them are still looping, and nothing else differs between the two
            orders. On a device, sorting also changes which memory each warp touches, which this model ignores.
          </p>
        </Translated>
      </Simplification>
      <LengthsBatch />
      <Translated
        vi={
          <p>
            Khi sắp xếp, mỗi warp chứa các độ dài gần bằng nhau và dừng lại xấp xỉ khi việc của nó xong. Khi xáo, gần như warp nào cũng chứa một trong các
            chuỗi dài và chạy tới hết chuỗi đó. Gom nhóm batch theo độ dài, việc bạn làm để giảm padding, có tác dụng phụ là làm các warp đồng nhất.
          </p>
        }
      >
        <p>
          Sorted, each warp holds near-equal lengths and stops about when its work is done. Shuffled, nearly every warp contains one of the
          long sequences and runs to it. Bucketing a batch by length, which you do to cut padding, makes warps uniform as a side effect.
        </p>
      </Translated>
      <Ask q={questionsS45['ch04/4.5/q-buckets']} />
      <Ask q={questionsS45['ch04/4.5/q-causal']} />

      <h2><Translated vi="Bạn có thể làm gì">What you can do about it</Translated></h2>
      <Translated
        vi={
          <p>
            Trước hết hãy quyết định có nên làm gì không. Kiểm tra biên trên dữ liệu lớn chỉ là nhiễu. Một nhánh có kết quả như nhau trong một warp
            không tốn gì dù được đi vào thường xuyên đến đâu. Điều đáng để bạn bỏ công là một nhánh hay một cận vòng lặp khác nhau giữa các luồng
            lân cận trên phần lớn lưới, với khối lượng việc thật trên mỗi nhánh.
          </p>
        }
      >
        <p>
          First decide whether to do anything. A boundary check on large data is noise. A branch whose outcome is the same across a warp
          costs nothing however often it is taken. What is worth your time is a branch or a loop bound that differs between neighbouring
          threads over much of the grid, with real work on each path.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Sau đó có hai cách chữa. Cách thứ nhất là sắp xếp công việc để mỗi warp đồng nhất: sắp xếp hoặc gom nhóm dữ liệu, hoặc tách các
            trường hợp thành những lần khởi chạy riêng. Cách thứ hai là bỏ lệnh rẽ nhánh bằng cách tính cả hai nhánh trong mọi luồng rồi chọn một kết quả.
            Dòng so sánh của trình mô phỏng cho thấy cách thứ hai tốn bao nhiêu: giờ mọi lane trả giá cho cả hai nhánh, điều mà một warp bị tách
            vốn đã trả, cộng thêm phép chọn. Vậy trong mô hình này nó không bao giờ thắng lệnh rẽ nhánh mà nó thay thế. Mô hình coi bản thân lệnh rẽ nhánh không tốn gì;
            chi phí đó trên thiết bị thật là bao nhiêu thì đó chính là chỗ phép chọn có thể thắng, nếu có, với các nhánh dài vài lệnh. Không nhà cung cấp nào công bố
            điều này, và trình biên dịch có thể tự hoán đổi mà không hỏi bạn.
          </p>
        }
      >
        <p>
          Then there are two remedies. The first is to arrange the work so that each warp is uniform: sort or bucket the data, or split the
          cases into separate launches. The second is to remove the branch by computing both paths in every thread and selecting a result.
          The simulator's comparison line shows what the second costs: every lane now pays for both paths, which is what a split warp was
          already paying, plus the select. So in this model it never beats the branch it replaces. The model gives the branch itself no cost;
          whatever that cost is on a device is where a select could win, if anywhere, for paths a few instructions long. No vendor documents
          it, and a compiler may make the swap without asking you.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách dừng lại ở việc đếm. Hướng dẫn của AMD liệt kê cả hai cách chữa, các lần khởi chạy riêng và “predication (branchless programming)”
              (facts B7); việc trình biên dịch hoán đổi là facts A9. Khi nào mỗi cách có lợi là nhận định của chúng tôi: lab 4.2 chỉ đo phép chọn trên nhánh dài, còn nhánh ngắn
              chưa được kiểm chứng ở đây.
            </p>
          }
        >
          <p>
            The book stops at counting. AMD's guide lists both remedies, separate launches and “predication (branchless programming)”
            (facts B7); the compiler's swap is facts A9. When each pays is ours: lab 4.2 times the select on long paths only, and short paths
            are untested here.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS45['ch04/4.5/q-select']} />

      <h2><Translated vi="Đo nó">Measure it</Translated></h2>
      <Translated
        vi={
          <p>
            Mọi thứ trên trang này đều đếm lượt. Lab 4.2 đo thời gian của chúng. Nó chạy một kernel có hai nhánh dài bằng nhau trên cùng dữ liệu theo
            nhiều cách bố trí: tất cả một nhánh, hai nhánh được xếp vào các warp riêng, và hai nhánh trộn lẫn trong mọi warp. Nó cũng
            đo thời gian phiên bản phép chọn, và một vòng lặp với số vòng lặp đã sắp xếp và đã xáo. Khối lượng việc giống hệt nhau trong mỗi phép so sánh; chỉ có việc ai
            chung warp với ai là thay đổi.
          </p>
        }
      >
        <p>
          Everything on this page counted passes. Lab 4.2 times them. It runs one kernel with two equally long paths over the same data in
          several arrangements: all one path, the two paths sorted into separate warps, and the two paths mixed inside every warp. It also
          times the select version, and a loop with sorted and shuffled trip counts. The work is identical in each comparison; only who
          shares a warp changes.
        </p>
      </Translated>
      <Predict
        id={DIVCOST_ID}
        prompt={
          <Translated
            vi={
              <>
                Cùng khối lượng việc, với hai nhánh trộn nửa này nửa kia trong mọi warp thay vì được sắp xếp để mỗi warp đồng nhất. Trên GPU
                {' '}<em>của bạn</em> nó sẽ mất thời gian gấp bao nhiêu lần? Hãy chốt một con số trước khi chạy lab 4.2.
              </>
            }
          >
            <>
              The same work, with the two paths mixed half and half inside every warp instead of sorted so that each warp is uniform. How many
              times as long will it take on <em>your</em> GPU? Commit a number before you run lab 4.2.
            </>
          </Translated>
        }
        unit={tr('times as long', 'lần thời gian')}
      >
        {(v) => <MeasuredDivergence predicted={v} />}
      </Predict>
      <Translated
        vi={
          <p>
            Lab nằm trên <a href={href('ch04', 'labs')}>trang lab</a>. Tiếp theo: vì sao một SM được giao nhiều warp hơn hẳn số nó có thể thực thi
            cùng lúc, và những warp không đang thực thi thì đang chờ gì.
          </p>
        }
      >
        <p>
          The lab is on the <a href={href('ch04', 'labs')}>labs page</a>. Next: why an SM is given far more warps than it can execute at
          once, and what the ones that are not executing are waiting for.
        </p>
      </Translated>
    </>
  );
}
