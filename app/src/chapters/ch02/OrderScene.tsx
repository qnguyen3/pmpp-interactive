import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import {
  OP_BODY,
  OP_NOTE,
  defaults,
  reads,
  runSequential,
  runThreads,
  threadOrder,
  type OrderName,
  type Params,
} from '../../sim/depend';
import { useNarrow } from './narrow';
import { Translated, tr } from '../../lib/i18n';

// "Same loop, another order": two loops over 8 numbers, each run with one thread
// per iteration in loop order and then backwards. Every number drawn comes from
// sim/depend.ts. Six beats, one narration script each
// (content/ch02/narration/order-N.txt).
//
// The 8 numbers are the ones the book uses for prefix sum in chapter 11 (p. 252),
// so the in-place loop's answer here, 3 4 11 11 15 16 22 25, is the book's too.
export const SCENE_X = [3, 1, 7, 0, 4, 1, 6, 3];
const N = SCENE_X.length;
type Loop = 'neighbour' | 'inplace';
const params = (op: Loop): Params => ({ ...defaults(op), n: N });
const DATA = { x: SCENE_X };
export const SCENE_WANT: Record<Loop, number[]> = {
  neighbour: runSequential('neighbour', DATA, params('neighbour')),
  inplace: runSequential('inplace', DATA, params('inplace')),
};

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

const MONO = { fontVariationSettings: "'MONO' 1, 'CASL' 0" }; // i18n-ignore: CSS value

// Two layouts. Wide: row labels to the left of the cells. Narrow (a phone): labels
// above the rows and narrower cells, so the drawing is scaled down far less and
// its text stays legible. Both draw the same numbers from the same model.
interface Geo {
  narrow: boolean;
  /** viewBox width and height */
  W: number;
  height: number;
  /** cell width and height, and the left edge of the cells */
  C: number;
  H: number;
  X0: number;
  headY: number;
  /** y of the three rows of a loop, and of the three rows of the summary */
  rows: [number, number, number];
  sumRows: [number, number, number];
  footY: number;
  /** longest line of the heading or the footer before it wraps, in characters */
  wrap: number;
}
const WIDE: Geo = { narrow: false, W: 720, height: 300, C: 56, H: 34, X0: 222, headY: 24, rows: [50, 128, 206], sumRows: [56, 140, 206], footY: 288, wrap: 999 };
const NARROW: Geo = { narrow: true, W: 364, height: 346, C: 42, H: 32, X0: 14, headY: 22, rows: [66, 156, 256], sumRows: [66, 146, 226], footY: 316, wrap: 46 };

/** Lines of at most `max` characters, each but the last keeping its trailing space, so that they join back to `text`. */
export function wrapText(text: string, max: number): string[] {
  const out: string[] = [];
  let line = '';
  for (const word of text.split(' ')) {
    const next = line ? line + ' ' + word : word;
    if (line && next.length > max) {
      out.push(line + ' ');
      line = word;
    } else line = next;
  }
  if (line) out.push(line);
  return out;
}

function Lines({ x, y, text, max }: { x: number; y: number; text: string; max: number }) {
  return (
    <text x={x} y={y} fontSize="13" fontWeight="600">
      {wrapText(text, max).map((l, i) => (
        <tspan key={i} x={x} dy={i ? 16 : 0}>{l}</tspan>
      ))}
    </text>
  );
}

interface CellLook {
  text: string;
  fill?: string;
  stroke?: string;
  ink?: string;
  bold?: boolean;
  width?: number;
}

function Row({ g, y, label, sub, cells }: { g: Geo; y: number; label: string; sub?: string; cells: CellLook[] }) {
  const { C, H, X0 } = g;
  return (
    <g>
      {g.narrow ? (
        <>
          <text x={X0 + 1} y={y - 8} fontSize="12.5" fontWeight="600">{label}</text>
          {sub ? (
            <text x={X0 + cells.length * C - 3} y={y - 8} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{sub}</text>
          ) : null}
        </>
      ) : (
        <>
          <text x={X0 - 12} y={y + (sub ? 15 : 21)} fontSize="12.5" fontWeight="600" textAnchor="end">{label}</text>
          {sub ? (
            <text x={X0 - 12} y={y + 29} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{sub}</text>
          ) : null}
        </>
      )}
      {cells.map((c, i) => (
        <g key={i}>
          <rect x={X0 + i * C + 1} y={y} width={C - 4} height={H} rx="2" fill={c.fill ?? 'var(--cell)'}
            stroke={c.stroke ?? 'var(--rule)'} strokeWidth={c.width ?? 1} />
          <text x={X0 + i * C + C / 2 - 1} y={y + H / 2 + 5} fontSize="14" textAnchor="middle" fontWeight={c.bold ? 700 : 400}
            style={{ fill: c.ink ?? 'var(--ink)', fontVariantNumeric: 'tabular-nums' }}>{c.text}</text>
        </g>
      ))}
    </g>
  );
}

function Frame({ g, children, head, foot }: { g: Geo; children: ReactNode; head: string; foot?: string }) {
  return (
    <svg viewBox={`0 0 ${g.W} ${g.height}`} role="img" aria-label={tr('Animation of two loops run by threads in two orders', 'Hoạt hình hai vòng lặp được các luồng chạy theo hai thứ tự')}
      data-layout={g.narrow ? 'narrow' : 'wide'} style={{ width: '100%', maxWidth: g.W, display: 'block' }}>
      <Lines x={16} y={g.headY} text={head} max={g.wrap} />
      {children}
      {foot ? <Lines x={16} y={g.footY} text={foot} max={g.wrap} /> : null}
    </svg>
  );
}

const plain = (v: number): CellLook => ({ text: Number.isNaN(v) ? '' : String(v) });
const muted = (v: number): CellLook => ({ text: String(v), ink: 'var(--ink-3)' });

/** One loop under one thread order, `p` of the way through. */
function Running({ op, order, p }: { op: Loop; order: OrderName; p: number }) {
  const g = useNarrow() ? NARROW : WIDE;
  const { C, H, X0 } = g;
  const ord = threadOrder(order, N);
  const done = Math.min(N, Math.floor(ramp(p, 0.1, 0.9) * (N + 0.999)));
  const state = runThreads(op, DATA, ord.slice(0, done), params(op));
  const last = done > 0 ? ord[done - 1] : null; // the thread that has just run
  const r = last == null ? null : reads(op, last, params(op));
  const want = SCENE_WANT[op];
  const written = new Set(ord.slice(0, done));
  const orderText = ord.map((k, i) => (i === done - 1 ? `[${k}]` : String(k))).join(' ');
  const wrong = [...written].filter((k) => state[k] !== want[k]).length;

  // What the thread that just ran read: its own inputs, and for the in-place loop its left neighbour as it was then.
  const readIdx = new Set<number>();
  if (r) {
    r.inputs.forEach((a) => a.indices.forEach((i) => readIdx.add(i)));
    r.outputs.forEach((i) => readIdx.add(i));
  }
  const result = (k: number): CellLook => {
    if (!written.has(k)) return op === 'inplace' ? plain(SCENE_X[k]) : plain(NaN);
    const ok = state[k] === want[k];
    return {
      text: String(state[k]),
      fill: ok ? 'var(--ok-wash)' : 'var(--fail-wash)',
      stroke: k === last ? 'var(--ink)' : ok ? 'var(--ok)' : 'var(--fail)',
      ink: ok ? 'var(--ink)' : 'var(--fail)',
      bold: k === last,
      width: k === last ? 2.2 : 1,
    };
  };

  const head = tr(`${op === 'neighbour' ? 'Loop 1' : 'Loop 2'}: ${OP_BODY[op]}`, `${op === 'neighbour' ? 'Vòng lặp 1' : 'Vòng lặp 2'}: ${OP_BODY[op]}`);
  const foot =
    done === 0
      ? tr(`threads will run in this order: ${ord.join(' ')}`, `các luồng sẽ chạy theo thứ tự này: ${ord.join(' ')}`)
      : done < N
        ? tr(`threads run so far: ${orderText}`, `các luồng đã chạy đến giờ: ${orderText}`)
        : wrong === 0
          ? tr('all 8 threads done: the same answer as the loop', 'cả 8 luồng đã xong: cùng kết quả với vòng lặp')
          : tr(`all 8 threads done: ${wrong} of 8 results differ from the loop`, `cả 8 luồng đã xong: ${wrong} trên 8 kết quả khác với vòng lặp`);
  const [y1, y2, y3] = g.rows;

  if (op === 'neighbour')
    return (
      <Frame g={g} head={head} foot={foot}>
        <Row g={g} y={y1} label="x" sub={tr('never written', 'không bao giờ bị ghi')}
          cells={SCENE_X.map((v, i) =>
            readIdx.has(i) ? { text: String(v), fill: 'var(--global)', stroke: 'var(--global)', ink: 'var(--page)', bold: true } : plain(v),
          )} />
        <Row g={g} y={y2} label="y" sub={tr('one element per thread', 'mỗi luồng một phần tử')} cells={SCENE_X.map((_, k) => result(k))} />
        <Row g={g} y={y3} label={tr('what the loop gives', 'kết quả của vòng lặp')} cells={want.map(muted)} />
      </Frame>
    );

  // In place: the neighbour read is either a value some thread already wrote, or the original.
  const left = last != null && last > 0 ? last - 1 : null;
  const leftWasWritten = left != null && ord.slice(0, done - 1).includes(left);
  // The note under the arrow is centred on it, kept inside the drawing on a narrow screen.
  const noteX = left == null ? 0 : g.narrow ? Math.max(126, Math.min(g.W - 126, X0 + left * C + C)) : X0 + left * C + C;
  return (
    <Frame g={g} head={head} foot={foot}>
      <Row g={g} y={y1} label={tr('x at the start', 'x lúc đầu')} cells={SCENE_X.map(muted)} />
      <Row g={g} y={y2} label={tr('x in memory now', 'x trong bộ nhớ lúc này')} sub={tr('read and written', 'vừa đọc vừa ghi')}
        cells={SCENE_X.map((_, k) => {
          const c = result(k);
          return k === left ? { ...c, stroke: 'var(--ink)', width: 2.2 } : c;
        })} />
      {left != null && last != null ? (
        <g>
          <path d={`M ${X0 + left * C + C / 2} ${y2 + H + 3} q ${C / 2} 16 ${C - 4} 0`} fill="none" stroke="var(--ink)" strokeWidth="1.6" />
          <text x={noteX} y={y2 + H + 30} fontSize="11.5" textAnchor="middle">
            {leftWasWritten ? tr(`thread ${last} read what thread ${left} wrote`, `luồng ${last} đọc thứ luồng ${left} đã ghi`) : tr(`thread ${last} read x[${left}] before thread ${left} ran`, `luồng ${last} đọc x[${left}] trước khi luồng ${left} chạy`)}
          </text>
        </g>
      ) : null}
      <Row g={g} y={y3} label={tr('what the loop gives', 'kết quả của vòng lặp')} cells={want.map(muted)} />
    </Frame>
  );
}

function Intro({ p }: { p: number }) {
  const g = useNarrow() ? NARROW : WIDE;
  // Wide: name, then the loop on the same line. Narrow: the loop on its own line below the name.
  const at = g.narrow
    ? { name: [124, 196], code: { x: 16, dy: 18 }, note: 36 }
    : { name: [140, 204], code: { x: 76, dy: 0 }, note: 20 };
  const loop = (n: 0 | 1, op: Loop, note: string, opacity: number) => (
    <g opacity={opacity}>
      <text x={16} y={at.name[n]} fontSize="13" fontWeight="600"><Translated vi={<>Vòng lặp {n + 1}</>}>Loop {n + 1}</Translated></text>
      <text x={at.code.x} y={at.name[n] + at.code.dy} fontSize="13" style={MONO}><Translated vi={<>với mỗi k: {OP_BODY[op]}</>}>for each k: {OP_BODY[op]}</Translated></text>
      <text x={at.code.x} y={at.name[n] + at.note} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>{note} (x[-1] được tính là 0)</>}>{note} ({OP_NOTE[op]})</Translated></text>
    </g>
  );
  return (
    <Frame g={g} head={tr('8 numbers and two loops', '8 số và hai vòng lặp')} foot={p > 0.7 ? tr('one thread per value of k: does the order of the threads matter?', 'mỗi giá trị của k một luồng: thứ tự của các luồng có quan trọng không?') : undefined}>
      <Row g={g} y={g.rows[0]} label="x" cells={SCENE_X.map(plain)} />
      {loop(0, 'neighbour', tr('writes a second array, y', 'ghi vào một mảng thứ hai, y'), ramp(p, 0.12, 0.3))}
      {loop(1, 'inplace', tr('writes back into x', 'ghi ngược vào x'), ramp(p, 0.32, 0.5))}
    </Frame>
  );
}

function Summary({ p }: { p: number }) {
  const g = useNarrow() ? NARROW : WIDE;
  const rev = (op: Loop) => runThreads(op, DATA, threadOrder('reversed', N), params(op));
  const cells = (op: Loop, got: number[]): CellLook[] =>
    got.map((v, k) =>
      v === SCENE_WANT[op][k]
        ? { text: String(v), fill: 'var(--ok-wash)', stroke: 'var(--ok)' }
        : { text: String(v), fill: 'var(--fail-wash)', stroke: 'var(--fail)', ink: 'var(--fail)' },
    );
  const bad = rev('inplace').filter((v, k) => v !== SCENE_WANT.inplace[k]).length;
  const [y1, y2, y3] = g.sumRows;
  return (
    <Frame g={g} head={tr('Run backwards, compared with what each loop gives', 'Chạy ngược, so với kết quả của từng vòng lặp')}
      foot={p > 0.5 ? tr('the test: does any thread read what another thread writes?', 'phép thử: có luồng nào đọc thứ mà một luồng khác ghi không?') : undefined}>
      <Row g={g} y={y1} label={tr('loop 1, backwards', 'vòng lặp 1, chạy ngược')} sub={tr('reads overlap, nobody writes x', 'các lần đọc chồng nhau, không ai ghi x')} cells={cells('neighbour', rev('neighbour'))} />
      <g opacity={ramp(p, 0.15, 0.3)}>
        <Row g={g} y={y2} label={tr('loop 2, backwards', 'vòng lặp 2, chạy ngược')} sub={tr(`${bad} of 8 differ`, `${bad} trên 8 khác`)} cells={cells('inplace', rev('inplace'))} />
        <Row g={g} y={y3} label={tr('loop 2, in loop order', 'vòng lặp 2, theo thứ tự vòng lặp')} cells={SCENE_WANT.inplace.map(muted)} />
      </g>
    </Frame>
  );
}

const beats: Beat[] = [
  {
    clip: 'order-1',
    get label() {
      return tr('Two loops', 'Hai vòng lặp');
    },
    draw: (p) => <Intro p={p} />,
  },
  {
    clip: 'order-2',
    get label() {
      return tr('Loop 1 in order', 'Vòng lặp 1 theo thứ tự');
    },
    draw: (p) => <Running op="neighbour" order="forward" p={p} />, // i18n-ignore: sim keys
  },
  {
    clip: 'order-3',
    get label() {
      return tr('Loop 1 backwards', 'Vòng lặp 1 chạy ngược');
    },
    draw: (p) => <Running op="neighbour" order="reversed" p={p} />, // i18n-ignore: sim keys
  },
  {
    clip: 'order-4',
    get label() {
      return tr('Loop 2 in order', 'Vòng lặp 2 theo thứ tự');
    },
    draw: (p) => <Running op="inplace" order="forward" p={p} />, // i18n-ignore: sim keys
  },
  {
    clip: 'order-5',
    get label() {
      return tr('Loop 2 backwards', 'Vòng lặp 2 chạy ngược');
    },
    draw: (p) => <Running op="inplace" order="reversed" p={p} />, // i18n-ignore: sim keys
  },
  {
    clip: 'order-6',
    get label() {
      return tr('The test', 'Phép thử');
    },
    draw: (p) => <Summary p={p} />,
  },
];

export function OrderScene() {
  return <Scene chapter="ch02" title={tr('Same loop, another order', 'Cùng vòng lặp, thứ tự khác')} beats={beats} />;
}
