import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, NoteGroup, Simplification, Source } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  BINS,
  BIN_LABEL,
  BIN_LONG,
  DECODE_ITEMS,
  OPS,
  OP_BODY,
  OP_LABEL,
  OP_NOTE,
  binCounts,
  classify,
  classifyDeps,
  combineReads,
  counts,
  defaults,
  reads,
  shape,
  stageItems,
  timesRead,
  underOrders,
  type Bin,
  type DecodeItem,
  type Op,
  type Params,
} from '../../sim/depend';
import { OrderScene } from './OrderScene';

const num = (v: number) => v.toLocaleString('en-US');
const plural = (n: number, one: string, many = one + 's') => `${num(n)} ${n === 1 ? one : many}`;

// ---------------------------------------------------------------- interactive A: what does one output read?
type Pick = { row: 'out' | 'combine'; i: number };

/** "x[4] and x[5]", "in[6] to in[10]", "I[9], I[10] and I[11]". */
function listCells(array: string, idx: number[]): string {
  const name = (i: number) => `${array}[${i}]`;
  if (idx.length === 0) return tr('nothing', 'không gì cả');
  if (idx.length === 1) return name(idx[0]);
  const contiguous = idx.every((v, i) => i === 0 || v === idx[i - 1] + 1);
  if (contiguous && idx.length > 3) return tr(`${name(idx[0])} to ${name(idx[idx.length - 1])}`, `${name(idx[0])} đến ${name(idx[idx.length - 1])}`);
  return idx.slice(0, -1).map(name).join(', ') + tr(' and ', ' và ') + name(idx[idx.length - 1]);
}

const outName = (op: Op, k: number) => (op === 'mean' ? tr('The mean', 'Giá trị trung bình') : `${shape(op).output.name}[${k}]`);
const combineName = (op: Op, c: number) =>
  op === 'mean' ? tr(`partial sum ${c}`, `tổng riêng phần ${c}`) : tr(`the sum of row ${c}`, `tổng của hàng ${c}`);

/** One sentence, built from the model, saying what the picked cell reads. */
function describe(op: Op, pick: Pick, p: Params): { text: string; bad: boolean } {
  if (pick.row === 'combine') {
    const r = combineReads(op, pick.i, p);
    const what = op === 'mean' ? tr(`Partial sum ${pick.i}`, `Tổng riêng phần ${pick.i}`) : tr(`The sum of row ${pick.i}`, `Tổng của hàng ${pick.i}`);
    return {
      text: tr(
        `${what} is built from ${listCells(r.array, r.indices)}: ${plural(r.indices.length, 'input')}, and no other ${op === 'mean' ? 'partial sum' : 'row sum'}. Each of these can be computed without the others.`,
        `${what} được tạo từ ${listCells(r.array, r.indices)}: ${num(r.indices.length)} đầu vào, và không từ ${op === 'mean' ? 'tổng riêng phần' : 'tổng hàng'} nào khác. Mỗi giá trị này có thể được tính mà không cần các giá trị kia.`,
      ),
      bad: false,
    };
  }
  const r = reads(op, pick.i, p);
  const c = counts(op, pick.i, p);
  const own = r.inputs.map((a) => listCells(a.array, a.indices)).join(tr(', and ', ', và '));
  const name = outName(op, pick.i);
  if (r.chain.length) {
    const first = r.outputs[0];
    const o = shape(op, p).output.name;
    return {
      text:
        tr(
          `${name} reads ${own} as it was before the loop, and it needs the new value of ${o}[${first}], which is another output. `,
          `${name} đọc ${own} như trước vòng lặp, và nó cần giá trị mới của ${o}[${first}], vốn là một đầu ra khác. `,
        ) +
        (r.chain.length > 1
          ? tr(
              `That one needs ${o}[${first - 1}], and so on back to ${o}[0]: ${plural(r.chain.length, 'output')} must exist first, in order.`,
              `Đầu ra đó lại cần ${o}[${first - 1}], và cứ thế lùi về ${o}[0]: ${num(r.chain.length)} đầu ra phải có trước, theo đúng thứ tự.`,
            )
          : tr(`1 output must exist first.`, `1 đầu ra phải có trước.`)),
      bad: true,
    };
  }
  if (c.combines) {
    const parts = r.combines.map((x) => combineName(op, x));
    const which = parts.length === 1 ? parts[0] : tr(`${plural(parts.length, 'partial sum')}`, `${num(parts.length)} tổng riêng phần`);
    return {
      text:
        (c.inputs
          ? tr(`${name} reads ${own} itself and needs `, `${name} tự đọc ${own} và cần `)
          : tr(`${name} reads no input itself. It needs `, `${name} không tự đọc đầu vào nào. Nó cần `)) +
        tr(
          `${which} first, built from ${plural(c.viaCombines, 'input')} in all. It needs no other output.`,
          `${which} trước, được tạo từ tổng cộng ${num(c.viaCombines)} đầu vào. Nó không cần đầu ra nào khác.`,
        ),
      bad: false,
    };
  }
  return {
    text: tr(
      `${name} reads ${own}: ${plural(c.inputs, 'input')}. It needs no other output, so its thread can run at any time.`,
      `${name} đọc ${own}: ${num(c.inputs)} đầu vào. Nó không cần đầu ra nào khác, nên luồng của nó có thể chạy vào bất kỳ lúc nào.`,
    ),
    bad: false,
  };
}

/**
 * At phone widths the drawing is wider than its panel and scrolls inside it.
 * Keep x (in viewBox units) in view, so the picked cell is never off to the side.
 */
function useKeepInView(x: number, viewBoxWidth: number) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    const svg = el?.querySelector('svg');
    if (!el || !svg || el.scrollWidth <= el.clientWidth + 1) return;
    const px = (x / viewBoxWidth) * svg.getBoundingClientRect().width;
    el.scrollLeft = Math.max(0, px - el.clientWidth / 2);
  }, [x, viewBoxWidth]);
  return ref;
}

const WRAP = { whiteSpace: 'normal' } as const;
const VBW = 760;
const LEFT = 104;
const CH = 30;

function ReadExplorer() {
  const [op, setOp] = useState<Op>('grayscale');
  const [radius, setRadius] = useState(1);
  const [chunk, setChunk] = useState(16);
  const [picked, setPicked] = useState<Pick>({ row: 'out', i: 3 });
  const p: Params = useMemo(() => ({ ...defaults(op), radius, chunk }), [op, radius, chunk]);
  const s = shape(op, p);
  const pick: Pick =
    picked.row === 'combine' && s.combine
      ? { row: 'combine', i: Math.min(picked.i, s.combine.length - 1) }
      : { row: 'out', i: Math.min(picked.i, s.output.length - 1) };

  const r = pick.row === 'out' ? reads(op, pick.i, p) : null;
  const c = pick.row === 'out' ? counts(op, pick.i, p) : null;
  const direct = new Map<string, Set<number>>();
  const via = new Set<number>(); // inputs of the first input array reached through a combined value
  const needCombine = new Set<number>();
  const chain = new Set<number>(r?.chain ?? []);
  if (r) {
    r.inputs.forEach((a) => direct.set(a.array, new Set(a.indices)));
    r.combines.forEach((x) => {
      needCombine.add(x);
      combineReads(op, x, p).indices.forEach((i) => via.add(i));
    });
  } else {
    const cr = combineReads(op, pick.i, p);
    direct.set(cr.array, new Set(cr.indices));
  }
  const said = describe(op, pick, p);
  const orders = useMemo(() => underOrders(op, undefined, p), [op, p]);
  const allMatch = orders.every((o) => o.matches);
  const bin = classify(op);

  // Geometry: one cell width for the longest input array; outputs and combined values span the inputs they sit under.
  const maxLen = Math.max(...s.inputs.map((a) => a.length));
  const cw = Math.min(40, (VBW - LEFT - 8) / maxLen);
  const span = (len: number) => (maxLen % len === 0 ? maxLen / len : 1);
  const cellX = (len: number, i: number, aligned: boolean) => LEFT + i * cw * (aligned ? span(len) : 1);
  const cellW = (len: number, aligned: boolean) => cw * (aligned ? span(len) : 1) - 3;
  const inY = (a: number) => 22 + a * 62;
  const combineY = inY(s.inputs.length) + 6;
  const outY = (s.combine ? combineY + 62 : inY(s.inputs.length)) + 6;
  const height = outY + CH + 34;
  const mid = (len: number, i: number, aligned: boolean) => cellX(len, i, aligned) + cellW(len, aligned) / 2;

  const cell = (
    key: string,
    x: number,
    y: number,
    w: number,
    label: string,
    state: 'idle' | 'read' | 'via' | 'needed' | 'chain' | 'picked',
    onPick?: () => void,
    aria?: string,
    sub?: string,
  ) => {
    const look = {
      idle: { fill: 'var(--cell)', stroke: 'var(--rule)', ink: 'var(--ink-2)', sw: 1 },
      read: { fill: 'var(--global)', stroke: 'var(--global)', ink: 'var(--page)', sw: 1 },
      via: { fill: 'var(--global-wash)', stroke: 'var(--global)', ink: 'var(--ink)', sw: 1 },
      needed: { fill: 'var(--page)', stroke: 'var(--ink)', ink: 'var(--ink)', sw: 2.2 },
      chain: { fill: 'var(--fail-wash)', stroke: 'var(--fail)', ink: 'var(--fail)', sw: 1.6 },
      picked: { fill: 'var(--ink)', stroke: 'var(--ink)', ink: 'var(--page)', sw: 1 },
    }[state];
    return (
      <g
        key={key}
        data-cell={key}
        data-state={state}
        role={onPick ? 'button' : undefined}
        tabIndex={onPick ? 0 : undefined}
        aria-label={aria}
        aria-pressed={onPick ? state === 'picked' : undefined}
        style={onPick ? { cursor: 'pointer' } : undefined}
        onClick={onPick}
        onKeyDown={
          onPick
            ? (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onPick();
                }
              }
            : undefined
        }
      >
        <rect x={x} y={y} width={w} height={CH} rx="2" fill={look.fill} stroke={look.stroke} strokeWidth={look.sw} />
        <text x={x + w / 2} y={y + CH / 2 + 4} fontSize="11.5" textAnchor="middle" fontWeight={state === 'idle' ? 400 : 650}
          style={{ fill: look.ink, fontVariantNumeric: 'tabular-nums', pointerEvents: 'none' }}>
          {label}
        </text>
        {sub != null ? (
          <text x={x + w / 2} y={y + CH + 11} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-3)', pointerEvents: 'none' }}>
            {sub}
          </text>
        ) : null}
      </g>
    );
  };

  const rowLabel = (y: number, main: string, sub?: string) => (
    <g>
      <text x={LEFT - 10} y={y + (sub ? 13 : 19)} fontSize="12" fontWeight="600" textAnchor="end">{main}</text>
      {sub ? (
        <text x={LEFT - 10} y={y + 26} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{sub}</text>
      ) : null}
    </g>
  );

  const lines: ReactNode[] = [];
  const outLen = s.output.length;
  if (pick.row === 'out' && r) {
    const fromX = mid(outLen, pick.i, true);
    // Without a combine row the inputs carry a count under each cell; stop the lines below it.
    const below = s.combine ? 0 : 15;
    r.inputs.forEach((a) => {
      const ai = s.inputs.findIndex((x) => x.name === a.array);
      a.indices.forEach((i) =>
        lines.push(
          <line key={`l-${a.array}-${i}`} x1={fromX} y1={outY} x2={mid(s.inputs[ai].length, i, false)} y2={inY(ai) + CH + below}
            stroke="var(--global)" strokeWidth="1.3" opacity="0.75" />,
        ),
      );
    });
    if (s.combine)
      r.combines.forEach((x) =>
        lines.push(
          <line key={`lc-${x}`} x1={fromX} y1={outY} x2={mid(s.combine!.length, x, true)} y2={combineY + CH}
            stroke="var(--ink)" strokeWidth="1.5" />,
        ),
      );
    // The chain: one hop from each needed output to the one before it, back to the start.
    [pick.i, ...r.chain].slice(0, -1).forEach((k) =>
      lines.push(
        <path key={`ch-${k}`} d={`M ${mid(outLen, k, true) - 4} ${outY + CH + 2} q ${-cw / 2 + 4} 13 ${-cw + 8} 0`}
          fill="none" stroke="var(--fail)" strokeWidth="1.6" markerEnd="url(#dep-arrow)" />,
      ),
    );
  }
  // A bracket from each lit combined value up to the inputs it is built from.
  if (s.combine) {
    const lit = pick.row === 'combine' ? [pick.i] : [...needCombine];
    lit.forEach((x) => {
      const idx = combineReads(op, x, p).indices;
      const x0 = cellX(maxLen, idx[0], false);
      const x1 = cellX(maxLen, idx[idx.length - 1], false) + cellW(maxLen, false);
      const y = inY(0) + CH + 16;
      lines.push(
        <path key={`br-${x}`} d={`M ${x0} ${y - 4} V ${y} H ${x1} V ${y - 4} M ${mid(s.combine!.length, x, true)} ${y} V ${combineY}`}
          fill="none" stroke="var(--global)" strokeWidth="1.3" />,
      );
    });
  }

  const choose = (next: Op) => {
    setOp(next);
    setPicked({ row: 'out', i: Math.min(5, shape(next).output.length - 1) });
  };
  const scroller = useKeepInView(
    pick.row === 'combine' && s.combine ? mid(s.combine.length, pick.i, true) : mid(outLen, pick.i, true),
    VBW,
  );

  return (
    <Bench
      title={tr('What does one output read?', 'Một đầu ra đọc những gì?')}
      task={tr(
        'Pick an operation, then click an output cell. Before you click, say how many inputs will light up and whether anything will turn red.',
        'Chọn một phép toán, rồi bấm vào một ô đầu ra. Trước khi bấm, hãy nói bao nhiêu đầu vào sẽ sáng lên và có ô nào chuyển sang đỏ không.',
      )}
      foot={tr(
        "Lit cells are input elements the output's own thread reads. A red cell is another output that must exist first. The small numbers under the inputs count how many outputs read each one.",
        'Ô sáng là các phần tử đầu vào mà chính luồng của đầu ra đó đọc. Ô đỏ là một đầu ra khác phải có trước. Các số nhỏ dưới đầu vào đếm xem bao nhiêu đầu ra đọc mỗi ô.',
      )}
    >
      <div data-explorer data-op={op}>
        <div className="row" role="group" aria-label={tr('Operation', 'Phép toán')} style={{ gap: '0.4rem', marginBottom: '0.8rem' }}>
          {OPS.map((o) => (
            <button key={o} className="quiet" data-op-choice={o} aria-pressed={o === op} onClick={() => choose(o)}>
              {OP_LABEL[o]}
            </button>
          ))}
        </div>
        <div className="row" style={{ marginBottom: '0.6rem', minHeight: '2.4rem' }}>
          <span>
            {s.output.length > 1 ? <span className="muted">{tr('for each k: ', 'với mỗi k: ')}</span> : null}
            <code data-out="body">{OP_BODY[op]}</code>
            {OP_NOTE[op] ? <span className="muted"> ({OP_NOTE[op]})</span> : null}
          </span>
          {op === 'blur' ? (
            <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
              <Translated
                vi={
                  <span>
                    Bán kính r = <b data-out="radius">{radius}</b>
                  </span>
                }
              >
                <span>
                  Radius r = <b data-out="radius">{radius}</b>
                </span>
              </Translated>
              <input type="range" min="0" max="4" step="1" value={radius} aria-label={tr('Blur radius', 'Bán kính làm mờ')}
                onChange={(e) => setRadius(Number(e.target.value))} />
            </label>
          ) : null}
          {op === 'mean' ? (
            <span className="row" style={{ gap: '0.5rem' }}>
              <span className="muted">
                <Translated vi="Chia tổng thành các tổng riêng phần gồm">Split the sum into partial sums of</Translated>
              </span>
              <span className="seg" role="group" aria-label={tr('Elements per partial sum', 'Số phần tử mỗi tổng riêng phần')}>
                {[16, 8, 4, 2].map((v) => (
                  <button key={v} data-chunk={v} aria-pressed={chunk === v} onClick={() => setChunk(v)}>
                    {v}
                  </button>
                ))}
              </span>
            </span>
          ) : null}
        </div>

        <div className="panel" style={{ overflowX: 'auto' }} ref={scroller}>
          <svg viewBox={`0 0 ${VBW} ${height}`} role="group" aria-label={tr(`Cells of ${OP_LABEL[op]}: inputs above, outputs below`, `Các ô của ${OP_LABEL[op]}: đầu vào ở trên, đầu ra ở dưới`)}
            style={{ width: '100%', minWidth: 620, maxWidth: VBW, display: 'block' }}>
            <defs>
              <marker id="dep-arrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute */}
                <path d="M 0 0 L 8 4 L 0 8 z" fill="var(--fail)" />
              </marker>
            </defs>
            {lines}
            {s.inputs.map((a, ai) => (
              <g key={a.name}>
                {rowLabel(
                  inY(ai),
                  s.inPlace ? tr(`${a.name} before`, `${a.name} trước`) : a.name,
                  op === 'grayscale' ? tr('input, r g b', 'đầu vào, r g b') : s.inPlace ? tr('the values read', 'các giá trị được đọc') : tr('input', 'đầu vào'),
                )}
                {Array.from({ length: a.length }, (_, i) => {
                  const isRead = direct.get(a.name)?.has(i);
                  const isVia = ai === 0 && via.has(i);
                  return cell(
                    `in:${a.name}:${i}`, // i18n-ignore: cell key
                    cellX(a.length, i, false),
                    inY(ai),
                    cellW(a.length, false),
                    op === 'grayscale' ? 'rgb'[i % 3] : String(i),
                    isRead ? 'read' : isVia ? 'via' : 'idle',
                    undefined,
                    undefined,
                    s.combine ? undefined : String(timesRead(op, a.name, i, p)),
                  );
                })}
              </g>
            ))}
            {s.combine ? (
              <g>
                {rowLabel(combineY, s.combine.name, tr('combined first', 'gộp trước'))}
                {Array.from({ length: s.combine.length }, (_, x) =>
                  cell(
                    `combine:${x}`, // i18n-ignore: cell key
                    cellX(s.combine!.length, x, true),
                    combineY,
                    cellW(s.combine!.length, true),
                    op === 'mean' ? tr(`sum ${x}`, `tổng ${x}`) : tr(`sum of row ${x}`, `tổng hàng ${x}`),
                    pick.row === 'combine' && pick.i === x ? 'picked' : needCombine.has(x) ? 'needed' : 'idle',
                    () => setPicked({ row: 'combine', i: x }),
                    tr(`${combineName(op, x)}: show what it reads`, `${combineName(op, x)}: xem nó đọc gì`),
                  ),
                )}
              </g>
            ) : null}
            <g>
              {rowLabel(
                outY,
                s.inPlace ? tr(`${s.output.name} after`, `${s.output.name} sau`) : s.output.name,
                s.inPlace ? tr('same memory', 'cùng vùng nhớ') : tr('output', 'đầu ra'),
              )}
              {Array.from({ length: outLen }, (_, k) =>
                cell(
                  `out:${k}`, // i18n-ignore: cell key
                  cellX(outLen, k, true),
                  outY,
                  cellW(outLen, true),
                  op === 'mean' ? tr('mean', 'trung bình') : String(k),
                  pick.row === 'out' && pick.i === k ? 'picked' : chain.has(k) ? 'chain' : 'idle',
                  () => setPicked({ row: 'out', i: k }),
                  tr(`Output ${k}: show what it reads`, `Đầu ra ${k}: xem nó đọc gì`),
                ),
              )}
            </g>
          </svg>
        </div>

        <div className="cols" style={{ marginTop: '0.8rem' }}>
          <div className={'narrate' + (said.bad ? ' bad' : '')} data-out="said" role="status">
            {said.text}
          </div>
          <div className="panel">
            <table className="threads">
              <tbody>
                <tr>
                  <td style={WRAP}>
                    <Translated vi="Đầu vào nó tự đọc">Inputs it reads itself</Translated>
                  </td>
                  <td>
                    <output data-out="inputs">{c ? c.inputs : combineReads(op, pick.i, p).indices.length}</output>
                  </td>
                </tr>
                {s.combine ? (
                  <tr>
                    <td style={WRAP}>
                      <Translated vi="Giá trị gộp cần có trước">Combined values needed first</Translated>
                    </td>
                    <td>
                      <output data-out="combines">{c ? c.combines : 0}</output>
                    </td>
                  </tr>
                ) : null}
                <tr>
                  <td style={WRAP}>
                    <Translated vi="Đầu ra cần có trước">Outputs needed first</Translated>
                  </td>
                  <td>
                    <output data-out="outputs-first" className={c && c.outputsFirst ? 't-fail' : undefined}>
                      {c ? c.outputsFirst : 0}
                    </output>
                  </td>
                </tr>
                <tr>
                  <td style={WRAP}>
                    <Translated vi="Dạng của phép toán">Shape of the operation</Translated>
                  </td>
                  <td style={WRAP}>
                    <output data-out="bin" data-bin={bin}>{BIN_LABEL[bin]}</output>
                  </td>
                </tr>
                <tr>
                  <td style={WRAP}>
                    <Translated vi={<>Một luồng cho mỗi đầu ra, {orders.length} thứ tự luồng</>}>
                      One thread per output, {orders.length} thread orders
                    </Translated>
                  </td>
                  <td>
                    <output data-out="orders" data-match={allMatch ? 'all' : 'some'}>
                      {allMatch ? (
                        <span className="chip ok">
                          <Translated vi="lần nào cũng cùng kết quả">same result every time</Translated>
                        </span>
                      ) : (
                        <span className="chip fail">
                          <Translated vi={<>{orders.filter((o) => !o.matches).length} trên {orders.length} khác với vòng lặp</>}>
                            {orders.filter((o) => !o.matches).length} of {orders.length} differ from the loop
                          </Translated>
                        </span>
                      )}
                    </output>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive B: sort a decode step
const SHORT: Record<Bin, string> = {
  get elementwise() {
    return tr('Per element', 'Theo từng phần tử');
  },
  get manyreads() {
    return tr('Per output, many reads', 'Theo từng đầu ra, đọc nhiều');
  },
  get combine() {
    return tr('Combine first', 'Gộp trước');
  },
  get chain() {
    return tr('Chain', 'Chuỗi');
  },
};

const course = (id: string, text: string) => <a href={href(id)}>{text}</a>;

/** What the reader sees once an item is in the right bin: its read set from the model, and where its kernel is written. */
const PLACED: Record<string, (it: DecodeItem) => ReactNode> = {
  residual: (it) => (
    <Translated
      vi={
        <>
          Một đầu ra đọc {it.deps.reads(4096)} đầu vào, <code>x[i]</code> và <code>r[i]</code>, bất kể kích thước của tensor. Đây là phép cộng vectơ của chương này dưới một tên khác, và là một trong ba phép toán của mốc dự án M0 (<a href={href('ch02', 'labs')}>lab 2.3</a>).
        </>
      }
    >
      One output reads {it.deps.reads(4096)} inputs, <code>x[i]</code> and <code>r[i]</code>, whatever the size of the
      tensor. This is the vector addition of this chapter under another name, and one of the three operations of
      milestone M0 (<a href={href('ch02', 'labs')}>lab 2.3</a>).
    </Translated>
  ),
  bias: (it) => (
    <Translated
      vi={
        <>
          Một đầu ra đọc {it.deps.reads(4096)} đầu vào: phần tử của chính nó và một giá trị bias. Mọi hàng đọc cùng các giá trị bias, và đọc chung thì vô hại. Mốc dự án M0 (<a href={href('ch02', 'labs')}>lab 2.3</a>).
        </>
      }
    >
      One output reads {it.deps.reads(4096)} inputs: its own element and one bias value. Every row reads the same bias
      values, and shared reads are harmless. Milestone M0 (<a href={href('ch02', 'labs')}>lab 2.3</a>).
    </Translated>
  ),
  scale: (it) => (
    <Translated
      vi={
        <>
          Một đầu ra đọc {it.deps.reads(4096)} đầu vào. Mốc dự án M0. Kernel trong bài tập 9 của sách,{' '}<code>b[i] = 2.7f*a[i] - 4.3f</code>, chính là phép toán này cộng thêm một độ lệch.
        </>
      }
    >
      One output reads {it.deps.reads(4096)} input. Milestone M0. The kernel in the book's exercise 9,{' '}
      <code>b[i] = 2.7f*a[i] - 4.3f</code>, is this operation with an offset added.
    </Translated>
  ),
  gelu: (it) => (
    <Translated
      vi={
        <>
          Một đầu ra đọc {it.deps.reads(4096)} đầu vào. Phép tính cho một phần tử dài hơn một phép cộng, và kernel có cùng dạng với kernel của chương này: một chỉ số, một phép chặn, một phần tử.
        </>
      }
    >
      One output reads {it.deps.reads(4096)} input. The arithmetic for one element is longer than an addition, and the
      kernel has the same shape as this chapter's: an index, a guard, one element.
    </Translated>
  ),
  linear: (it) => (
    <Translated
      vi={
        <>
          Các đầu ra không cần nhau, nhưng một đầu ra đọc {num(it.deps.reads(4096))} giá trị khi d = 4096: cả một hàng của <code>W</code> và toàn bộ <code>x</code>. Giờ mỗi luồng có một vòng lặp bên trong.{' '}{course('ch03', 'Chương 3')} viết kernel này, {course('ch05', 'chương 5')} cắt giảm lưu lượng bộ nhớ của nó và{' '}{course('ch15', 'chương 15')} tối ưu nó.
        </>
      }
    >
      Outputs do not need each other, but one output reads {num(it.deps.reads(4096))} values when d = 4096: a whole
      row of <code>W</code> and all of <code>x</code>. Each thread now has a loop inside it.{' '}
      {course('ch03', 'Chapter 3')} writes this kernel, {course('ch05', 'chapter 5')} cuts its memory traffic and{' '}
      {course('ch15', 'chapter 15')} optimises it.
    </Translated>
  ),
  layernorm: (it) => (
    <Translated
      vi={
        <>
          Mỗi đầu ra đọc {it.deps.reads(4096)} giá trị của riêng nó (phần tử của nó, một hệ số gain và một bias) và cần{' '}{it.deps.combines} giá trị được gộp từ cả hàng trước đó, là trung bình và phương sai. Tính chúng là một phép rút gọn (reduction), mẫu hình của {course('ch10', 'chương 10')}. Sau đó phần còn lại là theo từng phần tử.
        </>
      }
    >
      Each output reads {it.deps.reads(4096)} values of its own (its element, a gain and a bias) and needs{' '}
      {it.deps.combines} values combined from the whole row first, the mean and the variance. Computing those is a
      reduction, the pattern of {course('ch10', 'chapter 10')}. After that the rest is per element.
    </Translated>
  ),
  softmax: (it) => (
    <Translated
      vi={
        <>
          Mỗi đầu ra đọc {it.deps.reads(50000)} logit của riêng nó và cần {it.deps.combines} giá trị được gộp từ cả bộ từ vựng trước đó: logit lớn nhất m, được trừ đi trước khi lấy hàm mũ để không hàm mũ nào bị tràn số, và sau đó là tổng các hàm mũ. Mỗi giá trị là một phép rút gọn: {course('ch10', 'chương 10')}, mốc dự án M4. Trình khám phá ở trên dùng công thức thường, chỉ cần tổng.
        </>
      }
    >
      Each output reads {it.deps.reads(50000)} logit of its own and needs {it.deps.combines} values combined from the
      whole vocabulary first: the largest logit m, subtracted before the exponential so that no exponential
      overflows, and then the sum of the exponentials. Each is a reduction: {course('ch10', 'chapter 10')},
      milestone M4. The explorer above uses the plain formula, which needs only the sum.
    </Translated>
  ),
  argmax: (it) => (
    <Translated
      vi={
        <>
          Chỉ có một đầu ra, và bản thân nó là {it.deps.combines} giá trị được gộp từ cả bộ từ vựng. Một luồng cho mỗi đầu ra nghĩa là một luồng. Tính song song phải nằm bên trong bước gộp: {course('ch10', 'chương 10')}, mốc dự án M4.
        </>
      }
    >
      There is one output, and it is itself {it.deps.combines} value combined from the whole vocabulary. One thread
      per output would be one thread. The parallelism has to go inside the combine: {course('ch10', 'chapter 10')},
      milestone M4.
    </Translated>
  ),
  cumprob: () => (
    <Translated
      vi={
        <>
          <code>c[i]</code> cần <code>c[i-1]</code>: tổng cộng dồn tại chỗ của trình khám phá ở trên.{' '}{course('ch11', 'Chương 11')} thay chuỗi này bằng một thuật toán khác, và {course('x1', 'mô-đun X1')}{' '}dùng nó cho top-p.
        </>
      }
    >
      <code>c[i]</code> needs <code>c[i-1]</code>: the in-place running sum of the explorer above.{' '}
      {course('ch11', 'Chapter 11')} replaces the chain with a different algorithm, and {course('x1', 'module X1')}{' '}
      uses it for top-p.
    </Translated>
  ),
  kvappend: (it) => (
    <Translated
      vi={
        <>
          Một phép sao chép. Mỗi phần tử của các vectơ key và value mới được ghi vào hàng trống tiếp theo của cache: {it.deps.reads(4096)}{' '}lần đọc mỗi đầu ra, và không đầu ra nào cần đầu ra khác. Hàng nào trống là một con số host truyền vào. Cache dùng để làm gì là nội dung của {course('ch20', 'chương 20')}.
        </>
      }
    >
      A copy. Each element of the new key and value vectors goes to the next free row of the cache: {it.deps.reads(4096)}{' '}
      read per output, and no output needs another. Which row is free is a number the host passes in. What the cache
      is for is {course('ch20', 'chapter 20')}.
    </Translated>
  ),
  genloop: () => (
    <Translated
      vi={
        <>
          Token t được tính từ các token 0 đến t − 1, nên không kernel nào chạy được bước t trước khi bước t − 1 xong. Đây là vòng lặp bao quanh bước, không phải một phép toán bên trong nó: kernel làm mỗi bước nhanh hơn, còn các bước vẫn theo thứ tự. {course('ch20', 'Chương 20')} nói về những gì có thể giữ lại giữa các bước và cách gộp batch qua nhiều chuỗi.
        </>
      }
    >
      Token t is computed from tokens 0 to t − 1, so no kernel can run step t before step t − 1 has finished. This
      is the loop around the step, not an operation inside it: kernels make each step faster, and the steps stay in
      order. {course('ch20', 'Chapter 20')} covers what can be kept between steps and how to batch across sequences.
    </Translated>
  ),
};

/** Why a wrong bin is wrong, said from the item's dependency facts. */
function whyNot(it: DecodeItem, pick: Bin): ReactNode {
  const d = it.deps;
  const actual = classifyDeps(d);
  if (pick === 'chain')
    return (
      <Translated vi="Không đầu ra nào của phép toán này cần một đầu ra khác của cùng phép toán. Chuỗi chính là nhu cầu đó.">
        <>No output of this operation needs another output of the same operation. A chain is exactly that need.</>
      </Translated>
    );
  if (actual === 'chain')
    return (
      <Translated vi="Nhìn vào dòng công việc: kết quả bên trái cần một kết quả sớm hơn của cùng phép toán.">
        <>Look at the line of work: the result on the left needs an earlier result of the same operation.</>
      </Translated>
    );
  if (pick === 'combine')
    return (
      <Translated vi="Không có gì phải được gộp qua cả một hàng hay một mảng trước khi tính được một đầu ra. Mỗi đầu ra lấy đầu vào của nó trực tiếp.">
        <>Nothing has to be combined across a whole row or array before an output can be computed. Each output gets its inputs directly.</>
      </Translated>
    );
  if (actual === 'combine')
    return (
      <Translated
        vi={
          <>
            Một đầu ra không thể tính chỉ từ các đầu vào của riêng nó. Nó cần {num(d.combines)} giá trị được tạo từ cả hàng trước đó.
          </>
        }
      >
        <>
        An output cannot be computed from its own inputs alone. It needs {plural(d.combines, 'value')} built from the
        whole row first.
        </>
      </Translated>
    );
  if (pick === 'manyreads')
    return (
      <Translated
        vi={
          <>
            Một đầu ra đọc {num(d.reads(4096))} đầu vào dù tensor lớn đến đâu. Con số này không tăng theo {it.sizeName}.
          </>
        }
      >
        <>
        One output reads {plural(d.reads(4096), 'input')} however large the tensor is. The number does not grow with
        the {it.sizeName}.
        </>
      </Translated>
    );
  return (
    <Translated
      vi={
        <>
          Đếm xem một đầu ra đọc gì: {num(d.reads(4096))} giá trị khi {it.sizeName} là 4,096, và gấp đôi khi nó tăng gấp đôi. Đó là một vòng lặp bên trong mỗi luồng, không phải một dòng.
        </>
      }
    >
      <>
        Count what one output reads: {num(d.reads(4096))} values when the {it.sizeName} is 4,096, and twice as many when
        it doubles. That is a loop inside each thread, not one line.
      </>
    </Translated>
  );
}

function SortDecodeStep() {
  const [picks, setPicks] = useState<Record<string, Bin | undefined>>({});
  const placed = DECODE_ITEMS.filter((it) => picks[it.id] === classifyDeps(it.deps)).length;
  const tally = binCounts();
  const forward = stageItems('forward').length;
  const sampling = stageItems('sampling').length;
  const inForward = binCounts('forward');
  return (
    <Bench
      title={tr('Sort a decode step', 'Phân loại một bước decode')}
      task={tr(
        `Eleven items from one decode step, in order: ${forward} operations of the forward pass, ${sampling} of sampling, and the loop around the step. Put each in the bin that describes what one of its outputs needs.`,
        `Mười một mục từ một bước decode, theo thứ tự: ${forward} phép toán của lượt tính xuôi (forward pass), ${sampling} của lấy mẫu, và vòng lặp bao quanh bước. Đặt mỗi mục vào ngăn mô tả đúng điều mà một đầu ra của nó cần.`,
      )}
      foot={tr(
        "The list and the sorting are ours, not the book's. The numbers in the feedback come from the same dependency model as the explorer above.",
        'Danh sách và cách phân loại là của khóa học này, không phải của sách. Các con số trong phản hồi đến từ cùng mô hình phụ thuộc với trình khám phá ở trên.',
      )}
    >
      <div data-sort>
        <div className="panel" style={{ marginBottom: '0.8rem' }}>
          <table className="threads" style={{ whiteSpace: 'normal' }}>
            <tbody>
              {BINS.map((b) => (
                <tr key={b}>
                  <td style={{ whiteSpace: 'nowrap', fontWeight: 600 }}>{SHORT[b]}</td>
                  <td style={{ whiteSpace: 'normal' }}>{BIN_LONG[b]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <div className="row" style={{ justifyContent: 'space-between', marginBottom: '0.2rem' }}>
            <b>
              <Translated vi="Mười một mục">Eleven items</Translated>
            </b>
            <span className="muted" data-sort-score>
              <Translated vi={<>đã đặt {placed} trên {DECODE_ITEMS.length}</>}>
                {placed} of {DECODE_ITEMS.length} placed
              </Translated>
            </span>
          </div>
          {DECODE_ITEMS.map((it, n) => {
            const pick = picks[it.id];
            const answer = classifyDeps(it.deps);
            const done = pick === answer;
            return (
              <div key={it.id} data-card={it.id} data-solved={done ? 'true' : 'false'}
                style={{ padding: '0.75rem 0', borderTop: n ? '1px solid var(--rule-soft)' : undefined }}>
                <p style={{ margin: '0 0 0.5rem', maxWidth: '46rem', fontSize: '0.92rem' }}>
                  <span className="muted">{n + 1}.</span> <b>{it.label}</b> <code>{it.body}</code>
                </p>
                <div className="row" style={{ gap: '0.4rem' }} role="group" aria-label={tr(`Bin for ${it.label}`, `Ngăn cho ${it.label}`)}>
                  {BINS.map((b) => (
                    <button key={b} className="quiet" data-choice={b} aria-pressed={pick === b} disabled={done && b !== answer}
                      onClick={() => (done ? undefined : setPicks((s) => ({ ...s, [it.id]: b })))}>
                      {SHORT[b]}
                    </button>
                  ))}
                </div>
                {pick ? (
                  <p data-feedback={it.id} role="status" style={{ margin: '0.5rem 0 0', maxWidth: '46rem' }}>
                    {done ? (
                      <>
                        <span className="chip ok">
                          <Translated vi="đã đặt">placed</Translated>
                        </span>{' '}
                        {PLACED[it.id](it)}
                      </>
                    ) : (
                      <>
                        <span className="chip fail">
                          <Translated vi="không phải ngăn này">not this one</Translated>
                        </span>{' '}
                        {whyNot(it, pick)}
                      </>
                    )}
                  </p>
                ) : null}
              </div>
            );
          })}
        </div>
        <div className="row" style={{ marginTop: '0.8rem', justifyContent: 'flex-end' }}>
          <button className="quiet" onClick={() => setPicks({})}>
            <Translated vi="Làm lại">Start again</Translated>
          </button>
        </div>
        {placed === DECODE_ITEMS.length ? (
          <div className="panel" style={{ marginTop: '0.8rem' }} data-sort-done>
            <Translated
              vi={
                <p style={{ margin: 0 }}>
                  <b>Đã đặt cả mười một.</b> Đếm các ngăn: {tally.elementwise} theo từng phần tử, {tally.manyreads} đọc nhiều, {tally.combine} cần gộp trước, {tally.chain} chuỗi. Trong {forward} phép toán bên trong lượt tính xuôi, {inForward.elementwise} là theo từng phần tử: tính theo số phép toán, phần lớn lượt tính xuôi có đúng dạng kernel của chương này. Phép toán duy nhất đọc nhiều, phép matmul, là nơi tiêu tốn thời gian, và nó chiếm ba chương. Ngăn gộp chứa các phép rút gọn: layer norm bên trong lượt tính xuôi, và softmax với argmax của bước lấy mẫu. Ngăn chuỗi chứa tổng cộng dồn cho top-p và vòng lặp sinh, vốn là vòng lặp bao quanh bước chứ không phải một phép toán bên trong nó.
                </p>
              }
            >
              <p style={{ margin: 0 }}>
                <b>All eleven placed.</b> Count the bins: {tally.elementwise} per element, {tally.manyreads} with many
                reads, {tally.combine} that combine first, {tally.chain} chains. Of the {forward} operations inside the
                forward pass, {inForward.elementwise} are per element: by number of operations, most of the forward pass
                is the kernel shape of this chapter. The one operation with many reads, the matmul, is where the time
                goes, and it takes three chapters. The combine bin holds the reductions: layer norm inside the forward
                pass, and the softmax and argmax of sampling. The chain bin holds the running sum for top-p and the
                generation loop, which is the loop around the step rather than an operation inside it.
              </p>
            </Translated>
          </div>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
// Numbers the questions state, from the model.
const BLUR2: Params = { ...defaults('blur'), radius: 2 };
const BLUR2_READS = counts('blur', 8, BLUR2).inputs;
const BLUR2_TIMES = timesRead('blur', 'in', 8, BLUR2);

export const questionsS21: Record<string, Question> = {
  'ch02/2.1/q-criterion': {
    kind: 'choice',
    id: 'ch02/2.1/q-criterion',
    prompt: (
      <Translated vi="Điều kiện nào làm cho việc giao mỗi lần lặp của một vòng lặp cho một luồng riêng là an toàn?">
        <>Which condition makes it safe to give every iteration of a loop its own thread?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Các lần lặp đọc những tập đầu vào rời nhau, nên không phần tử đầu vào nào bị hai lần lặp cùng đọc.">
            The iterations read disjoint sets of inputs, so no input element is read by two of them.
          </Translated>
        ),
        why: (
          <Translated vi="Đủ cho các lần đọc, nhưng quá chặt. Các lần đọc của phép làm mờ chồng lên nhau rất nhiều mà vẫn ổn, như trình khám phá cho thấy.">
            Sufficient for the reads, and far too strong. Blur reads overlap heavily and it is fine, as the explorer shows.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không lần lặp nào đọc một vị trí mà một lần lặp khác của vòng lặp ghi vào.">
            No iteration reads a location that any other iteration of the loop writes.
          </Translated>
        ),
        why: (
          <Translated vi="Chỉ là một nửa. Vòng lặp for i: m = x[i] thỏa điều này, vì không gì nó đọc bị ghi vào, nhưng mọi lần lặp đều ghi m, và giá trị còn lại ở đó phụ thuộc vào luồng nào chạy sau cùng.">
            Only half of it. The loop for i: m = x[i] meets it, since nothing it reads is written, yet every iteration writes m, and the value left there depends on which thread ran last.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Mỗi lần lặp chỉ ghi vào những vị trí mà không lần lặp nào khác đọc hay ghi.">
            Each iteration writes only locations that no other iteration reads or writes.
          </Translated>
        ),
        correct: true,
        why: (
          <Translated vi="Khi đó không gì một luồng đọc có thể thay đổi dưới tay nó, và không gì nó ghi có thể bị ghi đè, nên kết quả không thể phụ thuộc vào thời điểm luồng nào chạy.">
            Then nothing a thread reads can change under it, and nothing it writes can be overwritten, so the result cannot depend on when any thread ran.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Mọi lần lặp làm cùng một lượng việc, nên không luồng nào xong muộn.">
            Every iteration does the same amount of work, so no thread finishes late.
          </Translated>
        ),
        why: (
          <Translated vi="Lượng việc không đều làm các luồng xong vào những thời điểm khác nhau. Nó không thay đổi điều mà luồng nào tính ra.">
            Uneven work makes threads finish at different times. It does not change what any of them computes.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Thân vòng lặp không có rẽ nhánh, nên mọi luồng đi cùng một đường.">
            The loop body has no branches, so every thread follows the same path.
          </Translated>
        ),
        why: (
          <Translated vi="Một nhánh là riêng của luồng đi vào nó. Phép làm mờ kiểm tra mép mảng trong mọi luồng.">
            A branch is private to the thread that takes it. Blur tests for the edge of the array in every thread.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong hình của sách, mỗi đầu ra có các mũi tên riêng từ đầu vào và không mũi tên nào đi từ một đầu ra sang đầu ra khác. Các lần đọc có thể chồng lên nhau tùy ý, vì một lần đọc không thay đổi gì. Điều kiện nằm ở các lần ghi, và nó có hai nửa: mỗi lần lặp chỉ ghi đầu ra của riêng nó, và không gì nó đọc bị một lần lặp khác ghi vào. Bỏ nửa đầu thì hai luồng có thể ghi cùng một vị trí, và luồng chạy sau cùng quyết định giá trị còn lại ở đó.
          </p>
        }
      >
        <p>
          In the book's picture every output has its own arrows from the input and no arrow runs from one output to
          another. Reads may overlap as much as they like, because a read changes nothing. The condition is about
          writes, and it has two halves: each iteration writes only its own output, and nothing it reads is written by
          another iteration. Drop the first half and two threads can write the same location, so the last one to run
          decides what stays there.
        </p>
      </Translated>
    ),
    source: '§2.1, pp. 21–23 (Fig. 2.2); the wording of the test is ours',
    vi: { source: '§2.1, pp. 21–23 (Fig. 2.2); cách phát biểu phép kiểm tra là của khóa học này' },
  },
  'ch02/2.1/q-inplace': {
    kind: 'choice',
    id: 'ch02/2.1/q-inplace',
    prompt: (
      <Translated
        vi={
          <>
            Vòng lặp 1 là <code>for i: y[i] = x[i-1] + x[i]</code>. Vòng lặp 2 là <code>for i: x[i] = x[i-1] + x[i]</code>. Vòng lặp nào có thể chạy với một luồng cho mỗi <code>i</code> và chắc chắn cho ra đáp án tuần tự?
          </>
        }
      >
        Loop 1 is <code>for i: y[i] = x[i-1] + x[i]</code>. Loop 2 is <code>for i: x[i] = x[i-1] + x[i]</code>. Which
        can be run with one thread for each <code>i</code> and be certain to give the sequential answer?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Chỉ vòng lặp 1: mỗi lần lặp ghi y[i] của riêng nó, và không gì nó đọc bị ghi vào.">
            Only loop 1: each iteration writes its own y[i], and nothing it reads is written.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Cả hai. Chúng làm cùng phép tính.">
            Both. They do the same arithmetic.
          </Translated>
        ),
        why: (
          <Translated vi="Cùng phép tính trên những giá trị khác nhau. Trong vòng lặp 2, x[i-1] là giá trị mà lần lặp i − 1 vừa ghi.">
            The same arithmetic on different values. In loop 2, x[i-1] is the value iteration i − 1 has just written.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ vòng lặp 2, vì nó không cần mảng thứ hai.">
            Only loop 2, because it needs no second array.
          </Translated>
        ),
        why: (
          <Translated vi="Ghi vào chính mảng nó đọc là điều làm nó hỏng.">
            Writing into the array it reads is exactly what breaks it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không vòng lặp nào, vì các lần lặp kề nhau đọc cùng một phần tử.">
            Neither, because neighbouring iterations read the same element.
          </Translated>
        ),
        why: (
          <Translated vi="Trong vòng lặp 1 cả hai đọc x[i-1], và không ai ghi vào nó. Đọc chung thì vô hại.">
            In loop 1 both read x[i-1], and nobody writes it. Shared reads are harmless.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Vòng lặp 2 đọc điều mà lần lặp trước vừa ghi, nên kết quả của nó phụ thuộc vào thứ tự các luồng chạy. Chạy theo thứ tự vòng lặp thì nó là tổng cộng dồn, còn gọi là tổng tiền tố (prefix sum). Chạy với luồng cuối cùng trước thì nó cho ra đáp án của vòng lặp 1, như hoạt hình cho thấy. Tổng tiền tố có thể tính song song, nhưng bằng một thuật toán khác, là nội dung của <a href={href('ch11')}>chương 11</a>.
          </p>
        }
      >
        <p>
          Loop 2 reads what the previous iteration wrote, so its result depends on the order in which the threads run.
          Run in loop order it is a running total, also called a prefix sum. Run with the last thread first it gives
          loop 1's answer, as the animation shows. A prefix sum can be computed in parallel, but by a different
          algorithm, which is <a href={href('ch11')}>chapter 11</a>.
        </p>
      </Translated>
    ),
    source: 'our example, by the method of §2.1 (pp. 21–23)',
    vi: { source: 'ví dụ của khóa học này, theo phương pháp của §2.1 (pp. 21–23)' },
  },
  'ch02/2.1/q-blur-reads': {
    kind: 'numeric',
    id: 'ch02/2.1/q-blur-reads',
    prompt: (
      <Translated
        vi={
          <>
            Một phép làm mờ 1D với bán kính 2 tính mỗi đầu ra từ {BLUR2_READS} đầu vào lân cận, với một luồng cho mỗi đầu ra. Tính trên mọi luồng, một phần tử đầu vào ở phần trong của mảng bị đọc bao nhiêu lần?
          </>
        }
      >
        A 1D blur with radius 2 computes each output from {BLUR2_READS} neighbouring inputs, with one thread per
        output. Across all threads, how many times is one input element in the interior of the array read?
      </Translated>
    ),
    answer: BLUR2_TIMES,
    unit: 'times',
    explain: (
      <Translated
        vi={
          <p>
            Một đầu vào ở vị trí j nằm trong vùng lân cận của các đầu ra từ j − 2 đến j + 2, tức {BLUR2_TIMES}{' '}đầu ra, nên {BLUR2_TIMES} luồng đọc nó. Phép làm mờ vẫn hoàn toàn song song dữ liệu: không luồng nào cần kết quả của luồng khác. Nhưng cùng những byte đó bị nạp từ bộ nhớ {BLUR2_TIMES} lần. Loại bỏ kiểu nạp lặp lại đó là chủ đề của <a href={href('ch05', '5.3')}>chương 5</a>.
          </p>
        }
      >
        <p>
          An input at position j lies in the neighbourhood of the outputs from j − 2 to j + 2, which is {BLUR2_TIMES}{' '}
          outputs, so {BLUR2_TIMES} threads read it. The blur is still fully data parallel: no thread needs another
          thread's result. But the same bytes are fetched from memory {BLUR2_TIMES} times. Removing that kind of
          repeated fetch is the subject of <a href={href('ch05', '5.3')}>chapter 5</a>.
        </p>
      </Translated>
    ),
    source: '§2.1, p. 21 (blur as the neighbourhood case); the count is ours',
    vi: { unit: 'lần', source: '§2.1, p. 21 (phép làm mờ là trường hợp vùng lân cận); phép đếm là của khóa học này' },
  },
  'ch02/2.1/q-average': {
    kind: 'choice',
    id: 'ch02/2.1/q-average',
    prompt: (
      <Translated
        vi={
          <>
            Sách nói độ sáng trung bình của một ảnh trông có vẻ toàn cục nhưng có thể chia thành những phép tính nhỏ hơn chạy độc lập. Điều gì làm việc đó khả thi?
          </>
        }
      >
        The book says the average brightness of an image appears global but can be broken into smaller computations
        that run independently. What makes that possible?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Phép cộng có thể nhóm lại: cộng từng đoạn độc lập, rồi gộp các tổng riêng phần.">
            Addition can be regrouped: sum chunks independently, then combine the partial sums.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Mỗi điểm ảnh có thể được giao một luồng riêng, luồng đó cộng thẳng giá trị của nó vào một tổng chung.">
            Each pixel can be given its own thread, which adds its value straight into one shared total.
          </Translated>
        ),
        why: (
          <Translated vi="Nhiều luồng cùng đọc và ghi một vị trí là một race condition (tranh chấp dữ liệu): các cập nhật bị mất. Chương 9 nói đúng về điều này.">
            Many threads reading and writing one location is a race: updates get lost. Chapter 9 is about exactly this.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Ảnh có thể được giảm mẫu trước, để số điểm ảnh cần đọc và cộng ít đi nhiều.">
            The image can be downsampled first, so that far fewer pixels need to be read and summed.
          </Translated>
        ),
        why: (
          <Translated vi="Cách đó tính trung bình của một ảnh khác.">
            That computes the average of a different image.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không thể làm chính xác. Sách đang mô tả một phép xấp xỉ.">
            It is not possible exactly. The book is describing an approximation.
          </Translated>
        ),
        why: (
          <Translated vi="Phép phân rã là chính xác trong số học số thực. Xem ghi chú về làm tròn trong phần giải thích.">
            The decomposition is exact in real arithmetic. See the note on rounding in the explanation.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Không tổng riêng phần nào cần tổng riêng phần khác, nên chúng độc lập, và một bước gộp ở cuối cho ra tổng. Hãy thử trong trình khám phá với tổng riêng phần gồm 4 phần tử. Một lưu ý cho sau này: phép cộng dấu phẩy động không có tính kết hợp chính xác, nên nhóm lại một tổng có thể làm thay đổi các bit cuối của nó. Sách bàn điều đó trong{' '}<a href={href('appA')}>phụ lục A</a>, và <a href={href('ch10')}>chương 10</a> xây dựng bước gộp một cách đúng đắn.
          </p>
        }
      >
        <p>
          No partial sum needs another partial sum, so they are independent, and one combine step at the end gives the
          total. Try it in the explorer with partial sums of 4. One caution for later: floating-point addition is not
          exactly associative, so regrouping a sum can change its last bits. The book takes that up in{' '}
          <a href={href('appA')}>appendix A</a>, and <a href={href('ch10')}>chapter 10</a> builds the combine step
          properly.
        </p>
      </Translated>
    ),
    source: '§2.1, p. 21; the rounding caution is from appendix A, pp. 593–594',
    vi: { source: '§2.1, p. 21; lưu ý về làm tròn lấy từ phụ lục A, pp. 593–594' },
  },
};

export function S21() {
  return (
    <>
      <Translated
        vi={
          <p>
            Trước khi viết một kernel, bạn cần nhìn được một phép tính và nói nó có thể cắt thành những phần không cần nhau hay không. Sách gọi tính chất đó là song song dữ liệu: các phép tính trên những phần khác nhau của dữ liệu có thể làm độc lập với nhau, và do đó làm cùng lúc. Điều này quan trọng vì khi một ứng dụng chạy chậm ngày nay, lý do thường gặp là lượng dữ liệu, và một ứng dụng giàu song song dữ liệu sẽ nhanh lên tương ứng với phần cứng được cấp cho nó.
          </p>
        }
      >
        <p>
          Before you write a kernel you need to be able to look at a computation and say whether it can be cut into
          pieces that do not need each other. The book calls that property data parallelism: computations on different
          parts of the data can be done independently of each other, and therefore at the same time. It matters because
          when an application is slow today the usual reason is the amount of data, and an application rich in data
          parallelism gets faster in proportion to the hardware it is given.
        </p>
      </Translated>
      <Source section="2.1" pages="21" />

      <h2>
        <Translated vi="Ba trường hợp, theo mức độ">Three cases, graded</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách phân cấp ý tưởng này bằng ba phép toán trên ảnh. Chuyển một điểm ảnh màu sang xám cần điểm ảnh đó và không gì khác. Làm mờ cần một vùng lân cận nhỏ các điểm ảnh quanh mỗi đầu ra. Độ sáng trung bình của cả một ảnh trông có vẻ toàn cục, vì một con số phụ thuộc vào mọi điểm ảnh, nhưng nó có thể chia thành những phép tính nhỏ hơn chạy độc lập.
          </p>
        }
      >
        <p>
          The book grades the idea with three image operations. Converting a colour pixel to gray needs that pixel and
          nothing else. Blurring needs a small neighbourhood of pixels around each output. The average brightness of a
          whole image looks global, since one number depends on every pixel, but it can be broken into smaller
          computations that run independently.
        </p>
      </Translated>
      <Source section="2.1" pages="21" />
      <Translated
        vi={
          <p>
            Xét chính xác trường hợp đầu tiên. Một điểm ảnh giữ ba phân số r, g và b, mỗi số nằm giữa 0 cho tối và 1 cho cường độ tối đa. Độ chói (luminance) của nó là một tổng có trọng số, <code>L = 0.299 r + 0.587 g + 0.114 b</code>. Với các điểm ảnh trong mảng đầu vào <code>I</code> và các độ chói trong mảng đầu ra <code>O</code>, giá trị{' '}<code>O[k]</code> phụ thuộc vào <code>I[k]</code> và không vào điểm ảnh nào khác. Trong bộ nhớ, một hàng điểm ảnh được lưu xen kẽ, r g b r g b và cứ thế, nên điểm ảnh k là ba giá trị liên tiếp.
          </p>
        }
      >
        <p>
          Take the first case exactly. A pixel holds three fractions r, g and b, each between 0 for dark and 1 for full
          intensity. Its luminance is a weighted sum, <code>L = 0.299 r + 0.587 g + 0.114 b</code>. With the pixels in an
          input array <code>I</code> and the luminances in an output array <code>O</code>, the value{' '}
          <code>O[k]</code> depends on <code>I[k]</code> and on no other pixel. In memory a row of pixels is stored
          interleaved, r g b r g b and so on, so pixel k is three consecutive values.
        </p>
      </Translated>
      <Source section="2.1" pages="21–23" figs={tr('Eq. 2.1, Fig. 2.2', 'Pt. 2.1, Hình 2.2')} />
      <Translated
        vi={
          <p>
            Đây là tính chất đó dưới dạng một phép kiểm tra bạn có thể áp dụng cho một vòng lặp: <b>không đầu ra nào được tính từ một đầu ra khác</b>. Đầu vào có thể được bao nhiêu luồng đọc cũng được, và các lần đọc của hai đầu ra có thể chồng lên nhau. Điều không được xảy ra là một luồng cần một giá trị mà một luồng khác của cùng lần khởi chạy tạo ra.
          </p>
        }
      >
        <p>
          Here is the property as a test you can apply to a loop: <b>no output is computed from another output</b>.
          Inputs may be read by as many threads as you like, and the reads of two outputs may overlap. What must not
          happen is one thread needing a value that another thread of the same launch produces.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Cách phát biểu phép kiểm tra là của khóa học này. Sách đưa ra định nghĩa và một hình trong đó không mũi tên nào đi giữa hai đầu ra, và để phép kiểm tra ở dạng ngầm hiểu.
            </p>
          }
        >
          <p>
            The wording of the test is ours. The book gives the definition and a picture in which no arrow runs between
            two outputs, and leaves the test implicit.
          </p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Một đầu ra đọc những gì">What one output reads</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trình khám phá áp dụng phép kiểm tra cho tám phép toán. Hãy bắt đầu với chuyển xám và cộng vectơ, rồi kéo bán kính làm mờ và xem con số nào trong hai con số thay đổi. Để hai phép tính tổng và softmax lại sau cùng.
          </p>
        }
      >
        <p>
          The explorer applies the test to eight operations. Start with grayscale and vector addition, then drag the
          blur radius and watch which of the two counts moves. Leave the two sums and the softmax for last.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Trình khám phá cho thấy phần tử nào được đọc. Nó không nói gì về một lần đọc mất bao lâu, hay lần đọc thứ hai của cùng phần tử có được phục vụ từ bộ nhớ đệm (cache) hay không. Các thứ tự luồng nó thử chạy hết trọn một luồng rồi đến luồng khác; luồng thật xen kẽ nhau mịn hơn.
            </p>
          }
        >
          <p>
            The explorer shows which elements are read. It says nothing about how long a read takes, or whether a
            second read of the same element is served from a cache. The thread orders it tries run one whole thread
            after another; real threads interleave more finely.
          </p>
        </Translated>
      </Simplification>
      <ReadExplorer />
      <Translated
        vi={
          <p>
            Ba điều rút ra từ nó. Đọc chồng lên nhau thì không sao: phép làm mờ bán kính 4 đọc chín đầu vào cho mỗi đầu ra và "đầu ra cần có trước" vẫn là 0. Trung bình và softmax cần một giá trị được gộp từ nhiều phần tử trước khi hoàn tất được bất kỳ đầu ra nào, và các phần của bước gộp đó tự chúng độc lập. Tổng cộng dồn tại chỗ là phép duy nhất có màu đỏ: mọi đầu ra đều chờ đầu ra trước nó.
          </p>
        }
      >
        <p>
          Three things to take from it. Overlapping reads are fine: a blur of radius 4 reads nine inputs per output and
          "outputs needed first" stays at 0. The mean and the softmax need a value combined from many elements before
          any output can be finished, and the pieces of that combine are themselves independent. The in-place running
          sum is the only one with red in it: every output waits for the one before.
        </p>
      </Translated>
      <Ask q={questionsS21['ch02/2.1/q-criterion']} />
      <Ask q={questionsS21['ch02/2.1/q-blur-reads']} label={tr('Work it out', 'Tự tính ra')} />
      <Ask q={questionsS21['ch02/2.1/q-average']} />

      <h2>
        <Translated vi="Vì sao kết quả của luồng khác là vấn đề">Why another thread's result is the problem</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Khi một kernel chạy, bạn không kiểm soát thứ tự các luồng của nó thực thi. Một phép toán chỉ đọc đầu vào cho cùng đáp án theo mọi thứ tự. Một phép toán trong đó một luồng đọc điều mà luồng khác ghi thì không. Hoạt hình chạy hai vòng lặp gần như giống hệt nhau, mỗi vòng theo hai thứ tự.
          </p>
        }
      >
        <p>
          When a kernel runs, you do not control the order in which its threads execute. An operation that reads only
          inputs gives the same answer in any order. An operation in which one thread reads what another writes does
          not. The animation runs two nearly identical loops in two orders each.
        </p>
      </Translated>
      <NoteGroup>
        <Aside>
          <Translated
            vi={
              <p>
                Ví dụ là của khóa học này. 8 con số là những số sách dùng khi giới thiệu tổng cộng dồn ở chương 11 (p. 252).
              </p>
            }
          >
            <p>
              The example is ours. The 8 numbers are the ones the book uses when it introduces the running total in
              chapter 11 (p. 252).
            </p>
          </Translated>
        </Aside>
        <Simplification>
          <Translated
            vi={
              <p>
                Mỗi thứ tự trong hoạt hình chạy hết trọn một luồng rồi đến luồng khác. Luồng thật xen kẽ nhau mịn hơn và một số chạy cùng lúc, điều đó chỉ thêm những thứ tự có thể khác với vòng lặp.
              </p>
            }
          >
            <p>
              Each order in the animation runs one whole thread after another. Real threads interleave more finely and
              some run at the same moment, which only adds orders that can differ from the loop.
            </p>
          </Translated>
        </Simplification>
      </NoteGroup>
      <OrderScene />
      <Ask q={questionsS21['ch02/2.1/q-inplace']} />

      <h2>
        <Translated vi="Một bước decode, phân loại theo sự phụ thuộc">One decode step, sorted by dependency</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Cùng phép kiểm tra đó phân loại những phép toán bạn đã biết. Lấy một bước decode của một transformer và hỏi, với mỗi phép toán, một đầu ra của nó cần gì. Bốn dạng bao quát tất cả. Dạng đầu tiên chính là kernel của chương này, và là thứ tạo nên mốc dự án M0 của dự án khóa học. Ba dạng còn lại chỉ ra các chương xử lý chúng.
          </p>
        }
      >
        <p>
          The same test sorts the operations you already know. Take one decode step of a transformer and ask, for each
          operation, what one of its outputs needs. Four shapes cover them. The first is exactly the kernel of this
          chapter, and it is what milestone M0 of the course project is made of. The other three name the chapters that
          deal with them.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Ví dụ của sách trong mục này là các phép toán trên ảnh. Việc ánh xạ sang các phép toán của một mô hình ngôn ngữ là của khóa học này.
            </p>
          }
        >
          <p>
            The book's examples in this section are image operations. The mapping to the operations of a language model
            is ours.
          </p>
        </Translated>
      </Beyond>
      <SortDecodeStep />

      <h2>
        <Translated vi="Song song dữ liệu và song song tác vụ">Data parallelism and task parallelism</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách tách song song dữ liệu khỏi song song tác vụ, vốn đến từ việc chia một ứng dụng thành các tác vụ khác nhau có thể chạy cùng lúc. Ví dụ của sách là một ứng dụng cần một phép cộng vectơ và một phép nhân ma trận với vectơ: hai tác vụ, và là song song tác vụ nếu hai việc đó chạy độc lập được. Truyền dữ liệu và I/O cũng là tác vụ. Trong hai loại, sách nói song song dữ liệu là nguồn chính của tốc độ và khả năng mở rộng. Nó tăng theo dữ liệu: gấp đôi điểm ảnh là gấp đôi lượng việc độc lập, trong khi số tác vụ vẫn như cũ. Theo cách nói của suy luận (ví dụ của khóa học này, không phải của sách), một batch là song song dữ liệu, còn tách token cho yêu cầu tiếp theo trong khi bước trước chạy trên GPU là song song tác vụ.
          </p>
        }
      >
        <p>
          The book separates data parallelism from task parallelism, which comes from splitting an application into
          different tasks that can run at the same time. Its example is an application that needs a vector addition and
          a matrix-vector multiplication: two tasks, and task parallelism if the two can run independently. Data
          transfers and I/O are tasks as well. Of the two kinds, the book says data parallelism is the main source of
          speed and of scalability. It grows with the data: twice the pixels is twice the independent work, while the
          number of tasks stays what it was. In inference terms (our example, not the book's), a batch is data
          parallelism, and tokenising the next request while the previous step runs on the GPU is task parallelism.
        </p>
      </Translated>
      <Source section="2.1" pages="23" />

      <h2>
        <Translated vi="Tư duy song song">Parallel thinking</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Viết mã song song dữ liệu nghĩa là tổ chức phép tính xoay quanh dữ liệu, để các phần độc lập lộ ra và có thể giao cho các luồng. Với chuyển xám, các phần đó hiển nhiên. Với trung bình, chúng phải được tìm ra. Sách gọi kỹ năng tìm ra chúng trong những trường hợp khó hơn là tư duy song song, và phần lớn những gì tiếp theo trong sách là luyện tập kỹ năng đó.
          </p>
        }
      >
        <p>
          Writing data-parallel code means organising the computation around the data, so that the independent pieces
          are visible and can be handed to threads. For grayscale the pieces are obvious. For the average they had to
          be found. The book calls the skill of finding them in harder cases parallel thinking, and most of what
          follows in the book is practice at it.
        </p>
      </Translated>
      <Source section="2.1" pages="21, 23" />
      <Translated
        vi={
          <p>
            Không gì trong mục này phụ thuộc vào GPU bạn dùng. Một đầu ra có cần đầu ra khác hay không là tính chất của phép tính, và đó là cùng một câu hỏi trên CUDA, HIP và Metal. Bản thân các kernel chuyển xám và làm mờ được viết ở <a href={href('ch03')}>chương 3</a>. Chương này dùng thứ đơn giản hơn, phép cộng hai vectơ, và trước kernel nó phải dựng chương trình bao quanh:{' '}<a href={href('ch02', '2.2')}>mục 2.2</a>.
          </p>
        }
      >
        <p>
          Nothing in this section depends on the GPU you use. Whether one output needs another is a property of the
          computation, and it is the same question on CUDA, HIP and Metal. The grayscale and blur kernels themselves
          are written in <a href={href('ch03')}>chapter 3</a>. This chapter uses something simpler, the addition of two
          vectors, and before the kernel it has to set up the program around it:{' '}
          <a href={href('ch02', '2.2')}>section 2.2</a>.
        </p>
      </Translated>
    </>
  );
}
