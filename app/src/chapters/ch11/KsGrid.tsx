import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Code } from '../../components/Code';
import {
  KIND_LABEL,
  active,
  firstFailure,
  iterations,
  kernelSource,
  nextRelease,
  rangeShort,
  rangeText,
  stateAt,
  windowCoef,
  type Coef,
  type Level,
  type Read,
  type Step,
  type Trace,
  type TreeNode,
} from '../../sim/ksblock';
import { Translated, tr } from '../../lib/i18n';

// Chapter 11's drawing of a Kogge-Stone scan as rows of the shared buffer: the load on
// top, then what each iteration wrote, the way Figs. 11.2 and 11.4 draw it. Sections 11.2
// and 11.3 and the scene "Doubling strides" all draw with it. It only draws: rows, cells,
// adders and arrows come from src/sim/ksblock.ts through the two builders at the bottom.

export type CellState = 'plain' | 'final' | 'new' | 'bad' | 'stale' | 'empty' | 'carried';
export interface GridCell {
  value: number;
  coef: Coef;
  state: CellState;
}
export type Tone = 'plain' | 'hot' | 'bad' | 'tree' | 'faint';
export interface GridArrow {
  from: [number, number];
  to: [number, number];
  kind: 'add' | 'copy' | 'carry';
  tone: Tone;
}
export interface GridMark {
  row: number;
  pos: number;
  tone: 'read' | 'bad' | 'write' | 'tree';
}

const LW = 112; // label column
const TOP = 18; // position numbers
const CH = 34; // cell height
const GAP = 30; // between rows: room for the adders

const show = (v: number) => (Number.isNaN(v) ? '?' : String(v));
const STROKE: Record<Tone, string> = {
  plain: 'var(--ink-3)',
  hot: 'var(--ink)',
  bad: 'var(--fail)',
  tree: 'var(--ink)',
  faint: 'var(--rule)',
};
const WIDTH: Record<Tone, number> = { plain: 1, hot: 2, bad: 2.2, tree: 2.6, faint: 1 };

export function KsGrid({
  rows,
  labels,
  arrows = [],
  marks = [],
  cellW = 60,
  ranges = true,
  label,
  extra,
  extraHeight = 0,
}: {
  rows: (GridCell | null)[][];
  /** One or two lines per row, drawn left of it. */
  labels: [string, string?][];
  arrows?: GridArrow[];
  marks?: GridMark[];
  cellW?: number;
  /** Write under each value the inputs it counts. */
  ranges?: boolean;
  label: string;
  /** Drawn on top, in the same coordinates (for a scene's headline, say). */
  extra?: (geo: { x: (pos: number) => number; y: (row: number) => number; width: number; height: number }) => ReactNode;
  extraHeight?: number;
}) {
  const B = rows[0]?.length ?? 0;
  const width = LW + B * cellW + 8;
  const height = TOP + rows.length * (CH + GAP) - GAP + 8 + extraHeight;
  const x = (pos: number) => LW + pos * cellW;
  const y = (row: number) => TOP + row * (CH + GAP);
  const cx = (pos: number) => x(pos) + (cellW - 4) / 2;
  const adders = new Map<string, Tone>();
  for (const a of arrows)
    if (a.kind === 'add') {
      const key = `${a.to[0]}:${a.to[1]}`;
      const rank: Tone[] = ['faint', 'plain', 'hot', 'tree', 'bad'];
      const prev = adders.get(key);
      if (!prev || rank.indexOf(a.tone) > rank.indexOf(prev)) adders.set(key, a.tone);
    }
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={label}
      data-part="ks-grid"
      style={{ width: '100%', maxWidth: width, minWidth: Math.round(width * 0.78), display: 'block' }}
    >
      {Array.from({ length: B }, (_, pos) => (
        <text key={pos} x={cx(pos)} y={11} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>
          {pos}
        </text>
      ))}
      {labels.map(([a, b], r) => (
        <g key={r}>
          <text x={4} y={y(r) + (b ? 14 : 21)} fontSize="11" fontWeight="600">{a}</text>
          {b ? <text x={4} y={y(r) + 28} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{b}</text> : null}
        </g>
      ))}
      {arrows.map((a, i) => {
        const [r0, p0] = a.from;
        const [r1, p1] = a.to;
        const x1 = cx(p0);
        const y1 = y(r0) + CH;
        const x2 = cx(p1);
        const y2 = a.kind === 'add' ? y(r1) - 13 : y(r1);
        return (
          <line
            key={i}
            data-part="arrow"
            data-kind={a.kind}
            data-tone={a.tone}
            data-from={`${r0}:${p0}`}
            data-to={`${r1}:${p1}`}
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke={STROKE[a.tone]}
            strokeWidth={WIDTH[a.tone]}
            strokeDasharray={a.kind === 'copy' ? '4 3' : a.kind === 'carry' ? '1 3' : undefined}
          />
        );
      })}
      {[...adders].map(([key, tone]) => {
        const [r, p] = key.split(':').map(Number);
        return (
          <g key={key} data-part="adder" data-row={r} data-pos={p}>
            <line x1={cx(p)} y1={y(r) - 7} x2={cx(p)} y2={y(r)} stroke={STROKE[tone]} strokeWidth={WIDTH[tone]} />
            <circle cx={cx(p)} cy={y(r) - 13} r={6} fill="var(--page)" stroke={STROKE[tone]} strokeWidth={WIDTH[tone]} />
            <text x={cx(p)} y={y(r) - 9.5} fontSize="10" textAnchor="middle" style={{ fill: STROKE[tone] }}>+</text>
          </g>
        );
      })}
      {rows.map((row, r) =>
        row.map((c, pos) => {
          if (!c) return null;
          const st = c.state;
          const fill = st === 'final' ? 'var(--shared-wash)' : st === 'bad' ? 'var(--fail-wash)' : st === 'empty' ? 'transparent' : 'var(--cell)';
          const stroke = st === 'bad' ? 'var(--fail)' : st === 'empty' || st === 'stale' || st === 'carried' ? 'var(--ink-3)' : st === 'new' ? 'var(--ink)' : 'var(--shared)';
          const muted = st === 'carried' || st === 'stale';
          return (
            <g key={`${r}:${pos}`} data-part="cell" data-row={r} data-pos={pos} data-state={st} data-value={show(c.value)} data-range={rangeShort(c.coef)}>
              <rect
                x={x(pos)}
                y={y(r)}
                width={cellW - 4}
                height={CH}
                rx={2}
                fill={fill}
                stroke={stroke}
                strokeWidth={st === 'new' ? 2 : 1}
                strokeDasharray={st === 'empty' || st === 'stale' || st === 'carried' ? '3 2' : undefined}
              />
              {st === 'empty' ? null : (
                <>
                  <text x={cx(pos)} y={y(r) + (ranges ? 14 : 21)} fontSize="12" fontWeight={st === 'final' ? 650 : 500} textAnchor="middle"
                    style={{ fill: st === 'bad' ? 'var(--fail)' : muted ? 'var(--ink-3)' : 'var(--ink)' }}>
                    {show(c.value)}
                  </text>
                  {ranges ? (
                    <text x={cx(pos)} y={y(r) + 27} fontSize={rangeShort(c.coef).length > 9 ? 7.5 : 8.5} textAnchor="middle"
                      style={{ fill: st === 'bad' ? 'var(--fail)' : 'var(--ink-2)' }}>
                      {rangeShort(c.coef)}
                    </text>
                  ) : null}
                </>
              )}
            </g>
          );
        }),
      )}
      {marks.map((m, i) => (
        <rect
          key={i}
          data-part="mark"
          data-tone={m.tone}
          data-row={m.row}
          data-pos={m.pos}
          x={x(m.pos) - 3}
          y={y(m.row) - 3}
          width={cellW + 2}
          height={CH + 6}
          rx={4}
          fill="none"
          stroke={m.tone === 'bad' ? 'var(--fail)' : 'var(--ink)'}
          strokeWidth={m.tone === 'tree' ? 1.6 : 2.4}
          strokeDasharray={m.tone === 'write' ? '5 3' : undefined}
        />
      ))}
      {extra ? extra({ x: cx, y, width, height }) : null}
    </svg>
  );
}

// ---------------------------------------------------------------- builders
const isWindowFinal = (c: Coef, pos: number, B: number) => !!c && c.every((v, k) => v === (k <= pos ? 1 : 0)) && c.length === B;
const sameCoef = (a: Coef, b: Coef) => !!a && !!b && a.every((v, k) => v === b[k]);

/** Fig. 11.2 for the levels shown so far: every position in lockstep. */
export function fromLevels(
  lv: Level[],
  upto: number,
  opts: { tree?: { nodes: TreeNode[]; edges: [TreeNode, TreeNode][] }; hot?: number; carries?: boolean } = {},
): { rows: (GridCell | null)[][]; labels: [string, string?][]; arrows: GridArrow[]; marks: GridMark[] } {
  const B = lv[0].values.length;
  const rows = lv.slice(0, upto + 1).map((l, r) =>
    l.values.map((value, pos): GridCell => {
      const coef = l.coef[pos];
      const fin = isWindowFinal(coef, pos, B);
      return { value, coef, state: fin ? 'final' : r > 0 && r === opts.hot && l.added[pos] ? 'new' : 'plain' };
    }),
  );
  const inTree = (r: number, p: number) => !!opts.tree?.edges.some(([a, b]) => b.level === r && b.pos === p && a.level === r - 1);
  const treeEdge = (a: [number, number], b: [number, number]) => !!opts.tree?.edges.some(([u, v]) => u.level === a[0] && u.pos === a[1] && v.level === b[0] && v.pos === b[1]);
  const arrows: GridArrow[] = [];
  for (let r = 1; r <= upto; r++) {
    const l = lv[r];
    const stride = l.stride!;
    for (let pos = 0; pos < B; pos++) {
      const tone = (from: [number, number]): Tone => (treeEdge(from, [r, pos]) ? 'tree' : r === opts.hot ? 'hot' : 'plain');
      if (l.added[pos]) {
        arrows.push({ from: [r - 1, pos], to: [r, pos], kind: 'add', tone: tone([r - 1, pos]) });
        arrows.push({ from: [r - 1, pos - stride], to: [r, pos], kind: 'add', tone: tone([r - 1, pos - stride]) });
      } else if (opts.carries !== false) arrows.push({ from: [r - 1, pos], to: [r, pos], kind: 'carry', tone: inTree(r, pos) ? 'tree' : 'faint' });
    }
  }
  const marks: GridMark[] = (opts.tree?.nodes ?? []).filter((n) => n.level > 0 && n.level <= upto).map((n) => ({ row: n.level, pos: n.pos, tone: 'tree' as const }));
  const labels: [string, string?][] = lv.slice(0, upto + 1).map((l) => (l.stride === null ? [tr('input', 'đầu vào')] : [tr(`stride ${l.stride}`, `stride ${l.stride}`), tr(`${B - l.stride} additions`, `${B - l.stride} phép cộng`)]));
  return { rows, labels, arrows, marks };
}

/**
 * What each iteration of a thread-level run has written so far, as rows (Fig. 11.4 for two
 * buffers), with the arrows of every finished addition or copy and the marks of the
 * current step. Reads that found the wrong write draw their arrow from where the value
 * really came from, in red.
 */
export function fromTrace(trace: Trace, n: number): { rows: (GridCell | null)[][]; labels: [string, string?][]; arrows: GridArrow[]; marks: GridMark[] } {
  const { cfg } = trace;
  const B = cfg.threads;
  const L = iterations(B);
  const st = stateAt(trace, n);
  const dbl = cfg.kind === 'double';
  const rows: (GridCell | null)[][] = [];
  for (let r = 0; r <= L; r++) {
    rows.push(
      Array.from({ length: B }, (_, pos): GridCell => {
        const s = st.rows[r][pos];
        if (s) {
          const right = sameCoef(s.coef, windowCoef(pos, r, B));
          return { value: s.value, coef: s.coef, state: !right ? 'bad' : isWindowFinal(s.coef, pos, B) ? 'final' : 'plain' };
        }
        if (r === 0) return { value: NaN, coef: null, state: 'empty' };
        const idle = !active(pos, r - 1);
        const th = st.threads[pos];
        const passed = th.done || th.iter > r - 1 || (th.iter === r - 1 && (th.last === 'swap' || th.last === 'barrier2'));
        if (!dbl && idle) {
          const above = rows[r - 1][pos];
          return above && above.state !== 'empty' ? { value: above.value, coef: above.coef, state: 'carried' } : { value: NaN, coef: null, state: 'empty' };
        }
        if (dbl && idle && !cfg.copy && passed) {
          const old = r >= 2 ? rows[r - 2][pos] : null;
          return old && old.state !== 'empty' ? { value: old.value, coef: old.coef, state: 'stale' } : { value: NaN, coef: null, state: 'stale' };
        }
        return { value: NaN, coef: null, state: 'empty' };
      }),
    );
  }
  const arrows: GridArrow[] = [];
  const steps = trace.steps.slice(0, Math.min(n, trace.steps.length));
  const lastRead = new Map<number, typeof steps[number]>();
  steps.forEach((s, i) => {
    if (s.thread < 0) return;
    if (s.op === 'read') lastRead.set(s.thread, s);
    const src = s.op === 'write' ? lastRead.get(s.thread) : s.op === 'add' || s.op === 'copy' ? s : undefined;
    if (!s.write || !src?.reads || s.op === 'load') return;
    const current = i === steps.length - 1;
    for (const rd of src.reads) {
      if (rd.found === null) continue;
      arrows.push({
        from: [rd.found + 1, rd.slot],
        to: [s.write.iter + 1, s.write.slot],
        kind: s.op === 'copy' ? 'copy' : 'add',
        tone: !rd.ok ? 'bad' : current ? 'hot' : 'plain',
      });
    }
  });
  const marks: GridMark[] = [];
  const cur = steps[steps.length - 1];
  if (cur && cur.thread >= 0) {
    for (const rd of cur.reads ?? []) {
      if (rd.found !== null) marks.push({ row: rd.found + 1, pos: rd.slot, tone: rd.ok ? 'read' : 'bad' });
      else marks.push({ row: rd.expected + 1, pos: rd.slot, tone: 'bad' });
    }
    if (cur.write) marks.push({ row: cur.write.iter + 1, pos: cur.write.slot, tone: 'write' });
  }
  const labels: [string, string?][] = Array.from({ length: L + 1 }, (_, r) => {
    if (!dbl) return r === 0 ? ['buffer_s', tr('after the load', 'sau khi nạp')] : ['buffer_s', tr(`after stride ${2 ** (r - 1)}`, `sau stride ${2 ** (r - 1)}`)];
    if (r === 0) return ['buffer1_s', tr('the load', 'lần nạp')];
    return [`buffer${(r % 2) + 1}_s`, tr(`iteration ${r - 1} wrote`, `lần lặp ${r - 1} đã ghi`)]; // i18n-ignore: buffer variable name
  });
  return { rows, labels, arrows, marks };
}

// ---------------------------------------------------------------- a thread-level stepper
const val = (v: number) => (Number.isNaN(v) ? tr('garbage', 'rác') : String(v));
const carried = (r: Read) => `${val(r.value)} (${r.coef ? rangeText(r.coef, true) : tr('garbage', 'rác')})`;

/** One sentence saying what a step did. */
export function describeStep(s: Step, trace: Trace): string {
  const { cfg } = trace;
  const buf = (b: number) => (cfg.kind === 'double' ? `buffer${b + 1}_s` : 'buffer_s'); // i18n-ignore: buffer variable name
  const stride = 2 ** s.iter;
  if (s.thread < 0) {
    if (cfg.kind === 'double')
      return s.iter === 0
        ? tr('The barrier opens: every thread has loaded its element into buffer1_s, so iteration 0 may read them.', 'Rào chắn mở: mọi luồng đã nạp phần tử của mình vào buffer1_s, nên lần lặp 0 được phép đọc chúng.')
        : tr(`The barrier opens: every thread has written its iteration-${s.iter - 1} value, so iteration ${s.iter} may read them.`, `Rào chắn mở: mọi luồng đã ghi giá trị của lần lặp ${s.iter - 1}, nên lần lặp ${s.iter} được phép đọc chúng.`);
    return s.barrier === 1
      ? tr(`Barrier 1 opens: every thread has arrived, so every write before it is done and iteration ${s.iter} (stride ${stride}) may read.`, `Rào chắn 1 mở: mọi luồng đã tới, nên mọi lần ghi trước nó đã xong và lần lặp ${s.iter} (stride ${stride}) được phép đọc.`)
      : tr(`Barrier 2 opens: every read of iteration ${s.iter} is done, so it is now safe to overwrite buffer_s.`, `Rào chắn 2 mở: mọi lần đọc của lần lặp ${s.iter} đã xong, nên giờ có thể ghi đè buffer_s an toàn.`);
  }
  const t = tr(`Thread ${s.thread}`, `Luồng ${s.thread}`);
  switch (s.op) {
    case 'load':
      return tr(`${t} loads ${s.write!.coef!.some(Boolean) ? `x${s.thread} = ${s.write!.value}` : 'the identity 0'} into ${buf(0)}[${s.thread}].`, `${t} nạp ${s.write!.coef!.some(Boolean) ? `x${s.thread} = ${s.write!.value}` : 'phần tử đơn vị 0'} vào ${buf(0)}[${s.thread}].`);
    case 'barrier1':
      return cfg.kind === 'double'
        ? tr(`${t} reaches the barrier of iteration ${s.iter} and waits for the whole block.`, `${t} tới rào chắn của lần lặp ${s.iter} và chờ cả khối.`)
        : tr(`${t} reaches barrier 1 of iteration ${s.iter} (stride ${stride}) and waits.`, `${t} tới rào chắn 1 của lần lặp ${s.iter} (stride ${stride}) và chờ.`);
    case 'barrier2':
      return cfg.temp
        ? tr(`${t} reaches barrier 2 of iteration ${s.iter} and waits until every thread has read.`, `${t} tới rào chắn 2 của lần lặp ${s.iter} và chờ đến khi mọi luồng đã đọc xong.`)
        : tr(`${t} reaches the barrier after its store, iteration ${s.iter}, and waits.`, `${t} tới rào chắn sau lần ghi của nó, lần lặp ${s.iter}, và chờ.`);
    case 'read': {
      const [a, b] = s.reads!;
      const sum = `${val(a.value + b.value)}`;
      return cfg.temp
        ? tr(`${t}, stride ${stride}: temp = buffer_s[${s.thread}] + buffer_s[${s.thread - stride}] = ${carried(a)} + ${carried(b)} = ${sum}.`, `${t}, stride ${stride}: temp = buffer_s[${s.thread}] + buffer_s[${s.thread - stride}] = ${carried(a)} + ${carried(b)} = ${sum}.`)
        : tr(`${t}, stride ${stride}: loads buffer_s[${s.thread}] = ${carried(a)} and buffer_s[${s.thread - stride}] = ${carried(b)} for its +=.`, `${t}, stride ${stride}: nạp buffer_s[${s.thread}] = ${carried(a)} và buffer_s[${s.thread - stride}] = ${carried(b)} cho phép += của nó.`);
    }
    case 'write':
      return cfg.temp
        ? tr(`${t} stores temp = ${val(s.write!.value)} (${rangeText(s.write!.coef, true)}) into buffer_s[${s.thread}].`, `${t} ghi temp = ${val(s.write!.value)} (${rangeText(s.write!.coef, true)}) vào buffer_s[${s.thread}].`)
        : tr(`${t} finishes its +=: buffer_s[${s.thread}] = ${val(s.write!.value)} (${rangeText(s.write!.coef, true)}).`, `${t} hoàn tất phép +=: buffer_s[${s.thread}] = ${val(s.write!.value)} (${rangeText(s.write!.coef, true)}).`);
    case 'add': {
      const [a, b] = s.reads!;
      return tr(`${t}, stride ${stride}: ${buf(s.write!.buf)}[${s.thread}] = ${buf(a.buf)}[${s.thread}] + ${buf(b.buf)}[${s.thread - stride}] = ${carried(a)} + ${carried(b)} = ${val(s.write!.value)}.`, `${t}, stride ${stride}: ${buf(s.write!.buf)}[${s.thread}] = ${buf(a.buf)}[${s.thread}] + ${buf(b.buf)}[${s.thread - stride}] = ${carried(a)} + ${carried(b)} = ${val(s.write!.value)}.`);
    }
    case 'copy':
      return tr(`${t} is idle at stride ${stride}, so it copies ${carried(s.reads![0])} from ${buf(s.reads![0].buf)} to ${buf(s.write!.buf)}.`, `${t} rảnh ở stride ${stride}, nên nó chép ${carried(s.reads![0])} từ ${buf(s.reads![0].buf)} sang ${buf(s.write!.buf)}.`);
    case 'swap':
      return tr(`${t} swaps its pointers: inBuffer_s is now ${buf(s.pointers!.inBuf)}, outBuffer_s ${buf(s.pointers!.outBuf)}.`, `${t} hoán đổi con trỏ: inBuffer_s giờ là ${buf(s.pointers!.inBuf)}, outBuffer_s là ${buf(s.pointers!.outBuf)}.`);
    case 'store':
      return tr(`${t} writes ${cfg.kind === 'double' ? `inBuffer_s[${s.thread}], that is ${buf(s.reads![0].buf)}[${s.thread}]` : `buffer_s[${s.thread}]`} = ${carried(s.reads![0])} to output[${s.thread}].`, `${t} ghi ${cfg.kind === 'double' ? `inBuffer_s[${s.thread}], tức ${buf(s.reads![0].buf)}[${s.thread}]` : `buffer_s[${s.thread}]`} = ${carried(s.reads![0])} ra output[${s.thread}].`);
    default:
      return '';
  }
}

const OP_WORD: Record<string, string> = {
  get load() { return tr('loaded', 'đã nạp'); },
  get barrier1() { return tr('past barrier 1', 'đã qua rào chắn 1'); },
  get barrier2() { return tr('past barrier 2', 'đã qua rào chắn 2'); },
  get read() { return tr('has read', 'đã đọc'); },
  get write() { return tr('has stored', 'đã ghi'); },
  get add() { return tr('has added', 'đã cộng'); },
  get copy() { return tr('has copied', 'đã chép'); },
  get swap() { return tr('has swapped', 'đã hoán đổi'); },
  get store() { return tr('finished', 'đã xong'); },
};

/**
 * Step through one run thread by thread: buttons, narration, the kernel with the current
 * line marked, the buffer drawn as rows, the threads, and the outcome. `trace` comes from
 * ksblock.run; a new trace starts again from step 0.
 */
export function TraceView({ trace, codeLabel = 'CUDA' }: { trace: Trace; codeLabel?: string }) {
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);
  const total = trace.steps.length;
  const at = Math.min(n, total);
  useEffect(() => {
    setN(0);
    setPlaying(false);
  }, [trace]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setN((x) => x + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);
  const st = useMemo(() => stateAt(trace, at), [trace, at]);
  const grid = useMemo(() => fromTrace(trace, at), [trace, at]);
  const src = useMemo(() => kernelSource(trace.cfg), [trace.cfg]);
  const cur: Step | undefined = at > 0 ? trace.steps[at - 1] : undefined;
  const bad = firstFailure(trace);
  const dbl = trace.cfg.kind === 'double';
  const lines = cur && cur.thread >= 0 ? src.line[cur.op as keyof typeof src.line] ?? [] : cur ? (src.line.barrier1 ?? src.line.barrier2 ?? []) : [];
  const done = at >= total;
  const doing = (t: number) => {
    const th = st.threads[t];
    if (th.done) return tr('finished', 'đã xong');
    if (th.waiting) return dbl ? tr('waiting at the barrier', 'đang chờ ở rào chắn') : tr(`waiting at barrier ${th.waiting}`, `đang chờ ở rào chắn ${th.waiting}`);
    if (!th.last) return tr('not started', 'chưa bắt đầu');
    return th.iter < 0 ? tr('loaded its element', 'đã nạp phần tử của mình') : tr(`iteration ${th.iter}: ${OP_WORD[th.last]}`, `lần lặp ${th.iter}: ${OP_WORD[th.last]}`);
  };
  return (
    <div data-part="trace-view" data-schedule={trace.schedule}>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setN(0); }} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => { setPlaying(false); setN(at - 1); }} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => { setPlaying(false); setN(at + 1); }} disabled={done}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => { setPlaying(false); setN(nextRelease(trace, at)); }} disabled={done}><Translated vi="Chạy tới rào chắn kế tiếp">Run to next barrier</Translated></button>
        <button onClick={() => setPlaying(!playing)} disabled={done}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Phát')}</button>
        <button onClick={() => { setPlaying(false); setN(total); }} disabled={done}><Translated vi="Chạy đến cuối">Run to the end</Translated></button>
        {bad ? <button onClick={() => { setPlaying(false); setN(bad); }}><Translated vi="Tới lỗi đầu tiên">Go to the first failure</Translated></button> : null}
        <span className="muted" aria-live="polite" data-readout="step"><Translated vi={<>bước {at} / {total}</>}>step {at} of {total}</Translated></span>
      </div>
      <div className={'narrate' + (cur?.violation ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }} data-readout="narrate">
        {cur ? (
          <>
            {describeStep(cur, trace)}
            {cur.violation ? (
              <>
                <br />
                <b className="t-fail"><Translated vi="Lỗi">Failure</Translated> ({KIND_LABEL[cur.violation.kind]}): {cur.violation.message}.</b>
              </>
            ) : null}
          </>
        ) : (
          <Translated vi={<>Chưa có gì chạy. {trace.cfg.threads} luồng, mỗi luồng một phần tử, scan x0 đến x{trace.cfg.threads - 1}. Bấm Bước.</>}>Nothing has run yet. {trace.cfg.threads} threads, one per element, scan x0 to x{trace.cfg.threads - 1}. Press Step.</Translated>
        )}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <KsGrid rows={grid.rows} labels={grid.labels} arrows={grid.arrows} marks={grid.marks} cellW={72}
          label={dbl ? tr('The two buffers, one row per iteration', 'Hai buffer, mỗi lần lặp một hàng') : tr('buffer_s, one row per iteration', 'buffer_s, mỗi lần lặp một hàng')} />
      </div>
      <Translated
        vi={
          <p className="muted" style={{ margin: '0.4rem 0 0.9rem' }}>
            Mỗi hàng là những gì một lần lặp đã ghi; ô nét đứt được giữ nguyên. Dưới mỗi giá trị: các đầu vào mà nó tính.
            Viền đậm đánh dấu chỗ bước hiện tại đọc, viền nét đứt là chỗ nó ghi; màu đỏ nghĩa là không phải giá trị mong muốn.
          </p>
        }
      >
        <p className="muted" style={{ margin: '0.4rem 0 0.9rem' }}>
          Each row is what one iteration wrote; dashed cells were left as they were. Under every value: the inputs it counts.
          A heavy outline marks what the current step reads, a dashed one what it writes; red means not the value intended.
        </p>
      </Translated>
      {done ? (
        <p style={{ margin: '0 0 0.9rem' }} data-readout="outcome" data-correct={trace.correct ? '1' : '0'}>
          {trace.correct ? (
            <span className="chip ok"><Translated vi="output đúng">output correct</Translated></span>
          ) : (
            <span className="chip fail"><Translated vi={<>{trace.wrong.length} trên {trace.cfg.threads} output sai</>}>{trace.wrong.length} of {trace.cfg.threads} outputs wrong</Translated></span>
          )}{' '}
          {trace.correct
            ? `output = ${trace.output.join(' ')}`
            : tr(`${trace.wrong.length > 1 ? 'positions' : 'position'} ${trace.wrong.join(', ')}: `, `vị trí ${trace.wrong.join(', ')}: `) + trace.wrong.map((i) => tr(`${val(trace.output[i])}${trace.outputCoef[i] ? ` (${rangeText(trace.outputCoef[i])})` : ''} instead of ${trace.reference[i]}`, `${val(trace.output[i])}${trace.outputCoef[i] ? ` (${rangeText(trace.outputCoef[i])})` : ''} thay vì ${trace.reference[i]}`)).join('; ')}
        </p>
      ) : null}
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 7fr) minmax(0, 4fr)' }}>
        <div style={{ minWidth: 0 }}>
          <Code tabs={[{ label: codeLabel, code: src.code }]} highlight={lines} />
        </div>
        <div style={{ minWidth: 0 }}>
          <table className="threads" aria-label={tr('Threads of the block', 'Các luồng của khối')}>
            <thead>
              <tr><th><Translated vi="luồng">thread</Translated></th><th><Translated vi="đang làm">doing</Translated></th>{dbl ? <th><Translated vi="buffer vào / ra">in / out buffer</Translated></th> : <th>temp{/* i18n-ignore: variable name */}</th>}</tr>
            </thead>
            <tbody>
              {st.threads.map((th, t) => (
                <tr key={t} className={cur?.thread === t ? 'cur' : ''}>
                  <td>{t}</td>
                  <td>{doing(t)}</td>
                  {dbl ? (
                    <td>{th.inBuf + 1} / {th.outBuf + 1}</td>
                  ) : (
                    <td className="t-reg">{th.temp ? `${val(th.temp.value)} (${rangeShort(th.temp.coef)})` : ''}</td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
