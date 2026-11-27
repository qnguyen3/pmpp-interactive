import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { BOOK_KEYS, bin, lsdSort, parallelPass, type SortPass } from '../../sim/radixsort';

// "Sorting one bit at a time" (section 14.4): the book's sixteen 4-bit keys through four
// stable 1-bit passes (Fig. 14.4), one unstable replay of pass 4, and the scan that gives
// every destination (Fig. 14.6). Six beats, one narration clip each
// (content/ch14/narration/lsd-passes-N.txt). Every row, arrow, bracket and number drawn
// comes from sim/radixsort.ts (lsdSort, parallelPass), pinned in radixsort.test.ts.
const RUN = lsdSort(BOOK_KEYS, 4, 1);
const BAD = lsdSort(BOOK_KEYS, 4, 1, { unstablePass: 4 });
const PP = parallelPass(BOOK_KEYS, 0, 4);
const P = RUN.passes;
/** The pair the third beat follows: 0101 (input 6) and 0100 (input 12), which differ only in bit 0. */
const PAIR = [6, 12];

const W = 720;
const H = 300;
/** A halo so that a label stays readable where arrows cross it. */
const HALO = { paintOrder: 'stroke', stroke: 'var(--page)', strokeWidth: 4, strokeLinejoin: 'round' } as const;
const AX = 24;
const CW = 42;
const N = BOOK_KEYS.length;
const TOP = 52;
const BOT = 184;

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const cx = (j: number) => AX + j * CW + (CW - 4) / 2;

/** One key: four bits, the bit of interest underlined, the low `shade` bits shaded. */
function Key({ x, y, k, mark, shade = 0, dim = false, hot = false, bad = false, row, slot, id }: {
  x: number; y: number; k: number; mark?: number; shade?: number; dim?: boolean; hot?: boolean; bad?: boolean;
  row: string; slot: number; id?: number;
}) {
  const s = bin(k, 4);
  return (
    <g data-row={row} data-slot={slot} data-key={s} data-id={id ?? ''} opacity={dim ? 0.3 : 1}>
      <rect x={x} y={y} width={CW - 4} height={24} rx="2"
        fill={bad ? 'var(--fail-wash)' : hot ? 'var(--ink)' : 'var(--cell)'}
        stroke={bad ? 'var(--fail)' : hot ? 'var(--ink)' : 'var(--rule)'} />
      {shade > 0 ? <rect x={x + 3 + (4 - shade) * 8} y={y + 3} width={shade * 8} height={18} rx="1" fill={hot ? 'none' : 'var(--rule)'} /> : null}
      {s.split('').map((ch, c) => {
        const isMark = mark != null && c === 3 - mark;
        return (
          <g key={c}>
            <text x={x + 7 + c * 8} y={y + 16.5} fontSize="12" textAnchor="middle" fontWeight={isMark ? 700 : 400}
              style={{ fill: hot ? 'var(--page)' : bad ? 'var(--fail)' : 'var(--ink)' }}>{ch}</text>
            {isMark ? <rect x={x + 3.5 + c * 8} y={y + 19.5} width={7} height={1.8} style={{ fill: hot ? 'var(--page)' : 'var(--ink)' }} /> : null}
          </g>
        );
      })}
    </g>
  );
}

function Row({ y, keys, ids, label, mark, shade, show = N, hot, dimExcept, badFrom, row }: {
  y: number; keys: number[]; ids: number[]; label: string; mark?: number; shade?: number; show?: number | boolean[];
  hot?: number[]; dimExcept?: number[]; badFrom?: number; row: string;
}) {
  return (
    <g>
      <text x={AX} y={y - 7} fontSize="11.5" fontWeight="600" style={HALO}>{label}</text>
      {keys.map((k, j) => {
        const visible = Array.isArray(show) ? show[j] : j < show;
        if (!visible)
          return <rect key={j} x={AX + j * CW} y={y} width={CW - 4} height={24} rx="2" fill="none" stroke="var(--rule)" strokeDasharray="3 3" />;
        return (
          <Key key={j} x={AX + j * CW} y={y} k={k} mark={mark} shade={shade} row={row} slot={j} id={ids[j]}
            hot={hot?.includes(ids[j])} dim={dimExcept ? !dimExcept.includes(ids[j]) : false} bad={badFrom != null && j >= badFrom} />
        );
      })}
    </g>
  );
}

/** Arrows from every input slot to its destination; `order` says which appear first, `show` how many. */
function Arrows({ pass, show, order, y1 = TOP + 26, y2 = BOT - 3, badBucket }: {
  pass: SortPass; show: number; order?: number[]; y1?: number; y2?: number; badBucket?: number;
}) {
  const seq = order ?? pass.dst.map((_, i) => i);
  return (
    <g>
      <defs>
        <marker id="lsd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute value */}
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--ink-2)' }} />
        </marker>
        <marker id="lsd-arrow-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute value */}
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--fail)' }} />
        </marker>
      </defs>
      {seq.map((i, k) => {
        const on = clamp(show * seq.length - k);
        if (on <= 0) return null;
        const bad = badBucket != null && ((pass.input[i] >>> pass.shift) & 1) === badBucket;
        return (
          <line key={i} data-arrow={i} data-to={pass.dst[i]} opacity={on}
            x1={cx(i)} y1={y1} x2={cx(pass.dst[i])} y2={y2}
            stroke={bad ? 'var(--fail)' : 'var(--ink-2)'} strokeWidth={bad ? 1.6 : 1.1}
            markerEnd={bad ? 'url(#lsd-arrow-bad)' : 'url(#lsd-arrow)'} />
        );
      })}
    </g>
  );
}

/** Brackets under a row: one per run of equal digit. */
function Runs({ y, groups, opacity = 1 }: { y: number; groups: { from: number; to: number; label: string }[]; opacity?: number }) {
  return (
    <g opacity={opacity} data-part="runs">
      {groups.map((g) => (
        <g key={g.from} data-run={g.label}>
          <line x1={AX + g.from * CW + 2} x2={AX + (g.to + 1) * CW - 6} y1={y} y2={y} stroke="var(--ink-2)" />
          <line x1={AX + g.from * CW + 2} x2={AX + g.from * CW + 2} y1={y - 4} y2={y} stroke="var(--ink-2)" />
          <line x1={AX + (g.to + 1) * CW - 6} x2={AX + (g.to + 1) * CW - 6} y1={y - 4} y2={y} stroke="var(--ink-2)" />
          <text x={AX + ((g.from + g.to + 1) * CW) / 2 - 2} y={y + 14} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{g.label}</text>
        </g>
      ))}
    </g>
  );
}
const bucketRuns = (pass: SortPass, name: (d: number) => string) =>
  pass.counts.flatMap((c, d) => (c ? [{ from: pass.starts[d], to: pass.starts[d] + c - 1, label: name(d) }] : []));

function Stage({ children, headline, sortedBy }: { children: ReactNode; headline?: ReactNode; sortedBy?: number | null }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Animation of four radix-sort passes over sixteen 4-bit keys', 'Hoạt hình bốn lượt radix sort trên mười sáu khóa 4 bit')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      {children}
      {sortedBy != null ? (
        <text x={W - 24} y={BOT + 76} fontSize="12" textAnchor="end" data-sorted-by={sortedBy} style={{ fill: 'var(--ink-2)' }}>
          {sortedBy === 4
            ? tr('sorted', 'đã sắp')
            : sortedBy === 0
              ? tr('not in order by its low bits', 'chưa theo thứ tự theo các bit thấp')
              : tr(`in order by its low ${['', 'bit', 'two bits', 'three bits'][sortedBy]}`, `đã theo thứ tự theo ${['', 'bit', 'hai bit', 'ba bit'][sortedBy]} thấp`)}
        </text>
      ) : null}
      {headline ? <text x={AX} y={H - 10} fontSize="13" fontWeight="600" data-headline="">{headline}</text> : null}
    </svg>
  );
}

/** One pass drawn as input row, arrows, output row; `p` from 0 to 1 animates the arrows. */
function PassView({ pass, p, a = 0.06, b = 0.8, order, headline, runs, runsAt = 0.85, label, badBucket, badFrom, final }: {
  pass: SortPass; p: number; a?: number; b?: number; order?: number[]; headline: ReactNode;
  runs?: { from: number; to: number; label: string }[]; runsAt?: number; label?: string; badBucket?: number; badFrom?: number; final?: number;
}) {
  const seq = order ?? pass.dst.map((_, i) => i);
  const t = ramp(p, a, b);
  const landed = Array(N).fill(false);
  seq.forEach((i, k) => {
    if (t * seq.length - k >= 1) landed[pass.dst[i]] = true;
  });
  const done = t >= 1;
  return (
    <Stage headline={headline} sortedBy={done ? (final ?? pass.sortedBy) : null}>
      <Row y={TOP} keys={pass.input} ids={pass.inIds} mark={pass.shift} row="top" // i18n-ignore: data-row key
        label={pass.index === 1 ? tr('input, bit 0 marked', 'đầu vào, đánh dấu bit 0') : tr(`after pass ${pass.index - 1}, bit ${pass.shift} marked`, `sau lượt ${pass.index - 1}, đánh dấu bit ${pass.shift}`)} />
      <Arrows pass={pass} show={t} order={order} badBucket={badBucket} />
      <Row y={BOT} keys={pass.output} ids={pass.outIds} row="bottom" show={landed} // i18n-ignore: data-row key
        shade={done ? (final ?? pass.sortedBy) : 0} badFrom={done ? badFrom : undefined}
        label={label ?? tr(`after pass ${pass.index}`, `sau lượt ${pass.index}`)} />
      {runs ? <Runs y={BOT + 34} groups={runs} opacity={ramp(p, runsAt, runsAt + 0.08)} /> : null}
    </Stage>
  );
}

const zerosFirst = (pass: SortPass) => {
  const idx = pass.input.map((_, i) => i);
  const bit = (i: number) => (pass.input[i] >>> pass.shift) & 1;
  return [...idx.filter((i) => bit(i) === 0), ...idx.filter((i) => bit(i) === 1)];
};

/** Beat 3: three rows with the pair joined across them. */
function Inherit({ p }: { p: number }) {
  const rows = [
    { keys: BOOK_KEYS, ids: BOOK_KEYS.map((_, i) => i), label: tr('input', 'đầu vào'), mark: 0 },
    { keys: P[0].output, ids: P[0].outIds, label: tr('after pass 1 (bit 0)', 'sau lượt 1 (bit 0)'), mark: 0 },
    { keys: P[1].output, ids: P[1].outIds, label: tr('after pass 2 (bit 1)', 'sau lượt 2 (bit 1)'), mark: 1 },
  ];
  const ys = [40, 120, 200];
  const step = p < 0.3 ? 0 : p < 0.62 ? 1 : 2;
  const pos = (r: number, id: number) => rows[r].ids.indexOf(id);
  const head = [
    tr('in the input, 0101 comes before 0100', 'trong đầu vào, 0101 đứng trước 0100'),
    tr('pass 1: their bit 0 differs, so 0100 goes ahead', 'lượt 1: bit 0 của chúng khác nhau, nên 0100 lên trước'),
    tr('pass 2: their bit 1 is the same, so the stable pass keeps 0100 ahead', 'lượt 2: bit 1 của chúng giống nhau, nên lượt ổn định giữ 0100 ở trước'),
  ][step];
  return (
    <Stage headline={head}>
      {rows.map((r, k) =>
        k <= step ? (
          <Row key={k} y={ys[k]} keys={r.keys} ids={r.ids} label={r.label} mark={k < 2 ? 0 : 1} row={`r${k}`} hot={PAIR} dimExcept={PAIR} />
        ) : null,
      )}
      {PAIR.map((id) =>
        [0, 1].map((k) =>
          k < step ? (
            <line key={`${id}-${k}`} data-pair={id} x1={cx(pos(k, id))} y1={ys[k] + 26} x2={cx(pos(k + 1, id))} y2={ys[k + 1] - 3}
              stroke="var(--ink)" strokeWidth="1.6" />
          ) : null,
        ),
      )}
      <text x={W - 24} y={ys[step] + 44} fontSize="11.5" textAnchor="end" data-pair-pos={`${pos(step, 12)},${pos(step, 6)}`} style={{ fill: 'var(--ink-2)' }}>
        <Translated vi={<>0100 ở {pos(step, 12)}, 0101 ở {pos(step, 6)}</>}>0100 at {pos(step, 12)}, 0101 at {pos(step, 6)}</Translated>
      </text>
    </Stage>
  );
}

/** Beat 6: Fig. 14.6's rows for pass 1, growing one by one. */
function ScanRows({ p }: { p: number }) {
  const L = 108;
  const x = (j: number) => L + j * 36;
  const rows: { name: string; vals: (string | number)[]; at: number }[] = [
    { name: tr('key', 'khóa'), vals: PP.keys.map((k) => bin(k, 4)), at: 0 },
    { name: tr('bit 0', 'bit 0'), vals: PP.bits, at: 0.12 },
    { name: tr('ones before', 'số bit 1 phía trước'), vals: PP.threads.map((t) => t.onesBefore), at: 0.32 },
    { name: tr('destination', 'đích đến'), vals: PP.threads.map((t) => t.dst!), at: 0.55 },
  ];
  return (
    <Stage
      headline={
        p < 0.75
          ? tr('one exclusive scan of the bits counts the ones before every key', 'một scan loại trừ trên các bit đếm số bit 1 đứng trước mỗi khóa')
          : tr('zeros: index − ones before; ones: n − total + ones before (section 14.5)', 'bit 0: index − số bit 1 phía trước; bit 1: n − tổng + số bit 1 phía trước (mục 14.5)')
      }
    >
      {rows.map((r, k) => (
        <g key={r.name} opacity={ramp(p, r.at, r.at + 0.08)} data-scan-row={r.name}>
          <text x={AX} y={58 + k * 44} fontSize="11.5" fontWeight="600">{r.name}</text>
          {r.vals.map((v, j) => (
            <g key={j}>
              <rect x={x(j)} y={40 + k * 44} width={32} height={26} rx="2" fill="var(--cell)" stroke="var(--rule)" />
              <text x={x(j) + 16} y={57.5 + k * 44} fontSize={k === 0 ? 10 : 12} textAnchor="middle" data-cell={`${r.name}:${j}`}>{v}</text>
            </g>
          ))}
        </g>
      ))}
      <g opacity={ramp(p, 0.75, 0.85)}>
        <text x={L} y={238} fontSize="12">
          <Translated vi={<>tổng số bit 1: {PP.ones}, mục 16 của scan (16 bit và một số 0 sau chúng)</>}>
            total ones: {PP.ones}, the scan's entry 16 (the 16 bits and a zero after them)
          </Translated>
        </text>
        <text x={L} y={262} fontSize="12">
          <Translated vi="một phép phân hoạch ổn định: bộ lọc ổn định của chương 12, với danh sách đầu ra thứ hai cho các bit 1">
            a stable partition: chapter 12's stable filter, with a second output list for the ones
          </Translated>
        </text>
      </g>
    </Stage>
  );
}

const beats: Beat[] = [
  {
    clip: 'lsd-passes-1',
    get label() {
      return tr('Pass 1', 'Lượt 1');
    },
    draw: (p) => (
      <PassView pass={P[0]} p={p} a={0.08} b={0.86} order={zerosFirst(P[0])} runsAt={0.88}
        runs={bucketRuns(P[0], (d) => `bit 0 = ${d}`)}
        headline={p < 0.5 ? tr('pass 1: keys whose bit 0 is 0 go to the front, in the order they come', 'lượt 1: các khóa có bit 0 bằng 0 lên đầu, theo thứ tự chúng đến') : tr('keys whose bit 0 is 1 go to the back, also in order', 'các khóa có bit 0 bằng 1 về cuối, cũng theo thứ tự')} />
    ),
  },
  {
    clip: 'lsd-passes-2',
    get label() {
      return tr('Pass 2', 'Lượt 2');
    },
    draw: (p) => (
      <PassView pass={P[1]} p={p} a={0.05} b={0.6} runsAt={0.66}
        runs={[0, 1, 2, 3].map((d) => ({ from: d * 4, to: d * 4 + 3, label: tr(`ends in ${bin(d, 2)}`, `tận cùng ${bin(d, 2)}`) }))}
        headline={p < 0.62 ? tr('pass 2 uses bit 1 and reorders nothing inside a group', 'lượt 2 dùng bit 1 và không đảo thứ tự gì bên trong một nhóm') : tr('in order by the low two bits: runs ending 00, 01, 10, 11', 'theo thứ tự theo hai bit thấp: các dải tận cùng 00, 01, 10, 11')} />
    ),
  },
  {
    clip: 'lsd-passes-3',
    get label() {
      return tr('One pass inherits', 'Một lượt thừa hưởng');
    },
    draw: (p) => <Inherit p={p} />,
  },
  {
    clip: 'lsd-passes-4',
    get label() {
      return tr('Passes 3 and 4', 'Lượt 3 và 4');
    },
    draw: (p) =>
      p < 0.5 ? (
        <PassView pass={P[2]} p={p * 2} a={0.05} b={0.75} headline={tr('pass 3 uses bit 2: in order by the low three bits', 'lượt 3 dùng bit 2: theo thứ tự theo ba bit thấp')} />
      ) : (
        <PassView pass={P[3]} p={(p - 0.5) * 2} a={0.05} b={0.75} runsAt={0.8}
          runs={bucketRuns(P[3], (d) => tr(`${P[3].counts[d]} keys with bit 3 = ${d}`, `${P[3].counts[d]} khóa có bit 3 = ${d}`))}
          headline={p < 0.9 ? tr('pass 4 uses bit 3: seven zeros in front of nine ones', 'lượt 4 dùng bit 3: bảy số 0 đứng trước chín số 1') : tr('four stable passes: sorted', 'bốn lượt ổn định: đã sắp')} />
      ),
  },
  {
    clip: 'lsd-passes-5',
    get label() {
      return tr('One unstable pass', 'Một lượt không ổn định');
    },
    draw: (p) => (
      <PassView pass={BAD.passes[3]} p={p} a={0.06} b={0.7} badBucket={1} badFrom={7} final={BAD.passes[3].sortedBy}
        label={tr('after pass 4, its ones bucket filled backwards', 'sau lượt 4, bucket bit 1 của nó được lấp ngược')}
        headline={p < 0.75 ? tr('pass 4 again, with the ones bucket filled backwards', 'lại lượt 4, với bucket bit 1 được lấp ngược') : tr(`not sorted: ${bin(BAD.result[7], 4)} lands before ${bin(BAD.result[8], 4)}`, `chưa sắp xong: ${bin(BAD.result[7], 4)} rơi vào trước ${bin(BAD.result[8], 4)}`)} />
    ),
  },
  {
    clip: 'lsd-passes-6',
    get label() {
      return tr('A stable partition', 'Một phép phân hoạch ổn định');
    },
    draw: (p) => <ScanRows p={p} />,
  },
];

export function LsdPassesScene() {
  return <Scene chapter="ch14" title={tr('Sorting one bit at a time', 'Sắp xếp từng bit một')} beats={beats} />;
}
