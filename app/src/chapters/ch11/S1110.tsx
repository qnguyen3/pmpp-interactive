import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  EX6,
  FIG116,
  FIG_11_20_PRINTED,
  additions,
  brentKung,
  closedForm,
  finalAfter,
  koggeStone,
  needsChain,
  printedDiffers,
  rangeLabel,
  ranges,
  readWriteOverlap,
  reverseLevels,
  run,
  span,
  upSteps,
  warpView,
  type Network,
  type Range,
} from '../../sim/prefixnet';
import { widthChoices } from '../../sim/warps';
import { BkSweepsScene } from './BkSweepsScene';

const SNIPPET = 'ch11/snippets/s1110_brent_kung.cu';
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const WRAP: CSSProperties = { flexWrap: 'wrap' };
const SHADE = [0, 0.2, 0.45, 0.75];
const list = (xs: (string | number)[]) =>
  xs.length <= 1 ? xs.join('') : tr(`${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`, `${xs.slice(0, -1).join(', ')} và ${xs[xs.length - 1]}`);
/** "2 to 13" for a run of consecutive positions, else a list. */
const span2 = (xs: number[]) =>
  xs.length > 2 && xs[xs.length - 1] - xs[0] === xs.length - 1 ? tr(`${xs[0]} to ${xs[xs.length - 1]}`, `${xs[0]} đến ${xs[xs.length - 1]}`) : list(xs);
const TIERS: Record<string, Tier> = { input: 'global', output: 'global', buffer_s: 'shared', i: 'reg', idx: 'reg', stride: 'reg' };
const INPUTS = [
  {
    key: 8,
    get label() {
      return tr('8: exercise 6', '8: bài tập 6');
    },
    values: EX6,
  },
  {
    key: 16,
    get label() {
      return tr('16: Fig. 11.6\'s values', '16: các giá trị của Fig. 11.6');
    },
    values: FIG116,
  },
];

// ---------------------------------------------------------------- a network, drawn
function NetSvg({
  net, k, rows, labels, finals, name, onCell, picked,
}: {
  net: Network;
  /** Steps done. */
  k: number;
  /** Row slots to reserve, so two networks line up step for step. */
  rows: number;
  labels: string[][];
  finals: Set<number>;
  name: string;
  onCell?: (i: number) => void;
  picked?: number | null;
}) {
  const n = net.n;
  const cw = n > 8 ? 32 : 52;
  const X0 = 56;
  const x = (i: number) => X0 + i * cw + cw / 2;
  const DY = 26;
  const y = (j: number) => 46 + j * DY;
  const cellY = y(rows) - 4;
  const width = X0 + n * cw + 8;
  const height = cellY + 38;
  const changed = new Set(k > 0 ? net.steps[k - 1].adders.map((a) => a.dst) : []);
  const two = labels.some((l) => l.length > 1);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={tr(`${name} network`, `Mạng ${name}`)} data-part="net" data-net={net.id} data-k={k}
      style={{ width: '100%', maxWidth: width * 1.45, minWidth: n > 8 ? 500 : 340, display: 'block' }}>
      {Array.from({ length: n }, (_, i) => (
        <g key={i}>
          <text x={x(i)} y={16} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{i}</text>
          <line x1={x(i)} y1={22} x2={x(i)} y2={cellY - 4} stroke="var(--rule)" />
        </g>
      ))}
      {Array.from({ length: rows }, (_, j) => {
        const st = net.steps[j];
        const label = st ? (st.phase === 'up' ? tr(`up ${j + 1}`, `lên ${j + 1}`) : st.phase === 'down' ? tr(`down ${st.level}`, `xuống ${st.level}`) : tr(`stride ${st.stride}`, `bước nhảy ${st.stride}`)) : '';
        const op = !st ? 0 : j < k - 1 ? 0.5 : j === k - 1 ? 1 : 0.14;
        return (
          <g key={j} opacity={st ? 1 : 0}>
            <text x={4} y={y(j) + 4} fontSize="10" style={{ fill: j === k - 1 ? 'var(--ink)' : 'var(--ink-3)', fontWeight: j === k - 1 ? 700 : 400 }}>{label}</text>
            <g opacity={op} data-part="row" data-step={j + 1} data-state={j < k - 1 ? 'done' : j === k - 1 ? 'current' : 'future'}>
              {st?.adders.map((a) => (
                <g key={a.dst}>
                  <line x1={x(a.src)} y1={y(j) - 10} x2={x(a.dst) - 5} y2={y(j) - 3} stroke="var(--ink)" strokeWidth={j === k - 1 ? 1.6 : 1} />
                  <circle cx={x(a.src)} cy={y(j) - 10} r={1.8} fill="var(--ink)" />
                  <circle cx={x(a.dst)} cy={y(j)} r={6.5} fill="var(--cell)" stroke="var(--ink)" strokeWidth={j === k - 1 ? 1.7 : 1} />
                  <text x={x(a.dst)} y={y(j) + 3.3} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink)' }}>+</text>
                </g>
              ))}
            </g>
          </g>
        );
      })}
      {k >= net.steps.length && net.steps.length < rows ? (
        <text x={X0 + 4} y={y(net.steps.length) + 4} fontSize="10.5" style={{ fill: 'var(--ok)' }} data-part="finished">
          <Translated vi={<>xong sau {net.steps.length} bước</>}>finished after {net.steps.length} steps</Translated>
        </text>
      ) : null}
      {labels.map((l, i) => {
        const fin = finals.has(i);
        return (
          <g key={i} data-part="cell" data-pos={i} data-label={l.join('')} data-final={fin ? 1 : 0}
            role={onCell ? 'button' : undefined} tabIndex={onCell ? 0 : undefined}
            aria-label={onCell ? tr(`position ${i}`, `vị trí ${i}`) : undefined}
            onMouseEnter={onCell ? () => onCell(i) : undefined} onFocus={onCell ? () => onCell(i) : undefined} onClick={onCell ? () => onCell(i) : undefined}
            style={onCell ? { cursor: 'pointer' } : undefined}>
            <rect x={x(i) - cw / 2 + 2} y={cellY} width={cw - 4} height={two ? 30 : 22} rx={3}
              fill={fin ? 'var(--ok-wash)' : 'var(--cell)'}
              stroke={picked === i ? 'var(--ink)' : fin ? 'var(--ok)' : changed.has(i) ? 'var(--ink)' : 'var(--rule)'}
              strokeWidth={changed.has(i) || picked === i ? 1.8 : 1} />
            {l.map((t, r) => (
              <text key={r} x={x(i)} y={cellY + (two ? 12.5 + r * 11 : 15)} fontSize={n > 8 ? 9 : 11} textAnchor="middle" style={{ fill: 'var(--ink)' }}>{t}</text>
            ))}
          </g>
        );
      })}
    </svg>
  );
}

const twoLines = (r: Range) => (r.lo === r.hi ? [`x${r.lo}`] : [`x${r.lo}..`, `x${r.hi}`]);

// ---------------------------------------------------------------- Fig. 11.20 as a heat map
function NeedsMap({ n, k, printed, picked, onPick }: { n: number; k: number; printed: boolean; picked: { row: number; pos: number } | null; onPick: (row: number, pos: number) => void }) {
  const locale = useLocale();
  const levels = useMemo(() => reverseLevels(n), [n, locale]);
  const diffs = useMemo(() => (n === 16 ? printedDiffers() : []), [n]);
  const isDiff = (row: number, col: number) => diffs.some((d) => d.row === row && d.col === col);
  const cw = n > 8 ? 34 : 52;
  const X0 = 58;
  const RH = 28;
  const width = X0 + n * cw + 6;
  const height = 22 + levels.length * RH + 4;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={tr('What each position still needs', 'Mỗi vị trí còn thiếu gì')} data-part="needs-map" data-shading={printed ? 'printed' : 'corrected'}
      style={{ width: '100%', maxWidth: width * 1.2, minWidth: n > 8 ? 540 : 340, display: 'block' }}>
      {Array.from({ length: n }, (_, i) => (
        <text key={i} x={X0 + i * cw + cw / 2} y={14} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{i}</text>
      ))}
      {levels.map((lv, row) => {
        const current = k === lv.step;
        return (
          <g key={row} data-part="needs-row" data-row={row} data-label={lv.label} data-current={current ? 1 : 0}>
            <text x={4} y={22 + row * RH + 17} fontSize="10.5" style={{ fill: current ? 'var(--ink)' : 'var(--ink-2)', fontWeight: current ? 700 : 400 }}><Translated vi={lv.label.replace('Initial', 'Ban đầu').replace('Level', 'Mức')}>{lv.label}</Translated></text>
            {lv.ranges.map((r, col) => {
              const v = printed && n === 16 ? FIG_11_20_PRINTED[row][col] : lv.needs[col];
              const show = row === 0 || lv.changed.includes(col);
              const dark = v >= 2;
              const sel = picked?.row === row && picked.pos === col;
              return (
                <g key={col} data-part="need-cell" data-row={row} data-pos={col} data-shade={v} data-needs={lv.needs[col]} data-differs={isDiff(row, col) ? 1 : 0}
                  role="button" tabIndex={0} aria-label={tr(`${lv.label}, position ${col}`, `${lv.label.replace('Initial', 'Ban đầu').replace('Level', 'Mức')}, vị trí ${col}`)} style={{ cursor: 'pointer' }}
                  onMouseEnter={() => onPick(row, col)} onFocus={() => onPick(row, col)} onClick={() => onPick(row, col)}>
                  <rect x={X0 + col * cw + 1} y={22 + row * RH} width={cw - 2} height={RH - 3} fill="var(--ink)" fillOpacity={SHADE[Math.min(3, v)]}
                    stroke={sel ? 'var(--ink)' : isDiff(row, col) ? 'var(--fail)' : current ? 'var(--ink-2)' : 'var(--rule)'}
                    strokeWidth={sel ? 2 : isDiff(row, col) ? 1.6 : 1} strokeDasharray={isDiff(row, col) && !sel ? '3 2' : undefined} />
                  {show ? (
                    <text x={X0 + col * cw + cw / 2} y={22 + row * RH + 16} fontSize={n > 8 ? 8 : 10} textAnchor="middle"
                      style={{ fill: dark ? 'var(--page)' : 'var(--ink)' }}>{rangeLabel(r)}</text>
                  ) : null}
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------- interactive 1: the stepper
function Stepper() {
  const [n, setN] = useState(16);
  const [show, setShow] = useState<'values' | 'ranges'>('ranges');
  const [step, setK] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [printed, setPrinted] = useState(false);
  const [picked, setPicked] = useState<{ row: number; pos: number } | null>(null);
  const values = (INPUTS.find((x) => x.key === n) ?? INPUTS[1]).values;
  const bk = useMemo(() => brentKung(n), [n]);
  const ks = useMemo(() => koggeStone(n), [n]);
  const bkV = useMemo(() => run(bk, values), [bk, values]);
  const ksV = useMemo(() => run(ks, values), [ks, values]);
  const bkR = useMemo(() => ranges(bk), [bk]);
  const ksR = useMemo(() => ranges(ks), [ks]);
  const locale = useLocale();
  const levels = useMemo(() => reverseLevels(n), [n, locale]);
  const total = bk.steps.length;
  const k = Math.min(step, total);
  const up = upSteps(bk);
  const kk = Math.min(k, ks.steps.length);

  useEffect(() => {
    if (!playing) return;
    if (k >= total) { setPlaying(false); return; }
    const id = window.setTimeout(() => setK((x) => x + 1), 900);
    return () => window.clearTimeout(id);
  }, [playing, k, total]);

  const labelsOf = (vals: number[], rs: Range[]) => (show === 'values' ? vals.map((v) => [String(v)]) : rs.map(twoLines));
  const addsSoFar = (net: Network, upto: number) => net.steps.slice(0, upto).reduce((s, x) => s + x.adders.length, 0);
  const bkStep = k > 0 ? bk.steps[k - 1] : null;
  const ksStep = k > 0 && k <= ks.steps.length ? ks.steps[k - 1] : null;
  const bkOverlap = bkStep ? readWriteOverlap(bkStep) : [];
  const ksOverlap = ksStep ? readWriteOverlap(ksStep) : [];

  const describeBK = () => {
    if (!bkStep) return tr(`Nothing has run. Each position holds its own input. Press Step.`, `Chưa có gì chạy. Mỗi vị trí giữ đầu vào của chính nó. Hãy nhấn Bước.`);
    const pairs = bkStep.adders.map((a) => tr(`${a.src} into ${a.dst}`, `${a.src} vào ${a.dst}`));
    const what = bkStep.phase === 'up'
      ? tr(`Reduction tree, step ${k}: positions of the form ${2 * bkStep.stride}n − 1 add the position ${bkStep.stride} to their left`, `Cây rút gọn, bước ${k}: các vị trí có dạng ${2 * bkStep.stride}n − 1 cộng vị trí cách chúng ${bkStep.stride} về bên trái`)
      : tr(`Reverse tree, level ${bkStep.level}: each partial sum moves ${bkStep.stride} ${bkStep.stride === 1 ? 'place' : 'places'} right`, `Cây đảo ngược, mức ${bkStep.level}: mỗi tổng riêng phần dời ${bkStep.stride} vị trí sang phải`);
    const fin = finalAfter(bk, k);
    return tr(`${what} (${bkStep.adders.length > 4 ? `${bkStep.adders.length} additions` : pairs.join(', ')}). Final now: ${span2(fin)}.`, `${what} (${bkStep.adders.length > 4 ? `${bkStep.adders.length} phép cộng` : pairs.join(', ')}). Đã xong: ${span2(fin)}.`);
  };

  // The readout for a picked cell of the needs map.
  const pickedText = (() => {
    if (!picked || picked.row >= levels.length || picked.pos >= n) return null;
    const lv = levels[picked.row];
    const chain = needsChain(lv.ranges, picked.pos);
    const own = rangeLabel(lv.ranges[picked.pos]);
    const pr = n === 16 ? FIG_11_20_PRINTED[picked.row][picked.pos] : null;
    const tail = pr != null && pr !== chain.length
      ? tr(` The book shades this cell for ${pr}, one level more than its ranges imply.`, ` Sách tô ô này cho ${pr}, đậm hơn một mức so với những gì các khoảng của nó cho thấy.`)
      : '';
    if (!chain.length) return tr(`${lv.label}, position ${picked.pos}: holds ${own}, which is final.${tail}`, `${lv.label.replace('Initial', 'Ban đầu').replace('Level', 'Mức')}, vị trí ${picked.pos}: giữ ${own}, đã là giá trị cuối.${tail}`);
    return tr(`${lv.label}, position ${picked.pos}: holds ${own} and still needs ${chain.length} partial ${chain.length === 1 ? 'sum' : 'sums'}: ${list(chain.map((c) => `${c} (${rangeLabel(lv.ranges[c])})`))}.${tail}`, `${lv.label.replace('Initial', 'Ban đầu').replace('Level', 'Mức')}, vị trí ${picked.pos}: giữ ${own} và vẫn cần ${chain.length} tổng riêng phần: ${list(chain.map((c) => `${c} (${rangeLabel(lv.ranges[c])})`))}.${tail}`);
  })();

  return (
    <Bench
      title={tr('Up-sweep, down-sweep', 'Quét lên, quét xuống')}
      task={
        <Translated vi="Chạy từng bước cả hai mạng trên cùng một đầu vào. Những vị trí nào đã xong khi cây rút gọn kết thúc, và mỗi mạng mất bao nhiêu bước?">
          Step both networks on the same input. Which positions are final when the reduction tree ends, and how many steps does each network take?
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Thời gian chạy xuống dưới, mỗi bước một hàng bộ cộng; một chấm đánh dấu nơi một giá trị được đọc, một vòng tròn là
              vị trí cộng nó vào. Ô xanh giữ giá trị cuối cùng của nó. Bảng ở dưới cùng là Fig. 11.20: ô càng tô đậm, vị trí
              của nó càng cần nhiều tổng riêng phần (không cần, rồi 1, 2 và 3; sách in trắng, xám nhạt, xám đậm và đen). Trỏ
              vào một ô để liệt kê chúng.
            </>
          }
        >
          <>
            Time runs down, one row of adders per step; a dot marks where a value is read, a circle the position that adds it.
            A green cell holds its final value. The table at the bottom is Fig. 11.20: the stronger a cell's shading, the
            more partial sums its position still needs (none, then 1, 2 and 3; the book prints white, light grey, dark grey
            and black). Point at a cell to list them.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Đầu vào">Input</Translated>
          <span className="seg" role="group" aria-label={tr('Input', 'Đầu vào')} style={WRAP}>
            {INPUTS.map((x) => (
              <button key={x.key} aria-pressed={n === x.key} onClick={() => { setN(x.key); setK(0); setPicked(null); setPlaying(false); }}>{x.label}</button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          <Translated vi="Ô hiển thị">Cells show</Translated>
          <span className="seg" role="group" aria-label={tr('Cells show', 'Ô hiển thị')}>
            {(['ranges', 'values'] as const).map((s) => <button key={s} aria-pressed={show === s} onClick={() => setShow(s)}><Translated vi={s === 'ranges' ? 'khoảng' : 'giá trị'}>{s}</Translated></button>)}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={() => { setPlaying(false); setK(0); }} disabled={k === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => { setPlaying(false); setK(k - 1); }} disabled={k === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => { setPlaying(false); setK(k + 1); }} disabled={k >= total}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => setPlaying(!playing)} disabled={k >= total}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Phát')}</button>
        <span className="muted" aria-live="polite" data-readout="step"><Translated vi={<>bước {k} trên {total}</>}>step {k} of {total}</Translated></span>
      </div>
      <div className="narrate" role="status" data-readout="narrate" style={{ marginBottom: '0.8rem' }}>
        <b>Brent-Kung.{/* i18n-ignore: algorithm name */}</b> {describeBK()}{' '}
        <b>Kogge-Stone.{/* i18n-ignore: algorithm name */}</b>{' '}
        {ksStep
          ? tr(`Stride ${ksStep.stride}: every position from ${ksStep.stride} on adds the one ${ksStep.stride} to its left (${ksStep.adders.length} additions).`, `Bước nhảy ${ksStep.stride}: mọi vị trí từ ${ksStep.stride} trở đi cộng vị trí cách nó ${ksStep.stride} về bên trái (${ksStep.adders.length} phép cộng).`)
          : k === 0 ? tr('Not started.', 'Chưa bắt đầu.') : tr(`Finished after ${ks.steps.length} steps.`, `Xong sau ${ks.steps.length} bước.`)}
      </div>
      <div className="cols" style={{ gridTemplateColumns: n > 8 ? 'minmax(0, 1fr)' : undefined }}>
        <div className="panel" data-panel="brent-kung">
          <Translated
            vi={
              <p style={{ margin: '0 0 0.4rem' }}>
                <b>Brent-Kung</b>: <span data-counter="bk-adds">{addsSoFar(bk, k)}</span> phép cộng trong{' '}
                <span data-counter="bk-steps">{k}</span> bước
                {k >= total ? <span className="chip ok" style={{ marginLeft: 6 }}>xong</span> : k === up ? <span className="chip" style={{ marginLeft: 6 }}>cây rút gọn xong</span> : null}
              </p>
            }
          >
            <p style={{ margin: '0 0 0.4rem' }}>
              <b>Brent-Kung</b>: <span data-counter="bk-adds">{addsSoFar(bk, k)}</span> additions in{' '}
              <span data-counter="bk-steps">{k}</span> steps
              {k >= total ? <span className="chip ok" style={{ marginLeft: 6 }}>done</span> : k === up ? <span className="chip" style={{ marginLeft: 6 }}>reduction tree done</span> : null}
            </p>
          </Translated>
          <div style={{ overflowX: 'auto' }}>
          <NetSvg net={bk} k={k} rows={total} name={tr('Brent-Kung', 'Brent-Kung')} labels={labelsOf(bkV[k], bkR[k])} finals={new Set(finalAfter(bk, k))}
            onCell={(i) => { if (k >= up) setPicked({ row: Math.min(levels.length - 1, k - up), pos: i }); }}
            picked={picked && levels[picked.row]?.step === k ? picked.pos : null} />
          </div>
        </div>
        <div className="panel" data-panel="kogge-stone">
          <Translated
            vi={
              <p style={{ margin: '0 0 0.4rem' }}>
                <b>Kogge-Stone</b>: <span data-counter="ks-adds">{addsSoFar(ks, kk)}</span> phép cộng trong{' '}
                <span data-counter="ks-steps">{kk}</span> bước
                {kk >= ks.steps.length ? <span className="chip ok" style={{ marginLeft: 6 }}>xong</span> : null}
              </p>
            }
          >
            <p style={{ margin: '0 0 0.4rem' }}>
              <b>Kogge-Stone</b>: <span data-counter="ks-adds">{addsSoFar(ks, kk)}</span> additions in{' '}
              <span data-counter="ks-steps">{kk}</span> steps
              {kk >= ks.steps.length ? <span className="chip ok" style={{ marginLeft: 6 }}>done</span> : null}
            </p>
          </Translated>
          <div style={{ overflowX: 'auto' }}>
            <NetSvg net={ks} k={kk} rows={total} name={tr('Kogge-Stone', 'Kogge-Stone')} labels={labelsOf(ksV[kk], ksR[kk])} finals={new Set(finalAfter(ks, kk))} />
          </div>
        </div>
      </div>
      <Translated
        vi={
          <p style={{ margin: '0.8rem 0' }} data-readout="overlap" data-bk={bkOverlap.length} data-ks={ksOverlap.length}>
            Đọc và ghi trong bước này:{' '}
            {bkStep ? (
              <>Brent-Kung: {bkOverlap.length ? `ghi đè ${span2(bkOverlap)} trong khi một bộ cộng khác đọc chúng` : <span className="chip ok">không vị trí nào do một bộ cộng ghi lại bị bộ cộng khác đọc</span>}{' '}</>
            ) : null}
            {ksStep ? (
              <>Kogge-Stone: {ksOverlap.length ? <span className="chip fail">ghi các vị trí {span2(ksOverlap)}, mà các bộ cộng khác cùng bước đọc</span> : 'không ghi gì mà một bộ cộng khác của bước này đọc.'}</>
            ) : null}
            {!bkStep && !ksStep ? 'chưa có gì.' : null}
          </p>
        }
      >
        <p style={{ margin: '0.8rem 0' }} data-readout="overlap" data-bk={bkOverlap.length} data-ks={ksOverlap.length}>
          Read and write in this step:{' '}
          {bkStep ? (
            <>Brent-Kung: {bkOverlap.length ? `overwrites ${span2(bkOverlap)} while another adder reads them` : <span className="chip ok">no position written by one adder is read by another</span>}{' '}</>
          ) : null}
          {ksStep ? (
            <>Kogge-Stone: {ksOverlap.length ? <span className="chip fail">writes positions {span2(ksOverlap)}, which other adders of the same step read</span> : 'writes nothing another adder of this step reads.'}</>
          ) : null}
          {!bkStep && !ksStep ? 'nothing yet.' : null}
        </p>
      </Translated>
      <div className="panel" data-panel="needs">
        <div className="row" style={{ marginBottom: '0.5rem', justifyContent: 'space-between' }}>
          <b><Translated vi="Mỗi vị trí còn thiếu gì (Fig. 11.20)">What each position still needs (Fig. 11.20)</Translated></b>
          {n === 16 ? (
            <span style={FIELD}>
              <Translated vi="Cách tô">Shading</Translated>
              <span className="seg" role="group" aria-label={tr('Shading', 'Cách tô')}>
                <button aria-pressed={!printed} onClick={() => setPrinted(false)}><Translated vi="từ các khoảng">from the ranges</Translated></button>
                <button aria-pressed={printed} onClick={() => setPrinted(true)}><Translated vi="như in trong sách">as printed</Translated></button>
              </span>
            </span>
          ) : null}
        </div>
        <div style={{ overflowX: 'auto' }}>
          <NeedsMap n={n} k={k} printed={printed} picked={picked} onPick={(row, pos) => setPicked({ row, pos })} />
        </div>
        <p className="muted" style={{ margin: '0.5rem 0 0' }} aria-live="polite" data-readout="needs">
          {pickedText ?? (k < up
            ? tr(`The table starts when the reduction tree ends, after step ${up}. Point at a cell.`, `Bảng bắt đầu khi cây rút gọn kết thúc, sau bước ${up}. Hãy trỏ vào một ô.`)
            : tr('Point at a cell, or at a cell of the Brent-Kung network, to list the partial sums it still needs.', 'Trỏ vào một ô, hoặc một ô của mạng Brent-Kung, để liệt kê các tổng riêng phần nó còn cần.'))}
        </p>
        {n === 16 ? (
          <p className="muted" style={{ margin: '0.3rem 0 0' }}>
            <Translated vi="Viền đỏ nét đứt: ô mà sách tô đậm hơn một mức so với những gì các khoảng của nó cho thấy.">
              Dashed red outline: a cell the book shades one level stronger than its ranges imply.
            </Translated>
          </p>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive 2: inside one warp
function LaneGrid({ v, upto, title }: { v: ReturnType<typeof warpView>; upto: number; title: string }) {
  const L = v.width > 32 ? 8 : 13;
  const G = v.width > 32 ? 1.5 : 2;
  const LX = 92;
  const width = LX + v.width * (L + G) + 6;
  const rows = v.steps.length;
  const height = 8 + rows * (L + G) + 4;
  return (
    <div>
      <p style={{ margin: '0 0 0.3rem' }}>
        <b>{title}</b>{' '}
        {upto >= v.finishesAt ? <span className="chip ok" data-part="status"><Translated vi={<>xong sau {v.finishesAt} bước</>}>done after {v.finishesAt} steps</Translated></span> : <span className="chip" data-part="status"><Translated vi={<>bước {upto} trên {v.finishesAt}</>}>step {upto} of {v.finishesAt}</Translated></span>}
      </p>
      <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={tr(`${title}: lanes per step`, `${title}: các lane theo từng bước`)} data-part="lanes" data-net={v.id}
        style={{ width: '100%', maxWidth: width * 1.3, minWidth: 460, display: 'block' }}>
        {v.steps.map((s, k) => (
          <g key={k} opacity={k < upto ? 1 : 0.15} data-part="lane-row" data-step={k + 1} data-adds={s.adds} data-idle={s.idle} data-shown={k < upto ? 1 : 0}>
            <text x={LX - 6} y={8 + k * (L + G) + L - 2} fontSize={v.width > 32 ? 8 : 9.5} textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
              <Translated vi={<>{s.adds} cộng · {s.idle} rảnh</>}>{s.adds} add · {s.idle} idle</Translated>
            </text>
            {Array.from({ length: v.width }, (_, l) => (
              <rect key={l} x={LX + l * (L + G)} y={8 + k * (L + G)} width={L} height={L} rx={1.2}
                fill={s.lanes.includes(l) ? 'var(--ink-2)' : 'none'} stroke="var(--rule)" />
            ))}
          </g>
        ))}
      </svg>
      </div>
    </div>
  );
}

function WarpExplorer() {
  const { reports } = useReports();
  const widths = useMemo(() => widthChoices(reports as Reports).filter((c) => [8, 16, 32, 64].includes(c.width)), [reports]);
  const [key, setKey] = useState('book');
  const W = (widths.find((c) => c.key === key) ?? widths[0]).width;
  const ks = useMemo(() => warpView('kogge-stone', W), [W]);
  const bk = useMemo(() => warpView('brent-kung', W), [W]);
  const [t, setT] = useState(0);
  useEffect(() => setT(0), [W]);
  const both = [ks, bk];
  return (
    <Bench
      title={tr('Inside one warp', 'Bên trong một warp')}
      task={
        <Translated vi="Kéo thời gian về phía trước. Khi nào mỗi mạng xong, và đến lúc đó warp đã tiêu tốn bao nhiêu suất lane?">
          Slide time forward. When is each network done, and how many lane slots has the warp spent by then?
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Một warp scan mỗi lane một giá trị. Ô vuông tô đặc là một lane có cộng ở bước đó; ô trống thì rảnh nhưng vẫn là
              một phần của warp mà bước đó được phát lệnh cho. Một suất lane là một lane trong một bước.
            </>
          }
        >
          <>
            One warp scans one value per lane. A filled square is a lane that adds in that step; an empty one idles but is part
            of the warp the step is issued for. A lane slot is one lane for one step.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={key} aria-label={tr('warp width', 'độ rộng warp')} onChange={(e) => setKey(e.target.value)}>
            {widths.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated
            vi={
              <span>
                Sau bước <b data-value="t">{t}</b>
              </span>
            }
          >
            <span>
              After step <b data-value="t">{t}</b>
            </span>
          </Translated>
          <input type="range" min={0} max={bk.finishesAt} value={t} aria-label={tr('Steps elapsed', 'Số bước đã qua')} onChange={(e) => setT(Number(e.target.value))} style={{ width: '14rem' }} />
        </label>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr)' }}>
        <LaneGrid v={ks} upto={t} title={tr('Kogge-Stone', 'Kogge-Stone')} />
        <LaneGrid v={bk} upto={t} title={tr('Brent-Kung', 'Brent-Kung')} />
      </div>
      <div style={{ overflowX: 'auto', marginTop: '0.8rem' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-part="warp-table">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}><Translated vi={<>{W} lane</>}>{W} lanes</Translated></th>
              <th className="num"><Translated vi="phép cộng">additions</Translated></th>
              <th className="num"><Translated vi="bước">steps</Translated></th>
              <th className="num"><Translated vi="suất lane">lane slots</Translated></th>
              <th className="num"><Translated vi="suất rảnh">idle slots</Translated></th>
              <th className="num"><Translated vi={<>suất đã tiêu đến bước {t}</>}>slots spent by step {t}</Translated></th>
            </tr>
          </thead>
          <tbody>
            {both.map((v) => (
              <tr key={v.id} data-row={v.id} data-adds={v.additions} data-steps={v.finishesAt} data-slots={v.slots} data-idle={v.idleSlots}>
                <td style={{ paddingLeft: '0.6rem' }}>{v.id === 'kogge-stone' ? 'Kogge-Stone' : 'Brent-Kung'}</td>
                <td className="num">{v.additions}</td>
                <td className="num">{v.finishesAt}</td>
                <td className="num">{v.slots}</td>
                <td className="num">{v.idleSlots}</td>
                <td className="num" data-cell="spent">{Math.min(t, v.finishesAt) * W}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Translated
        vi={
          <p className="narrate" role="status" style={{ margin: '0.8rem 0 0' }} data-readout="warp">
            Brent-Kung tiết kiệm {ks.additions - bk.additions} phép cộng trên {W} lane, nhưng nó phát {bk.finishesAt} bước
            trong khi Kogge-Stone phát {ks.finishesAt}, nên warp tiêu {bk.slots} suất lane cho nó so với {ks.slots}. Trong số
            các suất của Brent-Kung, {bk.idleSlots} là lane rảnh.
          </p>
        }
      >
        <p className="narrate" role="status" style={{ margin: '0.8rem 0 0' }} data-readout="warp">
          Brent-Kung saves {ks.additions - bk.additions} additions on {W} lanes, but it issues {bk.finishesAt} steps where
          Kogge-Stone issues {ks.finishesAt}, so the warp spends {bk.slots} lane slots on it against {ks.slots}. Of Brent-Kung's
          slots, {bk.idleSlots} are idle lanes.
        </p>
      </Translated>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const EX6_LEVEL1 = run(brentKung(8), EX6)[upSteps(brentKung(8)) + 1];
const BK1024 = brentKung(1024);

export const questionsS1110: Record<string, Question> = {
  'ch11/11.10/q-ex6': {
    kind: 'numeric',
    id: 'ch11/11.10/q-ex6',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 6 của sách: Brent-Kung trên <code>4 6 7 1 2 8 5 2</code>. Vị trí 5 giữ giá trị gì sau mức đầu tiên của cây
            đảo ngược?
          </>
        }
      >
        <>
          Book exercise 6: Brent-Kung on <code>4 6 7 1 2 8 5 2</code>. What does position 5 hold after the first level of the
          reverse tree?
        </>
      </Translated>
    ),
    answer: EX6_LEVEL1[5],
    explain: (
      <Translated
        vi={
          <p>
            Cây rút gọn cho 4 10 7 8 2 10 5 7, rồi 4 10 7 18 2 10 5 17, rồi 4 10 7 18 2 10 5 35. Với 8 phần tử, mức đầu tiên
            của cây đảo ngược làm việc ở khoảng cách 2: vị trí 5 cộng vị trí 3, 10 + 18 = {EX6_LEVEL1[5]}. Mức cuối cùng cộng
            1, 3, 5 vào 2, 4, 6: 4 10 17 18 20 28 33 35.
          </p>
        }
      >
        <p>
          The reduction tree gives 4 10 7 8 2 10 5 7, then 4 10 7 18 2 10 5 17, then 4 10 7 18 2 10 5 35. For 8 elements the
          reverse tree's first level works 2 apart: position 5 adds position 3, 10 + 18 = {EX6_LEVEL1[5]}. The last level adds
          1, 3, 5 into 2, 4, 6: 4 10 17 18 20 28 33 35.
        </p>
      </Translated>
    ),
    source: 'ch. 11 exercise 6, p. 286 (the answer is ours)',
    vi: { source: 'ch. 11 bài tập 6, p. 286 (đáp án là của khóa học này)' },
  },
  'ch11/11.10/q-ops': {
    kind: 'numeric',
    id: 'ch11/11.10/q-ops',
    prompt: <Translated vi={<>Brent-Kung thực hiện bao nhiêu phép cộng trên một đoạn 1,024 phần tử?</>}><>How many additions does Brent-Kung perform on a segment of 1,024 elements?</></Translated>,
    answer: additions(BK1024),
    unit: 'additions',
    explain: (
      <Translated
        vi={
          <p>
            2N − 2 − log₂N = 2048 − 2 − 10 = {additions(BK1024)}: 1,023 trong cây rút gọn và 1,013 trong cây đảo ngược.
            Kogge-Stone sẽ làm {closedForm('kogge-stone', 1024).toLocaleString('en-US')}.
          </p>
        }
      >
        <p>
          2N − 2 − log₂N = 2048 − 2 − 10 = {additions(BK1024)}: 1,023 in the reduction tree and 1,013 in the reverse tree.
          Kogge-Stone would do {closedForm('kogge-stone', 1024).toLocaleString('en-US')}.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.10, p. 284',
    vi: { unit: 'phép cộng' },
  },
  'ch11/11.10/q-steps': {
    kind: 'numeric',
    id: 'ch11/11.10/q-steps',
    prompt: <Translated vi={<>Và trong bao nhiêu bước, với đủ đơn vị cho mọi bộ cộng của một bước?</>}><>And in how many steps, with enough units for every adder of a step?</></Translated>,
    answer: BK1024.steps.length,
    unit: 'steps',
    explain: (
      <Translated
        vi={
          <p>
            log₂N = 10 trong cây rút gọn và log₂N − 1 = 9 trong cây đảo ngược: {BK1024.steps.length}, so với{' '}
            {span('kogge-stone', 1024)} của Kogge-Stone. Với 16 phần tử là 4 + 3 = 7, bảy hàng của Fig. 11.19.
          </p>
        }
      >
        <p>
          log₂N = 10 in the reduction tree and log₂N − 1 = 9 in the reverse tree: {BK1024.steps.length}, against Kogge-Stone's{' '}
          {span('kogge-stone', 1024)}. For 16 elements it is 4 + 3 = 7, the seven rows of Fig. 11.19.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.10, pp. 282–284 (the count 2·log₂N − 1 is ours, from Fig. 11.19)',
    vi: { unit: 'bước', source: 'ch. 11 §11.10, pp. 282–284 (con số 2·log₂N − 1 là của khóa học này, rút ra từ Fig. 11.19)' },
  },
  'ch11/11.10/q-needs': {
    kind: 'numeric',
    id: 'ch11/11.10/q-needs',
    prompt: (
      <Translated
        vi={
          <>
            N = 16. Sau cây rút gọn, vị trí 13 giữ x12..x13. Nó nhận tổng riêng phần còn thiếu từ vị trí nào?
          </>
        }
      >
        <>
          N = 16. After the reduction tree, position 13 holds x12..x13. From which position does it receive the partial sum it
          still lacks?
        </>
      </Translated>
    ),
    answer: needsChain(reverseLevels(16)[1].ranges, 13)[0],
    unit: '',
    explain: (
      <Translated
        vi={
          <p>
            Nó thiếu x0..x11. Sau mức 1 của cây đảo ngược, vị trí 11 giữ đúng phần đó (7 đã được cộng vào nó), nên ở mức 2 vị
            trí 13 cộng vị trí 11, cách nó hai vị trí về bên trái. Trước mức 1, nó sẽ cần hai mảnh, 11 (x8..x11) và 7 (x0..x7).
          </p>
        }
      >
        <p>
          It lacks x0..x11. After level 1 of the reverse tree position 11 holds exactly that (7 was added into it), so at level
          2 position 13 adds position 11, two places to its left. Before level 1 it would have needed two pieces, 11 (x8..x11)
          and 7 (x0..x7).
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.10, pp. 283–284, Fig. 11.20',
  },
  'ch11/11.10/q-warp': {
    kind: 'choice',
    id: 'ch11/11.10/q-warp',
    prompt: <Translated vi={<>Sau khi gộp việc cho luồng (thread coarsening), vì sao sách giữ Kogge-Stone cho phép scan mức warp, bất chấp phần việc thừa của nó?</>}><>After coarsening, why does the book keep Kogge-Stone for the warp-level scan, despite its extra work?</></Translated>,
    options: [
      { label: <Translated vi="Brent-Kung không thể viết bằng shuffle, vì các bước của nó cần những giá trị mà shuffle không với tới.">Brent-Kung cannot be written with shuffles, because its steps need values a shuffle cannot reach.</Translated>, why: <Translated vi="Cả hai mạng chỉ đọc giá trị cách một lũy thừa của hai về bên trái, thứ mà shuffle-up đưa tới được.">Both networks only ever read a value a power of two to the left, which a shuffle-up delivers.</Translated> },
      {
        label: <Translated vi="Trong một warp, các lane mà Brent-Kung để rảnh vẫn bị chiếm dụng, nên các bước thêm của nó tốn thời gian mà không tiết kiệm được gì.">In a warp the lanes Brent-Kung leaves idle are occupied anyway, so its extra steps cost time and save nothing.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Kogge-Stone làm ít phép cộng hơn trong một warp, vì mỗi lane chỉ cộng một lần mỗi bước nhảy.">Kogge-Stone does fewer additions within one warp, because each lane adds only once per stride.</Translated>, why: <Translated vi="Nó làm nhiều hơn: 129 so với 57 cho 32 lane.">It does more: 129 against 57 for 32 lanes.</Translated> },
      { label: <Translated vi="Brent-Kung cần rào chắn giữa các bước ngay cả bên trong một warp, còn Kogge-Stone thì không.">Brent-Kung needs a barrier between its steps even inside a warp, and Kogge-Stone does not.</Translated>, why: <Translated vi="Bên trong một warp, không mạng nào dùng rào chắn khối.">Inside a warp neither uses a block barrier.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đến lúc đó, phần lớn công việc là phép scan tuần tự của mỗi luồng, và phép scan song song chỉ chiếm một phần nhỏ
            thời gian chạy. Trong một warp, mỗi bước được phát lệnh cho mọi lane. Các phép cộng mà Brent-Kung tránh được biến
            thành những lane ngồi không nhưng vẫn dùng tài nguyên thực thi của warp, và nó cần 2·log₂N − 1 bước trong khi
            Kogge-Stone cần log₂N. Hãy dùng các lane và xong sớm hơn.
          </p>
        }
      >
        <p>
          By then most of the work is each thread's sequential scan, and the parallel scan is a small part of the run time.
          In a warp every step is issued for all the lanes. The additions Brent-Kung avoids turn into lanes that sit idle but
          still use the warp's execution resources, and it needs 2·log₂N − 1 steps where Kogge-Stone needs log₂N. Use the
          lanes and finish sooner.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.10, pp. 284–285',
  },
  'ch11/11.10/q-ex7': {
    kind: 'choice',
    id: 'ch11/11.10/q-ex7',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 7: một kernel Brent-Kung mức khối cập nhật <code>buffer_s[idx] += buffer_s[idx - stride]</code> (hoặc dạng
            cây đảo ngược của nó) cho các vị trí đang hoạt động của mỗi bước, tại chỗ. Vì sao một rào chắn mỗi lần lặp là đủ,
            trong khi Kogge-Stone tại chỗ của Fig. 11.3 cần hai?
          </>
        }
      >
        <>
          Exercise 7: a block-level Brent-Kung kernel updates <code>buffer_s[idx] += buffer_s[idx - stride]</code> (or its
          reverse-tree form) for the active positions of each step, in place. Why is one barrier per iteration enough, when
          Fig. 11.3's in-place Kogge-Stone needed two?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Brent-Kung có ít bước hơn, nên nó có thể bỏ rào chắn thứ hai mà race condition không bao giờ lộ ra.">Brent-Kung has fewer steps, so it can drop the second barrier without the race ever showing.</Translated>, why: <Translated vi="Nó có nhiều bước hơn, và bỏ một rào chắn cần thiết thì không bao giờ chấp nhận được.">It has more steps, and skipping a needed barrier is never affordable.</Translated> },
      {
        label: <Translated vi="Trong một bước Brent-Kung, không luồng nào đọc vị trí mà luồng khác ghi; chỉ phụ thuộc giữa các bước mới cần rào chắn.">In a Brent-Kung step no thread reads a position another thread writes; only the dependence between steps needs a barrier.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Bộ nhớ chia sẻ tuần tự hóa các lần ghi vào cùng một bank, nên một lần đọc và lần ghi của luồng khác không thể chồng lên nhau.">Shared memory serialises writes to the same bank, so a read and another thread's write cannot overlap.</Translated>, why: <Translated vi="Hành vi của bank liên quan đến tốc độ, không phải thứ tự đọc và ghi.">Bank behaviour is about speed, not about the order of reads and writes.</Translated> },
      { label: <Translated vi="Mỗi bước chỉ có một luồng hoạt động, nên không có luồng nào khác để tranh chấp.">Only one thread is active in each step, so no other thread is there to race with.</Translated>, why: <Translated vi="Bước đầu tiên của cây rút gọn có N/2 luồng hoạt động.">The first step of the reduction tree has N/2 active threads.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Rào chắn thứ hai của Fig. 11.3 áp đặt quan hệ ghi-sau-đọc: một luồng Kogge-Stone ghi đè lên một vị trí mà luồng
            khác cùng bước vẫn còn phải đọc. Trong một bước Brent-Kung, cây rút gọn ghi các vị trí 2·stride·k − 1 và đọc các vị
            trí đó trừ stride; cây đảo ngược ghi các vị trí đó cộng stride. Hai tập này không bao giờ gặp nhau, như dòng "đọc và
            ghi" của trình chạy từng bước cho thấy, nên phụ thuộc duy nhất còn lại là phụ thuộc thật giữa các bước liên tiếp.
          </p>
        }
      >
        <p>
          The second barrier of Fig. 11.3 enforces a write-after-read: a Kogge-Stone thread overwrites a position that another
          thread of the same step still has to read. In a Brent-Kung step the reduction tree writes positions 2·stride·k − 1 and
          reads those minus stride; the reverse tree writes those plus stride. The sets never meet, as the stepper's "read and
          write" line shows, so the only dependence left is the true one between consecutive steps.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.2, pp. 257–258; §11.3, pp. 258–259; exercise 7, p. 286 (the answer is ours)',
    vi: { source: 'ch. 11 §11.2, pp. 257–258; §11.3, pp. 258–259; bài tập 7, p. 286 (đáp án là của khóa học này)' },
  },
};

// ---------------------------------------------------------------- the section
export function S1110() {
  const q = questionsS1110;
  const cu = useLabFile(SNIPPET);
  const loaded = cu && !cu.startsWith('// could not load');
  const barrierLines = loaded ? cu.split('\n').flatMap((l, i) => (l.includes('__syncthreads();') ? [i + 1] : [])) : [];
  return (
    <>
      <Translated
        vi={
          <p>
            Mọi kernel trong chương này đều dùng Kogge-Stone, và phần 11.5 đã đếm cái giá của nó: N·log₂N − (N − 1) phép cộng
            trong khi vòng lặp chỉ cần N − 1. Phần này trình bày mạng cổ điển còn lại, Brent-Kung, đưa số phép cộng xuống khoảng
            2N với cái giá là nhiều bước hơn. Sách đưa nó vào cho đầy đủ và nói rõ như vậy: sau khi phần 11.6 gộp việc cho luồng
            (thread coarsening), phần việc thừa của Kogge-Stone không còn là mối lo lớn.
          </p>
        }
      >
        <p>
          Every kernel in this chapter has used Kogge-Stone, and section 11.5 counted what that costs: N·log₂N − (N − 1)
          additions where the loop needs N − 1. This section presents the other classic network, Brent-Kung, which brings
          the count down to about 2N at the price of more steps. The book includes it for completeness and says so: after
          section 11.6's coarsening, Kogge-Stone's extra work stopped being a major concern.
        </p>
      </Translated>
      <Source section="11.10" pages="282" />
      <Translated
        vi={
          <p>
            Brent-Kung, cũng như Kogge-Stone, khởi đầu là một thiết kế cho mạch cộng (tài liệu tham khảo [8] của sách). Nó bắt
            đầu từ một cây rút gọn, cây này tính tổng của N giá trị trong log₂N bước và, trên đường đi, tạo ra các tổng con mà
            một số đầu ra của phép scan có thể dùng. Kogge-Stone chạy một cây cho mỗi đầu ra và chia sẻ những gì có thể giữa
            chúng. Brent-Kung chạy một cây rồi trao các tổng con của nó cho những vị trí cần chúng.
          </p>
        }
      >
        <p>
          Brent-Kung, like Kogge-Stone, began as a design for adder circuits (the book's reference [8]). It starts from a
          reduction tree, which computes the total of N values in log₂N steps and, on the way, produces sub-sums that some of
          the scan's outputs could use. Kogge-Stone runs a tree for every output and shares what it can between them.
          Brent-Kung runs one tree and then hands its sub-sums to the positions that need them.
        </p>
      </Translated>

      <h2><Translated vi="Pha 1: cây rút gọn">Phase 1: the reduction tree</Translated></h2>
      <Translated
        vi={
          <p>
            Pha đầu tiên là <a href={href('ch10', '10.2')}>cây rút gọn của chương 10</a>, tính tại chỗ. Trên 16 phần tử nó
            mất 4 bước. Bước 1 cập nhật các vị trí có chỉ số dạng 2n − 1 (1, 3, 5, …, 15), mỗi vị trí cộng phần tử láng giềng
            bên trái. Bước 2 cập nhật các vị trí 4n − 1 (3, 7, 11, 15), mỗi vị trí cộng vị trí cách nó 2 về bên trái. Bước 3
            cập nhật 8n − 1, từ cách 4 về bên trái, và bước 4 chỉ riêng vị trí 15. Tổng cộng 8 + 4 + 2 + 1 = 15 phép cộng, và
            N − 1 trong trường hợp tổng quát.
          </p>
        }
      >
        <p>
          The first phase is <a href={href('ch10', '10.2')}>chapter 10's reduction tree</a>, computed in place. On 16
          elements it takes 4 steps. Step 1 updates the positions whose index has the form 2n − 1 (1, 3, 5, …, 15), each adding
          its left neighbour. Step 2 updates the positions 4n − 1 (3, 7, 11, 15), each adding the position 2 to its left. Step
          3 updates 8n − 1, from 4 to the left, and step 4 position 15 alone. That is 8 + 4 + 2 + 1 = 15 additions, and N − 1
          in general.
        </p>
      </Translated>
      <Source section="11.10" pages="282" figs={tr('Fig. 11.19, top', 'Fig. 11.19, phần trên')} />
      <Translated
        vi={
          <p>
            Giờ mỗi vị trí giữ một dãy đầu vào liên tiếp kết thúc tại chính nó, x<sub>i</sub>..x<sub>j</sub>. Các vị trí 0, 1,
            3, 7 và 15 giữ mọi thứ từ x0: chúng đã xong. Các vị trí khác giữ một phần đáp án của mình.
          </p>
        }
      >
        <p>
          Every position now holds a contiguous run of inputs ending at itself, x<sub>i</sub>..x<sub>j</sub>. Positions 0, 1,
          3, 7 and 15 hold everything from x0: they are final. The others hold a piece of their answer.
        </p>
      </Translated>

      <h2><Translated vi="Pha 2: cây đảo ngược">Phase 2: the reverse tree</Translated></h2>
      <Translated
        vi={
          <p>
            Hãy xem mỗi vị trí còn thiếu gì. Vị trí 14 chỉ giữ x14. Nó cần x0..x13, thứ mà cây đã để lại thành ba mảnh:
            x12..x13 ở vị trí 13, x8..x11 ở vị trí 11 và x0..x7 ở vị trí 7. Hai sự thật làm cho pha thứ hai khả thi. Không vị
            trí nào cần nhiều hơn log₂N − 1 mảnh, và các mảnh nằm cách nhau một lũy thừa của hai: 14 đến 13 là 1, 13 đến 11 là
            2, 11 đến 7 là 4.
          </p>
        }
      >
        <p>
          Look at what each position is missing. Position 14 holds only x14. It needs x0..x13, which the tree has left in three
          pieces: x12..x13 at position 13, x8..x11 at position 11 and x0..x7 at position 7. Two facts make the second phase
          possible. No position needs more than log₂N − 1 pieces, and the pieces sit a power of two apart: 14 to 13 is 1, 13
          to 11 is 2, 11 to 7 is 4.
        </p>
      </Translated>
      <Source section="11.10" pages="283–284" figs="Fig. 11.20" />
      <Translated
        vi={
          <p>
            Vì vậy cây đảo ngược làm việc từ khoảng cách lớn nhất đến nhỏ nhất. Mức 1 cộng vị trí 7 vào vị trí 11, cách 4, và
            11 đã xong. Mức 2 cộng 3, 7 và 11 vào 5, 9 và 13, cách 2. Mức 3 cộng mọi vị trí lẻ vào vị trí chẵn ngay bên phải
            nó. Với 16 phần tử, đó là thêm 1 + 3 + 7 = 11 phép cộng trong thêm 3 bước.
          </p>
        }
      >
        <p>
          So the reverse tree works from the largest distance to the smallest. Level 1 adds position 7 into position 11, 4
          apart, and 11 is final. Level 2 adds 3, 7 and 11 into 5, 9 and 13, 2 apart. Level 3 adds every odd position into the
          even position to its right. For 16 elements that is 1 + 3 + 7 = 11 more additions in 3 more steps.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Chỗ nhầm, pp. 283–284: pha rút gọn được gọi là "bottom portion" của Fig. 11.19 (thật ra nó ở trên), và các chỉ số
              dưới hai chữ số chỉ hạ chữ số đầu tiên ("x₁4" là x14).
            </p>
          }
        >
          <p>
            Slips, pp. 283–284: the reduction phase is called the "bottom portion" of Fig. 11.19 (it is the top), and two-digit
            subscripts lower only their first digit ("x₁4" is x14).
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Hãy chạy từng bước cả hai mạng trên cùng một đầu vào bên dưới. Tám giá trị của bài tập 6 cho phép bạn kiểm tra
            phép tính bằng tay; mười sáu giá trị của Fig. 11.6 cho ra đúng Fig. 11.19 và 11.20. Bảng dưới các mạng là
            Fig. 11.20, được tính lại từ các khoảng mà các ô giữ.
          </p>
        }
      >
        <p>
          Step both networks on the same input below. Exercise 6's eight values let you check the arithmetic by hand; Fig.
          11.6's sixteen give Fig. 11.19 and 11.20 exactly. The table under the networks is Fig. 11.20, recomputed from the
          ranges the cells hold.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mọi bộ cộng của một bước đọc các giá trị từ trước bước đó, như trong một mạch điện. Trình chạy từng bước đếm phép
              cộng và số bước; nó không có khái niệm thời gian mỗi bước, bộ nhớ hay rào chắn.
            </p>
          }
        >
          <p>
            Every adder of a step reads the values from before the step, as in a circuit. The stepper counts additions and
            steps; it has no notion of time per step, memory or barriers.
          </p>
        </Translated>
      </Simplification>
      <Stepper />
      <Translated
        vi={
          <p>
            Theo chính chú giải của sách, vị trí 9 (giữ x8..x9) và vị trí 11 (x8..x11) mỗi vị trí cần thêm một tổng riêng phần
            trước khi cây đảo ngược bắt đầu, tổng ở vị trí 7. Fig. 11.20 tô cả hai màu xám đậm, màu cho hai, và vị trí 9 vẫn
            xám đậm ở hàng Level 1. Mọi ô khác đều khớp. Chuyển cách tô sang "như in trong sách" để thấy ba ô đó.
          </p>
        }
      >
        <p>
          By the book's own legend, position 9 (holding x8..x9) and position 11 (x8..x11) each need one more partial sum
          before the reverse tree starts, the one at position 7. Fig. 11.20 shades both dark grey, the colour for two, and
          position 9 stays dark grey in the Level 1 row. Every other cell matches. Switch the shading to "as printed" to see
          the three cells.
        </p>
      </Translated>
      <Aside>
        <Translated vi={<p>Cách tô của Fig. 11.20, p. 284: cách chúng tôi đọc hình in so với chú giải của nó ở p. 283.</p>}>
          <p>Fig. 11.20 shading, p. 284: our reading of the printed figure against its legend on p. 283.</p>
        </Translated>
      </Aside>
      <Ask q={q['ch11/11.10/q-ex6']} label={tr('Work it out', 'Tự tính ra')} />
      <Ask q={q['ch11/11.10/q-needs']} label={tr('Work it out', 'Tự tính ra')} />

      <h2><Translated vi="Đếm khối lượng công việc">Counting the work</Translated></h2>
      <Translated
        vi={
          <p>
            Cây đảo ngược cộng (2 − 1) + (4 − 1) + … + (N/4 − 1) + (N/2 − 1) = N − 1 − log₂N. Cùng với N − 1 của cây rút gọn,
            tổng là 2N − 2 − log₂N phép cộng: 26 cho 16 phần tử, so với 49 của Kogge-Stone. Đó là O(N), bậc của vòng lặp, trong
            khi Kogge-Stone là O(N·log₂N). Cái giá là số bước: log₂N trong cây rút gọn và log₂N − 1 trong cây đảo ngược,
            2·log₂N − 1 tất cả, so với log₂N của Kogge-Stone.
          </p>
        }
      >
        <p>
          The reverse tree adds (2 − 1) + (4 − 1) + … + (N/4 − 1) + (N/2 − 1) = N − 1 − log₂N. With the reduction tree's N − 1
          that makes 2N − 2 − log₂N additions: 26 for 16 elements, against Kogge-Stone's 49. That is O(N), the order of the
          loop, where Kogge-Stone is O(N·log₂N). The price is steps: log₂N in the reduction tree and log₂N − 1 in the reverse
          tree, 2·log₂N − 1 in all, against Kogge-Stone's log₂N.
        </p>
      </Translated>
      <Source section="11.10" pages="284" />
      <Ask q={q['ch11/11.10/q-ops']} label={tr('Work it out', 'Tự tính ra')} />
      <Ask q={q['ch11/11.10/q-steps']} label={tr('Work it out', 'Tự tính ra')} />
      <Translated
        vi={
          <p>
            Hoạt hình chạy cả hai pha một lần trên mười sáu vị trí, rồi chuyển vào một warp duy nhất, nơi phép so sánh đảo
            chiều.
          </p>
        }
      >
        <p>
          The animation runs both phases once on sixteen positions, then moves into a single warp, where the comparison turns
          around.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Được vẽ từ mô hình của trình chạy từng bước. Trong phần xem warp, một bước là một lần phát lệnh cộng cho cả 32
              lane; một lane rảnh chỉ chiếm chỗ trong warp. Không có bộ nhớ, không có rào chắn, không có thời gian mỗi bước.
            </p>
          }
        >
          <p>
            Drawn from the stepper's model. In the warp view a step is one issue of the addition for all 32 lanes; an idle
            lane only occupies the warp. No memory, no barriers, no time per step.
          </p>
        </Translated>
      </Simplification>
      <BkSweepsScene />

      <h2><Translated vi="Vì sao Kogge-Stone vẫn thắng bên trong một warp">Why Kogge-Stone still wins inside a warp</Translated></h2>
      <Translated
        vi={
          <p>
            Hãy nhớ lại việc gộp việc cho luồng đã làm gì ở phần 11.6: mỗi luồng scan tuần tự đoạn phần tử của riêng nó, và phép
            scan song song duy nhất còn lại là trên các lane của một warp, một phần nhỏ của thời gian chạy. Ở đó, theo sách,
            Kogge-Stone chạy tốt hơn trong thực tế. Một warp thực thi mỗi bước cho mọi lane của nó. Các phép cộng mà Brent-Kung
            tránh được không biến mất; chúng trở thành những lane ngồi không trong một warp vẫn đang thực thi và vẫn dùng tài
            nguyên thực thi của nó. Tốt hơn là cho các lane đó làm việc, như Kogge-Stone làm, và xong trong ít bước hơn.
          </p>
        }
      >
        <p>
          Recall what coarsening did in section 11.6: each thread scans its own run of elements sequentially, and the only
          parallel scan left is across the lanes of a warp, a small part of the run time. There, the book reports, Kogge-Stone
          performs better in practice. A warp executes each step for all its lanes. The additions Brent-Kung avoids do not
          vanish; they become lanes that sit idle in a warp that is still executing and still use its execution resources.
          Better to put those lanes to work, as Kogge-Stone does, and finish in fewer steps.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A9, B7, C4; ch11-facts A4)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              NVIDIA và AMD đều ghi trong tài liệu rằng các lane bị che (masked) vẫn đi qua lệnh của warp. Apple không ghi mô
              hình thực thi nào cho các luồng SIMD group không hoạt động. Tài liệu <code>BlockScan</code> của CUB mô tả bước
              warp trong biến thể warp-scans của nó là "a shallow, but inefficient warp-synchronous Kogge-Stone style scan".
            </p>
          }
        >
          <p>
            NVIDIA and AMD both document masked lanes that still go through the warp's instruction. Apple documents no
            execution model for inactive SIMD-group threads. CUB's <code>BlockScan</code> docs describe the warp step of its
            warp-scans variant as "a shallow, but inefficient warp-synchronous Kogge-Stone style scan".
          </p>
        </Translated>
      </Today>
      <Translated vi={<p>Hãy thử cả hai độ rộng warp. Warp phát lệnh mỗi bước cho mọi lane của nó, dù rảnh hay không.</p>}>
        <p>Try both warp widths. The warp issues every step for all its lanes, idle or not.</p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một bước là một lần phát lệnh cộng cho cả warp. Shuffle, phép so sánh và chi phí của một lane rảnh ngoài việc "nó
              chiếm chỗ trong warp" không được mô hình hóa; NVIDIA cảnh báo rằng phần cứng có thể xử lý các lane bị che theo
              những cách mà chương trình không thấy được.
            </p>
          }
        >
          <p>
            One step is one issue of the addition for the whole warp. Shuffles, comparisons and the cost of an idle lane beyond
            "it occupies the warp" are not modelled; NVIDIA warns that hardware may treat masked lanes in ways a program cannot
            see.
          </p>
        </Translated>
      </Simplification>
      <WarpExplorer />
      <Ask q={q['ch11/11.10/q-warp']} />

      <h2><Translated vi="Bài tập 7: một kernel mức khối">Exercise 7: a block-level kernel</Translated></h2>
      <Translated
        vi={
          <p>
            Sách để lại kernel Brent-Kung cho người đọc. Đây là bản của chúng tôi. Một khối scan một đoạn <code>SEG_SIZE</code>{' '}
            phần tử với <code>SEG_SIZE/2</code> luồng, vì bước rộng nhất, bước đầu tiên của cây rút gọn, có đúng chừng đó phép
            cộng; mỗi luồng nạp và ghi hai phần tử. Trong cả hai vòng lặp, luồng t thực hiện phép cộng thứ t của bước, tại{' '}
            <code>idx = (t + 1)·2·stride − 1</code>, nên các luồng hoạt động luôn là những luồng có số thứ tự thấp nhất và cả
            warp cùng rời đi khi cây thu hẹp (bài học của <a href={href('ch10', '10.4')}>phần 10.4</a>).
          </p>
        }
      >
        <p>
          The book leaves a Brent-Kung kernel to the reader. Here is ours. A block scans a segment of <code>SEG_SIZE</code>{' '}
          elements with <code>SEG_SIZE/2</code> threads, since the widest step, the first of the reduction tree, has that many
          additions; each thread loads and stores two elements. In both loops thread t performs the t-th addition of the step,
          at <code>idx = (t + 1)·2·stride − 1</code>, so the active threads are always the lowest-numbered ones and whole warps
          drop out together as the tree narrows (the lesson of <a href={href('ch10', '10.4')}>section 10.4</a>).
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Kernel này là của chúng tôi; sách không in đáp án. Được dựng từ Fig. 11.19 và kiểm tra với mô hình của trình chạy
              từng bước trên các trình giả lập (emulator) của khóa học, kể cả một phiên bản bỏ các rào chắn trong vòng lặp và
              cho kết quả sai.
            </p>
          }
        >
          <p>
            The kernel is ours; the book prints no answer. Built from Fig. 11.19 and checked against the stepper's model on
            the course's emulators, including a version without the loop barriers that goes wrong.
          </p>
        </Translated>
      </Beyond>
      {cu ? (
        <Code
          tabs={[{ label: 'CUDA', code: cu, tiers: TIERS, status: 'emulator-checked and front-end-checked; unverified on CUDA hardware' }]}
          highlight={barrierLines}
          caption={
            <Translated
              vi={
                <>
                  Nạp từ <code>labs/{SNIPPET}</code>. Các dòng được tô sáng là các rào chắn của nó: một ở đầu mỗi lần lặp của cả
                  hai vòng lặp, và một trước các lệnh ghi. Nó chỉ dùng bộ nhớ chia sẻ, rào chắn và chỉ số luồng, những thứ mà bản
                  build HIP viết y hệt; nó chưa được biên dịch cho HIP.
                </>
              }
            >
              <>
                Loaded from <code>labs/{SNIPPET}</code>. The highlighted lines are its barriers: one at the top of each
                iteration of both loops, and one before the stores. It uses only shared memory, barriers and thread indices,
                which the HIP build spells the same way; it has not been compiled for HIP.
              </>
            </Translated>
          }
        />
      ) : null}
      <Translated
        vi={
          <p>
            Hãy so sánh các rào chắn của nó với Fig. 11.3. Kernel đó cần hai rào chắn mỗi lần lặp, vì một luồng Kogge-Stone ghi
            đè lên một vị trí mà luồng khác cùng bước chưa đọc. Dòng "đọc và ghi" của trình chạy từng bước cho thấy Kogge-Stone
            làm đúng điều đó ở mọi bước trừ bước cuối, còn Brent-Kung thì không bao giờ.
          </p>
        }
      >
        <p>
          Compare its barriers with Fig. 11.3's. That kernel needed two per iteration, because a Kogge-Stone thread overwrites a
          position that another thread of the same step has yet to read. The stepper's "read and write" line shows Kogge-Stone
          doing exactly that in every step but its last, and Brent-Kung never.
        </p>
      </Translated>
      <Source section="11.10" pages="285" figs={tr('exercise 7, p. 286; Fig. 11.3, pp. 256–258', 'bài tập 7, p. 286; Fig. 11.3, pp. 256–258')} />
      <Ask q={q['ch11/11.10/q-ex7']} />

      <h2><Translated vi="Một không gian thiết kế, không phải hai thuật toán">A design space, not two algorithms</Translated></h2>
      <Translated
        vi={
          <p>
            Kogge-Stone và Brent-Kung là hai điểm trong rất nhiều cách đi dây cho một tổng tiền tố (prefix sum), mỗi cách đánh
            đổi số phép cộng với số bước và với mức độ đồng đều khi chia việc cho các lane. Phép scan CUDA của PyTorch dọc theo
            chiều cuối, trường hợp batch nhân vocabulary, gọi bước song song của nó là "Sklansky method", một mạng thứ ba, và
            chú thích của nó dẫn link tới báo cáo decoupled look-back để xem sơ đồ. Trên Metal, bước mức warp là một lời gọi,{' '}
            <code>simd_prefix_inclusive_sum</code>, và Apple không ghi trong tài liệu nó dùng mạng nào hay cộng theo thứ tự nào.
          </p>
        }
      >
        <p>
          Kogge-Stone and Brent-Kung are two points among many ways to wire a prefix sum, each trading additions against steps
          and against how evenly the work spreads over lanes. PyTorch's CUDA scan along the last dimension, the
          batch-by-vocabulary case, labels its parallel step "Sklansky method", a third network, and its comment links the
          decoupled look-back report for the diagram. On Metal the warp-level step is one call,{' '}
          <code>simd_prefix_inclusive_sum</code>, and Apple does not document which network it uses or in what order it adds.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md D4, C1)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              PyTorch <code>ScanUtils.cuh</code>: "// Parallel reduction with Sklansky method", một <code>__syncthreads()</code>{' '}
              mỗi bước. Metal: thứ tự cộng của các hàm tiền tố SIMD group không có trong tài liệu.
            </p>
          }
        >
          <p>
            PyTorch <code>ScanUtils.cuh</code>: "// Parallel reduction with Sklansky method", one <code>__syncthreads()</code>{' '}
            per step. Metal: the summation order of the SIMD-group prefix functions is undocumented.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Đến đây là hết các thuật toán của chương. Các <a href={href('ch11', 'labs')}>lab</a> đưa các phép scan khối lên GPU
            của chính bạn, và <a href={href('ch11', 'wrap')}>phần tổng kết</a> ánh xạ các bài tập, kể cả bài 6 và 7 của trang
            này.
          </p>
        }
      >
        <p>
          That completes the chapter's algorithms. The <a href={href('ch11', 'labs')}>labs</a> put the block scans on your own
          GPU, and the <a href={href('ch11', 'wrap')}>summary</a> maps the exercises, including 6 and 7 from this page.
        </p>
      </Translated>
    </>
  );
}
