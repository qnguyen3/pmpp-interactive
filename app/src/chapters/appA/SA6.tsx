import { useEffect, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import {
  ACC_FORMAT,
  ACC_LABEL,
  BOOK_FOUR,
  BUG_HUNT,
  DATA,
  EX4_FORMATS,
  EX4_VALUES,
  METHODS,
  OUR_FOUR,
  TOY_VALUES,
  VERDICT_LABEL,
  absSum,
  bugHunt,
  bugHuntValues,
  challengeHolds,
  chunkSweep,
  ex4Order,
  generate,
  measure,
  reference,
  runningTrace,
  show,
  sumSequential,
  toySum,
  treeTrace,
  ulpText,
  pct,
  type Acc,
  type ChunkPoint,
  type DataKind,
  type Ex4Order,
  type Method,
  type Pairing,
  type ToyRun,
  type Verdict,
} from '../../sim/fpsum';
import { maxFinite } from '../../sim/minifloat';
import { functionSource } from '../../sim/toolchain';
import { KahanScene } from './KahanScene';

const FACTS = 'notes/today/appA-facts.md';
const CHECKED = (items: string) => `2026-11-20 (${FACTS}, ${items})`; // i18n-ignore: a date and a citation

// ================================================================ four values, two orders (p. 593)
const toyOptions = TOY_VALUES;

function AddRow({ a, k, shown }: { a: ToyRun['adds'][number]; k: number; shown: boolean }) {
  if (!shown) return <li className="muted" data-add={k}>·</li>;
  const exact = a.rounded === a.exact;
  return (
    <li data-add={k} data-rounded={a.rounded} data-lost={a.lost}>
      {show(a.left)} + {show(a.right)} = {show(a.exact)}{' '}
      {exact ? (
        <span className="muted"><Translated vi="chính xác">exactly</Translated></span>
      ) : (
        <Translated
          vi={
            <>
              → lưu <b>{show(a.rounded)}</b>{' '}
              <span className="t-fail">({a.lost > 0 ? `mất ${show(a.lost)}` : `dư ${show(-a.lost)}`}{a.saturated ? ', vượt 3.5: bị kẹp' : ''})</span>
            </>
          }
        >
          <>
            → stored <b>{show(a.rounded)}</b>{' '}
            <span className="t-fail">({a.lost > 0 ? `${show(a.lost)} lost` : `${show(-a.lost)} gained`}{a.saturated ? ', past 3.5: clamped' : ''})</span>
          </>
        </Translated>
      )}
    </li>
  );
}

function FourValues() {
  const [vals, setVals] = useState<number[]>(BOOK_FOUR);
  const [n, setN] = useState(0);
  const [tries, setTries] = useState(0);
  const [verdict, setVerdict] = useState<null | 'yes' | 'no'>(null);
  const seq = toySum(vals, 'sequential');
  const tree = toySum(vals, 'tree');
  const set = (i: number, v: number) => {
    setVals(vals.map((x, j) => (j === i ? v : x)));
    setN(0);
    setVerdict(null);
  };
  const check = () => {
    setN(3);
    if (challengeHolds(vals)) setVerdict('yes');
    else {
      setVerdict('no');
      setTries(tries + 1);
    }
  };
  const better = seq.error === tree.error ? tr('Both orders are equally accurate here.', 'Ở đây cả hai thứ tự chính xác như nhau.') : seq.error < tree.error ? tr('The loop is more accurate here.', 'Ở đây vòng lặp chính xác hơn.') : tr('The tree is more accurate here.', 'Ở đây cây chính xác hơn.');
  return (
    <Bench
      title={tr('Four values, two orders', 'Bốn giá trị, hai thứ tự')}
      task={tr(
        "Step through the book's example, then change the values until the tree is the worse order.",
        'Chạy từng bước ví dụ của sách, rồi thay đổi các giá trị cho đến khi cây là thứ tự tệ hơn.',
      )}
      foot={tr(
        "Every addition is rounded to the book's 5-bit format (Fig. A.4, with denormals), to nearest with ties to even. Its largest value is 3.5; it has no infinity, so a sum beyond 3.5 is clamped.",
        'Mỗi phép cộng được làm tròn về định dạng 5-bit của sách (Fig. A.4, có denormal), về gần nhất với hòa thì về số chẵn. Giá trị lớn nhất của nó là 3.5; nó không có vô cực, nên một tổng vượt quá 3.5 bị kẹp.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        {vals.map((v, i) => (
          <label className="field" key={i}>
            {'abcd'[i]}
            <select aria-label={tr(`value ${'abcd'[i]}`, `giá trị ${'abcd'[i]}`)} value={v} onChange={(e) => set(i, Number(e.target.value))}>
              {toyOptions.map((o) => (
                <option key={o} value={o}>{show(o)}</option>
              ))}
            </select>
          </label>
        ))}
        <button onClick={() => { setVals(BOOK_FOUR); setN(0); setVerdict(null); }}><Translated vi="Các giá trị của sách">The book's values</Translated></button>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setN(0)} disabled={n === 0}><Translated vi="Làm lại từ đầu">Restart</Translated></button>
        <button className="primary" onClick={() => setN(Math.min(3, n + 1))} disabled={n >= 3}><Translated vi="Bước tiếp">Step</Translated></button>
        <span className="muted" aria-live="polite"><Translated vi={<>phép cộng {n} trên 3 ở mỗi thứ tự</>}><>addition {n} of 3 in each order</></Translated></span>
      </div>
      <div className="cols">
        <div className="panel" data-order="sequential">
          <b><Translated vi="Theo thứ tự: ((a + b) + c) + d">In order: ((a + b) + c) + d</Translated></b>
          <ol style={{ margin: '0.4rem 0 0', paddingLeft: '1.3rem' }}>
            {seq.adds.map((a, k) => <AddRow key={k} a={a} k={k} shown={k < n} />)}
          </ol>
        </div>
        <div className="panel" data-order="tree">
          <b><Translated vi="Dạng cây: (a + b) + (c + d)">As a tree: (a + b) + (c + d)</Translated></b>
          <ol style={{ margin: '0.4rem 0 0', paddingLeft: '1.3rem' }}>
            {tree.adds.map((a, k) => <AddRow key={k} a={a} k={k} shown={k < n} />)}
          </ol>
        </div>
      </div>
      {n >= 3 ? (
        <Translated
          vi={
            <p data-summary style={{ margin: '0.8rem 0 0' }}>
              Tổng chính xác <b>{show(seq.exact)}</b>. Theo thứ tự <b data-result="sequential">{show(seq.result)}</b> (lệch {show(seq.error)}), dạng cây{' '}
              <b data-result="tree">{show(tree.result)}</b> (lệch {show(tree.error)}). {better}
            </p>
          }
        >
          <p data-summary style={{ margin: '0.8rem 0 0' }}>
            Exact sum <b>{show(seq.exact)}</b>. In order <b data-result="sequential">{show(seq.result)}</b> (off by {show(seq.error)}), as a tree{' '}
            <b data-result="tree">{show(tree.result)}</b> (off by {show(tree.error)}). {better}
          </p>
        </Translated>
      ) : null}
      <div className="row" style={{ marginTop: '0.9rem' }}>
        <button onClick={check}><Translated vi="Kiểm tra: cây có tệ hơn không?">Check: is the tree worse?</Translated></button>
        {tries >= 3 && verdict !== 'yes' ? (
          <button onClick={() => { setVals(OUR_FOUR); setN(3); setVerdict('yes'); }}><Translated vi="Cho xem một ví dụ">Show one</Translated></button>
        ) : null}
        {verdict === 'yes' ? (
          <span className="chip ok" data-challenge="yes"><Translated vi="Đúng: với các giá trị này, cây là thứ tự kém chính xác hơn.">Yes: the tree is the less accurate order for these values.</Translated></span>
        ) : verdict === 'no' ? (
          <span className="chip fail" data-challenge="no">
            {seq.saturated || tree.saturated ? tr('A sum went past 3.5; keep every partial sum in range.', 'Một tổng đã vượt quá 3.5; hãy giữ mọi tổng riêng phần trong phạm vi.') : tr('Not this time: the tree is at least as accurate.', 'Lần này chưa được: cây chính xác ít nhất là bằng.')}
          </span>
        ) : null}
      </div>
    </Bench>
  );
}

// ================================================================ thread to data (Exercise 4)
const ORDERS: Ex4Order[] = ['ascending', 'descending', 'shuffled'];

function ThreadToData() {
  const [acc, setAcc] = useState<Acc>('fp16');
  const [order, setOrder] = useState<Ex4Order>('ascending');
  const [pairing, setPairing] = useState<Pairing>('convergent');
  const [base, setBase] = useState<number[]>(EX4_VALUES.fp16);
  const [text, setText] = useState(EX4_VALUES.fp16.join(', '));
  const [err, setErr] = useState('');
  const [k, setK] = useState(0);
  const values = useMemo(() => ex4Order(base, order), [base, order]);
  const trace = useMemo(() => treeTrace(values, pairing, acc), [values, pairing, acc]);
  const other = useMemo(() => treeTrace(values, pairing === 'convergent' ? 'neighbours' : 'convergent', acc), [values, pairing, acc]);
  useEffect(() => setK(0), [values, pairing, acc]);
  const total = trace.steps.length;
  const cur = k > 0 ? trace.steps[k - 1] : undefined;
  const cells = cur ? cur.cells : trace.input;
  const level = cur ? cur.level : 0;
  const active = new Set(trace.levels[level].active);
  const lostSoFar = trace.steps.slice(0, k).reduce((s, x) => s + (Number.isFinite(x.lost) ? x.lost : 0), 0);
  const pickFormat = (a: Acc) => {
    setAcc(a);
    setBase(EX4_VALUES[a as 'fp16' | 'bf16']);
    setText(EX4_VALUES[a as 'fp16' | 'bf16'].join(', '));
    setErr('');
  };
  const apply = () => {
    const parts = text.split(/[\s,]+/).filter(Boolean).map(Number);
    if (parts.length !== 16 || parts.some((x) => !Number.isFinite(x))) {
      setErr(tr(`Give exactly 16 numbers (you gave ${parts.length}).`, `Hãy nhập đúng 16 số (bạn đã nhập ${parts.length}).`));
      return;
    }
    setErr('');
    setBase(parts);
  };
  const endOfLevel = () => {
    let j = k;
    while (j < total && (j === k || trace.steps[j].level === trace.steps[k].level)) j++;
    setK(Math.min(total, j));
  };
  const threadOf = (i: number) => (pairing === 'convergent' ? (i < 8 ? i : null) : i % 2 === 0 ? i / 2 : null);
  const cellCls = (i: number) => {
    if (!cur) return '';
    if (i === cur.dst) return cur.lost !== 0 ? 'bad' : 'hot';
    if (i === cur.src) return 'on';
    return '';
  };
  const describeStep = () => {
    if (!cur)
      return tr(
        `${trace.input.length} values in ${ACC_LABEL[acc]}, ${order}. Eight threads; ${pairing === 'convergent' ? 'thread i owns cell i and first adds cell i + 8 (Fig. 10.8)' : 'thread i owns cell 2i and first adds cell 2i + 1 (Fig. 10.6)'}. Press Step.`,
        `${trace.input.length} giá trị trong ${ACC_LABEL[acc]}, ${order === 'ascending' ? 'tăng dần' : order === 'descending' ? 'giảm dần' : 'xáo trộn'}. Tám luồng; ${pairing === 'convergent' ? 'luồng i sở hữu ô i và trước hết cộng ô i + 8 (Fig. 10.8)' : 'luồng i sở hữu ô 2i và trước hết cộng ô 2i + 1 (Fig. 10.6)'}. Hãy bấm Bước tiếp.`,
      );
    const lost = cur.lost === 0 ? tr('nothing lost', 'không mất gì') : Number.isFinite(cur.lost) ? tr(`${show(cur.lost)} lost`, `mất ${show(cur.lost)}`) : tr('overflow', 'tràn số');
    return tr(
      `Level ${cur.level + 1} (stride ${cur.stride}): thread ${cur.thread} adds cell ${cur.src} to cell ${cur.dst}: ${show(cur.a)} + ${show(cur.b)} = ${show(cur.exact)}, stored ${show(cur.rounded)}; ${lost}.`,
      `Tầng ${cur.level + 1} (bước nhảy ${cur.stride}): luồng ${cur.thread} cộng ô ${cur.src} vào ô ${cur.dst}: ${show(cur.a)} + ${show(cur.b)} = ${show(cur.exact)}, lưu ${show(cur.rounded)}; ${lost}.`,
    );
  };
  return (
    <Bench
      title={tr('Thread to data, with rounding', 'Luồng với dữ liệu, có làm tròn')}
      task={tr(
        'On ascending data, step the convergent tree to the end; then switch to neighbours and compare.',
        'Trên dữ liệu tăng dần, chạy từng bước cây hội tụ đến cuối; rồi chuyển sang ghép cặp lân cận và so sánh.',
      )}
      foot={
        <Translated
          vi={
            <>
              Mảng nằm trong bộ nhớ toàn cục (global memory) và được rút gọn tại chỗ, như ở chương 10. Mỗi phép cộng được làm
              tròn một lần về định dạng đã chọn. Ở mỗi bước, ô mà một luồng ghi được tô đậm và ô nó đọc được tô nhạt; ô được
              ghi chuyển sang đỏ khi tổng đã làm tròn đánh rơi một phần của tổng chính xác.
            </>
          }
        >
          <>
            The array lives in global memory and is reduced in place, as in chapter 10. Each addition is rounded once to the
            chosen format. At each step the cell a thread writes is solid and the cell it reads is shaded; the written cell
            turns red when the rounded sum dropped part of the exact one.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Format', 'Định dạng')}>
          {EX4_FORMATS.map((a) => (
            <button key={a} aria-pressed={acc === a} onClick={() => pickFormat(a)}>{ACC_LABEL[a]}</button>
          ))}
        </span>
        <label className="field">
          <Translated vi="Thứ tự">Order</Translated>
          <select aria-label={tr('Order', 'Thứ tự')} value={order} onChange={(e) => setOrder(e.target.value as Ex4Order)}>
            {ORDERS.map((o) => <option key={o} value={o}>{tr(o, o === 'ascending' ? 'tăng dần' : o === 'descending' ? 'giảm dần' : 'xáo trộn')}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label={tr('Pairing', 'Ghép cặp')}>
          <button aria-pressed={pairing === 'convergent'} onClick={() => setPairing('convergent')}><Translated vi="hội tụ (i, i + stride)">convergent (i, i + stride)</Translated></button>
          <button aria-pressed={pairing === 'neighbours'} onClick={() => setPairing('neighbours')}><Translated vi="lân cận (2i, 2i + stride)">neighbours (2i, 2i + stride)</Translated></button>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={{ flex: '1 1 18rem' }}>
          <Translated vi="16 giá trị (thứ tự bất kỳ; menu Thứ tự sẽ sắp xếp chúng)">The 16 values (any order; the Order menu arranges them)</Translated>
          <input
            type="text"
            aria-label={tr('values', 'các giá trị')}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && apply()}
            style={{ font: 'inherit', fontFamily: 'var(--sans)', padding: '0.3rem 0.4rem', width: '100%' }}
          />
        </label>
        <button onClick={apply}><Translated vi="Dùng các giá trị này">Use these values</Translated></button>
        {err ? <span className="t-fail">{err}</span> : null}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setK(0)} disabled={k === 0}><Translated vi="Làm lại từ đầu">Restart</Translated></button>
        <button onClick={() => setK(k - 1)} disabled={k === 0}><Translated vi="Lùi lại">Back</Translated></button>
        <button className="primary" onClick={() => setK(k + 1)} disabled={k >= total}><Translated vi="Bước tiếp">Step</Translated></button>
        <button onClick={endOfLevel} disabled={k >= total}><Translated vi="Hoàn tất tầng này">Finish this level</Translated></button>
        <button onClick={() => setK(total)} disabled={k >= total}><Translated vi="Chạy đến cuối">Run to the end</Translated></button>
        <span className="muted" aria-live="polite"><Translated vi={<>bước {k} trên {total}</>}><>step {k} of {total}</></Translated></span>
      </div>
      <div className={'narrate' + (cur && cur.lost !== 0 ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {describeStep()}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <div className="mx global" style={{ display: 'grid', gridTemplateColumns: 'repeat(8, minmax(2.2rem, 3.4rem))', rowGap: 4 }} aria-label={tr('Cells', 'Các ô')}>
          {cells.map((v, i) => {
            const t = threadOf(i);
            return (
              <span key={i} className={cellCls(i)} data-cell={i} data-value={v} style={{ height: '2.3rem' }}>
                {t != null ? (
                  <span
                    data-thread={t}
                    data-active={active.has(t) ? '1' : '0'}
                    style={{
                      position: 'absolute', left: 3, top: 1, fontSize: '0.56rem', lineHeight: 1.2,
                      opacity: active.has(t) ? 1 : 0.35,
                      fontWeight: cur?.thread === t ? 700 : 400,
                      textDecoration: cur?.thread === t ? 'underline' : 'none',
                    }}
                  >
                    t{t}
                  </span>
                ) : null}
                {show(v, 5)}
                <sub>{i}</sub>
              </span>
            );
          })}
        </div>
      </div>
      <Translated
        vi={
          <p className="muted" style={{ margin: '0.5rem 0 0' }}>
            Hàng trên là các ô 0 đến 7, hàng dưới 8 đến 15; t3 đánh dấu ô mà luồng 3 sở hữu, mờ đi khi luồng nghỉ ở tầng này.
            Đã mất đến giờ: <b data-lost-so-far={lostSoFar}>{show(lostSoFar)}</b>.
          </p>
        }
      >
        <p className="muted" style={{ margin: '0.5rem 0 0' }}>
          Top row cells 0 to 7, bottom row 8 to 15; t3 marks the cell thread 3 owns, faint when the thread is idle at this
          level. Lost so far: <b data-lost-so-far={lostSoFar}>{show(lostSoFar)}</b>.
        </p>
      </Translated>
      {k >= total ? (
        <table className="data" style={{ marginTop: '0.8rem', background: 'var(--page)' }} data-summary>
          <thead>
            <tr><th><Translated vi="Ghép cặp">Pairing</Translated></th><th><Translated vi="Kết quả">Result</Translated></th><th><Translated vi="Chính xác">Exact</Translated></th><th><Translated vi="Lệch">Off by</Translated></th></tr>
          </thead>
          <tbody>
            {[trace, other].map((t) => (
              <tr key={t.pairing} data-pairing={t.pairing}>
                <td>{t.pairing === 'convergent' ? tr('convergent (Fig. 10.8)', 'hội tụ (Fig. 10.8)') : tr('neighbours (Fig. 10.6)', 'lân cận (Fig. 10.6)')}</td>
                <td data-k="result">{show(t.result)}</td>
                <td>{show(t.exact)}</td>
                <td data-k="error">{show(t.error)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </Bench>
  );
}

// ================================================================ the summation explorer
const LOGN_MIN = 4;
const LOGN_MAX = 20;
const ACCS: Acc[] = ['fp32', 'fp16', 'bf16'];

function RunningPlot({ points, acc }: { points: { i: number; acc: number; exact: number }[]; acc: Acc }) {
  const W = 360;
  const H = 210;
  const L = 52;
  const B = 28;
  const n = points[points.length - 1].i;
  const finite = points.map((p) => p.acc).filter(Number.isFinite);
  const top = Math.max(...points.map((p) => p.exact), ...finite, 1e-30);
  const bottom = Math.min(0, ...points.map((p) => p.exact), ...finite);
  const x = (i: number) => L + (i / n) * (W - L - 8);
  const y = (v: number) => 10 + (H - B - 10) * (1 - (Math.min(v, top) - bottom) / (top - bottom || 1));
  const line = (pick: (p: (typeof points)[number]) => number) =>
    points.map((p) => `${x(p.i).toFixed(1)},${y(Number.isFinite(pick(p)) ? pick(p) : top).toFixed(1)}`).join(' ');
  const last = points[points.length - 1];
  const overflowAt = points.find((p) => !Number.isFinite(p.acc));
  return (
    <figure style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Running sum of the loop against the exact running sum', 'Tổng đang chạy của vòng lặp so với tổng đang chạy chính xác')} style={{ width: '100%', display: 'block' }} data-final-acc={last.acc} data-final-exact={last.exact}>
        <line x1={L} x2={L} y1={10} y2={H - B} stroke="var(--rule)" />
        <line x1={L} x2={W - 8} y1={H - B} y2={H - B} stroke="var(--rule)" />
        <text x={L - 6} y={14} fontSize="10.5" textAnchor="end">{show(top, 3)}</text>
        <text x={L - 6} y={H - B} fontSize="10.5" textAnchor="end">{show(bottom, 3)}</text>
        <text x={L} y={H - 10} fontSize="10.5">0</text>
        <text x={W - 8} y={H - 10} fontSize="10.5" textAnchor="end"><Translated vi={<>đã cộng {n} giá trị</>}><>{n} values added</></Translated></text>
        <polyline points={line((p) => p.exact)} fill="none" stroke="var(--ink-3)" strokeWidth="1.5" strokeDasharray="5 4" />
        <polyline points={line((p) => p.acc)} fill="none" stroke="var(--ink)" strokeWidth="2" />
        <g fontSize="10.5">
          <rect x={L + 4} y={8} width={170} height={38} rx="3" fill="var(--bench)" opacity="0.9" />
          <line x1={L + 10} x2={L + 30} y1={20} y2={20} stroke="var(--ink)" strokeWidth="2" />
          <text x={L + 34} y={23}><Translated vi={<>vòng lặp {ACC_LABEL[acc]}: {Number.isFinite(last.acc) ? show(last.acc, 6) : 'tràn số'}</>}><>{ACC_LABEL[acc]} loop: {Number.isFinite(last.acc) ? show(last.acc, 6) : 'overflow'}</></Translated></text>
          <line x1={L + 10} x2={L + 30} y1={36} y2={36} stroke="var(--ink-3)" strokeWidth="1.5" strokeDasharray="5 4" />
          <text x={L + 34} y={39} style={{ fill: 'var(--ink-2)' }}><Translated vi={<>chính xác: {show(last.exact, 6)}</>}><>exact: {show(last.exact, 6)}</></Translated></text>
        </g>
        {overflowAt ? <text x={x(overflowAt.i) + 4} y={56} fontSize="10.5" style={{ fill: 'var(--fail)' }}>{tr('overflow here', 'tràn ở đây')}</text> : null}
      </svg>
      <figcaption className="muted" style={{ fontSize: '0.85rem' }}><Translated vi="Tổng đang chạy của vòng lặp tuần tự (nét liền) so với tổng đang chạy chính xác (nét đứt).">The sequential loop's running sum (solid) against the exact running sum (dashed).</Translated></figcaption>
    </figure>
  );
}

function ChunkPlot({ sweep, c, active }: { sweep: ChunkPoint[] | null; c: number; active: boolean }) {
  const W = 360;
  const H = 210;
  const L = 52;
  const B = 28;
  if (!sweep) return <p className="muted" data-sweep="pending"><Translated vi="Đang tính sai số ở mọi c…">Computing the error at every c…</Translated></p>;
  const vals = sweep.flatMap((p) => [p.plain.ulps, p.kahan.ulps]).filter((v) => Number.isFinite(v) && v > 0);
  const lo = Math.floor(Math.log10(Math.max(1e-3, Math.min(...vals, 1))));
  const hi = Math.max(lo + 1, Math.ceil(Math.log10(Math.max(...vals, 1))));
  const maxK = Math.log2(sweep[sweep.length - 1].c) || 1;
  const x = (cc: number) => L + (Math.log2(cc) / maxK) * (W - L - 12);
  const y = (u: number) => {
    if (!Number.isFinite(u)) return 6;
    const v = Math.log10(Math.max(u, 10 ** lo));
    return 10 + (H - B - 10) * (1 - (v - lo) / (hi - lo));
  };
  const line = (pick: (p: ChunkPoint) => number) => sweep.map((p) => `${x(p.c).toFixed(1)},${y(pick(p)).toFixed(1)}`).join(' ');
  return (
    <figure style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Error against the number of values per thread', 'Sai số theo số giá trị mỗi luồng')} style={{ width: '100%', display: 'block' }} data-sweep="ready" data-points={JSON.stringify(sweep.map((p) => [p.c, p.plain.ulps, p.kahan.ulps]))}>
        <line x1={L} x2={L} y1={10} y2={H - B} stroke="var(--rule)" />
        <line x1={L} x2={W - 12} y1={H - B} y2={H - B} stroke="var(--rule)" />
        {Array.from({ length: hi - lo + 1 }, (_, j) => lo + j).map((e) => (
          <text key={e} x={L - 6} y={y(10 ** e) + 3} fontSize="10.5" textAnchor="end">{e >= -3 && e <= 4 ? String(10 ** e) : `1e${e}`}</text>
        ))}
        <text x={4} y={9} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>ULPs</text>
        <text x={L - 6} y={H - 10} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>c =</text>
        {sweep.filter((p, j) => j % 2 === 0 || j === sweep.length - 1).map((p) => (
          <text key={p.c} x={x(p.c)} y={H - 10} fontSize="10.5" textAnchor="middle">{p.c}</text>
        ))}
        {active && c <= sweep[sweep.length - 1].c ? <line x1={x(c)} x2={x(c)} y1={10} y2={H - B} stroke="var(--ink-2)" strokeDasharray="2 3" data-c={c} /> : null}
        <polyline points={line((p) => p.plain.ulps)} fill="none" stroke="var(--ink-3)" strokeWidth="2.5" />
        <polyline points={line((p) => p.kahan.ulps)} fill="none" stroke="var(--ink)" strokeWidth="2" />
        <g fontSize="10.5">
          <rect x={L + 4} y={8} width={250} height={18} rx="3" fill="var(--bench)" opacity="0.9" />
          <line x1={L + 10} x2={L + 30} y1={18} y2={18} stroke="var(--ink-3)" strokeWidth="2.5" />
          <text x={L + 34} y={21}>{tr('plain per thread', 'thường mỗi luồng')}</text>
          <line x1={L + 130} x2={L + 150} y1={18} y2={18} stroke="var(--ink)" strokeWidth="2" />
          <text x={L + 154} y={21}>{tr('Kahan per thread', 'Kahan mỗi luồng')}</text>
        </g>
        {sweep.some((p) => !Number.isFinite(p.plain.ulps) || !Number.isFinite(p.kahan.ulps)) ? (
          <text x={W - 14} y={36} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--fail)' }}>{tr('top edge: overflow', 'mép trên: tràn số')}</text>
        ) : null}
      </svg>
      <figcaption className="muted" style={{ fontSize: '0.85rem' }}>
        <Translated vi="Sai số của tổng chia đoạn theo c, số giá trị mỗi luồng cộng trước cây (thang log; c = 1 là cây thường).">
          Error of the chunked sum against c, the values each thread adds before the tree (log scales; c = 1 is the plain tree).
        </Translated>
      </figcaption>
    </figure>
  );
}

function Explorer() {
  const [kind, setKind] = useState<DataKind>('uniform');
  const [logn, setLogn] = useState(16);
  const [acc, setAcc] = useState<Acc>('fp32');
  const [method, setMethod] = useState<Method>('sequential');
  const [logc, setLogc] = useState(6);
  const n = 2 ** logn;
  const c = 2 ** logc;
  const chunked = method === 'chunked' || method === 'kahan-chunked';
  const data = useMemo(() => generate(kind, n), [kind, n]);
  const ref = useMemo(() => reference(data), [data]);
  const scale = useMemo(() => absSum(data), [data]);
  const e = useMemo(() => measure(data, { method, acc, chunk: c }, ref, scale), [data, method, acc, c, ref, scale]);
  const running = useMemo(() => runningTrace(data, acc), [data, acc]);
  const sweepKey = `${kind}|${n}|${acc}`;
  const [sweep, setSweep] = useState<{ key: string; points: ChunkPoint[] } | null>(null);
  useEffect(() => {
    // The sweep runs 13 chunk lengths twice over the data; let the rest of the bench paint first.
    const id = window.setTimeout(() => setSweep({ key: sweepKey, points: chunkSweep(data, acc) }), 30);
    return () => window.clearTimeout(id);
  }, [sweepKey, data, acc]);
  const points = sweep && sweep.key === sweepKey ? sweep.points : null;
  const fmt = ACC_FORMAT[acc];
  return (
    <Bench
      title={tr('Summation order and accumulator width', 'Thứ tự cộng và độ rộng bộ tích lũy')}
      task={tr(
        'Find the FP16 stall, then compare the loop, the tree and Kahan on each data set.',
        'Tìm chỗ FP16 bị kẹt, rồi so sánh vòng lặp, cây và Kahan trên từng tập dữ liệu.',
      )}
      foot={
        <Translated
          vi={
            <>
              Dữ liệu được sinh với seed cố định và lưu bằng FP32; tham chiếu là một tổng có bù trong binary64 (của Neumaier).
              Sai số tương đối được đo so với tổng các |x|. ULP là đơn vị ở vị trí cuối của định dạng bộ tích lũy tại giá trị
              tham chiếu. Giá trị lớn nhất của {ACC_LABEL[acc]} là {show(maxFinite(fmt))}.
            </>
          }
        >
          <>
            Data are seeded and stored in FP32; the reference is a compensated sum in binary64 (Neumaier's). Relative error is
            measured against the sum of |x|. ULPs are units in the last place of the accumulator format at the reference
            value. {ACC_LABEL[acc]}'s largest value is {show(maxFinite(fmt))}.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="Dữ liệu">Data</Translated>
          <select aria-label={tr('Data', 'Dữ liệu')} value={kind} onChange={(ev) => setKind(ev.target.value as DataKind)}>
            {DATA.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
          </select>
        </label>
        <label className="field">
          n = {n.toLocaleString('en-US')}
          <input type="range" aria-label={tr('n (log2)', 'n (log2)')} min={LOGN_MIN} max={LOGN_MAX} value={logn} onChange={(ev) => setLogn(Number(ev.target.value))} />
        </label>
        <span className="seg" role="group" aria-label={tr('Accumulator', 'Bộ tích lũy')}>
          {ACCS.map((a) => (
            <button key={a} aria-pressed={acc === a} onClick={() => setAcc(a)}>{ACC_LABEL[a]}</button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Phương pháp">Method</Translated>
          <select aria-label={tr('Method', 'Phương pháp')} value={method} onChange={(ev) => setMethod(ev.target.value as Method)}>
            {METHODS.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
        </label>
        {chunked ? (
          <label className="field">
            <Translated vi={<>c = {c} giá trị mỗi luồng ({Math.ceil(n / c).toLocaleString('en-US')} luồng)</>}>
              <>c = {c} values per thread ({Math.ceil(n / c).toLocaleString('en-US')} threads)</>
            </Translated>
            <input type="range" aria-label={tr('c (log2)', 'c (log2)')} min={0} max={12} value={logc} onChange={(ev) => setLogc(Number(ev.target.value))} />
          </label>
        ) : null}
        <span className="muted">{DATA.find((d) => d.id === kind)!.note}</span>
      </div>
      <div className="stats" aria-label={tr('Result', 'Kết quả')} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(9rem, 1fr))', gap: '0.5rem 1rem', marginBottom: '1rem', fontFamily: 'var(--sans)', fontSize: '0.9rem' }}>
        {[
          { k: 'result', label: tr(`Result (${ACC_LABEL[acc]})`, `Kết quả (${ACC_LABEL[acc]})`), v: e.result, text: Number.isFinite(e.result) ? show(e.result, 9) : e.result > 0 ? tr('+inf (overflow)', '+inf (tràn số)') : tr('−inf (overflow)', '−inf (tràn số)') },
          { k: 'reference', label: tr('Reference', 'Tham chiếu'), v: e.reference, text: show(e.reference, 12) },
          { k: 'abs', label: tr('Absolute error', 'Sai số tuyệt đối'), v: e.abs, text: e.overflow ? tr('overflow', 'tràn số') : show(e.abs, 4) },
          { k: 'rel', label: tr('Relative error', 'Sai số tương đối'), v: e.rel, text: pct(e.rel) },
          { k: 'ulps', label: tr(`Error in ULPs of ${ACC_LABEL[acc]}`, `Sai số tính bằng ULP của ${ACC_LABEL[acc]}`), v: e.ulps, text: ulpText(e.ulps) },
          { k: 'scale', label: tr('Sum of |x|', 'Tổng các |x|'), v: scale, text: show(scale, 9) },
        ].map((r) => (
          <div key={r.k} className="panel" style={{ padding: '0.4rem 0.6rem' }}>
            <div className="muted" style={{ fontSize: '0.78rem' }}>{r.label}</div>
            <div data-k={r.k} data-v={String(r.v)} style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>{r.text}</div>
          </div>
        ))}
      </div>
      <div className="cols">
        <RunningPlot points={running} acc={acc} />
        <ChunkPlot sweep={points} c={c} active={chunked} />
      </div>
    </Bench>
  );
}

function HalfLoopReveal({ v }: { v: number }) {
  const u = useMemo(() => generate('uniform', 65536), []);
  const got = sumSequential(u, 'fp16');
  const exact = reference(u);
  const close = Math.abs(v - got) <= 0.1 * got;
  return (
    <>
      <Translated
        vi={
          <p>
            Vòng lặp FP16 của mô hình trả về <b data-reveal="half-loop">{show(got)}</b>; tổng chính xác của 65,536 giá trị này
            là {show(exact, 7)}. {close ? 'Bạn đã đoán trước được chỗ kẹt.' : v > 10000 ? 'Phần lớn mọi người chờ đợi một giá trị gần tổng đúng.' : ''}
          </p>
        }
      >
        <p>
          The model's FP16 loop returns <b data-reveal="half-loop">{show(got)}</b>; the exact sum of these 65,536 values is{' '}
          {show(exact, 7)}. {close ? 'You saw the stall coming.' : v > 10000 ? 'Most people expect something near the true sum.' : ''}
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Dưới 2048, tổng đang chạy tăng dần: từ 1024 trở đi, các giá trị FP16 cách nhau 1, nên mỗi phép cộng làm tròn lên
            một đơn vị hoặc xuống thành không có gì. Ở 2048 khoảng cách thành 2. Mọi giá trị dưới 1 đều nhỏ hơn một nửa khoảng
            đó và bị làm tròn mất; một giá trị đúng bằng 1 là trường hợp hòa và quay về 2048, số có bit mantissa cuối bằng 0.
            Từ đó trở đi vòng lặp không cộng thêm được gì, khoảng 60,000 lần. Hãy đặt trình khám phá về FP16 và xem biểu đồ
            bên trái.
          </p>
        }
      >
        <p>
          Below 2048 the running sum grows: from 1024 on, FP16's values are 1 apart, so each addition rounds up by one or
          down to nothing. At 2048 the spacing becomes 2. Every value below 1 is less than half of that and rounds away; a
          value of exactly 1 is a tie that goes back to 2048, whose last mantissa bit is 0. From then on the loop adds
          nothing, about 60,000 times. Set the explorer to FP16 and look at the left plot.
        </p>
      </Translated>
    </>
  );
}

// ================================================================ Kahan, four ways
const VERDICTS: Verdict[] = ['kahan', 'plain', 'worse'];

function BugHunt() {
  const file = useLabFile('appA/snippets/kahan.h');
  const [marks, setMarks] = useState<Record<number, Verdict | undefined>>({});
  const [ran, setRan] = useState(false);
  const results = useMemo(() => (ran ? bugHunt(bugHuntValues(), 'fp32') : null), [ran]);
  const uniform = useMemo(() => (ran ? bugHunt(generate('uniform', 65536), 'fp32') : null), [ran]);
  const all = BUG_HUNT.every((b) => marks[b.version]);
  const right = results ? BUG_HUNT.filter((b, j) => marks[b.version] === results.runs[j].verdict).length : 0;
  return (
    <Bench
      title={tr('Kahan, four ways', 'Kahan, bốn cách')}
      task={tr(
        'Mark each version before you run them: like Kahan, like the plain sum, or worse than plain?',
        'Đánh dấu từng phiên bản trước khi chạy: giống Kahan, giống tổng thường, hay tệ hơn tổng thường?',
      )}
      foot={
        <Translated
          vi={
            <>
              Mã là <code>labs/appA/snippets/kahan.h</code>, được tải từ repository. Một unit test biên dịch nó bằng trình biên
              dịch của host và kiểm tra rằng mọi phiên bản cho ra đúng số float của mô hình này, từng bit. Dữ liệu: 1.0, rồi
              65,535 giá trị 2⁻²⁵, bằng FP32.
            </>
          }
        >
          <>
            The code is <code>labs/appA/snippets/kahan.h</code>, loaded from the repository. A unit test compiles it with the
            host compiler and checks that every version gives this model's float, bit for bit. Data: 1.0, then 65,535 values
            of 2⁻²⁵, in FP32.
          </>
        </Translated>
      }
    >
      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 24rem), 1fr))' }}>
        {BUG_HUNT.map((b, j) => {
          const src = file ? functionSource(file, `version${b.version}`) : null; // i18n-ignore: a function name in the lab file
          const run = results?.runs[j];
          const mark = marks[b.version];
          return (
            <div key={b.version} className="panel" data-version={b.version}>
              <b><Translated vi={<>Phiên bản {b.version}</>}><>Version {b.version}</></Translated></b>
              {src ? <Code tabs={[{ label: tr(`version${b.version}`, `version${b.version}`), code: src.text.replace(/^\/\/ Version \d+\n/, '') }]} /> : <p className="muted"><Translated vi="Đang tải labs/appA/snippets/kahan.h…">Loading labs/appA/snippets/kahan.h…</Translated></p>}
              <div className="opts" role="radiogroup" aria-label={tr(`Version ${b.version}`, `Phiên bản ${b.version}`)} style={{ display: 'grid', gap: '0.2rem', marginTop: '0.4rem' }}>
                {VERDICTS.map((v) => (
                  <label key={v} className="check">
                    <input
                      type="radio"
                      name={`kahan-v${b.version}`} // i18n-ignore: a form control name
                      checked={mark === v}
                      onChange={() => {
                        setMarks({ ...marks, [b.version]: v });
                        setRan(false);
                      }}
                    />
                    {VERDICT_LABEL[v]}
                  </label>
                ))}
              </div>
              {run ? (
                <p style={{ margin: '0.5rem 0 0' }} data-verdict={run.verdict} data-mark={mark === run.verdict ? 'right' : 'wrong'}>
                  <span className={'chip ' + (mark === run.verdict ? 'ok' : 'fail')}>{mark === run.verdict ? tr('right', 'đúng') : tr('not quite', 'chưa đúng')}</span>{' '}
                  <Translated
                    vi={
                      <>
                        {VERDICT_LABEL[run.verdict]}: kết quả {show(run.result.result, 9)}, lệch {ulpText(run.result.ulps)} ULP
                        {uniform ? <> (trên dữ liệu phân bố đều: {ulpText(uniform.runs[j].result.ulps)} ULP)</> : null}.
                      </>
                    }
                  >
                    <>
                      {VERDICT_LABEL[run.verdict]}: result {show(run.result.result, 9)}, off by {ulpText(run.result.ulps)} ULPs
                      {uniform ? <> (on uniform data: {ulpText(uniform.runs[j].result.ulps)} ULPs)</> : null}.
                    </>
                  </Translated>
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="row" style={{ marginTop: '0.9rem' }}>
        <button className="primary" disabled={!all} onClick={() => setRan(true)}><Translated vi="Chạy bốn phiên bản">Run the four versions</Translated></button>
        {results ? (
          <span data-score={right}>
            <Translated
              vi={
                <>
                  Đúng {right} trên 4. Vòng lặp thường: kết quả {show(results.plain.result, 9)}, lệch {ulpText(results.plain.ulps)} ULP
                  {uniform ? <> ({ulpText(uniform.plain.ulps)} trên dữ liệu phân bố đều)</> : null}.
                </>
              }
            >
              <>
                {right} of 4 right. The plain loop: result {show(results.plain.result, 9)}, off by {ulpText(results.plain.ulps)} ULPs
                {uniform ? <> ({ulpText(uniform.plain.ulps)} on uniform data)</> : null}.
              </>
            </Translated>
          </span>
        ) : (
          <span className="muted">{all ? tr('Ready.', 'Sẵn sàng.') : tr('Mark all four first.', 'Hãy đánh dấu cả bốn trước.')}</span>
        )}
      </div>
    </Bench>
  );
}

// ================================================================ questions
export const questionsSA6: Record<string, Question> = {
  'appA/A.6/q-ex4': {
    kind: 'choice',
    id: 'appA/A.6/q-ex4',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 4: mảng của một sinh viên luôn được sắp xếp từ nhỏ đến lớn. Để tránh phân kỳ luồng (divergence), anh ấy
            rút gọn nó bằng kernel hội tụ của chương 10, và tổng kém chính xác hơn anh ấy chờ đợi. Vì sao?
          </>
        }
      >
        <>
          Exercise 4: a student's array is always sorted from smallest to largest. To avoid divergence he reduces it with
          chapter 10's convergent kernel, and the sum is less accurate than he expected. Why?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Tầng đầu tiên của nó cộng phần tử i + n/2 vào phần tử i, nên giá trị nhỏ gặp giá trị lớn và biến mất.">
            Its first level adds element i + n/2 to element i, so small values meet large ones and vanish.
          </Translated>
        ),
        correct: true,
        why: (
          <Translated vi="Việc sắp xếp đặt các độ lớn tương tự cạnh nhau; cách ghép cặp này vươn ngang qua mảng và phá bỏ điều đó.">
            Sorting put similar magnitudes side by side; this pairing reaches across the array and undoes that.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không có phân kỳ, một warp cộng theo thứ tự khác, và dữ liệu đã sắp xếp làm tròn kém theo thứ tự đó.">
            Without divergence a warp adds in a different order, and sorted data rounds badly in that order.
          </Translated>
        ),
        why: (
          <Translated vi="Phân kỳ quyết định luồng nào chạy trong một vòng lặp, không quyết định phần tử nào được ghép cặp.">
            Divergence decides which threads run in an iteration, not which elements are paired.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Dữ liệu đã sắp xếp làm cây sâu hơn, nên mỗi giá trị đi qua nhiều lần làm tròn hơn.">
            Sorted data makes the tree deeper, so each value goes through more roundings.
          </Translated>
        ),
        why: <Translated vi="Độ sâu là log2 n với mọi dữ liệu.">The depth is log2 n for any data.</Translated>,
      },
      {
        label: (
          <Translated vi="Việc sắp xếp làm tròn mỗi giá trị thêm một lần trước khi phép rút gọn bắt đầu.">
            Sorting rounds every value once more before the reduction begins.
          </Translated>
        ),
        why: (
          <Translated vi="Sắp xếp di chuyển các giá trị; nó không thay đổi chúng.">
            Sorting moves values; it does not change them.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Văn bản của sách trỏ tới "thuật toán của Fig. A.4", một bảng các số biểu diễn được. Cách đọc của chúng tôi là
            kernel hội tụ của chương 10 (Fig. 10.8), kernel được viết để tránh phân kỳ. Tầng đầu tiên của nó ghép phần tử i
            với phần tử i + n/2: trên dữ liệu tăng dần, một giá trị nhỏ với một giá trị lớn hơn. Cách ghép cặp lân cận của
            Fig. 10.6 cộng các giá trị cùng cỡ với nhau và giữ được lợi ích của việc sắp xếp; trong trình chạy từng bước nó
            chính xác trên cùng dữ liệu đó.
          </p>
        }
      >
        <p>
          The book's text points to "the algorithm of Fig. A.4", a table of representable numbers. Our reading is chapter 10's
          convergent kernel (Fig. 10.8), the one written to avoid divergence. Its first level pairs element i with element i
          + n/2: on ascending data, a small value with a larger one. The neighbour pairing of Fig. 10.6 adds like with like
          and keeps the benefit of sorting; in the stepper it is exact on the same data.
        </p>
      </Translated>
    ),
    source: 'book Exercise 4, p. 600, with ch. 10 Figs. 10.6 and 10.8 (pp. 229–233); our model (fpsum.treeTrace)',
    vi: { source: 'sách Bài tập 4, p. 600, cùng ch. 10 Figs. 10.6 và 10.8 (pp. 229–233); mô hình của khóa học này (fpsum.treeTrace)' },
  },
  'appA/A.6/q-sign': {
    kind: 'choice',
    id: 'appA/A.6/q-sign',
    prompt: (
      <Translated vi={<>Dữ liệu có cả hai dấu và độ lớn chênh lệch rất nhiều. Bạn nên sắp xếp nó thế nào trước một phép cộng tuần tự hoặc dạng cây?</>}>
        <>The data has both signs and wildly different magnitudes. How should you sort it before a sequential or tree sum?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Theo độ lớn, |x| nhỏ nhất trước.">By magnitude, smallest |x| first.</Translated>, correct: true },
      {
        label: <Translated vi="Theo giá trị, tăng dần.">By value, ascending.</Translated>,
        why: (
          <Translated vi="Khi đó các giá trị âm lớn nhất đến trước, và các giá trị nhỏ ở giữa gặp một tổng đang chạy đã lớn.">
            Then the largest negative values come first, and the small values in the middle meet a running sum that is already large.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Theo giá trị, giảm dần.">By value, descending.</Translated>,
        why: (
          <Translated vi="Các giá trị dương lớn nhất đến trước, và mọi giá trị nhỏ gặp một tổng đang chạy lớn.">
            The largest positive values come first, and every small value meets a large running sum.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Một khi có hai dấu thì sắp xếp không giúp được gì.">Sorting cannot help once there are two signs.</Translated>,
        why: (
          <Translated vi="Nó vẫn đưa các giá trị cùng cỡ lại gần nhau; chỉ có khóa sắp xếp thay đổi.">
            It still brings values of similar size together; only the key changes.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sách nói phải tính đến dấu (p. 594). Điều quyết định một số hạng có biến mất hay không là độ lớn của nó so với thứ
            nó được cộng vào, nên khóa sắp xếp là |x|. Trong mô hình, trên dữ liệu "cancelling", vòng lặp sắp xếp theo độ lớn
            chính xác hơn nhiều lần so với cùng vòng lặp đó sắp xếp theo giá trị.
          </p>
        }
      >
        <p>
          The book says signs must be taken into account (p. 594). What decides whether an addend disappears is its size
          relative to what it is added to, so the key is |x|. In the model, on the "cancelling" data, the loop sorted by
          magnitude is many times more accurate than the same loop sorted by value.
        </p>
      </Translated>
    ),
    source: 'book §A.6, p. 594; our model (fpsum.test.ts, q-sign)',
    vi: { source: 'sách §A.6, p. 594; mô hình của khóa học này (fpsum.test.ts, q-sign)' },
  },
  'appA/A.6/q-half-tree': {
    kind: 'choice',
    id: 'appA/A.6/q-half-tree',
    prompt: (
      <Translated
        vi={
          <>
            Một bộ tích lũy FP16 cộng 262,144 giá trị lấy đều từ [0, 1): tổng đúng khoảng 131,000, gấp đôi giá trị lớn nhất của
            FP16. Một lần dạng vòng lặp, một lần dạng cây theo cặp. Kết quả là gì?
          </>
        }
      >
        <>
          An FP16 accumulator sums 262,144 values drawn uniformly from [0, 1): the true sum is about 131,000, twice FP16's
          largest value. Once as a loop, once as a pairwise tree. What comes out?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Vòng lặp trả về 2048; cây trả về vô cực.">The loop returns 2048; the tree returns infinity.</Translated>, correct: true },
      {
        label: <Translated vi="Cả hai trả về vô cực.">Both return infinity.</Translated>,
        why: (
          <Translated vi="Vòng lặp không bao giờ tới gần đỉnh của phạm vi: nó kẹt ở 2048, từ rất lâu trước đó.">
            The loop never gets near the top of the range: it stalls at 2048, long before.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Cả hai trả về khoảng 65,504, giá trị lớn nhất.">Both return about 65,504, the largest value.</Translated>,
        why: (
          <Translated vi="Làm tròn về gần nhất tràn thành vô cực chứ không kẹp lại.">
            Round to nearest overflows to infinity rather than clamping.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Vòng lặp trả về vô cực; cây trả về khoảng 131,000.">The loop returns infinity; the tree about 131,000.</Translated>,
        why: <Translated vi="Không giá trị FP16 nào gần 131,000 cả.">No FP16 value is anywhere near 131,000.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Vòng lặp kẹt ở chỗ khoảng cách của FP16 đạt 2, tại 2048, và ở yên đó: một đáp án hữu hạn, sai, và im lặng. Cây giữ
            mỗi tổng riêng phần chính xác, nên các phép cộng cuối của nó thực sự tạo ra các số lớn hơn giá trị lớn nhất của
            FP16, 65,504, và làm tròn về gần nhất biến chúng thành vô cực (mọi giá trị từ 65,520 trở lên). Thất bại của cây thì
            ồn ào; của vòng lặp thì không. Hãy thử trong trình khám phá với n = 2¹⁸.
          </p>
        }
      >
        <p>
          The loop stalls where FP16's spacing reaches 2, at 2048, and stays there: a quiet, wrong, finite answer. The tree
          keeps each partial sum accurate, so its last additions really do produce numbers above FP16's largest value,
          65,504, and round to nearest turns them into infinity (anything from 65,520 up). The tree's failure is loud; the
          loop's is not. Try it in the explorer with n = 2¹⁸.
        </p>
      </Translated>
    ),
    source: 'our model (fpsum.test.ts, q-half-tree); FP16 range from appA-facts A2',
    vi: { source: 'mô hình của khóa học này (fpsum.test.ts, q-half-tree); phạm vi FP16 từ appA-facts A2' },
  },
  'appA/A.6/q-kahan16': {
    kind: 'choice',
    id: 'appA/A.6/q-kahan16',
    prompt: (
      <Translated vi={<>Cộng 65,536 giá trị lấy đều từ [0, 1): cái nào chính xác hơn, vòng lặp Kahan bằng FP16 hay một vòng lặp thường bằng FP32?</>}>
        <>Summing 65,536 values drawn uniformly from [0, 1): which is more accurate, Kahan's loop in FP16 or a plain loop in FP32?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Vòng lặp FP32 thường.">The plain FP32 loop.</Translated>, correct: true },
      {
        label: <Translated vi="Kahan bằng FP16.">Kahan in FP16.</Translated>,
        why: (
          <Translated vi="Kahan lấy lại được phần mỗi phép cộng đánh rơi, nhưng không thể khiến FP16 chứa nhiều hơn 11 bit có nghĩa.">
            Kahan recovers what each addition drops, but it cannot make FP16 hold more than 11 significant bits.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Chúng gần như ngang nhau.">They are about the same.</Translated>,
        why: <Translated vi="Trong mô hình chúng chênh nhau khoảng hai bậc độ lớn.">In the model they differ by about two orders of magnitude.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong mô hình, Kahan bằng FP16 kết thúc trong khoảng nửa ULP FP16 quanh tổng đúng, tức là tốt nhất mà FP16 có thể
            lưu: khoảng 32,656 so với 32,664. Vòng lặp FP32 thường lệch khoảng 0.08. Phép bù sửa được sai số của các phép
            cộng; độ rộng của bộ tích lũy đặt ra giới hạn dưới. Bộ tích lũy rộng hơn trước, phép bù sau.
          </p>
        }
      >
        <p>
          In the model, Kahan in FP16 ends within half an FP16 ULP of the true sum, which is as good as FP16 can store it:
          about 32,656 against 32,664. The plain FP32 loop is about 0.08 off. Compensation fixes the error of the additions;
          the width of the accumulator sets the floor. Wider accumulator first, compensation second.
        </p>
      </Translated>
    ),
    source: 'our model (fpsum.test.ts, q-kahan16)',
    vi: { source: 'mô hình của khóa học này (fpsum.test.ts, q-kahan16)' },
  },
  'appA/A.6/q-fastmath': {
    kind: 'choice',
    id: 'appA/A.6/q-fastmath',
    prompt: (
      <Translated vi={<>Một kernel Kahan chạy qua được trên trình giả lập và trên HIP, nhưng trên Metal kết quả của nó không tốt hơn tổng thường. Vì sao?</>}>
        <>A Kahan kernel passes on the emulator and on HIP, but on Metal its result is no better than the plain sum. Why?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Chế độ toán mặc định cho phép trình biên dịch sắp xếp lại (reassociate), và c = (t − sum) − y đã thành 0.">
            The default math mode lets the compiler reassociate, and c = (t − sum) − y became 0.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Metal có thể làm tròn về phía 0, và phép bù không sửa được các phép cộng bị cắt bớt.">
            Metal may round toward zero, and compensation cannot correct truncated additions.
          </Translated>
        ),
        why: (
          <Translated vi="Chế độ làm tròn thay đổi cách mỗi phép cộng làm tròn; nó không xóa câu lệnh tính c. Ở đây phép bù đã biến mất hoàn toàn.">
            A rounding mode changes how each addition rounds; it does not remove the statement that computes c. Here the compensation is gone altogether.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Phép trừ float của Metal không được làm tròn đúng, nên c chứa nhiễu, không phải phần bị mất.">
            {"Metal's float subtraction is not correctly rounded, so c holds noise, not the lost part."}
          </Translated>
        ),
        why: (
          <Translated vi="Metal ghi trong tài liệu rằng + và − được làm tròn đúng ở mọi chế độ.">
            Metal documents + and − as correctly rounded in every mode.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Metal ép denormal về 0, và c gần như luôn là một denormal.">
            Metal flushes denormals to zero, and c is almost always a denormal.
          </Translated>
        ),
        why: (
          <Translated vi="c có cỡ khoảng các bit cuối của tổng, cao hơn nhiều so với phạm vi denormal với những tổng như thế này.">
            c is about the size of the last bits of the sum, far above the denormal range for sums like these.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Cả hai chế độ mặc định khả dĩ, fast và relaxed, đều cho phép sắp xếp lại, và đặc tả của Metal cảnh báo rằng điều
              đó phá vỡ mã dựa vào làm tròn, chẳng hạn (x + 2⁵²) − 2⁵². Hãy đặt <code>#pragma METAL fp math_mode(safe)</code> ở
              đầu tệp kernel. Clang, thứ HIP dùng, không sắp xếp lại trừ khi được bảo, nên bản build HIP giữ được nó.
            </p>
            <p>Trong lúc chương này được thiết kế, GPU Apple của khóa học đã làm đúng như vậy: chế độ mặc định xóa phép bù và pragma khôi phục nó.</p>
          </>
        }
      >
        <>
          <p>
            Both candidate default modes, fast and relaxed, allow reassociation, and Metal's specification warns that it
            breaks code relying on rounding, such as (x + 2⁵²) − 2⁵². Put <code>#pragma METAL fp math_mode(safe)</code> at
            the top of the kernel file. Clang, which HIP uses, does not reassociate unless told to, so the HIP build kept it.
          </p>
          <p>While this chapter was designed, the course's Apple GPU did exactly this: the default mode removed the compensation and the pragma restored it.</p>
        </>
      </Translated>
    ),
    source: `${FACTS} C2, C3 (MSL spec 1.6.3, 8.4), B5 (Clang); measured while designing this chapter`,
    vi: { source: `${FACTS} C2, C3 (đặc tả MSL 1.6.3, 8.4), B5 (Clang); đo trong lúc thiết kế chương này` },
  },
  'appA/A.6/q-coarse': {
    kind: 'choice',
    id: 'appA/A.6/q-coarse',
    prompt: (
      <Translated
        vi={
          <>
            Một hàng bắt đầu bằng một giá trị lớn theo sau là một triệu giá trị nhỏ ("one big, many small" của trình khám phá).
            Bạn tăng hệ số gộp việc của một tổng FP32 thường từ 4 lên 1024 giá trị mỗi luồng. Sai số thay đổi thế nào?
          </>
        }
      >
        <>
          A row starts with one large value followed by a million small ones (the explorer's "one big, many small"). You
          raise the coarsening factor of a plain FP32 sum from 4 to 1024 values per thread. What happens to the error?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nó tăng gần tỉ lệ với c; Kahan bên trong mỗi luồng đưa nó về lại khoảng một ULP.">
            It grows about in proportion to c; Kahan inside each thread brings it back to about one ULP.
          </Translated>
        ),
        correct: true,
        why: (
          <Translated vi="Luồng giữ giá trị lớn cộng khoảng một nghìn giá trị nhỏ vào nó và đánh rơi từng giá trị.">
            The thread that holds the large value adds about a thousand small ones to it and drops each.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không có gì thay đổi: cây ở cuối, chứ không phải các vòng lặp mỗi luồng, quyết định độ chính xác.">
            Nothing changes: the tree at the end, not the per-thread loops, decides how accurate it is.
          </Translated>
        ),
        why: (
          <Translated vi="Cây chỉ thấy các tổng của từng luồng; những gì một luồng đã đánh rơi thì đã mất.">
            The tree only sees the per-thread sums; what a thread dropped is already gone.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó giảm, vì ít luồng hơn thì ít tổng riêng phần hơn đi qua cây.">
            It shrinks, because with fewer threads fewer partial sums go through the tree.
          </Translated>
        ),
        why: (
          <Translated vi="Cây là phần chính xác; chính các vòng lặp dài hơn mới làm mất.">
            The tree is the accurate part; it is the longer loops that lose.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong mô hình, ở 2²⁰ giá trị, sai số đi từ khoảng 3 ULP ở c = 4 lên khoảng 260 ở c = 1024; với Kahan bên trong mỗi
            luồng nó giữ gần 1. Trên dữ liệu ngẫu nhiên không bắt đầu bằng một giá trị lớn, mức tăng nhỏ hơn nhiều. Chi phí
            của phép bù: bốn phép cộng mỗi phần tử thay vì một, đưa phép cộng từ 0.25 lên 1 phép toán mỗi byte, vẫn thấp hơn
            nhiều so với mức một kernel bị giới hạn bởi bộ nhớ cần để trở thành bị giới hạn bởi tính toán trên thiết bị ví dụ
            của sách (khoảng 20). Lab A.1 đo cả hai trên thiết bị của bạn.
          </p>
        }
      >
        <p>
          In the model, at 2²⁰ values, the error goes from about 3 ULPs at c = 4 to about 260 at c = 1024; with Kahan inside
          each thread it stays near 1. On random data that does not start with a large value the growth is much smaller. The
          cost of compensation: four additions per element instead of one, which moves the sum from 0.25 to 1 operation per
          byte, still far below what a memory-bound kernel needs to become compute-bound on the book's example device (about
          20). Lab A.1 measures both on yours.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.10 (pp. 246–248); our model (fpsum.test.ts, q-coarse); ch. 5 §5.1 for the book\'s example ratio',
    vi: { source: 'ch. 10 §10.10 (pp. 246–248); mô hình của khóa học này (fpsum.test.ts, q-coarse); ch. 5 §5.1 cho tỉ lệ trong ví dụ của sách' },
  },
};

// ================================================================ the page
export function SA6() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mục <a href={href('appA', 'A.4')}>A.4</a> kết thúc với việc một phép cộng mất tối đa nửa ULP và một vòng lặp
            gồm các phép cộng mất toàn bộ những gì nó cộng vào. Mục này nói về các tổng dài nơi điều đó xảy ra, và hai thứ bạn
            kiểm soát trong một kernel: thứ tự các phép cộng và bộ tích lũy mà chúng đi vào.
          </p>
        }
      >
        <p>
          Section <a href={href('appA', 'A.4')}>A.4</a> ended with one addition losing at most half a ULP and a loop of
          additions losing everything it adds. This section is about the long sums where that happens, and the two things you
          control in a kernel: the order of the additions and the accumulator they go into.
        </p>
      </Translated>

      <h2>
        <Translated vi="Tổng dài có ở khắp nơi">Long sums are everywhere</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Ví dụ của sách về một tổng dài là tích vô hướng bên trong phép nhân ma trận: mỗi phần tử đầu ra của một GEMM là một
            tích như vậy. Về mặt toán học phép cộng có tính kết hợp, nên thứ tự lẽ ra không quan trọng. Với độ chính xác hữu
            hạn, nó thay đổi kết quả.
          </p>
        }
      >
        <p>
          The book's example of a long sum is the dot product inside matrix multiplication: every element of a GEMM's output
          is one. Mathematically addition is associative, so the order should not matter. In finite precision it changes the
          result.
        </p>
      </Translated>
      <Source section="A.6" pages="593" />
      <Translated
        vi={
          <p>
            Một bước của bộ giải mã (decoder) cộng nhiều hơn thế: mẫu số chuẩn hóa của softmax trên một hàng logit, trung bình
            bình phương mà một phép chuẩn hóa (norm) lấy trên chiều ẩn, tổng tích lũy mà một bộ lấy mẫu top-p xây trên các xác
            suất đã sắp xếp. Mỗi cái là một tổng dài trong một định dạng bộ tích lũy nào đó.
          </p>
        }
      >
        <p>
          One decoder step adds up many more: the softmax normaliser over a row of logits, the mean of squares a norm takes
          over the hidden dimension, the running total a top-p sampler builds over sorted probabilities. Each is a long sum
          in some accumulator format.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Các ví dụ về suy luận là của khóa học này. Ví dụ của sách là tích vô hướng trong phép nhân ma trận.</p>}>
          <p>The inference examples are ours. The book's is the matrix-multiplication dot product.</p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Hai thứ tự, hai đáp án">Two orders, two answers</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách dùng định dạng 5-bit của mục <a href={href('appA', 'A.2')}>A.2</a>, có các giá trị giữa 2 và 3.5 cách nhau
            0.5. Cộng 1, 1, 0.25 và 0.25; tổng chính xác là 2.5. Theo thứ tự, 1 + 1 = 2, rồi 2 + 0.25 = 2.25 không biểu diễn
            được và bị làm tròn về 2, phép cộng cuối cũng vậy: các toán hạng nhỏ biến mất, và vòng lặp trả về 2.0. Dạng cây,
            (1 + 1) + (0.25 + 0.25) = 2 + 0.5 = 2.5, chính xác: hai giá trị nhỏ kết hợp trước thành một thứ đủ lớn để được tính.
          </p>
        }
      >
        <p>
          The book uses the 5-bit format of section <a href={href('appA', 'A.2')}>A.2</a>, whose values between 2 and 3.5
          are 0.5 apart. Add 1, 1, 0.25 and 0.25; the exact sum is 2.5. In order, 1 + 1 = 2, then 2 + 0.25 = 2.25 has no
          representation and rounds back to 2, and so does the last addition: the small operands disappear, and the loop
          returns 2.0. As a tree, (1 + 1) + (0.25 + 0.25) = 2 + 0.5 = 2.5, exactly: the two small values combine first into
          something large enough to count.
        </p>
      </Translated>
      <Source section="A.6" pages="593" />
      <Translated
        vi={
          <p>
            Sách nói thêm rằng một trường hợp gần đó làm cây thành thứ tự kém chính xác hơn, và dừng ở đó. Hãy tìm một trường
            hợp: thay đổi các giá trị trong trình chạy từng bước cho đến khi cây thua.
          </p>
        }
      >
        <p>
          The book adds that a nearby case makes the tree the less accurate order, and leaves it at that. Find one: change
          the values in the stepper until the tree loses.
        </p>
      </Translated>
      <Source section="A.6" pages="594" />
      <FourValues />
      <Translated
        vi={
          <p>
            Sau ba lần thử, trình chạy từng bước đưa ra một trường hợp của khóa học này. Dù bạn tìm được gì, hãy xem cây thua ở
            đâu: một trong các cặp của nó cộng một giá trị nhỏ vào một giá trị lớn trước khi các giá trị nhỏ có cơ hội kết hợp.
          </p>
        }
      >
        <p>
          After three tries the stepper offers a case of ours. Whatever you found, look at where the tree loses: one of its
          pairs adds a small value to a large one before the small values have had a chance to combine.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Trường hợp của chúng tôi, 0.25, 1, 0.25, 2, được tìm bằng tìm kiếm vét cạn trong lúc thiết kế trang này; các unit
              test xác nhận nó. Phép 0.25 + 2 = 2.25 của cây là một trường hợp hòa, và hòa thì về giá trị lân cận chẵn, 2 (mục{' '}
              <a href={href('appA', 'A.4')}>A.4</a>).
            </p>
          }
        >
          <p>
            Our case, 0.25, 1, 0.25, 2, was found by a brute-force search while designing this page; the unit tests confirm
            it. The tree's 0.25 + 2 = 2.25 is a tie, and ties go to the even neighbour, 2 (section{' '}
            <a href={href('appA', 'A.4')}>A.4</a>).
          </p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Không đáp án nào là tổng">Neither answer is the sum</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Cây là cây rút gọn của chương 10, vốn chỉ được phép vì phép cộng được giả định có tính kết hợp. Vì vậy một phép rút
            gọn trên GPU và một vòng lặp trên CPU trên cùng các số float trả về các con số khác nhau, và nói chung không cái nào
            là tổng chính xác.
          </p>
        }
      >
        <p>
          The tree is chapter 10's reduction tree, which is only allowed because addition is assumed associative. A GPU
          reduction and a CPU loop over the same floats therefore return different numbers, and in general neither is the
          exact sum.
        </p>
      </Translated>
      <Source section="A.6" pages="593" />
      <Translated
        vi={
          <p>
            Cả hai đều là kết quả dấu phẩy động hợp lệ; sự khác biệt không phải là lỗi của bên nào. Điều một bài kiểm thử nên
            kiểm tra là cả hai đều nằm trong một dung sai so với một tham chiếu chính xác hơn, tính theo tỉ lệ với tổng các |x|.
          </p>
        }
      >
        <p>
          Both are legitimate floating-point results; the difference is not a bug in either. What a test should check is
          that both are within a tolerance of a more accurate reference, scaled to the sum of |x|.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch10-facts.md, A9)">{/* i18n-ignore: a date and a citation */}
        <Translated
          vi={
            <p>
              Hướng dẫn của NVIDIA nêu cùng ý đó với các ví dụ ba giá trị và tích vô hướng của riêng họ, và nói thêm: "Sự hiện
              diện của khác biệt không nhất thiết có nghĩa là kết quả của GPU sai".
            </p>
          }
        >
          <p>
            NVIDIA's guide makes the same point with its own three-value and dot-product examples, and adds: "The presence
            of differences does not necessarily mean the GPU's result is incorrect".
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Bài tập 4: luồng với dữ liệu, có làm tròn">Exercise 4: thread to data, with rounding</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bài tập 4 của sách: một sinh viên cộng một mảng float lớn luôn được sắp xếp từ nhỏ đến lớn. Để tránh phân kỳ nhánh,
            anh ấy dùng "thuật toán của Fig. A.4", và kết quả kém chính xác hơn anh ấy chờ đợi. Fig. A.4 là một bảng các số biểu
            diễn được, nên tham chiếu này bị sai. Chúng tôi hiểu thuật toán được nhắm tới là kernel hội tụ của chương 10 (Fig.
            10.8), kernel được thiết kế để tránh phân kỳ.
          </p>
        }
      >
        <p>
          The book's exercise 4: a student sums a large float array that is always sorted from smallest to largest. To avoid
          branch divergence he uses "the algorithm of Fig. A.4", and the result is less accurate than he expected. Fig. A.4
          is a table of representable numbers, so the reference is broken. We take the intended algorithm to be chapter
          10's convergent kernel (Fig. 10.8), the one designed to avoid divergence.
        </p>
      </Translated>
      <Source pages="600" figs={tr('Exercise 4; ch. 10 Fig. 10.8, p. 233', 'Bài tập 4; ch. 10 Fig. 10.8, p. 233')} />
      <Translated
        vi={
          <p>
            Trong kernel đó, luồng i sở hữu phần tử i và cộng phần tử i + stride vào nó, với bước nhảy bắt đầu bằng nửa mảng và
            giảm một nửa ở mỗi tầng. Trước khi chạy từng bước, hãy tính xem cách ghép cặp đó làm gì với dữ liệu đã sắp xếp.
          </p>
        }
      >
        <p>
          In that kernel thread i owns element i and adds element i + stride to it, with the stride starting at half the
          array and halving at each level. Before you step through it, work out what that pairing does to sorted data.
        </p>
      </Translated>
      <Ask q={questionsSA6['appA/A.6/q-ex4']} />
      <Translated
        vi={
          <p>
            Giờ hãy kiểm tra đáp án của bạn. Chạy từng bước kernel bằng FP16 trên dữ liệu có sẵn (tám số 1, năm số 8, ba số
            2048), rồi chuyển cách ghép cặp sang lân cận, kernel đầu tiên của chương 10 (Fig. 10.6), và chạy lại.
          </p>
        }
      >
        <p>
          Now check your answer. Step through the kernel in FP16 on the preset data (eight 1s, five 8s, three 2048s), then
          switch the pairing to neighbours, chapter 10's first kernel (Fig. 10.6), and run it again.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi bước một phép cộng, làm tròn một lần, không có fused multiply-add. Các luồng chạy lần lượt từng luồng, cho
              cùng kết quả như mọi thứ tự thực vì không có hai luồng nào chạm cùng một ô trong một tầng. Định dạng 5-bit không
              được cung cấp: với 16 giá trị, giá trị lớn nhất 3.5 của nó không để lại dữ liệu nào mà hai cách ghép cặp khác nhau
              mà không bị kẹp.
            </p>
          }
        >
          <p>
            One addition per step, rounded once, no fused multiply-add. Threads run one at a time, which gives the same
            result as any real order because no two threads touch the same cell in one level. The 5-bit format is not
            offered: with 16 values its largest value, 3.5, leaves no data on which the two pairings differ without
            clamping.
          </p>
        </Translated>
      </Simplification>
      <ThreadToData />

      <h2>
        <Translated vi="Sắp xếp trước, theo độ lớn">Sort first, by magnitude</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Cách khắc phục của sách là sắp xếp dữ liệu tăng dần trước khi rút gọn. Khi đó các phần tử cạnh nhau có độ lớn tương
            tự, nên một cây kết hợp các phần tử lân cận sẽ cộng các giá trị cùng cỡ ở mọi tầng. Ở chỗ mỗi luồng trước hết cộng
            tuần tự nhóm của riêng nó, thứ tự tăng dần cũng giúp vòng lặp đó.
          </p>
        }
      >
        <p>
          The book's remedy is to sort the data ascending before reducing it. Then elements next to each other have
          similar magnitudes, so a tree that combines neighbours adds like with like at every level. Where each thread
          first sums its own group sequentially, ascending order helps that loop as well.
        </p>
      </Translated>
      <Source section="A.6" pages="594" />
      <Translated
        vi={
          <p>
            Sách nêu đây là một lý do khiến việc sắp xếp phổ biến đến vậy trong mã số học song song quy mô lớn, và nói thêm
            rằng phải tính đến dấu của các giá trị.
          </p>
        }
      >
        <p>
          The book gives this as one reason sorting is so common in massively parallel numerical code, and adds that the
          signs of the values have to be taken into account.
        </p>
      </Translated>
      <Beyond title={tr('Not in the book: the same effect without sorting', 'Không có trong sách: cùng hiệu ứng mà không cần sắp xếp')}>
        <Translated
          vi={
            <p>
              Phép sum trên CPU của PyTorch là dạng phân tầng, "chia tổng thành các nhóm từ 16 phần tử trở lên" và cộng các
              tổng đó theo cùng cách, để "chỉ những số có cùng bậc độ lớn mới được cộng với nhau". NumPy dùng phép cộng theo cặp
              một phần: luôn luôn khi không chỉ định trục, còn không thì chỉ dọc theo trục nhanh trong bộ nhớ ({FACTS} D1, D4).
            </p>
          }
        >
          <p>
            PyTorch's CPU sum is a cascade, "chunking the sum into groups of 16 or more elements" and summing those sums
            the same way, so "only numbers of a similar order of magnitude are added together". NumPy uses partial pairwise
            summation: always when no axis is given, otherwise only along the fast axis in memory ({FACTS} D1, D4).
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsSA6['appA/A.6/q-sign']} />

      <h2>
        <Translated vi="Độ rộng của bộ tích lũy">The accumulator's width</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Thứ tự là một đòn bẩy. Đòn bẩy bạn sẽ dùng thường xuyên nhất trong mã suy luận là cái còn lại: định dạng của bộ
            tích lũy. Đầu vào FP16 hay BF16 thì rẻ để nạp; câu hỏi là bạn cộng chúng vào cái gì. Hãy dự đoán một con số trước
            khi mở trình khám phá.
          </p>
        }
      >
        <p>
          Order is one lever. The one you will pull most often in inference code is the other: the format of the
          accumulator. Inputs in FP16 or BF16 are cheap to load; the question is what you add them into. Predict one number
          before you open the explorer.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Sách không bàn về độ rộng bộ tích lũy. Trình khám phá và mọi con số trên đó đến từ mô hình của khóa học này.</p>}>
          <p>Accumulator width is not discussed in the book. The explorer and every number on it come from our model.</p>
        </Translated>
      </Beyond>
      <Predict
        id="appA/A.6/p-half-loop"
        prompt={
          <Translated
            vi={
              <>
                Một bộ tích lũy FP16 cộng 65,536 giá trị lấy đều từ [0, 1), theo thứ tự. Tổng đúng khoảng 32,768. Vòng lặp trả
                về bao nhiêu?
              </>
            }
          >
            <>
              An FP16 accumulator adds 65,536 values drawn uniformly from [0, 1), in order. The true sum is about 32,768. What
              does the loop return?
            </>
          </Translated>
        }
      >
        {(v) => <HalfLoopReveal v={v} />}
      </Predict>
      <Ask q={questionsSA6['appA/A.6/q-half-tree']} />
      <Translated
        vi={
          <p>
            Giờ hãy khám phá. Bắt đầu với dữ liệu phân bố đều, FP32, vòng lặp tuần tự, rồi đổi bộ tích lũy; sau đó thử cây và
            hai phương pháp Kahan, và từng tập dữ liệu.
          </p>
        }
      >
        <p>
          Now explore. Start with uniform data, FP32, the sequential loop, then switch the accumulator; then try the tree
          and the two Kahan methods, and each data set.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi phép cộng làm tròn về số chẵn gần nhất trong định dạng bộ tích lũy, sau khi đầu vào được chuyển sang định
              dạng đó. Không fused multiply-add, không ép về 0, không có trình biên dịch sắp xếp lại. Một hình dạng cây, "halves"
              của chương 10. Thứ tự chia đoạn là phép gộp việc của chương 10: luồng t cộng các phần tử t, t + T, t + 2T và cứ
              thế, rồi T tổng đi qua cây.
            </p>
          }
        >
          <p>
            Every addition rounds to nearest even in the accumulator format, after the input is converted to it. No fused
            multiply-add, no flush-to-zero, no compiler reassociation. One tree shape, chapter 10's "halves". The chunked
            order is chapter 10's coarsening: thread t adds elements t, t + T, t + 2T and so on, then the T sums go
            through the tree.
          </p>
        </Translated>
      </Simplification>
      <Explorer />
      <Translated vi={<p>Những gì trình khám phá cho thấy, tất cả đều từ mô hình:</p>}>
        <p>What the explorer shows, all of it from the model:</p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>
              Một bộ tích lũy hẹp bị kẹt. Một tổng đang chạy gồm các giá trị dưới 1 dừng ở 2048 trong FP16 và ở 256 trong
              BF16, những điểm mà khoảng cách của mỗi định dạng đạt 2. BF16 có phạm vi của FP32 nhưng chỉ có 8 bit có nghĩa.
            </li>
            <li>
              Trên dữ liệu có độ lớn trộn lẫn, thứ tự tăng dần thắng giảm dần với vòng lặp, còn cây và Kahan thắng vòng lặp dài.
            </li>
            <li>
              Một cây FP16 giữ các tổng riêng phần chính xác, nên nó có thể tràn ở 65,504 trong khi vòng lặp chỉ bị kẹt.
            </li>
            <li>
              Độ dài đoạn đánh đổi tốc độ lấy sai số. Chương 10 kéo dài vòng lặp của mỗi luồng để cắt bớt các bước nhàn rỗi của
              cây; khi đó vòng lặp của mỗi luồng mất nhiều hơn. Trên dữ liệu FP32 ngẫu nhiên hiệu ứng nhỏ, trên "one big, many
              small" nó tăng theo c, và trong FP16 nó lớn.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            A narrow accumulator stalls. A running sum of values below 1 stops at 2048 in FP16 and at 256 in BF16, the
            points where each format's spacing reaches 2. BF16 has FP32's range but only 8 significant bits.
          </li>
          <li>
            On mixed magnitudes, ascending order beats descending for the loop, and the tree and Kahan beat the long loop.
          </li>
          <li>
            An FP16 tree keeps its partial sums accurate, so it can overflow at 65,504 where the loop merely stalls.
          </li>
          <li>
            The chunk length trades speed against error. Chapter 10 lengthens each thread's loop to cut the tree's idle
            steps; each thread's loop then loses more. On random FP32 data the effect is small, on "one big, many small" it
            grows with c, and in FP16 it is large.
          </li>
        </ul>
      </Translated>
      <Translated
        vi={
          <p>
            Các thư viện bạn gọi đưa ra lựa chọn này thay bạn, và chúng không phải lúc nào cũng chọn giống nhau. Khi một mô hình
            half-precision cho các con số hơi khác nhau trên hai framework, bộ tích lũy là chỗ đầu tiên cần xem.
          </p>
        }
      >
        <p>
          The libraries you call make this choice for you, and they do not all make the same one. When a half-precision
          model gives slightly different numbers in two frameworks, the accumulator is the first place to look.
        </p>
      </Translated>
      <Today checked={CHECKED('D1, D2, D3')}>
        <Translated
          vi={
            <ul>
              <li>
                PyTorch tích lũy half, bfloat16 và FP8 bằng float ở mọi nơi, và float bằng double trên CPU. Các GEMM half có
                thể cắt bớt một vài tổng riêng phần (một cờ, <code>allow_fp16_reduced_precision_reduction</code>); cờ BF16 "được
                bật theo mặc định".
              </li>
              <li>
                <code>softmax</code> của MLX tích lũy bằng kiểu đầu vào trừ khi <code>precise=True</code>; theo cách chúng tôi
                đọc các kernel của nó, <code>mx.sum</code> trên half tích lũy bằng half.
              </li>
              <li>
                llama.cpp yêu cầu tích lũy F32 trong flash attention, và cho phép BF16 cho tích KQ không dùng flash, bằng{' '}
                <code>ggml_prec_set_acc</code>.
              </li>
            </ul>
          }
        >
          <ul>
            <li>
              PyTorch accumulates half, bfloat16 and FP8 in float everywhere, and float in double on the CPU. Half GEMMs may
              truncate a few partial sums (a flag, <code>allow_fp16_reduced_precision_reduction</code>); the BF16 flag "is
              turned on by default".
            </li>
            <li>
              MLX's <code>softmax</code> accumulates in the input type unless <code>precise=True</code>; by our reading of
              its kernels, <code>mx.sum</code> on half accumulates in half.
            </li>
            <li>
              llama.cpp asks for F32 accumulation in flash attention, and allows BF16 for the non-flash KQ product, with{' '}
              <code>ggml_prec_set_acc</code>.
            </li>
          </ul>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Phép cộng có bù">Compensated summation</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Để có thứ chắc chắn hơn việc sắp xếp, sách trỏ tới phép cộng có bù của Kahan, mà không mô tả nó.
          </p>
        }
      >
        <p>
          For something sturdier than sorting, the book points to Kahan's compensated summation, without describing it.
        </p>
      </Translated>
      <Source section="A.6" pages="594" figs="ref. [5]" />
      <Translated
        vi={
          <p>
            Ý tưởng là giữ một biến thứ hai, c, chứa phần mà phép cộng vừa rồi đánh rơi, và đưa nó trở lại đầu vào tiếp theo
            trước khi cộng. Mỗi bước tính y = x − c, rồi t = sum + y. Hiệu t − sum là lượng mà phép cộng thực sự đã cộng; trừ y
            khỏi nó để lại phần bị mất, với dấu đảo ngược. Phần đó đi vào c. Chi phí là bốn phép cộng mỗi phần tử thay vì một.
          </p>
        }
      >
        <p>
          The idea is to keep a second variable, c, that holds what the last addition dropped, and to put it back into the
          next input before adding. Each step computes y = x − c, then t = sum + y. The difference t − sum is what the
          addition really added; subtracting y from it leaves the part that was lost, with its sign turned around. That
          goes into c. The cost is four additions per element instead of one.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Phần mô tả thuật toán, cảnh minh họa và bài săn lỗi là của khóa học này. Dữ liệu trong cả hai là các giá trị FP32
              gần 1, nơi mỗi bước có thể viết theo đơn vị 2⁻²⁵.
            </p>
          }
        >
          <p>
            The description of the algorithm, the scene and the bug hunt are ours. The data in both are FP32 values near 1,
            where each step can be written in units of 2⁻²⁵.
          </p>
        </Translated>
      </Beyond>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi bước làm tròn một lần về số chẵn gần nhất bằng FP32, không sắp xếp lại và không fused multiply-add: đúng như
              mã C++ của bài săn lỗi làm khi được biên dịch không có fast math.
            </p>
          }
        >
          <p>
            Every step rounds once to nearest even in FP32, with no reassociation and no fused multiply-add: what the bug
            hunt's C++ does when it is compiled without fast math.
          </p>
        </Translated>
      </Simplification>
      <KahanScene />
      <Translated
        vi={
          <p>
            Bốn câu lệnh phải đúng là những câu này, theo đúng thứ tự này. Bên dưới là bốn phiên bản của vòng lặp. Hãy quyết
            định mỗi phiên bản làm gì trước khi chạy chúng.
          </p>
        }
      >
        <p>
          The four statements have to be exactly these, in exactly this order. Below are four versions of the loop. Decide
          what each one does before you run them.
        </p>
      </Translated>
      <BugHunt />
      <Translated
        vi={
          <p>
            Phiên bản 1 trừ phần bị mất thay vì khôi phục nó. Trên dữ liệu này các phần hiệu chỉnh của nó triệt tiêu theo từng
            cặp và nó trả về đúng tổng thường; trên dữ liệu phân bố đều nó rơi vào giữa vòng lặp thường và Kahan, còn xa mới
            bằng Kahan. Phiên bản 3 cập nhật sum trước khi tính c, nên t − sum luôn bằng 0 và c luôn bằng −y: mỗi đầu vào được
            mang sang tất cả các đầu vào sau nó, và tổng chạy mất kiểm soát.
          </p>
        }
      >
        <p>
          Version 1 subtracts the lost part instead of restoring it. On these data its corrections cancel in pairs and it
          returns exactly the plain sum; on uniform data it lands between the plain loop and Kahan's, nowhere near the
          latter. Version 3 updates sum before it computes c, so t − sum is always 0 and c is always −y: every input is
          carried into all the inputs after it, and the sum runs away.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Phiên bản 4 là cái cần nhớ. Trong số học thực, (t − sum) − y bằng 0, nên một trình biên dịch được phép coi phép
            cộng là có tính kết hợp có thể xóa nó, và vòng lặp Kahan âm thầm trở thành vòng lặp thường. Nó có được phép hay
            không phụ thuộc vào backend và các cờ của bạn.
          </p>
        }
      >
        <p>
          Version 4 is the one to remember. In real arithmetic (t − sum) − y is zero, so a compiler that is allowed to treat
          addition as associative may delete it, and Kahan's loop silently becomes the plain loop. Whether it may depends on
          the backend and on your flags.
        </p>
      </Translated>
      <Today checked={CHECKED('A13, B5, C2, C3')}>
        <Translated
          vi={
            <ul>
              <li>
                CUDA: <code>__fadd_rn</code> và <code>__fsub_rn</code> không bao giờ bị gộp thành phép nhân-cộng. Việc nvcc
                có sắp xếp lại các phép cộng float thông thường hay không thì hướng dẫn của NVIDIA không khẳng định (mục 5.6.3
                và 5.6.1.8 mâu thuẫn nhau).
              </li>
              <li>HIP và Clang: <code>-fno-associative-math</code> là mặc định; <code>-ffast-math</code> có sắp xếp lại.</li>
              <li>
                Metal: cả hai chế độ mặc định khả dĩ, fast và relaxed, đều cho phép sắp xếp lại;{' '}
                <code>#pragma METAL fp math_mode(safe)</code> cấm điều đó.
              </li>
            </ul>
          }
        >
          <ul>
            <li>
              CUDA: <code>__fadd_rn</code> and <code>__fsub_rn</code> are never fused into multiply-adds. Whether nvcc
              reassociates plain float additions, NVIDIA's guide does not settle (sections 5.6.3 and 5.6.1.8 disagree).
            </li>
            <li>HIP and Clang: <code>-fno-associative-math</code> is the default; <code>-ffast-math</code> reassociates.</li>
            <li>
              Metal: both candidate default modes, fast and relaxed, allow reassociation;{' '}
              <code>#pragma METAL fp math_mode(safe)</code> forbids it.
            </li>
          </ul>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Trong lúc chương này được thiết kế, một vòng lặp có bù trên GPU Apple của khóa học đã hoạt động đúng như Phiên bản 4
            dưới chế độ mặc định, và như Kahan khi có pragma. Viết thành <code>#pragma metal fp math_mode(safe)</code>, bằng
            chữ thường, nó biên dịch không cảnh báo và không thay đổi gì. Lab A.1 kiểm tra thiết bị của bạn thay vì tin vào
            kết quả nào trong hai.
          </p>
        }
      >
        <p>
          While this chapter was designed, a compensated loop on the course's Apple GPU behaved exactly as Version 4 under
          the default mode, and like Kahan's with the pragma. Spelled <code>#pragma metal fp math_mode(safe)</code>, in
          lower case, it compiled without a warning and changed nothing. Lab A.1 checks your device rather than trusting
          either result.
        </p>
      </Translated>
      <Ask q={questionsSA6['appA/A.6/q-fastmath']} />
      <Ask q={questionsSA6['appA/A.6/q-kahan16']} />

      <h2>
        <Translated vi="Trở lại chương 10, và M4">Back to chapter 10, and M4</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Gộp việc cho luồng (thread coarsening, §10.10) khiến mỗi luồng cộng nhiều phần tử trong một vòng lặp trước cây. Mỗi
            vòng lặp đó là một tổng dài, nên hệ số lớn hơn có thể có nghĩa là sai số lớn hơn. Một vòng lặp Kahan cho mỗi luồng
            loại bỏ phần lớn sai số đó, và trong một kernel bị giới hạn bởi bộ nhớ, ba phép cộng thêm mỗi phần tử của nó tốn
            rất ít: phép cộng đi từ 0.25 lên 1 phép toán mỗi byte, vẫn thấp hơn nhiều so với mức khoảng 20 mà ở đó thiết bị ví
            dụ của sách thôi bị giới hạn bởi bộ nhớ. Lab A.1 đo cả hai trên GPU của bạn (<a href={href('appA', 'labs')}>các lab</a>).
          </p>
        }
      >
        <p>
          Thread coarsening (§10.10) makes each thread add many elements in a loop before the tree. Each of those loops is
          a long sum, so a larger factor can mean a larger error. A per-thread Kahan loop removes most of it, and in a
          memory-bound kernel its three extra additions per element cost little: the sum goes from 0.25 to 1 operation per
          byte, still far below the 20 or so at which the book's example device stops being memory-bound. Lab A.1
          measures both on your GPU (<a href={href('appA', 'labs')}>labs</a>).
        </p>
      </Translated>
      <Source pages="246–248" figs="ch. 10 §10.10; ch. 5 §5.1, p. 94" />
      <Ask q={questionsSA6['appA/A.6/q-coarse']} />
      <Translated
        vi={
          <p>
            Mốc dự án M4, softmax của lab 10.3, đã làm điều quan trọng nhất. Trừ giá trị lớn nhất của hàng làm mọi số hạng
            exp(x − max) nằm trong (0, 1], đều dương, nên không có triệt tiêu, và số hạng lớn nhất đúng bằng 1. Đòn bẩy còn lại
            là bộ tích lũy, và lab 10.3 giữ nó ở float. Mỗi luồng của nó cộng các số hạng của một hàng cách nhau một độ rộng
            khối: thứ tự chia đoạn của trình khám phá, với c bằng độ dài hàng chia cho kích thước khối (cây của nó sau đó là
            các phép warp shuffle, một hình dạng khác với cây của trình khám phá).
          </p>
        }
      >
        <p>
          Milestone M4, the softmax of lab 10.3, already does the most important thing. Subtracting the row's maximum makes
          every term exp(x − max) lie in (0, 1], all positive, so there is no cancellation, and the largest term is exactly
          1. The lever left is the accumulator, and lab 10.3 keeps it in float. Each of its threads adds the terms of a row
          one block-width apart: the chunked order of the explorer, with c equal to the row length divided by the block
          size (its tree after that is warp shuffles, a different shape from the explorer's).
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Câu đố ở trang mở đầu chính là tổng này: một mẫu số softmax trên 131,072 số hạng. Ở độ chính xác half, vòng lặp
              kẹt ở mức thấp hơn nhiều so với giá trị đúng; cây thì không.
            </p>
          }
        >
          <p>
            The start page's teaser was this sum: one softmax denominator over 131,072 terms. In half precision the loop
            stalled far below the true value; the tree did not.
          </p>
        </Translated>
      </Aside>
    </>
  );
}
