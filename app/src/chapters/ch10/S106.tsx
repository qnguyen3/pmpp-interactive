import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { href } from '../../lib/router';
import { Translated, tr, useLocale } from '../../lib/i18n';
import {
  FIG_VALUES,
  HUNT_FILE,
  OUTCOME_LABEL,
  VARIANT,
  VARIANTS,
  fmtValue,
  requests,
  runShared,
  stateAt,
  strides,
  type Outcome,
  type Schedule,
  type ThreadRun,
  type TStep,
  type Variant,
} from '../../sim/reduction';
import { SCHEDULE_LABEL } from '../../sim/tiled';
import { functionSource } from '../../sim/toolchain';

import { NoteGroup } from './NoteGroup';
import { STATUS, useLabStatus } from './status';
const LAB = 'ch10/lab1_sum';
const SCHEDULES: Schedule[] = ['lockstep', 'in-order', 'reversed', 'shuffled'];
const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', row: 'global', input_s: 'shared', buf_s: 'shared',
  t: 'reg', stride: 'reg', a: 'reg', b: 'reg', m: 'reg', bdim: 'reg',
};
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const missing = (s: string | null) => !s || s.startsWith('// could not load');
const ONES = new Array<number>(16).fill(1);
const SHORT: Record<Variant, string> = {
  get book() { return tr('as in the book', 'như trong sách'); },
  get bottom() { return tr('barrier at the bottom', 'rào chắn ở cuối'); },
  get none() { return tr('no barrier', 'không rào chắn'); },
  get 'inside-if'() { return tr('barrier inside the if', 'rào chắn trong if'); },
  get le() { return tr('<= instead of <', '<= thay cho <'); },
  get two() { return tr('max then sum, with the barrier', 'max rồi sum, có rào chắn'); },
  get 'two-race'() { return tr('max then sum, without it', 'max rồi sum, không có nó'); },
};
const ORDER_SHORT: Record<Schedule, string> = {
  get lockstep() { return tr('lockstep', 'lockstep'); },
  get 'in-order'() { return tr('thread 0 first', 'luồng 0 trước'); },
  get reversed() { return tr('last thread first', 'luồng cuối trước'); },
  get shuffled() { return tr('random', 'ngẫu nhiên'); },
};
type Guess = '' | 'correct' | 'some' | 'deadlock' | 'oob';
const GUESS_LABEL: Record<Exclude<Guess, ''>, string> = {
  get correct() { return tr('correct in every order', 'đúng ở mọi thứ tự'); },
  get some() { return tr('wrong in some orders', 'sai ở một số thứ tự'); },
  get deadlock() { return tr('stuck at a barrier', 'kẹt tại một rào chắn'); },
  get oob() { return tr('out of bounds', 'vượt biên'); },
};
/** The class of a whole row of outcomes, to compare with a prediction. */
function rowClass(outs: Outcome[]): Exclude<Guess, ''> {
  if (outs.every((o) => o === 'correct')) return 'correct';
  if (outs.some((o) => o === 'deadlock')) return 'deadlock';
  if (outs.some((o) => o === 'oob')) return 'oob';
  return 'some';
}

/** The two kernels of the hunt that are right; the other five are wrong on purpose. */
const CORRECT = new Set<Variant>(['book', 'two']);

// ---------------------------------------------------------------- the bug hunt
const arr = (v: Variant) => (VARIANT[v].two ? 'buf_s' : 'input_s');

function describe(v: Variant, run: ThreadRun, s: TStep): string {
  const A = arr(v);
  if (s.thread === -1) return tr(`The barrier on line ${s.site} opens: all ${run.threads} threads have arrived.`, `Rào chắn ở dòng ${s.site} mở: cả ${run.threads} luồng đã đến.`);
  if (s.thread === -2) return tr('Nothing can move.', 'Không gì có thể di chuyển.');
  const op = s.op!;
  const t = s.thread;
  const r = s.reads;
  const w = s.write ? fmtValue(s.write.value) : '';
  const test = VARIANT[v].le ? `${t} <= ${op.stride}` : `${t} < ${op.stride}`;
  switch (op.kind) {
    case 'load':
      return tr(`Thread ${t} loads input[${r[0].index}] (${fmtValue(r[0].value)}) and input[${r[1].index}] (${fmtValue(r[1].value)}) and stores their sum, ${w}, in ${A}[${t}].`, `Luồng ${t} nạp input[${r[0].index}] (${fmtValue(r[0].value)}) và input[${r[1].index}] (${fmtValue(r[1].value)}) rồi lưu tổng của chúng, ${w}, vào ${A}[${t}].`);
    case 'load-max':
      return tr(`Thread ${t} loads input[${r[0].index}] (${fmtValue(r[0].value)}) and input[${r[1].index}] (${fmtValue(r[1].value)}) into a and b, and stores the larger, ${w}, in ${A}[${t}].`, `Luồng ${t} nạp input[${r[0].index}] (${fmtValue(r[0].value)}) và input[${r[1].index}] (${fmtValue(r[1].value)}) vào a và b, rồi lưu giá trị lớn hơn, ${w}, vào ${A}[${t}].`);
    case 'skip':
      return tr(`Thread ${t}, ${op.phase === 0 && VARIANT[v].two ? 'max' : 'sum'} iteration ${op.iter} (stride ${op.stride}): ${test} is false, so it skips.`, `Luồng ${t}, vòng lặp ${op.phase === 0 && VARIANT[v].two ? 'max' : 'sum'} ${op.iter} (bước nhảy ${op.stride}): ${test} sai, nên nó bỏ qua.`);
    case 'add':
      return tr(`Thread ${t}, iteration ${op.iter} (stride ${op.stride}): ${test}, so ${A}[${t}] (${fmtValue(r[0].value)}) += ${A}[${r[1].index}] (${fmtValue(r[1].value)}): ${w}.`, `Luồng ${t}, vòng lặp ${op.iter} (bước nhảy ${op.stride}): ${test}, nên ${A}[${t}] (${fmtValue(r[0].value)}) += ${A}[${r[1].index}] (${fmtValue(r[1].value)}): ${w}.`);
    case 'max':
      return tr(`Thread ${t}, max iteration ${op.iter} (stride ${op.stride}): ${A}[${t}] = the larger of ${fmtValue(r[0].value)} and ${A}[${r[1].index}] (${fmtValue(r[1].value)}): ${w}.`, `Luồng ${t}, vòng lặp max ${op.iter} (bước nhảy ${op.stride}): ${A}[${t}] = giá trị lớn hơn giữa ${fmtValue(r[0].value)} và ${A}[${r[1].index}] (${fmtValue(r[1].value)}): ${w}.`);
    case 'read-max':
      return tr(`Thread ${t} reads the max from ${A}[0]: m = ${fmtValue(r[0].value)}.`, `Luồng ${t} đọc max từ ${A}[0]: m = ${fmtValue(r[0].value)}.`);
    case 'load-diff':
      return tr(`Thread ${t} starts the sum: ${A}[${t}] = (a - m) + (b - m) = ${w}.`, `Luồng ${t} bắt đầu phép tổng: ${A}[${t}] = (a - m) + (b - m) = ${w}.`);
    case 'barrier':
      return tr(`Thread ${t} reaches the barrier on line ${op.lines[0]} and waits.`, `Luồng ${t} đến rào chắn ở dòng ${op.lines[0]} và chờ.`);
    case 'store':
      return tr(`Thread 0 copies ${A}[0] = ${fmtValue(r[0].value)} to *output.`, `Luồng 0 chép ${A}[0] = ${fmtValue(r[0].value)} sang *output.`);
    default:
      return '';
  }
}

function OutcomeChip({ o, detail }: { o: Outcome; detail?: string }) {
  return (
    <>
      <span className={'chip ' + (o === 'correct' ? 'ok' : 'fail')}>{OUTCOME_LABEL[o]}</span>
      {detail ? <div className="muted" style={{ fontSize: '0.75rem' }}>{detail}</div> : null}
    </>
  );
}

function BarrierHunt() {
  const src = useLabFile(HUNT_FILE);
  const [data, setData] = useState<'ones' | 'mixed'>('ones');
  const values = data === 'ones' ? ONES : [...FIG_VALUES];
  const [guess, setGuess] = useState<Record<Variant, Guess>>(() => Object.fromEntries(VARIANTS.map((v) => [v, ''])) as Record<Variant, Guess>);
  const [shown, setShown] = useState(false);
  const [variant, setVariant] = useState<Variant>('book');
  const [schedule, setSchedule] = useState<Schedule>('in-order');
  const [n, setN] = useState(0);

  const locale = useLocale();
  const table = useMemo(
    () => Object.fromEntries(VARIANTS.map((v) => [v, SCHEDULES.map((s) => runShared(v, values, s))])) as Record<Variant, ThreadRun[]>,
    [data, locale], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const run = table[variant][SCHEDULES.indexOf(schedule)];
  const total = run.steps.length;
  const at = Math.min(n, total);
  const cur = at > 0 ? run.steps[at - 1] : undefined;
  const st = stateAt(run, at);
  const firstBad = run.issues.length ? run.issues[0].step + 1 : null;
  const fn = !missing(src) ? functionSource(src!, VARIANT[variant].kernel) : null;
  const B = run.threads;
  const L = strides('shared', B).length;
  useEffect(() => setN(0), [variant, schedule, data]);

  const nextBarrier = () => {
    for (let i = at; i < total; i++) if (run.steps[i].thread < 0) return setN(i + 1);
    setN(total);
  };
  const highlight = cur?.op ? cur.op.lines : cur?.site ? [cur.site] : [];
  const allGuessed = VARIANTS.every((v) => guess[v] !== '');
  const regs = (t: number) => {
    // m as the thread last read it, from the steps so far.
    let m: number | null = null;
    for (let i = 0; i < at; i++) {
      const s = run.steps[i];
      if (s.thread === t && s.op?.kind === 'read-max') m = s.reads[0].value;
    }
    return m;
  };

  return (
    <Bench
      title={tr('Where does the barrier go?', 'Rào chắn đặt ở đâu?')}
      task={tr("Predict each variant's outcome over all four thread orders, then run them and replay any cell.", 'Dự đoán kết quả của mỗi biến thể trên cả bốn thứ tự luồng, rồi chạy chúng và phát lại bất kỳ ô nào.')}
      foot={
        <>
          <Translated
            vi={
              <>
                Các thứ tự: mọi luồng chạy lockstep; mỗi luồng chạy xa nhất có thể, luồng 0 trước, hoặc luồng cuối trước; một cách
                xen kẽ ngẫu nhiên. Mỗi biến thể là một kernel trong <code>labs/{HUNT_FILE}</code>, chạy ở đây bằng một mô hình một khối {B} luồng. Một
                nhãn nhỏ trong một ô bộ nhớ chia sẻ cho biết nó giữ giá trị nào: phép rút gọn (max hay sum) và vòng lặp đã ghi nó,
                0 cho lần nạp. "Kẹt tại một rào chắn" đại diện cho hành vi không xác định: một GPU có thể treo, hoặc chạy tiếp và
                tính ra kết quả sai.
              </>
            }
          >
            The orders: all threads in lockstep; each thread runs as far as it can, thread 0 first, or the last thread first;
            a random interleaving. Every variant is a kernel in <code>labs/{HUNT_FILE}</code>, run here by a model of one block of {B} threads. A
            small label in a shared-memory slot says which value it holds: the reduction (max or sum) and the iteration that
            wrote it, 0 for the load. "Stuck at a barrier" stands for undefined behaviour: a GPU may hang, or carry on and
            compute something wrong.
          </Translated>
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span style={FIELD}>
          <Translated vi="Đầu vào (16 giá trị)">Input (16 values)</Translated>
          <span className="seg" role="group" aria-label={tr('Input', 'Đầu vào')}>
            <button aria-pressed={data === 'ones'} onClick={() => setData('ones')}><Translated vi="toàn số một">all ones</Translated></button>
            <button aria-pressed={data === 'mixed'} onClick={() => setData('mixed')}>3, 1, 7, 0, ...</button>
          </span>
        </span>
        <span className="muted">
          <Translated
            vi={
              <>
                Tổng phải là {fmtValue(runShared('book', values, 'lockstep').expected)}; với hai phép rút gọn, tổng của x − max phải là{' '}
                {fmtValue(runShared('two', values, 'lockstep').expected)}.
              </>
            }
          >
            The sum must be {fmtValue(runShared('book', values, 'lockstep').expected)}; for the two reductions, the sum of x − max must be{' '}
            {fmtValue(runShared('two', values, 'lockstep').expected)}.
          </Translated>
        </span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-part="hunt-table">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Biến thể">Variant</Translated></th>
              <th><Translated vi="Dự đoán của bạn">Your prediction</Translated></th>
              {SCHEDULES.map((s) => <th key={s} title={SCHEDULE_LABEL[s]}>{ORDER_SHORT[s]}</th>)}
            </tr>
          </thead>
          <tbody>
            {VARIANTS.map((v) => {
              const outs = table[v].map((r) => r.outcome);
              const cls = rowClass(outs);
              return (
                <tr key={v} data-variant={v}>
                  <td style={{ paddingLeft: '0.6rem' }}>
                    <button className="quiet" aria-pressed={variant === v} onClick={() => setVariant(v)} title={VARIANT[v].label} style={{ whiteSpace: 'nowrap' }}>{SHORT[v]}</button>
                  </td>
                  <td>
                    <select value={guess[v]} onChange={(e) => setGuess({ ...guess, [v]: e.target.value as Guess })} aria-label={tr(`prediction for ${SHORT[v]}`, `dự đoán cho ${SHORT[v]}`)} style={{ maxWidth: '9rem' }}>
                      <option value=""><Translated vi="chọn">choose</Translated></option>
                      {(Object.keys(GUESS_LABEL) as Exclude<Guess, ''>[]).map((g) => <option key={g} value={g}>{GUESS_LABEL[g]}</option>)}
                    </select>
                    {shown && guess[v] ? (
                      <div className={guess[v] === cls ? 't-ok' : 't-fail'} style={{ fontSize: '0.75rem' }} data-verdict={guess[v] === cls ? 'right' : 'wrong'}>
                        {guess[v] === cls ? tr('as you predicted', 'đúng như bạn dự đoán') : tr(`actually: ${GUESS_LABEL[cls]}`, `thực tế: ${GUESS_LABEL[cls]}`)}
                      </div>
                    ) : null}
                  </td>
                  {table[v].map((r, i) => (
                    <td key={i} data-outcome={shown ? r.outcome : 'hidden'}>
                      {shown ? (
                        <button className="quiet" style={{ padding: '0.15rem 0.3rem', textAlign: 'left' }}
                          onClick={() => { setVariant(v); setSchedule(SCHEDULES[i]); }} aria-label={tr(`replay ${SHORT[v]}, ${SCHEDULES[i]}`, `phát lại ${SHORT[v]}, ${SCHEDULES[i]}`)}>
                          <OutcomeChip o={r.outcome} detail={r.outcome === 'wrong' ? (Number.isNaN(r.output) ? tr('got garbage', 'ra rác') : tr(`got ${fmtValue(r.output)}`, `ra ${fmtValue(r.output)}`)) : r.outcome === 'correct' && r.issues.length ? tr('a read raced', 'một lần đọc bị tranh chấp') : undefined} />
                        </button>
                      ) : (
                        <span className="muted">?</span>
                      )}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="row" style={{ margin: '0.8rem 0 1.1rem' }}>
        <button className="primary" onClick={() => setShown(true)} disabled={shown}><Translated vi="Chạy cả bốn thứ tự">Run all four orders</Translated></button>
        {!allGuessed && !shown ? <span className="muted"><Translated vi="Chọn một dự đoán cho mọi hàng trước, hoặc cứ chạy.">Choose a prediction for every row first, or run anyway.</Translated></span> : null}
        {shown ? (
          <span data-readout="score">
            <Translated
              vi={
                <>
                  {VARIANTS.filter((v) => guess[v] && guess[v] === rowClass(table[v].map((r) => r.outcome))).length} trên {VARIANTS.length} dự đoán đúng.
                  Bấm vào một ô để phát lại nó bên dưới.
                </>
              }
            >
              {VARIANTS.filter((v) => guess[v] && guess[v] === rowClass(table[v].map((r) => r.outcome))).length} of {VARIANTS.length} predictions right.
              Click a cell to replay it below.
            </Translated>
          </span>
        ) : null}
      </div>

      <div className="panel" data-part="replay">
        <div className="row" style={{ marginBottom: '0.7rem' }}>
          <label className="field">
            <Translated vi="Biến thể">Variant</Translated>
            <select value={variant} onChange={(e) => setVariant(e.target.value as Variant)} aria-label={tr('Variant', 'Biến thể')}>
              {VARIANTS.map((v) => <option key={v} value={v}>{VARIANT[v].label}</option>)}
            </select>
          </label>
          <label className="field">
            <Translated vi="Thứ tự luồng">Thread order</Translated>
            <select value={schedule} onChange={(e) => setSchedule(e.target.value as Schedule)} aria-label={tr('Thread order', 'Thứ tự luồng')}>
              {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
            </select>
          </label>
        </div>
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <button onClick={() => setN(0)} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
          <button onClick={() => setN(at - 1)} disabled={at === 0}><Translated vi="Lùi lại">Back</Translated></button>
          <button className="primary" onClick={() => setN(at + 1)} disabled={at >= total}><Translated vi="Bước">Step</Translated></button>
          <button onClick={nextBarrier} disabled={at >= total}><Translated vi="Chạy đến rào chắn tiếp theo">Run to the next barrier</Translated></button>
          <button onClick={() => setN(total)} disabled={at >= total}><Translated vi="Chạy đến cuối">Run to the end</Translated></button>
          {firstBad ? <button onClick={() => setN(firstBad)}><Translated vi="Đến lỗi đầu tiên">Go to the first failure</Translated></button> : null}
          <span className="muted" aria-live="polite" data-readout="hunt-step"><Translated vi={<>bước {at} trên {total}</>}>step {at} of {total}</Translated></span>
        </div>
        <div className={'narrate' + (cur?.issue ? ' bad' : '')} role="status" data-readout="hunt-narrate" style={{ marginBottom: '0.9rem' }}>
          {cur ? (
            <>
              {describe(variant, run, cur)}
              {cur.issue ? <><br /><b className="t-fail"><Translated vi={<>Lỗi: {cur.issue.message}.</>}>Failure: {cur.issue.message}.</Translated></b></> : null}
            </>
          ) : (
            <Translated vi={<>Chưa có gì chạy. {B} luồng, {B * 2} giá trị, bộ nhớ chia sẻ chưa được ghi ("?"). Bấm Bước.</>}><>Nothing has run yet. {B} threads, {B * 2} values, shared memory not yet written ("?"). Press Step.</></Translated>
          )}
        </div>
        <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 7fr) minmax(0, 5fr)' }}>
          {fn ? (
            <Code
              tabs={[{ label: tr('the kernel', 'kernel'), code: fn.text, tiers: TIERS, status: CORRECT.has(variant) ? STATUS.snippet : STATUS.wrong }]}
              highlight={highlight}
              caption={<Translated vi={<><code>{VARIANT[variant].kernel}</code>, từ <code>labs/{HUNT_FILE}</code>.</>}><><code>{VARIANT[variant].kernel}</code>, from <code>labs/{HUNT_FILE}</code>.</></Translated>}
            />
          ) : (
            <p className="muted"><Translated vi={<>Đang nạp <code>labs/{HUNT_FILE}</code>.</>}>Loading <code>labs/{HUNT_FILE}</code>.</Translated></p>
          )}
          <div style={{ minWidth: 0 }}>
            <div className="mx-label t-global"><Translated vi="đầu vào (bộ nhớ toàn cục)">input (global memory)</Translated></div>
            <div className="mx global" style={{ gridTemplateColumns: 'repeat(8, auto)', marginBottom: '0.8rem' }} data-part="input">
              {values.map((x, i) => (
                <span key={i} className={cur?.reads.some((r) => r.arr === 'input' && r.index === i) ? 'hot' : ''}>{x}</span>
              ))}
            </div>
            <div className="mx-label t-shared"><Translated vi={<>{arr(variant)} (bộ nhớ chia sẻ, khối này)</>}>{arr(variant)} (shared memory, this block)</Translated></div>
            <div className="mx shared" style={{ gridTemplateColumns: `repeat(${B}, auto)` }} data-part="shared">
              {st.shared.map((x, i) => {
                const w = (() => {
                  for (let k = at - 1; k >= 0; k--) {
                    const s = run.steps[k];
                    if (s.write?.arr === 'shared' && s.write.index === i) return s.write.version;
                  }
                  return null;
                })();
                const bad = cur?.reads.some((r) => r.arr === 'shared' && r.index === i && r.issue);
                const hot = cur?.write?.arr === 'shared' && cur.write.index === i;
                const read = cur?.reads.some((r) => r.arr === 'shared' && r.index === i);
                return (
                  <span key={i} className={bad ? 'bad' : hot ? 'hot' : read ? 'on' : ''} data-slot={i} data-value={fmtValue(x)}>
                    {fmtValue(x)}
                    {w ? <sub>{VARIANT[variant].two ? (w.phase === 0 ? 'max' : 'sum') : ''}{w.level}</sub> : null}
                  </span>
                );
              })}
            </div>
            {cur?.reads.some((r) => r.issue === 'oob') ? (
              <p className="t-fail" style={{ margin: '0.4rem 0 0' }}><Translated vi={<>Lần đọc {arr(variant)}[{cur.reads.find((r) => r.issue === 'oob')!.index}] vượt quá cuối mảng.</>}>The read of {arr(variant)}[{cur.reads.find((r) => r.issue === 'oob')!.index}] is past the end of the array.</Translated></p>
            ) : null}
            <p style={{ margin: '0.8rem 0 0.5rem' }} data-readout="hunt-output">
              <span className="mono">*output</span>{/* i18n-ignore: code identifier */} = {Number.isNaN(st.output) ? (at >= total ? tr('garbage or never written', 'rác hoặc không bao giờ được ghi') : tr('not written yet', 'chưa được ghi')) : fmtValue(st.output)}
              {at >= total ? <> <OutcomeChip o={run.outcome} /></> : null}
            </p>
            <table className="threads" aria-label={tr('Threads of the block', 'Các luồng của khối')}>
              <thead>
                <tr><th><Translated vi="luồng">thread</Translated></th><th><Translated vi="trạng thái">status</Translated></th>{VARIANT[variant].two ? <th className="t-reg">m</th> : null}</tr>
              </thead>
              <tbody>
                {Array.from({ length: B }, (_, t) => (
                  <tr key={t} className={cur?.thread === t ? 'cur' : ''}>
                    <td>{t}</td>
                    <td>{st.status[t] === 'waiting' ? tr(`waiting on line ${run.steps[st.last[t]].op!.lines[0]}`, `đang chờ ở dòng ${run.steps[st.last[t]].op!.lines[0]}`) : tr(st.status[t], ({ 'not started': 'chưa bắt đầu', running: 'đang chạy', finished: 'đã xong' } as Record<string, string>)[st.status[t]] ?? st.status[t])}</td>
                    {VARIANT[variant].two ? <td className="t-reg">{regs(t) == null ? '·' : fmtValue(regs(t)!)}</td> : null}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted" style={{ margin: '0.5rem 0 0', fontSize: '0.78rem' }}><Translated vi={<>{L} vòng lặp của cây; bước nhảy {strides('shared', B).join(', ')}.</>}>{L} iterations of the tree; stride {strides('shared', B).join(', ')}.</Translated></p>
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- global traffic
const TRAFFIC_N = [64, 128, 256, 512, 1024, 2048];

function Traffic() {
  const [k, setK] = useState(2);
  const N = TRAFFIC_N[k];
  const rows = (['simple', 'convergent', 'shared'] as const).map((kind) => ({ kind, total: requests(kind, N, 32).total }));
  const max = Math.max(...rows.map((r) => r.total));
  const label = { simple: tr('simple (10.3)', 'đơn giản (10.3)'), convergent: tr('convergent (10.4)', 'hội tụ (10.4)'), shared: tr('shared memory (10.6)', 'bộ nhớ chia sẻ (10.6)') };
  return (
    <Bench title={tr('Global traffic, three kernels', 'Lưu lượng toàn cục, ba kernel')} task={tr('Slide N: how do the three counts grow?', 'Kéo N: ba con số tăng thế nào?')}
      foot={<Translated vi={<>Warp rộng 32, và quy tắc yêu cầu của mục 10.5 (của chúng tôi: một yêu cầu cho mỗi mẩu 32 float đã căn chỉnh mà một warp chạm tới trong một lần truy cập). Các công thức của sách đếm trọn warp, nên N bắt đầu từ 64.</>}><>32-wide warps, and the request rule of section 10.5 (ours: one request per aligned piece of 32 floats a warp touches in one access). The book's formulas count whole warps, so N starts at 64.</></Translated>}>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.6rem' }}>
          N
          <input type="range" min={0} max={TRAFFIC_N.length - 1} value={k} onChange={(e) => setK(Number(e.target.value))} aria-label="N" />
          <b data-readout="traffic-n">{N}</b> <Translated vi="giá trị">values</Translated>
        </label>
      </div>
      <div data-part="traffic">
        {rows.map((r) => (
          <div key={r.kind} className="row" style={{ gap: '0.8rem', marginBottom: '0.35rem', flexWrap: 'nowrap' }} data-kind={r.kind} data-total={r.total}>
            <span style={{ width: '10.5rem', flex: 'none' }}>{label[r.kind]}</span>
            <span style={{ flex: 1, minWidth: 0 }}>
              <span style={{ display: 'block', height: '0.9rem', width: `${Math.max(0.5, (r.total / max) * 100)}%`, background: 'var(--global)', borderRadius: 2 }} />
            </span>
            <span style={{ width: '4.5rem', textAlign: 'right', flex: 'none', fontVariantNumeric: 'tabular-nums' }}>{r.total}</span>
          </div>
        ))}
      </div>
      <p style={{ margin: '0.8rem 0 0' }} data-readout="traffic-formula">
        <Translated
          vi={
            <>
              Bộ nhớ chia sẻ: N/32 + 1 = {N}/32 + 1 = {N / 32} lần nạp được hợp nhất (hai cho mỗi warp, {N / 64} warp) + 1 lần lưu = <b>{N / 32 + 1}</b>.
              Kernel đơn giản cần gấp {(rows[0].total / rows[2].total).toFixed(1)} lần, kernel hội tụ gấp{' '}
              {(rows[1].total / rows[2].total).toFixed(1)} lần.
            </>
          }
        >
          Shared memory: N/32 + 1 = {N}/32 + 1 = {N / 32} coalesced loads (two per warp, {N / 64} warps) + 1 store = <b>{N / 32 + 1}</b>.
          The simple kernel needs {(rows[0].total / rows[2].total).toFixed(1)} times as many, the convergent one{' '}
          {(rows[1].total / rows[2].total).toFixed(1)} times.
        </Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS106: Record<string, Question> = {
  'ch10/10.6/q-barrier-top': {
    kind: 'choice',
    id: 'ch10/10.6/q-barrier-top',
    prompt: (
      <Translated vi={<>Vì sao Hình 10.10 đặt <code>__syncthreads()</code> ở đầu thân vòng lặp?</>}>
        <>Why does Fig. 10.10 put <code>__syncthreads()</code> at the top of the loop body?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Để vòng lặp nhanh hơn: ở đầu, các luồng đến phép cộng sớm hơn sau rào chắn.">To make the loop faster: at the top, threads reach their addition sooner after the barrier.</Translated>, why: <Translated vi="Một rào chắn chỉ luôn thêm thời gian chờ; câu hỏi là nó phải sắp thứ tự điều gì.">A barrier only ever adds waiting; the question is what it must order.</Translated> },
      {
        label: <Translated vi="Để các lần ghi bộ nhớ chia sẻ của lần nạp, và của mỗi vòng lặp, xong trước khi vòng lặp tiếp theo đọc chúng.">So the load's shared-memory writes, and each iteration's, are done before the next iteration reads them.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Không có lý do gì: đặt ở cuối vòng lặp cũng sắp thứ tự cùng các lần ghi và đọc đó tốt như vậy.">No reason: the bottom of the loop would order the same writes and reads just as well.</Translated>, why: <Translated vi={<>Khi đó không gì ngăn cách vòng lặp đầu tiên với lần nạp. Hãy chạy "rào chắn ở cuối" trong cuộc săn lỗi.</>}>Then nothing separates the first iteration from the load. Run "barrier at the bottom" in the hunt.</Translated> },
      { label: <Translated vi="Vì một rào chắn ở cuối sẽ deadlock ở vòng lặp cuối, khi chỉ luồng 0 hoạt động.">Because a barrier at the bottom would deadlock in the last iteration, when only thread 0 is active.</Translated>, why: <Translated vi="Nó không deadlock; mọi luồng vẫn đến được nó. Nó để lần nạp không được bảo vệ.">It does not deadlock; every thread still reaches it. It leaves the load unprotected.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Ở đầu, rào chắn đầu tiên ngăn cách lần nạp với vòng lặp đầu tiên, và mỗi rào chắn sau đó ngăn cách một vòng lặp với
            vòng trước nó. Không cần rào chắn sau vòng lặp: bên ghi duy nhất của vòng lặp cuối là luồng 0, và luồng 0 chính là
            luồng đọc <code>input_s[0]</code>. Sách phát biểu điểm thứ nhất (p. 236); điểm thứ hai là cách chúng tôi đọc code.
          </p>
        }
      >
        <p>
          At the top, the first barrier separates the load from the first iteration, and each later one separates an
          iteration from the one before. No barrier is needed after the loop: the last iteration's only writer is thread 0,
          and thread 0 is the one that reads <code>input_s[0]</code>. The book states the first point (p. 236); the second
          is our reading of the code.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.6, p. 236; notes/ch10.md (no barrier after the loop: inferred)',
    vi: { source: 'ch. 10 §10.6, p. 236; notes/ch10.md (không có rào chắn sau vòng lặp: suy ra)' },
  },
  'ch10/10.6/q-blockdim': {
    kind: 'choice',
    id: 'ch10/10.6/q-blockdim',
    prompt: (
      <Translated
        vi={
          <>
            <code>BLOCK_DIM</code> được biên dịch là 1024, nhưng kernel của Hình 10.10 được khởi chạy với 512 luồng trên một hàng
            1024 giá trị. Điều gì sai?
          </>
        }
      >
        <>
          <code>BLOCK_DIM</code> is compiled as 1024, but the kernel of Fig. 10.10 is launched with 512 threads on a row of
          1024 values. What goes wrong?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Không gì cả: vòng lặp bắt đầu tại blockDim.x/2, nên kernel thích nghi với kích thước được khởi chạy.">Nothing: the loop starts at blockDim.x/2, so the kernel adapts to the launched size.</Translated>, why: <Translated vi="Vòng lặp thì có. Lần nạp thì không: nó dùng BLOCK_DIM.">The loop does. The load does not: it uses BLOCK_DIM.</Translated> },
      { label: <Translated vi="Luồng t nạp input[t + 1024], vượt quá hàng, và các giá trị 512 đến 1023 không bao giờ được nạp.">Thread t loads input[t + 1024], past the row, and values 512 to 1023 are never loaded.</Translated>, correct: true },
      { label: <Translated vi="Lần khởi chạy thất bại, vì mảng chia sẻ lớn hơn khối.">The launch fails, because the shared array is larger than the block.</Translated>, why: <Translated vi="Một mảng chia sẻ được phép lớn hơn mức cần; không gì đối chiếu nó với khối.">A shared array may be larger than needed; nothing checks it against the block.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lần nạp ghép phần tử t với phần tử <code>t + BLOCK_DIM</code>, một hằng số lúc biên dịch, trong khi cây dùng{' '}
            <code>blockDim.x</code>, kích thước được khởi chạy. Chúng phải khớp nhau, và khối phải là lũy thừa của hai để cây chia
            đôi gọn gàng. Sách trộn lẫn hai thứ mà không nói ra; quy tắc này là cách chúng tôi đọc. Phiên bản của lab 10.1 dùng{' '}
            <code>blockDim.x</code> ở cả hai chỗ và định kích thước mảng chia sẻ cho khối lớn nhất.
          </p>
        }
      >
        <p>
          The load pairs element t with element <code>t + BLOCK_DIM</code>, a compile-time constant, while the tree uses{' '}
          <code>blockDim.x</code>, the launched size. They must agree, and the block must be a power of two for the tree to
          halve cleanly. The book mixes the two without saying so; this rule is our reading. Lab 10.1's version uses{' '}
          <code>blockDim.x</code> in both places and sizes the shared array for the largest block.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.6, Fig. 10.10, p. 237; notes/ch10.md "Unclear in the PDF" (inferred)',
    vi: { source: 'ch. 10 §10.6, Fig. 10.10, p. 237; notes/ch10.md "Unclear in the PDF" (suy ra)' },
  },
  'ch10/10.6/q-req-shared': {
    kind: 'numeric',
    id: 'ch10/10.6/q-req-shared',
    prompt: (
      <Translated vi={<>Kernel bộ nhớ chia sẻ trên 2048 giá trị, warp rộng 32. Nó tạo bao nhiêu yêu cầu bộ nhớ toàn cục?</>}>
        <>The shared-memory kernel on 2048 values, 32-wide warps. How many global-memory requests does it make?</>
      </Translated>
    ),
    answer: 65,
    explain: (
      <Translated
        vi={
          <p>
            1024 luồng trong 32 warp, mỗi warp nạp hai mẩu 32 float được hợp nhất (<code>input[t]</code> và{' '}
            <code>input[t + 1024]</code>): 64 yêu cầu. Rồi một lần lưu tổng. N/32 + 1 = 65, so với 204 của kernel hội tụ và
            1149 của kernel đơn giản.
          </p>
        }
      >
        <p>
          1024 threads in 32 warps, each warp loading two coalesced pieces of 32 floats (<code>input[t]</code> and{' '}
          <code>input[t + 1024]</code>): 64 requests. Then one store of the total. N/32 + 1 = 65, against 204 for the
          convergent kernel and 1149 for the simple one.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.6, pp. 236–237 (the formula); the 2048 case is ours',
    vi: { source: 'ch. 10 §10.6, pp. 236–237 (công thức); trường hợp 2048 là của khóa học này' },
  },
  'ch10/10.6/q-two-reductions': {
    kind: 'choice',
    id: 'ch10/10.6/q-two-reductions',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel softmax rút gọn max vào <code>scratch[0]</code>, mọi luồng đọc <code>scratch[0]</code>, rồi phép rút gọn
            tổng bắt đầu ghi <code>scratch[t]</code>. Cần gì giữa lần đọc và các lần ghi, và nó bảo vệ loại phụ thuộc nào?
          </>
        }
      >
        <>
          A softmax kernel reduces the max into <code>scratch[0]</code>, every thread reads <code>scratch[0]</code>, and then
          the sum reduction starts writing <code>scratch[t]</code>. What is needed between the read and the writes, and what
          kind of dependence does it protect?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Không cần gì: mỗi luồng đọc max trước khi ghi, theo thứ tự chương trình.">Nothing: each thread reads the max before it writes, in program order.</Translated>, why: <Translated vi="Thứ tự chương trình chỉ đúng trong một luồng. Luồng 0 có thể ghi scratch[0] trước khi luồng 5 đọc nó.">Program order holds within a thread. Thread 0 can write scratch[0] before thread 5 has read it.</Translated> },
      { label: <Translated vi="Một rào chắn, cho phụ thuộc đọc-sau-ghi (thật) trên max.">A barrier, for a read-after-write (true) dependence on the max.</Translated>, why: <Translated vi="Phụ thuộc đó đã được rào chắn trước lần đọc bảo vệ. Câu hỏi là về khoảng trống sau nó.">That one is protected by the barrier before the read. The question is about the gap after it.</Translated> },
      {
        label: <Translated vi="Một rào chắn, cho phụ thuộc ghi-sau-đọc (giả); một mảng thứ hai sẽ loại bỏ nó.">A barrier, for a write-after-read (false) dependence; a second array would remove it.</Translated>,
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Các lần ghi đầu tiên của phép tổng dùng lại chính ô mà mọi luồng vẫn đang đọc max. Bên ghi không cần gì từ bên đọc;
            chỉ là bên đọc không được mất giá trị. Đó là phụ thuộc giả của chương 5 (§5.4): cho phép tổng một mảng scratch riêng
            và rào chắn biến mất, với cái giá là thêm bộ nhớ chia sẻ. PyTorch và llama.cpp đều giữ một buffer và rào chắn, kèm một
            chú thích giải thích lý do.
          </p>
        }
      >
        <p>
          The sum's first writes reuse the slot every thread is still reading the max from. The writer needs nothing from
          the readers; they just must not lose the value. That is chapter 5's false dependence (§5.4): give the sum its own
          scratch array and the barrier goes away, at the cost of more shared memory. PyTorch and llama.cpp both keep one
          buffer and the barrier, with a comment saying why.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.4, pp. 110–111; notes/today/ch10-facts.md D1, D4', // i18n-ignore: citation only
  },
};

// ---------------------------------------------------------------- the page
export function S106() {
  const status = useLabStatus(LAB);
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  // The lab files separate their kernels with "// ---- (a)" rulers; leave those out of the listing.
  const strip = (t: string) => t.split('\n').filter((l, i, all) => !(i < all.length - 1 && /^\/\/ -{8,}/.test(l))).join('\n');
  const kf = !missing(cu) ? functionSource(cu!, 'SharedMemorySumReductionKernel') : null;
  const mf = !missing(metal) ? functionSource(metal!, 'shared_memory_sum') : null;
  const k = kf ? { ...kf, text: strip(kf.text) } : null;
  const m = mf ? { ...mf, text: strip(mf.text) } : null;
  const r256 = { c: requests('convergent', 256, 32).total, s: requests('shared', 256, 32).total };
  return (
    <>
      <Translated
        vi={
          <p>
            Kernel hội tụ vẫn ghi mọi tổng từng phần vào bộ nhớ toàn cục rồi đọc lại ở vòng lặp tiếp theo. Một phần lưu lượng đó
            có thể được bộ nhớ đệm cấp cuối phục vụ, nhưng ngay cả ở đó, độ trễ và băng thông, theo lời sách, vẫn là những giới
            hạn đáng kể. Bộ nhớ chia sẻ nhanh hơn cả hai rất nhiều. Vậy hãy giữ cây ở đó.
          </p>
        }
      >
        <p>
          The convergent kernel still writes every partial sum to global memory and reads it back in the next iteration. Some
          of that traffic may be served by the last-level cache, but even there latency and bandwidth are, in the book's
          words, significant limitations. Shared memory is much faster than both. So keep the tree there.
        </p>
      </Translated>
      <Source section="10.6" pages="236" />

      <h2><Translated vi="Cây trong bộ nhớ chia sẻ">The tree in shared memory</Translated></h2>
      <Translated
        vi={
          <p>
            Mỗi luồng nạp hai giá trị gốc của nó từ bộ nhớ toàn cục, cộng chúng, và lưu tổng vào một mảng{' '}
            <code>__shared__</code>. Phép cộng đó là tầng đầu tiên của cây, nên vòng lặp bắt đầu tại{' '}
            <code>blockDim.x/2</code>. Mọi tầng sau chỉ đọc và ghi bộ nhớ chia sẻ. Rào chắn chuyển lên đầu thân vòng lặp, nơi nó
            cũng ngăn cách lần nạp với vòng lặp đầu tiên. Cuối cùng luồng 0 ghi tổng vào bộ nhớ toàn cục. Truy cập toàn cục giảm
            còn N lần đọc và một lần ghi, và đầu vào không còn bị ghi đè.
          </p>
        }
      >
        <p>
          Each thread loads its two original values from global memory, adds them, and stores the sum in a{' '}
          <code>__shared__</code> array. That addition is the first level of the tree, so the loop starts at{' '}
          <code>blockDim.x/2</code>. Every later level reads and writes shared memory only. The barrier moves to the top of
          the loop body, where it also separates the load from the first iteration. At the end thread 0 writes the total to
          global memory. Global accesses drop to N reads and one write, and the input is no longer overwritten.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="10.6" pages="236–237" figs={tr('Figs. 10.9, 10.10', 'Hình 10.9, 10.10')} />
        <Aside>
          <Translated
            vi={
              <p>
                Phần đầu của Hình 10.10 chỉ có <code>float* input</code>, nhưng thân của nó ghi <code>*output</code>. Ở p. 236, văn
                bản dẫn "line 04" cho vòng lặp (thật ra là dòng 05) và gọi lần ghi cuối là "the final write to input[0]" (code ghi{' '}
                <code>*output</code>); ở p. 237, "the shared memory kernel in Fig. 10.9" nghĩa là Hình 10.10.
              </p>
            }
          >
            <p>
              Fig. 10.10's header has only <code>float* input</code>, but its body writes <code>*output</code>. On p. 236 the
              text cites "line 04" for the loop (it is line 05) and calls the last write "the final write to input[0]" (the code
              writes <code>*output</code>); on p. 237, "the shared memory kernel in Fig. 10.9" means Fig. 10.10.
            </p>
          </Translated>
        </Aside>
      </NoteGroup>
      {k ? (
        <Code
          tabs={[
            { label: tr('CUDA (lab 10.1)', 'CUDA (lab 10.1)'), code: k.text, tiers: TIERS, status: status.cu },
            ...(m ? [{ label: tr('Metal (lab 10.1)', 'Metal (lab 10.1)'), code: m.text, tiers: TIERS, status: status.metal }] : []),
          ]}
          caption={
            <Translated
              vi={
                <>
                  Lời giải tham chiếu của lab 10.1, từ <code>labs/{LAB}/solution/</code>. Hãy tự viết nó trong{' '}
                  <a href={href('ch10', 'labs')}>lab</a> trước khi nghiên cứu bản này. Ba khác biệt so với trang sách: tham số đầu ra
                  được khôi phục, <code>blockDim.x</code> thay cho <code>BLOCK_DIM</code> lúc biên dịch của sách (mảng chia sẻ được
                  định kích thước cho khối lớn nhất), và khối b làm việc trên hàng b, như ở mục 10.3 và 10.4.
                </>
              }
            >
              <>
                Lab 10.1's reference solution, from <code>labs/{LAB}/solution/</code>. Write it yourself in the{' '}
                <a href={href('ch10', 'labs')}>lab</a> before you study this one. Three differences from the page: the output
                parameter is restored, <code>blockDim.x</code> replaces the book's compile-time <code>BLOCK_DIM</code> (the
                shared array is sized for the largest block), and block b works on row b, as in sections 10.3 and 10.4.
              </>
            </Translated>
          }
        />
      ) : (
        <p className="muted"><Translated vi={<>Kernel được nạp từ <code>labs/{LAB}/solution/kernel.cu</code>.</>}>The kernel is loaded from <code>labs/{LAB}/solution/kernel.cu</code>.</Translated></p>
      )}
      <Translated
        vi={
          <p>
            Ba điều code dựa vào mà sách không nói ra. Không cần rào chắn sau vòng lặp: ở vòng lặp cuối chỉ luồng 0 ghi, và luồng
            0 chính là luồng đọc kết quả. <code>BLOCK_DIM</code> phải bằng <code>blockDim.x</code> được khởi chạy. Và khối phải
            là lũy thừa của hai, nếu không cây chia đôi sẽ đánh rơi giá trị.
          </p>
        }
      >
        <p>
          Three things the code relies on that the book leaves unsaid. No barrier is needed after the loop: in the last
          iteration only thread 0 writes, and thread 0 is the one that reads the result. <code>BLOCK_DIM</code> must equal
          the launched <code>blockDim.x</code>. And the block must be a power of two, or the halving tree drops values.
        </p>
      </Translated>
      <Beyond title={tr('Our reading', 'Cách chúng tôi đọc')}>
        <p><Translated vi="Ba điều này được suy ra từ code (notes/ch10.md đánh dấu như vậy), không được phát biểu trong sách.">These three are inferred from the code (notes/ch10.md marks them so), not stated in the book.</Translated></p>
      </Beyond>

      <h2><Translated vi="Rào chắn đặt ở đâu?">Where does the barrier go?</Translated></h2>
      <Translated
        vi={
          <p>
            Hình 10.10 và bốn cách làm sai: rào chắn dời xuống cuối vòng lặp, bị bỏ, hoặc đặt trong if, và phép kiểm tra hoạt
            động nới thành <code>&lt;=</code>. Rồi hai kernel chạy hai phép rút gọn qua một mảng scratch, như softmax làm (một
            max, rồi một phép tổng mà mọi luồng tính bằng max), có và không có rào chắn giữa chúng. Với mỗi kernel, hãy quyết
            định điều gì xảy ra trên bốn thứ tự luồng trước khi chạy: đúng ở mọi thứ tự, sai ở một số, kẹt tại một rào chắn, hay
            một lần đọc vượt biên. Bắt đầu với đầu vào toàn số một, khi đó tổng phải là 16.
          </p>
        }
      >
        <p>
          Fig. 10.10 and four ways to get it wrong: the barrier moved to the bottom of the loop, dropped, or put inside the
          if, and the active test loosened to <code>&lt;=</code>. Then two kernels that run two reductions through one scratch
          array, as a softmax does (a max, then a sum that every thread computes with the max), with and without a barrier
          between them. For each, decide what happens over the four thread orders before you run them: correct in every
          order, wrong in some, stuck at a barrier, or a read out of bounds. Start with all-ones input, where the sum must be
          16.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Cùng mô hình với trình chạy từng bước của mục 10.3: mỗi bước một phép toán mức mã nguồn, "lockstep" với mọi lần đọc
              của một vòng trước các lần ghi của nó, bộ nhớ chia sẻ chưa ghi được đọc là "?". Nó giữ quy tắc của sách rằng mọi luồng
              của một khối đều đến cùng một rào chắn, và báo một khối không thể rời một rào chắn là bị kẹt.
            </p>
          }
        >
          <p>
            The same model as section 10.3's stepper: one source-level operation per step, "lockstep" with all reads of a
            round before its writes, unwritten shared memory read as "?". It holds the book's rule that every thread of a
            block reaches the same barrier, and reports a block that cannot leave one as stuck.
          </p>
        </Translated>
      </Simplification>
      <BarrierHunt />
      <Translated
        vi={
          <p>
            Hai kết quả đáng xem lại. "Rào chắn ở cuối" và "không rào chắn" chỉ sai ở một số thứ tự: ở lockstep, và với luồng cao
            nhất chạy trước, mọi giá trị tình cờ đã nằm đúng chỗ trước khi được đọc. Còn "rào chắn trong if" bị kẹt ở mọi thứ tự.
            Theo quy tắc của sách, và của Metal, một rào chắn phải được mọi luồng của khối đến được, mà các luồng 4 đến 7 không
            bao giờ vào if. Trên GPU nó thậm chí có thể chạy xong, vì NVIDIA, và Apple cho Apple silicon, cho phép một luồng đã
            thoát không giữ rào chắn; nó vẫn sai.
          </p>
        }
      >
        <p>
          Two results deserve a second look. "Barrier at the bottom" and "no barrier" fail only in some orders: in lockstep,
          and with the highest thread first, every value happens to be in place before it is read. And "barrier inside the
          if" is stuck in every order. Under the book's rule, and Metal's, a barrier must be reached by every thread of the
          block, and threads 4 to 7 never enter the if. On a GPU it may even finish, since NVIDIA, and Apple for Apple
          silicon, let a thread that has exited off a barrier; it is still wrong.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch04-facts A6, B3, C3; ch05-facts C2)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              NVIDIA chỉ cho phép một rào chắn trong câu lệnh điều kiện nếu điều kiện giống nhau cho cả khối; Metal yêu cầu mọi
              luồng đều vào nó; AMD không nêu quy tắc nào.
            </p>
          }
        >
          <p>
            NVIDIA allows a barrier in a conditional only if the condition is the same for the whole block; Metal requires
            every thread to enter it; AMD states no rule.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Trình giả lập CPU của khóa học theo cách diễn đạt của NVIDIA: các luồng đã thoát không giữ rào chắn. Chạy trên nó,
            kernel rào-chắn-trong-if qua được cả ba thứ tự luồng của trình giả lập. Một bài kiểm tra qua được không làm một kernel
            đúng; nó chỉ nói thứ tự này, trên máy này, đã suôn sẻ. Trình giả lập có bắt được biến thể{' '}
            <code>&lt;=</code>: AddressSanitizer báo lần đọc vượt quá cuối <code>input_s</code> ở dòng của phép cộng.
          </p>
        }
      >
        <p>
          The course's CPU emulator follows NVIDIA's wording: exited threads do not hold a barrier. Run on it, the
          barrier-inside-the-if kernel passes in all three of the emulator's thread orders. A test that passes does not make
          a kernel correct; it says this order, on this machine, went well. The emulator does catch the{' '}
          <code>&lt;=</code> variant: AddressSanitizer reports the read past the end of <code>input_s</code> on the line of
          the addition.
        </p>
      </Translated>
      <Beyond title={tr('Run, not assumed', 'Đã chạy, không giả định')}>
        <Translated
          vi={
            <p>
              Cả hai kết quả trình giả lập đều đến từ việc chạy file snippet trên trình giả lập trong <code>app/src/sim/reduction.test.ts</code>,
              file này cũng kiểm tra rằng trình giả lập và mô hình này khớp nhau ở những thứ tự chúng cùng có.
            </p>
          }
        >
          <p>
            Both emulator results come from running the snippet file on the emulator in <code>app/src/sim/reduction.test.ts</code>,
            which also checks that the emulator and this model agree in the orders they share.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS106['ch10/10.6/q-barrier-top']} />
      <Ask q={questionsS106['ch10/10.6/q-blockdim']} />

      <h2><Translated vi="Nó tiết kiệm được gì">What it saves</Translated></h2>
      <Translated
        vi={
          <p>
            Với cây trong bộ nhớ chia sẻ, bộ nhớ toàn cục chỉ thấy N lần đọc và một lần ghi, tổng cộng N + 1 lần truy cập. Cả hai
            lần đọc của mỗi luồng đều được hợp nhất, nên chúng tốn N/32 yêu cầu, cộng một cho lần ghi. Với 256 giá trị: từ {r256.c} yêu cầu
            ở kernel hội tụ xuống 8 + 1 = {r256.s}, ít hơn bốn lần. Và mảng đầu vào được giữ nguyên.
          </p>
        }
      >
        <p>
          With the tree in shared memory, global memory sees N reads and one write, N + 1 accesses in all. Both reads of each
          thread are coalesced, so they take N/32 requests, plus one for the write. For 256 values: from {r256.c} requests in
          the convergent kernel to 8 + 1 = {r256.s}, four times fewer. And the input array is left as it was.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="10.6" pages="236–237" />
        <Simplification>
          <Translated
            vi={
              <p>
                Quy tắc yêu cầu của trình khám phá ở mục 10.5, của chúng tôi: không bộ nhớ đệm, không thời gian, không kích thước giao dịch.
              </p>
            }
          >
            <p>
              The request rule of section 10.5's explorer, ours: no caches, no timing, no transaction sizes.
            </p>
          </Translated>
        </Simplification>
      </NoteGroup>
      <Traffic />
      <Ask q={questionsS106['ch10/10.6/q-req-shared']} label={tr('Work it out', 'Tự tính ra')} />

      <h2><Translated vi="Hai phép rút gọn qua một mảng scratch">Two reductions through one scratch array</Translated></h2>
      <Translated
        vi={
          <p>
            Một softmax trên một hàng rút gọn hai lần: một max, thứ mà sau đó mọi luồng đều cần, và một tổng của exp(x − max).
            Trong cuộc săn lỗi, "max rồi sum, không có nó" chạy cả hai qua một mảng scratch chia sẻ mà không có rào chắn giữa
            chúng, và cho thấy điều gì sai: một luồng nhanh bắt đầu ghi tổng vào ô mà một luồng chậm chưa đọc max. Đó là một
            nguy cơ ghi-sau-đọc, phụ thuộc giả của chương 5 (§5.4).
          </p>
        }
      >
        <p>
          A softmax over a row reduces twice: a max, which every thread then needs, and a sum of exp(x − max). In the bug
          hunt, "max then sum, without it" runs both through one shared scratch array with no barrier between them, and shows
          what goes wrong: a fast thread starts writing the sum into the slot a slow thread has not read the max from yet.
          That is a write-after-read hazard, the false dependence of chapter 5 (§5.4).
        </p>
      </Translated>
      <Beyond>
        <p><Translated vi="Sách không bàn về softmax.">The book does not discuss softmax.</Translated></p>
      </Beyond>
      <Translated
        vi={
          <p>
            Các kernel production giữ một mảng và rào chắn, và nói lý do trong một chú thích. <code>BlockReduceSum</code> của{' '}
            PyTorch bắt đầu bằng một <code>__syncthreads()</code> để có thể được gọi hai lần liên tiếp, và softmax của llama.cpp
            đặt một rào chắn giữa max và sum của nó.
          </p>
        }
      >
        <p>
          Production kernels keep the one array and the barrier, and say why in a comment. PyTorch's{' '}
          <code>BlockReduceSum</code> starts with a <code>__syncthreads()</code> so that it can be called twice in a row, and
          llama.cpp's softmax puts one between its max and its sum.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch10-facts D1, D4)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              PyTorch: "ngăn tranh chấp khi các BlockReduce được gọi liên tiếp". llama.cpp: "cần sync vì chúng tôi dùng lại buf_iw
              giữa các lần gọi block_reduce".
            </p>
          }
        >
          <p>
            PyTorch: "prevent races when BlockReduces are called in a row". llama.cpp: "sync is needed as we reuse buf_iw
            across block_reduce invocations".
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS106['ch10/10.6/q-two-reductions']} />

      <h2><Translated vi="Cùng kernel đó trên các backend khác">The same kernel on other backends</Translated></h2>
      <Translated
        vi={
          <p>
            Chuyển tab trong đoạn code ở trên. Kernel này chỉ dùng những gì HIP viết giống CUDA (luồng, bộ nhớ chia sẻ, một rào
            chắn), nên file của lab được viết để biên dịch cho HIP nguyên trạng; điều đó là unverified, vì chưa có gì ở đây được
            build với ROCm. Trong Metal, mảng chia sẻ được khai báo{' '}
            <code>threadgroup</code> trong thân kernel, và lần này flag của rào chắn là{' '}
            <code>mem_flags::mem_threadgroup</code>: các tổng từng phần giờ nằm trong bộ nhớ threadgroup, đúng là thứ flag đó sắp
            thứ tự. So với mục 10.3, nơi chúng nằm trong bộ nhớ device và cần <code>mem_device</code>.
          </p>
        }
      >
        <p>
          Switch tabs in the listing above. This kernel uses only what HIP spells as CUDA does (threads, shared memory, a
          barrier), so the lab's file is written to compile for HIP as it stands; that is unverified, since nothing here has
          been built with ROCm. In Metal the shared array is declared{' '}
          <code>threadgroup</code> in the kernel body, and this time the barrier's flag is{' '}
          <code>mem_flags::mem_threadgroup</code>: the partial sums now live in threadgroup memory, which is what that flag
          orders. Compare section 10.3, where they lived in device memory and needed <code>mem_device</code>.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. notes/today/ch05-facts.md B4, C1, C2; ch04-facts.md A6, B3, C3; ch10-facts.md, "Where the
              backends really differ".
            </p>
          }
        >
          <p>
            The book uses CUDA only. notes/today/ch05-facts.md B4, C1, C2; ch04-facts.md A6, B3, C3; ch10-facts.md, "Where the
            backends really differ".
          </p>
        </Translated>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th><Translated vi="Khái niệm">Concept</Translated></th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="Mảng chia sẻ trong khối">Block-shared array</Translated></td>
            <td><code>__shared__ float input_s[..]</code></td>
            <td><Translated vi="cùng cách viết">same spelling</Translated></td>
            <td><code>threadgroup float input_s[..]</code> <Translated vi="trong thân kernel">in the kernel body</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Rào chắn trên bộ nhớ chia sẻ">Barrier over shared memory</Translated></td>
            <td><code>__syncthreads()</code></td>
            <td><Translated vi={<>cùng cách viết; "ngầm bao gồm một threadfence"</>}>same spelling; "implicitly include a threadfence"</Translated></td>
            <td><code>threadgroup_barrier(mem_flags::mem_threadgroup)</code></td>
          </tr>
          <tr>
            <td><Translated vi="Rào chắn trong câu lệnh điều kiện">Barrier in a conditional</Translated></td>
            <td><Translated vi="chỉ khi điều kiện giống nhau cho cả khối">only if the condition is the same for the whole block</Translated></td>
            <td><Translated vi="các trang của AMD không nêu quy tắc; hãy theo của CUDA">no rule stated in AMD's pages; follow CUDA's</Translated></td>
            <td><Translated vi="mọi luồng đều phải vào câu lệnh điều kiện">every thread must enter the conditional</Translated></td>
          </tr>
        </tbody>
      </table>
    </>
  );
}
