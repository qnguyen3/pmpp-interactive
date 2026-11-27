import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { href } from '../../lib/router';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { STATUS, useLabStatus } from './status';
import {
  EXERCISE_6,
  FIG_VALUES,
  INPLACE_FILE,
  INPLACE_KERNEL,
  firstIssueStep,
  fmtValue,
  historyAt,
  isActive,
  isPow2,
  ownedCell,
  runInPlace,
  stateAt,
  strides,
  type HistoryRow,
  type InPlaceKind,
  type Schedule,
  type ThreadRun,
  type TStep,
} from '../../sim/reduction';
import { SCHEDULE_LABEL } from '../../sim/tiled';
import { functionSource } from '../../sim/toolchain';

import { NoteGroup } from './NoteGroup';

// The lab whose runs give the status of the listings taken from its files (and from labs/ch10/shared/, which it runs).
const LAB10_1 = 'ch10/lab1_sum';
const SCHEDULES: Schedule[] = ['lockstep', 'in-order', 'reversed', 'shuffled'];
/** Identifiers coloured by the memory they live in. */
const TIERS: Record<string, Tier> = { input: 'global', output: 'global', row: 'global', i: 'reg', stride: 'reg', t: 'reg', b: 'reg', bdim: 'reg' };
const missing = (s: string | null) => !s || s.startsWith('// could not load');

/** The Metal kernel of a file, by name, or null. */
const metalFn = (src: string | null, name: string) => (missing(src) ? null : functionSource(src!, name));

// ---------------------------------------------------------------- the stepper
const DATA = [
  { key: 'fig', get label() { return tr('16 values, 8 threads (Fig. 10.6)', '16 giá trị, 8 luồng (Hình 10.6)'); } },
  { key: 'ex6', get label() { return tr('exercise 6: 6, 2, 7, 4, 5, 8, 3, 1', 'bài tập 6: 6, 2, 7, 4, 5, 8, 3, 1'); } },
  { key: 'own', get label() { return tr('your own values', 'giá trị của riêng bạn'); } },
] as const;
type DataKey = (typeof DATA)[number]['key'];

/** Parses 2, 4, 8 or 16 numbers. */
function parseValues(s: string): number[] | null {
  const parts = s.split(/[\s,]+/).filter(Boolean);
  const v = parts.map((p) => Number(p.replace('−', '-')));
  if (!v.length || v.some((x) => !Number.isFinite(x))) return null;
  return isPow2(v.length) && v.length >= 2 && v.length <= 16 ? v : null;
}

/** The iteration a barrier opening closes: the iteration of the last operation before it. */
function iterBefore(run: ThreadRun, index: number): number {
  for (let i = index - 1; i >= 0; i--) if (run.steps[i].op) return run.steps[i].op!.iter;
  return 0;
}

/** One sentence for a step, with the arithmetic spelled out. */
function describeInPlace(kind: InPlaceKind, run: ThreadRun, s: TStep): string {
  if (s.thread === -1) {
    const k = iterBefore(run, s.index);
    return tr(`The barrier opens: all ${run.threads} threads have finished iteration ${k}. Every partial sum of that iteration is in place before any thread starts the next.`, `Rào chắn mở: cả ${run.threads} luồng đã xong vòng lặp ${k}. Mọi tổng từng phần của vòng lặp đó đã nằm đúng chỗ trước khi bất kỳ luồng nào bắt đầu vòng tiếp theo.`);
  }
  const op = s.op!;
  const t = s.thread;
  const test = kind === 'simple' ? `${t} % ${op.stride}` : `${t} < ${op.stride}`;
  switch (op.kind) {
    case 'skip':
      return kind === 'simple'
        ? tr(`Thread ${t}, iteration ${op.iter} (stride ${op.stride}): ${test} is ${t % op.stride}, not 0, so it skips the addition.`, `Luồng ${t}, vòng lặp ${op.iter} (bước nhảy ${op.stride}): ${test} bằng ${t % op.stride}, khác 0, nên nó bỏ qua phép cộng.`)
        : tr(`Thread ${t}, iteration ${op.iter} (stride ${op.stride}): ${test} is false, so it skips the addition.`, `Luồng ${t}, vòng lặp ${op.iter} (bước nhảy ${op.stride}): ${test} sai, nên nó bỏ qua phép cộng.`);
    case 'add': {
      const [own, partner] = s.reads;
      return (
        tr(`Thread ${t}, iteration ${op.iter} (stride ${op.stride}): ${kind === 'simple' ? `${test} == 0` : `${test}`}, so it adds `, `Luồng ${t}, vòng lặp ${op.iter} (bước nhảy ${op.stride}): ${kind === 'simple' ? `${test} == 0` : `${test}`}, nên nó cộng `) +
        tr(`input[${partner.index}] (${fmtValue(partner.value)}) into input[${own.index}] (${fmtValue(own.value)}): input[${own.index}] = ${fmtValue(s.write!.value)}.`, `input[${partner.index}] (${fmtValue(partner.value)}) vào input[${own.index}] (${fmtValue(own.value)}): input[${own.index}] = ${fmtValue(s.write!.value)}.`)
      );
    }
    case 'barrier':
      return tr(`Thread ${t} reaches __syncthreads() at the end of iteration ${op.iter} and waits for the rest of the block.`, `Luồng ${t} đến __syncthreads() ở cuối vòng lặp ${op.iter} và chờ phần còn lại của khối.`);
    case 'store':
      return tr(`Thread 0 copies input[0] = ${fmtValue(s.write!.value)} to *output.`, `Luồng 0 chép input[0] = ${fmtValue(s.write!.value)} sang *output.`);
    default:
      return '';
  }
}

/** Fig. 10.6's layout: the input, then one row per iteration with the elements it wrote and the two each one came from. */
function HistoryGrid({ rows, threads, kind, cur }: { rows: HistoryRow[]; threads: number; kind: InPlaceKind; cur?: TStep }) {
  const N = rows[0].values.length;
  const CW = N > 8 ? 34 : 46;
  const CH = 24;
  const GAP = 24;
  const LW = 74;
  const TOP = 18;
  const y = (r: number) => TOP + r * (CH + GAP);
  const x = (c: number) => LW + c * CW;
  const W = LW + N * CW + 4;
  const H = y(rows.length - 1) + CH + 4;
  const curIter = cur?.op && cur.op.iter > 0 ? cur.op.iter : null;
  const curOwn = cur?.op?.kind === 'add' ? cur.reads[0].index : null;
  const curPartner = cur?.op?.kind === 'add' ? cur.reads[1].index : null;
  const activeNow = (t: number) => curIter != null && isActive(kind, t, strides(kind, threads)[curIter - 1]);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('The input and the array after each iteration', 'Đầu vào và mảng sau mỗi vòng lặp')} data-part="history"
      style={{ width: W, maxWidth: '100%', minWidth: Math.min(W, 460), display: 'block' }}>
      <defs>
        <marker id="s103-arrow" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute value */}
          <path d="M0,0 L6,3 L0,6 z" fill="var(--ink-3)" />
        </marker>
        <marker id="s103-arrow-hot" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute value */}
          <path d="M0,0 L6,3 L0,6 z" fill="var(--ink)" />
        </marker>
      </defs>
      {Array.from({ length: threads }, (_, t) => (
        <text key={t} x={x(ownedCell(kind, t)) + CW / 2} y={11} fontSize="9.5" textAnchor="middle" data-thread-label={t}
          style={{ fill: activeNow(t) ? 'var(--ink)' : 'var(--ink-3)', fontWeight: activeNow(t) ? 700 : 400 }}>
          t{t}
        </text>
      ))}
      {rows.map((row, r) => (
        <g key={r} data-row={r}>
          <text x={LW - 8} y={y(r) + CH / 2 + 4} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
            {r === 0 ? tr('input', 'đầu vào') : tr(`stride ${row.stride}`, `bước nhảy ${row.stride}`)}
          </text>
          {r > 0
            ? row.adds.map((a) => {
                const hot = curIter === r && curOwn === a.own && cur?.thread === a.thread;
                const stroke = hot ? 'var(--ink)' : 'var(--ink-3)';
                const marker = `url(#${hot ? 's103-arrow-hot' : 's103-arrow'})`;
                const tx = x(a.own) + CW / 2;
                return (
                  <g key={a.own} opacity={hot ? 1 : 0.75}>
                    <line x1={tx} y1={y(r - 1) + CH} x2={tx} y2={y(r) - 1} stroke={stroke} strokeWidth={hot ? 1.6 : 1} markerEnd={marker} />
                    <line x1={x(a.partner) + CW / 2} y1={y(r - 1) + CH} x2={tx + (a.partner > a.own ? 3 : -3)} y2={y(r) - 1}
                      stroke={stroke} strokeWidth={hot ? 1.6 : 1} markerEnd={marker} />
                  </g>
                );
              })
            : null}
          {row.values.map((v, c) => {
            const written = r === 0 || row.written[c];
            const bad = row.bad[c];
            const hot = (curIter === r && curOwn === c) || (curIter === r + 1 && (curPartner === c || curOwn === c) && cur?.op?.kind === 'add');
            return (
              <g key={c} data-cell={c} data-value={fmtValue(v)} data-written={written ? '1' : '0'} data-bad={bad ? '1' : '0'}>
                <rect x={x(c) + 1} y={y(r)} width={CW - 2} height={CH} rx="2"
                  fill={bad ? 'var(--fail-wash)' : written ? 'var(--global-wash)' : 'transparent'}
                  stroke={bad ? 'var(--fail)' : hot ? 'var(--ink)' : written ? 'var(--global)' : 'var(--rule)'}
                  strokeWidth={hot ? 2 : 1} strokeDasharray={written ? undefined : '3 2'} />
                <text x={x(c) + CW / 2} y={y(r) + CH / 2 + 4} fontSize="11" textAnchor="middle"
                  style={{ fill: bad ? 'var(--fail)' : written ? 'var(--ink)' : 'var(--ink-3)', fontWeight: written && r > 0 ? 650 : 400 }}>
                  {fmtValue(v)}
                </text>
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}

function Stepper() {
  const src = useLabFile(INPLACE_FILE);
  const [data, setData] = useState<DataKey>('fig');
  const [own, setOwn] = useState('5, 1, 4, 2, 8, 3, 6, 7');
  const [schedule, setSchedule] = useState<Schedule>('in-order');
  const [barrier, setBarrier] = useState(true);
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);

  const parsed = parseValues(own);
  const values = data === 'fig' ? [...FIG_VALUES] : data === 'ex6' ? [...EXERCISE_6] : parsed;
  const key = values ? values.join(',') : '';
  const locale = useLocale();
  const run = useMemo(() => (values ? runInPlace('simple', values, { barrier, schedule }) : null), [key, barrier, schedule, locale]); // eslint-disable-line react-hooks/exhaustive-deps
  const total = run ? run.steps.length : 0;
  const at = Math.min(n, total);
  const cur = run && at > 0 ? run.steps[at - 1] : undefined;
  const st = run ? stateAt(run, at) : null;
  const rows = run ? historyAt(run, at) : null;
  const firstBad = run ? firstIssueStep(run) : null;
  const fn = !missing(src) ? functionSource(src!, INPLACE_KERNEL.simple) : null;

  useEffect(() => {
    setN(0);
    setPlaying(false);
  }, [key, barrier, schedule]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setN((v) => v + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  const toBarrier = () => {
    if (!run) return;
    for (let i = at; i < total; i++) if (run.steps[i].thread === -1) return setN(i + 1);
    setN(total);
  };
  const highlight = cur?.op ? cur.op.lines : cur?.thread === -1 ? [7] : [];
  const done = run != null && at >= total;
  const T = run?.threads ?? 0;

  return (
    <Bench
      title={tr('Step the simple kernel', 'Chạy từng bước kernel đơn giản')}
      task={tr('Follow thread 0 to the first barrier. Then untick the barrier and find the first read that goes wrong.', 'Theo luồng 0 đến rào chắn đầu tiên. Rồi bỏ chọn rào chắn và tìm lần đọc đầu tiên bị sai.')}
      foot={
        <Translated
          vi={
            <>
              Mỗi hàng bên dưới đầu vào cho thấy những gì một vòng lặp đã ghi (nét liền) đè lên những gì vòng lặp trước để lại
              (nét đứt), với mũi tên từ hai phần tử mà một phép cộng đã đọc. Một phần tử màu đỏ được ghi từ một giá trị chưa phải
              giá trị cuối cùng. Các nhãn phía trên đầu vào là các luồng, đặt trên phần tử mà mỗi luồng sở hữu; những nhãn in
              đậm đang hoạt động trong vòng lặp hiện tại.
            </>
          }
        >
          <>
            Each row below the input shows what one iteration wrote (solid) over what the iteration before left (dashed),
            with arrows from the two elements an addition read. A red element was written from a value that was not final
            yet. The labels above the input are the threads, over the element each one owns; the bold ones are active in
            the current iteration.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Dữ liệu">Data</Translated>
          <select value={data} onChange={(e) => setData(e.target.value as DataKey)} aria-label={tr('Data', 'Dữ liệu')}>
            {DATA.map((d) => <option key={d.key} value={d.key}>{d.label}</option>)}
          </select>
        </label>
        {data === 'own' ? (
          <label className="field" style={{ minWidth: 0 }}>
            <Translated vi="2, 4, 8 hoặc 16 số">2, 4, 8 or 16 numbers</Translated>
            <input type="text" value={own} onChange={(e) => setOwn(e.target.value)} aria-label={tr('your values', 'giá trị của bạn')} style={{ width: '17rem', maxWidth: '100%' }} />
          </label>
        ) : null}
        <label className="field">
          <Translated vi="Thứ tự luồng">Thread order</Translated>
          <select value={schedule} onChange={(e) => setSchedule(e.target.value as Schedule)} aria-label={tr('Thread order', 'Thứ tự luồng')}>
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
        <label className="check"><input type="checkbox" checked={barrier} onChange={(e) => setBarrier(e.target.checked)} /> __syncthreads()</label>
      </div>
      {!run ? (
        <p><span className="chip fail"><Translated vi="Nhập 2, 4, 8 hoặc 16 số, cách nhau bằng dấu phẩy hoặc khoảng trắng.">Give 2, 4, 8 or 16 numbers, separated by commas or spaces.</Translated></span></p>
      ) : (
        <>
          <div className="row" style={{ marginBottom: '0.8rem' }}>
            <button onClick={() => { setPlaying(false); setN(0); }} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
            <button onClick={() => { setPlaying(false); setN(at - 1); }} disabled={at === 0}><Translated vi="Lùi lại">Back</Translated></button>
            <button className="primary" onClick={() => { setPlaying(false); setN(at + 1); }} disabled={at >= total}><Translated vi="Bước">Step</Translated></button>
            <button onClick={() => { setPlaying(false); toBarrier(); }} disabled={at >= total}>{barrier ? tr('Run to the barrier', 'Chạy đến rào chắn') : tr('Run to the end', 'Chạy đến cuối')}</button>
            <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Chạy')}</button>
            {firstBad ? <button onClick={() => { setPlaying(false); setN(firstBad); }}><Translated vi="Đến lỗi đầu tiên">Go to the first failure</Translated></button> : null}
            <span className="muted" aria-live="polite" data-readout="step"><Translated vi={<>bước {at} trên {total}</>}>step {at} of {total}</Translated></span>
          </div>
          <div className={'narrate' + (cur?.issue ? ' bad' : '')} role="status" data-readout="narrate" style={{ marginBottom: '0.9rem' }}>
            {cur ? (
              <>
                {describeInPlace('simple', run, cur)}
                {cur.issue ? <><br /><b className="t-fail"><Translated vi={<>Lỗi: {cur.issue.message}.</>}>Failure: {cur.issue.message}.</Translated></b></> : null}
              </>
            ) : (
              <>
                <Translated
                  vi={
                    <>
                      Chưa có gì chạy. {T} luồng, {T * 2} giá trị; luồng t sở hữu input[2t]. Bấm Bước, hoặc Chạy đến rào chắn để hoàn
                      tất một vòng lặp.
                    </>
                  }
                >
                  Nothing has run yet. {T} threads, {T * 2} values; thread t owns input[2t]. Press Step, or Run to the barrier to
                  finish an iteration.
                </Translated>
              </>
            )}
          </div>
          <div style={{ minWidth: 0, marginBottom: '0.6rem' }}>
            <div style={{ overflowX: 'auto' }}>
              <HistoryGrid rows={rows!} threads={T} kind="simple" cur={cur} />
            </div>
            <p style={{ margin: '0.7rem 0 0' }} data-readout="output">
              <span className="mono">*output</span>{/* i18n-ignore: code identifier */} = {Number.isNaN(st!.output) ? tr('not written yet', 'chưa được ghi') : fmtValue(st!.output)}
              {done ? (
                <>
                  {' '}
                  {run.outcome === 'correct' ? (
                    <span className="chip ok" data-outcome="correct"><Translated vi={<>tổng, {fmtValue(run.expected)}</>}>the sum, {fmtValue(run.expected)}</Translated></span>
                  ) : (
                    <span className="chip fail" data-outcome="wrong"><Translated vi={<>sai: tổng là {fmtValue(run.expected)}</>}>wrong: the sum is {fmtValue(run.expected)}</Translated></span>
                  )}
                  {run.issues.length ? <span className="muted"> <Translated vi={<>{run.issues.length} lần đọc một giá trị chưa phải cuối cùng.</>}>{run.issues.length} read{run.issues.length > 1 ? 's' : ''} of a value that was not final.</Translated></span> : null}
                </>
              ) : null}
            </p>
          </div>
          <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 7fr) minmax(0, 4fr)' }}>
            {fn ? (
              <Code
                tabs={[{ label: tr('Fig. 10.5, one block', 'Hình 10.5, một khối'), code: fn.text, tiers: TIERS, status: STATUS.snippet }]}
                highlight={highlight}
              />
            ) : (
              <p className="muted"><Translated vi={<>Đang nạp kernel từ <code>labs/{INPLACE_FILE}</code>.</>}>Loading the kernel from <code>labs/{INPLACE_FILE}</code>.</Translated></p>
            )}
            <table className="threads" aria-label={tr('Threads of the block', 'Các luồng của khối')} style={{ marginTop: '1.1rem' }}>
              <thead>
                <tr><th><Translated vi="luồng">thread</Translated></th><th><Translated vi="sở hữu">owns</Translated></th><th><Translated vi="trạng thái">status</Translated></th></tr>
              </thead>
              <tbody>
                {Array.from({ length: T }, (_, t) => (
                  <tr key={t} className={cur?.thread === t ? 'cur' : ''} data-thread={t}>
                    <td>{t}</td>
                    <td className="t-global">input[{2 * t}]</td>{/* i18n-ignore: code expression */}
                    <td>{st!.status[t] === 'waiting' ? tr('waiting at the barrier', 'đang chờ tại rào chắn') : tr(st!.status[t], ({ 'not started': 'chưa bắt đầu', running: 'đang chạy', finished: 'đã xong' } as Record<string, string>)[st!.status[t]] ?? st!.status[t])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS103: Record<string, Question> = {
  'ch10/10.3/q-ex6a': {
    kind: 'numeric',
    id: 'ch10/10.3/q-ex6a',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 6a của sách: chạy kernel đơn giản trên tám giá trị {'{'}6, 2, 7, 4, 5, 8, 3, 1{'}'}. Khi kernel kết thúc,{' '}
            <code>input[4]</code> giữ giá trị gì?
          </>
        }
      >
        <>
          Book exercise 6a: run the simple kernel on the eight values {'{'}6, 2, 7, 4, 5, 8, 3, 1{'}'}. What does{' '}
          <code>input[4]</code> hold when the kernel ends?
        </>
      </Translated>
    ),
    answer: 17,
    explain: (
      <Translated
        vi={
          <p>
            Bốn luồng, bước nhảy 1, 2 và 4. Sau bước nhảy 1 mảng là {'{'}8, 2, 11, 4, 13, 8, 4, 1{'}'}; sau bước nhảy 2,{' '}
            {'{'}19, 2, 11, 4, 17, 8, 4, 1{'}'}; sau bước nhảy 4 chỉ <code>input[0]</code> thay đổi, thành 36. Phần tử 4 giữ
            tổng từng phần của nửa sau, 5 + 8 + 3 + 1 = 17: đầu vào đã mất, và những gì còn lại là các tổng từng phần. Sách
            không in đáp án; đáp án này là của khóa học này, và trình chạy từng bước ở trên tái hiện nó.
          </p>
        }
      >
        <p>
          Four threads, strides 1, 2 and 4. After stride 1 the array is {'{'}8, 2, 11, 4, 13, 8, 4, 1{'}'}; after stride 2,{' '}
          {'{'}19, 2, 11, 4, 17, 8, 4, 1{'}'}; after stride 4 only <code>input[0]</code> changes, to 36. Element 4 keeps the
          partial sum of the second half, 5 + 8 + 3 + 1 = 17: the input is gone, and the leftovers are partial sums. The book
          prints no answer; this one is ours, and the stepper above reproduces it.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.12, exercise 6a, p. 250; answer ours',
    vi: { source: 'ch. 10 §10.12, bài tập 6a, p. 250; đáp án của khóa học này' },
  },
  'ch10/10.3/q-stale-read': {
    kind: 'choice',
    id: 'ch10/10.3/q-stale-read',
    prompt: (
      <Translated vi={<>Bỏ rào chắn và chạy kernel trên 16 giá trị với 8 luồng. Lần đọc nào có thể thấy một giá trị chưa phải giá trị cuối cùng?</>}>
        <>Remove the barrier and run the kernel on 16 values with 8 threads. Which read can see a value that is not final yet?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Ở bước nhảy 1, luồng 0 đọc input[1], phần tử mà luồng 1 đang cập nhật trong cùng vòng lặp.">At stride 1, thread 0 reads input[1], which thread 1 is updating in the same iteration.</Translated>, why: <Translated vi="Các phần tử lẻ không thuộc luồng nào. Không ai ghi vào input[1], nên không có gì để chờ.">Odd elements belong to no thread. Nobody ever writes input[1], so there is nothing to wait for.</Translated> },
      { label: <Translated vi="Ở bước nhảy 2, luồng 0 đọc input[2], phần tử mà luồng 1 ghi ở vòng lặp trước.">At stride 2, thread 0 reads input[2], which thread 1 writes in the iteration before.</Translated>, correct: true },
      { label: <Translated vi="Ở bước nhảy 2, luồng 1 đọc input[4], phần tử mà luồng 2 ghi ở vòng lặp trước.">At stride 2, thread 1 reads input[4], which thread 2 writes in the iteration before.</Translated>, why: <Translated vi="Luồng 1 không hoạt động ở bước nhảy 2: 1 % 2 khác 0.">Thread 1 is not active at stride 2: 1 % 2 is not 0.</Translated> },
      { label: <Translated vi="Không lần nào: mỗi luồng chỉ đọc các phần tử nó sở hữu hoặc các phần tử không ai ghi.">None: each thread reads only elements it owns or elements nobody writes.</Translated>, why: <Translated vi="Từ vòng lặp thứ hai trở đi, phần tử đối tác input[i + stride] thuộc về một luồng khác.">The partner element input[i + stride] belongs to another thread from the second iteration on.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Từ vòng lặp thứ hai trở đi, phần tử đối tác là tổng từng phần của một luồng khác: input[2] là của luồng 1, input[4]
            của luồng 2, và cứ thế. Không có rào chắn thì không gì ngăn luồng 0 đến phép cộng thứ hai của nó trước khi luồng 1
            làm xong phép cộng đầu tiên. Trong trình chạy từng bước, bỏ chọn rào chắn với thứ tự "mỗi luồng chạy xa nhất có thể,
            luồng 0 trước" và bấm <i>Đến lỗi đầu tiên</i>.
          </p>
        }
      >
        <p>
          From the second iteration on, the partner element is another thread's partial sum: input[2] is thread 1's, input[4]
          thread 2's, and so on. Without the barrier nothing stops thread 0 from reaching its second addition before thread 1
          has done its first. In the stepper, untick the barrier with the order "each thread runs as far as it can, thread 0
          first" and press <i>Go to the first failure</i>.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.3, pp. 229–230, Fig. 10.6',
  },
  'ch10/10.3/q-barrier-outside': {
    kind: 'choice',
    id: 'ch10/10.3/q-barrier-outside',
    prompt: (
      <Translated
        vi={
          <>
            Vì sao <code>__syncthreads()</code> nằm ngoài <code>if</code>, để cả những luồng bỏ qua phép cộng cũng thực thi
            nó?
          </>
        }
      >
        <>
          Why is <code>__syncthreads()</code> outside the <code>if</code>, so that threads that skip the addition execute it
          too?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Mọi luồng của khối phải đến được rào chắn; nếu nằm trong if, các luồng không hoạt động sẽ không bao giờ đến.">Every thread of the block must reach the barrier; inside the if, inactive threads never would.</Translated>, correct: true },
      { label: <Translated vi="Để các luồng không hoạt động có thể nhận làm những phép cộng mà các luồng hoạt động chưa xong.">So that the inactive threads can pick up additions the active ones have not finished.</Translated>, why: <Translated vi="Chúng không làm phép cộng nào; chúng chỉ chờ.">They do no addition; they only wait.</Translated> },
      { label: <Translated vi="Đó là lựa chọn phong cách: nằm trong if, nó sẽ chờ các luồng hoạt động, và chỉ cần vậy.">It is a style choice: inside the if, it would wait for the active threads, which is all it needs.</Translated>, why: <Translated vi="Một rào chắn mà chỉ một số luồng đến được thì vi phạm quy tắc của chương 4.">A barrier that only some threads reach breaks the rule of chapter 4.</Translated> },
      { label: <Translated vi="Để vòng lặp chạy nhanh hơn, vì khi đó mọi luồng cùng rời rào chắn.">To make the loop run faster, since all threads then leave the barrier together.</Translated>, why: <Translated vi="Một rào chắn chỉ luôn thêm thời gian chờ.">A barrier only ever adds waiting.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Quy tắc của chương 4: một rào chắn được thực thi bởi mọi luồng của khối hoặc không luồng nào. Đặt nó trong{' '}
            <code>if</code> thì từ vòng lặp thứ hai trở đi các luồng không hoạt động bỏ qua nó, nên các luồng hoạt động chờ những
            luồng không bao giờ tới. NVIDIA gọi một rào chắn nằm dưới một điều kiện không giống nhau cho cả khối là hành vi không
            xác định: nó có thể treo, hoặc chạy tiếp và tính ra kết quả sai. Cuộc săn lỗi của mục 10.6 chạy đúng sai lầm này.
          </p>
        }
      >
        <p>
          Chapter 4's rule: a barrier is executed by every thread of the block or by none. Put it inside the{' '}
          <code>if</code> and from the second iteration on the inactive threads skip it, so the active ones wait for threads
          that never come. NVIDIA calls a barrier under a condition that is not the same for the whole block undefined
          behaviour: it may hang, or carry on and compute something wrong. Section 10.6's bug hunt runs exactly this mistake.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.3; ch. 10 §10.3, p. 230; notes/today/ch04-facts.md A6', // i18n-ignore: citation only
  },
  'ch10/10.3/q-three-blocks': {
    kind: 'choice',
    id: 'ch10/10.3/q-three-blocks',
    prompt: (
      <Translated
        vi={
          <>
            Bạn cần tổng của 5000 giá trị và khởi chạy kernel của sách, không sửa gì, với 3 khối 1024 luồng. Điều gì xảy
            ra?
          </>
        }
      >
        <>
          You need the sum of 5000 values and launch the book's kernel, unchanged, with 3 blocks of 1024 threads. What
          happens?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Mỗi khối rút gọn một phần ba đầu vào của riêng nó, và khối xong cuối cùng ghi *output.">Each block reduces its own third of the input, and the last block to finish writes *output.</Translated>, why: <Translated vi="Không gì trong kernel dùng blockIdx. Mọi khối đều tính i = 2*threadIdx.x.">Nothing in the kernel uses blockIdx. Every block computes i = 2*threadIdx.x.</Translated> },
      { label: <Translated vi="Đáp án đúng, chỉ chậm hơn một khối, vì các khối lặp lại cùng một việc.">The answer is right, only slower than one block, because the blocks repeat the same work.</Translated>, why: <Translated vi="Ba khối ghi vào cùng các phần tử cùng lúc.">The three blocks write the same elements at the same time.</Translated> },
      {
        label: <Translated vi="Cả ba cùng lúc rút gọn tại chỗ 2048 giá trị đầu tiên, tranh chấp trên chúng và trên *output.">All three reduce the first 2048 values in place at once, racing on them and on *output.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Lần khởi chạy thất bại: một kernel gọi __syncthreads() chỉ được khởi chạy với một khối.">The launch fails: a kernel that calls __syncthreads() may only be launched with one block.</Translated>, why: <Translated vi="Lần khởi chạy là hợp lệ. Rào chắn đơn giản là không có tác dụng giữa các khối.">The launch is legal. The barrier simply has no effect between blocks.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Chỉ số chỉ phụ thuộc vào <code>threadIdx.x</code>, nên mọi khối đều làm việc trên các phần tử 0 đến 2047, mỗi khối
            ghi đè lên những tổng từng phần mà các khối khác vẫn đang đọc, và không rào chắn nào bao trùm các khối để sắp thứ
            tự chúng. Cả ba đều ghi <code>*output</code>. Các phần tử 2048 đến 4999 không bao giờ được chạm tới. Mục 10.9 gỡ
            giới hạn một khối bằng cách giao cho mỗi khối một đoạn riêng và kết hợp các tổng khối; phiên bản kernel này của lab
            10.1 đã dịch <code>input</code> theo khối, để có thể đo thời gian nhiều hàng độc lập cùng lúc.
          </p>
        }
      >
        <p>
          The index depends only on <code>threadIdx.x</code>, so every block works on elements 0 to 2047, each overwriting
          partial sums the others are still reading, and no barrier spans blocks to order them. All three write{' '}
          <code>*output</code>. Elements 2048 to 4999 are never touched. Section 10.9 removes the one-block limit by giving
          each block its own segment and combining the block sums; lab 10.1's version of this kernel already offsets{' '}
          <code>input</code> by the block, so that many independent rows can be timed at once.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.3, pp. 227–228; §10.9, p. 244', // i18n-ignore: citation only
  },
};

// ---------------------------------------------------------------- the page
export function S103() {
  const cu = useLabFile('ch10/shared/inplace_sum.cu');
  const metal = useLabFile('ch10/shared/inplace_sum.metal');
  const status = useLabStatus(LAB10_1);
  const k = !missing(cu) ? functionSource(cu!, INPLACE_KERNEL.simple) : null;
  const m = metalFn(metal, 'simple_sum');
  const barrierLine = m ? m.text.split('\n').findIndex((l) => l.includes('threadgroup_barrier')) + 1 : 0;
  const offsetLines = k ? k.text.split('\n').flatMap((l, i) => (/blockIdx|output\[/.test(l) ? [i + 1] : [])) : [];
  return (
    <>
      <Translated
        vi={
          <p>
            Cây của mục trước cần các luồng trao tổng từng phần cho nhau ở mỗi tầng: một luồng chỉ được cộng giá trị của đối
            tác khi đối tác đã xong tầng trước. Các luồng chỉ có thể chờ nhau bên trong một khối, bằng rào chắn từ chương 4. Vì
            vậy các kernel đầu tiên dùng một khối duy nhất. Với N giá trị, bạn khởi chạy một khối N/2 luồng; với tối đa 1024
            luồng trong một khối, tức là tối đa 2048 giá trị. Mục 10.9 gỡ bỏ giới hạn này.
          </p>
        }
      >
        <p>
          The tree of the last section needs threads that hand partial sums to each other at every level: a thread may only
          add its partner's value once the partner has finished the level before. Threads can only wait for each other inside
          a block, with the barrier from chapter 4. So the first kernels use a single block. For N values you launch one
          block of N/2 threads; with at most 1024 threads in a block, that is at most 2048 values. Section 10.9 lifts the
          limit.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="10.3" pages="227–228" />
        <Aside>
          <Translated
            vi={
              <p>
                Ở p. 228 sách nói giới hạn 2048 giá trị được gỡ "in Section 10.10". Thật ra nó được gỡ ở mục 10.9, về đầu vào độ
                dài tùy ý; 10.10 là gộp việc cho luồng (thread coarsening).
              </p>
            }
          >
            <p>
              The book says on p. 228 that the 2048-value limit goes "in Section 10.10". It goes in section 10.9, on inputs of
              any length; 10.10 is thread coarsening.
            </p>
          </Translated>
        </Aside>
      </NoteGroup>

      <h2><Translated vi="Chủ sở hữu tính">Owner computes</Translated></h2>
      <Translated
        vi={
          <p>
            Đầu vào nằm trong bộ nhớ toàn cục và kernel ghi đè lên nó. Luồng t <b>sở hữu</b> phần tử 2t: nó là luồng duy nhất
            từng ghi vào đó. Ở vòng lặp đầu tiên, mọi luồng cộng láng giềng bên phải vào phần tử của mình. Ở vòng thứ hai, các
            tổng từng phần cách nhau hai, nên bước nhảy gấp đôi thành 2, và chỉ cách một luồng mới có việc: luồng 0 cộng phần tử
            2 (đang giữ 2 + 3 cũ) vào phần tử 0. Bước nhảy tiếp tục gấp đôi, và những luồng làm việc trong một vòng lặp có bước
            nhảy s là những luồng có chỉ số là bội của s. Đó là phép kiểm tra{' '}
            <code>threadIdx.x % stride == 0</code>. Cuối cùng, phần tử 0 giữ tổng, và luồng 0 chép nó sang{' '}
            <code>*output</code>.
          </p>
        }
      >
        <p>
          The input sits in global memory and the kernel overwrites it. Thread t <b>owns</b> element 2t: it is the only
          thread that ever writes there. In the first iteration every thread adds its right-hand neighbour into its own
          element. In the second, the partial sums sit two apart, so the stride doubles to 2, and only every second thread
          has work: thread 0 adds element 2 (which holds the old 2 + 3) into element 0. The stride keeps doubling, and the
          threads that work in an iteration with stride s are those whose index is a multiple of s. That is the test{' '}
          <code>threadIdx.x % stride == 0</code>. At the end, element 0 holds the total, and thread 0 copies it to{' '}
          <code>*output</code>.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="10.3" pages="228–230" figs={tr('Figs. 10.5, 10.6', 'Hình 10.5, 10.6')} />
        <Aside>
          <Translated
            vi={
              <p>
                Ở p. 229 sách nói bước nhảy tăng cho đến khi lớn hơn <code>blockIdx.x</code>, "tổng số luồng trong khối". Code,
                một cách đúng đắn, kiểm tra <code>blockDim.x</code>.
              </p>
            }
          >
            <p>
              On p. 229 the book says the stride grows until it is greater than <code>blockIdx.x</code>, "the total number of
              threads in the block". The code, rightly, tests <code>blockDim.x</code>.
            </p>
          </Translated>
        </Aside>
      </NoteGroup>
      {k ? (
        <Code
          tabs={[
            { label: tr('CUDA (lab 10.1)', 'CUDA (lab 10.1)'), code: k.text, tiers: TIERS, status: status.cu, highlight: offsetLines },
            ...(m ? [{ label: tr('Metal (lab 10.1)', 'Metal (lab 10.1)'), code: m.text, tiers: TIERS, status: status.metal, highlight: barrierLine ? [barrierLine] : [] }] : []),
          ]}
          caption={
            <Translated
              vi={
                <>
                  <code>SimpleSumReductionKernel</code> như lab 10.1 cung cấp, từ <code>labs/ch10/shared/inplace_sum.cu</code>{' '}
                  (và <code>simple_sum</code> từ <code>inplace_sum.metal</code>). Một thay đổi so với trang sách, ở các dòng được
                  đánh dấu: khối b làm việc trên hàng riêng của nó gồm 2·blockDim.x giá trị (<code>input</code> dịch đến đầu hàng b,
                  kết quả ghi vào <code>output[blockIdx.x]</code>), để lab có thể đo thời gian nhiều hàng trong một lần khởi chạy. Với
                  một khối, nó chính là kernel của sách. HIP viết mọi thứ trong đó giống CUDA, nên file được viết để biên dịch cho
                  HIP nguyên trạng (unverified: chưa có gì ở đây được build với ROCm).
                </>
              }
            >
              <>
                <code>SimpleSumReductionKernel</code> as lab 10.1 provides it, from <code>labs/ch10/shared/inplace_sum.cu</code>{' '}
                (and <code>simple_sum</code> from <code>inplace_sum.metal</code>). One change from the page, on the marked
                lines: block b works on its own row of 2·blockDim.x values (<code>input</code> moves to the start of row b,
                the result goes to <code>output[blockIdx.x]</code>), so the lab can time many rows in one launch. With one
                block it is the book's kernel. HIP spells everything in it as CUDA does, so the file is written to compile
                for HIP as it stands (unverified: nothing here has been built with ROCm).
              </>
            </Translated>
          }
        />
      ) : (
        <p className="muted"><Translated vi={<>Kernel được nạp từ <code>labs/ch10/shared/inplace_sum.cu</code>.</>}>The kernel is loaded from <code>labs/ch10/shared/inplace_sum.cu</code>.</Translated></p>
      )}

      <h2><Translated vi="Chạy từng bước">Step it</Translated></h2>
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy kernel một khối của sách cho mọi luồng của khối. Bắt đầu với thứ tự "mỗi luồng chạy xa
            nhất có thể, luồng 0 trước": luồng 0 làm phép cộng đầu tiên rồi dừng ở rào chắn, vì các luồng khác chưa đuổi kịp.
            Chạy đến rào chắn vài lần và xem phần tử 0 gom dần tổng trong khi các phần tử chẵn khác giữ những tổng từng phần
            chúng tạo ra trên đường đi.
          </p>
        }
      >
        <p>
          The stepper runs the book's one-block kernel for every thread of the block. Start with the order "each thread runs
          as far as it can, thread 0 first": thread 0 does its first addition and then stops at the barrier, because the
          others have not caught up. Run to the barrier a few times and watch element 0 collect the total while the other
          even elements keep the partial sums they made on the way.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Ở đây một luồng tiến từng phép toán mức mã nguồn một, và một phép cộng đọc và ghi trong một bước. "Lockstep" chạy
              mọi luồng mỗi vòng một phép toán, mọi lần đọc trước mọi lần ghi. Không có warp, không có thời gian. Mô hình trung
              thực ở một điều: một luồng thấy giá trị nào khi nó đọc.
            </p>
          }
        >
          <p>
            A thread advances one source-level operation at a time here, and an addition reads and writes in one step.
            "Lockstep" runs every thread one operation per round, all reads before all writes. No warps, no timing. The
            model is faithful about one thing: which value a thread finds when it reads.
          </p>
        </Translated>
      </Simplification>
      <Stepper />
      <Ask q={questionsS103['ch10/10.3/q-ex6a']} label={tr("Work it out, then check it in the stepper", 'Tự tính ra, rồi kiểm tra trong trình chạy từng bước')} />

      <h2><Translated vi="Rào chắn bảo đảm điều gì">What the barrier guarantees</Translated></h2>
      <Translated
        vi={
          <p>
            <code>__syncthreads()</code> bên trong vòng lặp khiến mọi luồng xong một vòng lặp trước khi bất kỳ luồng nào bắt đầu
            vòng tiếp theo. Đó đúng là điều cây cần: các tổng từng phần của một tầng đều được ghi xong trước khi tầng tiếp theo
            đọc chúng. Sách thêm một điểm thứ hai. Một giá trị mà một luồng ghi vào bộ nhớ không nên được luồng khác đọc khi
            không có memory fence; vì ở đây bên ghi và bên đọc cùng một khối, rào chắn cũng đóng vai trò fence đó.
          </p>
        }
      >
        <p>
          The <code>__syncthreads()</code> inside the loop makes every thread finish one iteration before any thread starts
          the next. That is exactly what the tree needs: a level's partial sums are all written before the next level reads
          them. The book adds a second point. A value one thread writes to memory should not be read by another without a
          memory fence; because writer and reader are in the same block here, the barrier serves as that fence too.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="10.3" pages="230" />
        <Today checked="2026-11-20 (ch10-facts A1)">{/* i18n-ignore: date and citation */}
          <Translated
            vi={
              <p>
                Lời gọi này "strongly happens before" khi bất kỳ luồng tham gia nào rời khỏi nó. "Memory consistency model" của sách
                giờ là mục 5.8 trong hướng dẫn của NVIDIA, nơi một data race là hành vi không xác định.
              </p>
            }
          >
            <p>
              The call "strongly happens before" any participating thread leaves it. The book's "memory consistency model" is
              now section 5.8 of NVIDIA's guide, where a data race is undefined behaviour.
            </p>
          </Translated>
        </Today>
      </NoteGroup>
      <Translated
        vi={
          <p>
            Vì vậy một rào chắn bị bỏ sót không chỉ là khả năng đọc phải giá trị cũ: theo luật của NVIDIA, chương trình khi đó
            không có ý nghĩa xác định. Bỏ chọn rào chắn trong trình chạy từng bước, giữ nguyên thứ tự, và bấm <i>Đến lỗi đầu
            tiên</i>. Rồi thử các thứ tự khác. Một số vẫn cho ra tổng đúng, và đó là loại lỗi tệ nhất: một bài kiểm tra qua được
            không chứng minh gì về thiết bị tiếp theo.
          </p>
        }
      >
        <p>
          So a dropped barrier is not merely a chance of reading an old value: by NVIDIA's rules the program then has no
          defined meaning. Untick the barrier in the stepper, keep the same order, and press <i>Go to the first failure</i>. Then try the
          other orders. Some give the right total anyway, which is the worst kind of bug: a test that passes proves nothing
          about the next device.
        </p>
      </Translated>
      <Ask q={questionsS103['ch10/10.3/q-stale-read']} />
      <Ask q={questionsS103['ch10/10.3/q-barrier-outside']} />

      <h2><Translated vi="Kernel này tốn gì">What this kernel costs</Translated></h2>
      <Translated
        vi={
          <p>
            Nó chạy đúng, và là một bức tranh công bằng về cây, nhưng hãy mang theo ba cái giá sang các mục tiếp theo. Nó phá
            hủy đầu vào: mảng kết thúc đầy các tổng từng phần. Các luồng hoạt động của nó rải khắp mọi warp, nên phần lớn warp
            bị phân kỳ ở phần lớn vòng lặp; mục 10.4 đếm điều đó. Và các luồng kề nhau chạm vào những phần tử cách nhau 2, rồi
            4, rồi 8, nên các truy cập bộ nhớ của chúng không gộp lại được; mục 10.5 đếm điều đó. Cả ba đều đến từ cùng một lựa
            chọn: luồng t sở hữu phần tử 2t.
          </p>
        }
      >
        <p>
          It works, and it is a fair picture of the tree, but carry three costs into the next sections. It destroys its
          input: the array ends full of partial sums. Its active threads are spread through every warp, so most warps
          diverge in most iterations; section 10.4 counts that. And neighbouring threads touch elements that are 2, then 4,
          then 8 apart, so their memory accesses do not combine; section 10.5 counts that. All three come from the same
          choice, that thread t owns element 2t.
        </p>
      </Translated>
      <Source section="10.3–10.5" pages="228, 230, 234" />
      <Ask q={questionsS103['ch10/10.3/q-three-blocks']} />

      <h2><Translated vi="Cùng kernel đó trên các backend khác">The same kernel on other backends</Translated></h2>
      <Translated
        vi={
          <p>
            HIP nhận file CUDA như đã viết (unverified: chưa có gì ở đây được build với ROCm), với cùng{' '}
            <code>__syncthreads()</code>, mà AMD mô tả là một rào chắn ngầm bao gồm cả thread fence. Metal là nơi kernel này cần
            cẩn thận. Rào chắn của nó,{' '}
            <code>threadgroup_barrier(flags)</code>, chỉ sắp thứ tự các loại bộ nhớ được nêu trong flags. Flag thông dụng,{' '}
            <code>mem_flags::mem_threadgroup</code>, bao phủ bộ nhớ threadgroup, còn kernel này giữ tổng từng phần trong bộ nhớ
            device. Vì vậy phiên bản Metal của lab 10.1 truyền <code>mem_flags::mem_device</code> (dòng được đánh dấu trong tab
            Metal). Với flag threadgroup, nó vẫn biên dịch, chạy, và không có lý do nào được tài liệu hóa để đúng.
          </p>
        }
      >
        <p>
          HIP takes the CUDA file as written (unverified: nothing here has been built with ROCm), with the same{' '}
          <code>__syncthreads()</code>, which AMD documents as a barrier that implicitly includes a thread fence. Metal is where this kernel needs care. Its barrier,{' '}
          <code>threadgroup_barrier(flags)</code>, orders only the kinds of memory named in its flags. The usual flag,{' '}
          <code>mem_flags::mem_threadgroup</code>, covers threadgroup memory, and this kernel keeps its partial sums in device
          memory. So lab 10.1's Metal version passes <code>mem_flags::mem_device</code> (the marked line in the Metal tab).
          With the threadgroup flag it would compile, run, and have no documented reason to be right.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. HIP: notes/today/ch10-facts.md, "Where the backends really differ" (rào chắn khối như memory
              fence). Metal: notes/today/ch05-facts.md C2, bảng <code>mem_flags</code>.
            </p>
          }
        >
          <p>
            The book uses CUDA only. HIP: notes/today/ch10-facts.md, "Where the backends really differ" (block barrier as
            memory fence). Metal: notes/today/ch05-facts.md C2, the table of <code>mem_flags</code>.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Bạn chạy và đo thời gian kernel này ở <a href={href('ch10', 'labs')}>lab 10.1</a>, bên cạnh kernel hội tụ của mục
            10.4 và các kernel nhanh hơn của mục 10.6 đến 10.10.
          </p>
        }
      >
        <p>
          You run and time this kernel in <a href={href('ch10', 'labs')}>lab 10.1</a>, beside the convergent kernel of
          section 10.4 and the faster kernels of sections 10.6 to 10.10.
        </p>
      </Translated>
    </>
  );
}
