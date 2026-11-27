import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile, type CodeTab } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  PRESETS,
  STOP_LABEL,
  overlapStep,
  parseValues,
  runToSorted,
  threadsFor,
  type Interleaving,
  type OverlapEvent,
  type StepResult,
  type StopRule,
} from '../../sim/oddeven';
import { functionSource } from '../../sim/toolchain';
import { OddEvenWalkScene } from './OddEvenWalkScene';
import { useSolutionStatus } from './solutionStatus';

const SNIPPET = 'ch14/snippets/s142_oddeven.cu';
const LAB = 'ch14/lab1_twosorts/solution/';
const STATUS = {
  get snippet() {
    return tr(
      'front-end-checked; emulator-checked; unverified on CUDA and HIP hardware',
      'front-end-checked; emulator-checked; unverified trên phần cứng CUDA và HIP',
    );
  },
};
/** Where each value of the kernels lives. The lab tabs also hold a host loop: there only the device pointers are coloured. */
const TIERS: Record<string, Tier> = {
  data: 'global', hasChanged: 'global', flag: 'global',
  i: 'reg', tmp: 'reg', a: 'reg', b: 'reg',
};
/** Host code: its locals live in host memory, so only the device pointers get a tier. */
const HOST_TIERS: Record<string, Tier> = { data_d: 'global', changed_d: 'global' };
const missing = (s: string | null) => !s || s.startsWith('// could not load');
/** Long code in a table cell may wrap instead of widening the page. */
const WRAP = { whiteSpace: 'normal', overflowWrap: 'anywhere' } as const;
/** A function of a source file without the comment lines above it, so line 1 is the book's line 01. */
function bare(src: string | null, name: string): string | null {
  if (missing(src)) return null;
  const f = functionSource(src!, name);
  if (!f) return null;
  const lines = f.text.split('\n');
  let k = 0;
  while (k < lines.length && lines[k].trim().startsWith('//')) k++;
  return lines.slice(k).join('\n');
}

// ---------------------------------------------------------------- one row of values with pair brackets
function PairRow({ values, s, mark, predicted, onToggle, name }: {
  values: number[];
  /** The step whose pairs are bracketed. */
  s?: StepResult;
  /** Show which pairs swapped. */
  mark?: boolean;
  predicted?: Set<number>;
  onToggle?: (t: number) => void;
  name: string;
}) {
  const n = values.length;
  const swappedAt = new Set(mark && s ? s.threads.filter((t) => t.swapped).flatMap((t) => [t.i, t.i + 1]) : []);
  const pairAt = new Set(s ? s.threads.filter((t) => t.active).flatMap((t) => [t.i, t.i + 1]) : []);
  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(${n}, minmax(1.7rem, 2.6rem))`, columnGap: 4, rowGap: 4 }} data-row={name} data-values={values.join(' ')}>
      {values.map((v, k) => (
        <div key={'v' + k} className="mx global" style={{ display: 'block' }}>
          <span className={swappedAt.has(k) ? 'hot' : pairAt.has(k) ? 'on' : ''} style={{ minWidth: 0 }}>{v}</span>
        </div>
      ))}
      {s
        ? s.threads.map((t) =>
            t.active ? (
              <div key={'b' + t.t} style={{ gridColumn: `${t.i + 1} / span 2`, gridRow: 2, textAlign: 'center' }} data-pair={`${t.i}-${t.i + 1}`}
                data-swapped={mark ? (t.swapped ? 'yes' : 'no') : ''}>
                <div style={{ borderLeft: '1.5px solid var(--ink)', borderRight: '1.5px solid var(--ink)', borderBottom: `${mark && t.swapped ? 3 : 1.5}px solid var(--ink)`, height: 7, margin: '0 0.7rem' }} />
                {onToggle ? (
                  <button aria-pressed={predicted?.has(t.t)} onClick={() => onToggle(t.t)} style={{ fontSize: '0.72rem', padding: '0.1rem 0.3rem', marginTop: 2 }}>
                    t{t.t} <Translated vi="đổi chỗ?">swaps?</Translated>
                  </button>
                ) : (
                  <div className="ui muted" style={{ fontSize: '0.72rem' }}>t{t.t}, i={t.i}</div>
                )}
              </div>
            ) : null,
          )
        : null}
    </div>
  );
}

// ---------------------------------------------------------------- the stepper
function Stepper() {
  const [preset, setPreset] = useState(0);
  const [text, setText] = useState(PRESETS[0].values.join(' '));
  const [values, setValues] = useState(PRESETS[0].values);
  const [rule, setRule] = useState<StopRule>('ours');
  const [pos, setPos] = useState(0);
  const [predict, setPredict] = useState(false);
  const [marked, setMarked] = useState<Set<number>>(new Set());
  const [verdict, setVerdict] = useState<{ right: number; total: number; step: number } | null>(null);
  const run = useMemo(() => runToSorted(values, rule), [values, rule]);
  const at = Math.min(pos, run.launched);
  const cur = at === 0 ? values : run.rows[at - 1].after;
  const last = at > 0 ? run.rows[at - 1] : undefined;
  const next = at < run.launched ? run.rows[at] : undefined;
  const done = at >= run.launched;
  const comparisons = run.rows.slice(0, at).reduce((s, r) => s + r.comparisons, 0);
  const swaps = run.rows.slice(0, at).reduce((s, r) => s + r.swaps, 0);
  const bad = parseValues(text) === null;

  useEffect(() => {
    setPos(0);
    setMarked(new Set());
    setVerdict(null);
  }, [values, rule]);

  const doStep = () => {
    if (!next) return;
    if (predict) {
      const actual = new Set(next.threads.filter((t) => t.swapped).map((t) => t.t));
      const active = next.threads.filter((t) => t.active);
      const right = active.filter((t) => actual.has(t.t) === marked.has(t.t)).length;
      setVerdict({ right, total: active.length, step: at + 1 });
    } else setVerdict(null);
    setMarked(new Set());
    setPos(at + 1);
  };
  const toggle = (t: number) => {
    const m = new Set(marked);
    if (m.has(t)) m.delete(t);
    else m.add(t);
    setMarked(m);
  };

  return (
    <Bench
      title={tr('Step odd-even sort', 'Chạy odd-even sort từng bước')}
      task={tr(
        "Run Fig. 14.1, then the trap under each stopping rule. Then switch on prediction and call each step's swaps before you see them.",
        'Chạy Hình 14.1, rồi chạy cái bẫy dưới từng quy tắc dừng. Sau đó bật dự đoán và đoán các lần đổi chỗ của mỗi bước trước khi bạn thấy chúng.',
      )}
      foot={
        <Translated
          vi={
            <>
              Mảng dữ liệu nằm trong bộ nhớ toàn cục (global memory). Các ô có ngoặc là các cặp của bước đang hiển thị; ô
              được tô là ô đã đổi chỗ. Mỗi bước là một lần khởi chạy <code>sort_kernel</code> với {'⌊'}n/2{'⌋'} luồng.
            </>
          }
        >
          <>
            The data array lives in global memory. Bracketed cells are the pairs of the step shown; filled cells swapped.
            Each step is one launch of <code>sort_kernel</code> with {'⌊'}n/2{'⌋'} threads.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="Giá trị">Values</Translated>
          <select value={preset} onChange={(e) => { const k = +e.target.value; setPreset(k); setValues(PRESETS[k].values); setText(PRESETS[k].values.join(' ')); }}>
            {PRESETS.map((p, k) => <option key={p.id} value={k}>{tr(p.label, ({ fig: 'Hình 14.1: 4 7 2 3 8 5 9 6', reversed: 'đảo ngược: 9 8 7 6 5 4 3 2', sorted: 'đã sắp xếp sẵn', largest: 'lớn nhất đứng đầu: 8 1 2 3 4 5 6 7', trap: 'cái bẫy: 1 3 2 4 5 6 7 8' } as Record<string, string>)[p.id] ?? p.label)}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="hoặc gõ 2 đến 8 giá trị">or type 2 to 8 values</Translated>
          <input type="text" value={text} aria-label={tr('values', 'giá trị')} style={{ width: '11rem' }}
            onChange={(e) => { setText(e.target.value); const v = parseValues(e.target.value); if (v) setValues(v); }} />
        </label>
        {bad ? <span className="chip fail"><Translated vi="2 đến 8 số nguyên nhỏ hơn 100">2 to 8 whole numbers below 100</Translated></span> : null}
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="Host dừng">Host stops</Translated>
          <select value={rule} onChange={(e) => setRule(e.target.value as StopRule)}>
            <option value="book">{tr(STOP_LABEL.book, 'sau một bước yên lặng (đúng nguyên văn sách)')}</option>
            <option value="ours">{tr(STOP_LABEL.ours, 'sau hai bước yên lặng liên tiếp, một chẵn và một lẻ (của khóa học này)')}</option>
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={predict} onChange={(e) => { setPredict(e.target.checked); setMarked(new Set()); setVerdict(null); }} />{' '}
          <Translated vi="dự đoán hàng tiếp theo">predict
          the next row</Translated>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPos(0); setVerdict(null); setMarked(new Set()); }} disabled={at === 0}>{tr('Restart', 'Bắt đầu lại')}</button>
        <button onClick={() => { setPos(at - 1); setVerdict(null); setMarked(new Set()); }} disabled={at === 0}>{tr('Back', 'Lùi')}</button>
        <button className="primary" onClick={doStep} disabled={done}>{predict ? tr('Check and step', 'Kiểm tra và bước') : tr('Step', 'Bước')}</button>
        <button onClick={() => { setPos(run.launched); setVerdict(null); setMarked(new Set()); }} disabled={done}>{tr('Run', 'Chạy')}</button>
        <span className="muted" aria-live="polite" data-pos={at}><Translated vi={<>bước {at} / {run.launched}</>}>step {at} of {run.launched}</Translated></span>
      </div>

      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>
        {verdict ? (
          <>
            <span className={'chip ' + (verdict.right === verdict.total ? 'ok' : 'fail')} data-verdict={`${verdict.right}/${verdict.total}`}>
              <Translated vi={<>bạn đoán đúng {verdict.right} trên {verdict.total} cặp</>}>you called {verdict.right} of {verdict.total} pairs right</Translated>
            </span>{' '}
          </>
        ) : null}
        {last ? (
          <Translated
            vi={
              <>
                Bước {last.n} sắp thứ tự các cặp {last.parity ? 'lẻ' : 'chẵn'}: {last.comparisons} luồng đã so sánh, {last.swaps} luồng
                đã đổi chỗ, nên cờ là {last.flag ? 1 : 0}.{' '}
              </>
            }
          >
            <>
              Step {last.n} ordered the {last.parity ? 'odd' : 'even'} pairs: {last.comparisons} threads compared, {last.swaps} swapped, so the
              flag is {last.flag ? 1 : 0}.{' '}
            </>
          </Translated>
        ) : (
          <Translated vi={<>Chưa có gì chạy. Bước 1 sắp thứ tự các cặp chẵn. </>}>
            <>Nothing has run yet. Step 1 orders the even pairs. </>
          </Translated>
        )}
        {done ? (
          <span data-stop={run.sorted ? 'sorted' : 'unsorted'}>
            <Translated
              vi={
                <>
                  Host dừng sau bước {run.launched}{run.launched ? '' : ' (ít hơn hai giá trị: không có gì để khởi chạy)'}.{' '}
                </>
              }
            >
              The host stops after step {run.launched}{run.launched ? '' : ' (fewer than two values: nothing to launch)'}.{' '}
            </Translated>
            {run.sorted ? <span className="chip ok"><Translated vi="đã sắp xếp">sorted</Translated></span> : <span className="chip fail"><Translated vi="chưa sắp xếp">not sorted</Translated></span>}
          </span>
        ) : next ? (
          <Translated vi={<>Tiếp theo: bước {next.n}, các cặp {next.parity ? 'lẻ' : 'chẵn'}.</>}>
            <>Next: step {next.n}, {next.parity ? 'odd' : 'even'} pairs.</>
          </Translated>
        ) : null}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)' }}>
        <div>
          {last ? (
            <>
              <div className="mx-label">
                <Translated vi={<>Bước {last.n} (các cặp {last.parity ? 'lẻ' : 'chẵn'}), trước và sau</>}>
                  Step {last.n} ({last.parity ? 'odd' : 'even'} pairs), before and after
                </Translated>
              </div>
              <PairRow values={last.before} s={last} mark name="before" />{/* i18n-ignore: row id for checks */}
              <div style={{ height: 6 }} />
              <PairRow values={last.after} name="after" />{/* i18n-ignore: row id for checks */}
            </>
          ) : (
            <>
              <div className="mx-label"><Translated vi="Đầu vào">Input</Translated></div>
              <PairRow values={cur} name="after" />{/* i18n-ignore: row id for checks */}
            </>
          )}
          {predict && next ? (
            <div style={{ marginTop: '1rem' }}>
              <div className="mx-label">
                <Translated vi={<>Bước {next.n} (các cặp {next.parity ? 'lẻ' : 'chẵn'}): đánh dấu các cặp bạn nghĩ sẽ đổi chỗ</>}>
                  Step {next.n} ({next.parity ? 'odd' : 'even'} pairs): mark the pairs you expect to swap
                </Translated>
              </div>
              <PairRow values={cur} s={next} predicted={marked} onToggle={toggle} name="predict" />{/* i18n-ignore: row id for checks */}
            </div>
          ) : null}
          <div className="mx-label" style={{ marginTop: '1rem' }}><Translated vi="Mọi hàng cho đến giờ">Every row so far</Translated></div>
          <table className="threads" aria-label={tr('Rows so far', 'Các hàng cho đến giờ')} data-history={at}>
            <tbody>
              <tr><td><Translated vi="đầu vào">input</Translated></td><td className="mono">{values.join(' ')}</td><td /></tr>
              {run.rows.slice(0, at).map((r) => (
                <tr key={r.n} className={r.n === at ? 'cur' : ''}>
                  <td>
                    <Translated vi={<>bước {r.n}, {r.parity ? 'lẻ' : 'chẵn'}</>}>
                      step {r.n}, {r.parity ? 'odd' : 'even'}
                    </Translated>
                  </td>
                  <td className="mono">{r.after.join(' ')}</td>
                  <td>{r.flag ? tr(`flag 1`, `cờ 1`) : tr('quiet', 'yên lặng')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <div className="mx-label">
            <Translated
              vi={<>Các luồng của {last ? `bước ${last.n}` : 'bước tiếp theo'} (i = 2t + {(last ?? next)?.parity ?? 0})</>}
            >
              Threads of {last ? `step ${last.n}` : 'the next step'} (i = 2t + {(last ?? next)?.parity ?? 0})
            </Translated>
          </div>
          <table className="threads" aria-label={tr('Threads', 'Các luồng')}>
            <thead>
              <tr><th>t</th><th>i</th><th><Translated vi="cặp">pair</Translated></th><th><Translated vi="làm gì">does</Translated></th></tr>
            </thead>
            <tbody>
              {Array.from({ length: threadsFor(values.length) }, (_, t) => {
                const s = last ?? next;
                const th = s?.threads[t];
                if (!th) return null;
                return (
                  <tr key={t} data-thread={t}>
                    <td>{t}</td>
                    <td>{th.i}</td>
                    <td>{th.active ? `(${th.i}, ${th.i + 1})` : tr('out of range', 'ngoài phạm vi')}</td>
                    <td>{!th.active ? tr('nothing: i + 1 < n fails', 'không gì: i + 1 < n sai') : !last ? tr('waits', 'chờ') : th.swapped ? tr(`${th.a} > ${th.b}: swaps, sets flag`, `${th.a} > ${th.b}: đổi chỗ, đặt cờ`) : tr(`${th.a} ≤ ${th.b}: no swap`, `${th.a} ≤ ${th.b}: không đổi chỗ`)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <table className="threads" aria-label={tr('Counters', 'Bộ đếm')} style={{ marginTop: '0.8rem' }}>
            <tbody>
              <tr><td><Translated vi="số bước đã khởi chạy">steps launched</Translated></td><td data-counter="steps">{at}</td></tr>
              <tr><td><Translated vi="số lần so sánh">comparisons</Translated></td><td data-counter="comparisons">{comparisons}</td></tr>
              <tr><td><Translated vi="số lần đổi chỗ">swaps</Translated></td><td data-counter="swaps">{swaps}</td></tr>
              <tr><td><Translated vi="đã sắp xếp sau bước">sorted after step</Translated></td><td data-counter="sortedAfter">{run.sortedAfter === null ? tr('not yet', 'chưa') : run.sortedAfter <= at ? run.sortedAfter : tr('not yet', 'chưa')}</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the bug hunt
const OP_TEXT: Record<OverlapEvent['op'], string> = {
  get loadA() { return tr('load', 'nạp'); },
  get loadB() { return tr('load', 'nạp'); },
  get storeA() { return tr('store', 'ghi'); },
  get storeB() { return tr('store', 'ghi'); },
};

function describeEvent(e: OverlapEvent): string {
  const r = e.regs[e.t];
  const who = tr(`Thread ${e.t} (owns ${e.t} and ${e.t + 1})`, `Luồng ${e.t} (sở hữu ${e.t} và ${e.t + 1})`);
  if (e.op === 'loadA') return tr(`${who} loads data[${e.index}] = ${e.value} into a.`, `${who} nạp data[${e.index}] = ${e.value} vào a.`);
  if (e.op === 'loadB')
    return tr(
      `${who} loads data[${e.index}] = ${e.value} into b. ${r.swap ? `${r.a} > ${r.b}, so it will swap.` : `${r.a} ≤ ${r.b}: nothing to do.`}`,
      `${who} nạp data[${e.index}] = ${e.value} vào b. ${r.swap ? `${r.a} > ${r.b}, nên nó sẽ đổi chỗ.` : `${r.a} ≤ ${r.b}: không có gì phải làm.`}`,
    );
  if (e.skipped) return tr(`${who} has nothing to store.`, `${who} không có gì để ghi.`);
  return tr(
    `${who} stores ${e.op === 'storeA' ? 'b' : 'a'} = ${e.value} into data[${e.index}].`,
    `${who} ghi ${e.op === 'storeA' ? 'b' : 'a'} = ${e.value} vào data[${e.index}].`,
  );
}

function NeighbourHunt() {
  const [preset, setPreset] = useState(1);
  const [kind, setKind] = useState<Interleaving['kind']>('race');
  const [atEl, setAtEl] = useState(4);
  const [pos, setPos] = useState(0);
  const values = PRESETS[preset].values;
  const il: Interleaving = kind === 'race' ? { kind, at: Math.min(atEl, values.length - 2) } : { kind };
  const run = useMemo(() => overlapStep(values, il), [values, kind, atEl]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => setPos(0), [preset, kind, atEl]);
  const total = run.events.length;
  const at = Math.min(pos, total);
  const ev = at > 0 ? run.events[at - 1] : undefined;
  const data = ev ? ev.data : values;
  const regs = ev ? ev.regs : values.slice(1).map(() => ({ done: false }) as OverlapEvent['regs'][number]);
  const finished = at >= total;
  const dup = new Set(finished ? run.duplicated : []);
  return (
    <Bench
      title={tr('Neighbouring pairs at once', 'Các cặp kề nhau cùng lúc')}
      task={tr(
        'Pick the race, step the loads and stores, and find the value that disappears.',
        'Chọn kiểu tranh chấp, chạy từng lần nạp và ghi, rồi tìm giá trị bị biến mất.',
      )}
      foot={
        <Translated
          vi={
            <>
              Mỗi luồng t sở hữu cặp (t, t + 1) trong cùng một bước, như một bubble sort song song ngây thơ sẽ làm: các
              luồng kề nhau dùng chung một phần tử. Một luồng nạp cả hai giá trị, so sánh, và nếu cần thì ghi cả hai trở
              lại.
            </>
          }
        >
          <>
            Every thread t owns the pair (t, t + 1) in the same step, as a naive parallel bubble sort would: neighbouring
            threads share an element. A thread loads both values, compares, and if needed stores both back.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="Giá trị">Values</Translated>
          <select value={preset} onChange={(e) => setPreset(+e.target.value)}>
            {PRESETS.map((p, k) => <option key={p.id} value={k}>{tr(p.label, ({ fig: 'Hình 14.1: 4 7 2 3 8 5 9 6', reversed: 'đảo ngược: 9 8 7 6 5 4 3 2', sorted: 'đã sắp xếp sẵn', largest: 'lớn nhất đứng đầu: 8 1 2 3 4 5 6 7', trap: 'cái bẫy: 1 3 2 4 5 6 7 8' } as Record<string, string>)[p.id] ?? p.label)}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Thứ tự xen kẽ">Interleaving</Translated>
          <select value={kind} onChange={(e) => setKind(e.target.value as Interleaving['kind'])}>
            <option value="in-order">{tr('threads one after another, 0 first', 'các luồng lần lượt, luồng 0 trước')}</option>
            <option value="reversed">{tr('threads one after another, last first', 'các luồng lần lượt, luồng cuối trước')}</option>
            <option value="race">{tr('two threads race on one element', 'hai luồng tranh chấp một phần tử')}</option>
          </select>
        </label>
        {kind === 'race' ? (
          <label className="field">
            <Translated vi="Phần tử dùng chung">Shared element</Translated>
            <select value={il.kind === 'race' ? il.at : 1} onChange={(e) => setAtEl(+e.target.value)}>
              {Array.from({ length: values.length - 2 }, (_, k) => k + 1).map((e) => (
                <option key={e} value={e}><Translated vi={<>data[{e}]: luồng {e - 1} và {e}</>}>data[{e}]: threads {e - 1} and {e}</Translated></option>
              ))}
            </select>
          </label>
        ) : null}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setPos(0)} disabled={at === 0}>{tr('Restart', 'Bắt đầu lại')}</button>
        <button onClick={() => setPos(at - 1)} disabled={at === 0}>{tr('Back', 'Lùi')}</button>
        <button className="primary" onClick={() => setPos(at + 1)} disabled={finished}>{tr('Step', 'Bước')}</button>
        <button onClick={() => setPos(total)} disabled={finished}>{tr('Run to the end', 'Chạy đến cuối')}</button>
        <span className="muted" aria-live="polite"><Translated vi={<>thao tác {at} / {total}</>}>operation {at} of {total}</Translated></span>
      </div>
      <div className={'narrate' + (finished && !run.permutation ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {ev ? describeEvent(ev) : tr('Nothing has run yet.', 'Chưa có gì chạy.')}
        {finished ? (
          <span data-result={run.permutation ? 'permutation' : 'broken'}>
            {' '}
            {run.permutation ? (
              <span className="chip ok"><Translated vi="vẫn là một hoán vị của đầu vào">still a permutation of the input</Translated></span>
            ) : (
              <Translated
                vi={
                  <>
                    <span className="chip fail">không phải hoán vị</span> bị mất: {run.lost.join(', ')}; bị nhân đôi: {run.duplicated.join(', ')}.
                  </>
                }
              >
                <>
                  <span className="chip fail">not a permutation</span> lost: {run.lost.join(', ')}; duplicated: {run.duplicated.join(', ')}.
                </>
              </Translated>
            )}
          </span>
        ) : null}
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 6fr)' }}>
        <div>
          <div className="mx-label t-global"><Translated vi="data (bộ nhớ toàn cục)">data (global memory)</Translated></div>
          <div className="mx global" data-row="data" data-values={data.join(' ')} style={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
            {data.map((v, k) => (
              <span key={k} className={dup.has(v) ? 'bad' : ev && ev.index === k && !ev.skipped ? 'hot' : ''}>
                {v}
                <sub>{k}</sub>
              </span>
            ))}
          </div>
          <div className="mx-label" style={{ marginTop: '0.8rem' }}><Translated vi="Đầu vào">Input</Translated></div>
          <div className="mono muted">{values.join(' ')}</div>
        </div>
        <div>
          <table className="threads" aria-label={tr('Naive threads', 'Các luồng ngây thơ')}>
            <thead>
              <tr><th>t</th><th><Translated vi="sở hữu">owns</Translated></th><th>a</th><th>b</th><th><Translated vi="đổi chỗ?">swap?</Translated></th><th><Translated vi="việc vừa làm">last did</Translated></th></tr>
            </thead>
            <tbody>
              {regs.map((r, t) => {
                const lastOp = run.events.slice(0, at).filter((e) => e.t === t).pop();
                return (
                  <tr key={t} className={ev?.t === t ? 'cur' : ''}>
                    <td>{t}</td>
                    <td>({t}, {t + 1})</td>
                    <td className="t-reg">{r.a ?? '·'}</td>
                    <td className="t-reg">{r.b ?? '·'}</td>
                    <td>{r.swap === undefined ? '' : r.swap ? tr('yes', 'có') : tr('no', 'không')}</td>
                    <td>{lastOp ? (lastOp.skipped ? tr('skipped a store', 'bỏ qua một lần ghi') : `${OP_TEXT[lastOp.op]} data[${lastOp.index}]`) : ''}</td>{/* i18n-ignore: the verb comes from OP_TEXT, translated */}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS142: Record<string, Question> = {
  'ch14/14.2/q-threads': {
    kind: 'numeric',
    id: 'ch14/14.2/q-threads',
    prompt: (
      <Translated
        vi={
          <>
            Hình 14.2 được khởi chạy với n/2 = 5 luồng cho n = 10. Ở một bước lẻ, bao nhiêu luồng trong số đó so sánh một
            cặp?
          </>
        }
      >
        Fig. 14.2 is launched with n/2 = 5 threads for n = 10. On an odd step, how many of those threads compare a pair?
      </Translated>
    ),
    answer: 4,
    explain: (
      <Translated
        vi={
          <p>
            Ở một bước lẻ, luồng t lấy i = 2t + 1: 1, 3, 5, 7, 9. Cặp của luồng 4 sẽ là (9, 10), mà phần tử 10 không tồn
            tại, nên kiểm tra biên thất bại và nó không làm gì. Hình 14.1 cho thấy sự hụt tương tự với n = 8: bốn luồng ở
            bước chẵn, ba ở bước lẻ.
          </p>
        }
      >
        <p>
          On an odd step thread t takes i = 2t + 1: 1, 3, 5, 7, 9. Thread 4's pair would be (9, 10), and element 10 does not
          exist, so the bounds test fails and it does nothing. Fig. 14.1 shows the same drop for n = 8: four threads on even
          steps, three on odd ones.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.2, Fig. 14.2, p. 332',
  },
  'ch14/14.2/q-worst': {
    kind: 'numeric',
    id: 'ch14/14.2/q-worst',
    prompt: (
      <Translated
        vi={
          <>
            Đầu vào là [8, 1, 2, 3, 4, 5, 6, 7], và bước đầu tiên là bước chẵn, như trong Hình 14.1. Sau bao nhiêu bước
            thì danh sách được sắp xếp?
          </>
        }
      >
        The input is [8, 1, 2, 3, 4, 5, 6, 7], and the first step is even, as in Fig. 14.1. After how many steps is the list
        sorted?
      </Translated>
    ),
    answer: 7,
    explain: (
      <Translated
        vi={
          <p>
            Mỗi bước so sánh số 8 với đúng một láng giềng, nên nó dịch một vị trí mỗi bước: vị trí 1, 2, ..., 7 sau các
            bước 1 đến 7, và các giá trị nó vượt qua lùi xuống một chỗ phía sau nó. Đã sắp xếp sau 7 bước. Host chưa biết
            điều đó: nó dừng sau hai bước liên tiếp đều yên lặng, đó là bước 8 (lẻ) và 9 (chẵn), nên tổng cộng nó khởi chạy
            9 lần. Vòng lặp của lab 14.1 chỉ đọc cờ sau mỗi cặp chẵn-lẻ, nên nó chạy cặp bước 9 và 10 rồi mới dừng: 10 lần
            khởi chạy.
          </p>
        }
      >
        <p>
          Each step compares the 8 with exactly one neighbour, so it moves one place per step: positions 1, 2, ..., 7 after
          steps 1 to 7, and the values it passes shift down by one behind it. Sorted after 7 steps. The host does not know
          that yet: it stops after two steps in a row, both quiet, and those are steps 8 (odd) and 9 (even), so it launches 9
          in all. Lab 14.1's loop reads the flag only after each even-odd pair, so it runs the pair of steps 9 and 10
          before it stops: 10 launches.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.2, p. 333; src/sim/oddeven.ts', // i18n-ignore: citation and file path
  },
  'ch14/14.2/q-flag': {
    kind: 'choice',
    id: 'ch14/14.2/q-flag',
    prompt: (
      <Translated vi={<>Mọi luồng đổi chỗ đều ghi 1 vào cùng một cờ, trong cùng một bước. Phát biểu nào đúng?</>}>
        <>Every thread that swaps writes 1 to the same flag, in the same step. Which statement is right?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Các lần ghi cùng một giá trị không thể tranh chấp: lần ghi nào đến sau cùng thì cờ vẫn đọc ra 1, và host chỉ cần biết vậy.">
            Writes of the same value cannot race: whichever store lands last, the flag reads 1, which is all the host needs to know.
          </Translated>
        ),
        why: (
          <Translated vi="Race là chuyện các lần truy cập không có thứ tự vào một vị trí, ít nhất một lần là ghi; giá trị không liên quan.">
            A race is about unordered accesses to one location, at least one a write; the values do not enter into it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Đặt __syncthreads() trước lệnh ghi, để các lần ghi có thứ tự và lệnh ghi thường được xác định.">
            Put __syncthreads() before the write, so that the stores are ordered and the plain store is defined.
          </Translated>
        ),
        why: (
          <Translated vi="Rào chắn chỉ sắp thứ tự các luồng của một khối, còn các luồng ghi ở đây đến từ mọi khối của lưới.">
            A barrier orders threads of one block only, and the writers here come from every block of the grid.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một data race, hành vi không xác định trong CUDA dù mọi lần ghi đều ghi 1. Hãy dùng phép toán nguyên tử; thứ tự relaxed là đủ.">
            A data race, undefined behaviour in CUDA though every write stores 1. Make it atomic; relaxed order is enough.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không xác định trên phần cứng AMD, nhưng vô hại trên NVIDIA, vì mô hình bộ nhớ của NVIDIA cho phép các lần ghi cùng giá trị.">
            Undefined on AMD hardware, but benign on NVIDIA, whose memory model allows equal-value stores.
          </Translated>
        ),
        why: (
          <Translated vi="Quy tắc này nằm trong chính mô hình bộ nhớ của NVIDIA, và nó nói về ngôn ngữ, không phải phần cứng của một hãng.">
            The rule is in NVIDIA's own memory model, and it is about the language, not one vendor's hardware.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sách gọi race này là vô hại trong thực tế và gợi ý dùng phép toán nguyên tử cho các lập trình viên thận trọng.
            Hướng dẫn hiện tại của NVIDIA nói thẳng hơn: mọi data race đều là hành vi không xác định.{' '}
            <code>atomicExch(hasChanged, 1u)</code> tốn ít ở đây, vì nó chỉ chạy trong các luồng đã đổi chỗ, và nó làm
            chương trình được xác định. Không cần thứ tự nào ngoài tính nguyên tử: host đọc cờ sau khi lần khởi chạy đã
            xong.
          </p>
        }
      >
        <p>
          The book calls this race benign in practice and suggests an atomic for conservative programmers. NVIDIA's current
          guide is blunter: any data race is undefined behaviour. <code>atomicExch(hasChanged, 1u)</code> costs little here,
          since it runs only in threads that swapped, and it makes the program defined. No ordering beyond atomicity is
          needed: the host reads the flag after the launch has completed.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.2, p. 333; notes/today/ch10-facts.md A1; notes/today/ch14-facts.md A9', // i18n-ignore: citation and file paths
  },
  'ch14/14.2/q-stop': {
    kind: 'choice',
    id: 'ch14/14.2/q-stop',
    prompt: (
      <Translated vi={<>Đầu vào bốn phần tử nào khiến một vòng lặp host dừng ở bước yên lặng đầu tiên trả về một danh sách chưa sắp xếp?</>}>
        <>Which four-element input makes a host loop that stops at the first quiet step return an unsorted list?</>
      </Translated>
    ),
    options: [
      { label: '[4, 3, 2, 1]', why: <Translated vi="Bước chẵn đầu tiên đổi chỗ (4, 3) và (2, 1).">The first even step swaps (4, 3) and (2, 1).</Translated> },
      { label: '[2, 1, 3, 4]', why: <Translated vi="Bước chẵn đầu tiên đổi chỗ (2, 1), và sau đó danh sách đã được sắp xếp.">The first even step swaps (2, 1), and then the list is sorted.</Translated> },
      { label: '[1, 2, 4, 3]', why: <Translated vi="Bước chẵn đầu tiên đổi chỗ (4, 3).">The first even step swaps (4, 3).</Translated> },
      { label: '[1, 3, 2, 4]', correct: true },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bước chẵn đầu tiên so sánh (1, 3) và (2, 4): cả hai đều đúng thứ tự, không đổi chỗ, cờ 0. Cặp sai thứ tự (3, 2)
            là một cặp lẻ, mà bước đó không bao giờ nhìn tới. Một bước chẵn yên lặng chỉ cho biết các cặp chẵn đúng thứ
            tự. Hãy dừng sau khi hai bước liên tiếp, một chẵn và một lẻ theo thứ tự bất kỳ, đều yên lặng: khi đó mọi cặp kề
            nhau đã được kiểm tra mà không có gì dịch chuyển ở giữa.
          </p>
        }
      >
        <p>
          The first even step compares (1, 3) and (2, 4): both in order, no swap, flag 0. The out-of-order pair (3, 2) is an
          odd pair, which that step never looks at. A quiet even step says only that the even pairs are in order. Stop after
          two steps in a row, one even and one odd in either order, are both quiet: then every adjacent pair has been
          checked without anything moving in between.
        </p>
      </Translated>
    ),
    source: 'our reading of p. 331 (notes/ch14.md, "Unclear in the PDF" item 7)',
    vi: { source: 'cách đọc của khóa học này cho p. 331 (notes/ch14.md, "Unclear in the PDF" mục 7)' },
  },
  'ch14/14.2/q-work': {
    kind: 'numeric',
    id: 'ch14/14.2/q-work',
    prompt: (
      <Translated vi={<>Với cận của sách là N bước, mỗi bước N/2 phép so sánh, odd-even sort thực hiện bao nhiêu phép so sánh cho N = 1,024?</>}>
        <>With the book's bound of N steps of N/2 comparisons each, how many comparisons does odd-even sort make for N = 1,024?</>
      </Translated>
    ),
    answer: 524288,
    explain: (
      <Translated
        vi={
          <p>
            1,024 × 512 = 524,288. Một phép sắp xếp cần cỡ N log2 N phép so sánh chỉ cần khoảng 1,024 × 10 = 10,240: ít hơn
            khoảng năm mươi lần. Chạy mỗi bước hoàn toàn song song làm thời gian thành O(N); nó không làm khối lượng công
            việc nhỏ đi chút nào.
          </p>
        }
      >
        <p>
          1,024 × 512 = 524,288. A sort that makes on the order of N log2 N comparisons needs about 1,024 × 10 = 10,240: about
          fifty times fewer. Running each step fully in parallel makes the time O(N); it does not make the work any smaller.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.2, p. 333',
  },
};

// ---------------------------------------------------------------- the page
export function S142() {
  const snip = useLabFile(SNIPPET);
  const kernel = bare(snip, 'sort_kernel');
  const host = missing(snip) ? null : functionSource(snip!, 'oddEvenSort')?.text ?? null;
  const labCu = useLabFile(LAB + 'kernel.cu');
  const labMetal = useLabFile(LAB + 'kernel.metal');
  const labMm = useLabFile(LAB + 'launch.mm');
  const labStatus = useSolutionStatus('ch14/lab1_twosorts');
  // Only the odd-even part of each lab file: the kernel and its host loop.
  const part = (src: string | null, names: string[]) => {
    const fs = missing(src) ? [] : names.map((nm) => functionSource(src!, nm)?.text);
    return fs.length && fs.every(Boolean) ? fs.join('\n\n') : null;
  };
  const labCuPart = part(labCu, ['oddEvenStep', 'launchOddEvenSort']);
  const labMetalPart = part(labMetal, ['odd_even_step']);
  const labMmPart = part(labMm, ['runOddEvenSort']);
  const tabs: CodeTab[] = kernel ? [{ label: tr('Fig. 14.2, corrected', 'Hình 14.2, đã sửa'), code: kernel, tiers: TIERS, highlight: [5, 10], status: STATUS.snippet }] : [];
  if (kernel && labCuPart) tabs.push({ label: tr('Lab 14.1: CUDA and HIP', 'Lab 14.1: CUDA và HIP'), code: labCuPart, tiers: TIERS, status: labStatus.cu });
  if (kernel && labMetalPart) tabs.push({ label: tr('Lab 14.1: Metal kernel', 'Lab 14.1: kernel Metal'), code: labMetalPart, tiers: TIERS, status: labStatus.metal });
  if (kernel && labMetalPart && labMmPart) tabs.push({ label: tr('Lab 14.1: Metal host', 'Lab 14.1: host Metal'), code: labMmPart, tiers: {}, status: labStatus.metal });
  return (
    <>
      <Translated
        vi={
          <p>
            Bubble sort so sánh các phần tử kề nhau và đổi chỗ những cặp sai thứ tự, hết lượt này đến lượt khác, cho đến khi
            không còn gì thay đổi. Trong một lượt, thứ tự duyệt các cặp không quan trọng, điều này trông như một lời mời giao mỗi
            cặp cho một luồng riêng.
          </p>
        }
      >
        <p>
          Bubble sort compares neighbours and swaps those out of order, pass after pass, until nothing changes. Within a pass
          the order in which the pairs are visited does not matter, which looks like an invitation to give every pair to its
          own thread.
        </p>
      </Translated>
      <Source section="14.2" pages="331" />
      <Translated
        vi={
          <p>
            Không phải vậy. Nếu một luồng sở hữu (0, 1) và một luồng khác sở hữu (1, 2), và cả hai cùng quyết định đổi chỗ, thì cả
            hai đều đọc và ghi phần tử 1. Luồng nào ghi sau cùng sẽ thắng, và kết quả thậm chí không nhất thiết chứa cùng các giá
            trị như đầu vào. Hãy thử xem.
          </p>
        }
      >
        <p>
          It is not. If one thread owns (0, 1) and another owns (1, 2), and both decide to swap, both read and write element 1.
          Whichever writes last wins, and the result need not even contain the same values as the input. Try it.
        </p>
      </Translated>
      <NeighbourHunt />
      <Translated
        vi={
          <p>
            Khi các luồng chạy lần lượt, bước đó là một lượt bubble sort bình thường và các giá trị được giữ nguyên. Tranh chấp
            cần cả hai luồng nạp trước khi luồng nào ghi, và cả hai đều đổi chỗ: khi đó một giá trị bị ghi hai lần và một giá trị
            khác bị mất. Mô hình chạy mỗi lần nạp và ghi như một thao tác không thể chia nhỏ; các luồng thật xen kẽ ở mức mịn
            hơn, điều đó chỉ thêm nhiều cách để sai.
          </p>
        }
      >
        <p>
          With the threads one after another, the step is an ordinary bubble-sort pass and the values survive. The race needs
          both threads to load before either stores, and both to swap: then one value is written twice and another is lost.
          The model runs each load and store as one indivisible operation; real threads interleave more finely, which only
          adds ways to go wrong.
        </p>
      </Translated>

      <h2>
        <Translated vi="Các cặp rời nhau, luân phiên">Disjoint pairs, alternating</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Cách sửa là chỉ làm việc trên các cặp không dùng chung gì. Ở một <b>bước chẵn</b>, mọi cặp có chỉ số đầu chẵn được sắp
            thứ tự: (0, 1), (2, 3), v.v. Ở một <b>bước lẻ</b>, mọi cặp có chỉ số đầu lẻ: (1, 2), (3, 4), ... Luân phiên hai bước
            cho đến khi không còn gì được đổi chỗ. Trong một bước không có hai luồng nào chạm vào cùng một phần tử, nên thứ tự của
            các luồng không thể ảnh hưởng. Đây là odd-even transposition sort; Hình 14.1 chạy nó trên tám giá trị, bước cuối trong
            sáu bước không thay đổi gì.
          </p>
        }
      >
        <p>
          The fix is to work only on pairs that share nothing. On an <b>even step</b> every pair whose first index is even is
          ordered: (0, 1), (2, 3), and so on. On an <b>odd step</b>, every pair whose first index is odd: (1, 2), (3, 4), ...
          Alternate the two until nothing is swapped. Within a step no two threads touch the same element, so the order of the
          threads cannot matter. This is odd-even transposition sort; Fig. 14.1 runs it on eight values, the last of six steps
          changing nothing.
        </p>
      </Translated>
      <Source section="14.2" pages="331–332" figs={tr('Fig. 14.1', 'Hình 14.1')} />
      <Translated
        vi={
          <p>
            Kernel thực hiện một bước. Nó được khởi chạy một lần mỗi bước với số luồng bằng một nửa số phần tử. Luồng t lấy cặp
            bắt đầu tại i = 2t ở bước chẵn và i = 2t + 1 ở bước lẻ, kiểm tra cặp đó nằm trong danh sách, đổi chỗ nếu cần, và đặt
            một cờ để host biết có gì đó đã dịch chuyển.
          </p>
        }
      >
        <p>
          The kernel performs one step. It is launched once per step with half as many threads as elements. Thread t takes
          the pair starting at i = 2t on an even step and i = 2t + 1 on an odd one, checks that the pair is inside the list,
          swaps if needed, and sets a flag so the host knows something moved.
        </p>
      </Translated>
      <Source section="14.2" pages="331–332" figs={tr('Fig. 14.2', 'Hình 14.2')} />
      {kernel ? (
        <Code
          tabs={tabs}
          caption={
            <Translated
              vi={
                <>
                  Hình 14.2 với số dòng của trang sách, từ <code>labs/{SNIPPET}</code>. Hai dòng khác với trang sách: dòng 05 là{' '}
                  <code>if(i + 1 &lt; N)</code> trong khi trang sách có <code>if(i &lt; N - 1)</code>, và dòng 10 dùng{' '}
                  <code>atomicExch</code> trong khi trang sách có <code>*hasChanged = 1;</code>. HIP có <code>atomicExch</code> cùng
                  tên, nên tệp được viết để phục vụ HIP qua header của khóa học; nó chưa được build cho HIP ở đây. Các tab còn lại là
                  phần odd-even trong lời giải tham chiếu (solution) của lab 14.1: hãy thử làm lab trước khi đọc chúng.
                </>
              }
            >
              <>
                Fig. 14.2 with the page's line numbers, from <code>labs/{SNIPPET}</code>. Two lines differ from the page:
                line 05 reads <code>if(i + 1 &lt; N)</code> where the page has <code>if(i &lt; N - 1)</code>, and line 10 uses{' '}
                <code>atomicExch</code> where the page has <code>*hasChanged = 1;</code>. HIP has <code>atomicExch</code> under
                the same name, so the file is written to serve HIP through the course's header; it has not been built for
                HIP here. The other tabs are the odd-even part of lab 14.1's reference solution: try the lab before
                reading them.
              </>
            </Translated>
          }
        />
      ) : null}
      <Translated
        vi={
          <p>
            Vì sao có hai thay đổi. <code>N</code> là số không dấu, nên với danh sách rỗng <code>N - 1</code> quay vòng thành giá
            trị không dấu lớn nhất, mọi luồng vượt qua phép kiểm tra, và mọi luồng đọc quá cuối mảng; <code>i + 1 &lt; N</code>{' '}
            hỏi cùng câu hỏi đó mà không cần phép trừ. Cờ có phần riêng ở bên dưới, sau khi bạn đã xem phép sắp xếp chạy.
          </p>
        }
      >
        <p>
          Why the two changes. <code>N</code> is unsigned, so for an empty list <code>N - 1</code> wraps to the largest
          unsigned value, every thread passes the test, and every thread reads past the end; <code>i + 1 &lt; N</code> asks
          the same question without the subtraction. The flag gets its own part below, after you have watched the sort run.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cả hai thay đổi là của khóa học này. Việc quay vòng được ghi trong notes/ch14.md (code listings, Fig. 14.2, "Care");
              trình giả lập (emulator) của khóa học chạy kernel đã sửa trên một danh sách rỗng và kiểm tra rằng không có gì bị chạm
              vào.
            </p>
          }
        >
          <p>
            Both changes are ours. The wrap is noted in notes/ch14.md (code listings, Fig. 14.2, "Care"); the course
            emulator runs the corrected kernel on an empty list and checks that nothing is touched.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS142['ch14/14.2/q-threads']} label={tr('Work it out', 'Tự tính')} />

      <h2>
        <Translated vi="Xem nó chạy, rồi tự điều khiển">Watch it, then drive it</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hoạt hình cho thấy những gì trình chạy từng bước bên dưới cho bạn kiểm chứng: vì sao các cặp không bao giờ va nhau, một
            giá trị có thể đi xa bao nhiêu trong một bước, mỗi bước tốn bao nhiêu, và khi nào thì dừng an toàn.
          </p>
        }
      >
        <p>
          The animation shows what the stepper below lets you test: why the pairs never collide, how far a value can travel
          in one step, what each step costs, and when it is safe to stop.
        </p>
      </Translated>
      <OddEvenWalkScene />
      <Translated
        vi={
          <p>
            Giờ hãy tự điều khiển. Chạy Hình 14.1 trước và so từng hàng với hình của sách; rồi nạp cái bẫy và chạy nó dưới từng quy
            tắc dừng.
          </p>
        }
      >
        <p>
          Now drive it yourself. Run Fig. 14.1 first and compare each row with the book's figure; then load the trap and run
          it under each stopping rule.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Phép so sánh và đổi chỗ của một luồng là một bước duy nhất của mô hình; các luồng thật xen kẽ ở mức lệnh và chạy theo
              warp. Phân kỳ luồng (divergence) không được mô hình hóa: khi vài luồng của một warp đổi chỗ còn các luồng khác thì
              không, nhánh đổi chỗ có một chi phí mà mô hình bỏ qua (trên Metal chi phí đó không được tài liệu hóa:
              notes/today/ch04-facts.md C4). Không có thời gian.
            </p>
          }
        >
          <p>
            One thread's compare and swap is a single step of the model; real threads interleave at the instruction level and
            run in warps. Divergence is not modelled: when some threads of a warp swap and others do not, the swap branch
            has a cost the model ignores (on Metal that cost is not documented: notes/today/ch04-facts.md C4). No timing.
          </p>
        </Translated>
      </Simplification>
      <Stepper />

      <h2>
        <Translated vi="Một race ghi cùng một giá trị">A race that writes the same value</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Nhìn lại dòng 10 như in trong sách: mọi luồng đổi chỗ đều ghi 1 vào <code>hasChanged</code>, nhiều luồng trong cùng
            một bước, không có gì sắp thứ tự các lần ghi. Đó là một race condition (tranh chấp dữ liệu): một vị trí, các lần truy
            cập không có thứ tự, ít nhất một lần là ghi. Sách gọi nó là vô hại, vì mọi lần ghi đều ghi cùng một giá trị và ghi 1
            hai lần cũng như ghi một lần (phép toán là lũy đẳng), rồi thêm lưu ý quan trọng: trong mô hình bộ nhớ C++, data race
            là hành vi không xác định ngay cả khi nó lũy đẳng. Mã có thể chạy được; không gì bảo đảm điều đó. Theo sách, các lập
            trình viên thận trọng dùng phép toán nguyên tử (atomic).
          </p>
        }
      >
        <p>
          Look again at line 10 as printed: every thread that swaps writes 1 to <code>hasChanged</code>, many of them in the
          same step, with nothing ordering the writes. That is a data race: one location, unordered accesses, at least one a
          write. The book calls it benign, since every write stores the same value and storing 1 twice is the same as storing
          it once (the operation is idempotent), and then adds the caveat that matters: in the C++ memory model a data race is
          undefined behaviour even when it is idempotent. The code may work; nothing guarantees it. Conservative programmers,
          it says, use an atomic.
        </p>
      </Translated>
      <Source section="14.2" pages="333" />
      <Translated
        vi={
          <p>
            Hướng dẫn hiện tại của NVIDIA nêu thẳng quy tắc cho CUDA: mọi data race đều dẫn tới hành vi không xác định. Vì vậy
            đoạn mã làm lệnh ghi thành nguyên tử. Thứ tự relaxed là đủ: host chỉ đọc cờ sau khi lần khởi chạy đã xong, và không có
            gì khác phụ thuộc vào thời điểm lần ghi trở nên thấy được.
          </p>
        }
      >
        <p>
          NVIDIA's current guide states the rule for CUDA directly: any data race results in undefined behaviour. So the
          snippet makes the store atomic. Relaxed order is enough: the host reads the flag only after the launch has finished,
          and nothing else depends on when the write becomes visible.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch10-facts.md A1; ch14-facts.md A9, B7; infra-facts.md A1, A2, B1, B2, C1)', '2026-11-20 (notes/today/ch10-facts.md mục A1; ch14-facts.md mục A9, B7; infra-facts.md mục A1, A2, B1, B2, C1)')}>
        <Translated
          vi={
            <p>
              CUDA: <code>atomicExch</code> có thứ tự relaxed và phạm vi device; <code>cuda::atomic_ref</code> với{' '}
              <code>store(1u, cuda::memory_order_relaxed)</code> là dạng NVIDIA hiện khuyên dùng. HIP có <code>atomicExch</code>{' '}
              cùng tên; tài liệu của nó không nêu quy tắc nào cho các lần ghi thường có tranh chấp (chúng tôi không xác minh được quy
              tắc nào). Metal cần cờ được khai báo là một kiểu nguyên tử.
            </p>
          }
        >
          <p>
            CUDA: <code>atomicExch</code> is relaxed and device-scoped; <code>cuda::atomic_ref</code> with{' '}
            <code>store(1u, cuda::memory_order_relaxed)</code> is the form NVIDIA now recommends. HIP has{' '}
            <code>atomicExch</code> under the same name; its documentation states no rule for plain racy writes (we could not
            verify one). Metal needs the flag declared as an atomic type.
          </p>
        </Translated>
      </Today>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th><Translated vi="Đặt cờ">Setting the flag</Translated></th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="Lệnh ghi nguyên tử">The atomic store</Translated></td>
            <td><code>atomicExch(flag, 1u)</code>, <Translated vi="hoặc">or</Translated> <code style={WRAP}>cuda::atomic_ref&lt;unsigned int, cuda::thread_scope_device&gt;(*flag).store(1u, cuda::memory_order_relaxed)</code></td>
            <td>
              <Translated vi={<><code>atomicExch(flag, 1u)</code>; <code>cuda::atomic_ref</code> qua libhipcxx, một phần của ROCm từ 10.1</>}>
                <code>atomicExch(flag, 1u)</code>; <code>cuda::atomic_ref</code> through libhipcxx, part of ROCm from 10.1
              </Translated>
            </td>
            <td>
              <Translated vi={<><code style={WRAP}>atomic_store_explicit(flag, 1u, memory_order_relaxed)</code>, với <code>flag</code> là một <code>device atomic_uint*</code></>}>
                <code style={WRAP}>atomic_store_explicit(flag, 1u, memory_order_relaxed)</code>, with <code>flag</code> a <code>device atomic_uint*</code>
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Một lần ghi thường có tranh chấp">A plain racy write</Translated></td>
            <td><Translated vi="hành vi không xác định">undefined behaviour</Translated></td>
            <td><Translated vi="tài liệu HIP không nêu quy tắc nào">no rule stated in the HIP documentation</Translated></td>
            <td><Translated vi="không được định nghĩa cho các truy cập thường">not defined for plain accesses</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Đọc nó trên host">Reading it on the host</Translated></td>
            <td>
              <Translated vi={<><code>cudaMemcpy</code> ngược về: nó chỉ trả về khi việc sao chép xong, và việc sao chép chờ lần khởi chạy trước nó</>}>
                <code>cudaMemcpy</code> back: it returns only when the copy is done, and the copy waits for the launch before it
              </Translated>
            </td>
            <td>
              <Translated
                vi={
                  <>
                    <code>hipMemcpy</code> ngược về, qua cùng mã nguồn. HIP ghi trong tài liệu rằng nó chạy sau lần khởi chạy
                    trong cùng stream; việc nó chỉ trả về khi sao chép xong là ngầm hiểu, không được nêu (ch02-facts B3). Một
                    lệnh <code>hipDeviceSynchronize()</code> sau lần sao chép, trước khi đọc cờ, làm việc chờ trở nên tường
                    minh.
                  </>
                }
              >
                <code>hipMemcpy</code> back, through the same source. HIP documents that it runs after the launch in the same
                stream; that it returns only when the copy is done is implied, not stated (ch02-facts B3). A{' '}
                <code>hipDeviceSynchronize()</code> after the copy, before reading the flag, makes the wait explicit.
              </Translated>
            </td>
            <td>
              <Translated vi={<>sau khi command buffer đã hoàn tất (ví dụ <code>waitUntilCompleted</code>), đọc shared buffer</>}>
                after the command buffer has completed (for example <code>waitUntilCompleted</code>), read the shared buffer
              </Translated>
            </td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Bạn có thể xem quy tắc được thực thi. Trình giả lập phát hiện race của khóa học chạy các luồng của một lần khởi chạy
            thật sự đồng thời dưới ThreadSanitizer. Khi build với Hình 14.2 như in trong sách, một bước trên đầu vào đảo ngược báo
            một data race ở dòng <code>*hasChanged = 1;</code>, dù phép sắp xếp ra đúng; khi build với <code>atomicExch</code>{' '}
            của đoạn mã, cùng bước đó không báo gì.
          </p>
        }
      >
        <p>
          You can watch the rule being enforced. The course's race-detection emulator runs the threads of a launch truly
          concurrently under ThreadSanitizer. Built with Fig. 14.2 as printed, one step on reversed input reports a data race
          on the line <code>*hasChanged = 1;</code>, although the sort comes out right; built with the snippet's{' '}
          <code>atomicExch</code>, the same step reports nothing.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Phép kiểm tra đó nằm trong <code>app/src/chapters/ch14/S142.snippets.test.ts</code>, với chương trình điều khiển
              (driver) trong <code>labs/ch14/snippets/check/</code>. Nó là emulator-checked: không có GPU nào tham gia.
            </p>
          }
        >
          <p>
            That check is in <code>app/src/chapters/ch14/S142.snippets.test.ts</code>, with its driver in{' '}
            <code>labs/ch14/snippets/check/</code>. It is emulator-checked: no GPU was involved.
          </p>
        </Translated>
      </Aside>
      <Ask q={questionsS142['ch14/14.2/q-flag']} />

      <h2>
        <Translated vi="Khi nào dừng">When to stop</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách bảo cứ luân phiên cho đến khi không còn phần tử nào bị đổi chỗ, và không đưa ra vòng lặp host. Hiểu theo nghĩa
            đen, "dừng sau một bước không có lần đổi chỗ nào" là sai: cái bẫy trong trình chạy từng bước, hoặc đơn giản là [1, 3,
            2, 4], không đổi chỗ gì ở bước chẵn đầu tiên mà vẫn chưa được sắp xếp. Một bước chẵn yên lặng chỉ chứng minh các cặp
            chẵn đúng thứ tự. Điều thực sự chứng minh được: hai bước liên tiếp, một chẵn và một lẻ theo thứ tự bất kỳ, đều yên
            lặng. Khi đó mọi cặp kề nhau đã được kiểm tra mà không có gì dịch chuyển ở giữa.
          </p>
        }
      >
        <p>
          The book says to keep alternating until no elements are swapped, and does not show the host loop. Read literally,
          "stop after a step with no swap" is wrong: the trap in the stepper, or simply [1, 3, 2, 4], makes no swap on its
          first even step and is not sorted. A quiet even step proves only that the even pairs are in order. What does prove
          it: two steps in a row, one even and one odd in either order, both quiet. Then every adjacent pair has been
          checked with nothing moving in between.
        </p>
      </Translated>
      <Source section="14.2" pages="331" />
      <Ask q={questionsS142['ch14/14.2/q-stop']} />
      <Translated
        vi={
          <p>
            Đây là vòng lặp trong đoạn mã. Việc đọc cờ có một điều kiện tiên quyết mà sách để ngầm: bước đó phải đã xong. Trên CUDA,
            lần sao chép ngược về bảo đảm điều đó, vì <code>cudaMemcpy</code> không trả về cho đến khi xong và nó chạy sau lần khởi
            chạy trong cùng stream. Trên Metal, cờ nằm trong một shared buffer và có thể được đọc khi command buffer đã chạy bước
            đó hoàn tất.
          </p>
        }
      >
        <p>
          Here is the loop in the snippet. Reading the flag has a precondition the book leaves implicit: the step must have
          finished. On CUDA the copy back provides that, because <code>cudaMemcpy</code> does not return until it is done and
          it runs after the launch in the same stream. On Metal the flag lives in a shared buffer and may be read once the
          command buffer that ran the step has completed.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Vòng lặp host là của khóa học này, quy tắc dừng cũng vậy. Hành vi sao chép của CUDA: notes/today/ch02-facts.md A4 và
              A5. Metal: ch05-facts.md C6, infra-facts.md C12.
            </p>
          }
        >
          <p>
            The host loop is ours, as is the stopping rule. CUDA's copy behaviour: notes/today/ch02-facts.md A4 and A5.
            Metal: ch05-facts.md C6, infra-facts.md C12.
          </p>
        </Translated>
      </Beyond>
      {host ? (
        <Code
          tabs={[{ label: tr('host loop', 'vòng lặp host'), code: host, tiers: HOST_TIERS, status: STATUS.snippet }]}
          caption={
            <Translated vi={<>Từ cùng tệp đó. Bộ đếm bước yên lặng đặt lại mỗi khi một bước có đổi chỗ.</>}>
              <>From the same file. The quiet count resets whenever a step swaps.</>
            </Translated>
          }
        />
      ) : null}

      <h2>
        <Translated vi="Nó tốn bao nhiêu">What it costs</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trong trường hợp xấu nhất, giá trị lớn nhất bắt đầu ở đầu danh sách. Mỗi bước so sánh nó với một láng giềng, nên nó
            dịch một vị trí mỗi bước và cần khoảng N bước để tới cuối. Với đủ luồng để chạy trọn một bước song song, thời gian là
            O(N). Mỗi bước thực hiện khoảng N/2 phép so sánh, nên khối lượng công việc là O(N²): nhiều hơn hẳn cỡ N log N mà một
            phép sắp xếp tuần tự tốt cần.
          </p>
        }
      >
        <p>
          In the worst case the largest value starts at the front. Each step compares it with one neighbour, so it moves one
          place per step and needs about N steps to reach the end. With enough threads to run a step entirely in parallel, the
          time is O(N). Each step makes about N/2 comparisons, so the work is O(N²): far more than the order of N log N a
          good sequential sort needs.
        </p>
      </Translated>
      <Source section="14.2" pages="333" />
      <Ask q={questionsS142['ch14/14.2/q-worst']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS142['ch14/14.2/q-work']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Trên GPU có một chi phí mà O(N) che đi: mỗi bước là một lần khởi chạy, và đoạn mã theo sau mỗi lần khởi chạy bằng một
            lần sao chép ngược về để đọc cờ. Với một danh sách dài, đó là hàng nghìn lượt đi về. Lời giải tham chiếu của lab 14.1
            giảm một nửa số lần đọc: nó khởi chạy một bước chẵn và một bước lẻ liền nhau và đọc cờ một lần cho cả cặp. Cái giá là
            có thể thêm tối đa một bước, vì hai bước yên lặng nằm vắt qua hai cặp, một lẻ rồi một chẵn, không làm nó dừng: với [8,
            1, ..., 7] nó khởi chạy 10 bước trong khi vòng lặp của đoạn mã khởi chạy 9. Lab chạy nó trên GPU của bạn (xem{' '}
            <a href={href('ch14', 'labs')}>trang lab</a>); trước khi chạy, hãy quyết định xem một danh sách vài nghìn giá trị sẽ
            được sắp xếp nhanh hơn bởi kernel này hay bởi một luồng trên CPU.
          </p>
        }
      >
        <p>
          On a GPU there is a cost the O(N) hides: each step is a launch, and the snippet follows each launch with a copy
          back to read the flag. For a long list that is thousands of round trips. Lab 14.1's reference halves the reads: it
          launches an even and an odd step back to back and reads the flag once for the pair. The price is up to one more
          step, because two quiet steps that straddle two pairs, an odd then an even, do not stop it: for [8, 1, ..., 7] it
          launches 10 steps where the snippet's loop launches 9. The lab runs it on your GPU (see the{' '}
          <a href={href('ch14', 'labs')}>labs page</a>); before you run it, decide whether a list of a few thousand values
          would be sorted faster by this kernel or by one thread on the CPU.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Nơi odd-even sort thực sự có chỗ đứng là bên trong một luồng duy nhất. Phép sắp xếp GPU của MLX bắt đầu bằng việc mỗi
            luồng sắp xếp vài giá trị trong các thanh ghi (register) của chính nó, và vòng lặp nó dùng chính là vòng này: luân
            phiên các cặp chẵn và lẻ, một số lần cố định, không cờ và không lần khởi chạy nào. Bên trong một luồng không có tranh
            chấp nào cần tránh và không có host nào để hỏi, và với một nhúm giá trị thì N² vẫn nhỏ; mục 14.9 gọi các mẫu cố định
            như vậy là mạng sắp xếp.
          </p>
        }
      >
        <p>
          Where odd-even sort does earn its place is inside a single thread. MLX's GPU sort begins with each thread sorting
          the few values in its own registers, and the loop it uses is exactly this one: alternate even and odd pairs, a fixed
          number of times, with no flag and no launches. Inside one thread there is no race to avoid and no host to ask,
          and for a handful of values N² stays small; section 14.9 calls such fixed patterns sorting networks.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch14-facts.md, D4)', '2026-11-20 (notes/today/ch14-facts.md, mục D4)')}>
        <Translated
          vi={
            <p>
              <code>mlx/backend/metal/kernels/sort.h</code>, <code>ThreadSort</code>: <code>for (i = 0; i &lt; N_PER_THREAD; ++i)</code>{' '}
              trên <code>j = i &amp; 1, j + 2, ...</code>, đổi chỗ <code>vals[j]</code> và <code>vals[j + 1]</code> khi sai thứ tự.
            </p>
          }
        >
          <p>
            <code>mlx/backend/metal/kernels/sort.h</code>, <code>ThreadSort</code>: <code>for (i = 0; i &lt; N_PER_THREAD; ++i)</code>{' '}
            over <code>j = i &amp; 1, j + 2, ...</code>, swapping <code>vals[j]</code> and <code>vals[j + 1]</code> when out of
            order.
          </p>
        </Translated>
      </Today>
    </>
  );
}
