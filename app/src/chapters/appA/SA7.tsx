import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  ARITH_LABEL,
  EPS_EXPONENTS,
  bookPanels,
  busyFraction,
  captionText,
  diffPanels,
  equation,
  printedA11,
  system,
  trace,
  type ArithId,
  type Phase,
  type SystemId,
} from '../../sim/gauss';
import { GaussScene, STATE_LABEL, Timeline, timelineWidth } from './GaussScene';

const minus = (s: string) => s.replace(/-/g, '−');
const cellText = (t: string) => (t === '0' ? '·' : minus(t));
const bad = (v: number) => !Number.isFinite(v);

// ---------------------------------------------------------------- the stepper
interface Config {
  sys: SystemId;
  pivot: boolean;
  arith: ArithId;
  eps: number;
}

function opFor(p: Phase, t: number): string {
  const s = p.threads[t];
  if (p.kind === 'divide' && p.divisors?.[t]) return '÷ ' + minus(p.divisors[t]!.text);
  if (p.kind === 'subtract' && s === 'working') return tr(`− row ${p.col + 1}`, `− hàng ${p.col + 1}`);
  if (p.kind === 'pivot' && s === 'searching') {
    const j = p.pivot!.rows.indexOf(t);
    return `|${minus(p.pivot!.abs[j].text)}|` + (p.pivot!.winner === t ? tr(', the largest', ', lớn nhất') : '');
  }
  if (p.kind === 'swap' && s === 'working') return tr(`swaps with row ${(p.swap![0] === t ? p.swap![1] : p.swap![0]) + 1}`, `hoán đổi với hàng ${(p.swap![0] === t ? p.swap![1] : p.swap![0]) + 1}`);
  if (p.kind === 'backsub' && s === 'working') return tr('substitutes', 'thế ngược');
  return '';
}

function GaussStepper({ title, task, init }: { title: string; task: string; init: Config }) {
  const [cfg, setCfg] = useState<Config>(init);
  const [view, setView] = useState<'matrix' | 'equations'>('matrix');
  const [i, setI] = useState(0);
  const sys = useMemo(() => system(cfg.sys, cfg.eps), [cfg.sys, cfg.eps]);
  const locale = useLocale();
  const t = useMemo(() => trace(sys, { pivot: cfg.pivot, arith: cfg.arith }), [sys, cfg.pivot, cfg.arith, locale]);
  useEffect(() => setI(0), [t]);
  const last = t.phases.length - 1;
  const at = Math.min(i, last);
  const ph = t.phases[at];
  const n = t.n;
  const anyBad = ph.m.some((r) => r.some((c) => bad(c.v)));
  const upTo = { ...t, phases: t.phases.slice(0, at + 1) };
  const busySoFar = upTo.phases.some((p) => p.kind === 'divide' || p.kind === 'subtract') ? busyFraction(upTo) : null;
  const set = (patch: Partial<Config>) => setCfg((c) => ({ ...c, ...patch }));
  const lane = 30;
  const tw = timelineWidth(t.phases);
  const exactSol = sys.solution;
  const done = ph.kind === 'done';
  const exact = useMemo(() => trace(sys, { pivot: true, arith: 'exact' }).solution, [sys]);
  // Largest relative error of the computed solution; above 10% the answer counts as wrong.
  const relErr = Math.max(...t.solution.map((c, k) => Math.abs(c.v - exact[k].v) / Math.max(1e-30, Math.abs(exact[k].v))));
  const close = t.finite && relErr <= 0.1;
  return (
    <Bench
      title={title}
      task={task}
      foot={
        <Translated
          vi={
            <>
              Các hàng được đánh số như trong sách (hàng 1 ở trên cùng); luồng t sở hữu hàng t + 1. Phân số chính xác giữ vô cực
              và NaN như số học IEEE; FP32 và FP16 làm tròn mọi phép toán về chẵn gần nhất. Trên dòng thời gian, ô tối là luồng
              đang làm việc, ô nét đứt là luồng bỏ qua, ô xám là luồng rảnh, và các vạch dọc là rào chắn.
            </>
          }
        >
          <>
            Rows are numbered as in the book (row 1 at the top); thread t owns row t + 1. Exact fractions keep infinity and NaN
            as IEEE arithmetic would; FP32 and FP16 round every operation to nearest even. In the timeline, dark cells are
            threads working, dashed cells threads skipping, grey cells idle threads, and the vertical bars are barriers.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Hệ phương trình">System</Translated>
          <select value={cfg.sys} onChange={(e) => set({ sys: e.target.value as SystemId })} aria-label={tr('system', 'hệ phương trình')}>
            <option value="book1">{tr("The book's first system", 'Hệ thứ nhất của sách')}</option>
            <option value="book2">{tr("The book's second (no X in row 1)", 'Hệ thứ hai của sách (không có X ở hàng 1)')}</option>
            <option value="tiny">{tr('2 × 2 with a tiny pivot ε (ours)', '2 × 2 với phần tử chốt rất nhỏ ε (của khóa học này)')}</option>
          </select>
        </label>
        {cfg.sys === 'tiny' ? (
          <label className="field">
            ε = 1e−{cfg.eps}
            <input type="range" min={EPS_EXPONENTS[0]} max={EPS_EXPONENTS[EPS_EXPONENTS.length - 1]} value={cfg.eps} aria-label={tr('epsilon exponent', 'số mũ của epsilon')}
              onChange={(e) => set({ eps: +e.target.value })} />
          </label>
        ) : null}
        <label className="check">
          <input type="checkbox" checked={cfg.pivot} onChange={(e) => set({ pivot: e.target.checked })} /> <Translated vi="chọn phần tử chốt">pivoting</Translated>
        </label>
        <label className="field">
          <Translated vi="Số học">Arithmetic</Translated>
          <span className="seg" role="group" aria-label={tr('arithmetic', 'số học')}>
            {(['exact', 'fp32', 'fp16'] as ArithId[]).map((a) => (
              <button key={a} aria-pressed={cfg.arith === a} onClick={() => set({ arith: a })}><Translated vi={{ exact: 'phân số chính xác', fp32: 'FP32', fp16: 'FP16' }[a]}>{ARITH_LABEL[a]}</Translated></button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi="Chế độ xem">View</Translated>
          <span className="seg" role="group" aria-label={tr('view', 'chế độ xem')}>
            {(['matrix', 'equations'] as const).map((v) => (
              <button key={v} aria-pressed={view === v} onClick={() => setView(v)}><Translated vi={{ matrix: 'ma trận', equations: 'phương trình' }[v]}>{v}</Translated></button>
            ))}
          </span>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setI(0)} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => setI(at - 1)} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => setI(at + 1)} disabled={at >= last}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => setI(last)} disabled={at >= last}><Translated vi="Chạy đến cuối">Run to the end</Translated></button>
        <span className="muted" aria-live="polite" data-at={at} data-total={last}><Translated vi={<>giai đoạn {at} trên {last}: {{ start: 'bắt đầu', pivot: 'tìm phần tử chốt', swap: 'hoán đổi', divide: 'chia', subtract: 'trừ', barrier: 'rào chắn', backsub: 'thế ngược', done: 'xong' }[ph.kind]}</>}>phase {at} of {last}: {ph.kind === 'backsub' ? 'back substitution' : ph.kind}</Translated></span>
      </div>
      <div className={'narrate' + (anyBad ? ' bad' : '')} role="status" style={{ marginBottom: '0.8rem' }} data-phase={ph.kind}>
        {ph.note}
        {ph.kind === 'pivot' ? (
          <>
            <br />
            <span className="muted">
              <Translated vi="Rút gọn max:">Max reduction:</Translated>{' '}
              {ph.pivot!.levels
                .map((lvl) => lvl.map((r) => `|${minus(ph.pivot!.abs[ph.pivot!.rows.indexOf(r)].text)}|`).join(' '))
                .join('  →  ')}
            </span>
          </>
        ) : null}
      </div>
      {ph.kind === 'barrier' ? (
        <div data-barrier style={{ background: 'var(--ink)', color: 'var(--page)', borderRadius: 3, padding: '0.25rem 0.7rem', marginBottom: '0.4rem', fontFamily: 'var(--sans)', fontSize: '0.82rem' }}>
          <Translated vi={<>rào chắn: cả {n} luồng chờ ở đây</>}>barrier: all {n} threads wait here</Translated>
        </div>
      ) : null}
      <div style={{ overflowX: 'auto' }}>
      <table className="threads" aria-label={tr('Threads and their rows', 'Các luồng và hàng của chúng')} data-view={view}>
        <thead>
          <tr>
            <th><Translated vi="luồng">thread</Translated></th>
            <th><Translated vi="trạng thái">state</Translated></th>
            <th><Translated vi="giai đoạn này">this phase</Translated></th>
            <th>{view === 'matrix' ? tr(`row: ${sys.vars.join(', ')}, right-hand side`, `hàng: ${sys.vars.join(', ')}, vế phải`) : tr('equation', 'phương trình')}</th>
          </tr>
        </thead>
        <tbody>
          {ph.m.map((row, k) => (
            <tr key={k} data-thread={k} className={ph.threads[k] === 'working' ? 'cur' : ''}>
              <td><Translated vi={<>{k} · hàng {k + 1}</>}>{k} · row {k + 1}</Translated></td>
              <td data-state={ph.threads[k]}>{STATE_LABEL[ph.threads[k]]}</td>
              <td>{opFor(ph, k)}</td>
              <td>
                {view === 'matrix' ? (
                  <span className="mx" style={{ gridTemplateColumns: `repeat(${n + 1}, auto)` }}>
                    {row.map((c, j) => (
                      <span key={j} className={bad(c.v) ? 'bad' : ''} data-cell={`${k},${j}`} style={j === n ? { borderLeftWidth: 3 } : undefined}>
                        {cellText(c.text)}
                      </span>
                    ))}
                  </span>
                ) : (
                  <span className={row.some((c) => bad(c.v)) ? 't-fail' : ''} data-eq={k}>{equation(row.map((c) => c.text), sys.vars)}</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>
      <div style={{ marginTop: '0.8rem', overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${tw + 76} ${n * lane + 8}`} style={{ width: (tw + 76) * 1.3, maxWidth: '100%', display: 'block' }} role="img" aria-label={tr('Timeline of thread states', 'Dòng thời gian trạng thái các luồng')}>
          {Array.from({ length: n }, (_, k) => (
            <text key={k} x={0} y={8 + k * lane + 14} fontSize="11"><Translated vi={<>luồng {k}</>}>thread {k}</Translated></text>
          ))}
          <Timeline phases={t.phases.slice(0, at + 1)} n={n} x={66} y={8} lane={lane} />
        </svg>
      </div>
      <p className="muted" style={{ margin: '0.5rem 0 0' }} data-busy={busySoFar ?? ''}>
        {busySoFar == null
          ? tr('No divide or subtract step has run yet.', 'Chưa có bước chia hay trừ nào chạy.')
          : tr(`Thread slots doing work in the divide and subtract steps so far: ${Math.round(busySoFar * 100)}%.`, `Tỉ lệ ô luồng có làm việc trong các bước chia và trừ cho đến giờ: ${Math.round(busySoFar * 100)}%.`)}
      </p>
      {done ? (
        <p style={{ margin: '0.6rem 0 0' }} data-solution={t.solution.map((c) => c.text).join(',')}>
          {close ? (
            <span className="chip ok">{relErr === 0 ? tr('exact solution', 'nghiệm chính xác') : tr(`close: largest error ${Number((relErr * 100).toPrecision(2))}%`, `gần đúng: sai số lớn nhất ${Number((relErr * 100).toPrecision(2))}%`)}</span>
          ) : (
            <span className="chip fail">{t.finite ? tr('wrong answer', 'đáp án sai') : tr('no answer: NaN or infinity', 'không có đáp án: NaN hoặc vô cực')}</span>
          )}{' '}
          <Translated
            vi={
              <>
                Tính được: {sys.vars.map((v, k) => `${v} = ${cellText(t.solution[k].text)}`).join(', ')}. Chính xác: {sys.vars.map((v, k) => `${v} = ${minus(exactSol[k])}`).join(', ')}.
              </>
            }
          >
            Computed: {sys.vars.map((v, k) => `${v} = ${cellText(t.solution[k].text)}`).join(', ')}. Exact: {sys.vars.map((v, k) => `${v} = ${minus(exactSol[k])}`).join(', ')}.
          </Translated>
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the erratum hunt
function ErratumHunt() {
  const computed = useMemo(() => bookPanels('A.11'), []);
  const diffs = useMemo(() => diffPanels(printedA11, computed), [computed]);
  const wrong = useMemo(
    () => new Set(diffs.map((d) => `${printedA11.findIndex((p) => p.label === d.panel)}:${d.what === 'caption' ? 'cap' : `${d.row}:${d.col}`}`)),
    [diffs],
  );
  const [marks, setMarks] = useState<Set<string>>(new Set());
  const [revealed, setRevealed] = useState(false);
  const [showComputed, setShowComputed] = useState(false);
  const toggle = (k: string) => {
    if (revealed) return;
    setMarks((m) => {
      const x = new Set(m);
      if (x.has(k)) x.delete(k);
      else x.add(k);
      return x;
    });
  };
  const found = [...marks].filter((k) => wrong.has(k)).length;
  const falseAlarms = [...marks].filter((k) => !wrong.has(k)).length;
  const panels = showComputed ? computed : printedA11;
  return (
    <Bench
      title={tr('Fig. A.11 as printed', 'Fig. A.11 như bản in')}
      task={tr('Click every caption or number you think the printed figure gets wrong, then check.', 'Nhấn vào mọi chú thích hay con số mà bạn nghĩ hình in sai, rồi kiểm tra.')}
      foot={tr('Transcribed from p. 598 (book notes); captions shortened to their content. Rows are numbered from 1 and blank positions drawn as dots, as in the book.', 'Chép lại từ p. 598 (ghi chú về sách); chú thích được rút gọn còn nội dung. Các hàng được đánh số từ 1 và vị trí trống vẽ thành dấu chấm, như trong sách.')}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button className="primary" onClick={() => setRevealed(true)} disabled={revealed}><Translated vi="Kiểm tra các đánh dấu của tôi">Check my marks</Translated></button>
        <button onClick={() => { setMarks(new Set()); setRevealed(false); setShowComputed(false); }}><Translated vi="Xóa">Clear</Translated></button>
        {revealed ? (
          <label className="check">
            <input type="checkbox" checked={showComputed} onChange={(e) => setShowComputed(e.target.checked)} /> <Translated vi="hiện hình được tính toán">show the computed figure</Translated>
          </label>
        ) : null}
        <span className="muted" data-marks={marks.size}><Translated vi={<>đã đánh dấu {marks.size}</>}>{marks.size} marked</Translated></span>
      </div>
      {revealed ? (
        <p data-result={`${found},${wrong.size},${falseAlarms}`} style={{ margin: '0 0 0.8rem' }}>
          <span className={found === wrong.size ? 'chip ok' : 'chip fail'}><Translated vi={<>tìm được {found} trên {wrong.size}</>}>found {found} of {wrong.size}</Translated></span>{' '}
          {falseAlarms ? <span className="chip fail"><Translated vi={<>{falseAlarms} chỗ đánh dấu nhưng đúng</>}>{falseAlarms} marked but right</Translated></span> : null}{' '}
          <Translated
            vi={
              <>
                Step 1 trong bản in chia hàng 1 cho 3, nhưng hàng 1 là [2 3 1 11] và kết quả được in là phép chia cho 2. Step 5 và 6
                in hàng 1 của Fig. A.10 thuộc hệ thứ nhất (1, 5/3, 2/3, 19/3) thay vì của hệ này (1, 3/2, 1/2, 11/2). Đáp án, (1, 2,
                3), là đúng.
              </>
            }
          >
            The printed Step 1 divides
            row 1 by 3, but row 1 is [2 3 1 11] and the result shown is a division by 2. Steps 5 and 6 print row 1 of the first
            system's Fig. A.10 (1, 5/3, 2/3, 19/3) instead of this system's (1, 3/2, 1/2, 11/2). The answer, (1, 2, 3), is right.
          </Translated>
        </p>
      ) : null}
      <div style={{ display: 'grid', gap: '0.8rem', gridTemplateColumns: 'repeat(auto-fill, minmax(13.5rem, 1fr))' }}>
        {panels.map((p, i) => {
          const capKey = `${i}:cap`; // i18n-ignore: data key
          const capCls = revealed ? (wrong.has(capKey) ? (marks.has(capKey) ? 'found' : 'missed') : marks.has(capKey) ? 'false' : '') : '';
          return (
            <div key={p.label} className="panel" data-panel={p.label}>
              <div className="mx-label">{tr(p.label, p.label === 'Original' ? 'Ban đầu' : p.label === 'Pivoting' ? 'Chọn phần tử chốt' : p.label.replace(/^Step /, 'Bước '))}{p.solved ? <span className="muted"><Translated vi={<> · giải ra {p.solved}</>}> · solves {p.solved}</Translated></span> : null}</div>
              {p.caption.op !== 'none' ? (
                <button className="quiet" aria-pressed={marks.has(capKey)} data-key={capKey} data-verdict={capCls}
                  onClick={() => toggle(capKey)}
                  style={{ fontSize: '0.8rem', marginBottom: '0.45rem', textAlign: 'left', ...(capCls === 'found' || capCls === 'missed' ? { borderColor: 'var(--fail)', color: capCls === 'missed' ? 'var(--fail)' : undefined } : {}) }}>
                  {captionText(p.caption)}
                </button>
              ) : (
                <div className="muted" style={{ fontSize: '0.8rem', marginBottom: '0.45rem' }}><Translated vi="hệ phương trình">the system</Translated></div>
              )}
              <span className="mx" style={{ gridTemplateColumns: 'repeat(4, auto)', display: 'inline-grid' }}>
                {p.rows.flatMap((row, r) =>
                  row.map((v, c) => {
                    const k = `${i}:${r}:${c}`;
                    const isWrong = wrong.has(k);
                    const marked = marks.has(k);
                    const cls = revealed ? (isWrong ? (showComputed ? 'good' : 'bad') : marked ? 'void' : '') : '';
                    return (
                      <span key={k} role="button" tabIndex={0} data-key={k} data-marked={marked ? 1 : 0} className={cls}
                        onClick={() => toggle(k)} onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && toggle(k)}
                        style={{ cursor: revealed ? 'default' : 'pointer', ...(marked ? { outline: '2px solid var(--ink)', outlineOffset: -2 } : {}) }}>
                        {cellText(v)}
                      </span>
                    );
                  }),
                )}
              </span>
            </div>
          );
        })}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSA7: Record<string, Question> = {
  'appA/A.7/q-idle': {
    kind: 'numeric',
    id: 'appA/A.7/q-idle',
    prompt: (
      <Translated
        vi={
          <>
            8 phương trình, một luồng mỗi hàng, theo thiết kế của sách. Qua các bước chia và trừ của phép khử, bao nhiêu phần
            trăm ô luồng làm việc có ích?
          </>
        }
      >
        8 equations, one thread per row, the book's design. Over the divide and subtract steps of the elimination, what
        percentage of thread slots does useful work?
      </Translated>
    ),
    answer: 53.33,
    tol: 0.01,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            Bước chia k giữ n − k luồng làm việc và bước trừ k giữ n − k − 1, nên n + (n − 1) = 15 bước, mỗi bước 8 ô, chứa 8 + 7
            + 7 + 6 + ... + 1 = 64 = n² ô bận trên tổng 120: 53.3%. Tổng quát, tỉ lệ là n / (2n − 1), giảm dần về một nửa khi n
            tăng: 1,024 phương trình cho 1,024 / 2,047. Mọi thuật toán có khối lượng việc co lại trong sách này đều có dạng đó, và
            đó là lý do cây rút gọn của chương 10 được gộp việc cho luồng (thread coarsening).
          </p>
        }
      >
        <p>
          Divide step k keeps n − k threads working and subtract step k keeps n − k − 1, so the n + (n − 1) = 15 steps of 8
          slots hold 8 + 7 + 7 + 6 + ... + 1 = 64 = n² busy slots out of 120: 53.3%. In general the share is n / (2n − 1),
          which falls toward a half as n grows: 1,024 equations give 1,024 / 2,047. Every shrinking-work algorithm in this
          book has this shape, and it is why chapter 10's reduction tree gets coarsened.
        </p>
      </Translated>
    ),
    source: 'book §A.7, p. 597 (the design); our count',
    vi: { source: 'sách §A.7, p. 597 (thiết kế); phép đếm của chúng tôi' },
  },
  'appA/A.7/q-tiny': {
    kind: 'choice',
    id: 'appA/A.7/q-tiny',
    prompt: (
      <Translated
        vi={
          <>
            FP32, không chọn phần tử chốt, hệ [[ε, 1], [1, 1]] x = [1, 2] với ε = 10<sup>−8</sup>. Giá trị x1 thật vào khoảng 1.
            Phép khử cho x1 bằng bao nhiêu?
          </>
        }
      >
        FP32, no pivoting, the system [[ε, 1], [1, 1]] x = [1, 2] with ε = 10<sup>−8</sup>. The true x1 is about 1. What
        does elimination give for x1?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="0, vì 1 − 10⁸ và 2 − 10⁸ làm tròn thành cùng một số.">
            0, because 1 − 10⁸ and 2 − 10⁸ round to the same number.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Khoảng 1, đúng tới khoảng bảy chữ số, như FP32 hứa hẹn.">
            About 1, correct to about seven digits, as FP32 promises.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi phép toán chính xác tới nửa ULP; chính thuật toán khuếch đại các sai số.">
            Each operation is accurate to half an ULP; the algorithm amplifies the errors.
          </Translated>
        ),
      },
      {
        label: <Translated vi="NaN, vì ε làm tròn thành 0 trong FP32.">NaN, because ε rounds to 0 in FP32.</Translated>,
        why: (
          <Translated vi="10⁻⁸ lớn hơn rất nhiều so với số normal nhỏ nhất của FP32, khoảng 1.2 × 10⁻³⁸. Trong FP16 nó mới làm tròn thành 0.">
            10⁻⁸ is far above FP32's smallest normal number, about 1.2 × 10⁻³⁸. In FP16 it would round to 0.
          </Translated>
        ),
      },
      {
        label: <Translated vi="10⁸, nghịch đảo của phần tử chốt rất nhỏ.">10⁸, the reciprocal of the tiny pivot.</Translated>,
        why: (
          <Translated vi="Đó là hệ số sau phép chia, không phải nghiệm.">
            That is the coefficient after the division, not the solution.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hàng 1 chia cho ε chứa 10<sup>8</sup> hai lần. Trừ nó khỏi hàng 2 cho 1 − 10<sup>8</sup> và 2 − 10<sup>8</sup>, và
            gần 10<sup>8</sup> các giá trị FP32 cách nhau 8, nên cả hai làm tròn thành −10<sup>8</sup>: x2 = 1 chính xác. Phép thế
            ngược sau đó tính x1 = 10<sup>8</sup> − 10<sup>8</sup> · 1 = 0. Hoán đổi các hàng trước thì cùng các bước đó cho x1 =
            1.
          </p>
        }
      >
        <p>
          Row 1 divided by ε holds 10<sup>8</sup> twice. Subtracting it from row 2 gives 1 − 10<sup>8</sup> and 2 −
          10<sup>8</sup>, and near 10<sup>8</sup> FP32 values are 8 apart, so both round to −10<sup>8</sup>: x2 = 1 exactly.
          Back substitution then computes x1 = 10<sup>8</sup> − 10<sup>8</sup> · 1 = 0. Swap the rows first and the same
          steps give x1 = 1.
        </p>
      </Translated>
    ),
    source: 'our example of the book\'s "subtler case", §A.7, p. 597; checked by src/sim/gauss.test.ts',
    vi: { source: 'ví dụ của chúng tôi cho "trường hợp tinh vi hơn" của sách, §A.7, p. 597; kiểm tra bởi src/sim/gauss.test.ts' },
  },
  'appA/A.7/q-a11': {
    kind: 'choice',
    id: 'appA/A.7/q-a11',
    prompt: (
      <Translated
        vi={
          <>
            Chú thích Step 1 của Fig. A.11 nói chia hàng 1 cho 3. Sau khi hoán đổi, hàng 1 là [2 3 1 11], và hình cho thấy kết
            quả [1 3/2 1/2 11/2]. Số chia lẽ ra phải là bao nhiêu?
          </>
        }
      >
        Fig. A.11's Step 1 caption says to divide row 1 by 3. After the swap, row 1 is [2 3 1 11], and the figure shows
        the result [1 3/2 1/2 11/2]. What should the divisor be?
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="3, hệ số của Y ở hàng 1.">3, row 1's coefficient of Y.</Translated>,
        why: (
          <Translated vi="Step 1 khử X, nên số chia là hệ số của X.">
            Step 1 eliminates X, so the divisor is the coefficient of X.
          </Translated>
        ),
      },
      { label: <Translated vi="2, hệ số của X ở hàng 1.">2, row 1's coefficient of X.</Translated>, correct: true },
      {
        label: <Translated vi="5, hệ số lớn nhất còn lại trong ma trận.">5, the largest coefficient left in the matrix.</Translated>,
        why: (
          <Translated vi="Việc chọn phần tử chốt chọn ra hàng; phép chia dùng hệ số của chính hàng đó.">
            Pivoting chose the row; the division uses that row's own coefficient.
          </Translated>
        ),
      },
      {
        label: <Translated vi="1, vì phép hoán đổi đã chuẩn hóa hàng rồi.">1, since the swap already normalised the row.</Translated>,
        why: (
          <Translated vi="Hoán đổi chỉ di chuyển một hàng; nó không co giãn hàng đó.">
            A swap moves a row; it does not scale it.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Chia [2 3 1 11] cho 2 được [1 3/2 1/2 11/2], đúng kết quả được in; chia cho 3 sẽ được [2/3 1 1/3 11/3]. Chú thích là
            một lỗi sơ suất; chúng tôi đoán nó bị mang sang từ Fig. A.10, nơi hàng đầu tiên đúng là chia cho 3.
          </p>
        }
      >
        <p>
          Dividing [2 3 1 11] by 2 gives [1 3/2 1/2 11/2], exactly the printed result; dividing by 3 would give [2/3 1 1/3
          11/3]. The caption is a slip; our guess is that it was carried over from Fig. A.10, whose first row does divide by
          3.
        </p>
      </Translated>
    ),
    source: 'book §A.7, p. 598, Fig. A.11; notes/appA.md "Unclear in the PDF"',
    vi: { source: 'sách §A.7, p. 598, Fig. A.11; notes/appA.md "Unclear in the PDF"' },
  },
  'appA/A.7/q-pivot-cost': {
    kind: 'choice',
    id: 'appA/A.7/q-pivot-cost',
    prompt: (
      <Translated vi={<>Vì sao chọn phần tử chốt rẻ bên trong một khối mà đắt khi trải qua nhiều khối?</>}>
        <>Why is pivoting cheap inside one block and expensive across many blocks?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mỗi cột cần một giá trị max trên mọi hàng mà mọi luồng đều thấy: một rào chắn trong một khối, một lần đồng bộ hóa cả lưới khi qua nhiều khối.">
            Each column needs a max over all rows that every thread sees: a barrier in a block, a grid sync across blocks.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Hoán đổi hai hàng trong bộ nhớ toàn cục tốn thời gian hơn nhiều so với hoán đổi chúng trong bộ nhớ chia sẻ.">
            Swapping two rows in global memory takes far longer than swapping them in shared memory.
          </Translated>
        ),
        why: (
          <Translated vi="Di chuyển một hàng thì rẻ; cái giá nằm ở chỗ mọi luồng phải thống nhất là hàng nào, mỗi cột một lần.">
            Moving one row is cheap; the cost is that everyone must agree on which row, once per column.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các khối luồng không tự chạy được phép rút gọn max, nên việc tìm kiếm phải chuyển về host.">
            Thread blocks cannot run a max reduction on their own, so the search moves to the host.
          </Translated>
        ),
        why: (
          <Translated vi="Chúng làm được (chương 10); vấn đề là phải gộp đáp án của các khối trước khi ai đó được phép chia.">
            They can (chapter 10); the problem is combining the blocks' answers before anyone may divide.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chọn phần tử chốt làm số phép tính tăng gần gấp đôi, và nhiều khối làm cái giá đó lộ rõ.">
            Pivoting roughly doubles the arithmetic, and many blocks make that cost visible.
          </Translated>
        ),
        why: (
          <Translated vi="Một lần tìm kiếm thêm khoảng n phép so sánh mỗi cột, nhỏ so với chính phép khử.">
            A search adds about n comparisons per column, small next to the elimination itself.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bên trong một khối, việc tìm kiếm là một phép rút gọn trong bộ nhớ chia sẻ theo sau là một rào chắn. Các khối không
            thể chờ nhau trong một lần khởi chạy thông thường, nên khi qua nhiều khối, mỗi cột cần một lần đồng bộ hóa toàn lưới
            (một cooperative launch) hoặc một ranh giới kernel: mỗi cột một lần khởi chạy. Đó chính là việc truyền thông mà tài
            liệu tham khảo [6] của sách muốn tránh.
          </p>
        }
      >
        <p>
          Inside a block the search is a shared-memory reduction followed by a barrier. Blocks cannot wait for each other in
          an ordinary launch, so across blocks every column needs a grid-wide synchronisation (a cooperative launch) or a
          kernel boundary: a launch per column. That is the communication the book's reference [6] is about avoiding.
        </p>
      </Translated>
    ),
    source: 'book §A.7, pp. 598–599; chapter 10 §10.9 and notes/today/ch10-facts.md A8',
    vi: { source: 'sách §A.7, pp. 598–599; chương 10 §10.9 và notes/today/ch10-facts.md A8' },
  },
  'appA/A.7/q-partial': {
    kind: 'choice',
    id: 'appA/A.7/q-partial',
    prompt: (
      <Translated vi={<>getrf của cuSOLVER ghi trong tài liệu là "LU with partial pivoting". Nó làm gì trước khi khử mỗi cột?</>}>
        <>cuSOLVER's getrf documents "LU with partial pivoting". What does it do before eliminating each column?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Hoán đổi hàng sao cho phần tử chốt là phần tử có độ lớn lớn nhất của cột đó trong các hàng còn lại.">
            Swaps rows so that the pivot is the largest-magnitude entry of that column among the remaining rows.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Hoán đổi cả hàng và cột để đưa phần tử lớn nhất của ma trận con còn lại về vị trí chốt.">
            Swaps rows and columns to bring the largest entry of the remaining submatrix to the pivot.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là chọn phần tử chốt toàn phần, A = P L U Q (LAPACK getc2).">
            That is complete pivoting, A = P L U Q (LAPACK getc2).
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Tìm phần tử lớn nhất trong một nhóm cục bộ các hàng gần nhau, để ít dữ liệu phải trao đổi hơn.">
            Searches a localised group of nearby rows for the largest entry, so less data has to be exchanged.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là mô tả của sách về chọn phần tử chốt từng phần, không khớp với thuật ngữ của thư viện nào.">
            That is the book's description of partial pivoting, which matches neither library term.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không làm gì, trừ khi phần tử chốt đúng bằng không; khi đó nó hoán đổi với hàng kế tiếp.">
            Nothing, unless the pivot is exactly zero; then it swaps with the next row.
          </Translated>
        ),
        why: (
          <Translated vi="Nó tìm kiếm ở mọi cột; một phần tử chốt khác không nhưng rất nhỏ cũng là trường hợp nguy hiểm.">
            It searches every column; a tiny non-zero pivot is the dangerous case too.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong LAPACK, cuSOLVER, rocSOLVER và Metal Performance Shaders, chọn phần tử chốt từng phần (partial pivoting) nghĩa là
            hoán đổi hàng để chọn phần tử chốt trong cột hiện tại, cho ra A = P L U. Hoán đổi cả cột là chọn phần tử chốt toàn
            phần (complete pivoting). cuSOLVER cũng nói thẳng rằng chế độ không chọn phần tử chốt của nó "is not numerically
            stable" (không ổn định về số học).
          </p>
        }
      >
        <p>
          In LAPACK, cuSOLVER, rocSOLVER and Metal Performance Shaders, partial pivoting means row interchanges that choose
          the pivot within the current column, giving A = P L U. Swapping columns as well is complete pivoting. cuSOLVER
          also says plainly that its no-pivoting mode "is not numerically stable".
        </p>
      </Translated>
    ),
    source: 'notes/today/appA-facts.md A15 (LAPACK dgetrf and dgetc2, cuSOLVER, rocSOLVER, MPS)',
    vi: { source: 'notes/today/appA-facts.md A15 (LAPACK dgetrf và dgetc2, cuSOLVER, rocSOLVER, MPS)' },
  },
};

// ---------------------------------------------------------------- the section
export function SA7() {
  return (
    <>
      <Translated
        vi={
          <p>
            Trang này ở dạng cô đọng. §A.7 của sách giải các hệ tuyến tính nhỏ bằng phép khử Gauss, phác thảo một phiên bản song
            song, và cho thấy một số chia bằng không hoặc rất nhỏ làm hỏng nó thế nào. Ba điều chuyển được sang kernel suy luận:
            thiết kế mỗi luồng một hàng được giữ với nhau bằng một rào chắn sau mỗi bước, cái giá của việc các luồng rảnh dần
            khi khối lượng việc co lại, và thói quen tìm giá trị lớn nhất trước khi chia cho bất cứ thứ gì. Không có lab: với
            một bộ giải thật bạn gọi một thư viện, và cuối trang sẽ nói là thư viện nào.
          </p>
        }
      >
        <p>
          This page is condensed. The book's §A.7 solves small linear systems by Gaussian elimination, sketches a parallel
          version, and shows how a zero or tiny divisor breaks it. Three things transfer to inference kernels: a
          row-per-thread design held together by a barrier after every step, the cost of threads going idle as the work
          shrinks, and the habit of finding the largest value before dividing by anything. There is no lab: for a real
          solver you call a library, and the end of the page says which.
        </p>
      </Translated>
      <Source section="A.7" pages="594–599" />

      <h2><Translated vi="Ổn định và không ổn định">Stable and unstable</Translated></h2>
      <Translated
        vi={
          <p>
            Với một tổng, thứ tự các phép cộng làm thay đổi các bit thấp (§A.6). Với một bộ giải tuyến tính, nó có thể quyết định
            bạn có được đáp án hay không: một số đầu vào cần các phép toán theo một thứ tự khác thì nghiệm mới xuất hiện. Sách gọi
            một thuật toán là ổn định về số học nếu nó tìm được một thứ tự phù hợp, và do đó một nghiệm, cho mọi đầu vào có
            nghiệm, và không ổn định nếu ngược lại. Các yêu cầu về tính ổn định là một lý do khiến khó tìm được các bộ giải song
            song hiệu quả.
          </p>
        }
      >
        <p>
          For a sum, the order of the additions changes the low bits (§A.6). For a linear solver it can decide whether you
          get an answer at all: some inputs need the operations in a different order before a solution appears. The book
          calls an algorithm numerically stable if it can find a suitable order, and so a solution, for every input that has
          one, and unstable otherwise. Stability requirements are one reason efficient parallel solvers are hard to find.
        </p>
      </Translated>
      <Source section="A.7" pages="594" />

      <h2><Translated vi="Phép khử, mỗi luồng một hàng">Elimination, one row per thread</Translated></h2>
      <Translated
        vi={
          <p>
            Ví dụ của sách là ba phương trình, 3X + 5Y + 2Z = 19, 2X + 3Y + Z = 11 và X + 2Y + 2Z = 11: ba mặt phẳng cắt nhau tại
            một điểm. Chia mỗi phương trình cho hệ số X của nó, trừ phương trình thứ nhất khỏi hai phương trình kia, và X biến mất
            khỏi cả hai. Lặp lại cho Y trên hai phương trình cuối, và phương trình thứ ba thành 3Z = 9. Phép thế ngược sau đó chạy
            từ dưới lên: thay Z vào phương trình thứ hai được Y, thay Y và Z vào phương trình thứ nhất được X. Viết dưới dạng các
            hàng của một ma trận mở rộng, với vế phải là cột cuối cùng, mỗi phép toán trên một phương trình là một phép toán trên
            một hàng.
          </p>
        }
      >
        <p>
          The book's example is three equations, 3X + 5Y + 2Z = 19, 2X + 3Y + Z = 11 and X + 2Y + 2Z = 11: three planes that
          meet at one point. Divide each equation by its coefficient of X, subtract the first from the other two, and X is
          gone from both. Repeat for Y on the last two, and the third reads 3Z = 9. Back substitution then runs upward: Z
          into the second equation gives Y, Y and Z into the first give X. Written as the rows of an augmented matrix, with
          the right-hand side as a last column, each operation on an equation is an operation on a row.
        </p>
      </Translated>
      <Source section="A.7" pages="594–596" figs={tr('Figs. A.9 and A.10', 'Figs. A.9 và A.10')} />
      <Translated
        vi={
          <p>
            Thiết kế song song giao mỗi hàng cho một luồng, luồng đó làm toàn bộ việc trên hàng, và giữ ma trận trong bộ nhớ chia
            sẻ (shared memory) của một khối khi vừa. Các luồng cùng nhau đi qua từng cột. Sau mỗi phép chia là một rào chắn
            (barrier), vì phép trừ đọc hàng trên cùng, thứ mà một luồng khác vừa ghi. Sau mỗi phép trừ là một rào chắn nữa, để
            phép chia kế tiếp thấy được kết quả, và thêm một luồng không còn gì để làm cho đến phép thế ngược. Sách chỉ mô tả
            điều này bằng lời; nó không đưa ra kernel nào.
          </p>
        }
      >
        <p>
          The parallel design gives each row to one thread, which does all the work on it, and keeps the matrix in one
          block's shared memory when it fits. The threads step through the columns together. After each division comes a
          barrier, because the subtraction reads the top row, which another thread has just written. After each subtraction
          comes another, so that the next division sees the result, and one more thread is left with nothing to do until
          back substitution. The book describes this in prose only; it gives no kernel.
        </p>
      </Translated>
      <Source section="A.7" pages="597" />
      <GaussScene />
      <Translated
        vi={
          <p>
            Giờ bạn tự đi từng bước. Bắt đầu với hệ thứ nhất của sách ở chế độ xem ma trận, và so mỗi giai đoạn với Fig. A.10;
            chuyển sang phương trình để so với Fig. A.9. Sau đó thử FP32 và FP16 trên cùng hệ đó và xem đáp án lệch dần đi một
            chút so với (1, 2, 3).
          </p>
        }
      >
        <p>
          Now step through it yourself. Start with the book's first system in the matrix view, and compare each phase with
          Fig. A.10; switch to equations for Fig. A.9. Then try FP32 and FP16 on the same system and watch the answer drift a
          little from (1, 2, 3).
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một khối, một luồng mỗi hàng, không có thời gian và không có warp; các hàng vừa trong bộ nhớ chia sẻ, như sách giả
              định. Việc tìm phần tử chốt ở đây là một giai đoạn; trên GPU nó là một phép rút gọn gồm log2 n vòng với các rào
              chắn. Một phép nhân và một phép trừ làm tròn riêng rẽ (không có fused multiply-add).
            </p>
          }
        >
          <p>
            One block, one thread per row, no timing and no warps; the rows fit in shared memory, as the book assumes. A
            pivot search is one phase here; on a GPU it is a reduction of log2 n rounds with barriers. A multiply and a
            subtract round separately (no fused multiply-add).
          </p>
        </Translated>
      </Simplification>
      <GaussStepper
        title={tr('Elimination, one phase at a time', 'Phép khử, từng giai đoạn một')}
        task={tr('Step one phase at a time. Where do the barriers fall, and which threads have nothing to do?', 'Đi từng giai đoạn một. Các rào chắn rơi vào đâu, và luồng nào không có gì để làm?')}
        init={{ sys: 'book1', pivot: false, arith: 'exact', eps: 8 }}
      />
      <Translated
        vi={
          <p>
            Đếm các ô trên dòng thời gian. Với ba hàng, 9 trong 15 ô luồng ở các bước chia và trừ có làm việc. Đếm xem mỗi bước
            giữ bao nhiêu hàng bận, rồi tính xem tỉ lệ đó trở thành bao nhiêu trên một hệ lớn.
          </p>
        }
      >
        <p>
          Count the cells in the timeline. On three rows, 9 of the 15 thread slots in the divide and subtract steps do work.
          Count how many rows each step keeps busy, then work out what that share becomes on a large system.
        </p>
      </Translated>
      <Ask q={questionsSA7['appA/A.7/q-idle']} />

      <h2><Translated vi="Khi số chia bằng không, hoặc rất nhỏ">When the divisor is zero, or tiny</Translated></h2>
      <Translated
        vi={
          <p>
            Thay phương trình thứ nhất bằng 5Y + 2Z = 16. Hệ vẫn có nghiệm (1, 2, 3), nhưng bước 1 yêu cầu luồng 0 chia cho hệ số
            X của nó, giờ là 0. Trong số học IEEE không có gì dừng lại (§A.3): 5 / 0 là vô cực, 0 / 0 là NaN, và ở phép trừ kế
            tiếp mọi hàng đều nhận một NaN. Thuật toán đơn giản này không ổn định.
          </p>
        }
      >
        <p>
          Replace the first equation by 5Y + 2Z = 16. The system still has the solution (1, 2, 3), but step 1 asks thread 0
          to divide by its coefficient of X, which is now 0. In IEEE arithmetic nothing stops (§A.3): 5 / 0 is infinity, 0 /
          0 is NaN, and at the next subtraction every row picks up a NaN. The simple algorithm is unstable.
        </p>
      </Translated>
      <Source section="A.7" pages="597" />
      <Translated
        vi={
          <p>
            Trường hợp tinh vi hơn là một hệ số dẫn đầu không bằng không nhưng nhỏ, và đã mang sẵn sai số làm tròn từ các bước
            trước. Chia cho nó sẽ phóng đại sai số đó, và mọi bước sau đều thừa hưởng nó, vì thế tính không ổn định thường được đo
            bằng mức tăng của sai số từ bước này sang bước khác.
          </p>
        }
      >
        <p>
          The subtler case is a lead coefficient that is not zero but small, and already carries rounding error from earlier
          steps. Dividing by it magnifies that error, and every later step inherits it, which is why instability is often
          measured as the growth of error from step to step.
        </p>
      </Translated>
      <Source section="A.7" pages="597" />
      <Translated
        vi={
          <p>
            Một ví dụ hai phương trình làm điều này cụ thể: [[ε, 1], [1, 1]] x = [1, 2], có nghiệm x1 = 1 / (1 − ε) xấp xỉ 1 với
            ε nhỏ. Trong FP32 với ε = 10<sup>−8</sup> và không chọn phần tử chốt, hàng thứ nhất chia cho ε chứa 10<sup>8</sup>, và
            trừ nó khỏi hàng thứ hai còn lại 1 − 10<sup>8</sup> và 2 − 10<sup>8</sup>. Gần 10<sup>8</sup>, các giá trị FP32 cách
            nhau 8, nên cả hai làm tròn thành cùng một số, và thông tin của phương trình thứ hai mất đi.
          </p>
        }
      >
        <p>
          A two-equation example makes it concrete: [[ε, 1], [1, 1]] x = [1, 2], whose solution x1 = 1 / (1 − ε) is about 1
          for small ε. In FP32 with ε = 10<sup>−8</sup> and no pivoting, the first row divided by ε holds 10<sup>8</sup>, and
          subtracting it from the second leaves 1 − 10<sup>8</sup> and 2 − 10<sup>8</sup>. Near 10<sup>8</sup>, FP32 values
          are 8 apart, so both round to the same number, and the second equation's information is gone.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Hệ 2 × 2 là của khóa học này: một minh họa tiêu chuẩn cho "trường hợp tinh vi hơn" của sách.</p>}>
          <p>The 2 × 2 system is ours: a standard illustration of the book's "subtler case".</p>
        </Translated>
      </Beyond>
      <GaussStepper
        title={tr('Pivot or fail', 'Chọn phần tử chốt hoặc thất bại')}
        task={tr('Run it to the end, then change one thing at a time: the arithmetic, ε, pivoting.', 'Chạy đến cuối, rồi mỗi lần đổi một thứ: số học, ε, việc chọn phần tử chốt.')}
        init={{ sys: 'tiny', pivot: false, arith: 'fp32', eps: 8 }}
      />
      <Translated
        vi={
          <p>
            Với phân số chính xác, cùng các bước đó cho x1 = 100,000,000 / 99,999,999: thuật toán là chính xác, và chính việc
            làm tròn gây ra thiệt hại. Trong FP16 rắc rối bắt đầu sớm hơn nhiều, ở ε = 10<sup>−4</sup>; từ 10<sup>−5</sup> phép
            chia cho ε tràn thành vô cực, và từ 10<sup>−8</sup> trở xuống bản thân ε làm tròn thành 0, biến phần tử chốt rất nhỏ
            thành phần tử chốt bằng không ở trên. Sau đó chọn hệ thứ hai của sách không chọn phần tử chốt và đi đến phép trừ đầu
            tiên.
          </p>
        }
      >
        <p>
          With exact fractions the same steps give x1 = 100,000,000 / 99,999,999: the algorithm is exact, and the rounding
          does the damage. In FP16 the trouble starts much earlier, at ε = 10<sup>−4</sup>; from 10<sup>−5</sup> the division
          by ε overflows to infinity, and from 10<sup>−8</sup> down ε itself rounds to 0, which turns the tiny pivot into
          the zero pivot above. Then select the book's second system
          without pivoting and step to the first subtraction.
        </p>
      </Translated>
      <Ask q={questionsSA7['appA/A.7/q-tiny']} />

      <h2><Translated vi="Chọn phần tử chốt">Pivoting</Translated></h2>
      <Translated
        vi={
          <p>
            Cách sửa của sách: trước khi chia, chọn một phương trình còn lại có hệ số của biến hiện tại khác không, tốt nhất là
            lớn nhất về độ lớn, và hoán đổi nó lên trên cùng. Với hệ thứ hai, 2X + 3Y + Z = 11 lên trên cùng, và phương trình
            không có X thành phương trình thứ hai; luồng của nó bỏ qua cả phép chia lẫn phép trừ cho X, vì không có gì để khử. Sự
            bỏ qua đó là một nhánh mà vài luồng đi còn những luồng khác thì không: phân kỳ luồng (divergence) bên trong một warp
            (chương 4).
          </p>
        }
      >
        <p>
          The book's fix: before dividing, pick a remaining equation whose coefficient of the current variable is not zero,
          preferably the largest in magnitude, and swap it to the top. For the second system, 2X + 3Y + Z = 11 moves to the
          top, and the equation with no X becomes the second; its thread skips both the division and the subtraction for X,
          since there is nothing to eliminate. That skip is a branch some threads take and others do not: divergence inside a
          warp (chapter 4).
        </p>
      </Translated>
      <Source section="A.7" pages="597–598" figs="Fig. A.11" />
      <Translated
        vi={
          <p>
            Đánh dấu "chọn phần tử chốt" ở một trong hai trình chạy từng bước và chạy lại hệ thứ hai và hệ 2 × 2. Một tổ hợp vẫn
            thất bại: FP16 với ε từ 10<sup>−5</sup> đến 10<sup>−7</sup>, có chọn phần tử chốt.
          </p>
        }
      >
        <p>
          Tick "pivoting" in either stepper and run the second system and the 2 × 2 system again. One combination still
          fails: FP16 with ε from 10<sup>−5</sup> to 10<sup>−7</sup>, pivoting on.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Của khóa học này: thiết kế của sách chia mỗi hàng cho hệ số dẫn đầu của chính nó, kể cả hàng chứa ε, và khi đó 1 / ε
              vượt quá giá trị lớn nhất của FP16, 65,504. LU với chọn phần tử chốt từng phần thì chia phần tử của mỗi hàng cho phần
              tử chốt, nên mọi hệ số nhân có độ lớn không quá 1.
            </p>
          }
        >
          <p>
            Ours: the book's design divides every row by its own lead coefficient, including the row holding ε, and 1 / ε
            then exceeds FP16's largest value, 65,504. LU with partial pivoting divides each row's entry by the pivot instead,
            so every multiplier is at most 1 in magnitude.
          </p>
        </Translated>
      </Beyond>

      <h3><Translated vi="Fig. A.11, như bản in">Fig. A.11, as printed</Translated></h3>
      <Translated
        vi={
          <p>
            Sách vẽ lần chạy có chọn phần tử chốt của hệ thứ hai thành chín ô hình. Hai chỗ trong hình in là sai, một trong số đó
            sai hai lần. Hãy tìm chúng trước khi đọc tiếp: đánh dấu mọi chú thích hay con số bạn tin là sai, rồi kiểm tra. Trình
            chạy từng bước ở trên, đặt về hệ thứ hai có chọn phần tử chốt và phân số chính xác, tạo ra các ô hình đúng.
          </p>
        }
      >
        <p>
          The book draws the pivoting run of the second system as nine panels. Two things in the printed figure are wrong,
          one of them twice. Find them before reading on: mark every caption or number you believe is wrong, then check. The
          stepper above, set to the second system with pivoting and exact fractions, produces the right panels.
        </p>
      </Translated>
      <ErratumHunt />
      <Ask q={questionsSA7['appA/A.7/q-a11']} />

      <h2><Translated vi="Chọn phần tử chốt tốn gì khi chạy song song">What pivoting costs in parallel</Translated></h2>
      <Translated
        vi={
          <p>
            Chọn phần tử chốt mở đầu mỗi cột bằng một câu hỏi về mọi hàng còn lại: hàng nào giữ hệ số lớn nhất? Với một luồng mỗi
            hàng, đó là một phép rút gọn max giữa các luồng, cây của chương 10, theo sau là một phép hoán đổi mà mọi luồng phải
            thấy. Bên trong một khối thì chi phí đó chấp nhận được: một phép rút gọn trong bộ nhớ chia sẻ và một rào chắn mỗi cột,
            cùng chút cẩn thận để giữ phân kỳ trong các warp ở mức thấp.
          </p>
        }
      >
        <p>
          Pivoting starts every column with a question about all the remaining rows: which one holds the largest
          coefficient? With one thread per row that is a max reduction across threads, chapter 10's tree, followed by a swap
          that every thread must see. Inside one block that is affordable: a reduction in shared memory and a barrier per
          column, with some care to keep divergence within warps down.
        </p>
      </Translated>
      <Source section="A.7" pages="598–599" />
      <Translated
        vi={
          <p>
            Qua nhiều khối thì không. Các khối không thể chờ nhau trong một lần khởi chạy thông thường (
            <a href={href('ch10', '10.9')}>§10.9</a>), nên mỗi cột cần một lần đồng bộ hóa toàn lưới, chỉ có được với một
            cooperative launch, hoặc một ranh giới kernel: mỗi cột một lần khởi chạy. Qua các nút của một cụm máy thì còn tệ hơn
            nữa. Sách chỉ đến các thuật toán tránh truyền thông, vốn tái cấu trúc công việc để trao đổi ít hơn, và đến ngẫu nhiên
            hóa, thứ mà các nhà nghiên cứu nhận thấy có xu hướng giữ độ chính xác cao. Sách chỉ nêu tên cả hai, và trang này cũng
            vậy.
          </p>
        }
      >
        <p>
          Across blocks it is not. Blocks cannot wait for each other inside an ordinary launch (
          <a href={href('ch10', '10.9')}>§10.9</a>), so each column needs a grid-wide synchronisation, available only under a
          cooperative launch, or a kernel boundary: one launch per column. Across the nodes of a cluster it is worse again.
          The book points to communication-avoiding algorithms, which restructure the work to exchange less, and to
          randomisation, which researchers have found tends to keep accuracy high. It names both, and so does this page.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch10-facts.md A8 and its backend table)', '2026-11-20 (notes/today/ch10-facts.md A8 và bảng backend của nó)')}>
        <Translated
          vi={
            <p>
              CUDA và HIP có <code>grid.sync()</code> dưới một cooperative launch mà lưới vừa trên thiết bị cùng một lúc. Metal
              không có rào chắn toàn lưới: hãy dùng một lần dispatch thứ hai.
            </p>
          }
        >
          <p>
            CUDA and HIP have <code>grid.sync()</code> under a cooperative launch whose grid fits on the device at once.
            Metal has no grid-wide barrier: use a second dispatch.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsSA7['appA/A.7/q-pivot-cost']} />

      <h2><Translated vi="Các thư viện gọi nó là gì">What the libraries call it</Translated></h2>
      <Translated
        vi={
          <p>
            Sách định nghĩa chọn phần tử chốt từng phần là giới hạn các ứng viên trong một tập phương trình cục bộ, với cái giá
            nhỏ về độ chính xác, và quy tắc tổng quát của sách hoán đổi cả cột lẫn hàng. Các thư viện dùng từ ngữ khác. Ở đó,
            chọn phần tử chốt từng phần (partial pivoting) nghĩa là hoán đổi hàng để chọn phần tử chốt trong cột hiện tại, phép
            phân tích A = P L U; hoán đổi cả hàng và cột là chọn phần tử chốt toàn phần (complete pivoting), A = P L U Q. Tập cục
            bộ của sách không khớp với cái nào. Theo cách chúng tôi đọc, nó mô tả một biến thể tránh truyền thông, chủ đề của tài
            liệu tham khảo [6]; chúng tôi chưa kiểm tra điều đó.
          </p>
        }
      >
        <p>
          The book defines partial pivoting as restricting the candidates to a localised set of equations, at a slight cost
          in accuracy, and its general rule swaps columns as well as rows. The libraries use the words differently. Partial
          pivoting there means row interchanges that choose the pivot within the current column, the factorisation A = P L
          U; swapping rows and columns is complete pivoting, A = P L U Q. The book's localised set matches neither. Our
          reading is that it describes a communication-avoiding variant, the subject of its reference [6]; we have not
          checked that.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/appA-facts.md A15; infra-facts.md C11)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              cuSOLVER <code>getrf</code>, rocSOLVER <code>getrf</code> (và <code>getrf_npvt</code> không chọn phần tử chốt),{' '}
              <code>MPSMatrixDecompositionLU</code> của Metal và LAPACK của Accelerate trên CPU đều làm LU với chọn phần tử chốt
              từng phần theo nghĩa của thư viện. cuSOLVER: LU không chọn phần tử chốt "is not numerically stable" (không ổn định
              về số học).
            </p>
          }
        >
          <p>
            cuSOLVER <code>getrf</code>, rocSOLVER <code>getrf</code> (and <code>getrf_npvt</code> without pivoting), Metal's{' '}
            <code>MPSMatrixDecompositionLU</code> and Accelerate's LAPACK on the CPU all do LU with partial pivoting in the
            library sense. cuSOLVER: LU without pivoting "is not numerically stable".
          </p>
        </Translated>
      </Today>
      <Ask q={questionsSA7['appA/A.7/q-partial']} />
      <Translated
        vi={
          <p>
            Với một hệ tuyến tính trong mã của chính bạn, bạn gọi một trong các thư viện này; bạn không tự viết nó. Thứ bạn giữ lại
            từ mục này là khuôn mẫu, và nó xuất hiện trong các kernel suy luận dưới những cái tên khác.
          </p>
        }
      >
        <p>
          For a linear system in your own code you call one of these; you do not write it. What you keep from this section
          is the pattern, and it turns up in inference kernels under other names.
        </p>
      </Translated>

      <h2><Translated vi="Cùng nước đi đó trong một softmax">The same move in a softmax</Translated></h2>
      <Translated
        vi={
          <p>
            Một safe softmax trừ đi giá trị lớn nhất của hàng trước khi gọi exp, để không hàm mũ nào tràn số. Chọn phần tử chốt
            tìm số chia lớn nhất trước khi chia, để không phép chia nào bùng nổ. Co giãn theo khối MX (
            <a href={href('appA', 'A.5')}>§A.5</a>) tìm độ lớn lớn nhất trước khi chọn hệ số tỉ lệ. Cả ba đều chạy một phép rút gọn
            max trước, để phép toán nguy hiểm chỉ bao giờ gặp những độ lớn nó xử lý được.
          </p>
        }
      >
        <p>
          A safe softmax subtracts the row's maximum before calling exp, so no exponential overflows. Pivoting finds the
          largest divisor before dividing, so no division blows up. MX block scaling (<a href={href('appA', 'A.5')}>§A.5</a>)
          finds the largest magnitude before choosing the scale. All three run a max reduction first, so that the dangerous
          operation only ever sees magnitudes it can handle.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Phần nối này là của khóa học này; sách không nhắc đến softmax.</p>}>
          <p>This bridge is ours; the book does not mention softmax.</p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Những gì trang này bỏ qua">What this page leaves out</Translated></h2>
      <Translated
        vi={
          <p>
            Không có lab bộ giải, không có kernel LU, và không có gì ngoài cái tên cho các thuật toán tránh truyền thông và chọn
            phần tử chốt ngẫu nhiên hóa (p. 599). Tài liệu tham khảo [6] của sách (Ballard, Demmel, Holtz và Schwartz) là nơi bắt
            đầu cho hướng thứ nhất.
          </p>
        }
      >
        <p>
          No solver lab, no LU kernel, and nothing beyond a name for communication-avoiding algorithms and randomised
          pivoting (p. 599). The book's reference [6] (Ballard, Demmel, Holtz and Schwartz) is the place to start for the
          first.
        </p>
      </Translated>
    </>
  );
}
