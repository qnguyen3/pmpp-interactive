import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr, useLocale } from '../../lib/i18n';
import {
  BOOK_BLOCK,
  BOOK_NS,
  EXPRESSIONS,
  KERNELS,
  ORDERS,
  SCENE,
  blockCount,
  blockWork,
  ceilDiv,
  ex9,
  expression,
  launchWith,
  makeData,
  runKernel,
  schedule,
  showValue,
  smallestFailing,
  smallestWasting,
  verdict,
  waveCount,
  type ExprKey,
  type KernelKind,
  type LaunchResult,
  type Order,
  type Range,
} from '../../sim/launch';
import { OneLaunchScene } from './OneLaunchScene';

// Section 2.6, "Calling a kernel". Every number an interactive, a prediction or a
// question shows comes from src/sim/launch.ts, which is unit-tested against the
// book's worked values (pp. 38 to 39) and its exercises 4 and 9.

const num = (v: number) => v.toLocaleString('en-US');
const plural = (v: number, one: string, many = one + 's') => `${num(v)} ${v === 1 ? one : many}`;
const rangeText = (r: Range) =>
  r.first === r.last ? tr(`element ${num(r.first)}`, `phần tử ${num(r.first)}`) : tr(`elements ${num(r.first)} to ${num(r.last)}`, `các phần tử ${num(r.first)} đến ${num(r.last)}`);
const MONO = { fontVariationSettings: "'MONO' 1, 'CASL' 0" } as const; // i18n-ignore: CSS font axes

// Book exercise 9 (Fig. 2.15, p. 42), as stored in labs/ch02/snippets/ex9_foo.cu.
// S26.snippets.test.ts fails if this copy and that file drift apart.
const EX9 = `__global__ void foo_kernel(float* a, float* b, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        b[i] = 2.7f * a[i] - 4.3f;
    }
}
void foo(float* a_d, float* b_d) {
    unsigned int N = 200000;
    foo_kernel<<<(N + 128 - 1) / 128, 128>>>(a_d, b_d, N);
}`;

const LAB1 = 'ch02/lab1_vecadd';
const N_MAX = 2000000;

// ---------------------------------------------------------------- shared: a typed whole number
/** A number box that keeps what you type and reports it only when it is a whole number in range. */
function WholeNumber({
  value,
  onChange,
  min,
  max,
  label,
  width = '8.5rem',
}: {
  value: number | null;
  onChange: (v: number | null) => void;
  min: number;
  max: number;
  label: string;
  width?: string;
}) {
  const [text, setText] = useState(value == null ? '' : String(value));
  // A preset or a +1 button changed the value from outside: show it.
  useEffect(() => {
    if (value != null && Number(text) !== value) setText(String(value));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);
  return (
    <input
      type="number"
      min={min}
      max={max}
      step={1}
      value={text}
      aria-label={label}
      style={{ width }}
      onChange={(e) => {
        const t = e.target.value;
        setText(t);
        const v = Number(t);
        onChange(t.trim() !== '' && Number.isInteger(v) && v >= min && v <= max ? v : null);
      }}
    />
  );
}

// ---------------------------------------------------------------- shared: what a launch covers
interface WindowBlock {
  k: number;
  launched: boolean;
  first: number;
  last: number;
  /** Threads (or, in a block that was never created, elements) with an index below n. */
  working: number;
}

/** The last block that did something and whatever comes after it: at most two blocks for the five expressions. */
function endWindow(r: LaunchResult): WindowBlock[] {
  const needed = ceilDiv(r.n, r.B);
  const from = Math.max(0, Math.min(r.blocks, needed) - 1);
  const to = Math.max(r.blocks, needed, 1) - 1;
  return Array.from({ length: to - from + 1 }, (_, j) => {
    const w = blockWork(r.n, r.B, from + j);
    return { k: w.block, launched: w.block < r.blocks, first: w.first, last: w.last, working: w.working };
  });
}

/** Two bars: the whole array against the threads launched, and the end of it enlarged. */
function Coverage({ r }: { r: LaunchResult }) {
  const total = Math.max(r.n, r.threads, 1);
  const pct = (v: number) => (100 * v) / total + '%';
  const win = endWindow(r);
  const bar = { position: 'relative', height: 18, background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 2 } as const; // i18n-ignore: CSS value
  const part = { position: 'absolute', top: -1, bottom: -1, minWidth: 3, borderRadius: 2 } as const;
  return (
    <div data-coverage>
      <div className="mx-label">
        <Translated
          vi={
            <>
              Mảng, {r.n === 1 ? 'phần tử 0' : `các phần tử 0 đến ${num(r.n - 1)}`},{' '}
              {r.threads ? `so với ${num(r.threads)} luồng của lần khởi chạy` : 'so với một lần khởi chạy không có luồng nào'}
            </>
          }
        >
          The array, {r.n === 1 ? 'element 0' : `elements 0 to ${num(r.n - 1)}`},{' '}
          {r.threads ? `against the ${num(r.threads)} threads of the launch` : 'against a launch with no threads'}
        </Translated>
      </div>
      <div role="img" aria-label={tr('Elements processed, elements never processed and surplus threads', 'Các phần tử được xử lý, các phần tử không bao giờ được xử lý và các luồng thừa')} style={bar}>
        {r.working ? (
          <div style={{ ...part, left: 0, width: pct(r.working), background: 'var(--global-wash)', border: '1px solid var(--global)' }} />
        ) : null}
        {r.unprocessedCount ? (
          <div data-tail="lost" style={{ ...part, right: 0, width: pct(r.unprocessedCount), background: 'var(--fail)' }} />
        ) : null}
        {r.surplus ? (
          <div data-tail="surplus" style={{ ...part, right: 0, width: pct(r.surplus), background: 'var(--cell)', border: '1px dashed var(--ink-2)' }} />
        ) : null}
      </div>
      <div className="mx-label" style={{ marginTop: '0.8rem' }}>
        <Translated
          vi={
            <>
              Phần cuối, phóng to: {win.length === 1 ? `khối ${num(win[0].k)}` : `các khối ${num(win[0].k)} và ${num(win[win.length - 1].k)}`}
            </>
          }
        >
          The end of it, enlarged: {win.length === 1 ? `block ${num(win[0].k)}` : `blocks ${num(win[0].k)} and ${num(win[win.length - 1].k)}`}
        </Translated>
      </div>
      <div style={{ display: 'flex', gap: '0.8rem' }}>
        {win.map((b) => {
          const empty = b.launched && b.working === 0;
          const edge = b.launched ? 'var(--ink-2)' : 'var(--fail)';
          return (
            <div key={b.k} data-block={b.k} data-launched={b.launched ? 'yes' : 'no'} style={{ flex: '1 1 0', minWidth: 0 }}>
              <div style={{ fontSize: '0.76rem', marginBottom: '0.2rem', color: b.launched ? 'var(--ink-2)' : 'var(--fail)' }}>
                <b style={{ color: b.launched ? 'var(--ink)' : 'var(--fail)' }}><Translated vi={<>khối {num(b.k)}</>}>block {num(b.k)}</Translated></b>{' '}
                {b.launched ? (empty ? tr('launched, no thread works', 'đã khởi chạy, không luồng nào làm việc') : tr('launched', 'đã khởi chạy')) : tr('never created', 'không bao giờ được tạo')}
              </div>
              <div style={{ position: 'relative', height: 24, borderRadius: 2, background: b.launched ? 'var(--cell)' : 'transparent',
                border: `${empty ? 2 : 1}px ${b.launched && empty ? 'solid' : 'dashed'} ${edge}` }}>
                {b.working ? (
                  <div style={{ ...part, left: -1, width: `calc(${(100 * b.working) / r.B}% + 2px)`, maxWidth: 'calc(100% + 2px)',
                    background: b.launched ? 'var(--global-wash)' : 'var(--fail)', border: `1px solid ${b.launched ? 'var(--global)' : 'var(--fail)'}` }} />
                ) : null}
              </div>
              <div className="muted" style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', fontVariantNumeric: 'tabular-nums' }}>
                <span>{num(b.first)}</span>
                <span>{num(b.last)}</span>
              </div>
            </div>
          );
        })}
      </div>
      <ul style={{ margin: '0.5rem 0 0', paddingLeft: '1.2em' }}>
        {win.map((b) => {
          const rest = r.B - b.working;
          return (
            <li key={b.k} style={{ marginBottom: '0.2em' }}>
              <Translated
                vi={
                  <>
                    Khối {num(b.k)}, chỉ số {num(b.first)} đến {num(b.last)}:{' '}
                    {b.launched ? (
                      <>
                        {b.working ? <><span className="t-global">{num(b.working)} luồng ghi</span> một phần tử</> : <b>không luồng nào có phần tử để ghi</b>}
                        {rest ? <>{b.working ? '; ' : ', cả '}{num(rest)} luồng bị phép kiểm tra biên chặn lại</> : null}.
                      </>
                    ) : (
                      <>
                        <b className="t-fail">không có luồng nào cho {rangeText({ first: b.first, last: b.first + b.working - 1 })}</b>.
                      </>
                    )}
                  </>
                }
              >
                Block {num(b.k)}, indices {num(b.first)} to {num(b.last)}:{' '}
                {b.launched ? (
                  <>
                    {b.working ? <><span className="t-global">{plural(b.working, 'thread')} {b.working === 1 ? 'writes' : 'write'}</span> an element</> : <b>no thread has an element to write</b>}
                    {rest ? <>{b.working ? '; ' : ', all '}{plural(rest, 'thread')} {b.working && rest === 1 ? 'is' : 'are'} stopped by the guard</> : null}.
                  </>
                ) : (
                  <>
                    <b className="t-fail">no thread exists for {rangeText({ first: b.first, last: b.first + b.working - 1 })}</b>.
                  </>
                )}
              </Translated>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/** One line saying what a launch did, from the model. */
function summary(r: LaunchResult): string {
  const head = tr(`${plural(r.blocks, 'block')}, ${plural(r.threads, 'thread')}`, `${num(r.blocks)} khối, ${num(r.threads)} luồng`);
  if (r.unprocessedCount) return tr(`${head}; ${rangeText(r.unprocessed[0])} ${r.unprocessedCount === 1 ? 'is' : 'are'} never written.`, `${head}; ${rangeText(r.unprocessed[0])} không bao giờ được ghi.`);
  if (r.emptyBlocks) return tr(`${head}; every element is written, and ${plural(r.emptyBlocks, 'whole block')} of ${num(r.B)} threads has nothing to do.`, `${head}; mọi phần tử đều được ghi, và ${num(r.emptyBlocks)} khối trọn vẹn ${num(r.B)} luồng không có việc gì làm.`);
  return tr(`${head}; every element is written and ${r.idle === 0 ? 'no thread is idle' : `${plural(r.idle, 'thread')} ${r.idle === 1 ? 'is' : 'are'} stopped by the guard`}.`, `${head}; mọi phần tử đều được ghi và ${r.idle === 0 ? 'không luồng nào rảnh' : `${num(r.idle)} luồng bị phép kiểm tra biên chặn lại`}.`);
}

// ---------------------------------------------------------------- interactive A: launch calculator
function LaunchCalculator() {
  const [n, setN] = useState<number>(BOOK_NS[0]);
  const [typed, setTyped] = useState<number | null>(BOOK_NS[0]);
  const [B, setB] = useState<number>(BOOK_BLOCK);
  const [key, setKey] = useState<ExprKey>('int-ceil');
  const set = (v: number) => {
    const c = Math.max(1, Math.min(N_MAX, v));
    setN(c);
    setTyped(c);
  };
  const rows = useMemo(() => EXPRESSIONS.map((e) => ({ e, r: launchWith(e.key, n, B) })), [n, B]);
  const locale = useLocale();
  const count = useMemo(() => blockCount(key, n, B), [key, n, B, locale]);
  const r = rows.find((x) => x.e.key === key)!.r;
  const expr = expression(key);

  return (
    <Bench
      title={tr('Launch calculator', 'Máy tính lần khởi chạy')}
      task={tr('Which of the five expressions give enough blocks, and no block too many, for every n?', 'Biểu thức nào trong năm biểu thức cho đủ số khối, và không thừa khối nào, với mọi n?')}
      foot={
        <Translated
          vi={
            <>
              Một phần tử mỗi luồng, một chiều, <code>int n</code>. Bốn nút là các độ dài mà sách tính mẫu
              (pp. 38–39). <span className="t-global">Màu hổ phách</span> là bộ nhớ toàn cục của device mà một luồng ghi;{' '}
              <span className="t-fail">màu đỏ</span> là bộ nhớ không luồng nào chạm tới.
            </>
          }
        >
          <>
            One element per thread, one dimension, <code>int n</code>. The four buttons are the book's worked lengths
            (pp. 38–39). <span className="t-global">Amber</span> is device global memory that a thread writes;{' '}
            <span className="t-fail">red</span> is memory no thread reaches.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '1rem', alignItems: 'flex-end' }}>
        <label className="field">
          <span>
            <Translated vi={<>Số phần tử, <code>n</code></>}>
              Elements, <code>n</code>
            </Translated>
          </span>
          <WholeNumber
            value={typed}
            min={1}
            max={N_MAX}
            label="n"
            onChange={(v) => {
              setTyped(v);
              if (v != null) setN(v);
            }}
          />
        </label>
        <span className="row" style={{ gap: '0.3rem' }}>
          <button className="quiet" aria-label={tr('n minus 1', 'n trừ 1')} onClick={() => set(n - 1)} disabled={n <= 1}>
            −1
          </button>
          <button className="quiet" aria-label={tr('n plus 1', 'n cộng 1')} onClick={() => set(n + 1)} disabled={n >= N_MAX}>
            +1
          </button>
        </span>
        <label className="field">
          <span><Translated vi="Kéo n, từ 1 đến 4096">Drag n, 1 to 4096</Translated></span>
          <input type="range" min={1} max={4096} step={1} value={Math.min(n, 4096)} aria-label={tr('n slider', 'thanh trượt n')} style={{ width: '14rem' }}
            onChange={(e) => set(Number(e.target.value))} />
        </label>
        <label className="field">
          <span>
            <Translated vi={<>Số luồng mỗi khối, <code>B</code> = <b data-out="B">{num(B)}</b></>}>
              Threads per block, <code>B</code> = <b data-out="B">{num(B)}</b>
            </Translated>
          </span>
          <input type="range" min={32} max={1024} step={32} value={B} aria-label={tr('threads per block', 'số luồng mỗi khối')} style={{ width: '11rem' }}
            onChange={(e) => setB(Number(e.target.value))} />
        </label>
        <span className="row" role="group" aria-label={tr("The book's values of n", 'Các giá trị n của sách')} style={{ gap: '0.4rem' }}>
          <span className="muted"><Translated vi="Các n của sách">The book's n</Translated></span>
          {BOOK_NS.map((v) => (
            <button key={v} className="quiet" aria-pressed={n === v && B === BOOK_BLOCK}
              onClick={() => {
                set(v);
                setB(BOOK_BLOCK);
              }}>
              {num(v)}
            </button>
          ))}
        </span>
      </div>
      {typed == null ? (
        <p className="t-fail" role="alert" style={{ margin: '0 0 0.8rem' }}>
          <Translated vi={<>n phải là một số nguyên từ 1 đến {num(N_MAX)}. Vẫn đang hiển thị n = {num(n)}.</>}>
            n must be a whole number from 1 to {num(N_MAX)}. Still showing n = {num(n)}.
          </Translated>
        </p>
      ) : null}

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 7fr)' }}>
        <div>
          <div className="panel" style={{ overflowX: 'auto' }}>
            <table className="data" style={{ margin: 0 }} role="radiogroup" aria-label={tr('Block-count expression', 'Biểu thức số khối')}>
              <thead>
                <tr>
                  <th style={{ paddingLeft: '0.3rem' }}><Translated vi="Số khối, như được viết">Number of blocks, as written</Translated></th>
                  <th className="num"><Translated vi="Số khối">Blocks</Translated></th>
                  <th><Translated vi={<>Tại n = {num(n)}</>}>At n = {num(n)}</Translated></th>
                </tr>
              </thead>
              <tbody>
                {rows.map(({ e, r: x }) => (
                  <tr key={e.key} data-expr={e.key} data-blocks={x.blocks} data-lost={x.unprocessedCount} data-empty={x.emptyBlocks}
                    onClick={() => setKey(e.key)} style={{ cursor: 'pointer', background: e.key === key ? 'var(--rule-soft)' : undefined }}>
                    <td style={{ paddingLeft: '0.3rem' }}>
                      {/* On a phone the expression may wrap, at spaces only, so that the table fits. */}
                      <label className="check">
                        <input type="radio" name="s26-expr" checked={e.key === key} onChange={() => setKey(e.key)} aria-label={e.generic} />{/* i18n-ignore: radio group name */}
                        <code style={{ whiteSpace: 'normal', overflowWrap: 'normal' }}>{e.source(B)}</code>
                      </label>
                    </td>
                    <td className="num">{num(x.blocks)}</td>
                    <td>
                      {x.unprocessedCount ? (
                        <span className="chip fail"><Translated vi={<>{num(x.unprocessedCount)} không bao giờ được xử lý</>}>{num(x.unprocessedCount)} never processed</Translated></span>
                      ) : x.emptyBlocks ? (
                        <b><Translated vi="một khối không có việc">a block with no work</Translated></b>
                      ) : (
                        <span className="chip ok"><Translated vi="phủ đủ n">covers n</Translated></span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="panel" style={{ marginTop: '0.8rem' }} data-eval>
            <div className="mx-label">
              <Translated vi={<>C++ tính <code>{expr.source(B)}</code> với <code>int n = {n}</code> như thế nào</>}>
                How C++ evaluates <code>{expr.source(B)}</code> with <code>int n = {n}</code>
              </Translated>
            </div>
            <ol style={{ margin: 0, paddingLeft: '1.3em' }}>
              {count.steps.map((s, i) => (
                <li key={i} data-step={i} style={{ marginBottom: '0.35em' }}>
                  <Translated
                    vi={
                      <>
                        <code>{s.text}</code> là <b>{showValue(s.value, s.type)}</b>, một{' '}
                        <code>{s.type === 'unsigned' ? 'unsigned int' : s.type}</code>
                      </>
                    }
                  >
                    <code>{s.text}</code> is <b>{showValue(s.value, s.type)}</b>, {s.type === 'int' ? 'an' : 'a'}{' '}
                    <code>{s.type === 'unsigned' ? 'unsigned int' : s.type}</code>
                  </Translated>
                  <br />
                  <span className="muted">{s.rule}.</span>
                </li>
              ))}
            </ol>
            <p style={{ margin: '0.5rem 0 0' }}>
              <Translated
                vi={
                  <>
                    Lần khởi chạy nhận <b data-out="blocks">{num(count.blocks)}</b>. {num(n)} phần tử trong các khối {num(B)} luồng cần{' '}
                    <b data-out="needed">{num(count.needed)}</b>.
                  </>
                }
              >
                The launch gets <b data-out="blocks">{num(count.blocks)}</b>. {num(n)} elements in blocks of {num(B)} need{' '}
                <b data-out="needed">{num(count.needed)}</b>.
              </Translated>
            </p>
          </div>
        </div>

        <div>
          <div className="panel" style={{ overflowX: 'auto' }}>
            <code data-out="statement" style={{ whiteSpace: 'nowrap' }}>
              vecAddKernel&lt;&lt;&lt;{expr.source(B)}, {B}&gt;&gt;&gt;(A_d, B_d, C_d, n);
            </code>
            <table className="data" style={{ margin: '0.6rem 0 0' }}>
              <tbody>
                <tr>
                  <td><Translated vi="Số luồng được khởi chạy, số khối × B">Threads launched, blocks × B</Translated></td>
                  <td className="num" data-out="threads">{num(r.threads)}</td>
                </tr>
                <tr>
                  <td><Translated vi={<>Số luồng vượt qua phép kiểm tra <code>i &lt; n</code></>}>Threads that pass the guard <code>i &lt; n</code></Translated></td>
                  <td className="num" data-out="working">{num(r.working)}</td>
                </tr>
                <tr>
                  <td><Translated vi="Số luồng bị phép kiểm tra biên chặn">Threads the guard stops</Translated></td>
                  <td className="num" data-out="idle">{num(r.idle)}</td>
                </tr>
                <tr>
                  <td><Translated vi="Số phần tử không bao giờ được xử lý">Elements never processed</Translated></td>
                  <td className="num" data-out="lost">
                    {r.unprocessedCount ? <span className="chip fail">{num(r.unprocessedCount)}</span> : 0}
                  </td>
                </tr>
                <tr>
                  <td><Translated vi="Số khối trọn vẹn không có luồng nào làm việc">Whole blocks in which no thread works</Translated></td>
                  <td className="num" data-out="empty">{r.emptyBlocks ? <b>{num(r.emptyBlocks)}</b> : 0}</td>
                </tr>
              </tbody>
            </table>
            <p data-out="summary" role="status" style={{ margin: '0.6rem 0 0' }}>
              {r.correct && !r.emptyBlocks ? <span className="chip ok"><Translated vi="đúng với n này">right for this n</Translated></span> : r.correct ? <b><Translated vi="đúng, nhưng lãng phí:">correct, with waste:</Translated></b> : <span className="chip fail"><Translated vi="sai với n này">wrong for this n</Translated></span>}{' '}
              {summary(r)}
            </p>
          </div>
          {r.zero ? (
            <div className="panel" style={{ marginTop: '0.8rem', overflowX: 'auto' }} data-zero>
              <div className="mx-label"><Translated vi="Một lưới 0 khối. Lần khởi chạy làm gì, theo những gì mỗi nguồn nói">A grid of 0 blocks. What the launch does, as far as each source says</Translated></div>
              <table className="data" style={{ margin: 0 }}>
                <tbody>
                  {r.zero.map((z) => (
                    <tr key={z.backend} data-backend={z.backend}>
                      <td className="nowrap">{z.name}</td>
                      <td className="nowrap"><b>{tr(z.outcome, ({ 'not documented': 'không có trong tài liệu', 'error, documented': 'lỗi, có trong tài liệu', error: 'lỗi' } as Record<string, string>)[z.outcome] ?? z.outcome)}</b></td>
                      <td>{z.outcome === 'not documented' ? z.detail : <code>{z.detail}</code>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
          <div className="panel" style={{ marginTop: '0.8rem' }}>
            <Coverage r={r} />
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive B: find the n that breaks it
const HUNT: ExprKey[] = ['trunc', 'ceil-trunc', 'plus-one'];
const HUNT_B = BOOK_BLOCK;

interface HuntTarget {
  /** 'fail': the test can fail. 'waste': it never fails, and the defect is a block without work. */
  mode: 'fail' | 'waste';
  smallest: number;
}
function huntTarget(key: ExprKey): HuntTarget {
  const fails = smallestFailing(key, HUNT_B);
  return fails != null ? { mode: 'fail', smallest: fails } : { mode: 'waste', smallest: smallestWasting(key, HUNT_B)! };
}

function HuntCard({ k, index, onSolved }: { k: ExprKey; index: number; onSolved: (key: ExprKey, solved: boolean) => void }) {
  const target = useMemo(() => huntTarget(k), [k]);
  const [a, setA] = useState<number | null>(null);
  const [b, setB] = useState<number | null>(null);
  const [ran, setRan] = useState<{ a: number; b: number } | null>(null);
  const va = ran ? verdict(k, ran.a, HUNT_B) : null;
  const vb = ran ? verdict(k, ran.b, HUNT_B) : null;
  const okA = !!va && va.passes;
  const hit = !!vb && (target.mode === 'fail' ? !vb.passes : vb.wastes);
  const okB = hit && ran!.b === target.smallest;
  const solved = okA && okB;
  useEffect(() => onSolved(k, solved), [k, solved, onSolved]);
  const statement = `vecAddKernel<<<${expression(k).source(HUNT_B)}, ${HUNT_B}>>>(A_d, B_d, C_d, n);`;
  return (
    <div className="panel" data-hunt={k} data-solved={solved ? 'true' : 'false'} style={{ marginBottom: '0.8rem' }}>
      <p style={{ margin: '0 0 0.6rem' }}>
        <span className="muted">{index + 1}.</span> <code>{statement}</code>{' '}
        {solved ? <span className="chip ok"><Translated vi="đã tìm ra">found</Translated></span> : null}
      </p>
      <div className="row" style={{ alignItems: 'flex-end' }}>
        <label className="field">
          <span><Translated vi="Một n mà bài kiểm tra vượt qua">An n for which the test passes</Translated></span>
          <WholeNumber value={a} min={1} max={N_MAX} label={tr(`passing n for ${k}`, `n vượt qua cho ${k}`)} onChange={setA} />
        </label>
        <label className="field">
          <span>
            {target.mode === 'fail'
              ? tr('The smallest n, 1 or more, for which it fails', 'n nhỏ nhất, từ 1 trở lên, mà nó thất bại')
              : tr('The smallest n for which a whole block has no work', 'n nhỏ nhất mà một khối trọn vẹn không có việc')}
          </span>
          <WholeNumber value={b} min={1} max={N_MAX} label={tr(`smallest bad n for ${k}`, `n hỏng nhỏ nhất cho ${k}`)} onChange={setB} />
        </label>
        <button className="primary" disabled={a == null || b == null} onClick={() => setRan({ a: a!, b: b! })}>
          <Translated vi="Chạy cả hai lần khởi chạy">Run both launches</Translated>
        </button>
      </div>
      {ran && va && vb ? (
        <div role="status" style={{ marginTop: '0.7rem' }}>
          <p data-feedback="pass" style={{ margin: '0 0 0.4rem' }}>
            {okA ? <span className="chip ok"><Translated vi="vượt qua">passes</Translated></span> : <span className="chip fail"><Translated vi="thất bại">fails</Translated></span>} n = {num(ran.a)}:{' '}
            {summary(va.result)}{' '}
            {okA
              ? target.mode === 'fail'
                ? tr(`${num(ran.a)} is a multiple of ${HUNT_B}, so nothing was dropped.`, `${num(ran.a)} là bội số của ${HUNT_B}, nên không có gì bị bỏ sót.`)
                : tr('So does every other n: no element is ever lost with this expression, and a test of the result cannot tell it from the right one.', 'Mọi n khác cũng vậy: với biểu thức này không phần tử nào bị mất, và một bài kiểm tra kết quả không thể phân biệt nó với biểu thức đúng.')
              : tr('The comparison with the CPU loop finds the difference. Try another n.', 'Phép so sánh với vòng lặp CPU tìm ra khác biệt. Hãy thử một n khác.')}
          </p>
          <p data-feedback="bad" style={{ margin: 0 }}>
            {okB ? <span className="chip ok"><Translated vi="nhỏ nhất">smallest</Translated></span> : hit ? <b><Translated vi="chưa phải nhỏ nhất">not the smallest</Translated></b> : <span className="chip fail"><Translated vi="không phải n này">not this one</Translated></span>}{' '}
            n = {num(ran.b)}: {summary(vb.result)}{' '}
            {okB
              ? target.mode === 'fail'
                ? tr(`${ran.b} / ${HUNT_B} is 0 in integers. No block is created at all.`, `${ran.b} / ${HUNT_B} bằng 0 trong số nguyên. Không khối nào được tạo cả.`)
                : tr(`${ran.b} / ${HUNT_B} + 1 is ${vb.blocks}: one block does all the work and the other does none.`, `${ran.b} / ${HUNT_B} + 1 bằng ${vb.blocks}: một khối làm toàn bộ công việc và khối kia không làm gì.`)
              : hit
                ? target.mode === 'fail'
                  ? tr('It does fail here. A smaller n fails too.', 'Nó đúng là thất bại ở đây. Một n nhỏ hơn cũng thất bại.')
                  : tr('A block is wasted here. A smaller n wastes one too.', 'Một khối bị lãng phí ở đây. Một n nhỏ hơn cũng lãng phí một khối.')
                : target.mode === 'fail'
                  ? tr('The test passes at this n.', 'Bài kiểm tra vượt qua tại n này.')
                  : tr('No block is without work at this n.', 'Tại n này không khối nào thiếu việc.')}
          </p>
          {!okB || !okA ? null : (
            <div style={{ marginTop: '0.7rem' }}>
              <Coverage r={vb.result} />
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}

function FindTheN() {
  const [solved, setSolved] = useState<Record<string, boolean>>({});
  const onSolved = useMemo(() => (key: ExprKey, s: boolean) => setSolved((m) => (m[key] === s ? m : { ...m, [key]: s })), []);
  const found = HUNT.filter((k) => solved[k]).length;
  const multiple = 4 * HUNT_B;
  return (
    <Bench
      title={tr('Find the n that breaks it', 'Tìm n làm nó hỏng')}
      task={tr('For each launch: one n a test would pass, and the smallest n that exposes it.', 'Với mỗi lần khởi chạy: một n mà bài kiểm tra sẽ vượt qua, và n nhỏ nhất làm lộ ra lỗi.')}
      foot={
        <Translated
          vi={
            <>
              Kích thước khối {HUNT_B} cho cả ba. "Bài kiểm tra" so sánh mọi phần tử của C với vòng lặp CPU. Không có gì
              được hiển thị cho đến khi bạn chạy hai giá trị của mình.
            </>
          }
        >
          <>
            Block size {HUNT_B} in all three. "The test" compares every element of C with the CPU loop. Nothing is shown
            until you run your two values.
          </>
        </Translated>
      }
    >
      <p className="muted" data-hunt-score style={{ margin: '0 0 0.8rem' }}>
        <Translated vi={<>Đã tìm ra {found} trên {HUNT.length}</>}>
          {found} of {HUNT.length} found
        </Translated>
      </p>
      {HUNT.map((k, i) => (
        <HuntCard key={k} k={k} index={i} onSolved={onSolved} />
      ))}
      {found === HUNT.length ? (
        <div className="panel" data-hunt-done>
          <Translated
            vi={
              <p style={{ margin: 0 }}>
                <b>Đã tìm ra cả ba.</b> Một bài kiểm tra tại n = {num(multiple)} vượt qua với cả ba:{' '}
                {HUNT.map((k) => `${expression(k).source(HUNT_B)} cho ${verdict(k, multiple, HUNT_B).blocks} khối`).join(', ')}.
                Hai biểu thức đầu thất bại ở hai bên một ranh giới khối và dưới một khối. Biểu thức thứ ba không bao giờ làm
                một bài kiểm tra kết quả thất bại, và phải được bắt bằng cách đếm khối. Vì vậy kích thước kiểm tra nên nằm
                ngay dưới, đúng bằng và ngay trên một bội số của kích thước khối, và dưới một khối. Lab 2.1 kiểm tra n = 1,
                31, 32, 33, 255, 256, 257, 1000 và 4097 vì lý do đó.
              </p>
            }
          >
            <p style={{ margin: 0 }}>
              <b>All three found.</b> A test at n = {num(multiple)} passes for every one of them:{' '}
              {HUNT.map((k) => `${expression(k).source(HUNT_B)} gives ${verdict(k, multiple, HUNT_B).blocks} blocks`).join(', ')}.
              The first two fail on either side of a block boundary and below one block. The third never fails a test of
              the result, and has to be caught by counting blocks. Test sizes therefore belong just under, on and just
              over a multiple of the block size, and below one block. Lab 2.1 tests n = 1, 31, 32, 33, 255, 256, 257,
              1000 and 4097 for that reason.
            </p>
          </Translated>
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive C: any order, any device
const ORDER_LABEL: Record<Order, string> = {
  get ascending() {
    return tr('ascending', 'tăng dần');
  },
  get descending() {
    return tr('descending', 'giảm dần');
  },
  get shuffled() {
    return tr('shuffled', 'xáo trộn');
  },
};
const C_SIZES = [2, 3, 4];
const nRange = (B: number) => ({ min: 3 * B + 1, max: Math.min(12 * B, 32) });
const S_MAX = 8;
const LABEL_CELL = { border: 0, minWidth: '2.3rem', justifyItems: 'start' } as const;
const list = (xs: number[]) => (xs.length <= 1 ? xs.join('') : xs.slice(0, -1).join(', ') + tr(' and ', ' và ') + xs[xs.length - 1]);

function AnyOrder() {
  const [n, setN] = useState<number>(SCENE.n);
  const [B, setB] = useState<number>(4);
  const [S, setS] = useState<number>(SCENE.S);
  const [order, setOrder] = useState<Order>('shuffled');
  const [seed, setSeed] = useState<number>(SCENE.seed);
  const [kind, setKind] = useState<KernelKind>('independent');
  const [w, setW] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [all, setAll] = useState(false);

  const blocks = ceilDiv(n, B);
  const data = useMemo(() => makeData(n, B), [n, B]);
  const sched = useMemo(() => schedule(blocks, S, order, seed), [blocks, S, order, seed]);
  const run = useMemo(() => runKernel(kind, sched, data), [kind, sched, data]);
  const waves = sched.waves.length;
  const at = Math.min(w, waves);
  const finished = at === waves;
  const written = useMemo(() => new Set(run.written.slice(0, at).flat()), [run, at]);
  const fresh = useMemo(() => new Set(at > 0 ? run.written[at - 1] : []), [run, at]);
  const now = at > 0 ? run.after[at - 1] : data.C0;
  const wrong = new Set(finished ? run.wrong : []);
  const running = new Set(at > 0 && !finished ? sched.waves[at - 1] : []);

  useEffect(() => {
    setW(0);
    setPlaying(false);
  }, [n, B, S, order, seed, kind]);
  useEffect(() => {
    if (!playing) return;
    if (at >= waves) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setW((x) => x + 1), 650);
    return () => window.clearTimeout(id);
  }, [playing, at, waves]);

  const range = nRange(B);
  const grid = useMemo(
    () =>
      all
        ? ORDERS.map((o) => ({
            order: o,
            cells: Array.from({ length: S_MAX }, (_, i) => runKernel(kind, schedule(blocks, i + 1, o, seed), data).matches),
          }))
        : null,
    [all, kind, blocks, seed, data],
  );
  const same = grid ? grid.flatMap((g) => g.cells).filter(Boolean).length : 0;

  return (
    <Bench
      title={tr('Any order, any device', 'Thứ tự bất kỳ, device bất kỳ')}
      task={tr("First kernel: try to make C come out different. Second kernel: find the one setting that gives the loop's answer.", 'Kernel thứ nhất: hãy thử làm C ra kết quả khác. Kernel thứ hai: tìm thiết lập duy nhất cho ra đáp án của vòng lặp.')}
      foot={
        <Translated
          vi={
            <>
              S là một thiết lập của mô hình này, không phải thuộc tính của GPU nào. Ở đây C bắt đầu bằng 0 để có thể kiểm
              tra tổng bằng mắt; một vùng cấp phát thật không được xóa. Một dấu chấm là một phần tử chưa có gì ghi vào.
            </>
          }
        >
          <>
            S is a setting of this model, not a property of any GPU. C starts at 0 here so that the sums can be checked
            by eye; a real allocation is not cleared. A dot is an element nothing has written yet.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <span>
            <Translated vi={<>Số phần tử, <code>n</code> = <b data-out="n">{n}</b></>}>
              Elements, <code>n</code> = <b data-out="n">{n}</b>
            </Translated>
          </span>
          <input type="range" min={range.min} max={range.max} step={1} value={n} aria-label="n" style={{ width: '10rem' }}
            onChange={(e) => setN(Number(e.target.value))} />
        </label>
        <label className="field">
          <Translated vi="Số luồng mỗi khối">Threads per block</Translated>
          <select value={B} aria-label={tr('threads per block', 'số luồng mỗi khối')} onChange={(e) => {
            const nb = Number(e.target.value);
            const r = nRange(nb);
            setB(nb);
            setN(Math.max(r.min, Math.min(r.max, n)));
          }}>
            {C_SIZES.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <label className="field">
          <span>
            <Translated vi={<>Số khối device chạy cùng lúc, S = <b data-out="S">{S}</b></>}>
              Blocks the device runs at once, S = <b data-out="S">{S}</b>
            </Translated>
          </span>
          <input type="range" min={1} max={S_MAX} step={1} value={S} aria-label="S" style={{ width: '9rem' }}
            onChange={(e) => setS(Number(e.target.value))} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Kernel', 'Kernel')}>
          {KERNELS.map((kk) => (
            <button key={kk.kind} aria-pressed={kind === kk.kind} data-kernel={kk.kind} onClick={() => setKind(kk.kind)}>
              <code>{kk.body}</code>
            </button>
          ))}
        </span>
        <span className="seg" role="group" aria-label={tr('Order the device takes blocks in', 'Thứ tự device nhận các khối')}>
          {ORDERS.map((o) => (
            <button key={o} aria-pressed={order === o} data-order={o} onClick={() => setOrder(o)}>
              {ORDER_LABEL[o]}
            </button>
          ))}
        </span>
        <button className="quiet" onClick={() => { setOrder('shuffled'); setSeed((s) => s + 1); }}>
          <Translated vi="Xáo lại">Reshuffle</Translated>
        </button>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button className="primary" onClick={() => { if (finished) setW(0); setPlaying(!playing); }}>
          {playing ? tr('Pause', 'Tạm dừng') : finished ? tr('Play again', 'Phát lại') : at > 0 ? tr('Resume', 'Tiếp tục') : tr('Play', 'Phát')}
        </button>
        <button onClick={() => { setPlaying(false); setW(at + 1); }} disabled={finished}>
          <Translated vi="Đợt tiếp theo">Next wave</Translated>
        </button>
        <button onClick={() => { setPlaying(false); setW(waves); }} disabled={finished}>
          <Translated vi="Chạy đến cuối">Run to the end</Translated>
        </button>
        <button className="quiet" onClick={() => { setPlaying(false); setW(0); }} disabled={at === 0}>
          <Translated vi="Đặt lại">Reset</Translated>
        </button>
        <span className="muted" aria-live="polite" data-out="progress">
          <Translated vi={<>đợt {at} trên {waves}</>}>
            wave {at} of {waves}
          </Translated>
        </span>
      </div>

      <div className="narrate" role="status" data-out="status" style={{ marginBottom: '0.9rem', minHeight: '4.6em' }}>
        <Translated
          vi={
            <>
              <span data-out="blocks">{blocks}</span> khối {B} luồng, {S} khối mỗi lần: <b data-out="waves">{num(waveCount(blocks, S))} đợt</b>.
              Thứ tự: <span data-out="order" style={MONO}>{sched.waves.map((x) => x.join(' ')).join('  |  ')}</span>.
            </>
          }
        >
          <span data-out="blocks">{blocks}</span> blocks of {B} threads, {S} at a time: <b data-out="waves">{plural(waveCount(blocks, S), 'wave')}</b>.
          Order: <span data-out="order" style={MONO}>{sched.waves.map((x) => x.join(' ')).join('  |  ')}</span>.
        </Translated>
        <br />
        {at === 0
          ? tr('Nothing has run yet.', 'Chưa có gì chạy.')
          : tr(
              `Wave ${at}: block${sched.waves[at - 1].length === 1 ? '' : 's'} ${list(sched.waves[at - 1])} ${sched.waves[at - 1].length === 1 ? 'ran' : 'ran together'}.`,
              `Đợt ${at}: ${sched.waves[at - 1].length === 1 ? 'khối' : 'các khối'} ${list(sched.waves[at - 1])} ${sched.waves[at - 1].length === 1 ? 'đã chạy' : 'đã chạy cùng nhau'}.`,
            )}{' '}
        {finished ? (
          run.matches ? (
            <span className="chip ok" data-out="verdict"><Translated vi="C bằng kết quả của vòng lặp tuần tự">C equals the sequential loop's result</Translated></span>
          ) : (
            <span className="chip fail" data-out="verdict">
              <Translated vi={<>{run.wrong.length} trên {n} phần tử khác với vòng lặp tuần tự</>}>
                {run.wrong.length} of {n} elements differ from the sequential loop
              </Translated>
            </span>
          )
        ) : null}
      </div>

      <div className="row" style={{ alignItems: 'flex-start', gap: '0.9rem 0.8rem' }} data-strip>
        {Array.from({ length: blocks }, (_, k) => {
          const bwk = blockWork(n, B, k);
          const cells = Array.from({ length: B }, (_, t) => k * B + t);
          const live = running.has(k);
          return (
            <div key={k} data-block={k} data-wave={sched.waves.findIndex((x) => x.includes(k)) + 1}>
              <div className="mx-label" style={{ fontWeight: live ? 650 : 500, color: live ? 'var(--ink)' : 'var(--ink-2)' }}>
                <Translated vi={<>khối {k}</>}>
                  block {k}
                </Translated>
                <span className="muted" style={{ fontWeight: 400 }}>
                  <Translated vi={<> · đợt {sched.waves.findIndex((x) => x.includes(k)) + 1}</>}> · wave {sched.waves.findIndex((x) => x.includes(k)) + 1}</Translated>
                </span>
              </div>
              <div className="mx global" style={{ gridTemplateColumns: `auto repeat(${B}, auto)`, alignItems: 'center' }}>
                <span className="void" style={LABEL_CELL}>A</span>
                {cells.map((i) => (i < n ? <span key={i}>{data.A[i]}</span> : <span key={i} className="void" />))}
                {kind === 'independent' ? (
                  <>
                    <span className="void" style={LABEL_CELL}>B</span>
                    {cells.map((i) => (i < n ? <span key={i}>{data.Bv[i]}</span> : <span key={i} className="void" />))}
                  </>
                ) : null}
                <span className="void t-global" style={{ ...LABEL_CELL, fontWeight: 650 }}>C</span>
                {cells.map((i) =>
                  i < n ? (
                    <span key={i} data-c={i} data-state={!written.has(i) ? 'unwritten' : wrong.has(i) ? 'wrong' : 'written'}
                      className={!written.has(i) ? '' : wrong.has(i) ? 'bad' : fresh.has(i) && !finished ? 'hot' : 'on'}>
                      {written.has(i) ? now[i] : '·'}
                    </span>
                  ) : (
                    <span key={i} className="void" title={tr('a thread the guard stops', 'một luồng bị phép kiểm tra biên chặn')} />
                  ),
                )}
                <span className="void" style={LABEL_CELL}><Translated vi="vòng lặp">loop</Translated></span>
                {cells.map((i) => (i < n ? <span key={i} className="void" data-want={i}>{run.expected[i]}</span> : <span key={i} className="void" />))}
              </div>
              {bwk.working < B ? (
                <div className="muted" style={{ fontSize: '0.72rem', marginTop: '0.15rem' }}>
                  <Translated vi={<>{num(B - bwk.working)} luồng bị phép kiểm tra biên chặn</>}>
                    {plural(B - bwk.working, 'thread')} stopped by the guard
                  </Translated>
                </div>
              ) : null}
            </div>
          );
        })}
      </div>

      <div className="panel" style={{ marginTop: '1rem', overflowX: 'auto' }}>
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <span>
            <Translated
              vi={
                <>
                  Mọi thiết lập cùng lúc, cho <code>{KERNELS.find((x) => x.kind === kind)!.body}</code>: ba thứ tự, S từ 1 đến {S_MAX}.
                  Hãy quyết định bạn mong đợi gì trước.
                </>
              }
            >
              Every setting at once, for <code>{KERNELS.find((x) => x.kind === kind)!.body}</code>: three orders, S from 1 to {S_MAX}.
              Decide what you expect first.
            </Translated>
          </span>
          <button className="quiet" onClick={() => setAll(!all)}>
            {all ? tr('Hide the table', 'Ẩn bảng') : tr(`Try all ${ORDERS.length * S_MAX} settings`, `Thử cả ${ORDERS.length * S_MAX} thiết lập`)}
          </button>
        </div>
        {grid ? (
          <>
            <table className="data" style={{ margin: '0.6rem 0 0' }} data-all>
              <thead>
                <tr>
                  <th><Translated vi="Thứ tự">Order</Translated></th>
                  {Array.from({ length: S_MAX }, (_, i) => <th key={i}>S = {i + 1}</th>)}
                </tr>
              </thead>
              <tbody>
                {grid.map((g) => (
                  <tr key={g.order}>
                    <td>{ORDER_LABEL[g.order]}</td>
                    {g.cells.map((ok, i) => (
                      <td key={i} data-outcome={ok ? 'same' : 'differs'}>
                        {ok ? <span className="chip ok"><Translated vi="giống">same</Translated></span> : <span className="chip fail"><Translated vi="khác">differs</Translated></span>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <p style={{ margin: '0.5rem 0 0' }} data-out="all-summary">
              <Translated vi={<>{same} trên {ORDERS.length * S_MAX} thiết lập cho ra C của vòng lặp tuần tự.</>}>
                {same} of {ORDERS.length * S_MAX} settings give the sequential loop's C.
              </Translated>
            </p>
          </>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- numbers the page states, from the model
const B1000 = launchWith('book-ceil', 1000, BOOK_BLOCK);
const BOOK_REST = BOOK_NS.slice(1);
const EX4 = launchWith('int-ceil', 8000, 1024);
const BAD1000 = launchWith('ceil-trunc', 1000, BOOK_BLOCK);
const BAD1024 = launchWith('ceil-trunc', 1024, BOOK_BLOCK);
const E9 = ex9();
const E9_COUNT = blockCount('int-ceil', E9.N, E9.threadsPerBlock);

/** What a prediction of an exact count is told. */
function Exact({ got, want, children }: { got: number; want: number; children: ReactNode }) {
  return (
    <p>
      <b>{got === want ? tr('That is it.', 'Chính xác.') : tr(`It is ${num(want)}.`, `Đáp án là ${num(want)}.`)}</b> {children}
    </p>
  );
}

export const questionsS26: Record<string, Question> = {
  'ch02/2.6/q-threads-8000': {
    kind: 'numeric',
    id: 'ch02/2.6/q-threads-8000',
    prompt: (
      <Translated
        vi={
          <>
            Một vector 8000 phần tử, một phần tử mỗi luồng, khối 1024 luồng, và số khối nhỏ nhất phủ được vector. Lưới có
            bao nhiêu luồng?
          </>
        }
      >
        <>
          A vector of 8000 elements, one element per thread, blocks of 1024 threads, and the smallest number of blocks
          that covers the vector. How many threads are in the grid?
        </>
      </Translated>
    ),
    answer: EX4.threads,
    unit: 'threads',
    explain: (
      <Translated
        vi={
          <p>
            8000 / 1024 nhỏ hơn 8 một chút, nên cần {EX4.blocks} khối, và {EX4.blocks} × 1024 ={' '}
            {num(EX4.threads)}. Khối cuối có {EX4.idle} luồng có chỉ số từ 8000 trở lên; phép kiểm tra biên chặn chúng
            lại. Lưới luôn chứa một số nguyên khối, nên số luồng là bội số của kích thước khối và hiếm khi bằng n.
          </p>
        }
      >
        <p>
          8000 / 1024 is a little under 8, so {EX4.blocks} blocks are needed, and {EX4.blocks} × 1024 ={' '}
          {num(EX4.threads)}. The last block has {EX4.idle} threads whose index is 8000 or more; the guard stops them.
          The grid always holds a whole number of blocks, so the thread count is a multiple of the block size and is
          rarely n.
        </p>
      </Translated>
    ),
    source: 'book exercise 4, p. 41; method from §2.6, p. 38',
    vi: { unit: 'luồng', source: 'bài tập 4 của sách, p. 41; phương pháp từ §2.6, p. 38' },
  },
  'ch02/2.6/q-intdiv': {
    kind: 'choice',
    id: 'ch02/2.6/q-intdiv',
    prompt: (
      <Translated
        vi={
          <>
            Với <code>int n = 1000</code>, mã host chạy{' '}
            <code>vecAddKernel&lt;&lt;&lt;ceil(n/256), 256&gt;&gt;&gt;(A_d, B_d, C_d, n);</code> Điều gì xảy ra?
          </>
        }
      >
        <>
          With <code>int n = 1000</code>, host code runs{' '}
          <code>vecAddKernel&lt;&lt;&lt;ceil(n/256), 256&gt;&gt;&gt;(A_d, B_d, C_d, n);</code> What happens?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="4 khối được khởi chạy và C đúng: ceil làm tròn 3.90625 lên thành 4.">
            4 blocks are launched and C is correct: ceil rounds 3.90625 up to 4.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là điều ceil(n/256.0) làm. Ở đây cả hai toán hạng của phép chia đều là int.">
            That is what ceil(n/256.0) does. Here both operands of the division are int.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi={`${BAD1000.blocks} khối được khởi chạy, và các phần tử ${BAD1000.unprocessed[0].first} đến ${BAD1000.unprocessed[0].last} của C không bao giờ được ghi.`}>
            {`${BAD1000.blocks} blocks are launched, and elements ${BAD1000.unprocessed[0].first} to ${BAD1000.unprocessed[0].last} of C are never written.`}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nó không biên dịch được: ceil không thể nhận một int.">
            It does not compile: ceil cannot take an int.
          </Translated>
        ),
        why: (
          <Translated vi="Nó biên dịch được. Số int được chuyển sang double khi đi vào ceil, sau khi phép chia đã xảy ra.">
            It compiles. The int is converted to double on the way into ceil, after the division has already happened.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="4 khối được khởi chạy, nhưng các luồng của khối cuối đều không vượt qua phép kiểm tra biên và không làm gì.">
            4 blocks are launched, but the threads of the last block all fail the guard and do nothing.
          </Translated>
        ),
        why: (
          <Translated vi="Không có khối thứ tư. Phép kiểm tra biên không liên quan: mọi luồng tồn tại đều có chỉ số nhỏ hơn n.">
            There is no fourth block. The guard is not involved: every thread that exists has an index below n.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>n/256</code> chia hai số int, nên nó đã là {BAD1000.blocks} trước khi <code>ceil</code> thấy nó, và
            trần của {BAD1000.blocks} là {BAD1000.blocks}. Lần khởi chạy tạo {num(BAD1000.threads)} luồng cho{' '}
            {num(BAD1000.n)} phần tử. Không có gì báo lỗi, và bộ nhớ device từ <code>cudaMalloc</code> không được xóa, nên
            phần đuôi của C chứa các byte cũ. Một bài kiểm tra với n = 1024 sẽ vượt qua:{' '}
            {BAD1024.blocks} khối, {BAD1024.correct ? 'mọi phần tử đều được ghi' : 'thiếu phần tử'}.
          </p>
        }
      >
        <p>
          <code>n/256</code> divides two ints, so it is {BAD1000.blocks} before <code>ceil</code> sees it, and the
          ceiling of {BAD1000.blocks} is {BAD1000.blocks}. The launch creates {num(BAD1000.threads)} threads for{' '}
          {num(BAD1000.n)} elements. Nothing reports an error, and device memory from <code>cudaMalloc</code> is not
          cleared, so the tail of C holds old bytes. A test with n = 1024 would pass:{' '}
          {BAD1024.blocks} blocks, {BAD1024.correct ? 'every element written' : 'elements missing'}.
        </p>
      </Translated>
    ),
    source: '§2.6, p. 38 (why the book writes 256.0); "not cleared" from notes/today/ch02-facts.md, A2',
    vi: { source: '§2.6, p. 38 (lý do sách viết 256.0); "không được xóa" từ notes/today/ch02-facts.md, A2' },
  },
  'ch02/2.6/q-launch-check': {
    kind: 'choice',
    id: 'ch02/2.6/q-launch-check',
    prompt: (
      <Translated
        vi={
          <>
            Một lần khởi chạy yêu cầu 4096 luồng mỗi khối, nhiều hơn mức device cho phép. Mã host kiểm tra giá trị trả về
            của mọi lời gọi API theo mẫu của sách và không làm gì khác. Dòng bổ sung nào báo lỗi này ngay tại nơi nó xảy
            ra?
          </>
        }
      >
        <>
          A launch asks for 4096 threads per block, more than the device allows. The host code checks the return value
          of every API call with the book's pattern and does nothing else. Which additional line reports this failure
          at the place it happened?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi={<>Gán câu lệnh khởi chạy cho một <code>cudaError_t</code> và kiểm tra nó.</>}>
            <>Assign the launch statement to a <code>cudaError_t</code> and check that.</>
          </Translated>
        ),
        why: (
          <Translated vi="Một lần khởi chạy bằng dấu ngoặc nhọn ba không trả về cudaError_t. Không có gì để gán.">
            A launch with the chevrons returns no cudaError_t. There is nothing to assign.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi={<><code>CHECK(cudaGetLastError());</code> ngay sau lệnh khởi chạy.</>}>
            <><code>CHECK(cudaGetLastError());</code> immediately after the launch.</>
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi={<>Gọi <code>cudaGetErrorString()</code> sau lệnh khởi chạy.</>}>
            <>Call <code>cudaGetErrorString()</code> after the launch.</>
          </Translated>
        ),
        why: (
          <Translated vi="Hàm đó biến một mã lỗi thành văn bản. Bạn vẫn cần một mã để đưa vào nó.">
            That turns an error code into text. You still need a code to give it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi={<>Không cần gì thêm. Lời gọi <code>cudaMemcpy</code> có kiểm tra theo sau sẽ báo lỗi.</>}>
            <>Nothing more. The checked <code>cudaMemcpy</code> that follows will report it.</>
          </Translated>
        ),
        why: (
          <Translated vi="Không có tài liệu nào nói một lời gọi sau đó sẽ báo một lần khởi chạy bị từ chối, nên có thể nó sẽ không bao giờ được báo (lab 2.4 cho thấy điều này). Lỗi từ một kernel đã thực sự chạy thì có quay về từ các lời gọi sau, và đó là lý do chúng bị báo ở sai dòng. Hãy kiểm tra ngay sau lệnh khởi chạy.">
            Nothing documented says a later call reports a launch that was refused, so it may never be reported at all (lab 2.4 shows this). Errors from a kernel that did run do come back from later calls, which is how they end up reported on the wrong line. Check straight after the launch.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Câu lệnh khởi chạy là dòng duy nhất của hàm host không có giá trị trả về. Hướng dẫn của NVIDIA bảo hãy kiểm tra
            trạng thái lỗi ngay sau nó, và ví dụ của chính hướng dẫn là lần khởi chạy này: khối 4096 luồng, được macro
            kiểm tra của nó báo là "invalid argument". Thành công ở đó chỉ có nghĩa là cấu hình đã được chấp nhận. Nó không
            có nghĩa là kernel đã chạy đúng, hay đã bắt đầu.
          </p>
        }
      >
        <p>
          The launch statement is the one line of the host function with no return value. NVIDIA's guide says to check
          the error state immediately after it, and its own example is this launch: blocks of 4096 threads, reported
          as "invalid argument" by its check macro. Success there means only that the configuration was
          accepted. It does not mean the kernel ran correctly, or that it has started.
        </p>
      </Translated>
    ),
    source: 'not in the book; notes/today/ch02-facts.md, A2 and A6 (CUDA Programming Guide, sections 2.1.7.1 and 2.1.7.2)',
    vi: { source: 'không có trong sách; notes/today/ch02-facts.md, A2 và A6 (CUDA Programming Guide, mục 2.1.7.1 và 2.1.7.2)' },
  },
  'ch02/2.6/q-timer': {
    kind: 'choice',
    id: 'ch02/2.6/q-timer',
    prompt: (
      <Translated
        vi={
          <>
            Giả sử một bộ đếm thời gian phía host chỉ bao quanh câu lệnh khởi chạy đọc được 0.02 ms, và cùng bộ đếm đó bao
            quanh lệnh khởi chạy cộng <code>cudaDeviceSynchronize()</code> đọc được 3.1 ms. (Các số đo này được bịa ra cho
            câu hỏi.) Bộ đếm thứ nhất đã đo cái gì?
          </>
        }
      >
        <>
          Suppose a host timer around the launch statement alone reads 0.02 ms, and the same timer around the launch
          plus <code>cudaDeviceSynchronize()</code> reads 3.1 ms. (The readings are invented for this question.) What
          did the first timer measure?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Kernel. 3 ms còn lại là chi phí của chính lời gọi cudaDeviceSynchronize.">
            The kernel. The other 3 ms are the cost of the cudaDeviceSynchronize call itself.
          </Translated>
        ),
        why: (
          <Translated vi="cudaDeviceSynchronize chặn cho đến khi công việc đã phát lệnh hoàn tất. 3 ms đó chính là công việc ấy.">
            cudaDeviceSynchronize blocks until the work already issued has completed. The 3 ms is that work.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Việc giao lần khởi chạy cho runtime. Khi bộ đếm đó dừng, lưới chưa xong, có thể còn chưa bắt đầu.">
            Handing the launch to the runtime. When that timer stopped, the grid had not finished, perhaps not started.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Các luồng vượt qua phép kiểm tra biên. Phần thời gian còn lại là của các luồng rảnh.">
            The threads that pass the guard. The rest of the time is the idle threads.
          </Translated>
        ),
        why: (
          <Translated vi="Mọi luồng của lưới thuộc cùng một lần khởi chạy. Không bộ đếm nào trên host tách chúng ra.">
            All threads of the grid belong to the same launch. No timer on the host separates them.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Việc sao chép A và B sang device, thứ mà lần khởi chạy chờ.">
            The copy of A and B to the device, which the launch waits for.
          </Translated>
        ),
        why: (
          <Translated vi="Các lệnh sao chép là những lời gọi riêng thực hiện trước lần khởi chạy, nằm ngoài cả hai bộ đếm.">
            The copies are separate calls made before the launch, outside both timers.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một lần khởi chạy là bất đồng bộ đối với luồng host: kernel được thiết lập để chạy, và host đi tiếp mà không
            chờ nó hoàn tất hay thậm chí bắt đầu. Một lần đọc đồng hồ ngay sau câu lệnh đã đo việc bàn giao. Để đo kernel,
            hãy đồng bộ hóa trước khi đọc đồng hồ, và chạy lần khởi chạy một lần trước đó để chi phí khởi động một lần của
            runtime không nằm trong con số.
          </p>
        }
      >
        <p>
          A launch is asynchronous with respect to the host thread: the kernel is set up to run, and the host goes on
          without waiting for it to complete or even to start. A clock read straight after the statement has timed
          the hand-over. To time the kernel, synchronise before you read the clock, and run the launch once beforehand
          so that the runtime's one-time start-up cost is not in the figure.
        </p>
      </Translated>
    ),
    source: 'not in the book; notes/today/ch02-facts.md, A5 (CUDA Programming Guide, sections 2.1.2.2, 2.1.4, 2.1.6)',
    vi: { source: 'không có trong sách; notes/today/ch02-facts.md, A5 (CUDA Programming Guide, mục 2.1.2.2, 2.1.4, 2.1.6)' },
  },
  'ch02/2.6/q-order': {
    kind: 'choice',
    id: 'ch02/2.6/q-order',
    prompt: (
      <Translated vi={<>Thân kernel nào cho cùng kết quả bất kể các khối chạy theo thứ tự nào?</>}>
        <>Which kernel body gives the same result whatever order the blocks run in?</>
      </Translated>
    ),
    options: [
      { label: <><code>C[i] = A[i] + B[i]</code></>, correct: true },
      {
        label: <><code>C[i] = C[i-1] + A[i]</code></>,
        why: (
          <Translated vi="Luồng đầu tiên của một khối đọc phần tử mà luồng cuối của khối trước ghi. Nó thấy gì ở đó phụ thuộc vào khối nào chạy trước.">
            The first thread of a block reads the element the last thread of the previous block writes. What it finds there depends on which block ran first.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cả hai. Runtime chạy các khối theo thứ tự chỉ số.">
            Both. The runtime runs blocks in index order.
          </Translated>
        ),
        why: (
          <Translated vi="Không có thứ tự nào như vậy được hứa hẹn. Mô hình lập trình đòi hỏi các khối có thể chạy theo thứ tự bất kỳ, song song hoặc lần lượt.">
            No such order is promised. The programming model requires that blocks can run in any order, in parallel or one after another.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không cái nào. Mọi kernel đều có thể ra kết quả khác khi thứ tự thay đổi.">
            Neither. Any kernel can come out differently when the order changes.
          </Translated>
        ),
        why: (
          <Translated vi="Một kernel mà không luồng nào đọc thứ luồng khác ghi thì không thể.">
            A kernel in which no thread reads what another thread writes cannot.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong thân thứ nhất, mỗi luồng đọc hai đầu vào không ai ghi và ghi một đầu ra không ai khác chạm vào, nên thứ
            tự không thể quan trọng. Thân thứ hai là tổng tích lũy của <a href={href('ch02', '2.1')}>phần 2.1</a>: một đầu
            ra được tính từ một đầu ra khác. Trình khám phá ở trên chạy nó trong 24 thiết lập, và chỉ một trong số đó, mỗi
            lần một khối theo thứ tự tăng dần, tái tạo được vòng lặp. Ngay cả thiết lập đó cũng chỉ đúng trong mô hình, vốn
            chạy các luồng của một khối theo thứ tự chỉ số; một device cũng không hứa hẹn thứ tự nào bên trong một khối.
          </p>
        }
      >
        <p>
          In the first body each thread reads two inputs nobody writes and writes one output nobody else touches, so
          the order cannot matter. The second is the running sum of <a href={href('ch02', '2.1')}>section 2.1</a>: one
          output is computed from another. The explorer above runs it in 24 settings, and only one of them, one block
          at a time in ascending order, reproduces the loop. Even that one holds only in the model, which runs the
          threads of a block in index order; a device promises no order inside a block either.
        </p>
      </Translated>
    ),
    source: '§2.6, p. 39; threads inside a block, ch. 4, p. 74; notes/today/ch02-facts.md, A18',
    vi: { source: '§2.6, p. 39; luồng bên trong một khối, ch. 4, p. 74; notes/today/ch02-facts.md, A18' },
  },
  'ch02/2.6/q-scale': {
    kind: 'choice',
    id: 'ch02/2.6/q-scale',
    prompt: (
      <Translated
        vi={
          <>
            Phép cộng vector của bạn chạy trên một GPU thứ hai thực thi gấp bốn lần số khối cùng lúc. Cần thay đổi gì trong
            kernel hoặc trong câu lệnh khởi chạy?
          </>
        }
      >
        <>
          Your vector addition runs on a second GPU that executes four times as many blocks at once. What has to change
          in the kernel or in the launch statement?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Số khối: chia nó cho bốn.">The number of blocks: divide it by four.</Translated>,
        why: (
          <Translated vi="Số khối là trần của n chia cho kích thước khối. Nó mô tả công việc, không phải device.">
            The number of blocks is the ceiling of n over the block size. It describes the work, not the device.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Kích thước khối: nhân nó với bốn, để mỗi khối làm nhiều hơn.">
            The block size: multiply it by four, so that each block does more.
          </Translated>
        ),
        why: (
          <Translated vi="Kích thước khối nằm trong giới hạn số luồng mỗi khối và không liên quan gì đến việc bao nhiêu khối chạy cùng nhau.">
            The block size stays within the limit on threads per block and has nothing to do with how many blocks run together.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Không gì cả. Cùng một lần khởi chạy là đúng cho cả hai GPU.">Nothing. The same launch is right for both GPUs.</Translated>,
        correct: true,
      },
      {
        label: (
          <Translated vi="Không gì trong mã nguồn, nhưng n phải là bội số của số khối chạy đồng thời mới.">
            Nothing in the source, but n must be a multiple of the new number of concurrent blocks.
          </Translated>
        ),
        why: (
          <Translated vi="Lưới không gắn với việc bao nhiêu khối chạy cùng nhau. Một nhóm khối cuối, nhỏ hơn, đơn giản là chạy một mình.">
            The grid is not tied to how many blocks run together. A last, smaller group of blocks simply runs alone.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lần khởi chạy cho biết có bao nhiêu công việc, tính bằng khối. Device quyết định chạy bao nhiêu khối trong số đó
            cùng nhau. Vì không khối nào được phép phụ thuộc vào khối khác, chạy bao nhiêu khối mỗi lần cũng cho cùng kết
            quả, và một device chạy được nhiều khối hơn sẽ xong sớm hơn. Sách gọi đây là khả năng mở rộng (scalability) và
            quay lại với nó ở <a href={href('ch04')}>chương 4</a>.
          </p>
        }
      >
        <p>
          The launch says how much work there is, in blocks. The device decides how many of those it runs together.
          Because no block may depend on another, any number at a time gives the same result, and a device that runs
          more of them finishes sooner. The book calls this scalability and returns to it in{' '}
          <a href={href('ch04')}>chapter 4</a>.
        </p>
      </Translated>
    ),
    source: '§2.6, p. 39',
  },
};

// ---------------------------------------------------------------- listings
/** Lines of a lab file from the first one that is not part of the opening comment. */
function withoutHeader(src: string) {
  const lines = src.replace(/\n$/, '').split('\n');
  const from = Math.max(0, lines.findIndex((l) => !l.startsWith('//')));
  return { code: lines.slice(from).join('\n'), firstLine: from + 1, lines };
}
/** Lines of a lab file from the one that starts with `start`. */
function fromLine(src: string, start: string) {
  const lines = src.replace(/\n$/, '').split('\n');
  const from = Math.max(0, lines.findIndex((l) => l.startsWith(start)));
  return { code: lines.slice(from).join('\n'), firstLine: from + 1, lines };
}
/** 1-based numbers of the lines that contain `needle`. */
const linesWith = (lines: string[], needle: string) => lines.flatMap((l, i) => (l.includes(needle) ? [i + 1] : []));

// One CUDA statement, and what stands for each part of it in Metal. `needles` find the lines in launch.mm.
const spanText = (at: number[]) =>
  at.length === 0 ? '…' : at.length === 1 ? String(at[0]) : at[at.length - 1] - at[0] === at.length - 1 ? tr(`${at[0]} to ${at[at.length - 1]}`, `${at[0]} đến ${at[at.length - 1]}`) : at.slice(0, -1).join(', ') + tr(' and ', ' và ') + at[at.length - 1];
const METAL_STEPS: { cuda: ReactNode; metal: ReactNode; needles: string[]; note?: string }[] = [
  {
    get cuda() {
      return tr('The default stream: where a launch with no stream named is queued', 'Stream mặc định: nơi một lần khởi chạy không nêu tên stream được xếp hàng');
    },
    metal: (
      <Translated vi={<>Một command queue, tạo một lần bằng <code>newCommandQueue</code> và được truyền cho hàm</>}>
        <>A command queue, made once with <code>newCommandQueue</code> and handed to the function</>
      </Translated>
    ),
    needles: ['id<MTLCommandQueue> queue'],
    get note() {
      return tr('a parameter', 'một tham số');
    },
  },
  {
    get cuda() {
      return tr('Nothing', 'Không có');
    },
    metal: (
      <Translated vi={<>Một command buffer lấy từ queue</>}>
        <>A command buffer taken from the queue</>
      </Translated>
    ),
    needles: ['[queue commandBuffer]'], // i18n-ignore: source text searched for in launch.mm
  },
  {
    get cuda() {
      return tr('Nothing', 'Không có');
    },
    metal: (
      <Translated vi={<>Một compute encoder trên command buffer đó</>}>
        <>A compute encoder on that command buffer</>
      </Translated>
    ),
    needles: ['computeCommandEncoder]'],
  },
  {
    cuda: (
      <Translated vi={<>Tên hàm, <code>vecAddKernel</code></>}>
        <>The function name, <code>vecAddKernel</code></>
      </Translated>
    ),
    metal: (
      <Translated vi={<><code>setComputePipelineState:</code>, với kernel đã được biên dịch cho GPU này</>}>
        <><code>setComputePipelineState:</code>, with the kernel compiled for this GPU</>
      </Translated>
    ),
    needles: ['setComputePipelineState:'],
  },
  {
    cuda: (
      <Translated vi={<>Danh sách đối số, <code>(A_d, B_d, C_d, n)</code></>}>
        <>The argument list, <code>(A_d, B_d, C_d, n)</code></>
      </Translated>
    ),
    metal: (
      <Translated
        vi={
          <>
            <code>setBuffer:offset:atIndex:</code>, mỗi buffer một lời gọi. Một giá trị vô hướng như <code>n</code> sẽ đi
            qua <code>setBytes:length:atIndex:</code>, vốn nhận dữ liệu dưới 4 KB. Kernel này không cần cái nào
          </>
        }
      >
        <>
          <code>setBuffer:offset:atIndex:</code>, one call per buffer. A scalar such as <code>n</code> would go through{' '}
          <code>setBytes:length:atIndex:</code>, which takes data under 4 KB. This kernel needs none
        </>
      </Translated>
    ),
    needles: ['setBuffer:'],
  },
  {
    cuda: <><code>&lt;&lt;&lt;numBlocks, threadsPerBlock&gt;&gt;&gt;</code></>,
    metal: (
      <Translated vi={<><code>dispatchThreads:threadsPerThreadgroup:</code>, được cho số luồng</>}>
        <><code>dispatchThreads:threadsPerThreadgroup:</code>, given the number of threads</>
      </Translated>
    ),
    needles: ['dispatchThreads:'],
  },
  {
    get cuda() {
      return tr('Implicit in the statement', 'Ngầm định trong câu lệnh');
    },
    metal: (
      <Translated vi={<><code>endEncoding</code>, rồi <code>commit</code></>}>
        <><code>endEncoding</code>, then <code>commit</code></>
      </Translated>
    ),
    needles: ['endEncoding]', 'commandBuffer commit]'], // i18n-ignore: source text searched for in launch.mm
  },
  {
    cuda: <><code>cudaDeviceSynchronize()</code></>,
    metal: <><code>waitUntilCompleted</code></>,
    needles: ['waitUntilCompleted]'],
  },
  {
    cuda: (
      <Translated vi={<><code>cudaGetLastError()</code> và lời gọi đồng bộ hóa có kiểm tra</>}>
        <><code>cudaGetLastError()</code> and the checked synchronise</>
      </Translated>
    ),
    metal: (
      <Translated vi={<><code>status</code> và <code>error</code> của command buffer, đọc sau khi chờ</>}>
        <>The command buffer's <code>status</code> and <code>error</code>, read after the wait</>
      </Translated>
    ),
    needles: ['commandBuffer.status'],
  },
];

export function S26() {
  const cu = useLabFile(LAB1 + '/solution/kernel.cu');
  const mm = useLabFile(LAB1 + '/solution/launch.mm');
  const { reports } = useReports();
  // The same words as the labs page, from this machine's runs.
  const cuStatus = solutionTabStatus(reports, LAB1, 'kernel.cu');
  const mmStatus = solutionTabStatus(reports, LAB1, 'launch.mm');

  const host = cu ? withoutHeader(cu) : null;
  const launchLine = host ? linesWith(host.lines, 'vecAddKernel<<<')[0] : undefined;
  const mmPart = mm ? fromLine(mm, 'int vecAdd(') : null; // i18n-ignore: source text searched for in launch.mm
  const mmLine = (needles: string[]) => (mmPart ? needles.flatMap((x) => linesWith(mmPart.lines, x)) : []);
  const mmHighlight = METAL_STEPS.flatMap((s) => mmLine(s.needles));

  return (
    <>
      <Translated
        vi={
          <p>
            Kernel của <a href={href('ch02', '2.5')}>phần 2.5</a> không làm gì cho đến khi mã host gọi nó. Lời gọi là một
            câu lệnh, và nó mang hai con số mà không lời gọi hàm thông thường nào có: tạo bao nhiêu khối, và đặt bao nhiêu
            luồng vào mỗi khối. Phần này nói về việc tính đúng con số thứ nhất với mọi n, về việc tìm hiểu lần khởi chạy có
            thành công không, và về những gì bạn được và không được giả định về thứ tự mà các khối sau đó chạy.
          </p>
        }
      >
        <p>
          The kernel of <a href={href('ch02', '2.5')}>section 2.5</a> does nothing until host code calls it. The call is
          one statement, and it carries two numbers that no ordinary function call has: how many blocks to create, and
          how many threads to put in each. This section is about getting the first number right for any n, about
          finding out whether the launch worked, and about what you may assume, and may not, about the order in which
          the blocks then run.
        </p>
      </Translated>

      <h2><Translated vi="Hai con số giữa các dấu ngoặc nhọn ba">The two numbers between the chevrons</Translated></h2>
      <Translated
        vi={
          <p>
            Một lời gọi kernel đặt kích thước lưới của nó qua các tham số cấu hình thực thi (execution configuration
            parameters). Chúng được viết giữa <code>&lt;&lt;&lt;</code> và <code>&gt;&gt;&gt;</code>, sau tên kernel và
            trước danh sách đối số thông thường. Tham số thứ nhất là số khối trong lưới. Tham số thứ hai là số luồng trong
            mỗi khối. Với phép cộng vector, câu lệnh là{' '}
            <code>vecAddKernel&lt;&lt;&lt;numBlocks, 256&gt;&gt;&gt;(A_d, B_d, C_d, n);</code> và sách cố định kích thước
            khối ở 256, để chỉ số khối phụ thuộc vào n.
          </p>
        }
      >
        <p>
          A kernel call sets the size of its grid through execution configuration parameters. They are written between{' '}
          <code>&lt;&lt;&lt;</code> and <code>&gt;&gt;&gt;</code>, after the kernel's name and before the ordinary
          argument list. The first is the number of blocks in the grid. The second is the number of threads in each
          block. For the vector addition the statement is{' '}
          <code>vecAddKernel&lt;&lt;&lt;numBlocks, 256&gt;&gt;&gt;(A_d, B_d, C_d, n);</code> and the book fixes the
          block size at 256, so that only the number of blocks depends on n.
        </p>
      </Translated>
      <Source section="2.6" pages="37–39" figs="Fig. 2.12" />
      <Translated
        vi={
          <p>
            Cả hai con số là các biểu thức C++ thông thường mà host tính lúc chạy, trước khi tính các đối số trong ngoặc
            đơn. Vì vậy cách công việc được cắt thành các khối là một đối số như mọi đối số khác, và một giá trị sai không
            phải lỗi kiểu. Nó là một lần khởi chạy phủ quá ít dữ liệu, hoặc quá nhiều. Kiểu của cả hai là{' '}
            <code>dim3</code>, một bộ ba kích thước mà <a href={href('ch03')}>chương 3</a> dùng đầy đủ. Với một lưới một
            chiều, một số nguyên thông thường được chấp nhận.
          </p>
        }
      >
        <p>
          Both numbers are ordinary C++ expressions that the host evaluates at run time, before it evaluates the
          arguments in the parentheses. So the way the work is cut into blocks is an argument like any other, and a
          wrong value is not a type error. It is a launch that covers too little of the data, or too much. The type of
          both is <code>dim3</code>, a triple of sizes that <a href={href('ch03')}>chapter 3</a> uses in full. For a
          grid in one dimension a plain integer is accepted.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, A8; ch05-facts.md, A7)', '2026-11-20 (notes/today/ch02-facts.md, A8; và ch05-facts.md, A7)')}>
        <Translated
          vi={
            <p>
              Dạng đầy đủ trong hướng dẫn là{' '}
              <code>&lt;&lt;&lt;grid_dim, block_dim, dynamic_smem_bytes, stream&gt;&gt;&gt;</code>. Hai tham số cuối là
              tùy chọn: số byte bộ nhớ chia sẻ (chương 5) và một stream. Hai tham số đầu có kiểu <code>dim3</code>, số
              nguyên được chấp nhận cho một chiều, và cấu hình được tính trước các đối số của hàm.
            </p>
          }
        >
          <p>
            The guide's full form is{' '}
            <code>&lt;&lt;&lt;grid_dim, block_dim, dynamic_smem_bytes, stream&gt;&gt;&gt;</code>. The last two are
            optional: a byte count of shared memory (chapter 5) and a stream. The first two are of type{' '}
            <code>dim3</code>, integers are accepted for one dimension, and the configuration is evaluated before the
            function's arguments.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Bao nhiêu khối">How many blocks</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn muốn ít nhất n luồng, trong các khối B luồng. Vậy số khối là n / B làm tròn lên, tức hàm trần. Sách viết nó
            là <code>ceil(n/256.0)</code> và giải thích <code>.0</code>: nó biến phép chia thành phép chia dấu phẩy động.
            Với <code>256</code>, cả hai toán hạng sẽ là số nguyên và C++ sẽ bỏ phần thập phân trước khi <code>ceil</code>{' '}
            kịp thấy nó.
          </p>
        }
      >
        <p>
          You want at least n threads, in blocks of B. So the number of blocks is n / B rounded up, the ceiling. The
          book writes it as <code>ceil(n/256.0)</code> and explains the <code>.0</code>: it makes the division a
          floating-point division. With <code>256</code>, both operands would be integers and C++ would drop the
          fraction before <code>ceil</code> ever saw it.
        </p>
      </Translated>
      <Source section="2.6" pages="38" />
      <Predict
        id="ch02/2.6/p-blocks-1000"
        prompt={
          <Translated
            vi={
              <>
                n = 1000, với 256 luồng mỗi khối và <code>ceil(n/256.0)</code> khối. Lần khởi chạy tạo bao nhiêu luồng?
              </>
            }
          >
            <>
              n = 1000, with 256 threads per block and <code>ceil(n/256.0)</code> blocks. How many threads does the
              launch create?
            </>
          </Translated>
        }
        unit={tr('threads', 'luồng')}
      >
        {(v) => (
          <Exact got={v} want={B1000.threads}>
            <Translated
              vi={
                <>
                  1000 / 256.0 là 3.90625, nên lần khởi chạy có {B1000.blocks} khối, và {B1000.blocks} × 256 ={' '}
                  {num(B1000.threads)} luồng. {num(B1000.working)} luồng trong số đó cộng một cặp phần tử.{' '}
                  {B1000.idle} luồng còn lại có chỉ số từ 1000 trở lên và bị phép kiểm tra biên của phần 2.5 chặn lại.
                  {v === 1000 ? ' Một lưới không thể có đúng 1000 luồng với kích thước khối này: nó được tạo từ các khối trọn vẹn.' : ''}
                </>
              }
            >
              1000 / 256.0 is 3.90625, so the launch has {B1000.blocks} blocks, and {B1000.blocks} × 256 ={' '}
              {num(B1000.threads)} threads. {num(B1000.working)} of them add one pair of elements. The other{' '}
              {B1000.idle} have an index of 1000 or more and are stopped by the guard of section 2.5.
              {v === 1000 ? ' A grid cannot have exactly 1000 threads with this block size: it is made of whole blocks.' : ''}
            </Translated>
          </Exact>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Sách tính thêm ba độ dài với cùng kích thước khối: {BOOK_REST.slice(0, -1).map(num).join(', ')} và {num(BOOK_REST[BOOK_REST.length - 1])}.
            Hãy tính số khối cho mỗi độ dài trước khi bạn nạp nó vào máy tính bên dưới.
          </p>
        }
      >
        <p>
          The book works three more lengths with the same block size: {BOOK_REST.slice(0, -1).map(num).join(', ')} and {num(BOOK_REST[BOOK_REST.length - 1])}.
          Work out the number of blocks for each before you load it into the calculator below.
        </p>
      </Translated>
      <Source section="2.6" pages="38–39" />
      <Translated
        vi={
          <p>
            Khóa học này thay vào đó viết hàm trần bằng số nguyên: <code>(n + B - 1) / B</code>. Cộng B − 1 trước sẽ đẩy
            mọi n không phải bội số của B lên tới hoặc vượt qua bội số kế tiếp, và để một bội số của B nằm ngay dưới bội số
            sau nó, nên phép chia cắt cụt rơi đúng vào giá trị trần. Có ba lý do để ưu tiên nó. Đó là dạng mà hướng dẫn của
            NVIDIA dạy. Đó là dạng chính sách dùng trong mã của bài tập 9. Và đó là thứ bạn sẽ đọc thấy trong llama.cpp.
            Dạng của sách còn đưa một <code>double</code> vào một tham số có kiểu <code>dim3</code>, qua một phép chuyển đổi
            mà không trang tài liệu NVIDIA nào chúng tôi đọc có ghi. Hai dạng cho cùng số khối.
          </p>
        }
      >
        <p>
          This course writes the ceiling in integers instead: <code>(n + B - 1) / B</code>. Adding B − 1 first lifts
          every n that is not a multiple of B to or past the next multiple, and leaves a multiple of B just short of
          the one after it, so the truncating division lands on the ceiling. There are three reasons to prefer it. It is the form NVIDIA's guide teaches. It is what the book
          itself uses in the code of its exercise 9. And it is what you will read in llama.cpp. The book's form also
          hands a <code>double</code> to a parameter whose type is <code>dim3</code>, through a conversion that none of
          the NVIDIA pages we read documents. The two forms give the same number of blocks.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, A8 and D1)', '2026-11-20 (notes/today/ch02-facts.md, A8 và D1)')}>
        <Translated
          vi={
            <p>
              Hướng dẫn: <code>int blocks = (vectorLength + threads-1)/threads;</code>, hoặc{' '}
              <code>cuda::ceil_div</code> từ <code>&lt;cuda/cmath&gt;</code>. <code>scale.cu</code> của llama.cpp có cùng
              dạng với kích thước khối 256, một kích thước mà hướng dẫn gọi là tùy ý nhưng thường là điểm khởi đầu tốt.
            </p>
          }
        >
          <p>
            The guide: <code>int blocks = (vectorLength + threads-1)/threads;</code>, or{' '}
            <code>cuda::ceil_div</code> from <code>&lt;cuda/cmath&gt;</code>. llama.cpp's <code>scale.cu</code> has the
            same form with a block size of 256, a size the guide calls arbitrary but often a good one to start with.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Máy tính đánh giá năm cách viết số khối, từng bước một, theo quy tắc của C++ cho toán hạng số nguyên và dấu
            phẩy động. Hai cách là các dạng ở trên. Ba cách còn lại là những gì hay bị gõ nhầm. Hãy nạp bốn độ dài của
            sách. Rồi gõ một bội số của kích thước khối và dịch n đi một ở mỗi bên của nó, và thử một n nhỏ hơn một khối.
          </p>
        }
      >
        <p>
          The calculator evaluates five ways of writing the count, step by step, with C++'s rules for integer and
          floating-point operands. Two are the forms above. The other three are what gets typed by mistake. Load the
          book's four lengths. Then type a multiple of the block size and move n by one on either side of it, and try
          an n below one block.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Máy tính chỉ đếm và không khởi chạy gì. Nó bỏ qua giới hạn số luồng mỗi khối và số khối mỗi lưới (chương 3),
              và giả định n + B − 1 vừa trong một <code>int</code>. Một lưới 0 khối làm gì được lấy từ tài liệu và từ mã
              nguồn của trình giả lập của khóa học, không phải từ một lần chạy trên device.
            </p>
          }
        >
          <p>
            The calculator counts and launches nothing. It ignores the limits on threads per block and on blocks per
            grid (chapter 3), and assumes n + B − 1 fits in an <code>int</code>. What a grid of 0 blocks does is taken
            from documentation and from the course emulator's source, not from a run on a device.
          </p>
        </Translated>
      </Simplification>
      <LaunchCalculator />
      <Translated
        vi={
          <p>
            Hai trong năm cách đúng với mọi n. <code>n / B</code> và <code>ceil(n / B)</code> là cùng một số: chỉ đúng
            khi n là bội số của B, thiếu đúng phần dư trong trường hợp khác, và bằng không khi n nhỏ hơn B.{' '}
            <code>n / B + 1</code> không bao giờ mất phần tử, và khởi chạy thừa đúng một khối trọn vẹn khi n là bội số
            của B.
          </p>
        }
      >
        <p>
          Two of the five are right for every n. <code>n / B</code> and <code>ceil(n / B)</code> are the same number:
          right only when n is a multiple of B, short by the remainder otherwise, and zero when n is below B.{' '}
          <code>n / B + 1</code> never loses an element, and launches one whole block too many exactly when n is a
          multiple of B.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Dạng thứ tư có trong sách, trong một chú thích. Fig. 2.12 ghi "Launch ceil(n/256) blocks" phía trên đoạn mã
              dùng đúng <code>256.0</code>. Cũng hình đó đặt tên hàm là <code>vectAdd</code> và cho nó kiểu trả về{' '}
              <code>int</code> mà không có câu lệnh return, và chú thích của Fig. 2.12 và 2.13 đọc như bị đổi chỗ cho nhau.
            </p>
          }
        >
          <p>
            The fourth form is in the book, in a comment. Fig. 2.12 says "Launch ceil(n/256) blocks" above code that
            correctly uses <code>256.0</code>. The same figure names the function <code>vectAdd</code> and gives it the
            return type <code>int</code> with no return statement, and the captions of Figs. 2.12 and 2.13 read as if
            swapped.
          </p>
        </Translated>
      </Aside>
      <Ask q={questionsS26['ch02/2.6/q-intdiv']} />
      <Translated
        vi={
          <p>
            Hai kiểu lãng phí trông giống nhau trong máy tính nhưng không phải vậy. Luồng thừa trong khối cuối là bình
            thường: hướng dẫn của NVIDIA nói chúng tốn ít. Một khối trọn vẹn không có luồng nào làm việc là điều hướng dẫn
            bảo nên tránh. Và số khối bằng không là trường hợp thứ ba. HIP ghi trong tài liệu một lỗi cho nó, trình giả lập
            của khóa học báo một cấu hình không hợp lệ, và các trang CUDA chúng tôi đọc không nói điều gì xảy ra, đó là lý
            do máy tính hiển thị "not documented" ở đó.
          </p>
        }
      >
        <p>
          Two kinds of waste look alike in the calculator and are not. Surplus threads in the last block are normal:
          NVIDIA's guide says they cost little. A whole block in which no thread works is something the guide says to
          avoid. And a count of zero is a third case. HIP documents an error for it, the course emulator reports an
          invalid configuration, and the CUDA pages we read do not say what happens, which is why the calculator shows
          "not documented" there.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, A8 and B4)', '2026-11-20 (notes/today/ch02-facts.md, A8 và B4)')}>
        <Translated
          vi={
            <p>
              Hướng dẫn: khởi chạy thêm các luồng trong một khối mà không làm việc "does not incur a large overhead cost"
              (không gây chi phí lớn), nhưng các khối không có luồng nào làm việc "should be avoided" (nên tránh). HIP từ
              7.0: một chiều kích thước công việc toàn cục bằng không trả về <code>hipErrorInvalidValue</code>.
            </p>
          }
        >
          <p>
            The guide: launching extra threads in a block that do no work "does not incur a large overhead cost", but
            blocks in which no threads do work "should be avoided". HIP since 7.0: a global work size dimension of zero
            returns <code>hipErrorInvalidValue</code>.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Một số khối sai khó thấy vì hầu hết bài kiểm tra không thấy được nó. Trong trò săn lỗi, bạn đóng vai người viết
            bài kiểm tra: với mỗi lần khởi chạy sai trong ba lần, hãy nêu một độ dài che giấu lỗi và độ dài nhỏ nhất làm
            lộ ra nó.
          </p>
        }
      >
        <p>
          A wrong block count is hard to see because most tests cannot see it. In the bug hunt you play the test
          author: for each of the three wrong launches, name one length that hides the defect and the smallest length
          that shows it.
        </p>
      </Translated>
      <FindTheN />
      <Ask q={questionsS26['ch02/2.6/q-threads-8000']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Hàm host hoàn chỉnh">The complete host function</Translated></h2>
      <Translated
        vi={
          <p>
            Với lệnh khởi chạy đã vào chỗ, hàm host đã trọn vẹn. Mã này là lời giải tham chiếu của lab 2.1: mã host cuối
            cùng của sách với những thay đổi liệt kê bên dưới. <code>CHECK</code> là phép kiểm tra lỗi đã sửa của khung
            bên lề trong sách, dưới dạng macro in lỗi và khiến <code>vecAdd</code> trả về 1. Ba dòng được đánh dấu là lệnh
            khởi chạy và hai dòng theo sau nó.
          </p>
        }
      >
        <p>
          With the launch in place the host function is whole. The listing is the reference solution of lab 2.1: the
          book's final host code with the changes listed under it. <code>CHECK</code> is the corrected error check of
          the book's sidebar, as a macro that prints the error and makes <code>vecAdd</code> return 1. The three marked
          lines are the launch and the two lines that follow it.
        </p>
      </Translated>
      <Source section="2.6" pages="38" figs={tr('Fig. 2.13; the error check is the sidebar on p. 32', 'Fig. 2.13; phép kiểm tra lỗi là khung bên lề ở p. 32')} />
      {host ? (
        <Code
          tabs={[{ label: tr('CUDA and HIP', 'CUDA và HIP'), code: host.code, status: cuStatus }]}
          firstLine={host.firstLine}
          highlight={launchLine ? [launchLine, launchLine + 1, launchLine + 2] : []}
          caption={
            <Translated
              vi={
                <>
                  <code>labs/ch02/lab1_vecadd/solution/kernel.cu</code> không có chú thích mở đầu. Trong{' '}
                  <a href={href('ch02', 'labs')}>lab 2.1</a> bạn tự viết hàm này; hãy thử làm điều đó trước.
                </>
              }
            >
              <>
                <code>labs/ch02/lab1_vecadd/solution/kernel.cu</code> without its opening comment. In{' '}
                <a href={href('ch02', 'labs')}>lab 2.1</a> you write this function yourself; try that first.
              </>
            </Translated>
          }
        />
      ) : null}
      <Translated vi={<p>Những chỗ khác với trang sách:</p>}>
        <p>Where it differs from the page:</p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>
              Mọi lời gọi API đều được kiểm tra. Mã của sách không kiểm tra cái nào và có nói như vậy; khung bên lề của nó
              cho thấy mẫu.
            </li>
            <li>
              Lệnh khởi chạy được theo sau bởi <code>cudaGetLastError()</code> và bởi một{' '}
              <code>cudaDeviceSynchronize()</code> có kiểm tra. Không dòng nào trong hai dòng này có trong sách.
            </li>
            <li>
              Hàm trả về một trạng thái <code>int</code>. Hàm của sách trả về <code>void</code> và mẫu xử lý lỗi của nó
              gọi <code>exit</code>.
            </li>
            <li>
              Số khối là hàm trần số nguyên, không phải <code>ceil(n/256.0)</code>.
            </li>
            <li>Kích thước khối là một tham số. Sách viết cứng 256.</li>
            <li>
              Con trỏ host kết thúc bằng <code>_h</code> và con trỏ device bằng <code>_d</code> ở mọi nơi. Fig. 2.13 đặt
              tên các tham số host chỉ là <code>A</code>, <code>B</code>, <code>C</code>, trái với quy ước của chính sách.
            </li>
            <li>
              Các đầu vào là <code>const</code>, và số byte là một <code>size_t</code> ở chỗ sách dùng một{' '}
              <code>int</code>.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            Every API call is checked. The book's listing checks none and says so; its sidebar shows the pattern.
          </li>
          <li>
            The launch is followed by <code>cudaGetLastError()</code> and by a checked{' '}
            <code>cudaDeviceSynchronize()</code>. Neither line is in the book.
          </li>
          <li>
            The function returns an <code>int</code> status. The book's function returns <code>void</code> and its
            error pattern calls <code>exit</code>.
          </li>
          <li>
            The block count is the integer ceiling, not <code>ceil(n/256.0)</code>.
          </li>
          <li>The block size is a parameter. The book hard-wires 256.</li>
          <li>
            Host pointers end in <code>_h</code> and device pointers in <code>_d</code> throughout. Fig. 2.13 names its
            host parameters plain <code>A</code>, <code>B</code>, <code>C</code>, against the book's own convention.
          </li>
          <li>
            The inputs are <code>const</code>, and the byte count is a <code>size_t</code> where the book has an{' '}
            <code>int</code>.
          </li>
        </ul>
      </Translated>
      <Translated
        vi={
          <p>
            Hai dòng sau lệnh khởi chạy có mặt vì lệnh khởi chạy là câu lệnh duy nhất của hàm không có giá trị trả về để
            kiểm tra. <code>cudaGetLastError()</code> trả về trạng thái lỗi hiện tại của runtime và đặt lại nó. Thành công
            tại điểm đó chỉ có nghĩa là các tham số khởi chạy và cấu hình đã được chấp nhận: kernel có thể chưa bắt đầu.
            Một lỗi trong lúc kernel chạy được báo bởi lời gọi kế tiếp xem xét trạng thái lỗi, mà ở đây là{' '}
            <code>cudaDeviceSynchronize()</code> có kiểm tra. Lời gọi đó cũng chặn host cho đến khi lưới hoàn tất.
          </p>
        }
      >
        <p>
          The two lines after the launch are there because the launch is the one statement of the function with no
          return value to check. <code>cudaGetLastError()</code> returns the runtime's current error state and resets
          it. Success at that point means only that the launch parameters and the configuration were accepted: the
          kernel may not have started. A failure while the kernel runs is reported by the next call that looks at the
          error state, which here is the checked <code>cudaDeviceSynchronize()</code>. That call also blocks the host
          until the grid has finished.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, A6)', '2026-11-20 (notes/today/ch02-facts.md, A6)')}>
        <Translated
          vi={
            <p>
              Hướng dẫn: các lần khởi chạy bằng dấu ngoặc nhọn ba "do not return a cudaError_t" (không trả về
              cudaError_t), và kiểm tra trạng thái lỗi ngay sau đó là thực hành tốt. Mẫu của chính hướng dẫn là lệnh khởi
              chạy, rồi <code>cudaGetLastError()</code>, rồi <code>cudaDeviceSynchronize()</code>, mỗi cái nằm trong một
              macro kiểm tra. Trạng thái lỗi được giữ riêng cho từng luồng host.
            </p>
          }
        >
          <p>
            The guide: chevron launches "do not return a cudaError_t", and it is good practice to check the error state
            immediately after one. Its own pattern is the launch, then <code>cudaGetLastError()</code>, then{' '}
            <code>cudaDeviceSynchronize()</code>, each inside a check macro. The error state is kept per host thread.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Các dự án bạn muốn đọc coi đây là một quy tắc. Trong PyTorch,{' '}
            <code>C10_CUDA_KERNEL_LAUNCH_CHECK()</code> được định nghĩa là một phép kiểm tra <code>cudaGetLastError()</code>,
            và chú thích trên nó bảo hãy dùng nó ngay sau mọi lần khởi chạy kernel. llama.cpp đưa hầu hết các lần khởi chạy
            qua một hàm, <code>ggml_cuda_kernel_launch</code>, kết thúc bằng dấu ngoặc nhọn ba với số khối và kích thước
            khối mà nơi gọi đã tính.
          </p>
        }
      >
        <p>
          The projects you want to read treat this as a rule. In PyTorch,{' '}
          <code>C10_CUDA_KERNEL_LAUNCH_CHECK()</code> is defined as a check of <code>cudaGetLastError()</code>, and the
          comment on it says to use it directly after every kernel launch. llama.cpp sends most launches through one
          function, <code>ggml_cuda_kernel_launch</code>, which ends in the chevrons with the block counts and block
          sizes its caller computed.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, D4 and D1)', '2026-11-20 (notes/today/ch02-facts.md, D4 và D1)')}>
        <Translated
          vi={
            <p>
              Đọc trong <code>c10/cuda/CUDAException.h</code> trên nhánh main của PyTorch và trong{' '}
              <code>ggml/src/ggml-cuda/common.cuh</code> trên nhánh master của llama.cpp. Các lần khởi chạy của chính
              PyTorch dùng trực tiếp dấu ngoặc nhọn ba, với cả bốn tham số.
            </p>
          }
        >
          <p>
            Read in <code>c10/cuda/CUDAException.h</code> on PyTorch's main branch and in{' '}
            <code>ggml/src/ggml-cuda/common.cuh</code> on llama.cpp's master. PyTorch's own launches use the chevrons
            directly, with all four parameters.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS26['ch02/2.6/q-launch-check']} />

      <h2><Translated vi="Một lần khởi chạy, từ đầu đến cuối">One launch, start to finish</Translated></h2>
      <Translated
        vi={
          <p>
            Một lần khởi chạy là một dòng mã nguồn và một chuỗi sự kiện theo thời gian. Hoạt cảnh theo dõi một lần khởi
            chạy {SCENE.n} phần tử trong các khối {SCENE.B} luồng, trên một device chạy {SCENE.S} khối cùng lúc. Hãy chú ý
            ba điều: khoảnh khắc câu lệnh trả về, thứ tự mà C được điền, và, ở phần cuối, điều không bao giờ xảy ra.
          </p>
        }
      >
        <p>
          A launch is one line of source and a sequence of events in time. The animation follows a launch of{' '}
          {SCENE.n} elements in blocks of {SCENE.B}, on a device that runs {SCENE.S} blocks at once. Watch for three
          things: the moment the statement returns, the order in which C fills, and, in the last part, what never
          happens.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Device là một con số, số khối nó chạy cùng lúc, và nó chạy chúng theo các nhóm bắt đầu và kết thúc cùng nhau.
              Bên trong một khối, các luồng được vẽ hoàn tất theo thứ tự chỉ số. Không có gì trong đó được đo thời gian.
              Câu lệnh chờ được vẽ như một lời gọi đồng bộ hóa; một lệnh sao chép kiểu chặn cũng sẽ làm y như vậy.
            </p>
          }
        >
          <p>
            The device is one number, the blocks it runs at once, and it runs them in groups that start and finish
            together. Inside a block the threads are drawn finishing in index order. Nothing in it is timed. The
            statement that waits is drawn as a synchronise; a blocking copy would do the same.
          </p>
        </Translated>
      </Simplification>
      <OneLaunchScene />
      <Translated
        vi={
          <p>
            Phần 6 trả lời một câu hỏi mà mã của sách để ngỏ. Fig. 2.13 không có lời gọi đồng bộ hóa, mà lệnh sao chép
            của nó vẫn mang về một C hoàn chỉnh. Tài liệu giải thích điều này theo ba bước: một lần khởi chạy không nêu tên
            stream đi vào stream mặc định, các thao tác của một stream chạy theo thứ tự chúng được xếp hàng, và{' '}
            <code>cudaMemcpy</code> không trả về cho đến khi việc sao chép hoàn tất. Vì vậy lệnh sao chép chờ sau kernel.
            Bước cuối đó là suy luận của chúng tôi từ ba phát biểu; chúng tôi không tìm thấy một câu nào trong hướng dẫn
            nói điều đó.
          </p>
        }
      >
        <p>
          Part 6 answers a question the book's listing leaves open. Fig. 2.13 has no synchronise, and its copy still
          brings back a finished C. The documentation explains it in three steps: a launch with no stream named goes
          into the default stream, the operations of a stream run in the order they were queued, and{' '}
          <code>cudaMemcpy</code> does not return until its copy is complete. So the copy waits behind the kernel. That
          last step is our inference from the three statements; we did not find one sentence in the guide that says it.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, A5)', '2026-11-20 (notes/today/ch02-facts.md, A5)')}>
        <Translated
          vi={
            <p>
              Từ hướng dẫn, các mục 2.5.2, 2.5.2.3 và 2.5.5. Trong chương này sách không nói một lần khởi chạy có chặn hay
              không. <a href={href('ch02', '2.2')}>Phần 2.2</a> nói về những gì đã xảy ra khi câu lệnh trả về.
            </p>
          }
        >
          <p>
            From the guide, sections 2.5.2, 2.5.2.3 and 2.5.5. The book does not say in this chapter whether a launch
            blocks. <a href={href('ch02', '2.2')}>Section 2.2</a> is about what has happened when the statement returns.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Không có thứ tự giữa các khối">No order between blocks</Translated></h2>
      <Translated
        vi={
          <p>
            Trong hoạt cảnh, device nhận khối 2 trước khối 0. Nó được phép làm vậy. Quy tắc của sách là các khối của một
            lưới làm việc trên các phần dữ liệu khác nhau và có thể thực thi theo thứ tự bất kỳ, và lập trình viên không
            được giả định một thứ tự nào.
          </p>
        }
      >
        <p>
          In the animation the device took block 2 before block 0. It may. The book's rule is that the blocks of a grid
          work on different parts of the data and may execute in any order, and that a programmer must not assume one.
        </p>
      </Translated>
      <Source section="2.6" pages="39" />
      <Translated
        vi={
          <p>
            NVIDIA nêu điều đó như một yêu cầu đối với mã của bạn: phải có thể chạy các khối theo thứ tự bất kỳ, song song
            hoặc lần lượt, nên một khối không thể dựa vào kết quả từ một khối khác.
          </p>
        }
      >
        <p>
          NVIDIA states it as a requirement on your code: it must be possible to run the blocks in any order, in
          parallel or one after another, so a block cannot rely on a result from another block.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, A18)', '2026-11-20 (notes/today/ch02-facts.md, A18)')}>
        <Translated
          vi={
            <p>
              Hướng dẫn: "There is no guarantee of scheduling between thread blocks" (không có bảo đảm nào về lập lịch
              giữa các khối luồng), và mô hình lập trình "requires that it be possible to execute thread blocks in any
              order, in parallel or in series." (đòi hỏi phải có thể thực thi các khối luồng theo thứ tự bất kỳ, song song
              hoặc tuần tự.)
            </p>
          }
        >
          <p>
            The guide: "There is no guarantee of scheduling between thread blocks", and the programming model
            "requires that it be possible to execute thread blocks in any order, in parallel or in series."
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Quy tắc này mang lại một điều. Cùng một đoạn mã chạy trên một device thực thi ít khối mỗi lần và trên một device
            thực thi nhiều khối, chậm hơn trên cái thứ nhất và nhanh hơn trên cái thứ hai, mà không cần thay đổi mã nguồn.
            Ví dụ của sách là một GPU nhỏ chạy một hoặc hai khối mỗi lần và một GPU lớn chạy 128 hoặc 256. Sách gọi đây là
            khả năng mở rộng (scalability) và quay lại với nó ở <a href={href('ch04')}>chương 4</a>.
          </p>
        }
      >
        <p>
          The rule buys something. The same code runs on a device that executes few blocks at a time and on one that
          executes many, slower on the first and faster on the second, with no change to the source. The book's
          example is a small GPU that runs one or two blocks at a time and a large one that runs 128 or 256. It calls
          this scalability and comes back to it in <a href={href('ch04')}>chapter 4</a>.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, A18)', '2026-11-20 (notes/today/ch02-facts.md, A18)')}>
        <Translated
          vi={
            <p>
              Không nguồn nào đưa ra các con số của sách cho device cụ thể nào. Hãy đọc chúng là ít và nhiều: bao nhiêu khối
              chạy cùng nhau phụ thuộc vào device và vào kernel.
            </p>
          }
        >
          <p>
            No source gives the book's counts for any device. Read them as few and many: how many blocks run together
            depends on the device and on the kernel.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Trình khám phá chạy một lưới nhỏ trên một device nhận S khối mỗi lần, theo thứ tự bạn chọn. Hãy chạy kernel thứ
            nhất theo vài thứ tự và với vài giá trị S. Rồi chuyển sang kernel thứ hai, trong đó mỗi luồng đọc phần tử mà
            luồng láng giềng bên trái của nó ghi.
          </p>
        }
      >
        <p>
          The explorer runs a small grid on a device that takes S blocks at a time, in an order you choose. Run the
          first kernel in several orders and with several values of S. Then switch to the second kernel, in which each
          thread reads the element its left neighbour writes.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <>
              <p>
                Một device được rút gọn thành "S khối mỗi lần", và các khối của một đợt bắt đầu và kết thúc cùng nhau. Một
                device thật giao các khối cho các bộ xử lý của nó dưới nhiều giới hạn tài nguyên (chương 4), và bao nhiêu
                khối chạy cùng nhau phụ thuộc vào device và vào kernel. Đó là lý do không có giá trị S nào được hiển thị
                cho GPU của bạn.
              </p>
              <p>
                Bên trong một khối, mô hình chạy các luồng lần lượt theo thứ tự chỉ số, nên kernel thứ hai chỉ có thể sai
                ở chỗ một khối gặp khối kế tiếp. Trên một device, các luồng của một khối cũng không có thứ tự nào (sách,
                chương 4, p. 74).
              </p>
            </>
          }
        >
          <p>
            A device is reduced to "S blocks at a time", and the blocks of a wave start and finish together. A real
            device assigns blocks to its processors under several resource limits (chapter 4), and how many run
            together depends on the device and on the kernel. That is why no value of S is shown for your GPU.
          </p>
          <p>
            Inside a block the model runs the threads one after another in index order, so the second kernel can only
            go wrong where one block meets the next. On a device the threads of a block have no order either (book,
            chapter 4, p. 74).
          </p>
        </Translated>
      </Simplification>
      <AnyOrder />
      <Translated
        vi={
          <p>
            Với kernel thứ nhất, mọi thứ tự và mọi S cho cùng C, và chỉ số đợt thay đổi. Đó là tuyên bố về khả năng mở rộng
            thu nhỏ lại. Với kernel thứ hai, gần như mọi thiết lập cho một đáp án khác, kể cả thứ tự tăng dần ngay khi hai
            khối chạy cùng nhau. Thiết lập duy nhất khớp với vòng lặp chỉ khớp trong mô hình: trên một device, các luồng bên
            trong mỗi khối cũng chạy đua theo cùng cách. Một kernel như vậy không phải sai trên một số device. Nó sai, và
            vượt qua trên những device tình cờ chạy nó theo một thứ tự may mắn.
          </p>
        }
      >
        <p>
          For the first kernel every order and every S give the same C, and only the number of waves changes. That is
          the scalability claim in miniature. For the second, almost every setting gives a different answer, including
          ascending order as soon as two blocks run together. The one setting that matches the loop matches only in
          the model: on a device, the threads inside each block race in the same way. A kernel like that is not wrong
          on some devices. It is wrong, and passes on the devices that happen to run it in a lucky order.
        </p>
      </Translated>
      <Ask q={questionsS26['ch02/2.6/q-order']} />
      <Ask q={questionsS26['ch02/2.6/q-scale']} />

      <h2><Translated vi="Đếm luồng: bài tập 9 của sách">Count the threads: the book's exercise 9</Translated></h2>
      <Translated
        vi={
          <p>
            Bài tập thứ chín của sách là phần này dưới dạng năm phép đếm. Mã của nó khởi chạy một kernel trên{' '}
            {num(E9.N)} phần tử trong các khối {E9.threadsPerBlock} luồng, điều đó trả lời ý (a), số luồng mỗi khối. Hãy
            chốt bốn ý còn lại trước khi đọc tiếp. Số dòng của sách là số được hiển thị: dòng 2 tính chỉ số và dòng 4 là
            phép gán.
          </p>
        }
      >
        <p>
          The book's ninth exercise is this section as five counts. Its code launches a kernel over{' '}
          {num(E9.N)} elements in blocks of {E9.threadsPerBlock}, which answers part (a), the threads per block. Commit
          to the other four before you read on. The book's line numbers are the ones shown: line 2 computes the index
          and line 4 is the assignment.
        </p>
      </Translated>
      <Source section="2.9" pages="42" figs={tr('Fig. 2.15, exercise 9', 'Fig. 2.15, bài tập 9')} />
      <Code
        tabs={[{ label: 'CUDA', code: EX9, status: 'front-end-checked' }]}
        highlight={[2, 4, 9]}
        caption={
          <Translated
            vi={
              <>
                Fig. 2.15 của sách, với cách giãn dòng của chúng tôi. Được lưu tại <code>labs/ch02/snippets/ex9_foo.cu</code>.
              </>
            }
          >
            <>
              The book's Fig. 2.15, with our spacing. Stored as <code>labs/ch02/snippets/ex9_foo.cu</code>.
            </>
          </Translated>
        }
      />
      <Predict id="ch02/2.6/p-ex9-blocks" prompt={<Translated vi={<>Ý (c): lưới có bao nhiêu khối?</>}><>Part (c): how many blocks are in the grid?</></Translated>} unit={tr('blocks', 'khối')}>
        {(v) => (
          <Exact got={v} want={E9.blocks}>
            <Translated
              vi={
                <>
                  {E9_COUNT.steps[0].text} = {num(E9_COUNT.steps[0].value)}, và {E9_COUNT.steps[1].text} trong số nguyên là{' '}
                  {num(E9.blocks)}: thương chính xác là {showValue(E9_COUNT.steps[0].value / E9.threadsPerBlock, 'double')} và
                  phần thập phân bị bỏ.
                </>
              }
            >
              {E9_COUNT.steps[0].text} = {num(E9_COUNT.steps[0].value)}, and {E9_COUNT.steps[1].text} in integers is{' '}
              {num(E9.blocks)}: the exact quotient is {showValue(E9_COUNT.steps[0].value / E9.threadsPerBlock, 'double')} and
              the fraction is dropped.
            </Translated>
          </Exact>
        )}
      </Predict>
      <Predict id="ch02/2.6/p-ex9-threads" prompt={<Translated vi={<>Ý (b): lưới có bao nhiêu luồng?</>}><>Part (b): how many threads are in the grid?</></Translated>} unit={tr('threads', 'luồng')}>
        {(v) => (
          <Exact got={v} want={E9.threads}>
            <Translated
              vi={
                <>
                  {num(E9.blocks)} khối × {E9.threadsPerBlock} luồng = {num(E9.threads)}.
                  {v === E9.N ? ` Không phải ${num(E9.N)}: lưới được tạo từ các khối trọn vẹn.` : ''}
                </>
              }
            >
              {num(E9.blocks)} blocks × {E9.threadsPerBlock} threads = {num(E9.threads)}.
              {v === E9.N ? ` Not ${num(E9.N)}: the grid is made of whole blocks.` : ''}
            </Translated>
          </Exact>
        )}
      </Predict>
      <Predict id="ch02/2.6/p-ex9-line2" prompt={<Translated vi={<>Ý (d): bao nhiêu luồng thực thi dòng 2, phép tính chỉ số?</>}><>Part (d): how many threads execute line 2, the index computation?</></Translated>} unit={tr('threads', 'luồng')}>
        {(v) => (
          <Exact got={v} want={E9.line2}>
            <Translated vi={<>Mọi luồng của lưới đều tính <code>i</code> của riêng nó. Phép kiểm tra ở dòng 3 đến sau.</>}>
              Every thread of the grid computes its own <code>i</code>. The test on line 3 comes after.
            </Translated>
          </Exact>
        )}
      </Predict>
      <Predict id="ch02/2.6/p-ex9-line4" prompt={<Translated vi={<>Ý (e): bao nhiêu luồng thực thi dòng 4, phép gán?</>}><>Part (e): how many threads execute line 4, the assignment?</></Translated>} unit={tr('threads', 'luồng')}>
        {(v) => (
          <Exact got={v} want={E9.line4}>
            <Translated
              vi={
                <>
                  Chỉ những luồng có <code>i &lt; N</code>, mỗi phần tử một luồng. {num(E9.threads - E9.line4)} luồng còn
                  lại, tất cả trong khối cuối, tính một chỉ số rồi dừng.
                </>
              }
            >
              Only the threads with <code>i &lt; N</code>, one for each element. The other{' '}
              {num(E9.threads - E9.line4)}, all in the last block, compute an index and stop.
            </Translated>
          </Exact>
        )}
      </Predict>

      <h2><Translated vi="Đo thời gian một lần khởi chạy">Timing a launch</Translated></h2>
      <Translated
        vi={
          <p>
            Vì câu lệnh trả về trước khi lưới chạy, một bộ đếm thời gian bao quanh câu lệnh đo lần khởi chạy chứ không đo
            kernel. Ba quy tắc rút ra từ đó, và <a href={href('ch02', 'labs')}>lab 2.2</a> chính là các quy tắc này với số
            liệu của bạn. Gọi <code>cudaDeviceSynchronize()</code> trước khi đọc đồng hồ. Chạy lần khởi chạy một lần trước
            lần được đo, vì lời gọi đầu tiên vào runtime phải trả cho công việc khởi động mà các lời gọi sau không phải trả.
            Và để dành biến môi trường <code>CUDA_LAUNCH_BLOCKING=1</code>, biến đồng bộ hóa sau mọi lần khởi chạy, cho
            việc gỡ lỗi.
          </p>
        }
      >
        <p>
          Because the statement returns before the grid has run, a timer around the statement measures the launch and
          not the kernel. Three rules follow, and <a href={href('ch02', 'labs')}>lab 2.2</a> is these rules with your
          numbers. Call <code>cudaDeviceSynchronize()</code> before you read the clock. Run the launch once before the
          timed one, because the first call into the runtime pays for start-up work that later calls do not. And keep
          the environment variable <code>CUDA_LAUNCH_BLOCKING=1</code>, which synchronises after every launch, for
          debugging.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không đo thời gian gì trong chương này. Các quy tắc này lấy từ hướng dẫn của NVIDIA
              (notes/today/ch02-facts.md, A5), mà phương pháp riêng để đo một kernel là một cặp event được ghi trong stream.
            </p>
          }
        >
          <p>
            The book times nothing in this chapter. These rules are from NVIDIA's guide (notes/today/ch02-facts.md, A5),
            whose own method for timing a kernel is a pair of events recorded in the stream.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            HIP giống vậy với <code>hipDeviceSynchronize()</code>. Một khác biệt quan trọng khi bạn đo toàn bộ hàm host:{' '}
            <code>hipFree</code> luôn đồng bộ hóa trước, nên một bộ đếm bao quanh nó gồm cả thời gian chờ công việc còn
            đang chạy. Metal không cần bộ đếm phía host cho việc này: sau khi chờ, command buffer báo GPU đã bắt đầu và kết
            thúc thực thi nó lúc nào.
          </p>
        }
      >
        <p>
          HIP is the same with <code>hipDeviceSynchronize()</code>. One difference matters when you time the whole host
          function: <code>hipFree</code> always synchronises first, so a timer around it includes the wait for work
          still running. Metal needs no host timer for this: after the wait, the command buffer reports when the GPU
          started and finished executing it.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, A3, B1 and B3; ch05-facts.md, C9)', '2026-11-20 (notes/today/ch02-facts.md, A3, B1 và B3; ch05-facts.md, C9)')}>
        <Translated
          vi={
            <p>
              <code>hipFree</code>: "performs an implicit hipDeviceSynchronize() call" (thực hiện một lời gọi
              hipDeviceSynchronize() ngầm). <code>cudaFree</code> "may perform implicit synchronization" (có thể đồng bộ
              hóa ngầm). Metal: <code>GPUStartTime</code> và <code>GPUEndTime</code>, thời gian phía host tính bằng giây,
              bằng không cho đến khi command buffer hoàn tất, và tính cho từng command buffer, không phải từng dispatch.
            </p>
          }
        >
          <p>
            <code>hipFree</code>: "performs an implicit hipDeviceSynchronize() call". <code>cudaFree</code> "may perform
            implicit synchronization". Metal: <code>GPUStartTime</code> and <code>GPUEndTime</code>, host times in
            seconds, zero until the command buffer has finished, and per command buffer, not per dispatch.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS26['ch02/2.6/q-timer']} />

      <h2><Translated vi="Lần khởi chạy trên các backend khác">The launch on other backends</Translated></h2>
      <Translated
        vi={
          <p>
            Trên HIP, lần khởi chạy là cùng câu lệnh đó. Trình biên dịch của AMD chấp nhận dấu ngoặc nhọn ba với cùng bốn
            tham số, và tệp <code>.cu</code> của lab được build cho HIP qua một header đổi tên các lời gọi API. HIP ghi
            trong tài liệu hai lỗi khởi chạy: một chiều lưới bằng không trả về <code>hipErrorInvalidValue</code>, và số
            luồng mỗi khối nhiều hơn mức device cho phép trả về <code>hipErrorInvalidConfiguration</code>. Bạn hỏi về
            chúng theo cùng cách, bằng <code>hipGetLastError()</code> sau lệnh khởi chạy.
          </p>
        }
      >
        <p>
          On HIP the launch is the same statement. AMD's compiler accepts the chevrons with the same four parameters,
          and the lab's <code>.cu</code> file is built for HIP through a header that renames the API calls. HIP
          documents two launch errors: a grid dimension of zero returns <code>hipErrorInvalidValue</code>, and more
          threads per block than the device allows returns <code>hipErrorInvalidConfiguration</code>. You ask for them
          the same way, with <code>hipGetLastError()</code> after the launch.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. Phần này là của khóa học này, từ tài liệu của nhà cung cấp (notes/today/ch02-facts.md, B2
              và B4; ch05-facts.md, B3, kiểm tra ngày 2026-11-20). Bản build HIP của các lab chưa được biên dịch trên phần
              cứng AMD.
            </p>
          }
        >
          <p>
            The book uses CUDA only. This part is ours, from vendor documentation (notes/today/ch02-facts.md, B2 and B4;
            ch05-facts.md, B3, checked 2026-11-20). The HIP build of the labs has not been compiled on AMD hardware.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Trên Metal, một câu lệnh trở thành một chuỗi lời gọi trên các đối tượng. Không có gì trong đó mới về mặt khái
            niệm: mỗi phần của câu lệnh CUDA có một thứ tương ứng, và hai việc mà CUDA làm phía sau câu lệnh được viết ra
            tường minh. Bảng liệt kê chúng theo thứ tự, cùng các dòng trong hàm host Metal của lab 2.1 nơi mỗi việc xảy ra.
          </p>
        }
      >
        <p>
          On Metal the one statement becomes a sequence of calls on objects. Nothing in it is conceptually new: each
          part of the CUDA statement has a counterpart, and two things that CUDA does behind the statement are written
          out. The table lists them in order, with the lines of lab 2.1's Metal host function where each happens.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, C4, C5 and C8)', '2026-11-20 (notes/today/ch02-facts.md, C4, C5 và C8)')}>
        <Translated
          vi={
            <p>
              macOS 26 thêm một bộ kiểu lệnh thứ hai, song song, với tiền tố <code>MTL4</code>. Các lời gọi hiển thị ở đây
              là bộ gốc. Chúng vẫn hợp lệ, và là thứ mà mẫu của Apple, llama.cpp, MLX và PyTorch dùng trong các tệp được
              đọc cho khóa học này.
            </p>
          }
        >
          <p>
            macOS 26 added a second, parallel set of command types with an <code>MTL4</code> prefix. The calls shown
            here are the original ones. They remain valid, and they are what Apple's sample, llama.cpp, MLX and PyTorch
            use in the files read for this course.
          </p>
        </Translated>
      </Today>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-metal-steps>
        <thead>
          <tr>
            <th><Translated vi="Trong câu lệnh CUDA">In the CUDA statement</Translated></th>
            <th><Translated vi="Trong Metal">In Metal</Translated></th>
            <th className="num"><Translated vi="Các dòng bên dưới">Lines below</Translated></th>
          </tr>
        </thead>
        <tbody>
          {METAL_STEPS.map((s, i) => {
            const at = mmLine(s.needles);
            return (
              <tr key={i}>
                <td>{s.cuda}</td>
                <td>{s.metal}</td>
                <td className="num" data-lines={at.join(',')}>
                  {spanText(at)}
                  {s.note ? `, ${s.note}` : ''}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {mmPart ? (
        <Code
          tabs={[{ label: tr('Metal host', 'Host Metal'), code: mmPart.code, status: mmStatus }]}
          firstLine={mmPart.firstLine}
          highlight={mmHighlight}
          caption={
            <Translated
              vi={
                <>
                  <code>vecAdd</code> từ <code>labs/ch02/lab1_vecadd/solution/launch.mm</code>. Các dòng được đánh dấu là
                  các hàng của bảng.
                </>
              }
            >
              <>
                <code>vecAdd</code> from <code>labs/ch02/lab1_vecadd/solution/launch.mm</code>. The marked lines are the
                rows of the table.
              </>
            </Translated>
          }
        />
      ) : null}
      <Translated
        vi={
          <p>
            Một dòng của hàm CUDA hoàn toàn không có thứ tương ứng: số khối. <code>dispatchThreads</code> nhận số luồng, và
            Metal làm threadgroup cuối nhỏ hơn khi n không phải bội số của kích thước nhóm. Vì vậy không có hàm trần nào
            phải tính và không có phép kiểm tra biên trong kernel. Lời gọi dispatch còn lại,{' '}
            <code>dispatchThreadgroups</code>, nhận một số nhóm. Với nó, lưới bằng số nhóm nhân kích thước nhóm, và phép
            chia lấy trần cùng phép kiểm tra biên quay lại y như trong CUDA. Backend Metal của llama.cpp dispatch các
            threadgroup trọn vẹn, nên bạn sẽ đọc thấy cả hai kiểu.
          </p>
        }
      >
        <p>
          One line of the CUDA function has no counterpart at all: the block count. <code>dispatchThreads</code> takes
          the number of threads, and Metal makes the last threadgroup smaller when n is not a multiple of the group
          size. So there is no ceiling to compute and no guard in the kernel. The other dispatch call,{' '}
          <code>dispatchThreadgroups</code>, takes a number of groups. With it the grid is groups times group size, and
          the ceiling division and the guard come back exactly as in CUDA. llama.cpp's Metal backend dispatches whole
          threadgroups, so you will read both styles.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch05-facts.md, C5; notes/today/ch02-facts.md, D2)', '2026-11-20 (ch05-facts.md, C5; và notes/today/ch02-facts.md, D2)')}>
        <Translated
          vi={
            <p>
              Apple: với <code>dispatchThreads</code>, "Metal calculates the number of threadgroups needed, providing
              partial threadgroups if necessary" (Metal tính số threadgroup cần thiết, tạo threadgroup không đầy đủ nếu
              cần). Với <code>dispatchThreadgroups</code>, "perform boundary checks in your compute function" (hãy kiểm
              tra biên trong hàm tính toán của bạn). <code>dispatchThreads</code> cần một GPU hỗ trợ kích thước threadgroup
              không đồng đều, Apple GPU family Apple4 trở về sau.
            </p>
          }
        >
          <p>
            Apple: with <code>dispatchThreads</code>, "Metal calculates the number of threadgroups needed, providing
            partial threadgroups if necessary". With <code>dispatchThreadgroups</code>, "perform boundary checks in your
            compute function". <code>dispatchThreads</code> needs a GPU that supports nonuniform threadgroup sizes,
            Apple GPU family Apple4 and later.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Giờ bạn có thể khởi chạy một kernel cho độ dài bất kỳ và nói được mọi luồng của lưới làm gì. Thứ biến tệp mã
            nguồn thành thứ mà GPU có thể chạy là phần tiếp theo, ở <a href={href('ch02', '2.7')}>phần 2.7</a>. Còn chính
            lệnh khởi chạy là của bạn để viết trong <a href={href('ch02', 'labs')}>lab 2.1</a>.
          </p>
        }
      >
        <p>
          You can now launch a kernel for any length and say what every thread of the grid does. What turns the source
          file into something a GPU can run is next, in <a href={href('ch02', '2.7')}>section 2.7</a>. The launch
          itself is yours to write in <a href={href('ch02', 'labs')}>lab 2.1</a>.
        </p>
      </Translated>
    </>
  );
}
