import { useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Simplification, Source } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { CORRECT, makeProblem, naiveBlockReads, phases, runBlock, traffic, type Config, type Step } from '../../sim/tiled';
import { WhyTilingScene } from './WhyTilingScene';

// ---------------------------------------------------------------- drawing
const sum = (a: number[]) => a.reduce((x, y) => x + y, 0);
const plural = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;

/** Cell geometry shared by M, N and P, so rows and columns line up, with a wider gap between tiles. */
function cellBox(size: number, r: number, c: number, tile: number): CSSProperties {
  return {
    width: size,
    height: size,
    marginLeft: c > 0 && c % tile === 0 ? 5 : 0,
    marginTop: r > 0 && r % tile === 0 ? 5 : 0,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 0,
    borderRadius: 2,
    border: '1px solid var(--rule)', // i18n-ignore: CSS value
    background: 'var(--cell)',
    color: 'var(--ink)',
    fontSize: size >= 40 ? '0.76rem' : '0.72rem',
    fontWeight: 450,
    lineHeight: 1.15,
    fontVariantNumeric: 'tabular-nums',
  };
}
const READ: CSSProperties = { background: 'var(--global-wash)', border: '1px solid var(--global)' }; // i18n-ignore: CSS value
/** A caption above a group of buttons (a <label> would forward clicks on the caption to the first button). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };

/** An element name with its subscripts, e.g. M 0,3. */
function El({ a, r, c }: { a: string; r: number; c: number }) {
  return (
    <span className="nowrap">
      {a}
      <sub style={{ fontSize: '0.86em', verticalAlign: '-0.15em', lineHeight: 0 }}>
        {r},{c}
      </sub>
    </span>
  );
}

function Matrix({ name, n, note, children }: { name: string; n: number; note?: string; children: ReactNode }) {
  return (
    <figure style={{ margin: 0 }}>
      <figcaption style={{ marginBottom: 4 }}>
        <b className="t-global">{name}</b>
        {note ? <span className="muted"> {note}</span> : null}
      </figcaption>
      <div
        role="group"
        aria-label={tr(`matrix ${name}`, `ma trận ${name}`)}
        style={{ display: 'grid', gridTemplateColumns: `repeat(${n}, max-content)`, gap: 2, justifyContent: 'start' }}
      >
        {children}
      </div>
    </figure>
  );
}

const cells = (n: number) => Array.from({ length: n * n }, (_, i) => ({ i, r: Math.floor(i / n), c: i % n }));

// ---------------------------------------------------------------- interactive 1: who reads what
const W_CHOICES = [4, 6, 8];
const B_CHOICES = [2, 3, 4];

function WhoReadsWhat() {
  const [w, setW] = useState(4);
  const [b, setB] = useState(2);
  const [sel, setSel] = useState({ r: 0, c: 0 });
  const [view, setView] = useState<'thread' | 'block'>('thread');
  const pickW = (nw: number) => {
    setW(nw);
    if (nw % b !== 0) setB(B_CHOICES.find((x) => nw % x === 0)!);
    setSel((s) => ({ r: Math.min(s.r, nw - 1), c: Math.min(s.c, nw - 1) }));
  };
  const bx = Math.floor(sel.c / b);
  const by = Math.floor(sel.r / b);
  // One thread is a block of size 1 at its own position: the same count, no second code path.
  const reads = useMemo(
    () => (view === 'thread' ? naiveBlockReads(w, 1, sel.c, sel.r) : naiveBlockReads(w, b, bx, by)),
    [view, w, b, sel.r, sel.c, bx, by],
  );
  const loads = sum(reads.M) + sum(reads.N);
  const distinct = reads.M.filter((x) => x > 0).length + reads.N.filter((x) => x > 0).length;
  const ratio = loads / distinct;
  const size = w <= 4 ? 46 : w <= 6 ? 40 : 36;

  const input = (a: 'M' | 'N', counts: number[]) =>
    cells(w).map(({ i, r, c }) => (
      <div
        key={i}
        data-cell={`${a}-${r}-${c}`}
        data-count={counts[i]}
        style={{ ...cellBox(size, r, c, b), ...(counts[i] ? READ : null) }}
      >
        <El a={a} r={r} c={c} />
        {view === 'block' && counts[i] ? (
          <b className="t-global" style={{ marginTop: 2 }}>
            ×{counts[i]}
          </b>
        ) : null}
      </div>
    ));

  return (
    <Bench
      title={tr('Who reads what', 'Ai đọc gì')}
      task={tr(
        'Click an element of P. Then switch to its whole block and read off how often each input element is fetched.',
        'Bấm vào một phần tử của P. Sau đó chuyển sang cả khối của nó và đọc xem mỗi phần tử đầu vào được nạp bao nhiêu lần.',
      )}
      foot={tr(
        'Counts are loads the naive kernel requests from global memory: one element of M and one of N per loop iteration, per thread.',
        'Các con số là số lần nạp mà kernel ngây thơ yêu cầu từ bộ nhớ toàn cục (global memory): một phần tử của M và một của N cho mỗi vòng lặp, mỗi luồng.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span style={FIELD}>
          <Translated vi={<>Độ rộng W = {w}</>}>Width W = {w}</Translated>
          <span className="seg" role="group" aria-label={tr('Width W', 'Độ rộng W')}>
            {W_CHOICES.map((x) => (
              <button key={x} aria-pressed={w === x} onClick={() => pickW(x)}>
                {x}
              </button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          <Translated vi={<>Kích thước khối = {b} × {b}</>}>Block size = {b} × {b}</Translated>
          <span className="seg" role="group" aria-label={tr('Block size', 'Kích thước khối')}>
            {B_CHOICES.map((x) => (
              <button key={x} aria-pressed={b === x} disabled={w % x !== 0} onClick={() => setB(x)}>
                {x}
              </button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          <Translated vi="Hiện các lần đọc của">Show the reads of</Translated>
          <span className="seg" role="group" aria-label={tr('Show the reads of', 'Hiện các lần đọc của')}>
            <button aria-pressed={view === 'thread'} onClick={() => setView('thread')}>
              <Translated vi="một luồng">one thread</Translated>
            </button>
            <button aria-pressed={view === 'block'} onClick={() => setView('block')}>
              <Translated vi="cả khối của nó">its whole block</Translated>
            </button>
          </span>
        </span>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.2rem', alignItems: 'flex-start' }}>
        <div style={{ overflowX: 'auto', maxWidth: '100%', paddingBottom: 2 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'max-content max-content', gap: 14 }}>
            <div style={{ alignSelf: 'end', maxWidth: size * w, color: 'var(--ink-2)', fontSize: '0.78rem' }}>
              <span className="legend">
                <span>
                  <i style={{ background: 'var(--global-wash)', border: '1px solid var(--global)' }} />
                  <Translated vi="nạp từ bộ nhớ toàn cục">loaded from global memory</Translated>
                </span>
              </span>
            </div>
            <Matrix name="N" n={w}>
              {input('N', reads.N)}
            </Matrix>
            <Matrix name="M" n={w}>
              {input('M', reads.M)}
            </Matrix>
            <Matrix name="P" n={w} note={tr('(click an element)', '(bấm vào một phần tử)')}>
              {cells(w).map(({ i, r, c }) => {
                const inBlock = Math.floor(r / b) === by && Math.floor(c / b) === bx;
                const on = view === 'thread' ? r === sel.r && c === sel.c : inBlock;
                return (
                  <button
                    key={i}
                    aria-label={`P[${r}][${c}]`}
                    aria-pressed={on}
                    onClick={() => setSel({ r, c })}
                    style={{
                      ...cellBox(size, r, c, b),
                      cursor: 'pointer',
                      ...(on ? { background: 'var(--global-wash)', border: '2px solid var(--ink)' } : null),
                    }}
                  >
                    <El a="P" r={r} c={c} />
                  </button>
                );
              })}
            </Matrix>
          </div>
        </div>
        <div className="panel" style={{ flex: '1 1 15rem', minWidth: '14rem', maxWidth: '30rem' }} aria-live="polite">
          {view === 'thread' ? (
            <p style={{ margin: 0 }} data-readout="who">
              <Translated
                vi={
                  <>
                    <b>
                      Luồng cho P[{sel.r}][{sel.c}]
                    </b>{' '}
                    đọc hàng {sel.r} của M và cột {sel.c} của N.
                  </>
                }
              >
                <b>
                  The thread for P[{sel.r}][{sel.c}]
                </b>{' '}
                reads row {sel.r} of M and column {sel.c} of N.
              </Translated>
            </p>
          ) : (
            <p style={{ margin: 0 }} data-readout="who">
              <Translated
                vi={
                  <>
                    <b>
                      Khối (x {bx}, y {by})
                    </b>
                    : {b * b} luồng cho các hàng {by * b} đến {by * b + b - 1} và các cột {bx * b} đến {bx * b + b - 1}{' '}
                    của P, mỗi luồng chạy kernel ngây thơ.
                  </>
                }
              >
                <b>
                  Block (x {bx}, y {by})
                </b>
                : the {b * b} threads for rows {by * b} to {by * b + b - 1} and columns {bx * b} to {bx * b + b - 1} of P,
                each running the naive kernel.
              </Translated>
            </p>
          )}
          <table className="data" style={{ margin: '0.6rem 0' }}>
            <tbody>
              <tr>
                <td>
                  <Translated vi="Số lần nạp từ bộ nhớ toàn cục">Loads from global memory</Translated>
                </td>
                <td className="num" data-total="loads">
                  {loads}
                </td>
              </tr>
              <tr>
                <td>
                  <Translated vi="Số phần tử khác nhau cần dùng">Distinct elements needed</Translated>
                </td>
                <td className="num" data-total="distinct">
                  {distinct}
                </td>
              </tr>
              <tr>
                <td>
                  <Translated vi="Số lần nạp mỗi phần tử">Loads per element</Translated>
                </td>
                <td className="num" data-total="ratio">
                  <b>{Number.isInteger(ratio) ? ratio : ratio.toFixed(2)}</b>
                </td>
              </tr>
            </tbody>
          </table>
          {view === 'thread' ? (
            <p className="muted" style={{ margin: 0 }}>
              <Translated
                vi={
                  <>
                    Riêng một luồng thì không lãng phí gì: nó cần mỗi phần tử này một lần và nạp mỗi phần tử một lần. Sự lãng phí chỉ
                    lộ ra khi bạn nhìn sang các luồng lân cận. Hãy chuyển sang cả khối.
                  </>
                }
              >
                On its own a thread wastes nothing: it needs each of these elements once and loads each once. The waste
                only shows when you look at its neighbours. Switch to the whole block.
              </Translated>
            </p>
          ) : (
            <p className="muted" style={{ margin: 0 }}>
              <Translated
                vi={
                  <>
                    Con số trên một phần tử là số luồng của khối này nạp nó. Một khối nạp mỗi phần tử một lần rồi chia sẻ sẽ phát ra{' '}
                    {distinct} lần nạp thay vì {loads}.
                  </>
                }
              >
                The number on an element is how many threads of this block load it. A block that loaded each element once
                and shared it would issue {distinct} loads instead of {loads}.
              </Translated>
            </p>
          )}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive 2: phases
const PHASE_CASES = [
  { w: 4, t: 2 },
  { w: 6, t: 2 },
  { w: 6, t: 3 },
];
const isLoad = (s: Step) => s.op === 'loadM' || s.op === 'loadN';

function PhasesBench() {
  const [ci, setCi] = useState(0);
  const [ph, setPh] = useState(0);
  const { w, t } = PHASE_CASES[ci];
  const { prob, nPh, steps } = useMemo(() => {
    const cfg: Config = { ...CORRECT, width: w, tile: t };
    const prob = makeProblem(w);
    return { prob, nPh: phases(cfg), steps: runBlock(cfg, prob, 0, 0, 'lockstep').steps };
  }, [w, t]);

  // Everything shown is read off the simulator's trace of block (0,0).
  const loads = steps.filter((s) => isLoad(s) && s.ph === ph);
  const tile = (a: 'M' | 'N') => new Set(loads.filter((s) => s.global!.array === a).map((s) => s.global!.index));
  const threads = Array.from({ length: t * t }, (_, i) => {
    const mine = steps.filter((s) => s.thread === i);
    const muls = mine.filter((s) => s.op === 'mul' && s.ph === ph);
    const earlier = mine.filter((s) => s.op === 'mul' && s.ph < ph);
    return {
      m: mine.find((s) => s.op === 'loadM' && s.ph === ph)!,
      n: mine.find((s) => s.op === 'loadN' && s.ph === ph)!,
      muls,
      before: earlier.length ? earlier[earlier.length - 1].pvalue! : 0,
      after: muls[muls.length - 1].pvalue!,
      store: mine.find((s) => s.op === 'store')!,
    };
  });
  const writes = (a: 'Mds' | 'Nds', r: number, c: number) =>
    steps.filter((s) => s.write && s.write.array === a && s.write.r === r && s.write.c === c && s.ph <= ph);
  const last = ph === nPh - 1;
  const gsize = w <= 4 ? 40 : 34;

  const global = (a: 'M' | 'N', data: number[]) => {
    const on = tile(a);
    return (
      <Matrix name={a} n={w}>
        {cells(w).map(({ i, r, c }) => (
          <div
            key={i}
            data-cell={`${a}-${r}-${c}`}
            data-in-tile={on.has(i) ? 1 : 0}
            style={{ ...cellBox(gsize, r, c, t), ...(on.has(i) ? READ : { color: 'var(--ink-2)' }) }}
          >
            <span style={{ fontSize: '0.86em' }} className="muted">
              {r},{c}
            </span>
            <b>{data[i]}</b>
          </div>
        ))}
      </Matrix>
    );
  };

  const shared = (a: 'Mds' | 'Nds') => (
    <figure style={{ margin: 0 }}>
      <figcaption style={{ marginBottom: 4 }}>
        <b className="t-shared">{a}</b>
      </figcaption>
      <div
        role="group"
        aria-label={tr(`shared tile ${a}`, `ô chia sẻ ${a}`)}
        style={{ display: 'grid', gridTemplateColumns: `repeat(${t}, max-content)`, gap: 2 }}
      >
        {cells(t).map(({ i, r, c }) => {
          const ws = writes(a, r, c);
          const now = ws[ws.length - 1];
          const g = now.global!;
          return (
            <div
              key={i}
              data-slot={`${a}-${r}-${c}`}
              data-value={now.write!.value}
              data-writes={ws.length}
              style={{
                width: 88,
                height: 76,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                border: '1px solid var(--shared)',
                background: 'var(--shared-wash)',
                borderRadius: 2,
                fontSize: '0.7rem',
                lineHeight: 1.2,
                fontVariantNumeric: 'tabular-nums',
              }}
            >
              <span className="muted">
                {a}[{r}][{c}]
              </span>
              <b className="t-shared" style={{ fontSize: '1.15rem', lineHeight: 1.15 }}>
                {now.write!.value}
              </b>
              <span className="t-global">
                <Translated vi={<>từ {g.array}[{g.row}][{g.col}]</>}>
                  from {g.array}[{g.row}][{g.col}]
                </Translated>
              </span>
              <span className="muted" style={{ minHeight: '1.2em' }}>
                {ws.length > 1 ? (
                  <>
                        <Translated vi="trước là">was</Translated>{' '}
                    {ws.slice(0, -1).map((s, k) => (
                      <s key={k} style={{ marginRight: 3 }}>
                        {s.write!.value}
                      </s>
                    ))}
                  </>
                ) : null}
              </span>
            </div>
          );
        })}
      </div>
    </figure>
  );

  return (
    <Bench
      title={tr('Phases', 'Các pha')}
      task={tr(
        "Step through the phases of block (0,0). Watch where each thread's two loads go, and what happens to the shared slots.",
        'Đi qua từng pha của khối (0,0). Theo dõi hai lần nạp của mỗi luồng đi đâu, và chuyện gì xảy ra với các vị trí trong bộ nhớ chia sẻ.',
      )}
      foot={
        <Translated
          vi={
            <>
              Đầu vào là các số nguyên nhỏ, nên mọi tổng đều chính xác. Cả {t * t} luồng được vẽ cùng di chuyển; điều giữ
              cho các luồng thật đi cùng nhịp là chủ đề của phần 5.4.
            </>
          }
        >
          <>
            Inputs are small integers, so every sum is exact. All {t * t} threads are shown moving together; what keeps
            real threads in step is the subject of section 5.4.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Bài toán">Problem</Translated>
          <span className="seg" role="group" aria-label={tr('Problem', 'Bài toán')}>
            {PHASE_CASES.map((p, i) => (
              <button
                key={i}
                aria-pressed={ci === i}
                onClick={() => {
                  setCi(i);
                  setPh(0);
                }}
              >
                <Translated vi={<>{p.w}×{p.w}, ô {p.t}</>}>
                  {p.w}×{p.w}, tile {p.t}
                </Translated>
              </button>
            ))}
          </span>
        </span>
        <button onClick={() => setPh(ph - 1)} disabled={ph === 0}>
          <Translated vi="Pha trước">Previous phase</Translated>
        </button>
        <button className="primary" onClick={() => setPh(ph + 1)} disabled={last}>
          <Translated vi="Pha tiếp">Next phase</Translated>
        </button>
        <span data-readout="phase" aria-live="polite" style={{ paddingBottom: '0.36rem' }}>
          <Translated
            vi={
              <>
                <b>Pha {ph}</b> trên {nPh} (đánh số từ 0 đến {nPh - 1}){last ? ': pha cuối cùng' : ''}
              </>
            }
          >
            <b>Phase {ph}</b> of {nPh} (numbered 0 to {nPh - 1}){last ? ': the last one' : ''}
          </Translated>
        </span>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.2rem', alignItems: 'stretch', marginBottom: '1.2rem' }}>
        <div className="panel" style={{ flex: '0 1 auto', maxWidth: '100%', overflowX: 'auto' }}>
          <p style={{ margin: '0 0 0.6rem' }}>
            <Translated
              vi={
                <>
                  <b>Bộ nhớ toàn cục.</b> Hai ô mà pha này nạp.
                </>
              }
            >
              <b>Global memory.</b> The two tiles this phase loads.
            </Translated>
          </p>
          <div style={{ display: 'flex', gap: '1.2rem' }}>
            {global('M', prob.M)}
            {global('N', prob.N)}
          </div>
        </div>
        <div className="panel" style={{ flex: '0 1 auto', maxWidth: '100%', overflowX: 'auto' }}>
          <p style={{ margin: '0 0 0.6rem' }}>
            <Translated
              vi={
                <>
                  <b>Bộ nhớ chia sẻ của khối (0,0).</b> Vẫn {2 * t * t} vị trí đó trong mọi pha.
                </>
              }
            >
              <b>Shared memory of block (0,0).</b> The same {plural(2 * t * t, 'slot')} in every phase.
            </Translated>
          </p>
          <div style={{ display: 'flex', gap: '1.2rem' }}>
            {shared('Mds')}
            {shared('Nds')}
          </div>
          <p className="muted" style={{ margin: '0.6rem 0 0' }} data-readout="overwrites">
            {ph === 0
              ? tr('Each slot has been written once.', 'Mỗi vị trí đã được ghi một lần.')
              : tr(
                  `Each slot has now been written ${ph + 1} times. The crossed-out values are what earlier phases left; they are gone.`,
                  `Giờ mỗi vị trí đã được ghi ${ph + 1} lần. Các giá trị bị gạch là những gì các pha trước để lại; chúng đã mất.`,
                )}
          </p>
        </div>
      </div>
      <div className="panel" style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }}>
          <thead>
            <tr>
              <th>
                <Translated vi="Luồng">Thread</Translated>
              </th>
              <th>
                <Translated vi="Nạp từ M">Loads from M</Translated>
              </th>
              <th>
                <Translated vi="Nạp từ N">Loads from N</Translated>
              </th>
              <th>
                <Translated vi={<>Cộng trong pha {ph}</>}>Adds in phase {ph}</Translated>
              </th>
              <th className="num">
                <Translated vi="Tổng đang cộng dồn">Running sum</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            {threads.map((th) => {
              const slotOf = (s: Step) => `${s.write!.array}[${s.write!.r}][${s.write!.c}]`;
              const elOf = (s: Step) => `${s.global!.array}[${s.global!.row}][${s.global!.col}]`;
              return (
                <tr key={th.m.thread} data-thread={`${th.m.tx}-${th.m.ty}`}>
                  <td className="nowrap">
                    tx {th.m.tx}, ty {th.m.ty}{/* i18n-ignore: thread index names */}
                  </td>
                  <td className="nowrap" data-col="m">
                    <span className="t-global">{elOf(th.m)}</span> → <span className="t-shared">{slotOf(th.m)}</span>
                  </td>
                  <td className="nowrap" data-col="n">
                    <span className="t-global">{elOf(th.n)}</span> → <span className="t-shared">{slotOf(th.n)}</span>
                  </td>
                  <td className="nowrap" data-col="adds">
                    {th.muls.map((s) => `${s.reads![0].value}×${s.reads![1].value}`).join(' + ')}
                  </td>
                  <td className="num nowrap" data-col="sum">
                    {th.before} + {th.after - th.before} = <span className="chip reg">{th.after}</span>
                    {last ? (
                      <>
                        {' '}
                        = <span className="t-global">{elOf(th.store)}</span>
                      </>
                    ) : null}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
// Numbers come from the simulator's traffic model, not from arithmetic done by hand.
const NAIVE_96 = traffic(96, null);
const TILED_96 = traffic(96, 16);
const NAIVE_8 = traffic(8, null);
const TILED_8 = traffic(8, 4);

export const questionsS53: Record<string, Question> = {
  'ch05/5.3/q-requests': {
    kind: 'numeric',
    id: 'ch05/5.3/q-requests',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Nhân hai ma trận 96×96, trước bằng kernel ngây thơ rồi bằng một kernel phân ô với các ô 16×16. Theo dõi một phần
              tử cụ thể của M suốt cả phép nhân. Tính xem kernel ngây thơ yêu cầu nó từ bộ nhớ toàn cục bao nhiêu lần; rồi trả
              lời: kernel phân ô yêu cầu nó bao nhiêu lần?
            </>
          }
        >
          Two 96×96 matrices are multiplied, first by the naive kernel and then by a tiled kernel with 16×16 tiles. Follow
          one particular element of M through the whole multiplication. Work out how many times the naive kernel requests
          it from global memory; then answer: how many times does the tiled kernel request it?
        </Translated>
      </>
    ),
    answer: TILED_96.perElement,
    unit: 'requests',
    explain: (
      <Translated
        vi={
          <p>
            Một phần tử ở hàng r của M được {NAIVE_96.perElement} luồng tính hàng r của P cần đến, và trong kernel ngây thơ
            mỗi luồng tự nạp nó: {NAIVE_96.perElement} lần yêu cầu. Các luồng đó nằm trong 96 / 16 ={' '}
            {TILED_96.perElement} khối. Một khối phân ô nạp phần tử đó một lần, trong pha duy nhất có ô chứa nó, và cả 16 luồng
            của khối trên hàng đó dùng bản sao này. Vậy là {TILED_96.perElement} lần yêu cầu: N khi không phân ô, N/T với các
            ô T×T.
          </p>
        }
      >
        <p>
          An element in row r of M is needed by the {NAIVE_96.perElement} threads that compute row r of P, and in the
          naive kernel each of them fetches it for itself: {NAIVE_96.perElement} requests. Those threads sit in 96 / 16 ={' '}
          {TILED_96.perElement} blocks. A tiled block fetches the element once, in the one phase whose tile contains it,
          and all 16 of its threads in that row use the copy. So {TILED_96.perElement} requests: N without tiling, N/T
          with T×T tiles.
        </p>
      </Translated>
    ),
    source: 'ch. 5 exercise 8, p. 119; §5.3, p. 105',
    vi: { unit: 'lần yêu cầu', source: 'bài tập 8 chương 5, tr. 119; §5.3, tr. 105' },
  },
  'ch05/5.3/q-addition': {
    kind: 'choice',
    id: 'ch05/5.3/q-addition',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Phép cộng ma trận, <code>C[i][j] = A[i][j] + B[i][j]</code>, một luồng cho mỗi phần tử đầu ra. Nạp các ô của A và
              B vào bộ nhớ chia sẻ (shared memory) có cắt giảm được lưu lượng bộ nhớ toàn cục như với phép nhân không?
            </>
          }
        >
          Matrix addition, <code>C[i][j] = A[i][j] + B[i][j]</code>, one thread per output element. Could loading tiles of
          A and B into shared memory cut its global-memory traffic the way it does for multiplication?
        </Translated>
      </>
    ),
    options: [
      {
        label: (
          <Translated vi="Có, giảm đi số lần bằng độ rộng ô, như với phép nhân, vì mỗi khối nạp trọn các ô của A và B.">
            Yes, by a factor of the tile width, as for multiplication, since each block loads whole tiles of A and B.
          </Translated>
        ),
        why: (
          <Translated vi="Hệ số đó là số luồng trong một khối cần cùng một phần tử. Hãy đếm chúng ở đây.">
            That factor is the number of threads in a block that need the same element. Count them here.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Có, nhưng chỉ giảm 2 lần: một ô của A và một ô của B được cả khối chia sẻ.">
            Yes, but only by a factor of 2: one tile of A and one of B are shared by the block.
          </Translated>
        ),
        why: (
          <Translated vi="Phân ô không khai thác số lượng đầu vào. Nó khai thác việc nhiều luồng cần cùng một phần tử.">
            The number of inputs is not what tiling exploits. Several threads needing the same element is.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không: mỗi phần tử của A và B được một luồng dùng, một lần, nên không có lần nạp nào lặp lại.">
            No: each element of A and B is used by one thread, once, so no fetch repeats.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không, vì các ô của phép cộng quá lớn so với bộ nhớ chia sẻ.">
            No, because the tiles of an addition are too large for shared memory.
          </Translated>
        ),
        why: (
          <Translated vi="Ô có kích thước bao nhiêu là do bạn chọn. Kích thước không phải trở ngại; việc không có sự dùng lại mới là trở ngại.">
            A tile is whatever size you choose. Size is not the obstacle; the absence of reuse is.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Phân ô (tiling) không làm một lần nạp rẻ hơn. Nó thay nhiều lần nạp cùng một phần tử bằng một lần nạp và nhiều lần
            đọc một bản sao chia sẻ. Trong phép cộng, mỗi phần tử đầu vào vốn đã được nạp đúng một lần, bởi luồng duy nhất cần
            nó, nên lưu lượng vẫn là hai lần nạp và một lần ghi cho mỗi phép toán, và đi vòng qua bộ nhớ chia sẻ chỉ thêm việc.
            Câu cần hỏi với bất kỳ kernel nào trước khi dùng đến bộ nhớ chia sẻ: những phần tử nào được nhiều hơn một luồng
            của một khối cần đến?
          </p>
        }
      >
        <p>
          Tiling does not make a fetch cheaper. It replaces several fetches of the same element by one fetch and several
          reads of a shared copy. In addition every input element is already fetched exactly once, by the only thread
          that needs it, so the traffic stays at two loads and one store per operation and a detour through shared memory
          would only add work. The question to ask of any kernel before reaching for shared memory: which elements does
          more than one thread of a block need?
        </p>
      </Translated>
    ),
    source: 'ch. 5 exercise 1, p. 119; §5.1, p. 96',
    vi: { source: 'bài tập 1 chương 5, tr. 119; §5.1, tr. 96' },
  },
  'ch05/5.3/q-apply': {
    kind: 'numeric',
    id: 'ch05/5.3/q-apply',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Trình khám phá đã đếm một khối. Giờ hãy đếm cả một phép nhân: ma trận 8×8, ô 4×4, mọi khối, mọi pha. Tổng cộng
              kernel phân ô yêu cầu bao nhiêu lần nạp phần tử đầu vào từ bộ nhớ toàn cục? Tính cả tổng của kernel ngây thơ; phần
              giải thích sẽ so sánh hai con số.
            </>
          }
        >
          The explorer counted one block. Now count a whole multiplication: 8×8 matrices, 4×4 tiles, every block, every
          phase. How many loads of input elements does the tiled kernel request from global memory in total? Work out the
          naive kernel's total too; the explanation compares them.
        </Translated>
      </>
    ),
    answer: TILED_8.loads,
    unit: 'loads',
    explain: (
      <Translated
        vi={
          <p>
            Phân ô: 4 khối × 2 pha × 16 luồng × 2 lần nạp = {TILED_8.loads}. Ngây thơ: 64 luồng × 8 vòng lặp × 2 lần nạp
            = {NAIVE_8.loads}. Tỉ số là {NAIVE_8.loads / TILED_8.loads}, đúng bằng độ rộng ô. Nói cách khác, hai đầu vào có
            128 phần tử; kernel ngây thơ yêu cầu mỗi phần tử {NAIVE_8.perElement} lần còn kernel phân ô{' '}
            {TILED_8.perElement} lần. Với các ô 2×2 trên cùng các ma trận đó, tổng sẽ là {traffic(8, 2).loads}.
          </p>
        }
      >
        <p>
          Tiled: 4 blocks × 2 phases × 16 threads × 2 loads = {TILED_8.loads}. Naive: 64 threads × 8 iterations × 2 loads
          = {NAIVE_8.loads}. The ratio is {NAIVE_8.loads / TILED_8.loads}, the tile width. Put another way, the two
          inputs hold 128 elements; the naive kernel requests each one {NAIVE_8.perElement} times and the tiled kernel{' '}
          {TILED_8.perElement} times. With 2×2 tiles on the same matrices the total would be {traffic(8, 2).loads}.
        </p>
      </Translated>
    ),
    source: 'ch. 5 exercise 2, p. 119; §5.3, pp. 105–107',
    vi: { unit: 'lần nạp', source: 'bài tập 2 chương 5, tr. 119; §5.3, tr. 105–107' },
  },
};

// ---------------------------------------------------------------- the section
export function S53() {
  return (
    <>
      <Translated
        vi={
          <p>
            Bộ nhớ toàn cục (global memory) lớn và chậm. Bộ nhớ chia sẻ (shared memory) nhanh và nhỏ. Kernel ngây thơ chỉ dùng
            loại thứ nhất, và phần 5.1 đã đặt một con số cho cái giá đó: hai lần nạp từ bộ nhớ toàn cục cho mỗi phép nhân và
            cộng. Phân ô (tiling) là cách chuẩn để dùng cả hai loại bộ nhớ vào việc mà mỗi loại làm tốt. Cắt dữ liệu thành các
            mảnh, gọi là <b>ô (tile)</b>, mỗi mảnh đủ nhỏ để vừa bộ nhớ chia sẻ. Đưa một ô vào, làm hết mọi việc cần đến nó, rồi
            chuyển sang ô tiếp theo.
          </p>
        }
      >
        <p>
          Global memory is large and slow. Shared memory is fast and small. The naive kernel uses only the first, and
          section 5.1 put a number on the cost: two loads from global memory for every multiply and add. Tiling is the
          standard way to use both memories for what each is good at. Cut the data into pieces, called <b>tiles</b>, each
          small enough to fit in shared memory. Bring one tile in, do all the work that needs it, move on to the next.
        </p>
      </Translated>
      <Source section="5.3" pages="104" />
      <Translated
        vi={
          <p>
            Điều kiện nằm ẩn trong cụm "mọi việc cần đến nó". Phân ô chỉ áp dụng được khi phép tính có thể được sắp xếp sao cho
            việc trên một ô không cần các ô khác cùng lúc, và nó chỉ có lợi khi dữ liệu trong một ô được dùng nhiều hơn một
            lần trong lúc nằm đó. Không phải kernel nào cũng có dạng ấy. Phép nhân ma trận thì có, và bước đầu tiên là thấy nó
            nằm ở đâu.
          </p>
        }
      >
        <p>
          The condition is hidden in "all the work that needs it". Tiling only applies when the computation can be
          arranged so that the work on one tile does not need the others at the same moment, and it only pays when the
          data in a tile is used more than once while it is there. Not every kernel has that shape. Matrix multiplication
          does, and the first step is to see where.
        </p>
      </Translated>

      <h2>
        <Translated vi="Ai đọc gì">Who reads what</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lấy trường hợp nhỏ của sách: ma trận 4×4, các luồng được nhóm thành khối 2×2. Khối (0,0) chứa bốn luồng tính
            P<sub>0,0</sub>, P<sub>0,1</sub>, P<sub>1,0</sub> và P<sub>1,1</sub>. Luồng cho P<sub>0,0</sub> đi dọc hàng 0 của M
            và xuống cột 0 của N. Luồng cho P<sub>0,1</sub> lại đi dọc hàng 0 của M, và xuống cột 1 của N. Hai luồng cùng một
            hàng của khối đọc cùng một hàng của M; hai luồng cùng một cột đọc cùng một cột của N. Không có gì trong kernel ngây
            thơ cho phép một luồng dùng thứ mà luồng kia vừa nạp.
          </p>
        }
      >
        <p>
          Take the book's small case: 4×4 matrices, with the threads grouped into 2×2 blocks. Block (0,0) holds the four
          threads that compute P<sub>0,0</sub>, P<sub>0,1</sub>, P<sub>1,0</sub> and P<sub>1,1</sub>. The thread for P
          <sub>0,0</sub> walks along row 0 of M and down column 0 of N. The thread for P<sub>0,1</sub> walks along row 0
          of M again, and down column 1 of N. Two threads in the same row of a block read the same row of M; two threads
          in the same column read the same column of N. Nothing in the naive kernel lets one of them use what the other
          has just fetched.
        </p>
      </Translated>
      <Source section="5.3" pages="104–105" figs={tr('Figs. 5.5 and 5.6', 'Hình 5.5 và 5.6')} />
      <Predict
        id="ch05/5.3/p-overlap"
        prompt={
          <Translated vi="Phóng to lên: ma trận 8×8, khối 4×4, kernel ngây thơ. Một khối nạp mỗi phần tử của M mà nó cần bao nhiêu lần?">
            Scale that up: 8×8 matrices, 4×4 blocks, naive kernel. How many times does one block fetch each element of M that it needs?
          </Translated>
        }
        unit={tr('times', 'lần')}
      >
        {(v) => (
          <p>
            <Translated
              vi={
                <>
                  <b>Bốn.</b> 16 luồng của một khối phủ 4 hàng của P, và 4 luồng cùng một hàng đều đọc trọn hàng đó của M.
                </>
              }
            >
              <b>Four.</b> A block's 16 threads cover 4 rows of P, and the 4 threads that share a row each read that
              whole row of M.
            </Translated>{' '}
            {v === 4
              ? tr('That is what you had.', 'Đúng như bạn đã trả lời.')
              : v === 16
                ? tr(
                    'Sixteen would mean every thread of the block reads every element. Only the threads in the same row of P read a given row of M.',
                    'Mười sáu nghĩa là mọi luồng của khối đều đọc mọi phần tử. Chỉ những luồng trên cùng một hàng của P mới đọc một hàng nhất định của M.',
                  )
                : v === 8
                  ? tr(
                      'Eight is how many elements of M one thread loads, not how many threads load one element.',
                      'Tám là số phần tử của M mà một luồng nạp, không phải số luồng nạp một phần tử.',
                    )
                  : tr(
                      'Count the threads of the block that sit in one row of P.',
                      'Hãy đếm số luồng của khối nằm trên một hàng của P.',
                    )}{' '}
            <Translated vi="Đặt trình khám phá ở độ rộng 8 và kích thước khối 4 rồi kiểm tra.">
              Set the explorer to width 8 and block size 4 and check it.
            </Translated>
          </p>
        )}
      </Predict>
      <WhoReadsWhat />
      <Translated
        vi={
          <p>
            Dù bạn chọn gì, số lần nạp của khối chia cho số phần tử nó cần luôn ra đúng độ rộng của khối. Đó là kết quả tổng
            quát. Trong một khối T×T, có T luồng xin mỗi phần tử, nên một khối nạp mỗi phần tử một lần rồi chia sẻ sẽ cắt số
            yêu cầu đến bộ nhớ toàn cục đi T lần. Với các khối 2×2 trong ví dụ của sách, lưu lượng giảm một nửa; với khối
            32×32, nó giảm còn một phần ba mươi hai.
          </p>
        }
      >
        <p>
          Whatever you select, the block's loads divided by the elements it needs comes out as the width of the block.
          That is the general result. In a T×T block, T threads ask for each element, so a block that fetched each
          element once and shared it would cut its requests to global memory by a factor of T. With the 2×2 blocks of the
          book's example the traffic halves; with 32×32 blocks it falls to one thirty-second.
        </p>
      </Translated>
      <Source section="5.3" pages="105" />
      <Translated
        vi={
          <p>
            Hệ số đó cũng đúng cho cả phép nhân. Trong kernel ngây thơ, mỗi phần tử đầu vào được mọi luồng dùng nó yêu cầu
            một lần: W lần với ma trận W×W. Khi được chia sẻ trong các khối T×T, nó sẽ được mọi khối dùng nó yêu cầu một lần:
            W/T lần. Hãy nhớ cái đang được đếm. Đây là số yêu cầu, cũng là thứ mà <a href={href('ch05', '5.1')}>phần 5.1</a>{' '}
            đã đếm khi tính mọi lần nạp vào bộ nhớ toàn cục, và một yêu cầu không phải lúc nào cũng là một chuyến đi ra bộ nhớ
            ngoài chip.
          </p>
        }
      >
        <p>
          The same factor holds for the multiplication as a whole. In the naive kernel each input element is requested
          once by every thread that uses it: W times for W×W matrices. Shared inside T×T blocks, it would be requested
          once by every block that uses it: W/T times. Keep in mind what is being counted. These are requests, the same
          thing <a href={href('ch05', '5.1')}>section 5.1</a> counted when it charged every load to global memory, and a
          request is not always a trip to off-chip memory.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các con số là những gì kernel <em>yêu cầu</em> từ bộ nhớ toàn cục trong mô hình của sách, vốn không có bộ nhớ đệm
              (cache). Ở nơi bộ nhớ đệm trên chip đã hấp thụ các lần đọc lặp lại, mức tăng tốc đo được nhờ phân ô sẽ nhỏ hơn T.
              Lab 5.2 đo nó trên thiết bị của bạn.
            </p>
          }
        >
          <p>
            The counts are what the kernel <em>asks</em> of global memory in the book's model, which has no caches. Where
            on-chip caches already absorb repeated reads, the measured speed-up from tiling is smaller than T. Lab 5.2
            measures it on your device.
          </p>
        </Translated>
      </Simplification>
      <Ask q={questionsS53['ch05/5.3/q-requests']} />

      <h2>
        <Translated vi="Từ chia sẻ đến các pha">From sharing to phases</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một khối sắp xếp thế nào để nạp mỗi phần tử một lần? Nó không thể đơn giản chép mọi thứ nó cần vào bộ nhớ chia
            sẻ: đó là T hàng trọn vẹn của M và T cột trọn vẹn của N, và nó lớn dần theo ma trận. Vì thế công việc được cắt dọc
            theo tích vô hướng. Chia M và N thành các ô T×T, cùng kích thước với một khối. Tích vô hướng W số hạng của mỗi luồng
            trở thành W/T tổng riêng phần, mỗi tổng T số hạng, và khối tính chúng từng <b>pha</b> một. Một pha có hai bước:
          </p>
        }
      >
        <p>
          How does a block arrange to fetch each element once? It cannot simply copy everything it needs into shared
          memory: that is T whole rows of M and T whole columns of N, and it grows with the matrix. So the work is cut
          along the dot product instead. Divide M and N into T×T tiles, the same size as a block. Each thread's dot
          product of W terms becomes W/T partial sums of T terms each, and the block computes them one <b>phase</b> at a
          time. A phase has two steps:
        </p>
      </Translated>
      <Source section="5.3" pages="105–107" figs={tr('Figs. 5.7 and 5.8', 'Hình 5.7 và 5.8')} />
      <Translated
        vi={
          <ol>
            <li>
              Mỗi luồng nạp một phần tử của M và một phần tử của N từ bộ nhớ toàn cục vào hai mảng chia sẻ, <code>Mds</code>{' '}
              và <code>Nds</code>. Cùng nhau, T² luồng đưa vào trọn một ô của mỗi ma trận.
            </li>
            <li>
              Mỗi luồng cộng T tích vào tổng đang cộng dồn của nó, chỉ đọc <code>Mds</code> và <code>Nds</code>.
            </li>
          </ol>
        }
      >
        <ol>
          <li>
            Every thread loads one element of M and one element of N from global memory into two shared arrays,{' '}
            <code>Mds</code> and <code>Nds</code>. Between them the T² threads bring in one whole tile of each.
          </li>
          <li>
            Every thread adds T products to its running sum, reading only <code>Mds</code> and <code>Nds</code>.
          </li>
        </ol>
      </Translated>
      <Translated
        vi={
          <p>
            Ở pha 0, khối nạp ô ở đầu trái các hàng của nó trong M và ô ở đỉnh các cột của nó trong N. Mỗi giá trị đã nạp sau
            đó được đọc T lần: một phần tử của <code>Mds</code> bởi T luồng trên hàng của nó trong khối, một phần tử của{' '}
            <code>Nds</code> bởi T luồng trên cột của nó. Pha 1 dịch sang phải một ô trong M và xuống một ô trong N, cứ thế
            trong W/T pha. Một yêu cầu đến bộ nhớ toàn cục, T lần sử dụng.
          </p>
        }
      >
        <p>
          In phase 0 the block loads the tile at the left end of its rows of M and the tile at the top of its columns of
          N. Each loaded value is then read T times: an element of <code>Mds</code> by the T threads in its row of the
          block, an element of <code>Nds</code> by the T threads in its column. Phase 1 moves one tile to the right in M
          and one tile down in N, and so on for W/T phases. One request to global memory, T uses.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Ở đây các pha được đánh số từ 0, giống biến vòng lặp <code>ph</code> trong kernel. Hình 5.8 của sách cũng làm vậy,
              trong khi phần chữ ở tr. 107 gọi chính hai pha đó là 1 và 2.
            </p>
          }
        >
          <p>
            Phases are numbered from 0 here, like the loop variable <code>ph</code> in the kernel. The book's Fig. 5.8
            does the same, while its text on p. 107 calls the same two phases 1 and 2.
          </p>
        </Translated>
      </Aside>
      <WhyTilingScene />
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy khối (0,0) của một phép nhân nhỏ với số thật. Với mỗi pha, nó cho thấy hai ô đang được
            đọc từ bộ nhớ toàn cục, mỗi luồng nạp gì và vào vị trí nào, nội dung của hai ô chia sẻ, và tổng đang cộng dồn của
            mỗi luồng khi pha kết thúc. Hãy bước tới và để mắt đến các vị trí trong bộ nhớ chia sẻ.
          </p>
        }
      >
        <p>
          The stepper runs block (0,0) of a small multiplication with real numbers. For each phase it shows the two
          tiles being read from global memory, what every thread loads and into which slot, the contents of the two
          shared tiles, and each thread's running sum once the phase is over. Step forward and keep an eye on the shared
          slots.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Hoạt cảnh ở trên và trình chạy từng bước cho thấy trọn các pha, với mọi luồng của khối di chuyển cùng nhau. Luồng
              thật không di chuyển cùng nhau, và không có gì ở đây cho thấy thời gian hay bộ nhớ đệm. Phần 5.4 xử lý những gì điều
              đó làm thay đổi.
            </p>
          }
        >
          <p>
            The animation above and the stepper show whole phases, with all threads of the block moving together. Real
            threads do not move together, and nothing here shows timing or caches. Section 5.4 deals with what that
            changes.
          </p>
        </Translated>
      </Simplification>
      <PhasesBench />

      <h2>
        <Translated vi="Vì sao một bộ nhớ nhỏ là đủ">Why a small memory is enough</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hai điều trong trình chạy từng bước chứa đựng ý tưởng. Thứ nhất, <code>Mds</code> và <code>Nds</code> không bao giờ
            lớn lên. Mỗi pha ghi vào chính những vị trí mà pha trước đã dùng, vì đến lúc đó mọi luồng đã lấy xong thứ nó cần từ
            đó. Hai ô gồm T² giá trị phục vụ được phép nhân với độ rộng bất kỳ, và đó là cách một bộ nhớ chia sẻ nhỏ rốt cuộc
            đáp ứng phần lớn các lần đọc của kernel.
          </p>
        }
      >
        <p>
          Two things in the stepper carry the idea. First, <code>Mds</code> and <code>Nds</code> never grow. Each phase
          writes into the very slots the previous phase used, because by then every thread has taken what it needed from
          them. Two tiles of T² values serve a multiplication of any width, which is how a small shared memory ends up
          answering most of the kernel's reads.
        </p>
      </Translated>
      <Source section="5.3" pages="107" />
      <Translated
        vi={
          <p>
            Thứ hai, trong mỗi pha, mọi luồng của khối làm việc trên cùng một mảnh nhỏ của đầu vào. Thuộc tính đó gọi là{' '}
            <b>tính cục bộ</b>, và nó là lý do một bộ nhớ nhỏ mà nhanh có thể thay cho một bộ nhớ lớn mà chậm. Điều này không
            riêng gì GPU: trên một CPU đa lõi, bộ nhớ nhỏ mà nhanh là bộ nhớ đệm, và cách tổ chức lại tương tự cũng giúp ích ở đó.
          </p>
        }
      >
        <p>
          Second, in each phase all the threads of the block work on the same small piece of the input. That property is
          called <b>locality</b>, and it is the reason a small fast memory can stand in for a large slow one. It is not
          special to GPUs: on a multicore CPU the small fast memory is the cache, and the same restructuring helps there.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Dùng lại các vị trí có một cái giá mà trình chạy từng bước che đi bằng cách cho mọi luồng di chuyển cùng lúc. Không
            luồng nào được đọc một vị trí trước khi luồng bên cạnh đã nạp nó, và không luồng nào được nạp giá trị của pha sau
            trong khi luồng bên cạnh vẫn cần giá trị cũ. <a href={href('ch05', '5.4')}>Phần 5.4</a> viết kernel và trả cái giá
            đó.
          </p>
        }
      >
        <p>
          Reusing the slots has a price that the stepper hides by moving all threads at once. No thread may read a slot
          before its neighbour has loaded it, and no thread may load the next phase's value while a neighbour still needs
          the old one. <a href={href('ch05', '5.4')}>Section 5.4</a> writes the kernel and pays that price.
        </p>
      </Translated>
      <Ask q={questionsS53['ch05/5.3/q-addition']} />
      <Ask q={questionsS53['ch05/5.3/q-apply']} />
    </>
  );
}
