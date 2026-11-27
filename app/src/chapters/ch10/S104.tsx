import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { STATUS, useLabStatus } from './status';
import {
  BOOK_PRINTED,
  CONVERGENT_SPEC,
  EXERCISE_6,
  FIG_VALUES,
  KIND_LABEL,
  MIRROR_SPEC,
  OWNED,
  PARTNERS,
  TESTS,
  WRITERS,
  iterations,
  runSpec,
  type InPlaceKind,
  type Iteration,
  type OwnedKey,
  type PartnerKey,
  type Spec,
  type SpecRun,
  type TestKey,
  type Trace,
  type WriterKey,
} from '../../sim/reduction';
import { functionSource } from '../../sim/toolchain';
import { widthChoices } from '../../sim/warps';

import { NoteGroup } from './NoteGroup';

// The lab whose runs give the status of the listings taken from its files (and from labs/ch10/shared/, which it runs).
const LAB10_1 = 'ch10/lab1_sum';
const TIERS: Record<string, Tier> = { input: 'global', output: 'global', row: 'global', i: 'reg', stride: 'reg', t: 'reg', b: 'reg', bdim: 'reg' };
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const missing = (s: string | null) => !s || s.startsWith('// could not load');
const pct = (x: number) => `${Math.round(x * 100)}%`;
const SIZES = [16, 256, 512, 1024, 2048];

/** Lines of `b` (1-based) that differ from the same line of `a`, ignoring the function's name. */
function changedLines(a: string, b: string, nameA: string, nameB: string): number[] {
  const la = a.split('\n');
  return b.split('\n').flatMap((l, i) => (l.replace(nameB, nameA) !== la[i] ? [i + 1] : []));
}

// ---------------------------------------------------------------- the 256-element table
function BookTable() {
  const s = iterations('simple', 256, 32);
  const c = iterations('convergent', 256, 32);
  const div = (d: number) => <td className={'num' + (d ? ' t-fail' : '')}>{d}</td>;
  return (
    <table className="data" data-part="book-table" style={{ maxWidth: '40rem', whiteSpace: 'nowrap' }}>
      <thead>
        <tr>
          <th>
            <Translated vi="Vòng lặp">Iteration</Translated>
          </th>
          <th className="num">
            <Translated vi="Luồng hoạt động">Active threads</Translated>
          </th>
          <th className="num">
            <Translated vi="Đơn giản: bước nhảy">Simple: stride</Translated>
          </th>
          <th className="num">
            <Translated vi="warp">warps</Translated>
          </th>
          <th className="num">
            <Translated vi="phân kỳ">divergent</Translated>
          </th>
          <th className="num">
            <Translated vi="Hội tụ: bước nhảy">Convergent: stride</Translated>
          </th>
          <th className="num">
            <Translated vi="warp">warps</Translated>
          </th>
          <th className="num">
            <Translated vi="phân kỳ">divergent</Translated>
          </th>
        </tr>
      </thead>
      <tbody>
        {s.iterations.map((it, k) => (
          <tr key={k}>
            <td>{it.n}</td>
            <td className="num">{it.active}</td>
            <td className="num">{it.stride}</td>
            <td className="num">{it.activeWarps}</td>
            {div(it.divergentWarps)}
            <td className="num">{c.iterations[k].stride}</td>
            <td className="num">{c.iterations[k].activeWarps}</td>
            {div(c.iterations[k].divergentWarps)}
          </tr>
        ))}
        <tr>
          <td><b><Translated vi="Tổng">Total</Translated></b></td>
          <td className="num"><b>{s.results}</b></td>
          <td />
          <td className="num" data-total="simple" colSpan={2}><b>{s.slots / 32} × 32 = {s.slots}</b></td>
          <td />
          <td className="num" data-total="convergent" colSpan={2}><b>{c.slots / 32} × 32 = {c.slots}</b></td>
        </tr>
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------- the warp map
/** One iteration of one kernel: a row of lanes per warp, active lanes filled. */
function WarpRows({ it, width }: { it: Iteration; width: number }) {
  const cw = Math.max(4, Math.min(13, Math.floor(360 / width)));
  const rh = Math.max(8, cw + 3);
  const left = 34;
  const chipW = 62;
  const W = left + width * cw + chipW;
  const H = it.warps.length * rh + 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('One row per warp, one cell per lane; filled lanes are active', 'Mỗi hàng một warp, mỗi ô một lane; các lane tô đậm đang hoạt động')} data-part="warps"
      style={{ width: W, maxWidth: '100%', display: 'block' }}>
      <defs>
        <pattern id="s104-pad" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="4" stroke="var(--ink-3)" strokeWidth="1" />
        </pattern>
      </defs>
      {it.warps.map((w) => (
        <g key={w.warp} data-warp={w.warp} data-state={w.state}>
          <text x={left - 5} y={w.warp * rh + rh - 3} fontSize={Math.min(10, rh)} textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
            w{w.warp}
          </text>
          {w.lanes.map((l, i) => (
            <rect key={i} x={left + i * cw} y={w.warp * rh + 1} width={cw - 1} height={rh - 2} rx="1"
              fill={l === null ? 'url(#s104-pad)' : l ? 'var(--ink-2)' : 'var(--cell)'} stroke="var(--rule)" strokeWidth="0.5" />
          ))}
          <text x={left + width * cw + 6} y={w.warp * rh + rh - 3} fontSize={Math.min(10, rh)}
            style={{ fill: w.state === 'divergent' ? 'var(--fail)' : w.state === 'full' ? 'var(--ok)' : 'var(--ink-3)', fontWeight: w.state === 'idle' ? 400 : 650 }}>
            <Translated vi={{ full: 'đầy', divergent: 'phân kỳ', idle: 'rảnh' }[w.state]}>{w.state}</Translated>
          </text>
        </g>
      ))}
    </svg>
  );
}

/** Totals over the first n iterations. */
function soFar(t: Trace, n: number) {
  const its = t.iterations.slice(0, n);
  const slots = its.reduce((a, i) => a + i.slots, 0);
  const results = its.reduce((a, i) => a + i.active, 0);
  return { slots, results, efficiency: slots ? results / slots : 0 };
}

function KernelPanel({ t, n }: { t: Trace; n: number }) {
  const it = t.iterations[n - 1];
  const sf = soFar(t, n);
  return (
    <div className="panel" data-kernel={t.kind} data-iteration={n} style={{ minWidth: 0, overflowX: 'auto' }}>
      <div className="mx-label">
        <Translated
          vi={
            <>
              {{ simple: 'đơn giản (Fig. 10.5)', convergent: 'hội tụ (Fig. 10.8)', shared: 'bộ nhớ chia sẻ (Fig. 10.10)' }[t.kind]}: vòng
              lặp {it.n}, bước nhảy {it.stride}
            </>
          }
        >
          {KIND_LABEL[t.kind]}: iteration {it.n}, stride {it.stride}
        </Translated>
      </div>
      <p className="muted" style={{ margin: '0 0 0.5rem' }} data-readout="iteration">
        <Translated
          vi={
            <>
              {it.active} luồng hoạt động trong {it.activeWarps} trên {t.warpCount} warp;{' '}
              <span data-divergent={it.divergentWarps}>{it.divergentWarps} phân kỳ</span>.
            </>
          }
        >
          {it.active} active thread{it.active === 1 ? '' : 's'} in {it.activeWarps} of {t.warpCount} warp{t.warpCount === 1 ? '' : 's'};{' '}
          <span data-divergent={it.divergentWarps}>{it.divergentWarps} divergent</span>.
        </Translated>
      </p>
      <WarpRows it={it} width={t.width} />
      <table className="threads" style={{ marginTop: '0.7rem' }} aria-label={tr(`Iterations of the ${t.kind} kernel`, `Các vòng lặp của kernel ${t.kind === 'simple' ? 'đơn giản' : 'hội tụ'}`)}>
        <thead>
          <tr>
            <th><Translated vi="v.l.">it.</Translated></th>
            <th><Translated vi="bước nhảy">stride</Translated></th>
            <th><Translated vi="luồng hoạt động">active threads</Translated></th>
            <th><Translated vi="warp hoạt động">active warps</Translated></th>
            <th><Translated vi="phân kỳ">divergent</Translated></th>
            <th><Translated vi="slot">slots</Translated></th>
          </tr>
        </thead>
        <tbody>
          {t.iterations.map((x) => (
            <tr key={x.n} className={x.n === n ? 'cur' : ''} style={x.n > n ? { color: 'var(--ink-3)' } : undefined}>
              <td>{x.n}</td><td>{x.stride}</td><td>{x.active}</td><td>{x.activeWarps}</td>
              <td className={x.divergentWarps ? 't-fail' : ''}>{x.divergentWarps}</td><td>{x.slots}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ margin: '0.6rem 0 0' }} data-readout="totals" data-slots={t.slots} data-efficiency={t.efficiency.toFixed(4)}>
        <Translated
          vi={
            <>
              Tới giờ: {sf.slots} slot đã dùng, {sf.results} kết quả, hiệu suất {pct(sf.efficiency)}.{' '}
              <b>Cả vòng lặp: {t.slots} slot, {t.results} kết quả, {pct(t.efficiency)}</b>; {t.divergentIterations} trên{' '}
              {t.iterations.length} vòng lặp có một warp phân kỳ.
            </>
          }
        >
          So far: {sf.slots} slots consumed, {sf.results} results, efficiency {pct(sf.efficiency)}.{' '}
          <b>Whole loop: {t.slots} slots, {t.results} results, {pct(t.efficiency)}</b>; {t.divergentIterations} of {t.iterations.length} iterations
          have a divergent warp.
        </Translated>
      </p>
    </div>
  );
}

type Show = InPlaceKind | 'both';
interface MapState {
  N: number;
  widthKey: string;
  show: Show;
  n: number;
}

function DivergenceLab() {
  const { reports } = useReports();
  const choices = useMemo(() => widthChoices(reports), [reports]);
  const [s, setS] = useState<MapState>({ N: 256, widthKey: 'book', show: 'both', n: 1 });
  const chosen = choices.find((c) => c.key === s.widthKey) ?? choices[0];
  const width = chosen.width;
  const traces = useMemo(
    () => ({ simple: iterations('simple', s.N, width), convergent: iterations('convergent', s.N, width) }),
    [s.N, width],
  );
  const L = traces.simple.iterations.length;
  const n = Math.min(s.n, L);
  const kinds: InPlaceKind[] = s.show === 'both' ? ['simple', 'convergent'] : [s.show];
  const book = { s: iterations('simple', 256, 32), c: iterations('convergent', 256, 32) };

  // Quiz me: local only, not recorded.
  const [quiz, setQuiz] = useState<{ kind: InPlaceKind; N: number; width: number; n: number } | null>(null);
  const [guess, setGuess] = useState('');
  const [revealed, setRevealed] = useState(false);
  const draw = () => {
    const kind: InPlaceKind = Math.random() < 0.5 ? 'simple' : 'convergent';
    const N = [256, 512, 1024, 2048][Math.floor(Math.random() * 4)];
    const w = Math.random() < 0.5 ? 32 : 64;
    const its = iterations(kind, N, w).iterations.length;
    setQuiz({ kind, N, width: w, n: 1 + Math.floor(Math.random() * its) });
    setGuess('');
    setRevealed(false);
  };
  const answer = quiz ? iterations(quiz.kind, quiz.N, quiz.width).iterations[quiz.n - 1].divergentWarps : 0;
  const reveal = () => {
    if (!quiz) return;
    setRevealed(true);
    setS({ N: quiz.N, widthKey: quiz.width === 32 ? 'book' : 'w64', show: quiz.kind, n: quiz.n });
  };

  return (
    <Bench
      title={tr('Warp map', 'Bản đồ warp')}
      task={tr(
        'Same number of active threads in each iteration; count the warps that pay for them.',
        'Cùng số luồng hoạt động ở mỗi vòng lặp; hãy đếm các warp phải trả giá cho chúng.',
      )}
      foot={
        <Translated
          vi={
            <>
              Mỗi hàng là một warp, mỗi ô là một lane; các lane tô đậm đang hoạt động ở vòng lặp này, các lane gạch chéo không
              có luồng nào. Một warp có bất kỳ lane nào hoạt động đều tốn đủ {width} lane của nó cho vòng lặp đó, theo cách tính
              của sách.
            </>
          }
        >
          <>
            Each row is a warp, each cell a lane; filled lanes are active in this iteration, hatched lanes have no thread. A
            warp with any active lane costs all {width} of its lanes for the iteration, the book's accounting.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span style={FIELD}>
          <Translated vi="N (số giá trị)">N (values)</Translated>
          <span className="seg" role="group" aria-label="N">
            {SIZES.map((N) => (
              <button key={N} aria-pressed={s.N === N} onClick={() => setS({ ...s, N, n: 1 })}>{N}</button>
            ))}
          </span>
        </span>
        <label className="field" style={{ maxWidth: '100%', minWidth: 0 }}>
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={s.widthKey} onChange={(e) => setS({ ...s, widthKey: e.target.value, n: 1 })} aria-label={tr('Warp width', 'Độ rộng warp')}>
            {choices.map((c) => (
              <option key={c.key} value={c.key}>
                {tr(c.label, c.key === 'book' ? '32 của sách' : c.key === 'w64' ? '64, như HIP ghi cho gfx9' : c.label)}
              </option>
            ))}
          </select>
        </label>
        <span style={FIELD}>
          <Translated vi="Kernel">Kernel</Translated>
          <span className="seg" role="group" aria-label={tr('Kernel', 'Kernel')}>
            {(['simple', 'convergent', 'both'] as Show[]).map((k) => (
              <button key={k} aria-pressed={s.show === k} onClick={() => setS({ ...s, show: k })}>
                {k === 'both'
                  ? tr('side by side', 'cạnh nhau')
                  : tr(k, k === 'simple' ? 'đơn giản' : 'hội tụ')}
              </button>
            ))}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <button onClick={() => setS({ ...s, n: Math.max(1, n - 1) })} disabled={n <= 1}>
          <Translated vi="Lùi">Back</Translated>
        </button>
        <button className="primary" onClick={() => setS({ ...s, n: Math.min(L, n + 1) })} disabled={n >= L}>
          <Translated vi="Bước">Step</Translated>
        </button>
        <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
          <Translated vi="vòng lặp">iteration</Translated>
          <input type="range" min={1} max={L} value={n} onChange={(e) => setS({ ...s, n: Number(e.target.value) })} aria-label={tr('Iteration', 'Vòng lặp')} />
          <span data-readout="iteration-number">
            <Translated vi={<>{n} trên {L}</>}>
              {n} of {L}
            </Translated>
          </span>
        </label>
      </div>
      <div className="cols" style={{ gridTemplateColumns: kinds.length === 2 ? 'minmax(0, 1fr) minmax(0, 1fr)' : 'minmax(0, 1fr)' }}>
        {kinds.map((k) => <KernelPanel key={k} t={traces[k]} n={n} />)}
      </div>
      <p className="muted" style={{ margin: '0.9rem 0 0' }} data-readout="erratum">
        <Translated
          vi={
            <>
              Với 256 giá trị và warp rộng 32, sách in ở p. 234 {BOOK_PRINTED.simpleSlots} slot cho kernel đơn giản và số vòng
              lặp phân kỳ giảm "từ {BOOK_PRINTED.divergentBefore === 10 ? 'mười' : BOOK_PRINTED.divergentBefore} xuống năm".
              Chính công thức của sách ở p. 231 cho {book.s.slots}, và đếm từ mã cho {book.s.divergentIterations} vòng lặp phân
              kỳ so với {book.c.divergentIterations}: {book.s.slots} slot xuống {book.c.slots}.
            </>
          }
        >
          For 256 values and 32-wide warps the book prints, on p. 234, {BOOK_PRINTED.simpleSlots} slots for the simple kernel and
          divergent iterations falling "from {BOOK_PRINTED.divergentBefore === 10 ? 'ten' : BOOK_PRINTED.divergentBefore} to five". The
          book's own formula on p. 231 gives {book.s.slots}, and counting from the code gives {book.s.divergentIterations} divergent
          iterations against {book.c.divergentIterations}: {book.s.slots} slots to {book.c.slots}.
        </Translated>
      </p>

      <div className="panel" style={{ marginTop: '1rem' }} data-part="quiz">
        <div className="mx-label">
          <Translated vi="Đố tôi">Quiz me</Translated>
        </div>
        {!quiz ? (
          <p style={{ margin: '0 0 0.6rem' }}>
            <Translated vi="Rút một kernel, một kích thước, một độ rộng và một vòng lặp, rồi đếm các warp phân kỳ trước khi bản đồ cho thấy chúng. Không được ghi lại.">
              Draw a kernel, a size, a width and an iteration, then count the divergent warps before the map shows them. Not recorded.
            </Translated>
          </p>
        ) : (
          <p style={{ margin: '0 0 0.6rem' }} data-readout="quiz">
            <Translated
              vi={
                <>
                  Kernel <b>{quiz.kind === 'simple' ? 'đơn giản' : 'hội tụ'}</b> trên <b>{quiz.N}</b> giá trị ({quiz.N / 2} luồng),
                  warp rộng <b>{quiz.width}</b>, vòng lặp <b>{quiz.n}</b>. Có bao nhiêu warp phân kỳ?
                </>
              }
            >
              The <b>{quiz.kind}</b> kernel on <b>{quiz.N}</b> values ({quiz.N / 2} threads), <b>{quiz.width}</b>-wide warps,
              iteration <b>{quiz.n}</b>. How many warps diverge?
            </Translated>
          </p>
        )}
        <div className="row">
          <button onClick={draw}>{quiz ? tr('Another', 'Câu khác') : tr('Draw a question', 'Rút một câu hỏi')}</button>
          {quiz ? (
            <>
              <input type="number" value={guess} onChange={(e) => { setGuess(e.target.value); setRevealed(false); }} aria-label={tr('divergent warps', 'số warp phân kỳ')} />
              <button className="primary" onClick={reveal} disabled={guess.trim() === ''}>
                <Translated vi="Xem đáp án">Reveal</Translated>
              </button>
            </>
          ) : null}
        </div>
        {quiz && revealed ? (
          <p style={{ margin: '0.6rem 0 0' }} data-readout="quiz-answer" data-answer={answer}>
            <Translated
              vi={
                <>
                  {Number(guess) === answer ? <span className="chip ok">đúng</span> : <span className="chip fail">chưa đúng</span>}{' '}
                  {answer} warp phân kỳ: bước nhảy{' '}
                  {iterations(quiz.kind, quiz.N, quiz.width).iterations[quiz.n - 1].stride}, và bản đồ ở trên giờ đang hiện vòng lặp
                  đó.
                </>
              }
            >
              {Number(guess) === answer ? <span className="chip ok">right</span> : <span className="chip fail">not quite</span>}{' '}
              {answer} warp{answer === 1 ? '' : 's'} diverge{answer === 1 ? 's' : ''}: stride{' '}
              {iterations(quiz.kind, quiz.N, quiz.width).iterations[quiz.n - 1].stride}, and the map above now shows that iteration.
            </Translated>
          </p>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- mirror it (exercise 3)
const OWNED_KEYS: OwnedKey[] = ['t', '2t', 't+B', '2t+1'];
const TEST_KEYS: TestKey[] = ['lt', 'mod', 'ge', 'mirror'];
const PARTNER_KEYS: PartnerKey[] = ['plus', 'minus'];
const WRITER_KEYS: WriterKey[] = ['first', 'middle', 'last'];

function specCode(s: Spec): string {
  return [
    '__global__ void MyKernel(float* input, float* output) {',
    `    unsigned int i = ${OWNED[s.owned].code};`,
    '    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {',
    `        if (${TESTS[s.test].code}) {`,
    `            input[i] += ${PARTNERS[s.partner].code};`,
    '        }',
    '        __syncthreads();',
    '    }',
    `    ${WRITERS[s.writer].code}`,
    '}',
  ].join('\n');
}

/** The input and the array after each iteration of a spec run, with the additions as arrows. */
function SpecGrid({ run, values }: { run: SpecRun; values: readonly number[] }) {
  const N = values.length;
  const CW = N > 8 ? 34 : 46;
  const CH = 24;
  const GAP = 24;
  const LW = 74;
  const y = (r: number) => 4 + r * (CH + GAP);
  const x = (c: number) => LW + c * CW;
  const rows = [{ stride: 0, after: [...values], ops: [] as SpecRun['iterations'][number]['ops'], conflicts: [] as number[] }, ...run.iterations];
  const W = LW + N * CW + 4;
  const H = y(rows.length - 1) + CH + 4;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('The input and the array after each iteration of the assembled kernel', 'Đầu vào và mảng sau mỗi vòng lặp của kernel đã lắp ráp')} data-part="spec-grid"
      style={{ width: W, maxWidth: '100%', minWidth: Math.min(W, 460), display: 'block' }}>
      <defs>
        <marker id="s104-arrow" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute value */}
          <path d="M0,0 L6,3 L0,6 z" fill="var(--ink-3)" />
        </marker>
      </defs>
      {rows.map((row, r) => {
        const written = new Set(row.ops.map((o) => o.own));
        const bad = new Set([...row.conflicts, ...row.ops.filter((o) => o.oob).map((o) => o.own)]);
        return (
          <g key={r} data-row={r}>
            <text x={LW - 8} y={y(r) + CH / 2 + 4} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
              {r === 0 ? tr('input', 'đầu vào') : tr(`stride ${row.stride}`, `bước nhảy ${row.stride}`)}
            </text>
            {row.ops.map((o) => {
              const tx = x(o.own) + CW / 2;
              const inside = o.partner >= 0 && o.partner < N;
              return (
                <g key={o.t} opacity={0.8}>
                  <line x1={tx} y1={y(r - 1) + CH} x2={tx} y2={y(r) - 1} stroke="var(--ink-3)" markerEnd="url(#s104-arrow)" />
                  {inside ? (
                    <line x1={x(o.partner) + CW / 2} y1={y(r - 1) + CH} x2={tx + (o.partner > o.own ? 3 : -3)} y2={y(r) - 1}
                      stroke="var(--ink-3)" markerEnd="url(#s104-arrow)" />
                  ) : null}
                </g>
              );
            })}
            {row.after.map((v, c) => {
              const w = r === 0 || written.has(c);
              const b = bad.has(c);
              return (
                <g key={c} data-cell={c} data-value={Number.isNaN(v) ? '?' : v} data-written={w ? '1' : '0'} data-bad={b ? '1' : '0'}>
                  <rect x={x(c) + 1} y={y(r)} width={CW - 2} height={CH} rx="2"
                    fill={b ? 'var(--fail-wash)' : w ? 'var(--global-wash)' : 'transparent'}
                    stroke={b ? 'var(--fail)' : w ? 'var(--global)' : 'var(--rule)'} strokeDasharray={w ? undefined : '3 2'} />
                  <text x={x(c) + CW / 2} y={y(r) + CH / 2 + 4} fontSize="11" textAnchor="middle"
                    style={{ fill: b ? 'var(--fail)' : w ? 'var(--ink)' : 'var(--ink-3)', fontWeight: w && r > 0 ? 650 : 400 }}>
                    {Number.isNaN(v) ? '?' : v}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

function MirrorIt() {
  const [spec, setSpec] = useState<Spec>(CONVERGENT_SPEC);
  const [data, setData] = useState<'fig' | 'ex6'>('fig');
  const [showRef, setShowRef] = useState(false);
  const ref = useLabFile('ch10/snippets/s104_mirrored.cu');
  const values = data === 'fig' ? FIG_VALUES : EXERCISE_6;
  const run = useMemo(() => runSpec(spec, values), [spec, values]);
  const N = values.length;
  const isMirror = (Object.keys(MIRROR_SPEC) as (keyof Spec)[]).every((k) => spec[k] === MIRROR_SPEC[k]);
  const pick = <K extends keyof Spec>(k: K, v: Spec[K]) => setSpec({ ...spec, [k]: v });
  const sel = <K extends keyof Spec>(k: K, label: string, keys: Spec[K][], code: (x: Spec[K]) => string) => (
    <label className="field" style={{ minWidth: 0, maxWidth: '100%' }}>
      {label}
      <select value={spec[k] as string} onChange={(e) => pick(k, e.target.value as Spec[K])} aria-label={label}>
        {keys.map((x) => <option key={x as string} value={x as string}>{code(x)}</option>)}
      </select>
    </label>
  );
  return (
    <Bench
      title={tr('Mirror it', 'Lật ngược nó')}
      task={tr(
        'Exercise 3: change Fig. 10.8 so that threads own the second half and the total lands in the last element.',
        'Bài tập 3: sửa Fig. 10.8 để các luồng sở hữu nửa sau và tổng rơi vào phần tử cuối cùng.',
      )}
      foot={
        <Translated
          vi={
            <>
              Vòng lặp là của Fig. 10.8, bước nhảy từ blockDim.x xuống 1, cùng rào chắn của nó; bạn chọn bốn phần. Mỗi hàng cho
              thấy những gì một vòng lặp đã ghi, với mũi tên từ hai phần tử mà một phép cộng đã đọc. Màu đỏ đánh dấu một phần tử
              bị đọc trong cùng vòng lặp mà một luồng khác ghi vào nó, hoặc một chỉ số nằm ngoài đầu vào.
            </>
          }
        >
          <>
            The loop is Fig. 10.8's, stride from blockDim.x down to 1, with its barrier; you choose the four parts. Each row
            shows what one iteration wrote, with arrows from the two elements an addition read. Red marks an element read in
            the same iteration that another thread writes it, or an index outside the input.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        {sel('owned', tr('Owned element i =', 'Phần tử sở hữu i ='), OWNED_KEYS, (k) => OWNED[k].code)}
        {sel('test', tr('Active when', 'Hoạt động khi'), TEST_KEYS, (k) => TESTS[k].code)}
        {sel('partner', tr('Partner', 'Phần tử đối tác'), PARTNER_KEYS, (k) => PARTNERS[k].code)}
        {sel('writer', tr('After the loop', 'Sau vòng lặp'), WRITER_KEYS, (k) => WRITERS[k].code)}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi="Dữ liệu">Data</Translated>
          <select value={data} onChange={(e) => setData(e.target.value as 'fig' | 'ex6')} aria-label={tr('Data', 'Dữ liệu')}>
            <option value="fig">{tr("16 values (the exercise's figure)", '16 giá trị (hình của bài tập)')}</option>
            <option value="ex6">{tr('exercise 6: 6, 2, 7, 4, 5, 8, 3, 1', 'bài tập 6: 6, 2, 7, 4, 5, 8, 3, 1')}</option>
          </select>
        </label>
        <button onClick={() => setSpec(CONVERGENT_SPEC)}>
          <Translated vi="Quay lại Fig. 10.8">Back to Fig. 10.8</Translated>
        </button>
      </div>
      <div style={{ minWidth: 0 }}>
        <div style={{ overflowX: 'auto' }}>
          <SpecGrid run={run} values={values} />
        </div>
        <div style={{ marginTop: '0.7rem' }} data-readout="verdict" data-correct={run.correct ? '1' : '0'} data-last={run.totalInLast ? '1' : '0'}>
          {run.oob ? (
            <p style={{ margin: '0 0 0.3rem' }}>
              <Translated
                vi={
                  <>
                    <span className="chip fail">ngoài biên</span> {run.oob} lần đọc nằm ngoài đầu vào.
                  </>
                }
              >
                <span className="chip fail">out of bounds</span> {run.oob} read{run.oob > 1 ? 's' : ''} outside the input.
              </Translated>
            </p>
          ) : null}
          {run.conflicts ? (
            <p style={{ margin: '0 0 0.3rem' }}>
              <Translated
                vi={
                  <>
                    <span className="chip fail">xung đột</span> {run.conflicts} phần tử bị đọc trong cùng vòng lặp mà một luồng
                    khác ghi vào: đáp án sẽ phụ thuộc vào luồng nào chạy trước.
                  </>
                }
              >
                <span className="chip fail">conflict</span> {run.conflicts} element{run.conflicts > 1 ? 's' : ''} read in the same iteration that another thread
                writes {run.conflicts > 1 ? 'them' : 'it'}: the answer would depend on which thread goes first.
              </Translated>
            </p>
          ) : null}
          <p style={{ margin: '0 0 0.3rem' }}>
            <Translated
              vi={
                <>
                  <span className="mono">*output</span> = {Number.isNaN(run.output) ? '?' : run.output}; tổng là {run.total}.{' '}
                  {run.correct ? <span className="chip ok">tổng đúng</span> : <span className="chip fail">sai</span>}
                </>
              }
            >
              <span className="mono">*output</span> = {Number.isNaN(run.output) ? '?' : run.output}; the sum is {run.total}.{' '}
              {run.correct ? <span className="chip ok">right sum</span> : <span className="chip fail">wrong</span>}
            </Translated>
          </p>
          <p style={{ margin: 0 }}>
            {run.totalInLast ? (
              <span className="chip ok">
                <Translated vi={<>tổng nằm ở input[{N - 1}], phần tử cuối cùng</>}>
                  the total is in input[{N - 1}], the last element
                </Translated>
              </span>
            ) : (
              <span className="muted">
                <Translated
                  vi={
                    <>
                      Phần tử cuối, input[{N - 1}], kết thúc ở{' '}
                      {Number.isNaN(run.iterations[run.iterations.length - 1].after[N - 1]) ? '?' : run.iterations[run.iterations.length - 1].after[N - 1]},
                      không phải tổng.
                    </>
                  }
                >
                  The last element, input[{N - 1}], ends at {Number.isNaN(run.iterations[run.iterations.length - 1].after[N - 1]) ? '?' : run.iterations[run.iterations.length - 1].after[N - 1]}, not the total.
                </Translated>
              </span>
            )}
            {isMirror ? (
              <span className="t-ok">
                <Translated vi=" Đó là mẫu của bài tập 3."> That is exercise 3's pattern.</Translated>
              </span>
            ) : null}
          </p>
        </div>
      </div>
      <Code tabs={[{ label: tr('your kernel', 'kernel của bạn'), code: specCode(spec), tiers: TIERS }]} highlight={[2, 4, 5, 9]} />
      <div className="row" style={{ marginTop: '0.9rem' }}>
        <button onClick={() => setShowRef(!showRef)} aria-pressed={showRef}>
          <Translated vi={<>{showRef ? 'Ẩn' : 'Hiện'} đáp án của chúng tôi</>}>
            {showRef ? 'Hide' : 'Show'} our answer
          </Translated>
        </button>
      </div>
      {showRef && !missing(ref) ? (
        <Code
          tabs={[{ label: 's104_mirrored.cu', code: ref!, tiers: TIERS, status: STATUS.snippet }]}
          caption={
            <Translated
              vi={
                <>
                  Đáp án của chúng tôi, lưu ở <code>labs/ch10/snippets/s104_mirrored.cu</code>; sách không in đáp án nào. Một
                  unit test gắn các phần ở trên với tệp này, và chạy nó trên trình giả lập CPU của khóa học.
                </>
              }
            >
              <>
                Our answer, stored as <code>labs/ch10/snippets/s104_mirrored.cu</code>; the book prints none. A unit test ties
                the parts above to this file, and runs it on the course's CPU emulator.
              </>
            </Translated>
          }
        />
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS104: Record<string, Question> = {
  'ch10/10.4/q-ex1': {
    kind: 'numeric',
    id: 'ch10/10.4/q-ex1',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 1 của sách: kernel đơn giản (Fig. 10.5) trên 1024 giá trị, với warp rộng 32. Có bao nhiêu warp phân kỳ ở
            vòng lặp thứ 5?
          </>
        }
      >
        <>
          Book exercise 1: the simple kernel (Fig. 10.5) on 1024 values, with 32-wide warps. How many warps diverge in the
          5th iteration?
        </>
      </Translated>
    ),
    answer: 16,
    explain: (
      <Translated
        vi={
          <p>
            512 luồng tạo thành 16 warp. Vòng lặp thứ 5 có bước nhảy 16, nên các luồng hoạt động là bội của 16: lane 0 và 16
            của mọi warp. Hai lane hoạt động trên 32 ở mỗi warp trong 16 warp: cả 16 đều phân kỳ.
          </p>
        }
      >
        <p>
          512 threads make 16 warps. The 5th iteration has stride 16, so the active threads are the multiples of 16: lanes 0
          and 16 of every warp. Two active lanes out of 32 in each of the 16 warps: all 16 diverge.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.12, exercise 1, p. 249; answer ours',
    vi: { source: 'ch. 10 §10.12, bài tập 1, p. 249; đáp án của khóa học này' },
  },
  'ch10/10.4/q-ex2': {
    kind: 'numeric',
    id: 'ch10/10.4/q-ex2',
    prompt: (
      <Translated
        vi={<>Bài tập 2 của sách: kernel hội tụ (Fig. 10.8), cùng thiết lập. Có bao nhiêu warp phân kỳ ở vòng lặp thứ 5?</>}
      >
        <>Book exercise 2: the convergent kernel (Fig. 10.8), same setup. How many warps diverge in the 5th iteration?</>
      </Translated>
    ),
    answer: 0,
    explain: (
      <Translated
        vi={
          <p>
            Các bước nhảy là 512, 256, 128, 64, 32. Ở vòng lặp thứ 5, các luồng 0 đến 31 hoạt động: toàn bộ warp 0, và mọi warp
            khác hoàn toàn rảnh. Không warp nào phân kỳ. Phân kỳ bắt đầu ở vòng lặp thứ 6, chỉ trong warp 0.
          </p>
        }
      >
        <p>
          The strides go 512, 256, 128, 64, 32. In the 5th iteration threads 0 to 31 are active: all of warp 0, and every
          other warp entirely idle. No warp diverges. Divergence starts in the 6th iteration, in warp 0 alone.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.12, exercise 2, p. 249; answer ours',
    vi: { source: 'ch. 10 §10.12, bài tập 2, p. 249; đáp án của khóa học này' },
  },
  'ch10/10.4/q-eff-1024': {
    kind: 'numeric',
    id: 'ch10/10.4/q-eff-1024',
    prompt: (
      <Translated
        vi={
          <>
            Kernel hội tụ trên 1024 giá trị với warp rộng 32. Hiệu suất sử dụng tài nguyên thực thi của nó là bao nhiêu, tính
            theo phần trăm?
          </>
        }
      >
        <>The convergent kernel on 1024 values with 32-wide warps. What is its execution resource utilization efficiency, in percent?</>
      </Translated>
    ),
    answer: 89,
    tol: 0.02,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            16 warp. Khi còn từ 32 luồng hoạt động trở lên, các warp làm việc trọn vẹn: 16 + 8 + 4 + 2 + 1 = 31 lượt warp. Sau
            đó là năm vòng lặp với một warp phân kỳ. 36 × 32 = 1152 slot cho 1023 phép cộng: 89%. Hiệu suất tăng theo N, vì
            năm vòng lặp phân kỳ ở cuối là một chi phí cố định.
          </p>
        }
      >
        <p>
          16 warps. While 32 or more threads are active, whole warps work: 16 + 8 + 4 + 2 + 1 = 31 warp-iterations. Then five
          iterations with one divergent warp. 36 × 32 = 1152 slots for 1023 additions: 89%. The efficiency rises with N,
          because the five divergent iterations at the end are a fixed cost.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.4, p. 234 (the formula); the 1024 case is ours',
    vi: { source: 'ch. 10 §10.4, p. 234 (công thức); trường hợp 1024 là của khóa học này' },
  },
  'ch10/10.4/q-noncommutative': {
    kind: 'choice',
    id: 'ch10/10.4/q-noncommutative',
    prompt: (
      <Translated
        vi={<>Với toán tử nào, kernel hội tụ có thể cho đáp án khác kernel đơn giản, trong số học chính xác?</>}
      >
        <>For which operator could the convergent kernel give a different answer from the simple kernel, in exact arithmetic?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Max số nguyên.">Integer max.</Translated>,
        why: (
          <Translated vi="Max có tính giao hoán và kết hợp: thứ tự nào cũng cho cùng đáp án.">
            Max is commutative and associative: any order gives the same answer.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Tổng số nguyên (không tràn số).">Integer sum (with no overflow).</Translated>,
        why: <Translated vi="Có tính giao hoán và kết hợp.">Commutative and associative.</Translated>,
      },
      {
        label: <Translated vi="Tích số nguyên (không tràn số).">Integer product (with no overflow).</Translated>,
        why: <Translated vi="Có tính giao hoán và kết hợp.">Commutative and associative.</Translated>,
      },
      { label: <Translated vi="Nối chuỗi.">String concatenation.</Translated>, correct: true },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Kernel đơn giản kết hợp các phần tử kề nhau, nên các toán hạng giữ nguyên thứ tự ban đầu và chỉ có dấu ngoặc dịch
            chuyển: tính kết hợp là đủ. Kernel hội tụ kết hợp phần tử 0 với phần tử 8, phần tử 1 với phần tử 9, nên chính các
            toán hạng bị sắp xếp lại, và toán tử còn phải có tính giao hoán. Nối chuỗi có tính kết hợp nhưng không giao hoán:
            "ab" + "cd" không phải là "cd" + "ab". Phép cộng dấu phẩy động là trường hợp thường ngày mà thứ tự làm thay đổi các
            bit cuối; mục 10.2 đã cho thấy điều đó.
          </p>
        }
      >
        <p>
          The simple kernel combines neighbours, so the operands stay in their original order and only the brackets move:
          associativity is enough. The convergent kernel combines element 0 with element 8, element 1 with element 9, so the
          operands themselves are reordered, and the operator must also be commutative. Concatenation is associative but not
          commutative: "ab" + "cd" is not "cd" + "ab". Floating-point addition is the everyday case where the order changes
          the last bits; section 10.2 showed it.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.2, p. 225; §10.4, pp. 232–233', // i18n-ignore: citation only
  },
  'ch10/10.4/q-w64-slots': {
    kind: 'numeric',
    id: 'ch10/10.4/q-w64-slots',
    prompt: (
      <Translated
        vi={
          <>
            256 giá trị (128 luồng) trên một thiết bị có warp rộng 64. Kernel hội tụ dùng bao nhiêu slot luồng, theo cách tính
            của sách?
          </>
        }
      >
        <>
          256 values (128 threads) on a device whose warps are 64 wide. How many thread slots does the convergent kernel
          consume, in the book's accounting?
        </>
      </Translated>
    ),
    answer: 576,
    explain: (
      <Translated
        vi={
          <p>
            Hai warp 64. Bước nhảy 128: cả hai warp đầy. Bước nhảy 64: warp 0 đầy, warp 1 rảnh. Bước nhảy 32: warp 0 hoạt động
            một nửa, nên nó phân kỳ, sớm hơn một vòng lặp so với warp rộng 32. Sau đó 16, 8, 4, 2, 1: lại là warp 0. Tức là 2 +
            1 + 1 + 5 = 9 lượt warp, × 64 = 576 slot, hiệu suất 255/576 = 44%. Kernel đơn giản tốn 960. Mã không hề nhắc tới độ
            rộng; dù vậy hiệu suất của nó vẫn phụ thuộc vào độ rộng.
          </p>
        }
      >
        <p>
          Two warps of 64. Stride 128: both warps full. Stride 64: warp 0 full, warp 1 idle. Stride 32: warp 0 half active,
          so it diverges, one iteration earlier than with 32-wide warps. Then 16, 8, 4, 2, 1: warp 0 again. That is
          2 + 1 + 1 + 5 = 9 warp-iterations, × 64 = 576 slots, an efficiency of 255/576 = 44%. The simple kernel takes 960.
          The code never mentions the width; its efficiency depends on it all the same.
        </p>
      </Translated>
    ),
    source: 'ours, with the book\'s accounting (p. 231); widths: notes/today/ch05-facts.md B5',
    vi: { source: 'của khóa học này, theo cách tính của sách (p. 231); độ rộng: notes/today/ch05-facts.md B5' },
  },
  'ch10/10.4/q-ex6b': {
    kind: 'numeric',
    id: 'ch10/10.4/q-ex6b',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 6b của sách: chạy kernel hội tụ trên {'{'}6, 2, 7, 4, 5, 8, 3, 1{'}'}. Cuối cùng <code>input[1]</code>{' '}
            chứa giá trị gì?
          </>
        }
      >
        <>
          Book exercise 6b: run the convergent kernel on {'{'}6, 2, 7, 4, 5, 8, 3, 1{'}'}. What does <code>input[1]</code>{' '}
          hold at the end?
        </>
      </Translated>
    ),
    answer: 15,
    explain: (
      <Translated
        vi={
          <p>
            Các bước nhảy 4, 2, 1. Sau bước nhảy 4: {'{'}11, 10, 10, 5, 5, 8, 3, 1{'}'}. Sau bước nhảy 2: {'{'}21, 15, 10, 5,
            ...{'}'}. Sau bước nhảy 1, chỉ <code>input[0]</code> thay đổi, thành 36. Đặt dữ liệu của "Lật ngược nó" là bài tập
            6 với các phần của Fig. 10.8 để xem mọi hàng. Đáp án là của chúng tôi; sách không in đáp án nào.
          </p>
        }
      >
        <p>
          Strides 4, 2, 1. After stride 4: {'{'}11, 10, 10, 5, 5, 8, 3, 1{'}'}. After stride 2: {'{'}21, 15, 10, 5, ...{'}'}.
          After stride 1, only <code>input[0]</code> changes, to 36. Set the data of "Mirror it" to exercise 6 with Fig.
          10.8's parts to see every row. The answer is ours; the book prints none.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.12, exercise 6b, p. 250; answer ours',
    vi: { source: 'ch. 10 §10.12, bài tập 6b, p. 250; đáp án của khóa học này' },
  },
  'ch10/10.4/q-ex3': {
    kind: 'choice',
    id: 'ch10/10.4/q-ex3',
    prompt: (
      <Translated
        vi={
          <>
            Trong kernel lật ngược của bài tập 3, luồng t sở hữu <code>i = threadIdx.x + blockDim.x</code> và bước nhảy chia
            đôi từ <code>blockDim.x</code> xuống 1. Phép kiểm tra hoạt động nào đúng?
          </>
        }
      >
        <>
          In the mirrored kernel of exercise 3, thread t owns <code>i = threadIdx.x + blockDim.x</code> and the stride halves
          from <code>blockDim.x</code> to 1. Which active test is right?
        </>
      </Translated>
    ),
    options: [
      {
        label: <code>threadIdx.x &lt; stride</code>,
        why: (
          <Translated vi="Cách đó giữ các luồng thấp nhất hoạt động, những luồng sở hữu phần giữa của đầu vào, không phải phần cuối.">
            That keeps the lowest threads active, which own the middle of the input, not its end.
          </Translated>
        ),
      },
      {
        label: <code>threadIdx.x &gt;= stride</code>,
        why: (
          <Translated vi="Ở vòng lặp đầu (stride = blockDim.x) không luồng nào vượt qua nó.">
            In the first iteration (stride = blockDim.x) no thread passes it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated
            vi={
              <>
                <code>threadIdx.x &gt;= blockDim.x - stride</code>, cộng <code>input[i - stride]</code>
              </>
            }
          >
            <><code>threadIdx.x &gt;= blockDim.x - stride</code>, adding <code>input[i - stride]</code></>
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated
            vi={
              <>
                <code>threadIdx.x &gt;= blockDim.x - stride</code>, cộng <code>input[i + stride]</code>
              </>
            }
          >
            <><code>threadIdx.x &gt;= blockDim.x - stride</code>, adding <code>input[i + stride]</code></>
          </Translated>
        ),
        why: <Translated vi="i + stride chạy quá cuối đầu vào.">i + stride runs past the end of the input.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Các luồng hoạt động phải là <code>stride</code> luồng cuối cùng, để công việc dồn dần về phần tử cuối, và mỗi luồng
            cộng phần tử nằm dưới phần tử của mình <code>stride</code> vị trí. Với 8 luồng: bước nhảy 8 lấp các phần tử 8 đến
            15 từ 0 đến 7; bước nhảy 4 lấp 12 đến 15 từ 8 đến 11; bước nhảy 2 lấp 14 và 15; bước nhảy 1 lấp 15. Luồng cuối khi
            đó ghi <code>input[2*blockDim.x - 1]</code> vào <code>*output</code>.
          </p>
        }
      >
        <p>
          The active threads must be the last <code>stride</code> ones, so that the work closes in on the last element, and
          each adds the element <code>stride</code> below its own. With 8 threads: stride 8 fills elements 8 to 15 from 0 to
          7; stride 4 fills 12 to 15 from 8 to 11; stride 2 fills 14 and 15; stride 1 fills 15. The last thread then writes{' '}
          <code>input[2*blockDim.x - 1]</code> to <code>*output</code>.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.12, exercise 3, p. 249; answer ours (labs/ch10/snippets/s104_mirrored.cu)',
    vi: { source: 'ch. 10 §10.12, bài tập 3, p. 249; đáp án của khóa học này (labs/ch10/snippets/s104_mirrored.cu)' },
  },
};

// ---------------------------------------------------------------- the page
export function S104() {
  const cu = useLabFile('ch10/shared/inplace_sum.cu');
  const metal = useLabFile('ch10/shared/inplace_sum.metal');
  const status = useLabStatus(LAB10_1);
  const simpleCu = !missing(cu) ? functionSource(cu!, 'SimpleSumReductionKernel') : null;
  const convCu = !missing(cu) ? functionSource(cu!, 'ConvergentSumReductionKernel') : null;
  const simpleMt = !missing(metal) ? functionSource(metal!, 'simple_sum') : null;
  const convMt = !missing(metal) ? functionSource(metal!, 'convergent_sum') : null;
  const cuDiff = simpleCu && convCu ? changedLines(simpleCu.text, convCu.text, 'SimpleSumReductionKernel', 'ConvergentSumReductionKernel') : [];
  const mtDiff = simpleMt && convMt ? changedLines(simpleMt.text, convMt.text, 'simple_sum', 'convergent_sum') : [];
  const c256 = iterations('convergent', 256, 32);
  return (
    <>
      <Translated
        vi={
          <p>
            Hãy nhìn xem những luồng nào làm việc ở mỗi vòng lặp của kernel đơn giản. Ở vòng đầu, tất cả. Ở vòng hai, cứ cách
            một luồng: một nửa số lane của mọi warp. Tới vòng sáu, một lane trên 32. Một warp chỉ có một lane hoạt động vẫn
            chiếm trọn tài nguyên thực thi của một warp, và 31 lane của nó không tạo ra gì. Sách đo điều này bằng{' '}
            <b>hiệu suất sử dụng tài nguyên thực thi</b>: phần tài nguyên đã tiêu tốn thực sự tạo ra kết quả.
          </p>
        }
      >
        <p>
          Look at which threads work in each iteration of the simple kernel. In the first, all of them. In the second, every
          other thread: half the lanes of every warp. By the sixth, one lane in 32. A warp with one active lane still
          occupies a full warp's worth of execution resources, and 31 of its lanes produce nothing. The book measures this
          with the <b>execution resource utilization efficiency</b>: the share of the resources consumed that produce a
          result.
        </p>
      </Translated>
      <Source section="10.4" pages="230" />

      <h2>
        <Translated vi="Đếm phần lãng phí ở 256 giá trị">Counting the waste at 256 values</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lấy 256 giá trị: 128 luồng, 4 warp 32. Trong sáu vòng lặp đầu, mọi warp đều có ít nhất một luồng hoạt động, tất cả
            cùng một mẫu. Ở vòng thứ bảy chỉ các luồng 0 và 64 làm việc, nên warp 0 và 2 chạy với mỗi warp một lane còn warp 1
            và 3 rảnh; ở vòng thứ tám chỉ warp 0 chạy. Tính cho mỗi warp hoạt động đủ 32 lane và vòng lặp tiêu tốn (4 × 6 + 2 +
            1) × 32 = 864 slot luồng. Nó tạo ra 255 kết quả, mỗi phép cộng một kết quả, nên hiệu suất là 255/864, khoảng 0.30.
          </p>
        }
      >
        <p>
          Take 256 values: 128 threads, 4 warps of 32. For the first six iterations every warp has at least one active
          thread, all with the same pattern. In the seventh only threads 0 and 64 work, so warps 0 and 2 run with one lane
          each and warps 1 and 3 are idle; in the eighth only warp 0 runs. Charge every active warp its 32 lanes and the loop
          consumes (4 × 6 + 2 + 1) × 32 = 864 thread slots. It produces 255 results, one per addition, so the efficiency is
          255/864, about 0.30.
        </p>
      </Translated>
      <Source section="10.4" pages="230–231" />
      <BookTable />
      <Translated
        vi={
          <p>
            Các cột bên phải thuộc về cách sửa. Nó có đúng cùng số luồng hoạt động ở mọi vòng lặp, và ít hơn một nửa số warp
            hoạt động.
          </p>
        }
      >
        <p>
          The right-hand columns belong to the fix. It has exactly the same number of active threads in every iteration, and
          fewer than half the active warps.
        </p>
      </Translated>

      <h2>
        <Translated vi="Giữ các luồng hoạt động ở cạnh nhau">Keep the active threads together</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Vấn đề là các tổng bộ phận, và do đó các luồng sở hữu chúng, trôi xa nhau hơn sau mỗi vòng lặp. Hãy đảo ngược lại.
            Cho các luồng sở hữu nửa đầu của đầu vào (luồng t sở hữu phần tử t), bắt đầu bước nhảy từ <code>blockDim.x</code>{' '}
            và chia đôi nó. Ở vòng đầu, với 16 giá trị, luồng 0 cộng phần tử 8 vào phần tử 0, luồng 1 cộng phần tử 9 vào phần
            tử 1, và cứ thế; ở mỗi vòng sau, nửa thấp của các luồng đang hoạt động tiếp tục hoạt động, mỗi luồng vươn xa đúng
            bằng số luồng hoạt động. Phép kiểm tra hoạt động trở thành <code>threadIdx.x &lt; stride</code>. Tức là ba dòng
            thay đổi.
          </p>
        }
      >
        <p>
          The trouble is that the partial sums, and so the threads that own them, drift further apart in every iteration.
          Turn it around. Let the threads own the first half of the input (thread t owns element t), start the stride at{' '}
          <code>blockDim.x</code> and halve it. In the first iteration, with 16 values, thread 0 adds element 8 into element
          0, thread 1 adds element 9 into element 1, and so on; in each later iteration the lowest half of the threads that
          were active stay active, each reaching as far as there are active threads. The active test becomes{' '}
          <code>threadIdx.x &lt; stride</code>. That is three changed lines.
        </p>
      </Translated>
      <Source section="10.4" pages="231–233" figs={tr('Figs. 10.7, 10.8', 'Hình 10.7, 10.8')} />
      {convCu ? (
        <Code
          tabs={[
            { label: tr('CUDA (lab 10.1)', 'CUDA (lab 10.1)'), code: convCu.text, tiers: TIERS, status: status.cu, highlight: cuDiff },
            ...(convMt ? [{ label: tr('Metal (lab 10.1)', 'Metal (lab 10.1)'), code: convMt.text, tiers: TIERS, status: status.metal, highlight: mtDiff }] : []),
          ]}
          caption={
            <Translated
              vi={
                <>
                  <code>ConvergentSumReductionKernel</code> từ <code>labs/ch10/shared/inplace_sum.cu</code> (và{' '}
                  <code>convergent_sum</code> từ <code>inplace_sum.metal</code>). Các dòng được đánh dấu là những dòng khác với
                  kernel đơn giản ở mục 10.3; mọi thứ khác, kể cả độ lệch hàng của lab 10.1, đều không đổi.
                </>
              }
            >
              <>
                <code>ConvergentSumReductionKernel</code> from <code>labs/ch10/shared/inplace_sum.cu</code> (and{' '}
                <code>convergent_sum</code> from <code>inplace_sum.metal</code>). The marked lines are the ones that differ
                from the simple kernel in section 10.3; everything else, including lab 10.1's row offset, is unchanged.
              </>
            </Translated>
          }
        />
      ) : (
        <p className="muted">
          <Translated vi={<>Kernel được nạp từ <code>labs/ch10/shared/inplace_sum.cu</code>.</>}>
            <>The kernel is loaded from <code>labs/ch10/shared/inplace_sum.cu</code>.</>
          </Translated>
        </p>
      )}
      <Translated
        vi={
          <p>
            Cách này sắp xếp lại nhiều hơn dấu ngoặc. Giờ cây kết hợp phần tử 0 với phần tử 8 và phần tử 1 với phần tử 9, nên
            các toán hạng gặp nhau theo thứ tự x0, x8, x1, x9, và cứ thế. Lời cảnh báo của mục 10.2 tới hạn: toán tử phải có
            tính giao hoán cũng như tính kết hợp.
          </p>
        }
      >
        <p>
          This reorders more than the brackets. The tree now combines element 0 with element 8 and element 1 with element 9,
          so the operands meet in the order x0, x8, x1, x9, and so on. Section 10.2's warning comes due: the operator must be
          commutative as well as associative.
        </p>
      </Translated>
      <Source section="10.4" pages="232–233" />
      <Ask q={questionsS104['ch10/10.4/q-noncommutative']} />

      <h2>
        <Translated vi="Đếm thử">Count it</Translated>
      </h2>
      <Predict
        id="ch10/10.4/p-efficiency"
        prompt={
          <Translated
            vi={
              <>
                Trước khi bạn chạy từng bước kernel hội tụ bên dưới: trên 256 giá trị với warp rộng 32, bạn dự đoán hiệu suất
                bao nhiêu, theo phần trăm? (Hiệu suất của kernel đơn giản là 30%.)
              </>
            }
          >
            <>
              Before you step the convergent kernel below: on 256 values with 32-wide warps, what efficiency do you expect, in
              percent? (The simple kernel's is 30%.)
            </>
          </Translated>
        }
        unit="%"
      >
        {(v) => (
          <Translated
            vi={
              <p data-readout="p-efficiency">
                Mô hình đếm {c256.slots} slot cho {c256.results} kết quả: {pct(c256.efficiency)}.{' '}
                {Math.abs(v - c256.efficiency * 100) <= 5
                  ? 'Đúng như bạn dự đoán.'
                  : v < c256.efficiency * 100
                    ? 'Cao hơn bạn dự đoán: cả warp cùng rảnh một lúc, nên các luồng rảnh không còn tốn gì.'
                    : 'Thấp hơn bạn dự đoán: năm vòng lặp cuối vẫn chạy một warp với vài lane hoạt động.'}{' '}
                Giờ hãy chạy từng bước bên dưới và xem các slot đi đâu.
              </p>
            }
          >
            <p data-readout="p-efficiency">
              The model counts {c256.slots} slots for {c256.results} results: {pct(c256.efficiency)}.{' '}
              {Math.abs(v - c256.efficiency * 100) <= 5
                ? 'That is what you expected.'
                : v < c256.efficiency * 100
                  ? 'Higher than you expected: whole warps go idle together, so idle threads stop costing anything.'
                  : 'Lower than you expected: the last five iterations still run one warp with a few active lanes.'}{' '}
              Now step it below and see where the slots go.
            </p>
          </Translated>
        )}
      </Predict>
      <Simplification>
        <Translated
          vi={
            <p>
              Bản đồ đếm lane, không đếm thời gian: một warp có bất kỳ lane nào hoạt động bị tính đủ độ rộng cho vòng lặp đó,
              theo cách tính của sách. Một vòng lặp phân kỳ tốn bao nhiêu thời gian là việc của lab 10.1 đo. Không có gì ở đây
              phụ thuộc vào dữ liệu.
            </p>
          }
        >
          <p>
            The map counts lanes, not time: a warp with any active lane is charged its full width for the iteration, which is
            the book's accounting. What a divergent iteration costs in time is for lab 10.1 to measure. Nothing here depends
            on data.
          </p>
        </Translated>
      </Simplification>
      <DivergenceLab />

      <h2>
        <Translated vi="Các luồng hoạt động nằm ở đâu">Where the active threads are</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mỗi vòng lặp có số luồng hoạt động như trước; điều thay đổi là chúng nằm ở đâu. Ở vòng thứ hai, các luồng 0 đến 63
            làm việc còn 64 đến 127 thì không, nên warp 0 và 1 hoạt động trọn vẹn còn warp 2 và 3 hoàn toàn rảnh. Không warp
            nào phân kỳ. Điều đó đúng chừng nào còn ít nhất 32 luồng hoạt động. Ở năm vòng lặp cuối (16, 8, 4, 2 và 1 luồng
            hoạt động), warp còn lại duy nhất phân kỳ. Tổng cộng (4 + 2 + 1 + 5) × 32 = 384 slot, và 255/384 = 66%. Vì các warp
            của một khối thay phiên nhau trên phần cứng có tài nguyên hạn chế, ít slot hơn cũng có nghĩa là ít thời gian hơn.
          </p>
        }
      >
        <p>
          Each iteration has as many active threads as before; what changed is where they are. In the second iteration
          threads 0 to 63 work and 64 to 127 do not, so warps 0 and 1 are fully active and warps 2 and 3 entirely idle. No
          warp diverges. That holds as long as at least 32 threads are active. In the last five iterations (16, 8, 4, 2 and 1
          active threads) the one remaining warp diverges. In total (4 + 2 + 1 + 5) × 32 = 384 slots, and 255/384 = 66%.
          Because the warps of a block take turns on hardware with limited resources, fewer slots also means less time.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="10.4" pages="233–234" />
        <Aside>
          <Translated
            vi={
              <p>
                Ở p. 234, sách so sánh 384 với "736, the resources consumed by the kernel in Fig. 10.5", và nói các vòng lặp phân
                kỳ giảm "from ten to five". Chính công thức của sách ở p. 231 cho 864, và ở kích thước này mã phân kỳ trong bảy
                trên tám vòng lặp. Bản đồ warp cho thấy cả hai con số đếm.
              </p>
            }
          >
            <p>
              On p. 234 the book compares 384 with "736, the resources consumed by the kernel in Fig. 10.5", and says the
              divergent iterations drop "from ten to five". Its own formula on p. 231 gives 864, and the code diverges in seven of
              its eight iterations at this size. The warp map shows both counts.
            </p>
          </Translated>
        </Aside>
      </NoteGroup>
      <Translated
        vi={
          <p>
            Bài học của sách: một thay đổi ba dòng về việc luồng nào làm gì, với tác động lớn, và bạn chỉ có thể thực hiện nó
            một cách tự tin nếu bạn có hình dung về cách một warp thực thi.
          </p>
        }
      >
        <p>
          The book's moral: a change of three lines in which thread does what, with a large effect, and you can only make it
          with confidence if you have the picture of how a warp executes.
        </p>
      </Translated>
      <Source section="10.4" pages="234" />
      <Ask q={questionsS104['ch10/10.4/q-ex1']} label={tr('Work it out, then check it on the map', 'Tự tính, rồi kiểm tra trên bản đồ')} />
      <Ask q={questionsS104['ch10/10.4/q-ex2']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS104['ch10/10.4/q-eff-1024']} label={tr('Work it out', 'Tự tính')} />

      <h2>
        <Translated vi="Độ rộng không nằm trong mã">The width is not in the code</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Không kernel nào nêu tên 32, nhưng hiệu suất của cả hai đều phụ thuộc vào độ rộng warp. Trên một thiết bị có warp
            rộng 64, kernel hội tụ phân kỳ sớm hơn một vòng lặp: khi 32 luồng hoạt động, chúng chỉ lấp một nửa warp 0. Kernel
            đơn giản cũng thiệt. Đặt độ rộng 64 trong bản đồ và so sánh.
          </p>
        }
      >
        <p>
          Neither kernel names 32, but both efficiencies depend on the warp width. On a device with 64-wide warps the
          convergent kernel diverges one iteration earlier: when 32 threads are active they fill only half of warp 0. The
          simple kernel loses as well. Set the width to 64 in the map and compare.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch05-facts B5; ch04-facts C2)">{/* i18n-ignore: date and citation only */}
        <Translated
          vi={
            <p>
              NVIDIA: 32. AMD: 64 trên gfx9, 32 trên gfx10 trở về sau; đừng giả định bên nào. Metal: không có tài liệu, đọc từ
              pipeline.
            </p>
          }
        >
          <p>
            NVIDIA: 32. AMD: 64 on gfx9, 32 on gfx10 and later; do not assume either. Metal: undocumented, read from the
            pipeline.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS104['ch10/10.4/q-w64-slots']} />
      <Translated
        vi={
          <p>
            Một vòng lặp phân kỳ tốn bao nhiêu thời gian là câu hỏi cho thiết bị của bạn, không phải cho cách tính này.{' '}
            <a href={href('ch04', 'labs')}>Lab 4.2</a> của chương 4 đã đo riêng sự phân kỳ;{' '}
            <a href={href('ch10', 'labs')}>lab 10.1</a> đo thời gian kernel đơn giản so với kernel hội tụ.
          </p>
        }
      >
        <p>
          What a divergent iteration costs in time is a question for your device, not for this accounting. Chapter 4's{' '}
          <a href={href('ch04', 'labs')}>lab 4.2</a> measured divergence on its own; <a href={href('ch10', 'labs')}>lab 10.1</a>{' '}
          times the simple kernel against the convergent one.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Các lựa chọn độ rộng là 32 của sách, 64 mà AMD ghi cho gfx9, và bất kỳ độ rộng nào mà thiết bị CUDA hoặc HIP của
              riêng bạn đã báo cáo trong <code>reports/runs</code>.
            </p>
          }
        >
          <p>
            The width choices are the book's 32, the 64 AMD documents for gfx9, and any width your own CUDA or HIP device
            reported in <code>reports/runs</code>.
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Lật ngược nó">Mirror it</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bài tập 3 của sách yêu cầu cùng ý tưởng đó nhưng đảo ngược: các luồng sở hữu nửa sau của đầu vào, và tổng kết thúc
            ở phần tử cuối cùng. Bắt đầu từ các phần của Fig. 10.8, đã được chọn sẵn bên dưới, và thay đổi chúng cho tới khi
            tổng rơi vào phần tử cuối mà không có xung đột nào. Vòng lặp và rào chắn giữ nguyên.
          </p>
        }
      >
        <p>
          Book exercise 3 asks for the same idea turned around: the threads own the second half of the input, and the total
          ends in the last element. Start from Fig. 10.8's parts, which are selected below, and change them until the total
          lands in the last element with no conflict. The loop and the barrier stay as they are.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="10.12" pages="249" figs={tr('exercise 3 and its figure', 'bài tập 3 và hình của nó')} />
        <Simplification>
          <Translated
            vi={
              <p>
                Mỗi vòng lặp được áp dụng như thể mọi luồng hoạt động đều đọc trước khi bất kỳ luồng nào ghi, và một giá trị bị
                đọc trong cùng vòng lặp mà một luồng khác ghi vào được đánh dấu là xung đột thay vì cho chạy đua. Một khối,
                không warp, không tính thời gian.
              </p>
            }
          >
            <p>
              Each iteration is applied as if every active thread read before any wrote, and a value read in the same
              iteration that another thread writes is flagged as a conflict rather than raced. One block, no warps, no timing.
            </p>
          </Translated>
        </Simplification>
      </NoteGroup>
      <MirrorIt />
      <Ask q={questionsS104['ch10/10.4/q-ex3']} />
      <Ask q={questionsS104['ch10/10.4/q-ex6b']} label={tr('Work it out, then check it in Mirror it', 'Tự tính, rồi kiểm tra trong Lật ngược nó')} />
    </>
  );
}
