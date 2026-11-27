import { useMemo, useState, type KeyboardEvent, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import {
  BOOK_BLOCK_MULTIPLE,
  BOOK_MAX_BLOCK,
  MAPPING_KINDS,
  counts,
  coverage,
  makeData,
  mapping,
  snapBlock,
  span,
  stepKernel,
  working,
  type Line,
  type MappingKind,
  type ThreadState,
} from '../../sim/grid1d';
import { CALLERS, EXECUTOR_TEXT, QUALIFIER_TEXT, functionSource, legalCall, type Qualifiers } from '../../sim/toolchain';
import { LoopToGridScene } from './LoopToGridScene';

const LAB = 'ch02/lab1_vecadd';
const KERNEL_TIERS: Record<string, Tier> = { A: 'global', B: 'global', C: 'global', i: 'reg' };
const num = (v: number) => v.toLocaleString('en-US');
const plural = (n: number, one: string, many = one + 's') => `${num(n)} ${n === 1 ? one : many}`;
const onKey = (run: () => void) => (e: KeyboardEvent) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    run();
  }
};

// The three source lines the stepper walks through, found in the lab's kernel by
// their text (S25.snippets.test.ts checks that each occurs exactly once there).
const LINE_MARKS = ['int i =', 'if (i < n)', 'C[i] = A[i] + B[i];'];
// What the stepper calls those statements. It names them rather than numbering them,
// because the listing beside it numbers every line of the function, comments included.
const STATEMENTS = (): string[] => [tr('the index line', 'dòng chỉ số'), tr('the guard', 'phép kiểm tra biên'), tr('the addition', 'phép cộng')];

// Three lines of llama.cpp's scale kernel (ggml/src/ggml-cuda/scale.cu), as quoted in
// notes/today/ch02-facts.md, item D1. S25.snippets.test.ts ties this copy to the facts file.
const LLAMA_SCALE = `const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= k) { return; }
dst[i_dst] = ggml_cuda_cast<T>(scale * ggml_cuda_cast<float>(x[i_src]) + bias);`;
// AMD's own example kernel, as quoted in the same file, item B4.
const AMD_INDEX = `const std::size_t index = threadIdx.x + blockDim.x * blockIdx.x;`;
const AMD_GUARD = `if(index < size)`;

// ---------------------------------------------------------------- the kernel, from the lab
// The same words as the labs page (LabKit's solutionTabStatus), from this machine's runs.
function useKernelStatus() {
  const { reports } = useReports();
  return { cu: solutionTabStatus(reports, LAB, 'kernel.cu'), metal: solutionTabStatus(reports, LAB, 'kernel.metal') };
}

function KernelListing({ cu, metal }: { cu: string | null; metal: string | null }) {
  const status = useKernelStatus();
  const k = cu ? functionSource(cu, 'vecAddKernel') : null;
  const m = metal ? functionSource(metal, 'vec_add') : null;
  if (!k || !m)
    return (
      <Translated vi={<p className="muted">Đang tải kernel từ <code>labs/{LAB}/solution/</code>.</p>}>
        <p className="muted">Loading the kernel from <code>labs/{LAB}/solution/</code>.</p>
      </Translated>
    );
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: k.text, tiers: KERNEL_TIERS, status: status.cu, firstLine: k.firstLine, source: <code>labs/{LAB}/solution/kernel.cu</code> },
        { label: 'Metal', code: m.text, tiers: KERNEL_TIERS, status: status.metal, firstLine: m.firstLine, source: <code>labs/{LAB}/solution/kernel.metal</code> },
      ]}
      caption={
        <Translated
          vi={
            <>
              Hàm kernel trong lời giải tham chiếu (solution) của lab 2.1, đánh số dòng như trong mỗi tệp. Kernel CUDA
              là Fig. 2.10 của sách với <code>const</code> được thêm vào hai đầu vào. Hãy thử làm lab trước khi nghiên
              cứu phần này.
            </>
          }
        >
          <>
            The kernel function of lab 2.1's reference solution, numbered as the lines are in each file. The CUDA kernel
            is the book's Fig. 2.10 with <code>const</code> added to the two inputs. Try the lab before you study this.
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- interactive A: run the kernel on every thread
type Pick = { kind: 'thread'; g: number } | { kind: 'elem'; e: number };

const S_CW = 34;
const S_GAP = 14;
const S_X0 = 34;
const S_TY = 24; // thread boxes
const S_TH = 44;
const S_AY = 96; // first array row
const S_RH = 28;

/** What one thread is and has done so far, in words. */
function threadFacts(t: ThreadState, n: number, B: number, line: Line, guard: boolean): { label: string; value: ReactNode; cls?: string }[] {
  const out: { label: string; value: ReactNode; cls?: string }[] = [
    { label: tr('blockIdx.x (built-in)', 'blockIdx.x (có sẵn)'), value: t.block },
    { label: tr('threadIdx.x (built-in)', 'threadIdx.x (có sẵn)'), value: t.thread },
    { label: tr('blockDim.x (built-in)', 'blockDim.x (có sẵn)'), value: B },
    { label: tr('i (private)', 'i (riêng)'), value: t.i == null ? tr('not computed yet', 'chưa được tính') : `${working(t.block, t.thread, B)}`, cls: t.i == null ? 'muted' : 't-reg' },
  ];
  if (line >= 2)
    out.push({
      label: 'guard', // i18n-ignore: internal key, the reader sees "The guard:" instead
      value: !guard ? tr('removed, so nothing is tested', 'đã bị bỏ, nên không có gì được kiểm tra') : t.pass ? tr(`${t.i} < ${n}, so it passes`, `${t.i} < ${n}, nên nó vượt qua`) : tr(`${t.i} < ${n} is false, so the thread stops here`, `${t.i} < ${n} là sai, nên luồng dừng ở đây`),
      cls: guard && !t.pass ? 'muted' : undefined,
    });
  if (line >= 3 && t.store) {
    const ok = t.store.inRange;
    out.push({
      label: 'addition', // i18n-ignore: internal key, the reader sees "The addition:" instead
      value: ok
        ? tr(`loads A[${t.i}] = ${t.loads[0].value} and B[${t.i}] = ${t.loads[1].value}, stores ${t.store.value} in C[${t.i}]`, `nạp A[${t.i}] = ${t.loads[0].value} và B[${t.i}] = ${t.loads[1].value}, lưu ${t.store.value} vào C[${t.i}]`)
        : tr(`loads A[${t.i}] and B[${t.i}], stores to C[${t.i}]: all three are past the end of the arrays`, `nạp A[${t.i}] và B[${t.i}], lưu vào C[${t.i}]: cả ba đều nằm quá cuối mảng`),
      cls: ok ? undefined : 't-fail',
    });
  }
  return out;
}

function KernelStepper({ cu }: { cu: string | null }) {
  const [n, setN] = useState(10);
  const [B, setB] = useState(4);
  const [guard, setGuard] = useState(true);
  const [line, setLine] = useState<Line>(0);
  const [picked, setPicked] = useState<Pick>({ kind: 'thread', g: 6 });

  const cfg = useMemo(() => ({ n, blockDim: B, guard }), [n, B, guard]);
  const data = useMemo(() => makeData(n), [n]);
  const st = useMemo(() => stepKernel(cfg, line, data), [cfg, line, data]);
  const c = counts(cfg);
  const cov = useMemo(() => coverage(cfg), [cfg]);
  const idleI = cov.idleThreads.map((t) => t.i);
  const pick: Pick = picked.kind === 'thread' ? { kind: 'thread', g: Math.min(picked.g, c.launched - 1) } : { kind: 'elem', e: Math.min(picked.e, n - 1) };
  const selThread = pick.kind === 'thread' ? st.threads[pick.g] : st.threads[cov.writers[pick.e][0]];
  const stray = new Set(st.outOfRange);

  const kernel = cu ? functionSource(cu, 'vecAddKernel') : null;
  const kernelLines = kernel ? kernel.text.split('\n') : [];
  const lineNo = (k: number) => kernelLines.findIndex((l) => l.includes(LINE_MARKS[k - 1])) + 1;

  const x = (g: number) => S_X0 + g * S_CW + Math.floor(g / B) * S_GAP;
  const width = x(c.launched - 1) + S_CW + 12;
  const height = S_AY + 3 * S_RH + 18;

  let said: ReactNode;
  let bad = false;
  if (line === 0)
    said = (
      <Translated
        vi={
          <>
            Chưa có gì chạy. Lệnh khởi chạy đã tạo {num(c.blocks)} khối, mỗi khối {B} luồng: {num(c.launched)} luồng{' '}
            cho {num(n)} phần tử. Mọi luồng sắp chạy cùng ba câu lệnh: dòng chỉ số, phép kiểm tra biên và phép cộng. Hãy
            nhấn Bước.
          </>
        }
      >
        <>
          Nothing has run yet. The launch made {plural(c.blocks, 'block')} of {B} threads: {plural(c.launched, 'thread')}{' '}
          for {plural(n, 'element')}. Every thread is about to run the same three statements: the index line, the guard
          and the addition. Press Step.
        </>
      </Translated>
    );
  else if (line === 1)
    said = (
      <Translated
        vi={
          <>
            Dòng chỉ số. Mỗi luồng tự tính <code>i</code> của riêng nó từ hai số có sẵn. Với luồng {selThread.thread} của
            khối {selThread.block}: {working(selThread.block, selThread.thread, B)}. Có {num(c.launched)} luồng và{' '}
            {num(c.launched)} bản <code>i</code> riêng biệt.
          </>
        }
      >
        <>
          The index line. Every thread computes its own <code>i</code> from two built-in numbers. For thread {selThread.thread} of
          block {selThread.block}: {working(selThread.block, selThread.thread, B)}. There are {num(c.launched)} threads
          and {num(c.launched)} separate copies of <code>i</code>.
        </>
      </Translated>
    );
  else if (line === 2)
    said = !guard ? (
      <Translated vi={<>Phép kiểm tra biên đã bị bỏ, nên không luồng nào kiểm tra <code>i</code> của nó. Cả {num(c.launched)} luồng đi tiếp đến phép cộng.</>}>
        <>The guard has been removed, so no thread tests its <code>i</code>. All {num(c.launched)} go on to the addition.</>
      </Translated>
    ) : c.idle ? (
      <Translated
        vi={
          <>
            Phép kiểm tra biên. Mọi luồng so <code>i</code> của nó với n = {n}. {num(c.working)} luồng vượt qua.{' '}
            {num(c.idle)} luồng với i = {span(idleI)} không vượt qua và không làm gì thêm.
          </>
        }
      >
        <>
          The guard. Every thread tests its <code>i</code> against n = {n}. {plural(c.working, 'thread')} pass.{' '}
          {plural(c.idle, 'thread')} with i = {span(idleI)} {c.idle === 1 ? 'fails' : 'fail'} and {c.idle === 1 ? 'does' : 'do'} nothing more.
        </>
      </Translated>
    ) : (
      <Translated
        vi={
          <>
            Phép kiểm tra biên. Mọi luồng so <code>i</code> của nó với n = {n}, và cả {num(c.launched)} đều vượt qua: lần
            khởi chạy này có số luồng đúng bằng số phần tử.
          </>
        }
      >
        <>
          The guard. Every thread tests its <code>i</code> against n = {n}, and all {num(c.launched)} pass: this launch has
          exactly as many threads as elements.
        </>
      </Translated>
    );
  else if (st.outOfRange.length) {
    bad = true;
    said = (
      <Translated
        vi={
          <>
            Phép cộng, không có kiểm tra biên. {num(c.working)} luồng nạp <code>A[i]</code> và <code>B[i]</code> rồi lưu
            tổng vào <code>C[i]</code>. {c.idle} luồng còn lại làm y như vậy với i = {span(st.outOfRange)}:{' '}
            <b className="t-fail">{num(3 * st.outOfRange.length)} lần truy cập ra ngoài mảng</b>, mỗi luồng hai lần nạp
            và một lần lưu.
          </>
        }
      >
        <>
          The addition, with no guard. {plural(c.working, 'thread')} load <code>A[i]</code> and <code>B[i]</code> and store the
          sum in <code>C[i]</code>. The other {c.idle} do the same with i = {span(st.outOfRange)}:{' '}
          <b className="t-fail">{plural(3 * st.outOfRange.length, 'access', 'accesses')} outside the arrays</b>, two
          loads and one store each.
        </>
      </Translated>
    );
  } else
    said = (
      <Translated
        vi={
          <>
            Phép cộng. Mỗi luồng trong {num(c.working)} luồng đang hoạt động nạp <code>A[i]</code> và <code>B[i]</code> rồi
            lưu tổng vào <code>C[i]</code>. Mọi phần tử của C được ghi một lần, mỗi phần tử bởi một luồng khác nhau.
            {!guard ? ' Với số luồng đúng bằng số phần tử, việc thiếu phép kiểm tra biên không gây hại. Hãy đổi n đi 1.' : ''}
          </>
        }
      >
        <>
          The addition. Each of the {num(c.working)} live threads loads <code>A[i]</code> and <code>B[i]</code> and stores the
          sum in <code>C[i]</code>. Every element of C was written once, each by a different thread.
          {!guard ? ' With exactly as many threads as elements the missing guard did no harm. Change n by 1.' : ''}
        </>
      </Translated>
    );

  const threadState = (t: ThreadState): 'new' | 'has-i' | 'off' | 'stray' | 'done' => {
    if (t.i == null) return 'new';
    if (t.pass === false) return 'off';
    if (t.store) return t.store.inRange ? 'done' : 'stray';
    return 'has-i';
  };

  return (
    <Bench
      title={tr('Run the kernel on every thread', 'Chạy kernel trên mọi luồng')}
      task={tr('Step through the three statements. Before the guard, say which threads will stop. Then switch the guard off and step again.', 'Chạy từng bước qua ba câu lệnh. Trước phép kiểm tra biên, hãy nói những luồng nào sẽ dừng. Rồi tắt phép kiểm tra biên và chạy lại từng bước.')}
      foot={
        <Translated
          vi={
            <>
              Các giá trị là số nguyên nhỏ để bạn có thể kiểm tra tổng bằng mắt. Nhấn vào một luồng để xem các số có sẵn
              và <code>i</code> riêng của nó; nhấn vào một phần tử của C để xem ai đã ghi nó. <TierLegend tiers={['reg', 'global']} />
            </>
          }
        >
          <>
            Values are small whole numbers so you can check the sums by eye. Click a thread to see its built-in numbers
            and its private <code>i</code>; click an element of C to see who wrote it. <TierLegend tiers={['reg', 'global']} />
          </>
        </Translated>
      }
    >
      <div data-stepper data-line={line}>
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <label className="field">
            <span><Translated vi={<>Số phần tử n = <b data-out="n">{n}</b></>}>Elements n = <b data-out="n">{n}</b></Translated></span>
            <input type="range" min="1" max="24" step="1" value={n} aria-label={tr('Number of elements', 'Số phần tử')} onChange={(e) => setN(Number(e.target.value))} />
          </label>
          <label className="field">
            <span><Translated vi={<>Số luồng mỗi khối, blockDim.x = <b data-out="B">{B}</b></>}>Threads per block, blockDim.x = <b data-out="B">{B}</b></Translated></span>
            <input type="range" min="2" max="8" step="1" value={B} aria-label={tr('Threads per block', 'Số luồng mỗi khối')} onChange={(e) => setB(Number(e.target.value))} />
          </label>
          <label className="check">
            <input type="checkbox" checked={guard} onChange={(e) => setGuard(e.target.checked)} /> <Translated vi="kiểm tra biên">guard</Translated> <code>if (i &lt; n)</code>
          </label>
        </div>
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <button onClick={() => setLine(0)} disabled={line === 0}><Translated vi="Làm lại">Restart</Translated></button>
          <button onClick={() => setLine((line - 1) as Line)} disabled={line === 0}><Translated vi="Lùi">Back</Translated></button>
          <button className="primary" onClick={() => setLine((line + 1) as Line)} disabled={line === 3}><Translated vi="Bước">Step</Translated></button>
          <span className="muted" aria-live="polite">
            {line === 0 ? tr('before the first statement', 'trước câu lệnh đầu tiên') : tr(`after ${STATEMENTS()[line - 1]}, ${line} of 3`, `sau ${STATEMENTS()[line - 1]}, ${line} trên 3`)}
          </span>
          <span className="muted">
            <Translated
              vi={
                <>
                  <output data-out="blocks">{c.blocks}</output> khối, <output data-out="launched">{c.launched}</output> luồng,{' '}
                  <output data-out="working">{c.working}</output> có phần tử, <output data-out="idle">{c.idle}</output> không có
                </>
              }
            >
              <output data-out="blocks">{c.blocks}</output> blocks, <output data-out="launched">{c.launched}</output> threads,{' '}
              <output data-out="working">{c.working}</output> with an element, <output data-out="idle">{c.idle}</output> without
            </Translated>
          </span>
        </div>

        <div className={'narrate' + (bad ? ' bad' : '')} data-out="said" role="status" style={{ marginBottom: '0.8rem' }}>
          {said}
        </div>

        <div className="panel" style={{ overflowX: 'auto', marginBottom: '0.9rem' }}>
          <svg viewBox={`0 0 ${width} ${height}`} role="group" aria-label={tr('The grid of threads above the arrays A, B and C', 'Lưới luồng phía trên các mảng A, B và C')}
            style={{ width, maxWidth: 'none', display: 'block' }}>
            {Array.from({ length: c.blocks }, (_, b) => (
              <g key={b}>
                <rect x={x(b * B) - 3} y={S_TY - 4} width={B * S_CW + 2} height={S_TH + 8} rx="3" fill="none" stroke="var(--rule)" />
                <text x={x(b * B) - 2} y={13} fontSize="10.5" fontWeight="600">{B >= 3 ? tr(`block ${b}`, `khối ${b}`) : `b${b}`}</text>
              </g>
            ))}
            {st.threads.map((t) => {
              const state = threadState(t);
              const sel = selThread.g === t.g;
              const off = state === 'off';
              const strayed = state === 'stray';
              return (
                <g key={t.g} data-thread={t.g} data-state={state} role="button" tabIndex={0} style={{ cursor: 'pointer' }}
                  aria-label={tr(`Thread ${t.thread} of block ${t.block}`, `Luồng ${t.thread} của khối ${t.block}`)} aria-pressed={pick.kind === 'thread' && sel}
                  onClick={() => setPicked({ kind: 'thread', g: t.g })} onKeyDown={onKey(() => setPicked({ kind: 'thread', g: t.g }))}>
                  <rect x={x(t.g)} y={S_TY} width={S_CW - 4} height={S_TH} rx="2"
                    fill={off || t.i == null ? 'var(--cell)' : strayed ? 'var(--fail-wash)' : 'var(--reg-wash)'}
                    stroke={sel ? 'var(--ink)' : off ? 'var(--ink-3)' : strayed ? 'var(--fail)' : t.i == null ? 'var(--rule)' : 'var(--reg)'}
                    strokeWidth={sel ? 2.4 : 1} strokeDasharray={off ? '3 2' : undefined} />
                  <text x={x(t.g) + (S_CW - 4) / 2} y={S_TY + 14} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)', pointerEvents: 'none' }}>
                    t{t.thread}
                  </text>
                  {t.i != null ? (
                    <text x={x(t.g) + (S_CW - 4) / 2} y={S_TY + 32} fontSize="12" textAnchor="middle" fontWeight="650"
                      style={{ fill: off ? 'var(--ink-3)' : strayed ? 'var(--fail)' : 'var(--reg)', textDecoration: off ? 'line-through' : undefined, pointerEvents: 'none', fontVariantNumeric: 'tabular-nums' }}>
                      {t.i}
                    </text>
                  ) : null}
                </g>
              );
            })}
            {(['A', 'B', 'C'] as const).map((name, r) => {
              const y = S_AY + r * S_RH;
              return (
                <g key={name}>
                  <text x={8} y={y + 16} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}>{name}</text>
                  {Array.from({ length: c.launched }, (_, i) => {
                    const inside = i < n;
                    if (!inside && !stray.has(i)) return null;
                    const mine = pick.kind === 'elem' ? pick.e === i : selThread.i === i;
                    const touched = line >= 3 && (inside ? st.C[i] != null : true);
                    const value = !inside ? '?' : name === 'A' ? data.A[i] : name === 'B' ? data.B[i] : st.C[i];
                    const clickable = name === 'C' && inside;
                    return (
                      <g key={i} data-cell={`${name}:${i}`} data-touched={touched ? 'true' : 'false'}
                        {...(clickable
                          ? {
                              role: 'button', tabIndex: 0, style: { cursor: 'pointer' }, 'aria-label': tr(`C[${i}]: who writes it`, `C[${i}]: ai ghi nó`),
                              'aria-pressed': pick.kind === 'elem' && pick.e === i,
                              onClick: () => setPicked({ kind: 'elem', e: i }), onKeyDown: onKey(() => setPicked({ kind: 'elem', e: i })),
                            }
                          : {})}>
                        <rect x={x(i)} y={y} width={S_CW - 4} height={S_RH - 4} rx="2"
                          fill={!inside ? 'var(--fail-wash)' : touched ? 'var(--global-wash)' : 'var(--cell)'}
                          stroke={!inside ? 'var(--fail)' : mine ? 'var(--ink)' : touched ? 'var(--global)' : 'var(--rule)'}
                          strokeWidth={mine ? 2 : 1} strokeDasharray={!inside ? '3 2' : undefined} />
                        {value != null ? (
                          <text x={x(i) + (S_CW - 4) / 2} y={y + 15.5} fontSize="11.5" textAnchor="middle"
                            style={{ fill: !inside ? 'var(--fail)' : 'var(--ink)', pointerEvents: 'none', fontVariantNumeric: 'tabular-nums' }}>
                            {value}
                          </text>
                        ) : null}
                      </g>
                    );
                  })}
                </g>
              );
            })}
            {Array.from({ length: c.launched }, (_, i) =>
              i < n || stray.has(i) ? (
                <text key={i} x={x(i) + (S_CW - 4) / 2} y={S_AY + 3 * S_RH + 8} fontSize="9" textAnchor="middle"
                  style={{ fill: i < n ? 'var(--ink-3)' : 'var(--fail)', fontVariantNumeric: 'tabular-nums' }}>
                  {i}
                </text>
              ) : null,
            )}
          </svg>
        </div>

        {/* Side by side when the listing fits without scrolling, stacked otherwise. */}
        <div className="row" style={{ alignItems: 'flex-start', gap: '1.2rem' }}>
          <div style={{ flex: '2 1 40rem', minWidth: 0 }}>
            {kernel ? (
              <Code tabs={[{ label: 'vecAddKernel', code: kernel.text, tiers: KERNEL_TIERS }]} highlight={line ? [lineNo(line)] : []} />
            ) : (
              <Translated vi={<p className="muted">Danh sách mã kernel xuất hiện ở đây khi <code>labs/{LAB}/solution/kernel.cu</code> tải được.</p>}>
                <p className="muted">The kernel listing appears here once <code>labs/{LAB}/solution/kernel.cu</code> can be loaded.</p>
              </Translated>
            )}
            {!guard ? (
              <p className="muted" style={{ margin: 0 }} data-out="guard-off">
                <Translated vi={<>Phép kiểm tra biên đang tắt: hãy đọc mã như thể dòng {kernel ? lineNo(2) : 2} và dấu ngoặc đóng của nó đã bị xóa.</>}>
                  The guard is switched off: read the listing as if line {kernel ? lineNo(2) : 2} and its closing brace were deleted.
                </Translated>
              </p>
            ) : null}
          </div>
          <div className="panel" data-out="detail" style={{ flex: '1 1 16rem', minWidth: 0 }}>
            {pick.kind === 'elem' ? (
              <p style={{ margin: '0 0 0.5rem' }} data-out="writer">
                <b>C[{pick.e}]</b>{' '}
                <Translated
                  vi={
                    <>
                      {st.C[pick.e] == null ? 'chưa được ghi. Luồng duy nhất sẽ ghi nó là' : `chứa ${st.C[pick.e]}. Nó được ghi bởi`}{' '}
                      luồng {selThread.thread} của khối {selThread.block}, luồng duy nhất có <code>i</code> bằng {pick.e}.
                    </>
                  }
                >
                  {st.C[pick.e] == null ? 'has not been written yet. The one thread that will write it is' : `holds ${st.C[pick.e]}. It was written by`}{' '}
                  thread {selThread.thread} of block {selThread.block}, the only thread whose <code>i</code> is {pick.e}.
                </Translated>
              </p>
            ) : (
              <p style={{ margin: '0 0 0.5rem' }}>
                <b><Translated vi={<>Luồng {selThread.thread} của khối {selThread.block}</>}>Thread {selThread.thread} of block {selThread.block}</Translated></b>
              </p>
            )}
            {/* The four short values in a table; what the thread did in the guard and the addition as sentences below it. */}
            <table className="threads">
              <tbody>
                {threadFacts(selThread, n, B, line, guard).slice(0, 4).map((f) => (
                  <tr key={f.label}>
                    <td>{f.label}</td>
                    <td className={f.cls} style={{ whiteSpace: 'normal' }} data-fact={f.label.split(' ')[0]}>{f.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {threadFacts(selThread, n, B, line, guard).slice(4).map((f) => (
              <p key={f.label} style={{ margin: '0.5rem 0 0', fontSize: '0.85rem' }}>
                <span className="muted">{f.label === 'guard' ? tr('The guard: ', 'Phép kiểm tra biên: ') : tr('The addition: ', 'Phép cộng: ')}</span>
                <span className={f.cls} data-fact={f.label.split(' ')[0]}>{f.value}</span>
              </p>
            ))}
            {st.outOfRange.length ? (
              <p className="t-fail" style={{ margin: '0.6rem 0 0' }} data-out="oob">
                <Translated vi={<>Các chỉ số bị truy cập ngoài mảng: {st.outOfRange.join(', ')}.</>}>
                  Indices touched outside the arrays: {st.outOfRange.join(', ')}.
                </Translated>
              </p>
            ) : null}
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive B: thread index explorer
const UNLOCKS: Partial<Record<MappingKind, string>> = {
  'two-adjacent': 'ch02/2.5/q-two-adjacent',
  'two-sections': 'ch02/2.5/q-two-sections',
};
const PRESETS = [
  { n: 100, B: 32, page: 'p. 37' },
  { n: 1000, B: 256, page: 'pp. 38–39' },
];
const E_LEFT = 50;
const E_CW = 27;
const E_RH = 30;
const E_MAX_ROW = 32;

function IndexExplorer() {
  const { progress } = useProgress();
  const [n, setNRaw] = useState(PRESETS[0].n);
  const [B, setBRaw] = useState(PRESETS[0].B);
  const [snap, setSnap] = useState(true);
  const [kindWanted, setKind] = useState<MappingKind>('one');
  const [sel, setSel] = useState<{ b: number; t: number } | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [focus, setFocus] = useState<number | null>(null);

  const unlocked = (k: MappingKind) => !UNLOCKS[k] || !!progress.answers[UNLOCKS[k]!];
  const kind: MappingKind = unlocked(kindWanted) ? kindWanted : 'one';
  const m = mapping(kind);
  const c = counts({ n, blockDim: B, mapping: kind });
  const blockSpan = B * m.perThread; // elements one block is responsible for
  const reach = c.launched * m.perThread; // element positions the launched threads map to

  const setN = (v: number) => {
    setNRaw(Math.max(1, Math.min(2000, Math.round(v) || 1)));
    setFocus(null);
    setHover(null);
  };
  const setB = (v: number, snapNow = snap) => {
    const w = Math.max(1, Math.min(BOOK_MAX_BLOCK, Math.round(v) || 1));
    setBRaw(snapNow ? snapBlock(w) : w);
    setFocus(null);
    setHover(null);
  };

  // The thread in view: the one the reader chose, or the owner of the last element.
  const last = m.owner(n - 1, B);
  const cur = sel && sel.b < c.blocks && sel.t < B ? sel : { b: last.block, t: last.thread };
  const curElems = m.elements(cur.b, cur.t, B);
  const curSet = new Set(curElems);
  const inRange = curElems.map((e) => e < n);

  // The window of elements drawn in full: about two blocks, between one and eight rows.
  const perRow = blockSpan <= E_MAX_ROW ? blockSpan * Math.floor(E_MAX_ROW / blockSpan) : E_MAX_ROW;
  const want = Math.max(perRow, Math.min(256, 2 * blockSpan));
  const rowsShown = Math.ceil(want / perRow);
  const totalRows = Math.ceil(reach / perRow);
  const maxStartRow = Math.max(0, totalRows - rowsShown);
  const startRow = focus == null ? maxStartRow : Math.max(0, Math.min(maxStartRow, Math.floor(focus / perRow) - Math.floor(rowsShown / 2)));
  const start = startRow * perRow;
  const end = Math.min(reach, start + rowsShown * perRow);
  const rows = Math.ceil((end - start) / perRow);

  const owner = (e: number) => m.owner(e, B);
  /** 'in': below n. 'tail': past n, but its thread has another element in range. 'idle': past n, thread has nothing in range. */
  const cellState = (e: number): 'in' | 'tail' | 'idle' => {
    if (e < n) return 'in';
    const o = owner(e);
    return m.elements(o.block, o.thread, B).some((x) => x < n) ? 'tail' : 'idle';
  };

  const shownE = hover ?? null;
  const shownOwner = shownE != null ? owner(shownE) : null;

  const describeThread = (): ReactNode => {
    const head = (
      <>
        <b><Translated vi={<>Luồng {cur.t} của khối {cur.b}</>}>Thread {cur.t} of block {cur.b}</Translated></b>: i = <span className="t-reg" data-out="working">{working(cur.b, cur.t, B, kind)}</span>.{' '}
      </>
    );
    if (m.perThread === 1)
      return (
        <>
          {head}
          {inRange[0] ? tr(`It owns element ${curElems[0]}.`, `Nó sở hữu phần tử ${curElems[0]}.`) : tr(`${curElems[0]} is not below n = ${n}: this thread fails the guard and does nothing.`, `${curElems[0]} không nhỏ hơn n = ${n}: luồng này không vượt qua phép kiểm tra biên và không làm gì.`)}
        </>
      );
    return (
      <>
        {head}
        <Translated
          vi={
            <>
              Hai phần tử của nó là {curElems[0]} và {curElems[1]} (<code>{m.secondExpr}</code>).{' '}
              {inRange[0] && inRange[1]
                ? 'Cả hai đều nằm trong phạm vi.'
                : inRange[0]
                  ? <span className="t-fail">Phần tử {curElems[1]} nằm quá cuối (n = {n}): lần truy cập thứ hai cần phép kiểm tra riêng, <code>{m.guards[1]}</code>.</span>
                  : `Không phần tử nào nhỏ hơn n = ${n}: luồng này không làm gì.`}
            </>
          }
        >
          Its two elements are {curElems[0]} and {curElems[1]} (<code>{m.secondExpr}</code>).{' '}
          {inRange[0] && inRange[1]
            ? 'Both are in range.'
            : inRange[0]
              ? <span className="t-fail">Element {curElems[1]} is past the end (n = {n}): the second access needs its own test, <code>{m.guards[1]}</code>.</span>
              : `Neither is below n = ${n}: this thread does nothing.`}
        </Translated>
      </>
    );
  };

  const pickElement = (e: number) => {
    const o = owner(e);
    setSel({ b: o.block, t: o.thread });
  };

  const svgW = E_LEFT + E_MAX_ROW * E_CW + 6;
  const overviewX = (e: number) => (e / reach) * 1000;
  const drawBlocks = c.blocks <= 250;

  return (
    <Bench
      title={tr('Thread index explorer', 'Trình khám phá chỉ số luồng')}
      task={tr('Move n and the block size. Before you let go, say how many threads will have nothing to do.', 'Thay đổi n và kích thước khối. Trước khi thả tay, hãy nói bao nhiêu luồng sẽ không có việc gì làm.')}
      foot={tr('Shading alternates from block to block and carries no other meaning. A dashed cell is an index at or past n that some launched thread maps to. The upper strip is the whole launch; the frame on it marks the part drawn in full below.', 'Màu nền xen kẽ giữa các khối và không mang ý nghĩa nào khác. Một ô viền nét đứt là một chỉ số bằng hoặc vượt n mà một luồng đã khởi chạy nào đó ánh xạ tới. Dải phía trên là toàn bộ lần khởi chạy; khung trên đó đánh dấu phần được vẽ đầy đủ bên dưới.')}
    >
      <div data-explorer data-kind={kind}>
        <div className="row" style={{ marginBottom: '0.7rem' }}>
          <span className="muted"><Translated vi="Các trường hợp của sách:">The book's cases:</Translated></span>
          {PRESETS.map((p) => (
            <button key={p.n} className="quiet" data-preset={p.n} onClick={() => { setSnap(true); setNRaw(p.n); setBRaw(p.B); setFocus(null); setSel(null); setHover(null); }}>
              <Translated vi={<>n = {num(p.n)}, khối {p.B} luồng ({p.page})</>}>
                n = {num(p.n)}, blocks of {p.B} ({p.page})
              </Translated>
            </button>
          ))}
        </div>
        <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
          <label className="field">
            <Translated vi="Số phần tử n">Elements n</Translated>
            <span className="row" style={{ gap: '0.5rem' }}>
              <input type="range" min="1" max="2000" step="1" value={n} aria-label={tr('Number of elements, slider', 'Số phần tử, thanh trượt')} onChange={(e) => setN(Number(e.target.value))} />
              <input type="number" min="1" max="2000" value={n} data-in="n" aria-label={tr('Number of elements', 'Số phần tử')} style={{ width: '5.5rem' }} onChange={(e) => setN(Number(e.target.value))} />
            </span>
          </label>
          <label className="field">
            <Translated vi="Số luồng mỗi khối, blockDim.x">Threads per block, blockDim.x</Translated>
            <span className="row" style={{ gap: '0.5rem' }}>
              <input type="range" min={snap ? BOOK_BLOCK_MULTIPLE : 1} max={BOOK_MAX_BLOCK} step={snap ? BOOK_BLOCK_MULTIPLE : 1} value={B}
                aria-label={tr('Threads per block, slider', 'Số luồng mỗi khối, thanh trượt')} onChange={(e) => setB(Number(e.target.value))} />
              <input type="number" min="1" max={BOOK_MAX_BLOCK} step={snap ? BOOK_BLOCK_MULTIPLE : 1} value={B} data-in="B" aria-label={tr('Threads per block', 'Số luồng mỗi khối')} style={{ width: '5.5rem' }}
                onChange={(e) => setB(Number(e.target.value))} />
            </span>
          </label>
          <label className="check">
            <input type="checkbox" checked={snap} data-in="snap" onChange={(e) => { setSnap(e.target.checked); if (e.target.checked) setB(B, true); }} />{' '}
            <Translated vi={<>chỉ bội số của {BOOK_BLOCK_MULTIPLE}</>}>multiples of {BOOK_BLOCK_MULTIPLE} only</Translated>
          </label>
        </div>
        <div className="row" style={{ marginBottom: '0.3rem' }}>
          <span className="muted"><Translated vi="Mỗi luồng xử lý">Each thread handles</Translated></span>
          <span className="seg" role="group" aria-label={tr('Elements per thread', 'Số phần tử mỗi luồng')}>
            {MAPPING_KINDS.map((k) => (
              <button key={k} data-kind-choice={k} aria-pressed={kind === k} disabled={!unlocked(k)} onClick={() => { setKind(k); setFocus(null); setSel(null); setHover(null); }}
                title={unlocked(k) ? undefined : tr(`Answer book exercise ${mapping(k).exercise} below to unlock`, `Trả lời bài tập ${mapping(k).exercise} của sách bên dưới để mở khóa`)}>
                {tr(mapping(k).label, ({ one: 'một phần tử mỗi luồng', 'two-adjacent': 'hai phần tử kề nhau mỗi luồng', 'two-sections': 'hai đoạn mỗi khối' } as Record<string, string>)[k])}
              </button>
            ))}
          </span>
        </div>
        {MAPPING_KINDS.some((k) => !unlocked(k)) ? (
          <p className="muted" data-out="locked" style={{ margin: '0 0 0.7rem' }}>
            <Translated vi="Các lựa chọn bị làm mờ sẽ mở khi bạn trả lời bài tập 2 và 3 của sách, ngay bên dưới trình khám phá.">
              The greyed-out choices open when you answer book exercises 2 and 3, just below the explorer.
            </Translated>
          </p>
        ) : null}

        <div className="cols plot" style={{ margin: '0.6rem 0 0.8rem' }}>
          <div className="panel">
            <table className="threads">
              <tbody>
                <tr><td><Translated vi="Số khối cần">Blocks needed</Translated></td><td><output data-out="blocks">{num(c.blocks)}</output></td></tr>
                <tr><td><Translated vi="Số luồng được khởi chạy">Threads launched</Translated></td><td><output data-out="launched">{num(c.launched)}</output></td></tr>
                <tr><td><Translated vi="Số luồng có làm việc">Threads that do work</Translated></td><td><output data-out="working">{num(c.working)}</output></td></tr>
                <tr>
                  <td><Translated vi="Số luồng rảnh">Idle threads</Translated></td>
                  <td style={{ whiteSpace: 'normal' }}>
                    <output data-out="idle">{num(c.idle)}</output>
                    <span className="muted">{c.idle === 0 ? tr(': the launch fits exactly', ': lần khởi chạy vừa khít') : tr(`, fewer than one block of ${num(B)}`, `, ít hơn một khối ${num(B)} luồng`)}</span>
                  </td>
                </tr>
                {m.perThread === 2 ? (
                  <tr>
                    <td style={{ whiteSpace: 'normal' }}><Translated vi="Một phần tử trong phạm vi, một phần tử quá cuối">One element in range, one past the end</Translated></td>
                    <td><output data-out="partial" className={c.partial ? 't-fail' : undefined}>{num(c.partial)}</output></td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
          <div className="panel" data-out="formula">
            <p style={{ margin: '0 0 0.4rem' }}>
              <span className="muted">{m.perThread === 1 ? tr('Its element', 'Phần tử của nó') : tr('Its first element', 'Phần tử đầu tiên của nó')}</span>
              <br />
              <code style={{ whiteSpace: 'normal' }}>i = {m.firstExpr}</code>
            </p>
            {m.secondExpr ? (
              <p style={{ margin: '0 0 0.4rem' }}>
                <span className="muted"><Translated vi="Phần tử thứ hai của nó">Its second element</Translated></span>
                <br />
                <code>{m.secondExpr}</code>
              </p>
            ) : null}
            <p style={{ margin: 0 }}>
              <span className="muted">{m.guards.length === 1 ? tr('Guard', 'Kiểm tra biên') : tr('Guards, one per access', 'Các phép kiểm tra biên, mỗi lần truy cập một phép')}</span>
              <br />
              {m.guards.map((g, k) => (
                <span key={g}>
                  {k ? ', ' : ''}
                  <code>{g}</code>
                </span>
              ))}
            </p>
          </div>
        </div>

        <div className="panel" style={{ overflowX: 'auto', marginBottom: '0.8rem' }}>
          <svg viewBox="0 0 1000 50" role="img" data-overview aria-label={tr(`The whole launch: ${c.blocks} blocks over ${reach} element positions, ${n} of them real`, `Toàn bộ lần khởi chạy: ${c.blocks} khối trên ${reach} vị trí phần tử, ${n} trong số đó là thật`)}
            style={{ width: '100%', minWidth: 560, display: 'block', cursor: 'pointer', marginBottom: '0.5rem' }}
            onClick={(ev) => {
              const r = ev.currentTarget.getBoundingClientRect();
              setFocus(Math.max(0, Math.min(reach - 1, Math.floor(((ev.clientX - r.left) / r.width) * reach))));
            }}>
            {drawBlocks ? (
              Array.from({ length: c.blocks }, (_, b) => (
                <rect key={b} x={overviewX(b * blockSpan)} y={16} width={overviewX(blockSpan)} height={18}
                  fill={b % 2 ? 'var(--bench)' : 'var(--cell)'} stroke="var(--ink-3)" strokeWidth="0.6" />
              ))
            ) : (
              <rect x={0} y={16} width={1000} height={18} fill="var(--cell)" stroke="var(--ink-3)" strokeWidth="0.6" />
            )}
            {reach > n ? (
              <rect x={overviewX(n)} y={16} width={1000 - overviewX(n)} height={18} fill="var(--page)" opacity="0.75" stroke="var(--ink-2)" strokeDasharray="4 3" />
            ) : null}
            <line x1={overviewX(n)} y1={10} x2={overviewX(n)} y2={38} stroke="var(--ink)" strokeWidth="1.6" />
            <text x={Math.min(overviewX(n), 996)} y={8} fontSize="11" textAnchor={overviewX(n) > 60 ? 'end' : 'start'}>n = {num(n)}</text>
            <text x={0} y={47} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
              {drawBlocks ? tr(`block 0 to block ${num(c.blocks - 1)}`, `khối 0 đến khối ${num(c.blocks - 1)}`) : tr(`${num(c.blocks)} blocks, too narrow to draw one by one`, `${num(c.blocks)} khối, quá hẹp để vẽ từng khối`)}
            </text>
            <text x={1000} y={47} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
              <Translated vi={<>{num(c.launched)} luồng được khởi chạy</>}>
                {num(c.launched)} threads launched
              </Translated>
            </text>
            <rect x={overviewX(start)} y={13} width={Math.max(3, overviewX(end - start))} height={24} fill="none" stroke="var(--ink)" strokeWidth="2.2" />
          </svg>
          <svg viewBox={`0 0 ${svgW} ${rows * E_RH + 4}`} role="group" data-strip aria-label={tr(`Element positions ${start} to ${end - 1}`, `Vị trí phần tử ${start} đến ${end - 1}`)}
            style={{ width: '100%', minWidth: 620, maxWidth: svgW, display: 'block' }} onMouseLeave={() => setHover(null)}>
            {Array.from({ length: rows }, (_, r) => (
              <text key={'r' + r} x={E_LEFT - 8} y={r * E_RH + 18} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)', fontVariantNumeric: 'tabular-nums' }}>
                {start + r * perRow}
              </text>
            ))}
            {Array.from({ length: end - start }, (_, k) => {
              const e = start + k;
              const cx = E_LEFT + (k % perRow) * E_CW;
              const cy = Math.floor(k / perRow) * E_RH + 3;
              const state = cellState(e);
              const o = owner(e);
              const mine = curSet.has(e);
              return (
                <g key={e} data-e={e} data-state={state} data-block={o.block} data-thread={o.thread} data-mine={mine ? 'true' : 'false'}
                  role="button" tabIndex={0} style={{ cursor: 'pointer' }}
                  aria-label={tr(`Index ${e}: thread ${o.thread} of block ${o.block}`, `Chỉ số ${e}: luồng ${o.thread} của khối ${o.block}`)} aria-pressed={mine}
                  onClick={() => pickElement(e)} onKeyDown={onKey(() => pickElement(e))} onMouseEnter={() => setHover(e)}>
                  <rect x={cx} y={cy} width={E_CW - 3} height={22} rx="2"
                    fill={state === 'tail' ? 'var(--fail-wash)' : state === 'idle' ? 'transparent' : o.block % 2 ? 'var(--bench)' : 'var(--cell)'}
                    stroke={mine ? 'var(--ink)' : state === 'tail' ? 'var(--fail)' : state === 'idle' ? 'var(--ink-3)' : 'var(--rule)'}
                    strokeWidth={mine ? 2.4 : 1} strokeDasharray={state === 'in' || mine ? undefined : '3 2'} />
                  <text x={cx + (E_CW - 3) / 2} y={cy + 15} fontSize="9.5" textAnchor="middle" fontWeight={mine ? 700 : 400}
                    style={{ fill: state === 'tail' ? 'var(--fail)' : state === 'idle' ? 'var(--ink-3)' : 'var(--ink)', pointerEvents: 'none', fontVariantNumeric: 'tabular-nums' }}>
                    {e}
                  </text>
                  {e % blockSpan === 0 ? <line x1={cx - 1.5} y1={cy - 2} x2={cx - 1.5} y2={cy + 24} stroke="var(--ink)" strokeWidth="2" /> : null}
                  {kind === 'two-sections' && e % blockSpan === B ? (
                    <line x1={cx - 1.5} y1={cy - 2} x2={cx - 1.5} y2={cy + 24} stroke="var(--ink-2)" strokeWidth="1.4" strokeDasharray="3 2" />
                  ) : null}
                </g>
              );
            })}
          </svg>
          <p className="muted" style={{ margin: '0.5rem 0 0' }} data-out="window">
            <Translated
              vi={
                <>
                  Được vẽ đầy đủ: các vị trí {num(start)} đến {num(end - 1)}. Một vạch đậm đánh dấu phần tử đầu tiên của một khối
                  {kind === 'two-sections' ? ', một vạch nét đứt đánh dấu điểm bắt đầu phần thứ hai của nó' : ''}. Nhấn vào dải phía trên để xem chỗ khác.
                </>
              }
            >
              Drawn in full: positions {num(start)} to {num(end - 1)}. A heavy bar marks the first element of a block
              {kind === 'two-sections' ? ', a dashed bar the start of its second section' : ''}. Click the upper strip to look elsewhere.
            </Translated>
          </p>
        </div>

        <div className="cols plot">
          <div className="narrate" data-out="thread" role="status">
            {describeThread()}
          </div>
          <div className="panel">
            <div className="row" style={{ gap: '0.6rem 1rem' }}>
              <label className="field">
                blockIdx.x
                <input type="number" min="0" max={c.blocks - 1} value={cur.b} data-in="block" aria-label={tr('Block index', 'Chỉ số khối')} style={{ width: '5.5rem' }}
                  onChange={(e) => {
                    const b = Math.max(0, Math.min(c.blocks - 1, Math.round(Number(e.target.value)) || 0));
                    setSel({ b, t: cur.t });
                    setFocus(m.first(b, cur.t, B));
                  }} />
              </label>
              <label className="field">
                threadIdx.x
                <input type="number" min="0" max={B - 1} value={cur.t} data-in="thread" aria-label={tr('Thread index', 'Chỉ số luồng')} style={{ width: '5.5rem' }}
                  onChange={(e) => {
                    const t = Math.max(0, Math.min(B - 1, Math.round(Number(e.target.value)) || 0));
                    setSel({ b: cur.b, t });
                    setFocus(m.first(cur.b, t, B));
                  }} />
              </label>
            </div>
            <p className="muted" style={{ margin: '0.5rem 0 0', minHeight: '2.8em' }} data-out="hover">
              {shownE != null && shownOwner
                ? tr(
                    `Index ${num(shownE)}: block ${num(shownOwner.block)}, thread ${num(shownOwner.thread)}${m.perThread === 2 ? `, its ${shownOwner.slot ? 'second' : 'first'} element` : ''}${shownE >= n ? '. Past the end.' : '.'}`,
                    `Chỉ số ${num(shownE)}: khối ${num(shownOwner.block)}, luồng ${num(shownOwner.thread)}${m.perThread === 2 ? `, phần tử ${shownOwner.slot ? 'thứ hai' : 'đầu tiên'} của nó` : ''}${shownE >= n ? '. Quá cuối mảng.' : '.'}`,
                  )
                : tr('Type a thread, or point at a cell to see who owns it and click to keep it.', 'Nhập một luồng, hoặc trỏ vào một ô để xem ai sở hữu nó và nhấn để giữ lựa chọn.')}
            </p>
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- your own devices
/** What the reader's devices reported about the size of a block, from reports/runs. */
function YourLimits() {
  const { reports, loading } = useReports();
  const rows = Object.keys(reports)
    .filter((b) => b !== 'emu' && b !== 'race')
    .flatMap((b) => {
      const d = reports[b].device;
      const name = String(d.name ?? b);
      const block = Number(d.maxThreadsPerBlock);
      const along = Number(d['maxThreadsPerThreadgroup.width']);
      if (Number.isFinite(block) && block > 0) return [{ b, text: tr(`${name} (${b}) reported at most ${num(block)} threads per block.`, `${name} (${b}) báo cáo tối đa ${num(block)} luồng mỗi khối.`) }];
      if (Number.isFinite(along) && along > 0)
        return [{ b, text: tr(`${name} (${b}) reported at most ${num(along)} threads along one dimension of a threadgroup. Apple says that figure holds for a trivial kernel; the limit for a particular kernel is read from its pipeline state.`, `${name} (${b}) báo cáo tối đa ${num(along)} luồng theo một chiều của một threadgroup. Apple nói con số đó đúng cho một kernel tầm thường; giới hạn cho một kernel cụ thể được đọc từ pipeline state của nó.`) }];
      return [];
    });
  if (loading) return null;
  return (
    <p data-out="your-limits">
      {rows.length ? (
        <>
          <b><Translated vi="Trên máy này.">On this machine.</Translated></b> {rows.map((r) => r.text).join(' ')}
        </>
      ) : (
        <Translated
          vi={
            <span className="muted">
              Chưa có báo cáo device nào trên máy này. <code>python3 labs/pmpp.py doctor</code> in ra những gì GPU của bạn
              báo cáo. Hãy chạy một lab bất kỳ, chẳng hạn <code>python3 labs/pmpp.py run 2.1</code>, rồi tải lại trang này:
              lần chạy đầu tiên trên một backend ghi lại device của nó, và nó sẽ xuất hiện ở đây.
            </span>
          }
        >
          <span className="muted">
            No device report on this machine yet. <code>python3 labs/pmpp.py doctor</code> prints what your GPUs
            report. Run any lab, for example <code>python3 labs/pmpp.py run 2.1</code>, and reload this page: the
            first run on a backend records its device, and it appears here.
          </span>
        </Translated>
      )}
    </p>
  );
}

// ---------------------------------------------------------------- numbers the questions state, from the model
const OWN = mapping('one').first(3, 5, 256);
const P37 = counts({ n: 100, blockDim: 32 });
const NOGUARD = coverage({ n: 1000, blockDim: 256, guard: false });
const ADJ_ODD = counts({ n: 101, blockDim: 32, mapping: 'two-adjacent' });

export const questionsS25: Record<string, Question> = {
  'ch02/2.5/q-own': {
    kind: 'numeric',
    id: 'ch02/2.5/q-own',
    prompt: (
      <Translated
        vi={
          <>
            <code>blockDim.x</code> là 256 và mỗi luồng xử lý một phần tử. Luồng 5 của khối 3 sở hữu phần tử nào?
          </>
        }
      >
        <>
          <code>blockDim.x</code> is 256 and each thread handles one element. Which element does thread 5 of block 3 own?
        </>
      </Translated>
    ),
    answer: OWN,
    explain: (
      <Translated
        vi={
          <p>
            Các khối 0, 1 và 2 đứng trước và cùng sở hữu 3 × 256 = 768 phần tử, từ 0 đến 767. Luồng 5 nằm ở vị trí thứ
            năm tính vào trong khối 3: {working(3, 5, 256)}.
          </p>
        }
      >
        <p>
          Blocks 0, 1 and 2 come first and own 3 × 256 = 768 elements between them, 0 to 767. Thread 5 is five places
          into block 3: {working(3, 5, 256)}.
        </p>
      </Translated>
    ),
    source: '§2.5, pp. 33–34 (Fig. 2.9)',
  },
  'ch02/2.5/q-index': {
    kind: 'choice',
    id: 'ch02/2.5/q-index',
    prompt: (
      <Translated
        vi={
          <>
            Mỗi luồng của lưới tính một phần tử đầu ra của phép cộng vector. Biểu thức nào ánh xạ chỉ số luồng và chỉ số
            khối sang chỉ số dữ liệu <code>i</code>?
          </>
        }
      >
        <>
          Each thread of the grid computes one output element of a vector addition. Which expression maps the thread
          and block indices to the data index <code>i</code>?
        </>
      </Translated>
    ),
    options: [
      {
        label: <code>i = threadIdx.x + threadIdx.y;</code>,
        why: (
          <Translated vi="Trong một khối một chiều, threadIdx.y bằng 0 với mọi luồng, nên biểu thức này chỉ còn threadIdx.x: mọi khối sẽ làm việc trên blockDim.x phần tử đầu tiên và không gì khác.">
            In a one-dimensional block threadIdx.y is 0 for every thread, so this is threadIdx.x alone: every block would work on the first blockDim.x elements and nothing else.
          </Translated>
        ),
      },
      {
        label: <code>i = blockIdx.x + threadIdx.x;</code>,
        why: (
          <Translated vi="Các khối kề nhau chồng lấn: luồng 1 của khối 0 và luồng 0 của khối 1 đều nhận 1.">
            Neighbouring blocks overlap: thread 1 of block 0 and thread 0 of block 1 both get 1.
          </Translated>
        ),
      },
      { label: <code>i = blockIdx.x * blockDim.x + threadIdx.x;</code>, correct: true },
      {
        label: <code>i = blockIdx.x * threadIdx.x;</code>,
        why: (
          <Translated vi="Mọi luồng của khối 0 nhận 0, và luồng 0 của mọi khối khác cũng vậy.">
            Every thread of block 0 gets 0, and so does thread 0 of every other block.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Các khối đứng trước khối của một luồng chiếm <code>blockIdx.x * blockDim.x</code> phần tử, và{' '}
            <code>threadIdx.x</code> là vị trí của luồng trong khối của nó. Một ánh xạ đúng giao mỗi phần tử cho đúng một
            luồng; mỗi phương án trong ba phương án còn lại giao một phần tử nào đó cho hai luồng hoặc không cho luồng nào.
          </p>
        }
      >
        <p>
          The blocks before a thread's own block account for <code>blockIdx.x * blockDim.x</code> elements, and{' '}
          <code>threadIdx.x</code> is the thread's place inside its block. A correct mapping gives every element to
          exactly one thread; each of the other three gives some element to two threads or to none.
        </p>
      </Translated>
    ),
    source: 'book exercise 1, p. 41; §2.5, pp. 33–34. The book prints no answers: the reasoning is ours.',
    vi: { source: 'bài tập 1 của sách, p. 41; §2.5, pp. 33–34. Sách không in đáp án: lập luận là của khóa học này.' },
  },
  'ch02/2.5/q-two-adjacent': {
    kind: 'choice',
    id: 'ch02/2.5/q-two-adjacent',
    prompt: (
      <Translated
        vi={
          <>
            Giờ mỗi luồng tính <b>hai phần tử kề nhau</b> của phép cộng vector. Biểu thức nào cho chỉ số của phần tử đầu
            tiên mà một luồng xử lý?
          </>
        }
      >
        <>
          Now each thread computes <b>two adjacent</b> elements of the vector addition. Which expression gives the index
          of the first element a thread handles?
        </>
      </Translated>
    ),
    options: [
      {
        label: <code>i = blockIdx.x * blockDim.x + threadIdx.x + 2;</code>,
        why: (
          <Translated vi="Đây là chỉ số một phần tử dịch đi 2. Các luồng liên tiếp vẫn bắt đầu cách nhau một phần tử, nên các cặp của chúng chồng lấn, và không ai xử lý phần tử 0 và 1.">
            This is the one-element index moved along by 2. Consecutive threads still start one element apart, so their pairs overlap, and nobody handles elements 0 and 1.
          </Translated>
        ),
      },
      {
        label: <code>i = blockIdx.x * threadIdx.x * 2;</code>,
        why: (
          <Translated vi="Mọi luồng của khối 0 nhận 0, và luồng 0 của mọi khối khác cũng vậy.">
            Every thread of block 0 gets 0, and so does thread 0 of every other block.
          </Translated>
        ),
      },
      { label: <code>i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;</code>, correct: true },
      {
        label: <code>i = blockIdx.x * blockDim.x * 2 + threadIdx.x;</code>,
        why: (
          <Translated vi="Mỗi khối bắt đầu đúng chỗ, nhưng bên trong một khối các luồng liên tiếp bắt đầu cách nhau một phần tử, nên các cặp kề nhau sẽ chồng lấn. Đây là cách bố trí của bài tập tiếp theo.">
            Each block starts in the right place, but inside a block consecutive threads start one element apart, so adjacent pairs would overlap. This is the layout of the next exercise.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Đánh số các luồng của cả lưới là k = 0, 1, 2, v.v., chính là chỉ số cũ{' '}
              <code>blockIdx.x * blockDim.x + threadIdx.x</code>. Luồng k nhận các phần tử 2k và 2k + 1. Trình khám phá ở
              trên giờ có thêm ánh xạ này.
            </p>
            <p>
              Phép kiểm tra biên thay đổi theo. Khi n lẻ, luồng làm việc cuối cùng có phần tử đầu trong phạm vi và phần
              tử thứ hai quá cuối, nên mỗi lần truy cập cần một phép kiểm tra riêng: với n = 101 và khối 32 luồng, đó là{' '}
              {ADJ_ODD.partial} luồng. Các kernel Metal của MLX cho phép toán theo từng phần tử làm đúng như vậy: một luồng
              xử lý vài phần tử trong một vòng lặp ngắn, và vòng lặp kiểm tra <code>index + i &lt; size</code> cho từng
              phần tử.
            </p>
          </>
        }
      >
        <>
          <p>
            Number the threads of the whole grid k = 0, 1, 2 and so on, which is the old index{' '}
            <code>blockIdx.x * blockDim.x + threadIdx.x</code>. Thread k takes elements 2k and 2k + 1. The explorer above
            now offers this mapping.
          </p>
          <p>
            The guard changes with it. When n is odd, the last working thread has its first element in range and its
            second past the end, so each access needs its own test: with n = 101 and blocks of 32 that is{' '}
            {ADJ_ODD.partial} thread. MLX's Metal kernels for elementwise operations do exactly this: a thread handles
            several elements in a short loop, and the loop tests <code>index + i &lt; size</code> for each one.
          </p>
        </>
      </Translated>
    ),
    source: 'book exercise 2, p. 41 (answer ours); MLX: notes/today/ch02-facts.md, D3',
    vi: { source: 'bài tập 2 của sách, p. 41 (đáp án của khóa học này); MLX: notes/today/ch02-facts.md, D3' },
  },
  'ch02/2.5/q-two-sections': {
    kind: 'choice',
    id: 'ch02/2.5/q-two-sections',
    prompt: (
      <Translated
        vi={
          <>
            Mỗi luồng lại tính hai phần tử, nhưng bố trí khác. Một khối chịu trách nhiệm cho{' '}
            <code>2 * blockDim.x</code> phần tử liên tiếp chia thành hai đoạn. Tất cả luồng của nó trước tiên mỗi luồng xử
            lý một phần tử trong đoạn thứ nhất, rồi mỗi luồng một phần tử trong đoạn thứ hai. Biểu thức nào cho chỉ số
            phần tử <b>đầu tiên</b> của một luồng?
          </>
        }
      >
        <>
          Each thread again computes two elements, laid out differently. A block is responsible for{' '}
          <code>2 * blockDim.x</code> consecutive elements in two sections. All its threads first handle one element
          each in the first section, then one each in the second. Which expression gives the index of a thread's{' '}
          <b>first</b> element?
        </>
      </Translated>
    ),
    options: [
      {
        label: <code>i = blockIdx.x * blockDim.x + threadIdx.x + 2;</code>,
        why: (
          <Translated vi="Các khối sẽ bắt đầu cách nhau blockDim.x phần tử, nhưng giờ mỗi khối phủ gấp đôi số đó.">
            Blocks would start blockDim.x elements apart, but each block now covers twice that.
          </Translated>
        ),
      },
      {
        label: <code>i = blockIdx.x * threadIdx.x * 2;</code>,
        why: <Translated vi="Mọi luồng của khối 0 nhận 0.">Every thread of block 0 gets 0.</Translated>,
      },
      {
        label: <code>i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;</code>,
        why: (
          <Translated vi="Biểu thức đó cho mỗi luồng hai phần tử kề nhau, cách bố trí của bài tập trước. Ở đây hai phần tử của một luồng cách nhau blockDim.x.">
            That gives each thread two adjacent elements, the layout of the previous exercise. Here the two elements of a thread are blockDim.x apart.
          </Translated>
        ),
      },
      { label: <code>i = blockIdx.x * blockDim.x * 2 + threadIdx.x;</code>, correct: true },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Khối b bắt đầu ở phần tử <code>b * 2 * blockDim.x</code>. Trong đoạn thứ nhất, luồng t nhận phần tử ở vị trí
              thứ t tính từ đầu đoạn. Phần tử thứ hai của nó ở cùng vị trí trong đoạn thứ hai: <code>i + blockDim.x</code>.
              Hãy chuyển trình khám phá sang ánh xạ này và so sánh luồng nào sở hữu phần tử 1 trong mỗi ánh xạ của cả ba.
            </p>
            <p className="muted">
              Sách in cùng bốn phương án dưới bài tập 2 và 3, và không có đáp án. Cả hai đáp án ở đây là của khóa học này.
            </p>
          </>
        }
      >
        <>
          <p>
            Block b starts at element <code>b * 2 * blockDim.x</code>. In the first section thread t takes the element t
            places in. Its second element is the same place in the second section: <code>i + blockDim.x</code>. Switch
            the explorer to this mapping and compare which thread owns element 1 in each of the three.
          </p>
          <p className="muted">
            The book prints the same four options under exercises 2 and 3, and no answers. Both answers here are ours.
          </p>
        </>
      </Translated>
    ),
    source: 'book exercise 3, p. 41 (answer ours)',
    vi: { source: 'bài tập 3 của sách, p. 41 (đáp án của khóa học này)' },
  },
  'ch02/2.5/q-private': {
    kind: 'choice',
    id: 'ch02/2.5/q-private',
    prompt: (
      <Translated
        vi={
          <>
            Một lưới 10,000 luồng chạy một kernel có dòng đầu tiên là <code>int i = ...;</code>. Có bao nhiêu bản{' '}
            <code>i</code> tồn tại, và ai nhìn thấy mỗi bản?
          </>
        }
      >
        <>
          A grid of 10,000 threads runs a kernel whose first line is <code>int i = ...;</code>. How many copies of{' '}
          <code>i</code> exist, and who can see each?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Một, dùng chung cho cả lưới.">One, shared by the whole grid.</Translated>,
        why: (
          <Translated vi="Khi đó 10,000 luồng sẽ ghi đè chỉ số của nhau. Một biến khai báo bên trong kernel không được dùng chung.">
            Then 10,000 threads would be overwriting each other's index. A variable declared inside the kernel is not shared.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Một bản mỗi khối, dùng chung cho các luồng của khối đó.">One per block, shared by the threads of that block.</Translated>,
        why: (
          <Translated vi="Việc dùng chung trong một khối có tồn tại, nhưng phải yêu cầu bằng một từ khóa bạn sẽ gặp ở chương 5. Một biến cục bộ thông thường không được dùng chung.">
            Sharing inside a block exists, but it has to be asked for with a keyword you meet in chapter 5. A plain local variable is not shared.
          </Translated>
        ),
      },
      {
        label: <Translated vi="10,000: mỗi luồng một bản, chỉ luồng đó nhìn thấy.">10,000: one per thread, visible only to that thread.</Translated>,
        correct: true,
      },
      {
        label: (
          <Translated vi="10,000, và bất kỳ luồng nào cũng có thể đọc bản của luồng khác qua blockIdx và threadIdx.">
            10,000, and any thread can read another thread's copy through blockIdx and threadIdx.
          </Translated>
        ),
        why: (
          <Translated vi="Các biến có sẵn cho phép một luồng tính ra i của luồng khác phải là bao nhiêu. Chúng không cho truy cập vào biến của luồng khác.">
            The built-in variables let a thread compute what another thread's i must be. They do not give access to the other thread's variable.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>i</code> là một biến tự động, và mỗi luồng nhận phiên bản riêng của mỗi biến tự động. Một luồng không
            thể đọc <code>i</code> của luồng khác. Nó chỉ có thể suy ra giá trị đó phải là bao nhiêu, vì công thức và tọa
            độ của luồng kia không phải bí mật. Những biến như vậy nằm ở đâu là chủ đề của{' '}
            <a href={href('ch05', '5.2')}>phần 5.2</a>.
          </p>
        }
      >
        <p>
          <code>i</code> is an automatic variable, and every thread gets its own version of each automatic variable. A
          thread cannot read another thread's <code>i</code>. It can only work out what it must be, since the formula and
          the other thread's coordinates are no secret. Where such variables live is{' '}
          <a href={href('ch05', '5.2')}>section 5.2</a>.
        </p>
      </Translated>
    ),
    source: '§2.5, p. 37',
  },
  'ch02/2.5/q-noguard': {
    kind: 'choice',
    id: 'ch02/2.5/q-noguard',
    prompt: (
      <Translated
        vi={
          <>
            n = 1000, khối 256 luồng, và có người xóa <code>if (i &lt; n)</code> khỏi kernel. Điều gì xảy ra?
          </>
        }
      >
        <>
          n = 1000, blocks of 256, and someone deletes <code>if (i &lt; n)</code> from the kernel. What happens?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không có gì. Runtime cắt bớt khối cuối chỉ còn những luồng có phần tử.">
            Nothing. The runtime clips the last block to the threads that have an element.
          </Translated>
        ),
        why: (
          <Translated vi="Runtime không biết n. Nó được cho một số khối và một kích thước khối, và nó chạy tất cả.">
            The runtime does not know n. It was given a number of blocks and a block size, and it runs all of them.
          </Translated>
        ),
      },
      {
        label: (
          <Translated
            vi={
              <>
                {NOGUARD.idle} luồng đọc và ghi các phần tử {span(NOGUARD.outOfRange)}, nằm ngoài cả ba mảng. Theo mặc định
                không có gì báo lỗi này trên GPU; AddressSanitizer của trình giả lập dừng lại ở dòng đó.
              </>
            }
          >
            <>
              {NOGUARD.idle} threads read and write elements {span(NOGUARD.outOfRange)}, outside all three arrays. Nothing
              reports it by default on a GPU; the emulator's AddressSanitizer stops at the line.
            </>
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Lệnh khởi chạy thất bại với một mã lỗi, vì lưới lớn hơn dữ liệu.">
            The launch fails with an error code, because the grid is larger than the data.
          </Translated>
        ),
        why: (
          <Translated vi="Các tham số khởi chạy hợp lệ. Câu lệnh khởi chạy không trả về mã lỗi trong mọi trường hợp, và thứ tài liệu mô tả việc kiểm tra sau đó là cấu hình.">
            The launch parameters are valid. The launch statement returns no error code in any case, and what the docs describe checking afterwards is the configuration.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một lỗi biên dịch, vì chỉ số có thể chạy quá cuối mảng.">
            A compile error, because the index can run past the arrays.
          </Translated>
        ),
        why: (
          <Translated vi="Trình biên dịch không thể biết các mảng sẽ dài bao nhiêu lúc chạy.">
            The compiler cannot know how long the arrays will be at run time.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            {NOGUARD.blocks} khối 256 luồng là {num(NOGUARD.launched)} luồng, và mỗi luồng đều chạy phép cộng. Các luồng có
            i từ {span(NOGUARD.outOfRange)} nạp dữ liệu từ quá cuối A và B và lưu quá cuối C. NVIDIA ghi trong tài liệu một
            mã lỗi cho thao tác nạp hoặc lưu tại một địa chỉ không hợp lệ. Một chỉ số chỉ vừa quá cuối một mảng vẫn có thể
            rơi vào bộ nhớ device đã được ánh xạ, vốn không phải địa chỉ không hợp lệ, và không trang tài liệu NVIDIA nào
            được kiểm tra cho khóa học này nói rằng mảng trên device được kiểm tra biên. Vì vậy không có gì bảo đảm sẽ được
            báo, và dữ liệu khác có thể bị ghi đè trong im lặng. Đây là lý do các lab còn chạy kernel của bạn trên một trình
            giả lập CPU được build với AddressSanitizer, công cụ dừng lại ở lần truy cập như vậy đầu tiên và chỉ ra dòng
            mã.
          </p>
        }
      >
        <p>
          {NOGUARD.blocks} blocks of 256 are {num(NOGUARD.launched)} threads, and every one of them runs the addition.
          Threads with i from {span(NOGUARD.outOfRange)} load from past the end of A and B and store past the end of C.
          NVIDIA documents an error code for a load or store at an invalid address. An index just past the end of an
          array can still land inside mapped device memory, which is not an invalid address, and none of the NVIDIA
          pages checked for this course says that device arrays are bounds-checked. So nothing is guaranteed to be
          reported, and other data may be
          overwritten in silence. This is why the labs also run your kernel on a CPU emulator built with
          AddressSanitizer, which stops at the first such access and names the line.
        </p>
      </Translated>
    ),
    source: '§2.5, p. 37; notes/today/ch02-facts.md, A6',
    vi: { source: '§2.5, p. 37; và notes/today/ch02-facts.md, A6' },
  },
  'ch02/2.5/q-metal-guard': {
    kind: 'choice',
    id: 'ch02/2.5/q-metal-guard',
    prompt: (
      <Translated
        vi={
          <>
            Cùng thân kernel một dòng đó dưới dạng một kernel Metal lấy chỉ số từ <code>[[thread_position_in_grid]]</code>.
            Khi nào nó cần phép kiểm tra biên?
          </>
        }
      >
        <>
          The same one-line body as a Metal kernel that takes its index from <code>[[thread_position_in_grid]]</code>.
          When does it need the bounds guard?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không bao giờ: Metal kiểm tra mỗi lần truy cập với độ dài của buffer.">
            Never: Metal checks each access against the buffer's length.
          </Translated>
        ),
        why: (
          <Translated vi="Không phải theo mặc định. Apple ghi trong tài liệu rằng việc báo truy cập ngoài biên là việc của Shader Validation, một công cụ gỡ lỗi mà bạn phải bật lên.">
            Not by default. Apple documents out-of-bounds reporting as a job for Shader Validation, a debugging tool you switch on.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Luôn luôn, y hệt như trong CUDA, bất kể host dùng lời gọi dispatch nào.">
            Always, exactly as in CUDA, whichever dispatch call the host uses.
          </Translated>
        ),
        why: (
          <Translated vi="Với dispatchThreads trên n luồng, không luồng nào có chỉ số bằng hoặc vượt n.">
            With dispatchThreads over n threads no thread has an index at or past n.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không cần khi host gọi dispatchThreads trên đúng n luồng. Cần khi dùng dispatchThreadgroups, lời gọi khởi chạy các threadgroup trọn vẹn.">
            Not when the host calls dispatchThreads over exactly n threads. It does with dispatchThreadgroups, which launches whole threadgroups.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chỉ khi n lớn hơn threadgroup lớn nhất.">
            Only when n is larger than the largest threadgroup.
          </Translated>
        ),
        why: (
          <Translated vi="Kích thước của một nhóm không quyết định điều đó. Điều quyết định là lưới có được làm tròn lên thành các nhóm trọn vẹn hay không.">
            The size of a group does not decide it. What decides it is whether the grid is rounded up to whole groups.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>dispatchThreads</code> nhận kích thước lưới tính bằng luồng, và Metal làm threadgroup cuối nhỏ hơn khi n
            không phải bội số của kích thước nhóm. Mẫu cộng vector của chính Apple được viết theo cách đó và không có phép
            kiểm tra biên. <code>dispatchThreadgroups</code> nhận một số nhóm, nên lưới là bội số của kích thước nhóm, như
            trong CUDA, và tài liệu của Apple bảo bạn kiểm tra biên trong kernel. llama.cpp dùng dạng thứ hai, nên các
            kernel Metal của nó trông giống kernel CUDA.
          </p>
        }
      >
        <p>
          <code>dispatchThreads</code> takes the size of the grid in threads, and Metal makes the last threadgroup
          smaller when n is not a multiple of the group size. Apple's own vector addition sample is written that way and
          has no guard. <code>dispatchThreadgroups</code> takes a number of groups, so the grid is a multiple of the
          group size, as in CUDA, and Apple's documentation tells you to check bounds in the kernel. llama.cpp uses the
          second form, so its Metal kernels look like CUDA kernels.
        </p>
      </Translated>
    ),
    source: 'not in the book; notes/today/ch02-facts.md, C1, C4, D2 and ch05-facts.md, C5',
    vi: { source: 'không có trong sách; notes/today/ch02-facts.md, C1, C4, D2 và ch05-facts.md, C5' },
  },
};

// ---------------------------------------------------------------- the qualifier table (book Fig. 2.11), from the model
const QUALIFIER_ROWS: Qualifiers[] = ['host', 'global', 'device'];
function QualifierTable() {
  return (
    <table className="data" data-table="qualifiers" style={{ maxWidth: '46rem' }}>
      <thead>
        <tr>
          <th><Translated vi="Qualifier">Qualifier</Translated></th>
          <th><Translated vi="Gọi được từ">Callable from</Translated></th>
          <th><Translated vi="Chạy trên">Runs on</Translated></th>
          <th><Translated vi="Được thực thi bởi">Executed by</Translated></th>
        </tr>
      </thead>
      <tbody>
        {QUALIFIER_ROWS.map((q) => {
          const rules = CALLERS.map((caller) => legalCall(caller, q, 'cuda'));
          const sure = rules.filter((r) => r.book === 'legal');
          const maybe = rules.some((r) => r.book === 'dynamic-parallelism');
          const from = sure.some((r) => r.caller === 'host-fn') ? tr('host code', 'mã host') : tr('device code', 'mã device');
          return (
            <tr key={q} data-qualifier={q}>
              <td>
                <code>{QUALIFIER_TEXT[q]}</code>
                {q === 'host' ? <span className="muted"><Translated vi=" (mặc định)"> (the default)</Translated></span> : null}
              </td>
              <td>
                {from}
                {maybe ? <span className="muted"><Translated vi=" (hoặc mã device, với dynamic parallelism)"> (or device code, with dynamic parallelism)</Translated></span> : null}
              </td>
              <td>{q === 'host' ? tr('the host', 'host') : tr('the device', 'device')}</td>
              <td>{tr(EXECUTOR_TEXT[sure[0].executedBy!], ({ 'caller-host-thread': 'luồng host gọi hàm', 'caller-device-thread': 'luồng device gọi hàm', 'new-grid': 'một lưới luồng device mới' } as Record<string, string>)[sure[0].executedBy!])}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export function S25() {
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  return (
    <>
      <Translated
        vi={
          <p>
            Hàm host của <a href={href('ch02', '2.4')}>phần 2.4</a> cấp phát, sao chép vào, sao chép ra và giải phóng, với
            một lỗ hổng ở giữa nơi công việc lẽ ra phải nằm. Phần này viết thứ lấp vào lỗ hổng đó. Một hàm kernel là đoạn
            mã mà mọi luồng của lưới chạy trong phần song song của chương trình. Có một chương trình và có nhiều luồng, mỗi
            luồng áp nó lên một phần dữ liệu khác nhau, đó là lý do sách xếp CUDA C++ vào loại single-program,
            multiple-data, hay SPMD.
          </p>
        }
      >
        <p>
          The host function of <a href={href('ch02', '2.4')}>section 2.4</a> allocates, copies in, copies out and frees,
          with a hole in the middle where the work should be. This section writes what goes in the hole. A kernel
          function is the code that every thread of the grid runs during the parallel part of the program. There is one
          program and there are many threads, each applying it to a different part of the data, which is why the book
          files CUDA C++ under single-program, multiple-data, or SPMD.
        </p>
      </Translated>
      <Source section="2.5" pages="32–33" />
      <Translated
        vi={
          <p>
            Hãy bắt đầu bằng cách xem toàn bộ phép biến đổi một lần. Phần còn lại của mục này mổ xẻ nó: các luồng được
            nhóm lại ra sao, một luồng tìm phần tử của nó thế nào, và vòng lặp còn lại gì.
          </p>
        }
      >
        <p>
          Start by watching the whole transformation once. The rest of the section takes it apart: how threads are
          grouped, how a thread finds its element, and what is left of the loop.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Hoạt cảnh vẽ tất cả luồng cùng lúc và đồng bộ, không có thời gian và không có warp. Các con số của nó lấy từ
              cùng mô hình với trình chạy từng bước bên dưới. Phần Metal chỉ cho thấy lần khởi chạy bằng{' '}
              <code>dispatchThreads</code>.
            </p>
          }
        >
          <p>
            The animation draws all threads at once and in step, with no timing and no warps. Its numbers come from the
            same model as the stepper below. The Metal part shows the <code>dispatchThreads</code> launch only.
          </p>
        </Translated>
      </Simplification>
      <LoopToGridScene />

      <h2><Translated vi="Lưới, khối và kích thước của chúng">Grids, blocks and their sizes</Translated></h2>
      <Translated
        vi={
          <p>
            Các luồng của một lần khởi chạy được tổ chức theo hai cấp. Một lưới (grid) là một mảng các khối luồng (block),
            và mọi khối của một lưới có cùng số luồng. Host chọn cả hai con số, bao nhiêu khối và bao nhiêu luồng mỗi khối,
            mỗi lần nó gọi kernel, nên cùng một kernel có thể được khởi chạy với hình dạng này ở chỗ này và hình dạng khác
            ở chỗ khác.
          </p>
        }
      >
        <p>
          The threads of a launch are organised on two levels. A grid is an array of thread blocks, and every block of
          a grid has the same number of threads. The host chooses both numbers, how many blocks and how many threads in
          each, every time it calls the kernel, so the same kernel can be launched with one shape in one place and
          another shape somewhere else.
        </p>
      </Translated>
      <Source section="2.5" pages="33" />
      <Translated
        vi={
          <p>
            Sách đặt kích thước một khối tối đa 1024 luồng "trên các hệ thống hiện nay". NVIDIA vẫn ghi như vậy trong tài
            liệu, và tài liệu đưa ra lý do: mọi luồng của một khối nằm trên một streaming multiprocessor (SM) và dùng
            chung tài nguyên của nó. Streaming multiprocessor là gì sẽ có ở <a href={href('ch04')}>chương 4</a>.
          </p>
        }
      >
        <p>
          The book puts the size of a block at up to 1024 threads "on current systems". That is still what NVIDIA
          documents, and the documentation gives the reason: all threads of a block live on one streaming multiprocessor
          and share its resources. What a streaming multiprocessor is comes in <a href={href('ch04')}>chapter 4</a>.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, A10, B5; ch05-facts.md, C3)', '2026-11-20 (notes/today/ch02-facts.md, A10, B5; và ch05-facts.md, C3)')}>
        <Translated
          vi={
            <p>
              CUDA: 1024 cho mọi device trong bảng của NVIDIA. HIP: AMD nói "typically 1024" (thường là 1024) và bảo hãy
              truy vấn <code>maxThreadsPerBlock</code>. Metal: tối đa 1024, đọc từ <code>maxTotalThreadsPerThreadgroup</code>{' '}
              của pipeline state của kernel.
            </p>
          }
        >
          <p>
            CUDA: 1024 for every device NVIDIA's table lists. HIP: AMD says "typically 1024" and to query{' '}
            <code>maxThreadsPerBlock</code>. Metal: up to 1024, read from <code>maxTotalThreadsPerThreadgroup</code> of
            the kernel's pipeline state.
          </p>
        </Translated>
      </Today>
      <YourLimits />
      <Translated
        vi={
          <p>
            Sách cũng khuyên nên chọn kích thước khối là bội số của 32 "vì lý do hiệu quả phần cứng" và hoãn giải thích
            lý do. Lý do là warp, nhóm luồng mà phần cứng thực thi cùng nhau, và đó là chủ đề của chương 4. Kích thước nào
            cũng hợp lệ. Một kích thước không phải bội số sẽ để một phần warp cuối không được dùng.
          </p>
        }
      >
        <p>
          The book also advises making the block size a multiple of 32 "for hardware efficiency reasons" and postpones
          the reason. The reason is the warp, the group of threads the hardware executes together, and it is chapter 4's
          subject. Any size is legal. A size that is not a multiple leaves part of the last warp unused.
        </p>
      </Translated>
      <Source section="2.5" pages="34" />
      <Translated
        vi={
          <p>
            Con số 32 là của NVIDIA. Trên AMD, bội số cần tôn trọng là kích thước warp của device, là 64 trên một số device
            và 32 trên các device khác. Trên Metal đó là <code>threadExecutionWidth</code>, đọc từ pipeline state. Mã được
            viết để chạy được nhiều nơi sẽ hỏi device và không viết cứng 32.
          </p>
        }
      >
        <p>
          The number 32 is NVIDIA's. On AMD the multiple to respect is the device's warp size, which is 64 on some
          devices and 32 on others. On Metal it is <code>threadExecutionWidth</code>, read from the pipeline state. Code
          that is meant to travel asks the device and does not write 32.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, A11; ch05-facts.md, B5, C3)', '2026-11-20 (notes/today/ch02-facts.md, A11; và ch05-facts.md, B5, C3)')}>
        <Translated
          vi={
            <p>
              NVIDIA nêu quy tắc bội số của 32 như một khuyến nghị: "It is legal to use any number of threads" (dùng số
              luồng bất kỳ đều hợp lệ). Hướng dẫn chuyển mã của AMD nói đừng giả định 32 hay 64. Apple không ghi giá trị cố
              định nào cho <code>threadExecutionWidth</code>.
            </p>
          }
        >
          <p>
            NVIDIA states the multiple-of-32 rule as a recommendation: "It is legal to use any number of threads". AMD's
            porting guide says not to assume 32 or 64. Apple documents no fixed value for{' '}
            <code>threadExecutionWidth</code>.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Một luồng biết mình là ai như thế nào">How a thread learns who it is</Translated></h2>
      <Translated
        vi={
          <p>
            Mọi luồng chạy cùng một đoạn mã, nên phải có thứ gì đó khác nhau giữa các luồng, nếu không chúng sẽ làm cùng một
            việc. Thứ khác nhau là một nhóm nhỏ biến có sẵn. Runtime đặt chúng trước khi một luồng bắt đầu, luồng chỉ có
            thể đọc chúng, và các luồng khác nhau thấy các giá trị khác nhau. Ba biến quan trọng ở đây.{' '}
            <code>blockDim</code> là số luồng trong một khối, giống nhau với mọi luồng. <code>blockIdx</code> là vị trí khối
            của một luồng trong lưới, chung cho mọi luồng của khối đó. <code>threadIdx</code> là vị trí của luồng bên trong
            khối của nó. Mỗi biến có ba trường, <code>x</code>, <code>y</code> và <code>z</code>, để khối và lưới có thể
            có một, hai hoặc ba chiều. Một vector chỉ cần <code>x</code>;{' '}
            <a href={href('ch03', '3.1')}>chương 3</a> dùng phần còn lại.
          </p>
        }
      >
        <p>
          Every thread runs the same code, so something has to differ between threads or they would all do the same
          work. What differs is a handful of built-in variables. The runtime sets them before a thread starts, a thread
          can only read them, and different threads see different values. Three matter here. <code>blockDim</code> is
          the number of threads in a block, the same for everyone. <code>blockIdx</code> is the position of a thread's
          block in the grid, shared by all threads of that block. <code>threadIdx</code> is the thread's position inside
          its block. Each has three fields, <code>x</code>, <code>y</code> and <code>z</code>, so that blocks and grids
          can have one, two or three dimensions. A vector needs only <code>x</code>;{' '}
          <a href={href('ch03', '3.1')}>chapter 3</a> uses the rest.
        </p>
      </Translated>
      <Source section="2.5" pages="33–34" />
      <Translated
        vi={
          <p>
            Sách so sánh cặp này với một số điện thoại: <code>blockIdx</code> là mã vùng,{' '}
            <code>threadIdx</code> là số máy địa phương, và cả hai cùng nhau dẫn tới đúng một đường dây trong cả nước.
          </p>
        }
      >
        <p>
          The book compares the pair to a telephone number: <code>blockIdx</code> is the area code,{' '}
          <code>threadIdx</code> the local number, and the two together reach one line in the whole country.
        </p>
      </Translated>
      <Source section="2.5" pages="34–35" />
      <Translated
        vi={
          <p>
            Từ cặp này một luồng tính ra một con số mà không luồng nào khác trong lưới có:{' '}
            <code>i = blockIdx.x * blockDim.x + threadIdx.x</code>. Các khối đứng trước khối của tôi chiếm{' '}
            <code>blockIdx.x * blockDim.x</code> phần tử, và tôi ở vị trí <code>threadIdx.x</code> trong khối của mình. Với
            256 luồng mỗi khối, khối 0 nhận i từ 0 đến 255, khối 1 từ 256 đến 511, khối 2 từ 512 đến 767.
          </p>
        }
      >
        <p>
          From the pair a thread computes one number that no other thread of the grid has:{' '}
          <code>i = blockIdx.x * blockDim.x + threadIdx.x</code>. The blocks before mine account for{' '}
          <code>blockIdx.x * blockDim.x</code> elements, and I am <code>threadIdx.x</code> places into my own. With 256
          threads per block, block 0 gets i from 0 to 255, block 1 from 256 to 511, block 2 from 512 to 767.
        </p>
      </Translated>
      <Source section="2.5" pages="33–34" figs="Fig. 2.9" />
      <Ask q={questionsS25['ch02/2.5/q-own']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Kernel">The kernel</Translated></h2>
      <Translated
        vi={
          <p>
            Đây là toàn bộ kernel. Hãy đọc theo màu trước: <code>A</code>, <code>B</code> và <code>C</code> trỏ vào bộ nhớ
            toàn cục, nơi mọi luồng đều chạm tới được, còn <code>i</code> nằm trong một thanh ghi (register) thuộc về một
            luồng. Từ khóa <code>__global__</code> ở phía trước là thứ khiến hàm trở thành một kernel; cuối phần này sẽ
            quay lại với nó.
          </p>
        }
      >
        <p>
          Here is the whole kernel. Read it by colour first: <code>A</code>, <code>B</code> and <code>C</code> point into
          global memory, which every thread can reach, and <code>i</code> sits in a register that belongs to one thread.
          The keyword <code>__global__</code> in front is what makes the function a kernel; the end of this section
          comes back to it.
        </p>
      </Translated>
      <Source section="2.5" pages="35" figs="Fig. 2.10" />
      <KernelListing cu={cu} metal={metal} />
      <Translated
        vi={
          <p>
            Mã này khác Fig. 2.10 ở một chỗ: hai đầu vào là <code>const</code>, nên trình biên dịch từ chối một kernel lỡ
            ghi vào chúng. Nó giữ thứ tự của hình cho chỉ số,{' '}
            <code>threadIdx.x + blockDim.x * blockIdx.x</code>, tức công thức trong văn bản với các toán hạng đổi chỗ và
            cùng giá trị. Ở chỗ khác trên các trang này, khung mã trong Fig. 2.9 không có phép kiểm tra biên và viết{' '}
            <code>A_d</code>, <code>B_d</code>, <code>C_d</code>, và ở p. 37 văn bản chỉ bạn tới "Fig. 2.10" cho vòng
            lặp, trong khi ý là Fig. 2.4, và gọi kernel là <code>vecAddVecKernel</code>.
          </p>
        }
      >
        <p>
          The listing differs from Fig. 2.10 in one place: the two inputs are <code>const</code>, so the compiler
          rejects a kernel that writes to them by mistake. It keeps the figure's order for the index,{' '}
          <code>threadIdx.x + blockDim.x * blockIdx.x</code>, which is the text's formula with the operands swapped and
          the same value. Elsewhere on the pages, the code box inside Fig. 2.9 has no guard and writes{' '}
          <code>A_d</code>, <code>B_d</code>, <code>C_d</code>, and on p. 37 the text sends you to "Fig. 2.10" for the
          loop, where Fig. 2.4 is meant, and calls the kernel <code>vecAddVecKernel</code>.
        </p>
      </Translated>
      <Aside>
        <Translated vi={<p>Ba điểm cuối trông như lỗi in. <a href={href('ch02', 'wrap')}>Trang tổng kết</a> của chương liệt kê tất cả.</p>}>
          <p>The last three look like slips in the print. The chapter's <a href={href('ch02', 'wrap')}>summary page</a> lists them all.</p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Bên trong kernel, các tên đã mất <code>_h</code> và <code>_d</code>. Một kernel chạy trên device và các con
            trỏ nó nhận là địa chỉ device, nên không có gì cần phân biệt.
          </p>
        }
      >
        <p>
          Inside the kernel the names have lost their <code>_h</code> and <code>_d</code>. A kernel runs on the device
          and the pointers it is given are device addresses, so there is nothing to tell apart.
        </p>
      </Translated>
      <Source section="2.5" pages="35" />
      <Translated
        vi={
          <p>
            Sách thêm trong một chú thích cuối trang rằng một kernel có thể chạm tới bộ nhớ host qua unified virtual
            addressing, với băng thông thấp hơn.
          </p>
        }
      >
        <p>
          The book adds in a footnote that a kernel can reach host memory through unified virtual addressing, at lower
          bandwidth.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, A15)', '2026-11-20 (notes/today/ch02-facts.md, A15)')}>
        <Translated
          vi={
            <p>
              Chú thích đó lỏng lẻo so với cách diễn đạt ngày nay. Unified virtual addressing nghĩa là các vùng cấp phát
              host và device dùng chung một không gian địa chỉ. Thứ cho phép một kernel chạm vào bộ nhớ host là bộ nhớ host
              được ánh xạ (khóa trang), hoặc unified memory. Băng thông thấp hơn được xác nhận. Chi tiết ở{' '}
              <a href={href('appC')}>phụ lục C</a>.
            </p>
          }
        >
          <p>
            The footnote is loose by today's wording. Unified virtual addressing means host and device allocations share
            one address space. What lets a kernel touch host memory is mapped (page-locked) host memory, or unified
            memory. The lower bandwidth is confirmed. Details are in <a href={href('appC')}>appendix C</a>.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Chạy nó trên mọi luồng">Run it on every thread</Translated></h2>
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy ba câu lệnh của kernel trên mọi luồng của một lần khởi chạy nhỏ cùng lúc: dòng chỉ
            số, phép kiểm tra biên và phép cộng. Hãy bước một lần, tới dòng chỉ số, và nhấn vào vài luồng: cùng một đoạn
            mã, hai số có sẵn khác nhau, một <code>i</code> khác nhau. Trước khi bước tới phép kiểm tra biên, hãy nói những
            luồng nào sẽ dừng ở đó.
          </p>
        }
      >
        <p>
          The stepper runs the three statements of the kernel on every thread of a small launch at once: the index
          line, the guard and the addition. Step once, to the index line, and click a few threads: the same code, two
          different built-in numbers, a different <code>i</code>. Before you step to the guard, say which threads will
          stop there.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mọi luồng cùng tiến một dòng mã nguồn. Đó là một tiện lợi khi vẽ và không phải lời hứa nào về cách GPU lập
              lịch các luồng (chương 4). Không có thời gian và không có warp. Khi tắt phép kiểm tra biên, các truy cập quá
              cuối được đánh dấu ở đây; tài liệu không nói một GPU thật sẽ báo chúng.
            </p>
          }
        >
          <p>
            All threads advance one source line together. That is a drawing convenience and no promise about how a GPU
            schedules threads (chapter 4). There is no timing and there are no warps. With the guard off, accesses past
            the end are marked here; a real GPU is not documented to report them.
          </p>
        </Translated>
      </Simplification>
      <KernelStepper cu={cu} />
      <Translated
        vi={
          <p>
            <code>i</code> trong mỗi ô là ý nghĩa của dòng chỉ số. Nó được khai báo bên trong kernel, điều đó khiến nó là
            một biến tự động, và mỗi luồng nhận phiên bản riêng của nó. Một lưới 10,000 luồng có 10,000 phiên bản của{' '}
            <code>i</code>, và không phiên bản nào luồng khác nhìn thấy được.
          </p>
        }
      >
        <p>
          The <code>i</code> in each box is the point of the index line. It is declared inside the kernel, which makes it an
          automatic variable, and every thread gets its own version. A grid of 10,000 threads has 10,000 versions of{' '}
          <code>i</code>, and none of them is visible to any other thread.
        </p>
      </Translated>
      <Source section="2.5" pages="37" />
      <Ask q={questionsS25['ch02/2.5/q-private']} />
      <Translated
        vi={
          <p>
            Một chi tiết của khai báo đó. Các trường của biến có sẵn là số không dấu, nên lưu chỉ số vào một{' '}
            <code>int</code> sẽ chuyển đổi nó. Một chú thích cuối trang trong sách cân nhắc hai kiểu: <code>unsigned int</code>{' '}
            nói rõ ý định và có thể đem lại cảnh báo hữu ích, còn <code>int</code> tránh bất ngờ trong phép trừ và số học
            trộn kiểu và cho phép trình biên dịch giả định không tràn số. Sách dùng cả hai. Hướng dẫn của NVIDIA lưu chỉ số
            trong một <code>int</code>; kernel llama.cpp ở phía dưới dùng <code>uint32_t</code>.
          </p>
        }
      >
        <p>
          One detail of that declaration. The fields of the built-in variables are unsigned, so storing the index in an{' '}
          <code>int</code> converts it. A footnote in the book weighs the two types: <code>unsigned int</code> states
          the intent and can earn useful warnings, while <code>int</code> avoids surprises in subtraction and mixed
          arithmetic and lets the compiler assume no overflow. The book uses both. NVIDIA's guide stores the index in an{' '}
          <code>int</code>; the llama.cpp kernel further down uses <code>uint32_t</code>.
        </p>
      </Translated>
      <Source section="2.5" pages="37" />

      <h2><Translated vi="Vòng lặp đã đi đâu?">Where did the loop go?</Translated></h2>
      <Translated
        vi={
          <p>
            Kernel không có vòng lặp. Lưới thay thế nó: mỗi luồng là một lần lặp của vòng lặp gốc, và lệnh khởi chạy tạo
            ra tất cả. Sách gọi đây là song song vòng lặp (loop parallelism). Trong phần đầu vòng lặp, giá trị khởi đầu và
            bước tăng đã biến thành hai số có sẵn. Phần thứ ba, phép kiểm tra <code>i &lt; n</code>, vẫn còn đó, dưới dạng
            phép kiểm tra biên của kernel.
          </p>
        }
      >
        <p>
          The kernel has no loop. The grid replaces it: each thread is one iteration of the original loop, and the
          launch creates them all. The book calls this loop parallelism. Of the loop header, the starting value and the
          increment have turned into the two built-in numbers. The third part, the test <code>i &lt; n</code>, is still
          there, as the kernel's guard.
        </p>
      </Translated>
      <Source section="2.5" pages="37" />

      <h2><Translated vi="Phép kiểm tra biên">The guard</Translated></h2>
      <Translated
        vi={
          <p>
            Phép kiểm tra cần thiết vì các khối luôn đến trọn vẹn. Độ dài của một vector hiếm khi là bội số của kích thước
            khối, nên khối cuối thường có những luồng không có phần tử nào để làm. Trường hợp của sách là 100 phần tử với
            khối 32 luồng, kích thước mà sách gọi là kích thước khối hiệu quả nhỏ nhất.
          </p>
        }
      >
        <p>
          The test is needed because blocks come whole. The length of a vector is rarely a multiple of the block size,
          so the last block usually has threads with no element to work on. The book's case is 100 elements with blocks
          of 32, which it calls the smallest efficient block size.
        </p>
      </Translated>
      <Source section="2.5" pages="37" />
      <Predict
        id="ch02/2.5/p-idle"
        prompt={
          <Translated
            vi={
              <>
                n = 100 và mỗi khối có 32 luồng. Host khởi chạy số khối ít nhất phủ được dữ liệu. Bao nhiêu luồng không
                vượt qua phép kiểm tra biên?
              </>
            }
          >
            <>
              n = 100 and each block has 32 threads. The host launches the fewest blocks that cover the data. How many
              threads fail the guard?
            </>
          </Translated>
        }
        unit={tr('threads', 'luồng')}
      >
        {(v) => (
          <Translated
            vi={
              <>
                <p>
                  {P37.blocks - 1} khối sẽ cho {(P37.blocks - 1) * 32} luồng, không đủ, nên host khởi chạy {P37.blocks}{' '}
                  khối: {P37.blocks} × 32 = {P37.launched} luồng. Mỗi luồng đều tính <code>i</code> và kiểm tra nó. 100
                  luồng đầu làm một phép cộng. <b>{P37.idle}</b> luồng còn lại, tất cả trong khối {P37.blocks - 1}, không
                  làm gì.
                </p>
                <p className="muted">
                  {v === P37.idle
                    ? 'Đúng như bạn dự đoán.'
                    : v === P37.launched
                      ? `${P37.launched} là số luồng được khởi chạy. Câu hỏi là bao nhiêu luồng trong số đó không có phần tử.`
                      : `Bạn đã dự đoán ${v}. Hãy nạp trường hợp đầu tiên của sách vào trình khám phá bên dưới và đọc các bộ đếm.`}
                </p>
              </>
            }
          >
            <>
              <p>
                {P37.blocks - 1} blocks would give {(P37.blocks - 1) * 32} threads, which is not enough, so the host
                launches {P37.blocks}: {P37.blocks} × 32 = {P37.launched} threads. Every one of them computes{' '}
                <code>i</code> and tests it. The first 100 do an addition. The other <b>{P37.idle}</b>, all in block{' '}
                {P37.blocks - 1}, do nothing.
              </p>
              <p className="muted">
                {v === P37.idle
                  ? 'That is what you predicted.'
                  : v === P37.launched
                    ? `${P37.launched} is the number of threads launched. The question was how many of them have no element.`
                    : `You predicted ${v}. Load the first of the book's cases in the explorer below and read the counters.`}
              </p>
            </>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Chính phép kiểm tra đó cho phép một kernel duy nhất xử lý vector có độ dài bất kỳ. Trình khám phá đếm giống
            vậy ở mọi kích thước. Hãy nạp từng trường hợp trong hai trường hợp của sách, rồi kéo kích thước khối và quan sát
            hai điều: ranh giới khối rơi vào đâu trên dữ liệu, và số luồng rảnh thay đổi thế nào.
          </p>
        }
      >
        <p>
          That one test is what lets a single kernel handle a vector of any length. The explorer does the same counting
          at any size. Load each of the book's two cases, then drag the block size and watch two things: where the block
          boundaries fall on the data, and how the number of idle threads moves.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Trình khám phá đếm luồng và phần tử. Nó không nói gì về tốc độ chạy của bất kỳ kích thước khối nào, và không
              kiểm tra kích thước khối với giới hạn của bất kỳ device nào.
            </p>
          }
        >
          <p>
            The explorer counts threads and elements. It says nothing about how fast any block size runs, and it does
            not check the block size against any device's limit.
          </p>
        </Translated>
      </Simplification>
      <IndexExplorer />
      <Translated
        vi={
          <p>
            Ba bài tập đầu tiên của chương nói về ánh xạ này. Bài thứ nhất hỏi chính nó. Hai bài còn lại thay đổi những
            phần tử mà một luồng sở hữu, đây là cái nhìn đầu tiên về việc giao cho một luồng nhiều hơn một phần tử công
            việc.
          </p>
        }
      >
        <p>
          The first three exercises of the chapter are about this mapping. The first asks for it. The other two change
          which elements a thread owns, which is a first look at giving a thread more than one element of work.
        </p>
      </Translated>
      <Source pages="41" />
      <Ask q={questionsS25['ch02/2.5/q-index']} label={tr('Book exercise 1', 'Bài tập 1 của sách')} />
      <Ask q={questionsS25['ch02/2.5/q-two-adjacent']} label={tr('Book exercise 2', 'Bài tập 2 của sách')} />
      <Ask q={questionsS25['ch02/2.5/q-two-sections']} label={tr('Book exercise 3', 'Bài tập 3 của sách')} />
      <Translated
        vi={
          <p>
            Trong cả ba ánh xạ, số luồng rảnh ít hơn một khối, và phép kiểm tra biên có một phép thử cho mỗi lần truy cập.
            Giờ hãy bỏ phép kiểm tra biên đi. Tắt nó trong trình chạy từng bước ở trên với n = 10 và khối 4 luồng, bước tới
            phép cộng, rồi trả lời cho trường hợp lớn hơn của sách.
          </p>
        }
      >
        <p>
          In all three mappings the idle threads number fewer than one block, and the guard has one test per access.
          Now take the guard away. Switch it off in the stepper above with n = 10 and blocks of 4, step to the addition, and
          then answer for the book's larger case.
        </p>
      </Translated>
      <Ask q={questionsS25['ch02/2.5/q-noguard']} />

      <h2><Translated vi="Cùng kernel đó trên HIP và Metal">The same kernel on HIP and Metal</Translated></h2>
      <Translated
        vi={
          <p>
            Với kernel này, mã nguồn HIP chính là mã nguồn CUDA. <code>__global__</code>, ba biến có sẵn và phép kiểm tra
            biên được viết giống hệt, và kernel ví dụ của chính AMD có cùng hai dòng:{' '}
            <code>{AMD_INDEX}</code> theo sau là <code>{AMD_GUARD}</code>. Thứ duy nhất thêm vào là một header, vì HIP không
            định nghĩa các biến có sẵn khi thiếu <code>hip/hip_runtime.h</code>; các lab đã include nó cho bạn.
          </p>
        }
      >
        <p>
          For this kernel the HIP source is the CUDA source. <code>__global__</code>, the three built-in variables and
          the guard are spelled the same, and AMD's own example kernel has the same two lines:{' '}
          <code>{AMD_INDEX}</code> followed by <code>{AMD_GUARD}</code>. The one addition is a header, since HIP does
          not define the built-ins without <code>hip/hip_runtime.h</code>; the labs include it for you.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. Những gì phần này nói về HIP và Metal được kiểm tra với tài liệu của nhà cung cấp
              (notes/today/ch02-facts.md, B4, C1, C4; ch05-facts.md, C5). Kernel Metal chạy trong lab 2.1; không gì ở đây
              đã chạy trên phần cứng HIP.
            </p>
          }
        >
          <p>
            The book uses CUDA only. What this part says about HIP and Metal is checked against vendor documentation
            (notes/today/ch02-facts.md, B4, C1, C4; ch05-facts.md, C5). The Metal kernel runs in lab 2.1; nothing here
            has run on HIP hardware.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Metal khác ở ba chỗ, đều thấy được trên tab Metal của mã ở trên. Điểm vào được đánh dấu{' '}
            <code>kernel void</code>. Mỗi con trỏ mang một từ khóa không gian địa chỉ, <code>device</code>, và được gắn với
            một khe buffer có đánh số thay vì được truyền theo giá trị. Và chỉ số không được tính: runtime tính vị trí của
            mỗi luồng trong lưới và truyền nó vào một đối số được đánh dấu{' '}
            <code>[[thread_position_in_grid]]</code>. Đặc tả Metal Shading Language đưa ra công thức cho vị trí đó, và đó
            chính là công thức của phần này.
          </p>
        }
      >
        <p>
          Metal differs in three places, all visible on the Metal tab of the listing above. The entry point is marked{' '}
          <code>kernel void</code>. Every pointer carries an address-space keyword, <code>device</code>, and is bound to
          a numbered buffer slot instead of being passed by value. And the index is not computed: the runtime works out
          each thread's position in the grid and passes it in an argument marked{' '}
          <code>[[thread_position_in_grid]]</code>. The Metal Shading Language specification gives the formula for that
          position, and it is this section's formula.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Một kernel Metal có cần phép kiểm tra biên hay không phụ thuộc vào cách host khởi động nó.{' '}
            <code>dispatchThreads</code> nhận kích thước lưới tính bằng luồng, nên lưới có thể đúng bằng n.{' '}
            <code>dispatchThreadgroups</code> nhận một số nhóm trọn vẹn, như một lần khởi chạy CUDA, và khi đó phép kiểm
            tra biên quay trở lại.
          </p>
        }
      >
        <p>
          Whether a Metal kernel needs the guard depends on how the host starts it. <code>dispatchThreads</code> takes
          the size of the grid in threads, so the grid can be exactly n. <code>dispatchThreadgroups</code> takes a number
          of whole groups, as a CUDA launch does, and then the guard is back.
        </p>
      </Translated>
      <Ask q={questionsS25['ch02/2.5/q-metal-guard']} />
      <Translated
        vi={
          <p>
            Bạn đã gặp kernel này trước đây mà không thấy nó. Khi một mô hình nhân một tensor với một hằng số và cộng một
            độ lệch, backend CUDA của llama.cpp chạy một kernel có các dòng đầu là chỉ số và phép kiểm tra biên của phần
            này, với phép kiểm tra được viết dưới dạng return sớm, theo sau là một dòng số học:
          </p>
        }
      >
        <p>
          You have met this kernel before without seeing it. When a model multiplies a tensor by a constant and adds an
          offset, llama.cpp's CUDA backend runs a kernel whose first lines are the index and the guard of this section,
          with the guard written as an early return, followed by one line of arithmetic:
        </p>
      </Translated>
      <Code
        tabs={[{ label: tr('llama.cpp, ggml-cuda/scale.cu', 'llama.cpp, ggml-cuda/scale.cu'), code: LLAMA_SCALE, status: tr('unverified: quoted from the project source as read on 2026-11-20, not built here', 'unverified: trích từ mã nguồn của dự án như đã đọc ngày 2026-11-20, không build ở đây') }]}
        caption={
          <Translated
            vi={
              <>
                Ba dòng của kernel trong <code>ggml/src/ggml-cuda/scale.cu</code>; các dòng ở giữa được lược bỏ
                (notes/today/ch02-facts.md, D1). Ở đây <code>k</code> đóng vai trò của <code>n</code>. Mốc dự án M0 trong{' '}
                <a href={href('ch02', 'labs')}>lab 2.3</a> yêu cầu bạn viết phép toán này.
              </>
            }
          >
            <>
              Three lines of the kernel in <code>ggml/src/ggml-cuda/scale.cu</code>; the lines between them are left out
              (notes/today/ch02-facts.md, D1). Here <code>k</code> plays the part of <code>n</code>. Milestone M0 in{' '}
              <a href={href('ch02', 'labs')}>lab 2.3</a> has you write this operation.
            </>
          </Translated>
        }
      />

      <h2><Translated vi="Ba loại hàm">Three kinds of function</Translated></h2>
      <Translated
        vi={
          <p>
            <code>__global__</code> là một trong ba qualifier mà CUDA C++ thêm vào khai báo hàm. Bảng của sách cho biết,
            với mỗi qualifier, hàm có thể được gọi từ đâu, chạy ở đâu và luồng nào chạy nó. Một hàm không có qualifier là
            hàm host, đó là lý do mã C++ thông thường không cần thay đổi gì.
          </p>
        }
      >
        <p>
          <code>__global__</code> is one of three qualifiers CUDA C++ adds to function declarations. The book's table
          says, for each, where the function may be called from, where it runs and which thread runs it. A function with
          no qualifier is a host function, which is why ordinary C++ code needs no changes.
        </p>
      </Translated>
      <Source section="2.5" pages="35–36" figs="Fig. 2.11" />
      <QualifierTable />
      <Translated
        vi={
          <p>
            Một hàm có thể mang cùng lúc <code>__host__</code> và <code>__device__</code>, và khi đó trình biên dịch build
            một phiên bản cho mỗi bên. Một kernel phải trả về <code>void</code> và không thể là thành viên của một lớp. Gọi
            một kernel từ mã device cần dynamic parallelism, thứ mà sách ghi là có từ kiến trúc Kepler; ấn bản này không còn
            chương nào về nó. <a href={href('ch02', '2.7')}>Phần 2.7</a> cho bạn đi qua bảng từng ô một, bên cạnh những gì
            trình biên dịch làm với mỗi qualifier.
          </p>
        }
      >
        <p>
          A function may carry <code>__host__</code> and <code>__device__</code> together, and the compiler then builds
          a version for each side. A kernel must return <code>void</code> and cannot be a member of a class. Calling a
          kernel from device code needs dynamic parallelism, which the book dates to the Kepler architecture; this
          edition no longer has a chapter on it. <a href={href('ch02', '2.7')}>Section 2.7</a> has you work the table
          cell by cell, next to what the compiler does with each qualifier.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, A7, A13)', '2026-11-20 (notes/today/ch02-facts.md, A7 và A13)')}>
        <Translated
          vi={
            <p>
              Bảng hiện tại của NVIDIA có thêm hai hàng, <code>__tile__</code> và <code>__tile_global__</code>, cho CUDA
              Tile C++ (toolkit 13.3 trở đi). Khóa học này không dùng chúng. Không xác nhận được tên "Kepler": hướng dẫn
              lưu trữ ghi compute capability 3.5 trở lên.
            </p>
          }
        >
          <p>
            NVIDIA's current table has two more rows, <code>__tile__</code> and <code>__tile_global__</code>, for CUDA
            Tile C++ (toolkit 13.3 onward). This course does not use them. "Kepler" could not be confirmed by name: the
            archived guide says compute capability 3.5 and higher.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Kernel đã được viết. Thứ vẫn còn thiếu là câu lệnh khởi động nó, cùng hai con số định hình lưới:{' '}
            <a href={href('ch02', '2.6')}>phần 2.6</a>.
          </p>
        }
      >
        <p>
          The kernel is written. What is still missing is the statement that starts it, with the two numbers that shape
          the grid: <a href={href('ch02', '2.6')}>section 2.6</a>.
        </p>
      </Translated>
    </>
  );
}
