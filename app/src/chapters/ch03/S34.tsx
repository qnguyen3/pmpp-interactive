import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, MATMUL_TIERS, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Aside, Bench, Beyond, Simplification, Source, TierLegend, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { benchRows, fmt, labStatus, num, realBackends, useReports, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  CUDA_DOCUMENTED,
  blockInputs,
  cellsOf,
  counts,
  granularity,
  gridFor,
  makeProblem,
  makeRect,
  maxWidth,
  maxWidthInt32,
  owners,
  program,
  threadFor,
  trace,
  traceRect,
  type Layout,
  type Limits,
  type Mode,
  type Op,
  type YX,
} from '../../sim/naivemm';
import { SixteenThreadsScene } from './SixteenThreadsScene';

// The kernel of book Fig. 3.11 (p. 61), line for line, and the launch the book does
// not print. Both are labs/ch03/snippets/matmul_kernel.cu, which the CUDA front-end
// check parses; S34.test.ts fails if this copy and that file drift apart.
const KERNEL = `__global__ void MatrixMulKernel(float* M, float* N,
                                float* P, int Width) {
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if ((row < Width) && (col < Width)) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row*Width+k]*N[k*Width+col];
        }
        P[row*Width+col] = Pvalue;
    }
}`;
const LAUNCH = `#define BLOCK_WIDTH 16

// Not printed in the book: a square grid of square blocks, rounded up to whole blocks.
void launchMatrixMul(float* M_d, float* N_d, float* P_d, int Width) {
    int blocks = (Width + BLOCK_WIDTH - 1) / BLOCK_WIDTH;   // ceil(Width / BLOCK_WIDTH)
    dim3 dimGrid(blocks, blocks, 1);                        // x covers the columns of P, y its rows
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    MatrixMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
}`;
const SNIPPET = 'labs/ch03/snippets/matmul_kernel.cu';
// Book exercise 2 (p. 64): the four kernel bodies the question offers, one per kernel of
// labs/ch03/snippets/matvec_options.cu. S34.test.ts fails if these copies and that file drift apart.
const MATVEC_1 = `int i = blockIdx.x*blockDim.x + threadIdx.x;
float s = 0;
for (int j = 0; j < N; ++j) s += B[i*N + j]*c[j];
a[i] = s;`;
const MATVEC_2 = `int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < N) {
    float s = 0;
    for (int j = 0; j < N; ++j) s += B[i*N + j]*c[j];
    a[i] = s;
}`;
const MATVEC_3 = `int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < N) {
    float s = 0;
    for (int j = 0; j < N; ++j) s += B[j*N + i]*c[j];
    a[i] = s;
}`;
const MATVEC_4 = `int i = blockIdx.y*blockDim.y + threadIdx.y;
int j = blockIdx.x*blockDim.x + threadIdx.x;
if (i < N && j < N) a[i] += B[i*N + j]*c[j];`;
/** Line 08, the loop body, as the animation's last beat shows it. */
const LOOP_LINE = KERNEL.split('\n')[7].trim();
/** Source line of Fig. 3.11 that each stepper operation executes. */
const LINE: Record<Op['op'], number> = { row: 3, col: 4, guard: 5, init: 6, mul: 8, store: 10 };

const LAB3 = 'ch03/lab3_naive_gemm';
const LAB4 = 'ch03/lab4_granularity';

const big = (v: number) => v.toLocaleString('en-US');
const yx = (p: YX) => `(${p.y},${p.x})`;
const plural = (n: number, one: string, many = one + 's') => `${big(n)} ${n === 1 ? one : many}`;
const range = (xs: number[]) => (xs.length === 1 ? `${xs[0]}` : tr(`${xs[0]} to ${xs[xs.length - 1]}`, `${xs[0]} đến ${xs[xs.length - 1]}`));
/** A caption above a group of buttons (a <label> would forward clicks on the caption to the first button). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };

// ---------------------------------------------------------------- interactive 1: the hand trace
const TW = 4; // Width of the book's example (p. 62)
const TB = 2; // its BLOCK_WIDTH
const PAIRS: YX[] = [
  { y: 0, x: 0 },
  { y: 0, x: 1 },
  { y: 1, x: 0 },
  { y: 1, x: 1 },
];

/** One sentence per executed line, with the index arithmetic evaluated. */
function describe(op: Op, block: YX, thread: YX, row: number, col: number): string {
  switch (op.op) {
    case 'row':
      return tr(`Line 03. row = blockIdx.y·blockDim.y + threadIdx.y = ${block.y}·${TB} + ${thread.y} = ${op.value}.`, `Dòng 03. row = blockIdx.y·blockDim.y + threadIdx.y = ${block.y}·${TB} + ${thread.y} = ${op.value}.`);
    case 'col':
      return tr(`Line 04. col = blockIdx.x·blockDim.x + threadIdx.x = ${block.x}·${TB} + ${thread.x} = ${op.value}.`, `Dòng 04. col = blockIdx.x·blockDim.x + threadIdx.x = ${block.x}·${TB} + ${thread.x} = ${op.value}.`);
    case 'guard':
      return op.pass
        ? tr(`Line 05. (${row} < ${TW}) && (${col} < ${TW}) is true, so this thread owns P[${row}][${col}]. It will read row ${row} of M and column ${col} of N.`, `Dòng 05. (${row} < ${TW}) && (${col} < ${TW}) là true, nên luồng này sở hữu P[${row}][${col}]. Nó sẽ đọc hàng ${row} của M và cột ${col} của N.`)
        : tr(`Line 05. The guard is false: this thread owns no element and does nothing more.`, `Dòng 05. Phép kiểm tra là false: luồng này không sở hữu phần tử nào và không làm gì thêm.`);
    case 'init':
      return tr('Line 06. Pvalue = 0. It is a local variable, so every thread has its own copy in a register.', 'Dòng 06. Pvalue = 0. Nó là một biến cục bộ, nên mỗi luồng có bản sao riêng của nó trong một thanh ghi.');
    case 'mul':
      return tr(
        `Line 08, k = ${op.k}. M[row·Width + k] = M[${row}·${TW} + ${op.k}] = M[${op.mIndex}] = ${op.mValue}. ` +
        `N[k·Width + col] = N[${op.k}·${TW} + ${col}] = N[${op.nIndex}] = ${op.nValue}. ` +
        `Pvalue = ${op.sum - op.product} + ${op.mValue} × ${op.nValue} = ${op.sum}.`,
        `Dòng 08, k = ${op.k}. M[row·Width + k] = M[${row}·${TW} + ${op.k}] = M[${op.mIndex}] = ${op.mValue}. ` +
          `N[k·Width + col] = N[${op.k}·${TW} + ${col}] = N[${op.nIndex}] = ${op.nValue}. ` +
          `Pvalue = ${op.sum - op.product} + ${op.mValue} × ${op.nValue} = ${op.sum}.`,
      );
    case 'store':
      return tr(`Line 10. P[row·Width + col] = P[${row}·${TW} + ${col}] = P[${op.index}] = ${op.value}. One store, and the thread is finished.`, `Dòng 10. P[row·Width + col] = P[${row}·${TW} + ${col}] = P[${op.index}] = ${op.value}. Một lần ghi, và luồng đã xong.`);
  }
}

/** A matrix as the book draws it: rows and columns. */
function Mx2D({ name, values, cls, show }: { name: string; values: number[]; cls: (i: number) => string; show?: (i: number) => boolean }) {
  return (
    <div data-view="2d" data-array={name}>
      <div className="mx-label t-global">{name}</div>
      <div className="mx global" style={{ gridTemplateColumns: `repeat(${TW}, auto)` }}>
        {values.map((v, i) => (
          <span key={i} className={cls(i)} data-i={i}>
            {!show || show(i) ? v : ''}
          </span>
        ))}
      </div>
    </div>
  );
}

/** The same matrix as memory holds it: one row of slots, the flat index under each value. */
function MxFlat({ name, values, cls, show }: { name: string; values: number[]; cls: (i: number) => string; show?: (i: number) => boolean }) {
  return (
    <div data-view="flat" data-array={name} style={{ marginTop: '0.7rem' }}>
      <div className="mx-label t-global">
        <Translated vi={<>{name} khi lưu <span className="muted" style={{ fontWeight: 400 }}>(số nhỏ là chỉ số phẳng)</span></>}>
          {name} as stored <span className="muted" style={{ fontWeight: 400 }}>(the small number is the flat index)</span>
        </Translated>
      </div>
      <div className="mx global" style={{ gridTemplateColumns: `repeat(${TW * TW}, auto)` }}>
        {values.map((v, i) => (
          <span
            key={i}
            className={cls(i)}
            data-i={i}
            style={{
              minWidth: '1.5rem',
              height: '2.3rem',
              paddingBottom: '0.62rem',
              marginLeft: i > 0 && i % TW === 0 ? 5 : 0,
              fontSize: v >= 100 ? '0.66rem' : undefined,
            }}
          >
            {!show || show(i) ? v : ''}
            <sub>{i}</sub>
          </span>
        ))}
      </div>
    </div>
  );
}

function HandTrace() {
  const prob = useMemo(() => makeProblem(TW), []);
  const [block, setBlock] = useState<YX>(PAIRS[0]);
  const [thread, setThread] = useState<YX>(PAIRS[0]);
  const [n, setN] = useState(0);
  const ops = useMemo(() => program(prob, TB, block, thread), [prob, block, thread]);
  const { row, col } = threadFor(TB, block, thread);
  useEffect(() => setN(0), [block, thread]);

  const total = ops.length;
  const at = Math.min(n, total);
  const cur: Op | undefined = at > 0 ? ops[at - 1] : undefined;
  const done = ops.slice(0, at);
  const knowsRow = at >= 1;
  const knowsCol = at >= 2;
  const muls = done.filter((o): o is Extract<Op, { op: 'mul' }> => o.op === 'mul');
  const store = done.find((o): o is Extract<Op, { op: 'store' }> => o.op === 'store');
  const pvalue = muls.length ? muls[muls.length - 1].sum : done.some((o) => o.op === 'init') ? 0 : null;
  const isBook = block.y === 0 && block.x === 0 && thread.y === 0 && thread.x === 0;

  const mCls = (i: number) => (cur?.op === 'mul' && cur.mIndex === i ? 'hot' : knowsRow && Math.floor(i / TW) === row ? 'on' : '');
  const nCls = (i: number) => (cur?.op === 'mul' && cur.nIndex === i ? 'hot' : knowsCol && i % TW === col ? 'on' : '');
  const pCls = (i: number) => (store?.index === i ? 'good' : knowsCol && i === row * TW + col ? 'on' : 'void');
  const pShow = (i: number) => store?.index === i;
  const pValues = prob.ref.map((v, i) => (store?.index === i ? store.value : v));

  const pick = (kind: 'Block' | 'Thread', chosen: YX, set: (p: YX) => void) => (
    <span style={FIELD}>
      <Translated vi={<>{kind === 'Block' ? 'Khối' : 'Luồng'} {kind === 'Block' ? '(blockIdx.y, blockIdx.x)' : '(threadIdx.y, threadIdx.x)'}</>}>
        {kind} {kind === 'Block' ? '(blockIdx.y, blockIdx.x)' : '(threadIdx.y, threadIdx.x)'}
      </Translated>
      <span className="seg" role="group" aria-label={tr(kind, kind === 'Block' ? 'Khối' : 'Luồng')}>
        {PAIRS.map((p) => (
          <button key={yx(p)} aria-pressed={chosen.y === p.y && chosen.x === p.x} onClick={() => set(p)}>
            {yx(p)}
          </button>
        ))}
      </span>
    </span>
  );

  return (
    <Bench
      title={tr('Trace one thread by hand', 'Tự lần theo một luồng')}
      task={tr("Step the book's thread to the end. Then pick another one, and say each index before you press Step.", 'Chạy từng bước luồng của sách đến cuối. Rồi chọn một luồng khác, và nói từng chỉ số trước khi bấm Bước.')}
      foot={
        <Translated
          vi={
            <>
              Width = 4, BLOCK_WIDTH = 2, như trong Figs. 3.12 và 3.13 của sách. Các giá trị là số nguyên nhỏ để bạn có thể
              kiểm tra từng tổng. <TierLegend tiers={['reg', 'global']} />
            </>
          }
        >
          <>
            Width = 4, BLOCK_WIDTH = 2, as in the book's Figs. 3.12 and 3.13. Values are small integers so that you can
            check each sum. <TierLegend tiers={['reg', 'global']} />
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        {pick('Block', block, setBlock)}
        {pick('Thread', thread, setThread)}
        <span className="muted" data-readout="owner" style={{ paddingBottom: '0.36rem' }}>
          <Translated vi={<>Luồng {yx(thread)} của Khối {yx(block)}</>}>Thread {yx(thread)} of Block {yx(block)}</Translated>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setN(0)} disabled={at === 0}>
          <Translated vi="Bắt đầu lại">Restart</Translated>
        </button>
        <button onClick={() => setN(at - 1)} disabled={at === 0}>
          <Translated vi="Lùi">Back</Translated>
        </button>
        <button className="primary" onClick={() => setN(at + 1)} disabled={at >= total}>
          <Translated vi="Bước">Step</Translated>
        </button>
        <button onClick={() => setN(total)} disabled={at >= total}>
          <Translated vi="Chạy đến cuối">Run to the end</Translated>
        </button>
        <span className="muted" aria-live="polite" data-readout="step">
          <Translated vi={<>bước {at} trên {total}</>}>step {at} of {total}</Translated>
        </span>
      </div>

      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>
        {cur
          ? describe(cur, block, thread, row, col)
          : tr(`Nothing has run yet. Thread ${yx(thread)} of Block ${yx(block)} is about to find out which element of P is its own. Press Step.`, `Chưa có gì chạy. Luồng ${yx(thread)} của Khối ${yx(block)} sắp tìm ra phần tử nào của P là của nó. Bấm Bước.`)}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) max-content', alignItems: 'start' }}>
        <div style={{ minWidth: 0 }}>
          <Code tabs={[{ label: tr('kernel', 'kernel'), code: KERNEL, tiers: MATMUL_TIERS }]} highlight={cur ? [LINE[cur.op]] : []} />
        </div>
        <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'max-content max-content', gap: '0.8rem 1.2rem', alignItems: 'end', marginTop: '1.1rem' }}>
            <div className="panel" style={{ alignSelf: 'stretch', minWidth: '9.4rem' }} data-readout="registers">
              <div className="mx-label t-reg"><Translated vi="Thanh ghi của luồng này">This thread's registers</Translated></div>
              <div>row = {knowsRow ? row : '?'}</div>
              <div>col = {knowsCol ? col : '?'}</div>
              <div>k = {cur?.op === 'mul' ? cur.k : store ? TW : '?'}</div>
              <div>
                Pvalue = <b className="t-reg">{pvalue ?? '?'}</b>
              </div>
            </div>
            <Mx2D name="N" values={prob.N} cls={nCls} />
            <Mx2D name="M" values={prob.M} cls={mCls} />
            <Mx2D name="P" values={pValues} cls={pCls} show={pShow} />
          </div>
        </div>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'max-content minmax(0, 1fr)', alignItems: 'start', marginTop: '0.4rem' }}>
        <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
          <MxFlat name="M" values={prob.M} cls={mCls} />
          <MxFlat name="N" values={prob.N} cls={nCls} />
          <MxFlat name="P" values={pValues} cls={pCls} show={pShow} />
          {store ? (
            <p style={{ margin: '0.9rem 0 0' }} data-readout="result">
              {store.value === prob.ref[store.index] ? (
                <span className="chip ok">
                  <Translated vi={<>P[{store.index}] = {store.value} khớp với M × N</>}>P[{store.index}] = {store.value} matches M × N</Translated>
                </span>
              ) : (
                <span className="chip fail"><Translated vi={<>P[{store.index}] sai</>}>P[{store.index}] is wrong</Translated></span>
              )}
            </p>
          ) : null}
        </div>
        <div style={{ overflowX: 'auto', minWidth: 0, marginTop: '0.7rem' }}>
          <table className="threads" data-table="trace" aria-label={tr('Loop trace', 'Dấu vết vòng lặp')}>
            <caption className="muted" style={{ textAlign: 'left', captionSide: 'top', paddingBottom: '0.3rem', whiteSpace: 'normal' }}>
              {isBook
                ? tr("The book's table for this thread (p. 63), filling in as you step. The last two columns are ours.", 'Bảng của sách cho luồng này (p. 63), được điền dần khi bạn chạy từng bước. Hai cột cuối là của khóa học này.')
                : tr('The same table for the thread you picked.', 'Cùng bảng đó cho luồng bạn đã chọn.')}
            </caption>
            <thead>
              <tr>
                <th>k</th>
                <th>row*Width+k{/* i18n-ignore: index formula */}</th>
                <th>k*Width+col{/* i18n-ignore: index formula */}</th>
                <th><Translated vi="phần tử M">M element</Translated></th>
                <th><Translated vi="phần tử N">N element</Translated></th>
                <th><Translated vi="tích">product</Translated></th>
                <th>Pvalue{/* i18n-ignore: variable name */}</th>
              </tr>
            </thead>
            <tbody>
              {muls.map((s) => (
                <tr key={s.k} data-k={s.k} className={cur?.op === 'mul' && cur.k === s.k ? 'cur' : ''}>
                  <td>{s.k}</td>
                  <td data-col="m-index">
                    {row}·{TW}+{s.k} = {s.mIndex}
                  </td>
                  <td data-col="n-index">
                    {s.k}·{TW}+{col} = {s.nIndex}
                  </td>
                  <td className="t-global">
                    M[{s.mIndex}] = M<sub>{row},{s.k}</sub>
                  </td>
                  <td className="t-global">
                    N[{s.nIndex}] = N<sub>{s.k},{col}</sub>
                  </td>
                  <td>
                    {s.mValue} × {s.nValue}
                  </td>
                  <td className="t-reg" data-col="sum">
                    {s.sum}
                  </td>
                </tr>
              ))}
              {muls.length === 0 ? (
                <tr>
                  <td colSpan={7} className="muted">
                    <Translated vi="Vòng lặp chưa bắt đầu.">The loop has not started.</Translated>
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive 2: who feeds this element
/** Cell geometry shared by the three matrices, with a wider gap wherever a block boundary falls. */
function box(size: number, state: 'hot' | 'wash' | null, gapLeft: boolean, gapTop: boolean): CSSProperties {
  return {
    width: size,
    height: size,
    marginLeft: gapLeft ? 5 : 0,
    marginTop: gapTop ? 5 : 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 0,
    borderRadius: 2,
    border: state ? '1px solid var(--global)' : '1px solid var(--rule)', // i18n-ignore: CSS value
    background: state === 'hot' ? 'var(--global)' : state === 'wash' ? 'var(--global-wash)' : 'var(--cell)',
    color: state === 'hot' ? 'var(--page)' : 'var(--ink)',
    fontSize: size >= 30 ? '0.76rem' : size >= 24 ? '0.7rem' : '0.6rem',
    fontWeight: 450,
    lineHeight: 1,
    fontVariantNumeric: 'tabular-nums',
  };
}

function Matrix({ name, note, cols, children }: { name: string; note: string; cols: number; children: ReactNode }) {
  return (
    <figure style={{ margin: 0 }} data-matrix={name}>
      <figcaption style={{ marginBottom: 4 }}>
        <b className="t-global">{name}</b> <span className="muted">{note}</span>
      </figcaption>
      <div role="group" aria-label={tr(`matrix ${name}`, `ma trận ${name}`)} style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, max-content)`, gap: 2, justifyContent: 'start' }}>
        {children}
      </div>
    </figure>
  );
}

/** A buffer as memory holds it, drawn as one line of slots: only which slots are touched, not their values. */
function Strip({ id, name, cells, rowLen, lit }: { id: string; name: ReactNode; cells: number; rowLen: number; lit: number[] }) {
  const on = new Set(lit);
  return (
    <div style={{ marginTop: '0.8rem' }}>
      <div className="mx-label">{name}</div>
      <svg
        data-strip={id}
        data-lit={lit.join(',')}
        viewBox={`0 0 ${cells} 10`}
        preserveAspectRatio="none" /* i18n-ignore: SVG attribute value */
        role="img"
        aria-label={tr(`${cells} slots, ${lit.length} of them read`, `${cells} ô, ${lit.length} ô trong đó được đọc`)}
        style={{ display: 'block', width: `min(100%, ${cells * 22}px)`, height: 22 }}
      >
        {Array.from({ length: cells }, (_, i) => (
          <rect key={i} x={i + 0.08} y={0.5} width={0.84} height={9} fill={on.has(i) ? 'var(--global)' : 'var(--cell)'}
            stroke={on.has(i) ? 'var(--global)' : 'var(--rule)'} strokeWidth="1" vectorEffect="non-scaling-stroke" />
        ))}
        {Array.from({ length: Math.floor(cells / rowLen) + 1 }, (_, r) => (
          <line key={r} x1={r * rowLen} x2={r * rowLen} y1={0} y2={10} stroke="var(--ink-3)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
        ))}
      </svg>
    </div>
  );
}

const BW_CHOICES = [2, 4, 8];

function Feeds() {
  const [rect, setRect] = useState(false);
  const [w, setW] = useState(8);
  const [dims, setDims] = useState({ j: 3, k: 8, l: 6 });
  const [bw, setBw] = useState(2);
  const [layout, setLayout] = useState<Layout>('book');
  const [view, setView] = useState<'thread' | 'block'>('thread');
  const [sel, setSel] = useState({ row: 1, col: 2 });

  const j = rect ? dims.j : w;
  const k = rect ? dims.k : w;
  const l = rect ? dims.l : w;
  const r = useMemo(() => makeRect(j, k, l), [j, k, l]);
  const row = Math.min(sel.row, j - 1);
  const col = Math.min(sel.col, l - 1);
  const own = owners(bw, row, col);
  const inputs = blockInputs(j, l, bw, own.block);
  const t = traceRect(r, row, col, layout);
  const grid = gridFor(j, l, bw);
  const cnt = counts(j, k, l);

  // Which slots of each operand the selection reads, from the model's traces.
  const { litA, litB } = useMemo(() => {
    const a = new Set<number>();
    const b = new Set<number>();
    const pairs = view === 'thread' ? [[row, col]] : inputs.rows.flatMap((rr) => inputs.cols.map((cc) => [rr, cc]));
    for (const [rr, cc] of pairs)
      for (const s of traceRect(r, rr, cc, layout).steps) {
        a.add(s.aIndex);
        b.add(s.bIndex);
      }
    return { litA: [...a].sort((x, y) => x - y), litB: [...b].sort((x, y) => x - y) };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [r, row, col, layout, view, bw]);
  const inA = new Set(litA);
  const inB = new Set(litB);

  const nameA = rect ? 'A' : 'M';
  const nameB = layout === 'torch' ? 'W' : rect ? 'B' : 'N';
  const nameC = rect ? 'C' : 'P';
  const sizeName = (s: 'j' | 'k' | 'l') => (rect ? s : 'Width');
  const size = Math.max(j, k, l) <= 6 ? 30 : Math.max(j, k, l) <= 10 ? 24 : 18;
  // The second operand as drawn: k rows of l in the book's layout, l rows of k in PyTorch's.
  const bCols = layout === 'book' ? l : k;
  const second = layout === 'book' ? r.B : r.W;
  const secondLine = layout === 'book' ? tr(`column ${view === 'thread' ? col : range(inputs.cols)} of ${nameB}`, `cột ${view === 'thread' ? col : range(inputs.cols)} của ${nameB}`) : tr(`row ${view === 'thread' ? col : range(inputs.cols)} of ${nameB}`, `hàng ${view === 'thread' ? col : range(inputs.cols)} của ${nameB}`);
  const aSteps = t.steps.map((s) => s.aIndex);
  const bSteps = t.steps.map((s) => s.bIndex);

  const slider = (name: string, value: number, min: number, max: number, set: (v: number) => void) => (
    <label className="field">
      {name} = {value}
      <input type="range" min={min} max={max} value={value} aria-label={name} onChange={(e) => set(+e.target.value)} />
    </label>
  );

  return (
    <Bench
      title={tr('Who feeds this element', 'Ai cung cấp dữ liệu cho phần tử này')}
      task={tr(`Click an element of ${nameC}. Then switch on "rectangular" and find out which size does which job.`, `Bấm vào một phần tử của ${nameC}. Rồi bật "chữ nhật" và tìm xem kích thước nào làm việc nào.`)}
      foot={tr('Counts and indices only. Nothing here says how long any of it takes.', 'Chỉ có số đếm và chỉ số. Không có gì ở đây cho biết việc nào mất bao lâu.')}
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="check" style={{ paddingBottom: '0.36rem' }}>
          <input type="checkbox" checked={rect} onChange={(e) => setRect(e.target.checked)} /> <Translated vi="chữ nhật">rectangular</Translated>
        </label>
        {rect ? (
          <>
            {slider('j', dims.j, 1, 16, (v) => setDims({ ...dims, j: v }))}
            {slider('k', dims.k, 1, 16, (v) => setDims({ ...dims, k: v }))}
            {slider('l', dims.l, 1, 16, (v) => setDims({ ...dims, l: v }))}
          </>
        ) : (
          slider('Width', w, 4, 16, setW)
        )}
        <span style={FIELD}>
          BLOCK_WIDTH = {bw}
          <span className="seg" role="group" aria-label="BLOCK_WIDTH">
            {BW_CHOICES.map((x) => (
              <button key={x} aria-pressed={bw === x} onClick={() => setBw(x)}>
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
        <span style={FIELD}>
          <Translated vi="Bố cục trọng số">Weight layout</Translated>
          <span className="seg" role="group" aria-label={tr('Weight layout', 'Bố cục trọng số')}>
            <button aria-pressed={layout === 'book'} onClick={() => setLayout('book')}>
              <Translated vi="của sách: đọc dọc một cột">the book's: read down a column</Translated>
            </button>
            <button aria-pressed={layout === 'torch'} onClick={() => setLayout('torch')}>
              <Translated vi="của PyTorch: đọc dọc một hàng">PyTorch's: read along a row</Translated>
            </button>
          </span>
        </span>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.2rem', alignItems: 'flex-start' }}>
        <div style={{ overflowX: 'auto', maxWidth: '100%', paddingBottom: 2 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'max-content max-content', gap: 14, alignItems: 'end' }}>
            <div style={{ maxWidth: Math.max(size * k, 150), color: 'var(--ink-2)', fontSize: '0.78rem' }}>
              <span className="legend">
                <span>
                  <i style={{ background: 'var(--global-wash)', border: '1px solid var(--global)' }} />
                  <Translated vi="được phần đang chọn đọc">read by the selection</Translated>
                </span>
              </span>
            </div>
            <Matrix
              name={nameB}
              note={layout === 'book' ? `${sizeName('k')} × ${sizeName('l')} = ${k} × ${l}` : tr(`${sizeName('l')} × ${sizeName('k')} = ${l} × ${k}, one output per row`, `${sizeName('l')} × ${sizeName('k')} = ${l} × ${k}, mỗi hàng một đầu ra`)}
              cols={bCols}
            >
              {second.map((v, i) => {
                const rr = Math.floor(i / bCols);
                const cc = i % bCols;
                // Block boundaries follow the columns of the output: columns of B, rows of W.
                const gapL = layout === 'book' && cc > 0 && cc % bw === 0;
                const gapT = layout === 'torch' && rr > 0 && rr % bw === 0;
                return (
                  <div key={i} data-cell={`second-${i}`} data-on={inB.has(i) ? '1' : '0'} style={box(size, inB.has(i) ? 'wash' : null, gapL, gapT)}>
                    {v}
                  </div>
                );
              })}
            </Matrix>
            <Matrix name={nameA} note={`${sizeName('j')} × ${sizeName('k')} = ${j} × ${k}`} cols={k}>
              {r.A.map((v, i) => {
                const rr = Math.floor(i / k);
                return (
                  <div key={i} data-cell={`first-${i}`} data-on={inA.has(i) ? '1' : '0'} style={box(size, inA.has(i) ? 'wash' : null, false, rr > 0 && rr % bw === 0)}>
                    {v}
                  </div>
                );
              })}
            </Matrix>
            <Matrix name={nameC} note={tr(`${sizeName('j')} × ${sizeName('l')} = ${j} × ${l} (click an element)`, `${sizeName('j')} × ${sizeName('l')} = ${j} × ${l} (bấm vào một phần tử)`)} cols={l}>
              {r.C.map((_, i) => {
                const rr = Math.floor(i / l);
                const cc = i % l;
                const mine = rr === row && cc === col;
                const inBlock = Math.floor(rr / bw) === own.block.y && Math.floor(cc / bw) === own.block.x;
                const on = view === 'thread' ? mine : inBlock;
                return (
                  <button
                    key={i}
                    aria-label={`${nameC}[${rr}][${cc}]`}
                    aria-pressed={on}
                    onClick={() => setSel({ row: rr, col: cc })}
                    style={{
                      ...box(size, mine ? 'hot' : inBlock ? 'wash' : null, cc > 0 && cc % bw === 0, rr > 0 && rr % bw === 0),
                      cursor: 'pointer',
                      ...(mine ? { border: '2px solid var(--ink)' } : null),
                    }}
                  />
                );
              })}
            </Matrix>
          </div>
        </div>

        <div className="panel" style={{ flex: '1 1 17rem', minWidth: '15rem', maxWidth: '31rem' }} aria-live="polite">
          {view === 'thread' ? (
            <p style={{ margin: 0 }} data-readout="who">
              <Translated
                vi={
                  <>
                    <b>
                      {nameC}[{row}][{col}]
                    </b>{' '}
                    = {r.C[row * l + col]} thuộc về Luồng {yx(own.thread)} của Khối {yx(own.block)}. Nó đọc hàng {row} của{' '}
                    {nameA} và {secondLine}.
                  </>
                }
              >
                <b>
                  {nameC}[{row}][{col}]
                </b>{' '}
                = {r.C[row * l + col]} belongs to Thread {yx(own.thread)} of Block {yx(own.block)}. It reads row {row} of{' '}
                {nameA} and {secondLine}.
              </Translated>
            </p>
          ) : (
            <p style={{ margin: 0 }} data-readout="who">
              <Translated
                vi={
                  <>
                    <b>Khối {yx(own.block)}</b> phủ các hàng {range(inputs.rows)} và các cột {range(inputs.cols)} của {nameC}:{' '}
                    {inputs.active} trên {inputs.threads} luồng của nó qua được phép kiểm tra. Cùng nhau chúng đọc{' '}
                    {inputs.rows.length === 1 ? 'hàng' : 'các hàng'} {range(inputs.rows)} của {nameA} và{' '}
                    {layout === 'book' ? (inputs.cols.length === 1 ? 'cột' : 'các cột') : inputs.cols.length === 1 ? 'hàng' : 'các hàng'}{' '}
                    {range(inputs.cols)} của {nameB}.
                  </>
                }
              >
                <b>Block {yx(own.block)}</b> covers rows {range(inputs.rows)} and columns {range(inputs.cols)} of {nameC}:{' '}
                {inputs.active} of its {inputs.threads} threads pass the guard. Together they read{' '}
                {inputs.rows.length === 1 ? 'row' : 'rows'} {range(inputs.rows)} of {nameA} and{' '}
                {layout === 'book' ? (inputs.cols.length === 1 ? 'column' : 'columns') : inputs.cols.length === 1 ? 'row' : 'rows'}{' '}
                {range(inputs.cols)} of {nameB}.
              </Translated>
            </p>
          )}
          <table className="data" style={{ margin: '0.6rem 0' }} data-table="roles">
            <thead>
              <tr>
                <th><Translated vi="Việc">Job</Translated></th>
                <th><Translated vi="Kích thước đảm nhận">Size that does it</Translated></th>
                <th className="num"><Translated vi="Ở đây">Here</Translated></th>
              </tr>
            </thead>
            <tbody>
              <tr data-role="loop">
                <td><Translated vi="Cận vòng lặp: độ dài một tích vô hướng">Loop bound: length of one dot product</Translated></td>
                <td>{sizeName('k')}</td>
                <td className="num">{k}</td>
              </tr>
              <tr data-role="gridx">
                <td><Translated vi={<>Lưới x: các khối trải ngang qua các cột của {nameC}</>}>Grid x: blocks across the columns of {nameC}</Translated></td>
                <td>{/* i18n-ignore: formula */}
                  ceil({sizeName('l')} / {bw})
                </td>
                <td className="num">{grid.x}</td>
              </tr>
              <tr data-role="gridy">
                <td><Translated vi={<>Lưới y: các khối trải dọc xuống các hàng của {nameC}</>}>Grid y: blocks down the rows of {nameC}</Translated></td>
                <td>{/* i18n-ignore: formula */}
                  ceil({sizeName('j')} / {bw})
                </td>
                <td className="num">{grid.y}</td>
              </tr>
            </tbody>
          </table>
          <table className="data" style={{ margin: '0 0 0.6rem' }}>
            <tbody>
              <tr>
                <td><Translated vi="Số luồng sở hữu một phần tử">Threads that own an element</Translated></td>
                <td className="num" data-total="threads">
                  {big(cnt.threads)}
                </td>
              </tr>
              <tr>
                <td><Translated vi={<>Số luồng được khởi chạy (dư {big(grid.surplus)})</>}>Threads launched (surplus {big(grid.surplus)})</Translated></td>
                <td className="num" data-total="launched">
                  {big(grid.threads)}
                </td>
              </tr>
              <tr>
                <td><Translated vi="Số phép nhân-cộng mỗi luồng">Multiply-adds per thread</Translated></td>
                <td className="num" data-total="madds">
                  {big(cnt.multiplyAddsPerThread)}
                </td>
              </tr>
              <tr>
                <td><Translated vi="Số lần đọc bộ nhớ toàn cục mỗi luồng">Reads from global memory per thread</Translated></td>
                <td className="num" data-total="reads">
                  {big(cnt.readsPerThread)}
                </td>
              </tr>
            </tbody>
          </table>
          <p style={{ margin: 0 }} data-readout="indices">
            <Translated
              vi={
                <>
                  Luồng của {nameC}[{row}][{col}] đọc{' '}
                  <code>
                    {nameA}[row*{sizeName('k')} + i]
                  </code>
                  : các ô {range(aSteps)}, cách nhau 1. Từ toán hạng còn lại nó đọc{' '}
                  <code data-formula="second">
                    {layout === 'book' ? `${nameB}[i*${sizeName('l')} + col]` : `${nameB}[col*${sizeName('k')} + i]`}
                  </code>
                  : các ô {bSteps.length > 4 ? `${bSteps.slice(0, 3).join(', ')}, … ${bSteps[bSteps.length - 1]}` : bSteps.join(', ')},{' '}
                  <b data-stride={t.stride}>cách nhau {t.stride}</b>.
                </>
              }
            >
              The thread for {nameC}[{row}][{col}] reads{' '}
              <code>
                {nameA}[row*{sizeName('k')} + i]
              </code>
              : slots {range(aSteps)}, 1 apart. From the other operand it reads{' '}
              <code data-formula="second">
                {layout === 'book' ? `${nameB}[i*${sizeName('l')} + col]` : `${nameB}[col*${sizeName('k')} + i]`}
              </code>
              : slots {bSteps.length > 4 ? `${bSteps.slice(0, 3).join(', ')}, … ${bSteps[bSteps.length - 1]}` : bSteps.join(', ')},{' '}
              <b data-stride={t.stride}>{t.stride} apart</b>.
            </Translated>
          </p>
        </div>
      </div>

      <Strip id="first" name={<Translated vi={<><b className="t-global">{nameA}</b> khi lưu: {big(j * k)} float, cứ {k} phần tử một hàng</>}><><b className="t-global">{nameA}</b> as stored: {plural(j * k, 'float')}, a row every {k}</></Translated>} cells={j * k} rowLen={k} lit={litA} />
      <Strip
        id="second"
        name={<Translated vi={<><b className="t-global">{nameB}</b> khi lưu: {big(k * l)} float, cứ {bCols} phần tử một hàng</>}><><b className="t-global">{nameB}</b> as stored: {plural(k * l, 'float')}, a row every {bCols}</></Translated>}
        cells={k * l}
        rowLen={bCols}
        lit={litB}
      />
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive 3: thread granularity
const WIDTHS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
const GB = 16; // BLOCK_WIDTH of lab 3.4's 2D launch; its 1D launches use 16 * 16 = 256 threads per block
const DRAWN = 16;
const MODES: { mode: Mode; label: string; short: string }[] = [
  { mode: 'element', get label() { return tr('one thread per element', 'mỗi phần tử một luồng'); }, get short() { return tr('Per element', 'Mỗi phần tử'); } },
  { mode: 'row', get label() { return tr('one thread per row', 'mỗi hàng một luồng'); }, get short() { return tr('Per row', 'Mỗi hàng'); } },
  { mode: 'column', get label() { return tr('one thread per column', 'mỗi cột một luồng'); }, get short() { return tr('Per column', 'Mỗi cột'); } },
];

/** Launch limits a device reported through verify.sh, for the backends that report them. */
function reportedLimits(reports: Reports): { key: string; label: string; limits: Limits }[] {
  return realBackends(reports).flatMap((b) => {
    const d = reports[b].device;
    const n = (key: string) => {
      const v = Number(d[key]);
      return isFinite(v) && v > 0 ? v : null;
    };
    const gridX = n('maxGridSize.x');
    const gridY = n('maxGridSize.y');
    const blockX = n('maxThreadsDim.x');
    const blockY = n('maxThreadsDim.y');
    const threadsPerBlock = n('maxThreadsPerBlock');
    if (gridX == null || gridY == null || blockX == null || blockY == null || threadsPerBlock == null) return [];
    return [{ key: b, get label() { return tr(`Your ${String(d.name ?? b)} (${b})`, `${String(d.name ?? b)} của bạn (${b})`); }, limits: { gridX, gridY, blockX, blockY, threadsPerBlock } }];
  });
}

function GranularityBench() {
  const { reports } = useReports();
  const [mode, setMode] = useState<Mode>('element');
  const [wi, setWi] = useState(4);
  const [thread, setThread] = useState(0);
  const [limKey, setLimKey] = useState('doc');
  const width = WIDTHS[wi];
  const devices = useMemo(() => [{ key: 'doc', get label() { return tr("CUDA's documented limits", 'Giới hạn được tài liệu hóa của CUDA'); }, limits: CUDA_DOCUMENTED }, ...reportedLimits(reports)], [reports]);
  const dev = devices.find((d) => d.key === limKey) ?? devices[0];
  const hasMetal = !!reports.metal;

  const all = MODES.map((m) => ({ ...m, g: granularity(m.mode, width, GB), most: maxWidth(m.mode, dev.limits, GB) }));
  const g = all.find((a) => a.mode === mode)!.g;
  const tSel = Math.min(thread, g.working - 1);
  const mine = new Set(cellsOf(mode, width, tSel));
  const shown = Math.min(width, DRAWN);
  const size = shown <= 8 ? 30 : 20;
  const int32 = maxWidthInt32();
  const threadOf = (r: number, c: number) => (mode === 'element' ? r * width + c : mode === 'row' ? r : c);
  const what =
    mode === 'element'
      ? `P[${Math.floor(tSel / width)}][${tSel % width}]`
      : mode === 'row'
        ? tr(`row ${tSel} of P, left to right`, `hàng ${tSel} của P, từ trái sang phải`)
        : tr(`column ${tSel} of P, top to bottom`, `cột ${tSel} của P, từ trên xuống dưới`);

  return (
    <Bench
      title={tr('How much does one thread do?', 'Một luồng làm bao nhiêu việc?')}
      task={tr('Switch between the three kernels, then drag Width up. Which numbers grow, and how fast?', 'Chuyển qua lại giữa ba kernel, rồi kéo Width lên. Những con số nào tăng, và tăng nhanh đến đâu?')}
      foot={tr('The total arithmetic is the same in all three columns. This model does not say which finishes first; lab 3.4 measures that.', 'Tổng lượng phép tính như nhau ở cả ba cột. Mô hình này không nói cái nào xong trước; lab 3.4 đo điều đó.')}
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Kernel">Kernel</Translated>
          <span className="seg" role="group" aria-label={tr('Kernel', 'Kernel')}>
            {MODES.map((m) => (
              <button key={m.mode} aria-pressed={mode === m.mode} onClick={() => { setMode(m.mode); setThread(0); }}>
                {m.label}
              </button>
            ))}
          </span>
        </span>
        <label className="field">
          Width = {big(width)}
          <input type="range" min={0} max={WIDTHS.length - 1} value={wi} aria-label={tr('Width', 'Width (chiều rộng)')} onChange={(e) => { setWi(+e.target.value); setThread(0); }} />
        </label>
        <label className="field">
          <Translated vi="Giới hạn khởi chạy">Launch limits</Translated>
          <select value={dev.key} aria-label={tr('Launch limits', 'Giới hạn khởi chạy')} onChange={(e) => setLimKey(e.target.value)}>
            {devices.map((d) => (
              <option key={d.key} value={d.key}>
                {d.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.2rem', alignItems: 'flex-start' }}>
        <div style={{ overflowX: 'auto', maxWidth: '100%', paddingBottom: 2 }}>
          <Matrix name="P" note={width > DRAWN ? tr(`top-left ${DRAWN} × ${DRAWN} of ${big(width)} × ${big(width)}`, `góc trên bên trái ${DRAWN} × ${DRAWN} của ${big(width)} × ${big(width)}`) : tr(`${width} × ${width} (click an element)`, `${width} × ${width} (bấm vào một phần tử)`)} cols={shown}>
            {Array.from({ length: shown * shown }, (_, i) => {
              const r = Math.floor(i / shown);
              const c = i % shown;
              const on = mine.has(r * width + c);
              return (
                <button
                  key={i}
                  aria-label={`P[${r}][${c}]`}
                  aria-pressed={on}
                  onClick={() => setThread(threadOf(r, c))}
                  style={{ ...box(size, on ? 'hot' : null, false, false), cursor: 'pointer' }}
                />
              );
            })}
          </Matrix>
          <p className="muted" style={{ margin: '0.5rem 0 0', maxWidth: Math.max(shown * (size + 2), 260) }} data-readout="thread">
            <Translated
              vi={
                <>
                  Luồng {big(tSel)} ghi {what}: {big(g.elementsPerThread)} lần ghi
                  {g.storeStride == null ? '' : `, cách nhau ${big(g.storeStride)} trong bộ nhớ`}.
                </>
              }
            >
              Thread {big(tSel)} writes {what}: {plural(g.elementsPerThread, 'store')}
              {g.storeStride == null ? '' : `, ${big(g.storeStride)} apart in memory`}.
            </Translated>
          </p>
        </div>

        <div style={{ flex: '1 1 26rem', minWidth: 0, overflowX: 'auto' }}>
          <table className="data" style={{ margin: 0, background: 'var(--page)' }} data-table="granularity">
            <thead>
              <tr>
                <th style={{ paddingLeft: '0.6rem' }}>Width = {big(width)}</th>
                {all.map((a) => (
                  <th key={a.mode} className="num" style={a.mode === mode ? { boxShadow: 'inset 0 -2px 0 var(--ink)' } : undefined}>
                    {a.short}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(
                [
                  [tr('Grid', 'Lưới'), 'grid', (a) => (a.g.gridDims === 2 ? tr(`2D: ${big(a.g.grid.x)} × ${big(a.g.grid.y)} ${a.g.grid.x * a.g.grid.y === 1 ? 'block' : 'blocks'} of ${a.g.block.x} × ${a.g.block.y}`, `2D: ${big(a.g.grid.x)} × ${big(a.g.grid.y)} khối ${a.g.block.x} × ${a.g.block.y}`) : tr(`1D: ${plural(a.g.grid.x, 'block')} of ${a.g.block.x}`, `1D: ${big(a.g.grid.x)} khối ${a.g.block.x}`))],
                  [tr('Threads launched', 'Số luồng được khởi chạy'), 'launched', (a) => big(a.g.threads)],
                  [tr('Threads with work', 'Số luồng có việc'), 'working', (a) => big(a.g.working)],
                  [tr('Elements of P per thread', 'Số phần tử của P mỗi luồng'), 'elements', (a) => big(a.g.elementsPerThread)],
                  [tr('Multiply-adds per thread', 'Số phép nhân-cộng mỗi luồng'), 'madds', (a) => big(a.g.multiplyAddsPerThread)],
                  [tr('Stride of one thread\'s stores', 'Bước nhảy giữa các lần ghi của một luồng'), 'stride', (a) => (a.g.storeStride == null ? tr('one store', 'một lần ghi') : big(a.g.storeStride))],
                  [tr('Multiply-adds in total', 'Tổng số phép nhân-cộng'), 'total', (a) => big(a.g.totalMultiplyAdds)],
                  [tr('Largest Width one grid can cover', 'Width lớn nhất một lưới có thể phủ'), 'most', (a) => (a.most ? big(a.most) : tr('block shape not allowed', 'hình dạng khối không được phép'))],
                  [tr('Largest Width with int indices', 'Width lớn nhất với chỉ số int'), 'int32', () => big(int32)],
                ] as [string, string, (a: (typeof all)[number]) => string][]
              ).map(([name, key, cell]) => (
                <tr key={key} data-row={key}>
                  <td style={{ paddingLeft: '0.6rem' }}>{name}</td>
                  {all.map((a) => (
                    <td key={a.mode} className="num" data-mode={a.mode} style={a.mode === mode ? { fontWeight: 620 } : undefined}>
                      {cell(a)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.6rem 0 0' }} data-readout="limits">
            {dev.key === 'doc'
              ? tr(`Limits used: a grid of at most ${big(dev.limits.gridX)} blocks in x and ${big(dev.limits.gridY)} in y, a block of at most ${big(dev.limits.threadsPerBlock)} threads. These are the values NVIDIA documents and the book prints (p. 47); a device reports its own.`, `Giới hạn được dùng: một lưới tối đa ${big(dev.limits.gridX)} khối theo x và ${big(dev.limits.gridY)} theo y, một khối tối đa ${big(dev.limits.threadsPerBlock)} luồng. Đây là các giá trị NVIDIA ghi trong tài liệu và sách in ra (p. 47); mỗi thiết bị tự báo giá trị của nó.`)
              : tr(`Limits used: what ${dev.label.replace(/^Your /, 'your ')} reported: a grid of at most ${big(dev.limits.gridX)} blocks in x and ${big(dev.limits.gridY)} in y, a block of at most ${big(dev.limits.threadsPerBlock)} threads.`, `Giới hạn được dùng: những gì ${dev.label} đã báo: một lưới tối đa ${big(dev.limits.gridX)} khối theo x và ${big(dev.limits.gridY)} theo y, một khối tối đa ${big(dev.limits.threadsPerBlock)} luồng.`)}
            {hasMetal ? tr(' Your Metal device is not offered here: we found no documented maximum grid size for Metal.', ' Thiết bị Metal của bạn không có ở đây: chúng tôi không tìm thấy kích thước lưới tối đa nào được tài liệu hóa cho Metal.') : ''}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the committed prediction for lab 3.4
const RANK_ID = 'ch03/3.4/p-granularity-rank';
const RANK_OPTIONS = [
  () => tr('One thread per element, and both others take more than 10 times as long.', 'Mỗi phần tử một luồng, và cả hai cách còn lại mất hơn 10 lần thời gian.'),
  () => tr('One thread per element, but at least one of the others is within 10 times of it.', 'Mỗi phần tử một luồng, nhưng ít nhất một trong hai cách còn lại chậm không quá 10 lần.'),
  () => tr('One thread per row.', 'Mỗi hàng một luồng.'),
  () => tr('One thread per column.', 'Mỗi cột một luồng.'),
];
const MODE_KEYS = ['element', 'row', 'col'] as const;
const MODE_WORDS: Record<(typeof MODE_KEYS)[number], string> = {
  get element() { return tr('per element', 'mỗi phần tử'); },
  get row() { return tr('per row', 'mỗi hàng'); },
  get col() { return tr('per column', 'mỗi cột'); },
};

interface Lab4Result {
  backend: string;
  device: string;
  whose: string;
  width: number | undefined;
  seconds: Record<(typeof MODE_KEYS)[number], number>;
}

/** Lab 3.4's benchmark rows, per real backend: the reader's kernels where they pass, else the reference solution's. */
function lab4Results(reports: Reports): Lab4Result[] {
  return realBackends(reports).flatMap((b) => {
    const run = reports[b];
    const mine = labStatus(run, LAB4, 'work') === 'pass' ? benchRows(run, LAB4, 'work') : [];
    const rows: BenchRow[] = mine.length ? mine : benchRows(run, LAB4, 'solution');
    const s = (m: string) => num(rows.find((r) => r.name === 'granularity' && r.mode === m)?.seconds);
    const [e, r, c] = [s('element'), s('row'), s('col')];
    if (!e || !r || !c) return [];
    return [{
      backend: b,
      device: String(run.device.name ?? b),
      whose: mine.length ? tr('your kernels', 'các kernel của bạn') : tr('the reference solution', 'lời giải tham chiếu'),
      width: num(rows.find((x) => x.name === 'granularity')?.width),
      seconds: { element: e, row: r, col: c },
    }];
  });
}
/** Which of the four options a measurement corresponds to. */
function measuredOption(s: Lab4Result['seconds']): number {
  const fastest = MODE_KEYS.reduce((a, b) => (s[b] < s[a] ? b : a));
  if (fastest === 'row') return 2;
  if (fastest === 'col') return 3;
  return Math.min(s.row, s.col) / s.element > 10 ? 0 : 1;
}

function RankPrediction() {
  const { progress, answer } = useProgress();
  const { reports, loading, reload } = useReports();
  const prior = progress.answers[RANK_ID];
  const committed = typeof prior?.value === 'number' ? prior.value : null;
  const [pick, setPick] = useState<number | null>(null);
  const results = lab4Results(reports);
  return (
    <div className="q" data-predict={RANK_ID}>
      <div className="kind"><Translated vi="Dự đoán trước">Predict first</Translated></div>
      <div className="prompt">
        <Translated
          vi={
            <>
              Lab 3.4 đo thời gian ba kernel trên cùng các ma trận, trên GPU <em>của bạn</em>. Kernel nào sẽ nhanh nhất, và nhanh hơn
              bao nhiêu? Hãy chốt trước khi chạy.
            </>
          }
        >
          Lab 3.4 times the three kernels on the same matrices, on <em>your</em> GPU. Which will be fastest, and by how
          much? Commit before you run it.
        </Translated>
      </div>
      {committed == null ? (
        <>
          <div className="opts" role="radiogroup">
            {RANK_OPTIONS.map((o, i) => (
              <label key={i}>
                <input type="radio" name={RANK_ID} checked={pick === i} onChange={() => setPick(i)} />
                <span>{o()}</span>
              </label>
            ))}
          </div>
          <button className="primary" disabled={pick == null} onClick={() => pick != null && answer(RANK_ID, pick)}>
            <Translated vi="Chốt dự đoán">Commit prediction</Translated>
          </button>
        </>
      ) : (
        <div className="verdict info">
          <p>
            <b>
              <Translated vi={<>Bạn đã dự đoán: {RANK_OPTIONS[committed]?.() ?? 'một lựa chọn mà trang này không còn liệt kê'}</>}>
                You predicted: {RANK_OPTIONS[committed]?.() ?? 'an option this page no longer lists'}
              </Translated>
            </b>
          </p>
          {results.length ? (
            results.map((m) => {
              const fastest = MODE_KEYS.reduce((a, b) => (m.seconds[b] < m.seconds[a] ? b : a));
              return (
                <p key={m.backend} data-measured={m.backend} data-fastest={fastest}>
                  <Translated
                    vi={
                      <>
                        Đo trên {m.device} ({m.backend}) với {m.whose}
                        {m.width ? `, Width ${big(m.width)}` : ''}:{' '}
                        {MODE_KEYS.map((k) => `${MODE_WORDS[k]} ${fmt(m.seconds[k] * 1000)} ms`).join(', ')}. Nhanh nhất: {MODE_WORDS[fastest]};{' '}
                        {MODE_KEYS.filter((k) => k !== fastest)
                          .map((k) => `${MODE_WORDS[k]} mất ${fmt(m.seconds[k] / m.seconds[fastest])} lần thời gian`)
                          .join(', ')}
                        . {measuredOption(m.seconds) === committed ? 'Đó là lựa chọn bạn đã chọn.' : 'Đó không phải lựa chọn bạn đã chọn.'}
                      </>
                    }
                  >
                    Measured on {m.device} ({m.backend}) with {m.whose}
                    {m.width ? `, Width ${big(m.width)}` : ''}:{' '}
                    {MODE_KEYS.map((k) => `${MODE_WORDS[k]} ${fmt(m.seconds[k] * 1000)} ms`).join(', ')}. Fastest: {MODE_WORDS[fastest]};{' '}
                    {MODE_KEYS.filter((k) => k !== fastest)
                      .map((k) => `${MODE_WORDS[k]} took ${fmt(m.seconds[k] / m.seconds[fastest])} times as long`)
                      .join(', ')}
                    . {measuredOption(m.seconds) === committed ? 'That is the option you chose.' : 'That is not the option you chose.'}
                  </Translated>
                </p>
              );
            })
          ) : loading ? (
            <p data-measured="loading"><Translated vi="Đang đọc báo cáo.">Reading reports.</Translated></p>
          ) : (
            <p data-measured="none">
              <Translated
                vi={
                  <>
                    Chưa có phép đo nào trên máy này. Hãy làm lab 3.4 trên <a href={href('ch03', 'labs')}>trang lab</a>, rồi chạy{' '}
                    <code>./verify.sh</code> không kèm <code>--quick</code> để ghi lại benchmark của nó.{' '}
                  </>
                }
              >
                No measurement on this machine yet. Do lab 3.4 on the <a href={href('ch03', 'labs')}>labs page</a>, then run{' '}
                <code>./verify.sh</code> without <code>--quick</code> to record its benchmark.{' '}
              </Translated>
              <button className="quiet" onClick={reload}>
                <Translated vi="Tải lại kết quả">Reload results</Translated>
              </button>
            </p>
          )}
          <Translated
            vi={
              <p className="muted">
                Đây là các phép đo từ máy của bạn, không phải giá trị kỳ vọng. Các số đếm trong trình khám phá như nhau trên
                mọi thiết bị; thứ hạng thì không nhất thiết. Các chương 4 đến 6 cho bạn những lý do cần tìm.
              </p>
            }
          >
            <p className="muted">
              These are measurements from your machine, not expected values. The counts in the explorer are the same on
              every device; the ranking need not be. Chapters 4 to 6 give you the reasons to look for.
            </p>
          </Translated>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- lab 3.3's reference kernel, on request
function Lab3Listing() {
  const cu = useLabFile(LAB3 + '/solution/kernel.cu');
  const metal = useLabFile(LAB3 + '/solution/kernel.metal');
  const mm = useLabFile(LAB3 + '/solution/launch.mm');
  const { reports } = useReports();
  // The labs page's words for the same runs.
  const cuda = solutionTabStatus(reports, LAB3, 'kernel.cu');
  const onMetal = solutionTabStatus(reports, LAB3, 'kernel.metal');
  if (!cu || !metal || !mm) return null;
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu, tiers: MATMUL_TIERS, status: cuda },
        { label: tr('Metal kernel', 'Kernel Metal'), code: metal, tiers: MATMUL_TIERS, status: onMetal },
        { label: tr('Metal host', 'Host Metal'), code: mm, tiers: {}, status: onMetal },
      ]}
      caption={
        <Translated
          vi={
            <>
              Lời giải tham chiếu (solution) của lab 3.3, nạp từ <code>labs/{LAB3}/solution/</code>. Trạng thái bên cạnh các tab
              được đọc từ các lần chạy lab đã ghi trên máy này. Bản build HIP dùng nguyên tệp CUDA; nó
              chưa được biên dịch hay chạy trên phần cứng AMD.
            </>
          }
        >
          <>
            The reference solution of lab 3.3, loaded from <code>labs/{LAB3}/solution/</code>. The status beside the tabs
            is read from this machine's recorded lab runs. The HIP build takes the CUDA file unchanged; it
            has not yet been compiled or run on AMD hardware.
          </>
        </Translated>
      }
    />
  );
}

function Lab3Reference() {
  const [open, setOpen] = useState(false);
  return (
    <>
      <p className="ui" data-solution="3.3">
        <button className="quiet" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? tr('Hide the rectangular kernel', 'Ẩn kernel chữ nhật') : tr('Show the rectangular kernel (the answer to lab 3.3)', 'Hiện kernel chữ nhật (đáp án của lab 3.3)')}
        </button>{' '}
        <span className="muted"><Translated vi="Có giá trị hơn sau khi bạn đã tự viết bản của mình.">Worth more after you have written your own.</Translated></span>
      </p>
      {open ? <Lab3Listing /> : null}
    </>
  );
}

// ---------------------------------------------------------------- questions
const pre: CSSProperties = { whiteSpace: 'pre-wrap', display: 'block', background: 'none', padding: 0 };
const T10 = trace(makeProblem(TW), 1, 0); // Thread (1,0) of Block (0,0)
const T20 = trace(makeProblem(TW), 2, 0); // Thread (0,0) of Block (1,0)

export const questionsS34: Record<string, Question> = {
  'ch03/3.4/q-swapk': {
    kind: 'choice',
    id: 'ch03/3.4/q-swapk',
    prompt: (
      <Translated
        vi={
          <>
            Ai đó gõ dòng 08 thành <code>Pvalue += M[k*Width+row] * N[col*Width+k];</code>. Kernel biên dịch, chạy và
            điền mọi phần tử của P. Điền bằng gì?
          </>
        }
      >
        <>
          Someone types line 08 as <code>Pvalue += M[k*Width+row] * N[col*Width+k];</code>. The kernel compiles, runs and
          fills every element of P. With what?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={<>M · N, như trước.</>}><>M · N, as before.</></Translated>,
        why: (
          <Translated
            vi={
              <>
                Điều đó cần M được đọc dọc hàng <code>row</code> và N được đọc dọc xuống cột <code>col</code>. Cả hai biểu thức chỉ số
                đều đã thay đổi.
              </>
            }
          >
            <>
              That needs M read along row <code>row</code> and N read down column <code>col</code>. Both index expressions
              have changed.
            </>
          </Translated>
        ),
      },
      {
        label: <>Mᵀ · N.</>,
        why: (
          <Translated
            vi={
              <>
                Đúng một nửa: giờ M được đọc dọc xuống cột <code>row</code>, tức hàng <code>row</code> của Mᵀ. Nhưng N được đọc
                dọc hàng <code>col</code>, không phải dọc xuống cột <code>col</code>.
              </>
            }
          >
            <>
              Half right: M is now read down column <code>row</code>, which is row <code>row</code> of Mᵀ. But N is read
              along row <code>col</code>, not down column <code>col</code>.
            </>
          </Translated>
        ),
      },
      {
        label: <>N · M.</>,
        why: (
          <Translated vi="Gần đúng. Tổng là phần tử (col, row) của N · M, và kernel lưu nó tại (row, col).">
            Close. The sum is element (col, row) of N · M, and the kernel stores it at (row, col).
          </Translated>
        ),
      },
      { label: <>(N · M)ᵀ.</>, correct: true },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
            Đảo ngược việc làm phẳng. <code>M[k*Width+row]</code> là M[k][row] và <code>N[col*Width+k]</code> là N[col][k], nên
            luồng tính Σ N[col][k] · M[k][row] theo k. Đó là phần tử (col, row) của N · M, lưu tại (row, col):
            chuyển vị của N · M, cũng chính là Mᵀ · Nᵀ.
            </p>
            <p>
            Không có gì sập và không có gì ra ngoài biên, vì mọi chỉ số vẫn rơi vào bên trong mảng của nó. Một lỗi đổi vai trò chỉ số
            như thế này chỉ lộ ra khi so với tham chiếu, và chỉ khi các ma trận thử không đối xứng. Lab 3.5
            là một cuộc săn lỗi chỉ số.
            </p>
          </>
        }
      >
        <>
          <p>
            Undo the flattening. <code>M[k*Width+row]</code> is M[k][row] and <code>N[col*Width+k]</code> is N[col][k], so
            the thread computes Σ N[col][k] · M[k][row] over k. That is element (col, row) of N · M, stored at (row, col):
            the transpose of N · M, which is also Mᵀ · Nᵀ.
          </p>
          <p>
            Nothing crashes and nothing is out of bounds, because every index still lands inside its array. An index-role
            bug like this one only shows against a reference, and only if the test matrices are not symmetric. Lab 3.5
            is a hunt for index defects.
          </p>
        </>
      </Translated>
    ),
    source: 'ch. 3 §3.4, Fig. 3.11 line 08, p. 61; the expansion is ours',
    vi: { source: 'ch. 3 §3.4, Fig. 3.11 dòng 08, p. 61; phần khai triển là của khóa học này' },
  },
  'ch03/3.4/q-ex1c': {
    kind: 'choice',
    id: 'ch03/3.4/q-ex1c',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 1c của sách. So với mỗi phần tử một luồng, một kernel mỗi hàng một luồng thay đổi điều gì với{' '}
            <code>Width = 2048</code>?
          </>
        }
      >
        <>
          Book exercise 1c. Compared with one thread per element, what does a one-thread-per-row kernel change for{' '}
          <code>Width = 2048</code>?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="2,048 luồng có việc thay vì khoảng 4.2 triệu, mỗi luồng 2,048 tích vô hướng: một lưới có thể phủ một ma trận lớn hơn, nhưng số luồng chạy cùng lúc ít hơn rất nhiều.">
            2,048 threads with work instead of about 4.2 million, 2,048 dot products each: one grid can cover a larger matrix, but far fewer threads run at once.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nó làm ít phép tính hơn và ít lưu lượng bộ nhớ hơn về tổng thể, vì mỗi luồng nạp hàng M của nó một lần và dùng lại cho cả 2,048 cột.">
            It does less arithmetic and less memory traffic in total, because each thread loads its row of M once and reuses it for all 2,048 columns.
          </Translated>
        ),
        why: (
          <Translated vi="Tổng vẫn là Width³ phép nhân-cộng theo cách nào cũng vậy, và kernel vẫn đọc lại M[row*Width+k] từ bộ nhớ toàn cục cho mỗi cột.">
            The total is Width³ multiply-adds either way, and the kernel still reads M[row*Width+k] from global memory again for every column.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó vẫn cần một lưới 2D, vì P là hai chiều và mỗi luồng cần cả chỉ số hàng lẫn chỉ số cột.">
            It still needs a 2D grid, because P is two-dimensional and every thread needs both a row and a column index.
          </Translated>
        ),
        why: (
          <Translated vi="Tọa độ duy nhất của luồng là hàng của nó; cột đã trở thành một vòng lặp. Một lưới 1D là đủ phủ.">
            The thread's only coordinate is its row; the column has become a loop. A 1D grid covers it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó cần một rào chắn sau mỗi hàng, vì giờ một luồng ghi nhiều phần tử của P và các lần ghi đó phải có thứ tự.">
            It needs a barrier after each row, because one thread now writes several elements of P and those stores must be ordered.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi phần tử của P vẫn có đúng một luồng ghi và các đầu vào chỉ được đọc. Không có hai luồng nào chia sẻ thứ chúng ghi.">
            Every element of P still has exactly one writer and the inputs are only read. No two threads share anything they write.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bảng của trình khám phá có các số đếm: {big(granularity('row', 2048, 16).working)} luồng làm việc, mỗi luồng làm{' '}
            {big(granularity('row', 2048, 16).multiplyAddsPerThread)} phép nhân-cộng, so với{' '}
            {big(granularity('element', 2048, 16).working)} luồng mỗi luồng làm {big(granularity('element', 2048, 16).multiplyAddsPerThread)}.
            Kernel mỗi cột một luồng có cùng số đếm và khác ở một điểm: các lần ghi của nó cách nhau Width thay vì
            liền kề. Cái nào trong ba cái nhanh nhất là câu hỏi về bộ nhớ và lập lịch mà chương 3 không trả lời được.
            Hãy đo nó trong lab 3.4, rồi quay lại sau chương 6.
          </p>
        }
      >
        <p>
          The explorer's table has the counts: {big(granularity('row', 2048, 16).working)} working threads doing{' '}
          {big(granularity('row', 2048, 16).multiplyAddsPerThread)} multiply-adds each, against{' '}
          {big(granularity('element', 2048, 16).working)} doing {big(granularity('element', 2048, 16).multiplyAddsPerThread)}.
          The per-column kernel has the same counts and differs in one thing: its stores are Width apart instead of
          adjacent. Which of the three is fastest is a question about memory and scheduling that chapter 3 cannot answer.
          Measure it in lab 3.4, then come back to it after chapter 6.
        </p>
      </Translated>
    ),
    source: 'ch. 3 exercise 1c, p. 64; the book prints no answer, this one is ours',
    vi: { source: 'ch. 3 bài tập 1c, p. 64; sách không in đáp án, đáp án này là của khóa học này' },
  },
  'ch03/3.4/q-ex2': {
    kind: 'choice',
    id: 'ch03/3.4/q-ex2',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 2 của sách. Một phép nhân ma trận-vector <code>a = B c</code>: B là N × N, thứ tự hàng, độ chính xác đơn; mỗi
            phần tử của <code>a</code> một luồng. Thân kernel nào đúng?
          </>
        }
      >
        <>
          Book exercise 2. A matrix-vector product <code>a = B c</code>: B is N × N, row-major, single precision; one
          thread per element of <code>a</code>. Which kernel body is right?
        </>
      </Translated>
    ),
    options: [
      {
        label: <code style={pre}>{MATVEC_1}</code>,
        why: (
          <Translated vi="Không có phép kiểm tra. Lưới gồm các khối trọn vẹn, nên trừ khi N là bội của kích thước khối, sẽ có luồng có i ≥ N và đọc, ghi quá cuối các mảng.">
            No guard. The grid is whole blocks, so unless N is a multiple of the block size some threads have i ≥ N and read and write past the ends of the arrays.
          </Translated>
        ),
      },
      {
        label: <code style={pre}>{MATVEC_2}</code>,
        correct: true,
      },
      {
        label: <code style={pre}>{MATVEC_3}</code>,
        why: <Translated vi="B[j*N + i] đi dọc xuống cột i. Cái đó tính Bᵀ c.">B[j*N + i] walks down column i. That computes Bᵀ c.</Translated>,
      },
      {
        label: <code style={pre}>{MATVEC_4}</code>,
        why: (
          <Translated vi="Mỗi tích một luồng: giờ N luồng cùng cập nhật một a[i] mà không có gì sắp thứ tự cho chúng, nên các cập nhật bị mất. Nó cũng không phải mỗi phần tử đầu ra một luồng.">
            One thread per product: N threads now update the same a[i] with nothing to order them, so updates are lost. It is also not one thread per output element.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
            Mỗi luồng một phần tử đầu ra nghĩa là một lưới 1D và phép ánh xạ của chương 2; vòng lặp của luồng đi dọc hàng i của
            B, cách nhau 1, và vector c. Theo thuật ngữ của khung bên lề, đây là một hàm level 2. Nó cũng là kernel của
            mục này với một cột đầu ra: lab 3.3 kiểm tra đúng dạng đó (l = 1).
            </p>
            <p className="muted">
            Bốn thân hàm là các kernel trong <code>labs/ch03/snippets/matvec_options.cu</code> (front-end-checked). Thân hàm
            đúng là emulator-checked so với một tham chiếu CPU bởi <code>python3 labs/ch03/snippets/check.py</code>,
            công cụ này cũng xác nhận rằng thân hàm thứ ba tính Bᵀ c.
            </p>
          </>
        }
      >
        <>
          <p>
            One output element per thread means a 1D grid and the mapping of chapter 2; the thread's loop walks row i of
            B, 1 apart, and the vector c. In the sidebar's vocabulary this is a level 2 routine. It is also the kernel of
            this section with one output column: lab 3.3 tests exactly that shape (l = 1).
          </p>
          <p className="muted">
            The four bodies are the kernels of <code>labs/ch03/snippets/matvec_options.cu</code> (front-end-checked). The
            right one is emulator-checked against a CPU reference by <code>python3 labs/ch03/snippets/check.py</code>,
            which also confirms that the third computes Bᵀ c.
          </p>
        </>
      </Translated>
    ),
    source: 'ch. 3 exercise 2, p. 64; the book prints no answer, this one is ours',
    vi: { source: 'ch. 3 bài tập 2, p. 64; sách không in đáp án, đáp án này là của khóa học này' },
  },
  'ch03/3.4/q-rect': {
    kind: 'choice',
    id: 'ch03/3.4/q-rect',
    prompt: (
      <Translated
        vi={
          <>
            <code>C[j × l] = A[j × k] · B[k × l]</code>, mỗi phần tử của C một luồng, row lấy từ các trường y và col từ các
            trường x. Kích thước nào quyết định chiều x của lưới, kích thước nào quyết định chiều y, và kích thước nào là cận vòng lặp?
          </>
        }
      >
        <>
          <code>C[j × l] = A[j × k] · B[k × l]</code>, one thread per element of C, row from the y fields and col from the
          x fields. Which size sets the grid's x extent, which its y extent, and which the loop bound?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="x từ j, y từ l, cận vòng lặp k.">x from j, y from l, loop bound k.</Translated>,
        why: <Translated vi="x cung cấp col, và C có l cột. Với ma trận vuông, lỗi này vô hình.">x supplies col, and C has l columns. For a square matrix this mistake is invisible.</Translated>,
      },
      { label: <Translated vi="x từ l, y từ j, cận vòng lặp k.">x from l, y from j, loop bound k.</Translated>, correct: true },
      {
        label: <Translated vi="x từ k, y từ j, cận vòng lặp l.">x from k, y from j, loop bound l.</Translated>,
        why: (
          <Translated vi="k hoàn toàn không phải là một kích thước của C. Nó là độ dài của mỗi tích vô hướng, và không chiều nào của lưới phủ nó.">
            k is not a size of C at all. It is the length of each dot product, and no grid dimension covers it.
          </Translated>
        ),
      },
      {
        label: <Translated vi="x từ l, y từ k, cận vòng lặp j.">x from l, y from k, loop bound j.</Translated>,
        why: <Translated vi="y cung cấp row, và C có j hàng.">y supplies row, and C has j rows.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lưới phủ đầu ra, và đầu ra có j hàng × l cột: y cung cấp hàng, nên y phủ j; x cung cấp
            cột, nên x phủ l. Kích thước mà lưới không bao giờ thấy, k, là kích thước hai đầu vào chung nhau, và nó trở thành
            cận vòng lặp. Trong một lớp tuyến tính với batch đứng đầu, điều đó đọc là: y trên batch, x trên các đặc trưng đầu ra, một
            vòng lặp trên các đặc trưng đầu vào.
          </p>
        }
      >
        <p>
          The grid covers the output, and the output is j rows by l columns: y supplies the row, so y covers j; x supplies
          the column, so x covers l. The size the grid never sees, k, is the one both inputs share, and it becomes the
          loop bound. In a linear layer with the batch first that reads: y over the batch, x over the output features, a
          loop over the input features.
        </p>
      </Translated>
    ),
    source: 'ours, from the definition and the mapping on p. 60; the kernel is lab 3.3',
    vi: { source: 'của khóa học này, từ định nghĩa và phép ánh xạ ở p. 60; kernel này là lab 3.3' },
  },
  'ch03/3.4/q-linear': {
    kind: 'choice',
    id: 'ch03/3.4/q-linear',
    prompt: (
      <Translated
        vi={
          <>
            <code>nn.Linear</code> giữ <code>weight</code> với shape (out, in) và tính x Aᵀ. Bạn có kernel của
            sách, P = M N, với phép tính chỉ số giữ nguyên, và bạn truyền batch x làm M. Bạn truyền gì làm N?
          </>
        }
      >
        <>
          <code>nn.Linear</code> holds <code>weight</code> with shape (out, in) and computes x Aᵀ. You have the book's
          kernel, P = M N, with its index arithmetic unchanged, and you pass the batch x as M. What do you pass as N?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Buffer trọng số đúng như PyTorch lưu nó.">The weight buffer exactly as PyTorch stores it.</Translated>,
        why: (
          <Translated vi="Kernel sẽ đọc một buffer (out, in) như thể nó là (in, out). Với in ≠ out nó lấy sai phần tử; với in = out nó lặng lẽ tính x A thay vì x Aᵀ.">
            The kernel would read an (out, in) buffer as if it were (in, out). With in ≠ out it fetches the wrong elements; with in = out it silently computes x A instead of x Aᵀ.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Trọng số được chuyển vị một lần trên host thành một buffer (in, out) theo thứ tự hàng.">
            The weight transposed once on the host into an (in, out) row-major buffer.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Trọng số với các hàng theo thứ tự ngược lại.">The weight with its rows in reverse order.</Translated>,
        why: <Translated vi="Điều đó hoán vị các đầu ra. Nó không đổi hàng thành cột.">That permutes the outputs. It does not exchange rows with columns.</Translated>,
      },
      {
        label: (
          <Translated vi="Không cách nào dùng một kernel được: một kernel chuyển vị phải chạy trước mỗi phép nhân.">
            Nothing works with one kernel: a transpose kernel has to run before every multiplication.
          </Translated>
        ),
        why: (
          <Translated vi="Chuyển vị là một biểu thức chỉ số khác, hoặc một bản sao được tạo một lần khi nạp trọng số. Nó không phải một lượt chạy cho mỗi lần gọi.">
            A transpose is a different index expression, or a copy made once when the weights are loaded. It is not a pass per call.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Kernel của sách đọc toán hạng thứ hai dọc xuống một cột, <code>N[k*Width + col]</code>. PyTorch lưu mỗi hàng một
            đặc trưng đầu ra, nên cùng những con số đó nằm ở <code>W[col*in + k]</code>. Khi chỉ số giữ nguyên,
            dữ liệu phải di chuyển để khớp với kernel. Con đường kia, nằm ngoài câu hỏi này, là giữ buffer đã lưu
            và đổi chỉ số của kernel thành <code>W[col*in + k]</code>; tích vẫn như nhau. Lab 3.3 đi con đường thứ nhất để
            hợp đồng của nó khớp với lab 5.3, và chương trình điều khiển (driver) của nó kiểm tra kết quả với x Aᵀ tính từ một trọng số được lưu
            theo cách của PyTorch.
          </p>
        }
      >
        <p>
          The book's kernel reads its second operand down a column, <code>N[k*Width + col]</code>. PyTorch stores one
          output feature per row, so the same numbers sit at <code>W[col*in + k]</code>. With the index unchanged, the
          data has to move to match the kernel. The other route, outside this question, is to keep the stored buffer
          and change the kernel's index to <code>W[col*in + k]</code>; the product is the same. Lab 3.3 takes the first route so
          that its contract matches lab 5.3, and its driver checks the result against x Aᵀ computed from a weight stored
          PyTorch's way.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch03-facts.md, D2 and D6 (torch.nn.Linear in the PyTorch 2.15 docs; ggml.h); book p. 62 for the kernel\'s index',
    vi: { source: 'notes/today/ch03-facts.md, D2 và D6 (torch.nn.Linear trong tài liệu PyTorch 2.15; ggml.h); sách p. 62 cho chỉ số của kernel' },
  },
  'ch03/3.4/q-limit': {
    kind: 'numeric',
    id: 'ch03/3.4/q-limit',
    prompt: (
      <Translated
        vi={
          <>
            Lấy kernel như bản in, với chỉ số <code>int</code>, và giả sử bộ nhớ không phải trở ngại. <code>Width</code> lớn
            nhất là bao nhiêu để chỉ số phẳng cuối cùng, <code>Width*Width − 1</code>, vẫn vừa trong một <code>int</code> có dấu
            32-bit?
          </>
        }
      >
        <>
          Take the kernel as printed, with <code>int</code> indices, and suppose memory is no obstacle. What is the
          largest <code>Width</code> for which the last flat index, <code>Width*Width − 1</code>, still fits in a signed
          32-bit <code>int</code>?
        </>
      </Translated>
    ),
    answer: maxWidthInt32(),
    explain: (
      <Translated
        vi={
          <>
            <p>
            <code>int</code> lớn nhất là 2³¹ − 1 = 2,147,483,647, nên Width² tối đa là 2³¹. Căn bậc hai của 2³¹
            xấp xỉ 46,340.95: 46,340² = 2,147,395,600 vừa còn 46,341² = 2,147,488,281 thì không.
            </p>
            <p>
            Hãy so với lưới. Với các khối 16 × 16, 65,535 khối theo y được tài liệu hóa phủ được{' '}
            {big(maxWidth('element', CUDA_DOCUMENTED, 16))} hàng, gấp hơn 22 lần. Vậy với kernel này chỉ số cạn
            sớm hơn lưới rất nhiều, và cả hai cách khắc phục của sách (nhiều lưới, hoặc nhiều phần tử mỗi
            luồng) đều không đụng tới nó. Cách sửa cho chỉ số là một kiểu rộng hơn cho offset.
            </p>
          </>
        }
      >
        <>
          <p>
            The largest <code>int</code> is 2³¹ − 1 = 2,147,483,647, so Width² may be at most 2³¹. The square root of 2³¹
            is about 46,340.95: 46,340² = 2,147,395,600 fits and 46,341² = 2,147,488,281 does not.
          </p>
          <p>
            Compare that with the grid. With 16 × 16 blocks, the documented 65,535 blocks in y cover{' '}
            {big(maxWidth('element', CUDA_DOCUMENTED, 16))} rows, more than 22 times as many. So for this kernel the index gives
            out long before the grid does, and neither of the book's two remedies (several grids, or several elements per
            thread) touches it. The fix for the index is a wider type for the offset.
          </p>
        </>
      </Translated>
    ),
    source: 'ch. 3 §3.4, p. 64 for the grid limit and its remedies; the index limit is ours (ch03-facts A11, ch05-facts A5)',
    vi: { source: 'ch. 3 §3.4, p. 64 cho giới hạn lưới và các cách khắc phục; giới hạn chỉ số là của khóa học này (ch03-facts A11, ch05-facts A5)' },
  },
};

// ---------------------------------------------------------------- the section
export function S34() {
  return (
    <>
      <Translated
        vi={
          <p>
            Chuyển ảnh xám cho mỗi luồng một điểm ảnh. Làm mờ cho mỗi luồng một vòng lặp. Phép nhân ma trận giữ vòng lặp và
            biến nó thành toàn bộ công việc: mỗi luồng tính một tích vô hướng. Sách đặt nó ở đây vì phép nhân ma trận-ma trận
            là một hàm cốt lõi của chuẩn BLAS, nằm bên dưới các bộ giải hệ tuyến tính như phân rã LU,
            và là một trong những phép tính chính của học sâu, chủ đề sách quay lại với mạng tích chập ở
            chương 19 và với mô hình ngôn ngữ ở chương 20.
          </p>
        }
      >
        <p>
          Grayscale gave every thread one pixel. Blur gave every thread a loop. Matrix multiplication keeps the loop and
          makes it the whole job: each thread computes one dot product. The book puts it here because matrix-matrix
          multiplication is a core routine of the BLAS standard, sits underneath linear solvers such as LU decomposition,
          and is one of the main computations of deep learning, which it returns to for convolutional networks in
          chapter 19 and for language models in chapter 20.
        </p>
      </Translated>
      <Source section="3.4" pages="59" />
      <Translated
        vi={
          <p>
            Trước hết là một thuật ngữ, vì nó có trong tên hàm của mọi thư viện. BLAS (Basic Linear Algebra
            Subprograms) chia các hàm của nó thành ba level theo lượng việc chúng làm. Level 1 là vector với vector,{' '}
            <code>y = αx + y</code>; phép cộng vector của chương 2 là trường hợp α = 1. Level 2 là ma trận với vector,{' '}
            <code>y = αAx + βy</code>. Level 3 là ma trận với ma trận, <code>C = αAB + βC</code>, và kernel của mục này
            là level 3 với α = 1 và β = 0. Theo những thuật ngữ đó, và đây là cách đọc của chúng tôi chứ không phải của sách, một token đi
            qua một lớp tuyến tính là một phép toán level 2, còn một batch hay cả một prompt là level 3.
          </p>
        }
      >
        <p>
          One piece of vocabulary first, because it is in every library's function names. BLAS (Basic Linear Algebra
          Subprograms) sorts its routines into three levels by how much work they do. Level 1 is vector with vector,{' '}
          <code>y = αx + y</code>; chapter 2's vector addition is that with α = 1. Level 2 is matrix with vector,{' '}
          <code>y = αAx + βy</code>. Level 3 is matrix with matrix, <code>C = αAB + βC</code>, and this section's kernel
          is level 3 with α = 1 and β = 0. In those terms, and this is our reading and not the book's, one token going
          through a linear layer is a level 2 operation, and a batch or a whole prompt is level 3.
        </p>
      </Translated>
      <Source section="3.4" pages="59–60" />

      <h2><Translated vi="Định nghĩa, và một phép ánh xạ bạn đã có">The definition, and a mapping you already have</Translated></h2>
      <Translated
        vi={
          <p>
            Nhân một ma trận i × j M với một ma trận j × k N là bạn được một ma trận i × k P. Mỗi phần tử của P là tích
            vô hướng của một hàng của M với một cột của N: <code>P[row][col]</code> là tổng theo k của{' '}
            <code>M[row][k] · N[k][col]</code>. Với P[1][5] đó là M[1][0]·N[0][5] + M[1][1]·N[1][5] + M[1][2]·N[2][5]
            và cứ thế đến cuối hàng.
          </p>
        }
      >
        <p>
          Multiply an i × j matrix M by a j × k matrix N and you get an i × k matrix P. Each element of P is the dot
          product of one row of M with one column of N: <code>P[row][col]</code> is the sum over k of{' '}
          <code>M[row][k] · N[k][col]</code>. For P[1][5] that is M[1][0]·N[0][5] + M[1][1]·N[1][5] + M[1][2]·N[2][5]
          and so on to the end of the row.
        </p>
      </Translated>
      <Source section="3.4" pages="60" />
      <Translated
        vi={
          <p>
            Không cần gì mới để song song hóa nó. Cho mỗi phần tử của P một luồng riêng, đúng như mỗi điểm ảnh có
            luồng riêng ở <a href={href('ch03', '3.2')}>mục 3.2</a>: <code>row</code> từ các trường y,{' '}
            <code>col</code> từ các trường x. Vì phép ánh xạ là một-một, (row, col) của một luồng làm hai việc. Nó
            cho biết luồng ghi vào đâu, và cho biết luồng đọc gì: hàng <code>row</code> của M và cột{' '}
            <code>col</code> của N.
          </p>
        }
      >
        <p>
          Nothing new is needed to parallelise it. Give every element of P its own thread, exactly as every pixel had its
          own thread in <a href={href('ch03', '3.2')}>section 3.2</a>: <code>row</code> from the y fields,{' '}
          <code>col</code> from the x fields. Because the mapping is one to one, a thread's (row, col) does two jobs. It
          says where the thread writes, and it says what the thread reads: row <code>row</code> of M and column{' '}
          <code>col</code> of N.
        </p>
      </Translated>
      <Source section="3.4" pages="60, 63" />
      <Translated
        vi={
          <p>
            Kernel của sách chỉ xử lý ma trận vuông, nên một tham số duy nhất, <code>Width</code>, đại diện cho mọi
            kích thước có mặt. Hãy nhớ điều đó: phần sau của mục này sẽ cho thấy <code>Width</code> thực ra là ba kích thước
            khác nhau mang chung một tên.
          </p>
        }
      >
        <p>
          The book's kernel handles square matrices only, so a single parameter, <code>Width</code>, stands for every
          size in sight. Keep that in mind: later in this section <code>Width</code> turns out to be three different
          sizes wearing one name.
        </p>
      </Translated>
      <Source section="3.4" pages="61" figs="Fig. 3.10" />
      <Translated
        vi={
          <p>
            Sách cũng mô tả phép ánh xạ này như việc cắt P thành các ô (tile), mỗi khối một ô. Ở đây hãy hiểu "ô" đơn giản
            chỉ là phần đầu ra mà các luồng của một khối phủ.
          </p>
        }
      >
        <p>
          The book also describes this mapping as cutting P into tiles, one tile per block. Read "tile" here as nothing
          more than the patch of the output that one block's threads cover.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, A10)" /* i18n-ignore: date and citation */>
        <Translated
          vi={
            <p>
              Giờ "tile" có ba nghĩa mà bạn sẽ gặp. Nghĩa này: phần đầu ra của một khối. Nghĩa của chương 5: một mảnh của
              <em>đầu vào</em> được đưa vào bộ nhớ chia sẻ (shared memory). Và các tile kernel của NVIDIA (<code>__tile_global__</code>), một
              mô hình lập trình thứ hai trong đó trình biên dịch chọn kích thước khối và không có <code>threadIdx</code>.
              Mục này chỉ dùng nghĩa thứ nhất.
            </p>
          }
        >
          <p>
            "Tile" now has three meanings you will meet. This one: a block's patch of the output. Chapter 5's: a piece of
            the <em>inputs</em> staged in shared memory. And NVIDIA's tile kernels (<code>__tile_global__</code>), a
            second programming model in which the compiler picks the block size and there is no <code>threadIdx</code>.
            This section means only the first.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Kernel">The kernel</Translated></h2>
      <Translated
        vi={
          <p>
            Đây là nó, đủ mười hai dòng. Trước hết hãy đọc theo màu: tên màu hổ phách nằm trong bộ nhớ toàn cục, tên màu tím là
            riêng của một luồng. Màu hổ phách duy nhất trong vòng lặp là một lần đọc <code>M</code> và một lần đọc <code>N</code>.
          </p>
        }
      >
        <p>
          Here it is, all twelve lines. Read it by colour first: amber names live in global memory, violet names are
          private to one thread. The only amber inside the loop is one read of <code>M</code> and one of <code>N</code>.
        </p>
      </Translated>
      <Source section="3.4" pages="61" figs="Fig. 3.11" />
      <Code
        tabs={[{ label: 'CUDA', code: KERNEL, tiers: MATMUL_TIERS, status: `front-end-checked (${SNIPPET})` }]}
        caption={
          <Translated
            vi={
              <>
                Fig. 3.11 của sách, với số dòng của nó. Mã in đặt tên kernel là <code>MatrixMulKernel</code>; phần
                chữ của sách gọi nó là <code>matrixMulKernel</code>. Cùng mã nguồn này phục vụ HIP (unverified trên phần cứng AMD);
                phiên bản Metal nằm trong lab 3.3.
              </>
            }
          >
            <>
              The book's Fig. 3.11, with its line numbers. The listing names the kernel <code>MatrixMulKernel</code>; the
              book's text calls it <code>matrixMulKernel</code>. The same source serves HIP (unverified on AMD hardware);
              the Metal version is in lab 3.3.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Các dòng 03 và 04 là phép ánh xạ. Dòng 05 là phép kiểm tra cho các luồng dư, cần thiết bất cứ khi nào{' '}
            <code>Width</code> không phải là bội của cạnh khối. Một luồng qua được nó đặt biến cục bộ{' '}
            <code>Pvalue</code> bằng 0, cho <code>k</code> chạy từ 0 đến <code>Width − 1</code>, mỗi bước cộng thêm một tích,
            và thực hiện đúng một lần ghi, ở dòng 10.
          </p>
        }
      >
        <p>
          Lines 03 and 04 are the mapping. Line 05 is the guard for surplus threads, needed whenever{' '}
          <code>Width</code> is not a multiple of the block's side. A thread that passes it sets a local{' '}
          <code>Pvalue</code> to zero, runs <code>k</code> from 0 to <code>Width − 1</code> adding one product per step,
          and makes exactly one store, on line 10.
        </p>
      </Translated>
      <Source section="3.4" pages="60–62" />
      <Translated
        vi={
          <p>
            Dòng 08 là quy tắc thứ tự hàng của mục 3.2 áp dụng hai lần. Hàng <code>row</code> của M bắt đầu tại{' '}
            <code>M[row*Width]</code>, nên phần tử thứ k của nó là <code>M[row*Width + k]</code>. Cột <code>col</code> của N
            bắt đầu tại <code>N[col]</code>, ở hàng 0, và phần tử kế tiếp của cùng cột nằm xa hơn trọn một hàng,
            nên phần tử thứ k của nó là <code>N[k*Width + col]</code>. Khi k tăng, chỉ số vào M dịch đi 1 và chỉ số
            vào N dịch đi <code>Width</code>.
          </p>
        }
      >
        <p>
          Line 08 is the row-major rule of section 3.2 applied twice. Row <code>row</code> of M starts at{' '}
          <code>M[row*Width]</code>, so its k-th element is <code>M[row*Width + k]</code>. Column <code>col</code> of N
          starts at <code>N[col]</code>, in row 0, and the next element of the same column is one whole row further on,
          so its k-th element is <code>N[k*Width + col]</code>. As k advances, the index into M moves by 1 and the index
          into N moves by <code>Width</code>.
        </p>
      </Translated>
      <Source section="3.4" pages="62" />
      <Translated
        vi={
          <p>
            Đó là một phát biểu về việc một luồng làm gì theo thời gian, không hơn. Đừng vội suy ra chi phí từ nó.
            Chương 6 cho thấy điều quan trọng trên GPU là các luồng lân cận chạm vào gì cùng một lúc, và đó là một
            câu hỏi khác với bước nhảy trong vòng lặp của một luồng.
          </p>
        }
      >
        <p>
          That is a statement about what one thread does over time, and nothing more. Do not read a cost into it yet.
          Chapter 6 shows that what matters on a GPU is what neighbouring threads touch at the same moment, and that is a
          different question from the stride of one thread's loop.
        </p>
      </Translated>
      <Ask q={questionsS34['ch03/3.4/q-swapk']} />

      <h2><Translated vi="Tự lần theo bằng tay">Trace it by hand</Translated></h2>
      <Translated
        vi={
          <p>
            Sách tự tay lần theo một luồng, trên một tích 4 × 4 với <code>BLOCK_WIDTH = 2</code>: bốn khối, mỗi khối
            bốn luồng, những kích thước mà sách gọi là nhỏ phi thực tế và được chọn để vừa một hình vẽ. Trong ví dụ đó Luồng (0,0) của
            Khối (0,0) sở hữu P[0][0], và Luồng (0,0) của Khối (1,0) sở hữu P[2][0]. Nhãn là (y, x), thứ tự của sách
            từ mục 3.1.
          </p>
        }
      >
        <p>
          The book works one thread through by hand, on a 4 × 4 product with <code>BLOCK_WIDTH = 2</code>: four blocks of
          four threads, sizes it calls unrealistically small and chosen to fit a picture. In that example Thread (0,0) of
          Block (0,0) owns P[0][0], and Thread (0,0) of Block (1,0) owns P[2][0]. Labels are (y, x), the book's order
          from section 3.1.
        </p>
      </Translated>
      <Source section="3.4" pages="62–63" figs="Figs. 3.12, 3.13" />
      <Translated
        vi={
          <p>
            Trình chạy từng bước mở ra ở luồng của sách; chạy nó đến cuối là bạn đã dựng lại bảng của sách. Các ma trận
            được đặt như trong hình của sách, M bên trái P để các hàng thẳng nhau, N phía trên P để các cột thẳng nhau.
            Bên dưới chúng là cùng ba mảng đó như bộ nhớ lưu chúng.
          </p>
        }
      >
        <p>
          The stepper opens on the book's thread; step it to the end and you have rebuilt the book's table. The matrices
          sit as in the book's figures, M to the left of P so their rows line up, N above P so their columns line up.
          Under them are the same three arrays as memory holds them.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi lần một luồng. Trên thiết bị, cả mười sáu luồng chạy cùng nhau, theo một thứ tự bạn không kiểm soát; hoạt hình
              bên dưới cho thấy điều đó. Không có thời gian, không có bộ nhớ đệm (cache): trình chạy từng bước chính xác về việc mỗi bước đọc ô nào, ngoài ra không gì khác.
            </p>
          }
        >
          <p>
            One thread at a time. On a device all sixteen run together, in an order you do not control; the animation
            below shows that. No time, no caches: the stepper is exact about which slot each step reads and nothing else.
          </p>
        </Translated>
      </Simplification>
      <HandTrace />
      <Predict
        id="ch03/3.4/p-n-index"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy từng bước: lấy Luồng (1,0) của Khối (0,0). Tại <code>k = 2</code>, nó đọc chỉ số phẳng nào của{' '}
                <code>N</code>? Hãy tính cả chỉ số của nó vào <code>M</code> ở cùng bước đó.
              </>
            }
          >
            <>
              Before you step it: take Thread (1,0) of Block (0,0). At <code>k = 2</code>, which flat index of{' '}
              <code>N</code> does it read? Work out its index into <code>M</code> at the same step too.
            </>
          </Translated>
        }
      >
        {(v) => (
          <Translated
            vi={
              <>
                <p>
                  {v === T10.steps[2].nIndex ? 'Đúng. ' : `Đáp án là ${T10.steps[2].nIndex}. `}
                  Luồng có row 0·2 + 1 = 1 và col 0·2 + 0 = 0, nên nó đọc <code>N[k*Width + col]</code> = N[2·4 + 0] =
                  N[{T10.steps[2].nIndex}] và <code>M[row*Width + k]</code> = M[1·4 + 2] = M[{T10.steps[2].mIndex}]. Tọa
                  độ khối và BLOCK_WIDTH chỉ xuất hiện ở các dòng 03 và 04; sau đó luồng không biết gì ngoài
                  hàng và cột của nó.
                </p>
                <p className="muted">Chọn luồng đó trong trình chạy từng bước và chạy tới k = 2 để thấy cả hai ô sáng lên.</p>
              </>
            }
          >
            <>
              <p>
                {v === T10.steps[2].nIndex ? 'Yes. ' : `It is ${T10.steps[2].nIndex}. `}
                The thread has row 0·2 + 1 = 1 and col 0·2 + 0 = 0, so it reads <code>N[k*Width + col]</code> = N[2·4 + 0] =
                N[{T10.steps[2].nIndex}] and <code>M[row*Width + k]</code> = M[1·4 + 2] = M[{T10.steps[2].mIndex}]. The
                block coordinates and BLOCK_WIDTH appear only in lines 03 and 04; after that the thread knows nothing but
                its row and its column.
              </p>
              <p className="muted">Select that thread in the stepper and step to k = 2 to see both slots light up.</p>
            </>
          </Translated>
        )}
      </Predict>
      <Predict
        id="ch03/3.4/p-store"
        prompt={
          <Translated vi={<>Luồng (0,0) của Khối (1,0): nó ghi vào chỉ số phẳng nào của <code>P</code>?</>}>
            <>
              Thread (0,0) of Block (1,0): which flat index of <code>P</code> does it write?
            </>
          </Translated>
        }
      >
        {(v) => (
          <Translated
            vi={
              <p>
                {v === T20.store!.index ? 'Đúng. ' : `Đáp án là ${T20.store!.index}. `}
                Hàng của nó là blockIdx.y·blockDim.y + threadIdx.y = 1·2 + 0 = 2 và col của nó là 0, nên nó ghi vào{' '}
                <code>P[row*Width + col]</code> = P[2·4 + 0] = P[{T20.store!.index}], tức P[2][0] như Fig. 3.12 của sách
                đánh nhãn.
              </p>
            }
          >
            <p>
              {v === T20.store!.index ? 'Yes. ' : `It is ${T20.store!.index}. `}
              Its row is blockIdx.y·blockDim.y + threadIdx.y = 1·2 + 0 = 2 and its col is 0, so it writes{' '}
              <code>P[row*Width + col]</code> = P[2·4 + 0] = P[{T20.store!.index}], which is P[2][0] as the book's Fig. 3.12
              labels it.
            </p>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Sách cũng tính luồng thứ nhất trong hai luồng này, ở p. 63: hàng 1 và cột 0, nên nó tính P[1][0] từ hàng
            1 của M và cột 0 của N.
          </p>
        }
      >
        <p>
          The book works out the first of these threads too, on p. 63: row 1 and column 0, so it computes P[1][0] from row
          1 of M and column 0 of N.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              <b>Một chỗ sơ suất trên trang sách.</b> Với Luồng (1,0) của Khối (0,0), p. 63 in hàng là 0∗0+1 và cột là
              0∗0+0. Với <code>blockDim</code> = 2 các tích là 0·2+1 và 0·2+0. Kết quả, 1 và 0, đều như nhau
              theo cách nào cũng vậy.
            </p>
          }
        >
          <p>
            <b>A slip on the page.</b> For Thread (1,0) of Block (0,0), p. 63 prints the row as 0∗0+1 and the column as
            0∗0+0. With <code>blockDim</code> = 2 the products are 0·2+1 and 0·2+0. The results, 1 and 0, are the same
            either way.
          </p>
        </Translated>
      </Aside>

      <h2><Translated vi="Cả mười sáu cùng lúc">All sixteen at once</Translated></h2>
      <Translated
        vi={
          <p>
            Trình chạy từng bước bỏ qua điều duy nhất khiến đây là một chương trình GPU: mọi luồng chạy vòng lặp đó cùng
            lúc. Hãy xem một lần, rồi quay lại trình chạy từng bước nếu có bước nào trôi qua quá nhanh.
          </p>
        }
      >
        <p>
          The stepper leaves out the one thing that makes this a GPU program: every thread runs that loop at the same
          time. Watch it once, then come back to the stepper if any step went by too quickly.
        </p>
      </Translated>
      <SixteenThreadsScene loopLine={LOOP_LINE} />
      <Translated
        vi={
          <p>
            Hãy nhìn những gì kernel không chứa: không rào chắn (barrier), không khóa, không phép toán nguyên tử (atomic). Nó không cần cái nào. M và N
            chỉ được đọc, và mỗi phần tử của P có đúng một luồng ghi, vì phép ánh xạ là một-một. Các luồng không chia sẻ
            thứ gì chúng ghi thì không thể làm phiền nhau, nên các khối có thể chạy theo bất kỳ thứ tự nào, hoặc cùng lúc, và P ra
            vẫn như nhau. Hãy tận hưởng khi còn có thể: kernel phân ô của chương 5 bắt các luồng của một khối chia sẻ một
            buffer, và trả giá cho điều đó bằng hai rào chắn.
          </p>
        }
      >
        <p>
          Look at what the kernel does not contain: no barrier, no lock, no atomic operation. It needs none. M and N are
          only read, and each element of P has exactly one writer, because the mapping is one to one. Threads that share
          nothing they write cannot disturb each other, so the blocks may run in any order, or all at once, and P comes
          out the same. Enjoy it while it lasts: the tiled kernel of chapter 5 makes the threads of a block share a
          buffer, and pays for it with two barriers.
        </p>
      </Translated>
      <Beyond title={tr('Our inference', 'Suy luận của chúng tôi')}>
        <Translated
          vi={
            <p>
              Sách nêu phép ánh xạ một-một (p. 60) và không bao giờ nhắc đến đồng bộ hóa cho kernel này. Việc không cần
              đồng bộ hóa suy ra từ phép ánh xạ; mô hình đằng sau trang này chạy các khối theo nhiều thứ tự và kiểm tra
              rằng mỗi phần tử của P có một luồng ghi.
            </p>
          }
        >
          <p>
            The book states the one-to-one mapping (p. 60) and never mentions synchronisation for this kernel. That none
            is needed follows from the mapping; the model behind this page runs the blocks in several orders and checks
            that every element of P has one writer.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Lệnh khởi chạy mà sách không in">The launch the book does not print</Translated></h2>
      <Translated
        vi={
          <p>
            Chương này không bao giờ cho thấy mã host cho kernel này, cũng không định nghĩa <code>BLOCK_WIDTH</code>. Fig. 3.12
            ngầm cho biết cả hai: một lưới 2D với ceil(Width / BLOCK_WIDTH) khối mỗi cạnh, mỗi khối là hình vuông BLOCK_WIDTH
            luồng. Điều duy nhất cần làm đúng là chiều nào của lưới phủ chiều nào của dữ liệu.{' '}
            <code>col</code> lấy từ các trường x, nên chiều x của lưới phải phủ các cột của P, và chiều y
            phủ các hàng. Với ma trận vuông bạn không thể làm sai điều đó, và đó chính là cách trường hợp vuông che giấu
            lỗi.
          </p>
        }
      >
        <p>
          The chapter never shows the host code for this kernel, nor a definition of <code>BLOCK_WIDTH</code>. Fig. 3.12
          implies both: a 2D grid of ceil(Width / BLOCK_WIDTH) blocks on each side, each block BLOCK_WIDTH threads
          square. The one thing to get right is which grid dimension covers which dimension of the data.{' '}
          <code>col</code> comes from the x fields, so the grid's x extent has to cover the columns of P, and its y
          extent the rows. With a square matrix you cannot get that wrong, which is exactly how the square case hides the
          mistake.
        </p>
      </Translated>
      <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
        <Translated
          vi={
            <p>
              Lệnh khởi chạy này nhất quán với Fig. 3.12 (p. 62) và làm tròn lên bằng số học số nguyên, dạng mà hướng dẫn của
              NVIDIA dùng (ch03-facts A1). <code>launchNaiveGemm</code> của lab 3.3 là cùng hàm đó với các kích thước
              được tách riêng.
            </p>
          }
        >
          <p>
            This launch is consistent with Fig. 3.12 (p. 62) and rounds up with integer arithmetic, the form NVIDIA's
            guide uses (ch03-facts A1). Lab 3.3's <code>launchNaiveGemm</code> is the same function with the sizes
            separated.
          </p>
        </Translated>
      </Beyond>
      <Code
        tabs={[{ label: tr('CUDA host', 'Host CUDA'), code: LAUNCH, tiers: {}, status: `front-end-checked (${SNIPPET})` }]}
        firstLine={14}
        caption={tr('The rest of the same file. BLOCK_WIDTH 16 gives 256 threads per block.', 'Phần còn lại của cùng tệp. BLOCK_WIDTH 16 cho 256 luồng mỗi khối.')}
      />

      <h2><Translated vi="Ma trận chữ nhật, và lớp tuyến tính">Rectangular matrices, and the linear layer</Translated></h2>
      <Translated
        vi={
          <p>
            <code>Width</code> đã che giấu ba kích thước. Hãy viết định nghĩa với cả ba:{' '}
            <code>C[j × l] = A[j × k] · B[k × l]</code>. Khi đó j là số hàng của A và của C, l là số
            cột của B và của C, và k là kích thước hai đầu vào chung nhau: độ dài của mỗi tích vô hướng. Mỗi{' '}
            <code>Width</code> trong kernel trở thành một trong ba kích thước đó. Phép kiểm tra là <code>row &lt; j && col &lt; l</code>.
            Vòng lặp chạy tới k. Các chỉ số là <code>A[row*k + i]</code>, <code>B[i*l + col]</code> và{' '}
            <code>C[row*l + col]</code>. Lưới cần ceil(l / BLOCK_WIDTH) khối theo x và ceil(j / BLOCK_WIDTH) theo y.
          </p>
        }
      >
        <p>
          <code>Width</code> was hiding three sizes. Write the definition with all of them:{' '}
          <code>C[j × l] = A[j × k] · B[k × l]</code>. Then j is the number of rows of A and of C, l is the number of
          columns of B and of C, and k is the size the two inputs share: the length of every dot product. Each{' '}
          <code>Width</code> in the kernel becomes one of the three. The guard is <code>row &lt; j && col &lt; l</code>.
          The loop runs to k. The indices are <code>A[row*k + i]</code>, <code>B[i*l + col]</code> and{' '}
          <code>C[row*l + col]</code>. The grid needs ceil(l / BLOCK_WIDTH) blocks in x and ceil(j / BLOCK_WIDTH) in y.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách đưa định nghĩa cho một ma trận i × j nhân một ma trận j × k (p. 60) rồi giới hạn kernel của nó ở
              ma trận vuông (p. 61). Kernel chữ nhật là của khóa học này. Đó là lab 3.3, nửa đầu của mốc dự án M1.
            </p>
          }
        >
          <p>
            The book gives the definition for an i × j times a j × k matrix (p. 60) and then restricts its kernel to
            square ones (p. 61). The rectangular kernel is ours. It is lab 3.3, the first half of project milestone M1.
          </p>
        </Translated>
      </Beyond>
      <Feeds />
      <Ask q={questionsS34['ch03/3.4/q-rect']} />

      <h3><Translated vi="Trọng số được lưu theo chiều nào?">Which way round is the weight stored?</Translated></h3>
      <Translated
        vi={
          <p>
            Đây là nơi kernel gặp mã bạn đã dùng. <code>nn.Linear</code> lưu trọng số với shape
            (out_features, in_features) và tính <code>y = x Aᵀ + b</code>. Với batch đứng đầu, x là batch × in, và
            đầu ra (b, o) là tích vô hướng của hàng b của x với hàng o của trọng số: cả hai toán hạng đều được đọc dọc một hàng. P =
            M N của sách thì không: nó đọc toán hạng thứ hai dọc xuống một cột. Đó là cùng một phép nhân với ma trận thứ hai được lưu theo hai
            cách, và một kernel được viết cho một trong hai cách.
          </p>
        }
      >
        <p>
          This is where the kernel meets code you already use. <code>nn.Linear</code> stores its weight with shape
          (out_features, in_features) and computes <code>y = x Aᵀ + b</code>. With the batch first, x is batch × in, and
          output (b, o) is row b of x dotted with row o of the weight: both operands are read along a row. The book's P =
          M N does not: it reads its second operand down a column. It is one product with the second matrix stored two
          ways, and a kernel is written for one of them.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, D6)" /* i18n-ignore: date and citation */>
        <Translated
          vi={
            <p>
              PyTorch 2.15 mô tả <code>nn.Linear</code> là y = xAᵀ + b với trọng số có shape (out_features,
              in_features).
            </p>
          }
        >
          <p>
            PyTorch 2.15 documents <code>nn.Linear</code> as y = xAᵀ + b with weight of shape (out_features,
            in_features).
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Đổi bố cục trọng số trong trình khám phá ở trên và xem dải phía dưới. Theo bố cục của sách, luồng đang chọn
            chạm vào k ô cách nhau l, <code>i*l + col</code>. Theo bố cục của PyTorch nó chạm vào k ô liền kề,{' '}
            <code>col*k + i</code>. Mọi số đếm trong bảng đều giữ nguyên; chỉ các địa chỉ di chuyển. Khác biệt đó
            có tốn gì không thì mô hình này không thể cho bạn biết.
          </p>
        }
      >
        <p>
          Switch the weight layout in the explorer above and watch the lower strip. In the book's layout the selected
          thread touches k slots that are l apart, <code>i*l + col</code>. In PyTorch's layout it touches k neighbours,{' '}
          <code>col*k + i</code>. Every count in the panel stays the same; only the addresses move. Whether that
          difference costs anything is not something this model can tell you.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Lab 3.3 giữ quy ước của sách với batch đứng đầu: X là batch × in, W là in × out, và kernel
            tính X W. W đó là chuyển vị của tensor mà PyTorch lưu. <a href={href('ch05', 'labs')}>Lab 5.3</a>{' '}
            có cùng hợp đồng, nên kernel phân ô bạn viết ở đó sẽ thay thế kernel này mà không đụng tới bên gọi. Chương
            trình điều khiển (driver) của lab buộc bạn tuân theo: một trong các bài kiểm tra của nó bắt đầu từ một trọng số được lưu theo cách của PyTorch, chuyển vị nó trên
            host, và so sánh kết quả của bạn với x Aᵀ.
          </p>
        }
      >
        <p>
          Lab 3.3 keeps the book's convention with the batch first: X is batch × in, W is in × out, and the kernel
          computes X W. That W is the transpose of the tensor PyTorch stores. <a href={href('ch05', 'labs')}>Lab 5.3</a>{' '}
          has the same contract, so the tiled kernel you write there replaces this one without touching the caller. The
          lab's driver holds you to it: one of its tests starts from a weight stored PyTorch's way, transposes it on the
          host, and compares your result with x Aᵀ.
        </p>
      </Translated>
      <Ask q={questionsS34['ch03/3.4/q-linear']} />
      <Lab3Reference />

      <h2><Translated vi="Một luồng nên làm bao nhiêu việc?">How much should one thread do?</Translated></h2>
      <Translated
        vi={
          <p>
            Mỗi phần tử một luồng là một lựa chọn, không phải một định luật. Bài tập đầu tiên của sách yêu cầu hai cách khác: một kernel trong đó
            mỗi luồng tạo ra trọn một hàng của P, và một kernel trong đó mỗi luồng tạo ra trọn một cột. Ở cả hai, lưới
            trở thành 1D, tọa độ duy nhất của luồng là một hàng (hoặc một cột), và chiều mà lưới không còn
            cung cấp biến thành một vòng lặp thứ hai bên trong luồng.
          </p>
        }
      >
        <p>
          One thread per element is a choice, not a law. The book's first exercise asks for two others: a kernel in which
          each thread produces a whole row of P, and one in which each thread produces a whole column. In both the grid
          becomes 1D, the thread's single coordinate is a row (or a column), and the dimension the grid no longer
          supplies turns into a second loop inside the thread.
        </p>
      </Translated>
      <Source section="3.5" pages="64" figs={tr('exercise 1', 'bài tập 1')} />
      <Translated
        vi={
          <p>
            Trình khám phá đặt ba cách cạnh nhau. Nó đếm; nó không đo thời gian. Lần khởi chạy 2D dùng khối 16 × 16 và
            các lần khởi chạy 1D dùng khối 256, cùng những con số như lab 3.4, nên mọi cột có cùng số luồng
            mỗi khối.
          </p>
        }
      >
        <p>
          The explorer puts the three side by side. It counts; it does not time. The 2D launch uses 16 × 16 blocks and
          the 1D launches use blocks of 256, the same numbers as lab 3.4, so every column has the same number of threads
          per block.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Số luồng, số phép toán và địa chỉ. Không có lập lịch, không có hệ thống bộ nhớ, nên không có xếp hạng: mô hình
              không có ý kiến về kernel nào nhanh nhất. Hàng giới hạn dùng các giới hạn khởi chạy được tài liệu hóa hoặc được báo cáo, không phải lượng
              bộ nhớ một thiết bị có.
            </p>
          }
        >
          <p>
            Thread counts, operation counts and addresses. No scheduling, no memory system, so no ranking: the model has
            no opinion on which kernel is fastest. The limits row uses documented or reported launch limits, not how much
            memory a device has.
          </p>
        </Translated>
      </Simplification>
      <GranularityBench />
      <Ask q={questionsS34['ch03/3.4/q-ex1c']} />
      <RankPrediction />
      <Ask q={questionsS34['ch03/3.4/q-ex2']} label={tr('Write it in your head', 'Viết nó trong đầu')} />

      <h2><Translated vi="P có thể lớn đến đâu?">How large can P be?</Translated></h2>
      <Translated
        vi={
          <p>
            Một lưới có số khối tối đa theo mỗi chiều và một khối có số luồng tối đa, nên một
            lần khởi chạy kernel này chỉ có thể phủ một đầu ra tới một kích thước nào đó. Sách nêu hai lối thoát: cắt P thành
            các ma trận con và khởi chạy mỗi mảnh một lưới từ host, hoặc thay đổi kernel để một luồng tính nhiều
            hơn một phần tử.
          </p>
        }
      >
        <p>
          A grid has a maximum number of blocks in each dimension and a block a maximum number of threads, so a single
          launch of this kernel can only cover an output up to some size. The book names two ways out: cut P into
          sub-matrices and launch one grid per piece from the host, or change the kernel so that a thread computes more
          than one element.
        </p>
      </Translated>
      <Source section="3.4" pages="64" />
      <Translated
        vi={
          <p>
            Cách thứ hai là cái bạn vừa khám phá, và là cái mã production làm ("Trong các backend thực tế", bên dưới, có
            một ví dụ). Hàng "Width lớn nhất" của trình khám phá cho thấy mức độ hiệu ứng với kernel này: một
            lần khởi chạy mỗi hàng hoặc mỗi cột một luồng chỉ dùng chiều x của lưới, chiều rộng rãi nhất từ trước đến nay.
          </p>
        }
      >
        <p>
          The second way is what you just explored, and it is what production code does ("In real backends", below, has
          an example). The explorer's "largest Width" row shows the size of the effect for this kernel: a
          per-row or per-column launch uses only the grid's x dimension, which is by far the roomiest.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, C4; ch05-facts.md, A5; ch08-facts.md, B1)" /* i18n-ignore: date and citation */>
        <Translated
          vi={
            <p>
              NVIDIA ghi trong tài liệu một lưới tối đa 2³¹ − 1 khối theo x và 65,535 theo y và z cho mọi compute
              capability được liệt kê. Bảng của AMD cho 2³² − 1 theo x và 65,535 theo y và z cho các thế hệ nó liệt kê; hãy đọc{' '}
              <code>hipDeviceProp_t</code> trên thế hệ mới hơn. Chúng tôi không tìm thấy kích thước lưới tính toán tối đa nào được tài liệu hóa cho Metal.
            </p>
          }
        >
          <p>
            NVIDIA documents a grid of at most 2³¹ − 1 blocks in x and 65,535 in y and z for every listed compute
            capability. AMD's table gives 2³² − 1 in x and 65,535 in y and z for the generations it lists; read{' '}
            <code>hipDeviceProp_t</code> on a newer one. We found no documented maximum compute grid size for Metal.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Một giới hạn khác đến sớm hơn nhiều, và nó nằm trong chính phép tính của kernel. Mọi chỉ số ở các dòng 08 và 10
            là một <code>int</code>, và câu hỏi bên dưới tìm ra nơi nó cạn.
          </p>
        }
      >
        <p>
          A different limit arrives much sooner, and it is in the kernel's own arithmetic. Every index on lines 08 and 10
          is an <code>int</code>, and the question below finds where that runs out.
        </p>
      </Translated>
      <Ask q={questionsS34['ch03/3.4/q-limit']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Đã có thư viện">Libraries exist</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn sẽ không đưa kernel này vào sản phẩm. Mọi hãng đều có phép nhân ma trận đã tinh chỉnh: cuBLAS trên NVIDIA, hipBLAS trên nền
            rocBLAS trên AMD, và trên Apple là <code>MPSMatrixMultiplication</code> từ host hoặc, từ Metal 4,{' '}
            <code>matmul2d</code> gọi từ bên trong một kernel. Kernel ngây thơ có chỗ đứng theo cách khác. Nó là
            GEMM đúng nhỏ nhất, và là đối tượng mà mọi chương sau dùng làm mốc so sánh.
          </p>
        }
      >
        <p>
          You would not ship this kernel. Every vendor has a tuned matrix multiplication: cuBLAS on NVIDIA, hipBLAS over
          rocBLAS on AMD, and on Apple <code>MPSMatrixMultiplication</code> from the host or, since Metal 4,{' '}
          <code>matmul2d</code> called from inside a kernel. The naive kernel earns its place another way. It is the
          smallest correct GEMM, and it is the object every later chapter measures against.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, C8; ch05-facts.md, A9; infra-facts.md, B12)" /* i18n-ignore: date and citation */>
        <Translated
          vi={
            <p>
              <code>MPSMatrixMultiplication</code> tính C = α·op(A)·op(B) + β·C, dạng level 3 ở trên.{' '}
              <code>matmul2d</code> là một trong các Metal Performance Primitives ở phần 7 của đặc tả Metal Shading Language,
              mới có từ Metal 4. hipBLAS là một lớp giao diện gọi rocBLAS hoặc cuBLAS.
            </p>
          }
        >
          <p>
            <code>MPSMatrixMultiplication</code> computes C = α·op(A)·op(B) + β·C, the level 3 form above.{' '}
            <code>matmul2d</code> is one of the Metal Performance Primitives in section 7 of the Metal Shading Language
            specification, new with Metal 4. hipBLAS is a front end that calls rocBLAS or cuBLAS.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Những gì kernel này còn bỏ ngỏ">What this kernel leaves on the table</Translated></h2>
      <Translated
        vi={
          <p>
            Hai số đếm để mang sang chương 5, chưa kèm phân tích. Mỗi luồng thực hiện 2 · Width lần đọc từ
            bộ nhớ toàn cục (và một lần ghi), và mỗi hàng của M được nạp lại bởi mọi luồng dùng chung hàng đó. Chương 5 hỏi điều đó tốn
            bao nhiêu và loại bỏ phần lớn; chương 15 hoàn tất công việc.
          </p>
        }
      >
        <p>
          Two counts to carry into chapter 5, with no analysis attached yet. Every thread makes 2 · Width reads from
          global memory (and one store), and every row of M is fetched again by every thread that shares that row. Chapter 5 asks what that
          costs and removes most of it; chapter 15 finishes the job.
        </p>
      </Translated>
      <Beyond title={tr('Our counts', 'Các số đếm của chúng tôi')}>
        <Translated
          vi={
            <p>
              Sách chỉ nói rằng kernel này được tối ưu thêm ở chương 5 và được bàn sâu ở chương 15
              (p. 64). Hai số đếm là của chúng tôi, đọc ra từ kernel.
            </p>
          }
        >
          <p>
            The book says only that this kernel is optimised further in chapter 5 and treated in depth in chapter 15
            (p. 64). The two counts are ours, read off the kernel.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Giờ hãy viết nó. <a href={href('ch03', 'labs')}>Lab 3.3</a> là kernel này cho mọi kích thước, kèm lệnh khởi chạy, trên
            mọi backend bạn có. Lab 3.4 là các kernel mỗi hàng và mỗi cột một luồng, đo thời gian so với nó.
          </p>
        }
      >
        <p>
          Now write it. <a href={href('ch03', 'labs')}>Lab 3.3</a> is this kernel for any shape, with its launch, on
          every backend you have. Lab 3.4 is the per-row and per-column kernels, timed against it.
        </p>
      </Translated>

      <h2><Translated vi="Trong các backend thực tế">In real backends</Translated></h2>
      <Translated
        vi={
          <p>
            Những lựa chọn của trang này xuất hiện ở đâu trong mã bạn sẽ đọc sau này. Không phần nào trong đó cần cho các lab.
          </p>
        }
      >
        <p>
          Where the choices of this page turn up in code you will read later. None of it is needed for the labs.
        </p>
      </Translated>
      <ul data-list="backends">
        <Translated
          vi={
            <li>
              <b>Phép ánh xạ.</b> Kernel Metal <code>naive_bmm</code> của PyTorch (<code>mps/kernels/Matmul.metal</code>)
              ánh xạ luồng sang phần tử đầu ra như trang này: cột từ x, hàng từ y, một phép kiểm tra trước lệnh ghi, nhóm 16 × 16.
              Dù tên như vậy, tích vô hướng bên trong của nó là phiên bản phân ô của chương 5.
            </li>
          }
        >
          <li>
            <b>The mapping.</b> PyTorch's Metal kernel <code>naive_bmm</code> (<code>mps/kernels/Matmul.metal</code>)
            maps threads to output elements as this page does: column from x, row from y, a guard on the write, 16 × 16
            groups. Despite its name, its inner product is the tiled one of chapter 5.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Toán hạng thứ hai.</b> <code>ggml_mul_mat</code> của llama.cpp đọc cả hai toán hạng dọc một hàng, như{' '}
              <code>nn.Linear</code>; header của nó gọi toán hạng thứ hai là "k columns, m rows (i.e. we transpose it
              internally)". Khi llama.cpp trao các buffer thứ tự hàng của nó cho cuBLAS, thư viện lưu ma trận theo từng cột,
              nó không sao chép gì: nó gọi <code>cublasSgemm(…, CUBLAS_OP_T, CUBLAS_OP_N, …)</code>, và cùng lời gọi đó
              tới hipBLAS trên AMD qua một lần đổi tên.
            </li>
          }
        >
          <li>
            <b>The second operand.</b> llama.cpp's <code>ggml_mul_mat</code> reads both operands along a row, as{' '}
            <code>nn.Linear</code> does; its header calls the second one "k columns, m rows (i.e. we transpose it
            internally)". When llama.cpp hands its row-major buffers to cuBLAS, which stores matrices column by column,
            it copies nothing: it calls <code>cublasSgemm(…, CUBLAS_OP_T, CUBLAS_OP_N, …)</code>, and the same call
            reaches hipBLAS on AMD through a rename.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Giới hạn lưới.</b> <code>pad.cu</code> của llama.cpp, có lệnh khởi chạy được{' '}
              <a href={href('ch03', '3.1')}>mục 3.1</a> trình bày, giới hạn lưới của nó ở
              65,535 khối theo y và theo z và cho mỗi luồng nhảy theo bước qua bất cứ phần nào lưới không phủ được.
            </li>
          }
        >
          <li>
            <b>The grid limit.</b> llama.cpp's <code>pad.cu</code>, whose launch{' '}
            <a href={href('ch03', '3.1')}>section 3.1</a> shows, caps its grid at
            65,535 blocks in y and in z and has each thread stride over whatever the grid could not cover.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Chỉ số <code>int</code>.</b> Trình khởi chạy theo từng phần tử của PyTorch khẳng định{' '}
              <code>N &lt;= std::numeric_limits&lt;int32_t&gt;::max()</code>. llama.cpp tính offset phẳng là{' '}
              <code>const int64_t dst_idx = …</code>. MLX chuyển sang offset 64-bit khi{' '}
              <code>out.data_size() &gt; UINT32_MAX</code>.
            </li>
          }
        >
          <li>
            <b>The <code>int</code> index.</b> PyTorch's elementwise launcher asserts{' '}
            <code>N &lt;= std::numeric_limits&lt;int32_t&gt;::max()</code>. llama.cpp computes flat offsets as{' '}
            <code>const int64_t dst_idx = …</code>. MLX switches to 64-bit offsets when{' '}
            <code>out.data_size() &gt; UINT32_MAX</code>.
          </li>
        </Translated>
      </ul>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, D1, D2, D5, A7, A11; ch05-facts.md, B9)" /* i18n-ignore: date and citation */>
        <Translated
          vi={
            <p>
              Đọc trong mã nguồn của các dự án vào ngày kiểm tra. Một lời gọi <code>torch.matmul</code> cụ thể đi theo đường matmul nào
              trên MPS thì chưa được lần theo.
            </p>
          }
        >
          <p>
            Read in the projects' sources on the check date. Which matmul path a given <code>torch.matmul</code> call
            takes on MPS was not traced.
          </p>
        </Translated>
      </Today>
    </>
  );
}
