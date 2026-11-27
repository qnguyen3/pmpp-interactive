import { useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { useReports, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { widthChoices } from '../../sim/warps';
import {
  FIG17_1, FIG17_6_ORDER, FORMAT_NAME, ORDER_TERMS, PRESETS, ROW0_PRODUCTS,
  alignedStride, build, cooUpdates, idleLanes, interleavings, longestRow, matrix, matrixRuns, nnz, permutations,
  floatSum, rmwDone, rmwInit, rmwStep, RMW_NEXT, runTrace, shuffledOrder, spmvRef, spmvTrace, storage, summarize,
  toCSC, toCSR, toCOO, toELL, toHybrid, toJDS, toggle, xRuns,
  type ArrayName, type EllLoop, type Entry, type Format, type FormatId, type Lane, type Matrix, type RmwState, type Trace,
} from '../../sim/sparse';
import { EllScene } from './EllScene';

// ---------------------------------------------------------------- the code shown

/** The snippets of this page, loaded at run time; snippets-s172.test.ts pins the lines below in them. The page shows coo,
 * csr and ell; csc is named in the prose and checked with the others, not shown. */
export const SNIPPETS = {
  coo: 'ch17/snippets/spmv_coo.cu',
  csr: 'ch17/snippets/spmv_csr.cu',
  ell: 'ch17/snippets/spmv_ell.cu',
  csc: 'ch17/snippets/spmv_csc.cu',
} as const;
export const COO_ATOMIC_LINES = [
  'cuda::atomic_ref<float, cuda::thread_scope_device> y_ref(y[row]);',
  'y_ref.fetch_add(x[col]*value, cuda::memory_order_relaxed);', // i18n-ignore: CUDA source line
] as const;
export const CSR_LOOP_LINE = 'for (unsigned int i = csrMatrix.rowPtrs[row]; i < csrMatrix.rowPtrs[row + 1]; ++i) {';
export const ELL_INDEX_LINE = 'unsigned int i = t*ellMatrix.stride + row;';
export const ELL_ACCUMULATE_LINE = 'y[row] += sum;';
export const ELL_PAD_TEST_LINE = 'if (col >= 0) sum += x[col]*ellMatrix.value[i];';
export const CSC_LOAD_LINE = 'float inValue = x[col];';

const STATUS = 'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware';
const SPMV_TIERS: Record<string, Tier> = { x: 'global', y: 'global', sum: 'reg', inValue: 'reg', row: 'reg', col: 'reg', i: 'reg', t: 'reg' };

/** The file without its leading comment block (the caption carries what it says). */
export function body(src: string): string {
  const lines = src.split('\n');
  let k = 0;
  while (k < lines.length && (lines[k].startsWith('//') || lines[k].trim() === '')) k++;
  return lines.slice(k).join('\n');
}
/** 1-based numbers of the lines that contain any of the needles. */
const linesWith = (text: string, needles: readonly string[]): number[] =>
  text.split('\n').flatMap((l, i) => (needles.some((n) => l.includes(n)) ? [i + 1] : []));

function Snippet({ file, caption, marks }: { file: string; caption: ReactNode; marks: readonly string[] }) {
  const src = useLabFile(file);
  if (!src) return null;
  const code = body(src);
  return (
    <Code
      tabs={[{ label: file.split('/').pop()!, code, tiers: SPMV_TIERS, status: STATUS, highlight: linesWith(code, marks) }]}
      caption={caption}
    />
  );
}

// ---------------------------------------------------------------- explorer state

type CooOrder = 'row' | 'fig17_6' | 'shuffled';
interface ExState {
  m: Matrix;
  /** A preset's id, or 'edited'. */
  preset: string;
  tab: FormatId;
  step: number;
  K: number;
  ellLoop: EllLoop;
  aligned: boolean;
  order: CooOrder;
  seed: number;
  group: 'four' | 'rows';
}
const INITIAL: ExState = {
  m: FIG17_1, preset: 'fig17_1', tab: 'coo', step: 0, K: 2, ellLoop: 'nnzPerRow', aligned: false, order: 'row', seed: 1, group: 'four',
};
const EMPTY8 = {
  id: 'empty8',
  get label() {
    return tr('Empty 8 × 8', 'Trống 8 × 8');
  },
  get source() {
    return tr('start from nothing', 'bắt đầu từ con số không');
  },
  m: matrix(8, 8, []),
};
const TABS: FormatId[] = ['coo', 'csr', 'ell', 'hyb', 'jds', 'csc'];
const kOf = (s: ExState) => Math.max(1, Math.min(s.K, Math.max(1, longestRow(s.m))));
const widthOf = (s: ExState) => (s.group === 'four' ? 4 : Math.max(1, s.m.rows));

function formatOf(s: ExState): Format {
  const order =
    s.tab !== 'coo' ? undefined
      : s.order === 'fig17_6' && s.preset === 'fig17_1' ? FIG17_6_ORDER
        : s.order === 'shuffled' ? shuffledOrder(nnz(s.m), s.seed) : undefined;
  return build(s.tab, s.m, { K: kOf(s), stride: s.aligned ? alignedStride(s.m.rows, 8) : s.m.rows, order });
}

/** The non-zero (or pad) at an index of the array a step reads. */
function entryAt(f: Format, array: ArrayName, i: number): Entry | null {
  if (f.kind === 'hyb') return array === 'ell' ? f.ell.at[i] : f.coo.at[i];
  return f.at[i] ?? null;
}

type Hover = { kind: 'entry'; r: number; c: number } | { kind: 'row'; r: number } | { kind: 'col'; c: number } | null;
interface ArrCell { text: string; entry: Entry | null; pad?: boolean; row?: number; col?: number; end?: boolean }
interface ArrRow { name: string; array: ArrayName | null; cells: ArrCell[]; part?: string; stride?: number }

const entryCells = (at: readonly Entry[], text: (e: Entry) => string): ArrCell[] => at.map((e) => ({ text: text(e), entry: e }));
const ptrCells = (ptrs: readonly number[], kind: 'row' | 'col'): ArrCell[] =>
  ptrs.map((p, k) => ({ text: String(p), entry: null, end: k === ptrs.length - 1, ...(k < ptrs.length - 1 ? { [kind]: k } : {}) }));
function ellRows(e: Extract<Format, { kind: 'ell' }>, part = ''): ArrRow[] {
  const sfx = part ? ` (${part})` : '';
  return [
    { name: 'nnzPerRow' + sfx, array: null, cells: e.nnzPerRow.map((n, r) => ({ text: String(n), entry: null, row: r })) },
    { name: 'colIdx' + sfx, array: 'ell', stride: e.stride, cells: e.at.map((x, i) => ({ text: x ? String(x.c) : String(e.colIdx[i]), entry: x, pad: !x })) },
    { name: 'value' + sfx, array: 'ell', stride: e.stride, cells: e.at.map((x) => ({ text: x ? x.label : '0', entry: x, pad: !x })) },
  ];
}
function arraysOf(f: Format): ArrRow[] {
  switch (f.kind) {
    case 'coo':
      return [
        { name: 'rowIdx', array: 'coo', cells: entryCells(f.at, (e) => String(e.r)) },
        { name: 'colIdx', array: 'coo', cells: entryCells(f.at, (e) => String(e.c)) },
        { name: 'value', array: 'coo', cells: entryCells(f.at, (e) => e.label) }, // i18n-ignore: the array's name in the code
      ];
    case 'csr':
      return [
        { name: 'rowPtrs', array: null, cells: ptrCells(f.rowPtrs, 'row') },
        { name: 'colIdx', array: 'csr', cells: entryCells(f.at, (e) => String(e.c)) },
        { name: 'value', array: 'csr', cells: entryCells(f.at, (e) => e.label) }, // i18n-ignore: the array's name in the code
      ];
    case 'ell':
      return ellRows(f);
    case 'hyb':
      return [
        ...ellRows(f.ell, tr('ELL part', 'phần ELL')),
        { name: tr('rowIdx (COO part)', 'rowIdx (phần COO)'), array: 'coo', cells: entryCells(f.coo.at, (e) => String(e.r)) },
        { name: tr('colIdx (COO part)', 'colIdx (phần COO)'), array: 'coo', cells: entryCells(f.coo.at, (e) => String(e.c)) },
        { name: tr('value (COO part)', 'value (phần COO)'), array: 'coo', cells: entryCells(f.coo.at, (e) => e.label) },
      ];
    case 'jds':
      return [
        { name: 'row', array: null, cells: f.row.map((r) => ({ text: String(r), entry: null, row: r })) }, // i18n-ignore: the array's name in the code
        { name: 'iterPtr', array: null, cells: f.iterPtr.map((p, k) => ({ text: String(p), entry: null, end: k === f.iterPtr.length - 1 })) },
        { name: 'colIdx', array: 'jds', cells: entryCells(f.at, (e) => String(e.c)) },
        { name: 'value', array: 'jds', cells: entryCells(f.at, (e) => e.label) }, // i18n-ignore: the array's name in the code
      ];
    case 'csc':
      return [
        { name: 'colPtrs', array: null, cells: ptrCells(f.colPtrs, 'col') },
        { name: 'rowIdx', array: 'csc', cells: entryCells(f.at, (e) => String(e.r)) },
        { name: 'value', array: 'csc', cells: entryCells(f.at, (e) => e.label) }, // i18n-ignore: the array's name in the code
      ];
  }
}

const touches = (h: Hover, e: Entry | null, row?: number, col?: number): boolean => {
  if (!h) return false;
  if (h.kind === 'entry') return !!e && e.r === h.r && e.c === h.c;
  if (h.kind === 'row') return e ? e.r === h.r : row === h.r;
  return e ? e.c === h.c : col === h.c;
};

const MAPPING_TEXT: Record<FormatId, string> = {
  get coo() {
    return tr('one thread per non-zero: thread i takes the triple at position i', 'một luồng mỗi phần tử khác 0: luồng i lấy bộ ba ở vị trí i');
  },
  get csr() {
    return tr('one thread per row: thread r walks rowPtrs[r] up to rowPtrs[r + 1]', 'một luồng mỗi hàng: luồng r đi từ rowPtrs[r] tới rowPtrs[r + 1]');
  },
  get ell() {
    return tr('one thread per row: thread r reads slot t·stride + r in iteration t', 'một luồng mỗi hàng: luồng r đọc vị trí t·stride + r ở lần lặp t');
  },
  get hyb() {
    return tr('the ELL part, one thread per row, then the COO part, one thread per spilled non-zero', 'phần ELL, một luồng mỗi hàng, rồi phần COO, một luồng mỗi phần tử khác 0 bị tràn');
  },
  get jds() {
    return tr('one thread per sorted row: thread p reads slot iterPtr[t] + p in iteration t', 'một luồng mỗi hàng đã sắp xếp: luồng p đọc vị trí iterPtr[t] + p ở lần lặp t');
  },
  get csc() {
    return tr('one thread per column: thread c loads x[c] once, then walks colPtrs[c] up to colPtrs[c + 1]', 'một luồng mỗi cột: luồng c nạp x[c] một lần, rồi đi từ colPtrs[c] tới colPtrs[c + 1]');
  },
};

const LABEL_CELL: CSSProperties = { border: 'none', background: 'transparent', color: 'var(--ink-3)', minWidth: '1.4rem' };
const NEUTRAL: CSSProperties = { border: '1px solid var(--rule)' }; // i18n-ignore: CSS value
/** The cell under the pointer, and its partner in the other view. */
const HOVER_RING: CSSProperties = { boxShadow: '0 0 0 2px var(--ink)', zIndex: 1 }; // i18n-ignore: CSS value

// ---------------------------------------------------------------- explorer

function MatrixGrid({ m, hover, setHover, hot, onToggle }: {
  m: Matrix; hover: Hover; setHover: (h: Hover) => void; hot: Set<Entry>; onToggle: (r: number, c: number) => void;
}) {
  const byPos = new Map(m.entries.map((e) => [`${e.r},${e.c}`, e]));
  return (
    <div>
      <div className="mx-label t-global">
        <Translated vi={<>A, {m.rows} × {m.cols}, {nnz(m)} phần tử khác 0</>}>A, {m.rows} × {m.cols}, {nnz(m)} non-zeros</Translated>
      </div>
      <div className="mx global" data-matrix="" style={{ gridTemplateColumns: `auto repeat(${m.cols}, auto)` }}>
        <span style={LABEL_CELL} />
        {Array.from({ length: m.cols }, (_, c) => (
          <span key={'c' + c} style={LABEL_CELL} onMouseEnter={() => setHover({ kind: 'col', c })} onMouseLeave={() => setHover(null)}>{c}</span>
        ))}
        {Array.from({ length: m.rows }, (_, r) => [
          <span key={'r' + r} style={LABEL_CELL} onMouseEnter={() => setHover({ kind: 'row', r })} onMouseLeave={() => setHover(null)}>{r}</span>,
          ...Array.from({ length: m.cols }, (_, c) => {
            const e = byPos.get(`${r},${c}`) ?? null;
            const lit = !!e && touches(hover, e, r, c);
            const cls = e && hot.has(e) ? 'hot' : lit ? 'on' : '';
            return (
              <span
                key={`${r},${c}`} data-r={r} data-c={c} role="button" tabIndex={0} className={cls}
                aria-label={e ? tr(`row ${r}, column ${c}: ${e.label}; click to remove`, `hàng ${r}, cột ${c}: ${e.label}; bấm để xóa`) : tr(`row ${r}, column ${c}: zero; click to add a non-zero`, `hàng ${r}, cột ${c}: số 0; bấm để thêm một phần tử khác 0`)}
                style={{ cursor: 'pointer', ...(lit ? HOVER_RING : {}) }}
                onClick={() => onToggle(r, c)}
                onKeyDown={(ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); onToggle(r, c); } }}
                onMouseEnter={() => setHover(e ? { kind: 'entry', r, c } : null)}
                onMouseLeave={() => setHover(null)}
              >
                {e ? e.label : ''}
              </span>
            );
          }),
        ])}
      </div>
      <p className="muted" style={{ margin: '0.4rem 0 0', maxWidth: '18rem' }}>
        <Translated vi="Bấm vào một ô để thêm hoặc xóa một phần tử khác 0. Rê chuột lên số hàng hoặc số cột để làm sáng toàn bộ hàng hoặc cột đó.">
          Click a cell to add or remove a non-zero. Hover a row or column number to light up all of it.
        </Translated>
      </p>
    </div>
  );
}

function Arrays({ rows, hover, setHover, reading }: {
  rows: ArrRow[]; hover: Hover; setHover: (h: Hover) => void; reading: { array: ArrayName; lanes: Map<number, Lane> } | null;
}) {
  return (
    <div>
      {rows.map((a) => (
        <div key={a.name} style={{ marginBottom: '0.55rem' }}>
          <div className="mx-label t-global">{a.name}</div>
          <div className="mx global" data-array={a.name} style={{ display: 'flex', flexWrap: 'wrap' }}>
            {a.cells.map((c, i) => {
              const lane = reading && a.array === reading.array ? reading.lanes.get(i) : undefined;
              const on = touches(hover, c.entry, c.row, c.col);
              const cls = c.pad || c.end ? 'void' : lane ? 'hot' : on ? 'on' : '';
              const style: CSSProperties = on ? { ...HOVER_RING } : {};
              if (a.stride && i > 0 && i % a.stride === 0) style.marginLeft = '0.45rem';
              if (lane && c.pad) style.outline = '2px solid var(--global)'; // i18n-ignore: CSS value
              return (
                <span
                  key={i} data-i={i} className={cls} style={style}
                  onMouseEnter={() => setHover(c.entry ? { kind: 'entry', r: c.entry.r, c: c.entry.c } : c.row != null ? { kind: 'row', r: c.row } : c.col != null ? { kind: 'col', c: c.col } : null)}
                  onMouseLeave={() => setHover(null)}
                >
                  {c.text}
                  <sub style={lane && !c.pad ? { color: 'inherit', opacity: 0.8 } : undefined}>{i}</sub>
                </span>
              );
            })}
            {a.cells.length === 0 ? <span className="void" style={{ minWidth: '5rem' }}><Translated vi="rỗng">empty</Translated></span> : null}
          </div>
        </div>
      ))}
    </div>
  );
}

function threadName(f: Format, step: { array: ArrayName }, lane: Lane): string {
  if (step.array === 'coo') return tr(`thread ${lane.thread}`, `luồng ${lane.thread}`);
  if (f.kind === 'jds') return tr(`thread ${lane.thread} (row ${f.row[lane.thread]})`, `luồng ${lane.thread} (hàng ${f.row[lane.thread]})`);
  if (f.kind === 'csc') return tr(`thread ${lane.thread} (column ${lane.thread})`, `luồng ${lane.thread} (cột ${lane.thread})`);
  return tr(`thread ${lane.thread} (row ${lane.thread})`, `luồng ${lane.thread} (hàng ${lane.thread})`);
}

function Explorer({ st, set, setTab }: { st: ExState; set: (p: Partial<ExState>) => void; setTab: (t: FormatId) => void }) {
  const [hover, setHover] = useState<Hover>(null);
  const f = useMemo(() => formatOf(st), [st]);
  const trace: Trace = useMemo(() => spmvTrace(f, { ellLoop: st.ellLoop }), [f, st.ellLoop]);
  const width = widthOf(st);
  const sum = useMemo(() => summarize(trace, width, st.m.rows), [trace, width, st.m.rows]);
  const store = storage(f, st.m);
  const steps = trace.steps;
  const si = Math.min(st.step, Math.max(0, steps.length - 1));
  const cur = steps[si];
  const launch = cur ? trace.launches[cur.launch] : trace.launches[0];
  const lanesByIndex = new Map((cur?.lanes ?? []).filter((l) => l.index != null).map((l) => [l.index!, l]));
  const hot = new Set((cur?.lanes ?? []).flatMap((l) => (l.index != null ? [entryAt(f, cur!.array, l.index)].filter((e): e is Entry => !!e) : [])));
  const ones = Array(st.m.cols).fill(1);
  const zeros = Array(st.m.rows).fill(0);
  const y = runTrace(trace, ones, zeros);
  const ref = spmvRef(st.m, ones, zeros);
  const runsNow = cur ? matrixRuns(cur, width) : [];
  const xNow = cur ? xRuns(cur, width) : [];
  const idleNow = cur ? idleLanes(cur) : 0;
  const z = nnz(st.m);
  const K = longestRow(st.m);
  const showGroupChoice = st.m.rows !== 4;

  const loadPreset = (id: string) => {
    const p = PRESETS.find((x) => x.id === id) ?? EMPTY8;
    set({ m: p.m, preset: p.id, step: 0, order: st.order === 'fig17_6' && p.id !== 'fig17_1' ? 'row' : st.order });
  };
  const runsText = (rs: number[]) => {
    const n = (v: number) => tr(`${v} run${v === 1 ? '' : 's'}`, `${v} dải`);
    if (rs.length <= 1) return n(rs[0] ?? 0);
    return rs.every((v) => v === rs[0])
      ? tr(`${n(rs[0])} in each of ${rs.length} groups`, `${n(rs[0])} ở mỗi nhóm trong ${rs.length} nhóm`)
      : rs.map((v, g) => tr(`group ${g}: ${v ? n(v) : 'nothing'}`, `nhóm ${g}: ${v ? n(v) : 'không có'}`)).join(', ');
  };

  return (
    <Bench
      title={tr('Format explorer', 'Trình khám phá định dạng')}
      task={tr('Pick a format, step its kernel, and edit the matrix. Hover a cell to find it in the arrays.', 'Chọn một định dạng, chạy từng bước kernel của nó, và sửa ma trận. Rê chuột lên một ô để tìm nó trong các mảng.')}
      foot={
        <Translated
          vi={
            <>
              Mọi mảng và số đếm đều lấy từ <code>src/sim/sparse.ts</code>, mà các bài kiểm thử của nó gắn với các hình của sách. x bằng 1
              ở mọi chỗ và y bắt đầu bằng 0, nên y kết thúc bằng tổng các hàng.
            </>
          }
        >
          <>
            Every array and count comes from <code>src/sim/sparse.ts</code>, which its tests tie to the book's figures. x is 1
            everywhere and y starts at 0, so y ends as the row sums.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Matrices', 'Ma trận')}>
          {[...PRESETS, EMPTY8].map((p) => (
            <button key={p.id} data-preset={p.id} aria-pressed={st.preset === p.id} onClick={() => loadPreset(p.id)} title={tr(p.source, ({ 'p. 402, the running example': 'p. 402, ví dụ xuyên suốt', 'p. 415, two long rows; values ours': 'p. 415, hai hàng dài; giá trị của khóa học này', 'p. 416, letters; values ours': 'p. 416, chữ cái; giá trị của khóa học này' } as Record<string, string>)[p.source] ?? p.source)}>
              {tr(p.label, ({ 'Exercise 1': 'Bài tập 1' } as Record<string, string>)[p.label] ?? p.label)}
            </button>
          ))}
        </span>
        {st.preset === 'edited' ? <span className="muted"><Translated vi="đã sửa">edited</Translated></span> : null}
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'max-content minmax(0, 1fr)' }}>
        <MatrixGrid
          m={st.m} hover={hover} setHover={setHover} hot={hot}
          onToggle={(r, c) => set({ m: toggle(st.m, r, c), preset: 'edited', step: 0, order: st.order === 'fig17_6' ? 'row' : st.order })}
        />
        <div>
          <span className="seg" role="group" aria-label={tr('Formats', 'Định dạng')} style={{ flexWrap: 'wrap', rowGap: 4, marginBottom: '0.7rem' }}>
            {TABS.map((t) => (
              <button key={t} data-tab={t} aria-pressed={st.tab === t} onClick={() => setTab(t)}>{tr(FORMAT_NAME[t], t === 'hyb' ? 'Lai ELL-COO' : FORMAT_NAME[t])}</button>
            ))}
          </span>
          {st.tab === 'hyb' || st.tab === 'jds' ? (
            <p className="muted" style={{ margin: '0 0 0.6rem' }}>
              <Translated vi="Trang 17.5 giải thích định dạng này. Nó có mặt ở đây để mọi định dạng có thể được so sánh trên cùng một ma trận.">
                Page 17.5 explains this format. It is here so that every format can be compared on the same matrix.
              </Translated>
            </p>
          ) : null}
          <Arrays rows={arraysOf(f)} hover={hover} setHover={setHover} reading={cur ? { array: cur.array, lanes: lanesByIndex } : null} />
          <Translated
            vi={
              <p style={{ margin: '0.6rem 0 0' }} data-storage="">
                Đã lưu: <b data-k="total">{store.total}</b> số ({store.values} giá trị, {store.indices} chỉ số, {store.pointers} con trỏ và số đếm),
                {store.pads ? <> trong đó có <b data-k="pads">{store.pads}</b> vị trí đệm trong <code>value</code> và chừng ấy trong các chỉ số cột.</> : <> với <b data-k="pads">không</b> phần đệm nào.</>}{' '}
                Ma trận dày đặc là <span data-k="dense">{store.dense}</span>.
                <br />
                <span className="muted">
                Số đếm của bài tập 2: <code>{store.formula}</code> với m = {st.m.rows}, n = {st.m.cols}, z = {z},{' '}
                {f.kind === 'hyb'
                ? `K = ${f.ell.width} (độ rộng của phần ELL; hàng dài nhất có ${K}), s = ${f.coo.value.length} bị tràn`
                : `K = ${K} (hàng dài nhất)`}
                {f.kind === 'ell' && f.stride !== f.numRows ? `, stride = ${f.stride}` : ''}.
                </span>
              </p>
            }
          >
            <p style={{ margin: '0.6rem 0 0' }} data-storage="">
              Stored: <b data-k="total">{store.total}</b> numbers ({store.values} values, {store.indices} indices, {store.pointers} pointers and counts),
              {store.pads ? <> including <b data-k="pads">{store.pads}</b> pad slots in <code>value</code> and as many in the column indices.</> : <> with <b data-k="pads">no</b> pads.</>}{' '}
              The dense matrix is <span data-k="dense">{store.dense}</span>.
              <br />
              <span className="muted">
                Exercise 2's count: <code>{store.formula}</code> with m = {st.m.rows}, n = {st.m.cols}, z = {z},{' '}
                {f.kind === 'hyb'
                  ? `K = ${f.ell.width} (the ELL part's width; the longest row has ${K}), s = ${f.coo.value.length} spilled`
                  : `K = ${K} (the longest row)`}
                {f.kind === 'ell' && f.stride !== f.numRows ? `, stride = ${f.stride}` : ''}.
              </span>
            </p>
          </Translated>
        </div>
      </div>

      <div className="panel" style={{ marginTop: '1rem' }} data-kernel={st.tab}>
        <p style={{ margin: '0 0 0.6rem' }}>
          <b>SpMV/{FORMAT_NAME[st.tab].replace('Hybrid ELL-COO', 'hybrid')}</b>: {MAPPING_TEXT[st.tab]}.
        </p>
        <div className="row" style={{ marginBottom: '0.7rem' }}>
          <button onClick={() => set({ step: 0 })} disabled={si === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
          <button onClick={() => set({ step: si - 1 })} disabled={si === 0}><Translated vi="Lùi">Back</Translated></button>
          <button className="primary" onClick={() => set({ step: si + 1 })} disabled={si >= steps.length - 1}><Translated vi="Bước">Step</Translated></button>
          <span className="muted" aria-live="polite" data-step={si}>
            {steps.length ? (
              <Translated vi={<>bước {si + 1} trên {steps.length}: {launch.name.replace(' part', ' (phần)')}, lần lặp t = {cur.t}</>}>
                <>step {si + 1} of {steps.length}: {launch.name}, iteration t = {cur.t}</>
              </Translated>
            ) : tr('no non-zeros: nothing to run', 'không có phần tử khác 0: không có gì để chạy')}
          </span>
          {showGroupChoice ? (
            <label className="field">
              <Translated vi="Số luồng mỗi nhóm">Threads per group</Translated>
              <select value={st.group} onChange={(e) => set({ group: e.target.value as ExState['group'] })} aria-label={tr('Threads per group', 'Số luồng mỗi nhóm')}>
                <option value="four">{tr("4, as in the book's figures", '4, như trong các hình của sách')}</option>
                <option value="rows">{st.m.rows}{tr(', the row count', ', bằng số hàng')}</option>
              </select>
            </label>
          ) : null}
        </div>
        {st.tab === 'coo' ? (
          <div className="row" style={{ marginBottom: '0.7rem' }}>
            <span className="seg" role="group" aria-label={tr('COO order', 'Thứ tự COO')}>
              <button aria-pressed={st.order === 'row'} onClick={() => set({ order: 'row', step: 0 })}><Translated vi="thứ tự hàng (Hình 17.3)">row order (Fig. 17.3)</Translated></button>
              <button aria-pressed={st.order === 'fig17_6'} disabled={st.preset !== 'fig17_1'} onClick={() => set({ order: 'fig17_6', step: 0 })}><Translated vi="thứ tự của Hình 17.6">Fig. 17.6's order</Translated></button>
              <button aria-pressed={st.order === 'shuffled'} onClick={() => set({ order: 'shuffled', seed: st.order === 'shuffled' ? st.seed + 1 : st.seed, step: 0 })}>
                {st.order === 'shuffled' ? tr('shuffle again', 'xáo lại') : tr('shuffled', 'đã xáo')}
              </button>
            </span>
          </div>
        ) : null}
        {st.tab === 'ell' || st.tab === 'hyb' ? (
          <div className="row" style={{ marginBottom: '0.7rem' }}>
            <span className="seg" role="group" aria-label={tr('ELL loop bound', 'Cận vòng lặp ELL')}>
              <button aria-pressed={st.ellLoop === 'nnzPerRow'} onClick={() => set({ ellLoop: 'nnzPerRow', step: 0 })}><Translated vi="lặp tới nnzPerRow (sách)">loop to nnzPerRow (the book)</Translated></button>
              <button aria-pressed={st.ellLoop === 'full'} onClick={() => set({ ellLoop: 'full', step: 0 })}><Translated vi={<>toàn bộ độ rộng, kiểm tra col &gt;= 0</>}>full width, test col &gt;= 0</Translated></button>
            </span>
            {st.tab === 'ell' ? (
              <span className="seg" role="group" aria-label={tr('ELL stride', 'Stride của ELL')}>
                <button aria-pressed={!st.aligned} onClick={() => set({ aligned: false, step: 0 })}>stride = numRows</button>
                <button aria-pressed={st.aligned} onClick={() => set({ aligned: true, step: 0 })}><Translated vi="stride là bội của 8">stride a multiple of 8</Translated></button>
              </span>
            ) : (
              <label className="field">
                <Translated vi={<>K, độ rộng của phần ELL: {kOf(st)}</>}>K, the ELL part's width: {kOf(st)}</Translated>
                <input type="range" min={1} max={Math.max(1, K)} value={kOf(st)} aria-label="K"
                  onChange={(e) => set({ K: Number(e.target.value), step: 0 })} />
              </label>
            )}
          </div>
        ) : null}

        {cur ? (
          <table className="threads" aria-label={tr('Threads in this step', 'Các luồng trong bước này')} style={{ marginBottom: '0.7rem' }}>
            <thead>
              <tr>
                <th><Translated vi="luồng">thread</Translated></th>
                <th><Translated vi="đọc">reads</Translated></th>
                <th>x</th>
                <th><Translated vi="y của nó">its y</Translated></th>
                <th><Translated vi="cập nhật y">update of y</Translated></th>
              </tr>
            </thead>
            <tbody>
              {cur.lanes.map((l, k) => (
                <tr key={k} data-thread={l.thread} data-read={l.index ?? '-'} data-pad={l.pad ? 1 : 0}
                  style={k > 0 && k % width === 0 ? { borderTop: '2px solid var(--rule)' } : undefined}>
                  <td>{threadName(f, cur, l)}</td>
                  <td className={l.index != null ? 't-global' : 'muted'}>
                    {l.index == null ? tr('idle', 'rảnh') : tr(
                      `${cur.array === 'coo' && f.kind === 'hyb' ? 'COO part ' : ''}slot ${l.index}${l.pad ? ' (pad: colIdx only)' : ''}`,
                      `${cur.array === 'coo' && f.kind === 'hyb' ? 'phần COO, ' : ''}vị trí ${l.index}${l.pad ? ' (phần đệm: chỉ colIdx)' : ''}`,
                    )}
                  </td>
                  <td>{l.x != null ? `x[${l.x}]` : <span className="muted">–</span>}</td>
                  <td>{l.y != null ? `y[${l.y}]` : <span className="muted">–</span>}</td>
                  <td style={{ whiteSpace: 'normal' }}>
                    {l.atomic ? <span className="chip global"><Translated vi="cộng nguyên tử">atomic add</Translated></span>
                      : l.index != null || (l.y != null && !launch.atomic) ? <span className="muted"><Translated vi="tổng riêng, cộng một lần ở cuối">own sum, added once at the end</Translated></span> : <span className="muted">–</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}

        <div className="row" style={{ gap: '0.4rem 0.6rem' }}>
          <span className="chip" style={NEUTRAL} data-k="idle-now">
            <Translated vi={<>{idleNow} trên {cur?.lanes.length ?? 0} luồng rảnh trong bước này</>}>{idleNow} of {cur?.lanes.length ?? 0} threads idle in this step</Translated>
          </span>
          <span className="chip global" data-k="runs-now">
            <Translated vi={<>lần đọc ma trận trong bước này: {runsText(runsNow)}</>}>matrix reads in this step: {runsText(runsNow)}</Translated>
          </span>
          <span className="chip global" data-k="x-runs-now">
            <Translated vi={<>lần đọc x trong bước này: {runsText(xNow)}</>}>x reads in this step: {runsText(xNow)}</Translated>
          </span>
          <span className="chip global" data-k="atomics">
            <Translated vi={<>{sum.atomicUpdates} lần cập nhật nguyên tử lên y</>}>{sum.atomicUpdates} atomic updates of y</Translated>
          </span>
          <span className="chip" style={NEUTRAL} data-k="shared-y">
            <Translated
              vi={
                <>{sum.sharedY} phần tử y được nhiều hơn một luồng cập nhật</>
              }
            >
              {sum.sharedY} y element{sum.sharedY === 1 ? '' : 's'} updated by more than one thread
            </Translated>
          </span>
        </div>
        <Translated
          vi={
            <p style={{ margin: '0.7rem 0 0' }} data-k="divergence">
              Toàn bộ kernel, nhóm {width} luồng: các nhóm chạy tổng cộng <b data-k="group-iters">{sum.groupIterations}</b> lần lặp. Trong số các lượt lane-lặp,{' '}
              <b data-k="useful">{sum.useful}</b> lượt nhân một phần tử khác 0, <b data-k="waiting">{sum.idle}</b> lượt chờ một hàng dài hơn trong nhóm
              {sum.pad ? <>, và <b data-k="pad-work">{sum.pad}</b> lượt đọc một phần đệm</> : <span data-k="pad-work" data-v="0" />}.
            </p>
          }
        >
          <p style={{ margin: '0.7rem 0 0' }} data-k="divergence">
            Whole kernel, groups of {width}: the groups run <b data-k="group-iters">{sum.groupIterations}</b> iterations in all. Of the lane-iterations,{' '}
            <b data-k="useful">{sum.useful}</b> multiply a non-zero, <b data-k="waiting">{sum.idle}</b> wait for a longer row in the group
            {sum.pad ? <>, and <b data-k="pad-work">{sum.pad}</b> read a pad</> : <span data-k="pad-work" data-v="0" />}.
          </p>
        </Translated>
        <p style={{ margin: '0.4rem 0 0' }} data-k="y">
          <Translated vi={<>y khi kernel kết thúc: [{y.join(', ')}]</>}>y when the kernel ends: [{y.join(', ')}]</Translated>{' '}
          {y.every((v, i) => v === ref[i]) ? (
            <span className="chip ok"><Translated vi="tổng các hàng của A">the row sums of A</Translated></span>
          ) : (
            <span className="chip fail"><Translated vi="không phải A·x">not A·x</Translated></span>
          )}
        </p>
      </div>
    </Bench>
  );
}

function WidthsNote() {
  const { reports } = useReports();
  const choices = useMemo(() => widthChoices(reports as Reports), [reports]);
  return (
    <>
      {tr(
        choices.map((c) => c.label).join('; '),
        choices
          .map((c) => (c.key === 'book' ? 'con số 32 của sách' : c.key === 'w64' ? '64, như HIP ghi trong tài liệu cho gfx9' : c.label.replace(/^yours: /, 'của bạn: ')))
          .join('; '),
      )}
    </>
  );
}

// ---------------------------------------------------------------- two threads, one y[0]

const ALL_ORDERS = permutations(3);
const showTerm = (v: number) => (v < 0 ? '−' : '') + Math.abs(v).toLocaleString('en-US');

function RaceStepper() {
  const [atomic, setAtomic] = useState(false);
  const [s, setS] = useState<RmwState>(() => rmwInit());
  const [found, setFound] = useState<number[]>([]);
  const [showAll, setShowAll] = useState(false);
  const [picked, setPicked] = useState<number[]>([]);
  const [seed, setSeed] = useState(5);
  const done = rmwDone(s);
  const tally = useMemo(() => {
    const t = new Map<number, number>();
    for (const sch of interleavings([3, 3])) {
      const v = cooUpdates(sch).y;
      t.set(v, (t.get(v) ?? 0) + 1);
    }
    return [...t.entries()].sort((a, b) => a[0] - b[0]);
  }, []);
  const orderTally = useMemo(() => {
    const t = new Map<number, number>();
    for (const o of ALL_ORDERS) t.set(floatSum(ORDER_TERMS, o), (t.get(floatSum(ORDER_TERMS, o)) ?? 0) + 1);
    return [...t.entries()].sort((a, b) => a[0] - b[0]);
  }, []);
  const want = ROW0_PRODUCTS[0] + ROW0_PRODUCTS[1];

  const stepThread = (t: number) => {
    const n = rmwStep(s, t, ROW0_PRODUCTS, atomic);
    setS(n);
    if (rmwDone(n) && !atomic && !found.includes(n.y)) setFound([...found, n.y].sort((a, b) => a - b));
  };
  const reset = (a = atomic) => { setAtomic(a); setS(rmwInit()); };
  const bothLoad = () => {
    setAtomic(false);
    const n = cooUpdates([0, 1, 0, 1, 0, 1]);
    setS(n);
    if (!found.includes(n.y)) setFound([...found, n.y].sort((a, b) => a - b));
  };
  const sums = picked.map((_, k) => floatSum(ORDER_TERMS, picked.slice(0, k + 1)));

  return (
    <Bench
      title={tr('Two threads, one y[0]', 'Hai luồng, một y[0]')}
      task={tr('Interleave the two threads so that one update is lost. Then switch the atomic on and try again.', 'Xen kẽ hai luồng sao cho một lần cập nhật bị mất. Rồi bật atomic lên và thử lại.')}
      foot={tr(
        "Row 0 of Fig. 17.1 holds 1 and 7; with x all ones, the two COO threads of row 0 add 1 and 7 into y[0], which starts at 0. The second panel's three products are ours, chosen to show rounding.",
        'Hàng 0 của Hình 17.1 chứa 1 và 7; với x toàn số 1, hai luồng COO của hàng 0 cộng 1 và 7 vào y[0], vốn bắt đầu bằng 0. Ba tích ở khung thứ hai là của khóa học này, được chọn để cho thấy việc làm tròn.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="check"><input type="checkbox" checked={atomic} onChange={(e) => reset(e.target.checked)} /> <Translated vi="cộng nguyên tử">atomic add</Translated></label>
        <button onClick={() => reset()}><Translated vi="Đặt lại">Reset</Translated></button>
        <button onClick={bothLoad}><Translated vi="Cả hai nạp, rồi cả hai ghi">Both load, then both store</Translated></button>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(14rem, 1fr))' }}>
        {[0, 1].map((t) => (
          <div key={t} className="panel" data-rmw-thread={t}>
            <p style={{ margin: '0 0 0.4rem' }}>
              <Translated vi={<><b>Luồng {t}</b>: cộng {ROW0_PRODUCTS[t]} (value {ROW0_PRODUCTS[t]} × x[{t}])</>}>
                <b>Thread {t}</b>: adds {ROW0_PRODUCTS[t]} (value {ROW0_PRODUCTS[t]} × x[{t}])
              </Translated>
            </p>
            <p style={{ margin: '0 0 0.5rem' }} className="muted">
              <Translated vi="tiếp theo:">next:</Translated>{' '}
              {atomic && s.pc[t] < 3
                ? tr('atomic add (load, add, store as one step)', 'cộng nguyên tử (nạp, cộng, ghi trong một bước)')
                : tr(RMW_NEXT[Math.min(3, s.pc[t])], ['nạp y', 'cộng tích của nó', 'ghi y', 'xong'][Math.min(3, s.pc[t])])}
              ; <Translated vi="thanh ghi:">register:</Translated>{' '}
              <span className="t-reg" data-k={'reg' + t}>{s.reg[t] ?? '–'}</span>
            </p>
            <button className="primary" onClick={() => stepThread(t)} disabled={s.pc[t] >= 3}><Translated vi={<>Chạy một bước luồng {t}</>}>Step thread {t}</Translated></button>
          </div>
        ))}
        <div className="panel">
          <p style={{ margin: 0 }}>
            <span className="t-global"><Translated vi="y[0] trong bộ nhớ">y[0] in memory</Translated></span>: <b data-k="y0">{s.y}</b>
          </p>
          {done ? (
            <p style={{ margin: '0.5rem 0 0' }} data-k="rmw-result">
              {s.y === want ? (
                <span className="chip ok"><Translated vi={<>{s.y}: cả hai lần cập nhật đều được ghi nhận</>}>{s.y}: both updates landed</Translated></span>
              ) : (
                <span className="chip fail">
                  <Translated vi={<>{s.y} thay vì {want}: lần cập nhật của luồng {s.y === ROW0_PRODUCTS[0] ? 1 : 0} đã bị mất</>}>
                    {s.y} instead of {want}: thread {s.y === ROW0_PRODUCTS[0] ? 1 : 0}'s update was lost
                  </Translated>
                </span>
              )}
            </p>
          ) : null}
        </div>
      </div>
      <ol className="muted" style={{ margin: '0.7rem 0', paddingLeft: '1.4rem', minHeight: '1.4em' }} data-k="rmw-log">
        {s.events.map((e, k) => (
          <li key={k}>
            <Translated vi={<>luồng {e.thread} {({ load: 'nạp', add: 'cộng', store: 'ghi', 'atomic add': 'cộng nguyên tử' } as Record<string, string>)[e.op] ?? e.op}: thanh ghi {e.reg}, y[0] = {e.y}</>}>
              thread {e.thread} {e.op}: register {e.reg}, y[0] = {e.y}
            </Translated>
          </li>
        ))}
      </ol>
      <div className="row">
        <span><Translated vi="Các kết quả bạn đã tìm được khi không có atomic:">Results you have found without the atomic:</Translated></span>
        {found.length ? found.map((v) => <span key={v} className={'chip ' + (v === want ? 'ok' : 'fail')} data-found={v}>{v}</span>) : <span className="muted"><Translated vi="chưa có">none yet</Translated></span>}
        <button onClick={() => setShowAll(true)} disabled={showAll}><Translated vi="Hiện mọi cách xen kẽ">Show every interleaving</Translated></button>
      </div>
      {showAll ? (
        <Translated
          vi={
            <p data-k="all-interleavings">
              {tally.reduce((a, [, n]) => a + n, 0)} cách xen kẽ hai luồng, mỗi luồng ba bước, kết thúc ở{' '}
              {tally.map(([v, n]) => `${v} (${n} cách)`).join(', ')}. Với atomic, mỗi luồng có một bước không thể chia cắt, và cả hai
              thứ tự đều cho {want}.
            </p>
          }
        >
          <p data-k="all-interleavings">
            The {tally.reduce((a, [, n]) => a + n, 0)} ways to interleave two threads of three steps each end at{' '}
            {tally.map(([v, n]) => `${v} (${n} of them)`).join(', ')}. With the atomic, each thread has one indivisible step, and both
            orders give {want}.
          </p>
        </Translated>
      ) : null}

      <hr style={{ border: 0, borderTop: '1px dashed var(--rule)', margin: '1rem 0' }} />
      <Translated
        vi={
          <p style={{ marginTop: 0 }}>
            <b>Atomic đã bật. Giờ đến thứ tự.</b> Hàng 1 của Hình 17.1 có ba phần tử khác 0, nên với COO có ba luồng cộng vào y[1].
            Giả sử các tích của chúng là {ORDER_TERMS.map(showTerm).join(', ')}. Hãy chọn thứ tự các phép cộng nguyên tử của chúng được thực hiện.
          </p>
        }
      >
        <p style={{ marginTop: 0 }}>
          <b>The atomic is on. Now the order.</b> Row 1 of Fig. 17.1 has three non-zeros, so under COO three threads add into y[1].
          Suppose their products are {ORDER_TERMS.map(showTerm).join(', ')}. Pick the order in which their atomic adds land.
        </p>
      </Translated>
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        {ORDER_TERMS.map((v, k) => (
          <button key={k} data-term={k} onClick={() => setPicked([...picked, k])} disabled={picked.includes(k)}>{showTerm(v)}</button>
        ))}
        <button onClick={() => { setPicked(shuffledOrder(3, seed)); setSeed(seed + 1); }}><Translated vi="Xáo">Shuffle</Translated></button>
        <button onClick={() => setPicked([])} disabled={!picked.length}><Translated vi="Xóa">Clear</Translated></button>
      </div>
      <p style={{ margin: 0 }} data-k="float-sum">
        <Translated vi="tổng tích lũy float32: 0">float32 running sum: 0</Translated>
        {picked.map((k, i) => <span key={i}> + {showTerm(ORDER_TERMS[k])} → <b>{showTerm(sums[i])}</b></span>)}
        {picked.length === 3 ? (
          <> {' '}<span className={'chip ' + (sums[2] === 1 ? 'ok' : 'fail')} data-k="float-result">
              <Translated vi={<>{showTerm(sums[2])}; tổng chính xác là 1</>}>{showTerm(sums[2])}; the exact sum is 1</Translated>
            </span></>
        ) : null}
      </p>
      {picked.length === 3 ? (
        <Translated
          vi={
            <p className="muted" style={{ margin: '0.5rem 0 0' }} data-k="float-all">
              Trên cả {ALL_ORDERS.length} thứ tự: {orderTally.map(([v, n]) => `${showTerm(v)} ở ${n} thứ tự`).join(', ')}. Các giá trị float32 gần
              100,000,000 cách nhau 8, nên cộng 1 vào một giá trị như thế không thay đổi gì; số 1 chỉ còn lại khi hai số hạng lớn triệt tiêu nhau trước. Atomic khiến mọi
              phép cộng đều được tính; nó không chọn thứ tự. Ghi chú "Today" cạnh phần COO cho biết các thư viện đưa ra lựa chọn này thế nào.
            </p>
          }
        >
          <p className="muted" style={{ margin: '0.5rem 0 0' }} data-k="float-all">
            Over all {ALL_ORDERS.length} orders: {orderTally.map(([v, n]) => `${showTerm(v)} in ${n}`).join(', ')}. Float32 values near
            100,000,000 are 8 apart, so adding 1 to one changes nothing; the 1 survives only when the two large terms cancel first. The atomic made every
            add count; it did not choose the order. The "Today" note beside the COO section says how the libraries offer the choice.
          </p>
        </Translated>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- comparison

const BOOK_VERDICT: Record<string, { flex: string; access: string }> = {
  coo: {
    get flex() {
      return tr('best: append, or reorder the three arrays together (p. 406)', 'tốt nhất: nối thêm, hoặc sắp lại cả ba mảng cùng nhau (p. 406)');
    },
    get access() {
      return tr('non-zero to its row and column (p. 406)', 'từ phần tử khác 0 ra hàng và cột của nó (p. 406)');
    },
  },
  csr: {
    get flex() {
      return tr('poor: insert, shift later rows, bump later rowPtrs (p. 409)', 'kém: chèn, dời các hàng sau, tăng các rowPtrs sau (p. 409)');
    },
    get access() {
      return tr('row to its non-zeros (p. 409)', 'từ hàng ra các phần tử khác 0 của nó (p. 409)');
    },
  },
  ell: {
    get flex() {
      return tr('better than CSR: a pad slot can take a non-zero (p. 413)', 'tốt hơn CSR: một vị trí đệm có thể nhận một phần tử khác 0 (p. 413)');
    },
    get access() {
      return tr('both: row to non-zeros, slot i to row i % stride (p. 413)', 'cả hai chiều: từ hàng ra các phần tử khác 0, từ vị trí i ra hàng i % stride (p. 413)');
    },
  },
  hyb: {
    get flex() {
      return tr('better than ELL: an overflow goes to the COO part (p. 415)', 'tốt hơn ELL: phần tràn đi vào phần COO (p. 415)');
    },
    get access() {
      return tr('weaker than ELL for rows that spilled (p. 415)', 'yếu hơn ELL với các hàng bị tràn (p. 415)');
    },
  },
  jds: {
    get flex() {
      return tr('worst: a new non-zero may force a re-sort (p. 417)', 'tệ nhất: một phần tử khác 0 mới có thể buộc phải sắp xếp lại (p. 417)');
    },
    get access() {
      return tr('row to its non-zeros only (p. 418)', 'chỉ từ hàng ra các phần tử khác 0 của nó (p. 418)');
    },
  },
  csc: {
    get flex() {
      return tr('as CSR (p. 420)', 'như CSR (p. 420)');
    },
    get access() {
      return tr('column to its non-zeros (p. 420)', 'từ cột ra các phần tử khác 0 của nó (p. 420)');
    },
  },
};

export function scoreRows(m: Matrix, K: number) {
  const list: { key: string; book: string; name: string; f: Format; loop: EllLoop }[] = [
    { key: 'coo', book: 'coo', name: 'COO', f: toCOO(m), loop: 'nnzPerRow' },
    { key: 'csr', book: 'csr', name: 'CSR', f: toCSR(m), loop: 'nnzPerRow' },
    { key: 'ell', book: 'ell', name: 'ELL, loop to nnzPerRow', f: toELL(m), loop: 'nnzPerRow' }, // i18n-ignore: translated where ScoreTable renders it
    { key: 'ell-full', book: 'ell', name: 'ELL, full width', f: toELL(m), loop: 'full' }, // i18n-ignore: translated where ScoreTable renders it
    { key: 'hyb', book: 'hyb', name: `Hybrid, K = ${K}`, f: toHybrid(m, K), loop: 'nnzPerRow' }, // i18n-ignore: translated where ScoreTable renders it
    { key: 'jds', book: 'jds', name: 'JDS', f: toJDS(m), loop: 'nnzPerRow' },
    { key: 'csc', book: 'csc', name: 'CSC', f: toCSC(m), loop: 'nnzPerRow' },
  ];
  return list.map((x) => {
    const t = spmvTrace(x.f, { ellLoop: x.loop });
    const s = summarize(t, 4, m.rows);
    const st = storage(x.f, m);
    const worst = Math.max(0, ...t.steps.flatMap((step) => matrixRuns(step, 4)));
    return { ...x, mapping: [...new Set(t.launches.map((l) => l.mapping))].join(', then '), storage: st, summary: s, worst };
  });
}

function ScoreTable({ st, set }: { st: ExState; set: (p: Partial<ExState>) => void }) {
  const K = kOf(st);
  const rows = useMemo(() => scoreRows(st.m, K), [st.m, K]);
  return (
    <Bench
      title={tr('All six formats on one matrix', 'Cả sáu định dạng trên một ma trận')}
      task={tr('Load Fig. 17.13 and compare it with Fig. 17.1. Which columns change, and which formats keep their verdict?', 'Nạp Hình 17.13 và so sánh với Hình 17.1. Những cột nào thay đổi, và định dạng nào giữ nguyên đánh giá của nó?')}
      foot={
        <Translated
          vi={
            <>
              Nhóm 4 luồng, như trong các hình của sách. "Dải tệ nhất" là số dải địa chỉ liên tiếp riêng rẽ nhiều nhất mà các lần đọc ma trận
              của một nhóm tạo ra trong một lần lặp bất kỳ: 1 là liền mạch. Hai cột cuối là đánh giá của sách. Ma trận và K là của
              trình khám phá, ở đầu trang.
            </>
          }
        >
          <>
            Groups of 4 threads, as in the book's figures. "Worst runs" is the most separate runs of consecutive addresses one group's
            matrix reads form in any one iteration: 1 is contiguous. The last two columns are the book's verdicts. The matrix and K are the
            explorer's, at the top of the page.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Matrices for the table', 'Ma trận cho bảng')}>
          {PRESETS.map((p) => (
            <button key={p.id} aria-pressed={st.preset === p.id} onClick={() => set({ m: p.m, preset: p.id, step: 0, order: 'row' })}>
              {tr(p.label, ({ 'Exercise 1': 'Bài tập 1' } as Record<string, string>)[p.label] ?? p.label)}
            </button>
          ))}
        </span>
        {st.preset === 'edited' ? <span className="muted"><Translated vi="ma trận bạn đã sửa">your edited matrix</Translated></span> : null}
      </div>
      <table className="data" data-score="" style={{ background: 'var(--page)', margin: 0 }}>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Định dạng">Format</Translated></th>
            <th><Translated vi="Một luồng mỗi">One thread per</Translated></th>
            <th className="num"><Translated vi="Đã lưu">Stored</Translated></th>
            <th className="num"><Translated vi="Phần đệm">Pads</Translated></th>
            <th className="num"><Translated vi="Phép cộng nguyên tử">Atomic adds</Translated></th>
            <th className="num"><Translated vi="Dải tệ nhất">Worst runs</Translated></th>
            <th className="num"><Translated vi="Lane chờ">Lanes waiting</Translated></th>
            <th className="num"><Translated vi="Lane trên phần đệm">Lanes on pads</Translated></th>
            <th><Translated vi="Tính linh hoạt (sách)">Flexibility (book)</Translated></th>
            <th><Translated vi="Truy cập (sách)">Access (book)</Translated></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} data-format={r.key}>
              <td style={{ paddingLeft: '0.6rem' }}>
                <b>
                  {tr(
                    r.name,
                    ({ ell: 'ELL, lặp tới nnzPerRow', 'ell-full': 'ELL, toàn bộ độ rộng', hyb: `Lai, K = ${K}` } as Record<string, string>)[r.key] ?? r.name,
                  )}
                </b>
              </td>
              <td>
                {tr(
                  r.mapping.replace(/^per /, '').replace(', then per ', ', then '),
                  r.mapping
                    .split(', then ')
                    .map((m) => ({ 'per non-zero': 'phần tử khác 0', 'per row': 'hàng', 'per sorted row': 'hàng đã sắp xếp', 'per column': 'cột' } as Record<string, string>)[m] ?? m)
                    .join(', rồi '),
                )}
              </td>
              <td className="num" data-col="stored">{r.storage.total} <span className="muted">({Math.round((100 * r.storage.total) / Math.max(1, r.storage.dense))}%)</span></td>
              <td className="num" data-col="pads">{r.storage.pads}</td>
              <td className="num" data-col="atomics">{r.summary.atomicUpdates}{r.summary.sharedY ? <span className="muted"><Translated vi={<> ({r.summary.sharedY} y dùng chung)</>}> ({r.summary.sharedY} y shared)</Translated></span> : null}</td>
              <td className="num" data-col="worst">{r.worst}</td>
              <td className="num" data-col="waiting">{r.summary.idle}</td>
              <td className="num" data-col="padwork">{r.summary.pad}</td>
              <td className="muted">{BOOK_VERDICT[r.book].flex}</td>
              <td className="muted">{BOOK_VERDICT[r.book].access}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions

const csrFig = spmvTrace(toCSR(FIG17_1));
const ell1 = summarize(spmvTrace(toELL(FIG17_1)), 4, 4);
const ellFull1 = summarize(spmvTrace(toELL(FIG17_1), { ellLoop: 'full' }), 4, 4);

export const questionsS172: Record<string, Question> = {
  'ch17/17.2/q-coo-y1': {
    kind: 'numeric',
    id: 'ch17/17.2/q-coo-y1',
    prompt: (
      <>
        <Translated
          vi={
            <>
              SpMV/COO trên Hình 17.1, với các bộ ba theo thứ tự của Hình 17.3. Có bao nhiêu luồng cộng vào <code>y[1]</code>?
            </>
          }
        >
          SpMV/COO on Fig. 17.1, with the triples in Fig. 17.3's order. How many threads add into <code>y[1]</code>?
        </Translated>
      </>
    ),
    answer: 3,
    explain: (
      <Translated
        vi={
          <p>
            Hàng 1 chứa 5, 3 và 9, ở các vị trí 2, 3 và 4 của các mảng COO. Mỗi vị trí là một luồng, và cả ba đều cộng vào{' '}
            <code>y[1]</code>, nên phép cộng phải là phép toán nguyên tử (atomic). Với CSR, luồng 1 sở hữu hàng 1: nó cộng ba tích trong một thanh ghi và
            cộng tổng vào <code>y[1]</code> một lần, không cần atomic. Các nhãn của trình khám phá cho thấy sự khác biệt: 8 phép cộng nguyên tử và 3 phần tử
            y dùng chung với COO, không có gì với CSR.
          </p>
        }
      >
        <p>
          Row 1 holds 5, 3 and 9, at positions 2, 3 and 4 of the COO arrays. Each position is one thread, and all three add into{' '}
          <code>y[1]</code>, so the add must be atomic. Under CSR, thread 1 owns row 1: it sums the three products in a register and
          adds the sum to <code>y[1]</code> once, with no atomic. The explorer's chips show the difference: 8 atomic adds and 3 shared
          y elements under COO, none under CSR.
        </p>
      </Translated>
    ),
    source: 'ch. 17 §17.2, pp. 404–405; §17.3, pp. 408–409', // i18n-ignore: citation
  },
  'ch17/17.2/q-ell-index': {
    kind: 'numeric',
    id: 'ch17/17.2/q-ell-index',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Lưu Hình 17.1 dưới dạng ELL với stride 8 thay vì 4: bốn hàng đệm được nối thêm để mọi lần lặp bắt đầu ở một
              ranh giới 32 byte. Chỉ số phẳng của phần tử <code>t = 2</code> của hàng 1, tức số 9, là bao nhiêu?
            </>
          }
        >
          Store Fig. 17.1 as ELL with stride 8 instead of 4: four rows of pads are appended so that every iteration starts on a
          32-byte boundary. What is the flat index of row 1's element <code>t = 2</code>, the 9?
        </Translated>
      </>
    ),
    answer: 17,
    explain: (
      <Translated
        vi={
          <p>
            <code>i = t·stride + row = 2·8 + 1 = 17</code>. Với stride 4 của sách, nó sẽ là 9. Các hàng 4 đến 7 chỉ chứa phần đệm, và không
            luồng nào làm việc trên chúng: phép kiểm tra biên là <code>row &lt; numRows</code>, không phải <code>row &lt; stride</code>. Trong thẻ
            ELL của trình khám phá, chọn "stride là bội của 8" và tìm số 9 ở vị trí 17.
          </p>
        }
      >
        <p>
          <code>i = t·stride + row = 2·8 + 1 = 17</code>. With the book's stride of 4 it would be 9. Rows 4 to 7 hold only pads, and no
          thread works on them: the bounds check is <code>row &lt; numRows</code>, not <code>row &lt; stride</code>. In the explorer's
          ELL tab, choose "stride a multiple of 8" and find the 9 in slot 17.
        </p>
      </Translated>
    ),
    source: 'ch. 17 §17.4, pp. 411–413; the 32-byte unit from notes/today/ch17-facts.md A3',
    vi: { source: 'ch. 17 §17.4, pp. 411–413; đơn vị 32 byte lấy từ notes/today/ch17-facts.md A3' },
  },
  'ch17/17.2/q-ell-assign': {
    kind: 'choice',
    id: 'ch17/17.2/q-ell-assign',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Hình 17.12 kết thúc bằng <code>y[row] = sum;</code>. Bạn dựng một SpMV lai ELL-COO cho <code>y = A·x + y</code>, trong đó{' '}
              <code>y</code> chứa giá trị của lần lặp trước (khác 0), và khởi chạy phần COO trước, rồi kernel ELL đúng như
              in trong sách. Điều gì xảy ra?
            </>
          }
        >
          Fig. 17.12 ends with <code>y[row] = sum;</code>. You build a hybrid ELL-COO SpMV for <code>y = A·x + y</code>, where{' '}
          <code>y</code> holds the previous iterate (non-zero), and launch the COO part first, then the ELL kernel exactly as
          printed. What happens?
        </Translated>
      </>
    ),
    options: [
      { label: <Translated vi="Kết quả đúng theo cách nào cũng vậy: phần COO chỉ chứa những hàng mà phần ELL không đụng tới.">It is correct either way: the COO part only holds rows the ELL part does not touch.</Translated>, why: <Translated vi="Hàng nào cũng có phần ELL, dù toàn là phần đệm, và kernel ELL ghi y[row] cho mọi hàng.">Every row has an ELL part, even if it is all pads, and the ELL kernel writes y[row] for every row.</Translated> },
      {
        label: (
          <Translated vi={<>Mọi hàng đều mất những gì <code>y</code> chứa trước khi kernel ELL chạy, gồm phần COO và giá trị ban đầu của nó; cách sửa là <code>y[row] += sum</code>.</>}>
            <>Every row loses what <code>y</code> held before the ELL kernel ran, its COO part and its starting value; the fix is <code>y[row] += sum</code>.</>
          </Translated>
        ),
        correct: true,
      },
      { label: <Translated vi="Chỉ các hàng dài hơn K ra sai: kernel ELL ghi đè phần đuôi COO của chúng, còn mọi hàng ngắn hơn đều ra đúng.">Only the rows longer than K come out wrong: the ELL kernel overwrites their COO tails, and every shorter row comes out right.</Translated>, why: <Translated vi="Những hàng đó mất phần đuôi COO, nhưng mọi hàng cũng mất giá trị y ban đầu: SpMV là y = A·x + y.">Those lose their COO tails, but every row also loses the y it started with: SpMV is y = A·x + y.</Translated> },
      { label: <Translated vi="Kết quả thay đổi giữa các lần chạy, vì hai kernel tranh chấp trên y[row] mà không có atomic.">The result changes from run to run, because the two kernels race on y[row] without an atomic.</Translated>, why: <Translated vi="Hai lần khởi chạy chạy nối tiếp nhau, nên không có gì tranh chấp: kernel ELL chỉ đơn giản ghi đè lên những gì phần COO để lại, lần chạy nào cũng y như vậy.">The two launches run one after the other, so nothing races: the ELL kernel simply writes over what the COO part left, the same way every run.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lần chạy trên trình giả lập các đoạn mã của trang này cho thấy điều đó trên Hình 17.1 với K = 2: phần COO trước, rồi dòng của sách, 4 trên 4 hàng
            sai; với <code>+=</code>, không hàng nào sai. Chạy phần COO sau kernel ELL chỉ che giấu vấn đề khi <code>y</code> bắt đầu
            bằng 0. Với một kernel ELL cộng dồn, hai phần kết hợp được theo thứ tự nào cũng được.
          </p>
        }
      >
        <p>
          The emulator run of this page's snippets shows it on Fig. 17.1 with K = 2: COO part first, then the page's line, 4 of 4 rows
          wrong; with <code>+=</code>, none. Running the COO part after the ELL kernel hides the problem only when <code>y</code> starts
          at zero. With an accumulating ELL kernel the two parts compose in either order.
        </p>
      </Translated>
    ),
    source: 'notes/ch17.md "Unclear" item 8; notes/today/ch17-facts.md A10; labs/ch17/snippets/check/spmv_check.cu on the emulator',
    vi: { source: 'notes/ch17.md "Unclear" mục 8; notes/today/ch17-facts.md A10; labs/ch17/snippets/check/spmv_check.cu trên trình giả lập' },
  },
  'ch17/17.2/q-pad-marker': {
    kind: 'choice',
    id: 'ch17/17.2/q-pad-marker',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Một kernel ELL lặp qua toàn bộ độ rộng đã đệm, và các vị trí đệm chứa cột 0 và giá trị 0 (nên phép kiểm tra{' '}
              <code>col &gt;= 0</code> cho chúng đi qua). <code>x[0]</code> là +Inf. Kết quả ra sao?
            </>
          }
        >
          An ELL kernel loops over the full padded width, and the pad slots hold column 0 and value 0 (so a <code>col &gt;= 0</code>{' '}
          test lets them through). <code>x[0]</code> is +Inf. What comes out?
        </Translated>
      </>
    ),
    options: [
      { label: <Translated vi="NaN ở mọi hàng có phần đệm, vì 0 × Inf là NaN.">NaN in every row that has a pad, because 0 × Inf is NaN.</Translated>, correct: true },
      { label: <Translated vi="Kết quả đúng: phần đệm bằng 0, nên chúng không cộng thêm gì.">The correct result: the pads are zero, so they add nothing.</Translated>, why: <Translated vi="0 nhân một số hữu hạn bằng 0; 0 nhân vô cùng là NaN trong số học IEEE.">Zero times a finite number is zero; zero times infinity is NaN in IEEE arithmetic.</Translated> },
      { label: <Translated vi="Chỉ hàng 0 sai, hàng thực sự có cột 0.">Only row 0 is wrong, the row that really has column 0.</Translated>, why: <Translated vi="Mọi vị trí đệm đều trỏ vào cột 0, bất kể nó thuộc hàng nào.">Every pad slot points at column 0, whatever row it belongs to.</Translated> },
      { label: <Translated vi="Inf ở mọi hàng, vì x[0] là vô cùng.">Inf in every row, because x[0] is infinite.</Translated>, why: <Translated vi="Một số khác 0 nhân Inf là Inf, nhưng một phần đệm cho 0 × Inf = NaN, và NaN thắng mọi tổng mà nó tham gia.">A non-zero times Inf is Inf, but a pad gives 0 × Inf = NaN, and NaN wins any sum it enters.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trên Hình 17.1, các hàng 0, 2 và 3 có phần đệm và ra NaN; hàng 1 không có. Với cột −1 trong phần đệm cùng phép kiểm tra, không phần đệm nào
            chạm tới <code>x</code>: đó là quy ước của các thư viện. cuSPARSE mô tả cùng sự khác biệt đó giữa mã dày đặc và mã thưa: tích vô hướng
            dày đặc của [0, 1, 0] và [1, 1, NaN] là NaN, tích thưa là 1.
          </p>
        }
      >
        <p>
          On Fig. 17.1, rows 0, 2 and 3 have pads and come out NaN; row 1 has none. With column −1 in the pads and the test, no pad
          touches <code>x</code>: the libraries' convention. cuSPARSE describes the same split between dense and sparse code: a dense
          dot product of [0, 1, 0] and [1, 1, NaN] is NaN, a sparse one is 1.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch17-facts.md A4, A6; the model (src/sim/sparse.ts) and the emulator run of the snippets',
    vi: { source: 'notes/today/ch17-facts.md A4, A6; mô hình (src/sim/sparse.ts) và lần chạy các đoạn mã trên trình giả lập' },
  },
  'ch17/17.2/q-ell-loop': {
    kind: 'choice',
    id: 'ch17/17.2/q-ell-loop',
    prompt: (
      <>
        <Translated
          vi={
            <>
              ELL có thể dừng ở <code>nnzPerRow[row]</code> (kernel của sách) hoặc lặp qua toàn bộ độ rộng và bỏ qua phần đệm. Phát biểu
              nào đúng?
            </>
          }
        >
          ELL can stop at <code>nnzPerRow[row]</code> (the book's kernel) or loop over the full width and skip the pads. Which
          statement holds?
        </Translated>
      </>
    ),
    options: [
      { label: <Translated vi="nnzPerRow giữ phân kỳ luồng của CSR nhưng không đọc phần đệm nào; toàn bộ độ rộng có số vòng lặp cố định, do hàng dài nhất quyết định.">nnzPerRow keeps CSR's divergence but reads no pads; the full width has a fixed trip count, set by the longest row.</Translated>, correct: true },
      { label: <Translated vi="Toàn bộ độ rộng nhanh hơn, vì không lane nào chờ một hàng dài hơn và phần đệm không tốn gì.">The full width is faster, because no lane waits for a longer row and the pads cost nothing.</Translated>, why: <Translated vi="Nó đổi các lane phải chờ lấy việc xử lý phần đệm, và một hàng dài khiến mọi luồng phải trả giá cho hàng đó.">It trades waiting lanes for pad work, and one long row makes every thread pay for that row.</Translated> },
      { label: <Translated vi="Cả hai đều loại bỏ phân kỳ luồng, vì ELL cho mọi hàng cùng một độ rộng đã đệm.">Both remove divergence, because ELL gives every row the same padded width.</Translated>, why: <Translated vi="Với nnzPerRow, các lane dừng ở độ dài hàng của chúng y như trong CSR (pp. 413–414).">With nnzPerRow, lanes stop at their row's length exactly as in CSR (pp. 413–414).</Translated> },
      { label: <Translated vi="Dừng ở nnzPerRow phá vỡ việc hợp nhất truy cập bộ nhớ, vì các lane còn hoạt động không còn đọc một dải liên tiếp.">Stopping at nnzPerRow breaks coalescing, because the lanes that are still active no longer read one contiguous run.</Translated>, why: <Translated vi="Một lane đã xong để lại một khoảng trống, nhưng vị trí của các lane còn hoạt động vẫn nằm trong cùng vài đoạn như của vòng lặp toàn bộ độ rộng, nên các truy cập vẫn được hợp nhất. Một lane rảnh không đọc gì.">A finished lane leaves a gap, but the active lanes' slots still sit inside the same few segments as the full-width loop's, so the accesses still coalesce. An idle lane reads nothing.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trên Hình 17.1 trong một nhóm 4 luồng, cả hai phiên bản đều chạy {ell1.groupIterations} lần lặp và làm {ell1.useful} lượt lane-lặp
            có ích. Dừng ở nnzPerRow, {ell1.idle} lượt lane-lặp phải chờ; lặp toàn bộ độ rộng, {ellFull1.idle} lượt chờ và{' '}
            {ellFull1.pad} lượt đọc một phần đệm thay vào đó. Trong mô hình này, vốn không có thời gian, nhóm tiêu tốn {ell1.useful + ell1.idle}{' '}
            lượt lane-lặp theo cách nào cũng vậy; điều toàn bộ độ rộng mua được là một vòng lặp không có số vòng phụ thuộc dữ liệu, với cái giá là phải đọc{' '}
            <code>colIdx</code> ở mọi phần đệm.
          </p>
        }
      >
        <p>
          On Fig. 17.1 in one group of 4, both versions run {ell1.groupIterations} iterations and do {ell1.useful} useful
          lane-iterations. Stopping at nnzPerRow, {ell1.idle} lane-iterations wait; looping the full width, {ellFull1.idle} wait and{' '}
          {ellFull1.pad} read a pad instead. In this model, which has no timing, the group spends {ell1.useful + ell1.idle}{' '}
          lane-iterations either way; what the full width buys is a loop with no data-dependent trip count, at the price of reading{' '}
          <code>colIdx</code> at every pad.
        </p>
      </Translated>
    ),
    source: 'ch. 17 §17.4, pp. 411, 413–414', // i18n-ignore: citation
  },
  'ch17/17.2/q-csc-use': {
    kind: 'choice',
    id: 'ch17/17.2/q-csc-use',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Bạn tính <code>y = A·x + y</code> trong đó bản thân <code>x</code> là thưa: 2% số mục của nó khác 0, và A có
              một triệu cột. Bạn có thể lưu A dưới dạng COO, CSR hoặc CSC. Cách bố trí nào cho phép một kernel bỏ qua phần việc mà các số 0 của{' '}
              <code>x</code> làm cho vô nghĩa, và nó phải trả giá gì cho điều đó?
            </>
          }
        >
          You compute <code>y = A·x + y</code> where <code>x</code> is itself sparse: 2% of its entries are non-zero, and A has a
          million columns. You may store A as COO, CSR or CSC. Which layout lets a kernel skip the work that the zeros of{' '}
          <code>x</code> make pointless, and what does it pay for that?
        </Translated>
      </>
    ),
    options: [
      {
        label: (
          <Translated vi={<>CSC: các phần tử khác 0 của <code>x</code> chọn ra các cột, mỗi cột được lưu liền nhau; một kernel đọc khoảng 2% của A nhưng cần atomic trên <code>y</code>.</>}>
            <>CSC: <code>x</code>'s non-zeros pick the columns, each stored together; a kernel reads about 2% of A but needs atomics on <code>y</code>.</>
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi={<>CSR: một luồng mỗi hàng sở hữu <code>y[row]</code>, nên nó có thể bỏ qua các mục gặp một số 0 của <code>x</code> và không cần atomic.</>}>
            <>CSR: one thread per row owns <code>y[row]</code>, so it can skip the entries that meet a zero of <code>x</code> and needs no atomics.</>
          </Translated>
        ),
        why: (
          <Translated vi="Nó không cần atomic, nhưng nó không bỏ qua được gì: một mục x bằng 0 vô hiệu cả một cột, còn CSR nhóm theo hàng, nên mọi hàng vẫn phải được đọc để tìm xem mục nào của nó gặp một x khác 0.">
            It needs no atomics, but it skips nothing: a zero x entry kills a column, and CSR groups by row, so every row must still be read to find which of its entries meet a non-zero x.
          </Translated>
        ),
      },
      { label: <Translated vi="COO: các bộ ba của nó có thể nằm theo bất kỳ thứ tự nào, nên những bộ ba có ích có thể đứng trước và phần việc còn lại được cân bằng tốt nhất.">COO: its triples can sit in any order, so the useful ones can go first and the remaining work is balanced best.</Translated>, why: <Translated vi="Cân bằng đến sau khi tìm ra việc. Các mục của những cột được chọn rải khắp z bộ ba, nên tìm ra chúng nghĩa là phải đọc mọi bộ ba, trừ khi chúng được sắp xếp theo cột, mà đó chính là CSC với một mảng chỉ số dài hơn.">Balance comes after finding the work. The entries of the chosen columns are spread over all z triples, so finding them means reading every triple, unless they are sorted by column, which is CSC with a longer index array.</Translated> },
      { label: <Translated vi="CSC, không cần atomic, vì mỗi luồng sở hữu một cột và do đó sở hữu phần đầu ra của riêng nó.">CSC, with no atomics, because each thread owns one column and therefore its own share of the output.</Translated>, why: <Translated vi="Một luồng sở hữu một cột, nhưng đầu ra được đánh chỉ số theo hàng: hai cột có phần tử khác 0 trong cùng một hàng cộng vào cùng một phần tử y (pp. 419–420).">A thread owns a column, but the output is indexed by row: two columns with a non-zero in the same row add into the same y element (pp. 419–420).</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một tích <code>x[col]·A[row][col]</code> bằng 0 mỗi khi <code>x[col]</code> bằng 0, nên một số 0 trong <code>x</code> làm cho cả
            một cột trở nên vô nghĩa. Sách gọi tên trường hợp này, phép nhân ma trận thưa với vector thưa, là một trong hai công dụng của CSC: các
            phần tử khác 0 của cột được nhóm lại, nên bỏ qua nó rất rẻ (p. 421). Các atomic trên <code>y</code> vẫn còn, như trong kernel
            SpMV/CSC của sách. Công dụng còn lại là tích vector–ma trận <code>xᵀ·A</code>, trong đó mỗi đầu ra là tích vô hướng của{' '}
            <code>x</code> với một cột.
          </p>
        }
      >
        <p>
          A product <code>x[col]·A[row][col]</code> is zero whenever <code>x[col]</code> is, so a zero in <code>x</code> makes a
          whole column pointless. The book names this case, sparse-matrix–sparse-vector multiplication, as one of CSC's two uses: the
          column's non-zeros are grouped, so skipping it is cheap (p. 421). The atomics on <code>y</code> stay, as in the book's
          SpMV/CSC kernel. The other use is the vector–matrix product <code>xᵀ·A</code>, where each output is the dot product of{' '}
          <code>x</code> with one column.
        </p>
      </Translated>
    ),
    source: 'ch. 17 §17.7, pp. 419–421', // i18n-ignore: citation
  },
  'ch17/17.2/q-ex1-pads': {
    kind: 'numeric',
    id: 'ch17/17.2/q-ex1-pads',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Ma trận của bài tập 1, các hàng <code>[1,0,7,0]</code>, <code>[0,0,8,0]</code>, <code>[0,4,3,0]</code>, <code>[2,0,0,1]</code>, ở dạng
              ELL: <code>value</code> chứa bao nhiêu vị trí đệm?
            </>
          }
        >
          Exercise 1's matrix, rows <code>[1,0,7,0]</code>, <code>[0,0,8,0]</code>, <code>[0,4,3,0]</code>, <code>[2,0,0,1]</code>, in
          ELL: how many pad slots does <code>value</code> hold?
        </Translated>
      </>
    ),
    answer: 1,
    explain: (
      <Translated
        vi={
          <p>
            Hàng dài nhất có 2 phần tử khác 0, nên ELL chứa 4 × 2 = 8 vị trí cho 7 phần tử khác 0, và hàng 1 nhận phần đệm duy nhất. Nạp ma trận
            "Bài tập 1" trong trình khám phá để xem các định dạng khác của bài tập; thẻ CSR của nó cho thấy <code>rowPtrs = [0, 2, 3, 5, 7]</code>.
          </p>
        }
      >
        <p>
          The longest row has 2 non-zeros, so ELL holds 4 × 2 = 8 slots for 7 non-zeros, and row 1 gets the one pad. Load the
          "Exercise 1" matrix in the explorer to see the exercise's other formats; its CSR tab shows <code>rowPtrs = [0, 2, 3, 5, 7]</code>.
        </p>
      </Translated>
    ),
    source: 'ch. 17 §17.9, exercise 1, p. 422 (answer ours: notes/ch17.md, exercises)',
    vi: { source: 'ch. 17 §17.9, bài tập 1, p. 422 (lời giải của khóa học này: notes/ch17.md, phần bài tập)' },
  },
  'ch17/17.2/q-ex2-missing': {
    kind: 'choice',
    id: 'ch17/17.2/q-ex2-missing',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Bài tập 2: một ma trận số nguyên có m hàng, n cột và z phần tử khác 0. Số lượng số nguyên của những định dạng nào không thể tính chỉ từ m,
              n và z?
            </>
          }
        >
          Exercise 2: an integer matrix has m rows, n columns and z non-zeros. Which formats' integer counts cannot be computed from m,
          n and z alone?
        </Translated>
      </>
    ),
    options: [
      { label: <Translated vi="ELL và JDS: cả hai đều cần K, độ dài của hàng dài nhất.">ELL and JDS: both need K, the length of the longest row.</Translated>, correct: true },
      { label: <Translated vi="COO và CSR: cả hai phụ thuộc vào cách các hàng được lấp đầy.">COO and CSR: both depend on how rows are filled.</Translated>, why: <Translated vi="COO là 3z và CSR là 2z + m + 1: không cái nào phụ thuộc vào cách các phần tử khác 0 phân bố trên các hàng.">COO is 3z and CSR is 2z + m + 1: neither depends on how the non-zeros are spread over the rows.</Translated> },
      { label: <Translated vi="Cả bốn, vì mỗi định dạng đều cần n, số cột.">All four, because each needs n, the column count.</Translated>, why: <Translated vi="Không định dạng nào trong COO, CSR, ELL hay JDS lưu gì theo từng cột, nên n không xuất hiện trong số đếm nào.">None of COO, CSR, ELL or JDS stores anything per column, so n appears in none of the counts.</Translated> },
      { label: <Translated vi="Chỉ ELL: JDS không có phần đệm, nên m và z xác định được nó.">ELL alone: JDS has no pads, so m and z fix it.</Translated>, why: <Translated vi="JDS không có phần đệm, nhưng nó lưu iterPtr, mỗi lần lặp một mục cộng thêm một: K + 1 mục.">JDS has no pads, but it stores iterPtr, one entry per iteration plus one: K + 1 entries.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            COO 3z; CSR 2z + m + 1; ELL 2·m·K, cộng m nếu <code>nnzPerRow</code> được lưu như Hình 17.12 cần; JDS 2z + m + K + 1 (mảng{' '}
            <code>row</code> và <code>iterPtr</code>). Chỉ một định dạng theo cột như CSC mới cần n. Trình khám phá in số đếm của từng
            định dạng với chính m, n, z và K của ma trận.
          </p>
        }
      >
        <p>
          COO 3z; CSR 2z + m + 1; ELL 2·m·K, plus m if <code>nnzPerRow</code> is stored as Fig. 17.12 needs; JDS 2z + m + K + 1 (the{' '}
          <code>row</code> array and <code>iterPtr</code>). Only a column format such as CSC would need n. The explorer prints each
          format's count with the matrix's own m, n, z and K.
        </p>
      </Translated>
    ),
    source: 'ch. 17 §17.9, exercise 2, p. 422 (answer ours: notes/ch17.md, exercises)',
    vi: { source: 'ch. 17 §17.9, bài tập 2, p. 422 (lời giải của khóa học này: notes/ch17.md, phần bài tập)' },
  },
};

// ---------------------------------------------------------------- the page

function CsrReveal({ v }: { v: number }) {
  const t1 = csrFig.steps[1].lanes;
  return (
    <>
      <Translated
        vi={
          <p>
            {v === t1[2].index ? 'Đúng. ' : `Đáp án là ${t1[2].index}. `}Hàng 2 bắt đầu ở <code>rowPtrs[2] = {toCSR(FIG17_1).rowPtrs[2]}</code>,
            nên phần tử 1 của nó nằm ở {t1[2].index}. Ở lần lặp 1, bốn luồng đọc{' '}
            {t1.map((l) => (l.index == null ? `không gì cả (hàng của luồng ${l.thread} có một phần tử)` : `value[${l.index}]`)).join(', ')}.
          </p>
        }
      >
        <p>
          {v === t1[2].index ? 'Right. ' : `The answer is ${t1[2].index}. `}Row 2 starts at <code>rowPtrs[2] = {toCSR(FIG17_1).rowPtrs[2]}</code>,
          so its element 1 is at {t1[2].index}. In iteration 1 the four threads read{' '}
          {t1.map((l) => (l.index == null ? `nothing (thread ${l.thread}'s row has one element)` : `value[${l.index}]`)).join(', ')}.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chạy thẻ CSR của trình khám phá tới lần lặp 1 để thấy điều đó, rồi so nhãn "lần đọc ma trận" với nhãn của thẻ ELL ở cùng bước.
          </p>
        }
      >
        <p>
          Step the explorer's CSR tab to iteration 1 to see it, then compare the "matrix reads" chip with the ELL tab's at the same step.
        </p>
      </Translated>
    </>
  );
}

export function S172() {
  const [st, setSt] = useState<ExState>(INITIAL);
  const set = (p: Partial<ExState>) => setSt((s) => ({ ...s, ...p }));
  const setTab = (tab: FormatId) => set({ tab, step: 0 });
  /** Open a tab of the explorer at a step and bring it into view. */
  const show = (tab: FormatId, step = 0, extra: Partial<ExState> = {}) => {
    set({ tab, step, ...extra });
    document.querySelector(tr('section.bench[aria-label="Format explorer"]', 'section.bench[aria-label="Trình khám phá định dạng"]'))?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };
  const Open = ({ tab, step = 0, extra, children }: { tab: FormatId; step?: number; extra?: Partial<ExState>; children: ReactNode }) => (
    <button className="quiet" data-open={`${tab}:${step}`} onClick={() => show(tab, step, extra)}>{children}</button>
  );

  return (
    <>
      <Translated
        vi={
          <p>
            Chương này đặt một câu hỏi cho sáu định dạng lưu trữ. Khi phần lớn A là số 0, nên lưu <code>y = A·x + y</code> thế nào,
            và mỗi cách bố trí buộc kernel làm gì? Định dạng quyết định luồng (thread) nào sở hữu phần việc nào. Ba
            điều suy ra từ đó, và bạn có thể đọc cả ba ngay từ các mảng mà không cần chạy gì: liệu hai luồng có cập nhật cùng
            một phần tử <code>y</code> không, các luồng kề nhau có đọc các địa chỉ kề nhau không, và chúng có xong cùng lúc không.
            Trang này dựng bốn định dạng bằng tay và gán cho mỗi định dạng kernel của nó. Trang 17.5 bổ sung hai định dạng mua sự đều đặn của ELL với ít
            phần đệm hơn.
          </p>
        }
      >
        <p>
          This chapter asks one question of six storage formats. When most of A is zeros, how should <code>y = A·x + y</code> be
          stored, and what does each layout make the kernel do? The format decides which thread owns which piece of work. Three
          things follow from that, and you can read all three off the arrays without running anything: whether two threads update the
          same <code>y</code> element, whether neighbouring threads read neighbouring addresses, and whether they finish together.
          This page builds four formats by hand and gives each its kernel. Page 17.5 adds the two that buy ELL's regularity with less
          padding.
        </p>
      </Translated>
      <Source section="17.1" pages="401–404" />

      <h2>
        <Translated vi="Ví dụ xuyên suốt">The running example</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách đưa một ma trận 4 × 4 qua mọi định dạng (Hình 17.1): 8 phần tử khác 0 trong các hàng dài 2, 3, 2 và 1. Mỗi hàng là một
            phương trình chỉ liên quan đến vài ẩn số. Nó đủ nhỏ để mọi mảng nằm vừa trên một dòng và đủ không đều để
            cho thấy mọi vấn đề mà chương này bàn tới.
          </p>
        }
      >
        <p>
          The book works one 4 × 4 matrix through every format (Fig. 17.1): 8 non-zeros in rows of 2, 3, 2 and 1. Each row is one
          equation that involves only a few of the unknowns. It is small enough that every array fits on a line and uneven enough to
          show every problem the chapter is about.
        </p>
      </Translated>
      <Source section="17.1" pages="402" figs={tr('Fig. 17.1', 'Hình 17.1')} />
      <Translated
        vi={
          <p>
            Trình khám phá bên dưới là công cụ của trang này, và mỗi phần sau nó giao cho bạn một nhiệm vụ trong một thẻ của nó. Bấm vào một ô để
            thêm hoặc xóa một phần tử khác 0, hoặc nạp một ma trận khác của sách. Rê chuột lên bất kỳ ô nào, trong ma trận hay trong một mảng, để tìm nó ở
            bên kia. Chạy từng bước kernel để thấy, từng luồng một, mỗi luồng đọc vị trí nào và kết quả của nó đi đâu. Các thẻ lai và JDS
            thuộc về trang 17.5; chúng có mặt ở đây để phần so sánh cuối trang này phủ được cả sáu.
          </p>
        }
      >
        <p>
          The explorer below is this page's instrument, and each section after it gives you a task in one of its tabs. Click a cell to
          add or remove a non-zero, or load another of the book's matrices. Hover any cell, in the matrix or in an array, to find it in
          the other. Step the kernel to see, thread by thread, which slot each one reads and where its result goes. The hybrid and JDS
          tabs belong to page 17.5; they are here so that the comparison at the end of this page covers all six.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các luồng chạy theo nhóm 4 (hoặc một nhóm mỗi hàng), như các hình của sách vẽ. Nhóm thật rộng hơn; các độ rộng mà khóa
              học này biết: <WidthsNote />.
            </p>
          }
        >
          <p>
            Threads run in groups of 4 (or one group per row), as the book's figures draw them. Real groups are wider; the widths this
            course knows of: <WidthsNote />.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Trên một ma trận nhỏ như vậy, cách đếm sector của chương 6 sẽ xếp bốn lần đọc 4 byte vào một đoạn 32 byte ngay cả khi chúng
              rải rác (các vị trí 0, 2, 5 và 7 của CSR đều rơi vào byte 0 đến 31), nên thay vào đó trình khám phá đếm các dải địa chỉ liên tiếp:
              một dải là hình ảnh của sách về một lần đọc được hợp nhất (coalescing). Không có cache, không có thời gian, atomic không tốn gì.
            </p>
          }
        >
          <p>
            On a matrix this small, chapter 6's sector count would put four 4-byte reads in one 32-byte segment even when they are
            scattered (CSR's slots 0, 2, 5 and 7 all fall in bytes 0 to 31), so the explorer counts runs of consecutive addresses
            instead: one run is the book's picture of a coalesced read. No caches, no timing, no cost for an atomic.
          </p>
        </Translated>
      </Simplification>
      <Explorer st={st} set={set} setTab={setTab} />

      <h2>
        <Translated vi="COO: một luồng mỗi phần tử khác 0">COO: one thread per non-zero</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Định dạng tọa độ lưu mỗi phần tử khác 0 dưới dạng một bộ ba, trải trên ba mảng cùng độ dài: <code>rowIdx</code>,{' '}
            <code>colIdx</code> và <code>value</code>. Kernel tự nhiên giao cho mỗi luồng một bộ ba. Luồng nhân giá trị của nó với{' '}
            <code>x[col]</code> và cộng tích vào <code>y[row]</code>.
          </p>
        }
      >
        <p>
          The coordinate format stores each non-zero as a triple, spread over three arrays of equal length: <code>rowIdx</code>,{' '}
          <code>colIdx</code> and <code>value</code>. The natural kernel gives each thread one triple. It multiplies its value by{' '}
          <code>x[col]</code> and adds the product into <code>y[row]</code>.
        </p>
      </Translated>
      <Source section="17.2" pages="404–405" figs={tr('Figs. 17.3–17.5', 'Hình 17.3–17.5')} />
      <Translated
        vi={
          <p>
            Mọi phần tử khác 0 của một hàng đều nhắm vào cùng một phần tử <code>y</code>, nên nhiều luồng cộng vào một chỗ: phép cộng phải là
            phép toán nguyên tử (atomic). Đó là cái giá duy nhất của COO. Mọi thứ khác ở nó đều tốt. Các luồng liên tiếp đọc các phần tử liên tiếp của cả ba
            mảng, nên các lần nạp ma trận được hợp nhất. Mỗi luồng làm đúng một phép nhân, nên không luồng nào chờ luồng khác, trừ ở
            phép kiểm tra biên. <Open tab="coo">Mở thẻ COO</Open>
          </p>
        }
      >
        <p>
          All of a row's non-zeros target the same <code>y</code> element, so several threads add into one place: the add has to be
          atomic. That is COO's one cost. Everything else about it is good. Consecutive threads read consecutive elements of all three
          arrays, so the matrix loads coalesce. Every thread does exactly one multiply, so no thread waits for another, except at the
          bounds check. <Open tab="coo">Open the COO tab</Open>
        </p>
      </Translated>
      <Source section="17.2" pages="405–406" figs={tr('Fig. 17.4', 'Hình 17.4')} />
      <Ask q={questionsS172['ch17/17.2/q-coo-y1']} label={tr('Work it out', 'Tự tính ra')} />
      <Snippet
        file={SNIPPETS.coo}
        marks={COO_ATOMIC_LINES}
        caption={
          <Translated
            vi={
              <>
                Hình 17.5 từ <code>labs/ch17/snippets/spmv_coo.cu</code>. Nó thêm <code>#include &lt;cuda/atomic&gt;</code> mà trang sách
                bỏ sót và struct <code>COOMatrix</code> mà sách không bao giờ định nghĩa. Các dòng được tô sáng là phép cộng nguyên tử, với
                phạm vi device và thứ tự relaxed. Đã chạy trên trình giả lập của khóa học theo ba thứ tự luồng; trình giả lập race đánh dấu một bản sao không có
                atomic.
              </>
            }
          >
            <>
              Fig. 17.5 from <code>labs/ch17/snippets/spmv_coo.cu</code>. It adds the <code>#include &lt;cuda/atomic&gt;</code> the page
              leaves out and the <code>COOMatrix</code> struct the book never defines. The highlighted lines are the atomic add, with
              device scope and relaxed order. Run on the course's emulator in three thread orders; the race emulator flags a copy without
              the atomic.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            COO cũng là định dạng linh hoạt nhất. Các bộ ba có thể nằm theo bất kỳ thứ tự nào, miễn là ba mảng di chuyển cùng nhau: trong thẻ
            COO của trình khám phá, chuyển sang <Open tab="coo" step={0} extra={{ order: 'fig17_6', m: FIG17_1, preset: 'fig17_1' }}>thứ tự của Hình 17.6</Open>{' '}
            và so sánh <code>y</code> với thứ tự hàng. Thêm một phần tử khác 0 chỉ là nối thêm vào cuối. Đó là lý do ma trận thường được lắp ráp
            ở dạng COO rồi chuyển đổi sau. Điều COO không trả lời rẻ được là "những phần tử khác 0 nào nằm ở hàng r?".
          </p>
        }
      >
        <p>
          COO is also the most flexible format. The triples can sit in any order, as long as the three arrays move together: in the
          explorer's COO tab, switch to <Open tab="coo" step={0} extra={{ order: 'fig17_6', m: FIG17_1, preset: 'fig17_1' }}>Fig. 17.6's order</Open>{' '}
          and compare <code>y</code> with the row order. Adding a non-zero is an append. That is why matrices are often assembled
          in COO and converted later. What COO cannot answer cheaply is "which non-zeros are in row r?".
        </p>
      </Translated>
      <Source section="17.2" pages="405–406" figs={tr('Fig. 17.6', 'Hình 17.6')} />
      <Translated
        vi={
          <p>
            Sách nêu hai cái giá của atomic. Một là thông lượng: các luồng xếp hàng trên một địa chỉ. Cái kia là thứ tự. Các phép cộng nguyên tử vào
            một phần tử xảy ra từng cái một, theo thứ tự không xác định, và phép cộng số thực không có tính kết hợp, nên các bit thấp của{' '}
            <code>y</code> có thể thay đổi từ lần chạy này sang lần chạy khác.
          </p>
        }
      >
        <p>
          The book names two costs of the atomic. One is throughput: threads queue on one address. The other is order. Atomic adds into
          one element happen one at a time, in an undefined order, and float addition is not associative, so the low bits of{' '}
          <code>y</code> can change from one run to the next.
        </p>
      </Translated>
      <Source section="17.2" pages="406–407" />
      <Translated
        vi={
          <p>
            Các thư viện của nhà cung cấp biến cái giá thứ hai đó thành một công tắc: một thuật toán nhanh hơn mà kết quả có thể khác ở các bit cuối giữa
            các lần chạy, hoặc một thuật toán tất định nhưng chậm hơn. Thuật toán nào chạy được chọn trong lời gọi, không phải bởi định dạng.
          </p>
        }
      >
        <p>
          The vendor libraries turn that second cost into a switch: a faster algorithm whose result may differ in the last bits from
          run to run, or a deterministic one that is slower. Which one runs is chosen in the call, not by the format.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch17-facts.md, A2, A7, B5)', '2026-11-20 (notes/today/ch17-facts.md, mục A2, A7, B5)')}>
        <Translated
          vi={
            <p>
              <code>COO_ALG1</code> của cuSPARSE có thể khác nhau giữa các lần chạy, <code>COO_ALG2</code> thì tất định; rocSPARSE có cặp
              tương tự. cuSPARSE giả định COO được sắp theo hàng (A7); kernel của sách thì không.
            </p>
          }
        >
          <p>
            cuSPARSE's <code>COO_ALG1</code> may differ between runs, <code>COO_ALG2</code> is deterministic; rocSPARSE has the same
            pair. cuSPARSE assumes COO sorted by row (A7); the book's kernel does not.
          </p>
        </Translated>
      </Today>
      <Translated vi={<p>Trình chạy từng bước bên dưới làm cho cả hai cái giá trở nên cụ thể, trước hết là lần cập nhật bị mất rồi đến thứ tự.</p>}>
        <p>The stepper below makes both costs concrete, the lost update first and then the order.</p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một <code>+=</code> thường là ba bước, nạp, cộng và ghi, trên một <code>y[0]</code> dùng chung; atomic là một
              bước không thể chia cắt và không tốn gì. Không có cache, không có thời gian. Khung thứ tự làm tròn mọi phép cộng về float32.
            </p>
          }
        >
          <p>
            A plain <code>+=</code> is three steps, load, add and store, on one shared <code>y[0]</code>; the atomic is one
            indivisible step that costs nothing. No caches, no timing. The order panel rounds every add to float32.
          </p>
        </Translated>
      </Simplification>
      <RaceStepper />

      <h2>
        <Translated vi="CSR: một luồng mỗi hàng">CSR: one thread per row</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Nhóm các phần tử khác 0 theo hàng, và mảng hàng có thể thu lại còn một độ lệch mỗi hàng. CSR giữ <code>colIdx</code> và{' '}
            <code>value</code> theo thứ tự hàng và thay <code>rowIdx</code> (z mục) bằng <code>rowPtrs</code> (m + 1 mục): hàng r
            là lát từ <code>rowPtrs[r]</code> tới <code>rowPtrs[r+1]</code>, và mục thừa cuối cùng kết thúc hàng cuối.
          </p>
        }
      >
        <p>
          Group the non-zeros by row, and the row array can shrink to one offset per row. CSR keeps <code>colIdx</code> and{' '}
          <code>value</code> in row order and replaces <code>rowIdx</code> (z entries) with <code>rowPtrs</code> (m + 1 entries): row r
          is the slice from <code>rowPtrs[r]</code> up to <code>rowPtrs[r+1]</code>, and the extra last entry ends the last row.
        </p>
      </Translated>
      <Source section="17.3" pages="407–408" figs={tr('Fig. 17.7', 'Hình 17.7')} />
      <Translated
        vi={
          <p>
            Giờ một luồng có thể sở hữu một hàng. Nó cộng các tích của hàng trong một thanh ghi (register) và cộng tổng vào <code>y[row]</code> một lần. Không có hai
            luồng nào ghi cùng một phần tử, nên atomic biến mất.
          </p>
        }
      >
        <p>
          Now one thread can own a row. It sums the row's products in a register and adds the sum to <code>y[row]</code> once. No two
          threads write the same element, so the atomic is gone.
        </p>
      </Translated>
      <Source section="17.3" pages="408–409" figs={tr('Figs. 17.8–17.9', 'Hình 17.8–17.9')} />
      <Snippet
        file={SNIPPETS.csr}
        marks={[CSR_LOOP_LINE, 'y[row] += sum;']}
        caption={
          <Translated
            vi={
              <>
                Hình 17.9 từ <code>labs/ch17/snippets/spmv_csr.cu</code>, với struct <code>CSRMatrix</code> mà sách không bao giờ định nghĩa.
                Các cận vòng lặp dựa vào mục đánh dấu kết thúc; dòng cuối cộng dồn. Không có gì trong đó cần cách viết riêng cho HIP (ch05-facts
                B4, ch17-facts B1); chưa biên dịch cho HIP ở đây: unverified.
              </>
            }
          >
            <>
              Fig. 17.9 from <code>labs/ch17/snippets/spmv_csr.cu</code>, with the <code>CSRMatrix</code> struct the book never defines.
              The loop bounds lean on the end marker; the last line accumulates. Nothing in it needs a HIP-specific spelling (ch05-facts
              B4, ch17-facts B1); not compiled for HIP here: unverified.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Cái giá chuyển sang bộ nhớ và thời gian. Ở lần lặp t của vòng lặp, luồng r đọc vị trí <code>rowPtrs[r] + t</code>. Ở
            lần lặp 0, các luồng 0 đến 3 đọc <code>value[0]</code>, <code>value[2]</code>, <code>value[5]</code> và{' '}
            <code>value[7]</code>: bốn chỗ riêng rẽ, không phải một dải. Và vòng lặp chạy số lần bằng độ dài của hàng, nên trong một nhóm
            luồng, các hàng ngắn xong rồi chờ trong khi các hàng dài tiếp tục. Sách cho rằng điều đó xảy ra ở hầu hết hoặc mọi warp của một ma trận
            thực tế.
          </p>
        }
      >
        <p>
          The cost moves to memory and to time. In iteration t of the loop, thread r reads slot <code>rowPtrs[r] + t</code>. In
          iteration 0, threads 0 to 3 read <code>value[0]</code>, <code>value[2]</code>, <code>value[5]</code> and{' '}
          <code>value[7]</code>: four separate places, not one run. And the loop runs as many times as the row is long, so in a group
          of threads the short rows finish and wait while the long ones go on. The book expects that in most or all warps of a real
          matrix.
        </p>
      </Translated>
      <Source section="17.3" pages="409–410" />
      <Predict
        id="ch17/17.2/p-csr-t1"
        prompt={
          <>
            <Translated
              vi={
                <>
                  Ở lần lặp 1 của SpMV/CSR trên Hình 17.1, luồng 2 đọc chỉ số nào của <code>value</code>? Hãy chốt câu trả lời trước khi chạy thẻ
                  CSR của trình khám phá.
                </>
              }
            >
              In iteration 1 of SpMV/CSR on Fig. 17.1, which index of <code>value</code> does thread 2 read? Commit before you step the
              explorer's CSR tab.
            </Translated>
          </>
        }
        unit={tr('(an index)', '(một chỉ số)')}
      >
        {(v) => <CsrReveal v={v} />}
      </Predict>
      <Translated
        vi={
          <p>
            <Open tab="csr" step={1}>Xem CSR ở lần lặp 1</Open> CSR nhỏ hơn COO và không cần atomic. Nó kém linh hoạt hơn
            (chèn một phần tử khác 0 dời mọi hàng phía sau và tăng mọi mục <code>rowPtrs</code> phía sau) và không cho truy cập theo cột.
            Hai vấn đề của nó, các lần đọc rải rác và số vòng lặp không đều, là thứ các định dạng tiếp theo giải quyết.
          </p>
        }
      >
        <p>
          <Open tab="csr" step={1}>Show CSR at iteration 1</Open> CSR is smaller than COO and needs no atomics. It is less flexible
          (inserting a non-zero shifts every later row and bumps every later <code>rowPtrs</code> entry) and gives no access by column.
          Its two problems, scattered reads and uneven trip counts, are what the next formats work on.
        </p>
      </Translated>
      <Source section="17.3" pages="409–410" />

      <h2>
        <Translated vi="ELL: đệm, rồi xoay">ELL: pad, then turn</Translated>
      </h2>
      <Translated
        vi={
          <p>
            ELL (đặt theo tên gói ELLPACK) sửa các lần đọc bằng hai nước đi. Đệm cuối mỗi hàng cho dài bằng hàng dài nhất,
            để các phần tử khác 0 tạo thành một hình chữ nhật. Rồi lưu hình chữ nhật theo từng cột: phần tử 0 của mọi hàng, rồi phần tử 1
            của mọi hàng, và cứ thế. Phần tử t của hàng r nằm ở <code>t·stride + row</code>, và không còn cần <code>rowPtrs</code>.
            Sách viết <code>numRows</code> ở chỗ khóa học này viết <code>stride</code>; hai giá trị bằng nhau trừ khi thêm nguyên các hàng
            đệm để căn lề, như bên dưới.
          </p>
        }
      >
        <p>
          ELL (named after the ELLPACK package) fixes the reads in two moves. Pad every row at its end to the length of the longest row,
          so that the non-zeros form a rectangle. Then store the rectangle column by column: every row's element 0, then every row's
          element 1, and so on. Element t of row r lands at <code>t·stride + row</code>, and <code>rowPtrs</code> is no longer needed.
          The book writes <code>numRows</code> where this course writes <code>stride</code>; the two are equal unless whole rows of
          pads are added for alignment, as below.
        </p>
      </Translated>
      <Source section="17.4" pages="410–411" figs={tr('Fig. 17.10', 'Hình 17.10')} />
      <Translated vi={<p>Hoạt hình bên dưới chạy CSR trên ví dụ xuyên suốt, rồi dựng ELL của nó và chạy lại chính bốn luồng ấy.</p>}>
        <p>The animation below runs CSR on the running example, then builds its ELL and runs the same four threads again.</p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Bốn luồng trong một nhóm, như các hình của sách vẽ. Nó cho thấy mỗi luồng đọc vị trí nào và lane nào
              phải chờ, chứ không cho thấy khi nào: không có cache, không có thời gian.
            </p>
          }
        >
          <p>
            Four threads in one group, as the book's figures draw them. It shows which slot each thread reads and which lanes
            wait, not when: no caches, no timing.
          </p>
        </Translated>
      </Simplification>
      <EllScene />
      <Translated
        vi={
          <p>
            Kernel là kernel của CSR với chỉ số được thay đổi. Vì <code>i</code> là hàm của <code>row</code>, và <code>row</code> là hàm của{' '}
            <code>threadIdx.x</code>, các luồng liên tiếp chạm vào các vị trí liên tiếp ở mọi lần lặp. (Trang 413 nói chỉ số này "was
            calculated in Fig. 17.9", tức đã được tính trong Hình 17.9; thực ra nó ở Hình 17.12.) <Open tab="ell">Mở thẻ ELL</Open>
          </p>
        }
      >
        <p>
          The kernel is CSR's with the index changed. Because <code>i</code> is a function of <code>row</code>, and <code>row</code> of{' '}
          <code>threadIdx.x</code>, consecutive threads touch consecutive slots in every iteration. (Page 413 says this index "was
          calculated in Fig. 17.9"; it is in Fig. 17.12.) <Open tab="ell">Open the ELL tab</Open>
        </p>
      </Translated>
      <Source section="17.4" pages="411–413" figs={tr('Figs. 17.11–17.12', 'Hình 17.11–17.12')} />
      <Ask q={questionsS172['ch17/17.2/q-ell-index']} label={tr('Work it out', 'Tự tính ra')} />
      <Snippet
        file={SNIPPETS.ell}
        marks={[ELL_INDEX_LINE, ELL_ACCUMULATE_LINE, ELL_PAD_TEST_LINE]}
        caption={
          <Translated
            vi={
              <>
                Hình 17.12 từ <code>labs/ch17/snippets/spmv_ell.cu</code>, rồi cùng kernel đó lặp qua toàn bộ độ rộng. Nó khác
                trang sách ở ba điểm: <code>y[row] += sum</code> ở chỗ sách dùng phép gán; phần đệm mang cột −1 và giá trị 0, nên{' '}
                <code>colIdx</code> và <code>col</code> là kiểu có dấu; và <code>stride</code> thay cho <code>numRows</code>. Đã chạy trên
                trình giả lập, kể cả với <code>stride</code> làm tròn lên bội của 8.
              </>
            }
          >
            <>
              Fig. 17.12 from <code>labs/ch17/snippets/spmv_ell.cu</code>, then the same kernel looping the full width. It differs from
              the page in three ways: <code>y[row] += sum</code> where the page assigns; pads hold column −1 and value 0, so{' '}
              <code>colIdx</code> and <code>col</code> are signed; and <code>stride</code> for <code>numRows</code>. Run on the
              emulator, including with <code>stride</code> rounded up to a multiple of 8.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Hai trong số những khác biệt đó sửa những vấn đề thật. Hình 17.12 kết thúc bằng <code>{'y[row] = sum'}</code> ở chỗ Hình 17.9 có{' '}
            <code>+=</code>; SpMV cộng dồn, và với dòng của sách, bất cứ thứ gì có trong <code>y</code> đều bị mất. Và sách nói một
            kernel không có <code>nnzPerRow</code> có thể lặp qua toàn bộ độ rộng đã đệm, vì giá trị đệm bằng 0, nhưng không bao giờ nói cột
            của một phần đệm chứa gì, dù vòng lặp đó đọc nó (p. 411). Đoạn mã lưu −1 và kiểm tra nó.
          </p>
        }
      >
        <p>
          Two of those differences fix real problems. Fig. 17.12 ends with <code>{'y[row] = sum'}</code> where Fig. 17.9 has{' '}
          <code>+=</code>; SpMV accumulates, and with the page's line whatever was in <code>y</code> is lost. And the book says a
          kernel without <code>nnzPerRow</code> can loop over the full padded width, because pad values are zero, but never says what a
          pad's column holds, though that loop reads it (p. 411). The snippet stores −1 and tests it.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch17-facts.md, A6, A10)', '2026-11-20 (notes/today/ch17-facts.md, mục A6, A10)')}>
        <Translated
          vi={
            <p>
              Cả hai quy ước đều là của các thư viện. Sliced ELL của NVIDIA và ELL của AMD lưu cột −1 trong phần đệm, và vòng lặp tham chiếu của AMD
              kiểm tra chỉ số trước khi đọc <code>x</code>. Mọi SpMV của thư viện đều tính y = α·A·x + β·y, một phép cộng dồn.
            </p>
          }
        >
          <p>
            Both conventions are the libraries'. NVIDIA's Sliced ELL and AMD's ELL store column −1 in a pad, and AMD's reference loop
            tests the index before reading <code>x</code>. Every library SpMV computes y = α·A·x + β·y, an accumulate.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS172['ch17/17.2/q-ell-assign']} />
      <Ask q={questionsS172['ch17/17.2/q-pad-marker']} />
      <Translated
        vi={
          <p>
            Vậy một phần đệm trỏ vào cột 0 chỉ vô hại khi <code>x</code> còn hữu hạn: một <code>x[0]</code> vô cùng biến mọi
            hàng có đệm thành NaN, đó chính là khác biệt cuSPARSE mô tả giữa số học dày đặc và số học thưa. Giờ đến cận vòng lặp. Trong
            thẻ ELL của trình khám phá, chạy qua ba lần lặp với vòng lặp dừng ở <code>nnzPerRow</code>, rồi chuyển
            sang toàn bộ độ rộng (<Open tab="ell" extra={{ ellLoop: 'full' }}>toàn bộ độ rộng</Open>) và chạy lại. Các lần đọc vẫn là một dải mỗi
            lần lặp theo cách nào cũng vậy. Điều thay đổi là các lane của hàng ngắn làm gì: chờ, hay đọc phần đệm. ELL sửa được việc hợp nhất, không sửa được
            phân kỳ luồng.
          </p>
        }
      >
        <p>
          So a pad that points at column 0 is harmless only while <code>x</code> stays finite: one infinite <code>x[0]</code> turns every
          padded row into NaN, which is the difference cuSPARSE describes between dense and sparse arithmetic. Now the loop bound. In
          the explorer's ELL tab, step through the three iterations with the loop stopping at <code>nnzPerRow</code>, then switch
          to the full width (<Open tab="ell" extra={{ ellLoop: 'full' }}>full width</Open>) and step again. The reads stay one run per
          iteration either way. What changes is what the short rows' lanes do: wait, or read pads. ELL fixed coalescing, not
          divergence.
        </p>
      </Translated>
      <Source section="17.4" pages="411, 413–414" />
      <Ask q={questionsS172['ch17/17.2/q-ell-loop']} />
      <Translated
        vi={
          <p>
            ELL được nhiều hơn là việc hợp nhất. Một phần tử khác 0 mới có thể chiếm một vị trí đệm mà không phải dời gì, nên nó linh hoạt hơn CSR.
            Nó cho truy cập theo cả hai chiều: các phần tử của hàng r nằm ở r, r + stride và cứ thế, còn vị trí i thuộc về hàng{' '}
            <code>i % stride</code>. Cái giá là không gian, và chỉ riêng hàng dài nhất quyết định nó. Thêm vài phần tử khác 0 vào một hàng của
            ma trận trong trình khám phá và quan sát số phần đệm. Ví dụ của sách: một ma trận 1000 × 1000 với 1% phần tử khác 0 chiếm khoảng 2% kích thước
            dày đặc khi ở dạng CSR; nếu một hàng có 200 phần tử khác 0, ELL đệm mọi hàng lên 200 và chiếm 40%. Trang 17.5 bàn về vấn đề đó.
          </p>
        }
      >
        <p>
          ELL gains more than coalescing. A new non-zero can take a pad slot without moving anything, so it is more flexible than CSR.
          It gives access both ways: row r's elements are at r, r + stride and so on, and slot i belongs to row{' '}
          <code>i % stride</code>. The price is space, and the longest row sets it alone. Add a few non-zeros to one row of the
          explorer's matrix and watch the pad count. The book's example: a 1000 × 1000 matrix with 1% non-zeros takes about 2% of its
          dense size as CSR; if one row has 200 non-zeros, ELL pads every row to 200 and takes 40%. Page 17.5 is about that problem.
        </p>
      </Translated>
      <Source section="17.4" pages="412–413" />
      <Translated
        vi={
          <p>
            Sách thêm một mẹo căn lề (p. 413): nối thêm vài hàng đệm trước khi xoay hình chữ nhật, để mọi lần lặp
            bắt đầu ở một địa chỉ được căn lề. Theo cách nói của khóa học này, hãy làm <code>stride</code> lớn hơn <code>numRows</code>; công tắc stride
            của thẻ ELL làm việc đó (<Open tab="ell" extra={{ aligned: true }}>stride là bội của 8</Open>). Khi đó Hình 17.1 chiếm 24 vị trí thay vì
            12: lần lặp t bắt đầu ở vị trí 8t, và các hàng 4 đến 7 chỉ chứa phần đệm, không luồng nào ghé qua vì phép kiểm tra biên vẫn là{' '}
            <code>row &lt; numRows</code>. Các hàng đệm là cái giá; thứ chúng mua được là lần đọc đầu tiên của mọi lần lặp bắt đầu một
            đoạn mới đã căn lề thay vì nằm vắt qua một đoạn.
          </p>
        }
      >
        <p>
          The book adds an alignment trick (p. 413): append a few rows of pads before turning the rectangle, so that every iteration
          starts on an aligned address. In this course's terms, make <code>stride</code> larger than <code>numRows</code>; the ELL tab's stride
          switch does it (<Open tab="ell" extra={{ aligned: true }}>stride a multiple of 8</Open>). Fig. 17.1 then takes 24 slots instead
          of 12: iteration t starts at slot 8t, and rows 4 to 7 hold only pads, which no thread visits because the bounds check is still{' '}
          <code>row &lt; numRows</code>. The pad rows are the price; what they buy is that every iteration's first read begins a new
          aligned segment instead of straddling one.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch17-facts.md, A3)', '2026-11-20 (notes/today/ch17-facts.md, mục A3)')}>
        <Translated
          vi={
            <p>
              Đơn vị ví dụ của sách, "such as 64-bytes" (chẳng hạn 64 byte), không có trong tài liệu NVIDIA nào được đọc hôm nay. Trên mọi thiết bị mà CUDA hiện tại hỗ trợ,
              đơn vị là một sector 32 byte, nên mọi lần lặp ELL bắt đầu đã căn lề khi <code>stride</code> là bội của 8 số float;
              các dòng 64 hoặc 128 byte của AMD cần 16 hoặc 32. Apple không công bố đơn vị nào.
            </p>
          }
        >
          <p>
            The book's example unit, "such as 64-bytes", is in no NVIDIA document read today. On every device current CUDA supports
            the unit is a 32-byte sector, so every ELL iteration starts aligned when <code>stride</code> is a multiple of 8 floats;
            AMD's 64- or 128-byte lines need 16 or 32. Apple documents no unit.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Nếu bạn đã dùng một attention mask thưa theo khối, bạn đã dùng cách bố trí này. Một mask như vậy trên các khối query và key phần lớn là
            trống, và chương trình xử lý một hàng khối query phải biết cần ghé những khối key nào. Hãy lưu, cho mỗi hàng, một
            số đếm và một danh sách chỉ số khối có độ rộng cố định, được đệm ở chỗ hàng ngắn, và mỗi chương trình tìm thấy phần việc của hàng mình ở một
            độ lệch cố định mà không cần tìm kiếm. Đó là <code>nnzPerRow</code> cộng một hàng đã đệm có độ rộng cố định, khả năng truy cập của ELL (p. 413);
            lưu theo thứ tự hàng (row-major), vì một chương trình đi qua một hàng, nên lập luận về hợp nhất truy cập của sách không áp dụng được.
          </p>
        }
      >
        <p>
          If you have used a block-sparse attention mask, you have used this layout. Such a mask over query and key blocks is mostly
          empty, and the program that handles one row of query blocks has to know which key blocks to visit. Store, for every row, a
          count and a fixed-width list of block indices, padded where the row is short, and each program finds its row's work at a
          fixed offset with no search. That is <code>nnzPerRow</code> plus a fixed-width padded row, ELL's accessibility (p. 413);
          stored row-major, because one program walks one row, so the book's coalescing argument does not carry over.
        </p>
      </Translated>
      <Beyond title={tr('Not in the book: sparse attention masks', 'Không có trong sách: attention mask thưa')}>
        <Translated
          vi={
            <p>
              <code>BlockMask</code> của FlexAttention trong PyTorch là <code>kv_num_blocks</code> cộng một{' '}
              <code>kv_indices[ROWS, MAX]</code> đã đệm, theo thứ tự hàng vì một chương trình đi qua một hàng; lượt lan truyền ngược của nó sinh ra một bản sao
              chuyển vị, <code>q_indices</code> (bản sao theo cột của p. 409). Sparse flash attention cho CUDA và HIP của llama.cpp dựng
              các danh sách chỉ số có độ rộng cố định được đệm bằng −1, cộng các số đếm. Đọc ngày 2026-11-20 trong <code>torch/nn/attention/flex_attention.py</code> và{' '}
              <code>ggml/src/ggml-cuda/fattn.cu</code> (facts D3, D4); phép ánh xạ sang các định dạng của sách là của khóa học này.
            </p>
          }
        >
          <p>
            PyTorch's FlexAttention <code>BlockMask</code> is <code>kv_num_blocks</code> plus a padded{' '}
            <code>kv_indices[ROWS, MAX]</code>, row-major because one program walks one row; its backward pass generates a transposed
            copy, <code>q_indices</code> (p. 409's column-oriented copy). llama.cpp's CUDA and HIP sparse flash attention builds
            fixed-width index lists padded with −1, plus counts. Read 2026-11-20 in <code>torch/nn/attention/flex_attention.py</code> and{' '}
            <code>ggml/src/ggml-cuda/fattn.cu</code> (facts D3, D4); the mapping to the book's formats is ours.
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="CSC: cùng ý tưởng, theo cột">CSC: the same idea, by column</Translated>
      </h2>
      <Translated
        vi={
          <p>
            CSC là CSR với hàng và cột đổi chỗ: <code>value</code> được nhóm theo cột, <code>rowIdx</code> cho biết hàng của từng giá trị,
            và <code>colPtrs</code> với một mục mỗi cột cộng mục đánh dấu kết thúc.
          </p>
        }
      >
        <p>
          CSC is CSR with rows and columns swapped: <code>value</code> grouped by column, <code>rowIdx</code> giving each value's row,
          and <code>colPtrs</code> with one entry per column plus the end marker.
        </p>
      </Translated>
      <Source section="17.7" pages="418–419" figs={tr('Fig. 17.16', 'Hình 17.16')} />
      <Translated
        vi={
          <p>
            Sách đưa kernel SpMV của nó cho đầy đủ, và nó dạy bằng chính sự tệ của mình. Một luồng mỗi cột nạp{' '}
            <code>x[col]</code> một lần, rồi đi qua cột của nó và cộng mỗi tích vào <code>y[row]</code>. Các cột khác nhau đụng vào
            cùng các hàng, nên phép cộng lại phải là atomic. Các luồng 0 đến 3 đọc <code>value[0]</code>, <code>[2]</code>, <code>[4]</code>,{' '}
            <code>[6]</code>, rồi <code>[1]</code>, <code>[3]</code>, <code>[5]</code>, <code>[7]</code>: rải rác, như ở CSR. Số vòng
            lặp theo độ dài cột, nên nó phân kỳ. Nó gộp cái tệ nhất của COO và của CSR.{' '}
            <Open tab="csc">Mở thẻ CSC</Open>
          </p>
        }
      >
        <p>
          The book shows its SpMV kernel for completeness, and it teaches by being bad. One thread per column loads{' '}
          <code>x[col]</code> once, then walks its column and adds each product into <code>y[row]</code>. Different columns hit the
          same rows, so the add is atomic again. Threads 0 to 3 read <code>value[0]</code>, <code>[2]</code>, <code>[4]</code>,{' '}
          <code>[6]</code>, then <code>[1]</code>, <code>[3]</code>, <code>[5]</code>, <code>[7]</code>: scattered, as in CSR. Trip
          counts follow column lengths, so it diverges. It combines the worst of COO and of CSR.{' '}
          <Open tab="csc">Open the CSC tab</Open>
        </p>
      </Translated>
      <Source section="17.7" pages="419–421" figs={tr('Figs. 17.17–17.18', 'Hình 17.17–17.18')} />
      <Translated
        vi={
          <p>
            Bản thân kernel là vòng lặp của CSR trên một cột, với phép cộng nguyên tử của COO. Có hai chỗ sai trên trang của nó: đoạn mã đọc{' '}
            <code>rowIdxs</code> và <code>values</code> ở chỗ phần chữ và Hình 17.16 nói <code>rowIdx</code> và <code>value</code>,
            và chú thích của nó gọi nó là kernel SpMV/CSR. Một bản đã sửa, <code>labs/ch17/snippets/spmv_csc.cu</code>, chạy trên
            trình giả lập cùng ba bản kia.
          </p>
        }
      >
        <p>
          The kernel itself is CSR's loop over a column, with COO's atomic add. Two slips on its page: the listing reads{' '}
          <code>rowIdxs</code> and <code>values</code> where the text and Fig. 17.16 say <code>rowIdx</code> and <code>value</code>,
          and its caption calls it an SpMV/CSR kernel. A corrected copy, <code>labs/ch17/snippets/spmv_csc.cu</code>, runs on the
          emulator with the other three.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Nó có một ưu điểm. Các luồng kề nhau đọc các phần tử kề nhau của <code>x</code>, mỗi phần tử đúng một lần, trong khi mọi kernel khác ở đây
            đọc <code>x</code> ở bất cứ đâu cột của các phần tử khác 0 trỏ tới. Trong thẻ CSC, nhãn "lần đọc x" cho thấy một dải ở t = 0 và
            không gì sau đó.
          </p>
        }
      >
        <p>
          It has one virtue. Adjacent threads read adjacent elements of <code>x</code>, each exactly once, where every other kernel here
          reads <code>x</code> wherever the non-zeros' columns point. In the CSC tab, the "x reads" chip shows one run at t = 0 and
          nothing afterwards.
        </p>
      </Translated>
      <Source section="17.7" pages="421" />
      <Translated
        vi={
          <p>
            CSC hợp với mọi chỗ mà nguyên một cột là đơn vị công việc: <code>xᵀ·A</code>, trong đó mỗi đầu ra là tích vô hướng của một cột, và một
            <code>x</code> thưa, trong đó mọi cột có mục <code>x</code> bằng 0 đều có thể bỏ qua nguyên cả cột. Nó cũng là "second,
            column-oriented copy" (bản sao thứ hai, theo cột) mà p. 409 gợi ý giữ bên cạnh CSR khi một phép tính cần cả hai chiều.
          </p>
        }
      >
        <p>
          CSC fits wherever a whole column is the unit of work: <code>xᵀ·A</code>, where each output is a column's dot product, and a
          sparse <code>x</code>, where every column whose <code>x</code> entry is zero can be skipped whole. It is also the "second,
          column-oriented copy" p. 409 suggests keeping beside CSR when a computation needs both directions.
        </p>
      </Translated>
      <Source section="17.3, 17.7" pages="409, 421" />
      <Ask q={questionsS172['ch17/17.2/q-csc-use']} />

      <h2>
        <Translated vi="Cùng các kernel ấy trên các backend khác">The same kernels on other backends</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Các kernel mỗi-hàng-một-luồng không cần gì ngoài chỉ số luồng, nên chúng chạy được ở mọi nơi. Các kernel dùng atomic cần một phép cộng nguyên tử số thực,
            mà mỗi backend viết khác nhau và sắp thứ tự khác nhau. <a href={href('ch17', 'labs')}>Trang lab</a> có
            bảng, từng backend một.
          </p>
        }
      >
        <p>
          The per-row kernels need nothing beyond a thread index, so they port everywhere. The atomic kernels need a float atomic add,
          which each backend spells differently and orders differently. The <a href={href('ch17', 'labs')}>lab page</a> has the
          table, backend by backend.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Sách chỉ dùng CUDA. Không thứ gì trên trang này đã chạy trên GPU.</p>}>
          <p>The book uses CUDA only. Nothing on this page has run on a GPU.</p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Vậy thì định dạng nào">Which format, then</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bảng bên dưới là phần so sánh của chương, tính cho bất kỳ ma trận nào trình khám phá đang giữ. Hãy đọc nó khi đã nạp Hình 17.1,
            rồi với Hình 17.13, có hai hàng dài là điểm xuất phát của trang 17.5. Không hàng nào thắng ở mọi cột. COO mua sự linh hoạt và
            cân bằng bằng atomic. CSR mua không gian và quyền sở hữu <code>y</code> bằng các vòng lặp rải rác, không đều. ELL mua các lần đọc liền mạch
            bằng phần đệm, và giữ các vòng lặp không đều của CSR trừ khi nó trả giá cho toàn bộ độ rộng.
          </p>
        }
      >
        <p>
          The table below is the chapter's comparison, computed for whatever matrix the explorer holds. Read it with Fig. 17.1 loaded,
          then with Fig. 17.13, whose two long rows are page 17.5's starting point. No row wins every column. COO buys flexibility and
          balance with atomics. CSR buys space and ownership of <code>y</code> with scattered, uneven loops. ELL buys contiguous reads
          with pads, and keeps CSR's uneven loops unless it pays for the full width.
        </p>
      </Translated>
      <Source section="17.2–17.7" pages="405–421" />
      <ScoreTable st={st} set={set} />
      <Translated
        vi={
          <p>
            Định dạng nào nhanh nhất phụ thuộc vào cách các phần tử khác 0 phân bố, điều sách gọi là hiệu năng phụ thuộc dữ liệu, và mã thưa
            đạt ít FLOPS hơn nhiều so với mã dày đặc, trên CPU cũng như GPU. <a href={href('ch17', 'labs')}>Lab 17.1</a>, SpMV với CSR và ELL
            trên các hàng lệch, đo CSR so với ELL trên thiết bị của bạn. Hãy đưa ra dự đoán của nó trước khi chạy.
          </p>
        }
      >
        <p>
          Which one is fastest depends on how the non-zeros are spread, which the book calls data-dependent performance, and sparse code
          reaches far fewer FLOPS than dense code, on CPUs and GPUs alike. <a href={href('ch17', 'labs')}>Lab 17.1</a>, CSR and ELL
          SpMV on skewed rows, measures CSR against ELL on your device. Make its prediction before you run it.
        </p>
      </Translated>
      <Source section="17.8" pages="421–422" />
      <Ask q={questionsS172['ch17/17.2/q-ex1-pads']} label={tr('Work it out', 'Tự tính ra')} />
      <Ask q={questionsS172['ch17/17.2/q-ex2-missing']} />
    </>
  );
}
