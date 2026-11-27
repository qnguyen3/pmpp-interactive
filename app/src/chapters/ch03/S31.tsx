import { useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Code, useLabFile } from '../../components/Code';
import { Translated, tr } from '../../lib/i18n';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  AXES,
  BOOK_RULE,
  CUDA_DOCUMENTED,
  FIG31,
  FORMULAS,
  LAUNCH_ERROR,
  blockCount,
  checkLaunch,
  coverage,
  describeViolation,
  dim3,
  drillTargets,
  formatBlockLabel,
  formatDim3,
  formatThreadLabel,
  inside,
  isDim3,
  limitsFromReport,
  linearThread,
  linearThreadWorking,
  misread,
  sameIndex,
  sizeGrid,
  sizeGridMinusOne,
  threadFromLabel,
  threadsPerBlock,
  totalThreads,
  workingInBlock,
  type Dim3,
  type Limits,
  type Target,
  type Unchecked,
} from '../../sim/grid';

// ---------------------------------------------------------------- small helpers
const num = (v: number | bigint) => v.toLocaleString('en-US');
const plural = (n: number | bigint, one: string, many = one + 's') => `${num(n)} ${Number(n) === 1 ? one : many}`;
const ORIGIN: Dim3 = { x: 0, y: 0, z: 0 };
/** A caption above a group of controls (a <label> would forward clicks on the caption to the first control). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };

/** A limit as the lesson writes it: 2^31 - 1 stays symbolic. */
function Cap({ v }: { v: number }) {
  return v === 2 ** 31 - 1 ? (
    <span className="nowrap">
      2<sup>31</sup> − 1
    </span>
  ) : (
    <>{num(v)}</>
  );
}

const CODE_TIERS: Record<string, Tier> = {
  out: 'global', out_d: 'global', A_d: 'global', B_d: 'global', C_d: 'global', A: 'global', B: 'global', C: 'global',
  threadsPerBlock: 'reg', block: 'reg', thread: 'reg', i: 'reg', n: 'reg', err: 'reg',
};

// ---------------------------------------------------------------- limit sets the reader can judge by
type Kind = 'book' | 'cuda-doc' | 'cuda' | 'hip' | 'emu' | 'metal' | 'other';
interface Choice {
  key: string;
  kind: Kind;
  label: string;
  limits: Limits;
  note: string;
}
const BOOK_CHOICE: Choice = {
  key: 'book',
  kind: 'book',
  get label() {
    return tr("The book's rule (p. 47)", 'Quy tắc của sách (p. 47)');
  },
  limits: BOOK_RULE,
  get note() {
    return tr(
      "The book's rule (section 3.1, p. 47): at most 1024 threads in a block, spread over its dimensions in any way; gridDim.x up to 2^31 − 1; gridDim.y and gridDim.z up to 65,535.",
      'Quy tắc của sách (mục 3.1, p. 47): tối đa 1024 luồng trong một khối, phân bố trên các chiều của nó theo cách nào cũng được; gridDim.x tới 2^31 − 1; gridDim.y và gridDim.z tới 65,535.',
    );
  },
};
const CUDA_CHOICE: Choice = {
  key: 'cuda-doc',
  kind: 'cuda-doc',
  get label() {
    return tr("CUDA's documented limits", 'Giới hạn được tài liệu hóa của CUDA');
  },
  limits: CUDA_DOCUMENTED,
  get note() {
    return tr(
      "NVIDIA's documented limits, the same for every compute capability its table lists (checked 2026-11-20). Documented values, not read from a device.",
      'Giới hạn theo tài liệu của NVIDIA, như nhau cho mọi compute capability mà bảng của họ liệt kê (kiểm tra ngày 2026-11-20). Đây là giá trị theo tài liệu, không đọc từ một thiết bị.',
    );
  },
};

/** Limit sets from the reader's own device reports. */
function reported(reports: Reports): Choice[] {
  // Real devices first, then the emulator once (its two flavours report the same numbers).
  const emu = (b: string) => b === 'emu' || b === 'race';
  const keys = Object.keys(reports).sort((a, b) => Number(emu(a)) - Number(emu(b)) || a.localeCompare(b));
  return keys
    .filter((b) => b !== 'race' || !('emu' in reports))
    .flatMap((b): Choice[] => {
      const limits = limitsFromReport(reports[b].device);
      if (!limits) return [];
      const name = String(reports[b].device.name ?? b);
      const when = reports[b].generated.slice(0, 10);
      if (emu(b))
        return [
          {
            key: b,
            kind: 'emu',
            get label() {
              return tr("The course's CPU emulator (not a GPU)", 'Trình giả lập CPU của khóa học (không phải GPU)');
            },
            limits,
            get note() {
              return tr(
                'The numbers the course emulator reports about itself. It is not a GPU; it copies the limits in NVIDIA\'s table and enforces them.',
                'Những con số trình giả lập của khóa học báo về chính nó. Nó không phải GPU; nó chép các giới hạn trong bảng của NVIDIA và áp dụng chúng.',
              );
            },
          },
        ];
      if (b === 'metal')
        return [
          {
            key: b,
            kind: 'metal',
            get label() {
              return tr(`Yours: ${name} (metal)`, `Của bạn: ${name} (metal)`);
            },
            limits,
            get note() {
              return tr(
                `What this Metal device reported when ./verify.sh ran on ${when}: the most threads along each dimension of a threadgroup, which Apple says holds for a trivial kernel only.`,
                `Những gì thiết bị Metal này báo khi ./verify.sh chạy ngày ${when}: số luồng tối đa theo mỗi chiều của một threadgroup, mà Apple nói chỉ đúng với một kernel tầm thường.`,
              );
            },
          },
        ];
      return [
        {
          key: b,
          kind: b === 'cuda' ? 'cuda' : b === 'hip' ? 'hip' : 'other',
          get label() {
            return tr(`Yours: ${name} (${b})`, `Của bạn: ${name} (${b})`);
          },
          limits,
          get note() {
            return tr(
              `What this device reported in its device properties when ./verify.sh ran on ${when}.`,
              `Những gì thiết bị này báo trong thuộc tính thiết bị của nó khi ./verify.sh chạy ngày ${when}.`,
            );
          },
        },
      ];
    });
}

function uncheckedNote(u: Unchecked, kind: Kind): string {
  if (kind === 'book' && u === 'block-dim')
    return tr('The book gives no cap for a single dimension of a block, so none is tested.', 'Sách không cho giới hạn nào cho riêng một chiều của khối, nên không kiểm tra giới hạn nào.');
  if (kind === 'metal' && u === 'block-threads')
    return tr(
      'The total per threadgroup is not in a device record. Metal sets it for each compiled kernel (maxTotalThreadsPerThreadgroup).',
      'Tổng số luồng mỗi threadgroup không có trong bản ghi thiết bị. Metal đặt nó cho từng kernel đã biên dịch (maxTotalThreadsPerThreadgroup).',
    );
  if (kind === 'metal' && u === 'grid-dim')
    return tr('Apple documents no limit on the size of a grid, so none is tested.', 'Apple không ghi giới hạn nào cho kích thước lưới, nên không kiểm tra giới hạn nào.');
  return tr(
    `This record has no figure for ${u === 'block-threads' ? 'the threads in a block' : u === 'block-dim' ? 'a single dimension of a block' : 'the size of the grid'}, so it is not tested.`,
    `Bản ghi này không có số liệu cho ${u === 'block-threads' ? 'số luồng trong một khối' : u === 'block-dim' ? 'riêng một chiều của khối' : 'kích thước lưới'}, nên không kiểm tra.`,
  );
}

// What the course emulator itself does with a launch (labs/common/include/pmpp/emu/emu_runtime.h,
// launch_impl; checked by labs/ch03/snippets/check.py): it rejects a block of more than 1024
// threads, a block or grid dimension over NVIDIA's per-dimension caps, a zero dimension, and
// dynamic shared memory over its limit, all with cudaErrorInvalidConfiguration.
const EMU_ENFORCES = () =>
  tr(
    "The emulator enforces the limits it reports, which are NVIDIA's documented ones. It also rejects a zero dimension and too much dynamic shared memory.",
    'Trình giả lập áp dụng các giới hạn mà nó báo, tức là các giới hạn theo tài liệu của NVIDIA. Nó cũng từ chối một chiều bằng 0 và quá nhiều bộ nhớ chia sẻ động.',
  );

function errorNote(kind: Kind): string {
  const e = kind === 'hip' ? LAUNCH_ERROR.hip : LAUNCH_ERROR.cuda;
  switch (kind) {
    case 'book':
      return tr(
        'The book stops here. What a device does with such a launch is the subject of "What happens when you break a limit", further down this page.',
        'Sách dừng ở đây. Một thiết bị làm gì với lần khởi chạy như vậy là chủ đề của "Điều gì xảy ra khi bạn vượt giới hạn", ở phía dưới trang này.',
      );
    case 'emu':
      return tr(
        `The emulator rejects this launch: it sets ${e.name} (${e.value}) and prints a line. ${EMU_ENFORCES()}`,
        `Trình giả lập từ chối lần khởi chạy này: nó đặt ${e.name} (${e.value}) và in ra một dòng. ${EMU_ENFORCES()}`,
      );
    case 'metal':
      return tr(
        'Apple documents no error for a threadgroup over its limit. Do not rely on any particular behaviour.',
        'Apple không ghi lỗi nào cho một threadgroup vượt giới hạn. Đừng dựa vào bất kỳ hành vi cụ thể nào.',
      );
    default:
      return tr(
        `By the documentation, a launch like this fails and leaves ${e.name} (${e.value}) in the runtime's error state. The launch statement returns nothing, so ${e.query} is the only way to see it.`,
        `Theo tài liệu, một lần khởi chạy như thế này thất bại và để lại ${e.name} (${e.value}) trong trạng thái lỗi của runtime. Câu lệnh khởi chạy không trả về gì, nên ${e.query} là cách duy nhất để thấy nó.`,
      );
  }
}

// ---------------------------------------------------------------- the grid explorer
const DRAW = { blocks: { x: 8, y: 6, z: 3 }, threads: { x: 32, y: 16, z: 4 } } as const;

const cellStyle = (w: number, h: number): CSSProperties => ({
  width: w,
  height: h,
  padding: 0,
  borderColor: 'var(--rule)',
  borderRadius: 2,
  fontSize: '0.68rem',
  fontWeight: 450,
  fontVariantNumeric: 'tabular-nums',
  lineHeight: 1,
});
const HEAD: CSSProperties = {
  fontSize: '0.64rem',
  color: 'var(--ink-3)',
  textAlign: 'center',
  alignSelf: 'center',
  fontVariantNumeric: 'tabular-nums',
  whiteSpace: 'nowrap',
};

/** A table of cells with x across and y down, and the two index rulers. */
function Lattice({
  nx,
  ny,
  axis,
  label,
  cell,
}: {
  nx: number;
  ny: number;
  /** Name printed in the corner, e.g. "blockIdx" or "threadIdx". */
  axis: string;
  label: string;
  cell: (x: number, y: number) => ReactNode;
}) {
  const xs = Array.from({ length: nx }, (_, i) => i);
  const ys = Array.from({ length: ny }, (_, i) => i);
  return (
    <div
      role="group"
      aria-label={label}
      style={{ display: 'grid', gridTemplateColumns: `max-content repeat(${nx}, max-content)`, gap: 2, justifyContent: 'start' }}
    >
      <span style={{ ...HEAD, textAlign: 'left', paddingRight: 4 }} title={tr(`${axis}.y runs down, ${axis}.x runs across`, `${axis}.y chạy xuống, ${axis}.x chạy ngang`)}>
        y ↓ x →
      </span>
      {xs.map((x) => (
        <span key={x} style={HEAD}>
          {x}
        </span>
      ))}
      {ys.map((y) => [
        <span key={'h' + y} style={{ ...HEAD, paddingRight: 4, textAlign: 'right' }}>
          {y}
        </span>,
        ...xs.map((x) => cell(x, y)),
      ])}
    </div>
  );
}

interface Preset {
  key: string;
  label: string;
  grid?: Dim3;
  block: Dim3;
}
const LAUNCH_PRESETS: Preset[] = [
  {
    key: 'fig31',
    get label() {
      return tr('Fig. 3.1', 'Hình 3.1');
    },
    grid: FIG31.grid,
    block: FIG31.block,
  },
  {
    key: 'p46',
    get label() {
      return tr('32 blocks of 128 threads (p. 46)', '32 khối 128 luồng (p. 46)');
    },
    grid: dim3(32),
    block: dim3(128),
  },
];
const BOOK_SHAPES: Dim3[] = [dim3(512, 1, 1), dim3(8, 16, 4), dim3(32, 16, 2), dim3(32, 32, 2)];
const OUR_SHAPE = dim3(1, 1, 128);
const SHAPE_PRESETS: Preset[] = [...BOOK_SHAPES, OUR_SHAPE].map((b) => ({
  key: `b${b.x}-${b.y}-${b.z}`,
  label: formatDim3(b),
  block: b,
}));

type Triple = Record<'x' | 'y' | 'z', string>;
const toText = (d: Dim3): Triple => ({ x: String(d.x), y: String(d.y), z: String(d.z) });
const toDim = (t: Triple): Dim3 => {
  const one = (s: string) => (s.trim() === '' ? NaN : Number(s));
  return { x: one(t.x), y: one(t.y), z: one(t.z) };
};

function GridExplorer() {
  const { reports, loading } = useReports();
  const choices = useMemo(() => [BOOK_CHOICE, CUDA_CHOICE, ...reported(reports)], [reports]);
  const [key, setKey] = useState('book');
  const [g, setG] = useState<Triple>(toText(FIG31.grid));
  const [b, setB] = useState<Triple>(toText(FIG31.block));
  const [selBlock, setSelBlock] = useState<Dim3>({ x: 0, y: 1, z: 0 });
  const [selThread, setSelThread] = useState<Dim3 | null>(null);

  const choice = choices.find((c) => c.key === key) ?? BOOK_CHOICE;
  const grid = toDim(g);
  const block = toDim(b);
  const valid = isDim3(grid) && isDim3(block);
  const drawable = valid && AXES.every((a) => grid[a] >= 1 && block[a] >= 1);
  const verdict = valid ? checkLaunch(grid, block, choice.limits) : null;
  const sb = drawable && inside(selBlock, grid) ? selBlock : ORIGIN;
  const st = drawable && selThread && inside(selThread, block) ? selThread : null;

  const apply = (p: Preset) => {
    if (p.grid) setG(toText(p.grid));
    setB(toText(p.block));
    setSelThread(null);
  };
  const active = (p: Preset) => valid && sameIndex(block, p.block) && (!p.grid || sameIndex(grid, p.grid));

  const fields = (name: string, t: Triple, set: (t: Triple) => void) => (
    <span style={FIELD}>
      <span>
        <code>{name}</code> (x, y, z)
      </span>
      <span className="row" style={{ gap: '0.3rem', flexWrap: 'nowrap' }}>
        {AXES.map((a) => (
          <input
            key={a}
            type="number"
            min="0"
            step="1"
            value={t[a]}
            aria-label={`${name}.${a}`}
            style={{ width: '5.4rem' }}
            onChange={(e) => {
              set({ ...t, [a]: e.target.value });
              setSelThread(null);
            }}
          />
        ))}
      </span>
    </span>
  );

  const shown = {
    bx: drawable ? Math.min(grid.x, DRAW.blocks.x) : 0,
    by: drawable ? Math.min(grid.y, DRAW.blocks.y) : 0,
    bz: drawable ? Math.min(grid.z, DRAW.blocks.z) : 0,
    tx: drawable ? Math.min(block.x, DRAW.threads.x) : 0,
    ty: drawable ? Math.min(block.y, DRAW.threads.y) : 0,
    tz: drawable ? Math.min(block.z, DRAW.threads.z) : 0,
  };
  const blocksCut = drawable && (shown.bx < grid.x || shown.by < grid.y || shown.bz < grid.z);
  const threadsCut = drawable && (shown.tx < block.x || shown.ty < block.y || shown.tz < block.z);
  const layers = (n: number) => Array.from({ length: n }, (_, i) => i);
  const perBlock = valid ? threadsPerBlock(block) : 0n;

  return (
    <Bench
      title={tr('Grid explorer', 'Trình khám phá lưới')}
      task={tr('Pick a block, then one of its threads. Compare the three ways the page names it.', 'Chọn một khối, rồi một luồng của nó. So sánh ba cách trang này gọi tên nó.')}
      foot={<>{choice.note}</>}
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        {fields('gridDim', g, setG)}
        {fields('blockDim', b, setB)}
        <label className="field">
          <Translated vi="Đánh giá lần khởi chạy theo">Judge the launch by</Translated>
          <select value={choice.key} onChange={(e) => setKey(e.target.value)} aria-label={tr('Judge the launch by', 'Đánh giá lần khởi chạy theo')}>
            {choices.map((c) => (
              <option key={c.key} value={c.key}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.5rem' }} role="group" aria-label={tr('Launch presets', 'Cấu hình khởi chạy mẫu')}>
        <span className="muted"><Translated vi="Lần khởi chạy">Launches</Translated></span>
        {LAUNCH_PRESETS.map((p) => (
          <button key={p.key} className="quiet" data-preset={p.key} aria-pressed={active(p)} onClick={() => apply(p)}>
            {p.label}
          </button>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '1rem' }} role="group" aria-label={tr('Block shape presets', 'Hình dạng khối mẫu')}>
        <span className="muted"><Translated vi="Hình dạng khối">Block shapes</Translated></span>
        {SHAPE_PRESETS.map((p) => (
          <button key={p.key} className="quiet" data-preset={p.key} aria-pressed={active(p)} onClick={() => apply(p)}>
            {p.label}
          </button>
        ))}
        <span className="muted"><Translated vi="bốn cái đầu là của sách (p. 47), cái cuối là của khóa học này">the first four are the book's (p. 47), the last is ours</Translated></span>
      </div>
      {!loading && choices.length === 2 ? (
        <p className="muted" style={{ margin: '0 0 0.8rem' }}>
          <Translated
            vi={
              <>
                Máy này chưa có báo cáo thiết bị nào, nên danh sách đưa ra quy tắc của sách và giới hạn theo tài liệu của CUDA. Hãy
                chạy <code>./verify.sh</code> để thêm các thiết bị của chính bạn.
              </>
            }
          >
            No device report on this machine yet, so the list offers the book's rule and CUDA's documented limits. Run{' '}
            <code>./verify.sh</code> to add your own devices.
          </Translated>
        </p>
      ) : null}

      <div className="cols" style={{ marginBottom: '1.2rem' }}>
        <div className="panel">
          <p style={{ margin: '0 0 0.4rem' }}>
            <b><Translated vi="Lần khởi chạy bạn đang mô tả">The launch you are describing</Translated></b>
          </p>
          <pre className="mono" data-out="statement" style={{ margin: '0 0 0.7rem', fontSize: '0.78rem', lineHeight: 1.5, overflowX: 'auto' }}>
            {`dim3 dimGrid(${g.x || '?'}, ${g.y || '?'}, ${g.z || '?'});\ndim3 dimBlock(${b.x || '?'}, ${b.y || '?'}, ${b.z || '?'});\nKernelFunction<<<dimGrid, dimBlock>>>(...);`}
          </pre>
          {valid ? (
            <p style={{ margin: 0 }} aria-live="polite">
              <span data-out="blocks" data-value={String(blockCount(grid))}>
                {tr(plural(blockCount(grid), 'block'), `${num(blockCount(grid))} khối`)}
              </span>{' '}
              ×{' '}
              <span data-out="per-block" data-value={String(perBlock)}>
                {tr(plural(perBlock, 'thread'), `${num(perBlock)} luồng`)}
              </span>{' '}
              ={' '}
              <b data-out="threads" data-value={String(totalThreads(grid, block))}>
                {tr(plural(totalThreads(grid, block), 'thread'), `${num(totalThreads(grid, block))} luồng`)}
              </b>
              <br />
              <span className="muted"><Translated vi="Một lời gọi kernel tạo ra một lưới. Mọi khối trong đó đều có hình dạng này.">One kernel call creates one grid. Every block in it has this shape.</Translated></span>
            </p>
          ) : (
            <p style={{ margin: 0 }} data-out="invalid">
              <Translated vi="Mỗi ô nhận một số nguyên, từ 0 trở lên, vừa với một số nguyên không dấu 32 bit.">
                Each field takes a whole number, 0 or larger, that fits in an unsigned 32-bit integer.
              </Translated>
            </p>
          )}
        </div>
        <div className="panel" aria-live="polite" data-out="verdict" data-legal={verdict ? String(verdict.legal) : 'none'}>
          <p style={{ margin: '0 0 0.4rem' }}>
            <b><Translated vi="Kết luận">Verdict</Translated></b>{' '}
            <span className="muted">
              <Translated vi={<>theo {choice.label.replace(/^Của bạn: /, '')}</>}>by {choice.label.replace(/^The /, 'the ').replace(/^Yours: /, '')}</Translated>
            </span>
          </p>
          {!verdict ? (
            <p style={{ margin: 0 }} className="muted">
              <Translated vi="Chưa có kết luận cho đến khi cả sáu ô đều có số.">
                No verdict until all six fields hold numbers.
              </Translated>
            </p>
          ) : verdict.legal ? (
            <>
              <p style={{ margin: '0 0 0.4rem' }}>
                <span className="chip ok"><Translated vi="chấp nhận">accepted</Translated></span>{' '}
                <Translated vi="Không điều khoản nào của bộ giới hạn này bị vi phạm.">No clause of this limit set is broken.</Translated>
              </p>
              {verdict.unchecked.map((u) => (
                <p key={u} className="muted" style={{ margin: '0 0 0.3rem' }} data-out="unchecked">
                  {uncheckedNote(u, choice.kind)}
                </p>
              ))}
            </>
          ) : (
            <>
              <p style={{ margin: '0 0 0.4rem' }}>
                <span className="chip fail"><Translated vi="từ chối">rejected</Translated></span>{' '}
                {verdict.violations.length === 1 ? tr('One clause fails.', 'Một điều khoản không đạt.') : tr(`${verdict.violations.length} clauses fail.`, `${verdict.violations.length} điều khoản không đạt.`)}
              </p>
              <ul style={{ margin: '0 0 0.5rem', paddingLeft: '1.2em' }}>
                {verdict.violations.map((v) => (
                  <li key={v.field} data-out="violation" data-clause={v.clause}>
                    {tr(describeViolation(v), v.clause === 'zero' ? `${v.field} là ${num(v.value)}. Mọi chiều phải ít nhất là 1.` : v.clause === 'block-threads' ? `Một khối sẽ chứa ${num(v.value)} luồng. Giới hạn là ${num(v.limit)}.` : v.clause === 'block-dim' ? `${v.field} là ${num(v.value)}. Giới hạn cho chiều đó của một khối là ${num(v.limit)}.` : `${v.field} là ${num(v.value)}. Giới hạn cho chiều đó của một lưới là ${num(v.limit)}.`)}
                  </li>
                ))}
              </ul>
              <p className="muted" style={{ margin: 0 }} data-out="error">
                {errorNote(choice.kind)}
              </p>
            </>
          )}
        </div>
      </div>

      {drawable ? (
        <>
          <div className="panel" style={{ marginBottom: '1.2rem', overflowX: 'auto' }}>
            <p style={{ margin: '0 0 0.6rem' }}>
              <Translated
                vi={
                  <>
                    <b>Lưới.</b> Mỗi nút là một khối, được gọi tên như sách gọi:{' '}
                    {grid.z > 1 ? '(blockIdx.z, blockIdx.y, blockIdx.x)' : '(blockIdx.y, blockIdx.x)'}.
                  </>
                }
              >
                <b>The grid.</b> Each button is a block, named as the book names it:{' '}
                {grid.z > 1 ? '(blockIdx.z, blockIdx.y, blockIdx.x)' : '(blockIdx.y, blockIdx.x)'}.
              </Translated>
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.4rem' }}>
              {layers(shown.bz).map((z) => (
                <div key={z}>
                  {grid.z > 1 ? <div className="mx-label">blockIdx.z = {z}</div> : null}
                  <Lattice
                    nx={shown.bx}
                    ny={shown.by}
                    axis="blockIdx"
                    label={tr(`blocks, layer z ${z}`, `các khối, lớp z ${z}`)}
                    cell={(x, y) => (
                      <button
                        key={`${x}-${y}`}
                        data-block={`${x}-${y}-${z}`}
                        aria-pressed={sameIndex(sb, { x, y, z })}
                        aria-label={tr(formatBlockLabel({ x, y, z }, grid), formatBlockLabel({ x, y, z }, grid).replace('Block', 'Khối'))}
                        style={cellStyle(grid.z > 1 ? 56 : 44, 28)}
                        onClick={() => {
                          setSelBlock({ x, y, z });
                          setSelThread(null);
                        }}
                      >
                        {formatBlockLabel({ x, y, z }, grid).replace('Block ', '')}
                      </button>
                    )}
                  />
                </div>
              ))}
            </div>
            {blocksCut ? (
              <p className="muted" style={{ margin: '0.6rem 0 0' }} data-out="blocks-cut">
                <Translated
                  vi={
                    <>
                      Đã vẽ: {shown.bx} × {shown.by} × {shown.bz} khối đầu tiên (x × y × z) trong {num(grid.x)} × {num(grid.y)} ×{' '}
                      {num(grid.z)}. Phần còn lại trông giống vậy.
                    </>
                  }
                >
                  Drawn: the first {shown.bx} × {shown.by} × {shown.bz} blocks (x × y × z) of {num(grid.x)} × {num(grid.y)} ×{' '}
                  {num(grid.z)}. The rest look the same.
                </Translated>
              </p>
            ) : null}
          </div>

          <div className="cols plot">
            <div className="panel" style={{ overflowX: 'auto' }}>
              <p style={{ margin: '0 0 0.6rem' }} data-out="block-title">
                <Translated
                  vi={
                    <>
                      <b>{formatBlockLabel(sb, grid).replace('Block', 'Khối')}.</b> Theo thứ tự của hàm dựng, <code>blockIdx</code> của nó là{' '}
                      {formatDim3(sb)}. Mỗi nút là một luồng của nó và hiện số thứ tự của luồng trong khối.
                    </>
                  }
                >
                  <b>{formatBlockLabel(sb, grid)}.</b> In constructor order its <code>blockIdx</code> is {formatDim3(sb)}. Each
                  button is one of its threads and shows the thread's number within the block.
                </Translated>
              </p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.2rem' }}>
                {layers(shown.tz).map((z) => (
                  <div key={z}>
                    <div className="mx-label">threadIdx.z = {z}</div>
                    <Lattice
                      nx={shown.tx}
                      ny={shown.ty}
                      axis="threadIdx"
                      label={tr(`threads, layer z ${z}`, `các luồng, lớp z ${z}`)}
                      cell={(x, y) => (
                        <button
                          key={`${x}-${y}`}
                          data-thread={`${x}-${y}-${z}`}
                          aria-pressed={!!st && sameIndex(st, { x, y, z })}
                          aria-label={tr(formatThreadLabel({ x, y, z }), formatThreadLabel({ x, y, z }).replace('Thread', 'Luồng'))}
                          style={cellStyle(30, 24)}
                          onClick={() => setSelThread({ x, y, z })}
                        >
                          {linearThread({ x, y, z }, block)}
                        </button>
                      )}
                    />
                  </div>
                ))}
              </div>
              {threadsCut ? (
                <p className="muted" style={{ margin: '0.6rem 0 0' }} data-out="threads-cut">
                  <Translated
                    vi={
                      <>
                        Đã vẽ: {shown.tx} × {shown.ty} × {shown.tz} luồng đầu tiên (x × y × z) trong {num(block.x)} ×{' '}
                        {num(block.y)} × {num(block.z)}.
                      </>
                    }
                  >
                    Drawn: the first {shown.tx} × {shown.ty} × {shown.tz} threads (x × y × z) of {num(block.x)} ×{' '}
                    {num(block.y)} × {num(block.z)}.
                  </Translated>
                </p>
              ) : null}
            </div>
            <div className="panel" aria-live="polite" data-out="thread">
              {st ? (
                <>
                  <p style={{ margin: '0 0 0.5rem' }}>
                    <b data-out="label">
                      <Translated vi={<>{formatThreadLabel(st).replace('Thread', 'Luồng')} của {formatBlockLabel(sb, grid).replace('Block', 'Khối')}</>}>
                        {formatThreadLabel(st)} of {formatBlockLabel(sb, grid)}
                      </Translated>
                    </b>
                    <br />
                    <span className="muted"><Translated vi="nhãn của sách: chiều cao nhất trước">the book's label: highest dimension first</Translated></span>
                  </p>
                  <table className="data" style={{ margin: '0 0 0.6rem' }}>
                    <thead>
                      <tr>
                        <th><Translated vi="Biến dựng sẵn">Built-in</Translated></th>
                        <th>(x, y, z)</th>
                        <th><Translated vi="Cùng giá trị trong">Same value in</Translated></th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>
                          <code>threadIdx</code>
                        </td>
                        <td className="nowrap" data-out="threadIdx">
                          {formatDim3(st)}
                        </td>
                        <td><Translated vi="một luồng của mỗi khối">one thread of each block</Translated></td>
                      </tr>
                      <tr>
                        <td>
                          <code>blockIdx</code>
                        </td>
                        <td className="nowrap" data-out="blockIdx">
                          {formatDim3(sb)}
                        </td>
                        <td><Translated vi={<>{num(perBlock)} luồng của khối này</>}>the {plural(perBlock, 'thread')} of this block</Translated></td>
                      </tr>
                      <tr>
                        <td>
                          <code>blockDim</code>
                        </td>
                        <td className="nowrap">{formatDim3(block)}</td>
                        <td><Translated vi="mọi luồng của lưới">every thread of the grid</Translated></td>
                      </tr>
                      <tr>
                        <td>
                          <code>gridDim</code>
                        </td>
                        <td className="nowrap">{formatDim3(grid)}</td>
                        <td><Translated vi="mọi luồng của lưới">every thread of the grid</Translated></td>
                      </tr>
                    </tbody>
                  </table>
                  <p style={{ margin: 0 }}>
                    <Translated vi="Số thứ tự trong khối, x thay đổi nhanh nhất:">Number within the block, x fastest:</Translated>
                    <br />
                    <code>x + y·blockDim.x + z·blockDim.x·blockDim.y</code>
                    <br />= <span data-out="linear">{linearThreadWorking(st, block)}</span>
                  </p>
                </>
              ) : (
                <p className="muted" style={{ margin: 0 }}>
                  <Translated
                    vi={
                      <>
                        Bấm vào một luồng để xem bốn biến dựng sẵn của nó, nhãn sách dùng cho nó, và số thứ tự của nó trong khối.
                      </>
                    }
                  >
                    Click a thread to see its four built-in variables, the book's label for it, and its number within the
                    block.
                  </Translated>
                </p>
              )}
            </div>
          </div>
        </>
      ) : valid ? (
        <p className="muted" style={{ margin: 0 }} data-out="nothing">
          <Translated vi="Một chiều bằng 0 mô tả không có khối hay không có luồng nào. Không có gì để vẽ.">
            A dimension of 0 describes no blocks or no threads. There is nothing to draw.
          </Translated>
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- find the thread
const DRILL = FIG31;
const DRILL_COUNT = 3;

function FindThread() {
  const [seed, setSeed] = useState(1);
  const targets = useMemo(() => drillTargets(DRILL.grid, DRILL.block, DRILL_COUNT, seed), [seed]);
  const [at, setAt] = useState(0);
  const [click, setClick] = useState<Target | null>(null);
  const [misses, setMisses] = useState(0);
  const [found, setFound] = useState(0);

  const target = targets[Math.min(at, targets.length - 1)];
  const hit = !!click && sameIndex(click.block, target.block) && sameIndex(click.thread, target.thread);
  const finished = hit && at === targets.length - 1;
  const read = click && !hit ? misread(target, click) : null;

  const press = (t: Target) => {
    if (hit) return;
    setClick(t);
    if (sameIndex(t.block, target.block) && sameIndex(t.thread, target.thread)) setFound(found + 1);
    else setMisses(misses + 1);
  };
  const next = () => {
    setAt(at + 1);
    setClick(null);
  };
  const restart = () => {
    setSeed(seed + 1);
    setAt(0);
    setClick(null);
    setMisses(0);
    setFound(0);
  };

  const cell = (blk: Dim3, thr: Dim3) => {
    const me = { block: blk, thread: thr };
    const isClick = !!click && sameIndex(click.block, blk) && sameIndex(click.thread, thr);
    const style: CSSProperties = { ...cellStyle(30, 26) };
    if (isClick) {
      style.background = hit ? 'var(--ok-wash)' : 'var(--fail-wash)';
      style.borderColor = hit ? 'var(--ok)' : 'var(--fail)';
      style.color = hit ? 'var(--ok)' : 'var(--fail)';
    }
    return (
      <button
        key={`${thr.x}-${thr.y}`}
        data-cell={`${blk.x}-${blk.y}-${blk.z}-${thr.x}-${thr.y}-${thr.z}`}
        data-state={isClick ? (hit ? 'found' : 'missed') : undefined}
        aria-label={tr(`a thread of the block at blockIdx.x ${blk.x}, blockIdx.y ${blk.y}`, `một luồng của khối tại blockIdx.x ${blk.x}, blockIdx.y ${blk.y}`)}
        style={style}
        onClick={() => press(me)}
      >
        {isClick ? (hit ? '✓' : '×') : ''}
      </button>
    );
  };

  const blockXs = Array.from({ length: DRILL.grid.x }, (_, i) => i);
  const blockYs = Array.from({ length: DRILL.grid.y }, (_, i) => i);
  const zs = Array.from({ length: DRILL.block.z }, (_, i) => i);

  return (
    <Bench
      title={tr('Find the thread', 'Tìm luồng')}
      task={tr('The cells carry no names. Use the rulers: x runs across, y runs down, z picks the layer.', 'Các ô không có tên. Hãy dùng thước: x chạy ngang, y chạy xuống, z chọn lớp.')}
      foot={
        <>
          <Translated
            vi={
              <>
                Lần khởi chạy của Fig. 3.1 trong sách: <code>dimGrid(2, 2, 1)</code>, <code>dimBlock(4, 2, 2)</code>. Mục tiêu đầu
                tiên là ví dụ của chính sách (p. 48); các mục tiêu khác được chọn sao cho đọc ngược nhãn sẽ rơi vào một ô khác hoặc
                không rơi vào ô nào.
              </>
            }
          >
            The launch of the book's Fig. 3.1: <code>dimGrid(2, 2, 1)</code>, <code>dimBlock(4, 2, 2)</code>. The first
            target is the book's own example (p. 48); the others are drawn so that reading the label backwards lands on
            a different cell or on none.
          </Translated>
        </>
      }
    >
      <p style={{ margin: '0 0 0.8rem', fontSize: '1rem' }} data-out="target" aria-live="polite">
        <Translated
          vi={
            <>
              Mục tiêu {Math.min(at, targets.length - 1) + 1} trên {targets.length}: bấm vào{' '}
              <b>
                {formatThreadLabel(target.thread).replace('Thread', 'Luồng')} của {formatBlockLabel(target.block, DRILL.grid).replace('Block', 'Khối')}
              </b>
              .
            </>
          }
        >
          Target {Math.min(at, targets.length - 1) + 1} of {targets.length}: click{' '}
          <b>
            {formatThreadLabel(target.thread)} of {formatBlockLabel(target.block, DRILL.grid)}
          </b>
          .
        </Translated>
      </p>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `max-content repeat(${DRILL.grid.x}, max-content)`,
          gap: '0.7rem',
          alignItems: 'center',
          marginBottom: '1rem',
        }}
      >
        <span />
        {blockXs.map((x) => (
          <span key={x} className="mx-label" style={{ margin: 0, textAlign: 'center' }}>
            blockIdx.x = {x}
          </span>
        ))}
        {blockYs.map((y) => [
          <span key={'y' + y} className="mx-label" style={{ margin: 0 }}>
            blockIdx.y = {y}
          </span>,
          ...blockXs.map((x) => {
            const blk = { x, y, z: 0 };
            return (
              <div key={`${x}-${y}`} className="panel" data-drill-block={`${x}-${y}`} style={{ display: 'flex', gap: '0.9rem' }}>
                {zs.map((z) => (
                  <div key={z}>
                    <div className="mx-label">threadIdx.z = {z}</div>
                    <Lattice
                      nx={DRILL.block.x}
                      ny={DRILL.block.y}
                      axis="threadIdx"
                      label={tr(`block x ${x} y ${y}, layer z ${z}`, `khối x ${x} y ${y}, lớp z ${z}`)}
                      cell={(tx, ty) => cell(blk, { x: tx, y: ty, z })}
                    />
                  </div>
                ))}
              </div>
            );
          }),
        ])}
      </div>
      <div className={'narrate' + (read ? ' bad' : '')} data-out="feedback" aria-live="polite">
        {!click ? (
          <Translated vi="Chưa bấm ô nào.">Nothing clicked yet.</Translated>
        ) : hit ? (
          <>
            <Translated
              vi={
                <>
                  <b>Tìm thấy.</b> {formatThreadLabel(target.thread).replace('Thread', 'Luồng')} có z = {target.thread.z}, y ={' '}
                  {target.thread.y}, x = {target.thread.x}: số cuối của nhãn là x. Theo thứ tự của hàm dựng, <code>threadIdx</code> của
                  nó là {formatDim3(target.thread)} và <code>blockIdx</code> là {formatDim3(target.block)}.
                </>
              }
            >
              <b>Found.</b> {formatThreadLabel(target.thread)} is z = {target.thread.z}, y = {target.thread.y}, x ={' '}
              {target.thread.x}: the last number of the label is x. In constructor order its <code>threadIdx</code> is{' '}
              {formatDim3(target.thread)} and its <code>blockIdx</code> is {formatDim3(target.block)}.
            </Translated>
          </>
        ) : (
          <>
            <b>
              <Translated vi={<>Đó là {formatThreadLabel(click.thread).replace('Thread', 'Luồng')} của {formatBlockLabel(click.block, DRILL.grid).replace('Block', 'Khối')}.</>}>
                That is {formatThreadLabel(click.thread)} of {formatBlockLabel(click.block, DRILL.grid)}.
              </Translated>
            </b>{' '}
            {read!.block === 'reversed'
              ? tr(`You went to blockIdx.x = ${click.block.x}, blockIdx.y = ${click.block.y}. A block label is (y, x): its first number is y.`, `Bạn đã đến blockIdx.x = ${click.block.x}, blockIdx.y = ${click.block.y}. Nhãn khối là (y, x): số đầu tiên của nó là y.`)
              : read!.block === 'other'
                ? tr('The block is not the one asked for. A block label is (y, x).', 'Khối này không phải khối được yêu cầu. Nhãn khối là (y, x).')
                : tr('The block is right.', 'Khối đúng.')}{' '}
            {read!.thread === 'reversed'
              ? tr('For the thread you read the label as (x, y, z). It is (z, y, x): the last number is x.', 'Với luồng, bạn đã đọc nhãn là (x, y, z). Nó là (z, y, x): số cuối là x.')
              : read!.thread === 'other'
                ? tr('The thread is not the one asked for. A thread label is (z, y, x): layer first, then row, then column.', 'Luồng này không phải luồng được yêu cầu. Nhãn luồng là (z, y, x): lớp trước, rồi hàng, rồi cột.')
                : tr('The thread is right.', 'Luồng đúng.')}
          </>
        )}
      </div>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        {hit && !finished ? (
          <button className="primary" onClick={next}>
            <Translated vi="Mục tiêu tiếp theo">Next target</Translated>
          </button>
        ) : null}
        {finished ? (
          <button className="primary" onClick={restart}>
            <Translated vi="Ba mục tiêu mới">Three new targets</Translated>
          </button>
        ) : null}
        <span className="muted" data-out="score" data-found={found} data-misses={misses}>
          <Translated vi={<>Tìm thấy {found} trên {targets.length}, với {num(misses)} lần trượt.</>}>
            Found {found} of {targets.length}, with {plural(misses, 'miss', 'misses')}.
          </Translated>
        </span>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- four ways to size a grid
const SIZE_PRESETS = [
  { n: 1000, b: 256 },
  { n: 4000, b: 256 },
];

function SizingBench() {
  const [n, setN] = useState(1000);
  const [b, setB] = useState(256);
  const rows = FORMULAS.map((f) => ({ f, c: coverage(n, b, sizeGrid(n, b, f.key)) }));
  const scale = Math.max(n, ...rows.map((r) => r.c.threads));
  const rem = n % b;
  const pct = (v: number) => (100 * v) / scale + '%';

  return (
    <Bench
      title={tr('Four ways to size a grid', 'Bốn cách định kích thước lưới')}
      task={tr('Find an n for which the second formula launches a block that does nothing. Then say when the first one is right.', 'Tìm một n mà với nó công thức thứ hai khởi chạy một khối không làm gì. Rồi nói khi nào công thức thứ nhất đúng.')}
      foot={
        <>
          <Translated
            vi={
              <>
                Mỗi luồng một phần tử, một chiều. <code>n</code> và <code>b</code> là <code>int</code>, nên <code>/</code> giữa chúng
                cắt bỏ phần lẻ. Các cấu hình mẫu là hai ví dụ của sách (p. 46).
              </>
            }
          >
            One element per thread, in one dimension. <code>n</code> and <code>b</code> are <code>int</code>, so{' '}
            <code>/</code> between them truncates. The presets are the book's two examples (p. 46).
          </Translated>
        </>
      }
    >
      <div className="row" style={{ marginBottom: '1rem', alignItems: 'flex-end' }}>
        <label className="field">
          <span>
            <Translated vi="Số phần tử, ">Elements, </Translated><code>n</code> = <b data-out="n">{num(n)}</b>
          </span>
          <input type="range" min={1} max={5000} step={1} value={n} aria-label="n" style={{ width: '18rem' }} onChange={(e) => setN(Number(e.target.value))} />
        </label>
        <span className="row" style={{ gap: '0.3rem' }}>
          <button className="quiet" aria-label={tr('n minus 1', 'n trừ 1')} onClick={() => setN(Math.max(1, n - 1))}>
            −1
          </button>
          <button className="quiet" aria-label={tr('n plus 1', 'n cộng 1')} onClick={() => setN(Math.min(5000, n + 1))}>
            +1
          </button>
        </span>
        <label className="field">
          <span>
            <Translated vi="Số luồng mỗi khối, ">Threads per block, </Translated><code>b</code> = <b data-out="b">{num(b)}</b>
          </span>
          <input type="range" min={32} max={1024} step={32} value={b} aria-label="b" style={{ width: '12rem' }} onChange={(e) => setB(Number(e.target.value))} />
        </label>
        <span className="row" role="group" aria-label={tr('Presets', 'Cấu hình mẫu')} style={{ gap: '0.4rem' }}>
          <span className="muted"><Translated vi="Cấu hình mẫu">Presets</Translated></span>
          {SIZE_PRESETS.map((p) => (
            <button
              key={p.n}
              className="quiet"
              aria-pressed={n === p.n && b === p.b}
              onClick={() => {
                setN(p.n);
                setB(p.b);
              }}
            >
              n = {num(p.n)}, b = {p.b}
            </button>
          ))}
        </span>
      </div>
      <div className="panel" style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.3rem' }}><Translated vi="Số khối">Number of blocks</Translated></th>
              <th className="num"><Translated vi="Khối">Blocks</Translated></th>
              <th className="num"><Translated vi="Luồng">Threads</Translated></th>
              <th className="num"><Translated vi="Luồng dư">Surplus threads</Translated></th>
              <th className="num"><Translated vi="Phần tử không có luồng">Elements with no thread</Translated></th>
              <th className="num"><Translated vi="Khối không có việc">Blocks with no work</Translated></th>
              <th style={{ minWidth: '15rem', paddingLeft: '0.8rem' }}>
                <Translated vi={<>Số luồng so với {num(n)} phần tử</>}>Threads against the {num(n)} elements</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ f, c }) => (
              <tr
                key={f.key}
                data-formula={f.key}
                data-blocks={c.blocks}
                data-threads={c.threads}
                data-surplus={c.surplus}
                data-uncovered={c.uncovered}
                data-idle={c.idleBlocks}
              >
                <td className="nowrap" style={{ paddingLeft: '0.3rem' }}>
                  <code style={{ whiteSpace: 'nowrap' }}>{f.code}</code>
                </td>
                <td className="num">{num(c.blocks)}</td>
                <td className="num">{num(c.threads)}</td>
                <td className="num">{num(c.surplus)}</td>
                <td className="num">
                  {c.uncovered > 0 ? <span className="chip fail">{num(c.uncovered)}</span> : <span className="chip ok">0</span>}
                </td>
                <td className="num">{c.idleBlocks > 0 ? <b>{c.idleBlocks}</b> : 0}</td>
                <td style={{ paddingLeft: '0.8rem' }}>
                  <div aria-hidden="true" style={{ display: 'flex', height: 14, position: 'relative' }}>
                    {Array.from({ length: c.blocks }, (_, k) => {
                      const w = workingInBlock(n, b, k);
                      const fill = (100 * w) / b;
                      return (
                        <div
                          key={k}
                          style={{
                            flex: `0 0 ${pct(b)}`,
                            boxSizing: 'border-box',
                            borderRight: '1px solid var(--page)',
                            outline: w === 0 ? '1px dashed var(--ink-2)' : undefined,
                            outlineOffset: -1,
                            background:
                              w === 0
                                ? 'transparent'
                                : `linear-gradient(to right, var(--ink-2) ${fill}%, var(--rule) ${fill}%)`,
                          }}
                        />
                      );
                    })}
                    {c.uncovered > 0 ? (
                      <div
                        style={{
                          flex: `0 0 ${pct(c.uncovered)}`,
                          boxSizing: 'border-box',
                          background: 'var(--fail-wash)',
                          border: '1px solid var(--fail)',
                        }}
                      />
                    ) : null}
                    <div style={{ position: 'absolute', left: pct(n), top: -3, bottom: -3, width: 0, borderLeft: '2px solid var(--ink)' }} />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted" style={{ margin: '0.6rem 0 0' }}>
          <Translated
            vi={
              <>
                Trong các thanh, mỗi đoạn là một khối: tô đậm ở chỗ các luồng của nó có phần tử, nhạt ở chỗ chúng dư, nét đứt ở chỗ cả
                khối không có gì để làm. Vạch đánh dấu phần tử <code>n</code>. Màu đỏ là dữ liệu không luồng nào chạm tới.
              </>
            }
          >
            In the bars each segment is a block: filled where its threads have an element, faint where they are surplus,
            dashed where the whole block has nothing to do. The line marks element <code>n</code>. Red is data no thread
            reaches.
          </Translated>
        </p>
      </div>
      <p style={{ margin: '0.9rem 0 0' }} data-out="sizing-summary" aria-live="polite">
        {rem === 0 ? (
          <>
            <Translated
              vi={
                <>
                  <b>
                    {num(b)} chia hết {num(n)}.
                  </b>{' '}
                  Lần này cắt bỏ phần lẻ là đúng, còn <code>n / b + 1</code> khởi chạy {num(rows[1].c.blocks)} khối, khối cuối cùng có{' '}
                  {num(b)} luồng mà tất cả đều không qua được phép kiểm tra.
                </>
              }
            >
              <b>
                {num(b)} divides {num(n)} exactly.
              </b>{' '}
              Truncation is right this time, and <code>n / b + 1</code> launches {plural(rows[1].c.blocks, 'block')}, the
              last of which has {num(b)} threads that all fail the test.
            </Translated>
          </>
        ) : (
          <>
            <Translated
              vi={
                <>
                  <b>
                    {num(b)} không chia hết {num(n)}: còn dư {num(rem)} phần tử.
                  </b>{' '}
                  Cắt bỏ phần lẻ thì không khởi chạy luồng nào cho chúng. Ba công thức còn lại khởi chạy thêm một khối, trong đó{' '}
                  {num(b - rem)} luồng là dư.
                </>
              }
            >
              <b>
                {num(b)} does not divide {num(n)}: {plural(rem, 'element')} are left over.
              </b>{' '}
              Truncation launches no thread for them. The other three launch one more block, in which {num(b - rem)}{' '}
              threads are surplus.
            </Translated>
          </>
        )}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- tables computed from the model
const mark = (legal: boolean) =>
  legal ? (
    <span className="chip ok"><Translated vi="cho phép">allowed</Translated></span>
  ) : (
    <span className="chip fail"><Translated vi="từ chối">rejected</Translated></span>
  );

function ShapesTable() {
  const one = dim3();
  return (
    <table className="data" data-table="shapes">
      <thead>
        <tr>
          <th><Translated vi="Hình dạng khối">Block shape</Translated></th>
          <th className="num"><Translated vi="Luồng">Threads</Translated></th>
          <th><Translated vi="Quy tắc của sách">The book's rule</Translated></th>
          <th><Translated vi="Giới hạn được tài liệu hóa của CUDA">CUDA's documented limits</Translated></th>
        </tr>
      </thead>
      <tbody>
        {[...BOOK_SHAPES, OUR_SHAPE].map((s) => {
          const book = checkLaunch(one, s, BOOK_RULE);
          const cuda = checkLaunch(one, s, CUDA_DOCUMENTED);
          return (
            <tr key={formatDim3(s)} data-shape={`${s.x}-${s.y}-${s.z}`} data-book={String(book.legal)} data-cuda={String(cuda.legal)}>
              <td className="nowrap">
                <code>dim3{formatDim3(s)}</code>
                {s === OUR_SHAPE ? <span className="muted"><Translated vi=" của khóa học này"> ours</Translated></span> : null}
              </td>
              <td className="num nowrap">
                {s.x} × {s.y} × {s.z} = {num(threadsPerBlock(s))}
              </td>
              <td>{mark(book.legal)}</td>
              <td>
                {mark(cuda.legal)}
                {cuda.violations.length ? <span className="muted"> {tr(cuda.violations.map(describeViolation).join(' '), cuda.violations.map((v) => (v.clause === 'zero' ? `${v.field} là ${num(v.value)}. Mọi chiều phải ít nhất là 1.` : v.clause === 'block-threads' ? `Một khối sẽ chứa ${num(v.value)} luồng. Giới hạn là ${num(v.limit)}.` : v.clause === 'block-dim' ? `${v.field} là ${num(v.value)}. Giới hạn cho chiều đó của một khối là ${num(v.limit)}.` : `${v.field} là ${num(v.value)}. Giới hạn cho chiều đó của một lưới là ${num(v.limit)}.`)).join(' '))}</span> : null}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function LimitsTable() {
  const { reports, loading } = useReports();
  const mine = reported(reports);
  const triple = (d: Dim3 | null, none: string) =>
    d ? (
      <>
        <Cap v={d.x} />, <Cap v={d.y} />, <Cap v={d.z} />
      </>
    ) : (
      <span className="muted">{none}</span>
    );
  const rows = [
    { c: BOOK_CHOICE, name: tr('The book (p. 47)', 'Sách (p. 47)'), none: tr('not stated', 'không nêu') },
    { c: CUDA_CHOICE, name: tr("NVIDIA's documentation", 'Tài liệu của NVIDIA'), none: tr('not stated', 'không nêu') },
    ...mine.map((c) => ({ c, name: c.label, none: c.kind === 'metal' ? tr('not in a Metal device record', 'không có trong bản ghi thiết bị Metal') : tr('not reported', 'không được báo') })),
  ];
  return (
    <>
      <table className="data" data-table="limits">
        <thead>
          <tr>
            <th><Translated vi="Nguồn">Source</Translated></th>
            <th className="num"><Translated vi="Số luồng trong một khối">Threads in a block</Translated></th>
            <th><Translated vi="Khối, theo từng chiều (x, y, z)">Block, per dimension (x, y, z)</Translated></th>
            <th><Translated vi="Lưới, theo từng chiều (x, y, z)">Grid, per dimension (x, y, z)</Translated></th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ c, name, none }) => (
            <tr key={c.key} data-limits={c.key}>
              <td>{name}</td>
              <td className="num">
                {c.limits.maxThreadsPerBlock != null ? num(c.limits.maxThreadsPerBlock) : <span className="muted">{none}</span>}
              </td>
              <td>{triple(c.limits.maxBlockDim, none)}</td>
              <td>{triple(c.limits.maxGridDim, none)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {!loading && mine.length === 0 ? (
        <p className="muted">
          <Translated
            vi={
              <>
                Máy này chưa có báo cáo thiết bị nào. Hãy chạy <code>./verify.sh</code> và các thiết bị của chính bạn sẽ xuất hiện thành các hàng ở đây.
              </>
            }
          >
            No device report on this machine yet. Run <code>./verify.sh</code> and your own devices appear as rows here.
          </Translated>
        </p>
      ) : null}
    </>
  );
}

// ---------------------------------------------------------------- quoted source
// Quoted from llama.cpp, ggml/src/ggml-cuda/pad.cu, as recorded in notes/today/ch03-facts.md (D1).
// S31.test.ts fails if these lines drift from the facts file.
const PAD_KERNEL = `    // blockIdx.z: i3*ne2+i2
    // blockIdx.y: i1
    // blockIDx.x: i0 / CUDA_PAD_BLOCK_SIZE
    // gridDim.y and gridDim.z are capped at 65535, blocks stride over larger ne1 and ne2*ne3`;
const PAD_LAUNCH = `    int  num_blocks = (ne0 + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    dim3 gridDim(num_blocks, std::min(ne1, 65535), std::min(ne2 * ne3, 65535));
    pad_f32<<<gridDim, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(src, ...);`;

// ---------------------------------------------------------------- questions
const EX3 = {
  bd: dim3(16, 32),
  gd: dim3(sizeGridMinusOne(300, 16), sizeGridMinusOne(150, 32)),
};
const EX3_THREADS = Number(totalThreads(EX3.gd, EX3.bd));
const LABEL_Q = { thread: threadFromLabel([1, 0, 2]), block: dim3(4, 2, 2) };

export const questionsS31: Record<string, Question> = {
  'ch03/3.1/q-label': {
    kind: 'numeric',
    id: 'ch03/3.1/q-label',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Một khối được khởi chạy với <code>dim3(4, 2, 2)</code>. Các luồng của nó được đánh số từ 0 với x thay đổi nhanh nhất,
              rồi đến y, rồi z. <b>Thread (1,0,2)</b> của sách nhận số nào?
            </>
          }
        >
          A block is launched as <code>dim3(4, 2, 2)</code>. Its threads are numbered from 0 with x moving fastest, then
          y, then z. What number does the book's <b>Thread (1,0,2)</b> get?
        </Translated>
      </>
    ),
    answer: linearThread(LABEL_Q.thread, LABEL_Q.block),
    explain: (
      <Translated
        vi={
          <p>
            Nhãn là (z, y, x), nên x = 2, y = 0, z = 1, và số thứ tự là{' '}
            {linearThreadWorking(LABEL_Q.thread, LABEL_Q.block)}. Đọc nhãn theo thứ tự của hàm dựng cho 1 + 0×4 + 2×8 = 17, và
            một khối 16 luồng không có luồng 17. Đó là một tín hiệu báo động hữu ích: khi một nhãn đọc theo cách của bạn chỉ ra
            ngoài khối, thì bạn đã đọc ngược thứ tự.
          </p>
        }
      >
        <p>
          The label is (z, y, x), so x = 2, y = 0, z = 1, and the number is{' '}
          {linearThreadWorking(LABEL_Q.thread, LABEL_Q.block)}. Reading the label in constructor order gives 1 + 0×4 +
          2×8 = 17, and a block of 16 threads has no thread 17. That is a useful alarm: when a label read your way
          points outside the block, you have the order backwards.
        </p>
      </Translated>
    ),
    source: 'ch. 3 §3.1, p. 48, for the label; NVIDIA CUDA Programming Guide (checked 2026-11-20) for the numbering',
    vi: { source: 'ch. 3 §3.1, p. 48, cho nhãn; NVIDIA CUDA Programming Guide (kiểm tra ngày 2026-11-20) cho cách đánh số' },
  },
  'ch03/3.1/q-zcap': {
    kind: 'choice',
    id: 'ch03/3.1/q-zcap',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Theo các giới hạn được tài liệu hóa của NVIDIA, hình dạng khối nào trong số này bị từ chối dù có ít hơn 1024 luồng?
            </>
          }
        >
          Going by NVIDIA's documented limits, which of these block shapes is rejected although it has fewer than 1024
          threads?
        </Translated>
      </>
    ),
    options: [
      { label: <code>dim3(512, 1, 1)</code>, why: <Translated vi="512 luồng, và x có thể lên tới 1024.">512 threads, and x may go up to 1024.</Translated> },
      { label: <code>dim3(8, 16, 4)</code>, why: <Translated vi="512 luồng, và mọi chiều đều nằm trong giới hạn của nó.">512 threads, and every dimension is inside its cap.</Translated> },
      {
        label: <code>dim3(32, 16, 2)</code>,
        why: (
          <Translated vi="Đúng 1024 luồng, điều được phép, và z = 2 thấp hơn nhiều so với giới hạn của nó.">
            Exactly 1024 threads, which is allowed, and z = 2 is far below its cap.
          </Translated>
        ),
      },
      {
        label: <code>dim3(1, 1, 128)</code>,
        correct: true,
        why: (
          <Translated vi="Chỉ 128 luồng, nhưng chiều z của một khối bị giới hạn ở 64.">
            Only 128 threads, but the z dimension of a block is capped at 64.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một hình dạng khối phải qua hai phép kiểm tra: tích các chiều của nó tối đa là 1024, và mỗi chiều nằm trong giới hạn
            riêng của nó là 1024, 1024 và 64. Sách chỉ nêu phép thứ nhất. Các ví dụ của chính sách qua hoặc trượt cả hai phép
            như nhau, đó là lý do khoảng hở này không lộ ra ở đó.
          </p>
        }
      >
        <p>
          A block shape has to pass two tests: the product of its dimensions is at most 1024, and each dimension is
          inside its own cap of 1024, 1024 and 64. The book states only the first. Its own examples pass or fail both
          tests alike, which is why the gap does not show there.
        </p>
      </Translated>
    ),
    source: 'ch. 3 §3.1, p. 47, for the three allowed examples; NVIDIA CUDA Programming Guide, compute capabilities table (checked 2026-11-20), for the per-dimension caps',
    vi: { source: 'ch. 3 §3.1, p. 47, cho ba ví dụ được phép; NVIDIA CUDA Programming Guide, bảng compute capabilities (kiểm tra ngày 2026-11-20), cho giới hạn theo từng chiều' },
  },
  'ch03/3.1/q-silent': {
    kind: 'choice',
    id: 'ch03/3.1/q-silent',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Một hàm host khởi chạy một kernel với <code>dimBlock(32, 32, 2)</code>, sao chép buffer đầu ra về và in nó ra. Nó
              không bao giờ gọi <code>cudaGetLastError()</code>. Theo tài liệu của NVIDIA, điều gì xảy ra?
            </>
          }
        >
          A host function launches a kernel with <code>dimBlock(32, 32, 2)</code>, copies the output buffer back and
          prints it. It never calls <code>cudaGetLastError()</code>. Going by NVIDIA's documentation, what happens?
        </Translated>
      </>
    ),
    options: [
      {
        label: (
          <Translated vi="Tệp không biên dịch được: nvcc từ chối một khối 2048 luồng ở dòng khởi chạy.">
            The file does not compile: nvcc rejects a block of 2048 threads on the launch line.
          </Translated>
        ),
        why: (
          <Translated vi="Trình biên dịch chỉ thấy ba số nguyên. Giới hạn thuộc về thiết bị và được kiểm tra khi lần khởi chạy chạy.">
            The compiler sees three integers. The limit belongs to the device and is checked when the launch runs.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Câu lệnh khởi chạy ném ra một ngoại lệ nêu tên cấu hình sai.">
            The launch statement throws an exception that names the bad configuration.
          </Translated>
        ),
        why: (
          <Translated vi="Một câu lệnh khởi chạy không có giá trị và không ném gì cả. Thất bại đi vào một trạng thái lỗi mà bạn phải truy vấn.">
            A launch statement has no value and throws nothing. The failure goes into an error state that you have to query.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Runtime thu nhỏ khối về 1024 luồng mà thiết bị cho phép và chạy nó.">
            The runtime shrinks the block to the 1024 threads the device allows and runs it.
          </Translated>
        ),
        why: (
          <Translated vi="Không có gì được điều chỉnh giúp bạn. Lần khởi chạy thất bại toàn bộ.">
            Nothing is adjusted for you. The launch fails as a whole.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không luồng nào chạy, không có gì được báo, và chương trình in ra bất cứ thứ gì buffer đang chứa.">
            No thread runs, nothing is reported, and the program prints whatever the buffer held.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lần khởi chạy thất bại và ghi <code>cudaErrorInvalidConfiguration</code> vào trạng thái lỗi của runtime. Chương trình
            chưa bao giờ được bảo hãy nhìn vào đó, nên nó tiếp tục và in ra bộ nhớ cũ hoặc chưa khởi tạo như thể đó là kết quả.
            Đây là lý do mã khởi chạy thực tế hỏi lỗi sau mỗi lần khởi chạy. Nhận định này lấy từ tài liệu của NVIDIA; chúng tôi
            chưa chạy nó trên thiết bị CUDA (unverified trên thiết bị).
          </p>
        }
      >
        <p>
          The launch fails and records <code>cudaErrorInvalidConfiguration</code> in the runtime's error state. The
          program was never told to look there, so it carries on and prints stale or uninitialised memory as if it were
          a result. This is why real launch code asks for the error after every launch. The statement comes from
          NVIDIA's documentation; we have not run it on a CUDA device (unverified on device).
        </p>
      </Translated>
    ),
    source: 'NVIDIA CUDA Programming Guide and Runtime API reference (checked 2026-11-20); not in the book',
    vi: { source: 'NVIDIA CUDA Programming Guide và tài liệu tham chiếu Runtime API (kiểm tra ngày 2026-11-20); không có trong sách' },
  },
  'ch03/3.1/q-metal-size': {
    kind: 'choice',
    id: 'ch03/3.1/q-metal-size',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Trong CUDA, <code>dim3 dimGrid(42);</code> mô tả một lưới 1D gồm 42. Mã host Metal cho cùng kích thước đó là gì?
            </>
          }
        >
          In CUDA, <code>dim3 dimGrid(42);</code> describes a 1D grid of 42. What is the Metal host code for the same
          size?
        </Translated>
      </>
    ),
    options: [
      { label: <code>MTLSizeMake(42)</code>, why: <Translated vi="MTLSizeMake nhận ba đối số và không có giá trị mặc định.">MTLSizeMake takes three arguments and has no defaults.</Translated> },
      {
        label: <code>MTLSizeMake(42, 1, 1)</code>,
        correct: true,
        why: <Translated vi="Cả ba đều được cho, với 1 cho các chiều bạn không dùng.">All three are given, with 1 for the dimensions you do not use.</Translated>,
      },
      {
        label: <code>MTLSizeMake(42, 0, 0)</code>,
        why: (
          <Translated vi="Chiều không dùng là 1, như trong CUDA. Với số 0, kích thước sẽ là 42 × 0 × 0.">
            Unused dimensions are 1, as in CUDA. With zeros the size would be 42 × 0 × 0.
          </Translated>
        ),
      },
      {
        label: (
          <Translated
            vi={
              <>
                Một đối số <code>uint3</code> của hàm kernel
              </>
            }
          >
            <>
              A <code>uint3</code> argument of the kernel function
            </>
          </Translated>
        ),
        why: (
          <Translated vi="Kích thước của lần khởi chạy là mã host, được cho ở lời gọi dispatch. Đối số kernel là nơi một luồng Metal nhận vị trí của nó.">
            The size of a launch is host code, given at the dispatch call. Kernel arguments are where a Metal thread receives its position.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>MTLSize</code> là bản tương ứng của <code>dim3</code> trong Metal, với các trường <code>width</code>,{' '}
            <code>height</code> và <code>depth</code>. Quy tắc cho các chiều không dùng giống CUDA, nhưng không có gì tự điền
            chúng giúp bạn.
          </p>
        }
      >
        <p>
          <code>MTLSize</code> is Metal's counterpart of <code>dim3</code>, with fields <code>width</code>,{' '}
          <code>height</code> and <code>depth</code>. The rule for unused dimensions is the same as CUDA's, but nothing
          fills them in for you.
        </p>
      </Translated>
    ),
    source: 'Apple, MTLSize documentation (checked 2026-11-20); not in the book',
    vi: { source: 'Apple, tài liệu MTLSize (kiểm tra ngày 2026-11-20); không có trong sách' },
  },
  'ch03/3.1/q-ex3-threads': {
    kind: 'numeric',
    id: 'ch03/3.1/q-ex3-threads',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Mã host đặt <code>M = 150</code> và <code>N = 300</code> rồi khởi chạy một kernel với{' '}
              <code>dim3 bd(16, 32);</code> và <code>dim3 gd((N - 1)/16 + 1, (M - 1)/32 + 1);</code>. Mọi phép tính đều trên số
              nguyên. Lưới có bao nhiêu luồng?
            </>
          }
        >
          Host code sets <code>M = 150</code> and <code>N = 300</code> and launches a kernel with{' '}
          <code>dim3 bd(16, 32);</code> and <code>dim3 gd((N - 1)/16 + 1, (M - 1)/32 + 1);</code>. All the arithmetic is
          in integers. How many threads are in the grid?
        </Translated>
      </>
    ),
    answer: EX3_THREADS,
    unit: 'threads',
    explain: (
      <Translated
        vi={
          <>
            <p>
              Ba bước, chính là các phần (a), (c) và (b) của bài tập trong sách. <code>bd</code> có hai đối số, nên z của nó là
              1 và một khối chứa 16 × 32 = {num(threadsPerBlock(EX3.bd))} luồng. Lưới là (299/16 + 1, 149/32 + 1) = (18 + 1, 4 +
              1) = ({EX3.gd.x}, {EX3.gd.y}), tức là {num(blockCount(EX3.gd))} khối. Vậy lưới có {num(blockCount(EX3.gd))} ×{' '}
              {num(threadsPerBlock(EX3.bd))} = {num(EX3_THREADS)} luồng.
            </p>
            <p>
              Hãy để ý kích thước nào đi đâu: x được định kích thước theo <code>N</code>, 300 cột, và y theo <code>M</code>, 150
              hàng. Các luồng trải {EX3.gd.x * EX3.bd.x} × {EX3.gd.y * EX3.bd.y} cho dữ liệu 300 × 150. Bao nhiêu luồng trong số
              đó có phần tử để làm việc là chủ đề của mục tiếp theo.
            </p>
          </>
        }
      >
        <>
          <p>
            Three steps, which are parts (a), (c) and (b) of the book's exercise. <code>bd</code> has two arguments, so
            its z is 1 and a block holds 16 × 32 = {num(threadsPerBlock(EX3.bd))} threads. The grid is (299/16 + 1,
            149/32 + 1) = (18 + 1, 4 + 1) = ({EX3.gd.x}, {EX3.gd.y}), which is {num(blockCount(EX3.gd))} blocks. So the
            grid has {num(blockCount(EX3.gd))} × {num(threadsPerBlock(EX3.bd))} = {num(EX3_THREADS)} threads.
          </p>
          <p>
            Notice which size went where: x is sized from <code>N</code>, the 300 columns, and y from <code>M</code>, the
            150 rows. The threads span {EX3.gd.x * EX3.bd.x} by {EX3.gd.y * EX3.bd.y} for data of 300 by 150. How many of
            them have an element to work on is the next section's subject.
          </p>
        </>
      </Translated>
    ),
    source: 'ch. 3, exercise 3, p. 65',
    vi: { unit: 'luồng', source: 'ch. 3, bài tập 3, p. 65' },
  },
};

// ---------------------------------------------------------------- the section
const FIG31_TOTAL = Number(totalThreads(FIG31.grid, FIG31.block));
const FIG31_BLOCKS = Number(blockCount(FIG31.grid));
const FIG31_PER = Number(threadsPerBlock(FIG31.block));

export function S31() {
  const launch1d = useLabFile('ch03/snippets/launch_1d.cu');
  const fig31 = useLabFile('ch03/snippets/launch_fig31.cu');
  const checked = useLabFile('ch03/snippets/launch_checked.cu');
  return (
    <>
      <Translated
        vi={
          <p>
            Ở chương 2, một con số, <code>i</code>, cho một luồng biết phải làm gì. Một phần tử ma trận cần hai con số, và một
            phần tử của tensor <code>[batch, sequence, hidden]</code> cần ba. CUDA cho phép chính lần khởi chạy có tới ba chiều,
            để mỗi luồng được trao tọa độ theo hình dạng dữ liệu của nó. Mục này nói về lần khởi chạy: cách bạn nêu hình dạng
            của nó, mỗi luồng khi đó đọc được gì về chính mình, và thiết bị chấp nhận những hình dạng nào. Một luồng làm gì với
            tọa độ của nó là chủ đề của{' '}
            <a href={href('ch03', '3.2')}>mục 3.2</a>.
          </p>
        }
      >
        <p>
          In chapter 2 one number, <code>i</code>, told a thread what to do. A matrix element needs two numbers, and an
          element of a <code>[batch, sequence, hidden]</code> tensor needs three. CUDA lets the launch itself have up to
          three dimensions, so that a thread is handed coordinates in the shape of its data. This section is about the
          launch: how you state its shape, what each thread can then read about itself, and which shapes a device
          accepts. What a thread does with its coordinates is{' '}
          <a href={href('ch03', '3.2')}>section 3.2</a>.
        </p>
      </Translated>
      <Source section="3.1" pages="45–48" figs={tr('Fig. 3.1', 'Hình 3.1')} />

      <h2><Translated vi="Cùng mã, khác tọa độ">Same code, different coordinates</Translated></h2>
      <Translated
        vi={
          <p>
            Mọi luồng của một lưới chạy cùng một hàm kernel. Thứ duy nhất khác nhau giữa luồng này với luồng khác là giá trị của
            hai biến dựng sẵn, và một luồng dùng chúng để chọn phần dữ liệu của mình. Hai biến dựng sẵn nữa mang hình dạng của
            lần khởi chạy vào kernel và có cùng giá trị trong mọi luồng.
          </p>
        }
      >
        <p>
          Every thread of a grid runs the same kernel function. The only thing that differs from one thread to the next
          is the value of two built-in variables, and a thread uses them to pick its part of the data. Two more
          built-ins carry the shape of the launch into the kernel and read the same in every thread.
        </p>
      </Translated>
      <Source section="3.1" pages="45–47" />
      <table className="data">
        <thead>
          <tr>
            <th><Translated vi="Biến dựng sẵn">Built-in</Translated></th>
            <th><Translated vi="Chứa">Holds</Translated></th>
            <th><Translated vi="Khoảng giá trị của mỗi thành phần">Range of each component</Translated></th>
            <th><Translated vi="Cùng giá trị trong">Same value in</Translated></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <code>gridDim</code>
            </td>
            <td><Translated vi="kích thước của lưới, tính bằng khối">the size of the grid, in blocks</Translated></td>
            <td><Translated vi="đặt bởi lần khởi chạy">set by the launch</Translated></td>
            <td><Translated vi="mọi luồng của lưới">every thread of the grid</Translated></td>
          </tr>
          <tr>
            <td>
              <code>blockDim</code>
            </td>
            <td><Translated vi="kích thước của một khối, tính bằng luồng">the size of a block, in threads</Translated></td>
            <td><Translated vi="đặt bởi lần khởi chạy">set by the launch</Translated></td>
            <td><Translated vi="mọi luồng của lưới">every thread of the grid</Translated></td>
          </tr>
          <tr>
            <td>
              <code>blockIdx</code>
            </td>
            <td><Translated vi="vị trí của khối này trong lưới">where this block is in the grid</Translated></td>
            <td>
              <Translated vi={<>0 đến <code>gridDim</code> − 1</>}>0 to <code>gridDim</code> − 1</Translated>
            </td>
            <td><Translated vi="mọi luồng của một khối">all threads of one block</Translated></td>
          </tr>
          <tr>
            <td>
              <code>threadIdx</code>
            </td>
            <td><Translated vi="vị trí của luồng này trong khối của nó">where this thread is in its block</Translated></td>
            <td>
              <Translated vi={<>0 đến <code>blockDim</code> − 1</>}>0 to <code>blockDim</code> − 1</Translated>
            </td>
            <td><Translated vi="một luồng của mỗi khối">one thread of each block</Translated></td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Mỗi biến trong bốn biến có một <code>x</code>, một <code>y</code> và một <code>z</code>. Hệ phân cấp có hai tầng và
            không hơn: một lưới là một mảng ba chiều các khối, và một khối là một mảng ba chiều các luồng. Mọi khối của một lưới
            có cùng hình dạng. Bốn cái tên này thuộc về CUDA C++ và không thể đổi.
          </p>
        }
      >
        <p>
          Each of the four has an <code>x</code>, a <code>y</code> and a <code>z</code>. The hierarchy has two levels and
          no more: a grid is a three-dimensional array of blocks, and a block is a three-dimensional array of threads.
          All blocks of a grid have the same shape. The four names belong to CUDA C++ and cannot be changed.
        </p>
      </Translated>
      <Today checked="2026-11-20">
        <Translated
          vi={
            <p>
              NVIDIA ghi <code>gridDim</code> và <code>blockDim</code> có kiểu <code>dim3</code>, còn{' '}
              <code>blockIdx</code> và <code>threadIdx</code> có kiểu <code>uint3</code>. Cả hai đều là cấu trúc gồm ba giá trị
              không dấu. Tính không dấu quan trọng ở mục 3.3, nơi một chỉ số cần có thể xuống dưới 0.
            </p>
          }
        >
          <p>
            NVIDIA documents <code>gridDim</code> and <code>blockDim</code> as <code>dim3</code>, and{' '}
            <code>blockIdx</code> and <code>threadIdx</code> as <code>uint3</code>. Both are structures of three unsigned
            values. Unsigned matters in section 3.3, where an index has to be able to go below zero.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Cấu hình thực thi">The execution configuration</Translated></h2>
      <Translated
        vi={
          <p>
            Hình dạng được đặt ở lời gọi. Giữa <code>&lt;&lt;&lt;</code> và <code>&gt;&gt;&gt;</code> là hai giá trị kiểu{' '}
            <code>dim3</code>: trước là kích thước lưới, tính bằng khối, sau là kích thước một khối, tính bằng luồng. Một{' '}
            <code>dim3</code> là một vector ba số nguyên tên <code>x</code>, <code>y</code> và <code>z</code>. Một chiều bạn
            không dùng là 1, không bao giờ là 0.
          </p>
        }
      >
        <p>
          The shape is set at the call. Between <code>&lt;&lt;&lt;</code> and <code>&gt;&gt;&gt;</code> go two values of
          type <code>dim3</code>: first the size of the grid, counted in blocks, then the size of one block, counted in
          threads. A <code>dim3</code> is a vector of three integers named <code>x</code>, <code>y</code> and{' '}
          <code>z</code>. A dimension you do not use is 1, never 0.
        </p>
      </Translated>
      <Source section="3.1" pages="46–47" />
      <Code
        tabs={[{ label: 'CUDA', code: launch1d ?? tr('// loading…', '// đang tải…'), status: 'front-end-checked', tiers: CODE_TIERS }]}
        highlight={[5, 6, 12]}
        caption={tr(
          "The book's 1D example (p. 46): 32 blocks of 128 threads, which is 4096 threads. dimGrid and dimBlock are ordinary host variables with ordinary names; the book launches the same grid from two variables called dog and cat to make the point.",
          'Ví dụ 1D của sách (p. 46): 32 khối 128 luồng, tức là 4096 luồng. dimGrid và dimBlock là những biến host bình thường với tên bình thường; sách khởi chạy cùng lưới đó từ hai biến tên dog và cat để nhấn mạnh điều này.',
        )}
      />
      <Translated
        vi={
          <p>
            Hàm thứ hai chạy được vì hàm dựng của <code>dim3</code> gán 1 cho mọi thành phần bạn bỏ trống. Vì vậy một số nguyên
            duy nhất điền vào <code>x</code> và để <code>y</code> và <code>z</code> bằng 1. Bên trong kernel, hai giá trị đến
            dưới dạng <code>gridDim</code> và <code>blockDim</code>: với lần khởi chạy ở trên, mọi luồng đọc{' '}
            <code>gridDim.x</code> = 32 và <code>blockDim.x</code> = 128. Một khi lưới đã được khởi chạy, hình dạng của nó cố
            định cho đến khi cả lưới chạy xong.
          </p>
        }
      >
        <p>
          The second function works because the constructor of <code>dim3</code> gives 1 to every component you leave
          out. A single integer therefore fills <code>x</code> and leaves <code>y</code> and <code>z</code> at 1. Inside
          the kernel the two values arrive as <code>gridDim</code> and <code>blockDim</code>: with the launch above,
          every thread reads <code>gridDim.x</code> = 32 and <code>blockDim.x</code> = 128. Once a grid is launched its
          shape stays fixed until the whole grid has finished.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Lưới và khối không cần có cùng số chiều. Hình của sách cho mục này dùng một lưới hai chiều gồm các khối ba chiều.
          </p>
        }
      >
        <p>
          Grid and block do not need the same number of dimensions. The book's figure for this section uses a
          two-dimensional grid of three-dimensional blocks.
        </p>
      </Translated>
      <Source section="3.1" pages="47–48" figs={tr('Fig. 3.1', 'Hình 3.1')} />
      <Code
        tabs={[{ label: 'CUDA', code: fig31 ?? tr('// loading…', '// đang tải…'), status: 'front-end-checked', tiers: CODE_TIERS }]}
        highlight={[15, 16]}
        caption={tr(
          "The launch of the book's Fig. 3.1 (lines 15 and 16), with a kernel of ours behind it so that there is something to launch. Each thread writes one integer: its own number within its block. Line 9 comes back later in this section.",
          'Lần khởi chạy của Fig. 3.1 trong sách (dòng 15 và 16), với một kernel của khóa học này phía sau để có thứ mà khởi chạy. Mỗi luồng ghi một số nguyên: số thứ tự của chính nó trong khối. Dòng 9 sẽ quay lại ở phần sau của mục này.',
        )}
      />
      <Predict
        id="ch03/3.1/p-fig31"
        prompt={
          <>
            <Translated
              vi={
                <>
                  Lần khởi chạy ở dòng 15 và 16 tạo ra bao nhiêu luồng? Hãy cam kết một con số trước khi bạn đếm chúng trên trang.
                </>
              }
            >
              How many threads does the launch on lines 15 and 16 create? Commit to a number before you count them on the
              page.
            </Translated>
          </>
        }
        unit={tr('threads', 'luồng')}
      >
        {(v) => (
          <p>
            <Translated
              vi={
                <>
                  <b>{FIG31_TOTAL}.</b> Lưới là 2 × 2 × 1 = {FIG31_BLOCKS} khối và mỗi khối là 4 × 2 × 2 ={' '}
                  {FIG31_PER} luồng.{' '}
                  {v === FIG31_TOTAL
                    ? 'Đó đúng là con số của bạn.'
                    : v === FIG31_PER
                      ? 'Mười sáu là một khối. Lưới có bốn khối như vậy.'
                      : v === FIG31_BLOCKS
                        ? 'Bốn là số khối. Mỗi khối chứa mười sáu luồng.'
                        : v === 20 || v === 13
                          ? 'Các kích thước nhân với nhau. Mọi khối của lưới chứa trọn một bộ luồng.'
                          : 'Hãy nhân ba số của mỗi dim3, rồi nhân hai kết quả với nhau.'}{' '}
                  Trình khám phá bên dưới mở sẵn lần khởi chạy này. Lưới thực tế chứa từ hàng nghìn đến hàng triệu luồng; lưới này đủ
                  nhỏ để vẽ ra.
                </>
              }
            >
              <b>{FIG31_TOTAL}.</b> The grid is 2 × 2 × 1 = {FIG31_BLOCKS} blocks and each block is 4 × 2 × 2 ={' '}
              {FIG31_PER} threads.{' '}
              {v === FIG31_TOTAL
                ? 'That is what you had.'
                : v === FIG31_PER
                  ? 'Sixteen is one block. The grid has four of them.'
                  : v === FIG31_BLOCKS
                    ? 'Four is the number of blocks. Each holds sixteen threads.'
                    : v === 20 || v === 13
                      ? 'The sizes multiply. Every block of the grid holds a full set of threads.'
                      : 'Multiply the three numbers of each dim3, then multiply the two results.'}{' '}
              The explorer below opens on this launch. Real grids hold thousands to millions of threads; this one is
              small enough to draw.
            </Translated>
          </p>
        )}
      </Predict>

      <h2><Translated vi="Định kích thước lưới cho n phần tử">Sizing a grid for n elements</Translated></h2>
      <Translated
        vi={
          <p>
            Hình dạng hiếm khi là hằng số. Với <code>n</code> phần tử và các khối <code>b</code> luồng, bạn cần đủ khối để phủ
            hết <code>n</code>, nghĩa là chia rồi làm tròn lên. Sách viết <code>ceil(n/256.0)</code>: với khối 256,{' '}
            <code>n</code> = 1000 cho 4 khối và <code>n</code> = 4000 cho 16. Phần <code>.0</code> có tác dụng thật. Nó khiến
            phép chia diễn ra theo dấu phẩy động, để còn phần lẻ cho <code>ceil</code> làm tròn lên.
          </p>
        }
      >
        <p>
          Shapes are rarely constants. For <code>n</code> elements and blocks of <code>b</code> threads you need enough
          blocks to cover all of <code>n</code>, which means dividing and rounding up. The book writes{' '}
          <code>ceil(n/256.0)</code>: with blocks of 256, <code>n</code> = 1000 gives 4 blocks and <code>n</code> = 4000
          gives 16. The <code>.0</code> is doing work. It makes the division happen in floating point, so that there is
          a fraction left for <code>ceil</code> to round up.
        </p>
      </Translated>
      <Source section="3.1" pages="46" />
      <Translated
        vi={
          <p>
            Cách kia giữ nguyên số nguyên: <code>(n + b - 1) / b</code>. Chính các bài tập của sách dùng dạng số nguyên,{' '}
            <code>(N + 128 - 1)/128</code> ở chương 2 (bài tập 9, p. 42) và <code>(N - 1)/16 + 1</code> ở chương này (bài tập
            3, p. 65). Trình khám phá đặt cả hai cách cạnh hai lối tắt mà người ta hay nghĩ tới trước.
          </p>
        }
      >
        <p>
          The other way stays in integers: <code>(n + b - 1) / b</code>. The book's own exercises use integer forms,{' '}
          <code>(N + 128 - 1)/128</code> in chapter 2 (exercise 9, p. 42) and <code>(N - 1)/16 + 1</code> in this chapter
          (exercise 3, p. 65). The explorer puts both ways next to the two shortcuts people reach for first.
        </p>
      </Translated>
      <Today checked="2026-11-20">
        <Translated
          vi={
            <p>
              Hướng dẫn của NVIDIA định kích thước lưới bằng số nguyên, <code>(vectorLength + threads-1)/threads</code>, hoặc bằng{' '}
              <code>cuda::ceil_div</code> từ <code>&lt;cuda/cmath&gt;</code>. Dạng <code>ceil</code> không có trên trang nào của các
              hãng mà chúng tôi đã đọc. Nó truyền một <code>double</code> cho <code>dim3</code> và dựa vào một phép chuyển kiểu
              ngầm định, và chúng tôi chưa biên dịch nó bằng nvcc (unverified). Các lab dùng dạng số nguyên.
            </p>
          }
        >
          <p>
            NVIDIA's guide sizes its grids in integers, <code>(vectorLength + threads-1)/threads</code>, or with{' '}
            <code>cuda::ceil_div</code> from <code>&lt;cuda/cmath&gt;</code>. The <code>ceil</code> form is on no vendor
            page we read. It hands a <code>double</code> to <code>dim3</code> and relies on an implicit conversion, and
            we have not compiled it with nvcc (unverified). The labs use the integer form.
          </p>
        </Translated>
      </Today>
      <SizingBench />
      <Translated
        vi={
          <p>
            Có hai chỗ hỏng. Cắt bỏ phần lẻ để <code>n mod b</code> phần tử cuối cùng không có luồng, và không có gì báo lỗi:
            kernel đúng và phần đuôi của đầu ra đơn giản là không bao giờ được ghi. Cộng thêm một vô điều kiện thì đúng cho đến
            khi <code>b</code> chia hết <code>n</code>, và lúc đó nó khởi chạy một khối mà mọi luồng đều không qua được phép
            kiểm tra.
          </p>
        }
      >
        <p>
          Two things go wrong. Truncating leaves the last <code>n mod b</code> elements without a thread, and nothing
          complains: the kernel is correct and the tail of the output is simply never written. Adding one
          unconditionally is right until <code>b</code> divides <code>n</code>, and then it launches a block in which
          every thread fails the test.
        </p>
      </Translated>
      <Today checked="2026-11-20">
        <Translated
          vi={
            <p>
              Hướng dẫn của NVIDIA về chi phí của từng loại lãng phí: "Launching extra threads in a block that do no work does not
              incur a large overhead cost, however launching thread blocks in which no threads do work should be avoided."
            </p>
          }
        >
          <p>
            NVIDIA's guide on the cost of each kind of waste: "Launching extra threads in a block that do no work does
            not incur a large overhead cost, however launching thread blocks in which no threads do work should be
            avoided."
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Hai công thức cuối luôn cho cùng kết quả, và không bao giờ mắc lỗi nào trong hai lỗi trên. Luồng dư bên trong khối
            cuối cùng là cái giá bình thường của một lưới gồm nguyên các khối, đó là lý do kernel giữ phép <code>if</code> của nó.
          </p>
        }
      >
        <p>
          The last two formulas always agree, and they never do either. Surplus threads inside the last block are the
          normal price of a grid made of whole blocks, which is why the kernel keeps its <code>if</code>.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi luồng một phần tử, một chiều, và kích thước thấp hơn nhiều so với miền giá trị của <code>int</code>, nên phép
              tràn số mà <code>n + b - 1</code> có thể gặp gần đỉnh miền đó không xuất hiện. Hàng <code>ceil</code>{' '}
              được tính bằng độ chính xác kép và lấy thành số nguyên; phép chuyển kiểu mà C++ thực hiện khi kết quả khởi tạo một{' '}
              <code>dim3</code> không được mô phỏng.
            </p>
          }
        >
          <p>
            One element per thread, one dimension, and sizes far below the range of <code>int</code>, so the overflow
            that <code>n + b - 1</code> can run into near the top of that range does not appear. The <code>ceil</code>{' '}
            row is computed in double precision and taken as an integer; the conversion that C++ performs when the
            result initialises a <code>dim3</code> is not modelled.
          </p>
        </Translated>
      </Simplification>

      <h2><Translated vi="Cái bẫy thứ tự nhãn">The label order trap</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn viết hình dạng khởi chạy với x trước: <code>dim3(4, 2, 2)</code> rộng 4 theo x. Sách gọi tên khối và luồng theo
            chiều ngược lại, chiều cao nhất trước: Block (<code>blockIdx.y</code>, <code>blockIdx.x</code>) và Thread (
            <code>threadIdx.z</code>, <code>threadIdx.y</code>, <code>threadIdx.x</code>). Vậy Block (1,0) là khối có{' '}
            <code>blockIdx.y</code> = 1 và <code>blockIdx.x</code> = 0, còn Thread (1,0,2) có z = 1, y = 0 và x = 2. Mọi hình
            và ví dụ minh họa từ đây đến cuối sách đều dùng các nhãn này.
          </p>
        }
      >
        <p>
          You write a launch shape x first: <code>dim3(4, 2, 2)</code> is 4 wide in x. The book names blocks and threads
          the other way round, highest dimension first: Block (<code>blockIdx.y</code>, <code>blockIdx.x</code>) and
          Thread (<code>threadIdx.z</code>, <code>threadIdx.y</code>, <code>threadIdx.x</code>). So Block (1,0) is the
          block with <code>blockIdx.y</code> = 1 and <code>blockIdx.x</code> = 0, and Thread (1,0,2) has z = 1, y = 0
          and x = 2. Every figure and worked example from here to the end of the book uses these labels.
        </p>
      </Translated>
      <Source section="3.1" pages="48, 49" figs={tr('Fig. 3.1; footnote 1', 'Hình 3.1; chú thích 1')} />
      <Translated
        vi={
          <p>
            Lý do nằm ở thứ tự chỉ số của dữ liệu. Bạn viết một phần tử ma trận là <code>P[row][col]</code> và hình dạng tensor
            là <code>(batch, sequence, hidden)</code>: chỉ số thay đổi chậm nhất đứng đầu, chỉ số thay đổi nhanh nhất đứng cuối.
            Sách gán nhãn luồng theo cách bạn đánh chỉ số dữ liệu, để x, số cuối của nhãn, thẳng hàng với chỉ số cuối. Hãy đọc
            nó là "x là trục cuối" và sự đảo ngược thôi là cái bẫy. Chính sách cũng gọi hai thứ tự ngược nhau này là một nguồn
            gây nhầm lẫn.
          </p>
        }
      >
        <p>
          The reason is the order of data indices. You write a matrix element as <code>P[row][col]</code> and a tensor
          shape as <code>(batch, sequence, hidden)</code>: the index that varies slowest first, the one that varies
          fastest last. The book labels threads the way you index data, so that x, the last number of a label, lines up
          with the last index. Read it as "x is the last axis" and the reversal stops being a trap. The book itself
          calls the two opposite orders a source of confusion.
        </p>
      </Translated>
      <Today checked="2026-11-20">
        <Translated
          vi={
            <p>
              Thứ tự nhãn là quy ước của sách. Tài liệu của NVIDIA viết x trước ở mọi nơi. Nó cũng cố định cách đánh số các luồng
              của một khối: "the first index x moves the fastest, followed by y and then z", nên <code>threadIdx.y</code> có bước
              nhảy (stride) <code>blockDim.x</code> và <code>threadIdx.z</code> có bước nhảy{' '}
              <code>blockDim.x * blockDim.y</code>. Đó là dòng 9 của kernel ở trên.
            </p>
          }
        >
          <p>
            The label order is the book's convention. NVIDIA's documentation writes x first everywhere. It also fixes
            how the threads of a block are numbered: "the first index x moves the fastest, followed by y and then z", so
            that <code>threadIdx.y</code> has a stride of <code>blockDim.x</code> and <code>threadIdx.z</code> a stride
            of <code>blockDim.x * blockDim.y</code>. That is line 9 of the kernel above.
          </p>
        </Translated>
      </Today>
      <GridExplorer />
      <Translated
        vi={
          <p>
            Hãy thử lần lượt các hình dạng khối và theo dõi số thứ tự luồng. Chúng đếm dọc theo x, rồi sang hàng tiếp theo của
            y, rồi sang lớp tiếp theo của z. Một khối ba chiều là một dãy luồng, gập lại hai lần.
          </p>
        }
      >
        <p>
          Try the block shapes one after another and watch the thread numbers. They count along x, then start the next
          row of y, then move to the next layer of z. A three-dimensional block is one line of threads, folded twice.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Không có gì được khởi chạy. Một kết luận chỉ nói rằng sáu con số qua hay trượt một bảng giới hạn. Những giới hạn phụ
              thuộc vào nhiều thứ hơn hai hình dạng không được kiểm tra: bộ nhớ chia sẻ (shared memory) mỗi khối, số thanh ghi
              (register) một kernel dùng (một lần khởi chạy CUDA dưới 1024 luồng có thể thất bại vì lý do này), và trên Metal là
              giới hạn luồng của từng kernel đã biên dịch. Hình dạng lớn chỉ được vẽ một phần.
            </p>
          }
        >
          <p>
            Nothing is launched. A verdict says only that six numbers pass or fail a table of limits. Limits that depend
            on more than the two shapes are not tested: shared memory per block, the registers a kernel uses (a CUDA
            launch under 1024 threads can fail for that reason), and on Metal the thread limit of each compiled kernel.
            Large shapes are drawn in part.
          </p>
        </Translated>
      </Simplification>
      <Translated
        vi={
          <p>
            Cũng quy tắc gập đó là thứ <a href={href('ch03', '3.2')}>mục 3.2</a> áp dụng cho dữ liệu, và chương 4 dùng nó để cắt
            một khối thành các warp. Ngoài điều đó, số chiều bạn cho một lần khởi chạy chỉ là chuyện sổ sách: hãy chọn hình dạng
            giúp phép tính chỉ số của bạn dễ làm đúng.
          </p>
        }
      >
        <p>
          The same folding rule is what <a href={href('ch03', '3.2')}>section 3.2</a> applies to data, and chapter 4
          uses it to cut a block into warps. Beyond that, the number of dimensions you give a launch is bookkeeping:
          choose the shape that makes your index arithmetic easy to get right.
        </p>
      </Translated>
      <Today checked="2026-11-20">
        <Translated
          vi={
            <p>
              Hướng dẫn của NVIDIA: "The use of multi-dimensional thread blocks and grids is for convenience only and does not
              affect performance."
            </p>
          }
        >
          <p>
            NVIDIA's guide: "The use of multi-dimensional thread blocks and grids is for convenience only and does not
            affect performance."
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Giờ thì không còn tên trên các ô. Bài luyện dưới đây dùng lần khởi chạy của Fig. 3.1 và hỏi các luồng theo nhãn của
            sách.
          </p>
        }
      >
        <p>
          Now without the names on the cells. The drill below uses the launch of Fig. 3.1 and asks for threads by the
          book's labels.
        </p>
      </Translated>
      <FindThread />
      <Ask q={questionsS31['ch03/3.1/q-label']} />

      <h2><Translated vi="Một lần khởi chạy có thể lớn đến đâu">How large a launch can be</Translated></h2>
      <Translated
        vi={
          <p>
            Sách đưa ra ba giới hạn. Một khối chứa tối đa 1024 luồng tổng cộng trên các hệ thống hiện nay. Một lưới có tối
            đa 2<sup>31</sup> − 1 khối theo x và tối đa 65,535 theo y và theo z. Sách nói thêm rằng 1024 luồng có thể phân bố trên ba
            chiều theo cách nào cũng được, với bốn ví dụ: (512, 1, 1), (8, 16, 4) và (32, 16, 2) được phép, còn (32, 32, 2) thì
            không, vì tích của nó là 2048.
          </p>
        }
      >
        <p>
          The book gives three limits. A block holds at most 1024 threads in total on current systems. A grid has at
          most 2<sup>31</sup> − 1 blocks along x and at most 65,535 along y and along z. It adds that the 1024 threads
          can be spread over the three dimensions in any way, with four examples: (512, 1, 1), (8, 16, 4) and (32, 16, 2)
          are allowed, and (32, 32, 2) is not, because its product is 2048.
        </p>
      </Translated>
      <Source section="3.1" pages="47" />
      <Translated
        vi={
          <p>
            "Theo cách nào cũng được" là quá lỏng. Mỗi chiều của khối còn có giới hạn riêng, và giới hạn của z thì nhỏ.
          </p>
        }
      >
        <p>
          "In any way" is too loose. Each dimension of a block has a cap of its own as well, and the cap on z is small.
        </p>
      </Translated>
      <Today checked="2026-11-20">
        <Translated
          vi={
            <p>
              Bảng giới hạn của NVIDIA, như nhau cho mọi compute capability mà nó liệt kê: tối đa 1024 luồng trong một khối; một
              khối tối đa 1024 theo x, 1024 theo y và 64 theo z; một lưới tối đa 2<sup>31</sup> − 1 theo x và 65,535 theo y và z.
              Vậy (1, 1, 128) chỉ có 128 luồng mà vẫn không phải một khối hợp lệ. Bốn ví dụ của sách cho cùng kết quả dưới cả hai
              quy tắc.
            </p>
          }
        >
          <p>
            NVIDIA's table of limits, the same for every compute capability it lists: at most 1024 threads in a block; a
            block at most 1024 in x, 1024 in y and 64 in z; a grid at most 2<sup>31</sup> − 1 in x and 65,535 in y and
            z. So (1, 1, 128) has only 128 threads and is not a legal block. The book's four examples come out the same
            under both rules.
          </p>
        </Translated>
      </Today>
      <ShapesTable />
      <Translated
        vi={
          <p>
            Hãy đặt trình khám phá ở trên thành <code>(1, 1, 128)</code> và đổi tiêu chí đánh giá. Rồi hãy coi mọi con số này là
            tài liệu, không phải sự thật về thiết bị của bạn. Cách khả chuyển để có chúng là hỏi: một thiết bị CUDA hoặc HIP báo{' '}
            <code>maxThreadsPerBlock</code>, <code>maxThreadsDim</code> và <code>maxGridSize</code> trong thuộc tính thiết bị của
            nó.
          </p>
        }
      >
        <p>
          Set the explorer above to <code>(1, 1, 128)</code> and switch what it judges by. Then treat all of these
          numbers as documentation, not as facts about your device. The portable way to get them is to ask: a CUDA or
          HIP device reports <code>maxThreadsPerBlock</code>, <code>maxThreadsDim</code> and <code>maxGridSize</code> in
          its device properties.
        </p>
      </Translated>
      <LimitsTable />
      <Ask q={questionsS31['ch03/3.1/q-zcap']} />

      <h2><Translated vi="Điều gì xảy ra khi bạn vượt giới hạn">What happens when you break a limit</Translated></h2>
      <Translated
        vi={
          <p>
            Sách không nói. Điều này đáng biết, vì thất bại trông không giống thất bại: một chương trình với lần khởi chạy không
            hợp lệ vẫn biên dịch, chạy đến cuối và in ra các con số.
          </p>
        }
      >
        <p>
          The book does not say. It is worth knowing, because the failure does not look like one: a program with an
          illegal launch compiles, runs to the end and prints numbers.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Lấy từ tài liệu hiện hành của NVIDIA và AMD. Không có gì trong phần này được chạy trên thiết bị CUDA hay HIP
              (unverified trên thiết bị).
            </p>
          }
        >
          <p>
            From NVIDIA's and AMD's current documentation. Nothing in this part was run on a CUDA or HIP device
            (unverified on device).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Một lần khởi chạy với <code>dimBlock(32, 32, 2)</code> vẫn biên dịch, vì hình dạng là dữ liệu lúc chạy bình thường.
            Câu lệnh khởi chạy không có giá trị, nên không có gì để kiểm tra trên dòng đó. Lúc chạy, lời gọi thất bại. Không
            luồng nào chạy, và buffer đầu ra giữ nguyên thứ nó đang chứa. Runtime ghi thất bại là{' '}
            <code>cudaErrorInvalidConfiguration</code>, giá trị 9, vào một trạng thái lỗi mà nó giữ cho mỗi luồng host, và chương
            trình tiếp tục. Bạn chỉ thấy lỗi nếu bạn hỏi.
          </p>
        }
      >
        <p>
          A launch with <code>dimBlock(32, 32, 2)</code> compiles, because a shape is ordinary run-time data. The launch
          statement has no value, so there is nothing to test on that line. At run time the call fails. No thread runs,
          and the output buffer keeps whatever it held. The runtime records the failure as{' '}
          <code>cudaErrorInvalidConfiguration</code>, value 9, in an error state it keeps for each host thread, and the
          program continues. You see the error only if you ask for it.
        </p>
      </Translated>
      <Today checked="2026-11-20">
        <Translated
          vi={
            <p>
              NVIDIA: "The function call fails if grid_dim or block_dim exceeds the maximum sizes allowed for the device".{' '}
              <code>cudaGetLastError()</code> trả về trạng thái lỗi và đặt lại nó thành <code>cudaSuccess</code>;{' '}
              <code>cudaPeekAtLastError()</code> trả về mà không đặt lại. HIP có cả hai tương ứng, với{' '}
              <code>hipErrorInvalidConfiguration</code>, cũng bằng 9, và <code>hipGetLastError()</code>.
            </p>
          }
        >
          <p>
            NVIDIA: "The function call fails if grid_dim or block_dim exceeds the maximum sizes allowed for the device".{' '}
            <code>cudaGetLastError()</code> returns the error state and resets it to <code>cudaSuccess</code>;{' '}
            <code>cudaPeekAtLastError()</code> returns it without resetting. HIP mirrors both, with{' '}
            <code>hipErrorInvalidConfiguration</code>, also 9, and <code>hipGetLastError()</code>.
          </p>
        </Translated>
      </Today>
      <Code
        tabs={[{ label: 'CUDA', code: checked ?? tr('// loading…', '// đang tải…'), status: 'front-end-checked', tiers: CODE_TIERS }]}
        highlight={[11, 12]}
        caption={tr(
          "Asking after the launch. On the course's CPU emulator, blockShapeIsRejected returns true and the buffer is left untouched (emulator-checked: python3 labs/ch03/snippets/check.py). The emulator also prints a line of its own; the documentation describes no message from a device.",
          'Hỏi sau lần khởi chạy. Trên trình giả lập CPU của khóa học, blockShapeIsRejected trả về true và buffer không bị động đến (emulator-checked: python3 labs/ch03/snippets/check.py). Trình giả lập cũng in một dòng của riêng nó; tài liệu không mô tả thông báo nào từ thiết bị.',
        )}
      />
      <Translated
        vi={
          <p>
            PyTorch làm điều này sau mỗi lần khởi chạy. Các kernel CUDA của nó được theo sau bởi macro{' '}
            <code>C10_CUDA_KERNEL_LAUNCH_CHECK()</code>, được định nghĩa là <code>C10_CUDA_CHECK(cudaGetLastError())</code>
            . Hãy làm như vậy trong các hàm khởi chạy của chính bạn. Trình giả lập của khóa học áp dụng các giới hạn theo tài
            liệu của NVIDIA, theo từng chiều cũng như tổng cộng: một khối (1, 1, 128) và một lưới 65,536 khối theo y thất bại ở
            đó với cùng lỗi (emulator-checked bằng cùng script đó). Đó là các giới hạn theo tài liệu, không phải của thiết bị
            bạn; một thiết bị thật cũng có thể từ chối một lần khởi chạy nhỏ hơn, như phần so sánh dưới đây giải thích.
          </p>
        }
      >
        <p>
          PyTorch does this after every launch. Its CUDA kernels are followed by the macro{' '}
          <code>C10_CUDA_KERNEL_LAUNCH_CHECK()</code>, which is defined as <code>C10_CUDA_CHECK(cudaGetLastError())</code>
          . Do the same in your own launch functions. The course's emulator applies NVIDIA's documented limits, per
          dimension as well as in total: a block of (1, 1, 128) and a grid of 65,536 blocks along y fail there with the
          same error (emulator-checked by the same script). Those are the documented limits, not your device's; a
          real device can also refuse a smaller launch, as the comparison below explains.
        </p>
      </Translated>
      <Ask q={questionsS31['ch03/3.1/q-silent']} />

      <h2><Translated vi="Cùng lần khởi chạy trên HIP và Metal">The same launch on HIP and Metal</Translated></h2>
      <Translated
        vi={
          <p>
            Trên HIP, các dòng host của mục này là cùng một mã nguồn. Trên Metal, mọi ý tưởng đều có bản tương ứng và mọi cái
            tên đều khác.
          </p>
        }
      >
        <p>
          On HIP the host lines of this section are the same source. On Metal every idea has a counterpart and every
          name is different.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ nói về CUDA. Phần so sánh này là của khóa học này, từ tài liệu và header hiện hành của AMD và Apple. "Không
              có tài liệu" nghĩa là chúng tôi đã tìm và không thấy nhận định nào.
            </p>
          }
        >
          <p>
            The book covers CUDA only. This comparison is ours, from AMD's and Apple's current documentation and
            headers. "Not documented" means we looked and found no statement.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <table className="data" data-table="backends">
            <thead>
              <tr>
                <th />
                <th>CUDA</th>
                <th>HIP</th>
                <th>Metal</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <b>Kiểu kích thước</b>
                </td>
                <td>
                  <code>dim3</code>; các thành phần mặc định là 1
                </td>
                <td>
                  <code>dim3</code>, cùng trường, cùng mặc định
                </td>
                <td>
                  <code>MTLSize</code> với <code>width</code>, <code>height</code>, <code>depth</code>; không có mặc định, các
                  chiều không dùng phải là 1
                </td>
              </tr>
              <tr>
                <td>
                  <b>Khởi chạy</b>
                </td>
                <td>
                  <code>kernel&lt;&lt;&lt;grid, block&gt;&gt;&gt;(args)</code>
                </td>
                <td>cùng câu lệnh</td>
                <td>
                  trên một compute encoder: <code>dispatchThreadgroups</code> nhận số nhóm, như CUDA;{' '}
                  <code>dispatchThreads</code> nhận tổng lưới luồng
                </td>
              </tr>
              <tr>
                <td>
                  <b>Tọa độ trong kernel</b>
                </td>
                <td>
                  biến dựng sẵn; <code>gridDim</code> và <code>blockDim</code> là <code>dim3</code>,{' '}
                  <code>blockIdx</code> và <code>threadIdx</code> là <code>uint3</code>
                </td>
                <td>
                  cùng bốn cái tên, mỗi cái có kiểu riêng: hãy dùng <code>.x</code>, <code>.y</code>, <code>.z</code>{' '}
                  và đừng bao giờ sao chép nguyên một biến dựng sẵn
                </td>
                <td>
                  tham số kernel có thuộc tính: <code>threadgroups_per_grid</code>, <code>threads_per_threadgroup</code>,{' '}
                  <code>threadgroup_position_in_grid</code>, <code>thread_position_in_threadgroup</code>
                </td>
              </tr>
              <tr>
                <td>
                  <b>Số luồng trong một khối hoặc nhóm</b>
                </td>
                <td>
                  1024 cho thiết bị; theo từng chiều 1024, 1024, 64. Một kernel dùng nhiều thanh ghi có thể thất bại dưới 1024
                  với <code>cudaErrorLaunchOutOfResources</code>
                </td>
                <td>
                  1024; theo từng chiều 1024, 1024, 1024 trong bảng của AMD (hãy đọc <code>maxThreadsDim</code> trên thiết bị
                  mới hơn). Một kernel biên dịch với <code>__launch_bounds__</code> từ chối khối lớn hơn
                </td>
                <td>
                  đặt cho từng kernel đã biên dịch: <code>maxTotalThreadsPerThreadgroup</code> của pipeline state
                </td>
              </tr>
              <tr>
                <td>
                  <b>Kích thước lưới</b>
                </td>
                <td>
                  x tới 2<sup>31</sup> − 1; y và z tới 65,535
                </td>
                <td>
                  x tới 2<sup>32</sup> − 1, y và z tới 65,535 trong bảng của AMD; hãy đọc <code>maxGridSize</code>
                </td>
                <td>không có tài liệu</td>
              </tr>
              <tr>
                <td>
                  <b>Một lần khởi chạy vượt giới hạn</b>
                </td>
                <td>
                  <code>cudaErrorInvalidConfiguration</code>, thấy được qua <code>cudaGetLastError()</code>
                </td>
                <td>
                  <code>hipErrorInvalidConfiguration</code>, thấy được qua <code>hipGetLastError()</code>
                </td>
                <td>không có tài liệu</td>
              </tr>
            </tbody>
          </table>
        }
      >
        <table className="data" data-table="backends">
          <thead>
            <tr>
              <th />
              <th>CUDA</th>
              <th>HIP</th>
              <th>Metal</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <b>Size type</b>
              </td>
              <td>
                <code>dim3</code>; components default to 1
              </td>
              <td>
                <code>dim3</code>, same fields, same default
              </td>
              <td>
                <code>MTLSize</code> with <code>width</code>, <code>height</code>, <code>depth</code>; no defaults, unused
                ones must be 1
              </td>
            </tr>
            <tr>
              <td>
                <b>Launch</b>
              </td>
              <td>
                <code>kernel&lt;&lt;&lt;grid, block&gt;&gt;&gt;(args)</code>
              </td>
              <td>the same statement</td>
              <td>
                on a compute encoder: <code>dispatchThreadgroups</code> takes a count of groups, as CUDA does;{' '}
                <code>dispatchThreads</code> takes the total thread grid
              </td>
            </tr>
            <tr>
              <td>
                <b>Coordinates in the kernel</b>
              </td>
              <td>
                built-in variables; <code>gridDim</code> and <code>blockDim</code> are <code>dim3</code>,{' '}
                <code>blockIdx</code> and <code>threadIdx</code> are <code>uint3</code>
              </td>
              <td>
                the same four names, each with a type of its own: use <code>.x</code>, <code>.y</code>, <code>.z</code>{' '}
                and never copy a whole built-in
              </td>
              <td>
                attributed kernel parameters: <code>threadgroups_per_grid</code>, <code>threads_per_threadgroup</code>,{' '}
                <code>threadgroup_position_in_grid</code>, <code>thread_position_in_threadgroup</code>
              </td>
            </tr>
            <tr>
              <td>
                <b>Threads in a block or group</b>
              </td>
              <td>
                1024 for the device; per dimension 1024, 1024, 64. A kernel that uses many registers can fail below 1024
                with <code>cudaErrorLaunchOutOfResources</code>
              </td>
              <td>
                1024; per dimension 1024, 1024, 1024 in AMD's table (read <code>maxThreadsDim</code> on a newer device).
                A kernel compiled with <code>__launch_bounds__</code> refuses a larger block
              </td>
              <td>
                set for each compiled kernel: <code>maxTotalThreadsPerThreadgroup</code> of the pipeline state
              </td>
            </tr>
            <tr>
              <td>
                <b>Grid size</b>
              </td>
              <td>
                x up to 2<sup>31</sup> − 1; y and z up to 65,535
              </td>
              <td>
                x up to 2<sup>32</sup> − 1, y and z up to 65,535 in AMD's table; read <code>maxGridSize</code>
              </td>
              <td>not documented</td>
            </tr>
            <tr>
              <td>
                <b>A launch over a limit</b>
              </td>
              <td>
                <code>cudaErrorInvalidConfiguration</code>, seen through <code>cudaGetLastError()</code>
              </td>
              <td>
                <code>hipErrorInvalidConfiguration</code>, seen through <code>hipGetLastError()</code>
              </td>
              <td>not documented</td>
            </tr>
          </tbody>
        </table>
      </Translated>
      <Translated
        vi={
          <p>
            Có hai hàng đáng xem lại. Giới hạn luồng là mức trần cho thiết bị, và trên cả ba backend một kernel có thể nằm dưới
            mức đó. Một kernel CUDA cần nhiều thanh ghi mỗi luồng có thể không khởi chạy được với ít hơn 1024 luồng. Một kernel
            HIP biên dịch với launch bound từ chối khối lớn hơn. Một pipeline state của Metal báo giới hạn cho chính kernel đã
            biên dịch của nó. Vậy "hình dạng này có hợp lệ không" là câu hỏi về một kernel cũng như về một thiết bị; mục 4.7
            quay lại chuyện thanh ghi. Và bốn tham số Metal ở hàng thứ ba, theo thứ tự đã liệt kê, là <code>gridDim</code>,{' '}
            <code>blockDim</code>, <code>blockIdx</code> và <code>threadIdx</code>. Metal viết vị trí với x trước và không có
            nhãn (z, y, x).
          </p>
        }
      >
        <p>
          Two rows deserve a second look. The thread limit is a ceiling for the device, and on all three backends one
          kernel can sit below it. A CUDA kernel that needs many registers per thread can fail to launch with fewer than
          1024 threads. A HIP kernel compiled with a launch bound refuses a larger block. A Metal pipeline state reports
          the limit for its own compiled kernel. So "is this shape legal" is a question about a kernel as well as a
          device; section 4.7 comes back to registers. And the four Metal parameters in the third row are, in the order
          listed, <code>gridDim</code>, <code>blockDim</code>, <code>blockIdx</code> and <code>threadIdx</code>. Metal
          writes positions x first and has no (z, y, x) labels.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch08-facts.md, A4, B1, B4; ch03-facts.md, B3, C4)">{/* i18n-ignore: date and notes reference */}
        <Translated
          vi={
            <p>
              NVIDIA: <code>cudaErrorLaunchOutOfResources</code> bao gồm trường hợp "too many threads for the kernel's register
              count". AMD: một lần khởi chạy được kiểm tra theo <code>__launch_bounds__</code> của kernel, và{' '}
              <code>gridDim</code> nhân <code>blockDim</code> phải vừa trong 32 bit theo mỗi chiều. Apple: giới hạn của pipeline
              "varies depending on the complexity of the function".
            </p>
          }
        >
          <p>
            NVIDIA: <code>cudaErrorLaunchOutOfResources</code> covers "too many threads for the kernel's register
            count". AMD: a launch is checked against the kernel's <code>__launch_bounds__</code>, and{' '}
            <code>gridDim</code> times <code>blockDim</code> must fit in 32 bits per dimension. Apple: the pipeline's
            limit "varies depending on the complexity of the function".
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS31['ch03/3.1/q-metal-size']} />

      <h2><Translated vi="Bạn sẽ gặp điều này ở đâu: một lưới 3D trên một tensor 4D">Where you will meet this: a 3D grid over a 4D tensor</Translated></h2>
      <Translated
        vi={
          <p>
            llama.cpp đệm một tensor bằng một kernel mà lần khởi chạy dùng mọi thứ trong mục này. Một tensor ggml có bốn kích
            thước, <code>ne0</code> đến <code>ne3</code>, với <code>ne0</code> là chiều trong cùng, chiều thay đổi nhanh nhất
            trong bộ nhớ. Một lưới có ba chiều. Đây là cách bốn chiều được xếp vào ba.
          </p>
        }
      >
        <p>
          llama.cpp pads a tensor with a kernel whose launch uses everything in this section. A ggml tensor has four
          sizes, <code>ne0</code> to <code>ne3</code>, with <code>ne0</code> the innermost one, the one that varies
          fastest in memory. A grid has three dimensions. This is how four are fitted onto three.
        </p>
      </Translated>
      <Code
        tabs={[{ label: 'llama.cpp', code: PAD_KERNEL + '\n    ...\n' + PAD_LAUNCH }]}
        highlight={[7]}
        caption={tr(
          "Quoted from llama.cpp, ggml/src/ggml-cuda/pad.cu (master at b94959a, read 2026-11-20; blockIDx is spelled as in the source). Not our code and not compiled here. The comment is from the kernel, the three lines after the ellipsis from its host function.",
          'Trích từ llama.cpp, ggml/src/ggml-cuda/pad.cu (master tại b94959a, đọc ngày 2026-11-20; blockIDx được viết như trong mã nguồn). Không phải mã của chúng tôi và không được biên dịch ở đây. Chú thích lấy từ kernel, ba dòng sau dấu ba chấm lấy từ hàm host của nó.',
        )}
      />
      <Translated
        vi={
          <p>
            Chiều trong cùng của tensor nằm trên x, chia thành các khối bằng phép làm tròn lên số nguyên từ trình khám phá định
            kích thước. Chiều tiếp theo, <code>ne1</code>, được mỗi chỉ số một khối theo y. Hai chiều ngoài cùng được gộp thành
            một số duy nhất, <code>i3*ne2 + i2</code>, theo z. Hai lời gọi <code>std::min</code> ở dòng 7 chính là giới hạn 65,535
            của mục này trong mã sản phẩm: khi <code>ne1</code> hoặc <code>ne2 * ne3</code> lớn hơn, lưới dừng ở mức trần và mỗi
            khối lặp qua các chỉ số mà nó phải phủ. Cùng tệp này được biên dịch cho thiết bị AMD.
          </p>
        }
      >
        <p>
          The innermost tensor dimension goes on x, split into blocks with the integer round-up from the sizing explorer.
          The next one, <code>ne1</code>, gets one block per index along y. The two outer dimensions are folded into a
          single number, <code>i3*ne2 + i2</code>, along z. The two <code>std::min</code> calls on line 7 are the 65,535
          limit of this section in production code: when <code>ne1</code> or <code>ne2 * ne3</code> is larger, the grid
          stops at the cap and each block loops over the indices it has to cover. The same file is compiled for AMD
          devices.
        </p>
      </Translated>
      <Today checked="2026-11-20">
        <Translated
          vi={
            <p>
              Mô hình của chương này, mỗi luồng một phần tử với phép tính chỉ số viết bằng tay, là thứ hướng dẫn của NVIDIA nay
              gọi là kernel SIMT. Hướng dẫn cũng mô tả một mô hình thứ hai, kernel tile, trong đó trình biên dịch chọn kích thước
              khối và bốn biến dựng sẵn này không tồn tại. Khóa học này không dùng nó.
            </p>
          }
        >
          <p>
            The model of this chapter, one thread per element with the index arithmetic written by hand, is what
            NVIDIA's guide now calls SIMT kernels. The guide also documents a second model, tile kernels, in which the
            compiler picks the block size and these four built-ins do not exist. This course does not use it.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Kiểm tra: một lần khởi chạy, đọc từ mã host">Check: one launch, read from host code</Translated></h2>
      <Ask q={questionsS31['ch03/3.1/q-ex3-threads']} />
      <Translated
        vi={
          <p>
            Tiếp theo, <a href={href('ch03', '3.2')}>mục 3.2</a> biến tọa độ của một luồng thành một hàng và một cột, rồi biến
            chúng thành một offset duy nhất trong một mảng phẳng.
          </p>
        }
      >
        <p>
          Next, <a href={href('ch03', '3.2')}>section 3.2</a> turns a thread's coordinates into a row and a column, and
          those into one offset into a flat array.
        </p>
      </Translated>
    </>
  );
}
