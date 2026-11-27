import { useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  blockDependencies,
  dedupFlags,
  dedupMoves,
  moveByIndex,
  type Dependencies,
  type MatrixPattern,
  type Move,
} from '../../sim/inplace';

// Section 12.8, p. 301. Every destination, flag and dependency shown comes from sim/inplace.ts.

export const questionsS128: Record<string, Question> = {
  'ch12/12.8/q-dedup': {
    kind: 'numeric',
    id: 'ch12/12.8/q-dedup',
    prompt: (
      <Translated vi={<>Loại bỏ các phần tử trùng lặp khỏi danh sách đã sắp xếp <code>[3, 3, 5, 7, 7, 7, 9]</code>. Còn lại bao nhiêu khóa?</>}>
        Remove the duplicates from the sorted list <code>[3, 3, 5, 7, 7, 7, 9]</code>. How many keys remain?
      </Translated>
    ),
    answer: 4,
    unit: 'keys',
    vi: { unit: 'khóa' },
    explain: (
      <Translated
        vi={
          <p>
            Các cờ giữ lại là 1 0 1 1 0 0 1: một khóa được giữ khi nó khác khóa đứng trước, và khóa đầu tiên luôn được giữ. Scan
            loại trừ của chúng cho các khóa còn lại 3, 5, 7 và 9 các ô 0 đến 3. Đó là một bộ lọc ổn định có phép kiểm tra nhìn
            sang phần tử bên cạnh.
          </p>
        }
      >
        <p>
          The keep flags are 1 0 1 1 0 0 1: a key is kept when it differs from the key before it, and the first key is
          always kept. Their exclusive scan gives the survivors 3, 5, 7 and 9 the slots 0 to 3. It is a stable filter
          whose test looks at a neighbour.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.8, p. 301',
  },
  'ch12/12.8/q-noscan': {
    kind: 'choice',
    id: 'ch12/12.8/q-noscan',
    prompt: (
      <Translated vi={<>Vì sao xóa các cột của một ma trận theo chỉ số không cần scan?</>}>
        <>Why does removing columns of a matrix by index need no scan?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Vì các cột của ma trận nằm liền nhau trong bộ nhớ, nên một khối có thể chuyển mỗi cột bằng một phép sao chép thường.">Because the columns of a matrix are contiguous in memory, so a block can move each one with a plain copy.</Translated>,
        why: <Translated vi="Trong ma trận thứ tự hàng (row-major), một cột có bước nhảy (stride): các phần tử của nó cách nhau trọn một hàng.">In a row-major matrix a column is strided: its elements are a whole row apart.</Translated>,
      },
      {
        label: <Translated vi="Nó có cần scan, trên các cờ giữ lại của mọi phần tử.">It does need a scan, of the keep flags of every element.</Translated>,
        why: <Translated vi="Nó có thể dùng scan, nhưng không có gì trong dữ liệu quyết định ai còn lại, nên số đếm trước mỗi khóa đã biết từ chỉ số.">It could use one, but nothing about the data decides who survives, so the count before each key is known from the index.</Translated>,
      },
      {
        label: (
          <Translated vi="Một khóa có còn lại hay không, và có bao nhiêu cột bị xóa đứng trước nó, chỉ phụ thuộc vào chỉ số cột của nó, nên đích đến của nó là một công thức đóng.">
            Whether a key survives, and how many removed columns come before it, depend only on its column index, so its destination is a closed formula.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Scan tồn tại để đếm những phần tử còn lại do dữ liệu quyết định. Khi chỉ số quyết định, số cột bị xóa bên trái cột{' '}
            <i>c</i> là một hằng số cho mỗi cột, và chỉ số phẳng mới là{' '}
            <code>r × (C − m) + (c − removed before c)</code>.
          </p>
        }
      >
        <p>
          A scan exists to count survivors that the data decides. When the index decides, the count of removed columns
          to the left of column <i>c</i> is a constant per column, and the new flat index is{' '}
          <code>r × (C − m) + (c − removed before c)</code>.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.8, p. 301',
  },
  'ch12/12.8/q-pad': {
    kind: 'choice',
    id: 'ch12/12.8/q-pad',
    prompt: (
      <Translated
        vi={
          <>
            Bạn đệm mỗi hàng của một ma trận thứ tự hàng từ độ rộng 6 lên độ rộng 8, tại chỗ, mỗi khối luồng một hàng. Cần
            thứ tự nào giữa các khối?
          </>
        }
      >
        You pad every row of a row-major matrix from width 6 to width 8, in place, one row per thread block. Which
        ordering between blocks is required?
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Cùng thứ tự như khi lọc: mỗi khối chờ khối đứng trước nó, để các khối trước đọc xong trước khi các khối sau ghi.">The same order as filtering: each block waits for the block before it, so earlier blocks read before later ones write.</Translated>,
        why: <Translated vi="Điều đó bảo vệ các khóa khỏi những lần ghi đến từ phía sau. Ở đây các lần ghi đến từ phía trước.">That protects keys from writes coming from behind. Here the writes come from in front.</Translated>,
      },
      {
        label: (
          <Translated vi="Mỗi khối phải đọc xong trước khi bất kỳ khối nào đứng trước nó ghi, vì các khóa dịch ra ngoài: chuỗi chạy từ khối cuối về khối đầu.">
            Each block must finish reading before any earlier block writes, because keys move outwards: the chain runs from the last block to the first.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Không cần: không có gì bị xóa, nên không có gì bị ghi đè.">None: nothing is removed, so nothing is overwritten.</Translated>,
        why: <Translated vi="Hàng 1 chuyển từ các ô 6 đến 11 sang các ô 8 đến 13, mà các ô 12 và 13 vẫn đang giữ hàng 2, có thể chưa được đọc.">Row 1 moves from slots 6 to 11 into slots 8 to 13, and slots 12 and 13 still hold row 2, which may not have been read yet.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hàng <i>r</i> chuyển từ <code>6r</code> sang <code>8r</code>, nên nó rơi vào các ô của những hàng sau nó. Những hàng
            đó phải được đọc trước. Cách đồng bộ hóa một chiều của sách vẫn dùng được, chạy từ khối có các khóa cuối về phía khối
            có các khóa đầu.
          </p>
        }
      >
        <p>
          Row <i>r</i> moves from <code>6r</code> to <code>8r</code>, so it lands on the slots of rows after it. Those
          rows must be read first. The book's one-directional synchronisation still works, run from the block with the
          last keys towards the block with the first.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.8, p. 301 (with the erratum corrected)',
    vi: { source: 'ch. 12 §12.8, p. 301 (đã sửa lỗi in)' },
  },
  'ch12/12.8/q-order-only': {
    kind: 'choice',
    id: 'ch12/12.8/q-order-only',
    prompt: (
      <Translated vi={<>Bạn xóa các hàng của một ma trận theo chỉ số, tại chỗ, dùng cơ chế giữa các khối của scan một lượt. Mỗi khối chuyển gì cho khối kế tiếp?</>}>
        <>You remove rows of a matrix by index, in place, with the single-pass scan's machinery between blocks. What does each block pass on to the next?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Chỉ một cờ báo “tôi đã đọc xong”; không có tổng một phần.">Only a flag that says "I have finished reading"; no partial sum.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Số hàng nó giữ, cộng vào một tổng tích lũy, đúng như trong scan của bộ lọc ổn định.">The number of rows it kept, added to a running total, exactly as in the stable filter's scan.</Translated>,
        why: <Translated vi="Khối nào cũng tính được điều đó từ chỉ số của các hàng bị xóa. Không ai cần nó được chuyển tiếp.">Every block can compute that from the indices of the removed rows. Nobody needs it passed on.</Translated>,
      },
      {
        label: <Translated vi="Không gì cả: các đích đến đã biết, nên không khối nào phụ thuộc khối khác.">Nothing at all: the destinations are known, so no block depends on another.</Translated>,
        why: <Translated vi="Đích đến đã biết loại bỏ được scan, không loại bỏ được hiểm họa. Một khối sau vẫn không được ghi trước khi khối trước đã đọc.">Known destinations remove the scan, not the hazard. A later block still must not write before an earlier block has read.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Scan của mục 12.7 làm hai việc cùng lúc: nó tính độ lệch và nó sắp thứ tự các khối. Khi chỉ số cho sẵn độ lệch, chỉ
            còn việc sắp thứ tự, và riêng cờ làm được việc đó, với cùng release và acquire như trước.
          </p>
        }
      >
        <p>
          The scan of section 12.7 did two jobs at once: it computed offsets and it ordered the blocks. When the index
          gives the offsets, only the ordering job is left, and the flag alone does it, with the same release and acquire
          as before.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.8, p. 301',
  },
};

// ---------------------------------------------------------------- the explorer

type Kind = 'dedup' | 'removeRows' | 'removeCols' | 'padCols';
const KINDS: { kind: Kind; label: string }[] = [
  {
    kind: 'dedup',
    get label() {
      return tr('remove duplicates', 'loại bỏ trùng lặp');
    },
  },
  {
    kind: 'removeRows',
    get label() {
      return tr('remove rows', 'xóa hàng');
    },
  },
  {
    kind: 'removeCols',
    get label() {
      return tr('remove columns', 'xóa cột');
    },
  },
  {
    kind: 'padCols',
    get label() {
      return tr('pad columns', 'đệm cột');
    },
  },
];
const DEFAULT_KEYS = '1, 1, 2, 3, 3, 3, 4, 5, 5, 6, 7, 7';

function parseKeys(s: string): { keys: number[]; error: string | null } {
  const parts = s.split(/[\s,]+/).filter(Boolean);
  const keys = parts.map(Number);
  if (!keys.length) return { keys: [], error: tr('Type at least one key.', 'Hãy nhập ít nhất một khóa.') };
  if (keys.some((k) => !Number.isFinite(k))) return { keys: [], error: tr('Keys must be numbers.', 'Khóa phải là số.') };
  if (keys.length > 24) return { keys: [], error: tr('At most 24 keys.', 'Tối đa 24 khóa.') };
  if (keys.some((k, i) => i > 0 && k < keys[i - 1])) return { keys: [], error: tr('The list must be sorted: each key at least the one before it.', 'Danh sách phải được sắp xếp: mỗi khóa ít nhất bằng khóa đứng trước.') };
  return { keys, error: null };
}

function DirectionPanel({ deps, inPlace, unit }: { deps: Dependencies; inPlace: boolean; unit: string }) {
  if (!inPlace)
    return (
      <Translated
        vi={
          <p data-part="direction" data-direction="out-of-place">
            Khi không làm tại chỗ, mọi lần ghi đều vào mảng thứ hai, nơi không có khóa nào chờ được đọc. Không cần thứ tự
            nào giữa các khối.
          </p>
        }
      >
        <p data-part="direction" data-direction="out-of-place">
          Out of place, every write goes to the second array, where no key waits to be read. No order between blocks is
          needed.
        </p>
      </Translated>
    );
  const label =
    deps.direction === 'forward'
      ? tr('earlier blocks must read before later blocks write: run the chain from the first block to the last', 'các khối trước phải đọc trước khi các khối sau ghi: chạy chuỗi từ khối đầu tới khối cuối')
      : deps.direction === 'backward'
        ? tr('later blocks must read before earlier blocks write: run the chain from the last block to the first', 'các khối sau phải đọc trước khi các khối trước ghi: chạy chuỗi từ khối cuối về khối đầu')
        : deps.direction === 'none'
          ? tr('no block writes over another block\'s keys: no chain needed', 'không khối nào ghi đè lên khóa của khối khác: không cần chuỗi')
          : tr('dependencies run both ways', 'phụ thuộc chạy theo cả hai chiều');
  return (
    <div data-part="direction" data-direction={deps.direction}>
      <p style={{ margin: '0 0 0.4rem' }}>
        <span className={'chip' + (deps.direction === 'none' ? ' ok' : deps.direction === 'mixed' ? ' fail' : '')}>
          {deps.direction === 'forward' ? tr('forward', 'xuôi') : deps.direction === 'backward' ? tr('backward', 'ngược') : tr(deps.direction, { none: 'không có', mixed: 'hỗn hợp' }[deps.direction])}
        </span>{' '}
        {label}.
      </p>
      {deps.edges.length ? (
        <ul style={{ margin: '0 0 0.4rem', paddingLeft: '1.2rem' }} data-part="edges">
          {deps.edges.map((e) => (
            <li key={`${e.reader}-${e.writer}`} data-edge={`${e.reader}>${e.writer}`}>
              <Translated
                vi={
                  <>
                    khối {e.reader} đọc trước khi khối {e.writer} ghi ({e.writes} {unit === 'element' ? 'phần tử' : 'khóa'} rơi vào
                    vùng của khối {e.reader}) {e.reader < e.writer ? '→' : '←'}
                  </>
                }
              >
                block {e.reader} reads before block {e.writer} writes ({e.writes} {unit}
                {e.writes === 1 ? ' lands' : 's land'} in block {e.reader}'s range) {e.reader < e.writer ? '→' : '←'}
              </Translated>
            </li>
          ))}
        </ul>
      ) : null}
      <p className="muted" style={{ margin: 0 }}>
        <Translated
          vi={
            <>
              {deps.within} lần ghi bên trong chính khối của bên ghi (một rào chắn sắp thứ tự chúng)
              {deps.fresh ? `; ${deps.fresh} lần vào vùng nhớ mới sau điểm kết thúc ban đầu` : ''}.
            </>
          }
        >
          {deps.within} write{deps.within === 1 ? '' : 's'} inside the writer's own block (a barrier orders them)
          {deps.fresh ? `; ${deps.fresh} into new memory past the original end` : ''}.
        </Translated>
      </p>
    </div>
  );
}

function RelatedExplorer() {
  const [kind, setKind] = useState<Kind>('removeCols');
  const [rows, setRows] = useState(4);
  const [cols, setCols] = useState(6);
  const [rmRows, setRmRows] = useState<number[]>([1]);
  const [rmCols, setRmCols] = useState<number[]>([2]);
  const [pad, setPad] = useState(2);
  const [rowBlocks, setRowBlocks] = useState(4);
  const [inPlace, setInPlace] = useState(true);
  const [keysText, setKeysText] = useState(DEFAULT_KEYS);
  const [perBlock, setPerBlock] = useState(4);
  const [sel, setSel] = useState<number | null>(null);

  const parsed = parseKeys(keysText);
  const isMatrix = kind !== 'dedup';
  const pattern: MatrixPattern | null = !isMatrix
    ? null
    : kind === 'removeRows'
      ? { kind, rows, cols, remove: rmRows.filter((r) => r < rows) }
      : kind === 'removeCols'
        ? { kind, rows, cols, remove: rmCols.filter((c) => c < cols) }
        : { kind: 'padCols', rows, cols, pad };
  const moved = pattern ? moveByIndex(pattern) : null;
  const rowsPerBlock = Math.ceil(rows / Math.min(rowBlocks, rows));
  const dedup = parsed.error ? null : dedupFlags(parsed.keys);
  const moves: Move[] = isMatrix ? moved!.moves : dedup ? dedupMoves(parsed.keys) : [];
  const srcSize = isMatrix ? rows * cols : parsed.keys.length;
  const blockSize = isMatrix ? rowsPerBlock * cols : perBlock;
  const deps = blockDependencies(moves, srcSize, blockSize);
  const selMove = sel == null ? undefined : moves.find((m) => m.src === sel);

  const toggle = (list: number[], v: number) => (list.includes(v) ? list.filter((x) => x !== v) : [...list, v].sort((a, b) => a - b));
  const pick = (k: Kind) => {
    setKind(k);
    setSel(null);
  };

  return (
    <Bench
      title={tr('Which way does the dependency run?', 'Phụ thuộc chạy theo chiều nào?')}
      task={tr(
        'Predict the direction for each pattern before you switch in-place on. Then click cells to check destinations.',
        'Dự đoán chiều cho mỗi mẫu trước khi bật chế độ tại chỗ. Sau đó bấm vào các ô để kiểm tra đích đến.',
      )}
      foot={
        isMatrix
          ? tr('Row-major storage. Each cell of the source shows the flat index it moves to; click one for the formula.', 'Lưu theo thứ tự hàng (row-major). Mỗi ô của nguồn hiện chỉ số phẳng mà nó chuyển tới; bấm vào một ô để xem công thức.')
          : tr('Each key is kept when it differs from the key before it. The offsets are the exclusive scan of the keep flags.', 'Mỗi khóa được giữ khi nó khác khóa đứng trước. Các độ lệch là scan loại trừ của các cờ giữ lại.')
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Pattern', 'Mẫu')}>
          {KINDS.map((k) => (
            <button key={k.kind} aria-pressed={kind === k.kind} onClick={() => pick(k.kind)}>{k.label}</button>
          ))}
        </span>
        <label className="check"><input type="checkbox" checked={inPlace} onChange={(e) => setInPlace(e.target.checked)} /> <Translated vi="tại chỗ">in place</Translated></label>
      </div>

      {isMatrix ? (
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <label className="field">
            {tr('Rows', 'Số hàng')}
            <select value={rows} onChange={(e) => { setRows(+e.target.value); setSel(null); }} aria-label={tr('rows', 'số hàng')}>
              {[2, 3, 4, 5, 6, 7, 8].map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <label className="field">
            {tr('Columns', 'Số cột')}
            <select value={cols} onChange={(e) => { setCols(+e.target.value); setSel(null); }} aria-label={tr('columns', 'số cột')}>
              {[2, 3, 4, 5, 6, 7, 8].map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <label className="field">
            {tr('Row blocks', 'Số khối hàng')}
            <select value={Math.min(rowBlocks, rows)} onChange={(e) => setRowBlocks(+e.target.value)} aria-label={tr('row blocks', 'số khối hàng')}>
              {Array.from({ length: rows }, (_, i) => i + 1).map((v) => (
                <Translated key={v} vi={<option value={v}>{v} (mỗi khối {Math.ceil(rows / v)} hàng)</option>}>
                  <option key={v} value={v}>{v} ({Math.ceil(rows / v)} row{Math.ceil(rows / v) === 1 ? '' : 's'} each)</option>
                </Translated>
              ))}
            </select>
          </label>
          {kind === 'padCols' ? (
            <label className="field">
              {tr('Padding per row', 'Phần đệm mỗi hàng')}
              <select value={pad} onChange={(e) => setPad(+e.target.value)} aria-label={tr('padding', 'phần đệm')}>
                {[1, 2, 3].map((v) => <option key={v} value={v}>{v} {tr(`column${v === 1 ? '' : 's'}`, 'cột')}</option>)}
              </select>
            </label>
          ) : null}
          {kind === 'removeRows' ? <span className="muted"><Translated vi="Bấm vào số hàng để xóa hoặc giữ nó.">Click a row number to remove or keep it.</Translated></span> : null}
          {kind === 'removeCols' ? <span className="muted"><Translated vi="Bấm vào số cột để xóa hoặc giữ nó.">Click a column number to remove or keep it.</Translated></span> : null}
        </div>
      ) : (
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <label className="field" style={{ minWidth: 'min(22rem, 100%)' }}>
            {tr('Sorted keys', 'Khóa đã sắp xếp')}
            <input type="text" value={keysText} onChange={(e) => setKeysText(e.target.value)} aria-label={tr('sorted keys', 'khóa đã sắp xếp')} />
          </label>
          <label className="field">
            {tr('Keys per block', 'Số khóa mỗi khối')}
            <select value={perBlock} onChange={(e) => setPerBlock(+e.target.value)} aria-label={tr('keys per block', 'số khóa mỗi khối')}>
              {[2, 3, 4, 6, 8].map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          {parsed.error ? <span className="t-fail" data-part="error">{parsed.error}</span> : null}
        </div>
      )}

      {isMatrix && moved ? (
        <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
          <div>
            <div className="mx-label t-global"><Translated vi={<>nguồn, {rows} × {cols}: mỗi phần tử đi đâu</>}>source, {rows} × {cols}: where each element goes</Translated></div>
            <div className="mx global" style={{ gridTemplateColumns: `auto repeat(${cols}, auto)` }} data-part="source">
              <span className="void" />
              {Array.from({ length: cols }, (_, c) =>
                kind === 'removeCols' ? (
                  <span key={'h' + c} className={rmCols.includes(c) ? 'hot' : 'on'} role="button" tabIndex={0} style={{ cursor: 'pointer' }}
                    data-col-head={c} aria-label={tr(`column ${c}`, `cột ${c}`)} onClick={() => { setRmCols(toggle(rmCols, c)); setSel(null); }}
                    onKeyDown={(e) => e.key === 'Enter' && setRmCols(toggle(rmCols, c))}>{rmCols.includes(c) ? `c${c} ✕` : `c${c}`}</span>
                ) : (
                  <span key={'h' + c} className="void">c{c}</span>
                ),
              )}
              {Array.from({ length: rows }, (_, r) => [
                kind === 'removeRows' ? (
                  <span key={'r' + r} className={rmRows.includes(r) ? 'hot' : 'on'} role="button" tabIndex={0} style={{ cursor: 'pointer' }}
                    data-row-head={r} aria-label={tr(`row ${r}`, `hàng ${r}`)} onClick={() => { setRmRows(toggle(rmRows, r)); setSel(null); }}
                    onKeyDown={(e) => e.key === 'Enter' && setRmRows(toggle(rmRows, r))}>{rmRows.includes(r) ? `r${r} ✕` : `r${r}`}<sub>b{Math.floor(r / rowsPerBlock)}</sub></span>
                ) : (
                  <span key={'r' + r} className="void" title={tr(`block ${Math.floor(r / rowsPerBlock)}`, `khối ${Math.floor(r / rowsPerBlock)}`)}>r{r}<sub>b{Math.floor(r / rowsPerBlock)}</sub></span>
                ),
                ...Array.from({ length: cols }, (_, c) => {
                  const m = moved.moves[r * cols + c];
                  return (
                    <span key={r + '-' + c} data-src={m.src} data-dest={m.dest ?? ''}
                      className={sel === m.src ? 'hot' : m.dest == null ? 'void' : ''}
                      style={{ cursor: 'pointer' }}
                      onClick={() => setSel(m.src)}>
                      {m.dest == null ? '✕' : m.dest}
                    </span>
                  );
                }),
              ])}
            </div>
            <p className="muted" style={{ margin: '0.4rem 0 0', fontSize: '0.8rem' }}>
              <Translated vi={<>Mỗi khối {rowsPerBlock} hàng; chữ b nhỏ trên mỗi hàng là khối của nó.</>}>
                Blocks of {rowsPerBlock} row{rowsPerBlock === 1 ? '' : 's'}; the small b on each row is its block.
              </Translated>
            </p>
          </div>
          <div>
            <div className="mx-label t-global"><Translated vi={<>kết quả, {moved.newRows} × {moved.newCols}: phần tử nguồn nào rơi vào đây</>}>result, {moved.newRows} × {moved.newCols}: which source element lands here</Translated></div>
            <div className="mx global" style={{ gridTemplateColumns: `repeat(${Math.max(1, moved.newCols)}, auto)` }} data-part="result">
              {Array.from({ length: moved.size }, (_, d) => {
                const m = moved.moves.find((x) => x.dest === d);
                return (
                  <span key={d} className={m && sel === m.src ? 'hot' : m ? 'on' : 'void'} data-slot={d}>
                    {m ? m.src : tr('pad', 'đệm')}
                  </span>
                );
              })}
            </div>
          </div>
        </div>
      ) : null}

      {!isMatrix && dedup ? (
        <div style={{ overflowX: 'auto' }} data-part="dedup">
          {(
            [
              ['key', parsed.keys.map(String)],
              ['keep', dedup.keep.map(String)],
              ['offset', dedup.offset.map((o, i) => (dedup.keep[i] ? String(o) : '·'))],
              ['block', parsed.keys.map((_, i) => String(Math.floor(i / perBlock)))],
            ] as [string, string[]][]
          ).map(([name, vals]) => (
            <div key={name} className="row" style={{ gap: '0.5rem', marginBottom: '0.3rem', flexWrap: 'nowrap' }}>
              <span className="mx-label" style={{ width: '3.4rem', margin: 0 }}>{tr(name, (({ key: 'khóa', keep: 'giữ', offset: 'độ lệch', block: 'khối' }) as Record<string, string>)[name] ?? name)}</span>
              <div className="mx global" style={{ gridTemplateColumns: `repeat(${vals.length}, auto)` }} data-row={name}>
                {vals.map((v, i) => (
                  <span key={i} className={sel === i ? 'hot' : name !== 'block' && dedup.keep[i] ? 'on' : name === 'block' ? 'void' : ''}
                    onClick={() => setSel(i)} style={{ cursor: 'pointer' }}>{v}</span>
                ))}
              </div>
            </div>
          ))}
          <div className="row" style={{ gap: '0.5rem', flexWrap: 'nowrap' }}>
            <span className="mx-label" style={{ width: '3.4rem', margin: 0 }}><Translated vi="ra">out</Translated></span>
            <div className="mx global" style={{ gridTemplateColumns: `repeat(${Math.max(1, dedup.count)}, auto)` }} data-row="out">
              {dedup.out.map((v, i) => <span key={i} className="on">{v}</span>)}
            </div>
            <span className="muted" data-part="count"><Translated vi={<>{dedup.count} được giữ</>}>{dedup.count} kept</Translated></span>
          </div>
        </div>
      ) : null}

      <div className="narrate" role="status" style={{ margin: '0.8rem 0' }} data-part="formula">
        {selMove ? (
          isMatrix ? (
            <Translated
              vi={
                <>
                  Phần tử (hàng {selMove.r}, cột {selMove.c}), chỉ số phẳng {selMove.src}:{' '}
                  {selMove.dest == null ? <b>{kind === 'removeRows' ? `hàng ${selMove.r} bị xóa` : `cột ${selMove.c} bị xóa`}</b> : <>chuyển tới <b>{selMove.formula}</b>. Không cần scan: chỉ số quyết định.</>}
                </>
              }
            >
              Element (row {selMove.r}, column {selMove.c}), flat index {selMove.src}:{' '}
              {selMove.dest == null ? <b>{selMove.formula}</b> : <>moves to <b>{selMove.formula}</b>. No scan: the index decides.</>}
            </Translated>
          ) : (
            <Translated
              vi={
                <>
                  Khóa {selMove.src}:{' '}
                  {dedup && dedup.keep[selMove.src]
                    ? `offset[${selMove.src}] = ${dedup.offset[selMove.src]} (scan loại trừ của các cờ giữ lại)`
                    : `giống khóa ${selMove.src - 1}: bị bỏ`}
                  . Ở đây giá trị quyết định, nên các độ lệch cần một phép scan.
                </>
              }
            >
              <>Key {selMove.src}: {selMove.formula}. Here the value decides, so the offsets need a scan.</>
            </Translated>
          )
        ) : (
          <Translated vi={<>Bấm vào một ô để xem nó đi đâu và vì sao.</>}>
            <>Click a cell to see where it goes and why.</>
          </Translated>
        )}
      </div>

      {parsed.error && !isMatrix ? null : <DirectionPanel deps={deps} inPlace={inPlace} unit={isMatrix ? 'element' : 'key'} />}
    </Bench>
  );
}

// ---------------------------------------------------------------- the page

export function S128() {
  const q = questionsS128;
  return (
    <>
      <Translated
        vi={
          <p>
            Bộ lọc ổn định là một thành viên của một họ: các mẫu trong đó mọi khóa di chuyển theo một chiều. Xóa khóa đẩy các khóa
            còn lại về phía đầu; chèn khoảng trống đẩy các khóa về phía cuối. Những gì bạn đã học cho bộ lọc vẫn áp dụng được: gom
            tạm qua bộ nhớ chia sẻ (shared memory), gộp việc, và khi làm tại chỗ, một thứ tự giữa các khối.
          </p>
        }
      >
        <p>
          The stable filter is one member of a family: patterns in which every key moves in one direction. Removing keys
          moves the survivors towards the front; inserting gaps moves keys towards the back. What you learned for the
          filter carries over: staging through shared memory, coarsening, and, in place, an order between blocks.
        </p>
      </Translated>
      <Source section="12.8" pages="301" />

      <h2><Translated vi="Loại bỏ phần tử trùng lặp khỏi một danh sách đã sắp xếp">Removing duplicates from a sorted list</Translated></h2>
      <Translated
        vi={
          <p>
            Trong một danh sách đã sắp xếp, các khóa bằng nhau nằm cạnh nhau. Khi đó loại bỏ trùng lặp là một bộ lọc ổn định có
            phép kiểm tra giữ lại là "khóa này khác khóa đứng trước nó", với khóa đầu tiên luôn được giữ. Mọi thứ còn lại là mục
            12.5: cờ giữ lại, một scan loại trừ, một phép rải.
          </p>
        }
      >
        <p>
          In a sorted list, equal keys sit next to each other. Removing duplicates is then a stable filter whose keep
          test is "this key differs from the one before it", with the first key always kept. Everything else is section
          12.5: keep flags, an exclusive scan, a scatter.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch12-facts.md, A8)', '2026-11-20 (notes/today/ch12-facts.md, mục A8)')}>
        <Translated
          vi={
            <p>
              CUB có nó dưới tên <code>DeviceSelect::Unique</code>: "only the first key from each run is selectively copied" (chỉ
              khóa đầu tiên của mỗi dải được chọn để chép).
            </p>
          }
        >
          <p>
            CUB has it as <code>DeviceSelect::Unique</code>: "only the first key from each run is selectively copied".
          </p>
        </Translated>
      </Today>
      <Ask q={q['ch12/12.8/q-dedup']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Xóa hàng hoặc cột">Removing rows or columns</Translated></h2>
      <Translated
        vi={
          <p>
            Xóa hàng hoặc cột của một ma trận khác ở một điểm quan trọng: khóa được quyết định theo <i>chỉ số</i>, không theo giá
            trị. Một khóa có còn lại hay không, và có bao nhiêu khóa bị xóa đứng trước nó, suy ra từ hàng và cột của nó, nên mỗi
            luồng tính đích đến của khóa mình bằng một công thức. Không cần scan.
          </p>
        }
      >
        <p>
          Removing rows or columns of a matrix is different in one way that matters: keys go by <i>index</i>, not by
          value. Whether a key survives, and how many removed keys come before it, follow from its row and column, so
          each thread computes its key's destination with a formula. No scan.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Khi làm tại chỗ, hiểm họa của mục <a href={href('ch12', '12.7')}>12.7</a> vẫn còn đó: khóa của một khối sau rơi vào
            vùng của một khối trước, nên các khối trước vẫn phải đọc trước khi các khối sau ghi. Sách giữ cách đồng bộ hóa một
            chiều của scan một lượt cho việc đó, nhưng không có gì để cộng: các khối chuyển cho nhau một cờ và không có tổng một
            phần nào.
          </p>
        }
      >
        <p>
          In place, the hazard of section <a href={href('ch12', '12.7')}>12.7</a> is still there: a later block's keys
          land in an earlier block's range, so earlier blocks must still read before later blocks write. The book keeps
          the single-pass scan's one-directional synchronisation for that, but with nothing to add up: the blocks pass a
          flag and no partial sum.
        </p>
      </Translated>
      <Source section="12.8" pages="301" />
      <Predict
        id="ch12/12.8/p-col"
        prompt={
          <Translated vi={<>Một ma trận thứ tự hàng 4 × 6 bị mất cột 2. Phần tử (hàng 2, cột 4) chuyển tới chỉ số phẳng nào?</>}>
            A 4 × 6 row-major matrix loses column 2. Element (row 2, column 4) moves to which flat index?
          </Translated>
        }
      >
        {(v) => (
          <Translated
            vi={
              <p>
                Nó chuyển tới <b>13</b>: độ rộng mới là 5, cột 4 trở thành cột 3 vì có một cột bị xóa ở bên trái nó, và 2 × 5 + 3 =
                13.{' '}
                {v === 13 ? 'Bạn đã đúng.' : v === 15 ? 'Bạn có giữ độ rộng cũ không? 2 × 6 + 3 = 15 dùng độ rộng cũ.' : v === 14 ? 'Bạn có giữ cột cũ không? 2 × 5 + 4 = 14 dùng cột cũ.' : 'Xem trình khám phá bên dưới: chọn xóa cột và bấm vào (2, 4).'}
              </p>
            }
          >
            <p>
              It moves to <b>13</b>: the new width is 5, column 4 becomes column 3 because one removed column is to its
              left, and 2 × 5 + 3 = 13.{' '}
              {v === 13 ? 'You had it.' : v === 15 ? 'Did you keep the old width? 2 × 6 + 3 = 15 uses it.' : v === 14 ? 'Did you keep the old column? 2 × 5 + 4 = 14 does.' : 'Check the explorer below: select remove columns and click (2, 4).'}
            </p>
          </Translated>
        )}
      </Predict>

      <h2><Translated vi="Thêm hàng hoặc cột">Adding rows or columns</Translated></h2>
      <Translated
        vi={
          <p>
            Thêm hàng hoặc cột, chẳng hạn đệm mỗi hàng tới một độ rộng mà hệ thống bộ nhớ ưa thích, một lần nữa do chỉ số quyết
            định. Nhưng giờ các khóa dịch <i>ra ngoài</i>, về phía các ô cao hơn. Khi không làm tại chỗ, điều đó không thay đổi
            gì. Khi làm tại chỗ, nó đảo ngược yêu cầu: khóa của một khối rơi vào ô của các khối <i>sau</i>, nên các khối sau phải
            đọc xong trước khi các khối trước ghi, và chuỗi chạy từ khối có các khóa cuối về khối có các khóa đầu.
          </p>
        }
      >
        <p>
          Adding rows or columns, for example padding each row to a width the memory system prefers, is again decided by
          the index. But the keys now move <i>outwards</i>, towards higher slots. Out of place that changes nothing. In
          place it reverses the requirement: a block's keys land on slots of <i>later</i> blocks, so later blocks must
          finish reading before earlier blocks write, and the chain runs from the block with the last keys to the block
          with the first.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              <b>Trang sách ghi "earlier" hai lần.</b> Nó viết "the earlier thread blocks ... write to memory locations
              previously occupied by keys read by earlier thread blocks". Chữ "earlier" thứ hai lẽ ra phải là "later", như câu
              kế tiếp xác nhận. Trang này còn có hai lỗi đánh máy nhỏ ("in the section", "data in-place data movement").
            </p>
          }
        >
          <p>
            <b>The page says "earlier" twice.</b> It reads "the earlier thread blocks ... write to memory locations
            previously occupied by keys read by earlier thread blocks". The second "earlier" should be "later", as the
            next sentence confirms. The page also has two small typos ("in the section", "data in-place data movement").
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Trình khám phá có cả bốn mẫu. Với mỗi mẫu, hãy quyết định các mũi tên giữa các khối sẽ chỉ theo chiều nào trước khi
            bạn bật <i>tại chỗ</i>.
          </p>
        }
      >
        <p>
          The explorer has all four patterns. For each one, decide which way the arrows between blocks will point before
          you switch <i>in place</i> on.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các khối sở hữu nguyên các hàng (hoặc một dải khóa) và đọc hết chúng trước khi ghi bất kỳ cái nào. Trình khám phá cho
              thấy ai phải chờ ai, không phải khi nào việc gì xảy ra.
            </p>
          }
        >
          <p>
            Blocks own whole rows (or a run of keys) and read all of them before writing any. The explorer shows who must
            wait for whom, not when anything happens.
          </p>
        </Translated>
      </Simplification>
      <RelatedExplorer />
      <Ask q={q['ch12/12.8/q-noscan']} />
      <Ask q={q['ch12/12.8/q-order-only']} />
      <Ask q={q['ch12/12.8/q-pad']} />

      <h2><Translated vi="Những phép di chuyển khó hơn">Harder movements</Translated></h2>
      <Translated
        vi={
          <p>
            Một số phép di chuyển dữ liệu tại chỗ phức tạp hơn những phép này. Chuyển vị một ma trận hoặc một tensor tại chỗ là ví
            dụ của sách, và sách chỉ dẫn tới tài liệu tham khảo cho nó (tài liệu [2], Sung và cộng sự, 2014). Nó không được đề cập
            ở đây.
          </p>
        }
      >
        <p>
          Some in-place data movements are more complex than these. Transposing a matrix or a tensor in place is the
          book's example, and it only points to the literature for it (reference [2], Sung and others, 2014). It is not
          covered here.
        </p>
      </Translated>
      <Source section="12.8" pages="301–302" />
      <Translated
        vi={
          <p>
            Bạn sẽ gặp cả hai mẫu liên quan ở gần một mô hình. Đệm chiều từ vựng lên một bội số của độ rộng tile là thêm cột. Loại
            bỏ các id token trùng lặp khỏi một danh sách đã sắp xếp, để một repetition penalty đếm mỗi token khác nhau một lần,
            là bộ lọc trùng lặp.
          </p>
        }
      >
        <p>
          You will meet both related patterns close to a model. Padding a vocabulary dimension up to a multiple of a
          tile width is adding columns. Removing duplicate token ids from a sorted list, so that a repetition penalty
          counts each distinct token once, is the duplicate filter.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Cả hai ví dụ là của khóa học này. Ví dụ của sách là một danh sách đã sắp xếp và các bố cục ma trận được điều chỉnh để căn chỉnh.</p>}>
          <p>Both examples are ours. The book's examples are a sorted list and matrix layouts adjusted for alignment.</p>
        </Translated>
      </Beyond>
    </>
  );
}
