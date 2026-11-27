import { useMemo, useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useReports } from '../../lib/reports';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  ELEMENTS,
  INDEX,
  MARKS,
  arrayLimit,
  binary,
  bitsOf,
  memoryFigures,
  reach,
  type IndexType,
  type MemoryFigure,
} from '../../sim/addrwidth';

/** A long identifier that may break anywhere, for the narrow margin notes. */
function Id({ children }: { children: ReactNode }) {
  return <code style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{children}</code>;
}

const INDEX_LABEL: Record<IndexType, string> = { int32: 'int', uint32: 'unsigned int', int64: 'int64_t' }; // i18n-ignore: C type names
const FIELD_SHORT: Record<MemoryFigure['field'], string> = {
  get totalGlobalMem() {
    return tr('global memory', 'bộ nhớ toàn cục');
  },
  get maxBufferLength() {
    return tr('largest buffer', 'buffer lớn nhất');
  },
  get recommendedMaxWorkingSetSize() {
    return tr('working set', 'working set');
  },
};
const pow = (e: number) => (
  <>
    2<sup>{e}</sup>
  </>
);

// ---------------------------------------------------------------- your devices
function YourMemory() {
  const { reports, loading } = useReports();
  const locale = useLocale();
  const figs = useMemo(() => memoryFigures(reports), [reports, locale]);
  const intFloats = arrayLimit('int32', 4).bytes;
  return (
    <div className="panel" data-your-memory={figs.length} style={{ margin: '0.4rem 0 1.4rem' }}>
      <p className="ui" style={{ margin: '0 0 0.4rem' }}>
        <b>
          <Translated vi="Những gì thiết bị của bạn báo cáo">What your devices reported</Translated>
        </b>
      </p>
      {figs.length ? (
        <>
          <div style={{ overflowX: 'auto' }}>
            <table className="data" style={{ margin: 0 }}>
              <thead>
                <tr>
                  <th>
                    <Translated vi="Thiết bị">Device</Translated>
                  </th>
                  <th>
                    <Translated vi="Thuộc tính">Property</Translated>
                  </th>
                  <th className="num">
                    <Translated vi="Giá trị báo cáo">Reported</Translated>
                  </th>
                  <th className="num">
                    <Translated vi="So với 8 GiB">Against 8 GiB</Translated>
                  </th>
                </tr>
              </thead>
              <tbody>
                {figs.map((f) => (
                  <tr key={f.backend + f.field} data-figure={f.field} data-bytes={f.bytes}>
                    <td>
                      {f.name} <span className="muted">({f.backend})</span>
                    </td>
                    <td>
                      <code>{f.field}</code>
                      <br />
                      <span className="muted">{f.meaning}</span>
                    </td>
                    <td className="num">{binary(f.bytes).text}</td>
                    <td className="num">
                      {f.bytes > intFloats ? (
                        <span className="chip fail">
                          <Translated vi={<>gấp {Number((f.bytes / intFloats).toFixed(2))} lần</>}>
                            {Number((f.bytes / intFloats).toFixed(2))} times
                          </Translated>
                        </span>
                      ) : (
                        <span className="chip ok">
                          <Translated vi="nằm trong">within</Translated>
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Translated
            vi={
              <p className="ui muted" style={{ margin: '0.6rem 0 0' }}>
                Cột cuối so sánh mỗi giá trị với 8 GiB số float mà một chỉ số <code>int</code> với tới được. Một giá trị lớn
                hơn mức đó là lượng bộ nhớ mà một kernel đánh chỉ số bằng <code>int</code> trên một mảng float không thể với
                tới phần cuối. Trên Metal, giá trị cần so sánh là <code>maxBufferLength</code>, vì một mảng nằm trong một
                buffer.
              </p>
            }
          >
            <p className="ui muted" style={{ margin: '0.6rem 0 0' }}>
              The last column compares each figure with the 8 GiB of floats an <code>int</code> index reaches. A figure
              above it is memory an <code>int</code>-indexed kernel over one float array could not reach the end of. On
              Metal the figure to compare is <code>maxBufferLength</code>, since one array lives in one buffer.
            </p>
          </Translated>
        </>
      ) : (
        <p className="ui muted" style={{ margin: 0 }}>
          {loading ? (
            tr('Reading reports.', 'Đang đọc báo cáo.')
          ) : (
            <Translated
              vi={
                <>
                  Chưa có GPU nào được ghi nhận trên máy này. Hãy chạy{' '}
                  <code>python3 labs/pmpp.py run appC/lab1 --bench</code> rồi tải lại: lần chạy đầu tiên trên một backend
                  sẽ ghi lại những gì thiết bị báo cáo.
                </>
              }
            >
              <>
                No GPU is recorded on this machine yet. Run <code>python3 labs/pmpp.py run appC/lab1 --bench</code> and
                reload: the first run on a backend records what the device reports.
              </>
            </Translated>
          )}
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- the explorer
const LO = 16;
const HI = 64;
const W = 640;
const PAD = 26;
const x = (bits: number) => PAD + ((Math.min(HI, Math.max(LO, bits)) - LO) / (HI - LO)) * (W - 2 * PAD);

function BitsToBytes() {
  const { reports } = useReports();
  const figs = useMemo(() => memoryFigures(reports), [reports]);
  const [bits, setBits] = useState(32);
  const [elem, setElem] = useState(4);
  const [index, setIndex] = useState<IndexType>('int32');

  const r = reach(bits);
  const lim = arrayLimit(index, elem, bits);
  const elemLabel = ELEMENTS.find((e) => e.bytes === elem)?.label ?? tr(`${elem} bytes`, `${elem} byte`);
  const over = figs.filter((f) => f.bytes > r);
  const limBits = bitsOf(lim.indexBytes);

  return (
    <Bench
      title={tr('Bits to bytes', 'Từ bit tới byte')}
      task={tr(
        "Find the smallest width that names all of your device's memory, then the index type that reaches all of it.",
        'Tìm độ rộng nhỏ nhất gọi tên được toàn bộ bộ nhớ thiết bị của bạn, rồi tìm kiểu chỉ số với tới được toàn bộ bộ nhớ đó.',
      )}
      foot={
        <Translated
          vi={
            <>
              Tầm với là 2<sup>bits</sup> byte, theo đơn vị nhị phân (1 GiB = 2<sup>30</sup> byte). Các mốc lấy từ sách (32,
              48 và 49 bit), whitepaper Fermi (40) và tài liệu Linux (48 và 57); xem danh sách bên dưới thanh.
            </>
          }
        >
          <>
            Reach is 2<sup>bits</sup> bytes, in binary units (1 GiB = 2<sup>30</sup> bytes). The marks come from the book
            (32, 48 and 49 bits), the Fermi whitepaper (40) and the Linux documentation (48 and 57); see the list below the
            bar.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <label className="field" style={{ flex: '1 1 18rem' }}>
          <Translated vi={<>Độ rộng địa chỉ: {bits} bit</>}>Address width: {bits} bits</Translated>
          <input
            type="range"
            min={LO}
            max={HI}
            step={1}
            value={bits}
            aria-label={tr('Address width in bits', 'Độ rộng địa chỉ tính bằng bit')}
            onChange={(e) => setBits(Number(e.target.value))}
          />
        </label>
        <div aria-live="polite">
          <p style={{ margin: 0, fontSize: '1.5rem', fontWeight: 600, lineHeight: 1.2 }}>
            <output data-out="reach" data-bits={bits} data-bytes={r}>
              {binary(r).text}
            </output>
          </p>
          <p className="muted" style={{ margin: 0 }}>
            <Translated
              vi={
                <>
                  {pow(bits)} byte. Bớt một bit: {binary(reach(bits - 1)).text}. Thêm một bit: {binary(reach(bits + 1)).text}.
                </>
              }
            >
              {pow(bits)} bytes. One bit fewer: {binary(reach(bits - 1)).text}. One more: {binary(reach(bits + 1)).text}.
            </Translated>
          </p>
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <svg
          viewBox={`0 0 ${W} 150`}
          role="img"
          aria-label={tr(
            `Log scale of reach from ${LO} to ${HI} bits, filled to ${bits} bits`,
            `Thang logarit của tầm với từ ${LO} tới ${HI} bit, tô tới ${bits} bit`,
          )}
          style={{ width: '100%', minWidth: 520, display: 'block' }}
          data-part="bar"
        >
          <rect x={x(LO)} y={52} width={x(HI) - x(LO)} height={20} fill="var(--cell)" stroke="var(--rule)" />
          <rect x={x(LO)} y={52} width={x(bits) - x(LO)} height={20} fill="var(--global-wash)" stroke="var(--global)" data-part="fill" />
          {[16, 24, 32, 40, 48, 56, 64].map((b) => (
            <g key={b}>
              <line x1={x(b)} x2={x(b)} y1={72} y2={78} stroke="var(--ink-3)" />
              <text x={x(b)} y={90} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>
                {b}
              </text>
            </g>
          ))}
          {MARKS.map((m, i) => (
            <g key={m.bits} data-mark={m.bits}>
              <line x1={x(m.bits)} x2={x(m.bits)} y1={i % 2 ? 30 : 16} y2={72} stroke="var(--ink-2)" strokeDasharray="2 2" />
              <text x={x(m.bits)} y={i % 2 ? 26 : 12} fontSize="10.5" textAnchor={i === 3 ? 'start' : i === 2 ? 'end' : 'middle'} style={{ fill: 'var(--ink-2)' }}>
                {m.bits}: {m.label}
              </text>
            </g>
          ))}
          {figs.map((f, i) => {
            const fx = x(bitsOf(f.bytes));
            const y = 102 + (i % 3) * 15;
            return (
              <g key={f.backend + f.field} data-part="device" data-bits={bitsOf(f.bytes).toFixed(3)}>
                <path d={`M ${fx} 74 l -4 7 h 8 z`} fill={f.bytes > r ? 'var(--fail)' : 'var(--ok)'} />
                <line x1={fx} x2={fx} y1={81} y2={y - 8} stroke="var(--rule)" />
                <text x={fx} y={y} fontSize="10" textAnchor={fx > W / 2 ? 'end' : 'start'}>
                  {f.backend}: {FIELD_SHORT[f.field]}, {binary(f.bytes).text}
                </text>
              </g>
            );
          })}
          <g data-part="limit" data-bits={limBits.toFixed(3)}>
            <rect x={x(limBits) - 4} y={56} width={8} height={12} transform={`rotate(45 ${x(limBits)} 62)`} fill="var(--reg)" />
          </g>
          <text x={x(LO)} y={146} fontSize="10.5" style={{ fill: 'var(--reg)' }} data-part="limit-label">
            <Translated
              vi={
                <>
                  ◆ một chỉ số {INDEX_LABEL[index]} trên {elemLabel.split(',')[0]} với tới bao xa: {binary(lim.indexBytes).text}
                  {limBits > HI ? ' (vượt quá cuối thanh này)' : ''}
                </>
              }
            >
              ◆ how far an {INDEX_LABEL[index]} index over {elemLabel.split(',')[0]} reaches: {binary(lim.indexBytes).text}
              {limBits > HI ? ' (past the end of this bar)' : ''}
            </Translated>
          </text>
        </svg>
      </div>

      <div className="row" role="group" aria-label={tr('Marks', 'Các mốc')} style={{ margin: '0.6rem 0 1rem' }}>
        {MARKS.map((m) => (
          <button key={m.bits} className="quiet" data-goto={m.bits} aria-pressed={bits === m.bits} onClick={() => setBits(m.bits)} title={m.source}>
            <Translated vi={<>{m.bits} bit, {m.label}: {m.what}</>}>
              {m.bits} bits, {m.label}: {m.what}
            </Translated>
          </button>
        ))}
      </div>

      {over.length ? (
        <p data-out="wall" data-over={over.length} style={{ margin: '0 0 1rem' }}>
          <span className="chip fail">
            <Translated vi="bức tường">wall</Translated>
          </span>{' '}
          <Translated vi={<>Địa chỉ {bits}-bit không gọi tên được toàn bộ những gì thiết bị của bạn báo cáo:</>}>
            {bits}-bit addresses cannot name all of what your devices reported:
          </Translated>{' '}
          {over.map((f) => `${f.name} (${f.backend}), ${FIELD_SHORT[f.field]} ${binary(f.bytes).text}`).join('; ')}.
        </p>
      ) : figs.length ? (
        <p data-out="wall" data-over={0} style={{ margin: '0 0 1rem' }}>
          <span className="chip ok">
            <Translated vi="vừa">fits</Translated>
          </span>{' '}
          <Translated vi={<>Địa chỉ {bits}-bit gọi tên được mọi giá trị bộ nhớ mà thiết bị của bạn báo cáo.</>}>
            {bits}-bit addresses name every memory figure your devices reported.
          </Translated>
        </p>
      ) : null}

      <div className="cols plot">
        <div className="panel">
          <div className="row" style={{ alignItems: 'flex-start' }}>
            <label className="field">
              <Translated vi="Kiểu phần tử">Element type</Translated>
              <select value={elem} onChange={(e) => setElem(Number(e.target.value))} aria-label={tr('Element type', 'Kiểu phần tử')}>
                {ELEMENTS.map((e) => (
                  <option key={e.bytes} value={e.bytes}>
                    {e.label}
                  </option>
                ))}
              </select>
            </label>
            <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
              <Translated vi="Kiểu chỉ số">Index type</Translated>
              <span className="seg" role="group" aria-label={tr('Index type', 'Kiểu chỉ số')}>
                {(Object.keys(INDEX) as IndexType[]).map((k) => (
                  <button key={k} aria-pressed={index === k} onClick={() => setIndex(k)}>
                    {INDEX_LABEL[k]}
                  </button>
                ))}
              </span>
            </div>
          </div>
        </div>
        <div className="panel" aria-live="polite">
          <p className="muted" style={{ margin: 0 }}>
            <Translated vi="Mảng lớn nhất kernel này đánh chỉ số được">Largest array this kernel can index</Translated>
          </p>
          <p style={{ fontSize: '1.5rem', fontWeight: 600, margin: '0 0 0.2rem', lineHeight: 1.2 }}>
            <output data-out="limit" data-bytes={lim.bytes} data-binds={lim.binds}>
              {binary(lim.bytes).text}
            </output>
          </p>
          <p style={{ margin: 0 }} data-out="limit-why">
            {lim.binds === 'index' ? (
              <Translated
                vi={
                  <>
                    Chỉ số cạn trước: <code>{INDEX_LABEL[index]}</code> với tới {pow(INDEX[index].log2Count)} phần tử, và với{' '}
                    {elem === 0.5 ? 'nửa byte' : `${elem} byte`} mỗi phần tử thì đó là {binary(lim.indexBytes).text}.
                  </>
                }
              >
                <>
                  The index runs out first: <code>{INDEX_LABEL[index]}</code> reaches {pow(INDEX[index].log2Count)}{' '}
                  elements, and at {elem === 0.5 ? 'half a byte' : `${elem} byte${elem === 1 ? '' : 's'}`} each that is{' '}
                  {binary(lim.indexBytes).text}.
                </>
              </Translated>
            ) : (
              <Translated
                vi={
                  <>
                    Địa chỉ cạn trước: địa chỉ {bits}-bit gọi tên được {binary(r).text}, ít hơn{' '}
                    {binary(lim.indexBytes).text} mà một chỉ số <code>{INDEX_LABEL[index]}</code> có thể với tới.
                  </>
                }
              >
                <>
                  The address runs out first: {bits}-bit addresses name {binary(r).text}, less than the{' '}
                  {binary(lim.indexBytes).text} a <code>{INDEX_LABEL[index]}</code> index could reach.
                </>
              </Translated>
            )}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSC4: Record<string, Question> = {
  'appC/C.4/q-4gib': {
    kind: 'numeric',
    id: 'appC/C.4/q-4gib',
    prompt: (
      <Translated
        vi={
          <>
            Một địa chỉ byte rộng 32 bit. Nó gọi tên được bao nhiêu GiB bộ nhớ? (1 GiB là 2<sup>30</sup> byte.)
          </>
        }
      >
        <>
          A byte address is 32 bits wide. How many GiB of memory can it name? (1 GiB is 2<sup>30</sup> bytes.)
        </>
      </Translated>
    ),
    answer: 4,
    unit: 'GiB',
    explain: (
      <Translated
        vi={
          <p>
            2<sup>32</sup> = 2<sup>2</sup> × 2<sup>30</sup>, nên là 4 GiB. Sách đưa con số này làm trần cho hai thứ cùng lúc:
            một GPU đời đầu mang được bao nhiêu bộ nhớ, và một kernel làm việc được trên bao nhiêu dữ liệu.
          </p>
        }
      >
        <p>
          2<sup>32</sup> = 2<sup>2</sup> × 2<sup>30</sup>, so 4 GiB. The book gives this as the ceiling on two things at
          once: how much memory an early GPU could carry, and how much data one kernel could work on.
        </p>
      </Translated>
    ),
    source: 'App. C §C.4, pp. 625–626',
    vi: { source: 'Phụ lục C §C.4, pp. 625–626' },
  },
  'appC/C.4/q-why-special': {
    kind: 'choice',
    id: 'appC/C.4/q-why-special',
    prompt: (
      <Translated
        vi={
          <>
            Với địa chỉ GPU 32-bit, truy cập zero-copy cần một phép cấp phát đặc biệt (<code>cudaHostAlloc</code>) và một
            con trỏ thứ hai cho kernel, thay vì để kernel dùng bất kỳ con trỏ nào từ <code>malloc</code>. Mục này đưa ra lý
            do nào?
          </>
        }
      >
        <>
          With 32-bit GPU addresses, zero-copy access needed a special allocation (<code>cudaHostAlloc</code>) and a
          second pointer for the kernel, instead of letting a kernel use any pointer from <code>malloc</code>. Which reason
          does this section give?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Bộ nhớ ghim nhanh hơn bộ nhớ phân trang (pageable), nên GPU đòi dùng nó cho mọi buffer mà nó đọc tại chỗ.">
            Pinned memory is faster than pageable memory, so the GPU insisted on it for every buffer it read in place.
          </Translated>
        ),
        why: (
          <Translated vi="Ghim giữ các trang tại chỗ trong khi GPU đọc chúng (§C.2). Nó không giải thích vì sao kernel cần một giá trị con trỏ khác.">
            Pinning keeps the pages in place while the GPU reads them (§C.2). It does not explain why the kernel needed a
            different pointer value.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Bộ nhớ từ malloc không liên tục, nên GPU không thể đi qua nó bằng một con trỏ.">
            Memory from malloc is not contiguous, so the GPU could not walk it with one pointer.
          </Translated>
        ),
        why: (
          <Translated vi="Một dải ảo từ malloc theo định nghĩa là liên tục. Câu hỏi là liệu GPU có gọi tên được địa chỉ của nó hay không.">
            A virtual range from malloc is contiguous by definition. The question is whether the GPU can even name its
            address.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một địa chỉ ảo host 48-bit không thể viết vào một con trỏ GPU 32-bit, nên kernel cần một địa chỉ khác cho cùng buffer đó.">
            A 48-bit host virtual address cannot be written in a 32-bit GPU pointer, so the kernel needed a different
            address for the same buffer.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="GPU không đọc được DRAM gắn với CPU, nên dữ liệu phải được sao chép trước.">
            A GPU cannot read DRAM attached to the CPU, so the data had to be copied first.
          </Translated>
        ),
        why: (
          <Translated vi="Bộ nhớ zero-copy tồn tại chính là để kernel đọc được, qua đường kết nối (§C.2).">
            Zero-copy memory exists precisely so that a kernel can, over the interconnect (§C.2).
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một địa chỉ host 48-bit không vừa trong 32 bit. Vì vậy sách tính con trỏ GPU hẹp là một trong các lý do cho phép
            cấp phát đặc biệt, và cho hai giá trị con trỏ, một cho host và một cho kernel, mà §C.2 mô tả.
          </p>
        }
      >
        <p>
          A 48-bit host address does not fit in 32 bits. So the book counts the narrow GPU pointer among the reasons for
          the special allocation, and for the two pointer values, one for the host and one for the kernel, that §C.2
          describes.
        </p>
      </Translated>
    ),
    source: 'App. C §C.4, p. 626, with §C.2, p. 624',
    vi: { source: 'Phụ lục C §C.4, p. 626, cùng §C.2, p. 624' },
  },
  'appC/C.4/q-fermi-tib': {
    kind: 'numeric',
    id: 'appC/C.4/q-fermi-tib',
    prompt: (
      <Translated vi={<>Không gian địa chỉ hợp nhất của Fermi rộng 40 bit. Đó là bao nhiêu TiB?</>}>
        <>Fermi's unified address space was 40 bits wide. How many TiB is that?</>
      </Translated>
    ),
    answer: 1,
    unit: 'TiB',
    explain: (
      <Translated
        vi={
          <>
            <p>
              2<sup>40</sup> byte đúng bằng 1 TiB, thứ mà whitepaper Fermi của NVIDIA gọi là "a Terabyte". Đó là 2
              <sup>8</sup> = 256 lần bức tường 4 GiB.
            </p>
            <p>
              Đó cũng là lý do câu "Kepler là thế hệ đầu tiên" không đứng vững: thế hệ trước Kepler đã bỏ lại 32 bit phía sau.
            </p>
          </>
        }
      >
        <>
          <p>
            2<sup>40</sup> bytes is exactly 1 TiB, which NVIDIA's Fermi whitepaper calls "a Terabyte". That is 2
            <sup>8</sup> = 256 times the 4 GiB wall.
          </p>
          <p>
            It is also why "Kepler was first" does not hold: the generation before Kepler had already left 32 bits behind.
          </p>
        </>
      </Translated>
    ),
    source: 'Fermi whitepaper, checked 2026-11-20 (notes/today/appC-facts.md, A6)',
    vi: { source: 'Whitepaper Fermi, kiểm tra ngày 2026-11-20 (notes/today/appC-facts.md, A6)' },
  },
  'appC/C.4/q-int-index': {
    kind: 'numeric',
    id: 'appC/C.4/q-int-index',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel tính <code>int i = blockIdx.x * blockDim.x + threadIdx.x</code> và dùng <code>i</code> để đánh chỉ
            số một mảng float. Mảng chứa được khoảng bao nhiêu GiB trước khi <code>i</code> bị tràn?
          </>
        }
      >
        <>
          A kernel computes <code>int i = blockIdx.x * blockDim.x + threadIdx.x</code> and uses <code>i</code> to index a
          float array. About how many GiB can the array hold before <code>i</code> overflows?
        </>
      </Translated>
    ),
    answer: 8,
    tol: 0.01,
    unit: 'GiB',
    explain: (
      <Translated
        vi={
          <>
            <p>
              <code>i</code> chạy từ 0 tới 2<sup>31</sup> − 1, nên nó với tới 2<sup>31</sup> phần tử. Mỗi phần tử 4 byte thì
              đó là 2<sup>33</sup> byte: 8 GiB. Với độ chính xác một nửa (half precision) là 4 GiB. Độ rộng của con trỏ
              không liên quan gì ở đây; chỉ số mới là phần hẹp.
            </p>
            <p>
              Mã thực tế hoặc tính offset bằng 64 bit, hoặc kiểm tra kích thước trước khi khởi chạy.{' '}
              <a href={href('ch03', '3.4')}>Chương 3, §3.4</a> cho thấy PyTorch, llama.cpp và MLX làm điều đó thế nào.
            </p>
          </>
        }
      >
        <>
          <p>
            <code>i</code> runs from 0 to 2<sup>31</sup> − 1, so it reaches 2<sup>31</sup> elements. At 4 bytes each that
            is 2<sup>33</sup> bytes: 8 GiB. In half precision, 4 GiB. The width of the pointer has nothing to do with it;
            the index is the narrow part.
          </p>
          <p>
            Real code either computes the offset in 64 bits or checks the size before launching.{' '}
            <a href={href('ch03', '3.4')}>Chapter 3, §3.4</a> shows how PyTorch, llama.cpp and MLX do it.
          </p>
        </>
      </Translated>
    ),
    source: 'Ours: C++ integer limits; see ch. 3 §3.4 and notes/today/ch03-facts.md',
    vi: { source: 'Của chúng tôi: giới hạn số nguyên của C++; xem ch. 3 §3.4 và notes/today/ch03-facts.md' },
  },
};

// ---------------------------------------------------------------- the section
export function SC4() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mỗi con trỏ là một con số, và độ rộng của con số đó quyết định một chương trình gọi tên được bao nhiêu bộ nhớ. Các
            GPU CUDA đầu tiên dùng địa chỉ ảo 32-bit, và địa chỉ vật lý của chúng rộng tối đa 32 bit. Trước khi đọc tiếp, hãy
            tính xem điều đó cho phép bao nhiêu.
          </p>
        }
      >
        <p>
          Every pointer is a number, and the width of that number decides how much memory a program can name. The first
          CUDA GPUs used 32-bit virtual addresses, and their physical addresses were at most 32 bits wide. Before reading
          on, work out what that allowed.
        </p>
      </Translated>
      <Source section="C.4" pages="625–626" />
      <Ask q={questionsSC4['appC/C.4/q-4gib']} label={tr('Work it out', 'Tự tính')} />

      <h2>
        <Translated vi="Một bức tường ở 4 GiB">A wall at 4 GiB</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bốn gibibyte là bức tường ở cả hai phía. Độ rộng vật lý giới hạn lượng bộ nhớ một GPU mang được. Độ rộng của con
            trỏ giới hạn lượng dữ liệu một kernel làm việc được, dù dữ liệu nằm ở đâu, trong bộ nhớ device hay bộ nhớ host.
          </p>
        }
      >
        <p>
          Four gibibytes was a wall on two sides. The physical width capped how much memory a GPU could carry. The width
          of a pointer capped how much data one kernel could work on, wherever the data lived, in device memory or in host
          memory.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Nó cũng giữ host và device tách khỏi nhau. Sách ghi rằng một CPU 64-bit dùng 48 trong 64 bit địa chỉ của nó. Vì
            vậy một địa chỉ host hoàn toàn không vừa trong một con trỏ GPU 32-bit, và sách tính điều này là một trong các lý
            do khiến bộ nhớ zero-copy, ở <a href={href('appC', 'C.2')}>§C.2</a>, cần lời gọi cấp phát riêng và một con trỏ
            riêng cho kernel.
          </p>
        }
      >
        <p>
          It also kept the host and the device apart. The book notes that a 64-bit CPU uses 48 of its 64 address bits. A
          host address therefore did not fit in a 32-bit GPU pointer at all, and the book counts this among the reasons
          that zero-copy memory, in <a href={href('appC', 'C.2')}>§C.2</a>, needed its own allocation call and a separate
          pointer for the kernel.
        </p>
      </Translated>
      <Source section="C.4" pages="626" />
      <Ask q={questionsSC4['appC/C.4/q-why-special']} />

      <h2>
        <Translated vi="Sáu mươi tư bit, và một mốc ngày cần sửa">Sixty-four bits, and a date to correct</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bước tiếp theo của sách là kiến trúc Kepler: theo sách, từ Kepler trở đi, GPU dùng địa chỉ ảo 64-bit và địa chỉ
            vật lý ít nhất 40 bit. Một GPU có thể mang hơn 4 GiB, một kernel có thể làm việc trên dữ liệu lớn hơn, và, phần
            mà phần còn lại của phụ lục này xây dựng lên trên, host và device có thể dùng đúng cùng một giá trị con trỏ cho
            một mẩu dữ liệu, dù nó nằm ở đâu.
          </p>
        }
      >
        <p>
          The book's next step is the Kepler architecture: from Kepler on, it says, GPUs used 64-bit virtual addresses and
          physical addresses of at least 40 bits. A GPU could carry more than 4 GiB, a kernel could work on larger data,
          and, the part the rest of this appendix builds on, the host and the device could use exactly the same pointer
          value for a piece of data wherever it lived.
        </p>
      </Translated>
      <Source section="C.4" pages="626" />
      <Translated
        vi={
          <p>
            Hai chi tiết của đoạn đó không qua được phép đối chiếu với chính tài liệu của NVIDIA. Sách ghi Kepler vào năm 2013
            và trình bày nó như thế hệ đã bỏ lại 32 bit. NVIDIA ra mắt các GPU Kepler đầu tiên vào tháng 3 năm 2012. Và Fermi,
            thế hệ trước đó, đã có định địa chỉ nạp và ghi 64-bit, một không gian địa chỉ hợp nhất 40-bit, hơn 4 GiB bộ nhớ,
            và một không gian địa chỉ chung cho host và mọi device có compute capability 2.0 trở lên trong một tiến trình
            64-bit.
          </p>
        }
      >
        <p>
          Two details of that paragraph do not survive a check against NVIDIA's own documents. The book dates Kepler to
          2013 and presents it as the generation that left 32 bits behind. NVIDIA launched the first Kepler GPUs in March
          2012. And Fermi, the generation before, already had 64-bit load and store addressing, a 40-bit unified address
          space, more than 4 GiB of memory, and one address space for the host and every device of compute capability 2.0 or
          later in a 64-bit process.
        </p>
      </Translated>
      <Today
        checked={tr(
          "2026-11-20 against NVIDIA's Kepler launch release, the Fermi whitepaper and the archived CUDA 11.0 guide (appC-facts.md, A6)",
          'ngày 2026-11-20, đối chiếu với thông cáo ra mắt Kepler của NVIDIA, whitepaper Fermi và bản lưu trữ hướng dẫn CUDA 11.0 (appC-facts.md, A6)',
        )}
      >
        <Translated vi={<p>Kepler: tháng 3 năm 2012. Fermi (whitepaper 2009, sản phẩm 2010) là thế hệ đầu tiên vượt qua 32 bit.</p>}>
          <p>Kepler: March 2012. Fermi (whitepaper 2009, products 2010) was first past 32 bits.</p>
        </Translated>
        <Translated vi={<p>"Ít nhất 40-bit vật lý" là con số của sách; không trang nào của NVIDIA mà chúng tôi đọc nêu con số này.</p>}>
          <p>"At least 40-bit physical" is the book's figure; no NVIDIA page we read states one.</p>
        </Translated>
      </Today>
      <Ask q={questionsSC4['appC/C.4/q-fermi-tib']} />
      <Translated
        vi={
          <p>
            Con số 48 bit cho CPU của sách là bố cục x86-64 kinh điển: bốn cấp bảng trang, 256 TiB địa chỉ ảo. Các CPU và
            nhân hệ điều hành mới hơn có thể thêm cấp thứ năm và địa chỉ 57-bit, nhưng Linux vẫn đặt bộ nhớ của chương trình
            dưới mốc 47 bit trừ khi chương trình yêu cầu nhiều hơn. Chúng tôi không tìm ra được CUDA có xử lý được một con trỏ
            host vượt trên mốc đó hay không, nên khóa học này không khẳng định theo chiều nào cả.
          </p>
        }
      >
        <p>
          The book's 48 bits for a CPU is the classic x86-64 layout: four levels of page tables, 256 TiB of virtual
          addresses. Newer CPUs and kernels can add a fifth level and 57-bit addresses, but Linux still places a program's
          memory below the 47-bit line unless the program asks for more. Whether CUDA copes with a host pointer above that
          line we could not find out, so this course says nothing either way.
        </p>
      </Translated>
      <Today
        checked={tr(
          "2026-11-20 against the Linux kernel's 5-level paging documentation (appC-facts.md, A7)",
          'ngày 2026-11-20, đối chiếu với tài liệu phân trang 5 cấp của nhân Linux (appC-facts.md, A7)',
        )}
      >
        <Translated vi={<p>Phân trang 4 cấp: 256 TiB. 5 cấp: 128 PiB, và không gian người dùng 56-bit khi được yêu cầu.</p>}>
          <p>4-level paging: 256 TiB. 5-level: 128 PiB, and 56-bit user space on request.</p>
        </Translated>
        <Translated vi={<p>Cách CUDA xử lý con trỏ trên 47 bit: unverified.</p>}>
          <p>CUDA's handling of pointers above 47 bits: unverified.</p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Giới hạn 32-bit vẫn còn theo bạn">The 32-bit limit that is still with you</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Con trỏ 64-bit không xóa bỏ mọi giới hạn 32-bit khỏi kernel. Chúng chuyển nó vào mã do bạn viết. Các kernel của
            những chương đầu tính chỉ số dưới dạng <code>int</code>, và một <code>int</code> chứa tối đa 2
            <sup>31</sup> − 1. Dù device có bao nhiêu bộ nhớ, chỉ số đó dừng ở khoảng hai tỉ phần tử.
          </p>
        }
      >
        <p>
          Sixty-four-bit pointers did not remove every 32-bit limit from kernels. They moved it into code you write. The
          kernels of the early chapters compute their index as an <code>int</code>, and an <code>int</code> holds at most 2
          <sup>31</sup> − 1. However much memory the device has, that index stops at about two billion elements.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Giới hạn này thuần túy là C++, không phải một sự thật về GPU. <a href={href('ch03', '3.4')}>Chương 3, §3.4</a>{' '}
              cho thấy PyTorch, llama.cpp và MLX xử lý nó thế nào.
            </p>
          }
        >
          <p>
            The cap is plain C++, not a GPU fact. <a href={href('ch03', '3.4')}>Chapter 3, §3.4</a> shows how PyTorch,
            llama.cpp and MLX deal with it.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsSC4['appC/C.4/q-int-index']} />
      <Translated
        vi={
          <p>
            Điều đó có quan trọng hay không tùy vào device. Đây là những gì device của bạn báo cáo, đặt cạnh 8 GiB số float mà
            một <code>int</code> với tới được.
          </p>
        }
      >
        <p>
          Whether that matters depends on the device. Here is what yours reported, set against the 8 GiB of floats an{' '}
          <code>int</code> reaches.
        </p>
      </Translated>
      <YourMemory />
      <Translated
        vi={
          <p>
            Trình khám phá đặt hai giới hạn cạnh nhau. Kéo độ rộng địa chỉ và xem tầm với tăng gấp đôi theo mỗi bit; các độ
            rộng được đánh dấu là những độ rộng phụ lục này nhắc tới, và các giá trị của thiết bị bạn nằm dưới thanh. Sau đó
            chọn một kiểu phần tử và một kiểu chỉ số và xem cái nào cạn trước, chỉ số hay địa chỉ. Hãy thử một{' '}
            <code>int</code> trên float ở 32 bit, rồi ở 40.
          </p>
        }
      >
        <p>
          The explorer puts the two limits side by side. Drag the address width and watch the reach double with each bit;
          the marked widths are the ones this appendix mentions, and your devices' figures sit under the bar. Then choose an
          element type and an index type and see which runs out first, the index or the address. Try an{' '}
          <code>int</code> over floats at 32 bits, then at 40.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Tầm với chỉ đếm các byte định địa chỉ được. Thiết bị và hệ điều hành thật dành riêng một số phần của không gian
              địa chỉ, và độ rộng vật lý và ảo không được mô hình hóa riêng.
            </p>
          }
        >
          <p>
            Reach counts addressable bytes only. Real devices and operating systems reserve parts of an address space, and
            physical and virtual widths are not modelled separately.
          </p>
        </Translated>
      </Simplification>
      <BitsToBytes />

      <h2>
        <Translated vi="Trên GPU AMD và Apple">On AMD and Apple GPUs</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách chỉ nói về NVIDIA. Trên GPU AMD, trình biên dịch chỉ hỗ trợ không gian địa chỉ tiến trình 64-bit, và bộ nhớ
            chia sẻ (LDS) cùng scratch riêng của mỗi luồng là các cửa sổ 32-bit bên trong một không gian flat 64-bit. Trên
            Metal, mỗi buffer có một địa chỉ ảo GPU, <code>gpuAddress</code>, kiểu <code>MTLGPUAddress</code>, một giá trị
            64-bit.
          </p>
        }
      >
        <p>
          The book covers only NVIDIA. On AMD GPUs the compiler supports only 64-bit process address spaces, and shared
          memory (LDS) and per-thread scratch are 32-bit windows inside one flat 64-bit space. On Metal, every buffer has a
          GPU virtual address, <code>gpuAddress</code>, of type <code>MTLGPUAddress</code>, a 64-bit value.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Kiểm tra ngày 2026-11-20, đối chiếu với tài liệu AMDGPU của LLVM và các header của macOS 27.0 SDK (appC-facts.md,
              B1, C3).
            </p>
          }
        >
          <p>
            Checked 2026-11-20 against LLVM's AMDGPU documentation and the macOS 27.0 SDK headers (appC-facts.md, B1, C3).
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Việc <Id>gpuAddress</Id> của một buffer Metal có bằng con trỏ CPU của nó hay không không được ghi trong tài liệu.
            </p>
          }
        >
          <p>
            Whether a Metal buffer's <Id>gpuAddress</Id> equals its CPU pointer is not documented.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Quay lại <a href={href('appC', 'start')}>bản đồ</a>: C.4 và C.5 chung bước được đánh dấu C.4–C.5. Địa chỉ rộng hơn
            tự nó không thay đổi ô nào; những ô duy nhất mà bước đó thay đổi nằm ở hàng của một GPU ngang hàng (peer), đó là
            mục C.5.
          </p>
        }
      >
        <p>
          Back to <a href={href('appC', 'start')}>the map</a>: C.4 and C.5 share the step marked C.4–C.5. Wider addresses
          change no cell on their own; the only cells that step changes are in the row for a peer GPU, which is section C.5.
        </p>
      </Translated>
    </>
  );
}
