import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, NoteGroup as Notes, Simplification, Source, Today } from '../../components/notes';
import { fmt, realBackends, useReports, type BackendRun, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  BOOK_DEVICE,
  ELEM_LABEL,
  WELL_OPTIMISED,
  WRITTEN_LABEL,
  bestRegisterTiled,
  blockSizes,
  ceiling,
  copyGBs,
  intensityLimit,
  kernelRates,
  labRows,
  measuredBandwidth,
  scanBytes,
  threshold,
  toppBytesPerElement,
  type Elem,
  type KernelRate,
  type Row,
  type RunLike,
  type Written,
} from '../../sim/scantraffic';

// Section 11.8: a scan's operations per byte, the bandwidth ceiling in elements per second,
// and the 80% yardstick. Every number comes from src/sim/scantraffic.ts; the reader's own
// figures come from the lab benchmarks through useReports().

const LAB2 = 'ch11/lab2_';
const fmtG = (elementsPerSec: number) => fmt(elementsPerSec / 1e9);
const pct = (x: number) => `${Math.round(x * 100)}%`;


// ---------------------------------------------------------------- devices
interface Device {
  key: string;
  label: string;
  /** GB/s */
  bw: number;
  /** FLOP/B, when known */
  threshold?: number;
  note: string;
  /** Lab 11.2's rows on this backend, if any. */
  rows: Row[];
}
const BOOK: Device = {
  key: 'book',
  get label() {
    return tr("The book's example (H100 SXM, 3.35 TB/s)", 'Ví dụ của sách (H100 SXM, 3.35 TB/s)');
  },
  bw: BOOK_DEVICE.bandwidthGBs,
  threshold: threshold(BOOK_DEVICE.tflopsNvidia, BOOK_DEVICE.bandwidthGBs),
  get note() {
    return tr("The book's example device, p. 272: 3.35 TB/s and a threshold of 20 FLOP/B.", 'Thiết bị ví dụ của sách, p. 272: 3.35 TB/s và ngưỡng 20 FLOP/B.');
  },
  rows: [],
};

/** This machine's backends with a measured copy rate (lab 11.2's copy, else lab 11.1's, else lab 5.1's probe). */
function measuredDevices(reports: Reports): Device[] {
  return realBackends(reports).flatMap((b) => {
    const run = reports[b] as BackendRun & RunLike;
    const m = measuredBandwidth(run);
    if (!m) return [];
    return [
      {
        key: b,
        get label() {
          return tr(`Yours: ${String(run.device.name ?? b)} (${b})`, `Của bạn: ${String(run.device.name ?? b)} (${b})`);
        },
        bw: m.gbs,
        threshold: m.threshold,
        get note() {
          return tr(
            `Measured on ${run.generated.slice(0, 10)} by ${m.source}: ${fmt(m.gbs)} GB/s. That is what a copy achieves here, not the vendor's peak.`,
            `Đo ngày ${run.generated.slice(0, 10)} bằng ${m.source}: ${fmt(m.gbs)} GB/s. Đó là mức một phép sao chép đạt được ở đây, không phải đỉnh do nhà cung cấp công bố.`,
          );
        },
        rows: labRows(run, LAB2),
      },
    ];
  });
}

// ---------------------------------------------------------------- the share-of-copy chart
function ShareChart({ rates }: { rates: KernelRate[] }) {
  const W = 760;
  const ROW = 22;
  const L = 250;
  const top = 24;
  const H = top + rates.length * ROW + 10;
  const max = Math.max(1.1, ...rates.map((r) => r.share)) * 1.02;
  const X = (s: number) => L + (s / max) * (W - L - 16);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr("Each kernel's rate as a share of the copy rate", 'Tốc độ của mỗi kernel tính theo tỉ lệ so với tốc độ sao chép')} style={{ width: '100%', maxWidth: W, minWidth: 640, display: 'block' }}>
      <line x1={X(WELL_OPTIMISED)} x2={X(WELL_OPTIMISED)} y1={top - 6} y2={H - 6} stroke="var(--ink)" strokeDasharray="4 3" />
      <text x={X(WELL_OPTIMISED)} y={top - 10} fontSize="11" textAnchor="middle">80%</text>
      <line x1={X(1)} x2={X(1)} y1={top - 6} y2={H - 6} stroke="var(--ink-3)" />
      <text x={X(1)} y={top - 10} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi="sao chép">copy</Translated></text>
      {rates.map((r, i) => {
        const y = top + i * ROW;
        const good = r.share >= WELL_OPTIMISED;
        return (
          <g key={i} data-share={r.share.toFixed(4)} data-label={r.label}>
            <text x={L - 8} y={y + 13} fontSize="11.5" textAnchor="end">{r.label.replace(/, \d+ threads$/, '')}</text>
            <rect x={L} y={y + 3} width={Math.max(1, X(r.share) - L)} height={ROW - 8} rx="2"
              fill={good ? 'var(--ok)' : 'var(--ink-3)'} opacity={good ? 0.85 : 0.6} />
            <text x={X(r.share) + 5} y={y + 13} fontSize="11">{pct(r.share)}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------- explorer: the scan ceiling
function CeilingExplorer() {
  const { reports } = useReports();
  const devices = useMemo(() => [BOOK, ...measuredDevices(reports)], [reports]);
  const [key, setKey] = useState('book');
  const [custom, setCustom] = useState(800);
  const [elem, setElem] = useState<Elem>('float');
  const [written, setWritten] = useState<Written>('scan');
  const dev: Device =
    key === 'custom'
      ? { key: 'custom', label: tr('Custom', 'Tùy chỉnh'), bw: custom, note: tr('A bandwidth you typed.', 'Một băng thông bạn tự nhập.'), rows: [] }
      : (devices.find((d) => d.key === key) ?? BOOK);

  const bytes = scanBytes(elem, written);
  const opb = intensityLimit(bytes.total);
  const cap = ceiling(dev.bw, bytes.total);
  const copy = copyGBs(dev.rows);
  const sizes = blockSizes(dev.rows);
  const [blockPick, setBlock] = useState<number | null>(null);
  const block = blockPick != null && sizes.includes(blockPick) ? blockPick : (sizes[sizes.length - 1] ?? null);
  const rows = block == null ? dev.rows : dev.rows.filter((r) => r.block === block);
  const rates = copy ? kernelRates(rows, copy) : [];
  const best = bestRegisterTiled(rates);

  return (
    <Bench
      title={tr('The scan ceiling', 'Trần của scan')}
      task={tr("Find the ceiling for floats on the book's example, then switch to half precision and to your own device.", 'Tìm trần cho float trên ví dụ của sách, rồi chuyển sang độ chính xác half và sang thiết bị của chính bạn.')}
      foot={
        <Translated
          vi={
            <>
              Lưu lượng lý tưởng: mỗi phần tử được nạp một lần và mỗi kết quả được ghi một lần, không gì khác. Phép toán là các phép
              cộng của scan, N − 1 với N phần tử; với N lớn thì là 1 mỗi phần tử. {dev.note}
            </>
          }
        >
          <>
            Ideal traffic: every element loaded once and every result stored once, nothing else. Operations are the scan's
            additions, N − 1 for N elements; for large N that is 1 per element. {dev.note}
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi="Thiết bị">Device</Translated>
          <select value={key} onChange={(e) => setKey(e.target.value)} aria-label={tr('device', 'thiết bị')}>
            {devices.map((d) => <option key={d.key} value={d.key}>{d.label}</option>)}
            <option value="custom">{tr('Custom bandwidth', 'Băng thông tùy chỉnh')}</option>
          </select>
        </label>
        {key === 'custom' ? (
          <label className="field">
            <Translated vi="Băng thông (GB/s)">Bandwidth (GB/s)</Translated>
            <input type="number" min="1" value={custom} aria-label={tr('bandwidth', 'băng thông')} onChange={(e) => setCustom(Math.max(1, +e.target.value || 1))} />
          </label>
        ) : null}
        <label className="field">
          <Translated vi="Kiểu phần tử">Element type</Translated>
          <span className="seg" role="group" aria-label={tr('Element type', 'Kiểu phần tử')}>
            {(['float', 'half'] as Elem[]).map((e) => (
              <button key={e} aria-pressed={elem === e} onClick={() => setElem(e)}>{ELEM_LABEL[e]}</button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi="Kernel ghi ra gì">What the kernel writes</Translated>
          <span className="seg" role="group" aria-label={tr('What is written', 'Thứ được ghi ra')}>
            {(['scan', 'scan+mask'] as Written[]).map((w) => (
              <button key={w} aria-pressed={written === w} onClick={() => setWritten(w)}>{WRITTEN_LABEL[w]}</button>
            ))}
          </span>
        </label>
      </div>
      <div className="cols">
        <div className="panel" data-testid="ceiling">
          <Translated
            vi={
              <p style={{ margin: 0 }}>
                Mỗi phần tử: <b data-testid="load">{bytes.load}</b> byte vào, <b data-testid="store">{bytes.store}</b> byte ra, tổng
                cộng <b data-testid="bytes">{bytes.total}</b>.
              </p>
            }
          >
            <p style={{ margin: 0 }}>
              Per element: <b data-testid="load">{bytes.load}</b> bytes in, <b data-testid="store">{bytes.store}</b> bytes out,{' '}
              <b data-testid="bytes">{bytes.total}</b> in all.
            </p>
          </Translated>
          <Translated
            vi={
              <p style={{ margin: '0.4rem 0 0' }}>
                Cường độ tính toán: 1 phép cộng mỗi {bytes.total} byte = <b data-testid="opb">{fmt(opb)}</b> OP/B.
              </p>
            }
          >
            <p style={{ margin: '0.4rem 0 0' }}>
              Intensity: 1 addition per {bytes.total} bytes = <b data-testid="opb">{fmt(opb)}</b> OP/B.
            </p>
          </Translated>
          <p style={{ fontSize: '1.5rem', margin: '0.6rem 0 0.2rem', fontWeight: 600 }}>
            <span data-testid="cap">{fmtG(cap)}</span> <Translated vi="× 10⁹ phần tử/s">× 10⁹ elements/s</Translated>
          </p>
          <p className="muted" style={{ margin: 0 }}>
            <Translated
              vi={
                <>
                  {fmt(dev.bw)} GB/s ÷ {bytes.total} byte mỗi phần tử: mức nhiều nhất thiết bị này có thể scan mỗi giây, bất kể kernel
                  được viết thế nào.
                </>
              }
            >
              {fmt(dev.bw)} GB/s ÷ {bytes.total} bytes per element: the most this device can scan per second, however the kernel
              is written.
            </Translated>
          </p>
          {dev.threshold ? (
            <p style={{ margin: '0.6rem 0 0' }} data-testid="regime">
              <Translated
                vi={
                  <>
                    <span className="chip global">bị giới hạn bởi bộ nhớ</span> Ngưỡng của thiết bị là {fmt(dev.threshold)} FLOP/B
                    {key === 'book' ? '' : ' (lab 5.1)'}; scan cung cấp {fmt(opb)}, ít hơn khoảng <b>{fmt(dev.threshold / opb)}</b> lần.
                  </>
                }
              >
                <span className="chip global">memory-bound</span> The device's threshold is {fmt(dev.threshold)} FLOP/B{key === 'book' ? '' : ' (lab 5.1)'}; the scan
                supplies {fmt(opb)}, about <b>{fmt(dev.threshold / opb)}</b> times less.
              </Translated>
            </p>
          ) : null}
        </div>
        <div className="panel" data-testid="measured">
          {key === 'book' || key === 'custom' ? (
            <Translated
              vi={
                <p style={{ margin: 0 }}>
                  Chọn thiết bị của bạn để xem các kernel của lab 11.2 so với kernel sao chép của bạn. Một đỉnh được công bố cho biết bộ
                  nhớ có thể làm gì; kernel sao chép của bạn cho biết một kernel không có việc gì khác thực sự đạt được bao nhiêu.
                </p>
              }
            >
              <p style={{ margin: 0 }}>
                Pick your own device to see lab 11.2's kernels against your copy kernel. A published peak tells you what the
                memory could do; your copy kernel tells you what a kernel with nothing else to do actually gets.
              </p>
            </Translated>
          ) : rates.length ? (
            <Translated
              vi={
                <p style={{ margin: 0 }}>
                  <b>Lab 11.2 trên thiết bị này</b>: bên dưới là tốc độ của mỗi kernel tính theo tỉ lệ so với kernel sao chép ở cùng
                  kích thước khối (8 byte mỗi phần tử), so với vạch 80% của sách. Tốt nhất ở {block} luồng:{' '}
                  <b>{best!.rate.label.replace(/, \d+ threads$/, '')}</b>, {pct(best!.rate.share)}.
                </p>
              }
            >
              <p style={{ margin: 0 }}>
                <b>Lab 11.2 on this device</b>: below, each kernel's rate as a share of the copy kernel's at the same block size
                (8 bytes per element), against the book's 80% line. Best at {block} threads:{' '}
                <b>{best!.rate.label.replace(/, \d+ threads$/, '')}</b>, {pct(best!.rate.share)}.
              </p>
            </Translated>
          ) : (
            <Translated
              vi={
                <p style={{ margin: 0 }}>
                  Thiết bị này chưa có benchmark nào của lab 11.2. Hãy chạy <code>python3 labs/pmpp.py run 11.2 --solution --bench</code>,
                  rồi tải lại: mỗi kernel của nó sẽ hiện ở đây theo tỉ lệ so với tốc độ sao chép, đối chiếu với vạch 80%.
                </p>
              }
            >
              <p style={{ margin: 0 }}>
                No lab 11.2 benchmark on this device yet. Run <code>python3 labs/pmpp.py run 11.2 --solution --bench</code>, then
                reload: each of its kernels appears here as a share of the copy rate, against the 80% line.
              </p>
            </Translated>
          )}
        </div>
      </div>
      {key !== 'book' && key !== 'custom' && rates.length ? (
        <div style={{ marginTop: '0.9rem' }} data-testid="chart">
          {sizes.length > 1 ? (
            <div className="row" style={{ marginBottom: '0.5rem' }}>
              <span className="seg" role="group" aria-label={tr('Block size', 'Kích thước khối')}>
                {sizes.map((b) => (
                  <button key={b} aria-pressed={b === block} onClick={() => setBlock(b)}><Translated vi={<>{b} luồng</>}>{b} threads</Translated></button>
                ))}
              </span>
            </div>
          ) : null}
          <ShareChart rates={rates} />
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- measured reveal
function CopyReveal({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = realBackends(reports).flatMap((b) => {
    const rows = labRows(reports[b] as BackendRun & RunLike, LAB2);
    const copy = copyGBs(rows);
    const best = copy ? bestRegisterTiled(kernelRates(rows, copy)) : undefined;
    return best ? [{ b, device: String(reports[b].device.name ?? b), ...best }] : [];
  });
  if (!found.length)
    return (
      <Translated
        vi={
          <p>
            Máy này chưa có số đo nào. Hãy chạy <code>python3 labs/pmpp.py run 11.2 --solution --bench</code> trên một máy có GPU,
            rồi tải lại: ô này sẽ hiện tốc độ phân ô thanh ghi tốt nhất theo tỉ lệ so với kernel sao chép của bạn.
          </p>
        }
      >
        <p>
          No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 11.2 --solution --bench</code> on a machine with
          a GPU, then reload: this box will show the best register-tiled rate as a share of your copy kernel's.
        </p>
      </Translated>
    );
  return (
    <>
      {found.map((f) => (
        <p key={f.b} data-backend={f.b} data-share={f.rate.share.toFixed(4)}>
          <Translated
            vi={
              <>
                Trên <b>{f.device}</b> ({f.b}), {f.marked ? 'kernel phân ô thanh ghi' : 'kernel'} tốt nhất, <b>{f.rate.label}</b>, đạt{' '}
                <b>{pct(f.rate.share)}</b> tốc độ sao chép, {Math.abs(f.rate.share * 100 - predicted) <= 5 ? 'lệch không quá 5 điểm so với dự đoán của bạn' : f.rate.share * 100 > predicted ? `cao hơn dự đoán của bạn ${Math.round(f.rate.share * 100 - predicted)} điểm` : `thấp hơn dự đoán của bạn ${Math.round(predicted - f.rate.share * 100)} điểm`}
                {f.rate.share >= WELL_OPTIMISED ? ', vượt vạch 80% của sách.' : ', chưa tới vạch 80% của sách.'}
              </>
            }
          >
            On <b>{f.device}</b> ({f.b}) the best {f.marked ? 'register-tiled kernel' : 'kernel'}, <b>{f.rate.label}</b>, reached{' '}
            <b>{pct(f.rate.share)}</b> of the copy rate, {Math.abs(f.rate.share * 100 - predicted) <= 5 ? 'within 5 points of your prediction' : f.rate.share * 100 > predicted ? `${Math.round(f.rate.share * 100 - predicted)} points above your prediction` : `${Math.round(predicted - f.rate.share * 100)} points below your prediction`}
            {f.rate.share >= WELL_OPTIMISED ? ', past the book\'s 80% line.' : ', short of the book\'s 80% line.'}
          </Translated>
        </p>
      ))}
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS118: Record<string, Question> = {
  'ch11/11.8/q-half': {
    kind: 'numeric',
    id: 'ch11/11.8/q-half',
    prompt: (
      <Translated
        vi={
          <>
            Một scan đọc các giá trị half, cộng chúng trong một biến tích lũy float, và ghi kết quả half. Với N lớn, cường độ tính
            toán của nó tính theo OP/B là bao nhiêu?
          </>
        }
      >
        <>
          A scan reads half-precision values, adds them in a float accumulator, and writes half-precision results. What is its
          intensity in OP/B, for large N?
        </>
      </Translated>
    ),
    answer: 0.25,
    tol: 0.02,
    unit: 'OP/B',
    vi: { unit: 'OP/B', source: 'phép đếm của §11.8 (p. 272) áp dụng cho phần tử 2 byte; của khóa học này' },
    explain: (
      <Translated
        vi={
          <p>
            N − 1 phép cộng trên 2N byte đọc và 2N byte ghi: khoảng 1/4. Độ rộng của biến tích lũy không di chuyển byte nào; chỉ
            những gì được nạp từ và ghi vào bộ nhớ toàn cục mới tính. Giảm một nửa kích thước phần tử thì cường độ tính toán và trần
            tính theo phần tử mỗi giây tăng gấp đôi, còn scan vẫn bị giới hạn bởi bộ nhớ y như cũ: 0.25 vẫn thấp hơn ngưỡng của sách
            80 lần.
          </p>
        }
      >
        <p>
          N − 1 additions over 2N bytes read and 2N written: about 1/4. The accumulator's width moves no bytes; only what is
          loaded from and stored to global memory counts. Halving the element size doubles the intensity and the ceiling in
          elements per second, and leaves the scan just as memory-bound: 0.25 is still 80 times below the book's threshold.
        </p>
      </Translated>
    ),
    source: 'the §11.8 count (p. 272) applied to 2-byte elements; ours',
  },
  'ch11/11.8/q-ceiling': {
    kind: 'numeric',
    id: 'ch11/11.8/q-ceiling',
    prompt: (
      <Translated
        vi={
          <>
            Trên một thiết bị nào đó, một kernel sao chép đơn thuần đo được 800 GB/s. Một kernel một lượt scan float ở đó. Trần của
            nó là bao nhiêu, tính theo tỉ phần tử mỗi giây?
          </>
        }
      >
        <>
          On some device a plain copy kernel measures 800 GB/s. A single-pass kernel scans floats there. What is its ceiling, in
          billions of elements per second?
        </>
      </Translated>
    ),
    answer: 100,
    tol: 0.01,
    unit: '× 10⁹ elements/s',
    vi: { unit: '× 10⁹ phần tử/s', source: 'ch. 11 §11.8, p. 272 (phương pháp); thiết bị 800 GB/s là giả định' },
    explain: (
      <Translated
        vi={
          <p>
            Mỗi phần tử tốn ít nhất 8 byte: 4 đọc, 4 ghi. 800 × 10⁹ byte mỗi giây ÷ 8 = 100 × 10⁹ phần tử mỗi giây. Dùng tốc độ
            đo được của kernel sao chép thay vì đỉnh được công bố cho ra một trần mà thiết bị này thực sự đạt được với một kernel
            không làm gì khác.
          </p>
        }
      >
        <p>
          Each element costs at least 8 bytes: 4 read, 4 written. 800 × 10⁹ bytes per second ÷ 8 = 100 × 10⁹ elements per
          second. Using the copy kernel's measured rate instead of a published peak gives a ceiling this device actually
          reaches with a kernel that does nothing else.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.8, p. 272 (the method); the 800 GB/s device is made up',
  },
  'ch11/11.8/q-topp-bytes': {
    kind: 'numeric',
    id: 'ch11/11.8/q-topp-bytes',
    prompt: (
      <Translated
        vi={
          <>
            Kernel top-p gộp của lab 11.6 đọc một xác suất float mỗi phần tử và ghi một khối loại trừ float cùng một cờ giữ
            một byte mỗi phần tử. Nó di chuyển bao nhiêu byte mỗi phần tử?
          </>
        }
      >
        <>
          The fused top-p kernel of lab 11.6 reads one float probability per element and writes a float exclusive mass and a
          one-byte keep flag per element. How many bytes per element does it move?
        </>
      </Translated>
    ),
    answer: toppBytesPerElement(),
    unit: 'bytes',
    vi: { unit: 'byte', source: 'lab 11.6 (của khóa học này); phép đếm là của §11.8, p. 272' },
    explain: (
      <Translated
        vi={
          <p>
            4 vào, 4 + 1 ra: 9 byte. Trần của nó là băng thông chia 9, thấp hơn một chút so với băng thông chia 8 của một scan đơn
            thuần. Phương án khác, các kernel riêng cho scan và cho ngưỡng, sẽ đọc các xác suất hai lần, ghi khối xác suất rồi đọc
            lại: mỗi lượt đi qua hàng lại thêm những byte mà kernel gộp không bao giờ di chuyển.
          </p>
        }
      >
        <p>
          4 in, 4 + 1 out: 9 bytes. Its ceiling is the bandwidth over 9, a little below a plain scan's bandwidth over 8. The
          alternative, separate kernels for the scan and the threshold, would read the probabilities twice and write the mass
          and read it back: every pass over the row adds bytes the fused kernel never moves.
        </p>
      </Translated>
    ),
    source: 'lab 11.6 (ours); the counting is §11.8, p. 272',
  },
  'ch11/11.8/q-80': {
    kind: 'choice',
    id: 'ch11/11.8/q-80',
    prompt: (
      <Translated
        vi={
          <>
            Profile của kernel scan của bạn cho thấy 85% băng thông được công bố. Nhìn kỹ hơn, nó đọc toàn bộ đầu vào hai lần. Theo
            quy tắc của sách, nó có được tối ưu tốt không?
          </>
        }
      >
        <>
          Your scan kernel's profile says 85% of the published bandwidth. Looking closer, it reads its whole input twice. Is it
          well optimised by the book's rule?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Có: mọi thứ trên 80% băng thông công bố đều được coi là tối ưu tốt.">Yes: anything above 80% of the published bandwidth counts as well optimised.</Translated>,
        why: <Translated vi="Quy tắc có hai phần. Đọc lại p. 273: trên 80% và không có truy cập global thừa.">The rule has two parts. Read p. 273 again: above 80% and no redundant global accesses.</Translated>,
      },
      {
        label: <Translated vi="Không: quy tắc còn yêu cầu mỗi vị trí global chỉ bị chạm một lần, và kernel này không như vậy.">No: the rule also requires every global location to be touched once, and this kernel does not.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Không: một scan không bao giờ vượt 80%, nên số đo hẳn là sai.">No: a scan can never pass 80%, so the measurement must be wrong.</Translated>,
        why: <Translated vi="Sách nói Fig. 11.13 có thể vượt 80% (p. 273).">The book says Fig. 11.13 can plausibly exceed 80% (p. 273).</Translated>,
      },
      {
        label: <Translated vi="Không thể đánh giá nếu không biết FLOP/s của thiết bị.">{'It cannot be judged without knowing the device\'s FLOP/s.'}</Translated>,
        why: <Translated vi="Ở 1/8 OP/B, scan bị giới hạn bởi bộ nhớ trên mọi thiết bị mà sách xét; chỉ số byte là quan trọng.">At 1/8 OP/B the scan is memory-bound on any device the book considers; only bytes matter.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Băng thông đo số byte được di chuyển, không phải khối lượng việc đã làm. Đọc đầu vào hai lần biến 8 byte mỗi phần tử
            thành 12, nên 85% băng thông chỉ cho 85% × 8/12 ≈ 57% số phần tử mỗi giây mà một kernel chạm mỗi vị trí một lần có thể
            đạt. Nửa sau của quy tắc, không truy cập thừa, là thứ ngăn một kernel lãng phí trông có vẻ tốt.
          </p>
        }
      >
        <p>
          Bandwidth measures bytes moved, not work done. Reading the input twice turns 8 bytes per element into 12, so 85% of
          the bandwidth delivers 85% × 8/12 ≈ 57% of the elements per second a kernel touching each location once could reach.
          The second half of the rule, no redundant accesses, is what stops a wasteful kernel from looking good.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.8, pp. 272–273',
  },
};

export function S118() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mục 11.7 để lại cho bạn một scan khối chạm bộ nhớ chia sẻ một lần mỗi phần tử khi vào và một lần khi ra. Trước khi đi
            tiếp, hãy hỏi scan nhanh nhất có thể là gì. Với scan, câu trả lời chỉ cần ba con số và một phép chia.
          </p>
        }
      >
        <p>
          Section 11.7 left you with a block scan that touches shared memory once per element on the way in and once on the way
          out. Before going further, ask what the fastest possible scan would be. For a scan the answer takes three numbers and
          one division.
        </p>
      </Translated>
      <Source section="11.8" pages="272" />

      <h2><Translated vi="Đếm số byte">Count the bytes</Translated></h2>
      <Translated
        vi={
          <p>
            Scan N số float. Kernel phải đọc mọi đầu vào, 4N byte, và ghi mọi kết quả, thêm 4N byte. Ở giữa nó làm N − 1 phép cộng.
            Số phép toán trên mỗi byte lưu lượng global: (N − 1) / 8N, với mọi N thực tế là 1/8 OP/B. Đó là tính chất của phép
            tính, không phải của việc nó được viết khéo đến đâu: không kernel nào có thể làm ít phép cộng hơn hay di chuyển ít byte
            hơn.
          </p>
        }
      >
        <p>
          Scan N floats. The kernel has to read every input, 4N bytes, and write every result, another 4N bytes. In between it
          does N − 1 additions. Operations per byte of global traffic: (N − 1) / 8N, which for any real N is 1/8 OP/B. That is
          a property of the computation, not of how cleverly it is written: no kernel can do fewer additions or move fewer
          bytes.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Hãy so với một thiết bị. GPU ví dụ của sách làm 66.9 TFLOPS so với 3.35 TB/s, ngưỡng 20 FLOP/B (cùng phép tính như mục
            5.1). Một scan ở mức 0.125 thiếu 160 lần so với ngưỡng đó. Các đơn vị số học sẽ dành gần hết thời gian để chờ, nên với
            scan, băng thông là toàn bộ câu chuyện.
          </p>
        }
      >
        <p>
          Compare it with a device. The book's example GPU does 66.9 TFLOPS against 3.35 TB/s, a threshold of 20 FLOP/B (the
          same calculation as section 5.1). A scan at 0.125 is 160 times short of it. The arithmetic units will spend almost
          all their time waiting, so for a scan the bandwidth is the whole story.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 against nvidia.com (notes/today/ch11-facts.md A1; ch05 A8)', '2026-11-20, đối chiếu với nvidia.com (notes/today/ch11-facts.md A1; ch05 A8)')}>
        <Translated
          vi={
            <p>
              NVIDIA in FP32 của H100 SXM là 67 TFLOPS, không phải 66.9, và xác nhận 3.35 TB/s. Đằng nào ngưỡng cũng là 20 FLOP/B, và
              mọi trần bên dưới không đổi.
            </p>
          }
        >
          <p>
            NVIDIA prints FP32 for the H100 SXM as 67 TFLOPS, not 66.9, and confirms 3.35 TB/s. The threshold is 20 FLOP/B either
            way, and every ceiling below is unchanged.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Trần">The ceiling</Translated></h2>
      <Translated
        vi={
          <p>
            Với một kernel bị giới hạn bởi bộ nhớ, băng thông đỉnh quyết định hiệu năng đỉnh. Một scan phải di chuyển ít nhất 8 byte
            mỗi số float, nên số phần tử nhiều nhất nó xử lý được mỗi giây là băng thông chia 8. Trên ví dụ của sách, 3.35 × 10¹² /
            8 = 419 × 10⁹ phần tử mỗi giây. Thiết bị của bạn có con số riêng, và cách trung thực để có nó là lấy mức mà một kernel
            sao chép đơn thuần đạt được ở đó để chia, không phải con số trên trang của nhà cung cấp.
          </p>
        }
      >
        <p>
          For a memory-bound kernel, peak bandwidth sets peak performance. A scan must move at least 8 bytes per float, so the
          most elements it can process per second is the bandwidth divided by 8. On the book's example, 3.35 × 10¹² / 8 = 419 ×
          10⁹ elements per second. Your device has its own figure, and the honest way to get it is to divide what a plain copy
          kernel achieves there, not the number on the vendor's page.
        </p>
      </Translated>
      <Source section="11.8" pages="272" />
      <CeilingExplorer />
      <Translated
        vi={
          <p>
            Hai điều rút ra từ trình khám phá. Cường độ tính toán không bao giờ dịch đi xa: giảm một nửa kích thước phần tử thì nó
            tăng gấp đôi lên 1/4, và một cờ một byte hạ nó xuống 1/9, nhưng mọi lựa chọn đều ở rất xa dưới ngưỡng của sách và của
            thiết bị bạn, nên chế độ không bao giờ đổi. Thứ thực sự dịch chuyển là trần tính theo phần tử mỗi giây, đúng tỉ lệ với số
            byte mỗi phần tử. Với một kernel bị giới hạn bởi bộ nhớ, tiết kiệm một byte là tiết kiệm thời gian.
          </p>
        }
      >
        <p>
          Two things to take from the explorer. The intensity never moves far: halving the element size doubles it to 1/4, and a
          one-byte flag lowers it to 1/9, but every option stays far below the book's threshold and your own device's, so the
          regime never changes. What does move is the ceiling in elements per second, in exact proportion to the bytes per element. For a
          memory-bound kernel, a byte saved is time saved.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Trình khám phá tính mọi byte cho bộ nhớ toàn cục và không gì khác: không bộ nhớ đệm, không chi phí khởi chạy, không khối
              cuối dở dang. Đó là một trần, không phải một dự đoán.
            </p>
          }
        >
          <p>
            The explorer charges every byte to global memory and nothing else: no caches, no launch overhead, no partial last
            block. It is a ceiling, not a prediction.
          </p>
        </Translated>
      </Simplification>
      <Ask q={questionsS118['ch11/11.8/q-half']} />
      <Ask q={questionsS118['ch11/11.8/q-ceiling']} />

      <h2><Translated vi="Gần bao nhiêu là đủ gần">How close is close enough</Translated></h2>
      <Translated
        vi={
          <p>
            Chạm tới trần nghĩa là chạm mỗi vị trí global đúng một lần và giấu mọi phép cộng, mọi rào chắn và mọi truy cập bộ nhớ
            chia sẻ sau lưu lượng global. Sách nói điều đó không xảy ra: ngay cả một kernel chỉ sao chép cũng không đạt 100% băng
            thông đỉnh. Thay vào đó, thước đo của sách: một kernel bị giới hạn bởi bộ nhớ đạt hơn 80% băng thông,{' '}
            <em>không có truy cập global thừa</em>, được coi là tối ưu tốt.
          </p>
        }
      >
        <p>
          Reaching the ceiling would mean touching every global location exactly once and hiding every addition, every barrier
          and every shared-memory access behind the global traffic. The book says that does not happen: even a kernel that only
          copies does not reach 100% of peak bandwidth. Its yardstick instead: a memory-bound kernel that reaches more than 80%
          of the bandwidth, <em>with no redundant global accesses</em>, counts as well optimised.
        </p>
      </Translated>
      <Notes>
      <Source section="11.8" pages="272–273" />
      <Today checked={tr('2026-11-20 against the Nsight Compute documentation v2026.4.0 (notes/today/ch11-facts.md A2)', '2026-11-20, đối chiếu với tài liệu Nsight Compute v2026.4.0 (notes/today/ch11-facts.md A2)')}>
        <Translated
          vi={
            <p>
              Profiler của NVIDIA vạch cùng một đường: hướng dẫn phân loại của nó gọi thông lượng "Above ~80%: high (near-limit
              regime)" (trên ~80%: cao, gần giới hạn). Không tìm thấy nguồn nào của nhà cung cấp cho câu của sách rằng một phép sao
              chép không thể đạt 100%; hãy coi đó là nhận định của sách.
            </p>
          }
        >
          <p>
            NVIDIA's profiler draws the same line: its triage guide calls throughput "Above ~80%: high (near-limit regime)". No
            vendor source was found for the book's sentence that a copy cannot reach 100%; treat that as the book's.
          </p>
        </Translated>
      </Today>
      </Notes>
      <Translated
        vi={
          <p>
            Điều đó gợi ý đo hai lần: so với đỉnh được công bố, và so với kernel sao chép của chính bạn trên cùng thiết bị và cùng
            kích thước. Lần thứ hai cho biết bao nhiêu phần khoảng cách là lỗi của scan và bao nhiêu chỉ là mức hệ thống bộ nhớ cấp
            cho bất kỳ kernel nào. Vì lý do đó, lab 11.2 đo thời gian một phép sao chép bên cạnh mỗi scan nó chạy.
          </p>
        }
      >
        <p>
          That suggests measuring twice: against the published peak, and against your own copy kernel on the same device and
          the same size. The second tells you how much of the gap is the scan's fault and how much is simply what the memory
          system delivers to any kernel. Lab 11.2 times a copy beside every scan it runs, for this reason.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>So sánh với kernel sao chép của chính bạn bên cạnh đỉnh là gợi ý của chúng tôi, không phải của sách.</p>}>
          <p>Comparing with your own copy kernel as well as the peak is our suggestion, not the book's.</p>
        </Translated>
      </Beyond>
      <Predict
        id="ch11/11.8/p-copy"
        prompt={tr("What share of your own copy kernel's bandwidth will lab 11.2's register-tiled kernel reach at its best coarsening factor?", 'Kernel phân ô thanh ghi của lab 11.2 sẽ đạt bao nhiêu phần băng thông của kernel sao chép của chính bạn, ở hệ số gộp việc tốt nhất của nó?')}
        unit="%"
      >
        {(v) => <CopyReveal predicted={v} />}
      </Predict>
      <Ask q={questionsS118['ch11/11.8/q-80']} />

      <h2><Translated vi="Điều scan khối còn bỏ dở">What the block scan leaves undone</Translated></h2>
      <Translated
        vi={
          <p>
            Kernel phân ô thanh ghi của Fig. 11.13 đọc mỗi phần tử một lần và ghi một lần, nên theo đánh giá của sách nó có thể vượt
            80%. Nhưng nó chỉ scan riêng đoạn của từng khối. Biến các đoạn đó thành một scan của cả mảng đòi hỏi mỗi khối biết tổng
            của mọi khối đứng trước, và điều đó tốn thêm truy cập global và thêm đồng bộ hóa. Thêm bao nhiêu là chủ đề của{' '}
            <a href={href('ch11', '11.9')}>mục 11.9</a>, và nó được đếm đúng bằng đơn vị này: byte mỗi phần tử.
          </p>
        }
      >
        <p>
          The register-tiled kernel of Fig. 11.13 reads each element once and writes it once, so by the book's judgement it can
          plausibly pass 80%. But it only scans each block's segment on its own. Turning those segments into one scan of the
          whole array needs every block to learn the total of all the blocks before it, and that costs more global accesses and
          more synchronisation. How many more is the subject of <a href={href('ch11', '11.9')}>section 11.9</a>, and it is
          counted in exactly these units: bytes per element.
        </p>
      </Translated>
      <Source section="11.8" pages="273" />

      <h2><Translated vi="Bên dưới một bộ lấy mẫu">Underneath a sampler</Translated></h2>
      <Translated
        vi={
          <p>
            Tổng tích lũy trên một hàng xác suất, bước cốt lõi của lấy mẫu top-p, chính là phép tính này: một phép cộng mỗi phần tử,
            1/8 OP/B với float. Chạy nó bằng các kernel riêng cho chuẩn hóa softmax, tổng tích lũy và ngưỡng, thì mỗi kernel đọc
            hàng và ghi một kết quả mà kernel sau đọc lại. Gộp thành một lượt, hàng được đọc một lần và khối xác suất cùng các
            cờ giữ được ghi một lần: 9 byte mỗi phần tử trong lab 11.6. Đó là lập luận của mục 11.9, nạp một lần và ghi một lần, áp
            dụng cho mốc dự án M5.
          </p>
        }
      >
        <p>
          A cumulative sum over one row of probabilities, the step at the heart of top-p sampling, is this same computation: one
          addition per element, 1/8 OP/B in float. Run it as separate kernels for softmax normalisation, cumulative sum and
          threshold, and each kernel reads the row and writes a result the next one reads back. Fused into one pass, the row is
          read once and the mass and the keep flags are written once: 9 bytes per element in lab 11.6. That is section 11.9's
          argument, load once and store once, applied to milestone M5.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không nối scan với lấy mẫu. Phép ánh xạ và lab 11.6 là của chúng tôi; phép đếm là của sách (p. 272).
            </p>
          }
        >
          <p>
            The book does not connect scan to sampling. The mapping and lab 11.6 are ours; the counting is the book's (p. 272).
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS118['ch11/11.8/q-topp-bytes']} />
    </>
  );
}
