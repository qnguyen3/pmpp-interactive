import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  BOOK_MEMORY,
  elementRate,
  histogramBound,
  peakBandwidth,
  perLocationRate,
  skewSlowdown,
  uniformShare,
} from '../../sim/atomicrate';
import { histogram, hottest, preset } from '../../sim/histogram';
import { AtomicWaitScene } from './AtomicWaitScene';

const LAB = 'ch09/lab1_racehunt';

/** "2.5 M/s", "640 M/s", "32 G/s". */
function rate(v: number): string {
  if (!isFinite(v)) return 'n/a';
  if (v >= 1e9) return `${fmt(v / 1e9)} G/s`;
  if (v >= 1e6) return `${fmt(v / 1e6)} M/s`;
  if (v >= 1e3) return `${fmt(v / 1e3)} k/s`;
  return `${fmt(v)} /s`;
}

// ---------------------------------------------------------------- explorer
type Place = 'dram' | 'llc' | 'shared';
const PLACES: { id: Place; label: string; min: number; max: number; start: number; words: string }[] = [
  {
    id: 'dram',
    get label() {
      return tr('DRAM', 'DRAM');
    },
    min: 100,
    max: 800,
    start: 200,
    get words() {
      return tr("the book's example is 200", 'ví dụ của sách là 200');
    },
  },
  {
    id: 'llc',
    get label() {
      return tr('last-level cache', 'bộ nhớ đệm cấp cuối');
    },
    min: 10,
    max: 90,
    start: 30,
    get words() {
      return tr('the book: "tens of cycles"; starting value illustrative', 'sách: "hàng chục chu kỳ"; giá trị khởi đầu chỉ để minh họa');
    },
  },
  {
    id: 'shared',
    get label() {
      return tr('shared memory', 'bộ nhớ chia sẻ');
    },
    min: 1,
    max: 9,
    start: 4,
    get words() {
      return tr('the book: "a few cycles"; starting value illustrative', 'sách: "vài chu kỳ"; giá trị khởi đầu chỉ để minh họa');
    },
  },
];
type Dist = 'uniform' | 'bright' | 'constant' | 'custom';
const DISTS: { id: Dist; label: string }[] = [
  {
    id: 'uniform',
    get label() {
      return tr('uniform', 'đều');
    },
  },
  {
    id: 'bright',
    get label() {
      return tr('a bright image', 'một ảnh sáng');
    },
  },
  {
    id: 'constant',
    get label() {
      return tr('constant', 'hằng');
    },
  },
  {
    id: 'custom',
    get label() {
      return tr('set the busiest bin', 'đặt ô bận nhất');
    },
  },
];
const BIN_CHOICES = [1, 2, 4, 8, 16, 32, 64, 128, 256];
/** Share of the busiest bin in our sky picture, binned into `bins` equal intervals. */
const brightShare = (bins: number) => hottest(histogram(preset('sky').values, 256 / bins)).share;

function LogBars({ rows }: { rows: { label: string; v: number; tone: string }[] }) {
  const W = 440;
  const L = 118;
  const lo = 1e5;
  const hi = 1e11;
  const X = (v: number) => L + ((Math.log10(Math.max(lo, v)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (W - L - 22);
  return (
    <svg viewBox={`0 0 ${W} ${rows.length * 26 + 26}`} role="img" aria-label={tr('Rates on a log scale', 'Các tốc độ trên thang logarit')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      {[1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11].map((t) => (
        <g key={t}>
          <line x1={X(t)} x2={X(t)} y1={4} y2={rows.length * 26 + 6} stroke="var(--rule-soft)" />
          <text x={X(t)} y={rows.length * 26 + 20} fontSize="10.5" textAnchor="middle">{rate(t).replace('/s', '')}</text>
        </g>
      ))}
      {rows.map((r, i) => (
        <g key={r.label}>
          <text x={L - 6} y={i * 26 + 20} fontSize="11.5" textAnchor="end">{r.label}</text>
          <rect x={L} y={i * 26 + 8} width={Math.max(1, X(r.v) - L)} height={15} fill={r.tone} />
        </g>
      ))}
    </svg>
  );
}

function measuredSkew(reports: Reports) {
  return realBackends(reports).flatMap((b) => {
    const rows = benchRows(reports[b], LAB).filter((r) => r.name === 'basic');
    const uni = num(rows.find((r) => r.data === 'uniform')?.gpixels_per_s);
    const con = num(rows.find((r) => r.data === 'constant')?.gpixels_per_s);
    return uni && con ? [{ b, device: String(reports[b].device.name ?? b), uni, con, ratio: uni / con }] : [];
  });
}

// The explorer shows the model only. The reader's measured slowdown is the answer to the
// prediction further down (ch09/9.3/p-skew-slowdown), so it appears only in that
// prediction's reveal (MeasuredSkew), after the reader has committed a number.
function ThroughputExplorer() {
  const [place, setPlace] = useState<Place>('dram');
  const [lat, setLat] = useState<Record<Place, number>>({ dram: 200, llc: 30, shared: 4 });
  const [book, setBook] = useState(true);
  const [ghz, setGhz] = useState(1.5);
  const [bins, setBins] = useState(256);
  const [dist, setDist] = useState<Dist>('uniform');
  const [custom, setCustom] = useState(0.5);

  const P = PLACES.find((x) => x.id === place)!;
  const clockHz = book ? BOOK_MEMORY.clockHz : ghz * 1e9;
  const mem = { ...BOOK_MEMORY, clockHz };
  const cap = elementRate(mem);
  const one = perLocationRate(clockHz, lat[place]);
  const share =
    dist === 'uniform' ? uniformShare(bins) : dist === 'constant' ? 1 : dist === 'bright' ? brightShare(bins) : Math.max(custom, 1 / bins);
  const bound = histogramBound(one, share, cap);
  const reset = () => {
    setPlace('dram');
    setLat({ dram: 200, llc: 30, shared: 4 });
    setBook(true);
    setBins(256);
    setDist('uniform');
  };

  return (
    <Bench
      title={tr('Atomic throughput', 'Thông lượng atomic')}
      task={tr(
        "Start from the book's example. Then find what matters more: the number of bins, or the share of the busiest one?",
        'Bắt đầu từ ví dụ của sách. Rồi tìm xem điều gì quan trọng hơn: số ô, hay tỷ phần của ô bận nhất?',
      )}
      foot={
        <Translated
          vi={
            <>
              Hệ thống bộ nhớ là hệ thống giả định của sách (kênh 8 byte, double data rate, 8 kênh); tốc độ phần tử của nó tỷ lệ
              theo xung nhịp bạn chọn. Độ trễ tính cho mỗi lần truy cập, như nhau cho lần nạp và lần ghi.
            </>
          }
        >
          <>
            The memory system is the book's hypothetical one (8-byte channels, double data rate, 8 channels); its element
            rate scales with the clock you choose. Latencies are per access, the same for the load and the store.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={reset}>
          <Translated vi="Ví dụ của sách">Book's example</Translated>
        </button>
        <label className="field">
          <Translated vi="Nơi atomic được thực hiện">Where the atomic is performed</Translated>
          <span className="seg" role="group" aria-label={tr('Where the atomic is performed', 'Nơi atomic được thực hiện')}>
            {PLACES.map((x) => (
              <button key={x.id} aria-pressed={place === x.id} onClick={() => setPlace(x.id)}>{x.label}</button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi={<>Độ trễ mỗi lần truy cập = {lat[place]} chu kỳ ({P.words})</>}>
            Latency per access = {lat[place]} cycles ({P.words})
          </Translated>
          <input type="range" min={P.min} max={P.max} step={place === 'dram' ? 10 : 1} value={lat[place]} aria-label={tr('Latency per access', 'Độ trễ mỗi lần truy cập')}
            onChange={(e) => setLat({ ...lat, [place]: +e.target.value })} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Xung nhịp">Clock</Translated>
          <span className="seg" role="group" aria-label={tr('Clock', 'Xung nhịp')}>
            <button aria-pressed={book} onClick={() => setBook(true)}>
              <Translated vi="1 GHz của sách">the book's 1 GHz</Translated>
            </button>
            <button aria-pressed={!book} onClick={() => setBook(false)}>
              <Translated vi="tùy chỉnh">custom</Translated>
            </button>
          </span>
        </label>
        {!book ? (
          <label className="field">
            GHz
            <input type="number" min="0.1" max="5" step="0.1" value={ghz} onChange={(e) => setGhz(Math.max(0.1, +e.target.value || 0.1))} />
          </label>
        ) : null}
        <label className="field">
          <Translated vi="Số ô">Bins</Translated>
          <select value={bins} onChange={(e) => setBins(+e.target.value)} aria-label={tr('Bins', 'Số ô')}>
            {BIN_CHOICES.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Dữ liệu">Data</Translated>
          <span className="seg" role="group" aria-label={tr('Data', 'Dữ liệu')}>
            {DISTS.map((d) => (
              <button key={d.id} aria-pressed={dist === d.id} onClick={() => setDist(d.id)}>{d.label}</button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi={<>Tỷ phần của ô bận nhất = {fmt(100 * share)}%</>}>
            Busiest bin's share = {fmt(100 * share)}%
          </Translated>
          <input type="range" min="0" max="1" step="0.01" value={dist === 'custom' ? custom : share} disabled={dist !== 'custom'}
            aria-label={tr("Busiest bin's share", 'Tỷ phần của ô bận nhất')} onChange={(e) => setCustom(+e.target.value)} />
        </label>
      </div>

      <div className="cols">
        <div className="panel">
          <p style={{ margin: 0 }}>
            <Translated
              vi={
                <>
                  Một vị trí: {fmt(clockHz / 1e9)} GHz / ({lat[place]} + {lat[place]} chu kỳ) ={' '}
                  <b data-one={one}>{rate(one)}</b> atomic.
                </>
              }
            >
              One location: {fmt(clockHz / 1e9)} GHz / ({lat[place]} + {lat[place]} cycles) ={' '}
              <b data-one={one}>{rate(one)}</b> atomics.
            </Translated>
          </p>
          <p style={{ margin: '0.6rem 0 0' }}>
            <Translated vi={<>Cả histogram: {rate(one)} / {fmt(share, 3)} =</>}>
              The whole histogram: {rate(one)} / {fmt(share, 3)} =
            </Translated>{' '}
            <b style={{ fontSize: '1.3rem' }} data-bound={bound.rate}>{rate(bound.rate)}</b>
            {bound.capped ? (
              <>
                {' '}
                <span className="chip global">
                  <Translated vi="bị chặn bởi bộ nhớ">capped by memory</Translated>
                </span>
              </>
            ) : null}
          </p>
          <p className="muted" style={{ margin: '0.6rem 0 0' }}>
            <Translated
              vi={
                <>
                  Hệ thống bộ nhớ đưa tới {rate(cap).replace('/s', '')} phần tử bốn byte mỗi giây ({fmt(peakBandwidth(mem) / 1e9)} GB/s).{' '}
                </>
              }
            >
              The memory system delivers {rate(cap).replace('/s', '')} four-byte elements per second ({fmt(peakBandwidth(mem) / 1e9)} GB/s).{' '}
            </Translated>
            {bound.capped
              ? tr('The atomics keep up with it here.', 'Ở đây các atomic theo kịp nó.')
              : tr(`The atomics manage ${fmt(cap / bound.rate)} times fewer updates than that.`, `Các atomic chỉ làm được số lần cập nhật ít hơn thế ${fmt(cap / bound.rate)} lần.`)}
          </p>
          <p className="muted" style={{ margin: '0.4rem 0 0' }}>
            {dist === 'bright'
              ? tr(`Our sky picture from section 9.1, in ${bins} bins: the busiest holds ${fmt(100 * share)}% of the pixels.`, `Hình bầu trời của chúng tôi từ mục 9.1, chia ${bins} ô: ô bận nhất chứa ${fmt(100 * share)}% số điểm ảnh.`)
              : dist === 'uniform'
                ? tr(`Uniform data over ${bins} bins: each holds 1/${bins} of the updates.`, `Dữ liệu đều trên ${bins} ô: mỗi ô nhận 1/${bins} số lần cập nhật.`)
                : dist === 'constant'
                  ? tr('Every pixel the same: one bin takes every update.', 'Mọi điểm ảnh như nhau: một ô nhận mọi lần cập nhật.')
                  : tr('Your share for the busiest bin; the others share the rest.', 'Tỷ phần bạn đặt cho ô bận nhất; các ô khác chia phần còn lại.')}
          </p>
        </div>
        <div className="panel">
          <LogBars
            rows={[
              { label: tr('one location', 'một vị trí'), v: one, tone: 'var(--ink-3)' },
              { label: tr('whole histogram', 'cả histogram'), v: bound.rate, tone: 'var(--ink)' },
              { label: tr('memory, elements', 'bộ nhớ, phần tử'), v: cap, tone: 'var(--global)' },
            ]}
          />
        </div>
      </div>
    </Bench>
  );
}

function MeasuredSkew({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = measuredSkew(reports);
  if (!found.length)
    return (
      <Translated
        vi={
          <p data-part="skew-none">
            Chưa có số đo nào trên máy này. Hãy chạy <code>python3 labs/pmpp.py run 9.1 --solution --bench</code> rồi tải lại
            trang này: lệnh đó đo thời gian kernel cơ bản trên các byte phân bố đều và trên một ảnh hằng, và ô này sẽ so dự đoán
            của bạn với kết quả.
          </p>
        }
      >
        <p data-part="skew-none">
          No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 9.1 --solution --bench</code> and reload
          this page: it times the basic kernel on uniform bytes and on a constant image, and this box will compare your
          prediction with the result.
        </p>
      </Translated>
    );
  return (
    <>
      {found.map((f) => (
        <p key={f.b} data-part="skew-measured">
          <Translated
            vi={
              <>
                Trên <b>{f.device}</b> ({f.b}), kernel cơ bản chạy chậm hơn <b>{fmt(f.ratio)}×</b> trên ảnh hằng ({fmt(f.con)}{' '}
                so với {fmt(f.uni)} Gpixel/s).{' '}
              </>
            }
          >
            On <b>{f.device}</b> ({f.b}) the basic kernel ran <b>{fmt(f.ratio)}×</b> slower on the constant image ({fmt(f.con)}{' '}
            against {fmt(f.uni)} Gpixel/s).{' '}
          </Translated>
          {f.ratio < 1
            ? tr(
                'The constant image ran faster here, the opposite of the model: something merged the same-address updates or kept the busy bin close (see the note above), and which of these applies is not documented for every backend.',
                'Ở đây ảnh hằng lại chạy nhanh hơn, ngược với mô hình: có gì đó đã gộp các lần cập nhật cùng địa chỉ hoặc giữ ô bận ở gần (xem ghi chú ở trên), và điều nào trong số đó áp dụng thì không được ghi trong tài liệu cho mọi backend.',
              )
            : predicted > f.ratio * 2
            ? tr(
                'You expected considerably more. Something absorbed much of the contention the model charges for: caches, or the hardware or compiler merging same-address updates (see the note above). Which of these applies is not documented for every backend.',
                'Bạn đã dự đoán nhiều hơn đáng kể. Có gì đó đã hấp thụ phần lớn tranh chấp mà mô hình tính vào: bộ nhớ đệm, hoặc phần cứng hay trình biên dịch gộp các lần cập nhật cùng địa chỉ (xem ghi chú ở trên). Điều nào trong số đó áp dụng thì không được ghi trong tài liệu cho mọi backend.',
              )
            : predicted < f.ratio / 2
              ? tr(
                  'You expected considerably less: one busy location costs more on this device than you allowed for.',
                  'Bạn đã dự đoán ít hơn đáng kể: một vị trí bận tốn trên thiết bị này nhiều hơn bạn tính.',
                )
              : tr('That is close to what you predicted.', 'Kết quả đó gần với dự đoán của bạn.')}
        </p>
      ))}
      <Translated
        vi={<p className="muted">Đây là số đo từ máy của bạn, không phải giá trị kỳ vọng. Một thiết bị khác có thể chênh lệch nhiều theo cả hai hướng.</p>}
      >
        <p className="muted">These are measurements from your machine, not expected values. Another device can differ by a wide margin in either direction.</p>
      </Translated>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS93: Record<string, Question> = {
  'ch09/9.3/q-ex1': {
    kind: 'numeric',
    id: 'ch09/9.3/q-ex1',
    prompt: (
      <Translated vi={<>Mỗi phép toán nguyên tử trên một hệ thống DRAM mất tổng cộng 100 ns. Một biến toàn cục nhận được tối đa bao nhiêu atomic mỗi giây?</>}>
        <>Each atomic operation on a DRAM system takes 100 ns in total. What is the most atomics per second one global variable can take?</>
      </Translated>
    ),
    answer: 10,
    tol: 0.01,
    unit: 'M atomics/s',
    explain: (
      <Translated vi={<p>Lần lượt từng cái, mỗi cái 100 ns: 1 / 100 ns = 10 triệu mỗi giây, dù có bao nhiêu luồng đang chờ.</p>}>
        <p>One at a time, 100 ns each: 1 / 100 ns = 10 million per second, however many threads are waiting.</p>
      </Translated>
    ),
    source: 'ch. 9 exercise 1, p. 220',
    vi: { unit: 'M atomic/s', source: 'ch. 9 bài tập 1, p. 220' },
  },
  'ch09/9.3/q-ex2': {
    kind: 'numeric',
    id: 'ch09/9.3/q-ex2',
    prompt: (
      <>
        <Translated
          vi={
            <>
              GPU thực hiện atomic trong bộ nhớ đệm L2 của nó: 4 ns khi trúng, 100 ns khi giá trị phải lấy từ DRAM, và 90% số
              atomic trúng. Một biến toàn cục nhận được khoảng bao nhiêu atomic mỗi giây?
            </>
          }
        >
          The GPU performs atomics in its L2 cache: 4 ns on a hit, 100 ns when the value has to come from DRAM, and 90% of
          the atomics hit. Roughly how many atomics per second can one global variable take?
        </Translated>
      </>
    ),
    answer: 73.5,
    tol: 0.01,
    unit: 'M atomics/s',
    explain: (
      <Translated
        vi={
          <p>
            Thời gian trung bình là 0.9 × 4 + 0.1 × 100 = 3.6 + 10 = 13.6 ns, nên khoảng 73.5 triệu mỗi giây. 10% lần trượt
            tốn gần gấp ba tổng tất cả các lần trúng: trường hợp chậm chi phối một trung bình của các thời gian.
          </p>
        }
      >
        <p>
          The mean time is 0.9 × 4 + 0.1 × 100 = 3.6 + 10 = 13.6 ns, so about 73.5 million per second. The 10% of misses
          cost almost three times as much as all the hits together: the slow case dominates an average of times.
        </p>
      </Translated>
    ),
    source: 'ch. 9 exercise 2, p. 220',
    vi: { unit: 'M atomic/s', source: 'ch. 9 bài tập 2, p. 220' },
  },
  'ch09/9.3/q-ex3': {
    kind: 'numeric',
    id: 'ch09/9.3/q-ex3',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Với các atomic của bài tập 1 (mỗi cái 100 ns trên một biến), một kernel thực hiện 5 phép toán dấu phẩy động cho mỗi
              atomic. Tốc độ dấu phẩy động cao nhất mà các atomic cho phép là bao nhiêu?
            </>
          }
        >
          With exercise 1's atomics (100 ns each on one variable), a kernel performs 5 floating-point operations for every
          atomic. What is the highest floating-point rate the atomics allow?
        </Translated>
      </>
    ),
    answer: 50,
    tol: 0.01,
    unit: 'M FLOP/s',
    explain: (
      <Translated vi={<p>5 × 10 triệu = 50 triệu phép toán mỗi giây, bất kể các đơn vị số học làm được bao nhiêu. Atomic là nút thắt cổ chai, và phép tính phải chờ nó.</p>}>
        <p>5 × 10 million = 50 million operations per second, whatever the arithmetic units could do. The atomic is the bottleneck, and the arithmetic waits on it.</p>
      </Translated>
    ),
    source: 'ch. 9 exercise 3, p. 220',
    vi: { unit: 'M FLOP/s', source: 'ch. 9 bài tập 3, p. 220' },
  },
  'ch09/9.3/q-ex4': {
    kind: 'numeric',
    id: 'ch09/9.3/q-ex4',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Lại bài tập 3, nhưng biến được riêng tư hóa vào bộ nhớ chia sẻ, nơi một atomic mất tổng cộng 1 ns. Mọi atomic ban
              đầu trở thành atomic trên bộ nhớ chia sẻ; các lần ghi về biến toàn cục thêm vào cộng thêm 10% thời gian chạy. Vẫn 5
              phép toán dấu phẩy động mỗi atomic. Giờ tốc độ dấu phẩy động cao nhất là bao nhiêu?
            </>
          }
        >
          Exercise 3 again, but the variable is privatised into shared memory, where one atomic takes 1 ns in total. All
          the original atomics become shared-memory atomics; the extra commits to the global variable add 10% to the run
          time. Still 5 floating-point operations per atomic. What is the highest floating-point rate now?
        </Translated>
      </>
    ),
    answer: 4.55,
    tol: 0.02,
    unit: 'GFLOP/s',
    explain: (
      <Translated
        vi={
          <>
            <p>
              Giờ mỗi atomic tốn 1 ns × 1.1 = 1.1 ns, khoảng 909 triệu mỗi giây, và 5 × 909 triệu ≈ 4.5 GFLOP/s: gấp 90 lần
              bài tập 3.
            </p>
            <p>
              Cách diễn đạt này là của chúng tôi. Sách nói "shared memory access latency is 1 ns", điều đó cũng cho phép tính
              một lần đọc và một lần ghi, 2 ns mỗi atomic, như trong mô hình 200 + 200 chu kỳ ở trên. Cách hiểu đó cho 2.2 ns mỗi
              atomic và khoảng 2.3 GFLOP/s. Sách không nói nó muốn nói cách nào.
            </p>
          </>
        }
      >
        <>
          <p>
            Each atomic now costs 1 ns × 1.1 = 1.1 ns, about 909 million per second, and 5 × 909 million ≈ 4.5 GFLOP/s: 90
            times exercise 3.
          </p>
          <p>
            This wording is ours. The book says "shared memory access latency is 1 ns", which also allows counting a read
            and a write, 2 ns per atomic, as in the 200 + 200 cycle model above. That reading gives 2.2 ns per atomic and
            about 2.3 GFLOP/s. The book does not say which it means.
          </p>
        </>
      </Translated>
    ),
    source: 'ch. 9 exercise 4, p. 220; the ambiguity is noted in notes/ch09.md, "Unclear in the PDF"',
    vi: { source: 'ch. 9 bài tập 4, p. 220; chỗ mơ hồ được ghi trong notes/ch09.md, mục "Unclear in the PDF"' },
  },
  'ch09/9.3/q-half': {
    kind: 'numeric',
    id: 'ch09/9.3/q-half',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Mô hình DRAM của sách: 2.5 M atomic mỗi giây trên một vị trí. Một histogram 256 ô trong đó một nửa số điểm ảnh rơi
              vào một ô và phần còn lại trải đều trên 255 ô kia. Cả histogram xử lý được khoảng bao nhiêu atomic mỗi giây?
            </>
          }
        >
          The book's DRAM model: 2.5 M atomics per second on one location. A 256-bin histogram where half of the pixels
          fall in one bin and the rest are spread evenly over the other 255. Roughly how many atomics per second does the
          whole histogram get through?
        </Translated>
      </>
    ),
    answer: 5,
    tol: 0.02,
    unit: 'M atomics/s',
    explain: (
      <Translated
        vi={
          <p>
            Ô bận nhận 0.5<i>N</i> trong <i>N</i> lần cập nhật và xử lý chúng ở 2.5 M/s, nên nó cần 0.5<i>N</i> / 2.5 M giây.
            Mỗi ô khác nhận khoảng 0.002<i>N</i> và xong từ lâu trước đó. <i>N</i> lần cập nhật trong 0.2<i>N</i>{' '}
            micro giây là 5 M/s: chậm hơn 128 lần so với 640 M/s của dữ liệu đều, chỉ vì một ô.
          </p>
        }
      >
        <p>
          The busy bin receives 0.5<i>N</i> of the <i>N</i> updates and takes them at 2.5 M/s, so it needs 0.5<i>N</i> / 2.5
          M seconds. Each other bin receives about 0.002<i>N</i> and finishes long before. <i>N</i> updates in 0.2<i>N</i>{' '}
          microseconds is 5 M/s: 128 times slower than the uniform 640 M/s, because of one bin.
        </p>
      </Translated>
    ),
    source: 'our arithmetic on the §9.3 model, pp. 209–210',
    vi: { unit: 'M atomic/s', source: 'phép tính của chúng tôi trên mô hình của §9.3, pp. 209–210' },
  },
  'ch09/9.3/q-more-threads': {
    kind: 'choice',
    id: 'ch09/9.3/q-more-threads',
    prompt: (
      <Translated vi={<>Khởi chạy thêm luồng đã làm một kernel sao chép bị giới hạn bởi bộ nhớ chạy nhanh hơn. Liệu nó có tăng tốc kernel histogram cơ bản trên một ảnh toàn trắng không?</>}>
        <>Launching more threads made a memory-bound copy kernel faster. Will it speed up the basic histogram kernel on an all-white image?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Có: nhiều luồng hơn che được nhiều độ trễ hơn, đó chính là điều đã tăng tốc kernel sao chép.">Yes: more threads hide more latency, which is what sped up the copy kernel.</Translated>, why: <Translated vi="Che độ trễ cần các yêu cầu độc lập có thể chồng lên nhau. Atomic trên một vị trí thì không thể.">Latency hiding needs independent requests that can overlap. Atomics on one location cannot.</Translated> },
      { label: <Translated vi="Không: mỗi lúc chỉ một atomic trên ô đó, nên thêm luồng chỉ làm hàng đợi dài hơn.">No: one atomic at a time on that bin, so more threads only make the queue longer.</Translated>, correct: true },
      { label: <Translated vi="Chỉ khi các luồng thêm vào nằm ở các khối khác, để atomic của chúng bắt đầu trên các SM khác.">Only if the extra threads are in different blocks, so their atomics start on other SMs.</Translated>, why: <Translated vi="Atomic phạm vi thiết bị trên một vị trí đều phải xếp hàng, dù đến từ khối hay SM nào.">Device-scope atomics on one location queue whichever block or SM they come from.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Kernel sao chép nhanh hơn vì các yêu cầu của nó độc lập: càng nhiều yêu cầu đang bay thì càng chồng lên nhau nhiều.
            Mỗi lần cập nhật vào ô trắng phải chờ lần trước nó, nên tốc độ của ô bị độ trễ cố định, và thêm luồng chỉ thêm chờ
            đợi. Các cách chữa thay đổi đích đến của dữ liệu (bản sao riêng) hoặc độ trễ (bộ nhớ đệm, bộ nhớ chia sẻ), chứ không
            phải số luồng.
          </p>
        }
      >
        <p>
          The copy kernel got faster because its requests were independent: more of them in flight meant more overlap.
          Every update to the white bin waits for the one before it, so the bin's rate is fixed by the latency, and adding
          threads adds waiting. The cures change the data's destination (private copies) or the latency (caches, shared
          memory), not the thread count.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.3, p. 209',
  },
};

export function S93() {
  const q = questionsS93;
  const book = useMemo(() => {
    const one = perLocationRate(1e9, 200);
    return { one, uni: histogramBound(one, 1 / 256).rate };
  }, []);
  return (
    <>
      <Translated
        vi={
          <p>
            Bất kỳ phần nào của một chương trình song song quy mô lớn mà chạy từng bước một đều có thể chi phối thời gian chạy
            của nó. Atomic trên một vị trí theo thiết kế chạy từng cái một, nên câu hỏi là chúng tốn bao nhiêu.
          </p>
        }
      >
        <p>
          Any part of a massively parallel program that runs one step at a time can dominate its run time. Atomics on one
          location run one at a time by design, so the question is how much they cost.
        </p>
      </Translated>
      <Source section="9.3" pages="209" />
      <Translated
        vi={
          <p>
            Hãy nhớ lại cách GPU đối phó với bộ nhớ chậm. Một lần truy cập DRAM mất hàng trăm chu kỳ xung nhịp (
            <a href={href('ch05', '5.1')}>chương 5</a>). GPU che điều đó bằng cách giữ nhiều lần truy cập cùng đang bay: trong
            lúc một warp chờ, warp khác chạy (<a href={href('ch04', '4.6')}>chương 4</a>), và các kênh và bank của DRAM phục vụ
            nhiều yêu cầu cùng lúc (<a href={href('ch06', '6.2')}>chương 6</a>). Với đủ yêu cầu độc lập, tốc độ được quyết định
            bởi số byte mỗi giây bộ nhớ có thể đưa tới, chứ không phải bởi một lần truy cập kéo dài bao lâu.
          </p>
        }
      >
        <p>
          Recall how a GPU copes with slow memory. A DRAM access takes hundreds of clock cycles (
          <a href={href('ch05', '5.1')}>chapter 5</a>). The GPU hides that by keeping many accesses in flight: while one
          warp waits, another runs (<a href={href('ch04', '4.6')}>chapter 4</a>), and the DRAM's channels and banks serve
          many requests at once (<a href={href('ch06', '6.2')}>chapter 6</a>). With enough independent requests, the speed
          is set by how many bytes per second the memory can deliver, not by how long one access takes.
        </p>
      </Translated>
      <Source section="9.3" pages="209" />
      <Translated
        vi={
          <p>
            Atomic trên một vị trí phá vỡ chiến lược đó. Mỗi lúc chỉ một cái có thể đang diễn ra, và mỗi cái kéo dài khoảng một
            độ trễ nạp (để đọc giá trị cũ) cộng một độ trễ ghi (để ghi giá trị mới). Không có gì chồng lên nhau, nên vị trí đó
            nhận tối đa một atomic cho mỗi độ trễ nạp-cộng-ghi: khoảng 1 / (2 × độ trễ).
          </p>
        }
      >
        <p>
          Atomics on one location break that strategy. Only one can be in progress at a time, and each lasts about a load
          latency (to read the old value) plus a store latency (to write the new one). Nothing overlaps, so the location
          takes at most one atomic per load-plus-store latency: roughly 1 / (2 × latency).
        </p>
      </Translated>
      <Source section="9.3" pages="209" figs={tr('Fig. 9.7', 'Hình 9.7')} />

      <h2>
        <Translated vi="Vì sao một ô bận quyết định nhịp độ">Why one busy bin sets the pace</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hãy xem nó theo thời gian. Mỗi thanh bên dưới là một yêu cầu; điểm bắt đầu và kết thúc của nó được tính bằng cùng
            mô hình với trình khám phá ở phía dưới, với bộ nhớ 200 chu kỳ của sách.
          </p>
        }
      >
        <p>
          Watch it as time. Each bar below is one request; its start and end are computed by the same model as the
          explorer further down, with the book's 200-cycle memory.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi vị trí mỗi lúc một atomic, các vị trí khác nhau song song hoàn toàn, không có độ trễ truyền giữa lần nạp và lần
              ghi (phép tính của sách cũng bỏ qua nó), không có bộ nhớ đệm và trình biên dịch không gộp atomic. Tám luồng đại diện
              cho một warp; một warp thật có 32 lane trên NVIDIA và 32 hoặc 64 trên AMD. Các số liệu về bộ nhớ đệm, bộ nhớ chia sẻ
              và khe phát lệnh chỉ để minh họa: sách chỉ nói "hàng chục" và "vài" chu kỳ.
            </p>
          }
        >
          <p>
            One atomic per location at a time, different locations fully in parallel, no transfer delay between the load
            and the store (the book's arithmetic leaves it out too), no caches and no merging of atomics by the compiler.
            Eight threads stand in for a warp; a real warp has 32 lanes on NVIDIA and 32 or 64 on AMD.
            The cache, shared-memory and issue-slot figures are illustrative: the book gives only "tens" and "a few" cycles.
          </p>
        </Translated>
      </Simplification>
      <AtomicWaitScene />

      <h2>
        <Translated vi="Các con số của sách">The book's numbers</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Ví dụ của sách là một hệ thống bộ nhớ giả định: 8 kênh, mỗi kênh rộng 8 byte, double data rate, ở 1 GHz, với độ trễ
            truy cập 200 chu kỳ. Đỉnh của nó là 8 byte × 2 lần truyền × 1 G xung nhịp × 8 kênh = 128 GB/s, tức 32 G phần tử bốn
            byte mỗi giây. Một vị trí nhận một atomic mỗi 200 + 200 chu kỳ: 1 GHz / 400 ={' '}
            {rate(book.one)}. Với 256 ô được nhắm đều tuyệt đối, các ô làm việc song song và histogram xử lý được 256 × 2.5 M ={' '}
            {rate(book.uni)}, vẫn thấp hơn năm mươi lần so với những gì bộ nhớ có thể đưa tới. Và ảnh thật không phân bố đều:
            một bức ảnh sáng dồn phần lớn điểm ảnh vào vài ô, và mức lợi giảm xa dưới số ô.
          </p>
        }
      >
        <p>
          The book's example is a hypothetical memory system: 8 channels, each 8 bytes wide, double data rate, at 1 GHz,
          with an access latency of 200 cycles. Its peak is 8 bytes × 2 transfers × 1 G clocks × 8 channels = 128 GB/s,
          or 32 G four-byte elements per second. One location takes an atomic every 200 + 200 cycles: 1 GHz / 400 ={' '}
          {rate(book.one)}. With 256 bins hit perfectly evenly, the bins work in parallel and the histogram gets through
          256 × 2.5 M = {rate(book.uni)}, still fifty times below what the memory could deliver. And real images are not
          uniform: a bright picture puts most of its pixels in a few bins, and the gain falls far below the number of bins.
        </p>
      </Translated>
      <Source section="9.3" pages="209–210" />
      <Translated
        vi={
          <p>
            Trình khám phá tái tạo các con số này và cho bạn đổi từng giả định: nơi atomic được thực hiện, độ trễ của nó, xung
            nhịp, số ô và cách dữ liệu rơi vào chúng.
          </p>
        }
      >
        <p>
          The explorer reproduces these numbers and lets you change each assumption: where the atomic is performed, its
          latency, the clock, the number of bins and how the data falls into them.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Cùng mô hình với hoạt ảnh, dưới dạng công thức: một vị trí chạy ở xung nhịp / (độ trễ nạp + ghi); cả histogram chạy
              ở tốc độ đó chia cho tỷ phần của ô bận nhất, nhưng không bao giờ nhanh hơn tốc độ bộ nhớ đưa phần tử tới. Không có
              bộ nhớ đệm, trình biên dịch không gộp.
            </p>
          }
        >
          <p>
            The same model as the animation, as a formula: one location runs at clock / (load + store latency); the whole
            histogram at that rate divided by the busiest bin's share, but never faster than the memory delivers elements.
            No caches, no compiler aggregation.
          </p>
        </Translated>
      </Simplification>
      <ThroughputExplorer />
      <Ask q={q['ch09/9.3/q-half']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={q['ch09/9.3/q-more-threads']} />

      <h2>
        <Translated vi="Cách chữa 1: rút ngắn độ trễ">Remedy 1: shorten the latency</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Nếu tốc độ là một atomic cho mỗi độ trễ, cách chữa đầu tiên là độ trễ ngắn hơn. GPU hiện đại thực hiện atomic phạm
            vi thiết bị trong bộ nhớ đệm cấp cuối, nơi mọi SM dùng chung. Một giá trị đang có trong bộ nhớ đệm được cập nhật ở
            đó; giá trị chưa có thì được đưa vào trước. Một ô được cập nhật nhiều có xu hướng nằm lại trong bộ nhớ đệm.
          </p>
        }
      >
        <p>
          If the rate is one atomic per latency, the first remedy is a shorter latency. Modern GPUs perform device-scope
          atomics in the last-level cache, which all SMs share. A value that is in the cache is updated there; one that is
          not is brought in first. A heavily updated bin tends to stay in the cache.
        </p>
      </Translated>
      <Source section="9.3" pages="210" />
      <Translated
        vi={
          <p>
            Sách đặt một lần truy cập bộ nhớ đệm cấp cuối ở mức "tens of cycles rather than hundreds" (hàng chục chu kỳ thay vì
            hàng trăm), điều đó cải thiện thông lượng atomic "by at least an order of magnitude" (ít nhất một bậc độ lớn). Trong
            trình khám phá, hãy chuyển sang bộ nhớ đệm cấp cuối: tốc độ theo vị trí và tốc độ cả histogram thay đổi cùng một hệ
            số, cho đến khi histogram chạm mức bộ nhớ có thể đưa tới.
          </p>
        }
      >
        <p>
          The book puts a last-level cache access at "tens of cycles rather than hundreds", which improves atomic
          throughput "by at least an order of magnitude". In the explorer, switch to the last-level cache: the
          per-location and whole-histogram rates move by the same factor, until the histogram reaches what the memory can
          deliver.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch09-facts.md, A4, B2, C4)', '2026-11-20 (notes/today/ch09-facts.md, các mục A4, B2, C4)')}>
        <Translated
          vi={
            <p>
              Hướng dẫn Nsight Compute của NVIDIA: atomic toàn cục "are serialized at the L2 atomic stage"; chỉ số{' '}
              <code>lts__t_sectors_srcunit_tex_op_atom</code> cho thấy khối lượng. AMD: "Atomics execute directly in the L2
              cache". Apple không công bố atomic chạy ở đâu lẫn độ trễ của chúng. Không nhà cung cấp nào đưa ra độ trễ L2 hay DRAM
              tính bằng chu kỳ, nên 200 và "hàng chục" vẫn là ví dụ của sách.
            </p>
          }
        >
          <p>
            NVIDIA's Nsight Compute guide: global atomics "are serialized at the L2 atomic stage"; the metric{' '}
            <code>lts__t_sectors_srcunit_tex_op_atom</code> shows the volume. AMD: "Atomics execute directly in the L2
            cache". Apple documents neither where atomics run nor their latency. No vendor gives L2 or DRAM latency in
            cycles, so 200 and "tens" stay the book's example.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Bài tập 1 đến 4">Exercises 1 to 4</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bốn bài tập đầu của sách chính là phép tính của mục này. Bài tập 4 được đưa ra theo cách diễn đạt của chúng tôi; lời
            giải thích của nó nói vì sao.
          </p>
        }
      >
        <p>
          The book's first four exercises are this section's arithmetic. Exercise 4 is given in our wording; its
          explanation says why.
        </p>
      </Translated>
      <Ask q={q['ch09/9.3/q-ex1']} label={tr('Exercise 1', 'Bài tập 1')} />
      <Ask q={q['ch09/9.3/q-ex2']} label={tr('Exercise 2', 'Bài tập 2')} />
      <Ask q={q['ch09/9.3/q-ex3']} label={tr('Exercise 3', 'Bài tập 3')} />
      <Ask q={q['ch09/9.3/q-ex4']} label={tr('Exercise 4', 'Bài tập 4')} />

      <h2>
        <Translated vi="Dự đoán, rồi đo">Predict, then measure</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trong mô hình của sách, một ảnh có mọi điểm ảnh cùng một giá trị làm kernel cơ bản chậm hơn{' '}
            {fmt(skewSlowdown(book.one, 256, 32e9))} lần so với các byte đều trên 256 ô: mọi lần cập nhật xếp vào một hàng đợi
            thay vì 256. Phần cứng thật có thể không đồng ý, theo cả hai hướng. Lab 9.1 đo thời gian cả hai đầu vào.
          </p>
        }
      >
        <p>
          In the book's model, an image whose pixels all have one value makes the basic kernel {fmt(skewSlowdown(book.one, 256, 32e9))}{' '}
          times slower than uniform bytes over 256 bins: every update joins one queue instead of 256. Real hardware may
          disagree, in either direction. Lab 9.1 times both inputs.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch09-facts.md, A9, B5)', '2026-11-20 (notes/today/ch09-facts.md, các mục A9, B5)')}>
        <Translated
          vi={
            <p>
              Một bài blog của NVIDIA, cập nhật lần cuối năm 2017, nói nvcc gộp các atomic cùng địa chỉ trong một warp "in many
              cases" (trong nhiều trường hợp), mà không nói là trường hợp nào. Backend AMDGPU của LLVM chỉ gộp chúng khi địa chỉ
              giống nhau trên cả wavefront ngay lúc biên dịch, điều mà địa chỉ phụ thuộc dữ liệu của histogram không thỏa. Mức
              phạt đo được có thể thấp hơn nhiều so với của mô hình.
            </p>
          }
        >
          <p>
            An NVIDIA blog post, last updated in 2017, says nvcc merges same-address atomics within a warp "in many cases",
            without saying which. LLVM's AMDGPU backend merges them only when the address is the same across the wavefront
            at compile time, which a histogram's data-dependent address is not. A measured penalty may be far below the
            model's.
          </p>
        </Translated>
      </Today>
      <Predict
        id="ch09/9.3/p-skew-slowdown"
        prompt={
          <Translated vi={<>Trên GPU <em>của bạn</em>, kernel cơ bản sẽ chạy chậm hơn bao nhiêu lần trên một ảnh hằng so với trên các byte ngẫu nhiên phân bố đều? Hãy chốt trước khi chạy lab 9.1.</>}>
            <>On <em>your</em> GPU, how many times slower will the basic kernel run on a constant image than on uniform random bytes? Commit before you run lab 9.1.</>
          </Translated>
        }
        unit={tr('× slower', '× chậm hơn')}
      >
        {(v) => <MeasuredSkew predicted={v} />}
      </Predict>
      <Translated
        vi={
          <p>
            Còn một chi tiết nữa quyết định phần cứng chạy lệnh nào. Kernel cơ bản không bao giờ dùng giá trị mà{' '}
            <code>fetch_add</code> trả về.
          </p>
        }
      >
        <p>
          One more detail decides which instruction the hardware runs. The basic kernel never uses the value{' '}
          <code>fetch_add</code> returns.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              PTX có hai lệnh cho việc này: <code>atom</code>, trả về giá trị cũ, và <code>red</code>, không trả về. Một kết quả{' '}
              <code>fetch_add</code> không được dùng cho phép trình biên dịch dùng <code>red</code>; PyTorch đặt tên hàm trợ giúp
              của nó là <code>gpuAtomicAddNoReturn</code> vì lý do đó. nvcc sinh ra lệnh nào thì không có tài liệu: hãy đọc PTX
              (notes/today/ch09-facts.md A9, D1).
            </p>
          }
        >
          <p>
            PTX has two instructions for this: <code>atom</code>, which returns the old value, and <code>red</code>, which
            does not. An unused <code>fetch_add</code> result lets the compiler use <code>red</code>; PyTorch calls its
            helper <code>gpuAtomicAddNoReturn</code> for that reason. Which one nvcc emits is undocumented: read the PTX
            (notes/today/ch09-facts.md A9, D1).
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Cùng cái bẫy đó trong mã suy luận">The same trap in inference code</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một bộ tích lũy duy nhất trong bộ nhớ toàn cục mà mọi luồng cập nhật là trường hợp tệ nhất của mục này: một vị trí,
            mọi luồng trong hàng đợi của nó. Cộng mẫu số của softmax theo cách đó, mỗi luồng một phép cộng nguyên tử, đúng là như
            vậy. Phép rút gọn nhiều khối của chương 10 tránh điều đó: mỗi khối cộng đoạn của riêng nó một cách riêng tư và luồng
            dẫn đầu của nó ghi về một lần, nên hàng đợi trên tổng toàn cục chỉ có mỗi khối một mục (Fig. 10.18, pp. 244–245). Đó
            là hình dạng của mục tiếp theo.
          </p>
        }
      >
        <p>
          A single accumulator in global memory that every thread updates is the worst case of this section: one location,
          every thread in its queue. Summing the denominator of a softmax that way, one atomic add per thread, is exactly
          that. Chapter 10's multi-block reduction avoids it: each block sums its own segment privately and its leader
          commits once, so the queue on the global total has one entry per block (Fig. 10.18, pp. 244–245). That is the
          shape of the next section.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Ví dụ softmax là của chúng tôi; chương 10 xây kernel đó.</p>}>
          <p>The softmax example is ours; chapter 10 builds the kernel.</p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Lab 9.1 nằm ở <a href={href('ch09', 'labs')}>trang lab</a>; <a href={href('ch09', '9.4')}>mục 9.4</a> đi theo cách
            chữa thứ hai, ít luồng hơn trên mỗi vị trí.
          </p>
        }
      >
        <p>
          Lab 9.1 is on the <a href={href('ch09', 'labs')}>labs page</a>; <a href={href('ch09', '9.4')}>section 9.4</a> takes
          the second remedy, fewer threads per location.
        </p>
      </Translated>
    </>
  );
}
