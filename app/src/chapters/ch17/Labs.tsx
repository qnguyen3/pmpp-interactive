import type { Question } from '../../components/Ask';
import { Beyond, Source, Today, type Tier } from '../../components/notes';
import {
  BackendsAndEmulator,
  LabFiles,
  LabPrediction,
  LabResults,
  LabsIntro,
  LabsPage,
  ReferenceSolution,
  RunningLabs,
  VerificationStatus,
  defineLabs,
  measure,
  msCol,
  numCol,
  preferredBench,
  rawCol,
  useLabs,
  whose,
  type BenchSet,
  type Col,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 17's own content.
// No question on this page: the lab is the exercise. The two predictions are stored, not
// graded, under ch17/labs/p-lab1-uniform and ch17/labs/p-lab1-skewed.
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch17', [{ num: '17.1', dir: 'lab1_spmv', name: 'CSR and ELL SpMV', vi: { name: 'SpMV với CSR và ELL' } }]);
const [l1] = LABS;

/** Identifier colours in the reference solution: the matrix arrays and vectors live in global memory. */
const TIERS: Record<string, Tier> = {
  rowPtrs: 'global', colIdx: 'global', values: 'global', nnzPerRow: 'global', x: 'global', y: 'global',
  sum: 'reg', row: 'reg', col: 'reg', value: 'reg', i: 'reg', t: 'reg',
};

/** The benchmark's four matrices, in the driver's order. */
export const MATRICES = ['uniform', 'spread', 'spread_sorted', 'skewed'] as const;

// ---------------------------------------------------------------- reading the benchmark rows
/** The row for one matrix and one format, from one backend's preferred set. */
export const spmvRow = (s: BenchSet | undefined, matrix: string, format: string): BenchRow | undefined =>
  s?.rows.find((r) => r.name === 'spmv' && r.matrix === matrix && r.format === format);
const secs = (r: BenchRow | undefined) => num(r?.seconds);
const ms = (r: BenchRow | undefined) => {
  const v = secs(r);
  return v ? `${fmt(v * 1000)} ms` : '?'; // i18n-ignore: a number and its unit
};
const rowsOf = (r: BenchRow | undefined) => (num(r?.rows) ? tr(`${num(r?.rows)!.toLocaleString('en-US')} rows`, `${num(r?.rows)!.toLocaleString('en-US')} hàng`) : '');

/** Prediction 1: on 'uniform', CSR time over ELL time. Above 1, ELL was faster. */
const uniformRatio = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const csr = spmvRow(s, 'uniform', 'csr');
    const ell = spmvRow(s, 'uniform', 'ell');
    const a = secs(csr);
    const b = secs(ell);
    if (!a || !b) return null;
    return { value: a / b, detail: tr(`${ms(csr)} CSR against ${ms(ell)} ELL, ${rowsOf(csr)} of 16 entries`, `${ms(csr)} CSR so với ${ms(ell)} ELL, ${rowsOf(csr)} mỗi hàng 16 mục`) };
  });

/** Prediction 2: on 'skewed', ELL time over CSR time, with the hybrid's time beside it. */
const skewedRatio = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const csr = spmvRow(s, 'skewed', 'csr');
    const ell = spmvRow(s, 'skewed', 'ell');
    const hyb = spmvRow(s, 'skewed', 'hybrid');
    const a = secs(ell);
    const b = secs(csr);
    if (!a || !b) return null;
    const h = secs(hyb);
    const hybrid = h
      ? tr(`; the hybrid at K = 16 took ${ms(hyb)}, so CSR time / hybrid time is ${fmt(b / h)}`, `; dạng lai ở K = 16 mất ${ms(hyb)}, nên thời gian CSR / thời gian dạng lai là ${fmt(b / h)}`)
      : tr('; no hybrid row in this run', '; lần chạy này không có hàng nào cho dạng lai');
    return { value: a / b, detail: tr(`${ms(ell)} ELL at width ${String(ell?.width)} against ${ms(csr)} CSR${hybrid}`, `${ms(ell)} ELL ở độ rộng ${String(ell?.width)} so với ${ms(csr)} CSR${hybrid}`) };
  });

const MISSING = (
  <>
    <Translated
      vi={
        <>
          Phần này cần các hàng benchmark của lab 17.1 từ một thiết bị thật. Chạy <code>python3 labs/pmpp.py run 17.1 --bench</code>{' '}
          khi các kernel của bạn đã đạt (hoặc với <code>--solution</code> để đo thời gian bản tham chiếu). Trình giả lập không bao giờ được đo hiệu năng.
        </>
      }
    >
      This needs the benchmark rows of lab 17.1 from a real device. Run <code>python3 labs/pmpp.py run 17.1 --bench</code>{' '}
      once your kernels pass (or with <code>--solution</code> to time the reference). The emulator is never benchmarked.
    </Translated>
  </>
);

// ---------------------------------------------------------------- results table
const pct = (v: number | undefined) => (v == null ? '' : `${fmt(v * 100, 3)}%`);
const COLS: Col[] = [
  {
    get head() {
      return tr('Matrix', 'Ma trận');
    },
    cell: (r) => String(r.matrix ?? ''),
  },
  {
    get head() {
      return tr('Format', 'Định dạng');
    },
    cell: (r) => String(r.format ?? ''),
  },
  rawCol('Width', 'width'), // i18n-ignore: Vietnamese head supplied where LabResults renders
  rawCol('Stride', 'stride'), // i18n-ignore: Vietnamese head supplied where LabResults renders
  {
    get head() {
      return tr('Pad', 'Phần đệm');
    },
    num: true,
    cell: (r) => (r.format === 'csr' ? '' : pct(num(r.pad_fraction))),
  },
  numCol('COO entries', 'coo_entries'), // i18n-ignore: Vietnamese head supplied where LabResults renders
  {
    get head() {
      return tr('Matrix MB', 'MB ma trận');
    },
    num: true,
    cell: (r) => (num(r.matrix_bytes) ? fmt(num(r.matrix_bytes)! / 1e6) : ''),
  },
  msCol,
  numCol('GB/s, useful bytes', 'gbps'), // i18n-ignore: Vietnamese head supplied where LabResults renders
];

/** CSR time over each other format's time, per matrix: the driver's own summary line, from the reports. */
function Ratios() {
  const { reports, loading } = useLabs();
  if (loading) return null;
  const sets = preferredBench(reports, l1);
  const ratio = (s: BenchSet, m: string, f: string) => {
    const a = secs(spmvRow(s, m, 'csr'));
    const b = secs(spmvRow(s, m, f));
    return a && b ? fmt(a / b) : '';
  };
  const lines = sets.flatMap((s) => {
    const spread = spmvRow(s, 'spread', 'csr');
    const sorted = spmvRow(s, 'spread_sorted', 'csr');
    return secs(spread) && secs(sorted) ? [{ s, spread, sorted }] : [];
  });
  if (!sets.some((s) => MATRICES.some((m) => spmvRow(s, m, 'csr'))))
    return (
      <Translated
        vi={
          <p className="muted" data-ratios="none">
            Hai câu hỏi bạn không ghi lại sẽ được trả lời ở đây khi lab 17.1 có các hàng benchmark trên một thiết bị thật:{' '}
            <code>python3 labs/pmpp.py run 17.1 --bench</code>.
          </p>
        }
      >
        <p className="muted" data-ratios="none">
          The two questions you did not record are answered here once lab 17.1 has benchmark rows on a real device:{' '}
          <code>python3 labs/pmpp.py run 17.1 --bench</code>.
        </p>
      </Translated>
    );
  return (
    <>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-ratios="">
          <thead>
            <tr>
              <th><Translated vi="Backend">Backend</Translated></th>
              <th><Translated vi="Ma trận">Matrix</Translated></th>
              <th className="num"><Translated vi="Thời gian CSR">CSR time</Translated></th>
              <th className="num"><Translated vi="CSR / ell">CSR / ell</Translated></th>
              <th className="num"><Translated vi="CSR / ell_aligned">CSR / ell_aligned</Translated></th>
              <th className="num"><Translated vi="CSR / hybrid">CSR / hybrid</Translated></th>
            </tr>
          </thead>
          <tbody>
            {sets.flatMap((s) =>
              MATRICES.filter((m) => spmvRow(s, m, 'csr')).map((m) => (
                <tr key={s.backend + m} data-backend={s.backend} data-matrix={m}>
                  <td>
                    {s.backend} <span className="muted">({s.device}, {s.variant === 'work' ? tr('yours', 'của bạn') : tr('reference solution', 'lời giải tham chiếu')})</span>
                  </td>
                  <td>{m}</td>
                  <td className="num">{ms(spmvRow(s, m, 'csr'))}</td>
                  <td className="num" data-col="ell">{ratio(s, m, 'ell')}</td>
                  <td className="num" data-col="ell_aligned">{ratio(s, m, 'ell_aligned')}</td>
                  <td className="num" data-col="hybrid">{ratio(s, m, 'hybrid')}</td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
      {lines.map(({ s, spread, sorted }) => (
        <Translated
          vi={
            <p key={s.backend} data-sorted={s.backend} data-sorted-ratio={fmt(secs(spread)! / secs(sorted)!)}>
              Trên {s.device} ({s.backend}), với {whose(s)}, kernel CSR không đổi mất <b>{ms(sorted)}</b> trên{' '}
              <code>spread_sorted</code> so với <b>{ms(spread)}</b> trên <code>spread</code>: thời gian CSR trên spread chia cho thời gian CSR
              trên spread_sorted là {fmt(secs(spread)! / secs(sorted)!)}. Trên 1 nghĩa là việc sắp xếp các hàng làm CSR nhanh hơn. Dưới 1
              khớp với cái giá của việc sắp xếp: các luồng kề nhau giờ đọc <code>x</code> ở những chỗ cách xa nhau.
            </p>
          }
        >
          <p key={s.backend} data-sorted={s.backend} data-sorted-ratio={fmt(secs(spread)! / secs(sorted)!)}>
            On {s.device} ({s.backend}), with {whose(s)}, the unchanged CSR kernel took <b>{ms(sorted)}</b> on{' '}
            <code>spread_sorted</code> against <b>{ms(spread)}</b> on <code>spread</code>: CSR time on spread over CSR time
            on spread_sorted is {fmt(secs(spread)! / secs(sorted)!)}. Above 1, sorting the rows made CSR faster. Below 1
            is consistent with the cost of the sort: neighbouring threads now read <code>x</code> far apart.
          </p>
        </Translated>
      ))}
    </>
  );
}

/** One line on lab 17.1 on the race backend, which the status table does not show. */
function RaceStatus() {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const line = (variant: 'solution' | 'work') => {
    const rs = (run?.results.filter((x) => x.lab === l1.id && x.variant === variant) ?? []) as (RunResult & { tsan?: unknown[] })[];
    const tests = rs.flatMap((x) => x.tests);
    const found = rs.reduce((n, x) => n + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
    return { s: labStatus(run, l1.id, variant), tests, passed: tests.filter((t) => t.pass).length, found };
  };
  const sol = line('solution');
  const mine = line('work');
  return (
    <p data-race-status={loading ? 'loading' : sol.s} data-race-work={loading ? 'loading' : mine.s}>
      {loading ? (
        tr('Reading reports.', 'Đang đọc báo cáo.')
      ) : !run ? (
        tr('No run on the race backend on this machine yet.', 'Máy này chưa có lần chạy nào trên backend race.')
      ) : (
        <>
          {sol.s === 'none'
            ? tr('On this machine the reference solution has no run on the race backend yet.', 'Trên máy này, lời giải tham chiếu chưa có lần chạy nào trên backend race.')
            : sol.s === 'pass'
              ? tr(
                  `On this machine the reference solution passed all ${sol.tests.length} tests on the race backend with ${sol.found ? `${sol.found} ThreadSanitizer report(s)` : 'no race reported'}: emulator-checked on this flavour.`,
                  `Trên máy này, lời giải tham chiếu đạt cả ${sol.tests.length} bài kiểm thử trên backend race với ${sol.found ? `${sol.found} báo cáo ThreadSanitizer` : 'không báo cáo race nào'}: emulator-checked trên biến thể này.`,
                )
              : tr(
                  `On this machine the reference solution failed on the race backend (${sol.passed} of ${sol.tests.length} tests): unverified there.`,
                  `Trên máy này, lời giải tham chiếu thất bại trên backend race (${sol.passed} trên ${sol.tests.length} bài kiểm thử): unverified ở đó.`,
                )}{' '}
          {mine.s === 'none'
            ? tr('Your own kernels have no recorded run there, or are still the untouched starter.', 'Kernel của bạn chưa có lần chạy nào được ghi lại ở đó, hoặc vẫn là mã khởi đầu (starter) chưa động đến.')
            : mine.s === 'pass'
              ? tr(
                  `Your own kernels passed all ${mine.tests.length} tests there with ${mine.found ? `${mine.found} ThreadSanitizer report(s): read them` : 'no race reported'}.`,
                  `Kernel của bạn đạt cả ${mine.tests.length} bài kiểm thử ở đó với ${mine.found ? `${mine.found} báo cáo ThreadSanitizer: hãy đọc chúng` : 'không báo cáo race nào'}.`,
                )
              : tr(
                  `Your own kernels passed ${mine.passed} of ${mine.tests.length} tests there, with ${mine.found} ThreadSanitizer report(s) in the run.`,
                  `Kernel của bạn đạt ${mine.passed} trên ${mine.tests.length} bài kiểm thử ở đó, với ${mine.found} báo cáo ThreadSanitizer trong lần chạy.`,
                )}
        </>
      )}
    </p>
  );
}

/** What each backend calls the pieces of the chapter's kernels (notes/today/ch17-facts.md, "Where the backends really differ"). */
function BackendsTable() {
  return (
    <table className="data wide" style={{ maxWidth: '58rem' }}>
      <thead>
        <tr><th><Translated vi="Với các kernel này">For these kernels</Translated></th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><Translated vi="Một luồng mỗi hàng (CSR, ELL)">One thread per row (CSR, ELL)</Translated></td>
          <td><Translated vi={<><code>blockIdx.x*blockDim.x + threadIdx.x</code> và một phép kiểm tra biên</>}><code>blockIdx.x*blockDim.x + threadIdx.x</code> and a bounds check</Translated></td>
          <td><Translated vi="cùng mã nguồn đó">the same source</Translated></td>
          <td><Translated vi={<><code>[[thread_position_in_grid]]</code>; với <code>dispatchThreads</code> không luồng nào rơi ra ngoài lưới</>}><code>[[thread_position_in_grid]]</code>; with <code>dispatchThreads</code> no thread falls outside the grid</Translated></td>
        </tr>
        <tr>
          <td><Translated vi="Phép cộng nguyên tử số thực (COO, CSC)">Float atomic add (COO, CSC)</Translated></td>
          <td><Translated vi={<><code>cuda::atomic_ref&lt;float, cuda::thread_scope_device&gt;</code>, <code>fetch_add(v, cuda::memory_order_relaxed)</code>; hoặc <code>atomicAdd</code> kiểu cũ (legacy)</>}><code>cuda::atomic_ref&lt;float, cuda::thread_scope_device&gt;</code>, <code>fetch_add(v, cuda::memory_order_relaxed)</code>; or the legacy <code>atomicAdd</code></Translated></td>
          <td><Translated vi={<>cùng cách viết qua libhipcxx (ROCm 10.1 trở lên); hoặc <code>atomicAdd</code>, thực hiện an toàn, có thể dưới dạng vòng lặp compare-and-swap, trừ khi dựng với <code>-munsafe-fp-atomics</code></>}>the same spelling through libhipcxx (ROCm 10.1 and later); or <code>atomicAdd</code>, done safely, possibly as a compare-and-swap loop, unless built with <code>-munsafe-fp-atomics</code></Translated></td>
          <td><Translated vi={<><code>atomic_fetch_add_explicit</code> trên một buffer <code>device atomic_float*</code>, chỉ với <code>memory_order_relaxed</code>; Metal 3 và Apple7 trở lên</>}><code>atomic_fetch_add_explicit</code> on a <code>device atomic_float*</code> buffer, <code>memory_order_relaxed</code> only; Metal 3 and Apple7 or later</Translated></td>
        </tr>
        <tr>
          <td><Translated vi="Thứ tự các phép cộng nguyên tử số thực">Order of float atomic adds</Translated></td>
          <td><Translated vi="không xác định">undefined</Translated></td>
          <td><Translated vi="không xác định">undefined</Translated></td>
          <td><Translated vi="không có tài liệu">not documented</Translated></td>
        </tr>
        <tr>
          <td><Translated vi="Đơn vị dùng để đếm việc hợp nhất truy cập bộ nhớ (coalescing)">Unit that coalescing is counted in</Translated></td>
          <td><Translated vi="sector 32 byte">32-byte sectors</Translated></td>
          <td><Translated vi="dòng cache 64 hoặc 128 byte">64- or 128-byte cache lines</Translated></td>
          <td><Translated vi="không có tài liệu">not documented</Translated></td>
        </tr>
        <tr>
          <td><Translated vi="Phân kỳ luồng (các vòng lặp không đều của CSR)">Divergence (CSR's uneven loops)</Translated></td>
          <td><Translated vi="có tài liệu: các lượt chạy bị che mặt nạ theo từng warp">documented: masked passes per warp</Translated></td>
          <td><Translated vi="có tài liệu: mặt nạ EXEC, theo từng wavefront">documented: the EXEC mask, per wavefront</Translated></td>
          <td><Translated vi="không có tài liệu">not documented</Translated></td>
        </tr>
        <tr>
          <td><Translated vi="Struct chứa các con trỏ device của sách, dùng làm đối số">The book's struct of device pointers as an argument</Translated></td>
          <td><Translated vi="truyền theo giá trị; tổng mọi tham số tối đa 32,764 byte">by value; all parameters together at most 32,764 bytes</Translated></td>
          <td><Translated vi="cùng cú pháp; không tìm thấy giới hạn kích thước">the same syntax; no size limit found</Translated></td>
          <td><Translated vi="một argument buffer (Metal 2 trở lên)">an argument buffer (Metal 2 and later)</Translated></td>
        </tr>
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={TIERS}>
      <LabsIntro>
        <Translated
          vi={
            <>
              Một lab, trong <code>labs/ch17/lab1_spmv/</code>. Lab này là tùy chọn, vì chương này được rút gọn, và nó không thêm gì vào
              dự án xuyên suốt. Nó biến hai kernel mỗi-hàng-một-luồng của trang <a href={href('ch17', '17.2')}>17.2</a> thành
              mã bạn chạy trên mọi backend, và benchmark của nó đặt câu hỏi của trang{' '}
              <a href={href('ch17', '17.5')}>17.5</a> trên chính GPU của bạn: việc đệm, sắp xếp và tràn mua được gì trên các hàng có độ dài
              không đều.
            </>
          }
        >
          One lab, in <code>labs/ch17/lab1_spmv/</code>. It is optional, as this chapter is condensed, and it adds nothing to
          the through-line project. It turns the two per-row kernels of page <a href={href('ch17', '17.2')}>17.2</a> into
          code you run on every backend, and its benchmark asks the question of page{' '}
          <a href={href('ch17', '17.5')}>17.5</a> on your own GPU: what padding, sorting and spilling buy on rows of uneven
          length.
        </Translated>
      </LabsIntro>
      <RunningLabs example="17.1" />
      <BackendsAndEmulator />
      <VerificationStatus />

      {/* ------------------------------------------------------------ 17.1 */}
      <h2>
        <Translated vi="Lab 17.1: SpMV với CSR và ELL trên các hàng lệch">Lab 17.1: CSR and ELL SpMV on skewed rows</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn viết SpMV, <code>y = A·x + y</code>, hai lần, cả hai lần đều một luồng (thread) mỗi hàng. Trong CSR, một hàng là dải các mục
            nằm giữa <code>rowPtrs[row]</code> và <code>rowPtrs[row+1]</code>. Trong ELL, mọi hàng được đệm tới cùng một độ rộng và
            hình chữ nhật đã đệm được lưu theo thứ tự cột (column-major), nên vị trí <code>t</code> của hàng <code>r</code> nằm ở{' '}
            <code>t*stride + r</code> và các luồng kề nhau đọc các vị trí kề nhau ở mọi lần lặp. Cả hai kernel giữ
            tổng trong một thanh ghi (register) và cộng nó vào <code>y[row]</code> một lần.
          </p>
        }
      >
        <p>
          You write SpMV, <code>y = A·x + y</code>, twice, one thread per row both times. In CSR a row is the run of entries
          between <code>rowPtrs[row]</code> and <code>rowPtrs[row+1]</code>. In ELL every row is padded to the same width and
          the padded rectangle is stored column-major, so slot <code>t</code> of row <code>r</code> sits at{' '}
          <code>t*stride + r</code> and neighbouring threads read neighbouring slots in every iteration. Both kernels keep the
          sum in a register and add it to <code>y[row]</code> once.
        </p>
      </Translated>
      <Source section="17.3, 17.4" pages="407–414" figs={tr('Figs. 17.9, 17.12', 'Hình 17.9, 17.12')} />
      <Translated
        vi={
          <p>
            Chương trình điều khiển (driver) cũng chạy dạng lai của trang 17.5 với kernel ELL của bạn. Các hàng dài hơn K giữ K mục đầu trong
            ELL và cho phần còn lại tràn sang một danh sách COO, mà một kernel cung cấp sẵn, Hình 17.5 đúng như sách viết, cộng vào{' '}
            <code>y</code> bằng phép toán nguyên tử (atomic) trước khi kernel của bạn cộng phần ELL.
          </p>
        }
      >
        <p>
          The driver also runs the hybrid of page 17.5 with your ELL kernel. Rows longer than K keep their first K entries in
          ELL and spill the rest into a COO list, which a provided kernel, Fig. 17.5 as the book writes it, adds into{' '}
          <code>y</code> with atomics before your kernel adds the ELL part.
        </p>
      </Translated>
      <Source section="17.2, 17.5" pages="404–405, 414–415" figs={tr('Figs. 17.5, 17.13', 'Hình 17.5, 17.13')} />
      <Translated
        vi={
          <p>
            Với ELL, bạn chọn vòng lặp. Dừng ở <code>nnzPerRow[row]</code>, như sách làm, hoặc đi qua cả{' '}
            <code>width</code> vị trí và bỏ qua mọi vị trí có cột âm, như các vòng lặp tham chiếu của thư viện nhà cung cấp. Cả hai đều
            đúng; chúng khác nhau ở chỗ các lane của hàng ngắn làm gì trong lúc các hàng dài chạy xong. Cả hai hàm khởi chạy dùng các khối{' '}
            <code>SPMV_BLOCK</code> = 128 luồng, đủ số khối để phủ mọi hàng bằng phép chia nguyên làm tròn lên, và không khởi chạy
            gì khi <code>numRows</code> bằng 0.
          </p>
        }
      >
        <p>
          For ELL you choose the loop. Stop at <code>nnzPerRow[row]</code>, as the book does, or visit all{' '}
          <code>width</code> slots and skip any whose column is negative, as the vendor libraries' reference loops do. Both are
          correct; they differ in what the short rows' lanes do while the long rows finish. Both launchers use blocks of{' '}
          <code>SPMV_BLOCK</code> = 128 threads, enough of them to cover every row by integer ceiling division, and launch
          nothing when <code>numRows</code> is 0.
        </p>
      </Translated>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <>
              <Translated
                vi={
                  <>
                    <code>spmvCsrKernel</code>, <code>launchSpmvCsr</code>, <code>spmvEllKernel</code> và{' '}
                    <code>launchSpmvEll</code>, với con trỏ device thô và kích thước tường minh làm đối số
                  </>
                }
              >
                <code>spmvCsrKernel</code>, <code>launchSpmvCsr</code>, <code>spmvEllKernel</code> and{' '}
                <code>launchSpmvEll</code>, with raw device pointers and explicit sizes as arguments
              </Translated>
            </>,
          ],
          [
            'kernel.metal',
            <>
              <Translated
                vi={
                  <>
                    thân của <code>spmv_csr</code> và <code>spmv_ell</code>; chữ ký hàm và các ràng buộc buffer đã có sẵn
                  </>
                }
              >
                the bodies of <code>spmv_csr</code> and <code>spmv_ell</code>; the signatures and buffer bindings are given
              </Translated>
            </>,
          ],
          [
            'launch.mm',
            <>
              <Translated
                vi={
                  <>
                    <code>encodeSpmvCsr</code> và <code>encodeSpmvEll</code>: không encode gì khi <code>numRows</code> bằng 0,
                    nếu không thì một luồng mỗi hàng với <code>dispatchThreads</code>
                  </>
                }
              >
                <code>encodeSpmvCsr</code> and <code>encodeSpmvEll</code>: encode nothing when <code>numRows</code> is 0,
                otherwise one thread per row with <code>dispatchThreads</code>
              </Translated>
            </>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Chỉ đọc: <code>spmv.h</code> (<code>SPMV_BLOCK</code>), <code>provided/coo_spmv.cu</code>,{' '}
            <code>provided/coo_spmv.metal</code> và <code>provided/coo_spmv.mm</code> (kernel COO của dạng lai và các hàm
            khởi chạy của nó), và <code>main.cpp</code>: bộ sinh ma trận, các hàm dựng định dạng, bản tham chiếu CPU, các bài kiểm thử và
            benchmark.
          </p>
        }
      >
        <p>
          Read-only: <code>spmv.h</code> (<code>SPMV_BLOCK</code>), <code>provided/coo_spmv.cu</code>,{' '}
          <code>provided/coo_spmv.metal</code> and <code>provided/coo_spmv.mm</code> (the hybrid's COO kernel and its
          launchers), and <code>main.cpp</code>: the matrix generator, the format builders, the CPU reference, the tests and
          the benchmark.
        </p>
      </Translated>

      <h3>
        <Translated vi="Lab khác trang sách ở đâu">Where the lab differs from the page</Translated>
      </h3>
      <Translated
        vi={
          <p>
            <b>Con trỏ thô, không phải struct.</b> Sách truyền một struct <code>CSRMatrix</code> hoặc <code>ELLMatrix</code> cho mỗi
            kernel và không cho thấy cả struct lẫn mã host nào. Ở đây mỗi mảng là một đối số riêng, ánh xạ một-một
            sang các ràng buộc buffer của Metal. Các hàm khởi chạy là của khóa học này.
          </p>
        }
      >
        <p>
          <b>Raw pointers, not structs.</b> The book passes a <code>CSRMatrix</code> or <code>ELLMatrix</code> struct to each
          kernel and shows neither the struct nor any host code. Here each array is its own argument, which maps one to one
          onto Metal's buffer bindings. The launchers are ours.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>Kernel ELL cộng dồn.</b> Hình 17.12 kết thúc bằng <code>y[row] = sum</code>, Hình 17.9 bằng{' '}
            <code>y[row] += sum</code>, và SpMV là nhân rồi cộng dồn. Hãy viết <code>+=</code> ở cả hai. Đó cũng là điều cho phép
            dạng lai chạy phần COO trước: với <code>=</code>, kernel của bạn sẽ xóa sạch những gì kernel COO đã cộng vào.
          </p>
        }
      >
        <p>
          <b>The ELL kernel accumulates.</b> Fig. 17.12 ends with <code>y[row] = sum</code>, Fig. 17.9 with{' '}
          <code>y[row] += sum</code>, and SpMV is multiply and accumulate. Write <code>+=</code> in both. It is also what lets
          the hybrid run its COO part first: with <code>=</code>, your kernel would wipe out what the COO kernel added.
        </p>
      </Translated>
      <Source pages="403, 409, 412" figs={tr('Figs. 17.9, 17.12', 'Hình 17.9, 17.12')} />
      <Translated
        vi={
          <p>
            <b>Một stride, không phải numRows.</b> Hình 17.12 bước từ một vị trí của hàng sang vị trí kế tiếp bằng <code>numRows</code>; ở đây bằng{' '}
            <code>stride</code>, có thể lớn hơn. Đó chính là nhận xét của sách ở p. 413, thêm các hàng trước khi
            xoay hình chữ nhật để mọi lần lặp bắt đầu ở một địa chỉ được căn lề, nay được làm tường minh. Các bài kiểm thử dùng{' '}
            <code>numRows</code>, <code>numRows</code> làm tròn lên bội của 8, và <code>numRows + 37</code>.
          </p>
        }
      >
        <p>
          <b>A stride, not numRows.</b> Fig. 17.12 steps from one slot of a row to the next by <code>numRows</code>; here by{' '}
          <code>stride</code>, which may be larger. That is the book's own remark on p. 413, appending rows before the
          rectangle is turned so that every iteration starts on an aligned address, made explicit. The tests use{' '}
          <code>numRows</code>, <code>numRows</code> rounded up to 8, and <code>numRows + 37</code>.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>Phần đệm mang cột −1 và giá trị 0.</b> Sách nói giá trị đệm bằng 0 và không bao giờ nói cột của một phần đệm chứa gì.
            −1 là quy ước của các thư viện nhà cung cấp, nên một vòng lặp trên toàn bộ độ rộng phải kiểm tra cột trước khi đọc{' '}
            <code>x</code>.
          </p>
        }
      >
        <p>
          <b>Pads hold column −1 and value 0.</b> The book says pad values are zero and never says what a pad's column holds.
          −1 is the vendor libraries' convention, so a loop over the full width must test the column before it reads{' '}
          <code>x</code>.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch17-facts.md A6, A10, A11, B3, C2)', '2026-11-20 (notes/today/ch17-facts.md, mục A6, A10, A11, B3, C2)')}>
        <Translated
          vi={
            <p>
              Sliced ELL của cuSPARSE và ELL của rocSPARSE đệm bằng cột −1; mọi SpMV của thư viện đều tính y = α·op(A)·x + β·y.
              CUDA nhận một struct chứa con trỏ theo giá trị; Metal sẽ cần một argument buffer.
            </p>
          }
        >
          <p>
            cuSPARSE's Sliced ELL and rocSPARSE's ELL pad with column −1; every library SpMV computes y = α·op(A)·x + β·y.
            CUDA takes a struct of pointers by value; Metal would need an argument buffer.
          </p>
        </Translated>
      </Today>

      <h3>
        <Translated vi="Cùng các kernel ấy trên các backend khác">The same kernels on other backends</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Hai kernel của bạn không cần gì ngoài chỉ số luồng. Kernel COO của dạng lai, giống các kernel COO và CSC của sách,
            cần một phép cộng nguyên tử số thực, và đó là chỗ các backend rẽ theo những hướng khác nhau.
          </p>
        }
      >
        <p>
          Your two kernels need nothing beyond a thread index. The hybrid's COO kernel, like the book's COO and CSC kernels,
          needs a float atomic add, and that is where the backends part ways.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. Mọi thứ trong bảng này về HIP và Metal đều lấy từ notes/today/ch17-facts.md, "Where the
              backends really differ", kiểm tra ngày 2026-11-20.
            </p>
          }
        >
          <p>
            The book uses CUDA only. Everything in this table about HIP and Metal is from notes/today/ch17-facts.md, "Where the
            backends really differ", checked 2026-11-20.
          </p>
        </Translated>
      </Beyond>
      <BackendsTable />

      <h3>
        <Translated vi="Các bài kiểm thử kiểm tra gì">What the tests check</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 17.1 --backend emu</code> trước. Mọi định dạng được dựng trên host từ một
            ma trận sinh ra duy nhất, và bản tham chiếu CPU cộng dồn bằng double. Mỗi kernel được kiểm thử trên 0 hàng, 1 hàng, 7, 33, 65 và
            129 hàng (không số nào là bội của khối), một ma trận chữ nhật 300 x 305, cứ ba hàng có một hàng rỗng, mọi hàng rỗng,
            mọi hàng cùng độ dài, các hàng dài 1 đến 3 với một hàng dài 40, và độ dài ngẫu nhiên từ 0 đến 12. ELL chạy với ba
            stride và một lần với độ rộng lớn hơn mọi hàng, để có những cột toàn là phần đệm. Dạng lai chạy với K = 0
            (mọi thứ nằm trong COO), 1, độ dài hàng trung vị và hàng dài nhất (không có phần COO). Thiết bị thật thêm bốn
            ma trận của benchmark với 65,541 hàng.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 17.1 --backend emu</code> first. Every format is built on the host from one
          generated matrix, and the CPU reference accumulates in double. Each kernel is tested on 0 rows, 1 row, 7, 33, 65 and
          129 rows (none a multiple of the block), a rectangular 300 by 305 matrix, every third row empty, every row empty,
          every row the same length, rows of 1 to 3 with one row of 40, and random lengths from 0 to 12. ELL runs at three
          strides and once with a width larger than any row, so that whole columns are pads. The hybrid runs at K = 0
          (everything in COO), 1, the median row length and the longest row (no COO part). Real devices add the benchmark's
          four matrices at 65,541 rows.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Mỗi phép kiểm tra được dựng để bắt một lỗi. <code>y</code> bắt đầu bằng các giá trị ngẫu nhiên khác 0 và phải kết thúc bằng{' '}
            <code>y0 + A·x</code>, nên một phép gán sẽ thất bại, và thông báo cho biết khi mọi hàng sai chỉ chứa <code>A·x</code>{' '}
            mà thôi. 128 phần tử canh gác sau <code>y</code> phải còn nguyên, điều này bắt được việc thiếu phép kiểm tra{' '}
            <code>row &lt; numRows</code>. Trên CUDA, HIP và cả hai trình giả lập, <code>x</code> được truyền vào dưới dạng buffer + 1 với một NaN đứng trước{' '}
            <code>x[0]</code>, nên việc đọc <code>x</code> ở cột −1 của một phần đệm biến hàng đó thành NaN. Các hàng căn lề nằm sau{' '}
            <code>numRows</code> mang giá trị NaN, nên một kernel bước theo <code>numRows</code> thay vì <code>stride</code>{' '}
            sẽ thất bại. Một hàng đạt khi nằm trong khoảng <code>1e-4 × (1 + |y0| + Σ|a·x|)</code> quanh bản tham chiếu: một dung sai co giãn theo độ lớn
            của các số hạng trong hàng, không bao giờ chỉ là tương đối khi gần 0.
          </p>
        }
      >
        <p>
          Each check is built to catch one mistake. <code>y</code> starts as random non-zero values and must end as{' '}
          <code>y0 + A·x</code>, so an assignment fails, and the message says when every wrong row holds <code>A·x</code>{' '}
          alone. 128 sentinel elements after <code>y</code> must survive, which catches a missing <code>row &lt; numRows</code>{' '}
          test. On CUDA, HIP and both emulators, <code>x</code> is handed over as buffer + 1 with a NaN in front of{' '}
          <code>x[0]</code>, so reading <code>x</code> at a pad's column −1 turns the row into NaN. The alignment rows past{' '}
          <code>numRows</code> hold value NaN, so a kernel that steps by <code>numRows</code> instead of <code>stride</code>{' '}
          fails. A row passes within <code>1e-4 × (1 + |y0| + Σ|a·x|)</code> of the reference: a tolerance scaled by the size
          of the row's terms, never relative-only near zero.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch05-facts.md C7; ch10-facts A6)', '2026-11-20 (notes/today/ch05-facts.md, mục C7; ch10-facts A6)')}>
        <Translated
          vi={
            <p>
              Metal không mặc định dùng số học IEEE nghiêm ngặt, và một phép cộng nguyên tử số thực trên bộ nhớ toàn cục (global memory) làm tròn các số subnormal về
              0, nên một tổng COO có thể khác CPU khi gần 0.
            </p>
          }
        >
          <p>
            Metal does not default to strict IEEE arithmetic, and a float atomic add on global memory flushes subnormals to
            zero, so a COO sum can differ from the CPU near zero.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Trình giả lập (emulator) chạy từng luồng một theo ba thứ tự cố định dưới AddressSanitizer, và nó từ chối một lần khởi chạy không
            khối nào, như HIP. Biến thể <code>race</code> chạy các ma trận 33 và 65 hàng với các luồng chạy đồng thời dưới
            ThreadSanitizer: <code>python3 labs/pmpp.py run 17.1 --backend race</code>. Kernel của bạn không cần atomic, vì mỗi
            hàng thuộc về một luồng; kernel COO của dạng lai thì cần, và một <code>+=</code> thường ở đó sẽ bị báo cáo kèm dòng
            mã nguồn. Một lần chạy sạch là bằng chứng, không phải chứng minh: nó chỉ thấy những lịch chạy đã thực sự xảy ra.
          </p>
        }
      >
        <p>
          The emulator runs one thread at a time in three fixed orders under AddressSanitizer, and it rejects a launch of zero
          blocks, as HIP does. The <code>race</code> flavour runs the 33- and 65-row matrices with the threads concurrent under
          ThreadSanitizer: <code>python3 labs/pmpp.py run 17.1 --backend race</code>. Your kernels need no atomics, since one
          thread owns each row; the hybrid's COO kernel does, and a plain <code>+=</code> there is reported with its source
          line. A clean run is evidence, not proof: it sees only the schedules that happened.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md A8, B4)', '2026-11-20 (notes/today/ch02-facts.md, mục A8, B4)')}>
        <Translated
          vi={
            <p>
              HIP 7.0 trả về lỗi khi một chiều của lưới bằng 0. Tài liệu của CUDA không nói về trường hợp này, nên một lần khởi chạy
              không khối nào có thể trôi qua mà không ai để ý ở đó.
            </p>
          }
        >
          <p>
            HIP 7.0 returns an error for a grid dimension of zero. CUDA's documentation does not state the case, so a launch of
            zero blocks may pass unnoticed there.
          </p>
        </Translated>
      </Today>
      <RaceStatus />
      <Translated
        vi={
          <p>
            Trên Metal, <code>[[thread_position_in_grid]]</code> là hàng và <code>dispatchThreads</code> tạo đúng{' '}
            <code>numRows</code> luồng, nên phép kiểm tra hàng không bao giờ sai; dù vậy vẫn giữ nó. Kernel COO cung cấp sẵn khai báo{' '}
            <code>y</code> là <code>device atomic_float*</code> và cộng bằng một <code>atomic_fetch_add_explicit</code> relaxed;
            kernel ELL của bạn ràng buộc cùng buffer đó dưới dạng <code>device float*</code>. Hai lần dispatch dùng chung một compute encoder,
            chạy chúng theo thứ tự. Metal không có NaN canh gác trước <code>x</code>: fast math khiến việc lan truyền NaN
            không đáng tin, và không có cách nào đã kiểm chứng để đưa cho kernel <code>x + 1</code>. Một kernel đọc{' '}
            <code>x[-1]</code> ở đó sẽ đọc bất cứ thứ gì nằm trước buffer, nên hãy chạy <code>emu</code> để bắt lỗi đó. Trong các bài kiểm thử{' '}
            <code>numRows = 0</code>, chương trình điều khiển đặt một bộ đếm vào chỗ encoder, và bất kỳ lần dispatch nào cũng làm bài kiểm thử thất bại.
          </p>
        }
      >
        <p>
          On Metal, <code>[[thread_position_in_grid]]</code> is the row and <code>dispatchThreads</code> makes exactly{' '}
          <code>numRows</code> threads, so the row check is never false; keep it anyway. The provided COO kernel declares{' '}
          <code>y</code> as <code>device atomic_float*</code> and adds with a relaxed <code>atomic_fetch_add_explicit</code>;
          your ELL kernel binds the same buffer as <code>device float*</code>. The two dispatches share one compute encoder,
          which runs them in order. Metal gets no NaN guard in front of <code>x</code>: fast math makes NaN propagation
          unreliable, and there is no checked way to hand the kernel <code>x + 1</code>. A kernel that reads{' '}
          <code>x[-1]</code> there reads whatever lies before the buffer, so run <code>emu</code> for that mistake. In the{' '}
          <code>numRows = 0</code> tests the driver puts a counter in place of the encoder, and any dispatch fails the test.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch17-facts.md C1; ch05-facts C4, C5; ch02-facts C9)', '2026-11-20 (notes/today/ch17-facts.md, mục C1; ch05-facts C4, C5; ch02-facts C9)')}>
        <Translated
          vi={
            <p>
              Phép cộng nguyên tử số thực trên bộ nhớ device cần Metal 3 và Apple7, chỉ với thứ tự relaxed. Một compute encoder thường
              dispatch tuần tự, và các buffer lấy từ device được theo dõi hazard.
            </p>
          }
        >
          <p>
            Float atomic add on device memory needs Metal 3 and Apple7, with relaxed order only. A plain compute encoder
            dispatches serially, and buffers from the device are hazard-tracked.
          </p>
        </Translated>
      </Today>

      <h3>
        <Translated vi="Thí nghiệm">The experiment</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Làm việc này khi các kernel của bạn đã đạt, và trước khi bạn chạy <code>--bench</code>. Benchmark dựng bốn ma trận có
            cùng số hàng và trung bình khoảng 16 mục mỗi hàng, các cột nằm trong khoảng ±2048 quanh đường chéo.{' '}
            <code>uniform</code>: mọi hàng dài 16. <code>spread</code>: độ dài phân bố đều từ 1 đến 31, theo thứ tự ngẫu nhiên.{' '}
            <code>spread_sorted</code>: cùng các hàng đó, hàng dài nhất trước, kiểu sắp xếp của JDS. Mỗi hàng giữ nguyên các cột của nó, nên sau khi
            sắp xếp, các luồng kề nhau không còn đọc những phần kề nhau của <code>x</code>. <code>skewed</code>: các hàng dài 16 và cứ
            512 hàng có một hàng dài 256. Nó đo thời gian <code>csr</code>, <code>ell</code> (stride = <code>numRows</code>),{' '}
            <code>ell_aligned</code> (stride làm tròn lên bội của 32 số float) và, trên <code>spread</code> và{' '}
            <code>skewed</code>, <code>hybrid</code> ở K = 16 với lượt COO của nó được tính vào thời gian.
          </p>
        }
      >
        <p>
          Do this once your kernels pass, and before you run <code>--bench</code>. The benchmark builds four matrices with the
          same number of rows and about 16 entries per row on average, columns within ±2048 of the diagonal.{' '}
          <code>uniform</code>: every row 16 long. <code>spread</code>: lengths uniform on 1 to 31, in random order.{' '}
          <code>spread_sorted</code>: the same rows, longest first, the sort of JDS. Each row keeps its columns, so after the
          sort neighbouring threads no longer read neighbouring parts of <code>x</code>. <code>skewed</code>: rows of 16 and one
          row in 512 of length 256. It times <code>csr</code>, <code>ell</code> (stride = <code>numRows</code>),{' '}
          <code>ell_aligned</code> (stride rounded up to a multiple of 32 floats) and, on <code>spread</code> and{' '}
          <code>skewed</code>, <code>hybrid</code> at K = 16 with its COO pass timed in.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Mô hình của trang 17.2 nói các lần đọc của ELL được hợp nhất (coalescing) còn của CSR thì không, và mô hình của trang 17.5 nói một hàng dài khiến ELL trả giá cho
            độ rộng của nó ở mọi hàng. Cả hai đều đếm yêu cầu và bỏ qua bộ nhớ đệm (cache), và Apple hoàn toàn không công bố mô hình hợp nhất hay phân kỳ
            nào. Không câu trả lời nào bên dưới được bảo đảm bởi bất cứ điều gì bạn đã đọc. Hãy ghi lại hai con số, và, không ghi lại,
            thêm hai phỏng đoán: kernel CSR không đổi có chạy nhanh hơn trên <code>spread_sorted</code> so với trên{' '}
            <code>spread</code> không, và <code>ell_aligned</code> có thắng <code>ell</code> không?
          </p>
        }
      >
        <p>
          Page 17.2's model says ELL's reads coalesce and CSR's do not, and page 17.5's says one long row makes ELL pay for its
          width in every row. Both count requests and ignore caches, and Apple documents no coalescing or divergence model at
          all. Neither answer below is promised by anything you have read. Write down two numbers, and, without recording
          them, two more guesses: will the unchanged CSR kernel run faster on <code>spread_sorted</code> than on{' '}
          <code>spread</code>, and will <code>ell_aligned</code> beat <code>ell</code>?
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch17-facts.md A3, C3)', '2026-11-20 (notes/today/ch17-facts.md, mục A3, C3)')}>
        <Translated
          vi={
            <p>
              NVIDIA đếm việc hợp nhất theo sector 32 byte trên mọi thiết bị mà CUDA hiện tại hỗ trợ; AMD công bố dòng 64 và 128
              byte. Apple không công bố đơn vị, quy tắc căn lề hay mô hình phân kỳ nào.
            </p>
          }
        >
          <p>
            NVIDIA counts coalescing in 32-byte sectors on every device current CUDA supports; AMD documents 64- and 128-byte
            lines. Apple documents no unit, alignment rule or divergence model.
          </p>
        </Translated>
      </Today>
      <LabPrediction
        id="ch17/labs/p-lab1-uniform"
        prompt={
          <Translated
            vi={
              <>
                Trên <code>uniform</code>, thời gian CSR chia cho thời gian ELL trên GPU của bạn. (Trên 1 nghĩa là ELL nhanh hơn.)
              </>
            }
          >
            <>
              On <code>uniform</code>, CSR time divided by ELL time on your GPU. (Above 1 means ELL was faster.)
            </>
          </Translated>
        }
        unit={tr('(CSR time / ELL time)', '(thời gian CSR / thời gian ELL)')}
        measure={uniformRatio}
        missing={MISSING}
        reading={
          <Translated
            vi={
              <>
                Khi mọi hàng dài 16 thì không có gì phân kỳ, nên tỉ số này tách riêng ảnh hưởng của cách bố trí. Mỗi luồng CSR đi qua 64 byte
                liên tiếp của riêng nó trong 16 lần lặp, nên một cache giữ được các dòng đó có thể phục vụ phần lớn những gì
                cách đếm theo từng lần lặp gọi là rải rác.
              </>
            }
          >
            <>
              With every row 16 long nothing diverges, so the ratio isolates the layout. Each CSR thread walks 64 contiguous
              bytes of its own over 16 iterations, so a cache that keeps those lines can serve most of what the
              per-iteration count calls scattered.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch17/labs/p-lab1-skewed"
        prompt={
          <Translated
            vi={
              <>
                Trên <code>skewed</code>, thời gian ELL chia cho thời gian CSR. (Trên 1 nghĩa là ELL chậm hơn.) Rồi nói, không
                ghi lại, liệu dạng lai có lấy lại được thời gian của CSR hoặc tốt hơn không.
              </>
            }
          >
            <>
              On <code>skewed</code>, ELL time divided by CSR time. (Above 1 means ELL was slower.) Then say, without
              recording it, whether the hybrid gets back to CSR's time or better.
            </>
          </Translated>
        }
        unit={tr('(ELL time / CSR time)', '(thời gian ELL / thời gian CSR)')}
        measure={skewedRatio}
        missing={MISSING}
        reading={
          <Translated
            vi={
              <>
                ELL ở đây rộng 256 vị trí cho các hàng dài 16, và kernel tham chiếu dừng ở <code>nnzPerRow</code>, nên nó
                lưu phần đệm nhưng không đọc chúng. Điều nó vẫn giữ là phân kỳ của CSR: một hàng dài giữ chân cả
                nhóm luồng của nó trong 256 lần lặp, ở định dạng nào cũng vậy.
              </>
            }
          >
            <>
              ELL here is 256 slots wide for rows of 16, and the reference kernel stops at <code>nnzPerRow</code>, so it
              stores the pads but does not read them. What it does keep is CSR's divergence: one long row holds its whole
              group of threads for 256 iterations, in either format.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <LabResults
            lab={l1}
            cols={COLS.map((c) => ({
              ...c,
              head:
                ({
                  Width: 'Độ rộng',
                  Stride: 'Stride',
                  'COO entries': 'Số mục COO',
                  'GB/s, useful bytes': 'GB/s, byte hữu ích',
                } as Record<string, string>)[c.head] ?? c.head,
            }))}
          />
        }
      >
        <LabResults lab={l1} cols={COLS} />
      </Translated>
      <Translated
        vi={
          <p>
            Benchmark hoạt động thế nào. Kích thước lấy từ thiết bị của bạn: 2^16 + 5 hàng, nhân đôi chừng nào CSR trên{' '}
            <code>uniform</code> còn mất dưới 2 ms, tối đa 2^21 + 5, và chỉ chừng nào ma trận skewed ở dạng ELL còn vừa trong một phần tư
            bộ nhớ mà thiết bị báo cáo. Phần + 5 cố ý làm cho <code>stride = numRows</code> không phải bội của 8. Một định dạng
            không vừa sẽ được in là bị bỏ qua, kèm số byte nó cần. Mọi kết quả được kiểm tra với CPU trước khi
            đo thời gian, và một thời gian là lần nhanh nhất trong năm lần chạy sau một lượt khởi động. "MB ma trận" là những gì định dạng lưu; một kernel
            dừng ở <code>nnzPerRow</code> không đọc phần đệm. "GB/s, byte hữu ích" lấy 8 byte mỗi mục cộng 16 byte mỗi
            hàng (một con trỏ hoặc độ dài, <code>x</code> một lần, <code>y</code> đọc và ghi) chia cho thời gian: lượng tối thiểu mà định dạng nào
            cũng phải di chuyển, cùng một tử số cho mọi định dạng, không phải lượng mà hệ thống bộ nhớ thực sự đã di chuyển.
          </p>
        }
      >
        <p>
          How the benchmark works. The size comes from your device: 2^16 + 5 rows, doubled while CSR on{' '}
          <code>uniform</code> takes under 2 ms, up to 2^21 + 5, and only while the skewed matrix in ELL fits in a quarter of
          the memory the device reports. The + 5 makes <code>stride = numRows</code> deliberately not a multiple of 8. A format
          that would not fit is printed as skipped, with the bytes it needs. Every result is checked against the CPU before it
          is timed, and a time is the best of five runs after a warm-up. "Matrix MB" is what the format stores; a kernel that
          stops at <code>nnzPerRow</code> does not read the pads. "GB/s, useful bytes" divides 8 bytes per entry plus 16 per
          row (a pointer or length, <code>x</code> once, <code>y</code> read and written) by the time: the least any format
          could move, the same numerator for every format, not what the memory system moved.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chương trình điều khiển cũng in thời gian CSR chia cho thời gian của từng định dạng, theo từng ma trận. Cũng các tỉ số đó, từ báo cáo của máy này,
            trả lời hai phỏng đoán bạn không ghi lại.
          </p>
        }
      >
        <p>
          The driver also prints CSR time over each format's time per matrix. The same ratios, from this machine's reports,
          answer the two guesses you did not record.
        </p>
      </Translated>
      <Ratios />
      <Translated
        vi={
          <p>
            Thời gian của cùng một kernel và ma trận thay đổi từ lần chạy này sang lần chạy khác, nên hãy chạy benchmark hai lần trước khi tin vào một
            chênh lệch nhỏ giữa hai định dạng.
          </p>
        }
      >
        <p>
          Times for the same kernel and matrix vary from one run to the next, so run the benchmark twice before you trust a
          small difference between two formats.
        </p>
      </Translated>

      <h3>
        <Translated vi="Điều này nằm ở đâu dưới những gì bạn đã biết">Where this sits under what you know</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Một batch có độ dài lởm chởm là cùng bài toán đó trong bộ áo khác. Đệm mọi chuỗi cho dài bằng chuỗi dài nhất chính là ELL, với cùng
            cái giá do chuỗi dài nhất quyết định. Chia chuỗi vào các nhóm theo độ dài là phép sắp xếp của JDS, với một mảng chỉ số để đưa kết quả
            về chỗ cũ. Phục vụ số ít chuỗi rất dài qua một đường riêng chính là dạng lai.
          </p>
        }
      >
        <p>
          A ragged batch is the same problem in other clothes. Padding every sequence to the longest is ELL, with the same
          price set by the longest. Bucketing sequences by length is the JDS sort, with an index array to put the results
          back. Serving the few very long ones down a separate path is the hybrid.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Cách đọc theo batch lởm chởm là của khóa học này, không phải của sách; trang 17.5 thực hiện cùng phép ánh xạ cho việc định tuyến mixture-of-experts.</p>}>
          <p>The ragged-batch reading is ours, not the book's; page 17.5 makes the same mapping for mixture-of-experts routing.</p>
        </Translated>
      </Beyond>

      <h3>
        <Translated vi="Đi xa hơn">Go further</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Bài tập 5 yêu cầu một kernel JDS. Ma trận <code>spread_sorted</code> đã ở sẵn thứ tự hàng của JDS; phần còn lại là
            cách bố trí theo thứ tự cột không đệm và mảng <code>iterPtr</code>.{' '}
            <a href={href('ch17', 'wrap')}>Trang tóm tắt</a> cho một lời giải, đã kiểm tra trên trình giả lập. Bài tập 4 yêu cầu
            phần COO của dạng lai chạy trên host: đo thời gian phần đó so với kernel cung cấp sẵn ở K = 16 trên <code>skewed</code>, tính cả
            các lần sao chép. Và đổi vòng lặp ELL của bạn sang vòng lặp đúng còn lại, rồi đo cả hai trên <code>spread</code> và{' '}
            <code>skewed</code>. Vòng lặp nào đọc phần đệm?
          </p>
        }
      >
        <p>
          Exercise 5 asks for a JDS kernel. The <code>spread_sorted</code> matrix is already in JDS row order; what is left is
          the column-major layout without padding and the <code>iterPtr</code> array. The{' '}
          <a href={href('ch17', 'wrap')}>summary page</a> shows one answer, checked on the emulator. Exercise 4 asks for the
          hybrid's COO part on the host: time that against the provided kernel at K = 16 on <code>skewed</code>, copies
          included. And swap your ELL loop for the other correct one, then measure both on <code>spread</code> and{' '}
          <code>skewed</code>. Which one reads the pads?
        </p>
      </Translated>
      <Source section="17.9" pages="422" />
      <ReferenceSolution lab={l1} />
    </LabsPage>
  );
}
