import type { Question } from '../../components/Ask';
import { Beyond, Bench, Source, Today, type Tier } from '../../components/notes';
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
  benchRow,
  defineLabs,
  measure,
  msCol,
  preferredBench,
  rawCol,
  useLabs,
  whose,
  type BenchSet,
  type Col,
} from '../../components/LabKit';
import { fmt, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';

// The page is built from components/LabKit.tsx; what is here is chapter 23's own content.
// No question on this page: the lab is the exercise. The three predictions are stored, not
// graded, under ch23/labs/p-staged, ch23/labs/p-pageable and ch23/labs/p-metal (the ids the
// lab's README names). The bench rows read here (parts, firsttouch, serial, staged, inflight)
// are the contract with labs/ch23/lab1_overlap/main.cpp and with the stepper on page 23.3.
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch23', [{ num: '23.1', dir: 'lab1_overlap', name: 'hide the copies', vi: { name: 'giấu các lần sao chép' } }]);
const [l1] = LABS;
const RUN = 'python3 labs/pmpp.py run 23.1';

/** Identifier colours in the reference solution: the device arrays and buffers, and the kernels' registers. */
const TIERS: Record<string, Tier> = { d_in: 'global', d_out: 'global', in: 'global', out: 'global', x: 'reg', y: 'reg' };

// ---------------------------------------------------------------- reading the benchmark rows
const secs = (r: BenchRow | undefined) => num(r?.seconds);
const ms = (v: number | undefined) => (v == null ? '?' : `${fmt(v * 1000)} ms`); // i18n-ignore: number and unit
const row = benchRow;
/** The provided baseline, for one kind of host memory: pinned, pageable (CUDA, HIP) or shared (Metal). */
export const serialRow = (s: BenchSet | undefined, host: string) => row(s, 'serial', { host });
/** Your runStaged at one setting. */
export const stagedRow = (s: BenchSet | undefined, streams: number, chunks: number, order: string, host: string) =>
  row(s, 'staged', { streams, chunks, order, host });
/** Your runInFlight at one setting. */
export const inflightRow = (s: BenchSet | undefined, buffers: number, chunks: number) => row(s, 'inflight', { buffers, chunks });

/** Prediction 1 and 2: serial time over staged time at 4 streams, 16 chunks, depth first, for one kind of host memory. */
export const stagedSpeedup = (host: 'pinned' | 'pageable') => (reports: Reports) =>
  measure(reports, l1, (s) => {
    const a = secs(serialRow(s, host));
    const b = secs(stagedRow(s, 4, 16, 'depth', host));
    if (!a || !b) return null;
    const engines = num(row(s, 'parts', {})?.async_engines);
    const note =
      engines == null
        ? ''
        : s.backend === 'hip'
          ? tr(`; asyncEngineCount ${engines}, a constant on ROCm`, `; asyncEngineCount ${engines}, một hằng số trên ROCm`)
          : `; asyncEngineCount ${engines}`; // i18n-ignore: API name and number
    return {
      value: a / b,
      detail: tr(
        `${ms(a)} serial and ${ms(b)} staged, ${host} host memory${note}`,
        `${ms(a)} tuần tự và ${ms(b)} chia giai đoạn, bộ nhớ host ${host === 'pinned' ? 'ghim' : 'phân trang'}${note}`,
      ),
    };
  });

/** Prediction 3: in-flight time with 1 buffer over in-flight time with 3, at 16 chunks (Metal). */
export const inflightRatio = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const one = inflightRow(s, 1, 16);
    const three = inflightRow(s, 3, 16);
    const a = secs(one);
    const b = secs(three);
    if (!a || !b) return null;
    const serial = secs(serialRow(s, 'shared'));
    return {
      value: a / b,
      detail: tr(
        `${ms(a)} with 1 buffer and ${ms(b)} with 3${serial ? `; the serial path took ${ms(serial)}` : ''}`,
        `${ms(a)} với 1 buffer và ${ms(b)} với 3${serial ? `; đường tuần tự mất ${ms(serial)}` : ''}`,
      ),
    };
  });

/** The fourth guess, not recorded: breadth-first time over depth-first time, pinned, 16 chunks, per stream count. */
export const ORDER_STREAMS = [1, 2, 3, 4, 8] as const;
export const orderRatio = (s: BenchSet, streams: number) => {
  const d = secs(stagedRow(s, streams, 16, 'depth', 'pinned'));
  const b = secs(stagedRow(s, streams, 16, 'breadth', 'pinned'));
  return d && b ? { depth: d, breadth: b, ratio: b / d } : null;
};

// ---------------------------------------------------------------- what the benchmark measured first
function Parts() {
  const { reports, loading, reload } = useLabs();
  const sets = preferredBench(reports, l1).filter((s) => row(s, 'parts', {}));
  return (
    <Bench
      title={tr('The three parts, each alone', 'Ba phần, đo riêng từng phần')}
      task={
        <button className="quiet" onClick={reload}>
          <Translated vi="Tải lại kết quả">Reload results</Translated>
        </button>
      }
      foot={tr(
        "From this machine's recorded lab runs: each part over the whole array, timed before any pipeline runs. On Metal the kernel is the GPU time of the serial path's one command buffer; the fill and the drain are CPU time.",
        'Từ các lần chạy lab đã ghi trên máy này: mỗi phần chạy trên toàn bộ mảng, được đo trước khi bất kỳ pipeline nào chạy. Trên Metal, kernel là thời gian GPU của command buffer duy nhất trong đường tuần tự; phần nạp (fill) và phần rút (drain) là thời gian CPU.',
      )}
    >
      {sets.length ? (
        <div className="panel" style={{ overflowX: 'auto' }}>
          <table className="data" style={{ margin: 0 }} data-table="parts">
            <thead>
              <tr>
                <th>
                  <Translated vi="Backend">Backend</Translated>
                </th>
                <th className="num">
                  <Translated vi="Mảng, MiB">Array, MiB</Translated>
                </th>
                <th className="num">
                  <Translated vi="Bậc">Degree</Translated>
                </th>
                <th className="num">
                  <Translated vi="Chép vào hoặc nạp, ms">Copy in or fill, ms</Translated>
                </th>
                <th className="num">
                  <Translated vi="Kernel, ms">Kernel, ms</Translated>
                </th>
                <th className="num">
                  <Translated vi="Chép ra hoặc rút, ms">Copy out or drain, ms</Translated>
                </th>
                <th className="num">asyncEngineCount</th>
              </tr>
            </thead>
            <tbody>
              {sets.map((s) => {
                const p = row(s, 'parts', {})!;
                const metal = p.fill_s !== undefined;
                const engines = num(p.async_engines);
                const v = (k: string) => (num(p[k]) == null ? '' : fmt(num(p[k])! * 1000));
                return (
                  <tr key={s.backend} data-parts={s.backend}>
                    <td>
                      {s.backend} <span className="muted">({s.device}, {s.variant === 'work' ? tr('yours', 'của bạn') : tr('reference solution', 'lời giải tham chiếu')})</span>
                    </td>
                    <td className="num">{num(p.bytes) ? fmt(num(p.bytes)! / 2 ** 20) : ''}</td>
                    <td className="num">{String(p.degree ?? '')}</td>
                    <td className="num" data-col="in">{v(metal ? 'fill_s' : 'h2d_s')}</td>
                    <td className="num" data-col="kernel">{v('kernel_s')}</td>
                    <td className="num" data-col="out">{v(metal ? 'drain_s' : 'd2h_s')}</td>
                    <td className="num" data-col="engines">
                      {metal ? <span className="muted">
                          <Translated vi="không có copy engine">no copy engine</Translated>
                        </span> : engines == null || engines < 0 ? '' : engines}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted" style={{ margin: 0 }} data-parts-empty>
          {loading ? (
            tr('Reading reports.', 'Đang đọc báo cáo.')
          ) : (
            <Translated
              vi={
                <>
                  Máy này chưa có benchmark nào của lab này. Hãy chạy <code>{RUN} --bench</code> trên một máy có GPU (thêm{' '}
                  <code>--solution</code> để đo lời giải tham chiếu), rồi tải lại. Trình giả lập không bao giờ được benchmark.
                </>
              }
            >
              <>
                No benchmark of this lab on this machine yet. Run <code>{RUN} --bench</code> on a machine with a GPU (add{' '}
                <code>--solution</code> to time the reference), then reload. The emulator is never benchmarked.
              </>
            </Translated>
          )}
        </p>
      )}
      {sets.map((s) => {
        const f = row(s, 'firsttouch', {});
        const fresh = num(f?.fresh_fill_s);
        const used = num(f?.fill_s);
        return f && fresh && used ? (
          <Translated
            key={s.backend}
            vi={
              <p style={{ margin: '0.6rem 0 0' }} data-firsttouch={s.backend}>
                Lần chạm đầu trên {s.device}: tạo một shared buffer cỡ bằng mảng mất {ms(num(f.create_s))}; nạp nó lần đầu mất{' '}
                {ms(fresh)}, so với {ms(used)} cho một buffer đã dùng rồi, lâu gấp {fmt(fresh / used)} lần. Đó là lý do{' '}
                <code>slotBuffer</code> giữ các slot của bạn từ lần gọi này sang lần gọi sau.
              </p>
            }
          >
            <p key={s.backend} style={{ margin: '0.6rem 0 0' }} data-firsttouch={s.backend}>
              First touch on {s.device}: making a shared buffer of the array&rsquo;s size took {ms(num(f.create_s))}; filling
              it the first time took {ms(fresh)}, against {ms(used)} for a buffer already used, {fmt(fresh / used)} times as
              long. That is why <code>slotBuffer</code> keeps your slots from one call to the next.
            </p>
          </Translated>
        ) : null;
      })}
    </Bench>
  );
}

// ---------------------------------------------------------------- the results table
const PIPE_ROWS = new Set(['serial', 'staged', 'inflight']);

function OverlapResults() {
  const { reports } = useLabs();
  // The serial time each row is compared with: same backend, same kernel, same host memory.
  const base = new Map<string, number>();
  for (const b of Object.keys(reports))
    for (const x of reports[b].results.filter((x) => x.lab === l1.id))
      for (const r of x.bench.filter((r) => r.name === 'serial')) {
        const t = num(r.seconds);
        if (t) base.set(`${b}/${x.variant}/${String(r.host)}`, t);
      }
  const hostOf = (r: BenchRow) => (r.name === 'inflight' ? 'shared' : String(r.host ?? ''));
  const metal = Object.keys(reports).some((b) => reports[b].results.some((x) => x.lab === l1.id && x.bench.some((r) => r.name === 'inflight')));
  const cols: Col[] = [
    { head: tr('Row', 'Hàng'), cell: (r) => r.name },
    { head: tr('Streams or buffers', 'Stream hoặc buffer'), num: true, cell: (r) => String(r.streams ?? r.buffers ?? '') },
    rawCol(tr('Chunks', 'Khúc'), 'chunks'),
    { head: tr('Order', 'Thứ tự'), cell: (r) => String(r.order ?? '') },
    { head: tr('Host memory', 'Bộ nhớ host'), cell: hostOf },
    { ...msCol, head: tr('Whole call, ms', 'Cả lần gọi, ms') },
    {
      head: tr('Serial / this', 'Tuần tự / hàng này'),
      num: true,
      cell: (r) => {
        const t = num(r.seconds);
        const s = base.get(`${String(r.backend)}/${String(r.variant)}/${hostOf(r)}`);
        return t && s ? fmt(s / t) : '';
      },
    },
    ...(metal
      ? [
          {
            head: tr('GPU, ms', 'GPU, ms'),
            num: true,
            cell: (r: BenchRow) => (num(r.gpu_s) == null ? '' : fmt(num(r.gpu_s)! * 1000)),
          },
          {
            head: tr('Overlap seen', 'Thấy chồng lấp'),
            cell: (r: BenchRow) =>
              r.overlap_observed === undefined ? '' : (r.overlap_observed as unknown) === true ? tr('yes', 'có') : tr('no', 'không'),
          },
        ]
      : []),
  ];
  return <LabResults lab={l1} filter={(r) => PIPE_ROWS.has(r.name)} cols={cols} />;
}

// ---------------------------------------------------------------- the fourth guess: issue order
function OrderTable() {
  const { reports, loading } = useLabs();
  if (loading) return null;
  const sets = preferredBench(reports, l1).filter((s) => ORDER_STREAMS.some((k) => orderRatio(s, k)));
  if (!sets.length)
    return (
      <Translated
        vi={
          <p className="muted" data-order="none">
            Dự đoán bạn không ghi lại sẽ được trả lời ở đây khi lab 23.1 có các hàng <code>staged</code> từ CUDA hoặc HIP. Metal
            không có thứ tự phát lệnh nào để so sánh: pipeline của nó không có lần sao chép nào. Hãy chạy <code>{RUN} --bench</code>{' '}
            trên một máy có CUDA hoặc ROCm.
          </p>
        }
      >
        <p className="muted" data-order="none">
          The guess you did not record is answered here once lab 23.1 has <code>staged</code> rows from CUDA or HIP. Metal has
          no issue order to compare: its pipeline has no copies. Run <code>{RUN} --bench</code> on a machine with CUDA or
          ROCm.
        </p>
      </Translated>
    );
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      <table className="data" data-order="">
        <thead>
          <tr>
            <th>
              <Translated vi="Backend">Backend</Translated>
            </th>
            <th className="num">
              <Translated vi="Stream">Streams</Translated>
            </th>
            <th className="num">
              <Translated vi="Theo chiều sâu, ms">Depth first, ms</Translated>
            </th>
            <th className="num">
              <Translated vi="Theo chiều rộng, ms">Breadth first, ms</Translated>
            </th>
            <th className="num">
              <Translated vi="Rộng / sâu">Breadth / depth</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          {sets.flatMap((s) =>
            ORDER_STREAMS.flatMap((k) => {
              const o = orderRatio(s, k);
              return o
                ? [
                    <tr key={s.backend + k} data-backend={s.backend} data-streams={k}>
                      <td>
                        {s.backend} <span className="muted">({s.device}, {whose(s)})</span>
                      </td>
                      <td className="num">{k}</td>
                      <td className="num">{fmt(o.depth * 1000)}</td>
                      <td className="num">{fmt(o.breadth * 1000)}</td>
                      <td className="num" data-col="ratio">{fmt(o.ratio)}</td>
                    </tr>,
                  ]
                : [];
            }),
          )}
        </tbody>
      </table>
    </div>
  );
}

const ONLY_CUDA = (
  <Translated
    vi={
      <>
        Tỉ số này chỉ có trên CUDA và HIP: trên Apple silicon không có lần sao chép nào để chia giai đoạn, nên lần chạy Metal không có
        hàng <code>staged</code> nào. Hãy chạy <code>{RUN} --bench</code> trên một máy có CUDA hoặc ROCm, khi{' '}
        <code>runStaged</code> của bạn đã qua (hoặc với <code>--solution</code> để đo lời giải tham chiếu).
      </>
    }
  >
    <>
      This ratio exists only on CUDA and HIP: on Apple silicon there is no copy to stage, so the Metal run has no{' '}
      <code>staged</code> rows. Run <code>{RUN} --bench</code> on a machine with CUDA or ROCm, once your{' '}
      <code>runStaged</code> passes (or with <code>--solution</code> to time the reference).
    </>
  </Translated>
);

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={TIERS}>
      <LabsIntro>
        <Translated
          vi={
            <>
              Một lab, trong <code>labs/ch23/lab1_overlap/</code>. Lab này không bắt buộc, vì chương này được rút gọn, và nó không
              thêm gì vào dự án xuyên suốt. Nó biến vòng lặp chia giai đoạn (staged) của trang <a href={href('ch23', '23.3')}>23.3</a> thành mã
              bạn chạy trên mọi backend, và benchmark của nó đặt lại câu hỏi của trang đó trên chính GPU của bạn: sự chồng lấp
              giấu được bao nhiêu phần việc chép?
            </>
          }
        >
          One lab, in <code>labs/ch23/lab1_overlap/</code>. It is optional, as this chapter is condensed, and it adds nothing
          to the through-line project. It turns the staged loop of page <a href={href('ch23', '23.3')}>23.3</a> into code you
          run on every backend, and its benchmark asks that page&rsquo;s question on your own GPU: how much of the copying
          does the overlap hide?
        </Translated>
      </LabsIntro>
      <RunningLabs
        example="23.1"
        alsoShownIn={
          <Translated
            vi={
              <>
                trình chạy từng bước của <a href={href('ch23', '23.3')}>mục 23.3</a>, dưới dạng thiết lập sẵn &ldquo;thiết bị của
                bạn&rdquo;
              </>
            }
          >
            <>
              the stepper of <a href={href('ch23', '23.3')}>section 23.3</a>, as its &ldquo;your device&rdquo; preset
            </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <Translated
        vi={
          <p>
            Có hai điều quan trọng riêng với lab này. Trên trình giả lập (emulator), mọi lệnh gọi stream đều đồng bộ: một lần
            chép hay một lần khởi chạy đã xong khi lệnh gọi trả về. Vì thế trình giả lập kiểm tra mỗi chunk tính ra gì và
            kết quả nằm ở đâu, chứ không bao giờ kiểm tra có gì chồng lấp hay không. Và lab này không được build cho{' '}
            <code>race</code>, biến thể chạy các luồng đồng thời dưới ThreadSanitizer, vì không luồng nào ghi vào thứ mà một
            luồng khác chạm tới.
          </p>
        }
      >
        <p>
          Two things matter for this lab in particular. On the emulator every stream call is synchronous: a copy or a launch
          has finished by the time the call returns. So the emulator checks what each chunk computes and where it lands,
          never whether anything overlapped. And the lab is not built for <code>race</code>, the flavour that runs threads
          concurrently under ThreadSanitizer, because no thread writes anything another thread touches.
        </p>
      </Translated>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 23.1 */}
      <h2>
        <Translated vi="Lab 23.1: giấu các lần sao chép sau kernel">Lab 23.1: hide the copies behind the kernel</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Vòng khứ hồi của chương 2 làm ba việc nối tiếp nhau: chép đầu vào lên device, chạy kernel, chép đầu ra về. Trong lúc
            đường truyền chép, các lõi (core) chờ; trong lúc các lõi tính, đường truyền chờ. Cách chữa của mục 23.3, áp dụng
            cho một GPU: cắt công việc thành các chunk và để lần sao chép của một chunk chạy trong khi kernel của một chunk khác chạy.
            Bạn viết pipeline đó, rồi đo xem nó mang lại được gì.
          </p>
        }
      >
        <p>
          Chapter 2&rsquo;s round trip does three things one after another: copy the input to the device, run the kernel, copy
          the output back. While the link copies, the cores wait; while the cores compute, the link waits. Section 23.3&rsquo;s
          remedy, applied to one GPU: cut the work into chunks and let one chunk&rsquo;s copy run while another chunk&rsquo;s
          kernel does. You write that pipeline, and then you measure what it buys.
        </p>
      </Translated>
      <Source
        section="23.3"
        pages="553–563"
        figs={tr(
          'streams p. 556, cudaMemcpyAsync p. 560, pinned memory pp. 562–563',
          'stream tr. 556, cudaMemcpyAsync tr. 560, bộ nhớ ghim tr. 562–563',
        )}
      />
      <Translated
        vi={
          <p>
            Kernel đã được cung cấp. <code>polyKernel</code> tính một đa thức bậc <code>degree</code> tại mọi phần tử theo quy
            tắc Horner, mỗi bậc một phép nhân-cộng. Nó đại diện cho một lớp theo từng phần tử mà bạn chỉnh được chi phí, và
            benchmark chỉnh nó cho tới khi kernel mất khoảng bằng thời gian chép đầu vào lên, là chỗ sự chồng lấp có ý nghĩa
            nhất.
          </p>
        }
      >
        <p>
          The kernel is provided. <code>polyKernel</code> evaluates a polynomial of degree <code>degree</code> at every element
          by Horner&rsquo;s rule, one multiply-add per degree. It stands in for an elementwise layer whose cost you can dial,
          and the benchmark dials it until the kernel takes about as long as copying the input in, which is where overlap
          matters most.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Kernel, pipeline chia chunk, phiên bản Metal và mọi phép đo là của khóa học này. Mẫu chia giai đoạn là của NVIDIA (Best
              Practices Guide 10.1.2, ch23-facts A7); mẫu Metal là của Apple (C5).
            </p>
          }
        >
          <p>
            The kernel, the chunked pipeline, the Metal version and every measurement are ours. The staged pattern is
            NVIDIA&rsquo;s (Best Practices Guide 10.1.2, ch23-facts A7); the Metal pattern is Apple&rsquo;s (C5).
          </p>
        </Translated>
      </Beyond>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={
                <>
                  <code>runStaged(job, streams, chunks, order)</code>, bên dưới dòng <code>your code below</code>.{' '}
                  <code>polyKernel</code>, <code>launchPoly</code> và đường cơ sở <code>runSerial</code> phía trên đã được cung cấp
                </>
              }
            >
              <>
                <code>runStaged(job, streams, chunks, order)</code>, below the line <code>your code below</code>.{' '}
                <code>polyKernel</code>, <code>launchPoly</code> and the baseline <code>runSerial</code> above it are provided
              </>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated
              vi={
                <>
                  <code>runInFlight(job, buffers, chunks)</code>. <code>encodePoly</code>, <code>waitFor</code>,{' '}
                  <code>slotBuffer</code> và đường cơ sở <code>runSerialMetal</code> đã được cung cấp, và{' '}
                  <code>kernel.metal</code> đã hoàn chỉnh
                </>
              }
            >
              <>
                <code>runInFlight(job, buffers, chunks)</code>. <code>encodePoly</code>, <code>waitFor</code>,{' '}
                <code>slotBuffer</code> and the baseline <code>runSerialMetal</code> are provided, and{' '}
                <code>kernel.metal</code> is complete
              </>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Chỉ đọc: <code>overlap.h</code> (hai job, <code>Order</code> và mọi khai báo), <code>main.cpp</code> (chương trình
            điều khiển (driver), tham chiếu CPU, bước hiệu chỉnh và benchmark) và <code>probe.cu</code> (các công cụ đo cho
            CUDA và HIP: số copy engine, cấp phát bộ nhớ ghim, các phần được đo thời gian). Driver sở hữu các mảng trên device
            và các buffer trên host, và quyết định các buffer host được ghim (pinned) hay phân trang (pageable).
          </p>
        }
      >
        <p>
          Read-only: <code>overlap.h</code> (the two jobs, <code>Order</code> and every declaration), <code>main.cpp</code> (the
          driver, the CPU reference, the calibration and the benchmark) and <code>probe.cu</code> (the CUDA and HIP
          instruments: the copy-engine count, pinned allocation, the timed parts). The driver owns the device arrays and the
          host buffers, and decides whether the host buffers are pinned or pageable.
        </p>
      </Translated>

      <h3>
        <Translated vi="Bạn viết gì">What you write</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Một quy tắc chia chunk cho cả hai backend: chunk k là <code>[k*n/chunks, (k+1)*n/chunks)</code> theo phép chia số
            nguyên, với tích tính bằng 64 bit. Bỏ qua các chunk rỗng, xuất hiện khi <code>n</code> nhỏ hơn <code>chunks</code>, và
            đánh số các chunk bạn thực sự đưa vào hàng đợi là j = 0, 1, 2, cứ thế tiếp.
          </p>
        }
      >
        <p>
          One chunk rule for both backends: chunk k is <code>[k*n/chunks, (k+1)*n/chunks)</code> in integer division, with the
          product in 64 bits. Skip the empty ones, which appear when <code>n</code> is smaller than <code>chunks</code>, and
          number the chunks you do queue j = 0, 1, 2, and so on.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>CUDA, HIP và trình giả lập.</b> Tạo <code>streams</code> stream bằng{' '}
            <code>cudaStreamCreateWithFlags(&amp;s, cudaStreamNonBlocking)</code>. Đưa chunk j vào stream{' '}
            <code>j mod streams</code>: một <code>cudaMemcpyAsync</code> chép phần của nó trong <code>hostIn</code> vào{' '}
            <code>d_in</code>, <code>launchPoly</code> từ <code>d_in</code> vào <code>d_out</code>, và một{' '}
            <code>cudaMemcpyAsync</code> chép phần của nó trong <code>d_out</code> vào <code>hostOut</code>, tất cả ở cùng một
            offset. <code>Order::Depth</code> đưa vào hàng đợi từng chunk một; <code>Order::Breadth</code> đưa vào mọi lần sao chép
            vào, rồi mọi kernel, rồi mọi lần sao chép ra. Sau đó đồng bộ hóa mọi stream, hủy mỗi stream đúng một lần, và trả về 0,
            hoặc in lỗi ra và trả về 1.
          </p>
        }
      >
        <p>
          <b>CUDA, HIP and the emulator.</b> Create <code>streams</code> streams with{' '}
          <code>cudaStreamCreateWithFlags(&amp;s, cudaStreamNonBlocking)</code>. Queue chunk j in stream{' '}
          <code>j mod streams</code>: a <code>cudaMemcpyAsync</code> of its part of <code>hostIn</code> into <code>d_in</code>,{' '}
          <code>launchPoly</code> from <code>d_in</code> into <code>d_out</code>, and a <code>cudaMemcpyAsync</code> of its part
          of <code>d_out</code> into <code>hostOut</code>, all at the same offset. <code>Order::Depth</code> queues chunk by
          chunk; <code>Order::Breadth</code> queues every copy in, then every kernel, then every copy out. Then synchronise
          every stream, destroy each one once, and return 0, or print the error and return 1.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>Metal.</b> Apple silicon không có gì để chép, nên việc cần giấu là việc của CPU: nạp đầu vào của một chunk vào một
            shared buffer và rút đầu ra của nó. Lấy K = <code>buffers</code> slot, mỗi slot là một cặp shared buffer từ{' '}
            <code>slotBuffer</code>, đủ dài cho chunk lớn nhất, và một dispatch semaphore khởi đầu bằng K. Với chunk j: chờ trên
            semaphore; nếu slot <code>j mod K</code> vẫn còn giữ một chunk trước đó, <code>waitFor</code> command buffer của nó và
            chép đầu ra của nó sang <code>hostOut</code>; nạp slot từ <code>hostIn</code>; encode bằng <code>encodePoly</code>{' '}
            trong một command buffer riêng; thêm một completion handler chỉ làm mỗi việc báo hiệu semaphore; commit. Sau chunk
            cuối, chờ các slot còn đang chạy, cũ nhất trước, và rút chúng.
          </p>
        }
      >
        <p>
          <b>Metal.</b> Apple silicon has nothing to copy, so the work to hide is the CPU&rsquo;s: filling a chunk&rsquo;s input
          into a shared buffer and draining its output. Take K = <code>buffers</code> slots, each a pair of shared buffers from{' '}
          <code>slotBuffer</code>, long enough for the largest chunk, and a dispatch semaphore that starts at K. For chunk j:
          wait on the semaphore; if slot <code>j mod K</code> still holds an earlier chunk, <code>waitFor</code> its command
          buffer and copy its output to <code>hostOut</code>; fill the slot from <code>hostIn</code>; encode with{' '}
          <code>encodePoly</code> in a command buffer of its own; add a completion handler that only signals the semaphore;
          commit. After the last chunk, wait for the slots still in flight, oldest first, and drain them.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Hãy biết trước điều sẽ thấy trước khi đo. Một luồng CPU làm cả việc nạp lẫn việc rút, cái này sau cái kia, nên một
            máy Mac có hai làn chứ không phải ba: làn của CPU, nạp + rút, và làn của GPU. Chế độ xem Metal của trình chạy từng bước
            cho cận dưới: một lần chạy mất ít nhất n · max(nạp + rút, kernel) cho mỗi chunk. Trừ khi kernel dài, làn của CPU là
            làn dài hơn, vì nó di chuyển dữ liệu hai lần. Driver đặt kernel vào khoảng nạp + rút, trường hợp tốt nhất, và ngay
            cả khi đó sự chồng lấp cũng chỉ có thể giảm thời gian tuần tự đi tối đa một nửa. Chi phí của từng chunk, các lần chờ
            một slot rảnh, và việc CPU và GPU dùng chung một bộ nhớ đều ăn bớt phần đó; ăn bớt bao nhiêu là điều các hàng của bạn
            đo được.
          </p>
        }
      >
        <p>
          Know what to expect before you measure. One CPU thread does both the fill and the drain, one after the other, so a
          Mac has two lanes, not three: the CPU&rsquo;s, fill + drain, and the GPU&rsquo;s. The stepper&rsquo;s Metal view
          gives the bound: a run takes at least n · max(fill + drain, kernel) per chunk. Unless the kernel is long, the
          CPU&rsquo;s lane is the longer one, since it moves the data twice. The driver sets the kernel to about fill + drain,
          the best case, and even then overlap can at most halve the serial time. Per-chunk costs, the waits for a free slot
          and the CPU and GPU sharing one memory all eat into that; how much is what your rows measure.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch23-facts.md C4, C5; infra-facts.md C12)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Apple: &ldquo;finalize all CPU writes to one buffer instance before you commit a command buffer that references
              it.&rdquo; (hãy hoàn tất mọi lần ghi của CPU vào một buffer trước khi commit một command buffer tham chiếu tới nó).
              Các tệp facts ghi lại khi nào một completion handler chạy, chứ không ghi nó chạy trên luồng nào, nên việc rút vẫn
              nằm trên luồng của bạn.
            </p>
          }
        >
          <p>
            Apple: &ldquo;finalize all CPU writes to one buffer instance before you commit a command buffer that references
            it.&rdquo; The facts files record when a completion handler runs, not on which thread, so the drain stays on
            yours.
          </p>
        </Translated>
      </Today>

      <h3>
        <Translated vi="Lab khác sách ở đâu">Where the lab differs from the book</Translated>
      </h3>
      <Translated
        vi={
          <p>
            <b>Stream không chặn (non-blocking).</b> Sách tạo các stream bằng <code>cudaStreamCreate</code>, hoặc bằng{' '}
            <code>cudaStreamDefault</code> kèm một độ ưu tiên. Đó là các stream chặn (blocking): mọi thứ được đưa vào stream mặc
            định kiểu cũ (legacy default stream) đều chờ chúng và bắt chúng chờ. Đường cơ sở và các phép đo thời gian của driver
            dùng stream mặc định, nên các stream của lab này được tạo ở dạng không chặn.
          </p>
        }
      >
        <p>
          <b>Non-blocking streams.</b> The book creates its streams with <code>cudaStreamCreate</code>, or with{' '}
          <code>cudaStreamDefault</code> and a priority. Those are blocking streams: anything queued in the legacy default
          stream waits for them and makes them wait. The baseline and the driver&rsquo;s timing probes use the default stream,
          so this lab&rsquo;s streams are created non-blocking.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch23-facts.md A3, A9)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Stream mặc định kiểu cũ &ldquo;synchronizes with all other streams in the same CUcontext except for non-blocking
              streams&rdquo; (đồng bộ với mọi stream khác trong cùng CUcontext, trừ các stream không chặn).
            </p>
          }
        >
          <p>
            The legacy default stream &ldquo;synchronizes with all other streams in the same CUcontext except for non-blocking
            streams&rdquo;.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            <b>Không độ ưu tiên, không event.</b> Ở đây chỉ có một loại công việc, nên không có gì cần được ưu tiên; một độ ưu
            tiên dù sao cũng không tác động tới các lần sao chép, và ROCm mặc định bỏ qua độ ưu tiên. Sách cần event vì ba kernel ở
            ba stream dùng chung một chuẩn (norm). Ở đây ba thao tác của một chunk nằm cùng một stream, và thứ tự của stream đó
            là tất cả sự sắp thứ tự chúng cần.
          </p>
        }
      >
        <p>
          <b>No priorities, no events.</b> There is one kind of work here, so nothing needs to be preferred; a priority would
          not touch the copies anyway, and ROCm ignores priorities by default. The book needs events because three kernels in
          three streams share one norm. Here a chunk&rsquo;s three operations stay in one stream, whose order is all the
          ordering they need.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>&ldquo;Bắt buộc&rdquo; bộ nhớ ghim.</b> Sách nói <code>cudaMemcpyAsync</code> bắt buộc bộ nhớ host được ghim. Từ
            bộ nhớ phân trang, lệnh gọi vẫn chạy, nhưng nó có thể chờ lần sao chép xong, và khi đó nó không chồng lấp được gì. Các
            bài kiểm tra chạy cả hai loại buffer; benchmark đo xem sự khác biệt tốn bao nhiêu.
          </p>
        }
      >
        <p>
          <b>&ldquo;Requires&rdquo; pinned memory.</b> The book says <code>cudaMemcpyAsync</code> requires pinned host memory.
          From pageable memory the call still works, but it may wait for the copy, and then it overlaps nothing. The tests run
          both kinds of buffer; the benchmark measures what the difference costs.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch23-facts.md A2, A6, B2, B4)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              NVIDIA: một lần sao chép như vậy &ldquo;will revert to a synchronous behavior which will not overlap with other
              work&rdquo; (sẽ quay về hành vi đồng bộ, không chồng lấp với công việc khác). Header của AMD: &ldquo;the memory
              copy will be performed synchronously&rdquo; (lần sao chép bộ nhớ sẽ được thực hiện đồng bộ).
            </p>
          }
        >
          <p>
            NVIDIA: such a copy &ldquo;will revert to a synchronous behavior which will not overlap with other work&rdquo;.
            AMD&rsquo;s header: &ldquo;the memory copy will be performed synchronously&rdquo;.
          </p>
        </Translated>
      </Today>

      <h3>
        <Translated vi="Các bài kiểm tra kiểm tra gì">What the tests check</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Chạy <code>{RUN} --backend emu</code> trước, rồi <code>{RUN}</code>. Các kích thước là 1, 7, 1,000, 4,099 và 65,536
            số float, và thiết bị thật thêm kích thước của benchmark. CUDA, HIP và trình giả lập chạy (streams, chunks) = (1,
            1), (1, 4), (2, 2), (3, 5), (4, 16) và (8, 3), cộng thêm (2, 16) và (3, 16) ở n = 7, nơi các chunk rỗng nằm xen giữa
            các chunk đầy; cả hai thứ tự; buffer host được ghim và phân trang, vốn là cùng một vùng cấp phát trên trình giả lập.
            Metal chạy K = 1 tới 4 với cùng các số chunk. Với mỗi trường hợp:
          </p>
        }
      >
        <p>
          Run <code>{RUN} --backend emu</code> first, then <code>{RUN}</code>. The sizes are 1, 7, 1,000, 4,099 and 65,536
          floats, and real devices add the benchmark&rsquo;s size. CUDA, HIP and the emulator run (streams, chunks) = (1, 1),
          (1, 4), (2, 2), (3, 5), (4, 16) and (8, 3), plus (2, 16) and (3, 16) at n = 7, where empty chunks sit between full
          ones; both orders; pinned and pageable host buffers, which are the same allocation on the emulator. Metal runs K = 1
          to 4 with the same chunk counts. For each case:
        </p>
      </Translated>
      <ol>
        <li>
          <Translated
            vi={
              <>
                <code>hostOut</code> khớp với phép tính Horner của CPU trong <code>main.cpp</code> trong một dung sai tương đối,
                và <code>hostIn</code> không đổi. <code>hostOut</code> khởi đầu là NaN, nên một chunk chưa bao giờ được chép về
                sẽ lộ ra.
              </>
            }
          >
            <code>hostOut</code> matches the CPU&rsquo;s Horner evaluation in <code>main.cpp</code> within a relative tolerance,
            and <code>hostIn</code> is unchanged. <code>hostOut</code> starts as NaN, so a chunk never copied back shows.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Họ CUDA, từ những gì <code>launchPoly</code> ghi lại: mọi phần tử được tính đúng một lần, mỗi chunk không rỗng
                một lần khởi chạy trên đúng ranh giới của quy tắc, không lần nào bị từ chối vì số lượng bằng 0, không lần nào
                nằm trong stream mặc định, <code>min(streams, non-empty chunks)</code> stream khác nhau, và các chunk liền kề
                nằm ở các stream khác nhau.
              </>
            }
          >
            CUDA family, from what <code>launchPoly</code> recorded: every element computed exactly once, one launch per
            non-empty chunk on the rule&rsquo;s boundaries, none refused for a count of 0, none in the default stream,{' '}
            <code>min(streams, non-empty chunks)</code> distinct streams, and neighbouring chunks in different streams.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Metal, từ những gì <code>encodePoly</code> ghi lại: mỗi chunk không rỗng một dispatch, nhiều nhất K buffer đầu
                vào, các chunk liền kề nằm ở các buffer khác nhau, mỗi command buffer một <code>waitFor</code>, và chunk j được
                encode sau nhiều nhất <code>max(0, j − K + 1)</code> lần chờ. Kiểm tra cuối cùng đó là điều làm nó thành một
                pipeline.
              </>
            }
          >
            Metal, from what <code>encodePoly</code> recorded: one dispatch per non-empty chunk, at most K input buffers,
            neighbouring chunks in different ones, one <code>waitFor</code> per command buffer, and chunk j encoded after at
            most <code>max(0, j − K + 1)</code> waits. That last check is what makes it a pipeline.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Chỉ trình giả lập: AddressSanitizer chặn một chunk chạy vượt quá một mảng, kèm dòng mã nguồn của nó, và một
                stream bị hủy hai lần, kèm cả hai dòng. Một stream không bao giờ được hủy làm trường hợp đó thất bại qua số
                byte heap còn sống, vốn phải bằng nhau trước và sau lệnh gọi.
              </>
            }
          >
            Emulator only: AddressSanitizer stops a chunk that runs past an array, with its source line, and a stream destroyed
            twice, with both lines. A stream never destroyed fails the case through the count of live heap bytes, which must
            be the same after the call as before it.
          </Translated>
        </li>
      </ol>
      <Translated
        vi={
          <p>
            Như khi giao, <code>runStaged</code> và <code>runInFlight</code> trả về −1, nên mọi trường hợp đều thất bại. Những gì
            các bài kiểm tra không thấy được cũng quan trọng không kém:
          </p>
        }
      >
        <p>
          As shipped, <code>runStaged</code> and <code>runInFlight</code> return −1, so every case fails. What the tests cannot
          see matters as much:
        </p>
      </Translated>
      <ul>
        <li>
          <Translated
            vi={
              <>
                <b>Mọi thứ về thời gian trên trình giả lập.</b> Thiếu <code>cudaStreamSynchronize</code> trước khi trả về, một{' '}
                <code>cudaDeviceSynchronize</code> sau mỗi chunk, và buffer phân trang đều qua được ở đó. Trên GPU, cái đầu là một
                race condition (tranh chấp dữ liệu) có kết quả phụ thuộc vào thời điểm; hai cái còn lại chỉ lộ ra trong
                benchmark.
              </>
            }
          >
            <b>Anything about time on the emulator.</b> A missing <code>cudaStreamSynchronize</code> before returning, a{' '}
            <code>cudaDeviceSynchronize</code> after every chunk, and pageable buffers all pass there. On a GPU the first is a
            race whose outcome depends on timing; the other two show only in the benchmark.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Stream của các lần sao chép.</b> Chỉ các lần khởi chạy được ghi lại. Một lần sao chép được đưa vào một stream khác
                với kernel của chunk đó là một race condition trên GPU và vô hình trên trình giả lập.
              </>
            }
          >
            <b>The copies&rsquo; streams.</b> Only launches are recorded. A copy queued in a different stream from its
            chunk&rsquo;s kernel is a race on a GPU and invisible on the emulator.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Chiều sâu so với chiều rộng.</b> Cả hai thứ tự đưa vào cùng các thao tác. Benchmark có thể phân biệt được
                chúng.
              </>
            }
          >
            <b>Depth against breadth.</b> Both orders queue the same operations. The benchmark may tell them apart.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Semaphore.</b> Chờ trên chính command buffer của slot là đủ cho tính đúng, nên một{' '}
                <code>runInFlight</code> không có semaphore vẫn qua. Hãy giữ nó, và giữ cân bằng: đó là cơ chế kiểm soát nhịp
                độ mà Apple ghi trong tài liệu.
              </>
            }
          >
            <b>The semaphore.</b> Waiting on the slot&rsquo;s own command buffer is enough for correctness, so a{' '}
            <code>runInFlight</code> without one passes. Keep it, balanced: it is Apple&rsquo;s documented rate control.
          </Translated>
        </li>
      </ul>

      <h3>
        <Translated vi="Trước khi benchmark, hãy dự đoán">Before you benchmark, predict</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Dùng trình chạy từng bước ở trang <a href={href('ch23', '23.3')}>23.3</a>. Driver làm cho kernel mất khoảng bằng lần
            chép vào, nên hãy bắt đầu từ thiết lập sẵn &ldquo;các giai đoạn bằng nhau&rdquo; và đặt số chunk và số stream như
            bên dưới. Trên CUDA, hãy tra <code>asyncEngineCount</code> trước (<code>python3 labs/pmpp.py doctor</code> và các
            dòng mở đầu của benchmark in ra giá trị này, và bảng bên dưới các dự đoán này hiện nó khi bạn đã chạy) rồi đặt số
            copy engine của trình chạy từng bước cho khớp.
          </p>
        }
      >
        <p>
          Use the stepper on page <a href={href('ch23', '23.3')}>23.3</a>. The driver makes the kernel take about as long as the
          copy in, so start from its &ldquo;equal stages&rdquo; preset and set the chunks and streams below. On CUDA, look up{' '}
          <code>asyncEngineCount</code> first (<code>python3 labs/pmpp.py doctor</code> and the benchmark&rsquo;s opening
          lines print it, and the table under these predictions shows it once you have run) and set the stepper&rsquo;s
          copy engines to match.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch23-facts.md A7, B3)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Với một copy engine, một lần sao chép vào và một lần sao chép ra không thể chạy cùng lúc; với hai thì có thể. Trên ROCm,{' '}
              <code>asyncEngineCount</code> là một hằng số của runtime, giống nhau trên mọi thiết bị.
            </p>
          }
        >
          <p>
            With one copy engine a copy in and a copy out cannot run at the same time; with two they can. On ROCm{' '}
            <code>asyncEngineCount</code> is a runtime constant, the same on every device.
          </p>
        </Translated>
      </Today>
      <LabPrediction
        id="ch23/labs/p-staged"
        prompt={
          <Translated
            vi={
              <>
                CUDA hoặc HIP: thời gian tuần tự chia cho thời gian chia giai đoạn ở 4 stream, 16 chunk, theo chiều sâu, bộ nhớ host
                được ghim. (Trên 1 nghĩa là pipeline thắng.)
              </>
            }
          >
            <>
              CUDA or HIP: the serial time divided by the staged time at 4 streams, 16 chunks, depth first, pinned host
              memory. (Above 1 means the pipeline won.)
            </>
          </Translated>
        }
        unit={tr('× faster than serial', '× nhanh hơn tuần tự')}
        measure={stagedSpeedup('pinned')}
        missing={ONLY_CUDA}
        reading={
          <Translated
            vi={
              <>
                Thiết lập sẵn &ldquo;thiết bị của bạn&rdquo; của trình chạy từng bước đưa các phần đo được này vào mô hình, để
                bạn đặt mức tăng tốc của mô hình cạnh con số này. Thời gian đo được gồm cả việc tạo các stream và chi phí cố
                định của từng chunk, những thứ mà mô hình bỏ qua trừ khi bạn cho nó một chi phí cho mỗi thao tác.
              </>
            }
          >
            <>
              The stepper&rsquo;s &ldquo;your device&rdquo; preset puts these measured parts into the model, so you can set the
              model&rsquo;s speed-up beside this one. The measured time includes creating the streams and every chunk&rsquo;s
              fixed costs, which the model leaves out unless you give it a per-operation cost.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch23/labs/p-pageable"
        prompt={
          <Translated
            vi={
              <>
                Cùng tỉ số đó với buffer host phân trang: thời gian tuần tự chia cho thời gian chia giai đoạn, cả hai từ bộ nhớ kiểu{' '}
                <code>malloc</code>.
              </>
            }
          >
            <>
              The same ratio with pageable host buffers: serial time over staged time, both from <code>malloc</code>-style
              memory.
            </>
          </Translated>
        }
        unit={tr('× faster than serial', '× nhanh hơn tuần tự')}
        measure={stagedSpeedup('pageable')}
        missing={ONLY_CUDA}
        reading={
          <Translated
            vi={
              <>
                Đặt con số này cạnh tỉ số với bộ nhớ ghim ở trên. Từ bộ nhớ phân trang, mỗi lần sao chép bất đồng bộ có thể bắt host
                chờ, và khi đó công việc của chunk tiếp theo không thể được đưa vào hàng đợi cho tới khi lần sao chép xong. Công tắc
                pageable của trình chạy từng bước cho thấy lịch chạy kéo theo.
              </>
            }
          >
            <>
              Set this beside the pinned ratio above. From pageable memory each asynchronous copy may make the host wait, and
              then the next chunk&rsquo;s work cannot be queued until it has finished. The stepper&rsquo;s pageable switch shows
              the schedule that follows.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch23/labs/p-metal"
        prompt={
          <Translated
            vi={
              <>
                Metal: thời gian in-flight với 1 buffer chia cho thời gian in-flight với 3 buffer, ở 16 chunk. Dùng chế độ xem
                Metal của trình chạy từng bước, nơi S trở thành K.
              </>
            }
          >
            <>
              Metal: the in-flight time with 1 buffer divided by the in-flight time with 3 buffers, at 16 chunks. Use the
              stepper&rsquo;s Metal view, where S becomes K.
            </>
          </Translated>
        }
        unit={tr('× faster with 3 buffers', '× nhanh hơn với 3 buffer')}
        measure={inflightRatio}
        missing={
          <Translated
            vi={
              <>
                Tỉ số này chỉ có trên Metal. Hãy chạy <code>{RUN} --bench</code> trên một máy Mac, khi{' '}
                <code>runInFlight</code> của bạn đã qua (hoặc với <code>--solution</code> để đo lời giải tham chiếu).
              </>
            }
          >
            <>
              This ratio exists only on Metal. Run <code>{RUN} --bench</code> on a Mac, once your <code>runInFlight</code>{' '}
              passes (or with <code>--solution</code> to time the reference).
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                Với một buffer, CPU chờ từng command buffer rồi mới nạp được chunk tiếp theo, nên việc nạp, các kernel và việc rút
                lần lượt thay phiên nhau. Với ba, CPU nạp một chunk trong khi GPU tính một chunk khác. Cột &ldquo;Thấy chồng
                lấp&rdquo; bên dưới cho biết điều đó có xảy ra ở từng hàng hay không. Sau đó so hàng 3 buffer với hàng tuần tự:
                khi làn của CPU, nạp + rút, dài bằng kernel, sự chồng lấp chỉ có thể giảm thời gian tuần tự đi tối đa một nửa, và
                bất cứ thứ gì làm dài thêm một trong hai làn đều lấy bớt từ đó.
              </>
            }
          >
            <>
              With one buffer the CPU waits for each command buffer before it can fill the next chunk, so fills, kernels and
              drains take turns. With three, the CPU fills one while the GPU computes another. The &ldquo;Overlap seen&rdquo;
              column below says whether that happened in each row. Then compare the 3-buffer row with the serial row: with
              the CPU&rsquo;s lane, fill + drain, as long as the kernel, overlap can at most halve the serial time, and
              anything that lengthens either lane takes from that.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Thêm một câu nữa, không có con số: trên GPU của bạn, theo chiều rộng có khác theo chiều sâu không? Trình chạy từng
            bước nói là có thể, với một copy engine, nếu engine đó phục vụ các lần sao chép theo thứ tự chúng được phát lệnh. NVIDIA
            không ghi trong tài liệu cả thứ tự đó lẫn cách các hàng đợi chép của họ thay đổi nó, nên câu này là câu hỏi dành cho
            thiết bị của bạn.
          </p>
        }
      >
        <p>
          One more, without a number: will breadth first differ from depth first on your GPU? The stepper says it might, with
          one copy engine, if that engine serves copies in the order they were issued. NVIDIA documents neither that order nor
          how its copy queues change it, so this one is a question for your device.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch23-facts.md A7, A8, E2)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Theo mặc định, CUDA ánh xạ các stream lên 8 hàng đợi công việc phần cứng; nhiều stream hoạt động hơn số hàng đợi
              có thể khiến các stream không liên quan phải chờ nhau. HIP mặc định dùng nhiều nhất 4 hàng đợi cho mỗi thiết bị.
            </p>
          }
        >
          <p>
            CUDA maps streams onto 8 hardware work queues by default; more active streams than queues can make unrelated
            streams wait for each other. HIP uses at most 4 per device by default.
          </p>
        </Translated>
      </Today>

      <h3>
        <Translated vi="Máy của bạn đo được gì">What your machine measured</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Benchmark trước hết đo riêng từng phần trên toàn bộ mảng. Nó chọn kích thước mảng từ dung lượng bộ nhớ mà thiết bị
            báo cáo: số float vừa trong một phần mười sáu của dung lượng đó, tối đa 64 Mi float mỗi mảng. Mức trần này là một
            tham số của lab, đặt ra để giữ cho lần chạy ngắn, không phải một sự thật về thiết bị nào. Sau đó nó hiệu chỉnh: nó
            đo kernel ở các bậc 1, 2, 4 và cứ thế, và giữ bậc có thời gian kernel gần với lần sao chép vào nhất. Trên Metal là gần
            với nạp + rút nhất, toàn bộ làn của CPU, và kernel được đo bên trong đường tuần tự được cung cấp, đúng như hàng
            tuần tự được đo.
          </p>
        }
      >
        <p>
          The benchmark first times each part alone over the whole array. It picks the array size from the memory the device
          reports: as many floats as fit in a sixteenth of it, up to 64 Mi floats per array. The cap is a parameter of this lab,
          there to keep a run short, not a fact about any device. Then it calibrates: it times the kernel at degrees 1, 2, 4
          and so on, and keeps the degree whose kernel time is closest to the copy in. On Metal it is closest to fill +
          drain, the CPU&rsquo;s whole lane, and the kernel is timed inside the provided serial path, exactly as the serial
          row is.
        </p>
      </Translated>
      <Parts />
      <Translated
        vi={
          <p>
            Mỗi thời gian bên dưới là một lần gọi trọn vẹn, đã gồm việc tạo stream, là kết quả tốt nhất trong vài lần chạy sau
            một lần khởi động không đo giờ mà đầu ra được kiểm tra. &ldquo;Tuần tự / hàng này&rdquo; chia thời gian của đường cơ
            sở, với cùng loại bộ nhớ host, cho thời gian của hàng đó. Trên Metal, &ldquo;GPU&rdquo; cộng thời gian GPU của các
            command buffer mà lệnh gọi đã chờ, và &ldquo;Thấy chồng lấp&rdquo; cho biết CPU có encode một chunk trong khi GPU
            đang chạy command buffer của một chunk khác hay không. Cột cuối đó là một quan sát trên máy của bạn: Apple ghi trong
            tài liệu cái mẫu, chứ không ghi kết quả.
          </p>
        }
      >
        <p>
          Every time below is one whole call, stream creation included, the best of several after an untimed warm-up whose
          output is checked. &ldquo;Serial / this&rdquo; divides the baseline&rsquo;s time, with the same host memory, by the
          row&rsquo;s. On Metal, &ldquo;GPU&rdquo; adds up the GPU time of the command buffers the call waited for, and
          &ldquo;Overlap seen&rdquo; says whether the CPU encoded a chunk while the GPU was running another chunk&rsquo;s
          command buffer. That last column is an observation of your machine: Apple documents the pattern, not the outcome.
        </p>
      </Translated>
      <OverlapResults />
      <Translated
        vi={
          <p>
            Ở các bậc thấp, thời gian hiệu chỉnh của kernel có thể đứng yên, vì việc di chuyển mảng, chứ không phải phép tính số
            học, quyết định nó: tỉ số giữa số phép toán và số byte của chương 5. Nếu nó không bao giờ tăng, hoặc một bậc thấp
            mất lâu hơn một bậc cao, thì đã có thứ khác dùng GPU, hoặc trên Mac là CPU, trong lúc chạy. Hãy chạy lại trước khi so
            sánh bất cứ gì, và chạy hai lần trước khi tin vào một khác biệt nhỏ giữa hai hàng.
          </p>
        }
      >
        <p>
          At low degrees the kernel&rsquo;s calibration time may stay flat, because moving the array, not the arithmetic,
          sets it: chapter 5&rsquo;s ratio of operations to bytes. If it never grows, or a low degree takes longer than a
          high one, something else was using the GPU, or on a Mac the CPU, during the run. Run again before you compare
          anything, and run twice before you trust a small difference between two rows.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Dự đoán bạn không ghi lại, từ cùng các hàng đó: thời gian theo chiều rộng chia cho thời gian theo chiều sâu, bộ nhớ
            ghim, 16 chunk.
          </p>
        }
      >
        <p>The guess you did not record, from the same rows: breadth-first time over depth-first time, pinned, 16 chunks.</p>
      </Translated>
      <OrderTable />
      <Translated
        vi={
          <p>
            Khi bạn đã chạy, trình chạy từng bước ở trang <a href={href('ch23', '23.3')}>23.3</a> cung cấp các phần đo được trên
            thiết bị của bạn dưới dạng một thiết lập sẵn và vẽ các hàng chia giai đoạn cạnh mô hình của nó.
          </p>
        }
      >
        <p>
          Once you have run, the stepper on page <a href={href('ch23', '23.3')}>23.3</a> offers your device&rsquo;s measured
          parts as a preset and plots the staged rows beside its model.
        </p>
      </Translated>
      <ReferenceSolution lab={l1} />

      <h3>
        <Translated vi="Điều này nằm ở đâu bên dưới những gì bạn đã biết">Where this sits under what you know</Translated>
      </h3>
      <Translated
        vi={
          <p>
            <code>tensor.to(&quot;cuda&quot;, non_blocking=True)</code> của PyTorch là một <code>cudaMemcpyAsync</code>, và từ
            bộ nhớ phân trang, theo chính ghi chú của PyTorch, &ldquo;the call may block the host&rdquo; (lệnh gọi có thể chặn
            host). Đó là hàng pageable của benchmark này. Một data loader với <code>pin_memory=True</code> đưa cho bạn các tensor
            đã nằm sẵn trong bộ nhớ ghim, nên lần sao chép có thể chồng lấp với các kernel của batch trước.
          </p>
        }
      >
        <p>
          PyTorch&rsquo;s <code>tensor.to(&quot;cuda&quot;, non_blocking=True)</code> is a <code>cudaMemcpyAsync</code>, and from
          pageable memory, PyTorch&rsquo;s own notes say, &ldquo;the call may block the host&rdquo;. That is the pageable row of
          this benchmark. A data loader with <code>pin_memory=True</code> hands you tensors already in pinned memory, so the copy
          can overlap the previous batch&rsquo;s kernels.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md D4; appC-facts.md D1)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Kernel chép của PyTorch gọi <code>cudaMemcpyAsync</code> khi <code>non_blocking</code> được bật; data loader của
              nó chép các tensor vào bộ nhớ ghim khi <code>pin_memory</code> là true.
            </p>
          }
        >
          <p>
            PyTorch&rsquo;s copy kernel calls <code>cudaMemcpyAsync</code> when <code>non_blocking</code> is set; its data
            loader copies tensors into pinned memory when <code>pin_memory</code> is true.
          </p>
        </Translated>
      </Today>

      <h3>
        <Translated vi="Đi xa hơn">Go further</Translated>
      </h3>
      <ul>
        <li>
          <Translated
            vi={
              <>
                Cho <code>runStaged</code> một thứ tự thứ ba: chép vào chunk j + 1 trước khi khởi chạy chunk j. Trên GPU của bạn,
                nó rơi vào đâu giữa theo chiều sâu và theo chiều rộng?
              </>
            }
          >
            Give <code>runStaged</code> a third order: copy in chunk j + 1 before launching chunk j. Where does it land between
            depth and breadth on your GPU?
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Đo lại các hàng 8 stream với <code>CUDA_DEVICE_MAX_CONNECTIONS=4</code> và với <code>=16</code>, và tìm các phụ
                thuộc giả mà trang 23.3 mô tả.
              </>
            }
          >
            Time the 8-stream rows again with <code>CUDA_DEVICE_MAX_CONNECTIONS=4</code> and with <code>=16</code>, and look for
            the false dependencies page 23.3 describes.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Trên Metal, chuyển việc rút vào completion handler, và nói bạn cần biết gì về luồng chạy handler trước khi tin
                vào nó.
              </>
            }
          >
            On Metal, move the drain into the completion handler, and say what you would need to know about the
            handler&rsquo;s thread before you trusted it.
          </Translated>
        </li>
      </ul>
      <Translated
        vi={
          <p>
            Lab này không phải một mốc dự án của dự án khóa học, vốn chạy trên một GPU và không chia một mô hình ra nhiều thiết
            bị. Phần <a href={href('ch23', 'wrap')}>tóm tắt</a> có bài kiểm tra tổng hợp của chương, nơi mỗi bài tập của sách
            được giải, và những gì đã thay đổi từ khi sách được in.
          </p>
        }
      >
        <p>
          The lab is not a milestone of the course project, which runs on one GPU and does not split a model across devices.
          The <a href={href('ch23', 'wrap')}>summary</a> has the chapter&rsquo;s mixed check, where each of the book&rsquo;s
          exercises is worked, and what has changed since the book was printed.
        </p>
      </Translated>
    </LabsPage>
  );
}
