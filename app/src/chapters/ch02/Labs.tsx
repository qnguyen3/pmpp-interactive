import type { Question } from '../../components/Ask';
import { Predict } from '../../components/Ask';
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
  benchRow,
  defineLabs,
  measure,
  numCol,
  rawCol,
  useLabs,
  type BenchSet,
  type Col,
} from '../../components/LabKit';
import { Translated, tr } from '../../lib/i18n';
import { fmt, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 2's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch02', [
  { num: '2.1', dir: 'lab1_vecadd', name: 'vector addition', vi: { name: 'cộng vectơ' } },
  { num: '2.2', dir: 'lab2_timing', name: 'where the time goes', vi: { name: 'thời gian đi đâu' } },
  { num: '2.3', dir: 'lab3_elementwise', name: 'elementwise ops (M0)', vi: { name: 'phép toán theo từng phần tử (M0)' } },
  { num: '2.4', dir: 'lab4_bughunt', name: 'bug hunt', vi: { name: 'săn lỗi' } },
]);
const [l1, l2, l3, l4] = LABS;

// Identifier colours for the reference solutions: arrays on the device are global memory,
// a thread's own index is a register. Host arrays (the _h names) have no tier.
const tiers = (globals: string[], regs: string[]): Record<string, Tier> => ({
  ...Object.fromEntries(globals.map((n) => [n, 'global' as Tier])),
  ...Object.fromEntries(regs.map((n) => [n, 'reg' as Tier])),
});
const VEC_TIERS = tiers(['A', 'B', 'C', 'A_d', 'B_d', 'C_d'], ['i']);
const M0_TIERS = tiers(['X', 'Y', 'R', 'bias', 'x_d', 'y_d', 'r_d', 'bias_d', 'X_d', 'Y_d', 'R_d'], ['i']);
const HUNT_TIERS = tiers(['x', 'x_d'], ['i']);

// ---------------------------------------------------------------- reading the benchmark rows
const ms = (s: number) => `${fmt(s * 1000)} ms`; // i18n-ignore: number and unit
/** A column of seconds shown in milliseconds; blank when the row lacks the key. */
const msOf = (head: string, key: string): Col => ({
  head,
  num: true,
  cell: (r) => {
    const v = num(r[key]);
    return v == null ? '' : fmt(v * 1000);
  },
});
/** The row called `name` with the largest n (and, when given, the sync setting). */
const largest = (s: BenchSet, name: string, sync?: number): BenchRow | undefined =>
  s.rows
    .filter((r) => r.name === name && (sync === undefined || r.sync === undefined || num(r.sync) === sync))
    .reduce<BenchRow | undefined>((best, r) => (best && (num(best.n) ?? 0) >= (num(r.n) ?? 0) ? best : r), undefined);
/** The host-array row a prediction compares against: on CUDA and HIP the one with the synchronise. */
const hostArrays = (s: BenchSet) => largest(s, 'phases', 1);

/** Lab 2.1: CPU loop time over one whole vecAdd call, at the largest size measured. */
const cpuOverOffload = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const r = largest(s, 'end_to_end');
    const v = num(r?.speedup_vs_cpu);
    const cpu = num(r?.cpu_s);
    const off = num(r?.offload_s);
    if (!r || v == null || !cpu || !off) return null;
    return {
      value: v,
      detail: tr(
        `${ms(cpu)} for the CPU loop, ${ms(off)} for one whole call of vecAdd, at n = ${String(r.n)}`,
        `${ms(cpu)} cho vòng lặp CPU, ${ms(off)} cho trọn một lời gọi vecAdd, ở n = ${String(r.n)}`,
      ),
    };
  });

/** Lab 2.2: the kernel's share of the host-array call. CUDA family: the kernel phase; Metal: the GPU time the command buffer reports. */
const kernelShare = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const r = hostArrays(s);
    const total = num(r?.total_s);
    if (!r || !total) return null;
    const kernel = num(r.kernel_s);
    if (kernel != null)
      return {
        value: (100 * kernel) / total,
        detail: tr(
          `kernel phase ${ms(kernel)} of ${ms(total)} in all, with the synchronise, at n = ${String(r.n)}`,
          `giai đoạn kernel ${ms(kernel)} trên tổng ${ms(total)}, có đồng bộ hóa, ở n = ${String(r.n)}`,
        ),
      };
    const gpu = num(r.gpu_s);
    if (gpu == null) return null;
    return {
      value: (100 * gpu) / total,
      detail: tr(
        `${ms(gpu)} of GPU time reported by the command buffer, of ${ms(total)} in all, at n = ${String(r.n)}`,
        `${ms(gpu)} thời gian GPU do command buffer báo, trên tổng ${ms(total)}, ở n = ${String(r.n)}`,
      ),
    };
  });

/** Lab 2.2: the host-array call's total over the resident path's time, at the largest size. */
const hostOverResident = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const host = hostArrays(s);
    const total = num(host?.total_s);
    const res = host ? benchRow(s, 'resident', { n: num(host.n) ?? -1 }) : undefined;
    const t = num(res?.seconds);
    if (!host || !total || !t) return null;
    return {
      value: total / t,
      detail: tr(
        `${ms(total)} with host arrays in and out, ${ms(t)} on data already on the device, at n = ${String(host.n)}`,
        `${ms(total)} với mảng host vào và ra, ${ms(t)} trên dữ liệu đã có sẵn trên device, ở n = ${String(host.n)}`,
      ),
    };
  });

/** Lab 2.2, CUDA and HIP only: the kernel phase without the synchronise, as a percent of the one with it. */
const nosyncShare = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const r = largest(s, 'phases', 0);
    const v = num(r?.nosync_kernel_pct);
    if (!r || v == null || num(r.sync) !== 0) return null;
    return {
      value: v,
      detail: tr(
        `kernel phase ${ms(num(r.kernel_s) ?? NaN)} without the synchronise, at n = ${String(r.n)}`,
        `giai đoạn kernel ${ms(num(r.kernel_s) ?? NaN)} không có đồng bộ hóa, ở n = ${String(r.n)}`,
      ),
    };
  });

/** Lab 2.3: three round trips over one resident chain. */
const chainRatio = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const r = benchRow(s, 'chain');
    const v = num(r?.ratio);
    const res = num(r?.resident_s);
    const rt = num(r?.roundtrips_s);
    if (!r || v == null || !res || !rt) return null;
    return {
      value: v,
      detail: tr(
        `${ms(rt)} for three round trips, ${ms(res)} for one blockTail call, at ${String(r.rows)} × ${String(r.d)}`,
        `${ms(rt)} cho ba lượt đi về, ${ms(res)} cho một lời gọi blockTail, ở ${String(r.rows)} × ${String(r.d)}`,
      ),
    };
  });

// ---------------------------------------------------------------- lab 2.2: the columns depend on the backend
const PATH: Col = {
  get head() {
    return tr('Path', 'Đường đi');
  },
  cell: (r) =>
    r.name === 'phases'
      ? r.sync === undefined
        ? tr('host arrays in and out', 'mảng host vào và ra')
        : num(r.sync) === 1
          ? tr('host arrays, sync after launch', 'mảng host, đồng bộ sau khởi chạy')
          : tr('host arrays, no sync after launch', 'mảng host, không đồng bộ sau khởi chạy')
      : r.name === 'resident'
        ? tr('data already on the device', 'dữ liệu đã có sẵn trên device')
        : r.name === 'cpu'
          ? tr('CPU loop', 'vòng lặp CPU')
          : r.name,
};
// Built when the table renders, so the headings follow the selected language.
const CUDA_PHASES: () => Col[] = () => [
  msOf(tr('Allocate, ms', 'Cấp phát, ms'), 'alloc_s'),
  msOf(tr('Copy in, ms', 'Sao chép vào, ms'), 'h2d_s'),
  msOf(tr('Kernel, ms', 'Kernel, ms'), 'kernel_s'),
  msOf(tr('Copy back, ms', 'Sao chép về, ms'), 'd2h_s'),
  msOf(tr('Free, ms', 'Giải phóng, ms'), 'free_s'),
  numCol(tr('Kernel, % of total', 'Kernel, % tổng'), 'kernel_share_pct'),
  numCol(tr('Kernel phase, % of the synchronised one', 'Giai đoạn kernel, % so với bản có đồng bộ'), 'nosync_kernel_pct'),
];
const METAL_PHASES: () => Col[] = () => [
  msOf(tr('Buffers and copy in, ms', 'Buffer và sao chép vào, ms'), 'wrap_s'),
  msOf(tr('Encode and commit, ms', 'Encode và commit, ms'), 'encode_s'),
  msOf(tr('Wait, ms', 'Chờ, ms'), 'wait_s'),
  msOf(tr('Copy out, ms', 'Sao chép ra, ms'), 'readback_s'),
  msOf(tr('GPU time, ms', 'Thời gian GPU, ms'), 'gpu_s'),
];
const TOTAL: Col = {
  get head() {
    return tr('Total, ms', 'Tổng, ms');
  },
  num: true,
  cell: (r) => {
    const v = num(r.name === 'phases' ? r.total_s : r.seconds);
    return v == null ? '' : fmt(v * 1000);
  },
};

function Lab2Results() {
  const { reports } = useLabs();
  const rows = Object.keys(reports).flatMap((b) => reports[b].results.filter((x) => x.lab === l2.id).flatMap((x) => x.bench));
  const cuda = rows.some((r) => r.alloc_s !== undefined);
  const metal = rows.some((r) => r.wrap_s !== undefined);
  return (
    <LabResults
      lab={l2}
      filter={(r) => r.name === 'phases' || r.name === 'resident' || r.name === 'cpu'}
      cols={[rawCol('n', 'n'), PATH, ...(cuda ? CUDA_PHASES() : []), ...(metal ? METAL_PHASES() : []), TOTAL]}
    />
  );
}

// ---------------------------------------------------------------- lab 2.4: what reported itself
// Backends that are not a GPU: the emulator and its race-detection flavour.
const EMULATED = new Set(['emu', 'race']);
interface HuntRun {
  backend: string;
  device: string;
  date: string;
  untouched: boolean;
  total: number;
  failed: number;
  /** Failing tests in which scalePrefix itself returned a non-zero status. */
  returned: number;
  /** The run was stopped before its end: a crash, or AddressSanitizer on the emulator. */
  stopped: boolean;
}
/** Per backend, how the last run of the reader's work/ for lab 2.4 failed. */
function huntRuns(reports: Reports): HuntRun[] {
  return Object.keys(reports).flatMap((b) => {
    const run = reports[b];
    const rs = run.results.filter((x) => x.lab === l4.id && x.variant === 'work' && x.built && x.test_rc != null);
    if (!rs.length) return [];
    const tests = rs.flatMap((x) => x.tests);
    const bad = tests.filter((t) => !t.pass);
    return [{
      backend: b,
      device: String(run.device.name ?? b),
      date: run.generated.slice(0, 10),
      untouched: run.work_state?.[l4.id] === 'untouched',
      total: tests.length,
      failed: bad.length,
      returned: bad.filter((t) => /^scalePrefix returned -?[1-9]/.test(t.detail ?? '')).length,
      stopped: rs.some((x) => x.test_rc !== 0 && x.test_rc !== 1),
    }];
  });
}
/** Failures that announced themselves: a non-zero status, or a stopped run. */
const selfReported = (r: HuntRun) => r.returned + (r.stopped ? 1 : 0);

function HuntReveal() {
  const { reports, loading, reload } = useLabs();
  const runs = huntRuns(reports);
  return (
    <>
      {runs.length ? (
        runs.map((r) => (
          <Translated
            key={r.backend}
            vi={
              <p data-measured={selfReported(r)} data-backend={r.backend}>
                {EMULATED.has(r.backend) ? 'Trên trình giả lập' : `Trên ${r.device} (${r.backend})`}, trong lần chạy ngày {r.date},{' '}
                {r.untouched ? 'với work/ như khi phát hành' : 'sau khi work/ đã được sửa'}: {r.failed} trên {r.total} bài kiểm thử thất bại.{' '}
                <b>{selfReported(r)}</b> lỗi tự báo ra
                {r.returned ? `, ${r.returned} qua một giá trị trả về khác 0` : ''}
                {r.stopped ? `, ${EMULATED.has(r.backend) ? 'và lần chạy bị AddressSanitizer dừng lại' : 'và lần chạy dừng sớm'}` : ''}
                . {r.failed - r.returned > 0 ? `${r.failed - r.returned} lỗi là giá trị sai mà chỉ phép so sánh với bản tham chiếu CPU mới tìm ra.` : ''}
              </p>
            }
          >
            <p key={r.backend} data-measured={selfReported(r)} data-backend={r.backend}>
              {EMULATED.has(r.backend) ? 'On the emulator' : `On ${r.device} (${r.backend})`}, in the run of {r.date},{' '}
              {r.untouched ? 'with work/ as shipped' : 'after work/ had been edited'}: {r.failed} of {r.total} tests failed.{' '}
              <b>{selfReported(r)}</b>{' '}
              {selfReported(r) === 1 ? 'failure announced itself' : 'failures announced themselves'}
              {r.returned ? `, ${r.returned} through a non-zero return value` : ''}
              {r.stopped ? `, ${EMULATED.has(r.backend) ? 'and the run was stopped by AddressSanitizer' : 'and the run stopped early'}` : ''}
              . {r.failed - r.returned > 0 ? `${r.failed - r.returned} were wrong values that only the comparison with the CPU reference found.` : ''}
            </p>
          </Translated>
        ))
      ) : loading ? (
        <p data-measured="loading">
          <Translated vi="Đang đọc báo cáo.">Reading reports.</Translated>
        </p>
      ) : (
        <p data-measured="none">
          <Translated
            vi={
              <>
                Chưa có lần chạy nào cho bài làm của bạn ở lab này. Chạy <code>python3 labs/pmpp.py run 2.4 --backend emu</code>, rồi{' '}
                <code>python3 labs/pmpp.py run 2.4</code> trên một máy có GPU.
              </>
            }
          >
            No run of your work for this lab yet. Run <code>python3 labs/pmpp.py run 2.4 --backend emu</code>, then{' '}
            <code>python3 labs/pmpp.py run 2.4</code> on a machine with a GPU.
          </Translated>{' '}
          <button className="quiet" onClick={reload}>
            <Translated vi="Tải lại kết quả">Reload results</Translated>
          </button>
        </p>
      )}
      <Translated
        vi={
          <p className="muted">
            Một lần chạy kiểm thử đếm số bài kiểm thử thất bại, không đếm số lỗi, và một lỗi có thể làm hỏng nhiều bài kiểm thử hoặc che giấu một lỗi khác. Khi mọi bài kiểm thử của bạn đã qua, phần cuối của <code>README.md</code> của lab, "After the hunt", nói với mỗi lỗi tài liệu của từng backend hứa hẹn điều gì.
          </p>
        }
      >
        <p className="muted">
          A test run counts failing tests, not defects, and one defect can fail many tests or hide another. Once all
          your tests pass, the last section of the lab's <code>README.md</code>, "After the hunt", says for each defect
          what each backend's documentation promises.
        </p>
      </Translated>
    </>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <LabsIntro>
        <Translated
          vi={
            <>
              Bốn lab, mỗi lab trong một thư mục riêng dưới <code>labs/ch02/</code>. Lab đầu tiên là toàn bộ chương trình của chương, do bạn viết từ lần cấp phát đầu tiên đến lần giải phóng cuối cùng. Lab thứ hai đo thời gian của nó. Lab thứ ba xây mốc dự án đầu tiên của dự án khóa học, và lab thứ tư là một cuộc săn lỗi trong một hàm stub phía host.
            </>
          }
        >
          Four labs, each in its own directory under <code>labs/ch02/</code>. The first is the chapter's whole program,
          written by you from the first allocation to the last free. The second times it. The third builds the first
          milestone of the course project, and the fourth is a bug hunt in a host stub.
        </Translated>
      </LabsIntro>
      <RunningLabs
        example="2.1"
        alsoShownIn={
          <Translated
            vi={
              <>
                chuỗi được đo ở <a href={href('ch02', '2.3')}>mục 2.3</a> và thời gian các giai đoạn ở{' '}
                <a href={href('ch02', '2.4')}>mục 2.4</a>
              </>
            }
          >
            <>
              the measured chain of <a href={href('ch02', '2.3')}>section 2.3</a> and the phase times of{' '}
              <a href={href('ch02', '2.4')}>section 2.4</a>
            </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <Translated
        vi={
          <p>
            Không lab nào trong số này được build cho <code>race</code>, biến thể thứ hai của trình giả lập, chạy các luồng đồng thời dưới ThreadSanitizer. Không luồng nào trong chúng chạm vào một phần tử mà luồng khác ghi.
          </p>
        }
      >
        <p>
          None of these labs is built for <code>race</code>, the emulator's second flavour, which runs the threads
          concurrently under ThreadSanitizer. No thread in them touches an element that another thread writes.
        </p>
      </Translated>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 2.1 */}
      <h2>
        <Translated vi="Lab 2.1: cộng vectơ, từ đầu đến cuối">Lab 2.1: vector addition, end to end</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn viết toàn bộ chương trình của chương: kernel của Fig. 2.10 và hàm host của Fig. 2.13, mà chương trình điều khiển (driver) gọi với ba mảng host thông thường, theo cách <code>main</code> gọi nó trong Fig. 2.4. Cấp phát ba mảng trên device, sao chép A và B vào, tính số khối cho kích thước khối được giao, khởi chạy, kiểm tra lần khởi chạy, chờ, sao chép C về, giải phóng, và trả về 0 hoặc một trạng thái khác 0.
          </p>
        }
      >
        <p>
          You write the whole program of the chapter: the kernel of Fig. 2.10 and the host function of Fig. 2.13, which
          the driver calls with three ordinary host arrays, the way <code>main</code> calls it in Fig. 2.4. Allocate
          three device arrays, copy A and B in, compute the block count for the block size you are given, launch, check
          the launch, wait, copy C back, free, and return 0 or a non-zero status.
        </p>
      </Translated>
      <Source
        section="2.3–2.6"
        pages="26–39"
        figs={tr('Fig. 2.10, p. 35; Fig. 2.13, p. 38; the sidebar on p. 32', 'Hình 2.10, p. 35; Hình 2.13, p. 38; khung bên ở p. 32')}
      />
      <Translated
        vi={
          <p>
            Mọi lời gọi runtime đi qua một macro <code>CHECK</code> mà bạn viết một lần, từ phép kiểm tra lỗi ở p. 32. Đoạn mã đó không biên dịch được như khi in; <a href={href('ch02', '2.4')}>mục 2.4</a> yêu cầu bạn tìm ra lý do. Macro của bạn in chuỗi lỗi, tên tệp và số dòng, không giải phóng gì, và làm hàm trả về một giá trị khác 0. Nó không gọi <code>exit()</code>, vì chương trình điều khiển kiểm tra điều gì xảy ra sau một lần thất bại.
          </p>
        }
      >
        <p>
          Every runtime call goes through a <code>CHECK</code> macro that you write once, from the error check on p. 32.
          That snippet does not compile as printed; <a href={href('ch02', '2.4')}>section 2.4</a> has you find out why.
          Your macro prints the error string, the file and the line, frees nothing, and makes the function return a
          non-zero value. It does not call <code>exit()</code>, because the driver tests what happens after a failure.
        </p>
      </Translated>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <Translated vi={<><code>CHECK</code>, <code>vecAddKernel</code> và <code>vecAdd</code></>}>
              <><code>CHECK</code>, <code>vecAddKernel</code> and <code>vecAdd</code></>
            </Translated>,
          ],
          ['kernel.metal', <><code>vec_add</code></>],
          [
            'launch.mm',
            <Translated vi={<><code>vecAdd</code>: buffer, encoder, một lần dispatch đúng n luồng, commit, chờ, kiểm tra, sao chép ra. <code>makeVecAddPipeline</code> được cung cấp sẵn</>}>
              <><code>vecAdd</code>: buffers, encoder, a dispatch of exactly n threads, commit, wait, check, copy out. <code>makeVecAddPipeline</code> is provided</>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Trên Metal không có mã lỗi. Bạn kiểm tra mọi lần tạo xem có nil không, và trạng thái cùng lỗi của command buffer sau lệnh chờ. Các đầu vào đến buffer của chúng qua <code>newBufferWithBytes</code>, hàm này sao chép, vì các mảng của chương trình điều khiển là bộ nhớ heap thông thường và hàm bọc không sao chép cần bộ nhớ căn theo trang (<a href={href('ch02', '2.3')}>mục 2.3</a>).
          </p>
        }
      >
        <p>
          On Metal there are no error codes. You check every creation for nil, and the command buffer's status and
          error after the wait. The inputs reach their buffers through <code>newBufferWithBytes</code>, which copies,
          because the driver's arrays are ordinary heap memory and the copy-free wrapper needs page-aligned memory (
          <a href={href('ch02', '2.3')}>section 2.3</a>).
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Kernel và hàm host Metal là của khóa học này. Chương của sách chỉ có CUDA.</p>}>
          <p>The Metal kernel and host function are ours. The book's chapter is CUDA only.</p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 2.1</code>, bắt đầu với <code>--backend emu</code>. Các đầu vào nằm giữa 0.5 và 1.5, nên không tổng đúng nào gần 0, và <code>C_h</code> ban đầu được điền NaN. Các độ dài nằm ở cả hai phía của mỗi kích thước khối: 1, 31, 32, 33, 255, 256, 257, 1000 và 4097, và trên device còn có 2²⁰ + 3. Trên CUDA, HIP và trình giả lập, mỗi độ dài chạy với 32 và với 256 luồng mỗi khối, và một trường hợp dùng khối lớn nhất mà device báo. Thêm một trường hợp trên ba backend đó yêu cầu một khối lớn gấp đôi mức device cho phép: nó phải trả về khác 0 mà không kết thúc chương trình, và lời gọi hợp lệ thực hiện ngay sau đó phải trả về 0 với các tổng đúng.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 2.1</code>, starting with <code>--backend emu</code>. The inputs lie
          between 0.5 and 1.5, so no correct sum is near 0, and <code>C_h</code> starts filled with NaN. The lengths sit
          on both sides of each block size: 1, 31, 32, 33, 255, 256, 257, 1000 and 4097, and on a device also 2²⁰ + 3.
          On CUDA, HIP and the emulator each length runs with 32 and with 256 threads per block, and one case uses the
          largest block the device reports. One more case on those three asks for a block twice as large as the device
          allows: it must
          return non-zero without ending the program, and the valid call made right after it must return 0 with the
          right sums.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Hãy đọc một lần thất bại qua hình dạng của nó. NaN còn lại trong <code>C_h</code> nghĩa là không gì được sao chép về đó. Phần đuôi sai nghĩa là quá ít khối. Một báo cáo AddressSanitizer trên trình giả lập nghĩa là một lần truy cập ra ngoài một mảng, kèm dòng mã nguồn: thiếu phép chặn, kích thước tính theo phần tử trong khi cần tính theo byte, hoặc giải phóng trước khi sao chép về.
          </p>
        }
      >
        <p>
          Read a failure by its shape. NaN left in <code>C_h</code> means nothing was copied back there. A wrong tail
          means too few blocks. An AddressSanitizer report on the emulator means an access outside an array, with the
          source line: a missing guard, a size in elements where bytes were meant, or a free before the copy back.
        </p>
      </Translated>
      <LabPrediction
        id="ch02/labs/p-lab1-speedup"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy benchmark. Ở độ dài lớn nhất mà nó đạt tới, thời gian của vòng lặp CPU đơn giản của Fig. 2.4 chia cho thời gian của trọn một lời gọi <code>vecAdd</code> của bạn, kể cả cấp phát, sao chép và giải phóng, sẽ là bao nhiêu? Lớn hơn 1 nghĩa là bản GPU thắng.
              </>
            }
          >
            <>
            Before you run the benchmark. At the largest length it reaches, the time of the plain CPU loop of Fig. 2.4
            divided by the time of one whole call of your <code>vecAdd</code>, allocation, copies and free included,
            will be what? Above 1 means the GPU version wins.
            </>
          </Translated>
        }
        unit={tr('× as fast as the CPU loop', '× nhanh hơn vòng lặp CPU')}
        measure={cpuOverOffload}
        missing={
          <Translated
            vi={
              <>
                Chạy <code>python3 labs/pmpp.py run 2.1 --bench</code> trên một máy có GPU. Trình giả lập không bao giờ được đo hiệu năng.
              </>
            }
          >
            <>
            Run <code>python3 labs/pmpp.py run 2.1 --bench</code> on a machine with a GPU. The emulator is never
            benchmarked.
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                Kernel làm một phép cộng cho mỗi 12 byte nó di chuyển, và lời gọi còn sao chép từng byte đó sang device rồi quay về. Mục 2.3 đã đếm lưu lượng đó; con số này là cái giá của nó so với một luồng CPU trên máy của bạn. Lab 2.2 mổ xẻ lời gọi.
              </>
            }
          >
            <>
            The kernel does one addition for every 12 bytes it moves, and the call also copies every one of those
            bytes across to the device and back. Section 2.3 counted that traffic; this number is what it costs against
            one CPU thread on your machine. Lab 2.2 takes the call apart.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l1}
        filter={(r) => r.name === 'end_to_end'}
        cols={[
          rawCol('n', 'n'),
          msOf(tr('CPU loop, ms', 'Vòng lặp CPU, ms'), 'cpu_s'),
          msOf(tr('Your vecAdd, whole call, ms', 'vecAdd của bạn, trọn lời gọi, ms'), 'offload_s'),
          numCol(tr('CPU time over vecAdd time', 'Thời gian CPU trên thời gian vecAdd'), 'speedup_vs_cpu'),
        ]}
      />
      <ReferenceSolution lab={l1} tiers={VEC_TIERS} />

      {/* ------------------------------------------------------------ 2.2 */}
      <h2>
        <Translated vi="Lab 2.2: thời gian đi đâu?">Lab 2.2: where does the time go?</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách cảnh báo ở p. 28 rằng cách chuyển việc sang device này, sao chép vào, tính, sao chép ra, có thể rất kém hiệu quả, và rằng các chương trình thực tế giữ dữ liệu lớn trên device. Sách không đưa ra con số nào. Ở đây bạn đo chúng trên máy của chính mình. Mã khởi đầu (starter) chứa một <code>vecAdd</code> chạy được dưới một tên mới, là lời giải tham chiếu (solution) của lab 2.1 (dán bản của bạn vào nếu bạn thích), và bạn thêm các đồng hồ.
          </p>
        }
      >
        <p>
          The book warns on p. 28 that this way of moving work to the device, copy in, compute, copy out, can be very
          inefficient, and that real programs keep large data on the device. It gives no numbers. Here you take them on
          your own machine. The starter holds a working <code>vecAdd</code> under a new name, the reference solution of
          lab 2.1 (paste in your own if you prefer it), and you add the clocks.
        </p>
      </Translated>
      <Source section="2.3" pages="28" />
      <Translated
        vi={
          <p>
            Trên CUDA, HIP và trình giả lập, <code>vecAddTimed</code> điền năm thời gian giai đoạn: cấp phát, sao chép vào, kernel, sao chép về, giải phóng. Cờ <code>syncAfterLaunch</code> của nó quyết định giai đoạn kernel kết thúc sau một <code>cudaDeviceSynchronize()</code> có kiểm tra hay ngay sau câu lệnh khởi chạy, đó chính là thí nghiệm. Một <code>launchVecAdd</code> được cung cấp sẵn chạy cùng kernel trên các mảng mà chương trình điều khiển đã đặt sẵn trên device, và chương trình điều khiển đo thời gian nó với lần đồng bộ hóa riêng: đường cơ sở với dữ liệu nằm sẵn trên device. Trên Metal, <code>vecAddTimed</code> điền bốn giai đoạn phía host (buffer và sao chép vào, encode và commit, chờ, sao chép ra) cùng thời gian GPU mà command buffer báo, và bạn viết <code>vecAddResident</code> cho các buffer mà chương trình điều khiển đã tạo trong lưu trữ chia sẻ và điền qua <code>contents</code>.
          </p>
        }
      >
        <p>
          On CUDA, HIP and the emulator, <code>vecAddTimed</code> fills five phase times: allocation, copies in,
          kernel, copy back, free. Its flag <code>syncAfterLaunch</code> decides whether the kernel phase ends after a
          checked <code>cudaDeviceSynchronize()</code> or right after the launch statement, which is the experiment.
          A provided <code>launchVecAdd</code> runs the same kernel on arrays the driver already put on the device, and
          the driver times it with its own synchronise: the resident baseline. On Metal, <code>vecAddTimed</code> fills
          four host phases (buffers and copy in, encode and commit, wait, copy out) and the GPU time the command buffer
          reports, and you write <code>vecAddResident</code> for buffers the driver created in shared storage and filled
          through <code>contents</code>.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A3, A5, B1)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              Một lần khởi chạy trả về trước khi lưới của nó chạy, nên một bộ đếm thời gian chỉ bao quanh lần khởi chạy chỉ đo được lần khởi chạy.{' '}<code>cudaFree</code> “có thể thực hiện đồng bộ hóa ngầm”, và <code>hipFree</code> luôn đồng bộ hóa trước: một bộ đếm thời gian bao quanh một lần giải phóng có thể bao gồm cả thời gian chờ kernel.
            </p>
          }
        >
          <p>
            A launch returns before its grid has run, so a timer around the launch alone measures only the launch.{' '}
            <code>cudaFree</code> "may perform implicit synchronization", and <code>hipFree</code> always synchronises
            first: a timer around a free can include waiting for the kernel.
          </p>
        </Translated>
      </Today>
      <LabFiles
        lab={l2}
        files={[
          [
            'kernel.cu',
            <Translated vi={<><code>vecAddTimed</code>; kernel và <code>launchVecAdd</code> được cung cấp sẵn và chạy được</>}>
              <><code>vecAddTimed</code>; the kernel and <code>launchVecAdd</code> are provided and working</>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<><code>vec_add</code>, được cung cấp hoàn chỉnh</>}>
              <><code>vec_add</code>, provided complete</>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<><code>vecAddTimed</code> và <code>vecAddResident</code></>}>
              <><code>vecAddTimed</code> and <code>vecAddResident</code></>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Trên Metal không có <code>syncAfterLaunch</code>. Trên CUDA, lần sao chép về có chặn sẽ chờ kernel, nên bỏ lần đồng bộ hóa chỉ chuyển thời gian từ giai đoạn này sang giai đoạn khác. Trên Metal không gì chờ cho bạn, và đọc <code>contents</code> trước khi lệnh chờ trả về là một lỗi thứ tự, không phải một lựa chọn về đo thời gian.
          </p>
        }
      >
        <p>
          There is no <code>syncAfterLaunch</code> on Metal. On CUDA the blocking copy back waits for the kernel, so
          leaving the synchronise out only moves time from one phase to another. On Metal nothing waits for you, and
          reading <code>contents</code> before the wait has returned is an ordering error, not a timing choice.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 2.2</code>. Các bài kiểm thử kiểm tra các tổng với độ dài 1, 257, 1000 và 4097 (và 2²⁰ + 3 trên device) với cả hai giá trị của cờ, rồi đến thời gian các giai đoạn: mỗi giá trị đều được ghi, hữu hạn và không âm, và cộng lại không vượt quá đồng hồ riêng của chương trình điều khiển bao quanh lời gọi, với độ chênh 5% và 50 micro giây, và ít nhất bằng một nửa của nó. Chúng kiểm tra đường dữ liệu nằm sẵn trên device với các mảng mà chương trình điều khiển sở hữu. Lần đồng bộ hóa có đặt đúng chỗ hay không thì không phải là một bài kiểm thử. Trên trình giả lập mọi lời gọi đều đồng bộ, nên chỉ benchmark trên device mới cho thấy điều đó.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 2.2</code>. The tests check the sums for lengths 1, 257, 1000 and 4097 (and
          2²⁰ + 3 on a device) with both settings of the flag, then the phase times: each written, finite and not
          negative, and together no more than the driver's own clock around the call, with 5% and 50 microseconds of
          slack, and at least half of it. They check the resident path on arrays the driver owns. Whether a synchronise
          is in the right place is not a test. On the emulator every call is synchronous, so only the benchmark on a
          device can show it.
        </p>
      </Translated>
      <LabPrediction
        id="ch02/labs/p-lab2-share"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy benchmark. Ở độ dài lớn nhất, kernel chiếm bao nhiêu phần trăm của một lời gọi với mảng host? Trên CUDA và HIP đó là giai đoạn kernel có đồng bộ hóa của bạn; trên Metal là thời gian GPU mà command buffer báo.
              </>
            }
          >
            <>
            Before you run the benchmark. At the largest length, what percent of one host-array call is the kernel?
            On CUDA and HIP that is your kernel phase with the synchronise; on Metal, the GPU time the command buffer
            reports.
            </>
          </Translated>
        }
        unit={tr('percent of the call', 'phần trăm của lời gọi')}
        measure={kernelShare}
        missing={
          <Translated
            vi={
              <>
                Chạy <code>python3 labs/pmpp.py run 2.2 --bench</code> trên một máy có GPU.
              </>
            }
          >
            <>
              Run <code>python3 labs/pmpp.py run 2.2 --bench</code> on a machine with a GPU.
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                Phần còn lại của lời gọi là cấp phát, sao chép, chờ và giải phóng: phần mà sách gọi là cái giá của thuê ngoài trong suốt. Bảng có mọi giai đoạn, nên bạn thấy được giai đoạn nào chiếm ưu thế ở mỗi độ dài.
              </>
            }
          >
            <>
            The rest of the call is allocation, copies, waiting and freeing: the part the book calls the cost of
            transparent outsourcing. The table has every phase, so you can see which one dominates at each length.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch02/labs/p-lab2-resident"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy benchmark. Ở độ dài lớn nhất, thời gian của một lời gọi với mảng host chia cho thời gian của cùng phép cộng trên dữ liệu đã có sẵn trên device sẽ là bao nhiêu?
              </>
            }
          >
            <>
            Before you run the benchmark. At the largest length, the time of one host-array call divided by the time
            of the same addition on data already on the device will be what?
            </>
          </Translated>
        }
        unit={tr('× the resident time', '× thời gian khi dữ liệu nằm sẵn trên device')}
        measure={hostOverResident}
        missing={
          <Translated
            vi={
              <>
                Chạy <code>python3 labs/pmpp.py run 2.2 --bench</code> trên một máy có GPU.
              </>
            }
          >
            <>
              Run <code>python3 labs/pmpp.py run 2.2 --bench</code> on a machine with a GPU.
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                Cả hai hàng chạy cùng kernel trên cùng các độ dài. Thứ mà đường dữ liệu nằm sẵn bỏ qua là mọi thứ di chuyển dữ liệu giữa các mảng của bên gọi và device. Trên Metal nó hoàn toàn không sao chép gì: hàng của nó báo 0 byte được sao chép, một con số đếm điều mà đường mã đó làm, không phải một phép đo thời gian.
              </>
            }
          >
            <>
            Both rows run the same kernel on the same lengths. What the resident path leaves out is everything that
            moves data between the caller's arrays and the device. On Metal it copies nothing at all: its row reports
            0 bytes copied, a count of what that code path does, not a timing.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch02/labs/p-lab2-nosync"
        prompt={
          <Translated
            vi={
              <>
                CUDA và HIP. Khi bỏ lần đồng bộ hóa sau lần khởi chạy, giai đoạn kernel của bạn sẽ đo được bao nhiêu phần trăm so với giá trị có đồng bộ hóa? Hãy ghi lại cả giai đoạn nào bạn dự đoán sẽ tăng lên thay vào đó.
              </>
            }
          >
            <>
            CUDA and HIP. With the synchronise after the launch removed, your kernel phase will read what percent of
            its synchronised value? Write down too which phase you expect to grow in its place.
            </>
          </Translated>
        }
        unit={tr('percent of the synchronised kernel phase', 'phần trăm của giai đoạn kernel có đồng bộ hóa')}
        measure={nosyncShare}
        missing={
          <Translated
            vi={
              <>
                Phép đo này chỉ có trên CUDA và HIP. Metal không có lần đồng bộ hóa nào để bỏ, nên benchmark của nó không có hàng như vậy. Chạy <code>python3 labs/pmpp.py run 2.2 --bench</code> trên một máy có CUDA hoặc ROCm.
              </>
            }
          >
            <>
            This measurement exists only on CUDA and HIP. Metal has no synchronise to leave out, so its benchmark has
            no such row. Run <code>python3 labs/pmpp.py run 2.2 --bench</code> on a machine with CUDA or ROCm.
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                Không có lần đồng bộ hóa, đồng hồ dừng khi câu lệnh khởi chạy trả về. Kernel vẫn chạy, và thời gian của nó xuất hiện ở một giai đoạn sau có chờ nó. So sánh hai hàng trong bảng theo từng giai đoạn.
              </>
            }
          >
            <>
            Without the synchronise the clock stops when the launch statement returns. The kernel still runs, and its
            time turns up in a later phase that waits for it. Compare the two rows in the table phase by phase.
            </>
          </Translated>
        }
      />
      <Lab2Results />
      <ReferenceSolution lab={l2} tiers={VEC_TIERS} />

      {/* ------------------------------------------------------------ 2.3 */}
      <h2>
        <Translated vi="Lab 2.3: các phép toán theo từng phần tử mà một transformer cần">Lab 2.3: the elementwise operations a transformer needs</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn viết ba kernel, mỗi kernel có một hàm khởi chạy bên cạnh, và một hàm host nối chúng thành chuỗi trên device: một bias được cộng vào mọi hàng của một tensor <code>rows × d</code>, một phép nhân hệ số và dịch{' '}<code>alpha * x + beta</code>, và một residual được cộng tại chỗ. Mỗi kernel là phép cộng vectơ của Fig. 2.10 với một phần thân khác. Phép nhân hệ số đi qua <code>scaleShift</code>, một hàm phụ <code>__host__ __device__</code> mà trình biên dịch build hai lần, và một hàm host được cung cấp sẵn gọi bản build cho host để chương trình điều khiển có thể so sánh hai bản. Bạn đã gặp phép toán này: kernel của bài tập 9 của sách, mà{' '}<a href={href('ch02', '2.6')}>mục 2.6</a> đếm số luồng, tính <code>b[i] = 2.7f * a[i] - 4.3f</code>, một phép nhân hệ số và dịch với hai hằng số được viết cứng. Ở đây chúng trở thành đối số.
          </p>
        }
      >
        <p>
          You write three kernels, each with a launch function beside it, and a host function that chains them on the
          device: a bias added to every row of a <code>rows × d</code> tensor, a scale and shift{' '}
          <code>alpha * x + beta</code>, and a residual added in place. Each kernel is the vector addition of Fig. 2.10
          with another body. The scale goes through <code>scaleShift</code>, a <code>__host__ __device__</code> helper
          that the compiler builds twice, and a provided host function calls the host build so the driver can compare
          the two. You have met this operation already: the kernel of the book's exercise 9, whose threads{' '}
          <a href={href('ch02', '2.6')}>section 2.6</a> counts, computes <code>b[i] = 2.7f * a[i] - 4.3f</code>, a
          scale and shift with the two constants written in. Here they become arguments.
        </p>
      </Translated>
      <Source
        section="2.5"
        pages="35–36"
        figs={tr('Fig. 2.10, Fig. 2.11; the affine map is exercise 9, Fig. 2.15, p. 42', 'Hình 2.10, Hình 2.11; phép biến đổi affine là bài tập 9, Hình 2.15, p. 42')}
      />
      <Translated
        vi={
          <p>
            Đây là mốc dự án M0 của dự án khóa học, và nó tuân theo các quy ước mà mọi mốc dự án sau ghép nối vào: tensor <code>float</code> theo thứ tự hàng (row-major) với chiều batch đứng đầu, con trỏ device thô và kích thước tường minh làm đối số kernel, và một hàm <code>launch&lt;Name&gt;</code> bên cạnh mỗi kernel chỉ tính lưới và khởi chạy, không hơn. Việc chờ và đọc trạng thái lỗi là việc của bên gọi, để bên gọi có thể xếp hàng nhiều lần khởi chạy trên dữ liệu ở lại trên device. <code>blockTail</code> làm đúng điều đó:{' '}<code>y = alpha * (x + bias) + beta + r</code> với một lượt sao chép vào, ba lần khởi chạy liên tiếp, một lần kiểm tra và một lần chờ, và một lần sao chép ra.
          </p>
        }
      >
        <p>
          This is milestone M0 of the course project, and it follows the conventions every later milestone composes
          with: row-major <code>float</code> tensors with the batch first, raw device pointers and explicit sizes as
          kernel arguments, and a <code>launch&lt;Name&gt;</code> function beside each kernel that computes its grid
          and launches, nothing more. Waiting and reading the error state are the caller's job, so that a caller can
          queue several launches on data that stays on the device. <code>blockTail</code> does exactly that:{' '}
          <code>y = alpha * (x + bias) + beta + r</code> with one set of copies in, three launches back to back, one
          check and one wait, and one copy out.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Dự án xuyên suốt, <code>nanoinfer</code>, là của khóa học này, và việc dùng kernel của chương cho bias, nhân hệ số và residual cũng vậy. Sách chỉ áp dụng nó cho phép cộng vectơ, và chỉ dùng kernel của bài tập 9 để đếm luồng.
            </p>
          }
        >
          <p>
            The through-line project, <code>nanoinfer</code>, is ours, and so is using the chapter's kernel for bias,
            scale and residual. The book applies it to vector addition only, and uses the kernel of exercise 9 only to
            count threads.
          </p>
        </Translated>
      </Beyond>
      <LabFiles
        lab={l3}
        files={[
          [
            'kernel.cu',
            <Translated vi={<><code>scaleShift</code>, ba kernel, <code>launchAddBias</code>, <code>launchScaleShift</code>, <code>launchAddResidual</code> và <code>blockTail</code></>}>
              <><code>scaleShift</code>, the three kernels, <code>launchAddBias</code>, <code>launchScaleShift</code>, <code>launchAddResidual</code> and <code>blockTail</code></>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<>hàm phụ <code>scale_shift</code> và các kernel <code>add_bias</code>, <code>scale_shift_k</code>, <code>add_residual</code></>}>
              <>the helper <code>scale_shift</code> and the kernels <code>add_bias</code>, <code>scale_shift_k</code>, <code>add_residual</code></>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<><code>encodeAddBias</code>, <code>encodeScaleShift</code>, <code>encodeAddResidual</code>, và <code>blockTail</code> với cả ba lần dispatch trong một command buffer</>}>
              <><code>encodeAddBias</code>, <code>encodeScaleShift</code>, <code>encodeAddResidual</code>, and <code>blockTail</code> with all three dispatches in one command buffer</>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Trên Metal không có thứ tương ứng với bản build cho host của <code>scaleShift</code>: mọi hàm trong một tệp{' '}<code>.metal</code> đều là mã GPU, và Apple không mô tả cách nào biên dịch một định nghĩa cho cả hai phía (<a href={href('ch02', '2.7')}>mục 2.7</a>). Hàm phụ trong <code>kernel.metal</code> chỉ đóng vai trò của một hàm{' '}<code>__device__</code>.
          </p>
        }
      >
        <p>
          On Metal there is no counterpart of the host build of <code>scaleShift</code>: every function in a{' '}
          <code>.metal</code> file is GPU code, and Apple documents no way to compile one definition for both sides (
          <a href={href('ch02', '2.7')}>section 2.7</a>). The helper in <code>kernel.metal</code> plays the part of a{' '}
          <code>__device__</code> function only.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 2.3</code>. Mỗi hàm khởi chạy được kiểm thử riêng, trên các tensor mà chương trình điều khiển giữ trên device, với các hình dạng 1 × 1, 1 × 257, 3 × 100, 4 × 256 và 7 × 33, và 64 × 4099 trên device. Bias khác nhau giữa các cột và các hình dạng có ít hàng hơn cột, nên một bias được đánh chỉ số theo hàng sẽ cho giá trị sai. Đầu vào phải giữ nguyên, đầu ra bắt đầu là NaN, và trên device một khối giá trị đánh dấu nằm sau mỗi tensor và phải còn nguyên. Trên CUDA, HIP và trình giả lập, phép nhân hệ số của kernel còn được so sánh với <code>scaleShiftOnHost</code> từng phần tử một, trong một dung sai: một trình biên dịch có thể gộp phép nhân và phép cộng thành một phép toán chỉ làm tròn một lần, và hai bản build không nhất thiết thống nhất về điều đó.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 2.3</code>. Each launch function is tested alone, on tensors the driver
          keeps on the device, at shapes 1 × 1, 1 × 257, 3 × 100, 4 × 256 and 7 × 33, and 64 × 4099 on a device. The
          bias differs from column to column and the shapes have fewer rows than columns, so a bias indexed by the row
          gives wrong values. Inputs must come back unchanged, outputs start as NaN, and on a device a block of marker
          values sits behind every tensor and must survive. On CUDA, HIP and the emulator the kernel's scale is also
          compared with <code>scaleShiftOnHost</code> element by element, within a tolerance: a compiler may fuse the
          multiply and the add into one operation that rounds once, and the two builds need not agree on that.
        </p>
      </Translated>
      <LabPrediction
        id="ch02/labs/p-lab3-chain"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy benchmark. Ở kích thước nó chọn, thời gian của cùng ba phép toán làm thành ba lượt đi về riêng biệt, mỗi lượt có các lần sao chép vào và ra riêng, chia cho thời gian của một lời gọi{' '}<code>blockTail</code>, sẽ là bao nhiêu? Trước hết hãy đếm số lần sao chép cỡ một tensor mà mỗi phiên bản thực hiện: ở đây bias là một hàng và hệ số là hai con số.
              </>
            }
          >
            <>
            Before you run the benchmark. At the size it picks, the time of the same three operations done as three
            separate round trips, each with its own copies in and out, divided by the time of one{' '}
            <code>blockTail</code> call, will be what? First count the tensor-sized copies each version makes: here
            the bias is one row and the scale is two numbers.
            </>
          </Translated>
        }
        unit={tr('× the time of one blockTail call', '× thời gian của một lời gọi blockTail')}
        measure={chainRatio}
        missing={
          <Translated
            vi={
              <>
                Chạy <code>python3 labs/pmpp.py run 2.3 --bench</code> trên một máy có GPU.
              </>
            }
          >
            <>
              Run <code>python3 labs/pmpp.py run 2.3 --bench</code> on a machine with a GPU.
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                Số lần sao chép cho biết mỗi phiên bản di chuyển gì; tỉ lệ là cái giá của điều đó trên device của bạn. Trên CUDA và HIP các lượt đi về đi qua giữa bộ nhớ host và bộ nhớ device. Trên Metal không gì đi qua một bus: các lượt đi về sao chép mảng host vào các buffer chia sẻ rồi sao chép ngược lại, đó là cái giá của một giao diện nhận mảng host.
              </>
            }
          >
            <>
            The copy count says what each version moves; the ratio is what that costs on your device. On CUDA and HIP
            the round trips cross between host and device memory. On Metal nothing crosses a bus: the round trips copy
            host arrays into shared buffers and back, which is the cost of an interface that takes host arrays.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l3}
        filter={(r) => r.name === 'op' || r.name === 'chain'}
        cols={[
          {
            head: tr('Measured', 'Đối tượng đo'),
            cell: (r) =>
              r.name === 'chain'
                ? `blockTail, ${String(r.rows)} × ${String(r.d)}` // i18n-ignore: function name and shape
                : tr(`${String(r.kernel ?? '')} alone, data on the device`, `chỉ ${String(r.kernel ?? '')}, dữ liệu trên device`),
          },
          numCol(tr('Elements', 'Số phần tử'), 'elements', 8),
          {
            head: tr('Time, ms', 'Thời gian, ms'),
            num: true,
            cell: (r) => {
              const v = num(r.name === 'chain' ? r.resident_s : r.seconds);
              return v == null ? '' : fmt(v * 1000);
            },
          },
          numCol(tr('GB/s requested', 'GB/s yêu cầu'), 'gb_per_s'),
          msOf(tr('Three round trips, ms', 'Ba lượt đi về, ms'), 'roundtrips_s'),
          numCol(tr('Round trips over blockTail', 'Lượt đi về trên blockTail'), 'ratio'),
        ]}
      />
      <ReferenceSolution lab={l3} tiers={M0_TIERS} />

      {/* ------------------------------------------------------------ 2.4 */}
      <h2>
        <Translated vi="Lab 2.4: săn lỗi, một hàm stub gần như chạy được">Lab 2.4: bug hunt, a stub that almost works</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn được giao một chương trình hoàn chỉnh biên dịch được. <code>scalePrefix</code> nhân <code>len</code> số float đầu tiên trong <code>n</code> số với <code>s</code> trên device và phải giữ phần đệm từ{' '}<code>len</code> đến <code>n</code> y như cũ: hình dạng của một phép toán áp dụng lên phần hợp lệ của một batch có đệm. Bản CUDA có bốn lỗi độc lập, bản Metal có ba. Chúng là những sai lầm mà hàm stub phía host của chương này dễ dẫn tới, và tìm ra chúng trong mã bạn không viết là kỹ năng bạn sẽ cần cho mã của chính mình.
          </p>
        }
      >
        <p>
          You are handed a complete program that compiles. <code>scalePrefix</code> multiplies the first{' '}
          <code>len</code> of <code>n</code> floats by <code>s</code> on the device and must leave the padding from{' '}
          <code>len</code> to <code>n</code> exactly as it was: the shape of an operation applied to the valid part of
          a padded batch. The CUDA version has four independent defects, the Metal version three. They are the mistakes
          this chapter's host stub invites, and finding them in code you did not write is the skill you will need for
          your own.
        </p>
      </Translated>
      <Source section="2.4–2.6" pages="28–39" figs={tr('Fig. 2.10, Fig. 2.13; the sidebar on p. 32', 'Hình 2.10, Hình 2.13; khung bên ở p. 32')} />
      <LabFiles
        lab={l4}
        files={[
          [
            'kernel.cu',
            <Translated vi={<><code>scalePrefixKernel</code> và <code>scalePrefix</code>, với bốn lỗi nằm giữa chúng</>}>
              <><code>scalePrefixKernel</code> and <code>scalePrefix</code>, with the four defects between them</>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<><code>scale_prefix</code>, được dispatch theo nguyên threadgroup như backend Metal của llama.cpp làm, nên nó tự tính chỉ số của mình theo cách một kernel CUDA làm</>}>
              <><code>scale_prefix</code>, dispatched in whole threadgroups as llama.cpp's Metal backend does, so it computes its own index the way a CUDA kernel does</>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<><code>scalePrefix</code>, hàm host Metal</>}>
              <><code>scalePrefix</code>, the Metal host function</>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Mỗi bài kiểm thử điền mảng bằng các giá trị giữa 1 và 2, gọi <code>scalePrefix</code> với <code>s = 3</code>, và kiểm tra toàn bộ mảng: phần đầu được nhân, phần đệm giữ nguyên từng bit. Các mảng có đệm đến trước, rồi các mảng không có đệm, rồi, trên CUDA, HIP và trình giả lập, một lần khởi chạy mà device không chạy được, phải trả về khác 0.
          </p>
        }
      >
        <p>
          Each test fills the array with values between 1 and 2, calls <code>scalePrefix</code> with <code>s = 3</code>,
          and checks the whole array: the prefix scaled, the padding unchanged bit for bit. Arrays with padding come
          first, then arrays with none, then, on CUDA, HIP and the emulator, a launch the device cannot run, which must
          return non-zero.
        </p>
      </Translated>
      <Predict
        id="ch02/labs/p-lab4-reported"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy bất cứ thứ gì trên GPU của bạn. Đọc <code>work/kernel.cu</code> và quyết định các lỗi là gì. Trong số đó, bao nhiêu lỗi GPU sẽ tự báo ra, qua một mã lỗi hoặc một lần sập, thay vì là một giá trị sai mà chỉ một bài kiểm thử so với bản tham chiếu CPU mới phát hiện?
              </>
            }
          >
            <>
            Before you run anything on your GPU. Read <code>work/kernel.cu</code> and decide what the defects are. Of
            them, how many will the GPU report by itself, through an error code or a crash, as opposed to a wrong value
            that only a test against the CPU reference reveals?
            </>
          </Translated>
        }
        unit={tr('of the defects', 'trong số các lỗi')}
      >
        {() => <HuntReveal />}
      </Predict>
      <Translated vi={<p>Phương pháp quan trọng hơn các bản sửa:</p>}>
        <p>The method matters more than the fixes:</p>
      </Translated>
      <ol>
        <Translated
          vi={
            <li>
              Trình giả lập trước: <code>python3 labs/pmpp.py run 2.4 --backend emu</code>. Nó chạy kernel dưới AddressSanitizer, công cụ dừng ở lần truy cập đầu tiên ra ngoài một mảng device và chỉ ra dòng mã.
            </li>
          }
        >
          <li>
            Emulator first: <code>python3 labs/pmpp.py run 2.4 --backend emu</code>. It runs the kernel under
            AddressSanitizer, which stops at the first access outside a device array and names the line.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Sửa một chỗ rồi chạy lại. Các lỗi thay đổi theo tiến trình: một lỗi có thể che giấu lỗi khác cho đến khi nó được sửa.
            </li>
          }
        >
          <li>
            Fix one thing and run again. The failures change as you go: one defect can hide another until it is fixed.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Sau đó chạy trên GPU của bạn, <code>python3 labs/pmpp.py run 2.4</code>, và ghi lại lỗi nào nó tự cho bạn thấy và lỗi nào nó che giấu.
            </li>
          }
        >
          <li>
            Then run on your GPU, <code>python3 labs/pmpp.py run 2.4</code>, and note which defects it showed you by
            itself and which it hid.
          </li>
        </Translated>
      </ol>
      <LabResults
        lab={l4}
        noBench={tr(
          'This lab has no benchmark. The result is the line above: every test passing on the emulator and on a device. As shipped, the program is meant to fail.',
          'Lab này không có benchmark. Kết quả là dòng ở trên: mọi bài kiểm thử đều qua trên trình giả lập và trên một device. Như khi phát hành, chương trình được cố ý để thất bại.',
        )}
      />
      <ReferenceSolution lab={l4} tiers={HUNT_TIERS} />

      <h2>
        <Translated vi="Những gì các lab này để lại cho sau">What these labs leave for later</Translated>
      </h2>
      <Translated
        vi={
          <p>
            <code>launchAddBias</code>, <code>launchScaleShift</code> và <code>launchAddResidual</code> là những mảnh của mốc dự án M0 mà các mốc dự án sau gọi trên dữ liệu ở lại trên device. Chương 3 thêm mốc dự án tiếp theo, một phép nhân ma trận cho một lớp tuyến tính, với cùng các quy ước (<a href={href('ch03', 'labs')}>các lab của chương đó</a>). <a href={href('ch02', 'wrap')}>Trang tóm tắt</a> có bài kiểm tra tổng hợp của chương và danh sách những chỗ trang sách in khác với mã đã kiểm thử.
          </p>
        }
      >
        <p>
          <code>launchAddBias</code>, <code>launchScaleShift</code> and <code>launchAddResidual</code> are the pieces of
          milestone M0 that later milestones call on data that stays on the device. Chapter 3 adds the next milestone, a
          matrix multiplication for a linear layer, with the same conventions (<a href={href('ch03', 'labs')}>its
          labs</a>). The <a href={href('ch02', 'wrap')}>summary</a> has the chapter's mixed check and the list of places
          where the printed page differs from the tested code.
        </p>
      </Translated>
    </LabsPage>
  );
}
