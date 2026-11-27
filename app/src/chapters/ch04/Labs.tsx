import { Predict, type Question } from '../../components/Ask';
import { Beyond, Source, Today, type Tier } from '../../components/notes';
import {
  BackendsAndEmulator,
  LabFiles,
  LabPrediction,
  LabResults,
  LabsIntro,
  LabsPage,
  RaceStatus,
  ReferenceSolution,
  RunningLabs,
  VerificationStatus,
  benchRow,
  defineLabs,
  measure,
  msCol,
  numCol,
  rawCol,
  useLabs,
  type BenchSet,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';

// The page is built from components/LabKit.tsx; what is here is chapter 4's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch04', [
  { num: '4.1', dir: 'lab1_launchconfig', name: 'launch from queries (M1)', vi: { name: 'khởi chạy từ truy vấn (M1)' } },
  { num: '4.2', dir: 'lab2_divergence', name: 'divergence', vi: { name: 'phân kỳ' } },
  { num: '4.3', dir: 'lab3_barrierhunt', name: 'barrier hunt', vi: { name: 'săn lỗi rào chắn' } },
  { num: '4.4', dir: 'lab4_blocksize', name: 'block size', vi: { name: 'kích thước khối' } },
]);
const [l1, l2, l3, l4] = LABS;

// Lab 3.3 is the other half of milestone M1.
const M1_FIRST_HALF = 'ch03/lab3_naive_gemm';

// Identifier colours for the reference solutions: buffers are global memory, values a
// thread computes for itself are registers. Only names that mean one thing in all three files.
const tiers = (global: string[], reg: string[]): Record<string, Tier> => ({
  ...Object.fromEntries(global.map((n) => [n, 'global' as Tier])),
  ...Object.fromEntries(reg.map((n) => [n, 'reg' as Tier])),
});
const LINEAR_TIERS = tiers(['X', 'W', 'Y'], []);
const DIVERGENCE_TIERS = tiers(['which', 'trips'], []);
const BARRIER_TIERS = tiers([], ['mine', 'valid', 'base']);
const GATHER_TIERS = tiers(['table', 'idx'], ['sum']);

// ---------------------------------------------------------------- what each prediction is measured against
const ms = (s: number) => `${fmt(s * 1000)} ms`; // i18n-ignore: unit
const shape = (r: BenchRow | undefined, x: string, y: string) => (r ? `${String(r[x])} × ${String(r[y])}` : '');

/** Lab 4.1: at batch 1, the derived configuration's speed over fixed 16 × 16 blocks. */
const derivedOverFixed = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const v = num(benchRow(s, 'speedup', { batch: 1 })?.derived_over_fixed);
    const derived = s.rows.find((r) => r.name === 'linear' && r.config === 'derived' && r.batch === 1);
    const fixed = s.rows.find((r) => r.name === 'linear' && r.config === 'fixed16' && r.batch === 1);
    const a = num(derived?.seconds);
    const b = num(fixed?.seconds);
    if (!v || !a || !b) return null;
    return { value: v, detail: tr(`${ms(a)} with your ${shape(derived, 'block_x', 'block_y')} blocks, ${ms(b)} with 16 × 16, batch 1`, `${ms(a)} với khối ${shape(derived, 'block_x', 'block_y')} của bạn, ${ms(b)} với 16 × 16, batch 1`) };
  });

/** Lab 4.1: the width the device reported (warpSize, or the pipeline's threadExecutionWidth on Metal). */
const reportedWidth = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const queried = s.rows.filter((r) => r.name === 'queried');
    const row = queried.find((r) => r.device === 0) ?? queried[0];
    const metal = s.backend === 'metal';
    const v = num(metal ? row?.threadExecutionWidth : row?.warpSize);
    if (!v) return null;
    return {
      value: v,
      detail: metal ? tr('threadExecutionWidth of the linear-layer pipeline', 'threadExecutionWidth của pipeline lớp tuyến tính') : tr('warpSize of device 0, from your queryDevices', 'warpSize của device 0, từ queryDevices của bạn'),
    };
  });

const isMixed = (r: BenchRow) => String(r.pattern ?? '').startsWith('half of each'); // i18n-ignore: data key written by the driver
const mixedRow = (s: BenchSet, name: string) => s.rows.find((r) => r.name === name && isMixed(r));

/** Lab 4.2: the branch with both paths in every warp, over the same mix sorted. */
const divergenceCost = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const r = mixedRow(s, 'branch');
    const v = num(r?.ratio_to_sorted);
    const mixed = num(r?.seconds);
    const sorted = num(r?.sorted_seconds);
    if (!r || !v || !mixed || !sorted) return null;
    return { value: v, detail: tr(`${ms(mixed)} mixed, ${ms(sorted)} sorted, width ${String(r.width)}`, `${ms(mixed)} khi trộn, ${ms(sorted)} khi sắp xếp, độ rộng ${String(r.width)}`) };
  });

/** Lab 4.2: select over branch, on that same mixed pattern. */
const selectCost = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const sel = mixedRow(s, 'select');
    const br = mixedRow(s, 'branch');
    const v = num(sel?.ratio_to_branch);
    const a = num(sel?.seconds);
    const b = num(br?.seconds);
    if (!v || !a || !b) return null;
    return { value: v, detail: tr(`${ms(a)} for the select, ${ms(b)} for the branch`, `${ms(a)} cho select, ${ms(b)} cho nhánh`) };
  });

/** Lab 4.4: the slowest block size's time over the fastest's. */
const blockSizeSpread = (reports: Reports) =>
  measure(reports, l4, (s) => {
    const r = benchRow(s, 'summary');
    const v = num(r?.slowest_over_fastest);
    const noise = num(r?.largest_median_over_best);
    if (!r || !v) return null;
    return {
      value: v,
      detail:
        tr(`slowest ${String(r.slowest_threads_per_block)} threads per block, fastest ${String(r.fastest_threads_per_block)}`, `chậm nhất ${String(r.slowest_threads_per_block)} luồng mỗi khối, nhanh nhất ${String(r.fastest_threads_per_block)}`) +
        (noise ? tr(`; the median run of one size was up to ${fmt(noise)} times its own best`, `; lần chạy trung vị của một kích thước chậm tới ${fmt(noise)} lần so với lần tốt nhất của chính nó`) : ''),
    };
  });

// ---------------------------------------------------------------- lab 4.3: the experiment of exercise 5
const BARRIER_ID = 'ch04/labs/p-lab3-barrier';
const OUTCOMES = () => [
  tr('Neither fails: both block sizes pass every launch.', 'Không cái nào hỏng: cả hai kích thước khối đều qua mọi lần khởi chạy.'),
  tr('Only the block four warps wide fails.', 'Chỉ khối rộng bốn warp hỏng.'),
  tr('Only the block one warp wide fails.', 'Chỉ khối rộng một warp hỏng.'),
  tr('Both fail.', 'Cả hai đều hỏng.'),
] as const;
const GPU_BACKENDS = ['metal', 'cuda', 'hip'];

interface CaseResult {
  /** The test cases of one block size, as the driver named them. */
  cases: { name: string; pass: boolean; detail: string }[];
  failed: boolean;
}
interface BarrierRun {
  backend: string;
  device: string;
  date: string;
  untouched: boolean;
  one: CaseResult;
  four: CaseResult;
  outcome: number;
}
/** The two block sizes the experiment is about, from the last run of the reader's work/ on each GPU. */
function barrierRuns(reports: Reports): BarrierRun[] {
  return GPU_BACKENDS.flatMap((b) => {
    const run = reports[b];
    if (!run) return [];
    const tests = run.results.filter((x) => x.lab === l3.id && x.variant === 'work').flatMap((x) => x.tests);
    const pick = (label: string): CaseResult => {
      const cases = tests
        .filter((t) => t.case.includes(`(${label})`))
        .map((t) => ({ name: t.case, pass: t.pass, detail: t.detail ?? '' }));
      return { cases, failed: cases.some((c) => !c.pass) };
    };
    const one = pick('1 x width'); // i18n-ignore: test case name written by the driver
    const four = pick('4 x width'); // i18n-ignore: test case name written by the driver
    if (!one.cases.length || !four.cases.length) return [];
    return [{
      backend: b,
      device: String(run.device.name ?? b),
      date: run.generated.slice(0, 10),
      untouched: run.work_state?.[l3.id] === 'untouched',
      one,
      four,
      outcome: (one.failed ? 2 : 0) + (four.failed ? 1 : 0),
    }];
  });
}
/** "passed every launch" or the driver's own count of wrong launches. */
const verdict = (c: CaseResult) => {
  if (!c.failed) return tr('passed every launch', 'qua mọi lần khởi chạy');
  const counts = c.cases
    .filter((x) => !x.pass)
    .map((x) => /wrong in \d+ of \d+ launches/.exec(x.detail)?.[0] ?? tr('failed', 'hỏng'))
    .join('; ');
  return tr(`failed (${counts})`, `hỏng (${counts})`);
};

function BarrierReveal({ committed }: { committed: number }) {
  const { reports, loading, reload } = useLabs();
  const runs = barrierRuns(reports);
  return (
    <>
      {runs.length ? (
        runs.map((r) => (
          <div key={r.backend} data-measured={r.outcome} data-backend={r.backend}>
            <Translated
              vi={
                <p>
                  Trên {r.device} ({r.backend}), lần chạy cuối của <code>work/</code> của bạn vào {r.date}
                  {r.untouched ? ', khi đó vẫn là mã khởi đầu như được phát hành' : ''}: rộng một warp,{' '}
                  <span data-case="one">{verdict(r.one)}</span>; rộng bốn warp, <span data-case="four">{verdict(r.four)}</span>.
                </p>
              }
            >
              <p>
                On {r.device} ({r.backend}), the last run of your <code>work/</code> on {r.date}
                {r.untouched ? ', which was still the starter as shipped' : ''}: one warp wide,{' '}
                <span data-case="one">{verdict(r.one)}</span>; four warps wide, <span data-case="four">{verdict(r.four)}</span>.
              </p>
            </Translated>
            <p>
              <b>{OUTCOMES()[r.outcome]}</b>{' '}
              {r.untouched
                ? tr('The starter still has all three defects, so this is not the experiment yet.', 'Mã khởi đầu vẫn còn cả ba lỗi, nên đây chưa phải là thí nghiệm.')
                : r.outcome === committed
                  ? tr('That is the outcome you predicted.', 'Đó là kết quả bạn đã dự đoán.')
                  : tr('That is not the outcome you predicted.', 'Đó không phải là kết quả bạn đã dự đoán.')}
            </p>
          </div>
        ))
      ) : loading ? (
        <p data-measured="loading"><Translated vi="Đang đọc báo cáo.">Reading reports.</Translated></p>
      ) : (
        <p data-measured="none">
          <Translated
            vi={
              <>
                Chưa có lần chạy nào của bản làm của bạn cho lab này trên GPU. Hãy sửa ba lỗi, xóa rào chắn, và chạy{' '}
                <code>python3 labs/pmpp.py run 4.3</code> trên một máy có GPU.{' '}
              </>
            }
          >
            No run of your work for this lab on a GPU yet. Fix the three defects, delete the barrier, and run{' '}
            <code>python3 labs/pmpp.py run 4.3</code> on a machine with a GPU.{' '}
          </Translated>
          <button className="quiet" onClick={reload}>
            <Translated vi="Tải lại kết quả">Reload results</Translated>
          </button>
        </p>
      )}
      <Translated
        vi={
          <p className="muted">
            Phần này đọc lần chạy cuối của chính <code>work/</code> của bạn. Nó chỉ là thí nghiệm nếu lần chạy đó đã sửa hai
            lỗi kia và bỏ rào chắn; nếu vẫn còn lỗi nào, các thất bại không nói gì về rào chắn. Một kích thước khối qua mọi lần
            khởi chạy chỉ cho bạn thấy những lịch chạy mà GPU của bạn tình cờ thực hiện, không cho thấy kernel đúng.
          </p>
        }
      >
        <p className="muted">
          This reads the last run of your own <code>work/</code>. It is the experiment only if that run had the other two
          defects fixed and the barrier removed; with any defect still in, failures say nothing about the barrier. A
          block size that passed all its launches showed you the schedules your GPU happened to run, not that the kernel
          is correct.
        </p>
      </Translated>
    </>
  );
}

/** A prediction with four outcomes, stored as the outcome's written index. Stored, not graded. */
function BarrierPrediction() {
  return (
    <Predict
      id={BARRIER_ID}
      options={[...OUTCOMES()]}
      prompt={
        <Translated
          vi={
            <>
              Trước thí nghiệm. Với hai lỗi kia đã sửa và rào chắn đã bỏ, bạn chạy các bài kiểm thử trên GPU của mình. Điều
              gì xảy ra ở khối rộng một warp (trường hợp đánh dấu <code>1 x width</code>) và ở khối rộng bốn warp (
              <code>4 x width</code>)?
            </>
          }
        >
          <>
            Before the experiment. With the other two defects fixed and the barrier removed, you run the tests on your GPU.
            What happens at the block one warp wide (the case marked <code>1 x width</code>) and at the block four warps
            wide (<code>4 x width</code>)?
          </>
        </Translated>
      }
    >
      {(committed) => <BarrierReveal committed={committed} />}
    </Predict>
  );
}

// ---------------------------------------------------------------- milestone M1
function MilestoneStatus() {
  const { reports, loading } = useLabs();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const both = backends.filter(
    (b) => labStatus(reports[b], M1_FIRST_HALF, 'work') === 'pass' && labStatus(reports[b], l1.id, 'work') === 'pass',
  );
  const name = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  return (
    <p data-milestone={loading ? 'loading' : both.join(',') || 'none'}>
      {loading
        ? tr('Reading reports.', 'Đang đọc báo cáo.')
        : !backends.length
          ? tr('No lab results are recorded on this machine yet.', 'Chưa có kết quả lab nào được ghi trên máy này.')
          : both.length
            ? tr(`Your own labs 3.3 and 4.1 both pass on ${both.map(name).join(' and ')} on this machine: M1 is done there.`, `Lab 3.3 và 4.1 của chính bạn đều qua trên ${both.map(name).join(' và ')} trên máy này: M1 đã xong ở đó.`)
            : tr('Your own labs 3.3 and 4.1 do not both pass on any backend of this machine yet.', 'Lab 3.3 và 4.1 của bạn chưa cùng qua trên backend nào của máy này.')}
    </p>
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
              Bốn lab, mỗi lab trong thư mục riêng dưới <code>labs/ch04/</code>. Lab đầu hỏi thiết bị các giới hạn của nó và
              định kích thước một lần khởi chạy từ đó, hoàn thành mốc dự án đầu tiên. Lab thứ hai và thứ tư đo những gì phần
              4.5 và 4.7 chỉ có thể đếm. Lab thứ ba là một cuộc săn lỗi quanh một rào chắn.
            </>
          }
        >
          Four labs, each in its own directory under <code>labs/ch04/</code>. The first asks the device for its limits and
          sizes a launch from them, which finishes the project's first milestone. The second and the fourth measure what
          sections 4.5 and 4.7 can only count. The third is a bug hunt around a barrier.
        </Translated>
      </LabsIntro>
      <RunningLabs
        example="4.2"
        alsoShownIn={
          <Translated
            vi={
              <>
                các phần <a href={href('ch04', '4.4')}>4.4</a>, <a href={href('ch04', '4.5')}>4.5</a>,{' '}
                <a href={href('ch04', '4.7')}>4.7</a> và <a href={href('ch04', '4.8')}>4.8</a>
              </>
            }
          >
            <>
              sections <a href={href('ch04', '4.4')}>4.4</a>, <a href={href('ch04', '4.5')}>4.5</a>,{' '}
              <a href={href('ch04', '4.7')}>4.7</a> and <a href={href('ch04', '4.8')}>4.8</a>
            </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <Translated
        vi={
          <p>
            Trình giả lập có một biến thể thứ hai, <code>race</code>, mà lab 4.3 cũng được dựng cho nó. Nó chạy các luồng giả
            lập đồng thời dưới ThreadSanitizer và báo cáo hai luồng chạm vào cùng một phần tử mà không đồng bộ hóa, kèm cả hai
            dòng mã nguồn. Hãy yêu cầu nó bằng <code>--backend race</code>; <code>python3 labs/pmpp.py doctor</code> cho biết
            trình biên dịch của bạn có dựng được nó không. Bảng trạng thái bên dưới cho nó một cột riêng; các bảng khác trên
            trang này chỉ hiển thị biến thể thứ nhất.
          </p>
        }
      >
        <p>
          The emulator has a second flavour, <code>race</code>, which lab 4.3 is also built for. It runs the emulated
          threads concurrently under ThreadSanitizer and reports two threads that touch the same element without
          synchronisation, with both source lines. Ask for it with <code>--backend race</code>;{' '}
          <code>python3 labs/pmpp.py doctor</code> says whether your compiler can build it. The status table below gives
          it a column of its own; the other tables on this page show the first flavour only.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Hai phép đo của chương này, chi phí của một warp phân kỳ và ảnh hưởng của kích thước khối lên một kernel phải chờ
            bộ nhớ, chỉ tồn tại trên GPU. Trình giả lập kiểm tra rằng lab 4.2 và 4.4 tính ra giá trị đúng, và không nói gì về
            tốc độ của chúng.
          </p>
        }
      >
        <p>
          Two of this chapter's measurements, what a divergent warp costs and what block size does to a kernel that
          waits on memory, exist only on a GPU. The emulators check that labs 4.2 and 4.4 compute the right values, and
          nothing about their speed.
        </p>
      </Translated>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 4.1 */}
      <h2><Translated vi="Lab 4.1: hỏi thiết bị, rồi định kích thước lần khởi chạy">Lab 4.1: ask the device, then size the launch</Translated></h2>
      <Translated
        vi={
          <p>
            Lab 3.3 khởi chạy GEMM ngây thơ với khối 16 × 16 vì các hình của sách làm vậy. Ở đây thiết bị quyết định. Bạn viết
            vòng lặp truy vấn của <a href={href('ch04', '4.8')}>phần 4.8</a>, một hàm thuần biến những gì thiết bị báo cáo
            thành một khối và một lưới, và lần khởi chạy dùng cả hai. Kernel là của lab 3.3, được cho sẵn đầy đủ và viết lại
            với tên gọi của lớp tuyến tính: <code>Y[b][o]</code> là tổng theo <code>k</code> của{' '}
            <code>X[b][k] * W[k][o]</code>, thứ tự hàng, batch trước, mỗi đầu ra một luồng, cùng thứ tự đối số với{' '}
            <code>naiveGemmKernel</code>.
          </p>
        }
      >
        <p>
          Lab 3.3 launched the naive GEMM with 16 × 16 blocks because the book's figures do. Here the device decides. You
          write the query loop of <a href={href('ch04', '4.8')}>section 4.8</a>, a pure function that turns what the
          device reports into a block and a grid, and the launch that uses both. The kernel is lab 3.3's, given complete
          and restated with a linear layer's names: <code>Y[b][o]</code> is the sum over <code>k</code> of{' '}
          <code>X[b][k] * W[k][o]</code>, row-major, batch first, one thread per output, the same argument order as{' '}
          <code>naiveGemmKernel</code>.
        </p>
      </Translated>
      <Source section="4.8" pages="87–89" figs={tr('the slot arithmetic is §4.7, pp. 85–86; the warp cut §4.4, p. 75', 'phép tính ô ở §4.7, pp. 85–86; cách cắt warp ở §4.4, p. 75')} />
      <LabFiles
        lab={l1}
        files={[
          ['kernel.cu', <Translated vi={<><code>queryDevices</code> (vòng lặp của p. 88), <code>chooseLaunch</code> và <code>launchLinear</code></>}><><code>queryDevices</code> (the loop of p. 88), <code>chooseLaunch</code> and <code>launchLinear</code></></Translated>],
          ['kernel.metal', <Translated vi={<>kernel thăm dò <code>simd_probe</code>; kernel tuyến tính được cho sẵn</>}><>the probe kernel <code>simd_probe</code>; the linear kernel is given</></Translated>],
          ['launch.mm', <Translated vi={<><code>queryMetal</code>, <code>encodeLinear</code> và <code>encodeProbe</code></>}><><code>queryMetal</code>, <code>encodeLinear</code> and <code>encodeProbe</code></></Translated>],
        ]}
      />
      <Translated
        vi={
          <p>
            Một header nhỏ bên cạnh, <code>launch_config.h</code>, chứa các struct <code>DeviceInfo</code> và{' '}
            <code>MetalInfo</code>. Nó được cho sẵn và không cần sửa.
          </p>
        }
      >
        <p>
          A small header beside them, <code>launch_config.h</code>, holds the <code>DeviceInfo</code> and{' '}
          <code>MetalInfo</code> structs. It is given and not edited.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <code>chooseLaunch</code> phải trả về một khối nằm trong <code>maxThreadsPerBlock</code> và{' '}
            <code>maxThreadsDim</code>, là một số nguyên warp theo <code>warpSize</code> được báo cáo, có khả năng lấp đầy các ô
            luồng của một SM (không nhỏ đến mức ô khối cạn trước, cũng không là kích thước để thừa ô), không cao hơn số hàng,
            và một lưới phủ hết đầu ra trong giới hạn <code>maxGridSize</code>, hoặc trả về <code>false</code>. Một trường
            bằng 0 nghĩa là thiết bị không báo cáo nó, và quy tắc cần nó được bỏ qua. Các quy tắc là điểm xuất phát của khóa học
            này cho mô hình mỗi đầu ra một luồng, không phải khuyến nghị của nhà cung cấp. Trên Metal, threadgroup theo công thức
            của Apple: rộng bằng <code>threadExecutionWidth</code> của pipeline, cao đến mức{' '}
            <code>maxTotalThreadsPerThreadgroup</code> cho phép và batch cần, được dispatch bằng{' '}
            <code>dispatchThreads</code>.
          </p>
        }
      >
        <p>
          <code>chooseLaunch</code> must return a block within <code>maxThreadsPerBlock</code> and{' '}
          <code>maxThreadsDim</code>, a whole number of warps by the reported <code>warpSize</code>, able to fill an SM's
          thread slots (neither so small that the block slots run out first nor a size that leaves slots over), no taller
          than the number of rows, and a grid that covers the output within <code>maxGridSize</code>, or return{' '}
          <code>false</code>. A field of 0 means the device did not report it, and the rule that needs it is skipped. The
          rules are this course's starting point for one thread per output, not a vendor's recommendation. On Metal the
          threadgroup follows Apple's recipe instead: the pipeline's <code>threadExecutionWidth</code> wide, as tall as{' '}
          <code>maxTotalThreadsPerThreadgroup</code> allows and the batch needs, dispatched with{' '}
          <code>dispatchThreads</code>.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch04-facts.md, A18, C2, C6)', '2026-11-20 (notes/today/ch04-facts.md, mục A18, C2, C6)')}>
        <Translated
          vi={
            <p>
              Đừng đọc <code>devProp.clockRate</code>: CUDA 13.0 đã gỡ nó, còn HIP vẫn có, nên một mã nguồn đọc nó sẽ dựng được
              trên bên này mà không dựng được trên bên kia. Kho thanh ghi mỗi SM là <code>regsPerMultiprocessor</code>, không
              phải <code>regsPerBlock</code> của p. 89. Trên Metal, độ rộng thuộc về một pipeline đã biên dịch, không thuộc về
              thiết bị.
            </p>
          }
        >
          <p>
            Do not read <code>devProp.clockRate</code>: CUDA 13.0 removed it, and HIP still has it, so one source that
            reads it builds on one and not the other. The per-SM register pool is <code>regsPerMultiprocessor</code>, not
            the <code>regsPerBlock</code> of p. 89. On Metal the width belongs to a compiled pipeline, not to the device.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 4.1</code>, bắt đầu với <code>--backend emu</code>. Mọi trường mà{' '}
            <code>queryDevices</code> của bạn điền phải bằng những gì harness tự đọc được. <code>chooseLaunch</code> được chạy
            trên chín thiết bị tổng hợp, không cái nào là sản phẩm thật, kết hợp với bốn hình dạng đầu ra, và mỗi quy tắc được
            kiểm tra riêng để một thất bại gọi tên được quy tắc. Lần khởi chạy được so với tham chiếu CPU ở 1 × 1000, 3 × 40, 37
            × 53 và 64 × 257 đầu ra. Trên Metal, <code>queryMetal</code> được so trên ba pipeline, một trong số đó biên dịch với
            giới hạn luồng thấp hơn của thiết bị để một giá trị đọc nhầm đối tượng lộ ra; và các bản ghi của kernel thăm dò phải
            thỏa những gì đặc tả hứa về SIMD-group, và không hơn.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 4.1</code>, starting with <code>--backend emu</code>. Every field your{' '}
          <code>queryDevices</code> fills must equal what the harness reads on its own. <code>chooseLaunch</code> is run on
          nine synthetic devices, none of them a product, crossed with four output shapes, and each rule is checked on its
          own so a failure names the rule. The launch is compared with a CPU reference at 1 × 1000, 3 × 40, 37 × 53 and 64
          × 257 outputs. On Metal, <code>queryMetal</code> is compared on three pipelines, one of them compiled with a
          lower thread limit than the device's so that a value read from the wrong object shows; and the probe's records
          must satisfy what the specification promises about SIMD-groups, and nothing more.
        </p>
      </Translated>
      <LabPrediction
        id="ch04/labs/p-lab1-speedup"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy benchmark. Nó đo thời gian một lớp tuyến tính với 1024 đầu vào và 4096 đầu ra (lớp ví dụ của
                sách ở §20.6, mà lab 5.3 cũng dùng) ở batch 1, một lần với cấu hình suy ra của bạn và một lần với khối cố định
                16 × 16. Cấu hình của bạn sẽ nhanh gấp bao nhiêu lần? (1 nghĩa là cùng tốc độ, dưới 1 nghĩa là chậm hơn.)
              </>
            }
          >
            <>
              Before you run the benchmark. It times a linear layer with 1024 inputs and 4096 outputs (the book's example
              layer of §20.6, which lab 5.3 also uses) at batch 1, once with your derived configuration and once with
              fixed 16 × 16 blocks. How many times as fast will yours be? (1 means the same speed, below 1 means slower.)
            </>
          </Translated>
        }
        unit={tr('× the speed of 16 × 16', '× tốc độ của 16 × 16')}
        measure={derivedOverFixed}
        missing={
          <Translated
            vi={
              <>
                Khi lab của bạn đã qua, hãy chạy <code>python3 labs/pmpp.py run 4.1 --bench</code> trên một máy có GPU, rồi
                tải lại.
              </>
            }
          >
            <>
              Once your lab passes, run <code>python3 labs/pmpp.py run 4.1 --bench</code> on a machine with a GPU, then
              reload.
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                Trên CUDA và HIP, một khối 16 × 16 ở batch 1 có 15 hàng luồng không có việc gì làm. Trên Metal,{' '}
                <code>dispatchThreads</code> cắt threadgroup cho vừa lưới, nên 16 × 16 chạy như 16 × 1 ở đó. Hãy xem batch 64
                trong bảng trước khi quyết định con số nói gì về các quy tắc của bạn. Một cấu hình thua là một phát hiện về
                các quy tắc, và là lý do chúng nằm trong một hàm.
              </>
            }
          >
            <>
              On CUDA and HIP a 16 × 16 block at batch 1 has 15 rows of threads with nothing to do. On Metal,{' '}
              <code>dispatchThreads</code> trims a threadgroup to the grid, so 16 × 16 runs as 16 × 1 there. Look at batch
              64 in the table before you decide what the number says about your rules. A configuration that loses is a
              finding about the rules, and the reason they live in one function.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch04/labs/p-lab1-width"
        prompt={
          <Translated vi={<>Và độ rộng thiết bị của bạn sẽ báo cáo: bao nhiêu luồng trong một warp, hoặc trong một SIMD-group của Metal?</>}>
            <>And the width your device will report: how many threads in a warp, or in a Metal SIMD-group?</>
          </Translated>
        }
        unit={tr('threads', 'luồng')}
        measure={reportedWidth}
        missing={
          <Translated
            vi={
              <>
                Chạy <code>python3 labs/pmpp.py run 4.1 --bench</code> trên một máy có GPU, rồi tải lại. Độ rộng được ghi cùng
                benchmark.
              </>
            }
          >
            <>
              Run <code>python3 labs/pmpp.py run 4.1 --bench</code> on a machine with a GPU, then reload. The width is
              recorded with the benchmark.
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                CUDA ghi 32 cho mọi compute capability mà nó liệt kê, HIP báo 64 hoặc 32 tùy kiến trúc AMD, còn Apple không ghi
                giá trị nào, chỉ ghi nơi đọc nó (notes/today/ch04-facts.md, A8, B6, C2). Trên Metal, đây là độ rộng của một
                pipeline; kernel khác có thể báo giá trị khác.
              </>
            }
          >
            <>
              CUDA documents 32 on every compute capability it lists, HIP reports 64 or 32 depending on the AMD
              architecture, and Apple documents no value at all, only where to read it (notes/today/ch04-facts.md, A8, B6,
              C2). On Metal this is one pipeline's width; another kernel may report another.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l1}
        filter={(r) => r.name === 'linear'}
        cols={[
          { head: tr('Configuration', 'Cấu hình'), cell: (r) => (r.config === 'derived' ? tr('derived (from the queries)', 'suy ra (từ các truy vấn)') : tr('fixed 16 × 16', 'cố định 16 × 16')) },
          rawCol(tr('Batch', 'Batch'), 'batch'),
          { head: tr('Block', 'Khối'), num: true, cell: (r) => shape(r, 'block_x', 'block_y') },
          { head: tr('Grid', 'Lưới'), num: true, cell: (r) => shape(r, 'grid_x', 'grid_y') },
          msCol,
          numCol('GFLOP/s', 'gflops'),
        ]}
      />
      <Translated
        vi={
          <p>
            Cùng lần chạy đó ghi lại những gì các truy vấn của bạn trả về, mà <a href={href('ch04', '4.8')}>phần 4.8</a> trình
            bày từng trường một, và trên Metal là những gì kernel thăm dò tìm thấy về SIMD-group, mà trình tô warp của{' '}
            <a href={href('ch04', '4.4')}>phần 4.4</a> vẽ ra.
          </p>
        }
      >
        <p>
          The same run records what your queries returned, which <a href={href('ch04', '4.8')}>section 4.8</a> sets out
          field by field, and on Metal what the probe found about SIMD-groups, which the warp painter of{' '}
          <a href={href('ch04', '4.4')}>section 4.4</a> draws.
        </p>
      </Translated>
      <h3><Translated vi="Mốc dự án M1">Project milestone M1</Translated></h3>
      <Translated
        vi={
          <p>
            Kernel của lab 3.3 và lần khởi chạy của lab này cùng nhau tạo nên mốc dự án M1 của dự án xuyên suốt: một lớp tuyến
            tính có lần khởi chạy được suy ra từ những gì thiết bị báo cáo, không có gì về thiết bị được viết cứng trong mã
            nguồn. Lab 5.3 thay kernel bằng một kernel phân ô có cùng đối số; các quy tắc khởi chạy được giữ nguyên.
          </p>
        }
      >
        <p>
          Lab 3.3's kernel and this lab's launch together are milestone M1 of the through-line project: a linear layer
          whose launch is derived from what the device reports, with nothing about the device written into the source.
          Lab 5.3 replaces the kernel with a tiled one of the same arguments; the launch rules carry over.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Dự án xuyên suốt, <code>nanoinfer</code>, là của chúng tôi: một đường suy luận GPU nhỏ cho một transformer chỉ
              có decoder, lớn dần qua từng mốc.
            </p>
          }
        >
          <p>
            The through-line project, <code>nanoinfer</code>, is ours: one small GPU inference path for a decoder-only
            transformer, grown a milestone at a time.
          </p>
        </Translated>
      </Beyond>
      <MilestoneStatus />
      <ReferenceSolution lab={l1} tiers={LINEAR_TIERS} />

      {/* ------------------------------------------------------------ 4.2 */}
      <h2><Translated vi="Lab 4.2: một warp phân kỳ tốn bao nhiêu">Lab 4.2: what a divergent warp costs</Translated></h2>
      <Translated
        vi={
          <p>
            <a href={href('ch04', '4.5')}>Phần 4.5</a> đếm số lượt một warp phải chạy khi các luồng của nó bất đồng tại một
            nhánh. Nó không đưa ra giá. Lab này đo một cái giá. Bạn viết ba kernel: <code>branchKernel</code>, nơi mỗi luồng
            chạy một trong hai đường dài bằng nhau được chọn theo phần tử của nó trong <code>which</code> (dạng của Fig. 4.9, với
            dữ liệu thay cho <code>threadIdx.x &lt; 24</code>); <code>selectKernel</code>, cùng kết quả mà luồng điều khiển của
            không luồng nào phụ thuộc vào <code>which</code>; và <code>loopKernel</code>, nơi mỗi luồng lặp số lần của riêng nó
            (dạng của Fig. 4.10). Sau đó chương trình điều khiển xếp hai đường theo những mẫu khác nhau trên cùng dữ liệu.
          </p>
        }
      >
        <p>
          <a href={href('ch04', '4.5')}>Section 4.5</a> counts the passes a warp makes when its threads disagree at a
          branch. It gives no price. This lab measures one. You write three kernels: <code>branchKernel</code>, where
          each thread runs one of two equally long paths chosen by its element of <code>which</code> (the shape of Fig.
          4.9, with data in place of <code>threadIdx.x &lt; 24</code>); <code>selectKernel</code>, the same result with
          no thread's control flow depending on <code>which</code>; and <code>loopKernel</code>, where each thread loops
          its own number of times (the shape of Fig. 4.10). The driver then lays out the two paths in different
          patterns over the same data.
        </p>
      </Translated>
      <Source section="4.5" pages="79–82" figs={tr('Figs. 4.9 and 4.10; the kernels are ours', 'Figs. 4.9 và 4.10; các kernel là của chúng tôi')} />
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <Translated vi={<>ba kernel và <code>launchBranch</code>, <code>launchSelect</code>, <code>launchLoop</code></>}><>the three kernels and <code>launchBranch</code>, <code>launchSelect</code>, <code>launchLoop</code></></Translated>],
          ['kernel.metal', <Translated vi={<><code>branch_kernel</code>, <code>select_kernel</code> và <code>loop_kernel</code></>}><><code>branch_kernel</code>, <code>select_kernel</code> and <code>loop_kernel</code></></Translated>],
          ['launch.mm', <Translated vi={<><code>encodeBranch</code>, <code>encodeSelect</code> và <code>encodeLoop</code></>}><><code>encodeBranch</code>, <code>encodeSelect</code> and <code>encodeLoop</code></></Translated>],
        ]}
      />
      <Translated
        vi={
          <p>
            Hai đường được cho sẵn trong <code>paths.h</code> và <code>paths.metal.h</code>, cạnh các file của bạn và không cần
            sửa. Mỗi đường là một vòng lặp có trạng thái phụ thuộc dữ liệu, để trình biên dịch khó lặng lẽ biến nhánh thành
            select; không nhà cung cấp nào mô tả khi nào trình biên dịch làm vậy, nên đây là lập luận của chúng tôi, và benchmark
            là bằng chứng. Các mẫu được dựng từ độ rộng mà thiết bị của bạn báo cáo, không bao giờ từ hằng số 32: trên CUDA và
            HIP từ <code>warpSize</code>, trên Metal từ những gì một kernel thăm dò được cung cấp báo về việc luồng nào nằm trong
            SIMD-group nào, vì Apple để cách cắt đó cho cài đặt quyết định. Trình giả lập không chạy các luồng của một warp đồng
            bộ, nên nó chạy mọi mẫu ở các độ rộng giả định 4, 32 và 64 để kiểm tra giá trị của bạn.
          </p>
        }
      >
        <p>
          The two paths are given in <code>paths.h</code> and <code>paths.metal.h</code>, beside your files and not to be
          edited. Each is a loop with data-dependent state, so that the branch is hard for a compiler to turn into a
          select behind your back; no vendor documents when a compiler does that, so this is our reasoning, and the
          benchmark is the evidence. The patterns are built from the width your device reports, never from a literal 32: on CUDA
          and HIP from <code>warpSize</code>, on Metal from what a provided probe kernel reports about which thread sits
          in which SIMD-group, because Apple leaves that cut to the implementation. The emulator does not run the threads of a
          warp in step, so it runs every pattern at pretend widths of 4, 32 and 64 to check your values.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch04-facts.md, A9, B7, C4)', '2026-11-20 (notes/today/ch04-facts.md, mục A9, B7, C4)')}>
        <Translated
          vi={
            <p>
              NVIDIA và AMD đều mô tả các lượt: mỗi đường được thực thi trong khi các luồng khác bị che. Không bên nào đưa ra chi
              phí. Apple không mô tả cả cơ chế lẫn chi phí, nên trên Metal bảng của bạn là bằng chứng duy nhất.
            </p>
          }
        >
          <p>
            NVIDIA and AMD both document the passes: each path executed with the other threads masked off. Neither gives
            a cost. Apple documents neither the mechanism nor its cost, so on Metal your table is the only evidence.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 4.2 --backend emu</code>, rồi <code>python3 labs/pmpp.py run 4.2</code>. Các
            bài kiểm thử so cả ba kernel với tham chiếu CPU cho tám mẫu (toàn A, toàn B, nửa đầu mảng là A, warp xen kẽ, một
            nửa số lane của mỗi warp, lane xen kẽ, mỗi warp một lane, một nửa ngẫu nhiên có seed), ở các kích thước khối bằng
            một, hai và tám lần độ rộng, với độ dài có và không là bội số của khối. <code>loopKernel</code> được kiểm thử với
            số vòng lặp có cả 0, đã sắp xếp, xáo trộn và hằng số. Một bài kiểm thử có thể biết <code>selectKernel</code> của bạn
            tính đúng giá trị, nhưng không biết nó có tránh được nhánh không: chỉ benchmark cho thấy điều đó.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 4.2 --backend emu</code>, then <code>python3 labs/pmpp.py run 4.2</code>.
          The tests compare all three kernels with a CPU reference for eight patterns (all A, all B, the first half of the
          array A, alternating warps, half of each warp's lanes, alternating lanes, one lane per warp, a seeded random
          half), at block sizes of one, two and eight widths, with lengths that are and are not multiples of the block.{' '}
          <code>loopKernel</code> is tested with trip counts that include 0, sorted, shuffled and constant. A test can
          tell that your <code>selectKernel</code> computes the right values, not whether it avoids the branch: only the
          benchmark shows that.
        </p>
      </Translated>
      <LabPrediction
        id="ch04/4.5/p-divcost"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy benchmark (cùng câu dự đoán như ở phần 4.5: chốt nó ở trang nào cũng được). Cùng một công việc,
                với hai đường trộn nửa nọ nửa kia bên trong mỗi warp thay vì được sắp xếp để mỗi warp đồng nhất. Trên GPU của
                bạn nó sẽ mất thời gian gấp bao nhiêu lần?
              </>
            }
          >
            <>
              Before you run the benchmark (the same prediction as in section 4.5: commit it on either page). The same
              work, with the two paths mixed half and half inside every warp instead of sorted so that each warp is
              uniform. How many times as long will it take on your GPU?
            </>
          </Translated>
        }
        unit={tr('times as long', 'lần thời gian')}
        measure={divergenceCost}
        missing={
          <Translated
            vi={
              <>
                Khi các kernel của bạn đã qua, hãy chạy <code>python3 labs/pmpp.py run 4.2 --bench</code> trên một máy có
                GPU, rồi tải lại.
              </>
            }
          >
            <>
              Once your kernels pass, run <code>python3 labs/pmpp.py run 4.2 --bench</code> on a machine with a GPU, then
              reload.
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                Phép đếm lượt của phần 4.5 nói là hai lượt trong khi một warp đồng nhất chạy một lượt, với các đường dài và bằng
                nhau. Một thiết bị có thể ra thấp hơn hoặc cao hơn số lượt của nó, và không nhà cung cấp nào công bố con số để
                so sánh.
              </>
            }
          >
            <>
              Section 4.5's pass count says two passes where a uniform warp makes one, for long, equal paths. A device
              can come out lower or higher than its pass count, and no vendor publishes a figure to compare with.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch04/labs/p-lab2-select"
        prompt={
          <Translated
            vi={
              <>
                Trên cùng mẫu trộn đó, việc tính cả hai đường rồi select sẽ mất thời gian gấp bao nhiêu lần so với nhánh? (Dưới
                1 nghĩa là select nhanh hơn.)
              </>
            }
          >
            <>
              On that same mixed pattern, computing both paths and selecting will take how many times as long as the
              branch? (Below 1 means the select is faster.)
            </>
          </Translated>
        }
        unit={tr("× the branch's time", '× thời gian của nhánh')}
        measure={selectCost}
        missing={
          <Translated vi={<>Chạy <code>python3 labs/pmpp.py run 4.2 --bench</code> trên một máy có GPU, rồi tải lại.</>}>
            <>
              Run <code>python3 labs/pmpp.py run 4.2 --bench</code> on a machine with a GPU, then reload.
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                Select luôn làm việc của cả hai đường. Nhánh chỉ làm cả hai trong một warp chứa cả hai. Hãy so hàng của select
                trên một mẫu đồng nhất trong bảng: đó là cái giá của nó ở nơi nhánh không tốn thêm gì.
              </>
            }
          >
            <>
              The select always does both paths' work. The branch does both only in a warp that holds both. Compare the
              select's row on a uniform pattern in the table: that is its price where the branch costs nothing extra.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l2}
        filter={(r) => r.name === 'branch' || r.name === 'select' || r.name === 'loop'}
        cols={[
          { head: tr('Version', 'Phiên bản'), cell: (r) => r.name },
          { head: tr('Pattern', 'Mẫu'), cell: (r) => String(r.name === 'loop' ? tr(`trip counts ${String(r.order)}`, `số vòng lặp ${String(r.order)}`) : r.pattern) },
          rawCol(tr('Width', 'Độ rộng'), 'width'),
          msCol,
          numCol(tr('÷ same mix, sorted', '÷ cùng mẫu trộn, đã sắp xếp'), 'ratio_to_sorted'),
          numCol(tr('Select ÷ branch', 'Select ÷ nhánh'), 'ratio_to_branch'),
        ]}
      />
      <ReferenceSolution lab={l2} tiers={DIVERGENCE_TIERS} />

      {/* ------------------------------------------------------------ 4.3 */}
      <h2><Translated vi="Lab 4.3: săn lỗi rào chắn">Lab 4.3: barrier hunt</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn được giao một kernel đảo ngược tại chỗ phần mảng của mỗi khối: mọi luồng đọc phần tử của mình vào một thanh ghi,
            rồi ghi nó vào vị trí đối xứng trong phần của nó. Nó biên dịch được, và có ba lỗi độc lập, cùng ba lỗi đó ở cả bản
            CUDA lẫn bản Metal. Hãy tìm và sửa cả ba. Chúng là những loại lỗi mà một phép hoán đổi tại chỗ giữa các luồng hay
            gây ra: một về thứ tự đọc và ghi, một về kích thước khối, một về khối cuối chưa đầy.
          </p>
        }
      >
        <p>
          You are handed a kernel that reverses each block's slice of an array in place: every thread reads its element
          into a register, then writes it to the mirrored position of its slice. It compiles, and it has three
          independent defects, the same three in the CUDA and the Metal version. Find and fix all three. They are the
          kinds an in-place exchange between threads invites: one about the order of reads and writes, one about the
          block size, one about the last, partial block.
        </p>
      </Translated>
      <Source section="4.3" pages="72" figs={tr('the any-order rule §4.4, p. 74; exercise 5, p. 91; the kernel is ours', 'quy tắc thứ tự bất kỳ §4.4, p. 74; bài tập 5, p. 91; kernel là của chúng tôi')} />
      <LabFiles
        lab={l3}
        files={[
          ['kernel.cu', <Translated vi={<><code>reverseBlocksKernel</code>, với ba lỗi; lần khởi chạy của nó là đúng</>}><><code>reverseBlocksKernel</code>, with the three defects; its launch is correct</></Translated>],
          ['kernel.metal', <Translated vi={<><code>reverse_blocks</code>, với cùng ba lỗi đó</>}><><code>reverse_blocks</code>, with the same three</></Translated>],
          ['launch.mm', <Translated vi={<>đúng như khi phát hành; không có gì phải sửa</>}><>correct as shipped; nothing to change</></Translated>],
        ]}
      />
      <Translated vi={<p>Phương pháp quan trọng hơn các bản sửa:</p>}>
        <p>The method matters more than the fixes:</p>
      </Translated>
      <Translated
        vi={
          <ol>
            <li>
              Trình giả lập trước: <code>python3 labs/pmpp.py run 4.3 --backend emu</code>. Nó chạy một luồng cho đến khi tới
              một rào chắn hoặc return, rồi luồng kế tiếp, theo ba thứ tự cố định, nên một rào chắn bị thiếu sẽ sai mọi lần. Một
              lần ghi ra ngoài mảng dừng lần chạy với báo cáo AddressSanitizer: hãy đọc <code>WRITE</code>, dòng{' '}
              <code>kernel.cu</code> của frame đầu tiên, và địa chỉ vượt quá vùng bao nhiêu byte.
            </li>
            <li>
              Sửa một thứ rồi chạy lại. Những kích thước khối nào hỏng, và độ dài có là bội số của khối không, là bằng chứng của
              bạn cho lỗi kế tiếp.
            </li>
            <li>
              <code>python3 labs/pmpp.py run 4.3 --backend race</code> chạy các luồng đồng thời dưới ThreadSanitizer và gọi tên
              hai dòng mã nguồn của một lần đọc và ghi không đồng bộ.
            </li>
            <li>
              Rồi đến GPU: <code>python3 labs/pmpp.py run 4.3</code>. Mảng nằm giữa các phần tử canh gác, mỗi hình dạng được
              khởi chạy 40 lần trên dữ liệu mới, và một trường hợp hỏng cho biết nó sai trong bao nhiêu lần trên 40.
            </li>
          </ol>
        }
      >
        <ol>
          <li>
            Emulator first: <code>python3 labs/pmpp.py run 4.3 --backend emu</code>. It runs one thread until it reaches
            a barrier or returns, then the next, under three fixed orders, so a missing barrier is wrong every time. A
            write outside the array stops the run with an AddressSanitizer report: read <code>WRITE</code>, the first
            frame's <code>kernel.cu</code> line, and how many bytes past the region the address is.
          </li>
          <li>
            Fix one thing and run again. Which block sizes fail, and whether the length is a multiple of the block, is
            your evidence for the next defect.
          </li>
          <li>
            <code>python3 labs/pmpp.py run 4.3 --backend race</code> runs the threads concurrently under ThreadSanitizer
            and names the two source lines of an unsynchronised read and write.
          </li>
          <li>
            Then the GPU: <code>python3 labs/pmpp.py run 4.3</code>. The array sits between sentinel elements, each shape
            is launched 40 times on fresh data, and a failing case says in how many of the 40 it was wrong.
          </li>
        </ol>
      </Translated>
      <Translated
        vi={
          <p>
            Chương trình điều khiển chọn kích thước khối: 32, 48, 64 và 96 luồng, rồi một, hai và bốn lần độ rộng mà thiết bị
            của bạn báo cáo, mỗi cái với độ dài có và không là bội số của khối. Rào chắn của Metal nhận một cờ chỉ ra vùng nhớ
            mà nó sắp thứ tự; quyết định cờ nào thuộc về đây là một phần của lab.
          </p>
        }
      >
        <p>
          The driver picks the block sizes: 32, 48, 64 and 96 threads, then one, two and four times the width your
          device reports, each with lengths that are and are not multiples of the block. The Metal barrier takes a flag
          that names the memory it orders; deciding which one belongs here is part of the lab.
        </p>
      </Translated>
      <RaceStatus lab={l3} />
      <h3><Translated vi="Thí nghiệm của bài tập 5">The experiment of exercise 5</Translated></h3>
      <Translated
        vi={
          <p>
            Bài tập 5 cho rằng với 32 luồng trong một khối thì có thể bỏ rào chắn. Khi mọi bài kiểm thử của bạn đã qua, hãy xóa
            rào chắn lần nữa, chạy trên trình giả lập (mọi trường hợp đều hỏng, theo mọi thứ tự), rồi trên GPU của bạn năm lần,
            ghi lại kích thước khối nào hỏng và hỏng trong bao nhiêu trên 40 lần khởi chạy. Rồi đặt rào chắn trở lại.
          </p>
        }
      >
        <p>
          Exercise 5 claims that with 32 threads in a block the barrier can be left out. Once all your tests pass, delete
          the barrier again, run on the emulator (every case fails, under every order), then on your GPU five times,
          noting which block sizes fail and in how many of their 40 launches. Then put the barrier back.
        </p>
      </Translated>
      <BarrierPrediction />
      <Translated
        vi={
          <p>
            Dù GPU của bạn làm gì, hãy đọc nó đối chiếu với những gì các nhà cung cấp ghi trong tài liệu, vì đó là điều thiết bị
            tiếp theo có quyền làm. Độ rộng không phải hằng số. NVIDIA không hứa các luồng của một warp chạy đồng bộ, AMD thì
            hứa, và Apple không nói gì. Và một khối rộng hơn một warp là nhiều warp, thứ mà chỉ có rào chắn sắp thứ tự được với
            nhau.
          </p>
        }
      >
        <p>
          Whatever your GPU does, read it against what the vendors document, because that is what the next device is
          entitled to do. The width is not a constant. NVIDIA does not promise that the threads of a warp stay in step,
          AMD does, and Apple says nothing either way. And a block wider than one warp is several warps, which nothing
          but a barrier orders against each other.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch04-facts.md, A8, A9, A10, B5, B6, B8, C2)', '2026-11-20 (notes/today/ch04-facts.md, mục A8, A9, A10, B5, B6, B8, C2)')}>
        <Translated
          vi={
            <p>
              NVIDIA: từ compute capability 7.0, mỗi luồng có bộ đếm chương trình riêng; hãy đồng bộ hóa tường minh mã đồng bộ
              theo warp. AMD: "all warp lanes are executed in lockstep" (mọi lane của warp được thực thi lockstep). Apple:
              SIMD-group "make independent forward progress" (tiến triển độc lập) khi không có rào chắn threadgroup, và không có
              độ rộng nào được ghi trong tài liệu.
            </p>
          }
        >
          <p>
            NVIDIA: from compute capability 7.0 each thread has its own program counter; synchronise warp-synchronous
            code explicitly. AMD: "all warp lanes are executed in lockstep". Apple: SIMD-groups "make independent
            forward progress" without a threadgroup barrier, and no width is documented.
          </p>
        </Translated>
      </Today>
      <LabResults
        lab={l3}
        noBench={tr(
          'This lab has no benchmark. The result is the line above: every test passing on the emulators and on a device. As shipped, the kernel is meant to fail.',
          'Lab này không có benchmark. Kết quả là dòng ở trên: mọi bài kiểm thử đều qua trên các trình giả lập và trên thiết bị. Ở dạng được phát hành, kernel được thiết kế để thất bại.',
        )}
      />
      <ReferenceSolution lab={l3} tiers={BARRIER_TIERS} />

      {/* ------------------------------------------------------------ 4.4 */}
      <h2><Translated vi="Lab 4.4: kích thước khối trên một kernel phải chờ bộ nhớ">Lab 4.4: block size on a kernel that waits on memory</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn viết một phép gather có dạng embedding bag: đầu ra <code>i</code> là tổng của <code>k</code> phần tử bảng được
            chọn bởi một mảng chỉ số, <code>out[i] = Σ table[idx[i*k + j]]</code>. Các chỉ số là ngẫu nhiên và bảng lớn, nên gần
            như mỗi bước là một lần nạp từ bộ nhớ toàn cục mà không gì đoán trước được. Đó chính là kernel mà{' '}
            <a href={href('ch04', '4.6')}>phần 4.6</a> nói tới: nó dành thời gian để chờ, và thứ nó cần từ SM là các warp khác để
            chạy trong lúc đó. Kích thước khối là đối số lúc chạy của lần khởi chạy, và benchmark quét qua nó.
          </p>
        }
      >
        <p>
          You write a gather in the shape of an embedding bag: output <code>i</code> is the sum of <code>k</code> table
          entries chosen by an index array, <code>out[i] = Σ table[idx[i*k + j]]</code>. The indices are random and the
          table is large, so nearly every step is a load from global memory that nothing predicts. That is the kernel{' '}
          <a href={href('ch04', '4.6')}>section 4.6</a> is about: it spends its time waiting, and what it needs from the
          SM is other warps to run meanwhile. The block size is a run-time argument of the launch, and the benchmark
          sweeps it.
        </p>
      </Translated>
      <Source section="4.7" pages="85–87" figs={tr('latency tolerance §4.6, pp. 82–85; exercises 6 to 8, pp. 91–92; the kernel is ours', 'khả năng che độ trễ §4.6, pp. 82–85; bài tập 6 đến 8, pp. 91–92; kernel là của chúng tôi')} />
      <Translated
        vi={
          <p>
            Trên CUDA, HIP và trình giả lập, bạn cũng viết phép tính occupancy của sách thành hai hàm,{' '}
            <code>modelBlocksPerSm</code> và <code>modelOccupancy</code>: ô khối, ô luồng tính theo số warp nguyên, và thanh
            ghi, với giới hạn bằng 0 nghĩa là không được báo cáo. Benchmark đặt mô hình của bạn cạnh số đo và cạnh hàm occupancy
            của chính nhà cung cấp.
          </p>
        }
      >
        <p>
          On CUDA, HIP and the emulator you also write the book's occupancy arithmetic as two functions,{' '}
          <code>modelBlocksPerSm</code> and <code>modelOccupancy</code>: block slots, thread slots counted in whole warps,
          and registers, with a limit of 0 meaning not reported. The benchmark sets your model beside the measurement and
          beside the vendor's own occupancy function.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Embedding bag là cầu nối của chúng tôi: nó có dạng của phép tra embedding token ở đáy một mô hình, và của mọi
              kernel có các lần nạp không gì đoán trước được.
            </p>
          }
        >
          <p>
            An embedding bag is ours as a bridge: it is the shape of the token-embedding lookup at the bottom of a model,
            and of any kernel whose loads nothing predicts.
          </p>
        </Translated>
      </Beyond>
      <LabFiles
        lab={l4}
        files={[
          ['kernel.cu', <Translated vi={<><code>gatherSumKernel</code>, <code>launchGatherSum</code>, <code>modelBlocksPerSm</code> và <code>modelOccupancy</code></>}><><code>gatherSumKernel</code>, <code>launchGatherSum</code>, <code>modelBlocksPerSm</code> and <code>modelOccupancy</code></></Translated>],
          ['kernel.metal', <><code>gather_sum</code></>],
          ['launch.mm', <Translated vi={<><code>encodeGatherSum</code>, với kích thước threadgroup là một đối số</>}><><code>encodeGatherSum</code>, with the threadgroup size as an argument</></Translated>],
        ]}
      />
      <Translated
        vi={
          <p>
            <code>sm_limits.h</code> bên cạnh chứa struct giới hạn và được cho sẵn. File{' '}
            <code>provided/occupancy_api.cu</code> là của chương trình điều khiển: trên CUDA và HIP nó hỏi runtime của nhà cung
            cấp xem bao nhiêu khối kernel của bạn vừa trên một SM. Nó chưa bao giờ được biên dịch bởi nvcc hay trình biên dịch
            HIP, nên nó là unverified trên cả hai; nếu nó không dựng được với toolchain của bạn, <code>README.md</code> của lab
            hướng dẫn cách cấu hình không dùng nó.
          </p>
        }
      >
        <p>
          <code>sm_limits.h</code> beside them holds the limits struct and is given. The file{' '}
          <code>provided/occupancy_api.cu</code> is the driver's: on CUDA and HIP it asks the vendor's runtime how many
          blocks of your kernel fit on one SM. It has never been compiled by nvcc or a HIP compiler, so it is unverified on
          both; if it does not build with your toolchain, the lab's <code>README.md</code> says how to configure without it.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 4.4</code>, bắt đầu với <code>--backend emu</code>. Kernel được so với tham
            chiếu CPU ở nhiều kích thước và kích thước khối, phần lớn để lại một khối cuối chưa đầy, và trên trình giả lập
            AddressSanitizer dừng ở lần truy cập đầu tiên ra ngoài buffer. Mô hình được kiểm thử trên những giới hạn không cần ai
            sở hữu: bảy kích thước khối được giải trong sách, 31, 33 và 64 thanh ghi mỗi luồng, bài tập 6, 7 và 8, một thiết bị
            có 1536 ô luồng và 16 ô khối, các khối không phải số nguyên warp, độ rộng 64, và các giới hạn bằng 0. Trong bảng kết
            quả, các cột cho mô hình của bạn và hàm của nhà cung cấp để trống trên Metal và ở bất cứ đâu thiết bị không báo cáo
            một giới hạn.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 4.4</code>, starting with <code>--backend emu</code>. The kernel is compared
          with a CPU reference for several sizes and block sizes, most leaving a partial last block, and on the emulator
          AddressSanitizer stops at the first access outside a buffer. The model is tested on limits nobody has to own:
          the book's seven worked block sizes, 31 and 33 and 64 registers per thread, exercises 6, 7 and 8, a device with
          1536 thread slots and 16 block slots, blocks that are not whole warps, a width of 64, and limits of 0. In the
          results table, the columns for your model and the vendor's function stay empty on Metal and wherever the device
          left a limit unreported.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch04-facts.md, A13, B9, B10, C5)', '2026-11-20 (notes/today/ch04-facts.md, mục A13, B9, B10, C5)')}>
        <Translated
          vi={
            <p>
              Thanh ghi là phần yếu nhất của mô hình: NVIDIA không công bố độ chi tiết cấp phát, AMD cấp phát theo wavefront và
              trường ô khối của họ được tính từ ô luồng. Metal hoàn toàn không có con số occupancy, nên ở đó benchmark chỉ là một
              lần quét thời gian và không hơn.
            </p>
          }
        >
          <p>
            Registers are the model's weakest part: NVIDIA does not publish the granularity, AMD allocates per wavefront
            and its block-slot field is computed from the thread slots. Metal has no occupancy figure at all, so there the
            benchmark is a timing sweep and nothing else.
          </p>
        </Translated>
      </Today>
      <LabPrediction
        id="ch04/4.7/p-blocksize"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy benchmark (cùng câu dự đoán như ở phần 4.7: chốt nó ở trang nào cũng được). Lần quét chạy cùng
                phép gather ở 1, 2, 3, 4, 6, 8, 12, 16, 24 và 32 lần độ rộng của thiết bị, tối đa đến khối lớn nhất nó chấp
                nhận. Kích thước khối chậm nhất sẽ mất thời gian gấp bao nhiêu lần kích thước nhanh nhất? (1 nghĩa là kích thước
                khối không tạo khác biệt.)
              </>
            }
          >
            <>
              Before you run the benchmark (the same prediction as in section 4.7: commit it on either page). The sweep
              runs the same gather at 1, 2, 3, 4, 6, 8, 12, 16, 24 and 32 times the device's width, up to the largest
              block it accepts. How many times as long will the slowest block size take as the fastest? (1 means block
              size makes no difference.)
            </>
          </Translated>
        }
        unit={tr('times', 'lần')}
        measure={blockSizeSpread}
        missing={
          <Translated
            vi={
              <>
                Khi kernel của bạn đã qua, hãy chạy <code>python3 labs/pmpp.py run 4.4 --bench</code> trên một máy có GPU, rồi
                tải lại.
              </>
            }
          >
            <>
              Once your kernel passes, run <code>python3 labs/pmpp.py run 4.4 --bench</code> on a machine with a GPU, then
              reload.
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                Một khoảng cách giữa hai kích thước không lớn hơn độ dao động giữa các lần chạy trong cùng dòng là do máy, không
                phải do kích thước khối. Một lần quét phẳng cũng là một kết quả: trên thiết bị mà ngay cả khối nhỏ cũng lấp đầy
                các ô, kích thước khối còn rất ít thứ để thay đổi.
              </>
            }
          >
            <>
              A gap between two sizes no larger than the run-to-run spread in the same line is the machine, not the block
              size. A flat sweep is a result too: on a device where even small blocks fill the slots, block size has
              little left to change.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l4}
        filter={(r) => r.name === 'sweep'}
        cols={[
          rawCol(tr('Threads per block', 'Số luồng mỗi khối'), 'threads_per_block'),
          rawCol(tr('× width', '× độ rộng'), 'multiple'),
          { ...msCol, head: tr('Best, ms', 'Tốt nhất, ms') },
          {
            head: tr('Median, ms', 'Trung vị, ms'),
            num: true,
            cell: (r) => {
              const v = num(r.median_seconds);
              return v == null ? '' : fmt(v * 1000);
            },
          },
          {
            head: tr('M lookups/s', 'Triệu lần tra/s'),
            num: true,
            cell: (r) => {
              const v = num(r.lookups_per_s);
              return v == null ? '' : fmt(v / 1e6, 4);
            },
          },
          {
            head: tr('Your model', 'Mô hình của bạn'),
            num: true,
            cell: (r) => {
              const v = num(r.model_occupancy);
              return v == null ? '' : `${fmt(v * 100)}%`;
            },
          },
          rawCol(tr("Vendor's blocks per SM", 'Số khối mỗi SM theo nhà cung cấp'), 'vendor_blocks_per_sm'),
        ]}
      />
      <ReferenceSolution lab={l4} tiers={GATHER_TIERS} />
    </LabsPage>
  );
}
