import type { Question } from '../../components/Ask';
import { MATMUL_TIERS } from '../../components/Code';
import { Beyond, Source, Today, type Tier } from '../../components/notes';
import {
  BackendsAndEmulator,
  LabFiles,
  LabPrediction,
  LabResults,
  LabsIntro,
  LabsPage,
  ReferenceSolution,
  RaceStatus,
  RunningLabs,
  VerificationStatus,
  benchRow,
  defineLabs,
  measure,
  msCol,
  numCol,
  rawCol,
  useLabs,
  type Col,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 6's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch06', [
  { num: '6.1', dir: 'lab1_coalescing', name: 'coalescing', vi: { name: 'hợp nhất truy cập bộ nhớ' } },
  { num: '6.2', dir: 'lab2_vector', name: 'vector loads', vi: { name: 'nạp vector' } },
  { num: '6.3', dir: 'lab3_banks', name: 'bank conflicts', vi: { name: 'xung đột bank' } },
  { num: '6.4', dir: 'lab4_linear', name: 'linear layer (M2)', vi: { name: 'lớp tuyến tính (M2)' } },
  { num: '6.5', dir: 'lab5_doublebuf', name: 'double buffering', vi: { name: 'buffer kép' } },
]);
const [l1, l2, l3, l4, l5] = LABS;

// Identifier colours per lab: names mean different things in different labs (`in` is a
// buffer in lab 6.1 and a size in lab 6.4), so there is no page-wide map. Lab 6.2 has
// none: its buffers are x, y and z, and colouring those would colour every .x field.
const TIERS_61: Record<string, Tier> = { in: 'global', out: 'global', i: 'reg', row: 'reg', col: 'reg' };
const TIERS_63: Record<string, Tier> = { in: 'global', out: 'global', tile: 'shared', tx: 'reg', ty: 'reg', r: 'reg', c: 'reg', outRow: 'reg', outCol: 'reg' };
const TIERS_64: Record<string, Tier> = {
  X: 'global', W: 'global', Y: 'global', Xs: 'shared', Ws: 'shared',
  Yvalue: 'reg', tx: 'reg', ty: 'reg', Row: 'reg', Col: 'reg', ph: 'reg', xk: 'reg', wo: 'reg', wk: 'reg', i: 'reg',
};

// ---------------------------------------------------------------- reading the benchmark rows
// Several labs tell their rows apart by a string field (`variant` is the kernel's name);
// LabKit's benchRow() matches string and number fields alike.
const ms = (r?: BenchRow) => (num(r?.seconds) != null ? `${fmt(num(r!.seconds)! * 1000)} ms` : tr('no time', 'không có thời gian')); // i18n-ignore: number with unit ms

/** Lab 6.1: the strided read at stride 2, as a percentage of stride 1. */
const stride2 = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const two = benchRow(s, 'strided', { stride: 2 });
    const one = benchRow(s, 'strided', { stride: 1 });
    const v = num(two?.relative_to_stride1);
    if (!two || v == null) return null;
    return {
      value: v * 100,
      detail: tr(
        `${fmt(num(two.gb_per_s) ?? NaN)} GB/s at stride 2 against ${fmt(num(one?.gb_per_s) ?? NaN)} GB/s at stride 1`,
        `${fmt(num(two.gb_per_s) ?? NaN)} GB/s ở bước nhảy 2 so với ${fmt(num(one?.gb_per_s) ?? NaN)} GB/s ở bước nhảy 1`,
      ),
    };
  });
/** Lab 6.2: the float4 kernel's speed-up over one float per thread. */
const vec4Speedup = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const r = benchRow(s, 'vecadd', { variant: 'vec4' });
    const v = num(r?.speedup_vs_scalar);
    if (!r || v == null) return null;
    return {
      value: v,
      detail: tr(
        `${ms(r)} against ${ms(benchRow(s, 'vecadd', { variant: 'scalar' }))} at n = ${String(r.n)}`,
        `${ms(r)} so với ${ms(benchRow(s, 'vecadd', { variant: 'scalar' }))} ở n = ${String(r.n)}`,
      ),
    };
  });
/** Lab 6.3: the bank probe's time per access at stride 32, over stride 1. */
const probe32 = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const r = benchRow(s, 'bank_probe', { stride: 32 });
    const v = num(r?.relative_to_stride1);
    if (!r || v == null) return null;
    const one = benchRow(s, 'bank_probe', { stride: 1 });
    return {
      value: v,
      detail: tr(
        `${fmt(num(r.ns_per_access) ?? NaN)} ns per warp-wide access at stride 32, ${fmt(num(one?.ns_per_access) ?? NaN)} at stride 1, ${String(r.lanes)} lanes`,
        `${fmt(num(r.ns_per_access) ?? NaN)} ns mỗi lần truy cập toàn warp ở bước nhảy 32, ${fmt(num(one?.ns_per_access) ?? NaN)} ở bước nhảy 1, ${String(r.lanes)} lane`,
      ),
    };
  });
/** Lab 6.3: transpose time without padding over time with it, at T = 32. */
const padGain = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const plain = benchRow(s, 'transpose', { tile: 32, pitch: 32 });
    const padded = benchRow(s, 'transpose', { tile: 32, pitch: 33 });
    const a = num(plain?.seconds);
    const b = num(padded?.seconds);
    if (!a || !b) return null;
    return {
      value: a / b,
      detail: tr(
        `${ms(plain)} at pitch 32, ${ms(padded)} at pitch 33, on a ${String(plain!.rows)} × ${String(plain!.cols)} matrix`,
        `${ms(plain)} ở pitch 32, ${ms(padded)} ở pitch 33, trên ma trận ${String(plain!.rows)} × ${String(plain!.cols)}`,
      ),
    };
  });
/** Lab 6.4: the corner-turned, padded kernel over the provided baseline, batch 256, T = 16. */
const linearSpeedup = (reports: Reports) =>
  measure(reports, l4, (s) => {
    const r = benchRow(s, 'linear', { variant: 'tuned', tile: 16, batch: 256 });
    const v = num(r?.speedup_vs_baseline);
    if (!r || v == null) return null;
    return {
      value: v,
      detail: tr(
        `${fmt(num(r.gflops) ?? NaN)} GFLOP/s against the baseline's ${fmt(num(benchRow(s, 'linear', { variant: 'baseline', tile: 16, batch: 256 })?.gflops) ?? NaN)}, ${String(r.in)} → ${String(r.out)} layer`,
        `${fmt(num(r.gflops) ?? NaN)} GFLOP/s so với ${fmt(num(benchRow(s, 'linear', { variant: 'baseline', tile: 16, batch: 256 })?.gflops) ?? NaN)} của bản mốc, lớp ${String(r.in)} → ${String(r.out)}`,
      ),
    };
  });
/** Lab 6.5: one barrier per phase (double buffered) over two, at T = 16. */
const oneBarrier = (reports: Reports) =>
  measure(reports, l5, (s) => {
    const r = benchRow(s, 'matmul', { variant: 'double_buffered', tile: 16 });
    const v = num(r?.speedup_vs_two_barriers);
    if (!r || v == null) return null;
    return {
      value: v,
      detail: tr(
        `${ms(r)} against ${ms(benchRow(s, 'matmul', { variant: 'two_barriers', tile: 16 }))} at width ${String(r.width)}`,
        `${ms(r)} so với ${ms(benchRow(s, 'matmul', { variant: 'two_barriers', tile: 16 }))} ở độ rộng ${String(r.width)}`,
      ),
    };
  });

// ---------------------------------------------------------------- results columns
// Labs 6.1 and 6.3 record several kinds of row (a probe, a copy, the kernel itself, the
// SIMD width), so their tables say in the first column what each row measured.
const what61: Col = {
  get head() {
    return tr('Measurement', 'Phép đo');
  },
  cell: (r) =>
    r.name === 'copy'
      ? tr(`copy (provided), ${String(r.n)} floats`, `sao chép (có sẵn), ${String(r.n)} float`)
      : r.name === 'strided'
        ? tr(`strided read, stride ${String(r.stride)}`, `đọc theo bước nhảy, bước nhảy ${String(r.stride)}`)
        : r.name === 'transpose'
          ? tr(
              `naive transpose, ${r.kernel === 'write_rows' || r.variant === 'write_rows' ? 'rows written (strided loads)' : 'rows read (strided stores)'}, ${String(r.rows)} × ${String(r.cols)}`,
              `chuyển vị ngây thơ, ${r.kernel === 'write_rows' || r.variant === 'write_rows' ? 'ghi theo hàng (nạp theo bước nhảy)' : 'đọc theo hàng (ghi theo bước nhảy)'}, ${String(r.rows)} × ${String(r.cols)}`,
            )
          : r.name === 'simd_width'
            ? tr('SIMD width the probe used', 'Độ rộng SIMD mà phép thăm dò dùng')
            : r.name,
};
const what63: Col = {
  get head() {
    return tr('Measurement', 'Phép đo');
  },
  cell: (r) =>
    r.name === 'bank_probe'
      ? tr(`bank probe, stride ${String(r.stride)}`, `thăm dò bank, bước nhảy ${String(r.stride)}`)
      : r.name === 'copy'
        ? tr(`copy (provided), ${String(r.n)} floats`, `sao chép (có sẵn), ${String(r.n)} float`)
        : r.name === 'transpose'
          ? tr(
              `tiled transpose, T = ${String(r.tile)}, pitch ${String(r.pitch)}, ${String(r.rows)} × ${String(r.cols)}`,
              `chuyển vị phân ô, T = ${String(r.tile)}, pitch ${String(r.pitch)}, ${String(r.rows)} × ${String(r.cols)}`,
            )
          : r.name === 'simd_width'
            ? tr('SIMD width the probe used', 'Độ rộng SIMD mà phép thăm dò dùng')
            : r.name,
};
const resultCol: Col = {
  get head() {
    return tr('Result', 'Kết quả');
  },
  num: true,
  cell: (r) =>
    r.name === 'simd_width'
      ? String(r.width ?? '')
      : r.name === 'bank_probe'
        ? tr(`${fmt(num(r.ns_per_access) ?? NaN)} ns per access`, `${fmt(num(r.ns_per_access) ?? NaN)} ns mỗi lần truy cập`)
        : num(r.gb_per_s) != null
          ? `${fmt(num(r.gb_per_s)!)} GB/s`
          : '',
};
const vsStride1: Col = {
  get head() {
    return tr('Relative to stride 1', 'So với bước nhảy 1');
  },
  num: true,
  cell: (r) => {
    const v = num(r.relative_to_stride1);
    if (v == null) return '';
    return r.name === 'bank_probe' ? tr(`${fmt(v)} × the time`, `${fmt(v)} × thời gian`) : tr(`${fmt(v)} of the bandwidth`, `${fmt(v)} băng thông`);
  },
};
const msOrBlank: Col = { ...msCol, cell: (r) => (r.name === 'simd_width' ? '' : msCol.cell(r)) };

// ---------------------------------------------------------------- milestone M2
function MilestoneStatus() {
  const { reports, loading } = useLabs();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const pass = backends.filter((b) => labStatus(reports[b], l4.id, 'work') === 'pass');
  const name = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  return (
    <p data-milestone={loading ? 'loading' : pass.join(',') || 'none'}>
      {loading
        ? tr('Reading reports.', 'Đang đọc báo cáo.')
        : !backends.length
          ? tr('No lab results are recorded on this machine yet.', 'Máy này chưa ghi nhận kết quả lab nào.')
          : pass.length
            ? (
              <Translated
                vi={
                  <>
                    <code>linearKernel</code> của chính bạn vượt qua trên {pass.map(name).join(' và ')} ở máy này: lớp đã ở
                    bố cục của framework tại đó. Nó đã được tinh chỉnh hay chưa thì benchmark mới trả lời được.
                  </>
                }
              >
                <>Your own <code>linearKernel</code> passes on {pass.map(name).join(' and ')} on this machine: the layer is in the framework layout there. Whether it is tuned is the benchmark&rsquo;s to say.</>
              </Translated>
            )
            : (
              <Translated vi={<><code>linearKernel</code> của chính bạn chưa vượt qua trên backend nào của máy này.</>}>
                <>Your own <code>linearKernel</code> does not pass on any backend of this machine yet.</>
              </Translated>
            )}
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
              Năm lab, mỗi lab trong một thư mục riêng dưới <code>labs/ch06/</code>. Ba lab là phép đo một quy tắc mà bạn
              phát biểu được nhưng không dự đoán được trên thiết bị của mình: hợp nhất truy cập bộ nhớ (coalescing), nạp
              vector, xung đột bank (bank conflict). Lab thứ tư chuyển lớp tuyến tính của dự án sang bố cục trọng số mà các
              framework dùng và tinh chỉnh nó, mốc dự án M2. Lab thứ năm bỏ một rào chắn (barrier) khỏi kernel phân ô và để
              một bộ phát hiện race condition kiểm tra kết quả. Mã bạn không sửa, các bản mốc và các phép thăm dò, nằm
              trong thư mục <code>provided/</code> của mỗi lab.
            </>
          }
        >
          Five labs, each in its own directory under <code>labs/ch06/</code>. Three are measurements of a rule you can
          state but not predict on your device: coalescing, vector loads, bank conflicts. The fourth moves the
          project&rsquo;s linear layer to the weight layout frameworks use and tunes it, milestone M2. The fifth removes a
          barrier from the tiled kernel and has a race detector check the result. Code you do not edit, the baselines and
          the probes, sits in each lab&rsquo;s <code>provided/</code> folder.
        </Translated>
      </LabsIntro>
      <RunningLabs
        example="6.1"
        alsoShownIn={
          <Translated
            vi={
              <>
                các mục <a href={href('ch06', '6.1')}>6.1</a>, <a href={href('ch06', '6.3')}>6.3</a>,{' '}
                <a href={href('ch06', '6.4')}>6.4</a>, <a href={href('ch06', '6.7')}>6.7</a> và{' '}
                <a href={href('ch06', '6.9')}>6.9</a>
              </>
            }
          >
            <>
              sections <a href={href('ch06', '6.1')}>6.1</a>, <a href={href('ch06', '6.3')}>6.3</a>,{' '}
              <a href={href('ch06', '6.4')}>6.4</a>, <a href={href('ch06', '6.7')}>6.7</a> and{' '}
              <a href={href('ch06', '6.9')}>6.9</a>
            </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <Translated
        vi={
          <p>
            Trình giả lập (emulator) có một biến thể thứ hai, <code>race</code>, mà các lab 6.3, 6.4 và 6.5 cũng được dựng
            cho nó. Nó chạy các luồng giả lập đồng thời dưới ThreadSanitizer và báo hai luồng chạm vào cùng một phần tử mà
            không đồng bộ hóa, kèm cả hai dòng mã nguồn. Gọi nó bằng <code>--backend race</code>;{' '}
            <code>python3 labs/pmpp.py doctor</code> cho biết trình biên dịch của bạn có dựng được nó không. Các bảng trên
            trang này chỉ hiển thị biến thể thứ nhất.
          </p>
        }
      >
        <p>
          The emulator has a second flavour, <code>race</code>, which labs 6.3, 6.4 and 6.5 are also built for. It runs the emulated
          threads concurrently under ThreadSanitizer and reports two threads that touch the same element without
          synchronisation, with both source lines. Ask for it with <code>--backend race</code>;{' '}
          <code>python3 labs/pmpp.py doctor</code> says whether your compiler can build it. The tables on this page show
          the first flavour only.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Mọi chi phí chương này bàn tới, giao dịch bộ nhớ, lượt qua bank, rào chắn, chỉ tồn tại trên GPU. Các trình giả
            lập kiểm tra rằng mỗi kernel tính ra đúng giá trị và không chạm vào vùng nhớ nào nó không được chạm, và không nói
            gì về tốc độ của nó. Vì vậy mỗi lab yêu cầu bạn dự đoán trước khi chạy benchmark, và dự đoán được so với thiết bị
            của bạn, không bao giờ với thiết bị của sách.
          </p>
        }
      >
        <p>
          Every cost this chapter talks about, transactions, bank passes, barriers, exists only on a GPU. The emulators
          check that each kernel computes the right values and touches no memory it should not, and nothing about its
          speed. Each lab therefore asks you to predict before you benchmark, and the prediction is compared with your
          device, never with the book&rsquo;s.
        </p>
      </Translated>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 6.1 */}
      <h2>
        <Translated vi="Lab 6.1: hợp nhất truy cập bộ nhớ trên thiết bị của bạn">Lab 6.1: coalescing on your device</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn viết ba kernel chỉ khác nhau ở địa chỉ mà mỗi luồng chạm tới. Một phép đọc theo bước nhảy (stride): luồng{' '}
            <code>i</code> chép <code>in[i * stride]</code> sang <code>out[i]</code>, nên phép ghi liền mạch ở mọi bước nhảy
            còn phép đọc chỉ liền mạch ở bước nhảy 1. Rồi hai phép chuyển vị ngây thơ của một ma trận rows × cols, một bản
            đọc hàng và ghi cột, một bản đọc cột và ghi hàng. Một phép sao chép có sẵn, đọc và ghi liền mạch, là thước đo.
            Không cái nào trong số này là một listing trong sách: chúng là phép thăm dò quy tắc mà mục 6.1 phát biểu, và trên
            máy Mac chúng là cách duy nhất để biết quy tắc đó trông ra sao ở đó.
          </p>
        }
      >
        <p>
          You write three kernels that differ only in which address each thread touches. A strided read: thread{' '}
          <code>i</code> copies <code>in[i * stride]</code> to <code>out[i]</code>, so the write is contiguous at every
          stride and the read only at stride 1. Then two naive transposes of a rows × cols matrix, one that reads rows and
          writes columns, one that reads columns and writes rows. A provided copy, which reads and writes contiguously, is
          the yardstick. None of these is a listing in the book: they are probes of the rule section 6.1 states, and on a
          Mac they are the only way to learn what that rule looks like there.
        </p>
      </Translated>
      <Source section="6.1" pages="124–132" figs={tr('Figs. 6.2 and 6.3; stores, p. 130', 'Hình 6.2 và 6.3; phép ghi, p. 130')} />
      <LabFiles
        lab={l1}
        files={[
          ['kernel.cu', <Translated vi={<><code>stridedReadKernel</code>, <code>transposeReadRowsKernel</code>, <code>transposeWriteRowsKernel</code> và ba lần khởi chạy của chúng</>}><><code>stridedReadKernel</code>, <code>transposeReadRowsKernel</code>, <code>transposeWriteRowsKernel</code> and their three launches</></Translated>],
          ['kernel.metal', <Translated vi={<><code>strided_read</code>, <code>transpose_read_rows</code> và <code>transpose_write_rows</code>, đánh chỉ số bằng <code>int</code></>}><><code>strided_read</code>, <code>transpose_read_rows</code> and <code>transpose_write_rows</code>, indexing with <code>int</code></></Translated>],
          ['launch.mm', <Translated vi={<>các lệnh dispatch trong <code>encodeStridedRead</code>, <code>encodeTransposeReadRows</code> và <code>encodeTransposeWriteRows</code></>}><>the dispatches in <code>encodeStridedRead</code>, <code>encodeTransposeReadRows</code> and <code>encodeTransposeWriteRows</code></></Translated>],
        ]}
      />
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 6.1</code>, bắt đầu với <code>--backend emu</code>. Phép đọc theo bước nhảy
            được kiểm thử ở n = 1, 31, 32, 33 và 1000 với các bước nhảy 1, 2, 3, 8 và 32, và trên thiết bị còn ở n = 65536
            với bước nhảy 64. Đầu vào của nó chứa đúng <code>n * stride</code> float, nên trên trình giả lập AddressSanitizer
            chỉ ra dòng của bất kỳ phép đọc nào vượt quá nó. Các phép chuyển vị chạy trên 1 × 1, 1 × 37, 37 × 1, 32 × 32,
            33 × 65 và 100 × 7, và trên thiết bị trên 1000 × 513 và 2048 × 2048. Đầu ra khởi đầu là NaN, nên một phần tử bạn
            không bao giờ ghi sẽ thất bại, và phép so sánh là chính xác: các kernel này chỉ di chuyển giá trị chứ không tính
            gì. Với <code>--bench</code>, phép sao chép, phép đọc theo bước nhảy ở các bước nhảy 1 đến 64 trên một đầu vào
            2<sup>25</sup> float, và cả hai phép chuyển vị trên ma trận 4096 × 4096 được đo thời gian, và độ rộng SIMD mà
            thiết bị của bạn báo cáo được ghi lại.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 6.1</code>, starting with <code>--backend emu</code>. The strided read is
          tested at n = 1, 31, 32, 33 and 1000 with strides 1, 2, 3, 8 and 32, and on a device also at n = 65536 with
          stride 64. Its input holds exactly <code>n * stride</code> floats, so on the emulator AddressSanitizer names the
          line of any read past it. The transposes run on 1 × 1, 1 × 37, 37 × 1, 32 × 32, 33 × 65 and 100 × 7, and on a
          device on 1000 × 513 and 2048 × 2048. Outputs start as NaN, so an element you never write fails, and the
          comparison is exact: these kernels move values and compute nothing. With <code>--bench</code> the copy, the
          strided read at strides 1 to 64 over one input of 2<sup>25</sup> floats, and both transposes on a 4096 × 4096
          matrix are timed, and the SIMD width your device reports is recorded.
        </p>
      </Translated>
      <LabPrediction
        id="ch06/labs/p-lab1-stride2"
        prompt={
          <Translated
            vi={
              <>
                Trước khi bạn chạy benchmark. Con số lab in ra cho phép đọc theo bước nhảy là số byte hữu ích (đọc cộng ghi)
                mỗi giây, so với bước nhảy 1. Ở bước nhảy 2, bạn sẽ đo được bao nhiêu phần trăm của bước nhảy 1? Trên NVIDIA,
                một lần nạp bước nhảy 2 dùng một nửa của mỗi sector 32 byte mà nó nạp, và phép ghi liền mạch ở mọi bước nhảy:
                hãy bắt đầu từ đó, rồi quyết định bộ nhớ đệm (cache) của bạn có thể thay đổi điều gì.
              </>
            }
          >
            <>
              Before you run the benchmark. The figure the lab prints for the strided read is useful bytes (read plus
              written) per second, relative to stride 1. At stride 2, what percentage of stride 1 will you measure? On
              NVIDIA a stride-2 load uses half of every 32-byte sector it fetches, and the write is contiguous at every
              stride: start from there, then decide what your caches might change.
            </>
          </Translated>
        }
        unit={tr('% of stride 1', '% của bước nhảy 1')}
        measure={stride2}
        reading={
          <Translated
            vi={
              <>
                Nếu thời gian đi theo số byte được di chuyển, mô hình sector của NVIDIA sẽ đặt con số này ở hai phần ba, không
                phải một nửa: bước nhảy 1 di chuyển 8 byte mỗi phần tử cho 8 byte được dùng, bước nhảy 2 di chuyển 12 cho 8
                được dùng, vì phép nạp lấy mọi sector còn phép ghi không đổi. Một phép đo gần con số đó cho thấy thiết bị của
                bạn nạp những đơn vị trọn vẹn cỡ chừng ấy và không có gì trên chip cứu được nửa không dùng. Apple không công
                bố đơn vị nào cả, nên trên máy Mac con số này là mô tả duy nhất của quy tắc mà bạn có. Hãy đọc phần còn lại
                của dải quét trong bảng: đường cong ngừng giảm ở đâu, và ở bước nhảy nào?
              </>
            }
          >
            <>
              If time followed the bytes moved, NVIDIA&rsquo;s sector model would put this figure at two thirds, not
              one half: stride 1 moves 8 bytes per element for 8 used, stride 2 moves 12 for 8 used, because the load
              fetches every sector and the store does not change. A measurement near that says your device fetches whole
              units of about that size and nothing on chip rescues the unused half. Apple documents no unit at all, so
              on a Mac this number is the only description of the rule you have. Read the rest of the sweep in the table:
              where does the curve stop falling, and at what stride?
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Dự đoán thứ hai cho lab này, phép chuyển vị ngây thơ nào nhanh hơn, nằm ở{' '}
            <a href={href('ch06', '6.1')}>mục 6.1</a> dưới tiêu đề &ldquo;Phép ghi cũng tính&rdquo;, nơi sách đưa ra nhận
            định về phép ghi. Nó đọc cùng các hàng benchmark.
          </p>
        }
      >
        <p>
          The second prediction for this lab, which naive transpose is faster, is in{' '}
          <a href={href('ch06', '6.1')}>section 6.1</a> under &ldquo;Stores count too&rdquo;, where the book&rsquo;s
          claim about stores is made. It reads the same benchmark rows.
        </p>
      </Translated>
      <LabResults lab={l1} cols={[what61, msOrBlank, resultCol, vsStride1]} />
      <ReferenceSolution lab={l1} tiers={TIERS_61} />

      {/* ------------------------------------------------------------ 6.2 */}
      <h2>
        <Translated vi="Lab 6.2: nạp vector, xử lý biên cho đúng">Lab 6.2: vector loads, with the boundaries done right</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hình 6.12 cho mỗi luồng của phép cộng vector cộng bốn cặp float với một lần nạp 16 byte cho mỗi đầu vào và một
            lần ghi 16 byte, và để phần biên cho Bài tập 4: như được in, nó đọc quá cuối mảng mỗi khi độ dài không phải bội
            của bốn. Sau đó mục 6.5 gọi chính kernel đó là một ví dụ về gộp việc cho luồng (thread coarsening). Lab này tách
            hai ý tưởng ra. Bạn viết hai kernel trong đó mỗi luồng sở hữu cùng bốn phần tử: một kernel di chuyển chúng bằng
            một <code>float4</code> cho mỗi mảng (bốn phần tử mỗi lệnh), kernel kia bằng vòng lặp gộp việc của mục 6.5 và một
            float mỗi lần truy cập (bốn phần tử mỗi luồng). Một kernel có sẵn với một float mỗi luồng là bản mốc.
          </p>
        }
      >
        <p>
          Figure 6.12 makes each thread of vector addition add four pairs of floats with one 16-byte load per input and
          one 16-byte store, and leaves the boundaries to Exercise 4: as printed it reads past the end whenever the length
          is not a multiple of four. Section 6.5 then calls the same kernel an example of thread coarsening. This lab
          separates the two ideas. You write two kernels in which each thread owns the same four elements: one moves
          them with one <code>float4</code> per array (four elements per instruction), the other with section 6.5&rsquo;s
          coarsening loop and one float per access (four elements per thread). A provided kernel with one float per
          thread is the baseline.
        </p>
      </Translated>
      <Source
        section="6.3"
        pages="138–139"
        figs={tr('Fig. 6.12; Exercise 4, p. 155; coarsening, §6.5 pp. 141–143', 'Hình 6.12; Bài tập 4, p. 155; gộp việc, §6.5 pp. 141–143')}
      />
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <Translated vi={<><code>vecAdd4Kernel</code>, <code>vecAddCoarseKernel</code> và các lần khởi chạy của chúng</>}><><code>vecAdd4Kernel</code>, <code>vecAddCoarseKernel</code> and their launches</></Translated>],
          ['kernel.metal', <Translated vi={<><code>vec_add4</code>, nhìn mỗi buffer như <code>float4</code> cho các khúc trọn vẹn và như <code>float</code> cho phần đuôi, và <code>vec_add_coarse</code></>}><><code>vec_add4</code>, which sees each buffer as <code>float4</code> for full chunks and as <code>float</code> for the tail, and <code>vec_add_coarse</code></></Translated>],
          ['launch.mm', <Translated vi={<><code>encodeVecAdd4</code>, gắn mỗi buffer hai lần ở hai chỉ số, và <code>encodeVecAddCoarse</code></>}><><code>encodeVecAdd4</code>, which binds each buffer twice at two indices, and <code>encodeVecAddCoarse</code></></Translated>],
        ]}
      />
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 6.2</code>, bắt đầu với <code>--backend emu</code>. Cả hai kernel chạy với n
            = 1, 2, 3, 4, 5, 7, 8, 9, 255, 256, 257, 1023, 1024 và 1025, mọi số dư của n khi chia cho 4, và trên thiết bị còn
            với 2<sup>20</sup>, 2<sup>20</sup> + 3 và 10,000,001. Mỗi mảng chứa đúng n float, nên trên trình giả lập một{' '}
            <code>float4</code> vươn quá cuối mảng là một lỗi AddressSanitizer chỉ ra dòng; trên GPU cùng lỗi đó có thể
            không bị phát hiện. Đầu ra khởi đầu là NaN, nên phần đuôi bị bỏ sót sẽ thất bại. Các con trỏ chính là các vùng
            cấp phát, căn lề 16 byte: trình giả lập không phân biệt được một <code>float4</code> lệch lề với một cái đúng lề,
            nên không bài kiểm thử nào truyền <code>x + 1</code>. Với <code>--bench</code>, ba kernel được đo thời gian ở một
            kích thước, tăng từ 2<sup>22</sup> cho tới khi bản mốc mất hơn 5 ms, tối đa 2<sup>26</sup>.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 6.2</code>, starting with <code>--backend emu</code>. Both kernels run for n =
          1, 2, 3, 4, 5, 7, 8, 9, 255, 256, 257, 1023, 1024 and 1025, every remainder of n modulo 4, and on a device also
          for 2<sup>20</sup>, 2<sup>20</sup> + 3 and 10,000,001. Each array holds exactly n floats, so on the emulator a{' '}
          <code>float4</code> that reaches past the end is an AddressSanitizer error naming the line; on a GPU the same
          mistake may go unnoticed. The output starts as NaN, so a skipped tail fails. The pointers are the allocations
          themselves, 16-byte aligned: the emulator cannot tell a misaligned <code>float4</code> from an aligned one, so
          no test passes <code>x + 1</code>. With <code>--bench</code> the three kernels are timed at one size, grown from
          2<sup>22</sup> until the baseline takes over 5 ms, up to 2<sup>26</sup>.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md A11, B5, C3)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              Phép ép kiểu cần một con trỏ căn lề 16 byte. Blog của NVIDIA cho thấy điều đó, các hướng dẫn của họ thì không;
              tài liệu của AMD không mô tả nó. Trên Metal, cách được ghi trong tài liệu là khai báo buffer là{' '}
              <code>device const float4*</code>, đó là điều <code>vec_add4</code> làm.
            </p>
          }
        >
          <p>
            The cast needs a pointer aligned to 16 bytes. NVIDIA&rsquo;s blog shows it, not its guides; AMD&rsquo;s docs do
            not describe it. On Metal the documented route is to declare the buffer <code>device const float4*</code>,
            which is what <code>vec_add4</code> does.
          </p>
        </Translated>
      </Today>
      <LabPrediction
        id="ch06/labs/p-lab2-vec4"
        prompt={
          <Translated
            vi={
              <>
                Trước khi bạn chạy benchmark. Lập luận của sách là ít hơn 75% lệnh nạp cho cùng lượng dữ liệu (p. 139).
                Kernel một float vốn đã được hợp nhất, chạy ở occupancy đầy đủ và di chuyển 12 byte mỗi phép cộng. Bạn sẽ đo
                được mức tăng tốc bao nhiêu của kernel <code>float4</code> so với nó ở kích thước lớn nhất?
              </>
            }
          >
            <>
              Before you run the benchmark. The book&rsquo;s argument is 75% fewer load instructions for the same data
              (p. 139). The one-float kernel is already coalesced, runs at full occupancy and moves 12 bytes per
              addition. What speed-up of the <code>float4</code> kernel over it will you measure at the largest size?
            </>
          </Translated>
        }
        unit={tr('× the one-float kernel', '× kernel một float')}
        measure={vec4Speedup}
        reading={
          <Translated
            vi={
              <>
                Ít lệnh hơn chỉ tiết kiệm thời gian nếu lệnh là thứ kernel phải chờ. Một kernel vốn đã giữ hệ thống bộ nhớ
                bận rộn chẳng được lợi bao nhiêu khi phát yêu cầu thành ít mảnh hơn, rộng hơn; sách nêu trường hợp nó có lợi,
                một kernel không đạt được occupancy đầy đủ (p. 139). Rồi so sánh kernel gộp việc trong bảng: cùng bốn phần tử
                mỗi luồng, một float mỗi lần truy cập. Hãy áp dụng quy tắc của mục 6.1 cho một lệnh nạp của nó trước khi giải
                thích con số của nó. Mục <a href={href('ch06', '6.3')}>6.3</a> đọc cùng các hàng này.
              </>
            }
          >
            <>
              Fewer instructions save time only if instructions were what the kernel waited on. A kernel that already
              keeps the memory system busy has little to gain from issuing its requests in fewer, wider pieces; the book
              names the case where it pays, a kernel that cannot reach full occupancy (p. 139). Then compare the
              coarsened kernel in the table: same four elements per thread, one float per access. Apply section
              6.1&rsquo;s rule to one of its load instructions before you explain its number. Section{' '}
              <a href={href('ch06', '6.3')}>6.3</a> reads the same rows.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l2}
        filter={(r) => r.name === 'vecadd'}
        cols={[
          {
            head: tr('Kernel measured', 'Kernel được đo'),
            cell: (r) =>
              r.variant === 'scalar' ? tr('one float per thread (provided)', 'một float mỗi luồng (có sẵn)') : r.variant === 'vec4' ? tr('float4, four per instruction', 'float4, bốn mỗi lệnh') : r.variant === 'coarse4' ? tr('coarsened, four per thread', 'gộp việc, bốn mỗi luồng') : String(r.variant),
          },
          rawCol('n', 'n'),
          msCol,
          numCol(tr('GB/s (12 bytes per element)', 'GB/s (12 byte mỗi phần tử)'), 'gb_per_s'),
          numCol(tr('Speed-up over one float', 'Tăng tốc so với một float'), 'speedup_vs_scalar'),
        ]}
      />
      <ReferenceSolution lab={l2} />

      {/* ------------------------------------------------------------ 6.3 */}
      <h2>
        <Translated vi="Lab 6.3: gây ra xung đột bank, rồi chèn đệm để khử nó">Lab 6.3: provoke a bank conflict, then pad it away</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Ví dụ của mục 6.4 là phép ghi khi xoay góc (corner turning): các luồng của một warp ghi dọc một cột của một ô
            chia sẻ 32 × 32 và luồng nào cũng rơi vào bank 0. Sách chỉ cho thấy hai khai báo và không gì khác. Ở đây bạn viết
            một kernel mà toàn bộ việc của nó là phép xoay đó, một phép chuyển vị ma trận qua một ô chia sẻ: đọc ô đầu vào
            theo từng hàng, một rào chắn, ghi ô đầu ra theo từng hàng, nghĩa là đọc ô chia sẻ dọc theo một cột. Pitch của ô là
            một đối số lúc chạy, nên một kernel phục vụ cả bố cục không đệm (pitch T) lẫn bố cục có đệm (pitch T + 1). Hai
            công cụ đo được cung cấp sẵn: một phép thăm dò trong đó một warp đọc và ghi một mảng chia sẻ ở các bước nhảy của
            Bài tập 5, và một phép sao chép.
          </p>
        }
      >
        <p>
          Section 6.4&rsquo;s example is the corner-turning store: the threads of a warp write down a column of a 32 × 32
          shared tile and every one of them lands in bank 0. The book shows the two declarations and nothing else. Here
          you write a kernel whose whole job is that turn, a matrix transpose through a shared tile: read the input tile
          row by row, one barrier, write the output tile row by row, which means reading the shared tile down a column.
          The tile&rsquo;s pitch is a run-time argument, so one kernel serves both the unpadded layout (pitch T) and the
          padded one (pitch T + 1). Two instruments are provided: a probe in which one warp reads and writes a shared
          array at Exercise 5&rsquo;s strides, and a copy.
        </p>
      </Translated>
      <Source section="6.4" pages="139–141" figs={tr('Exercise 5, p. 155; turning a tile, Fig. 6.4', 'Bài tập 5, p. 155; xoay một ô, Hình 6.4')} />
      <LabFiles
        lab={l3}
        files={[
          ['kernel.cu', <Translated vi={<><code>tiledTransposeKernel</code> và <code>launchTiledTranspose</code>; ô chia sẻ đã được khai báo sẵn cho bạn</>}><><code>tiledTransposeKernel</code> and <code>launchTiledTranspose</code>; the shared tile is declared for you</></Translated>],
          ['kernel.metal', <Translated vi={<><code>tiled_transpose</code>, với độ rộng ô được cho lúc chạy</>}><><code>tiled_transpose</code>, with the tile width given at run time</></Translated>],
          ['launch.mm', <Translated vi={<>các threadgroup trong <code>encodeTiledTranspose</code></>}><>the threadgroups in <code>encodeTiledTranspose</code></></Translated>],
        ]}
      />
      <Translated
        vi={
          <p>
            Trước hết hãy làm Bài tập 5 bằng tay (mục <a href={href('ch06', '6.4')}>6.4</a> có bảng): các hàng của phép thăm
            dò theo cùng thứ tự. Rồi chạy <code>python3 labs/pmpp.py run 6.3</code>, bắt đầu với{' '}
            <code>--backend emu</code>. Phép chuyển vị của bạn chạy trên 1 × 1, 7 × 5, 16 × 16, 17 × 33, 64 × 48, 100 × 1 và
            1 × 100, và trên thiết bị trên 1000 × 1000 và 513 × 2049, mỗi trường hợp ở cả hai pitch, với đầu ra khởi đầu là
            NaN, và phép so sánh chính xác. Trên trình giả lập, một rào chắn bị thiếu sẽ thất bại ở ít nhất một trong ba thứ
            tự luồng, và <code>--backend race</code> chỉ ra hai dòng mà nó để không có thứ tự. Các bài kiểm thử không thấy
            được bạn đã dùng pitch nào: đầu ra như nhau dù thế nào, và chỉ benchmark mới phân biệt được. Với{' '}
            <code>--bench</code>, phép thăm dò chạy các bước nhảy 1, 2, 4, 7, 8, 12, 16, 24, 31, 32 và 33, và phép chuyển vị
            được đo thời gian ở T = 16 và 32 ở cả hai pitch bên cạnh phép sao chép.
          </p>
        }
      >
        <p>
          First work Exercise 5 by hand (section <a href={href('ch06', '6.4')}>6.4</a> has the table): the probe&rsquo;s
          rows are in the same order. Then run <code>python3 labs/pmpp.py run 6.3</code>, starting with{' '}
          <code>--backend emu</code>. Your transpose runs on 1 × 1, 7 × 5, 16 × 16, 17 × 33, 64 × 48, 100 × 1 and 1 × 100,
          and on a device on 1000 × 1000 and 513 × 2049, each at both pitches, against an output that starts as NaN, with
          an exact comparison. On the emulator a missing barrier fails in at least one of the three thread orders, and{' '}
          <code>--backend race</code> names the two lines it leaves unordered. The
          tests cannot see which pitch you used: the output is the same either way, and only the benchmark can tell. With{' '}
          <code>--bench</code> the probe runs strides 1, 2, 4, 7, 8, 12, 16, 24, 31, 32 and 33, and the transpose is timed
          at T = 16 and 32 at both pitches beside the copy.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md A12, B4, C2)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              NVIDIA: 32 bank, mỗi bank 4 byte, và các lane đọc cùng một từ 32 bit thì không xung đột. AMD: 32 hoặc 64 bank
              phục vụ 16 lane mỗi chu kỳ. Apple không công bố bank nào, và khuyên tránh để nhiều luồng đọc cùng một vị trí.
            </p>
          }
        >
          <p>
            NVIDIA: 32 banks of 4 bytes, and lanes reading one 32-bit word do not conflict. AMD: 32 or 64 banks serving 16
            lanes per cycle. Apple documents no banks, and advises against many threads reading one location.
          </p>
        </Translated>
      </Today>
      <LabPrediction
        id="ch06/labs/p-lab3-stride32"
        prompt={
          <Translated
            vi={
              <>
                Trước khi bạn chạy benchmark. Trong mô hình của NVIDIA, một warp ở bước nhảy 32 cần 32 lượt trong khi bước
                nhảy 1 cần một. Phép thăm dò đo thời gian một chuỗi truy cập phụ thuộc nhau của một warp, nên độ trễ và vòng
                lặp quanh mỗi lần truy cập có mặt trong mọi con số. Thiết bị của bạn sẽ mất thời gian gấp bao nhiêu lần mỗi
                lần truy cập ở bước nhảy 32 so với bước nhảy 1?
              </>
            }
          >
            <>
              Before you run the benchmark. In NVIDIA&rsquo;s model a warp at stride 32 needs 32 passes where stride 1
              needs one. The probe times a chain of dependent accesses by one warp, so latency and the loop around each
              access are in every number. How many times longer per access will your device take at stride 32 than at
              stride 1?
            </>
          </Translated>
        }
        unit={tr('× the time at stride 1', '× thời gian ở bước nhảy 1')}
        measure={probe32}
        reading={
          <Translated
            vi={
              <>
                Con số 32 đếm số lượt qua các bank; phép thăm dò còn đo mọi thứ khác, nên ngay cả trên NVIDIA tỉ lệ cũng bị
                pha loãng bởi chi phí cố định của mỗi lần truy cập. Hãy nhìn hình dạng của cả dải quét thay vì một hàng: các
                bước nhảy có chung ước số với 32 (2, 4, 8, 16, 24) có nổi bật so với các bước nhảy lẻ (7, 31, 33) không? Trên
                AMD, hãy nhớ 16 lane mỗi chu kỳ; trên máy Mac, không tài liệu nào nói nên chờ đợi điều gì, và đây là bằng
                chứng.
              </>
            }
          >
            <>
              The 32 counts passes through the banks; the probe measures everything else as well, so even on NVIDIA the
              ratio is diluted by the fixed cost of each access. Look at the shape of the whole sweep rather than one
              row: do strides that share a factor with 32 (2, 4, 8, 16, 24) stand out from the odd ones (7, 31, 33)? On
              AMD, remember the 16 lanes per cycle; on a Mac, no document says what to expect, and this is the evidence.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch06/labs/p-lab3-pad"
        prompt={
          <Translated
            vi={
              <>
                Và với phép chuyển vị ở T = 32, trường hợp của sách: thời gian của nó ở pitch 32 chia cho thời gian ở pitch
                33. (Lớn hơn 1 nghĩa là phần đệm làm nó nhanh hơn.)
              </>
            }
          >
            <>
              And for the transpose at T = 32, the book&rsquo;s case: its time at pitch 32 divided by its time at pitch 33.
              (Above 1 means the padding made it faster.)
            </>
          </Translated>
        }
        unit={tr('× the padded time', '× thời gian có đệm')}
        measure={padGain}
        missing={
          <Translated
            vi={
              <>
                Phần này cần các hàng chuyển vị T = 32. Chạy <code>python3 labs/pmpp.py run 6.3 --bench</code>; trên Metal
                chúng chỉ xuất hiện nơi pipeline chấp nhận 1024 luồng mỗi threadgroup.
              </>
            }
          >
            <>
              This needs the T = 32 transpose rows. Run <code>python3 labs/pmpp.py run 6.3 --bench</code>; on Metal they
              appear only where the pipeline accepts 1024 threads per threadgroup.
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                Xung đột nằm ở một lần đọc bộ nhớ chia sẻ mỗi phần tử, và phép chuyển vị còn đọc và ghi bộ nhớ toàn cục một
                lần mỗi phần tử. Hãy so sánh cả hai pitch với phép sao chép trong bảng: một phép chuyển vị vốn đã chạy gần
                tốc độ sao chép thì bố cục của nó chẳng còn gì để giành. Rồi làm cặp T = 16, sau khi tính ra một warp 32 lane
                chạm vào những gì khi khối rộng 16.
              </>
            }
          >
            <>
              The conflict is in one shared-memory read per element, and the transpose also reads and writes global
              memory once per element. Compare both pitches with the copy in the table: a transpose that already runs at
              about copy speed has nothing left for its layout to win. Then do the T = 16 pair, after working out what a
              32-lane warp touches when the block is 16 wide.
            </>
          </Translated>
        }
      />
      <LabResults lab={l3} cols={[what63, msOrBlank, resultCol, vsStride1]} />
      <ReferenceSolution lab={l3} tiers={TIERS_63} />

      {/* ------------------------------------------------------------ 6.4 */}
      <h2>
        <Translated vi="Lab 6.4: lớp tuyến tính trong bố cục của framework, xoay góc (mốc dự án M2)">
          Lab 6.4: the linear layer in the framework layout, corner-turned (milestone M2)
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lab 5.3 tính <code>C = A B</code> với B lưu dạng (k, l). Các framework không lưu một lớp tuyến tính như vậy:{' '}
            <code>nn.Linear</code> của PyTorch giữ trọng số dạng (out_features, in_features) và tính{' '}
            <code>y = x W^T</code>, và <code>ggml_mul_mat</code> của llama.cpp dùng cùng bố cục đó. Đọc theo cách của lab 5.3,{' '}
            <code>W^T</code> đó chính là N thứ tự cột (column-major) của sách (p. 128): các luồng kề nhau nạp những float cách
            nhau <code>in</code>.
          </p>
        }
      >
        <p>
          Lab 5.3 computed <code>C = A B</code> with B stored (k, l). Frameworks do not store a linear layer that way:
          PyTorch&rsquo;s <code>nn.Linear</code> keeps its weight as (out_features, in_features) and computes{' '}
          <code>y = x W^T</code>, and llama.cpp&rsquo;s <code>ggml_mul_mat</code> uses the same layout. Read the lab 5.3 way,
          that <code>W^T</code> is the book&rsquo;s column-major N (p. 128): neighbouring threads fetch floats{' '}
          <code>in</code> apart.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Bố cục trọng số lấy từ mã nguồn của PyTorch và llama.cpp (<code>notes/today/ch03-facts.md</code> D6). Kernel
              là lời giải của chúng tôi cho Bài tập 2; sách không đưa listing nào cho nó.
            </p>
          }
        >
          <p>
            The weight layout is from PyTorch&rsquo;s and llama.cpp&rsquo;s source (<code>notes/today/ch03-facts.md</code>{' '}
            D6). The kernel is our solution to Exercise 2; the book gives no listing for it.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Bạn viết <code>linearKernel(X, W, Y, batch, in, out)</code> tính <code>Y = X W^T</code> với hai thay đổi so với
            kernel lab 5.3 của bạn, cả hai ở ô W. Nạp nó theo kiểu xoay góc, để các <code>threadIdx.x</code> kề nhau đọc các
            float kề nhau của một hàng của W. Và đệm ô chia sẻ thành <code>[TILE_WIDTH][TILE_WIDTH + 1]</code>, vì phép ghi
            đã xoay ghi dọc theo một cột. Đây là Bài tập 2 của sách trong bố cục mà dự án cần.
          </p>
        }
      >
        <p>
          You write <code>linearKernel(X, W, Y, batch, in, out)</code> computing <code>Y = X W^T</code> with two changes
          to your lab 5.3 kernel, both in the W tile. Load it corner-turned, so neighbouring <code>threadIdx.x</code> read
          neighbouring floats of one row of W. And pad the shared tile to <code>[TILE_WIDTH][TILE_WIDTH + 1]</code>,
          because the turned store writes down a column. This is the book&rsquo;s Exercise 2 in the layout the project
          needs.
        </p>
      </Translated>
      <Source
        section="6.1"
        pages="129–130"
        figs={tr('Fig. 6.4; padding, §6.4 pp. 139–141; Exercise 2, p. 155', 'Hình 6.4; chèn đệm, §6.4 pp. 139–141; Bài tập 2, p. 155')}
      />
      <LabFiles
        lab={l4}
        files={[
          ['kernel.cu', <Translated vi={<>khai báo của ô W, phép nạp xoay góc của nó và dòng tích trong <code>linearKernel</code></>}><>the W tile&rsquo;s declaration, its corner-turned load and the product line in <code>linearKernel</code></></Translated>],
          ['kernel.metal', <Translated vi={<>cùng ba chỗ đó trong <code>linear</code></>}><>the same three places in <code>linear</code></></Translated>],
        ]}
      />
      <Translated
        vi={
          <p>
            <code>work/launch.mm</code> đã hoàn chỉnh. Chạy <code>python3 labs/pmpp.py run 6.4</code>, bắt đầu với{' '}
            <code>--backend emu</code>. Các hình dạng (batch, in, out) là 3 × 3 × 3, 1 × 1 × 1, 16 × 16 × 16, 32 × 48 × 16,
            17 × 33 × 9, một token qua một lớp 40 → 7, 5 × 1 × 9 và 33 × 65 × 50, và trên thiết bị còn 4 × 300 × 1000,
            257 × 129 × 511 và một token qua một lớp 1024 → 4096. Đầu ra khởi đầu được điền một giá trị đánh dấu và được so
            với một tham chiếu CPU trong phạm vi dung sai. Các bài kiểm thử chỉ kiểm tra kết quả: bản mốc có sẵn, cùng kernel
            phân ô nhưng đọc <code>W[Col * in + k]</code> không hợp nhất, cũng vượt qua chúng, và một kernel quên phần đệm
            cũng vậy. Phép nạp của bạn có được hợp nhất hay không sẽ hiện ra trong benchmark: với <code>--bench</code>, bản
            mốc và kernel của bạn lần lượt chạy trên lớp 1024 → 4096 của lab 5.3 ở batch 1, 16, 64, 256 và 1024, ở T = 16 và
            32.
          </p>
        }
      >
        <p>
          <code>work/launch.mm</code> is complete. Run <code>python3 labs/pmpp.py run 6.4</code>, starting with{' '}
          <code>--backend emu</code>. The shapes (batch, in, out) are 3 × 3 × 3, 1 × 1 × 1, 16 × 16 × 16, 32 × 48 × 16,
          17 × 33 × 9, one token through a 40 → 7 layer, 5 × 1 × 9 and 33 × 65 × 50, and on a device also 4 × 300 × 1000,
          257 × 129 × 511 and one token through a 1024 → 4096 layer. The output starts filled with a marker value and is
          compared with a CPU reference within a tolerance. The tests check results only: the provided baseline, the
          same tiled kernel reading <code>W[Col * in + k]</code> uncoalesced, passes them too, and so does a kernel that
          forgets the padding. Whether your load coalesces shows in the benchmark: with <code>--bench</code> the baseline
          and your kernel take turns on lab 5.3&rsquo;s 1024 → 4096 layer at batch 1, 16, 64, 256 and 1024, at T = 16
          and 32.
        </p>
      </Translated>
      <LabPrediction
        id="ch06/labs/p-lab4-speedup"
        prompt={
          <Translated
            vi={
              <>
                Trước khi bạn chạy benchmark. Ở T = 32, phép nạp W của một warp chạm 32 sector trong bản mốc và 4 khi xoay
                góc, theo mô hình của NVIDIA; ở T = 16 một warp 32 luồng phủ hai hàng của khối, nên hãy tính con số đó trước.
                Phép nạp W là một phần của mỗi pha. Bạn sẽ đo được mức tăng tốc bao nhiêu của kernel của bạn so với bản mốc ở
                batch 256, T = 16?
              </>
            }
          >
            <>
              Before you run the benchmark. At T = 32 one warp&rsquo;s W load touches 32 sectors in the baseline and 4
              corner-turned, in NVIDIA&rsquo;s model; at T = 16 a 32-thread warp covers two rows of the block, so work that
              figure out first. The W load is one part of each phase. What speed-up of your kernel over the baseline will
              you measure at batch 256, T = 16?
            </>
          </Translated>
        }
        unit={tr('× the baseline', '× bản mốc')}
        measure={linearSpeedup}
        reading={
          <Translated
            vi={
              <>
                Lab 5.2 cho thấy bộ nhớ đệm phục vụ lưu lượng mà mô hình đã tính phí. Ở đây, trong bản mốc, các warp khác
                của cùng khối cần phần còn lại của mọi sector mà một warp nạp, và chúng hỏi đến nó ngay sau đó. Mô hình đếm
                sector theo từng lệnh; thiết bị có thể phục vụ cả ô của khối từ một nắm dòng cache dù theo cách nào. Hãy so
                sánh các kích thước batch và cả hai độ rộng ô trong bảng trước khi giải thích con số, rồi bỏ cột đệm, chạy{' '}
                <code>--bench</code> lần nữa, và xem nó đã mua được gì.
              </>
            }
          >
            <>
              Lab 5.2 showed caches serving traffic the model charged for. Here, in the baseline, the other warps of the
              same block need the rest of every sector a warp fetches, and they ask for it moments later. The model
              counts sectors per instruction; the device may serve the block&rsquo;s whole tile from a handful of lines
              either way. Compare the batch sizes and both tile widths in the table before you explain the number, then
              remove the padding column, run <code>--bench</code> again, and see what it was buying.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l4}
        filter={(r) => r.name === 'linear'}
        cols={[
          {
            head: tr('Layer kernel', 'Kernel của lớp'),
            cell: (r) => (r.variant === 'baseline' ? tr('baseline (provided)', 'bản mốc (có sẵn)') : r.variant === 'tuned' ? tr('corner-turned, padded', 'xoay góc, có đệm') : String(r.variant)),
          },
          rawCol('T', 'tile'),
          rawCol(tr('Batch', 'Batch'), 'batch'),
          msCol,
          numCol('GFLOP/s', 'gflops'),
          numCol(tr('Speed-up over baseline', 'Tăng tốc so với bản mốc'), 'speedup_vs_baseline'),
        ]}
      />
      <ReferenceSolution lab={l4} tiers={TIERS_64} />
      <h3>
        <Translated vi="Dự án: mốc dự án M2">The project: milestone M2</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Từ đây trở đi, trọng số của dự án được lưu dạng (out, in) theo thứ tự hàng (row-major), như cách{' '}
            <code>nn.Linear</code> và ggml lưu chúng; tensor là <code>float</code> thứ tự hàng với batch đứng đầu; kernel
            nhận con trỏ device thô và kích thước tường minh. <code>launchLinear(X, W, Y, batch, in, out)</code> là chữ ký
            mà chương 15 xây tiếp, nơi kernel được gộp việc và phân ô thanh ghi thành mốc dự án M3.
          </p>
        }
      >
        <p>
          From here on the project&rsquo;s weights are stored (out, in) row-major, as <code>nn.Linear</code> and ggml
          store them; tensors are row-major <code>float</code> with the batch first; kernels take raw device pointers and
          explicit sizes. <code>launchLinear(X, W, Y, batch, in, out)</code> is the signature chapter 15 builds on, where
          the kernel is coarsened and register-tiled into milestone M3.
        </p>
      </Translated>
      <MilestoneStatus />

      {/* ------------------------------------------------------------ 6.5 */}
      <h2>
        <Translated vi="Lab 6.5: buffer kép, với một bộ phát hiện race condition theo dõi">Lab 6.5: double buffering, with a race detector watching</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn được giao phép nhân ma trận phân ô của lab 5.2 với rào chắn thứ hai đã bị xóa. Nó biên dịch được, và nó sai:
            một luồng làm xong tích vô hướng sẽ nạp ô của pha kế tiếp đè lên những ô mà các luồng láng giềng chưa đọc. Bạn làm
            nó đúng với một rào chắn mỗi pha, mà không đặt lại rào chắn thứ hai: cho mỗi toán hạng hai ô, để các pha luân
            phiên giữa hai cặp, giữ rào chắn sau các phép nạp và không có rào chắn nào sau tích vô hướng. Bản mốc có sẵn là
            kernel hai rào chắn của lab 5.2.
          </p>
        }
      >
        <p>
          You are handed lab 5.2&rsquo;s tiled matrix multiplication with its second barrier deleted. It compiles, and it
          is wrong: a thread that finishes its dot product loads the next phase&rsquo;s tile over slots its neighbours
          have not read yet. You make it correct with one barrier per phase, without putting the second barrier back:
          give each operand two tiles, let the phases alternate between the two pairs, keep the barrier after the loads
          and none after the dot product. The provided baseline is lab 5.2&rsquo;s two-barrier kernel.
        </p>
      </Translated>
      <Source section="6.7" pages="146–147" figs={tr('applied to Fig. 5.9, p. 108', 'áp dụng cho Hình 5.9, p. 108')} />
      <Translated
        vi={
          <p>
            Vòng lặp của lab 5.2 ghi trước và đọc sau, nên rào chắn canh giữ phụ thuộc giả (ghi-sau-đọc) là rào chắn thứ hai
            của nó. Trong vòng lặp minh họa của sách, phép đọc đến trước và đó là rào chắn thứ nhất, và p. 147 viết
            &ldquo;thứ hai&rdquo; trong khi ý là rào chắn thứ nhất đó (xem{' '}
            <a href={href('ch06', 'wrap')}>phần tổng kết</a>). Mục <a href={href('ch06', '6.7')}>6.7</a> đi qua từng bước cả
            hai.
          </p>
        }
      >
        <p>
          Lab 5.2&rsquo;s loop writes first and reads second, so the barrier that guards the false (write-after-read)
          dependence is its second one. In the book&rsquo;s toy loop the read comes first and it is the first one, and
          p. 147 says &ldquo;second&rdquo; where it means that first (see the{' '}
          <a href={href('ch06', 'wrap')}>summary</a>). Section <a href={href('ch06', '6.7')}>6.7</a> steps through both.
        </p>
      </Translated>
      <LabFiles
        lab={l5}
        files={[
          ['kernel.cu', <Translated vi={<>các khai báo ô và hai chỗ chọn một cặp trong <code>doubleBufferedMatmulKernel</code>, rồi TODO 4, một chú thích nói vì sao nó đúng</>}><>the tile declarations and the two places that choose a pair in <code>doubleBufferedMatmulKernel</code>, then TODO 4, a comment saying why it is correct</></Translated>],
          ['kernel.metal', <Translated vi={<>cùng thay đổi đó trong <code>double_buffered_matmul</code></>}><>the same change in <code>double_buffered_matmul</code></></Translated>],
        ]}
      />
      <Translated vi={<p>Phương pháp quan trọng hơn bản sửa:</p>}>
        <p>The method matters more than the fix:</p>
      </Translated>
      <Translated
        vi={
          <ol>
            <li>
              Trước khi thay đổi bất cứ điều gì, hãy chạy mã khởi đầu (starter) dưới ThreadSanitizer:{' '}
              <code>python3 labs/pmpp.py run 6.5 --backend race</code>. Báo cáo của nó chỉ ra một phép ghi vào một ô và phép
              nhân-cộng đọc các ô. Hãy viết ra luồng nào làm gì, theo thứ tự nào, để báo cáo đó có thể xảy ra.
            </li>
            <li>
              <code>python3 labs/pmpp.py run 6.5 --backend emu</code>. Mã khởi đầu vượt qua ở độ rộng 16 và thất bại ở 32 và
              64 dưới cả ba thứ tự luồng. Hãy giải thích trường hợp độ rộng 16 trước khi đi tiếp.
            </li>
            <li>
              Sửa <code>work/kernel.cu</code>, và chạy cả hai biến thể trình giả lập cho tới khi cả hai đều sạch. Rồi làm
              cùng thay đổi đó trong <code>work/kernel.metal</code> và chạy trên mọi backend bạn có. Trên GPU mỗi trường hợp
              được khởi chạy 10 lần, với đầu ra được điền lại NaN trước mỗi lần khởi chạy.
            </li>
          </ol>
        }
      >
        <ol>
          <li>
            Before you change anything, run the starter under ThreadSanitizer:{' '}
            <code>python3 labs/pmpp.py run 6.5 --backend race</code>. Its report names a store into a tile and the
            multiply-add that reads the tiles. Write down which thread does what, in which order, for that report to be
            possible.
          </li>
          <li>
            <code>python3 labs/pmpp.py run 6.5 --backend emu</code>. The starter passes at width 16 and fails at 32 and 64
            under all three thread orders. Explain the width 16 case before you go on.
          </li>
          <li>
            Fix <code>work/kernel.cu</code>, and run both emulator flavours until both are clean. Then make the same change
            in <code>work/kernel.metal</code> and run on every backend you have. On a GPU each case is launched 10 times
            with the output refilled with NaN before each launch.
          </li>
        </ol>
      </Translated>
      <Translated
        vi={
          <p>
            Các bài kiểm thử so sánh với một tham chiếu CPU ở độ rộng 16, 32 và 64 (T = 16) trên trình giả lập, 32 và 48 trên
            backend race, và 16, 32, 256 và 33 × T trên thiết bị. Ba lỗi khác đều đã được kiểm tra là tự chúng thất bại: hai
            cặp ô mà không có rào chắn nào, rào chắn đặt sau tích vô hướng thay vì sau các phép nạp, và tích vô hướng đọc cặp
            kia. Một lần chạy sạch là bằng chứng, không phải chứng minh: backend race chỉ thấy những lịch thực thi đã xảy ra,
            và mười lần khởi chạy là mười lịch thực thi. Lập luận trong TODO 4 mới là thứ cho thấy kernel đúng.
          </p>
        }
      >
        <p>
          The tests compare with a CPU reference at widths 16, 32 and 64 (T = 16) on the emulator, 32 and 48 on the race
          backend, and 16, 32, 256 and 33 × T on a device. Three other mistakes were each checked to fail on their own: two
          pairs of tiles with no barrier at all, the barrier after the dot product instead of after the loads, and the dot
          product reading the other pair. A clean run is evidence, not proof: the race backend sees only the schedules that
          happened, and ten launches are ten schedules. The argument in TODO 4 is what shows the kernel correct.
        </p>
      </Translated>
      <RaceStatus lab={l5} />
      <LabPrediction
        id="ch06/labs/p-lab5-onebarrier"
        prompt={
          <Translated
            vi={
              <>
                Trước khi bạn chạy benchmark. Một rào chắn mỗi pha thay vì hai, với cái giá là gấp đôi bộ nhớ chia sẻ mỗi
                khối. Ở T = 16, bạn sẽ đo được mức tăng tốc bao nhiêu so với kernel hai rào chắn? Hãy cân một rào chắn với 16
                phép nhân-cộng mà mỗi luồng làm giữa các rào chắn.
              </>
            }
          >
            <>
              Before you run the benchmark. One barrier per phase instead of two, at the price of twice the shared memory
              per block. At T = 16, what speed-up over the two-barrier kernel will you measure? Weigh one barrier against
              the 16 multiply-adds each thread does between barriers.
            </>
          </Translated>
        }
        unit={tr('× the two-barrier kernel', '× kernel hai rào chắn')}
        measure={oneBarrier}
        reading={
          <Translated
            vi={
              <>
                Một rào chắn tốn đúng bằng thời gian luồng chậm nhất của khối bắt các luồng khác chờ, và một pha 16 phép
                nhân-cộng mỗi luồng là ngắn. Cặp ô thứ hai cũng không miễn phí: chương 5 đã cho thấy bộ nhớ chia sẻ mỗi khối
                có thể giới hạn số khối cư trú. GEMM Metal dùng trong sản phẩm của MLX giữ hai rào chắn mỗi bước, tức vòng lặp
                &ldquo;trước&rdquo; của sách. Bất kể bạn đo được gì, đó là một sự thật về kernel này trên thiết bị của bạn,
                không phải một quy tắc về buffer kép.
              </>
            }
          >
            <>
              A barrier costs what the slowest thread of the block makes the others wait, and a phase of 16 multiply-adds
              per thread is short. The second pair of tiles is not free either: chapter 5 showed that shared memory per
              block can limit how many blocks are resident. MLX&rsquo;s production Metal GEMM keeps two barriers per step,
              the book&rsquo;s &ldquo;before&rdquo; loop. Whatever you measured is a fact about this kernel on your device,
              not a rule about double buffering.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l5}
        filter={(r) => r.name === 'matmul'}
        cols={[
          {
            head: tr('Kernel measured', 'Kernel được đo'),
            cell: (r) => (r.variant === 'two_barriers' ? tr('two barriers (provided)', 'hai rào chắn (có sẵn)') : r.variant === 'double_buffered' ? tr('double buffered, one barrier', 'buffer kép, một rào chắn') : String(r.variant)),
          },
          rawCol('T', 'tile'),
          rawCol(tr('Width', 'Độ rộng'), 'width'),
          msCol,
          numCol('GFLOP/s', 'gflops'),
          numCol(tr('Speed-up over two barriers', 'Tăng tốc so với hai rào chắn'), 'speedup_vs_two_barriers'),
        ]}
      />
      <ReferenceSolution lab={l5} tiers={MATMUL_TIERS} />
      <Translated
        vi={
          <p>
            Các kernel thực tế đưa ý tưởng đi thêm một bước: trong khi các luồng tính trên một cặp ô, cặp kế tiếp đã đang
            trên đường tới. Điều đó cần một phép sao chép mà các luồng không phải chờ, và đó là chỗ chương 15 tiếp nhận ý
            tưởng này.
          </p>
        }
      >
        <p>
          Real kernels take the idea one step further: while the threads compute on one pair of tiles, the next pair is
          already on its way. That needs a copy the threads do not wait for, and it is where chapter 15 picks the idea up.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md A16, B8, C7, D3)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              Ngày nay NVIDIA chủ yếu mô tả buffer kép bằng các phép sao chép bất đồng bộ vào bộ nhớ chia sẻ. AMD chỉ mô tả
              một phép sao chép như vậy cho CDNA 3 và 4. Metal không mô tả cái nào, nên trên máy Mac dạng đồng bộ của lab
              này là dạng bạn có. GEMM Metal của MLX giữ hai rào chắn mỗi bước.
            </p>
          }
        >
          <p>
            NVIDIA documents double buffering today mostly with asynchronous copies into shared memory. AMD documents
            such a copy only for CDNA 3 and 4. Metal documents none, so on a Mac the synchronous form of this lab is the
            one you have. MLX&rsquo;s Metal GEMM keeps two barriers per step.
          </p>
        </Translated>
      </Today>
    </LabsPage>
  );
}
