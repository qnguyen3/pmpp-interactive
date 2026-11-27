import type { Question } from '../../components/Ask';
import { Source, Today, type Tier } from '../../components/notes';
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
  rawCol,
  useLabs,
  type BenchSet,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 21's own content.
// No question on this page: the lab is the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch21', [{ num: '21.1', dir: 'lab1_dcs', name: 'scatter, gather, coarsened', vi: { name: 'scatter, gather, đã gộp việc' } }]);
const [l1] = LABS;

/** Identifier colours in the reference solution: where each value lives. */
const TIERS: Record<string, Tier> = {
  atoms: 'const',
  energygrid: 'global',
  energy: 'reg',
  energy0: 'reg',
  energy1: 'reg',
  energy2: 'reg',
  energy3: 'reg',
  charge: 'reg',
  dysqdzsq: 'reg',
  dx: 'reg',
  dy: 'reg',
  dz: 'reg',
  dx0: 'reg',
  dx1: 'reg',
  dx2: 'reg',
  dx3: 'reg',
  dy2: 'reg',
  dz2: 'reg',
  i: 'reg',
  j: 'reg',
  n: 'reg',
  x: 'reg',
  y: 'reg',
  z: 'reg',
};

// ---------------------------------------------------------------- what each prediction is measured against
// Comparison A: a 256 × 256 slice, chunk 256. Comparison B: a 1024 × 1024 slice at four chunk sizes.
const A = { nx: 256, chunk: 256 };
const B_NX = 1024;
const CHUNKS = [64, 256, 512, 1024];
const ms = (r: BenchRow | undefined) => {
  const v = num(r?.seconds);
  return v ? `${fmt(v * 1000)} ms` : '?'; // i18n-ignore: a number and a unit
};
const secs = (r: BenchRow | undefined) => num(r?.seconds) ?? 0;

/** Gather over scatter on comparison A. */
const gatherOverScatter = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const sc = benchRow(s, 'scatter', A);
    const ga = benchRow(s, 'gather', A);
    if (!secs(sc) || !secs(ga)) return null;
    return { value: secs(sc) / secs(ga), detail: tr(`scatter ${ms(sc)} against gather ${ms(ga)} per slice`, `scatter ${ms(sc)} so với gather ${ms(ga)} mỗi lát`) };
  });

/** Adjacent over strided at one chunk size of comparison B: above 1 means the strided kernel was faster. */
const stridedGain = (s: BenchSet, chunk: number) => {
  const adj = benchRow(s, 'coarse_adjacent', { nx: B_NX, chunk });
  const str = benchRow(s, 'coarse_strided', { nx: B_NX, chunk });
  return secs(adj) && secs(str) ? { adj, str, ratio: secs(adj) / secs(str) } : null;
};

/** Strided over adjacent at chunk 64, with the other chunk sizes in the detail. */
const stridedOverAdjacent = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const first = stridedGain(s, CHUNKS[0]);
    if (!first) return null;
    const rest = CHUNKS.slice(1).flatMap((c) => {
      const g = stridedGain(s, c);
      return g ? [tr(`${fmt(g.ratio)} at chunk ${c.toLocaleString('en-US')}`, `${fmt(g.ratio)} ở chunk ${c.toLocaleString('en-US')}`)] : [];
    });
    return {
      value: first.ratio,
      detail: tr(
        `strided ${ms(first.str)} against adjacent ${ms(first.adj)} at chunk 64${rest.length ? `; then ${rest.join(', ')}` : ''}`,
        `strided ${ms(first.str)} so với adjacent ${ms(first.adj)} ở chunk 64${rest.length ? `; rồi ${rest.join(', ')}` : ''}`,
      ),
    };
  });

const MISSING = (
  <Translated
    vi={
      <>
        Phần này cần các dòng benchmark của lab 21.1. Chạy <code>python3 labs/pmpp.py run 21.1 --bench</code> trên một
        máy có GPU khi các kernel của bạn đã qua (hoặc với <code>--solution</code> để đo thời gian lời giải tham chiếu).
      </>
    }
  >
    <>
      This needs lab 21.1's benchmark rows. Run <code>python3 labs/pmpp.py run 21.1 --bench</code> on a machine with a GPU
      once your kernels pass (or with <code>--solution</code> to time the reference).
    </>
  </Translated>
);

// ---------------------------------------------------------------- results table
const KERNEL_LABEL: Record<string, string> = {
  get scatter() {
    return tr('scatter (Fig. 21.5)', 'scatter (Fig. 21.5)');
  },
  get gather() {
    return tr('gather (Fig. 21.6)', 'gather (Fig. 21.6)');
  },
  get coarse_adjacent() {
    return tr('adjacent (provided, Fig. 21.8 corrected)', 'adjacent (có sẵn, Fig. 21.8 đã sửa)');
  },
  get coarse_strided() {
    return tr('strided (Fig. 21.10)', 'strided (Fig. 21.10)');
  },
};
const gpairs = (r: BenchRow) => {
  const v = num(r.pairs_per_s);
  return v == null ? '' : fmt(v / 1e9);
};

/** One line on lab 21.1 on the race backend, which the status table does not show. */
function RaceStatus() {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const line = (variant: 'solution' | 'work') => {
    const rs = (run?.results.filter((x) => x.lab === l1.id && x.variant === variant) ?? []) as (RunResult & { tsan?: unknown[] })[];
    const tests = rs.flatMap((x) => x.tests);
    const reportsFound = rs.reduce((n, x) => n + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
    return { s: labStatus(run, l1.id, variant), tests, passed: tests.filter((t) => t.pass).length, reportsFound };
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
                  `On this machine the reference solution passed all ${sol.tests.length} tests on the race backend with ${sol.reportsFound ? `${sol.reportsFound} ThreadSanitizer report(s)` : 'no race reported'}: emulator-checked on this flavour.`,
                  `Trên máy này, lời giải tham chiếu đã qua cả ${sol.tests.length} bài kiểm tra trên backend race với ${sol.reportsFound ? `${sol.reportsFound} báo cáo ThreadSanitizer` : 'không có race nào được báo'}: emulator-checked trên biến thể này.`,
                )
              : tr(
                  `On this machine the reference solution failed on the race backend (${sol.passed} of ${sol.tests.length} tests): unverified there.`,
                  `Trên máy này, lời giải tham chiếu không qua trên backend race (${sol.passed} trên ${sol.tests.length} bài kiểm tra): unverified ở đó.`,
                )}{' '}
          {mine.s === 'none'
            ? tr('Your own kernels have no recorded run there, or are still the untouched starter.', 'Các kernel của bạn chưa có lần chạy nào được ghi lại ở đó, hoặc vẫn là mã khởi đầu (starter) chưa sửa.')
            : mine.s === 'pass'
              ? tr(
                  `Your own kernels passed all ${mine.tests.length} tests there with ${mine.reportsFound ? `${mine.reportsFound} ThreadSanitizer report(s): read them` : 'no race reported'}.`,
                  `Các kernel của bạn đã qua cả ${mine.tests.length} bài kiểm tra ở đó với ${mine.reportsFound ? `${mine.reportsFound} báo cáo ThreadSanitizer: hãy đọc chúng` : 'không có race nào được báo'}.`,
                )
              : tr(
                  `Your own kernels passed ${mine.passed} of ${mine.tests.length} tests there, with ${mine.reportsFound} ThreadSanitizer report(s) in the run.`,
                  `Các kernel của bạn qua ${mine.passed} trên ${mine.tests.length} bài kiểm tra ở đó, với ${mine.reportsFound} báo cáo ThreadSanitizer trong lần chạy.`,
                )}
        </>
      )}
    </p>
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
              Một lab, trong <code>labs/ch21/lab1_dcs/</code>. Nó là tùy chọn, vì chương này được rút gọn, và nó không bổ
              sung gì cho dự án xuyên suốt: nó xây các kernel scatter, gather và kernel đã gộp việc của{' '}
              <a href={href('ch21', '21.2')}>trang 21.2</a> cho một lát của bản đồ thế và đo hai lựa chọn mà trang đó bàn
              luận.
            </>
          }
        >
          One lab, in <code>labs/ch21/lab1_dcs/</code>. It is optional, as this chapter is condensed, and it adds nothing
          to the through-line project: it builds the scatter, gather and coarsened kernels of{' '}
          <a href={href('ch21', '21.2')}>page 21.2</a> for one slice of the potential map and measures the two choices that
          page argues about.
        </Translated>
      </LabsIntro>
      <RunningLabs
        example="21.1"
        alsoShownIn={
          <Translated
            vi={
              <>
                độ rộng nhóm "yours" của trình khám phá câu lệnh ghi ở <a href={href('ch21', '21.2')}>21.2</a>
              </>
            }
          >
            <>
              the "yours" group width of the write-statement explorer on <a href={href('ch21', '21.2')}>21.2</a>
            </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <VerificationStatus />

      {/* ------------------------------------------------------------ 21.1 */}
      <h2>
        <Translated vi="Lab 21.1: một lát, ba cách">Lab 21.1: one slice, three ways</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mọi điểm của một lát trong lưới <code>nz × ny × nx</code> nhận tổng, trên mọi nguyên tử, của điện tích chia cho
            khoảng cách. Các nguyên tử đến device theo từng chunk đủ nhỏ cho bộ nhớ hằng, và mỗi lần khởi chạy cộng đóng góp của
            một chunk vào lát. Bạn viết ba kernel và các hàm khởi chạy của chúng. Kernel scatter của Fig. 21.5 giao cho mỗi luồng
            một nguyên tử của chunk, đi qua cả lát và cộng bằng một phép toán nguyên tử float. Kernel gather của Fig. 21.6 giao
            cho mỗi luồng một điểm, tính tổng của chunk trong một thanh ghi và cộng tổng đó bằng một phép <code>+=</code>{' '}
            thường. Kernel đã gộp việc của Fig. 21.10 giao cho mỗi luồng bốn điểm của một hàng, cách nhau{' '}
            <code>blockDim.x</code>, đọc mỗi nguyên tử một lần và dùng chung <code>dy*dy + dz*dz</code> cùng điện tích cho cả
            bốn.
          </p>
        }
      >
        <p>
          Every point of a slice of an <code>nz × ny × nx</code> grid gets the sum, over all atoms, of charge divided by
          distance. The atoms reach the device in chunks small enough for constant memory, and each launch adds one chunk's
          contribution to the slice. You write three kernels and their launch functions. The scatter kernel of Fig. 21.5
          gives each thread one atom of the chunk, walks the whole slice and adds with a float atomic. The gather kernel of
          Fig. 21.6 gives each thread one point, sums the chunk in a register and adds the sum with one plain{' '}
          <code>+=</code>. The coarsened kernel of Fig. 21.10 gives each thread four points of a row,{' '}
          <code>blockDim.x</code> apart, reads each atom once and shares <code>dy*dy + dz*dz</code> and the charge across
          the four.
        </p>
      </Translated>
      <Source section={tr('21.2 to 21.4', '21.2 đến 21.4')} pages="515–522" figs={tr('Figs. 21.5, 21.6, 21.10; exercise 1', 'Figs. 21.5, 21.6, 21.10; bài tập 1')} />
      <Translated
        vi={
          <p>
            Mỗi hàm khởi chạy là vòng lặp host ở p. 517, cũng là bài tập 1 của sách: chép nhiều nhất <code>chunk</code> nguyên
            tử vào mảng <code>__constant__</code> bằng <code>cudaMemcpyToSymbol</code>, khởi chạy, lặp lại cho đến khi mọi
            nguyên tử đã được cộng; chunk cuối thường không đầy. Hợp đồng nằm trong <code>dcs.h</code>: lưới là một con trỏ
            device theo thứ tự hàng (row-major) với x thay đổi nhanh nhất, mỗi nguyên tử là bốn float (x, y, z, điện tích) trong
            bộ nhớ host, một lần khởi chạy cộng vào lát <code>k</code> và để yên các lát khác, và <code>CHUNK_MAX</code>,{' '}
            <code>BLOCK_X</code>, <code>BLOCK_Y</code> và <code>COARSEN</code> được định nghĩa ở đó và không ở đâu khác.
          </p>
        }
      >
        <p>
          Each launch function is the host loop of p. 517, which is the book's exercise 1: copy at most{' '}
          <code>chunk</code> atoms into the <code>__constant__</code> array with <code>cudaMemcpyToSymbol</code>, launch,
          repeat until every atom has been added; the last chunk is usually partial. The contract is in{' '}
          <code>dcs.h</code>: the grid is a row-major device pointer with x fastest, the atoms are four floats each (x, y,
          z, charge) in host memory, a launch adds to slice <code>k</code> and leaves the others alone, and{' '}
          <code>CHUNK_MAX</code>, <code>BLOCK_X</code>, <code>BLOCK_Y</code> and <code>COARSEN</code> are defined there
          and nowhere else.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Một kernel được cung cấp sẵn và không sửa: <code>labs/ch21/shared/coarse_adjacent.cu</code> (cùng bản Metal của
            nó), là Fig. 21.8 với dòng 05 đã sửa, để mỗi luồng sở hữu bốn điểm liền kề. Kernel strided của bạn được đo thời gian
            so với nó. Nó giữ mảng hằng riêng, <code>atoms_adj</code>: dưới trình giả lập (emulator), một biến{' '}
            <code>__constant__</code> là một biến toàn cục bình thường, và hai biến cùng tên sẽ xung đột khi các tệp được liên
            kết.
          </p>
        }
      >
        <p>
          One kernel is provided and not edited: <code>labs/ch21/shared/coarse_adjacent.cu</code> (with its Metal pair),
          Fig. 21.8 with its line 05 corrected, so each thread owns four adjacent points. Your strided kernel is timed
          against it. It keeps its own constant array, <code>atoms_adj</code>: under the emulator a{' '}
          <code>__constant__</code> variable is an ordinary global, and two of the same name would collide when the files
          are linked.
        </p>
      </Translated>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={
                <>
                  mảng nguyên tử <code>__constant__</code>, <code>dcsScatterKernel</code>, <code>dcsGatherKernel</code>,{' '}
                  <code>dcsCoarsenedKernel</code> và các hàm khởi chạy của chúng <code>launchDcsScatter</code>,{' '}
                  <code>launchDcsGather</code> và <code>launchDcsCoarsened</code>
                </>
              }
            >
              <>
                the <code>__constant__</code> atom array, <code>dcsScatterKernel</code>, <code>dcsGatherKernel</code>,{' '}
                <code>dcsCoarsenedKernel</code> and their launches <code>launchDcsScatter</code>,{' '}
                <code>launchDcsGather</code> and <code>launchDcsCoarsened</code>
              </>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated
              vi={
                <>
                  <code>dcs_scatter</code>, <code>dcs_gather</code> và <code>dcs_coarsened</code>, với các nguyên tử dưới dạng{' '}
                  <code>constant float4*</code>
                </>
              }
            >
              <>
                <code>dcs_scatter</code>, <code>dcs_gather</code> and <code>dcs_coarsened</code>, with the atoms as{' '}
                <code>constant float4*</code>
              </>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated
              vi={
                <>
                  ba vòng lặp chunk, trong <code>encodeDcsScatter</code>, <code>encodeDcsGather</code> và{' '}
                  <code>encodeDcsCoarsened</code>
                </>
              }
            >
              <>
                the three chunk loops, in <code>encodeDcsScatter</code>, <code>encodeDcsGather</code> and{' '}
                <code>encodeDcsCoarsened</code>
              </>
            </Translated>,
          ],
        ]}
      />
      <Translated vi={<p>Những chỗ lab khác với các đoạn mã in trong sách:</p>}>
        <p>Where the lab differs from the printed listings:</p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>
              Dòng 05 của Fig. 21.8 cộng một <code>threadIdx.x</code> chưa nhân hệ số trong khi mỗi luồng ghi từ <code>i</code>{' '}
              đến <code>i+3</code>, nên các luồng cạnh nhau chồng lấn và phần lớn các điểm của mỗi khối không bao giờ được ghi (
              <a href={href('ch21', '21.2')}>21.2</a>, phần tìm lỗi in). Mốc cơ sở có sẵn nhân chỉ số với hệ số, như văn bản ở
              p. 521 mô tả. Cũng dòng 05 đó lại đúng trong Fig. 21.10, nơi quyền sở hữu theo bước nhảy, và kernel của bạn giữ
              nguyên nó. Đã kiểm thử: mọi trường hợp đều chạy mốc cơ sở đã sửa, và <code>labs/ch21/snippets/check.py</code> chạy
              dòng như bản in và cho thấy nó thất bại đúng như 21.2 nói.
            </li>
            <li>
              Lát là một <code>int k</code> và z của nó là <code>k * gridspacing</code>. Fig. 21.6 truyền một <code>z</code>{' '}
              kiểu float và tính <code>k = z / gridspacing</code>, phép này cắt cụt, còn Figs. 21.3 và 21.4 đánh chỉ số lát theo
              hai cách khác. Kích thước là <code>nx</code> và <code>ny</code> trong khi sách truyền một <code>dim3 grid</code>.
              Đã kiểm thử: độ lệch của lát, bằng các trường hợp sentinel bên dưới.
            </li>
            <li>
              Không đoạn mã nào kiểm tra biên, và Fig. 21.5 hoàn toàn không có số nguyên tử: các luồng vượt quá cuối một chunk
              không đầy sẽ cộng những nguyên tử mà chunk trước để lại trong bộ nhớ hằng. Ở đây lưới không được đệm thành các khối
              trọn vẹn, nên mọi kernel đều kiểm tra biên, kernel đã gộp việc kiểm tra một lần cho mỗi điểm. Đã kiểm thử: bằng các
              lát nhô ra và chunk không đầy bên dưới.
            </li>
            <li>
              Chương không nêu tên lời gọi điền bộ nhớ hằng; đó là <code>cudaMemcpyToSymbol</code>. <code>atomicAdd</code> kiểu
              float của sách được giữ nguyên: NVIDIA giờ gọi nó là phép toán nguyên tử legacy và khuyên dùng{' '}
              <code>cuda::atomic_ref</code>, thứ mà chương 9 đã dùng.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            Fig. 21.8's line 05 adds an unscaled <code>threadIdx.x</code> while each thread writes <code>i</code> to{' '}
            <code>i+3</code>, so neighbouring threads overlap and most of each block's points are never written (
            <a href={href('ch21', '21.2')}>21.2</a>, the erratum hunt). The provided baseline scales the index by the factor,
            as the text on p. 521 describes. The same line 05 is right in Fig. 21.10, whose ownership is strided, and your
            kernel keeps it. Tested: every case runs the corrected baseline, and{' '}
            <code>labs/ch21/snippets/check.py</code> runs the printed line and shows it failing as 21.2 says.
          </li>
          <li>
            The slice is an <code>int k</code> and its z is <code>k * gridspacing</code>. Fig. 21.6 passes a float{' '}
            <code>z</code> and computes <code>k = z / gridspacing</code>, which truncates, and Figs. 21.3 and 21.4 index the
            slice two other ways. The sizes are <code>nx</code> and <code>ny</code> where the book passes a{' '}
            <code>dim3 grid</code>. Tested: the slice offset, by the sentinel cases below.
          </li>
          <li>
            No listing tests a bound, and Fig. 21.5 has no atom count at all: the threads past the end of a partial chunk
            would add atoms the previous chunk left in constant memory. Here the grid is not padded to whole blocks, so every
            kernel tests its bounds, the coarsened one once per point. Tested: by the overhanging slices and the partial
            chunk below.
          </li>
          <li>
            The chapter does not name the call that fills constant memory; it is <code>cudaMemcpyToSymbol</code>. The
            book's float <code>atomicAdd</code> is kept: NVIDIA now calls it a legacy atomic and recommends{' '}
            <code>cuda::atomic_ref</code>, which chapter 9 used.
          </li>
        </ul>
      </Translated>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 21.1</code>, bắt đầu với <code>--backend emu</code>. Mỗi trường hợp chạy cả một
            lát qua hàm khởi chạy của bạn và so từng điểm với một tổng tính bằng double trong <code>main.cpp</code>, trong phạm
            vi <code>1e-4</code> của tổng trên mọi nguyên tử của |điện tích| / khoảng cách tại điểm đó. Điều đó chừa chỗ cho fast
            math của Metal và cho các phép toán nguyên tử float cộng không theo thứ tự cố định; không phép so sánh nào là theo
            từng bit. Các lát là 37 × 11 và 300 × 5, không lát nào là bội số của 4 × <code>BLOCK_X</code>; 150 nguyên tử được
            đưa vào trong ba lần khởi chạy 64, 64 và 22, và cũng trong một lần khởi chạy, và một trường hợp chỉ có một nguyên
            tử. Lát 1 hoặc 2 trong 3 được tính trong khi các lát còn lại giữ một giá trị sentinel phải được giữ nguyên từng bit,
            và lát 2 là lát cuối, nên một lần ghi vượt cuối hàng sẽ ra khỏi vùng cấp phát. Mọi nguyên tử nằm giữa hai mặt phẳng
            lát, nên không khoảng cách nào bằng không. Trên GPU còn chạy 1000 × 333 với 2,000 nguyên tử theo chunk 1,024. Cả bốn
            kernel chạy mọi trường hợp, và gather với kernel đã gộp việc còn phải khớp với nhau.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 21.1</code>, starting with <code>--backend emu</code>. Every case runs a whole
          slice through your launch function and compares each point with a sum in double in <code>main.cpp</code>, within{' '}
          <code>1e-4</code> of the sum over atoms of |charge| / distance at that point. That allows for Metal's fast math and
          for float atomics adding in no fixed order; no comparison is bitwise. The slices are 37 × 11 and 300 × 5, neither
          a multiple of 4 × <code>BLOCK_X</code>; 150 atoms go in three launches of 64, 64 and 22, and also in one launch,
          and one case has a single atom. Slice 1 or 2 of 3 is computed while the other slices hold a sentinel that must
          survive bit for bit, and slice 2 is the last, so a write past a row's end leaves the allocation. Every atom sits
          half-way between two slice planes, so no distance is zero. A GPU also runs 1000 × 333 with 2,000 atoms in chunks
          of 1,024. All four kernels run every case, and gather and coarsened must also agree with each other.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch21-facts.md A10, B2, C3)">{/* i18n-ignore: a date and a citation */}
        <Translated
          vi={
            <p>
              Vì sao cần dung sai: <code>/</code> và <code>sqrtf</code> của CUDA được làm tròn đúng theo mặc định; HIP báo sai
              khác trong phạm vi 1 ULP so với thư viện C++; fast math của Metal, mặc định của trình biên dịch, cho x / y trong
              phạm vi 2.5 ulp và tính sqrt bằng x·rsqrt(x).
            </p>
          }
        >
          <p>
            Why a tolerance: CUDA's <code>/</code> and <code>sqrtf</code> are correctly rounded by default; HIP reports
            differences within 1 ULP of the C++ library; Metal's fast math, the compiler default, gives x / y within 2.5 ulp
            and computes sqrt as x·rsqrt(x).
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Mỗi backend bắt được gì quan trọng ở đây hơn ở phần lớn các lab khác. Trình giả lập chạy từng luồng một và không bao
            giờ ngắt một phép <code>+=</code> giữa chừng, nên nó không thể thấy một phép toán nguyên tử bị thiếu: một kernel
            scatter với phép <code>+=</code> thường vẫn qua ở đó. Nó bắt được chỉ số sai, thiếu kiểm tra biên (AddressSanitizer
            chỉ ra dòng), nguyên tử cũ do thiếu số nguyên tử, <code>=</code> ở chỗ phải là <code>+=</code> (các trường hợp ba
            lần khởi chạy) và thiếu độ lệch lát (các trường hợp sentinel). Biến thể race,{' '}
            <code>python3 labs/pmpp.py run 21.1 --backend race</code>, chạy các luồng của một khối đồng thời dưới ThreadSanitizer
            trên các lát nhỏ: khi phép toán nguyên tử bị thay bằng <code>+=</code>, nó báo dòng ghi của kernel scatter, và với
            dòng 05 như bản in của Fig. 21.8, nó báo bốn lần ghi chồng lấn. Một lần chạy sạch là bằng chứng, không phải chứng
            minh: nó chỉ thấy những lịch chạy đã xảy ra. Trên GPU, một phép toán nguyên tử bị thiếu chỉ lộ ra dưới dạng tổng
            sai, và chỉ khi hai luồng va chạm về thời gian.
          </p>
        }
      >
        <p>
          What each backend can catch matters here more than in most labs. The emulator runs one thread at a time and never
          interrupts a <code>+=</code> half-way, so it cannot see a missing atomic: a scatter kernel with a plain{' '}
          <code>+=</code> passes there. It does catch a wrong index, a missing bound (AddressSanitizer names the line), stale
          atoms from a missing atom count, <code>=</code> where <code>+=</code> belongs (the three-launch cases) and a
          missing slice offset (the sentinel cases). The race flavour,{' '}
          <code>python3 labs/pmpp.py run 21.1 --backend race</code>, runs the threads of a block concurrently under
          ThreadSanitizer on small slices: with the atomic replaced by <code>+=</code> it reported the scatter kernel's
          write line, and with Fig. 21.8's printed line 05 it reported the four overlapping writes. A clean run is evidence,
          not proof: it sees only the schedules that happened. On a GPU a missing atomic shows only as wrong sums, and only
          when two threads collided in time.
        </p>
      </Translated>
      <RaceStatus />
      <Translated
        vi={
          <p>
            Một lỗ hổng mà các bài kiểm tra bỏ ngỏ: các trường hợp rộng 37 không thể thấy <code>dx1</code>, <code>dx2</code>{' '}
            hay <code>dx3</code> sai trong kernel đã gộp việc của bạn, vì một hàng 37 điểm chỉ chứa điểm đầu tiên của mỗi luồng.
            Các trường hợp rộng 300 thì thấy được, nên một kernel qua trường hợp đầu và trượt trường hợp sau có độ lệch và vị trí
            ghi lệch nhịp nhau.
          </p>
        }
      >
        <p>
          One gap the tests leave open: the 37-wide cases cannot see a wrong <code>dx1</code>, <code>dx2</code> or{' '}
          <code>dx3</code> in your coarsened kernel, because a 37-point row holds only each thread's first point. The
          300-wide cases can, so a kernel that passes the first and fails the second has its offsets and its write
          positions out of step.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Trên Metal không có symbol nào để chép vào. Mọi nguyên tử nằm trong một buffer <code>float4</code>, và một chunk là
            buffer đó được gắn lại tại nguyên tử đầu tiên của chunk, 16 byte mỗi nguyên tử. Kernel scatter khai báo lưới là{' '}
            <code>device atomic_float*</code> và cộng bằng <code>atomic_fetch_add_explicit</code>; hai kernel còn lại nhận cùng
            buffer đó dưới dạng <code>device float*</code>. Khởi chạy kernel đã gộp việc bằng <code>dispatchThreadgroups</code>,
            các threadgroup trọn vẹn <code>BLOCK_X × BLOCK_Y</code>, và đọc bước nhảy của nó từ{' '}
            <code>[[dispatch_threads_per_threadgroup]]</code>: <code>dispatchThreads</code> sẽ cắt bớt các threadgroup ở mép,
            và các luồng bị thiếu của một nhóm bị cắt sở hữu những điểm không ai tính. Một nhóm bị cắt cũng báo kích thước nhỏ
            hơn của nó trong <code>[[threads_per_threadgroup]]</code>, nên bước nhảy đọc từ đó sẽ sai ở mép.
          </p>
        }
      >
        <p>
          On Metal there is no symbol to copy to. All the atoms sit in one buffer of <code>float4</code>s, and a chunk is
          that buffer bound again at the chunk's first atom, 16 bytes per atom. The scatter kernel declares the grid{' '}
          <code>device atomic_float*</code> and adds with <code>atomic_fetch_add_explicit</code>; the other two take the same
          buffer as <code>device float*</code>. Launch the coarsened kernel with <code>dispatchThreadgroups</code>, whole
          threadgroups of <code>BLOCK_X × BLOCK_Y</code>, and read its stride from{' '}
          <code>[[dispatch_threads_per_threadgroup]]</code>: <code>dispatchThreads</code> would trim the edge threadgroups,
          and a trimmed group's missing threads own points that nobody computes. A trimmed group also reports its smaller
          size in <code>[[threads_per_threadgroup]]</code>, so a stride read from there is wrong at the edge.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch21-facts.md C1, C2; infra-facts C1, C6; ch05-facts C4, C5; ch02-facts C9)">{/* i18n-ignore: a date and a citation */}
        <Translated
          vi={
            <p>
              Phép toán nguyên tử float cần một kiểu atomic và Apple7 (M1) trở lên. Không có bộ nhớ đệm hằng nào được ghi nhận.
              Threadgroup ở mép của <code>dispatchThreads</code> nhỏ hơn. Các dispatch của cùng một encoder thấy được các lần ghi
              của nhau.
            </p>
          }
        >
          <p>
            Float atomics need an atomic type and Apple7 (M1) or later. No constant cache is documented. An edge
            threadgroup of <code>dispatchThreads</code> is smaller. Dispatches of one encoder see each other's writes.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Cùng tệp <code>kernel.cu</code> đó được viết để build cho HIP mà không cần sửa, vì nó chỉ dùng những phần chuyển được
            qua <code>pmpp/vendors/hip.h</code>: luồng và khối, bộ nhớ hằng, một phép toán nguyên tử legacy và các lời gọi
            runtime, với <code>cudaMemcpyToSymbol</code> được ánh xạ sang <code>hipMemcpyToSymbol</code>. Warp intrinsic,
            cooperative groups hay các header độ chính xác một nửa sẽ không chuyển được nguyên vẹn. Bản build đó là unverified:
            chưa có máy HIP nào build nó. Bản build HIP của khóa học không truyền <code>-munsafe-fp-atomics</code>, nên một phép
            đo scatter trên GPU AMD là đo dạng an toàn của phép toán nguyên tử float.
          </p>
        }
      >
        <p>
          The same <code>kernel.cu</code> is written to build for HIP unchanged, because it uses only the parts that carry over through{' '}
          <code>pmpp/vendors/hip.h</code>: threads and blocks, constant memory, a legacy atomic and runtime calls, with{' '}
          <code>cudaMemcpyToSymbol</code> mapped to <code>hipMemcpyToSymbol</code>. Warp intrinsics, cooperative groups or
          half-precision headers would not port unchanged. That build is unverified: no HIP machine has built it yet. The course's HIP build does not pass <code>-munsafe-fp-atomics</code>, so a scatter
          measurement on an AMD GPU is of the safe form of the float atomic.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/infra-facts.md B1, B5; ch21-facts B1)">{/* i18n-ignore: a date and a citation */}
        <Translated
          vi={
            <p>
              <code>atomicAdd</code> kiểu float của HIP là một phép toán an toàn, có thể là một vòng lặp compare-and-swap, trừ khi
              build với <code>-munsafe-fp-atomics</code>. AMD không ghi nhận kích thước bộ nhớ hằng.
            </p>
          }
        >
          <p>
            HIP's float <code>atomicAdd</code> is a safe operation, possibly a compare-and-swap loop, unless built with{' '}
            <code>-munsafe-fp-atomics</code>. AMD documents no constant-memory size.
          </p>
        </Translated>
      </Today>

      <h3>
        <Translated vi="Thí nghiệm">The experiment</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Làm phần này khi các kernel của bạn đã qua. So sánh A đo scatter so với gather trên một lát 256 × 256 với 1,024
            nguyên tử theo chunk 256; nó được giữ nhỏ vì scatter có một luồng cho mỗi nguyên tử và mỗi luồng đi qua cả lát. So
            sánh B đo gather, kernel adjacent có sẵn và kernel strided của bạn trên một lát 1024 × 1024 với 4,096 nguyên tử, ở các
            chunk 64, 256, 512 và 1,024. Chunk 512 là 8 KB nguyên tử, tập làm việc của bộ nhớ đệm hằng cho mỗi SM mà NVIDIA ghi
            nhận (ch21-facts A5); liệu có gì thay đổi ở đó hay không thì không được ghi nhận. Mỗi thời gian là cả một lát, gồm
            mọi chunk, lấy kết quả tốt nhất trong năm lần. Hãy ghi lại hai dự đoán trước khi bạn chạy <code>--bench</code>.
          </p>
        }
      >
        <p>
          Do this once your kernels pass. Comparison A times scatter against gather on a 256 × 256 slice with 1,024 atoms in
          chunks of 256; it is kept small because scatter has one thread per atom and each one walks the whole slice.
          Comparison B times gather, the provided adjacent kernel and your strided kernel on a 1024 × 1024 slice with 4,096
          atoms, at chunks of 64, 256, 512 and 1,024. Chunk 512 is 8 KB of atoms, the constant-cache working set per SM
          that NVIDIA documents (ch21-facts A5); whether anything changes there is not documented. Every time is one whole
          slice, all chunks included, the best of five. Write down two predictions before you run <code>--bench</code>.
        </p>
      </Translated>
      <Source section="21.2, 21.4" pages="518, 521–522" />
      <LabPrediction
        id="ch21/labs/p-scatter"
        prompt={
          <Translated
            vi={
              <>
                Ở so sánh A, kernel gather của bạn sẽ chạy nhanh hơn kernel scatter bao nhiêu lần trên GPU của bạn? Hãy nghĩ xem
                mọi luồng của một chunk đang làm gì ở cùng một thời điểm.
              </>
            }
          >
            <>
              On comparison A, how many times faster than your scatter kernel will your gather kernel run on your GPU? Think
              about what every thread of a chunk is doing at the same moment.
            </>
          </Translated>
        }
        unit={tr('× the scatter kernel', '× kernel scatter')}
        measure={gatherOverScatter}
        missing={MISSING}
        reading={
          <Translated
            vi={
              <>
                Sách chỉ nói rằng phép toán nguyên tử làm kernel scatter chậm đi "đáng kể" (p. 518), và không nhà cung cấp nào
                đưa ra tỷ lệ. Hai chi phí chồng lên nhau: mọi lane của một warp cộng vào cùng một điểm ở cùng một bước, nên các
                phép toán nguyên tử trên điểm đó bị tuần tự hóa, và mọi lane đọc một nguyên tử khác nhau, trường hợp chậm của bộ
                nhớ đệm hằng trên NVIDIA và AMD. Apple không ghi nhận chi phí nào trong hai chi phí đó, và trên AMD, dạng của phép
                toán nguyên tử phụ thuộc vào một cờ trình biên dịch.
              </>
            }
          >
            <>
              The book says only that atomics "significantly" slow the scatter kernel (p. 518), and no vendor gives a ratio.
              Two costs stack up: every lane of a warp adds to the same point at the same step, so the atomics on it are
              serialised, and every lane reads a different atom, the slow case of the constant cache on NVIDIA and AMD.
              Apple documents neither cost, and on AMD the atomic's form depends on a compiler flag.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch21/labs/p-strided"
        prompt={
          <Translated
            vi={
              <>
                Ở so sánh B với chunk 64, kernel strided của bạn sẽ chạy nhanh hơn kernel adjacent có sẵn bao nhiêu lần? Ghi lại
                luôn bạn nghĩ khoảng cách đó sẽ tăng hay giảm khi đến chunk 1,024; phần kết quả cho thấy cả bốn kích thước chunk.
              </>
            }
          >
            <>
              On comparison B at chunk 64, how many times faster than the provided adjacent kernel will your strided kernel
              run? Note down too whether you expect that gap to grow or shrink by chunk 1,024; the reveal shows all four
              chunk sizes.
            </>
          </Translated>
        }
        unit={tr('× the adjacent kernel', '× kernel adjacent')}
        measure={stridedOverAdjacent}
        missing={MISSING}
        reading={
          <Translated
            vi={
              <>
                Trên 1 nghĩa là kernel strided nhanh hơn. Trên NVIDIA, quy tắc được ghi nhận cho ra 64 sector cho bốn câu lệnh ghi
                mỗi warp của kernel adjacent và 16 cho kernel strided. Nhưng một lần khởi chạy đọc và ghi mỗi điểm một lần trong
                khi vòng lặp nguyên tử của nó chạy <code>chunk</code> lần cho mỗi điểm, nên dự đoán của chính chúng tôi, unverified,
                là khoảng cách thu hẹp khi chunk tăng. AMD ghi nhận việc gom thành các dòng 64 hoặc 128 byte (ch21-facts B3, E2),
                điều dự đoán được số dòng nhưng không dự đoán được thời gian. Apple không ghi nhận quy tắc hợp nhất nào. Các kích
                thước gần nhau có thể chênh vài phần trăm giữa các lần chạy; hãy chạy benchmark hai lần trước khi đọc quá nhiều
                vào một khoảng cách nhỏ.
              </>
            }
          >
            <>
              Above 1, the strided kernel was faster. On NVIDIA the documented rule gives 64 sectors for the adjacent
              kernel's four write statements per warp and 16 for the strided kernel's. But a launch reads and writes each
              point once while its atom loop runs <code>chunk</code> times per point, so our own prediction, unverified, is
              that the gap shrinks as the chunk grows. AMD documents grouping into 64- or 128-byte lines (ch21-facts B3, E2),
              which predicts the line counts but not the time. Apple documents no coalescing rule. Neighbouring sizes can differ by a few per cent between runs; run the benchmark twice before
              you read much into a small gap.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l1}
        cols={[
          { head: tr('What ran', 'Đã chạy gì'), cell: (r) => KERNEL_LABEL[r.name] ?? r.name },
          rawCol(tr('Comparison', 'So sánh'), 'comparison'),
          { head: tr('Slice', 'Lát'), cell: (r) => `${String(r.nx)} × ${String(r.ny)}` },
          rawCol(tr('Atoms', 'Nguyên tử'), 'atoms'),
          rawCol(tr('Chunk', 'Chunk'), 'chunk'),
          msCol,
          { head: tr('Gpairs/s', 'Gcặp/s'), num: true, cell: gpairs },
          rawCol(tr('Group width', 'Độ rộng nhóm'), 'group_width'),
        ]}
      />
      <Translated
        vi={
          <p>
            Một cặp là một nguyên tử và một điểm, nên số cặp mỗi giây so sánh được các kernel trên những lát kích thước khác
            nhau. Độ rộng nhóm là <code>warpSize</code> trên CUDA và HIP và, trên Metal, là <code>threadExecutionWidth</code>{' '}
            của pipeline kernel đã gộp việc; trình khám phá câu lệnh ghi ở <a href={href('ch21', '21.2')}>21.2</a> và bảng mặt
            nạ ở <a href={href('ch21', '21.5')}>21.5</a> cung cấp độ rộng đã ghi lại của Metal dưới mục "Apple". Trên CUDA, số
            sector mỗi yêu cầu của Nsight Compute cho các lần lưu toàn cục của kernel bạn và của kernel adjacent cho thấy khác
            biệt mà mô hình dự đoán; với truy cập 32 bit bởi cả một warp, mức tối ưu là 4 (ch21-facts A8). Không có gì ở đây đã
            chạy trên một device CUDA.
          </p>
        }
      >
        <p>
          A pair is one atom and one point, so pairs per second compares kernels on slices of different sizes. The group
          width is <code>warpSize</code> on CUDA and HIP and, on Metal, the <code>threadExecutionWidth</code> of the coarsened
          pipeline; the write-statement explorer on <a href={href('ch21', '21.2')}>21.2</a> and the mask panel on{' '}
          <a href={href('ch21', '21.5')}>21.5</a> offer Metal's recorded width under "Apple". On CUDA,
          Nsight Compute's sectors per request for the global stores of your kernel and of the adjacent one shows the
          difference the model predicts; for 32-bit accesses by a full warp the optimum is 4 (ch21-facts A8). Nothing here
          ran on a CUDA device.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Để đi xa hơn: PyTorch đưa ra đúng lựa chọn của lab này cho gradient của một bảng embedding, theo cách khác nhau trên
            hai backend. Trên CUDA và ROCm, nó sắp xếp các id token và cộng các hàng của mỗi id với một người ghi duy nhất, trừ
            khi một heuristic chọn đường nguyên tử của nó và các thuật toán tất định không được yêu cầu; trên MPS, nó cộng bằng
            phép toán nguyên tử float và từ chối ở chế độ tất định (ch21-facts D1, D2). Từ những gì bạn đo được ở so sánh A, bạn
            sẽ chọn cách nào để đưa vào sản phẩm, và khi nào?
          </p>
        }
      >
        <p>
          To go further: PyTorch makes this lab's choice for the gradient of an embedding table, differently on two backends.
          On CUDA and ROCm it sorts the token ids and sums each one's rows with a single writer, unless a heuristic prefers its
          atomic path and deterministic algorithms were not requested; on MPS it adds with float atomics and refuses in
          deterministic mode (ch21-facts D1, D2). From what you measured in comparison A, which would you ship, and when?
        </p>
      </Translated>
      <ReferenceSolution lab={l1} />
    </LabsPage>
  );
}
