import type { Question } from '../../components/Ask';
import { Predict } from '../../components/Ask';
import { MATMUL_TIERS } from '../../components/Code';
import { Aside, Beyond, Source, Today, type Tier } from '../../components/notes';
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
  rawCol,
  useLabs,
  type BenchSet,
} from '../../components/LabKit';
import { Translated, tr } from '../../lib/i18n';
import { fmt, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 3's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch03', [
  { num: '3.1', dir: 'lab1_grayscale', name: 'grayscale', vi: { name: 'ảnh xám' } },
  { num: '3.2', dir: 'lab2_blur', name: 'blur', vi: { name: 'làm mờ' } },
  { num: '3.3', dir: 'lab3_naive_gemm', name: 'naive GEMM (M1)', vi: { name: 'GEMM ngây thơ (M1)' } },
  { num: '3.4', dir: 'lab4_granularity', name: 'granularity', vi: { name: 'độ mịn công việc' } },
  { num: '3.5', dir: 'lab5_maphunt', name: 'map hunt', vi: { name: 'săn lỗi ánh xạ' } },
]);
const [l1, l2, l3, l4, l5] = LABS;

// Identifier colours for the reference solutions: buffers are global memory, everything
// a thread computes for itself is a register.
const reg = (...names: string[]): Record<string, Tier> => Object.fromEntries(names.map((n) => [n, 'reg' as Tier]));
const glob = (...names: string[]): Record<string, Tier> => Object.fromEntries(names.map((n) => [n, 'global' as Tier]));
const GRAY_TIERS = { ...glob('Pin', 'Pout'), ...reg('col', 'row', 'grayOffset', 'rgbOffset', 'r', 'g', 'b', 'gid') };
const BLUR_TIERS = { ...glob('in', 'out'), ...reg('col', 'row', 'pixVal', 'pixels', 'blurRow', 'blurCol', 'curRow', 'curCol', 'gid') };
const GEMM_TIERS = { ...MATMUL_TIERS, ...reg('gid', 'sum', 'Cvalue') };
const SEQ_TIERS = { ...glob('in', 'out'), ...reg('h', 't', 'b', 'sum', 'count', 'cur', 'd', 'gid') };

// ---------------------------------------------------------------- what each prediction is measured against
const seconds = (set: BenchSet, name: string, where: (r: BenchRow) => boolean) =>
  num(set.rows.find((r) => r.name === name && where(r))?.seconds);
const ms = (s: number) => `${fmt(s * 1000)} ms`; // i18n-ignore: number and unit

/** Lab 3.1: time with blocks of 1 by 256 over time with blocks of 256 by 1, same picture. */
const tallOverWide = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const wide = seconds(s, 'grayscale', (r) => r.block_w === 256 && r.block_h === 1);
    const tall = seconds(s, 'grayscale', (r) => r.block_w === 1 && r.block_h === 256);
    if (!wide || !tall) return null;
    return { value: tall / wide, detail: tr(`${ms(tall)} with blocks 1 wide and 256 high, ${ms(wide)} with blocks 256 wide and 1 high`, `${ms(tall)} với khối rộng 1 cao 256, ${ms(wide)} với khối rộng 256 cao 1`) };
  });
/** Lab 3.2: time at radius 4 over time at radius 1, same picture. */
const radiusCost = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const r1 = seconds(s, 'blur', (r) => r.radius === 1);
    const r4 = seconds(s, 'blur', (r) => r.radius === 4);
    if (!r1 || !r4) return null;
    return { value: r4 / r1, detail: tr(`${ms(r4)} at r = 4, ${ms(r1)} at r = 1`, `${ms(r4)} ở r = 4, ${ms(r1)} ở r = 1`) };
  });
/** Lab 3.3: time at width 1024 over time at width 512. */
const doublingCost = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const half = seconds(s, 'square', (r) => r.width === 512);
    const full = seconds(s, 'square', (r) => r.width === 1024);
    if (!half || !full) return null;
    return { value: full / half, detail: tr(`${ms(full)} at width 1024, ${ms(half)} at width 512`, `${ms(full)} ở chiều rộng 1024, ${ms(half)} ở chiều rộng 512`) };
  });
/** Lab 3.4: time of one thread per row over time of one thread per element, at the width the driver chose. */
const rowOverElement = (reports: Reports) =>
  measure(reports, l4, (s) => {
    const row = s.rows.find((r) => r.name === 'granularity' && r.mode === 'row');
    const element = seconds(s, 'granularity', (r) => r.mode === 'element');
    const perRow = num(row?.seconds);
    if (!row || !perRow || !element) return null;
    return { value: perRow / element, detail: tr(`${ms(perRow)} per row, ${ms(element)} per element, at width ${String(row.width)}`, `${ms(perRow)} khi mỗi luồng một hàng, ${ms(element)} khi mỗi luồng một phần tử, ở chiều rộng ${String(row.width)}`) };
  });

// ---------------------------------------------------------------- lab 3.5: which named tests failed
const NAMED = ['cube', 'wide', 'tall', 'odd batch', 'window'] as const; // i18n-ignore: test case names from the driver
// Backends that are not a GPU: the emulator and its race-detection flavour.
const EMULATED = new Set(['emu', 'race']);

interface NamedRun {
  backend: string;
  device: string;
  date: string;
  untouched: boolean;
  failed: string[];
  passed: string[];
}
/** Per device, how the five named tests of lab 3.5 went in the last run of the reader's work/. */
function namedRuns(reports: Reports): NamedRun[] {
  return Object.keys(reports)
    .filter((b) => !EMULATED.has(b))
    .flatMap((b) => {
      const run = reports[b];
      const tests = run.results.filter((x) => x.lab === l5.id && x.variant === 'work').flatMap((x) => x.tests);
      const of = (name: string) => tests.find((t) => t.case.startsWith(name + ':'));
      // A count means something only if all five were run.
      if (!NAMED.every((name) => of(name))) return [];
      return [{
        backend: b,
        device: String(run.device.name ?? b),
        date: run.generated.slice(0, 10),
        untouched: run.work_state?.[l5.id] === 'untouched',
        failed: NAMED.filter((name) => !of(name)!.pass),
        passed: NAMED.filter((name) => of(name)!.pass),
      }];
    });
}

function NamedReveal({ committed }: { committed: number }) {
  const { reports, loading, reload } = useLabs();
  const runs = namedRuns(reports);
  return (
    <>
      {runs.length ? (
        runs.map((r) => (
          <p key={r.backend} data-measured={r.failed.length} data-backend={r.backend}>
            <Translated
              vi={
                <>
                  Trên {r.device} ({r.backend}), trong lần chạy ngày {r.date},{' '}
                  {r.untouched ? 'khi work/ vẫn như lúc phát hành' : 'sau khi work/ đã được sửa'}: <b>{r.failed.length}</b> trên
                  5 phép kiểm thử có tên thất bại{r.failed.length ? ` (${r.failed.join(', ')})` : ''}
                  {r.passed.length ? `; ${r.passed.join(', ')} vượt qua` : ''}.{' '}
                  {committed === r.failed.length
                    ? 'Đó đúng là con số bạn đã dự đoán. Hãy kiểm tra xem các tên có khớp với những tên bạn đã ghi lại không.'
                    : 'Đó không phải con số bạn đã dự đoán. Hãy so các tên với những tên bạn đã ghi lại.'}
                </>
              }
            >
              On {r.device} ({r.backend}), in the run of {r.date},{' '}
              {r.untouched ? 'with work/ still as shipped' : 'after work/ had been edited'}: <b>{r.failed.length}</b> of
              the 5 named tests failed{r.failed.length ? ` (${r.failed.join(', ')})` : ''}
              {r.passed.length ? `; ${r.passed.join(', ')} passed` : ''}.{' '}
              {committed === r.failed.length
                ? 'That is the number you predicted. Check that the names match the ones you wrote down.'
                : 'That is not the number you predicted. Compare the names with the ones you wrote down.'}
            </Translated>
          </p>
        ))
      ) : loading ? (
        <p data-measured="loading"><Translated vi="Đang đọc báo cáo.">Reading reports.</Translated></p>
      ) : (
        <p data-measured="none">
          <Translated
            vi={
              <>
                Chưa có lần chạy nào bài làm của bạn cho lab này trên một thiết bị. Hãy chạy <code>python3 labs/pmpp.py run 3.5</code>{' '}
                trên một máy có GPU.{' '}
              </>
            }
          >
            No run of your work for this lab on a device yet. Run <code>python3 labs/pmpp.py run 3.5</code> on a machine
            with a GPU.{' '}
          </Translated>
          <button className="quiet" onClick={reload}>
            <Translated vi="Tải lại kết quả">Reload results</Translated>
          </button>
        </p>
      )}
      <p className="muted">
        <Translated
          vi={
            <>
              Con số này lấy từ một thiết bị, nơi mọi phép kiểm thử chạy đến cùng. Trên trình giả lập (emulator), một lần chạy dừng
              ở lần truy cập đầu tiên ra ngoài một buffer, nên các phép kiểm thử sau đó hoàn toàn không được chạy. Khi bạn bắt đầu
              sửa lỗi, con số giảm xuống; thứ bạn đã dự đoán là kernel như lúc phát hành.
            </>
          }
        >
          The count is from a device, where every test runs to the end. On the emulator a run stops at the first access
          outside a buffer, so the tests after that one are not run at all. Once you start fixing defects the count
          falls; what you predicted was the kernel as shipped.
        </Translated>
      </p>
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
              Năm lab, mỗi lab trong một thư mục riêng dưới <code>labs/ch03/</code>. Ba lab đầu là ba kernel của chương, do bạn
              viết: ảnh xám, làm mờ, và phép nhân ma trận cho mọi hình dạng. Lab thứ tư đo một lựa chọn thiết kế mà sách để lại
              làm bài tập. Lab thứ năm là một cuộc săn lỗi.
            </>
          }
        >
          Five labs, each in its own directory under <code>labs/ch03/</code>. The first three are the chapter's three
          kernels, written by you: grayscale, blur, and matrix multiplication for any shape. The fourth measures a
          design choice the book leaves as an exercise. The fifth is a bug hunt.
        </Translated>
      </LabsIntro>
      <RunningLabs
        example="3.3"
        alsoShownIn={
          <>
            <Translated
              vi={
                <>phần dự đoán xếp hạng của <a href={href('ch03', '3.4')}>mục 3.4</a></>
              }
            >
              the ranking prediction of <a href={href('ch03', '3.4')}>section 3.4</a>
            </Translated>
          </>
        }
      />
      <BackendsAndEmulator />
      <Translated
        vi={
          <p>
            Trình giả lập có một biến thể thứ hai, <code>race</code>, mà các lab 3.3, 3.4 và 3.5 cũng được dựng cho. Nó chạy các
            luồng giả lập đồng thời dưới ThreadSanitizer và báo hai luồng chạm vào cùng một phần tử mà không đồng bộ hóa, kèm
            dòng mã nguồn. Yêu cầu nó bằng <code>--backend race</code>;{' '}
            <code>python3 labs/pmpp.py doctor</code> cho biết trình biên dịch của bạn có dựng được nó không. Các bảng trên trang
            này chỉ hiện biến thể thứ nhất.
          </p>
        }
      >
        <p>
          The emulator has a second flavour, <code>race</code>, which labs 3.3, 3.4 and 3.5 are also built for. It runs
          the emulated threads concurrently under ThreadSanitizer and reports two threads that touch the same element
          without synchronisation, with the source line. Ask for it with <code>--backend race</code>;{' '}
          <code>python3 labs/pmpp.py doctor</code> says whether your compiler can build it. The tables on this page
          show the first flavour only.
        </p>
      </Translated>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 3.1 */}
      <h2><Translated vi="Lab 3.1: ảnh màu sang ảnh xám">Lab 3.1: colour to grayscale</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn viết kernel của <a href={href('ch03', '3.2')}>mục 3.2</a> và cấu hình khởi chạy của nó: mỗi luồng một điểm ảnh,
            một lưới 2D gồm các khối 2D làm tròn lên thành nguyên khối, phép kiểm tra phạm vi, và hai offset phẳng, một byte mỗi
            điểm ảnh đi ra và ba byte đi vào. Mức xám là L = 0.299 r + 0.587 g + 0.114 b, cắt bớt về một byte. Hình dạng khối là
            một đối số của cấu hình khởi chạy, nên không chỗ nào trong mã của bạn được giả định 16.
          </p>
        }
      >
        <p>
          You write the kernel of <a href={href('ch03', '3.2')}>section 3.2</a> and its launch: one thread per pixel, a
          2D grid of 2D blocks rounded up to whole blocks, the range test, and two flat offsets, one byte per pixel
          going out and three coming in. The grey level is L = 0.299 r + 0.587 g + 0.114 b, truncated to a byte. The
          block shape is an argument of the launch, so nothing in your code may assume 16.
        </p>
      </Translated>
      <Source section="3.2" pages="48–55" figs={tr('Fig. 3.4, p. 53; the launch is on p. 50', 'Hình 3.4, p. 53; cấu hình khởi chạy ở p. 50')} />
      <Translated
        vi={
          <p>
            Đừng sao chép Fig. 3.4. Kernel của lab khác bản in ở bốn chỗ: phép ép kiểu, các hằng số, thứ tự đối số và tên. Mục
            3.2 cho bạn tự tìm ra chúng, và <code>README.md</code> của lab liệt kê chúng. Nếu bạn không tìm ra, các phép kiểm thử
            sẽ tìm ra phép ép kiểu và các hằng số giúp bạn.
          </p>
        }
      >
        <p>
          Do not copy Fig. 3.4. The lab's kernel differs from the printed one in four places: the cast, the constants,
          the argument order and the name. Section 3.2 has you find them, and the lab's <code>README.md</code> lists
          them. The tests find the cast and the constants for you if you do not.
        </p>
      </Translated>
      <LabFiles
        lab={l1}
        files={[
          ['kernel.cu', <Translated vi={<><code>colorToGrayscaleKernel</code> và <code>launchGrayscale</code></>}><><code>colorToGrayscaleKernel</code> and <code>launchGrayscale</code></></Translated>],
          ['kernel.metal', <Translated vi={<>thân của <code>grayscale</code>, trên các buffer <code>uchar</code> thuần</>}><>the body of <code>grayscale</code>, over plain <code>uchar</code> buffers</></Translated>],
          ['launch.mm', <Translated vi={<>hai lời gọi <code>setBytes</code> và lệnh dispatch trong <code>encodeGrayscale</code></>}><>the two <code>setBytes</code> calls and the dispatch in <code>encodeGrayscale</code></></Translated>],
        ]}
      />
      <Translated
        vi={
          <p>
            Cấu hình khởi chạy Metal dùng <code>dispatchThreads</code>, nên lưới khớp đúng bức ảnh và bạn phải quyết định kernel
            có còn cần phép kiểm tra phạm vi không. Ở đó một điểm ảnh cũng là ba byte riêng rẽ: một{' '}
            <code>uchar3</code> chiếm bốn byte trong Metal Shading Language.
          </p>
        }
      >
        <p>
          The Metal launch uses <code>dispatchThreads</code>, so the grid is exactly the picture and you have to decide
          whether the kernel still needs its range test. A pixel is three separate bytes there too: a{' '}
          <code>uchar3</code> is four bytes in the Metal Shading Language.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Kernel và cấu hình khởi chạy Metal là của khóa học này. Chương của sách chỉ dùng CUDA.</p>}>
          <p>The Metal kernel and launch are ours. The book's chapter is CUDA only.</p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 3.1</code>, bắt đầu với <code>--backend emu</code>. Đầu ra của bạn được so với
            một kết quả tham chiếu trên CPU, cho phép lệch một mức xám về mỗi phía, vì phép ép kiểu cắt bớt và thứ tự của ba phép
            cộng có thể làm byte lệch đi một. Các ảnh đơn sắc chạy trước: đỏ, xanh lá và xanh dương thuần phải cho 76, 149 và 29,
            còn trắng là 255. Sau đó là một dải chuyển màu có nhiễu ở 62 × 76 (bức ảnh của sách), 16 × 16, 1 × 1, 1 × 300,
            300 × 1, 33 × 17 và 64 × 80, và trên thiết bị là 1500 × 2000, mỗi kích thước với các khối 16 × 16, 32 × 8 và 8 × 32
            luồng. Đầu ra được điền trước bằng một giá trị đánh dấu, nên một điểm ảnh bạn không bao giờ ghi sẽ bị đếm. Trên
            thiết bị, các buffer lớn hơn bức ảnh và các byte sau nó phải giữ nguyên.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 3.1</code>, starting with <code>--backend emu</code>. Your output is
          compared with a CPU reference, allowing one grey level either way, because the cast truncates and the order
          of three additions can move the byte by one. Flat pictures come first: pure red, green and blue must give 76,
          149 and 29, and white 255. Then a gradient with noise at 62 by 76 (the book's picture), 16 by 16, 1 by 1, 1
          by 300, 300 by 1, 33 by 17 and 64 by 80, and on a device 1500 by 2000, each with blocks of 16 by 16, 32 by 8
          and 8 by 32 threads. The output is filled with a marker value first, so a pixel you never write is counted.
          On a device the buffers are larger than the picture and the bytes after it must come back unchanged.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Sau đó benchmark đo thời gian một bức ảnh lớn với nhiều hình dạng khối. Ở hai trong số đó, một khối chứa 256 luồng:
            một dãy điểm ảnh dọc theo một hàng ở hình dạng này, một dãy xuống theo một cột ở hình dạng kia. Hướng dẫn của NVIDIA
            nói rằng "the use of multi-dimensional thread blocks and grids is for convenience only and does not affect
            performance", câu này nói về việc viết một phép ánh xạ ở một, hai hoặc ba chiều. Cùng đoạn đó tiếp tục đánh số các
            luồng của một khối với x thay đổi nhanh nhất và thêm: "This affects how threads are assigned to warps", một từ mà
            chương 4 giải thích. Không câu nào cho bạn biết hai hình dạng này sẽ cho kết quả ra sao.
          </p>
        }
      >
        <p>
          The benchmark then times one large picture under several block shapes. In two of them a block holds 256
          threads: a run of pixels along a row in one, a run down a column in the other. NVIDIA's guide says that "the
          use of multi-dimensional thread blocks and grids is for convenience only and does not affect performance",
          which is about writing one mapping in one, two or three dimensions. The same passage goes on to number a
          block's threads x fastest and adds: "This affects how threads are assigned to warps", a word chapter 4
          explains. Neither sentence tells you what these two shapes will do.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, A2)">{/* i18n-ignore: date and notes reference */}
        <Translated vi={<p>Cả hai trích dẫn đều từ hướng dẫn lập trình CUDA của NVIDIA.</p>}>
          <p>Both quotations are from NVIDIA's CUDA programming guide.</p>
        </Translated>
      </Today>
      <LabPrediction
        id="ch03/labs/p-lab1-tall-wide"
        prompt={
          <>
            <Translated
              vi={
                <>
                  Trước khi bạn chạy benchmark. Nó đo thời gian một bức ảnh lớn với nhiều hình dạng khối. Khối rộng 256 cao 1, và
                  khối rộng 1 cao 256, khởi chạy cùng số luồng, và mọi luồng làm cùng một việc. Thời gian với khối cao chia cho thời
                  gian với khối rộng sẽ là bao nhiêu? (1 nghĩa là như nhau.)
                </>
              }
            >
              Before you run the benchmark. It times one large picture with several block shapes. Blocks 256 wide and 1
              high, and blocks 1 wide and 256 high, launch the same number of threads, and every thread does the same
              work. The time with the tall blocks divided by the time with the wide ones will be what? (1 means the
              same.)
            </Translated>
          </>
        }
        unit={tr('× the wide-block time', '× thời gian với khối rộng')}
        measure={tallOverWide}
        missing={
          <>
            <Translated
              vi={
                <>
                  Chạy <code>python3 labs/pmpp.py run 3.1 --bench</code> trên một máy có GPU. Benchmark bỏ qua hình dạng khối mà thiết
                  bị không cho phép, và thiếu một trong hai hình dạng thì không có tỉ số.
                </>
              }
            >
              Run <code>python3 labs/pmpp.py run 3.1 --bench</code> on a machine with a GPU. The benchmark skips a block
              shape the device does not allow, and without both shapes there is no ratio.
            </Translated>
          </>
        }
        reading={
          <>
            <Translated
              vi={
                <>
                  Cả hai lần khởi chạy làm cùng phép tính số học trên cùng các byte. Điều khác nhau là các luồng của một khối giữ
                  những điểm ảnh nào: một dãy dọc theo một hàng, hay một dãy xuống theo một cột. Vì sao điều đó có thể quan trọng là
                  chủ đề của chương 4 và 6; hiện tại con số này là một quan sát để mang theo đến đó.
                </>
              }
            >
              Both launches do the same arithmetic on the same bytes. What differs is which pixels the threads of one
              block hold: a run along a row, or a run down a column. Why that can matter is the subject of chapters 4
              and 6; for now the number is an observation to carry there.
            </Translated>
          </>
        }
      />
      <LabResults
        lab={l1}
        filter={(r) => r.name === 'grayscale'}
        cols={[
          { head: tr('Block, wide × high', 'Khối, rộng × cao'), cell: (r) => `${String(r.block_w)} × ${String(r.block_h)}` },
          { head: tr('Picture, rows × columns', 'Ảnh, hàng × cột'), cell: (r) => `${String(r.height)} × ${String(r.width)}` },
          msCol,
          numCol(tr('Megapixels per second', 'Megapixel mỗi giây'), 'mpix_per_s'),
        ]}
      />
      <ReferenceSolution lab={l1} tiers={GRAY_TIERS} />

      {/* ------------------------------------------------------------ 3.2 */}
      <h2><Translated vi="Lab 3.2: làm mờ ảnh">Lab 3.2: image blur</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn viết kernel làm mờ của <a href={href('ch03', '3.3')}>mục 3.3</a> và cấu hình khởi chạy của nó. Phép ánh xạ là của
            lab 3.1, nhưng giờ mỗi luồng lặp: nó lấy trung bình các điểm ảnh đầu vào trong một vùng lân cận vuông quanh vị trí
            của chính nó và bỏ phần vùng lân cận nhô ra ngoài mép. Hai phép kiểm tra, một tổng cộng dồn, một số đếm, và một phép
            chia cho số đếm.
          </p>
        }
      >
        <p>
          You write the blur kernel of <a href={href('ch03', '3.3')}>section 3.3</a> and its launch. The mapping is lab
          3.1's, but now each thread loops: it averages the input pixels in a square patch around its own position and
          leaves out the part of the patch that hangs over an edge. Two tests, a running sum, a count, and a division
          by the count.
        </p>
      </Translated>
      <Source section="3.3" pages="55–59" figs={tr('Fig. 3.8, p. 57', 'Hình 3.8, p. 57')} />
      <Translated
        vi={
          <p>
            Một chỗ khác với trang sách: sách cố định bán kính vùng lân cận bằng một hằng số <code>BLUR_SIZE</code>. Ở đây bán
            kính là một đối số của kernel, <code>r</code>, để benchmark có thể thay đổi nó. Vùng lân cận có cạnh 2r + 1 điểm ảnh.
            Sách không in cấu hình khởi chạy nào cho kernel này; của bạn là cấu hình khởi chạy của ảnh xám với khối 16 × 16.
          </p>
        }
      >
        <p>
          One difference from the page: the book fixes the patch radius with a constant <code>BLUR_SIZE</code>. Here
          the radius is a kernel argument <code>r</code>, so the benchmark can change it. The patch is 2r + 1 pixels on
          a side. The book prints no launch for this kernel; yours is the grayscale launch with 16 by 16 blocks.
        </p>
      </Translated>
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <Translated vi={<><code>blurKernel</code> và <code>launchBlur</code></>}><><code>blurKernel</code> and <code>launchBlur</code></></Translated>],
          ['kernel.metal', <Translated vi={<>thân của <code>blur</code>. Vị trí Metal đưa cho bạn là số không dấu; các offset bên trong một vùng lân cận thì không</>}><>the body of <code>blur</code>. The position Metal hands you is unsigned; the offsets inside a patch are not</></Translated>],
          ['launch.mm', <Translated vi={<>lệnh dispatch trong <code>encodeBlur</code>, với <code>dispatchThreadgroups</code>, vốn đếm theo nhóm nên làm tròn lên như một lưới CUDA</>}><>the dispatch in <code>encodeBlur</code>, with <code>dispatchThreadgroups</code>, which counts groups and so rounds up like a CUDA grid</></Translated>],
        ]}
      />
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 3.2</code>. Phép so sánh lại cho phép lệch một mức xám. Các hình dạng là 62 × 76
            với r = 1, 2 và 3, một điểm ảnh, một hàng và một cột, 3 × 3 với r = 3 khi vùng lân cận lớn hơn bức ảnh, 33 × 17 với
            r = 7, và trên thiết bị là 1500 × 2000. Một lỗi được báo bằng hai số đếm, số điểm ảnh viền sai và số điểm ảnh bên
            trong sai. Viền sai mà bên trong đúng là dấu hiệu chỉ về phép kiểm tra bên trong hoặc về số bạn chia; bên trong sai
            là dấu hiệu chỉ về phép ánh xạ.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 3.2</code>. The comparison again allows one grey level. The shapes are 62
          by 76 with r = 1, 2 and 3, a single pixel, a single row and a single column, 3 by 3 with r = 3 where the patch
          is larger than the picture, 33 by 17 with r = 7, and on a device 1500 by 2000. A failure is reported as two
          counts, wrong border pixels and wrong interior pixels. A wrong border with a correct interior points at the
          inner test or at what you divide by; a wrong interior points at the mapping.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Phép kiểm thử đầu tiên đưa cho kernel 62 hàng trên cùng của một buffer 66 hàng. Một lần đọc vượt quá mép phải của
            một hàng sẽ rơi vào đầu hàng kế tiếp, vốn là bộ nhớ hợp lệ, nên không trình kiểm tra biên nào phản đối và chỉ các giá
            trị mới cho thấy lỗi. Trong một buffer vừa đúng kích thước, cùng lỗi đó chỉ bị bắt ở hàng cuối cùng.
          </p>
        }
      >
        <p>
          The first test hands the kernel the top 62 rows of a 66-row buffer. A read that runs past the right edge of a
          row lands on the start of the next row, which is valid memory, so no bounds checker objects and only the
          values can show it. In a buffer of exact size the same mistake is caught at the last row alone.
        </p>
      </Translated>
      <LabPrediction
        id="ch03/labs/p-lab2-radius"
        prompt={
          <>
            <Translated
              vi={
                <>
                  Trước khi bạn chạy benchmark. Nó làm mờ một bức ảnh lớn ở r = 1, 2, 4 và 8. Đi từ r = 1 lên r = 4, mỗi luồng đọc 81
                  giá trị thay vì 9. Thời gian ở r = 4 chia cho thời gian ở r = 1 sẽ là bao nhiêu?
                </>
              }
            >
              Before you run the benchmark. It blurs one large picture at r = 1, 2, 4 and 8. Going from r = 1 to r = 4,
              each thread reads 81 values where it read 9. The time at r = 4 divided by the time at r = 1 will be what?
            </Translated>
          </>
        }
        unit={tr('× the time at r = 1', '× thời gian ở r = 1')}
        measure={radiusCost}
        missing={
          <>
            <Translated
              vi={
                <>Chạy <code>python3 labs/pmpp.py run 3.2 --bench</code> trên một máy có GPU.</>
              }
            >
              Run <code>python3 labs/pmpp.py run 3.2 --bench</code> on a machine with a GPU.
            </Translated>
          </>
        }
        reading={
          <>
            <Translated
              vi={
                <>
                  Số lần đọc mỗi luồng gấp chín lần là một nhận định về thứ kernel yêu cầu. Tỉ số là thứ thiết bị của bạn tạo ra từ
                  đó. Hãy xem thêm r = 2 và r = 8 trong bảng trước khi giải thích con số, và chạy hai lần: trên một máy bận rộn, các
                  thời gian này dao động.
                </>
              }
            >
              Nine times the reads per thread is a statement about what the kernel asks for. The ratio is what your
              device made of it. Look at r = 2 and r = 8 in the table as well before you explain the number, and run it
              twice: on a busy machine these timings move.
            </Translated>
          </>
        }
      />
      <LabResults
        lab={l2}
        filter={(r) => r.name === 'blur'}
        cols={[
          rawCol('r', 'radius'),
          {
            head: tr('Values per patch', 'Số giá trị mỗi vùng lân cận'),
            num: true,
            cell: (r) => {
              const radius = num(r.radius);
              return radius == null ? '' : String((2 * radius + 1) ** 2);
            },
          },
          { head: tr('Picture, rows × columns', 'Ảnh, hàng × cột'), cell: (r) => `${String(r.height)} × ${String(r.width)}` },
          msCol,
          numCol(tr('Megapixels per second', 'Megapixel mỗi giây'), 'mpix_per_s'),
        ]}
      />
      <ReferenceSolution lab={l2} tiers={BLUR_TIERS} />

      {/* ------------------------------------------------------------ 3.3 */}
      <h2><Translated vi="Lab 3.3: GEMM ngây thơ cho một lớp tuyến tính">Lab 3.3: naive GEMM for a linear layer</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn viết kernel của Fig. 3.11 cho <code>C[j×l] = A[j×k] × B[k×l]</code>: <code>float</code> theo thứ tự hàng
            (row-major), mọi kích thước, mỗi luồng một phần tử của C, khối 16 × 16. Hình chỉ có một <code>Width</code>. Bạn có
            ba kích thước, và mỗi <code>Width</code> trong hình trở thành đúng một trong số chúng. x của lưới phủ l và y phủ j.
          </p>
        }
      >
        <p>
          You write the kernel of Fig. 3.11 for <code>C[j×l] = A[j×k] × B[k×l]</code>: row-major <code>float</code>, any
          sizes, one thread per element of C, 16 by 16 blocks. The figure has one <code>Width</code>. You have three
          sizes, and every <code>Width</code> in the figure becomes exactly one of them. The grid's x covers l and its
          y covers j.
        </p>
      </Translated>
      <Source section="3.4" pages="59–64" figs={tr('Fig. 3.11, p. 61; the rectangular definition is on p. 60', 'Hình 3.11, p. 61; định nghĩa cho ma trận chữ nhật ở p. 60')} />
      <Translated
        vi={
          <p>
            Đây là nửa đầu của mốc dự án M1 trong dự án xuyên suốt. Xét như một lớp tuyến tính, j là batch, đứng đầu, k là số đầu
            vào và l là số đầu ra, và B giữ trọng số dạng (in, out). Đó là chuyển vị của thứ PyTorch lưu, (out, in), và của thứ
            ggml nhân với; cả hai đều đọc hai toán hạng dọc theo hàng, trong khi kernel này đọc toán hạng thứ hai xuống theo một
            cột. Lab giữ quy ước của sách vì <a href={href('ch05', 'labs')}>lab 5.3</a> cũng giữ nó: cùng thứ tự đối số, cùng cách
            lưu, nên về sau kernel phân ô thay thế kernel này mà không phải động đến bên gọi. Một phép kiểm thử dựng một trọng số
            lưu theo cách của PyTorch, chuyển vị nó trên host và kiểm tra <code>y = x Wᵀ</code>.
          </p>
        }
      >
        <p>
          This is the first half of milestone M1 of the through-line project. As a linear layer, j is the batch, which
          comes first, k the number of inputs and l the number of outputs, and B holds the weight as (in, out). That is
          the transpose of what PyTorch stores, (out, in), and of what ggml multiplies by; both read their two operands
          along rows, where this kernel reads its second operand down a column. The lab keeps the book's convention
          because <a href={href('ch05', 'labs')}>lab 5.3</a> keeps it too: same argument order, same storage, so the
          tiled kernel replaces this one later without touching the caller. One of the tests builds a weight stored
          PyTorch's way, transposes it on the host and checks <code>y = x Wᵀ</code>.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Dự án xuyên suốt, <code>nanoinfer</code>, là của khóa học này, và kernel cho ma trận chữ nhật cùng cấu hình khởi chạy
              của nó cũng vậy: kernel của sách chỉ dành cho ma trận vuông và sách không in cấu hình khởi chạy nào cho nó.
            </p>
          }
        >
          <p>
            The through-line project, <code>nanoinfer</code>, is ours, and so are the rectangular kernel and its launch:
            the book's kernel is square only and it prints no launch for it.
          </p>
        </Translated>
      </Beyond>
      <LabFiles
        lab={l3}
        files={[
          ['kernel.cu', <Translated vi={<><code>naiveGemmKernel</code> và <code>launchNaiveGemm</code>, cả hai đều trống trong mã khởi đầu (starter)</>}><><code>naiveGemmKernel</code> and <code>launchNaiveGemm</code>, both empty in the starter</></Translated>],
          ['kernel.metal', <Translated vi={<>thân của <code>naive_gemm</code></>}><>the body of <code>naive_gemm</code></></Translated>],
          ['launch.mm', <Translated vi={<>lệnh dispatch ở cuối <code>encodeNaiveGemm</code></>}><>the dispatch at the end of <code>encodeNaiveGemm</code></></Translated>],
        ]}
      />
      <Translated
        vi={
          <p>
            Trên Metal, kernel nhận <code>[[thread_position_in_grid]]</code>, cấu hình khởi chạy làm tròn lên thành nguyên các
            threadgroup 16 × 16 với <code>dispatchThreadgroups</code>, và kernel giữ phép kiểm tra của nó. Backend MPS của PyTorch
            khởi chạy kernel matmul Metal đơn giản của nó theo cùng cách, và backend Metal của llama.cpp luôn dispatch nguyên các
            threadgroup. MLX gọi <code>dispatchThreads</code>, vốn khởi chạy đúng lưới bạn yêu cầu và không cần phép kiểm tra.
            Cách nào cũng vượt qua các phép kiểm thử.
          </p>
        }
      >
        <p>
          On Metal the kernel takes <code>[[thread_position_in_grid]]</code>, the launch rounds up to whole 16 by 16
          threadgroups with <code>dispatchThreadgroups</code>, and the kernel keeps its guard. PyTorch's MPS backend
          launches its plain Metal matmul kernel the same way, and llama.cpp's Metal backend dispatches whole
          threadgroups everywhere. MLX calls <code>dispatchThreads</code>, which launches exactly the grid you ask for and needs no
          guard. Either passes the tests.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, C3, D3, D4, D5)">{/* i18n-ignore: date and notes reference */}
        <Translated
          vi={
            <p>
              PyTorch làm tròn lên với các nhóm 16 × 16 và kiểm tra phạm vi trước khi ghi. Các kernel của llama.cpp giới hạn một
              vòng lặp có bước nhảy ở chỗ lab này có phép kiểm tra phạm vi. Các kernel mỗi luồng một phần tử của MLX không có phép
              kiểm tra.
            </p>
          }
        >
          <p>
            PyTorch rounds up with 16 × 16 groups and tests the range before its store. llama.cpp's kernels bound a
            strided loop where this lab has a range test. MLX's one-element-per-thread kernels have no guard.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 3.3</code>. Các hình dạng là ma trận vuông cỡ 4 (kích thước của Fig. 3.12 trong
            sách), 16, 33 và 64, rồi 5 × 7 × 3, một hàng duy nhất (một token đi qua một lớp), một cột đầu ra duy nhất (ma trận
            nhân vector, bài tập 2 của sách), chiều bên trong bằng 1 (một tích ngoài), 17 × 40 × 33, trường hợp bố cục PyTorch,
            và trên thiết bị là hai hình dạng lớn hơn. C được điền trước bằng một giá trị đánh dấu, nên phần tử bạn không bao giờ
            ghi sẽ bị báo, và trên thiết bị một giá trị ghi vượt quá cuối C cũng bị báo. Với <code>--bench</code>, các chiều rộng
            vuông 256, 512 và 1024 được đo thời gian, rồi đến lớp 1024 → 4096 của lab 5.3 ở batch 1, 16 và 256.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 3.3</code>. The shapes are square at 4 (the size of the book's Fig. 3.12),
          16, 33 and 64, then 5 × 7 × 3, a single row (one token through a layer), a single output column (matrix times
          vector, book exercise 2), an inner dimension of 1 (an outer product), 17 × 40 × 33, the PyTorch-layout case,
          and on a device two larger ones. C is filled with a marker value first, so an element you never write is
          reported, and on a device a value stored past the end of C is reported too. With <code>--bench</code>, square
          widths 256, 512 and 1024 are timed, then the 1024 → 4096 layer of lab 5.3 at batch 1, 16 and 256.
        </p>
      </Translated>
      <LabPrediction
        id="ch03/labs/p-lab3-doubling"
        prompt={
          <>
            <Translated
              vi={
                <>
                  Trước khi bạn chạy benchmark. Tăng gấp đôi chiều rộng từ 512 lên 1024 nhân lượng phép tính số học lên 8. Thời gian
                  của kernel ở chiều rộng 1024 chia cho thời gian ở chiều rộng 512 sẽ là bao nhiêu?
                </>
              }
            >
              Before you run the benchmark. Doubling the width from 512 to 1024 multiplies the arithmetic by 8. The
              kernel's time at width 1024 divided by its time at width 512 will be what?
            </Translated>
          </>
        }
        unit={tr('× the time at width 512', '× thời gian ở chiều rộng 512')}
        measure={doublingCost}
        missing={
          <>
            <Translated
              vi={
                <>Chạy <code>python3 labs/pmpp.py run 3.3 --bench</code> trên một máy có GPU.</>
              }
            >
              Run <code>python3 labs/pmpp.py run 3.3 --bench</code> on a machine with a GPU.
            </Translated>
          </>
        }
        reading={
          <>
            <Translated
              vi={
                <>
                  Tám là con số mà số phép toán dự đoán. Dù bạn đo được gì, cột GFLOP/s của bảng là mức nền mà chương 5 bắt đầu: lab
                  5.3 đo cùng lớp đó với cùng tên hàng, nên hai bảng có thể đọc song song.
                </>
              }
            >
              Eight is what the operation count predicts. Whatever you measured, the GFLOP/s column of the table is the
              baseline that chapter 5 starts from: lab 5.3 times the same layer with the same row names, so the two
              tables can be read side by side.
            </Translated>
          </>
        }
      />
      <LabResults
        lab={l3}
        filter={(r) => r.name === 'square' || r.name === 'linear_layer'}
        cols={[
          {
            head: tr('Shape, j × k × l', 'Hình dạng, j × k × l'),
            cell: (r) =>
              r.name === 'square'
                ? `${String(r.width)} × ${String(r.width)} × ${String(r.width)}`
                : tr(`${String(r.batch)} × ${String(r.k)} × ${String(r.l)} (layer, batch ${String(r.batch)})`, `${String(r.batch)} × ${String(r.k)} × ${String(r.l)} (lớp, batch ${String(r.batch)})`),
          },
          msCol,
          numCol('GFLOP/s', 'gflops'),
          numCol(tr('Rows per second', 'Số hàng mỗi giây'), 'rows_per_s'),
          numCol(tr('Time over half the width', 'Thời gian so với nửa chiều rộng'), 'time_vs_half_width'),
        ]}
      />
      <ReferenceSolution lab={l3} tiers={GEMM_TIERS} />

      {/* ------------------------------------------------------------ 3.4 */}
      <h2><Translated vi="Lab 3.4: mỗi luồng một phần tử, một hàng hay một cột">Lab 3.4: one thread per element, per row or per column</Translated></h2>
      <Translated
        vi={
          <p>
            Fig. 3.11 cho mỗi phần tử của P một luồng riêng. Bài tập đầu tiên của sách yêu cầu hai thiết kế thô hơn: một luồng
            tính trọn một hàng đầu ra, và một luồng tính trọn một cột đầu ra. Bạn viết cả hai, mỗi cái trên một lưới 1D gồm các
            khối 256 luồng có phép kiểm tra, cho ma trận vuông như trong bài tập. Kernel mỗi luồng một phần tử và cấu hình khởi
            chạy của nó được cung cấp sẵn trong cùng tệp.
          </p>
        }
      >
        <p>
          Fig. 3.11 gives every element of P its own thread. The book's first exercise asks for two coarser designs: one
          thread computes a whole output row, and one thread computes a whole output column. You write both, each on a
          1D grid of 256-thread blocks with a guard, for square matrices as in the exercise. The per-element kernel and
          its launch are provided in the same file.
        </p>
      </Translated>
      <Source section="3.5" pages="64" figs={tr('exercises 1a and 1b; the kernel is Fig. 3.11, p. 61', 'bài tập 1a và 1b; kernel là Hình 3.11, p. 61')} />
      <LabFiles
        lab={l4}
        files={[
          ['kernel.cu', <Translated vi={<><code>rowKernel</code> cùng <code>launchRow</code>, và <code>colKernel</code> cùng <code>launchCol</code></>}><><code>rowKernel</code> with <code>launchRow</code>, and <code>colKernel</code> with <code>launchCol</code></></Translated>],
          ['kernel.metal', <Translated vi={<><code>row_matmul</code> và <code>col_matmul</code>, mỗi cái nhận một <code>[[thread_position_in_grid]]</code> vô hướng</>}><><code>row_matmul</code> and <code>col_matmul</code>, each taking a scalar <code>[[thread_position_in_grid]]</code></></Translated>],
          ['launch.mm', <Translated vi={<>các lệnh dispatch 1D trong <code>encodeRow</code> và <code>encodeCol</code></>}><>the 1D dispatches in <code>encodeRow</code> and <code>encodeCol</code></></Translated>],
        ]}
      />
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 3.4</code>. Mỗi kernel trong ba kernel được kiểm thử và báo cáo riêng, ở các chiều
            rộng 1, 4, 16, 33 và 100, và trên thiết bị thêm 257 và 512. Như lúc phát hành, kernel được cung cấp vượt qua còn hai
            kernel của bạn thất bại. Phần lớn các chiều rộng này để một phần của khối 256 luồng không có việc, đó là lý do cần
            phép kiểm tra.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 3.4</code>. Each of the three kernels is tested and reported on its own, at
          widths 1, 4, 16, 33 and 100, and on a device also 257 and 512. As shipped, the provided kernel passes and your
          two fail. Most of these widths leave part of a 256-thread block without work, which is what the guard is for.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Với <code>--bench</code>, cả ba kernel được đo thời gian ở một chiều rộng. Chương trình điều khiển (driver) chọn nó:
            nó nhân đôi chiều rộng cho đến khi kernel mỗi luồng một phần tử chạy đủ lâu để đo, và dừng sớm hơn nếu một lần nhân
            đôi nữa sẽ đẩy kernel chậm nhất quá nửa giây mỗi lần chạy. Nó in ra chiều rộng đã chọn và lý do.
          </p>
        }
      >
        <p>
          With <code>--bench</code>, all three kernels are timed at one width. The driver picks it: it doubles the width
          until the per-element kernel runs long enough to time, and stops earlier if one more doubling would push the
          slowest kernel past half a second a run. It prints the width it chose and why.
        </p>
      </Translated>
      <LabPrediction
        id="ch03/labs/p-lab4-row"
        prompt={
          <>
            <Translated
              vi={
                <>
                  Trước khi bạn chạy benchmark. <a href={href('ch03', '3.4')}>Mục 3.4</a> đã yêu cầu bạn xếp hạng ba kernel. Giờ hãy cam
                  kết một hệ số: ở chiều rộng của benchmark, thời gian khi mỗi luồng một hàng chia cho thời gian khi mỗi luồng một phần
                  tử sẽ là bao nhiêu? Cả ba làm cùng phép tính số học. Chúng khác nhau ở số luồng chia nhau phần việc đó và ở hướng mà
                  mỗi luồng ghi.
                </>
              }
            >
              Before you run the benchmark. <a href={href('ch03', '3.4')}>Section 3.4</a> asked you to rank the three
              kernels. Now commit to a factor: at the benchmark's width, the time of one thread per row divided by the
              time of one thread per element will be what? All three do the same arithmetic. They differ in how many
              threads share it and in which direction each thread writes.
            </Translated>
          </>
        }
        unit={tr('× the per-element time', '× thời gian khi mỗi luồng một phần tử')}
        measure={rowOverElement}
        missing={
          <>
            <Translated
              vi={
                <>Chạy <code>python3 labs/pmpp.py run 3.4 --bench</code> trên một máy có GPU.</>
              }
            >
              Run <code>python3 labs/pmpp.py run 3.4 --bench</code> on a machine with a GPU.
            </Translated>
          </>
        }
        reading={
          <>
            <Translated
              vi={
                <>
                  Bảng có số luồng có việc bên cạnh mỗi thời gian. Bài tập 1c hỏi ưu và nhược điểm của mỗi thiết kế; hãy viết câu trả
                  lời của bạn ngay khi đã có số liệu, và ghi lại chiều rộng mà chúng được đo, vì tỉ số này không phải một hằng số của
                  thiết bị.
                </>
              }
            >
              The table has the number of threads with work beside each time. Exercise 1c asks for the pros and cons of
              each design; write yours now that you have numbers, and note the width they were taken at, because the
              ratio is not a constant of the device.
            </Translated>
          </>
        }
      />
      <LabResults
        lab={l4}
        filter={(r) => r.name === 'granularity'}
        cols={[
          {
            head: tr('One thread per', 'Mỗi luồng một'),
            cell: (r) => (r.mode === 'element' ? tr('element (provided)', 'phần tử (cung cấp sẵn)') : r.mode === 'row' ? tr('row', 'hàng') : r.mode === 'col' ? tr('column', 'cột') : String(r.mode ?? '')),
          },
          rawCol(tr('Width', 'Chiều rộng'), 'width'),
          numCol(tr('Threads with work', 'Số luồng có việc'), 'threads', 7),
          msCol,
          numCol('GFLOP/s', 'gflops'),
          numCol(tr('Time over per-element', 'Thời gian so với mỗi luồng một phần tử'), 'time_vs_element'),
        ]}
      />
      <ReferenceSolution lab={l4} tiers={GEMM_TIERS} />

      {/* ------------------------------------------------------------ 3.5 */}
      <h2><Translated vi="Lab 3.5: săn lỗi ánh xạ">Lab 3.5: map hunt</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn được giao một kernel và cấu hình khởi chạy của nó, bằng CUDA và bằng Metal, với bốn lỗi cài sẵn độc lập trong đó.
            Nó biên dịch được, và vượt qua phép kiểm thử đầu tiên. Đây là những lỗi mà phép tính chỉ số của chương này dễ dẫn tới,
            và tìm ra chúng trong mã không phải do bạn viết là kỹ năng bạn sẽ cần cho mã của chính mình.
          </p>
        }
      >
        <p>
          You are handed one kernel and its launch, in CUDA and in Metal, with four independent defects planted in them.
          It compiles, and it passes its first test. These are the mistakes this chapter's index arithmetic invites, and
          finding them in code you did not write is the skill you will need for your own.
        </p>
      </Translated>
      <Source section="3.2" pages="55" figs={tr('the 3D index form; the clipped loop is Fig. 3.8, §3.3, p. 57', 'dạng chỉ số 3D; vòng lặp bị cắt ở biên là Hình 3.8, §3.3, p. 57')} />
      <Translated
        vi={
          <p>
            Kernel lấy trung bình theo cửa sổ dọc trục chuỗi của một tensor <code>[batch, seq, hidden]</code>:{' '}
            <code>out[b][t][h]</code> là trung bình của <code>in[b][s][h]</code> với s từ t − r đến t + r, trên những s tồn tại.
            Hidden nằm trên x, seq trên y, batch trên z, trong các khối 8 × 8 × 4 luồng.
          </p>
        }
      >
        <p>
          The kernel takes a windowed mean along the sequence axis of a <code>[batch, seq, hidden]</code> tensor:{' '}
          <code>out[b][t][h]</code> is the mean of <code>in[b][s][h]</code> over s from t − r to t + r, for the s that
          exist. Hidden is on x, seq on y, batch on z, in blocks of 8 × 8 × 4 threads.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Kernel là của khóa học này, dựng từ hai thứ của sách: phép ánh xạ 3D ở p. 55 và vòng lặp bị cắt ở biên cùng bộ đếm của
              kernel làm mờ, thu về một trục.
            </p>
          }
        >
          <p>
            The kernel is ours, built from two things that are the book's: the 3D mapping of p. 55 and the blur
            kernel's clipped loop with its counter, kept to one axis.
          </p>
        </Translated>
      </Beyond>
      <LabFiles
        lab={l5}
        files={[
          ['kernel.cu', <Translated vi={<><code>seqMeanKernel</code> và <code>launchSeqMean</code>, với bốn lỗi nằm rải giữa hai hàm</>}><><code>seqMeanKernel</code> and <code>launchSeqMean</code>, with the four defects between them</></Translated>],
          ['kernel.metal', <Translated vi={<><code>seq_mean</code>, với cùng các lỗi đó</>}><><code>seq_mean</code>, with the same defects</></Translated>],
          ['launch.mm', <Translated vi={<><code>encodeSeqMean</code>, nơi có lỗi của cấu hình khởi chạy</>}><><code>encodeSeqMean</code>, which has the launch's defect</></Translated>],
        ]}
      />
      <Translated
        vi={
          <p>
            Năm phép kiểm thử đầu tiên có tên. Phép đầu là một khối lập phương, và mỗi phép trong bốn phép tiếp theo đi xa khỏi
            nó theo một hướng:
          </p>
        }
      >
        <p>
          The first five tests are named. The first is a cube, and each of the next four moves away from it in one
          direction:
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Các phép kiểm thử tên <code>mixed</code> sau năm phép đó là những tổ hợp kích thước. Chúng vượt qua khi cả bốn lỗi
              đã được sửa.
            </p>
          }
        >
          <p>
            The tests named <code>mixed</code> after the five are mixtures of sizes. They pass once all four defects are
            gone.
          </p>
        </Translated>
      </Aside>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-table="maphunt-tests">
          <thead>
            <tr><th><Translated vi="Phép kiểm thử">Test</Translated></th><th className="num"><Translated vi="batch">batch</Translated></th><th className="num"><Translated vi="seq">seq</Translated></th><th className="num"><Translated vi="hidden">hidden</Translated></th><th className="num">r</th></tr>
          </thead>
          <tbody>
            <tr><td><code>cube</code></td><td className="num">4</td><td className="num">16</td><td className="num">16</td><td className="num">0</td></tr>
            <tr><td><code>wide</code></td><td className="num">4</td><td className="num">6</td><td className="num">20</td><td className="num">0</td></tr>
            <tr><td><code>tall</code></td><td className="num">4</td><td className="num">20</td><td className="num">6</td><td className="num">0</td></tr>
            <tr><td><code>odd batch</code></td><td className="num">5</td><td className="num">16</td><td className="num">16</td><td className="num">0</td></tr>
            <tr><td><code>window</code></td><td className="num">4</td><td className="num">16</td><td className="num">16</td><td className="num">1</td></tr>
          </tbody>
        </table>
      </div>
      <Predict
        id="ch03/labs/p-lab5-failing"
        prompt={
          <>
            <Translated
              vi={
                <>
                  Trước lần chạy đầu tiên. Đọc kernel và cấu hình khởi chạy trong <code>work/kernel.cu</code>, và ghi lại những phép
                  kiểm thử có tên nào trong năm phép bạn dự đoán sẽ thất bại trên một thiết bị với mã như lúc phát hành. Đó là bao
                  nhiêu phép?
                </>
              }
            >
              Before the first run. Read the kernel and the launch in <code>work/kernel.cu</code>, and write down which
              of the five named tests you expect to fail on a device with the code as shipped. How many is that?
            </Translated>
          </>
        }
        unit={tr('of the 5 named tests', 'trên 5 phép kiểm thử có tên')}
      >
        {(v) => <NamedReveal committed={v} />}
      </Predict>
      <p><Translated vi="Phương pháp quan trọng hơn các bản sửa:">The method matters more than the fixes:</Translated></p>
      <Translated
        vi={
          <ol>
            <li>
              Trình giả lập trước: <code>python3 labs/pmpp.py run 3.5 --backend emu</code>. Sửa một chỗ rồi chạy lại. Các lỗi
              thay đổi theo từng bước.
            </li>
            <li>
              Hãy biết mỗi loại lỗi trông thế nào ở đó. Một lần truy cập ra ngoài tensor dừng lần chạy với một báo cáo
              AddressSanitizer: đọc xem nó ghi READ hay WRITE, khung đầu tiên chỉ tên một dòng của{' '}
              <code>kernel.cu</code>, và địa chỉ nằm trước hay sau buffer bao nhiêu byte. Một lần truy cập sai phần tử bên trong
              tensor là một địa chỉ hợp lệ, nên sanitizer im lặng và bạn nhận được giá trị sai, kèm số phần tử không luồng nào
              ghi. Một lỗi trong cấu hình khởi chạy là mã host bình thường tính vài số nguyên; không công cụ nào đánh dấu nó, và
              bạn chỉ thấy hệ quả của nó.
            </li>
            <li>
              Mỗi phép kiểm thử chạy dưới ba thứ tự luồng. Một kernel đúng cho cùng đầu ra dưới cả ba. Nếu một phép kiểm thử báo
              các giá trị sai khác nhau giữa các thứ tự, thì có hai luồng đang ghi cùng một phần tử.{' '}
              <code>--backend race</code> khi đó sẽ chỉ tên dòng.
            </li>
            <li>
              Hãy dùng các hình dạng. Một phép kiểm thử thất bại đã đi xa khỏi khối lập phương theo một hướng, và hướng đó là thứ
              lỗi cần để lộ ra.
            </li>
            <li>
              Sửa y như vậy trong cặp Metal và chạy trên thiết bị. Ở đó chương trình điều khiển đặt các phần tử canh gác trước và
              sau mỗi tensor, nên một lần ghi lạc chỗ rơi vào bộ nhớ mà phép kiểm thử sở hữu và bị đếm. Với mỗi lỗi, hãy ghi lại
              thiết bị cho bạn thấy gì: một giá trị sai, một phần tử không bao giờ được ghi, hay chỉ một lần ghi vào các phần tử
              canh gác. Một chương trình thật không có phần tử canh gác, và một lỗi thuộc loại cuối cùng để mọi kết quả bạn kiểm
              tra đều đúng.
            </li>
          </ol>
        }
      >
        <ol>
          <li>
            Emulator first: <code>python3 labs/pmpp.py run 3.5 --backend emu</code>. Fix one thing and run again. The
            failures change as you go.
          </li>
          <li>
            Know what each kind of failure looks like there. An access outside the tensor stops the run with an
            AddressSanitizer report: read whether it says READ or WRITE, the first frame naming a line of{' '}
            <code>kernel.cu</code>, and how many bytes before or after the buffer the address is. An access to the wrong
            element inside the tensor is a valid address, so the sanitizer is silent and you get wrong values, with a
            count of elements no thread wrote. A mistake in the launch is ordinary host code that computes a few
            integers; no tool flags it, and you see only its effect.
          </li>
          <li>
            Each test runs under three thread orders. A correct kernel gives the same output under all three. If a test
            reports different wrong values from one order to the next, two threads are writing the same element.{' '}
            <code>--backend race</code> then names the line.
          </li>
          <li>
            Use the shapes. A test that fails moved away from the cube in one direction, and that direction is what the
            defect needs in order to show.
          </li>
          <li>
            Make the same fixes in the Metal pair and run on the device. There the driver puts guard elements before and
            after each tensor, so a stray write lands in memory the test owns and is counted. For each defect, note what
            the device showed you: a wrong value, an element never written, or only a write into the guard elements. A
            real program has no guard elements, and a defect of the last kind leaves every result you check correct.
          </li>
        </ol>
      </Translated>
      <LabResults
        lab={l5}
        noBench={tr('This lab has no benchmark. The result is the line above: every test passing on the emulator and on a device. As shipped, the kernel is meant to fail.', 'Lab này không có benchmark. Kết quả là dòng ở trên: mọi phép kiểm thử vượt qua trên trình giả lập và trên một thiết bị. Như lúc phát hành, kernel được dự định là thất bại.')}
      />
      <ReferenceSolution lab={l5} tiers={SEQ_TIERS} />

      <h2><Translated vi="Những gì các lab này để lại cho sau">What these labs leave for later</Translated></h2>
      <Translated
        vi={
          <p>
            Lab 3.3 là một nửa của M1. Khối 16 × 16 của nó là một lựa chọn cố định; chương 4 hoàn thành mốc dự án bằng cách chọn
            cấu hình khởi chạy từ những gì thiết bị báo cáo. Sau đó chương 5 thay kernel đằng sau cùng những đối số đó bằng một
            kernel phân ô (<a href={href('ch05', 'labs')}>lab 5.3</a>, mốc dự án M2) và đo hai kernel với nhau. Phần{' '}
            <a href={href('ch03', 'wrap')}>tóm tắt</a> có bài kiểm tra tổng hợp của chương và danh sách những chỗ trang in khác
            với mã đã kiểm thử.
          </p>
        }
      >
        <p>
          Lab 3.3 is half of M1. Its 16 by 16 block is a fixed choice; chapter 4 completes the milestone by choosing the
          launch configuration from what the device reports. Chapter 5 then replaces the kernel behind the same
          arguments with a tiled one (<a href={href('ch05', 'labs')}>lab 5.3</a>, milestone M2) and measures the two
          against each other. The <a href={href('ch03', 'wrap')}>summary</a> has the chapter's mixed check and the list
          of places where the printed page differs from the tested code.
        </p>
      </Translated>
    </LabsPage>
  );
}
