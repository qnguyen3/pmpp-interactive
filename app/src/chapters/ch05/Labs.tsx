import type { Question } from '../../components/Ask';
import { Beyond, Source } from '../../components/notes';
import { MATMUL_TIERS } from '../../components/Code';
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
  numCol,
  preferredBench,
  rawCol,
  whose,
} from '../../components/LabKit';
import { fmt, num, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 5's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch05', [
  { num: '5.1', dir: 'lab1_roofline', name: 'roofline' },
  { num: '5.2', dir: 'lab2_tiled', name: 'tiled' },
  { num: '5.3', dir: 'lab3_gemm', name: 'gemm' },
  { num: '5.4', dir: 'lab4_racehunt', name: 'race hunt' },
  { num: '5.5', dir: 'lab5_dynamic', name: 'run-time tile' },
]);
const [l1, l2, l3, l4, l5] = LABS;

// ---------------------------------------------------------------- what each prediction is measured against
/** Lab 5.1: the naive kernel's throughput as a multiple of the model's ceiling. */
const ceilingRatio = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const r = benchRow(s, 'roofline');
    const naive = num(r?.naive_gflops);
    const ceiling = num(r?.naive_model_ceiling_gflops);
    if (!naive || !ceiling) return null;
    return { value: naive / ceiling, detail: tr(
        `${fmt(naive)} GFLOP/s against a model ceiling of ${fmt(ceiling)} GFLOP/s`,
        `${fmt(naive)} GFLOP/s so với mức trần mô hình ${fmt(ceiling)} GFLOP/s`,
      ),
    };
  });
/** Lab 5.2: speed-up of the tiled kernel over the naive one at T = 16. */
const tiledSpeedup = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const r = benchRow(s, 'tiled', { tile: 16 });
    const v = num(r?.speedup_vs_naive);
    if (!r || !v) return null;
    return { value: v, detail: tr(
        `${fmt(num(r.gflops) ?? NaN)} GFLOP/s tiled, width ${String(r.width)}`,
        `${fmt(num(r.gflops) ?? NaN)} GFLOP/s phân ô, độ rộng ${String(r.width)}`,
      ),
    };
  });
/** Lab 5.3: throughput at batch 256 over throughput at batch 1. */
const batchGain = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const one = num(benchRow(s, 'linear_layer', { batch: 1 })?.gflops);
    const many = num(benchRow(s, 'linear_layer', { batch: 256 })?.gflops);
    if (!one || !many) return null;
    return { value: many / one, detail: tr(`${fmt(many)} GFLOP/s at batch 256, ${fmt(one)} at batch 1`, `${fmt(many)} GFLOP/s ở batch 256, ${fmt(one)} ở batch 1`),
    };
  });
/** Lab 5.5: the run-time-tile kernel over lab 5.2's compile-time one, both at T = 16, at the
 *  matrix width lab 5.2's benchmark chose (lab 5.5's times every width that one can choose),
 *  same backend. A lab 5.5 report without that width gives no comparison. */
const dynamicCost = (reports: Reports) => {
  const fixed = preferredBench(reports, l2);
  return measure(reports, l5, (s) => {
    const f = fixed.find((x) => x.backend === s.backend);
    const fix = benchRow(f, 'tiled', { tile: 16 });
    const width = num(fix?.width);
    if (!f || !fix || width === undefined) return null;
    const dyn = benchRow(s, 'tiled_dynamic', { tile: 16, width });
    const a = num(dyn?.gflops);
    const b = num(fix.gflops);
    if (!dyn || !a || !b) return null;
    return {
      // whose() defaults to English; the Vietnamese branch asks for its Vietnamese wording.
      whose: tr(`lab 5.5 from ${whose(s)}, lab 5.2 from ${whose(f)}`, `lab 5.5 từ ${whose(s, 'vi')}, lab 5.2 từ ${whose(f, 'vi')}`),
      value: a / b,
      detail: tr(
        `${fmt(a)} GFLOP/s against ${fmt(b)} GFLOP/s, both at width ${String(width)}`,
        `${fmt(a)} GFLOP/s so với ${fmt(b)} GFLOP/s, cả hai ở độ rộng ${String(width)}`,
      ),
    };
  });
};

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={MATMUL_TIERS}>
      <LabsIntro>
        <Translated
          vi={
            <>
              Năm lab, mỗi lab trong một thư mục riêng dưới <code>labs/ch05/</code>. Bốn lab đầu đi theo chương: đo thiết
              bị của bạn, phân ô kernel, tổng quát hóa nó, gỡ lỗi một kernel. Lab thứ năm biến độ rộng ô thành một lựa chọn
              lúc chạy, đúng như cách mục 5.6 kết thúc.
            </>
          }
        >
          Five labs, each in its own directory under <code>labs/ch05/</code>. The first four follow the chapter: measure
          your device, tile the kernel, generalise it, debug one. The fifth makes the tile width a run-time choice, as
          section 5.6 ends.
        </Translated>
      </LabsIntro>
      <RunningLabs
        example="5.2"
        alsoShownIn={
          <Translated
            vi={
              <>
                biểu đồ roofline của <a href={href('ch05', '5.1')}>mục 5.1</a>
              </>
            }
          >
            <>
              the roofline of <a href={href('ch05', '5.1')}>section 5.1</a>
            </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <VerificationStatus />

      {/* ------------------------------------------------------------ 5.1 */}
      <h2>
        <Translated vi="Lab 5.1: roofline của thiết bị của bạn">Lab 5.1: the roofline of your device</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn viết kernel ngây thơ của chương 3 và phần khởi chạy của nó, bằng trí nhớ nếu được, bằng CUDA và bằng Metal. Hai
            kernel thăm dò được cung cấp sẵn: một phép cộng vector, có tốc độ gần như hoàn toàn do băng thông bộ nhớ quyết định, và
            một chuỗi phép nhân-cộng giữ trong thanh ghi (register), hầu như không chạm vào bộ nhớ. Cùng nhau, chúng đo hai con số
            tạo nên roofline của thiết bị của bạn, rồi benchmark đặt kernel của bạn cạnh mức trần mà mô hình 0.25 FLOP/B đặt cho
            nó. Đây là phép đo mà cả chương lập luận dựa trên, được thực hiện trên phần cứng của bạn chứ không phải của sách.
          </p>
        }
      >
        <p>
          You write the naive kernel of chapter 3 and its launch, from memory if you can, in CUDA and in Metal. Two
          probe kernels are provided: a vector addition, whose speed is set almost entirely by memory bandwidth, and a
          chain of multiply-adds held in registers, which hardly touches memory. Together they measure the two numbers
          of your device's roofline, and the benchmark then places your kernel against the ceiling that the 0.25 FLOP/B
          model gives it. This is the measurement the whole chapter argues from, taken on your hardware and not the
          book's.
        </p>
      </Translated>
      <Source section="5.1" pages="93–97" figs={tr('the kernel is Fig. 3.11, p. 61', 'kernel là Hình 3.11, p. 61')} />
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <Translated vi={<>thân của <code>matmulKernel</code> và phần khởi chạy trong <code>launchMatmul</code></>}>
              <>the body of <code>matmulKernel</code> and the launch in <code>launchMatmul</code></>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<>thân của <code>matmul</code></>}>
              <>the body of <code>matmul</code></>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<>phần gắn buffer và lệnh dispatch trong <code>encodeMatmul</code></>}>
              <>buffer bindings and the dispatch in <code>encodeMatmul</code></>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 5.1</code>, và thêm <code>--bench</code> khi nó đã qua. Các bài kiểm thử kiểm tra
            hai kernel thăm dò trước, rồi so kernel của bạn với một bản tham chiếu trên CPU, trong một dung sai, ở các độ rộng 1,
            5, 16, 33, 100, 257 và 512 trên thiết bị và một danh sách ngắn hơn trên trình giả lập (emulator). Những độ rộng không
            phải bội số của bất kỳ kích thước khối nào được đưa vào có chủ đích.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 5.1</code>, and add <code>--bench</code> once it passes. The tests check
          the two probes first, then your kernel against a CPU reference, within a tolerance, at widths 1, 5, 16, 33,
          100, 257 and 512 on a device and a shorter list on the emulator. Widths that are not a multiple of any block
          size are there on purpose.
        </p>
      </Translated>
      <LabPrediction
        id="ch05/labs/p-lab1-ceiling"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy benchmark. Mô hình đặt cho kernel ngây thơ mức trần bằng 0.25 FLOP/B nhân với băng thông
                đo được của thiết bị của bạn. GFLOP/s đo được của kernel bạn sẽ bằng bao nhiêu lần mức trần đó? (1 nghĩa là
                đúng bằng nó.)
              </>
            }
          >
            <>
              Before you run the benchmark. The model gives the naive kernel a ceiling of 0.25 FLOP/B times your
              device's measured bandwidth. Your kernel's measured GFLOP/s will be what multiple of that ceiling? (1
              means exactly at it.)
            </>
          </Translated>
        }
        unit={tr('× the model ceiling', '× mức trần mô hình')}
        measure={ceilingRatio}
        reading={
          <Translated
            vi={
              <>
                Trên 1, kernel đã vượt một mức trần tính mọi lần nạp vào bộ nhớ toàn cục, điều đó chỉ có thể nghĩa là một
                số lần nạp được phục vụ từ nơi nào đó nhanh hơn; sách ghi chú đúng điều này về bộ nhớ đệm (cache) (§5.1, p.
                97). Dưới 1, kernel vẫn còn chỗ dưới mái.
              </>
            }
          >
            <>
              Above 1, the kernel beat a ceiling that charges every load to global memory, which can only mean some
              loads were served from somewhere faster; the book notes exactly this about caches (§5.1, p. 97). Below
              1, the kernel has room left under the roof.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l1}
        filter={(r) => r.name === 'roofline'}
        cols={[
          numCol(tr('Bandwidth probe, GB/s', 'Thăm dò băng thông, GB/s'), 'bandwidth_gb_per_s'),
          numCol(tr('Arithmetic probe, GFLOP/s', 'Thăm dò số học, GFLOP/s'), 'compute_gflops'),
          numCol(tr('Threshold, FLOP/B', 'Ngưỡng, FLOP/B'), 'threshold_flop_per_byte'),
          numCol(tr('Naive matmul, GFLOP/s', 'Matmul ngây thơ, GFLOP/s'), 'naive_gflops'),
          rawCol(tr('at width', 'ở độ rộng'), 'naive_width'),
          numCol(tr('Model ceiling, GFLOP/s', 'Trần mô hình, GFLOP/s'), 'naive_model_ceiling_gflops'),
        ]}
      />
      <ReferenceSolution lab={l1} />

      {/* ------------------------------------------------------------ 5.2 */}
      <h2>
        <Translated vi="Lab 5.2: kernel phân ô">Lab 5.2: the tiled kernel</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn viết kernel phân ô (tiling) của mục 5.4: hai ô (tile) trong bộ nhớ chia sẻ, và ở mỗi giai đoạn là một lần nạp,
            một rào chắn (barrier), một tích vô hướng từng phần và một rào chắn thứ hai. Hợp đồng là của sách: ma trận vuông có độ
            rộng là bội số của độ rộng ô. Bản build định nghĩa <code>TILE_WIDTH</code> và biên dịch tệp của bạn cho nhiều giá trị,
            nên không chỗ nào trong mã của bạn được giả định là 16. Benchmark sau đó trả lời câu hỏi mà chương cứ hoãn lại: thiết bị
            của bạn biến được bao nhiêu phần trong hệ số T của mô hình thành tốc độ.
          </p>
        }
      >
        <p>
          You write the tiled kernel of section 5.4: two shared tiles, and for every phase a load, a barrier, a partial
          dot product and a second barrier. The contract is the book's, square matrices whose width is a multiple of
          the tile width. The build defines <code>TILE_WIDTH</code> and compiles your file for several values, so
          nothing in your code may assume 16. The benchmark then answers the question the chapter keeps deferring: how
          much of the model's factor of T your device turns into speed.
        </p>
      </Translated>
      <Source section="5.4" pages="108–112" figs={tr('Fig. 5.9', 'Hình 5.9')} />
      <LabFiles
        lab={l2}
        files={[
          [
            'kernel.cu',
            <Translated vi={<><code>tiledMatmulKernel</code> và <code>launchTiledMatmul</code></>}>
              <><code>tiledMatmulKernel</code> and <code>launchTiledMatmul</code></>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<><code>tiled_matmul</code>, cùng các bước đó theo cách gọi của Metal</>}>
              <><code>tiled_matmul</code>, the same steps in Metal's vocabulary</>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<>lệnh dispatch ở cuối <code>encodeTiledMatmul</code></>}>
              <>the dispatch at the end of <code>encodeTiledMatmul</code></>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 5.2</code>. Các bài kiểm thử so kernel của bạn với một bản tham chiếu trên CPU ở
            độ rộng ô 16 cho các độ rộng từ 16 đến 528 trên thiết bị (Metal thêm cả độ rộng ô 8), và ở 16, 32 và 64 trên trình giả
            lập với cả ba thứ tự luồng. Với <code>--bench</code>, một kernel ngây thơ được cung cấp sẵn và kernel phân ô của bạn
            được đo thời gian trên cùng các ma trận: trên Metal ở T = 4, 8, 16 và 32, trên CUDA và HIP ở các độ rộng ô mà bản build
            đã biên dịch (16, và cả 8 và 32).
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 5.2</code>. The tests compare your kernel with a CPU reference at tile
          width 16 for widths from 16 to 528 on a device (Metal also at tile width 8), and at 16, 32 and 64 on the
          emulator under all three thread orders. With <code>--bench</code>, a provided naive kernel and your tiled one
          are timed on the same matrices: on Metal at T = 4, 8, 16 and 32, on CUDA and HIP at the tile widths the build
          compiled (16, and also 8 and 32).
        </p>
      </Translated>
      <LabPrediction
        id="ch05/labs/p-lab2-speedup"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy benchmark. Ở T = 16, mô hình lưu lượng nói kernel phân ô đòi hỏi bộ nhớ toàn cục một phần mười
                sáu so với kernel ngây thơ. Bạn sẽ đo được mức tăng tốc bao nhiêu so với kernel ngây thơ trên thiết bị của
                mình ở T = 16?
              </>
            }
          >
            <>
              Before you run the benchmark. At T = 16 the traffic model says the tiled kernel asks global memory for
              one sixteenth of what the naive kernel asks. What speed-up over the naive kernel will you measure on
              your device at T = 16?
            </>
          </Translated>
        }
        unit={tr('× the naive kernel', '× kernel ngây thơ')}
        measure={tiledSpeedup}
        reading={
          <Translated
            vi={
              <>
                Hệ số 16 là một phát biểu về điều kernel đòi hỏi ở bộ nhớ. Mức tăng tốc là những gì thiết bị làm được từ đó,
                và nó phụ thuộc vào việc bộ nhớ đệm của thiết bị đã hấp thụ sẵn bao nhiêu lưu lượng của kernel ngây thơ. Hãy
                so sánh các độ rộng ô khác trong bảng trước khi giải thích con số này.
              </>
            }
          >
            <>
              The factor of 16 is a statement about what the kernel asks of memory. The speed-up is what the device
              made of it, which depends on how much of the naive kernel's traffic its caches were already
              absorbing. Compare the other tile widths in the table before you explain the number.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l2}
        filter={(r) => r.name === 'naive' || r.name === 'tiled'}
        cols={[
          {
            head: tr('Variant', 'Biến thể'),
            cell: (r) =>
              r.name === 'tiled'
                ? tr(`tiled, T = ${String(r.tile)}`, `phân ô, T = ${String(r.tile)}`)
                : tr('naive (provided)', 'ngây thơ (cung cấp sẵn)'),
          },
          rawCol(tr('Width', 'Độ rộng'), 'width'),
          msCol,
          numCol('GFLOP/s', 'gflops'),
          numCol(tr('Speed-up over naive', 'Tăng tốc so với ngây thơ'), 'speedup_vs_naive'),
        ]}
      />
      <ReferenceSolution lab={l2} />

      {/* ------------------------------------------------------------ 5.3 */}
      <h2>
        <Translated vi="Lab 5.3: một GEMM tổng quát">Lab 5.3: a general GEMM</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn biến kernel lab 5.2 của mình thành <code>C[j×l] = A[j×k] × B[k×l]</code> cho mọi kích thước. Có ba điều thay
            đổi. <code>Width</code> từng mang ba nghĩa khác nhau, và mỗi chỗ dùng nó trở thành j, k hoặc l. Kích thước không còn
            là bội số của ô, nên mỗi lần truy cập bộ nhớ cần phép kiểm tra biên riêng và số giai đoạn được làm tròn lên. Và lưới
            (grid) không còn vuông nữa.
          </p>
        }
      >
        <p>
          You turn your lab 5.2 kernel into <code>C[j×l] = A[j×k] × B[k×l]</code> for any sizes. Three things change.{' '}
          <code>Width</code> meant three different things, and each use becomes j, k or l. Sizes are no longer multiples
          of the tile, so every memory access needs its own bounds check and the number of phases rounds up. And the
          grid is no longer square.
        </p>
      </Translated>
      <Source section="5.5" pages="112–115" figs={tr('Fig. 5.13', 'Hình 5.13')} />
      <Translated
        vi={
          <p>
            Đây là kernel đứng sau mọi lớp tuyến tính, với j là kích thước batch, k là số đầu vào và l là số đầu ra. Sách dành cho
            ma trận chữ nhật một đoạn văn và để kernel lại làm bài tập (pp. 114–115). Ở đây nó là trọng tâm: nó là mốc dự án đầu
            tiên của dự án xuyên suốt được xây dựng, M2 trong kế hoạch, và là lý do benchmark đo thời gian một lớp chứ không phải
            một ma trận vuông.
          </p>
        }
      >
        <p>
          This is the kernel behind every linear layer, with j the batch size, k the inputs and l the outputs. The book
          gives rectangular matrices one paragraph and leaves the kernel as an exercise (pp. 114–115). Here it is the
          point: it is the first milestone of the through-line project to be built, M2 in the plan, and the reason the
          benchmark times a layer and not a square matrix.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Dự án xuyên suốt, <code>nanoinfer</code>, là của khóa học này: một đường suy luận nhỏ trên GPU cho một transformer chỉ
              có decoder, được phát triển dần qua từng mốc dự án.
            </p>
          }
        >
          <p>
            The through-line project, <code>nanoinfer</code>, is ours: one small GPU inference path for a decoder-only
            transformer, grown a milestone at a time.
          </p>
        </Translated>
      </Beyond>
      <LabFiles
        lab={l3}
        files={[
          [
            'kernel.cu',
            <Translated vi={<><code>gemmKernel</code> và <code>launchGemm</code>, cả hai đều trống trong mã khởi đầu (starter)</>}>
              <><code>gemmKernel</code> and <code>launchGemm</code>, both empty in the starter</>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<>thân của <code>gemm</code></>}>
              <>the body of <code>gemm</code></>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<>lệnh dispatch ở cuối <code>encodeGemm</code></>}>
              <>the dispatch at the end of <code>encodeGemm</code></>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 5.3</code>. Các bài kiểm thử là những hình dạng được chọn để làm hỏng một kernel
            vẫn còn nghĩ theo ma trận vuông: ví dụ 3×3 của sách với ô rộng hơn, một phần tử duy nhất, đúng một ô, một trường hợp
            chữ nhật mà mọi thứ đều là bội số của ô, một trường hợp không gì là bội số, một hàng duy nhất (một token qua một lớp
            tuyến tính), một tích ngoài, và trên thiết bị thêm hai trường hợp lớn và khó chịu. Buffer đầu ra được điền trước một
            giá trị đánh dấu, nên một phần tử mà kernel của bạn không bao giờ ghi sẽ bị phát hiện. Với{' '}
            <code>--bench</code>, một lớp có 1024 đầu vào và 4096 đầu ra được đo thời gian ở các kích thước batch từ 1 đến 1024.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 5.3</code>. The tests are shapes chosen to break a kernel that still
          thinks in squares: the book's 3×3 example with a wider tile, a single element, exactly one tile, a
          rectangular case where everything is a multiple of the tile, one where nothing is, a single row (one token
          through a linear layer), an outer product, and on a device two larger awkward ones. The output buffer is
          filled with a marker value first, so an element your kernel never writes is caught. With{' '}
          <code>--bench</code>, a layer of 1024 inputs and 4096 outputs is timed at batch sizes from 1 to 1024.
        </p>
      </Translated>
      <LabPrediction
        id="ch05/labs/p-lab3-batch"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy benchmark. Cùng kernel, cùng lớp 1024 → 4096: GFLOP/s của nó ở batch 256 chia cho GFLOP/s ở
                batch 1 sẽ bằng bao nhiêu?
              </>
            }
          >
            <>
              Before you run the benchmark. The same kernel, the same 1024 → 4096 layer: its GFLOP/s at batch 256
              divided by its GFLOP/s at batch 1 will be what?
            </>
          </Translated>
        }
        unit={tr('× the batch-1 throughput', '× thông lượng ở batch 1')}
        measure={batchGain}
        reading={
          <Translated
            vi={
              <>
                Mục 5.1 đã đếm vì sao thông lượng nên tăng theo batch: trọng số được nạp một lần và phục vụ mọi hàng. Tỉ lệ
                đo được cho biết kernel này và thiết bị của bạn hiện thực hóa được bao nhiêu phần trong đó. Hãy xem mức tăng
                dừng ở đâu trong bảng, và xem số hàng mỗi giây, con số mà một máy chủ quan tâm.
              </>
            }
          >
            <>
              Section 5.1 counted why throughput should rise with the batch: the weights are fetched once and serve
              every row. The measured ratio is how much of that this kernel and your device deliver. Look at where
              the rise stops in the table, and at rows per second, which is the figure a server cares about.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l3}
        filter={(r) => r.name === 'linear_layer'}
        cols={[
          rawCol(tr('Batch', 'Batch'), 'batch'),
          msCol,
          numCol('GFLOP/s', 'gflops'),
          numCol(tr('Rows per second', 'Số hàng mỗi giây'), 'rows_per_s'),
        ]}
      />
      <ReferenceSolution lab={l3} />

      {/* ------------------------------------------------------------ 5.4 */}
      <h2>
        <Translated vi="Lab 5.4: truy tìm race condition">Lab 5.4: race hunt</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn được giao một kernel phân ô, có kiểm tra biên, cho ma trận vuông với độ rộng bất kỳ, bằng CUDA và bằng Metal, có
            ba lỗi cài sẵn độc lập với nhau. Nó biên dịch được, và có thể qua khi độ rộng là bội số của độ rộng ô. Bạn tìm và sửa cả
            ba. Đây là những lỗi mà kernel của chương này dễ dẫn tới, và tìm ra chúng trong mã bạn không viết là kỹ năng bạn sẽ cần
            cho mã của chính mình.
          </p>
        }
      >
        <p>
          You are handed a tiled, boundary-checked kernel for square matrices of any width, in CUDA and in Metal, with
          three independent defects planted in it. It compiles, and it can pass when the width is a multiple of the
          tile width. You find and fix all three. These are the mistakes this chapter's kernel invites, and finding
          them in code you did not write is the skill you will need for your own.
        </p>
      </Translated>
      <Source section="5.5" pages="115" figs={tr('after Fig. 5.13', 'sau Hình 5.13')} />
      <LabFiles
        lab={l4}
        files={[
          [
            'kernel.cu',
            <Translated vi={<><code>tiledMatmulKernel</code>, với ba lỗi cài sẵn</>}>
              <><code>tiledMatmulKernel</code>, with the three defects</>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<><code>tiled_matmul</code>, với cùng ba lỗi đó</>}>
              <><code>tiled_matmul</code>, with the same three</>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<>được cung cấp sẵn và đúng; không cần thay đổi gì</>}>
              <>provided and correct; nothing to change</>
            </Translated>,
          ],
        ]}
      />
      <Translated vi={<p>Phương pháp quan trọng hơn các bản sửa:</p>}>
        <p>The method matters more than the fixes:</p>
      </Translated>
      <Translated
        vi={
          <ol>
            <li>
              Chạy trên trình giả lập trước: <code>python3 labs/pmpp.py run 5.4 --backend emu</code>. Các thứ tự luồng của nó cố
              định, nên một lỗi phụ thuộc thời điểm sẽ thất bại theo cùng một cách mỗi lần. Trên GPU, cùng lỗi đó có thể qua ở một
              số lần chạy.
            </li>
            <li>
              Nếu lần chạy bị hủy giữa chừng, hãy đọc báo cáo AddressSanitizer ở cuối đầu ra. Khung stack (stack frame) đầu tiên
              của nó chỉ ra hàm kernel và dòng của <code>kernel.cu</code> đã chạm vào vùng nhớ mà nó không được chạm.
            </li>
            <li>
              Sửa từng lỗi một và chạy lại. Ghi lại độ rộng nào thất bại và dưới thứ tự luồng nào, trước và sau mỗi lần sửa: mẫu
              hình đó là bằng chứng của bạn về lỗi tiếp theo là gì.
            </li>
            <li>
              Sửa đúng ba lỗi đó trong <code>kernel.metal</code> và chạy trên thiết bị.
            </li>
          </ol>
        }
      >
        <ol>
          <li>
            Run on the emulator first: <code>python3 labs/pmpp.py run 5.4 --backend emu</code>. Its thread orders are
            fixed, so a defect that depends on timing fails the same way every time. On a GPU the same defect may pass
            on some runs.
          </li>
          <li>
            If the run aborts, read the AddressSanitizer report at the end of the output. Its first stack frame names
            the kernel function and the line of <code>kernel.cu</code> that touched memory it should not have.
          </li>
          <li>
            Fix one defect at a time and run again. Note which widths fail and under which thread orders before and
            after each fix: that pattern is your evidence for what the next defect is.
          </li>
          <li>
            Make the same three fixes in <code>kernel.metal</code> and run on the device.
          </li>
        </ol>
      </Translated>
      <Translated
        vi={
          <p>
            Các bài kiểm thử dùng độ rộng 16, 32, 3, 24, 40 và 50, cùng với 200 và 333 trên thiết bị, mỗi độ rộng chạy hai lần
            trên dữ liệu mới và so với một bản tham chiếu trên CPU. Không có benchmark.
          </p>
        }
      >
        <p>
          The tests use widths 16, 32, 3, 24, 40 and 50, and 200 and 333 as well on a device, each run twice on fresh
          data and compared with a CPU reference. There is no benchmark.
        </p>
      </Translated>
      <LabResults
        lab={l4}
        noBench={tr(
          'This lab has no benchmark. The result is the line above: every test passing on the emulator and on a device. As shipped, the kernel is meant to fail.',
          'Lab này không có benchmark. Kết quả là dòng ở trên: mọi bài kiểm thử đều qua trên trình giả lập và trên thiết bị. Ở dạng được phát hành, kernel được thiết kế để thất bại.',
        )}
      />
      <ReferenceSolution lab={l4} />

      {/* ------------------------------------------------------------ 5.5 */}
      <h2>
        <Translated vi="Lab 5.5: độ rộng ô được chọn lúc chạy">Lab 5.5: tile width chosen at run time</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn viết kernel vuông phân ô, có kiểm tra biên, thêm một lần nữa, với độ rộng ô là một tham số và không có{' '}
            <code>#define</code>. Trong CUDA, điều đó nghĩa là một mảng <code>extern __shared__</code> không khai báo kích thước,
            được chia thành hai ô, với kích thước tính bằng byte là tham số khởi chạy thứ ba. Trong Metal, nó nghĩa là một tham số
            kernel <code>threadgroup float*</code> được gắn bằng <code>[[threadgroup(0)]]</code>, có kích thước do host đặt bằng{' '}
            <code>setThreadgroupMemoryLength:atIndex:</code>. Khi đó một tệp nhị phân phục vụ mọi độ rộng ô, và chương trình điều
            khiển chọn độ rộng lớn nhất mà thiết bị của bạn báo là còn chỗ.
          </p>
        }
      >
        <p>
          You write the tiled, boundary-checked square kernel once more, with the tile width as an argument and no{' '}
          <code>#define</code>. In CUDA that means one unsized <code>extern __shared__</code> array carved into two
          tiles, with its size in bytes as the third launch parameter. In Metal it means a{' '}
          <code>threadgroup float*</code> kernel parameter bound with <code>[[threadgroup(0)]]</code>, whose size the
          host sets with <code>setThreadgroupMemoryLength:atIndex:</code>. One binary then serves every tile width, and
          the driver picks the largest one your device reports room for.
        </p>
      </Translated>
      <Source section="5.6" pages="116–118" figs={tr('Fig. 5.14', 'Hình 5.14')} />
      <Translated
        vi={
          <p>
            Đây là bản chạy được tương ứng với Hình 5.14 của sách, vốn không biên dịch được như in.{' '}
            <a href={href('ch05', '5.6')}>Mục 5.6</a> cho bạn tìm chỗ sai trên trang sách; ở đây bạn viết phiên bản đúng.
          </p>
        }
      >
        <p>
          This is the working counterpart of the book's Fig. 5.14, which does not compile as printed.{' '}
          <a href={href('ch05', '5.6')}>Section 5.6</a> has you find what is wrong with the page; here you write the
          version that is right.
        </p>
      </Translated>
      <LabFiles
        lab={l5}
        files={[
          [
            'kernel.cu',
            <Translated vi={<><code>tiledMatmulDynamicKernel</code> và số byte trong <code>launchTiledMatmulDynamic</code></>}>
              <><code>tiledMatmulDynamicKernel</code> and the byte count in <code>launchTiledMatmulDynamic</code></>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<>thân của <code>tiled_matmul_dynamic</code>, hàm nhận các ô dưới dạng một con trỏ</>}>
              <>the body of <code>tiled_matmul_dynamic</code>, which receives the tiles as one pointer</>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<>độ dài bộ nhớ threadgroup và lệnh dispatch trong <code>encodeTiledMatmulDynamic</code></>}>
              <>the threadgroup memory length and the dispatch in <code>encodeTiledMatmulDynamic</code></>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 5.5</code>. Chương trình điều khiển trước hết tìm ô lũy thừa của hai lớn nhất mà
            khối của nó vừa với hai giới hạn thiết bị báo cáo, số luồng mỗi khối và bộ nhớ chia sẻ mỗi khối, rồi in chúng ra. Sau
            đó nó kiểm thử các độ rộng ô 4, 8 và 16, và cả ô lớn nhất đó nếu lớn hơn, mỗi ô ở các độ rộng T, 2T, 3 và 2T + 5, và ở
            200 và 333 trên thiết bị. Với <code>--bench</code>, nó đo thời gian các độ rộng 512, 1024, 2048 và 4096 ở mọi ô lũy
            thừa của hai từ 4 đến ô lớn nhất. Đó là những độ rộng mà benchmark của lab 5.2 có thể dừng lại (nó nhân đôi từ 512 cho
            đến khi kernel ngây thơ đủ chậm để đo, và dừng ở 4096), nên phép so sánh bên dưới được thực hiện ở độ rộng lab 5.2 đã
            dùng.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 5.5</code>. The driver first works out the largest power-of-two tile whose
          block fits two limits the device reports, threads per block and shared memory per block, and prints them. It
          then tests tile widths 4, 8 and 16, and that largest one if it is bigger, each at widths T, 2T, 3 and 2T + 5,
          and at 200 and 333 on a device. With <code>--bench</code> it times widths 512, 1024, 2048 and 4096 at every
          power-of-two tile from 4 up to the largest. Those are the widths lab 5.2's benchmark can settle on (it
          doubles from 512 until its naive kernel is slow enough to time, and stops at 4096), so the comparison below
          is made at the width lab 5.2 used.
        </p>
      </Translated>
      <LabPrediction
        id="ch05/labs/p-lab5-dynamic"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy benchmark. Ở T = 16 và cùng độ rộng ma trận, GFLOP/s của kernel này chia cho GFLOP/s của
                kernel lab 5.2, vốn có độ rộng ô cố định lúc biên dịch, sẽ bằng bao nhiêu? (1 nghĩa là cùng tốc độ, dưới 1
                nghĩa là phiên bản chọn lúc chạy chậm hơn.)
              </>
            }
          >
            <>
              Before you run the benchmark. At T = 16 and the same matrix width, this kernel's GFLOP/s divided by the
              GFLOP/s of lab 5.2's kernel, whose tile width is fixed at compile time, will be what? (1 means the same
              speed, below 1 means the run-time version is slower.)
            </>
          </Translated>
        }
        unit={tr('× the compile-time kernel', '× kernel cố định lúc biên dịch')}
        measure={dynamicCost}
        missing={
          <Translated
            vi={
              <>
                Phép so sánh này cần các dòng benchmark của lab 5.5 và của lab 5.2 từ cùng một backend, ở độ rộng ma trận
                mà lab 5.2 đã chọn. Hãy chạy <code>./verify.sh</code> không kèm <code>--quick</code> trên một máy có GPU.
                Một benchmark lab 5.5 được ghi trước 2026-11-21 chỉ đo độ rộng 2048; hãy chạy lại nếu lab 5.2 đã chọn độ
                rộng khác.
              </>
            }
          >
            <>
              This comparison needs benchmark rows for lab 5.5 and for lab 5.2 from the same backend, at the matrix
              width lab 5.2 chose. Run <code>./verify.sh</code> without <code>--quick</code> on a machine with a GPU. A
              lab 5.5 benchmark recorded before 2026-11-21 timed width 2048 only; run it again if lab 5.2 chose
              another width.
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                Cùng thuật toán, cùng độ rộng ô và độ rộng ma trận. Điều khác biệt là độ rộng ô giờ là một giá trị kernel
                đọc vào, trong khi trước đó nó là một hằng số trình biên dịch nhìn thấy. Bảng kết quả bên dưới cũng cho thấy
                tốc độ của kernel chọn lúc chạy thay đổi theo độ rộng ma trận: chỉ so sánh ở cùng một độ rộng.
              </>
            }
          >
            <>
              Same algorithm, tile width and matrix width. What differs is that the tile width is a value the kernel
              reads, where before it was a constant the compiler saw. The results table below shows how the run-time
              kernel's speed changes with the matrix width as well: compare at one width only.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l5}
        filter={(r) => r.name === 'tiled_dynamic'}
        cols={[
          rawCol('T', 'tile'),
          rawCol(tr('Width', 'Độ rộng'), 'width'),
          numCol(tr('Shared bytes per block', 'Số byte chia sẻ mỗi khối'), 'shared_bytes_per_block', 6),
          msCol,
          numCol('GFLOP/s', 'gflops'),
        ]}
      />
      <ReferenceSolution lab={l5} />
    </LabsPage>
  );
}
