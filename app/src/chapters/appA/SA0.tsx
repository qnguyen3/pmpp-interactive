import { useMemo } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Beyond, Source, Today } from '../../components/notes';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { href } from '../../lib/router';
import { TEASER_N, pct, show, teaser, ulpText } from '../../sim/fpsum';

const FACTS = 'notes/today/appA-facts.md';

// ---------------------------------------------------------------- the teaser's reveal
function TeaserReveal({ v }: { v: number }) {
  const locale = useLocale();
  const t = useMemo(() => teaser(), [locale]);
  const half = t.rows.find((r) => r.id === 'fp16-loop')!.e;
  const off = Math.abs(v - half.rel * 100);
  return (
    <>
      <table className="data" style={{ background: 'var(--page)', margin: '0.4rem 0 0.8rem' }} aria-label={tr('Four ways to sum one softmax denominator', 'Bốn cách cộng một mẫu số softmax')}>
        <thead>
          <tr>
            <th><Translated vi="Bộ tích lũy và thứ tự">Accumulator and order</Translated></th>
            <th><Translated vi="Kết quả">Result</Translated></th>
            <th><Translated vi="Sai số tương đối">Relative error</Translated></th>
            <th><Translated vi="Số ULP của bộ tích lũy">ULPs of the accumulator</Translated></th>
          </tr>
        </thead>
        <tbody>
          {t.rows.map((r) => (
            <tr key={r.id} data-row={r.id}>
              <td>
                <Translated
                  vi={{ 'fp32-loop': 'vòng lặp float', 'fp32-tree': 'cây float', 'fp16-loop': 'vòng lặp half', 'fp16-tree': 'cây half' }[r.id]}
                >
                  {r.label}
                </Translated>
              </td>
              <td data-k="result">{show(r.e.result, 9)}</td>
              <td data-k="rel">{pct(r.e.rel)}</td>
              <td data-k="ulps">{ulpText(r.e.ulps)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Giá trị tham chiếu, một tổng có bù ở độ chính xác kép (double), là <b data-k="reference">{show(t.reference, 10)}</b>.
            Vòng lặp half lệch <b>{pct(half.rel)}</b>
            {off < 10 ? ', gần với điều bạn dự đoán.' : v < half.rel * 100 ? ', nhiều hơn hẳn điều bạn dự đoán.' : ', ít hơn điều bạn dự đoán.'}{' '}
            Cũng những số hạng đó, ở cùng định dạng half, cộng theo một cây, cho kết quả lệch không quá một đơn vị ở vị trí cuối (ULP).
          </p>
        }
      >
        <p>
          The reference, a compensated sum in double precision, is <b data-k="reference">{show(t.reference, 10)}</b>. The half
          loop is off by <b>{pct(half.rel)}</b>
          {off < 10 ? ', close to what you predicted.' : v < half.rel * 100 ? ', far more than you predicted.' : ', less than you predicted.'}{' '}
          The same terms in the same half format, added as a tree, land within one unit in the last place.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Dữ liệu không đổi giữa các hàng; chỉ bộ tích lũy và thứ tự thay đổi. Vòng lặp half bị đứng lại: khi tổng đang cộng
            dồn của nó đạt vài trăm, phần lớn các số hạng nhỏ hơn một nửa khoảng cách giữa hai giá trị half kề nhau, và mỗi
            phép cộng làm tròn chúng mất đi. Mục <a href={href('appA', 'A.4')}>A.4</a> cho thấy một phép cộng như vậy, và mục{' '}
            <a href={href('appA', 'A.6')}>A.6</a> cho thấy cần làm gì với vòng lặp.
          </p>
        }
      >
        <p>
          The data did not change between the rows; the accumulator and the order did. The half loop stalls: once its
          running sum is a few hundred, most terms are smaller than half the gap between neighbouring half values, and
          each addition rounds them away. Section <a href={href('appA', 'A.4')}>A.4</a> shows one such addition, and
          section <a href={href('appA', 'A.6')}>A.6</a> shows what to do about the loop.
        </p>
      </Translated>
    </>
  );
}

// ---------------------------------------------------------------- prerequisite questions
export const questionsSA0: Record<string, Question> = {
  'appA/start/q-tolerance': {
    kind: 'choice',
    id: 'appA/start/q-tolerance',
    prompt: (
      <Translated
        vi={
          <>
            Một unit test so sánh tổng theo cây trên GPU của một triệu số float với một vòng lặp trên CPU qua cùng các số đó,
            dùng <code>==</code>. Nó thất bại: hai kết quả chênh nhau 3 ULP. Cách sửa đúng là gì?
          </>
        }
      >
        A unit test compares a GPU tree sum of a million floats with a CPU loop over the same floats, using{' '}
        <code>==</code>. It fails: the two differ by 3 ULPs. What is the right fix?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="So sánh cả hai với một giá trị tham chiếu chính xác, trong một dung sai tỉ lệ với tổng của |x|.">
            Compare both with an accurate reference, within a tolerance scaled to the sum of |x|.
          </Translated>
        ),
        correct: true,
        why: (
          <Translated vi="Không thứ tự nào cho ra tổng chính xác, nên không bên nào có thể làm chuẩn đối chiếu cho bên kia.">
            Neither order gives the exact sum, so neither can be the oracle for the other.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Bắt kernel cộng các phần tử theo thứ tự chỉ số, để GPU và vòng lặp CPU làm tròn giống nhau.">
            Make the kernel add the elements in index order, so the GPU and the CPU loop round alike.
          </Translated>
        ),
        why: (
          <Translated vi="Cách đó từ bỏ tính song song, và vòng lặp cũng đâu có chính xác.">
            That gives up the parallelism, and the loop is not exact either.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Tích lũy bằng double trên GPU, để các bit thấp không còn khác nhau.">
            Accumulate in double on the GPU, so the low bits no longer differ.
          </Translated>
        ),
        why: (
          <Translated vi="Nó thu hẹp khoảng chênh mà không xóa được, và Metal hoàn toàn không có double.">
            It shrinks the gap without closing it, and Metal has no double at all.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="So sánh mẫu bit của hai kết quả thay vì giá trị của chúng.">
            Compare the bit patterns of the two results instead of their values.
          </Translated>
        ),
        why: (
          <Translated vi="Các mẫu bit khác nhau đúng ở chỗ các giá trị khác nhau.">
            The bit patterns differ exactly where the values do.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Phép cộng dấu phẩy động không có tính kết hợp, nên một cây và một vòng lặp làm tròn khác nhau và cả hai đều là kết
            quả hợp lệ. Một bài kiểm thử nên chấp nhận mọi đáp án nằm trong dung sai so với một giá trị tham chiếu chính xác.
            Các lab đã làm như vậy.
          </p>
        }
      >
        <p>
          Floating-point addition is not associative, so a tree and a loop round differently and both are legitimate
          results. A test should accept any answer within a tolerance of an accurate reference. The labs already do this.
        </p>
      </Translated>
    ),
    source: `ch. 10 §10.2, p. 224 (and appendix A §A.6, p. 593); Metal has no double: ${FACTS} C1`,
    vi: { source: `ch. 10 §10.2, p. 224 (và phụ lục A §A.6, p. 593); Metal không có double: ${FACTS} C1` },
  },
  'appA/start/q-convergent': {
    kind: 'numeric',
    id: 'appA/start/q-convergent',
    prompt: (
      <Translated
        vi={
          <>
            Kernel hội tụ của chương 10 (Fig. 10.8) rút gọn 16 phần tử với 8 luồng. Trong lần lặp đầu tiên, luồng 2 cộng phần
            tử nào vào phần tử của chính nó?
          </>
        }
      >
        Chapter 10's convergent kernel (Fig. 10.8) reduces 16 elements with 8 threads. In the first iteration, which
        element does thread 2 add to its own?
      </Translated>
    ),
    answer: 10,
    unit: '(element index)',
    explain: (
      <Translated
        vi={
          <p>
            Luồng i sở hữu phần tử i, và bước nhảy (stride) bắt đầu bằng kích thước khối, 8: luồng i cộng phần tử i + 8. Trên
            dữ liệu đã sắp xếp, điều đó ghép một giá trị nhỏ với một giá trị lớn. Nó quay lại ở mục A.6 dưới dạng bài tập 4
            của sách.
          </p>
        }
      >
        <p>
          Thread i owns element i, and the stride starts at the block size, 8: thread i adds element i + 8. On sorted data
          that pairs a small value with a large one. It comes back in section A.6 as the book's exercise 4.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.4, pp. 232–233',
    vi: { unit: '(chỉ số phần tử)' },
  },
  'appA/start/q-bytes': {
    kind: 'numeric',
    id: 'appA/start/q-bytes',
    prompt: (
      <Translated
        vi={
          <>
            Một phép nhân ma trận-vectơ ở kích thước batch 1 lưu mỗi trọng số trong 1 byte. Tính một phép nhân và một phép cộng
            cho mỗi trọng số, nó thực hiện bao nhiêu phép toán trên mỗi byte lưu lượng trọng số?
          </>
        }
      >
        A matrix-vector product at batch size 1 stores its weights in 1 byte each. Counting a multiply and an add per
        weight, how many operations does it perform per byte of weight traffic?
      </Translated>
    ),
    answer: 2,
    unit: 'FLOP/B',
    explain: (
      <Translated
        vi={
          <p>
            Hai phép toán mỗi trọng số, một byte mỗi trọng số. Lớp tuyến tính của chương 20 ở batch 1 với trọng số 16 bit đạt 1
            FLOP/B; giảm một nửa số byte thì gấp đôi con số đó. Mục A.5 cộng thêm cái giá mà định dạng phải trả: các hệ số tỉ lệ
            lưu kèm mỗi khối trọng số.
          </p>
        }
      >
        <p>
          Two operations per weight, one byte per weight. Chapter 20's linear layer at batch 1 with 16-bit weights gets 1
          FLOP/B; halving the bytes doubles it. Section A.5 adds what the format costs on top: the scales stored with each
          block of weights.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.1, p. 94 (FLOP/B); ch. 20 §20.6, p. 505 (1 FLOP/B at batch 1, 16-bit)',
    vi: { source: 'ch. 5 §5.1, p. 94 (FLOP/B); ch. 20 §20.6, p. 505 (1 FLOP/B ở batch 1, 16 bit)' },
  },
  'appA/start/q-atomic': {
    kind: 'choice',
    id: 'appA/start/q-atomic',
    prompt: (
      <Translated
        vi={
          <>
            Một phép cộng float nhiều khối ghi tổng riêng phần của mỗi khối bằng một phép cộng nguyên tử (atomic). Bạn chạy nó
            20 lần trên cùng đầu vào. Các kết quả có thể khác nhau không?
          </>
        }
      >
        A multi-block float sum commits each block's partial sum with an atomic add. You run it 20 times on the same input.
        Can the results differ?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có: mỗi lần chạy các khối ghi theo một thứ tự khác, và phép cộng float không có tính kết hợp.">
            Yes: blocks commit in a different order each run, and float addition is not associative.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không: mỗi phép cộng atomic là không thể chia cắt, nên lần chạy nào cũng thực hiện cùng các phép cộng và ra cùng tổng.">
            No: each atomic add is indivisible, so every run performs the same additions and reaches the same total.
          </Translated>
        ),
        why: (
          <Translated vi="Các cập nhật không thể chia cắt vẫn đến theo thứ tự bất kỳ. Atomic không cố định thứ tự.">
            Indivisible updates still arrive in any order. Atomics do not fix the order.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ khi hai khối tình cờ ghi đúng cùng một thời điểm.">
            Only if two blocks happen to commit at exactly the same moment.
          </Translated>
        ),
        why: (
          <Translated vi="Các lần ghi đồng thời đằng nào cũng bị tuần tự hóa; chính thứ tự mới là thứ thay đổi.">
            Simultaneous commits are serialised either way; it is the order that varies.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Có, và một tổng số nguyên cũng sẽ khác nhau giữa các lần chạy theo cùng cách đó.">
            Yes, and an integer sum would differ from run to run in the same way.
          </Translated>
        ),
        why: (
          <Translated vi="Phép cộng số nguyên không tràn số có tính kết hợp: thứ tự nào cũng cho cùng một tổng.">
            Integer addition without overflow is associative: any order gives the same total.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một phép cộng atomic cho sự loại trừ lẫn nhau, không cho thứ tự, vì thế toán tử phải có tính kết hợp và giao hoán.
            Với float, điều đó chỉ đúng tới mức làm tròn, nên các bit thấp của tổng có thể đổi từ lần chạy này sang lần chạy khác.
          </p>
        }
      >
        <p>
          An atomic add gives mutual exclusion, not ordering, which is why the operator must be associative and
          commutative. For floats that holds only up to rounding, so the low bits of the total can change from run to run.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.9, p. 245',
  },
};

// ---------------------------------------------------------------- the page
export function SA0() {
  return (
    <>
      <Translated
        vi={
          <p>
            Một bước decoder của LLM, ngoài những thứ khác, là hàng nghìn phép cộng dài. Mỗi tích vô hướng bên trong một GEMM là
            một phép như vậy; mẫu số chuẩn hóa của softmax trên một hàng logits cũng vậy, và cả trung bình bình phương mà một
            phép chuẩn hóa (norm) lấy trên chiều ẩn. Mỗi phép chạy ở độ chính xác hữu hạn, vào một bộ tích lũy có định dạng do
            ai đó chọn. Phụ lục này cho bạn những gì cần để tự đưa ra lựa chọn đó và dự đoán cái giá của nó.
          </p>
        }
      >
        <p>
          One decoder step of an LLM is, among other things, thousands of long sums. Every dot product inside a GEMM is
          one; so is the softmax normaliser over a row of logits, and the mean of squares a norm takes over the hidden
          dimension. Each runs in finite precision, into an accumulator whose format somebody chose. This appendix gives
          you what you need to make that choice yourself and to predict what it costs.
        </p>
      </Translated>
      <Beyond title={tr('Not in the book: who chooses the accumulator', 'Không có trong sách: ai chọn bộ tích lũy')}>
        <Translated
          vi={
            <p>
              PyTorch tích lũy half, bfloat16 và FP8 bằng float trên CUDA, MPS và CPU, và float bằng double trên CPU. Softmax
              mặc định của MLX tích lũy theo kiểu của đầu vào trừ khi <code>precise=True</code>; theo cách chúng tôi đọc mã
              nguồn của nó, <code>mx.sum</code> trên half tích lũy bằng half. llama.cpp nêu rõ bộ tích lũy bắt buộc cho từng
              phép toán bằng <code>ggml_prec_set_acc</code> ({FACTS} D1 đến D3).
            </p>
          }
        >
          <p>
            PyTorch accumulates half, bfloat16 and FP8 in float on CUDA, MPS and CPU, and float in double on the CPU. MLX's
            default softmax accumulates in the input type unless <code>precise=True</code>; by our reading of its source,{' '}
            <code>mx.sum</code> on half accumulates in half. llama.cpp states a required accumulator per operation with{' '}
            <code>ggml_prec_set_acc</code> ({FACTS} D1 to D3).
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Một tổng, bốn cách">One sum, four ways</Translated></h2>
      <Translated
        vi={
          <p>
            Lấy một mẫu số softmax: một hàng {TEASER_N.toLocaleString('en-US')} logits, rút từ phân phối chuẩn tắc, và tổng
            của exp(z − max z) trên hàng đó. Mọi số hạng nằm trong (0, 1]. Cộng nó theo bốn cách: một vòng lặp và một cây cộng
            theo cặp, mỗi cách với một bộ tích lũy float (FP32) và một bộ tích lũy half (FP16), rồi so sánh từng kết quả với một
            giá trị tham chiếu chính xác.
          </p>
        }
      >
        <p>
          Take one softmax denominator: a row of {TEASER_N.toLocaleString('en-US')} logits, drawn from a standard normal
          distribution, and the sum of exp(z − max z) over the row. Every term lies in (0, 1]. Sum it four ways: a loop and a
          pairwise tree, each with a float (FP32) and with a half (FP16) accumulator, and compare each with an accurate
          reference.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Hàng dữ liệu và mọi con số trong phần kết quả đến từ mô hình của chúng tôi về bốn phép cộng (làm tròn về số chẵn
              gần nhất sau mỗi phép cộng; không có gì riêng của một GPU cụ thể). Lab A.1 đo sai số phép cộng trên GPU của bạn.
            </p>
          }
        >
          <p>
            The row and every number in the reveal come from our model of the four sums (round to nearest even after every
            addition; nothing about a particular GPU). Lab A.1 measures summation error on yours.
          </p>
        </Translated>
      </Beyond>
      <Predict
        id="appA/start/p-teaser"
        prompt={
          <Translated
            vi={
              <>
                Bạn đoán <b>vòng lặp</b> độ chính xác half sẽ lệch bao xa, tính theo phần trăm của tổng thật? Hãy chốt một con số
                trước khi bốn kết quả hiện ra.
              </>
            }
          >
            How far off do you expect the half-precision <b>loop</b> to be, as a percentage of the true sum? Commit to a
            number before the four results appear.
          </Translated>
        }
        unit="%"
      >
        {(v) => <TeaserReveal v={v} />}
      </Predict>

      <h2><Translated vi="Phụ lục này nói về điều gì">What this appendix is about</Translated></h2>
      <Translated
        vi={
          <p>
            Phụ lục của sách bắt đầu từ lý do dấu phẩy động đáng để bận tâm: với cùng số bit, nó phủ một dải giá trị rộng hơn
            nhiều so với một định dạng số nguyên, và nó biểu diễn các giá trị nhỏ một cách tinh tế. Sau đó nó tách ba ý tưởng
            dễ bị nhập nhằng. <b>Độ chính xác biểu diễn (precision)</b> là thuộc tính của một định dạng: giá trị gần nhất của nó
            có thể gần con số bạn muốn lưu đến mức nào. <b>Độ chính xác tính toán (accuracy)</b> là thuộc tính của một phép
            toán: một phép cộng hay phép chia thêm vào bao nhiêu sai số. <b>Tính ổn định (stability)</b> là thuộc tính của một
            thuật toán: nó có tìm ra nghiệm cho mọi đầu vào có nghiệm hay không (mục <a href={href('appA', 'A.7')}>A.7</a>).
          </p>
        }
      >
        <p>
          The book's appendix starts from why floating point is worth the trouble: for the same number of bits it covers a
          far wider range of values than an integer format, and it represents small values finely. It then separates
          three ideas that are easy to blur. <b>Precision</b> is a property of a format: how close its nearest value can be
          to the number you want to store. <b>Accuracy</b> is a property of an operation: how much error one addition or
          division adds. <b>Stability</b> is a property of an algorithm: whether it finds a solution for every input that
          has one (section <a href={href('appA', 'A.7')}>A.7</a>).
        </p>
      </Translated>
      <Source pages="583, 593–594" />
      <Translated
        vi={
          <p>
            Sách cũng lưu ý vì sao điều này quan trọng hơn trước: nhu cầu từ việc huấn luyện và suy luận LLM đã đẩy GPU tăng
            thông lượng dấu phẩy động một cách mạnh mẽ. Mục A.5 bổ sung rằng các GPU gần đây tính toán bằng dấu phẩy động 16,
            8 và 4 bit.
          </p>
        }
      >
        <p>
          The book also notes why this matters more now than it did: demand from LLM training and inference has pushed GPUs
          to raise their floating-point throughput dramatically. Section A.5 adds that recent GPUs compute in 16-, 8- and
          4-bit floating point.
        </p>
      </Translated>
      <Today checked={`2026-11-20 (${FACTS}, A1, A4)`}>
        <Translated
          vi={
            <p>
              Sách trích OCP MX v1.0; v1.1 (tháng 10 năm 2026) đổi tên E8M0 thành UE8M0, và NVIDIA, AMD và Metal đều hỗ trợ nó
              (§A.5). IEEE 754-2019 vẫn là chuẩn đang có hiệu lực.
            </p>
          }
        >
          <p>
            The book cites OCP MX v1.0; v1.1 (October 2026) renamed E8M0 to UE8M0, and NVIDIA, AMD and Metal all support
            it (§A.5). IEEE 754-2019 is still the active standard.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Bản đồ">The map</Translated></h2>
      <ul>
        <Translated
          vi={
            <li>
              <a href={href('appA', 'A.1')}>A.1</a> đến <a href={href('appA', 'A.3')}>A.3</a>: một định dạng lưu một giá trị
              thế nào, nó chứa được những giá trị nào, và các giá trị đặc biệt của nó (số không, denormal, vô cực, NaN).
            </li>
          }
        >
          <li>
            <a href={href('appA', 'A.1')}>A.1</a> to <a href={href('appA', 'A.3')}>A.3</a>: how a format stores a value,
            which values it can hold, and its special values (zero, denormals, infinity, NaN).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <a href={href('appA', 'A.4')}>A.4</a>: sai số của một phép toán, tính bằng đơn vị ở vị trí cuối (ULP).
            </li>
          }
        >
          <li>
            <a href={href('appA', 'A.4')}>A.4</a>: the error of one operation, in units in the last place.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <a href={href('appA', 'A.5')}>A.5</a>: các định dạng hẹp và việc co giãn theo khối, điểm xuất phát cho các kernel
              lượng tử hóa của module X2.
            </li>
          }
        >
          <li>
            <a href={href('appA', 'A.5')}>A.5</a>: narrow formats and block scaling, the starting point for the quantised
            kernels of module X2.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <a href={href('appA', 'A.6')}>A.6</a>: các tổng dài, phần cốt lõi của phụ lục này đối với các kernel suy luận. Nó
              nối lại với các phép rút gọn (reduction) của chương 10 và softmax của mốc dự án M4.
            </li>
          }
        >
          <li>
            <a href={href('appA', 'A.6')}>A.6</a>: long sums, the core of this appendix for inference kernels. It ties back to
            chapter 10's reductions and milestone M4's softmax.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <a href={href('appA', 'A.7')}>A.7</a>: tính ổn định số học, ở dạng cô đọng.
            </li>
          }
        >
          <li>
            <a href={href('appA', 'A.7')}>A.7</a>: numerical stability, condensed.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <a href={href('appA', 'labs')}>Lab A.1</a> đo sai số phép cộng, theo thứ tự, độ rộng bộ tích lũy và phép bù, trên
              GPU của bạn.
            </li>
          }
        >
          <li>
            <a href={href('appA', 'labs')}>Lab A.1</a> measures summation error, by order, accumulator width and
            compensation, on your GPU.
          </li>
        </Translated>
      </ul>

      <h2><Translated vi="Trước khi bắt đầu">Before you start</Translated></h2>
      <Translated
        vi={
          <p>
            Bốn câu hỏi về những gì phụ lục này giả định từ các chương 5, 10 và 20. Mỗi câu sẽ quay lại trong phần{' '}
            <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Four questions on what this appendix assumes from chapters 5, 10 and 20. Each one comes back in{' '}
          <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={questionsSA0['appA/start/q-tolerance']} label={tr('From chapter 10', 'Từ chương 10')} />
      <Ask q={questionsSA0['appA/start/q-convergent']} label={tr('From chapter 10', 'Từ chương 10')} />
      <Ask q={questionsSA0['appA/start/q-bytes']} label={tr('From chapters 5 and 20', 'Từ chương 5 và 20')} />
      <Ask q={questionsSA0['appA/start/q-atomic']} label={tr('From chapter 10', 'Từ chương 10')} />
    </>
  );
}
