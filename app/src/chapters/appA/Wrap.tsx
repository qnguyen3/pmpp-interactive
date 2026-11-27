import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// The numbers every question below states are pinned by Wrap.test.ts, computed with the
// same models (sim/minifloat.ts, sim/fpsum.ts) the sections use.
const FACTS = 'notes/today/appA-facts.md';

export const questionsWrap: Record<string, Question> = {
  'appA/wrap/q-rms-half': {
    kind: 'choice',
    id: 'appA/wrap/q-rms-half',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Một lớp chuẩn hóa (norm) cộng bình phương của 4,096 activation để lấy trung bình. Mỗi activation đúng bằng 4, nên mỗi bình
              phương là 16 và tổng thật là 65,536. Tổng được giữ ở FP16. Một vòng lặp thường và một cây chia đôi (mỗi tầng cộng từng
              cặp) trả về gì?
            </>
          }
        >
          A norm layer sums the squares of 4,096 activations to get their mean. Every activation is exactly 4, so each
          square is 16 and the true sum is 65,536. The sum is kept in FP16. What do a plain loop and a halving tree (each
          level adds pairs) return?
        </Translated>
      </>
    ),
    options: [
      {
        label: <Translated vi="Vòng lặp trả về 32,768; cây trả về vô cực.">The loop returns 32,768; the tree returns infinity.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Cả hai trả về 65,536: mọi tổng riêng phần đều là số nguyên.">Both return 65,536: every partial sum is a whole number.</Translated>,
        why: (
          <Translated vi="Không phải mọi số nguyên trên 2,048 đều là giá trị FP16, và 65,536 lớn hơn giá trị lớn nhất, 65,504.">
            Whole numbers above 2,048 are not all FP16 values, and 65,536 is above the largest one, 65,504.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Cả hai trả về vô cực: tổng thật quá lớn đối với FP16.">Both return infinity: the true sum is too large for FP16.</Translated>,
        why: (
          <Translated vi="Vòng lặp không bao giờ tới đó. Tại 32,768 khoảng cách giữa các giá trị FP16 là 32, và cộng 16 đúng bằng nửa khoảng cách.">
            The loop never gets there. At 32,768 the gap between FP16 values is 32, and adding 16 is exactly half a gap.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Vòng lặp trả về 65,504 và cây trả về vô cực.">The loop returns 65,504 and the tree returns infinity.</Translated>,
        why: (
          <Translated vi="Vòng lặp ngừng tăng từ sớm hơn nhiều. Ở đâu, và vì sao lại đúng ở đó?">
            The loop stops growing well before that. Where, and why exactly there?
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đến 32,768, mọi bội của 16 đều là giá trị FP16, nên vòng lặp chính xác cho đến đó. Trên 32,768 khoảng cách là 32, và
            32,768 + 16 nằm đúng giữa hai giá trị kề nhau; làm tròn về chẵn gần nhất chọn 32,768, và mọi phép cộng sau đó cũng vậy.
            Các tổng riêng phần của cây gấp đôi ở mỗi tầng, đều chính xác, cho đến phép cộng cuối, 32,768 + 32,768, bị tràn. Trung
            bình của vòng lặp là 8 trong khi lẽ ra là 16, nên căn bậc hai trung bình bình phương ra khoảng 2.83 thay vì 4 và mọi đầu
            ra lớn gấp khoảng 1.41 lần. Vô cực của cây làm mọi đầu ra bằng 0. Tích lũy bằng float sửa được cả hai, và đó là điều
            PyTorch làm với đầu vào half trên mọi thiết bị.
          </p>
        }
      >
        <p>
          Up to 32,768 every multiple of 16 is an FP16 value, so the loop is exact until then. Above 32,768 the spacing is
          32, and 32,768 + 16 lies exactly between two neighbours; round to nearest even picks 32,768, and every later
          addition does the same. The tree's partial sums double at each level, all exact, until the last addition,
          32,768 + 32,768, overflows. The loop's mean is 8 where it should be 16, so the root mean square comes out about
          2.83 instead of 4 and every output is about 1.41 times too large. The tree's infinity makes every output 0.
          Accumulating in float fixes both, which is what PyTorch does for half inputs on every device.
        </p>
      </Translated>
    ),
    source: `ours, from §A.2 (spacing, pp. 586–590), §A.3 (overflow to infinity, p. 590) and §A.6 (p. 593); ${FACTS} D1`,
    vi: { source: `của chúng tôi, từ §A.2 (khoảng cách, pp. 586–590), §A.3 (tràn thành vô cực, p. 590) và §A.6 (p. 593); ${FACTS} D1` },
  },
  'appA/wrap/q-bf16-argmax': {
    kind: 'numeric',
    id: 'appA/wrap/q-bf16-argmax',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Giải mã greedy lấy argmax của một hàng logits. Một kernel ép kiểu logits sang BF16 trước. Hai ứng viên có logits FP32 là
              10.01 và 10.03. Khoảng cách giữa các giá trị BF16 kề nhau gần 10 là bao nhiêu?
            </>
          }
        >
          Greedy decoding takes the argmax of a row of logits. A kernel casts the logits to BF16 first. Two candidates have
          FP32 logits of 10.01 and 10.03. What is the gap between neighbouring BF16 values near 10?
        </Translated>
      </>
    ),
    answer: 0.0625,
    tol: 0.001,
    explain: (
      <Translated
        vi={
          <p>
            BF16 giữ 7 bit phần định trị được lưu, nên giữa 8 và 16 các giá trị của nó cách nhau 2<sup>3</sup> × 2<sup>−7</sup> =
            0.0625. Cả hai logits đều làm tròn thành 10.0, và argmax giờ phải phân xử một trường hợp hòa mà FP32 không có; token nào
            thắng tùy vào quy tắc phân xử hòa của nó. Token greedy có thể đổi. Hãy lấy argmax trên logits FP32, hoặc trên kiểu mà
            GEMM cuối cùng đã tích lũy, trước bất kỳ phép ép kiểu nào.
          </p>
        }
      >
        <p>
          BF16 keeps 7 stored mantissa bits, so between 8 and 16 its values are 2<sup>3</sup> × 2<sup>−7</sup> = 0.0625
          apart. Both logits round to 10.0, and the argmax now has to break a tie that FP32 did not have; which token
          wins depends on its tie rule. The greedy token can change. Take the argmax on the FP32 logits, or on whatever the last GEMM
          accumulated in, before any cast.
        </p>
      </Translated>
    ),
    source: 'ours, from §A.2 (precision, pp. 586–590); BF16 field widths: notes/today/appA-facts.md A2',
    vi: { source: 'của chúng tôi, từ §A.2 (độ chính xác, pp. 586–590); độ rộng các trường BF16: notes/today/appA-facts.md A2' },
  },
  'appA/wrap/q-logprob-ftz': {
    kind: 'choice',
    id: 'appA/wrap/q-logprob-ftz',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Một bộ lấy mẫu cần log-xác suất. Thiết bị của bạn đẩy các denormal FP32 về không (phép dò của lab A.1 cho bạn biết thiết
              bị của bạn có làm vậy không). Logit của một token thấp hơn giá trị lớn nhất của hàng 90. Cách tính log-xác suất nào của
              nó cho ra đáp án hữu hạn ở đó?
            </>
          }
        >
          A sampler needs log-probabilities. Your device flushes FP32 denormals to zero (lab A.1's probe tells you whether
          yours does). A token's logit is 90 below the row's maximum. Which way of computing its log-probability gives a
          finite answer there?
        </Translated>
      </>
    ),
    options: [
      {
        label: (
          <Translated vi="Lấy log của xác suất softmax đúng như viết, log(exp(z − max) / Σ), bằng float.">
            Take the log of the softmax probability as written, log(exp(z − max) / Σ), in float.
          </Translated>
        ),
        why: (
          <Translated vi="exp(−90) khoảng 8.2 × 10⁻⁴⁰, nhỏ hơn số float normal nhỏ nhất. Bị đẩy về không, nó bằng 0, và log 0 là âm vô cực.">
            exp(−90) is about 8.2 × 10⁻⁴⁰, below the smallest normal float. Flushed, it is 0, and log 0 is minus infinity.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Tính (z − max) − log Σ exp(zⱼ − max), không tạo ra xác suất.">
            Compute (z − max) − log Σ exp(zⱼ − max), without forming the probability.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Cách nào cũng được: exp(−90) là một giá trị FP32 bình thường.">Either: exp(−90) is an ordinary FP32 value.</Translated>,
        why: <Translated vi="Chỉ dưới dạng denormal, thứ mà thiết bị này không giữ.">Only as a denormal, which this device does not keep.</Translated>,
      },
      {
        label: <Translated vi="Không cách nào: −90 nằm ngoài dải giá trị của FP32.">Neither: −90 is outside FP32's range.</Translated>,
        why: <Translated vi="Bản thân −90 là một số float bình thường. Chỉ có hàm mũ của nó là rất nhỏ.">−90 itself is an ordinary float. Only its exponential is tiny.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            exp(−90) khoảng 8.2 × 10<sup>−40</sup>. Giá trị normal nhỏ nhất của FP32 khoảng 1.18 × 10<sup>−38</sup>, nên xác suất này
            chỉ tồn tại dưới dạng denormal. Trên thiết bị giữ denormal, cả hai cách đều chạy; trên thiết bị đẩy chúng về không, cách
            thứ nhất cho âm vô cực, và một phép cộng hay một phép kiểm tra NaN sau đó có thể hành xử theo những cách khó lần ra. Cách
            thứ hai không bao giờ tính con số rất nhỏ đó: các số hạng của nó khoảng −90 và log của một tổng ít nhất bằng 1.
          </p>
        }
      >
        <p>
          exp(−90) is about 8.2 × 10<sup>−40</sup>. FP32's smallest normal value is about 1.18 × 10<sup>−38</sup>, so the
          probability exists only as a denormal. On a device that keeps denormals both forms work; on one that flushes
          them, the first gives minus infinity, and a later sum or a NaN test may then behave in ways that are hard to
          trace. The second form never computes the tiny number: its terms are about −90 and the log of a sum that is at
          least 1.
        </p>
      </Translated>
    ),
    source: `ours, from §A.2 (denormals, pp. 588–589); flushing: ${FACTS} A6, C2`,
    vi: { source: `của chúng tôi, từ §A.2 (denormal, pp. 588–589); đẩy về không: ${FACTS} A6, C2` },
  },
  'appA/wrap/q-e4m3-small': {
    kind: 'numeric',
    id: 'appA/wrap/q-e4m3-small',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Một KV cache lưu activation ở E4M3 không có hệ số tỉ lệ. Một activation là 0.0006. Giá trị nào được lưu?
            </>
          }
        >
          A KV cache stores activations in E4M3 with no scale factor. One activation is 0.0006. What value is stored?
        </Translated>
      </>
    ),
    answer: 0,
    explain: (
      <Translated
        vi={
          <p>
            Giá trị dương nhỏ nhất của E4M3 là denormal 2<sup>−9</sup>, khoảng 0.00195. 0.0006 nhỏ hơn một nửa giá trị đó, nên nó làm
            tròn thành 0. Nhân trước với một lũy thừa của hai, chẳng hạn 2<sup>10</sup>, đưa nó lên khoảng 0.61, nơi E4M3 giữ nó với
            sai lệch vài phần trăm; hệ số tỉ lệ được lưu bên cạnh khối và được chia ra khi đọc cache. Đó là lý do các định dạng hẹp đi
            kèm hệ số tỉ lệ, cho activation cũng như cho trọng số.
          </p>
        }
      >
        <p>
          E4M3's smallest positive value is the denormal 2<sup>−9</sup>, about 0.00195. 0.0006 is less than half of it,
          so it rounds to 0. Multiplying by a power of two first, say 2<sup>10</sup>, moves it to about 0.61, where E4M3
          keeps it to within a few percent; the scale is stored beside the block and divided out when the cache is read.
          That is the reason narrow formats come with scales, for activations as much as for weights.
        </p>
      </Translated>
    ),
    source: `ours, from §A.2 (gap near zero, pp. 588–589) and §A.5 (pp. 592–593); E4M3: ${FACTS} A3`,
    vi: { source: `của chúng tôi, từ §A.2 (khoảng trống gần không, pp. 588–589) và §A.5 (pp. 592–593); E4M3: ${FACTS} A3` },
  },
  'appA/wrap/q-batch-order': {
    kind: 'choice',
    id: 'appA/wrap/q-batch-order',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Server của bạn trả về logits hơi khác nhau cho cùng một prompt khi nó chạy riêng và khi nó chạy trong một batch 8. Mỗi
              cấu hình cho cùng logits mỗi lần chạy. Thư viện GEMM chọn một cách chia chiều k khác nhau cho mỗi kích thước batch. Lời
              giải thích khả dĩ nhất là gì?
            </>
          }
        >
          Your server returns slightly different logits for the same prompt when it runs alone and when it runs in a batch
          of 8. Each configuration gives the same logits every time it runs. The GEMM library chooses a different split of
          the k dimension for each batch size. What is the most likely explanation?
        </Translated>
      </>
    ),
    options: [
      {
        label: (
          <Translated vi="Hai cách chia cộng các tích riêng phần của mỗi tích vô hướng theo thứ tự khác nhau, và phép cộng float không có tính kết hợp.">
            The two splits add each dot product's partial products in different orders, and float addition is not associative.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Một race condition (tranh chấp dữ liệu) giữa các khối cộng các tổng riêng phần của chiều k.">
            A race condition between the blocks that add the partial sums of the k dimension.
          </Translated>
        ),
        why: (
          <Translated vi="Một race condition sẽ khiến một cấu hình khác nhau giữa các lần chạy. Ở đây mỗi cấu hình đều lặp lại được.">
            A race would make one configuration differ from run to run. Here each one is repeatable.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="GEMM theo batch đẩy denormal về không còn GEMM đơn lẻ thì không.">
            The batched GEMM flushes denormals and the single one does not.
          </Translated>
        ),
        why: (
          <Translated vi="Việc đẩy về không là thuộc tính của thiết bị và thiết lập trình biên dịch, không phải của kích thước batch.">
            Flushing is a property of the device and the compiler settings, not of the batch size.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một trong hai kết quả là sai: một GEMM đúng cho cùng các bit ở mọi kích thước batch.">
            One of the two results is wrong: a correct GEMM gives the same bits at every batch size.
          </Translated>
        ),
        why: (
          <Translated vi="Cả hai thứ tự đều là cách làm tròn hợp lệ của cùng một tổng. Không cái nào là đáp án chính xác.">
            Both orders are legitimate roundings of the same sum. Neither is the exact answer.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Chia k thành nhiều mảnh hơn biến một tổng dài thành vài tổng ngắn hơn được cộng ở cuối: một cây khác. Ý chính của mục A.6
            là mọi cây như vậy đều là kết quả hợp lệ và chúng khác nhau ở các bit thấp. Nếu bạn cần cùng logits bất kể batch, thứ tự
            rút gọn của mỗi đầu ra không được phụ thuộc vào kích thước batch, điều thường tốn chút tốc độ.
          </p>
        }
      >
        <p>
          Splitting k into more pieces turns one long sum into several shorter ones added at the end: a different tree.
          Section A.6's point is that every such tree is a valid result and they differ in the low bits. If you need the
          same logits whatever the batch, the reduction order of each output must not depend on the batch size, which
          usually costs some speed.
        </p>
      </Translated>
    ),
    source: 'book §A.6, p. 593 (order changes the sum); the GEMM setting is ours',
    vi: { source: 'sách §A.6, p. 593 (thứ tự làm thay đổi tổng); bối cảnh GEMM là của chúng tôi' },
  },
};

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports);
  const lab = backends.filter((b) => labStatus(reports[b], 'appA/lab1_sumerror', 'work') === 'pass');
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('appA/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  return (
    <>
      <h2 style={{ marginTop: 0 }}><Translated vi="Những gì cần mang theo">What to carry forward</Translated></h2>
      <ul>
        <Translated
          vi={
            <li>
              Một định dạng gồm dấu, số mũ và phần định trị. Bit số mũ mua dải giá trị, bit phần định trị mua độ chính xác, và
              khoảng cách giữa các giá trị kề nhau gấp đôi ở mỗi lũy thừa của hai.
            </li>
          }
        >
          <li>
            A format is a sign, an exponent and a mantissa. Exponent bits buy range, mantissa bits buy precision, and the
            gap between neighbouring values doubles at every power of two.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Gần không, một định dạng hoặc có một khoảng trống hoặc lấp nó bằng denormal. Backend nào cũng để denormal bị mất
              ở đâu đó: một cờ trình biên dịch, một chế độ fast-math, hoặc chính thiết bị. Phép dò của lab A.1 cho bạn biết
              thiết bị của bạn làm gì.
            </li>
          }
        >
          <li>
            Near zero a format either has a gap or fills it with denormals. Every backend lets denormals be lost somewhere:
            a compiler flag, a fast-math mode, or the device itself. Lab A.1's probe tells you what yours does.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Vô cực và NaN là giá trị, không phải lỗi. Không gì trên GPU dừng lại khi chúng xuất hiện, nên một softmax trừ giá
              trị lớn nhất trước và một bài kiểm thử kiểm tra đầu ra của nó.
            </li>
          }
        >
          <li>
            Infinity and NaN are values, not errors. Nothing on a GPU stops when one appears, so a softmax subtracts the
            maximum first and a test checks its outputs.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Độ chính xác biểu diễn thuộc về định dạng, độ chính xác tính toán thuộc về phép toán. Số học cơ bản được làm tròn
              đúng ở nơi nhà cung cấp nói vậy; các hàm fast-math và chế độ làm tròn của một số backend thì không.
            </li>
          }
        >
          <li>
            Precision belongs to the format, accuracy to the operation. Basic arithmetic is correctly rounded where the
            vendor says so; fast-math functions and some backends' rounding modes are not.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Định dạng hẹp đổi độ chính xác lấy số byte, và một bước decode bị giới hạn bởi bộ nhớ biến ít byte hơn cho mỗi
              trọng số thành tốc độ. Một hệ số tỉ lệ dùng chung cho mỗi khối lấy lại dải giá trị.
            </li>
          }
        >
          <li>
            Narrow formats trade precision for bytes, and a memory-bound decode step turns fewer bytes per weight into
            speed. A shared scale per block recovers the range.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Một tổng dài phụ thuộc vào thứ tự và bộ tích lũy của nó. Một cây, một thứ tự đã sắp xếp, phép bù của Kahan và một
              bộ tích lũy rộng hơn đều giúp ích; một bộ tích lũy hẹp thì bị đứng lại. Hãy so kết quả GPU với một giá trị tham
              chiếu chính xác và một dung sai, không bao giờ với một thứ tự khác.
            </li>
          }
        >
          <li>
            A long sum depends on its order and on its accumulator. A tree, a sorted order, Kahan's compensation and a
            wider accumulator each help; a narrow accumulator stalls. Compare GPU results with an accurate reference and a
            tolerance, never with another order.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Tính ổn định là thuộc tính của thuật toán. Chọn phần tử chốt giữ phép khử ổn định, và khi chạy song song nó tốn
              một phép rút gọn max cho mỗi cột.
            </li>
          }
        >
          <li>
            Stability is a property of the algorithm. Pivoting keeps elimination stable, and in parallel it costs a max
            reduction per column.
          </li>
        </Translated>
      </ul>
      <Source section="A.8" pages="599" />
      <Translated
        vi={
          <p>
            Phần tóm tắt của chính sách thêm một dòng đáng giữ: các thiết bị CUDA đời đầu không có denormal và các thiết bị sau
            này thì có, và các phép toán nhanh của "special function units" kém chính xác hơn các phép toán thông thường.
          </p>
        }
      >
        <p>
          The book's own summary adds one line worth keeping: early CUDA devices lacked denormals and later ones have them,
          and the fast operations of the "special function units" are less accurate than the ordinary ones.
        </p>
      </Translated>
      <Today checked={`2026-11-20 (${FACTS} A6, A14)`}>
        <Translated
          vi={
            <p>
              NVIDIA giữ denormal FP32 theo mặc định và đẩy chúng về không khi có <code>--ftz=true</code> hoặc{' '}
              <code>--use_fast_math</code>. Tài liệu hiện hành của hãng không dùng thuật ngữ "special function units"; nó nói các
              intrinsic nhanh hơn vì chúng "map to fewer native instructions" (ánh xạ sang ít lệnh gốc hơn).
            </p>
          }
        >
          <p>
            NVIDIA keeps FP32 denormals by default and flushes them under <code>--ftz=true</code> or{' '}
            <code>--use_fast_math</code>. Its current documentation does not use the term "special function units"; it
            says intrinsics are faster because they "map to fewer native instructions".
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Một bài kiểm tra tổng hợp">A mixed check</Translated></h2>
      <p>
        <Translated vi="Các câu này dựa trên toàn bộ phụ lục và trên những tình huống mà các mục chưa trình bày.">
          These draw on the whole appendix and on situations the sections did not show.
        </Translated>
        {asked
          ? tr(
              ` So far in this appendix you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.`,
              ` Đến giờ trong phụ lục này bạn đã trả lời đúng ${right} trên ${asked} câu hỏi, tính theo câu trả lời gần nhất của bạn cho mỗi câu.`,
            )
          : ''}
      </p>
      <Ask q={questionsWrap['appA/wrap/q-rms-half']} />
      <Ask q={questionsWrap['appA/wrap/q-bf16-argmax']} />
      <Ask q={questionsWrap['appA/wrap/q-logprob-ftz']} />
      <Ask q={questionsWrap['appA/wrap/q-e4m3-small']} />
      <Ask q={questionsWrap['appA/wrap/q-batch-order']} />

      <h2><Translated vi="Các bài tập của sách">The book's exercises</Translated></h2>
      <Translated
        vi={
          <p>
            Phụ lục kết thúc bằng năm bài tập (pp. 599–600). Mỗi bài được giải ở đâu đó trong khóa học này; cách diễn đạt của
            chính sách cũng đáng để làm.
          </p>
        }
      >
        <p>
          The appendix ends with five exercises (pp. 599–600). Each is worked somewhere in this course; the book's own
          wording is worth doing as well.
        </p>
      </Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-table="exercises">
          <thead>
            <tr>
              <th><Translated vi="Bài tập">Exercise</Translated></th>
              <th><Translated vi="Nội dung">About</Translated></th>
              <th><Translated vi="Được giải ở đâu trong khóa học">Where it is worked here</Translated></th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1</td>
              <td><Translated vi="Một định dạng 6 bit với bit phần định trị thứ ba">A 6-bit format with a third mantissa bit</Translated></td>
              <td>
                <Translated vi={<>trình khám phá trục số và câu hỏi của nó ở <a href={href('appA', 'A.2')}>A.2</a></>}>
                  the number-line explorer and its question in <a href={href('appA', 'A.2')}>A.2</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td><Translated vi="Một định dạng 6 bit với bit số mũ thứ ba">A 6-bit format with a third exponent bit</Translated></td>
              <td>
                <Translated vi={<>cùng trình khám phá đó và câu hỏi thứ hai của nó ở <a href={href('appA', 'A.2')}>A.2</a></>}>
                  the same explorer and its second question in <a href={href('appA', 'A.2')}>A.2</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td><Translated vi="Sai số lớn nhất của một bộ cộng chỉ cắt bỏ">The largest error of an adder that only truncates</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      trình chạy từng bước căn chỉnh đặt ở "toward zero", cùng câu hỏi của nó, ở <a href={href('appA', 'A.4')}>A.4</a>;
                      trên CUDA, bộ cộng cắt bỏ trong phần "go further" của <a href={href('appA', 'labs')}>lab A.1</a>
                    </>
                  }
                >
                  the alignment stepper set to "toward zero", and its question, in <a href={href('appA', 'A.4')}>A.4</a>;
                  on CUDA, the truncating adder in <a href={href('appA', 'labs')}>lab A.1</a>'s "go further"
                </Translated>
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td><Translated vi="Một mảng đã sắp xếp được rút gọn bằng cây hội tụ">A sorted array reduced by the convergent tree</Translated></td>
              <td>
                <Translated vi={<>trình chạy từng bước "thread to data, with rounding" và câu hỏi của nó ở <a href={href('appA', 'A.6')}>A.6</a></>}>
                  the "thread to data, with rounding" stepper and its question in <a href={href('appA', 'A.6')}>A.6</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>5</td>
              <td><Translated vi="Sai số ULP của một bộ tính sin() dừng sớm">The ULP error of a sin() unit that stops early</Translated></td>
              <td>
                <Translated vi={<>câu hỏi ở <a href={href('appA', 'A.4')}>A.4</a></>}>
                  the question in <a href={href('appA', 'A.4')}>A.4</a>
                </Translated>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <Translated
        vi={
          <p>
            Sách chỉ đến bài khảo sát của Goldberg (tài liệu tham khảo [1]) để tìm hiểu thêm về dấu phẩy động, và đến ghi chú năm
            1965 của Kahan ([5]) về phép cộng có bù. Tài liệu tham khảo [3] của sách giờ là đặc tả OCP MicroXcaling (MX), phiên
            bản 1.1, và tài liệu tham khảo [4], trang dấu phẩy động của NVIDIA, giờ trỏ đến bản white paper năm 2011: văn bản được
            duy trì là mục 5.6 của CUDA Programming Guide.
          </p>
        }
      >
        <p>
          The book points to Goldberg's survey (its reference [1]) for more on floating point, and to Kahan's 1965 note
          ([5]) for compensated summation. Its reference [3] is now the OCP MicroXcaling (MX) specification, version 1.1,
          and its reference [4], NVIDIA's floating-point page, now serves the 2011 white paper: the maintained text is
          section 5.6 of the CUDA Programming Guide.
        </p>
      </Translated>
      <Today checked={`2026-11-20 (${FACTS} A4, A12)`}>
        <Translated
          vi={
            <p>
              MX v1.1 (ngày 9 tháng 10 năm 2026) đổi tên đặc tả và kiểu hệ số tỉ lệ E8M0 của nó thành UE8M0. URL NVIDIA của sách
              dẫn đến bản white paper năm 2011, mà mục 5.6 của hướng dẫn liệt kê trong các tài liệu tham khảo.
            </p>
          }
        >
          <p>
            MX v1.1 (October 9, 2026) renamed the specification and its scale type E8M0 to UE8M0. The book's NVIDIA URL
            resolves to the 2011 white paper, which the guide's section 5.6 lists among its references.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Những chỗ PDF có vẻ sai">Where the PDF looks wrong</Translated></h2>
      <Translated
        vi={
          <p>
            Bản sao mà khóa học này dựa vào có một số lỗi sơ suất rõ ràng trong phụ lục này. Chúng được liệt kê để bạn không bị
            bối rối khi đọc sách song song. Phần lớn được bàn đến ở nơi chúng xuất hiện.
          </p>
        }
      >
        <p>
          The copy this course was built from has a number of apparent slips in this appendix. They are listed so that you
          are not thrown by them when you read the book alongside. Most are discussed where they come up.
        </p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>
              p. 589 nói denormal được bật "on or off" bằng "ftz = true (on) or ftz = false (off)". Cờ này nghĩa là flush to
              zero, nên <code>--ftz=true</code> tắt denormal. Xem <a href={href('appA', 'A.2')}>A.2</a>.
            </li>
          }
        >
          <li>
            p. 589 says denormals are switched "on or off" with "ftz = true (on) or ftz = false (off)". The flag means flush
            to zero, so <code>--ftz=true</code> turns denormals off. See <a href={href('appA', 'A.2')}>A.2</a>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 588 đặt số không bị thiếu vào "the no-zero column of Fig. A.5"; các cột nằm ở Fig. A.4. p. 589 chỉ đến Fig. A.6
              cho denormal, trong khi ý là Fig. A.7, và "previous interval" của nó là khoảng lớn hơn kế tiếp. Fig. A.7 vẽ một
              ngôi sao thứ 13 gần 4, thứ mà định dạng không chứa được, và Fig. A.5 không vẽ đúng tỉ lệ giữa 0 và 1. Xem{' '}
              <a href={href('appA', 'A.2')}>A.2</a>.
            </li>
          }
        >
          <li>
            p. 588 places the missing zero in "the no-zero column of Fig. A.5"; the columns are in Fig. A.4. p. 589 points to
            Fig. A.6 for denormals, where Fig. A.7 is meant, and its "previous interval" is the next larger one. Fig. A.7
            draws a 13th star near 4, which the format cannot hold, and Fig. A.5 is not drawn to scale between 0 and 1. See{' '}
            <a href={href('appA', 'A.2')}>A.2</a>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 590 nói thêm ba bit số mũ cho số khoảng nhiều gấp "2<sup>(2<sup>3</sup>)</sup> = 256 times"; thực ra là gấp
              2<sup>3</sup> = 8 lần. Xem <a href={href('appA', 'A.3')}>A.3</a>.
            </li>
          }
        >
          <li>
            p. 590 says three more exponent bits give "2<sup>(2<sup>3</sup>)</sup> = 256 times" as many intervals; they give
            2<sup>3</sup> = 8 times as many. See <a href={href('appA', 'A.3')}>A.3</a>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 591 cho độ chính xác đơn "2<sup>22</sup> − 1 possible NaN bit patterns". Đó là số NaN signaling của một dấu; tổng
              cộng có 2<sup>24</sup> − 2. Xem <a href={href('appA', 'A.3')}>A.3</a>.
            </li>
          }
        >
          <li>
            p. 591 gives single precision "2<sup>22</sup> − 1 possible NaN bit patterns". That is the number of signalling
            NaNs of one sign; there are 2<sup>24</sup> − 2 in all. See <a href={href('appA', 'A.3')}>A.3</a>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 585 lập luận rằng mẫu dành riêng 111 không thể là "either even number or odd number"; có vẻ ý là dương hay âm.
            </li>
          }
        >
          <li>
            p. 585 argues the reserved pattern 111 cannot be "either even number or odd number"; positive or negative seems
            to be meant.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 593 gọi 989 TFLOPS của H100 là "single-precision throughput using tensor cores". Con số 989 của NVIDIA là hàng
              TF32, có sparsity; con số FP8 3,958 bên cạnh cũng là số có sparsity. Xem <a href={href('appA', 'A.5')}>A.5</a>.
            </li>
          }
        >
          <li>
            p. 593 calls H100's 989 TFLOPS "single-precision throughput using tensor cores". NVIDIA's 989 is the TF32 row,
            with sparsity; the 3,958 FP8 figure beside it is also a with-sparsity number. See <a href={href('appA', 'A.5')}>A.5</a>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Chú thích Step 1 của Fig. A.11 nói "divide row 1 by 3" trong khi hàng được chia cho 2, và Step 5 và 6 của nó in hàng
              1 của ví dụ ở Fig. A.10. Chú thích Step 5 của Fig. A.10 nói "equation" ở chỗ các chú thích khác nói "row". Xem{' '}
              <a href={href('appA', 'A.7')}>A.7</a>.
            </li>
          }
        >
          <li>
            Fig. A.11's Step 1 caption says "divide row 1 by 3" where the row is divided by 2, and its Steps 5 and 6 print
            row 1 of Fig. A.10's example. Fig. A.10's Step 5 caption says "equation" where the others say "row". See{' '}
            <a href={href('appA', 'A.7')}>A.7</a>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              pp. 598–599 định nghĩa "partial pivoting" là chọn trong một tập phương trình cục bộ. Trong LAPACK, cuSOLVER,
              rocSOLVER và MPS nó nghĩa là hoán đổi hàng trong cột hiện tại; hoán đổi cả hàng và cột là chọn phần tử chốt toàn
              phần (complete pivoting). Xem <a href={href('appA', 'A.7')}>A.7</a>.
            </li>
          }
        >
          <li>
            pp. 598–599 define "partial pivoting" as choosing among a localized set of equations. In LAPACK, cuSOLVER,
            rocSOLVER and MPS it means swapping rows within the current column; swapping rows and columns is complete
            pivoting. See <a href={href('appA', 'A.7')}>A.7</a>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Bài tập 4 nhắc đến "the algorithm of Fig. A.4", vốn là một bảng. Chúng tôi hiểu ý là cây hội tụ của chương 10. Xem{' '}
              <a href={href('appA', 'A.6')}>A.6</a>.
            </li>
          }
        >
          <li>
            Exercise 4 refers to "the algorithm of Fig. A.4", which is a table. We take chapter 10's convergent tree to be
            meant. See <a href={href('appA', 'A.6')}>A.6</a>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Những điều nhỏ: Bài tập 1 và 2 liệt kê các trường theo thứ tự dấu, phần định trị, số mũ; Figs. A.2 và A.3 dùng chung
              một chú thích; và có lỗi đánh máy ở pp. 583, 585, 586, 590, 591, 592 và 597.
            </li>
          }
        >
          <li>
            Small things: Exercises 1 and 2 list the fields as sign, mantissa, exponent; Figs. A.2 and A.3 share a caption;
            and there are typos on pp. 583, 585, 586, 590, 591, 592 and 597.
          </li>
        </Translated>
      </ul>
      <Aside>
        <Translated vi={<p>Đây là cách chúng tôi đọc một bản PDF. Nếu bản của bạn khác, hãy tin bản của bạn.</p>}>
          <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
        </Translated>
      </Aside>

      <h2><Translated vi="Dự án đến lúc này">The project so far</Translated></h2>
      <p>
        <Translated
          vi={
            <>
              Phụ lục này không thêm mốc dự án nào của riêng nó. Thứ nó thêm vào là bằng chứng cho một quyết định mà hai mốc dự án
              phụ thuộc: kernel dùng bộ tích lũy nào. Lab A.1 đã đo, trên thiết bị của bạn, một bộ tích lũy half hoặc bfloat16 làm
              gì với một tổng dài và phép bù của Kahan tốn bao nhiêu.
            </>
          }
        >
          This appendix adds no milestone of its own. What it adds is the evidence for a decision two milestones depend
          on: which accumulator a kernel uses. Lab A.1 measured, on your device, what a half or bfloat16 accumulator does
          to a long sum and what Kahan's compensation costs.
        </Translated>{' '}
        {backends.length === 0
          ? tr('No lab results are recorded on this machine yet.', 'Chưa có kết quả lab nào được ghi trên máy này.')
          : lab.length
            ? tr(`Your own version passes on ${lab.join(' and ')} on this machine.`, `Bản làm của bạn chạy đạt trên ${lab.join(' và ')} trên máy này.`)
            : tr('Your own version has not passed on this machine yet.', 'Bản làm của bạn chưa chạy đạt trên máy này.')}{' '}
        <Translated vi={<>Xem các <a href={href('appA', 'labs')}>lab</a>.</>}>
          See the <a href={href('appA', 'labs')}>labs</a>.
        </Translated>
      </p>
      <Translated
        vi={
          <p>
            Module <a href={href('x2')}>X2</a> tiếp nối từ đây. Nó viết các GEMV trọng số int8 và 4 bit, nơi trọng số hẹp còn bộ
            tích lũy thì không được hẹp, và đưa chúng vào mốc dự án M9, bước decode lượng tử hóa, nơi đo xem ít byte hơn mang lại
            gì. Softmax của mốc dự án M4, lab 10.3, giữ tổng của nó bằng float vì những lý do mà các mục{' '}
            <a href={href('appA', 'A.3')}>A.3</a> và <a href={href('appA', 'A.6')}>A.6</a> đã nêu.
          </p>
        }
      >
        <p>
          Module <a href={href('x2')}>X2</a> picks this up. It writes int8- and 4-bit-weight GEMVs, where the weights are
          narrow and the accumulator must not be, and puts them into milestone M9, the quantised decode step, which
          measures what the fewer bytes buy. Milestone M4's softmax, lab
          10.3, keeps its sum in float for the reasons sections <a href={href('appA', 'A.3')}>A.3</a> and{' '}
          <a href={href('appA', 'A.6')}>A.6</a> gave.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Lượng tử hóa số nguyên, với các hệ số tỉ lệ và điểm không (zero point) của nó, và mã tensor-core hay matrix-unit cho
              các kiểu hẹp không có trong sách. Module X2 trình bày chúng.
            </p>
          }
        >
          <p>
            Integer quantisation, with its scales and zero points, and tensor-core or matrix-unit code for narrow types
            are not in the book. Module X2 covers them.
          </p>
        </Translated>
      </Beyond>
    </>
  );
}
