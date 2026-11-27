import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Source, Today } from '../../components/notes';
import { labStatus, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';

// Prerequisite questions drawn from chapters 4, 5, 6, 10 and 15 and their facts files. Each
// explanation names where the idea was taught and which section of this chapter uses it.
export const questionsS200: Record<string, Question> = {
  'ch20/start/q-gemv-ratio': {
    kind: 'numeric',
    id: 'ch20/start/q-gemv-ratio',
    prompt: (
      <Translated
        vi={
          <>
            Một bước decode nhân một hàng activation fp32 với một ma trận trọng số 4096 × 4096. Chỉ tính các byte của trọng
            số, nó thực hiện bao nhiêu phép toán dấu phẩy động trên mỗi byte?
          </>
        }
      >
        <>
          A decode step multiplies one fp32 activation row by a 4096 × 4096 weight matrix. Counting only the bytes of the
          weights, how many floating-point operations does it do per byte?
        </>
      </Translated>
    ),
    answer: 0.5,
    tol: 0.01,
    unit: 'FLOP/B',
    explain: (
      <Translated
        vi={
          <p>
            Mỗi trọng số được đọc một lần và dùng trong một phép nhân-cộng: 2 FLOP cho 4 byte, nên 0.5 FLOP/B. Hai vectơ thêm
            16 KiB so với 64 MiB trọng số và không thay đổi gì. Với một hàng thì không trọng số nào được dùng lần thứ hai, nên
            không kích thước ô nào nâng được tỷ lệ này; chỉ thêm hàng mới nâng được (batching). Được dạy ở{' '}
            <a href={href('ch05', '5.1')}>ch. 5 §5.1</a>. Mục 20.4 và 20.6 dựa trên nó: decode chính là phép toán này, lặp lại
            qua mọi lớp và qua toàn bộ KV cache.
          </p>
        }
      >
        <p>
          Each weight is read once and used in one multiply-add: 2 FLOP for 4 bytes, so 0.5 FLOP/B. The two vectors add
          16 KiB against 64 MiB of weights and change nothing. With one row there is no second use of any weight, so no
          tile size can raise the ratio; only more rows can (batching). Taught in{' '}
          <a href={href('ch05', '5.1')}>ch. 5 §5.1</a>. Sections 20.4 and 20.6 build on it: decode is this operation,
          repeated over every layer and over the whole KV cache.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.1, pp. 94–97; ch. 20 pp. 491–492, 504–505', // i18n-ignore: citation
  },
  'ch20/start/q-overflow': {
    kind: 'choice',
    id: 'ch20/start/q-overflow',
    prompt: (
      <Translated
        vi={
          <>
            Một hàng logits fp32, tất cả bằng 100, đi qua softmax <em>mà không</em> trừ giá trị lớn nhất của hàng. Kết quả ra
            là gì?
          </>
        }
      >
        <>
          A row of fp32 logits, all equal to 100, goes through softmax <em>without</em> subtracting the row max. What comes
          out?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="NaN ở mọi phần tử.">{'NaN in every entry.'}</Translated>, correct: true },
      { label: <Translated vi="1/n ở mọi phần tử.">{'1/n in every entry.'}</Translated>, why: <Translated vi="Đó là kết quả khi đã trừ giá trị lớn nhất. Không có phép trừ thì chẳng còn gì hữu hạn để chia.">{'That is the max-subtracted result. Without the subtraction nothing finite is left to divide.'}</Translated> },
      { label: <Translated vi="0 ở mọi phần tử.">{'0 in every entry.'}</Translated>, why: <Translated vi="Điều đó cần một tử số hữu hạn trên một tổng vô hạn. Ở đây các tử số cũng tràn số.">{'That would need a finite numerator over an infinite sum. Here the numerators overflow too.'}</Translated> },
      { label: <Translated vi="1 ở phần tử đầu, 0 ở các phần tử còn lại.">{'1 in the first entry, 0 elsewhere.'}</Translated>, why: <Translated vi="Mọi phần tử được đối xử như nhau; không có gì làm phần tử đầu khác biệt.">{'Every entry is treated alike; nothing singles out the first.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            exp(100) vượt quá số float lớn nhất (exp của mọi số lớn hơn khoảng 88.7 đều vậy), nên mọi tử số và tổng đều là vô
            cực, và vô cực chia vô cực là NaN. Trừ đi giá trị lớn nhất làm mọi số mũ nhiều nhất là 0 và mỗi phần tử của hàng ra
            1/n. Lab 10.3 kiểm tra đúng điều này với các hàng được cộng thêm +90 (
            <a href={href('ch10', 'labs')}>ch. 10 labs</a>). Eq. 20.2 (mục <a href={href('ch20', '20.2')}>20.2</a>) là cùng
            cách sửa đó, và kernel của mục <a href={href('ch20', '20.3')}>20.3</a> dùng nó.
          </p>
        }
      >
        <p>
          exp(100) is beyond the largest float (exp of anything above about 88.7 is), so every numerator and the sum are
          infinity, and infinity divided by infinity is NaN. Subtracting the max makes every exponent at most 0 and the
          row comes out as 1/n each. Lab 10.3 tests exactly this with rows offset by +90 (
          <a href={href('ch10', 'labs')}>ch. 10 labs</a>). Eq. 20.2 (section <a href={href('ch20', '20.2')}>20.2</a>) is the
          same fix, and section <a href={href('ch20', '20.3')}>20.3</a>&rsquo;s kernel uses it.
        </p>
      </Translated>
    ),
    source: 'lab 10.3 (labs/ch10/lab3_softmax_argmax); ch. 20 Eq. 20.2, p. 484', // i18n-ignore: citation
  },
  'ch20/start/q-lane0': {
    kind: 'choice',
    id: 'ch20/start/q-lane0',
    prompt: (
      <Translated
        vi={
          <>
            Sau một phép rút gọn warp kiểu shuffle-down (hoặc <code>cub::WarpReduce</code>), mọi lane phải chia các giá trị của
            chính nó cho tổng của warp. Còn thiếu gì?
          </>
        }
      >
        <>
          After a shuffle-down warp reduction (or <code>cub::WarpReduce</code>), every lane must divide its own values by
          the warp&rsquo;s sum. What is missing?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Chỉ lane 0 giữ tổng; các lane khác cần nó được phát (broadcast) tới.">{'Lane 0 alone holds the sum; the other lanes need it broadcast.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Không thiếu gì: mọi lane kết thúc phép rút gọn shuffle-down đều giữ tổng.">{'Nothing: every lane ends a shuffle-down reduction holding the sum.'}</Translated>,
        why: <Translated vi="Shuffle-down gấp các giá trị dồn về lane 0. Các lane khác kết thúc với tổng từng phần.">{'Shuffle-down folds the values towards lane 0. The other lanes end with partial sums.'}</Translated>,
      },
      {
        label: <Translated vi="Một __syncthreads() sau phép rút gọn, để mọi lane đọc được tổng.">{'A __syncthreads() after the reduction, so that every lane can read the sum.'}</Translated>,
        why: <Translated vi="Tổng nằm trong một thanh ghi của lane 0. Rào chắn không làm lane nào khác thấy được một thanh ghi.">{'The sum is in a register of lane 0. A barrier makes no other lane able to see a register.'}</Translated>,
      },
      {
        label: <Translated vi="Không thiếu gì, miễn là warp hội tụ hoàn toàn khi nó chia.">{'Nothing, provided the warp is fully converged when it divides.'}</Translated>,
        why: <Translated vi="Sự hội tụ quyết định ai thực thi cùng nhau, không quyết định thanh ghi nào giữ gì.">{'Convergence decides who executes together, not which register holds what.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Phép rút gọn shuffle-down gấp dồn về lane 0, và CUB ghi trong tài liệu rằng đầu ra của <code>WarpReduce</code>
            chỉ hợp lệ ở lane 0 (facts A2). Hãy phát nó đi, hoặc rút gọn bằng shuffle xor để mọi lane kết thúc với tổng đầy
            đủ. <code>simd_sum</code> của Metal đã trả nó về cho mọi lane (facts C3). Được dạy ở{' '}
            <a href={href('ch10', '10.7')}>ch. 10 §10.7</a>. Kernel flash của sách (mục 20.5, Hình 20.12 và 20.15) thực hiện
            lần phát này sau mỗi phép rút gọn warp.
          </p>
        }
      >
        <p>
          A shuffle-down reduction folds towards lane 0, and CUB documents <code>WarpReduce</code>&rsquo;s output as
          valid in lane 0 only (facts A2). Broadcast it, or reduce with xor shuffles so that every lane ends with the full
          sum. Metal&rsquo;s <code>simd_sum</code> already returns it to every lane (facts C3). Taught in{' '}
          <a href={href('ch10', '10.7')}>ch. 10 §10.7</a>. The book&rsquo;s flash kernel (section 20.5, Figs. 20.12 and
          20.15) does this broadcast after each warp reduction.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.7; notes/today/ch20-facts.md A2, C3; ch. 20 pp. 500–503', // i18n-ignore: citation
  },
  'ch20/start/q-two-reductions': {
    kind: 'choice',
    id: 'ch20/start/q-two-reductions',
    prompt: (
      <Translated
        vi={
          <>
            Trong softmax của lab 10.3, <code>blockReduceMax</code> để giá trị lớn nhất của hàng trong <code>scratch[0]</code>
            và mọi luồng đọc nó. Lời gọi tiếp theo, <code>blockReduceSum</code>, bắt đầu bằng việc lane 0 của mỗi warp ghi tổng
            từng phần của warp mình vào <code>scratch[warp]</code>. Vì sao phải có một <code>__syncthreads()</code> giữa hai
            lời gọi?
          </>
        }
      >
        <>
          In lab 10.3&rsquo;s softmax, <code>blockReduceMax</code> leaves the row max in <code>scratch[0]</code> and every
          thread reads it. The next call, <code>blockReduceSum</code>, starts with lane 0 of each warp writing its
          warp&rsquo;s partial sum into <code>scratch[warp]</code>. Why must a <code>__syncthreads()</code> sit between the
          two calls?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Warp 0 có thể ghi đè scratch[0] trước khi các warp chậm hơn kịp đọc giá trị lớn nhất.">{'Warp 0 may overwrite scratch[0] before slower warps have read the max.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Để giá trị lớn nhất nằm trong scratch[0] trước khi các luồng khác đọc nó.">{'So that the max is in scratch[0] before the other threads read it.'}</Translated>,
        why: <Translated vi="Lần đọc-sau-ghi đó đã được bảo vệ bởi rào chắn bên trong blockReduceMax, trước khi nó trả về.">{'That read-after-write is guarded by the barrier inside blockReduceMax, before its return.'}</Translated>,
      },
      {
        label: <Translated vi="Không có nó, trình biên dịch có thể đảo thứ tự hai lời gọi.">{'Without it the compiler may reorder the two calls.'}</Translated>,
        why: <Translated vi="Rào chắn sắp thứ tự các luồng với nhau; việc một luồng tự đảo thứ tự các câu lệnh của nó không phải là rủi ro ở đây.">{'The barrier orders threads against each other; reordering one thread\'s own statements is not the risk.'}</Translated>,
      },
      {
        label: <Translated vi="Không cần: các warp của một khối chạy đồng bộ từng bước (lockstep).">{'It is not needed: the warps of one block run in lockstep.'}</Translated>,
        why: <Translated vi="Không có gì hứa điều đó. Các warp của một khối chạy theo bất kỳ thứ tự nào giữa các rào chắn.">{'Nothing promises that. Warps of a block run in any order between barriers.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lần ghi không cần gì từ warp chậm; xung đột chỉ tồn tại vì vị trí bộ nhớ được dùng lại. Đó là ghi-sau-đọc, rào chắn
            thứ hai của phép nhân ma trận phân ô (<a href={href('ch05', '5.4')}>ch. 5 §5.4</a>), và chú thích của lab 10.3 nói
            rõ điều đó ngay tại rào chắn ấy. Mục <a href={href('ch20', '20.3')}>20.3</a> đặt kernel của sách, vốn phát giá trị
            qua một biến vô hướng riêng, cạnh kernel này và thấy rằng ở đó không phải chính rào chắn này giữ cho nó đúng.
          </p>
        }
      >
        <p>
          The write needs nothing from the slow warp; the conflict exists only because the location is reused. That is a
          write-after-read, the second barrier of the tiled matmul (<a href={href('ch05', '5.4')}>ch. 5 §5.4</a>), and
          lab 10.3&rsquo;s comment says so at that barrier. Section <a href={href('ch20', '20.3')}>20.3</a> puts the
          book&rsquo;s kernel, which broadcasts through a separate scalar, next to this one and finds that there the same
          barrier is not what keeps it right.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.4, pp. 110–111; labs/ch10/lab3_softmax_argmax/solution/kernel.cu', // i18n-ignore: citation
  },
  'ch20/start/q-qkt-linear': {
    kind: 'choice',
    id: 'ch20/start/q-qkt-linear',
    prompt: (
      <Translated
        vi={
          <>
            Q và K có thứ tự hàng (row-major) [N × d]. Lời gọi mốc dự án nào của bạn tính <code>S = Q K^T</code> mà không cần
            chuyển vị hay sao chép thêm?
          </>
        }
      >
        <>
          Q and K are row-major [N × d]. Which of your milestone calls computes <code>S = Q K^T</code> with no transpose
          or extra copy?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="launchLinear(Q, K, S, N, d, N) của lab 15.6.">{'Lab 15.6\'s launchLinear(Q, K, S, N, d, N).'}</Translated>, correct: true },
      {
        label: <Translated vi="launchGemm(Q, K, S, N, d, N) của lab 5.3.">{'Lab 5.3\'s launchGemm(Q, K, S, N, d, N).'}</Translated>,
        why: <Translated vi="launchGemm nhân với đối số thứ hai đúng như nó được lưu, [k × l]. K là [N × d], nên cách này cần K^T nằm trong bộ nhớ.">{'launchGemm multiplies by its second argument as stored, [k × l]. K is [N × d], so this needs K^T in memory.'}</Translated>,
      },
      {
        label: <Translated vi="Cái nào cũng được, vì cả hai đều tính tích theo thứ tự hàng.">{'Either one, since both compute row-major products.'}</Translated>,
        why: <Translated vi="Chúng khác nhau ở cách lưu toán hạng thứ hai, và đó chính là toàn bộ câu hỏi.">{'They differ in how the second operand is stored, which is the whole question.'}</Translated>,
      },
      {
        label: <Translated vi="Không cái nào, nếu không có một kernel chuyển vị trước.">{'Neither, without a transpose kernel first.'}</Translated>,
        why: <Translated vi="Một trong hai đã đọc toán hạng thứ hai của nó ở dạng chuyển vị.">{'One of them already reads its second operand transposed.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>launchLinear(X, W, Y, batch, in, out)</code> tính Y = X W<sup>T</sup> với W được lưu theo (out, in), giống
            một trọng số <code>nn.Linear</code>. K đúng là như vậy với N đầu ra rộng d. Nó không có đối số hệ số co giãn, nên
            1/√d phải đặt ở chỗ khác; <code>launchSoftmaxRows</code> của lab 10.3 đã nhận một hệ số như thế. P V thì ngược lại
            (V là [N × d], toán hạng thứ hai đúng như được lưu), tức là <code>launchGemm</code> của lab 5.3. Được dạy ở{' '}
            <a href={href('ch15', '15.1')}>ch. 15 §15.1</a> và <a href={href('ch15', 'labs')}>lab 15.6</a>; mục 20.3 xây
            pipeline không gộp từ hai lời gọi này.
          </p>
        }
      >
        <p>
          <code>launchLinear(X, W, Y, batch, in, out)</code> computes Y = X W<sup>T</sup> with W stored (out, in), like an{' '}
          <code>nn.Linear</code> weight. K is exactly that with N outputs of width d. It has no scale argument, so 1/√d
          goes elsewhere; lab 10.3&rsquo;s <code>launchSoftmaxRows</code> already takes one. P V is the other way round
          (V is [N × d], the second operand as stored), which is lab 5.3&rsquo;s <code>launchGemm</code>. Taught in{' '}
          <a href={href('ch15', '15.1')}>ch. 15 §15.1</a> and <a href={href('ch15', 'labs')}>lab 15.6</a>; section 20.3
          builds the unfused pipeline from these two calls.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.1; labs/ch15/lab6_linear/README.md; labs/ch05/lab3_gemm', // i18n-ignore: citation
  },
  'ch20/start/q-coalesce': {
    kind: 'choice',
    id: 'ch20/start/q-coalesce',
    prompt: (
      <Translated
        vi={
          <>
            Trong cách bố trí của sách, mỗi lane của một warp rộng 32 đọc <code>Q[row*d + lane + 32*k]</code> với cùng{' '}
            <code>k</code> và cùng <code>row</code>. Lần nạp đó có được hợp nhất (coalesced) không?
          </>
        }
      >
        <>
          In the book&rsquo;s layout, the lanes of a 32-wide warp each read <code>Q[row*d + lane + 32*k]</code> for the
          same <code>k</code> and the same <code>row</code>. Is the load coalesced?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Có: 32 float liên tiếp của một hàng.">{'Yes: 32 consecutive floats of one row.'}</Translated>, correct: true },
      {
        label: <Translated vi="Không: về sau mỗi lane còn đọc cả các cột khác.">{'No: each lane later reads other columns too.'}</Translated>,
        why: <Translated vi="Hợp nhất truy cập nói về một lệnh nạp trên cả warp, không phải về những gì một lane làm qua các vòng lặp.">{'Coalescing is about one load instruction across the warp, not what a lane does across iterations.'}</Translated>,
      },
      { label: <Translated vi="Chỉ khi d bằng 32.">{'Only when d is 32.'}</Translated>, why: <Translated vi="d nào cũng được: với một k, 32 địa chỉ là liên tiếp bất kể độ dài hàng.">{'Any d works: for one k the 32 addresses are consecutive whatever the row length.'}</Translated> },
      { label: <Translated vi="Chỉ với k = 0.">{'Only for k = 0.'}</Translated>, why: <Translated vi="Đổi k dịch cả dải đi 32 float; nó vẫn là một dải liên tiếp.">{'Changing k shifts the whole run by 32 floats; it stays one consecutive run.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Với một <code>k</code> cố định, các địa chỉ là <code>row*d + 32*k</code> cộng 0 đến 31: một dải liên tiếp 128
            byte. Đây là <code>load_Q</code> của sách (Hình 20.10), chia mỗi hàng của Q cho các lane theo thứ tự xen kẽ chính
            là để mọi lần nạp đều được hợp nhất. Được dạy ở{' '}
            <a href={href('ch06', '6.1')}>ch. 6 §6.1</a>; dùng trong mục 20.5.
          </p>
        }
      >
        <p>
          For a fixed <code>k</code> the addresses are <code>row*d + 32*k</code> plus 0 to 31: one consecutive run of
          128 bytes. This is the book&rsquo;s <code>load_Q</code> (Fig. 20.10), which deals each row of Q out to the lanes
          in interleaved order exactly so that every load is coalesced. Taught in{' '}
          <a href={href('ch06', '6.1')}>ch. 6 §6.1</a>; used in section 20.5.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.1; ch. 20 Fig. 20.10, p. 498', // i18n-ignore: citation
  },
  'ch20/start/q-wave64': {
    kind: 'choice',
    id: 'ch20/start/q-wave64',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel viết cho warp 32 lane gán cứng số 32 và gọi shuffle với độ rộng mặc định. Trên một GPU AMD gfx9,{' '}
            <code>warpSize</code> là bao nhiêu, và cái gì hỏng?
          </>
        }
      >
        <>
          A kernel written for 32-lane warps hard-codes 32 and calls shuffles with their default width. On an AMD gfx9
          GPU, what is <code>warpSize</code>, and what breaks?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="64; một lane ở nửa trên xin lane 5 sẽ nhận lane 5 của nửa dưới.">{'64; an upper-half lane asking for lane 5 gets the lower half\'s lane 5.'}</Translated>,
        correct: true,
      },
      { label: <Translated vi="32: HIP cũng báo 32 trên GPU AMD, nên kernel chạy không cần đổi.">{'32: HIP reports 32 on AMD GPUs too, so the kernel runs unchanged.'}</Translated>, why: <Translated vi="HIP báo độ rộng phần cứng: 64 trên gfx9, 32 trên gfx10 trở về sau.">{'HIP reports the hardware width: 64 on gfx9, 32 on gfx10 and later.'}</Translated> },
      { label: <Translated vi="64; trình biên dịch tách mỗi wavefront làm đôi, nên không có gì hỏng.">{'64; the compiler splits each wavefront in two, so nothing breaks.'}</Translated>, why: <Translated vi="Không có gì tách nó. Độ rộng mặc định là cả wavefront 64 lane.">{'Nothing splits it. The default width is the whole 64-lane wavefront.'}</Translated> },
      { label: <Translated vi="64; không có gì hỏng, vì lane nguồn của shuffle quay vòng ở 32.">{'64; nothing breaks, because shuffle source lanes wrap at 32.'}</Translated>, why: <Translated vi="Chúng quay vòng ở đối số width, mà mặc định là warpSize, tức 64.">{'They wrap at the width argument, which defaults to warpSize, 64.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Các GPU gfx9 của AMD (dòng Instinct) chạy wavefront 64 lane (ch05 facts B5). Shuffle của HIP tính lane nguồn là{' '}
            <code>(src_lane &amp; (width − 1)) + (self &amp; ~(width − 1))</code> với{' '}
            <code>width = warpSize</code>, nên lane phần cứng 40 xin lane 5 sẽ nhận lane phần cứng 5, vốn thuộc warp logic
            kia (facts B2). Cách sửa: biến độ rộng thành một tham số lúc biên dịch bằng đúng wavefront, hoặc truyền width 32 cho
            mọi shuffle. Được dạy ở <a href={href('ch04', '4.4')}>ch. 4 §4.4</a> và chương 10; mục 20.5 gặp lại nó trong kernel
            flash của sách.
          </p>
        }
      >
        <p>
          AMD&rsquo;s gfx9 GPUs (the Instinct line) run 64-lane wavefronts (ch05 facts B5). HIP&rsquo;s shuffle computes
          the source as <code>(src_lane &amp; (width − 1)) + (self &amp; ~(width − 1))</code> with{' '}
          <code>width = warpSize</code>, so hardware lane 40 asking for lane 5 gets hardware lane 5, which belongs to the
          other logical warp (facts B2). Fixes: make the width a compile-time parameter equal to the wavefront, or pass
          width 32 to every shuffle. Taught in <a href={href('ch04', '4.4')}>ch. 4 §4.4</a> and chapter 10; section 20.5
          meets it in the book&rsquo;s flash kernel.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch05-facts.md B5; notes/today/ch20-facts.md B2', // i18n-ignore: citation
  },
};

// ---------------------------------------------------------------- milestone readiness
// What the chapter composes: M2 and M3 are the GEMMs of the unfused pipeline, M4 the row
// softmax lab 20.1's kernel grows from. Pass/fail comes from this machine's reports only.
const MILESTONES: { m: string; lab: string; num: string; call: string; role: string }[] = [
  { m: 'M2', lab: 'ch05/lab3_gemm', num: '5.3', call: 'launchGemm', role: 'O = P V' },
  { m: 'M3', lab: 'ch15/lab6_linear', num: '15.6', call: 'launchLinear', role: 'S = Q K^T' },
  { m: 'M4', lab: 'ch10/lab3_softmax_argmax', num: '10.3', call: 'launchSoftmaxRows', get role() { return tr('the row softmax lab 20.1 extends', 'phép softmax theo hàng mà lab 20.1 mở rộng'); } },
];
const DEVICES: { b: string; name: string }[] = [
  { b: 'metal', name: 'Metal' },
  { b: 'cuda', name: 'CUDA' },
  { b: 'hip', name: 'HIP' },
];

function cell(reports: Reports, b: string, lab: string) {
  const run = reports[b];
  if (!run) return { work: tr('no run on this machine', 'máy này chưa chạy'), ref: '', tone: 'none' as const };
  const w = labStatus(run, lab, 'work');
  const r = labStatus(run, lab, 'solution');
  const untouched = run.work_state?.[lab] === 'untouched';
  const work = w === 'pass' ? tr('yours passes', 'bản của bạn đạt') : w === 'fail' ? tr('yours fails', 'bản của bạn trượt') : untouched ? tr('not started', 'chưa bắt đầu') : tr('no run of yours', 'bạn chưa chạy');
  const ref = r === 'pass' ? tr(`reference verified on ${DEVICES.find((d) => d.b === b)?.name}`, `lời giải tham chiếu verified on ${DEVICES.find((d) => d.b === b)?.name}`) : r === 'fail' ? tr('reference failed here', 'lời giải tham chiếu trượt ở đây') : tr('reference not run', 'lời giải tham chiếu chưa chạy');
  return { work, ref, tone: w === 'pass' ? ('ok' as const) : w === 'fail' ? ('fail' as const) : ('none' as const) };
}

function Readiness() {
  const { reports, loading, reload } = useReports();
  const any = DEVICES.some((d) => reports[d.b]);
  return (
    <Bench
      title={tr("What lab 20.1 builds on", 'Lab 20.1 dựa trên những gì')}
      task={
        <Translated vi="Tìm ba mốc dự án của bạn. Một mốc trượt ở đây cũng sẽ trượt bên trong attention.">
          Find your three milestones. A milestone that fails here will fail inside attention too.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Đọc từ các lần chạy đã ghi lại trên máy này (<code>reports/runs/</code>). Chạy{' '}
              <code>python3 labs/pmpp.py run 10.3</code> (hoặc 5.3, 15.6) để ghi lại kết quả của bạn.{' '}
              <button className="quiet" onClick={reload}>
                Tải lại kết quả
              </button>
            </>
          }
        >
          <>
            Read from this machine&rsquo;s recorded runs (<code>reports/runs/</code>). Run{' '}
            <code>python3 labs/pmpp.py run 10.3</code> (or 5.3, 15.6) to record yours.{' '}
            <button className="quiet" onClick={reload}>
              Reload results
            </button>
          </>
        </Translated>
      }
    >
      <table className="data" data-table="milestones" style={{ margin: 0, background: 'var(--page)' }}>
        <thead>
          <tr>
            <th><Translated vi="Mốc dự án">Milestone</Translated></th>
            <th><Translated vi="Lời gọi">Call</Translated></th>
            {DEVICES.map((d) => (
              <th key={d.b}>{d.name}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {MILESTONES.map((m) => (
            <tr key={m.m}>
              <td>
                <b>{m.m}</b>, <a href={href(m.lab.split('/')[0], 'labs')}><Translated vi={<>lab {m.num}</>}>lab {m.num}</Translated></a>
              </td>
              <td>
                <code>{m.call}</code>
                <div className="muted">{m.role}</div>
              </td>
              {DEVICES.map((d) => {
                const c = cell(reports, d.b, m.lab);
                return (
                  <td key={d.b} data-milestone={m.m} data-backend={d.b} data-work={loading ? 'loading' : c.work}>
                    {loading ? (
                      <span className="muted"><Translated vi="đang đọc báo cáo">reading reports</Translated></span>
                    ) : c.tone === 'none' ? (
                      <span className="muted">{c.work}</span>
                    ) : (
                      <span className={'chip ' + c.tone}>{c.work}</span>
                    )}
                    {!loading && c.ref ? <div className="muted">{c.ref}</div> : null}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="status-line" data-readiness={loading ? 'loading' : any ? 'runs' : 'none'} style={{ margin: '0.7rem 0 0' }}>
        {loading
          ? tr('Reading reports/runs.', 'Đang đọc reports/runs.')
          : any
            ? tr('Lab 20.1\'s benchmark builds the unfused pipeline from the reference solutions of M3 (Q K^T) and M2 (P V); your own M4 reduction is what you reuse in its kernel.', 'Benchmark của lab 20.1 xây pipeline không gộp từ lời giải tham chiếu của M3 (Q K^T) và M2 (P V); phép rút gọn M4 của chính bạn là thứ bạn dùng lại trong kernel của nó.')
            : tr('No runs on this machine yet. Lab 20.1\'s benchmark builds the unfused pipeline from the reference solutions of M3 (Q K^T) and M2 (P V); your own M4 reduction is what you reuse in its kernel.', 'Máy này chưa có lần chạy nào. Benchmark của lab 20.1 xây pipeline không gộp từ lời giải tham chiếu của M3 (Q K^T) và M2 (P V); phép rút gọn M4 của chính bạn là thứ bạn dùng lại trong kernel của nó.')}
      </p>
    </Bench>
  );
}

const ROADMAP: { sec: string; what: string; lab: string }[] = [
  { sec: '20.1, 20.2', get what() { return tr('Every block of a decoder layer as a kernel; the shape of every matrix in one head.', 'Mọi khối của một lớp decoder dưới dạng kernel; kích thước của mọi ma trận trong một head.'); }, lab: '' },
  { sec: '20.3', get what() { return tr('Mask and softmax with one block per row; what the unfused pipeline moves.', 'Mặt nạ và softmax với một khối cho mỗi hàng; những gì pipeline không gộp di chuyển.'); }, get lab() { return tr('Lab 20.1 (M6, part one)', 'Lab 20.1 (M6, phần một)'); } },
  { sec: '20.4', get what() { return tr('KV caching: a decode step is a matrix-vector pass over the whole cache.', 'KV caching: một bước decode là một lượt nhân ma trận-vectơ trên toàn bộ cache.'); }, lab: '' },
  { sec: '20.5', get what() { return tr('Flash attention: one fused kernel; S and P never reach memory.', 'Flash attention: một kernel gộp; S và P không bao giờ chạm tới bộ nhớ.'); }, get lab() { return tr('Labs 20.2 (M6, part two) and 20.3 (race hunt)', 'Lab 20.2 (M6, phần hai) và 20.3 (săn race condition)'); } },
  { sec: '20.6, 20.7', get what() { return tr('Cache size, decode intensity, and MQA, GQA and MLA, which shrink the cache.', 'Kích thước cache, cường độ khi decode, và MQA, GQA và MLA, những cách thu nhỏ cache.'); }, get lab() { return tr('Labs 20.4 (M7) and 20.5', 'Lab 20.4 (M7) và 20.5'); } },
];

export function S200() {
  return (
    <>
      <Translated
        vi={
          <p>
            Một mô hình ngôn ngữ chỉ làm một việc khi suy luận: với mọi token cho đến giờ, system prompt, đầu vào của bạn và
            những gì nó đã viết, nó dự đoán token tiếp theo. Attention là phần của mỗi lớp đọc ngữ cảnh đó và quyết định những
            token nào trước đó quan trọng cho token tiếp theo. Nó cũng là phần có chi phí tăng theo ngữ cảnh, và đó là lý do nó
            có riêng một chương.
          </p>
        }
      >
        <p>
          A language model does one thing at inference: given every token so far, the system prompt, your input and what
          it has already written, it predicts the next one. Attention is the part of each layer that reads that context
          and decides which earlier tokens matter for the next one. It is also the part whose cost grows with the
          context, which is why it gets a chapter.
        </p>
      </Translated>
      <Source pages="477–478" />
      <Translated
        vi={
          <p>
            Sách kể lịch sử trong một đoạn văn (transformer năm 2017, BERT năm 2018, các hướng thay thế như mô hình không gian
            trạng thái đang được khám phá). Phần còn lại của chương là kernel: một softmax, đại số của một cache, và một kernel
            attention gộp mà bạn sẽ viết. Chương này không in một con số thời gian nào, nên mọi con số trên các trang này
            không phải từ phép tính số học đều đến từ chính các lab của bạn.
          </p>
        }
      >
        <p>
          The book tells the history in a paragraph (transformers in 2017, BERT in 2018, alternatives such as state-space
          models being explored). The rest of the chapter is kernels: a softmax, the algebra of a cache, and a fused
          attention kernel you will write. The chapter prints no timings at all, so every number on these pages that is
          not arithmetic comes from your own labs.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              &ldquo;Kiến trúc&rdquo; trong chương này là kiến trúc của mạng, không phải của GPU (chú thích 1, p. 477).
            </p>
          }
        >
          <p>
            &ldquo;Architecture&rdquo; in this chapter means the network&rsquo;s, not the GPU&rsquo;s (footnote 1, p. 477).
          </p>
        </Translated>
      </Aside>

      <h2><Translated vi="Một con số cần nhớ">A number to hold on to</Translated></h2>
      <Translated
        vi={
          <p>
            Trước mọi thứ khác, một dự đoán. Mục 20.6 sẽ định kích thước bộ nhớ mà attention cần giữ lại giữa các token; hãy chốt
            dự đoán của bạn bây giờ và xem trực giác lệch bao xa.
          </p>
        }
      >
        <p>
          Before anything else, one prediction. Section 20.6 will size the memory that attention needs to keep around
          between tokens; commit your guess now and see how far off intuition is.
        </p>
      </Translated>
      <Predict
        id="ch20/start/p-kv-size"
        prompt={
          <Translated
            vi={
              <>
                Llama 2 7B, một cuộc hội thoại 4,096 token, số 16-bit. KV cache của nó (key và value của mọi lớp cho mọi token)
                chiếm bao nhiêu bộ nhớ?
              </>
            }
          >
            <>
              Llama 2 7B, one conversation of 4,096 tokens, 16-bit numbers. How much memory does its KV cache (every
              layer&rsquo;s keys and values for every token) take?
            </>
          </Translated>
        }
        unit="GiB"
      >
        {(v) => (
          <div data-reveal="kv-size">
            <Translated
              vi={
                <p>
                  Eq. 20.8 của sách: batch × số token × 2 (K và V) × số lớp × số chiều ẩn × số byte = 1 × 4096 × 2 × 32 × 4096 × 2 B
                  = 2<sup>31</sup> B = <b>2 GiB</b>, tức 512 KiB mỗi token.{' '}
                  {v >= 1 && v <= 4
                    ? 'Bạn đã đoán đúng khoảng.'
                    : v < 1
                      ? 'Bạn đoán thấp hơn. Mỗi token giữ một key và một value gồm 4,096 số ở mỗi lớp trong 32 lớp: 262,144 số, và số nào cũng được giữ cho đến hết cuộc hội thoại.'
                      : 'Bạn đoán cao hơn con số của sách.'}{' '}
                  Cache của một cuộc hội thoại bằng khoảng một phần bảy của 14 GB mà sách đưa ra cho trọng số của mô hình (2 GiB là
                  2.15 GB), và nó lớn thêm với mỗi token. Mục <a href={href('ch20', '20.6')}>20.6</a> cho bạn thay đổi từng thừa số.
                </p>
              }
            >
              <p>
                The book&rsquo;s Eq. 20.8: batch × tokens × 2 (K and V) × layers × hidden dimension × bytes = 1 × 4096 × 2 ×
                32 × 4096 × 2 B = 2<sup>31</sup> B = <b>2 GiB</b>, or 512 KiB per token.{' '}
                {v >= 1 && v <= 4
                  ? 'You were in the right range.'
                  : v < 1
                    ? 'You guessed lower. Each token keeps a key and a value of 4,096 numbers in each of 32 layers: 262,144 numbers, every one of them kept for the rest of the conversation.'
                    : 'You guessed higher than the book\'s figure.'}{' '}
                One conversation&rsquo;s cache is about a seventh of the 14 GB the book gives for the model&rsquo;s weights (2 GiB is 2.15 GB),
                and it grows with every token. Section <a href={href('ch20', '20.6')}>20.6</a> lets you vary each factor.
              </p>
            </Translated>
            <Translated
              vi={
                <p className="muted">
                  Sách pp. 505–506 (in là &ldquo;2 GB&rdquo;; đơn vị là nhị phân). 32 lớp và số chiều ẩn 4096 của Llama 2 7B được xác
                  nhận trong notes/today/ch20-facts.md E2 và F1.
                </p>
              }
            >
              <p className="muted">
                Book pp. 505–506 (printed &ldquo;2 GB&rdquo;; the unit is binary). Llama 2 7B&rsquo;s 32 layers and hidden
                size 4096 are confirmed in notes/today/ch20-facts.md E2 and F1.
              </p>
            </Translated>
          </div>
        )}
      </Predict>

      <h2><Translated vi="Chương này theo ngôn ngữ kernel">The chapter in kernel terms</Translated></h2>
      <Translated
        vi={
          <p>
            Mỗi mục hoặc giải thích một ma trận hoặc đưa cho bạn một kernel. Các lab ghép những mốc dự án bạn đã có thành đường
            đi attention của một engine suy luận.
          </p>
        }
      >
        <p>
          Each section either explains a matrix or gives you a kernel. The labs compose milestones you already have into
          the attention path of an inference engine.
        </p>
      </Translated>
      <table className="data" data-table="roadmap">
        <thead>
          <tr>
            <th><Translated vi="Mục">Section</Translated></th>
            <th><Translated vi="Bạn nhận được gì">What you get</Translated></th>
            <th><Translated vi="Lab">Lab</Translated></th>
          </tr>
        </thead>
        <tbody>
          {ROADMAP.map((r) => (
            <tr key={r.sec}>
              <td className="nowrap">{r.sec}</td>
              <td>{r.what}</td>
              <td>{r.lab ? <a href={href('ch20', 'labs')}>{r.lab}</a> : <span className="muted"><Translated vi="không có">none</Translated></span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Một tuyên bố phạm vi áp dụng cho cả chương. Attention của sách là nhân quả, một head, một chuỗi, số float 32-bit, với
            độ rộng head và kích thước ô cố định lúc biên dịch. Kernel thực tế thêm nhiều head, batch, kiểu 16-bit và 8-bit,
            cache phân trang và decode tách (split decoding); những chỗ mục 20.5 đến 20.7 nhắc đến chúng đều được đánh dấu là
            không có trong sách.
          </p>
        }
      >
        <p>
          One scope statement covers the whole chapter. The book&rsquo;s attention is causal, one head, one sequence,
          32-bit floats, with the head width and tile sizes fixed at compile time. Production kernels add heads, batches,
          16-bit and 8-bit types, paged caches and split decoding; where sections 20.5 to 20.7 mention those, they are
          marked as not in the book.
        </p>
      </Translated>
      <Source section="20.5" pages="495" />

      <h2><Translated vi="Bạn đang bắt đầu từ đâu">Where you are starting from</Translated></h2>
      <Translated
        vi={
          <p>
            Lab 20.1 không bắt đầu từ con số không. Benchmark của nó xây pipeline không gộp từ các mốc dự án trước của bạn,
            và kernel của nó phát triển từ softmax ở chương 10 của bạn.
          </p>
        }
      >
        <p>
          Lab 20.1 does not start from nothing. Its benchmark builds the unfused pipeline out of your earlier milestones,
          and its kernel grows out of your chapter 10 softmax.
        </p>
      </Translated>
      <Readiness />

      <h2><Translated vi="Điều này nằm ở đâu dưới PyTorch">What this is under PyTorch</Translated></h2>
      <Translated
        vi={
          <p>
            <code>torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)</code>, trên các tensor có số chiều
            head nằm thứ ba tính từ cuối, chính là Eq. 20.1 của chương này. Mã tham chiếu trong tài liệu của nó dựng một mặt nạ
            tam giác dưới, điền âm vô cực vào các logits bị che, lấy softmax và nhân với V: đúng từng bước của mục 20.2. Thứ
            thường chạy thực tế là một kernel gộp: trên CUDA, FlashAttention-2 hoặc một kernel tiết kiệm bộ nhớ, với một
            cài đặt C++ làm phương án dự phòng; trên GPU Apple, các kernel decode và prefill Metal của chính PyTorch, chuyển thể
            từ MLX.
          </p>
        }
      >
        <p>
          <code>torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)</code>, on tensors whose head
          dimension is third from the end, is Eq. 20.1 of this chapter. Its documented reference code builds a
          lower-triangular mask, fills the masked logits with minus infinity, takes a softmax and multiplies by V: the
          steps of section 20.2, literally. What usually runs is a fused kernel: on CUDA, FlashAttention-2 or a
          memory-efficient kernel, with a C++ implementation as the fallback; on Apple GPUs, PyTorch&rsquo;s own Metal
          decode and prefill kernels, adapted from MLX.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md D5, A6; infra-facts.md D1, D2)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              PyTorch 2.15 ghi trong tài liệu ba cài đặt đứng sau SDPA. Attention của llama.cpp nằm trong{' '}
              <code>ggml-cuda/fattn-*.cuh</code> (cũng được build cho HIP) và <code>ggml-metal/kernels/fa_*.metal</code>.
            </p>
          }
        >
          <p>
            PyTorch 2.15 documents three implementations behind SDPA. llama.cpp&rsquo;s attention lives in{' '}
            <code>ggml-cuda/fattn-*.cuh</code> (also built for HIP) and <code>ggml-metal/kernels/fa_*.metal</code>.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Một cái bẫy cần ghi nhớ cho decode. Với <code>is_causal=True</code> và một truy vấn ngắn hơn dãy key, PyTorch căn
            mặt nạ về góc trên bên trái, nên một token mới duy nhất chỉ chú ý tới key 0. FlashAttention và MLX căn nó về góc
            dưới bên phải, nơi truy vấn cuối thấy mọi key. Kernel lab của mục 20.3 theo quy ước thứ hai.
          </p>
        }
      >
        <p>
          One trap to file away for decode. With <code>is_causal=True</code> and a query shorter than the keys, PyTorch
          aligns the mask to the upper left, so a single new token attends to key 0 only. FlashAttention and MLX align it
          to the lower right, where the last query sees every key. Section 20.3&rsquo;s lab kernel takes the second
          convention.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>Không có trong sách. Tài liệu của PyTorch 2.15, changelog của FlashAttention (2.1) và tài liệu của MLX, kiểm tra ngày 2026-11-20 (facts D5, A8, D4).</p>
          }
        >
          <p>Not in the book. PyTorch 2.15&rsquo;s documentation, FlashAttention&rsquo;s changelog (2.1) and MLX&rsquo;s documentation, checked 2026-11-20 (facts D5, A8, D4).</p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Trước khi bắt đầu">Before you start</Translated></h2>
      <Translated
        vi={
          <p>
            Bảy câu hỏi về những gì chương này giả định. Mỗi đáp án nói ý tưởng đó được dạy ở đâu và quay lại ở đâu trong chương
            này; tất cả đều quay lại trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Seven questions on what this chapter assumes. Each answer says where the idea was taught and where in this
          chapter it returns; all of them come back in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={questionsS200['ch20/start/q-gemv-ratio']} label={tr("From chapter 5", 'Từ chương 5')} />
      <Ask q={questionsS200['ch20/start/q-overflow']} label={tr("From chapter 10", 'Từ chương 10')} />
      <Ask q={questionsS200['ch20/start/q-lane0']} label={tr("From chapter 10", 'Từ chương 10')} />
      <Ask q={questionsS200['ch20/start/q-two-reductions']} label={tr("From chapters 5 and 10", 'Từ chương 5 và 10')} />
      <Ask q={questionsS200['ch20/start/q-qkt-linear']} label={tr("From chapter 15", 'Từ chương 15')} />
      <Ask q={questionsS200['ch20/start/q-coalesce']} label={tr("From chapter 6", 'Từ chương 6')} />
      <Ask q={questionsS200['ch20/start/q-wave64']} label={tr("From chapters 4 and 10", 'Từ chương 4 và 10')} />
    </>
  );
}
