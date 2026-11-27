import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';

// Module X2's summary: what to carry forward, a mixed check on situations the sections did
// not show, where each of the module's tasks is worked (the book has no exercises for it),
// where the sources disagree (the module has no PDF errata), and milestone M9.

const FACTS = 'notes/today/x2-facts.md';

// The KV-cache question's numbers: llama.cpp's q8_0 cache type is 34 bytes per 32 values
// (ch20-facts D3), FP16 is 64; weights and cache 4 GB each in the question.
const KV_SPEEDUP = 8 / (4 + (4 * 34) / 64);

export const questionsWrap: Record<string, Question> = {
  'x2/wrap/q-kv-cache': {
    kind: 'numeric',
    id: 'x2/wrap/q-kv-cache',
    prompt: (
      <Translated
        vi={
          <>
            Một mô hình Q4_0 decode một chuỗi ở ngữ cảnh dài. Mỗi bước đọc trọng số của nó, 4 GB, và KV cache của nó, cũng 4
            GB, lưu ở FP16. Bạn chuyển cache sang kiểu cache <code>q8_0</code> của llama.cpp, 34 byte cho mỗi 32 giá trị.
            Theo mô hình byte, một bước decode nhanh hơn bao nhiêu lần?
          </>
        }
      >
        <>
          A Q4_0 model decodes one sequence at a long context. Each step reads its weights, 4 GB, and its KV cache, also 4
          GB, stored in FP16. You switch the cache to llama.cpp's <code>q8_0</code> cache type, 34 bytes per 32 values. By
          the bytes model, how many times faster is a decode step?
        </>
      </Translated>
    ),
    answer: Number(KV_SPEEDUP.toFixed(2)),
    tol: 0.02,
    unit: '×',
    explain: (
      <Translated
        vi={
          <p>
            FP16 lưu 32 giá trị trong 64 byte, nên cache co lại còn 34/64 của 4 GB, tức 2.125 GB. Bước đọc 4 + 2.125 =
            6.125 GB thay vì 8: 8 / 6.125 ≈ {KV_SPEEDUP.toFixed(2)}. Lượng tử hóa trọng số không giúp gì cho cache, và ở
            ngữ cảnh dài cache chiếm một nửa số byte, nên nó giới hạn những gì riêng định dạng trọng số có thể mang lại. Cùng
            phép đếm đó, với byte của cache thay cho byte của trọng số, chính là lập luận của mục X2.1.
          </p>
        }
      >
        <p>
          FP16 stores 32 values in 64 bytes, so the cache shrinks to 34/64 of 4 GB, 2.125 GB. The step reads 4 + 2.125 =
          6.125 GB instead of 8: 8 / 6.125 ≈ {KV_SPEEDUP.toFixed(2)}. Quantising the weights did nothing for the cache, and
          at a long context the cache is half the bytes, so it caps what weight formats alone can buy. The same count, with
          the cache's bytes in place of the weights', is the argument of section X2.1.
        </p>
      </Translated>
    ),
    source: 'ours: the bytes model of X2.1; notes/today/ch20-facts.md D3 (34 bytes per 32 values); x2-facts E1',
    vi: { source: 'của khóa học này: mô hình byte của X2.1; notes/today/ch20-facts.md D3 (34 byte cho mỗi 32 giá trị); x2-facts E1' },
  },
  'x2/wrap/q-own-layout': {
    kind: 'choice',
    id: 'x2/wrap/q-own-layout',
    prompt: (
      <Translated
        vi={
          <>
            Bạn thiết kế định dạng 4 bit của riêng mình: các khối 32 trọng số, mỗi khối gồm một hệ số tỉ lệ float 4 byte rồi đến 16
            byte mã, 20 byte mỗi khối, tất cả trong một mảng có điểm đầu căn chỉnh 16 byte. Những lần nạp mã nào của một khối
            được căn chỉnh?
          </>
        }
      >
        <>
          You design your own 4-bit format: blocks of 32 weights, each a 4-byte float scale followed by 16 bytes of codes,
          20 bytes per block, all in one array whose start is 16-byte aligned. Which loads of a block's codes are aligned?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nạp 4 byte ở mọi khối; nạp 16 byte chỉ ở mỗi khối thứ tư.">
            4-byte loads in every block; 16-byte loads only in every fourth.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Mọi lần nạp đều được căn chỉnh ở mọi khối, vì 20 là bội của 4.">
            Every load is aligned in every block, because 20 is a multiple of 4.
          </Translated>
        ),
        why: (
          <Translated vi="Bội của 4 là đủ cho lần nạp 4 byte, không đủ cho lần nạp 16 byte: mã của khối i bắt đầu ở 20i + 4, và 16 chia hết số đó chỉ khi i là 3, 7, 11, ...">
            A multiple of 4 is enough for 4-byte loads, not for 16-byte ones: block i's codes start at 20i + 4, and 16 divides that only when i is 3, 7, 11, ...
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cứ cách một khối lại có một khối mà lần nạp 4 byte bị lệch căn chỉnh, như với Q4_0.">
            A 4-byte load is misaligned in every other block, as it is in Q4_0.
          </Translated>
        ),
        why: (
          <Translated vi="Hệ số tỉ lệ của Q4_0 là 2 byte, khiến mã của nó nằm ở 2 mod 4 trong các khối chẵn. Một hệ số tỉ lệ 4 byte giữ mã của mọi khối ở bội của 4.">
            Q4_0's scale is 2 bytes, which leaves its codes at 2 mod 4 in even blocks. A 4-byte scale keeps every block's codes at a multiple of 4.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ nạp từng byte là an toàn: một định dạng đóng gói không bảo đảm căn chỉnh nào.">
            Only byte loads are safe: a packed format guarantees no alignment.
          </Translated>
        ),
        why: (
          <Translated vi="Căn chỉnh là phép tính trên địa chỉ, không phải thuộc tính của việc đóng gói. Hãy tính 20i + 4 cho từng độ rộng.">
            Alignment is arithmetic on addresses, not a property of packing. Work out 20i + 4 for each width.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Khối i bắt đầu ở byte 20i, và mã của nó ở 20i + 4. Số đó luôn là bội của 4. Nó là bội của 16 chỉ khi 4i + 4 là
            bội của 16, tức i = 3, 7, 11, v.v. Vậy word 4 byte an toàn ở mọi nơi còn lần nạp vector 16 byte thì không, và không
            có gì trong các trình giả lập báo cho bạn: chúng không kiểm tra căn chỉnh. Tách hệ số tỉ lệ ra một mảng riêng, như lab
            X2.1 và MLX làm, khiến mọi hàng mã được căn chỉnh như chính mảng.
          </p>
        }
      >
        <p>
          Block i starts at byte 20i, and its codes at 20i + 4. That is always a multiple of 4. It is a multiple of 16 only
          when 4i + 4 is, which is i = 3, 7, 11 and so on. So 4-byte words are safe everywhere and 16-byte vector loads are
          not, and nothing in the emulators would tell you: they do not check alignment. Splitting the scales into their
          own array, as lab X2.1 and MLX do, makes every row of codes as aligned as the array.
        </p>
      </Translated>
    ),
    source: 'ours: X2.3\'s alignment arithmetic; x2-facts A5; notes/today/infra-facts.md A6 (float4, int4 and uint4: size 16, alignment 16)',
    vi: { source: 'của khóa học này: phép tính căn chỉnh của X2.3; x2-facts A5; notes/today/infra-facts.md A6 (float4, int4 và uint4: kích thước 16, căn chỉnh 16)' },
  },
  'x2/wrap/q-row-int': {
    kind: 'choice',
    id: 'x2/wrap/q-row-int',
    prompt: (
      <Translated
        vi={
          <>
            Bạn chuyển đường số nguyên của lab X2.3 sang trọng số Q8_0 với activation Q8_1. Để bớt công tính float, bạn cộng
            tổng số nguyên của cả 128 khối của một hàng rộng 4096 vào một <code>int32</code>, rồi nhân một lần với d_w × d_x
            của khối đầu tiên. Sai ở đâu?
          </>
        }
      >
        <>
          You port lab X2.3's integer path to Q8_0 weights against Q8_1 activations. To save the float work, you add the
          integer sums of all 128 blocks of a 4096-wide row in one <code>int32</code>, then multiply once by the first
          block's d_w × d_x. What is wrong?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mỗi khối có d_w và d_x riêng, nên các tổng số nguyên mang đơn vị khác nhau.">
            Each block has its own d_w and d_x, so the integer sums are in different units.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="int32 bị tràn: 128 khối tích int8 vượt quá 2³¹.">
            The int32 overflows: 128 blocks of int8 products exceed 2³¹.
          </Translated>
        ),
        why: (
          <Translated vi="Một khối tối đa là 516,128; 128 khối là khoảng 66 triệu, thấp hơn 2³¹ rất nhiều. Một int32 chứa được khoảng 4,160 khối trong trường hợp xấu nhất.">
            One block is at most 516,128; 128 of them are about 66 million, far below 2³¹. An int32 holds about 4,160 worst-case blocks.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không sai gì: phép cộng số nguyên là chính xác, nên cách gộp nhóm không thể đổi tổng.">
            Nothing: integer addition is exact, so the grouping cannot change the sum.
          </Translated>
        ),
        why: (
          <Translated vi="Các số nguyên chính xác nhưng khác đơn vị cộng lại vẫn vô nghĩa. 3 mét cộng 2 feet không phải là 5 của thứ gì cả.">
            Exact integers in different units still add to nonsense. 3 metres plus 2 feet is not 5 of anything.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="d_w × d_x của khối đầu được tính ở FP16, nên bước đổi tỉ lệ mất độ chính xác.">
            The first block's d_w × d_x is formed in FP16, so the rescale loses precision.
          </Translated>
        ),
        why: (
          <Translated vi="Cả hai hệ số tỉ lệ được nới rộng sang float trước khi nhân, và một lần làm tròn là rất nhỏ. Sai số thật lớn hơn nhiều: 127 khối còn lại có hệ số tỉ lệ khác.">
            Both scales are widened to float before they are multiplied, and one rounding is tiny. The real error is far larger: the other 127 blocks have other scales.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tích vô hướng của một khối là d_w × d_x × Σ q_w q_x, và hệ số tỉ lệ thay đổi từ khối này sang khối khác. Số nguyên bên
            trong một khối, float giữa các khối: đó là hình dạng của <code>mmvq</code> của llama.cpp, và là lý do bước đổi tỉ
            lệ theo khối không thể đưa ra ngoài vòng lặp. Phiên bản trung thực của ý tưởng này là một định dạng khác: một hệ số tỉ lệ
            cho mỗi hàng cho phép bạn cộng cả hàng bằng số nguyên, và nó trả giá bằng độ chính xác thay vì công tính float,
            chính sai số theo hàng mà phép quét của X2.2 cho thấy tăng lên với phân phối đuôi nặng.
          </p>
        }
      >
        <p>
          A block's dot product is d_w × d_x × Σ q_w q_x, and the scales change from block to block. Integer inside a block,
          float across blocks: that is the shape of llama.cpp's <code>mmvq</code>, and the reason the per-block rescale cannot
          be hoisted. The honest version of the idea is a different format: one scale per row lets you sum the whole row in
          integers, and it costs accuracy instead of float work, the per-row error X2.2's sweep shows rising on heavy tails.
        </p>
      </Translated>
    ),
    source: 'x2-facts D3, E3; ours',
    vi: { source: 'x2-facts D3, E3; của khóa học này' },
  },
  'x2/wrap/q-torch-zero': {
    kind: 'choice',
    id: 'x2/wrap/q-torch-zero',
    prompt: (
      <Translated
        vi={
          <>
            Bạn muốn GEMV Q4_0 của mình phục vụ trọng số từ kernel int4 của PyTorch, với quy tắc: giá trị = q × scale + zero,
            trong đó q thuộc −8..7, một scale BF16 và một zero BF16 cho mỗi nhóm 128. Ngoài việc đóng gói lại mã thành q + 8,
            kernel của bạn phải thêm gì?
          </>
        }
      >
        <>
          You want your Q4_0 GEMV to serve weights from PyTorch's int4 kernel, whose rule is: value = q × scale + zero, with
          q in −8..7, a BF16 scale and a BF16 zero per group of 128. Apart from re-packing the codes as q + 8, what must
          your kernel add?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nhóm 128, giải mã hệ số tỉ lệ và zero BF16, và cộng zero × Σx một lần mỗi nhóm.">
            Groups of 128, BF16 scale and zero decoding, and zero × Σx added once per group.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chỉ giải mã BF16: q thuộc −8..7 là dải của Q4_0, nên zero luôn bằng 0.">
            BF16 decoding only: q in −8..7 is Q4_0's range, so the zero is always 0.
          </Translated>
        ),
        why: (
          <Translated vi="Zero của PyTorch là một số float được cộng sau khi nhân hệ số tỉ lệ, lưu theo từng nhóm. Nói chung nó không bằng 0.">
            PyTorch's zero is a float added after scaling, stored per group. It is not 0 in general.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Giải mã BF16, và trừ một zero point số nguyên khỏi mỗi mã trong vòng lặp.">
            BF16 decoding, and an integer zero point subtracted from each code in the loop.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là quy ước của MPSGraph, scale × (q − zeroPoint) với zero số nguyên. Của PyTorch là một số float cộng sau khi nhân hệ số tỉ lệ, và dù theo cách nào nó cũng thuộc về bên ngoài vòng lặp trong.">
            That is MPSGraph's convention, scale × (q − zeroPoint) with an integer zero. PyTorch's is a float added after scaling, and either way it belongs outside the inner loop.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Giải mã BF16, và một lượt đầu chuyển sang float32, vì zero float không thể gộp vào kernel.">
            BF16 decoding, and a first pass to float32, because a float zero cannot be fused.
          </Translated>
        ),
        why: (
          <Translated vi="Mọi độ lệch b theo nhóm đều tách ra được: Σ (d q + b) x = d Σ q x + b Σ x. Một lượt không gộp sẽ chậm hơn cả float32.">
            Any per-group offset b folds out: Σ (d q + b) x = d Σ q x + b Σ x. An unfused pass would be slower than float32.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Kernel của bạn giải mã một d FP16; các hệ số tỉ lệ và zero này là BF16, một bố cục 16 bit khác, nên nó cần một bộ giải mã
            thứ hai. Với một nhóm có hệ số tỉ lệ d và độ lệch b, Σ (d q + b) x = d Σ q x + b Σ x. b của Q4_0 là −8d, của Q4_1 là m, của MLX là
            bias của nó, của PyTorch là zero của nó. Vòng lặp trong vẫn là mã nhân activation; độ lệch tốn một phép nhân-cộng
            mỗi nhóm, từ một tổng của x mà mọi hàng đầu ra dùng chung. Bốn dự án, bốn quy ước cho cùng một ý tưởng, nên hãy đọc
            hàm giải lượng tử hóa trước khi dùng lại một bố cục.
          </p>
        }
      >
        <p>
          Your kernel decodes an FP16 d; these scales and zeros are BF16, a different 16-bit layout, so it needs a second
          decoder. For a group with scale d and offset b, Σ (d q + b) x = d Σ q x + b Σ x. Q4_0's b is −8d, Q4_1's is m, MLX's is its
          bias, PyTorch's is its zero. The inner loop stays codes times activations; the offset costs one multiply-add per
          group, from a sum of x that every output row shares. Four projects, four conventions for one idea, so read the
          dequantiser before you reuse a layout.
        </p>
      </Translated>
    ),
    source: 'x2-facts D6 (int4mm.cu), C4 (MPSGraph), D1, D5, E4', // i18n-ignore: a citation
  },
  'x2/wrap/q-prefill': {
    kind: 'choice',
    id: 'x2/wrap/q-prefill',
    prompt: (
      <Translated
        vi={
          <>
            Máy chủ của bạn chạy một prompt 512 token qua một mô hình Q4_0 trước khi decode, và bộ nhớ của nó không còn chỗ
            cho một bản sao float của trọng số. Kernel nào nên chạy mỗi lớp tuyến tính của lượt prefill đó?
          </>
        }
      >
        <>
          Your server runs a 512-token prompt through a Q4_0 model before decoding, and its memory has no room for a float
          copy of the weights. Which kernel should run each linear layer of that prefill?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Giải lượng tử hóa từng ô của W vào bộ nhớ chia sẻ (shared memory) và đưa vào GEMM phân ô của lab 15.6.">
            Dequantise one tile of W at a time into shared memory and feed lab 15.6's tiled GEMM.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="GEMV của lab X2.2, khởi chạy một lần cho mỗi nhóm 8 token.">
            Lab X2.2's GEMV, launched once per group of 8 tokens.
          </Translated>
        ),
        why: (
          <Translated vi="Cách đó đọc mỗi trọng số 64 lần. Ở 512 token, phép nhân bị giới hạn bởi tính toán, và một GEMV không tái sử dụng W giữa các hàng ngoài batch của nó.">
            That reads every weight 64 times. At 512 tokens the product is arithmetic-bound, and a GEMV has no reuse of W across rows beyond its batch.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Giải lượng tử hóa toàn bộ W sang float32 một lần, rồi chạy GEMM phân ô của lab 15.6.">
            Dequantise the whole of W to float32 once, then run lab 15.6's tiled GEMM.
          </Translated>
        ),
        why: (
          <Translated vi="Cách này đúng, nhưng cần chỗ cho một bản sao float32 của W, gấp 7.1 lần số byte của trọng số Q4_0, và thêm một lượt đi qua nó. Ở 512 token, lượt đó chỉ là một phần khiêm tốn của một GEMM bị giới hạn bởi tính toán, nên ở đây chính bộ nhớ, không phải thời gian, loại nó ra; cách làm từng ô tránh được cả hai, như mul_mm Metal của llama.cpp (D4).">
            It is correct, but it needs room for a float32 copy of W, 7.1 times the bytes of the Q4_0 weights, and an extra pass over it. At 512 tokens that pass is a modest share of an arithmetic-bound GEMM, so memory, not time, rules it out here; the tile-at-a-time route avoids both, as llama.cpp's Metal mul_mm does (D4).
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="torch._int_mm trên mã Q4_0, vốn ổn với 512 hàng activation.">
            torch._int_mm on the Q4_0 codes, which is fine with 512 rows of activations.
          </Translated>
        ),
        why: (
          <Translated vi="Nó nhân int8 với int8 ra int32; nó không biết gì về mã 4 bit hay hệ số tỉ lệ theo khối. Quy tắc M > 16 của nó không phải là trở ngại ở đây.">
            {'It multiplies int8 by int8 into int32; it knows nothing of 4-bit codes or block scales. Its M > 16 rule is not the obstacle here.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Ở 512 token, mỗi trọng số được dùng 512 lần, nên cường độ tính toán cao hơn mọi điểm gãy rất xa và bài toán là của
            chương 15, không phải của module này. Số byte mỗi trọng số không còn quyết định tốc độ; tái sử dụng mới quyết định.{' '}
            <code>mul_mm</code> Metal của llama.cpp làm đúng như vậy: nó giải lượng tử hóa từng ô vào bộ nhớ threadgroup và
            nhân bằng các lệnh ma trận. Định dạng lượng tử hóa tiết kiệm bộ nhớ và thời gian nạp, và tốn một chút công giải nén
            cho mỗi ô.
          </p>
        }
      >
        <p>
          At 512 tokens each weight is used 512 times, so the intensity is far above any ridge point and the problem is
          chapter 15's, not this module's. Bytes per weight no longer set the speed; reuse does. llama.cpp's Metal{' '}
          <code>mul_mm</code> does exactly this: it dequantises each tile into threadgroup memory and multiplies with matrix
          instructions. The quantised format saves memory and load time, and costs a little unpacking per tile.
        </p>
      </Translated>
    ),
    source: 'x2-facts D4 (mul_mm), A6; ch. 15 and its lab 15.6; ours',
    vi: { source: 'x2-facts D4 (mul_mm), A6; ch. 15 và lab 15.6 của nó; của khóa học này' },
  },
};

// ---------------------------------------------------------------- milestone M9
/** Every lab whose code runs in M9's step, with its directory. */
const M9: { num: string; chapter: string; title: string; dir: string }[] = [
  { num: '2.3', chapter: 'ch02', get title() { return tr('M0: bias and residual adds', 'M0: cộng bias và residual'); }, dir: 'ch02/lab3_elementwise' },
  { num: '10.3', chapter: 'ch10', get title() { return tr('M4: softmax and argmax over rows', 'M4: softmax và argmax theo hàng'); }, dir: 'ch10/lab3_softmax_argmax' },
  { num: '22.1', chapter: 'ch22', get title() { return tr('the float32 gather GEMV', 'GEMV gather float32'); }, dir: 'ch22/lab1_gemv' },
  { num: '22.2', chapter: 'ch22', get title() { return tr('M8: the batched decode step M9 starts from', 'M8: bước decode theo batch mà M9 xuất phát từ đó'); }, dir: 'ch22/lab2_batching' },
  { num: 'X2.1', chapter: 'x2', get title() { return tr('the int8-weight GEMV on Q8_0 codes and scales', 'GEMV trọng số int8 trên mã và hệ số tỉ lệ Q8_0'); }, dir: 'x2/lab1_q8gemv' },
  { num: 'X2.2', chapter: 'x2', get title() { return tr("the 4-bit GEMV on llama.cpp's Q4_0 blocks", 'GEMV 4 bit trên các khối Q4_0 của llama.cpp'); }, dir: 'x2/lab2_q4gemv' },
  { num: 'X2.4', chapter: 'x2', get title() { return tr('M9: the quantised decode step', 'M9: bước decode lượng tử hóa'); }, dir: 'x2/lab4_m9' },
];
const where = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
/** Backends (devices first, then the emulator) on which the reader's own work of a lab passes. */
const passingOn = (reports: Reports, lab: string) =>
  [...realBackends(reports), 'emu'].filter((b) => labStatus(reports[b], lab, 'work') === 'pass');

// ---------------------------------------------------------------- the page
export function Wrap() {
  const { progress } = useProgress();
  const { reports, loading } = useReports();
  const ran = Object.keys(reports).length > 0;
  const own = passingOn(reports, 'x2/lab4_m9');
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('x2/') && !k.includes('/p-'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}><Translated vi="Những gì cần mang theo">What to carry forward</Translated></h2>
      <ul>
        <Translated
          vi={
            <li>
              Ở decode, một lớp tuyến tính là một phép nhân ma trận-vector đọc mỗi trọng số một lần, nên tốc độ của nó do số
              byte mỗi trọng số quyết định, tính cả hệ số tỉ lệ: Q8_0 là 8.5 bit, Q4_0 là 4.5. Hãy dự đoán mức tăng tốc bằng tỉ lệ
              số byte di chuyển, rồi đo xem kernel đạt được bao nhiêu phần băng thông của thiết bị.
            </li>
          }
        >
          <li>
            At decode, a linear layer is a matrix-vector product that reads every weight once, so its speed is set by bytes
            per weight, scales included: Q8_0 is 8.5 bits, Q4_0 4.5. Predict the speed-up as the ratio of bytes moved, then
            measure how much of the device's bandwidth the kernel reached.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Gộp bước giải lượng tử hóa vào phép nhân. Một lượt riêng ghi trọng số float rồi đọc lại sẽ di chuyển nhiều
              byte hơn cả việc không lượng tử hóa.
            </li>
          }
        >
          <li>
            Fuse the dequantise into the product. A separate pass that writes float weights and reads them back moves more
            bytes than not quantising at all.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Một định dạng là một quy tắc, một bố cục và một kiểu hệ số tỉ lệ, và mỗi dự án có định dạng riêng: Q4_0 của llama.cpp
              chia giá trị cực trị có dấu cho −8 và đóng gói trọng số j cùng trọng số j + 16; định dạng affine của MLX giữ một
              hệ số tỉ lệ và một bias mỗi nhóm và đóng gói các phần tử kề nhau từ bit thấp lên. Hãy nói rõ bạn đang nói định dạng của
              ai. Mỗi bit thêm vào giảm sai số khoảng một nửa; khối nhỏ giúp nhiều nhất với phân phối đuôi nặng.
            </li>
          }
        >
          <li>
            A format is a rule, a layout and a scale type, and every project has its own: llama.cpp's Q4_0 divides the
            signed extreme by −8 and packs weight j with weight j + 16; MLX's affine format keeps a scale and a bias per
            group and packs neighbours from the low bits up. Name whose format you mean. Each extra bit roughly halves the
            error; small blocks help most on heavy tails.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Một khối 18 byte chỉ được căn chỉnh 2 byte. Hãy nạp mã của nó theo byte hoặc theo word 16 bit, hoặc giữ hệ số tỉ lệ và
              mã trong các mảng riêng. Không trình giả lập nào bắt được lần nạp lệch căn chỉnh; hãy kiểm tra bằng cách đọc mã.
            </li>
          }
        >
          <li>
            An 18-byte block is only 2-byte aligned. Load its codes as bytes or 16-bit words, or keep scales and codes in
            separate arrays. No emulator catches a misaligned load; check by reading.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Đưa độ lệch ra khỏi vòng lặp trong thông qua tổng của các activation, một lần mỗi khối. Sau đó chọn phép tích
              trong theo backend: phép nhân-cộng float trên các nibble đã lọc bằng mặt nạ với activation chia sẵn, ở nơi không có tích
              vô hướng số nguyên (Metal), hoặc activation lượng tử hóa sang int8 và một lệnh tích vô hướng byte (llama.cpp trên
              CUDA).
            </li>
          }
        >
          <li>
            Fold the offset out of the inner loop through the sum of the activations, once per block. Then choose the inner
            product by backend: float multiply-adds on masked nibbles with pre-divided activations where there is no
            integer dot (Metal), or activations quantised to int8 and a byte dot instruction (llama.cpp on CUDA).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Giữ các tổng ở độ rộng lớn: tổng số nguyên bên trong một khối ở int32, float giữa các khối, không bao giờ là half.
              Chặn trên từng cấp trước khi chọn kiểu cho nó.
            </li>
          }
        >
          <li>
            Keep the sums wide: an integer sum inside a block in int32, a float across blocks, never half. Bound each level
            before you choose its type.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Đánh giá một mô hình lượng tử hóa qua logits và token của nó, không qua sai số của trọng số.
            </li>
          }
        >
          <li>
            Judge a quantised model by its logits and its tokens, not by the error of its weights.
          </li>
        </Translated>
      </ul>
      <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
        <Translated
          vi={
            <p>
              §A.5 của phụ lục A nói rằng các định dạng nhỏ hơn nạp ít byte hơn cho mỗi phép toán, rồi dừng lại. Mọi thứ ở trên
              là của riêng module này, xây từ cường độ tính toán của chương 5, phép rút gọn (reduction) của chương 10 và GEMV
              của chương 22.
            </p>
          }
        >
          <p>
            Appendix A's §A.5 says smaller formats fetch fewer bytes per operation, and stops. Everything above is this
            module's own, built from chapter 5's intensity, chapter 10's reductions and chapter 22's GEMV.
          </p>
        </Translated>
      </Beyond>
      <Source section="A.5" pages="592–593" />

      <h2><Translated vi="Một bài kiểm tra tổng hợp">A mixed check</Translated></h2>
      <Translated
        vi={
          <p>
            Các câu này dựa trên toàn bộ module, với những tình huống mà các mục chưa đưa ra.
            {asked ? ` Đến giờ trong module này bạn đã trả lời đúng ${right} trên ${asked} câu, tính theo câu trả lời gần nhất của bạn cho mỗi câu.` : ''}
          </p>
        }
      >
        <p>
          These draw on the whole module, on situations the sections did not show.
          {asked ? ` So far in this module you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
        </p>
      </Translated>
      <Ask q={q['x2/wrap/q-kv-cache']} label={tr('Count the bytes', 'Đếm số byte')} />
      <Ask q={q['x2/wrap/q-own-layout']} />
      <Ask q={q['x2/wrap/q-row-int']} label={tr('Find the defect', 'Tìm lỗi')} />
      <Ask q={q['x2/wrap/q-torch-zero']} />
      <Ask q={q['x2/wrap/q-prefill']} />

      <h2><Translated vi="Các bài tập của module">The module's exercises</Translated></h2>
      <Translated
        vi={
          <p>
            Sách không có bài tập cho module này, vì sách không có kernel lượng tử hóa nào. Mục tiêu của kế hoạch cho X2 là
            viết một GEMM trọng số int8 và một GEMM trọng số 4 bit rồi đo lợi ích về băng thông; ở decode, nơi có lợi ích đó,
            phép nhân ấy là một GEMV. Đây là nơi từng phần được thực hiện.
          </p>
        }
      >
        <p>
          The book has no exercises for this module, because it has no quantised kernel. The plan's goal for X2 is to write
          an int8 and a 4-bit-weight GEMM and measure the bandwidth win; at decode, where the win is, that product is a
          GEMV. Here is where each part of that is worked.
        </p>
      </Translated>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="exercises">
        <thead>
          <tr><th><Translated vi="Nhiệm vụ">Task</Translated></th><th><Translated vi="Thực hiện trong một mục">Worked in a section</Translated></th><th><Translated vi="Xây trong một lab">Built in a lab</Translated></th></tr>
        </thead>
        <tbody>
          <tr><td><Translated vi="Dự đoán tốc độ của một GEMV decode từ số byte mỗi trọng số">Predict a decode GEMV's speed from bytes per weight</Translated></td><td><Translated vi={<>trình khám phá roofline và cảnh minh họa trong <a href={href('x2', 'X2.1')}>X2.1</a></>}>the roofline explorer and scene in <a href={href('x2', 'X2.1')}>X2.1</a></Translated></td><td><Translated vi="các dự đoán của lab X2.1 và lab X2.2">the predictions of lab X2.1 and lab X2.2</Translated></td></tr>
          <tr><td><Translated vi="Đổi một thời gian đo được thành tỉ lệ của băng thông">Turn a measured time into a fraction of bandwidth</Translated></td><td><Translated vi={<>công cụ tính băng thông trong <a href={href('x2', 'X2.1')}>X2.1</a></>}>the bandwidth calculator in <a href={href('x2', 'X2.1')}>X2.1</a></Translated></td><td><Translated vi={<>các cột "% của lab 5.1" trên <a href={href('x2', 'labs')}>trang lab</a></>}>the "% of lab 5.1" columns on the <a href={href('x2', 'labs')}>labs page</a></Translated></td></tr>
          <tr><td><Translated vi="Lượng tử hóa một khối theo quy tắc của llama.cpp và MLX">Quantise a block by llama.cpp's and MLX's rules</Translated></td><td><Translated vi={<>trình chạy từng bước theo khối trong <a href={href('x2', 'X2.2')}>X2.2</a></>}>the block stepper in <a href={href('x2', 'X2.2')}>X2.2</a></Translated></td><td><Translated vi="các kiểm tra bộ lượng tử hóa CPU của lab X2.1 và lab X2.2">the CPU quantiser checks of lab X2.1 and lab X2.2</Translated></td></tr>
          <tr><td><Translated vi="Đánh đổi sai số với số bit, kích thước khối và giá trị ngoại lai">Trade error against bits, block size and outliers</Translated></td><td><Translated vi={<>phép quét sai số trong <a href={href('x2', 'X2.2')}>X2.2</a></>}>the error sweep in <a href={href('x2', 'X2.2')}>X2.2</a></Translated></td><td><Translated vi="không có: sai số của một mô hình được đo trong lab X2.4">none: a model's error is measured in lab X2.4</Translated></td></tr>
          <tr><td><Translated vi="Đóng gói và giải nén hai mã mỗi byte mà không có lần nạp lệch căn chỉnh">Pack and unpack two codes per byte without a misaligned load</Translated></td><td><Translated vi={<>các bố cục, bước giải nén và cảnh minh họa trong <a href={href('x2', 'X2.3')}>X2.3</a></>}>the layouts, unpack steps and scene in <a href={href('x2', 'X2.3')}>X2.3</a></Translated></td><td><Translated vi="lab X2.2">lab X2.2</Translated></td></tr>
          <tr><td><Translated vi="Ánh xạ một warp lên các hàng khối lượng tử hóa">Map a warp onto rows of quantised blocks</Translated></td><td><Translated vi={<>trình chạy từng bước trong <a href={href('x2', 'X2.4')}>X2.4</a></>}>the stepper in <a href={href('x2', 'X2.4')}>X2.4</a></Translated></td><td><Translated vi="lab X2.1 và lab X2.2">lab X2.1 and lab X2.2</Translated></td></tr>
          <tr><td><Translated vi="Tách độ lệch; đường float so với đường số nguyên">Fold the offset; float path against integer path</Translated></td><td><Translated vi={<>trình chạy từng bước ba đường và cảnh minh họa trong <a href={href('x2', 'X2.5')}>X2.5</a></>}>the three-path stepper and scene in <a href={href('x2', 'X2.5')}>X2.5</a></Translated></td><td><Translated vi="lab X2.2 (phép tách) và lab X2.3">lab X2.2 (the fold) and lab X2.3</Translated></td></tr>
          <tr><td><Translated vi="Chọn bộ cộng dồn ở mỗi cấp">Choose the accumulator at each level</Translated></td><td><Translated vi={<>cái thang, cảnh minh họa và bài tìm lỗi trong <a href={href('x2', 'X2.6')}>X2.6</a></>}>the ladder, scene and bug spot in <a href={href('x2', 'X2.6')}>X2.6</a></Translated></td><td><Translated vi="lab X2.3, lỗi cài sẵn 2">lab X2.3, defect 2</Translated></td></tr>
          <tr><td><Translated vi="Viết một GEMV trọng số int8 và đo nó">Write an int8-weight GEMV and measure it</Translated></td><td><Translated vi={<><a href={href('x2', 'X2.1')}>X2.1</a> và <a href={href('x2', 'X2.4')}>X2.4</a></>}><a href={href('x2', 'X2.1')}>X2.1</a> and <a href={href('x2', 'X2.4')}>X2.4</a></Translated></td><td><Translated vi="lab X2.1">lab X2.1</Translated></td></tr>
          <tr><td><Translated vi="Viết một GEMV trọng số 4 bit và đo nó">Write a 4-bit-weight GEMV and measure it</Translated></td><td><Translated vi={<><a href={href('x2', 'X2.3')}>X2.3</a> và <a href={href('x2', 'X2.4')}>X2.4</a></>}><a href={href('x2', 'X2.3')}>X2.3</a> and <a href={href('x2', 'X2.4')}>X2.4</a></Translated></td><td><Translated vi="lab X2.2">lab X2.2</Translated></td></tr>
          <tr><td><Translated vi="Gỡ lỗi một kernel đường số nguyên">Debug an integer-path kernel</Translated></td><td><Translated vi={<><a href={href('x2', 'X2.5')}>X2.5</a> và <a href={href('x2', 'X2.6')}>X2.6</a></>}><a href={href('x2', 'X2.5')}>X2.5</a> and <a href={href('x2', 'X2.6')}>X2.6</a></Translated></td><td><Translated vi="lab X2.3">lab X2.3</Translated></td></tr>
          <tr><td><Translated vi="Lượng tử hóa một bước decode; đánh giá nó qua logits và token">Quantise a decode step; judge it by logits and tokens</Translated></td><td><Translated vi={<>trình khám phá bước và decoder đồ chơi trong <a href={href('x2', 'X2.7')}>X2.7</a></>}>the step explorer and toy decoder in <a href={href('x2', 'X2.7')}>X2.7</a></Translated></td><td><Translated vi="lab X2.4, mốc dự án M9">lab X2.4, milestone M9</Translated></td></tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Các bài tập của chính sách về từng phần được ánh xạ trên trang tóm tắt của chương tương ứng: định dạng số và phép
            cộng dồn ở <a href={href('appA', 'wrap')}>phụ lục A</a>, cường độ tính toán ở <a href={href('ch05', 'wrap')}>chương 5</a>,
            phép rút gọn ở <a href={href('ch10', 'wrap')}>chương 10</a> và decode ở <a href={href('ch22', 'wrap')}>chương 22</a>.
          </p>
        }
      >
        <p>
          The book's own exercises on the parts are mapped on their chapters' summary pages: number formats and summation
          in <a href={href('appA', 'wrap')}>appendix A</a>, intensity in <a href={href('ch05', 'wrap')}>chapter 5</a>,
          reduction in <a href={href('ch10', 'wrap')}>chapter 10</a> and decode in <a href={href('ch22', 'wrap')}>chapter 22</a>.
        </p>
      </Translated>

      <h2><Translated vi="Nơi các nguồn bất đồng">Where the sources disagree</Translated></h2>
      <Translated
        vi={
          <p>
            Module này không có PDF nào để sai: nó không nằm trong sách. Thay vào đó, nó có các dự án và nhà cung cấp bất đồng
            với nhau, và với điều người đọc sẽ giả định. Đây là những bất đồng mà module được xây quanh, tất cả đều đã được kiểm
            tra trong lượt kiểm tra dữ kiện.
          </p>
        }
      >
        <p>
          This module has no PDF to be wrong: it is not in the book. What it has instead are projects and vendors that
          disagree with each other, and with what a reader would assume. These are the disagreements the module was built
          around, all checked in the facts pass.
        </p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>
              <b>Độ lệch.</b> Bốn quy ước cho cùng một ý tưởng: Q4_1 của llama.cpp là d·q + m; affine của MLX là s·q + bias;
              kernel int4 của PyTorch là q·scale + zero, một số float cộng sau khi nhân hệ số tỉ lệ; của MPSGraph là
              scale·(q − zeroPoint) với một zero point số nguyên.
            </li>
          }
        >
          <li>
            <b>The offset.</b> Four conventions for one idea: llama.cpp's Q4_1 is d·q + m; MLX's affine is s·q + bias;
            PyTorch's int4 kernel is q·scale + zero, a float added after scaling; MPSGraph's is scale·(q − zeroPoint) with
            an integer zero point.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Tài liệu của MLX và mã của MLX.</b> Tài liệu lấy hệ số tỉ lệ từ max trừ min với giá trị nhỏ nhất làm bias. Mã lại lấy
              cực trị có độ lớn lớn hơn làm bias, cho phép hệ số tỉ lệ âm, và điều chỉnh hệ số tỉ lệ để số không biểu diễn được, chính xác
              trong số học thực và sai khác tối đa một lần làm tròn ở float32.
            </li>
          }
        >
          <li>
            <b>MLX's docs and MLX's code.</b> The documentation takes the scale from max minus min with the minimum as the
            bias. The code takes the larger-magnitude extreme as the bias, allows a negative scale, and adjusts the scale so
            that zero is representable, exactly in real arithmetic and up to one rounding in float32.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Thứ tự nibble.</b> Q4_0 của llama.cpp đóng gói trọng số j cùng trọng số j + 16; MLX đóng gói các phần tử kề nhau
              từ bit thấp của một word 32 bit lên; Q4_K đặt các nibble thấp của 32 byte lên trước trong mỗi 64 trọng số. Các kiểu
              4 bit mới của MSL hoàn toàn không ghi rõ thứ tự nào.
            </li>
          }
        >
          <li>
            <b>Nibble order.</b> llama.cpp's Q4_0 packs weight j with weight j + 16; MLX packs neighbours from the low bits of
            a 32-bit word up; Q4_K puts the low nibbles of 32 bytes first in each 64 weights. MSL's new 4-bit types document
            no order at all.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Tích vô hướng số nguyên.</b> CUDA có <code>__dp4a</code> từ sm_61. HIP không có <code>__dp4a</code>: chỉ có một
              hàm header không được ghi tài liệu và các builtin khác nhau theo target, chọn sai là lỗi biên dịch. MSL không có
              tích vô hướng số nguyên; <code>dot</code> của hai <code>half4</code> trả về <code>half</code>. Một file{' '}
              <code>.cu</code> build cho HIP không cần sửa đối với luồng, khối, bộ nhớ chia sẻ, rào chắn, atomic kiểu cũ và lời
              gọi runtime; tích vô hướng byte gia nhập danh sách những thứ không như vậy, cùng mặt nạ warp, cooperative groups,
              header half-precision, functor CUB và intrinsic ma trận.
            </li>
          }
        >
          <li>
            <b>The integer dot.</b> CUDA has <code>__dp4a</code> from sm_61. HIP has no <code>__dp4a</code>: an undocumented
            header function and builtins that differ by target, the wrong one a compile error. MSL has no integer dot; its{' '}
            <code>dot</code> of two <code>half4</code> returns <code>half</code>. One <code>.cu</code> builds for HIP unchanged
            for threads, blocks, shared memory, barriers, legacy atomics and runtime calls; a byte dot product joins warp
            masks, cooperative groups, half-precision headers, CUB functors and matrix intrinsics on the list where it
            does not.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Chính trang của PyTorch.</b> <code>torch.ao.quantization</code> đã được torchao thay thế, trong khi trang 2.15
              vẫn lên kế hoạch xóa nó "in 2.10".
            </li>
          }
        >
          <li>
            <b>PyTorch's own page.</b> <code>torch.ao.quantization</code> is superseded by torchao, while the 2.15 page still
            plans its deletion "in 2.10".
          </li>
        </Translated>
      </ul>
      <Today checked={tr(`2026-11-20 in the facts pass (${FACTS} B1, B2, C1, C2, C4, D1, D2, D5, D6)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS} B1, B2, C1, C2, C4, D1, D2, D5, D6)`)}>
        <Translated vi={<p>Mục "Summary of corrections" trong {FACTS} liệt kê mọi mục cùng nguồn của nó.</p>}>
          <p>The "Summary of corrections" in {FACTS} lists every item with its source.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Và đây là những chỗ file dữ kiện và thiết kế của chính chúng tôi đã sai, phát hiện trong khi xây module. File dữ
            kiện nói mã của Q4_0 bị lệch căn chỉnh cho lần nạp 4 byte ở các khối lẻ; với một mảng được căn chỉnh, chúng nằm ở
            2 mod 4 trong các khối chẵn, và A5 giờ mang một đính chính có ghi ngày. Nó nói mã của MLX khiến số không giải lượng
            tử hóa ra đúng 0; ở float32 điều đó đúng tới sai số làm tròn của một phép chia, và D5 ghi rõ như vậy. Nó đưa ra cận
            hàng int32 là khoảng 133,144 số hạng; chính xác 133,144 số hạng vừa đủ và số hạng tiếp theo bị tràn. Thiết kế dự
            kiến mức tăng tốc của bước decode ở batch 8 giảm nhẹ; trên thiết bị ví dụ của sách, mô hình byte khiến head Q4_0 bị
            giới hạn bởi tính toán ở đó, và mức tăng tốc dự đoán giảm khoảng một phần ba (mục{' '}
            <a href={href('x2', 'X2.7')}>X2.7</a>). Và bản nháp đầu của fixture trong các lab dự kiến một khối Q4_0 toàn số
            không lưu hệ số tỉ lệ là +0; quy tắc của llama.cpp chia 0 cho −8 và lưu −0, <code>0x8000</code>.
          </p>
        }
      >
        <p>
          And where our own facts file and design were wrong, found while building the module. The facts file said Q4_0's
          codes were misaligned for a 4-byte load in odd blocks; with an aligned array they are at 2 mod 4 in the even
          blocks, and A5 now carries a dated correction. It said MLX's code makes zero dequantise to exactly 0; in float32
          that holds up to the rounding of one division, and D5 says so. It gave the int32 row bound as about 133,144
          terms; exactly 133,144 fit and the next overflows. The design expected the decode step's speed-up at batch 8 to
          fall slightly; on the book's example device the bytes model makes the Q4_0 head arithmetic-bound there, and the
          predicted speed-up falls by about a third (section <a href={href('x2', 'X2.7')}>X2.7</a>). And the first draft of
          the labs' fixture expected an all-zero Q4_0 block to store a scale of +0; llama.cpp's rule divides 0 by −8 and
          stores −0, <code>0x8000</code>.
        </p>
      </Translated>
      <Aside>
        <Translated vi={<p>Đây là cách chúng tôi đọc các nguồn vào ngày 2026-11-20. Các dự án thay đổi; hãy kiểm tra file trước khi dựa vào một dòng nào.</p>}>
          <p>These are our readings of the sources on 2026-11-20. Projects move; check the file before you rely on a line.</p>
        </Translated>
      </Aside>

      <h2><Translated vi="Những gì module này bỏ qua">What this module leaves out</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn sẽ gặp những thứ này trong các hệ thống thực tế, và không thứ nào có ở đây: chọn hệ số tỉ lệ bằng hiệu chuẩn (GPTQ,
            AWQ, và importance matrix của llama.cpp, vốn chỉ thay đổi cách chọn d); kernel cho các K-quant, mà X2.2 và X2.3 chỉ
            cho thấy bố cục Q4_K; GEMM int4 kiểu Marlin; các định dạng số nguyên, FP8 và MX trên tensor core hoặc matrix core;
            xử lý các kênh ngoại lai trong activation; và KV cache lượng tử hóa vượt ra ngoài phép đếm byte trong câu hỏi đầu
            tiên của trang này.
          </p>
        }
      >
        <p>
          You will meet these in real stacks, and none of them is here: choosing scales by calibration (GPTQ, AWQ, and
          llama.cpp's importance matrix, which changes only how d is chosen); kernels for the K-quants, of which
          X2.2 and X2.3 show only the Q4_K layout; Marlin-style int4 GEMMs; integer, FP8 and MX formats on tensor or matrix cores; handling outlier channels in
          the activations; and quantised KV caches beyond the byte count in this page's first question.
        </p>
      </Translated>
      <Beyond title={tr('Not checked', 'Chưa kiểm tra')}>
        <Translated
          vi={
            <p>
              Lượt kiểm tra dữ kiện không đọc kernel nào trong số này: vLLM, Marlin, AWQ và GPTQ nằm trong danh sách "Not
              checked" của nó ({FACTS}). Danh sách này chỉ nêu tên chúng; mọi khẳng định về cách chúng hoạt động là unverified
              cho đến khi bạn đọc mã của chúng.
            </p>
          }
        >
          <p>
            The facts pass read none of these kernels: vLLM, Marlin, AWQ and GPTQ are on its "Not checked" list ({FACTS}).
            This list names them; any claim about how they work is unverified until you read their code.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Dự án đến lúc này">The project so far</Translated></h2>
      <Translated
        vi={
          <p>
            <code>launchQDecodeStep</code> của lab X2.4 hoàn thành mốc dự án M9: bước decode với trọng số int8 và 4 bit, mức
            tăng tốc được đo trên thiết bị của bạn và sai số được báo cáo so với float32, qua logits và qua token greedy.{' '}
            <span data-m9-own={loading ? 'loading' : own.join(',') || 'none'}>
              {loading
                ? 'Đang đọc báo cáo.'
                : !ran
                  ? 'Chưa có kết quả lab nào được ghi lại trên máy này.'
                  : own.length
                    ? `Bước của chính bạn chạy đạt trên ${own.map(where).join(' và ')} trên máy này.`
                    : 'Bước của chính bạn chưa chạy đạt trên backend nào của máy này.'}
            </span>{' '}
            Xem các <a href={href('x2', 'labs')}>lab</a>.
          </p>
        }
      >
        <p>
          Lab X2.4's <code>launchQDecodeStep</code> completes milestone M9: the decode step with int8 and 4-bit weights, its
          speed-up measured on your device and its error reported against float32, by logits and by greedy tokens.{' '}
          <span data-m9-own={loading ? 'loading' : own.join(',') || 'none'}>
            {loading
              ? 'Reading reports.'
              : !ran
                ? 'No lab results are recorded on this machine yet.'
                : own.length
                  ? `Your own step passes on ${own.map(where).join(' and ')} on this machine.`
                  : 'Your own step does not pass on any backend of this machine yet.'}
          </span>{' '}
          See the <a href={href('x2', 'labs')}>labs</a>.
        </p>
      </Translated>
      <table className="data" data-table="m9">
        <thead>
          <tr><th><Translated vi="Thành phần của M9">Piece of M9</Translated></th><th><Translated vi="Lab">Lab</Translated></th><th><Translated vi="Bài làm của bạn chạy đạt trên">Your own work passes on</Translated></th></tr>
        </thead>
        <tbody>
          {M9.map((m) => {
            const on = passingOn(reports, m.dir);
            return (
              <tr key={m.dir} data-m9={m.num} data-pass={loading ? 'loading' : on.join(',') || 'none'}>
                <td>{m.title}</td>
                <td className="nowrap">
                  <a href={href(m.chapter, 'labs')}><Translated vi={<>lab {m.num}</>}>lab {m.num}</Translated></a>
                </td>
                <td>{loading ? tr('reading reports', 'đang đọc báo cáo') : on.length ? on.map(where).join(', ') : <span className="muted"><Translated vi="chưa có gì">nothing yet</Translated></span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Tiếp theo theo thứ tự của khóa học là chương 7, chương 19 và phụ lục C, rồi đến{' '}
            <a href={href('x3')}>module X3, Đọc các backend thực tế</a>. X3 tìm các kernel đứng sau một op trong llama.cpp,
            MLX và PyTorch, bao gồm cả các kernel ma trận-vector lượng tử hóa mà module này đã dựng lại. Mốc dự án M10 của nó là
            một mô hình kiểu Llama được sinh ra, chạy từ đầu đến cuối trên các kernel của khóa học và được kiểm tra từng token
            một, và M9 ghép được với nó: một adapter có sẵn đặt <code>launchQLinear</code> của lab X2.4 vào vị trí lớp tuyến tính
            của M10, để cùng bước Llama đó chạy với trọng số Q8_0 và Q4_0. Các file GGUF dành cho llama.cpp lưu tensor của chúng theo các
            định dạng khối mà giờ bạn đã đọc được (x2-facts D1).
          </p>
        }
      >
        <p>
          Next in this course's order come chapters 7 and 19 and appendix C, then{' '}
          <a href={href('x3')}>module X3, Reading real backends</a>. X3 finds the kernels behind an op in llama.cpp,
          MLX and PyTorch, including the quantised matrix-vector kernels this module rebuilt. Its milestone M10 is a
          generated Llama-style model run end to end on the course's kernels and checked token by token, and M9 composes with
          it: a provided adapter puts lab X2.4's <code>launchQLinear</code> in M10's linear-layer slot, so the same Llama step
          runs with Q8_0 and Q4_0 weights. GGUF files made for llama.cpp store their tensors in the block formats you can now
          read (x2-facts D1).
        </p>
      </Translated>
    </>
  );
}
