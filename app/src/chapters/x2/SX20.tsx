import { Fragment, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import { FORMATS, bitsPerWeight, codeBitsPerWeight, layer, gemv, modelBytes, type FormatId } from '../../sim/qtraffic';

// Module X2's start page: what this module is (ours, anchored on appendix A §A.5 and
// chapter 5's intensity), which kernel a quantised linear layer ends in today (a small
// classification table, data only), the size of the problem (a Predict whose reveal is
// computed by qtraffic.ts), the plan, and six prerequisite questions from earlier chapters.

const FACTS = 'notes/today/x2-facts.md';
const PARAMS = 7e9;
const gb = (b: number) => (b / 1e9).toFixed(2);

export const questionsSX20: Record<string, Question> = {
  'x2/start/q-gemv-intensity': {
    kind: 'choice',
    id: 'x2/start/q-gemv-intensity',
    prompt: (
      <Translated
        vi={
          <>
            Lớp tuyến tính của sách, 1024 đầu vào sang 4096 đầu ra, làm khoảng 1 FLOP mỗi byte ở batch 1 với định dạng 16 bit (
            <a href={href('ch05', '5.1')}>mục 5.1</a>, và chương 20, tr. 505). Với trọng số và activation float32 thì con số đó
            là bao nhiêu, và điều gì có thể nâng nó lên?
          </>
        }
      >
        <>
          The book's linear layer, 1024 inputs to 4096 outputs, does about 1 FLOP per byte at batch 1 in 16-bit (
          <a href={href('ch05', '5.1')}>section 5.1</a>, and chapter 20, p. 505). What is it with float32 weights and
          activations, and what could raise it?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Khoảng 0.5 FLOP/B; chỉ batching hoặc ít byte hơn cho mỗi trọng số mới nâng được nó.">About 0.5 FLOP/B; only batching or fewer bytes per weight raise it.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Khoảng 0.5 FLOP/B; một ô bộ nhớ chia sẻ đủ lớn sẽ nâng nó lên.">About 0.5 FLOP/B; a large enough shared-memory tile raises it.</Translated>, why: <Translated vi="Phân ô loại bỏ những lần nạp lặp lại. Ở batch 1 không trọng số nào được nạp hai lần: không có gì để chia sẻ.">Tiling removes repeated fetches. At batch 1 no weight is fetched twice: there is nothing to share.</Translated> },
      { label: <Translated vi="Khoảng 2 FLOP/B: gấp đôi số byte nghĩa là gấp đôi khối lượng việc mỗi byte.">About 2 FLOP/B: twice the bytes means twice the work per byte.</Translated>, why: <Translated vi="Khối lượng việc vẫn vậy; số byte tăng gấp đôi, nên tỉ lệ giảm một nửa.">The work is the same; the bytes double, so the ratio halves.</Translated> },
      { label: <Translated vi="Khoảng 1 FLOP/B: định dạng số không làm thay đổi tỉ lệ.">About 1 FLOP/B: the number format does not change the ratio.</Translated>, why: <Translated vi="Tỉ lệ là số phép toán trên mỗi byte, và float32 di chuyển gấp đôi số byte cho cùng số phép toán.">The ratio is operations per byte, and float32 moves twice the bytes for the same operations.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            2 × 1024 × 4096 phép toán trên khoảng 4 × 1024 × 4096 byte trọng số là 0.5 FLOP/B, cộng thêm một chút cho vector
            đầu vào và đầu ra. Mỗi trọng số tham gia một phép nhân và một phép cộng cho mỗi token, nên chỉ có thêm hàng cho mỗi
            trọng số (batching) hoặc ít byte hơn cho mỗi trọng số mới làm tỉ lệ thay đổi. Đó là lý do module này tồn tại: định
            dạng là một trong hai núm vặn còn lại.
          </p>
        }
      >
        <p>
          2 × 1024 × 4096 operations over about 4 × 1024 × 4096 bytes of weights is 0.5 FLOP/B, plus a sliver for the
          input and output vectors. Every weight takes part in one multiply and one add per token, so only more rows per
          weight (batching) or fewer bytes per weight move the ratio. That is why this module
          exists: the format is one of the two knobs left.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.1, pp. 93–97; ch. 20 §20.6, p. 505', // i18n-ignore: a citation
  },
  'x2/start/q-gather': {
    kind: 'choice',
    id: 'x2/start/q-gather',
    prompt: (
      <Translated
        vi={
          <>
            Trong GEMV kiểu gather của lab 22.1 (<a href={href('ch22', 'labs')}>lab 22.1</a>), luồng nào ghi Y[b][j], và vì sao
            nó không cần phép toán nguyên tử (atomic)?
          </>
        }
      >
        <>
          In lab 22.1's gather GEMV (<a href={href('ch22', 'labs')}>lab 22.1</a>), which thread writes Y[b][j], and why does
          it need no atomic?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Lane 0 của warp duy nhất sở hữu đầu ra j, sau phép rút gọn bằng shuffle của nó.">Lane 0 of the one warp that owns output j, after its shuffle reduction.</Translated>, correct: true },
      { label: <Translated vi="Mọi lane của warp, mỗi lane với tổng riêng phần của nó, nên lần ghi là atomic.">Every lane of the warp, each with its partial sum, so the write is atomic.</Translated>, why: <Translated vi="Đó là split-K, phiên bản scatter của cùng lab đó. Phiên bản gather rút gọn trước.">That is split-K, the scatter version of the same lab. The gather version reduces first.</Translated> },
      { label: <Translated vi="Luồng j của lưới, tự mình lặp qua cả hàng.">Thread j of the grid, which loops over the whole row on its own.</Translated>, why: <Translated vi="Đó là một luồng cho mỗi đầu ra. Lab 22.1 giao cho mỗi đầu ra một warp để các lần nạp hàng được hợp nhất.">That is one thread per output. Lab 22.1 gives each output a warp so that the row loads coalesce.</Translated> },
      { label: <Translated vi="Warp cuối cùng của khối, sau khi một rào chắn gom các tổng riêng phần.">The last warp of the block, after a barrier collects the partials.</Translated>, why: <Translated vi="Không có bước nào giữa các warp: một warp sở hữu một đầu ra.">There is no cross-warp step: one warp owns one output.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một warp cho mỗi hàng đầu ra là một gather: các lane của warp đọc hàng cùng nhịp, một phép rút gọn bằng shuffle gom
            các tổng riêng phần của chúng về lane 0, và lane 0 ghi. Mục X2.4 giữ nguyên hình dạng này và chỉ đổi thứ mà một lane nạp.
          </p>
        }
      >
        <p>
          One warp per output row is a gather: the warp's lanes read the row in step, a shuffle reduction collects their
          partial sums in lane 0, and lane 0 writes. Section X2.4 keeps this shape and changes only what a lane loads.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.2, pp. 532–533; lab 22.1 README', // i18n-ignore: a citation
  },
  'x2/start/q-coalesce': {
    kind: 'choice',
    id: 'x2/start/q-coalesce',
    prompt: (
      <Translated
        vi={
          <>
            W được lưu dạng (out, in) theo thứ tự hàng (row-major) và lane L của warp cho đầu ra j đọc W[j·in + k0 + L]. Các lần
            nạp đó có được hợp nhất không? Còn khi W lưu dạng (in, out), lane L đọc W[(k0 + L)·out + j] thì sao? (<a href={href('ch06', '6.1')}>mục 6.1</a>)
          </>
        }
      >
        <>
          W is stored (out, in) row-major and lane L of the warp for output j reads W[j·in + k0 + L]. Are those loads
          coalesced? And with W stored (in, out), lane L reading W[(k0 + L)·out + j]? (<a href={href('ch06', '6.1')}>section 6.1</a>)
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Có với (out, in); không với (in, out), khi các lane cách nhau out số float.">Yes for (out, in); no for (in, out), where lanes are out floats apart.</Translated>, correct: true },
      { label: <Translated vi="Cả hai đều được hợp nhất, vì mỗi lane đọc đúng một số float.">Both are coalesced, because every lane reads exactly one float.</Translated>, why: <Translated vi="Hợp nhất truy cập liên quan đến địa chỉ của các lane kề nhau, không phải kích thước của mỗi lần đọc.">Coalescing is about the addresses of neighbouring lanes, not the size of each read.</Translated> },
      { label: <Translated vi="Không cái nào: mỗi trọng số chỉ được đọc một lần, nên không có gì để gộp.">Neither: each weight is read only once, so nothing can be combined.</Translated>, why: <Translated vi="Việc gộp diễn ra giữa các lane của một lệnh nạp, không phải theo thời gian.">Combining happens across the lanes of one load instruction, not across time.</Translated> },
      { label: <Translated vi="Không với (out, in), khi các lane nhảy qua một hàng; có với (in, out), nằm cạnh nhau.">No for (out, in), where lanes skip a row; yes for (in, out), side by side.</Translated>, why: <Translated vi="Ngược lại mới đúng. Với (out, in), lane L + 1 đọc số float ngay sau lane L; với (in, out), mỗi lane nhảy cả một hàng gồm out giá trị.">It is the other way round. In (out, in) lane L + 1 reads the float after lane L; in (in, out) the lanes jump a whole row of out values each.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Với (out, in), 32 lane của một warp chạm vào 128 byte liên tiếp, một dải segment liền nhau. Với (in, out), mỗi lane
            cách nhau một hàng và lane nào cũng cần segment riêng. llama.cpp và MLX đều lưu các hàng trọng số dạng (out, in), và
            mọi lab trong module này cũng vậy.
          </p>
        }
      >
        <p>
          With (out, in), the 32 lanes of a warp touch 128 consecutive bytes, one run of segments. With (in, out) each lane
          is a row apart and every lane needs its own segment. llama.cpp and MLX both store weight rows as (out, in), and so
          does every lab in this module.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.1, pp. 124–132; x2-facts project notes',
    vi: { source: 'ch. 6 §6.1, pp. 124–132; ghi chú dự án x2-facts' },
  },
  'x2/start/q-fp16-int': {
    kind: 'numeric',
    id: 'x2/start/q-fp16-int',
    prompt: (
      <Translated
        vi={
          <>
            FP16 có phần định trị 11 bit, tính cả bit ẩn (<a href={href('appA', 'A.2')}>mục A.2</a>). n lớn nhất bằng bao nhiêu
            để mọi số nguyên từ 0 đến n đều biểu diễn chính xác được trong FP16?
          </>
        }
      >
        <>
          FP16 has an 11-bit significand, counting the hidden bit (<a href={href('appA', 'A.2')}>section A.2</a>). Up to
          what n is every integer from 0 to n exactly representable in FP16?
        </>
      </Translated>
    ),
    answer: 2048,
    explain: (
      <Translated
        vi={
          <p>
            2¹¹ = 2048. Trên mức đó FP16 nhảy từng bước 2, rồi từng bước 4. Giá trị hữu hạn lớn nhất của nó là 65,504. Hãy nhớ cả
            hai con số: tổng của một khối tích int8 lớn hơn cả hai rất nhiều, và đó là chủ đề của mục X2.6.
          </p>
        }
      >
        <p>
          2¹¹ = 2048. Above that FP16 steps by 2, then by 4. Its largest finite value is 65,504. Keep both numbers: a block
          of int8 products sums to far more than either, which is the subject of section X2.6.
        </p>
      </Translated>
    ),
    source: 'appA §A.2, pp. 586–590; notes/today/appA-facts.md A2; x2-facts E3', // i18n-ignore: a citation
  },
  'x2/start/q-acc-order': {
    kind: 'choice',
    id: 'x2/start/q-acc-order',
    prompt: (
      <Translated
        vi={
          <>
            Vì sao cộng lần lượt 4096 tích lại mất mát nhiều hơn ở FP16 so với float32 (<a href={href('appA', 'A.6')}>mục A.6</a>)?
          </>
        }
      >
        <>
          Why does adding 4096 products one at a time lose more in FP16 than in float32 (<a href={href('appA', 'A.6')}>section A.6</a>)?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Các số hạng nhỏ biến mất khi cộng vào một tổng lớn, và FP16 đến mức đó sớm hơn nhiều.">Small addends vanish against a large sum, and FP16 gets there far sooner.</Translated>,
        correct: true,
      },
      { label: <Translated vi="FP16 bị tràn sau 2048 số hạng, số nguyên chính xác lớn nhất của nó; phần còn lại bị mất.">FP16 overflows after 2048 terms, its largest exact integer; the rest is lost.</Translated>, why: <Translated vi="2048 là nơi FP16 thôi giữ được mọi số nguyên, không phải nơi nó bị tràn: mức đó là 65,504, và nó phụ thuộc vào độ lớn của tổng, không phải số số hạng. Mất mát ở đây bắt đầu từ lâu trước khi có bất kỳ tràn số nào.">2048 is where FP16 stops holding every integer, not where it overflows: that is 65,504, and it depends on the size of the sum, not the number of terms. The loss here starts long before any overflow.</Translated> },
      { label: <Translated vi="FP16 làm tròn về phía 0, còn float32 làm tròn về giá trị gần nhất.">FP16 rounds toward zero, while float32 rounds to the nearest value.</Translated>, why: <Translated vi="Mặc định cả hai đều làm tròn về giá trị gần nhất. Khác biệt là mỗi bên giữ bao nhiêu bit.">Both round to nearest by default. The difference is how many bits each keeps.</Translated> },
      { label: <Translated vi="Không phải vậy: chỉ thứ tự các phép cộng là quan trọng.">It does not: the order of the additions is all that matters.</Translated>, why: <Translated vi="Thứ tự quan trọng ở cả hai. Định dạng hẹp hơn làm cùng một thứ tự trở nên tệ hơn.">Order matters in both. The narrower format makes the same order worse.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đây là toán hạng biến mất của sách: khi tổng tích lũy đã lớn, một số hạng nhỏ hơn nửa bước ở chữ số cuối của nó sẽ
            bị mất, và FP16 đến điểm đó sau ít số hạng hơn nhiều. Một bộ cộng dồn rộng hơn làm điểm bắt đầu đó đến muộn hơn. Mọi
            kernel lượng tử hóa trong module này cộng dồn bằng int32 hoặc float, không bao giờ bằng định dạng lưu trữ.
          </p>
        }
      >
        <p>
          This is the book's disappearing operand: once the running sum is large, an addend smaller than half its
          last-place step is lost, and FP16 reaches that point after far fewer terms. A wider
          accumulator delays the point where it starts. Every quantised kernel in this module accumulates in int32 or
          float, never in the stored format.
        </p>
      </Translated>
    ),
    source: 'appA §A.6, pp. 593–594', // i18n-ignore: a citation
  },
  'x2/start/q-warp-reduce': {
    kind: 'choice',
    id: 'x2/start/q-warp-reduce',
    prompt: (
      <Translated
        vi={
          <>
            Sau một phép rút gọn hình cánh bướm với <code>__shfl_xor_sync</code>, những lane nào giữ tổng của warp? Còn sau một
            phép rút gọn với <code>__shfl_down_sync</code> (<a href={href('ch10', '10.7')}>mục 10.7</a>)?
          </>
        }
      >
        <>
          After a butterfly reduction with <code>__shfl_xor_sync</code>, which lanes hold the warp's total? And after one
          with <code>__shfl_down_sync</code> (<a href={href('ch10', '10.7')}>section 10.7</a>)?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Mọi lane sau cánh bướm xor; chỉ lane 0 sau shuffle-down.">Every lane after the xor butterfly; only lane 0 after shuffle-down.</Translated>, correct: true },
      { label: <Translated vi="Chỉ lane 0 trong cả hai trường hợp, vì tổng kết thúc ở lane 0.">Only lane 0 in both cases, since lane 0 is where the sum ends.</Translated>, why: <Translated vi="Trong cánh bướm, mỗi lane trao đổi với một lane cặp và cả hai đều giữ tổng, nên mọi lane cuối cùng đều có nó.">In a butterfly each lane exchanges with a partner and both keep the sum, so all lanes end with it.</Translated> },
      { label: <Translated vi="Mọi lane trong cả hai: ở cả hai kiểu shuffle, mỗi lane đều lấy giá trị của một lane cặp.">Every lane in both: in either shuffle, each lane takes a partner's value.</Translated>, why: <Translated vi="Với shuffle-down, một lane có nguồn nằm quá cuối warp sẽ nhận lại chính giá trị của nó, nên các lane phía trên kết thúc với tổng riêng phần hoặc tổng bị nhân đôi; chỉ lane 0 chắc chắn giữ tổng.">With shuffle-down a lane whose source would be past the end of the warp gets its own value back, so the upper lanes end with partial or doubled sums; only lane 0 is sure to hold the total.</Translated> },
      { label: <Translated vi="Lane cuối cùng sau cánh bướm; lane 0 sau shuffle-down.">The last lane after the butterfly; lane 0 after shuffle-down.</Translated>, why: <Translated vi="Cánh bướm có tính đối xứng: không lane nào đặc biệt.">The butterfly is symmetric: no lane is special.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lab 22.1 dùng shuffle-down và để lane 0 ghi. <code>simd_sum</code> của Metal đưa tổng cho mọi lane đang hoạt động.
            Một GEMV chỉ cần một luồng ghi có thể dùng cách nào cũng được.
          </p>
        }
      >
        <p>
          Lab 22.1 uses shuffle-down and lets lane 0 write. Metal's <code>simd_sum</code> gives the sum to every active lane.
          A GEMV that only needs one writer can use either.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.7, pp. 237–241; notes/today/ch10-facts.md (table "Which lanes hold the warp result")',
    vi: { source: 'ch. 10 §10.7, pp. 237–241; notes/today/ch10-facts.md (bảng "Which lanes hold the warp result")' },
  },
};

// ---------------------------------------------------------------- what you run today
type Family = 'int4pack' | 'refused' | 'mlx-qmv' | 'mmvq' | 'mulmv';
const FAMILIES: { id: Family; label: string }[] = [
  { id: 'int4pack', get label() { return tr("PyTorch's weight-only int4 core op, _weight_int4pack_mm", 'op lõi int4 chỉ-trọng-số của PyTorch, _weight_int4pack_mm'); } },
  { id: 'refused', get label() { return tr('refused: no kernel serves it', 'bị từ chối: không kernel nào phục vụ nó'); } },
  { id: 'mlx-qmv', get label() { return tr("MLX's quantised matrix-vector kernels (affine_qmv_fast and its siblings)", 'các kernel ma trận-vector lượng tử hóa của MLX (affine_qmv_fast và các kernel anh em)'); } },
  { id: 'mmvq', get label() { return tr('mmvq: activations quantised to Q8_1, integer dot per block', 'mmvq: activation được lượng tử hóa về Q8_1, tích vô hướng số nguyên cho mỗi khối'); } },
  { id: 'mulmv', get label() { return tr('mul_mv: float multiply-adds on masked 4-bit codes', 'mul_mv: phép nhân-cộng số thực trên các mã 4 bit đã lọc bằng mặt nạ'); } },
];
interface Call {
  id: string;
  call: string;
  answer: Family;
  why: string;
}
// Data only: what each call ends in, with the facts items that say so.
const CALLS: Call[] = [
  {
    id: 'torch-int4',
    get call() {
      return tr(
        'PyTorch, int4 weights in groups of 128 with a BF16 scale and zero each, batch 1, on CUDA',
        'PyTorch, trọng số int4 theo nhóm 128 với mỗi nhóm một hệ số tỉ lệ và một điểm không BF16, batch 1, trên CUDA',
      );
    },
    answer: 'int4pack',
    get why() {
      return tr(
        "That is the rule of the core op _weight_int4pack_mm: its CUDA kernel dequantises each 4-bit value as q × scale + zero in BF16, and needs compute capability 8.0 or a ROCm build. Whether a torchao config such as Int4WeightOnlyConfig reaches this op depends on its packing format, which the facts pass did not check (D6).",
        'Đó là quy tắc của op lõi _weight_int4pack_mm: kernel CUDA của nó giải lượng tử hóa mỗi giá trị 4 bit thành q × scale + zero ở BF16, và cần compute capability 8.0 hoặc một bản build ROCm. Một cấu hình torchao như Int4WeightOnlyConfig có đến được op này hay không phụ thuộc vào định dạng đóng gói của nó, điều mà lượt kiểm tra dữ kiện chưa xác minh (D6).',
      );
    },
  },
  {
    id: 'int-mm',
    get call() {
      return tr(
        'torch._int_mm with M = 1, on CUDA',
        'torch._int_mm với M = 1, trên CUDA',
      );
    },
    answer: 'refused',
    get why() {
      return tr(
        'The CUDA implementation checks that the first dimension is greater than 16 and stops otherwise. The int8 GEMM is not a decode kernel (A6).',
        'Bản cài đặt CUDA kiểm tra chiều thứ nhất có lớn hơn 16 không và dừng nếu không. GEMM int8 không phải là một kernel decode (A6).',
      );
    },
  },
  {
    id: 'mlx',
    get call() {
      return tr(
        'mlx.nn.QuantizedLinear on a Mac',
        'mlx.nn.QuantizedLinear trên máy Mac',
      );
    },
    answer: 'mlx-qmv',
    get why() {
      return tr(
        "Defaults: group 64, 4 bits, affine. For one row its Metal kernels are the quantised matrix-vector family; affine_qmv_fast is the one read in the facts pass. Which sibling runs is chosen by shape, which the facts pass did not check (D5; infra-facts D3).",
        'Mặc định: nhóm 64, 4 bit, affine. Với một hàng, các kernel Metal của nó thuộc họ ma trận-vector lượng tử hóa; affine_qmv_fast là kernel được đọc trong lượt kiểm tra dữ kiện. Kernel anh em nào chạy được chọn theo hình dạng, điều mà lượt kiểm tra dữ kiện chưa xác minh (D5; infra-facts D3).',
      );
    },
  },
  {
    id: 'llama-cuda',
    get call() {
      return tr(
        'llama.cpp, a Q4_0 tensor, on CUDA',
        'llama.cpp, một tensor Q4_0, trên CUDA',
      );
    },
    answer: 'mmvq',
    get why() {
      return tr(
        'Up to 8 tokens llama.cpp uses matrix-vector kernels. mmvq first quantises the float activations to Q8_1, then does integer dot products per block and one float rescale per block (ch15-facts D1; D3).',
        'Với tối đa 8 token, llama.cpp dùng các kernel ma trận-vector. mmvq trước hết lượng tử hóa các activation số thực về Q8_1, rồi tính tích vô hướng số nguyên cho mỗi khối và một lần nhân lại tỉ lệ bằng số thực cho mỗi khối (ch15-facts D1; D3).',
      );
    },
  },
  {
    id: 'llama-metal',
    get call() {
      return tr(
        'llama.cpp, a Q4_0 tensor, on Metal',
        'llama.cpp, một tensor Q4_0, trên Metal',
      );
    },
    answer: 'mulmv',
    get why() {
      return tr(
        'Metal has no integer dot. llama.cpp masks the nibbles in place, divides the activations by 16, 256 or 4096 to make up for the missing shifts, and multiplies in float (D4; C1).',
        'Metal không có tích vô hướng số nguyên. llama.cpp lọc các nibble tại chỗ bằng mặt nạ, chia các activation cho 16, 256 hoặc 4096 để bù cho các phép dịch bit bị bỏ qua, và nhân bằng số thực (D4; C1).',
      );
    },
  },
];

function KernelClassify() {
  const [pick, setPick] = useState<Record<string, Family | ''>>({});
  const [checked, setChecked] = useState(false);
  const right = CALLS.filter((c) => pick[c.id] === c.answer).length;
  const all = CALLS.every((c) => pick[c.id]);
  return (
    <Bench
      title={tr('Which kernel serves this call?', 'Kernel nào phục vụ lời gọi này?')}
      task={<Translated vi="Với mỗi lời gọi, chọn họ kernel mà nó kết thúc ở đó lúc decode. Rồi kiểm tra.">For each call, pick the kernel family it ends in at decode time. Then check.</Translated>}
      foot={<Translated vi={<>Theo lượt kiểm tra dữ kiện ngày 2026-11-20 ({FACTS}, các mục A6, D3 đến D6). Các dự án này thay đổi thường xuyên; chữ cái của mục cho biết mỗi câu trả lời được đọc ở đâu.</>}><>From the facts pass of 2026-11-20 ({FACTS}, items A6, D3 to D6). These projects change often; the item letters say where each answer was read.</></Translated>}
    >
      <div style={{ overflowX: 'auto' }}>
        <table className="data" data-table="classify" style={{ margin: 0, background: 'var(--page)' }}>
          <thead>
            <tr>
              <th><Translated vi="Lời gọi">Call</Translated></th>
              <th><Translated vi="Kết thúc ở">Ends in</Translated></th>
              {checked ? <th><Translated vi="Kết quả">Verdict</Translated></th> : null}
            </tr>
          </thead>
          <tbody>
            {CALLS.map((c) => {
              const ok = pick[c.id] === c.answer;
              return (
                <Fragment key={c.id}>
                  <tr data-call={c.id} data-verdict={checked ? (ok ? 'right' : 'wrong') : ''}>
                    <td style={{ minWidth: '11rem' }}>{c.call}</td>
                    <td>
                      <select
                        aria-label={tr(`Kernel for ${c.call}`, `Kernel cho ${c.call}`)}
                        value={pick[c.id] ?? ''}
                        style={{ maxWidth: '100%' }}
                        onChange={(e) => {
                          setPick({ ...pick, [c.id]: e.target.value as Family });
                          setChecked(false);
                        }}
                      >
                        <option value="">{tr('choose', 'chọn')}</option>
                        {FAMILIES.map((f) => (
                          <option key={f.id} value={f.id}>{f.label}</option>
                        ))}
                      </select>
                    </td>
                    {checked ? <td>{ok ? <span className="chip ok"><Translated vi="đúng">right</Translated></span> : <span className="chip fail"><Translated vi="không phải">not this one</Translated></span>}</td> : null}
                  </tr>
                  {checked ? (
                    <tr data-why={c.id}>
                      <td colSpan={3} className="muted" style={{ whiteSpace: 'normal', paddingTop: 0 }}>
                        {ok ? '' : tr(`It is ${FAMILIES.find((f) => f.id === c.answer)!.label}. `, `Đó là ${FAMILIES.find((f) => f.id === c.answer)!.label}. `)}
                        {c.why}
                      </td>
                    </tr>
                  ) : null}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <button className="primary" disabled={!all} onClick={() => setChecked(true)}><Translated vi="Kiểm tra">Check</Translated></button>
        {checked ? <span data-score={right}><Translated vi={<>Đúng {right} trên {CALLS.length}.</>}>{right} of {CALLS.length} right.</Translated></span> : <span className="muted"><Translated vi="Chọn một đáp án cho mọi lời gọi để kiểm tra.">Pick one for every call to check.</Translated></span>}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the size of the problem
const SIZE_FORMATS: FormatId[] = ['f32', 'f16', 'q8_0', 'q4_0'];

function SizeTable() {
  return (
    <table className="data" data-table="sizes" style={{ margin: '0.6rem 0 0' }}>
      <thead>
        <tr>
          <th><Translated vi="Định dạng">Format</Translated></th>
          <th className="num"><Translated vi="Bit mỗi trọng số">Bits per weight</Translated></th>
          <th className="num"><Translated vi="trong đó hệ số tỉ lệ">of which scales</Translated></th>
          <th className="num"><Translated vi="7 tỉ trọng số, GB">7 billion weights, GB</Translated></th>
        </tr>
      </thead>
      <tbody>
        {SIZE_FORMATS.map((f) => (
          <tr key={f} data-format={f}>
            <td>{FORMATS[f].label}{f.startsWith('q') ? <span className="muted"> ({FORMATS[f].owner})</span> : null}</td>
            <td className="num" data-bits>{bitsPerWeight(f)}</td>
            <td className="num">{(bitsPerWeight(f) - codeBitsPerWeight(f)).toString()}</td>
            <td className="num" data-gb>{gb(modelBytes(PARAMS, f))}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function SX20() {
  const q = questionsSX20;
  const q4 = modelBytes(PARAMS, 'q4_0');
  const bookLayer = gemv('f16', layer('1024x4096'), 1, 2);
  return (
    <>
      <Translated
        vi={
          <p>
            Ở batch 1, mỗi lớp tuyến tính của một decoder là một phép nhân ma trận-vector, và mỗi trọng số của mô hình đi qua
            bus bộ nhớ một lần cho mỗi token được sinh ra. Bạn đã gặp điều này hai lần: trong roofline của chương 5 và trong bước
            decode của các chương 20 và 22. Module này vặn chiếc núm duy nhất mà các chương đó để yên: mỗi trọng số chiếm bao nhiêu byte.
          </p>
        }
      >
        <p>
          At batch 1, every linear layer of a decoder is a matrix-vector product, and every weight of the model crosses the
          memory bus once per generated token. You have met this twice: as the roofline of chapter 5 and as the decode step
          of chapters 20 and 22. This module turns the one knob those chapters left alone: how many bytes each weight takes.
        </p>
      </Translated>
      <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
        <Translated
          vi={
            <p>
              Sách không có lượng tử hóa số nguyên, hệ số tỉ lệ, điểm không hay kernel lượng tử hóa. X2 do chúng tôi xây dựng
              dựa trên các phương pháp của sách.
            </p>
          }
        >
          <p>
            The book has no integer quantisation, scales, zero points or quantised kernels. X2 is built by us on its
            methods.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Điểm neo trong sách là §A.5 của phụ lục A, hai đoạn văn: một định dạng nhỏ hơn di chuyển ít byte hơn cho mỗi phép
            toán, và vì thế nâng cường độ tính toán. Sách dừng ở đó. Chương 5 cung cấp phương pháp biến câu đó thành một dự đoán
            bạn có thể đo, còn các chương 10 và 22 cung cấp các phép rút gọn warp và GEMV kiểu gather mà module này dựa vào.
          </p>
        }
      >
        <p>
          The book's anchor is appendix A's §A.5, two paragraphs: a smaller format moves fewer bytes per operation, and so
          raises intensity. It stops there. Chapter 5 supplies the method that turns that sentence into a prediction you
          can measure, and chapters 10 and 22 the warp reductions and the gather GEMV this module builds on.
        </p>
      </Translated>
      <Source section="A.5" pages="592–593" />
      <Translated
        vi={
          <p>
            Mọi khẳng định ở đây về API của nhà cung cấp hay về một codebase đều được đọc trong lượt kiểm tra dữ kiện ngày
            2026-11-20 và trích dẫn mục của nó trong <code>{FACTS}</code>. Các dự án như llama.cpp thay đổi hằng tuần, nên chữ cái
            của mục cho bạn biết cần xem lại ở đâu.
          </p>
        }
      >
        <p>
          Every statement here about a vendor API or a codebase was read in the facts pass on 2026-11-20 and cites its item
          in <code>{FACTS}</code>. Projects like llama.cpp change weekly, so the item letters tell you where to look again.
        </p>
      </Translated>

      <h2><Translated vi="Những gì bạn đang chạy hôm nay">What you run today</Translated></h2>
      <Translated
        vi={
          <p>
            Có lẽ bạn đã từng lượng tử hóa một mô hình. Đây là nơi ba con đường phổ biến kết thúc trên GPU, với từng token một.
            Trong PyTorch, con đường đó giờ là torchao: <code>torch.ao.quantization</code> đã bị thay thế.
          </p>
        }
      >
        <p>
          You have probably quantised a model already. Here is where three common routes end on the GPU, for one token at a
          time. In PyTorch that route is torchao now: <code>torch.ao.quantization</code> is superseded.
        </p>
      </Translated>
      <Today checked={`2026-11-20 (${FACTS}, D6)`}>
        <Translated vi={<p>Trang tài liệu 2.15 của PyTorch vẫn ghi kế hoạch xóa "in 2.10". Hãy coi ngày đó là đã lỗi thời.</p>}>
          <p>PyTorch's 2.15 page still plans the deletion "in 2.10". Read that date as stale.</p>
        </Translated>
      </Today>
      <table className="data wide" data-table="today" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr>
            <th><Translated vi="Thứ bạn gọi">What you call</Translated></th>
            <th><Translated vi="Thứ chạy lúc decode">What runs at decode</Translated></th>
            <th><Translated vi="Dữ kiện">Facts</Translated></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi={<>torchao <code>Int4WeightOnlyConfig</code> (nhóm từ 32 đến 256, mặc định 128); <code>Int8WeightOnlyConfig</code></>}>torchao <code>Int4WeightOnlyConfig</code> (groups of 32 to 256, default 128); <code>Int8WeightOnlyConfig</code></Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    Các op lõi chỉ-trọng-số của PyTorch là <code>_weight_int4pack_mm</code> (CUDA với compute capability 8.0 hoặc một
                    bản build ROCm; MPS) và <code>_weight_int8pack_mm</code> (CPU, CUDA, MPS). Một cấu hình torchao đến được kernel nào
                    phụ thuộc vào định dạng đóng gói của nó: chưa kiểm tra
                  </>
                }
              >
                PyTorch's weight-only core ops are <code>_weight_int4pack_mm</code> (CUDA with compute capability 8.0 or a
                ROCm build; MPS) and <code>_weight_int8pack_mm</code> (CPU, CUDA, MPS). Which kernel a torchao config
                reaches depends on its packing format: not checked
              </Translated>
            </td>
            <td>D6</td>
          </tr>
          <tr>
            <td><Translated vi={<><code>mlx.nn.QuantizedLinear</code> (mặc định: nhóm 64, 4 bit, affine)</>}><code>mlx.nn.QuantizedLinear</code> (defaults: group 64, 4 bits, affine)</Translated></td>
            <td><Translated vi={<>một kernel ma trận-vector lượng tử hóa trên Metal, chẳng hạn <code>affine_qmv_fast</code></>}>a quantised matrix-vector kernel on Metal, such as <code>affine_qmv_fast</code></Translated></td>
            <td>D5</td>
          </tr>
          <tr>
            <td><Translated vi="llama.cpp với một tensor GGUF Q4_0 hoặc Q8_0">llama.cpp with a GGUF Q4_0 or Q8_0 tensor</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    CUDA và HIP: <code>mmvq</code> cho tối đa 8 token, sau khi lượng tử hóa các activation về Q8_1. Metal:{' '}
                    <code>mul_mv</code>, bằng số thực
                  </>
                }
              >
                CUDA and HIP: <code>mmvq</code> for up to 8 tokens, after quantising the activations to Q8_1. Metal:{' '}
                <code>mul_mv</code>, in float
              </Translated>
            </td>
            <td>D3, D4; ch15-facts D1{/* i18n-ignore: fact item ids */}</td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Đây là bốn kernel khác nhau cho cùng một ý tưởng. Hãy phân loại các lời gọi dưới đây trước khi đọc tiếp; phản hồi cho
            mỗi lời gọi là bản xem trước của một trang trong module này.
          </p>
        }
      >
        <p>
          These are four different kernels for one idea. Sort the calls below before reading on; the feedback for each one
          is a preview of a page of this module.
        </p>
      </Translated>
      <KernelClassify />

      <h2><Translated vi="Quy mô của vấn đề">The size of the problem</Translated></h2>
      <Translated
        vi={
          <p>
            Sách đưa ra một con số: một mô hình 7 tỉ tham số ở định dạng 16 bit chiếm khoảng 14 GB. Decode ở batch 1 đọc toàn bộ
            số đó cho mỗi token. Chính lớp của sách, 1024 sang 4096 ở 16 bit, làm{' '}
            {bookLayer.intensity.toFixed(3)} FLOP mỗi byte ở batch 1.
          </p>
        }
      >
        <p>
          The book puts a number on it: a model of 7 billion parameters in a 16-bit format takes about 14 GB. Decode at
          batch 1 reads all of it for every token. The book's own layer, 1024 to 4096 in 16-bit, does{' '}
          {bookLayer.intensity.toFixed(3)} FLOP per byte at batch 1.
        </p>
      </Translated>
      <Source section="20.6" pages="505–506" />
      <Predict
        id="x2/start/p-q4-gb"
        unit="GB"
        prompt={
          <Translated
            vi={
              <>
                Cũng 7 tỉ trọng số đó lưu dưới dạng Q4_0 của llama.cpp: mã 4 bit trong các khối 32, mỗi khối có một hệ số
                tỉ lệ 16 bit. Bao nhiêu GB?
              </>
            }
          >
            <>
              The same 7 billion weights stored as llama.cpp's Q4_0: 4-bit codes in blocks of 32, each block with one 16-bit
              scale. How many GB?
            </>
          </Translated>
        }
      >
        {(c) => (
          <div data-reveal="sizes">
            <Translated
              vi={
                <p>
                  <b>{gb(q4)} GB</b>
                  {Math.abs(c - q4 / 1e9) < 0.05 ? ', đúng bằng con số của bạn' : Math.abs(c - 3.5) < 0.05 ? ': 3.5 GB chỉ là phần mã' : ''}. Mỗi khối
                  gồm 16 byte mã và 2 byte hệ số tỉ lệ, 18 byte cho 32 trọng số, 4.5 bit mỗi trọng số. Cứ chín byte của một tệp Q4_0
                  thì có một byte là hệ số tỉ lệ, và bạn trả giá cho nó ở mỗi token.
                </p>
              }
            >
              <p>
                <b>{gb(q4)} GB</b>
                {Math.abs(c - q4 / 1e9) < 0.05 ? ', which is your number' : Math.abs(c - 3.5) < 0.05 ? ': 3.5 GB is the codes alone' : ''}. Each block
                is 16 bytes of codes and 2 bytes of scale, 18 bytes for 32 weights, 4.5 bits per weight. One byte in nine of
                a Q4_0 file is scale, and you pay for it on every token.
              </p>
            </Translated>
            <SizeTable />
            <Translated
              vi={
                <p style={{ marginTop: '0.6rem' }}>
                  Q8_0 là 8.5 bit, không phải 8, vì cùng lý do đó. Mục <a href={href('x2', 'X2.1')}>X2.1</a> biến các byte này thành
                  một mức tăng tốc dự đoán, và cho thấy một bước giải lượng tử hóa riêng làm gì với nó.
                </p>
              }
            >
              <p style={{ marginTop: '0.6rem' }}>
                Q8_0 is 8.5 bits, not 8, for the same reason. Section <a href={href('x2', 'X2.1')}>X2.1</a> turns these bytes
                into a predicted speed-up, and shows what a separate dequantise step does to it.
              </p>
            </Translated>
          </div>
        )}
      </Predict>

      <h2><Translated vi="Kế hoạch">The plan</Translated></h2>
      <Translated
        vi={
          <p>
            Bảy trang. <a href={href('x2', 'X2.1')}>X2.1</a> đếm byte và dự đoán tốc độ.{' '}
            <a href={href('x2', 'X2.2')}>X2.2</a> xây dựng hệ số tỉ lệ, điểm không và khối, còn <a href={href('x2', 'X2.3')}>X2.3</a>{' '}
            đóng gói hai mã mỗi byte. <a href={href('x2', 'X2.4')}>X2.4</a> ánh xạ một warp lên các hàng lượng tử hóa.{' '}
            <a href={href('x2', 'X2.5')}>X2.5</a> so sánh giải lượng tử hóa về số thực với nhân bằng số nguyên, và{' '}
            <a href={href('x2', 'X2.6')}>X2.6</a> chỉ ra nơi tổng phải giữ độ rộng lớn. <a href={href('x2', 'X2.7')}>X2.7</a> ghép
            tất cả lại thành một bước decode.
          </p>
        }
      >
        <p>
          Seven pages. <a href={href('x2', 'X2.1')}>X2.1</a> counts bytes and predicts speed.{' '}
          <a href={href('x2', 'X2.2')}>X2.2</a> builds scales, zero points and blocks, and <a href={href('x2', 'X2.3')}>X2.3</a>{' '}
          packs two codes per byte. <a href={href('x2', 'X2.4')}>X2.4</a> maps a warp onto quantised rows.{' '}
          <a href={href('x2', 'X2.5')}>X2.5</a> compares dequantising to float with multiplying in integers, and{' '}
          <a href={href('x2', 'X2.6')}>X2.6</a> says where the sum must stay wide. <a href={href('x2', 'X2.7')}>X2.7</a> puts
          it together into a decode step.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Bốn lab, trên <a href={href('x2', 'labs')}>trang lab</a>, mỗi lab có một dự đoán trước khi bạn đo:
          </p>
        }
      >
        <p>
          Four labs, on the <a href={href('x2', 'labs')}>labs page</a>, each with a prediction before you measure:
        </p>
      </Translated>
      <ul>
        <Translated vi={<li><b>Lab X2.1</b>, một GEMV trọng số int8 cho decode ở batch 1.</li>}><li><b>Lab X2.1</b>, an int8-weight GEMV for batch-1 decode.</li></Translated>
        <Translated vi={<li><b>Lab X2.2</b>, một GEMV 4 bit trên các khối Q4_0 của llama.cpp, đọc đúng như tệp lưu chúng.</li>}><li><b>Lab X2.2</b>, a 4-bit GEMV on llama.cpp's Q4_0 blocks, read as the file stores them.</li></Translated>
        <Translated vi={<li><b>Lab X2.3</b>, một cuộc săn lỗi trên đường số nguyên.</li>}><li><b>Lab X2.3</b>, a bug hunt on the integer path.</li></Translated>
        <Translated vi={<li><b>Lab X2.4</b>, mốc dự án M9: bước decode lượng tử hóa, kèm báo cáo mức sử dụng băng thông.</li>}><li><b>Lab X2.4</b>, milestone M9: the quantised decode step, with its bandwidth utilisation reported.</li></Translated>
      </ul>

      <h2><Translated vi="Trước khi bắt đầu">Before you start</Translated></h2>
      <Translated
        vi={
          <p>
            Sáu câu hỏi về những gì module này giả định từ các chương 5, 6, 10, 20 và 22 và phụ lục A. Mỗi câu liên kết tới mục
            mà nó xuất phát, và mỗi câu sẽ quay lại trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Six questions on what this module assumes from chapters 5, 6, 10, 20 and 22 and appendix A. Each links to the
          section it comes from, and each comes back in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={q['x2/start/q-gemv-intensity']} label={tr('From chapters 5 and 20', 'Từ các chương 5 và 20')} />
      <Ask q={q['x2/start/q-gather']} label={tr('From chapter 22', 'Từ chương 22')} />
      <Ask q={q['x2/start/q-coalesce']} label={tr('From chapter 6', 'Từ chương 6')} />
      <Ask q={q['x2/start/q-fp16-int']} label={tr('From appendix A', 'Từ phụ lục A')} />
      <Ask q={q['x2/start/q-acc-order']} label={tr('From appendix A', 'Từ phụ lục A')} />
      <Ask q={q['x2/start/q-warp-reduce']} label={tr('From chapter 10', 'Từ chương 10')} />
    </>
  );
}
