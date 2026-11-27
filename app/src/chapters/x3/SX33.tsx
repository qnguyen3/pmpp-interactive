import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Aside, Beyond, Bench, Simplification, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  EXP_LABEL,
  KERNELS,
  N_READS,
  PATH_LABEL,
  ROW16,
  STRUCTURE_LABEL,
  maxUlps,
  predictionOk,
  predictionSteps,
  rowBytes,
  runPath,
  ulps,
  type ExpFn,
  type PathId,
  type RowStep,
  type Structure,
} from '../../sim/rowreduce';
import { functionSource } from '../../sim/toolchain';
import type { Excerpt as ExcerptData } from '../../sim/x3sources';
import { Excerpt } from './Excerpt';
import { TwoSoftmaxesScene } from './TwoSoftmaxesScene';

const FACTS = 'notes/today/x3-facts.md';
const LAB = 'ch10/lab3_softmax_argmax/solution/kernel.cu';
// Quotations from llama.cpp and MLX, copied from notes/today/x3-facts.md B2 and B3 at the
// pinned commits and shown with the module's Excerpt (file, lines, permalink, licence). A line
// that is exactly "..." marks lines left out. Not course code: not compiled here.
// excerpts-g2.test.ts checks every fragment against the facts file.
export const EXCERPTS_SX33: Record<string, ExcerptData> = {
  rms: {
    id: 'x33-rms-norm-f32', project: 'llama.cpp', path: 'ggml/src/ggml-cuda/norm.cu', lines: [127, 272], factsItem: 'B2', lang: 'cuda',
    code: `float tmp = 0.0f; // partial sum for thread in warp
for (int col = tid; col < ncols; col += block_size) {
    const float xi = ggml_cuda_cast<float>(xc[col]);
    tmp += xi * xi;
}
// sum up partial sums
tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
const float mean = tmp / ncols;
const float scale = rsqrtf(mean + eps);
...
dstc[col] = ggml_cuda_cast<T>(scale * xi * mulc[mul_col]);`,
  },
  online: {
    id: 'x33-mlx-online-softmax', project: 'mlx', path: 'mlx/backend/cuda/softmax.cu', lines: [49, 62], factsItem: 'B3', lang: 'cuda',
    code: `prevmax = maxval;
for (int i = 0; i < N_READS; ++i) {
  maxval = max_op(maxval, static_cast<AccT>(vals[i]));
}
// Online normalizer calculation for softmax:
// https://github.com/NVIDIA/online-softmax
normalizer = normalizer * softmax_exp(prevmax - maxval);
for (int i = 0; i < N_READS; i++) {
  normalizer = normalizer + softmax_exp(static_cast<AccT>(vals[i]) - maxval);
}`,
  },
};

// ---------------------------------------------------------------- questions
export const questionsSX33: Record<string, Question> = {
  'x3/X3.3/q-excerpt-rms': {
    kind: 'choice',
    id: 'x3/X3.3/q-excerpt-rms',
    prompt: (
      <Translated
        vi={
          <>
            Hãy đọc vòng lặp và dòng <code>block_reduce</code> của <code>rms_norm_f32</code> ở trên. Đó là kernel nào của khóa học,
            và phiên bản của llama.cpp thêm gì?
          </>
        }
      >
        <>
          Read the loop and the <code>block_reduce</code> line of <code>rms_norm_f32</code> above. Which course kernel is it,
          and what does llama.cpp's version add?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="blockReduceSum của lab 10.3 được nạp x bình phương thay vì x; phép nhân với trọng số được gộp vào cùng kernel.">
            Lab 10.3's blockReduceSum fed with x squared instead of x; the weight multiply is fused into the same kernel.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="rowMaxKernel của lab 10.3: nó giữ bình phương lớn nhất.">Lab 10.3's rowMaxKernel: it keeps the largest square.</Translated>
        ),
        why: (
          <Translated vi="Vòng lặp cộng dồn, tmp += xi * xi, và phương pháp rút gọn là SUM.">
            The loop adds, tmp += xi * xi, and the reduction method is SUM.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một phép rút gọn trên toàn lưới: các khối ghi tổng riêng phần của mình bằng một phép toán nguyên tử (atomic), như kernel nhiều khối của chương 10.">
            A grid-wide reduction: blocks commit their partial sums with an atomic, as in chapter 10's multi-block kernel.
          </Translated>
        ),
        why: (
          <Translated vi="Một khối sở hữu một hàng (const int row = blockIdx.x), nên kết quả cấp khối là kết quả của hàng. Không cần atomic.">
            One block owns one row (const int row = blockIdx.x), so the block-level result is the row's result. No atomic is needed.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Histogram riêng tư hóa (privatization) của chương 9: mỗi luồng cộng vào một bộ đếm dùng chung.">
            Chapter 9's privatised histogram: each thread adds into a shared counter.
          </Translated>
        ),
        why: (
          <Translated vi="Ở đây không luồng nào cộng vào một bộ đếm dùng chung; bộ nhớ chia sẻ chỉ giữ các tổng riêng phần theo warp.">
            No thread adds into a shared counter here; shared memory only holds the per-warp partials.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một vòng lặp có bước nhảy qua hàng (gộp việc cho luồng, chương 10), một phép rút gọn trong warp, các tổng riêng phần theo
            warp trong bộ nhớ chia sẻ (shared memory), và giai đoạn thứ hai đọc chúng với kiểm tra biên và đệm bằng phần tử đơn vị, 0
            cho phép cộng. Hình 10.16 của sách không có kiểm tra biên; lab 10.3 và llama.cpp đều thêm nó (ch10 facts D1). Điều
            llama.cpp thêm vào là phần kết: căn bậc hai nghịch đảo, và phép nhân với trọng số ngay trong cùng kernel, nên hàng đã
            chuẩn hóa không bao giờ bị ghi ra rồi đọc lại.
          </p>
        }
      >
        <p>
          A strided loop over the row (coarsening, chapter 10), a warp reduction, the per-warp partials in shared memory, and a
          second stage that reads them with a guard and pads with the identity, 0 for a sum. The book's Fig. 10.16 has no
          guard; lab 10.3 and llama.cpp both add it (ch10 facts D1). What llama.cpp adds is the epilogue: the reciprocal square
          root, and the multiply by the weight in the same kernel, so the normalised row is never written and read back.
        </p>
      </Translated>
    ),
    source: `${FACTS} B2; notes/today/ch10-facts.md D1; lab 10.3 solution`,
    vi: { source: `${FACTS} B2; notes/today/ch10-facts.md D1; lời giải tham chiếu lab 10.3` },
  },
  'x3/X3.3/q-float4-576': {
    kind: 'choice',
    id: 'x3/X3.3/q-float4-576',
    prompt: (
      <Translated
        vi={
          <>
            llama.cpp CUDA chạy RMSNorm trên các hàng 576 float của SmolLM2-135M, và đích được căn chỉnh 16 byte. Nó đi theo đường
            nào?
          </>
        }
      >
        <>
          llama.cpp CUDA runs RMSNorm on SmolLM2-135M's rows of 576 floats, and the destination is 16-byte aligned. Which path
          does it take?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Đường float4: 576 là bội của 4 và đích được căn chỉnh.">
            The float4 path: 576 is a multiple of 4 and the destination is aligned.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Đường vô hướng, vì các hàng dưới 1024 phần tử dùng kernel vô hướng với 256 luồng.">
            The scalar path, because rows under 1024 elements take the scalar kernel with 256 threads.
          </Translated>
        ),
        why: (
          <Translated vi="Ngưỡng 1024 chọn kích thước khối của đường vô hướng (256 hoặc 1024 luồng); nó không quyết định giữa hai đường.">
            The 1024 threshold picks the scalar path's block size (256 or 1024 threads); it is not what decides between the two paths.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Đường float4, bất kể độ dài hàng, vì đích được căn chỉnh.">
            The float4 path, whatever the row length, because the destination is aligned.
          </Translated>
        ),
        why: (
          <Translated vi="Cả hai điều kiện phải cùng đúng: ncols % 4 == 0 cũng như phép kiểm tra căn chỉnh.">
            Both conditions must hold: ncols % 4 == 0 as well as the alignment test.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Đường vô hướng, vì các lần nạp float4 chỉ dùng cho trọng số, không bao giờ cho hàng.">
            The scalar path, because the float4 loads are used only for the weight, never for the row.
          </Translated>
        ),
        why: (
          <Translated vi="Hàng cũng được nạp qua rms_norm_load4; con trỏ trọng số cũng được ép sang float4.">
            The row is loaded through rms_norm_load4 too; the weight pointer is cast to float4 as well.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đường vector chỉ được chọn khi <code>ncols % 4 == 0</code> và{' '}
            <code>((uintptr_t) dst % (4*sizeof(T))) == 0</code>. Giống MLX và PyTorch, llama.cpp làm cho phép ép kiểu con trỏ sang
            kiểu vector của sách trở nên an toàn bằng cách kiểm tra căn chỉnh lúc chạy; không nhà cung cấp nào ghi trong tài liệu về
            chính phép ép kiểu đó (infra A6). Một hàng rộng 577 sẽ đi theo đường vô hướng.
          </p>
        }
      >
        <p>
          The vector path is chosen only when <code>ncols % 4 == 0</code> and{' '}
          <code>((uintptr_t) dst % (4*sizeof(T))) == 0</code>. Like MLX and PyTorch, llama.cpp makes the book's pointer cast
          to a vector type safe by testing alignment at run time; no vendor documents the cast itself (infra A6). A 577-wide
          row would take the scalar path.
        </p>
      </Translated>
    ),
    source: `${FACTS} B2 (norm.cu line 473), D4 (hidden_size 576)`,
    vi: { source: `${FACTS} B2 (norm.cu dòng 473), D4 (hidden_size 576)` },
  },
  'x3/X3.3/q-hf-cast': {
    kind: 'choice',
    id: 'x3/X3.3/q-hf-cast',
    prompt: (
      <Translated
        vi={
          <>
            Tham chiếu CPU của bạn cho một checkpoint bfloat16 làm theo llama2.c: nó nhân hàng đã chuẩn hóa với trọng số trong
            float. <code>LlamaRMSNorm</code> của Hugging Face làm tròn hàng đã chuẩn hóa về bfloat16 trước, rồi mới nhân. Phát biểu
            nào đúng?
          </>
        }
      >
        <>
          Your CPU reference for a bfloat16 checkpoint follows llama2.c: it multiplies the normalised row by the weight in
          float. Hugging Face's <code>LlamaRMSNorm</code> rounds the normalised row to bfloat16 first, then multiplies. Which
          statement holds?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Kết quả khác nhau, vượt xa sai số làm tròn float32: một tham chiếu phải nêu rõ thứ tự mà nó làm theo.">
            Different results, far beyond float32 rounding: a reference must name the order it follows.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chúng giống nhau đến mức làm tròn float32, vì nhân với trọng số trước hay sau một phép ép kiểu đều giao hoán.">
            They are the same up to float32 rounding, because multiplying by the weight before or after a cast commutes.
          </Translated>
        ),
        why: (
          <Translated vi="Phép ép kiểu không phải phép nhân: Hugging Face làm tròn về bfloat16, 8 bit có nghĩa, trước phép nhân, và phép làm tròn đó không được hoàn tác.">
            A cast is not a multiplication: Hugging Face rounds to bfloat16, 8 significant bits, before the multiply, and that rounding is not undone.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Chúng chỉ khác nhau khi trọng số bằng 1.">They differ only when the weight is 1.</Translated>,
        why: (
          <Translated vi="Với trọng số bằng 1, đầu ra của Hugging Face là hàng đã làm tròn, vẫn khác với hàng float.">
            With a weight of 1 the Hugging Face output is the rounded row, which still differs from the float one.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Hugging Face sai, và một kernel tốt nên khớp với llama2.c.">
            Hugging Face is wrong, and a good kernel should match llama2.c.
          </Translated>
        ),
        why: (
          <Translated vi="Không cái nào sai. Một mô hình được huấn luyện hoặc kiểm định theo một trong hai, và đó là cái cần khớp.">
            Neither is wrong. A model was trained or validated against one of them, and that is the one to match.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>return self.weight * hidden_states.to(input_dtype)</code>: phép ép về kiểu đầu vào diễn ra trước khi nhân trọng
            số. bfloat16 giữ 8 bit phần định trị, nên riêng phép làm tròn đó đã dịch mỗi giá trị tới khoảng 0.4%, cao hơn nhiều so
            với 1e-7 của float32. Kiểm thử một kernel với cái sai trong hai cái trông giống như một lỗi trong kernel.
          </p>
        }
      >
        <p>
          <code>return self.weight * hidden_states.to(input_dtype)</code>: the cast to the input type happens before the
          weight. bfloat16 keeps 8 bits of significand, so that rounding alone moves each value by up to about 0.4%, far
          above float32's 1e-7. Testing a kernel against the wrong one of the two looks like a bug in the kernel.
        </p>
      </Translated>
    ),
    source: `${FACTS} B2 (modeling_llama.py lines 62 to 67; run.c lines 182 to 195); appA A2 (bfloat16 layout)`,
    vi: { source: `${FACTS} B2 (modeling_llama.py dòng 62 đến 67; run.c dòng 182 đến 195); appA A2 (bố cục bfloat16)` },
  },
  'x3/X3.3/q-26385': {
    kind: 'choice',
    id: 'x3/X3.3/q-26385',
    prompt: (
      <Translated
        vi={
          <>
            <code>soft_max_f32</code> của llama.cpp đặt một rào chắn giữa phép rút gọn max và phép rút gọn tổng, với chú thích
            "sync is needed as we reuse buf_iw across block_reduce invocations, see #26385". Vì sao nó cần thiết?
          </>
        }
      >
        <>
          llama.cpp's <code>soft_max_f32</code> puts a barrier between its max reduction and its sum reduction, with the
          comment "sync is needed as we reuse buf_iw across block_reduce invocations, see #26385". Why is it needed?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một buffer dùng chung phục vụ cả hai phép rút gọn: tổng riêng phần của một warp nhanh có thể ghi đè một giá trị max mà một warp chậm chưa đọc.">
            One shared buffer serves both reductions: a fast warp’s partial sum could overwrite a max a slow warp has not read.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="expf không an toàn luồng trên GPU, nên các luồng của một khối phải lần lượt tính hàm mũ.">
            expf is not thread-safe on the GPU, so the threads of a block must take turns computing exponentials.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi luồng tự tính các hàm mũ của mình trong thanh ghi; không có gì được dùng chung ở đó.">
            Each thread computes its own exponentials in registers; nothing is shared there.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Giá trị max phải tới bộ nhớ toàn cục, nơi mọi khối thấy được, trước khi bất kỳ khối nào bắt đầu tính tổng.">
            The max must reach global memory, where every block can see it, before any block starts on the sum.
          </Translated>
        ),
        why: (
          <Translated vi="Giá trị max không bao giờ ra bộ nhớ toàn cục; nó ở lại trong bộ nhớ chia sẻ và thanh ghi.">
            The max never goes to global memory; it stays in shared memory and registers.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không cần trên CUDA, chỉ cần trong bản build HIP của cùng tệp nguồn, nơi một warp có thể rộng 64 luồng.">
            It is not needed on CUDA, only in the HIP build of the same source file, where a warp can be 64 threads wide.
          </Translated>
        ),
        why: (
          <Translated vi="Đây là tệp CUDA, cũng chính là tệp HIP, và mối nguy là như nhau ở mọi độ rộng warp.">
            This is the CUDA file, which is also the HIP file, and the hazard is the same at any warp width.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Dùng lại một buffer cho hai phép rút gọn biến một lần đọc kết quả thứ nhất thành mối nguy ghi-sau-đọc với phép thứ hai.{' '}
            <code>softmaxRowsKernel</code> của lab 10.3 có cùng rào chắn vì cùng lý do (chú thích phía trên lượt tính tổng của nó), và{' '}
            <code>BlockReduceSum</code> của PyTorch bắt đầu bằng một rào chắn: "prevent races when BlockReduces are called in a
            row". Bỏ nó đi và một bài kiểm thử vẫn có thể qua trên GPU của bạn; các thứ tự luồng của trình giả lập sẽ bắt được lỗi.
          </p>
        }
      >
        <p>
          Reusing one buffer for two reductions turns a read of the first result into a write-after-read hazard with the
          second. Lab 10.3's <code>softmaxRowsKernel</code> has the same barrier for the same reason (the comment above its
          sum pass), and PyTorch's <code>BlockReduceSum</code> starts with one: "prevent races when BlockReduces are called
          in a row". Remove it and a test may still pass on your GPU; the emulator's thread orders catch it.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch10-facts.md D1, D4; lab 10.3 solution',
    vi: { source: 'notes/today/ch10-facts.md D1, D4; lời giải tham chiếu lab 10.3' },
  },
  'x3/X3.3/q-excerpt-online': {
    kind: 'choice',
    id: 'x3/X3.3/q-excerpt-online',
    prompt: (
      <Translated vi={<>Các dòng của MLX CUDA ở trên: đây là ý tưởng nào của khóa học, và nó khác softmax của mốc dự án M4 ra sao?</>}>
        <>MLX CUDA's lines above: which idea from the course is this, and how does it differ from milestone M4's softmax?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Phép cập nhật (m, D) của chương 20 trên một hàng: hai lần đọc hàng, chỗ M4 dùng ba.">
            Chapter 20's (m, D) update on one row: two reads of the row where M4 makes three.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Một phép cộng có bù (Kahan) từ phụ lục A, mang sai số làm tròn của mỗi phần sang phần tiếp theo.">
            A compensated (Kahan) sum from appendix A, carrying each chunk's rounding error into the next.
          </Translated>
        ),
        why: (
          <Translated vi="Thừa số exp(prevmax − maxval) co giãn lại tổng theo một giá trị max mới; nó không mang sai số làm tròn đi tiếp.">
            The factor exp(prevmax − maxval) rescales the sum to a new maximum; it does not carry a rounding error forward.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chính M4, với tên khác: max trên hàng trước, rồi tổng, rồi ghi.">
            M4 itself, with other names: the maximum over the row first, then the sum, then the write.
          </Translated>
        ),
        why: (
          <Translated vi="M4 tìm max của cả hàng trước khi tính bất kỳ hàm mũ nào. Ở đây các hàm mũ được tính theo một giá trị max vẫn có thể tăng.">
            M4 finds the maximum of the whole row before it computes a single exponential. Here exponentials are taken against a maximum that may still rise.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Phép trộn của flash-decoding gộp các kết quả riêng phần mà nhiều khối đã tính trên các phần của hàng.">
            Flash-decoding's merge of the partial results that several blocks computed over parts of the row.
          </Translated>
        ),
        why: (
          <Translated vi="Phép trộn đó dùng cùng đại số, nhưng giữa các khối. Ở đây một khối đọc một hàng.">
            That merge uses the same algebra, but across blocks. Here one block reads one row.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>normalizer * softmax_exp(prevmax - maxval)</code> là phép co giãn lại D khi giá trị max thay đổi (chương 20,
            Eq. 20.4 với một vế là một phần). Vì max và tổng được tìm trong cùng một lượt quét, lượt quét thứ hai duy nhất là lượt
            ghi. Cái giá là mỗi phần một hàm mũ cho phép co giãn, được tính ngay cả khi max không tăng (khi đó thừa số đúng bằng 1).
          </p>
        }
      >
        <p>
          <code>normalizer * softmax_exp(prevmax - maxval)</code> is the rescale of D when the maximum moves (chapter 20,
          Eq. 20.4 with one side a chunk). Because the maximum and the sum are found in the same sweep, the only second
          sweep is the write. The price is one exponential per chunk for the rescale, taken even when the maximum did not
          rise (the factor is then exactly 1).
        </p>
      </Translated>
    ),
    source: `${FACTS} B3 (softmax.cu lines 49 to 62); notes/ch20.md §20.5`,
    vi: { source: `${FACTS} B3 (softmax.cu dòng 49 đến 62); notes/ch20.md §20.5` },
  },
  'x3/X3.3/q-bytes': {
    kind: 'numeric',
    id: 'x3/X3.3/q-bytes',
    prompt: (
      <Translated
        vi={
          <>
            Một hàng gồm 32,000 logits float32. Softmax của lab 10.3 đọc nó ba lần và ghi một lần; kernel online của MLX CUDA đọc nó
            hai lần và ghi một lần. Số byte của kernel online bằng bao nhiêu phần của lab 10.3?
          </>
        }
      >
        <>
          A row of 32,000 float32 logits. Lab 10.3's softmax reads it three times and writes once; MLX CUDA's online kernel
          reads it twice and writes once. The online kernel's bytes, as a fraction of lab 10.3's?
        </>
      </Translated>
    ),
    answer: 0.75,
    tol: 0.01,
    explain: (
      <Translated
        vi={
          <p>
            32,000 × 4 byte = 128,000 byte mỗi lượt quét. Ba lần đọc và một lần ghi: {rowBytes(32000, 3, 1).toLocaleString('en-US')}{' '}
            byte. Hai lần đọc và một lần ghi: {rowBytes(32000, 2, 1).toLocaleString('en-US')}. Tỉ lệ là 3 / 4. Cả hai kernel đều bị
            giới hạn bởi bộ nhớ, nên khi số byte ít đi một phần tư, thời gian giảm được tối đa một phần tư (tăng tốc tối đa 4/3), và
            ít hơn nếu một bộ nhớ đệm (cache) vốn đã phục vụ các lần đọc lặp lại.
          </p>
        }
      >
        <p>
          32,000 × 4 bytes = 128,000 bytes per sweep. Three reads and a write: {rowBytes(32000, 3, 1).toLocaleString('en-US')}{' '}
          bytes. Two reads and a write: {rowBytes(32000, 2, 1).toLocaleString('en-US')}. The ratio is 3 / 4. Both kernels are
          memory-bound, so with a quarter fewer bytes the time can drop by at most a quarter (a speed-up of at most 4/3), and
          by less if a cache was serving the repeated reads anyway.
        </p>
      </Translated>
    ),
    source: 'ours; read counts from lab 10.3 and x3 facts B3',
    vi: { source: 'của chúng tôi; số lần đọc lấy từ lab 10.3 và x3 facts B3' },
  },
  'x3/X3.3/q-exp-choice': {
    kind: 'choice',
    id: 'x3/X3.3/q-exp-choice',
    prompt: (
      <Translated
        vi={
          <>
            Bạn so một softmax Metal với một tham chiếu CPU tính bằng double. Theo các dữ kiện, kernel của framework nào cần dung
            sai lỏng hơn?
          </>
        }
      >
        <>
          You compare a Metal softmax with a CPU reference computed in double. By the facts, which framework's kernel needs
          the looser tolerance?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Của MLX, vốn gọi fast::exp ở chỗ PyTorch MPS gọi precise::exp; và mọi kernel Metal đều cần một dung sai nào đó.">
            MLX's, which calls fast::exp where PyTorch MPS calls precise::exp; and any Metal kernel needs some tolerance.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Của PyTorch MPS, vì precise::exp là đường chậm hơn và các bước thêm của nó tự gây thêm sai số làm tròn.">
            PyTorch MPS's, because precise::exp is the slower path and its extra steps add rounding errors of their own.
          </Translated>
        ),
        why: (
          <Translated vi="Tốc độ không phải độ chính xác. Biến thể precise có cận chặt hơn.">
            Speed is not accuracy. The precise variant has the tighter bound.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không cái nào: cả hai chạy trên cùng GPU, nên chúng cho cùng kết quả.">
            Neither: both run on the same GPU, so they give the same results.
          </Translated>
        ),
        why: (
          <Translated vi="Cùng GPU chạy bất kỳ hàm mũ nào kernel yêu cầu, và hai kernel yêu cầu hai hàm khác nhau.">
            The same GPU runs whichever exponential the kernel asks for, and the two kernels ask for different ones.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không cái nào cần: softmax chia mọi số hạng cho cùng một tổng, nên sai số tương đối trong các hàm mũ triệt tiêu.">
            Neither needs one: softmax divides every term by the same sum, so a relative error in the exponentials cancels.
          </Translated>
        ),
        why: (
          <Translated vi="Sai số tương đối trong mỗi số hạng không triệt tiêu trừ khi mọi số hạng có cùng sai số đó.">
            A relative error in each term does not cancel unless every term has the same one.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đặc tả MSL giới hạn sai số của <code>exp</code> độ chính xác đơn bản precise ở 4 ulp, và của bản fast ở 3 + floor(|2x|) ulp, tăng theo
            độ lớn của đối số. Softmax giữ đối số ở mức 0 trở xuống, đó là lý do MLX nêu trong chú thích của nó, nhưng các đối số âm
            lớn vẫn mất độ chính xác. Thêm vào đó, các tùy chọn biên dịch mặc định của Metal không tuân thủ IEEE chặt chẽ (ch05 C7),
            nên hãy đặt một dung sai bất kể hàm mũ nào.
          </p>
        }
      >
        <p>
          The MSL specification bounds the precise single-precision <code>exp</code> by 4 ulp, and the fast one by 3 +
          floor(|2x|) ulp, which grows with the size of the argument. Softmax keeps the argument at or below 0, which is
          MLX's reason in its comment, but large negative arguments still lose accuracy. On top of that, Metal's default
          compile options are not IEEE-strict (ch05 C7), so set a tolerance whatever the exponential.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch10-facts.md C4, D3; x3 facts B3; ch05 facts C7', // i18n-ignore: citation
  },
  'x3/X3.3/q-cg-port': {
    kind: 'choice',
    id: 'x3/X3.3/q-cg-port',
    prompt: (
      <Translated vi={<>Bạn có thể đặt softmax CUDA của MLX vào các lab của khóa học này và build nó cho HIP và trình giả lập mà không sửa gì không?</>}>
        <>Could you drop MLX CUDA's softmax into this course's labs and build it for HIP and the emulator unchanged?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không: các header cooperative-groups của nó có tên khác trên HIP, và trình giả lập của khóa học không mô phỏng cg::reduce.">
            No: its cooperative-groups headers have other names on HIP, and the course's emulator does not model cg::reduce.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có: vendors/hip.h đổi tên mọi định danh CUDA, kể cả cooperative groups, nên tệp biên dịch cho HIP nguyên trạng.">
            Yes: vendors/hip.h renames every CUDA identifier, cooperative groups included, so the file compiles for HIP as it is.
          </Translated>
        ),
        why: (
          <Translated vi="Một #define có thể đổi tên một định danh, không phải một đường dẫn #include, và cg::reduce nằm trong một header có tên khác trên HIP.">
            A #define can rename an identifier, not an #include path, and cg::reduce lives in a header whose name differs on HIP.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Có, vì HIP hoàn toàn không hỗ trợ cooperative groups, nên các lời gọi bị bỏ qua.">
            Yes, because HIP does not support cooperative groups at all, so the calls are skipped.
          </Translated>
        ),
        why: (
          <Translated vi="HIP có hỗ trợ chúng, với cùng namespace và tên lớp nhưng tên header khác.">
            HIP does support them, with the same namespace and class names but different header names.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ trên GPU AMD có wavefront rộng 32.">Only on AMD GPUs with 32-wide wavefronts.</Translated>
        ),
        why: (
          <Translated vi="Độ rộng warp là câu hỏi tiếp theo; bản build thất bại trước khi điều đó có ý nghĩa, ngay ở lệnh include.">
            The warp width is a further question; the build fails before it matters, at the include.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>&lt;cooperative_groups.h&gt;</code> và <code>&lt;cooperative_groups/reduce.h&gt;</code> của CUDA là{' '}
            <code>&lt;hip/hip_cooperative_groups.h&gt;</code> và <code>&lt;hip/cooperative_groups/hip_reduce.h&gt;</code> trên
            HIP, nên một mã nguồn chung cần một <code>#if</code> quanh lệnh include, như softmax.cu của llama.cpp có. Trình giả lập
            hỗ trợ <code>tiled_partition</code> tới 32 nhưng liệt kê <code>reduce</code> là không được mô phỏng (labs/README.md).
            Bản thân MLX không có backend AMD.
          </p>
        }
      >
        <p>
          CUDA's <code>&lt;cooperative_groups.h&gt;</code> and <code>&lt;cooperative_groups/reduce.h&gt;</code> are{' '}
          <code>&lt;hip/hip_cooperative_groups.h&gt;</code> and <code>&lt;hip/cooperative_groups/hip_reduce.h&gt;</code> on
          HIP, so a single source needs an <code>#if</code> around the include, as llama.cpp's softmax.cu has. The emulator
          supports <code>tiled_partition</code> up to 32 but lists <code>reduce</code> as not modelled (labs/README.md).
          MLX itself has no AMD backend.
        </p>
      </Translated>
    ),
    source: 'notes/today/infra-facts.md B4; x3 facts B3, B9; ch10 facts D1; labs/README.md (emulator coverage)',
    vi: { source: 'notes/today/infra-facts.md B4; x3 facts B3, B9; ch10 facts D1; labs/README.md (phạm vi của trình giả lập)' },
  },
};

// ---------------------------------------------------------------- the stepper
const f4 = (v: number) => (v === -Infinity ? '−∞' : String(Number(v.toFixed(4))));
const reads = (s?: RowStep) => (s ? s.counters.reads / ROW16.length : 0);
const fmtReads = (r: number) => (Number.isInteger(r) ? String(r) : r.toFixed(2));

function describe(s: RowStep): string {
  const cols = s.cols.length ? `x[${s.cols[0]}..${s.cols[s.cols.length - 1]}]` : '';
  switch (s.op) {
    case 'max':
      return tr(
        `Pass 1 reads ${cols} and keeps the larger value. Running max: ${f4(s.max)}.`,
        `Lượt 1 đọc ${cols} và giữ giá trị lớn hơn. Max hiện tại: ${f4(s.max)}.`,
      );
    case 'sum':
      return tr(
        `Pass 2 reads ${cols} again and adds exp(x − ${f4(s.max)}) for each. Running sum: ${f4(s.sum)}.`,
        `Lượt 2 đọc lại ${cols} và cộng exp(x − ${f4(s.max)}) cho từng giá trị. Tổng hiện tại: ${f4(s.sum)}.`,
      );
    case 'inv':
      return tr(
        `One reciprocal for the whole row: 1 / ${f4(s.sum)} = ${f4(s.inv!)}. No memory traffic.`,
        `Một phép nghịch đảo cho cả hàng: 1 / ${f4(s.sum)} = ${f4(s.inv!)}. Không có lưu lượng bộ nhớ.`,
      );
    case 'write':
      return tr(
        `Pass ${s.pass} reads ${cols} a ${s.path === 'twopass' ? 'third' : 'second'} time and writes exp(x − ${f4(s.max)}) × ${f4(s.sum === 0 ? 0 : 1 / s.sum)} for each.`,
        `Lượt ${s.pass} đọc ${cols} lần thứ ${s.path === 'twopass' ? 'ba' : 'hai'} và ghi exp(x − ${f4(s.max)}) × ${f4(s.sum === 0 ? 0 : 1 / s.sum)} cho từng giá trị.`,
      );
    case 'chunk-max':
      return s.chunk === 0
        ? tr(
            `Chunk 0: read ${cols} into registers. Its max, ${f4(s.max)}, is the first running max.`,
            `Phần 0: đọc ${cols} vào thanh ghi. Max của nó, ${f4(s.max)}, là max hiện tại đầu tiên.`,
          )
        : s.max > s.prevMax!
          ? tr(
              `Chunk ${s.chunk}: read ${cols}. The max rises from ${f4(s.prevMax!)} to ${f4(s.max)}, so the terms added so far are too large by a factor exp(${f4(s.max)} − ${f4(s.prevMax!)}).`,
              `Phần ${s.chunk}: đọc ${cols}. Max tăng từ ${f4(s.prevMax!)} lên ${f4(s.max)}, nên các số hạng đã cộng đến giờ lớn quá một thừa số exp(${f4(s.max)} − ${f4(s.prevMax!)}).`,
            )
          : tr(
              `Chunk ${s.chunk}: read ${cols}. Its values stay below the running max, ${f4(s.max)}.`,
              `Phần ${s.chunk}: đọc ${cols}. Các giá trị của nó vẫn dưới max hiện tại, ${f4(s.max)}.`,
            );
    case 'rescale':
      return s.chunk === 0
        ? tr(
            `The rescale runs on every chunk: here normalizer × exp(−∞ − ${f4(s.max)}) = 0 × 0 = 0, nothing to rescale yet.`,
            `Phép co giãn chạy ở mọi phần: ở đây normalizer × exp(−∞ − ${f4(s.max)}) = 0 × 0 = 0, chưa có gì để co giãn.`,
          )
        : tr(
            `normalizer × exp(${f4(s.prevMax!)} − ${f4(s.max)}) = ${f4(s.before!)} × ${f4(s.factor!)} = ${f4(s.sum)}.${s.factor === 1 ? ' The max did not rise: the factor is exactly 1, and the exponential was computed anyway.' : ''}`,
            `normalizer × exp(${f4(s.prevMax!)} − ${f4(s.max)}) = ${f4(s.before!)} × ${f4(s.factor!)} = ${f4(s.sum)}.${s.factor === 1 ? ' Max không tăng: thừa số đúng bằng 1, và hàm mũ vẫn được tính.' : ''}`,
          );
    case 'add':
      return tr(
        `Add exp(x − ${f4(s.max)}) for the chunk's 4 values. Normaliser: ${f4(s.sum)}.`,
        `Cộng exp(x − ${f4(s.max)}) cho 4 giá trị của phần. Normaliser: ${f4(s.sum)}.`,
      );
  }
}

function RowCells({ values, hot, kind }: { values: (number | null)[]; hot: Set<number>; kind: 'in' | 'out' }) {
  return (
    <div className="row" style={{ gap: '0.5rem', alignItems: 'flex-start' }}>
      {[0, 1, 2, 3].map((k) => (
        <div key={k} className="mx global" style={{ gridTemplateColumns: 'repeat(4, auto)' }} data-chunk={k}>
          {values.slice(k * N_READS, (k + 1) * N_READS).map((v, j) => {
            const i = k * N_READS + j;
            return (
              <span key={i} data-col={i} className={hot.has(i) ? 'hot' : kind === 'out' && v == null ? 'void' : ''} style={{ minWidth: kind === 'out' ? '3.1rem' : undefined }}>
                {v == null ? '' : kind === 'in' ? v.toFixed(1) : v.toFixed(4)}
              </span>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function TwoSoftmaxStepper() {
  const [path, setPath] = useState<PathId>('twopass');
  const [n, setN] = useState(0);
  const [guess, setGuess] = useState('');
  const [answered, setAnswered] = useState<Record<string, { value: number | null; ok: boolean }>>({});
  const trace = useMemo(() => runPath(path), [path]);
  const other = useMemo(() => runPath(path === 'twopass' ? 'online' : 'twopass'), [path]);
  const total = trace.steps.length;
  const cur = n > 0 ? trace.steps[n - 1] : undefined;
  const asks = useMemo(() => new Set(predictionSteps(trace)), [trace]);
  const key = `${path}:${n}`;
  const pending = asks.has(n) && !answered[key];
  const next = trace.steps[n];
  const done = n >= total;
  const tp = path === 'twopass' ? trace : other;
  const on = path === 'online' ? trace : other;

  const choose = (p: PathId) => {
    setPath(p);
    setN(0);
    setGuess('');
  };
  const check = (skip: boolean) => {
    const v = Number(guess);
    const ok = !skip && guess.trim() !== '' && !isNaN(v) && predictionOk(v, next.sum);
    setAnswered((a) => ({ ...a, [key]: { value: skip || guess.trim() === '' ? null : v, ok } }));
    setN(n + 1);
    setGuess('');
  };
  const last = answered[`${path}:${n - 1}`];

  return (
    <Bench
      title={tr('One row, two softmaxes', 'Một hàng, hai softmax')}
      task={tr(
        'Step through both paths. On the online path, predict the normaliser before each rescale.',
        'Chạy từng bước qua cả hai đường. Trên đường online, hãy dự đoán normaliser trước mỗi lần co giãn.',
      )}
      foot={
        <Translated
          vi={
            <>
              Hàng là 16 logits của cảnh, đọc theo từng phần {N_READS} giá trị (N_READS của MLX CUDA). Mọi giá trị là float32, được
              làm tròn sau mỗi phép toán. Số lần đọc đếm các phần tử lấy từ hàng trong bộ nhớ toàn cục; thanh ghi giữ max và tổng
              hiện tại.
            </>
          }
        >
          <>
            The row is the scene's 16 logits, read in chunks of {N_READS} (MLX CUDA's N_READS). Every value is float32, rounded
            after each operation. Reads count elements fetched from the row in global memory; registers hold the running max
            and sum.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Path', 'Đường')}>
          {(['twopass', 'online'] as PathId[]).map((p) => (
            <button key={p} aria-pressed={path === p} onClick={() => choose(p)}>{PATH_LABEL[p]}</button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setN(0); setGuess(''); }} disabled={n === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => { setN(n - 1); setGuess(''); }} disabled={n === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => { setN(n + 1); setGuess(''); }} disabled={done || pending}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => { let k = n; while (k < total && !(asks.has(k) && !answered[`${path}:${k}`])) k++; setN(k); setGuess(''); }} disabled={done || pending}>
          {path === 'online' ? tr('Run to the next prediction', 'Chạy tới dự đoán tiếp theo') : tr('Run to the end', 'Chạy đến cuối')}
        </button>
        <span className="muted" aria-live="polite"><Translated vi={<>bước {n} trên {total}</>}>step {n} of {total}</Translated></span>
      </div>

      <div className="narrate" role="status" style={{ marginBottom: '0.8rem' }} data-part="narrate">
        {cur ? (
          describe(cur)
        ) : (
          <Translated vi={<>Chưa có gì chạy. Hàng nằm trong bộ nhớ toàn cục; khối chưa đọc nó. Nhấn Bước.</>}>
            <>Nothing has run yet. The row is in global memory; the block has not read it. Press Step.</>
          </Translated>
        )}
        {last && cur?.op === 'rescale' ? (
          <>
            <br />
            {last.value == null ? (
              <span className="muted"><Translated vi="Bạn đã bỏ qua dự đoán này.">You skipped this prediction.</Translated></span>
            ) : last.ok ? (
              <b className="t-ok" data-verdict="right"><Translated vi={<>Bạn dự đoán {last.value}: đúng.</>}>You predicted {last.value}: right.</Translated></b>
            ) : (
              <b className="t-fail" data-verdict="wrong">
                <Translated vi={<>Bạn dự đoán {last.value}; mô hình cho {f4(cur.sum)}.</>}>You predicted {last.value}; the model gives {f4(cur.sum)}.</Translated>
              </b>
            )}
          </>
        ) : null}
      </div>

      {pending ? (
        <div className="q" data-part="predict" style={{ marginBottom: '0.8rem' }}>
          <div className="kind"><Translated vi="Dự đoán trước">Predict first</Translated></div>
          <div className="prompt">
            <Translated
              vi={
                <>
                  Phần {next.chunk} vừa đặt max thành {f4(next.max)} (trước đó là {f4(next.prevMax!)}). Normaliser là{' '}
                  {f4(next.before!)}. Sau phép co giãn nó bằng bao nhiêu?
                </>
              }
            >
              <>
                Chunk {next.chunk} just set the max to {f4(next.max)} (it was {f4(next.prevMax!)}). The normaliser is{' '}
                {f4(next.before!)}. What is it after the rescale?
              </>
            </Translated>
          </div>
          <div className="row">
            <input type="number" step="any" value={guess} aria-label={tr('predicted normaliser', 'normaliser dự đoán')} onChange={(e) => setGuess(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && check(false)} />
            <button className="primary" onClick={() => check(false)} disabled={guess.trim() === ''}><Translated vi="Kiểm tra và bước tiếp">Check and step</Translated></button>
            <button className="quiet" onClick={() => check(true)}><Translated vi="Bỏ qua">Skip</Translated></button>
          </div>
        </div>
      ) : null}

      <div className="mx-label t-global"><Translated vi="hàng x (bộ nhớ toàn cục)">the row x (global memory)</Translated></div>
      <RowCells values={ROW16} hot={new Set(cur?.cols ?? [])} kind="in" />
      <table className="threads" style={{ margin: '0.8rem 0', maxWidth: '40rem' }} aria-label={tr('State and traffic', 'Trạng thái và lưu lượng')}>
        <tbody>
          <tr><td><Translated vi="max hiện tại (thanh ghi)">running max (register)</Translated></td><td className="t-reg" data-k="max">{cur ? f4(cur.max) : '−∞'}</td></tr>
          <tr>
            <td>
              <Translated vi={<>{path === 'online' ? 'normaliser' : 'tổng hiện tại'} (thanh ghi)</>}>
                {path === 'online' ? 'normaliser' : 'running sum'} (register)
              </Translated>
            </td>
            <td className="t-reg" data-k="sum">{cur ? f4(cur.sum) : '0'}</td>
          </tr>
          <tr><td><Translated vi="số lần đọc hàng">reads of the row</Translated></td><td data-k="reads">{fmtReads(reads(cur))}</td></tr>
          <tr><td><Translated vi="byte đọc / ghi">bytes read / written</Translated></td><td data-k="bytes">{(cur?.counters.reads ?? 0) * 4} / {(cur?.counters.writes ?? 0) * 4}</td></tr>
          <tr><td><Translated vi="số hàm mũ">exponentials</Translated></td><td data-k="exps">{cur?.counters.exps ?? 0}</td></tr>
        </tbody>
      </table>
      <div className="mx-label t-global"><Translated vi="đầu ra y (bộ nhớ toàn cục)">the output y (global memory)</Translated></div>
      <RowCells values={cur?.out ?? new Array(ROW16.length).fill(null)} hot={new Set(cur?.op === 'write' ? cur.cols : [])} kind="out" />

      {done ? (
        <div data-part="compare" style={{ marginTop: '1rem' }}>
          <p style={{ margin: '0 0 0.5rem' }}>
            <span className="chip ok">
              <Translated
                vi={<>{path === 'online' ? 'online' : 'max, rồi tổng'}: {trace.rowReads} lần đọc, 1 lần ghi, {trace.counters.exps} hàm mũ</>}
              >
                {path === 'online' ? 'online' : 'max, then sum'}: {trace.rowReads} reads, 1 write, {trace.counters.exps} exponentials
              </Translated>
            </span>{' '}
            <span className="chip">
              <Translated
                vi={<>{path === 'online' ? 'max, rồi tổng' : 'online'}: {other.rowReads} lần đọc, 1 lần ghi, {other.counters.exps} hàm mũ</>}
              >
                {path === 'online' ? 'max, then sum' : 'online'}: {other.rowReads} reads, 1 write, {other.counters.exps} exponentials
              </Translated>
            </span>
          </p>
          <table className="threads" aria-label={tr('Both outputs', 'Cả hai đầu ra')} style={{ maxWidth: '40rem' }}>
            <thead>
              <tr>
                <th>c</th><th>x</th>
                <th><Translated vi="max, rồi tổng">max, then sum</Translated></th>
                <th><Translated vi="online">online</Translated></th>
                <th><Translated vi="chênh lệch">difference</Translated></th>
              </tr>
            </thead>
            <tbody>
              {ROW16.map((x, i) => (
                <tr key={i} data-col={i}>
                  <td>{i}</td><td>{x.toFixed(1)}</td>
                  <td>{tp.out[i].toPrecision(8)}</td><td>{on.out[i].toPrecision(8)}</td>
                  <td data-ulps={ulps(tp.out[i], on.out[i])}>{ulps(tp.out[i], on.out[i]) === 0 ? tr('equal', 'bằng nhau') : tr(`${ulps(tp.out[i], on.out[i])} ulp`, `${ulps(tp.out[i], on.out[i])} ulp`)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.5rem 0 0' }} data-part="verdict" data-maxulps={maxUlps(tp.out, on.out)}>
            <Translated
              vi={
                <>
                  Chênh lệch lớn nhất: {maxUlps(tp.out, on.out)} đơn vị ở vị trí cuối (ulp). Hai tổng là {tp.sum.toPrecision(8)} và{' '}
                  {on.sum.toPrecision(8)}: cùng một số, được cộng theo thứ tự khác nhau.
                </>
              }
            >
              <>
                Largest difference: {maxUlps(tp.out, on.out)} units in the last place. The sums are {tp.sum.toPrecision(8)} and{' '}
                {on.sum.toPrecision(8)}: the same number, added in a different order.
              </>
            </Translated>
          </p>
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the classification
const STRUCTURES: Structure[] = ['max-then-sum', 'online', 'one-sum', 'not-read'];
const EXPS: ExpFn[] = ['expf', '__expf', 'fast::exp', 'precise::exp', 'exp', 'none', 'not-read'];

function SortKernels() {
  const [s, setS] = useState<Record<string, Structure | ''>>({});
  const [e, setE] = useState<Record<string, ExpFn | ''>>({});
  const [checked, setChecked] = useState(false);
  const right = KERNELS.reduce((n, k) => n + (s[k.id] === k.structure ? 1 : 0) + (e[k.id] === k.exp ? 1 : 0), 0);
  return (
    <Bench
      title={tr('Sort the kernels', 'Phân loại các kernel')}
      task={tr(
        "For each kernel, pick what it reduces and which exponential it calls. Where the facts are silent, the right answer is 'not read'.",
        "Với mỗi kernel, hãy chọn nó rút gọn những gì và gọi hàm mũ nào. Chỗ các dữ kiện không nói gì, đáp án đúng là 'chưa đọc'.",
      )}
      foot={
        <Translated vi={<>Đáp án lấy từ {FACTS} B2, B3 và notes/today/ch10-facts.md D1, D3, D4, đọc ở các commit liệt kê ở đó.</>}>
          <>Answers from {FACTS} B2, B3 and notes/today/ch10-facts.md D1, D3, D4, read at the commits listed there.</>
        </Translated>
      }
    >
      <div style={{ display: 'grid', gap: '0.6rem' }}>
        {KERNELS.map((k) => (
          <div key={k.id} className="panel" data-kernel={k.id}>
            <div style={{ marginBottom: '0.45rem' }}>
              <b>{k.project}</b>, {k.backend}: <code>{k.kernel}</code> <span className="muted">({k.op})</span>
            </div>
            <div className="row" style={{ alignItems: 'flex-start' }}>
              <label className="field">
                <Translated vi="các phép rút gọn">reductions</Translated>
                <select aria-label={tr(`reductions of ${k.kernel}`, `các phép rút gọn của ${k.kernel}`)} value={s[k.id] ?? ''} onChange={(ev) => { setS({ ...s, [k.id]: ev.target.value as Structure }); setChecked(false); }}>
                  <option value="">{tr('choose', 'chọn')}</option>
                  {STRUCTURES.map((x) => <option key={x} value={x}>{STRUCTURE_LABEL[x]}</option>)}
                </select>
                {checked ? <span>{s[k.id] === k.structure ? <span className="chip ok"><Translated vi="đúng">right</Translated></span> : <span className="chip fail">{STRUCTURE_LABEL[k.structure]}</span>}</span> : null}
              </label>
              <label className="field">
                <Translated vi="hàm mũ">exponential</Translated>
                <select aria-label={tr(`exponential of ${k.kernel}`, `hàm mũ của ${k.kernel}`)} value={e[k.id] ?? ''} onChange={(ev) => { setE({ ...e, [k.id]: ev.target.value as ExpFn }); setChecked(false); }}>
                  <option value="">{tr('choose', 'chọn')}</option>
                  {EXPS.map((x) => <option key={x} value={x}>{EXP_LABEL[x]}</option>)}
                </select>
                {checked ? <span>{e[k.id] === k.exp ? <span className="chip ok"><Translated vi="đúng">right</Translated></span> : <span className="chip fail">{EXP_LABEL[k.exp]}</span>}</span> : null}
              </label>
            </div>
            {checked ? <p className="muted" style={{ margin: '0.45rem 0 0' }}>{k.why} ({k.source})</p> : null}
          </div>
        ))}
      </div>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <button className="primary" onClick={() => setChecked(true)}><Translated vi="Kiểm tra">Check</Translated></button>
        {checked ? (
          <span data-part="score">
            <Translated vi={<>{right} trên {KERNELS.length * 2} đúng.</>}>{right} of {KERNELS.length * 2} right.</Translated>
          </span>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page
/** The line of a lab function on which the second stage reads the per-warp slots behind a guard. */
const guardLine = (fn: { text: string; firstLine: number }) =>
  fn.firstLine + fn.text.split('\n').findIndex((l) => l.includes('laneIdx() < warpsInBlock()'));
export function SX33() {
  const cu = useLabFile(LAB);
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs.
  const { reports } = useReports();
  const CU_STATUS = solutionTabStatus(reports, 'ch10/lab3_softmax_argmax', 'kernel.cu');
  const reduce = cu ? functionSource(cu, 'blockReduceSum') : null;
  const softmax = cu ? functionSource(cu, 'softmaxRowsKernel') : null;
  return (
    <>
      <Translated
        vi={
          <p>
            Hai phép rút gọn theo hàng chạy trong mọi lớp Llama. RMSNorm chuẩn hóa vector ẩn của mỗi token hai lần mỗi lớp, một lần
            trước attention và một lần trước khối feed-forward, và thêm một lần sau lớp cuối. Softmax chạy bên trong attention trên
            mỗi hàng điểm số, và trên bộ từ vựng khi bạn lấy mẫu. Cả hai là phép rút gọn theo khối của chương 10 với một toán tử
            khác và một phần kết khác. Mục này đọc chúng trong llama.cpp, MLX và PyTorch, và hỏi mỗi kernel hai câu: nó đọc hàng bao
            nhiêu lần, và nó gọi hàm mũ nào?
          </p>
        }
      >
        <p>
          Two row reductions run in every Llama layer. RMSNorm normalises each token's hidden vector twice per layer, once
          before attention and once before the feed-forward block, and once more after the last layer. Softmax runs inside
          attention over each row of scores, and over the vocabulary when you sample. Both are chapter 10's block reduction
          with a different operator and a different epilogue. This section reads them in llama.cpp, MLX and PyTorch, and asks
          two questions of each kernel: how many times does it read the row, and which exponential does it call?
        </p>
      </Translated>
      <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
        <Translated
          vi={
            <p>
              Học phần X3 không có trong sách. Mọi phát biểu về một dự án là cách đọc của chúng tôi đối với mã nguồn của nó ở commit
              được ghim trong {FACTS}, vào ngày 2026-11-20. Các đoạn trích được trích ngắn, kèm tệp, một permalink và giấy phép của dự án.
            </p>
          }
        >
          <p>
            Module X3 is not in the book. Every statement about a project is our reading of its source at the commit pinned
            in {FACTS}, on 2026-11-20. Excerpts are quoted short, with the file, a permalink and the project's licence.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="RMSNorm: mỗi hàng một khối, một phép cộng">RMSNorm: one block per row, one sum</Translated></h2>
      <Translated
        vi={
          <p>
            RMSNorm tính y = x · rsqrt(mean(x²) + ε) · w cho mỗi hàng x, với một trọng số học được w. Mọi dự án giao cho mỗi hàng một
            khối riêng trên CUDA, hoặc một threadgroup riêng trên Metal. Khối cộng các bình phương của hàng, rồi mọi luồng cần kết quả
            để co giãn phần hàng của mình. Đó là phép rút gọn theo khối của lab 10.3 với kết quả được trao cho mọi luồng, đúng như
            điều softmax của nó cũng cần.
          </p>
        }
      >
        <p>
          RMSNorm computes y = x · rsqrt(mean(x²) + ε) · w for each row x, with a learned weight w. Every project gives a row
          its own block on CUDA, or its own threadgroup on Metal. The block sums the squares of its row, and every thread then needs the result to
          scale its share of the row. That is lab 10.3's block reduction with the result handed to every thread, which is
          exactly what its softmax needed too.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x3-facts.md B2)">{/* i18n-ignore: date and citation */}
        <Translated vi={<p>norm.cu của llama.cpp thay đổi lần cuối ngày 2026-11-18, rms_norm.metal của MLX ngày 2026-08-19 và rms_norm.cu ngày 2026-08-13.</p>}>
          <p>llama.cpp norm.cu last changed 2026-11-18, MLX's rms_norm.metal 2026-08-19 and rms_norm.cu 2026-08-13.</p>
        </Translated>
      </Today>
      <Translated vi={<p>Đây là kernel CUDA của llama.cpp, cũng là kernel HIP của nó. Hãy đọc nó trước câu hỏi bên dưới.</p>}>
        <p>Here is llama.cpp's CUDA kernel, which is also its HIP kernel. Read it before the question below.</p>
      </Translated>
      <Excerpt
        x={EXCERPTS_SX33.rms}
        caption={
          <Translated
            vi={
              <>
                Đường vô hướng. Mỗi hàng một khối (<code>const int row = blockIdx.x;</code>), các tổng riêng phần trong{' '}
                <code>extern __shared__ float s_sum[];</code>.
              </>
            }
          >
            <>
              The scalar path. One block per row (<code>const int row = blockIdx.x;</code>), partials in{' '}
              <code>extern __shared__ float s_sum[];</code>.
            </>
          </Translated>
        }
      />
      <Ask q={questionsSX33['x3/X3.3/q-excerpt-rms']} label={tr('Read the excerpt', 'Đọc đoạn trích')} />
      {reduce ? (
        <Code
          tabs={[{ label: tr('lab 10.3, blockReduceSum', 'lab 10.3, blockReduceSum'), code: reduce.text, status: CU_STATUS }]}
          firstLine={reduce.firstLine}
          caption={
            <Translated
              vi={
                <>
                  Phía khóa học, từ <code>labs/{LAB}</code> (dòng {reduce.firstLine} đến {reduce.lastLine}). Dòng{' '}
                  {guardLine(reduce)} là lệnh kiểm tra biên mà <code>block_reduce</code> của llama.cpp cũng có: các lane vượt quá warp
                  cuối nhận 0, phần tử đơn vị của phép cộng.
                </>
              }
            >
              <>
                The course's side, from <code>labs/{LAB}</code> (lines {reduce.firstLine} to {reduce.lastLine}). Line{' '}
                {guardLine(reduce)} is the guard llama.cpp's <code>block_reduce</code> has too: lanes past the last warp
                take 0, the identity of the sum.
              </>
            </Translated>
          }
        />
      ) : null}
      <Translated
        vi={
          <p>
            Có hai chi tiết đáng biết trước khi bạn đọc thêm norm.cu. Phép nhân trọng số thường được gộp: lượt xử lý đồ thị của
            llama.cpp gộp <code>RMS_NORM</code> theo sau là <code>MUL</code> (và một <code>ADD</code>) vào một lần khởi chạy, nên
            kernel chạy cho một nút RMSNorm là kernel đã gộp trừ khi bạn đặt <code>GGML_CUDA_DISABLE_FUSION</code>. Và có hai
            đường. Đường <code>float4</code> chỉ được dùng khi độ dài hàng là bội của 4 và đích được căn chỉnh 16 byte; nó chọn kích
            thước khối theo độ dài hàng. Đường vô hướng dùng 256 luồng cho các hàng dưới 1024 phần tử và 1024 luồng trong trường hợp
            còn lại.
          </p>
        }
      >
        <p>
          Two details are worth knowing before you read more of norm.cu. The weight multiply is usually fused: llama.cpp's
          graph pass merges <code>RMS_NORM</code> followed by <code>MUL</code> (and an <code>ADD</code>) into one launch, so
          the kernel that runs for an RMSNorm node is the fused one unless you set <code>GGML_CUDA_DISABLE_FUSION</code>.
          And there are two paths. A <code>float4</code> path is taken only when the row length is a multiple of 4 and the
          destination is 16-byte aligned; it picks its block size by row length. The scalar path uses 256 threads for rows
          under 1024 elements and 1024 threads otherwise.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Lượt gộp nằm trong ggml-cuda.cu (dòng 5268 đến 5271, x3 facts A1). Đó là thứ đầu tiên cần kiểm tra khi kernel bạn
              đang đọc không khớp với những gì profiler cho thấy.
            </p>
          }
        >
          <p>
            The fusion pass is in ggml-cuda.cu (lines 5268 to 5271, x3 facts A1). It is the first thing to check when the
            kernel you are reading does not match what the profiler shows.
          </p>
        </Translated>
      </Aside>
      <Ask q={questionsSX33['x3/X3.3/q-float4-576']} />

      <h3><Translated vi="Cùng kernel đó trên Metal, và trong MLX và PyTorch">The same kernel on Metal, and in MLX and PyTorch</Translated></h3>
      <Translated
        vi={
          <p>
            Kernel Metal của llama.cpp, <code>kernel_rms_norm_fuse_impl</code> trong norm.metal, là cùng phép rút gọn đó viết bằng các
            hàm SIMD group. SIMD group 0 trước tiên đặt các ô threadgroup về 0, nên giai đoạn thứ hai đọc được 0 ở bất cứ đâu không có
            SIMD group nào ghi. Một vòng lặp có bước nhảy tích lũy <code>dot(x[i00], x[i00])</code>, <code>simd_sum</code> rút gọn
            trong mỗi SIMD group, một rào chắn, lane 0 của mỗi SIMD group ghi ô của nó, một rào chắn thứ hai, và mọi lane đọc một ô rồi{' '}
            <code>simd_sum</code> lần nữa. Hệ số co giãn của nó là <code>1.0f/sqrt(mean + args.eps)</code>.
          </p>
        }
      >
        <p>
          llama.cpp's Metal kernel, <code>kernel_rms_norm_fuse_impl</code> in norm.metal, is the same reduction in SIMD-group
          functions. SIMD-group 0 first zeroes the threadgroup slots, so the second stage reads 0 wherever no SIMD-group wrote.
          A strided loop accumulates <code>dot(x[i00], x[i00])</code>, <code>simd_sum</code> reduces within each SIMD-group, a
          barrier, lane 0 of each SIMD-group writes its slot, a second barrier, and every lane reads a slot and{' '}
          <code>simd_sum</code>s again. Its scale is <code>1.0f/sqrt(mean + args.eps)</code>.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Mức gộp là một đối số template F: 1 cho phép chuẩn hóa đơn thuần, 2 kèm trọng số, 3 kèm trọng số và một phép cộng.
            Metal tìm một kernel bằng một chuỗi tên, nên mỗi bản khởi tạo được đặt một tên: dòng khởi tạo{' '}
            <code>kernel_rms_norm_fuse_impl&lt;float, 2&gt;</code> mang{' '}
            <code>[[host_name("kernel_rms_norm_mul_f32")]]</code> (norm.metal dòng 179). Để tìm kernel Metal đứng sau một lời gọi
            từ host, hãy tìm trong các tệp .metal chuỗi mà host định dạng, chứ không tìm tên hàm.
          </p>
        }
      >
        <p>
          The fusion level is a template argument F: 1 for the plain norm, 2 with the weight, 3 with the weight and an add.
          Metal finds a kernel by a name string, so each instantiation is given one: the line that instantiates{' '}
          <code>kernel_rms_norm_fuse_impl&lt;float, 2&gt;</code> carries{' '}
          <code>[[host_name("kernel_rms_norm_mul_f32")]]</code> (norm.metal line 179). To find the Metal kernel behind a host call, search the .metal files for the string the host formats, not for a
          function name.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x3-facts.md B2, C2)">{/* i18n-ignore: date and citation */}
        <Translated vi={<p>Đặc tả MSL ghi nhận <code>[[host_name(name)]]</code> từ Metal 2.2, chính cho việc này: một tên khác cho mỗi bản chuyên biệt hóa template.</p>}>
          <p>The MSL specification documents <code>[[host_name(name)]]</code> from Metal 2.2, for exactly this: a different name for each template specialisation.</p>
        </Translated>
      </Today>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr>
            <th><Translated vi="Dự án, backend">Project, backend</Translated></th>
            <th><Translated vi="Kernel">Kernel</Translated></th>
            <th><Translated vi="Phép rút gọn">Reduction</Translated></th>
            <th><Translated vi="Hệ số co giãn">Scale</Translated></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="llama.cpp, CUDA và HIP">llama.cpp, CUDA and HIP</Translated></td>
            <td><code>rms_norm_f32</code></td>
            <td><Translated vi={<>warp shuffle, rồi <code>block_reduce</code> qua bộ nhớ chia sẻ</>}>warp shuffles, then <code>block_reduce</code> over shared memory</Translated></td>
            <td><code>rsqrtf(mean + eps)</code></td>
          </tr>
          <tr>
            <td>llama.cpp, Metal{/* i18n-ignore: project and backend names */}</td>
            <td><code>kernel_rms_norm_fuse_impl</code></td>
            <td><Translated vi={<><code>simd_sum</code>, các ô threadgroup, <code>simd_sum</code></>}><code>simd_sum</code>, threadgroup slots, <code>simd_sum</code></Translated></td>
            <td><code>1.0f/sqrt(mean + args.eps)</code></td>
          </tr>
          <tr>
            <td>MLX, Metal{/* i18n-ignore: project and backend names */}</td>
            <td><Translated vi={<><code>rms_single_row</code>, <code>rms_looped</code> cho các hàng dài</>}><code>rms_single_row</code>, <code>rms_looped</code> for long rows</Translated></td>
            <td><Translated vi={<><code>simd_sum</code> và <code>local_sums</code> threadgroup</>}><code>simd_sum</code> and threadgroup <code>local_sums</code></Translated></td>
            <td><code>metal::precise::rsqrt(acc / axis_size + eps)</code></td>
          </tr>
          <tr>
            <td>MLX, CUDA{/* i18n-ignore: project and backend names */}</td>
            <td><code>rms_norm_small</code>, <code>rms_norm</code></td>
            <td>
              <Translated vi={<><code>BlockBroadcastReduce</code>: "Similar to cub::BlockReduce, but result is broadcasted to every thread." (kết quả được phát tới mọi luồng)</>}>
                <code>BlockBroadcastReduce</code>: "Similar to cub::BlockReduce, but result is broadcasted to every thread."
              </Translated>
            </td>
            <td><code>rsqrt(normalizer / axis_size + eps)</code></td>
          </tr>
          <tr>
            <td>PyTorch, CUDA{/* i18n-ignore: project and backend names */}</td>
            <td><code>_fused_rms_norm_cuda</code></td>
            <td><Translated vi={<>các kernel của LayerNorm, với một cờ template <code>bool rms_norm</code></>}>LayerNorm's kernels, with a template flag <code>bool rms_norm</code></Translated></td>
            <td><Translated vi="chưa đọc">not read</Translated></td>
          </tr>
          <tr>
            <td>PyTorch, MPS{/* i18n-ignore: project and backend names */}</td>
            <td>RMSNorm.metal</td>
            <td><Translated vi={<>cấu trúc của MLX: <code>simd_sum</code>, <code>local_sums</code></>}>MLX's structure: <code>simd_sum</code>, <code>local_sums</code></Translated></td>
            <td><code>metal::precise::rsqrt(acc / axis_size + eps)</code></td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            MLX chuyển từ kernel một hàng sang kernel lặp khi độ dài hàng vượt một ngưỡng đặt trong mã host. Đường CUDA của PyTorch
            lùi về một tổ hợp các op thông thường cho đầu vào số phức, và cảnh báo "Cannot dispatch to fused implementation." khi
            trọng số và đầu vào có dtype khác nhau: một đường không gộp rất dễ vô tình gặp phải với trọng số float32 trên một mô
            hình bfloat16.
          </p>
        }
      >
        <p>
          MLX switches from the single-row kernel to the looped one above a row length set in its host code. PyTorch's CUDA
          path falls back to a composite of ordinary ops for complex inputs, and warns "Cannot dispatch to fused
          implementation." when the weight and the input have different dtypes: an unfused path that is easy to hit by
          accident with a float32 weight on a bfloat16 model.
        </p>
      </Translated>

      <h3><Translated vi="Hai tham chiếu, hai hàm">Two references, two functions</Translated></h3>
      <Translated
        vi={
          <p>
            Các tham chiếu không thống nhất, và khác biệt có ý nghĩa ngay khi trọng số không phải float32.{' '}
            <code>LlamaRMSNorm</code> của Hugging Face chuyển hàng sang float32, chuẩn hóa nó, chuyển ngược về dtype đầu vào, và chỉ
            sau đó mới nhân với trọng số. llama2.c, vốn dùng float32 xuyên suốt, tính{' '}
            <code>o[j] = weight[j] * (ss * x[j])</code> trong float. Cho một checkpoint bfloat16 chạy qua cả hai và chúng lệch nhau
            nhiều hơn hẳn sai số làm tròn float32.
          </p>
        }
      >
        <p>
          The references disagree, and the difference matters as soon as the weights are not float32. Hugging Face's{' '}
          <code>LlamaRMSNorm</code> converts the row to float32, normalises it, converts it back to the input dtype, and only
          then multiplies by the weight. llama2.c, which is float32 throughout, computes{' '}
          <code>o[j] = weight[j] * (ss * x[j])</code> in float. Run a bfloat16 checkpoint through both and they disagree by
          far more than float32 rounding.
        </p>
      </Translated>
      <Ask q={questionsSX33['x3/X3.3/q-hf-cast']} />

      <h2><Translated vi="Softmax: một op, hai dạng kernel">Softmax: one op, two shapes of kernel</Translated></h2>
      <Translated
        vi={
          <p>
            <code>soft_max_f32</code> CUDA của llama.cpp là softmax của lab 10.3: một phép rút gọn theo khối để tìm max vào một buffer
            dùng chung <code>buf_iw</code>, rồi <code>expf(vals[col] - max_val)</code>, rồi một phép rút gọn theo khối để tính tổng
            vào cùng buffer đó. Giữa hai phép là một rào chắn với chú thích trỏ tới một báo cáo lỗi.
          </p>
        }
      >
        <p>
          llama.cpp's CUDA <code>soft_max_f32</code> is lab 10.3's softmax: a block reduction for the maximum into a shared
          buffer <code>buf_iw</code>, then <code>expf(vals[col] - max_val)</code>, then a block reduction for the sum into
          the same buffer. Between the two sits a barrier with a comment pointing at a bug report.
        </p>
      </Translated>
      <Ask q={questionsSX33['x3/X3.3/q-26385']} />
      <Translated
        vi={
          <p>
            Kernel Metal của llama.cpp có cùng cấu trúc hai lượt: <code>simd_max</code> với các ô không dùng đặt thành{' '}
            <code>-INFINITY</code>, hai rào chắn, <code>simd_max</code> lần nữa; rồi <code>exp(...)</code>, <code>simd_sum</code>{' '}
            với các ô đặt thành 0, hai rào chắn, <code>simd_sum</code> lần nữa. <code>softmax_single_row</code> Metal của MLX cũng
            lấy max trước và tổng sau. Kernel CUDA của MLX thì không.
          </p>
        }
      >
        <p>
          llama.cpp's Metal kernel has the same two-pass structure: <code>simd_max</code> with the unused slots set to{' '}
          <code>-INFINITY</code>, two barriers, <code>simd_max</code> again; then <code>exp(...)</code>, <code>simd_sum</code>{' '}
          with the slots set to 0, two barriers, <code>simd_sum</code> again. MLX's Metal <code>softmax_single_row</code> also
          takes the maximum first and the sum second. MLX's CUDA kernel does not.
        </p>
      </Translated>
      <Excerpt
        x={EXCERPTS_SX33.online}
        caption={
          <Translated
            vi={
              <>
                Bên trong vòng lặp qua hàng theo từng phần <code>N_READS</code>. Sau vòng lặp: một <code>cg::reduce</code> trong warp
                trên các giá trị max với các normaliser được co giãn lại, một giai đoạn thứ hai qua các mảng <code>__shared__</code> được
                đệm bằng phần tử đơn vị, và một lượt ghi <code>softmax_exp(x - maxval) * normalizer</code>.
              </>
            }
          >
            <>
              Inside the loop over the row in chunks of <code>N_READS</code>. After the loop: a warp <code>cg::reduce</code> of the
              maxima with the normalisers rescaled, a second stage through <code>__shared__</code> arrays padded with the
              identity, and one pass that writes <code>softmax_exp(x - maxval) * normalizer</code>.
            </>
          </Translated>
        }
      />
      <Ask q={questionsSX33['x3/X3.3/q-excerpt-online']} label={tr('Read the excerpt', 'Đọc đoạn trích')} />
      {softmax ? (
        <Code
          tabs={[{ label: tr('lab 10.3, softmaxRowsKernel', 'lab 10.3, softmaxRowsKernel'), code: softmax.text, status: CU_STATUS }]}
          firstLine={softmax.firstLine}
          caption={
            <Translated vi={<>Softmax của M4, từ <code>labs/{LAB}</code>: ba vòng lặp qua hàng, và rào chắn giữa hai phép rút gọn.</>}>
              <>M4's softmax, from <code>labs/{LAB}</code>: three loops over the row, and the barrier between the two reductions.</>
            </Translated>
          }
        />
      ) : null}

      <h3><Translated vi="Chạy từng bước qua một hàng">Step through one row</Translated></h3>
      <Translated
        vi={
          <p>
            Lấy một hàng 16 logits và chạy nó theo cả hai cách. Trên đường online, trình chạy từng bước dừng trước mỗi lần co giãn và
            hỏi bạn normaliser mới. Sau đó so sánh hai đầu ra, và lưu lượng.
          </p>
        }
      >
        <p>
          Take one row of 16 logits and run it both ways. On the online path, the stepper stops before each rescale and asks
          you for the new normaliser. Then compare the two outputs, and the traffic.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một khối và không có warp: phép rút gọn theo warp và phép trộn các cặp (max, normaliser) của các warp được gộp thành
              một lần cập nhật tuần tự cho mỗi phần. Hàm mũ là exp của JavaScript làm tròn về float32, không phải fast::exp, expf hay
              __expf. Giá trị khởi đầu của max hiện tại trong MLX không có trong các dữ kiện; mô hình bắt đầu ở âm vô cùng.
            </p>
          }
        >
          <p>
            One block and no warps: the per-warp reduction and the merge of the warps' (max, normaliser) pairs are collapsed
            into one sequential update per chunk. The exponential is JavaScript's exp rounded to float32, not fast::exp, expf
            or __expf. MLX's starting value for the running max is not in the facts; the model starts at minus infinity.
          </p>
        </Translated>
      </Simplification>
      <TwoSoftmaxStepper />
      <Ask q={questionsSX33['x3/X3.3/q-bytes']} label={tr('Work it out', 'Tự tính')} />
      <TwoSoftmaxesScene />

      <h3><Translated vi="Hàm mũ nào">Which exponential</Translated></h3>
      <Translated
        vi={
          <p>
            Lựa chọn còn lại mà mỗi kernel đưa ra là hàm mũ của nó. Kernel CUDA của MLX gọi <code>__expf</code>, với chú thích
            "Softmax doesn't need high precision exponential cause x is gonna be in (-oo, 0] anyway and subsequently it will be
            divided by sum(exp(x_i))." Kernel Metal của MLX gọi <code>fast::exp</code>. Softmax MPS của PyTorch gọi{' '}
            <code>precise::exp</code> xuyên suốt. Hai framework trên cùng một GPU Apple đưa ra lựa chọn ngược nhau cho cùng một op.
            Kernel Metal của llama.cpp gọi <code>exp</code> thông thường, có độ chính xác tùy theo các tùy chọn biên dịch.
          </p>
        }
      >
        <p>
          The other choice each kernel makes is its exponential. MLX's CUDA kernel calls <code>__expf</code>, with the
          comment "Softmax doesn't need high precision exponential cause x is gonna be in (-oo, 0] anyway and subsequently
          it will be divided by sum(exp(x_i))." MLX's Metal kernel calls <code>fast::exp</code>.
          PyTorch's MPS softmax calls <code>precise::exp</code> throughout. Two frameworks on the same Apple GPU made opposite
          choices for the same op. llama.cpp's Metal kernel calls plain <code>exp</code>, whose precision follows the compile
          options.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch10-facts.md C4; ch05-facts.md C7)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              MSL 4.1 giới hạn sai số của <code>exp</code> độ chính xác đơn bản precise ở 4 ulp và của bản fast ở 3 + floor(|2x|) ulp. Các nguồn
              của Apple không thống nhất về chế độ toán học nào là mặc định; cả hai ứng viên đều cho phép kết hợp lại (reassociation).
            </p>
          }
        >
          <p>
            MSL 4.1 bounds precise single-precision <code>exp</code> by 4 ulp and the fast one by 3 + floor(|2x|) ulp. Apple's
            sources disagree about which math mode is the default; both candidates allow reassociation.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Softmax CUDA của PyTorch chọn kernel theo kích thước hàng: các hàng tối đa 2048 phần tử và 8192 byte đi tới một kernel
            persistent với mỗi hàng một warp, "only __shfl_ instructions are required for reductions"; các hàng dài hơn đi tới một
            kernel dùng bộ nhớ chia sẻ hoặc một kernel toàn cục. Hãy kiểm tra những gì bạn đã đọc đến giờ.
          </p>
        }
      >
        <p>
          PyTorch's CUDA softmax picks a kernel by row size: rows of at most 2048 elements and 8192 bytes go to a persistent
          kernel with one warp per row, "only __shfl_ instructions are required for reductions"; longer rows go to a
          shared-memory kernel or a global one. Test what you have read so far.
        </p>
      </Translated>
      <SortKernels />
      <Ask q={questionsSX33['x3/X3.3/q-exp-choice']} />

      <h2><Translated vi="Trên AMD">On AMD</Translated></h2>
      <Translated
        vi={
          <p>
            Bản build HIP của llama.cpp biên dịch cùng softmax.cu và norm.cu. Hai nhánh theo nhà cung cấp hiện ra trong chúng:{' '}
            <code>WARP_SIZE</code> của nó là 32 logic ngay cả trên wavefront AMD rộng 64, với độ rộng vật lý được hỏi riêng (ch04 D1),
            và softmax.cu include <code>&lt;hip/hip_cooperative_groups.h&gt;</code> dưới{' '}
            <code>#ifdef GGML_USE_HIP</code> cho đường cooperative dành cho các hàng rất dài (ch10 D1). Mã nguồn CUDA của PyTorch được
            hipify lúc build. MLX hoàn toàn không có backend AMD, và softmax CUDA của nó sẽ không port được nguyên trạng.
          </p>
        }
      >
        <p>
          llama.cpp's HIP build compiles the same softmax.cu and norm.cu. Two vendor branches are visible in them: its{' '}
          <code>WARP_SIZE</code> is a logical 32 even on 64-wide AMD wavefronts, with the physical width asked for separately
          (ch04 D1), and softmax.cu includes <code>&lt;hip/hip_cooperative_groups.h&gt;</code> under{' '}
          <code>#ifdef GGML_USE_HIP</code> for its cooperative path for very long rows (ch10 D1). PyTorch's CUDA sources are
          hipified at build time. MLX has no AMD backend at all, and its CUDA softmax would not port unchanged.
        </p>
      </Translated>
      <Ask q={questionsSX33['x3/X3.3/q-cg-port']} />

      <h2><Translated vi="Chỗ của phần này bên dưới mô hình của bạn">Where this sits under your model</Translated></h2>
      <Translated
        vi={
          <p>
            Một bước decode Llama chạy RMSNorm 2 lần mỗi lớp cộng một lần ở cuối: 61 lần mỗi token cho 30 lớp của SmolLM2-135M, mỗi
            lần trên một hàng rộng 576. Softmax trên bộ từ vựng ở cuối là hàng mà bộ lấy mẫu của học phần X1 làm việc trên đó: xem{' '}
            <a href={href('x1', 'X1.2')}>X1.2</a> để biết temperature được gộp vào nó ra sao. Mục{' '}
            <a href={href('x3', 'X3.7')}>X3.7</a> đưa RMSNorm vào bước Llama của chính khóa học.
          </p>
        }
      >
        <p>
          A Llama decode step runs RMSNorm 2 times per layer plus once at the end: 61 times per token for SmolLM2-135M's 30
          layers, each over a 576-wide row. The vocabulary softmax at the end is the row the sampler of module X1 works on:
          see <a href={href('x1', 'X1.2')}>X1.2</a> for how temperature folds into it. Section{' '}
          <a href={href('x3', 'X3.7')}>X3.7</a> puts RMSNorm into the course's own Llama step.
        </p>
      </Translated>
    </>
  );
}
