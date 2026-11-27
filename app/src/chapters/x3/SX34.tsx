import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Aside, Bench, Beyond, Simplification, Today } from '../../components/notes';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { fmt, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  CODE_SWITCHES,
  DEFAULT_SHAPE,
  PROJECTS,
  PROJECT_LABEL,
  ROUTES,
  WEIGHT_LABEL,
  readerKnees,
  route,
  type Project,
  type Shape,
  type Verdict,
  type WeightType,
} from '../../sim/matmulroute';
import { functionSource } from '../../sim/toolchain';
import { permalink, type Excerpt as ExcerptData } from '../../sim/x3sources';
import { Excerpt } from './Excerpt';

const FACTS = 'notes/today/x3-facts.md';
const LAB = 'ch22/lab1_gemv/solution/kernel.cu';
const LLAMA_MUL_MAT = permalink('llama.cpp', 'ggml/src/ggml-cuda/ggml-cuda.cu', [2535, 2596]);

// MLX CUDA's gemv_impl, copied from notes/today/x3-facts.md B4 at the pinned commit and shown
// with the module's Excerpt. Not course code: not compiled here (excerpts-g2.test.ts).
export const EXCERPTS_SX34: Record<string, ExcerptData> = {
  gemv: {
    id: 'x34-mlx-gemv-impl', project: 'mlx', path: 'mlx/backend/cuda/gemms/gemv.cu', lines: [56, 83], factsItem: 'B4', lang: 'cuda',
    code: `int row = g_idx.x * rows_per_block + t_idx.y;
if (row < rows) {
  using Acc = typename GemvAccType<T>::type;
  Acc sum = Acc(0);
  for (int col = n_per_thread * warp.thread_rank(); col < cols;
       col += (WARP_SIZE * n_per_thread)) {
    auto local_mat = unsafe_load_vector<n_per_thread>(mat + row * cols + col, 0);
    auto local_vec = unsafe_load_vector<n_per_thread>(vec + col, 0);
    for (int j = 0; j < n_per_thread; ++j) {
      sum += static_cast<Acc>(local_mat[j]) * static_cast<Acc>(local_vec[j]);
    }
  }
  sum = cg::reduce(warp, sum, cg::plus<Acc>{});
  if (warp.thread_rank() == 0) { out[row] = static_cast<T>(sum); }
}`,
  },
};

// ---------------------------------------------------------------- questions
export const questionsSX34: Record<string, Question> = {
  'x3/X3.4/q-one-token': {
    kind: 'choice',
    id: 'x3/X3.4/q-one-token',
    prompt: (
      <Translated vi={<>llama.cpp trên CUDA, activation F32 và trọng số F16, một token. Phép nhân ma trận này có tới cuBLAS không?</>}>
        <>llama.cpp on CUDA, F32 activations and F16 weights, one token. Does this matrix product reach cuBLAS?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Chỉ khi mọi bước kiểm tra trước nó đều từ chối; mmvf là ứng viên đầu tiên, nên hãy dự kiến một GEMV.">
            Only if every check before it declines; mmvf is the first candidate, so expect a GEMV.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Có: trọng số là F16, không phải F32, và bước kiểm tra đầu tiên trong chuỗi gửi mọi phép nhân không phải F32 tới cuBLAS.">Yes: the weights are F16, not F32, and the first check in the chain sends any non-F32 product to cuBLAS.</Translated>,
        why: (
          <Translated vi="Bước kiểm tra đầu tiên đọc src1->type và dst->type: activation và đầu ra, ở đây đều là F32. Trọng số là src0, và kiểu của chúng là thứ các bước kiểm tra sau đọc.">
            {'The first check reads src1->type and dst->type: the activations and the output, both F32 here. The weights are src0, and their type is what the later checks read.'}
          </Translated>
        ),
      },
      {
        label: <Translated vi="Không: ở một token, chuỗi chạy mmvf, bất kể GPU và kiểu trọng số.">No: at one token the chain runs mmvf, whatever the GPU and the weight type.</Translated>,
        why: <Translated vi="Trong giới hạn batch, should_use_mmvf vẫn quyết định, với các nhánh cho từng GPU cụ thể. Các dữ kiện chưa đọc hết hàm này.">Within the batch limit, should_use_mmvf still decides, with branches for particular GPUs. The facts did not read all of it.</Translated>,
      },
      {
        label: <Translated vi="Có: cuBLAS là mặc định, còn các kernel khác dành cho trọng số đã lượng tử hóa.">Yes: cuBLAS is the default, and the other kernels are for quantised weights.</Translated>,
        why: <Translated vi="mmvf và mmf dành cho trọng số float; cuBLAS là thứ còn lại khi mọi bước kiểm tra đều từ chối.">mmvf and mmf are for float weights; cuBLAS is what is left when every check declines.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Chuỗi chạy theo thứ tự trong mã nguồn: cuBLAS nếu activation (<code>src1</code>) hoặc đầu ra không phải F32, rồi{' '}
            <code>should_use_mmvf</code>, <code>mmf</code>, <code>mmvq</code>, <code>mmq</code>, một đường FP8, và cuBLAS sau
            cùng. Một token nằm trong <code>MMVF_MAX_BATCH_SIZE</code>, nên mmvf là lộ trình đầu tiên không bị loại. Nó có được
            chọn hay không được quyết định bên trong <code>ggml_cuda_should_use_mmvf</code>, hàm mà module này chưa đọc hết: vì
            vậy "hãy dự kiến một GEMV", chứ không phải "nó chạy mmvf".
          </p>
        }
      >
        <p>
          The chain runs in source order: cuBLAS if the activations (<code>src1</code>) or the output are not F32, then <code>should_use_mmvf</code>, <code>mmf</code>,{' '}
          <code>mmvq</code>, <code>mmq</code>, an FP8 path, and cuBLAS last. One token is within{' '}
          <code>MMVF_MAX_BATCH_SIZE</code>, so mmvf is the first route not ruled out. Whether it is taken is decided inside{' '}
          <code>ggml_cuda_should_use_mmvf</code>, which this module did not read in full: so "expect a GEMV", not "it runs
          mmvf".
        </p>
      </Translated>
    ),
    source: `${FACTS} B4 (ggml-cuda.cu lines 2535 to 2596); notes/today/ch22-facts.md D6`,
    vi: { source: `${FACTS} B4 (ggml-cuda.cu các dòng 2535 đến 2596); notes/today/ch22-facts.md D6` },
  },
  'x3/X3.4/q-nine': {
    kind: 'choice',
    id: 'x3/X3.4/q-nine',
    prompt: (
      <Translated vi={<>Cùng mô hình đó, một batch 9 token. Những lộ trình nào bị loại?</>}>
        <>The same model, a batch of 9 tokens. Which routes are ruled out?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="mmvf và mmvq: 9 vượt giới hạn batch 8 của chúng. Còn lại mmf, mmq hoặc cuBLAS.">mmvf and mmvq: 9 is above their batch limit of 8. mmf, mmq or cuBLAS remain.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Chỉ cuBLAS: trên 8 token llama.cpp luôn dùng GEMM riêng của nó.">Only cuBLAS: above 8 tokens llama.cpp always uses its own GEMM.</Translated>,
        why: <Translated vi="cuBLAS vẫn là phương án dự phòng khi mmf và mmq từ chối.">cuBLAS is still the fallback when mmf and mmq decline.</Translated>,
      },
      {
        label: <Translated vi="Không lộ trình nào: giới hạn 8 chỉ áp dụng trên GPU NVIDIA, không áp dụng cho bản build HIP.">None: the limit of 8 applies only on NVIDIA GPUs, not to the HIP build.</Translated>,
        why: <Translated vi="Các hằng số nằm trong mmvf.cuh và mmvq.cuh, mà bản build HIP cũng biên dịch.">The constants are in mmvf.cuh and mmvq.cuh, which the HIP build compiles too.</Translated>,
      },
      {
        label: <Translated vi="mmf, vì kernel của nó cần số token chẵn.">mmf, because its kernel needs an even number of tokens.</Translated>,
        why: <Translated vi="Không dữ kiện nào kiểm tra tính chẵn lẻ.">Nothing in the facts tests parity.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>#define MMVF_MAX_BATCH_SIZE 8</code> và <code>#define MMVQ_MAX_BATCH_SIZE 8</code>, cả hai đều có chú thích
            "Max. batch size for which to use ... kernels". Với một số kiểu lượng tử hóa trên một số GPU, giới hạn của mmvq còn
            thấp hơn nữa ("tuned on RTX 4090"). Vượt quá nó, các kernel GEMM hoặc thư viện sẽ tiếp quản.
          </p>
        }
      >
        <p>
          <code>#define MMVF_MAX_BATCH_SIZE 8</code> and <code>#define MMVQ_MAX_BATCH_SIZE 8</code>, both commented "Max. batch
          size for which to use ... kernels". For some quantised types on some GPUs the mmvq limit is lower still ("tuned on
          RTX 4090"). Above it the GEMM kernels or the library take over.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch22-facts.md D6',// i18n-ignore: citation only
  },
  'x3/X3.4/q-mlx-transposed': {
    kind: 'choice',
    id: 'x3/X3.4/q-mlx-transposed',
    prompt: (
      <Translated
        vi={
          <>
            MLX trên CUDA, <code>y = x @ W.T</code> với W được lưu (out, in) như cách <code>nn.Linear</code> lưu, và x là một
            hàng, không có bias. GEMV hay cuBLASLt?
          </>
        }
      >
        <>
          MLX on CUDA, <code>y = x @ W.T</code> with W stored (out, in) as <code>nn.Linear</code> stores it, and x one row,
          no bias. GEMV or cuBLASLt?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="GEMV: M bằng 1 và b, tức view W.T, được chuyển vị.">GEMV: M is 1 and b, the view W.T, is transposed.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="cuBLASLt: b được chuyển vị, còn GEMV cần bố cục thông thường.">cuBLASLt: b is transposed, and the GEMV needs plain layouts.</Translated>,
        why: <Translated vi="Đọc điều kiện: M == 1 && b_transposed. View chuyển vị chính là thứ nó muốn: mỗi đầu ra là một hàng liền kề của W.">Read the condition: M == 1 && b_transposed. The transposed view is exactly what it wants: each output is a contiguous row of W.</Translated>,
      },
      {
        label: <Translated vi="GEMV chỉ khi N cũng bằng 1.">GEMV only if N is also 1.</Translated>,
        why: <Translated vi="Hai trường hợp được nối bằng ||: hoặc M == 1 với b chuyển vị, hoặc N == 1 với a không chuyển vị.">The two cases are joined by ||: either M == 1 with b transposed, or N == 1 with a not transposed.</Translated>,
      },
      {
        label: <Translated vi="cuBLASLt, vì MLX luôn dùng thư viện trên CUDA.">cuBLASLt, because MLX always uses the library on CUDA.</Translated>,
        why: <Translated vi="Nó dùng thư viện mỗi khi can_use_gemv là false hoặc có bias được gộp, nhưng ở đây thì không.">It does whenever can_use_gemv is false or a bias is fused, but not here.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>return (M == 1 &amp;&amp; b_transposed) || (N == 1 &amp;&amp; !a_transposed);</code> Với b = W.T được lưu
            dưới dạng W, tích vô hướng cho đầu ra j đi dọc hàng j của W, liền kề trong bộ nhớ: đúng bố cục mà một GEMV mỗi warp
            một hàng cần. Một bias được gộp sẽ gửi cùng hình dạng đó tới cuBLASLt, vì nhánh là{' '}
            <code>!bias &amp;&amp; can_use_gemv(...)</code>.
          </p>
        }
      >
        <p>
          <code>return (M == 1 &amp;&amp; b_transposed) || (N == 1 &amp;&amp; !a_transposed);</code> With b = W.T stored as
          W, the dot product for output j walks row j of W, contiguous in memory: the layout a warp-per-row GEMV wants. A
          fused bias sends the same shape to cuBLASLt, because the branch is <code>!bias &amp;&amp; can_use_gemv(...)</code>.
        </p>
      </Translated>
    ),
    source: `${FACTS} B4 (matmul.cpp line 99, gemv.cu line 188)`,
    vi: { source: `${FACTS} B4 (matmul.cpp dòng 99, gemv.cu dòng 188)` },
  },
  'x3/X3.4/q-excerpt-gemv': {
    kind: 'choice',
    id: 'x3/X3.4/q-excerpt-gemv',
    prompt: (
      <Translated vi={<>So sánh <code>gemv_impl</code> của MLX với kernel gather của lab 22.1. Phát biểu nào đúng?</>}>
        <>Compare MLX's <code>gemv_impl</code> with lab 22.1's gather kernel. Which statement is right?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Cùng thiết kế, mỗi hàng của W một warp; MLX nạp vector mà không kiểm tra căn chỉnh và rút gọn bằng cg::reduce, không phải cây shuffle.">
            Same design, a warp per row of W; MLX loads vectors with no alignment test and reduces with cg::reduce, not a shuffle tree.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="MLX chia mỗi hàng cho nhiều warp và cộng các tổng riêng phần của chúng vào out bằng atomic, như kernel split-K của lab 22.1.">MLX splits each row across several warps and adds their partial sums into out with atomics, like lab 22.1's split-K kernel.</Translated>,
        why: <Translated vi="Một warp sở hữu một hàng và lane 0 ghi out[row]: không warp nào khác chạm vào nó, nên không cần atomic.">One warp owns one row and lane 0 writes out[row]: no other warp touches it, so no atomic.</Translated>,
      },
      {
        label: <Translated vi="MLX giao cho mỗi luồng cả một hàng của W, luồng đó tự đi hết hàng, nên không cần rút gọn giữa các lane.">MLX gives each thread a whole row of W, which it walks alone, so no reduction across lanes is needed.</Translated>,
        why: <Translated vi="Các lane cùng nhau nhảy bước dọc theo hàng (col bắt đầu tại n_per_thread * warp.thread_rank()).">The lanes stride along the row together (col starts at n_per_thread * warp.thread_rank()).</Translated>,
      },
      {
        label: <Translated vi="Chúng khác nhau ở thứ chúng tính: MLX nhân với W, lab 22.1 nhân với W chuyển vị.">They differ in what they compute: MLX multiplies by W, lab 22.1 by W transposed.</Translated>,
        why: <Translated vi="Cả hai đều tính tích vô hướng của một hàng của W với vector đầu vào; lab 22.1 là Y = X W^T với W được lưu (out, in).">Both compute the dot product of a row of W with the input vector; lab 22.1 is Y = X W^T with W stored (out, in).</Translated>,
      },
    ],
    explain: (
      <>
        <Translated
          vi={
            <p>
              Cả hai đều là kiểu gather hướng đầu ra của chương 22: mỗi đầu ra một warp, các lane nhảy bước dọc theo K, một phép
              rút gọn warp, lane 0 ghi kết quả. Khác biệt: <code>unsafe_load_vector</code> của MLX đọc n_per_thread phần tử một
              lúc mà không kiểm tra căn chỉnh lúc chạy (<code>load_vector</code> của nó thì có); half và bfloat16 tích lũy bằng
              float thông qua <code>GemvAccType</code>; phép rút gọn là <code>cg::reduce</code>. Lab 22.1 còn phục vụ tới 8
              hàng batch cho mỗi lần đọc W, điều mà gemv của MLX, cho M == 1, không cần.
            </p>
          }
        >
          <p>
            Both are chapter 22's output-centric gather: one warp per output, lanes striding along K, a warp reduction, lane 0
            writes. The differences: MLX's <code>unsafe_load_vector</code> reads n_per_thread elements at once with no run-time
            alignment test (its <code>load_vector</code> has one); half and bfloat16 accumulate in float through{' '}
            <code>GemvAccType</code>; the reduction is <code>cg::reduce</code>. Lab 22.1 also serves up to 8 batch rows per read
            of W, which MLX's gemv, for M == 1, does not need.
          </p>
        </Translated>
      </>
    ),
    source: `${FACTS} B4, B1 (load_vector); lab 22.1 solution`,
    vi: { source: `${FACTS} B4, B1 (load_vector); lời giải tham chiếu lab 22.1` },
  },
  'x3/X3.4/q-static-assert': {
    kind: 'choice',
    id: 'x3/X3.4/q-static-assert',
    prompt: (
      <Translated
        vi={
          <>
            <code>GEMVKernel</code> Metal của MLX chứa{' '}
            <code>static_assert(SM * SN == 32, "simdgroup can only have 32 threads");</code>. Nó giả định điều gì, và khóa học
            tránh giả định đó thế nào?
          </>
        }
      >
        <>
          MLX's Metal <code>GEMVKernel</code> contains <code>static_assert(SM * SN == 32, "simdgroup can only have 32 threads");</code>.
          What does it assume, and how does the course avoid the assumption?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="32 luồng mỗi SIMD group. Khóa học đọc độ rộng từ threadExecutionWidth của pipeline (lab 15.6).">
            32 threads per SIMD-group. The course reads the width from the pipeline's threadExecutionWidth (lab 15.6).
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Một threadgroup đúng 32 luồng; khóa học tránh nó bằng cách khởi chạy threadgroup 256 luồng.">A threadgroup of exactly 32 threads; the course avoids it by launching threadgroups of 256 threads.</Translated>,
        why: <Translated vi="SM × SN là hình dạng của một SIMD group trong kernel, không phải của threadgroup.">SM × SN is the shape of one SIMD-group in the kernel, not of the threadgroup.</Translated>,
      },
      {
        label: <Translated vi="Không gì cả: Apple ghi trong tài liệu độ rộng SIMD group là 32 cho GPU của họ, nên assert chỉ nhắc lại một đảm bảo.">Nothing: Apple documents a SIMD-group width of 32 for its GPUs, so the assert only restates a guarantee.</Translated>,
        why: <Translated vi="Apple không ghi trong tài liệu độ rộng cố định nào; nó được đọc lúc chạy (ch04 D3).">Apple documents no fixed width; it is read at run time (ch04 D3).</Translated>,
      },
      {
        label: <Translated vi="Một ma trận 32 hàng; khóa học tránh nó bằng cách đệm mọi ma trận lên bội số của 32 hàng.">A matrix of 32 rows; the course avoids it by padding every matrix to a multiple of 32 rows.</Translated>,
        why: <Translated vi="Assert nói về luồng, như thông điệp của nó cho thấy.">The assert is about threads, as its message says.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            MLX viết cứng độ rộng trong một số kernel và hỏi nó trong các kernel khác (<code>[[threads_per_simdgroup]]</code>{' '}
            trong argmax của nó). Quy tắc của khóa học là không bao giờ dùng số 32 viết cứng ngoài một chú thích: lab 15.6 biên
            dịch <code>SIMD_WIDTH</code> từ <code>threadExecutionWidth</code> của pipeline.
          </p>
        }
      >
        <p>
          MLX hard-codes the width in some kernels and asks for it in others (<code>[[threads_per_simdgroup]]</code> in its
          argmax). The course's rule is never a literal 32 outside a comment: lab 15.6 compiles <code>SIMD_WIDTH</code> from
          the pipeline's <code>threadExecutionWidth</code>.
        </p>
      </Translated>
    ),
    source: `${FACTS} B4 (gemv.h line 50); notes/today/ch04-facts.md D3; lab 15.6 README`,
    vi: { source: `${FACTS} B4 (gemv.h dòng 50); notes/today/ch04-facts.md D3; README của lab 15.6` },
  },
  'x3/X3.4/q-wide': {
    kind: 'choice',
    id: 'x3/X3.4/q-wide',
    prompt: (
      <Translated vi={<>Vì sao backend Metal của MLX truyền trọng số một lần cho tối đa 5 vector đầu vào thay vì đệm chúng vào một ô GEMM?</>}>
        <>Why does MLX's Metal backend stream the weights once per up to 5 input vectors instead of padding them into a GEMM tile?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một ô GEMM được đệm cũng tính cả các hàng 0; một GEMV đọc W một lần và chỉ làm việc trên các hàng thật.">
            A padded GEMM tile computes the zero rows too; a GEMV reads W once and works on the real rows only.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Metal không có lệnh ma trận SIMD group cho ô nhỏ như vậy, nên GEMM sẽ phải lùi về mã vô hướng.">Metal has no SIMD-group matrix instructions for tiles this small, so a GEMM would fall back to scalar code.</Translated>,
        why: <Translated vi="GEMM steel của MLX dùng ma trận SIMD group; chú thích đưa ra lý do về lưu lượng bộ nhớ, không phải một lệnh bị thiếu.">MLX's steel GEMM uses SIMD-group matrices; the comment gives a traffic reason, not a missing instruction.</Translated>,
      },
      {
        label: <Translated vi="Một ô GEMM sẽ cho kết quả sai với ít hơn 5 hàng, vì các lần nạp của nó giả định ô đầy.">A GEMM tile would give wrong results for fewer than 5 rows, since its loads assume full tiles.</Translated>,
        why: <Translated vi="Một ô được đệm cho đáp án đúng; nó chỉ lãng phí công việc.">A padded tile gives the right answer; it wastes work.</Translated>,
      },
      {
        label: <Translated vi="Vì 5 vector là batch lớn nhất mà một GPU Apple có thể giữ trong bộ nhớ threadgroup cùng lúc.">Because 5 vectors is the largest batch an Apple GPU can hold in its threadgroup memory at once.</Translated>,
        why: <Translated vi="5 là một hằng số trong mã host của MLX, do các tác giả của nó chọn, không phải một số liệu thiết bị.">5 is a constant in MLX's host code, chosen by its authors, not a device figure.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Chú thích ở matmul.cpp dòng 1620: "The wide gemv route streams the weight matrix once per &lt;= 5 input vectors
            instead of running a row-padded GEMM tile." Lab 22.2 cho thấy cùng hiệu ứng từ phía bên kia: ô đầu ra của kernel M3
            của nó là 128 hàng, nên mọi batch từ 1 đến 128 khởi chạy cùng một lưới và làm cùng lượng tính toán trên các hàng
            được điền 0 (ch22 E3).
          </p>
        }
      >
        <p>
          The comment at matmul.cpp line 1620: "The wide gemv route streams the weight matrix once per &lt;= 5 input vectors
          instead of running a row-padded GEMM tile." Lab 22.2 shows the same effect from the other side: its M3 kernel's
          output tile is 128 rows, so every batch from 1 to 128 launches the same grid and does the same arithmetic on
          zero-filled rows (ch22 E3).
        </p>
      </Translated>
    ),
    source: `${FACTS} B4; notes/today/ch22-facts.md E3`,// i18n-ignore: citation only
  },
  'x3/X3.4/q-knee': {
    kind: 'choice',
    id: 'x3/X3.4/q-knee',
    prompt: (
      <Translated vi={<>Điểm gãy (knee) M8 bạn đo được nằm cao hơn hẳn 8 token. Điều đó có mâu thuẫn với điểm chuyển ở 8 của llama.cpp không?</>}>
        <>Your measured M8 knee is well above 8 tokens. Does that contradict llama.cpp's switch at 8?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không: con số 8 của llama.cpp được tinh chỉnh cho kernel của họ trên GPU của họ; điểm gãy của bạn đo kernel của bạn trên thiết bị của bạn.">
            No: llama.cpp's 8 was tuned for its own kernels on its GPUs; your knee measures your kernels on your device.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Có: một trong hai phép đo là sai.">Yes: one of the two measurements is wrong.</Translated>,
        why: <Translated vi="Chúng đo những thứ khác nhau: chỗ một kernel của llama.cpp thắng một kernel khác, và chỗ thời gian mỗi bước của bạn bắt đầu tăng.">They measure different things: where one of llama.cpp's kernels beats another, and where your step time starts to grow.</Translated>,
      },
      {
        label: <Translated vi="Có: điểm gãy là thuộc tính của băng thông và năng lực tính toán của GPU, nên mọi dự án đều nên chuyển kernel ở cùng một batch.">Yes: the knee is a property of the GPU’s bandwidth and compute, so every project should switch kernels at the same batch.</Translated>,
        why: <Translated vi="Các ngoại lệ theo từng GPU và từng kiểu trong should_use_mmvq cho thấy ngay cả một dự án cũng chuyển ở những điểm khác nhau.">The per-GPU and per-type exceptions in should_use_mmvq show that even one project switches at different points.</Translated>,
      },
      {
        label: <Translated vi="Không, vì llama.cpp đếm batch theo chuỗi, không theo token, nên số 8 của nó nghĩa là 8 chuỗi trọn vẹn.">No, because llama.cpp counts batches in sequences, not tokens, so its 8 means 8 whole sequences.</Translated>,
        why: <Translated vi="ne11 là số cột của activation: tức là token.">ne11 is the number of columns of the activations: tokens.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            "8" là một hằng số trong mã, không phải một số liệu thiết bị, và các ngoại lệ theo GPU ("tuned on RTX 4090", "tuned
            on RTX 5090") cho thấy điểm giao được đo, không được suy ra. Điểm gãy của bạn là batch mà tại đó bước decode của bạn
            mất 1.5 lần thời gian của batch nhỏ nhất: một sự đổi chế độ trong kernel của bạn trên GPU của bạn. Hai con số trả
            lời hai câu hỏi khác nhau.
          </p>
        }
      >
        <p>
          "8" is a code constant, not a device figure, and the per-GPU exceptions ("tuned on RTX 4090", "tuned on RTX 5090")
          show that the crossover is measured, not derived. Your knee is the batch at which your decode step takes 1.5 times
          its smallest-batch time: a regime change in your kernel on your GPU. The two answer different questions.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch22-facts.md D6; the knee as chapter 22 defines it (sim/batching.ts kneeOf)',
    vi: { source: 'notes/today/ch22-facts.md D6; điểm gãy theo định nghĩa của chương 22 (sim/batching.ts kneeOf)' },
  },
};

// ---------------------------------------------------------------- the explorer
const TOKENS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 32, 64, 128, 256, 512];
const NS = [1, 576, 4096, 11008];
const VERDICT: Record<Verdict, string> = {
  get true() {
    return tr('true', 'đúng');
  },
  get false() {
    return tr('false', 'sai');
  },
  get 'not-modelled'() {
    return tr('not modelled', 'không mô phỏng');
  },
};

function Strip({ M, knees }: { M: number; knees: { label: string; knee: number }[] }) {
  const W = 640;
  const X0 = 20;
  const X1 = W - 20;
  const x = (t: number) => X0 + (Math.log2(t) / Math.log2(1024)) * (X1 - X0);
  const marks = [...CODE_SWITCHES.map((c) => ({ label: c.label, tokens: c.tokens, kind: 'code' as const })), ...knees.map((k) => ({ label: k.label, tokens: k.knee, kind: 'knee' as const }))];
  const H = 70 + marks.length * 16;
  return (
    <div style={{ overflowX: 'auto' }}>
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Switch points in code and your measured knees', 'Các điểm chuyển trong mã và các điểm gãy bạn đo được')} data-part="strip"
      style={{ width: '100%', maxWidth: W, minWidth: 520, display: 'block', marginTop: '0.6rem' }}>
      <line x1={X0} y1={30} x2={X1} y2={30} stroke="var(--rule)" />
      {[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].map((t) => (
        <g key={t}>
          <line x1={x(t)} y1={26} x2={x(t)} y2={34} stroke="var(--rule)" />
          <text x={x(t)} y={20} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{t}</text>
        </g>
      ))}
      <g data-part="tokens" data-m={M}>
        <line x1={x(M)} y1={24} x2={x(M)} y2={H - 6} stroke="var(--ink)" strokeDasharray="3 3" />
      </g>
      {marks.map((m, k) => (
        <g key={m.label} data-mark={m.kind} data-tokens={m.tokens}>
          <circle cx={x(m.tokens)} cy={30} r={4} fill={m.kind === 'code' ? 'var(--ink)' : 'var(--page)'} stroke="var(--ink)" strokeWidth="1.5" />
          <line x1={x(m.tokens)} y1={34} x2={x(m.tokens)} y2={54 + k * 16} stroke="var(--rule)" />
          <text x={Math.min(x(m.tokens) + 4, X1 - 4)} y={58 + k * 16} fontSize="10.5" textAnchor={x(m.tokens) > W * 0.6 ? 'end' : 'start'} style={{ fill: 'var(--ink)' }}>
            {m.kind === 'code' ? `${m.label}: ${m.tokens}` : tr(`your knee, lab 22.2 with ${m.label}: ${m.tokens}`, `điểm gãy của bạn, lab 22.2 với ${m.label}: ${m.tokens}`)}
          </text>
        </g>
      ))}
    </svg>
    </div>
  );
}

function WhichKernel() {
  const [project, setProject] = useState<Project>('llama-cuda');
  const [mi, setMi] = useState(0);
  const [shape, setShape] = useState<Shape>(DEFAULT_SHAPE);
  const M = TOKENS[mi];
  const s = useMemo(() => ({ ...shape, M }), [shape, M]);
  const locale = useLocale();
  const r = useMemo(() => route(project, s), [project, s, locale]);
  const all = useMemo(() => PROJECTS.map((p) => route(p, s)), [s, locale]);
  const { reports, loading } = useReports();
  const knees = useMemo(() => readerKnees(reports), [reports, locale]);
  const set = (o: Partial<Shape>) => setShape({ ...shape, ...o });
  const cand = ROUTES[r.candidate];

  return (
    <Bench
      title={tr('Which kernel runs', 'Kernel nào chạy')}
      task={tr(
        'Start at one token in each project. Then raise the tokens past 5, 8 and 9 and watch which checks change.',
        'Bắt đầu ở một token trong mỗi dự án. Sau đó tăng số token vượt 5, 8 và 9 và xem những bước kiểm tra nào thay đổi.',
      )}
      foot={
        <Translated
          vi={
            <>
              Mỗi bước kiểm tra là điều kiện được trích trong {FACTS} B4 và notes/today/ch22-facts.md D6, theo thứ tự trong mã
              nguồn. "Không mô phỏng" nghĩa là nó được quyết định bên trong một hàm mà các dữ kiện chưa đọc; khi đó lộ trình
              hiển thị là lộ trình đầu tiên không bị loại.
            </>
          }
        >
          <>
            Each check is the condition quoted in {FACTS} B4 and notes/today/ch22-facts.md D6, in source order. "Not modelled"
            means it is decided inside a function the facts did not read; the route shown is then the first one not ruled out.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="Dự án và backend">Project and backend</Translated>
          <select value={project} onChange={(e) => setProject(e.target.value as Project)} aria-label={tr('Project and backend', 'Dự án và backend')}>
            {PROJECTS.map((p) => <option key={p} value={p}>{PROJECT_LABEL[p]}</option>)}
          </select>
        </label>
        <label className="field" style={{ flex: '1 1 14rem' }}>
          <Translated vi={<>Số token M = {M}</>}>Tokens M = {M}</Translated>
          <input type="range" min={0} max={TOKENS.length - 1} value={mi} aria-label={tr('Tokens', 'Số token')}
            onChange={(e) => setMi(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
        <label className="field">
          <Translated vi="N (số đầu ra)">N (outputs)</Translated>
          <select value={shape.N} onChange={(e) => set({ N: Number(e.target.value) })} aria-label="N">
            {NS.map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Kiểu trọng số">Weight type</Translated>
          <select value={shape.weight} onChange={(e) => set({ weight: e.target.value as WeightType })} aria-label={tr('Weight type', 'Kiểu trọng số')}>
            {(Object.keys(WEIGHT_LABEL) as WeightType[]).map((w) => <option key={w} value={w}>{WEIGHT_LABEL[w]}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        {project === 'llama-cuda' ? (
          <label className="check"><input type="checkbox" checked={shape.inputF32} onChange={(e) => set({ inputF32: e.target.checked })} />{' '}<Translated vi="activation (src1) là F32">activations (src1) are F32</Translated></label>
        ) : null}
        {project === 'mlx-cuda' || project === 'mlx-metal' ? (
          <>
            <label className="check"><input type="checkbox" checked={shape.bTransposed} onChange={(e) => set({ bTransposed: e.target.checked })} />{' '}<Translated vi="b chuyển vị (x @ W.T, W được lưu (out, in))">b transposed (x @ W.T, W stored (out, in))</Translated></label>
            <label className="check"><input type="checkbox" checked={shape.aTransposed} onChange={(e) => set({ aTransposed: e.target.checked })} />{' '}<Translated vi="a chuyển vị">a transposed</Translated></label>
          </>
        ) : null}
        {project === 'mlx-cuda' ? (
          <label className="check"><input type="checkbox" checked={shape.bias} onChange={(e) => set({ bias: e.target.checked })} />{' '}<Translated vi="bias được gộp">fused bias</Translated></label>
        ) : null}
      </div>

      <ol className="trace" data-part="trace" style={{ margin: '0 0 0.8rem', paddingLeft: '1.4rem' }}>
        {r.checks.map((c, k) => (
          <li key={k} data-verdict={c.verdict} data-route={c.route} style={{ marginBottom: '0.35rem', fontWeight: k === r.at ? 650 : undefined, opacity: k > r.at && r.certain ? 0.5 : 1 }}>
            <code>{c.label}</code> <span className="chip" style={{ border: '1px solid var(--rule)' }}>{VERDICT[c.verdict]}</span>{' '}
            → {ROUTES[c.route].name}
            <div className="muted" style={{ fontWeight: 400 }}>{c.detail} <span>({c.source})</span></div>
          </li>
        ))}
      </ol>
      <div className="narrate" role="status" data-part="result" data-candidate={r.candidate} data-kind={r.kind} data-certain={r.certain ? 1 : 0}>
        {r.certain ? (
          <Translated vi={<>Chạy: <b>{cand.name}</b></>}><>Runs: <b>{cand.name}</b></></Translated>
        ) : (
          <Translated vi={<>Ứng viên đầu tiên: <b>{cand.name}</b>, nếu bước kiểm tra của nó chấp nhận; nếu không thì {r.alternatives.map((a) => ROUTES[a].name).join(', rồi ')}</>}>
            <>First candidate: <b>{cand.name}</b>, if its check accepts; otherwise {r.alternatives.map((a) => ROUTES[a].name).join(', then ')}</>
          </Translated>
        )}
        {' '}<span className="muted">({cand.where})</span>
        {r.notes.map((n) => <div key={n} className="muted">{n}</div>)}
      </div>

      <h4 style={{ margin: '1rem 0 0.4rem' }}><Translated vi="Hình dạng này trong mọi dự án">This shape in every project</Translated></h4>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-part="all">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}><Translated vi="dự án">project</Translated></th>
              <th><Translated vi="lộ trình">route</Translated></th>
              <th><Translated vi="loại">kind</Translated></th>
              <th><Translated vi="đã quyết định">decided</Translated></th>
            </tr>
          </thead>
          <tbody>
            {all.map((x) => (
              <tr key={x.project} data-project={x.project} data-kind={x.kind}>
                <td style={{ paddingLeft: '0.6rem' }}>{PROJECT_LABEL[x.project]}</td>
                <td>{ROUTES[x.candidate].name}</td>
                <td>{x.kind === 'gemv' ? 'GEMV' : x.kind === 'gemm' ? 'GEMM' : x.kind === 'library' ? tr('vendor library', 'thư viện của hãng') : tr('not read', 'chưa đọc')}</td>
                <td>{x.certain ? tr('yes', 'có') : tr('first candidate', 'ứng viên đầu tiên')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h4 style={{ margin: '1rem 0 0.2rem' }}><Translated vi="Các điểm chuyển trong mã, và điểm gãy của bạn">Switch points in code, and your knee</Translated></h4>
      <Strip M={M} knees={knees.filter((k) => k.knee != null).map((k) => ({ label: new Set(knees.map((x) => x.backend)).size > 1 ? `${k.backend}, ${k.short}` : k.short, knee: k.knee! }))} />
      <div data-part="knees">
        {loading ? null : knees.length === 0 ? (
          <Translated
            vi={
              <p className="muted" style={{ margin: '0.4rem 0 0' }}>
                Chưa có phép đo lab 22.2 nào trên máy này. Hãy chạy{' '}
                <code>python3 labs/pmpp.py run 22.2 --solution --bench</code> rồi quay lại: các điểm gãy của bạn sẽ hiện cạnh các
                hằng số.
              </p>
            }
          >
            <p className="muted" style={{ margin: '0.4rem 0 0' }}>
              No lab 22.2 measurement on this machine yet. Run <code>python3 labs/pmpp.py run 22.2 --solution --bench</code> and
              come back: your knees will appear beside the constants.
            </p>
          </Translated>
        ) : (
          <ul style={{ margin: '0.4rem 0 0' }}>
            {knees.map((k) => (
              <li key={k.backend + k.label} data-knee={k.knee ?? 'none'}>
                {k.device} ({k.backend}), {k.label}:{' '}
                {k.knee != null ? (
                  <Translated vi={<>điểm gãy ở batch {k.knee}</>}><>knee at batch {k.knee}</></Translated>
                ) : (
                  <Translated vi={<>không có điểm gãy tới batch {k.upTo}</>}><>no knee up to batch {k.upTo}</></Translated>
                )}
                {k.points.length ? <span className="muted"> ({k.points.slice(0, 3).map((p) => tr(`${fmt(p.seconds * 1e3)} ms at ${p.batch}`, `${fmt(p.seconds * 1e3)} ms ở ${p.batch}`)).join(', ')}, …)</span> : null}
              </li>
            ))}
          </ul>
        )}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page
export function SX34() {
  const cu = useLabFile(LAB);
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs.
  const { reports } = useReports();
  const CU_STATUS = solutionTabStatus(reports, 'ch22/lab1_gemv', 'kernel.cu');
  const dots = cu ? functionSource(cu, 'warpRowDots') : null;
  const sum = cu ? functionSource(cu, 'warpSum') : null;
  const gather = cu ? functionSource(cu, 'gemvGatherKernel') : null;
  return (
    <>
      <Translated
        vi={
          <p>
            <code>torch.nn.Linear</code> là một lời gọi bất kể batch. Bên dưới, không dự án nào trong ba dự án gửi mọi phép nhân
            ma trận tới một kernel duy nhất. Với một hoặc vài token, công việc là một phép nhân ma trận với vector, bị giới hạn
            bởi bộ nhớ dù bạn làm gì (<a href={href('ch05', '5.1')}>câu hỏi về decode</a> của chương 5), và mỗi dự án có một
            GEMV cho nó. Với một batch đầy đủ, đó là một GEMM, và dự án chạy kernel phân ô riêng hoặc gọi thư viện của hãng. Chỗ
            chuyển đổi xảy ra là một hằng số trong mã.
          </p>
        }
      >
        <p>
          <code>torch.nn.Linear</code> is one call whatever the batch. Underneath, none of the three projects sends every matrix
          product to one kernel. For one token or a few, the work is a matrix-vector product, memory-bound whatever you do
          (chapter 5's <a href={href('ch05', '5.1')}>decode question</a>), and each project has a GEMV for it. For a full batch
          it is a GEMM, and the project runs its own tiled kernel or calls the vendor's library. Where the switch happens is a
          constant in code.
        </p>
      </Translated>
      <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
        <Translated
          vi={
            <p>
              Các chuỗi dispatch là cách chúng tôi đọc mã nguồn của từng dự án tại các commit trong {FACTS}. Chỗ nào một quyết
              định xảy ra bên trong một hàm chưa được đọc, trang này nói rõ thay vì đoán.
            </p>
          }
        >
          <p>
            The dispatch chains are our reading of each project's source at the commits in {FACTS}. Where a decision happens
            inside a function that was not read, this page says so instead of guessing.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="llama.cpp: một chuỗi kiểm tra theo thứ tự mã nguồn">llama.cpp: a chain of checks in source order</Translated></h2>
      <Translated
        vi={
          <p>
            <code>ggml_cuda_mul_mat</code> trong ggml-cuda.cu (các dòng 2535 đến 2596) thử, theo thứ tự: cuBLAS nếu activation
            (<code>src1</code>) hoặc đầu ra không phải F32 (trọng số, <code>src0</code>, không thuộc phép kiểm tra này); mmvf,
            một GEMV float; mmf; mmvq, một GEMV lượng tử hóa; mmq, một GEMM lượng tử hóa; một đường FP8; và cuBLAS nếu không có
            gì khác chấp nhận. Mỗi ứng viên có hàm <code>ggml_cuda_should_use_*</code> riêng. Hai kernel GEMV dừng ở batch 8:{' '}
            <code>#define MMVF_MAX_BATCH_SIZE 8</code> và <code>#define MMVQ_MAX_BATCH_SIZE 8</code>. Bên trong các hàm có các
            ngoại lệ theo từng GPU và từng kiểu, với chú thích như "tuned on RTX 4090", và các nhánh cho RDNA của AMD.
          </p>
        }
      >
        <p>
          <code>ggml_cuda_mul_mat</code> in ggml-cuda.cu (lines 2535 to 2596) tries, in order: cuBLAS if the activations (<code>src1</code>) or the output are not F32 (the weights, <code>src0</code>, are not part of this test);
          mmvf, a float GEMV; mmf; mmvq, a quantised GEMV; mmq, a quantised GEMM; an FP8 path; and cuBLAS if nothing else
          accepted. Each candidate has its own <code>ggml_cuda_should_use_*</code> function. The two GEMV kernels stop at a batch
          of 8: <code>#define MMVF_MAX_BATCH_SIZE 8</code> and <code>#define MMVQ_MAX_BATCH_SIZE 8</code>. Inside the functions
          are exceptions per GPU and per type, with comments such as "tuned on RTX 4090", and branches for AMD's RDNA.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x3-facts.md B4; ch22-facts.md D6; ch15-facts.md D1)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Cùng file đó là backend HIP: chuỗi trên AMD giống hệt, với các nhánh AMD bên trong should_use_mmvf. Trên Metal,
              llama.cpp dùng mul_mv cho một token, mul_mv_ext cho 2 đến 8 (4 đến 8 cho k-quant), và mul_mm với ma trận SIMD
              group ở trên mức đó, khi <code>ggml_metal_op_mul_mat_use_mm</code> cho phép.
            </p>
          }
        >
          <p>
            The same file is the HIP backend: the chain is the same on AMD, with AMD branches inside should_use_mmvf. On Metal,
            llama.cpp uses mul_mv for one token, mul_mv_ext for 2 to 8 (4 to 8 for k-quants), and mul_mm with SIMD-group
            matrices above, when <code>ggml_metal_op_mul_mat_use_mm</code> allows.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Hãy đọc hằng số đó đúng như bản chất của nó: một con số trong một header, được các tác giả llama.cpp chọn cho kernel
            của họ. Các ngoại lệ, được tinh chỉnh trên từng GPU cụ thể, cho thấy điểm giao thay đổi theo GPU và kiểu trọng
            số. Nó được đo, không được suy ra từ phần cứng.
          </p>
        }
      >
        <p>
          Read the constant as what it is: a number in a header, chosen by llama.cpp's authors for their kernels. The
          exceptions, tuned on particular GPUs, show that the crossover moves with the GPU and the weight type. It is
          measured, not derived from the hardware.
        </p>
      </Translated>

      <h2><Translated vi="MLX: một dòng quyết định">MLX: one line decides</Translated></h2>
      <Translated
        vi={
          <p>
            MLX trên CUDA có một phép kiểm tra (matmul.cpp dòng 99): <code>if (!bias &amp;&amp; cu::can_use_gemv(M, N, K, a_transposed,
            b_transposed))</code> chạy GEMV riêng của nó, và mọi trường hợp khác đi tới cuBLASLt. Điều kiện là{' '}
            <code>(M == 1 &amp;&amp; b_transposed) || (N == 1 &amp;&amp; !a_transposed)</code>. GEMV giao cho mỗi hàng của ma
            trận một warp:
          </p>
        }
      >
        <p>
          MLX on CUDA has one test (matmul.cpp line 99): <code>if (!bias &amp;&amp; cu::can_use_gemv(M, N, K, a_transposed,
          b_transposed))</code> runs its own GEMV, and anything else goes to cuBLASLt. The condition is{' '}
          <code>(M == 1 &amp;&amp; b_transposed) || (N == 1 &amp;&amp; !a_transposed)</code>. The GEMV gives each row of the
          matrix one warp:
        </p>
      </Translated>
      <Excerpt
        x={EXCERPTS_SX34.gemv}
        caption={
          <Translated vi={<>Half và bfloat16 tích lũy bằng float (<code>GemvAccType</code>).</>}>
            <>Half and bfloat16 accumulate in float (<code>GemvAccType</code>).</>
          </Translated>
        }
      />
      {dots && sum && gather ? (
        <Code
          tabs={[
            { label: tr('lab 22.1, warpRowDots', 'lab 22.1, warpRowDots'), code: dots.text, status: CU_STATUS },
            { label: 'gemvGatherKernel', code: gather.text, status: CU_STATUS },
            { label: 'warpSum', code: sum.text, status: CU_STATUS },
          ]}
          caption={
            <Translated
              vi={<>Phía khóa học: kernel gather của lab 22.1, được nạp từ <code>labs/{LAB}</code>. Mỗi đầu ra một warp, cùng hình dạng.</>}
            >
              <>The course's side: lab 22.1's gather, loaded from <code>labs/{LAB}</code>. One warp per output, the same shape.</>
            </Translated>
          }
        />
      ) : null}
      <Ask q={questionsSX34['x3/X3.4/q-excerpt-gemv']} label={tr('Compare the two', 'So sánh hai bên')} />
      <Translated
        vi={
          <p>
            MLX trên Metal có ba lộ trình trong matmul.cpp. Đầu tiên là một GEMV "rộng": "The wide gemv route streams the weight
            matrix once per &lt;= 5 input vectors instead of running a row-padded GEMM tile." (dòng 1620). Sau đó{' '}
            <code>if (std::min(M, N) == 1) {'{'} return gemv(</code> (dòng 1647). Nếu không thì là GEMM steel, với hình dạng ô
            chọn theo từng lớp thiết bị (ch15 D3).
          </p>
        }
      >
        <p>
          MLX on Metal has three routes in matmul.cpp. First a "wide" GEMV: "The wide gemv route streams the weight matrix once
          per &lt;= 5 input vectors instead of running a row-padded GEMM tile." (line 1620). Then{' '}
          <code>if (std::min(M, N) == 1) {'{'} return gemv(</code> (line 1647). Otherwise the steel GEMM, with tile shapes
          chosen per device class (ch15 D3).
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Kernel GEMV Metal của MLX assert <code>SM * SN == 32</code>, một độ rộng SIMD được viết thẳng vào mã. Các lab của
              khóa học thì đọc độ rộng từ pipeline.
            </p>
          }
        >
          <p>
            MLX's Metal GEMV kernel asserts <code>SM * SN == 32</code>, a SIMD width written into the code. The course's labs
            read the width from the pipeline instead.
          </p>
        </Translated>
      </Aside>
      <Ask q={questionsSX34['x3/X3.4/q-wide']} />
      <Ask q={questionsSX34['x3/X3.4/q-static-assert']} />

      <h2><Translated vi="PyTorch: thư viện trên CUDA, GEMV riêng trên MPS">PyTorch: a library on CUDA, its own GEMV on MPS</Translated></h2>
      <Translated
        vi={
          <p>
            PyTorch trên CUDA giao các phép nhân ma trận float cho cuBLAS hoặc cuBLASLt, tùy backend BLAS ưu tiên của nó, và
            trên ROCm có thể cho Composable Kernel. Điểm vào <code>gemv</code> của nó tắt TF32, với chú thích "gemv is bw bound,
            and does not benefit from TF32". Kernel nào thư viện chạy sau đó thì không phải thứ bạn đọc được trong mã nguồn.
            Trên MPS, <code>Linear.mm</code> dùng <code>MPSNDArrayMatrixMultiplication</code>, và ghi chú rằng "addmm fuses the
            bias and routes rank-1 shapes to the GEMV kernels": <code>gemv_t</code> và <code>gemv_nt</code> trong Gemv.h, một{' '}
            <code>simd_sum</code> của các tổng riêng phần theo từng SIMD group.
          </p>
        }
      >
        <p>
          PyTorch on CUDA hands float matmuls to cuBLAS or cuBLASLt, chosen by its preferred BLAS backend, and on ROCm possibly
          to Composable Kernel. Its <code>gemv</code> entry point turns TF32 off, with the comment "gemv is bw bound, and does
          not benefit from TF32". Which kernel the library then runs is not something you can read in source. On MPS,{' '}
          <code>Linear.mm</code> uses <code>MPSNDArrayMatrixMultiplication</code>, and notes that "addmm fuses the bias and
          routes rank-1 shapes to the GEMV kernels": <code>gemv_t</code> and <code>gemv_nt</code> in Gemv.h, a{' '}
          <code>simd_sum</code> of per-SIMD-group partials.
        </p>
      </Translated>

      <h2><Translated vi="Thử các hình dạng">Try shapes</Translated></h2>
      <Translated
        vi={
          <p>
            Trình khám phá chạy chuỗi của từng dự án trên một hình dạng bạn chọn. Hãy dự đoán trước khi kéo thanh trượt: ở một
            token, mỗi dự án chọn loại kernel nào? Ở 6 token, những dự án nào đã đổi ý?
          </p>
        }
      >
        <p>
          The explorer runs each project's chain on a shape you choose. Predict before you move the slider: at one token, which
          kind of kernel does each project pick? At 6 tokens, which projects have changed their mind?
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Chỉ các điều kiện được trích trong dữ kiện mới được đánh giá. mmf, mmq, đường FP8, use_mm, các điều kiện khác của
              GEMV rộng và lựa chọn riêng của các thư viện được hiển thị như các quyết định có tên, không được đánh giá. Việc
              chọn ô của steel và cuBLAS, stride và batch 3 chiều bị bỏ qua. Việc các hàm "_f" (mmvf, mmf) phục vụ trọng số float
              và các hàm "_q" (mmvq, mmq) phục vụ trọng số lượng tử hóa là cách chúng tôi đọc tên của chúng, và loại trừ lớp còn
              lại.
            </p>
          }
        >
          <p>
            Only the conditions quoted in the facts are evaluated. mmf, mmq, the FP8 path, use_mm, the wide GEMV's other
            conditions and the libraries' own choices are shown as named decisions, not evaluated. Steel and cuBLAS tile
            selection, strides and 3-D batching are left out. That the "_f" functions (mmvf, mmf) serve float weights and the
            "_q" ones (mmvq, mmq) quantised weights is our reading of their names, and rules out the other class.
          </p>
        </Translated>
      </Simplification>
      <WhichKernel />
      <Ask q={questionsSX34['x3/X3.4/q-one-token']} />
      <Ask q={questionsSX34['x3/X3.4/q-nine']} />
      <Ask q={questionsSX34['x3/X3.4/q-mlx-transposed']} />

      <h2><Translated vi="Điểm gãy của bạn so với hằng số của họ">Your knee against their constants</Translated></h2>
      <Translated
        vi={
          <p>
            Lab 22.2 đã đo chỗ bước decode của chính bạn thôi phẳng khi batch tăng. Dải trong trình khám phá đặt điểm gãy đó cạnh
            số 8 của llama.cpp và số 5 của MLX. Các dự án tinh chỉnh theo từng thiết bị và từng hình dạng; cách duy nhất để biết
            điểm chuyển nên nằm ở đâu trên GPU của bạn là đo thời gian kernel của bạn trên nó. Đó cũng là lý do các GEMM của khóa
            học được tinh chỉnh với kích thước ô cỡ NVIDIA của sách không nhất thiết thắng trên GPU Apple: các lab cho thấy thiết
            bị của bạn đo được gì, và không gì khác.
          </p>
        }
      >
        <p>
          Lab 22.2 measured where your own decode step stops being flat as the batch grows. The strip in the explorer puts that
          knee beside llama.cpp's 8 and MLX's 5. Projects tune per device and per shape; the only way to know where the switch
          belongs on your GPU is to time your kernels on it. That is also why the course's GEMMs tuned with the book's
          NVIDIA-sized tiles need not win on an Apple GPU: the labs show what your device measured, and nothing else.
        </p>
      </Translated>
      <Ask q={questionsSX34['x3/X3.4/q-knee']} />
      <Translated
        vi={
          <p>
            Trên AMD bức tranh vẫn vậy với những cái tên khác: llama.cpp biên dịch cùng chuỗi với tinh chỉnh cho AMD, và PyTorch
            gọi hipBLAS hoặc Composable Kernel. MLX không có backend AMD. Hãy tự đọc chuỗi trong{' '}
            <a href={LLAMA_MUL_MAT}>ggml-cuda.cu</a> tại commit được ghim, và thực hiện phép đo trên{' '}
            <a href={href('ch22', 'labs')}>trang lab của chương 22</a>.
          </p>
        }
      >
        <p>
          On AMD the picture is the same with other names: llama.cpp compiles the same chain with AMD tuning, and PyTorch calls
          hipBLAS or Composable Kernel. MLX has no AMD backend. Read the chain yourself in{' '}
          <a href={LLAMA_MUL_MAT}>ggml-cuda.cu</a> at the pinned commit, and take the measurement
          on the <a href={href('ch22', 'labs')}>chapter 22 labs page</a>.
        </p>
      </Translated>
    </>
  );
}
