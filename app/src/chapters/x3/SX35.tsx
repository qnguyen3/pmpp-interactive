import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { useReports } from '../../lib/reports';
import { Aside, Bench, Beyond, Simplification, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { href } from '../../lib/router';
import {
  ATTENTION_HOME,
  ATTN_LABEL,
  ATTN_PROJECTS,
  DEFAULT_INPUT,
  KV_CACHE_STEP,
  OWN_EVERYWHERE,
  TORCH_BACKEND_LABEL,
  CUDA_HOME_OF,
  attnRoute,
  cudaHome,
  type AttnInput,
  type CudaHome,
  type AttnProject,
  type Home,
  type Mask,
  type TorchBackend,
  type Verdict,
} from '../../sim/attnroute';
import { ATTN_HOME_OPTIONS, PROJECTS, PROJECT_NAME, type Project as HomeProject } from '../../sim/backendmap';
import { functionSource } from '../../sim/toolchain';
import { permalink, type Excerpt as ExcerptData } from '../../sim/x3sources';
import { Excerpt } from './Excerpt';

const FACTS = 'notes/today/x3-facts.md';
const LAB = 'ch20/lab4_decode/solution/kernel.cu';
const GITMODULES = permalink('pytorch', '.gitmodules', [115, 117]);
/** The start page's prediction this section reveals. */
export const START_PREDICTION = 'x3/start/p-attention-home';

// MLX CUDA's decode kernel, fragments copied from notes/today/x3-facts.md B5 at the pinned
// commit and shown with the module's Excerpt. Not course code: not compiled here.
export const EXCERPTS_SX35: Record<string, ExcerptData> = {
  sdpav: {
    id: 'x35-mlx-sdpav-exp2', project: 'mlx', path: 'mlx/backend/cuda/scaled_dot_product_attention.cu', lines: [67, 144], factsItem: 'B5', lang: 'cuda',
    code: `const U scale_log2 = params.scale * M_LOG2E;
...
U factor = exp2f(max_score - new_max);
U exp_score = exp2f(score - new_max);
...
cg::reduce(warp, score, cg::plus<U>())`,
  },
};

// ---------------------------------------------------------------- questions
export const questionsSX35: Record<string, Question> = {
  'x3/X3.5/q-excerpt-sdpav': {
    kind: 'choice',
    id: 'x3/X3.5/q-excerpt-sdpav',
    prompt: (
      <Translated vi={<>So sánh các dòng <code>kernel_sdpav</code> của MLX với kernel decode của lab 20.4. Phát biểu nào đúng?</>}>
        <>Compare MLX's <code>kernel_sdpav</code> lines with lab 20.4's decode kernel. Which statement is right?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Cùng một online softmax trên các key; MLX làm việc ở cơ số 2 với log2 e được gộp vào hệ số scale, và cộng bằng cg::reduce.">
            The same online softmax over the keys; MLX works in base 2 with log2 e folded into the scale, and sums with cg::reduce.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="MLX tính trước mọi score cho các key trong cache, rồi một softmax trên chúng, nên hoàn toàn không cần giá trị cực đại chạy.">MLX computes all the scores for the cached keys first, then one softmax over them, so it needs no running maximum at all.</Translated>,
        why: <Translated vi="factor = exp2f(max_score − new_max) là phép co giãn lại theo một giá trị cực đại chạy: chính là bước cập nhật online.">factor = exp2f(max_score − new_max) is the rescale of a running maximum: the online update.</Translated>,
      },
      {
        label: <Translated vi="MLX dùng cơ số 2 vì exp2f là chính xác tuyệt đối.">MLX uses base 2 because exp2f is exact.</Translated>,
        why: <Translated vi="exp2f cũng không chính xác tuyệt đối; cơ số 2 là một cách viết khác của cùng một hàm, không phải một cách chính xác hơn.">exp2f is not exact either; base 2 is a different spelling of the same function, not a more accurate one.</Translated>,
      },
      {
        label: <Translated vi="Lab 20.4 không có online softmax: nó ghi mọi score ra bộ nhớ toàn cục (global memory) và chuẩn hóa chúng trong một kernel thứ hai.">Lab 20.4 has no online softmax: it writes every score to global memory and normalises them in a second kernel.</Translated>,
        why: <Translated vi="Lab 20.4 giữ m, D và đầu ra chưa chuẩn hóa trong thanh ghi (register), cập nhật từng key một.">Lab 20.4 keeps m, D and the un-normalised output in registers, updated one key at a time.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lab 20.4 cập nhật (m, D, Õ) cho mỗi key: <code>alpha = expf(m - mNew)</code>, <code>p = expf(s - mNew)</code>, D = D ·
            alpha + p. MLX làm điều tương tự với <code>exp2f</code>, sau khi nhân hệ số scale với log2 e một lần. Mỗi score là một
            tích vô hướng được cộng trên toàn warp: lab 20.4 dùng butterfly <code>__shfl_xor_sync</code> với một mask có kiểu, MLX
            dùng <code>cg::reduce</code>, một cách viết chỉ có trên CUDA theo các quy tắc tính khả chuyển của khóa học này.
          </p>
        }
      >
        <p>
          Lab 20.4 updates (m, D, Õ) per key: <code>alpha = expf(m - mNew)</code>, <code>p = expf(s - mNew)</code>, D = D ·
          alpha + p. MLX does the same with <code>exp2f</code>, after multiplying the scale by log2 e once. Each score is a
          dot product summed across the warp: lab 20.4 with a <code>__shfl_xor_sync</code> butterfly and a typed mask, MLX
          with <code>cg::reduce</code>, a CUDA-only spelling in this course's portability rules.
        </p>
      </Translated>
    ),
    source: `${FACTS} B5; lab 20.4 solution`,
    vi: { source: `${FACTS} B5; lời giải tham chiếu (solution) của lab 20.4` },
  },
  'x3/X3.5/q-exp2': {
    kind: 'choice',
    id: 'x3/X3.5/q-exp2',
    prompt: (
      <Translated
        vi={
          <>
            MLX tính <code>exp2f(s · scale · log2 e − m₂)</code>, trong đó giá trị cực đại chạy m₂ được giữ theo cùng đơn vị
            cơ số 2 đã co giãn. So với <code>exp(s · scale − m)</code>, kết quả thay đổi ở đâu?
          </>
        }
      >
        <>
          MLX computes <code>exp2f(s · scale · log2 e − m₂)</code>, where the running maximum m₂ is kept in the same scaled
          base-2 units. Compared with <code>exp(s · scale − m)</code>, what changes in the result?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không gì cả trong số học chính xác: 2^(x · log2 e) = e^x, và giá trị cực đại cùng đơn vị; chỉ sai số làm tròn khác nhau.">
            Nothing in exact arithmetic: 2^(x · log2 e) = e^x, and the maximum is in the same units; only rounding differs.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Tổng chuẩn hóa ra bị nhân với log2 e, nên kernel phải chia nó cho log2 e trước khi chuẩn hóa đầu ra.">The normaliser comes out multiplied by log2 e, so the kernel must divide it by log2 e before it normalises the output.</Translated>,
        why: <Translated vi="Mỗi số hạng 2^(x log2 e − m₂) bằng e^(x − m), nên tổng của chúng vẫn là cùng một D. Không có gì cần hoàn tác.">Each term 2^(x log2 e − m₂) equals e^(x − m), so their sum is the same D. Nothing needs undoing.</Translated>,
      },
      {
        label: <Translated vi="Softmax trở thành softmax cơ số 2, sắc hơn: score lớn nhất chiếm nhiều trọng số hơn.">The softmax becomes a base-2 softmax, which is sharper: the largest score takes more of the weight.</Translated>,
        why: <Translated vi="Một softmax cơ số 2 trên các score chưa co giãn sẽ khác; gộp log2 e vào hệ số scale làm nó thành cùng một hàm.">A base-2 softmax of the unscaled scores would be different; folding log2 e into the scale makes it the same function.</Translated>,
      },
      {
        label: <Translated vi="Nó có thể tràn số sớm hơn, vì lũy thừa của 2 tăng nhanh hơn.">It can overflow sooner, because powers of 2 grow faster.</Translated>,
        why: <Translated vi="2^y tăng chậm hơn e^y với cùng một y; và dù sao giá trị cực đại cũng được trừ đi trước.">2^y grows more slowly than e^y for the same y; and the maximum is subtracted first anyway.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Với x = s · scale, 2^(x · log2 e) = e^x một cách chính xác. Giá trị cực đại chạy được lấy trên cùng các giá trị đã co
            giãn, nên m₂ = m · log2 e và mọi số mũ là cùng một số viết ở cơ số khác; phép co giãn lại{' '}
            <code>exp2f(max_score - new_max)</code> bằng e^(m_old − m_new). Mô hình trong src/sim/rowreduce.ts chạy cả hai và
            ra cùng một tổng chuẩn hóa, sai khác trong phạm vi làm tròn float32. Vì sao MLX chọn cơ số 2 thì các dữ kiện không
            nói; tiết kiệm một phép nhân cho mỗi hàm mũ, vì đằng nào cũng cần hệ số scale, là phỏng đoán của chúng tôi và unverified.
          </p>
        }
      >
        <p>
          With x = s · scale, 2^(x · log2 e) = e^x exactly. The running maximum is taken over the same scaled values, so m₂ =
          m · log2 e and every exponent is the same number written in another base; the rescale{' '}
          <code>exp2f(max_score - new_max)</code> equals e^(m_old − m_new). The model in src/sim/rowreduce.ts runs both and
          finds the same normaliser to within float32 rounding. Why MLX chose base 2 the facts do not say; saving a multiply
          per exponential, since the scale was needed anyway, is our guess and unverified.
        </p>
      </Translated>
    ),
    source: `${FACTS} B5 (scale_log2 = params.scale * M_LOG2E; exp2f); the identity is ours`,
    vi: { source: `${FACTS} B5 (scale_log2 = params.scale * M_LOG2E; exp2f); đồng nhất thức là của chúng tôi` },
  },
  'x3/X3.5/q-decode-100': {
    kind: 'choice',
    id: 'x3/X3.5/q-decode-100',
    prompt: (
      <Translated vi={<>MLX trên CUDA, một query mới, không mask, 100 key trong cache. Kernel nào?</>}>
        <>MLX on CUDA, one new query, no mask, 100 cached keys. Which kernel?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Kernel vector riêng của MLX; cuDNN chỉ tiếp quản khi cache dài hơn.">MLX's own vector kernel; cuDNN takes over only for a longer cache.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="SDPA của cuDNN: backend CUDA của MLX gửi mọi lời gọi attention tới cuDNN mỗi khi shape được hỗ trợ.">cuDNN's SDPA: MLX's CUDA backend sends every attention call to cuDNN whenever the shapes are supported.</Translated>,
        why: <Translated vi={'Với một query, MLX giữ kernel riêng của nó dưới 256 key: "for small sequence the overhead would kill the advantage."'}>For one query MLX keeps its own kernel below 256 keys: "for small sequence the overhead would kill the advantage."</Translated>,
      },
      {
        label: <Translated vi="Một kernel prefill phân ô, vì cache chỉ 100 key vừa trong một ô (tile).">A tiled prefill kernel, because a cache of only 100 keys fits in a single tile.</Translated>,
        why: <Translated vi="Một query là một bước decode, bất kể cache dài bao nhiêu.">One query is a decode step, whatever the cache length.</Translated>,
      },
      {
        label: <Translated vi="Đường math, ghép từ một matmul, một softmax và một matmul thứ hai.">The math path, composed from a matmul, a softmax and a second matmul.</Translated>,
        why: <Translated vi="Đó là một lộ trình của PyTorch; MLX có các kernel vector riêng.">That is a PyTorch route; MLX has its own vector kernels.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>use_cudnn_for_decoding</code> chỉ nhận một query khi không có mask dạng mảng và có ít nhất{' '}
            <code>kv_cache_step = 256</code> key, một con số mà chú thích của nó nói là lấy từ mlx-lm: kích thước các bước mà
            mlx-lm dùng để mở rộng KV cache. Đó là một hằng số phần mềm, không phải số liệu phần cứng. Dưới ngưỡng đó, MLX chạy{' '}
            <code>kernel_sdpav_1pass</code> hoặc <code>kernel_sdpav_2pass_1</code>.
          </p>
        }
      >
        <p>
          <code>use_cudnn_for_decoding</code> accepts one query only with no array mask and at least{' '}
          <code>kv_cache_step = 256</code> keys, a number its comment says comes from mlx-lm: the size of the steps in which
          mlx-lm grows its KV cache. It is a software constant, not a hardware figure. Below it, MLX runs{' '}
          <code>kernel_sdpav_1pass</code> or <code>kernel_sdpav_2pass_1</code>.
        </p>
      </Translated>
    ),
    source: `${FACTS} B5 (lines 72 to 88, 544); notes/today/ch20-facts.md D3 (mlx-lm step = 256)`,
    vi: { source: `${FACTS} B5 (các dòng 72 đến 88, 544); notes/today/ch20-facts.md D3 (bước của mlx-lm = 256)` },
  },
  'x3/X3.5/q-flash-where': {
    kind: 'choice',
    id: 'x3/X3.5/q-flash-where',
    prompt: (
      <Translated
        vi={
          <>
            Bạn mở <code style={{ overflowWrap: 'anywhere' }}>aten/src/ATen/native/transformers/cuda/flash_attn/</code> và chỉ thấy flash_api.cpp, flash_api.h,
            philox.cuh và static_switch.h. Kernel nằm ở đâu?
          </>
        }
      >
        <>
          You open <code style={{ overflowWrap: 'anywhere' }}>aten/src/ATen/native/transformers/cuda/flash_attn/</code> and find only flash_api.cpp, flash_api.h,
          philox.cuh and static_switch.h. Where is the kernel?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Trong submodule flash-attention tại third_party/flash-attention; flash_api.cpp gọi run_mha_fwd_ từ đó.">
            In the flash-attention submodule at third_party/flash-attention; flash_api.cpp calls run_mha_fwd_ from it.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Trong cuDNN: backend flash của PyTorch là một lớp bọc mỏng gọi fused attention của thư viện cho các shape được hỗ trợ.">In cuDNN: PyTorch’s flash backend is a thin wrapper that calls the library’s fused attention for supported shapes.</Translated>,
        why: <Translated vi="cuDNN attention là một backend riêng trong thứ tự ưu tiên; flash là FlashAttention-2.">cuDNN attention is a separate backend in the priority order; flash is FlashAttention-2.</Translated>,
      },
      {
        label: <Translated vi="Nó được torchgen sinh ra lúc build, như kernel add.">It is generated at build time by torchgen, like the add kernel.</Translated>,
        why: <Translated vi="torchgen viết các kernel ufunc. flash_api.cpp include <flash.h> từ một submodule.">{'torchgen writes the ufunc kernels. flash_api.cpp includes <flash.h> from a submodule.'}</Translated>,
      },
      {
        label: <Translated vi="Trong static_switch.h: các kernel là template ở đó, được khởi tạo theo kích thước head và tính causal.">In static_switch.h: the kernels are templates there, instantiated per head size and causality.</Translated>,
        why: <Translated vi="Header đó chuyển giữa các bản khởi tạo template; các kernel mà nó khởi tạo nằm trong submodule.">That header switches between template instantiations; the kernels it instantiates are in the submodule.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>flash_api.cpp</code> include <code>&lt;flash.h&gt;</code> và gọi{' '}
            <code>run_mha_fwd_&lt;elem_type, kHeadDim, Is_causal&gt;</code> hoặc <code>run_mha_fwd_splitkv_dispatch</code>. Những
            hàm đó đến từ <code>third_party/flash-attention</code>, url https://github.com/Dao-AILab/flash-attention.git, được ghim ở
            bb25ba3 vào ngày kiểm tra dữ kiện. Thư mục trong cây mã của PyTorch chỉ là một lớp đệm (shim).
          </p>
        }
      >
        <p>
          <code>flash_api.cpp</code> includes <code>&lt;flash.h&gt;</code> and calls{' '}
          <code>run_mha_fwd_&lt;elem_type, kHeadDim, Is_causal&gt;</code> or <code>run_mha_fwd_splitkv_dispatch</code>. Those
          come from <code>third_party/flash-attention</code>, url https://github.com/Dao-AILab/flash-attention.git, pinned at
          bb25ba3 on the day of the facts pass. The directory in PyTorch's tree is a shim.
        </p>
      </Translated>
    ),
    source: `${FACTS} B5 (flash_api.cpp lines 294 to 301; .gitmodules lines 115 to 117)`,
    vi: { source: `${FACTS} B5 (flash_api.cpp các dòng 294 đến 301; .gitmodules các dòng 115 đến 117)` },
  },
  'x3/X3.5/q-2pass': {
    kind: 'choice',
    id: 'x3/X3.5/q-2pass',
    prompt: (
      <Translated
        vi={
          <>
            MLX trên Metal, GQA với 4 head query cho mỗi head K/V, một query, 5000 key, trên một thiết bị có chữ cái phân lớp
            không phải d cũng không phải s. Kernel vector nào?
          </>
        }
      >
        <>
          MLX on Metal, GQA with 4 query heads per K/V head, one query, 5000 keys, on a device whose class letter is neither
          d nor s. Which vector kernel?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Kernel 2-pass: số head K/V ít hơn số head query, và n_keys ít nhất là 4096.">The 2-pass kernel: there are fewer K/V heads than query heads, and n_keys is at least 4096.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Kernel 1-pass: kernel 2-pass chỉ dành cho thiết bị 'd' hoặc 's'.">The 1-pass kernel: the 2-pass one is only for 'd' or 's' devices.</Translated>,
        why: <Translated vi="Đó là một trong hai điều kiện nối với nhau bằng ||. Điều kiện kia là GQA với ít nhất 4096 key.">That is one of two conditions joined by ||. The other is GQA with at least 4096 keys.</Translated>,
      },
      {
        label: <Translated vi="Steel attention, kernel phân ô, vì cache 5000 key là dài.">Steel attention, the tiled kernel, because a cache of 5000 keys is long.</Translated>,
        why: <Translated vi="Steel được chọn theo số query, nhiều hơn 8, không theo số key.">Steel is chosen by the number of queries, more than 8, not by the number of keys.</Translated>,
      },
      {
        label: <Translated vi="cuDNN, như với mọi cache dài cỡ này.">cuDNN, as for any cache this long.</Translated>,
        why: <Translated vi="MLX chỉ dùng cuDNN trên backend CUDA của nó.">MLX uses cuDNN only on its CUDA backend.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>if (((devc == 'd' || devc == 's') &amp;&amp; n_keys &gt;= 1024) || (k.shape(1) &lt; q.shape(1) &amp;&amp; n_keys &gt;= 4096))</code>{' '}
            chọn <code>sdpa_vector_2pass</code>. k.shape(1) và q.shape(1) là số head, nên điều kiện thứ hai là GQA với một
            cache dài. Chip Apple nào báo chữ cái nào thì tệp không ghi.
          </p>
        }
      >
        <p>
          <code>if (((devc == 'd' || devc == 's') &amp;&amp; n_keys &gt;= 1024) || (k.shape(1) &lt; q.shape(1) &amp;&amp; n_keys &gt;= 4096))</code>{' '}
          selects <code>sdpa_vector_2pass</code>. k.shape(1) and q.shape(1) are the head counts, so the second condition is GQA
          with a long cache. Which Apple chips report which letter is not documented in the file.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch20-facts.md D4 (lines 1885 to 1896); ch15 D3',
    vi: { source: 'notes/today/ch20-facts.md D4 (các dòng 1885 đến 1896); ch15 D3' },
  },
  'x3/X3.5/q-order': {
    kind: 'choice',
    id: 'x3/X3.5/q-order',
    prompt: (
      <Translated
        vi={
          <>
            scaled_dot_product_attention trên CUDA nhận một shape mà cả flash lẫn memory-efficient attention đều từ chối, trên
            một máy mà <code>check_prefer_cudnn_attention()</code> là false. Backend nào được thử tiếp theo, và điều gì thay đổi
            nếu cuDNN được ưu tiên?
          </>
        }
      >
        <>
          scaled_dot_product_attention on CUDA gets a shape that flash and memory-efficient attention both reject, on a machine
          where <code>check_prefer_cudnn_attention()</code> is false. Which backend is tried next, and what changes if cuDNN is
          preferred?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Math, trước cuDNN. Khi cuDNN được ưu tiên, cuDNN lên đầu, trước cả flash.">Math, before cuDNN. With cuDNN preferred, cuDNN moves to the front, ahead of flash.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="cuDNN: thư viện của hãng là phương án dự phòng sau các kernel gộp, nên ưu tiên nó không thay đổi gì ở đây.">cuDNN: a vendor library is the fallback after the fused kernels, so preferring it changes nothing here.</Translated>,
        why: <Translated vi="Trong thứ tự mặc định, math đứng trước cuDNN. cuDNN chỉ được thử sớm hơn khi nó được ưu tiên.">In the default order math comes before cuDNN. cuDNN is tried earlier only when it is preferred.</Translated>,
      },
      {
        label: <Translated vi="Không cái nào: PyTorch đo hiệu năng các backend còn lại ở lời gọi đầu tiên và lưu lại cái nhanh nhất.">Neither: PyTorch benchmarks the remaining backends on the first call and caches the fastest.</Translated>,
        why: <Translated vi="select_sdp_backend đi qua một danh sách cố định và lấy backend đầu tiên vượt qua bước kiểm tra use_<backend> của nó.">{'select_sdp_backend walks a fixed list and takes the first backend whose use_<backend> check passes.'}</Translated>,
      },
      {
        label: <Translated vi="Math, và ưu tiên cuDNN không thay đổi gì ở đây: math luôn đứng trước mọi thư viện của hãng.">Math, and preferring cuDNN changes nothing here: math always comes before any vendor library.</Translated>,
        why: <Translated vi="Ưu tiên cuDNN thay thứ tự bằng một thứ tự bắt đầu với cuDNN, nên nó được thử đầu tiên.">A preference for cuDNN replaces the order with one that starts with cuDNN, so it is tried first of all.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>sdp_priority_order = {'{'} flash_attention, efficient_attention, math, cudnn_attention, overrideable {'}'}</code>{' '}
            (Context.h các dòng 509 đến 514). Vì vậy một shape mà cả flash lẫn efficient đều từ chối sẽ rơi xuống đường math
            trước khi cuDNN được thử. Khi <code>check_prefer_cudnn_attention()</code> là true, việc duyệt bắt đầu với cuDNN thay
            vào đó (sdp_utils.cpp các dòng 121 đến 131). Hàm <code>use_&lt;backend&gt;</code> của mỗi backend quyết định nó có
            chạy được với các tham số đã cho hay không (Note [SDPA Runtime Dispatch]).
          </p>
        }
      >
        <p>
          <code>sdp_priority_order = {'{'} flash_attention, efficient_attention, math, cudnn_attention, overrideable {'}'}</code>{' '}
          (Context.h lines 509 to 514). So a shape that flash and efficient both reject falls to the math path before cuDNN is
          tried. When <code>check_prefer_cudnn_attention()</code> is true, the walk starts with cuDNN instead (sdp_utils.cpp
          lines 121 to 131). Each backend's <code>use_&lt;backend&gt;</code> function decides whether it can run with the given parameters
          (Note [SDPA Runtime Dispatch]).
        </p>
      </Translated>
    ),
    source: `${FACTS} B5 (Context.h lines 509 to 514; sdp_utils.cpp lines 48 to 58, 121 to 131)`,
    vi: { source: `${FACTS} B5 (Context.h các dòng 509 đến 514; sdp_utils.cpp các dòng 48 đến 58, 121 đến 131)` },
  },
};

// ---------------------------------------------------------------- the explorer
const KEYS = [1, 16, 64, 100, 128, 255, 256, 300, 512, 1000, 1023, 1024, 2048, 4095, 4096, 5000, 8192];
const QUERIES: { label: string; value: number }[] = [
  { label: '1', value: 1 },
  {
    get label() {
      return tr('2 to 8', '2 đến 8');
    },
    value: 8,
  },
  {
    get label() {
      return tr('more (prefill)', 'nhiều hơn (prefill)');
    },
    value: 128,
  },
];
const HOME_LABEL: Record<Home, string> = {
  get own() {
    return tr('own source', 'mã nguồn riêng');
  },
  get library() {
    return tr('vendor library', 'thư viện của hãng');
  },
  get submodule() {
    return tr('submodule', 'submodule');
  },
  get unknown() {
    return tr('not read', 'chưa đọc');
  },
};
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
  get reader() {
    return tr('your answer', 'câu trả lời của bạn');
  },
};
const BACKENDS: TorchBackend[] = ['flash', 'efficient', 'math', 'cudnn', 'overrideable'];

function WhichAttention() {
  const [project, setProject] = useState<AttnProject>('mlx-cuda');
  const [ki, setKi] = useState(KEYS.indexOf(100));
  const [x, setX] = useState<AttnInput>(DEFAULT_INPUT);
  const input = useMemo(() => ({ ...x, keys: KEYS[ki] }), [x, ki]);
  const locale = useLocale();
  const r = useMemo(() => attnRoute(project, input), [project, input, locale]);
  const all = useMemo(() => ATTN_PROJECTS.map((p) => attnRoute(p, input)), [input, locale]);
  const set = (o: Partial<AttnInput>) => setX({ ...x, ...o });
  return (
    <Bench
      title={tr('Which attention kernel', 'Kernel attention nào')}
      task={
        <Translated vi="Bắt đầu với MLX trên CUDA, một query, 100 key. Kéo số key vượt quá 256, rồi chuyển sang prefill, rồi thử các dự án khác.">
          Start with MLX on CUDA, one query, 100 keys. Move the keys past 256, then switch to prefill, then try the other projects.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Mỗi bước kiểm tra là điều kiện đọc được trong {FACTS} B5 và notes/today/ch20-facts.md D4 và D5. "Câu trả lời của
              bạn" đánh dấu một bước kiểm tra có điều kiện thật chưa được đọc: bạn quyết định nó bằng các công tắc.
            </>
          }
        >
          <>
            Each check is the condition read in {FACTS} B5 and notes/today/ch20-facts.md D4 and D5. "Your answer" marks a check
            whose real conditions were not read: you decide it with the switches.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="Dự án và backend">Project and backend</Translated>
          <select value={project} onChange={(e) => setProject(e.target.value as AttnProject)} aria-label={tr('Project and backend', 'Dự án và backend')}>
            {ATTN_PROJECTS.map((p) => <option key={p} value={p}>{ATTN_LABEL[p]}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Số query mỗi chuỗi">Queries per sequence</Translated>
          <span className="seg" role="group" aria-label={tr('Queries', 'Query')}>
            {QUERIES.map((q) => (
              <button key={q.label} aria-pressed={x.queries === q.value} onClick={() => set({ queries: q.value })}>{q.label}</button>
            ))}
          </span>
        </label>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          <Translated vi={<>Số key trong cache = {KEYS[ki]}</>}>Keys in the cache = {KEYS[ki]}</Translated>
          <input type="range" min={0} max={KEYS.length - 1} value={ki} aria-label={tr('Keys', 'Key')} onChange={(e) => setKi(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
        <label className="field">
          <Translated vi="Mask">Mask</Translated>
          <span className="seg" role="group" aria-label={tr('Mask', 'Mask')}>
            {(['none', 'causal', 'array'] as Mask[]).map((m) => (
              <button key={m} aria-pressed={x.mask === m} onClick={() => set({ mask: m })}>{m}</button>
            ))}
          </span>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="check"><input type="checkbox" checked={x.gqa} onChange={(e) => set({ gqa: e.target.checked })} />{' '}<Translated vi="GQA (số head K/V ít hơn số head query)">GQA (fewer K/V heads than query heads)</Translated></label>
        {project === 'mlx-metal' ? (
          <label className="check"><input type="checkbox" checked={x.deviceDS} onChange={(e) => set({ deviceDS: e.target.checked })} /> devc == 'd' || devc == 's'</label>
        ) : null}
        {project === 'mlx-cuda' ? (
          <label className="check"><input type="checkbox" checked={x.cudnnSupports} onChange={(e) => set({ cudnnSupports: e.target.checked })} />{' '}<Translated vi="supports_sdpa_cudnn chấp nhận phần còn lại">supports_sdpa_cudnn accepts the rest</Translated></label>
        ) : null}
        {project === 'torch-cuda' ? (
          <>
            <label className="check"><input type="checkbox" checked={x.preferCudnn} onChange={(e) => set({ preferCudnn: e.target.checked })} /> check_prefer_cudnn_attention()</label>
            {BACKENDS.map((b) => (
              <label key={b} className="check">
                <input type="checkbox" checked={x.use[b]} onChange={(e) => set({ use: { ...x.use, [b]: e.target.checked } })} /> use_{/* i18n-ignore: identifier prefix */}{TORCH_BACKEND_LABEL[b]}
              </label>
            ))}
          </>
        ) : null}
      </div>
      {r.order ? (
        <p className="muted" style={{ margin: '0 0 0.5rem' }} data-part="order"><Translated vi="Thứ tự đã duyệt:">Order walked:</Translated> {r.order.map((b) => TORCH_BACKEND_LABEL[b]).join(', ')}</p>
      ) : null}
      <ol data-part="trace" style={{ margin: '0 0 0.8rem', paddingLeft: '1.4rem' }}>
        {r.checks.map((c, k) => (
          <li key={k} data-verdict={c.verdict} style={{ marginBottom: '0.35rem' }}>
            <code>{c.label}</code> <span className="chip" style={{ border: '1px solid var(--rule)' }}>{VERDICT[c.verdict]}</span>
            <div className="muted">{c.detail} ({c.source})</div>
          </li>
        ))}
      </ol>
      <div className="narrate" role="status" data-part="result" data-home={r.kernel.home}>
        <Translated vi="Chạy:">Runs:</Translated> <b>{r.kernel.name}</b> <span className="chip" style={{ border: '1px solid var(--rule)' }}>{HOME_LABEL[r.kernel.home]}</span>
        {r.kernel.where ? <div className="muted">{r.kernel.where}</div> : null}
        {r.notes.map((n) => <div key={n} className="muted">{n}</div>)}
      </div>
      <h4 style={{ margin: '1rem 0 0.4rem' }}><Translated vi="Lời gọi này trong mọi dự án">This call in every project</Translated></h4>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-part="all">
          <thead><tr><th style={{ paddingLeft: '0.6rem' }}><Translated vi="dự án">project</Translated></th><th><Translated vi="kernel">kernel</Translated></th><th><Translated vi="nằm ở đâu">where it lives</Translated></th></tr></thead>
          <tbody>
            {all.map((a) => (
              <tr key={a.project} data-project={a.project} data-home={a.kernel.home}>
                <td style={{ paddingLeft: '0.6rem' }}>{ATTN_LABEL[a.project]}</td>
                <td>{a.kernel.name}</td>
                <td>{HOME_LABEL[a.kernel.home]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

function StartReveal() {
  const { progress } = useProgress();
  const prior = progress.answers[START_PREDICTION]?.value as Partial<Record<HomeProject, CudaHome>> | undefined;
  const committed = prior && typeof prior === 'object' && PROJECTS.every((p) => prior[p]) ? prior : null;
  const label = (h: CudaHome) => ATTN_HOME_OPTIONS.find((o) => o.id === h)?.label ?? h;
  return (
    <div className="q" data-part="reveal">
      <div className="kind"><Translated vi="Dự đoán của bạn từ trang bắt đầu">Your prediction from the start page</Translated></div>
      <p style={{ margin: '0 0 0.6rem' }} data-predicted={committed ? 'yes' : 'none'}>
        {committed ? (
          <Translated vi={<>Trang bắt đầu đã hỏi các kernel attention CUDA của mỗi dự án đến từ đâu. Đây là những gì bạn đã chốt, đối chiếu với những gì phần này tìm thấy.</>}>
            <>The start page asked where each project's CUDA attention kernels come from. Here is what you committed, against what this section found.</>
          </Translated>
        ) : (
          <Translated
            vi={
              <>
                <a href={href('x3', 'start')}>Trang bắt đầu</a> hỏi các kernel attention CUDA của mỗi dự án đến từ đâu; bạn chưa
                chốt câu trả lời ở đó. Đây là những gì phần này tìm thấy.
              </>
            }
          >
            <>
              The <a href={href('x3', 'start')}>start page</a> asks where each project's CUDA attention kernels come from; you
              have not committed an answer there yet. Here is what this section found.
            </>
          </Translated>
        )}
      </p>
      <table className="data" style={{ margin: '0 0 0.6rem' }} data-part="cuda-homes">
        <thead><tr><th><Translated vi="dự án">project</Translated></th>{committed ? <th><Translated vi="bạn đã nói">you said</Translated></th> : null}<th><Translated vi="theo dữ kiện">by the facts</Translated></th></tr></thead>
        <tbody>
          {PROJECTS.map((p) => {
            const right = cudaHome(CUDA_HOME_OF[p]);
            const ok = committed ? committed[p] === right : null;
            return (
              <tr key={p} data-project={p} data-right={ok == null ? '' : ok ? '1' : '0'} data-home={right}>
                <td>{PROJECT_NAME[p]}</td>
                {committed ? <td>{label(committed[p]!)} {ok ? <span className="chip ok"><Translated vi="đúng">right</Translated></span> : <span className="chip fail"><Translated vi="sai">no</Translated></span>}</td> : null}
                <td>{label(right)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p style={{ margin: '0 0 0.4rem' }}>
        <Translated
          vi={
            <>
              Trên mọi backend GPU, không chỉ CUDA, {OWN_EVERYWHERE.length} trong ba dự án, {OWN_EVERYWHERE.join(', ')}, chạy các kernel
              attention từ cây mã của chính nó:
            </>
          }
        >
          <>
            Across every GPU backend, not only CUDA, {OWN_EVERYWHERE.length} of the three projects, {OWN_EVERYWHERE.join(', ')}, runs attention
            kernels from its own tree:
          </>
        </Translated>
      </p>
      <table className="data" style={{ margin: 0 }} data-part="homes">
        <thead><tr><th><Translated vi="dự án">project</Translated></th><th><Translated vi="backend">backend</Translated></th><th><Translated vi="attention nằm ở đâu">where attention lives</Translated></th><th><Translated vi="toàn bộ là mã nguồn riêng">all own source</Translated></th></tr></thead>
        <tbody>
          {ATTENTION_HOME.map((h) => (
            <tr key={h.project + h.backend} data-own={h.allOwn ? 1 : 0}>
              <td>{h.project}</td><td>{tr(h.backend, ({ 'CUDA and HIP': 'CUDA và HIP' } as Record<string, string>)[h.backend] ?? h.backend)}</td><td>{h.home}</td><td>{h.allOwn ? tr('yes', 'có') : tr('no', 'không')}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------- the page
/** Lines of a lab function that contain any of the given fragments, as file line numbers. */
const linesWith = (fn: { text: string; firstLine: number }, parts: string[]) =>
  fn.text.split('\n').flatMap((l, i) => (parts.some((p) => l.includes(p)) ? [fn.firstLine + i] : []));

export function SX35() {
  const cu = useLabFile(LAB);
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs.
  const { reports } = useReports();
  const CU_STATUS = solutionTabStatus(reports, 'ch20/lab4_decode', 'kernel.cu');
  const decode = cu ? functionSource(cu, 'decodeAttentionKernel') : null;
  const butterfly = cu ? functionSource(cu, 'warpSumAll') : null;
  return (
    <>
      <Translated
        vi={
          <p>
            <code>F.scaled_dot_product_attention</code> trong một mô hình Hugging Face, hay <code>mx.fast.scaled_dot_product_attention</code>{' '}
            trong mlx-lm, là một lời gọi. Đằng sau nó, mỗi dự án chọn một kernel theo số query, độ dài cache, mask và thiết bị,
            và kernel không phải lúc nào cũng nằm trong cây mã của chính dự án. Phần này tìm ra nó trong từng trường hợp, và đọc
            kernel decode CUDA của MLX đối chiếu với lab 20.4.
          </p>
        }
      >
        <p>
          <code>F.scaled_dot_product_attention</code> in a Hugging Face model, or <code>mx.fast.scaled_dot_product_attention</code>{' '}
          in mlx-lm, is one call. Behind it, each project picks a kernel by the number of queries, the length of the cache, the
          mask and the device, and the kernel is not always in the project's own tree. This section finds it in each case, and
          reads MLX's CUDA decode kernel against lab 20.4.
        </p>
      </Translated>
      <Beyond title={tr("Ours, not the book's", 'Của chúng tôi, không phải của sách')}>
        <Translated
          vi={
            <p>
              Các quy tắc dispatch là cách chúng tôi đọc mã nguồn của mỗi dự án tại các commit trong {FACTS}. Có hai điều chưa
              được kiểm tra: MLX cần phiên bản cuDNN nào, và <code>supports_sdpa_cudnn</code> kiểm tra gì ngoài những dòng đã đọc.
            </p>
          }
        >
          <p>
            The dispatch rules are our reading of each project's source at the commits in {FACTS}. Two things were not
            checked: which cuDNN version MLX needs, and what <code>supports_sdpa_cudnn</code> checks beyond the lines read.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Kernel riêng: llama.cpp ở mọi nơi, MLX trên Metal">Own kernels: llama.cpp everywhere, MLX on Metal</Translated></h2>
      <Translated
        vi={
          <p>
            llama.cpp chạy kernel attention riêng của nó trên mọi backend. Trên CUDA, và trên HIP từ cùng các tệp đó, fattn-vec.cuh
            chứa online softmax của chương 20 trong vài dòng, một bước kết hợp split-KV giữa các khối, và một tỉ lệ GQA cho phép
            nhiều head query dùng chung một head K/V đã nạp. Trên Metal, fa_common.metal và fa_aux.metal làm điều tương tự bằng
            các hàm SIMD group, và phân loại các ô mask để bỏ qua những ô bị che hoàn toàn.
          </p>
        }
      >
        <p>
          llama.cpp runs its own attention kernels on every backend. On CUDA, and on HIP from the same files, fattn-vec.cuh
          carries chapter 20's online softmax in a handful of lines, a split-KV combine across blocks, and a GQA ratio that lets
          several query heads share one loaded K/V head. On Metal, fa_common.metal and fa_aux.metal do the same with SIMD-group
          functions, and classify mask tiles so that fully masked ones are skipped.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            MLX trên Metal cũng có kernel riêng, chia theo shape. Từ tám query trở xuống dùng các kernel vector. Trong số đó, một
            kernel 2-pass chia các key được chọn khi, theo chú thích của nó, "The device is large and the sequence length
            long" (chữ cái phân lớp thiết bị là d hoặc s, ít nhất 1024 key) hoặc "The sequence length is even longer and we have gqa" (ít
            nhất 4096 key). Nhiều
            query hơn thì đi tới kernel steel attention phân ô.
          </p>
        }
      >
        <p>
          MLX on Metal also has its own kernels, split by shape. Eight or fewer queries take the vector kernels. Among those, a
          2-pass kernel that splits the keys is chosen when, in its comments, "The device is large and the sequence length
          long" (device class letter d or s, at least 1024 keys) or "The sequence length is even longer and we have gqa" (at
          least 4096 keys). More
          queries go to the tiled steel attention kernel.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md D1, D2, D4)">{/* i18n-ignore: date and citation */}
        <Translated vi={<p>Chip Apple nào báo chữ cái phân lớp thiết bị nào thì tệp của MLX không ghi; hãy đọc nó từ thiết bị của bạn thay vì giả định.</p>}>
          <p>Which Apple chips report which device class letter is not documented in MLX's file; read it from your device rather than assume it.</p>
        </Translated>
      </Today>

      <h2><Translated vi="MLX trên CUDA: một thư viện cho prefill, kernel riêng cho các bước decode ngắn">MLX on CUDA: a library for prefill, its own kernel for short decodes</Translated></h2>
      <Translated
        vi={
          <p>
            Trên CUDA, <code>ScaledDotProductAttention::eval_gpu</code> gọi attention của cuDNN, dựng bằng API đồ thị của cuDNN
            frontend, mỗi khi <code>supports_sdpa_cudnn</code> chấp nhận lời gọi. Với một query duy nhất, nó thêm một điều kiện
            riêng: không có mask dạng mảng, và ít nhất 256 key. Chú thích giải thích: "The cuDNN SDPA is faster than vector kernel but for small
            sequence the overhead would kill the advantage." Ngưỡng là <code>constexpr int kv_cache_step = 256; // number
            is from mlx-lm</code>. Dưới ngưỡng đó, MLX chạy các kernel vector riêng, và online softmax của chúng được viết ở cơ số 2:
          </p>
        }
      >
        <p>
          On CUDA, <code>ScaledDotProductAttention::eval_gpu</code> calls cuDNN's attention, built with the cuDNN frontend graph
          API, whenever <code>supports_sdpa_cudnn</code> accepts the call. For a single query it adds a condition of its own:
          no array mask, and at least 256 keys. The comment explains: "The cuDNN SDPA is faster than vector kernel but for small
          sequence the overhead would kill the advantage." The threshold is <code>constexpr int kv_cache_step = 256; // number
          is from mlx-lm</code>. Below it, MLX runs its own vector kernels, and their online softmax is written in base 2:
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              MLX không có backend AMD: một pull request ROCm thử nghiệm đang mở và chưa được merge (x3 facts B9). Attention CUDA
              của nó cũng phụ thuộc vào frontend của cuDNN và vào cooperative groups.
            </p>
          }
        >
          <p>
            MLX has no AMD backend: an experimental ROCm pull request is open and unmerged (x3 facts B9). Its CUDA attention
            also depends on cuDNN's frontend and on cooperative groups.
          </p>
        </Translated>
      </Aside>
      <Excerpt
        x={EXCERPTS_SX35.sdpav}
        caption={
          <Translated
            vi={
              <>
                Các dòng 67, 143 và 144, trong kernel đầu tiên của hai kernel decode (<code>kernel_sdpav_1pass</code> bắt đầu ở dòng 42,{' '}
                <code>kernel_sdpav_2pass_1</code> ở dòng 195). Kernel Metal sdpa_vector.h có cùng công thức truy hồi với{' '}
                <code>fast::exp</code>.
              </>
            }
          >
            <>
              Lines 67, 143 and 144, in the first of its two decode kernels (<code>kernel_sdpav_1pass</code> starts at line 42,{' '}
              <code>kernel_sdpav_2pass_1</code> at 195). The Metal kernel sdpa_vector.h has the same recurrence with{' '}
              <code>fast::exp</code>.
            </>
          </Translated>
        }
      />
      {decode && butterfly ? (
        <Code
          tabs={[
            { label: tr('lab 20.4, decodeAttentionKernel', 'lab 20.4, decodeAttentionKernel'), code: decode.text, status: CU_STATUS, highlight: linesWith(decode, ['mNew', 'alpha', 'expf(s - mNew)']) },
            { label: 'warpSumAll', code: butterfly.text, status: CU_STATUS },
          ]}
          firstLine={decode.firstLine}
          caption={
            <Translated vi={<>Phía khóa học, từ <code>labs/{LAB}</code>: bước cập nhật online cho mỗi key được tô sáng.</>}>
              <>The course's side, from <code>labs/{LAB}</code>: the online update per key is highlighted.</>
            </Translated>
          }
        />
      ) : null}
      <Ask q={questionsSX35['x3/X3.5/q-excerpt-sdpav']} label={tr('Compare the two', 'So sánh hai bên')} />
      <Ask q={questionsSX35['x3/X3.5/q-exp2']} />

      <h2><Translated vi="PyTorch trên CUDA: một thứ tự ưu tiên và một submodule">PyTorch on CUDA: a priority order and a submodule</Translated></h2>
      <Translated
        vi={
          <p>
            <code>select_sdp_backend</code> đi qua một danh sách ưu tiên, mặc định là flash attention, memory-efficient attention,
            đường math, cuDNN, và một vị trí overrideable; cuDNN lên đầu khi <code>check_prefer_cudnn_attention()</code> nói
            vậy. "Each backend defines a use_&lt;backend&gt; function that returns true if the backend can be run with the given SDP
            parameters." Backend đầu tiên trả về true sẽ chạy.
          </p>
        }
      >
        <p>
          <code>select_sdp_backend</code> walks a priority list, by default flash attention, memory-efficient attention, the math
          path, cuDNN, and an overrideable slot; cuDNN moves to the front when <code>check_prefer_cudnn_attention()</code> says
          so. "Each backend defines a use_&lt;backend&gt; function that returns true if the backend can be run with the given SDP
          parameters." The first that returns true runs.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x3-facts.md B5; ch20-facts.md B5, D5)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Trên ROCm, PyTorch chạy AOTriton hoặc Composable Kernel cho attention, từ <code>native/transformers/hip/</code>. Trên MPS
              nó có kernel Metal riêng, <code>sdpa_vector_2pass_mps</code>, <code>sdpa_vector_fast_mps</code> và{' '}
              <code>sdpa_prefill_mps</code>, mà mã nguồn mở đầu bằng "Largely influeneced by" (sic) mã của MLX.
            </p>
          }
        >
          <p>
            On ROCm PyTorch runs AOTriton or Composable Kernel for attention, from <code>native/transformers/hip/</code>. On MPS it
            has its own Metal kernels, <code>sdpa_vector_2pass_mps</code>, <code>sdpa_vector_fast_mps</code> and{' '}
            <code>sdpa_prefill_mps</code>, whose source begins "Largely influeneced by" (sic) MLX's.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Rồi hãy tìm kernel flash. <code>native/transformers/cuda/flash_attn/</code> chứa bốn tệp: flash_api.cpp,
            flash_api.h, philox.cuh và static_switch.h. Không tệp nào trong số đó là kernel.
          </p>
        }
      >
        <p>
          Then look for the flash kernel. <code>native/transformers/cuda/flash_attn/</code> holds four files: flash_api.cpp,
          flash_api.h, philox.cuh and static_switch.h. None of them is the kernel.
        </p>
      </Translated>
      <Ask q={questionsSX35['x3/X3.5/q-flash-where']} />
      <Ask q={questionsSX35['x3/X3.5/q-order']} />

      <h2><Translated vi="Một quy tắc đọc mã">A reading rule</Translated></h2>
      <Translated
        vi={
          <p>
            Khi một dispatch kết thúc ở một thư mục chỉ chứa một tệp API và vài header, đó là một lớp đệm (shim): hãy đọc{' '}
            <code>.gitmodules</code> và các include của lớp đệm, rồi lần theo chúng tới submodule ở commit được ghim. Khi nó kết
            thúc ở một thư viện của hãng (cuDNN, cuBLAS), không có mã nguồn kernel để đọc, và thứ đáng đọc là bước kiểm tra điều
            kiện đứng trước nó: <code>supports_sdpa_cudnn</code>, <code>use_cudnn_for_decoding</code>,{' '}
            <code>use_&lt;backend&gt;</code>. Chính bước kiểm tra đó quyết định shape của bạn có chạy một kernel mà bạn đọc được hay không.
          </p>
        }
      >
        <p>
          When a dispatch ends in a directory that holds only an API file and some headers, it is a shim: read{' '}
          <code>.gitmodules</code> and the shim's includes, and follow them to the submodule at its pinned commit. When it ends
          in a vendor library (cuDNN, cuBLAS), there is no kernel source to read, and the useful reading is the eligibility
          check in front of it: <code>supports_sdpa_cudnn</code>, <code>use_cudnn_for_decoding</code>,{' '}
          <code>use_&lt;backend&gt;</code>. That check is what decides whether your shape runs a kernel you can read.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Quy tắc này là của chúng tôi, rút ra từ các trường hợp ở trên. Xem <a href={GITMODULES}>.gitmodules của PyTorch</a> ở commit được ghim.</p>}>
          <p>This rule is ours, drawn from the cases above. See <a href={GITMODULES}>PyTorch's .gitmodules</a> at the pinned commit.</p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Thử các shape">Try shapes</Translated></h2>
      <Translated
        vi={
          <p>
            Dự đoán trước: với một query và cache 100 key, những dự án nào chạy một kernel có mã nguồn bạn mở được? Rồi chuyển
            sang prefill.
          </p>
        }
      >
        <p>
          Predict first: with one query and a cache of 100 keys, which projects run a kernel whose source you can open? Then
          switch to prefill.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              supports_sdpa_cudnn và các bước kiểm tra use_&lt;backend&gt; của PyTorch là bất cứ giá trị nào bạn đặt: điều kiện
              thật của chúng chưa được đọc. Cách MLX CUDA chọn kernel vector 1-pass hay 2-pass, nó làm gì với nhiều query khi cuDNN
              từ chối, biến thể nào của llama.cpp chạy, và các ngưỡng của MPS được nêu tên, không được tính. Kích thước head và
              dtype bị bỏ qua.
            </p>
          }
        >
          <p>
            supports_sdpa_cudnn and PyTorch's use_&lt;backend&gt; checks are whatever you set: their real conditions were not
            read. How MLX CUDA picks its 1-pass or 2-pass vector kernel, what it does for several queries when cuDNN declines,
            which llama.cpp variant runs, and the MPS thresholds are named, not evaluated. Head sizes and dtypes are left out.
          </p>
        </Translated>
      </Simplification>
      <WhichAttention />
      <Ask q={questionsSX35['x3/X3.5/q-decode-100']} />
      <Ask q={questionsSX35['x3/X3.5/q-2pass']} />
      <Translated
        vi={
          <p>
            Con số 256 đáng xem lại lần nữa. Nó là bước mà mlx-lm dùng để mở rộng KV cache, mỗi lần {KV_CACHE_STEP} vị trí, được
            dùng lại làm ngưỡng. Nó không nói gì về GPU.
          </p>
        }
      >
        <p>
          The 256 is worth a second look. It is the step in which mlx-lm grows its KV cache, {KV_CACHE_STEP} positions at a
          time, reused as a threshold. It says nothing about the GPU.
        </p>
      </Translated>
      <StartReveal />

      <h2><Translated vi="Phần này nằm ở đâu bên dưới mô hình của bạn">Where this sits under your model</Translated></h2>
      <Translated
        vi={
          <p>
            Ở batch 1 trong decode, <code>F.scaled_dot_product_attention</code> là một query đối với toàn bộ cache: bài toán của
            lab 20.4 (<a href={href('ch20', 'labs')}>các lab của chương 20</a>). Trên AMD, llama.cpp chạy cùng mã nguồn như trên
            NVIDIA, PyTorch chuyển sang các thư viện khác, còn MLX thì không có gì. Khi đầu ra của một mô hình khác nhau giữa máy
            Mac của bạn và một máy CUDA, kernel attention có thể thậm chí không cùng thuật toán, nói gì đến cùng mã.
          </p>
        }
      >
        <p>
          At batch 1 in decode, <code>F.scaled_dot_product_attention</code> is one query against the whole cache: lab 20.4's
          problem (<a href={href('ch20', 'labs')}>chapter 20 labs</a>). On AMD, llama.cpp runs the same source as on NVIDIA,
          PyTorch switches to other libraries, and MLX has nothing. When a model's output differs between your Mac and a CUDA
          box, the attention kernel may not even be the same algorithm, let alone the same code.
        </p>
      </Translated>
    </>
  );
}
