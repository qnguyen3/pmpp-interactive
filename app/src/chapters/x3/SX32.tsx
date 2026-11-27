import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabExcerpt } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { MECHANISMS, PROJECT_NAME, BACKEND_NAME, VARIANT_MECHANISMS, gradeSort, type Mechanism } from '../../sim/backendmap';
import {
  BLOCK_CHOICES,
  N_READS_CHOICES,
  PATH_LABEL,
  STRIDE_GROUPS,
  STYLES,
  WARP,
  divergentWarps,
  footprint,
  launch,
  pathCounts,
  tailSummary,
  type Launch,
  type LaunchCfg,
  type Path,
  type Style,
} from '../../sim/elemshapes';
import type { Excerpt as ExcerptData } from '../../sim/x3sources';
import { Excerpt } from './Excerpt';
import { OpMap } from './OpMap';

// Section X3.2: elementwise and gather ops in the three projects, read as chapter 2's kernel
// with chapter 6's coarsening and vector loads, and how each project picks a variant. The
// explorer runs on sim/elemshapes.ts, the sort on VARIANT_MECHANISMS in sim/backendmap.ts. The
// excerpts are checked against notes/today/x3-facts.md by excerpts-g1.test.ts.

const FACTS = 'notes/today/x3-facts.md';
const CHECKED = (items: string) =>
  tr(`2026-11-20 in the facts pass (${FACTS}, ${items})`, `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, ${items})`);

export const EXCERPTS_SX32: Record<string, ExcerptData> = {
  unary: {
    id: 'x32-unary', project: 'llama.cpp', path: 'ggml/src/ggml-cuda/unary.cu', lines: [116, 127], factsItem: 'B1', lang: 'cuda',
    code: `template <float (*op)(float), typename T>
static __global__ void unary_op_kernel(const T * x, T * dst, const int k) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    ggml_cuda_pdl_sync();
    dst[i] = ggml_cuda_cast<T>(op(ggml_cuda_cast<float>(x[i])));
}`,
  },
  unaryMetal: {
    id: 'x32-unary-metal', project: 'llama.cpp', path: 'ggml/src/ggml-metal/kernels/unary.metal', lines: [3, 134], factsItem: 'B1', lang: 'metal',
    code: `constant short FC_unary_op [[function_constant(FC_UNARY + 0)]];
...
if (FC_OP == OP_UNARY_NUM_SILU) {
    dst_ptr[i0] = ggml_metal_unary_cast<T>(x / (1 + exp(-x)));
}`,
  },
  swiglu: {
    id: 'x32-swiglu-loop', project: 'llama.cpp', path: 'ggml/src/ggml-metal/kernels/unary.metal', lines: [290, 309], factsItem: 'B1', lang: 'metal',
    code: `for (int i0 = tgpig.x*ntg.x + tpitg.x; i0 < args.ne0; i0 += ntg.x*ntgs.x)`,
  },
  binaryVv: {
    id: 'x32-binary-vv', project: 'mlx', path: 'mlx/backend/cuda/binary/binary.cuh', lines: [93, 118], factsItem: 'B1', lang: 'cuda',
    code: `template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ __launch_bounds__(BINARY_MAX_BLOCK_DIM) void binary_vv(
    const In* a, const In* b, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();
  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = Op{}(a[i], b[i]);
    }
  } else {
    auto a_vec = load_vector<N_READS>(a, index);
    auto b_vec = load_vector<N_READS>(b, index);
...
    store_vector<N_READS>(out, index, out_vec);
  }
}`,
  },
  loadVector: {
    id: 'x32-load-vector', project: 'mlx', path: 'mlx/backend/cuda/device/utils.cuh', lines: [33, 70], factsItem: 'B1', lang: 'cuda',
    code: `struct alignas(sizeof(T) * N) AlignedVector
...
if (is_aligned<N>(ptr)) {
  auto* from = reinterpret_cast<const AlignedVector<T, N>*>(ptr);
  return from[offset];
} else {
...
}`,
  },
  siluTorch: {
    id: 'x32-silu-torch', project: 'pytorch', path: 'aten/src/ATen/native/cuda/ActivationSiluKernel.cu', lines: [30, 33], factsItem: 'A3', lang: 'cuda',
    code: `gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
  using opmath_t = at::opmath_type<scalar_t>;
  const opmath_t x_acc = static_cast<opmath_t>(x);
  return x_acc / (opmath_t(1) + ::exp(-x_acc));
});`,
  },
  embedTorch: {
    id: 'x32-embedding-torch', project: 'pytorch', path: 'aten/src/ATen/native/Embedding.cpp', lines: [35, 55], factsItem: 'B7', lang: 'cpp',
    code: `return weight.index_select(0, indices.reshape(-1)).view_symint(size);`,
  },
};

export const questionsSX32: Record<string, Question> = {
  'x3/X3.2/q-read-unary': {
    kind: 'choice',
    id: 'x3/X3.2/q-read-unary',
    prompt: (
      <Translated vi={<><code>unary_op_kernel</code> là kernel nào của khóa học, và nó làm gì khác đi?</>}>
        <>Which course kernel is <code>unary_op_kernel</code>, and what does it do differently?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Kernel theo từng phần tử của lab 2.3, với op là một tham số template: mỗi op một ngôi (unary) một bản khởi tạo.">
            Lab 2.3’s elementwise kernel, with the op as a template parameter: one instantiation per unary op.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Kernel gộp việc cho luồng (thread coarsening) của lab 6.2: mỗi luồng xử lý nhiều phần tử.">
            Lab 6.2’s coarsened kernel: each thread handles several elements.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi luồng tính i một lần và ghi dst[i]: một phần tử, như ở lab 2.3.">
            Each thread computes i once and writes dst[i]: one element, as in lab 2.3.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Softmax theo hàng của lab 10.3, với phép rút gọn bị bỏ đi.">
            Lab 10.3’s row softmax, with the reduction removed.
          </Translated>
        ),
        why: (
          <Translated vi="Không có hàng nào và không có phép rút gọn nào: đó là một chỉ số phẳng với một lệnh kiểm tra biên.">
            There is no row and no reduction: it is a flat index with a guard.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Kernel của lab 2.3, nhưng có vòng lặp grid-stride để một số khối cố định phủ được bất kỳ số phần tử nào.">
            Lab 2.3’s kernel, but with a grid-stride loop so that a fixed number of blocks covers any number of elements.
          </Translated>
        ),
        why: (
          <Translated vi="Không có vòng lặp nào. Lần khởi chạy tính đủ số khối cho k phần tử, như lab 2.3.">
            There is no loop. The launch computes enough blocks for k elements, as lab 2.3 does.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Chỉ số toàn cục, kiểm tra biên, một phần tử: ba dòng của lab 2.3. Op là <code>template &lt;float (*op)(float), typename T&gt;</code>,
            nên <code>ggml_cuda_op_silu</code> là <code>ggml_cuda_op_unary&lt;op_silu&gt;</code>, và mỗi op một ngôi là một bản
            khởi tạo riêng được biên dịch vào thư viện. Hai lời gọi <code>pdl</code> là điểm khác còn lại; chúng không làm gì trừ
            khi programmatic dependent launch được bật.
          </p>
        }
      >
        <p>
          Global index, guard, one element: lab 2.3’s three lines. The op is <code>template &lt;float (*op)(float), typename T&gt;</code>,
          so <code>ggml_cuda_op_silu</code> is <code>ggml_cuda_op_unary&lt;op_silu&gt;</code>, and every unary op is a separate
          instantiation compiled into the library. The two <code>pdl</code> calls are the other difference; they do nothing unless
          programmatic dependent launch is on.
        </p>
      </Translated>
    ),
    source: `${FACTS} B1; lab 2.3`,
    vi: { source: `${FACTS} B1; bài lab 2.3` },
  },
  'x3/X3.2/q-read-binvv': {
    kind: 'choice',
    id: 'x3/X3.2/q-read-binvv',
    prompt: (
      <Translated vi={<><code>vecAdd4Kernel</code> của lab 6.2 và <code>binary_vv</code> của MLX trông giống nhau. <code>binary_vv</code> thêm gì?</>}>
        <>Lab 6.2’s <code>vecAdd4Kernel</code> and MLX’s <code>binary_vv</code> look alike. What does <code>binary_vv</code> add?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một phép kiểm tra căn chỉnh lúc chạy: lần nạp vector chỉ được dùng khi con trỏ được căn chỉnh phù hợp.">
            A run-time alignment test: the vector load is taken only when the pointer is aligned for it.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Một lần khởi chạy thứ hai cho phần đuôi, để không luồng nào của lần khởi chạy chính phải rẽ nhánh ở cuối mảng.">
            A second launch for the tail, so that no thread of the main launch has to branch on the end of the array.
          </Translated>
        ),
        why: (
          <Translated vi="Không cái nào khởi chạy hai lần: cả hai xử lý phần dở dang trong luồng cuối bằng một vòng lặp vô hướng.">
            Neither launches twice: both handle the partial chunk in the last thread with a scalar loop.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Bộ nhớ chia sẻ (shared memory): mỗi khối đưa các vector của nó vào đó trước, để các lần nạp từ bộ nhớ toàn cục được hợp nhất.">
            Shared memory: each block stages its vectors there first, so that its global loads coalesce.
          </Translated>
        ),
        why: (
          <Translated vi="Cả hai đều không dùng bộ nhớ chia sẻ: mỗi luồng đọc phần của mình thẳng từ bộ nhớ toàn cục (global memory).">
            There is no shared memory in either: each thread reads its own chunk straight from global memory.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không gì cả: đó là cùng một kernel với một template cho op.">
            Nothing: it is the same kernel with a template for the op.
          </Translated>
        ),
        why: (
          <Translated vi="Template là mới, nhưng phép kiểm tra căn chỉnh cũng vậy, và lab 6.2 không có nó.">
            The template is new, but so is the alignment test, which lab 6.2 does not have.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cả hai giao cho mỗi luồng N phần tử liên tiếp, dùng một lần nạp vector cho cả một phần trọn vẹn, và hoàn tất phần cuối
            dở dang từng phần tử một trong cùng luồng đó. Lab 6.2 giả định các con trỏ của nó được căn chỉnh 16 byte, điều này đúng
            với một buffer <code>cudaMalloc</code> mới cấp phát. MLX không giả định vậy:{' '}
            <code>load_vector</code> kiểm tra <code>is_aligned&lt;N&gt;(ptr)</code> trước và lùi về các lần nạp vô hướng.
          </p>
        }
      >
        <p>
          Both give each thread N consecutive elements, take a vector load for a whole chunk, and finish a partial last chunk one
          element at a time in the same thread. Lab 6.2 assumes its pointers are 16-byte aligned, which holds for a fresh{' '}
          <code>cudaMalloc</code> buffer. MLX does not assume it:{' '}
          <code>load_vector</code> tests <code>is_aligned&lt;N&gt;(ptr)</code> first and falls back to scalar loads.
        </p>
      </Translated>
    ),
    source: `${FACTS} B1; notes/today/infra-facts.md A6; lab 6.2`, // i18n-ignore: citation
  },
  'x3/X3.2/q-read-fconst': {
    kind: 'choice',
    id: 'x3/X3.2/q-read-fconst',
    prompt: (
      <Translated vi={<>Trong <code>unary.metal</code> của llama.cpp, thứ gì đóng vai trò mà tham số template <code>op</code> đóng trong <code>unary.cu</code>?</>}>
        <>In llama.cpp’s <code>unary.metal</code>, what plays the part that the template parameter <code>op</code> plays in <code>unary.cu</code>?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Hằng hàm (function constant) FC_unary_op, được kiểm tra bên trong thân kernel duy nhất.">
            The function constant FC_unary_op, tested inside the one kernel body.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Một tham số template của Metal, y hệt như trong CUDA.">
            A Metal template parameter, exactly as in CUDA.
          </Translated>
        ),
        why: (
          <Translated vi="MSL có template, và llama.cpp dùng chúng cho kiểu dữ liệu, nhưng ở đây bản thân op là một hằng hàm.">
            MSL has templates, and llama.cpp uses them for types, but the op itself is a function constant here.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một đối số kernel đọc từ buffer args lúc chạy.">
            A kernel argument read from the args buffer at run time.
          </Translated>
        ),
        why: (
          <Translated vi="Một đối số có thể thay đổi theo từng lần dispatch; một hằng hàm được cố định khi pipeline được tạo.">
            An argument could change per dispatch; a function constant is fixed when the pipeline is made.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một macro tiền xử lý được đặt khi thư viện được biên dịch.">
            A preprocessor macro set when the library is compiled.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là cách tiếp cận ubershader mà hằng hàm thay thế, theo đúng lời của đặc tả.">
            That is the ubershader approach that function constants replace, in the spec’s own words.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một template, <code>kernel_unary_impl</code>, phục vụ mọi op một ngôi, và <code>if (FC_OP == OP_UNARY_NUM_SILU)</code>{' '}
            chọn phần thân. Giá trị được đặt khi pipeline được tạo, nên mỗi pipeline là một biến thể làm cho một op.
          </p>
        }
      >
        <p>
          One template, <code>kernel_unary_impl</code>, serves every unary op, and <code>if (FC_OP == OP_UNARY_NUM_SILU)</code>{' '}
          chooses the body. The value is set when the pipeline is created, so each pipeline is a variant made for one op.
        </p>
      </Translated>
    ),
    source: `${FACTS} B1, C1`,
  },
  'x3/X3.2/q-tail': {
    kind: 'numeric',
    id: 'x3/X3.2/q-tail',
    prompt: (
      <Translated
        vi={
          <>
            <code>binary_vv</code> của MLX với N_READS = 4 chạy trên 1002 phần tử. Giả sử lần khởi chạy có ceil(1002 / 4) = 251
            luồng (lượt kiểm tra dữ kiện không đọc phần tính toán khởi chạy của MLX). Có bao nhiêu phần tử đi qua vòng lặp vô hướng
            ở phần đuôi?
          </>
        }
      >
        <>
          MLX’s <code>binary_vv</code> with N_READS = 4 runs on 1002 elements. Assume the launch has ceil(1002 / 4) = 251
          threads (the facts did not read MLX’s launch computation). How many elements go through the scalar tail loop?
        </>
      </Translated>
    ),
    answer: 2,
    unit: 'elements',
    explain: (
      <Translated
        vi={
          <p>
            Luồng 250 có (250 + 1) × 4 = 1004 &gt; 1002, nên nó đi vào nhánh phần đuôi và lặp qua các phần tử 1000 và 1001.
            Các luồng 0 đến 249 có (index + 1) × 4 ≤ 1000 và đi theo đường vector. Phần đuôi luôn chứa n mod N_READS phần tử, trong
            một luồng. Trình khám phá bên dưới cho thấy điều đó với mọi n.
          </p>
        }
      >
        <p>
          Thread 250 has (250 + 1) × 4 = 1004 &gt; 1002, so it takes the tail branch and loops over elements 1000 and 1001.
          Threads 0 to 249 have (index + 1) × 4 ≤ 1000 and take the vector path. The tail always holds n mod N_READS elements, in
          one thread. The explorer below shows it for any n.
        </p>
      </Translated>
    ),
    source: `${FACTS} B1; sim/elemshapes.ts`, // i18n-ignore: citation
    vi: { unit: 'phần tử' },
  },
  'x3/X3.2/q-misaligned': {
    kind: 'choice',
    id: 'x3/X3.2/q-misaligned',
    prompt: (
      <Translated vi={<>Cùng lời gọi đó, N_READS = 4, nhưng trên một view bắt đầu lệch một float so với buffer của nó. Điều gì thay đổi?</>}>
        <>The same call, N_READS = 4, but on a view that starts one float into its buffer. What changes?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="is_aligned thất bại ở mọi luồng, nên mỗi luồng nạp bốn phần tử của nó từng cái một; kết quả không đổi.">
            is_aligned fails in every thread, so each loads its four elements one at a time; the result is unchanged.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Kết quả sai: các lần nạp vector đọc vắt qua điểm bắt đầu của view.">
            The results are wrong: the vector loads read across the view’s start.
          </Translated>
        ),
        why: (
          <Translated vi="Phép kiểm tra tồn tại chính để ngăn điều đó. Không lần nạp vector lệch căn chỉnh nào được phát ra.">
            The test exists to stop that. No misaligned vector load is issued.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Kernel bị lỗi ở lần nạp lệch căn chỉnh đầu tiên.">
            The kernel faults on the first misaligned load.
          </Translated>
        ),
        why: (
          <Translated vi="Nó không bao giờ thực hiện lần nạp như vậy: vòng lặp vô hướng chạy thay thế.">
            It never makes one: the scalar loop runs instead.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ luồng cuối đổi đường, vì độ lệch chỉ dời chỗ phần đuôi bắt đầu còn các phần khác vẫn trọn vẹn.">
            Only the last thread changes path, because an offset only moves where the tail starts and the other chunks stay whole.
          </Translated>
        ),
        why: (
          <Translated vi="Độ lệch dời mọi phần, nên mọi lần nạp vector đều sẽ lệch căn chỉnh.">
            The offset moves every chunk, so every vector load would be misaligned.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>load_vector</code> chỉ diễn giải lại con trỏ thành một <code>AlignedVector</code> sau khi <code>is_aligned&lt;N&gt;(ptr)</code>.
            Bị dời 4 byte, không điểm bắt đầu phần nào được căn chỉnh 16 byte, nên mọi luồng đi vào vòng lặp vô hướng. Mỗi luồng vẫn
            sở hữu bốn phần tử như cũ, nên trong một lệnh nạp, một warp giờ đọc cứ mỗi float thứ tư thay vì một dải liền mạch. Hãy
            thử độ lệch trong trình khám phá.
          </p>
        }
      >
        <p>
          <code>load_vector</code> reinterprets the pointer as an <code>AlignedVector</code> only after <code>is_aligned&lt;N&gt;(ptr)</code>.
          Shifted by 4 bytes, no chunk start is 16-byte aligned, so every thread takes the scalar loop. Each thread still owns the
          same four elements, so within one load instruction a warp now reads every fourth float rather than one unbroken run. Try
          the offset in the explorer.
        </p>
      </Translated>
    ),
    source: `${FACTS} B1; sim/elemshapes.ts`, // i18n-ignore: citation
  },
  'x3/X3.2/q-pdl': {
    kind: 'choice',
    id: 'x3/X3.2/q-pdl',
    prompt: (
      <Translated vi={<><code>ggml_cuda_pdl_lc()</code> và <code>ggml_cuda_pdl_sync()</code> làm gì trong một bản build HIP của llama.cpp?</>}>
        <>What do <code>ggml_cuda_pdl_lc()</code> and <code>ggml_cuda_pdl_sync()</code> do in a HIP build of llama.cpp?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không gì cả: programmatic dependent launch chỉ có trên CUDA, và các lời gọi không làm gì trừ khi nó được bật.">
            Nothing: programmatic dependent launch is CUDA-only, and the calls are no-ops unless it is enabled.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chúng trở thành các lời gọi dependent-launch riêng của HIP thông qua vendors/hip.h.">
            They become HIP’s own dependent-launch calls through vendors/hip.h.
          </Translated>
        ),
        why: (
          <Translated vi="Macro PDL chỉ được định nghĩa khi GGML_USE_HIP không được định nghĩa (ch18 D3).">
            The PDL macro is defined only when GGML_USE_HIP is not (ch18 D3).
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chúng là các rào chắn (barrier) mà mọi luồng của khối phải tới được.">
            They are barriers that every thread of the block must reach.
          </Translated>
        ),
        why: (
          <Translated vi="Chúng liên quan đến thứ tự giữa các kernel liên tiếp, không phải giữa các luồng.">
            They are about the order between consecutive kernels, not between threads.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chúng không biên dịch được, nên bản build HIP bỏ qua unary.cu.">
            They fail to compile, so the HIP build leaves out unary.cu.
          </Translated>
        ),
        why: (
          <Translated vi="Bản build HIP biên dịch mọi ggml-cuda/*.cu, kể cả unary.cu.">
            The HIP build compiles every ggml-cuda/*.cu, unary.cu included.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            PDL cho phép một kernel phụ thuộc khởi chạy trước khi kernel mà nó phụ thuộc, trong cùng stream, kết thúc. llama.cpp chỉ định nghĩa{' '}
            <code>GGML_CUDA_USE_PDL</code> khi <code>#if !defined(GGML_USE_HIP) &amp;&amp; …</code>, và các hàm hỗ trợ phía device chỉ
            có tác dụng trên Hopper trở về sau. Trên HIP, hai dòng đó biên dịch thành không có gì, và kernel chính là kernel của lab 2.3.
          </p>
        }
      >
        <p>
          PDL lets a dependent kernel launch before the kernel it depends on, in the same stream, has finished. llama.cpp defines{' '}
          <code>GGML_CUDA_USE_PDL</code> only <code>#if !defined(GGML_USE_HIP) &amp;&amp; …</code>, and the device helpers act only on
          Hopper or later. On HIP the two lines compile to nothing, and the kernel is lab 2.3’s.
        </p>
      </Translated>
    ),
    source: `${FACTS} B1; notes/today/ch18-facts.md D3`, // i18n-ignore: citation
  },
  'x3/X3.2/q-fconst': {
    kind: 'choice',
    id: 'x3/X3.2/q-fconst',
    prompt: (
      <Translated
        vi={
          <>
            Kernel một ngôi CUDA của llama.cpp biến op thành một tham số template; kernel Metal của nó biến op thành một hằng hàm.
            Hằng hàm đem lại điều gì trên Metal?
          </>
        }
      >
        <>
          llama.cpp’s CUDA unary kernel makes the op a template parameter; its Metal kernel makes it a function constant. What
          does the function constant buy on Metal?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một thư viện đã biên dịch; mỗi biến thể được cố định khi pipeline state của nó được tạo, chứ không biên dịch trước (offline).">
            One compiled library; each variant is fixed when its pipeline state is created, not compiled offline.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Op có thể thay đổi giữa các luồng lúc chạy, nên một lần dispatch có thể trộn nhiều op trong một lưới.">
            The op can change from thread to thread at run time, so one dispatch can mix several ops in one grid.
          </Translated>
        ),
        why: (
          <Translated vi="Một hằng hàm có một giá trị cho mỗi pipeline, giống nhau với mọi luồng.">
            A function constant has one value per pipeline, the same for every thread.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Toán nhanh hơn: hằng hàm bật fast::exp.">
            Faster math: function constants enable fast::exp.
          </Translated>
        ),
        why: (
          <Translated vi="Chế độ toán học là một tùy chọn biên dịch riêng (ch05 C7).">
            The math mode is a compile option of its own (ch05 C7).
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó cho phép kernel CUDA và kernel Metal dùng chung một tệp nguồn, với op được chọn bằng một hằng ở mỗi bên.">
            It lets the CUDA and Metal kernels share one source file, with the op chosen by a constant on each side.
          </Translated>
        ),
        why: (
          <Translated vi="Chúng là các tệp riêng: unary.cu và unary.metal.">
            They are separate files: unary.cu and unary.metal.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            MSL 4.1, mục 5.8: hằng hàm “provide the same ease of use as preprocessor macros but moves the generation of
            the specific variants to the creation of the pipeline state, so you don’t have to compile the variants offline.” Giá
            trị đến từ một <code>MTLFunctionDescriptor</code>, không phải từ mã nguồn.
          </p>
        }
      >
        <p>
          MSL 4.1, section 5.8: function constants “provide the same ease of use as preprocessor macros but moves the generation of
          the specific variants to the creation of the pipeline state, so you don’t have to compile the variants offline.” The
          value comes from an <code>MTLFunctionDescriptor</code>, not from the source.
        </p>
      </Translated>
    ),
    source: `${FACTS} B1, C1 (MSL 4.1 §5.8, p. 190)`, // i18n-ignore: citation
  },
  'x3/X3.2/q-silu-bits': {
    kind: 'choice',
    id: 'x3/X3.2/q-silu-bits',
    prompt: (
      <Translated vi="SiLU trên Metal của llama.cpp và SiLU trên Metal của MLX có cho đầu ra giống nhau đến từng bit với cùng một đầu vào float không?">
        <>Will llama.cpp’s Metal SiLU and MLX’s Metal SiLU give bit-identical outputs for the same float input?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không nhất thiết: x / (1 + exp(−x)) và x · sigmoid(x) có thể làm tròn khác nhau, và chế độ toán học của Metal còn cho thêm tự do.">
            Not necessarily: x / (1 + exp(−x)) and x · sigmoid(x) can round differently, and Metal’s math mode adds more freedom.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có: cả hai đều tính SiLU, và SiLU trên float có một đáp án làm tròn đúng duy nhất, mà mọi kernel float cẩn thận đều phải tạo ra.">
            Yes: both compute SiLU, and float SiLU has one correctly rounded answer, which any careful float kernel must produce.
          </Translated>
        ),
        why: (
          <Translated vi="Giá trị chính xác có một đáp án; hai công thức float có thể làm tròn về hai giá trị lân cận khác nhau của nó.">
            The exact value has one answer; two float formulas can round to different neighbours of it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Có, vì cả hai chạy trên cùng GPU, qua cùng trình biên dịch Metal, ở cùng chế độ toán học.">
            Yes, because both run on the same GPU, through the same Metal compiler, in the same math mode.
          </Translated>
        ),
        why: (
          <Translated vi="Cùng phần cứng, khác phép toán: một phép chia so với một phép nhân với sigmoid.">
            Same hardware, different operations: a division against a multiply by a sigmoid.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không: hai công thức luôn khác nhau ở bit cuối, bất kể đầu vào.">
            No: the two formulas always differ in the last bit, whatever the input.
          </Translated>
        ),
        why: (
          <Translated vi="“Không nhất thiết khớp” không phải là “luôn khác”. Thường thì chúng sẽ khớp. Ở đây không ai đo điều đó.">
            “Need not agree” is not “always differ”. Often they will match. Nobody measured it here.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hai công thức bằng nhau trong số học chính xác. Trên float, phép chia x cho 1 + exp(−x) và tích của x với một sigmoid được
            làm tròn riêng là hai chuỗi phép toán làm tròn khác nhau. Trên Metal, chế độ toán học mặc định cho trình biên dịch thêm
            tự do. Chúng tôi chưa đo chúng khác nhau thường xuyên đến mức nào.
          </p>
        }
      >
        <p>
          The two are equal in exact arithmetic. In float, a division of x by 1 + exp(−x) and a product of x with a separately
          rounded sigmoid are different sequences of rounded operations. On Metal the default math mode allows the compiler
          further liberties. We have not measured how often they differ.
        </p>
      </Translated>
    ),
    source: `${FACTS} B1; notes/today/ch05-facts.md C7`, // i18n-ignore: citation
  },
  'x3/X3.2/q-getrows-quant': {
    kind: 'choice',
    id: 'x3/X3.2/q-getrows-quant',
    prompt: (
      <Translated vi="Vì sao GET_ROWS của llama.cpp trên một bảng embedding lượng tử hóa không thể là một phép sao chép hàng đơn thuần?">
        <>Why can llama.cpp’s GET_ROWS on a quantised embedding table not be a plain row copy?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Các hàng được lưu là các khối lượng tử hóa, nên kernel giải lượng tử hóa từng hàng sang float khi gather nó.">
            The stored rows are quantised blocks, so the kernel dequantises each row to float as it gathers it.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Các hàng lượng tử hóa được lưu theo thứ tự cột (column-major), nên một hàng không liền mạch.">
            Quantised rows are stored column-major, so a row is not contiguous.
          </Translated>
        ),
        why: (
          <Translated vi="Không dữ kiện nào nói vậy. Khác biệt nằm ở cách mã hóa, không phải bố cục.">
            Nothing in the facts says so. The difference is the encoding, not the layout.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các token id phải được sắp xếp trước, để các hàng được đọc theo thứ tự.">
            The token ids must be sorted first, so that rows are read in order.
          </Translated>
        ),
        why: (
          <Translated vi="Một phép gather đọc các hàng theo bất kỳ thứ tự nào; mỗi id một hàng đầu ra.">
            A gather reads rows in any order; one output row per id.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Có thể: đầu ra vẫn ở dạng lượng tử hóa và các kernel sau giải lượng tử hóa nó.">
            It can: the output stays quantised and later kernels dequantise it.
          </Translated>
        ),
        why: (
          <Translated vi="Đầu ra của GET_ROWS đi vào các kernel float, và các dữ kiện nói các hàng được giải lượng tử hóa ngay khi đọc.">
            The output of GET_ROWS feeds float kernels, and the facts say rows are dequantised on the fly.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>getrows.cu</code> có <code>k_get_rows</code> cho các hàng lượng tử hóa bên cạnh <code>k_get_rows_float</code> và{' '}
            <code>k_get_rows_float_vec</code>; trên Metal, <code>kernel_get_rows_q</code> nằm cạnh <code>kernel_get_rows_f</code>.
            Kernel lượng tử hóa giải mã từng khối khi sao chép. Học phần <a href={href('x2', 'start')}>X2</a> xây các định dạng khối đó.
          </p>
        }
      >
        <p>
          <code>getrows.cu</code> has <code>k_get_rows</code> for quantised rows next to <code>k_get_rows_float</code> and{' '}
          <code>k_get_rows_float_vec</code>; on Metal <code>kernel_get_rows_q</code> sits next to <code>kernel_get_rows_f</code>.
          The quantised one decodes each block as it copies. Module <a href={href('x2', 'start')}>X2</a> builds those block formats.
        </p>
      </Translated>
    ),
    source: `${FACTS} B7`,
  },
};

// ---------------------------------------------------------------- course kernels, loaded from the labs
// ---------------------------------------------------------------- the launch-shapes explorer
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)', fontFamily: 'var(--sans)', fontSize: '0.84rem' };
const CW = 13;
/** Fewer elements per row below 600 px, so the cells stay large enough to read. */
function usePerRow(): number {
  const q = '(max-width: 600px)'; // i18n-ignore: media query
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && window.matchMedia(q).matches);
  useEffect(() => {
    const m = window.matchMedia(q);
    const on = () => setNarrow(m.matches);
    m.addEventListener('change', on);
    return () => m.removeEventListener('change', on);
  }, []);
  return narrow ? 20 : 50;
}

function fillOf(path: Path | null): { fill: string; stroke: string; dash?: string; width?: number } {
  switch (path) {
    case 'vector':
      return { fill: 'var(--global)', stroke: 'var(--global)' };
    case 'scalar':
      return { fill: 'var(--global-wash)', stroke: 'var(--global)' };
    case 'tail':
      return { fill: 'var(--global-wash)', stroke: 'var(--ink)', width: 2 };
    case 'misaligned':
      return { fill: 'var(--global-wash)', stroke: 'var(--global)', dash: '2 2' };
    default:
      return { fill: 'var(--cell)', stroke: 'var(--rule)' };
  }
}

function Swatch({ path }: { path: Path }) {
  const f = fillOf(path);
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', marginRight: '0.9rem' }}>
      <svg width="14" height="14" aria-hidden="true">
        <rect x="1" y="1" width="12" height="12" rx="1" fill={f.fill} stroke={f.stroke} strokeWidth={f.width ?? 1} strokeDasharray={f.dash} />
      </svg>
      {PATH_LABEL[path]}
    </span>
  );
}

function Elements({ l }: { l: Launch }) {
  const PER_ROW = usePerRow();
  const n = l.cfg.n;
  const rows = Math.ceil(n / PER_ROW);
  return (
    <svg viewBox={`0 0 ${PER_ROW * CW + 40} ${rows * (CW + 4) + 18}`} style={{ width: '100%', maxWidth: 720, display: 'block' }} role="img"
      aria-label={tr('Elements, coloured by the path that wrote them', 'Các phần tử, tô màu theo đường đã ghi chúng')} data-part="elements">
      <text x={0} y={11} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi={<>các phần tử 0 đến {n - 1}, {PER_ROW} mỗi hàng</>}>elements 0 to {n - 1}, {PER_ROW} per row</Translated>
      </text>
      {l.pathOf.map((p, e) => {
        const r = Math.floor(e / PER_ROW);
        const c = e % PER_ROW;
        const f = fillOf(p);
        const newOwner = e > 0 && l.writer[e] !== l.writer[e - 1] && c !== 0;
        return (
          <rect key={e} x={36 + c * CW + (newOwner ? 1 : 0)} y={18 + r * (CW + 4)} width={CW - 2 - (newOwner ? 1 : 0)} height={CW} rx="1"
            fill={f.fill} stroke={f.stroke} strokeWidth={f.width ?? 0.8} strokeDasharray={f.dash} data-path={p ?? 'none'} data-writer={l.writer[e]} />
        );
      })}
      {Array.from({ length: rows }, (_, r) => (
        <text key={r} x={0} y={18 + r * (CW + 4) + 10} fontSize="10" style={{ fill: 'var(--ink-3)' }}>{r * PER_ROW}</text>
      ))}
    </svg>
  );
}

function Warps({ l }: { l: Launch }) {
  const TW = 18;
  const div = new Set(divergentWarps(l));
  return (
    <svg viewBox={`0 0 ${WARP * TW + 150} ${l.warps.length * 22 + 18}`} style={{ width: '100%', maxWidth: 720, display: 'block' }} role="img"
      aria-label={tr('Threads by warp, coloured by path', 'Các luồng theo warp, tô màu theo đường')} data-part="warps">
      <text x={0} y={11} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi={<>các luồng, {WARP} mỗi hàng: một warp CUDA</>}>threads, {WARP} per row: a CUDA warp</Translated>
      </text>
      {l.warps.map((w) => (
        <g key={w.w} data-warp={w.w} data-diverges={w.diverges ? '1' : '0'}>
          <text x={0} y={18 + w.w * 22 + 13} fontSize="10" style={{ fill: 'var(--ink-3)' }}><Translated vi={<>warp {w.w}</>}>warp {w.w}</Translated></text>
          {l.plans.slice(w.threads[0], w.threads[1] + 1).map((p) => {
            const f = fillOf(p.path === 'idle' ? null : p.path);
            return <rect key={p.t} x={50 + (p.t % WARP) * TW} y={18 + w.w * 22} width={TW - 3} height={17} rx="1.5" fill={f.fill} stroke={f.stroke}
              strokeWidth={f.width ?? 0.8} strokeDasharray={f.dash} />;
          })}
          {div.has(w.w) ? <text x={50 + WARP * TW + 6} y={18 + w.w * 22 + 13} fontSize="10.5"><Translated vi="nhánh tách">branch splits</Translated></text> : null}
        </g>
      ))}
    </svg>
  );
}

function LaunchShapes() {
  const [style, setStyle] = useState<Style>('binary_vv');
  const [n, setN] = useState(150);
  const [block, setBlock] = useState(32);
  const [nReads, setNReads] = useState(4);
  const [offset, setOffset] = useState(0);
  const cfg: LaunchCfg = { style, n, block, nReads, offset };
  const l = useMemo(() => launch(cfg), [style, n, block, nReads, offset]); // eslint-disable-line react-hooks/exhaustive-deps
  const counts = pathCounts(l);
  const tail = tailSummary(l);
  const div = divergentWarps(l);
  const vv = style === 'binary_vv';
  const paths: Path[] = vv ? ['vector', 'tail', 'misaligned', 'idle'] : ['scalar', 'idle'];
  const shownWarps = l.warps.slice(0, 3);

  return (
    <Bench
      title={tr('Launch shapes', 'Hình dạng khởi chạy')}
      task={tr(
        'Pick a kernel style, then move n, the block size and the pointer offset. Watch which thread writes what, and by which path.',
        'Chọn một kiểu kernel, rồi thay đổi n, kích thước khối và độ lệch con trỏ. Quan sát luồng nào ghi gì, và theo đường nào.',
      )}
      foot={
        <Translated
          vi={
            <>
              Vẽ từ sim/elemshapes.ts. Một khoảng hở tách các phần tử của một luồng với luồng kế tiếp. Một warp được vẽ rộng 32, như
              trên CUDA; Metal báo độ rộng SIMD group theo từng pipeline. Chỉ các lần nạp của đầu vào thứ nhất được liệt kê. Lần khởi
              chạy binary_vv ở đây là của chúng tôi: ceil(n / N_READS) luồng làm tròn lên thành số khối nguyên. Kiểu grid-stride dùng{' '}
              {STRIDE_GROUPS} nhóm, cũng là lựa chọn của chúng tôi.
            </>
          }
        >
          <>
            Drawn from sim/elemshapes.ts. A gap separates one thread’s elements from the next. A warp is drawn 32 wide, as on
            CUDA; Metal reports its SIMD-group width per pipeline. Only loads of the first input are listed. The binary_vv launch here is ours: ceil(n /
            N_READS) threads rounded up to whole blocks. The grid-stride style uses {STRIDE_GROUPS} groups, also our choice.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Kiểu kernel">Kernel style</Translated>
          <span className="seg" role="group" aria-label={tr('Kernel style', 'Kiểu kernel')} style={{ flexWrap: 'wrap' }}>
            {STYLES.map((s) => (
              <button key={s.id} aria-pressed={style === s.id} onClick={() => setStyle(s.id)} title={s.who} data-style={s.id}>{s.label}</button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          <Translated vi="Kích thước khối">Block size</Translated>
          <span className="seg" role="group" aria-label={tr('Block size', 'Kích thước khối')}>
            {BLOCK_CHOICES.map((b) => <button key={b} aria-pressed={block === b} onClick={() => setBlock(b)}>{b}</button>)}
          </span>
        </span>
        {vv ? (
          <span style={FIELD}>
            N_READS
            <span className="seg" role="group" aria-label="N_READS">
              {N_READS_CHOICES.map((r) => <button key={r} aria-pressed={nReads === r} onClick={() => setNReads(r)}>{r}</button>)}
            </span>
          </span>
        ) : null}
      </div>
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <label className="field" style={{ flex: '1 1 14rem' }}>
          n = {n}
          <input type="range" min={1} max={300} value={n} aria-label="n" onChange={(e) => setN(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
        {vv ? (
          <label className="field" style={{ flex: '1 1 10rem' }}>
            <Translated vi={<>độ lệch con trỏ = {offset} float ({offset * 4} byte)</>}>
              pointer offset = {offset} float{offset === 1 ? '' : 's'} ({offset * 4} bytes)
            </Translated>
            <input type="range" min={0} max={3} value={offset} aria-label={tr('pointer offset', 'độ lệch con trỏ')} onChange={(e) => setOffset(Number(e.target.value))} style={{ width: '100%' }} />
          </label>
        ) : null}
      </div>
      <p className="ui muted" style={{ margin: '0 0 0.4rem', fontSize: '0.8rem' }}>
        {STYLES.find((s) => s.id === style)!.who}. {paths.map((p) => <Swatch key={p} path={p} />)}
      </p>
      <Elements l={l} />
      <div style={{ height: '0.6rem' }} />
      <Warps l={l} />
      <p style={{ margin: '0.6rem 0 0.3rem' }} data-readout="summary" data-threads={l.threads} data-tail-elements={tail.elements} data-tail-threads={tail.threads}
        data-divergent={div.join(',')} data-vector={counts.vector.threads} data-misaligned={counts.misaligned.threads} data-idle={counts.idle.threads}
        data-aligned={l.aligned ? '1' : '0'}>
        <Translated vi={<><b>{l.threads}</b> luồng trong <b>{l.blocks}</b> khối cho {n} phần tử.</>}>
          <b>{l.threads}</b> threads in <b>{l.blocks}</b> block{l.blocks === 1 ? '' : 's'} for {n} elements.
        </Translated>{' '}
        {vv ? (
          <Translated
            vi={
              <>
                {counts.vector.threads} luồng dùng lần nạp vector, {counts.misaligned.threads} luồng dùng vòng lặp vô hướng vì con trỏ
                không được căn chỉnh, {tail.threads} luồng xử lý phần đuôi ({tail.elements} phần tử), {counts.idle.threads} luồng
                không có việc gì.{' '}
              </>
            }
          >
            <>
              {counts.vector.threads} take the vector load, {counts.misaligned.threads} the scalar loop because the pointer is
              not aligned, {tail.threads} the tail ({tail.elements} element{tail.elements === 1 ? '' : 's'}), {counts.idle.threads} have
              nothing to do.{' '}
            </>
          </Translated>
        ) : style === 'per-element' ? (
          <Translated vi={<>{counts.scalar.threads} luồng mỗi luồng ghi một phần tử, {counts.idle.threads} luồng dừng ở lệnh kiểm tra biên. </>}>
            <>{counts.scalar.threads} write one element each, {counts.idle.threads} stop at the guard. </>
          </Translated>
        ) : (
          <Translated vi={<>Mỗi luồng lặp với bước nhảy {l.threads}: tối đa {Math.ceil(n / l.threads)} lượt. </>}>
            <>Each thread loops with a stride of {l.threads}: up to {Math.ceil(n / l.threads)} trips. </>
          </Translated>
        )}
        {div.length ? (
          <Translated vi={<>Nhánh chỉ tách ở warp {div.join(', ')}.</>}>
            <>The branch splits in warp {div.join(', ')} only.</>
          </Translated>
        ) : (
          <Translated vi={<>Không warp nào bị tách: mọi warp đi theo một nhánh.</>}>
            <>No warp splits: every warp takes one branch.</>
          </Translated>
        )}
      </p>
      <div className="ui" style={{ fontSize: '0.82rem' }} data-readout="loads">
        {shownWarps.map((w) => (
          <p key={w.w} style={{ margin: '0 0 0.2rem' }} data-warp-loads={w.w}>
            <span className="muted"><Translated vi={<>warp {w.w}, các lần nạp a:</>}>warp {w.w}, loads of a:</Translated></span>{' '}
            {w.loads.length === 0
              ? tr('none', 'không có')
              : w.loads.slice(0, 4).map((ld, k) => {
                  const f = footprint(ld, vv ? offset : 0);
                  return (
                    <span key={k} data-contiguous={f.contiguous ? '1' : '0'}>
                      {k ? '; ' : ''}
                      {ld.lanes[0].width > 1
                        ? tr(`one ${ld.lanes[0].width}-wide load per lane`, `mỗi lane một lần nạp rộng ${ld.lanes[0].width}`)
                        : tr(`load ${ld.k + 1}`, `lần nạp ${ld.k + 1}`)}
                      <Translated vi={<>: {f.read} float</>}>: {f.read} float{f.read === 1 ? '' : 's'}</Translated>{' '}
                      {f.contiguous
                        ? tr(`in one run, ${f.lo} to ${f.hi}`, `trong một dải liền, ${f.lo} đến ${f.hi}`)
                        : tr(`spread over ${f.lo} to ${f.hi}`, `rải trên ${f.lo} đến ${f.hi}`)}
                    </span>
                  );
                })}
            {w.loads.length > 4 ? '; …' : ''}
          </p>
        ))}
        {l.warps.length > 3 ? (
          <p className="muted" style={{ margin: 0 }}>
            <Translated vi="Các warp từ 3 trở lên không được liệt kê.">Warps 3 and up are not listed.</Translated>
          </p>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the variant sort
function VariantSort() {
  const [picks, setPicks] = useState<Record<string, Mechanism | undefined>>({});
  const [checked, setChecked] = useState(false);
  const g = gradeSort(picks);
  const all = VARIANT_MECHANISMS.every((v) => picks[v.id]);
  return (
    <Bench
      title={tr('How each project picks a variant', 'Mỗi dự án chọn biến thể như thế nào')}
      task={tr('Sort each line by the moment its variant is chosen. Then check.', 'Xếp mỗi dòng theo thời điểm biến thể của nó được chọn. Rồi kiểm tra.')}
      foot={
        <Translated vi={<>Mọi dòng đều được trích trong notes/today/x3-facts.md. Bài xếp và đáp án của nó là VARIANT_MECHANISMS trong sim/backendmap.ts.</>}>
          <>Every line is quoted in notes/today/x3-facts.md. The sort and its answers are VARIANT_MECHANISMS in sim/backendmap.ts.</>
        </Translated>
      }
    >
      <div style={{ display: 'grid', gap: '0.5rem' }}>
        {VARIANT_MECHANISMS.map((v) => {
          const ok = checked && picks[v.id] === v.mechanism;
          const bad = checked && picks[v.id] !== v.mechanism;
          return (
            <div key={v.id} className="panel" data-item={v.id} data-result={checked ? (ok ? 'right' : 'wrong') : ''}
              style={{ padding: '0.5rem 0.7rem', borderColor: ok ? 'var(--ok)' : bad ? 'var(--fail)' : undefined }}>
              <div className="row" style={{ justifyContent: 'space-between', gap: '0.4rem 1rem' }}>
                <code style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', flex: '1 1 18rem' }}>{v.code}</code>
                <span className="muted ui" style={{ fontSize: '0.78rem' }}>{PROJECT_NAME[v.project]}, {BACKEND_NAME[v.backend]}</span>
                <select aria-label={tr('mechanism for ', 'cơ chế cho ') + v.id} value={picks[v.id] ?? ''} onChange={(e) => { setPicks({ ...picks, [v.id]: e.target.value as Mechanism }); setChecked(false); }}>
                  <option value="" disabled>{tr('chosen when…', 'được chọn khi…')}</option>
                  {MECHANISMS.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
                </select>
              </div>
              {checked ? (
                <p className="ui" style={{ margin: '0.35rem 0 0', fontSize: '0.82rem' }}>
                  <span className={ok ? 't-ok' : 't-fail'}>
                    {ok
                      ? tr('Right.', 'Đúng.')
                      : tr(
                          `It is: ${MECHANISMS.find((m) => m.id === v.mechanism)!.label}.`,
                          `Đáp án là: ${MECHANISMS.find((m) => m.id === v.mechanism)!.label}.`,
                        )}
                  </span>{' '}
                  {v.why} <span className="muted">({v.facts})</span>
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="row" style={{ marginTop: '0.7rem' }}>
        <button className="primary" disabled={!all} onClick={() => setChecked(true)}><Translated vi="Kiểm tra">Check</Translated></button>
        {checked ? (
          <span data-readout="sort-score">
            <Translated vi={<><b>{g.right} trên {g.total}</b> đúng.</>}>
              <b>{g.right} of {g.total}</b> right.
            </Translated>
          </span>
        ) : (
          <span className="muted ui">{all ? '' : tr('Choose for every line first.', 'Hãy chọn cho mọi dòng trước.')}</span>
        )}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the section
export function SX32() {
  const q = questionsSX32;
  const x = EXCERPTS_SX32;
  // Each kernel without its comment, numbered as in its file.
  const lab23 = useLabExcerpt('ch02/lab3_elementwise/solution/kernel.cu', 'addResidualKernel', { comment: false });
  const lab62 = useLabExcerpt('ch06/lab2_vector/solution/kernel.cu', 'vecAdd4Kernel', { comment: false });
  return (
    <>
      <Translated
        vi={
          <p>
            <code>torch.nn.functional.silu</code> là cổng của mọi MLP Llama: <code>LlamaMLP</code> của Hugging Face tính{' '}
            <code>down_proj(act_fn(gate_proj(x)) * up_proj(x))</code>, và SiLU cùng phép nhân giữa hai phép chiếu là các op theo
            từng phần tử (elementwise). Mọi phép cộng residual cũng vậy. Mục này đọc các kernel mà ba dự án chạy cho chúng, và cho
            phép tra embedding, vốn là một phép gather. Bạn đã viết tất cả: chúng là kernel của chương 2, đôi khi kèm gộp việc cho
            luồng (thread coarsening) và nạp vector của chương 6.
          </p>
        }
      >
        <p>
          <code>torch.nn.functional.silu</code> is the gate of every Llama MLP: Hugging Face’s <code>LlamaMLP</code> computes{' '}
          <code>down_proj(act_fn(gate_proj(x)) * up_proj(x))</code>, and the SiLU and the product between the two projections are
          elementwise ops. So is every residual add. This section reads the kernels the three projects run for them, and for the
          embedding lookup, which is a gather. You have written all of them: they are chapter 2’s kernel, sometimes with chapter
          6’s coarsening and vector loads.
        </p>
      </Translated>
      <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
        <Translated
          vi={
            <p>
              Cách đối chiếu với khóa học (mỗi kernel là lab nào) là của chúng tôi. Các kernel được đọc trong lượt kiểm tra dữ kiện
              ngày 2026-11-20 ở các commit được ghim; không cái nào được chạy.
            </p>
          }
        >
          <p>
            The course mapping (which lab each kernel is) is ours. The kernels were read in the facts pass on 2026-11-20 at the
            pinned commits; none was run.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="llama.cpp trên CUDA: lab 2.3 với op là một template">llama.cpp on CUDA: lab 2.3 with the op as a template</Translated></h2>
      <Translated
        vi={
          <p>
            Đây là toàn bộ kernel đứng sau SiLU, GELU và các op một ngôi khác của llama.cpp trên CUDA. Hãy đọc nó cạnh phép cộng
            residual của lab 2.3 trước khi đọc tiếp.
          </p>
        }
      >
        <p>
          Here is the whole kernel behind SiLU, GELU and llama.cpp’s other unary ops on CUDA. Read it next to lab 2.3’s residual add
          before you read on.
        </p>
      </Translated>
      <Excerpt x={x.unary} caption={<Translated vi="Kernel một ngôi của llama.cpp.">llama.cpp’s unary kernel.</Translated>} />
      <Code tabs={[{ label: tr('lab 2.3, addResidualKernel', 'lab 2.3, addResidualKernel'), code: lab23?.text ?? '// loading…', firstLine: lab23?.firstLine }]}
        caption={
          <Translated vi={<>Kernel của khóa học, nạp từ <code>labs/ch02/lab3_elementwise/solution/kernel.cu</code>.</>}>
            <>The course’s kernel, loaded from <code>labs/ch02/lab3_elementwise/solution/kernel.cu</code>.</>
          </Translated>
        } />
      <Ask q={q['x3/X3.2/q-read-unary']} label={tr('Read the excerpt', 'Đọc đoạn trích')} />
      <Translated
        vi={
          <p>
            Lần khởi chạy cũng là của lab 2.3: <code>(k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE</code> khối. Bản thân SiLU là{' '}
            <code>x / (1.0f + expf(-x))</code>. SwiGLU là một kernel thứ hai, <code>unary_gated_op_kernel</code>, vẫn mỗi đầu ra một
            luồng, tính <code>op(x[j0]) * g[j1]</code>. Các phép cộng residual đi qua <code>k_bin_bcast</code>, và{' '}
            <code>ggml_cuda_op_fused_add</code> gộp 2 đến 8 phép cộng liên tiếp vào một lần khởi chạy.
          </p>
        }
      >
        <p>
          The launch is lab 2.3’s too: <code>(k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE</code> blocks. SiLU itself is{' '}
          <code>x / (1.0f + expf(-x))</code>. SwiGLU is a second kernel, <code>unary_gated_op_kernel</code>, still one thread per
          output, computing <code>op(x[j0]) * g[j1]</code>. Residual adds go through <code>k_bin_bcast</code>, and{' '}
          <code>ggml_cuda_op_fused_add</code> fuses 2 to 8 consecutive adds into one launch.
        </p>
      </Translated>
      <Today checked={CHECKED('B1; ch18 D3')}>{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              <code>unary.cu</code> dòng 116 đến 127, khởi chạy ở dòng 131; <code>ggml_cuda_op_silu_single</code> trong{' '}
              <code>unary.cuh</code>. PDL: chỉ được định nghĩa khi không build cho HIP, có hiệu lực từ Hopper trở đi.
            </p>
          }
        >
          <p>
            <code>unary.cu</code> lines 116 to 127, launch at line 131; <code>ggml_cuda_op_silu_single</code> in{' '}
            <code>unary.cuh</code>. PDL: defined only when not building for HIP, active from Hopper on.
          </p>
        </Translated>
      </Today>
      <Ask q={q['x3/X3.2/q-pdl']} />

      <h2><Translated vi="llama.cpp trên Metal: một kernel, một hằng hàm">llama.cpp on Metal: one kernel, a function constant</Translated></h2>
      <Translated
        vi={
          <p>
            Phía Metal có một template, <code>kernel_unary_impl</code>, cho mọi op một ngôi. Chỗ CUDA biến op thành một tham số
            template, Metal biến nó thành một hằng hàm (function constant), khai báo trong mã nguồn không kèm giá trị và được kiểm
            tra bên trong thân kernel:
          </p>
        }
      >
        <p>
          The Metal side has one template, <code>kernel_unary_impl</code>, for every unary op. Where CUDA makes the op a template
          parameter, Metal makes it a function constant, declared in the source without a value and tested inside the body:
        </p>
      </Translated>
      <Excerpt x={x.unaryMetal} caption={<Translated vi="Bộ chọn op và nhánh SiLU của unary.metal.">The op selector and the SiLU branch of unary.metal.</Translated>} />
      <Ask q={q['x3/X3.2/q-read-fconst']} label={tr('Read the excerpt', 'Đọc đoạn trích')} />
      <Translated
        vi={
          <p>
            Giá trị được cung cấp khi host tạo pipeline, nên một thư viện đã biên dịch chứa mọi op và mỗi pipeline được chuyên biệt
            hóa cho một op. Một bản khởi tạo <code>float4</code>, <code>kernel_unary_f32_f32_4</code>, xử lý bốn phần tử mỗi
            luồng. SwiGLU trên Metal lại được viết theo cách khác nữa, dưới dạng một vòng lặp grid-stride trên một lưới cố định:
          </p>
        }
      >
        <p>
          The value is supplied when the host creates the pipeline, so one compiled library holds every op and each pipeline is
          specialised for one of them. A <code>float4</code> instantiation, <code>kernel_unary_f32_f32_4</code>, does
          four elements per thread. SwiGLU on Metal is written differently again, as a grid-stride loop over a fixed grid:
        </p>
      </Translated>
      <Today checked={CHECKED('C1')}>
        <Translated
          vi={
            <p>
              MSL 4.1 §5.8: “Metal 1.2 and later support function constants.” Chúng “provide the same ease of use as preprocessor
              macros but moves the generation of the specific variants to the creation of the pipeline state, so you don’t have to
              compile the variants offline.”
            </p>
          }
        >
          <p>
            MSL 4.1 §5.8: “Metal 1.2 and later support function constants.” They “provide the same ease of use as preprocessor
            macros but moves the generation of the specific variants to the creation of the pipeline state, so you don’t have to
            compile the variants offline.”
          </p>
        </Translated>
      </Today>
      <Excerpt
        x={x.swiglu}
        caption={
          <Translated vi="Vòng lặp của kernel_swiglu: bắt đầu ở chỉ số toàn cục, bước theo toàn bộ lưới.">
            kernel_swiglu’s loop: start at the global index, step by the whole grid.
          </Translated>
        }
      />
      <Ask q={q['x3/X3.2/q-fconst']} />

      <h2>
        <Translated vi="MLX trên CUDA: gộp việc cho luồng của chương 6, với một lần nạp vector có kiểm tra">
          MLX on CUDA: chapter 6’s coarsening, with a guarded vector load
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            Kernel nhị phân CUDA của MLX giao cho mỗi luồng <code>N_READS</code> phần tử liên tiếp. Một luồng có phần việc sẽ vượt
            quá điểm cuối thì dùng một vòng lặp vô hướng qua phần còn lại; mọi luồng khác nạp phần của mình như một vector. Không có
            lần khởi chạy thứ hai cho phần đuôi.
          </p>
        }
      >
        <p>
          MLX’s CUDA binary kernel gives each thread <code>N_READS</code> consecutive elements. A thread whose chunk would run past
          the end takes a scalar loop over what is left; every other thread loads its chunk as one vector. There is no second
          launch for the tail.
        </p>
      </Translated>
      <Excerpt
        x={x.binaryVv}
        caption={
          <Translated vi="binary_vv của MLX. Các dòng áp dụng Op lên các vector được lược bỏ.">
            MLX’s binary_vv. The lines that apply Op to the vectors are left out.
          </Translated>
        }
      />
      <Translated vi={<p>Lab 6.2 đã yêu cầu bạn viết đúng kernel này. Hãy so sánh phần thân của nó:</p>}>
        <p>
          Lab 6.2 asked you for the same kernel. Compare its body:
        </p>
      </Translated>
      <Code tabs={[{ label: tr('lab 6.2, vecAdd4Kernel', 'lab 6.2, vecAdd4Kernel'), code: lab62?.text ?? '// loading…', firstLine: lab62?.firstLine }]}
        caption={
          <Translated vi={<>Nạp từ <code>labs/ch06/lab2_vector/solution/kernel.cu</code>.</>}>
            <>Loaded from <code>labs/ch06/lab2_vector/solution/kernel.cu</code>.</>
          </Translated>
        } />
      <Translated
        vi={
          <p>
            Khác biệt nằm bên trong <code>load_vector</code>. Nó thực hiện phép ép kiểu con trỏ như sách, nhưng chỉ sau khi kiểm tra
            con trỏ được căn chỉnh theo kích thước của vector, và lùi về một vòng lặp vô hướng nếu không:
          </p>
        }
      >
        <p>
          The difference is inside <code>load_vector</code>. It does the book’s pointer cast, but only after testing that the
          pointer is aligned to the vector’s size, and falls back to a scalar loop if not:
        </p>
      </Translated>
      <Today checked={CHECKED('B1; infra A6; ch06 D4')}>{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              Không nhà cung cấp nào ghi trong tài liệu rằng ép <code>float*</code> sang một con trỏ vector cho ra một lần nạp rộng;
              thay vào đó khóa học đo điều này (infra A6). <code>MemoryAccess.cuh</code> của PyTorch đưa ra cùng lựa chọn lúc chạy
              dựa trên căn chỉnh con trỏ.
            </p>
          }
        >
          <p>
            No vendor documents that casting <code>float*</code> to a vector pointer gives one wide load; the course measures it
            instead (infra A6). PyTorch’s <code>MemoryAccess.cuh</code> makes the same run-time choice from pointer alignment.
          </p>
        </Translated>
      </Today>
      <Excerpt
        x={x.loadVector}
        caption={<Translated vi="load_vector của MLX: phép ép kiểu, được kiểm tra lúc chạy.">MLX’s load_vector: the cast, guarded at run time.</Translated>}
      />
      <Ask q={q['x3/X3.2/q-read-binvv']} label={tr('Read the excerpt', 'Đọc đoạn trích')} />

      <h3><Translated vi="Khám phá các hình dạng">Explore the shapes</Translated></h3>
      <Translated
        vi={
          <p>
            Hãy dự đoán trước khi chạm vào các nút điều khiển. Với N_READS = 4 và n = 150, có bao nhiêu luồng xử lý phần đuôi, và nó
            chứa bao nhiêu phần tử? Sau đó dời con trỏ đi một float. Những luồng nào đổi đường, và có phần tử nào đổi chủ không?
          </p>
        }
      >
        <p>
          Predict before you touch the controls. With N_READS = 4 and n = 150, how many threads take the tail, and how many
          elements does it hold? Then shift the pointer by one float. Which threads change path, and does any element change
          owner?
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Không có thời gian, không có cache, không có giao dịch bộ nhớ: mô hình cho thấy mỗi luồng thực hiện những truy cập nào,
              không phải chúng tốn bao nhiêu. Chỉ các lần nạp của đầu vào thứ nhất được vẽ. Phần tính toán khởi chạy của MLX không
              được đọc, nên lưới của binary_vv ở đây là của chúng tôi.
            </p>
          }
        >
          <p>
            No timing, no caches, no memory transactions: the model shows which accesses each thread makes, not what they cost.
            Only loads of the first input are drawn. MLX’s launch computation was not read, so binary_vv’s grid here is ours.
          </p>
        </Translated>
      </Simplification>
      <LaunchShapes />
      <Translated
        vi={
          <p>
            Điều bạn lẽ ra đã thấy. Phần đuôi chứa n mod N_READS phần tử trong một luồng, và nhánh tách ở tối đa một warp. Độ lệch
            một float đẩy mọi luồng sang đường vô hướng; mỗi luồng giữ nguyên các phần tử của mình, nên kết quả như cũ, nhưng trong
            mỗi lệnh nạp, các lane của một warp giờ đọc cứ mỗi float thứ tư thay vì một dải liền mạch (với N_READS = 2, độ lệch hai
            float vẫn được căn chỉnh). Kiểu grid-stride phủ mọi n với cùng một lưới cố định.
          </p>
        }
      >
        <p>
          What you should have seen. The tail holds n mod N_READS elements in one thread, and the branch splits in at most one
          warp. A one-float offset sends every thread down the scalar path; each thread keeps its elements, so the result is the
          same, but in each load instruction the lanes of a warp now read every fourth float instead of one unbroken run (with
          N_READS = 2, a two-float offset is still aligned). The grid-stride style covers any n with the same fixed grid.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Các kernel Metal của MLX thực hiện cùng kiểu gộp việc đó trong <code>binary.h</code>: <code>WorkPerThread&lt;T&gt;::n</code> phần
              tử mỗi luồng, với lệnh kiểm tra biên <code>index + i &lt; size</code> (ch02 D3).
            </p>
          }
        >
          <p>
            MLX’s Metal kernels do the same coarsening in <code>binary.h</code>: <code>WorkPerThread&lt;T&gt;::n</code> elements per
            thread, with a guard <code>index + i &lt; size</code> (ch02 D3).
          </p>
        </Translated>
      </Aside>
      <Ask q={q['x3/X3.2/q-tail']} />
      <Ask q={q['x3/X3.2/q-misaligned']} />

      <h2><Translated vi="PyTorch: một lambda trên một vòng lặp elementwise">PyTorch: a lambda over one elementwise loop</Translated></h2>
      <Translated
        vi={
          <p>
            Các op elementwise CUDA của PyTorch là các lambda truyền cho một hàm khởi chạy, <code>gpu_kernel</code>, hàm này chạy{' '}
            <code>elementwise_kernel</code>: một lưới 1D bất kể hình dạng tensor, nhiều phần tử mỗi luồng, kích thước khối cố định
            trong mã nguồn (128 luồng trên CUDA, 256 dưới ROCm), và độ rộng vector được chọn từ căn chỉnh của mỗi con trỏ lúc chạy.
            Lambda của SiLU tính trong <code>at::opmath_type</code>, kiểu tích lũy, và ép về kiểu của tensor ở cuối.
          </p>
        }
      >
        <p>
          PyTorch’s CUDA elementwise ops are lambdas passed to one launcher, <code>gpu_kernel</code>, which runs{' '}
          <code>elementwise_kernel</code>: a 1D grid whatever the tensor’s shape, several elements per thread, a block size fixed in
          source (128 threads on CUDA, 256 under ROCm), and a vector width chosen from each pointer’s alignment at run time. SiLU’s
          lambda computes in <code>at::opmath_type</code>, the accumulate type, and casts back to the tensor’s type at the end.
        </p>
      </Translated>
      <Today checked={CHECKED('A3; ch03 D6, ch06 D4')}>{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              <code>num_threads()</code> là <code>C10_WARP_SIZE * 4</code> trên CUDA và 256 dưới <code>USE_ROCM</code>.
              {' '}<code>can_vectorize_up_to</code> trả về 8, 4, 2 hoặc 1 từ địa chỉ của con trỏ.
            </p>
          }
        >
          <p>
            <code>num_threads()</code> is <code>C10_WARP_SIZE * 4</code> on CUDA and 256 under <code>USE_ROCM</code>.
            {' '}<code>can_vectorize_up_to</code> returns 8, 4, 2 or 1 from the pointer’s address.
          </p>
        </Translated>
      </Today>
      <Excerpt
        x={x.siluTorch}
        caption={
          <Translated vi="SiLU của PyTorch trên CUDA, đăng ký bằng REGISTER_DISPATCH(silu_stub, &silu_kernel).">
            PyTorch’s SiLU on CUDA, registered with REGISTER_DISPATCH(silu_stub, &silu_kernel).
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Trên MPS, PyTorch gọi kernel bằng tên: <code>lib.exec_binary_kernel(iter, "add")</code> chọn{' '}
            <code>add_functor</code> hoặc <code>add_alpha_functor</code> trong <code>kernels/BinaryKernel.metal</code>.
          </p>
        }
      >
        <p>
          On MPS, PyTorch names the kernel instead: <code>lib.exec_binary_kernel(iter, "add")</code> picks{' '}
          <code>add_functor</code> or <code>add_alpha_functor</code> in <code>kernels/BinaryKernel.metal</code>.
        </p>
      </Translated>

      <h3><Translated vi="Cùng một hàm, ba công thức">The same function, three formulas</Translated></h3>
      <Translated
        vi={
          <p>
            llama.cpp trên CUDA và Metal, PyTorch trên CUDA và llama2.c tính SiLU là x / (1 + exp(−x)). MLX tính x ·
            sigmoid(x). Hai công thức bằng nhau trong số học chính xác và không nhất thiết làm tròn như nhau trên float. Khi bạn so
            kernel của mình với một trong số này, hãy so với một dung sai; mục <a href={href('x3', 'X3.9')}>X3.9</a> nói các dự án
            làm điều đó ra sao.
          </p>
        }
      >
        <p>
          llama.cpp on CUDA and Metal, PyTorch on CUDA and llama2.c compute SiLU as x / (1 + exp(−x)). MLX computes x ·
          sigmoid(x). The two are equal in exact arithmetic and need not round alike in float. When you compare your kernel with
          one of these, compare with a tolerance; section <a href={href('x3', 'X3.9')}>X3.9</a> says how the projects do it.
        </p>
      </Translated>
      <Beyond title={tr('Ours, not measured', 'Của chúng tôi, chưa đo')}>
        <Translated vi={<p>Việc hai công thức có thể làm tròn khác nhau là cách đọc của chúng tôi. Chưa ai trong khóa học này đo chúng khác nhau thường xuyên đến mức nào.</p>}>
          <p>That the two formulas can round differently is our reading. Nobody in this course has measured how often they do.</p>
        </Translated>
      </Beyond>
      <Ask q={q['x3/X3.2/q-silu-bits']} />

      <h2><Translated vi="Embedding ở đâu cũng là một phép gather">Embedding is a gather everywhere</Translated></h2>
      <Translated
        vi={
          <p>
            Mọi dự án tra embedding bằng một phép gather theo hàng: mỗi token id một hàng đầu ra. llama.cpp có một op đồ thị cho nó,{' '}
            <code>GGML_OP_GET_ROWS</code>, với <code>k_get_rows_float</code> và <code>k_get_rows_float_vec</code> cho bảng float
            và <code>k_get_rows</code> cho bảng lượng tử hóa, kernel này giải lượng tử hóa từng hàng khi sao chép; trên Metal, host yêu cầu{' '}
            <code>kernel_get_rows_</code> cộng với tên kiểu. PyTorch hoàn toàn không có kernel embedding riêng cho từng backend: op này
            là một tổ hợp của <code>index_select</code>.
          </p>
        }
      >
        <p>
          Every project looks up embeddings as a row gather: one output row per token id. llama.cpp has a graph op for it,{' '}
          <code>GGML_OP_GET_ROWS</code>, with <code>k_get_rows_float</code> and <code>k_get_rows_float_vec</code> for float tables
          and <code>k_get_rows</code> for quantised ones, which dequantises each row as it copies; on Metal the host asks for{' '}
          <code>kernel_get_rows_</code> plus the type’s name. PyTorch has no embedding kernel per backend at all: the op is a
          composite of <code>index_select</code>.
        </p>
      </Translated>
      <Excerpt
        x={x.embedTorch}
        caption={<Translated vi="embedding_symint của PyTorch, toàn bộ lượt forward.">PyTorch’s embedding_symint, the whole forward.</Translated>}
      />
      <Translated
        vi={
          <p>
            <code>nn.Embedding</code> của MLX là <code>self.weight[x]</code>. Cả ba đều làm điều mà <code>launchEmbed</code> của lab 22.2{' '}
            làm, một phép gather theo hàng với mỗi token id một hàng đầu ra. Cách mỗi dự án ánh xạ luồng tới đầu ra chưa được đọc.
          </p>
        }
      >
        <p>
          MLX’s <code>nn.Embedding</code> is <code>self.weight[x]</code>. All three do what lab 22.2’s <code>launchEmbed</code>{' '}
          does, a row gather with one output row per token id. How each maps threads to outputs was not read.
        </p>
      </Translated>
      <Today checked={CHECKED('B7')}>
        <Translated
          vi={
            <p>
              <code>getrows.cu</code> dòng 6, 73, 105; <code>quantize.metal</code> dòng 266 và 297; PyTorch{' '}
              <code>native_functions.yaml</code> dòng 2261 đến 2263 (<code>CompositeExplicitAutograd: embedding_symint</code>).
            </p>
          }
        >
          <p>
            <code>getrows.cu</code> lines 6, 73, 105; <code>quantize.metal</code> lines 266 and 297; PyTorch{' '}
            <code>native_functions.yaml</code> lines 2261 to 2263 (<code>CompositeExplicitAutograd: embedding_symint</code>).
          </p>
        </Translated>
      </Today>
      <Ask q={q['x3/X3.2/q-getrows-quant']} />
      <OpMap op="embedding" project="llama.cpp" title={tr('Op map: embedding', 'Bản đồ op: embedding')} />{/* i18n-ignore: op id */}

      <h2><Translated vi="Mỗi dự án chọn biến thể như thế nào">How each project picks a variant</Translated></h2>
      <Translated
        vi={
          <p>
            Mọi kernel ở trên tồn tại dưới nhiều biến thể: theo op, theo kiểu, theo mức gộp, theo độ rộng vector. Các dự án chọn
            giữa chúng ở những thời điểm khác nhau. Trên CUDA và HIP, lựa chọn thường được đưa ra khi thư viện được biên dịch; trên
            Metal, nó được đưa ra khi một pipeline được tạo, hoặc qua cái tên mà host yêu cầu; MLX và PyTorch còn sinh mã, lúc chạy
            và lúc build.
          </p>
        }
      >
        <p>
          Every kernel above exists in several variants: per op, per type, per fusion level, per vector width. The projects choose
          among them at different moments. On CUDA and HIP the choice is usually made when the library is compiled; on Metal it
          is made when a pipeline is created, or by the name the host asks for; MLX and PyTorch also generate code, at run time
          and at build time.
        </p>
      </Translated>
      <VariantSort />
      <Translated
        vi={
          <p>
            Quy tắc thực hành khi đọc: một biến thể CUDA hay HIP là một bản khởi tạo template mà bạn có thể tìm bằng tên template của
            nó. Một biến thể Metal là một chuỗi hoặc giá trị của một hằng hàm, được đặt bởi mã host mà bạn phải đọc trước. Một biến thể
            được sinh ra thì không nằm trong tệp nào cả.
          </p>
        }
      >
        <p>
          The practical rule for reading: a CUDA or HIP variant is a template instantiation you can find by its template’s name.
          A Metal variant is a string or a function constant’s value, set by host code you have to read first. A generated
          variant is in no file at all.
        </p>
      </Translated>
    </>
  );
}
