import { useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  DIFFER_ROWS,
  ENTRY_POINTS,
  PROJECT_NAME,
  ROUTE_CHOICES,
  hopOrder,
  route,
  scoreWalk,
  type Route,
} from '../../sim/backendmap';
import { linesText, permalink, shortCommit, type Excerpt as ExcerptData } from '../../sim/x3sources';
import { Excerpt } from './Excerpt';
import { OneCallScene } from './OneCallScene';
import { OpMap } from './OpMap';

// Section X3.1: the reading method. Each project has one place where an op becomes a backend
// call; start there, follow one name, and on Metal follow a string. The op map, the
// "Follow the call" stepper and the one-call scene all run on sim/backendmap.ts. The
// excerpts below are checked against notes/today/x3-facts.md by excerpts-g1.test.ts.

const FACTS = 'notes/today/x3-facts.md';
const CHECKED = (items: string) => tr(`2026-11-20 in the facts pass (${FACTS}, ${items})`, `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, ${items})`);

export const EXCERPTS_SX31: Record<string, ExcerptData> = {
  switch: {
    id: 'x31-switch', project: 'llama.cpp', path: 'ggml/src/ggml-cuda/ggml-cuda.cu', lines: [2812, 3042], factsItem: 'A1', lang: 'cuda',
    code: `static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
...
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
...
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
...
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
...`,
  },
  fusion: {
    id: 'x31-fusion', project: 'llama.cpp', path: 'ggml/src/ggml-cuda/ggml-cuda.cu', lines: [5268, 5271], factsItem: 'A1', lang: 'cuda',
    code: `if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
    ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
    return 1;
}`,
  },
  snprintf: {
    id: 'x31-snprintf', project: 'llama.cpp', path: 'ggml/src/ggml-metal/ggml-metal-device.cpp', lines: 154, factsItem: 'A1', lang: 'cpp',
    code: `snprintf(base, 256, "kernel_get_rows_%s", ggml_type_name(tsrc));`,
  },
  hostName: {
    id: 'x31-host-name', project: 'llama.cpp', path: 'ggml/src/ggml-metal/kernels/norm.metal', lines: 179, factsItem: 'C2', lang: 'metal',
    code: `template [[host_name("kernel_rms_norm_mul_f32")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 2>;`,
  },
  rmsNormPy: {
    id: 'x31-mlx-rmsnorm', project: 'mlx', path: 'python/mlx/nn/layers/normalization.py', lines: 166, factsItem: 'A2', lang: 'python',
    code: `return mx.fast.rms_norm(x, self["weight"], self.eps)`,
  },
  siluPy: {
    id: 'x31-mlx-silu', project: 'mlx', path: 'python/mlx/nn/layers/activations.py', lines: [136, 145], factsItem: 'A2', lang: 'python',
    code: `@partial(mx.compile, shapeless=True)
def silu(x):
...
    return x * mx.sigmoid(x)`,
  },
  addH: {
    id: 'x31-ufunc-add', project: 'pytorch', path: 'aten/src/ATen/native/ufunc/add.h', lines: null, factsItem: 'A3', lang: 'cpp',
    code: `return self + alpha * other;`,
  },
};

export const questionsSX31: Record<string, Question> = {
  'x3/X3.1/q-no-add-file': {
    kind: 'choice',
    id: 'x3/X3.1/q-no-add-file',
    prompt: (
      <Translated
        vi={
          <>
            Bạn grep thư mục <code>aten/src/ATen/native/cuda</code> của PyTorch để tìm kernel CUDA đứng sau{' '}
            <code>add.out</code> và không thấy gì. Vì sao?
          </>
        }
      >
        <>
          You grep PyTorch’s <code>aten/src/ATen/native/cuda</code> for the CUDA kernel behind <code>add.out</code> and find
          nothing. Why?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="add.out là một op ufunc: torchgen viết file CUDA của nó, UfuncCUDA_add.cu, lúc build từ ufunc/add.h.">
            add.out is a ufunc op: torchgen writes its CUDA file, UfuncCUDA_add.cu, at build time from ufunc/add.h.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Phép cộng được chuyển cho cuBLAS, nên PyTorch không giữ kernel add riêng nào trong native/cuda.">Adds go to cuBLAS, so PyTorch keeps no add kernel of its own in native/cuda.</Translated>,
        why: <Translated vi="Không có thư viện nào tham gia. File được sinh ra gọi vòng lặp gpu_kernel của chính PyTorch.">No library is involved. The generated file calls PyTorch’s own gpu_kernel loop.</Translated>,
      },
      {
        label: <Translated vi="Kernel đã chuyển sang cây MPS, nơi giờ phục vụ mọi GPU.">The kernel moved to the MPS tree, which serves every GPU now.</Translated>,
        why: <Translated vi="MPS chỉ là backend của Apple, và nó có add_out_mps viết tay riêng chính vì lý do này.">MPS is the Apple backend only, and it has its own hand-written add_out_mps for exactly this reason.</Translated>,
      },
      {
        label: <Translated vi="add được cài đặt bằng Python và được trace thành kernel lúc chạy.">add is implemented in Python and traced into a kernel at run time.</Translated>,
        why: <Translated vi="Điều đó không mô tả op nào của PyTorch ở đây. Thân vô hướng là C++ trong native/ufunc/add.h.">That describes no PyTorch op here. The scalar body is C++ in native/ufunc/add.h.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>native_functions.yaml</code> cho <code>add.out</code> một <code>ufunc_inner_loop</code> và không có khóa{' '}
            <code>CUDA:</code>. torchgen sinh <code>UfuncCUDA_add.cu</code> trong lúc build, chứa{' '}
            <code>gpu_kernel(iter, ufunctor);</code> và <code>REGISTER_DISPATCH</code> nối nó vào. File này chỉ tồn tại trong
            một cây build.
          </p>
        }
      >
        <p>
          <code>native_functions.yaml</code> gives <code>add.out</code> a <code>ufunc_inner_loop</code> and no <code>CUDA:</code>{' '}
          key. torchgen generates <code>UfuncCUDA_add.cu</code> during the build, containing <code>gpu_kernel(iter, ufunctor);</code>{' '}
          and the <code>REGISTER_DISPATCH</code> that wires it up. The file exists only in a build tree.
        </p>
      </Translated>
    ),
    source: `${FACTS} A3`,
  },
  'x3/X3.1/q-fusion-breakpoint': {
    kind: 'choice',
    id: 'x3/X3.1/q-fusion-breakpoint',
    prompt: (
      <Translated
        vi={
          <>
            Bạn chạy một mô hình Llama trên backend CUDA của llama.cpp với một breakpoint trong{' '}
            <code>ggml_cuda_op_rms_norm</code>. Nó không bao giờ dừng, dù mô hình rõ ràng có chuẩn hóa. Lý do khả dĩ nhất là
            gì, và bạn xác nhận nó thế nào?
          </>
        }
      >
        <>
          You run a Llama model on llama.cpp’s CUDA backend with a breakpoint in <code>ggml_cuda_op_rms_norm</code>. It never
          hits, though the model plainly normalises. What is the likeliest reason, and how do you confirm it?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Phép gộp (fusion) đã gửi RMS_NORM và MUL của nó tới ggml_cuda_op_rms_norm_fused; chạy lại với GGML_CUDA_DISABLE_FUSION được đặt.">
            Fusion sent RMS_NORM and its MUL to ggml_cuda_op_rms_norm_fused; rerun with GGML_CUDA_DISABLE_FUSION set.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Op đã lùi về backend CPU, và chạy ở đó. Kiểm tra ggml_backend_cuda_device_supports_op cho RMS_NORM.">The op fell back to the CPU backend, which ran it there. Check ggml_backend_cuda_device_supports_op for RMS_NORM.</Translated>,
        why: <Translated vi="Backend CUDA chấp nhận RMS_NORM. supports_op sẽ cho bạn biết nếu nó không chấp nhận, nhưng đây không phải nguyên nhân khả dĩ.">The CUDA backend accepts RMS_NORM. supports_op would tell you if it did not, but this is not the likely cause.</Translated>,
      },
      {
        label: <Translated vi="File mô hình lưu trọng số đã chuẩn hóa sẵn, nên không cần chạy phép chuẩn hóa nào lúc suy luận; kiểm tra metadata GGUF.">The model file stores pre-normalised weights, so no norm needs to run at inference time; check the GGUF metadata.</Translated>,
        why: <Translated vi="RMSNorm chia mỗi hàng activation cho căn trung bình bình phương của chính nó, vốn phụ thuộc vào đầu vào. Trọng số không thể làm việc đó trước được.">RMSNorm divides each activation row by its own root mean square, which depends on the input. Weights cannot do that ahead of time.</Translated>,
      },
      {
        label: <Translated vi="Bản build HIP đã được nạp thay cho bản CUDA. Kiểm tra thư viện dùng chung nào tiến trình thực sự đã mở.">The HIP build was loaded instead of the CUDA one. Check which shared library the process actually opened.</Translated>,
        why: <Translated vi="Bản build HIP biên dịch cùng norm.cu và chạy cùng lượt gộp, nên điều này cũng không giải thích được.">The HIP build compiles the same norm.cu and runs the same fusion pass, so this would not explain it either.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trước switch trong <code>ggml_cuda_compute_forward</code>, một lượt gộp (fusion) xem xét các chuỗi node.
            RMS_NORM theo sau bởi MUL trở thành một lời gọi tới <code>ggml_cuda_op_rms_norm_fused</code>, và RMS_NORM, MUL,
            ADD, được thử sớm hơn một bước, thành một lời gọi tới <code>ggml_cuda_op_rms_norm_fused_add</code>. Biến môi
            trường <code>GGML_CUDA_DISABLE_FUSION</code> tắt lượt này, và khi đó case thường sẽ chạy.
          </p>
        }
      >
        <p>
          Before the switch in <code>ggml_cuda_compute_forward</code>, a fusion pass looks at runs of nodes. RMS_NORM followed by
          MUL becomes one call to <code>ggml_cuda_op_rms_norm_fused</code>, and RMS_NORM, MUL, ADD, tried one step earlier,
          one call to <code>ggml_cuda_op_rms_norm_fused_add</code>. The environment variable{' '}
          <code>GGML_CUDA_DISABLE_FUSION</code> turns the pass off, and then the plain case runs.
        </p>
      </Translated>
    ),
    source: `${FACTS} A1`,
  },
  'x3/X3.1/q-metal-string': {
    kind: 'choice',
    id: 'x3/X3.1/q-metal-string',
    prompt: (
      <Translated
        vi={
          <>
            Host Metal của llama.cpp cần kernel RMSNorm được gộp với phép nhân trọng số, cho các hàng float. Để tìm mã
            nguồn của nó, bạn tìm gì trong các file <code>.metal</code>?
          </>
        }
      >
        <>
          llama.cpp’s Metal host wants the RMSNorm kernel fused with the weight multiply, for float rows. To find its source,
          what do you search the <code>.metal</code> files for?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Chuỗi mà host yêu cầu, kernel_rms_norm_mul_f32, được một [[host_name(...)]] gán cho một bản khởi tạo.">The string the host asks for, kernel_rms_norm_mul_f32, which a [[host_name(...)]] gives to one instantiation.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Tên C++ của template, kernel_rms_norm_fuse_impl, vì mọi bản khởi tạo mà pipeline có thể chạy đều mang tên đó.">The template’s C++ name, kernel_rms_norm_fuse_impl, since every instantiation the pipeline can run carries that name.</Translated>,
        why: <Translated vi="Template đó phục vụ mọi mức gộp và mọi kiểu. Tên của nó không cho bạn biết bản khởi tạo nào chạy.">That template serves every fusion level and type. Its name does not tell you which instantiation runs.</Translated>,
      },
      {
        label: <Translated vi="Tên hàm host, ggml_metal_op_norm, được định nghĩa trong file .metal.">The host function’s name, ggml_metal_op_norm, which the .metal file defines.</Translated>,
        why: <Translated vi="ggml_metal_op_norm là mã host C++ trong ggml-metal-ops.cpp; đó là nơi bạn vừa đi qua.">ggml_metal_op_norm is C++ host code in ggml-metal-ops.cpp; it is where you came from.</Translated>,
      },
      {
        label: <Translated vi="Enum op GGML_OP_RMS_NORM, mà file .metal dùng để switch.">The op enum GGML_OP_RMS_NORM, which the .metal file switches on.</Translated>,
        why: <Translated vi="Switch nằm ở host. Mã nguồn kernel không bao giờ thấy enum op.">The switch is on the host. Kernel source never sees the op enum.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Metal tìm kernel bằng một chuỗi tên. Trong <code>norm.metal</code> dòng 179,{' '}
            <code>template [[host_name("kernel_rms_norm_mul_f32")]] … kernel_rms_norm_fuse_impl&lt;float, 2&gt;;</code> gán
            cho bản khởi tạo có mức gộp 2 cái tên mà host dựng ra. Template cần điều này: mỗi bản chuyên biệt hóa có tên
            riêng.
          </p>
        }
      >
        <p>
          Metal finds a kernel by a name string. In <code>norm.metal</code> line 179,{' '}
          <code>template [[host_name("kernel_rms_norm_mul_f32")]] … kernel_rms_norm_fuse_impl&lt;float, 2&gt;;</code> gives the
          instantiation with fusion level 2 the name the host builds. Templates need it: each specialisation gets its own name.
        </p>
      </Translated>
    ),
    source: `${FACTS} A1, C2; MSL 4.1 on [[host_name]]`,
    vi: { source: `${FACTS} A1, C2; MSL 4.1 về [[host_name]]` },
  },
  'x3/X3.1/q-silu-file': {
    kind: 'choice',
    id: 'x3/X3.1/q-silu-file',
    prompt: (
      <Translated vi={<>Mã nguồn Metal của <code>nn.silu</code> trong MLX nằm ở đâu?</>}>
        <>Where is the Metal source of MLX’s <code>nn.silu</code>?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không ở đâu trong cây mã: nn.silu nằm dưới mx.compile, nên mã nguồn kernel của nó được viết ra lúc chạy.">
            Nowhere in the tree: nn.silu is under mx.compile, so its kernel source is written at run time.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Trong mlx/backend/metal/kernels/unary.h, cạnh các op một ngôi khác như exp và sigmoid.">In mlx/backend/metal/kernels/unary.h, beside the other unary ops such as exp and sigmoid.</Translated>,
        why: <Translated vi="Một chỗ hợp lý, nhưng các dữ kiện cho thấy nn.silu là một hàm mx.compile, vốn không có file kernel.">A plausible place, but the facts show nn.silu is an mx.compile function, which has no kernel file.</Translated>,
      },
      {
        label: <Translated vi="Trong một file .metal mà quá trình build của MLX sinh ra từ một template, như torchgen sinh các kernel ufunc của PyTorch.">In a .metal file that MLX’s build generates from a template, as torchgen generates PyTorch’s ufunc kernels.</Translated>,
        why: <Translated vi="Được sinh ra, đúng, nhưng lúc chạy ở lần gọi đầu tiên, không phải lúc build.">Generated, yes, but at run time on first call, not at build time.</Translated>,
      },
      {
        label: <Translated vi="Trong MPS: MLX gọi Metal Performance Shaders của Apple cho các hàm kích hoạt.">In MPS: MLX calls Apple’s Metal Performance Shaders for activations.</Translated>,
        why: <Translated vi="Không dữ kiện nào cho thấy MLX chuyển các hàm kích hoạt sang MPS.">Nothing in the facts routes MLX activations to MPS.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>nn.silu</code> là <code>@partial(mx.compile, shapeless=True)</code>. <code>Compiled::eval_gpu</code> trong{' '}
            <code>metal/compiled.cpp</code> viết <code>[[host_name("…")]] [[kernel]] void …(</code> vào một chuỗi mã nguồn và
            biên dịch nó bằng <code>d.get_library(...)</code>. Muốn đọc nó, bạn phải in mã nguồn được sinh ra; không có file
            nào để mở.
          </p>
        }
      >
        <p>
          <code>nn.silu</code> is <code>@partial(mx.compile, shapeless=True)</code>. <code>Compiled::eval_gpu</code> in{' '}
          <code>metal/compiled.cpp</code> writes <code>[[host_name("…")]] [[kernel]] void …(</code> into a source string and compiles
          it with <code>d.get_library(...)</code>. To read it, you would print the generated source; there is no file to open.
        </p>
      </Translated>
    ),
    source: `${FACTS} A2`,
  },
  'x3/X3.1/q-mlx-hip': {
    kind: 'choice',
    id: 'x3/X3.1/q-mlx-hip',
    prompt: (
      <Translated
        vi={
          <>
            Bạn có thể build backend CUDA của MLX cho GPU AMD bằng cách cho nó qua một bước hipify, như PyTorch làm với mã
            nguồn CUDA của nó không?
          </>
        }
      >
        <>
          Could you build MLX’s CUDA backend for an AMD GPU by running it through a hipify step, as PyTorch does with its CUDA
          sources?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không phải nguyên trạng: nó gọi cuBLASLt, frontend của cuDNN và cooperative groups, mỗi thứ cần một bản port chứ không chỉ đổi tên.">
            Not as is: it calls cuBLASLt, cuDNN’s frontend and cooperative groups, each needing a port rather than a rename.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Có: hipify đổi tên mọi lời gọi runtime và thư viện CUDA sang cách viết HIP, nên khi đó mọi mã nguồn CUDA đều build được cho AMD.">Yes: hipify renames every CUDA runtime and library call to its HIP spelling, so any CUDA source then builds for AMD.</Translated>,
        why: <Translated vi="Thư viện và cooperative groups là chỗ khẳng định đơn giản đó không còn đúng (ghi chú về tính khả chuyển của khóa học, infra B4).">Libraries and cooperative groups are where the simple claim stops being true (the course’s portability notes, infra B4).</Translated>,
      },
      {
        label: <Translated vi="Có: MLX đã cung cấp sẵn bước đó trong mlx/backend/rocm.">Yes: MLX already ships that step under mlx/backend/rocm.</Translated>,
        why: <Translated vi="Không có thư mục nào như vậy ở commit được ghim; một pull request ROCm thử nghiệm đang mở và chưa được merge.">There is no such directory at the pinned commit; an experimental ROCm pull request is open and unmerged.</Translated>,
      },
      {
        label: <Translated vi="Không, vì GPU AMD không có warp để các phép rút gọn cấp warp của nó chạy trên đó.">No, because AMD GPUs have no warps for its warp-level reductions to run on.</Translated>,
        why: <Translated vi="Chúng có: rộng 64 hoặc 32. Đó không phải trở ngại.">They do: 64 or 32 wide. That is not the obstacle.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            PyTorch hipify được vì cây CUDA của nó được viết với bước đó trong đầu và ghép nó với các thư viện AMD. Backend
            CUDA của MLX gọi cuBLASLt cho GEMM, frontend của cuDNN cho attention, và cooperative groups trong các phép rút gọn
            của nó. Mỗi thứ cần một bản port, không chỉ đổi tên. <code>mlx/backend</code> không có thư mục HIP, và pull request
            #2300 vẫn đang mở.
          </p>
        }
      >
        <p>
          PyTorch can hipify because its CUDA tree was written with that step in mind and pairs it with AMD libraries. MLX’s CUDA
          backend calls cuBLASLt for GEMM, cuDNN’s frontend for attention, and cooperative groups in its reductions. Each needs a
          port, not a rename. <code>mlx/backend</code> has no HIP directory, and pull request #2300 is still open.
        </p>
      </Translated>
    ),
    source: `${FACTS} B9; notes/today/infra-facts.md B4`,// i18n-ignore: citation only
  },
};

// ---------------------------------------------------------------- Follow the call
function FollowTheCall() {
  const [choice, setChoice] = useState(ROUTE_CHOICES[0].id);
  const [fusion, setFusion] = useState(true);
  const c = ROUTE_CHOICES.find((x) => x.id === choice)!;
  const r: Route = route(c.variants ? (fusion ? c.variants.on : c.variants.off) : c.id);
  const [hop, setHop] = useState(0);
  const [first, setFirst] = useState<(number | null)[]>([]);
  const [pick, setPick] = useState<number | null>(null);

  const reset = (id: string, f = fusion) => {
    setChoice(id);
    setFusion(f);
    setHop(0);
    setFirst([]);
    setPick(null);
  };
  const done = hop >= r.hops.length;
  const h = done ? null : r.hops[hop];
  const order = h ? hopOrder(r.id, hop, h.options.length) : [];
  const right = h && pick != null ? !!h.options[pick].correct : false;
  const choose = (i: number) => {
    setPick(i);
    setFirst((f) => {
      if (f[hop] != null) return f;
      const n = [...f];
      n[hop] = i;
      return n;
    });
  };
  const next = () => {
    setHop(hop + 1);
    setPick(null);
  };
  const score = scoreWalk(r, first);

  return (
    <Bench
      title={tr('Follow the call', 'Lần theo lời gọi')}
      task={tr('Choose what to search for at each hop. Then the hop is revealed, with its file, lines and permalink.', 'Chọn thứ cần tìm ở mỗi bước. Sau đó bước đó được hé lộ, cùng file, các dòng và permalink của nó.')}
      foot={
        <Translated vi={<>Mỗi bước là một lần tìm kiếm trong cây mã thật ở commit được ghim. Các lộ trình và mục dữ kiện của chúng là dữ liệu trong sim/backendmap.ts (ROUTES).</>}>
          <>Each hop is one search in the real tree at the pinned commit. The routes and their facts items are data in sim/backendmap.ts (ROUTES).</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem', alignItems: 'flex-end' }}>
        <span className="seg" role="group" aria-label={tr('Route', 'Lộ trình')} style={{ flexWrap: 'wrap' }}>
          {ROUTE_CHOICES.map((x) => (
            <button key={x.id} aria-pressed={choice === x.id} onClick={() => reset(x.id)} data-route={x.id}>{x.label}</button>
          ))}
        </span>
        {c.variants ? (
          <label className="check">
            <input type="checkbox" checked={fusion} onChange={(e) => reset(choice, e.target.checked)} />{' '}
            <Translated vi="bật gộp (mặc định)">fusion on (the default)</Translated>
          </label>
        ) : null}
      </div>
      <p className="ui" style={{ margin: '0 0 0.6rem', fontSize: '0.88rem' }} data-readout="start">
        <b><Translated vi="Bắt đầu.">Start.</Translated></b> {r.start}
      </p>
      <ol data-part="trail" style={{ margin: '0 0 0.6rem', paddingLeft: '1.3em', fontFamily: 'var(--sans)', fontSize: '0.84rem' }}>
        {r.hops.slice(0, hop).map((x, i) => (
          <li key={i} style={{ marginBottom: '0.3rem' }} data-hop={i}>
            <span className={first[i] != null && x.options[first[i]!].correct ? 't-ok' : 't-fail'}>
              {first[i] != null && x.options[first[i]!].correct ? tr('right first time', 'đúng ngay lần đầu') : tr('found on a later try', 'tìm ra ở lần thử sau')}
            </span>
            {': '}
            {x.found}{' '}
            <a href={permalink(x.project, x.path, x.lines ?? null, !/\.\w+$/.test(x.path))} target="_blank" rel="noreferrer">
              {x.path.split('/').pop()}{x.lines != null ? ', ' + linesText(x.lines) : ''}
            </a>
          </li>
        ))}
      </ol>
      {h ? (
        <div className="panel" data-part="hop" data-hop={hop}>
          <p style={{ margin: '0 0 0.5rem' }}>
            <b><Translated vi={<>Bước {hop + 1} trên {r.hops.length}.</>}>Hop {hop + 1} of {r.hops.length}.</Translated></b> {h.prompt}
          </p>
          <div style={{ display: 'grid', gap: '0.35rem' }}>
            {order.map((i) => {
              const o = h.options[i];
              const state = pick === i ? (o.correct ? 'right' : 'wrong') : '';
              return (
                <button key={i} className="quiet" onClick={() => choose(i)} disabled={right} data-option={o.label}
                  style={{
                    textAlign: 'left', fontWeight: 450,
                    borderColor: state === 'right' ? 'var(--ok)' : state === 'wrong' ? 'var(--fail)' : undefined,
                    background: state === 'right' ? 'var(--ok-wash)' : state === 'wrong' ? 'var(--fail-wash)' : undefined,
                    color: 'var(--ink)',
                  }}>
                  {o.label}
                  {o.kind ? <span className="muted"> ({tr(o.kind, ({ 'function name': 'tên hàm', 'yaml key': 'khóa yaml', 'switch case': 'case của switch', 'formatted string': 'chuỗi được định dạng', 'file name': 'tên tệp', 'environment variable': 'biến môi trường' } as Record<string, string>)[o.kind] ?? o.kind)})</span> : null}
                </button>
              );
            })}
          </div>
          {pick != null && !right ? (
            <p className="t-fail" style={{ margin: '0.5rem 0 0' }} data-readout="why">{h.options[pick].why} <Translated vi="Thử lựa chọn khác.">Try another.</Translated></p>
          ) : null}
          {right ? (
            <div style={{ marginTop: '0.6rem' }} data-readout="found">
              <p style={{ margin: '0 0 0.4rem' }}>
                <span className="t-ok"><b><Translated vi="Đã tìm thấy.">Found.</Translated></b></span> {h.found}
              </p>
              <p className="muted" style={{ margin: '0 0 0.5rem', fontSize: '0.84rem' }}>
                <a href={permalink(h.project, h.path, h.lines ?? null, !/\.\w+$/.test(h.path))} target="_blank" rel="noreferrer" data-part="permalink">
                  {h.path}{h.lines != null ? ', ' + linesText(h.lines) : ''}
                </a>{' '}
                <Translated vi="tại">at</Translated> {shortCommit(h.project)}; {h.facts.map((f) => (f.includes(' ') ? f : 'x3 ' + f)).join(', ')}.
              </p>
              <button className="primary" onClick={next}>{hop + 1 < r.hops.length ? tr('Next hop', 'Bước tiếp') : tr('Finish', 'Hoàn tất')}</button>
            </div>
          ) : null}
        </div>
      ) : (
        <div className="panel" data-readout="end">
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>
              <Translated vi={<>Xong: {score.right} trên {score.answered} bước đúng ngay lựa chọn đầu tiên.</>}>
                Done: {score.right} of {score.answered} hops right on the first choice.
              </Translated>
            </b> {r.lesson}
          </p>
          <button onClick={() => reset(choice)}><Translated vi="Đi lại lần nữa">Walk it again</Translated></button>
        </div>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- the section
export function SX31() {
  const q = questionsSX31;
  const x = EXCERPTS_SX31;
  return (
    <>
      <Translated
        vi={
          <p>
            Mọi framework bạn dùng đều có một chỗ mà op bạn gọi trở thành một lời gọi vào một backend: một kernel CUDA, một
            kernel HIP, một pipeline Metal. Tìm ra chỗ đó, và mỗi op chỉ còn cách một lần tìm kiếm ngắn. Mục này chỉ ra chỗ đó
            nằm ở đâu trong llama.cpp, MLX và PyTorch, cùng hai thứ làm hỏng cách tìm đơn giản: gộp (fusion) và mã được
            sinh tự động.
          </p>
        }
      >
        <p>
          Every framework you use has a place where an op you called becomes a call into one backend: a CUDA kernel, a HIP
          kernel, a Metal pipeline. Find that place, and each op is a short search away. This section shows where it is in
          llama.cpp, MLX and PyTorch, and the two things that break the simple search: fusion and code that is generated.
        </p>
      </Translated>
      <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
        <Translated
          vi={
            <p>
              Module X3 không có trong sách. Mọi khẳng định về một codebase trên trang này đều được đọc trong lượt kiểm tra dữ
              kiện ngày 2026-11-20 tại các commit ghi ở <a href={href('x3', 'start')}>trang bắt đầu</a>. Không có gì được build
              hay chạy. Các quy tắc đọc là của khóa học này.
            </p>
          }
        >
          <p>
            Module X3 is not in the book. Every claim about a codebase on this page was read in the facts pass on 2026-11-20 at
            the commits on the <a href={href('x3', 'start')}>start page</a>. Nothing was built or run. The reading rules are ours.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="llama.cpp: mỗi backend một switch">llama.cpp: one switch per backend</Translated></h2>
      <Translated
        vi={
          <p>
            Trong llama.cpp, một mô hình là một đồ thị các tensor, và op của mỗi node là một giá trị enum{' '}
            <code>GGML_OP_*</code>. Backend CUDA biến một node thành công việc trong một hàm duy nhất,{' '}
            <code>ggml_cuda_compute_forward</code>, vốn là một switch duy nhất trên giá trị đó. Mỗi case gọi một hàm host đặt
            tên theo op, và mỗi hàm host nằm trong file <code>.cu</code> cùng tên: <code>getrows.cu</code>,{' '}
            <code>norm.cu</code>, <code>softmax.cu</code>, <code>rope.cu</code>, <code>fattn.cu</code>. Backend HIP là cùng
            switch đó, vì nó là cùng file được biên dịch dưới dạng HIP.
          </p>
        }
      >
        <p>
          In llama.cpp a model is a graph of tensors, and each node’s op is a <code>GGML_OP_*</code> enum value. The CUDA
          backend turns a node into work in one function, <code>ggml_cuda_compute_forward</code>, which is a single switch on
          that value. Each case calls a host function named after the op, and each host function lives in the{' '}
          <code>.cu</code> file of the same name: <code>getrows.cu</code>, <code>norm.cu</code>, <code>softmax.cu</code>,{' '}
          <code>rope.cu</code>, <code>fattn.cu</code>. The HIP backend is the same switch, because it is the same file compiled
          as HIP.
        </p>
      </Translated>
      <Today checked={CHECKED('A1, B9')}>
        <Translated
          vi={
            <p>
              <code>ggml_cuda_compute_forward</code> ở dòng 2812; <code>ggml_backend_cuda_device_supports_op</code> (dòng 6336)
              cho biết backend có chấp nhận một op hay không. <code>ggml-hip/CMakeLists.txt</code> biên dịch{' '}
              <code>../ggml-cuda/*.cu</code> với ngôn ngữ HIP.
            </p>
          }
        >
          <p>
            <code>ggml_cuda_compute_forward</code> at line 2812; <code>ggml_backend_cuda_device_supports_op</code> (line 6336)
            says whether the backend accepts an op at all. <code>ggml-hip/CMakeLists.txt</code> compiles{' '}
            <code>../ggml-cuda/*.cu</code> as language HIP.
          </p>
        </Translated>
      </Today>
      <Excerpt
        x={x.switch}
        caption={
          <Translated vi="Dispatch của CUDA: một switch trên enum op, ba case của nó.">
            The CUDA dispatch: one switch over the op enum, three of its cases.
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Có một điểm cần lưu ý. Trước khi switch đó thấy một node, một lượt gộp xem xét các chuỗi node. Nếu một
            RMS_NORM được theo sau bởi một MUL (trọng số của phép chuẩn hóa), cặp đó đi tới một lời gọi gộp, và case
            riêng của switch cho RMS_NORM không bao giờ chạy. Trong một đồ thị Llama, kernel chạy cho RMS_NORM thường là kernel
            gộp, nên hàm host bạn tìm thấy đầu tiên không phải là hàm thực sự chạy.
          </p>
        }
      >
        <p>
          There is a catch. Before that switch sees a node, a fusion pass looks at runs of nodes. If an RMS_NORM is followed by
          a MUL (the norm’s weight), the pair goes to one fused call, and the switch’s own case for RMS_NORM never runs. In a
          Llama graph the kernel that runs for RMS_NORM is usually the fused one, so the host function you find first is not
          the one that runs.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              <code>GGML_CUDA_DISABLE_FUSION</code>, được đọc ở dòng 181, tắt lượt này. Dùng nó khi bạn muốn đường đi thường:
              để đặt breakpoint, hoặc để so sánh hai đường.
            </p>
          }
        >
          <p>
            <code>GGML_CUDA_DISABLE_FUSION</code>, read at line 181, switches the pass off. Use it when you want the plain path:
            to set a breakpoint, or to compare the two.
          </p>
        </Translated>
      </Aside>
      <Excerpt
        x={x.fusion}
        caption={
          <Translated vi="Lượt gộp, trước switch. RMS_NORM, MUL, ADD được thử sớm hơn một bước.">
            The fusion pass, before the switch. RMS_NORM, MUL, ADD is tried one step earlier.
          </Translated>
        }
      />

      <h3><Translated vi="Metal tìm kernel bằng chuỗi">Metal finds kernels by string</Translated></h3>
      <Translated
        vi={
          <p>
            Backend Metal có switch riêng, <code>ggml_metal_op_encode_impl</code>, trên cùng enum. Mỗi case mã hóa công việc
            và trả về <code>n_fuse</code>, số node đồ thị mà nó đã tiêu thụ. Để lấy một pipeline, host dựng tên kernel dưới
            dạng một chuỗi và yêu cầu nó bằng tên đó:
          </p>
        }
      >
        <p>
          The Metal backend has its own switch, <code>ggml_metal_op_encode_impl</code>, over the same enum. Each case encodes
          the work and returns <code>n_fuse</code>, the number of graph nodes it consumed. To get a pipeline, the host builds
          the kernel’s name as a string and asks for it by that name:
        </p>
      </Translated>
      <Excerpt
        x={x.snprintf}
        caption={<Translated vi="Host dựng tên của kernel mà nó cần.">The host builds the name of the kernel it wants.</Translated>}
      />
      <Translated
        vi={
          <p>
            Trong mã nguồn <code>.metal</code>, thuộc tính <code>[[host_name(...)]]</code> gán cho một hàm, hoặc một bản khởi
            tạo của một template, cái tên mà host sẽ yêu cầu. Vì vậy tên template C++ được mọi biến thể dùng chung, còn chuỗi
            cho bạn biết biến thể nào chạy.
          </p>
        }
      >
        <p>
          In the <code>.metal</code> source, the attribute <code>[[host_name(...)]]</code> gives a function, or one
          instantiation of a template, the name the host will ask for. So the C++ template name is shared by every variant,
          and the string tells you which one runs.
        </p>
      </Translated>
      <Today checked={CHECKED('C2')}>
        <Translated
          vi={
            <p>
              MSL 4.1: “Từ Metal 2.2 trở đi, bạn có thể ghi đè tên mặc định mà Metal Framework API dùng để chỉ một hàm đủ điều
              kiện. Thêm thuộc tính [[host_name(name)]]…” Template cần “một tên khác nhau cho mỗi bản chuyên biệt hóa.”
            </p>
          }
        >
          <p>
            MSL 4.1: “In Metal 2.2 and later, you can override the default name that the Metal Framework API uses to refer to a
            qualified function. Add the [[host_name(name)]] attribute…” Templates need “a different name for each
            specialization.”
          </p>
        </Translated>
      </Today>
      <Beyond title={tr('Reading rule (ours)', 'Quy tắc đọc (của khóa học này)')}>
        <Translated vi={<p>Để tìm kernel Metal đứng sau một lời gọi từ host, hãy tìm trong các file .metal chuỗi mà host định dạng, không phải tên một hàm C++.</p>}>
          <p>To find the Metal kernel behind a host call, search the .metal files for the string the host formats, not for a C++ function name.</p>
        </Translated>
      </Beyond>
      <Excerpt
        x={x.hostName}
        caption={
          <Translated vi="Một bản khởi tạo của template RMSNorm, mức gộp 2, được đặt tên cho host.">
            One instantiation of the RMSNorm template, fusion level 2, named for the host.
          </Translated>
        }
      />

      <h2><Translated vi="MLX: mỗi primitive một phương thức, cho mỗi backend">MLX: one method per primitive, per backend</Translated></h2>
      <Translated
        vi={
          <p>
            Các lớp Python của MLX gọi các op lõi hoặc <code>mx.fast.*</code>. <code>nn.RMSNorm</code> chỉ là một dòng,{' '}
            <code>mx.fast.rms_norm</code>; <code>nn.Embedding</code> là <code>self.weight[x]</code>, một phép gather. Mỗi op trở
            thành một lớp con <code>Primitive</code> trong đồ thị, và mỗi thư mục backend cài đặt <code>eval_gpu</code> của
            primitive đó một lần: <code>RMSNorm::eval_gpu</code> nằm trong <code>metal/normalization.cpp</code> và trong{' '}
            <code>cuda/rms_norm.cu</code>, <code>Softmax::eval_gpu</code> trong <code>cuda/softmax.cu</code> và{' '}
            <code>metal/softmax.cpp</code>. Vậy thứ cần tìm là <code>Name::eval_gpu</code> trong{' '}
            <code>mlx/backend/cuda</code> hoặc <code>mlx/backend/metal</code>. Các op theo từng phần tử được dập ra bởi một
            macro, <code>BINARY_GPU(Add)</code>, nên bạn tìm chỗ dùng macro đó thay vào.
          </p>
        }
      >
        <p>
          MLX’s Python layers call core ops or <code>mx.fast.*</code>. <code>nn.RMSNorm</code> is one line,{' '}
          <code>mx.fast.rms_norm</code>; <code>nn.Embedding</code> is <code>self.weight[x]</code>, a gather. Each op becomes a{' '}
          <code>Primitive</code> subclass in the graph, and each backend directory implements that primitive’s{' '}
          <code>eval_gpu</code> once: <code>RMSNorm::eval_gpu</code> is in <code>metal/normalization.cpp</code> and in{' '}
          <code>cuda/rms_norm.cu</code>, <code>Softmax::eval_gpu</code> in <code>cuda/softmax.cu</code> and{' '}
          <code>metal/softmax.cpp</code>. So the search is for <code>Name::eval_gpu</code> in{' '}
          <code>mlx/backend/cuda</code> or <code>mlx/backend/metal</code>. Elementwise ops are stamped out by a macro,{' '}
          <code>BINARY_GPU(Add)</code>, so you search for the macro’s use instead.
        </p>
      </Translated>
      <Excerpt
        x={x.rmsNormPy}
        caption={<Translated vi="nn.RMSNorm.__call__: toàn bộ lớp này.">nn.RMSNorm.__call__: the whole layer.</Translated>}
      />
      <Translated
        vi={
          <p>
            Ngoại lệ là mọi thứ nằm dưới <code>mx.compile</code>. <code>nn.silu</code> là một hàm như vậy, và{' '}
            <code>swiglu</code> của mlx-lm cũng thế. Kernel của nó được viết thành văn bản mã nguồn ở lần chạy đầu tiên, bởi{' '}
            <code>Compiled::eval_gpu</code>, với <code>[[host_name]]</code> trên Metal và <code>__global__</code> trên CUDA, và
            được biên dịch khi đó. Không có file nào để mở.
          </p>
        }
      >
        <p>
          The exception is anything under <code>mx.compile</code>. <code>nn.silu</code> is such a function, and so is mlx-lm’s{' '}
          <code>swiglu</code>. Its kernel is written as source text the first time it runs, by <code>Compiled::eval_gpu</code>,
          with <code>[[host_name]]</code> on Metal and <code>__global__</code> on CUDA, and compiled then. There is no file to
          open.
        </p>
      </Translated>
      <Today checked={CHECKED('A2, B9')}>
        <Translated
          vi={
            <p>
              <code>metal/compiled.cpp</code> dòng 257 viết <code>[[host_name("{'{0}'}")]]</code>; <code>cuda/compiled.cpp</code>{' '}
              dòng 225 viết <code>__global__ void {'{}'}(</code> và build thông qua <code>cu::get_jit_module</code>. MLX không có
              backend HIP: <code>mlx/backend</code> chứa common, cpu, cuda, gpu, metal, no_cpu, no_gpu; pull request ROCm #2300
              đang mở.
            </p>
          }
        >
          <p>
            <code>metal/compiled.cpp</code> line 257 writes <code>[[host_name("{'{0}'}")]]</code>; <code>cuda/compiled.cpp</code> line
            225 writes <code>__global__ void {'{}'}(</code> and builds through <code>cu::get_jit_module</code>. MLX has no HIP
            backend: <code>mlx/backend</code> holds common, cpu, cuda, gpu, metal, no_cpu, no_gpu; ROCm pull request #2300 is open.
          </p>
        </Translated>
      </Today>
      <Excerpt
        x={x.siluPy}
        caption={
          <Translated vi="nn.silu: được biên dịch, nên kernel của nó được sinh ra lúc chạy.">
            nn.silu: compiled, so its kernel is generated at run time.
          </Translated>
        }
      />

      <h2><Translated vi="PyTorch: file yaml chính là bản đồ">PyTorch: the yaml is the map</Translated></h2>
      <Translated
        vi={
          <p>
            Trong PyTorch, bản đồ từ op tới kernel là một file, <code>aten/src/ATen/native/native_functions.yaml</code>. Mỗi op
            có một mục <code>- func:</code>, và các khóa <code>dispatch:</code> của nó nêu tên một hàm cho mỗi backend:{' '}
            <code>CUDA: softmax_cuda_out</code>, <code>MPS: softmax_mps_out</code>. ROCm không có khóa riêng: cùng các mã nguồn{' '}
            <code>native/cuda</code> được hipify khi PyTorch được build cho AMD.
          </p>
        }
      >
        <p>
          In PyTorch the map from op to kernel is one file, <code>aten/src/ATen/native/native_functions.yaml</code>. Each op has
          a <code>- func:</code> entry, and its <code>dispatch:</code> keys name a function per backend:{' '}
          <code>CUDA: softmax_cuda_out</code>, <code>MPS: softmax_mps_out</code>. ROCm has no key of its own: the same{' '}
          <code>native/cuda</code> sources are hipified when PyTorch is built for AMD.
        </p>
      </Translated>
      <Beyond title={tr('Reading rule (ours)', 'Quy tắc đọc (của khóa học này)')}>
        <Translated
          vi={
            <p>
              Tìm dòng <code>- func:</code> của op và đọc <code>dispatch:</code>. Nếu không có khóa cho backend của bạn, hãy lần
              theo <code>structured_delegate</code>, <code>ufunc_inner_loop</code>, hoặc <code>_stub</code> tới{' '}
              <code>REGISTER_DISPATCH</code> của nó.
            </p>
          }
        >
          <p>
            Find the op’s <code>- func:</code> line and read <code>dispatch:</code>. With no key for your backend, follow{' '}
            <code>structured_delegate</code>, <code>ufunc_inner_loop</code>, or the <code>_stub</code> to its{' '}
            <code>REGISTER_DISPATCH</code>.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Bạn biết <code>torch.nn.functional.rms_norm</code> từ một lớp Llama của Hugging Face. Nó gọi{' '}
            <code>rms_norm_symint</code>, hàm này gọi operator <code>at::_fused_rms_norm</code>, có mục ghi{' '}
            <code>CUDA: _fused_rms_norm_cuda</code>, <code>MPS: _fused_rms_norm_mps</code>. Hai lần tìm nữa dẫn tới{' '}
            <code>layer_norm_kernel.cu</code> trên CUDA và <code>RMSNorm.metal</code> trên MPS. Trình chạy từng bước bên dưới
            đi qua đường đó.
          </p>
        }
      >
        <p>
          You know <code>torch.nn.functional.rms_norm</code> from a Hugging Face Llama layer. It calls{' '}
          <code>rms_norm_symint</code>, which calls the operator <code>at::_fused_rms_norm</code>, whose entry reads{' '}
          <code>CUDA: _fused_rms_norm_cuda</code>, <code>MPS: _fused_rms_norm_mps</code>. Two more searches land in{' '}
          <code>layer_norm_kernel.cu</code> on CUDA and <code>RMSNorm.metal</code> on MPS. The stepper below walks it.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Không phải op nào cũng có khóa backend. Op phổ biến nhất, <code>add.out</code>, là một op ufunc: mục của nó có một{' '}
            <code>ufunc_inner_loop</code> và không có dòng <code>CUDA:</code>. Thân vô hướng của nó là một dòng C++, và kernel
            CUDA của nó là một file mà torchgen viết ra trong lúc build, <code>UfuncCUDA_add.cu</code>, gọi{' '}
            <code>gpu_kernel(iter, ufunctor);</code>. Nó không nằm trong file mã nguồn nào bạn có thể grep.
          </p>
        }
      >
        <p>
          Not every op has a backend key. The most common one, <code>add.out</code>, is a ufunc op: its entry has a{' '}
          <code>ufunc_inner_loop</code> and no <code>CUDA:</code> line. Its scalar body is one line of C++, and its CUDA kernel
          is a file that torchgen writes during the build, <code>UfuncCUDA_add.cu</code>, calling{' '}
          <code>gpu_kernel(iter, ufunctor);</code>. It is in no source file you can grep.
        </p>
      </Translated>
      <Today checked={CHECKED('A3, B2; ch01 D4, ch24 D3')}>{/* i18n-ignore: facts item ids */}
        <Translated
          vi={
            <p>
              <code>add.out</code>: <code>structured: True</code>, <code>ufunc_inner_loop: Generic: add (...)</code> (các dòng
              565 đến 571). MPS: “add.out là một op ufunc, và codegen ufunc chỉ sinh bản cài đặt CPU/CUDA, nên MPS cần mục này.”
              ROCm: <code>tools/amd_build/build_amd.py</code> hipify <code>native/cuda/*</code>.
            </p>
          }
        >
          <p>
            <code>add.out</code>: <code>structured: True</code>, <code>ufunc_inner_loop: Generic: add (...)</code> (lines 565 to
            571). MPS: “add.out is a ufunc op, and ufunc codegen only emits CPU/CUDA impls, so MPS needs this entry.” ROCm:{' '}
            <code>tools/amd_build/build_amd.py</code> hipifies <code>native/cuda/*</code>.
          </p>
        </Translated>
      </Today>
      <Excerpt
        x={x.addH}
        caption={
          <Translated vi="Phần mã nguồn riêng cho op duy nhất của kernel add CUDA: thân vô hướng của nó trong native/ufunc/add.h. torchgen sinh phần còn lại lúc build.">
            The only op-specific source of the CUDA add kernel: its scalar body in native/ufunc/add.h. torchgen generates the rest at build time.
          </Translated>
        }
      />

      <h2><Translated vi="Ba điểm vào đặt cạnh nhau">The three entry points side by side</Translated></h2>
      <table className="data wide" data-part="entry-points" style={{ maxWidth: '62rem' }}>
        <thead>
          <tr>
            <th><Translated vi="Dự án">Project</Translated></th>
            <th><Translated vi="Một op là">An op is</Translated></th>
            <th>CUDA</th>
            <th>HIP</th>
            <th>Metal</th>
          </tr>
        </thead>
        <tbody>
          {ENTRY_POINTS.map((e) => (
            <tr key={e.project} data-project={e.project}>
              <td><b>{PROJECT_NAME[e.project]}</b></td>
              <td>{e.what}</td>
              <td>{e.cuda}</td>
              <td>{e.hip}</td>
              <td>{e.metal}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Ba hàng trong bảng “Where the backends really differ” của file dữ kiện nói cùng một điều trên các dự án. Trên CUDA
            và HIP, host gọi tên kernel bằng symbol của nó và các biến thể là tham số template; trên Metal, host gọi tên nó bằng
            một chuỗi, và các biến thể là function constant hoặc các bản khởi tạo có tên. Mục{' '}
            <a href={href('x3', 'X3.2')}>X3.2</a> đọc cả hai cơ chế trong các kernel thật.
          </p>
        }
      >
        <p>
          Three rows of the facts file’s table “Where the backends really differ” say the same thing across projects. On
          CUDA and HIP the host names a kernel by its symbol and variants are template parameters; on Metal the host names it
          by a string, and variants are function constants or named instantiations. Section{' '}
          <a href={href('x3', 'X3.2')}>X3.2</a> reads both mechanisms in real kernels.
        </p>
      </Translated>
      <table className="data wide" data-part="differ" style={{ maxWidth: '62rem' }}>
        <thead>
          <tr>
            <th><Translated vi="Tính năng">Feature</Translated></th>
            <th>CUDA</th>
            <th>HIP (ROCm)</th>{/* i18n-ignore: backend name */}
            <th>Metal</th>
          </tr>
        </thead>
        <tbody>
          {DIFFER_ROWS.map((r) => (
            <tr key={r.feature}>
              <td>{r.feature}</td>
              {[r.cuda, r.hip, r.metal].map((t, i) => (
                <td key={i}>{t.split('`').map((s, k) => (k % 2 ? <code key={k}>{s}</code> : s))}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      <h2><Translated vi="Tự tìm lấy">Find it yourself</Translated></h2>
      <Translated
        vi={
          <p>
            Bản đồ op chứa kết quả của việc lần theo mọi op trong chuyến tham quan, ở mọi dự án và backend. Trước khi mở nó,
            hãy dự đoán hai cột: HIP chứa gì cho llama.cpp và cho MLX, và những ô nào hoàn toàn không có file.
          </p>
        }
      >
        <p>
          The op map holds the result of following every op in the tour in every project and backend. Before you open it,
          predict two columns: what HIP holds for llama.cpp and for MLX, and which cells have no file at all.
        </p>
      </Translated>
      <OpMap />
      <Translated
        vi={
          <p>
            Bạn hẳn đã thấy rằng HIP là file CUDA trong llama.cpp, là mã nguồn CUDA đã hipify trong PyTorch (trừ chỗ nó gọi
            thư viện AMD), và trống trong MLX; và rằng một số ô không có file: add CUDA của PyTorch, SiLU của MLX, RoPE của
            PyTorch, vốn hoàn toàn không phải một operator.
          </p>
        }
      >
        <p>
          You should have found that HIP is the CUDA file in llama.cpp, the hipified CUDA source in PyTorch (except where it
          calls AMD libraries), and empty in MLX; and that some cells have no file: PyTorch’s CUDA add, MLX’s SiLU, PyTorch’s
          RoPE, which is not an operator at all.
        </p>
      </Translated>
      <Ask q={q['x3/X3.1/q-no-add-file']} />
      <Ask q={q['x3/X3.1/q-silu-file']} />
      <Translated
        vi={
          <p>
            Bản đồ op cho thấy bạn sẽ đến đâu. Trình chạy từng bước bắt bạn tự đi từng bước: ở mỗi bước bạn chọn thứ cần tìm
            trước khi nó được hé lộ.
          </p>
        }
      >
        <p>
          The op map shows where you end up. The stepper makes you take the hops: at each one you choose what to search for
          before it is revealed.
        </p>
      </Translated>
      <FollowTheCall />
      <Ask q={q['x3/X3.1/q-fusion-breakpoint']} />
      <Ask q={q['x3/X3.1/q-metal-string']} />

      <h2><Translated vi="Một lời gọi, quan sát từ đầu đến cuối">One call, watched</Translated></h2>
      <Translated
        vi={
          <p>
            Hoạt cảnh theo một RMSNorm đi qua các backend CUDA và Metal của llama.cpp, theo thứ tự: lượt gộp tiêu thụ hai
            node, kernel nó khởi chạy, và chuỗi mà Metal tra cứu.
          </p>
        }
      >
        <p>
          The animation follows one RMSNorm through llama.cpp’s CUDA and Metal backends, in order: the fusion pass consuming two
          nodes, the kernel it launches, and the string Metal looks up.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Đồ thị là phần đầu của một lớp Llama như llama.cpp sẽ giữ nó: thứ tự của bốn node là của khóa học này. Tên kernel,
              các dòng và quy tắc gộp là của file dữ kiện.
            </p>
          }
        >
          <p>
            The graph is the start of a Llama layer as llama.cpp would hold it: the order of the four nodes is ours. The kernel
            names, lines and fusion rule are the facts file’s.
          </p>
        </Translated>
      </Aside>
      <OneCallScene />

      <h2><Translated vi="HIP và Metal, trong một đoạn">HIP and Metal, in one paragraph</Translated></h2>
      <Translated
        vi={
          <p>
            Với HIP thì có rất ít thứ để tìm. Trong llama.cpp, kernel HIP chính là file CUDA. Trong PyTorch, nó là file CUDA
            sau một bước chuyển đổi, trừ các thư viện ma trận và attention, vốn khác trên AMD. MLX không có backend HIP. Với
            Metal thì luôn có một chuỗi ở giữa: llama.cpp định dạng một chuỗi, MLX viết một chuỗi vào mã nguồn được sinh ra, và
            mã MPS của PyTorch dựng một chuỗi bằng <code>fmt::format</code> (<code>RMSNorm.mm</code>) trước khi yêu cầu
            pipeline.
          </p>
        }
      >
        <p>
          For HIP there is little to find. In llama.cpp the HIP kernel is the CUDA file. In PyTorch it is the CUDA file after a
          translation step, except for the matrix and attention libraries, which differ on AMD. MLX has no HIP backend. For
          Metal there is always a string in the middle: llama.cpp formats one, MLX writes one into generated source, and
          PyTorch’s MPS code builds one with <code>fmt::format</code> (<code>RMSNorm.mm</code>) before asking for the pipeline.
        </p>
      </Translated>
      <Ask q={q['x3/X3.1/q-mlx-hip']} />
    </>
  );
}
