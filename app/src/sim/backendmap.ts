// Where an op becomes a kernel in llama.cpp, MLX and PyTorch, on CUDA, HIP and Metal, as read
// in the X3 facts pass (notes/today/x3-facts.md, 2026-11-20, at the commits in x3sources.ts).
//
// Every cell cites the facts items it rests on. Line numbers appear only where x3-facts gives
// them, at the pinned commit. Cells that rest on another chapter's facts item (read at master
// the same day) say so with readAt: 'master' and carry that item's master URL, never pinned
// line numbers. A cell whose file was not read says so with the tag 'not read in the facts
// pass' instead of guessing.
//
// Also here, because the lessons draw them from the same data: the side-by-side entry points
// (ENTRY_POINTS), the routes the "Follow the call" stepper walks (ROUTES), the variant
// mechanisms X3.2 sorts (VARIANT_MECHANISMS), and the graph and names the "One RMSNorm call"
// scene animates (ONE_CALL, fuseGraph).
//
// Reader text is held in getters that call tr(), so it follows the language chosen when it is
// read, not the one in force when this module was imported.

import { permalink, type Lines, type ProjectId } from './x3sources';
import { tr } from '../lib/locale.ts';

export type OpId = 'embedding' | 'add' | 'silu' | 'rmsnorm' | 'softmax' | 'matmul' | 'rope' | 'attention' | 'sampling';
export type Project = 'llama.cpp' | 'mlx' | 'pytorch';
export type Backend = 'cuda' | 'hip' | 'metal';
export const PROJECTS: Project[] = ['llama.cpp', 'mlx', 'pytorch'];
export const BACKENDS: Backend[] = ['cuda', 'hip', 'metal'];
export const PROJECT_NAME: Record<Project, string> = { 'llama.cpp': 'llama.cpp', mlx: 'MLX', pytorch: 'PyTorch' };
export const BACKEND_NAME: Record<Backend, string> = { cuda: 'CUDA', hip: 'HIP', metal: 'Metal' };

export const TAGS = [
  'own kernel', // i18n-ignore: identifier also shown to the reader
  'same source as CUDA', // i18n-ignore: identifier also shown to the reader
  'hipified at build time', // i18n-ignore: identifier also shown to the reader
  'generated at build time', // i18n-ignore: identifier also shown to the reader
  'generated at run time (mx.compile)', // i18n-ignore: identifier also shown to the reader
  'vendor library', // i18n-ignore: identifier also shown to the reader
  'third-party submodule', // i18n-ignore: identifier also shown to the reader
  'composite of other ops', // i18n-ignore: identifier also shown to the reader
  'fused before dispatch', // i18n-ignore: identifier also shown to the reader
  'found by name string', // i18n-ignore: identifier also shown to the reader
  'not read in the facts pass', // i18n-ignore: identifier also shown to the reader
  'no backend', // i18n-ignore: identifier also shown to the reader
] as const;
export type Tag = (typeof TAGS)[number];

/** A further permalink at the pinned commit: an entry point, a host function, a build file. */
export interface Where {
  label: string;
  path: string;
  lines?: Lines;
  /** Defaults to the cell's project. */
  project?: ProjectId;
  tree?: boolean;
}

export interface Cell {
  /** Where the op enters this backend. */
  entry: string;
  /** The host function that launches or encodes the work. */
  host: string | null;
  /** The kernel's file, relative to the repository root; null when there is no file to open. */
  file: string | null;
  /** The kernel function(s), or what stands in for them. */
  kernel: string | null;
  /** Lines of `file`, only where x3-facts.md gives them. */
  lines?: Lines;
  /** 'pinned': read at the commit in x3sources. 'master': from another chapter's facts item. */
  readAt: 'pinned' | 'master';
  /** For readAt 'master': the URL that facts item gives. */
  url?: string;
  also?: Where[];
  /** Facts items: "A1" for x3-facts.md, "ch10 D2" for notes/today/ch10-facts.md. */
  facts: string[];
  tags: Tag[];
  note?: string;
}

export interface CourseLab {
  num: string;
  /** Chapter id for the link to its labs page. */
  chapter: string;
  /** Directory under labs/, or null for a lab still planned. */
  dir: string | null;
  what: string;
  planned?: boolean;
}

export interface OpRow {
  id: OpId;
  label: string;
  /** The tour's eight ops group add and SiLU as "elementwise". */
  tour: string;
  /** The section of module X3 that reads it. */
  section: string;
  course: CourseLab[];
  cells: Record<Project, Record<Backend, Cell>>;
}

// ---------------------------------------------------------------- helpers for the HIP columns
const LLAMA_HIP_NOTE = 'ggml-hip/CMakeLists.txt compiles ../ggml-cuda/*.cu with LANGUAGE HIP (lines 60 to 70).'; // i18n-ignore: translated by tr() where it is read
const LLAMA_HIP_CMAKE: Where = { label: 'ggml-hip/CMakeLists.txt', path: 'ggml/src/ggml-hip/CMakeLists.txt', lines: [60, 70] };

/** llama.cpp's HIP cell: the CUDA file itself. */
function llamaHip(c: Cell): Cell {
  return { ...c, also: [...(c.also ?? []), LLAMA_HIP_CMAKE], facts: [...c.facts, 'B9'], tags: ['same source as CUDA'], get note() { return tr(LLAMA_HIP_NOTE, 'ggml-hip/CMakeLists.txt biên dịch ../ggml-cuda/*.cu với LANGUAGE HIP (dòng 60 đến 70).'); } }; // i18n-ignore: identifier also shown to the reader
}
/** PyTorch's HIP cell: the CUDA source after the build-time hipify step. */
function torchHip(c: Cell): Cell {
  return {
    ...c,
    facts: [...c.facts, 'B9', 'ch24 D3'], // i18n-ignore: identifier also shown to the reader
    tags: [...c.tags.filter((t) => t === 'composite of other ops'), 'hipified at build time'], // i18n-ignore: identifier also shown to the reader
    get note() { return tr('tools/amd_build/build_amd.py translates aten/src/ATen/native/cuda/* at build time (ch24 D3).', 'tools/amd_build/build_amd.py dịch aten/src/ATen/native/cuda/* lúc build (ch24 D3).'); },
  };
}
const MLX_NO_HIP: Cell = {
  get entry() { return tr('none', 'không có'); }, host: null, file: null, kernel: null, readAt: 'pinned',
  also: [{ get label() { return tr('mlx/backend at the pinned commit', 'mlx/backend tại commit được ghim'); }, path: 'mlx/backend', tree: true }],
  facts: ['B9'], tags: ['no backend'], // i18n-ignore: identifier also shown to the reader
  get note() { return tr('mlx/backend holds common, cpu, cuda, gpu, metal, no_cpu and no_gpu. An experimental ROCm pull request (#2300) is open and unmerged.', 'mlx/backend chứa common, cpu, cuda, gpu, metal, no_cpu và no_gpu. Một pull request thử nghiệm cho ROCm (#2300) đang mở và chưa được merge.'); },
};

// ---------------------------------------------------------------- llama.cpp entry points
const CU = 'ggml/src/ggml-cuda/';
const MT = 'ggml/src/ggml-metal/';
const CUDA_SWITCH = (line: number): Where => ({ get label() { return tr(`ggml_cuda_compute_forward, line ${line}`, `ggml_cuda_compute_forward, dòng ${line}`); }, path: CU + 'ggml-cuda.cu', lines: line });
const METAL_SWITCH = (line?: number): Where => ({ get label() { return tr(`ggml_metal_op_encode_impl${line ? ', line ' + line : ' (line 187)'}`, `ggml_metal_op_encode_impl${line ? ', dòng ' + line : ' (dòng 187)'}`); }, path: MT + 'ggml-metal-ops.cpp', lines: line ?? 187 });
const MLX_CU = 'mlx/backend/cuda/';
const MLX_MT = 'mlx/backend/metal/';
const YAML = 'aten/src/ATen/native/native_functions.yaml';
const TN = 'aten/src/ATen/native/';

// The rows below are built when this module is imported, so `what` is translated in the getter,
// when it is read: the Vietnamese of each course lab is looked up by the English the call passes.
const lab = (num: string, chapter: string, dir: string | null, what: string, planned = false): CourseLab => ({
  num,
  chapter,
  dir,
  get what() {
    return tr(
      what,
      ({
        'the Llama pieces the course has not built': 'những phần của Llama mà khóa học chưa xây dựng',
        'launchEmbed, one output row per token': 'launchEmbed, mỗi token một hàng đầu ra',
        'launchAddResidual, one thread per element': 'launchAddResidual, mỗi phần tử một luồng',
        'four elements per thread, float4 loads': 'bốn phần tử mỗi luồng, nạp float4',
        'the elementwise kernel SiLU is': 'loại kernel theo từng phần tử mà SiLU thuộc về',
        'the two-stage row reduction (launchRowMax)': 'phép rút gọn (reduction) theo hàng hai giai đoạn (launchRowMax)',
        'launchSoftmaxRows, max then sum': 'launchSoftmaxRows, max rồi sum',
        'the tiled GEMM': 'GEMM phân ô',
        'M3, the linear layer': 'M3, lớp tuyến tính',
        'GEMV for one token': 'GEMV cho một token',
        'flash attention': 'flash attention',
        'decode with a KV cache': 'decode với KV cache',
        'GQA decode': 'decode GQA',
        'M5, the GPU sampler': 'M5, bộ lấy mẫu trên GPU',
      } as Record<string, string>)[what] ?? what,
    );
  },
  planned,
});
const LAB_X31 = lab('X3.1', 'x3', null, 'the Llama pieces the course has not built', true); // i18n-ignore: translated in lab() when read

export const OPS: OpRow[] = [
  {
    id: 'embedding', get label() { return tr('embedding', 'embedding'); }, tour: 'embedding', section: 'X3.2',
    course: [lab('22.2', 'ch22', 'ch22/lab2_batching', 'launchEmbed, one output row per token')], // i18n-ignore: translated in lab() when read
    cells: {
      'llama.cpp': (() => {
        const cuda: Cell = {
          get entry() { return tr('case GGML_OP_GET_ROWS in ggml_cuda_compute_forward', 'case GGML_OP_GET_ROWS trong ggml_cuda_compute_forward'); }, host: 'ggml_cuda_op_get_rows',
          file: CU + 'getrows.cu', get kernel() { return tr('k_get_rows (line 6), k_get_rows_float (73), k_get_rows_float_vec (105)', 'k_get_rows (dòng 6), k_get_rows_float (73), k_get_rows_float_vec (105)'); }, lines: [6, 105],
          readAt: 'pinned', also: [CUDA_SWITCH(2826)], facts: ['A1', 'B7'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
          get note() { return tr('Quantised rows are dequantised as they are gathered.', 'Các hàng đã lượng tử hóa được giải lượng tử hóa ngay khi được gom về.'); },
        };
        return {
          cuda,
          get hip() { return llamaHip(cuda); },
          metal: {
            get entry() { return tr('case GGML_OP_GET_ROWS in ggml_metal_op_encode_impl', 'case GGML_OP_GET_ROWS trong ggml_metal_op_encode_impl'); },
            get host() { return tr('ggml_metal_op_get_rows, which asks for "kernel_get_rows_%s" with the type name', 'ggml_metal_op_get_rows, hàm yêu cầu "kernel_get_rows_%s" kèm tên kiểu'); },
            file: MT + 'kernels/quantize.metal', get kernel() { return tr('kernel_get_rows_f (line 266), kernel_get_rows_q (297)', 'kernel_get_rows_f (dòng 266), kernel_get_rows_q (297)'); }, lines: [266, 297],
            readAt: 'pinned', also: [METAL_SWITCH(), { get label() { return tr('the snprintf, ggml-metal-device.cpp line 154', 'lệnh snprintf, ggml-metal-device.cpp dòng 154'); }, path: MT + 'ggml-metal-device.cpp', lines: 154 }],
            facts: ['A1', 'B7', 'C2'], tags: ['own kernel', 'found by name string'], // i18n-ignore: identifier also shown to the reader
          },
        };
      })(),
      mlx: {
        cuda: {
          get entry() { return tr('nn.Embedding: return self.weight[x], a gather', 'nn.Embedding: return self.weight[x], một phép gather'); }, host: null, file: null,
          get kernel() { return tr('the gather primitive’s CUDA kernel', 'kernel CUDA của primitive gather'); }, readAt: 'pinned',
          also: [{ get label() { return tr('embedding.py line 32', 'embedding.py dòng 32'); }, path: 'python/mlx/nn/layers/embedding.py', lines: 32 }],
          facts: ['A2', 'B7'], tags: ['composite of other ops', 'not read in the facts pass'], // i18n-ignore: identifier also shown to the reader
        },
        hip: MLX_NO_HIP,
        metal: {
          get entry() { return tr('nn.Embedding: return self.weight[x], a gather', 'nn.Embedding: return self.weight[x], một phép gather'); }, host: null, file: null,
          get kernel() { return tr('the gather primitive’s Metal kernel', 'kernel Metal của primitive gather'); }, readAt: 'pinned',
          also: [{ get label() { return tr('embedding.py line 32', 'embedding.py dòng 32'); }, path: 'python/mlx/nn/layers/embedding.py', lines: 32 }],
          facts: ['A2', 'B7'], tags: ['composite of other ops', 'not read in the facts pass'], // i18n-ignore: identifier also shown to the reader
        },
      },
      pytorch: (() => {
        const cuda: Cell = {
          entry: '- func: embedding, dispatch CompositeExplicitAutograd: embedding_symint', // i18n-ignore: a native_functions.yaml line, quoted as code
          host: 'embedding_symint: weight.index_select(0, indices.reshape(-1))', file: null, // i18n-ignore: code
          get kernel() { return tr('index_select’s CUDA kernel', 'kernel CUDA của index_select'); }, readAt: 'pinned',
          also: [{ get label() { return tr('native_functions.yaml lines 2261 to 2263', 'native_functions.yaml dòng 2261 đến 2263'); }, path: YAML, lines: [2261, 2263] }, { get label() { return tr('Embedding.cpp lines 35 to 55', 'Embedding.cpp dòng 35 đến 55'); }, path: TN + 'Embedding.cpp', lines: [35, 55] }],
          facts: ['A3', 'B7'], tags: ['composite of other ops', 'not read in the facts pass'], // i18n-ignore: identifier also shown to the reader
        };
        return {
          cuda,
          get hip() { return torchHip(cuda); },
          get metal() { return { ...cuda, get kernel() { return tr('index_select’s MPS kernel', 'kernel MPS của index_select'); }, get note() { return tr('The same composite: there is no embedding entry per backend.', 'Vẫn là op tổ hợp đó: không có mục embedding riêng cho từng backend.'); } }; },
        };
      })(),
    },
  },
  {
    id: 'add', get label() { return tr('add', 'add'); }, tour: 'elementwise', section: 'X3.2',
    course: [lab('2.3', 'ch02', 'ch02/lab3_elementwise', 'launchAddResidual, one thread per element'), lab('6.2', 'ch06', 'ch06/lab2_vector', 'four elements per thread, float4 loads')], // i18n-ignore: translated in lab() when read
    cells: {
      'llama.cpp': (() => {
        const cuda: Cell = {
          get entry() { return tr('case GGML_OP_ADD in ggml_cuda_compute_forward', 'case GGML_OP_ADD trong ggml_cuda_compute_forward'); }, host: 'ggml_cuda_op_add', file: CU + 'binbcast.cu', kernel: 'k_bin_bcast',
          readAt: 'pinned', also: [CUDA_SWITCH(2847)], facts: ['A1', 'B1', 'ch03 D1'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
          get note() { return tr('ggml_cuda_op_fused_add fuses 2 to 8 consecutive adds.', 'ggml_cuda_op_fused_add gộp 2 đến 8 phép add liên tiếp.'); },
        };
        return {
          cuda,
          get hip() { return llamaHip(cuda); },
          metal: {
            get entry() { return tr('case GGML_OP_ADD (with SUB, MUL, DIV) in ggml_metal_op_encode_impl', 'case GGML_OP_ADD (cùng SUB, MUL, DIV) trong ggml_metal_op_encode_impl'); }, host: 'ggml_metal_op_bin',
            file: MT + 'kernels/binbcast.metal', get kernel() { return tr('kernel_bin_fuse_impl, which also fuses several ops', 'kernel_bin_fuse_impl, kernel này cũng gộp nhiều op'); }, readAt: 'pinned',
            also: [METAL_SWITCH(291)], facts: ['A1', 'B1'], tags: ['own kernel', 'found by name string'], // i18n-ignore: identifier also shown to the reader
          },
        };
      })(),
      mlx: {
        cuda: {
          get entry() { return tr('BINARY_GPU(Add) in cuda/binary/add.cu', 'BINARY_GPU(Add) trong cuda/binary/add.cu'); }, get host() { return tr('Add::eval_gpu, stamped out by the BINARY_GPU macro', 'Add::eval_gpu, được macro BINARY_GPU sinh ra'); },
          file: MLX_CU + 'binary/binary.cuh', kernel: 'binary_vv<Op, In, Out, IdxT, N_READS>', lines: [93, 118], readAt: 'pinned',
          also: [{ get label() { return tr('the macro, binary.cuh lines 399 to 406', 'macro đó, binary.cuh dòng 399 đến 406'); }, path: MLX_CU + 'binary/binary.cuh', lines: [399, 406] }],
          facts: ['A2', 'B1'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
        },
        hip: MLX_NO_HIP,
        metal: {
          get entry() { return tr('BINARY_GPU(Add) in metal/binary.cpp', 'BINARY_GPU(Add) trong metal/binary.cpp'); }, get host() { return tr('Add::eval_gpu, stamped out by the same macro name', 'Add::eval_gpu, được sinh ra bởi macro cùng tên'); },
          file: MLX_MT + 'kernels/binary.h', get kernel() { return tr('binary_ss, binary_vv with WorkPerThread elements per thread', 'binary_ss, binary_vv với WorkPerThread phần tử mỗi luồng'); }, readAt: 'master',
          url: 'https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/binary.h',
          also: [{ get label() { return tr('binary.cpp line 217', 'binary.cpp dòng 217'); }, path: MLX_MT + 'binary.cpp', lines: 217 }],
          facts: ['A2', 'B1', 'ch02 D3'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
        },
      },
      pytorch: {
        cuda: {
          get entry() { return tr('- func: add.out: structured, ufunc_inner_loop, no CUDA: key', '- func: add.out: structured, ufunc_inner_loop, không có khóa CUDA:'); }, get host() { return tr('the scalar body in ufunc/add.h: return self + alpha * other;', 'phần thân vô hướng trong ufunc/add.h: return self + alpha * other;'); },
          file: null, get kernel() { return tr('UfuncCUDA_add.cu, written by torchgen at build time, calling gpu_kernel(iter, ufunctor)', 'UfuncCUDA_add.cu, do torchgen viết lúc build, gọi gpu_kernel(iter, ufunctor)'); }, readAt: 'pinned',
          also: [
            { get label() { return tr('native_functions.yaml lines 565 to 571', 'native_functions.yaml dòng 565 đến 571'); }, path: YAML, lines: [565, 571] },
            { label: 'ufunc/add.h', path: TN + 'ufunc/add.h' },
            { get label() { return tr('torchgen/gen.py line 2536', 'torchgen/gen.py dòng 2536'); }, path: 'torchgen/gen.py', lines: 2536 },
            { get label() { return tr('torchgen/dest/ufunc.py lines 279 to 328', 'torchgen/dest/ufunc.py dòng 279 đến 328'); }, path: 'torchgen/dest/ufunc.py', lines: [279, 328] },
          ],
          facts: ['A3'], tags: ['generated at build time'], // i18n-ignore: identifier also shown to the reader
          get note() { return tr('grep native/cuda for an add kernel and you find nothing: the file exists only in a build tree.', 'grep native/cuda để tìm một kernel add và bạn sẽ không thấy gì: tệp đó chỉ tồn tại trong một cây build.'); },
        },
        hip: {
          get entry() { return tr('- func: add.out: structured, ufunc_inner_loop, no CUDA: key', '- func: add.out: structured, ufunc_inner_loop, không có khóa CUDA:'); }, host: null, file: null,
          get kernel() { return tr('how the ROCm build gets its add kernel', 'cách bản build ROCm có được kernel add của nó'); }, readAt: 'pinned',
          facts: ['A3', 'B9', 'ch24 D3'], tags: ['generated at build time', 'not read in the facts pass'], // i18n-ignore: identifier also shown to the reader
          get note() { return tr('The CUDA kernel is generated by torchgen; whether hipify sees the generated file or torchgen writes a HIP one was not read.', 'Kernel CUDA do torchgen sinh ra; việc hipify thấy tệp được sinh ra hay torchgen viết một tệp HIP thì chưa được đọc.'); },
        },
        metal: {
          get entry() { return tr('add_out_mps, hand-written: “ufunc codegen only emits CPU/CUDA impls, so MPS needs this entry”', 'add_out_mps, viết tay: “ufunc codegen only emits CPU/CUDA impls, so MPS needs this entry” (codegen của ufunc chỉ sinh bản cài đặt CPU/CUDA, nên MPS cần mục này)'); },
          get host() { return tr('add_stub → lib.exec_binary_kernel(iter, "add") or "add_alpha"', 'add_stub → lib.exec_binary_kernel(iter, "add") hoặc "add_alpha"'); }, file: TN + 'mps/kernels/BinaryKernel.metal',
          kernel: 'add_functor, add_alpha_functor', lines: [22, 25], readAt: 'pinned', // i18n-ignore: kernel names
          also: [{ get label() { return tr('BinaryOps.mm lines 16 to 19', 'BinaryOps.mm dòng 16 đến 19'); }, path: TN + 'mps/operations/BinaryOps.mm', lines: [16, 19] }, { get label() { return tr('BinaryKernel.mm lines 182 to 187', 'BinaryKernel.mm dòng 182 đến 187'); }, path: TN + 'mps/operations/BinaryKernel.mm', lines: [182, 187] }],
          facts: ['A3'], tags: ['own kernel', 'found by name string'], // i18n-ignore: identifier also shown to the reader
        },
      },
    },
  },
  {
    id: 'silu', label: 'SiLU', tour: 'elementwise', section: 'X3.2',
    course: [lab('2.3', 'ch02', 'ch02/lab3_elementwise', 'the elementwise kernel SiLU is'), LAB_X31], // i18n-ignore: translated in lab() when read
    cells: {
      'llama.cpp': (() => {
        const cuda: Cell = {
          get entry() { return tr('case GGML_UNARY_OP_SILU in ggml_cuda_compute_forward', 'case GGML_UNARY_OP_SILU trong ggml_cuda_compute_forward'); }, host: 'ggml_cuda_op_silu = ggml_cuda_op_unary<op_silu>',
          file: CU + 'unary.cu', get kernel() { return tr('unary_op_kernel<op, T>, the op a template parameter', 'unary_op_kernel<op, T>, op là một tham số template'); }, lines: [116, 127], readAt: 'pinned',
          also: [CUDA_SWITCH(2882), { get label() { return tr('ggml_cuda_op_silu_single, unary.cuh lines 98 to 100', 'ggml_cuda_op_silu_single, unary.cuh dòng 98 đến 100'); }, path: CU + 'unary.cuh', lines: [98, 100] }],
          facts: ['A1', 'B1'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
          get note() { return tr('SwiGLU is case GGML_GLU_OP_SWIGLU (line 2945): unary_gated_op_kernel, one thread per output (lines 307 to 321).', 'SwiGLU là case GGML_GLU_OP_SWIGLU (dòng 2945): unary_gated_op_kernel, mỗi đầu ra một luồng (dòng 307 đến 321).'); },
        };
        return {
          cuda,
          get hip() { return llamaHip(cuda); },
          metal: {
            get entry() { return tr('case GGML_OP_UNARY in ggml_metal_op_encode_impl', 'case GGML_OP_UNARY trong ggml_metal_op_encode_impl'); }, host: 'ggml_metal_op_unary',
            file: MT + 'kernels/unary.metal', get kernel() { return tr('kernel_unary_impl, the op a function constant FC_unary_op', 'kernel_unary_impl, op là một function constant FC_unary_op'); }, lines: [132, 134], readAt: 'pinned',
            also: [METAL_SWITCH(319)], facts: ['A1', 'B1', 'C1'], tags: ['own kernel', 'found by name string'], // i18n-ignore: identifier also shown to the reader
            get note() { return tr('SwiGLU is case GGML_OP_GLU (line 327): kernel_swiglu, a grid-stride loop (lines 290 to 309).', 'SwiGLU là case GGML_OP_GLU (dòng 327): kernel_swiglu, một vòng lặp grid-stride (dòng 290 đến 309).'); },
          },
        };
      })(),
      mlx: {
        cuda: {
          get entry() { return tr('nn.silu is decorated with mx.compile', 'nn.silu được gắn decorator mx.compile'); }, get host() { return tr('Compiled::eval_gpu, cuda/compiled.cpp line 225', 'Compiled::eval_gpu, cuda/compiled.cpp dòng 225'); }, file: null,
          get kernel() { return tr('source written at run time (__global__ void {}( ...), built through cu::get_jit_module', 'mã nguồn được viết lúc chạy (__global__ void {}( ...), build qua cu::get_jit_module'); }, readAt: 'pinned',
          also: [{ get label() { return tr('activations.py lines 136 to 145', 'activations.py dòng 136 đến 145'); }, path: 'python/mlx/nn/layers/activations.py', lines: [136, 145] }, { get label() { return tr('cuda/compiled.cpp lines 225 to 245', 'cuda/compiled.cpp dòng 225 đến 245'); }, path: MLX_CU + 'compiled.cpp', lines: [225, 245] }],
          facts: ['A2'], tags: ['generated at run time (mx.compile)'], // i18n-ignore: identifier also shown to the reader
          get note() { return tr('There is no file to open: the kernel exists after the first call. MLX computes x * sigmoid(x).', 'Không có tệp nào để mở: kernel chỉ tồn tại sau lần gọi đầu tiên. MLX tính x * sigmoid(x).'); },
        },
        hip: MLX_NO_HIP,
        metal: {
          get entry() { return tr('nn.silu is decorated with mx.compile', 'nn.silu được gắn decorator mx.compile'); }, get host() { return tr('Compiled::eval_gpu, metal/compiled.cpp line 257', 'Compiled::eval_gpu, metal/compiled.cpp dòng 257'); }, file: null,
          get kernel() { return tr('source written at run time with [[host_name("{0}")]], compiled through d.get_library(...)', 'mã nguồn được viết lúc chạy với [[host_name("{0}")]], biên dịch qua d.get_library(...)'); }, readAt: 'pinned',
          also: [{ get label() { return tr('activations.py lines 136 to 145', 'activations.py dòng 136 đến 145'); }, path: 'python/mlx/nn/layers/activations.py', lines: [136, 145] }, { get label() { return tr('metal/compiled.cpp lines 257 to 290', 'metal/compiled.cpp dòng 257 đến 290'); }, path: MLX_MT + 'compiled.cpp', lines: [257, 290] }],
          facts: ['A2', 'C2'], tags: ['generated at run time (mx.compile)', 'found by name string'], // i18n-ignore: identifier also shown to the reader
          get note() { return tr('There is no file to open: the kernel exists after the first call.', 'Không có tệp nào để mở: kernel chỉ tồn tại sau lần gọi đầu tiên.'); },
        },
      },
      pytorch: (() => {
        const cuda: Cell = {
          entry: '- func: silu.out, dispatch CPU, CUDA, MPS, MTIA, XPU: silu_out', get host() { return tr('silu_stub, filled by REGISTER_DISPATCH(silu_stub, &silu_kernel)', 'silu_stub, được điền bởi REGISTER_DISPATCH(silu_stub, &silu_kernel)'); }, // i18n-ignore: a native_functions.yaml line, quoted as code
          file: TN + 'cuda/ActivationSiluKernel.cu', get kernel() { return tr('a lambda passed to gpu_kernel, computed in at::opmath_type', 'một lambda truyền cho gpu_kernel, tính trong at::opmath_type'); }, lines: [30, 57], readAt: 'pinned',
          also: [{ get label() { return tr('native_functions.yaml line 5300', 'native_functions.yaml dòng 5300'); }, path: YAML, lines: 5300 }], facts: ['A3', 'B1', 'ch03 D6'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
        };
        return {
          cuda,
          get hip() { return torchHip(cuda); },
          metal: {
            entry: '- func: silu.out, dispatch CPU, CUDA, MPS, MTIA, XPU: silu_out', host: null, file: null, get kernel() { return tr('the MPS implementation behind silu_stub', 'phần cài đặt MPS đằng sau silu_stub'); }, // i18n-ignore: a native_functions.yaml line, quoted as code
            readAt: 'pinned', also: [{ get label() { return tr('native_functions.yaml line 5300', 'native_functions.yaml dòng 5300'); }, path: YAML, lines: 5300 }], facts: ['A3'], tags: ['not read in the facts pass'], // i18n-ignore: identifier also shown to the reader
          },
        };
      })(),
    },
  },
  {
    id: 'rmsnorm', get label() { return tr('RMSNorm', 'RMSNorm'); }, tour: 'RMSNorm', section: 'X3.3',
    course: [lab('10.3', 'ch10', 'ch10/lab3_softmax_argmax', 'the two-stage row reduction (launchRowMax)'), LAB_X31], // i18n-ignore: translated in lab() when read
    cells: {
      'llama.cpp': (() => {
        const cuda: Cell = {
          get entry() { return tr('the fusion pass first (lines 5268 to 5271); else case GGML_OP_RMS_NORM', 'lượt gộp (fusion pass) chạy trước (dòng 5268 đến 5271); nếu không thì case GGML_OP_RMS_NORM'); },
          get host() { return tr('ggml_cuda_op_rms_norm_fused when RMS_NORM is followed by MUL; ggml_cuda_op_rms_norm otherwise', 'ggml_cuda_op_rms_norm_fused khi theo sau RMS_NORM là MUL; nếu không thì ggml_cuda_op_rms_norm'); },
          file: CU + 'norm.cu', get kernel() { return tr('rms_norm_f32, one block per row', 'rms_norm_f32, mỗi hàng một khối'); }, lines: [127, 272], readAt: 'pinned',
          also: [{ get label() { return tr('the fusion pass, lines 5255 to 5272', 'lượt gộp, dòng 5255 đến 5272'); }, path: CU + 'ggml-cuda.cu', lines: [5255, 5272] }, CUDA_SWITCH(2997), { get label() { return tr('GGML_CUDA_DISABLE_FUSION, line 181', 'GGML_CUDA_DISABLE_FUSION, dòng 181'); }, path: CU + 'ggml-cuda.cu', lines: 181 }],
          facts: ['A1', 'B2'], tags: ['own kernel', 'fused before dispatch'], // i18n-ignore: identifier also shown to the reader
          get note() { return tr('The host function you would expect is often not the one that runs: set GGML_CUDA_DISABLE_FUSION to check.', 'Hàm host bạn chờ đợi thường không phải hàm thực sự chạy: đặt GGML_CUDA_DISABLE_FUSION để kiểm tra.'); },
        };
        return {
          cuda,
          hip: { ...llamaHip(cuda), tags: ['same source as CUDA', 'fused before dispatch'] }, // i18n-ignore: identifier also shown to the reader
          metal: {
            get entry() { return tr('case GGML_OP_RMS_NORM in ggml_metal_op_encode_impl', 'case GGML_OP_RMS_NORM trong ggml_metal_op_encode_impl'); }, get host() { return tr('ggml_metal_op_norm, which returns how many nodes it consumed', 'ggml_metal_op_norm, hàm trả về số nút nó đã tiêu thụ'); },
            file: MT + 'kernels/norm.metal', get kernel() { return tr('kernel_rms_norm_fuse_impl<float, F>; F = 2 is "kernel_rms_norm_mul_f32" (line 179)', 'kernel_rms_norm_fuse_impl<float, F>; F = 2 là "kernel_rms_norm_mul_f32" (dòng 179)'); }, lines: [111, 185],
            readAt: 'pinned', also: [METAL_SWITCH(414)], facts: ['A1', 'B2', 'C2'], tags: ['own kernel', 'fused before dispatch', 'found by name string'], // i18n-ignore: identifier also shown to the reader
          },
        };
      })(),
      mlx: {
        cuda: {
          get entry() { return tr('nn.RMSNorm calls mx.fast.rms_norm', 'nn.RMSNorm gọi mx.fast.rms_norm'); }, get host() { return tr('RMSNorm::eval_gpu, rms_norm.cu line 539', 'RMSNorm::eval_gpu, rms_norm.cu dòng 539'); }, file: MLX_CU + 'rms_norm.cu',
          get kernel() { return tr('rms_norm_small, rms_norm, with a broadcasting block reduce', 'rms_norm_small, rms_norm, với một phép rút gọn trong khối có broadcast'); }, lines: 539, readAt: 'pinned',
          also: [{ get label() { return tr('normalization.py line 166', 'normalization.py dòng 166'); }, path: 'python/mlx/nn/layers/normalization.py', lines: 166 }], facts: ['A2', 'B2'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
        },
        hip: MLX_NO_HIP,
        metal: {
          get entry() { return tr('nn.RMSNorm calls mx.fast.rms_norm', 'nn.RMSNorm gọi mx.fast.rms_norm'); }, get host() { return tr('RMSNorm::eval_gpu, metal/normalization.cpp line 17', 'RMSNorm::eval_gpu, metal/normalization.cpp dòng 17'); }, file: MLX_MT + 'kernels/rms_norm.metal',
          get kernel() { return tr('rms_single_row (line 13), rms_looped (84) above RMS_LOOPED_LIMIT', 'rms_single_row (dòng 13), rms_looped (84) khi vượt RMS_LOOPED_LIMIT'); }, lines: [13, 84], readAt: 'pinned',
          also: [{ get label() { return tr('normalization.cpp lines 17 to 60', 'normalization.cpp dòng 17 đến 60'); }, path: MLX_MT + 'normalization.cpp', lines: [17, 60] }], facts: ['A2', 'B2'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
        },
      },
      pytorch: (() => {
        const cuda: Cell = {
          entry: '- func: _fused_rms_norm, dispatch CUDA: _fused_rms_norm_cuda', get host() { return tr('_fused_rms_norm_cuda, layer_norm_kernel.cu line 2009', '_fused_rms_norm_cuda, layer_norm_kernel.cu dòng 2009'); }, // i18n-ignore: a native_functions.yaml line, quoted as code
          file: TN + 'cuda/layer_norm_kernel.cu', get kernel() { return tr('LayerNorm’s kernels with a flag: RowwiseMomentsCUDAKernel<T, T_ACC, rms_norm> (line 57)', 'các kernel của LayerNorm kèm một cờ: RowwiseMomentsCUDAKernel<T, T_ACC, rms_norm> (dòng 57)'); }, lines: [57, 270],
          readAt: 'pinned',
          also: [{ get label() { return tr('native_functions.yaml lines 3304 to 3309', 'native_functions.yaml dòng 3304 đến 3309'); }, path: YAML, lines: [3304, 3309] }, { get label() { return tr('rms_norm_symint, layer_norm.cpp lines 329 to 372', 'rms_norm_symint, layer_norm.cpp dòng 329 đến 372'); }, path: TN + 'layer_norm.cpp', lines: [329, 372] }],
          facts: ['A3', 'B2'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
        };
        return {
          cuda,
          get hip() { return torchHip(cuda); },
          metal: {
            entry: '- func: _fused_rms_norm, dispatch MPS: _fused_rms_norm_mps', get host() { return tr('_fused_rms_norm_mps (RMSNorm.mm line 22), names the kernel with fmt::format', '_fused_rms_norm_mps (RMSNorm.mm dòng 22), đặt tên kernel bằng fmt::format'); }, // i18n-ignore: a native_functions.yaml line, quoted as code
            file: TN + 'mps/kernels/RMSNorm.metal', get kernel() { return tr('simd_sum, local_sums, metal::precise::rsqrt: MLX’s structure', 'simd_sum, local_sums, metal::precise::rsqrt: cấu trúc của MLX'); }, lines: [54, 74], readAt: 'pinned',
            also: [{ get label() { return tr('RMSNorm.mm lines 22 to 49', 'RMSNorm.mm dòng 22 đến 49'); }, path: TN + 'mps/operations/RMSNorm.mm', lines: [22, 49] }],
            facts: ['A3', 'B2', 'C2'], tags: ['own kernel', 'found by name string'], // i18n-ignore: identifier also shown to the reader
          },
        };
      })(),
    },
  },
  {
    id: 'softmax', get label() { return tr('softmax', 'softmax'); }, tour: 'softmax', section: 'X3.3',
    course: [lab('10.3', 'ch10', 'ch10/lab3_softmax_argmax', 'launchSoftmaxRows, max then sum')], // i18n-ignore: translated in lab() when read
    cells: {
      'llama.cpp': (() => {
        const cuda: Cell = {
          get entry() { return tr('case GGML_OP_SOFT_MAX in ggml_cuda_compute_forward', 'case GGML_OP_SOFT_MAX trong ggml_cuda_compute_forward'); }, host: 'ggml_cuda_op_soft_max', file: CU + 'softmax.cu', get kernel() { return tr('soft_max_f32: max, expf, barrier, sum', 'soft_max_f32: max, expf, rào chắn, sum'); },
          readAt: 'pinned', also: [CUDA_SWITCH(3042)], facts: ['A1', 'B3', 'ch10 D1'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
        };
        return {
          cuda,
          get hip() { return { ...llamaHip(cuda), get note() { return tr(LLAMA_HIP_NOTE + ' softmax.cu includes <hip/hip_cooperative_groups.h> under #ifdef GGML_USE_HIP (lines 5 to 10).', 'ggml-hip/CMakeLists.txt biên dịch ../ggml-cuda/*.cu với LANGUAGE HIP (dòng 60 đến 70). softmax.cu include <hip/hip_cooperative_groups.h> dưới #ifdef GGML_USE_HIP (dòng 5 đến 10).'); } }; },
          metal: {
            get entry() { return tr('case GGML_OP_SOFT_MAX in ggml_metal_op_encode_impl', 'case GGML_OP_SOFT_MAX trong ggml_metal_op_encode_impl'); }, host: 'ggml_metal_op_soft_max', file: MT + 'kernels/softmax.metal',
            get kernel() { return tr('kernel_soft_max: simd_max, two barriers, simd_sum', 'kernel_soft_max: simd_max, hai rào chắn, simd_sum'); }, lines: [42, 99], readAt: 'pinned', also: [METAL_SWITCH()],
            facts: ['A1', 'B3'], tags: ['own kernel', 'found by name string'], // i18n-ignore: identifier also shown to the reader
          },
        };
      })(),
      mlx: {
        cuda: {
          get entry() { return tr('the Softmax primitive', 'primitive Softmax'); }, get host() { return tr('Softmax::eval_gpu, cuda/softmax.cu line 108', 'Softmax::eval_gpu, cuda/softmax.cu dòng 108'); }, file: MLX_CU + 'softmax.cu',
          get kernel() { return tr('one block per row, an online normaliser in one pass, __expf', 'mỗi hàng một khối, một bộ chuẩn hóa trực tuyến (online normaliser) trong một lượt, __expf'); }, lines: [49, 62], readAt: 'pinned', facts: ['A2', 'B3'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
          get note() { return tr('Not the algorithm of MLX’s own Metal softmax.', 'Không phải thuật toán của softmax Metal do chính MLX viết.'); },
        },
        hip: MLX_NO_HIP,
        metal: {
          get entry() { return tr('the Softmax primitive', 'primitive Softmax'); }, host: 'Softmax::eval_gpu, metal/softmax.cpp', file: MLX_MT + 'kernels/softmax.h', // i18n-ignore: code
          get kernel() { return tr('softmax_single_row: max, then sum, fast::exp', 'softmax_single_row: max, rồi sum, fast::exp'); }, readAt: 'master', url: 'https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/softmax.h',
          facts: ['A2', 'B3', 'ch10 D3'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
        },
      },
      pytorch: (() => {
        const cuda: Cell = {
          entry: '- func: _softmax.out, dispatch CUDA: softmax_cuda_out', host: 'host_softmax', file: TN + 'cuda/SoftMax.cu', // i18n-ignore: a native_functions.yaml line, quoted as code
          get kernel() { return tr('a warp per row up to 2048 elements; cunn_SoftMaxForwardSmem or cunn_SoftMaxForward above', 'mỗi hàng một warp khi có tới 2048 phần tử; cunn_SoftMaxForwardSmem hoặc cunn_SoftMaxForward khi lớn hơn'); }, lines: [1125, 1129], readAt: 'pinned',
          also: [{ get label() { return tr('native_functions.yaml lines 5607 to 5609', 'native_functions.yaml dòng 5607 đến 5609'); }, path: YAML, lines: [5607, 5609] }], facts: ['A3', 'B3', 'ch10 D4'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
        };
        return {
          cuda,
          get hip() { return torchHip(cuda); },
          metal: {
            entry: '- func: _softmax.out, dispatch MPS: softmax_mps_out', host: 'softmax_mps_out', file: TN + 'mps/kernels/SoftMax.metal', // i18n-ignore: a native_functions.yaml line, quoted as code
            get kernel() { return tr('precise::exp throughout, simd_max and simd_sum', 'precise::exp ở mọi chỗ, simd_max và simd_sum'); }, lines: [30, 39], readAt: 'pinned',
            also: [{ get label() { return tr('native_functions.yaml lines 5607 to 5609', 'native_functions.yaml dòng 5607 đến 5609'); }, path: YAML, lines: [5607, 5609] }], facts: ['A3', 'B3'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
          },
        };
      })(),
    },
  },
  {
    id: 'matmul', get label() { return tr('matmul', 'matmul'); }, tour: 'GEMV and GEMM', section: 'X3.4', // i18n-ignore: identifier also shown to the reader
    course: [lab('5.3', 'ch05', 'ch05/lab3_gemm', 'the tiled GEMM'), lab('15.6', 'ch15', 'ch15/lab6_linear', 'M3, the linear layer'), lab('22.1', 'ch22', 'ch22/lab1_gemv', 'GEMV for one token')], // i18n-ignore: translated in lab() when read
    cells: {
      'llama.cpp': (() => {
        const cuda: Cell = {
          get entry() { return tr('case GGML_OP_MUL_MAT in ggml_cuda_compute_forward', 'case GGML_OP_MUL_MAT trong ggml_cuda_compute_forward'); }, get host() { return tr('ggml_cuda_mul_mat: a chain of “should use” tests', 'ggml_cuda_mul_mat: một chuỗi phép thử “should use”'); },
          file: CU + 'ggml-cuda.cu', get kernel() { return tr('mmvf, mmf, mmvq, mmq or cuBLAS, picked by shape and weight type', 'mmvf, mmf, mmvq, mmq hoặc cuBLAS, được chọn theo hình dạng và kiểu trọng số'); }, lines: [2535, 2596], readAt: 'pinned',
          also: [CUDA_SWITCH(3003)], facts: ['A1', 'B4', 'ch22 D6'], tags: ['own kernel', 'vendor library'], // i18n-ignore: identifier also shown to the reader
          get note() { return tr('Up to 8 tokens go to the vector kernels: MMVF_MAX_BATCH_SIZE and MMVQ_MAX_BATCH_SIZE are 8 (ch22 D6).', 'Tối đa 8 token đi vào các kernel vector: MMVF_MAX_BATCH_SIZE và MMVQ_MAX_BATCH_SIZE bằng 8 (ch22 D6).'); },
        };
        return {
          cuda,
          get hip() { return llamaHip(cuda); },
          metal: {
            get entry() { return tr('case GGML_OP_MUL_MAT in ggml_metal_op_encode_impl', 'case GGML_OP_MUL_MAT trong ggml_metal_op_encode_impl'); }, host: 'ggml_metal_op_mul_mat', file: MT + 'kernels/mul_mv.metal',
            get kernel() { return tr('mul_mv for one token, mul_mv_ext for 2 to 8, kernel_mul_mm (mul_mm.metal) above', 'mul_mv cho một token, mul_mv_ext cho 2 đến 8, kernel_mul_mm (mul_mm.metal) khi nhiều hơn'); }, readAt: 'pinned', also: [METAL_SWITCH()],
            facts: ['A1', 'B4', 'ch22 D6'], tags: ['own kernel', 'found by name string'], // i18n-ignore: identifier also shown to the reader
          },
        };
      })(),
      mlx: {
        cuda: {
          get entry() { return tr('the Matmul primitive', 'primitive Matmul'); }, get host() { return tr('Matmul::eval_gpu, cuda/matmul.cpp line 143', 'Matmul::eval_gpu, cuda/matmul.cpp dòng 143'); }, file: MLX_CU + 'gemms/gemv.cu',
          get kernel() { return tr('gemv_impl, a warp per row, when M or N is 1; cuBLASLt otherwise', 'gemv_impl, mỗi hàng một warp, khi M hoặc N bằng 1; nếu không thì cuBLASLt'); }, lines: [56, 83], readAt: 'pinned',
          also: [{ get label() { return tr('matmul.cpp lines 95 to 135', 'matmul.cpp dòng 95 đến 135'); }, path: MLX_CU + 'matmul.cpp', lines: [95, 135] }, { get label() { return tr('can_use_gemv, gemv.cu line 188', 'can_use_gemv, gemv.cu dòng 188'); }, path: MLX_CU + 'gemms/gemv.cu', lines: 188 }],
          facts: ['A2', 'B4'], tags: ['own kernel', 'vendor library'], // i18n-ignore: identifier also shown to the reader
        },
        hip: MLX_NO_HIP,
        metal: {
          get entry() { return tr('the Matmul primitive', 'primitive Matmul'); }, host: 'Matmul::eval_gpu, metal/matmul.cpp', file: MLX_MT + 'kernels/gemv.h', // i18n-ignore: code
          get kernel() { return tr('a wide GEMV for up to 5 vectors, GEMVKernel when min(M, N) is 1, steel GEMM otherwise', 'một GEMV rộng cho tối đa 5 vector, GEMVKernel khi min(M, N) bằng 1, nếu không thì steel GEMM'); }, lines: [30, 60], readAt: 'pinned',
          also: [{ get label() { return tr('matmul.cpp lines 1584 to 1660', 'matmul.cpp dòng 1584 đến 1660'); }, path: MLX_MT + 'matmul.cpp', lines: [1584, 1660] }], facts: ['A2', 'B4', 'ch15 D3'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
        },
      },
      pytorch: {
        cuda: {
          entry: '- func: mm.out, dispatch CUDA: mm_out_cuda', get host() { return tr('mm_out_cuda, then gemm_internal', 'mm_out_cuda, rồi gemm_internal'); }, file: 'aten/src/ATen/cuda/CUDABlas.cpp', // i18n-ignore: a native_functions.yaml line, quoted as code
          get kernel() { return tr('cuBLAS or cuBLASLt; gemv with TF32 disabled', 'cuBLAS hoặc cuBLASLt; gemv khi TF32 bị tắt'); }, readAt: 'master', url: 'https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDABlas.cpp',
          also: [{ get label() { return tr('native_functions.yaml line 4151', 'native_functions.yaml dòng 4151'); }, path: YAML, lines: 4151 }], facts: ['A3', 'B4', 'ch15 D4'], tags: ['vendor library'], // i18n-ignore: identifier also shown to the reader
        },
        hip: {
          get entry() { return tr('- func: mm.out, dispatch CUDA: mm_out_cuda (the same key on ROCm)', '- func: mm.out, dispatch CUDA: mm_out_cuda (cùng khóa đó trên ROCm)'); }, host: 'gemm_internal', file: 'aten/src/ATen/cuda/CUDABlas.cpp',
          get kernel() { return tr('hipBLAS or Composable Kernel', 'hipBLAS hoặc Composable Kernel'); }, readAt: 'master', url: 'https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDABlas.cpp',
          facts: ['ch15 D4', 'B9'], tags: ['vendor library', 'hipified at build time'], // i18n-ignore: identifier also shown to the reader
        },
        metal: {
          entry: '- func: mm.out, dispatch MPS: mm_out_mps', get host() { return tr('Matmul.mm builds names gemv_t_..., gemv_nt_... (lines 116 to 141)', 'Matmul.mm dựng các tên gemv_t_..., gemv_nt_... (dòng 116 đến 141)'); }, // i18n-ignore: a native_functions.yaml line, quoted as code
          file: TN + 'mps/kernels/Gemv.h', get kernel() { return tr('gemv_t (line 71) for rank-1 shapes; MPSNDArrayMatrixMultiplication otherwise', 'gemv_t (dòng 71) cho hình dạng hạng 1; nếu không thì MPSNDArrayMatrixMultiplication'); }, lines: 71, readAt: 'pinned',
          also: [{ get label() { return tr('native_functions.yaml line 4153', 'native_functions.yaml dòng 4153'); }, path: YAML, lines: 4153 }, { get label() { return tr('Matmul.mm lines 94 to 160', 'Matmul.mm dòng 94 đến 160'); }, path: TN + 'mps/operations/Matmul.mm', lines: [94, 160] }],
          facts: ['B4'], tags: ['own kernel', 'vendor library', 'found by name string'], // i18n-ignore: identifier also shown to the reader
        },
      },
    },
  },
  {
    id: 'rope', label: 'RoPE', tour: 'RoPE and SwiGLU', section: 'X3.7', // i18n-ignore: identifier also shown to the reader
    course: [LAB_X31],
    cells: {
      'llama.cpp': (() => {
        const cuda: Cell = {
          get entry() { return tr('case GGML_OP_ROPE in ggml_cuda_compute_forward', 'case GGML_OP_ROPE trong ggml_cuda_compute_forward'); }, host: 'ggml_cuda_op_rope', file: CU + 'rope.cu',
          get kernel() { return tr('rope_norm (line 44), consecutive pairs; rope_neox (126), pairs half a head apart', 'rope_norm (dòng 44), các cặp liền kề; rope_neox (126), các cặp cách nhau nửa head'); }, lines: [44, 126], readAt: 'pinned',
          also: [CUDA_SWITCH(3048)], facts: ['A1', 'B6'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
        };
        return {
          cuda,
          get hip() { return llamaHip(cuda); },
          metal: {
            get entry() { return tr('case GGML_OP_ROPE in ggml_metal_op_encode_impl', 'case GGML_OP_ROPE trong ggml_metal_op_encode_impl'); }, host: 'ggml_metal_op_rope', file: MT + 'kernels/rope.metal',
            get kernel() { return tr('kernel_rope_norm (line 48), kernel_rope_neox (106)', 'kernel_rope_norm (dòng 48), kernel_rope_neox (106)'); }, lines: [48, 106], readAt: 'pinned', also: [METAL_SWITCH()],
            facts: ['A1', 'B6'], tags: ['own kernel', 'found by name string'], // i18n-ignore: identifier also shown to the reader
          },
        };
      })(),
      mlx: {
        cuda: {
          entry: 'mx.fast.rope(a, dims, *, traditional, base, scale, offset)', get host() { return tr('RoPE::eval_gpu, cuda/rope.cu line 249', 'RoPE::eval_gpu, cuda/rope.cu dòng 249'); }, file: MLX_CU + 'rope.cu',
          get kernel() { return tr('the pairing a template parameter: template <typename T, bool traditional, bool forward>', 'cách ghép cặp là một tham số template: template <typename T, bool traditional, bool forward>'); }, lines: 15, readAt: 'pinned',
          also: [{ get label() { return tr('fast.cpp lines 211 to 245', 'fast.cpp dòng 211 đến 245'); }, path: 'python/src/fast.cpp', lines: [211, 245] }], facts: ['A2', 'B6', 'C1'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
        },
        hip: MLX_NO_HIP,
        metal: {
          entry: 'mx.fast.rope(a, dims, *, traditional, base, scale, offset)', host: 'RoPE::eval_gpu, metal/rope.cpp', file: MLX_MT + 'kernels/rope.metal', // i18n-ignore: code
          get kernel() { return tr('the pairing a function constant: constant bool traditional [[function_constant(2)]]', 'cách ghép cặp là một function constant: constant bool traditional [[function_constant(2)]]'); }, lines: 8, readAt: 'pinned',
          also: [{ get label() { return tr('fast.cpp lines 211 to 245', 'fast.cpp dòng 211 đến 245'); }, path: 'python/src/fast.cpp', lines: [211, 245] }], facts: ['A2', 'B6', 'C1'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
        },
      },
      pytorch: (() => {
        const none: Cell = {
          get entry() { return tr('none: “rotary” does not occur in native_functions.yaml', 'không có: “rotary” không xuất hiện trong native_functions.yaml'); }, get host() { return tr('Hugging Face’s rotate_half, run as ordinary tensor ops', 'rotate_half của Hugging Face, chạy như các op tensor thông thường'); }, file: null, kernel: null,
          readAt: 'pinned', also: [{ get label() { return tr('transformers modeling_llama.py lines 73 to 123', 'transformers modeling_llama.py dòng 73 đến 123'); }, path: 'src/transformers/models/llama/modeling_llama.py', lines: [73, 123], project: 'transformers' }],
          facts: ['B6'], tags: ['composite of other ops'], // i18n-ignore: identifier also shown to the reader
        };
        return { cuda: none, get hip() { return { ...none, facts: ['B6', 'B9'] }; }, metal: none };
      })(),
    },
  },
  {
    id: 'attention', get label() { return tr('attention', 'attention'); }, tour: 'attention', section: 'X3.5',
    course: [lab('20.2', 'ch20', 'ch20/lab2_flash', 'flash attention'), lab('20.4', 'ch20', 'ch20/lab4_decode', 'decode with a KV cache'), lab('20.5', 'ch20', 'ch20/lab5_gqa_decode', 'GQA decode')], // i18n-ignore: translated in lab() when read
    cells: {
      'llama.cpp': (() => {
        const cuda: Cell = {
          get entry() { return tr('case GGML_OP_FLASH_ATTN_EXT in ggml_cuda_compute_forward', 'case GGML_OP_FLASH_ATTN_EXT trong ggml_cuda_compute_forward'); }, host: 'ggml_cuda_flash_attn_ext', file: CU + 'fattn.cu',
          get kernel() { return tr('its own: the vec, tile and mma kernels (fattn-vec.cuh, fattn-tile, fattn-mma-f16.cuh)', 'kernel của chính nó: các kernel vec, tile và mma (fattn-vec.cuh, fattn-tile, fattn-mma-f16.cuh)'); }, readAt: 'pinned', also: [CUDA_SWITCH(3111)],
          facts: ['A1', 'ch20 D1', 'infra D1'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
        };
        return {
          cuda,
          get hip() { return llamaHip(cuda); },
          metal: {
            get entry() { return tr('case GGML_OP_FLASH_ATTN_EXT in ggml_metal_op_encode_impl', 'case GGML_OP_FLASH_ATTN_EXT trong ggml_metal_op_encode_impl'); }, host: 'ggml_metal_op_flash_attn_ext', file: MT + 'kernels/fa_common.metal',
            get kernel() { return tr('its own: one file per K/V type (fa_f32.metal, fa_f16.metal, ...), vec variants, fa_aux.metal', 'kernel của chính nó: mỗi kiểu K/V một tệp (fa_f32.metal, fa_f16.metal, ...), các biến thể vec, fa_aux.metal'); }, readAt: 'pinned', also: [METAL_SWITCH(487)],
            facts: ['A1', 'ch20 D2', 'infra D2'], tags: ['own kernel', 'found by name string'], // i18n-ignore: identifier also shown to the reader
          },
        };
      })(),
      mlx: {
        cuda: {
          entry: 'mx.fast.scaled_dot_product_attention', get host() { return tr('ScaledDotProductAttention::eval_gpu, line 623: sdpa_cudnn if supported, else sdpa_vector', 'ScaledDotProductAttention::eval_gpu, dòng 623: sdpa_cudnn nếu được hỗ trợ, nếu không thì sdpa_vector'); },
          file: MLX_CU + 'scaled_dot_product_attention.cu', get kernel() { return tr('its own for short decodes: kernel_sdpav_1pass (line 42), kernel_sdpav_2pass_1 (195); cuDNN SDPA for prefill and long decodes', 'kernel của chính nó cho decode ngắn: kernel_sdpav_1pass (dòng 42), kernel_sdpav_2pass_1 (195); cuDNN SDPA cho prefill và decode dài'); },
          lines: [42, 195], readAt: 'pinned', also: [{ get label() { return tr('scaled_dot_product_attention.cpp lines 623 to 660', 'scaled_dot_product_attention.cpp dòng 623 đến 660'); }, path: MLX_CU + 'scaled_dot_product_attention.cpp', lines: [623, 660] }],
          facts: ['B5'], tags: ['own kernel', 'vendor library'], // i18n-ignore: identifier also shown to the reader
        },
        hip: MLX_NO_HIP,
        metal: {
          entry: 'mx.fast.scaled_dot_product_attention', host: 'ScaledDotProductAttention::eval_gpu, metal/scaled_dot_product_attention.cpp', file: MLX_MT + 'kernels/sdpa_vector.h', // i18n-ignore: code
          get kernel() { return tr('its own: vector kernels for 8 or fewer queries, steel attention otherwise', 'kernel của chính nó: kernel vector cho 8 query trở xuống, nếu không thì steel attention'); }, readAt: 'master',
          url: 'https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sdpa_vector.h', facts: ['ch20 D4'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
        },
      },
      pytorch: {
        cuda: {
          entry: '- func: _scaled_dot_product_flash_attention, dispatch CUDA: _scaled_dot_product_flash_attention_cuda', get host() { return tr('select_sdp_backend (sdp_utils.cpp line 1337): flash, efficient, math, cuDNN', 'select_sdp_backend (sdp_utils.cpp dòng 1337): flash, efficient, math, cuDNN'); }, // i18n-ignore: a native_functions.yaml line, quoted as code
          file: TN + 'transformers/cuda/flash_attn/flash_api.cpp', get kernel() { return tr('run_mha_fwd_ and run_mha_fwd_splitkv_dispatch, from the third_party/flash-attention submodule', 'run_mha_fwd_ và run_mha_fwd_splitkv_dispatch, từ submodule third_party/flash-attention'); }, lines: [294, 301],
          readAt: 'pinned',
          also: [{ get label() { return tr('native_functions.yaml lines 15280 to 15282', 'native_functions.yaml dòng 15280 đến 15282'); }, path: YAML, lines: [15280, 15282] }, { get label() { return tr('sdp_utils.cpp lines 1337 to 1350', 'sdp_utils.cpp dòng 1337 đến 1350'); }, path: TN + 'transformers/cuda/sdp_utils.cpp', lines: [1337, 1350] }, { get label() { return tr('.gitmodules lines 115 to 117', '.gitmodules dòng 115 đến 117'); }, path: '.gitmodules', lines: [115, 117] }],
          facts: ['B5'], tags: ['third-party submodule', 'vendor library'], // i18n-ignore: identifier also shown to the reader
        },
        hip: {
          get entry() { return tr('the same dispatch, hipified', 'cùng dispatch đó, đã qua hipify'); }, host: 'select_sdp_backend', file: TN + 'transformers/hip',
          get kernel() { return tr('AOTriton or Composable Kernel (transformers/hip: aotriton_adapter.h, flash_attn/{aot,ck})', 'AOTriton hoặc Composable Kernel (transformers/hip: aotriton_adapter.h, flash_attn/{aot,ck})'); }, readAt: 'master',
          url: 'https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/transformers/hip', facts: ['B9', 'ch20 B5'], tags: ['vendor library', 'hipified at build time'], // i18n-ignore: identifier also shown to the reader
        },
        metal: {
          get entry() { return tr('scaled_dot_product_attention on MPS: Attention.mm', 'scaled_dot_product_attention trên MPS: Attention.mm'); }, get host() { return tr('sdpa_vector_2pass_mps, sdpa_vector_fast_mps or sdpa_prefill_mps', 'sdpa_vector_2pass_mps, sdpa_vector_fast_mps hoặc sdpa_prefill_mps'); }, file: TN + 'mps/kernels/Attention.metal',
          get kernel() { return tr('its own Metal kernels, “largely influenced” by MLX’s', 'các kernel Metal của chính nó, “largely influenced” (chịu ảnh hưởng lớn) bởi kernel của MLX'); }, readAt: 'master', url: 'https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/kernels/Attention.metal',
          facts: ['B5', 'ch20 D5'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
        },
      },
    },
  },
  {
    id: 'sampling', get label() { return tr('argmax and sampling', 'argmax và lấy mẫu'); }, tour: 'sampling', section: 'X3.6',
    course: [lab('10.3', 'ch10', 'ch10/lab3_softmax_argmax', 'launchArgmaxRows'), lab('X1.3', 'x1', 'x1/lab3_sampler', 'M5, the GPU sampler')], // i18n-ignore: translated in lab() when read
    cells: {
      'llama.cpp': (() => {
        const cuda: Cell = {
          get entry() { return tr('the host chain by default (src/llama-sampler.cpp); the experimental backend greedy sampler builds ggml_argmax', 'mặc định là chuỗi lấy mẫu trên host (src/llama-sampler.cpp); bộ lấy mẫu greedy thử nghiệm trên backend dựng ggml_argmax'); }, get host() { return tr('llama_sampler_dist_apply on the host', 'llama_sampler_dist_apply trên host'); },
          file: CU + 'argmax.cu', get kernel() { return tr('argmax_f32: ties go to whichever lane held the index', 'argmax_f32: khi hòa, kết quả là chỉ số do bất kỳ lane nào đang giữ'); }, readAt: 'master', url: 'https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/argmax.cu',
          facts: ['B8', 'ch10 D2', 'x1 D3'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
        };
        return {
          cuda,
          hip: { ...cuda, facts: [...cuda.facts, 'B9'], tags: ['same source as CUDA'], get note() { return tr(LLAMA_HIP_NOTE, 'ggml-hip/CMakeLists.txt biên dịch ../ggml-cuda/*.cu với LANGUAGE HIP (dòng 60 đến 70).'); } }, // i18n-ignore: identifier also shown to the reader
          metal: {
            get entry() { return tr('the experimental backend greedy sampler builds ggml_argmax', 'bộ lấy mẫu greedy thử nghiệm trên backend dựng ggml_argmax'); }, host: null, file: MT + 'kernels/misc.metal',
            get kernel() { return tr('kernel_argmax_f32: the largest index among lanes holding the maximum', 'kernel_argmax_f32: chỉ số lớn nhất trong các lane đang giữ giá trị lớn nhất'); }, readAt: 'master', url: 'https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/misc.metal',
            facts: ['ch10 D2', 'x1 D3'], tags: ['own kernel'], // i18n-ignore: identifier also shown to the reader
          },
        };
      })(),
      mlx: {
        cuda: {
          get entry() { return tr('mlx-lm sample_utils.py: mx.random.categorical or mx.argmax', 'mlx-lm sample_utils.py: mx.random.categorical hoặc mx.argmax'); }, host: null, file: MLX_CU + 'random.cu',
          get kernel() { return tr('threefry2x32_hash (line 27), rbitsc (47), rbits (87); the rest is ordinary MLX ops', 'threefry2x32_hash (dòng 27), rbitsc (47), rbits (87); phần còn lại là các op MLX thông thường'); }, lines: [27, 87], readAt: 'pinned',
          facts: ['B8', 'x1 D7', 'x1 D8'], tags: ['own kernel', 'composite of other ops'], // i18n-ignore: identifier also shown to the reader
        },
        hip: MLX_NO_HIP,
        metal: {
          get entry() { return tr('mlx-lm sample_utils.py: mx.random.categorical or mx.argmax', 'mlx-lm sample_utils.py: mx.random.categorical hoặc mx.argmax'); }, host: null, file: MLX_MT + 'kernels/random.metal',
          get kernel() { return tr('threefry2x32_hash, 20 rounds; the rest is ordinary MLX ops', 'threefry2x32_hash, 20 vòng; phần còn lại là các op MLX thông thường'); }, readAt: 'master', url: 'https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/random.metal',
          facts: ['B8', 'x1 D7', 'x1 D8'], tags: ['own kernel', 'composite of other ops'], // i18n-ignore: identifier also shown to the reader
        },
      },
      pytorch: (() => {
        const cuda: Cell = {
          entry: '- func: multinomial, dispatch CPU, CUDA, MPS, XPU: multinomial', get host() { return tr('multinomial_out: one sample is argmax(p / q) with q exponential', 'multinomial_out: một mẫu là argmax(p / q) với q có phân phối mũ'); }, // i18n-ignore: a native_functions.yaml line, quoted as code
          file: TN + 'cuda/MultinomialKernel.cu', get kernel() { return tr('the inverse-CDF search, used only with replacement and several samples', 'phép tìm kiếm theo CDF nghịch đảo, chỉ dùng khi lấy có hoàn lại và nhiều mẫu'); }, readAt: 'master',
          url: 'https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/MultinomialKernel.cu', facts: ['x1 D5', 'B8'], tags: ['composite of other ops', 'own kernel'], // i18n-ignore: identifier also shown to the reader
        };
        return {
          cuda,
          get hip() { return torchHip(cuda); },
          metal: {
            ...cuda, file: TN + 'mps/operations/Distributions.mm', get kernel() { return tr('with replacement: cumsum, then searchsorted', 'khi có hoàn lại: cumsum, rồi searchsorted'); }, url: 'https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Distributions.mm',
            tags: ['composite of other ops'], // i18n-ignore: identifier also shown to the reader
          },
        };
      })(),
    },
  },
];

export const OP_IDS = OPS.map((o) => o.id);
export function op(id: OpId): OpRow {
  const o = OPS.find((x) => x.id === id);
  if (!o) throw new Error('no op ' + id);
  return o;
}
export const cell = (o: OpId, p: Project, b: Backend) => op(o).cells[p][b];

/** The facts file an item lives in: "B1" → x3-facts.md, "ch10 D2" → ch10-facts.md. */
export function factsFile(ref: string): { file: string; item: string } {
  const m = /^(\w+) ([A-Z]\d+)$/.exec(ref);
  return m ? { file: `notes/today/${m[1]}-facts.md`, item: m[2] } : { file: 'notes/today/x3-facts.md', item: ref };
}

/** The link for a cell's kernel file: a pinned permalink, the master URL of its facts item, or none. */
export function cellLink(project: Project, c: Cell): string | null {
  if (c.readAt === 'master') return c.url ?? null;
  if (!c.file) return null;
  return permalink(project, c.file, c.lines ?? null, !/\.\w+$/.test(c.file));
}
export const whereLink = (project: Project, w: Where) => permalink(w.project ?? project, w.path, w.lines ?? null, !!w.tree);

// ---------------------------------------------------------------- the three entry points
export interface EntryPoint {
  project: Project;
  what: string;
  cuda: string;
  hip: string;
  metal: string;
  rule: string;
  facts: string[];
  where: Where[];
}
export const ENTRY_POINTS: EntryPoint[] = [
  {
    project: 'llama.cpp', get what() { return tr('an op is a GGML_OP_* value in a graph', 'một op là một giá trị GGML_OP_* trong một đồ thị'); },
    get cuda() { return tr('ggml_cuda_compute_forward: one switch on dst->op (after the fusion pass)', 'ggml_cuda_compute_forward: một switch trên dst->op (sau lượt gộp)'); }, get hip() { return tr('the same switch: ggml-cuda/*.cu compiled as HIP', 'cùng switch đó: ggml-cuda/*.cu được biên dịch thành HIP'); },
    get metal() { return tr('ggml_metal_op_encode_impl: its own switch; each case returns n_fuse and asks for a pipeline by name', 'ggml_metal_op_encode_impl: switch riêng của nó; mỗi case trả về n_fuse và yêu cầu một pipeline theo tên'); },
    get rule() { return tr('Find the case for the op, open the .cu named after it. Check the fusion pass first. On Metal, follow the formatted name string.', 'Tìm case của op, mở tệp .cu mang tên nó. Kiểm tra lượt gộp trước. Trên Metal, lần theo chuỗi tên được định dạng.'); },
    facts: ['A1', 'B9', 'C2'],
    where: [{ get label() { return tr('ggml-cuda.cu line 2812', 'ggml-cuda.cu dòng 2812'); }, path: CU + 'ggml-cuda.cu', lines: 2812 }, { get label() { return tr('ggml-metal-ops.cpp line 187', 'ggml-metal-ops.cpp dòng 187'); }, path: MT + 'ggml-metal-ops.cpp', lines: 187 }],
  },
  {
    project: 'mlx', get what() { return tr('an op is a Primitive subclass', 'một op là một lớp con của Primitive'); },
    get cuda() { return tr('Primitive::eval_gpu in mlx/backend/cuda', 'Primitive::eval_gpu trong mlx/backend/cuda'); }, get hip() { return tr('none: MLX has no AMD backend', 'không có: MLX không có backend AMD'); },
    get metal() { return tr('Primitive::eval_gpu in mlx/backend/metal; anything under mx.compile is source written at run time', 'Primitive::eval_gpu trong mlx/backend/metal; mọi thứ dưới mx.compile là mã nguồn được viết lúc chạy'); },
    get rule() { return tr('Search for Name::eval_gpu in the backend’s directory; elementwise ops come from a macro; compiled functions have no file.', 'Tìm Name::eval_gpu trong thư mục của backend; các op theo từng phần tử đến từ một macro; các hàm đã biên dịch (compiled) không có tệp.'); },
    facts: ['A2', 'B9'],
    where: [{ label: 'mlx/backend', path: 'mlx/backend', tree: true }],
  },
  {
    project: 'pytorch', get what() { return tr('an op is a - func: entry in native_functions.yaml', 'một op là một mục - func: trong native_functions.yaml'); },
    get cuda() { return tr('the CUDA: dispatch key; or structured_delegate, ufunc_inner_loop, or a _stub filled by REGISTER_DISPATCH', 'khóa dispatch CUDA:; hoặc structured_delegate, ufunc_inner_loop, hoặc một _stub được REGISTER_DISPATCH điền'); }, get hip() { return tr('the same CUDA sources, hipified at build time', 'cùng các mã nguồn CUDA đó, được hipify lúc build'); },
    get metal() { return tr('the MPS: dispatch key, an operations/*.mm file that names a kernel in kernels/*.metal by a string', 'khóa dispatch MPS:, một tệp operations/*.mm gọi tên một kernel trong kernels/*.metal bằng một chuỗi'); },
    get rule() { return tr('Find the - func: line, read dispatch:; with no backend key, follow structured_delegate, ufunc_inner_loop or the stub to REGISTER_DISPATCH.', 'Tìm dòng - func:, đọc dispatch:; nếu không có khóa backend, lần theo structured_delegate, ufunc_inner_loop hoặc stub tới REGISTER_DISPATCH.'); },
    facts: ['A3', 'B9', 'C2', 'ch24 D3'], // i18n-ignore: identifier also shown to the reader
    where: [{ label: 'native_functions.yaml', path: YAML }],
  },
];

/** Three rows of x3-facts "Where the backends really differ", verbatim. */
export const DIFFER_ROWS: { feature: string; cuda: string; hip: string; metal: string }[] = [
  {
    get feature() { return tr('Op to kernel map', 'Ánh xạ từ op sang kernel'); },
    get cuda() { return tr('llama.cpp `ggml_cuda_compute_forward`; MLX `eval_gpu` in `backend/cuda`; PyTorch `CUDA:` dispatch key', 'llama.cpp `ggml_cuda_compute_forward`; MLX `eval_gpu` trong `backend/cuda`; PyTorch khóa dispatch `CUDA:`'); },
    get hip() { return tr('Same as CUDA', 'Giống CUDA'); },
    get metal() { return tr('llama.cpp `ggml_metal_op_encode_impl`; MLX `eval_gpu` in `backend/metal`; PyTorch `MPS:` key', 'llama.cpp `ggml_metal_op_encode_impl`; MLX `eval_gpu` trong `backend/metal`; PyTorch khóa `MPS:`'); },
  },
  {
    get feature() { return tr('Kernel variants', 'Biến thể kernel'); }, get cuda() { return tr('Template parameters', 'Tham số template'); }, get hip() { return tr('Same templates', 'Cùng các template'); },
    get metal() { return tr('Function constants (`[[function_constant]]`, Metal 1.2+) and `[[host_name]]` instantiations', 'Function constant (`[[function_constant]]`, Metal 1.2+) và các bản khởi tạo `[[host_name]]`'); },
  },
  { get feature() { return tr('Finding a kernel from the host', 'Tìm một kernel từ host'); }, get cuda() { return tr('Function symbol', 'Ký hiệu hàm'); }, get hip() { return tr('Same', 'Giống'); }, get metal() { return tr('The name string passed to the pipeline lookup', 'Chuỗi tên được truyền cho bước tra cứu pipeline'); } },
];

// ---------------------------------------------------------------- "Follow the call" routes
export type SearchKind = 'function name' | 'yaml key' | 'switch case' | 'formatted string' | 'REGISTER_DISPATCH' | 'macro' | 'file name' | 'environment variable';
export interface HopOption {
  label: string;
  kind?: SearchKind;
  correct?: boolean;
  why?: string;
}
export interface Hop {
  prompt: string;
  options: HopOption[];
  /** What you find when you follow the right option. */
  found: string;
  project: ProjectId;
  path: string;
  lines?: Lines;
  facts: string[];
}
export interface Route {
  id: string;
  label: string;
  project: Project;
  backend: Backend;
  op: OpId;
  start: string;
  hops: Hop[];
  /** What this route teaches, shown at the end. */
  lesson: string;
}

export const ROUTES: Route[] = [
  {
    id: 'torch-rmsnorm-cuda', get label() { return tr('PyTorch rms_norm on CUDA', 'PyTorch rms_norm trên CUDA'); }, project: 'pytorch', backend: 'cuda', op: 'rmsnorm',
    get start() { return tr('torch.nn.functional.rms_norm(x, (d,), weight) on CUDA tensors of the same dtype.', 'torch.nn.functional.rms_norm(x, (d,), weight) trên các tensor CUDA có cùng dtype.'); },
    hops: [
      {
        get prompt() { return tr('The Python function calls into C++. What do you look for in aten/src/ATen/native?', 'Hàm Python gọi vào C++. Bạn tìm gì trong aten/src/ATen/native?'); },
        options: [
          { get label() { return tr('The C++ function it binds to, rms_norm_symint', 'Hàm C++ mà nó gắn tới, rms_norm_symint'); }, kind: 'function name', correct: true },
          { get label() { return tr('A switch on an op enum', 'Một switch trên một enum op'); }, kind: 'switch case', get why() { return tr('That is llama.cpp’s shape. PyTorch has no op switch to search.', 'Đó là kiểu của llama.cpp. PyTorch không có switch op nào để tìm.'); } },
          { get label() { return tr('A kernel-name string', 'Một chuỗi tên kernel'); }, kind: 'formatted string', get why() { return tr('Name strings are how Metal pipelines are found, not how a Python call reaches C++.', 'Chuỗi tên là cách tìm pipeline Metal, không phải cách một lời gọi Python đến được C++.'); } },
        ],
        get found() { return tr('rms_norm_symint in layer_norm.cpp calls at::_fused_rms_norm(...) (line 372). It falls back to rms_norm_composite for complex input, and warns “Cannot dispatch to fused implementation.” when weight and input dtypes differ.', 'rms_norm_symint trong layer_norm.cpp gọi at::_fused_rms_norm(...) (dòng 372). Nó quay về rms_norm_composite với đầu vào số phức, và cảnh báo “Cannot dispatch to fused implementation.” khi dtype của weight và của đầu vào khác nhau.'); },
        project: 'pytorch', path: TN + 'layer_norm.cpp', lines: [329, 372], facts: ['B2'],
      },
      {
        get prompt() { return tr('at::_fused_rms_norm is an operator. Where is the map from it to a CUDA function?', 'at::_fused_rms_norm là một operator. Ánh xạ từ nó sang một hàm CUDA nằm ở đâu?'); },
        options: [
          { get label() { return tr('Its - func: _fused_rms_norm entry in native_functions.yaml', 'Mục - func: _fused_rms_norm của nó trong native_functions.yaml'); }, kind: 'yaml key', correct: true },
          { get label() { return tr('A REGISTER_DISPATCH for a _fused_rms_norm_stub', 'Một REGISTER_DISPATCH cho một _fused_rms_norm_stub'); }, kind: 'REGISTER_DISPATCH', get why() { return tr('Some ops do go through a stub, but this one names its backends directly in the yaml.', 'Một số op đúng là đi qua một stub, nhưng op này gọi tên các backend của nó trực tiếp trong yaml.'); } },
          { get label() { return tr('A file called _fused_rms_norm.cu', 'Một tệp tên _fused_rms_norm.cu'); }, kind: 'file name', get why() { return tr('PyTorch does not name files after operators.', 'PyTorch không đặt tên tệp theo operator.'); } },
        ],
        get found() { return tr('dispatch: CUDA: _fused_rms_norm_cuda, MPS: _fused_rms_norm_mps, CompositeImplicitAutograd: rms_norm_composite (lines 3304 to 3309).', 'dispatch: CUDA: _fused_rms_norm_cuda, MPS: _fused_rms_norm_mps, CompositeImplicitAutograd: rms_norm_composite (dòng 3304 đến 3309).'); },
        project: 'pytorch', path: YAML, lines: [3304, 3309], facts: ['A3'],
      },
      {
        get prompt() { return tr('The CUDA key names _fused_rms_norm_cuda. What do you search for now?', 'Khóa CUDA gọi tên _fused_rms_norm_cuda. Giờ bạn tìm gì?'); },
        options: [
          { get label() { return tr('That symbol, _fused_rms_norm_cuda, in native/cuda', 'Chính ký hiệu đó, _fused_rms_norm_cuda, trong native/cuda'); }, kind: 'function name', correct: true },
          { get label() { return tr('A generated UfuncCUDA_ file in the build tree', 'Một tệp UfuncCUDA_ được sinh ra trong cây build'); }, kind: 'file name', get why() { return tr('Only ufunc ops (add and friends) are generated. This one has a CUDA key and a hand-written function.', 'Chỉ các op ufunc (add và các op cùng loại) được sinh ra. Op này có khóa CUDA và một hàm viết tay.'); } },
          { get label() { return tr('The MPS operations file', 'Tệp operations của MPS'); }, kind: 'file name', get why() { return tr('That is the MPS: key, another backend.', 'Đó là khóa MPS:, một backend khác.'); } },
        ],
        get found() { return tr('_fused_rms_norm_cuda is in native/cuda/layer_norm_kernel.cu (line 2009).', '_fused_rms_norm_cuda nằm trong native/cuda/layer_norm_kernel.cu (dòng 2009).'); },
        project: 'pytorch', path: TN + 'cuda/layer_norm_kernel.cu', lines: 2009, facts: ['B2'],
      },
      {
        get prompt() { return tr('Predict: what does the kernel it launches look like?', 'Dự đoán: kernel mà nó khởi chạy trông như thế nào?'); },
        options: [
          { get label() { return tr('LayerNorm’s kernels, with RMSNorm selected by a template flag', 'Các kernel của LayerNorm, với RMSNorm được chọn bằng một cờ template'); }, correct: true },
          { get label() { return tr('A separate RMSNorm kernel file', 'Một tệp kernel RMSNorm riêng'); }, get why() { return tr('There is none: the two norms share kernels.', 'Không có: hai phép chuẩn hóa dùng chung kernel.'); } },
          { get label() { return tr('A cuDNN normalisation call', 'Một lời gọi chuẩn hóa của cuDNN'); }, get why() { return tr('No library is involved here; these are PyTorch’s own kernels.', 'Không có thư viện nào ở đây; đây là các kernel của chính PyTorch.'); } },
        ],
        get found() { return tr('template <typename T, typename T_ACC, bool rms_norm> __global__ void RowwiseMomentsCUDAKernel( (line 57) and vectorized_layer_norm_kernel_impl (line 270): one source, two norms.', 'template <typename T, typename T_ACC, bool rms_norm> __global__ void RowwiseMomentsCUDAKernel( (dòng 57) và vectorized_layer_norm_kernel_impl (dòng 270): một mã nguồn, hai phép chuẩn hóa.'); },
        project: 'pytorch', path: TN + 'cuda/layer_norm_kernel.cu', lines: [57, 270], facts: ['B2'],
      },
    ],
    get lesson() { return tr('One name at a time: Python function, C++ function, yaml key, dispatch target, kernel. The yaml is the map.', 'Mỗi lần một cái tên: hàm Python, hàm C++, khóa yaml, đích dispatch, kernel. Tệp yaml chính là bản đồ.'); },
  },
  {
    id: 'llama-rmsnorm-fused', get label() { return tr('llama.cpp RMS_NORM on CUDA, fusion on', 'llama.cpp RMS_NORM trên CUDA, bật gộp'); }, project: 'llama.cpp', backend: 'cuda', op: 'rmsnorm',
    get start() { return tr('A Llama layer’s graph: an RMS_NORM node, then a MUL by the norm weight, on the CUDA backend with default settings.', 'Đồ thị của một lớp Llama: một nút RMS_NORM, rồi một MUL với trọng số chuẩn hóa, trên backend CUDA với thiết lập mặc định.'); },
    hops: [
      {
        get prompt() { return tr('Before ggml_cuda_compute_forward’s switch sees the RMS_NORM node, what looks at it?', 'Trước khi switch của ggml_cuda_compute_forward thấy nút RMS_NORM, cái gì xem xét nó?'); },
        options: [
          { get label() { return tr('A fusion pass over runs of nodes', 'Một lượt gộp chạy qua các dãy nút'); }, kind: 'function name', correct: true },
          { get label() { return tr('Nothing: the switch is first', 'Không gì cả: switch là bước đầu tiên'); }, kind: 'switch case', get why() { return tr('That is true only with fusion disabled.', 'Điều đó chỉ đúng khi tắt gộp.'); } },
          { get label() { return tr('ggml_backend_cuda_device_supports_op, which runs it', 'ggml_backend_cuda_device_supports_op, hàm chạy nó'); }, kind: 'function name', get why() { return tr('supports_op (line 6336) only says whether the backend accepts the op. It runs nothing.', 'supports_op (dòng 6336) chỉ cho biết backend có nhận op hay không. Nó không chạy gì cả.'); } },
        ],
        get found() { return tr('if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) { ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]); return 1; } (lines 5268 to 5271). RMS_NORM, MUL, ADD is tried one step earlier.', 'if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) { ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]); return 1; } (dòng 5268 đến 5271). RMS_NORM, MUL, ADD được thử sớm hơn một bước.'); },
        project: 'llama.cpp', path: CU + 'ggml-cuda.cu', lines: [5255, 5272], facts: ['A1'],
      },
      {
        get prompt() { return tr('Where is ggml_cuda_op_rms_norm_fused?', 'ggml_cuda_op_rms_norm_fused nằm ở đâu?'); },
        options: [
          { get label() { return tr('In norm.cu, the .cu file named after the op family', 'Trong norm.cu, tệp .cu mang tên họ op'); }, kind: 'file name', correct: true },
          { get label() { return tr('In a fusion.cu with every fused kernel', 'Trong một fusion.cu chứa mọi kernel gộp'); }, kind: 'file name', get why() { return tr('llama.cpp keeps a fused variant beside the plain one.', 'llama.cpp giữ biến thể gộp ngay bên cạnh biến thể thường.'); } },
          { get label() { return tr('Nowhere: it is generated at build time', 'Không ở đâu cả: nó được sinh ra lúc build'); }, get why() { return tr('llama.cpp generates template instances, not this function.', 'llama.cpp sinh các bản khởi tạo template, không sinh hàm này.'); } },
        ],
        get found() { return tr('norm.cu, rms_norm_f32 (lines 127 to 272): one block per row, a sum of squares, block_reduce, rsqrtf(mean + eps), and the weight multiply when it is fused.', 'norm.cu, rms_norm_f32 (dòng 127 đến 272): mỗi hàng một khối, một tổng bình phương, block_reduce, rsqrtf(mean + eps), và phép nhân trọng số khi được gộp.'); },
        project: 'llama.cpp', path: CU + 'norm.cu', lines: [127, 272], facts: ['B2'],
      },
      {
        get prompt() { return tr('A breakpoint in plain ggml_cuda_op_rms_norm never hits. How do you confirm why?', 'Một breakpoint trong ggml_cuda_op_rms_norm bản thường không bao giờ dừng lại. Làm sao bạn xác nhận lý do?'); },
        options: [
          { get label() { return tr('Rerun with GGML_CUDA_DISABLE_FUSION set', 'Chạy lại với GGML_CUDA_DISABLE_FUSION được đặt'); }, kind: 'environment variable', correct: true },
          { get label() { return tr('Rebuild for HIP', 'Build lại cho HIP'); }, get why() { return tr('The HIP build compiles the same files and runs the same fusion pass.', 'Bản build HIP biên dịch cùng các tệp đó và chạy cùng lượt gộp đó.'); } },
          { get label() { return tr('Remove the MUL from the model file', 'Xóa MUL khỏi tệp mô hình'); }, get why() { return tr('That changes the model, not the dispatch.', 'Việc đó thay đổi mô hình, không phải dispatch.'); } },
        ],
        get found() { return tr('ggml_cuda_env_flag("GGML_CUDA_DISABLE_FUSION") is read at line 181. With it set, the switch’s case GGML_OP_RMS_NORM (line 2997) runs ggml_cuda_op_rms_norm.', 'ggml_cuda_env_flag("GGML_CUDA_DISABLE_FUSION") được đọc ở dòng 181. Khi biến này được đặt, case GGML_OP_RMS_NORM của switch (dòng 2997) chạy ggml_cuda_op_rms_norm.'); },
        project: 'llama.cpp', path: CU + 'ggml-cuda.cu', lines: [176, 183], facts: ['A1'],
      },
      {
        get prompt() { return tr('And on an AMD GPU?', 'Còn trên GPU AMD?'); },
        options: [
          { get label() { return tr('The same norm.cu, compiled as HIP', 'Cùng tệp norm.cu đó, biên dịch thành HIP'); }, correct: true },
          { get label() { return tr('ggml-hip/norm.hip, a separate port', 'ggml-hip/norm.hip, một bản port riêng'); }, kind: 'file name', get why() { return tr('ggml-hip holds build files only.', 'ggml-hip chỉ chứa các tệp build.'); } },
          { get label() { return tr('A copy hipified at build time', 'Một bản sao được hipify lúc build'); }, get why() { return tr('That is PyTorch’s approach. llama.cpp renames with #defines in vendors/hip.h.', 'Đó là cách của PyTorch. llama.cpp đổi tên bằng các #define trong vendors/hip.h.'); } },
        ],
        get found() { return tr('ggml-hip/CMakeLists.txt globs ../ggml-cuda/*.cu and sets LANGUAGE HIP (lines 60 to 70).', 'ggml-hip/CMakeLists.txt glob ../ggml-cuda/*.cu và đặt LANGUAGE HIP (dòng 60 đến 70).'); },
        project: 'llama.cpp', path: 'ggml/src/ggml-hip/CMakeLists.txt', lines: [60, 70], facts: ['B9'],
      },
    ],
    get lesson() { return tr('The fusion pass runs before the switch, so the host function you expect may never run.', 'Lượt gộp chạy trước switch, nên hàm host bạn chờ đợi có thể không bao giờ chạy.'); },
  },
  {
    id: 'llama-rmsnorm-plain', get label() { return tr('llama.cpp RMS_NORM on CUDA, fusion off', 'llama.cpp RMS_NORM trên CUDA, tắt gộp'); }, project: 'llama.cpp', backend: 'cuda', op: 'rmsnorm',
    get start() { return tr('The same graph, run with GGML_CUDA_DISABLE_FUSION set.', 'Cùng đồ thị đó, chạy với GGML_CUDA_DISABLE_FUSION được đặt.'); },
    hops: [
      {
        get prompt() { return tr('With fusion off, what do you search ggml-cuda.cu for?', 'Khi tắt gộp, bạn tìm gì trong ggml-cuda.cu?'); },
        options: [
          { get label() { return tr('case GGML_OP_RMS_NORM in ggml_cuda_compute_forward', 'case GGML_OP_RMS_NORM trong ggml_cuda_compute_forward'); }, kind: 'switch case', correct: true },
          { label: 'ggml_cuda_op_rms_norm_fused', kind: 'function name', get why() { return tr('That is the fused path, which is now off.', 'Đó là đường gộp, giờ đã tắt.'); } }, // i18n-ignore: function name
          { get label() { return tr('A kernel-name string', 'Một chuỗi tên kernel'); }, kind: 'formatted string', get why() { return tr('CUDA host code calls the kernel by its symbol.', 'Mã host CUDA gọi kernel bằng ký hiệu của nó.'); } },
        ],
        get found() { return tr('case GGML_OP_RMS_NORM: ggml_cuda_op_rms_norm(ctx, dst); (line 2997), inside switch (dst->op) of ggml_cuda_compute_forward (line 2812).', 'case GGML_OP_RMS_NORM: ggml_cuda_op_rms_norm(ctx, dst); (dòng 2997), bên trong switch (dst->op) của ggml_cuda_compute_forward (dòng 2812).'); },
        project: 'llama.cpp', path: CU + 'ggml-cuda.cu', lines: 2997, facts: ['A1'],
      },
      {
        get prompt() { return tr('Where is ggml_cuda_op_rms_norm?', 'ggml_cuda_op_rms_norm nằm ở đâu?'); },
        options: [
          { label: 'norm.cu', kind: 'file name', correct: true },
          { label: 'rmsnorm.cu', kind: 'file name', get why() { return tr('The file is named for the op family: norm.cu holds every norm.', 'Tệp được đặt tên theo họ op: norm.cu chứa mọi phép chuẩn hóa.'); } },
          { get label() { return tr('ggml-cuda.cu itself', 'Chính ggml-cuda.cu'); }, get why() { return tr('ggml-cuda.cu holds the switch; each host function lives in its op’s file.', 'ggml-cuda.cu chứa switch; mỗi hàm host nằm trong tệp của op tương ứng.'); } },
        ],
        get found() { return tr('norm.cu, the same rms_norm_f32 kernel (lines 127 to 272), now without the fused multiply.', 'norm.cu, cùng kernel rms_norm_f32 đó (dòng 127 đến 272), giờ không có phép nhân gộp.'); },
        project: 'llama.cpp', path: CU + 'norm.cu', lines: [127, 272], facts: ['A1', 'B2'],
      },
      {
        get prompt() { return tr('What runs the MUL by the weight now?', 'Giờ cái gì chạy phép MUL với trọng số?'); },
        options: [
          { get label() { return tr('A second launch, from the switch’s own case for MUL: a binary op', 'Một lần khởi chạy thứ hai, từ case riêng của switch cho MUL: một op nhị phân'); }, kind: 'switch case', correct: true },
          { get label() { return tr('Nothing: RMS_NORM always multiplies', 'Không gì cả: RMS_NORM luôn nhân'); }, get why() { return tr('Only the fused call multiplies; the plain one does not.', 'Chỉ lời gọi gộp mới nhân; lời gọi thường thì không.'); } },
          { get label() { return tr('The CPU backend', 'Backend CPU'); }, get why() { return tr('The CUDA backend accepts MUL; nothing falls back here.', 'Backend CUDA nhận MUL; ở đây không có gì quay về dự phòng.'); } },
        ],
        get found() { return tr('Binary ops go to k_bin_bcast in binbcast.cu (ch03 D1). Two launches and a round trip through memory where the fused path had one.', 'Các op nhị phân đi đến k_bin_bcast trong binbcast.cu (ch03 D1). Hai lần khởi chạy và một vòng đi về qua bộ nhớ, ở chỗ đường gộp chỉ có một.'); },
        project: 'llama.cpp', path: CU + 'binbcast.cu', facts: ['A1', 'B1', 'ch03 D1'], // i18n-ignore: identifier also shown to the reader
      },
    ],
    get lesson() { return tr('With fusion off, the one-name rule works as is: switch case, host function, the .cu file of the same name.', 'Khi tắt gộp, quy tắc một tên dùng được nguyên vẹn: switch case, hàm host, tệp .cu cùng tên.'); },
  },
  {
    id: 'llama-getrows-metal', get label() { return tr('llama.cpp GET_ROWS on Metal', 'llama.cpp GET_ROWS trên Metal'); }, project: 'llama.cpp', backend: 'metal', op: 'embedding',
    get start() { return tr('The embedding lookup of a llama.cpp graph, a GGML_OP_GET_ROWS node with F16 rows, on the Metal backend.', 'Bước tra embedding của một đồ thị llama.cpp, một nút GGML_OP_GET_ROWS với các hàng F16, trên backend Metal.'); },
    hops: [
      {
        get prompt() { return tr('Where does the Metal backend turn the node into work?', 'Backend Metal biến nút thành công việc ở đâu?'); },
        options: [
          { get label() { return tr('A case in ggml_metal_op_encode_impl’s switch', 'Một case trong switch của ggml_metal_op_encode_impl'); }, kind: 'switch case', correct: true },
          { get label() { return tr('The CUDA switch, compiled for Metal', 'Switch của CUDA, biên dịch cho Metal'); }, get why() { return tr('Metal has its own host code and its own kernels.', 'Metal có mã host riêng và kernel riêng.'); } },
          { get label() { return tr('A Metal function named ggml_get_rows', 'Một hàm Metal tên ggml_get_rows'); }, kind: 'function name', get why() { return tr('Metal kernels are found by a name string the host builds, and the kernels are called kernel_*.', 'Kernel Metal được tìm bằng một chuỗi tên do host dựng, và các kernel có tên kernel_*.'); } },
        ],
        get found() { return tr('static int ggml_metal_op_encode_impl(ggml_metal_op_t ctx, int idx) (line 187): GGML_OP_GET_ROWS goes to ggml_metal_op_get_rows. Each case returns n_fuse, the number of nodes it consumed.', 'static int ggml_metal_op_encode_impl(ggml_metal_op_t ctx, int idx) (dòng 187): GGML_OP_GET_ROWS đi đến ggml_metal_op_get_rows. Mỗi case trả về n_fuse, số nút nó đã tiêu thụ.'); },
        project: 'llama.cpp', path: MT + 'ggml-metal-ops.cpp', lines: [187, 490], facts: ['A1'],
      },
      {
        get prompt() { return tr('ggml_metal_op_get_rows needs a pipeline. How does it ask for one?', 'ggml_metal_op_get_rows cần một pipeline. Nó yêu cầu pipeline bằng cách nào?'); },
        options: [
          { get label() { return tr('By a kernel name it formats as a string', 'Bằng một tên kernel mà nó định dạng thành chuỗi'); }, kind: 'formatted string', correct: true },
          { get label() { return tr('By a C++ function pointer to the kernel', 'Bằng một con trỏ hàm C++ tới kernel'); }, kind: 'function name', get why() { return tr('The host cannot call an MSL function by symbol; it asks the library for a function by name.', 'Host không thể gọi một hàm MSL bằng ký hiệu; nó xin thư viện một hàm theo tên.'); } },
          { get label() { return tr('By including the .metal file', 'Bằng cách include tệp .metal'); }, kind: 'file name', get why() { return tr('The kernels are compiled into a library; the host looks functions up in it by name.', 'Các kernel được biên dịch vào một thư viện; host tra cứu hàm trong đó theo tên.'); } },
        ],
        get found() { return tr('snprintf(base, 256, "kernel_get_rows_%s", ggml_type_name(tsrc)); (ggml-metal-device.cpp line 154).', 'snprintf(base, 256, "kernel_get_rows_%s", ggml_type_name(tsrc)); (ggml-metal-device.cpp dòng 154).'); },
        project: 'llama.cpp', path: MT + 'ggml-metal-device.cpp', lines: 154, facts: ['A1'],
      },
      {
        get prompt() { return tr('What do you search the .metal files for?', 'Bạn tìm gì trong các tệp .metal?'); },
        options: [
          { get label() { return tr('The string the host builds: kernel_get_rows_ followed by the type’s name', 'Chuỗi mà host dựng: kernel_get_rows_ theo sau là tên kiểu'); }, kind: 'formatted string', correct: true },
          { label: 'ggml_metal_op_get_rows', kind: 'function name', get why() { return tr('That is the host function you came from.', 'Đó là hàm host bạn vừa đi qua.'); } }, // i18n-ignore: function name
          { get label() { return tr('The template’s C++ name only', 'Chỉ tên C++ của template'); }, kind: 'function name', get why() { return tr('One template serves every type; the string tells you which instantiation runs.', 'Một template phục vụ mọi kiểu; chuỗi cho bạn biết bản khởi tạo nào chạy.'); } },
        ],
        get found() { return tr('kernels/quantize.metal holds kernel_get_rows_f (line 266) for float rows and kernel_get_rows_q (line 297) for quantised rows; the string matches the [[host_name(...)]] of an instantiation. The exact instantiation string for F16 was not read in the facts pass.', 'kernels/quantize.metal chứa kernel_get_rows_f (dòng 266) cho hàng float và kernel_get_rows_q (dòng 297) cho hàng đã lượng tử hóa; chuỗi khớp với [[host_name(...)]] của một bản khởi tạo. Chuỗi khởi tạo chính xác cho F16 chưa được đọc trong lượt kiểm tra dữ kiện.'); },
        project: 'llama.cpp', path: MT + 'kernels/quantize.metal', lines: [266, 297], facts: ['A1', 'B7', 'C2'],
      },
    ],
    get lesson() { return tr('On Metal, the name is a string: search the .metal files for what the host formats.', 'Trên Metal, tên là một chuỗi: tìm trong các tệp .metal thứ mà host định dạng.'); },
  },
  {
    id: 'mlx-softmax-cuda', get label() { return tr('MLX softmax on CUDA', 'MLX softmax trên CUDA'); }, project: 'mlx', backend: 'cuda', op: 'softmax',
    get start() { return tr('MLX’s Softmax primitive on an NVIDIA GPU. (How the Python softmax builds the primitive was not read in the facts pass.)', 'Primitive Softmax của MLX trên GPU NVIDIA. (Cách hàm softmax Python dựng primitive chưa được đọc trong lượt kiểm tra dữ kiện.)'); },
    hops: [
      {
        get prompt() { return tr('MLX ops are Primitive subclasses. What do you search mlx/backend/cuda for?', 'Các op MLX là lớp con của Primitive. Bạn tìm gì trong mlx/backend/cuda?'); },
        options: [
          { label: 'Softmax::eval_gpu', kind: 'function name', correct: true }, // i18n-ignore: function name
          { get label() { return tr('A dispatch key in a yaml file', 'Một khóa dispatch trong một tệp yaml'); }, kind: 'yaml key', get why() { return tr('That is PyTorch’s map.', 'Đó là bản đồ của PyTorch.'); } },
          { get label() { return tr('case SOFTMAX in a switch', 'case SOFTMAX trong một switch'); }, kind: 'switch case', get why() { return tr('That is llama.cpp’s shape.', 'Đó là kiểu của llama.cpp.'); } },
        ],
        get found() { return tr('void Softmax::eval_gpu( in mlx/backend/cuda/softmax.cu (line 108).', 'void Softmax::eval_gpu( trong mlx/backend/cuda/softmax.cu (dòng 108).'); },
        project: 'mlx', path: MLX_CU + 'softmax.cu', lines: 108, facts: ['A2'],
      },
      {
        get prompt() { return tr('Predict: how does that kernel reduce a row?', 'Dự đoán: kernel đó rút gọn một hàng như thế nào?'); },
        options: [
          { get label() { return tr('In one pass, rescaling a running sum whenever the max grows', 'Trong một lượt, co giãn lại tổng đang tích lũy mỗi khi max tăng'); }, correct: true },
          { get label() { return tr('Max, then sum, as lab 10.3 and MLX’s Metal kernel do', 'Max, rồi sum, như lab 10.3 và kernel Metal của MLX'); }, get why() { return tr('That is the Metal side. The two backends of one project differ here.', 'Đó là phía Metal. Hai backend của cùng một dự án khác nhau ở đây.'); } },
          { get label() { return tr('Through cuDNN’s softmax', 'Qua softmax của cuDNN'); }, get why() { return tr('MLX uses cuDNN for attention, not for this op.', 'MLX dùng cuDNN cho attention, không dùng cho op này.'); } },
        ],
        get found() { return tr('normalizer = normalizer * softmax_exp(prevmax - maxval); then add softmax_exp(vals[i] - maxval) (lines 49 to 62): the online normaliser of chapter 20, with __expf and cooperative groups.', 'normalizer = normalizer * softmax_exp(prevmax - maxval); rồi cộng softmax_exp(vals[i] - maxval) (dòng 49 đến 62): bộ chuẩn hóa trực tuyến của chương 20, với __expf và cooperative groups.'); },
        project: 'mlx', path: MLX_CU + 'softmax.cu', lines: [20, 106], facts: ['B3'],
      },
      {
        get prompt() { return tr('Is there an AMD version of this file?', 'Có phiên bản AMD của tệp này không?'); },
        options: [
          { get label() { return tr('No: MLX has no AMD backend', 'Không: MLX không có backend AMD'); }, correct: true },
          { get label() { return tr('Yes, the same file hipified at build time', 'Có, cùng tệp đó được hipify lúc build'); }, get why() { return tr('That is PyTorch. MLX’s backend directory has no HIP or ROCm entry.', 'Đó là PyTorch. Thư mục backend của MLX không có mục HIP hay ROCm.'); } },
          { get label() { return tr('Yes, under mlx/backend/rocm', 'Có, trong mlx/backend/rocm'); }, kind: 'file name', get why() { return tr('There is no such directory at the pinned commit.', 'Không có thư mục như vậy ở commit được ghim.'); } },
        ],
        get found() { return tr('mlx/backend lists common, cpu, cuda, gpu, metal, no_cpu, no_gpu. An experimental ROCm pull request (#2300) is open and unmerged; the CUDA backend also uses cooperative groups, cuBLASLt and cuDNN’s frontend.', 'mlx/backend liệt kê common, cpu, cuda, gpu, metal, no_cpu, no_gpu. Một pull request thử nghiệm cho ROCm (#2300) đang mở và chưa được merge; backend CUDA cũng dùng cooperative groups, cuBLASLt và frontend của cuDNN.'); },
        project: 'mlx', path: 'mlx/backend', facts: ['B9'],
      },
    ],
    get lesson() { return tr('In MLX, search for Name::eval_gpu in the backend directory. Do not assume its two backends compute alike.', 'Trong MLX, tìm Name::eval_gpu trong thư mục backend. Đừng giả định hai backend của nó tính toán giống nhau.'); },
  },
  {
    id: 'torch-add-cuda', get label() { return tr('PyTorch add on CUDA', 'PyTorch add trên CUDA'); }, project: 'pytorch', backend: 'cuda', op: 'add',
    get start() { return tr('add.out, the out= form of torch.add, on CUDA tensors.', 'add.out, dạng out= của torch.add, trên các tensor CUDA.'); },
    hops: [
      {
        get prompt() { return tr('Where do you start?', 'Bạn bắt đầu từ đâu?'); },
        options: [
          { get label() { return tr('Its - func: add.out entry in native_functions.yaml', 'Mục - func: add.out của nó trong native_functions.yaml'); }, kind: 'yaml key', correct: true },
          { get label() { return tr('A file Add.cu in native/cuda', 'Một tệp Add.cu trong native/cuda'); }, kind: 'file name', get why() { return tr('Try it: there is none.', 'Thử xem: không có tệp nào như vậy.'); } },
          { get label() { return tr('REGISTER_DISPATCH(add_stub, ...) in native/cuda', 'REGISTER_DISPATCH(add_stub, ...) trong native/cuda'); }, kind: 'REGISTER_DISPATCH', get why() { return tr('Close, but the registration is in a file that does not exist in the source tree.', 'Gần đúng, nhưng phần đăng ký nằm trong một tệp không tồn tại trong cây mã nguồn.'); } },
        ],
        get found() { return tr('structured: True, ufunc_inner_loop: Generic: add (AllAndComplex, BFloat16, Half, ComplexHalf), and no CUDA: key (lines 565 to 571).', 'structured: True, ufunc_inner_loop: Generic: add (AllAndComplex, BFloat16, Half, ComplexHalf), và không có khóa CUDA: (dòng 565 đến 571).'); },
        project: 'pytorch', path: YAML, lines: [565, 571], facts: ['A3'],
      },
      {
        get prompt() { return tr('ufunc_inner_loop names add. Where is the scalar body?', 'ufunc_inner_loop gọi tên add. Phần thân vô hướng nằm ở đâu?'); },
        options: [
          { label: 'native/ufunc/add.h', kind: 'file name', correct: true },
          { label: 'native/cuda/BinaryAddKernel.cu', kind: 'file name', get why() { return tr('A guess at a name; the ufunc body is device-neutral and lives in native/ufunc.', 'Một tên đoán bừa; phần thân ufunc không phụ thuộc thiết bị và nằm trong native/ufunc.'); } },
          { get label() { return tr('cuBLAS’s axpy', 'axpy của cuBLAS'); }, get why() { return tr('No library: the loop is PyTorch’s own.', 'Không có thư viện nào: vòng lặp là của chính PyTorch.'); } },
        ],
        get found() { return tr('return self + alpha * other; in aten/src/ATen/native/ufunc/add.h.', 'return self + alpha * other; trong aten/src/ATen/native/ufunc/add.h.'); },
        project: 'pytorch', path: TN + 'ufunc/add.h', facts: ['A3'],
      },
      {
        get prompt() { return tr('So where is the CUDA kernel?', 'Vậy kernel CUDA nằm ở đâu?'); },
        options: [
          { get label() { return tr('In UfuncCUDA_add.cu, which torchgen writes at build time', 'Trong UfuncCUDA_add.cu, tệp do torchgen viết lúc build'); }, correct: true },
          { get label() { return tr('In native/cuda under another name', 'Trong native/cuda dưới một tên khác'); }, get why() { return tr('Search all you like: it is not in the source tree.', 'Tìm bao nhiêu tùy bạn: nó không có trong cây mã nguồn.'); } },
          { get label() { return tr('Nowhere: add runs on the CPU', 'Không ở đâu cả: add chạy trên CPU'); }, get why() { return tr('It runs on the GPU, from a file that exists only after a build.', 'Nó chạy trên GPU, từ một tệp chỉ tồn tại sau khi build.'); } },
        ],
        get found() { return tr('torchgen/gen.py writes UfuncCUDA_{name}.cu (line 2536); torchgen/dest/ufunc.py emits gpu_kernel(iter, ufunctor); and REGISTER_DISPATCH(...) (lines 279, 288, 328).', 'torchgen/gen.py viết UfuncCUDA_{name}.cu (dòng 2536); torchgen/dest/ufunc.py sinh ra gpu_kernel(iter, ufunctor); và REGISTER_DISPATCH(...) (dòng 279, 288, 328).'); },
        project: 'pytorch', path: 'torchgen/dest/ufunc.py', lines: [279, 328], facts: ['A3'],
      },
      {
        get prompt() { return tr('And on MPS?', 'Còn trên MPS?'); },
        options: [
          { get label() { return tr('A hand-written add_out_mps, because ufunc codegen emits only CPU and CUDA', 'Một add_out_mps viết tay, vì codegen của ufunc chỉ sinh CPU và CUDA'); }, correct: true },
          { get label() { return tr('The same generated file, compiled for Metal', 'Cùng tệp được sinh ra đó, biên dịch cho Metal'); }, get why() { return tr('Codegen does not emit an MPS version, and the MPS source says so.', 'Codegen không sinh phiên bản MPS, và mã nguồn MPS nói rõ điều đó.'); } },
          { get label() { return tr('MPSGraph only', 'Chỉ MPSGraph'); }, get why() { return tr('PyTorch MPS has its own Metal kernel for add.', 'PyTorch MPS có kernel Metal riêng cho add.'); } },
        ],
        get found() { return tr('“add.out is a ufunc op, and ufunc codegen only emits CPU/CUDA impls, so MPS needs this entry” (BinaryOps.mm lines 16 to 19), then lib.exec_binary_kernel(iter, "add") and add_alpha_functor in kernels/BinaryKernel.metal (lines 22 to 25).', '“add.out is a ufunc op, and ufunc codegen only emits CPU/CUDA impls, so MPS needs this entry” (add.out là một op ufunc, và codegen của ufunc chỉ sinh bản cài đặt CPU/CUDA, nên MPS cần mục này) (BinaryOps.mm dòng 16 đến 19), rồi lib.exec_binary_kernel(iter, "add") và add_alpha_functor trong kernels/BinaryKernel.metal (dòng 22 đến 25).'); },
        project: 'pytorch', path: TN + 'mps/operations/BinaryOps.mm', lines: [16, 19], facts: ['A3'],
      },
    ],
    get lesson() { return tr('Some PyTorch CUDA kernels exist in no source file. When the yaml has no CUDA key, follow ufunc_inner_loop or the stub.', 'Một số kernel CUDA của PyTorch không nằm trong tệp mã nguồn nào. Khi yaml không có khóa CUDA, hãy lần theo ufunc_inner_loop hoặc stub.'); },
  },
];
export function route(id: string): Route {
  const r = ROUTES.find((x) => x.id === id);
  if (!r) throw new Error('no route ' + id);
  return r;
}
/** The five routes the stepper offers; RMS_NORM's two are one route with a fusion switch. */
export const ROUTE_CHOICES: { id: string; label: string; variants?: { on: string; off: string } }[] = [
  { id: 'torch-rmsnorm-cuda', label: 'PyTorch rms_norm, CUDA' }, // i18n-ignore: project, op and backend names, the same in Vietnamese
  { id: 'llama-rmsnorm', label: 'llama.cpp RMS_NORM, CUDA', variants: { on: 'llama-rmsnorm-fused', off: 'llama-rmsnorm-plain' } }, // i18n-ignore: project, op and backend names, the same in Vietnamese
  { id: 'llama-getrows-metal', label: 'llama.cpp GET_ROWS, Metal' }, // i18n-ignore: project, op and backend names, the same in Vietnamese
  { id: 'mlx-softmax-cuda', label: 'MLX softmax, CUDA' }, // i18n-ignore: project, op and backend names, the same in Vietnamese
  { id: 'torch-add-cuda', label: 'PyTorch add, CUDA' }, // i18n-ignore: project, op and backend names, the same in Vietnamese
];

/** Score a walk: for each hop answered, whether the first pick was right. */
export function scoreWalk(r: Route, firstPicks: (number | null)[]): { answered: number; right: number } {
  let answered = 0;
  let right = 0;
  r.hops.forEach((h, i) => {
    const p = firstPicks[i];
    if (p == null) return;
    answered++;
    if (h.options[p]?.correct) right++;
  });
  return { answered, right };
}

// ---------------------------------------------------------------- variant mechanisms (X3.2)
export type Mechanism = 'compile-time' | 'pipeline-constant' | 'name-string' | 'registration' | 'build-time-codegen' | 'run-time-codegen';
export const MECHANISMS: { id: Mechanism; label: string }[] = [
  { id: 'compile-time', get label() { return tr('compile time: a template or a macro', 'lúc biên dịch: một template hoặc một macro'); } },
  { id: 'pipeline-constant', get label() { return tr('fixed when the pipeline is created', 'cố định khi pipeline được tạo'); } },
  { id: 'name-string', get label() { return tr('looked up by a name string', 'tra cứu bằng một chuỗi tên'); } },
  { id: 'registration', get label() { return tr('a registration table', 'một bảng đăng ký'); } },
  { id: 'build-time-codegen', get label() { return tr('generated at build time', 'được sinh ra lúc build'); } },
  { id: 'run-time-codegen', get label() { return tr('generated at run time', 'được sinh ra lúc chạy'); } },
];
export interface VariantItem {
  id: string;
  code: string;
  project: Project;
  backend: Backend;
  mechanism: Mechanism;
  why: string;
  facts: string;
}
export const VARIANT_MECHANISMS: VariantItem[] = [
  {
    id: 'llama-template', code: 'template <float (*op)(float), typename T>', project: 'llama.cpp', backend: 'cuda', mechanism: 'compile-time',
    get why() { return tr('Each op is a separate instantiation of unary_op_kernel, compiled into the binary. HIP compiles the same template.', 'Mỗi op là một bản khởi tạo riêng của unary_op_kernel, được biên dịch vào tệp nhị phân. HIP biên dịch cùng template đó.'); }, facts: 'B1',
  },
  {
    id: 'llama-fc', code: 'constant short FC_unary_op [[function_constant(FC_UNARY + 0)]];', project: 'llama.cpp', backend: 'metal', mechanism: 'pipeline-constant',
    get why() { return tr('Its value is supplied when the specialised function and pipeline are created; the source is compiled once.', 'Giá trị của nó được cung cấp khi hàm chuyên biệt hóa và pipeline được tạo; mã nguồn chỉ được biên dịch một lần.'); }, facts: 'B1, C1',
  },
  {
    id: 'llama-hostname', code: 'template [[host_name("kernel_rms_norm_mul_f32")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 2>;',
    project: 'llama.cpp', backend: 'metal', mechanism: 'name-string',
    get why() { return tr('The instantiation is compiled ahead, and the host finds it by the string in [[host_name]].', 'Bản khởi tạo được biên dịch trước, và host tìm nó bằng chuỗi trong [[host_name]].'); }, facts: 'C2',
  },
  {
    id: 'llama-snprintf', code: 'snprintf(base, 256, "kernel_get_rows_%s", ggml_type_name(tsrc));', project: 'llama.cpp', backend: 'metal', mechanism: 'name-string',
    get why() { return tr('The host side of the same mechanism: it builds the string it will look up.', 'Phía host của cùng cơ chế đó: nó dựng chuỗi mà nó sẽ tra cứu.'); }, facts: 'A1',
  },
  {
    id: 'mlx-binary-gpu', code: 'BINARY_GPU(Add)', project: 'mlx', backend: 'cuda', mechanism: 'compile-time',
    get why() { return tr('A macro that stamps out Add::eval_gpu when MLX is compiled.', 'Một macro sinh ra Add::eval_gpu khi MLX được biên dịch.'); }, facts: 'A2',
  },
  {
    id: 'mlx-rope-template', code: 'template <typename T, bool traditional, bool forward>', project: 'mlx', backend: 'cuda', mechanism: 'compile-time',
    get why() { return tr('The RoPE pairing as a template parameter, on CUDA.', 'Cách ghép cặp của RoPE dưới dạng tham số template, trên CUDA.'); }, facts: 'B6, C1',
  },
  {
    id: 'mlx-rope-fc', code: 'constant bool traditional [[function_constant(2)]];', project: 'mlx', backend: 'metal', mechanism: 'pipeline-constant',
    get why() { return tr('The same choice on Metal, as a function constant.', 'Cùng lựa chọn đó trên Metal, dưới dạng function constant.'); }, facts: 'B6, C1',
  },
  {
    id: 'torch-register', code: 'REGISTER_DISPATCH(silu_stub, &silu_kernel)', project: 'pytorch', backend: 'cuda', mechanism: 'registration',
    get why() { return tr('The stub is a table entry per device; the CUDA file fills it when the library loads.', 'Stub là một mục trong bảng cho mỗi thiết bị; tệp CUDA điền nó khi thư viện được nạp.'); }, facts: 'A3',
  },
  {
    id: 'torch-ufunc', code: 'gpu_kernel(iter, ufunctor);', project: 'pytorch', backend: 'cuda', mechanism: 'build-time-codegen',
    get why() { return tr('torchgen writes this line into UfuncCUDA_add.cu while PyTorch builds.', 'torchgen viết dòng này vào UfuncCUDA_add.cu trong lúc PyTorch build.'); }, facts: 'A3',
  },
  {
    id: 'torch-fmt', code: 'fmt::format("{}_{}", name, scalarToMetalTypeString(output))', project: 'pytorch', backend: 'metal', mechanism: 'name-string',
    get why() { return tr('PyTorch MPS builds a kernel name and asks for its pipeline with getPipelineStateForFunc.', 'PyTorch MPS dựng một tên kernel và xin pipeline của nó bằng getPipelineStateForFunc.'); }, facts: 'C2',
  },
  {
    id: 'mlx-compiled', code: '[[host_name(\\"{0}\\")]]\\n[[kernel]] void {0}(', project: 'mlx', backend: 'metal', mechanism: 'run-time-codegen',
    get why() { return tr('Compiled::eval_gpu writes this into kernel source the first time an mx.compile function runs, and compiles it.', 'Compiled::eval_gpu viết đoạn này vào mã nguồn kernel ở lần đầu một hàm mx.compile chạy, rồi biên dịch nó.'); }, facts: 'A2',
  },
];
export function gradeSort(picks: Record<string, Mechanism | undefined>): { right: number; total: number; wrong: string[] } {
  const wrong = VARIANT_MECHANISMS.filter((v) => picks[v.id] !== v.mechanism).map((v) => v.id);
  return { right: VARIANT_MECHANISMS.length - wrong.length, total: VARIANT_MECHANISMS.length, wrong };
}

// ---------------------------------------------------------------- the "One RMSNorm call" scene
export type GraphOp = 'GET_ROWS' | 'RMS_NORM' | 'MUL' | 'ADD' | 'MUL_MAT';
/** The CUDA fusion rules read in A1 that start with RMS_NORM, longest first, as the pass tries them. */
export const CUDA_FUSIONS: { ops: GraphOp[]; call: string }[] = [
  // A1 quotes this rule but not the name of the function it calls.
  { ops: ['RMS_NORM', 'MUL', 'ADD'], get call() { return tr('the fused RMS_NORM, MUL, ADD call', 'lời gọi gộp RMS_NORM, MUL, ADD'); } },
  { ops: ['RMS_NORM', 'MUL'], call: 'ggml_cuda_op_rms_norm_fused' },
];
export const PLAIN_CALL: Record<GraphOp, string> = {
  GET_ROWS: 'ggml_cuda_op_get_rows',
  RMS_NORM: 'ggml_cuda_op_rms_norm',
  get MUL() { return tr('binary op (binbcast.cu)', 'op nhị phân (binbcast.cu)'); },
  ADD: 'ggml_cuda_op_add',
  MUL_MAT: 'ggml_cuda_mul_mat',
};
export interface Dispatch {
  nodes: number[];
  call: string;
  fused: boolean;
}
/** What the CUDA backend calls for each run of nodes, with or without the fusion pass. */
export function fuseGraph(graph: GraphOp[], fusion: boolean): Dispatch[] {
  const out: Dispatch[] = [];
  for (let i = 0; i < graph.length; ) {
    const rule = fusion ? CUDA_FUSIONS.find((f) => f.ops.every((o, k) => graph[i + k] === o)) : undefined;
    if (rule) {
      out.push({ nodes: rule.ops.map((_, k) => i + k), call: rule.call, fused: true });
      i += rule.ops.length;
    } else {
      out.push({ nodes: [i], call: PLAIN_CALL[graph[i]], fused: false });
      i++;
    }
  }
  return out;
}
/** llama.cpp Metal's RMSNorm instantiations (norm.metal lines 178 to 185, float forms), by fusion level F. */
export const METAL_NORM_NAMES: { F: number; ops: GraphOp[]; name: string }[] = [
  { F: 1, ops: ['RMS_NORM'], name: 'kernel_rms_norm_f32' }, // i18n-ignore: kernel name
  { F: 2, ops: ['RMS_NORM', 'MUL'], name: 'kernel_rms_norm_mul_f32' }, // i18n-ignore: kernel name
  { F: 3, ops: ['RMS_NORM', 'MUL', 'ADD'], name: 'kernel_rms_norm_mul_add_f32' }, // i18n-ignore: kernel name
];
/** The Metal name for a run of nodes starting at an RMS_NORM: the longest matching fusion level. */
export function metalNormName(graph: GraphOp[], i: number): { F: number; consumed: number; name: string } {
  const m = [...METAL_NORM_NAMES].reverse().find((x) => x.ops.every((o, k) => graph[i + k] === o))!;
  return { F: m.F, consumed: m.ops.length, name: m.name };
}
export const ONE_CALL = {
  /** The part of a Llama layer the scene shows (ours: the order of a pre-norm attention block's start). */
  graph: ['GET_ROWS', 'RMS_NORM', 'MUL', 'MUL_MAT'] as GraphOp[],
  cudaKernel: { file: 'norm.cu', kernel: 'rms_norm_f32', lines: [127, 272] as [number, number] },
  metalSwitch: { fn: 'ggml_metal_op_encode_impl', line: 187, host: 'ggml_metal_op_norm', caseLine: 414 },
  metalTemplate: 'kernel_rms_norm_fuse_impl',
};

// ---------------------------------------------------------------- the start page's prediction
/** Where a project's CUDA attention kernels come from, read off the attention cells' tags. */
export type AttnHome = 'own' | 'mixed' | 'submodule';
export const ATTN_HOME_OPTIONS: { id: AttnHome; label: string }[] = [
  { id: 'own', get label() { return tr('its own source tree, for every shape', 'cây mã nguồn của chính nó, cho mọi hình dạng'); } },
  { id: 'mixed', get label() { return tr('its own kernels for some shapes, a vendor library for others', 'kernel của chính nó cho một số hình dạng, thư viện của nhà cung cấp cho các hình dạng khác'); } },
  { id: 'submodule', get label() { return tr('first choice is another project’s kernels, pulled in as a git submodule', 'lựa chọn đầu tiên là kernel của một dự án khác, được kéo vào dưới dạng git submodule'); } },
];
export function attentionHome(p: Project): AttnHome {
  const t = cell('attention', p, 'cuda').tags;
  if (t.includes('third-party submodule')) return 'submodule';
  if (t.includes('vendor library')) return 'mixed';
  return 'own';
}

/**
 * The order a hop's options are shown in: a permutation seeded by the route and hop, the same
 * on every visit, so the right option is not always where the data lists it (first).
 */
export function hopOrder(routeId: string, hop: number, n: number): number[] {
  let h = 2166136261;
  const key = `${routeId}#${hop}`;
  for (let i = 0; i < key.length; i++) h = Math.imul(h ^ key.charCodeAt(i), 16777619) >>> 0;
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
    const j = h % (i + 1);
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx;
}
