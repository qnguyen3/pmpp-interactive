// Which kernel a matrix product runs, by project and shape (module X3, section X3.4). Not in
// the book. Every check below is one read in notes/today/x3-facts.md (B4), ch22-facts.md (D6)
// or ch15-facts.md (D1 to D4), at the commits listed there; the line numbers are the facts'.
//
// What it models: the decision chains in their source order, for a product
//   Y[M x N] = X[M x K] * W^T   (M tokens, W stored (N, K) as nn.Linear stores it).
// Each check is evaluated as true or false only when the condition the facts quote decides it;
// everything decided inside a function the facts did not read (ggml_cuda_should_use_mmf,
// ggml_cuda_should_use_mmq, the rest of should_use_mmvf and should_use_mmvq, the FP8 path,
// ggml_metal_op_mul_mat_use_mm, MLX's wide-GEMV conditions other than its limit of 5, cuBLAS's
// own algorithm choice) is returned as 'not-modelled'. The route is then the first check not
// ruled out, and it is certain only when that check is true. One reading is ours, not the
// facts': mmvf and mmf (the "_f" functions) serve float weights and mmvq and mmq (the "_q"
// ones) quantised weights, so a weight of the other class rules a candidate out.
//
// What it leaves out: steel and cuBLAS tile selection, data types other than the weight's
// class, strides and batching of 3-D operands, and every per-GPU exception (it names them).

import { tr } from '../lib/locale.ts';
import { gpuBackends, stepSeries, type ReportsLike } from './batching';

export type Project = 'llama-cuda' | 'llama-metal' | 'mlx-cuda' | 'mlx-metal' | 'torch-cuda' | 'torch-mps';
export const PROJECTS: Project[] = ['llama-cuda', 'llama-metal', 'mlx-cuda', 'mlx-metal', 'torch-cuda', 'torch-mps'];
export const PROJECT_LABEL: Record<Project, string> = {
  get 'llama-cuda'() {
    return tr('llama.cpp, CUDA and HIP', 'llama.cpp, CUDA và HIP');
  },
  'llama-metal': 'llama.cpp, Metal', // i18n-ignore: project and backend names
  'mlx-cuda': 'MLX, CUDA', // i18n-ignore: project and backend names
  'mlx-metal': 'MLX, Metal', // i18n-ignore: project and backend names
  get 'torch-cuda'() {
    return tr('PyTorch, CUDA (and ROCm)', 'PyTorch, CUDA (và ROCm)');
  },
  get 'torch-mps'() {
    return tr('PyTorch, MPS (addmm and mm)', 'PyTorch, MPS (addmm và mm)');
  },
};

export type WeightType = 'f32' | 'f16' | 'quant' | 'kquant';
export const WEIGHT_LABEL: Record<WeightType, string> = {
  f32: 'F32',
  f16: 'F16',
  get quant() {
    return tr('quantised (not a k-quant)', 'lượng tử hóa (không phải k-quant)');
  },
  get kquant() {
    return tr('quantised (k-quant)', 'lượng tử hóa (k-quant)');
  },
};
const isQuant = (w: WeightType) => w === 'quant' || w === 'kquant';

export interface Shape {
  /** Tokens: rows of X (llama.cpp's ne11). */
  M: number;
  /** Output features: rows of W. */
  N: number;
  /** Input features. */
  K: number;
  weight: WeightType;
  /** llama.cpp CUDA: are the activations (src1) F32? ggml-cuda.cu 2547 tests src1->type and dst->type, never src0 (x3 B4). The output is F32 throughout this model. */
  inputF32: boolean;
  /** MLX: b is a transposed view (x @ W.T with W stored (out, in)). */
  bTransposed: boolean;
  /** MLX: a is a transposed view. */
  aTransposed: boolean;
  /** MLX CUDA: a bias is fused (addmm); the GEMV branch requires none. */
  bias: boolean;
}

export const DEFAULT_SHAPE: Shape = { M: 1, N: 4096, K: 4096, weight: 'f16', inputF32: true, bTransposed: true, aTransposed: false, bias: false };

export type RouteId =
  | 'cublas' | 'mmvf' | 'mmf' | 'mmvq' | 'mmq' | 'fp8'
  | 'mul_mv' | 'mul_mv_ext' | 'mul_mm' | 'metal-unread'
  | 'mlx-gemv' | 'cublaslt'
  | 'wide' | 'gemv' | 'steel'
  | 'torch-blas'
  | 'mps-gemv' | 'mps-matmul';
export type Kind = 'gemv' | 'gemm' | 'library' | 'unknown';

export const ROUTES: Record<RouteId, { name: string; kind: Kind; where: string }> = {
  cublas: {
    name: 'cuBLAS (ggml_cuda_mul_mat_cublas)', // i18n-ignore: kernel and API names
    kind: 'library',
    get where() {
      return tr('ggml/src/ggml-cuda/ggml-cuda.cu (the HIP build goes through the renames in vendors/hip.h)', 'ggml/src/ggml-cuda/ggml-cuda.cu (bản build HIP đi qua các phép đổi tên trong vendors/hip.h)');
    },
  },
  mmvf: {
    get name() {
      return tr('mmvf: float GEMV (ggml_cuda_mul_mat_vec_f)', 'mmvf: GEMV số thực (ggml_cuda_mul_mat_vec_f)');
    },
    kind: 'gemv',
    where: 'ggml/src/ggml-cuda/mmvf.cu',
  },
  mmf: { name: 'mmf (ggml_cuda_mul_mat_f)', kind: 'gemm', where: 'ggml/src/ggml-cuda/mmf.cu' }, // i18n-ignore: kernel and API names, file path
  mmvq: {
    get name() {
      return tr('mmvq: quantised GEMV (ggml_cuda_mul_mat_vec_q)', 'mmvq: GEMV lượng tử hóa (ggml_cuda_mul_mat_vec_q)');
    },
    kind: 'gemv',
    where: 'ggml/src/ggml-cuda/mmvq.cu',
  },
  mmq: {
    get name() {
      return tr('mmq: quantised GEMM (ggml_cuda_mul_mat_q)', 'mmq: GEMM lượng tử hóa (ggml_cuda_mul_mat_q)');
    },
    kind: 'gemm',
    where: 'ggml/src/ggml-cuda/mmq.cu',
  },
  fp8: {
    get name() {
      return tr('the FP8 path', 'nhánh FP8');
    },
    kind: 'unknown',
    where: 'ggml/src/ggml-cuda/ggml-cuda.cu',
  },
  mul_mv: {
    get name() {
      return tr('mul_mv (matrix-vector)', 'mul_mv (ma trận với vector)');
    },
    kind: 'gemv',
    where: 'ggml/src/ggml-metal/kernels/mul_mv.metal',
  },
  mul_mv_ext: {
    get name() {
      return tr('mul_mv_ext (small-batch matrix-vector)', 'mul_mv_ext (ma trận với vector, batch nhỏ)');
    },
    kind: 'gemv',
    where: 'ggml/src/ggml-metal/ggml-metal-ops.cpp',
  },
  mul_mm: {
    get name() {
      return tr('mul_mm (SIMD-group matrices)', 'mul_mm (ma trận SIMD group)');
    },
    kind: 'gemm',
    where: 'ggml/src/ggml-metal/kernels/mul_mm.metal',
  },
  'metal-unread': {
    get name() {
      return tr('a path the facts did not read', 'một nhánh mà facts chưa đọc');
    },
    kind: 'unknown',
    where: 'ggml/src/ggml-metal/ggml-metal-ops.cpp',
  },
  'mlx-gemv': {
    get name() {
      return tr('gemv: a warp per row (gemv_impl)', 'gemv: một warp cho mỗi hàng (gemv_impl)');
    },
    kind: 'gemv',
    where: 'mlx/backend/cuda/gemms/gemv.cu',
  },
  cublaslt: { name: 'cuBLASLt (CublasGemm)', kind: 'library', where: 'mlx/backend/cuda/gemms/cublas_gemm.cpp' }, // i18n-ignore: kernel and API names, file path
  wide: {
    get name() {
      return tr('the wide GEMV', 'GEMV rộng');
    },
    kind: 'gemv',
    where: 'mlx/backend/metal/matmul.cpp',
  },
  gemv: { name: 'gemv (GEMVKernel)', kind: 'gemv', where: 'mlx/backend/metal/kernels/gemv.h' }, // i18n-ignore: kernel and API names, file path
  steel: {
    get name() {
      return tr('steel GEMM, tiles chosen per device', 'steel GEMM, ô được chọn theo từng device');
    },
    kind: 'gemm',
    where: 'mlx/backend/metal/kernels/steel/gemm/',
  },
  'torch-blas': {
    get name() {
      return tr('cuBLAS or cuBLASLt (CK on ROCm), by the preferred BLAS backend', 'cuBLAS hoặc cuBLASLt (CK trên ROCm), theo backend BLAS được ưu tiên');
    },
    kind: 'library',
    where: 'aten/src/ATen/cuda/CUDABlas.cpp',
  },
  'mps-gemv': {
    get name() {
      return tr('gemv_t / gemv_nt kernels', 'các kernel gemv_t / gemv_nt');
    },
    kind: 'gemv',
    where: 'aten/src/ATen/native/mps/kernels/Gemv.h',
  },
  'mps-matmul': {
    get name() {
      return tr('PyTorch\'s Metal matmul kernels or MPSGraph (which one was not read)', 'các kernel matmul Metal của PyTorch hoặc MPSGraph (chưa đọc là cái nào)');
    },
    kind: 'gemm',
    where: 'aten/src/ATen/native/mps/operations/Matmul.mm',
  },
};

export type Verdict = 'true' | 'false' | 'not-modelled';
export interface Check {
  label: string;
  verdict: Verdict;
  /** What runs when the check is true. */
  route: RouteId;
  detail: string;
  source: string;
}
export interface RouteResult {
  project: Project;
  checks: Check[];
  /** Index of the first check not ruled out. */
  at: number;
  /** The first route not ruled out. */
  candidate: RouteId;
  kind: Kind;
  /** True when that check is true, not merely undecided. */
  certain: boolean;
  /** Routes still possible after the candidate (undecided checks and the fallback). */
  alternatives: RouteId[];
  notes: string[];
}

// Constants read in the facts.
/** mmvf.cuh: "Max. batch size for which to use MMVF kernels." (ch22 D6) */
export const MMVF_MAX_BATCH_SIZE = 8;
/** mmvq.cuh: "Max. batch size for which to use MMVQ kernels." (ch22 D6) */
export const MMVQ_MAX_BATCH_SIZE = 8;
/** llama.cpp Metal mul_mv_ext: (ne11 >= 2 && ne11 <= 8), or (ne11 >= 4 && ne11 <= 8) for k-quants (ch22 D6). */
export const METAL_EXT = { lo: 2, loKQuant: 4, hi: 8 };
/** MLX Metal: "streams the weight matrix once per <= 5 input vectors" (x3 B4, matmul.cpp line 1620). */
export const MLX_WIDE_MAX = 5;

const T = (b: boolean): Verdict => (b ? 'true' : 'false');

/** MLX CUDA, gemv.cu line 188, as quoted: (M == 1 && b_transposed) || (N == 1 && !a_transposed). */
export const canUseGemv = (M: number, N: number, aT: boolean, bT: boolean) => (M === 1 && bT) || (N === 1 && !aT);

function checksFor(p: Project, s: Shape): { checks: Check[]; notes: string[] } {
  switch (p) {
    case 'llama-cuda': {
      const overVec = s.M > MMVF_MAX_BATCH_SIZE;
      return {
        checks: [
          { label: tr('the activations (src1) or the output are not F32', 'activation (src1) hoặc đầu ra không phải F32'), verdict: T(!s.inputF32), route: 'cublas', detail: s.inputF32 ? tr('F32: go on', 'F32: đi tiếp') : tr('not F32: straight to cuBLAS', 'không phải F32: thẳng tới cuBLAS'), source: tr('x3 B4, ggml-cuda.cu 2547 to 2548 (src1->type, dst->type)', 'x3 B4, ggml-cuda.cu 2547 đến 2548 (src1->type, dst->type)') },
          {
            label: 'ggml_cuda_should_use_mmvf', // i18n-ignore: function name
            verdict: overVec ? 'false' : isQuant(s.weight) ? 'false' : 'not-modelled',
            route: 'mmvf',
            detail: overVec
              ? tr(`${s.M} tokens > MMVF_MAX_BATCH_SIZE ${MMVF_MAX_BATCH_SIZE}: ruled out`, `${s.M} token > MMVF_MAX_BATCH_SIZE ${MMVF_MAX_BATCH_SIZE}: bị loại`)
              : isQuant(s.weight)
                ? tr('a quantised weight: mmvf is the float GEMV (our reading of the names in the facts)', 'trọng số lượng tử hóa: mmvf là GEMV số thực (cách chúng tôi đọc các tên trong facts)')
                : tr(`${s.M} ≤ ${MMVF_MAX_BATCH_SIZE}; the rest, including AMD branches, is decided inside should_use_mmvf (not modelled)`, `${s.M} ≤ ${MMVF_MAX_BATCH_SIZE}; phần còn lại, kể cả các nhánh AMD, được quyết định bên trong should_use_mmvf (không mô hình hóa)`),
            source: 'x3 B4, ch22 D6', // i18n-ignore: citation
          },
          {
            label: 'ggml_cuda_should_use_mmf', // i18n-ignore: function name
            verdict: isQuant(s.weight) ? 'false' : 'not-modelled',
            route: 'mmf',
            detail: isQuant(s.weight) ? tr('a quantised weight: mmf is the float matmul, beside mmq (our reading of the names)', 'trọng số lượng tử hóa: mmf là phép nhân ma trận số thực, bên cạnh mmq (cách chúng tôi đọc các tên)') : tr('decided inside should_use_mmf (not modelled)', 'được quyết định bên trong should_use_mmf (không mô hình hóa)'),
            source: tr('x3 B4, ggml-cuda.cu 2575 to 2576', 'x3 B4, ggml-cuda.cu 2575 đến 2576'),
          },
          {
            label: 'ggml_cuda_should_use_mmvq', // i18n-ignore: function name
            verdict: s.M > MMVQ_MAX_BATCH_SIZE ? 'false' : !isQuant(s.weight) ? 'false' : 'not-modelled',
            route: 'mmvq',
            detail: s.M > MMVQ_MAX_BATCH_SIZE
              ? tr(`${s.M} tokens > MMVQ_MAX_BATCH_SIZE ${MMVQ_MAX_BATCH_SIZE}: ruled out`, `${s.M} token > MMVQ_MAX_BATCH_SIZE ${MMVQ_MAX_BATCH_SIZE}: bị loại`)
              : !isQuant(s.weight)
                ? tr('a float weight: mmvq is the quantised GEMV (our reading of the names in the facts)', 'trọng số số thực: mmvq là GEMV lượng tử hóa (cách chúng tôi đọc các tên trong facts)')
                : tr(`${s.M} ≤ ${MMVQ_MAX_BATCH_SIZE}, but the limit is lower for some types on some GPUs ("tuned on RTX 4090"): not modelled`, `${s.M} ≤ ${MMVQ_MAX_BATCH_SIZE}, nhưng giới hạn thấp hơn với một số kiểu trên một số GPU ("tuned on RTX 4090"): không mô hình hóa`),
            source: 'x3 B4, ch22 D6', // i18n-ignore: citation
          },
          {
            label: 'ggml_cuda_should_use_mmq', // i18n-ignore: function name
            verdict: isQuant(s.weight) ? 'not-modelled' : 'false',
            route: 'mmq',
            detail: isQuant(s.weight) ? tr('decided inside should_use_mmq (not modelled)', 'được quyết định bên trong should_use_mmq (không mô hình hóa)') : tr('a float weight: mmq is the quantised matmul (our reading of the names)', 'trọng số số thực: mmq là phép nhân ma trận lượng tử hóa (cách chúng tôi đọc các tên)'),
            source: tr('x3 B4, ggml-cuda.cu 2589 to 2590', 'x3 B4, ggml-cuda.cu 2589 đến 2590'),
          },
          { label: tr('an FP8 path', 'một nhánh FP8'), verdict: 'not-modelled', route: 'fp8', detail: tr('its condition was not read', 'điều kiện của nó chưa được đọc'), source: 'x3 B4' }, // i18n-ignore: verdict identifier
          { label: tr('otherwise', 'ngược lại'), verdict: 'true', route: 'cublas', detail: 'ggml_cuda_mul_mat_cublas', source: 'x3 B4, ggml-cuda.cu 2596' }, // i18n-ignore: verdict identifier, function name, citation
        ],
        notes: [tr('HIP compiles the same file: the chain is the same, with AMD branches inside should_use_mmvf (ch22 D6, x3 B9).', 'HIP biên dịch cùng tệp này: chuỗi kiểm tra giống hệt, với các nhánh AMD bên trong should_use_mmvf (ch22 D6, x3 B9).')],
      };
    }
    case 'llama-metal': {
      const lo = s.weight === 'kquant' ? METAL_EXT.loKQuant : METAL_EXT.lo;
      const inExt = s.M >= lo && s.M <= METAL_EXT.hi;
      return {
        checks: [
          { label: tr('one token', 'một token'), verdict: T(s.M === 1), route: 'mul_mv', detail: s.M === 1 ? 'mul_mv.metal' : tr(`${s.M} tokens`, `${s.M} token`), source: tr('x3 B4 ("mul_mv.metal for one"; where this test sits in the code was not read)', 'x3 B4 ("mul_mv.metal for one"; chưa đọc phép kiểm tra này nằm ở đâu trong mã)') },
          {
            label: tr(`${lo} ≤ tokens ≤ ${METAL_EXT.hi}${s.weight === 'kquant' ? ' (k-quants)' : ''}`, `${lo} ≤ token ≤ ${METAL_EXT.hi}${s.weight === 'kquant' ? ' (k-quant)' : ''}`),
            verdict: inExt ? 'not-modelled' : 'false',
            route: 'mul_mv_ext',
            detail: inExt
              ? tr('in the range; the condition is "for most types" and the full type list was not read', 'trong khoảng; điều kiện là "for most types" và danh sách kiểu đầy đủ chưa được đọc')
              : tr(`${s.M} tokens: outside ${lo} to ${METAL_EXT.hi}`, `${s.M} token: ngoài khoảng ${lo} đến ${METAL_EXT.hi}`),
            source: 'ch22 D6, ggml-metal-ops.cpp', // i18n-ignore: citation and file name
          },
          { label: 'ggml_metal_op_mul_mat_use_mm', verdict: 'not-modelled', route: 'mul_mm', detail: tr('decided inside use_mm (not modelled)', 'được quyết định bên trong use_mm (không mô hình hóa)'), source: 'ch22 D6, ch15 D2' }, // i18n-ignore: function name, verdict identifier, citation
          { label: tr('otherwise', 'ngược lại'), verdict: 'true', route: 'metal-unread', detail: tr('what runs when use_mm declines was not read', 'chưa đọc cái gì chạy khi use_mm từ chối'), source: 'ch22 D6' }, // i18n-ignore: verdict identifier, citation
        ],
        notes: [],
      };
    }
    case 'mlx-cuda': {
      const g = !s.bias && canUseGemv(s.M, s.N, s.aTransposed, s.bTransposed);
      return {
        checks: [
          {
            label: '!bias && can_use_gemv: (M == 1 && b transposed) || (N == 1 && !a transposed)',
            verdict: T(g),
            route: 'mlx-gemv',
            detail: s.bias
              ? tr('a bias: the GEMV branch needs none', 'có bias: nhánh GEMV yêu cầu không có bias')
              : tr(`M == 1: ${s.M === 1}, b transposed: ${s.bTransposed}, N == 1: ${s.N === 1}, a transposed: ${s.aTransposed}`, `M == 1: ${s.M === 1}, b chuyển vị: ${s.bTransposed}, N == 1: ${s.N === 1}, a chuyển vị: ${s.aTransposed}`),
            source: tr('x3 B4, matmul.cpp line 99, gemv.cu line 188', 'x3 B4, matmul.cpp dòng 99, gemv.cu dòng 188'),
          },
          { label: tr('otherwise', 'ngược lại'), verdict: 'true', route: 'cublaslt', detail: '"Invoke cublasLt"', source: tr('x3 B4, matmul.cpp line 115', 'x3 B4, matmul.cpp dòng 115') }, // i18n-ignore: verdict identifier; the detail quotes a source comment
        ],
        notes: [tr('No AMD backend: MLX has no HIP build (x3 B9).', 'Không có backend AMD: MLX không có bản build HIP (x3 B9).')],
      };
    }
    case 'mlx-metal': {
      return {
        checks: [
          {
            label: tr(`the wide GEMV: at most ${MLX_WIDE_MAX} input vectors`, `GEMV rộng: tối đa ${MLX_WIDE_MAX} vector đầu vào`),
            verdict: s.M <= MLX_WIDE_MAX ? 'not-modelled' : 'false',
            route: 'wide',
            detail: s.M <= MLX_WIDE_MAX ? tr(`${s.M} ≤ ${MLX_WIDE_MAX}; the route's other conditions were not read`, `${s.M} ≤ ${MLX_WIDE_MAX}; các điều kiện khác của nhánh này chưa được đọc`) : tr(`${s.M} vectors > ${MLX_WIDE_MAX}`, `${s.M} vector > ${MLX_WIDE_MAX}`),
            source: tr('x3 B4, matmul.cpp line 1620', 'x3 B4, matmul.cpp dòng 1620'),
          },
          { label: 'std::min(M, N) == 1', verdict: T(Math.min(s.M, s.N) === 1), route: 'gemv', detail: `min(${s.M}, ${s.N}) = ${Math.min(s.M, s.N)}`, source: tr('x3 B4, matmul.cpp line 1647', 'x3 B4, matmul.cpp dòng 1647') },
          { label: tr('otherwise', 'ngược lại'), verdict: 'true', route: 'steel', detail: tr('tile shape picked per device class (not evaluated)', 'hình dạng ô được chọn theo lớp device (không đánh giá)'), source: 'ch15 D3' }, // i18n-ignore: verdict identifier, citation
        ],
        notes: [tr('GEMVKernel asserts SM * SN == 32: a SIMD width of 32 written into the code (x3 B4, ch04 D3).', 'GEMVKernel khẳng định SM * SN == 32: độ rộng SIMD 32 được viết cứng vào mã (x3 B4, ch04 D3).')],
      };
    }
    case 'torch-cuda':
      return {
        checks: [
          {
            label: tr('gemm_internal: the preferred BLAS backend', 'gemm_internal: backend BLAS được ưu tiên'),
            verdict: 'true', // i18n-ignore: verdict identifier
            route: 'torch-blas',
            detail: tr('the library picks its own kernel; a gemv entry point exists, with TF32 turned off, but which shapes reach it was not read', 'thư viện tự chọn kernel của nó; có một điểm vào gemv, với TF32 bị tắt, nhưng chưa đọc những hình dạng nào đi tới đó'),
            source: 'ch15 D4, CUDABlas.cpp', // i18n-ignore: citation and file name
          },
        ],
        notes: [tr('On ROCm the same sources are hipified; gemm_internal may prefer Composable Kernel (ch15 D4, x3 B9).', 'Trên ROCm, cùng mã nguồn đó được hipify; gemm_internal có thể ưu tiên Composable Kernel (ch15 D4, x3 B9).')],
      };
    case 'torch-mps':
      return {
        checks: [
          { label: tr('a rank-1 shape: min(M, N) == 1', 'hình dạng hạng 1: min(M, N) == 1'), verdict: T(Math.min(s.M, s.N) === 1), route: 'mps-gemv', detail: `min(${s.M}, ${s.N}) = ${Math.min(s.M, s.N)}`, source: tr('x3 B4, Matmul.mm lines 116 to 141, Linear.mm line 161', 'x3 B4, Matmul.mm dòng 116 đến 141, Linear.mm dòng 161') },
          { label: tr('otherwise', 'ngược lại'), verdict: 'true', route: 'mps-matmul', detail: tr('not read in detail (infra D4)', 'chưa đọc chi tiết (infra D4)'), source: 'x3 B4' }, // i18n-ignore: verdict identifier, citation
        ],
        notes: [
          tr(
            "nn.Linear's own path (Linear.mm) uses MPSNDArrayMatrixMultiplication (lines 68 to 82); its comment says addmm routes rank-1 shapes to the GEMV kernels. Which shapes Linear.mm hands to addmm was not read.",
            'Nhánh riêng của nn.Linear (Linear.mm) dùng MPSNDArrayMatrixMultiplication (dòng 68 đến 82); chú thích của nó nói addmm chuyển các hình dạng hạng 1 tới các kernel GEMV. Chưa đọc Linear.mm giao những hình dạng nào cho addmm.',
          ),
        ],
      };
  }
}

export function route(p: Project, s: Shape): RouteResult {
  const { checks, notes } = checksFor(p, s);
  const at = checks.findIndex((c) => c.verdict !== 'false');
  const c = checks[at];
  const alternatives = checks.slice(at + 1).filter((x) => x.verdict !== 'false').map((x) => x.route);
  return {
    project: p,
    checks,
    at,
    candidate: c.route,
    kind: ROUTES[c.route].kind,
    certain: c.verdict === 'true',
    alternatives: c.verdict === 'true' ? [] : [...new Set(alternatives)],
    notes,
  };
}

/** Projects whose shape rule the facts read (PyTorch CUDA hands every shape to a library). */
export const SHAPE_RULE_READ: Project[] = ['llama-cuda', 'llama-metal', 'mlx-cuda', 'mlx-metal', 'torch-mps'];

/** Switch points written in code, for the strip beside the reader's knee. */
export const CODE_SWITCHES = [
  {
    get label() {
      return tr('llama.cpp\'s MMVF_MAX_BATCH_SIZE', 'MMVF_MAX_BATCH_SIZE của llama.cpp');
    },
    tokens: MMVF_MAX_BATCH_SIZE,
    source: 'ch22 D6', // i18n-ignore: citation
  },
  {
    get label() {
      return tr('MLX Metal\'s wide GEMV limit', 'giới hạn GEMV rộng của MLX Metal');
    },
    tokens: MLX_WIDE_MAX,
    source: 'x3 B4',
  },
];

// ---------------------------------------------------------------- the reader's knee (ch22's definition)
export interface ReaderKnee {
  backend: string;
  device: string;
  /** Series label, e.g. "lab 22.2, M3 (lab 15.6's register-tiled kernel)". */
  label: string;
  /** The kernel's short name, e.g. "M3". */
  short: string;
  /** First measured batch at 1.5 times the smallest batch's step time; null if none in range. */
  knee: number | null;
  /** Largest batch measured. */
  upTo: number;
  /** The measured (batch, seconds per step) points, by batch. */
  points: { batch: number; seconds: number }[];
}

/** Lab 22.2's 'step' rows on every GPU backend, one entry per GEMM kernel, with ch22's knee (sim/batching.ts kneeOf). */
export function readerKnees(reports: ReportsLike): ReaderKnee[] {
  return gpuBackends(reports).flatMap((b) =>
    stepSeries(reports, b)
      .filter((s) => s.points.length >= 2)
      .map((s) => ({
        backend: b,
        device: String(reports[b].device?.name ?? b),
        label: s.label,
        short: s.label.replace(/^lab 22\.2, /, '').replace(/ \(.*$/, ''),
        knee: s.knee,
        upTo: Math.max(...s.points.map((p) => p.batch)),
        points: s.points,
      })),
  );
}
