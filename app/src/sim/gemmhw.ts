// What a GEMM kernel can reach on each target: matrix instructions, asynchronous copies,
// template libraries and the vendor baseline (book §15.9, pp. 368–369, checked against
// notes/today/ch15-facts.md and the files it refers to). Every cell names the facts item
// it rests on; "not documented" means the facts files found nothing to point to, which is
// a statement about the documentation, not a claim that the hardware lacks it.
//
// Fact references: a bare id ("A6") is ch15-facts.md; others name their file
// ("infra C10" = infra-facts.md, "ch06 C7" = ch06-facts.md, "ch03 A7" = ch03-facts.md).

import { tr } from '../lib/locale.ts';

export type Backend = 'CUDA' | 'HIP' | 'Metal';
export type FeatureId = 'warpMma' | 'bigMma' | 'asyncCopy' | 'bulkCopy' | 'templateLib' | 'vendorGemm';
/** yes: documented and available; no: documented as unavailable here; undocumented: nothing usable is documented. */
export type Status = 'yes' | 'no' | 'undocumented';

export const FEATURES: readonly { id: FeatureId; label: string; book: string }[] = [
  { id: 'warpMma', get label() { return tr('Warp-level matrix multiply', 'Phép nhân ma trận cấp warp'); }, book: 'WMMA (p. 369)' }, // i18n-ignore: book citation
  { id: 'bigMma', get label() { return tr('Larger matrix multiply (warp group, tensor memory, or a library operation in the kernel)', 'Phép nhân ma trận lớn hơn (warp group, tensor memory, hoặc một phép toán của thư viện trong kernel)'); }, book: 'WGMMA, TMEM (p. 369)' }, // i18n-ignore: book citation
  { id: 'asyncCopy', get label() { return tr('Asynchronous global-to-shared copy', 'Sao chép bất đồng bộ từ bộ nhớ toàn cục sang bộ nhớ chia sẻ'); }, book: 'LDGSTS (p. 369)' }, // i18n-ignore: book citation
  { id: 'bulkCopy', get label() { return tr('Bulk multi-dimensional tensor copy', 'Sao chép bulk một tensor nhiều chiều'); }, book: 'TMA (p. 369)' }, // i18n-ignore: book citation
  { id: 'templateLib', get label() { return tr('Template GEMM library', 'Thư viện GEMM dạng template'); }, book: 'CUTLASS, cuTile (p. 368)' }, // i18n-ignore: book citation
  { id: 'vendorGemm', get label() { return tr('Vendor GEMM to measure against (FP32)', 'GEMM của hãng để đo so sánh (FP32)'); }, book: 'cuBLAS (p. 368)' }, // i18n-ignore: book citation
];

export interface Cell {
  status: Status;
  /** What is there, or why nothing is. */
  what: string;
  /** Facts items behind the cell. */
  items: string[];
}

export interface Target {
  id: string;
  backend: Backend;
  label: string;
  /** CUDA: compute capability; the generation tests below compare it numerically. */
  cc?: number;
  /** CUDA: built for the architecture-specific "a" target (sm_90a). */
  archSpecific?: boolean;
  /** HIP: gfx generation (9 = CDNA, 11 = RDNA3, 12 = RDNA4: ch15-facts B1, B4) and, on CDNA, its version. */
  gfx?: number;
  cdna?: number;
  /** Metal: compiled for Metal 4. */
  metal4?: boolean;
  /** A word about the target itself. */
  note?: string;
}

export const TARGETS: readonly Target[] = [
  { id: 'cuda-7.0', backend: 'CUDA', get label() { return tr('CUDA, compute capability 7.0 (Volta)', 'CUDA, phiên bản compute capability 7.0 (Volta)'); }, cc: 7.0 },
  { id: 'cuda-7.5', backend: 'CUDA', get label() { return tr('CUDA, compute capability 7.5 (Turing)', 'CUDA, phiên bản compute capability 7.5 (Turing)'); }, cc: 7.5 },
  { id: 'cuda-8.0', backend: 'CUDA', get label() { return tr('CUDA, compute capability 8.0', 'CUDA, phiên bản compute capability 8.0'); }, cc: 8.0 },
  { id: 'cuda-8.6', backend: 'CUDA', get label() { return tr('CUDA, compute capability 8.6 or 8.9 (RTX 30, RTX 40)', 'CUDA, phiên bản compute capability 8.6 hoặc 8.9 (RTX 30, RTX 40)'); }, cc: 8.6 },
  { id: 'cuda-9.0', backend: 'CUDA', get label() { return tr('CUDA, compute capability 9.0, built as sm_90', 'CUDA, phiên bản compute capability 9.0, biên dịch dạng sm_90'); }, cc: 9.0 },
  { id: 'cuda-9.0a', backend: 'CUDA', get label() { return tr('CUDA, compute capability 9.0, built as sm_90a', 'CUDA, phiên bản compute capability 9.0, biên dịch dạng sm_90a'); }, cc: 9.0, archSpecific: true },
  { id: 'cuda-10.0', backend: 'CUDA', get label() { return tr('CUDA, compute capability 10.0 (data-centre Blackwell)', 'CUDA, phiên bản compute capability 10.0 (Blackwell cho trung tâm dữ liệu)'); }, cc: 10.0 },
  { id: 'cuda-12', backend: 'CUDA', get label() { return tr('CUDA, compute capability 12.x (RTX 50)', 'CUDA, phiên bản compute capability 12.x (RTX 50)'); }, cc: 12.0 },
  { id: 'hip-cdna2', backend: 'HIP', label: 'HIP, CDNA 2', gfx: 9, cdna: 2 }, // i18n-ignore: product and target names
  { id: 'hip-cdna3', backend: 'HIP', label: 'HIP, CDNA 3', gfx: 9, cdna: 3 }, // i18n-ignore: product and target names
  { id: 'hip-cdna4', backend: 'HIP', label: 'HIP, CDNA 4 (gfx950)', gfx: 9, cdna: 4 }, // i18n-ignore: product and target names
  { id: 'hip-rdna3', backend: 'HIP', label: 'HIP, RDNA 3 (gfx11)', gfx: 11 }, // i18n-ignore: product and target names
  { id: 'hip-rdna4', backend: 'HIP', label: 'HIP, RDNA 4 (gfx12)', gfx: 12 }, // i18n-ignore: product and target names
  { id: 'metal-apple7', backend: 'Metal', get label() { return tr('Metal, Apple7 or later, without Metal 4', 'Metal, Apple7 trở lên, không có Metal 4'); }, metal4: false },
  { id: 'metal-apple7-m4', backend: 'Metal', get label() { return tr('Metal, Apple7 or later, with Metal 4', 'Metal, Apple7 trở lên, có Metal 4'); }, metal4: true },
];

export const target = (id: string): Target => {
  const t = TARGETS.find((x) => x.id === id);
  if (!t) throw new Error('unknown target ' + id);
  return t;
};

// ---------------------------------------------------------------- the rules, one per fact
/** nvcc 13.5 lists real architectures from sm_75 up; sm_70 is absent (A5). */
export const cuda13Targets = (cc: number) => cc >= 7.5;
/** nvcuda::wmma: compute capability 7.0 or higher (infra A12). PTX mma: "Requires sm_70 or higher" (A6). */
export const hasWmma = (cc: number) => cc >= 7.0;
/** PTX mma .f16 m16n8k16 and the .tf32 shapes require sm_80 (A6). */
export const hasMmaTf32 = (cc: number) => cc >= 8.0;
/** wgmma "Requires sm_90a"; "a" targets cannot run on later generations (A6). */
export const hasWgmma = (cc: number, archSpecific: boolean) => cc === 9.0 && archSpecific;
/** tcgen05 (with TMEM): sm_100 and sm_110 families only; sm_120 is not in the list (A6). */
export const hasTcgen05 = (cc: number) => (cc >= 10.0 && cc < 11.0) || (cc >= 11.0 && cc < 12.0);
/** LDGSTS / cp.async: CC 8.0+ (A7). */
export const hasLdgsts = (cc: number) => cc >= 8.0;
/** TMA: CC 9.0+ (A8). */
export const hasTma = (cc: number) => cc >= 9.0;
/** BF16x9 emulation in cuBLAS: compute capability 10.0 and 10.3 only (A10). */
export const hasBf16x9 = (cc: number) => cc === 10.0 || cc === 10.3;

/** MFMA builtins: the "mai-insts" feature of the CDNA (gfx9 MI) sets (B4). */
export const hasMfma = (t: Target) => t.backend === 'HIP' && t.gfx === 9;
/** The xf32 MFMA form: llama.cpp uses it on CDNA3; "CDNA4 (gfx950) does not support xf32 MFMA" (B4, D1). */
export const hasXf32Mfma = (t: Target) => t.backend === 'HIP' && t.cdna === 3;
/** WMMA builtins: wmma-256b-insts in the gfx11 sets, wmma-128b-insts in gfx12 (B4). */
export const hasRdnaWmma = (t: Target) => t.backend === 'HIP' && (t.gfx === 11 || t.gfx === 12);
/** global_load_lds: "vmem-to-lds-load-insts", in the GFX9 and GFX10 generation sets, not GFX11/12 (B5). */
export const hasGlobalLoadLds = (t: Target) => t.backend === 'HIP' && (t.gfx === 9 || t.gfx === 10);
/** The data movement engine is described for CDNA 3 and 4 (B5). */
export const hasDme = (t: Target) => t.backend === 'HIP' && (t.cdna === 3 || t.cdna === 4);

// ---------------------------------------------------------------- the table
function cuda(t: Target, f: FeatureId): Cell {
  const cc = t.cc!;
  switch (f) {
    case 'warpMma':
      return {
        status: 'yes',
        what: hasMmaTf32(cc)
          ? tr('nvcuda::wmma in <mma.h>, still documented (not "pre-Hopper only"); PTX mma.sync, including the f16 m16n8k16 and tf32 shapes.', 'nvcuda::wmma trong <mma.h>, vẫn có trong tài liệu (không phải "chỉ trước Hopper"); PTX mma.sync, gồm cả các hình dạng f16 m16n8k16 và tf32.')
          : tr('nvcuda::wmma in <mma.h>; PTX mma.sync. The f16 m16n8k16 and tf32 shapes need sm_80.', 'nvcuda::wmma trong <mma.h>; PTX mma.sync. Các hình dạng f16 m16n8k16 và tf32 cần sm_80.'),
        items: ['A6', 'infra A12'], // i18n-ignore: facts-file item ids
      };
    case 'bigMma':
      if (hasWgmma(cc, !!t.archSpecific))
        return {
          status: 'yes',
          what: tr('WGMMA (wgmma.mma_async): four warps; A from registers or shared memory, B from shared memory; issue, commit and wait are separate. Code built for sm_90a cannot run on later generations.', 'WGMMA (wgmma.mma_async): bốn warp; A lấy từ thanh ghi hoặc bộ nhớ chia sẻ, B lấy từ bộ nhớ chia sẻ; issue, commit và wait là các bước riêng. Mã biên dịch cho sm_90a không chạy được trên các thế hệ sau.'),
          items: ['A6'],
        };
      if (hasTcgen05(cc))
        return {
          status: 'yes',
          what: tr('tcgen05.mma with tensor memory (TMEM): the accumulator is written to TMEM, out of the register file. WGMMA is not available: it is sm_90a only, and "a" code does not run here.', 'tcgen05.mma với tensor memory (TMEM): bộ tích lũy (accumulator) được ghi vào TMEM, nằm ngoài tệp thanh ghi. Không có WGMMA: nó chỉ dành cho sm_90a, và mã "a" không chạy ở đây.'),
          items: ['A6'],
        };
      if (cc === 9.0)
        return { status: 'no', what: tr('WGMMA "Requires sm_90a": built as plain sm_90 the kernel cannot use it.', 'WGMMA "Requires sm_90a": khi biên dịch dạng sm_90 thường, kernel không dùng được nó.'), items: ['A6'] };
      if (cc >= 12.0)
        return {
          status: 'no',
          what: tr('No TMEM: tcgen05 targets are the sm_100 and sm_110 families, and sm_120 is not in the list. No WGMMA either (sm_90a only).', 'Không có TMEM: các target của tcgen05 là họ sm_100 và sm_110, và sm_120 không có trong danh sách. Cũng không có WGMMA (chỉ sm_90a).'),
          items: ['A6'],
        };
      return { status: 'no', what: tr('WGMMA needs sm_90a and tcgen05 the sm_100 or sm_110 families.', 'WGMMA cần sm_90a, còn tcgen05 cần họ sm_100 hoặc sm_110.'), items: ['A6'] };
    case 'asyncCopy':
      return hasLdgsts(cc)
        ? {
            status: 'yes',
            what: tr('LDGSTS / cp.async (CC 8.0+). cuda::memcpy_async does not guarantee it; the __pipeline_memcpy_async primitives do. Check the SASS or Nsight Compute\'s memory chart.', 'LDGSTS / cp.async (CC 8.0+). cuda::memcpy_async không bảo đảm dùng nó; các primitive __pipeline_memcpy_async thì bảo đảm. Hãy kiểm tra SASS hoặc biểu đồ bộ nhớ của Nsight Compute.'),
            items: ['A7'],
          }
        : { status: 'no', what: tr('LDGSTS needs compute capability 8.0 (cp.async "Requires sm_80 or higher").', 'LDGSTS cần compute capability 8.0 (cp.async "Requires sm_80 or higher").'), items: ['A7'] };
    case 'bulkCopy':
      return hasTma(cc)
        ? {
            status: 'yes',
            what: tr('TMA (CC 9.0+). A 2D tile needs a tensor map built on the host (cuTensorMapEncodeTiled) and cuda::ptx::cp_async_bulk_tensor; cuda::memcpy_async reaches TMA only for 1D bulk copies.', 'TMA (CC 9.0+). Một ô 2D cần một tensor map dựng trên host (cuTensorMapEncodeTiled) và cuda::ptx::cp_async_bulk_tensor; cuda::memcpy_async chỉ dùng tới TMA cho các lần sao chép bulk 1D.'),
            items: ['A8'],
          }
        : { status: 'no', what: tr('TMA needs compute capability 9.0.', 'TMA cần compute capability 9.0.'), items: ['A8'] };
    case 'templateLib':
      return {
        status: 'yes',
        what: tr('CUTLASS; CUDA Tile (cuTile Python, and CUDA Tile C++ from toolkit 13.3). Which generations each supports is not in our facts files.', 'CUTLASS; CUDA Tile (cuTile Python, và CUDA Tile C++ từ toolkit 13.3). Các tệp facts của chúng tôi không ghi mỗi thư viện hỗ trợ những thế hệ nào.'),
        items: ['A9', 'infra A12'], // i18n-ignore: facts-file item ids
      };
    case 'vendorGemm':
      return {
        status: 'yes',
        what:
          tr('cublasSgemm (cublas_v2.h, CUDA::cublas), column-major. The default math mode keeps FP32; TF32 is opt-in', 'cublasSgemm (cublas_v2.h, CUDA::cublas), thứ tự cột (column-major). Chế độ toán mặc định giữ FP32; TF32 phải chủ động bật') +
          (hasBf16x9(cc) ? tr(', and so is BF16x9 emulation, which exists on this compute capability.', ', và giả lập BF16x9 cũng vậy, vốn có trên compute capability này.') : '.'),
        items: ['A10', 'ch03 A7'], // i18n-ignore: facts-file item ids
      };
  }
}

function hip(t: Target, f: FeatureId): Cell {
  switch (f) {
    case 'warpMma':
      if (hasMfma(t))
        return {
          status: 'yes',
          what:
            tr('MFMA builtins, including FP32 in and out (__builtin_amdgcn_mfma_f32_16x16x4f32); rocWMMA as the C++ route. ', 'Các builtin MFMA, gồm cả FP32 vào và ra (__builtin_amdgcn_mfma_f32_16x16x4f32); rocWMMA là con đường C++. ') +
            (hasXf32Mfma(t)
              ? tr('The reduced-precision xf32 form exists here (llama.cpp uses it on CDNA3).', 'Dạng xf32 độ chính xác giảm có ở đây (llama.cpp dùng nó trên CDNA3).')
              : t.cdna === 4
                ? tr('No xf32 form: "CDNA4 (gfx950) does not support xf32 MFMA".', 'Không có dạng xf32: "CDNA4 (gfx950) does not support xf32 MFMA".')
                : tr('llama.cpp takes the plain f32 path here, not xf32.', 'llama.cpp đi đường f32 thường ở đây, không phải xf32.')),
          items: ['B4', 'D1', 'infra B12'], // i18n-ignore: facts-file item ids
        };
      return {
        status: 'yes',
        what:
          (t.gfx === 12 ? tr('__builtin_amdgcn_wmma_*_w32_gfx12 builtins (wmma-128b-insts).', 'Các builtin __builtin_amdgcn_wmma_*_w32_gfx12 (wmma-128b-insts).') : tr('__builtin_amdgcn_wmma_*_w32 builtins (wmma-256b-insts).', 'Các builtin __builtin_amdgcn_wmma_*_w32 (wmma-256b-insts).')) +
          tr(' The HIP features table shows no Matrix Cores on RDNA, but the ROCm 10.2 compiler has these. No MFMA.', ' Bảng tính năng của HIP không ghi Matrix Cores trên RDNA, nhưng trình biên dịch ROCm 10.2 có các builtin này. Không có MFMA.'),
        items: ['B4'],
      };
    case 'bigMma':
      return { status: 'undocumented', what: tr('Nothing beyond MFMA or WMMA is documented.', 'Ngoài MFMA hoặc WMMA, tài liệu không ghi gì thêm.'), items: ['ch15 table'] }; // i18n-ignore: facts-file item ids
    case 'asyncCopy':
      return hasGlobalLoadLds(t)
        ? {
            status: 'undocumented',
            what: tr('__builtin_amdgcn_global_load_lds exists in the compiler (GFX9 generation) but not in the HIP guide, so it is not taught. The documented name, __builtin_amdgcn_async_work_group_copy, does not exist in ROCm 10.2.', '__builtin_amdgcn_global_load_lds có trong trình biên dịch (thế hệ GFX9) nhưng không có trong hướng dẫn HIP, nên khóa học không dạy nó. Tên có trong tài liệu, __builtin_amdgcn_async_work_group_copy, không tồn tại trong ROCm 10.2.'),
            items: ['B5'],
          }
        : {
            status: 'undocumented',
            what: tr('None documented. The compiler\'s global-to-LDS load feature is in the GFX9 and GFX10 sets, not GFX11 or GFX12.', 'Tài liệu không ghi cách nào. Tính năng nạp từ global sang LDS của trình biên dịch nằm trong các tập GFX9 và GFX10, không có ở GFX11 hay GFX12.'),
            items: ['B5'],
          };
    case 'bulkCopy':
      return hasDme(t)
        ? {
            status: 'undocumented',
            what: tr('AMD describes a data movement engine on CDNA 3 and 4, but the intrinsic it names does not exist in ROCm 10.2: no usable documented API.', 'AMD mô tả một data movement engine trên CDNA 3 và 4, nhưng intrinsic mà AMD nêu tên không tồn tại trong ROCm 10.2: không có API nào trong tài liệu dùng được.'),
            items: ['B5'],
          }
        : { status: 'undocumented', what: tr('None documented.', 'Tài liệu không ghi cách nào.'), items: ['B5'] };
    case 'templateLib':
      return {
        status: 'yes',
        what:
          tr('Composable Kernel (with CK Tile); rocWMMA.', 'Composable Kernel (kèm CK Tile); rocWMMA.') +
          (t.gfx === 11 || t.gfx === 12 ? tr(' PyTorch\'s code notes "gfx11 and gfx12 have no CK fp32 GEMM".', ' Mã của PyTorch có ghi chú "gfx11 and gfx12 have no CK fp32 GEMM".') : ''),
        items: ['B7', 'infra B12', 'D4'], // i18n-ignore: facts-file item ids
      };
    case 'vendorGemm':
      return {
        status: 'yes',
        what: tr('hipblasSgemm (roc::hipblas): the same call shape as cublasSgemm, column-major, FP32 by default; XF32 is opt-in.', 'hipblasSgemm (roc::hipblas): cùng dạng lời gọi như cublasSgemm, thứ tự cột (column-major), mặc định FP32; XF32 phải chủ động bật.'),
        items: ['B6', 'ch03 B6'], // i18n-ignore: facts-file item ids
      };
  }
}

function metal(t: Target, f: FeatureId): Cell {
  switch (f) {
    case 'warpMma':
      return {
        status: 'yes',
        what: tr('simdgroup_float8x8 with simdgroup_multiply_accumulate (8 x 8 only, Apple7 and later). Apple documents no matrix unit behind it: not "tensor cores".', 'simdgroup_float8x8 với simdgroup_multiply_accumulate (chỉ 8 x 8, Apple7 trở lên). Tài liệu của Apple không nêu đơn vị ma trận nào đứng sau nó: không phải "tensor cores".'),
        items: ['C5', 'infra C10'], // i18n-ignore: facts-file item ids
      };
    case 'bigMma':
      return t.metal4
        ? {
            status: 'yes',
            what: tr('Metal Performance Primitives matmul2d over one or more SIMD-groups, float x float -> float from Metal 4; half x int4 formats from Metal 4 with OS 26.4. Apple recommends it over simdgroup_matrix.', 'Metal Performance Primitives matmul2d trên một hoặc nhiều SIMD-group, float x float -> float từ Metal 4; các định dạng half x int4 từ Metal 4 với OS 26.4. Apple khuyên dùng nó thay cho simdgroup_matrix.'),
            items: ['C5', 'infra C10'], // i18n-ignore: facts-file item ids
          }
        : { status: 'no', what: tr('MPP matmul2d needs Metal 4.', 'MPP matmul2d cần Metal 4.'), items: ['infra C10'] }; // i18n-ignore: facts-file item ids
    case 'asyncCopy':
      return { status: 'undocumented', what: tr('None documented: the MSL 4.1 specification has no "async".', 'Tài liệu không ghi cách nào: đặc tả MSL 4.1 không có "async".'), items: ['ch06 C7'] }; // i18n-ignore: facts-file item ids
    case 'bulkCopy':
      return { status: 'undocumented', what: tr('None documented.', 'Tài liệu không ghi cách nào.'), items: ['ch15 table'] }; // i18n-ignore: facts-file item ids
    case 'templateLib':
      return { status: 'undocumented', what: tr('No template GEMM library; MPP is the nearest thing.', 'Không có thư viện GEMM dạng template; MPP là thứ gần nhất.'), items: ['ch15 table'] }; // i18n-ignore: facts-file item ids
    case 'vendorGemm':
      return {
        status: 'yes',
        what: tr('MPSMatrixMultiplication (-framework MetalPerformanceShaders), row-major, with transpose flags. Uses higher internal precision by default; reduced precision is an option you set.', 'MPSMatrixMultiplication (-framework MetalPerformanceShaders), thứ tự hàng (row-major), có cờ chuyển vị. Mặc định dùng độ chính xác nội bộ cao hơn; độ chính xác giảm là một tùy chọn bạn tự đặt.'),
        items: ['C6', 'ch03 C7'], // i18n-ignore: facts-file item ids
      };
  }
}

export function cell(t: Target, f: FeatureId): Cell {
  return t.backend === 'CUDA' ? cuda(t, f) : t.backend === 'HIP' ? hip(t, f) : metal(t, f);
}

export interface TargetSummary {
  target: Target;
  /** CUDA only: whether today's toolkit (CUDA 13) compiles for it at all. */
  toolkit?: Cell;
  cells: Record<FeatureId, Cell>;
}

export function summarize(id: string): TargetSummary {
  const t = target(id);
  const cells = Object.fromEntries(FEATURES.map((f) => [f.id, cell(t, f.id)])) as Record<FeatureId, Cell>;
  const toolkit =
    t.backend === 'CUDA'
      ? cuda13Targets(t.cc!)
        ? { status: 'yes' as const, what: tr('A CUDA 13 target (nvcc 13.5 lists sm_75 and later).', 'Một target của CUDA 13 (nvcc 13.5 liệt kê sm_75 trở lên).'), items: ['A5'] }
        : {
            status: 'no' as const,
            what: tr('Tensor cores date from Volta, but nvcc 13.5 does not list sm_70 and NVIDIA\'s libraries dropped pre-Turing GPUs. (We found no single toolkit-wide sentence saying so.)', 'Tensor core có từ Volta, nhưng nvcc 13.5 không liệt kê sm_70 và các thư viện của NVIDIA đã bỏ các GPU trước Turing. (Chúng tôi không tìm thấy một câu nào áp dụng cho cả toolkit nói điều này.)'),
            items: ['A5'],
          }
      : undefined;
  return { target: t, toolkit, cells };
}
