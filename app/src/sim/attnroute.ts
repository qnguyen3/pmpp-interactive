// Which attention kernel each project runs (module X3, section X3.5). Not in the book. Every
// check is one read in notes/today/x3-facts.md (B5, B9) or ch20-facts.md (B5, D1, D2, D4, D5),
// at the commits listed there.
//
// What it models:
//   - MLX CUDA: cuDNN's SDPA when supports_sdpa_cudnn accepts the call; for one query that
//     needs no array mask and at least 256 keys (use_cudnn_for_decoding, kv_cache_step = 256,
//     "number is from mlx-lm"); otherwise its own sdpa_vector kernels for one query.
//   - MLX Metal: vector kernels for 8 or fewer queries; among them the 2-pass kernel when the
//     device class is 'd' or 's' with at least 1024 keys, or with GQA from 4096 keys; tiled
//     steel attention for more queries.
//   - PyTorch CUDA: select_sdp_backend walks a priority order (default flash, efficient, math,
//     cuDNN, overrideable; cuDNN first when it is preferred) and takes the first backend whose
//     use_<backend> check passes.
//   - llama.cpp: its own kernels on every backend.
//
// What it leaves out: the real conditions of supports_sdpa_cudnn and of PyTorch's use_<backend>
// checks (not read: the reader answers them), how MLX CUDA picks its 1-pass or 2-pass vector
// kernel, what MLX CUDA does for several queries when cuDNN declines, which llama.cpp variant
// (vector, tile, MMA) runs for a shape, the MPS kernels' thresholds, head sizes and dtypes.

import { tr } from '../lib/locale.ts';

export type AttnProject = 'llama' | 'mlx-cuda' | 'mlx-metal' | 'torch-cuda' | 'torch-rocm' | 'torch-mps';
export const ATTN_PROJECTS: AttnProject[] = ['llama', 'mlx-cuda', 'mlx-metal', 'torch-cuda', 'torch-rocm', 'torch-mps'];
export const ATTN_LABEL: Record<AttnProject, string> = {
  llama: 'llama.cpp (CUDA, HIP, Metal)', // i18n-ignore: product and backend names
  'mlx-cuda': 'MLX, CUDA', // i18n-ignore: product and backend names
  'mlx-metal': 'MLX, Metal', // i18n-ignore: product and backend names
  'torch-cuda': 'PyTorch, CUDA', // i18n-ignore: product and backend names
  'torch-rocm': 'PyTorch, ROCm', // i18n-ignore: product and backend names
  'torch-mps': 'PyTorch, MPS', // i18n-ignore: product and backend names
};

export type Mask = 'none' | 'causal' | 'array';
export type TorchBackend = 'flash' | 'efficient' | 'math' | 'cudnn' | 'overrideable';
export const TORCH_BACKEND_LABEL: Record<TorchBackend, string> = {
  flash: 'flash_attention',
  efficient: 'efficient_attention',
  math: 'math',
  cudnn: 'cudnn_attention',
  overrideable: 'overrideable',
};

export interface AttnInput {
  /** Queries per sequence (q.shape(2) in MLX). */
  queries: number;
  keys: number;
  mask: Mask;
  /** Fewer K/V heads than query heads. */
  gqa: boolean;
  /** MLX Metal: the device class letter is 'd' or 's' (devc == 'd' || devc == 's'). */
  deviceDS: boolean;
  /** MLX CUDA: the reader's answer to supports_sdpa_cudnn for the other conditions (not read). */
  cudnnSupports: boolean;
  /** PyTorch CUDA: check_prefer_cudnn_attention(). */
  preferCudnn: boolean;
  /** PyTorch CUDA: the reader's answers to each use_<backend> check (not read). */
  use: Record<TorchBackend, boolean>;
}

export const DEFAULT_INPUT: AttnInput = {
  queries: 1,
  keys: 1000,
  mask: 'none',
  gqa: false,
  deviceDS: false,
  cudnnSupports: true,
  preferCudnn: false,
  use: { flash: true, efficient: true, math: true, cudnn: true, overrideable: false },
};

/** Context.h lines 509 to 514. */
export const SDP_PRIORITY_ORDER: TorchBackend[] = ['flash', 'efficient', 'math', 'cudnn', 'overrideable'];
/** use_cudnn_for_decoding: "constexpr int kv_cache_step = 256; // number is from mlx-lm" */
export const KV_CACHE_STEP = 256;
/** MLX Metal: q_pre.shape(2) <= 8 selects the vector kernels. */
export const MLX_VECTOR_MAX_QUERIES = 8;
/** MLX Metal 2-pass: n_keys >= 1024 on a 'd' or 's' device, or GQA with n_keys >= 4096. */
export const MLX_2PASS_KEYS = 1024;
export const MLX_2PASS_GQA_KEYS = 4096;

export type Verdict = 'true' | 'false' | 'not-modelled' | 'reader';
export interface AttnCheck {
  label: string;
  verdict: Verdict;
  detail: string;
  source: string;
}
export type Home = 'own' | 'library' | 'submodule' | 'unknown';
export interface AttnKernel {
  name: string;
  where: string;
  home: Home;
}
export interface AttnResult {
  project: AttnProject;
  checks: AttnCheck[];
  kernel: AttnKernel;
  /** PyTorch CUDA: the order walked. */
  order?: TorchBackend[];
  notes: string[];
}

const T = (b: boolean): Verdict => (b ? 'true' : 'false');

export const TORCH_KERNEL: Record<TorchBackend, AttnKernel> = {
  flash: {
    get name() {
      return tr('FlashAttention-2 (run_mha_fwd_ or run_mha_fwd_splitkv_dispatch)', 'FlashAttention-2 (run_mha_fwd_ hoặc run_mha_fwd_splitkv_dispatch)');
    },
    get where() {
      return tr(
        'third_party/flash-attention, a submodule pinned at bb25ba3; PyTorch holds the shim flash_attn/flash_api.cpp',
        'third_party/flash-attention, một submodule được ghim ở bb25ba3; PyTorch giữ lớp shim flash_attn/flash_api.cpp',
      );
    },
    home: 'submodule',
  },
  efficient: {
    get name() {
      return tr('memory-efficient attention', 'attention tiết kiệm bộ nhớ (memory-efficient)');
    },
    where: 'aten/src/ATen/native/transformers/cuda/mem_eff_attention/',
    home: 'own',
  },
  math: {
    get name() {
      return tr('the math path: "a PyTorch implementation defined in C++" (the docs)', 'đường math: "một bản cài đặt PyTorch được định nghĩa bằng C++" (theo tài liệu)');
    },
    get where() {
      return tr('in PyTorch itself; its file was not read', 'ngay trong PyTorch; tệp của nó chưa được đọc');
    },
    home: 'own',
  },
  cudnn: {
    get name() {
      return tr("cuDNN's attention", 'attention của cuDNN');
    },
    get where() {
      return tr('the cuDNN library (no source)', 'thư viện cuDNN (không có mã nguồn)');
    },
    home: 'library',
  },
  overrideable: {
    get name() {
      return tr('an overrideable backend', 'một backend overrideable');
    },
    get where() {
      return tr('registered from outside (not read)', 'được đăng ký từ bên ngoài (chưa đọc)');
    },
    home: 'unknown',
  },
};

export function attnRoute(p: AttnProject, x: AttnInput): AttnResult {
  switch (p) {
    case 'llama':
      return {
        project: p,
        checks: [
          {
            label: tr('which variant for this shape', 'biến thể nào cho shape này'),
            verdict: 'not-modelled', // i18n-ignore: verdict id, the page shows its own label
            detail: tr('vector, tile or MMA kernels: the selection was not read', 'kernel vector, tile hoặc MMA: phần lựa chọn chưa được đọc'),
            source: 'ch20 D1, D2', // i18n-ignore: citation
          },
        ],
        kernel: {
          name: tr(
            'its own flash-attention kernels: fattn-vec (online softmax, split-KV combine, GQA ratio) and the other fattn-* variants; Metal fa_common and fa_aux',
            'các kernel flash-attention của riêng nó: fattn-vec (online softmax, kết hợp split-KV, tỉ lệ GQA) và các biến thể fattn-* khác; Metal fa_common và fa_aux',
          ),
          where: tr(
            'ggml/src/ggml-cuda/fattn-*.cu (also the HIP build); ggml/src/ggml-metal/kernels/fa_*.metal',
            'ggml/src/ggml-cuda/fattn-*.cu (cả bản build HIP); ggml/src/ggml-metal/kernels/fa_*.metal',
          ),
          home: 'own',
        },
        notes: [tr('The HIP backend compiles the same fattn files (x3 B9).', 'Backend HIP biên dịch chính các tệp fattn đó (x3 B9).')],
      };
    case 'mlx-cuda': {
      const one = x.queries === 1;
      const decodeOk = !one || (x.mask !== 'array' && x.keys >= KV_CACHE_STEP);
      const cudnn = x.cudnnSupports && decodeOk;
      const checks: AttnCheck[] = [
        {
          label: tr('supports_sdpa_cudnn (its other conditions)', 'supports_sdpa_cudnn (các điều kiện khác của nó)'),
          verdict: 'reader', // i18n-ignore: verdict id, the page shows its own label
          detail: x.cudnnSupports
            ? tr('you said yes; the real checks were not read', 'bạn trả lời có; các kiểm tra thật chưa được đọc')
            : tr('you said no', 'bạn trả lời không'),
          source: tr('x3 B5, scaled_dot_product_attention.cpp 650 to 654', 'x3 B5, scaled_dot_product_attention.cpp 650 đến 654'),
        },
      ];
      if (one)
        checks.push({
          label: tr(
            `one query: no array mask and at least ${KV_CACHE_STEP} keys (use_cudnn_for_decoding)`,
            `một query: không có mask dạng mảng và ít nhất ${KV_CACHE_STEP} key (use_cudnn_for_decoding)`,
          ),
          verdict: T(decodeOk),
          detail: tr(
            `mask: ${x.mask}, keys: ${x.keys}. "for small sequence the overhead would kill the advantage"; 256 is mlx-lm's cache step`,
            `mask: ${x.mask}, số key: ${x.keys}. "for small sequence the overhead would kill the advantage" (với chuỗi ngắn, chi phí phụ sẽ triệt tiêu lợi thế); 256 là bước cache của mlx-lm`,
          ),
          source: tr('x3 B5, lines 72 to 88', 'x3 B5, dòng 72 đến 88'),
        });
      if (cudnn)
        return {
          project: p,
          checks,
          kernel: {
            name: tr("cuDNN's SDPA, built with the cuDNN frontend graph API (sdpa_cudnn)", 'SDPA của cuDNN, dựng bằng graph API của cuDNN frontend (sdpa_cudnn)'),
            where: tr(
              'the cuDNN library (no source); MLX holds the graph in scaled_dot_product_attention.cpp',
              'thư viện cuDNN (không có mã nguồn); MLX giữ graph trong scaled_dot_product_attention.cpp',
            ),
            home: 'library',
          },
          notes: [tr('Which cuDNN version MLX needs was not verified (x3 B5).', 'Phiên bản cuDNN mà MLX cần chưa được kiểm chứng (x3 B5).')],
        };
      if (one)
        return {
          project: p,
          checks: [
            ...checks,
            {
              label: tr('1-pass or 2-pass', '1-pass hoặc 2-pass'),
              verdict: 'not-modelled', // i18n-ignore: verdict id, the page shows its own label
              detail: tr('kernel_sdpav_1pass or kernel_sdpav_2pass_1: the choice was not read', 'kernel_sdpav_1pass hoặc kernel_sdpav_2pass_1: phần lựa chọn chưa được đọc'),
              source: 'x3 B5', // i18n-ignore: citation
            },
          ],
          kernel: {
            name: tr(
              'its own sdpa_vector kernels (kernel_sdpav_1pass, kernel_sdpav_2pass_1), online softmax in base 2',
              'các kernel sdpa_vector của riêng nó (kernel_sdpav_1pass, kernel_sdpav_2pass_1), online softmax cơ số 2',
            ),
            where: 'mlx/backend/cuda/scaled_dot_product_attention.cu',
            home: 'own',
          },
          notes: [],
        };
      return {
        project: p,
        checks,
        kernel: {
          name: tr(
            'not read: the facts give no MLX CUDA kernel for several queries when cuDNN declines',
            'chưa đọc: các dữ kiện không nêu kernel MLX CUDA nào cho nhiều query khi cuDNN từ chối',
          ),
          where: 'mlx/backend/cuda/scaled_dot_product_attention.cpp',
          home: 'unknown',
        },
        notes: [],
      };
    }
    case 'mlx-metal': {
      const vector = x.queries <= MLX_VECTOR_MAX_QUERIES;
      const checks: AttnCheck[] = [
        {
          label: tr(`q_pre.shape(2) <= ${MLX_VECTOR_MAX_QUERIES} (queries)`, `q_pre.shape(2) <= ${MLX_VECTOR_MAX_QUERIES} (số query)`),
          verdict: T(vector),
          detail: tr(`${x.queries} queries`, `${x.queries} query`),
          source: tr('ch20 D4, line 1820', 'ch20 D4, dòng 1820'),
        },
      ];
      if (!vector)
        return {
          project: p,
          checks,
          kernel: {
            name: tr('tiled steel attention (sdpa_full_self_attention_metal)', 'steel attention phân ô (sdpa_full_self_attention_metal)'),
            where: tr(
              'called from mlx/backend/metal/scaled_dot_product_attention.cpp, line 1929',
              'được gọi từ mlx/backend/metal/scaled_dot_product_attention.cpp, dòng 1929',
            ),
            home: 'own',
          },
          notes: [],
        };
      const big = x.deviceDS && x.keys >= MLX_2PASS_KEYS;
      const gqaLong = x.gqa && x.keys >= MLX_2PASS_GQA_KEYS;
      checks.push(
        {
          label: tr(`device class 'd' or 's' and n_keys >= ${MLX_2PASS_KEYS}`, `lớp thiết bị 'd' hoặc 's' và n_keys >= ${MLX_2PASS_KEYS}`),
          verdict: T(big),
          detail: tr(`devc d or s: ${x.deviceDS}, keys: ${x.keys}`, `devc d hoặc s: ${x.deviceDS}, số key: ${x.keys}`),
          source: tr('ch20 D4, lines 1885 to 1896', 'ch20 D4, dòng 1885 đến 1896'),
        },
        {
          label: tr(`k.shape(1) < q.shape(1) (GQA) and n_keys >= ${MLX_2PASS_GQA_KEYS}`, `k.shape(1) < q.shape(1) (GQA) và n_keys >= ${MLX_2PASS_GQA_KEYS}`),
          verdict: T(gqaLong),
          detail: tr(`GQA: ${x.gqa}, keys: ${x.keys}`, `GQA: ${x.gqa}, số key: ${x.keys}`),
          source: tr('ch20 D4, lines 1885 to 1896', 'ch20 D4, dòng 1885 đến 1896'),
        },
      );
      return {
        project: p,
        checks,
        kernel: big || gqaLong
          ? {
              name: tr('sdpa_vector_2pass (split over keys, then merged)', 'sdpa_vector_2pass (chia theo key, rồi ghép lại)'),
              where: tr(
                'called from mlx/backend/metal/scaled_dot_product_attention.cpp, lines 1885 to 1896',
                'được gọi từ mlx/backend/metal/scaled_dot_product_attention.cpp, dòng 1885 đến 1896',
              ),
              home: 'own',
            }
          : { name: tr('sdpa_vector (one pass, fast::exp)', 'sdpa_vector (một lượt, fast::exp)'), where: 'mlx/backend/metal/kernels/sdpa_vector.h', home: 'own' },
        notes: [
          tr(
            'Which Apple chips report which device class letter is not documented in the file (ch15 D3).',
            'Chip Apple nào báo chữ cái lớp thiết bị nào thì tệp không ghi lại (ch15 D3).',
          ),
        ],
      };
    }
    case 'torch-cuda': {
      const order: TorchBackend[] = x.preferCudnn ? ['cudnn', ...SDP_PRIORITY_ORDER.filter((b) => b !== 'cudnn')] : SDP_PRIORITY_ORDER.slice();
      const checks: AttnCheck[] = [];
      let chosen: TorchBackend | null = null;
      for (const b of order) {
        if (chosen) {
          checks.push({
            label: `use_${TORCH_BACKEND_LABEL[b]}`, // i18n-ignore: function name
            verdict: 'not-modelled', // i18n-ignore: verdict id, the page shows its own label
            detail: tr('not reached', 'chưa tới lượt'),
            source: 'x3 B5', // i18n-ignore: citation
          });
          continue;
        }
        checks.push({
          label: `use_${TORCH_BACKEND_LABEL[b]}`, // i18n-ignore: function name
          verdict: 'reader', // i18n-ignore: verdict id, the page shows its own label
          detail: x.use[b] ? tr('you said it passes', 'bạn trả lời là đạt') : tr('you said it fails', 'bạn trả lời là không đạt'),
          source: tr('x3 B5, sdp_utils.cpp 48 to 58 (Note [SDPA Runtime Dispatch])', 'x3 B5, sdp_utils.cpp 48 đến 58 (Note [SDPA Runtime Dispatch])'),
        });
        if (x.use[b]) chosen = b;
      }
      return {
        project: p,
        checks,
        order,
        kernel: chosen ? TORCH_KERNEL[chosen] : { name: tr('no backend accepts the call', 'không backend nào nhận lời gọi'), where: '', home: 'unknown' },
        notes: x.preferCudnn
          ? [
              tr(
                'When cuDNN is preferred, the order of the others was not read; they are shown in the default order.',
                'Khi cuDNN được ưu tiên, thứ tự của các backend còn lại chưa được đọc; chúng được hiển thị theo thứ tự mặc định.',
              ),
            ]
          : [],
      };
    }
    case 'torch-rocm':
      return {
        project: p,
        checks: [
          {
            label: tr('AOTriton or Composable Kernel', 'AOTriton hoặc Composable Kernel'),
            verdict: 'not-modelled', // i18n-ignore: verdict id, the page shows its own label
            detail: tr('the selection on ROCm was not read', 'phần lựa chọn trên ROCm chưa được đọc'),
            source: 'ch20 B5, x3 B9', // i18n-ignore: citation
          },
        ],
        kernel: {
          name: tr('AOTriton or CK flash attention', 'flash attention của AOTriton hoặc CK'),
          where: tr(
            'aten/src/ATen/native/transformers/hip/ (aotriton_adapter.h, flash_attn/{aot,ck}); AOTriton pinned in cmake/External/aotriton.cmake',
            'aten/src/ATen/native/transformers/hip/ (aotriton_adapter.h, flash_attn/{aot,ck}); AOTriton được ghim trong cmake/External/aotriton.cmake',
          ),
          home: 'library',
        },
        notes: [tr('The CUDA sources are hipified at build time (x3 B9).', 'Mã nguồn CUDA được hipify lúc build (x3 B9).')],
      };
    case 'torch-mps':
      return {
        project: p,
        checks: [
          {
            label: tr('long cache, short query or prefill', 'cache dài, query ngắn hoặc prefill'),
            verdict: 'not-modelled', // i18n-ignore: verdict id, the page shows its own label
            detail: tr('Attention.mm routes to three kernels; the thresholds were not read', 'Attention.mm chuyển tới ba kernel; các ngưỡng chưa được đọc'),
            source: tr('ch20 D5, Attention.mm 403 to 410', 'ch20 D5, Attention.mm 403 đến 410'),
          },
        ],
        kernel: {
          name: x.queries === 1
            ? tr(
                'its own decode kernels: sdpa_vector_2pass_mps (long caches) or sdpa_vector_fast_mps',
                'các kernel decode của riêng nó: sdpa_vector_2pass_mps (cache dài) hoặc sdpa_vector_fast_mps',
              )
            : tr(
                'its own kernels: sdpa_vector_fast_mps (short queries) or sdpa_prefill_mps',
                'các kernel của riêng nó: sdpa_vector_fast_mps (query ngắn) hoặc sdpa_prefill_mps',
              ),
          where: tr(
            'aten/src/ATen/native/mps/kernels/Attention.metal ("Largely influeneced by" MLX)',
            'aten/src/ATen/native/mps/kernels/Attention.metal ("Largely influeneced by" MLX: phần lớn chịu ảnh hưởng của MLX)',
          ),
          home: 'own',
        },
        notes: [],
      };
  }
}

// ---------------------------------------------------------------- the start page's prediction
export interface HomeRow {
  project: 'llama.cpp' | 'MLX' | 'PyTorch';
  backend: string;
  /** Where the attention kernels live, by the facts. */
  home: string;
  /** Every kernel for every shape is source in the project's own tree. */
  allOwn: boolean;
}
/** Where each project's attention kernels live, backend by backend (x3 B5, ch20 B5, D4, D5). */
export const ATTENTION_HOME: HomeRow[] = [
  {
    project: 'llama.cpp',
    backend: 'CUDA and HIP', // i18n-ignore: identifier also shown to the reader (matched with includes('CUDA') and pinned by the test)
    get home() {
      return tr('its own fattn-* kernels', 'các kernel fattn-* của riêng nó');
    },
    allOwn: true,
  },
  {
    project: 'llama.cpp',
    backend: 'Metal',
    get home() {
      return tr('its own fa_common / fa_aux kernels', 'các kernel fa_common / fa_aux của riêng nó');
    },
    allOwn: true,
  },
  {
    project: 'MLX',
    backend: 'Metal',
    get home() {
      return tr('its own vector and steel kernels', 'các kernel vector và steel của riêng nó');
    },
    allOwn: true,
  },
  {
    project: 'MLX',
    backend: 'CUDA',
    get home() {
      return tr("cuDNN for prefill and long decodes, its own vector kernels for short decodes", 'cuDNN cho prefill và decode dài, các kernel vector của riêng nó cho decode ngắn');
    },
    allOwn: false,
  },
  {
    project: 'PyTorch',
    backend: 'CUDA',
    get home() {
      return tr('FlashAttention-2 from a submodule, its own memory-efficient and math paths, cuDNN', 'FlashAttention-2 từ một submodule, các đường memory-efficient và math của riêng nó, cuDNN');
    },
    allOwn: false,
  },
  {
    project: 'PyTorch',
    backend: 'ROCm',
    get home() {
      return tr('AOTriton or Composable Kernel', 'AOTriton hoặc Composable Kernel');
    },
    allOwn: false,
  },
  {
    project: 'PyTorch',
    backend: 'MPS',
    get home() {
      return tr('its own Metal kernels', 'các kernel Metal của riêng nó');
    },
    allOwn: true,
  },
];
/** Projects whose own tree holds the attention kernel on every GPU backend they support. */
export const OWN_EVERYWHERE = [...new Set(ATTENTION_HOME.map((r) => r.project))].filter((p) =>
  ATTENTION_HOME.filter((r) => r.project === p).every((r) => r.allOwn),
);

/** Where a project's CUDA attention kernels come from, as the start page's prediction asks it. */
export type CudaHome = 'own' | 'mixed' | 'submodule';
/** Every shape the explorer offers, with the library checks answered yes (the defaults). */
const SHAPES: AttnInput[] = [1, 8, 128].flatMap((queries) =>
  [1, 100, 256, 1000, 5000].flatMap((keys) =>
    (['none', 'causal', 'array'] as Mask[]).flatMap((mask) => [false, true].map((gqa) => ({ ...DEFAULT_INPUT, queries, keys, mask, gqa }))),
  ),
);
/**
 * Derived from attnRoute over every shape: 'submodule' when the default first choice is another
 * project's kernels, 'own' when every shape runs the project's own source, else 'mixed'.
 */
export function cudaHome(p: 'llama' | 'mlx-cuda' | 'torch-cuda'): CudaHome {
  if (attnRoute(p, DEFAULT_INPUT).kernel.home === 'submodule') return 'submodule';
  return SHAPES.every((x) => attnRoute(p, x).kernel.home === 'own') ? 'own' : 'mixed';
}
/** The start page stores its pick by these project ids (sim/backendmap.ts). */
export const CUDA_HOME_OF = { 'llama.cpp': 'llama', mlx: 'mlx-cuda', pytorch: 'torch-cuda' } as const;
