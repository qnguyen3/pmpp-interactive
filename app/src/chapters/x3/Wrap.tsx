import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';

// Module X3's summary: what to carry forward, a mixed check on situations the sections did
// not show, where each of the module's tasks is worked (the book has no exercises for it),
// where the sources disagree (the module has no PDF errata), and milestone M10.

export const questionsWrap: Record<string, Question> = {
  'x3/wrap/q-permute-row': {
    kind: 'numeric',
    id: 'x3/wrap/q-permute-row',
    prompt: (
      <Translated
        vi={
          <>
            Bạn chuyển <code>k_proj.weight</code> của SmolLM2-135M, 192 hàng (3 head KV, mỗi head 64) × 576, sang cách ghép cặp
            NORM của llama.cpp bằng <code>permute(weights, n_head = 9, n_head_kv = 3)</code> của bộ chuyển đổi. Trong một head, nó
            đưa hàng k × 32 + j tới hàng 2j + k. Hàng 100 của tensor Hugging Face rơi vào hàng nào của tensor đã chuyển?
          </>
        }
      >
        <>
          You convert SmolLM2-135M's <code>k_proj.weight</code>, 192 rows (3 KV heads of 64) by 576, for llama.cpp's NORM
          pairing with its converter's <code>permute(weights, n_head = 9, n_head_kv = 3)</code>. Within a head it sends row
          k × 32 + j to row 2j + k. Row 100 of the Hugging Face tensor lands at which row of the converted one?
        </>
      </Translated>
    ),
    answer: 73,
    unit: 'row',
    explain: (
      <Translated
        vi={
          <p>
            Với K, hàm chuyển sang <code>n_head_kv</code>, nên mỗi head có 192 / 3 = 64 hàng. Hàng 100 là hàng 36 của head 1, và
            36 = 1 × 32 + 4, nên k = 1, j = 4, và nó tới hàng 2 × 4 + 1 = 9 của head đó: 64 + 9 = 73. Hàng cặp với nó theo cách
            ghép nửa-chia của Hugging Face, hàng 68 (hàng 4 của head 1), tới hàng 72. Hai hàng mà một phép quay trộn với nhau giờ
            nằm cạnh nhau, đúng như cách ghép cặp liên tiếp mong đợi. Nếu permute K với 9 head thay vào đó thì 192 / 9 / 2 không
            phải số nguyên, nên phép reshape thất bại.
          </p>
        }
      >
        <p>
          For K the function switches to <code>n_head_kv</code>, so the heads are 192 / 3 = 64 rows. Row 100 is row 36 of
          head 1, and 36 = 1 × 32 + 4, so k = 1, j = 4, and it goes to 2 × 4 + 1 = 9 of that head: 64 + 9 = 73. Its partner
          under Hugging Face's half-split pairing, row 68 (row 4 of head 1), goes to row 72. The two rows that one rotation
          mixes now sit next to each other, which is what the consecutive pairing expects. Permute K with 9 heads instead and
          192 / 9 / 2 is not a whole number, so the reshape fails.
        </p>
      </Translated>
    ),
    source: 'notes/today/x3-facts.md B6 (permute), D4 (shapes), D6; the arithmetic is ours',
    vi: { unit: 'hàng', source: 'notes/today/x3-facts.md B6 (permute), D4 (shape), D6; phần tính toán là của chúng tôi' },
  },
  'x3/wrap/q-batch-flip': {
    kind: 'choice',
    id: 'x3/wrap/q-batch-flip',
    prompt: (
      <Translated
        vi={
          <>
            Với llama.cpp trên GPU NVIDIA và trọng số float32, một prompt decode riêng lẻ và cùng prompt đó decode như một trong
            16 chuỗi của một batch cho cùng các token greedy cho tới bước 30, rồi khác nhau. Không lần chạy nào có lỗi. Lý do
            nhiều khả năng là gì?
          </>
        }
      >
        <>
          With llama.cpp on an NVIDIA GPU and float32 weights, one prompt decoded alone and the same prompt decoded as one of
          16 sequences in a batch give the same greedy tokens up to step 30, then different ones. Neither run has a bug. What
          is the likely reason?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Ở batch 16, một kernel khác chạy các phép nhân, cộng theo thứ tự khác, và khoảng cách của một bước nhỏ hơn mức chênh đó.">
            At batch 16 another kernel runs the products, adding in another order, and one step’s margin was below that difference.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Trong decode theo batch, các hàng cache của 15 chuỗi còn lại rò vào attention của chuỗi này khi cache dài ra.">
            In a batched decode, cache rows of the other 15 sequences leak into this sequence’s attention once the cache grows long.
          </Translated>
        ),
        why: (
          <Translated vi="Đó sẽ là một lỗi, và nó sẽ làm thay đổi logit ở mọi bước, chứ không chỉ lật một token ở bước 30.">
            That would be a bug, and it would change the logits at every step, not flip one token at step 30.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Giải mã greedy vẫn rút một số ngẫu nhiên mỗi bước, và batch đã thay đổi số mà chuỗi này nhận được.">
            Greedy decoding still draws one random number per step, and the batch changed which number this sequence received.
          </Translated>
        ),
        why: (
          <Translated vi="Greedy lấy argmax: không có số ngẫu nhiên nào được rút (llama2.c với -t 0.0; M4).">
            Greedy takes the argmax: no random number is drawn (llama2.c at -t 0.0; M4).
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Phép cộng số thực trên GPU làm tròn khác nhau mỗi lần chạy, nên hai lần chạy của bất kỳ mô hình nào cũng trôi xa nhau sau vài chục bước decode.">
            GPU float addition rounds differently on every run, so two runs of any model drift apart after a few dozen decoding steps.
          </Translated>
        ),
        why: (
          <Translated vi="Cùng một kernel cộng cùng các giá trị theo cùng thứ tự cho ra cùng các bit. Thứ thay đổi ở đây là kernel, và do đó là thứ tự.">
            The same kernel adding the same values in the same order gives the same bits. What changed here is the kernel, and so the order.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            llama.cpp chọn kernel nhân ma trận theo số token: kernel vector tới 8, sau đó là một kernel ma trận hoặc cuBLAS. Mỗi
            kernel cộng các tích theo thứ tự riêng, và phép cộng số thực không có tính kết hợp, nên logit khác nhau ở những bit
            cuối. Điều đó vô hại cho tới khi hai logit cao nhất của một bước gần nhau hơn mức chênh đó; khi ấy token bị lật, và mọi
            bước sau đó thấy một đầu vào khác. Đây là lý do lab X3.2 kiểm tra tính bất biến theo batch trên một kernel và chỉ kiểm
            tra token ở những nơi khoảng cách lớn.
          </p>
        }
      >
        <p>
          llama.cpp picks its matrix-product kernel by the number of tokens: vector kernels up to 8, then a matrix kernel or
          cuBLAS. Each adds the products in its own order, and float addition is not associative, so the logits differ in
          their last bits. That is harmless until a step's top two logits are closer than the difference; then the token
          flips, and every later step sees another input. This is why lab X3.2 checks batch invariance on one kernel and
          checks tokens only where the margin is large.
        </p>
      </Translated>
    ),
    source: 'notes/today/x3-facts.md B4, D7; ch22-facts.md D6; ch10-facts.md A9',// i18n-ignore: citation only
  },
  'x3/wrap/q-nmse-zero': {
    kind: 'choice',
    id: 'x3/wrap/q-nmse-zero',
    prompt: (
      <Translated
        vi={
          <>
            Bạn thêm vào llama.cpp một op có đầu ra đúng bằng 0 ở hầu hết mọi phần tử, vì phần lớn đầu vào bị che. Kernel CUDA của
            bạn lệch so với CPU trong phạm vi 1e-6 ở mọi nơi, vậy mà <code>test-backend-ops</code> báo NMSE vượt xa
            1e-7. Chuyện gì đang xảy ra?
          </>
        }
      >
        <>
          You add an op to llama.cpp whose output is exactly 0 in almost every element, because most inputs are masked. Your
          CUDA kernel is within 1e-6 of the CPU everywhere, yet <code>test-backend-ops</code> reports an NMSE far above
          1e-7. What is going on?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="NMSE chia sai số bình phương cho tổng bình phương của tham chiếu, ở đây rất nhỏ, nên sai số nhỏ cho ra tỉ số lớn.">
            NMSE divides the squared error by the reference’s sum of squares, tiny here, so small errors give a large ratio.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Kernel của bạn hẳn đã ghi một NaN ở đâu đó, vì NMSE lớn như vậy không thể chỉ đến từ sai số làm tròn.">
            Your kernel must have written a NaN somewhere, because an NMSE that large cannot come from rounding errors alone.
          </Translated>
        ),
        why: (
          <Translated vi="Một NaN sẽ làm NMSE thành NaN. Sai số bình thường có thể cho ra bất kỳ tỉ số nào khi mẫu số nhỏ.">
            A NaN would make the NMSE NaN. Ordinary errors can give any ratio when the denominator is small.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Công cụ so kernel của bạn với một backend GPU khác trên máy, và cách làm tròn của backend đó khác của bạn.">
            The tool compares your kernel with another GPU backend on the machine, and that backend’s rounding differs from yours.
          </Translated>
        ),
        why: (
          <Translated vi="Mặc định nó so mỗi backend với backend CPU.">By default it compares each backend with the CPU backend.</Translated>
        ),
      },
      {
        label: (
          <Translated vi="Ngưỡng 1e-7 dành cho cả một mô hình; một op riêng lẻ phải đạt một ngưỡng riêng chặt hơn nhiều.">
            The 1e-7 threshold is meant for a whole model; a single op is held to a much stricter threshold of its own.
          </Translated>
        ),
        why: (
          <Translated vi="Ngược lại: 1e-7 là mặc định cho mỗi op, và bài kiểm tra của một op có thể ghi đè nó.">
            The other way round: 1e-7 is the per-op default, which an op’s test may override.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>nmse</code> cộng (a − b)² trên các phần tử và chia cho tổng a², tức độ lớn của chính tham chiếu. Điều đó biến nó
            thành một sai số tương đối, phù hợp với các đầu ra có độ lớn nhất định. Ở đây mẫu số là bình phương của vài đầu ra không
            bị che, nên sai số 1e-6 ở mọi chỗ khác có thể lấn át nó. Một op như thế cần ngưỡng riêng, thứ mà công cụ cho phép một bài
            kiểm tra ghi đè, hoặc đầu vào kiểm tra có đầu ra không phần lớn là 0.
          </p>
        }
      >
        <p>
          <code>nmse</code> sums (a − b)² over the elements and divides by the sum of a², the reference's own size. That makes
          it a relative error, which suits outputs with some magnitude. Here the denominator is the squares of the few
          unmasked outputs, so errors of 1e-6 everywhere else can dominate it. An op like this needs its own threshold, which
          the tool lets a test override, or test inputs whose output is not mostly zero.
        </p>
      </Translated>
    ),
    source: 'notes/today/x3-facts.md D7 (nmse, max_nmse_err); the diagnosis is ours',
    vi: { source: 'notes/today/x3-facts.md D7 (nmse, max_nmse_err); phần chẩn đoán là của chúng tôi' },
  },
  'x3/wrap/q-rms-dtype': {
    kind: 'choice',
    id: 'x3/wrap/q-rms-dtype',
    prompt: (
      <Translated
        vi={
          <>
            Bạn profile một mô hình PyTorch trên CUDA. Bạn kỳ vọng mỗi <code>torch.nn.functional.rms_norm</code> là một kernel hợp
            nhất, nhưng bạn thấy vài kernel nhỏ cho mỗi phép norm và một cảnh báo. Trọng số của norm là bfloat16 còn activation là
            float32. Bạn kiểm tra gì trước tiên?
          </>
        }
      >
        <>
          You profile a PyTorch model on CUDA. You expect each <code>torch.nn.functional.rms_norm</code> to be one fused
          kernel, but you see several small kernels per norm and a warning. The norm's weight is bfloat16 and the activations
          are float32. What do you check first?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Các dtype: khi trọng số và đầu vào khác nhau, rms_norm cảnh báo “Cannot dispatch to fused implementation.”">
            The dtypes: when weight and input differ, rms_norm warns “Cannot dispatch to fused implementation.”
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="GPU này có số học bfloat16 hay không, vì kernel gộp chỉ chạy được nơi bfloat16 được hỗ trợ gốc.">
            Whether this GPU has bfloat16 arithmetic, since the fused kernel can only run where bfloat16 is native.
          </Translated>
        ),
        why: (
          <Translated vi="Đường dispatch đã đọc trong lượt kiểm tra dữ kiện kiểm tra dtype của tensor, không kiểm tra thiết bị.">
            The dispatch path read in the facts pass tests the tensors’ dtypes, not the device.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Mô hình có thật sự ở trên CUDA không, vì RMSNorm gộp chỉ được đăng ký cho GPU Apple.">
            Whether the model is really on CUDA, since the fused RMSNorm is registered only for Apple GPUs.
          </Translated>
        ),
        why: (
          <Translated vi="native_functions.yaml liệt kê CUDA: _fused_rms_norm_cuda bên cạnh MPS: _fused_rms_norm_mps.">
            native_functions.yaml lists CUDA: _fused_rms_norm_cuda beside MPS: _fused_rms_norm_mps.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các hàng có dài hơn 2,048 phần tử không, giới hạn của kernel mỗi-warp-một-hàng mà PyTorch dùng cho hàng ngắn.">
            Whether the rows are longer than 2,048 elements, the limit of the warp-per-row kernel PyTorch uses for short rows.
          </Translated>
        ),
        why: (
          <Translated vi="Giới hạn đó thuộc về kernel persistent của softmax (ch10-facts D4), không phải RMSNorm.">
            That limit belongs to softmax’s persistent kernel (ch10-facts D4), not to RMSNorm.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>rms_norm</code> gọi <code>at::_fused_rms_norm</code>, hàm dispatch tới <code>_fused_rms_norm_cuda</code>
            . Nó lùi về một tổ hợp các phép toán tensor thông thường với đầu vào số phức, và cảnh báo "Cannot dispatch to fused
            implementation." khi dtype của trọng số khác của đầu vào: vài kernel thay vì một. Cho trọng số cùng dtype với
            activation, hoặc ngược lại, và kernel gộp có thể chạy.
          </p>
        }
      >
        <p>
          <code>rms_norm</code> calls <code>at::_fused_rms_norm</code>, which dispatches to <code>_fused_rms_norm_cuda</code>
          . It falls back to a composite of ordinary tensor ops for complex input, and warns "Cannot dispatch to fused
          implementation." when the weight's dtype differs from the input's: several kernels instead of one. Give the weight
          the activations' dtype, or the other way round, and the fused kernel can run.
        </p>
      </Translated>
    ),
    source: 'notes/today/x3-facts.md A3, B2 (layer_norm.cpp lines 345 to 372)',
    vi: { source: 'notes/today/x3-facts.md A3, B2 (layer_norm.cpp các dòng 345 đến 372)' },
  },
};

// ---------------------------------------------------------------- milestone M10
/** Every lab M10's step runs, as lab X3.2's provided/pieces.cmake copies them. */
const M10: { num: string; chapter: string; title: string; dir: string; need: 'required' | 'if present' | 'yours' }[] = [
  { num: '2.3', chapter: 'ch02', get title() { return tr('M0: the residual add', 'M0: phép cộng residual'); }, dir: 'ch02/lab3_elementwise', need: 'required' },
  { num: '6.4', chapter: 'ch06', get title() { return tr('M2: the linear layer', 'M2: lớp tuyến tính'); }, dir: 'ch06/lab4_linear', need: 'required' },
  { num: '10.3', chapter: 'ch10', get title() { return tr('M4: argmax', 'M4: argmax'); }, dir: 'ch10/lab3_softmax_argmax', need: 'required' },
  { num: '20.4', chapter: 'ch20', get title() { return tr('M7: the KV append and decode attention', 'M7: ghi nối KV và attention decode'); }, dir: 'ch20/lab4_decode', need: 'required' },
  { num: '22.2', chapter: 'ch22', get title() { return tr('M8: the embedding gather', 'M8: phép gather embedding'); }, dir: 'ch22/lab2_batching', need: 'required' },
  { num: 'X3.1', chapter: 'x3', get title() { return tr('RMSNorm, RoPE and SwiGLU', 'RMSNorm, RoPE và SwiGLU'); }, dir: 'x3/lab1_llama_pieces', need: 'required' },
  { num: '15.6', chapter: 'ch15', get title() { return tr('M3: the register-tiled linear layer', 'M3: lớp tuyến tính phân ô thanh ghi'); }, dir: 'ch15/lab6_linear', need: 'if present' },// i18n-ignore: data key, shown translated
  { num: '20.5', chapter: 'ch20', get title() { return tr('M7: grouped-query decode attention', 'M7: attention decode grouped-query'); }, dir: 'ch20/lab5_gqa_decode', need: 'if present' },// i18n-ignore: data key, shown translated
  { num: '22.1', chapter: 'ch22', get title() { return tr('The gather GEMV', 'GEMV gather'); }, dir: 'ch22/lab1_gemv', need: 'if present' },// i18n-ignore: data key, shown translated
  { num: 'X1.3', chapter: 'x1', get title() { return tr('M5: the sampler', 'M5: bộ lấy mẫu'); }, dir: 'x1/lab3_sampler', need: 'if present' },// i18n-ignore: data key, shown translated
  { num: 'X2.4', chapter: 'x2', get title() { return tr('M9: the quantised linear layer, in the M9 checks', 'M9: lớp tuyến tính lượng tử hóa, trong các bước kiểm tra của M9'); }, dir: 'x2/lab4_m9', need: 'if present' },// i18n-ignore: data key, shown translated
  { num: 'X3.2', chapter: 'x3', get title() { return tr('M10: the step that composes them', 'M10: bước ghép chúng lại'); }, dir: 'x3/lab2_tinyllama', need: 'yours' },
];
const where = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập (emulator)') : b === 'metal' ? 'Metal' : b.toUpperCase());
/** Backends (devices first, then the emulator) on which the reader's own work of a lab passes. */
const passingOn = (reports: Reports, lab: string) =>
  [...realBackends(reports), 'emu'].filter((b) => labStatus(reports[b], lab, 'work') === 'pass');

// ---------------------------------------------------------------- the page
export function Wrap() {
  const { progress } = useProgress();
  const { reports, loading } = useReports();
  const ran = Object.keys(reports).length > 0;
  const own = passingOn(reports, 'x3/lab2_tinyllama');
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('x3/') && !k.includes('/p-'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}><Translated vi="Những gì cần mang theo">What to carry forward</Translated></h2>
      <Translated
        vi={
          <ul>
            <li>
              Mỗi dự án có một chỗ nơi một op trở thành một lời gọi backend. Trong llama.cpp đó là một câu lệnh switch trên{' '}
              <code>GGML_OP_*</code> cho mỗi backend, sau một lượt gộp; trong MLX là một <code>eval_gpu</code> cho mỗi primitive
              trong thư mục của từng backend; trong PyTorch là các dispatch key của <code>native_functions.yaml</code>, và với một số
              op là một kernel được sinh ra lúc build. Hãy bắt đầu ở đó và lần theo một cái tên.
            </li>
            <li>
              Trên Metal, hãy lần theo chuỗi, không phải tên hàm: host yêu cầu một pipeline bằng tên trong một{' '}
              <code>[[host_name]]</code>, và các biến thể là function constant ở chỗ mã CUDA dùng tham số template. HIP biên dịch
              mã nguồn CUDA của llama.cpp và PyTorch; MLX không có backend AMD.
            </li>
            <li>
              Các kernel là của khóa học. Một op từng phần tử là kernel của chương 2, được gộp việc với các lần nạp vector trong MLX và
              PyTorch; RMSNorm và softmax là phép rút gọn khối hai giai đoạn của chương 10; một GEMV decode là của lab 22.1. Hãy đọc
              để tìm chỗ khác: hàm mũ, số lượt đi qua hàng, phần đuôi.
            </li>
            <li>
              Shape chọn kernel: một kernel vector cho một hoặc vài token, một kernel phân ô hoặc thư viện của hãng cho nhiều hơn.
              Attention có thể nằm trong cây mã, trong một thư viện hoặc trong một submodule.
            </li>
            <li>
              Một bước Llama là các kernel của khóa học cộng thêm RMSNorm, RoPE và SwiGLU. RoPE có hai cách ghép cặp; chuyển một
              checkpoint giữa chúng sẽ hoán vị các hàng Q và K, với số head KV cho K. Một checkpoint thật còn khác về dtype, kích
              thước head và các hằng số, và kích thước của nó dùng để kiểm tra header.
            </li>
            <li>
              So sánh với một tham chiếu bằng một thước đo sai số trên logit, và chỉ yêu cầu token greedy giống nhau ở nơi khoảng
              cách giữa hai logit cao nhất lớn hơn sai số đó. Một lần lật ở một bước sát nút làm thay đổi mọi token sau đó mà không
              kernel nào sai.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            Every project has one place where an op becomes a backend call. In llama.cpp it is a switch over{' '}
            <code>GGML_OP_*</code> per backend, after a fusion pass; in MLX an <code>eval_gpu</code> per primitive in each
            backend's directory; in PyTorch the dispatch keys of <code>native_functions.yaml</code>, and for some ops a kernel
            generated at build time. Start there and follow one name.
          </li>
          <li>
            On Metal, follow the string, not the function name: the host asks for a pipeline by the name in a{' '}
            <code>[[host_name]]</code>, and variants are function constants where CUDA code uses template parameters. HIP
            compiles llama.cpp's and PyTorch's CUDA sources; MLX has no AMD backend.
          </li>
          <li>
            The kernels are the course's. An elementwise op is chapter 2's kernel, coarsened with vector loads in MLX and
            PyTorch; RMSNorm and softmax are chapter 10's two-stage block reduction; a decode GEMV is lab 22.1's. Read for
            what differs: the exponential, the number of passes over the row, the tail.
          </li>
          <li>
            The shape picks the kernel: a vector kernel for one or a few tokens, a tiled kernel or the vendor's library for
            more. Attention may be in the tree, in a library or in a submodule.
          </li>
          <li>
            A Llama step is the course's kernels plus RMSNorm, RoPE and SwiGLU. RoPE has two pairings; moving a checkpoint
            between them permutes the Q and K rows, with the KV head count for K. A real checkpoint also differs in dtype,
            head size and constants, and its size checks its header.
          </li>
          <li>
            Compare with a reference by an error measure on the logits, and require equal greedy tokens only where the top-two
            margin is larger than that error. A flip at a close step changes every later token without any kernel being
            wrong.
          </li>
        </ul>
      </Translated>
      <Beyond title={tr("Ours, not the book's", 'Của chúng tôi, không phải của sách')}>
        <Translated
          vi={
            <p>
              Sách không đọc mã nguồn của dự án nào. Mọi khẳng định ở trên về llama.cpp, MLX và PyTorch đều được đọc trong lượt kiểm tra
              dữ kiện ngày 2026-11-20 tại các commit mà <a href={href('x3', 'start')}>trang bắt đầu</a> liệt kê.
            </p>
          }
        >
          <p>
            The book reads no project's source. Every claim above about llama.cpp, MLX and PyTorch was read in the facts pass
            on 2026-11-20 at the commits the <a href={href('x3', 'start')}>start page</a> lists.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Một bài kiểm tra tổng hợp">A mixed check</Translated></h2>
      <p>
        <Translated vi="Các câu này dựa trên toàn bộ module, vào những tình huống mà các phần chưa cho thấy.">These draw on the whole module, on situations the sections did not show.</Translated>
        {asked ? tr(` So far in this module you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.`, ` Đến giờ trong module này bạn đã trả lời đúng ${right} trên ${asked} câu, tính theo câu trả lời gần nhất của bạn cho mỗi câu.`) : ''}
      </p>
      <Ask q={q['x3/wrap/q-permute-row']} label={tr('Follow the row', 'Lần theo hàng')} />
      <Ask q={q['x3/wrap/q-batch-flip']} />
      <Ask q={q['x3/wrap/q-nmse-zero']} />
      <Ask q={q['x3/wrap/q-rms-dtype']} label={tr('Read the profile', 'Đọc kết quả profile')} />

      <h2><Translated vi="Bài tập của module">The module's exercises</Translated></h2>
      <Translated
        vi={
          <p>
            Sách không có bài tập cho module này, vì sách không đọc backend thật nào. Mục tiêu của kế hoạch cho X3 là tìm kernel cho
            một op cho trước trong mỗi codebase thật trong ba codebase và giải thích nó, và mốc dự án M10 là một mô hình kiểu Llama được
            kiểm tra từng token một so với một tham chiếu. Đây là nơi từng phần của việc đó được thực hiện.
          </p>
        }
      >
        <p>
          The book has no exercises for this module, because it reads no real backend. The plan's goal for X3 is to find the
          kernel for a given op in each of three real codebases and explain it, and its milestone M10 is a Llama-style model
          checked token by token against a reference. Here is where each part of that is worked.
        </p>
      </Translated>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="exercises">
        <thead>
          <tr><th><Translated vi="Nhiệm vụ">Task</Translated></th><th><Translated vi="Thực hiện trong một phần">Worked in a section</Translated></th><th><Translated vi="Xây dựng trong một lab">Built in a lab</Translated></th></tr>
        </thead>
        <tbody>
          <tr><td><Translated vi="Tìm kernel cho một op trong mỗi dự án">Find the kernel for an op in each project</Translated></td><td><Translated vi={<>bản đồ op, các lộ trình và scene trong <a href={href('x3', 'X3.1')}>X3.1</a></>}>the op map, routes and scene in <a href={href('x3', 'X3.1')}>X3.1</a></Translated></td><td><Translated vi="không có: chỉ đọc">none: reading only</Translated></td></tr>
          <tr><td><Translated vi="Gọi tên bản tương ứng HIP và Metal của một kernel CUDA">Name the HIP and Metal counterpart of a CUDA kernel</Translated></td><td><Translated vi={<>cột HIP của bản đồ op trong <a href={href('x3', 'X3.1')}>X3.1</a>; các biến thể trong <a href={href('x3', 'X3.2')}>X3.2</a></>}>the op map's HIP column in <a href={href('x3', 'X3.1')}>X3.1</a>; variants in <a href={href('x3', 'X3.2')}>X3.2</a></Translated></td><td><Translated vi={<>lab X3.1: một <code>.cu</code> cho CUDA, HIP và các trình giả lập, cùng các tệp Metal của nó</>}>lab X3.1: one <code>.cu</code> for CUDA, HIP and the emulators, and its Metal files</Translated></td></tr>
          <tr><td><Translated vi="Giải thích một kernel từng phần tử hoặc gather">Explain an elementwise or gather kernel</Translated></td><td><Translated vi={<>trình khám phá launch shape trong <a href={href('x3', 'X3.2')}>X3.2</a></>}>the launch-shape explorer in <a href={href('x3', 'X3.2')}>X3.2</a></Translated></td><td><Translated vi="SwiGLU của lab X3.1">lab X3.1's SwiGLU</Translated></td></tr>
          <tr><td><Translated vi="Giải thích một phép rút gọn theo hàng">Explain a row reduction</Translated></td><td><Translated vi={<>trình chạy từng bước và scene trong <a href={href('x3', 'X3.3')}>X3.3</a></>}>stepper and scene in <a href={href('x3', 'X3.3')}>X3.3</a></Translated></td><td><Translated vi="RMSNorm của lab X3.1, và phần xem xét bản chuyển ở phần B">lab X3.1's RMSNorm, and the port review of part B</Translated></td></tr>
          <tr><td><Translated vi="Dự đoán kernel nhân ma trận từ shape">Predict the matrix-product kernel from the shape</Translated></td><td><Translated vi={<>trình khám phá lộ trình trong <a href={href('x3', 'X3.4')}>X3.4</a></>}>the route explorer in <a href={href('x3', 'X3.4')}>X3.4</a></Translated></td><td><Translated vi="benchmark của lab X3.2, gather so với M2 và M3">lab X3.2's benchmark, gather against M2 and M3</Translated></td></tr>
          <tr><td><Translated vi="Tìm một kernel attention, trong cây mã hay không">Find an attention kernel, in the tree or not</Translated></td><td><a href={href('x3', 'X3.5')}>X3.5</a></td><td><Translated vi="lab X3.2 chạy kernel decode của lab 20.4 hoặc 20.5">lab X3.2 runs lab 20.4's or 20.5's decode kernel</Translated></td></tr>
          <tr><td><Translated vi="Dự đoán token của bộ lấy mẫu khi hòa">Predict a sampler's token on a tie</Translated></td><td><Translated vi={<>trình chạy từng bước cho trường hợp hòa trong <a href={href('x3', 'X3.6')}>X3.6</a></>}>the tie stepper in <a href={href('x3', 'X3.6')}>X3.6</a></Translated></td><td><Translated vi="bước greedy của lab X3.2, chỉ số nhỏ nhất của M4">lab X3.2's greedy step, M4's lowest index</Translated></td></tr>
          <tr><td><Translated vi="Bố trí một bước Llama; tách bạch các cách ghép cặp RoPE">Lay out a Llama step; keep the RoPE pairings apart</Translated></td><td><Translated vi={<>trình chạy từng bước, trình khám phá và các scene trong <a href={href('x3', 'X3.7')}>X3.7</a></>}>stepper, explorer and scenes in <a href={href('x3', 'X3.7')}>X3.7</a></Translated></td><td><Translated vi="RoPE của lab X3.1, bước của lab X3.2 và bài kiểm tra ghép cặp của nó">lab X3.1's RoPE, lab X3.2's step and its pairing test</Translated></td></tr>
          <tr><td><Translated vi="Đọc header của một checkpoint; kiểm tra một bản tải về">Read a checkpoint's header; check a download</Translated></td><td><Translated vi={<>bộ giải mã và trình khám phá trong <a href={href('x3', 'X3.8')}>X3.8</a></>}>decoder and explorer in <a href={href('x3', 'X3.8')}>X3.8</a></Translated></td><td><Translated vi="bộ nạp của lab X3.2 và bước kiểm tra kích thước của nó">lab X3.2's loader and its size check</Translated></td></tr>
          <tr><td><Translated vi="Kiểm tra so với một tham chiếu; chuẩn bị một đóng góp">Test against a reference; prepare a contribution</Translated></td><td><Translated vi={<>trình khám phá và scene trong <a href={href('x3', 'X3.9')}>X3.9</a></>}>explorer and scene in <a href={href('x3', 'X3.9')}>X3.9</a></Translated></td><td><Translated vi="các bài kiểm tra teacher forcing và chạy tự do của lab X3.2">lab X3.2's teacher-forced and free-running tests</Translated></td></tr>
          <tr><td><Translated vi="Chạy một checkpoint thật nhỏ và so sánh token">Run a small real checkpoint and compare tokens</Translated></td><td><Translated vi={<>các phép chuyển đổi trong <a href={href('x3', 'X3.8')}>X3.8</a></>}>the conversions in <a href={href('x3', 'X3.8')}>X3.8</a></Translated></td><td><Translated vi="lab X3.2 chạy một mô hình được sinh ra; README của nó mô tả đường đi với checkpoint thật, unverified">lab X3.2 runs a generated model; its README describes the real-checkpoint path, unverified</Translated></td></tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Các bài tập riêng của sách về những kernel mà các dự án dùng được ánh xạ trên trang tóm tắt của chương tương ứng: kernel
            từng phần tử ở <a href={href('ch02', 'wrap')}>chương 2</a>, gộp việc và nạp vector ở{' '}
            <a href={href('ch06', 'wrap')}>chương 6</a>, phép rút gọn ở <a href={href('ch10', 'wrap')}>chương 10</a>, nhân ma
            trận ở <a href={href('ch15', 'wrap')}>chương 15</a> và attention ở <a href={href('ch20', 'wrap')}>chương
            20</a>.
          </p>
        }
      >
        <p>
          The book's own exercises on the kernels the projects use are mapped on their chapters' summary pages: elementwise
          kernels in <a href={href('ch02', 'wrap')}>chapter 2</a>, coarsening and vector loads in{' '}
          <a href={href('ch06', 'wrap')}>chapter 6</a>, reduction in <a href={href('ch10', 'wrap')}>chapter 10</a>, matrix
          products in <a href={href('ch15', 'wrap')}>chapter 15</a> and attention in <a href={href('ch20', 'wrap')}>chapter
          20</a>.
        </p>
      </Translated>

      <h2><Translated vi="Nơi các nguồn bất đồng">Where the sources disagree</Translated></h2>
      <Translated
        vi={
          <p>
            Module này không có PDF nào để sai: nó không có trong sách. Thay vào đó nó có ba dự án bất đồng với nhau, và với những gì
            người đọc sẽ giả định. Đây là những bất đồng mà module được xây quanh, tất cả đều được đọc trong lượt kiểm tra dữ kiện.
          </p>
        }
      >
        <p>
          This module has no PDF to be wrong: it is not in the book. What it has instead are three projects that disagree
          with each other, and with what a reader would assume. These are the disagreements the module was built around, all
          read in the facts pass.
        </p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>
              <b>Kernel nằm ở đâu.</b> <code>add</code> CUDA của PyTorch không có tệp mã nguồn: torchgen viết nó lúc build.
              Flash attention CUDA của PyTorch nằm trong một submodule, và attention CUDA của MLX gọi cuDNN cho prefill và các bước
              decode dài.
            </li>
            <li>
              <b>Một dự án, hai backend.</b> Softmax của MLX lấy cực đại rồi lấy tổng trên Metal, với{' '}
              <code>fast::exp</code>, và là một phép chuẩn hóa online một lượt trên CUDA, với <code>__expf</code>. Trên Metal, softmax
              của PyTorch dùng <code>precise::exp</code> ở chỗ MLX dùng <code>fast::exp</code>.
            </li>
            <li>
              <b>Một kernel cho mỗi op.</b> Không dự án nào gửi mọi phép nhân ma trận tới một kernel: cả ba đều chọn theo shape.
            </li>
            <li>
              <b>RoPE.</b> llama.cpp chạy Llama với các cặp liên tiếp và hoán vị các hàng Q và K của Hugging Face cho khớp; Hugging
              Face, mặc định của mlx-lm và SmolLM2 dùng cách ghép nửa-chia; llama2.c dùng cặp liên tiếp; PyTorch không có op RoPE.{' '}
              <code>export.py</code> của llama2.c hoán vị K với số head query, điều mà theo cách đọc của chúng tôi không khớp với một
              mô hình grouped-query.
            </li>
            <li>
              <b>Trường hợp hòa và AMD.</b> llama2.c và <code>torch.argmax</code> giữ cực đại đầu tiên; các kernel argmax GPU của
              llama.cpp thì không. Các tệp CUDA của llama.cpp và PyTorch chính là tệp HIP của chúng; MLX không có backend AMD.
            </li>
            <li>
              <b>Định dạng.</b> GGUF là thứ llama.cpp nạp, nhưng không phải định dạng đơn giản nhất để bắt đầu; không checkpoint thật
              nhỏ nào có kích thước head 128.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            <b>Where the kernel is.</b> PyTorch's CUDA <code>add</code> has no source file: torchgen writes it at build time.
            PyTorch's CUDA flash attention is in a submodule, and MLX's CUDA attention calls cuDNN for prefill and long
            decodes.
          </li>
          <li>
            <b>One project, two backends.</b> MLX's softmax takes the maximum and then the sum on Metal, with{' '}
            <code>fast::exp</code>, and is a one-pass online normaliser on CUDA, with <code>__expf</code>. On Metal, PyTorch's
            softmax uses <code>precise::exp</code> where MLX uses <code>fast::exp</code>.
          </li>
          <li>
            <b>One kernel per op.</b> No project sends every matrix product to one kernel: all three pick by shape.
          </li>
          <li>
            <b>RoPE.</b> llama.cpp runs Llama with consecutive pairs and permutes Hugging Face's Q and K rows to fit; Hugging
            Face, mlx-lm's default and SmolLM2 use the half-split pairing; llama2.c consecutive pairs; PyTorch has no RoPE op.
            llama2.c's <code>export.py</code> permutes K with the query head count, which by our reading does not fit a
            grouped-query model.
          </li>
          <li>
            <b>Ties and AMD.</b> llama2.c and <code>torch.argmax</code> keep the first maximum; llama.cpp's GPU argmax kernels
            do not. llama.cpp's and PyTorch's CUDA files are their HIP files; MLX has no AMD backend.
          </li>
          <li>
            <b>Formats.</b> GGUF is what llama.cpp loads, but not the simplest first format; none of the small real
            checkpoints has head size 128.
          </li>
        </ul>
      </Translated>
      <Today checked={tr('2026-11-20 in the facts pass (notes/today/x3-facts.md A3, B3 to B6, B9, D5, D6, D9, E2; ch10-facts.md D2)', '2026-11-20 trong lượt kiểm tra dữ kiện (notes/today/x3-facts.md A3, B3 đến B6, B9, D5, D6, D9, E2; ch10-facts.md D2)')}>
        <Translated vi={<p>Mục "Summary of corrections" trong notes/today/x3-facts.md liệt kê mọi mục kèm nguồn của nó.</p>}>
          <p>The "Summary of corrections" in notes/today/x3-facts.md lists every item with its source.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Ba chỗ mà chính kế hoạch và thiết kế của chúng tôi đã sai, được phát hiện trong khi xây dựng module. M10 trong kế hoạch nói
            "run a small real checkpoint"; không checkpoint nào khớp kích thước head 128 của M7 và bản build không được tải gì về, nên
            lab X3.2 chạy một mô hình được sinh ra và mô tả đường đi thật, unverified. Thiết kế yêu cầu một bài kiểm tra rằng bất kỳ
            offset khác 0 nào cũng đưa mọi luồng của <code>binary_vv</code> của MLX sang đường vô hướng; điều đó sai với các lần nạp
            rộng hai phần tử có offset hai float, và phần <a href={href('x3', 'X3.2')}>X3.2</a> dùng quy tắc chính xác. Và thiết kế
            nói softmax Metal của MLX đọc hàng ba lần; các dữ kiện chỉ cho biết thứ tự các phép rút gọn của nó, nên phần{' '}
            <a href={href('x3', 'X3.3')}>X3.3</a> không đếm số lần đọc của nó.
          </p>
        }
      >
        <p>
          Three places where our own plan and design were wrong, found while building the module. The plan's M10 says "run a
          small real checkpoint"; none fits M7's head size of 128 and the build must download nothing, so lab X3.2 runs a
          generated model and describes the real path, unverified. The design asked for a test that any nonzero offset
          sends every thread of MLX's <code>binary_vv</code> to its scalar path; that is false for two-wide loads with an
          offset of two floats, and section <a href={href('x3', 'X3.2')}>X3.2</a> uses the exact rule. And the design said
          MLX's Metal softmax reads the row three times; the facts give only the order of its reductions, so section{' '}
          <a href={href('x3', 'X3.3')}>X3.3</a> does not count its reads.
        </p>
      </Translated>
      <Aside>
        <Translated vi={<p>Đây là cách chúng tôi đọc các nguồn vào ngày 2026-11-20. Các dự án thay đổi; hãy kiểm tra tệp trước khi dựa vào một dòng.</p>}>
          <p>These are our readings of the sources on 2026-11-20. Projects move; check the file before you rely on a line.</p>
        </Translated>
      </Aside>

      <h2><Translated vi="Dự án xuyên suốt cho tới nay">The project so far</Translated></h2>
      <p>
        <Translated
          vi={
            <>
              <code>launchLlamaStep</code> của lab X3.2 hoàn thành mốc dự án M10: một decoder kiểu Llama chạy trọn vẹn trên các kernel
              của chính khóa học và được kiểm tra từng logit, từng token so với một tham chiếu CPU.
            </>
          }
        >
          <>
            Lab X3.2's <code>launchLlamaStep</code> completes milestone M10: a Llama-style decoder runs end to end on the
            course's own kernels and is checked logit by logit and token by token against a CPU reference.
          </>
        </Translated>{' '}
        <span data-m10-own={loading ? 'loading' : own.join(',') || 'none'}>
          {loading
            ? tr('Reading reports.', 'Đang đọc báo cáo.')
            : !ran
              ? tr('No lab results are recorded on this machine yet.', 'Chưa có kết quả lab nào được ghi trên máy này.')
              : own.length
                ? tr(`Your own step passes on ${own.map(where).join(' and ')} on this machine.`, `Bước của chính bạn vượt qua trên ${own.map(where).join(' và ')} trên máy này.`)
                : tr('Your own step does not pass on any backend of this machine yet.', 'Bước của chính bạn chưa vượt qua trên backend nào của máy này.')}
        </span>{' '}
        <Translated
          vi={
            <>
              Xem <a href={href('x3', 'labs')}>các lab</a>. Bước này chạy lời giải tham chiếu (solution) của mỗi lab trước đó; bảng cho
              thấy phiên bản của chính bạn cho mỗi lab vượt qua ở đâu.
            </>
          }
        >
          <>
            See the <a href={href('x3', 'labs')}>labs</a>. The step runs each earlier lab's reference solution; the table shows
            where your own version of each passes.
          </>
        </Translated>
      </p>
      <table className="data" data-table="m10">
        <thead>
          <tr><th><Translated vi="Phần của M10">Piece of M10</Translated></th><th><Translated vi="Lab">Lab</Translated></th><th><Translated vi="Trong bước">In the step</Translated></th><th><Translated vi="Bản làm của bạn vượt qua trên">Your own work passes on</Translated></th></tr>
        </thead>
        <tbody>
          {M10.map((m) => {
            const on = passingOn(reports, m.dir);
            return (
              <tr key={m.dir} data-m10={m.num} data-pass={loading ? 'loading' : on.join(',') || 'none'}>
                <td>{m.title}</td>
                <td className="nowrap">
                  <a href={href(m.chapter, 'labs')}><Translated vi={<>lab {m.num}</>}>lab {m.num}</Translated></a>
                </td>
                <td>{tr(m.need, m.need === 'required' ? 'bắt buộc' : m.need === 'yours' ? 'của bạn' : 'nếu có')}</td>
                <td>{loading ? tr('reading reports', 'đang đọc báo cáo') : on.length ? on.map(where).join(', ') : <span className="muted"><Translated vi="chưa có gì">nothing yet</Translated></span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            M10 chạy ở float32, và với lớp lượng tử hóa của M9 ở vị trí lớp tuyến tính. Hàm khởi chạy của M9 nhận một descriptor ma
            trận lượng tử hóa, không phải trọng số float mà vị trí đó cần, nên một adapter được cung cấp sẵn gắn mỗi buffer trọng
            số float của mô hình với bản sao lượng tử hóa của nó. Bước của bạn không thay đổi, và các token Q8_0 và Q4_0 của nó được
            kiểm tra so với tham chiếu CPU trên cùng trọng số lượng tử hóa.
          </p>
        }
      >
        <p>
          M10 runs in float32, and with M9's quantised layer in its linear-layer slot. M9's launch function takes a quantised
          matrix descriptor, not the float weights the slot expects, so a provided adapter binds each of the model's float
          weight buffers to its quantised copy. Your step does not change, and its Q8_0 and Q4_0 tokens are checked against
          the CPU reference on the same quantised weights.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Đây là module cuối cùng theo thứ tự của khóa học này. Sau nó chỉ còn các chương rút gọn (8, 13, 16 đến 18, 21,
            23 và 24; phần <a href={href('ch01', '1.7')}>1.7</a> cho thấy toàn bộ thứ tự). Để chọn nội dung cần xem lại, trang
            tóm tắt của <a href={href('ch24', 'wrap')}>chương 24</a>, chương kết của sách (pp. 577–580), liệt kê
            từng module cùng các mục tiêu mà nó phục vụ.
          </p>
        }
      >
        <p>
          This is the last module in this course's order. Only the condensed chapters come after it (8, 13, 16 to 18, 21,
          23 and 24; section <a href={href('ch01', '1.7')}>1.7</a> shows the whole order). To choose what to revisit, the
          summary page of <a href={href('ch24', 'wrap')}>chapter 24</a>, the book's closing chapter (pp. 577–580), lists
          each module with the goals it serves.
        </p>
      </Translated>
    </>
  );
}
