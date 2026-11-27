import { useEffect, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  CARDS, CUDNN_FWD_ALGOS, DISPATCH_PRESETS, LIMITS, LLAMA_GUESSES, LLAMA_PATH_TEXT, METAL_BUFFER_FLOOR, MIOPEN_FWD_ALGOS,
  NAMES, PRESETS, bookGemm, chapterOutputDim, clampField, cudnnValues, duplication, flops, grade, inputPerSample,
  libraryGemm, llamaCuda, mlx, outputDimGgml, outputDimSteps, outputDimTorch, shape, slow2d, unfoldSize,
  type Check, type Layer, type LlamaPath,
} from '../../sim/convlib';

// ---------------------------------------------------------------- questions
export const questionsS194: Record<string, Question> = {
  'ch19/19.4/q-names': {
    kind: 'choice',
    id: 'ch19/19.4/q-names',
    prompt: (
      <Translated vi={<>Một lời gọi cuDNN mô tả bộ lọc của nó với <code>K = 64</code> và <code>R = S = 3</code>, và đầu vào với{' '}<code>C = 32</code>. Theo cách gọi tên của chương, lớp đó là gì?</>}>
      <>
        A cuDNN call describes its filters with <code>K = 64</code> and <code>R = S = 3</code>, and its input with{' '}
        <code>C = 32</code>. What is that layer in the chapter&rsquo;s names?
      </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="M = 64 bản đồ đầu ra, bộ lọc 3 × 3 (K = 3), C = 32 bản đồ đầu vào.">M = 64 output maps, a 3 × 3 filter (K = 3), C = 32 input maps.</Translated>, correct: true },
      { label: <Translated vi="K = 64 bản đồ đầu ra, M = 3 (cạnh bộ lọc), C = 32 bản đồ đầu vào.">K = 64 output maps, M = 3 (the filter side), C = 32 input maps.</Translated>, why: <Translated vi="Đó là giữ nguyên ký hiệu của cuDNN. Trong chương, K là cạnh của bộ lọc và M đếm số bản đồ đầu ra.">That keeps cuDNN’s letters. In the chapter, K is the side of the filter and M counts output maps.</Translated> },
      { label: <Translated vi="M = 64 bản đồ đầu ra, bộ lọc có K = 9 tap, C = 32 bản đồ đầu vào.">M = 64 output maps, a filter of K = 9 taps, C = 32 input maps.</Translated>, why: <Translated vi="R × S = 9 là số tap trên mỗi bản đồ đầu vào, tức K². K của chương là cạnh, 3.">R × S = 9 is the number of taps per input map, K². The chapter’s K is the side, 3.</Translated> },
      { label: <Translated vi="M = 32 bản đồ đầu ra, bộ lọc 3 × 3 (K = 3), C = 64 bản đồ đầu vào.">M = 32 output maps, a 3 × 3 filter (K = 3), C = 64 input maps.</Translated>, why: <Translated vi="C có cùng nghĩa trong cả hai quy ước: 32 bản đồ đầu vào. 64 bản đồ đầu ra là M của chương.">C means the same in both conventions: 32 input maps. The 64 output maps are the chapter’s M.</Translated> },
    ],
    explain: (
      <Translated vi={<p>Bảng 19.1 đổi tên hai thứ. <code>K</code> của cuDNN đếm số bản đồ đặc trưng đầu ra, tức <code>M</code> của chương; <code>R</code> và <code>S</code> là chiều cao và chiều rộng của bộ lọc, trong khi chương dùng một <code>K</code> cho bộ lọc vuông. Chú thích của bảng cảnh báo đúng điều này.</p>}>
      <p>
        Table 19.1 renames two things. cuDNN&rsquo;s <code>K</code> counts output feature maps, the chapter&rsquo;s{' '}
        <code>M</code>; <code>R</code> and <code>S</code> are the filter&rsquo;s height and width, where the chapter uses
        one <code>K</code> for a square filter. The table&rsquo;s caption warns about exactly this.
      </p>
      </Translated>
    ),
    source: 'ch. 19 §19.4, Table 19.1, p. 473', // i18n-ignore: citation
  },
  'ch19/19.4/q-padding': {
    kind: 'choice',
    id: 'ch19/19.4/q-padding',
    prompt: (
      <Translated vi={<>Sách nói phần đệm (padding) có mặt để căn chỉnh bộ nhớ tốt hơn hoặc để thực thi vector hóa. <code>pad_h</code> của cuDNN làm gì?</>}>
      <>
        The book says padding is there for better memory alignment or vectorised execution. What does cuDNN&rsquo;s{' '}
        <code>pad_h</code> do?
      </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Nó thêm pad_h hàng số 0 phía trên và phía dưới mỗi đầu vào (các ô ma (ghost cell) của chương 7), nên P thay đổi.">It adds pad_h zero rows above and below every input (chapter 7’s ghost cells), so P changes.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Nó làm tròn mỗi hàng của tensor lên một độ dài được căn chỉnh; kích thước đầu ra không đổi.">It rounds each row of the tensor up to an aligned length; the output size does not change.</Translated>,
        why: <Translated vi="Kiểu đệm đó có tồn tại bên trong cuDNN, cho Tensor Cores, và bạn không yêu cầu nó. pad_h xuất hiện trong công thức kích thước đầu ra.">Padding of that kind exists inside cuDNN, for Tensor Cores, and you do not ask for it. pad_h appears in the output-size formula.</Translated>,
      },
      {
        label: <Translated vi="Nó đệm đầu ra: P tăng thêm 2·pad_h, các số 0 được ghi quanh Y.">It pads the output: P grows by 2·pad_h, the zeros written around Y.</Translated>,
        why: <Translated vi="Các số 0 nằm quanh đầu vào. P tăng vì vừa thêm nhiều cửa sổ hơn trong đầu vào đã đệm, khoảng 2·pad_h / u hàng.">The zeros go around the input. P grows because more windows fit in the padded input, by about 2·pad_h / u rows.</Translated>,
      },
      {
        label: <Translated vi="Không làm gì trong lượt tiến (forward); nó được lượt lùi (backward) dùng khi huấn luyện.">Nothing in the forward pass; it is used by the backward pass in training.</Translated>,
        why: <Translated vi="Nó có trong công thức của lượt tiến: đầu vào được đọc tại p·u + r − pad, với số 0 bên ngoài ảnh.">It is in the forward formula: the input is read at p·u + r − pad, with zeros outside the image.</Translated>,
      },
    ],
    explain: (
      <Translated vi={<p>cuDNN ghi trong tài liệu <code>pad_h</code> là các hàng số 0 được ngầm thêm vào phía trên và phía dưới ảnh đầu vào (<code>pad_w</code> tương tự ở bên trái và bên phải). Chúng xuất hiện trong công thức kích thước đầu ra, nên chúng thay đổi{' '}<code>P</code> và <code>Q</code>. Phần đệm để căn chỉnh có tồn tại, nhưng bên trong thư viện: với dữ liệu NCHW đóng gói có số kênh không phải bội số của 8, cuDNN tự đệm các tensor để dùng Tensor Cores.</p>}>
      <p>
        cuDNN documents <code>pad_h</code> as rows of zeros implicitly added at the top and at the bottom of the input
        images (<code>pad_w</code> likewise at the left and right). They appear in the output-size formula, so they change{' '}
        <code>P</code> and <code>Q</code>. Alignment padding does exist, but inside the library: for packed NCHW data whose
        channel count is not a multiple of 8, cuDNN pads the tensors itself to use Tensor Cores.
      </p>
      </Translated>
    ),
    source: 'ch. 19 p. 473; ch19-facts A2', // i18n-ignore: citation
  },
  'ch19/19.4/q-workspace': {
    kind: 'choice',
    id: 'ch19/19.4/q-workspace',
    prompt: (
      <Translated vi={<>Thuật toán forward nào của cuDNN (<code>CUDNN_CONVOLUTION_FWD_ALGO_</code> theo sau là một tên) là phép mở rộng tường minh mà sách phản đối ở p. 467, và workspace của nó chứa gì?</>}>
      <>
        Which cuDNN forward algorithm (<code>CUDNN_CONVOLUTION_FWD_ALGO_</code> followed by a name) is the explicit
        unfolding the book argues against on p. 467, and what does its workspace hold?
      </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="GEMM; workspace chứa ma trận đầu vào đã mở rộng.">GEMM; the workspace holds the unfolded input matrix.</Translated>, correct: true },
      { label: <Translated vi="IMPLICIT_GEMM; workspace chứa các ô của ma trận đã mở rộng.">IMPLICIT_GEMM; the workspace holds tiles of the unfolded matrix.</Translated>, why: <Translated vi="IMPLICIT_GEMM không bao giờ tạo ra ma trận. Đó là Hình 19.11: các ô nằm trong bộ nhớ trên chip.">IMPLICIT_GEMM never forms the matrix. It is Fig. 19.11: tiles live in on-chip memory.</Translated> },
      { label: <Translated vi="IMPLICIT_PRECOMP_GEMM; workspace chứa ma trận đã mở rộng.">IMPLICIT_PRECOMP_GEMM; the workspace holds the unfolded matrix.</Translated>, why: <Translated vi="Workspace của nó chứa các chỉ số được tính trước, không phải ma trận.">Its workspace holds precomputed indices, not the matrix.</Translated> },
      { label: <Translated vi="DIRECT; workspace chứa tập bộ lọc.">DIRECT; the workspace holds the filter bank.</Translated>, why: <Translated vi="DIRECT hoàn toàn không làm phép nhân ma trận; nó là loại kernel mà mục 19.2 viết.">DIRECT does no matrix multiplication at all; it is the kind of kernel section 19.2 writes.</Translated> },
    ],
    explain: (
      <Translated vi={<p>Tài liệu mô tả <code>GEMM</code> là một tích ma trận tường minh cần một workspace bộ nhớ đáng kể cho ma trận dữ liệu đầu vào: chính <code>B</code> của sách, với sự trùng lặp của Eq. 19.4 và thêm một lượt qua bộ nhớ toàn cục để tạo nó. Đường chậm trên CUDA của PyTorch và đồ thị tổng quát của llama.cpp đi cùng con đường tường minh đó.</p>}>
      <p>
        The documentation describes <code>GEMM</code> as an explicit matrix product that needs a significant memory
        workspace for the matrix of input data: the book&rsquo;s <code>B</code>, with the duplication of Eq. 19.4 and an
        extra pass over global memory to build it. PyTorch&rsquo;s slow CUDA path and llama.cpp&rsquo;s generic graph take
        the same explicit route.
      </p>
      </Translated>
    ),
    source: 'ch. 19 p. 467; ch19-facts A3, D2, D4', // i18n-ignore: citation
  },
  'ch19/19.4/q-precomp': {
    kind: 'choice',
    id: 'ch19/19.4/q-precomp',
    prompt: (
      <Translated vi={<><code>IMPLICIT_PRECOMP_GEMM</code> cần một workspace mà <code>IMPLICIT_GEMM</code> không cần. Trong đó có gì, và vì sao nó có ích?</>}>
      <>
        <code>IMPLICIT_PRECOMP_GEMM</code> needs a workspace that <code>IMPLICIT_GEMM</code> does not. What is in it, and
        why would it help?
      </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Các chỉ số vào đầu vào được tính trước, để tiết kiệm các phép chia và lấy dư của Eq. 19.5 cho mỗi lần nạp.">Precomputed indices into the input, to save Eq. 19.5’s divides and remainders per load.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Một bản sao của đầu vào đã mở rộng, để có thể gọi một GEMM đã tinh chỉnh thông thường trên nó.">A copy of the unfolded input, so it can call an ordinary tuned GEMM on it.</Translated>, why: <Translated vi="Đó là GEMM, thuật toán tường minh. PRECOMP vẫn không bao giờ tạo ra ma trận.">That is GEMM, the explicit algorithm. PRECOMP still never forms the matrix.</Translated> },
      { label: <Translated vi="Các trọng số bộ lọc được sắp xếp lại theo thứ tự ô của GEMM.">The filter weights rearranged into the GEMM’s tile order.</Translated>, why: <Translated vi="Tài liệu nói workspace chứa các chỉ số giúp tạo ma trận đầu vào một cách ngầm định.">The documentation says the workspace holds indices that help build the input matrix implicitly.</Translated> },
      { label: <Translated vi="Các tổng riêng phần từ một phép rút gọn được tách trên C·K·K, được gộp lại sau GEMM.">Partial sums from a split reduction over C·K·K, combined after the GEMM.</Translated>, why: <Translated vi="Mô tả của nó chỉ nêu các chỉ số được tính trước, không gì khác.">Its description names precomputed indices, nothing else.</Translated> },
    ],
    explain: (
      <Translated vi={<p>cuDNN nói nó cần một ít workspace để tính trước các chỉ số cho việc tạo ngầm ma trận đầu vào. Eq. 19.5 tốn một phép chia nguyên và một phép lấy dư cho mỗi phần tử được nạp, và hướng dẫn best practices của CUDA nói mỗi phép có thể biên dịch ra tới 20 lệnh. CUTLASS ghi lại cùng sự đánh đổi đó: các độ lệch con trỏ được tính trước trên host, và một phép chia-lấy-dư nhanh cho phần còn lại. Vậy câu &ldquo;computes the mapping dynamically&rdquo; (tính ánh xạ một cách động) của sách mô tả một trong ba thuật toán dựa trên GEMM của cuDNN.</p>}>
      <p>
        cuDNN says it needs some workspace to precompute indices for the implicit construction of the input matrix.
        Eq. 19.5 costs an integer divide and a remainder per loaded element, and CUDA&rsquo;s best-practices guide says
        each can compile to up to 20 instructions. CUTLASS documents the same trade: pointer deltas precomputed on the
        host, and a fast divide-and-modulo for what is left. So the book&rsquo;s &ldquo;computes the mapping
        dynamically&rdquo; describes one of cuDNN&rsquo;s three GEMM-based algorithms.
      </p>
      </Translated>
    ),
    source: 'ch. 19 p. 474; ch19-facts A3, A4, A7, A8', // i18n-ignore: citation
  },
  'ch19/19.4/q-tf32': {
    kind: 'choice',
    id: 'ch19/19.4/q-tf32',
    prompt: (
      <Translated vi={<>Kernel tích chập FP32 của bạn qua được kiểm tra với một tham chiếu CPU, vậy mà trên GPU NVIDIA nó lệch khỏi{' '}<code>torch.nn.functional.conv2d</code> khoảng 1e-3 tương đối. Bạn kiểm tra điều gì trước tiên?</>}>
      <>
        Your FP32 convolution kernel passes against a CPU reference, yet on an NVIDIA GPU it differs from{' '}
        <code>torch.nn.functional.conv2d</code> by about 1e-3 relative. What do you check first?
      </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Liệu cuDNN có chạy tích chập ở TF32 không, điều PyTorch cho phép theo mặc định.">Whether cuDNN ran the convolution in TF32, which PyTorch allows by default.</Translated>, correct: true },
      { label: <Translated vi="Liệu cuDNN có lật bộ lọc không, tính tích chập thật thay vì tương quan chéo (cross-correlation).">Whether cuDNN flipped the filter, computing a true convolution instead of cross-correlation.</Translated>, why: <Translated vi="PyTorch chọn tương quan chéo, công thức của sách. Một bộ lọc bị lật sẽ cho những con số hoàn toàn khác, chứ không phải một sai số tương đối nhỏ.">PyTorch selects cross-correlation, the book’s formula. A flipped filter would give different numbers altogether, not a small relative error.</Translated> },
      {
        label: <Translated vi="Liệu cudnn.benchmark có chọn một thuật toán không tất định không.">Whether cudnn.benchmark chose a non-deterministic algorithm.</Translated>,
        why: <Translated vi="cuDNN ghi trong tài liệu mọi thuật toán forward đều tất định. Chọn thuật toán khác thay đổi tốc độ và phạm vi hỗ trợ.">cuDNN documents every forward algorithm as deterministic. Choosing another changes speed and support.</Translated>,
      },
      {
        label: <Translated vi="Không gì cả: chỉ riêng thứ tự cộng đã giải thích được.">Nothing: summation order alone explains it.</Translated>,
        why: <Translated vi="Thứ tự đúng là thay đổi vài bit cuối (phụ lục A), nên bạn so sánh với một dung sai. Hãy loại trừ độ chính xác thấp hơn trước.">Order does change the last bits (appendix A), so you compare with a tolerance. Rule out a lower precision first.</Translated>,
      },
    ],
    explain: (
      <Translated vi={<p><code>torch.backends.cudnn.allow_tf32</code> mặc định là True, nên các tích chập cuDNN trên tensor FP32 có thể chạy ở TF32. PyTorch nói cờ này sẽ bị ngừng khuyến nghị để thay bằng{' '}<code>torch.backends.cudnn.conv.fp32_precision</code> (&ldquo;ieee&rdquo; hoặc &ldquo;tf32&rdquo;), mà giá trị mặc định không được trang của nó nêu, nên hãy in ra giá trị bạn nhận được. Trên Apple, <code>MPSCNNConvolution</code> dùng trọng số FP16 trừ khi nguồn dữ liệu của nó chọn FP32.</p>}>
      <p>
        <code>torch.backends.cudnn.allow_tf32</code> defaults to True, so cuDNN convolutions on FP32 tensors may run in
        TF32. PyTorch says the flag is going to be deprecated in favour of{' '}
        <code>torch.backends.cudnn.conv.fp32_precision</code> (&ldquo;ieee&rdquo; or &ldquo;tf32&rdquo;), whose default
        its page does not state, so print what you got. On Apple, <code>MPSCNNConvolution</code> uses FP16 weights unless
        its data source opts into FP32.
      </p>
      </Translated>
    ),
    source: 'ch19-facts A3, A6, C4', // i18n-ignore: citation
  },
  'ch19/19.4/q-find': {
    kind: 'choice',
    id: 'ch19/19.4/q-find',
    prompt: (
      <Translated vi={<>Bước Find của MIOpen và <code>torch.backends.cudnn.benchmark</code> của PyTorch có điểm gì chung?</>}>
      <>
        What do MIOpen&rsquo;s Find step and PyTorch&rsquo;s <code>torch.backends.cudnn.benchmark</code> have in common?
      </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Cả hai đều đo thời gian vài thuật toán trên mỗi hình dạng mới và giữ lại cái nhanh nhất.">Both time several algorithms on each new shape and keep the fastest one.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Cả hai đều làm kết quả tất định, để hai lần chạy cùng một mô hình cho ra cùng các bit.">Both make results deterministic, so two runs of the same model give the same bits.</Translated>, why: <Translated vi="PyTorch có một cờ riêng cho điều đó, và việc đo benchmark thậm chí có thể chọn các thuật toán khác nhau ở các lần chạy khác nhau.">PyTorch has a separate flag for that, and benchmarking may even pick different algorithms on different runs.</Translated> },
      { label: <Translated vi="Cả hai đều là cách tùy chọn để tiết kiệm bộ nhớ bằng cách chọn thuật toán có workspace nhỏ hơn.">Both are optional ways to save memory by picking algorithms with smaller workspaces.</Translated>, why: <Translated vi="Find là bắt buộc trước miopenConvolutionForward trừ khi bạn dùng immediate mode, và không cái nào liên quan đến bộ nhớ.">Find is mandatory before miopenConvolutionForward unless you use immediate mode, and neither is about memory.</Translated> },
      { label: <Translated vi="Cả hai đều chọn bố cục dữ liệu của các tensor.">Both choose the data layout of the tensors.</Translated>, why: <Translated vi="Bố cục nằm trong các tensor descriptor bạn truyền vào. Hai thứ này chọn thuật toán.">The layout is in the tensor descriptors you pass. These choose the algorithm.</Translated> },
    ],
    explain: (
      <Translated vi={<p>Khi đặt <code>benchmark</code>, cuDNN đo benchmark vài thuật toán tích chập cho mỗi bộ kích thước mới và chọn cái nhanh nhất. <code>miopenFindConvolutionForwardAlgorithm</code> của MIOpen biên dịch và đo thời gian các solver của nó rồi trả về chúng đã sắp xếp; nó là bắt buộc trước <code>miopenConvolutionForward</code>, và nó lưu kết quả vào bộ nhớ đệm (cache) trên đĩa để các phiên sau khởi động nhanh hơn. Immediate mode bỏ qua Find và tra cứu câu trả lời thay vào đó.</p>}>
      <p>
        With <code>benchmark</code> set, cuDNN benchmarks several convolution algorithms for each new set of sizes and
        selects the fastest. MIOpen&rsquo;s <code>miopenFindConvolutionForwardAlgorithm</code> compiles and times its
        solvers and returns them sorted; it is mandatory before <code>miopenConvolutionForward</code>, and it caches its
        results on disk so later sessions start faster. Immediate mode skips Find and looks the answer up instead.
      </p>
      </Translated>
    ),
    source: 'ch19-facts A6, B3', // i18n-ignore: citation
  },
  'ch19/19.4/q-dispatch': {
    kind: 'choice',
    id: 'ch19/19.4/q-dispatch',
    prompt: (
      <Translated vi={<>Backend CUDA của llama.cpp nhận một tích chập 1 × 1 với stride 1 và không đệm, từ 256 lên 512 kênh trên một bản đồ 14 × 14. Nó chạy gì?</>}>
      <>
        llama.cpp&rsquo;s CUDA backend receives a 1 × 1 convolution with stride 1 and no padding, from 256 to 512
        channels on a 14 × 14 map. What does it run?
      </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Một GEMM của cuBLAS trực tiếp trên đầu vào.">A cuBLAS GEMM directly on the input.</Translated>, correct: true },
      { label: <Translated vi="Một phép mở rộng tường minh thành B, rồi một GEMM của cuBLAS.">An explicit unfold into B, then a cuBLAS GEMM.</Translated>, why: <Translated vi="Với K = 1, phép mở rộng sẽ sao chép nguyên đầu vào (Eq. 19.4 cho đúng 1), nên không có gì để tạo.">With K = 1 the unfold would copy the input unchanged (Eq. 19.4 gives exactly 1), so there is nothing to build.</Translated> },
      { label: <Translated vi="Kernel implicit-GEMM của nó, Hình 19.11 trong production.">Its implicit-GEMM kernel, Fig. 19.11 in production.</Translated>, why: <Translated vi="Đó là đường tổng quát. Hình dạng này có phép kiểm tra riêng: không có gì để gom.">That is the general path. This shape has a test of its own: there is nothing to gather.</Translated> },
      { label: <Translated vi="Kernel trực tiếp, vì bản đồ chỉ có 196 pixel.">The direct kernel, since the map has only 196 pixels.</Translated>, why: <Translated vi="Kernel trực tiếp dành cho các phép rút gọn ngắn trên bản đồ nhỏ; ở đây C·K·K = 256 không dưới 64.">The direct kernel is for short reductions on small maps; here C·K·K = 256 is not below 64.</Translated> },
    ],
    explain: (
      <Translated vi={<p>Với K = 1, stride 1 và không đệm, Eq. 19.5 rút gọn thành <code>B[u, v] = X[n, u, v / W_out, v % W_out]</code>:{' '}<code>B</code> là <code>X</code> nhìn như một ma trận C × H·W. Phép kiểm tra <code>pointwise</code> của llama.cpp gửi trường hợp này tới <code>cublasGemmEx</code>, và đường chậm của PyTorch bỏ qua bước mở rộng với cùng các hình dạng đó. Một tích chập 1 × 1 là một lớp tuyến tính áp dụng tại mọi pixel, chính là điều các phép chiếu của một LLM làm với mọi token.</p>}>
      <p>
        With K = 1, stride 1 and no padding, Eq. 19.5 reduces to <code>B[u, v] = X[n, u, v / W_out, v % W_out]</code>:{' '}
        <code>B</code> is <code>X</code> viewed as a C × H·W matrix. llama.cpp&rsquo;s <code>pointwise</code> test sends
        this case to <code>cublasGemmEx</code>, and PyTorch&rsquo;s slow path skips its unfold for the same shapes. A 1 × 1
        convolution is a linear layer applied at every pixel, which is what an LLM&rsquo;s projections do to every
        token.
      </p>
      </Translated>
    ),
    source: 'ch19-facts D1, D4; notes, "the K = 1 case"',
    vi: { source: 'ch19-facts D1, D4; notes, "the K = 1 case" (trường hợp K = 1)' },
  },
};

// ---------------------------------------------------------------- helpers
const n = (v: number) => v.toLocaleString('en-US');
const mib = (bytes: number) => `${(bytes / 2 ** 20).toLocaleString('en-US', { maximumFractionDigits: 1 })} MiB`; // i18n-ignore: byte unit
const ratio = (v: number) => v.toLocaleString('en-US', { maximumFractionDigits: 2 });
const gflop = (v: number) => (v >= 1e9 ? `${(v / 1e9).toLocaleString('en-US', { maximumFractionDigits: 2 })} GFLOP` : `${n(v)} FLOP`); // i18n-ignore: units
const sameLayer = (a: Layer, b: Layer) => (Object.keys(a) as (keyof Layer)[]).every((k) => a[k] === b[k]);
const describeLayer = (l: Layer) =>
  tr(`N ${l.N}, C ${l.C}, ${l.H} × ${l.W} input, M ${l.M}, ${l.K} × ${l.K} filter, pad ${l.pad}, stride ${l.stride}, dilation ${l.dilation}`, `N ${l.N}, C ${l.C}, đầu vào ${l.H} × ${l.W}, M ${l.M}, bộ lọc ${l.K} × ${l.K}, pad ${l.pad}, stride ${l.stride}, dilation ${l.dilation}`);

/** A number box that lets you type freely and commits clamped integers to the layer. */
function NumField({ k, label, hint, value, onChange }: { k: keyof Layer; label: string; hint: string; value: number; onChange: (v: number) => void }) {
  const [text, setText] = useState(String(value));
  const [focus, setFocus] = useState(false);
  useEffect(() => {
    if (!focus) setText(String(value));
  }, [value, focus]);
  return (
    <label className="field" style={{ minWidth: 0 }}>
      <span>
        <b className="mono">{label}</b> <span className="muted">{hint}</span>
      </span>
      <input
        type="number"
        min={LIMITS[k][0]}
        max={LIMITS[k][1]}
        step={1}
        value={text}
        data-field={k}
        aria-label={`${label}, ${hint}`}
        style={{ width: '6.5rem' }}
        onFocus={() => setFocus(true)}
        onBlur={() => {
          setFocus(false);
          setText(String(value));
        }}
        onChange={(e) => {
          setText(e.target.value);
          const v = Number(e.target.value);
          if (e.target.value.trim() !== '' && Number.isFinite(v)) onChange(clampField(k, v));
        }}
      />
    </label>
  );
}

const BOOK_FIELDS: { k: keyof Layer; hint: string }[] = [
  { k: 'N', get hint() { return tr('samples', 'số mẫu'); } },
  { k: 'C', get hint() { return tr('input maps', 'số bản đồ đầu vào'); } },
  { k: 'H', get hint() { return tr('input height', 'chiều cao đầu vào'); } },
  { k: 'W', get hint() { return tr('input width', 'chiều rộng đầu vào'); } },
  { k: 'M', get hint() { return tr('output maps', 'số bản đồ đầu ra'); } },
  { k: 'K', get hint() { return tr('filter side', 'cạnh bộ lọc'); } },
];
const LIB_FIELDS: { k: keyof Layer; hint: string }[] = [
  { k: 'pad', get hint() { return tr('zeros per side', 'số 0 mỗi bên'); } },
  { k: 'stride', get hint() { return tr('step', 'bước'); } },
  { k: 'dilation', get hint() { return tr('tap spacing', 'khoảng cách giữa các tap'); } },
];

// ---------------------------------------------------------------- explorer 1: describe a layer to a library
function NamesTable({ layer }: { layer: Layer }) {
  const values = cudnnValues(layer);
  return (
    <table className="data" style={{ margin: 0 }} data-names>
      <thead>
        <tr>
          <th>cuDNN</th>
          <th><Translated vi="Nghĩa">Means</Translated></th>
          <th><Translated vi="Chương">Chapter</Translated></th>
          <th className="num"><Translated vi="Giá trị">Value</Translated></th>
        </tr>
      </thead>
      <tbody>
        {NAMES.filter((r) => r.cudnn in values).map((r) => (
          <tr key={r.cudnn} data-name={r.cudnn}>
            <td>
              <code>{r.cudnn}</code>
            </td>
            <td>
              {r.meaning}
              {r.from !== 'Table 19.1' ? <span className="muted"><Translated vi=" (không có trong Bảng 19.1)"> (not in Table 19.1)</Translated></span> : null}
            </td>
            <td data-chapter={r.chapter ?? ''}>
              {r.chapter ? (
                r.cudnn === 'K' ? (
                  <b>
                    <Translated vi={<><code>{r.chapter}</code>, không phải K của chương</>}>
                    <code>{r.chapter}</code>, not the chapter&rsquo;s K
                    </Translated>
                  </b>
                ) : (
                  <code>{r.chapter}</code>
                )
              ) : (
                <span className="muted"><Translated vi={<>không có: {r.none}</>}>none: {r.none}</Translated></span>
              )}
            </td>
            <td className="num" data-value={values[r.cudnn]}>
              {n(values[r.cudnn])}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function OutputSize({ layer }: { layer: Layer }) {
  const sh = shape(layer);
  const axis = (name: 'P' | 'Q', input: number, inName: 'H' | 'W', padName: string, strideName: string) => {
    const st = outputDimSteps(input, layer.K, layer.pad, layer.stride, layer.dilation);
    const filt = name === 'P' ? 'R' : 'S';
    return (
      <div data-axis={name} style={{ marginBottom: '0.6rem' }}>
        <div className="mono" style={{ fontSize: '0.86rem' }}>
          {name} = 1 + ({inName} + 2·{padName} − (({filt} − 1)·dilation + 1)) / {strideName}{/* i18n-ignore: formula */}
        </div>
        <div className="mono" style={{ fontSize: '0.86rem' }} data-steps>
          {st.quotient == null ? (
            <>
              <Translated vi={<>= không có đầu ra: các tap trải trên {st.span}, đầu vào đã đệm có {input + 2 * layer.pad}</>}>= no output: the taps span {st.span}, the padded input has {input + 2 * layer.pad}</Translated>
            </>
          ) : (
            <>
              = 1 + ({input} + 2·{layer.pad} − (({layer.K} − 1)·{layer.dilation} + 1)) / {layer.stride} = 1 + {st.room} /{' '}
              {layer.stride} = <b data-result>{st.result}</b>
            </>
          )}
        </div>
      </div>
    );
  };
  const torch = outputDimTorch(layer.H, layer.K, layer.pad, layer.stride, layer.dilation);
  const ggml = outputDimGgml(layer.H, layer.K, layer.pad, layer.stride, layer.dilation);
  const chap = chapterOutputDim(layer.H, layer.K);
  return (
    <div className="panel" data-outsize data-p={sh.P} data-q={sh.Q}>
      <p style={{ margin: '0 0 0.5rem' }}>
        <b><Translated vi="Kích thước đầu ra">Output size</Translated></b> <span className="muted"><Translated vi="(công thức của cuDNN, phép chia nguyên)">(cuDNN&rsquo;s formula, integer division)</Translated></span>
      </p>
      {axis('P', layer.H, 'H', 'pad_h', 'u')}
      {layer.W !== layer.H ? axis('Q', layer.W, 'W', 'pad_w', 'v') : null}
      {sh.valid ? (
        <>
          <p style={{ margin: '0 0 0.4rem' }} data-pq>
            <Translated vi={<>Đầu ra: <b>{sh.P} × {sh.Q}</b> mỗi bản đồ.</>}>Output: <b>{sh.P} × {sh.Q}</b> per map.</Translated>
            {layer.W === layer.H ? <span className="muted"><Translated vi=" Q cũng vậy, vì W = H."> Q is the same, since W = H.</Translated></span> : null}
          </p>
          <p style={{ margin: '0 0 0.4rem' }} data-others>
            <Translated vi={<>Công thức của PyTorch cho {torch} và của llama.cpp cho {ggml}.</>}>
            PyTorch&rsquo;s formula gives {torch} and llama.cpp&rsquo;s gives {ggml}.
            </Translated>
          </p>
          <p style={{ margin: 0 }} data-chapter-formula={sh.chapterApplies ? 'applies' : 'differs'}>
            <Translated vi={<>H − K + 1 của chương = {chap}:</>}>The chapter&rsquo;s H − K + 1 = {chap}:</Translated>{' '}
            {sh.chapterApplies ? (
              <span className="chip ok"><Translated vi="như nhau, khi không đệm, stride 1, dilation 1">the same, with no padding, stride 1, dilation 1</Translated></span>
            ) : chap === sh.P ? (
              <span className="chip ok"><Translated vi="ở đây như nhau, do tình cờ của các thiết lập">the same here, by coincidence of the settings</Translated></span>
            ) : (
              <span className="chip fail"><Translated vi="sai với lớp này, vốn có đệm, stride hoặc dilation">wrong for this layer, which pads, strides or dilates</Translated></span>
            )}
          </p>
        </>
      ) : null}
    </div>
  );
}

function GemmPanel({ layer }: { layer: Layer }) {
  const b = bookGemm(layer);
  const g = libraryGemm(layer);
  return (
    <div className="panel" data-gemm>
      <p style={{ margin: '0 0 0.5rem' }}>
        <b><Translated vi="Dưới dạng một phép nhân ma trận">As a matrix multiplication</Translated></b>
      </p>
      <p style={{ margin: '0 0 0.5rem' }} data-book-gemm={`${b.rows}x${b.inner}x${b.cols}x${b.count}`}>
        <span className="muted"><Translated vi="Hướng của sách, mỗi mẫu (pp. 464–466):">The book&rsquo;s orientation, per sample (pp. 464–466):</Translated></span>
        <Translated vi={<><br />F ({n(b.rows)} × {n(b.inner)}) nhân B ({n(b.inner)} × {n(b.cols)}) cho Y ({n(b.rows)} × {n(b.cols)}),{' '}{b.count === 1 ? 'một lần' : `${n(b.count)} lần, mỗi mẫu một lần`}.</>}>
        <br />F ({n(b.rows)} × {n(b.inner)}) times B ({n(b.inner)} × {n(b.cols)}) gives Y ({n(b.rows)} × {n(b.cols)}),{' '}
        {b.count === 1 ? 'once' : `${n(b.count)} times, one per sample`}.
        </Translated>
      </p>
      <p style={{ margin: '0 0 0.5rem' }} data-lib-gemm={`${g.rows}x${g.inner}x${g.cols}`}>
        <span className="muted"><Translated vi="Hướng của thư viện mà CUTLASS ghi trong tài liệu, một tích cho cả batch:">The library orientation CUTLASS documents, one product for the batch:</Translated></span>
        <br />
        <Translated vi={<>N·P·Q × C·R·S nhân C·R·S × K: {n(g.rows)} × {n(g.inner)} nhân {n(g.inner)} × {n(g.cols)}, các pixel đầu ra là các hàng.</>}>
        N·P·Q × C·R·S times C·R·S × K: {n(g.rows)} × {n(g.inner)} times {n(g.inner)} × {n(g.cols)}, output pixels as
        rows.
        </Translated>
      </p>
      <p style={{ margin: 0 }} data-flops={flops(layer)}>
        <Translated vi={<>Dù cách nào, {gflop(flops(layer))} cho cả lớp.</>}>Either way, {gflop(flops(layer))} for the layer.</Translated>
      </p>
    </div>
  );
}

function UnfoldPanel({ layer }: { layer: Layer }) {
  const sh = shape(layer);
  const u = unfoldSize(layer);
  const d = duplication(layer);
  const b = bookGemm(layer);
  const reading =
    layer.K === 1 && layer.stride === 1 && layer.pad === 0
      ? tr('B is X itself: with a 1 × 1 filter there is nothing to unfold.', 'B chính là X: với bộ lọc 1 × 1 không có gì để mở rộng.')
      : d > 1
        ? tr('Overlapping windows copy pixels into several columns (zeros from padding count too).', 'Các cửa sổ chồng lấn sao chép pixel vào nhiều cột (các số 0 từ phần đệm cũng được tính).')
        : d === 1
          ? tr('No pixel is copied twice: the windows tile the input without overlapping.', 'Không pixel nào bị sao chép hai lần: các cửa sổ lát kín đầu vào mà không chồng lấn.')
          : tr('Some pixels are never read: the stride skips past them.', 'Một số pixel không bao giờ được đọc: stride nhảy qua chúng.');
  return (
    <div className="panel" data-unfold data-ratio={d}>
      <p style={{ margin: '0 0 0.5rem' }}>
        <b><Translated vi="Đầu vào đã mở rộng B">The unfolded input B</Translated></b>
      </p>
      <p style={{ margin: '0 0 0.4rem' }}>
        <Translated vi={<>Mỗi mẫu: C·K·K × P·Q = {n(b.inner)} × {n(sh.P * sh.Q)} = {n(u.perSample)} phần tử, so với {n(inputPerSample(layer))} trong X.</>}>
        Per sample: C·K·K × P·Q = {n(b.inner)} × {n(sh.P * sh.Q)} = {n(u.perSample)} elements, against {n(inputPerSample(layer))} in X.
        </Translated>
      </p>
      <p style={{ margin: '0 0 0.4rem' }} data-dup>
        <Translated vi={<>Hệ số trùng lặp <b>{ratio(d)}×</b>. {reading}</>}>Duplication <b>{ratio(d)}×</b>. {reading}</Translated>
      </p>
      <p style={{ margin: '0 0 0.4rem' }}>
        <Translated vi={<>Mở rộng cho cả batch cùng lúc: {n(u.batch)} phần tử, {mib(u.batchBytes)} ở FP32.</>}>
        Unfolded for the whole batch at once: {n(u.batch)} elements, {mib(u.batchBytes)} in FP32.
        </Translated>
      </p>
      <p style={{ margin: 0 }} data-limits={`${u.overInt32 ? 'int32' : ''}${u.overMetalFloor ? ' metal' : ''}`.trim()}>
        {u.overInt32 ? (
          <span className="chip fail"><Translated vi="nhiều phần tử hơn mức một int 32-bit có thể đánh chỉ số">more elements than a 32-bit int can index</Translated></span>
        ) : (
          <span className="muted"><Translated vi="Nằm trong tầm của chỉ số 32-bit.">Within reach of 32-bit indices.</Translated></span>
        )}{' '}
        {u.overMetalFloor ? (
          <span className="chip fail"><Translated vi={<>vượt mức {n(METAL_BUFFER_FLOOR / 2 ** 20)} MiB mà Apple bảo đảm cho một buffer Metal</>}>over the {n(METAL_BUFFER_FLOOR / 2 ** 20)} MiB Apple guarantees for one Metal buffer</Translated></span>
        ) : (
          <span className="muted"><Translated vi={<>Dưới mức {n(METAL_BUFFER_FLOOR / 2 ** 20)} MiB mà Apple bảo đảm cho một buffer Metal.</>}>Under the {n(METAL_BUFFER_FLOOR / 2 ** 20)} MiB Apple guarantees for one Metal buffer.</Translated></span>
        )}
      </p>
    </div>
  );
}

function Flashcards() {
  const [i, setI] = useState(0);
  const [pick, setPick] = useState<string | null>(null);
  const [right, setRight] = useState(0);
  const [done, setDone] = useState(0);
  const card = CARDS[i];
  const ok = pick != null && grade(card, pick);
  const choose = (o: string) => {
    if (pick != null) return;
    setPick(o);
    setDone((x) => x + 1);
    if (grade(card, o)) setRight((x) => x + 1);
  };
  const next = () => {
    if (i + 1 < CARDS.length) setI(i + 1);
    else {
      setI(0);
      setRight(0);
      setDone(0);
    }
    setPick(null);
  };
  const other = card.from === 'chapter' ? tr('cuDNN’s', 'của cuDNN') : tr('the chapter’s', 'của chương');
  return (
    <div className="panel" data-card={card.id}>
      <p className="muted" style={{ margin: '0 0 0.4rem' }}>
        <Translated vi={<>Thẻ {i + 1} trên {CARDS.length}. <span data-score>Đến giờ đúng {right} trên {done}.</span></>}>
        Card {i + 1} of {CARDS.length}. <span data-score>{right} of {done} right so far.</span>
        </Translated>
      </p>
      <p style={{ margin: '0 0 0.7rem' }}>
        <Translated vi={<>Ký hiệu{' '}<b className="mono" style={{ fontSize: '1.5rem', padding: '0 0.3rem', border: '1px solid var(--ink)', borderRadius: 3 }}>{card.symbol}</b>{' '}{card.from === 'chapter' ? 'của chương' : 'của cuDNN'} là gì theo cách gọi tên {other}?</>}>
        {card.from === 'chapter' ? 'The chapter’s' : 'cuDNN’s'}{' '}
        <b className="mono" style={{ fontSize: '1.5rem', padding: '0 0.3rem', border: '1px solid var(--ink)', borderRadius: 3 }}>
          {card.symbol}
        </b>{' '}
        is what in {other} names?
        </Translated>
      </p>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        {card.options.map((o) => (
          <button key={o} aria-pressed={pick === o} disabled={pick != null && pick !== o} onClick={() => choose(o)} data-option={o}>
            {tr(o, ({ 'no counterpart': 'không có tương ứng', 'R and S': 'R và S', 'P and Q': 'P và Q', 'u and v': 'u và v' } as Record<string, string>)[o] ?? o)}
          </button>
        ))}
      </div>
      {pick != null ? (
        <p style={{ margin: '0 0 0.7rem' }} data-verdict={ok ? 'right' : 'wrong'}>
          <span className={'chip ' + (ok ? 'ok' : 'fail')}>{ok ? tr('Right', 'Đúng') : tr(`No: ${card.answer}`, `Không: ${({ 'no counterpart': 'không có tương ứng', 'R and S': 'R và S', 'P and Q': 'P và Q', 'u and v': 'u và v' } as Record<string, string>)[card.answer] ?? card.answer}`)}</span> {card.why}
        </p>
      ) : null}
      <button className="primary" onClick={next} disabled={pick == null}>
        {i + 1 < CARDS.length ? tr('Next card', 'Thẻ tiếp') : tr('Start again', 'Bắt đầu lại')}
      </button>
    </div>
  );
}

function LayerFields({ layer, setLayer }: { layer: Layer; setLayer: (l: Layer) => void }) {
  const field = (f: { k: keyof Layer; hint: string }) => (
    <NumField key={f.k} k={f.k} label={f.k} hint={f.hint} value={layer[f.k]} onChange={(v) => setLayer({ ...layer, [f.k]: v })} />
  );
  return (
    <>
      <p className="ui muted" style={{ margin: '0 0 0.3rem', fontSize: '0.85rem' }}>
        <Translated vi={<>Cách gọi tên của chương</>}>
        The chapter&rsquo;s names
        </Translated>
      </p>
      <div className="row" style={{ marginBottom: '0.7rem', flexWrap: 'wrap' }} data-book-fields>
        {BOOK_FIELDS.map(field)}
      </div>
      <p className="ui muted" style={{ margin: '0 0 0.3rem', fontSize: '0.85rem' }}>
        <Translated vi={<>Chỉ có ở thư viện: không có trong các kernel của sách</>}>
        Library only: not in the book&rsquo;s kernels
        </Translated>
      </p>
      <div className="row" style={{ marginBottom: '0.9rem', flexWrap: 'wrap' }} data-lib-fields>
        {LIB_FIELDS.map(field)}
      </div>
    </>
  );
}

function DescribeBench({ layer, setLayer }: { layer: Layer; setLayer: (l: Layer) => void }) {
  const [mode, setMode] = useState<'explore' | 'cards'>('explore');
  const presetId = PRESETS.find((p) => sameLayer(p.layer, layer))?.id ?? '';
  const sh = shape(layer);
  return (
    <Bench
      title={tr("Describe a layer to a library", 'Mô tả một lớp cho thư viện')}
      task={tr("Find cuDNN’s name for M, a setting that changes P without touching H or K, and a stride that ends all duplication.", 'Tìm tên cuDNN cho M, một thiết lập thay đổi P mà không động đến H hay K, và một stride chấm dứt mọi sự trùng lặp.')}
      foot={
        <>
          <Translated vi={<>Được tính bởi một mô hình đã kiểm thử (src/sim/convlib.ts) từ Bảng 19.1 và công thức được ghi trong tài liệu của cuDNN (ch19-facts A2, A7, C2). Kích thước là số phần tử; không có gì ở đây là phép đo.</>}>
          Computed by a tested model (src/sim/convlib.ts) from Table 19.1 and cuDNN&rsquo;s documented formula
          (ch19-facts A2, A7, C2). Sizes are counts of elements; nothing here is a measurement.
          </Translated>
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', flexWrap: 'wrap' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Cấu hình có sẵn">Preset</Translated>
          <select
            aria-label={tr("Preset", 'Cấu hình có sẵn')}
            value={presetId}
            onChange={(e) => {
              const p = PRESETS.find((x) => x.id === e.target.value);
              if (p) setLayer(p.layer);
            }}
            style={{ maxWidth: '100%' }}
          >
            <option value="" disabled>
              {tr('your own layer', 'lớp của bạn')}
            </option>
            {PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <span className="seg" role="group" aria-label={tr("Mode", 'Chế độ')} style={{ alignSelf: 'flex-end' }}>
          <button aria-pressed={mode === 'explore'} onClick={() => setMode('explore')}>
            <Translated vi="Khám phá">Explore</Translated>
          </button>
          <button aria-pressed={mode === 'cards'} onClick={() => setMode('cards')}>
            <Translated vi="Thẻ ghi nhớ">Flashcards</Translated>
          </button>
        </span>
      </div>
      {mode === 'cards' ? (
        <Flashcards />
      ) : (
        <>
          <LayerFields layer={layer} setLayer={setLayer} />
          <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)' }}>
            <div>
              <NamesTable layer={layer} />
              <p className="muted" style={{ fontSize: '0.85rem', margin: '0.5rem 0 0' }}>
                <Translated vi={<>Sách gọi tensor đầu vào của cuDNN là <code>D</code> (<code>X</code> của chương), tensor bộ lọc là <code>F</code>, và đầu ra là <code>O</code> (<code>Y</code> của chương).</>}>
                The book calls cuDNN&rsquo;s input tensor <code>D</code> (the chapter&rsquo;s <code>X</code>), its filter
                tensor <code>F</code>, and its output <code>O</code> (the chapter&rsquo;s <code>Y</code>).
                </Translated>
              </p>
            </div>
            <div style={{ display: 'grid', gap: '0.8rem' }}>
              <OutputSize layer={layer} />
              {sh.valid ? (
                <>
                  <GemmPanel layer={layer} />
                  <UnfoldPanel layer={layer} />
                </>
              ) : (
                <p className="panel" style={{ margin: 0 }} data-invalid>
                  <Translated vi={<><span className="chip fail">không có đầu ra</span> Các tap của bộ lọc trải trên nhiều hàng hoặc cột hơn đầu vào đã đệm có, nên không có gì để nhân.</>}>
                  <span className="chip fail">no output</span> The filter&rsquo;s taps span more rows or columns than
                  the padded input has, so there is nothing to multiply.
                  </Translated>
                </p>
              )}
            </div>
          </div>
        </>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer 2: which kernel would they run?
function Checks({ checks }: { checks: Check[] }) {
  return (
    <ul style={{ margin: '0 0 0.5rem', paddingLeft: '1.1rem' }}>
      {checks.map((c) => (
        <li key={c.label} data-check={c.label} data-holds={c.holds ? 1 : 0}>
          {c.label} <span className="muted"><Translated vi={<>(ở đây <span className="mono">{c.shown}</span>)</>}>(here <span className="mono">{c.shown}</span>)</Translated></span>:{' '}
          {c.holds ? <b><Translated vi="đúng">holds</Translated></b> : <span className="muted"><Translated vi="không đúng">does not hold</Translated></span>}
        </li>
      ))}
    </ul>
  );
}

function BackendCard({ id, title, file, children }: { id: string; title: string; file?: string; children: ReactNode }) {
  return (
    <div className="panel" data-backend={id}>
      <p style={{ margin: '0 0 0.4rem' }}>
        <b>{title}</b>
        {file ? (
          <>
            {' '}
            <span className="muted" style={{ fontSize: '0.85rem' }}>
              ({file})
            </span>
          </>
        ) : null}
      </p>
      {children}
    </div>
  );
}

function DispatchBench({ layer, setLayer }: { layer: Layer; setLayer: (l: Layer) => void }) {
  const [presetId, setPresetId] = useState<string | null>(null);
  const [guess, setGuess] = useState<LlamaPath | null>(null);
  // Nothing is revealed until the first prediction: the page's starting layer is one of the presets.
  const [started, setStarted] = useState(false);
  const active = presetId ? DISPATCH_PRESETS.find((p) => p.id === presetId) : undefined;
  const onPreset = active != null && sameLayer(active.layer, layer);
  const pending = onPreset && guess == null;
  const sh = shape(layer);
  const llama = llamaCuda(layer);
  const torch = slow2d(layer);
  const apple = mlx(layer);
  const b = bookGemm(layer);
  return (
    <Bench
      title={tr("Which kernel would they run?", 'Chúng sẽ chạy kernel nào?')}
      task={tr("Pick a preset and predict llama.cpp’s choice before you see it; then change the layer above and watch the rules.", 'Chọn một cấu hình có sẵn và dự đoán lựa chọn của llama.cpp trước khi xem; rồi thay đổi lớp ở trên và quan sát các quy tắc.')}
      foot={
        <>
          <Translated vi={<>Các điều kiện về hình dạng như được đọc trong mã nguồn của từng dự án ngày 2026-11-20 (ch19-facts D1, D3, D4), trong một mô hình đã kiểm thử (src/sim/convlib.ts). Kiểu dữ liệu, kiểm tra phần cứng và mọi thứ tệp facts không ghi lại đều không được mô hình hóa. Mã sẽ thay đổi; hãy đọc lại trước khi dựa vào một quy tắc.</>}>
          Shape conditions as read in each project&rsquo;s source on 2026-11-20 (ch19-facts D1, D3, D4), in a tested model
          (src/sim/convlib.ts). Data types, hardware checks and anything the facts file does not record are not
          modelled. The code changes; read it again before relying on a rule.
          </Translated>
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', flexWrap: 'wrap' }} data-presets>
        {DISPATCH_PRESETS.map((p) => (
          <button
            key={p.id}
            aria-pressed={presetId === p.id && onPreset}
            data-preset={p.id}
            onClick={() => {
              setLayer(p.layer);
              setPresetId(p.id);
              setGuess(null);
            }}
          >
            {p.label}
          </button>
        ))}
      </div>
      <p style={{ margin: '0 0 0.8rem' }} data-layer>
        <span className="muted"><Translated vi="Lớp:">Layer:</Translated></span> {describeLayer(layer)}
        {sh.valid ? (
          <>
            <Translated vi={<>{' '}→ đầu ra {sh.P} × {sh.Q}.</>}>
            {' '}
            → output {sh.P} × {sh.Q}.
            </Translated>
          </>
        ) : (
          <><Translated vi=" → không có đầu ra."> → no output.</Translated></>
        )}{' '}
        <span className="muted"><Translated vi="Thay đổi nó trong trình khám phá ở trên.">Change it in the explorer above.</Translated></span>
      </p>

      {pending ? (
        <div className="panel" data-predict>
          <p style={{ margin: '0 0 0.6rem' }}>
            <Translated vi={<><b>Trước khi xem:</b> backend CUDA của llama.cpp chạy gì cho {active!.label}?</>}>
            <b>Before you look:</b> what does llama.cpp&rsquo;s CUDA backend run for {active!.label}?
            </Translated>
          </p>
          <div className="row" style={{ flexWrap: 'wrap' }}>
            {LLAMA_GUESSES.map((g) => (
              <button
                key={g}
                data-guess={g}
                onClick={() => {
                  setGuess(g);
                  setStarted(true);
                }}
                style={{ textAlign: 'left' }}
              >
                {LLAMA_PATH_TEXT[g]}
              </button>
            ))}
          </div>
        </div>
      ) : !started ? (
        <p className="panel" style={{ margin: 0 }} data-not-started>
          <Translated vi={<>Hãy bắt đầu với một trong ba cấu hình có sẵn. Bạn dự đoán trước khi bất cứ thứ gì được hiện ra; sau đó, mọi thay đổi của lớp cập nhật các backend ngay lập tức.</>}>
          Start with one of the three presets. You predict before anything is shown; after that, every change to the
          layer updates the backends at once.
          </Translated>
        </p>
      ) : !sh.valid ? (
        <p className="panel" data-invalid>
          <span className="chip fail"><Translated vi="không có đầu ra">no output</Translated></span><Translated vi=" Lớp này không có đầu ra, nên không backend nào chạy gì."> This layer has no output, so no backend runs anything.</Translated>
        </p>
      ) : (
        <>
          {onPreset && guess ? (
            <p style={{ margin: '0 0 0.8rem' }} data-guess-verdict={guess === llama.path ? 'right' : 'wrong'}>
              <span className={'chip ' + (guess === llama.path ? 'ok' : 'fail')}>{guess === llama.path ? tr('You had it', 'Bạn đoán đúng') : tr('Not that', 'Không phải')}</span>{' '}
              <Translated vi={<>Bạn đã chọn {LLAMA_PATH_TEXT[guess]}.</>}>
              You said {LLAMA_PATH_TEXT[guess]}.
              </Translated>
            </p>
          ) : null}
          <div style={{ display: 'grid', gap: '0.7rem' }}>
            <BackendCard id="llama" title={tr("llama.cpp, CUDA and HIP", 'llama.cpp, CUDA và HIP')} file="ggml-cuda/conv2d.cu">
              <Checks checks={llama.checks} />
              <p style={{ margin: 0 }} data-choice={llama.path}>
                <b><Translated vi="Chạy:">Runs:</Translated></b> {LLAMA_PATH_TEXT[llama.path]}.{' '}
                {llama.path === 'cublas'
                  ? tr('With K = 1 the unfolded matrix is the input itself, so there is nothing to gather.', 'Với K = 1, ma trận đã mở rộng chính là đầu vào, nên không có gì để gom.')
                  : llama.path === 'direct'
                    ? tr('A short reduction on a small map: the code’s comment says such layers do not amortise conversion and launch costs, the p. 472 argument about small matrices.', 'Một phép rút gọn ngắn trên một bản đồ nhỏ: chú thích trong mã nói những lớp như vậy không bù được chi phí chuyển đổi và khởi chạy, đúng lập luận ở p. 472 về các ma trận nhỏ.')
                    : llama.path === 'implicit'
                      ? tr('For long reductions on small maps it can also split the C·K·K sum across blocks (split-K); that condition is not modelled.', 'Với các phép rút gọn dài trên bản đồ nhỏ, nó còn có thể tách tổng C·K·K cho nhiều khối (split-K); điều kiện đó không được mô hình hóa.')
                      : tr('The facts file records both tests but not which the code makes first, so this explorer will not guess.', 'Tệp facts ghi lại cả hai phép kiểm tra nhưng không ghi mã thực hiện phép nào trước, nên trình khám phá này sẽ không đoán.')}
              </p>
            </BackendCard>
            <BackendCard id="pytorch" title={tr("PyTorch’s CUDA fallback, Slow2d", 'Đường dự phòng CUDA của PyTorch, Slow2d')} file="ConvolutionMM2d.cu">
              <p style={{ margin: '0 0 0.3rem' }} className="muted">
                <Translated vi={<>requires_columns là true nếu bất kỳ điều kiện nào sau đây đúng:</>}>
                requires_columns is true if any of these holds:
                </Translated>
              </p>
              <Checks checks={torch.checks} />
              <p style={{ margin: 0 }} data-columns={!torch.modelled ? 'unknown' : torch.requiresColumns ? 'yes' : 'no'}>
                {!torch.modelled ? (
                  <>
                    <Translated vi={<><b>Không được mô hình hóa:</b> phép kiểm tra này không có dilation, và backend nào phục vụ một lớp có dilation thì chưa được truy ra trong tệp facts.</>}>
                    <b>Not modelled:</b> this test has no dilation, and which backend serves a dilated layer is not
                    traced in the facts file.
                    </Translated>
                  </>
                ) : torch.requiresColumns ? (
                  <>
                    <Translated vi={<><b>Mở rộng</b> mỗi mẫu thành một ma trận cột {n(b.inner)} × {n(b.cols)}, rồi chạy mỗi mẫu một GEMM: mục 19.3, một cách tường minh.</>}>
                    <b>Unfolds</b> each sample into a {n(b.inner)} × {n(b.cols)} column matrix, then runs one GEMM per
                    sample: section 19.3, explicitly.
                    </Translated>
                  </>
                ) : (
                  <>
                    <Translated vi={<><b>Bỏ qua bước mở rộng:</b> GEMM đọc đầu vào nguyên như nó vốn có.</>}>
                    <b>Skips the unfold:</b> the GEMM reads the input as it is.
                    </Translated>
                  </>
                )}{' '}
                <span className="muted"><Translated vi="Các backend CUDA khác của PyTorch gồm Cudnn và Miopen.">PyTorch&rsquo;s other CUDA backends include Cudnn and Miopen.</Translated></span>
              </p>
            </BackendCard>
            <BackendCard id="mlx" title="MLX, Metal" file="backend/metal/conv.cpp">{/* i18n-ignore: product names */}
              <Checks checks={apple.checks} />
              <p style={{ margin: 0 }} data-mlx={apple.path}>
                {apple.path === 'winograd-candidate' ? (
                  <>
                    <Translated vi={<><b>Một ứng viên Winograd.</b> MLX còn yêu cầu đầu vào đủ lớn, một điều kiện không được mô hình hóa ở đây.{apple.dilationUnknown ? ' Tệp facts không nói phép kiểm tra này có xét dilation hay không.' : ''}</>}>
                    <b>A Winograd candidate.</b> MLX also asks for large enough inputs, a condition not modelled here.
                    {apple.dilationUnknown ? ' The facts file does not say whether this test looks at dilation.' : ''}
                    </Translated>
                  </>
                ) : (
                  <>
                    <Translated vi={<><b>Không phải Winograd:</b> một implicit GEMM (một kernel chuyên biệt hoặc tổng quát; các điều kiện không có trong tệp facts), hoặc, như phương án cuối cùng, một phép mở rộng tường minh làm theo từng dải hàng vừa một buffer Metal.</>}>
                    <b>Not Winograd:</b> an implicit GEMM (a specialised or a general kernel; the conditions are not in
                    the facts file), or, as the last resort, an explicit unfold done in row tiles that fit one Metal buffer.
                    </Translated>
                  </>
                )}
              </p>
            </BackendCard>
            <BackendCard id="libraries" title={tr("cuDNN and MIOpen", 'cuDNN và MIOpen')}>
              <p style={{ margin: 0 }}>
                <Translated vi={<>Được chọn lúc chạy, không theo một quy tắc bạn có thể đọc: người gọi chỉ định một thuật toán (API legacy của cuDNN), heuristic chọn một engine (graph API của nó), <code>cudnn.benchmark</code> của PyTorch đo thời gian vài cái, Find của MIOpen đo thời gian các solver của nó hoặc immediate mode tra cứu một cái.</>}>
                Chosen at run time, not by a rule you can read: the caller names an algorithm (cuDNN&rsquo;s legacy API),
                heuristics pick an engine (its graph API), PyTorch&rsquo;s <code>cudnn.benchmark</code> times several,
                MIOpen&rsquo;s Find times its solvers or immediate mode looks one up.
                </Translated>
              </p>
            </BackendCard>
            <BackendCard id="apple" title="Apple: MPSGraph, MPSCNNConvolution, MPP convolution2d">{/* i18n-ignore: product names */}
              <p style={{ margin: 0 }} data-apple>
                <Translated vi="Không thuật toán nào được công khai hay ghi trong tài liệu.">No algorithm is exposed or documented.</Translated>
              </p>
            </BackendCard>
          </div>
        </>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- the p-outsize reveal
const STEM = PRESETS.find((p) => p.id === 'stem')!.layer;
function OutsizeReveal({ committed, onLoad }: { committed: number; onLoad: () => void }) {
  const st = outputDimSteps(STEM.H, STEM.K, STEM.pad, STEM.stride, STEM.dilation);
  const answer = st.result;
  const chap = chapterOutputDim(STEM.H, STEM.K);
  return (
    <>
      <p data-outsize-answer={answer}>
        <Translated vi={<>Công thức của cuDNN: P = 1 + ({STEM.H} + 2·{STEM.pad} − (({STEM.K} − 1)·{STEM.dilation} + 1)) / {STEM.stride} = 1 +{' '}{st.room} / {STEM.stride} = 1 + {st.quotient} = <b>{answer}</b>, với phép chia nguyên.{' '}{committed === answer ? 'Đúng như bạn đã dự đoán.' : committed === chap ? `${chap} là H − K + 1 của chương, chỉ đúng khi không đệm và stride 1.` : committed === answer + 1 ? `${st.room} / ${STEM.stride} là ${st.room / STEM.stride}; phép chia cắt bớt phần lẻ, nó không làm tròn lên.` : `Hãy kiểm tra từng số hạng: phần đệm thêm ${2 * STEM.pad} hàng, bộ lọc trải trên ${st.span}, và stride chia phần còn lại.`}</>}>
        cuDNN&rsquo;s formula: P = 1 + ({STEM.H} + 2·{STEM.pad} − (({STEM.K} − 1)·{STEM.dilation} + 1)) / {STEM.stride} = 1 +{' '}
        {st.room} / {STEM.stride} = 1 + {st.quotient} = <b>{answer}</b>, with integer division.{' '}
        {committed === answer
          ? 'That is what you predicted.'
          : committed === chap
            ? `${chap} is the chapter’s H − K + 1, which holds only with no padding and stride 1.`
            : committed === answer + 1
              ? `${st.room} / ${STEM.stride} is ${st.room / STEM.stride}; the division truncates, it does not round up.`
              : `Check each term: the padding adds ${2 * STEM.pad} rows, the filter spans ${st.span}, and the stride divides what is left.`}
        </Translated>
      </p>
      <p style={{ margin: 0 }}>
        <Translated vi={<>Cùng công thức đó định kích thước đầu ra trong PyTorch ({outputDimTorch(STEM.H, STEM.K, STEM.pad, STEM.stride)}) và llama.cpp ({outputDimGgml(STEM.H, STEM.K, STEM.pad, STEM.stride)}).{' '}</>}>
        The same formula sizes outputs in PyTorch ({outputDimTorch(STEM.H, STEM.K, STEM.pad, STEM.stride)}) and llama.cpp (
        {outputDimGgml(STEM.H, STEM.K, STEM.pad, STEM.stride)}).{' '}
        </Translated>
        <button onClick={onLoad} data-load-stem>
          <Translated vi="Nạp lớp này vào trình khám phá">Load this layer into the explorer</Translated>
        </button>
      </p>
    </>
  );
}

// ---------------------------------------------------------------- the backend table, cut to the library rows (ch19-facts, last table)
const DIFFER: [() => string, () => string, () => string, () => string][] = [
  [
    () => tr('Vendor library', 'Thư viện của nhà sản xuất'),
    () => tr('cuDNN 9.28.1: graph API (C++ and Python frontend, C backend); legacy cudnnConvolutionForward deprecated since 9.0', 'cuDNN 9.28.1: graph API (frontend C++ và Python, backend C); cudnnConvolutionForward legacy bị ngừng khuyến nghị từ 9.0'),
    () => tr('MIOpen 3.6.2: miopenConvolutionForward after Find, or immediate mode; hipDNN at an early stage', 'MIOpen 3.6.2: miopenConvolutionForward sau Find, hoặc immediate mode; hipDNN ở giai đoạn đầu'),
    () => tr('MPSGraph convolution2D; MPSCNNConvolution on MPSImage; MPP convolution2d inside shaders (Metal 4)', 'MPSGraph convolution2D; MPSCNNConvolution trên MPSImage; MPP convolution2d bên trong shader (Metal 4)'),
  ],
  [
    () => tr('Forward algorithms', 'Thuật toán forward'),
    () => tr(`${CUDNN_FWD_ALGOS.length}: GEMM, IMPLICIT_GEMM, IMPLICIT_PRECOMP_GEMM, DIRECT, FFT, FFT_TILING, WINOGRAD (not on Hopper and later), WINOGRAD_NONFUSED`, `${CUDNN_FWD_ALGOS.length}: GEMM, IMPLICIT_GEMM, IMPLICIT_PRECOMP_GEMM, DIRECT, FFT, FFT_TILING, WINOGRAD (không có trên Hopper trở về sau), WINOGRAD_NONFUSED`),
    () => `${MIOPEN_FWD_ALGOS.length}: GEMM, Direct, FFT, Winograd, ImplicitGEMM`, // i18n-ignore: algorithm names
    () => tr('None exposed or documented', 'Không có thuật toán nào được công khai hay ghi trong tài liệu'),
  ],
  [
    () => tr('Choosing one', 'Chọn một thuật toán'),
    () => tr('The caller passes algo (legacy) or heuristics pick an engine; PyTorch’s cudnn.benchmark times several', 'Người gọi truyền algo (legacy) hoặc heuristic chọn một engine; cudnn.benchmark của PyTorch đo thời gian vài cái'),
    () => tr('Find times solvers and caches on disk; immediate mode uses a database and heuristics', 'Find đo thời gian các solver và lưu vào bộ nhớ đệm trên đĩa; immediate mode dùng một cơ sở dữ liệu và heuristic'),
    () => tr('Not exposed', 'Không được công khai'),
  ],
  [() => tr('Explicit im2col as an API', 'im2col tường minh dưới dạng API'), () => tr('cudnnIm2Col (deprecated in 9.0), one row per output pixel', 'cudnnIm2Col (bị ngừng khuyến nghị từ 9.0), mỗi pixel đầu ra một hàng'), () => tr('None in miopen.h', 'Không có trong miopen.h'), () => tr('MPSGraph imToCol (macOS 14 and later)', 'MPSGraph imToCol (macOS 14 trở về sau)')],
  [
    () => tr('Layouts', 'Bố cục'),
    () => tr('NCHW, NHWC, NCHW_VECT_C, any strides (some routines refuse); Tensor Cores require NHWC, converted internally', 'NCHW, NHWC, NCHW_VECT_C, stride bất kỳ (một số routine từ chối); Tensor Cores yêu cầu NHWC, được chuyển đổi bên trong'),
    () => tr('NCHW and NHWC fully supported; CHWN not supported', 'NCHW và NHWC được hỗ trợ đầy đủ; CHWN không được hỗ trợ'),
    () => tr('MPSGraph NCHW or NHWC (weights OIHW or HWIO); MPSCNN weights OHWI; MPP NHWC and HWIO only', 'MPSGraph NCHW hoặc NHWC (trọng số OIHW hoặc HWIO); trọng số MPSCNN OHWI; MPP chỉ NHWC và HWIO'),
  ],
  [() => tr('Filter flip', 'Lật bộ lọc'), () => tr('Cross-correlation mode in its formulas; frameworks select it', 'Chế độ tương quan chéo (cross-correlation) trong các công thức của nó; các framework chọn chế độ này'), () => tr('Cross-correlation', 'Tương quan chéo'), () => tr('Not documented; PyTorch passes weights unflipped to MPSGraph', 'Không có trong tài liệu; PyTorch truyền trọng số không lật cho MPSGraph')],
  [
    () => tr('Precision trap against an FP32 reference', 'Bẫy độ chính xác khi so với tham chiếu FP32'),
    () => tr('PyTorch lets cuDNN convolutions use TF32 by default', 'PyTorch cho phép các tích chập cuDNN dùng TF32 theo mặc định'),
    () => tr('Not checked here', 'Chưa được kiểm tra ở đây'),
    () => tr('MPSCNNConvolution weights FP16 unless the data source opts into FP32; MPP relaxed_precision off by default; MSL fast math', 'Trọng số MPSCNNConvolution là FP16 trừ khi nguồn dữ liệu chọn FP32; relaxed_precision của MPP tắt theo mặc định; fast math của MSL'),
  ],
];

// ---------------------------------------------------------------- the page
export function S194() {
  const q = questionsS194;
  const [layer, setLayer] = useState<Layer>(PRESETS.find((p) => p.id === 'lenet-c1')!.layer);
  const loadStem = () => {
    setLayer(STEM);
    document.querySelector(tr('section.bench[aria-label="Describe a layer to a library"]', 'section.bench[aria-label="Mô tả một lớp cho thư viện"]'))?.scrollIntoView({ block: 'start' });
  };
  return (
    <>
      <Translated vi={<p>Một framework thường không chạy tích chập bằng một kernel của riêng nó. Nó mô tả lớp cho một thư viện của nhà sản xuất, và thư viện quyết định cách tính. Trang này nói về sự chuyển giao đó: cần cho cuDNN và các thư viện tương ứng của AMD và Apple biết những gì, tên gọi của chúng ứng với tên gọi của chương ra sao, những kernel nào của chương ẩn sau các tùy chọn của chúng, và chúng quyết định gì mà không hỏi bạn.</p>}>
      <p>
        A framework usually does not run a convolution with a kernel of its own. It describes the layer to a vendor
        library, and the library decides how to compute it. This page is about that hand-off: what cuDNN and its AMD and
        Apple counterparts need to be told, how their names map onto the chapter&rsquo;s, which of the chapter&rsquo;s
        kernels hide behind their options, and what they decide without asking you.
      </p>
      </Translated>

      <h2><Translated vi="cuDNN là gì">What cuDNN is</Translated></h2>
      <Translated vi={<p>cuDNN là thư viện các routine đã tinh chỉnh của NVIDIA cho những khối xây dựng của học sâu, được làm ra để các framework có thể dùng GPU mà không phải viết kernel. Giao diện của nó là C. Đầu vào và đầu ra phải nằm sẵn trong bộ nhớ device, như với cuBLAS. Nó an toàn luồng (thread-safe): bạn có thể gọi nó từ nhiều luồng host. Tích chập lượt tiến và lượt lùi dùng chung một descriptor của lớp, và tensor cùng bộ lọc được mô tả bằng các descriptor mờ (opaque) cho phép bạn nêu bố cục dưới dạng một stride tùy ý cho mỗi chiều. Cốt lõi của nó là một tích chập theo batch, và sách trình bày chiều tiến (forward).</p>}>
      <p>
        cuDNN is NVIDIA&rsquo;s library of tuned routines for the building blocks of deep learning, made so that
        frameworks can use GPUs without writing kernels. Its interface is C. Inputs and outputs must already sit in
        device memory, as with cuBLAS. It is thread-safe: you can call it from several host threads. The forward and
        the backward convolution share one descriptor of the layer, and tensors and filters are described by opaque
        descriptors that let you give the layout as an arbitrary stride per dimension. At its heart is a batched
        convolution, and the book covers the forward direction.
      </p>
      </Translated>
      <Source section="19.4" pages="472" />
      <Translated vi={<p>Tất cả những điều đó vẫn có trong tài liệu, nhưng giờ đây là thứ NVIDIA gọi là API legacy. Hãy đọc nó như mô hình của một lời gọi thư viện, chứ không phải lời gọi bạn sẽ viết hôm nay.</p>}>
      <p>
        All of that is still documented, but it is now what NVIDIA calls the legacy API. Read it as the model of a
        library call, not as the call you would write today.
      </p>
      </Translated>
      <Today checked="2026-11-20 (ch19-facts A1)">{/* i18n-ignore: date and fact citation */}
        <Translated vi={<p><code>cudnnConvolutionForward</code> và các descriptor của nó đã bị ngừng khuyến nghị từ cuDNN 9.0 và tiếp tục tồn tại trong các thư viện legacy. NVIDIA khuyên dùng graph API của nó, qua frontend C++ và Python. Ba điều kiện sách bỏ qua: các luồng không được dùng cùng một handle cuDNN cùng lúc, một tensor bị giới hạn ở 2 Giga-elements (2 tỉ phần tử), và một số routine từ chối các stride bất thường.</p>}>
        <p>
          <code>cudnnConvolutionForward</code> and its descriptors have been deprecated since cuDNN 9.0 and live on in
          the legacy libraries. NVIDIA recommends its graph API, through the C++ and Python frontend. Three conditions the
          book leaves out: threads must not use one cuDNN handle at the same time, a tensor is capped at 2
          Giga-elements, and some routines reject unusual strides.
        </p>
        </Translated>
      </Today>
      <Translated vi={<p>Một chỗ sơ suất ở p. 472: sách nói yêu cầu về bộ nhớ device đã được bàn ở phần trước của chương. Không có phần nào trước đó trong chương 19 bàn về nó.</p>}>
      <p>
        One slip on p. 472: the book says the device-memory requirement was discussed earlier in the chapter. Nothing
        earlier in chapter 19 discusses it.
      </p>
      </Translated>

      <h2><Translated vi="Ai gọi nó">Who calls it</Translated></h2>
      <Translated vi={<p>Sách nêu tên PyTorch, Caffe, TensorFlow và MXNet là các framework xây trên cuDNN, và nói phần lớn các framework học sâu dùng nó, nên người dùng của chúng có được các cài đặt này mà không phải viết CUDA.</p>}>
      <p>
        The book names PyTorch, Caffe, TensorFlow and MXNet as frameworks built on cuDNN, and says most deep-learning
        frameworks use it, so their users get these implementations without writing CUDA.
      </p>
      </Translated>
      <Source section="19.4" pages="472, 474" />
      <Translated vi={<p>Hai trong bốn cái đó đã dừng. MXNet bị khai tử năm 2023 và Caffe không có commit nào kể từ 2020. Danh sách của chính NVIDIA hôm nay có PyTorch, JAX và XLA, Megatron Core và Transformer Engine của họ, và, cho phục vụ LLM, vLLM và SGLang thông qua FlashInfer. Con đường cuối cùng đó là cho attention, không phải tích chập.</p>}>
      <p>
        Two of those four have stopped. MXNet was retired in 2023 and Caffe has had no commit since 2020. NVIDIA&rsquo;s
        own list today has PyTorch, JAX and XLA, its Megatron Core and Transformer Engine, and, for LLM serving, vLLM and
        SGLang through FlashInfer. That last route is for attention, not convolution.
      </p>
      </Translated>
      <Today checked="2026-11-20 (ch19-facts A6)">{/* i18n-ignore: date and fact citation */}
        <Translated vi={<p>Apache MXNet: bị khai tử tháng 9 năm 2023, chuyển vào Attic tháng 2 năm 2024. BVLC/caffe: commit cuối 2020-02-13.</p>}>
        <p>Apache MXNet: retired September 2023, moved to the Attic February 2024. BVLC/caffe: last commit 2020-02-13.</p>
        </Translated>
      </Today>

      <h2><Translated vi="Hai bộ tên gọi">Two sets of names</Translated></h2>
      <Translated vi={<p>Bảng 19.1 liệt kê các tham số của cuDNN, và chú thích của nó cảnh báo rằng chúng không phải là tên của chương. Phần lớn các chữ cái giữ nguyên: <code>N</code> mẫu, <code>C</code> bản đồ đầu vào, <code>H</code> và <code>W</code> cho chiều cao và chiều rộng của đầu vào. Có hai chữ thì không. <code>K</code> của cuDNN là số bản đồ đặc trưng đầu ra, mà chương gọi là <code>M</code>. Chiều cao và chiều rộng của bộ lọc là <code>R</code> và{' '}<code>S</code>, trong khi chương dùng một <code>K</code> cho bộ lọc vuông. Bốn tham số còn lại,{' '}<code>u</code>, <code>v</code>, <code>pad_h</code> và <code>pad_w</code>, là stride và phần đệm, những thứ các kernel của chương không có. Sách gọi các tensor là <code>D</code> (đầu vào), <code>F</code> (bộ lọc) và <code>O</code> (đầu ra), và chiều cao, chiều rộng của đầu ra là <code>P</code> và <code>Q</code>.</p>}>
      <p>
        Table 19.1 lists cuDNN&rsquo;s parameters, and its caption warns that they are not the chapter&rsquo;s. Most
        letters carry over: <code>N</code> samples, <code>C</code> input maps, <code>H</code> and <code>W</code> for the
        input&rsquo;s height and width. Two do not. cuDNN&rsquo;s <code>K</code> is the number of output feature maps,
        which the chapter calls <code>M</code>. The filter&rsquo;s height and width are <code>R</code> and{' '}
        <code>S</code>, where the chapter uses one <code>K</code> for a square filter. The remaining four,{' '}
        <code>u</code>, <code>v</code>, <code>pad_h</code> and <code>pad_w</code>, are strides and padding, which the
        chapter&rsquo;s kernels do not have. The book calls the tensors <code>D</code> (input), <code>F</code> (filters)
        and <code>O</code> (output), and the output&rsquo;s height and width <code>P</code> and <code>Q</code>.
      </p>
      </Translated>
      <Source section="19.4" pages="472–473" figs={tr("Table 19.1", 'Bảng 19.1')} />
      <Ask q={q['ch19/19.4/q-names']} />

      <h2><Translated vi="Kích thước đầu ra">The output size</Translated></h2>
      <Translated vi={<p>Về kích thước đầu ra, sách chỉ cho một sự phụ thuộc, <code>P = f(H; R; u; pad_h)</code>, không có công thức. cuDNN ghi trong tài liệu một công thức: <code>P = 1 + (H + 2·pad_h − ((R − 1)·dilation + 1)) / u</code>, với phép chia nguyên. Số hạng{' '}<code>(R − 1)·dilation + 1</code> là số hàng đầu vào mà các tap của bộ lọc trải trên: R khi dilation bằng 1, nhiều hơn khi các tap được giãn ra. PyTorch ghi trong tài liệu cùng công thức với phép làm tròn xuống, và llama.cpp tính nó trong{' '}<code>ggml_calc_conv_output_size</code>. Các kernel của chương là trường hợp pad 0, stride 1, dilation 1, khi đó nó rút gọn thành <code>H − K + 1</code>.</p>}>
      <p>
        For the output size the book gives only a dependence, <code>P = f(H; R; u; pad_h)</code>, with no formula. cuDNN
        documents one: <code>P = 1 + (H + 2·pad_h − ((R − 1)·dilation + 1)) / u</code>, with integer division. The term{' '}
        <code>(R − 1)·dilation + 1</code> is how many input rows the filter&rsquo;s taps span: R when dilation is 1, more
        when the taps are spread apart. PyTorch documents the same formula with a floor, and llama.cpp computes it in{' '}
        <code>ggml_calc_conv_output_size</code>. The chapter&rsquo;s kernels are the case pad 0, stride 1, dilation 1,
        where it reduces to <code>H − K + 1</code>.
      </p>
      </Translated>
      <Today checked="2026-11-20 (ch19-facts A2)">{/* i18n-ignore: date and fact citation */}
        <Translated vi={<p>Từ <code>cudnnGetConvolution2dForwardOutputDim</code>, legacy từ 9.0 như phần còn lại. Dilation và groups là các tham số của cuDNN mà Bảng 19.1 không liệt kê.</p>}>
        <p>
          From <code>cudnnGetConvolution2dForwardOutputDim</code>, legacy since 9.0 like the rest. Dilation and groups are
          cuDNN parameters that Table 19.1 does not list.
        </p>
        </Translated>
      </Today>
      <Ask q={q['ch19/19.4/q-padding']} />
      <Translated vi={<p>Phần đệm cần được sửa lại. Sách nói nó phục vụ việc căn chỉnh bộ nhớ hoặc thực thi vector hóa. Đó không phải điều tham số này ghi trong tài liệu. <code>pad_h</code> là số hàng số 0 mà cuDNN thêm vào phía trên và phía dưới mỗi ảnh đầu vào, và <code>pad_w</code> là số cột số 0 ở bên trái và bên phải: các ô ma (ghost cell) của chương 7. Chúng tham gia vào công thức, nên chúng thay đổi kích thước đầu ra; đệm một lớp 3 × 3 thêm 1 giữ đầu ra lớn bằng đầu vào. Phần đệm để căn chỉnh cũng có tồn tại, nhưng bên trong cuDNN, vốn tự đệm các tensor NCHW khi số kênh không phải bội số của 8, để có thể dùng Tensor Cores.</p>}>
      <p>
        Padding needs a correction. The book says it serves memory alignment or vectorised execution. That is not what
        the parameter documents. <code>pad_h</code> is the number of zero rows cuDNN adds at the top and at the bottom of
        each input image, and <code>pad_w</code> the zero columns at the left and right: chapter 7&rsquo;s ghost cells.
        They enter the formula, so they change the output size; padding a 3 × 3 layer by 1 keeps its output as large as
        its input. Alignment padding exists too, but inside cuDNN, which pads NCHW tensors itself when the channel count
        is not a multiple of 8, so that Tensor Cores can be used.
      </p>
      </Translated>
      <Source section="19.4" pages="473" />
      <Predict
        id="ch19/19.4/p-outsize"
        prompt={
          <Translated vi={<>Một lớp có <code>H = 224</code>, bộ lọc 7 × 7 (<code>R = 7</code>), stride dọc <code>u = 2</code>,{' '}<code>pad_h = 3</code> và dilation 1. <code>P</code>, chiều cao đầu ra, là bao nhiêu?</>}>
          <>
            A layer has <code>H = 224</code>, a 7 × 7 filter (<code>R = 7</code>), vertical stride <code>u = 2</code>,{' '}
            <code>pad_h = 3</code> and dilation 1. What is <code>P</code>, the output height?
          </>
          </Translated>
        }
        unit={tr("rows", 'hàng')}
      >
        {(v) => <OutsizeReveal committed={v} onLoad={loadStem} />}
      </Predict>
      <Translated vi={<p>Bây giờ hãy tự đặt các lớp: sáu tên của chương, rồi ba tham số mà các kernel của nó không có. Hãy tìm một thiết lập thay đổi <code>P</code> mà không động đến <code>H</code> hay <code>K</code>, và một stride làm sự trùng lặp của mục 19.3 biến mất. Các thẻ ghi nhớ luyện hai bộ tên gọi.</p>}>
      <p>
        Now set layers of your own: the chapter&rsquo;s six names, then the three parameters its kernels lack. Look for
        a setting that changes <code>P</code> without touching <code>H</code> or <code>K</code>, and a stride that makes
        the duplication of section 19.3 disappear. The flashcards drill the two sets of names.
      </p>
      </Translated>
      <Simplification>
        <Translated vi={<p>Chỉ 2D, groups 1, bộ lọc vuông, và mỗi loại phần đệm, stride và dilation một giá trị cho cả hai trục; cuDNN nhận chúng theo từng trục. Hình dạng và số đếm là chính xác cho lớp. Không có gì ở đây mô hình hóa thời gian.</p>}>
        <p>
          2D only, groups 1, a square filter, and one value each of padding, stride and dilation for both axes; cuDNN
          takes them per axis. Shapes and counts are exact for the layer. Nothing here models time.
        </p>
        </Translated>
      </Simplification>
      <DescribeBench layer={layer} setLayer={setLayer} />

      <h2><Translated vi="Các thuật toán">The algorithms</Translated></h2>
      <Translated vi={<p>cuDNN cung cấp vài thuật toán cho một lớp tích chập: các thuật toán dựa trên phép nhân ma trận (sách xếp cả GEMM và Winograd vào đây), các thuật toán dựa trên FFT, và những loại khác. Thuật toán dựa trên GEMM của nó là mục 19.3. Nó không gom đầu vào đã mở rộng vào bộ nhớ toàn cục rồi gọi một phép nhân ma trận; nó tạo các phần nó cần trong lúc chạy và chỉ nạp chúng vào bộ nhớ trên chip.</p>}>
      <p>
        cuDNN offers several algorithms for a convolutional layer: ones based on matrix multiplication (the book files
        both GEMM and Winograd here), ones based on the FFT, and others. Its GEMM-based algorithm is section 19.3. It
        does not gather the unfolded input in global memory and then call a matrix multiply; it generates the parts it
        needs as it goes and loads them only into on-chip memory.
      </p>
      </Translated>
      <Source section="19.4" pages="473" />
      <Translated vi={<p>Tài liệu hôm nay liệt kê tám thuật toán forward, và ba trong số đó là chương này. <code>GEMM</code> là phép mở rộng tường minh, với một workspace đáng kể để chứa đầu vào đã mở rộng. <code>IMPLICIT_GEMM</code> là Hình 19.11. <code>IMPLICIT_PRECOMP_GEMM</code> cũng là ngầm định, nhưng trước tiên tính trước một số chỉ số vào workspace, nên câu &ldquo;computes the mapping dynamically&rdquo; (tính ánh xạ một cách động) của sách chỉ mô tả một trong ba.{' '}<code>DIRECT</code> là loại kernel mà mục 19.2 viết. FFT và Winograd là toán học khác, nằm ngoài chương này. Mọi thuật toán forward đều được ghi trong tài liệu là tất định; chọn một thuật toán thay đổi tốc độ và những trường hợp được hỗ trợ.</p>}>
      <p>
        The documentation lists eight forward algorithms today, and three of them are this chapter. <code>GEMM</code> is
        the explicit unfold, with a significant workspace to hold the unfolded input. <code>IMPLICIT_GEMM</code> is Fig.
        19.11. <code>IMPLICIT_PRECOMP_GEMM</code> is implicit too, but first precomputes some indices into a workspace,
        so the book&rsquo;s &ldquo;computes the mapping dynamically&rdquo; describes only one of the three.{' '}
        <code>DIRECT</code> is the kind of kernel section 19.2 writes. FFT and Winograd are other mathematics, outside
        this chapter. Every forward algorithm is documented as deterministic; choosing one changes speed and which cases
        are supported.
      </p>
      </Translated>
      <Today checked="2026-11-20 (ch19-facts A3, A4)">{/* i18n-ignore: date and fact citation */}
        <Translated vi={<p><code>WINOGRAD</code> không được hỗ trợ trên Hopper và các GPU về sau. <code>cudnnIm2Col</code>, bị ngừng khuyến nghị từ 9.0, từng là một phép mở rộng tường minh công khai; nó tạo ra chuyển vị của <code>B</code> của sách, mỗi pixel đầu ra của mọi mẫu một hàng.</p>}>
        <p>
          <code>WINOGRAD</code> is not supported on Hopper and later GPUs. <code>cudnnIm2Col</code>, deprecated in 9.0,
          was a public explicit unfold; it built the transpose of the book&rsquo;s <code>B</code>, one row per output
          pixel of every sample.
        </p>
        </Translated>
      </Today>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-algos>
          <thead>
            <tr>
              <th><Translated vi="Thuật toán forward của cuDNN">cuDNN forward algorithm</Translated></th>
              <th><Translated vi="Trong chương này">In this chapter</Translated></th>
              <th><Translated vi="Workspace, theo tài liệu">Workspace, as documented</Translated></th>
              <th><Translated vi="HIPIFY sang MIOpen">HIPIFY to MIOpen</Translated></th>
            </tr>
          </thead>
          <tbody>
            {CUDNN_FWD_ALGOS.map((a) => (
              <tr key={a.name} data-algo={a.name}>
                <td>
                  <code>{a.name.replace('CUDNN_CONVOLUTION_FWD_ALGO_', '')}</code>
                  {a.note ? <div className="muted">{a.note}</div> : null}
                </td>
                <td>{a.chapter}</td>
                <td>{a.workspace}</td>
                <td>{a.hipify ? <code>{a.hipify.replace('miopenConvolutionFwdAlgo', '')}</code> : <span className="muted"><Translated vi="không có trong danh sách">none listed</Translated></span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted" style={{ fontSize: '0.85rem', marginTop: '-0.8rem' }}>
          <Translated vi={<>Mỗi tên có tiền tố <code>CUDNN_CONVOLUTION_FWD_ALGO_</code>, mỗi giá trị MIOpen có tiền tố{' '}<code>miopenConvolutionFwdAlgo</code>. Từ ch19-facts A3 và B4.</>}>
          Each name is prefixed with <code>CUDNN_CONVOLUTION_FWD_ALGO_</code>, each MIOpen value with{' '}
          <code>miopenConvolutionFwdAlgo</code>. From ch19-facts A3 and B4.
          </Translated>
        </p>
      </div>
      <Ask q={q['ch19/19.4/q-workspace']} />
      <Ask q={q['ch19/19.4/q-precomp']} />

      <h2><Translated vi="Phép nhân ma trận chạy thế nào">How the matrix multiplication runs</Translated></h2>
      <Translated vi={<p>Bên dưới, sách nói, là một phép nhân ma trận đạt một tỉ lệ cao so với thông lượng đỉnh số học của thiết bị. Nó đọc các ô kích thước cố định của cả hai đầu vào vào bộ nhớ trên chip và tính một ô của đầu ra từ chúng, và trong khi tính trên các ô hiện tại thì nó đã nạp các ô kế tiếp. Đó là software pipelining của mục <a href={href('ch15', '15.8')}>15.8</a>: thời gian chờ bộ nhớ được che sau phép tính số học.</p>}>
      <p>
        Underneath, the book says, is a matrix multiplication that reaches a high fraction of the device&rsquo;s peak
        arithmetic. It reads fixed-size tiles of both inputs into on-chip memory and computes a tile of the output from
        them, and while it computes on the current tiles it is already fetching the next ones. That is the software
        pipelining of section <a href={href('ch15', '15.8')}>15.8</a>: the wait for memory hides behind arithmetic.
      </p>
      </Translated>
      <Source section="19.4" pages="473–474" />
      <Translated vi={<p>Việc phân ô không biết gì về tích chập, nên chỗ các cạnh của một ô rơi vào trong đầu vào không đơn giản. cuDNN tính sự tương ứng đó trong lúc tính toán và dùng nó để nạp đúng các phần tử: thêm phép tính chỉ số, đổi lại được dùng lại toàn bộ cỗ máy nhân ma trận. Sách nói thêm rằng routine của NVIDIA giống một DGEMM năm 2011 cho GPU Fermi, tài liệu tham khảo số 10 của sách. cuDNN không ghi trong tài liệu kernel nào của nó, nên phép so sánh đó không kiểm chứng được.</p>}>
      <p>
        The tiling knows nothing about convolutions, so where a tile&rsquo;s edges fall in the input is not simple.
        cuDNN computes that correspondence as the computation runs and uses it to load the right elements: extra index
        arithmetic, in exchange for reusing the whole matrix-multiplication engine. The book adds that NVIDIA&rsquo;s
        routine is similar to a 2011 DGEMM for Fermi GPUs, its reference 10. cuDNN documents none of its kernels, so that
        comparison cannot be checked.
      </p>
      </Translated>
      <Aside>
        <Translated vi={<p>Unverified: cuDNN không ghi trong tài liệu chi tiết bên trong kernel nào, và không tải được bài báo năm 2011 (ch19-facts A4, A10).</p>}>
        <p>Unverified: cuDNN documents no kernel internals, and the 2011 paper could not be fetched (ch19-facts A4, A10).</p>
        </Translated>
      </Aside>
      <Translated vi={<p>NVIDIA có ghi lại kỹ thuật này, trong CUTLASS, thư viện GEMM mã nguồn mở của họ. Tích chập implicit-GEMM của nó tạo các ô của ma trận tích chập ngay trong lúc chạy, khi dữ liệu di chuyển từ bộ nhớ toàn cục sang bộ nhớ chia sẻ, rồi giao chúng cho các thành phần GEMM cấp warp. Nó tách chỉ số rút gọn đúng như Eq. 19.5. Và nó nói việc tính trực tiếp các phép chia và lấy dư đó sẽ quá tốn, nên nó tính trước các độ lệch con trỏ trên host và dùng một phép chia-lấy-dư nhanh cho phần còn lại. GEMM của nó là chuyển vị của GEMM trong sách: N·P·Q hàng là các pixel đầu ra, K bản đồ đầu ra là các cột, một phép rút gọn trên C·R·S, trên dữ liệu NHWC.</p>}>
      <p>
        NVIDIA does document the technique, in CUTLASS, its open-source GEMM library. Its implicit-GEMM convolution forms
        tiles of the convolution matrix on the fly, as data moves from global to shared memory, and hands them to its
        warp-level GEMM components. It splits the reduction index exactly as Eq. 19.5 does. And it says that computing
        those divisions and remainders directly would cost too much, so it precomputes pointer deltas on the host and
        uses a fast divide-and-modulo for the rest. Its GEMM is the transpose of the book&rsquo;s: N·P·Q rows of output
        pixels, the K output maps as columns, a reduction over C·R·S, on NHWC data.
      </p>
      </Translated>
      <Today checked="2026-11-20 (ch19-facts A7, A8)">{/* i18n-ignore: date and fact citation */}
        <Translated vi={<p>Hướng dẫn best practices của CUDA: phép chia và lấy dư số nguyên có thể biên dịch ra tới 20 lệnh. CUTLASS khuyến nghị các tensor NHWC căn chỉnh 128-bit, với C và K là bội số của 32.</p>}>
        <p>
          CUDA&rsquo;s best-practices guide: integer division and modulo can compile to up to 20 instructions. CUTLASS
          recommends 128-bit aligned NHWC tensors, with C and K multiples of 32.
        </p>
        </Translated>
      </Today>

      <h2><Translated vi="Bước cuối: bố cục">The last step: layout</Translated></h2>
      <Translated vi={<p>Sau khi tính xong, sách nói, cuDNN chuyển vị kết quả sang bất kỳ bố cục dữ liệu nào người dùng yêu cầu.</p>}>
      <p>
        After computing, the book says, cuDNN transposes the result into whatever data layout the user asked for.
      </p>
      </Translated>
      <Source section="19.4" pages="474" />
      <Translated vi={<p>Tài liệu gắn việc chuyển đổi đó với Tensor Cores và không gì khác. Tensor Cores yêu cầu NHWC, nên cuDNN chuyển NCHW sang NHWC và ngược lại khi phép tính Tensor Core được yêu cầu và thực sự được dùng, và không làm vậy trong các trường hợp khác. Một yêu cầu chỉ là một yêu cầu: Tensor Cores có thể không được dùng.</p>}>
      <p>
        The documentation ties that conversion to Tensor Cores and nothing else. Tensor Cores require NHWC, so cuDNN
        converts NCHW to NHWC and back when Tensor Core math is requested and actually used, and not otherwise. A request
        is only a request: Tensor Cores may not be used.
      </p>
      </Translated>
      <Today checked="2026-11-20 (ch19-facts A5)">{/* i18n-ignore: date and fact citation */}
        <Translated vi={<p>Các định dạng của cuDNN là NCHW, NHWC và NCHW_VECT_C, với các thứ tự khác qua stride. NHWC chỉ được hỗ trợ hạn chế trong lời gọi forward legacy.</p>}>
        <p>
          cuDNN&rsquo;s formats are NCHW, NHWC and NCHW_VECT_C, with other orders through strides. NHWC has limited
          support in the legacy forward call.
        </p>
        </Translated>
      </Today>

      <h2><Translated vi="Chọn thuật toán trong thực tế">Choosing an algorithm in practice</Translated></h2>
      <Translated vi={<p>Với API legacy, bạn chỉ định thuật toán trong lời gọi. Trong thực tế, một framework sẽ chọn, và PyTorch cho bạn điều khiển nó. Khi đặt <code>torch.backends.cudnn.benchmark</code>, cuDNN đo benchmark vài thuật toán ở lần đầu gặp một bộ kích thước mới và giữ cái nhanh nhất; <code>torch.backends.cudnn.deterministic</code> giới hạn nó vào các thuật toán tất định. Đó là lý do lần lặp đầu tiên của một mô hình bật benchmark thì chậm, và vì sao hai lần chạy trên cùng một máy có thể kết thúc với các kernel khác nhau.</p>}>
      <p>
        With the legacy API you name the algorithm in the call. In practice a framework chooses, and PyTorch lets you
        steer it. With <code>torch.backends.cudnn.benchmark</code> set, cuDNN benchmarks several algorithms the first time
        it meets a new set of sizes and keeps the fastest; <code>torch.backends.cudnn.deterministic</code> restricts it to
        deterministic ones. That is why the first iteration of a model with benchmarking on is slow, and why two runs on
        the same machine can end up with different kernels.
      </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Từ tài liệu của PyTorch 2.15 về <code>torch.backends</code> và các ghi chú về tính ngẫu nhiên (ch19-facts A6).</p>}>
        <p>From PyTorch 2.15&rsquo;s documentation of <code>torch.backends</code> and its notes on randomness (ch19-facts A6).</p>
        </Translated>
      </Beyond>
      <Translated vi={<p>Độ chính xác cũng là một lựa chọn. PyTorch cho cuDNN chạy các tích chập FP32 ở TF32 theo mặc định:{' '}<code>torch.backends.cudnn.allow_tf32</code> là True. PyTorch nói cờ đó sẽ bị ngừng khuyến nghị để thay bằng{' '}<code>torch.backends.cudnn.conv.fp32_precision</code>, đặt thành &ldquo;ieee&rdquo; hoặc &ldquo;tf32&rdquo;, và trang này không nêu giá trị mặc định của nó. Khi so sánh kernel của bạn với <code>conv2d</code> trên GPU NVIDIA, hãy in thiết lập đó ra và so sánh với một dung sai.</p>}>
      <p>
        Precision is a choice too. PyTorch lets cuDNN run FP32 convolutions in TF32 by default:{' '}
        <code>torch.backends.cudnn.allow_tf32</code> is True. PyTorch says that flag is going to be deprecated in favour of{' '}
        <code>torch.backends.cudnn.conv.fp32_precision</code>, set to &ldquo;ieee&rdquo; or &ldquo;tf32&rdquo;, and the
        page does not state its default. When you compare your kernel with <code>conv2d</code> on an NVIDIA GPU, print
        the setting and compare with a tolerance.
      </p>
      </Translated>
      <Today checked="2026-11-20 (ch19-facts A6)">{/* i18n-ignore: date and fact citation */}
        <Translated vi={<p>Các thiết lập độ chính xác chi tiết hơn xuất hiện sau PyTorch 2.9. Các đường dự phòng tích chập chậm hoặc ngây thơ xây trên GEMM của CUDA thừa hưởng thiết lập độ chính xác của matmul.</p>}>
        <p>
          The finer-grained precision settings arrived after PyTorch 2.9. Slow or naive convolution fallbacks built on
          CUDA GEMMs inherit the matmul precision setting.
        </p>
        </Translated>
      </Today>
      <Ask q={q['ch19/19.4/q-tf32']} />

      <h2><Translated vi="Trên AMD: MIOpen">On AMD: MIOpen</Translated></h2>
      <Translated vi={<p>Thư viện tương ứng của AMD có hình dáng của API legacy của cuDNN: một handle, các descriptor mờ, và một lời gọi forward nhận một thuật toán và một workspace. MIOpen 3.6.2 có năm thuật toán forward, GEMM, direct, FFT, Winograd và implicit GEMM, và nó tính tương quan chéo, như sách. Hai khác biệt quan trọng trong thực tế. Một bước Find, biên dịch và đo thời gian một tập solver cho bài toán của bạn, là bắt buộc trước{' '}<code>miopenConvolutionForward</code>; nó lưu kết quả vào bộ nhớ đệm (cache) trên đĩa để các phiên sau chạy nhanh hơn, và immediate mode thay nó bằng một lần tra cứu trong cơ sở dữ liệu. Và trong ba bố cục của Bài tập 2, NCHW và NHWC được hỗ trợ đầy đủ còn CHWN hoàn toàn không được hỗ trợ.</p>}>
      <p>
        AMD&rsquo;s counterpart has the shape of cuDNN&rsquo;s legacy API: a handle, opaque descriptors, and a forward
        call that takes an algorithm and a workspace. MIOpen 3.6.2 has five forward algorithms, GEMM, direct, FFT,
        Winograd and implicit GEMM, and it computes cross-correlation, as the book does. Two differences matter in
        practice. A Find step, which compiles and times a set of solvers for your problem, is mandatory before{' '}
        <code>miopenConvolutionForward</code>; it caches its results on disk so later sessions run faster, and immediate
        mode replaces it with a lookup in a database. And of Exercise 2&rsquo;s three layouts, NCHW and NHWC are fully
        supported and CHWN is not supported at all.
      </p>
      </Translated>
      <Today checked="2026-11-20 (ch19-facts B3)">{/* i18n-ignore: date and fact citation */}
        <Translated vi={<p>Descriptor của MIOpen nhận phần đệm, stride và dilation như của cuDNN. Không tìm thấy phát biểu chung nào về an toàn luồng của MIOpen (unverified); header của nó nói việc đo thời gian kernel trên một handle không an toàn luồng.</p>}>
        <p>
          MIOpen&rsquo;s descriptor takes padding, stride and dilation like cuDNN&rsquo;s. No general thread-safety
          statement for MIOpen was found (unverified); its header says kernel timings on one handle are not thread-safe.
        </p>
        </Translated>
      </Today>
      <Translated vi={<p>Chuyển một lời gọi cuDNN không phải việc máy móc. HIPIFY ánh xạ <code>cudnnConvolutionForward</code> sang{' '}<code>miopenConvolutionForward</code> và năm trong tám giá trị thuật toán, nhưng không liệt kê bản tương ứng MIOpen nào cho hàm đặt descriptor <code>cudnnSetConvolution2dDescriptor</code> (hàm khởi tạo của MIOpen nhận chế độ trước và không có compute type), cũng như cho <code>IMPLICIT_PRECOMP_GEMM</code>, <code>FFT_TILING</code> hay{' '}<code>WINOGRAD_NONFUSED</code>. Thư viện kiểu graph của AMD, hipDNN, tự nói về mình là đang ở giai đoạn đầu phát triển.</p>}>
      <p>
        Porting a cuDNN call is not mechanical. HIPIFY maps <code>cudnnConvolutionForward</code> to{' '}
        <code>miopenConvolutionForward</code> and five of the eight algorithm values, but lists no MIOpen counterpart for
        the descriptor setter <code>cudnnSetConvolution2dDescriptor</code> (MIOpen&rsquo;s initialiser takes the mode
        first and has no compute type), nor for <code>IMPLICIT_PRECOMP_GEMM</code>, <code>FFT_TILING</code> or{' '}
        <code>WINOGRAD_NONFUSED</code>. AMD&rsquo;s graph-style library, hipDNN, says of itself that it is in the early
        stages of development.
      </p>
      </Translated>
      <Today checked="2026-11-20 (ch19-facts B4)">{/* i18n-ignore: date and fact citation */}
        <Translated vi={<p>Từ bảng API cuDNN được MIOpen hỗ trợ của HIPIFY, và README của hipDNN.</p>}>
        <p>From HIPIFY&rsquo;s table of the cuDNN API supported by MIOpen, and hipDNN&rsquo;s README.</p>
        </Translated>
      </Today>
      <Ask q={q['ch19/19.4/q-find']} />

      <h2><Translated vi="Trên Apple: ba con đường, không có menu">On Apple: three routes, no menu</Translated></h2>
      <Translated vi={<p>Apple có ba cách chạy một tích chập, và không cách nào công khai lựa chọn thuật toán. <code>convolution2D</code> của MPSGraph{' '}nhận dữ liệu NCHW hoặc NHWC và trọng số OIHW hoặc HWIO; NCHW với OIHW là bố cục của sách. MPSGraph còn có <code>imToCol</code> (macOS 14 trở về sau), một phép mở rộng tường minh, chính <code>B</code> được hiện thực hóa của sách, dưới dạng một phép toán đồ thị. <code>MPSCNNConvolution</code> làm việc trên các đối tượng{' '}<code>MPSImage</code> thay vì buffer thô và chỉ nhận trọng số OHWI. <code>convolution2d</code> của Metal Performance Primitives chạy bên trong shader của chính bạn trên Metal 4, được cả một threadgroup dùng, chỉ với activation NHWC, trọng số HWIO và groups bằng 1.</p>}>
      <p>
        Apple has three ways to run a convolution, and none of them exposes an algorithm choice. MPSGraph&rsquo;s{' '}
        <code>convolution2D</code> takes NCHW or NHWC data and OIHW or HWIO weights; NCHW with OIHW is the book&rsquo;s
        layout. MPSGraph also has <code>imToCol</code> (macOS 14 and later), an explicit unfold, the book&rsquo;s
        materialised <code>B</code>, as a graph operation. <code>MPSCNNConvolution</code> works on{' '}
        <code>MPSImage</code> objects rather than raw buffers and takes OHWI weights only. Metal Performance
        Primitives&rsquo; <code>convolution2d</code> runs inside your own shader on Metal 4, used by a whole threadgroup,
        with NHWC activations, HWIO weights and groups of 1 only.
      </p>
      </Translated>
      <Today checked="2026-11-20 (ch19-facts C3, C4, C5)">{/* i18n-ignore: date and fact citation */}
        <Translated vi={<p>Từ các header của SDK macOS 27.0 và mục 7.2.2 của đặc tả Metal Shading Language 4.1.</p>}>
        <p>From the macOS 27.0 SDK headers and section 7.2.2 of the Metal Shading Language specification 4.1.</p>
        </Translated>
      </Today>
      <Translated vi={<p>Hai cái bẫy. <code>MPSCNNConvolution</code> dùng trọng số FP16 trừ khi nguồn dữ liệu của nó báo FP32, nên khi so với một tham chiếu FP32 nó có thể lệch nhiều hơn bạn nghĩ. Và có hai điều không được ghi trong tài liệu: tích chập của MPSGraph có lật bộ lọc hay không (PyTorch truyền trọng số <code>nn.Conv2d</code> cho nó mà không đảo ngược, nên trong thực tế nó tính tương quan chéo), và <code>convolution2d</code> của MPP đặt cửa sổ của nó ở đâu so với độ lệch bạn đưa vào (kernel của PyTorch ghi lại một quy tắc mà họ đã đo).</p>}>
      <p>
        Two traps. <code>MPSCNNConvolution</code> uses FP16 weights unless its data source reports FP32, so against an
        FP32 reference it can be further off than you expect. And two things are not documented: whether MPSGraph&rsquo;s
        convolution flips the filter (PyTorch passes <code>nn.Conv2d</code> weights to it unreversed, so in practice it
        computes cross-correlation), and where MPP&rsquo;s <code>convolution2d</code> places its window relative to the
        offset you give it (PyTorch&rsquo;s kernel records a rule it measured).
      </p>
      </Translated>
      <Aside>
        <Translated vi={<p>Hai chỗ sơ suất trong các header của Apple: descriptor im2col gọi NCHW là &ldquo;batch x channels x width x height&rdquo;, và header tích chập cho chiều rộng kernel có dilation là (dilationRateInX − 1)·kernelWidth + 1 trong khi ý muốn nói là (kernelWidth − 1)·dilation + 1 (ch19-facts C3).</p>}>
        <p>
          Two slips in Apple&rsquo;s headers: the im2col descriptor calls NCHW &ldquo;batch x channels x width x
          height&rdquo;, and the convolution header gives the dilated kernel width as (dilationRateInX − 1)·kernelWidth + 1
          where (kernelWidth − 1)·dilation + 1 is meant (ch19-facts C3).
        </p>
        </Translated>
      </Aside>

      <h2><Translated vi="Các backend thực tế chạy gì">What real backends run</Translated></h2>
      <Translated vi={<p>Cho mục tiêu thứ ba của bạn, đọc các backend thực tế: đây là nơi mỗi backend gửi một tích chập 2D tới. Danh sách backend tích chập của PyTorch có <code>Cudnn</code>, <code>Miopen</code>, <code>Mps</code> và <code>Slow2d</code>, đường dự phòng CUDA của chính nó. <code>Slow2d</code> là mục 19.3 với một phép mở rộng tường minh, từng mẫu một, và nó bỏ qua bước mở rộng khi bộ lọc là 1 × 1 với stride 1 và không đệm.</p>}>
      <p>
        For your third goal, reading real backends: here is where each one sends a 2D convolution. PyTorch&rsquo;s list
        of convolution backends has <code>Cudnn</code>, <code>Miopen</code>, <code>Mps</code> and <code>Slow2d</code>, its
        own CUDA fallback. <code>Slow2d</code> is section 19.3 with an explicit unfold, one sample at a time, and it skips
        the unfold when the filter is 1 × 1 with stride 1 and no padding.
      </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Đọc từ nhánh mặc định của từng dự án ngày 2026-11-20 (ch19-facts D1 đến D4). Số dòng sẽ thay đổi.</p>}>
        <p>Read from each project&rsquo;s default branch on 2026-11-20 (ch19-facts D1 to D4). Line numbers drift.</p>
        </Translated>
      </Beyond>
      <Translated vi={<p>Backend CUDA của llama.cpp, cũng dựng được cho HIP, có <code>conv2d_implicit_gemm_f16</code>: Hình 19.11 trong production. Lưới của nó chính là lưới chúng ta đã suy ra cho Hình 19.11 ở mục 19.3 (sách chỉ nói z là mẫu): x chạy trên các ô pixel đầu ra, y trên các ô kênh đầu ra, z trên batch nhân với số phần của một phép rút gọn được tách. Nó thêm phép chia làm tròn lên và mọi kiểm tra biên mà Hình 19.11 bỏ qua. Nó gom ô đầu vào bằng Eq. 19.5, dùng số chia là hằng cho bộ lọc 3 × 3 và 1 × 1 và một phép chia nhanh được tính trước cho các trường hợp khác, và nó xử lý phần đệm từ trước bằng cách ghi một bản sao đầu vào đã đệm số 0. Backend Metal của nó có một phép mở rộng tường minh, một kernel trực tiếp và <code>kernel_conv_2d_mm</code>, một implicit GEMM trên các ma trận simdgroup. MLX làm việc với kênh nằm cuối và chọn giữa Winograd, implicit GEMM và, như phương án cuối cùng, một phép mở rộng tường minh làm theo từng dải hàng để mỗi dải vừa một buffer Metal.</p>}>
      <p>
        llama.cpp&rsquo;s CUDA backend, which also builds for HIP, has <code>conv2d_implicit_gemm_f16</code>: Fig. 19.11
        in production. Its grid is the one we derived for Fig. 19.11 in section 19.3 (the book states only that z is the
        sample): x over tiles of output pixels, y over tiles of output channels, z over the batch times the pieces of
        a split reduction. It adds ceiling division and every bounds check Fig. 19.11 leaves out. It gathers the input tile with
        Eq. 19.5, using literal divisors for 3 × 3 and 1 × 1 filters and a precomputed fast division otherwise, and it
        handles padding beforehand by writing a zero-padded copy of the input. Its Metal backend has an explicit unfold, a
        direct kernel and <code>kernel_conv_2d_mm</code>, an implicit GEMM on simdgroup matrices. MLX works channels-last
        and picks among Winograd, implicit GEMM and, as the last resort, an explicit unfold done in row tiles so that each
        fits one Metal buffer.
      </p>
      </Translated>
      <Translated vi={<p>Các quy tắc dispatch của llama.cpp và MLX là nơi lập luận của sách ở p. 472, rằng cách diễn đạt GEMM đáng giá khi các ma trận lớn, biến thành mã. Trình khám phá áp dụng chúng cho lớp bạn đặt ở trên.</p>}>
      <p>
        The dispatch rules of llama.cpp and MLX are where the book&rsquo;s argument on p. 472, that the GEMM formulation
        pays when the matrices are large, turns into code. The explorer applies them to the layer you set above.
      </p>
      </Translated>
      <Source section="19.3" pages="472" />
      <Translated vi={<p>Hãy chọn một trong ba cấu hình có sẵn, chốt điều mà backend CUDA của llama.cpp chạy, rồi mới so sánh các backend khác. Sau đó, thay đổi lớp trong trình khám phá ở trên và quan sát những điều kiện nào đổi.</p>}>
      <p>
        Pick one of the three presets, commit to what llama.cpp&rsquo;s CUDA backend runs, and only then compare the
        other backends. After that, change the layer in the explorer above and watch which conditions flip.
      </p>
      </Translated>
      <Simplification>
        <Translated vi={<p>Chỉ có các điều kiện về hình dạng, như được đọc ngày 2026-11-20. Kiểu dữ liệu, kiểm tra phần cứng, điều kiện split-K và mọi thứ khác mà tệp facts không ghi lại đều không được mô hình hóa, và trình khám phá nói rõ điều đó ở chỗ cần thiết.</p>}>
        <p>
          Shape conditions only, as read on 2026-11-20. Data types, hardware checks, the split-K condition and anything
          else the facts file does not record are not modelled, and the explorer says so where it matters.
        </p>
        </Translated>
      </Simplification>
      <DispatchBench layer={layer} setLayer={setLayer} />
      <Ask q={q['ch19/19.4/q-dispatch']} />

      <h2><Translated vi="Các backend khác nhau ở đâu">Where the backends differ</Translated></h2>
      <Translated vi={<p>Các hàng về thư viện trong bảng backend của tệp facts của chương này. Mỗi ô là một sự thật có trong tài liệu hoặc được đánh dấu là chưa kiểm tra; không có gì ở đây được chạy trên thiết bị.</p>}>
      <p>
        The library rows of the backend table in this chapter&rsquo;s facts file. Every cell is a documented fact or is
        marked as not checked; nothing here was run on a device.
      </p>
      </Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-differ>
          <thead>
            <tr>
              <th><Translated vi="Đặc điểm">Feature</Translated></th>
              <th>CUDA</th>
              <th>HIP</th>
              <th>Metal</th>
            </tr>
          </thead>
          <tbody>
            {DIFFER.map(([f, c, h, m]) => (
              <tr key={f()} data-row={f()}>
                <td>{f()}</td>
                <td>{c()}</td>
                <td>{h()}</td>
                <td>{m()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Translated vi={<p>Tiếp theo: <a href={href('ch19', 'labs')}>lab</a>, nơi bạn tự xây con đường tường minh và đo thời gian nó so với kernel trực tiếp, và <a href={href('ch19', 'wrap')}>phần tóm tắt</a> của chương.</p>}>
      <p>
        Next: the <a href={href('ch19', 'labs')}>lab</a>, where you build the explicit route yourself and time it against
        the direct kernel, and the chapter&rsquo;s <a href={href('ch19', 'wrap')}>summary</a>.
      </p>
      </Translated>
    </>
  );
}
