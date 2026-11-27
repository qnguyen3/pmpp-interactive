import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Beyond, Source } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { hOut, lenetLayer, wOut } from '../../sim/cnnlayer';
import { geometry } from '../../sim/convgrid';
import { BOOK_SHAPE, gemmShape, xToB } from '../../sim/unfold';

// A kernel of our own, for one question: the backward pass of Exercise 3 written as a
// scatter. Checked copy: labs/ch19/snippets/wrap_backward_scatter.cu (Wrap.snippets.test.ts
// ties the two together and pins the numbers below to the models).
const SCATTER = `// For a question on chapter 19's summary page (Exercise 3, the backward pass); not built
// or run. The kernel is ours, not the book's. It computes dE/dX, the gradient with respect
// to the layer's input, by giving each thread one element of dE/dY and scattering that
// element's contributions into dE/dX, which is zeroed before the launch. Grid: x covers
// the H_out * W_out pixels of one output map, y the M output maps, z the N samples.
// Layouts as in lab 19.1: NCHW, row-major float. The question asks what is wrong with it.
__global__ void convBackwardXScatter(const float* dY, const float* F, float* dX,
                                     int C, int H, int W, int M, int K) {
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    int n = blockIdx.z;
    if (pixel >= H_out * W_out) return;
    int h = pixel / W_out;
    int w = pixel % W_out;
    float g = dY[((n * M + m) * H_out + h) * W_out + w];
    for (int c = 0; c < C; ++c)
        for (int p = 0; p < K; ++p)
            for (int q = 0; q < K; ++q)
                dX[((n * C + c) * H + h + p) * W + w + q] += g * F[((m * C + c) * K + p) * K + q];
}`;

// ---------------------------------------------------------------- the numbers the questions take from the models
const C5 = lenetLayer('C5')!;
const c5Launch = (division: 'floor' | 'ceil') =>
  geometry({ N: 1, M: C5.M, Hout: hOut(C5), Wout: wOut(C5), tile: 16, division });
/** LeNet C5 on the direct kernel at batch 1, TILE_WIDTH 16 (src/sim/convgrid.ts). */
export const C5_DIRECT = {
  outputs: C5.M * hOut(C5) * wOut(C5),
  threads: c5Launch('ceil').threads,
  printedBlocks: c5Launch('floor').blocks,
  pct: (100 * C5.M * hOut(C5) * wOut(C5)) / c5Launch('ceil').threads,
  inner: C5.C * C5.K * C5.K,
};
/** Threads of the scatter kernel that add into one dX element, book example (src/sim/unfold.ts). */
export const SCATTER_WRITERS = {
  centre: BOOK_SHAPE.M * xToB(BOOK_SHAPE, 0, 1, 1).length,
  corner: BOOK_SHAPE.M * xToB(BOOK_SHAPE, 0, 0, 0).length,
};
/** A depthwise layer, 256 channels of 14 × 14 with 3 × 3 filters, per channel; and the dense layer of the same size. */
export const DEPTHWISE = {
  per: gemmShape({ N: 1, C: 1, H: 14, W: 14, M: 1, K: 3 }),
  dense: gemmShape({ N: 1, C: 256, H: 14, W: 14, M: 256, K: 3 }),
};
/** Fig. 19.5 as printed on LeNet C1's 28 × 28 output maps (src/sim/convgrid.ts). */
const C1 = lenetLayer('C1')!;
const c1Printed = geometry({ N: 1, M: 1, Hout: hOut(C1), Wout: wOut(C1), tile: 16, division: 'floor' });
export const C1_PRINTED = {
  pixels: hOut(C1) * wOut(C1),
  missed: hOut(C1) * wOut(C1) - c1Printed.threads,
};
const r2 = (v: number) => Math.round(v * 100) / 100;
const shape = (g: { rows: number; inner: number; cols: number }) => `${g.rows} × ${g.inner} × ${g.cols}`;

export const questionsWrap: Record<string, Question> = {
  'ch19/wrap/q-c5-direct': {
    kind: 'numeric',
    id: 'ch19/wrap/q-c5-direct',
    prompt: (
      <Translated vi={<>Lớp C5 của LeNet biến 16 bản đồ 5 × 5 thành {C5.M} bản đồ 1 × 1 với bộ lọc 5 × 5. Bạn chạy nó ở batch 1 trên kernel trực tiếp của lab 19.1: <code>TILE_WIDTH</code> 16, số ô được làm tròn lên, mọi đầu ra đều được kiểm tra biên. Bao nhiêu phần trăm số luồng được khởi chạy thực sự tính một đầu ra?</>}>
      <>
        LeNet&rsquo;s C5 turns 16 maps of 5 × 5 into {C5.M} maps of 1 × 1 with 5 × 5 filters. You run it at batch 1 on
        lab 19.1&rsquo;s direct kernel: <code>TILE_WIDTH</code> 16, the tile count rounded up, every output
        bounds-checked. What percentage of the launched threads compute an output?
      </>
      </Translated>
    ),
    answer: r2(C5_DIRECT.pct),
    tol: 0.02,
    unit: '%',
    explain: (
      <Translated vi={<p>Mỗi bản đồ đầu ra là một pixel, nên nó cần một ô: lưới là ({C5.M}, 1, 1) khối 256 luồng,{' '}{C5_DIRECT.threads.toLocaleString('en-US')} luồng cho {C5_DIRECT.outputs} đầu ra, {r2(C5_DIRECT.pct)}%. Hình 19.5 như được in còn tệ hơn: 1 / 16 bị cắt còn 0 ô, nên nó khởi chạy {C5_DIRECT.printedBlocks} khối và không tính gì cả. C5 là một lớp tuyến tính trá hình. Mỗi đầu ra là tích vô hướng của cả {C5_DIRECT.inner} đầu vào của mẫu với một hàng của F, và F vốn đã được lưu dạng ({C5.M}, {C5_DIRECT.inner}). Vì vậy{' '}<code>launchLinear(X, F, Y, N, {C5_DIRECT.inner}, {C5.M})</code>{' '}tính lớp này cho cả batch mà không cần unroll gì cả, và đặt batch vào chỗ mà GEMM có thể dùng lại mỗi trọng số. Mục <a href={href('ch19', '19.2')}>19.2</a> có lưới; cách đọc C5 này là của khóa học này.</p>}>
      <p>
        Each output map is one pixel, so it needs one tile: the grid is ({C5.M}, 1, 1) blocks of 256 threads,{' '}
        {C5_DIRECT.threads.toLocaleString('en-US')} threads for {C5_DIRECT.outputs} outputs, {r2(C5_DIRECT.pct)}%. Fig.
        19.5 as printed is worse: 1 / 16 truncates to 0 tiles, so it launches {C5_DIRECT.printedBlocks} blocks and computes
        nothing. C5 is a linear layer in disguise. Each output is a dot product of all {C5_DIRECT.inner} inputs of the
        sample with one row of F, and F is already stored ({C5.M}, {C5_DIRECT.inner}). So{' '}
        <code>
          launchLinear(X, F, Y, N, {C5_DIRECT.inner}, {C5.M})
        </code>{' '}
        computes the layer for the whole batch with no unroll at all, and puts the batch where the GEMM can reuse each
        weight. Section <a href={href('ch19', '19.2')}>19.2</a> has the grid; the C5 reading is ours.
      </p>
      </Translated>
    ),
    source: 'Figs. 19.5 and 19.7, pp. 461–462; LeNet C5, p. 458; the linear-layer reading is ours',
    vi: { source: 'Figs. 19.5 và 19.7, pp. 461–462; LeNet C5, p. 458; cách đọc như lớp tuyến tính là của khóa học này' },
  },
  'ch19/wrap/q-subsample': {
    kind: 'choice',
    id: 'ch19/wrap/q-subsample',
    prompt: (
      <Translated vi={<>Bài tập 1 yêu cầu lớp lấy mẫu con (subsampling) của LeNet: mỗi pixel đầu ra là trung bình của một cửa sổ 2 × 2 trên một bản đồ đầu vào, các cửa sổ không chồng lấn, và theo sau là một bias riêng cho mỗi bản đồ và một hàm sigmoid. Bạn giao mỗi luồng một pixel đầu ra, như kernel trực tiếp. Đưa các ô đầu vào vào bộ nhớ chia sẻ (shared memory) sẽ được lợi gì?</>}>
      <>
        Exercise 1 asks for LeNet&rsquo;s subsampling layer: each output pixel is the mean of a 2 × 2 window of one
        input map, the windows do not overlap, and a per-map bias and a sigmoid follow. You give each thread one output
        pixel, as the direct kernel does. What would staging input tiles in shared memory gain?
      </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Khoảng 4 lần ít lần nạp toàn cục hơn, vì mỗi đầu ra đọc 4 đầu vào.">About a factor of 4 in global loads, since each output reads 4 inputs.</Translated>,
        why: <Translated vi="Bộ nhớ chia sẻ chỉ tiết kiệm một lần nạp khi có luồng thứ hai cần cùng giá trị. Ở đây không đầu vào nào nằm trong hai cửa sổ.">Shared memory saves a load only when a second thread wants the same value. Here no input is in two windows.</Translated>,
      },
      {
        label: <Translated vi="Sự dùng lại giữa các bản đồ đầu ra, điều mà kernel tích chập trực tiếp không có được.">Reuse across output maps, which the direct convolution kernel could not get.</Translated>,
        why: <Translated vi="Bản đồ đầu ra m của một lớp lấy mẫu con chỉ đọc bản đồ đầu vào m. Không có gì để chia sẻ giữa các bản đồ.">Output map m of a subsampling layer reads input map m only. There is nothing to share across maps.</Translated>,
      },
      {
        label: <Translated vi="Không gì cả: mỗi giá trị đầu vào được một luồng đọc, một lần, vốn đã là mức tối thiểu.">Nothing: each input value is read by one thread, once, which is already the minimum.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Tính đúng đắn: không có nó, 4 luồng cộng vào một đầu ra sẽ gây race condition (tranh chấp dữ liệu).">Correctness: without it the 4 threads that add into one output would race.</Translated>,
        why: <Translated vi="Một luồng sở hữu mỗi đầu ra và cộng 4 đầu vào của nó trong một thanh ghi. Không có hai luồng nào ghi cùng một phần tử.">One thread owns each output and sums its 4 inputs in a register. No two threads write the same element.</Translated>,
      },
    ],
    explain: (
      <Translated vi={<p>Chồng lấn là điều làm phân ô (tiling) đáng giá ở chương 7 và làm B lớn hơn X ở mục{' '}<a href={href('ch19', '19.3')}>19.3</a>. Một cửa sổ có stride bằng kích thước của nó thì không có chồng lấn: tỉ số mở rộng là 1, như thiết lập stride = K trong trình khám phá của 19.3 cho thấy. Giống phép cộng ma trận trong bài tập đầu tiên của chương 5, kernel này về bản chất bị giới hạn bởi bộ nhớ, và việc cần làm là khiến các lần nạp của nó được hợp nhất. Trong một mạng thực tế, bạn sẽ gộp (fuse) nó vào phép tích chập đứng trước, để đầu ra của phép tích chập không bao giờ phải đi một vòng tới bộ nhớ toàn cục rồi quay về.</p>}>
      <p>
        Overlap is what made tiling pay in chapter 7 and what made B larger than X in section{' '}
        <a href={href('ch19', '19.3')}>19.3</a>. A window whose stride equals its size has none: the expansion ratio is 1,
        as the stride = K setting of 19.3&rsquo;s explorer shows. Like the matrix addition of chapter 5&rsquo;s first
        exercise, this kernel is memory-bound by nature, and the work is in making its loads coalesce. In a real network you would fuse it
        into the convolution before it, so the convolution&rsquo;s output never makes the round trip to global memory.
      </p>
      </Translated>
    ),
    source: 'Exercise 1, p. 474; the subsampling layer, Appendix B pp. 616–617 (Fig. B.7); the fusion remark is ours',
    vi: { source: 'Exercise 1, p. 474; lớp lấy mẫu con, Appendix B pp. 616–617 (Fig. B.7); nhận xét về việc gộp là của khóa học này' },
  },
  'ch19/wrap/q-backward': {
    kind: 'choice',
    id: 'ch19/wrap/q-backward',
    prompt: (
      <Translated vi={<>Kernel ở trên là lần thử đầu tiên cho gradient theo đầu vào của Bài tập 3. <code>dX</code> được đặt về 0 trước khi khởi chạy, các chỉ số đều đúng, và nó chạy không lỗi. Nó có đúng không?</>}>
      <>
        The kernel above is a first attempt at the input gradient of Exercise 3. <code>dX</code> is zeroed before the
        launch, the indices are right, and it runs without errors. Is it correct?
      </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Có: mỗi luồng chỉ đọc phần tử dY của riêng nó, nên các luồng độc lập với nhau.">Yes: each thread reads only its own element of dY, so the threads are independent.</Translated>,
        why: <Translated vi="Các lần đọc không phải vấn đề. Hãy xem hai luồng khác nhau ghi vào những phần tử nào của dX.">The reads are not the problem. Look at which elements of dX two different threads write.</Translated>,
      },
      {
        label: <Translated vi="Có, miễn là mỗi khối chứa trọn một bản đồ đầu ra, để không có hai khối nào gặp nhau.">Yes, as long as each block holds one whole output map, so no two blocks meet.</Translated>,
        why: <Translated vi="Các luồng của cùng một bản đồ vẫn va chạm mỗi khi K > 1, vì các pixel đầu ra kề nhau có các vùng chồng lấn, và các khối cho các bản đồ m khác nhau cũng cộng vào cùng các phần tử dX.">{'Threads of one map still collide whenever K > 1, because neighbouring output pixels have overlapping patches, and blocks for different maps m also add into the same dX elements.'}</Translated>,
      },
      {
        label: <Translated vi="Không: nó cần một __syncthreads() trước các vòng lặp, để mọi lần đọc dY xong trước.">No: it needs a __syncthreads() before the loops, so all reads of dY finish first.</Translated>,
        why: <Translated vi="Các luồng va chạm cũng nằm ở các khối khác nhau (m khác nhau), và dù sao một rào chắn (barrier) cũng không làm cho hai lần đọc-cộng-ghi trên một phần tử trở thành nguyên tử.">The colliding threads are also in different blocks (different m), and a barrier would not make two read-add-writes of one element atomic anyway.</Translated>,
      },
      {
        label: <Translated vi="Không: các vùng chồng lấn, và các bản đồ đầu ra khác nhau, cộng vào một phần tử dX mà không có phép toán nguyên tử.">No: overlapping patches, and different output maps, add into one dX element without atomics.</Translated>,
        correct: true,
      },
    ],
    explain: (
      <Translated vi={<p>Trong ví dụ của sách, pixel ở giữa của mỗi bản đồ đầu vào nằm trong 4 vùng của mỗi bản đồ trong 2 bản đồ đầu ra, nên {SCATTER_WRITERS.centre} luồng đọc, cộng và ghi nó; một pixel ở góc có {SCATTER_WRITERS.corner}. Đó chính là nhận xét của mục <a href={href('ch19', '19.2')}>19.2</a> về các vòng lặp c, p và q nhìn từ phía bên kia. Hai cách sửa: biến bước cập nhật thành một <code>atomicAdd</code>, hoặc đảo ngược kernel và giao mỗi luồng một phần tử của dX, gom qua m, p và q các số hạng <code>dY[n, m, h − p, w − q] · F[m, c, p, q]</code> có pixel đầu ra tồn tại. Phép gom có một luồng ghi cho mỗi phần tử và không cần phép toán nguyên tử, và đó là dạng mà Eq. B.22 của Phụ lục B được viết. Biến thể <code>race</code> của trình giả lập, chạy các luồng đồng thời dưới ThreadSanitizer, là công cụ sẽ báo lỗi kernel này; nó chưa được chạy trên kernel này.</p>}>
      <p>
        In the book&rsquo;s example the centre pixel of each input map lies in 4 patches of each of the 2 output maps,
        so {SCATTER_WRITERS.centre} threads read, add and write it; a corner pixel has {SCATTER_WRITERS.corner}. It is
        section <a href={href('ch19', '19.2')}>19.2</a>&rsquo;s remark about the c, p and q loops seen from the other
        side. Two fixes: make the update an <code>atomicAdd</code>, or turn the kernel round and give each thread one
        element of dX, gathering over m, p and q the terms <code>dY[n, m, h − p, w − q] · F[m, c, p, q]</code> whose
        output pixel exists. The gather has one writer per element and no atomics, and it is the form Appendix B&rsquo;s
        Eq. B.22 is written in. The emulator&rsquo;s <code>race</code> flavour, which runs threads concurrently
        under ThreadSanitizer, is the tool that would report this kernel; it was not run on it.
      </p>
      </Translated>
    ),
    source: 'Exercise 3, p. 475; atomics for the inner loops, p. 460; Eq. B.22, Appendix B p. 618; the kernel is ours',
    vi: { source: 'Exercise 3, p. 475; phép toán nguyên tử cho các vòng lặp trong, p. 460; Eq. B.22, Appendix B p. 618; kernel là của khóa học này' },
  },
  'ch19/wrap/q-depthwise': {
    kind: 'choice',
    id: 'ch19/wrap/q-depthwise',
    prompt: (
      <Translated vi={<>Ngoài sách: một tích chập depthwise giao cho mỗi kênh trong 256 kênh một bộ lọc 3 × 3 riêng, nên bản đồ đầu ra m chỉ đọc bản đồ đầu vào m. Trên các bản đồ 14 × 14 không đệm, khi được hạ xuống theo cách của chương, nó là một GEMM nhỏ cho mỗi kênh. Mỗi GEMM có hình dạng gì (số hàng × độ dài rút gọn × số cột), và điều gì xảy ra tiếp theo?</>}>
      <>
        Beyond the book: a depthwise convolution gives each of 256 channels its own 3 × 3 filter, so output map m reads
        input map m only. On 14 × 14 maps with no padding, lowered the chapter&rsquo;s way, it is one small GEMM per
        channel. What shape is each (rows × reduction × columns), and what follows?
      </>
      </Translated>
    ),
    options: [
      {
        get label() { return tr(`${shape(DEPTHWISE.dense)}, the same as a dense layer of this size, so it is just as good a fit.`, `${shape(DEPTHWISE.dense)}, giống một lớp dày đặc cùng kích thước, nên nó cũng phù hợp y như vậy.`); },
        why: <Translated vi="Đó là lớp dày đặc, nơi mỗi bản đồ đầu ra đọc cả 256 bản đồ đầu vào. Ở đây mỗi bản đồ chỉ đọc một.">That is the dense layer, where every output map reads all 256 input maps. Here each reads one.</Translated>,
      },
      {
        get label() { return tr(`${shape(DEPTHWISE.per)}: a matrix-vector product per channel, so each B element is used once.`, `${shape(DEPTHWISE.per)}: mỗi kênh một phép nhân ma trận với vector, nên mỗi phần tử B chỉ được dùng một lần.`); },
        correct: true,
      },
      {
        get label() { return tr(`256 × 9 × ${DEPTHWISE.per.cols}: one GEMM for all channels, with one B shared by every filter.`, `256 × 9 × ${DEPTHWISE.per.cols}: một GEMM cho mọi kênh, với một B dùng chung cho mọi bộ lọc.`); },
        why: <Translated vi="Mỗi kênh mở rộng một bản đồ đầu vào khác nhau, nên không có một B duy nhất cho cả 256 hàng của F.">Each channel unfolds a different input map, so there is no single B for all 256 rows of F.</Translated>,
      },
      {
        get label() { return tr(`1 × 2304 × ${DEPTHWISE.per.cols}, since C·K·K is still 256 · 9.`, `1 × 2304 × ${DEPTHWISE.per.cols}, vì C·K·K vẫn là 256 · 9.`); },
        why: <Translated vi="Mỗi đầu ra đọc một kênh, nên tích vô hướng của nó có 9 số hạng, không phải 2304.">Each output reads one channel, so its dot product has 9 terms, not 2304.</Translated>,
      },
    ],
    explain: (
      <Translated vi={<p>Mỗi kênh một bản đồ đầu ra cho F một hàng: {shape(DEPTHWISE.per)}, so với {shape(DEPTHWISE.dense)} của lớp dày đặc. Một GEMM phân ô dùng lại mỗi phần tử B đã nạp cho các hàng của ô đầu ra, và ở đây chỉ có một hàng, nên 15 trong mỗi 16 hàng của một ô 16 × 16 không chứa gì. Đây là lập luận ở p. 472 của mục{' '}<a href={href('ch19', '19.3')}>19.3</a> ở mức cực đoan: các ma trận nhỏ chịu thiệt. Một kernel trực tiếp với các ô bộ nhớ chia sẻ của chương 7 hợp với một lớp depthwise, vì các cửa sổ của nó chồng lấn và mỗi đầu vào được 9 đầu ra trong cùng kênh của nó dùng.</p>}>
      <p>
        One output map per channel gives F one row: {shape(DEPTHWISE.per)}, against {shape(DEPTHWISE.dense)} for the
        dense layer. A tiled GEMM reuses each loaded B element across the rows of its output tile, and here there is
        one row, so 15 of every 16 rows of a 16 × 16 tile hold nothing. This is section{' '}
        <a href={href('ch19', '19.3')}>19.3</a>&rsquo;s p. 472 argument at its extreme: small matrices lose. A direct
        kernel with chapter 7&rsquo;s shared-memory tiles fits a depthwise layer, since its windows overlap and each
        input is used by 9 outputs of its own channel.
      </p>
      </Translated>
    ),
    source: 'reuse and matrix size, p. 472; tiling, ch. 7 §7.5; depthwise convolution is ours',
    vi: { source: 'sự dùng lại và kích thước ma trận, p. 472; phân ô, ch. 7 §7.5; tích chập depthwise là của khóa học này' },
  },
  'ch19/wrap/q-int8-scale': {
    kind: 'choice',
    id: 'ch19/wrap/q-int8-scale',
    prompt: (
      <Translated vi={<>Ngoài sách: bạn lưu F dưới dạng int8 với một hệ số tỉ lệ float cho mỗi bản đồ đầu ra, còn X vẫn là float. Trong implicit GEMM đã sửa của mục 19.3, đâu là chỗ rẻ nhất và đúng để áp dụng hệ số tỉ lệ?</>}>
      <>
        Beyond the book: you store F as int8 with one float scale per output map, and X stays float. In the corrected
        implicit GEMM of section 19.3, where is the cheapest correct place to apply the scale?
      </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Trong phép gom X: nhân mỗi giá trị của X với hệ số tỉ lệ khi nó được nạp.">In the X gather: multiply each value of X by the scale as it is loaded.</Translated>,
        why: <Translated vi="Một giá trị X đi vào mọi bản đồ đầu ra trong ô, và mỗi bản đồ có hệ số tỉ lệ riêng. Một phép nhân ở đây sẽ cho tất cả chúng cùng một hệ số.">An X value feeds every output map in the tile, and each map has its own scale. One multiply here would give all of them the same one.</Translated>,
      },
      {
        label: <Translated vi="Trong bước nạp ô F: chuyển mỗi int8 sang float và nhân nó với hệ số tỉ lệ của hàng nó.">In the F tile load: convert each int8 to float and multiply it by its row’s scale.</Translated>,
        why: <Translated vi={`Đúng, nhưng nó tốn một phép nhân cho mỗi phần tử F mà một luồng nạp, mỗi pha một lần (${DEPTHWISE.dense.phases} pha trên một lớp 3 × 3 có 256 kênh), cho một hệ số giống hệt nhau dọc cả hàng.`}>{`Correct, but it costs a multiply for every F element a thread loads, once per phase (${DEPTHWISE.dense.phases} phases on a 256-channel 3 × 3 layer), for a factor that is the same along the whole row.`}</Translated>,
      },
      {
        label: <Translated vi="Một lần cho mỗi đầu ra: nhân Pvalue với scale[Row] ngay trước bước ghi.">Once per output: multiply Pvalue by scale[Row] just before the store.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Bên trong vòng lặp trong theo k, để các tích nằm trong phạm vi.">Inside the inner loop over k, so the products stay in range.</Translated>,
        why: <Translated vi="Pvalue là một biến tích lũy float, nên phạm vi không phải vấn đề, và cách này sẽ tốn 16 phép nhân mỗi pha.">Pvalue is a float accumulator, so range is not the issue, and this would cost 16 multiplies per phase.</Translated>,
      },
    ],
    explain: (
      <Translated vi={<p>Hàng m của Y là tích vô hướng của hàng m của F với một cột của B. Một hệ số giống nhau cho mọi phần tử của hàng có thể đưa ra ngoài tổng: Σ s·f·b = s·Σ f·b. Vì vậy các giá trị int8 được chuyển đổi trong bước nạp ô F, phép tổng chạy như trước, và hệ số tỉ lệ được áp dụng một lần cho mỗi đầu ra. Nếu hệ số tỉ lệ thay đổi dọc theo phép rút gọn, chẳng hạn mỗi nhóm kênh đầu vào một hệ số, thì nó không còn đưa ra ngoài được nữa, và chỗ nạp ô chính là nơi nó sẽ nằm. Chỗ đó, nơi Hình 19.11 thực hiện phép gom, cũng là nơi một GEMM lượng tử hóa thực hiện việc giải lượng tử hóa.</p>}>
      <p>
        Row m of Y is the dot product of row m of F with a column of B. A factor that is the same for every element of
        the row comes out of the sum: Σ s·f·b = s·Σ f·b. So the int8 values are converted in the F tile load, the sum
        runs as before, and the scale is applied once per output. If the scale changed along the reduction, one per
        group of input channels say, it would no longer factor out, and the tile-load slot is where it would go. That
        slot, where Fig. 19.11 does its gather, is where a quantised GEMM does its dequantising.
      </p>
      </Translated>
    ),
    source: 'Fig. 19.11, p. 471; quantisation is ours (notes/ch19.md, relevance section)',
    vi: { source: 'Fig. 19.11, p. 471; lượng tử hóa là của khóa học này (notes/ch19.md, phần liên hệ)' },
  },
};

// ---------------------------------------------------------------- the page
export function Wrap() {
  const { progress } = useProgress();
  const { reports, loading } = useReports();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const passing = (id: string) => backends.filter((b) => labStatus(reports[b], id, 'work') === 'pass');
  const lab = passing('ch19/lab1_convgemm');
  const m2 = passing('ch06/lab4_linear');
  const name = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch19/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}><Translated vi="Những gì cần mang theo">What to carry forward</Translated></h2>
      <Translated vi={<p>Phần tóm tắt của sách điểm lại lớp tích chập, kernel trực tiếp, dạng GEMM với đầu vào được unroll ngầm định, và cuDNN, và lưu ý rằng người dùng framework có được tất cả những điều đó mà không phải viết kernel. Những phần cần giữ lại:</p>}>
      <p>
        The book&rsquo;s summary recaps the layer, the direct kernel, the GEMM form with its input unrolled implicitly,
        and cuDNN, and notes that framework users get all of it without writing a kernel. The parts to keep:
      </p>
      </Translated>
      <Source section="19.5" pages="474" />
      <Translated vi={<ul><li>Một lớp tích chập là <code>Y[n, m, h, w] = Σ X[n, c, h + p, w + q] · F[m, c, p, q]</code> trên c, p và q: không lật, không đệm, đầu ra nhỏ hơn đầu vào K − 1 hàng và K − 1 cột. X và Y mang chiều batch; F thì không.</li><li>Bốn trong bảy vòng lặp của nó, n, m, h và w, ghi các đầu ra khác nhau và có thể song song hóa tự do. Ba vòng còn lại cộng vào một đầu ra: giữ chúng tuần tự trong một thanh ghi, hoặc trả giá cho phép toán nguyên tử.</li><li>Kernel trực tiếp giao mỗi luồng một đầu ra trên một lưới (bản đồ đầu ra, ô, mẫu). Hãy làm tròn số ô lên và kiểm tra mọi đầu ra. Nó yêu cầu 0.25 FLOP/B, và nó không thể chia sẻ một ô đầu vào giữa các bản đồ đầu ra vì mỗi khối sở hữu một bản đồ.</li><li>F vốn đã là một ma trận <code>M × C·K·K</code> mà không cần sao chép. Mở rộng đầu vào thành B,{' '}<code>C·K·K × H_out·W_out</code>, và lớp trở thành <code>Y = F · B</code>, vốn đã ở dạng NCHW. Nếu lưu lại, B tốn tới K² lần đầu vào và thêm một lượt đi qua bộ nhớ; nếu gom từng ô với Eq. 19.5, nó chỉ tốn phép tính số nguyên trong bước nạp ô và không gì khác.</li><li>Bước nạp ô của một GEMM phân ô là một chỗ mà một toán hạng có thể được biến đổi mà không động đến vòng lặp trong: một phép gom ở đây, một phép giải lượng tử hóa trong một kernel lượng tử hóa.</li><li>Dạng GEMM đáng giá khi các ma trận lớn. Các lớp nhỏ, các GEMM một hàng và các lần khởi chạy theo từng mẫu có lợi cho kernel trực tiếp, và các thư viện chọn theo từng hình dạng: cuDNN và MIOpen chọn giữa các thuật toán của chúng, llama.cpp theo một quy tắc dựa trên C·K·K và kích thước đầu ra.</li><li>Hãy đọc tên gọi của một thư viện trước các con số của nó: K của cuDNN là M của chương, còn R và S của nó là K của chương. Kích thước đầu ra của nó có một công thức, và lựa chọn thuật toán của nó có thể bao gồm cả độ chính xác.</li></ul>}>
      <ul>
        <li>
          A convolutional layer is <code>Y[n, m, h, w] = Σ X[n, c, h + p, w + q] · F[m, c, p, q]</code> over c, p and q:
          no flip, no padding, an output K − 1 rows and columns smaller than its input. X and Y carry the batch; F does not.
        </li>
        <li>
          Four of its seven loops, n, m, h and w, write different outputs and are free to parallelise. The other three
          add into one output: keep them serial in a register, or pay for atomics.
        </li>
        <li>
          The direct kernel gives each thread one output on a grid of (output map, tile, sample). Round the tile count
          up and check every output. It requests 0.25 FLOP/B, and it cannot share an input tile across output maps
          because each block owns one map.
        </li>
        <li>
          F is already an <code>M × C·K·K</code> matrix with no copy. Unfold the input into B,{' '}
          <code>C·K·K × H_out·W_out</code>, and the layer is <code>Y = F · B</code>, already in NCHW. Stored, B costs up
          to K² times the input and an extra pass; gathered tile by tile with Eq. 19.5, it costs integer arithmetic in
          the tile load and nothing else.
        </li>
        <li>
          The tile load of a tiled GEMM is a slot where an operand can be transformed without touching the inner loop:
          a gather here, a dequantisation in a quantised kernel.
        </li>
        <li>
          The GEMM form pays when the matrices are large. Small layers, one-row GEMMs and per-sample launches favour a
          direct kernel, and libraries choose per shape: cuDNN and MIOpen among their algorithms, llama.cpp by a rule on
          C·K·K and the output size.
        </li>
        <li>
          Read a library&rsquo;s names before its numbers: cuDNN&rsquo;s K is the chapter&rsquo;s M, and its R and S are
          the chapter&rsquo;s K. Its output size has a formula, and its choice of algorithm can include the precision.
        </li>
      </ul>
      </Translated>

      <h2><Translated vi="Kiểm tra tổng hợp">A mixed check</Translated></h2>
      <p>
        <Translated vi={<>Những câu này dùng cả chương cho các lớp và kernel mà các mục chưa trình bày.{asked ? ` Đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu hỏi, tính theo câu trả lời gần nhất của bạn cho mỗi câu.` : ''}</>}>
        These use the whole chapter on layers and kernels the sections did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
        </Translated>
      </p>
      <Beyond>
        <Translated vi={<p>Lớp depthwise, bộ lọc int8 và kernel scatter là của khóa học này, được chọn cho các câu hỏi.</p>}>
        <p>The depthwise layer, the int8 filters and the scatter kernel are ours, picked for the questions.</p>
        </Translated>
      </Beyond>
      <Ask q={q['ch19/wrap/q-c5-direct']} />
      <Ask q={q['ch19/wrap/q-subsample']} label={tr("Exercise 1, as a design question", 'Bài tập 1, dưới dạng một câu hỏi thiết kế')} />
      <Code
        tabs={[
          {
            label: tr('CUDA (not a lab file: for the next question only)', 'CUDA (không phải tệp lab: chỉ cho câu hỏi tiếp theo)'),
            code: SCATTER,
            status: tr('front-end-checked; never run', 'front-end-checked; chưa bao giờ chạy'),
            tiers: { dY: 'global', F: 'global', dX: 'global', g: 'reg', h: 'reg', w: 'reg', pixel: 'reg' },
          },
        ]}
        caption={
          <>
            <Translated vi={<>Từ <code>labs/ch19/snippets/wrap_backward_scatter.cu</code>.</>}>
            From <code>labs/ch19/snippets/wrap_backward_scatter.cu</code>.
            </Translated>
          </>
        }
      />
      <Ask q={q['ch19/wrap/q-backward']} label={tr("Exercise 3, as a design question", 'Bài tập 3, dưới dạng một câu hỏi thiết kế')} />
      <Ask q={q['ch19/wrap/q-depthwise']} />
      <Ask q={q['ch19/wrap/q-int8-scale']} />

      <h2><Translated vi="Các bài tập của sách">The book&rsquo;s exercises</Translated></h2>
      <Translated vi={<p>Chương này kết thúc bằng bốn bài tập (pp. 474–475), và sách không đưa ra lời giải cho bài nào. Hai bài được giải trong các mục; hai bài yêu cầu viết trọn kernel, lớp lấy mẫu con và lượt lùi, được giải ở đây dưới dạng câu hỏi thiết kế và không được xây thành lab.</p>}>
      <p>
        The chapter ends with four exercises (pp. 474–475), and the book gives no solution for any of them. Two are
        worked in the sections; the two that ask for whole kernels, the subsampling layer and the backward pass, are
        worked here as design questions and not built as labs.
      </p>
      </Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }} data-table="exercises">
          <thead>
            <tr>
              <th><Translated vi="Bài tập">Exercise</Translated></th>
              <th><Translated vi="Nội dung">About</Translated></th>
              <th><Translated vi="Được giải ở đâu trong khóa học">Where it is worked here</Translated></th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1</td>
              <td><Translated vi="Lượt tiến của lớp lấy mẫu con">The forward pass of the subsampling layer</Translated></td>
              <td>
                <Translated vi={<>câu hỏi về lấy mẫu con ở trên; tham chiếu tuần tự là Hình B.7 của Phụ lục B (p. 617), và lưới là lưới của <a href={href('ch19', '19.2')}>19.2</a> trên bản đồ nhỏ hơn</>}>
                the subsampling question above; the sequential reference is Appendix B&rsquo;s Fig. B.7 (p. 617), and the
                grid is the one of <a href={href('ch19', '19.2')}>19.2</a> on the smaller map
                </Translated>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td><Translated vi="Bố cục NHWC và CHWN thay cho NCHW">NHWC and CHWN layouts instead of NCHW</Translated></td>
              <td>
                <Translated vi={<>trình khám phá các lần đọc của warp và câu hỏi về bố cục trong <a href={href('ch19', '19.2')}>19.2</a>, kể cả chế độ các luồng kề nhau xử lý các mẫu khác nhau mà CHWN cần</>}>
                the warp-reads explorer and the layout question in <a href={href('ch19', '19.2')}>19.2</a>, including
                the mode with neighbouring threads on different samples that CHWN needs
                </Translated>
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td><Translated vi="Lượt lùi của lớp tích chập">The backward pass of the convolutional layer</Translated></td>
              <td>
                <Translated vi={<>câu hỏi về kernel scatter ở trên, và cách sửa bằng phép gom; Hình B.10 và B.11 của Phụ lục B (pp. 619–620) là các tham chiếu tuần tự</>}>
                the scatter-kernel question above, and its gather fix; Appendix B&rsquo;s Figs. B.10 and B.11 (pp.
                619–620) are the sequential references
                </Translated>
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td><Translated vi="Hợp nhất truy cập và xung đột bank trong Hình 19.11">Coalescing and bank conflicts in Fig. 19.11</Translated></td>
              <td>
                <Translated vi={<>trình khám phá warp và câu hỏi Bài tập 4 trong <a href={href('ch19', '19.3')}>19.3</a>, trên 32 bank mà NVIDIA ghi trong tài liệu</>}>
                the warp explorer and the Exercise 4 question in <a href={href('ch19', '19.3')}>19.3</a>, on NVIDIA&rsquo;s
                documented 32 banks
                </Translated>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2><Translated vi="Những chỗ PDF có vẻ sai">Where the PDF looks wrong</Translated></h2>
      <Translated vi={<p>Bản sao mà khóa học này dựa vào có một vài chỗ có vẻ sơ suất trong chương 19. Ba chỗ nằm trong mã và quan trọng: các mục và lab 19.1 dùng phiên bản đã sửa và nói rõ chúng khác ở đâu. Những chỗ còn lại được liệt kê để bạn không bị bối rối khi đọc sách song song.</p>}>
      <p>
        The copy this course was built from has a few apparent slips in chapter 19. Three are in the code and matter:
        the sections and lab 19.1 use corrected versions and say where they differ. The rest are listed so that you are
        not thrown by them when you read the book alongside.
      </p>
      </Translated>
      <Aside>
        <Translated vi={<p>Đây là cách đọc của chúng tôi trên một bản PDF. Nếu bản của bạn khác, hãy tin bản của bạn.</p>}>
        <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
        </Translated>
      </Aside>
      <Translated vi={<ul data-list="errata"><li>Hình 19.5 (p. 461) chia kích thước đầu ra cho <code>TILE_WIDTH</code> bằng phép cắt bớt, nên đầu ra 28 × 28 của LeNet C1 chỉ được một ô mỗi chiều và {C1_PRINTED.missed} trong {C1_PRINTED.pixels} pixel của nó không bao giờ được tính. Mục{' '}<a href={href('ch19', '19.2')}>19.2</a> và lab làm tròn lên.</li><li>Hình 19.7 (p. 462) không có kiểm tra biên, và chữ ký của nó thiếu H, W và M mà các chỉ số cần khi được tuyến tính hóa. Hình 19.5 (p. 461) cũng để <code>W_grid</code>, <code>H_grid</code> và <code>T</code> không có kiểu, và Hình 19.6 (p. 462) ghi T là <code>W_grid * H_grid</code> trong khi mã có <code>H_grid * W_grid</code> (cùng giá trị).</li><li>Hình 19.11 (p. 471) cắt bớt số pha, nên C·K·K = 25 chỉ chạy một pha và bỏ sót 9 số hạng của mỗi đầu ra; nó không có kiểm tra biên và không trình bày lần khởi chạy. Phần săn lỗi của mục <a href={href('ch19', '19.3')}>19.3</a> cho thấy từng lỗi; lab cung cấp một phiên bản đã sửa và đã kiểm tra.</li><li>p. 468 liệt kê ô đầu ra của Hình 19.9 là bốn pixel của bản đồ đầu ra 0. Khung đỏ và ô B mà nó dùng cho ra <code>Y[n, 0, 0, 0]</code>, <code>Y[n, 0, 0, 1]</code>, <code>Y[n, 1, 0, 0]</code> và{' '}<code>Y[n, 1, 0, 1]</code>: các giá trị 14, 20 / 12, 24.</li><li>p. 469 nói &ldquo;u is equal to 2&rdquo; (u bằng 2) cho <code>B[10, 2]</code>. Thực ra u = 10 và v = 2; số 2 mà phép tính cho ra là kênh c. Cùng trang đó viết <code>B[3.0]</code> thay cho <code>B[3, 0]</code>.</li><li>Eq. 19.1 viết mỗi số hạng là vùng · bộ lọc còn Eq. 19.2 là bộ lọc · vùng, và câu sau Eq. 19.3 nêu các toán hạng theo thứ tự ngược với phương trình. Giá trị, 14, vẫn như nhau. Ở p. 464, &ldquo;the expression for generating the Y0,0,0&rdquo; bỏ mất chỉ số dưới n được dùng ở dòng trước, và &ldquo;the layout of C matrices&rdquo; nghĩa là mảng trong ngôn ngữ C, không phải một ma trận tên là C.</li><li>p. 458 nói tập bộ lọc là tham số mảng <code>W</code>; Hình 19.3 gọi nó là F, còn W là chiều rộng. Trang đó cũng nói một chỉ số giữ chỗ chạy &ldquo;from 0 to K&rdquo; (từ 0 đến K): các chỉ số bộ lọc dừng ở K − 1.</li><li>p. 458 gọi ma trận trọng số của F6 là 120 × 84, trong khi <code>W · X</code> với 120 đầu vào và 84 đầu ra cần 84 × 120 theo thứ tự thông thường. Hình 19.1 ghi C5 là &ldquo;Full connection&rdquo; (kết nối đầy đủ) trong khi phần chữ gọi nó là một lớp tích chập; cả hai đều đúng với một bộ lọc 5 × 5 trên một bản đồ 5 × 5.</li><li>p. 458 để lớp kết nối đầy đủ lại &ldquo;as an exercise&rdquo; (làm bài tập), và Bài tập 1 nói lớp lấy mẫu con được giới thiệu ở mục 19.1. Không bài tập nào yêu cầu cái thứ nhất, và Phụ lục B mô tả cái thứ hai.</li><li>p. 454 hứa hẹn &ldquo;a few optimizations&rdquo; (một vài tối ưu) cho kernel cơ bản. Tối ưu duy nhất được phát triển là dạng GEMM; việc phân ô kernel trực tiếp được để lại cho bạn (p. 463).</li><li>p. 472 nói cuDNN cần dữ liệu nằm trong bộ nhớ device &ldquo;as we discussed earlier in this chapter&rdquo; (như đã bàn ở phần trước của chương); chúng tôi không tìm thấy phần bàn luận nào như vậy. p. 473 chỉ cho kích thước đầu ra dưới dạng <code>P = f(H; R; u; pad_h)</code>; mục{' '}<a href={href('ch19', '19.4')}>19.4</a> đưa ra công thức của cuDNN. Cùng trang đó nêu mục đích của phần đệm là căn chỉnh bộ nhớ hoặc thực thi vector hóa, trong khi cuDNN ghi trong tài liệu <code>pad_h</code> và <code>pad_w</code> là các hàng và cột số 0 quanh đầu vào (lại là 19.4), và xếp Winograd vào nhóm các thuật toán nhân ma trận.</li><li>p. 460 mở đầu bằng việc so sánh &ldquo;training&rdquo; (huấn luyện) một mạng với phép nhân ma trận, trước một mục về đường đi tiến (forward). Phần chữ viết CUDNN trong khi tài liệu tham khảo của nó viết cuDNN, và &ldquo;unfolding&rdquo; của mục 19.3 trở thành &ldquo;unrolling&rdquo; trong phần tóm tắt.</li></ul>}>
      <ul data-list="errata">
        <li>
          Fig. 19.5 (p. 461) divides the output size by <code>TILE_WIDTH</code> with truncation, so LeNet C1&rsquo;s 28 ×
          28 output gets one tile per side and {C1_PRINTED.missed} of its {C1_PRINTED.pixels} pixels are never computed. Section{' '}
          <a href={href('ch19', '19.2')}>19.2</a> and the lab round up.
        </li>
        <li>
          Fig. 19.7 (p. 462) has no bounds check, and its signature lacks the H, W and M that its indices need once they
          are linearised. Fig. 19.5 (p. 461) also leaves <code>W_grid</code>, <code>H_grid</code> and <code>T</code> untyped, and
          Fig. 19.6 (p. 462) labels T as <code>W_grid * H_grid</code> where the code has <code>H_grid * W_grid</code> (the same
          value).
        </li>
        <li>
          Fig. 19.11 (p. 471) truncates the phase count, so C·K·K = 25 runs one phase and drops 9 terms of every output;
          it has no bounds checks and does not show its launch. Section <a href={href('ch19', '19.3')}>19.3</a>&rsquo;s
          erratum hunt shows each failure; the lab provides a corrected, tested version.
        </li>
        <li>
          p. 468 lists the output tile of Fig. 19.9 as four pixels of output map 0. The red box and the B tile it uses
          give <code>Y[n, 0, 0, 0]</code>, <code>Y[n, 0, 0, 1]</code>, <code>Y[n, 1, 0, 0]</code> and{' '}
          <code>Y[n, 1, 0, 1]</code>: the values 14, 20 / 12, 24.
        </li>
        <li>
          p. 469 says &ldquo;u is equal to 2&rdquo; for <code>B[10, 2]</code>. It is u = 10 and v = 2; the 2 that the
          arithmetic produces is the channel c. The same page writes <code>B[3.0]</code> for <code>B[3, 0]</code>.
        </li>
        <li>
          Eq. 19.1 writes each term as patch · filter and Eq. 19.2 as filter · patch, and the sentence after Eq. 19.3
          names its operands in the opposite order to the equation. The value, 14, is the same either way. On p. 464,
          &ldquo;the expression for generating the Y0,0,0&rdquo; drops the n subscript used a line earlier, and
          &ldquo;the layout of C matrices&rdquo; means arrays in the C language, not a matrix named C.
        </li>
        <li>
          p. 458 says the filter bank is the array parameter <code>W</code>; Fig. 19.3 calls it F, and W is the width. It
          also says a placeholder index runs &ldquo;from 0 to K&rdquo;: filter indices stop at K − 1.
        </li>
        <li>
          p. 458 calls F6&rsquo;s weight matrix 120 × 84, where <code>W · X</code> with 120 inputs and 84 outputs needs
          84 × 120 in the usual order. Fig. 19.1 labels C5 &ldquo;Full connection&rdquo; while the text calls it a
          convolutional layer; both are right for a 5 × 5 filter on a 5 × 5 map.
        </li>
        <li>
          p. 458 leaves the fully connected layer &ldquo;as an exercise&rdquo;, and Exercise 1 says the subsampling
          layer is introduced in section 19.1. No exercise asks for the first, and Appendix B describes the second.
        </li>
        <li>
          p. 454 promises &ldquo;a few optimizations&rdquo; of the basic kernel. The only one developed is the GEMM form;
          tiling the direct kernel is left to you (p. 463).
        </li>
        <li>
          p. 472 says cuDNN needs its data in device memory &ldquo;as we discussed earlier in this chapter&rdquo;; we found
          no such discussion. p. 473 gives the output size only as <code>P = f(H; R; u; pad_h)</code>; section{' '}
          <a href={href('ch19', '19.4')}>19.4</a> gives cuDNN&rsquo;s formula. The same page gives padding&rsquo;s purpose
          as memory alignment or vectorised execution, where cuDNN documents <code>pad_h</code> and <code>pad_w</code> as
          zero rows and columns around the input (19.4 again), and groups Winograd under the matrix-multiplication
          algorithms.
        </li>
        <li>
          p. 460 opens by comparing &ldquo;training&rdquo; a network with matrix multiplication, before a section about
          the forward path. The text writes CUDNN where its reference writes cuDNN, and section 19.3&rsquo;s
          &ldquo;unfolding&rdquo; becomes &ldquo;unrolling&rdquo; in the summary.
        </li>
      </ul>
      </Translated>

      <h2><Translated vi="Những gì khóa học này rút gọn">What this course shortened</Translated></h2>
      <Translated vi={<p>Mỗi mục của chương đều có trang riêng ở đây. Chỉ giữ lại một câu hoặc một liên kết: lịch sử của CNN (pp. 454–455); các lớp lấy mẫu con, kết nối đầy đủ và Gaussian của LeNet-5, được nêu tên nhưng không được tính (pp. 455, 458–459, cùng Phụ lục B, phần khóa học này bỏ qua); đường đi lùi (p. 455), chỉ xuất hiện dưới dạng câu hỏi Bài tập 3 ở trên; và tích chập FFT và Winograd, được nêu tên trong bảng của 19.4 (p. 473). Được giải thích nhưng không được xây dựng: phân ô bộ nhớ chia sẻ cho kernel trực tiếp (p. 463), điều mà mục{' '}<a href={href('ch07', '7.5')}>7.5</a> của chương 7 và lab 7.2 dạy; phép nhân ma trận bên trong cuDNN (pp. 473–474), thứ chương 15 xây dựng; và các tích chập có stride, có đệm, có dilation và theo nhóm, xuất hiện trong máy tính của 19.4 và trong các ghi chú bên cạnh các trình khám phá nhưng không có trong mã lab nào. Implicit GEMM của Hình 19.11 được cung cấp sẵn trong lab, đã sửa và đã kiểm tra, thay vì để bạn viết.</p>}>
      <p>
        Every section of the chapter has its own page here. Kept to a sentence or a pointer: the history of CNNs (pp.
        454–455); LeNet-5&rsquo;s subsampling, fully connected and Gaussian layers, named but not computed (pp. 455,
        458–459, with Appendix B, which this course skips); the backward path (p. 455), which appears only as the
        Exercise 3 question above; and FFT and Winograd convolution, named in 19.4&rsquo;s table (p. 473). Explained but
        not built: shared-memory tiling of the direct kernel (p. 463), which chapter 7&rsquo;s section{' '}
        <a href={href('ch07', '7.5')}>7.5</a> and lab 7.2 teach; the matmul inside cuDNN (pp. 473–474), which chapter 15
        builds; and strided, padded, dilated and grouped convolutions, which appear in 19.4&rsquo;s calculator and in
        notes beside the explorers but in no lab code. The implicit GEMM of Fig. 19.11 is provided in the lab, corrected
        and tested, rather than written by you.
      </p>
      </Translated>

      <h2><Translated vi="Dự án xuyên suốt cho đến giờ">The project so far</Translated></h2>
      <p>
        <Translated vi={<>Chương này không thêm mốc dự án nào. Lab 19.1 gọi mốc M2, hàm{' '}<code>launchLinear(X, W, Y, batch, in, out)</code> của lab 6.4, một lần cho mỗi mẫu với F ở vị trí của X và đầu vào đã unroll ở vị trí của W, và không có gì trong GEMM phải thay đổi. Đó là mục đích của các quy ước của dự án: <code>float</code> theo thứ tự hàng, con trỏ device thô, kích thước tường minh, và trọng số lưu dạng (out, in).{' '}</>}>
        This chapter adds no milestone. Lab 19.1 calls milestone M2, lab 6.4&rsquo;s{' '}
        <code>launchLinear(X, W, Y, batch, in, out)</code>, once per sample with F in the place of X and the unrolled
        input in the place of W, and nothing in the GEMM had to change. That is what the project&rsquo;s conventions
        were for: row-major <code>float</code>, raw device pointers, explicit sizes, and weights stored (out, in).{' '}
        </Translated>
        <span data-lab19={loading ? 'loading' : lab.join(',') || 'none'}>
          {loading
            ? tr('Reading reports.', 'Đang đọc báo cáo.')
            : backends.length === 0
              ? tr('No lab results are recorded on this machine yet.', 'Chưa có kết quả lab nào được ghi lại trên máy này.')
              : lab.length
                ? <><Translated vi={<>Các kernel lab 19.1 của bạn qua trên {lab.map(name).join(' và ')} trên máy này.</>}>Your own lab 19.1 kernels pass on {lab.map(name).join(' and ')} on this machine.</Translated></>
                : <><Translated vi={<>Các kernel lab 19.1 của bạn chưa qua trên máy này.</>}>Your own lab 19.1 kernels have not passed on this machine yet.</Translated></>}
        </span>{' '}
        <span data-m2={loading ? 'loading' : m2.join(',') || 'none'}>
          {loading || backends.length === 0
            ? ''
            : m2.length
              ? <><Translated vi={<>M2 của bạn qua trên {m2.map(name).join(' và ')}, nên <code>PMPP_CH19_GEMM=work</code> chạy lớp này trên GEMM của bạn.</>}>Your own M2 passes on {m2.map(name).join(' and ')}, so <code>PMPP_CH19_GEMM=work</code> runs the layer on your GEMM.</Translated></>
              : <><Translated vi={<>M2 của bạn chưa qua; cho đến khi qua, lab gọi bản tham chiếu của lab 6.4.</>}>Your own M2 has not passed yet; until it does, the lab calls lab 6.4&rsquo;s reference.</Translated></>}
        </span>{' '}
        <Translated vi={<>Xem <a href={href('ch19', 'labs')}>các lab</a>.</>}>See the <a href={href('ch19', 'labs')}>labs</a>.</Translated>
      </p>
      <Translated vi={<p><a href={href('ch20')}>Chương 20</a> là mô hình mà khóa học này hướng tới. Các lớp tuyến tính của nó là M2 và M3 như hiện có, điểm attention của nó là các tích ma trận, và các bước decode của nó tạo ra những hình dạng mỏng, batch 1 mà lập luận ở p. 472 của chương này cảnh báo. Các mốc M6 và M7 bắt đầu từ đó.</p>}>
      <p>
        <a href={href('ch20')}>Chapter 20</a> is the model this course is about. Its linear layers are M2 and M3 as they
        stand, its attention scores are matrix products, and its decode steps make the thin, batch-1 shapes that this
        chapter&rsquo;s p. 472 argument warns about. Milestones M6 and M7 begin there.
      </p>
      </Translated>
    </>
  );
}
