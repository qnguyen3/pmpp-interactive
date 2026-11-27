import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { stripWorst } from '../../sim/gemmbanks';
import { SCHEDULES, run } from '../../sim/gemmpipe';
import { readsPerFma } from '../../sim/gemmreads';
import { intensity } from '../../sim/gemmreuse';

// A kernel of our own, for one question: section 15.8's toy loop with its two calls
// swapped. Checked copy: labs/ch15/snippets/wrap_load_first.cu (Wrap.snippets.test.ts
// ties the two together and pins the numbers below to the models).
const LOAD_FIRST = `// Section 15.8's pipelined toy (s158_pipeline_toy.cu) with the two calls in the loop
// swapped: the loads of tile + 1 come before mm on tile. For a question on the
// chapter's summary page; not built or run.
#define THREADS 4

__device__ __forceinline__ void loadTile(float* buf, const float* tiles, unsigned int tile) {
    buf[threadIdx.x] = tiles[tile * THREADS + threadIdx.x];
}

__device__ __forceinline__ void mm(const float* buf, float& sum) {
    for (unsigned int j = 0; j < THREADS; ++j) sum += buf[j];
}

__global__ void loadFirstToy(const float* tiles, float* sums, unsigned int numTiles) {
    __shared__ float buf[2][THREADS];
    float sum = 0.0f;
    if (numTiles == 0) {
        sums[threadIdx.x] = sum;
        return;
    }
    loadTile(buf[0], tiles, 0);
    __syncthreads();
    for (unsigned int tile = 0; tile < numTiles - 1; ++tile) {
        loadTile(buf[(tile + 1) & 1], tiles, tile + 1);
        mm(buf[tile & 1], sum);
        __syncthreads();
    }
    mm(buf[(numTiles - 1) & 1], sum);
    sums[threadIdx.x] = sum;
}`;

// ---------------------------------------------------------------- the numbers the questions take from the models
/** A batch-4 step through a linear layer: a 4-row output tile, 128 wide (src/sim/gemmreuse.ts). */
export const DECODE4 = { m: 4, n: 128, fpb: intensity(4, 128), bound: 0.5 * 4 };
/** Leading dimensions of A_s for bK = 16, §15.5 layout, 32 banks (src/sim/gemmbanks.ts). The A-strip
 *  conflict depends only on the row stride, so the model's eight steps along i are enough. */
export const BK16 = Object.fromEntries(
  [16, 17, 20, 24].map((ld) => [ld, stripWorst({ layout: 'remap', strip: 'A', ld, banks: 32, W: 32 }).degree]),
) as Record<16 | 17 | 20 | 24, number>;
/** The load-first loop under every thread order, 1 to 6 tile pairs (src/sim/gemmpipe.ts). */
export const LOAD_FIRST_OK = SCHEDULES.every((s) =>
  [1, 2, 3, 4, 5, 6].every((n) => run({ threads: 4, tiles: n, variant: 'load-first' }, s).correct),
);
/** 8 x 8 against 4 x 4 thread tiles, 256 threads (src/sim/gemmreads.ts, gemmreuse.ts). */
export const SMALLER = {
  reads8: readsPerFma({ tM: 8, tN: 8, bK: 8 }, 'v2'),
  reads4: readsPerFma({ tM: 4, tN: 4, bK: 8 }, 'v2'),
  fpb128: intensity(128, 128),
  fpb64: intensity(64, 64),
};
const r2 = (v: number) => Math.round(v * 100) / 100;

export const questionsWrap: Record<string, Question> = {
  'ch15/wrap/q-decode4': {
    kind: 'numeric',
    id: 'ch15/wrap/q-decode4',
    prompt: (
      <Translated
        vi={
          <>
            Speculative decoding kiểm tra 4 token nháp cùng lúc, nên mọi lớp tuyến tính chạy ở batch 4. Trong kernel của chương này,
            ô đầu ra của một khối khi đó chỉ cao 4 hàng, bằng batch, và rộng 128. Cường độ tính toán của ô đó với phần tử 4 byte là
            bao nhiêu?
          </>
        }
      >
        <>
          Speculative decoding checks 4 drafted tokens at once, so every linear layer runs at batch 4. In this
          chapter&rsquo;s kernel a block&rsquo;s output tile is then only 4 rows tall, the batch, and 128 wide. What is that
          tile&rsquo;s intensity with 4-byte elements?
        </>
      </Translated>
    ),
    answer: r2(DECODE4.fpb),
    tol: 0.02,
    unit: 'FLOP/B',
    explain: (
      <Translated
        vi={
          <p>
            0.5 · 4 · 128 / (4 + 128) = {r2(DECODE4.fpb)} FLOP/B, và 124 trong 128 hàng của khối không tính gì cả. Một ô rộng hơn
            không cứu được: 0.5·m·n/(m + n) vẫn dưới 0.5·m, ở đây là {DECODE4.bound}, dù n lớn đến đâu. Trọng số đều chỉ được đọc một
            lần dù theo cách nào, nên khi lớp còn bị giới hạn bởi bộ nhớ, kiểm tra bốn token tốn khoảng chừng ấy thời gian như sinh
            một token: đó là canh bạc mà speculative decoding đặt cược. Mục{' '}
            <a href={href('ch15', '15.2')}>15.2</a> có công thức; bối cảnh speculative decoding và cách hiểu đó là của khóa học này.
          </p>
        }
      >
        <p>
          0.5 · 4 · 128 / (4 + 128) = {r2(DECODE4.fpb)} FLOP/B, and 124 of the block&rsquo;s 128 rows compute nothing. A
          wider tile does not rescue it: 0.5·m·n/(m + n) stays below 0.5·m, here {DECODE4.bound}, however large n grows.
          The weights are read once either way, so while the layer is memory-bound, checking four tokens takes about as
          long as generating one: that is the bet speculative decoding makes. Section{' '}
          <a href={href('ch15', '15.2')}>15.2</a> has the formula; the speculative-decoding setting and that reading of it
          are ours.
        </p>
      </Translated>
    ),
    source: 'formula of ch. 15 §15.2, p. 351; the setting is ours',
    vi: { source: "công thức của ch. 15 §15.2, p. 351; bối cảnh là của khóa học này" },
  },
  'ch15/wrap/q-bk16': {
    kind: 'choice',
    id: 'ch15/wrap/q-bk16',
    prompt: (
      <Translated
        vi={
          <>
            Bạn tăng gấp đôi bK lên 16 trong kernel của lab 15.4 (cách bố trí mục 15.5, warp 32 lane). Chiều đầu nào của{' '}
            <code>A_s</code> làm các lần đọc dải A không xung đột trên 32 bank rộng 4 byte?
          </>
        }
      >
        <>
          You double bK to 16 in lab 15.4&rsquo;s kernel (the section 15.5 layout, 32-lane warps). Which leading dimension
          of <code>A_s</code> makes the A-strip reads conflict-free on 32 banks of 4 bytes?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"16: ô rộng 16, nên các hàng của nó đã cách nhau 16 word."}>{'16: the tile is 16 wide, so its rows are already 16 words apart.'}</Translated>,
        why: <Translated vi={`Các hàng lane cách nhau 4 hàng bắt đầu cách nhau 64 word, tức 0 mod 32: một bank cho cả tám, xung đột ${BK16[16]} chiều.`}>{`Lane rows 4 rows apart start 64 words apart, which is 0 mod 32: one bank for all eight, a ${BK16[16]}-way conflict.`}</Translated>,
      },
      { label: <Translated vi={"17."}>{'17.'}</Translated>, correct: true },
      {
        label: <Translated vi={"20: bốn word đệm, giống B_s của dự án."}>{'20: four words of padding, like the project’s B_s.'}</Translated>,
        why: <Translated vi={`Các hàng cách nhau 4 bắt đầu cách nhau 80 word, tức 16 mod 32: tám hàng lane dùng chung hai bank, xung đột ${BK16[20]} chiều. Pitch B_s của lab 15.6 giải quyết một kiểu truy cập khác.`}>{`Rows 4 apart start 80 words apart, 16 mod 32: the eight lane rows share two banks, a ${BK16[20]}-way conflict. The B_s pitch of lab 15.6 solves a different access.`}</Translated>,
      },
      {
        label: <Translated vi={"24: thêm nửa hàng, để các hàng rơi vào các nửa khác nhau của dãy bank."}>{'24: half a row more, so rows land on different halves of the banks.'}</Translated>,
        why: <Translated vi={`Các hàng cách nhau 4 bắt đầu cách nhau 96 word, tức 0 mod 32: xung đột ${BK16[24]} chiều, tệ như không đệm.`}>{`Rows 4 apart start 96 words apart, 0 mod 32: a ${BK16[24]}-way conflict, as bad as no padding.`}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tám hàng lane của một warp đọc các hàng cách nhau 4, nên hàng lane g đọc word 4·g·ld cộng một hằng số. Tám word đó
            rơi vào tám bank khác nhau chỉ khi 4·g·ld mod 32 nhận tám giá trị khác nhau với g = 0 đến 7, điều xảy ra đúng khi
            ld lẻ (các lane trong một hàng lane đọc cùng một word, một lần phát quảng bá). Với bK = 16, lựa chọn lẻ nhỏ nhất là
            17 ({BK16[17] === 1 ? 'không xung đột' : `${BK16[17]} chiều`} trong mô hình), cùng một cột đệm như con số 9 của
            sách cho bK = 8. Trình khám phá của mục <a href={href('ch15', '15.6')}>15.6</a> cho cùng quy tắc với mọi độ rộng;
            trên 64 bank, mô hình chỉ yêu cầu ld không chia hết cho 4.
          </p>
        }
      >
        <p>
          The eight lane rows of a warp read rows 4 apart, so lane row g reads word 4·g·ld plus a constant. The eight land
          in eight different banks only when 4·g·ld mod 32 takes eight different values for g = 0 to 7, which happens
          exactly when ld is odd (the lanes within one lane row read the same word, a broadcast). With bK = 16 the smallest odd choice is 17 ({BK16[17] === 1 ? 'conflict-free' : `${BK16[17]}-way`}{' '}
          in the model), the same one column of padding as the book&rsquo;s 9 for bK = 8. Section{' '}
          <a href={href('ch15', '15.6')}>15.6</a>&rsquo;s explorer gives the same rule for any width; on 64 banks the
          model asks only that ld not be divisible by 4.
        </p>
      </Translated>
    ),
    source: 'rule of ch. 15 §15.6, pp. 361–363, and ch15-facts E1 (our arithmetic with NVIDIA’s bank rule); bK = 16 is ours',
    vi: { source: "quy tắc của ch. 15 §15.6, pp. 361–363, và ch15-facts E1 (phép tính của khóa học này với quy tắc bank của NVIDIA); bK = 16 là của khóa học này" },
  },
  'ch15/wrap/q-load-first': {
    kind: 'choice',
    id: 'ch15/wrap/q-load-first',
    prompt: (
      <Translated
        vi={
          <>
            Kernel ở trên là ví dụ đồ chơi của mục 15.8 với hai lời gọi trong vòng lặp bị đổi chỗ: các lần nạp cặp{' '}
            <code>tile + 1</code> giờ đứng trước <code>mm</code> trên cặp <code>tile</code>, và rào chắn duy nhất vẫn ở cuối vòng lặp.
            Nó còn đúng không?
          </>
        }
      >
        <>
          The kernel above is section 15.8&rsquo;s toy with the two calls in the loop swapped: the loads of pair{' '}
          <code>tile + 1</code> now come before <code>mm</code> on pair <code>tile</code>, and the one barrier stays at the
          end of the iteration. Is it still correct?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Không: các lần nạp của một luồng nhanh giờ đứng trước mm của nó, nên chúng có thể ghi đè lên buffer mà mm của một luồng chậm hơn đang đọc."}>{'No: a fast thread\'s loads now come before its mm, so they can overwrite the buffer a slower thread\'s mm is reading.'}</Translated>,
        why: <Translated vi={"Các lần nạp ghi buf[(tile + 1) & 1] còn mm đọc buf[tile & 1]: hai buffer khác nhau ở mọi vòng lặp."}>{'The loads write buf[(tile + 1) & 1] and mm reads buf[tile & 1]: different buffers in every iteration.'}</Translated>,
      },
      {
        label: <Translated vi={"Có: các lần nạp lấp đầy buffer đã được giải phóng trước rào chắn vừa rồi, và mm đọc buffer mà rào chắn đó đã hoàn tất."}>{'Yes: the loads fill the buffer freed before the last barrier, and mm reads the one that barrier completed.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Không: nó cần một rào chắn thứ hai giữa các lần nạp và mm."}>{'No: it needs a second barrier between the loads and mm.'}</Translated>,
        why: <Translated vi={"Một rào chắn ở đó sẽ sắp các lần nạp trước mm, nhưng mm không đọc gì mà các lần nạp này ghi."}>{'A barrier there would order the loads before mm, but mm does not read anything these loads write.'}</Translated>,
      },
      {
        label: <Translated vi={"Chỉ khi numTiles chẵn, để phần kết đọc buffer mà lần nạp cuối đã lấp đầy."}>{'Only when numTiles is even, so that the epilogue reads the buffer the last load filled.'}</Translated>,
        why: <Translated vi={"Tính chẵn lẻ luân phiên sau mỗi vòng lặp, bất kể số lượng; phần kết đọc (numTiles - 1) & 1 trong cả hai trường hợp."}>{'The parity alternates every iteration, whatever the count; the epilogue reads (numTiles - 1) & 1 either way.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Ở vòng lặp <code>tile</code>, các lần nạp ghi <code>buf[(tile + 1) &amp; 1]</code>, buffer được{' '}
            <code>mm</code> đọc lần cuối ở vòng lặp <code>tile - 1</code>, trước rào chắn của vòng lặp đó; <code>mm</code> đọc{' '}
            <code>buf[tile &amp; 1]</code>, buffer đã được lấp đầy trước cùng rào chắn đó. Thứ tự nào bên trong vòng lặp cũng không
            ảnh hưởng tới tính đúng đắn. Mô hình mức luồng của mục <a href={href('ch15', '15.8')}>15.8</a>{' '}
            {LOAD_FIRST_OK ? 'không tìm thấy tổng sai nào' : 'tìm thấy một tổng sai'} dưới mọi thứ tự luồng của nó cho 1 đến 6 ô.
            Điều thứ tự thay đổi là đoạn mã được giao cho trình biên dịch, đó là lý do lab 15.5 yêu cầu bạn đo cả hai.
          </p>
        }
      >
        <p>
          In iteration <code>tile</code> the loads write <code>buf[(tile + 1) &amp; 1]</code>, which was last read by{' '}
          <code>mm</code> in iteration <code>tile - 1</code>, before that iteration&rsquo;s barrier; <code>mm</code> reads{' '}
          <code>buf[tile &amp; 1]</code>, which was filled before the same barrier. Neither order inside the iteration
          matters to correctness. The thread-level model of section <a href={href('ch15', '15.8')}>15.8</a>{' '}
          {LOAD_FIRST_OK ? 'finds no wrong sum' : 'finds a wrong sum'} under any of its thread orders for 1 to 6 tiles.
          What the order changes is the code the compiler is handed, which is why lab 15.5 asks you to measure both.
        </p>
      </Translated>
    ),
    source: 'Fig. 15.14, p. 367; the variant and the toy are ours (src/sim/gemmpipe.ts)',
    vi: { source: "Fig. 15.14, p. 367; biến thể và ví dụ đồ chơi là của khóa học này (src/sim/gemmpipe.ts)" },
  },
  'ch15/wrap/q-epilogue': {
    kind: 'choice',
    id: 'ch15/wrap/q-epilogue',
    prompt: (
      <Translated
        vi={
          <>
            Các lớp của mô hình bạn tính y = gelu(x Wᵀ + b). Hiện tại bạn chạy M3, rồi một kernel thứ hai đọc Y, cộng b, áp dụng GELU
            và ghi lại Y. Bias và hàm kích hoạt có thể đặt ở đâu để Y được ghi một lần và không bao giờ phải đọc lại?
          </>
        }
      >
        <>
          Your model&rsquo;s layers compute y = gelu(x Wᵀ + b). Today you run M3, then a second kernel that reads Y, adds b,
          applies GELU and writes Y again. Where can the bias and the activation go so that Y is written once and never
          read back?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Vào loadTile: cộng b vào mỗi phần tử của ô X khi nó đến bộ nhớ chia sẻ."}>{'Into loadTile: add b to each element of the X tile as it arrives in shared memory.'}</Translated>,
        why: <Translated vi={"Bias được cộng một lần cho mỗi đầu ra. Cộng nó vào một phần tử đầu vào sẽ thay đổi mọi tích mà phần tử đó tham gia."}>{'The bias is added once per output. Adding it to an input element changes every product that element enters.'}</Translated>,
      },
      {
        label: <Translated vi={"Vào mm: cộng b[col] vào mỗi biến tích lũy ở mọi bước của vòng lặp trong theo k."}>{'Into mm: add b[col] to each accumulator at every step of the inner loop over k.'}</Translated>,
        why: <Translated vi={"Cách đó cộng bias ở mọi bước dọc chiều trong: K lần thay vì một lần."}>{'That adds the bias at every step along the inner dimension: K times instead of once.'}</Translated>,
      },
      {
        label: <Translated vi={"Vào writeTile: cộng b[col] và áp dụng GELU trong thanh ghi trước khi ghi."}>{'Into writeTile: add b[col] and apply GELU in registers before the store.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Không chỗ nào trong M3: gộp một epilogue cần một thư viện như cuBLASLt."}>{'Nowhere in M3: fusing an epilogue needs a library such as cuBLASLt.'}</Translated>,
        why: <Translated vi={"Kernel của bạn giữ mọi đầu ra trong thanh ghi cho đến writeTile. Đó là chỗ dễ gộp nhất trong mọi GEMM, kể cả của bạn."}>{'Your kernel holds every output in registers until writeTile. That is the easiest place in any GEMM to fuse, yours included.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cho đến <code>writeTile</code>, mỗi đầu ra chỉ nằm trong <code>C_r</code> của luồng của nó. Cộng{' '}
            <code>b[col]</code> ở đó (đọc một lần cho mỗi cột luồng sở hữu) và áp dụng GELU tốn vài lệnh cho mỗi đầu ra và tiết kiệm
            cả một lượt thứ hai qua Y: một lần đọc và một lần ghi batch × out float. Dạng GEMM của chính sách, D = αA × B + βC, là
            cùng ý tưởng cho một bias với α = β = 1. Đây là lý do &ldquo;epilogue được gộp&rdquo; để tự viết kernel mà mục{' '}
            <a href={href('ch15', '15.9')}>15.9</a> đưa ra; CUTLASS cũng gọi giai đoạn ghi là epilogue.
          </p>
        }
      >
        <p>
          Until <code>writeTile</code>, each output lives only in its thread&rsquo;s <code>C_r</code>. Adding{' '}
          <code>b[col]</code> there (read once per column the thread owns) and applying GELU costs a few instructions per
          output and saves a whole second pass over Y: one read and one write of batch × out floats. The book&rsquo;s own
          GEMM form, D = αA × B + βC, is the same idea for a bias with α = β = 1. This is the &ldquo;fused epilogue&rdquo;
          reason to write your own kernel that section <a href={href('ch15', '15.9')}>15.9</a> gives; CUTLASS calls the
          store phase the epilogue too.
        </p>
      </Translated>
    ),
    source: 'GEMM form, ch. 15 §15.1, p. 350; writeTile, Fig. 15.7, p. 357; the fusion is ours; CUTLASS epilogue, ch15-facts A12',
    vi: { source: "dạng GEMM, ch. 15 §15.1, p. 350; writeTile, Fig. 15.7, p. 357; phép gộp là của khóa học này; epilogue của CUTLASS, ch15-facts A12" },
  },
  'ch15/wrap/q-triage': {
    kind: 'choice',
    id: 'ch15/wrap/q-triage',
    prompt: (
      <Translated
        vi={
          <>
            Một profiler trên kernel M3 của bạn báo occupancy đạt được 12.5%, pipe nhân-cộng gộp (FMA) bận phần lớn thời gian, và
            hầu như không có xung đột bank bộ nhớ chia sẻ. Một đồng nghiệp đề xuất ô luồng 4 × 4, để thanh ghi cho phép bốn khối mỗi
            SM. Bạn kỳ vọng điều gì?
          </>
        }
      >
        <>
          A profiler on your M3 kernel reports 12.5% achieved occupancy, the fused multiply-add pipe busy most of the
          time, and almost no shared-memory bank conflicts. A colleague proposes 4 × 4 thread tiles, so that the registers
          allow four blocks per SM. What do you expect?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Nhanh hơn: bốn khối mỗi SM cho gấp bốn số warp, và che giấu được gấp bốn độ trễ."}>{'Faster: four blocks per SM give four times the warps, and four times the latency is hidden.'}</Translated>,
        why: <Translated vi={"Che giấu độ trễ giúp ích khi các đơn vị phải chờ. Ở đây pipe đã bận, và hướng dẫn chẩn đoán của NVIDIA gọi warp thêm là phản tác dụng trong trường hợp đó."}>{'Latency hiding helps when the units wait. Here the pipe is already busy, and NVIDIA’s triage guide calls extra warps counter-productive in that case.'}</Translated>,
      },
      {
        label: <Translated vi={"Không tốt hơn, có lẽ còn tệ hơn: các đơn vị đã bận, và ô nhỏ hơn dùng lại mỗi giá trị được nạp ít hơn."}>{'No better, probably worse: the units are already busy, and smaller tiles reuse each loaded value less.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Nhanh hơn, vì khi đó mỗi khối cần ít bộ nhớ chia sẻ hơn, nên vừa được nhiều khối hơn."}>{'Faster, because each block then needs less shared memory, so more of them fit.'}</Translated>,
        why: <Translated vi={"Bộ nhớ chia sẻ chưa bao giờ là giới hạn (mục 15.7). Tính theo mỗi đầu ra, ô nhỏ hơn nạp nhiều hơn, không phải ít hơn."}>{'Shared memory was never the limit (section 15.7). Per output, smaller tiles load more, not less.'}</Translated>,
      },
      {
        label: <Translated vi={"Không thay đổi: trong một kernel bị giới hạn bởi tính toán, occupancy không ảnh hưởng đến tốc độ."}>{'No change: in a compute-bound kernel, occupancy has no effect on speed.'}</Translated>,
        why: <Translated vi={"Quá mạnh. Đề xuất này thay đổi nhiều hơn occupancy: nó thay đổi mức độ mỗi giá trị được nạp được dùng lại."}>{'Too strong. The proposal changes more than occupancy: it changes how much each loaded value is reused.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Với 256 luồng, ô luồng 4 × 4 tạo thành ô khối 64 × 64: cường độ tính toán giảm từ{' '}
            {r2(SMALLER.fpb128)} xuống {r2(SMALLER.fpb64)} FLOP/B, và số lần đọc chia sẻ mỗi phép nhân-cộng của mỗi luồng tăng từ{' '}
            {r2(SMALLER.reads8)} lên {r2(SMALLER.reads4)}. Sự đánh đổi của mục <a href={href('ch15', '15.7')}>15.7</a> cố ý đi theo
            chiều ngược lại. Hãy nâng occupancy khi các đơn vị chờ dữ liệu, không phải khi chúng đang bận.
          </p>
        }
      >
        <p>
          With 256 threads, 4 × 4 thread tiles make a 64 × 64 block tile: the intensity falls from{' '}
          {r2(SMALLER.fpb128)} to {r2(SMALLER.fpb64)} FLOP/B, and each thread&rsquo;s shared reads per multiply-add rise
          from {r2(SMALLER.reads8)} to {r2(SMALLER.reads4)}. Section <a href={href('ch15', '15.7')}>15.7</a>&rsquo;s trade
          runs the other way on purpose. Raise occupancy when the units wait for data, not when they are busy.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.7, pp. 363–364; Nsight Compute triage guide, ch15-facts A3; the profile is made up',
    vi: { source: "ch. 15 §15.7, pp. 363–364; hướng dẫn chẩn đoán Nsight Compute, ch15-facts A3; bản profile là giả định" },
  },
};

// ---------------------------------------------------------------- the page
export function Wrap() {
  const { progress } = useProgress();
  const { reports, loading } = useReports();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const m3 = backends.filter((b) => labStatus(reports[b], 'ch15/lab6_linear', 'work') === 'pass');
  const name = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch15/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>
        <Translated vi="Những gì cần mang theo">What to carry forward</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương này áp dụng các tối ưu hóa của chương 6 lên một kernel, và mỗi tối ưu hóa gỡ bỏ nút thắt mà tối ưu hóa trước đó để
            lộ ra. Những phần cần giữ lại:
          </p>
        }
      >
        <p>
          The chapter applied chapter 6&rsquo;s optimisations to one kernel, and each one removed the bottleneck the
          previous one exposed. The parts to keep:
        </p>
      </Translated>
      <Source section="15.10" pages="370" />
      <ul>
        <Translated
          vi={
            <li>
              Một khối sở hữu một ô đầu ra m × n thực hiện 0.5·m·n/(m + n) FLOP cho mỗi byte đầu vào nó nạp. Độ sâu của một bước triệt
              tiêu, nên k có thể nhỏ tùy ý, và không gì về ô giúp được khi m bằng 1: một lớp tuyến tính batch 1 bị giới hạn bởi bộ nhớ
              dù viết thế nào.
            </li>
          }
        >
          <li>
            A block that owns an m × n output tile does 0.5·m·n/(m + n) FLOP per byte of input it loads. The depth of a
            step cancels, so k is free to be small, and nothing about the tile helps when m is 1: a batch-1 linear layer
            is memory-bound however it is written.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Ô lớn nghĩa là luồng được gộp việc (thread coarsening): 256 luồng sở hữu một ô 128 × 128, mỗi luồng 64 đầu ra. Những đầu
              ra đó chỉ ở lại trong thanh ghi nếu mọi chỉ số đều là hằng, nên mọi vòng lặp chạm vào chúng đều được trải hết và mọi hàm
              phụ trợ đều được inline.
            </li>
          }
        >
          <li>
            Large tiles mean coarse threads: 256 threads own a 128 × 128 tile, 64 outputs each. Those outputs stay in
            registers only if every index is a constant, so every loop that touches them is fully unrolled and every
            helper inlined.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Phân ô (tiling) có tác dụng ở mọi cấp nơi hai bộ nhớ khác nhau về tốc độ. Phân ô thanh ghi là một phép hoán đổi vòng lặp:
              đi theo chiều trong ở vòng ngoài cùng, giữ một dải của A và một dải của B trong thanh ghi, và làm một cập nhật hạng 1.
            </li>
          }
        >
          <li>
            Tiling works at every level where two memories differ in speed. Register tiling is a loop interchange: walk
            the inner dimension outermost, hold one strip of A and one of B in registers, and do a rank-1 update.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Mỗi cách sửa có thể dời nút thắt đi chỗ khác. Ánh xạ lại các luồng vào ô con 4 × 4 làm các lần ghi được hợp nhất và gây
              ra một xung đột bộ nhớ chia sẻ; một cột đệm (mọi chiều đầu lẻ, trên 32 bank) loại bỏ nó, cùng với một dòng mã mà sách
              không nhắc tới.
            </li>
          }
        >
          <li>
            Each fix can move the bottleneck. Re-mapping threads to 4 × 4 sub-tiles makes the stores coalesce and raises a
            shared-memory conflict; one padding column (any odd leading dimension, on 32 banks) removes it, with one line
            of code the book does not mention.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Kernel này cố ý chạy ở occupancy thấp: thanh ghi là giới hạn, và tái sử dụng đáng giá hơn số warp. Truy cập rộng hơn, trải
              hết vòng lặp và một vòng lặp pipeline thay thế cho những warp mà nó từ bỏ.
            </li>
          }
        >
          <li>
            This kernel runs at low occupancy on purpose: registers are the limit, and reuse is worth more than warps.
            Wider accesses, full unrolling and a pipelined loop stand in for the warps it gives up.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Hai buffer loại bỏ phụ thuộc giả; xoay vòng lặp đặt việc nạp cặp kế tiếp và phép tính trên cặp hiện tại vào cùng một vòng
              lặp, dưới một rào chắn. Bản thân sự chồng lấp là việc của trình biên dịch, hoặc của phần cứng với sao chép bất đồng bộ ở
              backend có hỗ trợ.
            </li>
          }
        >
          <li>
            Two buffers remove the false dependence; rotating the loop puts the loads of the next pair and the arithmetic
            on this one in the same iteration, under one barrier. The overlap itself is the compiler&rsquo;s to make, or
            the hardware&rsquo;s with asynchronous copies where a backend has them.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Hãy gọi thư viện trừ khi nó thiếu phép toán của bạn, và đằng nào cũng đo so với nó. Với FP32, khoảng cách nằm ở kỹ thuật
              (sao chép bất đồng bộ, chọn kernel theo hình dạng, tinh chỉnh sâu hơn; trên AMD CDNA có thể là lệnh ma trận FP32), và những
              gì mỗi backend cho phép mã của chính bạn dùng là khác nhau. Tensor core chỉ vào cuộc khi bạn chủ động chọn độ chính xác
              thấp hơn (mục 15.9).
            </li>
          }
        >
          <li>
            Call the library unless it lacks your operation, and measure against it either way. In FP32 the gap is
            technique (asynchronous copies, per-shape kernel selection, deeper tuning; on AMD CDNA possibly FP32 matrix
            instructions), and what each backend lets your own code use differs. Tensor cores enter only when you opt
            into a lower precision (section 15.9).
          </li>
        </Translated>
      </ul>

      <h2>
        <Translated vi="Bài kiểm tra tổng hợp">A mixed check</Translated>
      </h2>
      <p>
        <Translated vi="Các câu này dùng cả chương cho những tình huống mà các mục chưa cho thấy.">
          These use the whole chapter on situations the sections did not show.
        </Translated>
        {asked
          ? tr(
              ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.`,
              ` Đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu hỏi, tính theo câu trả lời mới nhất của bạn cho mỗi câu.`,
            )
          : ''}
      </p>
      <Beyond>
        <Translated vi={<p>Các bối cảnh, kernel đồ chơi và bản profile trong những câu hỏi này là của khóa học này, được chọn riêng cho các câu hỏi.</p>}>
          <p>The settings, the toy kernel and the profile in these questions are ours, picked for the questions.</p>
        </Translated>
      </Beyond>
      <Ask q={q['ch15/wrap/q-decode4']} />
      <Ask q={q['ch15/wrap/q-bk16']} />
      <Code
        tabs={[
          {
            label: tr('CUDA (not a lab file: for the next question only)', 'CUDA (không phải tệp lab: chỉ cho câu hỏi tiếp theo)'),
            code: LOAD_FIRST,
            status: tr('front-end-checked; never run', 'front-end-checked; chưa từng chạy'),
            tiers: { tiles: 'global', sums: 'global', buf: 'shared', sum: 'reg' },
          },
        ]}
        caption={
          <Translated vi={<>Từ <code>labs/ch15/snippets/wrap_load_first.cu</code>.</>}>
            <>From <code>labs/ch15/snippets/wrap_load_first.cu</code>.</>
          </Translated>
        }
      />
      <Ask q={q['ch15/wrap/q-load-first']} />
      <Ask q={q['ch15/wrap/q-epilogue']} />
      <Ask q={q['ch15/wrap/q-triage']} />

      <h2>
        <Translated vi="Các bài tập của sách">The book&rsquo;s exercises</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương kết thúc bằng ba bài tập lập trình (p. 370), và sách không đưa lời giải cho bài nào. Mỗi bài được giải ở đây, thành
            một lab có kiểm thử và benchmark.
          </p>
        }
      >
        <p>
          The chapter ends with three coding exercises (p. 370), and the book gives no solution for any of them. Each is
          worked here, as a lab with tests and a benchmark.
        </p>
      </Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }} data-table="exercises">
          <thead>
            <tr>
              <th>
                <Translated vi="Bài tập">Exercise</Translated>
              </th>
              <th>
                <Translated vi="Về">About</Translated>
              </th>
              <th>
                <Translated vi="Được giải ở đâu">Where it is worked here</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1</td>
              <td>
                <Translated vi={<>Lệnh nạp vector trong <code>loadTile</code></>}>
                  Vector loads in <code>loadTile</code>
                </Translated>
              </td>
              <Translated
                vi={
                  <td>
                    chế độ vector của trình chạy từng bước loadTile ở <a href={href('ch15', '15.3')}>15.3</a>, đánh dấu các địa chỉ không
                    căn chỉnh; lab 15.4 trên <a href={href('ch15', 'labs')}>trang lab</a>, và lab 15.6 cho ô W
                  </td>
                }
              >
                <td>
                  the loadTile stepper&rsquo;s vector mode in <a href={href('ch15', '15.3')}>15.3</a>, which flags
                  misaligned addresses; lab 15.4 on the <a href={href('ch15', 'labs')}>labs page</a>, and lab 15.6 for the
                  W tile
                </td>
              </Translated>
            </tr>
            <tr>
              <td>2</td>
              <td>
                <Translated vi={<>Lệnh ghi vector trong <code>writeTile</code></>}>
                  Vector stores in <code>writeTile</code>
                </Translated>
              </td>
              <Translated
                vi={
                  <td>
                    trình khám phá kiểu ghi ở <a href={href('ch15', '15.5')}>15.5</a>, cho thấy vì sao chỉ riêng lệnh ghi vector vẫn để lại
                    khoảng trống; lab 15.3
                  </td>
                }
              >
                <td>
                  the store-pattern explorer in <a href={href('ch15', '15.5')}>15.5</a>, which shows why vector stores
                  alone leave gaps; lab 15.3
                </td>
              </Translated>
            </tr>
            <tr>
              <td>3</td>
              <td>
                <Translated vi="Kernel với bốn ô con 4 × 4 mỗi luồng của mục 15.5">
                  The kernel with section 15.5&rsquo;s four 4 × 4 sub-tiles per thread
                </Translated>
              </td>
              <Translated
                vi={
                  <td>
                    bộ chọn lane và hoạt hình có thuyết minh ở <a href={href('ch15', '15.5')}>15.5</a>, cho warp rộng 32 và 64; lab 15.3, rồi
                    các lab 15.4 đến 15.6, vốn giữ nguyên cách bố trí
                  </td>
                }
              >
                <td>
                  the lane picker and the narrated animation in <a href={href('ch15', '15.5')}>15.5</a>, for 32- and 64-wide
                  warps; lab 15.3, then labs 15.4 to 15.6, which keep the layout
                </td>
              </Translated>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>
        <Translated vi="Những chỗ bản PDF có vẻ sai">Where the PDF looks wrong</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bản sách dùng để xây khóa học này có vài lỗi rõ ràng và chỗ bỏ ngỏ trong chương 15. Chúng được liệt kê để bạn không bị bối
            rối khi đọc sách song song.
          </p>
        }
      >
        <p>
          The copy this course was built from has a few apparent slips and loose ends in chapter 15. They are listed so
          that you are not thrown by them when you read the book alongside.
        </p>
      </Translated>
      <Aside>
        <Translated vi={<p>Đây là cách đọc của chúng tôi trên một bản PDF. Nếu bản của bạn khác, hãy tin bản của bạn.</p>}>
          <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
        </Translated>
      </Aside>
      <ul data-list="errata">
        <Translated
          vi={
            <li>
              pp. 362–363: đệm <code>A_s</code> thay đổi khai báo của nó và chiều đầu truyền cho{' '}
              <code>loadTile</code> và <code>mm</code>, nhưng offset bắt đầu của luồng <code>&amp;A_s[tRow*bK]</code>{' '}
              cũng phải đổi sang bK + 1, và sách không nói điều đó. Để nguyên như bản in, các luồng đọc sai hàng. Xem{' '}
              <a href={href('ch15', '15.6')}>15.6</a>; các phép kiểm thử của lab 15.4 bắt được lỗi này.
            </li>
          }
        >
          <li>
            pp. 362–363: padding <code>A_s</code> changes its declaration and the leading dimension passed to{' '}
            <code>loadTile</code> and <code>mm</code>, but the thread&rsquo;s start offset <code>&amp;A_s[tRow*bK]</code>{' '}
            must change to bK + 1 as well, and the book does not say so. Left as printed, threads read the wrong rows. See{' '}
            <a href={href('ch15', '15.6')}>15.6</a>; lab 15.4&rsquo;s tests catch it.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 363: &ldquo;B không cần thay đổi&rdquo; chỉ đúng trong cách bố trí của mục 15.5. Trong cách bố trí của mục 15.3 và 15.4,
              lần đọc dải B là xung đột 4 chiều trên 32 bank, đó là lý do các lab ánh xạ lại trước khi đệm.
            </li>
          }
        >
          <li>
            p. 363: &ldquo;B needs no change&rdquo; holds in section 15.5&rsquo;s layout only. In the layout of sections
            15.3 and 15.4 the B-strip read is a 4-way conflict on 32 banks, which is why the labs remap before they pad.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 365 nói các lệnh đầu tiên &ldquo;load the input tiles A0 and B0 from shared memory&rdquo; (nạp các ô đầu vào A0 và B0 từ
              bộ nhớ chia sẻ); ý đúng là từ bộ nhớ toàn cục vào bộ nhớ chia sẻ, như p. 366 nói.
            </li>
          }
        >
          <li>
            p. 365 says the first instructions &ldquo;load the input tiles A0 and B0 from shared memory&rdquo;; from global
            memory into shared memory is meant, as p. 366 says.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              pp. 352 và 354: phần văn mô tả một khối 16 × 16 luồng, nhưng Hình 15.3 tính cả hai tọa độ ô chỉ từ{' '}
              <code>threadIdx.x</code>, điều cần một khối một chiều 256 luồng. Mục{' '}
              <a href={href('ch15', '15.3')}>15.3</a> cho thấy một lần khởi chạy 16 × 16 sẽ làm gì.
            </li>
          }
        >
          <li>
            pp. 352 and 354: the prose describes a block of 16 × 16 threads, but Fig. 15.3 computes both tile coordinates
            from <code>threadIdx.x</code> alone, which needs a one-dimensional block of 256. Section{' '}
            <a href={href('ch15', '15.3')}>15.3</a> shows what a 16 × 16 launch would do.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Hình 15.14 (p. 367) là một sơ đồ: các đối số của nó bị lược bớt và theo thứ tự khác với Hình 15.5 và 15.6, các buffer và{' '}
              <code>numTiles</code> của nó không được khai báo, và các kiểu con trỏ 2D của nó không khớp với các ô một chiều của Hình
              15.3. Lab 15.5 có một phiên bản đã biên dịch và kiểm thử.
            </li>
          }
        >
          <li>
            Fig. 15.14 (p. 367) is a schematic: its arguments are elided and in another order than Figs. 15.5 and 15.6,
            its buffers and <code>numTiles</code> are not declared, and its 2D pointer types do not match Fig. 15.3&rsquo;s
            one-dimensional tiles. Lab 15.5 has a compiled, tested version.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Hình 15.3 khai báo các ô chia sẻ bên trong thân vòng lặp (dòng 23 và 24). Các lab khai báo chúng một lần, ở đầu kernel.
            </li>
          }
        >
          <li>
            Fig. 15.3 declares the shared tiles inside the loop body (lines 23 and 24). The labs declare them once, at the
            top of the kernel.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 362: các bank được nêu cho &ldquo;các phần tử thứ hai&rdquo; (1 khi không đệm; 1, 5, …, 29 khi có đệm) khớp với cột 1
              của cùng các hàng, tức bước kế tiếp dọc chiều trong, không phải hàng kế tiếp. Cùng trang đó mời kiểm tra &ldquo;all the 16
              elements accessed by each 4-thread group&rdquo; (cả 16 phần tử mà mỗi nhóm 4 luồng truy cập) mà không nói là 16 phần tử
              nào.
            </li>
          }
        >
          <li>
            p. 362: the banks quoted for the &ldquo;second elements&rdquo; (1 unpadded; 1, 5, …, 29 padded) fit column 1 of
            the same rows, the next step along the inner dimension, not the next row. The same page invites a check of
            &ldquo;all the 16 elements accessed by each 4-thread group&rdquo; without saying which 16.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 364 nói lệnh nạp và ghi vector đã được thấy &ldquo;in Section 15.3&rdquo; (ở mục 15.3); mục đó chỉ nêu tên chúng như bài
              tập, và không có mã vector hóa nào xuất hiện trong chương.
            </li>
          }
        >
          <li>
            p. 364 says vector loads and stores were seen &ldquo;in Section 15.3&rdquo;; that section only names them as
            exercises, and no vectorised code appears in the chapter.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              pp. 359 và 364 ghi công cho phân ô thanh ghi việc nâng cường độ tính toán. Theo phép đếm của mục 15.2, cường độ tính toán
              đến từ kích thước ô đầu ra, thứ mà thanh ghi giữ; phân ô thanh ghi cho đầu vào cắt giảm số lần đọc bộ nhớ chia sẻ.
            </li>
          }
        >
          <li>
            pp. 359 and 364 credit register tiling with raising arithmetic intensity. By section 15.2&rsquo;s count the
            intensity comes from the size of the output tile, which registers hold; register tiling of the inputs cuts
            shared-memory reads.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 353 nói một mảng cục bộ mặc định được đặt &ldquo;in the global memory&rdquo; (trong bộ nhớ toàn cục); chương 5 và NVIDIA
              gọi nơi đó là bộ nhớ cục bộ (local memory), về vật lý nằm trong bộ nhớ thiết bị.
            </li>
          }
        >
          <li>
            p. 353 says a local array is placed &ldquo;in the global memory&rdquo; by default; chapter 5 and NVIDIA call
            the place local memory, physically in device memory.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Lỗi nhỏ: dòng đầu của Hình 15.6 bị cắt trước dấu phẩy sau <code>unsigned int k</code> (Hình 15.9 có hiện nó); Hình 15.7
              không có dấu ngoặc mở; p. 363 viết <code>bk</code> trong khi mã có{' '}
              <code>bK</code>; p. 369 có &ldquo;multidimensional tenors&rdquo; thay cho tensors.
            </li>
          }
        >
          <li>
            Small slips: Fig. 15.6&rsquo;s first line is clipped before the comma after <code>unsigned int k</code> (Fig.
            15.9 shows it); Fig. 15.7 shows no opening brace; p. 363 writes <code>bk</code> where the code has{' '}
            <code>bK</code>; p. 369 has &ldquo;multidimensional tenors&rdquo; for tensors.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Chương không báo cáo số đo nào cho bước nào và không cho thấy mã host hay cấu hình khởi chạy nào. Các lab bổ sung cả hai.
            </li>
          }
        >
          <li>
            The chapter reports no measurement for any step and shows no host code or launch configuration. The labs fill
            both in.
          </li>
        </Translated>
      </ul>
      <Translated
        vi={
          <p>
            Một số phát biểu gắn với thời điểm trong mục 15.7 và 15.9 đã thay đổi kể từ khi sách được viết, và một phát biểu không
            thể xác nhận. Mục <a href={href('ch15', '15.7')}>15.7</a> và <a href={href('ch15', '15.9')}>15.9</a> đi qua từng điều;
            tóm tắt:
          </p>
        }
      >
        <p>
          Several dated statements in sections 15.7 and 15.9 have moved since the book was written, and one could not be
          confirmed. Sections <a href={href('ch15', '15.7')}>15.7</a> and <a href={href('ch15', '15.9')}>15.9</a> go
          through them; in short:
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch15-facts A3, A5 to A9)', '2026-11-20 (ch15-facts A3, A5 đến A9)')}>
        <Translated
          vi={
            <p>
              2048 luồng thường trú mỗi SM chỉ đúng trên một số GPU (1536 trên RTX 30 đến 50). CUDA 13 nhắm vào compute capability 7.5
              trở lên. WMMA vẫn được hỗ trợ; WGMMA chỉ có trên Hopper; TMEM không có trên RTX 50. Không trang nào của NVIDIA nói nvcc gộp
              một lệnh nạp và ghi vector thành LDGSTS. cuTile giờ là CUDA Tile.
            </p>
          }
        >
          <p>
            2048 resident threads per SM holds only on some GPUs (1536 on RTX 30 to 50). CUDA 13 targets compute
            capability 7.5 and later. WMMA is still supported; WGMMA is Hopper-only; TMEM is not on RTX 50. No NVIDIA page
            says nvcc fuses a vector load and store into LDGSTS. cuTile is now CUDA Tile.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Những gì khóa học này rút gọn">What this course shortened</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mỗi mục của chương có trang riêng ở đây. Chỉ giữ lại một câu hoặc một con trỏ: ví dụ đồ họa của mục 15.1 (p. 350); α và β
            của dạng GEMM đầy đủ, thứ sách cũng bỏ qua (p. 350); các tổ hợp thứ tự lưu trữ khác với cặp thứ tự hàng của sách và trọng
            số (out, in) của dự án; và các ô mức GPU và mức cụm máy của hệ phân cấp (pp. 352, 359–360). Được giải thích nhưng không
            xây: chuyên biệt hóa warp (warp specialisation, p. 368), thứ cần đồng bộ hóa producer và consumer mà sách không cho thấy;
            sao chép bất đồng bộ (p. 369), thứ chỉ CUDA công bố ở dạng dùng được và các trình giả lập không mô hình hóa; tensor core
            và các giao diện của chúng (pp. 368–369), khác nhau theo nhà cung cấp (mục{' '}
            <a href={href('x2', 'X2.5')}>X2.5</a> nêu tên các dạng số nguyên của chúng; không lab nào dùng chúng); và cuDNN, chỉ nêu
            tên (mục <a href={href('ch19', '19.3')}>19.3</a> hạ tích chập xuống GEMM, và <a href={href('ch19', '19.4')}>19.4</a> nói
            về cuDNN).
          </p>
        }
      >
        <p>
          Every section of the chapter has its own page here. Kept to a sentence or a pointer: the graphics example of
          section 15.1 (p. 350); the α and β of the full GEMM form, which the book also leaves out (p. 350); storage-order
          combinations other than the book&rsquo;s row-major pair and the project&rsquo;s (out, in) weights; and the GPU-
          and cluster-level tiles of the hierarchy (pp. 352, 359–360). Explained but not built: warp specialisation (p.
          368), which needs producer and consumer synchronisation the book does not show; asynchronous copies (p. 369),
          which only CUDA documents in a usable form and the emulators do not model; tensor cores and their interfaces
          (pp. 368–369), which differ by vendor (section{' '}
          <a href={href('x2', 'X2.5')}>X2.5</a> names their integer forms; no lab uses them); and cuDNN, named only (section <a href={href('ch19', '19.3')}>19.3</a> lowers convolution to GEMM,
          and <a href={href('ch19', '19.4')}>19.4</a> is about cuDNN).
        </p>
      </Translated>

      <h2>
        <Translated vi="Dự án đến lúc này">The project so far</Translated>
      </h2>
      <p>
        <Translated
          vi={
            <>
              Lab 15.6 là mốc dự án M3: lớp tuyến tính của dự án với kernel của mục 15.3 đến 15.8 bên trong, đằng sau cùng{' '}
              <code>launchLinear(X, W, Y, batch, in, out)</code> như M2, và được đo so với M2 và thư viện của nhà cung cấp.
            </>
          }
        >
          Lab 15.6 is milestone M3: the project&rsquo;s linear layer with the kernel of sections 15.3 to 15.8 inside,
          behind the same <code>launchLinear(X, W, Y, batch, in, out)</code> as M2, and benchmarked against M2 and the
          vendor library.
        </Translated>{' '}
        <span data-m3={loading ? 'loading' : m3.join(',') || 'none'}>
          {loading
            ? tr('Reading reports.', 'Đang đọc báo cáo.')
            : backends.length === 0
              ? tr('No lab results are recorded on this machine yet.', 'Chưa có kết quả lab nào được ghi lại trên máy này.')
              : m3.length
                ? (
                  <Translated vi={<><code>launchLinear</code> của chính bạn đã qua trên {m3.map(name).join(' và ')} trên máy này.</>}>
                    <>Your own <code>launchLinear</code> passes on {m3.map(name).join(' and ')} on this machine.</>
                  </Translated>
                )
                : (
                  <Translated vi={<><code>launchLinear</code> của chính bạn chưa qua trên máy này.</>}>
                    <>Your own <code>launchLinear</code> has not passed on this machine yet.</>
                  </Translated>
                )}
        </span>{' '}
        <Translated
          vi={
            <>
              Xem <a href={href('ch15', 'labs')}>các lab</a>. Nên gọi M2 hay M3 ở một kích thước batch nhất định là câu trả lời
              của benchmark cho thiết bị của bạn, không phải của chương.
            </>
          }
        >
          See the <a href={href('ch15', 'labs')}>labs</a>. Which of M2 and M3 to call at a given batch size is the
          benchmark&rsquo;s answer for your device, not the chapter&rsquo;s.
        </Translated>
      </p>
      <Translated
        vi={
          <p>
            Phần 3 của sách bắt đầu với <a href={href('ch16')}>chương 16</a>, về những phép tính mà các ô phải chờ các ô trước đó. Với
            dự án, nơi tiếp theo dùng kernel này là <a href={href('ch20')}>chương 20</a>: các tích Q Kᵀ và P V của attention là các
            GEMM của mục 15.1, và mốc dự án M6 xây trên chúng.
          </p>
        }
      >
        <p>
          The book&rsquo;s Part 3 begins with <a href={href('ch16')}>chapter 16</a>, on computations whose cells wait for
          earlier cells. For the project, the next user of this kernel is <a href={href('ch20')}>chapter 20</a>: the Q Kᵀ
          and P V products of attention are section 15.1&rsquo;s GEMMs, and milestone M6 builds on them.
        </p>
      </Translated>
    </>
  );
}
