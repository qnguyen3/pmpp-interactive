import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { fmt, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr, useLocale } from '../../lib/i18n';
import {
  BOOK_DEVICE,
  M8_DEFAULT,
  PROJ_1024,
  PROJ_4096,
  WEIGHT_BYTES,
  batchRange,
  bStar,
  bStarSearch,
  bookThroughput,
  intensity,
  kneeOf,
  m8Shape,
  measuredDevices,
  precisionFor,
  projectionSeries,
  stepAt,
  stepBStar,
  stepCurve,
  stepSeries,
  thresholdOf,
  weightBytes,
  type Device,
  type Padding,
  type Point,
  type ReportsLike,
  type Series,
  type Shape,
  type StepResult,
  type WeightBytes,
} from '../../sim/batching';
import { MIXES, SHORT_MAX, makeRequests, serve, type Mix, type Policy, type ServeResult } from '../../sim/serving';
import { ContinuousBatchingScene } from './ContinuousBatchingScene';
import { DecodeStepScene } from './DecodeStepScene';

// Section 22.5, Batching: latency versus throughput (book pp. 539–540). The book's part is
// the two motivations, the LLM batching argument with its two throughput formulas, and the
// CPU-versus-GPU remark. The roofline arithmetic (notes/today/ch22-facts.md group E), the
// serving bridge (D1 to D4) and the library notes (A11, B3, C3, D6) are ours and labelled.

// ---------------------------------------------------------------- questions
export const questionsS225: Record<string, Question> = {
  'ch22/22.5/q-scaling': {
    kind: 'choice',
    id: 'ch22/22.5/q-scaling',
    prompt: (
      <Translated vi={<>Động cơ nào trong hai động cơ của mục này là weak scaling, theo nghĩa của định luật Gustafson?</>}>
        <>Which of the section's two motivations is weak scaling, in the sense of Gustafson's law?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Động cơ thứ hai: hoàn thành nhiều thể hiện hơn trong thời gian có được.">The second: more instances finished in the time available.</Translated>, correct: true },
      { label: <Translated vi={"Động cơ thứ nhất: một bài toán, bản phân tích 10 giờ, xong sớm hơn."}>The first: one problem, the 10-hour analysis, finished sooner.</Translated>, why: <Translated vi={"Bài toán cố định, nhiều bộ xử lý hơn, ít thời gian hơn: đó là strong scaling, bối cảnh của định luật Amdahl."}>Fixed problem, more processors, less time: that is strong scaling, the setting of Amdahl's law.</Translated> },
      { label: <Translated vi={"Cả hai: bất kỳ mức tăng tốc nào cho phép hoàn thành nhiều việc hơn đều là weak scaling."}>Both: any speed-up that lets more work finish is weak scaling.</Translated>, why: <Translated vi={"Hai cách khác nhau ở thứ được giữ cố định: kích thước bài toán, hay thời gian."}>The two differ in what is held fixed: the problem size, or the time.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Strong scaling giữ bài toán cố định và hỏi nó xong sớm hơn bao nhiêu (bản phân tích 10 giờ phải còn 4). Weak scaling
            để bài toán lớn lên cùng cỗ máy và hỏi thêm được bao nhiêu trong cùng thời gian (thêm lần chạy bản phân tích, thêm
            truy vấn của người dùng). Việc gắn hai khái niệm này với hai động cơ của sách là của khóa học này.
          </p>
        }
      >
        <p>
          Strong scaling keeps the problem fixed and asks how much sooner it finishes (the 10-hour analysis that must take
          4). Weak scaling lets the problem grow with the machine and asks how much more fits in the same time (more runs of
          the analysis, more users' queries). The mapping onto the book's two motivations is ours.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.5, p. 539; CUDA Best Practices Guide §4.1.3 (ch22-facts A9)', // i18n-ignore: citation
  },
  'ch22/22.5/q-plateau': {
    kind: 'choice',
    id: 'ch22/22.5/q-plateau',
    prompt: (
      <Translated
        vi={
          <>
            Lab 22.2 chạy với M3 (GEMM của lab 15.6) cho thấy thời gian mỗi bước phẳng từ batch 1 đến 128. Điều đó có chứng tỏ
            bước này bị giới hạn bởi băng thông trên cả khoảng đó không?
          </>
        }
      >
        <>
          Lab 22.2 run with M3 (lab 15.6's GEMM) shows a time per step that is flat from batch 1 to 128. Does that show the
          step is bandwidth-bound across that range?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Tự nó thì không. Ô (tile) khối của M3 cao 128 hàng và tính 128 hàng bất kể B, nên dưới 128 cả lưới lẫn khối lượng phép tính của nó đều không đổi."}>Not by itself. M3's block tile is 128 rows tall and computes 128 rows whatever B is, so below 128 neither its grid nor its arithmetic changes.</Translated>,
        correct: true,
      },
      { label: <Translated vi={"Có: thời gian mỗi bước giữ phẳng khi thêm hàng là dấu hiệu của một bước bị giới hạn bởi băng thông."}>Yes: a time per step that stays flat as rows are added is the signature of a bandwidth-bound step.</Translated>, why: <Translated vi={"Một đường phẳng nghĩa là có thứ cố định quyết định thời gian. Với M3, đó có thể là 128 hàng phép tính mà ô vẫn làm dù sao đi nữa."}>A flat line means something fixed sets the time. With M3 it can be the 128 rows of arithmetic the tile does anyway.</Translated> },
      { label: <Translated vi={"Không: một đường phẳng nghĩa là bước bị giới hạn bởi tính toán, với các đơn vị số học là chi phí cố định."}>No: a flat line means the step is compute-bound, with the arithmetic units as the fixed cost.</Translated>, why: <Translated vi={"Có thể là cả hai. Chỉ riêng độ phẳng không cho biết chi phí cố định nào quyết định thời gian."}>It may be either. Flatness alone cannot tell which fixed cost sets the time.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lượng tử hóa theo ô (tile quantisation): khi batch không lấp đầy ô khối, các hàng đệm vẫn tốn phép tính của chúng
            (ch22-facts E3, từ hướng dẫn nhân ma trận của NVIDIA). Để phân biệt hai nguyên nhân, hãy so với M2, có ô 16 hàng tạo
            ra các bậc 16, và tính băng thông trọng số hiệu dụng, tức số byte trọng số mỗi bước chia cho thời gian đo được, so
            với băng thông của lab 5.1. Trong trình khám phá ở trên, đặt phần đệm thành 128 hàng và xem các phép chiếu nằm ở chế
            độ nào tại batch 1.
          </p>
        }
      >
        <p>
          Tile quantisation: when the batch does not fill the block tile, the padding rows still cost their arithmetic
          (ch22-facts E3, from NVIDIA's matrix multiplication guide). To tell the two causes apart, compare with M2, whose
          16-row tile makes steps of 16, and compute an effective weight bandwidth, weight bytes per step divided by the
          measured time, against lab 5.1's bandwidth. In the explorer above, set the padding to 128 rows and look at the
          regime the projections are in at batch 1.
        </p>
      </Translated>
    ),
    source: 'ch22-facts E3; labs/ch15/lab6_linear (bM = 128)', // i18n-ignore: citation
  },
  'ch22/22.5/q-half-bytes': {
    kind: 'choice',
    id: 'ch22/22.5/q-half-bytes',
    prompt: (
      <Translated vi={<>Trọng số và activation chuyển từ float32 sang 16-bit. Điểm gãy B* của phép chiếu dịch đi đâu?</>}>
        <>Weights and activations go from float32 to 16-bit. Where does the projection's knee B* move?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Còn khoảng một nửa: từ khoảng 2T hàng xuống khoảng T hàng.">To about half: from about 2T rows to about T rows.</Translated>, correct: true },
      { label: <Translated vi={"Nó tăng gấp đôi, lên khoảng 4T hàng: giờ mỗi hàng mang ít byte hơn để chia sẻ trọng số."}>It doubles, to about 4T rows: each row now brings fewer bytes to share the weights with.</Translated>, why: <Translated vi={"Giảm một nửa số byte làm số phép toán mỗi byte tăng gấp đôi ở mọi batch, nên ngưỡng đạt được sớm hơn, không phải muộn hơn."}>Halving the bytes doubles the operations per byte at every batch, so the threshold is reached sooner, not later.</Translated> },
      { label: <Translated vi={"Nó không dịch chuyển: khối lượng phép tính vẫn vậy."}>It does not move: the arithmetic is the same.</Translated>, why: <Translated vi={"Khối lượng phép tính vẫn vậy còn số byte giảm một nửa. Cường độ tính toán là tỉ số của hai đại lượng đó."}>The arithmetic is the same and the bytes are halved. Intensity is the ratio of the two.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Khi B nhỏ hơn nhiều so với K và N, I(B) khoảng B/2 FLOP/B với float32 và khoảng B với 16-bit, nên ngưỡng T đạt được
            gần 2T hàng so với gần T hàng. Dưới điểm gãy, bước chính là luồng đọc trọng số, nên thời gian của nó cũng giảm một
            nửa: đó là lập luận băng thông cho trọng số lượng tử hóa mà module X2 và mốc dự án M9 đi sâu hơn.
          </p>
        }
      >
        <p>
          While B is much smaller than K and N, I(B) is about B/2 FLOP/B in float32 and about B in 16-bit, so the threshold
          T is reached near 2T rows against near T rows. Below the knee the step is the weight stream, so its time also
          halves: the bandwidth argument for quantised weights that module X2 and milestone M9 take further.
        </p>
      </Translated>
    ),
    source: 'ch22-facts E1 (our arithmetic)',
    vi: { source: "ch22-facts E1 (phép tính của khóa học này)" },
  },
  'ch22/22.5/q-attention': {
    kind: 'choice',
    id: 'ch22/22.5/q-attention',
    prompt: (
      <Translated vi={<>Những phần nào của một bước decode làm nhiều phép toán hơn trên mỗi byte khi batch lớn lên?</>}>
        <>Which parts of a decode step do more operations per byte as the batch grows?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Các phép chiếu có trọng số: Q, K, V, đầu ra attention, MLP và LM head.">The projections that carry weights: Q, K, V, the attention output, the MLP and the LM head.</Translated>, correct: true },
      { label: <Translated vi={"Mọi phần của bước, vì giờ mọi kernel xử lý B hàng trong một lần khởi chạy."}>Every part of the step, since every kernel now processes B rows in one launch.</Translated>, why: <Translated vi={"Bias, residual, softmax và argmax làm một lượng việc cố định cho mỗi phần tử của mỗi hàng, bất kể có bao nhiêu hàng (E2)."}>Bias, residual, softmax and argmax do a fixed amount of work per element of each row, however many rows there are (E2).</Translated> },
      { label: <Translated vi={"Attention nhiều nhất, đó là lý do p. 539 nêu các attention head là thứ được batching làm tăng cường độ."}>Attention most of all, which is why p. 539 names the attention heads as what batching makes more intense.</Translated>, why: <Translated vi={"Bản thân attention đọc key và value riêng của từng request: không có gì trong đó được chia sẻ giữa các request, nên không có gì được tái sử dụng."}>Attention proper reads each request's own keys and values: nothing in it is shared between requests, so nothing is reused.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Batching có lợi ở chỗ một byte được nạp có thể phục vụ nhiều hàng, và chỉ trọng số là được chia sẻ. Lượng đọc K và V
            của attention tăng theo batch, mỗi request một cache, giống các phần theo từng phần tử. Cụm "attention heads" ở trang
            539 chỉ đúng nếu "attention" bao gồm cả phép chiếu QKV và phép chiếu đầu ra, một định nghĩa mở rộng mà các tác giả
            của Orca nêu trong một chú thích. Chương 20 (p. 505) nói bản thân các lớp attention không được lợi.
          </p>
        }
      >
        <p>
          Batching pays where one fetched byte can serve many rows, and only the weights are shared. Attention's reads of K
          and V grow with the batch, one cache per request, like the elementwise pieces. Page 539's "attention heads" is
          right only if "attention" includes the QKV and output projections, an extended definition Orca's authors name
          in a footnote. Chapter 20 (p. 505) says the attention layers themselves do not benefit.
        </p>
      </Translated>
    ),
    source: 'ch. 22 p. 539 against ch. 20 p. 505; ch22-facts A6 (Orca, OSDI 2022)',
    vi: { source: "ch. 22 p. 539 so với ch. 20 p. 505; ch22-facts A6 (Orca, OSDI 2022)" },
  },
  'ch22/22.5/q-who-pays': {
    kind: 'choice',
    id: 'ch22/22.5/q-who-pays',
    prompt: (
      <Translated
        vi={
          <>
            Tám người dùng đang chờ token kế tiếp. Phương án A: decode từng người một, xoay vòng, nên một vòng là tám bước batch
            1. Phương án B: một bước ở batch 8, tốn gấp 3 lần một bước batch 1. Phương án nào cho mỗi người dùng ít thời gian
            hơn giữa các token?
          </>
        }
      >
        <>
          Eight users are waiting for their next token. Option A: decode one user at a time, round-robin, so one round is
          eight batch-1 steps. Option B: one step at batch 8, which takes 3 times as long as a batch-1 step. Which gives each
          user less time between tokens?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="B: 3 đơn vị mỗi token thay vì 8.">B: 3 units per token instead of 8.</Translated>, correct: true },
      { label: <Translated vi={"A, vì mỗi bước của nó ngắn hơn."}>A, because each of its steps is shorter.</Translated>, why: <Translated vi={"Mỗi người dùng nhận một token mỗi vòng, và một vòng của A dài tám bước."}>Each user gets one token per round, and a round of A is eight steps long.</Translated> },
      { label: <Translated vi={"Như nhau: khối lượng việc như nhau."}>The same: the work is the same.</Translated>, why: <Translated vi={"Khối lượng phép tính như nhau; lưu lượng trọng số thì không. A đọc trọng số tám lần mỗi vòng, B chỉ một lần."}>The arithmetic is the same; the weight traffic is not. A reads the weights eight times per round, B once.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            "Độ trễ dài hơn" mà sách nói cho mỗi truy vấn được so với một server chỉ có một người dùng, nơi một bước batch 1 là
            tất cả những gì ai đó phải chờ. So với việc phục vụ cùng những người dùng đó lần lượt từng người, batching thắng với
            mọi người dùng, và thắng càng nhiều khi batch càng nằm xa dưới điểm gãy.
          </p>
        }
      >
        <p>
          The book's "longer latency" for each query is measured against a server with a single user, where a batch-1
          step is all anyone waits. Against serving the same users one at a time, batching wins for every user, and by
          more the further below the knee the batch is.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.5, p. 539; our arithmetic',
    vi: { source: "ch. 22 §22.5, p. 539; phép tính của khóa học này" },
  },
  'ch22/22.5/q-cpu-gpu': {
    kind: 'choice',
    id: 'ch22/22.5/q-cpu-gpu',
    prompt: (
      <Translated
        vi={
          <>
            Một server có hai nhánh decode. Nhánh A chạy một request mỗi bước, và một bước tốn t. Nhánh B luôn chạy một bước 64
            hàng, đệm các hàng trống, và một bước tốn 4t. Mỗi nhánh phục vụ mục tiêu thiết kế nào, và một người dùng đến khi
            server đang trống nhận được gì từ mỗi nhánh?
          </>
        }
      >
        <>
          A server has two decode paths. Path A runs one request per step, and a step takes t. Path B always runs a step of
          64 rows, padding empty ones, and a step takes 4t. Which design goal does each serve, and what does a user who
          arrives at an empty server get from each?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"A phục vụ độ trễ, B phục vụ thông lượng. Một người dùng đơn lẻ nhận một token mỗi t trên A và mỗi 4t trên B; với 64 người dùng, B cho số token mỗi giây gấp 16 lần A."}>A serves latency, B throughput. A lone user gets a token every t on A and every 4t on B; with 64 users, B gives 16 times A's tokens per second.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"B phục vụ cả hai. Với 64 người dùng nó cho số token mỗi giây gấp 16 lần A, nên ngay cả một người dùng đơn lẻ trên server trống cũng nhận mỗi token sớm hơn trên B."}>B serves both. With 64 users it delivers 16 times A's tokens per second, so even a lone user on an empty server gets each token sooner on B.</Translated>,
        why: <Translated vi={"Con số 16 lần là số token mỗi giây của device, chia cho 64 hàng. Một người dùng đơn lẻ lấp một hàng của B và vẫn phải chờ 4t cho mỗi token, so với t trên A."}>The 16 times is tokens per second for the device, spread over 64 rows. A lone user fills one row of B and still waits 4t for each token, against t on A.</Translated>,
      },
      {
        label: <Translated vi={"A phục vụ thông lượng còn B phục vụ độ trễ. A xong 4 bước trong thời gian B xong một, nên nó tạo ra nhiều token mỗi giây hơn."}>A serves throughput and B latency. A finishes 4 steps in the time B finishes one, so it produces more tokens per second.</Translated>,
        why: <Translated vi={"A xong nhiều bước hơn, nhưng mỗi bước chỉ cho một token. Một bước đầy của B cho 64 token trong 4t, tức 16 mỗi t so với 1 của A."}>A finishes more steps, but each yields one token. A full step of B yields 64 tokens in 4t, which is 16 per t against A's 1.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Nhánh A là mục tiêu thiết kế của CPU trong phép so sánh của sách: thời gian ngắn nhất cho mỗi phần tử, ở đây là token
            của mỗi người dùng. Nhánh B là của GPU: nhiều việc nhất mỗi giây trên nhiều phần tử, đổi lại mỗi phần tử tốn lâu
            hơn. Cái nào tốt hơn phụ thuộc vào tải, không phụ thuộc vào nhánh. Best Practices Guide của NVIDIA nói về phía phần
            cứng theo cùng cách: lõi CPU "minimize latency for a small number of threads", còn GPU xử lý nhiều luồng nhẹ "to
            maximize throughput".
          </p>
        }
      >
        <p>
          Path A is the CPU's design goal in the book's analogy: the shortest time for each element, here each user's token.
          Path B is the GPU's: the most work per second over many elements, paid for by each one taking longer. Which is
          better depends on the load, not on the path. NVIDIA's Best Practices Guide puts the hardware side the same way:
          CPU cores "minimize latency for a small number of threads", GPUs handle many lightweight threads "to maximize
          throughput".
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.5, pp. 539–540; CUDA Best Practices Guide §3.1 (ch22-facts A8); our arithmetic',
    vi: { source: "ch. 22 §22.5, pp. 539–540; CUDA Best Practices Guide §3.1 (ch22-facts A8); phép tính của khóa học này" },
  },
  'ch22/22.5/q-continuous': {
    kind: 'choice',
    id: 'ch22/22.5/q-continuous',
    prompt: (
      <Translated vi={<>Vì sao continuous batching (batching ở mức vòng lặp) quan trọng với sự đánh đổi của mục này?</>}>
        <>Why does continuous (iteration-level) batching matter for this section's trade-off?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Nó nhận và trả request ở mỗi bước, nên batch luôn gần giới hạn slot; khi đó mỗi người dùng trả thời gian bước của batch đó cho mỗi token."}>It admits and retires requests every step, so the batch stays near the slot limit; each user then pays that batch's step time per token.</Translated>,
        correct: true,
      },
      { label: <Translated vi={"Nó làm attention được batch: các request được decode cùng nhau dùng chung một lượt đọc key và value."}>It makes attention batch: requests decoded together share one pass over the keys and values.</Translated>, why: <Translated vi={"Bản thân attention vẫn đọc K và V riêng của từng request. Orca, nơi giới thiệu kỹ thuật này, chạy attention theo từng request."}>Attention proper still reads each request's own K and V. Orca, which introduced the technique, runs attention per request.</Translated> },
      { label: <Translated vi={"Nó xóa bỏ cái giá của batching về thời gian mỗi token, vì không request nào phải chờ các request khác trong batch của nó xong."}>It removes batching's cost in time per token, because no request waits for the others in its batch to finish.</Translated>, why: <Translated vi={"Nó xóa bỏ việc chờ các request khác xong. Mỗi bước vẫn tốn thời gian của batch mà nó mang."}>It removes the waiting for other requests to finish. Each step still takes the time of the batch it carries.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Static batching lãng phí slot cho các request đã xong và bắt request mới phải chờ, nên batch mà server thực sự chạy
            nhỏ hơn giới hạn của nó. Continuous batching giữ nó đầy, và đó là điều đặt một server bận rộn vào đúng chỗ đường
            cong của mô hình chỉ ra: số token mỗi giây gần trần, và thời gian mỗi token của mỗi người dùng do bước ở batch đó
            quyết định.
          </p>
        }
      >
        <p>
          Static batching wastes slots on finished requests and makes new ones wait, so the batch the server actually runs
          is smaller than its limit. Continuous batching keeps it full, which is what puts a busy server where the model's
          curve says: tokens per second near their ceiling, and each user's time per token set by the step at that batch.
        </p>
      </Translated>
    ),
    source: 'ch22-facts D1 to D4 (ours, not the book\'s)',
    vi: { source: "ch22-facts D1 đến D4 (của khóa học này, không phải của sách)" },
  },
  'ch22/22.5/q-batched-gemm': {
    kind: 'choice',
    id: 'ch22/22.5/q-batched-gemm',
    prompt: (
      <Translated
        vi={
          <>
            <code>cublasGemmStridedBatchedEx</code> chạy nhiều GEMM trong một lời gọi. Đó có phải là cách cài đặt việc batch các
            truy vấn người dùng của sách không?
          </>
        }
      >
        <>
          <code>cublasGemmStridedBatchedEx</code> runs many GEMMs in one call. Is it how the book's batching of user queries
          is implemented?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Không. Sách xếp chồng các hàng của request thành một GEMM với M = B nhân với một ma trận trọng số dùng chung; batched GEMM dành cho nhiều cặp độc lập."}>No. The book stacks the requests' rows into one GEMM with M = B against one shared weight matrix; batched GEMM is for many independent pairs.</Translated>,
        correct: true,
      },
      { label: <Translated vi={"Có: mỗi request một thể hiện của lời gọi batched, mỗi thể hiện nhân hàng của request đó với ma trận trọng số, tất cả trong một lời gọi."}>Yes: one instance of the batched call per request, each multiplying that request's row by the weight matrix, all in one call.</Translated>, why: <Translated vi={"Mỗi thể hiện sẽ mang bản sao riêng của toán hạng trọng số, một phép nhân ma trận với vector cho mỗi request: chính là lưu lượng trọng số thừa mà batching sinh ra để loại bỏ."}>Each instance would carry its own copy of the weight operand, a matrix-vector product per request: the redundant weight traffic batching exists to remove.</Translated> },
      { label: <Translated vi={"Có trên HIP: hipBLAS định nghĩa batched GEMM là một phép nhân trên các hàng xếp chồng, khác với cuBLAS."}>Yes on HIP: hipBLAS defines its batched GEMM as one product over the stacked rows, unlike cuBLAS.</Translated>, why: <Translated vi={"hipBLAS có cùng các entry point batched như cuBLAS, và ở đó chúng mang cùng ý nghĩa: C_i = A_i B_i cho mỗi thể hiện i."}>hipBLAS has the same batched entry points as cuBLAS, and they mean the same thing there: C_i = A_i B_i for each instance i.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Batched GEMM hợp với những phép nhân không dùng chung toán hạng, chẳng hạn điểm attention của mỗi chuỗi với key riêng
            của nó. Backend CUDA của llama.cpp gửi phép nhân một trọng số 2 chiều với một batch token tới một GEMM thường và
            dành các lời gọi batched cho toán hạng 3 chiều và 4 chiều.
          </p>
        }
      >
        <p>
          Batched GEMM fits products that do not share an operand, such as each sequence's attention scores against its own
          keys. llama.cpp's CUDA backend sends a 2-D weight times a batch of tokens to one plain GEMM and keeps the batched
          calls for 3-D and 4-D operands.
        </p>
      </Translated>
    ),
    source: 'ch22-facts A11 (cuBLAS), B3 (hipBLAS), C3 (MPS), D6 (llama.cpp)', // i18n-ignore: citation
  },
};

// ---------------------------------------------------------------- shared setup
type ShapeKey = 'm8' | 'p4096' | 'p1024';
interface Setup {
  devKey: string;
  custom: { peak: number; bw: number };
  shapeKey: ShapeKey;
  wBytes: WeightBytes;
  pad: Padding;
  batch: number;
}
interface DevOption {
  key: string;
  label: string;
  dev: Device;
  note: string;
  backend?: string;
}
const BOOK_OPTION: DevOption = {
  key: 'book',
  get label() {
    return tr("The book's example (H100)", 'Ví dụ của sách (H100)');
  },
  dev: BOOK_DEVICE,
  get note() {
    return tr(
      "The book's figures for its example GPU (ch. 5, p. 94): 66.9 TFLOPS and 3.35 TB/s, and the threshold it prints for them, 20.0 FLOP/B. The book's example only.",
      'Số liệu của sách cho GPU ví dụ của nó (ch. 5, p. 94): 66.9 TFLOPS và 3.35 TB/s, cùng ngưỡng sách in cho chúng, 20.0 FLOP/B. Chỉ là ví dụ của sách.',
    );
  },
};

const fmtTime = (s: number) => (!isFinite(s) ? tr('n/a', 'không có') : s < 1e-3 ? `${fmt(s * 1e6)} µs` : s < 1 ? `${fmt(s * 1e3)} ms` : `${fmt(s)} s`); // i18n-ignore: units
const fmtRate = (v: number) => fmt(v, 3);

function useSetup() {
  const { reports } = useReports();
  const rep = reports as unknown as ReportsLike;
  const [setup, setSetup] = useState<Setup>({ devKey: 'book', custom: { peak: 1000, bw: 100 }, shapeKey: 'm8', wBytes: 4, pad: 1, batch: 1 });
  const options: DevOption[] = useMemo(
    () => [
      BOOK_OPTION,
      ...measuredDevices(rep).map((m) => ({
        key: m.backend,
        backend: m.backend,
        get label() {
          return tr(`Yours: ${m.name} (${m.backend})`, `Của bạn: ${m.name} (${m.backend})`);
        },
        dev: { peak: m.peak, bw: m.bw },
        get note() {
          return tr(
            `Measured by lab 5.1 on ${m.date}: ${fmt(m.peak)} GFLOP/s and ${fmt(m.bw)} GB/s from two probe kernels, threshold ${fmt(m.peak / m.bw)} FLOP/B.`,
            `Đo bởi lab 5.1 ngày ${m.date}: ${fmt(m.peak)} GFLOP/s và ${fmt(m.bw)} GB/s từ hai kernel thăm dò, ngưỡng ${fmt(m.peak / m.bw)} FLOP/B.`,
          );
        },
      })),
    ],
    [rep],
  );
  const option: DevOption =
    setup.devKey === 'custom'
      ? { key: 'custom', label: tr('Custom', 'Tùy chỉnh'), dev: setup.custom, note: tr('Numbers you typed.', 'Các số bạn đã nhập.') }
      : (options.find((o) => o.key === setup.devKey) ?? BOOK_OPTION);
  const m8 = useMemo(() => m8Shape(rep, option.backend), [rep, option.backend]);
  const shape: Shape = useMemo(() => (setup.shapeKey === 'm8' ? m8 : setup.shapeKey === 'p4096' ? PROJ_4096 : PROJ_1024), [setup.shapeKey, m8]);
  const prec = useMemo(() => precisionFor(setup.wBytes), [setup.wBytes]);
  return { reports: rep, setup, set: (p: Partial<Setup>) => setSetup((s) => ({ ...s, ...p })), options, option, shape, prec, m8 };
}
type SetupState = ReturnType<typeof useSetup>;

function shapeLabel(shape: Shape, fromRows: boolean) {
  if (shape.kind === 'proj') return tr(`one ${shape.k} → ${shape.n} projection`, `một phép chiếu ${shape.k} → ${shape.n}`);
  return tr(
    `lab 22.2's step: d = ${shape.d}, ${shape.layers} layers, vocabulary ${fmt(shape.vocab, 6)}${fromRows ? ' (from your rows)' : ' (the lab\'s default)'}`,
    `bước của lab 22.2: d = ${shape.d}, ${shape.layers} lớp, từ vựng ${fmt(shape.vocab, 6)}${fromRows ? ' (từ các hàng của bạn)' : ' (mặc định của lab)'}`,
  );
}

// ---------------------------------------------------------------- the roofline explorer's plots
const MARKS = ['diamond', 'square', 'triangle', 'ring'] as const;
function Marker({ kind, x, y, filled = true }: { kind: (typeof MARKS)[number]; x: number; y: number; filled?: boolean }) {
  const f = filled ? 'var(--ink)' : 'var(--page)';
  if (kind === 'diamond') return <rect x={x - 4.5} y={y - 4.5} width={9} height={9} transform={`rotate(45 ${x} ${y})`} fill={f} stroke="var(--ink)" />;
  if (kind === 'square') return <rect x={x - 4} y={y - 4} width={8} height={8} fill={f} stroke="var(--ink)" />;
  if (kind === 'triangle') return <path d={`M${x},${y - 5.5} L${x + 5},${y + 4} L${x - 5},${y + 4} Z`} fill={f} stroke="var(--ink)" />;
  return <circle cx={x} cy={y} r={4.5} fill="none" stroke="var(--ink)" strokeWidth={1.8} />;
}

const POW2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

function BatchPlot({ kind, curve, batch, bstar, series }: { kind: 'time' | 'tokens'; curve: StepResult[]; batch: number; bstar: number | null; series: Series[] }) {
  const W = 440;
  const H = 280;
  const pad = { l: 64, r: 12, t: 14, b: 42 };
  const val = (r: StepResult) => (kind === 'time' ? r.seconds : r.tokensPerSecond);
  const mval = (p: Point) => (kind === 'time' ? p.seconds : p.batch / p.seconds);
  const ys = [...curve.map(val), ...series.flatMap((s) => s.points.map(mval))].filter((v) => v > 0 && isFinite(v));
  const lo = Math.min(...ys) / 1.6;
  const hi = Math.max(...ys) * 1.6;
  const X = (b: number) => pad.l + (Math.log2(Math.min(Math.max(b, 1), 1024)) / 10) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (H - pad.t - pad.b);
  const ticks: number[] = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    for (const m of Math.log10(hi / lo) < 2.2 ? [1, 2, 5] : [1]) {
      const v = m * 10 ** e;
      if (v >= lo && v <= hi) ticks.push(v);
    }
  const label = (v: number) => (kind === 'time' ? fmtTime(v) : fmt(v, 2));
  const at = curve[Math.min(batch, curve.length) - 1];
  const values = POW2.filter((b) => b <= curve.length).map((b) => [b, val(curve[b - 1])]);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={kind === 'time' ? tr('Time per step against batch', 'Thời gian mỗi bước theo batch') : tr('Tokens per second against batch', 'Số token mỗi giây theo batch')}
      style={{ width: '100%', maxWidth: W, display: 'block' }} data-plot={kind}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {bstar != null && bstar <= 1024 ? (
        <rect x={pad.l} y={pad.t} width={X(bstar) - pad.l} height={H - pad.t - pad.b} fill="var(--global-wash)" />
      ) : null}
      {POW2.map((b) => (
        <g key={b}>
          <line x1={X(b)} x2={X(b)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(b)} y={H - pad.b + 14} fontSize="10.5" textAnchor="middle">{b}</text>
        </g>
      ))}
      {ticks.map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 5} y={Y(v) + 3.5} fontSize="10" textAnchor="end">{label(v)}</text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 8} fontSize="11" textAnchor="middle"><Translated vi="batch B (thang log)">batch B (log scale)</Translated></text>
      <text x={13} y={(H - pad.b + pad.t) / 2} fontSize="11" textAnchor="middle" transform={`rotate(-90 13 ${(H - pad.b + pad.t) / 2})`}>
        {kind === 'time' ? tr('time per step', 'thời gian mỗi bước') : tr('tokens per second', 'số token mỗi giây')}
      </text>
      {bstar != null && bstar <= 1024 ? (
        <g data-bstar-line={bstar.toFixed(1)}>
          <line x1={X(bstar)} x2={X(bstar)} y1={pad.t} y2={H - pad.b} stroke="var(--ink)" strokeDasharray="4 3" />
          <text x={X(bstar) + 4} y={pad.t + 13} fontSize="10.5">B* {bstar.toFixed(1)}</text>
          <text x={pad.l + 5} y={H - pad.b - 6} fontSize="10" style={{ fill: 'var(--global)' }}><Translated vi="GEMM bị giới hạn bởi băng thông">bandwidth-bound GEMMs</Translated></text>
        </g>
      ) : null}
      <polyline data-series="model" data-values={JSON.stringify(values)}
        points={curve.map((r) => `${X(r.batch).toFixed(1)},${Y(val(r)).toFixed(1)}`).join(' ')}
        fill="none" stroke="var(--ink)" strokeWidth={2} />
      <circle cx={X(at.batch)} cy={Y(val(at))} r={5} fill="var(--reg)" stroke="var(--page)" strokeWidth={1.5} data-current={at.batch} />
      {series.map((s, i) => (
        <g key={s.key} data-measured={s.key} data-points={s.points.length}>
          {s.points.map((p) => (
            <g key={p.batch}>
              <Marker kind={MARKS[i % MARKS.length]} x={X(p.batch)} y={Y(mval(p))} />
            </g>
          ))}
          {s.knee != null ? (
            <g data-measured-knee={s.knee}>
              <circle cx={X(s.knee)} cy={Y(mval(s.points.find((p) => p.batch === s.knee)!))} r={9} fill="none" stroke="var(--ink)" strokeDasharray="2 2" />
            </g>
          ) : null}
        </g>
      ))}
    </svg>
  );
}

// ---------------------------------------------------------------- explorer 1: the step on the roofline
function DecodeRoofline({ st }: { st: SetupState }) {
  const { setup, set, options, option, shape, prec, reports, m8 } = st;
  const dev = option.dev;
  const T = thresholdOf(dev);
  const locale = useLocale();
  const curve = useMemo(() => stepCurve(shape, prec, setup.pad, dev, batchRange()), [shape, prec, setup.pad, dev, locale]);
  const r = curve[setup.batch - 1];
  const bstar = stepBStar(shape, prec, dev);
  const modelKnee = kneeOf(curve.map((c) => ({ batch: c.batch, seconds: c.seconds })));
  // The reader's own measurements, when this device is one lab 5.1 ran on; they are float32.
  const backend = option.backend;
  const allSeries: Series[] = backend
    ? shape.kind === 'step'
      ? stepSeries(reports, backend)
      : projectionSeries(reports, backend, shape.k, shape.n)
    : [];
  const series = setup.wBytes === 4 ? allSeries : [];
  const fromRows = shape.kind === 'step' && m8 !== M8_DEFAULT;
  const step = (k: ShapeKey, label: string) => (
    <button key={k} aria-pressed={setup.shapeKey === k} onClick={() => set({ shapeKey: k })}>{label}</button>
  );
  return (
    <Bench
      title={tr('A decode step on the roofline', 'Một bước decode trên roofline')}
      task={tr(
        'Find where time per step stops being flat, then change the bytes per weight and the tile padding.',
        'Tìm chỗ thời gian mỗi bước thôi phẳng, rồi thay đổi số byte mỗi trọng số và phần đệm ô.',
      )}
      foot={
        <Translated
          vi={
            <>
              Mỗi phần ở cận roofline theo lưu lượng tối thiểu của chính nó, các phần nối tiếp nhau. Dải tô màu là nơi các GEMM
              bị giới hạn bởi băng thông; đường nét đứt là B* của mô hình. Các điểm tô đặc là số đo của bạn (chỉ float32), các
              vòng chấm là điểm gãy của chúng, nơi thời gian mỗi bước đã tăng thêm một nửa. {option.note}
            </>
          }
        >
          <>
            Each piece at the roofline bound of its own minimum traffic, pieces one after another. The shaded band is
            where the GEMMs are bandwidth-bound; the dashed line is the model's B*. Filled markers are your measurements
            (float32 only), dotted rings their knees, where time per step has grown by half. {option.note}
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi="Thiết bị">Device</Translated>
          <select aria-label={tr('device', 'thiết bị')} value={setup.devKey} onChange={(e) => set({ devKey: e.target.value })}>
            {options.map((o) => (
              <option key={o.key} value={o.key}>{o.label}</option>
            ))}
            <option value="custom">{tr('Custom numbers', 'Số tùy chỉnh')}</option>
          </select>
        </label>
        {setup.devKey === 'custom' ? (
          <>
            <label className="field">
              <Translated vi="Đỉnh (GFLOP/s)">Peak (GFLOP/s)</Translated>
              <input type="number" min="1" aria-label={tr('custom peak', 'đỉnh tùy chỉnh')} value={setup.custom.peak}
                onChange={(e) => set({ custom: { ...setup.custom, peak: Math.max(1, +e.target.value || 1) } })} />
            </label>
            <label className="field">
              <Translated vi="Băng thông (GB/s)">Bandwidth (GB/s)</Translated>
              <input type="number" min="1" aria-label={tr('custom bandwidth', 'băng thông tùy chỉnh')} value={setup.custom.bw}
                onChange={(e) => set({ custom: { ...setup.custom, bw: Math.max(1, +e.target.value || 1) } })} />
            </label>
          </>
        ) : null}
        <label className="field">
          <Translated vi="Mô hình">Model</Translated>
          <span className="seg" role="group" aria-label={tr('model', 'mô hình')}>
            {step('m8', tr('lab 22.2 step', 'bước của lab 22.2'))}
            {step('p4096', '4096 → 4096')}
            {step('p1024', '1024 → 4096')}
          </span>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi="Số byte mỗi trọng số">Bytes per weight</Translated>
          <span className="seg" role="group" aria-label={tr('bytes per weight', 'số byte mỗi trọng số')}>
            {WEIGHT_BYTES.map((b) => (
              <button key={b} aria-pressed={setup.wBytes === b} onClick={() => set({ wBytes: b })}>
                {b === 4 ? '4 (float32)' : b === 2 ? '2 (16-bit)' : b === 1 ? '1 (8-bit)' : '0.5 (4-bit)'}{/* i18n-ignore: numeric formats */}
              </button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi="Phần đệm ô GEMM">GEMM tile padding</Translated>
          <span className="seg" role="group" aria-label={tr('tile padding', 'phần đệm ô')}>
            {([1, 16, 128] as Padding[]).map((p) => (
              <button key={p} aria-pressed={setup.pad === p} onClick={() => set({ pad: p })}>
                {p === 1 ? tr('none', 'không') : p === 16 ? tr('16 rows (M2)', '16 hàng (M2)') : tr('128 rows (M3)', '128 hàng (M3)')}
              </button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi={<>Batch B = {setup.batch}</>}>Batch B = {setup.batch}</Translated>
          <input type="range" min="0" max="10" step="1" aria-label={tr('batch', 'batch')} value={Math.log2(setup.batch)}
            onChange={(e) => set({ batch: 2 ** +e.target.value })} />
        </label>
      </div>
      <div className="cols plot">
        <div className="panel">
          <p style={{ margin: '0 0 0.5rem' }}>
            <b>{shapeLabel(shape, fromRows)}</b>
            <Translated vi={<>, trọng số {prec.weight} B, activation {prec.act} B, batch {setup.batch}</>}>
              , weights {prec.weight} B, activations {prec.act} B, batch {setup.batch}
            </Translated>
          </p>
          <table className="data" style={{ margin: 0 }} data-pieces="">
            <thead>
              <tr>
                <th><Translated vi="Phần">Piece</Translated></th>
                <th className="num"><Translated vi="Thời gian">Time</Translated></th>
                <th className="num"><Translated vi="Tỉ lệ">Share</Translated></th>
                <th><Translated vi="Bị giới hạn bởi">Limited by</Translated></th>
              </tr>
            </thead>
            <tbody>
              {r.pieces.map((p) => (
                <tr key={p.key} data-piece={p.key}>
                  <td><Translated vi={({ proj: 'các phép chiếu của lớp', head: 'LM head', rowwise: 'cộng bias và residual', softmax: 'softmax trên từ vựng', argmax: 'argmax', embed: 'tra cứu embedding' } as Record<string, string>)[p.key] ?? p.label}>{p.label}</Translated></td>
                  <td className="num">{fmtTime(p.seconds)}</td>
                  <td className="num">{fmt((100 * p.seconds) / r.seconds, 2)}%</td>
                  <td>{p.bound === 'memory' ? <span className="chip global"><Translated vi="băng thông">bandwidth</Translated></span> : <span className="chip reg"><Translated vi="phép tính">arithmetic</Translated></span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <Translated
            vi={
              <p style={{ margin: '0.7rem 0 0' }} data-step-seconds={r.seconds}>
                Thời gian mỗi bước <b>{fmtTime(r.seconds)}</b>; {fmtRate(r.tokensPerSecond)} token mỗi giây; mỗi người dùng chờ{' '}
                <b>{fmtTime(r.perToken)}</b> cho mỗi token.
              </p>
            }
          >
            <p style={{ margin: '0.7rem 0 0' }} data-step-seconds={r.seconds}>
              Time per step <b>{fmtTime(r.seconds)}</b>; {fmtRate(r.tokensPerSecond)} tokens per second; each user waits{' '}
              <b>{fmtTime(r.perToken)}</b> per token.
            </p>
          </Translated>
          <Translated
            vi={
              <p style={{ margin: '0.5rem 0 0' }}>
                Các GEMM làm <b data-intensity={r.intensity}>{fmt(r.intensity)}</b> FLOP/B so với ngưỡng{' '}
                <b data-threshold={T}>{fmt(T)}</b>. B* của chúng = <b data-bstar={bstar == null ? '' : bstar.toFixed(1)}>{bstar == null ? 'không bao giờ' : bstar.toFixed(1)}</b>
                {modelKnee != null ? <>; thời gian mỗi bước của mô hình bằng 1.5 lần giá trị ở batch 1 tại B = {modelKnee}</> : null}.
              </p>
            }
          >
            <p style={{ margin: '0.5rem 0 0' }}>
              The GEMMs do <b data-intensity={r.intensity}>{fmt(r.intensity)}</b> FLOP/B against a threshold of{' '}
              <b data-threshold={T}>{fmt(T)}</b>. Their B* = <b data-bstar={bstar == null ? '' : bstar.toFixed(1)}>{bstar == null ? 'never' : bstar.toFixed(1)}</b>
              {modelKnee != null ? <>; the model's time per step is 1.5 times its batch-1 value at B = {modelKnee}</> : null}.
            </p>
          </Translated>
          {series.length ? (
            <p style={{ margin: '0.5rem 0 0' }} data-overlay-note="">
              <Translated vi="Số đo:">Measured:</Translated> {series.map((s, i) => (
                <span key={s.key}>
                  {i ? '; ' : ''}
                  <Translated vi={<>{s.label} ({({ diamond: 'hình thoi', square: 'hình vuông', triangle: 'tam giác', ring: 'vòng tròn' } as Record<string, string>)[MARKS[i % MARKS.length]]}), {s.knee == null ? 'không có điểm gãy đến ' + s.points.at(-1)!.batch : `điểm gãy ở ${s.knee}`}</>}>
                    {s.label} ({MARKS[i % MARKS.length]}), {s.knee == null ? 'no knee up to ' + s.points.at(-1)!.batch : `knee at ${s.knee}`}
                  </Translated>
                </span>
              ))}.
            </p>
          ) : backend && allSeries.length ? (
            <p className="muted" style={{ margin: '0.5rem 0 0' }}><Translated vi="Các số đo của bạn là float32: chọn 4 byte mỗi trọng số để xem chúng.">Your measurements are float32: choose 4 bytes per weight to see them.</Translated></p>
          ) : backend ? (
            <p className="muted" style={{ margin: '0.5rem 0 0' }}>
              <Translated
                vi={
                  <>
                    Chưa có số đo nào cho dạng này trên {backend}.{' '}
                    {shape.kind === 'step' ? <>Lab 22.2 đo thời gian nó (<a href={href('ch22', 'labs')}>lab</a>).</> : <>Lab 5.3 và 15.6 đo thời gian lớp 1024 → 4096; 15.6 đo cả 4096 → 4096.</>}
                  </>
                }
              >
                No measurements of this shape on {backend} yet.{' '}
                {shape.kind === 'step' ? <>Lab 22.2 times it (<a href={href('ch22', 'labs')}>labs</a>).</> : <>Labs 5.3 and 15.6 time the 1024 → 4096 layer; 15.6 also 4096 → 4096.</>}
              </Translated>
            </p>
          ) : null}
        </div>
        <div>
          <BatchPlot kind="time" curve={curve} batch={setup.batch} bstar={bstar} series={series} />
          <BatchPlot kind="tokens" curve={curve} batch={setup.batch} bstar={bstar} series={series} />
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the Predict's curve
function IntensityCurve() {
  const prec = precisionFor(4);
  const W = 420;
  const H = 170;
  const pad = { l: 40, r: 10, t: 10, b: 30 };
  const max = 96;
  const X = (b: number) => pad.l + ((b - 1) / (max - 1)) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - (v / 50) * (H - pad.t - pad.b);
  const bs = Array.from({ length: max }, (_, i) => i + 1);
  const b = bStar(4096, 4096, 20, prec)!;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Intensity of a 4096 by 4096 projection against batch', 'Cường độ tính toán của một phép chiếu 4096 x 4096 theo batch')} style={{ width: '100%', maxWidth: W, display: 'block', marginTop: '0.5rem' }}
      data-intensity-curve={b.toFixed(1)}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {[0, 10, 20, 30, 40, 50].map((v) => (
        <text key={v} x={pad.l - 5} y={Y(v) + 3.5} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink)' }}>{v}</text>
      ))}
      {[1, 16, 32, 48, 64, 80, 96].map((v) => (
        <text key={v} x={X(v)} y={H - pad.b + 13} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink)' }}>{v}</text>
      ))}
      <text x={(W + pad.l) / 2} y={H - 3} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink)' }}><Translated vi="batch B">batch B</Translated></text>
      <line x1={pad.l} x2={W - pad.r} y1={Y(20)} y2={Y(20)} stroke="var(--ink)" strokeDasharray="4 3" />
      <text x={W - pad.r - 4} y={Y(20) - 5} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink)' }}><Translated vi="ngưỡng 20 FLOP/B">threshold 20 FLOP/B</Translated></text>
      <polyline points={bs.map((x) => `${X(x)},${Y(intensity(4096, 4096, x, prec))}`).join(' ')} fill="none" stroke="var(--reg)" strokeWidth={2} />
      <line x1={X(b)} x2={X(b)} y1={Y(20)} y2={H - pad.b} stroke="var(--ink-3)" />
      <text x={X(b) + 4} y={H - pad.b - 6} fontSize="10.5" style={{ fill: 'var(--ink)' }}>B* = {b.toFixed(1)}</text>
    </svg>
  );
}

// ---------------------------------------------------------------- explorer 2: batch slots on a server
const COUNT = 1000;
const WINDOW = 240;
const LOADS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256];

/** The window of steps the chart shows: WINDOW steps from the middle of the run, where the batch has settled. */
function chartWindow(steps: number): [number, number] {
  const from = Math.max(0, Math.floor(steps / 2) - WINDOW / 2);
  return [from, Math.min(steps, from + WINDOW)];
}

function SlotChart({ run }: { run: ServeResult }) {
  const W = 680;
  const H = 190;
  const l = 52;
  const [from, to] = chartWindow(run.steps.length);
  const cw = (W - l - 6) / WINDOW;
  const rh = (H - 22) / run.slots;
  const x = (k: number) => l + (k - from) * cw;
  const gap = rh >= 5 ? 0.6 : 0;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Slots against decode steps', 'Các slot theo các bước decode')} style={{ width: '100%', minWidth: 520, display: 'block' }}
      data-slot-chart={run.policy} data-slots={run.slots} data-from={from} data-to={to}>
      <rect x={l} y={0} width={WINDOW * cw} height={H - 22} fill="var(--page)" stroke="var(--rule)" />
      {run.served
        .filter((r) => r.admitStep < to && r.returnStep > from)
        .map((r) => {
          const start = Math.max(r.admitStep, from);
          const genEnd = Math.min(r.lastStep + 1, to);
          const heldStart = Math.max(r.lastStep + 1, from);
          const heldEnd = Math.min(r.returnStep, to);
          return (
            <g key={r.id}>
              {genEnd > start ? (
                <rect x={x(start)} y={r.slot * rh + gap} width={Math.max(0.5, (genEnd - start) * cw - gap)} height={Math.max(0.5, rh - 2 * gap)}
                  fill="var(--ink-2)" opacity={r.length <= SHORT_MAX ? 0.45 : 1} data-short={r.length <= SHORT_MAX ? '1' : '0'} />
              ) : null}
              {heldEnd > heldStart ? (
                <rect x={x(heldStart)} y={r.slot * rh + rh / 2 - Math.min(1.5, rh / 4)} width={(heldEnd - heldStart) * cw} height={Math.max(0.5, Math.min(3, rh / 2))}
                  fill="var(--ink-3)" data-held="" />
              ) : null}
            </g>
          );
        })}
      {[0, 60, 120, 180, 240].filter((k) => from + k <= to).map((k) => (
        <text key={k} x={x(from + k)} y={H - 8} fontSize="10" textAnchor={k === 0 ? 'start' : k === 240 ? 'end' : 'middle'}>{from + k}</text>
      ))}
      <text x={l - 6} y={10} fontSize="10" textAnchor="end"><Translated vi="slot 1">slot 1</Translated></text>
      <text x={l - 6} y={H - 26} fontSize="10" textAnchor="end">{run.slots}</text>
    </svg>
  );
}

function ServerSlots({ st }: { st: SetupState }) {
  const { option, shape, prec, setup } = st;
  const [policy, setPolicy] = useState<Policy>('continuous');
  const [slots, setSlots] = useState(16);
  const [loadIdx, setLoadIdx] = useState(LOADS.indexOf(64));
  const [mix, setMix] = useState<Mix>('long');
  const [seed, setSeed] = useState(1);
  const dev = option.dev;
  const stepTime = useMemo(() => {
    const table: number[] = [];
    return (a: number) => (table[a] ??= stepAt(shape, a, prec, setup.pad, dev).seconds);
  }, [shape, prec, setup.pad, dev]);
  const t1 = stepTime(1);
  const cap1 = 1 / (t1 * MIXES[mix].mean);
  const load = LOADS[loadIdx];
  const rate = load * cap1;
  const runs = useMemo(() => {
    const requests = makeRequests({ count: COUNT, rate, mix, seed });
    return {
      static: serve({ requests, slots, policy: 'static', stepTime }),
      continuous: serve({ requests, slots, policy: 'continuous', stepTime }),
    };
  }, [rate, mix, seed, slots, stepTime]);
  const shown = runs[policy];
  const ratio = runs.continuous.tokensPerSecond / runs.static.tokensPerSecond;
  const row = (p: Policy) => {
    const r = runs[p];
    const cell = (k: string, v: number | null, text: string) => (
      <td className="num" data-k={k} data-v={v ?? ''}>{text}</td>
    );
    return (
      <tr key={p} data-policy={p} style={p === policy ? { background: 'var(--rule-soft)' } : undefined}>
        <td><Translated vi={p === 'static' ? 'tĩnh' : 'liên tục'}>{p}</Translated></td>
        {cell('tps', r.tokensPerSecond, fmtRate(r.tokensPerSecond))}
        {cell('tpt', r.meanTimePerToken, fmtTime(r.meanTimePerToken))}
        {cell('worst', r.worstTimePerToken, fmtTime(r.worstTimePerToken))}
        {cell('short', r.meanFinishShort, r.meanFinishShort == null ? tr('none', 'không có') : fmtTime(r.meanFinishShort))}
        {cell('util', r.utilisation, `${fmt(100 * r.utilisation, 2)}%`)}
        {cell('batch', r.meanBatch, fmt(r.meanBatch, 3))}
      </tr>
    );
  };
  return (
    <Bench
      title={tr('Batch slots on a server', 'Các slot batch trên một server')}
      task={tr(
        'At heavy load, compare the two policies at the same slot limit; then raise the limit past the knee.',
        'Ở tải nặng, so sánh hai chính sách với cùng giới hạn slot; rồi nâng giới hạn vượt qua điểm gãy.',
      )}
      foot={
        <Translated
          vi={
            <>
              {COUNT} request từ một bộ sinh có seed; thời gian bước lấy từ trình khám phá ở trên ({option.label}, {shapeLabel(shape, false).replace(' (mặc định của lab)', '')},
              trọng số {prec.weight} B, phần đệm {setup.pad === 1 ? 'không' : setup.pad + ' hàng'}). Biểu đồ hiển thị {WINDOW} bước của
              chính sách bạn chọn, từ giữa lần chạy, khi batch đã ổn định: thanh đậm là các request dài hơn {SHORT_MAX} token, thanh nhạt hơn là các request ngắn hơn, và một đường mảnh là một
              request đã xong nhưng vẫn giữ slot của nó.
            </>
          }
        >
          <>
            {COUNT} requests from a seeded generator; step times from the explorer above ({option.label}, {shapeLabel(shape, false).replace(' (the lab\'s default)', '')},
            weights {prec.weight} B, padding {setup.pad === 1 ? 'none' : setup.pad + ' rows'}). The chart shows {WINDOW} steps of the
            policy you picked, from the middle of the run, where the batch has settled: solid bars are requests of more than {SHORT_MAX} tokens, fainter bars shorter ones, and a thin line is a
            finished request still holding its slot.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi={<>Lượt đến: {fmt(rate, 3)} request mỗi giây ({load}× lượng một slot ở batch 1 phục vụ được)</>}>
            Arrivals: {fmt(rate, 3)} requests per second ({load}× what one slot at batch 1 serves)
          </Translated>
          <input type="range" min="0" max={LOADS.length - 1} step="1" aria-label={tr('arrival rate', 'tốc độ đến')} value={loadIdx}
            onChange={(e) => setLoadIdx(+e.target.value)} />
        </label>
        <label className="field">
          <Translated vi="Độ dài đầu ra">Output lengths</Translated>
          <span className="seg" role="group" aria-label={tr('output lengths', 'độ dài đầu ra')}>
            {(['short', 'long'] as Mix[]).map((m) => (
              <button key={m} aria-pressed={mix === m} onClick={() => setMix(m)} title={tr(MIXES[m].label, m === 'short' ? 'độ dài tương tự (8 đến 40 token)' : 'đuôi dài (4 trên 5 ngắn, 1 trên 5 dài 128 đến 512 token)')}>
                {m === 'short' ? tr('similar', 'tương tự') : tr('long-tailed', 'đuôi dài')}
              </button>
            ))}
          </span>
        </label>
        <button className="quiet" onClick={() => setSeed(seed + 1)}><Translated vi="Lượt đến mới">New arrivals</Translated></button>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi={<>Giới hạn slot = {slots}</>}>Slot limit = {slots}</Translated>
          <input type="range" min="0" max="8" step="1" aria-label={tr('slot limit', 'giới hạn slot')} value={Math.log2(slots)}
            onChange={(e) => setSlots(2 ** +e.target.value)} />
        </label>
        <label className="field">
          <Translated vi="Biểu đồ hiển thị">Chart shows</Translated>
          <span className="seg" role="group" aria-label={tr('policy', 'chính sách')}>
            {(['static', 'continuous'] as Policy[]).map((p) => (
              <button key={p} aria-pressed={policy === p} onClick={() => setPolicy(p)}><Translated vi={p === 'static' ? 'tĩnh' : 'liên tục'}>{p}</Translated></button>
            ))}
          </span>
        </label>
      </div>
      <SlotChart run={shown} />
      <p className="muted" style={{ margin: '0.2rem 0 0.8rem', fontSize: '0.78rem' }}>
        <Translated vi={<>các bước decode theo chiều ngang ({chartWindow(shown.steps.length).join(' đến ')}, từ giữa một lần chạy {shown.steps.length} bước), các slot theo chiều dọc</>}>
          decode steps across ({chartWindow(shown.steps.length).join(' to ')}, from the middle of a run of {shown.steps.length}), slots down
        </Translated>
      </p>
      <table className="data" style={{ margin: 0 }} data-serving="">
        <thead>
          <tr>
            <th><Translated vi="Chính sách">Policy</Translated></th>
            <th className="num"><Translated vi="Token mỗi giây">Tokens per second</Translated></th>
            <th className="num"><Translated vi="Thời gian mỗi token, trung bình">Time per token, mean</Translated></th>
            <th className="num"><Translated vi="tệ nhất">worst</Translated></th>
            <th className="num"><Translated vi="Request ngắn xong sau">Short requests done after</Translated></th>
            <th className="num"><Translated vi="Slot bận">Slots busy</Translated></th>
            <th className="num"><Translated vi="Batch trung bình">Mean batch</Translated></th>
          </tr>
        </thead>
        <tbody>
          {row('static')}
          {row('continuous')}
        </tbody>
      </table>
      <p style={{ margin: '0.7rem 0 0' }} data-verdict={ratio}>
        <Translated
          vi={
            <>
              Với {slots} slot,{' '}
              {Math.abs(ratio - 1) < 0.02 ? 'hai chính sách cho cùng số token mỗi giây.' : `continuous batching cho số token mỗi giây gấp ${fmt(ratio, 3)} lần static.`}
              {runs.continuous.meanBatch < 0.5 * slots ? ' Các slot phần lớn trống: ở tải này, giới hạn không phải là thứ quyết định batch.' : ''}
            </>
          }
        >
          At {slots} {slots === 1 ? 'slot' : 'slots'},{' '}
          {Math.abs(ratio - 1) < 0.02 ? 'the two policies make the same tokens per second.' : `continuous batching makes ${fmt(ratio, 3)} times static's tokens per second.`}
          {runs.continuous.meanBatch < 0.5 * slots ? ' The slots are mostly empty: at this load the limit is not what sets the batch.' : ''}
        </Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the section
export function S225() {
  const st = useSetup();
  const f32 = precisionFor(4);
  const b8 = bookThroughput(8, 3, 1);
  const bBook = bStar(4096, 4096, 20, f32)!;
  const bWhole = bStarSearch(4096, 4096, 20, f32)!;
  const wb = weightBytes(M8_DEFAULT, f32);
  // The per-row pieces' share of lab 22.2's default step at batch 256 (book's device, float32).
  const r256 = stepAt(M8_DEFAULT, 256, f32, 1, BOOK_DEVICE);
  const rowShare = (100 * r256.rowSeconds) / r256.seconds;
  return (
    <>
      <Translated
        vi={
          <p>
            Mọi kỹ thuật trong sách cho đến giờ đều có một mục tiêu: lấy cùng một đầu vào và xử lý xong nó sớm hơn. Sách gọi
            đây là tối ưu độ trễ. Các ví dụ của nó đến từ chương 11 (double buffering, một kỹ thuật ở mức warp, và look-back của
            scan một lượt) và từ chương 20, nơi flash attention rút ngắn thời gian của một transformer trên một đầu vào cho trước
            bằng cách loại bỏ các lần đọc và ghi thừa vào bộ nhớ toàn cục (global memory).
          </p>
        }
      >
        <p>
          Every technique in this book so far has had one aim: take the same input and finish it sooner. The book calls
          this latency optimisation. Its examples come from chapter 11 (double buffering, a warp-level technique, and the
          look-back of the single-pass scan) and from chapter 20, where flash attention shortens a transformer's time on a
          given input by removing redundant reads and writes of global memory.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Danh sách của sách ghi "double buffering, warp-level shuffling register tiling, and look-back". Mục ở giữa trông như
              hai kỹ thuật của chương 11 bị viết dính vào nhau, warp-level shuffle (§11.4) và phân ô thanh ghi (§11.7). Cách đọc
              này là của khóa học này.
            </p>
          }
        >
          <p>
            The book's list reads "double buffering, warp-level shuffling register tiling, and look-back". The middle item
            looks like two of chapter 11's techniques run together, warp-level shuffles (§11.4) and register tiling
            (§11.7). That reading is ours.
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Xử lý xong một đầu vào sớm hơn không phải lý do duy nhất để tối ưu. Sách nêu hai lý do. Lý do thứ nhất là lý do cuốn
            sách này phục vụ từ đầu: giải một bài toán cho trước trong ít thời gian hơn. Ví dụ của nó là một công ty đầu tư có
            bản phân tích rủi ro qua đêm cho mọi danh mục đầu tư tốn 10 giờ, trong khi các nhà quản lý danh mục muốn nó xong
            trong 4 giờ. Lý do thứ hai ít hiển nhiên hơn. Cùng công ty đó muốn chạy bản phân tích nhiều lần, với các tham số
            khác nhau, và chúng không vừa hết trong khung thời gian. Tối ưu cho nhiều thể hiện mang đến một sự đánh đổi mà mục
            tiêu thứ nhất chưa từng có: độ trễ của mỗi thể hiện so với độ trễ của tất cả chúng gộp lại.
          </p>
        }
      >
        <p>
          Finishing one input sooner is not the only reason to optimise. The book names two. The first is the one this book
          has served all along: solve a given problem in less time. Its example is an investment firm whose overnight risk
          analysis of all its portfolios takes 10 hours, when the portfolio managers want it done in 4. The second is less
          obvious. The same firm wants many runs of the analysis, with different parameters, and they do not all fit in the
          window. Optimising for many instances brings a trade-off the first goal never had: the latency of each instance
          against the latency of all of them together.
        </p>
      </Translated>
      <Source section="22.5" pages="539" />
      <Translated
        vi={
          <p>
            Nếu bạn biết thuật ngữ về khả năng mở rộng song song, thì đây là hai loại của nó. Xử lý xong một bài toán cố định sớm
            hơn trên nhiều phần cứng hơn là strong scaling, bối cảnh của định luật Amdahl (chương 1 và mục 22.3). Nhét thêm bài
            toán vào cùng khoảng thời gian, với bài toán lớn lên cùng cỗ máy, là weak scaling, nơi định luật Gustafson áp dụng.
          </p>
        }
      >
        <p>
          If you know the vocabulary of parallel scaling, these are its two kinds. Finishing a fixed problem sooner on more
          hardware is strong scaling, the setting of Amdahl's law (chapter 1 and section 22.3). Fitting more problems into
          the same time, with the problem growing as the machine does, is weak scaling, where Gustafson's law applies.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Best Practices Guide của NVIDIA dùng cả hai tên này (§4.1.3). Việc gắn hai động cơ của sách với chúng là của khóa
              học này (ch22-facts A9).
            </p>
          }
        >
          <p>
            NVIDIA's Best Practices Guide uses both names (§4.1.3). Mapping the book's two motivations onto them is ours
            (ch22-facts A9).
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS225['ch22/22.5/q-scaling']} />

      <h2><Translated vi={"Batch các truy vấn biến nhiều phép nhân thành một"}>Batching queries turns many products into one</Translated></h2>
      <Translated
        vi={
          <p>
            Suy luận LLM là trường hợp sách chọn cho mục tiêu thứ hai. Trả lời truy vấn của một người dùng là một thể hiện của
            bài toán; một server có nhiều người dùng. Nếu server xếp chồng hàng hiện tại của truy vấn của nhiều người dùng thành
            một ma trận, mỗi phép chiếu vốn là một phép nhân vector với ma trận cho mỗi người dùng trở thành một phép nhân ma
            trận cho tất cả. Trọng số giống nhau cho mọi người dùng, nên một trọng số được nạp một lần giờ phục vụ mọi hàng. Phép
            nhân ma trận đó tốn lâu hơn một phép nhân vector với ma trận, nhưng ít hơn nhiều so với tất cả các phép đó nối tiếp
            nhau.
          </p>
        }
      >
        <p>
          LLM inference is the book's case of the second goal. Answering one user's query is one instance of the problem;
          a server has many users. If the server stacks the current row of several users' queries into one matrix, each
          projection that was a vector-matrix product per user becomes one matrix product for all of them. The weights are
          the same for every user, so a weight fetched once now serves every row. That matrix product takes longer than one
          vector-matrix product, but much less than all of them one after another.
        </p>
      </Translated>
      <Source section="22.5" pages="539" />
      <Translated vi={<p>Sách phát biểu lợi ích cho phép chiếu QKV dưới dạng hai thông lượng:</p>}>
        <p>The book states the gain for the QKV projection as two throughputs:</p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Lời văn của sách nói "vector-matrix" còn công thức của nó nói "matrix-vector". Cùng một phép toán: một hàng
              activation nhân với một ma trận trọng số.
            </p>
          }
        >
          <p>
            The book's prose says "vector-matrix" and its formula says "matrix-vector". Same operation: one row of
            activations against a weight matrix.
          </p>
        </Translated>
      </Aside>
      <table className="data">
        <tbody>
          <tr>
            <td><Translated vi="có batching">with batching</Translated></td>
            <td><Translated vi="thông lượng = số truy vấn ÷ độ trễ của phép nhân ma trận">throughput = number of queries ÷ latency of the matrix multiplication</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="không có">without</Translated></td>
            <td><Translated vi="thông lượng = số truy vấn ÷ tổng độ trễ của mọi phép nhân vector với ma trận">throughput = number of queries ÷ total latency of all the vector-matrix multiplications</Translated></td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Cái đầu cao hơn. Cái giá, như sách nói, là mỗi truy vấn thấy độ trễ phép chiếu QKV dài hơn, trong khi cả nhóm xong
            sớm hơn.
          </p>
        }
      >
        <p>
          The first is higher. The price, as the book puts it, is that each query sees a longer QKV projection latency,
          while the group as a whole finishes sooner.
        </p>
      </Translated>
      <Predict
        id="ch22/22.5/p-batch8"
        prompt={tr(
          'A batch of 8 queries: the matrix multiplication takes 3 times as long as one vector-matrix multiplication. By what factor does throughput change, against eight separate vector-matrix multiplications?',
          'Một batch 8 truy vấn: phép nhân ma trận tốn gấp 3 lần một phép nhân vector với ma trận. Thông lượng thay đổi bao nhiêu lần, so với tám phép nhân vector với ma trận riêng rẽ?',
        )}
        unit="×"
      >
        {(v) => (
          <Translated
            vi={
              <p>
                Có batch: 8 truy vấn trong 3 đơn vị. Riêng rẽ: 8 truy vấn trong 8 đơn vị. Thông lượng tăng 8 / 3 ={' '}
                <b data-gain={b8.gain}>{fmt(b8.gain)}×</b>, trong khi phép chiếu của riêng mỗi truy vấn đi từ 1 đơn vị lên{' '}
                {b8.latencyFactor}.{' '}
                {Math.abs(v - b8.gain) < 0.05
                  ? 'Đúng như bạn đã trả lời.'
                  : Math.abs(v - 3) < 0.05 || Math.abs(v - 1 / 3) < 0.05
                    ? 'Ba là hệ số độ trễ; thông lượng là số truy vấn mỗi đơn vị thời gian, và cả hai bên đều phục vụ 8 truy vấn.'
                    : 'Dùng hai công thức ở trên: tử số bằng nhau, nên tỉ số là nghịch đảo tỉ số thời gian.'}
              </p>
            }
          >
            <p>
              Batched: 8 queries in 3 units. Separate: 8 queries in 8 units. Throughput rises by 8 / 3 ={' '}
              <b data-gain={b8.gain}>{fmt(b8.gain)}×</b>, while each query's own projection went from 1 unit to{' '}
              {b8.latencyFactor}.{' '}
              {Math.abs(v - b8.gain) < 0.05
                ? 'That is what you had.'
                : Math.abs(v - 3) < 0.05 || Math.abs(v - 1 / 3) < 0.05
                  ? 'Three is the latency factor; throughput is queries per unit time, and both sides served 8 queries.'
                  : 'Use the two formulas above: the numerators are equal, so the ratio is the inverse ratio of the times.'}
            </p>
          </Translated>
        )}
      </Predict>

      <h2><Translated vi={"Xem một bước"}>Watch one step</Translated></h2>
      <Translated
        vi={
          <p>
            Vì sao phép nhân ma trận lại rẻ như vậy? Hãy xem một bước decode của bộ decoder trong lab 22.2 khi batch lớn lên.
            Thời gian được vẽ theo số lượt đọc qua trọng số ({fmt(wb / 1e6, 3)} MB trọng số ở float32 cho mô hình mặc định của
            lab), nên hình này đúng với mọi thiết bị có ngưỡng bằng 20 FLOP/B như ví dụ của sách.
          </p>
        }
      >
        <p>
          Why is the matrix product so cheap? Watch one decode step of lab 22.2's decoder as the batch grows. Time is drawn
          in passes over the weights ({fmt(wb / 1e6, 3)} MB of them in float32 for the lab's default model), so the picture
          holds for any device whose threshold is the book's example 20 FLOP/B.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Cùng mô hình với trình khám phá roofline bên dưới: mỗi phần ở cận theo lưu lượng tối thiểu của chính nó, các phần nối
              tiếp nhau, không có chi phí khởi chạy, không có bộ nhớ đệm, không có attention. Chỉ thời gian tương đối được vẽ, nên
              không có số đo thời gian của thiết bị nào.
            </p>
          }
        >
          <p>
            The same model as the roofline explorer below: each piece at the bound of its own minimum traffic, pieces in
            sequence, no launch cost, no cache, no attention. Only relative time is drawn, so no device timing appears.
          </p>
        </Translated>
      </Simplification>
      <DecodeStepScene />

      <h2><Translated vi={"Tìm điểm gãy trên roofline"}>Find the knee on the roofline</Translated></h2>
      <Translated
        vi={
          <p>
            Chương 5 cho ta công cụ. Một kernel bị kìm bởi băng thông chừng nào nó thực hiện ít phép toán trên mỗi byte hơn ngưỡng
            T của thiết bị, tức thông lượng đỉnh chia băng thông đỉnh (<a href={href('ch05', '5.1')}>§5.1</a>). Đếm một phép chiếu
            K → N trên một batch B hàng ở float32, mỗi trọng số, đầu vào và đầu ra được di chuyển một lần: 2BKN phép toán cho
            4(KN + BK + BN) byte, nên <code>I(B) = BKN / (2(KN + BK + BN))</code>. Khi B nhỏ hơn nhiều so với K và N, giá trị đó
            khoảng B/2.
          </p>
        }
      >
        <p>
          Chapter 5 gives the tool. A kernel is held back by bandwidth while it performs fewer operations per byte than the
          device's threshold T, peak throughput over peak bandwidth (<a href={href('ch05', '5.1')}>§5.1</a>). Count a K → N
          projection over a batch of B rows in float32, each weight, input and output moved once: 2BKN operations for
          4(KN + BK + BN) bytes, so <code>I(B) = BKN / (2(KN + BK + BN))</code>. While B is much smaller than K and N, that
          is about B/2.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cách đếm này là của khóa học này (ch22-facts E1), theo phương pháp của chương 5. Ở 16-bit nó cho ra con số 1 FLOP/B
              ở batch 1 và 315 ở batch 512 của chương 20 cho lớp 1024 → 4096 của nó.
            </p>
          }
        >
          <p>
            The count is ours (ch22-facts E1), by chapter 5's method. In 16-bit it gives chapter 20's 1 FLOP/B at batch 1
            and 315 at batch 512 for its 1024 → 4096 layer.
          </p>
        </Translated>
      </Beyond>
      <Predict
        id="ch22/22.5/p-book-knee"
        prompt={tr(
          "On the book's example device (threshold 20 FLOP/B, ch. 5 p. 94), what is the smallest batch at which a 4096 → 4096 float32 projection reaches the threshold?",
          'Trên thiết bị ví dụ của sách (ngưỡng 20 FLOP/B, ch. 5 p. 94), batch nhỏ nhất mà tại đó một phép chiếu 4096 → 4096 float32 đạt ngưỡng là bao nhiêu?',
        )}
        unit={tr('rows', 'hàng')}
      >
        {(v) => (
          <>
            <Translated
              vi={
                <p>
                  <b data-knee-answer={bWhole}>{bWhole}</b>. Giải I(B) = 20 cho B* = {bBook.toFixed(1)}: cao hơn 2T một chút,
                  vì activation thêm byte khi B lớn lên.{' '}
                  {v === bWhole ? 'Chính xác.' : Math.abs(v - 40) <= 1 ? 'Gần đúng: B/2 = 20 cho 40, và các hàng đầu vào, đầu ra đẩy nó lên 40.8.' : 'Dùng I(B) ≈ B/2 để ước lượng đầu tiên, rồi hiệu chỉnh cho activation.'}
                </p>
              }
            >
              <p>
                <b data-knee-answer={bWhole}>{bWhole}</b>. Solving I(B) = 20 gives B* = {bBook.toFixed(1)}: a little above
                2T, because the activations add bytes as B grows.{' '}
                {v === bWhole ? 'Exactly.' : Math.abs(v - 40) <= 1 ? 'Close: B/2 = 20 gives 40, and the input and output rows push it to 40.8.' : 'Use I(B) ≈ B/2 for a first estimate, then correct for the activations.'}
              </p>
            </Translated>
            <IntensityCurve />
          </>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Giờ đến cả bước. Trình khám phá đưa mọi phần của bước trong lab 22.2 qua cách đếm đó: embedding, phép chiếu của mỗi
            lớp, bias và residual, LM head, softmax và argmax. Chạy nó trên thiết bị ví dụ của sách, trên thiết bị của bạn nếu
            bạn đã chạy lab 5.1, hoặc trên các số bạn nhập. Chọn "4096 → 4096" để xem riêng phép chiếu.
          </p>
        }
      >
        <p>
          Now the whole step. The explorer runs every piece of lab 22.2's step through that count: the embedding, each
          layer's projection, bias and residual, the LM head, softmax and argmax. Run it on the book's example device, on
          yours if you have run lab 5.1, or on numbers you type. Choose "4096 → 4096" for the projection alone.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch22-facts A4, infra-facts C13)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              Các hướng dẫn hiệu năng của NVIDIA (2023) đưa ra lập luận này với chính công thức này. Apple không công bố số liệu
              đỉnh nào: trên máy Mac, số đo của lab 5.1 là ngưỡng duy nhất.
            </p>
          }
        >
          <p>
            NVIDIA's performance guides (2023) make this argument with this formula. Apple publishes no peak figures: on a
            Mac, lab 5.1's measurement is the only threshold.
          </p>
        </Translated>
      </Today>
      <DecodeRoofline st={st} />
      <Translated
        vi={
          <p>
            Điều cần tìm. Thời gian mỗi bước gần như phẳng cho đến B*, khoảng 2T hàng ở float32 và khoảng T ở 16-bit, trong khi
            số token mỗi giây tăng tỉ lệ với B. Qua điểm đó, số token mỗi giây chững lại và thời gian mỗi token của mọi người dùng
            tăng lên. Các phần theo từng hàng (bias, residual, softmax trên từ vựng, argmax) tăng ngay từ đầu và không ai dùng
            chung; trong bước của lab 22.2, với từ vựng {fmt(M8_DEFAULT.vocab, 6)}, chúng vẫn chỉ chiếm khoảng{' '}
            {fmt(rowShare, 2)}% của bước ở batch 256 trên thiết bị ví dụ của sách. Với phần đệm của M3, đường cong phẳng đến 128
            vì một lý do không liên quan đến băng thông. Và nếu một điểm đo nhanh hơn đường của mô hình, thiết bị đã di chuyển ít
            byte hơn mức mô hình tính, đó là bộ nhớ đệm đang làm việc như ở §5.1 (hoặc một phép thăm dò băng thông bị đo thấp).
          </p>
        }
      >
        <p>
          What to look for. Time per step stays nearly flat until B*, about 2T rows in float32 and about T in 16-bit, while
          tokens per second rise in proportion to B. Past it, tokens per second flatten and every user's time per token
          grows. The per-row pieces (bias, residual, softmax over the vocabulary, argmax) grow from the start and
          are shared by nobody; in lab 22.2's step, with its vocabulary of {fmt(M8_DEFAULT.vocab, 6)}, they are still only
          about {fmt(rowShare, 2)}% of the step at batch 256 on the book's example device. With M3's padding the curve is flat to 128 for a reason
          unrelated to bandwidth. And if a measured point is faster than the model's line, the device moved fewer bytes
          than the model charges, which is caches at work as in §5.1 (or a bandwidth probe that undershot).
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi phần chạy ở cận roofline theo lưu lượng tối thiểu của chính nó; các phần không chồng lấp; không có chi phí khởi
              chạy, không có bộ nhớ đệm; attention bị bỏ qua (nó không được batch). Chính NVIDIA gọi phép so sánh này là "a
              simplified rule of thumb". Với trọng số 8 và 4 bit, mô hình giữ activation ở 16 bit.
            </p>
          }
        >
          <p>
            Each piece runs at the roofline bound of its own minimum traffic; pieces do not overlap; no launch cost, no
            cache; attention is left out (it does not batch). NVIDIA itself calls this comparison "a simplified rule of
            thumb". With 8- and 4-bit weights the model keeps activations at 16 bits.
          </p>
        </Translated>
      </Simplification>
      <Ask q={questionsS225['ch22/22.5/q-plateau']} />
      <Ask q={questionsS225['ch22/22.5/q-half-bytes']} />

      <h2><Translated vi={"Những gì batching không batch được"}>What batching does not batch</Translated></h2>
      <Translated
        vi={
          <p>
            Trang 539 nói batching làm tăng cường độ tính toán "of the attention heads". Chương 20 nói các lớp attention "do not
            benefit from batching" (p. 505). Cả hai đều đúng, tùy vào việc "attention" có bao gồm phép chiếu QKV và phép chiếu
            đầu ra hay không. Những phép đó có trọng số, và chúng được batch. Bản thân attention nhân query của mỗi request với
            key và value riêng của request đó: không có gì được chia sẻ, nên lượng đọc của nó tăng theo batch, giống các phần
            theo từng phần tử. Phiên bản của chương 20 là cái nên giữ.
          </p>
        }
      >
        <p>
          Page 539 says batching raises the arithmetic intensity "of the attention heads". Chapter 20 says the attention
          layers "do not benefit from batching" (p. 505). Both hold, depending on whether "attention" includes the QKV and
          output projections. Those carry weights, and they batch. Attention proper multiplies each request's query by that
          request's own keys and values: nothing is shared, so its reads grow with the batch, like the elementwise pieces.
          Chapter 20's version is the one to keep.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Orca (OSDI 2022) batch mọi thứ trừ attention, vốn không có "model parameters" nào để tái sử dụng, và một chú thích
              nêu cả hai định nghĩa của attention (ch22-facts A6).
            </p>
          }
        >
          <p>
            Orca (OSDI 2022) batches everything but attention, which has no "model parameters" to reuse, and a footnote
            names both definitions of attention (ch22-facts A6).
          </p>
        </Translated>
      </Aside>
      <Ask q={questionsS225['ch22/22.5/q-attention']} />

      <h2><Translated vi={"Ai trả giá"}>Who pays</Translated></h2>
      <Translated
        vi={
          <p>
            Sách nói mỗi truy vấn chịu độ trễ phép chiếu dài hơn. Dài hơn so với cái gì? So với chính truy vấn đó chạy một mình,
            ở batch 1. Đó là lập luận của sách; không tài liệu nào của nhà cung cấp mà chúng tôi đã kiểm tra nói điều đó, và lab
            22.2 đo nó. Nhưng một server có tám người dùng không được phép phục vụ riêng từng người. Lựa chọn thực tế của nó là
            phục vụ họ lần lượt, và khi đó mọi người dùng chờ cả một vòng giữa các token.
          </p>
        }
      >
        <p>
          The book says each query experiences a longer projection latency. Longer than what? Than the same query alone, at
          batch 1. That is the book's reasoning; no vendor document we checked states it, and lab 22.2 measures it. But a
          server with eight users does not get to serve each of them alone. Its real alternative is to serve them one after
          another, and then every user waits a whole round between tokens.
        </p>
      </Translated>
      <Source section="22.5" pages="539" />
      <Ask q={questionsS225['ch22/22.5/q-who-pays']} />

      <h2><Translated vi={"Cùng sự đánh đổi như CPU so với GPU"}>The same trade-off as CPU against GPU</Translated></h2>
      <Translated
        vi={
          <p>
            Sách khép lại lập luận bằng một liên hệ với chương 1. Batching đổi độ trễ của mỗi thể hiện lấy thông lượng của cả
            nhóm, đó chính là sự đánh đổi phân biệt thiết kế của CPU với của GPU: CPU có độ trễ ngắn hơn cho mỗi phần tử dữ liệu,
            GPU có thông lượng cao hơn trên nhiều phần tử.
          </p>
        }
      >
        <p>
          The book closes the argument with a link to chapter 1. Batching trades the latency of each instance for the
          throughput of the group, which is the trade-off that separates a CPU's design from a GPU's: the CPU has the
          shorter latency for each data element, the GPU the higher throughput over many.
        </p>
      </Translated>
      <Source section="22.5" pages="539–540" />
      <Ask q={questionsS225['ch22/22.5/q-cpu-gpu']} />

      <h2><Translated vi={"Bạn gặp nó ở đâu: continuous batching"}>Where you meet it: continuous batching</Translated></h2>
      <Translated
        vi={
          <p>
            Một server chỉ sống ở B lớn nếu batch của nó luôn đầy, và các request không hợp tác: chúng đến ngẫu nhiên và muốn số
            token khác nhau. Chính sách đơn giản, static batching, bắt đầu một batch và chạy nó cho đến khi request dài nhất xong.
            Chính sách mà các server thực tế dùng nhận và trả request ở mọi bước.
          </p>
        }
      >
        <p>
          A server only lives at a large B if its batch stays full, and requests do not cooperate: they arrive at random and
          want different numbers of tokens. The simple policy, static batching, starts a batch and runs it until its longest
          request is done. The policy real servers use admits and retires requests at every step.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không bàn về chính sách phục vụ trong chương này; chương 20 có nhắc đến in-flight batching (p. 506). Phần này,
              hoạt ảnh và trình khám phá là của khóa học này.
            </p>
          }
        >
          <p>
            The book does not discuss serving policies in this chapter; chapter 20 mentions in-flight batching (p. 506).
            This part, the animation and the explorer are ours.
          </p>
        </Translated>
      </Beyond>
      <ContinuousBatchingScene />
      <Translated
        vi={
          <p>
            Trình khám phá chạy một nghìn request qua một server có thời gian bước lấy từ trình khám phá ở trên, nên thiết bị, mô
            hình, số byte mỗi trọng số và phần đệm bạn chọn ở đó cũng áp dụng ở đây.
          </p>
        }
      >
        <p>
          The explorer runs a thousand requests through a server whose step time comes from the explorer above, so the
          device, model, bytes per weight and padding you chose there apply here.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Không có prefill: một request bắt đầu decode khi được nhận. Thời gian bước chỉ phụ thuộc vào số request trong bước.
              Không có giới hạn bộ nhớ KV cache, thứ vốn giới hạn các batch thực tế (ch. 20, p. 505). Một request đã xong trong một
              batch tĩnh không tốn gì. Các lượt đến được sinh từ một seed.
            </p>
          }
        >
          <p>
            No prefill: a request starts decoding when admitted. Step time depends only on the number of requests in the
            step. No KV-cache memory limit, which is what caps real batches (ch. 20, p. 505). A finished request in a static
            batch costs nothing. Arrivals come from a seed.
          </p>
        </Translated>
      </Simplification>
      <ServerSlots st={st} />
      <Translated
        vi={
          <p>
            Ở tải nặng và cùng giới hạn slot, continuous batching giữ các slot đầy và trả các request ngắn sớm. Nâng giới hạn làm
            tăng số token mỗi giây cho đến khi bước vượt qua điểm gãy; qua đó, thời gian mỗi token của mỗi người dùng tăng lên mà
            thông lượng chẳng thêm bao nhiêu. Ở tải nhẹ, giới hạn hầu như không quan trọng, vì các slot không bao giờ đầy.
          </p>
        }
      >
        <p>
          At heavy load and the same slot limit, continuous batching keeps the slots full and returns short requests early.
          Raising the limit raises tokens per second until the step crosses the knee; past it, each user's time per token
          grows for little more throughput. Under light load the limit hardly matters, because the slots are never full.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Các server bạn có khả năng triển khai đều cho chỉnh giới hạn đó. vLLM (CUDA và ROCm) gọi nó là số chuỗi tối đa mỗi
            vòng lặp. Server của llama.cpp, trên CUDA, HIP và Metal, gọi nó là slot, với continuous batching bật theo mặc định.
            Trên máy Mac, server của mlx-lm batch liên tục qua một BatchGenerator, mặc định 32 request. Tên các tùy chọn là ảnh
            chụp mã nguồn của các dự án tại một thời điểm; sự đánh đổi thì là của mục này.
          </p>
        }
      >
        <p>
          The servers you are likely to deploy expose that limit. vLLM (CUDA and ROCm) calls it the maximum number of sequences
          per iteration. llama.cpp's server, on CUDA, HIP and Metal, calls it slots, with continuous batching on by
          default. On a Mac, mlx-lm's server batches continuously through a BatchGenerator, 32 requests by default. The
          option names are a snapshot of the projects' sources; the trade-off is this section's.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch22-facts D1 to D4)', '2026-11-20 (ch22-facts D1 đến D4)')}>
        <Translated
          vi={
            <p>
              Lập lịch ở mức vòng lặp, in-flight batching và continuous batching là cùng một kỹ thuật (tài liệu TensorRT-LLM). Giới
              hạn: <code>max_num_seqs</code> của vLLM; số slot <code>-np</code> của llama-server;{' '}
              <code>--decode-concurrency</code> của mlx-lm.
            </p>
          }
        >
          <p>
            Iteration-level scheduling, in-flight and continuous batching are one technique (TensorRT-LLM docs). The
            limit: vLLM's <code>max_num_seqs</code>; llama-server's <code>-np</code> slots; mlx-lm's{' '}
            <code>--decode-concurrency</code>.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS225['ch22/22.5/q-continuous']} />

      <h2><Translated vi={"Batched GEMM là chuyện khác"}>Batched GEMM is something else</Translated></h2>
      <Translated
        vi={
          <p>
            Các thư viện dùng chữ "batched" cho một thứ khác. <code>cublasGemmStridedBatchedEx</code> và{' '}
            <code>cublasGemmBatchedEx</code>, <code>hipblasGemmStridedBatchedEx</code> và{' '}
            <code>hipblasGemmBatchedEx</code> của hipBLAS, và <code>MPSMatrixMultiplication</code> của Metal Performance Shaders
            trên một khoảng đặt bởi <code>batchStart</code> và <code>batchSize</code> đều chạy nhiều GEMM độc lập, mỗi GEMM có
            toán hạng riêng. Batching của sách là một GEMM nhân với một ma trận trọng số dùng chung, với M bằng số request.
            Batched GEMM hợp với những phép nhân không chia sẻ gì, chẳng hạn attention của mỗi chuỗi với key riêng của nó.
          </p>
        }
      >
        <p>
          Libraries use the word "batched" for a different thing. <code>cublasGemmStridedBatchedEx</code> and{' '}
          <code>cublasGemmBatchedEx</code>, hipBLAS's <code>hipblasGemmStridedBatchedEx</code> and{' '}
          <code>hipblasGemmBatchedEx</code>, and Metal Performance Shaders' <code>MPSMatrixMultiplication</code> over a range
          set by <code>batchStart</code> and <code>batchSize</code> all run many independent GEMMs, each with its own
          operands. The book's batching is one GEMM against one shared weight matrix, with M equal to the number of
          requests. Batched GEMM fits products that share nothing, such as each sequence's attention against its own keys.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Bạn sẽ thấy cả hai trong llama.cpp. Backend CUDA và HIP của nó gửi phép nhân một trọng số 2 chiều với một batch token
            tới một GEMM thường. Đến 8 token, nó thường dùng các kernel nhân ma trận với vector của riêng nó thay vì GEMM (
            <code>MMVF_MAX_BATCH_SIZE</code> bằng 8); với một số kiểu lượng tử hóa trên những GPU cụ thể, giới hạn thấp hơn, vì
            kernel ma trận thắng sớm hơn. Backend Metal của nó làm tương tự với{' '}
            <code>mul_mv_ext</code> cho 2 đến 8 token, với hầu hết các kiểu trọng số. Tám là một hằng số trong mã, không phải số
            liệu của thiết bị: điểm giao được đo, và đó là ý của lab 22.2.
          </p>
        }
      >
        <p>
          You will see both in llama.cpp. Its CUDA and HIP backend sends a 2-D weight times a batch of tokens to one plain
          GEMM. Up to 8 tokens it normally uses its own matrix-vector kernels instead of a GEMM (
          <code>MMVF_MAX_BATCH_SIZE</code> is 8); for some quantised types on particular GPUs the limit is lower, because
          the matrix kernel wins sooner. Its Metal backend does the same with{' '}
          <code>mul_mv_ext</code> for 2 to 8 tokens, for most weight types. Eight is a code constant, not a device figure: the crossover is
          measured, which is lab 22.2's point.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch22-facts A11, B3, C3, D6)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              Các entry point có trong cuBLAS 13.5, hipBLAS ở ROCm 10.2 và macOS 27 SDK. Các ngưỡng của llama.cpp được đọc từ nhánh
              mặc định của nó và thay đổi thường xuyên.
            </p>
          }
        >
          <p>
            The entry points are in cuBLAS 13.5, hipBLAS at ROCm 10.2 and the macOS 27 SDK. llama.cpp's thresholds were read
            from its default branch and change often.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS225['ch22/22.5/q-batched-gemm']} />

      <h2><Translated vi={"Trên thiết bị của bạn"}>On your device</Translated></h2>
      <Translated
        vi={
          <p>
            Lab 22.2, mốc dự án M8, đo thời gian bước decode này với các kernel của chính bạn (M2, M3 hoặc gather của lab 22.1,
            phép cộng của M0, softmax và argmax của M4) ở các kích thước batch từ 1 trở lên, và ghi lại thời gian mỗi bước và số
            token mỗi giây. Hãy đưa ra dự đoán của nó trên <a href={href('ch22', 'labs')}>trang lab</a> từ trình khám phá ở trên,
            với thiết bị của bạn được chọn, trước khi chạy. Sau đó các hàng của nó sẽ xuất hiện trong trình khám phá đó.
          </p>
        }
      >
        <p>
          Lab 22.2, milestone M8, times this decode step with your own kernels (M2, M3 or lab 22.1's gather, M0's adds,
          M4's softmax and argmax) at batch sizes from 1 upward, and records time per step and tokens per second. Make its predictions on
          the <a href={href('ch22', 'labs')}>labs page</a> from the explorer above, with your device selected, before you
          run it. Its rows then appear in that explorer.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Phép đo tương tự trong một engine thực tế là <code>llama-batched-bench</code> của llama.cpp. Với <code>-tgs</code> nó
            decode các chuỗi lần lượt từng chuỗi; không có cờ đó, mỗi bước chứa một token từ mọi chuỗi: các phép nhân riêng rẽ so
            với một phép nhân của sách, trên một mô hình thực.
          </p>
        }
      >
        <p>
          The same measurement in a real engine is llama.cpp's <code>llama-batched-bench</code>. With <code>-tgs</code> it
          decodes the sequences one at a time; without it, each step holds one token from every sequence: the book's
          separate products against one product, on a real model.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch22-facts D3)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              <code>-tgs</code> nằm trong bộ phân tích tham số của công cụ, không có trong README của nó. Bảng trong README là số
              liệu của một máy.
            </p>
          }
        >
          <p>
            <code>-tgs</code> is in the tool's argument parser, not its README. The README's table is one machine's numbers.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Sách kết thúc mục này, và cả lập luận của nó, bằng quan điểm về cách người ta học: trước hết là khái niệm trong một
            mô hình lập trình, để có nền tảng vững chắc, rồi mới tổng quát hóa. Khóa học này thêm HIP và Metal trên cùng nền tảng
            đó.
          </p>
        }
      >
        <p>
          The book ends the section, and its own argument, on how it thinks people learn: concepts first in one
          programming model, for solid footing, then generalised. This course adds HIP and Metal on the same footing.
        </p>
      </Translated>
      <Source section="22.5" pages="540" />
    </>
  );
}
