import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Beyond, Source } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { benchRows, fmt, labStatus, num, realBackends, useReports, type BackendRun } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import { cubeAI, stencilIdealAI } from '../../sim/stencil3d';
import { sharedBytes, slabAI } from '../../sim/stencilslab';

// A kernel of our own, for one question. Checked copy: labs/ch08/snippets/regtile_two_planes.cu
// (Wrap.snippets.test.ts ties the two and pins the numbers the other questions take from the models).
const TWO_PLANES = `// Ours, not the book's: Fig. 8.12 (corrected) with its one shared plane
// doubled. Plane i is read from inCurr_s[b] and plane i + 1 is written
// to inCurr_s[1 - b], so the loop has one barrier where Fig. 8.12 has two.
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
__constant__ float c0, c1, c2, c3, c4, c5, c6;

__global__ void stencil_two_planes(const float* in, float* out,
                                   unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    int ty = threadIdx.y, tx = threadIdx.x;
    __shared__ float inCurr_s[2][IN_TILE_DIM][IN_TILE_DIM];
    float inPrev = 0.0f, inCurr = 0.0f, inNext = 0.0f;
    bool jk = j >= 0 && j < N && k >= 0 && k < N;
    if (jk && iStart >= 1 && iStart - 1 < N)
        inPrev = in[(iStart - 1)*N*N + j*N + k];
    if (jk && iStart < N)
        inCurr = in[iStart*N*N + j*N + k];
    inCurr_s[0][ty][tx] = inCurr;
    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        int b = (i - iStart) % 2;   // the buffer that holds plane i
        if (jk && i + 1 < N)
            inNext = in[(i + 1)*N*N + j*N + k];
        __syncthreads();            // the only barrier
        bool inGrid = i >= 1 && i < N - 1 && j >= 1 && j < N - 1
                   && k >= 1 && k < N - 1;
        bool inner = ty >= 1 && ty < IN_TILE_DIM - 1
                  && tx >= 1 && tx < IN_TILE_DIM - 1;
        if (inGrid && inner)
            out[i*N*N + j*N + k] = c0*inCurr
              + c1*inCurr_s[b][ty][tx - 1] + c2*inCurr_s[b][ty][tx + 1]
              + c3*inCurr_s[b][ty - 1][tx] + c4*inCurr_s[b][ty + 1][tx]
              + c5*inPrev + c6*inNext;
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[1 - b][ty][tx] = inNext;
    }
}`;

// ---------------------------------------------------------------- the numbers the questions take from the models
/**
 * Share of the ideal FLOP/B that one interior 2D tile of side t reaches (order 1): the
 * ideal is one load and one store per output, the tile loads t² and stores (t − 2)².
 * The FLOP per output cancel, so this holds for any stencil of order 1 in 2D.
 */
export const share2D = (t: number) => (2 * (t - 2) ** 2) / (t ** 2 + (t - 2) ** 2);
export const TILE_2D = 32;
/** Per cent, one decimal: 93.6 for the 32 × 32 tile. */
export const SHARE_2D_PCT = Math.round(1000 * share2D(TILE_2D)) / 10;
/** The 3D figures it is compared with: the 8 × 8 × 8 cube and the 32-wide slab as coded, as % of 13/8. */
export const SHARE_CUBE_PCT = Math.round((100 * cubeAI(8)) / stencilIdealAI(7));
export const SHARE_SLAB_PCT = Math.round((100 * slabAI(32, 30)) / stencilIdealAI(7));
/** Shared memory per 32 × 32 block: one plane (Fig. 8.12), two (the kernel above), three (Fig. 8.10). */
export const ONE_PLANE = sharedBytes('registers', 32);
export const TWO_PLANES_BYTES = 2 * ONE_PLANE;
export const THREE_PLANES = sharedBytes('planes', 32);

const n = (v: number) => v.toLocaleString('en-US');

export const questionsWrap: Record<string, Question> = {
  'ch08/wrap/q-2d-tile': {
    kind: 'numeric',
    id: 'ch08/wrap/q-2d-tile',
    prompt: (
      <Translated
        vi={
          <>
            Một stencil 2D 5 điểm được phân ô theo cách Hình 8.8 phân ô stencil 3D: một luồng cho mỗi phần tử đầu vào, các ô đầu vào{' '}
            {TILE_2D} × {TILE_2D}, {TILE_2D - 2} × {TILE_2D - 2} đầu ra mỗi khối. Một khối bên trong đạt được bao nhiêu phần FLOP/B
            lý tưởng của stencil 2D? Trả lời bằng phần trăm.
          </>
        }
      >
        <>
          A 2D 5-point stencil is tiled the way Fig. 8.8 tiles the 3D one: one thread per input element, {TILE_2D} ×{' '}
          {TILE_2D} input tiles, {TILE_2D - 2} × {TILE_2D - 2} outputs per block. What share of the 2D stencil's ideal
          FLOP/B does one interior block reach? Answer in per cent.
        </>
      </Translated>
    ),
    answer: SHARE_2D_PCT,
    tol: 0.01,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            Mỗi khối: {n((TILE_2D - 2) ** 2)} đầu ra, {n(TILE_2D ** 2)} lần nạp và {n((TILE_2D - 2) ** 2)} lần lưu. Mức lý tưởng là
            một lần nạp và một lần lưu cho mỗi đầu ra, nên khối đạt 2 × {n((TILE_2D - 2) ** 2)} / ({n(TILE_2D ** 2)} +{' '}
            {n((TILE_2D - 2) ** 2)}) = {SHARE_2D_PCT}% của mức đó; số phép toán mỗi đầu ra triệt tiêu nhau. Khối lập phương 8 × 8 × 8
            của trang 8.4 đạt {SHARE_CUBE_PCT}% mức lý tưởng của nó, và tấm rộng 32 của trang 8.5, như đã viết mã, đạt {SHARE_SLAB_PCT}%.
            Trong hai chiều, giới hạn số luồng vẫn cho phép một ô rộng, nên chỉ riêng phân ô (tiling) đã tiến gần mức đó. Gộp việc
            cho luồng (thread coarsening) là lời giải cho chiều thứ ba.
          </p>
        }
      >
        <p>
          Per block: {n((TILE_2D - 2) ** 2)} outputs, {n(TILE_2D ** 2)} loads and {n((TILE_2D - 2) ** 2)} stores. The ideal
          is one load and one store per output, so the block reaches 2 × {n((TILE_2D - 2) ** 2)} / ({n(TILE_2D ** 2)} +{' '}
          {n((TILE_2D - 2) ** 2)}) = {SHARE_2D_PCT}% of it; the operations per output cancel. The 8 × 8 × 8 cube of page
          8.4 reached {SHARE_CUBE_PCT}% of its ideal, and the 32-wide slab of page 8.5, as coded, {SHARE_SLAB_PCT}%. In two
          dimensions the thread limit still allows a wide tile, so tiling alone gets close. Coarsening is the answer to the
          third dimension.
        </p>
      </Translated>
    ),
    source: 'method of ch. 8 §8.4, p. 192 (the 2D ideal, 1.125 FLOP/B, is on p. 190); the 2D case is ours',
    vi: { source: 'phương pháp của ch. 8 §8.4, p. 192 (mức lý tưởng 2D, 1.125 FLOP/B, ở p. 190); trường hợp 2D là của chúng tôi' },
  },
  'ch08/wrap/q-box27': {
    kind: 'choice',
    id: 'ch08/wrap/q-box27',
    prompt: (
      <Translated
        vi={
          <>
            Một stencil 27 điểm đọc mọi điểm của hộp 3 × 3 × 3 quanh tâm, kể cả cạnh và góc. Bạn muốn áp dụng cả hai bước của Hình
            8.12 cho nó: cho một khối 2D tiến dọc z, và giữ các mặt phẳng trước và sau mặt phẳng hiện tại trong thanh ghi (register).
            Điều gì vẫn còn hiệu lực?
          </>
        }
      >
        <>
          A 27-point stencil reads every point of the 3 × 3 × 3 box around the centre, edges and corners included. You
          want both of Fig. 8.12's moves for it: march a 2D block through z, and keep the planes before and after the
          current one in registers. What still works?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Cả hai bước, không đổi: cửa sổ vẫn sâu ba mặt phẳng.">{'Both moves, unchanged: the window is still three planes deep.'}</Translated>,
        why: <Translated vi="Trong Hình 8.12, một luồng chỉ đọc điểm của chính nó trong các mặt phẳng trước và sau. Ở đây nó đọc chín điểm của mỗi mặt phẳng, tám điểm trong số đó do các luồng khác nạp.">{'In Fig. 8.12 a thread reads only its own point of the planes before and after. Here it reads nine points of each, eight of them loaded by other threads.'}</Translated>,
      },
      {
        label: <Translated vi="Tiến theo z vẫn được, nhưng các mặt phẳng trước và sau phải ở lại bộ nhớ chia sẻ: có tới chín luồng đọc mỗi giá trị của chúng.">{'Marching works, but the planes before and after must stay in shared memory: up to nine threads read each of their values.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Không bước nào: một stencil hình hộp đọc các góc, nên nó cần ô lập phương của Hình 8.8.">{'Neither: a box stencil reads corners, so it needs the cube tile of Fig. 8.8.'}</Translated>,
        why: <Translated vi="Tiến theo z không phụ thuộc vào hình dạng của stencil. Khối vẫn giữ một cửa sổ luân phiên ba mặt phẳng; nó chỉ đọc nhiều điểm hơn từ mỗi mặt phẳng.">{'Marching does not depend on the stencil\'s shape. The block still holds a rolling window of three planes; it just reads more of each.'}</Translated>,
      },
      {
        label: <Translated vi="Phân ô thanh ghi được, nhưng tiến theo z thì không: các góc nằm trong những mặt phẳng mà cửa sổ không giữ.">{'Register tiling works, but marching does not: the corners lie in planes the window does not hold.'}</Translated>,
        why: <Translated vi="Stencil vẫn có bậc 1 theo z: các mặt phẳng z − 1, z và z + 1 chứa mọi giá trị một đầu ra cần, kể cả các góc.">{'The stencil still has order 1 in z: planes z − 1, z and z + 1 hold every value an output needs, corners included.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sách giới hạn phân ô thanh ghi cho những stencil có các láng giềng nằm trên các trục đi qua tâm (p. 196), và lý do là
            ai đọc một giá trị. Trên các trục, chỉ luồng ở cùng (x, y) đọc một giá trị của các mặt phẳng trước và sau, nên giá trị đó
            có thể nằm trong thanh ghi của luồng ấy. Khi stencil có cả cạnh và góc, luồng ở (x, y) còn đọc (x ± 1, y ± 1) của các mặt
            phẳng đó, vốn do các luồng khác nạp, nên cả ba mặt phẳng đều phải được chia sẻ. Gộp việc cho luồng vẫn có lợi như trước:
            khối nạp đúng các mặt phẳng ấy, chỉ là đọc nhiều điểm hơn từ chúng.
          </p>
        }
      >
        <p>
          The book limits register tiling to stencils whose neighbours lie on the axes through the centre (p. 196), and
          the reason is who reads a value. On the axes, only the thread at the same (x, y) reads a value of the planes
          before and after, so it can live in that thread's registers. With edges and corners in the stencil, the thread
          at (x, y) also reads (x ± 1, y ± 1) of those planes, which other threads loaded, so all three planes must be
          shared. Coarsening still pays as before: the block loads exactly the same planes, and only reads more of them.
        </p>
      </Translated>
    ),
    source: 'ch. 8 §8.6, p. 196 (axis-only stencils); the 27-point case is ours',
    vi: { source: 'ch. 8 §8.6, p. 196 (stencil chỉ theo trục); trường hợp 27 điểm là của chúng tôi' },
  },
  'ch08/wrap/q-two-planes': {
    kind: 'choice',
    id: 'ch08/wrap/q-two-planes',
    prompt: (
      <Translated
        vi={
          <>
            Kernel ở trên là Hình 8.12 với mặt phẳng chia sẻ được nhân đôi: mặt phẳng i được đọc từ <code>inCurr_s[b]</code>, và
            mặt phẳng i + 1 được ghi vào <code>inCurr_s[1 - b]</code>. Nó có một rào chắn (barrier) mỗi vòng lặp, trong khi Hình 8.12
            có hai. Nó có đúng với mọi thứ tự luồng không, và thay đổi này tốn gì?
          </>
        }
      >
        <>
          The kernel above is Fig. 8.12 with its shared plane doubled: plane i is read from <code>inCurr_s[b]</code>, and
          plane i + 1 is written into <code>inCurr_s[1 - b]</code>. It has one barrier per iteration where Fig. 8.12 has
          two. Is it correct under every thread order, and what did the change cost?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Sai: một luồng nhanh có thể ghi đè mặt phẳng mà các láng giềng của nó vẫn đang đọc.">{'Wrong: a fast thread can overwrite the plane its neighbours are still reading.'}</Translated>,
        why: <Translated vi="Nó ghi vào buffer kia. Buffer đang được đọc chỉ được ghi lại ở cuối vòng lặp kế tiếp, sau một rào chắn mà không luồng nào vượt qua cho đến khi mọi luồng đã đọc xong trong vòng lặp này.">{'It writes the other buffer. The buffer being read is written again only at the end of the next iteration, after a barrier that no thread passes until every thread has finished this iteration\'s reads.'}</Translated>,
      },
      {
        label: <Translated vi="Đúng, và miễn phí: rào chắn thứ hai của Hình 8.12 vốn không bao giờ cần.">{'Correct, and free: Fig. 8.12\'s second barrier was never needed.'}</Translated>,
        why: <Translated vi="Trong Hình 8.12, lần ghi sau phép tính đi vào chính inCurr_s mà các láng giềng đang đọc. Không có rào chắn đó, một luồng nhanh sẽ ghi đè một giá trị mà một láng giềng chậm chưa đọc.">{'In Fig. 8.12 the write after the compute goes into the same inCurr_s the neighbours are reading. Without that barrier a fast thread overwrites a value a slow neighbour has not read yet.'}</Translated>,
      },
      {
        label: <Translated vi={`Đúng, với cái giá là một mặt phẳng bộ nhớ chia sẻ thứ hai: ${n(TWO_PLANES_BYTES)} byte mỗi khối 32 × 32 thay vì ${n(ONE_PLANE)}.`}>{`Correct, at the price of a second plane of shared memory: ${n(TWO_PLANES_BYTES)} bytes per 32 × 32 block instead of ${n(ONE_PLANE)}.`}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Sai: với một rào chắn, một luồng có thể đọc mặt phẳng i + 1 trước khi các láng giềng của nó ghi xong.">{'Wrong: with one barrier a thread can read plane i + 1 before its neighbours have written it.'}</Translated>,
        why: <Translated vi="Đó chính là điều rào chắn còn lại ngăn chặn. Nó nằm giữa các lần ghi mặt phẳng i + 1, ở cuối một vòng lặp, và các lần đọc nó ở vòng kế tiếp.">{'That is what the remaining barrier prevents. It sits between the writes of plane i + 1, at the end of one iteration, and the reads of it in the next.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Rào chắn thứ hai của Hình 8.12 bảo vệ một phụ thuộc giả (ghi sau đọc): mặt phẳng kế tiếp được ghi vào đúng mảng mà các
            láng giềng vẫn đang đọc. <a href={href('ch06', '6.7')}>Buffer kép của chương 6</a> loại bỏ nó bằng cách ghi vào chỗ khác;
            rào chắn còn lại bảo vệ phụ thuộc thật. Cái giá là thêm một mặt phẳng {n(ONE_PLANE)} byte, vẫn để khối thấp hơn ba mặt
            phẳng của Hình 8.10 ({n(THREE_PLANES)} byte). Việc bớt một rào chắn có bù được mặt phẳng thêm hay không là chuyện phải đo,
            không phải điều mô hình nói ra.
          </p>
        }
      >
        <p>
          Fig. 8.12's second barrier protects a false dependence (write after read): the next plane is written into the
          array the neighbours are still reading. <a href={href('ch06', '6.7')}>Chapter 6's double buffering</a> removes it by writing somewhere else; the
          barrier that remains protects the true dependence. The price is one more {n(ONE_PLANE)}-byte plane, which still
          leaves the block below Fig. 8.10's three planes ({n(THREE_PLANES)} bytes). Whether one barrier fewer pays for
          the extra plane is a measurement, not something the model says.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.7, pp. 146–147 (double buffering); ch. 8 §8.6, p. 197; the kernel is ours',
    vi: { source: 'ch. 6 §6.7, pp. 146–147 (buffer kép); ch. 8 §8.6, p. 197; kernel là của chúng tôi' },
  },
  'ch08/wrap/q-decode-placement': {
    kind: 'choice',
    id: 'ch08/wrap/q-decode-placement',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel attention ở bước decode giao một khối cho một head. Các key được cache của head mỗi key gồm 128 số float, lưu
            nối tiếp nhau. Mỗi luồng chấm điểm một số key với vector query của head, cũng 128 số float, và giữ một giá trị cực đại
            chạy và một tổng chạy trên các key của riêng nó. Theo quy tắc của chương này, mỗi giá trị nên nằm ở đâu?
          </>
        }
      >
        <>
          A decode-step attention kernel gives one block to one head. The head's cached keys are 128 floats each, stored
          one after another. Each thread scores some of the keys against the head's query vector, also 128 floats, and
          keeps a running maximum and a running sum over its own keys. By this chapter's rule, where does each value
          belong?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Tất cả trong bộ nhớ chia sẻ, query, các key và các giá trị chạy, vì bộ nhớ chia sẻ nhanh hơn bộ nhớ toàn cục.">{'All of it in shared memory, the query, the keys and the running values, since shared memory is faster than global.'}</Translated>,
        why: <Translated vi="Bộ nhớ chia sẻ dành cho các giá trị nhiều luồng đọc, hoặc để trung chuyển dữ liệu mà các lần nạp toàn cục sẽ không được hợp nhất. Một giá trị cực đại chạy chỉ do một luồng đọc và thay đổi ở mỗi key: giữ nó trong bộ nhớ chia sẻ chỉ thêm lưu lượng, không được gì khác.">{'Shared memory is for values several threads read, or for staging data whose global loads would not coalesce. A running maximum is read by one thread only and changes at every key: keeping it in shared memory adds traffic and nothing else.'}</Translated>,
      },
      {
        label: <Translated vi="Query trong bộ nhớ chia sẻ, giá trị cực đại chạy và tổng chạy của mỗi luồng trong thanh ghi của nó, và không chia sẻ key nào.">{"The query in shared memory, each thread's running maximum and sum in its registers, and no key shared."}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Query được chép vào thanh ghi của mọi luồng, mỗi luồng 128 số float, để không luồng nào phải chờ bộ nhớ chia sẻ.">{'The query copied into every thread\'s registers, 128 floats each, so that no thread waits on shared memory.'}</Translated>,
        why: <Translated vi="Mọi luồng sẽ giữ cùng 128 giá trị: ngân sách thanh ghi của trang 8.5 bị tiêu vào các bản sao. Ở 1,024 luồng, CUDA cho phép tối đa 64 thanh ghi mỗi luồng.">{'Every thread would hold the same 128 values: the register budget of page 8.5 spent on copies. At 1,024 threads, CUDA allows at most 64 registers per thread.'}</Translated>,
      },
      {
        label: <Translated vi="Các key trong bộ nhớ chia sẻ, để khối có thể dùng lại chúng giữa các luồng, và các giá trị chạy trong thanh ghi.">{'The keys in shared memory, so that the block can reuse them across threads, and the running values in registers.'}</Translated>,
        why: <Translated vi="Ở đây không key nào được dùng lại: mỗi key được chấm điểm một lần, với một query. Việc dùng lại xuất hiện khi một khối chấm điểm nhiều query với cùng các key, chẳng hạn nhiều query head dùng chung một bộ key; khi đó chia sẻ chúng mới có lợi. (Trung chuyển key qua bộ nhớ chia sẻ vẫn có thể có lợi vì một lý do khác; xem phần giải thích.)">{'No key is reused here: each is scored once, against one query. Reuse appears when one block scores several queries against the same keys, for instance several query heads that share one set of keys; then sharing them pays. (Staging keys through shared memory can still pay for another reason; see the explanation.)'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Hãy hỏi ai đọc mỗi giá trị. Mọi luồng đều đọc query: bộ nhớ chia sẻ, nạp một lần mỗi khối. Chỉ luồng sở hữu nó mới đọc
              một giá trị cực đại chạy hay tổng chạy: thanh ghi, giống <code>inPrev</code> và <code>inNext</code>. Một luồng đọc mỗi
              key, một lần: theo quy tắc của chương này thì không có gì để chia sẻ.
            </p>
            <p>
              Quy tắc này giải quyết chuyện chia sẻ. Nó không giải quyết chuyện một key đi từ bộ nhớ toàn cục đến luồng của nó ra sao,
              và bố cục này làm sai điều đó: với một luồng cho mỗi key, các lane kề nhau đọc những địa chỉ cách nhau 128 số float (512
              byte), một lần nạp không được hợp nhất theo cách nói của chương 6. Chương 6 đưa ra hai lối thoát. Đổi cách ánh xạ, để
              các lane của một warp chia mỗi key theo chiều head và các lane kề nhau đọc các số float kề nhau (sau đó chúng cộng các
              tích vô hướng từng phần lại). Hoặc nạp các key vào bộ nhớ chia sẻ bằng các lần nạp được hợp nhất rồi thực hiện các lần
              đọc theo bước nhảy ở đó, như kỹ thuật corner turning (§6.1), dù mỗi key chỉ có một người đọc. Việc áp dụng chúng cho
              attention là phác thảo của chúng tôi, không phải khảo sát các kernel thực tế; chương 20 xây các kernel attention cho mốc
              dự án M6.
            </p>
          </>
        }
      >
        <>
          <p>
            Ask who reads each value. Every thread reads the query: shared memory, loaded once per block. Only its own
            thread reads a running maximum or sum: registers, like <code>inPrev</code> and <code>inNext</code>. One thread
            reads each key, once: by this chapter's rule there is nothing to share.
          </p>
          <p>
            The rule settles sharing. It does not settle how a key gets from global memory to its thread, and this layout
            gets that wrong: with one thread per key, neighbouring lanes read addresses 128 floats (512 bytes) apart, an
            uncoalesced load in chapter 6's terms. Chapter 6 gives two ways out. Change the mapping, so that a warp's lanes
            split each key across the head dimension and neighbouring lanes read neighbouring floats (they then add up
            their partial dot products). Or load the keys into shared memory with coalesced loads and do the strided reads
            there, as corner turning does (§6.1), even though each key has a single reader. Applying them to attention is
            our sketch, not a survey of real kernels; chapter 20 builds attention kernels for milestone M6.
          </p>
        </>
      </Translated>
    ),
    source:
      'the rule of ch. 8 §8.6, p. 196; coalescing strategies in ch. 6 §6.1, p. 129; the attention setting and the layouts are ours; register budget in notes/today/ch08-facts.md A4',
    vi: {
      source:
        'quy tắc của ch. 8 §8.6, p. 196; các chiến lược hợp nhất truy cập trong ch. 6 §6.1, p. 129; bối cảnh attention và các bố cục là của chúng tôi; ngân sách thanh ghi trong notes/today/ch08-facts.md A4',
    },
  },
  'ch08/wrap/q-cache-basic': {
    kind: 'choice',
    id: 'ch08/wrap/q-cache-basic',
    prompt: (
      <Translated
        vi={
          <>
            Lab 8.1 báo tốc độ của mỗi kernel dưới dạng "GB/s ở 8 B/điểm": số điểm bên trong mỗi giây nhân 8 byte, như thể mỗi
            điểm tốn một lần nạp và một lần lưu. Giả sử con số của kernel cơ bản xấp xỉ băng thông mà phép thăm dò của lab 5.1 đo
            được trên cùng thiết bị. Điều gì suy ra từ đó?
          </>
        }
      >
        <>
          Lab 8.1 reports each kernel's speed as "GB/s at 8 B/point": interior points per second times 8 bytes, as if every
          point cost one load and one store. Suppose the basic kernel's figure comes out close to the bandwidth that lab
          5.1's probe measured on the same device. What follows?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Kernel cơ bản chỉ yêu cầu 8 byte mỗi điểm, nên con số 32 mà mô hình đếm là sai.">{'The basic kernel requests only 8 bytes per point, so the model\'s count of 32 was wrong.'}</Translated>,
        why: <Translated vi="Nó yêu cầu 32: bảy lần nạp và một lần lưu. Mã quyết định cái gì được yêu cầu; phần cứng quyết định bao nhiêu trong số đó đến được DRAM.">{'It requests 32: seven loads and one store. The code fixes what is requested; the hardware decides how much of it reaches DRAM.'}</Translated>,
      },
      {
        label: <Translated vi="Phần lớn trong 32 byte được yêu cầu mỗi điểm không bao giờ đến DRAM, nên phân ô thanh ghi còn lại rất ít lưu lượng DRAM để tiết kiệm.">{'Most of its 32 requested bytes per point never reach DRAM, so register tiling has little DRAM traffic left to save.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Thiết bị còn dư băng thông, nên kernel phân ô thanh ghi sẽ chạy nhanh hơn khoảng 3.6 lần, như số byte dự đoán.">{'The device has bandwidth to spare, so the register-tiled kernel will run about 3.6 times faster, as the byte count predicts.'}</Translated>,
        why: <Translated vi="Kernel đã di chuyển dữ liệu ở khoảng tốc độ mà phép thăm dò tìm được. Nhanh hơn 3.6 lần nghĩa là khoảng 2 byte lưu lượng DRAM mỗi điểm, ít hơn một lần nạp và một lần lưu.">{'The kernel already moves data at about the rate the probe found. 3.6 times faster would mean about 2 bytes of DRAM traffic per point, less than one load and one store.'}</Translated>,
      },
      {
        label: <Translated vi="Phép đo hẳn là sai, vì 0.41 FLOP/B chặn kernel ở một phần tư mức đó.">{'The measurement must be wrong, since 0.41 FLOP/B caps the kernel at a quarter of that.'}</Translated>,
        why: <Translated vi="Mức chặn đó giả định mọi byte được yêu cầu đều đến từ DRAM. Bộ nhớ đệm phá vỡ giả định này, như sách lưu ý (p. 189).">{'That cap assumes every requested byte comes from DRAM. Caches break the assumption, as the book notes (p. 189).'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mọi lượt quét đúng trên một lưới lớn hơn nhiều so với bộ nhớ đệm (cache) đều đọc mỗi đầu vào từ DRAM và ghi mỗi đầu ra
            vào đó ít nhất một lần: khoảng 8 byte mỗi điểm. Nếu số điểm mỗi giây của kernel cơ bản nhân 8 byte đã gần băng thông mà
            thiết bị cung cấp, nó không thể di chuyển nhiều hơn thế bao nhiêu, dù nó yêu cầu bao nhiêu. Phân ô thanh ghi cắt giảm
            yêu cầu, không phải mức sàn đó. Trên một thiết bị như vậy, mọi lợi ích phải đến từ việc ít truy cập cache hơn.
          </p>
        }
      >
        <p>
          Every correct sweep of a grid much larger than the caches reads each input from DRAM and writes each output to
          it at least once: about 8 bytes per point. If the basic kernel's points per second times 8 bytes is already near
          the bandwidth the device delivers, it cannot be moving much more than that, whatever it requests. Register
          tiling cuts requests, not that floor. On such a device any gain has to come from fewer cache accesses.
        </p>
      </Translated>
    ),
    source: 'ch. 8 §8.3, p. 189 (caches raise the measured ratio); the reasoning is ours; measurements from labs 5.1 and 8.1',
    vi: { source: 'ch. 8 §8.3, p. 189 (bộ nhớ đệm làm tăng tỉ lệ đo được); lập luận là của chúng tôi; số đo từ lab 5.1 và 8.1' },
  },
};

// ---------------------------------------------------------------- this machine's numbers for the last question
/** The rows of one lab on one backend: the reader's own where they passed, else the reference's. */
function preferredRows(run: BackendRun | undefined, lab: string) {
  const mine = labStatus(run, lab, 'work') === 'pass' ? benchRows(run, lab, 'work') : [];
  return mine.length ? mine : benchRows(run, lab, 'solution');
}

function OnThisMachine() {
  const { reports, loading } = useReports();
  const lines = realBackends(reports).flatMap((b) => {
    const basic = preferredRows(reports[b], 'ch08/lab1_stencil3d').find((r) => r.name === 'basic');
    const roof = preferredRows(reports[b], 'ch05/lab1_roofline').find((r) => r.name === 'roofline');
    const a = num(basic?.gbps_compulsory);
    const bw = num(roof?.bandwidth_gb_per_s);
    if (!a || !bw) return [];
    return [{ b, device: String(reports[b].device.name ?? b), a, bw, n: basic?.n }];
  });
  if (loading) return null;
  return lines.length ? (
    <>
      {lines.map((l) => (
        <p key={l.b} data-machine={l.b} data-ratio={fmt(l.a / l.bw)}>
          <Translated
            vi={
              <>
                Số liệu của chính bạn, {l.device} ({l.b}): kernel cơ bản của lab 8.1 chạy ở <b>{fmt(l.a)} GB/s ở 8 B/điểm</b> trên
                một lưới cạnh {String(l.n)}, và phép thăm dò băng thông của lab 5.1 đo được <b>{fmt(l.bw)} GB/s</b>: tỉ lệ{' '}
                {fmt(l.a / l.bw)}. Trang lab có con số của kernel phân ô thanh ghi bên cạnh.
              </>
            }
          >
            Your own numbers, {l.device} ({l.b}): the basic kernel of lab 8.1 ran at <b>{fmt(l.a)} GB/s at 8 B/point</b> on a
            grid of side {String(l.n)}, and lab 5.1's bandwidth probe measured <b>{fmt(l.bw)} GB/s</b>: a ratio of{' '}
            {fmt(l.a / l.bw)}. The labs page has the register-tiled kernel's figure beside it.
          </Translated>
        </p>
      ))}
    </>
  ) : (
    <p className="muted" data-machine="none">
      <Translated
        vi={
          <>
            Để đưa thiết bị của chính bạn vào câu hỏi này, bạn cần các hàng benchmark từ lab 8.1 và từ lab 5.1 trên cùng
            backend: <code>python3 labs/pmpp.py run 8.1 --bench</code> và <code>python3 labs/pmpp.py run 5.1 --bench</code>.
          </>
        }
      >
        To put your own device into this question you need benchmark rows from lab 8.1 and from lab 5.1 on the same
        backend: <code>python3 labs/pmpp.py run 8.1 --bench</code> and <code>python3 labs/pmpp.py run 5.1 --bench</code>.
      </Translated>
    </p>
  );
}

// ---------------------------------------------------------------- the page
export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports);
  const lab = 'ch08/lab1_stencil3d';
  const passing = [...backends, 'emu'].filter((b) => labStatus(reports[b], lab, 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch08/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>
        <Translated vi="Những gì cần mang theo">What to carry forward</Translated>
      </h2>
      <ul>
        <li>
          <Translated
            vi={
              <>
                Tỉ lệ của một ô do vùng biên (halo) của nó quyết định. Một khối lập phương nhỏ chủ yếu là vùng biên (58% ở 8 × 8 × 8),
                và với một luồng cho mỗi phần tử, giới hạn số luồng, chứ không phải công thức, quyết định khối lập phương lớn được đến đâu.
              </>
            }
          >
            A tile's ratio is set by its halo. A small cube is mostly halo (58% at 8 × 8 × 8), and with one thread per
            element the thread limit, not the formula, decides how large the cube can get.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Gộp việc theo một trục để thoát khỏi giới hạn đó: một khối là một mặt phẳng của ô tiến qua chiều thứ ba, giữ một cửa sổ
                luân phiên các mặt phẳng trên chip. Ô lớn lên mà không cần thêm luồng, và độ sâu của tấm trở thành một núm vặn đánh đổi
                tỉ lệ lấy số khối.
              </>
            }
          >
            Coarsen along one axis to escape that limit: a block that is one plane of the tile marches through the third
            dimension, holding a rolling window of planes on chip. The tile grows without more threads, and the slab depth
            becomes a knob that trades ratio for the number of blocks.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Một buffer chia sẻ được dùng lại qua các vòng lặp cần hai rào chắn mỗi vòng: một trước các lần đọc (đọc sau ghi), một
                trước lần ghi đè (ghi sau đọc). Mọi luồng phải đến được cả hai, nên hãy bọc điều kiện quanh các lần nạp và lưu và đừng
                bao giờ return sớm. Một buffer thứ hai loại bỏ rào chắn thứ hai.
              </>
            }
          >
            A shared buffer reused across iterations needs two barriers per iteration: one before the reads (read after
            write), one before the overwrite (write after read). Every thread must reach both, so guard loads and stores and
            never return early. A second buffer removes the second barrier.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Một giá trị chỉ đúng một luồng đọc thuộc về thanh ghi của luồng đó; chỉ những giá trị nhiều luồng đọc mới cần bộ nhớ chia
                sẻ (shared memory). Điều đó tiết kiệm bộ nhớ chia sẻ và lưu lượng trên chip, tốn thanh ghi, và để nguyên lưu lượng toàn
                cục.
              </>
            }
          >
            A value that exactly one thread reads belongs in that thread's registers; only values that several threads read
            need shared memory. That saves shared memory and on-chip traffic, costs registers, and leaves global traffic
            unchanged.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Giữ cửa sổ của mỗi luồng ở các chỉ số hằng, dưới dạng các biến vô hướng có tên hoặc một vòng lặp được unroll hoàn toàn.
                NVIDIA ghi nhận một mảng được đánh chỉ số lúc chạy là ứng viên cho bộ nhớ cục bộ (local memory), vốn là bộ nhớ thiết bị.
              </>
            }
          >
            Keep a per-thread window at constant indices, as named scalars or a fully unrolled loop. NVIDIA documents a
            run-time-indexed array as a candidate for local memory, which is device memory.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Thanh ghi là ngân sách của cả khối. Ở 1024 luồng, CUDA cho phép tối đa 64 thanh ghi mỗi luồng và cho lần khởi chạy thất
                bại nếu vượt quá mà không có <code>__launch_bounds__</code>; HIP giới hạn số thanh ghi và có thể tràn (spill); Metal có
                thể hạ giới hạn số luồng của pipeline. Hãy đọc các giới hạn; đừng giả định chúng.
              </>
            }
          >
            Registers are a block budget. At 1024 threads CUDA allows at most 64 per thread and fails the launch above that
            without <code>__launch_bounds__</code>; HIP limits registers and may spill; Metal may lower the pipeline's
            thread cap. Read the limits;
            do not assume them.
          </Translated>
        </li>
        <li>
          <Translated vi="Mô hình đếm số byte được yêu cầu; bộ nhớ đệm quyết định cái gì đến được DRAM. Hãy dự đoán từ mô hình, rồi đo.">
            The model counts requested bytes; caches decide what reaches DRAM. Predict from the model, then measure.
          </Translated>
        </li>
      </ul>
      <Source section="8.7" pages="198" />

      <h2>
        <Translated vi="Bài kiểm tra tổng hợp">A mixed check</Translated>
      </h2>
      <p>
        <Translated vi="Các câu này dựa vào cả chương và vào những tình huống mà các trang chưa trình bày.">
          These draw on the whole chapter and on situations the pages did not show.
        </Translated>
        {asked
          ? tr(
              ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.`,
              ` Đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu hỏi, tính theo câu trả lời gần nhất của bạn cho mỗi câu.`,
            )
          : ''}
      </p>
      <Beyond>
        <Translated
          vi={
            <p>
              Các trường hợp 2D và 27 điểm, kernel hai mặt phẳng và bối cảnh attention là của chúng tôi. Kernel hai mặt phẳng là
              front-end-checked và chưa được chạy; một unit test chạy một mô hình vòng lặp của nó từng luồng một theo ba thứ tự.
            </p>
          }
        >
          <p>
            The 2D and 27-point cases, the two-plane kernel and the attention setting are ours. The two-plane kernel is
            front-end-checked and has not been run; a unit test runs a model of its loop one thread at a time under three
            orders.
          </p>
        </Translated>
      </Beyond>
      <Ask q={q['ch08/wrap/q-2d-tile']} label={tr('Work it out', 'Tự tính ra')} />
      <Ask q={q['ch08/wrap/q-box27']} />
      <Code
        tabs={[
          {
            label: tr('CUDA (not a lab file: for this question only)', 'CUDA (không phải tệp lab: chỉ dành cho câu hỏi này)'),
            code: TWO_PLANES,
            status: tr('front-end-checked; never run', 'front-end-checked; chưa từng chạy'),
            tiers: {
              in: 'global', out: 'global', inCurr_s: 'shared',
              inPrev: 'reg', inCurr: 'reg', inNext: 'reg', i: 'reg', j: 'reg', k: 'reg', iStart: 'reg', b: 'reg', ty: 'reg', tx: 'reg', jk: 'reg',
              c0: 'const', c1: 'const', c2: 'const', c3: 'const', c4: 'const', c5: 'const', c6: 'const',
            },
          },
        ]}
      />
      <Ask q={q['ch08/wrap/q-two-planes']} label={tr('Read the code', 'Đọc mã')} />
      <Ask q={q['ch08/wrap/q-decode-placement']} />
      <Ask q={q['ch08/wrap/q-cache-basic']} />
      <OnThisMachine />

      <h2>
        <Translated vi="Bài tập của sách">The book's exercises</Translated>
      </h2>
      <p>
        <Translated
          vi={
            <>
              Chương kết thúc bằng hai bài tập gồm chín phần (pp. 198–199), cả hai đều về định kích thước cho các kernel của chương:
              bài thứ nhất cho một lưới 120 × 120 × 120, bài thứ hai cho một khối đã gộp việc. Mọi phần đều được giải ở trang 8.4 hoặc
              8.5; sách không in đáp án, nên đáp án ở đó là của chúng tôi.
            </>
          }
        >
          The chapter ends with two exercises in nine parts (pp. 198–199), both about sizing the chapter's kernels: the
          first for a 120 × 120 × 120 grid, the second for one coarsened block. Every part is worked on page 8.4 or 8.5;
          the book prints no answers, so the answers there are ours.
        </Translated>
      </p>
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
                <Translated vi="Được giải ở đâu tại đây">Where it is worked here</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1a</td>
              <td>
                <Translated vi="Số điểm đầu ra được tính mỗi lượt quét">Output points computed per sweep</Translated>
              </td>
              <td>
                <Translated vi={<>một câu hỏi trong <a href={href('ch08', '8.4')}>8.4</a></>}>
                  a question in <a href={href('ch08', '8.4')}>8.4</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>1b</td>
              <td>
                <Translated vi="Số khối cho kernel cơ bản với khối 8 × 8 × 8">Blocks for the basic kernel with 8 × 8 × 8 blocks</Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      một câu hỏi trong <a href={href('ch08', '8.4')}>8.4</a>; phần khởi chạy cơ bản của{' '}
                      <a href={href('ch08', 'labs')}>lab 8.1</a>
                    </>
                  }
                >
                  a question in <a href={href('ch08', '8.4')}>8.4</a>; the basic launch of{' '}
                  <a href={href('ch08', 'labs')}>lab 8.1</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>1c</td>
              <td>
                <Translated vi="Số khối cho kernel phân ô lập phương với khối 8 × 8 × 8">
                  Blocks for the cube-tiled kernel with 8 × 8 × 8 blocks
                </Translated>
              </td>
              <td>
                <Translated vi={<>một câu hỏi trong <a href={href('ch08', '8.4')}>8.4</a></>}>
                  a question in <a href={href('ch08', '8.4')}>8.4</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>1d</td>
              <td>
                <Translated vi="Số khối cho kernel gộp việc theo z với khối 32 × 32">
                  Blocks for the z-coarsened kernel with 32 × 32 blocks
                </Translated>
              </td>
              <td>
                <Translated vi={<>một câu hỏi trong <a href={href('ch08', '8.5')}>8.5</a></>}>
                  a question in <a href={href('ch08', '8.5')}>8.5</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>2a, 2b</td>
              <td>
                <Translated vi="Số phần tử đầu vào và đầu ra của một khối gộp việc theo hệ số 16">
                  Input and output elements of a block coarsened by 16
                </Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      mỗi phần một câu hỏi trong <a href={href('ch08', '8.5')}>8.5</a>, và mục "Số lần nạp và lưu mỗi khối" của trình khám
                      phá tấm ở D = 16
                    </>
                  }
                >
                  a question each in <a href={href('ch08', '8.5')}>8.5</a>, and the slab explorer's "Loads and stores per
                  block" at D = 16
                </Translated>
              </td>
            </tr>
            <tr>
              <td>2c</td>
              <td>
                <Translated vi="FLOP/B của khối đó">That block's FLOP/B</Translated>
              </td>
              <td>
                <Translated vi={<>một câu hỏi trong <a href={href('ch08', '8.5')}>8.5</a>, và trình khám phá tấm ở D = 16</>}>
                  a question in <a href={href('ch08', '8.5')}>8.5</a>, and the slab explorer at D = 16
                </Translated>
              </td>
            </tr>
            <tr>
              <td>2d, 2e</td>
              <td>
                <Translated vi="Bộ nhớ chia sẻ mỗi khối khi không có và khi có phân ô thanh ghi">
                  Shared memory per block without and with register tiling
                </Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      một câu hỏi trong <a href={href('ch08', '8.5')}>8.5</a>, và phần đọc bộ nhớ chia sẻ của trình khám phá cho cả hai kernel
                    </>
                  }
                >
                  a question in <a href={href('ch08', '8.5')}>8.5</a>, and the explorer's shared-memory readout for both
                  kernels
                </Translated>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>
        <Translated vi="Những chỗ PDF có vẻ sai">Where the PDF looks wrong</Translated>
      </h2>
      <p>
        <Translated
          vi={
            <>
              Bản sách mà khóa học này dựa vào có vài chỗ có vẻ sai sót ở chương 8, và vài khẳng định mà tài liệu hiện hành không ủng
              hộ. Chúng được liệt kê để bạn không bị bối rối khi đọc sách song song.
            </>
          }
        >
          The copy of the book this course was built from has a few apparent slips in chapter 8, and a few statements that
          current documentation does not support. They are listed so that you are not thrown by them when you read the
          book alongside.
        </Translated>
      </p>
      <ul data-list="errata">
        <li>
          <Translated
            vi={
              <>
                Hình 8.10 và 8.12 ghép <code>c3</code> với láng giềng y + 1 và <code>c4</code> với y − 1, ngược với Hình 8.6 và 8.8;
                kết quả khác nhau trừ khi hai trọng số bằng nhau. Khóa học và lab 8.1 theo Hình 8.6 và 8.8 (
                <a href={href('ch08', '8.4')}>8.4</a>, <a href={href('ch08', '8.5')}>8.5</a>).
              </>
            }
          >
            Figs. 8.10 and 8.12 pair <code>c3</code> with the y + 1 neighbour and <code>c4</code> with y − 1, the reverse
            of Figs. 8.6 and 8.8; the results differ unless the two weights are equal. The course and lab 8.1 follow Figs.
            8.6 and 8.8 (<a href={href('ch08', '8.4')}>8.4</a>, <a href={href('ch08', '8.5')}>8.5</a>).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Dòng cuối của Hình 8.12 lưu <code>inNext_s</code>, thứ mà kernel đó không khai báo; ý là thanh ghi{' '}
                <code>inNext</code>. Số dòng gần cuối của nó chạy 33, 34, 35, 34, 35, 35, và "dòng 34" trong văn bản là số 34 thứ hai (
                <a href={href('ch08', '8.5')}>8.5</a>).
              </>
            }
          >
            Fig. 8.12's last line stores <code>inNext_s</code>, which that kernel does not declare; the register{' '}
            <code>inNext</code> is meant. Its line numbers near the end run 33, 34, 35, 34, 35, 35, and the text's "line
            34" is the second 34 (<a href={href('ch08', '8.5')}>8.5</a>).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 195 cho kernel đã gộp việc 1.52 FLOP/B ở t = 32. Chính công thức của sách và Hình 8.10 như đã viết mã cho 1.47;
                1.52 là giới hạn cho một tấm có độ sâu không chặn (<a href={href('ch08', '8.5')}>8.5</a>).
              </>
            }
          >
            p. 195 gives the coarsened kernel 1.52 FLOP/B at t = 32. The book's own formula and Fig. 8.10 as coded give
            1.47; 1.52 is the limit for a slab of unbounded depth (<a href={href('ch08', '8.5')}>8.5</a>).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 196 nói phân ô thanh ghi giảm bộ nhớ chia sẻ xuống một phần ba "so với kernel trong Hình 8.12"; ý là Hình 8.10.
              </>
            }
          >
            p. 196 says register tiling cuts shared memory to a third of "that by the kernel in Fig. 8.12"; Fig. 8.10 is
            meant.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <code>IN_TILE_DIM</code> và <code>OUT_TILE_DIM</code> được dùng trong Hình 8.8, 8.10 và 8.12 nhưng không hình nào định
                nghĩa; văn bản cho đầu ra = đầu vào − 2 (p. 192). Lab 8.1 định nghĩa cả hai. Không listing nào cho thấy mã host.
              </>
            }
          >
            <code>IN_TILE_DIM</code> and <code>OUT_TILE_DIM</code> are used in Figs. 8.8, 8.10 and 8.12 and defined in
            none; the text gives output = input − 2 (p. 192). Lab 8.1 defines both. No listing shows host code.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 186 nói các stencil trong "Hình 8.3(a), (b), và (c)" có bậc 1, 2 và 3; ý là Hình 8.2 (chú thích của Hình 8.3 cho
                1, 2, 1, 2).
              </>
            }
          >
            p. 186 says the stencils in "Fig. 8.3(a), (b), and (c)" have orders 1, 2 and 3; Fig. 8.2 is meant (Fig. 8.3's
            caption gives 1, 2, 1, 2).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 187 mô tả stencil 2D 5 điểm là dùng hai điểm lưới ở mỗi bên của tâm dọc theo mỗi trục. Thực ra nó dùng một điểm mỗi
                bên, hai điểm mỗi trục.
              </>
            }
          >
            p. 187 describes the 2D 5-point stencil as using two grid points on each side of the centre along each axis.
            It uses one on each side, two per axis.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 184 gọi 0.87 là "giá trị x tương ứng với phần tử F[2]"; 0.87 là giá trị được lưu, sin(2π/6), còn giá trị x là
                2π/6.
              </>
            }
          >
            p. 184 calls 0.87 "the x value that corresponds to element F[2]"; 0.87 is the stored value, sin(2π/6), and the
            x value is 2π/6.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Sai sót nhỏ: "compared 6.25 FLOP/B" (p. 190), "a copy the current plane" (p. 196), và{' '}
                <code>N - 1&amp;&amp; k</code> thiếu dấu cách trong Hình 8.6.
              </>
            }
          >
            Small slips: "compared 6.25 FLOP/B" (p. 190), "a copy the current plane" (p. 196), and{' '}
            <code>N - 1&amp;&amp; k</code> without a space in Fig. 8.6.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Không phải lỗi, nhưng không được giải thích: Hình 8.5 để bốn ô góc của lưới đầu vào không tô. Một stencil 5 điểm không
                bao giờ đọc chúng, điểm "không có góc" của <a href={href('ch08', '8.4')}>8.4</a>.
              </>
            }
          >
            Not an error, but unexplained: Fig. 8.5 leaves the input grid's four corner cells unshaded. A 5-point stencil
            never reads them, the "no corners" point of <a href={href('ch08', '8.4')}>8.4</a>.
          </Translated>
        </li>
      </ul>
      <Translated vi={<p>Và những chỗ tài liệu hiện hành của các nhà cung cấp nói khác:</p>}>
        <p>And where current vendor documentation says something else:</p>
      </Translated>
      <ul data-list="dated">
        <li>
          <Translated
            vi={
              <>
                p. 192, "giới hạn thực tế của t là 8": một khối 10 × 10 × 10 gồm 1000 luồng nằm trong mọi giới hạn CUDA và HIP được
                ghi nhận. 8 là lựa chọn lũy thừa của hai của sách (<a href={href('ch08', '8.4')}>8.4</a>).
              </>
            }
          >
            p. 192, "the practical limit on t is 8": a 10 × 10 × 10 block of 1000 threads is within every documented CUDA
            and HIP limit. 8 is the book's power-of-two choice (<a href={href('ch08', '8.4')}>8.4</a>).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 193, các lần nạp của ô 8 × 8 × 8 "không được hợp nhất": theo mô hình 32 byte của NVIDIA, chúng đạt một nửa hiệu suất,
                và khối 32 × 8 × 4 được gợi ý đạt 80%, không phải 100%, vì phần − 1 của vùng biên (
                <a href={href('ch08', '8.4')}>8.4</a>).
              </>
            }
          >
            p. 193, the 8 × 8 × 8 tile's loads are "not coalesced": by NVIDIA's 32-byte model they are half efficient, and
            the suggested 32 × 8 × 4 block reaches 80%, not 100%, because of the halo's − 1 (
            <a href={href('ch08', '8.4')}>8.4</a>).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                pp. 184–185, độ chính xác đơn và bán chính xác có thông lượng "cao hơn nhiều" so với double: tỉ lệ FP32 so với FP64 của
                NVIDIA chạy từ 2:1 đến 64:1 tùy compute capability, và Metal không có <code>double</code>.
              </>
            }
          >
            pp. 184–185, single and half precision have "much higher" throughput than double: NVIDIA's FP32 to FP64 ratio
            runs from 2:1 to 64:1 depending on the compute capability, and Metal has no <code>double</code>.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 197, thanh ghi có "độ trễ thấp hơn đáng kể và băng thông cao hơn" bộ nhớ chia sẻ: NVIDIA ghi nhận thứ tự độ trễ và
                không có con số nào; Apple không ghi nhận gì về thanh ghi.
              </>
            }
          >
            p. 197, registers have "significantly lower latency and higher bandwidth" than shared memory: NVIDIA documents
            the latency order and no figures; Apple documents nothing about registers.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Sách thiếu điều này: trên CUDA, một khối 32 × 32 có kernel cần hơn 64 thanh ghi mỗi luồng sẽ không khởi chạy được trừ
                khi kernel mang <code>__launch_bounds__</code> (<a href={href('ch08', '8.5')}>8.5</a>).
              </>
            }
          >
            Missing from the book: on CUDA a 32 × 32 block whose kernel needs more than 64 registers per thread fails to
            launch unless the kernel carries <code>__launch_bounds__</code> (<a href={href('ch08', '8.5')}>8.5</a>).
          </Translated>
        </li>
      </ul>
      <Aside>
        <Translated
          vi={
            <p>
              Đây là cách đọc của chúng tôi đối với một bản PDF, được đối chiếu với tài liệu hiện hành của các nhà cung cấp ngày
              2026-11-20 (notes/today/ch08-facts.md). Nếu bản của bạn khác, hãy tin bản của bạn.
            </p>
          }
        >
          <p>
            These are our readings of one PDF, checked against current vendor documentation on 2026-11-20
            (notes/today/ch08-facts.md). If your copy differs, trust your copy.
          </p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Những gì chương này lược bỏ">What this chapter left out</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương này được rút gọn trong khóa học, nên đây là những gì không được đề cập và nơi tìm chúng. Phần lớn bối cảnh của
            §8.1 (pp. 184–187): lấy mẫu sin x trên một lưới (Hình 8.1), lưới có cấu trúc và không có cấu trúc, nội suy, đạo hàm
            sai phân hữu hạn và sai số của nó, và các hình vẽ stencil của Hình 8.2 đến 8.4; trang bắt đầu giữ lại hai câu về nó.
            Phần lớn danh sách khác biệt so với tích chập (convolution) ở phần mở đầu (p. 183): rằng stencil phục vụ các bộ giải lặp,
            rằng trọng số của chúng đến từ phương trình vi phân, và các ràng buộc thứ tự mà một số bộ giải áp đặt giữa các đầu ra,
            trong đó trang bắt đầu chỉ giữ giả định loại trừ chúng (§8.2). Hai điểm của danh sách đó vẫn còn: các mẫu stencil cho
            phép những tối ưu mà tích chập không cho phép (phân ô thanh ghi, trang 8.5), và độ chính xác, mà cùng với phần bàn về
            nó ở §8.1 (pp. 184–185) trở thành thiết lập số byte mỗi giá trị trong các trình khám phá. Điểm về lưới 3D là của phần
            tóm tắt §8.7 (p. 198), và là lý do có trang 8.5.
          </p>
        }
      >
        <p>
          The chapter is condensed in this course, so here is what is not covered and where to find it. Most of §8.1's
          background (pp. 184–187): sampling sin x on a grid (Fig. 8.1), structured and unstructured grids, interpolation,
          the finite-difference derivative and its error, and the stencil drawings of Figs. 8.2 to 8.4; the start page
          keeps two sentences of it. Most of the opening's list of differences from convolution (p. 183): that stencils
          serve iterative solvers, that their weights come from the differential equation, and the ordering constraints
          some solvers impose between outputs, of which the start page keeps only the assumption that rules them out
          (§8.2). Two points of that list do survive: stencil patterns that allow optimisations convolution does not
          (register tiling, page 8.5), and precision, which with §8.1's discussion of it (pp. 184–185) becomes the
          bytes-per-value setting in the explorers. The 3D-grid point is the §8.7 summary's (p. 198), and it is the
          reason for page 8.5.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Hình 8.9 và 8.11 được vẽ lại bằng hoạt cảnh và trình chạy từng bước của trang 8.5 chứ không sao chép lại. Bạn không viết
            kernel phân ô lập phương của Hình 8.8 hay kernel ba mặt phẳng của Hình 8.10: lab 8.1 cung cấp cả hai, đã sửa và đã kiểm
            thử. Stencil 3D bậc cao hơn chỉ xuất hiện trong các trình khám phá và trong vài câu hỏi, và khối 32 × 8 × 4 chỉ trong
            trình khám phá warp của trang 8.4; không kernel nào được xây cho cả hai. Không phần nào về sau trong khóa học phụ thuộc
            vào những gì đã bị lược bỏ.
          </p>
        }
      >
        <p>
          Figs. 8.9 and 8.11 are redrawn by page 8.5's animation and stepper rather than reproduced. You do not write the
          cube-tiled kernel of Fig. 8.8 or the three-plane kernel of Fig. 8.10: lab 8.1 provides both, corrected and tested.
          Higher-order 3D stencils appear only in the explorers and in a few questions, and the 32 × 8 × 4 block only in
          page 8.4's warp explorer; no kernel is built for either. Nothing later in the course depends on what was left
          out.
        </p>
      </Translated>

      <h2>
        <Translated vi="Dự án đến nay">The project so far</Translated>
      </h2>
      <p>
        <Translated
          vi={
            <>
              Chương này không thêm mốc dự án nào vào dự án xuyên suốt. Hai kỹ thuật của nó nằm trên đường đến một mốc:
              phép nhân ma trận gộp việc, phân ô thanh ghi của <a href={href('ch15', 'start')}>chương 15</a> là mốc dự án M3,
              xây trên cùng quy tắc về việc ai đọc một giá trị. Lab 8.1 là tùy chọn.{' '}
            </>
          }
        >
          This chapter adds no milestone to the through-line project. Its two techniques are on the way to one:{' '}
          <a href={href('ch15', 'start')}>chapter 15</a>'s coarsened, register-tiled matrix multiplication is milestone M3,
          built on the same rule about who reads a value.
          Lab 8.1 is optional.{' '}
        </Translated>
        <span data-lab81={passing.join(',') || 'none'}>
          {backends.length === 0 && !reports['emu']
            ? tr('No lab results are recorded on this machine yet.', 'Máy này chưa ghi kết quả lab nào.')
            : passing.length
              ? tr(
                  `Your own version passes on ${passing.map(where).join(' and ')} on this machine.`,
                  `Phiên bản của chính bạn đã qua trên ${passing.map(where).join(' và ')} trên máy này.`,
                )
              : tr('Your own version has not passed on this machine yet.', 'Phiên bản của chính bạn chưa qua trên máy này.')}
        </span>{' '}
        <Translated vi={<>Xem <a href={href('ch08', 'labs')}>lab</a>.</>}>
          See the <a href={href('ch08', 'labs')}>lab</a>.
        </Translated>
      </p>
      <Translated
        vi={
          <p>
            Chương 9 bỏ lại một thói quen của mọi kernel cho đến nay: mỗi đầu ra thuộc về đúng một luồng. Trong một histogram, ô
            (bin) mà một luồng cập nhật phụ thuộc vào dữ liệu nó đọc, nên các luồng va chạm trên đầu ra. Điều đó cần các phép toán
            nguyên tử (atomic) và, để chúng nhanh, riêng tư hóa (privatization): các bản sao đầu ra cho mỗi khối trong bộ nhớ chia
            sẻ. Kernel của nó cũng là mảnh đầu tiên của đường lấy mẫu, các số đếm token đằng sau một repetition penalty (mốc dự án
            M5). Bắt đầu ở <a href={href('ch09', 'start')}>chương 9</a>.
          </p>
        }
      >
        <p>
          Chapter 9 leaves a habit of every kernel so far behind: each output belonged to exactly one thread. In a
          histogram, which bin a thread updates depends on the data it reads, so threads collide on outputs. That needs
          atomic operations and, to make them fast, privatisation: per-block copies of the output in shared memory. Its
          kernel is also the first piece of the sampling path, the token counts behind a repetition penalty (milestone M5).
          Start at <a href={href('ch09', 'start')}>chapter 9</a>.
        </p>
      </Translated>
    </>
  );
}
