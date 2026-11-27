import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Source } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';

// A kernel of our own, in the spirit of the book's exercise 10 (p. 120): each block
// transposes one B x B tile of a matrix in place, through shared memory.
const TRANSPOSE = `#define B 8   // tile width; the matrix width is a multiple of B

__global__ void transposeTiles(float* A, int width) {
    __shared__ float tile[B][B];
    int idx = (blockIdx.y * B + threadIdx.y) * width + blockIdx.x * B + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = A[idx];
    A[idx] = tile[threadIdx.x][threadIdx.y];
}`;

export const questionsWrap: Record<string, Question> = {
  'ch05/wrap/q-transpose': {
    kind: 'choice',
    id: 'ch05/wrap/q-transpose',
    prompt: (
      <Translated
        vi={
          <>
            Kernel ở trên được khởi chạy với <code>B × B</code> luồng mỗi khối. Với những giá trị nào của <code>B</code>{' '}
            thì nó chắc chắn cho ra các ô đã chuyển vị?
          </>
        }
      >
        The kernel above is launched with <code>B × B</code> threads per block. For which values of <code>B</code> is
        it guaranteed to produce the transposed tiles?
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Mọi B.">Any B.</Translated>,
        why: (
          <Translated vi="Mỗi luồng đọc một ô mà một luồng khác ghi. Điều gì bảo đảm lần ghi đó xảy ra trước?">
            Each thread reads a slot that a different thread writes. What makes that write happen first?
          </Translated>
        ),
      },
      {
        label: <Translated vi="Mọi B không vượt quá độ rộng của một warp.">Any B up to the width of a warp.</Translated>,
        why: (
          <Translated vi="Việc thực thi đồng bộ từng bước trong một warp có thể che giấu vấn đề, nhưng ngôn ngữ không hứa điều đó, và một khối với B = 8 đã có 64 luồng.">
            Lockstep execution inside one warp may hide the problem, but the language does not promise it, and a block with B = 8 already has 64 threads.
          </Translated>
        ),
      },
      { label: <Translated vi="Chỉ B = 1.">Only B = 1.</Translated>, correct: true },
      {
        label: <Translated vi="Không có B nào cả.">No B at all.</Translated>,
        why: (
          <Translated vi="Với một luồng mỗi khối, ô duy nhất được đọc là ô mà chính luồng đó vừa ghi.">
            With one thread per block the only slot read is the one the thread itself just wrote.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Luồng (x, y) đọc <code>tile[x][y]</code>, ô do luồng (y, x) ghi. Không có gì sắp thứ tự hai việc đó, nên lần
            đọc có thể thấy bộ nhớ chưa khởi tạo: một phụ thuộc đọc-sau-ghi mà không có rào chắn (barrier). Chỉ khi luồng
            đọc chính ô của nó (x = y, điều mà với mọi luồng có nghĩa là B = 1) thì mới an toàn. Cách sửa là một{' '}
            <code>__syncthreads()</code> giữa hai câu lệnh. Không cần rào chắn thứ hai, vì ô không bao giờ bị ghi đè.
          </p>
        }
      >
        <p>
          Thread (x, y) reads <code>tile[x][y]</code>, which thread (y, x) writes. Nothing orders the two, so the read
          can see uninitialised memory: a read-after-write dependence with no barrier. Only when the thread reads its own
          slot (x = y, which for every thread means B = 1) is it safe. The fix is one <code>__syncthreads()</code>{' '}
          between the two statements. No second barrier is needed, because the tile is never overwritten.
        </p>
      </Translated>
    ),
    source: 'after ch. 5 exercise 10, pp. 119–120; dependence vocabulary from §5.4, pp. 110–111',
    vi: { source: 'theo bài tập 10 chương 5, tr. 119–120; thuật ngữ về phụ thuộc từ §5.4, tr. 110–111' },
  },
  'ch05/wrap/q-flopb': {
    kind: 'numeric',
    id: 'ch05/wrap/q-flopb',
    prompt: (
      <Translated
        vi={
          <>
            Mỗi luồng của một kernel nạp sáu số float từ bộ nhớ toàn cục (global memory), nạp thêm hai số từ bộ nhớ chia
            sẻ (shared memory), thực hiện chín phép nhân và tám phép cộng, rồi ghi hai số float ra bộ nhớ toàn cục. Tỉ số
            FLOP trên mỗi byte lưu lượng bộ nhớ toàn cục của nó là bao nhiêu?
          </>
        }
      >
        Per thread, a kernel loads six floats from global memory, loads two more from shared memory, performs nine
        multiplications and eight additions, and stores two floats to global memory. What is its ratio in FLOP per byte
        of global-memory traffic?
      </Translated>
    ),
    answer: 0.53,
    tol: 0.03,
    unit: 'FLOP/B',
    explain: (
      <Translated
        vi={
          <p>
            17 phép toán. Lưu lượng toàn cục: sáu lần nạp và hai lần ghi, mỗi lần 4 byte = 32 byte. 17 / 32 = 0.53 FLOP/B.
            Hai lần nạp từ bộ nhớ chia sẻ không tốn gì vào băng thông toàn cục, và đó chính là lý do để đặt dữ liệu được
            dùng lại ở đó.
          </p>
        }
      >
        <p>
          17 operations. Global traffic: six loads and two stores of 4 bytes = 32 bytes. 17 / 32 = 0.53 FLOP/B. The two
          shared-memory loads cost nothing against global bandwidth, which is the whole reason to put reused data there.
        </p>
      </Translated>
    ),
    source: 'method of ch. 5 §5.1, pp. 94–97; compare exercise 11(f), p. 121',
    vi: { source: 'phương pháp của chương 5 §5.1, tr. 94–97; so sánh bài tập 11(f), tr. 121' },
  },
  'ch05/wrap/q-notile': {
    kind: 'choice',
    id: 'ch05/wrap/q-notile',
    prompt: (
      <Translated
        vi={
          <>
            Bạn được nhờ tăng tốc một kernel tính <code>y[i] = a * x[i] + y[i]</code> cho một vector rất dài. Nạp{' '}
            <code>x</code> và <code>y</code> vào các ô (tile) trong bộ nhớ chia sẻ có giúp được không?
          </>
        }
      >
        You are asked to speed up a kernel that computes <code>y[i] = a * x[i] + y[i]</code> for a very long vector.
        Would loading <code>x</code> and <code>y</code> into shared memory tiles help?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có: bộ nhớ chia sẻ nhanh hơn bộ nhớ toàn cục, nên kernel nào dùng nó cũng được lợi.">
            Yes: shared memory is faster than global memory, so any kernel gains from using it.
          </Translated>
        ),
        why: (
          <Translated vi="Dữ liệu vẫn phải đến từ bộ nhớ toàn cục một lần. Chép nó vào bộ nhớ chia sẻ trước chỉ thêm việc, trừ khi có gì đó dùng lại nó.">
            The data still has to come from global memory once. Copying it to shared memory first adds work unless something reuses it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không: mỗi phần tử được một luồng đọc, một lần, nên không có lần nạp nào lặp lại.">
            No: each element is read by one thread, once, so no fetch repeats.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có, nhưng chỉ khi ô rộng ít nhất bằng một warp.">
            Yes, but only if the tile is at least as wide as a warp.
          </Translated>
        ),
        why: (
          <Translated vi="Kích thước ô không tạo ra sự dùng lại mà bản thân phép tính không có.">
            Tile size does not create reuse that the computation does not have.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Phân ô (tiling) loại bỏ lưu lượng <em>dư thừa</em>. Nó cần dữ liệu mà nhiều luồng của một khối cùng muốn, hoặc
            một luồng muốn nhiều lần. Ở đây mỗi byte chỉ được chạm một lần: 2 FLOP cho 12 byte, 0.17 FLOP/B, và đó đã là
            lưu lượng tối thiểu cho phép tính này. Kernel này vốn bị giới hạn bởi bộ nhớ, và câu hỏi hữu ích trở thành nó
            chạy gần tốc độ ánh sáng (speed of light) đến mức nào.
          </p>
        }
      >
        <p>
          Tiling removes <em>redundant</em> traffic. It needs data that several threads of a block want, or that one
          thread wants several times. Here every byte is touched once: 2 FLOP for 12 bytes, 0.17 FLOP/B, and that is
          already the minimum traffic for this computation. The kernel is memory-bound by nature, and the useful question
          becomes how close to the speed of light it runs.
        </p>
      </Translated>
    ),
    source: 'ch. 5 exercise 1, p. 119, and §5.1, p. 96',
    vi: { source: 'bài tập 1 chương 5, tr. 119, và §5.1, tr. 96' },
  },
  'ch05/wrap/q-measured': {
    kind: 'choice',
    id: 'ch05/wrap/q-measured',
    prompt: (
      <Translated
        vi={
          <>
            Trên một thiết bị nào đó, kernel matmul ngây thơ đo được nhanh gấp tám lần, <em>vượt trên</em> mức trần mà
            0.25 FLOP/B và băng thông đo được dự đoán. Trước khi chạy thêm gì khác, bạn nên chờ đợi gì từ việc phân ô với
            T = 16 trên thiết bị đó?
          </>
        }
      >
        On some device the naive matmul kernel measures eight times <em>above</em> the ceiling that 0.25 FLOP/B and the
        measured bandwidth predict. Before running anything else, what should you expect from tiling with T = 16 there?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nhanh hơn khoảng 16×: mức giảm lưu lượng của mô hình ở T = 16 vẫn áp dụng trọn vẹn.">
            About 16× faster: the model's traffic reduction at T = 16 still applies in full.
          </Translated>
        ),
        why: (
          <Translated vi="Con số 16× của mô hình là mức giảm lưu lượng được yêu cầu. Rõ ràng phần lớn lưu lượng đó vốn đã không đến bộ nhớ toàn cục.">
            The model's 16× is a reduction in requested traffic. Much of that traffic was evidently not reaching global memory in the first place.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Thấp hơn hẳn 16×: bộ nhớ đệm (cache) vốn đã hấp thụ phần lớn lưu lượng lặp lại.">
            Well under 16×: caches were already absorbing much of the repeated traffic.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không được lợi gì, vì kernel ngây thơ đã chạy vượt trên mức trần của nó.">
            No gain at all, since the naive kernel already runs above its ceiling.
          </Translated>
        ),
        why: (
          <Translated vi="Phân ô vẫn bớt được việc (và cũng có thể cải thiện kiểu truy cập), nên nhiều khả năng vẫn có lợi; chỉ là không bằng hệ số của mô hình.">
            Tiling still removes work (and can also improve access patterns), so some gain is likely; it is just not the model's factor.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một kernel ngây thơ chạy vượt xa mức trần của mô hình đang cho bạn biết tỉ số hiệu dụng của nó đã cao hơn 0.25
            nhiều, vì bộ nhớ đệm trên chip hấp thụ các lần đọc lặp lại. Phân ô biến sự dùng lại đó thành tường minh và
            đáng tin cậy, nhưng nó chỉ tiết kiệm được lưu lượng thực sự đang xảy ra. Dự đoán trước, rồi đo: lab 5.2 cho
            thấy phân ô được lợi bao nhiêu trên thiết bị của bạn.
          </p>
        }
      >
        <p>
          A naive kernel running far above its model ceiling is telling you its effective ratio is already much higher
          than 0.25, because on-chip caches absorb repeated reads. Tiling makes that reuse explicit and reliable, but it
          can only save traffic that was really happening. Predict, then measure: lab 5.2 shows what tiling gains on
          your device.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.1, p. 97 (caches raise the real ratio); measurements from lab 5.1 and 5.2',
    vi: { source: 'chương 5 §5.1, tr. 97 (bộ nhớ đệm nâng tỉ số thực); số đo từ lab 5.1 và 5.2' },
  },
  'ch05/wrap/q-llm': {
    kind: 'choice',
    id: 'ch05/wrap/q-llm',
    prompt: (
      <Translated
        vi={
          <>
            Sinh từng token một với batch size 1 dành phần lớn thời gian cho các phép nhân ma trận với vector. Thay đổi
            nào nâng tỉ số phép toán trên mỗi byte của các phép nhân đó?
          </>
        }
      >
        Token-by-token generation at batch size 1 spends most of its time in matrix-vector products. Which change
        raises the operations-per-byte ratio of those products?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Độ rộng ô lớn hơn, để mỗi trọng số được nạp phục vụ nhiều luồng hơn.">
            A larger tile width, so each loaded weight serves more threads.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi trọng số được dùng một lần cho mỗi token; không có lần nạp lặp lại nào để một ô chia sẻ.">
            Each weight is used once per token; there is no repeated fetch for a tile to share.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nhiều luồng hơn mỗi khối, để nhiều trọng số được nạp cùng lúc hơn.">
            More threads per block, so more weights are in flight at once.
          </Translated>
        ),
        why: (
          <Translated vi="Điều đó thay đổi cách chia việc, chứ không thay đổi số byte mỗi phép toán cần.">
            That changes how the work is divided, not how many bytes each operation needs.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Lưu trọng số bằng 8 bit thay vì 32.">Storing the weights in 8 bits instead of 32.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Thêm một rào chắn thứ hai.">Adding a second barrier.</Translated>,
        why: (
          <Translated vi="Rào chắn liên quan đến tính đúng đắn, không phải lưu lượng.">
            Barriers are about correctness, not traffic.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hai phép toán cho mỗi trọng số, một lần nạp trọng số đó: tỉ số là 2 chia cho số byte mỗi trọng số. Float32 cho
            0.5 FLOP/B, int8 cho 2. Đó là mức trần tăng gấp bốn cho một kernel bị giới hạn bởi bộ nhớ, trước khi phép tính
            nào rẻ đi. Đòn bẩy còn lại là batching, cho phép một trọng số đã nạp phục vụ nhiều hàng.
          </p>
        }
      >
        <p>
          Two operations per weight, one fetch of that weight: the ratio is 2 divided by the bytes per weight. Float32
          gives 0.5 FLOP/B, int8 gives 2. That is a fourfold rise in the ceiling for a memory-bound kernel, before any
          arithmetic gets cheaper. The other lever is batching, which lets one fetched weight serve several rows.
        </p>
      </Translated>
    ),
    source: 'our analysis with the §5.1 method; the book makes the batching argument in §20.6, p. 505',
    vi: { source: 'phân tích của chúng tôi theo phương pháp §5.1; sách trình bày lập luận về batching ở §20.6, tr. 505' },
  },
};

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports);
  const m2 = backends.filter((b) => labStatus(reports[b], 'ch05/lab3_gemm', 'work') === 'pass');
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch05/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>
        <Translated vi="Những gì cần mang theo">What to carry forward</Translated>
      </h2>
      <Translated
        vi={
          <ul>
            <li>
              Một kernel có một tỉ số, số phép toán trên mỗi byte lưu lượng bộ nhớ toàn cục (global memory), còn một
              thiết bị có một ngưỡng, thông lượng đỉnh chia cho băng thông đỉnh. Dưới ngưỡng, bộ nhớ quyết định bạn chạy
              nhanh đến đâu.
            </li>
            <li>
              Mô hình đếm những gì kernel yêu cầu. Bộ nhớ đệm (cache) quyết định bao nhiêu phần trong đó đến được bộ nhớ
              toàn cục, nên mô hình cho hướng đi, còn phép đo cho độ lớn.
            </li>
            <li>
              Dữ liệu có thể nằm trong thanh ghi (register) (một luồng), bộ nhớ chia sẻ (shared memory) (một khối), bộ
              nhớ toàn cục hoặc bộ nhớ hằng (constant memory) (tất cả mọi người). Bộ nhớ càng nhanh càng nhỏ, và dùng
              chúng tốn occupancy (mức chiếm dụng).
            </li>
            <li>
              Phân ô (tiling) chuyển dữ liệu được dùng lại vào một bộ nhớ nhanh hơn. Với matmul dùng các ô{' '}
              <code>T × T</code>, nó cắt lưu lượng được yêu cầu đi <code>T</code> lần, và cần hai rào chắn (barrier) mỗi
              pha: một cho phụ thuộc thật, một cho phụ thuộc giả.
            </li>
            <li>
              Mỗi lần truy cập bộ nhớ có phép kiểm tra biên riêng. Một luồng không sở hữu đầu ra nào vẫn có thể phải nạp
              dữ liệu cho khối của nó.
            </li>
            <li>Dung lượng là thuộc tính của từng thiết bị. Hãy truy vấn chúng; đừng giả định.</li>
          </ul>
        }
      >
        <ul>
          <li>
            A kernel has a ratio, operations per byte of global-memory traffic, and a device has a threshold, peak
            throughput over peak bandwidth. Below the threshold, memory decides how fast you go.
          </li>
          <li>
            The model counts what a kernel asks for. Caches decide how much of that reaches global memory, so the model
            gives direction and a measurement gives magnitude.
          </li>
          <li>
            Data can live in registers (one thread), shared memory (one block), global or constant memory (everyone).
            Faster memories are smaller, and using them costs occupancy.
          </li>
          <li>
            Tiling moves reused data into a faster memory. For matmul with <code>T × T</code> tiles it cuts requested
            traffic by a factor of <code>T</code>, and it needs two barriers per phase: one for a true dependence, one
            for a false one.
          </li>
          <li>
            Every memory access gets its own bounds check. A thread that owns no output may still have to load for its
            block.
          </li>
          <li>Capacities are properties of a device. Query them; do not assume them.</li>
        </ul>
      </Translated>
      <Source section="5.7" pages="118–119" />

      <h2>
        <Translated vi="Kiểm tra tổng hợp">A mixed check</Translated>
      </h2>
      <p>
        <Translated vi="Các câu này dựa trên cả chương và trên những tình huống mà các phần chưa trình bày.">
          These draw on the whole chapter and on situations the sections did not show.
        </Translated>
        {asked
          ? tr(
              ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.`,
              ` Đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu, tính theo câu trả lời gần nhất của bạn cho mỗi câu.`,
            )
          : ''}
      </p>
      <Code
        tabs={[
          {
            label: tr('CUDA (not a lab file: for this question only)', 'CUDA (không phải tệp lab: chỉ cho câu hỏi này)'),
            code: TRANSPOSE,
            tiers: { A: 'global', tile: 'shared', idx: 'reg' },
            status: 'front-end-checked (labs/ch05/snippets/transpose_tiles_defect.cu)',
          },
        ]}
      />
      <Ask q={questionsWrap['ch05/wrap/q-transpose']} label={tr('Find the defect', 'Tìm lỗi')} />
      <Ask q={questionsWrap['ch05/wrap/q-flopb']} />
      <Ask q={questionsWrap['ch05/wrap/q-notile']} />
      <Ask q={questionsWrap['ch05/wrap/q-measured']} />
      <Ask q={questionsWrap['ch05/wrap/q-llm']} />

      <h2>
        <Translated vi="Bài tập của sách">The book's exercises</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương kết thúc bằng mười hai bài tập (tr. 119–121). Bài nào cũng được đề cập ở đâu đó trong khóa học này;
            làm cả theo cách sách diễn đạt cũng đáng công.
          </p>
        }
      >
        <p>
          The chapter ends with twelve exercises (pp. 119–121). Each is covered somewhere in this course; the book's own
          wording is worth doing as well.
        </p>
      </Translated>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr>
            <th><Translated vi="Bài tập">Exercise</Translated></th>
            <th><Translated vi="Về">About</Translated></th>
            <th><Translated vi="Được giải ở đâu trong khóa học">Where it is worked here</Translated></th>
          </tr>
        </thead>
        <tbody>
          <tr><td>1</td><td><Translated vi="Bộ nhớ chia sẻ cho phép cộng ma trận">Shared memory for matrix addition</Translated></td><td><Translated vi={<><a href={href('ch05', '5.3')}>5.3</a>, và câu hỏi saxpy ở trên</>}><a href={href('ch05', '5.3')}>5.3</a>, and the saxpy question above</Translated></td></tr>
          <tr><td>2</td><td><Translated vi="8×8 với các ô 2×2 và 4×4">8×8 with 2×2 and 4×4 tiles</Translated></td><td><Translated vi={<>trình khám phá "ai đọc gì" trong <a href={href('ch05', '5.3')}>5.3</a></>}>the "who reads what" explorer in <a href={href('ch05', '5.3')}>5.3</a></Translated></td></tr>
          <tr><td>3</td><td><Translated vi="Quên một rào chắn">Forgetting a barrier</Translated></td><td><Translated vi={<>trình chạy từng bước và bảng kết quả trong <a href={href('ch05', '5.4')}>5.4</a>; lab 5.4</>}>stepper and outcome table in <a href={href('ch05', '5.4')}>5.4</a>; lab 5.4</Translated></td></tr>
          <tr><td>4</td><td><Translated vi="Bộ nhớ chia sẻ so với thanh ghi">Shared memory versus registers</Translated></td><td><a href={href('ch05', '5.2')}>5.2</a></td></tr>
          <tr><td>5, 8</td><td><Translated vi="Giảm lưu lượng nhờ phân ô">Traffic reduction with tiles</Translated></td><td><a href={href('ch05', '5.3')}>5.3</a></td></tr>
          <tr><td>6, 7</td><td><Translated vi="Một biến có bao nhiêu bản sao">How many copies of a variable</Translated></td><td><a href={href('ch05', '5.2')}>5.2</a></td></tr>
          <tr><td>9</td><td><Translated vi="Bị giới hạn bởi tính toán hay bởi bộ nhớ trên hai thiết bị">Compute- or memory-bound on two devices</Translated></td><td><a href={href('ch05', '5.1')}>5.1</a></td></tr>
          <tr><td>10</td><td><Translated vi="Một phép chuyển vị ô có lỗi">A tile transpose with a defect</Translated></td><td><Translated vi="câu hỏi đầu tiên ở trên (kernel của chúng tôi theo cùng tinh thần)">the first question above (our own kernel in the same spirit)</Translated></td></tr>
          <tr><td>11</td><td><Translated vi="Số bản sao, số byte chia sẻ và tỉ số của một kernel">Copies, shared bytes and ratio of a kernel</Translated></td><td><Translated vi={<><a href={href('ch05', '5.2')}>5.2</a>, và câu hỏi về tỉ số ở trên</>}><a href={href('ch05', '5.2')}>5.2</a>, and the ratio question above</Translated></td></tr>
          <tr><td>12</td><td><Translated vi="Occupancy dưới ba mức ngân sách">Occupancy under three budgets</Translated></td><td><Translated vi={<>công cụ tính ngân sách trong <a href={href('ch05', '5.6')}>5.6</a></>}>the budget calculator in <a href={href('ch05', '5.6')}>5.6</a></Translated></td></tr>
        </tbody>
      </table>

      <h2>
        <Translated vi="Những chỗ PDF có vẻ sai">Where the PDF looks wrong</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Ấn bản này còn mới và bản sao mà khóa học dựa vào có vài chỗ có vẻ nhầm ở chương 5. Chúng được liệt kê ở đây
            để bạn không bị bối rối khi đọc sách song song.
          </p>
        }
      >
        <p>
          This edition is new and the copy this course was built from has a few apparent slips in chapter 5. They are
          listed so that you are not thrown by them when you read the book alongside.
        </p>
      </Translated>
      <ul>
        <Translated vi={<li>Tr. 97 ghi ngưỡng của thiết bị ví dụ là "25.1 FLOP/B or 20.0 FLOP/B". Ở mọi chỗ khác nó là 20.0, đúng bằng kết quả chia hai giá trị đỉnh.</li>}>
          <li>p. 97 gives the example device's threshold as "25.1 FLOP/B or 20.0 FLOP/B". Everywhere else it is 20.0, which is what the two peaks divide to.</li>
        </Translated>
        <Translated vi={<li>Hình 5.8 đánh số các pha là 0 và 1; phần chữ bên cạnh gọi chúng là 1 và 2. Mã đếm từ 0, và khóa học này cũng vậy.</li>}>
          <li>Fig. 5.8 numbers the phases 0 and 1; the text beside it calls them 1 and 2. The code counts from 0, and so does this course.</li>
        </Translated>
        <Translated vi={<li>Tr. 106–107 nói mức giảm là "by a factor of N if the tiles are N × N". Ở đây N là độ rộng ô, không phải ma trận N được nhắc đến một dòng trước đó.</li>}>
          <li>pp. 106–107 say the reduction is "by a factor of N if the tiles are N × N". Here N is the tile width, not the matrix N named a line earlier.</li>
        </Translated>
        <Translated vi={<li>Hình 5.14 (bộ nhớ chia sẻ có kích thước động) có vài lỗi trong bản in; <a href={href('ch05', '5.6')}>phần 5.6</a> đi qua từng lỗi và lab 5.5 có một phiên bản đã sửa và đã kiểm thử.</li>}>
          <li>Fig. 5.14 (dynamically sized shared memory) has several errors as printed; <a href={href('ch05', '5.6')}>section 5.6</a> goes through them and lab 5.5 has a corrected, tested version.</li>
        </Translated>
        <Translated vi={<li>Tr. 113 nói luồng (1,0) của khối (1,1) phải nạp M[2][1] dù nó không sở hữu đầu ra nào. Theo chính cách gọi (hàng, cột) của sách thì luồng đó sẽ nạp M[3][0]; luồng nạp M[2][1] là (0,1). Ý mà nó minh họa vẫn đúng.</li>}>
          <li>p. 113 says thread (1,0) of block (1,1) must load M[2][1] although it owns no output. By the book's own (row, column) naming that thread would load M[3][0]; the thread that loads M[2][1] is (0,1). The point it illustrates stands.</li>
        </Translated>
        <Translated vi={<li>Tr. 116 mô tả <code>sharedMemPerBlock</code> là bộ nhớ chia sẻ của một SM. Tài liệu CUDA định nghĩa nó theo từng khối.</li>}>
          <li>p. 116 describes <code>sharedMemPerBlock</code> as the shared memory of an SM. The CUDA documentation defines it per block.</li>
        </Translated>
        <Translated vi={<li>Đề bài tập 10 viết <code>BLOCK_WIDTH</code> trong khi mã của nó viết <code>BLOCK_SIZE</code>; hai cái có cùng nghĩa.</li>}>
          <li>Exercise 10's text says <code>BLOCK_WIDTH</code> where its code says <code>BLOCK_SIZE</code>; they mean the same thing.</li>
        </Translated>
        <Translated vi={<li>Đoạn mã của bài tập 11 có một lỗi dàn trang ở dòng 18, và <code>foo</code> nhận <code>int*</code> trong khi <code>foo_kernel</code> nhận <code>float*</code>.</li>}>
          <li>Exercise 11's listing has a typesetting artefact in line 18, and <code>foo</code> takes <code>int*</code> where <code>foo_kernel</code> takes <code>float*</code>.</li>
        </Translated>
        <Translated vi={<li>Bài tập 12 ghi lượng bộ nhớ chia sẻ dùng "per SM" cho một kernel; gần như chắc chắn ý là theo từng khối.</li>}>
          <li>Exercise 12 states shared-memory use "per SM" for a kernel; per block is almost certainly meant.</li>
        </Translated>
      </ul>
      <Aside>
        <Translated vi={<p>Đây là cách chúng tôi đọc một bản PDF. Nếu bản của bạn khác, hãy tin bản của bạn.</p>}>
          <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Dự án đến lúc này">The project so far</Translated>
      </h2>
      <p>
        <Translated
          vi={
            <>
              Kernel của lab 5.3, một phép nhân ma trận phân ô cho mọi kích thước, là mốc dự án M2 của dự án xuyên suốt:
              nó là phép tính đứng sau mọi lớp tuyến tính của mô hình transformer nhỏ mà cuối cùng bạn sẽ chạy.
            </>
          }
        >
          Lab 5.3's kernel, a tiled matrix multiplication for any shapes, is milestone M2 of the through-line
          project: it is the computation behind every linear layer of the small transformer you will end up running.
        </Translated>{' '}
        {backends.length === 0
          ? tr('No lab results are recorded on this machine yet.', 'Chưa có kết quả lab nào được ghi lại trên máy này.')
          : m2.length
            ? tr(
                `Your own version passes on ${m2.join(' and ')} on this machine.`,
                `Phiên bản của bạn đã qua trên ${m2.join(' và ')} trên máy này.`,
              )
            : tr('Your own version has not passed on this machine yet.', 'Phiên bản của bạn chưa qua trên máy này.')}{' '}
        <Translated
          vi={
            <>
              Xem các <a href={href('ch05', 'labs')}>bài thực hành</a>.
            </>
          }
        >
          See the <a href={href('ch05', 'labs')}>labs</a>.
        </Translated>
      </p>
      <Translated
        vi={
          <p>
            Chương 6 tiếp nhận một chi tiết mà chương này đã bước vòng qua. Kernel ngây thơ đọc M dọc theo bộ nhớ và N cắt
            ngang nó. Hai kiểu truy cập đó không tốn như nhau, và lý do, cách các yêu cầu bộ nhớ của một warp được hợp nhất
            hay không, là đòn bẩy lớn tiếp theo sau phân ô:{' '}
            <a href={href('ch06', '6.1')}>phần 6.1, Hợp nhất truy cập bộ nhớ toàn cục (coalescing)</a>.
          </p>
        }
      >
        <p>
          Chapter 6 picks up a detail this chapter stepped around. The naive kernel reads M along memory and N across it.
          Those two patterns do not cost the same, and the reason, how a warp's memory requests are merged or not, is the
          next large lever after tiling: <a href={href('ch06', '6.1')}>section 6.1, Global memory access coalescing</a>.
        </p>
      </Translated>
    </>
  );
}
