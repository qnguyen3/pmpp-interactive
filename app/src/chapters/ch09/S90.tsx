import { Ask, type Question } from '../../components/Ask';
import { Beyond, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// The opening bridge (notes/today/ch09-facts.md D5) and the prerequisite check, drawn
// from the notes of chapters 4, 5 and 6.
export const questionsS90: Record<string, Question> = {
  'ch09/start/q-mlx': {
    kind: 'choice',
    id: 'ch09/start/q-mlx',
    prompt: (
      <>
        <pre
          className="mono"
          style={{ margin: '0 0 0.6rem', padding: '0.5rem 0.7rem', background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 3, overflowX: 'auto' }}
        >
          {`import mlx.core as mx
a = mx.array([0, 0])
idx = mx.array([0, 1, 0, 1])
a[idx] += 1`}
        </pre>
        <Translated vi={<>Sau đó <code>a</code> chứa gì?</>}>
          What does <code>a</code> hold afterwards?
        </Translated>
      </>
    ),
    options: [
      { label: '[2, 2]', why: <Translated vi="Đó là kết quả của a.at[idx].add(1). Phép += thông thường không áp dụng mọi lần cập nhật.">That is what a.at[idx].add(1) gives. The plain += does not apply every update.</Translated> },
      { label: '[1, 1]', correct: true },
      { label: <Translated vi="Một lỗi, vì các chỉ số lặp lại.">An error, because the indices repeat.</Translated>, why: <Translated vi="MLX chấp nhận chỉ số lặp lại và có tài liệu mô tả dòng này làm gì.">MLX accepts repeated indices and documents what this line does.</Translated> },
      { label: <Translated vi="Nó thay đổi giữa các lần chạy.">It varies from run to run.</Translated>, why: <Translated vi="Tài liệu của MLX cho thấy một kết quả xác định cho đúng ví dụ này.">MLX's documentation shows one definite result for exactly this example.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Dòng này gồm ba bước: một lần gather, một phép cộng và một lần scatter. <code>a[idx]</code> đọc ra{' '}
              <code>[0, 0, 0, 0]</code>, cộng 1 được <code>[1, 1, 1, 1]</code>, và lần scatter ghi 1 vào <code>a[0]</code> hai
              lần và vào <code>a[1]</code> hai lần. Cả hai lần ghi vào một chỉ số đều mang cùng giá trị cũ cộng một, nên mỗi chỉ
              số mất một lần tăng.
            </p>
            <p>
              MLX ghi rõ đúng kết quả này trong tài liệu và cung cấp <code>a.at[idx].add(1)</code>, phép này áp dụng mọi lần
              cập nhật và cho ra <code>[2, 2]</code>. PyTorch nói cùng điều đó thẳng thắn hơn: với chỉ số trùng lặp và{' '}
              <code>accumulate=False</code>, <code>index_put_</code> là không xác định.
            </p>
          </>
        }
      >
        <>
          <p>
            The line is three steps: a gather, an add and a scatter. <code>a[idx]</code> reads <code>[0, 0, 0, 0]</code>,
            adding 1 gives <code>[1, 1, 1, 1]</code>, and the scatter writes 1 into <code>a[0]</code> twice and into{' '}
            <code>a[1]</code> twice. Both writes to an index carry the same old value plus one, so one increment per index
            is lost.
          </p>
          <p>
            MLX documents exactly this result and offers <code>a.at[idx].add(1)</code>, which applies every update and
            gives <code>[2, 2]</code>. PyTorch says the same thing more bluntly: with duplicate indices and{' '}
            <code>accumulate=False</code>, <code>index_put_</code> is undefined.
          </p>
        </>
      </Translated>
    ),
    source: 'notes/today/ch09-facts.md D5 (MLX docs for array.at; PyTorch 2.15 docs for Tensor.index_put_)',
    vi: { source: 'notes/today/ch09-facts.md D5 (tài liệu MLX cho array.at; tài liệu PyTorch 2.15 cho Tensor.index_put_)' },
  },
  'ch09/start/q-barrier-scope': {
    kind: 'choice',
    id: 'ch09/start/q-barrier-scope',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Mỗi khối của một kernel cộng kết quả của nó vào một mảng trong bộ nhớ toàn cục rồi gọi{' '}
              <code>__syncthreads()</code>. Sau rào chắn của mình, khối 1 có thể tin chắc là thấy được các phép cộng của khối 0
              không?
            </>
          }
        >
          Every block of a kernel adds its results into one array in global memory and then calls{' '}
          <code>__syncthreads()</code>. After its barrier, can block 1 rely on seeing block 0's additions?
        </Translated>
      </>
    ),
    options: [
      { label: <Translated vi="Có: rào chắn chờ mọi luồng chạm vào mảng, dù luồng đó thuộc khối nào.">Yes: the barrier waits for every thread that touches the array, whichever block it is in.</Translated>, why: <Translated vi="Một rào chắn chờ các luồng của chính khối nó, và chỉ những luồng đó.">A barrier waits for the threads of its own block, and only those.</Translated> },
      { label: <Translated vi="Có, vì khối 0 có chỉ số nhỏ hơn, và các khối bắt đầu và kết thúc theo thứ tự chỉ số.">Yes, because block 0 has the lower index, and blocks start and finish in index order.</Translated>, why: <Translated vi="Không có thứ tự nào giữa các khối được hứa hẹn. Khối 1 có thể xong trước khi khối 0 bắt đầu.">No order between blocks is promised. Block 1 may finish before block 0 starts.</Translated> },
      { label: <Translated vi="Không: rào chắn chỉ bao các luồng của một khối, và các khối chạy theo thứ tự bất kỳ, không phải tất cả cùng lúc.">No: the barrier covers one block's threads, and blocks run in any order, not all at once.</Translated>, correct: true },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Rào chắn là điểm hẹn của các luồng trong một khối (<a href={href('ch04', '4.3')}>mục 4.3</a>). Runtime giao các
            khối cho phần cứng theo một thứ tự không được hứa hẹn, và một lưới lớn không bao giờ nằm hết trên thiết bị cùng lúc.
            Vì vậy không thể dùng rào chắn để sắp thứ tự kết quả từ các khối khác nhau. Chương này gộp chúng bằng phép toán
            nguyên tử (atomic), vốn không cần thứ tự nào cả.
          </p>
        }
      >
        <p>
          A barrier is a meeting point for the threads of one block (<a href={href('ch04', '4.3')}>section 4.3</a>). The
          runtime assigns blocks to the hardware in no promised order, and a large grid is never all on the device at once.
          So results from different blocks cannot be put in order with barriers. This chapter merges them with atomics,
          which need no order at all.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.3, pp. 70–74',
  },
  'ch09/start/q-shared-init': {
    kind: 'choice',
    id: 'ch09/start/q-shared-init',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Một kernel khai báo <code>__shared__ unsigned int counts[256];</code> và các luồng của nó lập tức bắt đầu tăng các
              phần tử. Các giá trị đếm bắt đầu từ đâu?
            </>
          }
        >
          A kernel declares <code>__shared__ unsigned int counts[256];</code> and its threads immediately start
          incrementing entries. What did the counts start at?
        </Translated>
      </>
    ),
    options: [
      { label: <Translated vi="Không: mảng chia sẻ được xóa về 0 khi một khối bắt đầu.">Zero: shared arrays are cleared when a block starts.</Translated>, why: <Translated vi="Không có gì xóa chúng. Trình giả lập của khóa học này điền bộ nhớ chia sẻ bằng các byte 0xCD để lỗi này lần nào cũng lộ ra.">Nothing clears them. The course's emulator fills shared memory with 0xCD bytes so that this mistake shows every time.</Translated> },
      { label: <Translated vi="Bất cứ thứ gì vùng nhớ đó đang chứa: không ai khởi tạo bản sao của khối này.">Whatever that memory held: nobody initialised this block's copy.</Translated>, correct: true },
      { label: <Translated vi="Các giá trị đếm mà khối trước đó trên cùng SM để lại.">The counts the previous block on the same SM left behind.</Translated>, why: <Translated vi="Có thể, có thể không. Không gì hứa hẹn một nội dung cụ thể nào, nên hãy coi nó là không biết.">Maybe, maybe not. Nothing promises any particular content, so treat it as unknown.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi khối có bản sao riêng của một biến chia sẻ, tồn tại trong lúc kernel chạy, và không có giá trị khởi đầu (
            <a href={href('ch05', '5.2')}>mục 5.2</a>). Vì vậy các histogram riêng của mục 9.4 xóa bản sao của chúng trước
            tiên, với cả khối cùng làm, và đặt một rào chắn sau bước xóa: thiếu nó, một luồng nhanh có thể đếm vào một phần tử
            mà một luồng chậm chưa xóa xong.
          </p>
        }
      >
        <p>
          Each block gets its own copy of a shared variable, alive while the kernel runs, and no initial value (
          <a href={href('ch05', '5.2')}>section 5.2</a>). The private histograms of section 9.4 therefore clear their
          copy first, with the whole block helping, and put a barrier after the clearing: without it a fast thread could
          count into an entry that a slow thread has not cleared yet.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.2, pp. 98–104; ch. 9 Fig. 9.10, p. 214; the emulator\'s 0xCD fill: labs/README.md',
    vi: { source: 'ch. 5 §5.2, pp. 98–104; ch. 9 Fig. 9.10, p. 214; việc trình giả lập điền 0xCD: labs/README.md' },
  },
  'ch09/start/q-latency': {
    kind: 'choice',
    id: 'ch09/start/q-latency',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Một lần truy cập DRAM mất hàng trăm chu kỳ xung nhịp, vậy mà một kernel cộng vector vẫn có thể chạy gần băng thông
              đỉnh của bộ nhớ. Vì sao?
            </>
          }
        >
          A DRAM access takes hundreds of clock cycles, yet a vector-addition kernel can run close to the memory's peak
          bandwidth. Why?
        </Translated>
      </>
    ),
    options: [
      { label: <Translated vi="Bộ nhớ đệm che độ trễ của mọi lần truy cập sau lần đầu, nên chỉ một lần nạp phải trả nó.">Caches hide the latency of every access after the first, so only one load pays it.</Translated>, why: <Translated vi="Phép cộng vector đọc mỗi phần tử một lần. Không có gì để bộ nhớ đệm dùng lại.">Vector addition reads each element once. There is nothing for a cache to reuse.</Translated> },
      { label: <Translated vi="Nhiều warp giữ các yêu cầu độc lập cùng đang bay, nên độ trễ của chúng chồng lên nhau.">Many warps keep independent requests in flight, so their latencies overlap.</Translated>, correct: true },
      { label: <Translated vi="Trình biên dịch sắp xếp lại các lần nạp để mỗi lần được phát sớm và tốn ít thời gian hơn.">The compiler reorders the loads so that each one is issued early and takes less time.</Translated>, why: <Translated vi="Sắp xếp lại không làm một lần truy cập ngắn đi. Điều giúp ích là chồng nhiều lần truy cập lên nhau.">Reordering does not shorten an access. Overlapping many of them is what helps.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong lúc một warp chờ dữ liệu, SM chạy một warp khác (<a href={href('ch04', '4.6')}>mục 4.6</a>), và các kênh
            và bank của DRAM xử lý nhiều yêu cầu cùng lúc (<a href={href('ch06', '6.2')}>mục 6.2</a>). Điều đó chỉ hiệu quả
            khi các yêu cầu độc lập với nhau. Mục 9.3 nói về trường hợp chúng không độc lập: các lần cập nhật vào một vị trí
            phải lần lượt chờ đến phiên.
          </p>
        }
      >
        <p>
          While one warp waits for its data, the SM runs another (<a href={href('ch04', '4.6')}>section 4.6</a>), and the
          DRAM's channels and banks work on many requests at the same time (<a href={href('ch06', '6.2')}>section 6.2</a>).
          That only works when the requests are independent. Section 9.3 is about the case where they are not: updates to
          one location have to take turns.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.6, pp. 82–85; ch. 6 §6.2, pp. 132–137', // i18n-ignore: citation
  },
  'ch09/start/q-coalesce': {
    kind: 'choice',
    id: 'ch09/start/q-coalesce',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Mỗi luồng đọc một byte của <code>image</code> ở mỗi bước <code>k</code> của một vòng lặp. Với chỉ số nào thì các
              lần nạp của một warp trong một bước rơi vào các byte liên tiếp?
            </>
          }
        >
          Each thread reads one byte of <code>image</code> in every step <code>k</code> of a loop. With which index do the
          loads of a warp in one step fall on consecutive bytes?
        </Translated>
      </>
    ),
    options: [
      { label: <code>image[(blockIdx.x*blockDim.x + threadIdx.x) + k*blockDim.x*gridDim.x]</code>, correct: true },
      {
        label: <code>image[(blockIdx.x*blockDim.x + threadIdx.x)*K + k]</code>,
        why: (
          <Translated vi="Ở đây các luồng liền kề cách nhau K byte ở mỗi bước, nên các lần nạp của warp trải trên vùng nhớ lớn gấp K lần.">
            Here neighbouring threads are K bytes apart in every step, so the warp's loads are spread over K times as much memory.
          </Translated>
        ),
      },
      { label: <Translated vi="Cả hai: phần cứng gộp mọi lần nạp mà một warp thực hiện trong cùng một bước.">Both: the hardware merges any loads a warp makes in the same step.</Translated>, why: <Translated vi="Nó gộp các lần nạp rơi vào cùng những mảnh bộ nhớ. Các lần nạp trải rộng cần nhiều mảnh hơn.">It merges loads that fall in the same pieces of memory. Spread-out loads need more pieces.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Ở biểu thức thứ nhất, luồng <i>t</i> + 1 đọc byte ngay sau luồng <i>t</i> ở mỗi bước, nên các lần nạp của warp
            được hợp nhất thành ít giao dịch bộ nhớ (<a href={href('ch06', '6.1')}>mục 6.1</a>). Biểu thức thứ hai cho mỗi
            luồng một dải <code>K</code> byte liên tiếp của riêng nó, một bố cục tốt cho luồng CPU và tệ cho một warp. Mục 9.5
            so sánh đúng hai biểu thức chỉ số này.
          </p>
        }
      >
        <p>
          In the first, thread <i>t</i> + 1 reads the byte after thread <i>t</i> in every step, so the warp's loads merge
          into few memory transactions (<a href={href('ch06', '6.1')}>section 6.1</a>). The second gives each thread its
          own contiguous run of <code>K</code> bytes, a good layout for a CPU thread and a bad one for a warp. Section 9.5
          compares exactly these two index expressions.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.1, pp. 124–132',
  },
  'ch09/start/q-coarsen': {
    kind: 'choice',
    id: 'ch09/start/q-coarsen',
    prompt: (
      <Translated vi={<>Khi nào việc giao cho mỗi luồng nhiều phần tử (gộp việc cho luồng, thread coarsening) làm kernel nhanh hơn?</>}>
        <>When does giving each thread several elements (thread coarsening) make a kernel faster?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Ít luồng hơn nghĩa là phần cứng ít việc lập lịch hơn, nên nó giúp ích bất kể kernel làm gì.">Fewer threads means less scheduling work for the hardware, so it helps whatever the kernel does.</Translated>, why: <Translated vi="Với các luồng hoàn toàn độc lập và không có chi phí như vậy thì chẳng được bao nhiêu, và quá ít khối để một phần thiết bị ngồi không.">With fully independent threads and no such cost there is little to gain, and too few blocks leave part of the device idle.</Translated> },
      { label: <Translated vi="Khi một chi phí theo từng khối bị trả cho những khối mà đằng nào phần cứng cũng chạy lần lượt.">When a per-block cost is paid for blocks the hardware would have run one after another anyway.</Translated>, correct: true },
      { label: <Translated vi="Khi kernel bị giới hạn bởi tính toán, vì gộp việc làm tăng lượng phép tính mỗi luồng thực hiện.">When the kernel is compute-bound, since coarsening raises the arithmetic each thread does.</Translated>, why: <Translated vi="Phép thử là việc song song hóa có tốn một chi phí mà gộp việc loại bỏ được hay không, chứ không phải cường độ tính toán của kernel.">The test is whether parallelising costs something that coarsening removes, not the kernel's arithmetic intensity.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Gộp việc chỉ có lợi khi có một chi phí phụ để cắt bỏ (<a href={href('ch06', '6.5')}>mục 6.5</a>). Mục 9.5 tìm
            thấy một chi phí như vậy: mỗi histogram riêng của mục 9.4 phải được xóa và gộp một lần cho mỗi khối, nên ít khối
            hơn, mỗi khối bận hơn, thì làm ít việc đó hơn.
          </p>
        }
      >
        <p>
          Coarsening pays only when there is an overhead to cut (<a href={href('ch06', '6.5')}>section 6.5</a>). Section
          9.5 finds one: each private histogram of section 9.4 has to be cleared and merged once per block, so fewer,
          busier blocks do less of that work.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.5, pp. 141–143',
  },
};

const ROADMAP: { sec: string; title: string; does: string }[] = [
  {
    sec: '9.2',
    get title() {
      return tr('Atomic operations', 'Phép toán nguyên tử');
    },
    get does() {
      return tr('the correct baseline: every update to a bin made indivisible', 'mốc đúng đắn: mọi lần cập nhật vào một ô được làm thành không thể chia cắt');
    },
  },
  {
    sec: '9.3',
    get title() {
      return tr('Latency and throughput of atomics', 'Độ trễ và thông lượng của atomic');
    },
    get does() {
      return tr('why that baseline crawls when many threads hit one bin', 'vì sao mốc đó bò chậm khi nhiều luồng cùng nhắm một ô');
    },
  },
  {
    sec: '9.4',
    get title() {
      return tr('Privatisation', 'Riêng tư hóa');
    },
    get does() {
      return tr('a private copy of the bins per block, merged at the end', 'mỗi khối một bản sao riêng của các ô, gộp lại ở cuối');
    },
  },
  {
    sec: '9.5',
    get title() {
      return tr('Thread coarsening', 'Gộp việc cho luồng');
    },
    get does() {
      return tr('fewer blocks, so fewer copies to clear and merge', 'ít khối hơn, nên ít bản sao phải xóa và gộp hơn');
    },
  },
  {
    sec: '9.6',
    get title() {
      return tr('Thread-level privatisation', 'Riêng tư hóa ở mức luồng');
    },
    get does() {
      return tr('a private count in each thread for a run of equal values', 'một bộ đếm riêng trong mỗi luồng cho một dãy giá trị bằng nhau');
    },
  },
];

export function S90() {
  const q = questionsS90;
  return (
    <>
      <Translated
        vi={
          <p>
            Hãy bắt đầu với một dòng Python mà có thể tuần trước bạn đã viết. Trả lời trước khi đọc tiếp.
          </p>
        }
      >
        <p>
          Start with a line of Python you could have written last week. Answer before reading on.
        </p>
      </Translated>
      <Ask q={q['ch09/start/q-mlx']} label={tr('Predict', 'Dự đoán')} />
      <Translated
        vi={
          <p>
            Dòng này trông như một phép toán nhưng thực ra là ba: đọc các phần tử được chọn, cộng một, ghi chúng trở lại. Khi
            một chỉ số xuất hiện hai lần, hai lần ghi mang cùng một giá trị cũ và một lần tăng biến mất. Không có gì thất bại;
            thư viện đã làm đúng điều câu lệnh nói. Một kernel GPU trong đó nhiều luồng cập nhật một mảng dùng chung có cùng
            hình dạng đó, ở quy mô lớn hơn nhiều và không có thư viện nào chốt lại kết quả. Chương này là lỗi đó trên GPU, và
            cách chữa nó.
          </p>
        }
      >
        <p>
          The line looks like one operation and is three: read the selected entries, add one, write them back. When an
          index appears twice, two writes carry the same stale value and one increment disappears. Nothing failed; the
          library did what the statement says. A GPU kernel in which many threads update a shared array has the same
          shape, at a much larger scale and without a library to pin down the result. This chapter is that bug on the
          GPU, and its cure.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch09-facts.md, D5)', '2026-11-20 (notes/today/ch09-facts.md, mục D5)')}>
        <Translated
          vi={
            <p>
              Tài liệu của MLX cho <code>array.at</code> dùng đúng ví dụ này. mlx-lm cố ý dùng cả hai dạng: một phép trừ thông
              thường cho presence penalty, nơi các bản trùng nên gộp làm một, và <code>.at[...].subtract</code> cho frequency
              penalty, nơi mọi lần xuất hiện đều được tính.
            </p>
          }
        >
          <p>
            MLX's documentation for <code>array.at</code> uses this exact example. mlx-lm uses both forms on purpose: a
            plain subtraction for the presence penalty, where duplicates should collapse, and{' '}
            <code>.at[...].subtract</code> for the frequency penalty, where every occurrence counts.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Điều gì mới trong chương này">What is new in this chapter</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mọi kernel cho đến giờ đều tuân theo một quy tắc: mỗi phần tử đầu ra thuộc về một luồng, luồng đó tính và ghi nó.
            Sách gọi đây là owner-computes (người sở hữu tính). Các luồng không bao giờ ghi vào cùng một chỗ, nên chúng không
            bao giờ phải phối hợp. Histogram phá vỡ quy tắc đó. Luồng cập nhật ô nào phụ thuộc vào điểm ảnh nó đọc, nên luồng
            nào cũng có thể cập nhật ô nào, và nhiều luồng có thể cập nhật cùng một ô vào cùng một lúc. Sách gọi kết quả đó là
            output interference (giao thoa đầu ra).
          </p>
        }
      >
        <p>
          Every kernel so far obeyed one rule: each output element belongs to one thread, which computes it and writes
          it. The book calls this owner-computes. Threads never wrote to the same place, so they never had to coordinate.
          A histogram breaks the rule. Which bin a thread updates depends on the pixel it reads, so any thread may update
          any bin, and several may update the same one at the same moment. The book calls the result output interference.
        </p>
      </Translated>
      <Source pages="201" />
      <Translated vi={<p>Chương này xây một kernel đúng trước rồi làm cho nó nhanh, qua năm bước:</p>}>
        <p>The chapter builds a correct kernel first and then makes it fast, in five steps:</p>
      </Translated>
      <table className="data" data-part="roadmap">
        <thead>
          <tr>
            <th>
              <Translated vi="Mục">Section</Translated>
            </th>
            <th>
              <Translated vi="Nó thêm gì">What it adds</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          {ROADMAP.map((r) => (
            <tr key={r.sec}>
              <td>
                <a href={href('ch09', r.sec)}>
                  {r.sec} {r.title}
                </a>
              </td>
              <td>{r.does}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Sách nhấn mạnh rằng mỗi tối ưu có đáng hay không phụ thuộc vào phần cứng và vào dữ liệu đầu vào. Một ô mà mọi
            luồng đều muốn hành xử rất khác với các ô được nhắm đều nhau. Đó là lý do mọi lab trong chương này đo hai loại đầu
            vào: các byte phân bố đều và một ảnh bị lệch.
          </p>
        }
      >
        <p>
          The book insists that whether each optimisation pays off depends on the hardware and on the input data. A bin
          that every thread wants behaves very differently from bins that are hit evenly. That is why every lab in this
          chapter measures two kinds of input: uniform bytes and a skewed image.
        </p>
      </Translated>
      <Source pages="201" />

      <h2>
        <Translated vi="Điều này nằm ở đâu trong suy luận LLM">Where this sits in LLM inference</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Phép nhân ma trận và attention là owner-computes: mỗi phần tử đầu ra có một người ghi. Các lần ghi phụ thuộc dữ
            liệu xuất hiện quanh chúng. Repetition penalty cần đếm mỗi token id đã xuất hiện bao nhiêu lần, và đếm token id là
            một histogram có các ô là bộ từ vựng (mốc dự án M5, <a href={href('ch09', 'labs')}>lab 9.3</a>). Phép chọn top-k
            theo bucket đếm các logits vào các bucket trước. Và mọi scatter-add, một đầu ra được đánh chỉ số bằng dữ liệu, đều
            có cùng race condition và cùng cách chữa.
          </p>
        }
      >
        <p>
          Matrix multiplication and attention are owner-computes: every output element has one writer. Data-dependent
          writes appear around them. Repetition penalties need a count of how often each token id has appeared, and
          counting token ids is a histogram whose bins are the vocabulary (milestone M5, <a href={href('ch09', 'labs')}>lab 9.3</a>).
          Bucketed top-k selection counts the logits into buckets first. And any scatter-add, an output indexed by data,
          has the same race and takes the same cure.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không liên hệ gì với mô hình ngôn ngữ trong chương này. Cách đối chiếu là của chúng tôi (notes/ch09.md). Các
              mục và trang tóm tắt nêu tên mã thật đang làm những việc này ngày nay (notes/today/ch09-facts.md D1 đến D6).
            </p>
          }
        >
          <p>
            The book makes no link to language models in this chapter. The mapping is ours (notes/ch09.md). The sections
            and the summary page name the real code that does these things today (notes/today/ch09-facts.md D1 to D6).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Điều mang sang được nhiều nhất không phải bản thân histogram. Đó là khả năng ước lượng các lần cập nhật nguyên tử
            có thể nhanh đến đâu, và hình dạng riêng-tư-hóa-rồi-ghi-về của các kernel nhanh. Chương 10 dùng lại cả hai: phép
            rút gọn nhiều khối của nó để mỗi khối hoàn tất riêng rồi ghi về bằng một atomic (Fig. 10.18, pp. 244–245).
          </p>
        }
      >
        <p>
          What transfers most is not the histogram itself. It is the ability to estimate how fast atomic updates can go,
          and the privatise-then-commit shape of the fast kernels. Chapter 10 reuses both: its multi-block reduction lets
          each block finish privately and commit with one atomic (Fig. 10.18, pp. 244–245).
        </p>
      </Translated>

      <h2>
        <Translated vi="Kiểm tra những gì chương này giả định">Check what this chapter assumes</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Năm câu hỏi về những gì chương này giả định từ chương 4 đến 6. Mỗi lời giải thích dẫn ngược về nơi điều đó được
            dạy, và mỗi câu sẽ quay lại trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Five questions on what this chapter assumes from chapters 4 to 6. Each explanation links back to where it was
          taught, and each comes back in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={q['ch09/start/q-barrier-scope']} label={tr('From chapter 4', 'Từ chương 4')} />
      <Ask q={q['ch09/start/q-shared-init']} label={tr('From chapter 5', 'Từ chương 5')} />
      <Ask q={q['ch09/start/q-latency']} label={tr('From chapters 4 and 6', 'Từ chương 4 và 6')} />
      <Ask q={q['ch09/start/q-coalesce']} label={tr('From chapter 6', 'Từ chương 6')} />
      <Ask q={q['ch09/start/q-coarsen']} label={tr('From chapter 6', 'Từ chương 6')} />

      <h2>
        <Translated vi="Nhân chứng chính của bạn: backend race">Your main witness: the race backend</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lab 4.3 đã giới thiệu biến thể thứ hai của trình giả lập, <code>race</code>. Trong chương này nó trở thành nhân
            chứng chính. Trình giả lập thông thường, <code>emu</code>, chạy từng luồng một và chỉ chuyển luồng tại rào chắn,
            nên các bước đọc, cộng và ghi của một luồng không bao giờ bị tách ra: một atomic bị thiếu lần nào cũng qua được ở
            đó. Backend <code>race</code> chạy các luồng đồng thời dưới ThreadSanitizer và chỉ ra cả hai dòng mã nguồn của mọi
            cặp truy cập không đồng bộ, ngay cả trong một lần chạy tình cờ cho ra số đếm đúng. Hãy dùng{' '}
            <code>python3 labs/pmpp.py run 9.1 --backend race</code> trước khi tin một kernel histogram.
          </p>
        }
      >
        <p>
          Lab 4.3 introduced the emulator's second flavour, <code>race</code>. In this chapter it becomes the main witness.
          The ordinary emulator, <code>emu</code>, runs one thread at a time and switches only at barriers, so a thread's
          read, add and write are never split: a missing atomic passes there every time. The <code>race</code> backend runs
          the threads concurrently under ThreadSanitizer and names both source lines of every unsynchronised pair of
          accesses, even in a run that happened to produce the right counts. Use{' '}
          <code>python3 labs/pmpp.py run 9.1 --backend race</code> before you trust a histogram kernel.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cả hai trình giả lập đều là công cụ của khóa học này, không phải của sách. Mỗi cái thấy được gì và không thấy được
              gì được liệt kê trong labs/README.md.
            </p>
          }
        >
          <p>
            Both emulators are this course's tools, not the book's. What each one can and cannot see is listed in
            labs/README.md.
          </p>
        </Translated>
      </Beyond>
    </>
  );
}
