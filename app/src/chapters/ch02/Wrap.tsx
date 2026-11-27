import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { ceilDiv, launch } from '../../sim/launch';

// Two kernels of our own, each shown only for a question. The checked copies are
// labs/ch02/snippets/saxpy_stub_defect.cu and labs/ch02/snippets/count_above_race.cu
// (Wrap.snippets.test.ts fails if a copy here and its file drift apart).
const SAXPY = `// y = a*x + y for n floats, on the device. It has one defect, on purpose.
// No error checks, to keep the listing short, as in the book's figures.
__global__ void saxpyKernel(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

void saxpy(float a, const float* x_h, float* y_h, int n) {
    float *x_d, *y_d;
    size_t size = (size_t)n * sizeof(float);
    cudaMalloc((void**)&x_d, size);
    cudaMalloc((void**)&y_d, size);
    cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);

    saxpyKernel<<<(n + 255) / 256, 256>>>(a, x_d, y_d, n);

    cudaMemcpy(y_h, y_d, size, cudaMemcpyDeviceToHost);
    cudaFree(x_d);
    cudaFree(y_d);
}`;

const COUNT = `// Count the probabilities above a threshold t. *count is 0 before the launch.
__global__ void countAbove(const float* p, float t, int* count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && p[i] > t) {
        *count = *count + 1;
    }
}`;

// The numbers the questions ask for.
/** One token's logits through a one-thread-per-element kernel in blocks of 256 (the vocabulary size is our example). */
const VOCAB = 50257;
const LOGITS = launch(VOCAB, 256, ceilDiv(VOCAB, 256));
/** An embedding table of 150,000 tokens by 4,096 dimensions, in float. */
const TABLE = { rows: 150000, cols: 4096 };
const TABLE_N = TABLE.rows * TABLE.cols;
const TABLE_BYTES = TABLE_N * 4;
const INT_MAX = 2 ** 31 - 1;
/** What a 32-bit int holds after the conversion: C++20 defines it modulo 2^32. */
const WRAPPED = TABLE_BYTES > INT_MAX ? TABLE_BYTES - 2 ** 32 : TABLE_BYTES;

const n = (v: number) => v.toLocaleString('en-US');

export const questionsWrap: Record<string, Question> = {
  'ch02/wrap/q-inout': {
    kind: 'choice',
    id: 'ch02/wrap/q-inout',
    prompt: (
      <Translated
        vi={
          <>
            <code>saxpy</code> ở trên được gọi trên một GPU CUDA với mọi <code>x_h[i]</code> bằng 1, mọi{' '}<code>y_h[i]</code> bằng 2, và <code>a = 3</code>. Sau đó <code>y_h</code> chứa gì?
          </>
        }
      >
        <code>saxpy</code> above is called on a CUDA GPU with every <code>x_h[i]</code> equal to 1, every{' '}
        <code>y_h[i]</code> equal to 2, and <code>a = 3</code>. What does <code>y_h</code> hold afterwards?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="5 ở mọi phần tử. Mọi lời gọi đều thành công và mọi phần tử đều có luồng của nó.">
            5 everywhere. Every call succeeds and every element has its thread.
          </Translated>
        ),
        why: (
          <Translated vi="Lần khởi chạy và phép chặn đều đúng. Nhưng kernel đọc y[i] trước khi ghi nó, và không gì đưa y của bên gọi lên device.">
            The launch and the guard are right. But the kernel reads y[i] before it writes it, and nothing put the caller's y on the device.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="3 cộng với bất cứ thứ gì vùng cấp phát của y_d tình cờ chứa, ở mọi phần tử.">
            3 plus whatever the allocation of y_d happened to hold, in every element.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Đúng 3 ở mọi phần tử, vì cudaMalloc cấp phát bộ nhớ đã được xóa về 0.">
            Exactly 3 everywhere, because cudaMalloc hands out zeroed memory.
          </Translated>
        ),
        why: (
          <Translated vi="NVIDIA ghi điều ngược lại: “The memory is not cleared.” Số 0 là thứ newBufferWithLength của Metal cho ra, và là thứ mẫu điền của trình giả lập (emulator) gần giống.">
            NVIDIA documents the opposite: "The memory is not cleared." Zero is what Metal's newBufferWithLength gives, and what the emulator's fill pattern comes close to.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không gì cả: lời gọi thất bại, vì kernel đọc bộ nhớ mà không ai ghi.">
            Nothing: the call fails, because the kernel reads memory that nobody wrote.
          </Translated>
        ),
        why: (
          <Translated vi="Không lời gọi API nào báo cáo việc đọc bộ nhớ device chưa khởi tạo. Mọi lời gọi ở đây đều trả về thành công.">
            No API call reports a read of uninitialised device memory. Every call here returns success.
          </Translated>
        ),
      },
    ],
    explain: (
      <>
        <Translated
          vi={
            <p>
              Hàm stub có hình dạng của hàm stub cộng vectơ, nơi C chỉ được ghi và vì thế không bao giờ được sao chép vào (<a href={href('ch02', '2.3')}>mục 2.3</a>). Ở đây <code>y</code> vừa được đọc vừa được ghi, nên nó cần lần sao chép vào mà C không cần: <code>cudaMemcpy(y_d, y_h, size, cudaMemcpyHostToDevice)</code> trước lần khởi chạy. Không có nó, kernel cộng <code>a * x[i]</code> vào bất cứ thứ gì <code>y_d</code> đang chứa, thứ mà CUDA không xóa.
            </p>
          }
        >
          <p>
            The stub has the shape of the vector addition stub, where C is only written and so is never copied in (
            <a href={href('ch02', '2.3')}>section 2.3</a>). Here <code>y</code> is read as well as written, so it needs
            the copy in that C did not: <code>cudaMemcpy(y_d, y_h, size, cudaMemcpyHostToDevice)</code> before the
            launch. Without it the kernel adds <code>a * x[i]</code> to whatever <code>y_d</code> held, which CUDA does
            not clear.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Phần nguy hiểm là đáp án sai có thể trông như đúng. Trên trình giả lập của khóa học, nơi bộ nhớ device mới cấp phát đọc ra như một số float khoảng −2.9 × 10⁻¹⁶, một lần chạy riêng lẻ tệp này cho đúng 3 ở cả 1000 phần tử (emulator-checked; không có GPU nào tham gia). Trên Metal, một buffer tạo bằng{' '}<code>newBufferWithLength</code> bắt đầu từ 0 và cho kết quả tương tự. Một bài kiểm thử có <code>y</code> bắt đầu từ 0 sẽ qua. Điều tương tự đúng với mọi phép toán làm tại chỗ, trong đó có <code>addResidual</code> của lab 2.3: một toán hạng được cập nhật cũng là một đầu vào.
            </p>
          }
        >
          <p>
            The dangerous part is that the wrong answer can look right. On the course's emulator, whose fresh device
            memory reads as a float of about −2.9 × 10⁻¹⁶, a one-off run of this file gave exactly 3 in all 1000
            elements (emulator-checked; no GPU was involved). On Metal a buffer made with{' '}
            <code>newBufferWithLength</code> starts at zero and gives the same. A test whose <code>y</code> starts at 0
            passes. The same holds for any operation done in place, lab 2.3's <code>addResidual</code> among them: an
            operand that is updated is an input too.
          </p>
        </Translated>
      </>
    ),
    source: 'our kernel; ch. 2 §2.3, p. 27 and §2.4, pp. 30–31; cudaMalloc: notes/today/ch02-facts.md, A2; Metal: C2',
    vi: { source: 'kernel của khóa học này; ch. 2 §2.3, p. 27 và §2.4, pp. 30–31; cudaMalloc: notes/today/ch02-facts.md, A2; Metal: C2' },
  },
  'ch02/wrap/q-logits': {
    kind: 'numeric',
    id: 'ch02/wrap/q-logits',
    prompt: (
      <Translated
        vi={
          <>
            Một bộ lấy mẫu chia mỗi logit trong {n(VOCAB)} logit của một token cho một nhiệt độ (temperature), một luồng cho mỗi logit, trong các khối 256 luồng và số khối ít nhất đủ phủ chúng. Bao nhiêu luồng được khởi chạy dừng lại ở phép chặn?
          </>
        }
      >
        A sampler divides each of the {n(VOCAB)} logits of one token by a temperature, one thread per logit, in
        blocks of 256 threads and the fewest blocks that cover them. How many of the launched threads stop at the
        guard?
      </Translated>
    ),
    answer: LOGITS.surplus,
    unit: 'threads',
    explain: (
      <Translated
        vi={
          <p>
            ({n(VOCAB)} + 255) / 256 = {LOGITS.blocks} khối theo phép chia nguyên, nên có {n(LOGITS.threads)} luồng.{' '}{n(LOGITS.working)} luồng trong số đó có một logit và {LOGITS.surplus} luồng dừng ở <code>if (i &lt; n)</code>, tất cả trong khối cuối. Kích thước bộ từ vựng là ví dụ của khóa học này; mọi kích thước không phải bội của 256 đều để lại một khối cuối chỉ đầy một phần, đó là lý do mọi kernel loại này đều mang phép chặn. Trên Metal,{' '}<code>dispatchThreads</code> với một lưới đúng {n(VOCAB)} luồng không khởi chạy luồng thừa nào.
          </p>
        }
      >
        <p>
          ({n(VOCAB)} + 255) / 256 = {LOGITS.blocks} blocks in integer division, so {n(LOGITS.threads)} threads.{' '}
          {n(LOGITS.working)} of them have a logit and {LOGITS.surplus} stop at <code>if (i &lt; n)</code>, all in the
          last block. The vocabulary size is our example; any size that is not a multiple of 256 leaves a partly
          filled last block, which is why every kernel of this kind carries the guard. On Metal,{' '}
          <code>dispatchThreads</code> with a grid of exactly {n(VOCAB)} threads launches none spare.
        </p>
      </Translated>
    ),
    source: 'method of ch. 2 §2.5, p. 37 and §2.6, p. 38 (the book\'s n = 1000 case); the sampler is ours',
    vi: {
      unit: 'luồng',
      source: 'phương pháp của ch. 2 §2.5, p. 37 và §2.6, p. 38 (trường hợp n = 1000 của sách); bộ lấy mẫu là của khóa học này',
    },
  },
  'ch02/wrap/q-count': {
    kind: 'choice',
    id: 'ch02/wrap/q-count',
    prompt: (
      <Translated
        vi={
          <>
            <code>countAbove</code> ở trên được khởi chạy với 4 khối 256 luồng trên 1000 xác suất, trong đó 300 xác suất lớn hơn <code>t</code>. Bạn có thể kỳ vọng gì ở <code>*count</code> sau khi kernel đã xong?
          </>
        }
      >
        <code>countAbove</code> above is launched with 4 blocks of 256 threads over 1000 probabilities, of which 300
        are above <code>t</code>. What can you expect of <code>*count</code> after the kernel has finished?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nó là 300: mọi luồng qua phép kiểm tra đều cộng một.">
            It is 300: every thread that passes the test adds one.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi lần cập nhật là một lần đọc, một phép cộng và một lần ghi. Hai luồng có thể đọc cùng một giá trị cũ và cả hai ghi giá trị đó cộng một, và một trong hai phép cộng bị mất.">
            Each update is a read, an addition and a write. Two threads can read the same old value and both write that value plus one, and one of the additions is lost.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó là 300 nếu cả 300 phần tử rơi vào một khối, vì một khối chạy các luồng của nó theo thứ tự.">
            It is 300 if all 300 elements fall in one block, because a block runs its threads in order.
          </Translated>
        ),
        why: (
          <Translated vi="Cũng không gì sắp thứ tự các luồng của một khối. Một khối dùng chung một cấu hình khởi chạy, không dùng chung một lịch chạy.">
            Nothing orders the threads of a block either. A block shares a launch configuration, not a schedule.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó có thể nhỏ hơn 300, và có thể thay đổi giữa các lần chạy.">
            It can be less than 300, and it can change from run to run.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nó là 0: mỗi luồng cộng một vào bản sao count riêng của nó.">
            It is 0: each thread adds one to its own private copy of count.
          </Translated>
        ),
        why: (
          <Translated vi="Con trỏ count là riêng của mỗi luồng, giống i. Số int mà nó trỏ tới là một vị trí trong bộ nhớ toàn cục (global memory), dùng chung bởi mọi luồng.">
            The pointer count is private to each thread, like i. The int it points to is one location in global memory, shared by every thread.
          </Translated>
        ),
      },
    ],
    explain: (
      <>
        <Translated
          vi={
            <p>
              Phép kiểm tra của chương cho một luồng mỗi đầu ra thất bại ở đây: chỉ có một đầu ra, và 300 luồng tính nó từ những gì các luồng khác đã ghi. Không có thứ tự nào giữa các luồng được hứa hẹn, trong một khối hay giữa các khối, nên các cập nhật có thể bị mất. Cách chữa là phép toán nguyên tử (atomic) (chương 9) và phép rút gọn (reduction) (chương 10), cũng là cách tính tổng đằng sau một softmax.
            </p>
          }
        >
          <p>
            The chapter's test for one thread per output fails here: there is one output, and 300 threads compute it
            from what the others wrote. No order between the threads is promised, inside a block or between blocks, so
            updates can be lost. The cures are atomic operations (chapter 9) and a reduction (chapter 10), which is also
            how the sum behind a softmax is computed.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Trình giả lập của khóa học cho thấy vì sao bạn không thể thoát khỏi chuyện này bằng kiểm thử. Trong các lần chạy riêng lẻ tệp này khi trang này được viết, biến thể <code>emu</code>, chạy mỗi lần một luồng và không bao giờ ngắt một lần cập nhật, lần nào cũng cho 300. Biến thể <code>race</code>, nơi các luồng chạy cùng lúc, cho 188, 191 và 202 trong ba lần chạy, và ThreadSanitizer chỉ ra dòng cập nhật. Cả hai đều không phải GPU.
            </p>
          }
        >
          <p>
            The course's emulator shows why you cannot test your way out of this. In one-off runs of this file made
            when this page was written, the <code>emu</code> flavour, which runs one thread at a time and never
            interrupts an update, gave 300 every time. The <code>race</code> flavour, whose threads run at once, gave
            188, 191 and 202 in three runs, and ThreadSanitizer named the line of the update. Neither is a GPU.
          </p>
        </Translated>
      </>
    ),
    source: 'our kernel; criterion of ch. 2 §2.1, pp. 21–23; no order between blocks, §2.6, p. 39, nor between the threads of a block, ch. 4 §4.4, p. 74; private variables, §2.5, p. 37',
    vi: {
      source:
        'kernel của khóa học này; tiêu chí của ch. 2 §2.1, pp. 21–23; không có thứ tự giữa các khối, §2.6, p. 39, cũng không giữa các luồng của một khối, ch. 4 §4.4, p. 74; biến riêng, §2.5, p. 37',
    },
  },
  'ch02/wrap/q-bytes-int': {
    kind: 'choice',
    id: 'ch02/wrap/q-bytes-int',
    prompt: (
      <Translated
        vi={
          <>
            Bạn sao chép hàm stub của Fig. 2.8, với <code>int size = n * sizeof(float);</code> của nó, để đưa một bảng embedding lên device: {n(TABLE.rows)} token nhân {n(TABLE.cols)} chiều kiểu <code>float</code>, nên{' '}<code>int n = {n(TABLE_N)}</code>. Điều gì hỏng?
          </>
        }
      >
        You copy the stub of Fig. 2.8, with its <code>int size = n * sizeof(float);</code>, to put an embedding table
        on the device: {n(TABLE.rows)} tokens by {n(TABLE.cols)} dimensions of <code>float</code>, so{' '}
        <code>int n = {n(TABLE_N)}</code>. What goes wrong?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không gì cả: sizeof làm cho tích là một size_t, và size_t đủ lớn cho mọi bảng.">
            Nothing: sizeof makes the product a size_t, and a size_t is large enough for any table.
          </Translated>
        ),
        why: (
          <Translated vi="Tích được tính dưới dạng size_t và đúng. Sau đó nó được lưu vào một int, và đó là chỗ nó không còn đúng.">
            The product is computed as a size_t and is right. It is then stored in an int, and that is where it stops being right.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi={`size không chứa được ${n(TABLE_BYTES)}, nên cudaMalloc và cả hai lần sao chép nhận một số byte sai.`}>
            {`size cannot hold ${n(TABLE_BYTES)}, so cudaMalloc and both copies get a wrong byte count.`}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="int i của kernel bị tràn với các phần tử cuối, nên chúng bị ghi vào sai chỗ.">
            The kernel's int i overflows for the last elements, so they are written to the wrong place.
          </Translated>
        ),
        why: (
          <Translated
            vi={`Chỉ số lớn nhất là ${n(TABLE_N - 1)}, thấp hơn nhiều so với ${n(INT_MAX)}. Một chỉ số vào bảng cỡ này vẫn vừa một int; kích thước theo byte của nó thì không.`}
          >
            {`The largest index is ${n(TABLE_N - 1)}, well below ${n(INT_MAX)}. An index into a table of this size still fits in an int; its size in bytes does not.`}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Dòng đó không còn biên dịch được.">
            The line no longer compiles.
          </Translated>
        ),
        why: (
          <Translated vi="Chuyển một size_t sang int là một phép chuyển đổi thu hẹp mà C++ cho phép không một lời cảnh báo. Một số trình biên dịch cảnh báo về nó khi được yêu cầu.">
            Converting a size_t to an int is a narrowing conversion that C++ allows without a word. Some compilers warn about it when asked to.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            {n(TABLE_N)} số float là {n(TABLE_BYTES)} byte. Giá trị lớn nhất của một <code>int</code> 32 bit là{' '}{n(INT_MAX)}. Từ C++20, phép chuyển đổi được định nghĩa là quay vòng theo modulo 2³², nên <code>size</code> chứa{' '}{n(WRAPPED)}, và khi giá trị đó được truyền vào chỗ <code>cudaMalloc</code> chờ một <code>size_t</code>, nó trở thành một số không dấu khổng lồ. Yêu cầu đã sai trước khi runtime kịp thấy nó. Fig. 2.8 vẫn ổn với các vectơ của sách; các lab viết <code>size_t size = (size_t)n * sizeof(float);</code>, và lời giải tham chiếu (solution) của lab 2.1 ghi chú rằng một số byte kiểu <code>int</code> bị tràn khi n đạt 2²⁹ số float.
          </p>
        }
      >
        <p>
          {n(TABLE_N)} floats are {n(TABLE_BYTES)} bytes. The largest value of a 32-bit <code>int</code> is{' '}
          {n(INT_MAX)}. Since C++20 the conversion is defined to wrap modulo 2³², so <code>size</code> holds{' '}
          {n(WRAPPED)}, and when that is passed where <code>cudaMalloc</code> expects a <code>size_t</code> it becomes an
          enormous unsigned number. The request is wrong before the runtime ever sees it. Fig. 2.8 is fine for the
          book's vectors; the labs write <code>size_t size = (size_t)n * sizeof(float);</code>, and lab 2.1's solution
          notes that an <code>int</code> byte count overflows once n reaches 2²⁹ floats.
        </p>
      </Translated>
    ),
    source: 'ch. 2 Fig. 2.8, p. 31; int versus unsigned, footnote 4, p. 37; the conversion rule is C++\'s, ours',
    vi: { source: 'ch. 2 Fig. 2.8, p. 31; int so với unsigned, chú thích 4, p. 37; quy tắc chuyển đổi là của C++, phần áp dụng là của khóa học này' },
  },
  'ch02/wrap/q-overlap': {
    kind: 'choice',
    id: 'ch02/wrap/q-overlap',
    prompt: (
      <Translated
        vi={
          <>
            Một hàm host khởi chạy một kernel chạy trong 3 ms, rồi dành 2 ms thời gian CPU để tách token cho yêu cầu tiếp theo, rồi gọi <code>cudaMemcpy</code> để lấy kết quả của kernel. Bỏ qua chi phí khởi chạy và bản thân lần sao chép, khoảng bao lâu sau lần khởi chạy thì <code>cudaMemcpy</code> trả về? (Các thời gian được đặt ra cho câu hỏi này.)
          </>
        }
      >
        A host function launches a kernel that runs for 3 ms, then spends 2 ms of CPU time tokenising the next
        request, then calls <code>cudaMemcpy</code> to fetch the kernel's result. Leaving out the launch overhead and
        the copy itself, about how long after the launch does <code>cudaMemcpy</code> return? (The times are invented
        for this question.)
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Khoảng 5 ms: việc của CPU bắt đầu khi kernel đã xong.">
            About 5 ms: the CPU work starts once the kernel has finished.
          </Translated>
        ),
        why: (
          <Translated vi="Lần khởi chạy trả về trước khi kernel chạy, nên việc của CPU bắt đầu ngay.">
            The launch returns before the kernel has run, so the CPU work starts at once.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Khoảng 3 ms: việc của CPU chạy trong khi kernel chạy, và lần sao chép chờ 1 ms cuối.">
            About 3 ms: the CPU work runs while the kernel runs, and the copy waits for the last 1 ms.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Khoảng 2 ms: lần sao chép bắt đầu ngay sau việc của CPU và lấy bất cứ thứ gì có ở đó.">
            About 2 ms: the copy starts right after the CPU work and fetches whatever is there.
          </Translated>
        ),
        why: (
          <Translated vi="Lần sao chép đi vào cùng stream mặc định theo thứ tự với kernel và không trả về cho đến khi hoàn tất, nên nó đến sau kernel.">
            The copy goes into the same in-order default stream as the kernel and does not return until it has completed, so it comes after the kernel.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không thể nói được: không có cudaDeviceSynchronize thì không gì xếp lần sao chép sau kernel.">
            It cannot be said: without cudaDeviceSynchronize nothing orders the copy after the kernel.
          </Translated>
        ),
        why: (
          <Translated vi="Stream mặc định làm việc đó: công việc trong nó chạy theo thứ tự được phát lệnh.">
            The default stream does: work in it runs in the order it was issued.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đây là sự chồng lấp mà Fig. 2.3 của sách gợi ý nhưng không bao giờ nói ra. NVIDIA ghi ba điều: một lần khởi chạy trả về trước khi kernel hoàn tất, công việc phát lệnh không kèm stream chạy theo thứ tự trong stream mặc định, và{' '}<code>cudaMemcpy</code> không trả về cho đến khi sao chép xong. Việc lần sao chép do đó chờ kernel là cách khóa học này hiểu ba phát biểu đó; không có câu nào nói điều đó. Trên Metal, <code>commit</code> trả về ngay theo cùng cách, và <code>waitUntilCompleted</code> là lệnh chờ. Một máy chủ suy luận dùng đúng điều này để chuẩn bị yêu cầu tiếp theo trong khi GPU làm việc.
          </p>
        }
      >
        <p>
          This is the overlap the book's Fig. 2.3 hints at and never states. NVIDIA documents three things: a launch
          returns before the kernel completes, work issued without a stream runs in order in the default stream, and{' '}
          <code>cudaMemcpy</code> does not return until the copy is done. That the copy therefore waits for the kernel is
          our reading of those three statements; no single sentence says it. On Metal, <code>commit</code> returns at
          once in the same way, and <code>waitUntilCompleted</code> is the wait. An inference server uses exactly this
          to prepare the next request while the GPU works.
        </p>
      </Translated>
    ),
    source: 'ch. 2 §2.2, p. 24, Fig. 2.3; notes/today/ch02-facts.md, A5 (the inference is ours) and C4',
    vi: { source: 'ch. 2 §2.2, p. 24, Fig. 2.3; notes/today/ch02-facts.md, A5 (suy luận là của khóa học này) và C4' },
  },
};

const M0 = 'ch02/lab3_elementwise';
// Backends that are not a GPU: the emulator and its race-detection flavour.
const EMULATED = new Set(['emu', 'race']);

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const devices = Object.keys(reports).filter((b) => !EMULATED.has(b));
  const m0 = devices.filter((b) => labStatus(reports[b], M0, 'work') === 'pass');
  const m0emu = labStatus(reports.emu, M0, 'work') === 'pass';
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch02/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>
        <Translated vi="Những gì cần mang theo">What to carry forward</Translated>
      </h2>
      <ul>
        <Translated
          vi={
            <li>
              Một phép toán là song song dữ liệu khi không đầu ra nào được tính từ một đầu ra khác. Khi đó nó có thể có một luồng cho mỗi đầu ra, và không thứ tự nào của các luồng làm thay đổi kết quả.
            </li>
          }
        >
          <li>
            An operation is data parallel when no output is computed from another output. Then it can have one thread
            per output, and no order of the threads changes the result.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Một chương trình là một luồng host khởi chạy các lưới luồng device, được tổ chức thành các khối. Một lần khởi chạy trả về trước khi lưới của nó chạy. Việc chờ là thứ bạn phải yêu cầu: <code>cudaDeviceSynchronize()</code>, một lời gọi chặn trong cùng stream, hoặc <code>waitUntilCompleted</code> trên Metal.
            </li>
          }
        >
          <li>
            A program is one host thread that launches grids of device threads, organised in blocks. A launch returns
            before its grid has run. Waiting is something you ask for: <code>cudaDeviceSynchronize()</code>, a blocking
            call in the same stream, or <code>waitUntilCompleted</code> on Metal.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Một hàm stub phía host có ba phần: cấp phát và sao chép vào, khởi chạy, sao chép về và giải phóng. Mỗi lần đi qua ranh giới đều tốn kém, đó là lý do mã thực tế giữ dữ liệu trên device. Lab 2.2 đo xem tốn bao nhiêu.
            </li>
          }
        >
          <li>
            A host stub has three parts: allocate and copy in, launch, copy back and free. Every crossing of the
            boundary costs, which is why real code keeps its data on the device. Lab 2.2 measures by how much.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Kích thước tính bằng byte, đích đứng trước, và trên CUDA một con trỏ device chỉ được đưa cho các lời gọi API và kernel. CUDA không xóa bộ nhớ nó cấp phát.
            </li>
          }
        >
          <li>
            Sizes are in bytes, the destination comes first, and on CUDA a device pointer goes only to API calls and
            kernels. CUDA does not clear the memory it allocates.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Mỗi luồng tìm phần tử của nó bằng <code>i = blockIdx.x * blockDim.x + threadIdx.x</code> và kiểm tra{' '}<code>i &lt; n</code>. Các biến tự động của nó là của riêng nó. Lưới thay thế cho vòng lặp.
            </li>
          }
        >
          <li>
            Each thread finds its element with <code>i = blockIdx.x * blockDim.x + threadIdx.x</code> and tests{' '}
            <code>i &lt; n</code>. Its automatic variables are its own. The grid takes the place of the loop.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Số khối là <code>(n + B - 1) / B</code> theo số nguyên. Một câu lệnh khởi chạy không trả về mã lỗi: hãy đọc trạng thái lỗi ngay sau nó, rồi kiểm tra lệnh chờ. Các khối chạy theo thứ tự bất kỳ.
            </li>
          }
        >
          <li>
            The block count is <code>(n + B - 1) / B</code> in integers. A launch statement returns no error code: read
            the error state right after it, then check the wait. Blocks run in any order.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Qualifier quyết định ai biên dịch một hàm, ai được gọi nó và ai chạy nó. Một định nghĩa{' '}<code>__host__ __device__</code> được build hai lần. Metal không có hàm như vậy: kernel của nó là một ngôn ngữ riêng, được hoàn tất cho GPU của bạn khi pipeline được tạo.
            </li>
          }
        >
          <li>
            Qualifiers decide who compiles a function, who may call it and who runs it. One{' '}
            <code>__host__ __device__</code> definition is built twice. Metal has no such function: its kernels are a
            separate language, finished for your GPU when the pipeline is created.
          </li>
        </Translated>
      </ul>
      <Source section="2.8" pages="40" />
      <Translated
        vi={
          <p>
            Sách kết thúc chương bằng cách xếp những gì nó giới thiệu vào bốn nhóm phần mở rộng. Đây là chúng cùng với các thứ tương ứng trên hai backend còn lại.
          </p>
        }
      >
        <p>
          The book ends the chapter by sorting what it introduced into four groups of extensions. Here they are with
          their counterparts on the other two backends.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, B1, B4, C1 to C5, C7)', '2026-11-20 (notes/today/ch02-facts.md, B1, B4, C1 đến C5, C7)')}>
        <Translated
          vi={
            <p>
              HIP giữ cách viết của CUDA khi <code>&lt;hip/hip_runtime.h&gt;</code> được include. Một buffer Metal lưu trữ chia sẻ được đọc và ghi qua <code>contents</code> và được giải phóng cùng với tham chiếu cuối cùng của nó.
            </p>
          }
        >
          <p>
            HIP keeps CUDA's spelling once <code>&lt;hip/hip_runtime.h&gt;</code> is included. A shared-storage Metal
            buffer is read and written through <code>contents</code> and is released with its last reference.
          </p>
        </Translated>
      </Today>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-table="four-groups">
          <thead>
            <tr>
              <th>
                <Translated vi="Nhóm của sách">The book's group</Translated>
              </th>
              <th>CUDA</th>
              <th>HIP</th>
              <th>Metal</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <Translated vi="Khai báo hàm">Function declarations</Translated>
              </td>
              <td><code>__global__</code>, <code>__device__</code>, <code>__host__</code></td>
              <td>
                <Translated vi="cùng ba từ khóa đó">the same three</Translated>
              </td>
              <td>
                <Translated vi={<><code>kernel</code>; mọi hàm khác là mã GPU; không gì biên dịch một hàm cho cả hai phía</>}>
                  <><code>kernel</code>; any other function is GPU code; nothing compiles one function for both sides</>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>
                <Translated vi="Gọi kernel và khởi chạy lưới">Kernel call and grid launch</Translated>
              </td>
              <td><code>kernel&lt;&lt;&lt;blocks, threads&gt;&gt;&gt;(args)</code></td>
              <td>
                <Translated vi="cùng cú pháp">the same syntax</Translated>
              </td>
              <td>
                <Translated vi={<>một encoder: pipeline, buffer, <code>dispatchThreads</code> hoặc <code>dispatchThreadgroups</code>, rồi <code>commit</code></>}>
                  <>an encoder: pipeline, buffers, <code>dispatchThreads</code> or <code>dispatchThreadgroups</code>, then <code>commit</code></>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>
                <Translated vi="Biến dựng sẵn">Built-in variables</Translated>
              </td>
              <td><code>threadIdx</code>, <code>blockIdx</code>, <code>blockDim</code></td>
              <td>
                <Translated vi="cùng tên">the same names</Translated>
              </td>
              <td>
                <Translated vi={<>các đối số kernel có thuộc tính, như <code>[[thread_position_in_grid]]</code></>}>
                  <>kernel arguments with attributes, such as <code>[[thread_position_in_grid]]</code></>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>
                <Translated vi="Runtime API">Runtime API</Translated>
              </td>
              <td><code>cudaMalloc</code>, <code>cudaMemcpy</code>, <code>cudaFree</code></td>
              <td><code>hipMalloc</code>, <code>hipMemcpy</code>, <code>hipFree</code></td>
              <td>
                <Translated vi={<><code>newBufferWithLength</code>, <code>newBufferWithBytes</code>, <code>contents</code>; không có lời gọi giải phóng</>}>
                  <><code>newBufferWithLength</code>, <code>newBufferWithBytes</code>, <code>contents</code>; no free call</>
                </Translated>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>
        <Translated vi="Một bài kiểm tra tổng hợp">A mixed check</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Các câu hỏi này dựa trên cả chương và trên những tình huống các mục chưa trình bày.
        {asked ? ` Cho đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu hỏi, tính theo câu trả lời gần nhất của bạn cho mỗi câu.` : ''}
          </p>
        }
      >
        <p>
          These draw on the whole chapter and on situations the sections did not show.
          {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
        </p>
      </Translated>
      <Code
        tabs={[
          {
            label: tr('CUDA (not a lab file: for this question only)', 'CUDA (không phải tệp lab: chỉ cho câu hỏi này)'),
            code: SAXPY,
            tiers: { x: 'global', y: 'global', x_d: 'global', y_d: 'global', i: 'reg' },
            status: 'front-end-checked; has a defect on purpose',
          },
        ]}
        caption={
          <Translated
            vi={
              <>
                Của khóa học này, không phải của sách. Bản đã kiểm tra là <code>labs/ch02/snippets/saxpy_stub_defect.cu</code>.
              </>
            }
          >
            <>
              Ours, not the book's. The checked copy is <code>labs/ch02/snippets/saxpy_stub_defect.cu</code>.
            </>
          </Translated>
        }
      />
      <Ask q={questionsWrap['ch02/wrap/q-inout']} label={tr('Find the defect', 'Tìm lỗi')} />
      <Ask q={questionsWrap['ch02/wrap/q-logits']} />
      <Code
        tabs={[
          {
            label: tr('CUDA (not a lab file: for this question only)', 'CUDA (không phải tệp lab: chỉ cho câu hỏi này)'),
            code: COUNT,
            tiers: { p: 'global', count: 'global', i: 'reg' },
            status: 'front-end-checked; has a defect on purpose',
          },
        ]}
        caption={
          <Translated
            vi={
              <>
                Của khóa học này, không phải của sách. Bản đã kiểm tra là <code>labs/ch02/snippets/count_above_race.cu</code>.
              </>
            }
          >
            <>
              Ours, not the book's. The checked copy is <code>labs/ch02/snippets/count_above_race.cu</code>.
            </>
          </Translated>
        }
      />
      <Ask q={questionsWrap['ch02/wrap/q-count']} />
      <Ask q={questionsWrap['ch02/wrap/q-bytes-int']} />
      <Ask q={questionsWrap['ch02/wrap/q-overlap']} />

      <h2>
        <Translated vi="Bài tập của sách">The book's exercises</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương kết thúc với mười bài tập (pp. 41–42). Mỗi bài được giải ở đâu đó trong khóa học này; sách không in đáp án, nên đáp án ở đây là của khóa học này.
          </p>
        }
      >
        <p>
          The chapter ends with ten exercises (pp. 41–42). Each is worked somewhere in this course; the book prints no
          answers, so the answers here are ours.
        </p>
      </Translated>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="exercises">
        <thead>
          <tr>
            <th>
              <Translated vi="Bài tập">Exercise</Translated>
            </th>
            <th>
              <Translated vi="Nội dung">About</Translated>
            </th>
            <th>
              <Translated vi="Được giải ở đâu tại đây">Where it is worked here</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>1</td>
            <td>
              <Translated vi="Biểu thức chỉ số cho một phần tử mỗi luồng">The index expression for one element per thread</Translated>
            </td>
            <td>
              <Translated vi={<>một câu hỏi trong <a href={href('ch02', '2.5')}>2.5</a></>}>
                <>a question in <a href={href('ch02', '2.5')}>2.5</a></>
              </Translated>
            </td>
          </tr>
          <tr>
            <td>2, 3</td>
            <td>
              <Translated vi="Hai phần tử mỗi luồng, liền kề hoặc ở hai đoạn">Two elements per thread, adjacent or in two sections</Translated>
            </td>
            <td>
              <Translated vi={<>hai câu hỏi trong <a href={href('ch02', '2.5')}>2.5</a>, mở khóa các cách ánh xạ đó trong trình khám phá chỉ số của mục</>}>
                <>two questions in <a href={href('ch02', '2.5')}>2.5</a>, which unlock those mappings in its index explorer</>
              </Translated>
            </td>
          </tr>
          <tr>
            <td>4</td>
            <td>
              <Translated vi="Số luồng trong lưới cho 8000 phần tử với các khối 1024">Threads in the grid for 8000 elements in blocks of 1024</Translated>
            </td>
            <td>
              <Translated vi={<>một câu hỏi trong <a href={href('ch02', '2.6')}>2.6</a>, và công cụ tính số khối của mục</>}>
                <>a question in <a href={href('ch02', '2.6')}>2.6</a>, and its block-count calculator</>
              </Translated>
            </td>
          </tr>
          <tr>
            <td>5, 6</td>
            <td>
              <Translated vi={<>Hai đối số của <code>cudaMalloc</code></>}>
                <>The two arguments of <code>cudaMalloc</code></>
              </Translated>
            </td>
            <td>
              <Translated vi={<>hai câu hỏi trong <a href={href('ch02', '2.4')}>2.4</a></>}>
                <>two questions in <a href={href('ch02', '2.4')}>2.4</a></>
              </Translated>
            </td>
          </tr>
          <tr>
            <td>7</td>
            <td>
              <Translated vi={<>Lời gọi <code>cudaMemcpy</code> cho một lần sao chép sang device</>}>
                <>The <code>cudaMemcpy</code> call for a copy to the device</>
              </Translated>
            </td>
            <td>
              <Translated vi={<>một câu hỏi trong <a href={href('ch02', '2.4')}>2.4</a>, với bốn lời gọi của khóa học này</>}>
                <>a question in <a href={href('ch02', '2.4')}>2.4</a>, with four calls of our own</>
              </Translated>
            </td>
          </tr>
          <tr>
            <td>8</td>
            <td>
              <Translated vi="Kiểu nhận kết quả của một lời gọi API">The type that receives an API call's result</Translated>
            </td>
            <Translated
              vi={
                <td>
                  câu hỏi về tham số đầu ra của <code>cudaMalloc</code> trong <a href={href('ch02', '2.4')}>2.4</a>, và macro{' '}<code>CHECK</code> bạn viết trong lab 2.1 (<a href={href('ch02', 'labs')}>lab</a>)
                </td>
              }
            >
              <td>
                the question on <code>cudaMalloc</code>'s out-parameter in <a href={href('ch02', '2.4')}>2.4</a>, and the{' '}
                <code>CHECK</code> macro you write in lab 2.1 (<a href={href('ch02', 'labs')}>labs</a>)
              </td>
            </Translated>
          </tr>
          <tr>
            <td>9</td>
            <td>
              <Translated vi="Số khối, số luồng và số luồng làm việc của Fig. 2.15">Blocks, threads and working threads of Fig. 2.15</Translated>
            </td>
            <td>
              <Translated vi={<>“Đếm số luồng” trong <a href={href('ch02', '2.6')}>2.6</a></>}>
                <>"Count the threads" in <a href={href('ch02', '2.6')}>2.6</a></>
              </Translated>
            </td>
          </tr>
          <tr>
            <td>10</td>
            <td>
              <Translated vi="Một lập trình viên viết mỗi hàm hai lần">A programmer who writes every function twice</Translated>
            </td>
            <td>
              <Translated vi={<>một câu hỏi trong <a href={href('ch02', '2.7')}>2.7</a>; <code>scaleShift</code> của lab 2.3 làm đúng cách</>}>
                <>a question in <a href={href('ch02', '2.7')}>2.7</a>; lab 2.3's <code>scaleShift</code> does it right</>
              </Translated>
            </td>
          </tr>
        </tbody>
      </table>

      <h2>
        <Translated vi="Những chỗ bản PDF có vẻ sai">Where the PDF looks wrong</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Ấn bản này còn mới và bản mà khóa học này dựa vào có những chỗ sơ suất rõ ràng trong chương 2. Ba chỗ đầu nằm trong mã: như được in, hai chỗ không biên dịch được và chỗ thứ ba dẫn đến lỗi chia nguyên. Đó là lý do các bài học chỉ hiển thị các tệp đã kiểm thử và không bao giờ gõ lại một đoạn mã. Những chỗ còn lại được liệt kê để bạn không bị bối rối khi đọc sách song song.
          </p>
        }
      >
        <p>
          This edition is new and the copy this course was built from has apparent slips in chapter 2. The first three
          are in code: as printed, two of them do not compile and the third invites the integer-division bug. That is
          why the lessons show only tested files and never retype a listing. The rest are listed so that you are not
          thrown by them when you read the book alongside.
        </p>
      </Translated>
      <ul data-list="errata">
        <Translated
          vi={
            <li>
              Fig. 2.5, dòng 03 (p. 27): <code>float *d_A *d_B, *d_C;</code> thiếu dấu phẩy sau <code>*d_A</code>, và dùng tiền tố <code>d_</code> trong khi mọi chỗ khác dùng hậu tố <code>_d</code>.{' '}<a href={href('ch02', '2.4')}>Mục 2.4</a> yêu cầu bạn tìm ra nó.
            </li>
          }
        >
          <li>
            Fig. 2.5, line 03 (p. 27): <code>float *d_A *d_B, *d_C;</code> lacks a comma after <code>*d_A</code>, and uses
            a <code>d_</code> prefix where everything else uses the <code>_d</code> suffix.{' '}
            <a href={href('ch02', '2.4')}>Section 2.4</a> has you find it.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Phép kiểm tra lỗi ở p. 32 khai báo <code>cudaError_t err</code> rồi lại kiểm tra <code>error</code>. Mục 2.4 cũng yêu cầu bạn tìm ra nó, và macro <code>CHECK</code> của lab 2.1 là phiên bản đã sửa.
            </li>
          }
        >
          <li>
            The error check on p. 32 declares <code>cudaError_t err</code> and then tests <code>error</code>. Section
            2.4 has you find it too, and lab 2.1's <code>CHECK</code> macro is the corrected version.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Fig. 2.12 (p. 38) đặt tên hàm là <code>vectAdd</code>, cho nó kiểu trả về <code>int</code> mà không có{' '}<code>return</code>, và chú thích của nó ghi <code>ceil(n/256)</code>, mà khi đọc như C++ với một{' '}<code>n</code> nguyên chính là lỗi mà <a href={href('ch02', '2.6')}>mục 2.6</a> mổ xẻ. Ở mọi chỗ khác hàm là <code>void vecAdd</code>.
            </li>
          }
        >
          <li>
            Fig. 2.12 (p. 38) names the function <code>vectAdd</code>, gives it return type <code>int</code> with no{' '}
            <code>return</code>, and its comment says <code>ceil(n/256)</code>, which read as C++ with an integer{' '}
            <code>n</code> is the bug <a href={href('ch02', '2.6')}>section 2.6</a> takes apart. Everywhere else the
            function is <code>void vecAdd</code>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Chú thích của Figs. 2.12 và 2.13 (p. 38) đọc như bị đổi chỗ: hình chỉ hiển thị lần khởi chạy được gọi là ví dụ mã host, còn hàm host hoàn chỉnh lại được gọi là câu lệnh gọi kernel.
            </li>
          }
        >
          <li>
            The captions of Figs. 2.12 and 2.13 (p. 38) read as if swapped: the one that shows only the launch is called
            a host code example, and the complete host function is called a kernel call statement.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Cách đặt tên <code>_h</code> và <code>_d</code> ở p. 26 không được các đoạn mã giữ. Figs. 2.4 và 2.8 dùng{' '}<code>A_h</code>, <code>B_h</code>, <code>C_h</code>; Figs. 2.5, 2.12 và 2.13 dùng <code>A</code>,{' '}<code>B</code>, <code>C</code> trơn cho các mảng host. Bên trong kernel phần chữ bỏ các hậu tố (p. 35), như Fig. 2.10, nhưng hộp mã của Fig. 2.9 viết <code>A_d</code>, <code>B_d</code>, <code>C_d</code>.
            </li>
          }
        >
          <li>
            The <code>_h</code> and <code>_d</code> naming of p. 26 is not kept by the listings. Figs. 2.4 and 2.8 use{' '}
            <code>A_h</code>, <code>B_h</code>, <code>C_h</code>; Figs. 2.5, 2.12 and 2.13 use plain <code>A</code>,{' '}
            <code>B</code>, <code>C</code> for the host arrays. Inside the kernel the text drops the suffixes (p. 35),
            as Fig. 2.10 does, but the code box of Fig. 2.9 writes <code>A_d</code>, <code>B_d</code>, <code>C_d</code>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Hộp mã của Fig. 2.9 (p. 33) không có <code>if (i &lt; n)</code>; Fig. 2.10 thì có.
            </li>
          }
        >
          <li>The code box of Fig. 2.9 (p. 33) has no <code>if (i &lt; n)</code>; Fig. 2.10 has it.</li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 37 nói kernel của “Fig. 2.10” không có vòng lặp tương ứng với vòng lặp trong Fig. 2.10; vòng lặp nằm ở Fig. 2.4. Cùng trang đó gọi kernel là <code>vecAddVecKernel</code>; thực ra nó là <code>vecAddKernel</code>.
            </li>
          }
        >
          <li>
            p. 37 says the kernel of "Fig. 2.10" has no loop corresponding to the one in Fig. 2.10; the loop is in Fig.
            2.4. The same page calls the kernel <code>vecAddVecKernel</code>; it is <code>vecAddKernel</code>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Fig. 2.3 (p. 25) vẽ luồng host thành từng mảnh bên cạnh mỗi lưới và không cho thấy nó chờ hay làm việc trong khi lưới chạy, và chương này không bao giờ nói một lần khởi chạy có chặn hay không.{' '}<a href={href('ch02', '2.2')}>Mục 2.2</a> trả lời điều đó từ tài liệu của các nhà cung cấp.
            </li>
          }
        >
          <li>
            Fig. 2.3 (p. 25) draws the host thread in pieces beside each grid and does not show whether it waits or
            works while the grid runs, and the chapter never says whether a launch blocks.{' '}
            <a href={href('ch02', '2.2')}>Section 2.2</a> answers it from the vendors' documentation.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Đoạn mã ở p. 30 in <code>float *A_d</code> mà thiếu dấu chấm phẩy.
            </li>
          }
        >
          <li>The snippet on p. 30 prints <code>float *A_d</code> without its semicolon.</li>
        </Translated>
        <Translated
          vi={
            <li>
              Bài tập 3 (p. 41) ghi “will first process a section first”, và in cùng bốn lựa chọn như bài tập 2. Đáp án dự định khác nhau: (c) cho bài tập 2, (d) cho bài tập 3, theo cách suy ra của khóa học này.
            </li>
          }
        >
          <li>
            Exercise 3 (p. 41) reads "will first process a section first", and prints the same four options as exercise
            2. The intended answers differ: (c) for exercise 2, (d) for exercise 3, by our derivation.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Những điều nhỏ hơn: “discussed in 5” ở p. 28 thiếu chữ Chapter; một câu ở p. 36 thiếu chữ “in”; chú thích 2 (p. 29) nói <code>cudaMalloc</code> “returns a generic object”, trong khi phần chữ giải thích rằng nó ghi con trỏ qua tham số đầu tiên; tài liệu tham khảo [1] (p. 43) ghi ba năm cho một cuốn sách.
            </li>
          }
        >
          <li>
            Smaller things: "discussed in 5" on p. 28 is missing the word Chapter; a sentence on p. 36 is missing the
            word "in"; footnote 2 (p. 29) says <code>cudaMalloc</code> "returns a generic object", where the text explains
            that it writes the pointer through its first parameter; reference [1] (p. 43) gives three years for one
            book.
          </li>
        </Translated>
      </ul>
      <Aside>
        <Translated vi={<p>Đây là cách khóa học này đọc một bản PDF. Nếu bản của bạn khác, hãy tin bản của bạn.</p>}>
          <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Có hai điều trông như sơ suất nhưng không phải. Phần chữ và Fig. 2.9 viết chỉ số là{' '}<code>blockIdx.x * blockDim.x + threadIdx.x</code> còn Fig. 2.10 viết{' '}<code>threadIdx.x + blockDim.x * blockIdx.x</code>: cùng một giá trị. Và hai lựa chọn của bài tập 5 dùng một{' '}<code>n</code> mà câu hỏi không hề định nghĩa; chúng là phương án gây nhiễu.
          </p>
        }
      >
        <p>
          Two things look like slips and are not. The text and Fig. 2.9 write the index as{' '}
          <code>blockIdx.x * blockDim.x + threadIdx.x</code> and Fig. 2.10 as{' '}
          <code>threadIdx.x + blockDim.x * blockIdx.x</code>: the same value. And two options of exercise 5 use an{' '}
          <code>n</code> the question never defines; they are distractors.
        </p>
      </Translated>

      <h2>
        <Translated vi="Những chỗ tài liệu hiện nay khác với trang sách">Where today's documentation differs from the page</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Tài liệu tham khảo của sách cho CUDA là hướng dẫn lập trình của NVIDIA bản phát hành 13.0 (tài liệu tham khảo [3] của sách). Các trang hiện hành của nhà cung cấp bổ sung cho chương hoặc nói khác đi ở những chỗ sau, mỗi chỗ được bàn trong mục được nêu tên.
          </p>
        }
      >
        <p>
          The book's reference for CUDA is NVIDIA's programming guide of release 13.0 (its reference [3]). The vendors'
          current pages add to the chapter or say things differently in these places, each taken up in the section
          named.
        </p>
      </Translated>
      <ul data-list="today">
        <Translated
          vi={
            <li>
              Một lần khởi chạy là bất đồng bộ: nó trả về trước khi kernel chạy. Sách không bao giờ nói vậy trong chương này; các đoạn mã của sách chạy đúng vì lần sao chép về có chặn chạy sau kernel trong stream mặc định (<a href={href('ch02', '2.2')}>2.2</a>, <a href={href('ch02', '2.6')}>2.6</a>, lab 2.2).
            </li>
          }
        >
          <li>
            A launch is asynchronous: it returns before the kernel has run. The book never says so in this chapter; its
            listings work because the blocking copy back runs after the kernel in the default stream (
            <a href={href('ch02', '2.2')}>2.2</a>, <a href={href('ch02', '2.6')}>2.6</a>, lab 2.2).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Một câu lệnh khởi chạy không trả về mã lỗi, và phép kiểm tra của sách không bao quát nó. Mẫu của NVIDIA là{' '}<code>cudaGetLastError()</code> ngay sau lần khởi chạy và một <code>cudaDeviceSynchronize()</code> có kiểm tra{' '}sau đó (<a href={href('ch02', '2.6')}>2.6</a>, lab 2.1).
            </li>
          }
        >
          <li>
            A launch statement returns no error code, and the book's check does not cover it. NVIDIA's pattern is{' '}
            <code>cudaGetLastError()</code> right after the launch and a checked <code>cudaDeviceSynchronize()</code>{' '}
            after that (<a href={href('ch02', '2.6')}>2.6</a>, lab 2.1).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              NVIDIA dạy cách tính số khối là số nguyên <code>(n + threads - 1) / threads</code>, hoặc{' '}<code>cuda::ceil_div</code>, không phải <code>ceil(n/256.0)</code>. CUDA làm gì với một lưới không khối nào thì không có trong tài liệu; HIP trả về lỗi (<a href={href('ch02', '2.6')}>2.6</a>).
            </li>
          }
        >
          <li>
            NVIDIA teaches the block count as the integer <code>(n + threads - 1) / threads</code>, or{' '}
            <code>cuda::ceil_div</code>, not <code>ceil(n/256.0)</code>. What CUDA does with a grid of zero blocks is
            not documented; HIP returns an error (<a href={href('ch02', '2.6')}>2.6</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Phép ép kiểu <code>(void**)</code> trên <code>cudaMalloc</code> không còn bắt buộc: mã C++ có thể dùng một overload có kiểu. Phép ép kiểu vẫn biên dịch được ở mọi nơi, và các lab giữ nó (<a href={href('ch02', '2.4')}>2.4</a>).
            </li>
          }
        >
          <li>
            The <code>(void**)</code> cast on <code>cudaMalloc</code> is no longer required: C++ code can use a typed
            overload. The cast still compiles everywhere, and the labs keep it (<a href={href('ch02', '2.4')}>2.4</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Có một loại sao chép thứ năm, <code>cudaMemcpyDefault</code>, tự suy ra hướng từ các con trỏ và được NVIDIA hiện khuyến nghị (<a href={href('ch02', '2.4')}>2.4</a>).
            </li>
          }
        >
          <li>
            There is a fifth copy kind, <code>cudaMemcpyDefault</code>, which infers the direction from the pointers and
            which NVIDIA now recommends (<a href={href('ch02', '2.4')}>2.4</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Bảng qualifier đã có thêm hai hàng, <code>__tile__</code> và <code>__tile_global__</code>, cho các kernel CUDA Tile. Ba hàng của sách không thay đổi (<a href={href('ch02', '2.7')}>2.7</a>).
            </li>
          }
        >
          <li>
            The qualifier table has grown two rows, <code>__tile__</code> and <code>__tile_global__</code>, for CUDA
            Tile kernels. The book's three rows are unchanged (<a href={href('ch02', '2.7')}>2.7</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Mặc định <code>nvcc</code> biên dịch mã device sang mã máy lúc build và đóng gói nó cùng PTX. Driver, không phải một phần của NVCC, biên dịch PTX lúc chạy, và chỉ khi không có mã máy nào hợp với GPU (<a href={href('ch02', '2.7')}>2.7</a>).
            </li>
          }
        >
          <li>
            By default <code>nvcc</code> compiles device code to machine code at build time and packs it with PTX. The
            driver, not a part of NVCC, compiles the PTX at run time, and only when no machine code fits the GPU (
            <a href={href('ch02', '2.7')}>2.7</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Unified Virtual Addressing, theo cách diễn đạt hiện nay của NVIDIA, là chuyện địa chỉ. Thứ cho phép một kernel đọc bộ nhớ host là bộ nhớ được ánh xạ (mapped) hoặc unified memory (<a href={href('ch02', '2.5')}>2.5</a>).
            </li>
          }
        >
          <li>
            Unified Virtual Addressing, in NVIDIA's current wording, is about addresses. What lets a kernel read host
            memory is mapped or unified memory (<a href={href('ch02', '2.5')}>2.5</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Hướng dẫn của bản phát hành 13.0 nay là một bản lưu trữ; hướng dẫn hiện hành mở đầu bằng chính phép cộng vectơ của chương này. Các số liệu về device của sách là minh họa riêng của sách: “một hoặc hai” và “128 hoặc 256” khối cùng lúc không có nguồn, và 80 GB với 94 GB của H100 là hai sản phẩm.
            </li>
          }
        >
          <li>
            The guide of release 13.0 is now an archive; the current guide opens with this chapter's vector addition.
            The book's device figures are its own illustrations: "one or two" and "128 or 256" blocks at once have no
            source, and the H100's 80 GB and 94 GB are two products.
          </li>
        </Translated>
      </ul>
      <Today
        checked={tr(
          '2026-11-20 (notes/today/ch02-facts.md, A1, A2, A4 to A8, A12, A14, A15, A18)',
          '2026-11-20 (notes/today/ch02-facts.md, A1, A2, A4 đến A8, A12, A14, A15, A18)',
        )}
      >
        <Translated
          vi={
            <p>
              Đã đối chiếu với hướng dẫn lập trình v13.5 và tài liệu tham khảo runtime API của NVIDIA, tài liệu HIP của AMD và tài liệu Metal của Apple. Các khẳng định của sách về số chu kỳ tạo luồng và về việc CUDA là framework trưởng thành nhất không thể kiểm chứng theo hướng nào.
            </p>
          }
        >
          <p>
            Checked against NVIDIA's programming guide v13.5 and runtime API reference, AMD's HIP documentation and
            Apple's Metal documentation. The book's claims on thread-creation cycles and on CUDA being the most mature
            framework could not be checked either way.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Những gì chương này bỏ qua">What this chapter left out</Translated>
      </h2>
      <ul>
        <Translated
          vi={
            <li>
              Từ mục 2.1: danh sách các ứng dụng nặng dữ liệu (p. 21), bức ảnh của Fig. 2.1, phần lớn khung bên về RGB (p. 22) và danh sách tác vụ động lực học phân tử của khung bên về dữ liệu so với tác vụ (p. 23). Bản thân kernel chuyển xám là nội dung chương 3.
            </li>
          }
        >
          <li>
            From section 2.1: the list of data-heavy applications (p. 21), the photograph of Fig. 2.1, most of the RGB
            sidebar (p. 22) and the molecular-dynamics task list of the data-versus-task sidebar (p. 23). The grayscale
            kernel itself is chapter 3.
          </li>
        </Translated>
        <Translated vi={<li>Khung bên về con trỏ trong C++ (pp. 26–27).</li>}>
          <li>The sidebar on pointers in C++ (pp. 26–27).</li>
        </Translated>
        <Translated
          vi={
            <li>
              Mảng nhiều chiều với <code>cudaMalloc</code> (chú thích 2, p. 29): chương 3. Các hàm cấp phát nhanh hơn hoặc không cần sao chép mà sách hoãn lại (p. 29) được nêu tên trong <a href={href('ch02', '2.4')}>mục 2.4</a> theo tài liệu hiện nay; phần còn lại là <a href={href('ch23')}>chương 23</a> và <a href={href('appC')}>phụ lục C</a>.
            </li>
          }
        >
          <li>
            Multidimensional arrays with <code>cudaMalloc</code> (footnote 2, p. 29): chapter 3. The faster or copy-free
            allocation functions the book defers (p. 29) are named in <a href={href('ch02', '2.4')}>section 2.4</a> from today's
            documentation; the rest is <a href={href('ch23')}>chapter 23</a> and <a href={href('appC')}>appendix C</a>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Phép so sánh số điện thoại của khung bên về phân cấp (pp. 34–35) và chú thích về <code>int</code> so với{' '}<code>unsigned</code> (p. 37) mỗi cái được một câu; chú thích về UVA (p. 35) là một ghi chú bên lề, với chi tiết ở phụ lục C. Song song động (p. 35) được một đoạn trong mục 2.7: ấn bản này đã bỏ chương của nó.
            </li>
          }
        >
          <li>
            The telephone-number analogy of the hierarchy sidebar (pp. 34–35) and the <code>int</code> versus{' '}
            <code>unsigned</code> footnote (p. 37) are one sentence each; the UVA footnote (p. 35) is a margin note, with
            the detail in appendix C. Dynamic parallelism (p. 35) gets one paragraph in section 2.7: this edition
            removed its chapter.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Fig. 2.14 được thay bằng một quy trình cho mỗi backend trong ba backend, theo tài liệu hiện nay.
            </li>
          }
        >
          <li>
            Fig. 2.14 is replaced by a pipeline for each of the three backends that follows today's documentation.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Phép cộng vectơ gộp việc cho luồng (thread coarsening) của bài tập 2 và 3 được giải như phép tính chỉ số, không phải như một lab. Việc gộp việc được đo một cách đúng đắn ở chương 6 và 15.
            </li>
          }
        >
          <li>
            The coarsened vector addition of exercises 2 and 3 is worked as index arithmetic, not as a lab. Coarsening is
            measured properly in chapters 6 and 15.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Được nêu tên nhưng không dạy ở đây: <code>cudaMemcpyDefault</code>, stream và event,{' '}<code>cudaLaunchKernelEx</code>, kernel CUDA Tile, các loại lệnh của Metal 4, lưu trữ private và sao chép blit của Metal, và các trình biên dịch lúc chạy hiprtc và nvrtc.
            </li>
          }
        >
          <li>
            Named and not taught here: <code>cudaMemcpyDefault</code>, streams and events,{' '}
            <code>cudaLaunchKernelEx</code>, CUDA Tile kernels, Metal 4's command types, Metal private storage and blit
            copies, and the run-time compilers hiprtc and nvrtc.
          </li>
        </Translated>
      </ul>

      <h2>
        <Translated vi="Dự án cho đến giờ">The project so far</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Các kernel của lab 2.3, cộng bias, nhân hệ số và dịch, và cộng residual, mỗi kernel có một hàm khởi chạy bên cạnh, là mốc dự án M0 của dự án xuyên suốt: các phép toán theo từng phần tử mà mọi khối của transformer nhỏ bạn sẽ chạy đều cần.{' '}
        <span data-m0>
          {devices.length === 0 && !reports.emu
            ? 'Chưa có kết quả lab nào được ghi trên máy này.'
            : m0.length
              ? `Phiên bản của bạn qua trên ${m0.join(' và ')} trên máy này.`
              : m0emu
                ? 'Phiên bản của bạn qua trên trình giả lập; nó chưa qua trên một device nào trên máy này.'
                : 'Phiên bản của bạn chưa qua trên máy này.'}
        </span>{' '}
        Xem các <a href={href('ch02', 'labs')}>lab</a>.
          </p>
        }
      >
        <p>
          Lab 2.3's kernels, bias add, scale and shift, and residual add, with a launch function beside each, are
          milestone M0 of the through-line project: the elementwise operations every block of the small transformer you
          will end up running needs.{' '}
          <span data-m0>
            {devices.length === 0 && !reports.emu
              ? 'No lab results are recorded on this machine yet.'
              : m0.length
                ? `Your own version passes on ${m0.join(' and ')} on this machine.`
                : m0emu
                  ? 'Your own version passes on the emulator; it has not passed on a device on this machine yet.'
                  : 'Your own version has not passed on this machine yet.'}
          </span>{' '}
          See the <a href={href('ch02', 'labs')}>labs</a>.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Dự án, <code>nanoinfer</code>, và các mốc dự án của nó là của khóa học này, không phải của sách.
            </p>
          }
        >
          <p>The project, <code>nanoinfer</code>, and its milestones are ours, not the book's.</p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            <a href={href('ch03', 'start')}>Chương 3</a> đưa lưới từ một chiều lên hai và ba chiều. Nó viết kernel chuyển xám mà mục 2.1 dùng làm ví dụ, rồi một phép làm mờ, rồi phép nhân ma trận đằng sau một lớp tuyến tính, là mốc dự án tiếp theo, M1. Mọi thứ trong hàm stub phía host của chương này giữ nguyên; điều thay đổi là cách một luồng tìm phần tử của nó.
          </p>
        }
      >
        <p>
          <a href={href('ch03', 'start')}>Chapter 3</a> takes the grid from one dimension to two and three. It writes
          the grayscale kernel that section 2.1 used as its example, then a blur, then the matrix multiplication behind
          a linear layer, which is the next milestone, M1. Everything in this chapter's host stub carries over
          unchanged; what changes is how a thread finds its element.
        </p>
      </Translated>
    </>
  );
}
