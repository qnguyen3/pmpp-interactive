import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, type Tier } from '../../components/notes';
import { Code, useLabExcerpt } from '../../components/Code';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import { ceiling, speedup } from '../../sim/amdahl';

const LAB1 = 'ch02/lab1_vecadd/solution/kernel.cu';

const VEC_TIERS: Record<string, Tier> = { A: 'global', B: 'global', C: 'global', i: 'reg' };

// The book's own worked case (ch. 1, p. 9) and the ceiling this page asks for, from the model chapter 1 runs on.
const BOOK_CASE = speedup(0.3, 100);
const BOOK_LIMIT = ceiling(0.3);
const DECODE_LIMIT = Math.round(ceiling(0.9));

// Prerequisite questions drawn from chapter 1 (from the notes for that chapter; it has no code).
export const questionsS20: Record<string, Question> = {
  'ch02/start/q-amdahl': {
    kind: 'numeric',
    id: 'ch02/start/q-amdahl',
    prompt: (
      <Translated
        vi={
          <>
            Trong một bước decode của một mô hình, 90% thời gian dành cho các phép toán mà bạn có thể chuyển vào
            kernel. 10% còn lại vẫn ở trên CPU: tách token, logic lấy mẫu, chính vòng lặp. Giả sử phần kernel trở nên
            nhanh vô hạn. Mức tăng tốc tổng thể lớn nhất là bao nhiêu lần?
          </>
        }
      >
        <>
          In one decode step of a model, 90% of the time is spent in operations you could move into kernels. The other
          10% stays on the CPU: tokenising, the sampling logic, the loop itself. Suppose the kernel part became
          infinitely fast. What is the largest overall speed-up, as a factor?
        </>
      </Translated>
    ),
    answer: DECODE_LIMIT,
    tol: 0.001,
    unit: '×',
    explain: (
      <Translated
        vi={
          <>
            <p>
              10% kia vẫn còn đó, nên thời gian chạy chỉ có thể rút xuống một phần mười và không hơn: 1 / 0.1 ={' '}
              {DECODE_LIMIT}. Sách nêu ý này bằng một trường hợp khắc nghiệt hơn: khi 30% của một lần chạy được làm
              nhanh hơn 100 lần, cả lần chạy chỉ nhanh hơn {BOOK_CASE.toFixed(2)} lần, và với 30% nhanh vô hạn thì là{' '}
              {BOOK_LIMIT.toFixed(2)}. Cách đặt vấn đề theo bước decode là của khóa học này; phép tính là của sách.
            </p>
            <p>
              Hãy giữ cuốn sổ cái này mở suốt chương. Các lần sao chép giữa host và device mà bạn sắp viết không phải
              là việc của kernel. Chúng nằm ở phía 10% và làm phần đó lớn thêm, đó là lý do sách cảnh báo rằng chương
              trình nó xây dựng ở đây là một khuôn mẫu kém hiệu quả (
              <a href={href('ch02', '2.3')}>mục 2.3</a>).
            </p>
          </>
        }
      >
        <>
          <p>
            The 10% is still there, so the run can shrink to one tenth of its time and no further: 1 / 0.1 ={' '}
            {DECODE_LIMIT}. The book makes the point with a harsher case: when 30% of a run is made 100 times faster,
            the whole run is only {BOOK_CASE.toFixed(2)} times faster, and with an infinitely fast 30% it is{' '}
            {BOOK_LIMIT.toFixed(2)}. The decode-step framing is ours; the arithmetic is the book's.
          </p>
          <p>
            Keep this ledger open for the whole chapter. The copies between host and device that you are about to
            write are not kernel work. They sit on the 10% side and make it larger, which is why the book warns that
            the program it builds here is inefficient as a pattern (
            <a href={href('ch02', '2.3')}>section 2.3</a>).
          </p>
        </>
      </Translated>
    ),
    source: 'ch. 1 §1.3, p. 9',
  },
  'ch02/start/q-design': {
    kind: 'choice',
    id: 'ch02/start/q-design',
    prompt: (
      <Translated
        vi={
          <>
            Một lõi CPU được xây dựng để hoàn thành công việc của một luồng nhanh nhất có thể. Một GPU được xây dựng
            để hoàn thành nhiều việc nhất mỗi giây trên rất nhiều luồng, mỗi luồng chậm hơn. Khối lượng công việc nào
            thuộc về phía GPU?
          </>
        }
      >
        <>
          A CPU core is built to finish one thread's work as soon as possible. A GPU is built to finish the most work
          per second across very many threads, each of them slower. Which workload belongs on the GPU side?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một chuỗi dài các bước, trong đó bước nào cũng cần kết quả của bước trước.">
            One long chain of steps in which every step needs the result of the one before.
          </Translated>
        ),
        why: (
          <Translated vi="Không bao giờ có quá một việc sẵn sàng để chạy, nên chỉ tốc độ của một luồng là quan trọng. Đó là thứ một lõi CPU được xây dựng để làm.">
            There is never more than one thing ready to run, so only the speed of a single thread matters. That is what a CPU core is built for.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cùng một phép tính nhỏ trên hàng triệu phần tử không cần đến nhau.">
            The same small computation over millions of elements that do not need each other.
          </Translated>
        ),
        correct: true,
        why: (
          <Translated vi="Hàng triệu phần việc sẵn sàng cùng lúc, nên thông lượng là yếu tố quyết định.">
            Millions of pieces of work are ready at once, so throughput decides.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một nhóm nhỏ các tác vụ lớn, độc lập, mỗi tác vụ rẽ nhánh nhiều và khó đoán.">
            A handful of large, independent tasks, each with heavy and unpredictable branching.
          </Translated>
        ),
        why: (
          <Translated vi="Ít luồng, cùng bộ dự đoán rẽ nhánh và bộ nhớ đệm (cache) lớn phục vụ chúng, chính là thứ CPU đã dành diện tích chip cho.">
            A few threads, and the branch prediction and large caches that serve them are what the CPU spent its area on.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Bất kỳ trường hợp nào trong ba, vì một GPU có số đơn vị số học gấp nhiều lần một lõi CPU.">
            Any of the three, since a GPU has many times the arithmetic units of a CPU core.
          </Translated>
        ),
        why: (
          <Translated vi="Với một hoặc vài luồng, CPU thắng. Sách nói rõ rằng ứng dụng nên dùng cả hai.">
            With one or a few threads a CPU wins. The book is explicit that applications should use both.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Khối lượng công việc thứ hai có tên gọi, song song dữ liệu (data parallelism), và đó là chủ đề của{' '}
            <a href={href('ch02', '2.1')}>mục 2.1</a>. Trường hợp thứ nhất là thứ nên học cách nhận ra sớm: GPU không
            có gì để giúp một chuỗi. Vòng lặp sinh của một mô hình ngôn ngữ chính là một chuỗi như vậy, từ token này
            sang token kế tiếp, đó là lý do các kernel bạn viết tăng tốc công việc bên trong một bước chứ không giảm số
            bước.
          </p>
        }
      >
        <p>
          The second workload has a name, data parallelism, and it is the subject of{' '}
          <a href={href('ch02', '2.1')}>section 2.1</a>. The first is the case to learn to recognise early: a GPU has
          nothing to offer a chain. The generation loop of a language model is such a chain from one token to the
          next, which is why the kernels you write speed up the work inside a step and not the number of steps.
        </p>
      </Translated>
    ),
    source: 'ch. 1 §1.1, pp. 3–6 (the generation-loop remark is ours)',
    vi: { source: 'ch. 1 §1.1, pp. 3–6 (nhận xét về vòng lặp sinh là của khóa học này)' },
  },
  'ch02/start/q-challenge': {
    kind: 'choice',
    id: 'ch02/start/q-challenge',
    prompt: (
      <Translated
        vi={
          <>
            Chương 1 nêu bốn khó khăn của lập trình song song. Một phép cộng vector được viết hoàn hảo,{' '}
            <code>C[i] = A[i] + B[i]</code> với một luồng cho mỗi <code>i</code>, sẽ gặp khó khăn nào trong số đó trước
            tiên?
          </>
        }
      >
        <>
          Chapter 1 names four difficulties of parallel programming. A perfectly written vector addition,{' '}
          <code>C[i] = A[i] + B[i]</code> with one thread for each <code>i</code>, runs into one of them first. Which?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Khối lượng thuật toán: phiên bản song song thực hiện nhiều phép toán hơn vòng lặp.">
            Algorithm work: the parallel version performs more operations than the loop.
          </Translated>
        ),
        why: (
          <Translated vi="Nó thực hiện đúng n phép cộng, bằng với vòng lặp.">
            It performs exactly n additions, the same as the loop.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Thực thi bị giới hạn bởi bộ nhớ: di chuyển các byte mất nhiều thời gian hơn các phép cộng.">
            Memory-bound execution: moving the bytes takes longer than the additions.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nhạy cảm với dữ liệu đầu vào: một số phần tử tốn kém hơn số khác, nên các luồng xong không đều nhau.">
            Input sensitivity: some elements cost more than others, so the threads finish unevenly.
          </Translated>
        ),
        why: (
          <Translated vi="Phần tử nào cũng tốn như nhau, bất kể giá trị là gì.">
            Every element costs the same, whatever the values are.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chi phí đồng bộ hóa: các luồng mất thời gian chờ nhau tại các rào chắn (barrier).">
            Synchronisation overhead: threads spend their time waiting for each other at barriers.
          </Translated>
        ),
        why: (
          <Translated vi="Không luồng nào cần gì từ luồng khác. Sách gọi loại công việc như vậy là song song hiển nhiên (embarrassingly parallel).">
            No thread needs anything from another thread. The book calls such work embarrassingly parallel.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi luồng thực hiện một phép cộng và ba lần truy cập bộ nhớ, mỗi lần 4 byte: hai lần nạp và một lần ghi. Tức
            là 12 byte được di chuyển cho một phép toán, nên kernel chỉ có thể chạy nhanh bằng tốc độ các byte đi giữa
            bộ nhớ của GPU và các đơn vị số học của nó. Bạn sẽ viết kernel này trong chương này và đo nó. Phần phân tích
            cho bạn biết nó còn cách giới hạn bao xa nằm ở <a href={href('ch05', '5.1')}>chương 5, mục 5.1</a>.
          </p>
        }
      >
        <p>
          Each thread does one addition and three memory accesses of 4 bytes each: two loads and a store. That is 12
          bytes moved for one operation, so the kernel can go only as fast as bytes travel between the GPU's memory and
          its arithmetic units. You will write this kernel in this chapter and measure it. The analysis that tells you
          how far from the limit it is comes in <a href={href('ch05', '5.1')}>chapter 5, section 5.1</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 1 §1.4, pp. 11–12; the 12 bytes per operation are from ch. 5 §5.1, p. 96',
    vi: { source: 'ch. 1 §1.4, pp. 11–12; con số 12 byte mỗi phép toán lấy từ ch. 5 §5.1, p. 96' },
  },
  'ch02/start/q-speedup': {
    kind: 'numeric',
    id: 'ch02/start/q-speedup',
    prompt: (
      <Translated vi={<>Một công việc mất 200 s trên hệ thống A và 10 s trên hệ thống B. Mức tăng tốc của B so với A là bao nhiêu?</>}>
        <>A job takes 200 s on system A and 10 s on system B. What is the speed-up of B over A?</>
      </Translated>
    ),
    answer: 20,
    unit: '×',
    explain: (
      <Translated
        vi={
          <p>
            Thời gian trên hệ thống chậm hơn chia cho thời gian trên hệ thống nhanh hơn: 200 / 10 = 20. Sách thêm một
            lời lưu ý áp dụng cho mọi con số như vậy. Con số đó còn phản ánh hệ thống A kém phù hợp với công việc đến
            mức nào, nên một con số lớn có thể có nghĩa là một kernel GPU tốt hoặc một mốc so sánh CPU kém.{' '}
            <a href={href('ch02', 'labs')}>Lab 2.1</a> yêu cầu bạn dự đoán đúng một tỉ lệ như vậy cho máy của mình
            trước khi đo.
          </p>
        }
      >
        <p>
          Time on the slower system divided by time on the faster one: 200 / 10 = 20. The book adds a caution that
          applies to every such figure. It also reflects how badly system A suited the job, so a large number can mean
          a good GPU kernel or a poor CPU baseline. <a href={href('ch02', 'labs')}>Lab 2.1</a> asks you to predict
          exactly such a ratio for your own machine before you measure it.
        </p>
      </Translated>
    ),
    source: 'ch. 1 §1.3, pp. 9–10',
  },
};

export function S20() {
  // The kernel and the comment above it, numbered as in the file (components/Code.tsx, lib/excerpt.ts).
  const kernel = useLabExcerpt(LAB1, 'vecAddKernel');
  return (
    <>
      <Translated
        vi={
          <p>
            Trong PyTorch, <code>c = a + b</code> trên hai tensor CUDA chỉ là một dòng. Bên dưới nó, và bên dưới lời
            gọi <code>.to("cuda")</code> đã đưa <code>a</code> và <code>b</code> lên GPU ngay từ đầu, là bốn thứ bạn
            chưa bao giờ phải tự viết: bộ nhớ được cấp phát trên device (thiết bị), một lần sao chép dữ liệu qua ranh
            giới giữa bộ nhớ của CPU và bộ nhớ của GPU, một hàm thực hiện phép cộng trên GPU, và một lời gọi khởi chạy
            các luồng (thread) GPU chạy hàm đó. Chương này tự tay viết bốn thứ ấy, ở dạng đơn giản nhất: mỗi phần tử
            một luồng.
          </p>
        }
      >
        <p>
          In PyTorch, <code>c = a + b</code> on two CUDA tensors is one line. Under it, and under the{' '}
          <code>.to("cuda")</code> that put <code>a</code> and <code>b</code> on the GPU in the first place, are four
          things you have never had to write: memory allocated on the device, a copy of data across the boundary between
          the CPU's memory and the GPU's, a function that does the addition on the GPU, and a call that starts the GPU
          threads that run it. This chapter writes those four things by hand, in their simplest form: one thread for
          every element.
        </p>
      </Translated>
      <Source pages="21" />
      <Translated
        vi={
          <p>
            Sách xây dựng một chương trình hoàn chỉnh cộng hai vector. Nó chia chương trình giữa CPU, gọi là host, và
            GPU, gọi là device. Nó cấp phát bộ nhớ trên device, sao chép dữ liệu đầu vào vào, viết hàm mà mọi luồng
            chạy, khởi chạy các luồng, rồi sao chép kết quả ra. Sau đó bạn có thể viết và khởi chạy bất kỳ kernel theo
            từng phần tử một chiều nào, chọn kích thước lần khởi chạy cho mọi độ dài, và nói chính xác có bao nhiêu
            luồng chạy và bao nhiêu luồng trong số đó làm việc có ích.
          </p>
        }
      >
        <p>
          The book builds one complete program that adds two vectors. It splits the program between the CPU, called
          the host, and the GPU, called the device. It allocates device memory, copies the inputs in, writes the
          function every thread runs, launches the threads, and copies the result out. After it you can write and
          launch any one-dimensional elementwise kernel, size its launch for any length, and say exactly how many
          threads run and how many of them do useful work.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Mối liên hệ với PyTorch là của khóa học này. Hai phần của nó đã được đọc trong mã nguồn PyTorch: bộ cấp
              phát CUDA của nó kết thúc ở <code>cudaMalloc</code>, bên dưới một lớp bộ nhớ đệm khiến phần lớn tensor
              không bao giờ chạm tới đó, và <code>tensor.to("cuda")</code> kết thúc ở một lần sao chép bộ nhớ loại từ
              host sang device. Cả hai lời gọi đều được dạy ở <a href={href('ch02', '2.4')}>mục 2.4</a>. Cách kernel
              cộng của chính PyTorch chia các phần tử cho các luồng không nằm trong lần đọc đó. Đã đọc ngày 2026-11-20
              (ch02-facts D4).
            </p>
          }
        >
          <p>
            The PyTorch connection is ours. Two parts of it were read in PyTorch's source: its CUDA allocator ends in{' '}
            <code>cudaMalloc</code>, under a caching layer so that most tensors never reach it, and{' '}
            <code>tensor.to("cuda")</code> ends in a memory copy of the host-to-device kind. Both calls are taught in{' '}
            <a href={href('ch02', '2.4')}>section 2.4</a>. How PyTorch's own addition kernel shares the elements out
            among threads was not part of that reading. Read on 2026-11-20 (ch02-facts D4).
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Kernel bạn sẽ viết được">The kernel you will be able to write</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đây là nơi chương này đi đến. Mọi luồng chạy hàm này một lần. Nó tìm ra phần tử nào là của mình, kiểm tra
            rằng phần tử đó tồn tại, và cộng một cặp số.
          </p>
        }
      >
        <p>
          This is where the chapter ends up. Every thread runs this function once. It works out which element is its
          own, checks that the element exists, and adds one pair of numbers.
        </p>
      </Translated>
      <Source section="2.5" pages="35" figs={tr('Fig. 2.10', 'Hình 2.10')} />
      <Translated
        vi={
          <p>
            Đừng nghiên cứu nó vội. Chỉ cần để ý nó ngắn đến mức nào, và không có vòng lặp nào trong đó. Màu trên các
            tên cho biết mỗi phần dữ liệu nằm ở đâu, trong đoạn mã này và trong mọi đoạn mã khác của khóa học.
          </p>
        }
      >
        <p>
          Do not study it yet. Notice only how little there is, and that there is no loop anywhere in it. The colours
          on the names say where each piece of data lives, in this listing and in every other listing of the course.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              <b>Màu trong khóa học này</b> cho biết dữ liệu nằm ở đâu. Chương này dùng hai trong bốn màu:{' '}
              <span className="chip reg">thanh ghi (register)</span> cho các biến riêng của một luồng và{' '}
              <span className="chip global">bộ nhớ toàn cục (global memory)</span> cho các mảng trên device. Màu đỏ
              luôn có nghĩa là có gì đó thất bại.
            </p>
          }
        >
          <p>
            <b>Colour in this course</b> means where data lives. This chapter uses two of the four:{' '}
            <span className="chip reg">register</span> for a thread's private variables and{' '}
            <span className="chip global">global memory</span> for arrays on the device. Red always means something
            failed.
          </p>
        </Translated>
      </Aside>
      <Code
        tabs={[{ label: 'CUDA', code: kernel?.text ?? '// loading labs/' + LAB1, tiers: VEC_TIERS }]}
        firstLine={kernel?.firstLine ?? 1}
        caption={
          <Translated
            vi={
              <>
                Kernel từ lời giải tham chiếu (solution) của lab 2.1, <code>labs/{LAB1}</code>, hiển thị không kèm mã
                host xung quanh. Đó là Hình 2.10 của sách với <code>const</code> được thêm vào hai mảng chỉ được đọc.
                Tên các mảng mang màu của bộ nhớ toàn cục, còn <code>i</code> mang màu của một thanh ghi.
              </>
            }
          >
            <>
              The kernel from the reference solution of lab 2.1, <code>labs/{LAB1}</code>, shown without the host code
              around it. It is the book's Fig. 2.10 with <code>const</code> added to the two arrays that are only read.
              The array names carry the colour of global memory and <code>i</code> the colour of a register.
            </>
          </Translated>
        }
      />

      <h2>
        <Translated vi="Lộ trình qua chương">The road through the chapter</Translated>
      </h2>
      <ul>
        <Translated
          vi={
            <li>
              <b>Song song dữ liệu</b>: tính chất của một phép tính cho phép mỗi đầu ra một luồng, và cách kiểm tra
              nó (<a href={href('ch02', '2.1')}>mục 2.1</a>).
            </li>
          }
        >
          <li>
            <b>Data parallelism</b>: the property of a computation that makes one thread per output possible, and how
            to check for it (<a href={href('ch02', '2.1')}>section 2.1</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Host và device</b>: một luồng host khởi chạy các lưới (grid) luồng device, và điều gì đã xảy ra khi
              một câu lệnh khởi chạy trả về (<a href={href('ch02', '2.2')}>mục 2.2</a>).
            </li>
          }
        >
          <li>
            <b>Host and device</b>: one host thread that launches grids of device threads, and what has happened when
            a launch statement returns (<a href={href('ch02', '2.2')}>section 2.2</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Hàm stub phía host</b>: ba phần của hàm host bao quanh một kernel, và vì sao các lượt đi về giữa hai
              bộ nhớ lại tốn kém (<a href={href('ch02', '2.3')}>mục 2.3</a>).
            </li>
          }
        >
          <li>
            <b>The host stub</b>: the three parts of the host function around a kernel, and why round trips between
            the two memories cost (<a href={href('ch02', '2.3')}>section 2.3</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Bộ nhớ device và truyền dữ liệu</b>: cấp phát, sao chép theo từng chiều, giải phóng, kiểm tra lỗi (
              <a href={href('ch02', '2.4')}>mục 2.4</a>).
            </li>
          }
        >
          <li>
            <b>Device memory and transfers</b>: allocating, copying in each direction, freeing, checking errors (
            <a href={href('ch02', '2.4')}>section 2.4</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Kernel và chỉ số của nó</b>: cách một luồng tìm ra phần tử của mình, và phép kiểm tra chặn các luồng
              dư (<a href={href('ch02', '2.5')}>mục 2.5</a>).
            </li>
          }
        >
          <li>
            <b>The kernel and its index</b>: how a thread finds its element, and the test that stops surplus threads (
            <a href={href('ch02', '2.5')}>section 2.5</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Lần khởi chạy</b>: hai con số trong câu lệnh khởi chạy và một độ dài cho trước cần bao nhiêu khối (
              <a href={href('ch02', '2.6')}>mục 2.6</a>).
            </li>
          }
        >
          <li>
            <b>The launch</b>: the two numbers in the launch statement and how many blocks a given length needs (
            <a href={href('ch02', '2.6')}>section 2.6</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Trình biên dịch làm gì</b> với một tệp chứa mã cho hai bộ xử lý (
              <a href={href('ch02', '2.7')}>mục 2.7</a>).
            </li>
          }
        >
          <li>
            <b>What the compiler does</b> with a file that holds code for two processors (
            <a href={href('ch02', '2.7')}>section 2.7</a>).
          </li>
        </Translated>
      </ul>

      <h2>
        <Translated vi="Cùng chương trình trên GPU AMD và Apple">The same program on AMD and Apple GPUs</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trên HIP của AMD, chương trình giữ nguyên cấu trúc này: kernel được viết y hệt, và các lời gọi cấp phát và
            sao chép là cùng những lời gọi đó với <code>hip</code> thay cho <code>cuda</code>. Trên Apple silicon, một
            buffer Metal ở chế độ lưu trữ dùng chung là một vùng nhớ duy nhất mà cả CPU và GPU đều thấy. Khi dữ liệu
            được ghi vào một buffer như vậy ngay từ đầu, hai bước sao chép không còn gì để làm và biến mất.
          </p>
        }
      >
        <p>
          On AMD's HIP the program keeps this structure exactly: the kernel is spelled the same, and the allocation
          and copy calls are the same calls with <code>hip</code> in place of <code>cuda</code>. On Apple silicon a
          Metal buffer in shared storage is one piece of memory that the CPU and the GPU both see. When the data is
          written into such a buffer from the start, the two copy steps have nothing to do and disappear.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ đề cập CUDA. Mỗi mục của chương này đều nói HIP và Metal khác ở đâu, đã đối chiếu với tài liệu
              của AMD và Apple ngày 2026-11-20 (ở đây: ch02-facts B1, B4, C3; ch05-facts C6).
            </p>
          }
        >
          <p>
            The book covers CUDA only. Each section of this chapter says where HIP and Metal differ, checked against
            AMD's and Apple's documentation on 2026-11-20 (here: ch02-facts B1, B4, C3; ch05-facts C6).
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Các lab yêu cầu gì ở bạn">What the labs ask of you</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trong các lab này, phía host cũng là của bạn chứ không chỉ kernel: cấp phát, cả hai lần sao chép, lần khởi
            chạy, các bước kiểm tra lỗi và giải phóng. Có <a href={href('ch02', 'labs')}>bốn lab</a>:
          </p>
        }
      >
        <p>
          In these labs the host side is yours as well as the kernel: allocation, both copies, the launch, the error
          checks and the free. There are <a href={href('ch02', 'labs')}>four labs</a>:
        </p>
      </Translated>
      <ul>
        <li>
          <Translated vi="2.1, phép cộng vector từ lần cấp phát đầu tiên đến lần giải phóng cuối cùng;">
            2.1, vector addition from the first allocation to the last free;
          </Translated>
        </li>
        <li>
          <Translated vi="2.2, thời gian đi đâu giữa các lần sao chép và kernel;">
            2.2, where the time goes between copies and kernel;
          </Translated>
        </li>
        <li>
          <Translated vi="2.3, mốc dự án M0 của dự án xuyên suốt khóa học: các phép toán theo từng phần tử mà một transformer cần (cộng bias, cộng residual, nhân hệ số);">
            2.3, milestone M0 of the course project: the elementwise operations a transformer needs (bias add,
            residual add, scale);
          </Translated>
        </li>
        <li>
          <Translated vi="2.4, săn lỗi trong một hàm stub phía host gần như chạy được.">
            2.4, a bug hunt in a host stub that almost works.
          </Translated>
        </li>
      </ul>

      <h2>
        <Translated vi="Kiểm tra lại chương 1">A check on chapter 1</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bốn câu hỏi về những gì chương này giả định từ chương 1. Chương đó không có mã, nên các câu hỏi này là về
            việc suy luận với các con số và từ vựng của nó. Sai ở đây bây giờ còn hơn sai về sau; mỗi câu sẽ quay lại
            trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Four questions on what this chapter assumes from chapter 1. That chapter has no code, so these are about
          reasoning with its numbers and its vocabulary. They are worth getting wrong now rather than later; each one
          comes back in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={questionsS20['ch02/start/q-amdahl']} label={tr('From chapter 1', 'Từ chương 1')} />
      <Ask q={questionsS20['ch02/start/q-design']} label={tr('From chapter 1', 'Từ chương 1')} />
      <Ask q={questionsS20['ch02/start/q-challenge']} label={tr('From chapter 1', 'Từ chương 1')} />
      <Ask q={questionsS20['ch02/start/q-speedup']} label={tr('From chapter 1', 'Từ chương 1')} />
    </>
  );
}
