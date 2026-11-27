import { Ask, type Question } from '../../components/Ask';
import { Aside, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { fmt, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { BOOK_CASES, ceiling, remaining, speedup, untouchedShare } from '../../sim/amdahl';
import { chaptersBefore, goal, leadsTo, prerequisites } from '../../sim/bookmap';
import { design, finish } from '../../sim/designs';
import { BUDGET } from './TwoDesignsScene';

// Chapter 1 has no code, so this page shows no kernel and needs no snippet file.
// Every number its questions use comes from the models the sections run on;
// Wrap.test.ts pins them.

const n = (v: number) => v.toLocaleString('en-US', { maximumFractionDigits: 2 });
const and = (xs: (string | number)[]) =>
  xs.length <= 1 ? xs.join('') : xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1];

// ---- q-reprofile: Amdahl's law applied twice (sim/amdahl.ts).
export const PROFILE = { p: 0.9, s: 9 };
const LEFT = remaining(PROFILE.p, PROFILE.s);
const HOST_SHARE = untouchedShare(PROFILE.p, PROFILE.s);
const REACHED = speedup(PROFILE.p, PROFILE.s);
const TOP = ceiling(PROFILE.p);

// ---- q-both: the two designs of the start page on a job with two phases (sim/designs.ts).
const FAST = design(BUDGET, 2);
const MANY = design(BUDGET, 0);
export const JOB = { chain: 1200, crowd: 1024, length: 100 };
const last = (d: typeof FAST, chains: number, length: number) => finish(d, chains, length).last;
const CHAIN = { fast: last(FAST, 1, JOB.chain), many: last(MANY, 1, JOB.chain) };
const CROWD = { fast: last(FAST, JOB.crowd, JOB.length), many: last(MANY, JOB.crowd, JOB.length) };
const ALONE_FAST = CHAIN.fast + CROWD.fast;
const ALONE_MANY = CHAIN.many + CROWD.many;
const SPLIT = CHAIN.fast + CROWD.many;
const WRONG_WAY = CHAIN.many + CROWD.fast;
const CROWD_WAVES = { fast: finish(FAST, JOB.crowd, JOB.length).waves, many: finish(MANY, JOB.crowd, JOB.length).waves };

// ---- q-skip9: what the book's arrows put behind chapter 9 (sim/bookmap.ts).
const SKIPPED = 9;
const BEHIND_SKIPPED = leadsTo([SKIPPED]);
const SAMPLING = goal('sampling').chapters;
const SAMPLING_BLOCKED = SAMPLING.filter((c) => prerequisites([c]).includes(SKIPPED));
const SAMPLING_FREE = SAMPLING.filter((c) => !SAMPLING_BLOCKED.includes(c));

// ---- the reading-path figures quoted under "What to carry forward".
const BEFORE_MATMUL = chaptersBefore([15]);
const BEFORE_ATTENTION = chaptersBefore([20]);

export const questionsWrap: Record<string, Question> = {
  'ch01/wrap/q-reprofile': {
    kind: 'numeric',
    id: 'ch01/wrap/q-reprofile',
    prompt: (
      <Translated
        vi={
          <>
            Một bản profile của một bước decode cho thấy {n(PROFILE.p * 100)}% thời gian nằm trong các kernel và phần còn lại ở
            host. Bạn làm các kernel nhanh hơn {PROFILE.s} lần, không đổi gì ở host, rồi profile lại. Phần host giờ chiếm bao
            nhiêu phần thời gian chạy?
          </>
        }
      >
        <>
          A profile of a decode step shows {n(PROFILE.p * 100)}% of the time in kernels and the rest on the host. You
          make the kernels {PROFILE.s} times faster, change nothing on the host, and profile again. What share of the
          run time does the host part take now?
        </>
      </Translated>
    ),
    answer: 50,
    tol: 0.005,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            Các kernel co từ {n(PROFILE.p * 100)}% thời gian cũ xuống {fmt(LEFT.accelerated * 100)}%. Phần host vẫn là{' '}
            {fmt(LEFT.untouched * 100)}% thời gian cũ, nên bước này giờ mất {fmt(LEFT.total * 100)}% so với trước (
            {fmt(REACHED)}× tổng thể), và {fmt(LEFT.untouched * 100)} trong số {fmt(LEFT.total * 100)} điểm phần trăm đó là của
            host: {fmt(HOST_SHARE * 100)}%. Trần cho bước này là {fmt(TOP)}×. Bạn mới đi được nửa đường, và mọi công sức tối ưu
            kernel còn lại cũng chỉ có thể gấp đôi tốc độ thêm một lần nữa là cùng. Một bản profile mô tả chương trình như nó
            là lúc bạn đo. Sau mỗi bước có hiệu quả, hãy profile lại trước khi chọn bước tiếp theo.
          </p>
        }
      >
        <p>
          The kernels shrink from {n(PROFILE.p * 100)}% of the old time to {fmt(LEFT.accelerated * 100)}%. The host part
          is still {fmt(LEFT.untouched * 100)}% of the old time, so the step now takes {fmt(LEFT.total * 100)}% of what
          it did ({fmt(REACHED)}× overall), and {fmt(LEFT.untouched * 100)} of those {fmt(LEFT.total * 100)} points are
          the host's: {fmt(HOST_SHARE * 100)}%. The ceiling for this step was {fmt(TOP)}×. You are halfway
          to it, and all the kernel work still to come can at most double the speed again. A profile describes the
          program as it was when you took it. After every step that works, take another before you choose the next.
        </p>
      </Translated>
    ),
    source: "our numbers; the law is the book's, ch. 1 §1.3, p. 9",
    vi: { source: 'số liệu của chúng tôi; định luật là của sách, ch. 1 §1.3, p. 9' },
  },
  'ch01/wrap/q-both': {
    kind: 'choice',
    id: 'ch01/wrap/q-both',
    prompt: (
      <Translated
        vi={
          <>
            Một công việc có hai pha. Đầu tiên là một chuỗi {n(JOB.chain)} phép toán, mỗi phép cần kết quả của phép trước. Sau
            đó là một đám đông: {n(JOB.crowd)} chuỗi độc lập, mỗi chuỗi {JOB.length} phép toán. Một máy có cả hai thiết kế của
            trang bắt đầu: A với {FAST.units} đơn vị tốc độ {FAST.speed}, và B với {MANY.units} đơn vị tốc độ {MANY.speed}.
            Trong mô hình của trang đó, cách bố trí nào xong công việc trước?
          </>
        }
      >
        <>
          A job has two phases. First one chain of {n(JOB.chain)} operations, each needing the result of the one before.
          Then a crowd: {n(JOB.crowd)} independent chains of {JOB.length} operations each. A machine has both designs of the
          start page: A with {FAST.units} units of speed {FAST.speed}, and B with {MANY.units} units of speed{' '}
          {MANY.speed}. In that page's model, which arrangement finishes the job first?
        </>
      </Translated>
    ),
    options: [
      {
        get label() {
          return tr(
            'Both phases on A.',
            'Cả hai pha trên A.',
          );
        },
        get why() {
          return tr(
            `A finishes the chain at ${n(CHAIN.fast)}. Then it needs ${n(CROWD_WAVES.fast)} waves for the crowd, ${n(CROWD.fast)} more: ${n(ALONE_FAST)} in all.`,
            `A xong chuỗi lúc ${n(CHAIN.fast)}. Sau đó nó cần ${n(CROWD_WAVES.fast)} đợt cho đám đông, thêm ${n(CROWD.fast)}: tổng cộng ${n(ALONE_FAST)}.`,
          );
        },
      },
      {
        get label() {
          return tr(
            'Both phases on B.',
            'Cả hai pha trên B.',
          );
        },
        get why() {
          return tr(
            `B clears the crowd in ${n(CROWD_WAVES.many)} waves, but the single chain keeps one slow unit busy until ${n(CHAIN.many)}: ${n(ALONE_MANY)} in all.`,
            `B xử lý xong đám đông trong ${n(CROWD_WAVES.many)} đợt, nhưng chuỗi đơn giữ một đơn vị chậm bận đến ${n(CHAIN.many)}: tổng cộng ${n(ALONE_MANY)}.`,
          );
        },
      },
      {
        get label() {
          return tr(
            'The chain on A, then the crowd on B.',
            'Chuỗi trên A, rồi đám đông trên B.',
          );
        },
        correct: true,
      },
      {
        get label() {
          return tr(
            'The chain on B, then the crowd on A.',
            'Chuỗi trên B, rồi đám đông trên A.',
          );
        },
        get why() {
          return tr(
            `That gives each phase to the design that suits it least: ${n(CHAIN.many)} + ${n(CROWD.fast)} = ${n(WRONG_WAY)}.`,
            `Cách đó giao mỗi pha cho thiết kế ít hợp với nó nhất: ${n(CHAIN.many)} + ${n(CROWD.fast)} = ${n(WRONG_WAY)}.`,
          );
        },
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            A xong chuỗi lúc {n(CHAIN.fast)}. Rồi B xử lý xong đám đông trong {n(CROWD_WAVES.many)} đợt, thêm {n(CROWD.many)}:{' '}
            {n(SPLIT)}, so với {n(ALONE_MANY)} khi chỉ dùng B và {n(ALONE_FAST)} khi chỉ dùng A. Đây là lý do sách lập trình hai
            bộ xử lý cùng nhau: với một hoặc vài luồng thì CPU thắng, với nhiều luồng thì GPU thắng, nên các phần tuần tự ở lại
            trên host và các phần nặng về tính toán số đi sang device. Đó cũng là định luật Amdahl nhìn từ phía phần cứng. Khi
            chỉ dùng B, chuỗi chiếm {n(CHAIN.many)} trong {n(ALONE_MANY)}, và không số đơn vị nào rút ngắn được nó. Mô hình bỏ
            qua cái giá của một sự phân chia thật: di chuyển dữ liệu giữa hai bên. Chương 2 làm cho cái giá đó hiện ra.
          </p>
        }
      >
        <p>
          A finishes the chain at {n(CHAIN.fast)}. B then clears the crowd in {n(CROWD_WAVES.many)} waves,{' '}
          {n(CROWD.many)} more: {n(SPLIT)}, against {n(ALONE_MANY)} for B alone and {n(ALONE_FAST)} for A alone. This is
          the book's reason for programming the two processors together: with one or a few threads the CPU wins, with
          many the GPU wins, so the sequential parts stay on the host and the numerically heavy parts go to the device.
          It is also Amdahl's law seen from the hardware side. On B alone the chain is {n(CHAIN.many)} of the{' '}
          {n(ALONE_MANY)}, and no number of units shortens it. The model leaves out what a real split costs: moving the
          data between the two. Chapter 2 makes that cost visible.
        </p>
      </Translated>
    ),
    source: "our model of the book's argument, ch. 1 §1.1, pp. 5–6",
    vi: { source: 'mô hình của chúng tôi cho lập luận của sách, ch. 1 §1.1, pp. 5–6' },
  },
  'ch01/wrap/q-morethreads': {
    kind: 'choice',
    id: 'ch01/wrap/q-morethreads',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel khởi chạy đúng một luồng cho mỗi đơn vị số học của thiết bị. Mỗi luồng dành phần lớn thời gian để chờ, vì
            mỗi truy cập bộ nhớ mất rất lâu mới trả về. Bộ nhớ còn xa mới bão hòa: nó có thể cung cấp số byte mỗi giây gấp vài
            lần mức kernel này yêu cầu. Một thiết kế hướng thông lượng mong bạn thay đổi điều gì?
          </>
        }
      >
        <>
          A kernel launches exactly one thread per arithmetic unit of the device. Each thread spends most of its time
          waiting, because every memory access takes a long time to come back. Memory is far from saturated: it could
          deliver several times more bytes per second than this kernel asks for. What does a throughput-oriented design
          expect you to change?
        </>
      </Translated>
    ),
    options: [
      {
        get label() {
          return tr(
            'Nothing. A unit executes one thread at a time, so more threads than units cannot help.',
            'Không gì cả. Một đơn vị thực thi một luồng mỗi lúc, nên nhiều luồng hơn số đơn vị không giúp được gì.',
          );
        },
        get why() {
          return tr(
            'True at any one instant. But a waiting thread is not executing, and while it waits its unit could run another thread if there were one.',
            'Đúng ở bất kỳ khoảnh khắc nào. Nhưng một luồng đang chờ thì không thực thi, và trong lúc nó chờ, đơn vị của nó có thể chạy một luồng khác nếu có.',
          );
        },
      },
      {
        get label() {
          return tr(
            'Launch many more threads than there are units, so that a unit always has one that is ready.',
            'Khởi chạy nhiều luồng hơn hẳn số đơn vị, để một đơn vị luôn có một luồng sẵn sàng.',
          );
        },
        correct: true,
      },
      {
        get label() {
          return tr(
            "Launch fewer threads, so that each one's memory accesses come back sooner.",
            'Khởi chạy ít luồng hơn, để các truy cập bộ nhớ của mỗi luồng trả về sớm hơn.',
          );
        },
        get why() {
          return tr(
            'An access is slow here because the design allows it to be, not because the threads compete. Memory has throughput to spare.',
            'Ở đây một truy cập chậm vì thiết kế cho phép nó chậm, không phải vì các luồng tranh nhau. Bộ nhớ còn dư thông lượng.',
          );
        },
      },
      {
        get label() {
          return tr(
            'Make the arithmetic of each thread faster, so it reaches its next access sooner.',
            'Làm phép tính của mỗi luồng nhanh hơn, để nó đến lần truy cập kế tiếp sớm hơn.',
          );
        },
        get why() {
          return tr(
            'The units already sit idle most of the time. Shorter arithmetic only makes the waits a larger share.',
            'Các đơn vị vốn đã ngồi không phần lớn thời gian. Phép tính ngắn hơn chỉ làm thời gian chờ chiếm tỷ lệ lớn hơn.',
          );
        },
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Thiết kế mua thông lượng bằng cách để từng phép toán đơn lẻ mất nhiều thời gian, cả ở số học lẫn bộ nhớ. Điều đó chỉ
            có lợi nếu phần mềm cung cấp đủ luồng để lấp các khoảng chờ, và sách nói mã GPU được kỳ vọng có số luồng lớn chính vì
            lý do này. Hãy để ý điều kiện trong câu hỏi: bộ nhớ chưa bão hòa. Nếu nó bão hòa, byte không thể đến nhanh hơn dù số
            luồng là bao nhiêu, đó là phần phẳng của trình khám phá "Điều gì chặn trần kernel" ở mục 1.3. Độ trễ dài và thông
            lượng bão hòa là hai vấn đề khác nhau, và thêm luồng chỉ chữa được vấn đề thứ nhất. Chương 4 cho thấy phần cứng làm
            việc lấp khoảng chờ đó.
          </p>
        }
      >
        <p>
          The design buys throughput by letting single operations take long, in arithmetic and in memory. That pays off
          only if software supplies enough threads to fill the waits, and the book says GPU code is expected to have a
          large number of threads for exactly this reason. Notice the condition in the question: memory was not
          saturated. If it were, bytes could not arrive any faster whatever the number of threads, which is the flat
          part of the "What caps the kernel" explorer in section 1.3. A long latency and a saturated throughput are
          different problems, and more threads cure only the first. Chapter 4 shows the hardware that does the filling.
        </p>
      </Translated>
    ),
    source: 'ch. 1 §1.1, p. 5; the mechanism is ch. 4 §4.6, pp. 82–85',
    vi: { source: 'ch. 1 §1.1, p. 5; cơ chế ở ch. 4 §4.6, pp. 82–85' },
  },
  'ch01/wrap/q-skip9': {
    kind: 'choice',
    id: 'ch01/wrap/q-skip9',
    prompt: (
      <Translated
        vi={
          <>
            Để tiết kiệm thời gian, bạn quyết định bỏ qua chương {SKIPPED}, histogram. Theo các mũi tên tiên quyết của sách, mục
            tiêu nào của bạn giờ có một lỗ hổng phía trước?
          </>
        }
      >
        <>
          To save time you decide to skip chapter {SKIPPED}, histogram. By the book's prerequisite arrows, which of your
          goals now has a gap in front of it?
        </>
      </Translated>
    ),
    options: [
      {
        get label() {
          return tr(
            'Fast matrix multiplication, chapter 15.',
            'Phép nhân ma trận nhanh, chương 15.',
          );
        },
        get why() {
          return tr(
            `Chapter 15 needs chapters ${and(BEFORE_MATMUL)} and nothing else.`,
            `Chương 15 cần các chương ${BEFORE_MATMUL.join(', ')} và không gì khác.`,
          );
        },
      },
      {
        get label() {
          return tr(
            'Attention, chapter 20.',
            'Attention, chương 20.',
          );
        },
        get why() {
          return tr(
            `The chart's road to chapter 20 is chapters ${and(BEFORE_ATTENTION)}. Chapter ${SKIPPED} is on another branch.`,
            `Con đường của sơ đồ tới chương 20 là các chương ${BEFORE_ATTENTION.join(', ')}. Chương ${SKIPPED} nằm trên một nhánh khác.`,
          );
        },
      },
      {
        get label() {
          return tr(
            `Sampling, chapters ${and(SAMPLING)}.`,
            `Lấy mẫu, các chương ${SAMPLING.join(', ')}.`,
          );
        },
        correct: true,
      },
      {
        get label() {
          return tr(
            'None of them.',
            'Không mục tiêu nào.',
          );
        },
        get why() {
          return tr(
            `The chart puts ${BEHIND_SKIPPED.length} chapters behind chapter ${SKIPPED}: ${and(BEHIND_SKIPPED)}.`,
            `Sơ đồ đặt ${BEHIND_SKIPPED.length} chương phía sau chương ${SKIPPED}: ${BEHIND_SKIPPED.join(', ')}.`,
          );
        },
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Chương {SKIPPED} có một mũi tên vào chương 11 (scan), 11 có một mũi tên vào 12 (lọc), và 12 vào 14 (sắp xếp). Vì thế{' '}
            {SAMPLING_BLOCKED.length} trong {SAMPLING.length} chương mà mục tiêu lấy mẫu kết thúc ở đó nằm phía sau nó:{' '}
            {SAMPLING_BLOCKED.join(', ')}. Chương {SAMPLING_FREE.join(', ')} chỉ gắn với chương {SKIPPED} qua mũi tên nét đứt. Sơ
            đồ không đưa ra lý do. Lời nói đầu gợi ý một lý do: chương {SKIPPED} là nơi sách giới thiệu giao diện atomic của nó,
            và các chương scan và lọc đều dùng phép toán nguyên tử (atomic). Mối liên hệ đó là cách đọc của chúng tôi. Điểm
            chung là một sơ đồ tiên quyết trả lời hai câu hỏi: một chương cần gì, và bỏ qua một chương thì điều gì nằm ngoài tầm
            với.
          </p>
        }
      >
        <p>
          Chapter {SKIPPED} has an arrow into chapter 11 (scan), 11 has one into 12 (filter), and 12 into 14 (sort). So{' '}
          {SAMPLING_BLOCKED.length} of the {SAMPLING.length} chapters that the sampling goal ends at sit behind it:{' '}
          {and(SAMPLING_BLOCKED)}. Chapter {and(SAMPLING_FREE)} is tied to chapter {SKIPPED} only by the dashed arrow.
          The chart gives no reasons. The Preface suggests one: chapter {SKIPPED} is where the book introduces its
          atomics interface, and the scan and filter chapters both use atomic operations. That link is our reading. The
          general point is that a prerequisite chart answers two questions: what a chapter needs, and what skipping a
          chapter puts out of reach.
        </p>
      </Translated>
    ),
    source: 'Fig. P.1, p. xxiii; Preface, pp. xviii–xx; §1.7, p. 16',
    vi: { source: 'Fig. P.1, p. xxiii; Lời nói đầu, pp. xviii–xx; §1.7, p. 16' },
  },
  'ch01/wrap/q-sharedmemory': {
    kind: 'choice',
    id: 'ch01/wrap/q-sharedmemory',
    prompt: (
      <Translated
        vi={
          <>
            Công thức của sách để chuyển một phép tính sang GPU có sáu bước: (1) chọn phần cần song song hóa, (2) cấp phát bộ nhớ
            device, (3) chép dữ liệu sang device, (4) viết kernel, (5) khởi chạy nó, (6) chép kết quả về. Hình 1.1 của sách cho
            GPU một bộ nhớ riêng. Trên một máy mà CPU và GPU dùng chung một bộ nhớ, những bước nào mất đi lý do tồn tại của
            chúng?
          </>
        }
      >
        <>
          The book's recipe for moving a computation to a GPU has six steps: (1) pick the part to parallelise, (2)
          allocate device memory, (3) copy the data to the device, (4) write the kernel, (5) launch it, (6) copy the
          results back. Its Fig. 1.1 gives the GPU a memory of its own. On a machine where the CPU and the GPU share one
          memory, which steps lose the reason they were there for?
        </>
      </Translated>
    ),
    options: [
      {
        get label() {
          return tr(
            'Steps 2 and 5: with one memory there is nothing to allocate and nothing to launch.',
            'Bước 2 và 5: với một bộ nhớ thì không có gì để cấp phát và không có gì để khởi chạy.',
          );
        },
        get why() {
          return tr(
            'The GPU still needs a buffer it is allowed to use, and its threads still have to be started.',
            'GPU vẫn cần một buffer mà nó được phép dùng, và các luồng của nó vẫn phải được khởi động.',
          );
        },
      },
      {
        get label() {
          return tr(
            'Steps 3 and 6, the two copies: both processors read the same memory.',
            'Bước 3 và 6, hai lần sao chép: cả hai bộ xử lý đọc cùng một bộ nhớ.',
          );
        },
        correct: true,
      },
      {
        get label() {
          return tr(
            'Step 4: the kernel can then be an ordinary host function.',
            'Bước 4: khi đó kernel có thể là một hàm host bình thường.',
          );
        },
        get why() {
          return tr(
            'The kernel is the computation itself, written for many threads. Where the data lives does not change who runs it.',
            'Kernel là chính phép tính, được viết cho nhiều luồng. Dữ liệu nằm ở đâu không thay đổi ai chạy nó.',
          );
        },
      },
      {
        get label() {
          return tr(
            'None. The recipe does not depend on how memory is arranged.',
            'Không bước nào. Công thức không phụ thuộc vào cách bộ nhớ được bố trí.',
          );
        },
        get why() {
          return tr(
            'Two of the six steps do nothing except carry bytes from one memory to another.',
            'Hai trong sáu bước không làm gì ngoài việc mang byte từ bộ nhớ này sang bộ nhớ khác.',
          );
        },
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hai bộ nhớ nghĩa là đầu vào phải được mang sang và kết quả phải được mang về. Với một bộ nhớ mà cả hai bộ xử lý đều
            đánh địa chỉ được, một buffer có thể hiện ra với cả hai mà không cần chép: Apple ghi điều này trong tài liệu cho GPU
            của mình, nơi một buffer ở chế độ lưu trữ shared được "allocated in system memory" và CPU với GPU "share access" tới
            nó. Điều không mất đi là thứ tự: host không được đọc một kết quả trước khi công việc GPU tạo ra nó đã xong. Một
            chương trình vẫn có thể chép, khi dữ liệu của nó ban đầu nằm trong vùng nhớ không được cấp phát như một buffer như
            vậy. Khi đó đó là lựa chọn của chương trình, không phải đòi hỏi của phần cứng. Cấp phát, kernel và khởi chạy đều còn
            nguyên. Chương 2 bàn tiếp chuyện này.
          </p>
        }
      >
        <p>
          Two memories mean the inputs have to be carried across and the results carried back. With one memory that both
          processors can address, a buffer can be visible to both with no copy: Apple documents this for its GPUs, where
          a buffer in shared storage mode is "allocated in system memory" and the CPU and GPU "share access" to it. What
          does not go away is order: the host must not read a result before the GPU work that produces it has finished.
          A program may still copy, when its data starts out in memory that was not allocated as such a buffer. That is
          then a choice of the program, not a demand of the hardware. Allocation, the kernel and the launch all remain.
          Chapter 2 takes this up.
        </p>
      </Translated>
    ),
    source:
      'book §1.7, p. 15, and Fig. 1.1, p. 4; Apple documentation checked 2026-11-20 (notes/today/ch05-facts.md, C6; ch01-facts.md, C3)',
    vi: {
      source:
        'sách §1.7, p. 15, và Fig. 1.1, p. 4; tài liệu của Apple kiểm tra ngày 2026-11-20 (notes/today/ch05-facts.md, C6; ch01-facts.md, C3)',
    },
  },
};

// Backends that are not a GPU: the emulator and its race-detection flavour.
const EMULATED = new Set(['emu', 'race']);
const NAME: Record<string, string> = { metal: 'Metal', cuda: 'CUDA', hip: 'HIP' };
// The runtime writes device properties as strings ('1' or 'true'); booleans are accepted too.
const yes = (v: unknown) => v === true || v === '1' || v === 'true';

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const devices = Object.keys(reports).filter((b) => !EMULATED.has(b));
  const unified = reports.metal?.device?.hasUnifiedMemory;
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch01/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  return (
    <>
      <h2 style={{ marginTop: 0 }}><Translated vi="Những gì cần mang theo">What to carry forward</Translated></h2>
      <ul>
        <li>
          <Translated
            vi={
              <>
                CPU được xây dựng cho độ trễ của một luồng, GPU cho tổng thông lượng của rất nhiều luồng. Theo quy tắc chi phí của
                sách, độ trễ đắt hơn thông lượng rất nhiều, nên một ngân sách không mua được cả hai. Thiết kế nào xong một công việc
                trước phụ thuộc vào bao nhiêu phần của nó độc lập, không phụ thuộc vào nó chứa bao nhiêu phép tính.
              </>
            }
          >
            A CPU is built for the latency of one thread, a GPU for the total throughput of very many. By the book's cost
            rule, latency is far more expensive to buy than throughput, so one budget cannot buy both. Which design
            finishes a piece of work first depends on how many of its pieces are independent, not on how much arithmetic
            it holds.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Một bộ xử lý hướng thông lượng để từng phép toán đơn lẻ mất nhiều thời gian, nên nó mong có nhiều luồng hơn hẳn số
                nó có thể thực thi trong một khoảnh khắc. Các phần tuần tự ở lại trên host và các phần song song dữ liệu nặng đi sang
                device, đó là lý do mô hình lập trình có cả hai.
              </>
            }
          >
            A throughput-oriented processor lets single operations take a long time, so it expects many more threads
            than it can execute at one instant. Sequential parts stay on the host and the heavy data-parallel parts go
            to the device, which is why the programming model has both.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Tốc độ đỉnh mô tả phần cứng, không mô tả một chương trình. Trước khi dùng một con số như vậy, hãy tìm hiểu nó thuộc
                về loại đơn vị nào và nó giả định điều gì.
              </>
            }
          >
            A peak rate describes hardware, not a program. Before you use one, find out which kind of unit it belongs to
            and what it assumes.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Mức tăng tốc là một thời gian chia cho một thời gian khác, và nó nói về mốc so sánh nhiều ngang với về bản chuyển
                đổi. Nếu một phần p của lần chạy được làm nhanh hơn s lần, toàn bộ được lợi 1 / ((1 − p) + p / s) và không bao giờ
                hơn 1 / (1 − p). Hãy tính trần đó trước khi viết kernel. Sau đó profile lại, vì p đã thay đổi.
              </>
            }
          >
            A speed-up is one time divided by another, and it says as much about the baseline as about the port. If a
            fraction p of the run is made s times faster, the whole gains 1 / ((1 − p) + p / s) and never more than 1 /
            (1 − p). Work out that ceiling before you write the kernel. Profile again afterwards, because p has changed.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Một bản chuyển đổi thẳng thắn thường dừng lại ở chỗ bộ nhớ bão hòa. Con số của sách cho điểm đó, khoảng 10×, là một
                quy tắc kinh nghiệm. Vượt qua nó, lợi ích đến từ ít byte hơn trên mỗi phép toán, không phải từ phép tính nhanh hơn.
              </>
            }
          >
            A straightforward port tends to stop where memory saturates. The book's figure for that point, about 10×, is
            a rule of thumb. Past it the gain comes from fewer bytes per operation, not from faster arithmetic.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Bốn điều làm phiên bản nhanh trở nên khó: một thuật toán song song làm nhiều việc hơn thuật toán tuần tự, việc thực
                thi bị giới hạn bởi bộ nhớ, sự nhạy cảm với đầu vào, và chi phí đồng bộ hóa. Hãy nói một kernel đang mắc điều nào
                trước khi bạn thay đổi nó.
              </>
            }
          >
            Four things make the fast version hard: a parallel algorithm that does more work than the sequential one,
            memory-bound execution, sensitivity to the input, and synchronisation overhead. Say which of them a kernel
            suffers from before you change it.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Sách vẽ các chương của nó thành một sơ đồ tiên quyết, mà khóa học này coi là lời khuyên. Theo các mũi tên của nó,
                matmul nhanh cần các chương {BEFORE_MATMUL.join(', ')} và không gì khác, attention cần {BEFORE_ATTENTION.length}{' '}
                chương, và lấy mẫu là con đường dài nhất. Từ chương 12 trở lên, số chương mang nghĩa khác trong ấn bản thứ 4.
              </>
            }
          >
            The book draws its chapters as a chart of prerequisites, which this course treats as advice. By its arrows,
            fast matmul needs chapters {and(BEFORE_MATMUL)} and nothing else,
            attention needs {BEFORE_ATTENTION.length} chapters, and sampling is the longest road. From chapter 12 upward,
            a chapter number means something else in the 4th edition.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Tính đúng đắn đi theo mã nguồn còn hiệu năng thì không. HIP là mô hình lập trình của CUDA trên GPU AMD, còn Metal là
                một code base thứ hai. Trên mọi thiết bị, hãy đo, và truy vấn những gì lẽ ra bạn sẽ giả định.
              </>
            }
          >
            Correctness travels with the source and performance does not. HIP is CUDA's programming model on AMD GPUs,
            and Metal is a second code base. On every device, measure, and query what you would otherwise assume.
          </Translated>
        </li>
      </ul>
      <Source pages="xvii–xxvi, 1–18" />

      <h2><Translated vi="Một bài kiểm tra tổng hợp">A mixed check</Translated></h2>
      <Translated
        vi={
          <p>
            Các câu này dựa trên cả chương và trên những tình huống các mục chưa trình bày.
            {asked ? ` Đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu hỏi, tính theo câu trả lời gần nhất của bạn cho mỗi câu.` : ''}
          </p>
        }
      >
        <p>
          These draw on the whole chapter and on situations the sections did not show.
          {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
        </p>
      </Translated>
      <Ask q={questionsWrap['ch01/wrap/q-reprofile']} />
      <Ask q={questionsWrap['ch01/wrap/q-both']} />
      <Ask q={questionsWrap['ch01/wrap/q-morethreads']} />
      <Ask q={questionsWrap['ch01/wrap/q-skip9']} />
      <Ask q={questionsWrap['ch01/wrap/q-sharedmemory']} />

      <h2><Translated vi="Bài tập của sách">The book's exercises</Translated></h2>
      <Translated
        vi={
          <p>
            Không có bài nào. Chương 1 không có phần bài tập và kết thúc bằng tài liệu tham khảo ở p. 18. Cái nó có là các con
            số được tính sẵn, và mỗi con số đều được tính lại ở đâu đó trên các trang này. Tự kiểm tra với chúng là gợi ý của
            chúng tôi, không phải của sách.
          </p>
        }
      >
        <p>
          There are none. Chapter 1 has no exercise section and ends with its references on p. 18. What it has are
          worked numbers, and each of them is worked somewhere on these pages. Checking yourself against them is our
          suggestion, not the book's.
        </p>
      </Translated>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="worked">
        <thead>
          <tr><th><Translated vi="Con số của sách">The book's number</Translated></th><th><Translated vi="Trang">Page</Translated></th><th><Translated vi="Nơi nó được tính ở đây">Where it is worked here</Translated></th></tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="Gấp đôi thông lượng với gấp đôi công suất; một nửa độ trễ với gấp bốn công suất">Twice the throughput for twice the power; half the latency for four times the power</Translated></td>
            <td>5</td>
            <td><Translated vi={<>hoạt hình và trình khám phá "Tiêu một ngân sách công suất" trên <a href={href('ch01', 'start')}>trang bắt đầu</a></>}>the animation and the "Spend a power budget" explorer on the <a href={href('ch01', 'start')}>start page</a></Translated></td>
          </tr>
          <tr>
            <td><Translated vi="34, 67 và 1,979 TFLOPS trên GPU ví dụ của sách">34, 67 and 1,979 TFLOPS on the book's example GPU</Translated></td>
            <td>3</td>
            <td><Translated vi={<>câu hỏi về con số 16 bit, <a href={href('ch01', 'start')}>trang bắt đầu</a></>}>the question on the 16-bit figure, <a href={href('ch01', 'start')}>start page</a></Translated></td>
          </tr>
          <tr>
            <td><Translated vi="200 s so với 10 s là mức tăng tốc 20×">200 s against 10 s is a speed-up of 20×</Translated></td>
            <td>9</td>
            <td><Translated vi={<>"Tăng tốc" trong <a href={href('ch01', '1.3')}>1.3</a></>}>"Speed-up" in <a href={href('ch01', '1.3')}>1.3</a></Translated></td>
          </tr>
          <tr>
            <td><Translated vi={<>30% lần chạy được làm nhanh hơn 100 lần: {BOOK_CASES[0].book.replace('no more than', 'không quá').replace('about', 'khoảng')}×</>}>30% of the run made 100 times faster: {BOOK_CASES[0].book}×</Translated></td>
            <td>9</td>
            <td><Translated vi={<>dự đoán đầu tiên trong <a href={href('ch01', '1.3')}>1.3</a>, và một cấu hình sẵn của trình khám phá Amdahl ở đó</>}>the first prediction in <a href={href('ch01', '1.3')}>1.3</a>, and a preset of its Amdahl explorer</Translated></td>
          </tr>
          <tr>
            <td><Translated vi={<>Cùng 30% đó được làm nhanh vô hạn: {BOOK_CASES[1].book.replace('no more than', 'không quá').replace('about', 'khoảng')}×</>}>The same 30% made infinitely fast: {BOOK_CASES[1].book}×</Translated></td>
            <td>9</td>
            <td><Translated vi="phần đáp án của dự đoán đó, và một cấu hình sẵn">the reveal of that prediction, and a preset</Translated></td>
          </tr>
          <tr>
            <td><Translated vi={<>99% lần chạy được làm nhanh hơn 100 lần: {BOOK_CASES[2].book.replace('no more than', 'không quá').replace('about', 'khoảng')}×</>}>99% of the run made 100 times faster: {BOOK_CASES[2].book}×</Translated></td>
            <td>9</td>
            <td><Translated vi={<>dự đoán thứ hai trong <a href={href('ch01', '1.3')}>1.3</a>, và một cấu hình sẵn</>}>the second prediction in <a href={href('ch01', '1.3')}>1.3</a>, and a preset</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Kết quả trên 100× cần hơn 99.9% công việc nằm trong phần song song">Results above 100× need more than 99.9% of the work in the parallel part</Translated></td>
            <td>9–10</td>
            <td><Translated vi={<>nhiệm vụ ở đầu trình khám phá Amdahl và câu hỏi sau nó, trong <a href={href('ch01', '1.3')}>1.3</a></>}>the task in the header of the Amdahl explorer and the question after it, in <a href={href('ch01', '1.3')}>1.3</a></Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Khoảng 10× khi một bản chuyển đổi thẳng thắn làm bão hòa bộ nhớ">About 10× when a straightforward port saturates memory</Translated></td>
            <td>10</td>
            <td><Translated vi={<>trình khám phá "Điều gì chặn trần kernel" trong <a href={href('ch01', '1.3')}>1.3</a>; đo trên thiết bị của chính bạn từ chương 5 trở đi</>}>the "What caps the kernel" explorer in <a href={href('ch01', '1.3')}>1.3</a>; measured on your own device from chapter 5 on</Translated></td>
          </tr>
        </tbody>
      </table>

      <h2><Translated vi="Những chỗ sách đã lỗi thời">Where the book has aged</Translated></h2>
      <Translated
        vi={
          <p>
            Một chương mở đầu đầy ngày tháng và con số đếm, và chúng lỗi thời nhanh hơn ý tưởng. Ghi chú riêng của các mục đã
            bàn về các con số của H100 (<a href={href('ch01', 'start')}>trang bắt đầu</a>), các quy tắc kinh nghiệm 10× (
            <a href={href('ch01', '1.3')}>1.3</a>), và các thế hệ GPU, phiên bản CUDA, song song động (dynamic parallelism) và
            mục lục của nhà xuất bản (<a href={href('ch01', '1.7')}>1.7</a>). Bảng dưới đây có phần còn lại của những gì chúng
            tôi đã kiểm tra. Không điều nào trong đó thay đổi lập luận của chương.
          </p>
        }
      >
        <p>
          A first chapter is full of dates and counts, and they age faster than the ideas. The sections' own notes cover
          the H100 figures (<a href={href('ch01', 'start')}>start page</a>), the 10× rules of thumb (
          <a href={href('ch01', '1.3')}>1.3</a>), and GPU generations, CUDA versions, dynamic parallelism and the
          publisher's contents list (<a href={href('ch01', '1.7')}>1.7</a>). The table has the rest of what we checked.
          None of it changes an argument of the chapter.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch01-facts.md, A3, A4, A6, A7, A10 to A12, A15; ch05-facts.md, A9)">{/* i18n-ignore: date and notes reference */}
        <Translated
          vi={
            <p>
              Mọi thứ ở đây đến từ chính các trang của nhà sản xuất và tổ chức tiêu chuẩn. Một vài trang trong số đó chỉ đọc được
              qua một công cụ trả về văn bản đã trích xuất. Không có gì được chạy trên phần cứng, và không điều gì trong chương này
              dựa trên một phép đo.
            </p>
          }
        >
          <p>
            Everything here comes from vendors' and standards bodies' own pages. A few of those pages could only be read
            through a tool that returns extracted text. Nothing was run on hardware, and nothing in this chapter
            rests on a measurement.
          </p>
        </Translated>
      </Today>
      <table className="data wide" style={{ maxWidth: '62rem' }} data-table="aged">
        <thead>
          <tr><th style={{ minWidth: '13rem' }}><Translated vi="Sách nói">The book says</Translated></th><th><Translated vi="Những gì chúng tôi tìm thấy">What we found</Translated></th></tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="NVIDIA giới thiệu CUDA năm 2007, với hỗ trợ phần cứng bắt đầu từ G80 (pp. 5–7).">NVIDIA introduced CUDA in 2007, with hardware support starting from the G80 (pp. 5–7).</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    Hướng dẫn lập trình hiện hành của NVIDIA và dòng thời gian của công ty đều ghi CUDA ra đời năm 2006. Bộ toolkit cũ
                    nhất trong kho lưu trữ của họ, 1.0, đề ngày tháng 6 năm 2007. Chúng tôi không tìm thấy G80 được nhắc tên trên một
                    trang NVIDIA hiện hành nào, nên phần đó chỉ là phát biểu của sách.
                  </>
                }
              >
                NVIDIA's current programming guide and its corporate timeline both date CUDA to 2006. The oldest toolkit
                in its archive, 1.0, is dated June 2007. We could not find the G80 named on a current NVIDIA page, so
                that part is the book's statement only.
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Hơn 1 tỷ GPU hỗ trợ CUDA đang được dùng, và có hàng triệu lập trình viên CUDA (pp. xvii, 6).">More than 1 billion CUDA-enabled GPUs are in use, and there are millions of CUDA programmers (pp. xvii, 6).</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    Chúng tôi không xác nhận được con số một tỷ. Bài keynote tháng 3 năm 2026 của NVIDIA nói về "hundreds of millions
                    of GPUs and computing systems around the world that run CUDA". Hàng triệu nhà phát triển cũng là con số của chính
                    NVIDIA.
                  </>
                }
              >
                We could not confirm the billion. NVIDIA's keynote of March 2026 speaks of "hundreds of millions of GPUs
                and computing systems around the world that run CUDA". Millions of developers is NVIDIA's own figure
                too.
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Bộ xử lý máy chủ Intel và AMD cho mỗi lõi hai luồng phần cứng (hyper-threading) (p. 3).">Intel and AMD server processors give each core two hardware threads (hyper-threading) (p. 3).</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    Đúng với một số dòng của Intel và AMD, không đúng với Xeon E-core của Intel mà chúng tôi đã kiểm tra, có số luồng
                    bằng số lõi. Ampere, ví dụ ARM của sách, gọi các lõi của mình là đơn luồng. Số lõi trong sách từ đó đã bị vượt qua,
                    điều này vẫn giữ nguyên sự tương phản của sách giữa vài luồng hạng nặng và rất nhiều luồng nhẹ.
                  </>
                }
              >
                True of some Intel and AMD parts, not of the Intel E-core Xeon we checked, whose threads equal its cores.
                Ampere, the book's ARM example, calls its cores single-threaded. The book's core counts have since been
                passed, which leaves its contrast between a few heavyweight threads and very many light ones intact.
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi='Một CPU máy chủ năm 2022 đạt đỉnh "vài TFLOPS" (p. 3).'>A server CPU of 2022 peaks at "a few TFLOPS" (p. 3).</Translated></td>
            <td><Translated vi="Đúng bậc độ lớn, theo các con số tuân thủ xuất khẩu mà Intel công bố cho bộ xử lý của mình.">The right order of magnitude, going by the export-compliance figures Intel publishes for its processors.</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="CPU và GPU cùng nhau đã đưa tính toán exascale đến các cụm máy (p. 10).">CPUs and GPUs together have brought exascale computing to clusters (p. 10).</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    Exascale đúng: danh sách TOP500 tháng 11 năm 2026 có năm hệ thống trên một exaflop. Văn bản của nó nêu bộ tăng tốc
                    GPU cho một số trong năm hệ thống chứ không phải tất cả, nên đừng đọc câu đó thành "mọi máy exascale đều ghép CPU
                    với GPU".
                  </>
                }
              >
                Exascale holds: the TOP500 list of November 2026 has five systems above one exaflop. Its text names GPU
                accelerators for some of the five and not for all, so do not read the sentence as "every exascale
                machine pairs CPUs with GPUs".
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="OpenMP được thiết kế cho CPU và mở rộng sang GPU, và các trình biên dịch của nó vẫn đang cải thiện (pp. 12–13).">OpenMP was designed for CPUs and extended to GPUs, and its compilers are still improving (pp. 12–13).</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    Lịch sử đó đúng. Một báo cáo về các chỉ thị cho bộ tăng tốc gắn kèm xuất hiện vào tháng 11 năm 2012 và được thay thế
                    bởi OpenMP 4.0 vào tháng 7 năm 2013; đặc tả hiện hành là 6.1. "Vẫn đang cải thiện" là ý kiến của các tác giả, và
                    chúng tôi không kiểm tra nó.
                  </>
                }
              >
                The history holds. A report on directives for attached accelerators appeared in November 2012 and was
                superseded by OpenMP 4.0 in July 2013; the current specification is 6.1. "Still improving" is the
                authors' opinion, and we did not check it.
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="OpenCL được Apple, Intel, AMD/ATI và NVIDIA cùng phát triển năm 2009 (p. 13).">OpenCL was developed jointly in 2009 by Apple, Intel, AMD/ATI and NVIDIA (p. 13).</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    Khronos phê chuẩn và phát hành OpenCL 1.0 ngày 9 tháng 12 năm 2008, đúng năm mà tài liệu tham khảo [9] của chính sách
                    ghi, và danh sách các công ty đứng sau nó dài hơn nhiều. OpenCL vẫn được duy trì, ở phiên bản 3.1.
                  </>
                }
              >
                Khronos ratified and released OpenCL 1.0 on 9 December 2008, the year the book's own reference [9]
                gives, and the list of companies behind it is much longer. OpenCL is still maintained, at version 3.1.
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Tài liệu tham khảo [4] là CUDA C++ Programming Guide, Release 13.0, 2025 (p. 18).">Reference [4] is the CUDA C++ Programming Guide, Release 13.0, 2025 (p. 18).</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    NVIDIA đã tái cấu trúc hướng dẫn từ đó. Địa chỉ cũ không còn, liên kết đến các mục từ tài liệu cũ không còn hoạt
                    động, và văn bản 13.0 được giữ như một bản lưu trữ mà hướng dẫn mới liên kết tới.
                  </>
                }
              >
                NVIDIA has restructured the guide since. The old address is gone, section links from older material no
                longer resolve, and the 13.0 text is kept as an archive that the new guide links to.
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Ấn bản thứ 4 đề năm 2022 (p. xvii); dòng bản quyền của ấn bản này ghi 2027 (p. 1).">The 4th edition is dated 2022 (p. xvii); the copyright line of this one reads 2027 (p. 1).</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    Nhà xuất bản ghi ấn bản thứ 4 ngày 28 tháng 5 năm 2022. Với ấn bản thứ 5, các trang của họ đưa ra hai ngày xuất bản,
                    27 tháng 2 và 3 tháng 6 năm 2026. Khi trích dẫn sách, hãy ghi ấn bản, không ghi năm.
                  </>
                }
              >
                The publisher dates the 4th edition May 28, 2022. For the 5th its pages give two publication dates,
                February 27 and June 3, 2026. When you cite the book, give the edition, not the year.
              </Translated>
            </td>
          </tr>
        </tbody>
      </table>

      <h2><Translated vi="Những chỗ bản PDF có vẻ sai">Where the PDF looks wrong</Translated></h2>
      <Translated
        vi={
          <p>
            Ấn bản này còn mới và bản mà khóa học này dựa vào có vài chỗ có vẻ sơ suất trong Lời nói đầu và chương 1. Không chỗ
            nào chạm đến một lập luận. Chúng được liệt kê để bạn không bị bối rối khi đọc sách song song.
          </p>
        }
      >
        <p>
          This edition is new and the copy this course was built from has a few apparent slips in the Preface and
          chapter 1. None of them touches an argument. They are listed so that you are not thrown by them when you read
          the book alongside.
        </p>
      </Translated>
      <ul data-list="errata">
        <li>
          <Translated
            vi={
              <>
                Lời nói đầu nói sách được tổ chức "into four major parts" (p. xxii), rồi mô tả ba phần và nói phần thứ tư cũ đã bị bỏ
                (p. xxiv). Sơ đồ ở p. xxiii và §1.7 (p. 15) đều cho thấy ba. Mục lục trước khi xuất bản của nhà xuất bản có bốn phần,
                có lẽ cụm từ đó bắt nguồn từ đây.
              </>
            }
          >
            The Preface says the book is organised "into four major parts" (p. xxii), then describes three and says the
            old fourth part is gone (p. xxiv). The chart on p. xxiii and §1.7 (p. 15) both show three. The publisher's
            pre-publication contents list had four, which is probably where the phrase comes from.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                §1.7 nói radix sort "relies on the scan pattern covered in Chapter 14" (p. 16). Ở mọi chỗ khác scan là chương 11, còn
                chương 14 là sắp xếp. Lời nói đầu lại nói khác: radix sort xây trên phép lọc ổn định của chương 12 (p. xx). Hai cách
                nói chỉ khớp nhau gián tiếp, vì phép lọc ổn định dùng scan.
              </>
            }
          >
            §1.7 says radix sort "relies on the scan pattern covered in Chapter 14" (p. 16). Scan is chapter 11
            everywhere else, and chapter 14 is sort. The Preface puts it differently again: radix sort builds on the
            stable filter of chapter 12 (p. xx). The two agree only indirectly, because the stable filter uses scan.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Văn bản nhắc đến "Fig. 1.1(a)" và "Fig. 1.1(b)" (pp. 4–5). Hình không có nhãn nào như vậy, chỉ có CPU và GPU. Một
                thanh mỏng dưới các hàng của GPU không có nhãn, và "small cache memories" mà văn bản nhắc tới có thể là thanh đó, các
                khối nhỏ ở đầu mỗi hàng, hoặc cả hai. Bản vẽ lại của chúng tôi trên{' '}
                <a href={href('ch01', 'start')}>trang bắt đầu</a> có ghi nhãn cho các dải.
              </>
            }
          >
            The text refers to "Fig. 1.1(a)" and "Fig. 1.1(b)" (pp. 4–5). The figure has no such labels, only CPU and
            GPU. A thin bar under the GPU's rows is unlabelled, and the "small cache memories" the text mentions could
            be that bar, the small blocks at the start of each row, or both. Our redrawing on the{' '}
            <a href={href('ch01', 'start')}>start page</a> labels its bands.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Trong sơ đồ của Fig. P.1 (p. xxiii), đường từ chương 9 đến chương 17 và 21 đi ngay dưới ô của chương 14. Chúng tôi
                đọc nó là bắt đầu từ chương 9. Hình vẽ chật đến mức không loại trừ được một mũi tên từ 14, và trình khám phá trong{' '}
                <a href={href('ch01', '1.7')}>1.7</a> đánh dấu cả hai mũi tên là không chắc chắn.
              </>
            }
          >
            In the chart of Fig. P.1 (p. xxiii), the line from chapter 9 to chapters 17 and 21 passes directly under the
            box of chapter 14. We read it as starting at chapter 9. The drawing is tight enough that an arrow from 14
            cannot be ruled out, and the explorer in <a href={href('ch01', '1.7')}>1.7</a> marks both arrows as
            uncertain.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Tên chương khác nhau tùy chỗ. Chương 16 là "Wavefront Parallelism" trong Lời nói đầu và "Dynamic Progr. and
                Wavefront Parallelism" trong sơ đồ. Chương 23 là "Multi-GPU Programming" trong Lời nói đầu và "Programming a
                Heterogeneous Cluster" trong sơ đồ. Chương 22 là "Problem Decomposition" trong sơ đồ, trong khi §1.7 mô tả nó như
                một phần giới thiệu về tư duy tính toán. Ví dụ của chương 16 là một thuật toán trong Lời nói đầu và hai thuật toán
                trong §1.7 (p. 17).
              </>
            }
          >
            Chapter titles differ from place to place. Chapter 16 is "Wavefront Parallelism" in the Preface and "Dynamic
            Progr. and Wavefront Parallelism" in the chart. Chapter 23 is "Multi-GPU Programming" in the Preface and
            "Programming a Heterogeneous Cluster" in the chart. Chapter 22 is "Problem Decomposition" in the chart,
            while §1.7 describes it as an introduction to computational thinking. Chapter 16's example is one algorithm
            in the Preface and two in §1.7 (p. 17).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Kế hoạch giảng dạy trong Lời nói đầu nói giai đoạn đầu "consists of 14 chapters", tức là chương 2 đến 15, rồi xếp
                chương 7 đến 14 cho năm tuần cuối (p. xxiv). Chương 15 bị bỏ lại không có tuần nào.
              </>
            }
          >
            The Preface's teaching plan says its first phase "consists of 14 chapters", which would be chapters 2 to 15,
            and then schedules chapters 7 through 14 for the last five weeks (p. xxiv). Chapter 15 is left without a
            week.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                §1.5 ghi OpenCL năm 2009 (p. 13), trong khi tài liệu tham khảo [9] của chính sách ghi đặc tả 1.0 là năm 2008 (p.
                18). Bảng ở trên có những gì Khronos nói.
              </>
            }
          >
            §1.5 dates OpenCL to 2009 (p. 13), while the book's own reference [9] gives the 1.0 specification as 2008
            (p. 18). The table above has what Khronos says.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Những chỗ nhỏ hơn ở pp. 16–17: "coverts" thay cho converts, "filter patter", "divide-and-concur", và "leverage
                techniques scatter-to-gather transformations". Ở p. 10 sách có một lần nói "many-core GPUs" trong khi phần còn lại
                của chương nói many-thread; nghĩa vẫn như nhau.
              </>
            }
          >
            Smaller things on pp. 16–17: "coverts" for converts, "filter patter", "divide-and-concur", and "leverage
            techniques scatter-to-gather transformations". On p. 10 the book says "many-core GPUs" once where the rest
            of the chapter says many-thread; it means the same thing.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Không phải sơ suất, dù trông giống: 1,979 TFLOPS cho số học 16 bit bên cạnh 67 cho 32 bit (p. 3). Chương không giải
                thích. <a href={href('ch01', 'start')}>Trang bắt đầu</a> thì có.
              </>
            }
          >
            Not a slip, though it looks like one: 1,979 TFLOPS for 16-bit arithmetic next to 67 for 32-bit (p. 3). The
            chapter gives no explanation. The <a href={href('ch01', 'start')}>start page</a> does.
          </Translated>
        </li>
      </ul>
      <Aside>
        <Translated vi={<p>Đây là cách đọc của chúng tôi trên một bản PDF. Nếu bản của bạn khác, hãy tin bản của bạn.</p>}>
          <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
        </Translated>
      </Aside>

      <h2><Translated vi="Những gì chương này bỏ qua">What this chapter left out</Translated></h2>
      <Translated
        vi={
          <p>
            Đây là một chương rút gọn. Nó giữ lập luận về thiết kế, phép tính tăng tốc, bốn khó khăn và bản đồ của sách. Đây là
            phần còn lại, và nơi tìm thấy nó.
          </p>
        }
      >
        <p>
          This is a condensed chapter. It keeps the design argument, the speed-up arithmetic, the four difficulties and
          the map of the book. Here is the rest, and where to find it.
        </p>
      </Translated>
      <ul data-list="omitted">
        <li>
          <Translated
            vi={
              <>
                Sự kết thúc của việc mở rộng CPU đơn (pp. 1–3): vài câu trên trang bắt đầu. "Vòng tuần hoàn tốt đẹp" (virtuous
                cycle), mô hình von Neumann và bộ đếm chương trình, và "cuộc cách mạng đồng thời" (concurrency revolution): không
                dạy.
              </>
            }
          >
            The end of single-CPU scaling (pp. 1–3): a few sentences on the start page. The "virtuous cycle", the von
            Neumann model and the program counter, and the "concurrency revolution": not taught.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Lượng máy đã cài đặt và kiểu dáng như những lý do GPU thắng, với ví dụ chẩn đoán hình ảnh y tế; tính toán qua giao
                diện đồ họa trước CUDA; G80; FPGA như những bộ tăng tốc khác (pp. 6–7). Không dạy.
              </>
            }
          >
            Installed base and form factor as reasons GPUs won, with the medical-imaging example; computing through
            graphics interfaces before CUDA; the G80; FPGAs as other accelerators (pp. 6–7). Not taught.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                §1.2, "Why more speed or parallelism?" (pp. 7–9): chuyến tham quan các ứng dụng, từ sinh học phân tử và mã hóa video
                đến trò chơi, bản sao số (digital twin) và lịch sử học sâu. Chỉ các quy tắc kinh nghiệm ở p. 7 được giữ lại, trong
                mục 1.3.
              </>
            }
          >
            §1.2, "Why more speed or parallelism?" (pp. 7–9): the tour of applications, from molecular biology and video
            coding to games, digital twins and the history of deep learning. Only its rules of thumb from p. 7 are kept,
            in section 1.3.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Quả đào của Fig. 1.2 (pp. 10–11): một đoạn ngắn trong mục 1.3 và không có hình vẽ. Câu chuyện mỗi giao diện chạm tới
                được bao nhiêu phần quả đào bị bỏ.
              </>
            }
          >
            The peach of Fig. 1.2 (pp. 10–11): a short paragraph in section 1.3 and no drawing. Its story of how much
            of the peach each interface could reach is dropped.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                §1.5 (pp. 12–13): cách các chỉ thị và pragma của OpenMP hoạt động, phân rã miền của MPI, và thiết kế của OpenCL. Một
                đoạn trong mục 1.7. MPI, NCCL và NVSHMEM mỗi thứ có một đoạn trên một trang của chương 23 của khóa học này (
                <a href={href('ch23', '23.4')}>23.3–23.5</a>), chương giữ lại stream, sao chép bất đồng bộ và bộ nhớ ghim (pinned
                memory).
              </>
            }
          >
            §1.5 (pp. 12–13): how OpenMP's directives and pragmas work, MPI's domain decomposition, and the design of
            OpenCL. One paragraph in section 1.7. MPI, NCCL and NVSHMEM get a paragraph each on one page of this
            course's chapter 23 (<a href={href('ch23', '23.4')}>23.3–23.5</a>), which keeps streams, asynchronous copies
            and pinned memory.
          </Translated>
        </li>
        <li><Translated vi="§1.6, ba mục tiêu (p. 14): mỗi mục tiêu một câu trong mục 1.7.">§1.6, the three goals (p. 14): one sentence each in section 1.7.</Translated></li>
        <li>
          <Translated
            vi={
              <>
                Phần giới thiệu từng chương của §1.7 (pp. 15–18): được thay bằng trình khám phá lộ trình đọc. Công thức sáu bước ở p.
                15 chỉ được nêu tên; chương 2 đi qua nó.
              </>
            }
          >
            The chapter-by-chapter previews of §1.7 (pp. 15–18): replaced by the reading-path explorer. The six-step
            recipe on p. 15 is only named; chapter 2 works through it.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                "How to use the book" trong Lời nói đầu (pp. xxiv–xxvi): lịch sử khóa học của các tác giả, hai giai đoạn giảng dạy,
                dự án cuối khóa, cuộc thi trong lớp, bộ tài liệu giảng dạy và trang web đi kèm. Cả lời cảm ơn (p. xxvii) và tài liệu
                tham khảo (p. 18).
              </>
            }
          >
            "How to use the book" in the Preface (pp. xxiv–xxvi): the history of the authors' course, its two teaching
            phases, the final project, the class competition, the teaching kit and the companion site. Also the
            acknowledgments (p. xxvii) and the references (p. 18).
          </Translated>
        </li>
      </ul>

      <h2><Translated vi="Dự án đến lúc này">The project so far</Translated></h2>
      <Translated
        vi={
          <p>
            Chưa có gì. Chương 1 không có mã, nên nó không có lab và không có mốc dự án nào của dự án xuyên suốt, một transformer
            nhỏ mà cuối cùng bạn sẽ chạy. Mốc đầu tiên, M0, thuộc chương 2: các phép toán theo từng phần tử. Ở đây cũng không có
            gì để chạy từng bước, làm hỏng hay đo. Việc đó cũng bắt đầu ở chương 2.
          </p>
        }
      >
        <p>
          Nothing yet. Chapter 1 has no code, so it has no lab and no milestone of the through-line project, the small
          transformer you will end up running. The first milestone, M0, is chapter 2's: the elementwise operations.
          There was also nothing here to step through, break or measure. That starts in chapter 2 as well.
        </p>
      </Translated>
      <p>
        <Translated
          vi={
            <>
              Có một việc đáng làm ngay bây giờ. Từ thư mục gốc của repository, chạy <code>python3 labs/pmpp.py doctor</code>.
              Nó liệt kê các backend mà khóa học này có thể build trên máy của bạn, cho biết vì sao backend nào đó bị bỏ qua,
              và in ra những gì mỗi thiết bị báo cáo về chính nó.
            </>
          }
        >
          One thing is worth doing now. From the repository root, run <code>python3 labs/pmpp.py doctor</code>. It lists
          the backends this course can build on your machine, says why any of them is skipped, and prints what each
          device reports about itself.
        </Translated>{' '}
        <span data-devices>
          {devices.length
            ? tr(
                `Lab results recorded on this machine so far come from ${and(devices.map((b) => `${NAME[b] ?? b} (${String(reports[b].device?.name ?? 'unnamed device')})`))}.`,
                `Kết quả lab ghi nhận trên máy này đến nay đến từ ${devices.map((b) => `${NAME[b] ?? b} (${String(reports[b].device?.name ?? 'thiết bị không tên')})`).join(', ')}.`,
              )
            : tr(
                'No lab results from a GPU are recorded on this machine yet. The first will come from chapter 2.',
                'Chưa có kết quả lab nào từ GPU được ghi nhận trên máy này. Kết quả đầu tiên sẽ đến từ chương 2.',
              )}
        </span>{' '}
        {unified !== undefined ? (
          <span data-unified>
            {yes(unified) ? (
              <Translated
                vi={
                  <>
                    Thiết bị Metal của nó báo <code>hasUnifiedMemory</code> là true: một bộ nhớ cho cả CPU và GPU, đúng trường hợp của
                    câu hỏi cuối ở trên.
                  </>
                }
              >
                <>
                  Its Metal device reports <code>hasUnifiedMemory</code> as true: one memory for CPU and GPU, the case of
                  the last question above.
                </>
              </Translated>
            ) : (
              <Translated
                vi={
                  <>
                    Thiết bị Metal của nó báo <code>hasUnifiedMemory</code> là false: GPU có bộ nhớ riêng, như trong Hình 1.1.
                  </>
                }
              >
                <>
                  Its Metal device reports <code>hasUnifiedMemory</code> as false: the GPU has memory of its own, as in
                  Fig. 1.1.
                </>
              </Translated>
            )}
          </span>
        ) : null}
      </p>
      <Translated
        vi={
          <p>
            <a href={href('ch02', 'start')}>Chương 2</a> viết kernel đầu tiên. Nó biến host và device thành mã, đi qua sáu bước
            mà mục 1.7 chỉ nêu tên, và biến phép làm tròn lên của câu hỏi hai đợt trên trang bắt đầu thành công thức cho kích
            thước của một lần khởi chạy. Đó cũng là nơi một máy có một bộ nhớ chung cho CPU và GPU lần đầu thay đổi những gì bạn
            viết. Lab đầu tiên của nó đo thời gian một vòng lặp CPU một luồng bên cạnh kernel của bạn, để mốc so sánh của mọi
            mức tăng tốc về sau là mốc do chính bạn đo.
          </p>
        }
      >
        <p>
          <a href={href('ch02', 'start')}>Chapter 2</a> writes the first kernel. It turns host and device into code,
          works through the six steps that section 1.7 only named, and turns the rounding up of the two-wave question on
          the start page into the formula for the size of a launch. It is also where a machine with one memory for CPU
          and GPU first changes what you write. Its first lab times a one-thread CPU loop beside your kernel, so the
          baseline of every later
          speed-up is one you measured yourself.
        </p>
      </Translated>
    </>
  );
}
