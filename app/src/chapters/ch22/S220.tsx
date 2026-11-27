import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source } from '../../components/notes';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';

// Prerequisite questions drawn from chapters 1, 5, 9, 10 and 15 (their notes, labs/README.md
// and lab 15.6's README). Each explanation names where the idea was taught and where this
// chapter uses it.
export const questionsS220: Record<string, Question> = {
  'ch22/start/q-decode-amdahl': {
    kind: 'numeric',
    id: 'ch22/start/q-decode-amdahl',
    prompt: (
      <Translated
        vi={
          <>
            Một vòng lặp decode dành 80% thời gian của mỗi token cho các kernel GPU và 20% cho host (lấy mẫu, khởi chạy
            kernel). Bạn làm mọi kernel nhanh gấp 4 lần và không đổi gì ở host. Tốc độ tăng tổng thể cho mỗi token là bao
            nhiêu?
          </>
        }
      >
        <>
          A decode loop spends 80% of each token&rsquo;s time in GPU kernels and 20% on the host (sampling, kernel
          launches). You make every kernel 4 times faster and change nothing on the host. What is the overall speed-up
          per token?
        </>
      </Translated>
    ),
    answer: 2.5,
    tol: 0.01,
    unit: '×',
    explain: (
      <Translated
        vi={
          <p>
            1 / (0.2 + 0.8 / 4) = 1 / 0.4 = 2.5. Sau thay đổi, phần 0.2 của host lớn bằng phần 0.2 của các kernel, nên một
            nửa thời gian của mỗi token giờ là phần việc bạn chưa động tới, và thêm 4× nữa cho các kernel cũng chỉ đưa bạn
            đến 1 / 0.25 = 4. Mục <a href={href('ch22', '22.3')}>22.3</a> lập luận điều này cho cả một ứng dụng. Đã học ở{' '}
            <a href={href('ch01', '1.3')}>ch. 1 §1.3</a>.
          </p>
        }
      >
        <p>
          1 / (0.2 + 0.8 / 4) = 1 / 0.4 = 2.5. After the change the host&rsquo;s 0.2 is as large as the kernels&rsquo;
          0.2, so half of every token is now work you did not touch, and another 4× on the kernels would take you only to
          1 / 0.25 = 4. Section <a href={href('ch22', '22.3')}>22.3</a> makes this argument for a whole application.
          Taught in <a href={href('ch01', '1.3')}>ch. 1 §1.3</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 1 §1.3, p. 9 (the formula: notes/today/ch01-facts.md A16)',
    vi: { source: 'ch. 1 §1.3, tr. 9 (công thức: notes/today/ch01-facts.md A16)' },
  },
  'ch22/start/q-gemv-ceiling': {
    kind: 'numeric',
    id: 'ch22/start/q-gemv-ceiling',
    prompt: (
      <Translated
        vi={
          <>
            Một phép nhân ma trận-vector float32 thực hiện 2 phép toán dấu phẩy động cho mỗi trọng số 4 byte mà nó đọc. Trên
            một thiết bị có ngưỡng 20 FLOP/B (ví dụ của sách), nó dùng được tối đa bao nhiêu phần trăm năng lực tính toán
            đỉnh?
          </>
        }
      >
        <>
          A float32 matrix-vector product does 2 floating-point operations for each 4-byte weight it reads. On a device
          whose threshold is 20 FLOP/B (the book&rsquo;s example), at most what percentage of the peak arithmetic can it
          use?
        </>
      </Translated>
    ),
    answer: 2.5,
    tol: 0.01,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            2 / 4 = 0.5 FLOP/B, và 0.5 / 20 = 2.5%. Không mẹo kernel nào thay đổi được con số này, vì mỗi trọng số chỉ
            được dùng một lần cho mỗi token. Đòn bẩy duy nhất là thêm hàng cho mỗi trọng số (batching, mục{' '}
            <a href={href('ch22', '22.5')}>22.5</a>) hoặc bớt byte cho mỗi trọng số (lượng tử hóa). Đã học ở{' '}
            <a href={href('ch05', '5.1')}>ch. 5 §5.1</a>.
          </p>
        }
      >
        <p>
          2 / 4 = 0.5 FLOP/B, and 0.5 / 20 = 2.5%. No kernel trick moves it, because each weight is used once per token.
          The only levers are more rows per weight (batching, section <a href={href('ch22', '22.5')}>22.5</a>) or fewer
          bytes per weight (quantisation). Taught in <a href={href('ch05', '5.1')}>ch. 5 §5.1</a>.
        </p>
      </Translated>
    ),
    source: "ch. 5 §5.1, pp. 94–97 (20 FLOP/B is the book's H100 example)",
    vi: { source: 'ch. 5 §5.1, tr. 94–97 (20 FLOP/B là ví dụ H100 của sách)' },
  },
  'ch22/start/q-emu-blind': {
    kind: 'choice',
    id: 'ch22/start/q-emu-blind',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel histogram đếm bằng một lệnh <code>bins[b]++</code> thông thường và qua mọi bài kiểm tra trên backend{' '}
            <code>emu</code>. Vì sao, và cái gì sẽ bắt được lỗi này?
          </>
        }
      >
        <>
          A histogram kernel counts with a plain <code>bins[b]++</code> and passes every test on the <code>emu</code>{' '}
          backend. Why, and what would catch the bug?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated
            vi={
              <>
                <code>emu</code> chạy từng luồng cho tới khi nó gặp một điểm đồng bộ hóa, nên không phép tăng nào bị ngắt
                giữa chừng; backend <code>race</code> báo các dòng va chạm nhau.
              </>
            }
          >
            <>
              <code>emu</code> runs each thread until it reaches a synchronisation point, so no increment is ever
              interrupted; the <code>race</code> backend reports the colliding lines.
            </>
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated
            vi={
              <>
                <code>emu</code> cho mỗi luồng một bản sao riêng của <code>bins</code> rồi cộng các bản sao lại ở cuối, nên
                các phép tăng không bao giờ gặp nhau; bất kỳ GPU thật nào cũng sẽ bắt được lỗi ngay lần chạy đầu.
              </>
            }
          >
            <>
              <code>emu</code> gives every thread a private copy of <code>bins</code> and adds the copies at the end,
              so increments never meet; any real GPU would catch the bug on its first run.
            </>
          </Translated>
        ),
        why: (
          <Translated vi="Chỉ có một mảng, được AddressSanitizer kiểm tra. Và một GPU thật cũng có thể qua: race condition chỉ lộ ra khi hai phép tăng thực sự chồng lên nhau.">
            {'There is one array, checked by AddressSanitizer. And a real GPU may pass too: a race only shows when two increments actually overlap.'}
          </Translated>
        ),
      },
      {
        label: tr('Increments commute, so the order of the threads cannot matter: the kernel is correct and needs no other test.', 'Các phép tăng có tính giao hoán, nên thứ tự của các luồng không quan trọng: kernel đúng và không cần kiểm tra nào khác.'),
        why: tr('Commuting makes the order irrelevant, not the read-modify-write safe. Two threads that load the same old count lose one increment.', 'Tính giao hoán khiến thứ tự không quan trọng, chứ không làm cho thao tác đọc-sửa-ghi an toàn. Hai luồng cùng nạp một giá trị đếm cũ sẽ làm mất một phép tăng.'),
      },
      {
        label: (
          <Translated
            vi={
              <>
                <code>emu</code> chỉ kiểm tra biên của mảng, nên nó cũng sẽ cho qua một kernel đếm vào sai bin.
              </>
            }
          >
            <>
              <code>emu</code> only checks array bounds, so it would pass a kernel that counted into the wrong bins as well.
            </>
          </Translated>
        ),
        why: tr('Every lab compares the result with a CPU reference, so wrong bins fail. What emu cannot produce is an interleaving inside one increment.', 'Mọi lab đều so kết quả với một bản tham chiếu trên CPU, nên sai bin sẽ thất bại. Điều emu không tạo ra được là một sự đan xen bên trong một phép tăng.'),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>bins[b]++</code> là một lần nạp, một phép cộng và một lần ghi. Một cập nhật bị mất cần một luồng khác chạy
            xen vào giữa lần nạp và lần ghi, và <code>emu</code> không bao giờ chuyển luồng ở đó. Vì vậy mọi lab về cập
            nhật dùng chung đều build thêm cho <code>race</code>, backend chạy các luồng đồng thời dưới ThreadSanitizer.
            Mục <a href={href('ch22', '22.2')}>22.2</a> gọi mẫu này là scatter: hễ hai luồng có thể ghi cùng một đầu ra,
            bạn cần một phép toán nguyên tử (atomic), và một công cụ thấy được khi nó vắng mặt. Đã học ở{' '}
            <a href={href('ch09', '9.2')}>ch. 9 §9.2</a>.
          </p>
        }
      >
        <p>
          <code>bins[b]++</code> is a load, an add and a store. A lost update needs another thread to run between the
          load and the store, and <code>emu</code> never switches threads there. That is why every lab about shared
          updates also builds for <code>race</code>, which runs the threads concurrently under ThreadSanitizer. Section{' '}
          <a href={href('ch22', '22.2')}>22.2</a> calls this pattern scatter: whenever two threads may write one output,
          you need an atomic, and a tool that can see its absence. Taught in <a href={href('ch09', '9.2')}>ch. 9 §9.2</a>.
        </p>
      </Translated>
    ),
    source: 'labs/README.md (the emulators); ch. 9 §9.2, pp. 203–209',
    vi: { source: 'labs/README.md (các trình giả lập); ch. 9 §9.2, tr. 203–209' },
  },
  'ch22/start/q-tree-writers': {
    kind: 'choice',
    id: 'ch22/start/q-tree-writers',
    prompt: (
      <Translated vi="Trong một bước của cây rút gọn theo khối ở chương 10, có bao nhiêu luồng ghi vào mỗi ô tổng riêng phần?">
        <>In one step of chapter 10&rsquo;s block reduction tree, how many threads write each partial-sum slot?</>
      </Translated>
    ),
    options: [
      { label: tr('Exactly one: a barrier between steps, and no atomics.', 'Đúng một: một rào chắn giữa các bước, và không cần atomic.'), correct: true },
      {
        label: tr('Two, one for each operand, so every slot needs an atomic add.', 'Hai, mỗi toán hạng một luồng, nên mỗi ô cần một phép cộng nguyên tử.'),
        why: tr('Two values are read, but only the thread that owns the slot writes it.', 'Hai giá trị được đọc, nhưng chỉ luồng sở hữu ô mới ghi vào nó.'),
      },
      {
        label: tr('Every thread of the warp, so the warp must take turns at the slot.', 'Mọi luồng của warp, nên warp phải lần lượt ghi vào ô.'),
        why: tr('Each active thread has its own slot in a step; the warp does not share one.', 'Trong một bước, mỗi luồng đang hoạt động có ô riêng của nó; warp không dùng chung một ô.'),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Luồng t đọc hai tổng riêng phần và ghi vào một ô mà không luồng nào khác ghi trong bước đó: chủ sở hữu tính
            toán. Luồng đọc và luồng ghi của một ô chỉ gặp nhau qua các bước, và rào chắn (barrier) tách chúng ra. Mục{' '}
            <a href={href('ch22', '22.2')}>22.2</a> gọi cách này là lấy đầu ra làm trung tâm, xét từng vòng lặp một: mỗi
            luồng gom (gather) các đầu vào của đúng một đầu ra mà nó sở hữu. Đã học ở{' '}
            <a href={href('ch10', '10.3')}>ch. 10 §10.3</a>.
          </p>
        }
      >
        <p>
          Thread t reads two partial sums and writes one slot that no other thread writes in that step: the owner
          computes. A slot&rsquo;s reader and writer meet only across steps, and the barrier separates those. Section{' '}
          <a href={href('ch22', '22.2')}>22.2</a> calls this output-centric, judged one iteration at a time: each thread
          gathers the inputs of the one output it owns. Taught in <a href={href('ch10', '10.3')}>ch. 10 §10.3</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.3–10.4, pp. 227–234; ch. 22 §22.2, p. 534',
    vi: { source: 'ch. 10 §10.3–10.4, tr. 227–234; ch. 22 §22.2, tr. 534' },
  },
  'ch22/start/q-m3-rows': {
    kind: 'numeric',
    id: 'ch22/start/q-m3-rows',
    prompt: (
      <Translated
        vi={
          <>
            Mốc dự án M3 (lab 15.6) tính <code>Y</code> theo các ô khối 128 hàng × 128 cột, và các hàng chính là batch. Ở
            batch 3, bao nhiêu phần trăm số hàng mà một khối tính là hàng thật?
          </>
        }
      >
        <>
          Milestone M3 (lab 15.6) computes <code>Y</code> in block tiles of 128 rows by 128 columns, and the rows are the
          batch. At batch 3, what percentage of the rows a block computes are real?
        </>
      </Translated>
    ),
    answer: 2.34,
    tol: 0.05,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            3 / 128 = 2.34%. 125 hàng còn lại là phần đệm bằng 0 và kernel vẫn tính toán trên chúng, nên mọi batch từ 1 đến
            128 đều khởi chạy cùng một lưới và làm cùng một lượng việc. Vì thế một phép đo batching trên M3 cho thấy thời
            gian mỗi bước phẳng cho tới batch 128, dù kernel có bị giới hạn bởi bộ nhớ hay không. Mục{' '}
            <a href={href('ch22', '22.5')}>22.5</a> và lab 22.2 quay lại điểm này. Đã học ở{' '}
            <a href={href('ch15', 'labs')}>các lab của chương 15</a>.
          </p>
        }
      >
        <p>
          3 / 128 = 2.34%. The other 125 rows are zero padding and the kernel does their arithmetic anyway, so every
          batch from 1 to 128 launches the same grid and does the same work. A batching measurement on M3 therefore shows
          a flat time per step up to batch 128, whether or not the kernel is limited by memory. Section{' '}
          <a href={href('ch22', '22.5')}>22.5</a> and lab 22.2 come back to this. Taught in{' '}
          <a href={href('ch15', 'labs')}>chapter 15&rsquo;s labs</a>.
        </p>
      </Translated>
    ),
    source: 'labs/ch15/lab6_linear/README.md; notes/today/ch22-facts.md E3', // i18n-ignore: file-path citation
  },
};

// The chapter's four steps as the summary names them (p. 540), then the two sections that
// widen the frame.
const STEPS: { step: string; where: string; link?: string; what: string }[] = [
  { get step() { return tr('Problem formulation', 'Phát biểu bài toán'); }, where: '22.4', link: '22.4', get what() { return tr('Pose the problem so that a fast algorithm exists, even if that means giving up some exactness.', 'Đặt bài toán sao cho tồn tại một thuật toán nhanh, kể cả khi phải hy sinh một phần độ chính xác.'); } },
  { get step() { return tr('Algorithm selection', 'Chọn thuật toán'); }, where: '22.1', link: '22.1', get what() { return tr('Choose among algorithms for the same problem by what each trades on your hardware.', 'Chọn giữa các thuật toán cho cùng một bài toán theo điều mỗi thuật toán đánh đổi trên phần cứng của bạn.'); } },
  { get step() { return tr('Problem decomposition', 'Phân rã bài toán'); }, where: '22.2', link: '22.2', get what() { return tr('Decide which thread owns which output, or which input, and pay for that choice.', 'Quyết định luồng nào sở hữu đầu ra nào, hoặc đầu vào nào, và trả giá cho lựa chọn đó.'); } },
  { get step() { return tr('Optimisation', 'Tối ưu hóa'); }, get where() { return tr('chapters 2 to 21', 'chương 2 đến 21'); }, get what() { return tr('Everything else in the book: tiling, coalescing, privatisation, coarsening and the rest.', 'Mọi thứ còn lại trong sách: phân ô (tiling), hợp nhất truy cập bộ nhớ (coalescing), riêng tư hóa (privatization), gộp việc cho luồng (thread coarsening) và những kỹ thuật khác.'); } },
];
const FRAME: { step: string; where: string; what: string }[] = [
  { get step() { return tr("Amdahl's law for a whole application", 'Định luật Amdahl cho cả một ứng dụng'); }, where: '22.3', get what() { return tr('Decide which modules go on the GPU; whatever stays behind caps the speed-up.', 'Quyết định module nào chạy trên GPU; phần nào ở lại sẽ giới hạn tốc độ tăng.'); } },
  { get step() { return tr('Batching', 'Batching'); }, where: '22.5', get what() { return tr('Optimise one request’s latency, or the throughput of many requests at once.', 'Tối ưu độ trễ của một yêu cầu, hoặc thông lượng của nhiều yêu cầu cùng lúc.'); } },
];

export function S220() {
  const q = questionsS220;
  return (
    <>
      <Translated
        vi={
          <p>
            Mỗi chương cho đến giờ đều trao cho bạn một kỹ thuật: phân ô (tiling), riêng tư hóa (privatization), gộp việc cho
            luồng (thread coarsening), một mạng scan, một phép trộn (merge) bằng co-rank. Chương này lùi lại một bước, về
            những quyết định bạn đưa ra trước tất cả các kỹ thuật đó. Chính xác thì bài toán là gì? Thuật toán nào giải nó?
            Bạn cắt nó thành những phần mà các luồng có thể sở hữu như thế nào? Và kết quả có nhanh ở chỗ quan trọng, tức
            trong toàn bộ ứng dụng, hay không? Sách gọi đây là tư duy song song, và coi người lập trình là một cộng sự có
            thể định hình lại bài toán mà một nhà khoa học chuyên ngành mang đến, chứ không chỉ cài đặt nó.
          </p>
        }
      >
        <p>
          Every chapter so far has handed you a technique: tiling, privatisation, coarsening, a scan network, a merge by
          co-ranks. This one steps back to the decisions you make before any of them. What exactly is the problem? Which
          algorithm solves it? How do you cut it into pieces that threads can own? And is the result fast where it
          counts, in the whole application? The book calls this parallel thinking, and it casts the programmer as a
          collaborator who can reshape the problem a domain scientist brings, not only implement it.
        </p>
      </Translated>
      <Source pages="529" />
      <Translated
        vi={
          <p>
            Phần tóm tắt của chương nêu các bước theo thứ tự, và bảng dưới đây dùng chúng làm bản đồ. Hai mục mở rộng khung
            nhìn: định luật Amdahl áp dụng cho cả một chương trình, và lựa chọn giữa độ trễ và thông lượng.
          </p>
        }
      >
        <p>
          The chapter&rsquo;s summary names the steps in order, and the table below uses them as a map. Two sections widen
          the frame: Amdahl&rsquo;s law applied to a whole program, and the choice between latency and throughput.
        </p>
      </Translated>
      <Source section="22.6" pages="540" />
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-map>
          <thead>
            <tr>
              <th><Translated vi="Bước">Step</Translated></th>
              <th><Translated vi="Ở đâu">Where</Translated></th>
              <th><Translated vi="Quyết định">The decision</Translated></th>
            </tr>
          </thead>
          <tbody>
            {STEPS.map((s) => (
              <tr key={s.step} data-step={s.where}>
                <td>{s.step}</td>
                <td style={{ whiteSpace: 'nowrap' }}>{s.link ? <a href={href('ch22', s.link)}>{s.where}</a> : s.where}</td>
                <td>{s.what}</td>
              </tr>
            ))}
            {FRAME.map((s) => (
              <tr key={s.step} data-step={s.where}>
                <td>{s.step}</td>
                <td>
                  <a href={href('ch22', s.where)}>{s.where}</a>
                </td>
                <td>{s.what}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Translated
        vi={
          <p>
            Phát biểu bài toán đứng đầu trong danh sách của phần tóm tắt nhưng là mục thứ tư. Khóa học này giữ thứ tự của
            sách, nên bạn sẽ thấy việc chọn thuật toán và phân rã bài toán hoạt động trước khi gặp nó. Cho bước này, chương
            dẫn tài liệu tham khảo duy nhất của nó, một bài báo về điều mà tác giả gọi là tư duy điện toán.
          </p>
        }
      >
        <p>
          Formulation is first in the summary&rsquo;s list but fourth among the sections. This course keeps the
          book&rsquo;s order, so you will have watched selection and decomposition at work before you meet it. For it the
          chapter cites its single reference, a paper on what its author calls computational thinking.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              J. M. Wing, &ldquo;Computational thinking&rdquo;, <i>Communications of the ACM</i> 49(3), 2006, tr. 33–35. Tài
              liệu tham khảo duy nhất của chương (tr. 540; đã đối chiếu với bản ghi DOI, ch22-facts A10).
            </p>
          }
        >
          <p>
            J. M. Wing, &ldquo;Computational thinking&rdquo;, <i>Communications of the ACM</i> 49(3), 2006, pp. 33–35. The
            chapter&rsquo;s one reference (p. 540; checked against its DOI record, ch22-facts A10).
          </p>
        </Translated>
      </Aside>

      <h2><Translated vi="Một chương không có mã">A chapter without code</Translated></h2>
      <Translated
        vi={
          <p>
            Chương này không có đoạn mã nào, không có bài tập và không có số liệu thiết bị. Nó là phần lập luận về các quyết
            định đằng sau những chương khác, được minh họa bằng chính các chương đó. Vì vậy khóa học thêm hai lab của riêng
            mình, trên <a href={href('ch22', 'labs')}>trang lab</a>. Lab 22.1 cắt một phép nhân ma trận-vector lúc decode
            theo hai cách. Lab 22.2 là mốc dự án M8, một nghiên cứu về batching đo độ trễ và thông lượng trên chính thiết bị
            của bạn.
          </p>
        }
      >
        <p>
          There are no code listings in this chapter, no exercises and no device figures. It is reasoning about the
          decisions behind the other chapters, illustrated with them. So the course adds two labs of its own, on the{' '}
          <a href={href('ch22', 'labs')}>labs page</a>. Lab 22.1 cuts a decode-time matrix-vector product two ways. Lab
          22.2 is milestone M8, a batching study that measures latency and throughput on your own device.
        </p>
      </Translated>

      <h2><Translated vi="Bạn sẽ gặp những quyết định này ở đâu">Where you will meet these decisions</Translated></h2>
      <Translated
        vi={<p>Mỗi quyết định đều được đưa ra, một cách tường minh hoặc theo mặc định, trong đường đi GPU của một engine suy luận:</p>}
      >
        <p>
          Every one of them is made, explicitly or by default, in the GPU path of an inference engine:
        </p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>
              Kernel nào nhân trọng số ở kích thước batch nào là chuyện chọn thuật toán: llama.cpp giữ một kernel ma
              trận-vector và một kernel ma trận cho cùng một phép toán và chuyển đổi ở một batch cố định khoảng 8 token, hạ
              thấp hơn cho một số kiểu trọng số trên những GPU cụ thể nơi điểm giao được đo là đến sớm hơn (
              <a href={href('ch22', '22.1')}>22.1</a>).
            </li>
            <li>
              Luồng nào sở hữu đầu ra nào của một phép nhân trong bước decode, và các tổng riêng phần có gặp nhau trong một
              phép toán nguyên tử hay không, là chuyện phân rã (<a href={href('ch22', '22.2')}>22.2</a>).
            </li>
            <li>
              Những gì ở lại trên host trong vòng lặp decode, như lấy mẫu và khởi chạy kernel, là phần tuần tự giới hạn tốc
              độ tăng của mọi kernel bạn viết (<a href={href('ch22', '22.3')}>22.3</a>).
            </li>
            <li>
              Một server gom bao nhiêu yêu cầu vào batch, và mỗi người dùng phải chờ thêm bao lâu vì điều đó, là sự đánh
              đổi giữa độ trễ và thông lượng (<a href={href('ch22', '22.5')}>22.5</a>).
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            Which kernel multiplies the weights at which batch size is algorithm selection: llama.cpp keeps a
            matrix-vector kernel and a matrix kernel for the same operation and switches at a fixed batch of about 8 tokens,
            lowered for some weight types on particular GPUs where it was measured to cross sooner (
            <a href={href('ch22', '22.1')}>22.1</a>).
          </li>
          <li>
            Which thread owns which output of a decode-step product, and whether partial sums meet in an atomic, is
            decomposition (<a href={href('ch22', '22.2')}>22.2</a>).
          </li>
          <li>
            What stays on the host in the decode loop, such as sampling and kernel launches, is the serial part that caps
            the speed-up of every kernel you write (<a href={href('ch22', '22.3')}>22.3</a>).
          </li>
          <li>
            How many requests a server batches, and how much longer each user waits for it, is the trade between latency
            and throughput (<a href={href('ch22', '22.5')}>22.5</a>).
          </li>
        </ul>
      </Translated>
      <Translated
        vi={
          <p>
            Các ví dụ của chính sách là scan, sắp xếp, histogram và một bản đồ tĩnh điện, còn truy vấn LLM chỉ xuất hiện ở
            mục 22.5. Các ví dụ về suy luận trên những trang này là của khóa học này, và mỗi ví dụ đều được ghi rõ.
          </p>
        }
      >
        <p>
          The book&rsquo;s own examples are scans, sorts, histograms and an electrostatic map, with LLM queries only in
          section 22.5. The inference examples on these pages are the course&rsquo;s, and each is labelled.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={<p>Bốn cách ánh xạ sang suy luận này là của chúng tôi. Điểm chuyển của llama.cpp được kiểm chứng ở mục 22.1 (ch22-facts D6).</p>}
        >
          <p>These four mappings onto inference are ours. The llama.cpp switch is checked in section 22.1 (ch22-facts D6).</p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Trước khi bắt đầu">Before you start</Translated></h2>
      <Translated
        vi={
          <p>
            Năm câu hỏi về những gì chương này dựa vào từ các chương 1, 5, 9, 10 và 15. Mỗi lời giải thích chỉ ra nơi ý tưởng
            đã được dạy và mục ở đây dùng đến nó. Chúng sẽ quay lại trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Five questions on what this chapter leans on from chapters 1, 5, 9, 10 and 15. Each explanation points to where
          the idea was taught and to the section here that uses it. They come back in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={q['ch22/start/q-decode-amdahl']} label={tr('From chapter 1', 'Từ chương 1')} />
      <Ask q={q['ch22/start/q-gemv-ceiling']} label={tr('From chapter 5', 'Từ chương 5')} />
      <Ask q={q['ch22/start/q-emu-blind']} label={tr('From chapter 9 and the labs', 'Từ chương 9 và các lab')} />
      <Ask q={q['ch22/start/q-tree-writers']} label={tr('From chapter 10', 'Từ chương 10')} />
      <Ask q={q['ch22/start/q-m3-rows']} label={tr('From chapter 15', 'Từ chương 15')} />
    </>
  );
}
