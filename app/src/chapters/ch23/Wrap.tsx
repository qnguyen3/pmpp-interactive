import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { DEFAULTS, simulate } from '../../sim/chunkpipe';

// The summary page shows one host sequence of its own, for a question. Its checked copy is the
// region [shown] of labs/ch23/snippets/rerecord.cu, which the CUDA front-end check parses;
// Wrap.snippets.test.ts fails if the two drift apart and pins the pipeline numbers below to
// the stepper's model (src/sim/chunkpipe.ts).
export const REREC = `produceA<<<grid, block, 0, s1>>>(x, n);      // x = 1
cudaEventRecord(done, s1);
cudaStreamWaitEvent(s2, done, 0);
produceB<<<grid, block, 0, s1>>>(x, n);      // x = x + 1
cudaEventRecord(done, s1);
consumeC<<<grid, block, 0, s2>>>(y, x, n);   // y = 2 * x`;

// q-pp: two pipeline stages on two GPUs with a link between them, 8 micro-batches. It is the
// stepper's three-lane pipeline with the lanes renamed (GPU 0 = copy-in, link = kernel,
// GPU 1 = copy-out); the model takes whole-job totals, so each per-micro-batch time is times 8.
export const PP = { micro: 8, gpu: 4, link: 1 };
const ppRun = simulate({
  ...DEFAULTS,
  a: PP.gpu * PP.micro,
  b: PP.link * PP.micro,
  c: PP.gpu * PP.micro,
  n: PP.micro,
  streams: PP.micro,
  engines: 2,
  order: 'depth',
  overhead: 0,
  host: 'pinned',
  queues: Infinity,
  view: 'cuda',
});
export const PP_RESULT = { total: ppRun.total, serial: ppRun.serial, bubble: ppRun.bubble, fill: 2 * PP.gpu + PP.link };

export const questionsWrap: Record<string, Question> = {
  'ch23/wrap/q-rerecord': {
    kind: 'choice',
    id: 'ch23/wrap/q-rerecord',
    prompt: (
      <Translated
        vi={
          <>
            Trong chuỗi lệnh ở trên, <code>s1</code> và <code>s2</code> là hai stream và <code>consumeC</code> được định đọc{' '}
            <code>x</code> sau khi <code>produceB</code> đã cộng vào nó. Điều đó có được bảo đảm không?
          </>
        }
      >
        In the sequence above, <code>s1</code> and <code>s2</code> are two streams and <code>consumeC</code> is meant to
        read <code>x</code> after <code>produceB</code> has added to it. Is that guaranteed?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có: s2 chờ done, và done được ghi lần cuối sau produceB.">
            Yes: s2 waits on done, and done was last recorded after produceB.
          </Translated>
        ),
        why: (
          <Translated
            vi={
              <>
                Lần chờ được phát ở dòng 3. Nó lấy <code>done</code> như lúc đó; lần ghi ở dòng 5 thay đổi những gì các lần chờ
                về sau thấy, không phải lần chờ này.
              </>
            }
          >
            <>
              The wait was issued on line 3. It took <code>done</code> as it stood then; the record on line 5 changes what
              later waits see, not this one.
            </>
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không: lần chờ lấy done như lúc ở dòng 3, sau produceA.">
            No: the wait took done as it stood on line 3, after produceA.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có: host phát consumeC sau cùng, nên nó chạy sau cùng.">
            Yes: the host issued consumeC last, so it runs last.
          </Translated>
        ),
        why: (
          <Translated vi="Chỉ bên trong một stream. Hai stream không có thứ tự nào giữa chúng trừ khi một event hoặc một lần chờ của host đặt ra thứ tự.">
            Only inside one stream. Two streams have no order between them unless an event or a host wait gives them one.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không: ghi một event hai lần là lỗi, nên dòng 5 thất bại.">
            No: recording one event twice is an error, so line 5 fails.
          </Translated>
        ),
        why: (
          <Translated vi="NVIDIA cho phép điều đó: một lần ghi mới ghi đè thứ event đã chụp. Fig. 23.14 ghi lại ba event của nó trong mọi vòng lặp.">
            NVIDIA allows it: a new record overwrites what the event captured. Fig. 23.14 records its three events again in every iteration.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>cudaStreamWaitEvent</code> bắt công việc về sau trong <code>s2</code> chờ bất cứ thứ gì <code>done</code> đã
            chụp khi lần chờ được phát, mà ở dòng 3 là <code>produceA</code>. Một lần ghi về sau chỉ được thấy bởi các lần chờ
            phát sau nó, và ở đây không có lần nào. Vậy <code>consumeC</code> có thể chạy trong khi <code>produceB</code> vẫn
            đang cộng vào <code>x</code>, và các phần tử khác nhau của <code>y</code> có thể ra 2 hoặc 4. Hãy phát lần chờ sau
            lần ghi mà nó cần. Các vòng lặp của sách làm vậy: mỗi vòng lặp ghi, rồi chờ. Một event Metal lại hoạt động khác
            nữa: một bộ đếm mà lần chờ yêu cầu theo giá trị.
          </p>
        }
      >
        <p>
          <code>cudaStreamWaitEvent</code> makes later work in <code>s2</code> wait for whatever <code>done</code> had
          captured when the wait was issued, which on line 3 is <code>produceA</code>. A later record is seen only by waits
          issued after it, and there are none. So <code>consumeC</code> may run while <code>produceB</code> is still adding to{' '}
          <code>x</code>, and different elements of <code>y</code> may come out 2 or 4. Issue the wait after the record it
          needs. The book&rsquo;s loops do: each iteration records, then waits. A Metal event works differently again: a
          counter that a wait asks for by value.
        </p>
      </Translated>
    ),
    source: "our sequence; NVIDIA's runtime reference in notes/today/ch23-facts.md, A5 (Metal: C2); events as in §23.3, p. 560",
    vi: { source: 'chuỗi lệnh của khóa học này; tài liệu tham khảo runtime của NVIDIA trong notes/today/ch23-facts.md, A5 (Metal: C2); event như ở §23.3, p. 560' },
  },
  'ch23/wrap/q-decode': {
    kind: 'choice',
    id: 'ch23/wrap/q-decode',
    prompt: (
      <Translated
        vi={
          <>
            Một vòng lặp decode tạo một token mỗi bước. Host khởi chạy các kernel của bước; <code>cudaMemcpy</code> sao chép id
            của token được lấy mẫu vào một <code>int</code> trên host; host kiểm tra nó xem đã hết chuỗi chưa rồi khởi chạy bước
            tiếp theo. Một bản trace cho thấy GPU rảnh giữa các bước. Thay đổi nào giữ cho nó bận?
          </>
        }
      >
        A decode loop makes one token per step. The host launches the step&rsquo;s kernels; <code>cudaMemcpy</code> copies
        the sampled token id into a host <code>int</code>; the host checks it for the end of the sequence and launches the
        next step. A trace shows the GPU idle between steps. Which change keeps it busy?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Khởi chạy kernel lấy mẫu trong một stream có độ ưu tiên cao hơn.">
            Launch the sampling kernel in a stream of higher priority.
          </Translated>
        ),
        why: (
          <Translated vi="Trong khi host chờ lần sao chép thì không có gì khác được xếp hàng, nên độ ưu tiên, một gợi ý về các kernel đang chờ, chẳng có gì để ưu tiên.">
            While the host waits for the copy nothing else is queued, so a priority, a hint about pending kernels, has nothing to prefer.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Xếp hàng bước n + 1 trước khi đọc token n, và kiểm tra kết thúc chuỗi trễ một bước.">
            Queue step n + 1 before reading token n, and check for the end of the sequence one step late.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chuyển sang cudaMemcpyAsync vào cùng int đó, rồi đồng bộ ngay sau nó, trước lần khởi chạy tiếp theo.">
            Switch to cudaMemcpyAsync into the same int, then synchronise right after it, before the next launch.
          </Translated>
        ),
        why: (
          <Translated vi="Host vẫn chờ lần sao chép trước khi xếp hàng bất cứ thứ gì, và từ bộ nhớ phân trang (pageable) lần sao chép có thể hoàn toàn không bất đồng bộ.">
            The host still waits for the copy before it queues anything, and from pageable memory the copy may not be asynchronous at all.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Rải các kernel của mỗi bước ra nhiều stream hơn để chúng chồng lấp.">
            Spread each step's kernels over more streams so that they overlap.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi kernel của một bước cần kernel trước nó, nên có ít thứ chạy song song được, và khoảng trống giữa các bước vẫn còn.">
            Each kernel of a step needs the one before it, so little can run side by side, and the gap between steps stays.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Thời gian rảnh là một vòng khứ hồi: GPU xong, lần sao chép tới nơi, host đọc token, và chỉ khi đó bước tiếp theo mới được
            xếp hàng. Nếu các kernel của bước n + 1 đọc token từ bộ nhớ device, host không cần giá trị của nó để khởi chạy
            chúng. Hãy xếp hàng chúng trước, sao chép token vào bộ nhớ ghim bằng <code>cudaMemcpyAsync</code>, và chỉ chờ ở nơi
            bạn đọc nó. Nếu token đó kết thúc chuỗi, một bước thừa bị bỏ đi. Đó là lần sao chép của chuẩn trong chương này, được xếp hàng sớm
            và chờ một lần, áp dụng cho decode; vòng lặp generate của mlx-lm làm điều tương tự trên Metal, xếp hàng bước tiếp
            theo bằng <code>async_eval</code> trước khi đọc token hiện tại.
          </p>
        }
      >
        <p>
          The idle time is a round trip: the GPU finishes, the copy lands, the host reads the token, and only then is the
          next step queued. If step n + 1&rsquo;s kernels read the token from device memory, the host does not need its
          value to launch them. Queue them first, copy the token into pinned memory with <code>cudaMemcpyAsync</code>, and
          wait only where you read it. If the token ended the sequence, one extra step is thrown away. It is the
          chapter&rsquo;s norm copy, queued early and waited for once, applied to decoding; mlx-lm&rsquo;s generate loop
          does the same on Metal, queueing the next step with <code>async_eval</code> before it reads the current token.
        </p>
      </Translated>
    ),
    source: 'the norm copy of §23.3, pp. 560–561; notes/today/ch23-facts.md, A2, A6 and D6 (mlx-lm); the mapping to decoding is ours',
    vi: { source: 'lần sao chép của chuẩn ở §23.3, pp. 560–561; notes/today/ch23-facts.md, A2, A6 và D6 (mlx-lm); việc đối chiếu sang decode là của khóa học này' },
  },
  'ch23/wrap/q-pp': {
    kind: 'numeric',
    id: 'ch23/wrap/q-pp',
    prompt: (
      <Translated
        vi={
          <>
            Hai GPU mỗi GPU giữ một nửa số lớp của một mô hình, và một batch được cắt thành {PP.micro} micro-batch. Mỗi
            micro-batch tốn {PP.gpu} đơn vị thời gian trên GPU 0, {PP.link} đơn vị để đi qua đường kết nối, rồi {PP.gpu} đơn vị
            trên GPU 1. Mỗi GPU, và đường kết nối, xử lý một micro-batch mỗi lúc, theo thứ tự. Khi nào micro-batch cuối cùng rời
            GPU 1?
          </>
        }
      >
        Two GPUs each hold half of a model&rsquo;s layers, and a batch is cut into {PP.micro} micro-batches. Each
        micro-batch takes {PP.gpu} time units on GPU 0, {PP.link} unit to cross the link, then {PP.gpu} units on GPU 1.
        Each GPU, and the link, handles one micro-batch at a time, in order. When does the last micro-batch leave GPU 1?
      </Translated>
    ),
    answer: PP_RESULT.total,
    tol: 0,
    unit: 'units',
    explain: (
      <Translated
        vi={
          <p>
            Đó là pipeline của phần 23.3 với các làn được đổi tên: GPU 0 là copy-in, đường kết nối là kernel, GPU 1 là
            copy-out. Micro-batch đầu tiên cần {PP.gpu} + {PP.link} + {PP.gpu} = {PP_RESULT.fill} đơn vị để đi hết; sau đó giai
            đoạn chậm nhất cho ra một micro-batch mỗi {PP.gpu}: {PP_RESULT.fill} + {PP.micro - 1} × {PP.gpu} ={' '}
            {PP_RESULT.total}, so với {PP_RESULT.serial} khi làm lần lượt. Đường kết nối được giấu vì nó nhanh hơn cả hai GPU.
            Mỗi GPU vẫn rảnh {PP_RESULT.bubble} đơn vị trong lúc pipeline được nạp hoặc rút, dù có bao nhiêu
            micro-batch: càng nhiều micro-batch thì tỉ lệ của bong bóng đó càng nhỏ, chứ không phải kích thước của nó.
          </p>
        }
      >
        <p>
          It is section 23.3&rsquo;s pipeline with the lanes renamed: GPU 0 is the copy in, the link the kernel, GPU 1 the
          copy out. The first micro-batch needs {PP.gpu} + {PP.link} + {PP.gpu} = {PP_RESULT.fill} units to get through;
          after it the slowest stage releases one every {PP.gpu}: {PP_RESULT.fill} + {PP.micro - 1} × {PP.gpu} ={' '}
          {PP_RESULT.total}, against {PP_RESULT.serial} one after another. The link is hidden because it is faster than
          either GPU. Each GPU still idles {PP_RESULT.bubble} units while the pipeline fills or drains, however many
          micro-batches there are: more of them shrink that bubble&rsquo;s share, not its size.
        </p>
      </Translated>
    ),
    source: "the stepper's model (ours, notes/today/ch23-facts.md, E1) on the overlap of §23.3, pp. 553–555; \"bubble\" as PyTorch's pipelining page uses it, D5",
    vi: { unit: 'đơn vị', source: 'mô hình của trình chạy từng bước (của khóa học này, notes/today/ch23-facts.md, E1) về sự chồng lấp ở §23.3, pp. 553–555; "bubble" (bong bóng) theo nghĩa trang pipelining của PyTorch dùng, D5' },
  },
  'ch23/wrap/q-ex2': {
    kind: 'numeric',
    id: 'ch23/wrap/q-ex2',
    prompt: (
      <Translated
        vi={
          <>
            Lời gọi <code>MPI_Send(ptr_a, 1000, MPI_FLOAT, 2000, 4, MPI_COMM_WORLD)</code> đã chuyển 4,000 byte. Mỗi phần tử
            có kích thước bao nhiêu?
          </>
        }
      >
        The call <code>MPI_Send(ptr_a, 1000, MPI_FLOAT, 2000, 4, MPI_COMM_WORLD)</code> moved 4,000
        bytes. What is the size of each element?
      </Translated>
    ),
    answer: 4,
    tol: 0,
    unit: 'bytes',
    explain: (
      <Translated
        vi={
          <p>
            Số lượng là đối số thứ hai, 1,000; 2,000 là rank đích và 4 là tag. 4,000 / 1,000 = 4 byte, phương án (c) của sách.
            Một <code>MPI_FLOAT</code> có kích thước của một <code>float</code> trên host.
          </p>
        }
      >
        <p>
          The count is the second argument, 1,000; 2,000 is the destination rank and 4 the tag. 4,000 / 1,000 = 4 bytes,
          the book&rsquo;s option (c). An <code>MPI_FLOAT</code> has the size of a <code>float</code> on the host.
        </p>
      </Translated>
    ),
    source: 'ch. 23 §23.7, exercise 2, p. 576; parameters from Fig. 23.8 and the datatype sizes, p. 550',
    vi: { unit: 'byte', source: 'ch. 23 §23.7, bài tập 2, p. 576; tham số từ Fig. 23.8 và kích thước các kiểu dữ liệu, p. 550' },
  },
  'ch23/wrap/q-ex3': {
    kind: 'choice',
    id: 'ch23/wrap/q-ex3',
    prompt: (
      <Translated
        vi={
          <>
            Câu nào đúng? (a) <code>MPI_Send()</code> mặc định là chặn. (b) <code>MPI_Recv()</code> mặc định là chặn. (c)
            Thông điệp MPI phải có ít nhất 128 byte.
          </>
        }
      >
        Which is true? (a) <code>MPI_Send()</code> is blocking by default. (b) <code>MPI_Recv()</code> is
        blocking by default. (c) MPI messages must be at least 128 bytes.
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Chỉ (a), về MPI_Send.">Only (a), about MPI_Send.</Translated>,
        why: (
          <Translated
            vi={
              <>
                Đúng, nhưng <code>MPI_Recv</code> cũng ở cùng vị trí: sách đưa ra <code>MPI_Irecv</code> làm bản không chặn
                tương ứng.
              </>
            }
          >
            <>
              True, but <code>MPI_Recv</code> is in the same position: the book offers <code>MPI_Irecv</code> as its
              non-blocking counterpart.
            </>
          </Translated>
        ),
      },
      {
        label: <Translated vi="Chỉ (b), về MPI_Recv.">Only (b), about MPI_Recv.</Translated>,
        why: (
          <Translated
            vi={
              <>
                Đúng, nhưng <code>MPI_Send</code> cũng ở cùng vị trí: sách đưa ra <code>MPI_Isend</code> làm bản không chặn
                tương ứng.
              </>
            }
          >
            <>
              True, but <code>MPI_Send</code> is in the same position: the book offers <code>MPI_Isend</code> as its
              non-blocking counterpart.
            </>
          </Translated>
        ),
      },
      {
        label: <Translated vi="Chỉ (c), mức tối thiểu 128 byte.">Only (c), the 128-byte minimum.</Translated>,
        why: (
          <Translated vi="Không có gì trong chương đặt ra mức tối thiểu, và phép rút gọn chuẩn của nó gửi một số float 4 byte duy nhất.">
            Nothing in the chapter sets a minimum, and its norm reduction sends a single 4-byte float.
          </Translated>
        ),
      },
      { label: <Translated vi="Cả (a) và (b); (c) sai.">Both (a) and (b); (c) is false.</Translated>, correct: true },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Chương không bao giờ nói thẳng điều đó, nhưng nó trình bày <code>MPI_Isend</code> và <code>MPI_Irecv</code>, theo
            sau là <code>MPI_Wait</code>, như các phiên bản không chặn của hai lời gọi (p. 551), và nó gọi{' '}
            <code>MPI_Sendrecv</code> là chặn (p. 567). Vậy (a) và (b) đều đúng và (c) sai. Bài tập yêu cầu một câu; chúng tôi
            hiểu là nó có hai đáp án đúng và nói rõ như vậy, thay vì chọn một.
          </p>
        }
      >
        <p>
          The chapter never says it in so many words, but it presents <code>MPI_Isend</code> and <code>MPI_Irecv</code>,
          followed by <code>MPI_Wait</code>, as the non-blocking versions of the two calls (p. 551), and it calls{' '}
          <code>MPI_Sendrecv</code> blocking (p. 567). So (a) and (b) are both true and (c) is false. The exercise asks for
          one statement; we read it as having two right answers and say so, rather than pick one.
        </p>
      </Translated>
    ),
    source: 'ch. 23 §23.7, exercise 3, p. 576; §23.2, pp. 551, 553; §23.4, p. 567',
    vi: { source: 'ch. 23 §23.7, bài tập 3, p. 576; §23.2, pp. 551, 553; §23.4, p. 567' },
  },
};

const NAME: Record<string, string> = { metal: 'Metal', cuda: 'CUDA', hip: 'HIP', get emu() { return tr('the emulator', 'trình giả lập'); } };

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const q = questionsWrap;
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch23/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const backends = realBackends(reports);
  const tried = [...backends, ...(reports.emu ? ['emu'] : [])];
  const passing = tried.filter((b) => labStatus(reports[b], 'ch23/lab1_overlap', 'work') === 'pass');
  return (
    <>
      <h2 style={{ marginTop: 0 }}>
        <Translated vi="Những gì cần mang theo">What to carry forward</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách dùng một stencil để dạy ba cách trải công việc ra nhiều GPU. Thứ khóa học này giữ lại là phần chạy được trên
            một GPU, và thói quen hỏi, với mỗi lần chờ, ai đang chờ và chờ cái gì.
          </p>
        }
      >
        <p>
          The book uses one stencil to teach three ways of spreading work over many GPUs. What this course keeps is the part
          that works on one GPU, and the habit of asking, for every wait, who is waiting and for what.
        </p>
      </Translated>
      <Source section="23.6" pages="574–575" />
      <Translated
        vi={
          <ul data-carry>
            <li>
              Một lần khởi chạy, một lần sao chép bất đồng bộ và một lời gọi NCCL trả về khi công việc đã được xếp hàng. Việc công
              việc đã xong là điều chỉ một lần đồng bộ hoặc một event mới cho bạn biết.
            </li>
            <li>
              Một stream là một hàng đợi có thứ tự; công việc ở các stream khác nhau có thể chồng lấp. Để giấu các lần sao chép, hãy
              cắt công việc thành các chunk trên nhiều stream: tổng thời gian trở thành thời gian nạp pipeline cộng với giai
              đoạn chậm nhất của một chunk cho mỗi chunk tiếp theo. Chunk nhỏ hơn thu nhỏ bong bóng nạp và rút, nhưng
              mỗi chunk thêm chi phí cố định, nên có một số chunk tốt nhất cần đo.
            </li>
            <li>
              Một lần sao chép bất đồng bộ chỉ chồng lấp khi đi từ bộ nhớ host được ghim (pinned memory). Từ bộ nhớ phân trang nó vẫn
              hợp lệ, và lặng lẽ thôi bất đồng bộ.
            </li>
            <li>
              Event sắp thứ tự công việc giữa các stream mà không cần host. Một lần chờ lấy event như nó được ghi tại thời
              điểm đó.
            </li>
            <li>
              Độ ưu tiên của stream là một gợi ý cho các kernel còn đang chờ bắt đầu. Hãy phát công việc gấp trước, và đo. ROCm
              mặc định bỏ qua độ ưu tiên, và Metal không có độ ưu tiên.
            </li>
            <li>
              Việc chờ có thể dời chỗ. Với MPI host chờ mọi phép trao đổi; NCCL xếp phép trao đổi vào một stream; NVSHMEM gửi
              từ bên trong kernel và cần một rào chắn (barrier) để biết dữ liệu đã tới.
            </li>
            <li>
              Trên Apple silicon không có lần sao chép nào để giấu. Chồng lấp ở đó là CPU nạp và rút các buffer dùng chung trong khi
              GPU tính, với nhiều buffer đang dùng.
            </li>
          </ul>
        }
      >
        <ul data-carry>
          <li>
            A launch, an asynchronous copy and an NCCL call return once the work is queued. That the work is done is
            something only a synchronise or an event can tell you.
          </li>
          <li>
            A stream is an ordered queue; work in different streams may overlap. To hide copies, cut the job into chunks
            over several streams: the total becomes the time to fill the pipeline plus one chunk&rsquo;s slowest stage per
            further chunk. Smaller chunks shrink the fill-and-drain bubble, but every chunk adds fixed costs, so there is a
            best chunk count to measure.
          </li>
          <li>
            An asynchronous copy overlaps only from pinned host memory. From pageable memory it is legal, and it quietly
            stops being asynchronous.
          </li>
          <li>
            Events order work across streams without the host. A wait takes the event as it was recorded at that moment.
          </li>
          <li>
            A stream priority is a hint for kernels still waiting to start. Issue urgent work first, and measure. ROCm
            ignores priorities by default, and Metal has none.
          </li>
          <li>
            The waiting can move. With MPI the host waits for every exchange; NCCL queues the exchange in a stream;
            NVSHMEM sends from inside the kernel and needs a barrier to know the data has arrived.
          </li>
          <li>
            On Apple silicon there is no copy to hide. The overlap there is the CPU filling and draining shared buffers
            while the GPU computes, with several buffers in flight.
          </li>
        </ul>
      </Translated>

      <h2>
        <Translated vi="Bài kiểm tra tổng hợp">A mixed check</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Các câu này dựa trên cả chương và trên những tình huống mà các trang chưa trình bày, tiếp theo là hai bài tập của
            sách mà không trang nào khác giải.
            {asked ? ` Đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu, tính theo câu trả lời gần nhất của bạn cho mỗi câu.` : ''}
          </p>
        }
      >
        <p>
          These draw on the whole chapter and on situations the pages did not show, followed by the two of the book&rsquo;s
          exercises that no other page works.
          {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Chuỗi event, vòng lặp decode và pipeline hai GPU là của khóa học này. Chuỗi lệnh là front-end-checked và chưa
              bao giờ được chạy.
            </p>
          }
        >
          <p>
            The event sequence, the decode loop and the two-GPU pipeline are ours. The sequence is front-end-checked and has
            never been run.
          </p>
        </Translated>
      </Beyond>
      <Code
        tabs={[
          {
            label: tr('CUDA (not a lab file: for this question only)', 'CUDA (không phải tệp lab: chỉ dùng cho câu hỏi này)'),
            code: REREC,
            status: 'front-end-checked; never run',
            tiers: { x: 'global', y: 'global' },
          },
        ]}
      />
      <Ask q={q['ch23/wrap/q-rerecord']} label={tr('Read the code', 'Đọc mã')} />
      <Ask q={q['ch23/wrap/q-decode']} />
      <Ask q={q['ch23/wrap/q-pp']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={q['ch23/wrap/q-ex2']} label={tr('Book exercise 2', 'Bài tập 2 của sách')} />
      <Ask q={q['ch23/wrap/q-ex3']} label={tr('Book exercise 3', 'Bài tập 3 của sách')} />

      <h2>
        <Translated vi="Các bài tập của sách">The book&rsquo;s exercises</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương kết thúc bằng ba bài tập (p. 576). Sách không in đáp án, nên đáp án ở đây là của khóa học này. Hai bài có
            nhiều hơn một đáp án bảo vệ được, và các trang nói rõ điều đó thay vì chọn một.
          </p>
        }
      >
        <p>
          The chapter ends with three exercises (p. 576). The book prints no answers, so the answers here are ours. Two of
          them have more than one defensible answer, and the pages say so instead of picking one.
        </p>
      </Translated>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="exercises">
        <thead>
          <tr>
            <th><Translated vi="Bài tập">Exercise</Translated></th>
            <th><Translated vi="Về">About</Translated></th>
            <th><Translated vi="Được giải ở đâu trong khóa học">Where it is worked here</Translated></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>1a–1d</td>
            <td><Translated vi="Số điểm được tính, số điểm vùng biên, số điểm rìa và bên trong mỗi tiến trình, cho một lưới 64 × 512 trên 16 rank">Points computed, halo points, boundary and internal points per process, for a 64 × 512 grid on 16 ranks</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    trình khám phá ở <a href={href('ch23', 'start')}>trang bắt đầu</a>, thiết lập sẵn &ldquo;Bài tập 1&rdquo;,
                    cho thấy cả hai cách hiểu: có và không có hai cột rìa mà kernel không bao giờ ghi
                  </>
                }
              >
                the explorer on the <a href={href('ch23', 'start')}>start page</a>, preset &ldquo;Exercise 1&rdquo;, which
                shows both readings: with and without the two edge columns the kernel never writes
              </Translated>
            </td>
          </tr>
          <tr>
            <td>1e</td>
            <td><Translated vi="Số byte mỗi tiến trình gửi ở giai đoạn 2">Bytes each process sends in stage 2</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    câu hỏi &ldquo;Bài tập 1e của sách&rdquo; ở <a href={href('ch23', 'start')}>trang bắt đầu</a>, và phần dự
                    đoán sau nó cho 32 GPU
                  </>
                }
              >
                the question &ldquo;Book exercise 1e&rdquo; on the <a href={href('ch23', 'start')}>start page</a>, and the
                prediction after it for 32 GPUs
              </Translated>
            </td>
          </tr>
          <tr>
            <td>2</td>
            <td><Translated vi={<>Kích thước phần tử từ một lời gọi <code>MPI_Send</code> và số byte của nó</>}>Element size from an <code>MPI_Send</code> call and its byte count</Translated></td>
            <td><Translated vi="câu hỏi “Bài tập 2 của sách” ở trên">the question &ldquo;Book exercise 2&rdquo; above</Translated></td>
          </tr>
          <tr>
            <td>3</td>
            <td><Translated vi="Câu nào về MPI là đúng">Which MPI statement is true</Translated></td>
            <td><Translated vi="câu hỏi “Bài tập 3 của sách” ở trên, với cả hai câu đúng">the question &ldquo;Book exercise 3&rdquo; above, with both true statements</Translated></td>
          </tr>
        </tbody>
      </table>

      <h2>
        <Translated vi="Những chỗ PDF có vẻ sai">Where the PDF looks wrong</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bản sách mà khóa học này dựa vào có một số lỗi có vẻ là sơ suất ở chương 23. Phần lớn là nhỏ; dấu ngoặc ở dòng 19
            sẽ khiến mã không biên dịch được.
          </p>
        }
      >
        <p>
          The copy this course was built from has a number of apparent slips in chapter 23. Most are small; the line-19
          parenthesis would stop the code compiling.
        </p>
      </Translated>
      <Translated
        vi={
          <ul data-list="errata">
            <li>p. 546 nói Fig. 23.5 trình bày năm hàm MPI thiết yếu; nó chỉ trình bày bốn.</li>
            <li>
              Fig. 23.14 dòng 19 (p. 557) thừa một dấu đóng ngoặc. Trang <a href={href('ch23', '23.4')}>23.3–23.5</a> trình
              bày dòng đã sửa, và unit test của nó chứng minh dòng được in không biên dịch được.
            </li>
            <li>p. 559 đặt lần chờ trên stream dưới ở dòng 32; trong hình nó là dòng 31, như p. 564 nói.</li>
            <li>
              pp. 561–562 nói về &ldquo;the cudaMemset operation&rdquo; trong khi mã và Fig. 23.15 liên quan đến{' '}
              <code>cudaMemsetAsync</code>.
            </li>
            <li>
              Định danh bị lệch: <code>l2Norm</code> (p. 561) thay cho <code>l2norm</code>, <code>cudaStreamsynchronize</code>{' '}
              (p. 562), <code>nccl_comm</code> trong Fig. 23.20 thay cho <code>ncclComm</code> trong Fig. 23.17, và{' '}
              <code>PE</code> cùng <code>numPEs</code> trong Fig. 23.25, không bao giờ được định nghĩa.
            </li>
            <li>
              <code>launch_jacobi_kernel</code> và <code>blockReduce</code> được dùng mà không bao giờ được định nghĩa, và không
              có kích thước khởi chạy nào được cho. Các đoạn mã của khóa học khai báo chúng.
            </li>
            <li>p. 562 gọi các mũi tên đồng bộ host của Fig. 23.15 là chấm đậm; chú giải vẽ chúng là nét đứt.</li>
            <li>
              Câu về biên tuần hoàn (p. 551) có thể hiểu theo cả hai cách. Mã thì vòng quanh; Figs. 23.2, 23.3 và 23.12 vẽ GPU
              0 không có lân cận phía trên.
            </li>
            <li>
              Bài tập 1 gọi kernel là stencil 5 điểm; Fig. 23.1 lấy trung bình bốn lân cận và chỉ đọc điểm giữa để tính phần
              dư.
            </li>
            <li>
              Kernel tích lũy tổng bình phương phần dư vào một biến tên <code>l2norm</code>, và văn bản lúc nói chuẩn, lúc nói
              bình phương của nó (p. 553).
            </li>
            <li>Mọi vòng lặp host đều kiểm tra <code>l2norm &gt; tol</code> trước vòng lặp đầu tiên; giá trị khởi đầu không bao giờ được trình bày.</li>
            <li>
              Fig. 23.13(b) bắt đầu phép trao đổi phía dưới ở chỗ phép phía trên kết thúc, đúng như một lời gọi MPI chặn làm,
              nhưng hình này đứng trước phần stream và không nói nó vẽ phiên bản nào.
            </li>
            <li>Bài tập 1 không nói cột rìa có được tính hay không; bài tập 3 có hai câu đúng mà chỉ yêu cầu một.</li>
            <li>
              Câu chữ, như được in: &ldquo;when may be interested&rdquo; (p. 558), &ldquo;responsible these rows&rdquo; (p.
              544), &ldquo;intialize&rdquo; (p. 566), &ldquo;calls nvshmem_finalize releases&rdquo; (p. 572), và thiếu dấu
              chấm sau &ldquo;[1]&rdquo; (p. 551).
            </li>
            <li>
              Không phải lỗi: chiều của độ ưu tiên. Sách không nói; NVIDIA thì có, số nhỏ hơn là độ ưu tiên cao hơn, và thứ tự{' '}
              <code>low, high</code> của sách là đúng.
            </li>
          </ul>
        }
      >
        <ul data-list="errata">
          <li>p. 546 says Fig. 23.5 shows five essential MPI functions; it shows four.</li>
          <li>
            Fig. 23.14 line 19 (p. 557) has an extra closing parenthesis. Page <a href={href('ch23', '23.4')}>23.3–23.5</a> shows
            the corrected line, and its unit test proves the printed one does not compile.
          </li>
          <li>p. 559 places the wait on the bottom stream at line 32; in the figure it is line 31, as p. 564 says.</li>
          <li>
            pp. 561–562 speak of &ldquo;the cudaMemset operation&rdquo; while the code and Fig. 23.15 concern{' '}
            <code>cudaMemsetAsync</code>.
          </li>
          <li>
            Identifiers drift: <code>l2Norm</code> (p. 561) for <code>l2norm</code>, <code>cudaStreamsynchronize</code> (p.
            562), <code>nccl_comm</code> in Fig. 23.20 for <code>ncclComm</code> in Fig. 23.17, and <code>PE</code> and{' '}
            <code>numPEs</code> in Fig. 23.25, never defined.
          </li>
          <li>
            <code>launch_jacobi_kernel</code> and <code>blockReduce</code> are used and never defined, and no launch
            dimensions are given. The course&rsquo;s snippets declare them.
          </li>
          <li>p. 562 calls the host-synchronisation arrows of Fig. 23.15 thick dotted; the legend draws them dashed.</li>
          <li>
            The sentence on periodic boundaries (p. 551) can be read either way. The code wraps around; Figs. 23.2, 23.3 and
            23.12 draw GPU 0 with no top neighbour.
          </li>
          <li>
            Exercise 1 calls the kernel a 5-point stencil; Fig. 23.1 averages the four neighbours and reads the centre only
            for the residue.
          </li>
          <li>
            The kernel accumulates the sum of squared residues in a variable called <code>l2norm</code>, and the text
            alternates between the norm and its square (p. 553).
          </li>
          <li>Every host loop tests <code>l2norm &gt; tol</code> before the first iteration; the starting value is never shown.</li>
          <li>
            Fig. 23.13(b) starts the bottom exchange where the top exchange ends, which is what a blocking MPI call does, but
            the figure comes before streams and does not say which version it draws.
          </li>
          <li>Exercise 1 does not say whether edge columns count; exercise 3 has two true statements and asks for one.</li>
          <li>
            Wording, as printed: &ldquo;when may be interested&rdquo; (p. 558), &ldquo;responsible these rows&rdquo; (p.
            544), &ldquo;intialize&rdquo; (p. 566), &ldquo;calls nvshmem_finalize releases&rdquo; (p. 572), and a missing full
            stop after &ldquo;[1]&rdquo; (p. 551).
          </li>
          <li>
            Not an error: which way priorities run. The book does not say; NVIDIA does, a lower number is a higher priority,
            and the book&rsquo;s <code>low, high</code> order is right.
          </li>
        </ul>
      </Translated>
      <Aside>
        <Translated vi={<p>Đây là cách đọc của chúng tôi về một bản PDF. Nếu bản của bạn khác, hãy tin bản của bạn.</p>}>
          <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Một số điều chương này nói đã thay đổi kể từ khi được viết, hoặc chưa bao giờ hoàn toàn như vậy. Mỗi mục đã được
            kiểm tra với tài liệu hiện hành của nhà cung cấp.
          </p>
        }
      >
        <p>
          Some of what the chapter says has changed since it was written, or was never quite so. Each item was checked
          against current vendor documentation.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch23-facts.md A1 to A16, B2, B3)', '2026-11-20 (notes/today/ch23-facts.md A1 đến A16, B2, B3)')}>
        <Translated
          vi={
            <p>
              ROCm 10.2 chấp nhận độ ưu tiên stream và mặc định bỏ qua chúng, và <code>asyncEngineCount</code> của nó là một
              hằng số, giống nhau trên mọi thiết bị. Sách không nói gì về AMD.
            </p>
          }
        >
          <p>
            ROCm 10.2 accepts stream priorities and ignores them by default, and its <code>asyncEngineCount</code> is a
            constant, the same on every device. The book says nothing about AMD.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <ul data-list="dated">
            <li>
              &ldquo;<code>cudaMemset</code> is blocking&rdquo; (p. 559) sai với bộ nhớ device: nó không bắt host chờ. Lý do
              dùng <code>cudaMemsetAsync</code> là lời gọi thông thường được xếp hàng trong default stream kiểu cũ, stream này
              chờ mọi stream chặn và bắt chúng chờ nó.
            </li>
            <li>
              &ldquo;<code>cudaMemcpyAsync()</code> requires&rdquo; bộ nhớ ghim (p. 563): tài liệu tham khảo của NVIDIA nói một
              lần sao chép từ bộ nhớ phân trang vẫn hoạt động nhưng quay về hành vi đồng bộ. Trên các hệ thống mà GPU đọc bảng trang
              của host, <code>cudaMallocHost</code> có thể hoàn toàn không khóa trang.
            </li>
            <li>
              Độ ưu tiên như cách chữa cho một kernel chiếm GPU (p. 558): NVIDIA gọi độ ưu tiên là một gợi ý không chiếm quyền
              công việc đang chạy, không bảo đảm thứ tự nào và không chạm vào lần sao chép.
            </li>
            <li>
              &ldquo;Most MPI implementations ... are built with&rdquo; khả năng hỗ trợ CUDA &ldquo;enabled&rdquo; (p. 552): chỉ
              MPICH tự bật nó, khi tìm thấy CUDA lúc cấu hình; Open MPI cần được build cho việc đó. Dòng GPU của MVAPICH2 giờ là
              MVAPICH-Plus. Tài liệu NCCL cảnh báo rằng dùng NCCL và MPI hỗ trợ CUDA trên cùng các GPU cùng lúc có thể gây
              deadlock.
            </li>
            <li>
              NCCL &ldquo;may add support for one-sided communication soon&rdquo; (p. 575): nó đã có rồi, từ phiên bản 2.29, bên
              cạnh các cửa sổ đối xứng (2.27) và một API phía device (2.28).
            </li>
            <li>
              Phần thiết lập mà các hình bỏ qua: <code>ncclCommInitRank</code> cần device CUDA được đặt trước (Fig. 23.17), và
              NVSHMEM khuyến nghị khởi tạo <code>nvshmemx_init_attr_t</code> bằng{' '}
              <code>NVSHMEMX_INIT_ATTR_INITIALIZER</code> (Fig. 23.24).
            </li>
            <li>
              Tài liệu tham khảo (p. 576): liên kết CUB không còn hoạt động (tài liệu đã chuyển sang{' '}
              <code>/cccl/unstable/</code>), và liên kết NVSHMEM chuyển hướng. Kho &ldquo;Multi GPU programming models&rdquo;
              của NVIDIA, mã của chính sách cho các vòng lặp này, vẫn hoạt động.
            </li>
          </ul>
        }
      >
        <ul data-list="dated">
          <li>
            &ldquo;<code>cudaMemset</code> is blocking&rdquo; (p. 559) is wrong for device memory: it does not make the host
            wait. The reason to use <code>cudaMemsetAsync</code> is that the plain call is queued in the legacy default
            stream, which waits for every blocking stream and makes them wait for it.
          </li>
          <li>
            &ldquo;<code>cudaMemcpyAsync()</code> requires&rdquo; pinned memory (p. 563): NVIDIA&rsquo;s reference says a
            copy from pageable memory still works but reverts to synchronous behaviour. On systems where the GPU reads host
            page tables, <code>cudaMallocHost</code> may not page-lock at all.
          </li>
          <li>
            Priorities as the fix for a kernel that occupies the GPU (p. 558): NVIDIA calls a priority a hint that does not
            preempt running work, guarantees no order and does not touch copies.
          </li>
          <li>
            &ldquo;Most MPI implementations ... are built with&rdquo; CUDA awareness &ldquo;enabled&rdquo; (p. 552): only
            MPICH turns it on by itself, when it finds CUDA at configure time; Open MPI needs to be built for it.
            MVAPICH2&rsquo;s GPU line is now MVAPICH-Plus. NCCL&rsquo;s documentation warns that using NCCL and CUDA-aware
            MPI on the same GPUs at once can deadlock.
          </li>
          <li>
            NCCL &ldquo;may add support for one-sided communication soon&rdquo; (p. 575): it already has, since version
            2.29, beside symmetric windows (2.27) and a device-side API (2.28).
          </li>
          <li>
            Setup the figures leave out: <code>ncclCommInitRank</code> needs the CUDA device set first (Fig. 23.17), and
            NVSHMEM recommends initialising <code>nvshmemx_init_attr_t</code> with{' '}
            <code>NVSHMEMX_INIT_ATTR_INITIALIZER</code> (Fig. 23.24).
          </li>
          <li>
            References (p. 576): the CUB link no longer resolves (the documentation moved under <code>/cccl/unstable/</code>
            ), and the NVSHMEM link redirects. NVIDIA&rsquo;s &ldquo;Multi GPU programming models&rdquo; repository, the
            book&rsquo;s own code for these loops, is live.
          </li>
        </ul>
      </Translated>

      <h2>
        <Translated vi="Những gì chương này bỏ ra ngoài">What this chapter left out</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương này được rút gọn. Nó giữ những gì chạy được trên một GPU và để phần ống nước của cụm máy lại trong sách.
            Không gì ở đây chạy MPI, NCCL hay NVSHMEM.
          </p>
        }
      >
        <p>
          This chapter is condensed. It keeps what works on one GPU and leaves the cluster plumbing in the book. Nothing
          here runs MPI, NCCL or NVSHMEM.
        </p>
      </Translated>
      <Translated
        vi={
          <ul data-left>
            <li>
              Bản thân kernel Jacobi được mô tả, không được dạy (Fig. 23.1, pp. 542–543). Phép rút gọn khối và cách ghi kết quả
              nguyên tử của nó là của chương 10.
            </li>
            <li>
              Phân rã 1D so với 2D, ngoài một công tắc trong trình khám phá ở{' '}
              <a href={href('ch23', 'start')}>trang bắt đầu</a> (p. 545).
            </li>
            <li>
              MPI đầy đủ (phần 23.2, pp. 545–553, Figs. 23.4 đến 23.11): bộ nhớ dùng chung so với phân tán, tiến trình, rank
              và communicator, <code>MPI_Init</code> cùng các hàm đi kèm, <code>MPI_Send</code>, <code>MPI_Recv</code> và{' '}
              <code>MPI_Sendrecv</code> với các tham số của chúng, <code>MPI_Isend</code> và <code>MPI_Irecv</code>, biên tuần
              hoàn, MPI hỗ trợ CUDA và <code>MPI_Allreduce</code>. Một đoạn ở trang{' '}
              <a href={href('ch23', '23.4')}>23.3–23.5</a> và vòng lặp đầu tiên bằng lời ở trang bắt đầu thay cho nó.
            </li>
            <li>
              Phần thiết lập của NCCL, các lời gọi gửi và nhận, lời gọi nhóm và toàn bộ vòng lặp NCCL (phần 23.4, pp. 563–568,
              Figs. 23.16 đến 23.21), được giữ dưới dạng một đoạn, các dòng trao đổi, một cột của bảng so sánh và một phần của
              hoạt cảnh vòng lặp ở trang 23.3–23.5. Trình khám phá ở đó chỉ mô hình hóa hai vòng lặp MPI.
            </li>
            <li>
              Symmetric heap của NVSHMEM, các lệnh put và get trong kernel, việc khởi tạo cùng MPI, rào chắn trên stream, gộp
              kernel và put hạt mịn so với hạt thô (phần 23.5, pp. 568–574, Figs. 23.22 đến 23.25), được giữ dưới dạng một đoạn, một
              cột của cùng bảng đó và cùng phần hoạt cảnh đó.
            </li>
            <li>Các khung chữ ký API (Figs. 23.5, 23.8 đến 23.11, 23.16, 23.18, 23.19 và 23.22) và tài liệu tham khảo (p. 576).</li>
          </ul>
        }
      >
        <ul data-left>
          <li>
            The Jacobi kernel itself is described, not taught (Fig. 23.1, pp. 542–543). Its block reduction and atomic commit
            are chapter 10&rsquo;s.
          </li>
          <li>
            The 1D against 2D decomposition beyond one switch of the <a href={href('ch23', 'start')}>start page</a>
            &rsquo;s explorer (p. 545).
          </li>
          <li>
            MPI in full (section 23.2, pp. 545–553, Figs. 23.4 to 23.11): shared against distributed memory, processes,
            ranks and communicators, <code>MPI_Init</code> and its companions, <code>MPI_Send</code>, <code>MPI_Recv</code>{' '}
            and <code>MPI_Sendrecv</code> with their parameters, <code>MPI_Isend</code> and <code>MPI_Irecv</code>, periodic
            boundaries, CUDA-aware MPI and <code>MPI_Allreduce</code>. One paragraph on page{' '}
            <a href={href('ch23', '23.4')}>23.3–23.5</a> and the first loop in words on the start page stand in for it.
          </li>
          <li>
            NCCL&rsquo;s setup, its send and receive calls, group calls and the full NCCL loop (section 23.4, pp. 563–568,
            Figs. 23.16 to 23.21), kept as one paragraph, the exchange lines, a column of the comparison table and a beat
            of the loop animation on page 23.3–23.5. The explorer there models only the two MPI loops.
          </li>
          <li>
            NVSHMEM&rsquo;s symmetric heap, its in-kernel puts and gets, its initialisation with MPI, the on-stream barrier,
            kernel fusion and fine- against coarse-grained puts (section 23.5, pp. 568–574, Figs. 23.22 to 23.25), kept as
            one paragraph, a column of the same table and the same beat.
          </li>
          <li>The API signature boxes (Figs. 23.5, 23.8 to 23.11, 23.16, 23.18, 23.19 and 23.22) and the references (p. 576).</li>
        </ul>
      </Translated>
      <Translated
        vi={
          <p>
            Nếu bạn cần các phiên bản cho cụm máy, kho &ldquo;Multi GPU programming models&rdquo; của NVIDIA, tài liệu tham
            khảo số 4 của sách, có mã chạy được cho từng vòng lặp.
          </p>
        }
      >
        <p>
          If you need the cluster versions, NVIDIA&rsquo;s &ldquo;Multi GPU programming models&rdquo; repository, the
          book&rsquo;s reference 4, holds runnable code for each loop.
        </p>
      </Translated>

      <h2>
        <Translated vi="Dự án đến lúc này">The project so far</Translated>
      </h2>
      <Translated
        vi={
          <p data-project={passing.join(',') || 'none'}>
            Chương 23 không thêm mốc dự án nào. Dự án của khóa học chạy trên một GPU và không bao giờ chia một mô hình ra nhiều
            thiết bị, và lab 23.1 là tùy chọn.{' '}
            {tried.length === 0
              ? 'Chưa có kết quả lab nào được ghi trên máy này.'
              : passing.length
                ? `runStaged hoặc runInFlight của chính bạn đã qua trên ${passing.map((b) => NAME[b] ?? b).join(' và ')} trên máy này.`
                : 'Phiên bản lab 23.1 của chính bạn chưa qua trên máy này.'}{' '}
            Xem <a href={href('ch23', 'labs')}>lab</a>.
          </p>
        }
      >
        <p data-project={passing.join(',') || 'none'}>
          Chapter 23 adds no milestone. The course project runs on one GPU and never splits a model across devices, and lab
          23.1 is optional.{' '}
          {tried.length === 0
            ? 'No lab results are recorded on this machine yet.'
            : passing.length
              ? `Your own runStaged or runInFlight passes on ${passing.map((b) => NAME[b] ?? b).join(' and ')} on this machine.`
              : 'Your own version of lab 23.1 has not passed on this machine yet.'}{' '}
          See the <a href={href('ch23', 'labs')}>lab</a>.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Thứ thực sự đi vào dự án là thói quen từ trang <a href={href('ch23', '23.3')}>23.3</a>: khi mô hình của bạn sao
            chép một token về mỗi bước, hãy xếp hàng lần sao chép sớm, giữ buffer host được ghim, và chỉ chờ ở nơi cần giá trị.
          </p>
        }
      >
        <p>
          What does carry into the project is the habit from page <a href={href('ch23', '23.3')}>23.3</a>: when your model
          copies a token back every step, queue the copy early, keep the host buffer pinned, and wait only where the value
          is needed.
        </p>
      </Translated>

      <h2>
        <Translated vi="Tiếp theo">Where next</Translated>
      </h2>
      <Translated
        vi={
          <p>
            <a href={href('ch24', 'start')}>Chương 24</a> khép lại cuốn sách. Nó không có mã: nó đặt mọi chương vào ba bước của
            sách và đặt một bảng phần cứng năm 2007 cạnh năm 2025, kể cả đường kết nối host mà chương này đã giấu chi phí của
            nó.
          </p>
        }
      >
        <p>
          <a href={href('ch24', 'start')}>Chapter 24</a> closes the book. It has no code: it places every chapter in the
          book&rsquo;s three steps and sets one table of 2007 hardware against 2025, including the host link whose cost
          this chapter has been hiding.
        </p>
      </Translated>
    </>
  );
}
