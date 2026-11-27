import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';

// Prerequisite questions drawn from the notes of chapters 4, 6 and 9.
export const questionsS100: Record<string, Question> = {
  'ch10/start/q-lane': {
    kind: 'numeric',
    id: 'ch10/start/q-lane',
    prompt: (
      <Translated
        vi={
          <>
            Một khối 256 luồng được cắt thành các warp 32 luồng. Luồng 75 nằm trong một warp nào đó, ở một vị trí nào đó,
            gọi là <b>lane</b> của nó. Chỉ số lane của nó là bao nhiêu?
          </>
        }
      >
        A block of 256 threads is cut into warps of 32. Thread 75 sits in some warp at some position, its <b>lane</b>.
        What is its lane index?
      </Translated>
    ),
    answer: 11,
    explain: (
      <Translated
        vi={
          <p>
            Warp <i>n</i> chứa các luồng 32<i>n</i> đến 32<i>n</i> + 31, nên 75 = 2 × 32 + 11: warp 2, lane 11. Lane là{' '}
            <code>threadIdx.x % 32</code> và warp là <code>threadIdx.x / 32</code>. Trên một thiết bị AMD gfx9, nơi warp
            rộng 64, cùng luồng đó là lane 11 của warp 1. Đó là lý do các kernel cấp warp ở phần sau của chương này tính
            lane và warp từ độ rộng warp, không bao giờ từ một hằng số 32 viết cứng.
          </p>
        }
      >
        <p>
          Warp <i>n</i> holds threads 32<i>n</i> to 32<i>n</i> + 31, so 75 = 2 × 32 + 11: warp 2, lane 11. The lane is{' '}
          <code>threadIdx.x % 32</code> and the warp <code>threadIdx.x / 32</code>. On an AMD gfx9 device, whose warps are
          64 wide, the same thread is lane 11 of warp 1. That is why the warp-level kernels later in this chapter compute
          lane and warp from the warp width, never from a literal 32.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.4, p. 75; AMD warp width: notes/today/ch05-facts.md B5',
    vi: { source: 'ch. 4 §4.4, p. 75; độ rộng warp của AMD: notes/today/ch05-facts.md B5' },
  },
  'ch10/start/q-divwarps': {
    kind: 'numeric',
    id: 'ch10/start/q-divwarps',
    prompt: (
      <Translated
        vi={
          <>
            Một khối 128 luồng chạy <code>if (threadIdx.x &lt; 48) {'{ ... }'}</code>, với warp 32 luồng. Có bao nhiêu warp
            bị phân kỳ tại nhánh này?
          </>
        }
      >
        A 128-thread block runs <code>if (threadIdx.x &lt; 48) {'{ ... }'}</code>, with warps of 32. How many warps
        diverge at this branch?
      </Translated>
    ),
    answer: 1,
    unit: 'warp',
    explain: (
      <Translated
        vi={
          <p>
            Warp 0 (luồng 0 đến 31) đi vào nhánh cả cụm. Warp 1 (32 đến 63) tách ra tại 48: luồng 32 đến 47 đi vào, 48 đến
            63 thì không, nên nó chạy cả hai đường. Warp 2 và 3 bỏ qua nhánh cả cụm. Điều quyết định phân kỳ là điều kiện
            có khác nhau <em>bên trong</em> một warp hay không, chứ không phải có bao nhiêu luồng đi vào nhánh. Mục 10.4
            sắp xếp lại một kernel dựa đúng trên quy tắc này.
          </p>
        }
      >
        <p>
          Warp 0 (threads 0 to 31) takes the branch as a whole. Warp 1 (32 to 63) splits at 48: threads 32 to 47 take it,
          48 to 63 do not, so it runs both paths. Warps 2 and 3 skip it as a whole. What decides divergence is whether
          the condition differs <em>inside</em> a warp, not how many threads take the branch. Section 10.4 rearranges a
          kernel on exactly this rule.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.5, pp. 79–82',
    vi: { unit: 'warp' },
  },
  'ch10/start/q-gridbarrier': {
    kind: 'choice',
    id: 'ch10/start/q-gridbarrier',
    prompt: (
      <Translated vi={<>Vì sao các khối của một lần khởi chạy kernel thông thường không thể chờ nhau tại một rào chắn (barrier)?</>}>
        <>Why can the blocks of an ordinary kernel launch not wait for each other at a barrier?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Các khối chạy không theo thứ tự nào được hứa trước và không nhất thiết đều thường trú cùng lúc, nên một khối đang chờ có thể không bao giờ được giải phóng.">
            Blocks run in no promised order and need not all be resident, so a waiting block may never be released.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Mỗi khối có bộ nhớ chia sẻ riêng, nên các khối không giữ dữ liệu chung nào mà rào chắn có thể bảo vệ.">
            Each block has its own shared memory, so blocks hold no common data that a barrier could protect.
          </Translated>
        ),
        why: (
          <Translated vi="Các khối vẫn có thể chia sẻ kết quả qua bộ nhớ toàn cục. Trở ngại là việc lập lịch, không phải việc thiếu dữ liệu để trao đổi.">
            Blocks could still share results through global memory. The obstacle is scheduling, not the lack of data to exchange.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một rào chắn giữa các khối sẽ đúng, nhưng quá chậm nên không đáng cung cấp.">
            A barrier across blocks would be correct, but too slow to be worth providing.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là một lập luận về chi phí. Vấn đề nằm ở tính đúng đắn: một lần chờ như thế có thể không bao giờ kết thúc.">
            That is a cost argument. The problem is correctness: such a wait may never end.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các khối chạy theo thứ tự khởi chạy, nên khi khối b chạy thì mọi khối có số nhỏ hơn đã xong.">
            Blocks run in launch order, so by the time block b runs, every lower-numbered block has finished.
          </Translated>
        ),
        why: (
          <Translated vi="Không có thứ tự nào được hứa. Khối 7 có thể xong trước khi khối 0 bắt đầu.">
            No order is promised. Block 7 may finish before block 0 starts.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Runtime giao trọn từng khối cho phần cứng khi tài nguyên được giải phóng, không theo thứ tự nào được hứa, và
            một lưới (grid) lớn không bao giờ thường trú toàn bộ cùng lúc. Một rào chắn chỉ an toàn giữa những luồng chắc
            chắn đang chạy cùng nhau: các luồng của một khối. Sách có nhắc đến một rào chắn toàn lưới qua cooperative
            groups, vốn chỉ hoạt động dưới những hạn chế khiến mọi khối đều thường trú. Mục 10.9 phải hoàn tất một phép
            tổng giữa các khối mà không có rào chắn như vậy.
          </p>
        }
      >
        <p>
          The runtime assigns whole blocks to the hardware as resources free up, in no promised order, and a large grid
          is never all resident at once. A barrier is safe only among threads that are guaranteed to be running
          together: the threads of one block. The book mentions a grid-wide barrier through cooperative groups, which
          works only under restrictions that make every block resident. Section 10.9 has to finish a sum across blocks
          without such a barrier.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.3, pp. 70–73',
  },
  'ch10/start/q-sectors': {
    kind: 'numeric',
    id: 'ch10/start/q-sectors',
    prompt: (
      <Translated
        vi={
          <>
            32 luồng của một warp thực thi <code>x = in[2*t];</code> với <code>t</code> = 0 đến 31, trên một mảng float
            4 byte đã căn chỉnh. NVIDIA ghi trong tài liệu rằng bộ nhớ toàn cục được truy cập theo các giao dịch 32 byte.
            Một lần nạp này chạm tới bao nhiêu mẩu bộ nhớ 32 byte?
          </>
        }
      >
        The 32 threads of a warp execute <code>x = in[2*t];</code> with <code>t</code> = 0 to 31, on an aligned array
        of 4-byte floats. NVIDIA documents that global memory is accessed in 32-byte transactions. How many 32-byte
        pieces of memory does this one load touch?
      </Translated>
    ),
    answer: 8,
    unit: 'transactions',
    explain: (
      <Translated
        vi={
          <p>
            Warp đọc các phần tử 0, 2, 4, …, 62: byte 0 đến 251, trải trên tám mẩu 32 byte. Nếu đọc{' '}
            <code>in[t]</code> thay vào đó thì chỉ chạm các phần tử 0 đến 31, byte 0 đến 127: bốn mẩu. Bước nhảy (stride)
            2 làm gấp đôi số giao dịch và bỏ phí một nửa số byte được chuyển tới. Mục 10.5 đếm điều này cho một kernel rút
            gọn mà các luồng đang hoạt động đọc xa nhau hơn sau mỗi bước.
          </p>
        }
      >
        <p>
          The warp reads elements 0, 2, 4, …, 62: bytes 0 to 251, which span eight 32-byte pieces. Reading{' '}
          <code>in[t]</code> instead would touch elements 0 to 31, bytes 0 to 127: four pieces. A stride of 2 doubles the
          transactions and leaves half of every byte delivered unused. Section 10.5 counts this for a reduction kernel
          whose active threads read further apart with every step.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.1, pp. 124–132; the 32-byte unit is NVIDIA\'s (notes/today/ch06-facts.md A2)',
    vi: { unit: 'giao dịch', source: 'ch. 6 §6.1, pp. 124–132; đơn vị 32 byte là của NVIDIA (notes/today/ch06-facts.md A2)' },
  },
  'ch10/start/q-lostupdate': {
    kind: 'choice',
    id: 'ch10/start/q-lostupdate',
    prompt: (
      <Translated
        vi={
          <>
            256 luồng, mỗi luồng thực thi <code>*sum += x[threadIdx.x];</code> trên một <code>float</code> trong bộ nhớ
            toàn cục (global memory), không dùng atomic. Cuối cùng <code>*sum</code> có thể giữ giá trị gì?
          </>
        }
      >
        256 threads each execute <code>*sum += x[threadIdx.x];</code> on one <code>float</code> in global memory,
        without an atomic. What can <code>*sum</code> end up holding?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Tổng đúng: mỗi += là một câu lệnh duy nhất, nên các luồng lần lượt thay phiên.">
            The correct total: each += is a single statement, so the threads take turns.
          </Translated>
        ),
        why: (
          <Translated vi="Một câu lệnh, ba bước: đọc, cộng, ghi. Các luồng khác có thể chen vào giữa.">
            One statement, three steps: read, add, write. Other threads can come between them.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Tổng đúng, sai khác do làm tròn, vì phép cộng float có tính giao hoán và thứ tự không quan trọng.">
            The correct total up to rounding, because float addition is commutative and the order does not matter.
          </Translated>
        ),
        why: (
          <Translated vi="Tính giao hoán nói về thứ tự các toán hạng. Ở đây cả những lần cập nhật biến mất.">
            Commutativity is about the order of operands. Here whole updates disappear.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Tổng của một số giá trị, từ cả 256 xuống chỉ còn một: các lần cập nhật chồng nhau bị mất.">
            The sum of some of the values, from all 256 down to one: overlapping updates are lost.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chỉ x của luồng cuối cùng, không bao giờ là một tổng từng phần, vì mỗi lần ghi thay thế lần ghi trước.">
            Only the last thread's own x, never a partial sum, because each write replaces the one before.
          </Translated>
        ),
        why: (
          <Translated vi="Một luồng ghi old + x, nên giá trị nào nó đã đọc thì vẫn còn sống bên trong lần ghi của nó.">
            A thread writes old + x, so whatever value it read survives inside its write.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi <code>+=</code> đọc vị trí đó, cộng, rồi ghi lại. Nếu hai luồng đọc cùng một giá trị cũ, một lần ghi đè
            lên lần kia và phần đóng góp đó mất. Một phép toán nguyên tử (atomic) làm mỗi lần đọc-sửa-ghi trở nên không
            thể chia cắt, nên không gì bị mất. Nó không cố định <em>thứ tự</em> mà các lần cập nhật của các luồng được
            ghi vào, điều này quan trọng với một phép tổng float và sẽ quay lại ở mục 10.9.
          </p>
        }
      >
        <p>
          Each <code>+=</code> reads the location, adds, and writes back. If two threads read the same old value, one
          write overwrites the other and that contribution is gone. An atomic makes each read-modify-write indivisible,
          so nothing is lost. It does not fix the <em>order</em> in which the threads' updates land, which matters for a
          float sum and comes back in section 10.9.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.2, pp. 203–209',
  },
};

const KERNELS: { kernel: string; fig: string; sec: string; removes: string }[] = [
  { get kernel() { return tr('Simple', 'Đơn giản'); }, fig: 'Fig. 10.5', sec: '10.3', get removes() { return tr('nothing yet: the baseline, one block, in place', 'chưa gì cả: mốc cơ sở, một khối, tại chỗ'); } },
  { get kernel() { return tr('Convergent', 'Hội tụ'); }, fig: 'Fig. 10.8', sec: '10.4', get removes() { return tr('control divergence, and with it uncoalesced access (10.5)', 'phân kỳ điều khiển, và cùng với nó là truy cập không được hợp nhất (10.5)'); } },
  { get kernel() { return tr('Shared memory', 'Bộ nhớ chia sẻ'); }, fig: 'Fig. 10.10', sec: '10.6', get removes() { return tr('repeated trips to global memory', 'những chuyến đi lặp lại tới bộ nhớ toàn cục'); } },
  { get kernel() { return tr('Warp shuffles at the end', 'Shuffle warp ở cuối'); }, fig: 'Fig. 10.14', sec: '10.7', get removes() { return tr('barriers and shared memory in the last iterations', 'rào chắn và bộ nhớ chia sẻ ở các vòng lặp cuối'); } },
  { get kernel() { return tr('Two-stage warp reduction', 'Rút gọn warp hai giai đoạn'); }, fig: 'Fig. 10.16', sec: '10.8', get removes() { return tr('all but one barrier and one shared-memory hand-off', 'mọi rào chắn trừ một, và mọi lần chuyển giao qua bộ nhớ chia sẻ trừ một'); } },
  { get kernel() { return tr('Multi-block with an atomic', 'Nhiều khối với một atomic'); }, fig: 'Fig. 10.18', sec: '10.9', get removes() { return tr('the one-block limit of 2048 elements', 'giới hạn một khối 2048 phần tử'); } },
  { get kernel() { return tr('Coarsened', 'Gộp việc'); }, fig: 'Fig. 10.20', sec: '10.10', get removes() { return tr('hardware sitting idle in every block\'s shrinking tail', 'phần cứng ngồi không ở phần đuôi co dần của mỗi khối'); } },
];

export function S100() {
  const q = questionsS100;
  return (
    <>
      <Translated
        vi={
          <p>
            Mỗi bước decode đều kết thúc theo cùng một cách. Mô hình tạo ra một logit cho mỗi mục trong từ vựng, và phải
            có thứ gì đó biến hàng đó thành một token. Một <code>torch.softmax(logits / T, -1)</code> ổn định về số học
            cần logit lớn nhất trong hàng rồi một phép tổng trên cả hàng. Giải mã tham lam (greedy) là{' '}
            <code>torch.argmax(logits, -1)</code>. Bên trong mô hình, LayerNorm và RMSNorm cộng dọc theo từng hàng
            activation. Mỗi thao tác này nhận nhiều giá trị và trả về một. Kiểu tính toán đó là một{' '}
            <b>phép rút gọn (reduction)</b>, và chương này nói về cách viết tốt một phép như vậy trên GPU.
          </p>
        }
      >
        <p>
          Every decode step ends the same way. The model produces one logit per vocabulary entry, and something has to
          turn that row into a token. A numerically stable <code>torch.softmax(logits / T, -1)</code> needs the largest
          logit in the row and then a sum over the whole row. Greedy decoding is <code>torch.argmax(logits, -1)</code>.
          Inside the model, LayerNorm and RMSNorm sum along every row of activations. Each of these takes many values and
          returns one. That kind of computation is a <b>reduction</b>, and this chapter is about writing one well on a
          GPU.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không nhắc đến mô hình ngôn ngữ trong chương này. Mối liên hệ với softmax, argmax và các lớp chuẩn hóa
              là của khóa học này.
            </p>
          }
        >
          <p>
            The book does not mention language models in this chapter. The link to softmax, argmax and normalisation
            layers is ours.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Sách xây mọi kernel của mình cho phép rút gọn giản dị nhất có thể, một phép tổng, và nói lý do. Các luồng trong
            một phép rút gọn phải hợp tác, và mọi cách khiến chúng hợp tác đều có giá. Một phép tổng đủ nhỏ để từng cái giá
            có thể được phơi ra và loại bỏ lần lượt, mỗi lần một kernel. Những gì bạn học được áp dụng cho max, argmax và
            các phép rút gọn khác, dưới hai điều kiện về toán tử mà mục{' '}
            <a href={href('ch10', '10.1')}>10.1</a> và <a href={href('ch10', '10.2')}>10.2</a> phát biểu chính xác.
          </p>
        }
      >
        <p>
          The book builds all of its kernels for the plainest reduction there is, a sum, and says why. Threads in a
          reduction have to cooperate, and every way of making them cooperate has a cost. A sum is small enough that
          each cost can be exposed and removed in turn, one kernel at a time. What you learn carries over to max, argmax
          and the other reductions, under two conditions on the operator that sections{' '}
          <a href={href('ch10', '10.1')}>10.1</a> and <a href={href('ch10', '10.2')}>10.2</a> make precise.
        </p>
      </Translated>
      <Source pages="221" />

      <h2><Translated vi="Bảy kernel, mỗi kernel một nút thắt">Seven kernels, one bottleneck each</Translated></h2>
      <Translated
        vi={
          <p>
            Sau hai mục về bản thân phép rút gọn, chương viết bảy kernel. Mỗi kernel giữ lại những gì kernel trước đã làm
            đúng và gỡ thêm một trở ngại. Hãy giữ bảng này trong đầu như tấm bản đồ của chương.
          </p>
        }
      >
        <p>
          After two sections on reduction itself, the chapter writes seven kernels. Each one keeps what the previous one
          got right and removes one more obstacle. Keep this table in mind as the map of the chapter.
        </p>
      </Translated>
      <table className="data" data-part="kernel-map">
        <thead>
          <tr>
            <th><Translated vi="Kernel">Kernel</Translated></th>
            <th><Translated vi="Sách">Book</Translated></th>
            <th><Translated vi="Nó gỡ bỏ điều gì">What it removes</Translated></th>
          </tr>
        </thead>
        <tbody>
          {KERNELS.map((k) => (
            <tr key={k.fig}>
              <td>
                <a href={href('ch10', k.sec)}>{k.kernel}</a>
              </td>
              <td className="nowrap">
                {k.fig}, §{k.sec}
              </td>
              <td>{k.removes}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Phần lớn các trở ngại đã quen thuộc từ chương 4 đến 6: warp phân kỳ, truy cập bộ nhớ rải rác, lưu lượng tới bộ
            nhớ toàn cục, chờ tại rào chắn. Có hai cái mới. Các khối không thể chờ nhau, nên một phép tổng trên nhiều hơn
            một khối cần một cách khác để kết hợp kết quả của chúng. Và một cây phép cộng có ngày càng ít việc ở mỗi bước,
            nên phần lớn phần cứng giao cho nó ngồi không phần lớn thời gian.
          </p>
        }
      >
        <p>
          Most of the obstacles are familiar from chapters 4 to 6: divergent warps, scattered memory accesses, traffic to
          global memory, waiting at barriers. Two are new. Blocks cannot wait for each other, so a sum over more than one
          block needs another way to combine their results. And a tree of additions has less and less work in each step,
          so most of the hardware assigned to it is idle most of the time.
        </p>
      </Translated>
      <Source pages="227–248" />

      <h2><Translated vi="Chương kết thúc ở đâu">Where it ends</Translated></h2>
      <Translated
        vi={
          <p>
            Ba lab theo sau các mục: <a href={href('ch10', 'labs')}>một phép tổng viết theo ba cách, một cuộc săn các phép
            rút gọn bị hỏng, và lab 10.3</a>, chính là mốc dự án M4 của dự án khóa học: một softmax và argmax theo hàng trên
            logits, để giải mã tham lam chạy trọn vẹn từ đầu đến cuối trên các kernel của chính bạn.
          </p>
        }
      >
        <p>
          Three labs follow the sections: <a href={href('ch10', 'labs')}>one sum written three ways, a hunt for broken
          reductions, and lab 10.3</a>, which is milestone M4 of the course project: a row-wise softmax and argmax over
          logits, so that greedy decoding works end to end on your own kernels.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              <b>Màu sắc trong khóa học này</b> cho biết dữ liệu nằm ở đâu:{' '}
              <span className="chip reg">thanh ghi</span> <span className="chip shared">bộ nhớ chia sẻ</span>{' '}
              <span className="chip global">bộ nhớ toàn cục</span> <span className="chip const">bộ nhớ hằng</span>. Màu đỏ
              luôn nghĩa là có gì đó thất bại, màu xanh lá là khớp.
            </p>
          }
        >
          <p>
            <b>Colour in this course</b> means where data lives:{' '}
            <span className="chip reg">register</span> <span className="chip shared">shared memory</span>{' '}
            <span className="chip global">global memory</span> <span className="chip const">constant memory</span>. Red
            always means something failed, green that it matched.
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Sách chỉ viết CUDA. Các trang này bổ sung HIP và Metal ở bất cứ đâu chúng khác biệt, và nói rõ điều đó. Khác
            biệt lớn nhất nằm ở mục 10.7 và 10.8, ở cấp warp: HIP từ chối mặt nạ lane 32-bit của sách và có thể chạy warp
            rộng 64, còn Metal cung cấp phép tổng float một lệnh gọi trên một SIMD-group mà CUDA không có.
          </p>
        }
      >
        <p>
          The book writes CUDA only. These pages add HIP and Metal wherever they differ, and say so. The largest
          differences come in sections 10.7 and 10.8, at the warp level: HIP rejects the book's 32-bit lane mask and may
          run 64-wide warps, and Metal offers a one-call float sum across a SIMD-group that CUDA does not have.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch10-facts A4, B1, C1)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Các hàm <code>_sync</code> của HIP yêu cầu mặt nạ 64-bit; Metal có <code>simd_sum</code> và{' '}
              <code>simd_max</code> trên float; các hàm warp reduce của CUDA chỉ nhận số nguyên.
            </p>
          }
        >
          <p>
            HIP's <code>_sync</code> functions require a 64-bit mask; Metal has <code>simd_sum</code> and{' '}
            <code>simd_max</code> on floats; CUDA's warp reduce functions take integers only.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Trước khi bắt đầu">Before you start</Translated></h2>
      <Translated
        vi={
          <p>
            Năm câu hỏi về những gì chương này giả định từ chương 4, 6 và 9. Mỗi đáp án sẽ quay lại ở một mục sau; mỗi câu
            hỏi sẽ quay lại trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Five questions on what this chapter assumes from chapters 4, 6 and 9. Each answer comes back in a later section;
          each question comes back in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={q['ch10/start/q-lane']} label={tr('From chapter 4', 'Từ chương 4')} />
      <Ask q={q['ch10/start/q-divwarps']} label={tr('From chapter 4', 'Từ chương 4')} />
      <Ask q={q['ch10/start/q-gridbarrier']} label={tr('From chapter 4', 'Từ chương 4')} />
      <Ask q={q['ch10/start/q-sectors']} label={tr('From chapter 6', 'Từ chương 6')} />
      <Ask q={q['ch10/start/q-lostupdate']} label={tr('From chapter 9', 'Từ chương 9')} />
    </>
  );
}
