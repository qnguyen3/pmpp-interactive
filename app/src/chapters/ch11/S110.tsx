import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// Chapter 11's start page: what scan does in an inference stack, the chapter as a cost
// ledger of the book's formulas, a prerequisite check drawn from chapters 4 to 10, and
// the six labs. The ledger is a static table of the book's formulas with their pages.

const LEDGER: { what: string; formula: string; pages: string; sec: string; secLabel: string }[] = [
  {
    get what() {
      return tr('Barriers in one block scan, Fig. 11.3', 'Số rào chắn trong một lần scan khối, Fig. 11.3');
    },
    formula: '2 · log₂B', pages: 'p. 257', sec: '11.2', secLabel: '11.2', // i18n-ignore: formula
  },
  {
    get what() {
      return tr('Barriers in one block scan, Fig. 11.5 (two buffers)', 'Số rào chắn trong một lần scan khối, Fig. 11.5 (hai buffer)');
    },
    formula: 'log₂B', pages: 'pp. 260–261', sec: '11.3', secLabel: '11.3',
  },
  {
    get what() {
      return tr('Barriers in one block scan, Fig. 11.9 (warp scans)', 'Số rào chắn trong một lần scan khối, Fig. 11.9 (scan warp)');
    },
    formula: '2', pages: 'p. 265', sec: '11.4', secLabel: '11.4',
  },
  {
    get what() {
      return tr('Additions, Kogge-Stone on N elements', 'Số phép cộng, Kogge-Stone trên N phần tử');
    },
    formula: 'N · log₂N − (N − 1)', pages: 'p. 266', sec: '11.5', secLabel: '11.5', // i18n-ignore: formula
  },
  {
    get what() {
      return tr('Additions, coarsened, P threads', 'Số phép cộng, có gộp việc, P luồng');
    },
    get formula() {
      return tr('2N + P · log₂P − P − N/P, about 2N when P ≪ N', '2N + P · log₂P − P − N/P, khoảng 2N khi P ≪ N');
    },
    pages: 'pp. 269–270', sec: '11.6', secLabel: '11.6',
  },
  {
    get what() {
      return tr('Additions, Brent-Kung on N elements', 'Số phép cộng, Brent-Kung trên N phần tử');
    },
    formula: '2N − 2 − log₂N', pages: 'p. 284', sec: '11.10', secLabel: '11.10', // i18n-ignore: formula
  },
  {
    get what() {
      return tr('Global traffic, three kernels: scan, scan, add', 'Lưu lượng toàn cục, ba kernel: scan, scan, cộng');
    },
    get formula() {
      return tr('16 bytes per element', '16 byte mỗi phần tử');
    },
    pages: 'p. 274', sec: '11.9', secLabel: '11.9',
  },
  {
    get what() {
      return tr('Global traffic, three kernels: reduce, scan, scan', 'Lưu lượng toàn cục, ba kernel: rút gọn, scan, scan');
    },
    get formula() {
      return tr('12 bytes per element', '12 byte mỗi phần tử');
    },
    pages: 'pp. 275–276', sec: '11.9', secLabel: '11.9',
  },
  {
    get what() {
      return tr('Global traffic, one pass: load once, store once', 'Lưu lượng toàn cục, một lượt: nạp một lần, ghi một lần');
    },
    get formula() {
      return tr('8 bytes per element', '8 byte mỗi phần tử');
    },
    pages: 'p. 272', sec: '11.8', secLabel: '11.8, 11.9',
  },
];

export const questionsS110: Record<string, Question> = {
  'ch11/start/q-assoc': {
    kind: 'choice',
    id: 'ch11/start/q-assoc',
    prompt: (
      <Translated
        vi={
          <>
            Một cây rút gọn nhóm lại các phép cộng của một vòng lặp. Nó cho ra đúng kết quả của vòng lặp với mọi đầu vào
            khi toán tử có tính:
          </>
        }
      >
        A reduction tree regroups the additions of a loop. It gives the loop's answer for every input when the operator
        is:
      </Translated>
    ),
    options: [
      { label: <Translated vi="kết hợp">associative</Translated>, correct: true },
      {
        label: <Translated vi="giao hoán">commutative</Translated>,
        why: (
          <Translated vi="Tính giao hoán cho phép bạn đổi chỗ hai toán hạng. Một cây giữ nguyên thứ tự các toán hạng chỉ nhóm lại chúng, và việc nhóm lại là điều tính kết hợp cho phép.">
            Commutativity lets you swap the two operands. A tree that keeps the operands in order only regroups them, and regrouping is what associativity allows.
          </Translated>
        ),
      },
      {
        label: <Translated vi="có phần tử đơn vị">one that has an identity</Translated>,
        why: (
          <Translated vi="Phần tử đơn vị cho các luồng rảnh biết phải góp giá trị gì. Nó không nói gì về việc nhóm lại có làm đổi kết quả hay không.">
            An identity tells idle threads what to contribute. It says nothing about whether regrouping changes the result.
          </Translated>
        ),
      },
      {
        label: <Translated vi="bất kỳ tính chất nào ở trên">any of these</Translated>,
        why: (
          <Translated vi="Chỉ một trong ba tính chất nói về việc nhóm lại.">
            Only one of the three is about regrouping.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một cây thay đổi vị trí đặt dấu ngoặc, điều này an toàn đúng khi toán tử có tính kết hợp. Tính giao hoán chỉ
            cần khi cây còn đổi cả thứ tự các toán hạng, như kernel hội tụ của chương 10. Phép cộng số thực dấu phẩy động
            chỉ có tính kết hợp một cách xấp xỉ, nên một cây làm thay đổi vài bit cuối của tổng số thực. Scan thừa hưởng
            đúng yêu cầu này: mỗi đầu ra là phép rút gọn của một tiền tố (p. 255).
          </p>
        }
      >
        <p>
          A tree changes where the parentheses go, which is safe exactly when the operator is associative. Commutativity is
          needed only when the tree also changes the order of the operands, as the convergent kernel of chapter 10 does.
          Float addition is associative only approximately, so a tree changes the last bits of a float sum. Scan inherits
          exactly this requirement: each output is a reduction of a prefix (p. 255).
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.2, p. 224; ch. 11 p. 255; float addition: notes/today/ch10-facts.md A9',
    vi: { source: 'ch. 10 §10.2, p. 224; ch. 11 p. 255; phép cộng số thực: notes/today/ch10-facts.md A9' },
  },
  'ch11/start/q-shfl-edge': {
    kind: 'choice',
    id: 'ch11/start/q-shfl-edge',
    prompt: (
      <Translated
        vi={
          <>
            Trong một warp 32 lane, lane 30 gọi <code>__shfl_down_sync(0xffffffff, v, 4)</code>. Nó nhận lại gì?
          </>
        }
      >
        In a warp of 32 lanes, lane 30 calls <code>__shfl_down_sync(0xffffffff, v, 4)</code>. What does it get back?
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Giá trị của lane 34">Lane 34's value</Translated>,
        why: <Translated vi="Không có lane 34 trong một warp 32 lane.">There is no lane 34 in a 32-lane warp.</Translated>,
      },
      { label: <Translated vi="Chính v của nó">Its own v</Translated>, correct: true },
      {
        label: '0',
        why: (
          <Translated vi="Không có gì được điền vào. Số không sẽ là phần tử đơn vị của +, nhưng phép shuffle không biết bạn dùng toán tử nào.">
            Nothing is filled in. Zero would be the identity of +, but the shuffle does not know which operator you use.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Giá trị của lane 2 (nó quay vòng)">Lane 2's value (it wraps around)</Translated>,
        why: (
          <Translated vi="Đó là điều __shfl_sync làm với một lane nằm ngoài phạm vi, không phải phép shuffle xuống.">
            That is what __shfl_sync does with an out-of-range lane, not the down shuffle.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            NVIDIA ghi rõ rằng lane nguồn "will not wrap around" (sẽ không quay vòng), nên các lane phía trên "effectively
            remain unchanged" (thực chất giữ nguyên): lane 30 nhận lại giá trị của chính nó. Chương này dùng phiên bản
            đối xứng, <code>__shfl_up_sync</code>, với các lane phía dưới giữ giá trị của chính chúng. Đó là lý do vòng
            lặp của nó cần phép kiểm tra <code>laneIdx() &gt;= stride</code>: không có nó, những lane đó sẽ cộng giá trị
            của mình vào chính nó.
          </p>
        }
      >
        <p>
          NVIDIA documents that the source lane "will not wrap around", so the upper lanes "effectively remain unchanged":
          lane 30 receives its own value. This chapter uses the mirror image, <code>__shfl_up_sync</code>, whose lower
          lanes keep their own value. That is why its loop needs the test <code>laneIdx() &gt;= stride</code>: without it
          those lanes would add their own value to itself.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.7; notes/today/infra-facts.md A3; ch11-facts A3', // i18n-ignore: citation
  },
  'ch11/start/q-false-dep': {
    kind: 'choice',
    id: 'ch11/start/q-false-dep',
    prompt: (
      <Translated
        vi={
          <>
            Một vòng lặp đọc ô của láng giềng, chờ ở một rào chắn, ghi vào ô của chính nó, rồi lại chờ. Bạn cho nó hai buffer đổi vai trò mỗi lần lặp. Rào chắn nào có thể bỏ đi?
          </>
        }
      >
        A loop reads a neighbour's slot, waits at a barrier, writes its own slot, and waits again. You give it two
        buffers that swap roles every iteration. Which barrier can go?
      </Translated>
    ),
    options: [
      { label: <Translated vi="Rào chắn sau lần đọc">The one after the read</Translated>, correct: true },
      {
        label: <Translated vi="Rào chắn sau lần ghi">The one after the write</Translated>,
        why: (
          <Translated vi="Rào chắn đó bắt người đọc kế tiếp chờ giá trị nó cần. Hai buffer không thay đổi điều đó.">
            That one makes the next reader wait for the value it needs. Two buffers do not change that.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Cả hai">Both</Translated>,
        why: (
          <Translated vi="Một trong hai bảo vệ dữ liệu phải đến trước khi bất kỳ ai đọc nó.">
            One of them guards data that has to arrive before anyone reads it.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Không cái nào">Neither</Translated>,
        why: (
          <Translated vi="Một trong hai tồn tại chỉ vì cùng một ô được dùng lại.">
            One of them exists only because the same slot is reused.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Rào chắn sau lần đọc ngăn một luồng ghi phá hủy giá trị mà ai đó vẫn còn phải đọc: một phụ thuộc
            ghi-sau-đọc (write-after-read), hay phụ thuộc giả. Nó tồn tại chỉ vì một ô chứa cả giá trị cũ lẫn giá trị mới.
            Với hai buffer, giá trị mới đi sang chỗ khác và rào chắn đó không còn gì để bảo vệ. Rào chắn sau lần ghi bảo
            vệ một phụ thuộc thật (đọc-sau-ghi, read-after-write) và vẫn ở lại. Mục 11.3 làm đúng điều này với kernel scan.
          </p>
        }
      >
        <p>
          The barrier after the read stops a writer from destroying a value someone still has to read: a write-after-read,
          or false, dependence. It exists only because one slot holds both the old and the new value. With two buffers the
          new value goes elsewhere and that barrier has nothing to protect. The barrier after the write guards a true
          (read-after-write) dependence and stays. Section 11.3 does exactly this to the scan kernel.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.4, pp. 110–111; ch. 6 §6.7, pp. 146–147', // i18n-ignore: citation
  },
  'ch11/start/q-sectors': {
    kind: 'numeric',
    id: 'ch11/start/q-sectors',
    prompt: (
      <Translated
        vi={
          <>
            Mỗi lane trong 32 lane của một warp nạp số thực tại chỉ số <code>4*lane</code> của một mảng được căn chỉnh,
            giống như khi mỗi luồng tự nạp trực tiếp đoạn 4 phần tử của riêng nó. NVIDIA di chuyển bộ nhớ toàn cục theo
            các sector 32 byte. Lần nạp này chạm vào bao nhiêu sector?
          </>
        }
      >
        The 32 lanes of a warp each load the float at index <code>4*lane</code> of an aligned array, as they would if
        every thread loaded its own run of 4 elements directly. NVIDIA moves global memory in 32-byte sectors. How many
        sectors does this one load touch?
      </Translated>
    ),
    answer: 16,
    unit: 'sectors',
    explain: (
      <Translated
        vi={
          <p>
            Lane 31 đọc phần tử 124, các byte 496 đến 499, nên warp trải từ byte 0 đến 499: mười sáu sector 32 byte, trong
            đó nó chỉ dùng 4 byte trên mỗi 16 byte. Các chỉ số liên tiếp sẽ chỉ chạm 4 sector. Mục 11.6 gặp đúng điều này
            khi mỗi luồng sở hữu một đoạn phần tử, và nạp theo từng khúc để tránh nó (p. 269).
          </p>
        }
      >
        <p>
          Lane 31 reads element 124, bytes 496 to 499, so the warp spans bytes 0 to 499: sixteen 32-byte sectors, of which
          it uses 4 bytes in every 16. Consecutive indices would touch 4 sectors. Section 11.6 hits exactly this when each
          thread owns a run of elements, and loads chunk by chunk to avoid it (p. 269).
        </p>
      </Translated>
    ),
    source: "ch. 6 §6.1, pp. 124–132; the 32-byte sector is NVIDIA's (notes/today/ch06-facts.md A2); ch. 11 p. 269",
    vi: {
      unit: 'sector',
      source: 'ch. 6 §6.1, pp. 124–132; sector 32 byte là của NVIDIA (notes/today/ch06-facts.md A2); ch. 11 p. 269',
    },
  },
  'ch11/start/q-block-order': {
    kind: 'choice',
    id: 'ch11/start/q-block-order',
    prompt: (
      <Translated vi={<>Trong một lưới có nhiều khối, khối 7 quay vòng chờ cho đến khi khối 6 đặt một cờ trong bộ nhớ toàn cục. Như vậy có an toàn không?</>}>
        In a grid of many blocks, block 7 spins in a loop until block 6 sets a flag in global memory. Is that safe?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không: khối 6 có thể chưa được lập lịch, và nếu mọi khối thường trú đều quay chờ, nó sẽ không bao giờ được lập lịch.">
            No: block 6 may not be scheduled yet, and if every resident block spins, it never will be.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có: các khối bắt đầu theo thứ tự blockIdx.x, nên khối 6 đã chạy rồi khi khối 7 quay chờ.">
            Yes: blocks start in order of blockIdx.x, so block 6 is already running by the time block 7 spins.
          </Translated>
        ),
        why: (
          <Translated vi="NVIDIA nói bộ lập lịch không bảo đảm thứ tự nào cả.">
            NVIDIA says the scheduler makes no ordering guarantees.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Có, miễn là cờ được đọc bằng một phép toán nguyên tử (atomic) và được ghi với thứ tự release.">
            Yes, as long as the flag is read with an atomic and written with release order.
          </Translated>
        ),
        why: (
          <Translated vi="Một atomic và một thứ tự bộ nhớ làm cho lần đọc được định nghĩa rõ và dữ liệu nhìn thấy được; chúng không làm khối 6 chạy.">
            An atomic and an order make the read well defined and the data visible; they do not make block 6 run.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Chỉ khi cả hai khối ở trong cùng một warp.">Only if both blocks are in the same warp.</Translated>,
        why: <Translated vi="Các khối không bao giờ ở trong cùng một warp.">Blocks are never in the same warp.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Các khối có thể chạy theo bất kỳ thứ tự nào và không cần thường trú cùng lúc. Nếu thiết bị đầy những khối đều
            đang chờ các khối có số nhỏ hơn mà chưa bắt đầu, sẽ không có gì giải phóng một chỗ và lưới bị treo. Mục 11.9
            cần đúng kiểu chờ này cho scan một lượt của nó, và cố định thứ tự bằng cách phát chỉ số khối khi các khối thực
            sự bắt đầu.
          </p>
        }
      >
        <p>
          Blocks may run in any order and need not be resident together. If the device is full of blocks that all wait for
          lower-numbered blocks that have not started, nothing frees a slot and the grid hangs. Section 11.9 needs
          exactly this kind of waiting for its single-pass scan, and fixes the order by handing out block indices as blocks
          actually start.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.3, p. 73; notes/today/ch04-facts.md A5; ch. 11 p. 277', // i18n-ignore: citation
  },
  'ch11/start/q-relaxed': {
    kind: 'choice',
    id: 'ch11/start/q-relaxed',
    prompt: (
      <Translated
        vi={
          <>
            Vì sao <code>cuda::memory_order_relaxed</code> là đủ cho <code>fetch_add</code> của histogram ở chương 9?
          </>
        }
      >
        Why was <code>cuda::memory_order_relaxed</code> enough for the histogram's <code>fetch_add</code> in chapter 9?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không có truy cập nào khác của luồng cần được sắp thứ tự quanh nó; dù sao lần nạp điểm ảnh cũng đến trước.">
            No other access of the thread needs ordering around it; the pixel load comes first anyway.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Relaxed là thứ tự nhanh nhất, và về tính đúng đắn thì chọn thứ tự nào cũng không bao giờ quan trọng.">
            Relaxed is the fastest order, and for correctness it never matters which order you pick.
          </Translated>
        ),
        why: (
          <Translated vi="Các thứ tự khác nhau ở điều chúng hứa về những truy cập khác quanh phép atomic. Chương 11 cần một thứ tự mạnh hơn.">
            Orders differ in what they promise about the other accesses around the atomic. Chapter 11 needs a stronger one.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các bin của histogram nằm trong bộ nhớ chia sẻ, nơi thứ tự các truy cập quanh một atomic không quan trọng.">
            Histogram bins live in shared memory, where the order of the accesses around an atomic does not matter.
          </Translated>
        ),
        why: (
          <Translated vi="Kernel cơ bản cập nhật các bin trong bộ nhớ toàn cục, và câu hỏi về thứ tự là về những truy cập khác, không phải về nơi bin nằm.">
            The basic kernel updates bins in global memory, and the order question is about other accesses, not where the bin lives.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các atomic trên GPU nhất quán tuần tự dù bạn truyền thứ tự nào cho chúng.">
            GPU atomics are sequentially consistent whatever order you pass to them.
          </Translated>
        ),
        why: (
          <Translated vi="Thứ tự là một tham số chính vì chúng không nhất thiết như vậy: relaxed yêu cầu ít hơn, và phần cứng có thể cho ít hơn.">
            The order is a parameter precisely because they need not be: relaxed asks for less, and the hardware may give less.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một thứ tự bộ nhớ ràng buộc cách một atomic có thể bị sắp xếp lại so với các truy cập khác của cùng luồng.
            Luồng histogram nạp một điểm ảnh, và giá trị đó chọn bin, nên dù sao lần nạp cũng phải đến trước. Ở mục 11.9,
            một khối công bố một tổng rồi đặt một cờ, và một khối khác đọc cờ rồi mới đọc tổng. Ở đó thứ tự là quan trọng,
            và sách chuyển sang release và acquire.
          </p>
        }
      >
        <p>
          A memory order constrains how an atomic may be reordered with the same thread's other accesses. The histogram
          thread loads a pixel, and that value chooses the bin, so the load must come first anyway. In section 11.9 a block
          publishes a sum and then sets a flag, and another block reads the flag and then the sum. There the order matters,
          and the book switches to release and acquire.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.2, pp. 208–209; ch. 11 pp. 278–279', // i18n-ignore: citation
  },
};

export function S110() {
  const q = questionsS110;
  return (
    <>
      <Translated
        vi={
          <p>
            Scan là một tổng chạy (running total). Đầu ra <i>i</i> là tổng các đầu vào tính đến <i>i</i>, nên mỗi đầu ra
            dường như cần đầu ra đứng trước nó: tuần tự hết mức mà một phép tính có thể trông như vậy. Chương này cho thấy
            nhiều luồng vẫn có thể tính nó như thế nào, và cái giá phải trả là gì. Quy tắc kinh nghiệm của sách khá rộng:
            khi mỗi phần tử của một phép tính được định nghĩa từ phần tử đứng trước nó, phép tính đó thường có thể viết
            thành một scan và chạy song song.
          </p>
        }
      >
        <p>
          A scan is a running total. Output <i>i</i> is the sum of the inputs up to <i>i</i>, so every output seems to
          need the one before it: about as sequential as a computation can look. The chapter shows how many threads can
          compute it anyway, and what that costs. The book's rule of thumb is broad: when each item of a computation is
          defined from the item before it, the computation can often be written as a scan and run in parallel.
        </p>
      </Translated>
      <Source pages="251" />
      <Translated
        vi={
          <p>
            Scan cũng là một khối xây dựng. Lọc (chương 12) và radix sort (chương 14) được xây trên nó, và sách liệt kê
            quick sort, so sánh chuỗi, tính giá trị đa thức, công thức truy hồi và các phép toán trên cây là những nơi dùng
            nó. Và đây là một bài học về một sự đánh đổi mà bạn chưa gặp ở dạng này. Các scan song song nhanh làm tổng cộng
            nhiều phép cộng hơn vòng lặp thông thường; trên phần cứng có ít đơn vị thực thi, phần việc thừa đó có thể khiến
            chúng chậm hơn. Chương này luôn để mắt đến cả hai con số: tổng khối lượng công việc và số bước (span).
          </p>
        }
      >
        <p>
          Scan is also a building block. Filtering (chapter 12) and radix sort (chapter 14) are built on it, and the book
          lists quick sort, string comparison, polynomial evaluation, recurrences and tree operations as further users.
          And it is a study in a trade-off you have not met in this form yet. The fast parallel scans do more additions in
          total than the plain loop; on hardware with few execution units, that extra work can make them slower. The
          chapter keeps both numbers in view: the total work and the number of steps (the span).
        </p>
      </Translated>
      <Source pages="251–252" />

      <h2>
        <Translated vi="Vì sao bạn sẽ quan tâm">Why you will care</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lấy mẫu top-p sắp xếp xác suất của token kế tiếp và giữ tiền tố ngắn nhất có xác suất tích lũy đạt tới p. Xác
            suất tích lũy đó là một scan, và việc token vượt qua p có được giữ hay không chính là khác biệt giữa hai loại
            scan được định nghĩa ở mục 11.1. Lab 11.6 xây mặt nạ giữ lại đó; nó là phần top-p của mốc dự án M5 của dự án
            khóa học, vốn bắt đầu từ phần đếm token ở chương 9. Scan cũng biến kích thước thành vị trí bắt đầu (offset): số
            token của từng chuỗi trong một batch trở thành vị trí bắt đầu của mỗi chuỗi trong một buffer đóng gói, và ở
            chương 12 một scan trên các cờ giữ lại 0/1 cho mỗi token được giữ một chỗ trong danh sách đã nén gọn.
          </p>
        }
      >
        <p>
          Top-p sampling sorts the next token's probabilities and keeps the shortest prefix whose cumulative probability
          reaches p. That cumulative probability is a scan, and whether the token that crosses p is kept is the difference
          between the two kinds of scan defined in section 11.1. Lab 11.6 builds that keep mask; it is the top-p piece of
          milestone M5 of the course project, which chapter 9's token counts began. Scans also turn sizes into offsets: the per-sequence token counts of a
          batch become the positions where each sequence starts in one packed buffer, and in chapter 12 a scan of 0/1 keep
          flags gives every kept token its slot in a compacted list.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không bao giờ nối scan với lấy mẫu hay batching. Những cách dùng này là của khóa học này (ghi chú,
              "Relevance to LLM-inference kernels").
            </p>
          }
        >
          <p>
            The book never connects scan to sampling or batching. These uses are ours (notes, "Relevance to LLM-inference
            kernels").
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Điều gì đang chạy bên dưới torch.cumsum ngày nay">What runs under torch.cumsum today</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Gọi <code>torch.cumsum(probs, dim=-1)</code> trên một batch các hàng trên thiết bị CUDA, PyTorch sẽ scan mỗi
            hàng dọc theo chiều cuối cùng bằng một kernel riêng: vài hàng mỗi khối, một scan bộ nhớ chia sẻ (shared memory)
            cho mỗi khúc, và một tổng chạy được mang từ khúc này sang khúc kế tiếp. Một tensor một chiều thì đi sang thư
            viện CUB của NVIDIA, trừ khi bạn đã yêu cầu thuật toán tất định. Trên GPU của Apple (MPS), các hàng dài được
            chia cho nhiều threadgroup và scan trong ba lượt: rút gọn từng khối, scan các tổng khối, rồi cộng chúng trở
            lại. Đến cuối chương bạn sẽ nhận ra từng mảnh trong số này.
          </p>
        }
      >
        <p>
          Call <code>torch.cumsum(probs, dim=-1)</code> on a batch of rows on a CUDA device and PyTorch scans each row along
          its last dimension with a kernel of its own: several rows per block, a shared-memory scan per chunk, and a
          running total carried from chunk to chunk. A one-dimensional tensor goes to NVIDIA's CUB library instead, unless
          you have asked for deterministic algorithms. On Apple GPUs (MPS), long rows are split across threadgroups and
          scanned in three passes: reduce each block, scan the block sums, add them back. You will recognise every one of
          these pieces by the end of the chapter.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch11-facts D4)', '2026-11-20 (ch11-facts, mục D4)')}>
        <Translated
          vi={
            <p>
              CUDA: <code>scan_innermost_dim</code> cho chiều cuối cùng, "Parallel reduction with Sklansky method" bên
              trong một khối; tensor 1-D đi qua CUB, hoặc một scan tất định gồm hai kernel khi bật{' '}
              <code>use_deterministic_algorithms</code>. MPS: "Three-pass scan (block_reduce -&gt; scan_block_sums -&gt;
              block_carry)".
            </p>
          }
        >
          <p>
            CUDA: <code>scan_innermost_dim</code> for the last dimension, "Parallel reduction with Sklansky method" inside
            a block; 1-D tensors through CUB, or a deterministic two-kernel scan under{' '}
            <code>use_deterministic_algorithms</code>. MPS: "Three-pass scan (block_reduce -&gt; scan_block_sums -&gt;
            block_carry)".
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Chương này như một sổ cái chi phí">The chapter as a cost ledger</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách xây một kernel rồi lần lượt loại bỏ từng chi phí của nó: rào chắn, phép cộng thừa, rồi các chuyến đi tới
            bộ nhớ toàn cục. Đây là những con số sách sẽ đưa ra, để bạn nhìn từng con số giảm xuống. B là số phần tử một
            khối scan, N là độ dài của toàn bộ đầu vào, P là số luồng.
          </p>
        }
      >
        <p>
          The book builds one kernel and then removes its costs one at a time: barriers, extra additions, then trips to
          global memory. Here are the numbers it will give, so you can watch each one fall. B is the number of elements one
          block scans, N the length of the whole input, P the number of threads.
        </p>
      </Translated>
      <table className="data wide" data-part="ledger" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr>
            <th>
              <Translated vi="Chi phí gì">What it costs</Translated>
            </th>
            <th>
              <Translated vi="Công thức">Formula</Translated>
            </th>
            <th>
              <Translated vi="Sách">Book</Translated>
            </th>
            <th>
              <Translated vi="Mục">Section</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          {LEDGER.map((r) => (
            <tr key={r.what}>
              <td>{r.what}</td>
              <td className="nowrap">{r.formula}</td>
              <td className="nowrap">{r.pages}</td>
              <td>
                <a href={href('ch11', r.sec)}>{r.secLabel}</a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Có hai hàng đáng nhìn qua ngay bây giờ. Kogge-Stone, phép scan song song mà chương này bắt đầu, làm khoảng
            log₂N lần số phép cộng của vòng lặp: với 512 phần tử là nhiều gấp 8 đến 9 lần (p. 266). Gộp việc cho luồng
            (thread coarsening) đưa con số đó về khoảng hai phép cộng mỗi phần tử. Và các hàng lưu lượng cho thấy scan toàn
            cục tốt nhất di chuyển 8 byte mỗi phần tử, bằng một nửa so với thiết kế ba kernel hiển nhiên.
          </p>
        }
      >
        <p>
          Two rows deserve a first look now. Kogge-Stone, the parallel scan this chapter starts from, does about log₂N
          times the additions of the loop: for 512 elements between 8 and 9 times as many (p. 266). Coarsening brings that
          back to about two additions per element. And the traffic rows say that the best global scan moves 8 bytes per
          element, half of what the obvious three-kernel design moves.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Số rào chắn đến từ chính phép tính của các vòng lặp: Fig. 11.3 chạy log₂B lần lặp, mỗi lần hai rào chắn, Fig.
              11.5 mỗi lần một (pp. 257, 261).
            </p>
          }
        >
          <p>
            The barrier counts are the loops' own arithmetic: Fig. 11.3 runs log₂B iterations with two barriers each, Fig.
            11.5 one each (pp. 257, 261).
          </p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Trước khi bắt đầu">Before you start</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sáu câu hỏi về những gì chương này giả định từ chương 4 đến chương 10. Trong chương 10, cây rút gọn (
            <a href={href('ch10', '10.2')}>10.2</a>), shuffle warp (<a href={href('ch10', '10.7')}>10.7</a>,{' '}
            <a href={href('ch10', '10.8')}>10.8</a>) và gộp việc cho luồng (<a href={href('ch10', '10.10')}>10.10</a>) là
            những trang chương này dựa vào nhiều nhất. Mỗi câu hỏi sẽ quay lại trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Six questions on what this chapter assumes from chapters 4 to 10. Of chapter 10, the reduction trees (
          <a href={href('ch10', '10.2')}>10.2</a>), warp shuffles (<a href={href('ch10', '10.7')}>10.7</a>,{' '}
          <a href={href('ch10', '10.8')}>10.8</a>) and coarsening (<a href={href('ch10', '10.10')}>10.10</a>) are the pages
          this chapter leans on most. Each question comes back in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={q['ch11/start/q-assoc']} label={tr('From chapter 10', 'Từ chương 10')} />
      <Ask q={q['ch11/start/q-shfl-edge']} label={tr('From chapter 10', 'Từ chương 10')} />
      <Ask q={q['ch11/start/q-false-dep']} label={tr('From chapters 5 and 6', 'Từ chương 5 và 6')} />
      <Ask q={q['ch11/start/q-sectors']} label={tr('From chapter 6', 'Từ chương 6')} />
      <Ask q={q['ch11/start/q-block-order']} label={tr('From chapter 4', 'Từ chương 4')} />
      <Ask q={q['ch11/start/q-relaxed']} label={tr('From chapter 9', 'Từ chương 9')} />

      <h2>
        <Translated vi="Các lab">The labs</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sáu lab đi theo các mục, tất cả nằm trên <a href={href('ch11', 'labs')}>trang lab</a>. Mỗi lab yêu cầu một dự
            đoán trước khi bạn đo, và mỗi lab kiểm tra với kết quả tham chiếu trên CPU tính bằng vòng lặp thông thường từ
            trái sang phải.
          </p>
        }
      >
        <p>
          Six labs follow the sections, all on the <a href={href('ch11', 'labs')}>labs page</a>. Each asks for a prediction
          before you measure, and each tests against a CPU reference computed with the plain left-to-right loop.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>Lab 11.1, một lần scan khối, ba cách.</b> Kernel hai rào chắn của mục 11.2, kernel buffer kép của mục 11.3
            và kernel scan warp của mục 11.4, trên một khối, được đo so với nhau.
          </p>
        }
      >
        <p>
          <b>Lab 11.1, one block scan, three ways.</b> The two-barrier kernel of section 11.2, the double-buffered kernel
          of section 11.3 and the warp-scan kernel of section 11.4, on one block, measured against each other.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>Lab 11.2, gộp việc cho luồng và phân ô thanh ghi.</b> Mỗi luồng trước hết scan đoạn phần tử của riêng nó,
            tuần tự và trong thanh ghi (mục 11.6 và 11.7), để phép scan cấp khối có ít giá trị hơn nhiều cần kết hợp.
          </p>
        }
      >
        <p>
          <b>Lab 11.2, coarsening and register tiling.</b> Each thread first scans its own run of elements, sequentially
          and in registers (sections 11.6 and 11.7), so that the block-level scan has far fewer values to combine.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>Lab 11.3, săn race trong scan.</b> Một scan theo hàng có dạng như llama.cpp, MLX và PyTorch phát hành, với
            ba lỗi cần tìm và sửa. Các trình giả lập (emulator) cho thấy một số lỗi mà GPU có thể che giấu; kernel đã sửa là
            phần lõi của lab 11.6.
          </p>
        }
      >
        <p>
          <b>Lab 11.3, scan race hunt.</b> A row-wise scan in the shape llama.cpp, MLX and PyTorch ship, with three
          defects to find and fix. The emulators show some that a GPU can hide; the fixed kernel is the core of lab 11.6.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>Lab 11.4, scan với độ dài bất kỳ.</b> Các lần scan khối được ghép thành một scan của toàn bộ đầu vào bằng các
            kernel riêng biệt (mục 11.9), thiết kế chạy được trên mọi backend.
          </p>
        }
      >
        <p>
          <b>Lab 11.4, a scan of any length.</b> Block scans stitched into a scan of the whole input with separate
          kernels (section 11.9), the design that works on every backend.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>Lab 11.5, scan một lượt với lookback.</b> Không bắt buộc, và chỉ dành cho CUDA và HIP: các khối chờ nhau
            qua các cờ trong bộ nhớ toàn cục (mục 11.9). Metal không có tài liệu nào bảo đảm điều này an toàn.
          </p>
        }
      >
        <p>
          <b>Lab 11.5, single-pass scan with lookback.</b> Optional, and for CUDA and HIP only: blocks wait for each other
          through flags in global memory (section 11.9). Metal documents no guarantee that makes this safe.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>Lab 11.6, mặt nạ giữ lại top-p.</b> Phần top-p của mốc dự án M5: với mỗi hàng xác suất đã sắp xếp, một scan
            và một ngưỡng giữ lại các token có tổng khối lượng xác suất đứng trước chúng nhỏ hơn p. Nó nêu rõ quy tắc xử lý
            giá trị bằng nhau và dải làm tròn quanh p mà trong đó câu trả lời nào cũng được chấp nhận.
          </p>
        }
      >
        <p>
          <b>Lab 11.6, top-p keep mask.</b> The top-p piece of milestone M5: for each row of sorted probabilities, a scan and a
          threshold that keep the tokens whose mass before them is below p. It states its tie rule and the rounding band
          around p inside which either answer is accepted.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch11-facts C5, B3)', '2026-11-20 (ch11-facts, mục C5, B3)')}>
        <Translated
          vi={
            <p>
              Đặc tả của Metal không bảo đảm tiến triển (forward progress) giữa các threadgroup, và một kỹ sư Apple đã gọi
              một scan nối chuỗi có quay chờ là "undefined behavior". AMD cũng không có tài liệu nào bảo đảm; phiên bản HIP
              vẫn là unverified cho đến khi nó chạy trên phần cứng AMD.
            </p>
          }
        >
          <p>
            Metal's spec gives no forward-progress guarantee between threadgroups, and an Apple engineer called a spinning
            chained scan "undefined behavior". AMD documents none either; the HIP version stays unverified until it runs on
            AMD hardware.
          </p>
        </Translated>
      </Today>
    </>
  );
}
