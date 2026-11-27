import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { SEGMENT_MODELS, warpAccess } from '../../sim/coalesce';

// The start page of chapter 21: what the chapter adds (four decisions about one all-pairs
// sum), the domain in one paragraph, one bridge to PyTorch with a prediction, a map, and a
// check on the five ideas from chapters 4, 6, 7, 9 and appendix A that the chapter uses.
// The questions come in the order the chapter uses them. The sector count comes from the
// tested coalescing model (src/sim/coalesce.ts); the matrix size is plain arithmetic.

// The sector question is about NVIDIA only, where a warp is 32 threads
// (notes/today/ch05-facts.md A5) and the unit is a 32-byte sector (ch06-facts A2).
const NVIDIA_WARP = 32;
const SECTOR = SEGMENT_MODELS.nvidia.segmentBytes ?? 32;
const F32 = 4;
const lanes = Array.from({ length: NVIDIA_WARP }, (_, t) => t);
/** Lane t stores element base + 4t: the stride of one write statement in Fig. 21.8 as the text describes it. */
const SECTORS_STRIDE4 = warpAccess(lanes.map((t) => 4 * t * F32), F32, SECTOR).transactions; // 16
/** Lane t stores element base + t: Fig. 21.10. */
const SECTORS_STRIDE1 = warpAccess(lanes.map((t) => t * F32), F32, SECTOR).transactions; // 4

// The prediction: the scale is the book's (notes/ch22.md, p. 536: hundreds of thousands of
// atoms, millions of grid points); the round numbers are ours.
const POINTS = 1_000_000;
const ATOMS = 100_000;
const BYTES = POINTS * ATOMS * F32;
const MATRIX_GB = BYTES / 1e9; // 400
const MATRIX_GIB = BYTES / 2 ** 30; // 372.5
const RESULT_MB = (POINTS * F32) / 1e6; // 4
const fmt = (v: number) => (Math.round(v * 10) / 10).toLocaleString('en-US');

export const questionsS210: Record<string, Question> = {
  'ch21/start/q-float-order': {
    kind: 'choice',
    id: 'ch21/start/q-float-order',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel cộng 10,000 đóng góp kiểu float vào một vị trí đầu ra bằng <code>atomicAdd</code>. Bạn chạy nó
            hai lần. Cả hai kết quả đều khớp với một bản tham chiếu CPU độ chính xác kép trong phạm vi dung sai của bài
            kiểm tra, nhưng chúng khác nhau ở vài chữ số cuối. Đó có phải là lỗi không?
          </>
        }
      >
        A kernel adds 10,000 float contributions into one output location with <code>atomicAdd</code>. You run it
        twice. Both results agree with a double-precision CPU reference to within your test's tolerance, but they differ
        from each other in the last few digits. Is that a bug?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không. Các phép toán nguyên tử trên cùng một địa chỉ không đến theo thứ tự cố định nào, và phép cộng float không có tính kết hợp, nên mỗi lần chạy làm tròn khác nhau.">
            No. Same-address atomics land in no fixed order and float addition is not associative, so each run rounds
            differently.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Có. Một số cập nhật bị mất: khi hai lane của một warp chạm cùng một địa chỉ trong cùng một lệnh, phần cứng chỉ giữ một giá trị.">Yes. Some updates are being lost: when two lanes of a warp hit one address in the same instruction, the hardware keeps only one value.</Translated>,
        why: <Translated vi="Không làm mất cập nhật chính là điều một phép toán nguyên tử bảo đảm. Một cập nhật bị mất làm rơi cả một đóng góp, và khi đó tổng sẽ lệch khỏi bản tham chiếu, chứ không chỉ khác lần chạy kia.">Not losing updates is exactly what an atomic guarantees. A lost update drops a whole contribution, and the sum would then miss the reference, not just the other run.</Translated>,
      },
      {
        label: <Translated vi="Có. Một kernel đúng chạy hai lần trên cùng đầu vào cho ra cùng các bit, nên mọi khác biệt đều nghĩa là có race condition ở đâu đó.">Yes. A correct kernel run twice on the same inputs gives the same bits, so any difference means a race somewhere.</Translated>,
        why: <Translated vi="Cùng các bit là tính tái lập, không phải tính đúng. Cả hai lần chạy đều khớp bản tham chiếu; những phép cộng có làm tròn thông thường, thực hiện theo một thứ tự khác, đã đủ cho ra các chữ số cuối khác nhau.">Same bits is reproducibility, not correctness. Both runs match the reference; ordinary rounded additions, done in a different order, already give different last digits.</Translated>,
      },
      {
        label: <Translated vi="Không, và điều đó không thể xảy ra: GPU áp dụng các phép toán nguyên tử theo thứ tự chỉ số luồng, nên khác biệt phải đến từ host.">No, and it cannot happen: the GPU applies atomics in thread-index order, so the difference must come from the host.</Translated>,
        why: <Translated vi="NVIDIA nói thứ tự các phép toán nguyên tử trên một địa chỉ là không xác định, và sách nói phép toán nguyên tử không áp đặt thứ tự nào giữa các luồng.">NVIDIA says the order of atomics on one address is undefined, and the book says atomics enforce no order between threads.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sách nói phép toán nguyên tử không áp đặt thứ tự nào giữa các luồng; mỗi phép chỉ chờ phép trước nó xong.
            NVIDIA nói thêm rằng khi nhiều luồng của một warp cập nhật một vị trí, các cập nhật được tuần tự hóa "nhưng
            thứ tự chúng xảy ra là không xác định". Phụ lục A cho thấy thứ tự các phép cộng float làm thay đổi tổng sau
            làm tròn. Gộp lại: một tổng xây từ các phép toán nguyên tử float đúng đến mức làm tròn nhưng không tái lập
            được từng bit, còn một luồng cộng các số hạng của nó theo thứ tự cố định thì lần nào cũng ra cùng đáp án.
            Trang <a href={href('ch21', '21.2')}>21.2</a> gặp cả hai: kernel scatter cộng vào lưới bằng{' '}
            <code>atomicAdd</code>, kernel gather tính tổng trong một thanh ghi, và PyTorch cũng vạch đúng ranh giới đó
            khi bạn yêu cầu các thuật toán tất định. Được dạy ở <a href={href('ch09', '9.2')}>ch. 9 §9.2</a> và phụ lục
            A §A.6.
          </p>
        }
      >
        <p>
          The book says atomics enforce no order between threads; each one only waits for the one before it to finish.
          NVIDIA adds that when several threads of a warp update one location the updates are serialised "but the order
          in which they occur is undefined". Appendix A shows that the order of float additions changes the rounded sum.
          Together: a sum built from float atomics is correct to rounding but not bit-for-bit reproducible, while a thread
          that adds its terms in a fixed order gets the same answer every time. Page{' '}
          <a href={href('ch21', '21.2')}>21.2</a> meets both: the scatter kernel adds into the grid with{' '}
          <code>atomicAdd</code>, the gather kernel sums in a register, and PyTorch draws the same line when you ask it
          for deterministic algorithms. Taught in <a href={href('ch09', '9.2')}>ch. 9 §9.2</a> and appendix A §A.6.
        </p>
      </Translated>
    ),
    source:
      'ch. 9 §9.2, pp. 203–209 (no order: p. 206); appendix A §A.6, pp. 593–594; notes/today/infra-facts.md A1; ch10-facts A9',
    vi: {
      source:
        'ch. 9 §9.2, pp. 203–209 (không có thứ tự: p. 206); phụ lục A §A.6, pp. 593–594; notes/today/infra-facts.md A1; ch10-facts A9',
    },
  },
  'ch21/start/q-const-lanes': {
    kind: 'choice',
    id: 'ch21/start/q-const-lanes',
    prompt: (
      <Translated
        vi={
          <>
            Một bảng nằm trong bộ nhớ <code>__constant__</code>. Trong kernel A, mọi lane của một warp đọc{' '}
            <code>table[n]</code> với cùng một <code>n</code>. Trong kernel B, lane <code>t</code> đọc{' '}
            <code>table[t]</code>. NVIDIA ghi gì trong tài liệu về hai lần đọc này?
          </>
        }
      >
        A table sits in <code>__constant__</code> memory. In kernel A every lane of a warp reads <code>table[n]</code>{' '}
        with the same <code>n</code>. In kernel B lane <code>t</code> reads <code>table[t]</code>. What does NVIDIA
        document about the two reads?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="A: một lần truy cập cho cả warp, có thể nhanh như một thanh ghi. B: một lần truy cập cho mỗi địa chỉ khác nhau, tức mỗi lane một lần.">
            A: one access for the whole warp, which can be as fast as a register. B: one access per distinct address,
            so one per lane.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Cả hai đều được phát rộng tới warp trong một lần truy cập, vì dữ liệu hằng là chỉ đọc và không cần lưu lượng đảm bảo nhất quán.">Both are broadcast to the warp in one access, because constant data is read-only and needs no coherence traffic.</Translated>,
        why: <Translated vi="Việc chỉ đọc là điều cho phép bộ nhớ đệm nhỏ và rẻ. Chi phí vẫn là một lần truy cập cho mỗi địa chỉ khác nhau trong warp.">Being read-only is what lets the cache be small and cheap. The cost is still one access per distinct address in the warp.</Translated>,
      },
      {
        label: <Translated vi="B nhanh hơn: giống bộ nhớ chia sẻ (shared memory), bộ nhớ đệm hằng có các bank, và 32 địa chỉ khác nhau trải ra trên chúng mà không xung đột.">B is faster: like shared memory, the constant cache has banks, and 32 distinct addresses spread over them with no conflict.</Translated>,
        why: <Translated vi="Đó là quy tắc của bộ nhớ chia sẻ. Bộ nhớ đệm hằng phục vụ các địa chỉ khác nhau lần lượt từng cái.">That is the rule for shared memory. The constant cache serves distinct addresses one after another.</Translated>,
      },
      {
        label: <Translated vi="A và B: mỗi cái một lần truy cập bộ nhớ đệm cho mỗi warp khi bảng đã nằm trong bộ nhớ đệm; chỉ một lần trượt mới làm B đắt hơn.">A and B: one cache access per warp each once the table is in the cache; only a miss makes B dearer.</Translated>,
        why: <Translated vi="Ngay cả khi mọi lần đọc đều trúng, các địa chỉ khác nhau trong một warp vẫn được phục vụ lần lượt từng cái.">Even when every read hits, distinct addresses in one warp are served one at a time.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            NVIDIA: "Các truy cập tới những địa chỉ khác nhau bởi các luồng trong một warp được tuần tự hóa, do đó chi phí
            tăng tuyến tính theo số địa chỉ khác nhau", và khi mọi luồng của một warp đọc một vị trí, bộ nhớ hằng "có
            thể nhanh như một truy cập thanh ghi". AMD ghi nhận cùng sự phân chia đó cho bộ nhớ đệm vô hướng (scalar
            cache) của mình, vốn chỉ phục vụ các lần đọc đồng nhất trên toàn wavefront; Apple không ghi nhận bộ nhớ đệm
            hằng nào cả. Ở trang <a href={href('ch21', '21.2')}>21.2</a>, các nguyên tử nằm trong bộ nhớ hằng. Trong
            kernel gather, mọi lane của một warp đọc cùng một nguyên tử cùng lúc: trường hợp A. Trong kernel scatter,
            mỗi luồng đọc nguyên tử của riêng nó: trường hợp B. Được dạy ở <a href={href('ch07', '7.4')}>ch. 7 §7.4</a>.
          </p>
        }
      >
        <p>
          NVIDIA: "Accesses to different addresses by threads within a warp are serialized, thus the cost scales linearly
          with the number of unique addresses", and when every thread of a warp reads one location, constant memory "can
          be as fast as a register access". AMD documents the same split for its scalar cache, which serves only reads
          that are uniform across the wavefront; Apple documents no constant cache at all. On page{' '}
          <a href={href('ch21', '21.2')}>21.2</a> the atoms sit in constant memory. In the gather kernel every lane of a
          warp reads the same atom at the same time: case A. In the scatter kernel each thread reads its own atom: case B.
          Taught in <a href={href('ch07', '7.4')}>ch. 7 §7.4</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 7 §7.4, pp. 168–172; notes/today/ch07-facts.md A3 (NVIDIA), C2 (Apple); ch21-facts B1 (AMD)', // i18n-ignore: a citation
  },
  'ch21/start/q-coarsen-cost': {
    kind: 'choice',
    id: 'ch21/start/q-coarsen-cost',
    prompt: (
      <Translated
        vi={
          <>
            Mỗi luồng của một kernel tính một đầu ra bằng cách lặp qua một danh sách chỉ đọc mà mọi luồng đều đọc. Bạn
            gộp việc cho luồng (thread coarsening): giờ mỗi luồng tính bốn đầu ra, nạp mỗi phần tử danh sách một lần và
            dùng nó bốn lần, và bạn khởi chạy một phần tư số luồng. Bài toán lấp đầy GPU nhiều lần. Điều gì vẫn có thể
            làm kernel đã gộp việc chậm hơn?
          </>
        }
      >
        Each thread of a kernel computes one output by looping over a read-only list that every thread reads. You
        coarsen it: each thread now computes four outputs, loads each list item once and uses it four times, and you
        launch a quarter of the threads. The problem fills the GPU many times over. What can still make the coarsened
        kernel slower?
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Mỗi luồng giữ nhiều giá trị sống hơn trong thanh ghi, nên ít warp hơn vừa trên một SM để che độ trễ bộ nhớ.">Each thread keeps more values live in registers, so fewer warps fit on an SM to hide memory latency.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Giờ mỗi đầu ra cần nhiều phép tính hơn trước, vì luồng còn phải tính độ lệch của bốn đầu ra của nó.">Each output now needs more arithmetic than before, because the thread must also compute the offsets of its four outputs.</Translated>,
        why: <Translated vi="Gộp việc loại bỏ công việc lặp lại. Tính trên mỗi đầu ra, luồng làm cùng lượng việc hoặc ít hơn.">Coarsening removes repeated work. Per output, the thread does the same work or less.</Translated>,
      },
      {
        label: <Translated vi="Bốn đầu ra của một luồng cần một rào chắn giữa chúng, nên luồng phải chờ warp của nó ở mỗi đầu ra.">The four outputs of a thread need a barrier between them, so the thread waits for its warp at every output.</Translated>,
        why: <Translated vi="Một luồng tính cả bốn theo tuần tự. Không có luồng nào khác để chờ.">One thread computes all four in sequence. There is no other thread to wait for.</Translated>,
      },
      {
        label: <Translated vi="Còn lại quá ít khối để giữ mọi SM bận, vì một phần tư số luồng nghĩa là một phần tư số khối.">Too few blocks are left to keep every SM busy, since a quarter of the threads means a quarter of the blocks.</Translated>,
        why: <Translated vi="Đó là cái bẫy thứ hai của sách, gộp việc quá nhiều, và nó gây hại khi bài toán nhỏ. Bài toán này lấp đầy thiết bị nhiều lần.">That is the book's second pitfall, too much coarsening, and it bites when the problem is small. This one fills the device many times over.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cái bẫy thứ ba của sách: gộp việc có thể làm tăng số thanh ghi mỗi luồng cần, và nếu điều đó làm giảm
            occupancy (mức chiếm dụng) thì phần mất có thể lớn hơn phần được. Tệp thanh ghi của một SM là cố định, nên
            luồng nặng hơn nghĩa là ít warp thường trú hơn để luân phiên trong khi các warp khác chờ bộ nhớ. Hiệu ứng nào
            thắng là chuyện đo đạc, không phải quy tắc. Kernel đã gộp việc ở trang{' '}
            <a href={href('ch21', '21.2')}>21.2</a> giữ bốn tổng đang chạy, bốn khoảng cách và các số hạng mà bốn điểm
            dùng chung sống suốt vòng lặp nguyên tử. Sách nói số thanh ghi của nó nằm trong giới hạn nhưng không đưa con
            số; bạn kiểm tra bản build của mình bằng <code>--resource-usage</code> với <code>nvcc</code> hoặc{' '}
            <code>hipcc</code>, còn Metal không báo số thanh ghi. Được dạy ở{' '}
            <a href={href('ch06', '6.5')}>ch. 6 §6.5</a>.
          </p>
        }
      >
        <p>
          The book's third pitfall: coarsening can raise the registers each thread needs, and if that lowers occupancy the
          loss can outweigh the gain. The register file of an SM is fixed, so heavier threads mean fewer resident warps to
          switch between while others wait on memory. Which effect wins is a measurement, not a rule. The coarsened kernel
          on page <a href={href('ch21', '21.2')}>21.2</a> keeps four running sums, four distances and the terms the four
          points share live for the whole atom loop. The book says its register count stays within the limit but gives no
          count; you check your own build with <code>--resource-usage</code> under <code>nvcc</code> or{' '}
          <code>hipcc</code>, and Metal reports no register count. Taught in{' '}
          <a href={href('ch06', '6.5')}>ch. 6 §6.5</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.5, pp. 141–143 (pitfall 3: p. 143); ch. 21 §21.3, p. 521; notes/today/ch21-facts.md A7',
    vi: { source: 'ch. 6 §6.5, pp. 141–143 (cái bẫy 3: p. 143); ch. 21 §21.3, p. 521; notes/today/ch21-facts.md A7' },
  },
  'ch21/start/q-sectors': {
    kind: 'numeric',
    id: 'ch21/start/q-sectors',
    prompt: (
      <Translated
        vi={
          <>
            Trên GPU NVIDIA, một warp có 32 lane. Mỗi lane lưu một float: lane <code>t</code> ghi phần tử{' '}
            <code>base + 4*t</code>, và phần tử <code>base</code> bắt đầu tại một ranh giới 128 byte. Lần lưu của warp
            chạm tới bao nhiêu sector 32 byte?
          </>
        }
      >
        On an NVIDIA GPU a warp has 32 lanes. Each lane stores one float: lane <code>t</code> writes element{' '}
        <code>base + 4*t</code>, and element <code>base</code> starts on a 128-byte boundary. How many 32-byte sectors
        does the warp's store touch?
      </Translated>
    ),
    answer: SECTORS_STRIDE4,
    unit: 'sectors',
    explain: (
      <Translated
        vi={
          <p>
            Các lane phủ các phần tử 0 đến 124 của một đoạn 128 phần tử, tức 512 byte. Mỗi sector 32 byte trong đó chứa
            hai float được lưu và sáu float không ai cần: {SECTORS_STRIDE4} sector cho 128 byte hữu ích. Khi lane{' '}
            <code>t</code> ghi <code>base + t</code>, cùng 32 float đó lấp vừa khít {SECTORS_STRIDE1} sector. Sector
            32 byte là đơn vị của NVIDIA; AMD gom địa chỉ của một wavefront thành các dòng bộ nhớ đệm, còn Apple không
            ghi nhận quy tắc nào. Ở trang <a href={href('ch21', '21.2')}>21.2</a>, một luồng đã gộp việc sở hữu bốn điểm
            lưới liền kề, nên mỗi câu lệnh ghi trong bốn câu lệnh của nó có bước nhảy (stride) bốn này trên toàn warp:{' '}
            {4 * SECTORS_STRIDE4} sector cho cả bốn, trong khi cách sửa của sách cần {4 * SECTORS_STRIDE1}. Được dạy ở{' '}
            <a href={href('ch06', '6.1')}>ch. 6 §6.1</a>.
          </p>
        }
      >
        <p>
          The lanes cover elements 0 to 124 of a 128-element stretch, 512 bytes. Every 32-byte sector in it holds two of
          the stored floats and six that nobody asked for: {SECTORS_STRIDE4} sectors for 128 useful bytes. With lane{' '}
          <code>t</code> writing <code>base + t</code>, the same 32 floats fill {SECTORS_STRIDE1} sectors exactly. The
          32-byte sector is NVIDIA's unit; AMD groups a wavefront's addresses into cache lines, and Apple documents no
          rule. On page <a href={href('ch21', '21.2')}>21.2</a> a coarsened thread owns four adjacent grid points, so
          each of its four write statements has this stride of four across the warp: {4 * SECTORS_STRIDE4} sectors for
          the four, where the book's fix needs {4 * SECTORS_STRIDE1}. Taught in{' '}
          <a href={href('ch06', '6.1')}>ch. 6 §6.1</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.1, pp. 124–132; notes/today/ch06-facts.md A2 (NVIDIA\'s unit), B1, C1; ch21-facts E2',
    vi: { unit: 'sector', source: 'ch. 6 §6.1, pp. 124–132; notes/today/ch06-facts.md A2 (đơn vị của NVIDIA), B1, C1; ch21-facts E2' },
  },
  'ch21/start/q-divergent-test': {
    kind: 'choice',
    id: 'ch21/start/q-divergent-test',
    prompt: (
      <Translated
        vi={
          <>
            Một warp 32 lane chạy <code>if (r2 &lt; c2) e += q / sqrtf(r2);</code> và 8 lane của nó qua được phép kiểm
            tra. Warp thực thi những gì?
          </>
        }
      >
        A warp of 32 lanes runs <code>if (r2 &lt; c2) e += q / sqrtf(r2);</code> and 8 of its lanes pass the test.
        What does the warp execute?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Warp chạy thân lệnh một lần với 8 lane hoạt động và 24 lane bị che, mất khoảng chừng ấy thời gian như khi cả 32 lane đều qua.">
            The warp runs the body once with 8 lanes active and 24 masked off, taking about as long as if all 32 passed.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Thân lệnh 8 lần, mỗi lần cho một lane qua được, nên tốn một phần tư so với khi 32 lane đều qua.">The body 8 times, once for each passing lane, so it costs a quarter of what 32 passing lanes would.</Translated>,
        why: <Translated vi="Một warp chạy một lệnh cho mọi lane đang hoạt động cùng lúc. Số lượt chạy tính theo nhánh, không theo lane.">A warp runs one instruction for all its active lanes at once. The passes are per path, not per lane.</Translated>,
      },
      {
        label: <Translated vi="Không gì cả: warp bỏ qua một nhánh mà phần lớn lane của nó không đi vào, và 8 lane qua được giữ nguyên tổng cũ.">Nothing: a warp skips a branch that most of its lanes do not take, and the 8 passing lanes keep their old sums.</Translated>,
        why: <Translated vi="Warp chỉ bỏ qua một nhánh khi không lane nào đi vào. Chỉ một lane là đủ để cả warp chạy thân lệnh.">A warp skips a path only when none of its lanes take it. One lane is enough for the whole warp to run the body.</Translated>,
      },
      {
        label: <Translated vi="Thân lệnh một lần, trong khoảng một phần tư thời gian: chỉ 8 trên 32 lane hoạt động.">The body once, in about a quarter of the time: only 8 of the 32 lanes are active.</Translated>,
        why: <Translated vi="Các lane bị che vẫn chiếm chỗ của chúng suốt lượt chạy. Sách tính đó là một trong hai chi phí của phân kỳ luồng (divergence).">The masked lanes still occupy their slots for the whole pass. The book counts that as one of the two costs of divergence.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            NVIDIA: warp "thực thi từng nhánh được đi vào, vô hiệu hóa các luồng không nằm trên nhánh đó". Ở đây có một
            nhánh, thân lệnh, và 24 lane không qua ngồi không suốt nhánh đó. NVIDIA không đưa con số cho chi phí một lượt
            chạy, và lưu ý rằng trình biên dịch có thể biến một thân lệnh ngắn như thế này thành các lệnh có điều kiện vị
            từ (predicated) để không warp nào phân kỳ; cả hai cách mô tả đều không làm thân lệnh rẻ hơn khi ít lane cần nó
            hơn. Ở trang <a href={href('ch21', '21.5')}>21.5</a>, mọi luồng kiểm tra mọi nguyên tử của một ngăn lân cận
            với bán kính cắt quanh điểm lưới của riêng nó, và với các con số mẫu của sách, khoảng ba trên bốn phép kiểm
            tra không qua (số đếm của chúng tôi). Được dạy ở <a href={href('ch04', '4.5')}>ch. 4 §4.5</a>.
          </p>
        }
      >
        <p>
          NVIDIA: the warp "executes each branch path taken, disabling threads that are not on that path". Here there is
          one path, the body, and the 24 failing lanes sit idle through it. NVIDIA gives no figure for what a pass costs,
          and notes that a compiler may turn a short body like this into predicated instructions so that no warp diverges
          at all; neither description makes the body cheaper when fewer lanes need it. On page{' '}
          <a href={href('ch21', '21.5')}>21.5</a> every thread tests every atom of a neighbourhood bin against the cutoff
          around its own grid point, and at the book's worked numbers about three tests in four fail (our count). Taught
          in <a href={href('ch04', '4.5')}>ch. 4 §4.5</a>.
        </p>
      </Translated>
    ),
    source:
      'ch. 4 §4.5, pp. 79–82; notes/today/ch04-facts.md A9; ch06-facts A14 (predication); ch21-facts E4 (our count at the p. 525 numbers)',
    vi: {
      source:
        'ch. 4 §4.5, pp. 79–82; notes/today/ch04-facts.md A9; ch06-facts A14 (điều kiện vị từ); ch21-facts E4 (số đếm của chúng tôi với các con số ở p. 525)',
    },
  },
};

const near = (v: number, x: number, rel: number) => Math.abs(v - x) <= Math.abs(x) * rel;

/** The reveal after the matrix-size prediction, written for the common slips. */
function MaterialiseReveal({ v }: { v: number }) {
  const right = near(v, MATRIX_GB, 0.1) || near(v, MATRIX_GIB, 0.05);
  const gib = right && Math.abs(v - MATRIX_GIB) < Math.abs(v - MATRIX_GB);
  const noBytes = near(v, MATRIX_GB / F32, 0.1); // 100: elements, not bytes
  const fewPoints = near(v, MATRIX_GB / 10, 0.1); // 40: 10^5 points
  const twoMatrices = near(v, 2 * MATRIX_GB, 0.1); // 800: distances and quotients
  const terabytes = near(v, MATRIX_GB / 1000, 0.1); // 0.4: TB
  const decades = v > 0 ? Math.log10(v / MATRIX_GB) : NaN;
  const powerOfTen = !isNaN(decades) && Math.round(decades) !== 0 && Math.abs(decades - Math.round(decades)) < 0.05;
  return (
    <>
      <Translated
        vi={
          <p>
            <b>
              10<sup>6</sup> điểm × 10<sup>5</sup> nguyên tử × 4 byte = 4 × 10<sup>11</sup> byte: {fmt(MATRIX_GB)} GB.
            </b>{' '}
            Tức là {fmt(MATRIX_GIB)} GiB nếu bạn đếm theo lũy thừa của hai.
          </p>
        }
      >
        <p>
          <b>
            10<sup>6</sup> points × 10<sup>5</sup> atoms × 4 bytes = 4 × 10<sup>11</sup> bytes: {fmt(MATRIX_GB)} GB.
          </b>{' '}
          That is {fmt(MATRIX_GIB)} GiB if you count in powers of two.
        </p>
      </Translated>
      {right ? (
        <p>
          {gib ? tr('That is what you predicted, in GiB. ', 'Đúng như bạn dự đoán, tính bằng GiB. ') : tr('That is what you predicted. ', 'Đúng như bạn dự đoán. ')}
          <Translated vi={<>Mỗi cặp có một phần tử riêng, và có 10<sup>11</sup> cặp.</>}>
            Every pair gets its own entry, and there are 10<sup>11</sup> pairs.
          </Translated>
        </p>
      ) : noBytes ? (
        <Translated
          vi={
            <p>
              Bạn đã đếm số phần tử, không phải số byte: 10<sup>11</sup> phần tử là 100 tỷ. Mỗi phần tử là một float32 4
              byte, nên tính bằng byte, ma trận lớn gấp bốn lần con số đó.
            </p>
          }
        >
          <p>
            You counted entries, not bytes: 10<sup>11</sup> entries is 100 billion. Each entry is a float32 of 4 bytes, so
            the matrix is four times that in bytes.
          </p>
        </Translated>
      ) : fewPoints ? (
        <Translated
          vi={
            <p>
              Đó là 10<sup>5</sup> × 10<sup>5</sup> × 4 byte. Lưới có số điểm nhiều gấp mười lần số nguyên tử:
              10<sup>6</sup>.
            </p>
          }
        >
          <p>
            That is 10<sup>5</sup> × 10<sup>5</sup> × 4 bytes. The grid has ten times more points than there are atoms:
            10<sup>6</sup>.
          </p>
        </Translated>
      ) : twoMatrices ? (
        <Translated
          vi={
            <p>
              Bạn đã đếm hai ma trận, ma trận khoảng cách và ma trận thương. Viết theo cách này, biểu thức đúng là có thể
              tạo ra cả hai; câu hỏi chỉ hỏi về một.
            </p>
          }
        >
          <p>
            You counted two matrices, the distances and the quotients. Written this way the expression can indeed create
            both; the question asked for one.
          </p>
        </Translated>
      ) : terabytes ? (
        <Translated vi={<p>Đó là kích thước tính bằng terabyte. Tính bằng gigabyte, con số lớn gấp một nghìn lần.</p>}>
          <p>That is the size in terabytes. In gigabytes it is a thousand times the number.</p>
        </Translated>
      ) : powerOfTen ? (
        <Translated
          vi={
            <p>
              Bạn lệch một lũy thừa của mười. Hãy đếm số chữ số 0: 6 cho số điểm, 5 cho số nguyên tử, rồi nhân với 4 byte
              và chia cho 10<sup>9</sup> để ra gigabyte.
            </p>
          }
        >
          <p>
            You are off by a power of ten. Count the zeros: 6 for the points, 5 for the atoms, then multiply by 4 bytes
            and divide by 10<sup>9</sup> for gigabytes.
          </p>
        </Translated>
      ) : (
        <Translated
          vi={
            <p>
              Số phần tử là một phần tử cho mỗi cặp: 10<sup>6</sup> × 10<sup>5</sup> = 10<sup>11</sup> phần tử, mỗi phần
              tử 4 byte.
            </p>
          }
        >
          <p>
            The count is one entry per pair: 10<sup>6</sup> × 10<sup>5</sup> = 10<sup>11</sup> entries, 4 bytes each.
          </p>
        </Translated>
      )}
      <Translated
        vi={
          <p>
            Kết quả cuối cùng nó rút gọn về là một float cho mỗi điểm lưới, {fmt(RESULT_MB)} MB. Để đến được đó, ma trận
            phải được ghi ít nhất một lần và đọc lại ít nhất một lần: ít nhất {fmt(2 * MATRIX_GB)} GB lưu lượng bộ nhớ
            cho một kết quả {fmt(RESULT_MB)} MB, trước cả câu hỏi liệu nó có vừa bộ nhớ hay không.
          </p>
        }
      >
        <p>
          The answer it reduces to is one float per grid point, {fmt(RESULT_MB)} MB. To get there the matrix has to be
          written at least once and read back at least once: at least {fmt(2 * MATRIX_GB)} GB of memory traffic for a{' '}
          {fmt(RESULT_MB)} MB result, before any question of whether it fits.
        </p>
      </Translated>
    </>
  );
}

export function S210() {
  const q = questionsS210;
  return (
    <>
      <Translated
        vi={
          <p>
            Chương này đưa một phép tính qua một chuỗi kernel: thế tĩnh điện tại mọi điểm của một lưới (grid) 3D, do một
            tập các nguyên tử mang điện gây ra. Sách gọi phương pháp này là tổng Coulomb trực tiếp (direct Coulomb
            summation), viết tắt là DCS. Phân tử không đóng vai trò gì trong suy luận LLM, và lĩnh vực này chỉ có một
            đoạn ở đây. Điều mang sang được là chuỗi bốn quyết định mà sách đưa ra về một phép tổng trên mọi cặp:
          </p>
        }
      >
        <p>
          This chapter takes one computation through a sequence of kernels: the electrostatic potential at every point of
          a 3D grid, caused by a set of charged atoms. The book calls the method direct Coulomb summation, DCS for short.
          Molecules play no part in LLM inference, and the domain gets one paragraph here. What transfers is the sequence
          of four decisions the book makes about one all-pairs sum:
        </p>
      </Translated>
      <Source section="21 (opening), 21.6" pages="513, 527–528" />
      <Translated
        vi={
          <ul>
            <li>
              <b>Ai sở hữu mỗi đầu ra.</b> Một luồng (thread) cho mỗi nguyên tử thì rải (scatter) các đóng góp của nó lên
              lưới và cần một phép toán nguyên tử (atomic) cho mỗi lần cập nhật. Một luồng cho mỗi điểm lưới thì gom
              (gather) từ mọi nguyên tử và chỉ ghi một lần (§21.2, trang <a href={href('ch21', '21.2')}>21.2</a>).
            </li>
            <li>
              <b>Tái sử dụng bên trong một luồng.</b> Gộp việc cho luồng (thread coarsening): một luồng tính nhiều điểm
              lưới, nạp mỗi nguyên tử một lần và tái sử dụng những gì các điểm dùng chung (§21.3, trang{' '}
              <a href={href('ch21', '21.2')}>21.2</a>).
            </li>
            <li>
              <b>Một luồng sở hữu những đầu ra nào.</b> Giao cho một luồng các điểm liền kề làm mọi câu lệnh ghi không
              được hợp nhất; đặt chúng cách nhau một bề rộng khối thì sửa được điều đó (§21.4, trang{' '}
              <a href={href('ch21', '21.2')}>21.2</a>).
            </li>
            <li>
              <b>Đổi thuật toán để nó mở rộng được.</b> Khối lượng việc trên mọi cặp tăng theo bình phương thể tích của
              hệ. Một bán kính cắt (cutoff) cùng các ngăn không gian (spatial bin) làm nó tăng tuyến tính, đổi lại một
              chút độ chính xác (§21.5, trang <a href={href('ch21', '21.5')}>21.5</a>).
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            <b>Who owns each output.</b> A thread per atom scatters its contributions over the grid and needs an atomic
            for every update. A thread per grid point gathers from every atom and writes once (§21.2, page{' '}
            <a href={href('ch21', '21.2')}>21.2</a>).
          </li>
          <li>
            <b>Reuse inside one thread.</b> Coarsening: one thread computes several grid points, loads each atom once and
            reuses what the points share (§21.3, page <a href={href('ch21', '21.2')}>21.2</a>).
          </li>
          <li>
            <b>Which outputs a thread owns.</b> Giving a thread adjacent points makes every write statement uncoalesced;
            spacing them a block's width apart fixes it (§21.4, page <a href={href('ch21', '21.2')}>21.2</a>).
          </li>
          <li>
            <b>Changing the algorithm so it scales.</b> All-pairs work grows with the square of the system's volume. A
            cutoff with spatial bins makes it grow linearly, at a small cost in accuracy (§21.5, page{' '}
            <a href={href('ch21', '21.5')}>21.5</a>).
          </li>
        </ul>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Phần mở đầu (p. 513) liệt kê một "biến đổi gather sang scatter" trong số các kỹ thuật của chương. Chương lại
              đi theo chiều ngược lại, từ scatter sang gather, và khuyên dùng gather. Cùng danh sách đó nêu "kiểm tra điều
              kiện biên nhanh", điều mà không mục nào phát triển.
            </p>
          }
        >
          <p>
            The opening (p. 513) lists a "gather-to-scatter transformation" among the chapter's techniques. The chapter
            goes the other way, from scatter to gather, and recommends gather. The same list names "fast boundary
            condition checking", which no section develops.
          </p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Bài toán trong một đoạn">The problem in one paragraph</Translated>
      </h2>
      <Translated
        vi={
          <p>
            VMD là một chương trình để hiển thị, tạo hoạt ảnh và phân tích các mô hình phân tử sinh học. Một thứ nó tính
            là bản đồ thế tĩnh điện, được dùng chẳng hạn để tìm chỗ các ion vừa khít quanh một phân tử trước một mô phỏng
            động lực học phân tử.
          </p>
        }
      >
        <p>
          VMD is a program for displaying, animating and analysing models of biomolecules. One thing it computes is an
          electrostatic potential map, used for example to find where ions fit around a molecule before a
          molecular-dynamics simulation.
        </p>
      </Translated>
      <Source section="21.1" pages="513–514" figs="Fig. 21.2, p. 515" />
      <Translated
        vi={
          <p>
            Thế tại điểm lưới <i>j</i> là tổng trên mọi nguyên tử của điện tích mỗi nguyên tử chia cho khoảng cách từ nó
            đến điểm đó:{' '}
            <span style={{ whiteSpace: 'nowrap' }}>
              <i>V</i>
              <sub>j</sub> = Σ<sub>i</sub> <i>q</i>
              <sub>i</sub> / <i>r</i>
              <sub>ij</sub>
            </span>
            . Mọi nguyên tử đóng góp vào mọi điểm, nên khối lượng việc là số nguyên tử nhân số điểm lưới. Đó là lý do phép
            tính này xưa nay chạy như một tác vụ batch, và vì sao làm nó đủ nhanh để dùng tương tác là đáng để dùng GPU.
          </p>
        }
      >
        <p>
          The potential at grid point <i>j</i> is a sum over all atoms of each atom's charge divided by its distance to
          the point:{' '}
          <span style={{ whiteSpace: 'nowrap' }}>
            <i>V</i>
            <sub>j</sub> = Σ<sub>i</sub> <i>q</i>
            <sub>i</sub> / <i>r</i>
            <sub>ij</sub>
          </span>
          . Every atom contributes to every point, so the work is the number of atoms times the number of
          grid points. That is why the computation has traditionally run as a batch job, and why making it fast enough to
          use interactively was worth a GPU.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch21-facts.md, A1)">{/* i18n-ignore: a date and a citation */}
        <Translated
          vi={
            <p>
              VMD vẫn được phát hành: phiên bản 2.0.0 ra ngày 2026-03-25, với một bản build Linux dùng CUDA. Trang GPU
              của nó liệt kê việc đặt ion và tổng Coulomb trong số các tính năng được GPU tăng tốc.
            </p>
          }
        >
          <p>
            VMD is still released: version 2.0.0 came out on 2026-03-25, with a Linux build that uses CUDA. Its GPU page
            lists ion placement and Coulomb summation among the GPU-accelerated features.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Theo cách nói của PyTorch">In PyTorch terms</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Với các điểm lưới là một tensor <code>(P, 3)</code>, vị trí các nguyên tử là <code>(A, 3)</code> và điện tích
            là <code>(A,)</code>, một dòng tính ra bản đồ: <code>V = (q / torch.cdist(points, atoms)).sum(-1)</code>. Nó
            đúng, và nó dựng một ma trận khoảng cách P{' '}×{' '}A trước khi rút gọn bất cứ thứ gì. Chương 22 cho biết quy
            mô của một hệ thực tế: hàng trăm nghìn nguyên tử và hàng triệu điểm lưới.
          </p>
        }
      >
        <p>
          With the grid points as a <code>(P, 3)</code> tensor, the atom positions as <code>(A, 3)</code> and the charges
          as <code>(A,)</code>, one line computes the map: <code>V = (q / torch.cdist(points, atoms)).sum(-1)</code>. It
          is correct, and it builds a P{' '}×{' '}A matrix of distances before it reduces anything. Chapter 22 gives the scale of
          a realistic system: hundreds of thousands of atoms and millions of grid points.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Dòng PyTorch và phép so sánh với attention là của khóa học này. Quy mô là của sách (ch. 22 §22.2, p. 536),
              cũng như việc FlashAttention không bao giờ ghi ma trận điểm số của nó ra bộ nhớ toàn cục (ch. 20 §20.5, p.
              497).
            </p>
          }
        >
          <p>
            The PyTorch line and the comparison with attention are ours. The scale is the book's (ch. 22 §22.2, p. 536),
            as is the fact that FlashAttention never writes its score matrix to global memory (ch. 20 §20.5, p. 497).
          </p>
        </Translated>
      </Beyond>
      <Predict
        id="ch21/start/p-materialise"
        prompt={
          <Translated
            vi={
              <>
                Lấy 1,000,000 điểm lưới và 100,000 nguyên tử. Ma trận float32 mà{' '}
                <code>q / torch.cdist(points, atoms)</code> dựng ra trước khi <code>.sum(-1)</code> rút gọn nó lớn đến
                đâu? Hãy chốt một kích thước tính bằng GB.
              </>
            }
          >
            Take 1,000,000 grid points and 100,000 atoms. How large is the float32 matrix that{' '}
            <code>q / torch.cdist(points, atoms)</code> builds before <code>.sum(-1)</code> reduces it? Commit a size
            in GB.
          </Translated>
        }
        unit="GB"
      >
        {(v) => <MaterialiseReveal v={v} />}
      </Predict>
      <Translated
        vi={
          <p>
            Không kernel nào trong chương này tạo ra ma trận đó. Đóng góp của mỗi cặp được tính rồi cộng thẳng vào một
            tổng: vào một thanh ghi trong các kernel gather, vào lưới qua một phép toán nguyên tử trong kernel scatter.
            Một kernel attention gộp (fused) tránh ghi ma trận điểm số của nó vì cùng lý do, dù softmax làm việc đó
            khó hơn ở đấy: chương 20 xây dựng một kernel như vậy (<a href={href('ch20', '20.5')}>§20.5</a>).
          </p>
        }
      >
        <p>
          No kernel in this chapter forms that matrix. Each pair's contribution is computed and added straight into a
          sum: into a register in the gather kernels, into the grid through an atomic in the scatter kernel. A fused
          attention kernel avoids writing its score matrix for the same reason, though softmax makes it harder there:
          chapter 20 builds one (<a href={href('ch20', '20.5')}>§20.5</a>).
        </p>
      </Translated>

      <h2>
        <Translated vi="Chương này trong khóa học">The chapter in this course</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hai trang. Trang <a href={href('ch21', '21.2')}>21.2</a> bao quát §§21.2 đến 21.4 của sách: mã tuần tự và
            phép hoán đổi vòng lặp của nó, kernel scatter với phép toán nguyên tử, kernel gather với các nguyên tử nằm
            trong bộ nhớ hằng (constant memory), gộp việc theo hệ số bốn, và việc đổi quyền sở hữu để hợp nhất các lần
            ghi (pp. 515–522). Trang <a href={href('ch21', '21.5')}>21.5</a> bao quát §21.5: vì sao tổng trực tiếp không
            mở rộng được, bán kính cắt, các ngăn không gian và danh sách lân cận mà mỗi khối duyệt qua, và các ngăn kích
            thước cố định được đệm bằng nguyên tử giả, với một danh sách tràn do host xử lý nốt (pp. 523–527). Sau đó là
            một <a href={href('ch21', 'labs')}>lab</a> tùy chọn, xây các kernel scatter, gather và kernel đã gộp việc rồi
            chạy chúng trên GPU của bạn, và trang <a href={href('ch21', 'wrap')}>tóm tắt</a> với các bài tập của sách và
            những chỗ có vẻ sai trong PDF.
          </p>
        }
      >
        <p>
          Two pages. Page <a href={href('ch21', '21.2')}>21.2</a> covers the book's §§21.2 to 21.4: the sequential code
          and its loop interchange, the scatter kernel with atomics, the gather kernel with its atoms in constant memory,
          coarsening by four, and the change of ownership that coalesces the writes (pp. 515–522). Page{' '}
          <a href={href('ch21', '21.5')}>21.5</a> covers §21.5: why direct summation does not scale, the cutoff, spatial
          bins and the neighbourhood list each block scans, and fixed-size bins padded with dummy atoms, with an overflow
          list the host finishes (pp. 523–527). Then one optional <a href={href('ch21', 'labs')}>lab</a>, which builds
          the scatter, gather and coarsened kernels and runs them on your GPU, and the{' '}
          <a href={href('ch21', 'wrap')}>summary</a> with the book's exercises and the PDF's apparent slips.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Phần được rút gọn: phần nền của §21.1 (pp. 513–514), với danh sách tính năng được tăng tốc của VMD và hình
            phân tử được kết xuất ở Fig. 21.1, được thu lại thành đoạn văn ở trên. Trang tóm tắt liệt kê những gì khác bị
            bỏ qua, kèm số trang.
          </p>
        }
      >
        <p>
          What is condensed: the background of §21.1 (pp. 513–514), with VMD's list of accelerated features and the
          rendered molecule of Fig. 21.1, is reduced to the paragraph above. The summary page lists what else is left
          out, with pages.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Quyết định đầu tiên quay lại ở chương 22. Mục <a href={href('ch22', '22.2')}>§22.2</a> của nó phân loại mọi
            mẫu trong sách theo quyết định này, lấy đầu ra làm trung tâm (gather) hay lấy đầu vào làm trung tâm (scatter),
            và dùng chương này làm trường hợp gather thắng vì nó không cần phép toán nguyên tử.
          </p>
        }
      >
        <p>
          The first decision comes back in chapter 22. Its <a href={href('ch22', '22.2')}>§22.2</a> sorts every pattern
          in the book by it, output-centric (gather) or input-centric (scatter), and uses this chapter as the case where
          gather wins because it needs no atomics.
        </p>
      </Translated>
      <Source section="22.2" pages="532–536" />

      <h2>
        <Translated vi="Trước khi bắt đầu">Before you start</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Năm câu hỏi về các ý từ chương 4, 6, 7 và 9 và phụ lục A, theo thứ tự chương này dùng đến chúng. Mỗi lời giải
            thích kết thúc bằng trang dựa vào ý đó, và mỗi câu sẽ quay lại trong phần{' '}
            <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Five questions on ideas from chapters 4, 6, 7 and 9 and appendix A, in the order this chapter uses them. Each
          explanation ends with the page that relies on it, and each one comes back in{' '}
          <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={q['ch21/start/q-float-order']} label={tr('From chapter 9 and appendix A', 'Từ chương 9 và phụ lục A')} />
      <Ask q={q['ch21/start/q-const-lanes']} label={tr('From chapter 7', 'Từ chương 7')} />
      <Ask q={q['ch21/start/q-coarsen-cost']} label={tr('From chapter 6', 'Từ chương 6')} />
      <Ask q={q['ch21/start/q-sectors']} label={tr('From chapter 6', 'Từ chương 6')} />
      <Ask q={q['ch21/start/q-divergent-test']} label={tr('From chapter 4', 'Từ chương 4')} />
    </>
  );
}
