import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Beyond, Source } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { fmt, labStatus, useReports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { BOOK_MEMORY, perLocationRate } from '../../sim/atomicrate';
import { functionSource } from '../../sim/toolchain';

// The kernel for the first question is a snippet of our own, loaded at run time, so there
// is no copy to drift: labs/ch09/snippets/wrap_early_exit.cu. Wrap.snippets.test.ts runs it
// on the course's emulator and compares the bins it gets wrong with earlyExitLost below.
export const EARLY_FILE = 'ch09/snippets/wrap_early_exit.cu';
export const EARLY = { n: 1000, block: 256 } as const;
/** The image of the question and of the check: pixel i has value (37 i + 11) mod 256. */
export const earlyImage = (n: number): number[] => Array.from({ length: n }, (_, i) => (37 * i + 11) % 256);

/**
 * What histogramEarlyExit loses. Thread t of a block zeroes and commits the bins b with
 * b mod blockDim = t. In the last block only `live` threads stay, so the counts of that
 * block's pixels whose bin belongs to a thread that left are never committed. Returns the
 * number of threads that stay in the last block and, per bin, how many counts go missing.
 */
export function earlyExitLost(values: number[], block: number, bins = 256): { live: number; lost: Map<number, number> } {
  const n = values.length;
  const live = n % block;
  const lost = new Map<number, number>();
  if (live === 0) return { live: block, lost };
  for (let i = n - live; i < n; i++) {
    const b = values[i];
    if (b % block >= live && b < bins) lost.set(b, (lost.get(b) ?? 0) + 1);
  }
  return { live, lost };
}

/** Mixture-of-experts routing counts: the question's numbers. */
export const MOE = { tokens: 16384, topK: 2, experts: 64, block: 256 } as const;
export const moeAtomics = () => {
  const updates = MOE.tokens * MOE.topK;
  const blocks = Math.ceil(updates / MOE.block);
  return { updates, blocks, commits: blocks * MOE.experts };
};

/** Appending candidates through one global counter, in the book's DRAM model. */
export const APPEND = { candidates: 100000 } as const;
export const appendMs = () => (APPEND.candidates / perLocationRate(BOOK_MEMORY.clockHz, BOOK_MEMORY.latencyCycles)) * 1e3;

const early = earlyExitLost(earlyImage(EARLY.n), EARLY.block);
const lostBins = [...early.lost.keys()].sort((a, b) => a - b);
const lostRange = () =>
  lostBins.length
    ? tr(`${lostBins[0]} to ${lostBins[lostBins.length - 1]}`, `${lostBins[0]} đến ${lostBins[lostBins.length - 1]}`)
    : tr('none', 'không có');
const moe = moeAtomics();

export const questionsWrap: Record<string, Question> = {
  'ch09/wrap/q-early-exit': {
    kind: 'choice',
    id: 'ch09/wrap/q-early-exit',
    prompt: (
      <Translated
        vi={
          <>
            Kernel ở trên được khởi chạy với {EARLY.block} luồng mỗi khối trên một ảnh {EARLY.n.toLocaleString('en-US')}{' '}
            điểm ảnh, điểm ảnh <i>i</i> giữ giá trị (37<i>i</i> + 11) mod 256, nên mỗi bin chứa 3 hoặc 4 điểm ảnh. Nó biên dịch được và
            không bao giờ treo. Số đếm nào ra sai?
          </>
        }
      >
        <>
          The kernel above is launched with {EARLY.block} threads per block on an image of {EARLY.n.toLocaleString('en-US')}{' '}
          pixels, pixel <i>i</i> holding (37<i>i</i> + 11) mod 256, so that every bin holds 3 or 4 pixels. It compiles and
          never hangs. Which counts come out wrong?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không cái nào: một luồng đã return thì không giữ các luồng khác lại ở rào chắn.">
            None: a thread that has returned does not hold the others at a barrier.
          </Translated>
        ),
        why: (
          <Translated vi="Phần đó đúng trên NVIDIA, nên các rào chắn vẫn hoạt động. Nhưng các luồng đã rời đi còn có việc khác ngoài điểm ảnh của chúng.">
            That part is true on NVIDIA, so the barriers work. But the threads that left also had jobs besides their pixel.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi={`Các bin ${lostRange()}, mỗi bin thiếu một: trong khối cuối, không ai xóa hay ghi chúng.`}>
            {`Bins ${lostRange()}, one count short each: in the last block, nobody zeroes or commits them.`}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi={`Các bin 0 đến ${early.live - 1}: chỉ ${early.live} luồng còn lại trong khối cuối đếm điểm ảnh của chúng.`}>
            {`Bins 0 to ${early.live - 1}: only the ${early.live} threads left in the last block count their pixels.`}
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi luồng còn lại đếm điểm ảnh của chính nó, bất kể giá trị. Cái thiếu là ai đó xóa và ghi các bin còn lại.">
            Each remaining thread counts its own pixel, whatever its value. What is missing is somebody to zero and commit the other bins.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Tất cả, một cách ngẫu nhiên: các rào chắn không còn được mọi luồng của mọi khối đi tới.">
            All of them, at random: the barriers are no longer reached by every thread of every block.
          </Translated>
        ),
        why: (
          <Translated vi="Rào chắn của NVIDIA chờ các luồng chưa thoát, nên ở đó kernel có hành vi xác định; thiệt hại là tất định và chỉ giới hạn trong khối cuối.">
            NVIDIA's barrier waits for the threads that have not exited, so the kernel is well defined there; the damage is deterministic and confined to the last block.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Các vòng lặp xóa và ghi chia bin theo chỉ số luồng: luồng <i>t</i> sở hữu mọi bin{' '}
              <i>b</i> có <i>b</i> mod {EARLY.block} = <i>t</i>. Ba khối đầu đầy đủ. Khối cuối có{' '}
              {EARLY.n.toLocaleString('en-US')} − {EARLY.n - early.live} = {early.live} điểm ảnh, nên các luồng {early.live} đến{' '}
              {EARLY.block - 1} return ngay lập tức, và không ai xóa hay ghi các bin {lostRange()} trong bản sao của khối đó. Các
              luồng còn lại vẫn đếm các điểm ảnh có giá trị đó vào các bin này, và số đếm nằm lại ở đó. Trình giả lập cho thấy
              đúng {lostBins.length} bin này thiếu một, dưới cả ba thứ tự luồng, và ở 128 luồng thì là một tập khác.
            </p>
            <p>
              Một luồng không có điểm ảnh nào vẫn có việc: phần của nó trong bước xóa và bước ghi. Vì vậy Hình 9.10 chỉ canh
              phép cập nhật bằng <code>if(i &lt; width*height)</code>, và Hình 9.15 cho phép một luồng có điểm ảnh đầu tiên nằm ngoài
              phạm vi bỏ qua các điểm ảnh của nó mà vẫn tham gia rào chắn và bước ghi. Rào chắn của NVIDIA đúng là chỉ chờ các luồng
              chưa thoát, nhưng quy tắc của sách, mọi luồng đều tới mọi rào chắn, mới là thứ giữ bạn khỏi
              cái bẫy này.
            </p>
          </>
        }
      >
        <>
          <p>
            The zeroing and the commit loops hand out the bins by thread index: thread <i>t</i> owns every bin{' '}
            <i>b</i> with <i>b</i> mod {EARLY.block} = <i>t</i>. The first three blocks are full. The last one has{' '}
            {EARLY.n.toLocaleString('en-US')} − {EARLY.n - early.live} = {early.live} pixels, so threads {early.live} to{' '}
            {EARLY.block - 1} return at once, and nobody zeroes or commits bins {lostRange()} of that block's copy. The
            remaining threads still count pixels with those values into them, and the counts stay there. The emulator shows
            exactly these {lostBins.length} bins one short, under all three thread orders, and at 128 threads a different set.
          </p>
          <p>
            A thread with no pixel still has work: its share of the zeroing and of the commit. Fig. 9.10 therefore guards
            only the update with <code>if(i &lt; width*height)</code>, and Fig. 9.15 lets a thread whose first pixel is out
            of range skip its pixels and still join the barrier and the commit. NVIDIA's barrier does wait only for threads
            that have not exited, but the book's rule, every thread reaches every barrier, is the one that keeps you out of
            this trap.
          </p>
        </>
      </Translated>
    ),
    source: 'ours, after ch. 9 Fig. 9.10, p. 214, and Fig. 9.15, pp. 218–219; run on the course emulator by Wrap.snippets.test.ts; notes/today/ch04-facts.md A6',
    vi: { source: 'của khóa học này, dựa theo ch. 9 Hình 9.10, p. 214, và Hình 9.15, pp. 218–219; chạy trên trình giả lập của khóa học bởi Wrap.snippets.test.ts; notes/today/ch04-facts.md A6' },
  },
  'ch09/wrap/q-float-weights': {
    kind: 'choice',
    id: 'ch09/wrap/q-float-weights',
    prompt: (
      <Translated
        vi={
          <>
            Trên CUDA, <code>torch.bincount(ids, weights=w)</code> với trọng số float phát ra một cảnh báo khi bạn yêu cầu PyTorch dùng
            thuật toán tất định. Không có trọng số thì không. Vì sao?
          </>
        }
      >
        <>
          On CUDA, <code>torch.bincount(ids, weights=w)</code> with float weights raises an alert when you ask PyTorch for
          deterministic algorithms. Without weights it does not. Why?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Phép toán nguyên tử float có thể làm rơi một cập nhật khi tranh chấp nặng, điều mà phép toán nguyên tử số nguyên không làm.">
            Float atomics can drop an update under heavy contention, which integer atomics do not.
          </Translated>
        ),
        why: (
          <Translated vi="Cả hai đều nguyên tử: mọi cập nhật đều tới. Điều khác nhau là thứ tự đến ảnh hưởng thế nào tới kết quả.">
            Both are atomic: every update arrives. What differs is what the order of arrival does to the result.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các phép toán nguyên tử đến theo thứ tự không cố định, và tổng float làm tròn ở mỗi bước, nên các bit cuối có thể thay đổi.">
            Atomics land in no fixed order, and float sums round at every step, so the last bits can change.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Bin có trọng số cần một bản sao riêng lớn hơn, khiến việc đếm chuyển từ bộ nhớ chia sẻ sang bộ nhớ toàn cục.">
            Weighted bins need a larger private copy, which moves the counting from shared to global memory.
          </Translated>
        ),
        why: (
          <Translated vi="Bản sao nằm ở đâu thì ảnh hưởng tốc độ. Ở chỗ nào thì các phép toán nguyên tử, và các lần ghi của các khối, vẫn đến theo thứ tự không cố định.">
            Where the copy lives changes the speed. In either place the atomics, and the blocks' commits, still arrive in no fixed order.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Có trọng số thì khối nào đếm một phần tử thay đổi giữa các lần chạy, nên các tổng từng phần khác nhau.">
            With weights, which block counts an element changes from run to run, so partial sums differ.
          </Translated>
        ),
        why: (
          <Translated vi="Cách chia được cố định bởi các chỉ số. Mỗi phần tử được cùng một luồng đếm đúng một lần; chỉ thứ tự các phép cộng là thay đổi.">
            The partitioning is fixed by the indices. Every element is counted once by the same thread; only the order of the additions varies.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một phép toán nguyên tử làm mỗi cập nhật không thể chia cắt và không sắp thứ tự gì cả (p. 206). Sách trộn các bản sao riêng "theo bất kỳ
            thứ tự nào" vì phép cộng có tính giao hoán và kết hợp (p. 211). Điều đó chính xác với số đếm nguyên miễn là
            chúng không thể tràn. Phép cộng float làm tròn sau mỗi bước, nên nó không kết hợp một cách chính xác, và hai lần chạy có thể
            cộng cùng các trọng số theo thứ tự khác nhau. <code>bincount</code> CUDA của PyTorch phát cảnh báo chính vì
            lý do này, "because of floating point atomicAdd usage". Điều tương tự đúng với mọi scatter-add float, như
            gradient của một bảng embedding.
          </p>
        }
      >
        <p>
          An atomic makes each update indivisible and orders nothing (p. 206). The book merges private copies "in any
          order" because addition is commutative and associative (p. 211). That is exact for integer counts as long as
          they cannot overflow. Float addition rounds after each step, so it is not exactly associative, and two runs can
          add the same weights in different orders. PyTorch's CUDA <code>bincount</code> raises its alert for exactly this
          reason, "because of floating point atomicAdd usage". The same holds for any float scatter-add, such as an
          embedding table's gradient.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.2 p. 206 and §9.4 p. 211; notes/today/ch09-facts.md D1 (PyTorch SummaryOps.cu)',
    vi: { source: 'ch. 9 §9.2 p. 206 và §9.4 p. 211; notes/today/ch09-facts.md D1 (PyTorch SummaryOps.cu)' },
  },
  'ch09/wrap/q-moe': {
    kind: 'choice',
    id: 'ch09/wrap/q-moe',
    prompt: (
      <Translated
        vi={
          <>
            Một lớp mixture-of-experts gửi mỗi token trong {MOE.tokens.toLocaleString('en-US')} token tới {MOE.topK} trong{' '}
            {MOE.experts} expert và cần biết số token mỗi expert nhận được. Các lựa chọn của router không đều: một vài
            expert nhận nhiều hơn hẳn phần của chúng. Kernel nào?
          </>
        }
      >
        <>
          A mixture-of-experts layer sends each of {MOE.tokens.toLocaleString('en-US')} tokens to {MOE.topK} of{' '}
          {MOE.experts} experts and needs the number of tokens each expert received. The router's choices are uneven: a few
          experts get far more than their share. Which kernel?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi={`Một luồng cho mỗi lựa chọn, mỗi luồng làm một fetch_add phạm vi device trên một trong ${MOE.experts} bộ đếm trong bộ nhớ toàn cục.`}>
            {`One thread per choice, each doing a device-scope fetch_add on one of ${MOE.experts} counters in global memory.`}
          </Translated>
        ),
        why: (
          <Translated vi={`Đúng nhưng chậm chính trên đầu vào này: ${moe.updates.toLocaleString('en-US')} phép toán nguyên tử xếp hàng trên vài bộ đếm bận, trường hợp dữ liệu lệch của mục 9.3.`}>
            {`Correct but slow on exactly this input: ${moe.updates.toLocaleString('en-US')} atomics queue on a few busy counters, the skewed case of section 9.3.`}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi={`Một bản sao riêng của ${MOE.experts} bộ đếm cho mỗi khối trong bộ nhớ chia sẻ, rồi ghi các bộ đếm khác không bằng phép toán nguyên tử phạm vi device.`}>
            {`A private copy of the ${MOE.experts} counters per block in shared memory, then a commit of the non-zero ones with device-scope atomics.`}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi={`${MOE.experts} luồng, mỗi luồng một expert, mỗi luồng đọc mọi lựa chọn và đếm phần của mình.`}>
            {`${MOE.experts} threads, one per expert, each reading every choice and counting its own.`}
          </Translated>
        ),
        why: (
          <Translated vi={`Lại là owner-computes, nhưng ${MOE.experts} luồng không thể lấp đầy một GPU, và mỗi luồng đọc cả ${moe.updates.toLocaleString('en-US')} lựa chọn.`}>
            {`Owner-computes again, but ${MOE.experts} threads cannot fill a GPU, and each reads all ${moe.updates.toLocaleString('en-US')} choices.`}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một luồng cho mỗi lựa chọn với một phép tăng thường: số đếm chỉ dùng để định kích thước buffer, nên vài cập nhật bị mất cũng không sao.">
            One thread per choice with a plain increment: the counts are only used to size buffers, so a few lost updates do not matter.
          </Translated>
        ),
        why: (
          <Translated vi="Một cập nhật bị mất làm một số đếm quá nhỏ, và buffer được định kích thước từ nó cũng quá nhỏ: token bị ghi vượt quá cuối buffer.">
            A lost update makes a count too small, and a buffer sized from it too small: tokens are written past its end.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            {MOE.experts} bộ đếm 4 byte là 256 byte, vừa với bộ nhớ chia sẻ của bất kỳ khối nào. Với các khối {MOE.block}{' '}
            luồng, mỗi luồng một lựa chọn, {moe.updates.toLocaleString('en-US')} cập nhật đi vào {moe.blocks} bản sao riêng, và
            nhiều nhất {moe.blocks} × {MOE.experts} = {moe.commits.toLocaleString('en-US')} phép toán nguyên tử tới bộ nhớ toàn cục, một phần tư
            so với kernel cơ bản. Quan trọng hơn, hàng đợi của mỗi expert bận được chia ra cho các khối đang chạy, và
            việc định tuyến không đều vốn làm kernel cơ bản khổ nhất lại là nơi riêng tư hóa có lợi nhất (mục 9.3 và 9.4). Với một
            batch lớn hơn, gộp việc (mục 9.5) cắt giảm số lần ghi thêm nữa.
          </p>
        }
      >
        <p>
          {MOE.experts} counters of 4 bytes are 256 bytes, which fits any block's shared memory. With blocks of {MOE.block}{' '}
          threads, one choice each, the {moe.updates.toLocaleString('en-US')} updates go to {moe.blocks} private copies, and at
          most {moe.blocks} × {MOE.experts} = {moe.commits.toLocaleString('en-US')} atomics reach global memory, a quarter
          of the basic kernel's. More to the point, each busy expert's queue is split across the blocks in flight, and the
          uneven routing that hurts the basic kernel most is where privatisation pays most (sections 9.3 and 9.4). With a
          larger batch, coarsening (section 9.5) cuts the commits further.
        </p>
      </Translated>
    ),
    source: 'ours; the design is ch. 9 §9.4, pp. 211–214, and §9.5, p. 214',
    vi: { source: 'của khóa học này; thiết kế là ch. 9 §9.4, pp. 211–214, và §9.5, p. 214' },
  },
  'ch09/wrap/q-narrow-counts': {
    kind: 'choice',
    id: 'ch09/wrap/q-narrow-counts',
    prompt: (
      <Translated
        vi={
          <>
            Để chứa bốn bản sao riêng mỗi khối, một đồng nghiệp đổi các bin chia sẻ của Hình 9.14 thành{' '}
            <code>unsigned char bins_s[256]</code>, một byte cho mỗi số đếm, giữ 256 luồng và hệ số gộp việc 4.
            Điều gì sai?
          </>
        }
      >
        <>
          To fit four private copies per block, a colleague changes Fig. 9.14's shared bins to{' '}
          <code>unsigned char bins_s[256]</code>, one byte per count, keeping 256 threads and a coarsening factor of 4.
          What goes wrong?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không gì cả: 256 bin một byte dễ dàng vừa, kể cả bốn bản sao, trong bộ nhớ chia sẻ của bất kỳ khối nào.">
            Nothing: 256 one-byte bins fit easily, four copies included, in any block's shared memory.
          </Translated>
        ),
        why: <Translated vi="Vừa hay không chưa bao giờ là vấn đề. Phạm vi giá trị của một số đếm mới là vấn đề.">Fitting was never the problem. The range of a count is.</Translated>,
      },
      {
        label: (
          <Translated vi="Một bin có thể nhận 1,024 lần đếm và quay vòng qua 255, còn phép toán nguyên tử 8-bit thì là giả lập chậm hoặc không có.">
            A bin can receive 1,024 counts and wrap past 255, and 8-bit atomics are slow emulations or absent.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chỉ bước ghi: một phép toán nguyên tử phạm vi device không thể cộng một giá trị 8-bit vào một bin công khai 32-bit.">
            Only the commit: a device-scope atomic cannot add an 8-bit value to a 32-bit public bin.
          </Translated>
        ),
        why: (
          <Translated vi="Bước ghi đọc số đếm và cộng nó vào một bin unsigned int như mọi giá trị khác. Thiệt hại đã xảy ra trước đó.">
            The commit reads the count and adds it to an unsigned int bin like any other value. The damage is done before it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó chậm hơn vì xung đột bank giữa các byte kế nhau, nhưng mọi số đếm đều đúng.">
            It is slower because of bank conflicts between neighbouring bytes, but every count is right.
          </Translated>
        ),
        why: (
          <Translated vi="Một ảnh hằng dồn cả 1,024 điểm ảnh của một khối vào một bin, và một byte giữ được nhiều nhất 255.">
            A constant image puts all 1,024 of a block's pixels in one bin, and one byte holds at most 255.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hãy định kích thước bộ đếm riêng cho số cập nhật nhiều nhất mà một bản sao có thể nhận: ở đây 256 luồng × 4 điểm ảnh = 1,024, cần
            nhiều hơn 8 bit. Phép toán nguyên tử hẹp cũng tốn tốc độ, hoặc không tồn tại: libcu++ nói các kiểu nhỏ hơn 4 byte "are not natively
            supported by the underlying hardware" (không được phần cứng bên dưới hỗ trợ gốc) và được giả lập, "drastically slower" (chậm hơn rất nhiều), với tranh chấp trên
            từ 4 byte bao quanh trở nên tệ hơn. Metal không có kiểu atomic 8 hay 16 bit; PyTorch và MLX dựng chúng từ một vòng lặp
            compare-and-swap trên một từ 32-bit. Với một bộ từ vựng, nơi bộ đếm hẹp hấp dẫn nhất, điều tương tự cũng đúng.
          </p>
        }
      >
        <p>
          Size a private counter for the most updates one copy can receive: here 256 threads × 4 pixels = 1,024, which needs
          more than 8 bits. Narrow atomics also cost speed, or do not exist: libcu++ says types smaller than 4 bytes "are not natively
          supported by the underlying hardware" and are emulated, "drastically slower", with contention on the surrounding
          4-byte word made worse. Metal has no 8- or 16-bit atomic type; PyTorch and MLX build them from a compare-and-swap
          loop on a 32-bit word. For a vocabulary, where a narrow counter is most tempting, the same applies.
        </p>
      </Translated>
    ),
    source: 'ours, after ch. 9 Fig. 9.14, p. 217; notes/today/ch09-facts.md A1, C2, D7',
    vi: { source: 'của khóa học này, dựa theo ch. 9 Hình 9.14, p. 217; notes/today/ch09-facts.md A1, C2, D7' },
  },
  'ch09/wrap/q-append': {
    kind: 'numeric',
    id: 'ch09/wrap/q-append',
    prompt: (
      <Translated
        vi={
          <>
            Một sampler ghi chỉ số của {APPEND.candidates.toLocaleString('en-US')} ứng viên còn lại vào một mảng,
            mỗi luồng lấy ô của mình bằng <code>pos = counter.fetch_add(1)</code> trên một bộ đếm toàn cục. Trong mô hình DRAM
            của sách (độ trễ 200 chu kỳ, 1 GHz, không có gì được cache), các phép toán nguyên tử đó mất bao nhiêu mili giây?
          </>
        }
      >
        <>
          A sampler writes the indices of its {APPEND.candidates.toLocaleString('en-US')} surviving candidates to one array,
          each thread taking its slot with <code>pos = counter.fetch_add(1)</code> on one global counter. In the book's DRAM
          model (200-cycle latency, 1 GHz, nothing cached), how many milliseconds do those atomics take?
        </>
      </Translated>
    ),
    answer: Number(appendMs().toFixed(3)),
    tol: 0.02,
    unit: 'ms',
    explain: (
      <Translated
        vi={
          <p>
            Một vị trí nhận một phép toán nguyên tử mỗi 200 + 200 chu kỳ: 2.5 triệu mỗi giây. {APPEND.candidates.toLocaleString('en-US')}{' '}
            / 2.5 M = {fmt(appendMs())} ms, trong lúc đó phần còn lại của GPU chờ trên một hàng đợi, dù bạn khởi chạy
            bao nhiêu luồng. Đây là lần đầu dùng giá trị trả về của <code>fetch_add</code>, thứ mà sách nói chương 12 và 18
            cần (p. 209). Cách chữa có hình dạng của chương này: đếm riêng các ứng viên của một khối, rồi giữ chỗ cho cả
            dải của khối bằng một <code>fetch_add</code> duy nhất với số đếm đó, một phép toán nguyên tử mỗi khối. Chương 12 xây dựng nó, ở{' '}
            <a href={href('ch12', '12.4')}>mục 12.4</a>.
          </p>
        }
      >
        <p>
          One location takes one atomic per 200 + 200 cycles: 2.5 million per second. {APPEND.candidates.toLocaleString('en-US')}{' '}
          / 2.5 M = {fmt(appendMs())} ms, during which the rest of the GPU waits on one queue, however many threads you
          launch. This is the first use of <code>fetch_add</code>'s return value, which the book says chapters 12 and 18
          need (p. 209). The cure has this chapter's shape: count a block's candidates privately, then reserve the block's
          whole range with a single <code>fetch_add</code> of that count, one atomic per block. Chapter 12 builds it, in{' '}
          <a href={href('ch12', '12.4')}>section 12.4</a>.
        </p>
      </Translated>
    ),
    source: 'the §9.3 model, pp. 209–210; p. 209 for the return value; the sampler is ours',
    vi: { source: 'mô hình của §9.3, pp. 209–210; p. 209 cho giá trị trả về; sampler là của khóa học này' },
  },
};

/** The kernel of the first question, from the snippet file. */
function EarlyExitKernel() {
  const file = useLabFile(EARLY_FILE);
  const fn = file && !file.startsWith('// could not load') ? functionSource(file, 'histogramEarlyExit') : null;
  return (
    <Code
      tabs={[
        {
          label: tr('CUDA (not a lab file: for this question only)', 'CUDA (không phải tệp lab: chỉ cho câu hỏi này)'),
          code: fn ? fn.text : file ?? `// loading labs/${EARLY_FILE}`,
          status: tr('emulator-checked; wrong on purpose', 'emulator-checked; cố ý sai'),
          tiers: { image: 'global', bins: 'global', bins_s: 'shared', i: 'reg', b: 'reg' },
        },
      ]}
      caption={
        <Translated
          vi={
            <>
              Từ <code>labs/{EARLY_FILE}</code>: Hình 9.10 với một thay đổi. Các bài kiểm tra đơn vị chạy nó trên trình giả lập của khóa học.
            </>
          }
        >
          <>
            From <code>labs/{EARLY_FILE}</code>: Fig. 9.10 with one change. The unit tests run it on the course's emulator.
          </>
        </Translated>
      }
    />
  );
}

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const m5 = backends.filter((b) => labStatus(reports[b], 'ch09/lab3_tokencounts', 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch09/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}><Translated vi="Những gì cần mang theo">What to carry forward</Translated></h2>
      <Translated
        vi={
          <p>
            Phần tóm tắt của sách, mở rộng với những gì khóa học này thêm vào về các backend và công cụ, theo thứ tự bạn sẽ
            cần đến.
          </p>
        }
      >
        <p>
          The book's summary, extended with what this course added on the backends and the tools, in the order you will
          need it.
        </p>
      </Translated>
      <Source section="9.7" pages="219–220" />
      <ul>
        <Translated
          vi={
            <li>
              Khi dữ liệu quyết định một luồng ghi vào đâu, owner-computes không còn nữa: luồng nào cũng có thể cập nhật đầu ra nào. Một
              cập nhật thường là một lần đọc, một phép cộng và một lần ghi, và hai cập nhật như vậy trên một vị trí có thể làm mất một. Về hình thức đó là một data
              race, nên một số đếm bị mất là kết quả điển hình, không phải kết quả được hứa hẹn.
            </li>
          }
        >
          <li>
            When the data decides where a thread writes, owner-computes is gone: any thread may update any output. A plain
            update is a read, an add and a write, and two of them on one location can lose one. Formally that is a data
            race, so a lost count is the typical result, not a promised one.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Một phép toán nguyên tử làm một cập nhật không thể chia cắt và không sắp thứ tự gì cả. Chọn phạm vi của nó theo các luồng dùng chung
              vị trí đó: khối cho bản sao riêng của một khối, device cho các bin mà nhiều khối cập nhật. Luôn truyền phạm vi và
              thứ tự bộ nhớ; nếu bỏ đi, chúng mặc định là rộng nhất và chặt nhất.
            </li>
          }
        >
          <li>
            An atomic makes one update indivisible and orders nothing. Choose its scope from the threads that share the
            location: block for a block's private copy, device for bins that several blocks update. Pass the scope and the
            memory order every time; left out, they default to the widest and the strictest.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Các phép toán nguyên tử trên một vị trí chạy lần lượt từng cái một, khoảng một cái cho mỗi độ trễ nạp-cộng-ghi, và bin bận nhất quyết định
              nhịp độ của cả kernel. Dữ liệu lệch là trường hợp tệ nhất, và đó lại là trường hợp phổ biến.
            </li>
          }
        >
          <li>
            Atomics on one location run one at a time, at about one per load-plus-store latency, and the busiest bin sets
            the pace of the whole kernel. Skewed data is the worst case, and it is the common one.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Riêng tư hóa: cho một tập con luồng bản sao riêng của nó và trộn các bản sao ở cuối. Một bản sao mỗi khối, trong bộ nhớ
              chia sẻ khi vừa: xóa về không, rào chắn, đếm, rào chắn, ghi các bin khác không.
            </li>
          }
        >
          <li>
            Privatise: give a subset of threads its own copy and merge the copies at the end. One copy per block, in shared
            memory when it fits: zero it, barrier, count, barrier, commit the non-zero bins.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Mỗi bản sao riêng tốn một lần xóa và một lần ghi. Gộp việc cho đến khi vừa đủ số khối, và giữ chỉ số
              xen kẽ để các lần đọc của một warp vẫn nằm kề nhau. Chỉ gộp chuỗi trong một thanh ghi ở nơi một luồng thấy các chuỗi dài
              của cùng một giá trị.
            </li>
          }
        >
          <li>
            Each private copy costs a zeroing and a commit. Coarsen until there are just enough blocks, and keep the
            interleaved index so a warp's reads stay neighbours. Merge runs in a register only where one thread sees long
            runs of one value.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Một bộ từ vựng không vừa trên chip. Hãy đếm theo từng request, trong bộ nhớ toàn cục, và cập nhật số đếm mỗi bước decode một
              lần thay vì đếm lại. Kiểm thử mọi phép toán nguyên tử trên backend race; kiểm tra mọi phạm vi bằng cách đọc mã.
            </li>
          }
        >
          <li>
            A vocabulary does not fit on chip. Count per request, in global memory, and update the counts one decode step at
            a time rather than recounting. Test every atomic on the race backend; check every scope by reading it.
          </li>
        </Translated>
      </ul>

      <h2><Translated vi="Một bài kiểm tra tổng hợp">A mixed check</Translated></h2>
      <Translated
        vi={
          <p>
            Các câu này dựa trên cả chương và trên những tình huống mà các mục chưa trình bày.
            {asked ? ` Đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu hỏi, tính theo câu trả lời gần nhất của bạn cho mỗi câu.` : ''}
          </p>
        }
      >
        <p>
          These draw on the whole chapter and on situations the sections did not show.
          {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Kernel, router, sampler và các bộ đếm hẹp là của khóa học này.</p>}>
          <p>The kernel, the router, the sampler and the narrow counters are ours.</p>
        </Translated>
      </Beyond>
      <EarlyExitKernel />
      <Ask q={q['ch09/wrap/q-early-exit']} label={tr('Find the defect', 'Tìm lỗi')} />
      <Ask q={q['ch09/wrap/q-float-weights']} />
      <Ask q={q['ch09/wrap/q-moe']} />
      <Ask q={q['ch09/wrap/q-narrow-counts']} />
      <Ask q={q['ch09/wrap/q-append']} />

      <h2><Translated vi="Bài tập của sách">The book's exercises</Translated></h2>
      <Translated
        vi={
          <p>
            Chương kết thúc với năm bài tập (p. 220), bài cuối gồm ba phần. Mỗi bài đều được giải ở đâu đó trong khóa học này;
            sách không in đáp án, nên đáp án ở đây là của khóa học này.
          </p>
        }
      >
        <p>
          The chapter ends with five exercises (p. 220), the last in three parts. Each is worked somewhere in this course;
          the book prints no answers, so the answers here are ours.
        </p>
      </Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }} data-table="exercises">
          <thead>
            <tr>
              <th><Translated vi="Bài tập">Exercise</Translated></th>
              <th><Translated vi="Nội dung">About</Translated></th>
              <th><Translated vi="Được giải ở đâu trong khóa học">Where it is worked here</Translated></th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1</td>
              <td><Translated vi="Số phép toán nguyên tử mỗi giây trên một biến, mỗi phép 100 ns">Atomics per second on one variable at 100 ns each</Translated></td>
              <Translated
                vi={
                  <td>
                    một câu hỏi ở <a href={href('ch09', '9.3')}>9.3</a>, và trình khám phá thông lượng ở đó
                  </td>
                }
              >
                <td>
                  a question in <a href={href('ch09', '9.3')}>9.3</a>, and the throughput explorer there
                </td>
              </Translated>
            </tr>
            <tr>
              <td>2</td>
              <td><Translated vi="Như trên với phép toán nguyên tử trong L2: 4 ns khi trúng, 90% trúng">The same with atomics in L2: 4 ns on a hit, 90% hits</Translated></td>
              <Translated
                vi={
                  <td>
                    một câu hỏi ở <a href={href('ch09', '9.3')}>9.3</a>
                  </td>
                }
              >
                <td>
                  a question in <a href={href('ch09', '9.3')}>9.3</a>
                </td>
              </Translated>
            </tr>
            <tr>
              <td>3</td>
              <td><Translated vi="Tốc độ dấu phẩy động bị giới hạn bởi các phép toán nguyên tử của bài tập 1">Floating-point rate limited by exercise 1's atomics</Translated></td>
              <Translated
                vi={
                  <td>
                    một câu hỏi ở <a href={href('ch09', '9.3')}>9.3</a>
                  </td>
                }
              >
                <td>
                  a question in <a href={href('ch09', '9.3')}>9.3</a>
                </td>
              </Translated>
            </tr>
            <tr>
              <td>4</td>
              <td><Translated vi="Như trên sau khi riêng tư hóa vào bộ nhớ chia sẻ">The same after privatising into shared memory</Translated></td>
              <Translated
                vi={
                  <td>
                    một câu hỏi ở <a href={href('ch09', '9.3')}>9.3</a>, với cả hai cách hiểu bài tập
                  </td>
                }
              >
                <td>
                  a question in <a href={href('ch09', '9.3')}>9.3</a>, with both readings of the exercise
                </td>
              </Translated>
            </tr>
            <tr>
              <td>5a</td>
              <td><Translated vi="Số phép toán nguyên tử toàn cục của kernel cơ bản, 524,288 phần tử">Global atomics of the basic kernel, 524,288 elements</Translated></td>
              <Translated
                vi={
                  <td>
                    một câu hỏi ở <a href={href('ch09', '9.4')}>9.4</a>
                  </td>
                }
              >
                <td>
                  a question in <a href={href('ch09', '9.4')}>9.4</a>
                </td>
              </Translated>
            </tr>
            <tr>
              <td>5b</td>
              <td><Translated vi="Số nhiều nhất khi riêng tư hóa vào bộ nhớ chia sẻ">The most for shared-memory privatisation</Translated></td>
              <Translated
                vi={
                  <td>
                    một câu hỏi và trình khám phá ở <a href={href('ch09', '9.4')}>9.4</a>
                  </td>
                }
              >
                <td>
                  a question and the explorer in <a href={href('ch09', '9.4')}>9.4</a>
                </td>
              </Translated>
            </tr>
            <tr>
              <td>5c</td>
              <td><Translated vi="Số nhiều nhất với hệ số gộp việc 4">The most with coarsening factor 4</Translated></td>
              <Translated
                vi={
                  <td>
                    một câu hỏi và máy tính số lần ghi ở <a href={href('ch09', '9.5')}>9.5</a>
                  </td>
                }
              >
                <td>
                  a question and the commit calculator in <a href={href('ch09', '9.5')}>9.5</a>
                </td>
              </Translated>
            </tr>
          </tbody>
        </table>
      </div>

      <h2><Translated vi="Những chỗ bản PDF có vẻ sai">Where the PDF looks wrong</Translated></h2>
      <Translated
        vi={
          <p>
            Bản sách mà khóa học này dựa vào có vài chỗ sai rõ ràng trong chương 9, phần lớn sót lại từ
            ví dụ văn bản của một ấn bản trước. Chúng được liệt kê để bạn không bị bối rối khi đọc sách
            song song. Ở chỗ nào lỗi ảnh hưởng tới mã, các lab có bản đã sửa và đã kiểm thử.
          </p>
        }
      >
        <p>
          The copy of the book this course was built from has a few apparent slips in chapter 9, most of them left over from
          an earlier edition's text example. They are listed so that you are not thrown by them when you read the book
          alongside. Where one changes code, the labs have a corrected, tested version.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Đây là cách đọc của chúng tôi với một bản PDF, được đối chiếu với tài liệu hiện hành của nhà cung cấp vào 2026-11-20
              (notes/today/ch09-facts.md). Nếu bản của bạn khác, hãy tin bản của bạn.
            </p>
          }
        >
          <p>
            These are our readings of one PDF, checked against current vendor documentation on 2026-11-20
            (notes/today/ch09-facts.md). If your copy differs, trust your copy.
          </p>
        </Translated>
      </Aside>
      <ul data-list="errata">
        <Translated
          vi={
            <li>
              p. 211 viết "In our text histogram example"; ví dụ của ấn bản này là một ảnh. p. 216 viết "the four threads
              access characters 8 through 15", trong khi Hình 9.13 và câu ngay trước nó có tám luồng và các điểm ảnh.
            </li>
          }
        >
          <li>
            p. 211 says "In our text histogram example"; this edition's example is an image. p. 216 says "the four threads
            access characters 8 through 15", where Fig. 9.13 and the sentence before it have eight threads and pixels.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 215 viết "In Fig. 9.11, each thread block is assigned to a contiguous block segment" gồm{' '}
              <code>COARSE_FACTOR × blockDim.x</code> phần tử. Câu đó mô tả mã của Hình 9.12; Hình 9.11 là
              hình vẽ, mà hệ số 3 của nó văn bản không bao giờ nêu ra.
            </li>
          }
        >
          <li>
            p. 215 says "In Fig. 9.11, each thread block is assigned to a contiguous block segment" of{' '}
            <code>COARSE_FACTOR × blockDim.x</code> elements. That describes the code of Fig. 9.12; Fig. 9.11 is the
            drawing, whose factor of 3 the text never states.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 213 đặt phép toán nguyên tử đầu tiên trên <code>bins_s</code> ở "line 14" của Hình 9.10. Tham chiếu được dựng ở các dòng
              13 và 14, còn <code>fetch_add</code> nằm ở dòng 15.
            </li>
          }
        >
          <li>
            p. 213 puts the first atomic on <code>bins_s</code> at "line 14" of Fig. 9.10. The reference is built on lines
            13 and 14 and <code>fetch_add</code> is on line 15.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Khai báo của <code>cuda::atomic_ref</code> (p. 207) và chữ ký của <code>fetch_add</code> (p. 208)
              được in mà không có giá trị mặc định: phạm vi system và <code>seq_cst</code>. Các listing luôn truyền cả hai, và mọi
              kernel ở đây cũng vậy (<a href={href('ch09', '9.2')}>9.2</a>).
            </li>
          }
        >
          <li>
            The declaration of <code>cuda::atomic_ref</code> (p. 207) and the signature of <code>fetch_add</code> (p. 208)
            are printed without their defaults: system scope and <code>seq_cst</code>. The listings always pass both, and so
            does every kernel here (<a href={href('ch09', '9.2')}>9.2</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Chữ ký ở p. 208 viết thứ tự là <code>cuda::std::memory_order</code>, còn các listing viết{' '}
              <code>cuda::memory_order_relaxed</code>. Không phải lỗi: cả hai cách viết đều hợp lệ.
            </li>
          }
        >
          <li>
            The signature on p. 208 spells the order <code>cuda::std::memory_order</code>, the listings{' '}
            <code>cuda::memory_order_relaxed</code>. Not an error: both spellings are valid.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Không listing nào xóa <code>bins</code> hay <code>bins_pool</code> của Hình 9.9 về không, và không có mã host hay cấu hình
              khởi chạy nào được trình bày. Trong các lab, bên gọi xóa cả hai, và các hàm khởi chạy là của khóa học này.
            </li>
          }
        >
          <li>
            No listing zeroes <code>bins</code> or Fig. 9.9's <code>bins_pool</code>, and no host code or launch
            configuration is shown. In the labs the caller zeroes both, and the launchers are ours.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Bài tập 4 cho "access latency" của bộ nhớ chia sẻ là 1 ns mà không nói một phép toán nguyên tử tốn một lần truy cập hay
              hai (một lần đọc và một lần ghi, như trong mô hình của §9.3). <a href={href('ch09', '9.3')}>Mục 9.3</a> nêu
              cách hiểu nó dùng và đưa ra cách còn lại.
            </li>
          }
        >
          <li>
            Exercise 4 gives a shared-memory "access latency" of 1 ns without saying whether one atomic costs one access or
            two (a read and a write, as in the model of §9.3). <a href={href('ch09', '9.3')}>Section 9.3</a> states the
            reading it uses and gives the other.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 213 gọi bộ nhớ chia sẻ là "private to each SM". Nó nằm trên SM, nhưng mỗi khối được cấp, và chỉ thấy, phần
              cấp phát của riêng nó. <a href={href('ch05', 'wrap')}>Phần tóm tắt của chương 5</a> chỉ ra cùng chỗ sai này ở p. 116 và trong bài tập 12.
            </li>
          }
        >
          <li>
            p. 213 calls shared memory "private to each SM". It sits on the SM, but each block is given, and sees, its own
            allocation. <a href={href('ch05', 'wrap')}>Chapter 5's summary</a> flags the same slip on p. 116 and in exercise 12.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Hình 9.15 khai báo <code>b</code> là <code>unsigned int</code> và <code>bNext</code> là{' '}
              <code>unsigned char</code>. Vô hại: cả hai đều giữ 0 đến 255, và phép so sánh nâng kiểu <code>bNext</code>.
            </li>
          }
        >
          <li>
            Fig. 9.15 declares <code>b</code> as <code>unsigned int</code> and <code>bNext</code> as{' '}
            <code>unsigned char</code>. Harmless: both hold 0 to 255, and the comparison promotes <code>bNext</code>.
          </li>
        </Translated>
      </ul>
      <Translated
        vi={
          <p>
            Một số phát biểu thì lỗi thời chứ không sai. DRAM 200 chu kỳ, "tens of cycles" (hàng chục chu kỳ) cho bộ nhớ đệm cấp cuối và "a
            few cycles" (vài chu kỳ) cho bộ nhớ chia sẻ là ví dụ của sách: ngày nay không nhà cung cấp nào công bố độ trễ theo chu kỳ. 1,024
            luồng mỗi khối vẫn đúng. Phạm vi thứ ba của sách, "across all GPU devices", là phạm vi system của CUDA, vốn
            bao gồm cả CPU. Và gợi ý đọc bốn điểm ảnh, một từ 32-bit, mỗi lần lặp (p. 216) không có trong tài liệu hiện hành nào của
            NVIDIA.
          </p>
        }
      >
        <p>
          Some statements are dated rather than wrong. The 200-cycle DRAM, "tens of cycles" for the last-level cache and "a
          few cycles" for shared memory are the book's example: no vendor publishes latencies in cycles today. 1,024
          threads per block still holds. The book's third scope, "across all GPU devices", is CUDA's system scope, which
          includes the CPU. And the hint to read four pixels, a 32-bit word, per iteration (p. 216) is in no current NVIDIA
          document.
        </p>
      </Translated>
      <Source pages="207–216" />

      <h2><Translated vi="Những gì khóa học này rút gọn">What this course shortened</Translated></h2>
      <Translated
        vi={
          <p>
            Một số phần của sách được bỏ đi có chủ đích, để bạn phân biệt được sự lược bỏ với lỗ hổng. Hai phép so sánh đời thường
            cho một cập nhật bị mất, sơ đồ ghế máy bay và các ki-ốt cửa hàng (pp. 204–205), được thay bằng trình chạy từng bước
            xen kẽ của <a href={href('ch09', '9.2')}>9.2</a>. Danh sách các ứng dụng của histogram (pp. 202–203) chỉ còn một câu
            ở <a href={href('ch09', '9.1')}>9.1</a>, và hình cái cây của Hình 9.1 được thay bằng hình của riêng chúng tôi.
            Phần nhập môn về lớp và template C++ ở chú thích 1 (p. 206) bị bỏ qua. Cách chia hai bản sao và bốn bản sao (p. 211)
            là các thiết lập của trình khám phá ở <a href={href('ch09', '9.4')}>9.4</a>, và bản sao theo cluster trong bộ nhớ chia sẻ
            phân tán (p. 213) được mô tả ở đó cùng ví dụ hiện hành của NVIDIA nhưng không được xây dựng: không trình giả lập nào mô hình hóa
            cluster, Metal không có, và CUDA cần compute capability 9.0. Gợi ý bốn điểm ảnh (p. 216) được nhắc đến, không
            được hiện thực. Một histogram riêng trọn vẹn mỗi luồng cho rất ít bin (p. 217) là một câu hỏi ở{' '}
            <a href={href('ch09', '9.6')}>9.6</a>. Hình 9.10 xuất hiện dưới dạng kernel của lab 9.2 ở hệ số gộp việc 1, và Hình
            9.12 là bản đối chứng được cung cấp của lab đó; <code>COARSE_FACTOR</code> lúc biên dịch của sách là một đối số lúc chạy
            trong các lab. Không có trong sách và cũng không làm ở đây: phạt lặp có xét dấu (sign-aware repetition penalty), thứ mà mục{' '}
            <a href={href('x1', 'X1.3')}>X1.3</a> bổ sung.
          </p>
        }
      >
        <p>
          Some of the book is left out on purpose, so that you can tell an omission from a gap. The two everyday analogies
          for a lost update, the airline seat map and the store kiosks (pp. 204–205), are replaced by the interleaving
          stepper of <a href={href('ch09', '9.2')}>9.2</a>. The list of uses for histograms (pp. 202–203) is one sentence
          in <a href={href('ch09', '9.1')}>9.1</a>, and the tree image of Fig. 9.1 is replaced by our own pictures.
          Footnote 1's primer on C++ classes and templates (p. 206) is skipped. The two-copy and four-copy schemes (p. 211)
          are settings of the explorer in <a href={href('ch09', '9.4')}>9.4</a>, and the per-cluster copy in distributed
          shared memory (p. 213) is described there with NVIDIA's current example but not built: neither emulator models
          clusters, Metal has none, and CUDA needs compute capability 9.0. The four-pixels hint (p. 216) is mentioned, not
          implemented. A full private histogram per thread for very few bins (p. 217) is one question in{' '}
          <a href={href('ch09', '9.6')}>9.6</a>. Fig. 9.10 appears as lab 9.2's kernel at coarsening factor 1, and Fig.
          9.12 as that lab's provided comparator; the book's compile-time <code>COARSE_FACTOR</code> is a run-time argument
          in the labs. Not in the book and not done here either: the sign-aware repetition penalty, which section{' '}
          <a href={href('x1', 'X1.3')}>X1.3</a> adds.
        </p>
      </Translated>

      <h2><Translated vi="Nơi bạn sẽ gặp nó trong mã thật">Where you will meet it in real code</Translated></h2>
      <Translated
        vi={
          <p>
            Mục 9.4 đã cho thấy <code>torch.histc</code> của PyTorch đưa ra lựa chọn chia-sẻ-hay-toàn-cục của sách. Thêm ba
            chỗ đáng đọc, mỗi chỗ mang hình dạng của chương này trong mã production:
          </p>
        }
      >
        <p>
          Section 9.4 showed PyTorch's <code>torch.histc</code> making the book's shared-or-global choice. Three more
          places worth reading, each this chapter's shape in production code:
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Không cái nào trong số này có trong sách. Đọc vào 2026-11-20 (notes/today/ch09-facts.md D3, D4); chúng là ảnh chụp tại một thời điểm, nên
              hãy theo tệp, đừng theo số dòng.
            </p>
          }
        >
          <p>
            None of these is in the book. Read on 2026-11-20 (notes/today/ch09-facts.md D3, D4); they are snapshots, so
            follow the files, not line numbers.
          </p>
        </Translated>
      </Beyond>
      <ul data-list="real-code">
        <Translated
          vi={
            <li>
              <code>DeviceHistogram</code> của CUB giữ tối đa 256 bin trong bộ nhớ chia sẻ; vượt quá đó, nó giữ một bản sao riêng
              mỗi khối trong bộ nhớ toàn cục, kích thước số khối × số bin: chính là <code>bins_pool</code> của Hình 9.9 trong sách.
            </li>
          }
        >
          <li>
            CUB's <code>DeviceHistogram</code> keeps up to 256 bins in shared memory; above that it keeps one private copy
            per block in global memory, sized blocks × bins: the book's <code>bins_pool</code> of Fig. 9.9.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Top-k trên Metal của llama.cpp (<code>kernel_top_k_f32_i32</code> trong <code>argsort.metal</code>) lấp đầy một histogram
              256 bin trong bộ nhớ threadgroup, mỗi lần một chữ số 8-bit của mẫu bit của mỗi logit, để tìm bucket
              chứa giá trị thứ k.
            </li>
          }
        >
          <li>
            llama.cpp's Metal top-k (<code>kernel_top_k_f32_i32</code> in <code>argsort.metal</code>) fills a 256-bin
            histogram in threadgroup memory, one 8-bit digit of each logit's bit pattern at a time, to find the bucket that
            holds the k-th value.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>count_equal</code> của llama.cpp là riêng tư hóa ở mức luồng đẩy tới mức một bin: mỗi luồng đếm số lần khớp
              trong một thanh ghi qua các phần tử nhảy bước của nó, <code>warp_reduce_sum</code> cộng gộp một warp, và một lane phát
              một <code>atomicAdd</code> duy nhất mỗi warp. Bản Metal cộng bằng <code>simd_sum</code>, kết hợp các tổng từng phần
              của threadgroup và phát một <code>atomic_fetch_add_explicit</code>.
            </li>
          }
        >
          <li>
            llama.cpp's <code>count_equal</code> is thread-level privatisation pushed to one bin: each thread counts matches
            in a register over its strided elements, <code>warp_reduce_sum</code> adds up a warp, and one lane issues a
            single <code>atomicAdd</code> per warp. The Metal version sums with <code>simd_sum</code>, combines the threadgroup's
            partial sums and issues one <code>atomic_fetch_add_explicit</code>.
          </li>
        </Translated>
      </ul>

      <h2><Translated vi="Dự án đến giờ">The project so far</Translated></h2>
      <Translated
        vi={
          <p>
            Lab 9.3 là phần đếm của mốc dự án M5, lấy mẫu trên GPU: số đếm token theo từng request trên{' '}
            <code>[batch × vocab]</code>, cập nhật mỗi bước decode một lần, và các hình phạt tần suất và hiện diện được áp
            lên logits.{' '}
            <span data-m5={m5.join(',') || 'none'}>
              {backends.length === 0
                ? 'Chưa có kết quả lab nào được ghi lại trên máy này.'
                : m5.length
                  ? `Lab 9.3 của chính bạn qua trên ${m5.map(where).join(' và ')} trên máy này.`
                  : 'Lab 9.3 của chính bạn chưa qua trên backend nào của máy này.'}
            </span>{' '}
            Xem các <a href={href('ch09', 'labs')}>lab</a>. Chương 11, 12 và 14 bổ sung các phần mà top-p và top-k cần, và
            module X1 lắp ráp M5 trong <a href={href('x1', 'labs')}>lab X1.3</a> của nó.
          </p>
        }
      >
        <p>
          Lab 9.3 is the counting piece of milestone M5, sampling on the GPU: per-request token counts over{' '}
          <code>[batch × vocab]</code>, updated one decode step at a time, and the frequency and presence penalties applied
          to the logits.{' '}
          <span data-m5={m5.join(',') || 'none'}>
            {backends.length === 0
              ? 'No lab results are recorded on this machine yet.'
              : m5.length
                ? `Your own lab 9.3 passes on ${m5.map(where).join(' and ')} on this machine.`
                : 'Your own lab 9.3 does not pass on any backend of this machine yet.'}
          </span>{' '}
          See the <a href={href('ch09', 'labs')}>labs</a>. Chapters 11, 12 and 14 add the pieces top-p and top-k need, and
          module X1 assembles M5 in its <a href={href('x1', 'labs')}>lab X1.3</a>.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chương 10 giữ hình dạng của chương này và thay đổi thứ được trộn. Phép rút gọn (reduction) kết hợp mọi thứ thành một giá trị:
            giá trị lớn nhất và hệ số chuẩn hóa của softmax, và argmax của greedy decoding. Phiên bản nhiều khối của nó cho mỗi
            khối hoàn tất riêng rồi ghi một lần bằng cùng <code>atomic_ref</code> và <code>fetch_add</code>, và
            sách mở đầu nó bằng cách gọi phép rút gọn, giống histogram, là một mẫu tóm tắt một tập dữ liệu lớn (p. 221). Bắt đầu ở{' '}
            <a href={href('ch10')}>chương 10</a>.
          </p>
        }
      >
        <p>
          Chapter 10 keeps this chapter's shape and changes what is merged. A reduction combines everything into one value:
          the softmax's maximum and its normaliser, and the argmax of greedy decoding. Its multi-block version lets each
          block finish privately and commit once with the same <code>atomic_ref</code> and <code>fetch_add</code>, and the
          book opens it by calling reduction, like histogram, a pattern that summarises a large data set (p. 221). Start at{' '}
          <a href={href('ch10')}>chapter 10</a>.
        </p>
      </Translated>
    </>
  );
}
