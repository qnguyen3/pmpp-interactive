import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Beyond, Source } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { functionSource } from '../../sim/toolchain';

// The kernel for the first question is a snippet of our own, loaded at run time, so there
// is no copy to drift: labs/ch12/snippets/wrap_pairs.cu. Wrap.snippets.test.ts runs it on
// the course's emulator (where it passes, which is the point of the question) and pins the
// numbers below.
export const PAIRS_FILE = 'ch12/snippets/wrap_pairs.cu';

/**
 * Global traffic per input key of exercise 1's three kernels carrying (column, logit)
 * pairs: kernel 1 reads the logit and writes its flag (8), kernel 2 reads the flag and
 * writes the offset (8, not the scan's own traffic), kernel 3 reads logit, flag and offset
 * (12) and writes 8 bytes per kept pair. The column is the thread's index, never read.
 */
export const threeKernelPairBytes = (f: number) => 8 + 8 + 12 + 8 * f;
/** The same filter that reads each logit once and writes each kept pair once (lab 12.4's walk). */
export const onePassPairBytes = (f: number) => 4 + 8 * f;
export const PAIR_FRACTION = 0.25;

export const questionsWrap: Record<string, Question> = {
  'ch12/wrap/q-pairs': {
    kind: 'choice',
    id: 'ch12/wrap/q-pairs',
    prompt: (
      <Translated
        vi={
          <>
            Kernel ở trên gom các ứng viên của một hàng logits thành các cặp (column, logit), không ổn định. Cả hai bộ đếm
            bắt đầu từ 0. Bài kiểm tra của nó xác nhận số đếm đúng và mọi <code>outVal[k]</code> là logit tại cột{' '}
            <code>outIdx[k]</code>. Bài kiểm tra đó có thể thất bại ở đâu?
          </>
        }
      >
        The kernel above collects the candidates of one row of logits as (column, logit) pairs, unstable. Both counters
        start at 0. Its test checks that the count is right and that every <code>outVal[k]</code> is the logit at column{' '}
        <code>outIdx[k]</code>. Where can that test fail?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không ở đâu cả: mỗi bộ đếm phát mỗi ô đúng một lần, nên mọi cột được giữ và mọi logit được giữ đều được ghi đúng một lần.">
            Nowhere: each counter hands out every slot exactly once, so every kept column and every kept logit is written exactly once.
          </Translated>
        ),
        why: (
          <Translated vi="Riêng từng mảng là một bộ lọc không ổn định đúng. Vấn đề nằm ở các cặp: ô k của mảng này và ô k của mảng kia đến từ hai bộ đếm khác nhau.">
            Each array on its own is a correct unstable filter. The pairs are the problem: slot k of one array and slot k of the other come from two different counters.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Ở bất cứ đâu mà một luồng khác có thể lấy một ô giữa hai atomic của một luồng: trên GPU hoặc trình giả lập race, không bao giờ trên emu.">
            Wherever another thread can take a slot between one thread's two atomics: on a GPU or the race emulator, never on emu.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Trên emu, với thứ tự luồng đảo ngược.">
            On emu, under the reversed thread order.
          </Translated>
        ),
        why: (
          <Translated vi="Đảo ngược thay đổi luồng nào nhận ô 0, nhưng emu chạy mỗi luồng tới hết rồi mới tới luồng kế, nên mọi luồng nhận cùng một ô từ cả hai bộ đếm.">
            Reversed changes which thread gets slot 0, but emu runs each thread to its end before the next, so every thread gets the same slot from both counters.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ trên AMD, nơi trình biên dịch biến mỗi atomic thành một atomic cho mỗi wavefront.">
            Only on AMD, where the compiler turns each atomic into one per wavefront.
          </Translated>
        ),
        why: (
          <Translated vi="Việc gộp xếp hạng các lane của một wavefront theo thứ tự lane ở cả hai bộ đếm, nên bên trong một wavefront chúng khớp nhau. Thứ có thể khác là thứ tự các warp tới mỗi bộ đếm, trên bất kỳ nhà cung cấp nào.">
            Aggregation ranks the lanes of a wavefront in lane order at both counters, so inside one wavefront they agree. What can differ is the order in which warps reach each counter, on any vendor.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Một atomic làm cho mỗi cập nhật trọn vẹn; nó không cho các cập nhật vào một địa chỉ thứ tự nào, và không cho các cập
              nhật vào hai địa chỉ thứ tự nào so với nhau. Luồng A có thể lấy ô 3 của <code>idxCount</code> và ô 5 của{' '}
              <code>valCount</code>, nên <code>outVal[3]</code> thuộc về một cột khác. Các số đếm vẫn đúng và mọi cột, mọi logit
              đều có mặt: chỉ việc ghép cặp bị hỏng, đó là lý do một bài kiểm tra so sánh riêng hai mảng, như các danh sách đã sắp
              xếp, sẽ vượt qua.
            </p>
            <p>
              Trình giả lập (emulator) của khóa học chạy một luồng cho tới một rào chắn, một lời gọi <code>_sync</code> của warp
              hoặc tới khi kết thúc, nên không có gì chạy giữa hai atomic của một luồng: bài kiểm tra đơn vị chạy kernel này ở đó
              với cả ba thứ tự luồng, và mọi cặp đều khớp. Cũng không công cụ nào chỉ ra nó: mọi truy cập đều là atomic hoặc vào một
              ô mà một luồng sở hữu, nên không có race condition (tranh chấp dữ liệu) nào để ThreadSanitizer báo. Cách sửa là mỗi
              cặp một ô: một <code>fetch_add</code>, rồi cả hai lần ghi tại ô đó.
            </p>
          </>
        }
      >
        <>
          <p>
            An atomic makes each update whole; it gives updates to one address no order, and updates to two addresses no
            order relative to each other. Thread A can take slot 3 of <code>idxCount</code> and slot 5 of{' '}
            <code>valCount</code>, so <code>outVal[3]</code> belongs to some other column. The counts stay right and every
            column and every logit is there: only the pairing breaks, which is why a test that compared the two arrays
            separately, as sorted lists, would pass.
          </p>
          <p>
            The course's emulator runs one thread until a barrier, a warp <code>_sync</code> call or its end, so nothing
            runs between a thread's two atomics: the unit test runs this kernel there in all three thread orders, and
            every pair matches. No tool names it either: every access is atomic or to a slot one thread owns, so there is
            no data race for ThreadSanitizer to report. The fix is one slot per pair: one <code>fetch_add</code>, then
            both stores at that slot.
          </p>
        </>
      </Translated>
    ),
    source: 'ours; atomics on one address in no fixed order: notes/today/ch12-facts.md (settled table, ch09 A3); the emulator\'s schedule: labs/README.md; run by Wrap.snippets.test.ts',
    vi: { source: 'của khóa học này; các atomic trên một địa chỉ không theo thứ tự cố định: notes/today/ch12-facts.md (bảng đã chốt, ch09 A3); lịch chạy của trình giả lập: labs/README.md; chạy bởi Wrap.snippets.test.ts' },
  },
  'ch12/wrap/q-unstable-inplace': {
    kind: 'choice',
    id: 'ch12/wrap/q-unstable-inplace',
    prompt: (
      <Translated
        vi={
          <>
            Kernel của Fig. 12.2, loại một atomic cho mỗi khóa được giữ, được gọi với <code>output == input</code>. Nó có
            nén gọn các khóa tại chỗ được không?
          </>
        }
      >
        Fig. 12.2's kernel, the one atomic per kept key, is called with <code>output == input</code>. Can it compact
        the keys in place?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Được: một khóa được giữ không bao giờ chuyển tới chỉ số cao hơn, giống như trong bộ lọc ổn định.">
            Yes: a kept key never moves to a higher index, just as in the stable filter.
          </Translated>
        ),
        why: (
          <Translated vi="Điều đó đúng với bộ lọc ổn định, nơi ô chỉ đếm các khóa được giữ đứng trước nó. Một bộ đếm phát ô theo thứ tự bất kỳ: luồng 0 có thể được phục vụ cuối cùng và nhận ô cao nhất.">
            That holds for the stable filter, whose slot counts only the kept keys before it. A counter hands slots out in any order: thread 0 can be served last and get the highest slot.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Được, nếu mỗi khối trước hết đọc các khóa của mình vào bộ nhớ chia sẻ và vượt qua một rào chắn trước khi lấy ô.">
            Yes, if every block first reads its keys into shared memory and passes a barrier before taking slots.
          </Translated>
        ),
        why: (
          <Translated vi="Điều đó sắp các lần đọc của một khối trước các lần ghi của chính nó. Các ô của nó vẫn có thể rơi vào tile của một khối sau, có thể chưa được đọc.">
            That orders one block's reads before its own writes. Its slots can still land in a later block's tile, which may not have been read yet.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không: một luồng có thể được phát một ô cao hơn chỉ số của chính nó, nơi một khóa chưa đọc có thể vẫn còn nằm đó.">
            No: a thread can be handed a slot above its own index, where an unread key may still live.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chỉ ở nơi trình biên dịch gộp atomic theo warp.">
            Only where the compiler aggregates the atomic per warp.
          </Translated>
        ),
        why: (
          <Translated vi="Việc gộp thay đổi số atomic tới bộ đếm, không thay đổi những ô nào được trả về: các warp vẫn tới bộ đếm theo thứ tự bất kỳ.">
            Aggregation changes how many atomics reach the counter, not which slots come back: the warps still reach the counter in any order.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Quy tắc của mục 12.7, không bao giờ ghi một ô trước khi khóa cũ của nó được đọc, dễ giữ trong một bộ lọc ổn định vì ô
            ≤ chỉ số: các khóa chỉ dịch về phía đầu, nên chỉ các lần đọc của khối trước là quan trọng. Một bộ lọc không ổn định
            không hứa giới hạn nào như vậy. Làm tại chỗ, nó sẽ cần mọi lần đọc trong lưới xong trước bất kỳ lần ghi nào, một rào
            chắn toàn lưới, thứ chỉ có với khởi chạy cooperative trên CUDA và HIP và hoàn toàn không có trên Metal. Một bộ lọc ổn
            định là cách rẻ hơn để nén gọn tại chỗ.
          </p>
        }
      >
        <p>
          Section 12.7's rule, never write a slot before its old key has been read, is easy to keep in a stable filter
          because slot ≤ index: keys only move toward the front, so only earlier blocks' reads matter. An unstable filter
          promises no such bound. In place it would need every read in the grid done before any write, a grid-wide
          barrier, which exists only under a cooperative launch on CUDA and HIP and not at all on Metal. A stable filter
          is the cheaper way to compact in place.
        </p>
      </Translated>
    ),
    source: 'ours, from ch. 12 §12.2, p. 291 (slots in arbitrary order) and §12.7, p. 300; grid barriers: ch12-facts settled table (ch04 A7, B11, C3)',
    vi: { source: 'của khóa học này, từ ch. 12 §12.2, p. 291 (ô theo thứ tự tùy ý) và §12.7, p. 300; rào chắn toàn lưới: bảng đã chốt của ch12-facts (ch04 A7, B11, C3)' },
  },
  'ch12/wrap/q-chain': {
    kind: 'choice',
    id: 'ch12/wrap/q-chain',
    prompt: (
      <Translated
        vi={
          <>
            Một bộ lấy mẫu lọc một hàng các cặp (token id, logit) ba lần: nó bỏ các token bị cấm bằng một bộ lọc ổn định, áp
            dụng min-p bằng một bộ lọc atomic không ổn định vì nhanh hơn, rồi giữ các cặp bằng hoặc trên ngưỡng top-k bằng
            một bộ lọc ổn định. Các ứng viên cuối cùng theo thứ tự nào?
          </>
        }
      >
        A sampler filters a row of (token id, logit) pairs three times: it drops banned tokens with a stable filter,
        applies min-p with an unstable atomic filter because it is faster, then keeps the pairs at or above a top-k
        threshold with a stable filter. In what order are the final candidates?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Thứ tự cột: bộ lọc cuối cùng ổn định.">
            Column order: the last filter is stable.
          </Translated>
        ),
        why: (
          <Translated vi="Một bộ lọc ổn định giữ thứ tự đầu vào của nó. Ở đây đầu vào của nó là bất kỳ thứ tự nào bộ lọc atomic tạo ra.">
            A stable filter keeps the order of its input. Its input here is whatever order the atomic filter produced.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Thứ tự cột, vì bộ lọc đầu tiên ổn định.">
            Column order, because the first filter was stable.
          </Translated>
        ),
        why: (
          <Translated vi="Bộ lọc không ổn định ở giữa đảo thứ tự những gì bộ lọc đầu đã giữ.">
            The unstable filter in the middle reorders what the first one kept.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Thứ tự logit giảm dần: min-p và top-k đều kiểm tra logit.">
            Descending logit order: min-p and top-k both test the logit.
          </Translated>
        ),
        why: (
          <Translated vi="Một bộ lọc chỉ loại bỏ; nó không bao giờ sắp xếp. Sắp theo logit là việc của chương 14.">
            A filter only removes; it never sorts. Ordering by logit is chapter 14's job.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Một thứ tự có thể thay đổi giữa các lần chạy.">An order that can change from run to run.</Translated>,
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tính ổn định là một lời hứa về đầu vào của chính bộ lọc, không phải về hàng ban đầu: một bước không ổn định ở bất kỳ
            đâu cũng làm thứ tự cuối cùng thành tùy ý. Cả ba phép kiểm tra ở đây đều xét từng cặp một, nên cùng một tập đó ra từ
            một bộ lọc ổn định duy nhất trên ba phép kiểm tra AND với nhau, theo thứ tự cột, với một lần đọc hàng thay vì ba. Một
            phép kiểm tra phụ thuộc vào thứ tự của danh sách, như khối lượng tích lũy của top-p, không thể gộp vào theo cách đó.
          </p>
        }
      >
        <p>
          Stability is a promise about a filter's own input, not about the original row: one unstable step anywhere makes
          the final order arbitrary. All three tests here look at one pair at a time, so the same set comes out of a
          single stable filter on the three tests ANDed together, in column order, with one read of the row instead of
          three. A test that depends on the order of the list, such as top-p's cumulative mass, could not be folded in
          that way.
        </p>
      </Translated>
    ),
    source: 'ours, from ch. 12 §12.1, p. 290 (stable and unstable) and the notes\' "Relevance to LLM-inference kernels" (combining criteria)',
    vi: { source: 'của khóa học này, từ ch. 12 §12.1, p. 290 (ổn định và không ổn định) và mục "Relevance to LLM-inference kernels" của ghi chú (kết hợp tiêu chí)' },
  },
  'ch12/wrap/q-rightalign': {
    kind: 'choice',
    id: 'ch12/wrap/q-rightalign',
    prompt: (
      <Translated
        vi={
          <>
            Bạn cần một bộ lọc ổn định dồn các khóa được giữ về <em>cuối</em> mảng, tại chỗ, với thiết kế một lượt của mục
            12.7. Chuỗi giữa các khối phải chạy theo chiều nào?
          </>
        }
      >
        You need a stable filter that packs the kept keys at the <em>end</em> of the array, in place, with the
        single-pass design of section 12.7. Which way must the chain between blocks run?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Từ khối cuối về khối đầu: mỗi khối chờ khối đứng sau nó, khối đã đọc các khóa của mình và biết có bao nhiêu khóa được giữ theo sau.">
            Last block to first: each block waits for the one after it, which has read its keys and knows how many kept keys follow.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Từ khối đầu tới khối cuối, như ở mục 12.7.">
            First block to last, as in section 12.7.
          </Translated>
        ),
        why: (
          <Translated vi="Điều đó sắp các lần ghi của mỗi khối sau các lần đọc của các khối trước. Ở đây các khóa dịch về phía cuối, nên một khối ghi đè lên ô của các khối sau.">
            That orders each block's writes after the earlier blocks' reads. Here keys move toward the end, so a block writes over slots of later blocks.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không cần chuỗi: các khóa dịch ra ngoài không bao giờ rơi vào một khóa chưa được đọc.">
            No chain: keys that move outward never land on a key that has not been read.
          </Translated>
        ),
        why: (
          <Translated vi="Chúng rơi vào khóa của các khối sau, có thể chưa được đọc.">
            They land on keys of later blocks, which may not have been read yet.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chiều nào cũng được, miễn là mỗi khối có một rào chắn giữa các lần đọc và các lần ghi của nó.">
            Either way, provided each block has a barrier between its reads and its writes.
          </Translated>
        ),
        why: (
          <Translated vi="Rào chắn sắp thứ tự đọc và ghi bên trong một khối. Các khóa còn vượt sang các khối khác, và chỉ chuỗi mới sắp thứ tự những lần đó.">
            The barrier orders reads and writes inside one block. Keys also cross into other blocks, and only the chain orders those.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Giờ ô của một khóa được xác định bởi các khóa được giữ <em>sau</em> nó, một phép scan từ cuối, và mọi khóa dịch về phía
            cuối. Các lần ghi của một khối rơi vào tile của chính nó hoặc của các khối sau, nên không khối nào được ghi trước khi
            mọi khối sau đã đọc. Đó là trường hợp thêm hàng và cột của mục 12.8: chuỗi một chiều chạy từ khối có các khóa cuối tới
            khối có các khóa đầu, và trên CUDA các số động của nó được phát theo đúng thứ tự đó.
          </p>
        }
      >
        <p>
          A key's slot is now set by the kept keys <em>after</em> it, a scan from the end, and every key moves toward the
          end. A block's writes land in its own tile or in later ones, so no block may write before every later block has
          read. That is section 12.8's case of adding rows and columns: the one-directional chain runs from the block with
          the last keys to the block with the first, and on CUDA its dynamic numbers are handed out in that order.
        </p>
      </Translated>
    ),
    source: 'ours, from ch. 12 §12.7, p. 300 and §12.8, p. 301 (the reversed constraint for keys that move outward)',
    vi: { source: 'của khóa học này, từ ch. 12 §12.7, p. 300 và §12.8, p. 301 (ràng buộc đảo ngược cho các khóa dịch ra ngoài)' },
  },
  'ch12/wrap/q-pairbytes': {
    kind: 'numeric',
    id: 'ch12/wrap/q-pairbytes',
    prompt: (
      <Translated
        vi={
          <>
            Lại ba kernel của bài tập 1, giờ mang theo các cặp: mỗi khóa là một logit 4 byte, mỗi ứng viên được giữ được ghi
            thành một cặp (column, logit) 8 byte, và cột là chỉ số của luồng, không bao giờ được đọc. Cờ và độ lệch là 4
            byte. Khi một phần tư số logit được giữ, có bao nhiêu byte lưu lượng bộ nhớ toàn cục cho mỗi logit đầu vào, không
            tính lưu lượng riêng của phép scan?
          </>
        }
      >
        Exercise 1's three kernels again, now carrying pairs: each key is a 4-byte logit, each kept candidate is written
        as a (column, logit) pair of 8 bytes, and the column is the thread's index, never read. Flags and offsets are 4
        bytes. With a quarter of the logits kept, how many bytes of global traffic per input logit, not counting the
        scan's own traffic?
      </Translated>
    ),
    answer: threeKernelPairBytes(PAIR_FRACTION),
    unit: 'bytes',
    explain: (
      <Translated
        vi={
          <p>
            Kernel 1 đọc logit và ghi cờ, 8; kernel 2 đọc cờ và ghi độ lệch, 8; kernel 3 đọc logit, cờ và độ lệch, 12, và ghi 8 ×
            0.25 = 2 cho mỗi đầu vào. 8 + 8 + 12 + 2 ={' '}
            {threeKernelPairBytes(PAIR_FRACTION)}. Một kernel đọc mỗi logit một lần và ghi mỗi cặp được giữ một lần di chuyển 4 + 2
            = {onePassPairBytes(PAIR_FRACTION)}, bằng một phần năm: đây là lý do khối của lab 12.4 đi qua hàng của nó một lần và
            không bao giờ ghi cờ hay độ lệch ra bộ nhớ.
          </p>
        }
      >
        <p>
          Kernel 1 reads the logit and writes the flag, 8; kernel 2 reads the flag and writes the offset, 8; kernel 3 reads
          the logit, the flag and the offset, 12, and writes 8 × 0.25 = 2 per input. 8 + 8 + 12 + 2 ={' '}
          {threeKernelPairBytes(PAIR_FRACTION)}. A kernel that reads each logit once and writes each kept pair once moves 4
          + 2 = {onePassPairBytes(PAIR_FRACTION)}, a fifth as much: this is why lab 12.4's block walks its row once and
          never writes a flag or an offset to memory.
        </p>
      </Translated>
    ),
    source: 'ours, with the accounting of section 12.5 (exercise 1, p. 302); the pairs are lab 12.4\'s',
    vi: { unit: 'byte', source: 'của khóa học này, với cách tính của mục 12.5 (bài tập 1, p. 302); các cặp là của lab 12.4' },
  },
};

/** The kernel of the first question, from the snippet file. */
function PairsKernel() {
  const file = useLabFile(PAIRS_FILE);
  const fn = file && !file.startsWith('// could not load') ? functionSource(file, 'candidatePairs') : null;
  return (
    <Code
      tabs={[
        {
          label: tr('CUDA (not a lab file: for this question only)', 'CUDA (không phải tệp lab: chỉ cho câu hỏi này)'),
          code: fn ? fn.text : file ?? `// loading labs/${PAIRS_FILE}`,
          status: tr('front-end-checked; wrong on purpose', 'front-end-checked; cố ý sai'),
          tiers: { logits: 'global', outIdx: 'global', outVal: 'global', idxCount: 'global', valCount: 'global', v: 'reg', c: 'reg' },
        },
      ]}
      caption={
        <Translated vi={<>Từ <code>labs/{PAIRS_FILE}</code>. Các bài kiểm tra đơn vị chạy nó trên trình giả lập của khóa học.</>}>
          From <code>labs/{PAIRS_FILE}</code>. The unit tests run it on the course's emulator.
        </Translated>
      }
    />
  );
}

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const m5 = backends.filter((b) => labStatus(reports[b], 'ch12/lab4_candidates', 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch12/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}><Translated vi="Những gì cần mang theo">What to carry forward</Translated></h2>
      <ul>
        <li>
          <Translated
            vi={
              <>
                Một bộ lọc trả lời một câu hỏi cho mỗi khóa được giữ: ô đầu ra nào là của tôi? Một bộ lọc không ổn định lấy ô trống
                kế tiếp từ một bộ đếm, bằng một atomic. Một bộ lọc ổn định đếm các khóa được giữ đứng trước nó, một scan loại trừ của
                các cờ 0/1. Hãy trả giá cho tính ổn định khi thứ tự đầu vào có ý nghĩa: một danh sách đã sắp xếp, một quy tắc phá
                hòa, một lần rút có seed.
              </>
            }
          >
            A filter answers one question per kept key: which output slot is mine? An unstable filter takes the next free
            slot from a counter, with an atomic. A stable one counts the kept keys before it, an exclusive scan of 0/1
            flags. Pay for stability when the input order means something: a sorted list, a tie rule, a seeded draw.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Trên một bộ đếm, tranh chấp là chi phí, nên hãy chạm vào nó ít hơn: mỗi warp một lần bằng một lần bỏ phiếu (một
                trưởng nhóm, một popcount, một lần phát tán, và một tổng tiền tố (prefix sum) nhị phân cho hạng của mỗi lane), mỗi
                khối một lần bằng một danh sách riêng trong bộ nhớ chia sẻ (shared memory). Cả hai cũng đặt các khóa kề nhau vào các ô
                kề nhau, nên các lần ghi được hợp nhất.
              </>
            }
          >
            On one counter, contention is the cost, so touch it less: once per warp with a vote (a leader, a popcount, a
            broadcast, and a binary prefix sum for each lane's rank), once per block with a private list in shared memory.
            Both also put neighbouring keys in neighbouring slots, so the stores coalesce.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Mã warp chuyển được giữa các nền tảng: một ballot mà mọi lane đều thực hiện trong luồng điều khiển đồng nhất, không
                phải <code>__activemask()</code> bên trong một nhánh; một mặt nạ có kiểu riêng, 64 bit trên HIP, được đếm bằng popcount
                của kiểu đó; độ rộng được đọc lúc chạy. Trên Metal, <code>simd_sum</code> và{' '}
                <code>simd_prefix_exclusive_sum</code> là số đếm và hạng.
              </>
            }
          >
            Warp code that ports: a ballot that every lane takes in uniform control flow, not{' '}
            <code>__activemask()</code> inside a branch; a mask with its own type, 64 bits on HIP, counted with the
            popcount of that type; the width read at run time. On Metal, <code>simd_sum</code> and{' '}
            <code>simd_prefix_exclusive_sum</code> are the count and the rank.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Một bộ lọc ổn định là giữ, scan loại trừ, rải. Gom tạm các khóa còn lại của mỗi khối trong bộ nhớ chia sẻ ghi ra các
                dải liền; gộp việc làm các dải dài hơn và phép scan rẻ hơn. Đòn bẩy lớn hơn là số lượt đi qua bộ nhớ: khi giữ một nửa,
                ba kernel của bài tập 1 di chuyển 30 byte mỗi khóa, thiết kế đếm-rồi-rải 10, một lượt 6.
              </>
            }
          >
            A stable filter is keep, exclusive scan, scatter. Staging each block's survivors in shared memory writes
            contiguous runs; coarsening makes the runs longer and the scan cheaper. The bigger lever is passes over memory:
            at half kept, exercise 1's three kernels move 30 bytes per key, a count-then-scatter design 10, one pass 6.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Khi làm tại chỗ, các khóa chỉ dịch về phía đầu, nên không khối nào được ghi trước khi mọi khối trước đã đọc. Chuỗi của
                scan một lượt chỉ đem lại điều đó với một release, một acquire và một rào chắn sau các lần đọc, và chỉ ở nơi việc chờ
                một khối trước được ghi trong tài liệu: CUDA, với số khối lấy từ một bộ đếm. Trên HIP và Metal, hãy nén gọn tại chỗ
                bằng một khối (threadgroup) đi qua vùng của nó theo thứ tự, chỉ dùng rào chắn; nhiều lần dispatch là thiết kế được tài
                liệu bảo đảm an toàn khi không làm tại chỗ.
              </>
            }
          >
            In place, keys only move toward the front, so no block may write before every earlier block has read. The
            single-pass scan's chain gives that only with a release, an acquire and a barrier after the reads, and only
            where waiting on an earlier block is documented: CUDA, with block numbers from a counter. On HIP and Metal,
            compact in place with one block (threadgroup) that walks its range in order, with barriers only; several
            dispatches are the documented-safe design out of place.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Khi chỉ số quyết định vị trí của một khóa, không cần scan, và chuỗi tại chỗ chạy theo chiều các khóa di chuyển. Bộ lọc
                của các thư viện đều ổn định; các framework không thể trả về một shape phụ thuộc vào dữ liệu thì dùng mặt nạ thay vì
                nén gọn, và một kernel ghi ra một danh sách và một số đếm là thứ cho phép kernel kế tiếp bỏ qua những phần tử bị loại.
              </>
            }
          >
            When the index decides a key's position, no scan is needed, and the in-place chain runs the way the keys move.
            Libraries' filters are stable; frameworks that cannot return a shape that depends on the data mask instead of
            compacting, and a kernel that writes a list and a count is what lets the next kernel skip the losers.
          </Translated>
        </li>
      </ul>
      <Source section="12.9" pages="302" />

      <h2><Translated vi="Trong mã bạn sẽ đọc">In code you will read</Translated></h2>
      <Translated vi={<p>Cùng những lựa chọn đó xuất hiện trong mã GPU của các thư viện bạn đã gọi. Năm chỗ để xem.</p>}>
        <p>The same choices show in the GPU code of libraries you already call. Five places to look.</p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Không có trong sách. Đọc trong mã nguồn và tài liệu của PyTorch và llama.cpp ngày 2026-11-20
              (notes/today/ch12-facts.md D1, D2, D4); chỗ nào ghi "theo cách đọc của chúng tôi", đó là của khóa học này.
            </p>
          }
        >
          <p>
            Not in the book. Read in PyTorch's and llama.cpp's source and documentation on 2026-11-20
            (notes/today/ch12-facts.md D1, D2, D4); where a line says "by our reading", it is ours.
          </p>
        </Translated>
      </Beyond>
      <ul data-list="real-code">
        <li>
          <Translated
            vi={
              <>
                <b>Ổn định vì có bên gọi dựa vào nó.</b> <code>torch.nonzero</code> của PyTorch hứa rằng kết quả của nó "is sorted
                lexicographically, with the last index changing the fastest (C-style)" (được sắp theo thứ tự từ điển, chỉ số cuối thay
                đổi nhanh nhất). Trên CUDA nó được xây trên <code>DeviceSelect::Flagged</code> của CUB, một bộ lọc ổn định.
              </>
            }
          >
            <b>Stable because a caller relies on it.</b> PyTorch's <code>torch.nonzero</code> promises that its result
            "is sorted lexicographically, with the last index changing the fastest (C-style)". On CUDA it is built on
            CUB's <code>DeviceSelect::Flagged</code>, a stable filter.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Chọn thứ tự thay vì atomic.</b> Bước gather CUDA của <code>torch.topk</code> đếm các phần tử khớp của mỗi warp bằng
                một ballot, và mã nguồn của nó nói một cách giành chỗ dựa trên atomicAdd "would make the order depend on warp
                scheduling" (sẽ làm thứ tự phụ thuộc vào việc lập lịch warp). Đó là mục 12.2 đấu với mục 12.5, được quyết định vì tính
                tất định.
              </>
            }
          >
            <b>Order chosen over atomics.</b> <code>torch.topk</code>'s CUDA gather counts each warp's matches with a
            ballot, and its source says an atomicAdd-based reservation "would make the order depend on warp
            scheduling". That is section 12.2 against section 12.5, decided for determinism.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Dạng ballot của mục 12.3.</b> Top-k CUDA của llama.cpp đặt các khóa hòa nhau tại ngưỡng bằng một ballot và một
                popcount có mặt nạ trong luồng điều khiển đồng nhất, theo thứ tự cột; <code>topk</code> của PyTorch cũng đếm bằng một
                ballot.
              </>
            }
          >
            <b>The ballot form of section 12.3.</b> llama.cpp's CUDA top-k places the keys tied at its threshold with a
            ballot and a masked popcount in uniform control flow, in column order; PyTorch's <code>topk</code> counts
            with a ballot too.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Gom tạm để ghi được hợp nhất.</b> <code>nonzero_static</code> của PyTorch trên CUDA là bài tập 1 có gộp việc, và
                trước khi rải nó sắp xếp lại các khóa bằng một phép trao đổi trong khối "for a slightly better coalescing for writes"
                (để các lần ghi được hợp nhất tốt hơn một chút): ý tưởng của mục 12.6 trong mã thư viện.
              </>
            }
          >
            <b>Staging for coalesced writes.</b> PyTorch's <code>nonzero_static</code> on CUDA is exercise 1 with
            coarsening, and before it scatters it rearranges its keys with a block exchange "for a slightly better
            coalescing for writes": section 12.6's idea in library code.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Tất định không có nghĩa là ổn định.</b> Top-k Metal của llama.cpp nén gọn mỗi hàng trong một threadgroup bằng các
                tổng và tổng tiền tố của SIMD group, theo cách không phụ thuộc vào lập lịch. Tuy vậy, theo cách đọc của chúng tôi,
                mỗi lane sở hữu một luồng con có bước nhảy (stride) và các lane được xếp hạng trước các phần tử của chúng, nên các khóa
                trên ngưỡng ra theo thứ tự ưu tiên lane, không phải thứ tự cột: giống nhau mọi lần chạy, mà vẫn không ổn định. Điều
                làm một phép nén gọn ổn định là xếp hạng mỗi khóa được giữ theo số khóa được giữ đứng trước nó theo thứ tự đầu vào, như
                Figs. 12.8 đến 12.10 đều làm; nhánh chính của llama.cpp lại xếp hạng lane trước phần tử. (Chú thích của nó nói nó xếp
                hạng "by column order"; nhánh xử lý hòa của nó thì đúng như vậy.)
              </>
            }
          >
            <b>Deterministic is not stable.</b> llama.cpp's Metal top-k compacts each row in one threadgroup with
            SIMD-group sums and prefix sums, in a way that does not depend on scheduling. By our reading, though, each
            lane owns a strided substream and lanes are ranked before their elements, so the keys above the threshold
            come out in lane-major order, not column order: the same every run, and still not stable. What makes a
            compaction stable is ranking every kept key by the number of kept keys before it in input order, as Figs.
            12.8 to 12.10 all do; llama.cpp's main path ranks lanes before elements instead. (Its comment says it ranks
            "by column order"; its tie path does.)
          </Translated>
        </li>
      </ul>

      <h2><Translated vi="Bài kiểm tra tổng hợp">A mixed check</Translated></h2>
      <Translated
        vi={
          <p>
            Những câu này dựa trên cả chương và trên những tình huống mà các mục chưa trình bày.
            {asked ? ` Đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu, tính theo câu trả lời gần nhất của bạn cho mỗi câu.` : ''}
          </p>
        }
      >
        <p>
          These draw on the whole chapter and on situations the sections did not show.
          {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Kernel và các tình huống bên dưới là của khóa học này.</p>}>
          <p>The kernel and the situations below are ours.</p>
        </Translated>
      </Beyond>
      <PairsKernel />
      <Ask q={q['ch12/wrap/q-pairs']} label={tr('Find the defect', 'Tìm lỗi')} />
      <Ask q={q['ch12/wrap/q-unstable-inplace']} />
      <Ask q={q['ch12/wrap/q-chain']} />
      <Ask q={q['ch12/wrap/q-rightalign']} />
      <Ask q={q['ch12/wrap/q-pairbytes']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Bài tập của sách">The book's exercises</Translated></h2>
      <Translated
        vi={
          <p>
            Chương kết thúc với bốn bài tập (p. 302), tất cả về bộ lọc ổn định. Mỗi bài được giải ở đâu đó trong khóa học này;
            sách không in lời giải, nên lời giải ở đây là của khóa học này.
          </p>
        }
      >
        <p>
          The chapter ends with four exercises (p. 302), all about the stable filter. Each is worked somewhere in this
          course; the book prints no answers, so the answers here are ours.
        </p>
      </Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }} data-table="exercises">
          <thead>
            <tr>
              <th><Translated vi="Bài tập">Exercise</Translated></th>
              <th><Translated vi="Nội dung">About</Translated></th>
              <th><Translated vi="Được giải ở đâu">Where it is worked here</Translated></th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1</td>
              <td><Translated vi="Bộ lọc ổn định dưới dạng ba kernel: cờ giữ lại, scan loại trừ, đầu ra">The stable filter as three kernels: keep flags, exclusive scan, output</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      trình khám phá giữ-scan-rải và đoạn mã ở <a href={href('ch12', '12.5')}>12.5</a>, và phần A của{' '}
                      <a href={href('ch12', 'labs')}>lab 12.3</a>
                    </>
                  }
                >
                  the keep-scan-scatter explorer and the listing in <a href={href('ch12', '12.5')}>12.5</a>, and part A of{' '}
                  <a href={href('ch12', 'labs')}>lab 12.3</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td><Translated vi="Bộ lọc ổn định dưới dạng một kernel trên scan một lượt, và các ưu điểm của nó">The stable filter as one kernel on a single-pass scan, and its advantages</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      phép đếm byte ở <a href={href('ch12', '12.5')}>12.5</a>, câu hỏi về ưu điểm ở{' '}
                      <a href={href('ch12', '12.7')}>12.7</a>, và <a href={href('ch12', 'labs')}>lab 12.5</a> (tùy chọn, CUDA và HIP)
                    </>
                  }
                >
                  the byte count in <a href={href('ch12', '12.5')}>12.5</a>, the advantages question in{' '}
                  <a href={href('ch12', '12.7')}>12.7</a>, and <a href={href('ch12', 'labs')}>lab 12.5</a> (optional, CUDA
                  and HIP)
                </Translated>
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td><Translated vi="Riêng tư hóa kernel của bài tập 2">Privatisation of exercise 2's kernel</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      trình khám phá gom tạm và hoạt cảnh ở <a href={href('ch12', '12.6')}>12.6</a>; phần B của{' '}
                      <a href={href('ch12', 'labs')}>lab 12.3</a> với nhiều kernel, lab 12.5 với một kernel
                    </>
                  }
                >
                  the staging explorer and scene in <a href={href('ch12', '12.6')}>12.6</a>; part B of{' '}
                  <a href={href('ch12', 'labs')}>lab 12.3</a> in several kernels, lab 12.5 in one
                </Translated>
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td><Translated vi="Gộp việc cho luồng với kernel của bài tập 3">Thread coarsening of exercise 3's kernel</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      các câu hỏi về gộp việc ở <a href={href('ch12', '12.6')}>12.6</a>; hệ số coarse của phần B{' '}
                      <a href={href('ch12', 'labs')}>lab 12.3</a> và của lab 12.5
                    </>
                  }
                >
                  the coarsening questions in <a href={href('ch12', '12.6')}>12.6</a>; the coarse factor of{' '}
                  <a href={href('ch12', 'labs')}>lab 12.3</a> part B and of lab 12.5
                </Translated>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2><Translated vi="Chỗ bản PDF có vẻ sai">Where the PDF looks wrong</Translated></h2>
      <Translated
        vi={
          <p>
            Bản sách mà khóa học này được xây dựng từ đó có một số chỗ có vẻ nhầm lẫn trong chương 12, và vài đoạn mã mà tài liệu
            hiện nay hiểu khác. Chúng được liệt kê để bạn không bị bối rối khi đọc sách song song. Chỗ nào làm thay đổi mã, các lab
            và đoạn mã của khóa học có một bản đã sửa và đã kiểm thử.
          </p>
        }
      >
        <p>
          The copy of the book this course was built from has a number of apparent slips in chapter 12, and a few
          listings that current documentation reads differently. They are listed so that you are not thrown by them when
          you read the book alongside. Where one changes code, the course's labs and snippets have a corrected, tested
          version.
        </p>
      </Translated>
      <ul data-list="errata">
        <li>
          <Translated vi={<>Fig. 12.1 đặt chú thích "(a)" cho cả hai khung; khung thứ hai, bộ lọc ổn định, lẽ ra phải là (b).</>}>
            Fig. 12.1 captions both panels "(a)"; the second, the stable filter, should be (b).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 292 dẫn "Chapter 5" cho cách phần cứng hợp nhất các lần đọc và ghi của một warp. Ấn bản này dạy điều đó ở chương 6,{' '}
                <a href={href('ch06', '6.1')}>mục 6.1</a>.
              </>
            }
          >
            p. 292 cites "Chapter 5" for the way the hardware coalesces a warp's loads and stores. This edition teaches it
            in chapter 6, <a href={href('ch06', '6.1')}>section 6.1</a>.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Figs. 12.3 và 12.4 khai báo <code>unsigned int j;</code> ở dòng 08 và chỉ gán giá trị cho nó trong trưởng nhóm, vậy mà
                mọi lane đều truyền nó vào lệnh shuffle. Đọc một biến chưa khởi tạo là hành vi không xác định trong C++, và bản build
                trình giả lập của khóa học đã tận dụng quyền đó: mọi lane vượt qua đều chạy nhánh của trưởng nhóm. Các đoạn mã và lab
                viết <code>j = 0</code> (<a href={href('ch12', '12.3')}>12.3</a>).
              </>
            }
          >
            Figs. 12.3 and 12.4 declare <code>unsigned int j;</code> on line 08 and give it a value only in the leader, yet
            every lane passes it to the shuffle. Reading an uninitialised variable is undefined in C++, and the course's
            emulator build took that licence: every passing lane ran the leader's branch. The snippets and labs write{' '}
            <code>j = 0</code> (<a href={href('ch12', '12.3')}>12.3</a>).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Fig. 12.3 coi <code>__activemask()</code> bên trong nhánh là tập các lane đã vượt qua. Hướng dẫn của NVIDIA giờ nói
                không thể dùng nó theo cách đó: kernel vẫn đúng, nhưng "một atomic cho mỗi warp" không được bảo đảm. Đoạn mã cũng chỉ
                dành cho CUDA: HIP từ chối mặt nạ shuffle 32 bit của nó, và chỉ mở rộng mặt nạ đó thì vẫn sai trên wavefront 64 lane
                (<a href={href('ch12', '12.3')}>12.3</a>).
              </>
            }
          >
            Fig. 12.3 treats <code>__activemask()</code> inside the branch as the set of lanes that passed. NVIDIA's guide
            now says it cannot be used that way: the kernel stays correct, but "one atomic per warp" is not guaranteed.
            The listing is also CUDA-only: HIP rejects its 32-bit shuffle mask, and widening only that leaves it wrong on
            64-lane wavefronts (<a href={href('ch12', '12.3')}>12.3</a>).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Fig. 12.4 không cho thấy header hay namespace nào cho <code>coalesced_group</code>, và <code>size()</code> của nó giờ
                là một bí danh kiểu cũ của <code>num_threads()</code>.
              </>
            }
          >
            Fig. 12.4 shows no header and no namespace for <code>coalesced_group</code>, and its <code>size()</code> is
            now a legacy alias of <code>num_threads()</code>.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 294 nói trình biên dịch đã tự gộp các atomic, và không làm vậy khi chỉ có một luồng hoạt động, mà không nêu tên
                trình biên dịch hay phiên bản nào. NVIDIA chỉ nói điều thứ nhất trong một bài blog cập nhật lần cuối năm 2017; điều thứ
                hai không có trong nguồn nào của NVIDIA được tìm thấy.
              </>
            }
          >
            p. 294 says the compiler already coalesces atomics, and refrains when only one thread is active, without
            naming a compiler or version. NVIDIA says the first only in a blog post last updated in 2017; the second is in
            no NVIDIA source found.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 295 in <code>__ballot_sync</code> không có dấu ngoặc và không cho tham số nào cho cả ba lệnh bỏ phiếu mà nó nêu.
              </>
            }
          >
            p. 295 prints <code>__ballot_sync</code> without parentheses and gives no parameters for any of the three
            votes it names.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Fig. 12.6: lỗi đánh máy "intialize" trong chú thích; biến chia sẻ <code>j</code> ở dòng 26 dùng lại tên của biến cục
                bộ <code>j</code> ở dòng 19 và phá quy ước <code>_s</code> (lab 12.1 gọi nó là <code>j_s</code>); và luồng 0 thực hiện
                atomic công khai của nó kể cả khi khối không giữ gì. Bộ đếm đầu ra chỉ được nói là đặt về 0 trước khi khởi chạy cho
                Fig. 12.2 (p. 291); Figs. 12.3, 12.4 và 12.6 cũng cần điều đó.
              </>
            }
          >
            Fig. 12.6: the comment typo "intialize"; the shared <code>j</code> of line 26 reuses the name of the local{' '}
            <code>j</code> of line 19 and breaks the <code>_s</code> convention (lab 12.1 calls it <code>j_s</code>); and
            thread 0 makes its public atomic even when the block kept nothing. The output counter is said to be zeroed
            before the launch only for Fig. 12.2 (p. 291); Figs. 12.3, 12.4 and 12.6 need it too.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Fig. 12.5 ghi các khối theo thứ tự xanh lá, cam, vàng, xanh dương: một thứ tự tùy ý trong nhiều thứ tự, điều mà văn
                bản không giải thích.
              </>
            }
          >
            Fig. 12.5 commits the blocks in the order green, orange, yellow, blue: one arbitrary order of many, which the
            text does not explain.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Fig. 12.8 gọi <code>gridExclusiveScan</code>, hàm không bao giờ được định nghĩa, và đọc <code>input[i]</code> mà không
                kiểm tra <code>i &lt; N</code>.
              </>
            }
          >
            Fig. 12.8 calls <code>gridExclusiveScan</code>, which is never defined, and reads <code>input[i]</code> with
            no <code>i &lt; N</code> check.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 300 nói luồng 1 phải xong việc "reading k0" trước khi luồng 3 ghi k3; theo chính logic của câu đó thì phải là k1. Ví
                dụ dẫn Fig. 12.10, nơi bốn luồng xử lý tám khóa, nên số thứ tự luồng của nó không khớp với khóa
                (<a href={href('ch12', '12.7')}>12.7</a>).
              </>
            }
          >
            p. 300 says thread 1 must finish "reading k0" before thread 3 writes k3; by the sentence's own logic it is k1.
            The example cites Fig. 12.10, where four threads handle eight keys, so its thread numbers do not map onto keys
            (<a href={href('ch12', '12.7')}>12.7</a>).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 301: "earlier thread blocks write to memory locations previously occupied by keys read by earlier thread blocks";
                chữ "earlier" thứ hai lẽ ra phải là "later", như câu kế tiếp xác nhận. Trang này còn có "in the section" và "more
                complex data in-place data movement patterns".
              </>
            }
          >
            p. 301: "earlier thread blocks write to memory locations previously occupied by keys read by earlier thread
            blocks"; the second "earlier" should be "later", as the next sentence confirms. The page also has "in the
            section" and "more complex data in-place data movement patterns".
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Tài liệu tham khảo [4], đường dẫn CUB ở p. 302, trả về 404; các trang đó giờ nằm dưới{' '}
                <code>nvidia.github.io/cccl/unstable/</code>.
              </>
            }
          >
            Reference [4], the CUB link on p. 302, returns 404; the pages now live under{' '}
            <code>nvidia.github.io/cccl/unstable/</code>.
          </Translated>
        </li>
      </ul>
      <Aside>
        <Translated
          vi={
            <p>
              Đây là cách đọc của chúng tôi trên một bản PDF, đã đối chiếu với tài liệu hiện hành của các nhà cung cấp ngày
              2026-11-20 (notes/today/ch12-facts.md A1, A5, A7, A8, B1). Nếu bản của bạn khác, hãy tin bản của bạn.
            </p>
          }
        >
          <p>
            These are our readings of one PDF, checked against current vendor documentation on 2026-11-20
            (notes/today/ch12-facts.md A1, A5, A7, A8, B1). If your copy differs, trust your copy.
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Một phần của sách được cố ý bỏ qua, để bạn phân biệt được một sự lược bỏ với một lỗ hổng. Việc nén gọn heap ở §12.1
            (pp. 289–290) là động lực cho chương, nhưng không lab nào xây một bộ cấp phát. Chuyển vị tại chỗ ma trận và tensor
            (§12.8, p. 301, tài liệu [2]) được nhắc tên, không được đề cập. Kernel cooperative groups của Fig. 12.4 được trình bày
            và chạy trên trình giả lập, không được xây trong lab: Metal không có cooperative groups, và lần bỏ phiếu của lab 12.1
            bao quát cùng các bước trên mọi backend. Phép scan không được định nghĩa của Fig. 12.8 không được tái hiện; lab 12.3
            và 12.5 hiện thực nó theo hai cách. Bài tập 2 đến 4 dưới dạng một kernel chỉ được làm đầy đủ trong lab 12.5 tùy chọn.
            Thrust và CUB, cùng các thư viện tương ứng trên AMD và Apple, được mô tả kèm các bảo đảm của chúng, không được gọi.
            Tài liệu [1] và [2] được trích dẫn nhưng chưa được kiểm tra.
          </p>
        }
      >
        <p>
          Some of the book is left out on purpose, so that you can tell an omission from a gap. The heap compaction of
          §12.1 (pp. 289–290) motivates the chapter, but no lab builds an allocator. In-place transposition of matrices and
          tensors (§12.8, p. 301, reference [2]) is named, not covered. Fig. 12.4's cooperative-groups kernel is shown and
          run on the emulator, not built in a lab: Metal has no cooperative groups, and lab 12.1's vote covers the same
          steps on every backend. Fig. 12.8's undefined scan is not reproduced; labs 12.3 and 12.5 implement it two ways.
          Exercises 2 to 4 as one kernel are done in full only in the optional lab 12.5. Thrust and CUB, and their
          counterparts on AMD and Apple, are described with their guarantees, not called. References [1] and [2] are cited
          but not checked.
        </p>
      </Translated>

      <h2><Translated vi="Dự án đến lúc này">The project so far</Translated></h2>
      <Translated
        vi={
          <p>
            Phần nén gọn ứng viên của lab 12.4 là một phần của mốc dự án M5 trong dự án xuyên suốt, lấy mẫu trên GPU: các cặp
            (column, logit) ổn định và một số đếm cho mỗi hàng của logits <code>[batch × vocab]</code>, bước nằm giữa một ngưỡng và
            một lần rút. Lab 9.3 cho M5 số đếm token, lab 10.3 cho giá trị lớn nhất của hàng và softmax, và lab 11.6 cho mặt nạ
            giữ lại top-p; lab 14.3 sắp xếp các ứng viên; module X1 thêm top-k radix (lab X1.1), min-p và các bộ lọc gộp
            (lab X1.2), và lần rút, trong bộ lấy mẫu theo batch của lab X1.3.{' '}
            <span data-m5={m5.join(',') || 'none'}>
              {backends.length === 0
                ? 'Máy này chưa ghi nhận kết quả lab nào.'
                : m5.length
                  ? `Lab 12.4 của chính bạn vượt qua trên ${m5.map(where).join(' và ')} trên máy này.`
                  : 'Lab 12.4 của chính bạn chưa vượt qua trên backend nào của máy này.'}
            </span>{' '}
            Xem <a href={href('ch12', 'labs')}>các lab</a>.
          </p>
        }
      >
        <p>
          Lab 12.4's candidate compaction is a piece of milestone M5 of the through-line project, sampling on the GPU:
          stable (column, logit) pairs and a count for every row of <code>[batch × vocab]</code> logits, the step between a
          threshold and a draw. Lab 9.3 gave M5 its token counts, lab 10.3 its row max and softmax, and lab 11.6 its
          top-p keep mask; lab 14.3 sorts the candidates; module X1 adds a radix top-k (lab X1.1), min-p and the fused
          filters (lab X1.2), and the draw, in lab X1.3's batched sampler.{' '}
          <span data-m5={m5.join(',') || 'none'}>
            {backends.length === 0
              ? 'No lab results are recorded on this machine yet.'
              : m5.length
                ? `Your own lab 12.4 passes on ${m5.map(where).join(' and ')} on this machine.`
                : 'Your own lab 12.4 does not pass on any backend of this machine yet.'}
          </span>{' '}
          See the <a href={href('ch12', 'labs')}>labs</a>.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chương 13 đi theo chiều ngược lại. Sách gọi việc lọc là nghịch đảo của việc trộn (merge) (p. 289): một phép trộn đan
            hai danh sách đã sắp xếp thành một, theo thứ tự. Đó là mẫu đầu tiên trong đó một luồng không thể suy ra đầu vào của nó
            từ chỉ số, và là một khối xây dựng cho phép sắp xếp của chương 14, thứ cho M5 các ứng viên đã sắp xếp. Bắt đầu ở{' '}
            <a href={href('ch13')}>chương 13</a>.
          </p>
        }
      >
        <p>
          Chapter 13 runs the other way. The book calls filtering the inverse of merging (p. 289): a merge interleaves two
          sorted lists into one, in order. It is the first pattern in which a thread cannot work out its inputs from its
          index, and a building block of chapter 14's sorting, which gives M5 its sorted candidates. Start at{' '}
          <a href={href('ch13')}>chapter 13</a>.
        </p>
      </Translated>
    </>
  );
}
