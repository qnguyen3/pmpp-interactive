import { Ask, type Question } from '../../components/Ask';
import { Beyond, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { RUNNING, coRankByMerge, mergeStable, mergeWith, outputsPerBlock, tag, type Item } from '../../sim/mergetile';

// The start page of chapter 13: what the chapter is for and what the course keeps of
// it, a question before any theory, the book's running example and its tie rule, a map,
// and a check on the ideas from chapters 2, 5 and 6 that the chapter builds on.
// Every number about the running example comes from src/sim/mergetile.ts.

const A = [...RUNNING.A];
const B = [...RUNNING.B];
const C = mergeStable(A, B);
const C_LT = mergeWith(A, B, '<');
const TIE_DIFF = C.filter((x, k) => tag(x) !== tag(C_LT[k])).length; // 5
const I3 = coRankByMerge(3, A, B); // 2
const I6 = coRankByMerge(6, A, B); // 5
const list = (xs: readonly number[]) => '[' + xs.join(', ') + ']';
const seq = (items: Item[]) => items.map(tag).join(' ');

// Fig. 13.9, line 03, with m + n = 9 and 4 threads: the share as printed and as meant.
const PER_PRINTED = outputsPerBlock(9, 4, 'printed'); // 2
const PER_MEANT = outputsPerBlock(9, 4, 'corrected'); // 3
const WRITTEN = Math.min(9, 4 * PER_PRINTED); // 8

export const questionsS130: Record<string, Question> = {
  'ch13/start/q-hook': {
    kind: 'choice',
    id: 'ch13/start/q-hook',
    prompt: (
      <Translated
        vi={
          <>
            <code>A = {list(A)}</code> và <code>B = {list(B)}</code> đã được sắp xếp. <code>C</code> là kết quả trộn của
            chúng, 9 phần tử, và phép trộn ổn định: khi hai khóa bằng nhau, phần tử từ A đứng trước. Ba luồng chia đều C,
            mỗi luồng ba đầu ra, nên luồng 1 tạo ra <code>C[3]</code>, <code>C[4]</code> và <code>C[5]</code>. Luồng 1 cần
            những phần tử đầu vào nào?
          </>
        }
      >
        <code>A = {list(A)}</code> and <code>B = {list(B)}</code> are sorted. <code>C</code> is their merge, 9 elements,
        and it is stable: where two keys are equal, the element from A comes first. Three threads split C evenly, three
        outputs each, so thread 1 produces <code>C[3]</code>, <code>C[4]</code> and <code>C[5]</code>. Which input
        elements does thread 1 need?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated
            vi={
              <>
                <code>A[1]</code>, <code>A[2]</code> và <code>A[3]</code> (7, 8 và 9), và không gì từ B.
              </>
            }
          >
            <code>A[1]</code>, <code>A[2]</code> and <code>A[3]</code> (7, 8 and 9), and nothing from B.
          </Translated>
        ),
        why: (
          <Translated vi="A[1] = 7 là C[1]. Số 1 và cả hai số 7 đứng trước 8, nên luồng 1 bắt đầu ở C[3] = 8.">
            A[1] = 7 is C[1]. The 1 and both 7s come before 8, so thread 1 starts at C[3] = 8.
          </Translated>
        ),
      },
      {
        label: (
          <Translated
            vi={
              <>
                <code>A[2]</code> và <code>A[3]</code> (8 và 9), và <code>B[1]</code> (10).
              </>
            }
          >
            <code>A[2]</code> and <code>A[3]</code> (8 and 9), and <code>B[1]</code> (10).
          </Translated>
        ),
        why: (
          <Translated vi="Cùng ba giá trị, nhưng quy tắc phá hòa quyết định số 10 nào là C[5]: A[4] = 10 đứng trước B[1] = 10. Vậy C[5] là A[4], còn B[1] là C[6], đầu ra đầu tiên của luồng 2.">
            The same three values, but the tie rule decides which 10 is C[5]: A[4] = 10 goes before B[1] = 10. So C[5] is
            A[4], and B[1] is C[6], the first output of thread 2.
          </Translated>
        ),
      },
      {
        label: (
          <Translated
            vi={
              <>
                <code>A[2]</code>, <code>A[3]</code> và <code>A[4]</code> (8, 9 và 10), và không gì từ B.
              </>
            }
          >
            <code>A[2]</code>, <code>A[3]</code> and <code>A[4]</code> (8, 9 and 10), and nothing from B.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Hai phần tử của A và một của B, vì B giữ gần một nửa dữ liệu.">
            Two elements of A and one of B, since B holds almost half the data.
          </Translated>
        ),
        why: (
          <Translated vi="Điều đó chỉ đúng khi hai mảng xen kẽ đều nhau. Phần của một luồng trong mỗi mảng phụ thuộc vào các giá trị, và có thể bằng không.">
            That holds only where the two arrays interleave evenly. One thread's share of each array depends on the
            values, and it can be zero.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đếm xem bao nhiêu trong 3 đầu ra đầu tiên đến từ mỗi mảng: {I3} từ A, {3 - I3} từ B. Trong 6 đầu ra đầu tiên:{' '}
            {I6} từ A, {6 - I6} từ B. Luồng 1 lấy phần chênh lệch, từ A[{I3}] đến A[{I6 - 1}] và không gì từ B. Các cặp
            đó, ({I3}, {3 - I3}) và ({I6}, {6 - I6}), là co-rank của 3 và 6. Không phép tính nào trên chỉ số luồng tạo ra
            chúng: chúng phụ thuộc vào các giá trị. Trang <a href={href('ch13', '13.3')}>13.3</a> tìm chúng bằng tìm kiếm
            nhị phân thay vì một phép trộn.
          </p>
        }
      >
        <p>
          Count how many of the first 3 outputs come from each array: {I3} from A, {3 - I3} from B. Of the first 6:{' '}
          {I6} from A, {6 - I6} from B. Thread 1 takes the difference, A[{I3}] up to A[{I6 - 1}] and nothing from B. Those
          pairs, ({I3}, {3 - I3}) and ({I6}, {6 - I6}), are the co-ranks of 3 and 6. No arithmetic on the thread index
          produces them: they depend on the values. Page <a href={href('ch13', '13.3')}>13.3</a> finds them with a binary
          search instead of a merge.
        </p>
      </Translated>
    ),
    source: 'ch. 13 §13.3 and §13.5, pp. 308, 314 (the book works this three-thread split)',
    vi: { source: 'ch. 13 §13.3 và §13.5, pp. 308, 314 (sách giải đúng cách chia ba luồng này)' },
  },
  'ch13/start/q-tie': {
    kind: 'choice',
    id: 'ch13/start/q-tie',
    prompt: (
      <Translated
        vi={
          <>
            Giả sử <code>merge_sequential</code> so sánh bằng <code>A[i] &lt; B[j]</code> thay vì{' '}
            <code>A[i] &lt;= B[j]</code>. Điều gì xảy ra với C trong ví dụ xuyên suốt?
          </>
        }
      >
        Suppose <code>merge_sequential</code> compared with <code>A[i] &lt; B[j]</code> instead of{' '}
        <code>A[i] &lt;= B[j]</code>. What happens to C for the running example?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="C không còn được sắp xếp ở bất cứ chỗ nào hai mảng có khóa bằng nhau.">
            C is no longer sorted wherever the two arrays hold equal keys.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi bước vẫn chép phần tử đầu nhỏ hơn. Khi hòa, lựa chọn nào cũng giữ C được sắp xếp.">
            Each step still copies the smaller front element. On a tie either choice keeps C sorted.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không có gì thay đổi: các khóa bằng nhau không thể phân biệt, nên C vẫn như trước.">
            Nothing changes: equal keys cannot be told apart, so C is the same as before.
          </Translated>
        ),
        why: (
          <Translated vi={`Khóa thì không; phần tử thì có. Gắn nhãn nguồn gốc cho mỗi phần tử và C khác đi ở ${TIE_DIFF} trong 9 vị trí của nó.`}>
            {`Their keys cannot; the elements can. Tag each element with its origin and C differs in ${TIE_DIFF} of its 9 places.`}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Hai số 10 của B đổi chỗ cho nhau, và số 10 của A dời ra sau cả hai.">
            B's two 10s swap places, and A's 10 moves after both of them.
          </Translated>
        ),
        why: (
          <Translated vi="Số 10 của A đúng là dời ra sau chúng, nhưng trong một danh sách, phép trộn chép theo thứ tự: j chỉ luôn tiến lên. Chỉ thứ tự giữa hai danh sách thay đổi.">
            A's 10 does move after them, but within one list the merge copies in order: j only ever moves forward. Only
            the order across the two lists changes.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các khóa giữ nguyên thứ tự, nhưng số 7 của B đứng trước số 7 của A và số 10 của A đứng sau hai số 10 của B.">
            Keys in the same order, but B's 7 before A's 7 and A's 10 after B's two 10s.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Với <code>&lt;</code>, khi hòa thì phần tử của B đi trước, và C thành <code>{seq(C_LT)}</code> thay vì{' '}
            <code>{seq(C)}</code>. Nó vẫn được sắp xếp và vẫn ổn định trong mỗi danh sách, nhưng không ổn định giữa hai
            danh sách. Phép tìm co-rank cũng phải mã hóa lựa chọn của sách: một trong hai phép thử của nó là chặt và phép
            kia thì không (<code>&gt;</code> và <code>&gt;=</code>, p. 311).
          </p>
        }
      >
        <p>
          With <code>&lt;</code> a tie sends B's element first, and C becomes <code>{seq(C_LT)}</code> instead of{' '}
          <code>{seq(C)}</code>. It is sorted and still stable within each list, but not across them. The co-rank search
          has to encode the book's choice too: one of its two tests is strict and the other is not (<code>&gt;</code> and{' '}
          <code>&gt;=</code>, p. 311).
        </p>
      </Translated>
    ),
    source: 'ch. 13 §13.1 and §13.2, pp. 304–305 (the < variant is ours)',
    vi: { source: 'ch. 13 §13.1 và §13.2, pp. 304–305 (biến thể < là của khóa học này)' },
  },
  'ch13/start/q-ceil': {
    kind: 'numeric',
    id: 'ch13/start/q-ceil',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel chia cho mỗi luồng phần đầu ra của nó bằng{' '}
            <code>int per = ceil((m + n) / (blockDim.x * gridDim.x));</code> và luồng <code>t</code> ghi các đầu ra từ{' '}
            <code>t*per</code> đến <code>(t+1)*per - 1</code>. Với <code>m + n = 9</code> và tổng cộng 4 luồng, bao nhiêu
            trong 9 đầu ra được ghi?
          </>
        }
      >
        A kernel gives each thread its share of the outputs with{' '}
        <code>int per = ceil((m + n) / (blockDim.x * gridDim.x));</code> and thread <code>t</code> writes outputs{' '}
        <code>t*per</code> up to <code>(t+1)*per - 1</code>. With <code>m + n = 9</code> and 4 threads in all, how many of
        the 9 outputs get written?
      </Translated>
    ),
    answer: WRITTEN,
    unit: 'outputs',
    explain: (
      <Translated
        vi={
          <p>
            <code>(m + n) / (blockDim.x * gridDim.x)</code> chia số nguyên, nên 9 / 4 đã là {PER_PRINTED} khi{' '}
            <code>ceil</code> nhìn thấy nó. Bốn luồng mỗi luồng ghi {PER_PRINTED}, từ <code>C[0]</code> đến{' '}
            <code>C[{WRITTEN - 1}]</code>, và <code>C[8]</code> không bao giờ được ghi. Không có gì báo lỗi; chỉ đơn giản
            là thiếu một đầu ra. Chương 2 tránh điều này bằng một hằng số dấu phẩy động, <code>ceil(n/256.0)</code>; NVIDIA
            dạy dạng số nguyên <code>(a + b - 1) / b</code>, ở đây cho {PER_MEANT}. Dòng này là của chính sách, từ kernel
            trộn cơ bản của nó (Fig. 13.9, dòng 03, nơi <code>per</code> được gọi là <code>elementsPerThread</code>), và
            cùng lỗi đó khiến kernel phân ô của sách chạy 3 lần lặp trong khi văn bản nói 4.
          </p>
        }
      >
        <p>
          <code>(m + n) / (blockDim.x * gridDim.x)</code> divides integers, so 9 / 4 is already {PER_PRINTED} when{' '}
          <code>ceil</code> sees it. Four threads write {PER_PRINTED} each, <code>C[0]</code> to <code>C[{WRITTEN - 1}]</code>
          , and <code>C[8]</code> is never written. Nothing fails; one output is simply missing. Chapter 2 avoided this
          with a floating-point literal, <code>ceil(n/256.0)</code>; NVIDIA teaches the integer form{' '}
          <code>(a + b - 1) / b</code>, which gives {PER_MEANT} here. The line is the book's own, from its basic merge
          kernel (Fig. 13.9, line 03, where <code>per</code> is called <code>elementsPerThread</code>), and the same slip
          makes its tiled kernel run 3 iterations where the text says 4.
        </p>
      </Translated>
    ),
    source: 'ch. 2 §2.6, p. 38; ch02-facts A8; ch. 13 Fig. 13.9 line 03 (notes/ch13.md, "Unclear in the PDF" item 14; ch13-facts A4)',
    vi: {
      unit: 'đầu ra',
      source: 'ch. 2 §2.6, p. 38; ch02-facts A8; ch. 13 Fig. 13.9 dòng 03 (notes/ch13.md, mục 14 của "Unclear in the PDF"; ch13-facts A4)',
    },
  },
  'ch13/start/q-stride': {
    kind: 'numeric',
    id: 'ch13/start/q-stride',
    prompt: (
      <Translated
        vi={
          <>
            Trên GPU NVIDIA, bộ nhớ toàn cục (global memory) được truy cập theo các đoạn 32 byte và một warp có 32 luồng.
            Trong một lần nạp, lane <code>t</code> của một warp đọc số float 4 byte ở chỉ số <code>2*t</code> của một mảng
            mà phần tử 0 nằm ở đầu một đoạn. Lần nạp đó chạm tới bao nhiêu đoạn?
          </>
        }
      >
        On an NVIDIA GPU, global memory is accessed in 32-byte segments and a warp has 32 threads. In one load, lane{' '}
        <code>t</code> of a warp reads the 4-byte float at index <code>2*t</code> of an array whose element 0 starts a
        segment. How many segments does that one load touch?
      </Translated>
    ),
    answer: 8,
    unit: 'segments',
    explain: (
      <Translated
        vi={
          <p>
            Các lane đọc chỉ số 0, 2, ..., 62: byte 0 đến 251, nằm trong một khoảng 256 byte, tức 8 đoạn. Một nửa của mỗi
            đoạn không được dùng, nên một nửa số byte được chuyển bị lãng phí; tài liệu hướng dẫn của NVIDIA cũng cho con số
            50% đó với bước nhảy (stride) bằng 2. Trang 13.3 gặp một trường hợp tệ hơn: trong kernel trộn cơ bản của sách,
            mỗi luồng đi qua đoạn A và B của riêng nó, nên các lane kề nhau đọc những địa chỉ cách nhau vài phần tử, với
            khoảng cách phụ thuộc vào dữ liệu (§13.5, p. 314).
          </p>
        }
      >
        <p>
          The lanes read indices 0, 2, ..., 62: bytes 0 to 251, inside a 256-byte span, which is 8 segments. Half of every
          segment goes unused, so half the bytes moved are wasted; NVIDIA's guide gives the same 50% for a stride of 2.
          Page 13.3 meets a worse case: in the book's basic merge kernel each thread walks its own stretch of A and B, so
          neighbouring lanes read addresses several elements apart, by amounts that depend on the data (§13.5, p. 314).
        </p>
      </Translated>
    ),
    source: "ch. 6 §6.1, pp. 124–132; NVIDIA's 32-byte unit and the stride-2 figure: ch06-facts A2",
    vi: {
      unit: 'đoạn',
      source: 'ch. 6 §6.1, pp. 124–132; đơn vị 32 byte của NVIDIA và con số cho bước nhảy 2: ch06-facts A2',
    },
  },
  'ch13/start/q-strategy': {
    kind: 'choice',
    id: 'ch13/start/q-strategy',
    prompt: (
      <Translated
        vi={
          <>
            Mọi luồng của một khối tìm kiếm nhị phân trên cùng một mảng đã sắp xếp 1,024 phần tử, nằm trong bộ nhớ toàn cục.
            Mỗi lần tìm đọc khoảng mười phần tử, ở những vị trí phụ thuộc vào khóa nó đang tìm. Thay đổi nào làm cho các lần
            đọc bộ nhớ toàn cục của khối được hợp nhất (coalesced)?
          </>
        }
      >
        Every thread of a block binary-searches the same sorted 1,024-element array, which sits in global memory. Each
        search reads about ten elements, at places that depend on the key it looks for. Which change makes the block's
        global-memory reads coalesced?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Gán khóa cho các luồng sao cho các luồng kề nhau tìm các khóa kề nhau.">
            Assign keys to threads so that neighbouring threads search for neighbouring keys.
          </Translated>
        ),
        why: (
          <Translated vi="Các lần dò của chúng có thể rơi gần nhau hơn, nhưng một lần tìm đọc những phần tử nào vẫn phụ thuộc vào dữ liệu. Không có gì xếp các lane thẳng hàng trên các địa chỉ liên tiếp.">
            Their probes may land closer together, but which elements a search reads still depends on the data. Nothing
            lines the lanes up on consecutive addresses.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cho khối chép mảng vào bộ nhớ chia sẻ (shared memory), theo cách được hợp nhất, rồi tìm ở đó.">
            Have the block copy the array into shared memory, coalesced, and search it there.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Lưu mảng theo một bố cục khác, như bạn làm với một ma trận chuyển vị.">
            Store the array in a different layout, as you would a transposed matrix.
          </Translated>
        ),
        why: (
          <Translated vi="Đổi bố cục có ích khi mẫu truy cập cố định, như với ma trận chuyển vị. Ở đây những phần tử một lần tìm đọc phụ thuộc vào khóa nó tìm, nên một cách sắp xếp lại cố định không thể bảo đảm các lane của một warp đọc những địa chỉ kề nhau.">
            Changing the layout helps when the access pattern is fixed, as with a transposed matrix. Here the elements a
            search reads depend on the keys it looks for, so a fixed rearrangement cannot promise that the lanes of a warp
            read neighbouring addresses.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Khởi chạy nhiều luồng hơn, để hệ thống bộ nhớ có nhiều yêu cầu hơn để gộp.">
            Launch more threads, so the memory system has more requests to combine.
          </Translated>
        ),
        why: (
          <Translated vi="Nhiều yêu cầu không đều hơn thì vẫn không đều. Hợp nhất truy cập bộ nhớ cần các lane của một warp chạm cùng các đoạn trong cùng một lần nạp.">
            More irregular requests are still irregular. Coalescing needs the lanes of one warp to hit the same segments
            in the same load.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đây là chiến lược thứ ba của chương 6: chuyển dữ liệu giữa bộ nhớ toàn cục và bộ nhớ chia sẻ theo mẫu được hợp
            nhất, các luồng liên tiếp trên các phần tử liên tiếp, rồi thực hiện các truy cập không đều trong bộ nhớ chia
            sẻ, nơi không cần hợp nhất. Danh sách kiểm tra của chương 6 lấy chính chương này làm ví dụ. Kernel trộn phân ô
            của §13.6 làm điều đó với một ô (tile) của A và một ô của B.
          </p>
        }
      >
        <p>
          This is the third of chapter 6's strategies: move the data between global and shared memory in a coalesced
          pattern, consecutive threads on consecutive elements, then make the irregular accesses in shared memory, which
          does not need coalescing. Chapter 6's
          checklist names this very chapter as its example. The tiled merge kernel of §13.6 does it with a tile of A and a
          tile of B.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.1, p. 129 (strategy 3), and §6.8, p. 150, which names chapter 13',
    vi: { source: 'ch. 6 §6.1, p. 129 (chiến lược 3), và §6.8, p. 150, nơi nêu tên chương 13' },
  },
  'ch13/start/q-barriers': {
    kind: 'choice',
    id: 'ch13/start/q-barriers',
    prompt: (
      <Translated
        vi={
          <>
            Một khối chạy một vòng lặp. Trong mỗi lần lặp, các luồng của nó nạp một ô vào một mảng chia sẻ, rồi mọi luồng
            tìm kiếm nhị phân trên cả ô, rồi lần lặp tiếp theo nạp ô kế tiếp vào cùng mảng đó. Mỗi lần lặp cần những lời
            gọi <code>__syncthreads()</code> nào?
          </>
        }
      >
        A block runs a loop. In each iteration its threads load a tile into a shared array, then every thread
        binary-searches the whole tile, then the next iteration loads the next tile into the same array. Which{' '}
        <code>__syncthreads()</code> calls does each iteration need?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một, sau các lần nạp, để không luồng nào tìm trên một ô mới nạp một nửa.">
            One, after the loads, so that no thread searches a half-loaded tile.
          </Translated>
        ),
        why: (
          <Translated vi="Lời gọi đó ngăn các luồng tìm trên một ô mới nạp một nửa. Không có lời gọi thứ hai, một luồng nhanh bắt đầu nạp ô kế tiếp đè lên những phần tử mà một luồng chậm hơn vẫn đang tìm.">
            That one keeps threads from searching a half-loaded tile. Without a second, a fast thread starts loading the
            next tile over elements a slower thread is still searching.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một, sau các lần tìm, để không luồng nào ghi đè một ô đang được dùng.">
            One, after the searches, so that no thread overwrites a tile in use.
          </Translated>
        ),
        why: (
          <Translated vi="Khi đó một luồng có thể tìm trên ô trước khi các luồng khác ghi xong nó.">
            Then a thread may search the tile before the other threads have finished writing it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Hai: một sau các lần nạp và một sau các lần tìm.">
            Two: one after the loads and one after the searches.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không cần lời gọi nào, nếu chỉ một warp vừa nạp vừa tìm trên ô.">
            None, if a single warp both loads and searches the tile.
          </Translated>
        ),
        why: (
          <Translated vi="Trên GPU NVIDIA có compute capability 7.0 trở lên, các luồng của một warp không được bảo đảm chạy đồng bộ từng bước (ch04-facts A10). Warp không cho bạn thứ tự nào để dựa vào; rào chắn (barrier) thì có.">
            On NVIDIA GPUs of compute capability 7.0 and later the threads of a warp are not guaranteed to run in
            lockstep (ch04-facts A10). A warp gives you no ordering to rely on; the barrier does.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Rào chắn thứ nhất canh một phụ thuộc thật: mọi luồng đọc những phần tử mà các luồng khác đã ghi (đọc sau ghi).
            Rào chắn thứ hai canh một phụ thuộc giả: các lần nạp tiếp theo không cần gì từ các lần tìm, chúng chỉ không được
            ghi đè những phần tử vẫn đang được đọc (ghi sau đọc). Kernel trộn phân ô có cả hai (Fig. 13.12 dòng 36 và Fig.
            13.13 dòng 50), cùng hai rào chắn nữa trong Fig. 13.11 quanh việc chia sẻ co-rank của khối qua bộ nhớ chia sẻ.
          </p>
        }
      >
        <p>
          The first barrier guards a true dependence: every thread reads elements that other threads wrote (read after
          write). The second guards a false one: the next loads need nothing from the searches, they only must not
          overwrite elements still being read (write after read). The tiled merge kernel has both (Fig. 13.12 line 36 and
          Fig. 13.13 line 50), and two more in Fig. 13.11 around sharing the block's co-ranks through shared memory.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.4, pp. 110–111; ch. 13 §13.6, pp. 316–318', // i18n-ignore: citation
  },
  'ch13/start/q-coarsen': {
    kind: 'numeric',
    id: 'ch13/start/q-coarsen',
    prompt: (
      <Translated
        vi={
          <>
            Mỗi luồng của một kernel thực hiện 40 lần đọc toàn cục để chuẩn bị trước khi tạo ra bất cứ thứ gì, rồi 1 lần đọc
            toàn cục cho mỗi đầu ra. Có 1,000,000 đầu ra. Với 8 đầu ra mỗi luồng, cả lưới (grid) thực hiện bao nhiêu lần đọc
            toàn cục, tính bằng triệu?
          </>
        }
      >
        Every thread of a kernel makes 40 global reads of setup before it produces anything, then 1 global read per
        output. There are 1,000,000 outputs. With 8 outputs per thread, how many global reads does the whole grid make, in
        millions?
      </Translated>
    ),
    answer: 6,
    tol: 0.001,
    unit: 'million reads',
    explain: (
      <Translated
        vi={
          <p>
            1,000,000 / 8 = 125,000 luồng, mỗi luồng trả 40 lần đọc, tức 5 triệu, cộng 1 triệu cho các đầu ra: 6 triệu. Với
            một đầu ra mỗi luồng thì là 41 triệu, gần như toàn bộ là phần chuẩn bị. Đó là gộp việc cho luồng (thread
            coarsening): khi công việc song song tốn một chi phí cho mỗi luồng, hãy giao cho mỗi luồng nhiều việc hơn. Trong
            phép trộn, phần chuẩn bị của mỗi luồng là phép tìm co-rank, và sách gộp việc cho mọi kernel trộn trong chương vì
            lý do đó (§13.8, p. 327).
          </p>
        }
      >
        <p>
          1,000,000 / 8 = 125,000 threads pay 40 reads each, 5 million, plus 1 million for the outputs: 6 million. At one
          output per thread it is 41 million, almost all of it setup. That is thread coarsening: when parallel work costs
          something per thread, give each thread more work. In a merge the per-thread setup is the co-rank search, and the
          book coarsens every merge kernel in the chapter for that reason (§13.8, p. 327).
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.5, pp. 141–143; ch. 13 §13.8, p. 327', // i18n-ignore: citation
    vi: { unit: 'triệu lần đọc' },
  },
};

// ---------------------------------------------------------------- the running example, redrawn
const CELL = 40;
const STEP = 46;
const X0 = 34;
const cellX = (k: number) => X0 + k * STEP;

/** Runs of equal keys in C that hold more than one element: where the tie rule decides. */
function tieGroups(c: Item[]) {
  const groups: { from: number; to: number; items: Item[] }[] = [];
  for (let k = 0; k < c.length; ) {
    let e = k;
    while (e + 1 < c.length && c[e + 1].key === c[k].key) e++;
    if (e > k) groups.push({ from: k, to: e, items: c.slice(k, e + 1) });
    k = e + 1;
  }
  return groups;
}

function Row({ name, y, cells }: { name: string; y: number; cells: Item[] }) {
  return (
    <g data-row={name}>
      <text x={8} y={y + 25} fontSize="15" fontWeight="600" style={{ fill: 'var(--ink)' }}>
        {name}
      </text>
      {cells.map((it, k) => (
        <g key={k} data-tag={tag(it)}>
          <text x={cellX(k) + CELL / 2} y={y - 6} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
            {tag(it)}
          </text>
          <rect
            x={cellX(k)}
            y={y}
            width={CELL}
            height={36}
            rx={3}
            fill="var(--global-wash)"
            stroke="var(--global)"
            strokeWidth={1.5}
            strokeDasharray={it.src === 'B' ? '4 3' : undefined}
          />
          <text x={cellX(k) + CELL / 2} y={y + 24} fontSize="15" textAnchor="middle" style={{ fill: 'var(--ink)' }}>
            {it.key}
          </text>
        </g>
      ))}
    </g>
  );
}

function RunningExample() {
  const aItems: Item[] = A.map((key, idx) => ({ key, src: 'A', idx }));
  const bItems: Item[] = B.map((key, idx) => ({ key, src: 'B', idx }));
  const ties = tieGroups(C);
  const yC = 176;
  return (
    <figure style={{ margin: '0.4rem 0 1.4rem' }}>
      <svg
        viewBox="0 0 456 252"
        role="img"
        aria-label={tr(
          `A is ${list(A)}, B is ${list(B)}, and their stable merge C is ${list(C.map((x) => x.key))}, taken in the order ${seq(C)}.`,
          `A là ${list(A)}, B là ${list(B)}, và kết quả trộn ổn định C của chúng là ${list(C.map((x) => x.key))}, lấy theo thứ tự ${seq(C)}.`,
        )}
        style={{ width: '100%', maxWidth: 456, display: 'block', fontFamily: 'var(--sans)' }}
      >
        <Row name="A" y={24} cells={aItems} />
        <Row name="B" y={96} cells={bItems} />
        <Row name="C" y={yC} cells={C} />
        {ties.map((g) => {
          const x1 = cellX(g.from) + 2;
          const x2 = cellX(g.to) + CELL - 2;
          const yb = yC + 46;
          return (
            <g key={g.from} data-tie={g.items[0].key}>
              <path d={`M${x1},${yb} v6 H${x2} v-6`} fill="none" stroke="var(--ink-2)" strokeWidth={1.2} />
              <text x={(x1 + x2) / 2} y={yb + 22} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                {g.items.map(tag).join(tr(' before ', ' trước '))}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption className="muted" style={{ fontFamily: 'var(--sans)', fontSize: '0.8rem' }}>
        <Translated
          vi={
            <>
              Ví dụ xuyên suốt của sách (Fig. 13.1), vẽ lại. Nhãn phía trên mỗi phần tử cho biết nguồn gốc của nó (A1 là
              A[1]), và viền nét đứt đánh dấu một phần tử từ B. Các dấu ngoặc đánh dấu những khóa bằng nhau, nơi quy tắc phá
              hòa quyết định. Cả ba mảng đều mang màu của bộ nhớ toàn cục: các kernel của chương giữ chúng ở đó (p. 313).
            </>
          }
        >
          The book's running example (Fig. 13.1), redrawn. The tag above each element names its origin (A1 is A[1]), and a
          dashed outline marks an element from B. The brackets mark equal keys, where the tie rule decides. All three
          arrays are in the global-memory colour: the chapter's kernels keep them there (p. 313).
        </Translated>
      </figcaption>
    </figure>
  );
}

export function S130() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mọi kernel cho đến giờ đều tìm đầu vào của nó bằng phép tính trên chính chỉ số của nó. Luồng <code>i</code> của
            phép cộng vectơ đọc phần tử <code>i</code>; một khối nhân ma trận phân ô nạp những ô mà <code>blockIdx</code>{' '}
            của nó chỉ tới. Trộn hai mảng đã sắp xếp phá vỡ điều đó. Những đầu vào nào nuôi đầu ra <code>k</code> phụ thuộc
            vào giá trị trong cả hai mảng, nên một luồng phải tìm xem đầu vào của nó bắt đầu ở đâu trước khi có thể làm bất
            cứ việc gì.
          </p>
        }
      >
        <p>
          Every kernel so far found its inputs by arithmetic on its own index. Thread <code>i</code> of a vector add reads
          element <code>i</code>; a tiled matmul block loads the tiles its <code>blockIdx</code> names. Merging two sorted
          arrays breaks that. Which inputs feed output <code>k</code> depends on the values in both arrays, so a thread has
          to search for where its inputs start before it can do any work.
        </p>
      </Translated>
      <Source section="13 (opening), 13.3" pages="303, 308" />
      <Translated
        vi={
          <p>
            Chương này được rút gọn trong khóa học vì một lý do đơn giản: không kernel nhân ma trận, attention hay lượng tử
            hóa nào trộn hai danh sách đã sắp xếp. Có hai ý của nó chuyển sang được, và đó là những gì khóa học giữ lại:
          </p>
        }
      >
        <p>
          The chapter is condensed in this course for a plain reason: no matmul, attention or quantised kernel merges two
          sorted lists. Two of its ideas do transfer, and they are what the course keeps:
        </p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>
              <b>phép tìm co-rank</b>: chia cho mọi luồng một lát đầu ra bằng nhau, rồi tìm kiếm nhị phân hai khoảng đầu vào
              tạo ra lát đó. Khi đó mỗi luồng có cùng một lượng việc, bất kể dữ liệu trông thế nào;
            </li>
          }
        >
          <li>
            <b>the co-rank search</b>: give every thread an equal slice of the output, then binary-search for the two
            input ranges that produce it. Each thread then has the same amount of work, whatever the data looks like;
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>đưa các lần đọc không đều qua bộ nhớ chia sẻ (shared memory)</b>: khi một khối sẽ đọc đầu vào ở những vị
              trí phụ thuộc vào dữ liệu, trước tiên nó chép phần nó cần vào bộ nhớ chia sẻ bằng các lần nạp được hợp nhất
              (coalesced), mỗi lần một ô, rồi thực hiện các lần đọc không đều ở đó.
            </li>
          }
        >
          <li>
            <b>staging irregular reads through shared memory</b>: when a block will read its inputs at places that depend
            on the data, it first copies the part it needs into shared memory with coalesced loads, one tile at a time,
            and makes the irregular reads there.
          </li>
        </Translated>
      </ul>

      <Translated
        vi={
          <p>
            Trên GPU Apple, phép tìm này đã nằm bên dưới mã mà bạn có thể đang dùng. <code>mx.sort</code> của MLX là một
            merge sort, và <code>mx.argpartition</code> cùng <code>mx.topk</code> đi qua chính phép sắp xếp đó. Các lượt
            trộn của nó chia việc bằng co-rank, trước theo từng khối trên bộ nhớ thiết bị, rồi theo từng luồng trên một ô
            trong bộ nhớ threadgroup: gần với kernel phân ô của sách. Bộ lấy mẫu top-k của mlx-lm gọi{' '}
            <code>mx.argpartition</code> và bộ lấy mẫu top-p của nó gọi <code>mx.sort</code>, nên lấy mẫu từ một mô hình
            MLX với top-k hay top-p là chạy phép tìm của chương này. Backend MPS của PyTorch có một merge sort xây theo cùng
            cách phía sau <code>torch.sort</code> và <code>torch.topk</code>, được chọn theo từng lời gọi so với một radix
            sort.
          </p>
        }
      >
        <p>
          On an Apple GPU this search already sits under code you may use. MLX's <code>mx.sort</code> is a merge sort, and{' '}
          <code>mx.argpartition</code> and <code>mx.topk</code> go through the same sort. Its merge passes split the work
          with a co-rank, first per block on device memory, then per thread on a tile in threadgroup memory: close to the
          book's tiled kernel. mlx-lm's top-k sampler calls <code>mx.argpartition</code> and its top-p sampler{' '}
          <code>mx.sort</code>, so sampling from an MLX model with top-k or top-p runs this chapter's search. PyTorch's MPS backend has
          a merge sort built the same way behind <code>torch.sort</code> and <code>torch.topk</code>, chosen per call
          against a radix sort.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không nhắc đến MLX hay PyTorch, cũng không nhắc đến lấy mẫu. Mối liên hệ này là của chúng tôi, đọc từ mã
              nguồn của các dự án (ch13-facts D1, D2).
            </p>
          }
        >
          <p>
            The book mentions neither MLX nor PyTorch, nor sampling. The link is ours, read from the projects' source
            code (ch13-facts D1, D2).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Khi đọc mã đó, hãy tìm một cái tên khác: các thư viện gọi co-rank là <i>merge path</i> (CUB, rocPRIM) hoặc{' '}
            <code>merge_partition</code> (MLX, PyTorch). Và trên GPU NVIDIA, top-k thông thường hoàn toàn không phải phép
            trộn: PyTorch và llama.cpp dùng radix select hoặc bitonic sort ở đó.
          </p>
        }
      >
        <p>
          When you read that code, look for another name: the libraries call the co-rank <i>merge path</i> (CUB, rocPRIM)
          or <code>merge_partition</code> (MLX, PyTorch). And on NVIDIA GPUs the usual top-k is not a merge at all:
          PyTorch and llama.cpp use radix select or bitonic sort there.
        </p>
      </Translated>
      <Today
        checked={tr(
          "2026-11-20 in the projects' source (ch13-facts A1, D1, D2, D4)",
          '2026-11-20 trong mã nguồn của các dự án (ch13-facts A1, D1, D2, D4)',
        )}
      >
        <Translated
          vi={
            <p>
              MLX: <code>mb_block_partition</code> và <code>mb_block_merge</code> trong <code>kernels/sort.h</code>. PyTorch
              MPS: <code>merge_partition</code> trong <code>kernels/SortMerge.h</code>.
            </p>
          }
        >
          <p>
            MLX: <code>mb_block_partition</code> and <code>mb_block_merge</code> in <code>kernels/sort.h</code>. PyTorch
            MPS: <code>merge_partition</code> in <code>kernels/SortMerge.h</code>.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Một câu hỏi trước mọi lý thuyết">A question before any theory</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hãy thử câu này trước khi đọc tiếp. Nó là cả chương trong một ví dụ nhỏ, và bạn không cần gì hơn phép trộn bạn
            đã biết.
          </p>
        }
      >
        <p>
          Try this before reading on. It is the whole chapter in one small example, and you need nothing more than the
          merge you already know.
        </p>
      </Translated>
      <Ask q={questionsS130['ch13/start/q-hook']} label={tr('Before any theory', 'Trước mọi lý thuyết')} />

      <h2>
        <Translated vi="Ví dụ xuyên suốt">The running example</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách dùng hai mảng này cho cả chương, và khóa học này cũng vậy. Đây là chúng cùng kết quả trộn, mỗi phần tử được
            gắn nhãn nơi nó đến.
          </p>
        }
      >
        <p>
          The book uses these two arrays for the whole chapter, and so does this course. Here they are with their merge,
          each element tagged with where it came from.
        </p>
      </Translated>
      <RunningExample />
      <Translated
        vi={
          <p>
            Phép trộn tuần tự là phép trộn của merge sort: đi qua cả hai mảng, mỗi mảng một chỉ số, chép phần tử đầu nhỏ hơn
            sang C, và khi một mảng hết, chép phần còn lại của mảng kia. Mỗi phần tử được chạm một lần, nên chi phí là O(m +
            n). Chi tiết quan trọng trong chương này là phép so sánh, <code>A[i] &lt;= B[j]</code>: khi khóa bằng nhau,
            phần tử của A đi trước. Điều đó làm phép trộn <b>ổn định</b>. Các khóa bằng nhau giữ thứ tự của chúng trong mỗi
            danh sách (hai số 10 của B giữ nguyên thứ tự) và giữa hai danh sách (số 7 của A trước số 7 của B, số 10 của A
            trước cả hai số 10 của B).
          </p>
        }
      >
        <p>
          The sequential merge is the one from merge sort: walk both arrays with one index each, copy the smaller front
          element to C, and when one array runs out, copy the rest of the other. Each element is touched once, so it costs
          O(m + n). The detail that matters in this chapter is the comparison, <code>A[i] &lt;= B[j]</code>: on equal
          keys, A's element goes first. That makes the merge <b>stable</b>. Equal keys keep their order within each list
          (B's two 10s stay in order) and across the lists (A's 7 before B's 7, A's 10 before both of B's).
        </p>
      </Translated>
      <Source section="13.1, 13.2" pages="303–306" figs="Figs. 13.1, 13.2" />
      <Translated
        vi={
          <p>
            Vì sao phải giữ thứ tự khi hòa nếu các khóa bằng nhau trông như nhau? Vì phần tử mang nhiều hơn khóa của nó: các
            bản ghi đã được sắp xếp trước đó theo một khóa khác giữ nguyên thứ tự ấy qua một phép trộn ổn định, nên công việc
            trước đó không bị mất. Quy tắc phá hòa cũng xuất hiện lại trong phép tìm co-rank dưới dạng một cặp phép thử lệch
            nhau.
          </p>
        }
      >
        <p>
          Why keep a tie order when equal keys look the same? Because elements carry more than their key: records sorted
          earlier by another key keep that order through a stable merge, so the earlier work is not lost. The tie rule also
          reappears in the co-rank search as a pair of lopsided tests.
        </p>
      </Translated>
      <Ask q={questionsS130['ch13/start/q-tie']} />

      <h2>
        <Translated vi="Chương này trong khóa học">The chapter in this course</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một trang giảng, một lab tùy chọn và một bản tóm tắt. <a href={href('ch13', '13.3')}>Trang 13.3</a> bao quát
            những gì chương dạy sau trang này: hàm co-rank và phép tìm kiếm nhị phân của nó (§13.3 và §13.4, pp. 306–313),
            kernel cơ bản xây trên nó (§13.5), kernel phân ô đưa cả hai đầu vào qua bộ nhớ chia sẻ (§13.6), và vì sao mọi
            kernel trộn trong chương đều được gộp việc (§13.8). <a href={href('ch13', 'labs')}>Lab</a> trộn hai mảng đã sắp
            xếp trên GPU của chính bạn bằng cách phân chia theo co-rank. <a href={href('ch13', 'wrap')}>Bản tóm tắt</a> có
            một bài kiểm tra tổng hợp, bốn bài tập của sách và những chỗ nhầm trong bản PDF.
          </p>
        }
      >
        <p>
          One teaching page, an optional lab and a summary. <a href={href('ch13', '13.3')}>Page 13.3</a> covers what the
          chapter teaches after this page: the co-rank function and its binary search (§13.3 and §13.4, pp. 306–313), the
          basic kernel built on it (§13.5), the tiled kernel that stages both inputs through shared memory (§13.6), and why
          every merge kernel in the chapter is coarsened (§13.8). The <a href={href('ch13', 'labs')}>lab</a> merges two
          sorted arrays on your own GPU with co-rank partitioning. The <a href={href('ch13', 'wrap')}>summary</a> has a
          mixed check, the book's four exercises and the slips in the PDF.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Những gì bị bỏ qua: §13.7, kernel buffer vòng (pp. 321–327), giữ lại nửa mỗi ô mà kernel phân ô vứt đi. Trang
            13.3 mô tả nó trong một đoạn, và trang tóm tắt nói vì sao khóa học không đi xa hơn. Hai phần nền tảng, §13.1 và
            §13.2, là trang này.
          </p>
        }
      >
        <p>
          What is left out: §13.7, the circular-buffer kernel (pp. 321–327), which keeps the half of each tile that the
          tiled kernel throws away. Page 13.3 describes it in one paragraph, and the summary page says why the course goes
          no further. The two background sections, §13.1 and §13.2, are this page.
        </p>
      </Translated>

      <h2>
        <Translated vi="Trước khi bắt đầu">Before you start</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Năm câu hỏi về những gì chương này lấy từ chương 2, 5 và 6, theo thứ tự chương dùng đến chúng. Mỗi câu sẽ quay
            lại trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Five questions on what this chapter takes from chapters 2, 5 and 6, in the order the chapter uses them. Each one
          comes back in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={questionsS130['ch13/start/q-ceil']} label={tr('From chapter 2', 'Từ chương 2')} />
      <Ask q={questionsS130['ch13/start/q-stride']} label={tr('From chapter 6', 'Từ chương 6')} />
      <Ask q={questionsS130['ch13/start/q-strategy']} label={tr('From chapter 6', 'Từ chương 6')} />
      <Ask q={questionsS130['ch13/start/q-barriers']} label={tr('From chapter 5', 'Từ chương 5')} />
      <Ask q={questionsS130['ch13/start/q-coarsen']} label={tr('From chapter 6', 'Từ chương 6')} />
    </>
  );
}
