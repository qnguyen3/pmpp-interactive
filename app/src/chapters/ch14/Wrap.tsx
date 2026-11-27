import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Beyond, Source, type Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { functionSource } from '../../sim/toolchain';

// The kernel of the first question is a snippet of our own, loaded at run time so there is no
// copy to drift: labs/ch14/snippets/wrap_blocksort.cu. Wrap.snippets.test.ts runs it, and the
// corrected copy in labs/ch14/snippets/check/wrap_check.cu, on the course's emulator, and pins
// what the question says about both.
export const BLOCKSORT_FILE = 'ch14/snippets/wrap_blocksort.cu';
const TIERS: Record<string, Tier> = { keys: 'global', s: 'shared', t: 'reg', i: 'reg', step: 'reg', tmp: 'reg', n: 'reg' };

/** MSD select: keys per top-8-bit bin, from the top bin down, and the k of q-select-bin. */
export const SELECT_BINS = [0, 3, 12, 40, 97];
export const SELECT_K = 50;
/** The bin holding the k-th largest key, the keys above it, and how many it must still give. */
export function selectBin(bins: number[], k: number): { bin: number; above: number; still: number } {
  let above = 0;
  for (let b = 0; b < bins.length; ++b) {
    if (above + bins[b] >= k) return { bin: b, above, still: k - above };
    above += bins[b];
  }
  return { bin: -1, above, still: k - above };
}
/** Passes of an LSD radix sort with d-bit digits over the low `bits` bits of a key. */
export const passes = (bits: number, d: number) => Math.ceil(bits / d);
/** Bits of row id that `rows` rows need. */
export const rowBits = (rows: number) => Math.ceil(Math.log2(rows));

const SEL = selectBin(SELECT_BINS, SELECT_K);
const ROWS = 64;
const PACKED = passes(32 + rowBits(ROWS), 8);

export const questionsWrap: Record<string, Question> = {
  'ch14/wrap/q-blocksort': {
    kind: 'choice',
    id: 'ch14/wrap/q-blocksort',
    prompt: (
      <Translated
        vi={
          <>
            Kernel ở trên được khởi chạy thành một khối gồm <code>(n + 1) / 2</code> luồng. Một bài kiểm thử trên GPU dùng nó sắp xếp
            64 khóa ngẫu nhiên và chạy đạt. Nó sai ở đâu?
          </>
        }
      >
        <>
          The kernel above is launched as one block of <code>(n + 1) / 2</code> threads. A test on a GPU sorts 64 random keys
          with it and passes. What is wrong with it?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không có gì: trong một bước các cặp rời nhau, nên không có hai luồng nào chạm vào cùng một phần tử, và bài kiểm thử trên GPU xác nhận điều đó.">
            Nothing: within a step the pairs are disjoint, so no two threads touch the same element, and the GPU test confirms it.
          </Translated>
        ),
        why: (
          <Translated vi="Trong một bước thì đúng. Nhưng phần tử 2t + 2 thuộc luồng t + 1 ở bước chẵn và thuộc luồng t ở bước lẻ ngay sau đó, và không có gì buộc bước chẵn phải xong trước.">
            Within one step, yes. But element 2t + 2 belongs to thread t + 1 on an even step and to thread t on the odd step after it, and nothing makes the even step finish first.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một luồng có thể bắt đầu một bước trong khi luồng bên cạnh vẫn đang ở bước trước; một __syncthreads() mỗi bước sẽ sửa được.">
            A thread can start a step while its neighbour is still on the last one; one __syncthreads() per step fixes it.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Các lần đổi chỗ phải là nguyên tử, vì hai luồng của cùng một bước vẫn có thể đổi chỗ cùng một phần tử.">
            The swaps must be atomic, because two threads of one step can still swap the same element.
          </Translated>
        ),
        why: (
          <Translated vi="Không có hai luồng nào ghi cùng một phần tử trong cùng một bước. Điều còn thiếu là thứ tự giữa các bước, thứ mà rào chắn cho được còn phép toán nguyên tử thì không.">
            No two threads write the same element in the same step. What is missing is an order between steps, which a barrier gives and an atomic does not.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="n bước là quá ít cho một phép sắp xếp cấp khối; nó cần cờ của mục 14.2 và một vòng lặp chạy đến khi không còn gì thay đổi.">
            n steps are too few for a block-level sort; it needs section 14.2's flag and a loop that runs until nothing changes.
          </Translated>
        ),
        why: (
          <Translated vi="Chính số bước cố định làm cho đây là một mạng sắp xếp (mục 14.9). Kernel đã sửa sắp xếp mọi trường hợp bài kiểm thử đưa ra trong n bước, kể cả cả 256 danh sách gồm tám số 0 và 1.">
            A fixed number of steps is what makes this a sorting network (section 14.9). The corrected kernel sorts every case the test gives it in n steps, including all 256 lists of eight 0s and 1s.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Các cặp của mỗi bước chồng lên các cặp của bước trước. Không có rào chắn thì ngôn ngữ không hứa gì về việc hai luồng của
              một khối cách nhau bao xa, nên luồng t có thể so sánh một cặp mà phần tử thứ hai luồng t + 1 chưa kịp đổi chỗ, hoặc cả
              hai có thể cùng lúc đổi chỗ phần tử 2t + 2 và làm mất một giá trị: tranh chấp các cặp kề nhau của mục 14.2, chuyển từ
              bên trong một bước sang giữa hai bước. Một khối 32 luồng là một warp trên NVIDIA, nơi các luồng rất có thể giữ nhịp với
              nhau; ngôn ngữ không hứa điều đó, và một bài kiểm thử chạy đạt ở đó không chứng minh được gì.
            </p>
            <p>
              Trên trình giả lập của khóa học, vốn chạy từng luồng một giữa các rào chắn, kernel như trên để lại 64, 129 và 256 khóa
              chưa sắp xếp trong cả ba thứ tự luồng. Với <code>__syncthreads()</code> là câu lệnh cuối của thân vòng lặp, nó sắp xếp
              được tất cả. Đó là chi phí của một mạng trong bộ nhớ chia sẻ: một rào chắn mỗi tầng.
            </p>
          </>
        }
      >
        <>
          <p>
            Every step's pairs overlap the previous step's. Without a barrier the language promises nothing about how far
            apart two threads of a block are, so thread t can compare a pair whose second element thread t + 1 has not yet
            swapped, or both can swap element 2t + 2 at the same moment and lose a value: section 14.2's neighbouring-pairs
            race, moved from inside one step to between two. A block of 32 threads is one warp on NVIDIA, where the threads
            may well keep in step; the language does not promise it, and a test that passes there proves nothing.
          </p>
          <p>
            On the course's emulator, which runs one thread at a time between barriers, the kernel as shown leaves 64, 129 and
            256 keys unsorted in all three thread orders. With <code>__syncthreads()</code> as the last statement of the loop
            body it sorts them all. That is the cost of a network in shared memory: a barrier per stage.
          </p>
        </>
      </Translated>
    ),
    source: 'our kernel, after §14.2, pp. 331–333, and §14.9, p. 345; barriers: ch. 5 §5.4, pp. 110–111',
    vi: { source: 'kernel của khóa học này, dựa theo §14.2, pp. 331–333, và §14.9, p. 345; rào chắn: ch. 5 §5.4, pp. 110–111' },
  },
  'ch14/wrap/q-atomic-scatter': {
    kind: 'choice',
    id: 'ch14/wrap/q-atomic-scatter',
    prompt: (
      <Translated
        vi={
          <>
            Để bỏ qua phép scan trong một lượt radix một bit, ai đó giữ hai bộ đếm trong bộ nhớ toàn cục, một cho bucket 0 bắt đầu
            từ 0 và một cho bucket 1 bắt đầu từ số bit 0, rồi để mỗi luồng lấy đích đến cho khóa của mình bằng{' '}
            <code>atomicAdd</code> trên bộ đếm của bucket tương ứng. Sau cả 32 lượt, đầu ra chứa gì?
          </>
        }
      >
        <>
          To skip the scan in a one-bit radix pass, someone keeps two counters in global memory, one for the zero bucket
          starting at 0 and one for the one bucket starting at the number of zeros, and lets each thread take its key's
          destination with <code>atomicAdd</code> on its bucket's counter. After all 32 passes, what does the output hold?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Các khóa được sắp theo giá trị, nhưng các khóa bằng nhau không theo thứ tự được hứa nào: phép toán nguyên tử chỉ làm mất tính ổn định.">
            The keys sorted by value, but equal keys in no promised order: the atomics cost only stability.
          </Translated>
        ),
        why: (
          <Translated vi="Điều đó mô tả một lượt. Một phép sắp xếp LSD cần mọi lượt giữ thứ tự trước đó bên trong mỗi bucket, và thiếu điều đó thì các lượt sau phá đi công của các lượt trước.">
            That describes one pass. An LSD sort needs every pass to keep the previous order inside each bucket, and without that the later passes undo the earlier ones.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Mỗi khóa đúng một lần, nhưng nói chung không được sắp xếp: phép toán nguyên tử điền mỗi bucket theo thứ tự đến, vứt bỏ công của các lượt trước.">
            Every key once, but in general not sorted: atomics fill each bucket in arrival order, discarding the earlier passes' work.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Khóa bị mất hoặc bị nhân đôi: hai luồng của một bucket có thể đọc cùng một giá trị bộ đếm trước khi luồng nào cộng, và cả hai ghi vào ô đó.">
            Keys lost or duplicated: two threads of one bucket can read the same counter value before either adds, and both write that slot.
          </Translated>
        ),
        why: (
          <Translated vi="atomicAdd trả về một giá trị cũ khác nhau cho mỗi luồng, nên mỗi ô được lấy đúng một lần.">
            atomicAdd returns a different old value to every thread, so every slot is taken exactly once.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các khóa được sắp xếp, như với phép scan, chỉ chậm hơn: bộ đếm của mỗi bucket tuần tự hóa các luồng của nó.">
            The keys sorted, as with the scan, only slower: each bucket's counter serialises its threads.
          </Translated>
        ),
        why: (
          <Translated vi="Phép scan cho mỗi khóa ô giữ đúng thứ tự đầu vào; phép toán nguyên tử cho nó một ô theo thứ tự đến. Cùng bucket, khác vị trí trong đó.">
            The scan gives each key the slot that keeps input order; the atomic gives it a slot in arrival order. Same bucket, different place in it.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sau lượt cuối, các khóa được chia đúng theo bit 31, và đó là tất cả những gì được bảo đảm. Bên trong mỗi nửa, thứ tự lẽ
            ra phải là thứ tự mà 31 lượt đầu đã xây, và phép toán nguyên tử đã thay nó bằng thứ tự đến. Tính ổn định không phải thứ
            trang trí trong radix sort LSD: nó là thứ mang công của mỗi lượt sang lượt kế tiếp (mục 14.4). Đầu ra vẫn là một hoán
            vị, vì vậy một bài kiểm thử chỉ kiểm tra "mọi khóa đều có mặt" sẽ cho nó qua.
          </p>
        }
      >
        <p>
          After the last pass the keys are split correctly by bit 31, and that is all that is guaranteed. Inside each half
          the order should be the one the first 31 passes built, and the atomics replaced it with arrival order. Stability is
          not a nicety in an LSD radix sort: it is what carries each pass's work into the next (section 14.4). The output is
          still a permutation, which is why a test that checks only "every key is there" would pass it.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.4, pp. 335–336, and §14.5, pp. 337–338; atomics: ch. 9',
    vi: { source: 'ch. 14 §14.4, pp. 335–336, và §14.5, pp. 337–338; phép toán nguyên tử: ch. 9' },
  },
  'ch14/wrap/q-packed-rows': {
    kind: 'numeric',
    id: 'ch14/wrap/q-packed-rows',
    prompt: (
      <Translated
        vi={
          <>
            Sắp xếp mọi hàng của một batch {ROWS} hàng bằng một radix sort LSD thông thường: cho mỗi phần tử một khóa 64 bit có 32 bit
            thấp là khóa giảm dần của xác suất và các bit kế tiếp chứa id hàng. Phép sắp xếp dùng chữ số 8 bit và bỏ qua các bit cao
            không khóa nào dùng. Nó thực hiện bao nhiêu lượt?
          </>
        }
      >
        <>
          Sort every row of a batch of {ROWS} rows with one ordinary LSD radix sort: give each element a 64-bit key whose low
          32 bits are its probability's descending key and whose next bits hold its row id. The sort uses 8-bit digits and
          skips the high bits no key uses. How many passes does it make?
        </>
      </Translated>
    ),
    answer: PACKED,
    unit: 'passes',
    vi: {
      unit: 'lượt',
      source: 'phép tính của khóa học này, dựa theo §14.1, p. 330, và §14.7, pp. 342–344; khoảng bit của CUB: notes/today/ch14-facts.md A2',
    },
    explain: (
      <Translated
        vi={
          <p>
            {ROWS} hàng cần {rowBits(ROWS)} bit id hàng, nên các khóa dùng 32 + {rowBits(ROWS)} = {32 + rowBits(ROWS)} bit, và chữ
            số 8 bit cần ceil({32 + rowBits(ROWS)} / 8) = {PACKED} lượt. Thứ tự LSD làm phần còn lại: bốn lượt đầu sắp theo xác suất,
            lượt cuối theo hàng, và tính ổn định mang thứ tự xác suất vào từng hàng. Đó là chuỗi sắp xếp của mục 14.1, khóa phụ trước
            và khóa chính sau, bên trong một phép sắp xếp. Sắp xếp riêng từng hàng tốn 4 lượt; đóng gói tốn thêm một lượt và khóa
            rộng gấp đôi, đổi lại tiết kiệm được một phép sắp xếp phân đoạn. Radix sort của CUB nhận <code>begin_bit</code> và{' '}
            <code>end_bit</code>, nên một khóa 64 bit đóng gói như thế này có thể được sắp xếp chỉ trên {32 + rowBits(ROWS)} bit
            thấp của nó (A2).
          </p>
        }
      >
        <p>
          {ROWS} rows need {rowBits(ROWS)} bits of row id, so the keys use 32 + {rowBits(ROWS)} = {32 + rowBits(ROWS)} bits,
          and 8-bit digits need ceil({32 + rowBits(ROWS)} / 8) = {PACKED} passes. LSD order does the rest: the first four
          passes sort by probability, the last by row, and stability carries the probability order into each row. That is
          section 14.1's cascade, secondary key first and primary key last, inside one sort. Sorting each row on its own
          takes 4 passes; packing costs a pass and keys twice as wide, and saves a segmented sort. CUB's radix sort takes{' '}
          <code>begin_bit</code> and <code>end_bit</code>, so a packed 64-bit key like this one can be sorted on its low{' '}
          {32 + rowBits(ROWS)} bits only (A2).
        </p>
      </Translated>
    ),
    source: 'our arithmetic, after §14.1, p. 330, and §14.7, pp. 342–344; CUB bit range: notes/today/ch14-facts.md A2',
  },
  'ch14/wrap/q-select-bin': {
    kind: 'numeric',
    id: 'ch14/wrap/q-select-bin',
    prompt: (
      <Translated
        vi={
          <>
            Một radix select tìm {SELECT_K} khóa lớn nhất của một hàng. Histogram 8 bit cao nhất của nó, đọc từ bin trên cùng xuống,
            chứa {SELECT_BINS.join(', ')}, ... khóa. Nó còn phải chọn bao nhiêu khóa bên trong bin chứa khóa lớn thứ {SELECT_K}?
          </>
        }
      >
        <>
          A radix select looks for the {SELECT_K} largest keys of a row. Its histogram of the top 8 bits, read from the top
          bin down, holds {SELECT_BINS.join(', ')}, ... keys. How many keys must it still pick from inside the bin where the{' '}
          {SELECT_K}th largest key lies?
        </>
      </Translated>
    ),
    answer: SEL.still,
    unit: 'keys',
    vi: { unit: 'khóa', source: 'phép tính của khóa học này, dựa theo §14.9, p. 346; llama.cpp: notes/today/ch14-facts.md D2' },
    explain: (
      <Translated
        vi={
          <p>
            Tổng chạy từ trên xuống: {SELECT_BINS.slice(0, SEL.bin + 1).map((_, b) => SELECT_BINS.slice(0, b + 1).reduce((x, y) => x + y, 0)).join(', ')}.
            Khóa lớn thứ {SELECT_K} nằm ở bin thứ {SEL.bin + 1} tính từ trên. Mọi khóa của các bin phía trên nó đều thuộc đáp án,{' '}
            {SEL.above} khóa, và không khóa nào phía dưới thuộc đáp án, nên phép chọn giữ chúng mà không cần sắp thứ tự và chỉ đi
            tiếp sang chữ số kế tiếp bên trong bin đó, cho {SEL.still} khóa còn thiếu: xem lại {SELECT_BINS[SEL.bin]} khóa thay vì cả
            hàng. Đó là ý tưởng chữ số có nghĩa lớn nhất của mục 14.9 áp dụng cho phép chọn; <code>top_k_radix_cuda</code> của
            llama.cpp duyệt các bin 8 bit từ trên xuống theo cùng cách.
          </p>
        }
      >
        <p>
          Running totals from the top: {SELECT_BINS.slice(0, SEL.bin + 1).map((_, b) => SELECT_BINS.slice(0, b + 1).reduce((x, y) => x + y, 0)).join(', ')}. The{' '}
          {SELECT_K}th largest key is in bin {SEL.bin + 1} from the top. Every key of the bins above it is in the answer,{' '}
          {SEL.above} of them, and no key below it is, so the select keeps those without ever ordering them and goes on to the
          next digit only inside that bin, for the {SEL.still} still missing: {SELECT_BINS[SEL.bin]} keys looked at again
          instead of the whole row. That is section 14.9's most-significant-digit idea applied to selection; llama.cpp's{' '}
          <code>top_k_radix_cuda</code> walks 8-bit bins from the top the same way.
        </p>
      </Translated>
    ),
    source: 'our arithmetic, after §14.9, p. 346; llama.cpp: notes/today/ch14-facts.md D2',
  },
  'ch14/wrap/q-topp-head': {
    kind: 'choice',
    id: 'ch14/wrap/q-topp-head',
    prompt: (
      <Translated
        vi={
          <>
            Một bộ lấy mẫu top-p với p = 0.9 phải cắt mọi hàng đúng chỗ mà một phép sắp xếp đầy đủ sẽ cắt. Trên hàng trước mặt bạn,
            200 token có khả năng cao nhất đã chiếm 0.95 xác suất. Khối lượng sắp thứ tự ít nhất mà bộ lấy mẫu luôn có thể dựa vào
            là gì?
          </>
        }
      >
        <>
          A top-p sampler with p = 0.9 must cut every row exactly where a full sort would. On the row in front of you, the
          200 most likely tokens already hold 0.95 of the probability. What is the least ordering work the sampler can
          always rely on?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Sắp xếp toàn bộ hàng: điểm cắt được định nghĩa trên thứ tự đã sắp, nên phải biết hạng của mọi token.">
            A full sort of the row: the cut is defined on the sorted order, so every token's rank must be known.
          </Translated>
        ),
        why: (
          <Translated vi="Chỉ phần đầu của thứ tự đó là quan trọng. Các token sau điểm cắt bị bỏ dù thứ tự của chúng ra sao.">
            Only the head of that order matters. Tokens past the cut are dropped whatever their order.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chọn một tập chắc chắn chứa điểm cắt, ở đây là 200 giá trị lớn nhất, và chỉ sắp xếp chúng; nới rộng tập nếu tổng của chúng chưa đạt p.">
            Select a set sure to contain the cut, here the 200 largest, and sort only those; widen it if their sum falls short of p.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không cần gì: cộng dồn xác suất theo thứ tự token cho đến khi tổng chạy đạt 0.9, rồi cắt ở đó.">
            None: add up probabilities in token order until the running sum reaches 0.9, then cut there.
          </Translated>
        ),
        why: (
          <Translated vi="Cách đó giữ các token đầu tiên theo id, không phải các token có khả năng cao nhất.">
            That keeps the first tokens by id, not the most likely ones.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Sắp xếp 200 giá trị lớn nhất trên mọi hàng và bỏ qua bước kiểm tra: hàng này cho thấy 200 token chiếm hơn p.">
            Sort the 200 largest on every row and skip the check: this row shows that 200 tokens hold more than p.
          </Translated>
        ),
        why: (
          <Translated vi="Một hàng phẳng có thể trải 0.9 ra hàng nghìn token; chỉ tổng chạy mới cho biết phần đầu đã đủ dài hay chưa.">
            A flat row can spread 0.9 over thousands of tokens; only the running sum shows whether the head was long enough.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Điểm cắt là tiền tố ngắn nhất của thứ tự giảm dần có tổng đạt p, nên chỉ tiền tố đó cần có thứ tự, và chỉ cần biết rằng
            không có gì ngoài các ứng viên thuộc về nó. Top-p phía host của llama.cpp làm đúng như vậy: với hơn 1,024 ứng viên, nó
            sắp xếp một phần 256 giá trị đầu, lấy tổng chạy, và chỉ nới rộng phép sắp xếp nếu chưa đạt p. Một bộ lấy mẫu trên GPU
            có thể làm tương tự bằng một radix select và một phép sắp xếp các ứng viên.
          </p>
        }
      >
        <p>
          The cut is the shortest prefix of the descending order whose sum reaches p, so only that prefix needs an order, and
          only the knowledge that nothing outside the candidates belongs in it. llama.cpp's host top-p does exactly this: with
          more than 1,024 candidates it partially sorts the top 256, takes the running sum, and widens the sort only if p was
          not reached. A GPU sampler can do the same with a radix select and a sort of the candidates.
        </p>
      </Translated>
    ),
    source: 'our reasoning; llama.cpp: notes/today/ch14-facts.md D3',
    vi: { source: 'lập luận của khóa học này; llama.cpp: notes/today/ch14-facts.md D3' },
  },
};

// ---------------------------------------------------------------- milestone M5 so far
const M5: { lab: string; num: string; ch: string; what: string }[] = [
  { lab: 'ch09/lab3_tokencounts', num: '9.3', ch: 'ch09', get what() { return tr('token counts for the repetition penalty', 'số đếm token cho repetition penalty'); } },
  { lab: 'ch11/lab6_topp', num: '11.6', ch: 'ch11', get what() { return tr('top-p keep mask over a sorted row', 'mặt nạ giữ top-p trên một hàng đã sắp xếp'); } },
  { lab: 'ch12/lab4_candidates', num: '12.4', ch: 'ch12', get what() { return tr('candidate compaction', 'nén gọn (compaction) ứng viên'); } },
  { lab: 'ch14/lab3_sortrows', num: '14.3', ch: 'ch14', get what() { return tr('sorted candidates, every row descending and stable', 'ứng viên đã sắp xếp, mọi hàng giảm dần và ổn định'); } },
];
const where = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
/** Backends (devices first, then the emulator) on which the reader's own work of a lab passes. */
export const passingOn = (reports: Reports, lab: string) =>
  [...realBackends(reports), 'emu'].filter((b) => labStatus(reports[b], lab, 'work') === 'pass');

export function Wrap() {
  const { progress } = useProgress();
  const { reports, loading } = useReports();
  const snip = useLabFile(BLOCKSORT_FILE);
  const kernel = snip && !snip.startsWith('// could not load') ? functionSource(snip, 'blockOddEvenSort')?.text ?? null : null;
  const ran = Object.keys(reports).length > 0;
  const own143 = passingOn(reports, 'ch14/lab3_sortrows');
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch14/') && !k.includes('/p-'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>
        <Translated vi="Những gì mang theo">What to carry forward</Translated>
      </h2>
      <ul>
        <Translated
          vi={
            <li>
              Một phép sắp xếp nợ bạn hai thứ: một thứ tự và một hoán vị. Một phép sắp xếp ổn định còn giữ các khóa bằng nhau theo thứ
              tự đầu vào, đó là điều giúp sắp xếp theo hai khóa hoạt động và cố định thứ tự của bộ lấy mẫu giữa các xác suất bằng nhau.
            </li>
          }
        >
          <li>
            A sort owes you two things: an order and a permutation. A stable sort also keeps equal keys in input order, which
            is what makes a sort by two keys work and what fixes a sampler's order among tied probabilities.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Odd-even sort sắp thứ tự các cặp rời nhau, luân phiên chẵn và lẻ: O(N) bước và khối lượng công việc O(N²), một lần khởi
              chạy mỗi bước và một cờ mà host phải đọc. Là một vòng lặp cố định bên trong một luồng hay một khối, nó là mạng sắp xếp
              đơn giản nhất, và trong một khối nó cần một rào chắn mỗi bước.
            </li>
          }
        >
          <li>
            Odd-even sort orders disjoint pairs, alternating even and odd: O(N) steps and O(N²) work, one launch per step and a
            flag the host must read. As a fixed loop inside a thread or a block it is the simplest sorting network, and in a
            block it needs a barrier per step.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Merge sort sắp xếp các đoạn, rồi trộn chúng theo một cây. Các tầng đầu chạy nhiều phép trộn song song; các tầng sau chia
              vài phép trộn lớn cho các khối bằng phép tìm co-rank của chương 13. Nó ổn định nếu mọi phép trộn lấy từ dãy bên trái khi
              bằng nhau.
            </li>
          }
        >
          <li>
            Merge sort sorts segments, then merges them in a tree. Early stages run many merges side by side; late stages split
            a few large merges between blocks with chapter 13's co-rank search. It is stable if every merge takes from the left
            run on ties.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Một lượt radix là một phép phân hoạch ổn định theo một chữ số. Vị trí của mỗi khóa đến từ một phép scan loại trừ trên các
              bit, với tổng ở chỉ số n; mọi lượt phải ổn định thì các lượt mới cộng dồn được; float được sắp xếp sau một phép biến đổi
              bit, và thứ tự giảm dần đến từ việc đảo bit khóa, không phải lật ngược kết quả.
            </li>
          }
        >
          <li>
            A radix pass is a stable partition by one digit. Each key's place comes from an exclusive scan of the bits, with
            the total at index n; every pass must be stable for the passes to add up; floats sort after a bit transform, and
            descending order comes from inverting the key, not reversing the result.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Phép scan ở giữa kernel của sách không có dạng khả chuyển. Hãy cắt mỗi lượt thành các lần khởi chạy (trích hoặc đếm,
              scan, phân tán); Metal không có cách nào khác, còn CUDA và HIP chỉ có cách khác khi dùng lần khởi chạy cooperative.
            </li>
          }
        >
          <li>
            The book's scan in the middle of a kernel has no portable form. Cut each pass into launches (extract or count,
            scan, scatter); Metal has no other way, and CUDA and HIP only under a cooperative launch.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Phân tán trực tiếp ghi tối đa 2^r đoạn mỗi warp. Sắp xếp cục bộ khóa của mỗi khối trong bộ nhớ chia sẻ trước, với một
              bảng kích thước bucket bố trí theo chữ số được scan để ra điểm bắt đầu, làm các lần ghi được hợp nhất. Chữ số rộng hơn
              nghĩa là ít lượt hơn nhưng nhiều đoạn hơn và bảng lớn hơn; nhiều khóa mỗi luồng hơn nghĩa là đoạn dài hơn và bảng nhỏ
              hơn, cho đến khi hết bộ nhớ chia sẻ.
            </li>
          }
        >
          <li>
            A direct scatter writes up to 2^r runs per warp. Sorting each block's keys locally in shared memory first, with a
            digit-major table of bucket sizes scanned for their starts, makes the writes coalesce. Wider digits mean fewer passes
            but more runs and a bigger table; more keys per thread mean longer runs and a smaller table, until shared memory
            runs out.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Nơi nào có phép sắp xếp của thư viện, hãy dùng nó: CUB ghi trong tài liệu các radix sort ổn định, giảm dần, phân đoạn, và
              hipCUB có cùng các lời gọi trên nền rocPRIM, mà radix sort của nó được ghi là ổn định. Metal không có phép sắp xếp ổn định
              ở cấp kernel. Top-k là một phép chọn, không phải sắp xếp.
            </li>
          }
        >
          <li>
            Where a library sort exists, use it: CUB documents stable, descending, segmented radix sorts, and hipCUB has the
            same calls over rocPRIM, whose radix sort is documented as stable. Metal has no stable sort at kernel level. Top-k
            is a selection, not a sort.
          </li>
        </Translated>
      </ul>
      <Source section="14.10" pages="346–347" />

      <h2>
        <Translated vi="Bài kiểm tra tổng hợp">A mixed check</Translated>
      </h2>
      <p>
        <Translated vi="Các câu này dựa trên toàn chương và trên những tình huống mà các mục chưa cho thấy.">
          These draw on the whole chapter and on situations the sections did not show.
        </Translated>
        {asked
          ? tr(
              ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.`,
              ` Đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu hỏi, tính theo câu trả lời gần nhất của mỗi câu.`,
            )
          : ''}
      </p>
      {kernel ? (
        <Code
          tabs={[
            {
              label: tr('CUDA (not a lab file: for this question only)', 'CUDA (không phải tệp lab: chỉ dành cho câu hỏi này)'),
              code: kernel,
              tiers: TIERS,
              status: tr('front-end-checked; emulator-checked; unverified on CUDA and HIP hardware', 'front-end-checked; emulator-checked; unverified trên phần cứng CUDA và HIP'),
            },
          ]}
          caption={
            <Translated
              vi={
                <>
                  Từ <code>labs/{BLOCKSORT_FILE}</code>. Một khối sắp xếp tối đa 1,024 khóa trong bộ nhớ chia sẻ với n bước odd-even cố
                  định: không cờ và không vòng lặp host.
                </>
              }
            >
              <>
                From <code>labs/{BLOCKSORT_FILE}</code>. One block sorts up to 1,024 keys in shared memory with n fixed
                odd-even steps: no flag and no host loop.
              </>
            </Translated>
          }
        />
      ) : null}
      <Ask q={questionsWrap['ch14/wrap/q-blocksort']} label={tr('Find the defect', 'Tìm lỗi')} />
      <Ask q={questionsWrap['ch14/wrap/q-atomic-scatter']} />
      <Ask q={questionsWrap['ch14/wrap/q-packed-rows']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsWrap['ch14/wrap/q-select-bin']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsWrap['ch14/wrap/q-topp-head']} />

      <h2>
        <Translated vi="Bài tập của sách">The book's exercises</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương kết thúc bằng bốn bài tập lập trình (p. 347). Ba bài đầu là một lab ở đây; bài thứ tư được phác thảo và dựa trên
            phép trộn của chương 13.
          </p>
        }
      >
        <p>
          The chapter ends with four coding exercises (p. 347). The first three are one lab here; the fourth is sketched and
          rests on chapter 13's merge.
        </p>
      </Translated>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="exercises">
        <thead>
          <tr>
            <th><Translated vi="Bài tập">Exercise</Translated></th>
            <th><Translated vi="Về">About</Translated></th>
            <th><Translated vi="Được giải ở đâu tại đây">Where it is worked here</Translated></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>1</td>
            <td><Translated vi="Hình 14.7 với bộ nhớ chia sẻ, để các lần ghi được hợp nhất">Fig. 14.7 with shared memory, for coalesced writes</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    trình khám phá warp và bảng bucket ở <a href={href('ch14', '14.6')}>14.6</a>; lab 14.2 với r = 1 trên{' '}
                    <a href={href('ch14', 'labs')}>trang lab</a>
                  </>
                }
              >
                the warp explorer and bucket table in <a href={href('ch14', '14.6')}>14.6</a>; lab 14.2 at r = 1 on the{' '}
                <a href={href('ch14', 'labs')}>labs page</a>
              </Translated>
            </td>
          </tr>
          <tr>
            <td>2</td>
            <td><Translated vi="Radix nhiều bit">A multi-bit radix</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    trình khám phá độ rộng radix ở <a href={href('ch14', '14.7')}>14.7</a>; lab 14.2, r từ 1 đến 8
                  </>
                }
              >
                the radix-width explorer in <a href={href('ch14', '14.7')}>14.7</a>; lab 14.2, r from 1 to 8
              </Translated>
            </td>
          </tr>
          <tr>
            <td>3</td>
            <td><Translated vi="Gộp việc cho luồng (thread coarsening)">Thread coarsening</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    Hình 14.13 và trình khám phá số khóa mỗi luồng ở <a href={href('ch14', '14.8')}>14.8</a>; lab 14.2, c từ 1 đến 8
                  </>
                }
              >
                Fig. 14.13 and the keys-per-thread explorer in <a href={href('ch14', '14.8')}>14.8</a>; lab 14.2, c from 1 to 8
              </Translated>
            </td>
          </tr>
          <tr>
            <td>4</td>
            <td><Translated vi="Merge sort với phép trộn song song của chương 13">Merge sort with chapter 13's parallel merge</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    được phác thảo trong "Exercise 4, in outline" ở <a href={href('ch14', '14.3')}>14.3</a>, không được xây; bản thân phép
                    trộn là lab 13.1 trên <a href={href('ch13', 'labs')}>trang lab của chương 13</a>
                  </>
                }
              >
                sketched in "Exercise 4, in outline" on <a href={href('ch14', '14.3')}>14.3</a>, not built; the merge itself is
                lab 13.1 on <a href={href('ch13', 'labs')}>chapter 13's labs page</a>
              </Translated>
            </td>
          </tr>
        </tbody>
      </table>

      <h2>
        <Translated vi="Chỗ PDF có vẻ sai">Where the PDF looks wrong</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bản sách mà khóa học này dựa vào có vài chỗ có vẻ sơ suất ở chương 14. Năm chỗ đầu nằm trên trang sách; bốn chỗ cuối là
            cách chúng tôi đọc những chỗ trang sách để mập mờ.
          </p>
        }
      >
        <p>
          The copy this course was built from has a few apparent slips in chapter 14. The first five are on the page; the
          last four are our reading of places the page leaves unclear.
        </p>
      </Translated>
      <ul data-list="errata">
        <Translated vi={<li>p. 329 nói bộ lọc ổn định "will be discussed further in Chapter 12" (sẽ được bàn thêm ở Chương 12), trong khi chương đó đứng trước chương này. p. 336 viết đúng.</li>}>
          <li>p. 329 says the stable filter "will be discussed further in Chapter 12", which comes before this chapter. p. 336 has it right.</li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 329 nói "transposition sort and sampling sort are also discussed" như các phương pháp khác. Mục 14.9 bàn về mạng sắp
              xếp, sample sort và radix sort MSD; odd-even transposition sort là thuật toán của chính mục 14.2.
            </li>
          }
        >
          <li>
            p. 329 says "transposition sort and sampling sort are also discussed" as other methods. Section 14.9 covers sorting
            networks, sample sort and MSD radix sort; odd-even transposition sort is section 14.2's own algorithm.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 337 giới thiệu công thức đích đến thứ hai "for keys mapping to the zero bucket". Công thức đó dành cho bucket 1, như
              hình ghi nhãn (<a href={href('ch14', '14.5')}>14.5</a>).
            </li>
          }
        >
          <li>
            p. 337 introduces the second destination formula "for keys mapping to the zero bucket". The formula is for the one
            bucket, as the figure labels it (<a href={href('ch14', '14.5')}>14.5</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Hình 14.7 đọc <code>bits[N]</code> làm tổng số bit 1, điều này cần một phép scan loại trừ trên N + 1 phần tử; Hình 14.6
              cho thấy N, và giá trị cuối của nó bằng tổng chỉ vì bit của khóa cuối bằng 0. Lab 14.1 scan n + 1 phần tử.
            </li>
          }
        >
          <li>
            Fig. 14.7 reads <code>bits[N]</code> as the number of ones in total, which needs an exclusive scan of N + 1 entries;
            Fig. 14.6 shows N, and its last value equals the total only because the last key's bit is 0. Lab 14.1 scans n + 1
            entries.
          </li>
        </Translated>
        <Translated vi={<li>p. 339: "indeed to right destination indices" phải là "to the right destination indices".</li>}>
          <li>p. 339: "indeed to right destination indices" should read "to the right destination indices".</li>
        </Translated>
        <Translated
          vi={
            <li>
              Cách chúng tôi đọc: <code>gridExclusiveScan</code> trong Hình 14.7 là mã giả. Nó không bao giờ được định nghĩa, và mọi
              khối phải ghi xong bit của mình trước nó và phải chờ nó kết thúc, điều mà một lần khởi chạy thông thường không sắp đặt
              được. Mục 14.5 và các lab tách lượt thành các lần khởi chạy.
            </li>
          }
        >
          <li>
            Our reading: <code>gridExclusiveScan</code> in Fig. 14.7 is pseudocode. It is never defined, and every block must
            have written its bits before it and must wait for its end, which an ordinary launch cannot arrange. Section 14.5 and
            the labs split the pass into launches.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Cách chúng tôi đọc: vòng lặp host của odd-even không được đưa ra, và "until no elements are swapped" nếu hiểu là "cho đến
              khi một bước không đổi chỗ gì" sẽ dừng <code>[1, 3, 2, 4]</code> khi chưa sắp xếp. Hai bước liên tiếp, một chẵn và một
              lẻ theo thứ tự bất kỳ, đều yên lặng, mới cho thấy danh sách đã được sắp xếp (<a href={href('ch14', '14.2')}>14.2</a>).
            </li>
          }
        >
          <li>
            Our reading: the odd-even host loop is not shown, and "until no elements are swapped" read as "until one step makes
            no swap" stops <code>[1, 3, 2, 4]</code> unsorted. Two steps in a row, one even and one odd in either order,
            both quiet, show that the list is sorted (<a href={href('ch14', '14.2')}>14.2</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Cách chúng tôi đọc: Hình 14.13 vẽ các luồng của mỗi khối trên một nửa số khóa của nó và không nói luồng nào sở hữu khóa
              nào; tám khóa cho một khối bốn luồng nghĩa là hai khóa mỗi luồng (<a href={href('ch14', '14.8')}>14.8</a>).
            </li>
          }
        >
          <li>
            Our reading: Fig. 14.13 draws each block's threads over half of its keys and does not say which thread owns which;
            eight keys per four-thread block means two per thread (<a href={href('ch14', '14.8')}>14.8</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Cách chúng tôi đọc: p. 334 cho mỗi tầng trộn thời gian O(log N) và khối lượng công việc O(N log N) "as seen in Chapter
              13", nhưng chương 13 nêu O(m + n) cho phép trộn tuần tự và khoảng log2 N bước cho một phép tìm co-rank, chưa bao giờ nêu
              cận cho cả một phép trộn song song (<a href={href('ch14', '14.3')}>14.3</a>).
            </li>
          }
        >
          <li>
            Our reading: p. 334 gives each merge stage O(log N) time and O(N log N) work "as seen in Chapter 13", but chapter 13
            states O(m + n) for the sequential merge and about log2 N steps for one co-rank search, never a bound for a whole
            parallel merge (<a href={href('ch14', '14.3')}>14.3</a>).
          </li>
        </Translated>
      </ul>
      <Translated vi={<p>Hai điều các đoạn mã để lại cho bạn, mà các bài học và lab sửa và kiểm thử:</p>}>
        <p>Two things the listings leave to you, which the lessons and labs correct and test:</p>
      </Translated>
      <ul data-list="gaps">
        <Translated
          vi={
            <li>
              Hình 14.2 kiểm tra bằng <code>i &lt; N - 1</code> trên một N không dấu, vốn quay vòng khi N bằng 0;{' '}
              <code>i + 1 &lt; N</code> thì không thể.
            </li>
          }
        >
          <li>
            Fig. 14.2 guards with <code>i &lt; N - 1</code> on an unsigned N, which wraps when N is 0; <code>i + 1 &lt; N</code>{' '}
            cannot.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Hình 14.2 đặt <code>hasChanged</code> bằng một lệnh ghi thường từ nhiều luồng. Chính sách cảnh báo rằng đây là một data
              race và là hành vi không xác định trong mô hình bộ nhớ C++; các bài học và lab dùng lệnh ghi nguyên tử.
            </li>
          }
        >
          <li>
            Fig. 14.2 sets <code>hasChanged</code> with a plain store from many threads. The book itself warns that this is a
            data race and undefined behaviour in the C++ memory model; the lessons and labs use an atomic store.
          </li>
        </Translated>
      </ul>
      <Aside>
        <Translated
          vi={
            <p>
              Đây là cách chúng tôi đọc một bản PDF, từ notes/ch14.md, "Unclear in the PDF". Nếu bản của bạn khác, hãy tin bản của bạn.
            </p>
          }
        >
          <p>
            These are our readings of one PDF, from notes/ch14.md, "Unclear in the PDF". If your copy differs, trust your copy.
          </p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Những gì chương này bỏ qua">What this chapter left out</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương 14 ở đây là một chương mức Standard: mỗi mục có một trang, và những phần sau của sách được giữ ngắn hoặc bỏ qua.
          </p>
        }
      >
        <p>
          Chapter 14 is a Standard chapter here: every section has a page, and these parts of the book were kept short or left
          out.
        </p>
      </Translated>
      <ul data-list="left-out">
        <Translated vi={<li>Merge sort của Cole, thời gian O(log N) và khối lượng công việc O(N log N) (p. 334), chỉ được nêu tên, bởi sách và bởi chúng tôi.</li>}>
          <li>Cole's merge sort, O(log N) time and O(N log N) work (p. 334), is named only, by the book and by us.</li>
        </Translated>
        <Translated
          vi={
            <li>
              Merge sort song song không được xây (bài tập 4, p. 347). Trang 14.3 phác thảo nó; phép sắp xếp Metal của MLX là một merge
              sort có dạng này để bạn đọc.
            </li>
          }
        >
          <li>
            Parallel merge sort is not built (exercise 4, p. 347). Page 14.3 sketches it; MLX's Metal sort is a merge sort of
            this shape to read.
          </li>
        </Translated>
        <Translated vi={<li>Mạng odd-even merge sort của Batcher (p. 345) được nêu tên; chỉ mạng bitonic được chạy từng bước, ở trang 14.9.</li>}>
          <li>Batcher's odd-even merge sort network (p. 345) is named; only the bitonic network is stepped, on page 14.9.</li>
        </Translated>
        <Translated
          vi={
            <li>
              Sample sort (p. 346) có một câu hỏi. Lời khuyên của sách dùng nó cho dữ liệu trải trên bộ nhớ của nhiều GPU được thuật
              lại như của sách và không được kiểm chứng.
            </li>
          }
        >
          <li>
            Sample sort (p. 346) gets one question. The book's advice to use it for data spread over several GPUs' memories is
            reported as the book's and not checked.
          </li>
        </Translated>
        <Translated vi={<li>Radix sort MSD (p. 346) được mô tả và dùng để giải thích vì sao top-k là một phép chọn; không có phép sắp xếp MSD nào được xây.</li>}>
          <li>MSD radix sort (p. 346) is described and used to explain why top-k is a selection; no MSD sort is built.</li>
        </Translated>
        <Translated
          vi={
            <li>
              Thrust (p. 347) được nêu tên và không bao giờ được chạy: cả trình giả lập lẫn Metal đều không build được nó, nên bảng thư
              viện ở 14.9 đưa ra tên các phương thức, unverified trên phần cứng ở đây.
            </li>
          }
        >
          <li>
            Thrust (p. 347) is named and never run: neither the emulator nor Metal builds it, so the library table on 14.9
            gives the method names, unverified on hardware here.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Một lượt radix một kernel với rào chắn toàn lưới dưới lần khởi chạy cooperative không được xây. Thiết kế một lượt
              OneSweep mà sách dẫn (p. 336) cũng vậy: Metal không ghi nhận trong tài liệu bảo đảm tiến triển nào giữa các threadgroup.
            </li>
          }
        >
          <li>
            A one-kernel radix pass with a grid-wide barrier under a cooperative launch is not built. Neither is the
            single-pass OneSweep design the book cites (p. 336): Metal documents no forward progress between threadgroups.
          </li>
        </Translated>
        <Translated vi={<li>Các tài liệu tham khảo của chương ngoài OneSweep (pp. 347–348) không được bàn đến.</li>}>
          <li>The chapter's references other than OneSweep (pp. 347–348) are not discussed.</li>
        </Translated>
      </ul>
      <Beyond>
        <Translated
          vi={
            <p>
              Giữ ngắn những phần nào là lựa chọn của chúng tôi, theo mức của chương này trong PLAN.md. Dữ kiện về Metal nằm trong
              notes/today/ch14-facts.md C4.
            </p>
          }
        >
          <p>
            Which parts to keep short is our choice, from PLAN.md's tier for this chapter. The Metal fact is in
            notes/today/ch14-facts.md C4.
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Dự án đến lúc này">The project so far</Translated>
      </h2>
      <p>
        <Translated
          vi={
            <>
              Các hàng đã sắp xếp của lab 14.3 là các ứng viên đã sắp xếp của mốc dự án M5, lấy mẫu trên GPU: với mỗi hàng của các
              xác suất <code>[batch × vocab]</code>, các id token theo thứ tự lớn nhất trước, bằng nhau thì theo id tăng dần, và các
              xác suất theo thứ tự đó.{' '}
            </>
          }
        >
          Lab 14.3's sorted rows are the sorted candidates of milestone M5, sampling on the GPU: for every row of{' '}
          <code>[batch × vocab]</code> probabilities, the token ids largest first, ties by increasing id, and the
          probabilities in that order.{' '}
        </Translated>
        <span data-lab143={loading ? 'loading' : own143.join(',') || 'none'}>
          {loading
            ? tr('Reading reports.', 'Đang đọc báo cáo.')
            : !ran
              ? tr('No lab results are recorded on this machine yet.', 'Chưa có kết quả lab nào được ghi trên máy này.')
              : own143.length
                ? tr(`Your own lab 14.3 passes on ${own143.map(where).join(' and ')} on this machine.`, `Lab 14.3 của bạn chạy đạt trên ${own143.map(where).join(' và ')} trên máy này.`)
                : tr('Your own lab 14.3 does not pass on any backend of this machine yet.', 'Lab 14.3 của bạn chưa chạy đạt trên backend nào của máy này.')}
        </span>{' '}
        <Translated vi={<>Xem <a href={href('ch14', 'labs')}>các lab</a>.</>}>
          See the <a href={href('ch14', 'labs')}>labs</a>.
        </Translated>
      </p>
      <table className="data" data-table="m5">
        <thead>
          <tr>
            <th><Translated vi="Phần của M5">Piece of M5</Translated></th>
            <th><Translated vi="Lab">Lab</Translated></th>
            <th><Translated vi="Bài làm của bạn chạy đạt trên">Your own work passes on</Translated></th>
          </tr>
        </thead>
        <tbody>
          {M5.map((m) => {
            const on = passingOn(reports, m.lab);
            return (
              <tr key={m.lab} data-m5={m.num} data-pass={loading ? 'loading' : on.join(',') || 'none'}>
                <td>{m.what}</td>
                <td className="nowrap">
                  <a href={href(m.ch, 'labs')}><Translated vi={<>lab {m.num}</>}>lab {m.num}</Translated></a>
                </td>
                <td>{loading ? tr('reading reports', 'đang đọc báo cáo') : on.length ? on.map(where).join(', ') : <span className="muted"><Translated vi="chưa có gì">nothing yet</Translated></span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Phần còn lại của M5 là chính bộ lấy mẫu: module X1 ghép các mảnh này, thay phép sắp xếp đầy đủ bằng một phép chọn cho
            top-k, và rút token. Trong sách, chương 15 đến tiếp theo và quay lại phép nhân ma trận, phép tính đằng sau mọi lớp tuyến
            tính, cho mốc dự án M3. Bắt đầu ở <a href={href('ch15')}>chương 15</a>.
          </p>
        }
      >
        <p>
          What is left of M5 is the sampler itself: module X1 joins these pieces, replaces the full sort with a selection for
          top-k, and draws the token. In the book, chapter 15 comes next and returns to matrix multiplication, the computation
          behind every linear layer, for milestone M3. Start at <a href={href('ch15')}>chapter 15</a>.
        </p>
      </Translated>
    </>
  );
}
