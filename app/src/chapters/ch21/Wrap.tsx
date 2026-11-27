import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Beyond, Source } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { benchRows, fmt, labStatus, num, realBackends, useReports, type BackendRun, type BenchRow } from '../../lib/reports';
import { href } from '../../lib/router';
import { warpAccess } from '../../sim/coalesce';
import { acceptedShare, counts, superRadius } from '../../sim/cutoff';
import { arithmetic, opsGather } from '../../sim/dcs';

// A kernel of our own, for one question. Checked copy: labs/ch21/snippets/bias_gather.cu
// (Wrap.snippets.test.ts ties the two and pins the numbers the other questions take from
// the models).
const BIAS = `// Ours: a logit bias in the gather form of Fig. 21.6. The list holds n
// (token id, bias) pairs, and a token may appear in it more than once.
__global__ void biasGather(float* logits, int vocab,
                           const int* ids, const float* bias, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's logit
    if (v >= vocab) return;
    float add = 0.0f;
    for (int e = 0; e < n; ++e)   // every thread scans every pair
        if (ids[e] == v) add += bias[e];
    logits[v] += add;             // one writer per logit: no atomic
}`;

// ---------------------------------------------------------------- the numbers the questions take from the models
/** The bias question's sizes (ours). */
export const VOCAB = 128_000;
export const PAIRS = 50;

/** Gather kernel, per point per launch: arithmetic per atom (dcs.ts, counted from Fig. 21.6), over 8 bytes of +=. */
export const OPS_PER_ATOM = arithmetic(opsGather(1)); // 10
export const CHUNK = 64;
export const RMW_BYTES = 8; // one float read and one written by the +=
export const INTENSITY = (CHUNK * OPS_PER_ATOM) / RMW_BYTES; // 80
export const INTENSITY_1024 = (1024 * OPS_PER_ATOM) / RMW_BYTES; // 1,280

/** Coarsening by 8 on an NVIDIA warp: sectors per write statement (warpAccess, chapter 6's model). */
const NVIDIA_WARP = 32;
const SECTOR = 32;
const F32 = 4;
export const FACTOR = 8;
const lanes = Array.from({ length: NVIDIA_WARP }, (_, t) => t);
export const SECTORS_ADJ_STMT = warpAccess(lanes.map((t) => FACTOR * t * F32), F32, SECTOR).transactions; // 32
export const SECTORS_STR_STMT = warpAccess(lanes.map((t) => t * F32), F32, SECTOR).transactions; // 4
export const SECTORS_ADJ_ALL = FACTOR * SECTORS_ADJ_STMT; // 256
export const SECTORS_STR_ALL = FACTOR * SECTORS_STR_STMT; // 32
/** Factor 4, for comparison: page 21.2's 16 per statement. */
export const SECTORS_ADJ_F4 = warpAccess(lanes.map((t) => 4 * t * F32), F32, SECTOR).transactions; // 16

/** The book's worked case (p. 525), 3D: 12 Å cutoff, 4 Å bins; then 2 Å bins. */
const CUTOFF = 12;
const listFor = (b: number) => counts(superRadius(CUTOFF, b, 3, 'half') / b, 3).total;
const pct = (v: number) => Math.round(100 * v);
export const BINS_4 = listFor(4); // 461
export const BINS_2 = listFor(2); // 1,863
export const SHARE_4 = pct(acceptedShare(CUTOFF, 4, 3)); // 25
export const SHARE_2 = pct(acceptedShare(CUTOFF, 2, 3)); // 49
export const SCANNED_4 = BINS_4 * 4 ** 3; // Å³ of bins a thread scans
export const SCANNED_2 = BINS_2 * 2 ** 3;

const n = (v: number) => v.toLocaleString('en-US');

export const questionsWrap: Record<string, Question> = {
  'ch21/wrap/q-sparse-bias': {
    kind: 'choice',
    id: 'ch21/wrap/q-sparse-bias',
    prompt: (
      <Translated
        vi={
          <>
            Kernel ở trên cộng một danh sách {PAIRS} cặp (id token, bias) vào {n(VOCAB)} logits trước khi lấy mẫu. Nó đúng,
            không cần phép toán nguyên tử và cho cùng các bit ở mọi lần chạy. Nếu có, bạn sẽ thay đổi điều gì?
          </>
        }
      >
        <>
          The kernel above adds a list of {PAIRS} (token id, bias) pairs to {n(VOCAB)} logits before sampling. It is
          correct, needs no atomic and gives the same bits on every run. What, if anything, would you change?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Không gì cả: một người ghi cho mỗi đầu ra và không phép toán nguyên tử chính là dạng mà chương này khuyên dùng thay cho scatter.">Nothing: one writer per output and no atomics is exactly the form this chapter recommends over scatter.</Translated>,
        why: tr(
          `It pays for every pair: ${n(VOCAB)} threads each test ${PAIRS} entries, ${n(VOCAB * PAIRS)} comparisons for ${PAIRS} useful additions. Gather won in this chapter because every atom contributes to every grid point.`,
          `Nó trả giá cho mọi cặp: ${n(VOCAB)} luồng, mỗi luồng kiểm tra ${PAIRS} mục, ${n(VOCAB * PAIRS)} phép so sánh cho ${PAIRS} phép cộng hữu ích. Gather thắng trong chương này vì mọi nguyên tử đều đóng góp vào mọi điểm lưới.`,
        ),
      },
      {
        label: tr(
          `One thread per pair, adding its bias with an atomic; if the bits must stay fixed, sort the pairs by id first.`,
          `Mỗi cặp một luồng, cộng bias của nó bằng một phép toán nguyên tử; nếu các bit phải giữ cố định, hãy sắp xếp các cặp theo id trước.`,
        ),
        correct: true,
      },
      {
        label: <Translated vi="Làm phép cộng cuối cùng thành nguyên tử: hai luồng tranh chấp trên một logit khi một token xuất hiện hai lần trong danh sách.">Make the final add atomic: two threads race on one logit when a token appears in the list twice.</Translated>,
        why: <Translated vi="Chỉ luồng v từng ghi logit v. Một token xuất hiện hai lần được cộng bên trong chính luồng đó.">Only thread v ever writes logit v. A token that appears twice is summed inside that one thread.</Translated>,
      },
      {
        label: <Translated vi="Chuyển ids và bias vào bộ nhớ chia sẻ: mọi luồng đọc chúng ở một địa chỉ khác nhau, nên các lần nạp không được hợp nhất.">Move ids and bias to shared memory: every thread reads them at a different address, so the loads are uncoalesced.</Translated>,
        why: <Translated vi="Ở mỗi bước, mọi lane của một warp đọc cùng ids[e] và bias[e]: một địa chỉ cho cả warp. Lần lưu vào logits là liên tục trên toàn warp.">At every step all the lanes of a warp read the same ids[e] and bias[e]: one address for the whole warp. The store to logits is contiguous across the warp.</Translated>,
      },
    ],
    explain: (
      <>
        <Translated
          vi={
            <p>
              Mỗi cặp một luồng tạo ra {PAIRS} cập nhật hiếm khi va chạm, so với {n(VOCAB * PAIRS)} phép so sánh. Chương ưu
              tiên gather vì trong tổng trực tiếp, mọi đầu vào đến được mọi đầu ra. Khi đó các lane của kernel scatter va chạm
              trên một điểm ở mỗi bước, còn gather làm cùng số cặp mà không cần một phép toán nguyên tử nào. Ở đây mỗi đầu vào
              chỉ đến một đầu ra. Kernel gather vẫn ghé mọi cặp, với một phép so sánh bằng thay cho 1/r, trong khi một scatter
              chỉ làm một cập nhật cho mỗi đầu vào và các phép toán nguyên tử của nó hầu như không bao giờ gặp nhau. Chương 22
              cũng phân biệt như vậy: một luồng lấy đầu vào làm trung tâm có thể cập nhật một đầu ra, một tập con hoặc tất cả.
            </p>
          }
        >
          <p>
            A thread per pair makes {PAIRS} updates that rarely collide, against {n(VOCAB * PAIRS)} comparisons. The
            chapter prefers gather because in direct summation every input reaches every output. The scatter kernel's
            lanes then collide on one point at every step, and gather does the same number of pairs without a single
            atomic. Here each input reaches one output. The gather kernel still visits every pair, with an equality test
            in place of 1/r, while a scatter does one update per input and its atomics almost never meet. Chapter 22 makes
            the same distinction: an input-centric thread may update one output, a subset or all of them.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Sắp xếp các cặp theo id là phân ngăn với mỗi token một ngăn: khi đó mỗi luồng tìm trong một đoạn ngắn thay vì cả
              danh sách, và thứ tự của tổng là cố định. Đó là gradient embedding CUDA của PyTorch (sắp xếp, rồi một người ghi
              cho mỗi hàng). PyTorch liệt kê <code>index_add</code> và <code>scatter_add_</code> trên CUDA trong số các phép
              toán không tất định trừ khi bạn yêu cầu thuật toán tất định. <code>get_rows_back</code> của llama.cpp là dạng
              gather ở trên, dùng cho huấn luyện.
            </p>
          }
        >
          <p>
            Sorting the pairs by id is binning with one bin per token: each thread then searches a short run instead of the
            whole list, and the sum's order is fixed. That is PyTorch's CUDA embedding gradient (sort, then one writer per
            row). PyTorch lists <code>index_add</code> and <code>scatter_add_</code> on CUDA among the operations that are
            nondeterministic unless you ask for deterministic algorithms. llama.cpp's <code>get_rows_back</code> is the
            gather form above, for training.
          </p>
        </Translated>
      </>
    ),
    source:
      'ch. 21 §21.2, pp. 517–519; ch. 22 §22.2, p. 532 (notes/ch22.md); the kernel and the bias setting are ours; notes/today/ch21-facts.md D1, D3',
    vi: {
      source:
        'ch. 21 §21.2, pp. 517–519; ch. 22 §22.2, p. 532 (notes/ch22.md); kernel và bối cảnh bias là của khóa học này; notes/today/ch21-facts.md D1, D3',
    },
  },
  'ch21/wrap/q-intensity': {
    kind: 'numeric',
    id: 'ch21/wrap/q-intensity',
    prompt: (
      <Translated
        vi={
          <>
            Đếm phép tính của kernel gather cho mỗi nguyên tử ở mỗi điểm từ Fig. 21.6: {OPS_PER_ATOM} phép toán, không tính
            căn bậc hai. Với một chunk {CHUNK} nguyên tử, một luồng thực hiện bao nhiêu phép toán đó cho mỗi byte lưu lượng
            bộ nhớ toàn cục trong một lần khởi chạy? Truy cập toàn cục duy nhất của nó là phép <code>+=</code> vào điểm của
            nó.
          </>
        }
      >
        <>
          Count the gather kernel's arithmetic per atom per point from Fig. 21.6: {OPS_PER_ATOM} operations, square root
          not counted. With a chunk of {CHUNK} atoms, how many of those operations does one thread do per byte of
          global-memory traffic in one launch? Its only global access is the <code>+=</code> into its point.
        </>
      </Translated>
    ),
    answer: INTENSITY,
    tol: 0.02,
    unit: 'operations per byte',
    vi: {
      unit: 'phép toán mỗi byte',
      source:
        'đếm từ Fig. 21.6, p. 518 (notes/today/ch21-facts.md E1, E3); ch. 5 §5.1, p. 94 (notes/ch05.md); lập luận là của khóa học này',
    },
    explain: (
      <Translated
        vi={
          <p>
            {CHUNK} × {OPS_PER_ATOM} = {n(CHUNK * OPS_PER_ATOM)} phép toán so với {RMW_BYTES} byte, một float đọc và một
            float ghi: {INTENSITY} mỗi byte. Ở chunk 1,024 con số là {n(INTENSITY_1024)}. Các lần đọc nguyên tử đi vào bộ
            nhớ đệm hằng, thứ mà sách nói loại bỏ phần lớn các truy cập DRAM của chúng. GPU ví dụ của chương 5, H100 của
            sách, cần khoảng 20 để ra khỏi vùng bị giới hạn bởi bộ nhớ (p. 94), nên sách đúng khi nói băng thông toàn cục
            không giới hạn kernel này (p. 519). Điều đó cũng có nghĩa là mẫu ghi, chủ đề của §21.4, chiếm một phần nhỏ thời
            gian, và càng nhỏ khi chunk càng lớn: cơ sở cho dự đoán của chúng tôi ở lab 21.1.
          </p>
        }
      >
        <p>
          {CHUNK} × {OPS_PER_ATOM} = {n(CHUNK * OPS_PER_ATOM)} operations against {RMW_BYTES} bytes, one float read and one
          written: {INTENSITY} per byte. At a chunk of 1,024 it is {n(INTENSITY_1024)}. The atom reads go to the constant
          cache, which the book says removes most of their DRAM accesses. Chapter 5's example GPU, the book's H100, needs
          about 20 to leave the memory-bound regime (p. 94), so the book is right that global bandwidth does not limit this
          kernel (p. 519). It also means the
          write pattern, the subject of §21.4, is a small share of the time, and a smaller one the larger the chunk: the
          basis of our prediction for lab 21.1.
        </p>
      </Translated>
    ),
    source: 'counts from Fig. 21.6, p. 518 (notes/today/ch21-facts.md E1, E3); ch. 5 §5.1, p. 94 (notes/ch05.md); the reasoning is ours',
  },
  'ch21/wrap/q-factor8': {
    kind: 'numeric',
    id: 'ch21/wrap/q-factor8',
    prompt: (
      <Translated
        vi={
          <>
            Gộp việc theo hệ số {FACTOR} với quyền sở hữu liền kề, chỉ số luồng được nhân hệ số: luồng <code>t</code> sở hữu
            các điểm <code>{FACTOR}t</code> đến <code>{FACTOR}t + {FACTOR - 1}</code> và ghi chúng bằng {FACTOR} câu lệnh.
            Trên một warp NVIDIA 32 lane, với hàng bắt đầu tại một ranh giới 128 byte, {FACTOR} câu lệnh ghi gộp lại chạm tới
            bao nhiêu sector 32 byte?
          </>
        }
      >
        <>
          Coarsen by {FACTOR} with adjacent ownership, the thread index scaled: thread <code>t</code> owns points{' '}
          <code>{FACTOR}t</code> to <code>{FACTOR}t + {FACTOR - 1}</code> and writes them with {FACTOR} statements. On an
          NVIDIA warp of 32 lanes, with the row starting on a 128-byte boundary, how many 32-byte sectors do the{' '}
          {FACTOR} write statements touch together?
        </>
      </Translated>
    ),
    answer: SECTORS_ADJ_ALL,
    unit: 'sectors',
    vi: {
      unit: 'sector',
      source: 'ch. 21 §21.4, pp. 521–522; sector 32 byte của NVIDIA (notes/today/ch06-facts.md A2; ch21-facts E2); hệ số 8 là của khóa học này',
    },
    explain: (
      <Translated
        vi={
          <p>
            Trong một câu lệnh, lane <code>t</code> ghi phần tử <code>base + {FACTOR}t + m</code>: các lane cạnh nhau cách
            nhau {FACTOR} float, 32 byte, nên mỗi lane có riêng một sector. {SECTORS_ADJ_STMT} sector mỗi câu lệnh,{' '}
            {SECTORS_ADJ_ALL} cho cả {FACTOR}. Ở hệ số 4, con số là {SECTORS_ADJ_F4} mỗi câu lệnh, hai lane một sector. Phần
            lãng phí tăng theo hệ số cho đến khi mỗi lane cần sector riêng, mức tối đa mà một câu lệnh 32 lane ghi float có
            thể chạm tới. Với bước nhảy của Fig. 21.10, mọi câu lệnh giữ ở mức {SECTORS_STR_STMT}, tổng cộng{' '}
            {SECTORS_STR_ALL}, bất kể hệ số.
          </p>
        }
      >
        <p>
          In one statement lane <code>t</code> writes element <code>base + {FACTOR}t + m</code>: neighbouring lanes are{' '}
          {FACTOR} floats, 32 bytes, apart, so each lane has a sector to itself. {SECTORS_ADJ_STMT} sectors per statement,{' '}
          {SECTORS_ADJ_ALL} for the {FACTOR}. At a factor of 4 it was {SECTORS_ADJ_F4} per statement, two lanes to a sector.
          The waste grows with the factor until every lane needs its own sector, which is the most a 32-lane statement of
          floats can touch. With Fig. 21.10's stride every statement stays at {SECTORS_STR_STMT}, {SECTORS_STR_ALL} in all,
          whatever the factor.
        </p>
      </Translated>
    ),
    source: 'ch. 21 §21.4, pp. 521–522; NVIDIA\'s 32-byte sectors (notes/today/ch06-facts.md A2; ch21-facts E2); factor 8 is ours',
  },
  'ch21/wrap/q-small-bins': {
    kind: 'choice',
    id: 'ch21/wrap/q-small-bins',
    prompt: (
      <Translated
        vi={
          <>
            Trong trường hợp mẫu của sách, ngăn 4 Å và bán kính cắt 12 Å, danh sách của mỗi khối có {n(BINS_4)} ngăn và
            khoảng {SHARE_4} % số nguyên tử mà một luồng kiểm tra qua được. Bạn giảm cạnh ngăn xuống một nửa còn 2 Å, giữ quy
            tắc của sách rằng các điểm của một khối là một ngăn. Điều gì xảy ra?
          </>
        }
      >
        <>
          In the book's worked case, 4 Å bins and a 12 Å cutoff, each block's list has {n(BINS_4)} bins and about{' '}
          {SHARE_4} % of the atoms a thread tests pass. You halve the bin side to 2 Å, keeping the book's rule that a block's
          points are one bin. What happens?
        </>
      </Translated>
    ),
    options: [
      {
        label: tr(
          `Fewer tests are wasted, since about ${SHARE_2} % of tested atoms pass, but the list grows to ${n(BINS_2)} bins and a block has 64 threads.`,
          `Ít phép kiểm tra bị lãng phí hơn, vì khoảng ${SHARE_2} % số nguyên tử được kiểm tra qua được, nhưng danh sách tăng lên ${n(BINS_2)} ngăn và một khối có 64 luồng.`,
        ),
        correct: true,
      },
      {
        label: <Translated vi="Không gì thay đổi: hình cầu bán kính cắt vẫn vậy, nên mọi luồng kiểm tra cùng các nguyên tử, chỉ được gom vào nhiều ngăn hơn.">Nothing changes: the cutoff sphere is the same, so every thread tests the same atoms, only grouped into more bins.</Translated>,
        why: tr(
          `A thread tests every atom in its list's bins, not in its sphere. Smaller bins follow the sphere more closely: ${n(SCANNED_2)} Å³ scanned instead of ${n(SCANNED_4)}.`,
          `Một luồng kiểm tra mọi nguyên tử trong các ngăn của danh sách, không phải trong hình cầu của nó. Ngăn nhỏ hơn bám sát hình cầu hơn: quét ${n(SCANNED_2)} Å³ thay vì ${n(SCANNED_4)}.`,
        ),
      },
      {
        label: <Translated vi="Ít nguyên tử được kiểm tra qua được hơn, vì giờ nhiều ngăn trong danh sách chỉ nằm một phần trong hình cầu bán kính cắt.">Fewer of the tested atoms pass, because more of the bins in the list now lie only partly inside the cutoff sphere.</Translated>,
        why: tr(
          `More bins are partial, but each is smaller, and the volume scanned falls from ${n(SCANNED_4)} to ${n(SCANNED_2)} Å³. The share that passes rises to about ${SHARE_2} %.`,
          `Nhiều ngăn nằm một phần hơn, nhưng mỗi ngăn nhỏ hơn, và thể tích được quét giảm từ ${n(SCANNED_4)} xuống ${n(SCANNED_2)} Å³. Tỷ lệ qua được tăng lên khoảng ${SHARE_2} %.`,
        ),
      },
      {
        label: <Translated vi="Danh sách ngắn lại, vì mỗi ngăn nhỏ hơn và chứa ít nguyên tử hơn, nên ít ngăn chạm tới hình cầu hơn.">The list gets shorter, because each bin is smaller and holds fewer atoms, so fewer bins reach the sphere.</Translated>,
        why: tr(
          `It takes more small bins to cover the same super sphere: ${n(BINS_2)} against ${n(BINS_4)}.`,
          `Cần nhiều ngăn nhỏ hơn để phủ cùng hình cầu lớn đó: ${n(BINS_2)} so với ${n(BINS_4)}.`,
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tính bằng mô hình bán kính cắt của chương này: bán kính hình cầu lớn là bán kính cắt cộng nửa đường chéo ngăn,
            nên ngăn nhỏ hơn thu hẹp phần đệm quanh hình cầu và phần được quét nằm trong nó nhiều hơn. Cái giá nằm ở chỗ khác.
            Mỗi mục trong danh sách là một lần nạp cộng tác cùng các rào chắn của nó, và số mục tăng khoảng bốn lần. Một khối
            64 luồng nằm ở đáy dải mà NVIDIA khuyến nghị. Và một ngăn có thể tích bằng một phần tám thì trung bình chứa một
            phần tám số nguyên tử, nên số lượng của chúng dao động nhiều hơn so với mức trung bình đó, và một sức chứa vừa với
            phần lớn các ngăn sẽ phải đệm nhiều hơn. Phân kỳ giảm còn chi phí phụ tăng; bên nào thắng là chuyện đo đạc.
          </p>
        }
      >
        <p>
          Computed with this chapter's cutoff model: the super-sphere radius is the cutoff plus half the bin diagonal, so
          smaller bins shrink the padding around the sphere and more of what is scanned is inside it. The price is
          elsewhere. Every list entry means a cooperative load and its barriers, and there are about four times as many. A block of
          64 threads is at the bottom of NVIDIA's suggested range. And a bin an eighth the volume holds an eighth as many
          atoms on average, so their counts vary more relative to that average, and a capacity that fits most bins pads
          more. Divergence falls and overhead rises; which wins is a measurement.
        </p>
      </Translated>
    ),
    source:
      'ch. 21 §21.5, pp. 525–527; src/sim/cutoff.ts (notes/today/ch21-facts.md E4); block size advice ch21-facts A4; halving the bins is ours',
    vi: {
      source:
        'ch. 21 §21.5, pp. 525–527; src/sim/cutoff.ts (notes/today/ch21-facts.md E4); khuyến nghị kích thước khối ch21-facts A4; việc giảm nửa ngăn là của khóa học này',
    },
  },
  'ch21/wrap/q-bitwise': {
    kind: 'choice',
    id: 'ch21/wrap/q-bitwise',
    prompt: (
      <Translated
        vi={
          <>
            Lab 21.1 so sánh kernel gather và kernel đã gộp việc của bạn với nhau trong phạm vi một dung sai, không phải từng
            bit. Không kernel nào dùng phép toán nguyên tử, và mỗi kernel cho cùng các bit ở mọi lần chạy. Vì sao không thể
            kỳ vọng hai kernel khớp nhau chính xác?
          </>
        }
      >
        <>
          Lab 21.1 compares your gather and coarsened kernels with each other within a tolerance, not bit for bit. Neither
          uses an atomic, and each gives the same bits on every run. Why can the two not be expected to agree exactly?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Chúng tính cùng các khoảng cách bằng những phép toán float khác nhau, và mỗi phép toán đều làm tròn.">
            They compute the same distances by different float operations, and each operation rounds.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Chúng cộng các nguyên tử của một chunk theo thứ tự khác nhau, nên các tổng đang chạy làm tròn khác nhau.">They add the atoms of a chunk in different orders, so their running sums round differently.</Translated>,
        why: <Translated vi="Cả hai đều đi qua mỗi chunk từ n = 0 trở lên, và cả hai đều được khởi chạy một lần cho mỗi chunk theo cùng thứ tự.">Both walk each chunk from n = 0 upwards, and both are launched once per chunk in the same order.</Translated>,
      },
      {
        label: <Translated vi="Kernel đã gộp việc cộng vào lưới một lần cho mỗi bốn điểm, còn gather một lần cho mỗi điểm.">The coarsened kernel adds into the grid once per four points, gather once per point.</Translated>,
        why: <Translated vi="Cả hai đều làm một phép += cho mỗi điểm mỗi lần khởi chạy: kernel đã gộp việc có bốn câu lệnh ghi, mỗi câu cho một điểm của nó.">Both make one += per point per launch: the coarsened kernel has four write statements, one for each of its points.</Translated>,
      },
      {
        label: <Translated vi="Ngay cả không có phép toán nguyên tử, các warp được lập lịch khác nhau ở mỗi lần chạy, nên các bit của một kernel có thể thay đổi.">Even without atomics, warps are scheduled differently on every run, so a kernel's bits can change.</Translated>,
        why: <Translated vi="Các phép cộng của chính một luồng diễn ra theo thứ tự chương trình. Không có phép toán nguyên tử và không luồng nào khác ghi điểm của nó, cùng một bản build trên cùng một thiết bị sẽ lặp lại các bit của nó.">A thread's own additions happen in program order. With no atomic and no other thread writing its point, the same build on the same device repeats its bits.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Kernel đã gộp việc có <code>dx1</code> đến <code>dx3</code> bằng cách cộng các bội số của{' '}
            <code>blockDim.x * gridspacing</code> vào <code>dx0</code>, trong khi gather trừ x của nguyên tử khỏi x của điểm
            của chính nó. <code>(x − a) + 64·s</code> và <code>(x + 64·s) − a</code> bằng nhau khi là số thực nhưng không phải
            lúc nào cũng bằng nhau khi là float, vì mỗi phép toán đều làm tròn (phụ lục A). Trên Metal, fast math còn thêm phép
            chia và căn bậc hai lỏng hơn của riêng nó. Vì vậy hai kernel đúng khớp nhau đến một dung sai, còn một bài kiểm tra
            tính tất định so một kernel với chính nó. Đó là ranh giới PyTorch vạch ra: chế độ tất định hứa cùng đầu ra cho cùng
            đầu vào trên cùng phần mềm và phần cứng, không phải cùng đầu ra với một thuật toán khác.
          </p>
        }
      >
        <p>
          The coarsened kernel gets <code>dx1</code> to <code>dx3</code> by adding multiples of{' '}
          <code>blockDim.x * gridspacing</code> to <code>dx0</code>, where gather subtracts the atom's x from its own
          point's x. <code>(x − a) + 64·s</code> and <code>(x + 64·s) − a</code> are equal as real numbers and not always as floats,
          because each operation rounds (appendix A). On Metal, fast math adds its own looser division and square root. So
          two correct kernels agree to a tolerance, while a determinism test compares a kernel with itself. That is the
          line PyTorch draws: deterministic mode promises the same output for the same input on the same software and
          hardware, not the same output as another algorithm.
        </p>
      </Translated>
    ),
    source:
      'Figs. 21.6 and 21.10, pp. 518, 522; appendix A §A.6; notes/today/ch10-facts.md A9, ch21-facts C3, D1; the lab\'s check is ours',
    vi: {
      source:
        'Figs. 21.6 và 21.10, pp. 518, 522; phụ lục A §A.6; notes/today/ch10-facts.md A9, ch21-facts C3, D1; phép kiểm tra của lab là của khóa học này',
    },
  },
};

// ---------------------------------------------------------------- this machine's numbers for the intensity question
/** The rows of lab 21.1 on one backend: the reader's own where they passed, else the reference's. */
function preferredRows(run: BackendRun | undefined) {
  const lab = 'ch21/lab1_dcs';
  const mine = labStatus(run, lab, 'work') === 'pass' ? benchRows(run, lab, 'work') : [];
  return mine.length ? { rows: mine, whose: tr('your kernels', 'các kernel của bạn') } : { rows: benchRows(run, lab, 'solution'), whose: tr('the reference solution', 'lời giải tham chiếu') };
}
const at = (rows: BenchRow[], name: string, chunk: number) =>
  num(rows.find((r) => r.name === name && num(r.nx) === 1024 && num(r.chunk) === chunk)?.seconds);

function OnThisMachine() {
  const { reports, loading } = useReports();
  const lines = realBackends(reports).flatMap((b) => {
    const { rows, whose } = preferredRows(reports[b]);
    const r = (chunk: number) => {
      const adj = at(rows, 'coarse_adjacent', chunk);
      const str = at(rows, 'coarse_strided', chunk);
      return adj && str ? adj / str : null;
    };
    const r64 = r(64);
    const r1024 = r(1024);
    if (r64 == null || r1024 == null) return [];
    return [{ b, device: String(reports[b].device.name ?? b), r64, r1024, whose }];
  });
  if (loading) return null;
  return lines.length ? (
    <>
      {lines.map((l) => (
        <Translated
          vi={
            <p key={l.b} data-machine={l.b} data-r64={fmt(l.r64)} data-r1024={fmt(l.r1024)}>
              Số liệu của chính bạn, {l.device} ({l.b}), {l.whose}: trong lab 21.1, kernel strided chạy nhanh gấp{' '}
              <b>{fmt(l.r64)}</b> lần kernel adjacent ở chunk 64 và <b>{fmt(l.r1024)}</b> lần ở chunk 1,024. Trên 1 nghĩa là
              kernel strided nhanh hơn. <a href={href('ch21', 'labs')}>Trang lab</a> có bảng đầy đủ.
            </p>
          }
        >
          <p key={l.b} data-machine={l.b} data-r64={fmt(l.r64)} data-r1024={fmt(l.r1024)}>
            Your own numbers, {l.device} ({l.b}), {l.whose}: in lab 21.1 the strided kernel ran <b>{fmt(l.r64)}</b> times as
            fast as the adjacent one at chunk 64 and <b>{fmt(l.r1024)}</b> times at chunk 1,024. Above 1 the strided kernel
            was faster. The <a href={href('ch21', 'labs')}>labs page</a> has the full table.
          </p>
        </Translated>
      ))}
    </>
  ) : (
    <Translated
      vi={
        <p className="muted" data-machine="none">
          Để đặt thiết bị của chính bạn bên cạnh câu hỏi này, bạn cần các dòng benchmark của lab 21.1:{' '}
          <code>python3 labs/pmpp.py run 21.1 --bench</code>.
        </p>
      }
    >
      <p className="muted" data-machine="none">
        To put your own device beside this question you need lab 21.1's benchmark rows:{' '}
        <code>python3 labs/pmpp.py run 21.1 --bench</code>.
      </p>
    </Translated>
  );
}

// ---------------------------------------------------------------- the page
export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports);
  const lab = 'ch21/lab1_dcs';
  const passing = [...backends, 'emu'].filter((b) => labStatus(reports[b], lab, 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch21/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>
        <Translated vi="Những gì cần mang theo">What to carry forward</Translated>
      </h2>
      <Translated
        vi={
          <ul>
            <li>
              Quyết định ai sở hữu mỗi đầu ra trước khi viết kernel. Mỗi đầu vào một luồng là scatter: các đầu ra của nó va chạm,
              mọi cập nhật cần một phép toán nguyên tử, các phép toán nguyên tử cùng địa chỉ bị tuần tự hóa, và một tổng float
              xây từ chúng đổi các bit cuối giữa các lần chạy. Mỗi đầu ra một luồng là gather: một lần ghi thường, theo một thứ
              tự cố định.
            </li>
            <li>
              Gather thắng hoàn toàn khi mọi đầu vào đến được mọi đầu ra, như trong một tổng trên mọi cặp. Khi mỗi đầu vào chỉ
              đến vài đầu ra, gather phải trả cho những cặp không tồn tại; khi đó hãy phân ngăn hoặc sắp xếp đầu vào trước, hoặc
              chấp nhận một scatter.
            </li>
            <li>
              Mã tuần tự tốt nhất có thể có thứ tự vòng lặp sai cho GPU. Hãy song song hóa thứ tự lấy đầu ra làm trung tâm, chấp
              nhận một chút phép tính thừa, rồi lấy lại sự tái sử dụng bên trong mỗi luồng.
            </li>
            <li>
              Gộp việc cho luồng (thread coarsening): một luồng tính nhiều đầu ra, nạp mỗi toán hạng dùng chung một lần và tính
              các số hạng dùng chung một lần. Nó không thể loại bỏ những gì mỗi đầu ra cần cho riêng mình (ở đây là một phép
              chia và một căn bậc hai cho mỗi cặp), và nó tốn thanh ghi.
            </li>
            <li>
              Một luồng đã gộp việc sở hữu những đầu ra nào quyết định việc hợp nhất truy cập. Hãy chia chúng cách nhau{' '}
              <code>blockDim.x</code>, hoặc ghi chúng bằng một lần lưu vector; thay đổi độ lệch ghi và phép tính của mỗi đầu ra
              cùng nhau. Fig. 21.8 như bản in sai về quyền sở hữu.
            </li>
            <li>
              Một giá trị mà mọi lane đọc cùng lúc là trường hợp tốt nhất của bộ nhớ đệm hằng trên NVIDIA và AMD. Bộ nhớ hằng
              nhỏ, nên hãy đưa dữ liệu vào theo từng chunk và để mỗi lần khởi chạy cộng phần của nó.
            </li>
            <li>
              Khối lượng việc trên mọi cặp tăng theo bình phương kích thước bài toán. Bán kính cắt cùng các ngăn làm nó tuyến
              tính: một danh sách lân cận gồm các độ lệch tính một lần cho mỗi khối trên host, các ngăn được nạp vào bộ nhớ chia
              sẻ, một phép kiểm tra trên mỗi luồng có phân kỳ, các ngăn cùng một sức chứa được đệm bằng nguyên tử giả, và một
              đường chậm hơn cho phần tràn.
            </li>
            <li>
              Cả sách lẫn nhà cung cấp nào cũng không đưa con số cho những lợi ích này, và Apple không ghi nhận quy tắc hợp nhất
              nào hay chi phí nào cho phép toán nguyên tử. Hãy dự đoán từ mô hình, rồi đo.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            Decide who owns each output before you write the kernel. A thread per input scatters: its outputs collide, every
            update needs an atomic, same-address atomics are serialised, and a float sum built from them changes its last
            bits from run to run. A thread per output gathers: one plain write, in a fixed order.
          </li>
          <li>
            Gather wins outright when every input reaches every output, as in an all-pairs sum. When each input reaches only
            a few outputs, gather pays for pairs that do not exist; then bin or sort the inputs first, or accept a scatter.
          </li>
          <li>
            The best sequential code can have the wrong loop order for a GPU. Parallelise the output-centric order, accept a
            little redundant arithmetic, then win the reuse back inside each thread.
          </li>
          <li>
            Coarsening: one thread computes several outputs, loads each shared operand once and computes shared terms once.
            It cannot remove what each output needs for itself (here a division and a square root per pair), and it costs
            registers.
          </li>
          <li>
            Which outputs a coarsened thread owns decides coalescing. Hand them out <code>blockDim.x</code> apart, or write
            them with one vector store; change the write offsets and each output's arithmetic together. Fig. 21.8 as printed
            gets the ownership wrong.
          </li>
          <li>
            A value every lane reads at once is the constant cache's best case on NVIDIA and AMD. Constant memory is small,
            so feed it in chunks and let each launch add its part.
          </li>
          <li>
            All-pairs work grows with the square of the problem. A cutoff with bins makes it linear: a neighbourhood list of
            offsets computed once per block on the host, bins staged in shared memory, a per-thread test that diverges,
            bins of one capacity padded with dummies, and a slower path for the overflow.
          </li>
          <li>
            Neither the book nor any vendor puts a number on these gains, and Apple documents no coalescing rule and no cost
            for atomics. Predict from the model, then measure.
          </li>
        </ul>
      </Translated>
      <Source section="21.6" pages="527–528" />

      <h2>
        <Translated vi="Một bài kiểm tra tổng hợp">A mixed check</Translated>
      </h2>
      <p>
        <Translated vi="Các câu này dựa trên cả chương và trên những tình huống mà các trang chưa cho thấy.">
          These draw on the whole chapter and on situations the pages did not show.
        </Translated>
        {asked
          ? tr(
              ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.`,
              ` Đến giờ trong chương này, bạn đã trả lời đúng ${right} trên ${asked} câu hỏi, tính theo câu trả lời gần nhất của mỗi câu.`,
            )
          : ''}
      </p>
      <Beyond>
        <Translated
          vi={
            <p>
              Kernel bias, hệ số 8 và các ngăn giảm nửa là của khóa học này. Kernel bias là front-end-checked và chưa bao giờ
              được chạy.
            </p>
          }
        >
          <p>
            The bias kernel, the factor of 8 and the halved bins are ours. The bias kernel is front-end-checked and has
            never been run.
          </p>
        </Translated>
      </Beyond>
      <Code
        tabs={[
          {
            label: tr('CUDA (not a lab file: for this question only)', 'CUDA (không phải tệp lab: chỉ cho câu hỏi này)'),
            code: BIAS,
            status: tr('front-end-checked; never run', 'front-end-checked; chưa bao giờ chạy'),
            tiers: { logits: 'global', ids: 'global', bias: 'global', v: 'reg', add: 'reg', e: 'reg' },
          },
        ]}
      />
      <Ask q={q['ch21/wrap/q-sparse-bias']} label={tr('Read the code', 'Đọc mã')} />
      <Ask q={q['ch21/wrap/q-intensity']} label={tr('Work it out', 'Tự tính')} />
      <OnThisMachine />
      <Ask q={q['ch21/wrap/q-factor8']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={q['ch21/wrap/q-small-bins']} />
      <Ask q={q['ch21/wrap/q-bitwise']} />

      <h2>
        <Translated vi="Bài tập của sách">The book's exercises</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương kết thúc bằng năm bài tập (p. 528). Mỗi bài được giải ở đâu đó trong khóa học này; sách không in đáp án, nên
            các đáp án ở đây là của khóa học này.
          </p>
        }
      >
        <p>
          The chapter ends with five exercises (p. 528). Each is worked somewhere in this course; the book prints no answers,
          so the answers here are ours.
        </p>
      </Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }} data-table="exercises">
          <thead>
            <tr>
              <th>
                <Translated vi="Bài tập">Exercise</Translated>
              </th>
              <th>
                <Translated vi="Nội dung">About</Translated>
              </th>
              <th>
                <Translated vi="Được giải ở đâu trong khóa học">Where it is worked here</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1</td>
              <td>
                <Translated vi="Mã host cấu hình lưới và khởi chạy kernel gather">Host code that configures the grid and launches the gather kernel</Translated>
              </td>
              <Translated
                vi={
                  <td>
                    <code>launchDcsGather</code> trong <a href={href('ch21', 'labs')}>lab 21.1</a>, với vòng lặp chunk ở p. 517;
                    luồng dữ liệu ở <a href={href('ch21', '21.2')}>21.2</a>
                  </td>
                }
              >
                <td>
                  <code>launchDcsGather</code> in <a href={href('ch21', 'labs')}>lab 21.1</a>, with the chunk loop of p. 517;
                  the data flow on <a href={href('ch21', '21.2')}>21.2</a>
                </td>
              </Translated>
            </tr>
            <tr>
              <td>2</td>
              <td>
                <Translated vi="Số phép toán mỗi lần lặp ở hệ số gộp việc tám">Operations per iteration at a coarsening factor of eight</Translated>
              </td>
              <Translated
                vi={
                  <td>
                    một câu hỏi ở <a href={href('ch21', '21.2')}>21.2</a>, và bảng phép toán của trình khám phá ở F = 8
                  </td>
                }
              >
                <td>
                  a question on <a href={href('ch21', '21.2')}>21.2</a>, and the explorer's operation table at F = 8
                </td>
              </Translated>
            </tr>
            <tr>
              <td>3</td>
              <td>
                <Translated vi="Hai bất lợi của việc giao nhiều việc hơn cho mỗi luồng">Two disadvantages of more work per thread</Translated>
              </td>
              <Translated
                vi={
                  <td>
                    một câu hỏi ở <a href={href('ch21', '21.2')}>21.2</a> (thanh ghi, cùng độ thô phân chia từ Fig. 21.9); chi phí
                    hợp nhất ở mục "Coalescing: who owns which four" của trang đó và trình chạy từng bước câu lệnh ghi; chi phí
                    thanh ghi cũng có ở <a href={href('ch21', 'start')}>trang bắt đầu</a>
                  </td>
                }
              >
                <td>
                  a question on <a href={href('ch21', '21.2')}>21.2</a> (registers, with granularity from Fig. 21.9); the
                  coalescing cost in its section "Coalescing: who owns which four" and the write-statement stepper; the
                  register cost also on the <a href={href('ch21', 'start')}>start page</a>
                </td>
              </Translated>
            </tr>
            <tr>
              <td>4</td>
              <td>
                <Translated vi="Lưu vector cho kernel đã gộp việc">Vector stores for the coarsened kernel</Translated>
              </td>
              <Translated
                vi={
                  <td>
                    một câu hỏi ở <a href={href('ch21', '21.2')}>21.2</a>, và "one float4 store" trong trình khám phá câu lệnh ghi
                    của trang đó; không thuộc lab
                  </td>
                }
              >
                <td>
                  a question on <a href={href('ch21', '21.2')}>21.2</a>, and "one float4 store" in its write-statement
                  explorer; not part of the lab
                </td>
              </Translated>
            </tr>
            <tr>
              <td>5</td>
              <td>
                <Translated vi="Một ngăn lân cận gây ra phân kỳ điều khiển như thế nào">How a neighbourhood bin causes control divergence</Translated>
              </td>
              <Translated
                vi={
                  <td>
                    một câu hỏi ở <a href={href('ch21', '21.5')}>21.5</a>, bảng mặt nạ của trình khám phá và hoạt ảnh khối phân
                    ngăn
                  </td>
                }
              >
                <td>
                  a question on <a href={href('ch21', '21.5')}>21.5</a>, the explorer's mask panel and the binned-block
                  animation
                </td>
              </Translated>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>
        <Translated vi="Những chỗ PDF có vẻ sai">Where the PDF looks wrong</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bản sách mà khóa học này dựa vào có vài chỗ có vẻ sai trong chương 21, và vài nhận định mà tài liệu hiện hành không
            ủng hộ. Chúng được liệt kê để bạn không bị bối rối khi đọc sách song song.
          </p>
        }
      >
        <p>
          The copy of the book this course was built from has a few apparent slips in chapter 21, and a few statements that
          current documentation does not support. They are listed so that you are not thrown by them when you read the book
          alongside.
        </p>
      </Translated>
      <Translated
        vi={
          <ul data-list="errata">
            <li>
              Dòng 05 của Fig. 21.8 cộng một <code>threadIdx.x</code> chưa nhân hệ số trong khi mỗi luồng ghi từ <code>i</code>{' '}
              đến <code>i+3</code>, nên các luồng cạnh nhau chồng lấn và phần lớn các điểm của mỗi khối không bao giờ được ghi.
              Văn bản ở p. 521 mô tả các luồng cách nhau bốn điểm, điều cần chỉ số được nhân với hệ số (
              <a href={href('ch21', '21.2')}>21.2</a>; mốc cơ sở của lab 21.1 là kernel đã sửa).
            </li>
            <li>
              Các con số phép toán ở pp. 519–521 không khớp: 9 phép toán mỗi điểm ở p. 519 và 12 ở p. 521; 12 phép chia cho bốn
              điểm trong khi mã có 4; 11 phép cộng và 6 phép nhân cho Fig. 21.8 trong khi đoạn mã có 12 và 8; z bị bỏ khỏi các
              lần nạp được loại bỏ; không phép căn bậc hai nào được đếm ở bất cứ đâu (
              <a href={href('ch21', '21.2')}>21.2</a> cho thấy các con số đếm được bên cạnh con số của sách).
            </li>
            <li>p. 521 nói hai vấn đề gây ra các lần ghi không được hợp nhất nhưng chỉ mô tả một.</li>
            <li>
              Fig. 21.9 có các luồng tính "up to 8 potentials, skipping by half-warps", trong khi mã dùng hệ số 4 và bước nhảy{' '}
              <code>blockDim.x</code>. "Padding waste" của nó không bao giờ được bàn trong văn bản (
              <a href={href('ch21', '21.2')}>21.2</a>).
            </li>
            <li>
              Fig. 21.13 ghi nhãn bán kính là "cutoff distance + length of bin diagonal"; văn bản và Fig. 21.14 nói nửa đường
              chéo. "9 fully covered" của Fig. 21.14 chỉ đúng tại một bán kính duy nhất, khoảng 2.12 bề rộng ngăn; trên phần còn
              lại của dải mà danh sách có 21 ngăn, chỉ 5 ngăn nằm trọn bên trong (<a href={href('ch21', '21.5')}>21.5</a>).
            </li>
            <li>
              Phần mở đầu (p. 513) liệt kê một "gather-to-scatter transformation"; chương lại đi từ scatter sang gather. Cùng
              danh sách đó nêu "fast boundary condition checking", điều mà không mục nào phát triển (
              <a href={href('ch21', 'start')}>trang bắt đầu</a>).
            </li>
            <li>
              Lát được đánh chỉ số theo ba cách: <code>grid.x*grid.y*z</code> với z kiểu float (Fig. 21.3),{' '}
              <code>(grid.x*grid.y*z) / gridspacing</code> (Figs. 21.4, 21.5) và <code>int k = z / gridspacing</code> (Fig.
              21.6 trở đi). Lab 21.1 truyền <code>k</code> dưới dạng số nguyên.
            </li>
            <li>
              Không kernel nào kiểm tra biên, và Fig. 21.5 không có số nguyên tử, nên các luồng vượt quá cuối một chunk không
              đầy sẽ cộng các nguyên tử cũ. Các đoạn mã ở <a href={href('ch21', '21.2')}>21.2</a> và lab 21.1 thêm cả hai.
            </li>
            <li>
              Lỗi nhỏ: cả <code>atom[]</code> và <code>atoms[]</code> đều có ở p. 519; "implicit method" (phương pháp ngầm) cho
              các nguyên tử ở xa (p. 523) không bao giờ được mô tả (<a href={href('ch21', '21.5')}>21.5</a> nêu tên phương pháp
              mà VMD dùng).
            </li>
          </ul>
        }
      >
        <ul data-list="errata">
          <li>
            Fig. 21.8's line 05 adds an unscaled <code>threadIdx.x</code> while each thread writes <code>i</code> to{' '}
            <code>i+3</code>, so neighbouring threads overlap and most of each block's points are never written. The text on
            p. 521 describes threads four points apart, which needs the index scaled by the factor (
            <a href={href('ch21', '21.2')}>21.2</a>; lab 21.1's baseline is the corrected kernel).
          </li>
          <li>
            The operation counts on pp. 519–521 do not add up: 9 operations per point on p. 519 and 12 on p. 521; 12
            divisions for four points where the code has 4; 11 additions and 6 multiplications for Fig. 21.8 where the
            listing has 12 and 8; z left out of the eliminated loads; no square roots counted anywhere (
            <a href={href('ch21', '21.2')}>21.2</a> shows the counted numbers beside the book's).
          </li>
          <li>p. 521 says two problems cause the uncoalesced writes and describes only one.</li>
          <li>
            Fig. 21.9 has threads computing "up to 8 potentials, skipping by half-warps", where the code uses a factor of 4
            and a stride of <code>blockDim.x</code>. Its "Padding waste" is never discussed in the text (
            <a href={href('ch21', '21.2')}>21.2</a>).
          </li>
          <li>
            Fig. 21.13 labels its radius "cutoff distance + length of bin diagonal"; the text and Fig. 21.14 say half the
            diagonal. Fig. 21.14's "9 fully covered" holds at a single radius, about 2.12 bin widths; over the rest of the
            range where the list has 21 bins, only 5 are fully inside (<a href={href('ch21', '21.5')}>21.5</a>).
          </li>
          <li>
            The opening (p. 513) lists a "gather-to-scatter transformation"; the chapter goes from scatter to gather. The
            same list names "fast boundary condition checking", which no section develops (
            <a href={href('ch21', 'start')}>start page</a>).
          </li>
          <li>
            The slice is indexed three ways: <code>grid.x*grid.y*z</code> with a float z (Fig. 21.3),{' '}
            <code>(grid.x*grid.y*z) / gridspacing</code> (Figs. 21.4, 21.5) and <code>int k = z / gridspacing</code> (Fig.
            21.6 on). Lab 21.1 passes <code>k</code> as an integer.
          </li>
          <li>
            No kernel tests a bound, and Fig. 21.5 has no atom count, so the threads past the end of a partial chunk would add
            stale atoms. The snippets on <a href={href('ch21', '21.2')}>21.2</a> and lab 21.1 add both.
          </li>
          <li>
            Small slips: <code>atom[]</code> and <code>atoms[]</code> both on p. 519; the far atoms' "implicit method" (p.
            523) is never described (<a href={href('ch21', '21.5')}>21.5</a> names the one VMD uses).
          </li>
        </ul>
      </Translated>
      <Translated vi={<p>Và những chỗ tài liệu hiện hành của nhà cung cấp nói khác, hoặc không nói gì:</p>}>
        <p>And where current vendor documentation says something else, or nothing:</p>
      </Translated>
      <Translated
        vi={
          <ul data-list="dated">
            <li>
              "half-warps" của Fig. 21.9 là đơn vị hợp nhất trong hướng dẫn năm 2007 của NVIDIA. Ngày nay NVIDIA mô tả việc hợp
              nhất theo từng warp, bằng sector 32 byte; đừng học quy tắc half-warp. "64-256 threads" mỗi khối của nó vẫn khớp với
              khuyến nghị của NVIDIA và AMD.
            </li>
            <li>
              <code>atomicAdd</code> kiểu float của Fig. 21.5 vẫn còn dùng được, nhưng NVIDIA giờ gọi nó là phép toán nguyên tử
              legacy và khuyên dùng <code>cuda::atomic_ref</code>. Trên HIP, dạng của nó phụ thuộc vào{' '}
              <code>-munsafe-fp-atomics</code>; trên Metal, lưới phải có kiểu <code>atomic_float</code>.
            </li>
            <li>
              "Significantly reduces" (p. 518) và "faster" (p. 522) không được sách hay nhà cung cấp nào định lượng; lab 21.1 đo
              cả hai trên GPU của bạn.
            </li>
            <li>
              p. 521, số thanh ghi của kernel đã gộp việc "stays within the allowed limit": không có con số, trình biên dịch hay
              thiết bị nào được nêu, nên không thể kiểm tra. <code>--resource-usage</code> cho thấy con số của bạn trên CUDA và
              HIP; Metal không cung cấp số thanh ghi.
            </li>
            <li>
              Bộ nhớ đệm hằng làm cho các lần đọc nguyên tử của kernel gather rẻ (p. 519) là của NVIDIA. Bộ nhớ đệm vô hướng
              (scalar cache) của AMD chỉ phục vụ các lần đọc giống nhau trên toàn wavefront, và Apple không ghi nhận bộ nhớ đệm
              hằng nào cả.
            </li>
          </ul>
        }
      >
        <ul data-list="dated">
          <li>
            Fig. 21.9's "half-warps" is the coalescing unit of NVIDIA's 2007 guide. Today NVIDIA states coalescing per warp,
            in 32-byte sectors; do not learn a half-warp rule. Its "64-256 threads" per block still matches NVIDIA's and
            AMD's advice.
          </li>
          <li>
            Fig. 21.5's float <code>atomicAdd</code> is current, but NVIDIA now calls it a legacy atomic and recommends{' '}
            <code>cuda::atomic_ref</code>. On HIP its form depends on <code>-munsafe-fp-atomics</code>; on Metal the grid must
            be typed <code>atomic_float</code>.
          </li>
          <li>
            "Significantly reduces" (p. 518) and "faster" (p. 522) are not quantified by the book or by any vendor; lab 21.1
            measures both on your GPU.
          </li>
          <li>
            p. 521, the coarsened kernel's register count "stays within the allowed limit": no count, compiler or device is
            given, so it cannot be checked. <code>--resource-usage</code> shows yours on CUDA and HIP; Metal exposes no
            register count.
          </li>
          <li>
            The constant cache that makes the gather kernel's atom reads cheap (p. 519) is NVIDIA's. AMD's scalar cache
            serves only reads that are the same across the wavefront, and Apple documents no constant cache at all.
          </li>
        </ul>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Đây là cách đọc của chúng tôi về một bản PDF, được đối chiếu với tài liệu hiện hành của các nhà cung cấp vào ngày
              2026-11-20 (notes/today/ch21-facts.md). Nếu bản của bạn khác, hãy tin bản của bạn.
            </p>
          }
        >
          <p>
            These are our readings of one PDF, checked against current vendor documentation on 2026-11-20
            (notes/today/ch21-facts.md). If your copy differs, trust your copy.
          </p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Những gì chương này bỏ qua">What this chapter left out</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương này được rút gọn trong khóa học, nên đây là những gì không được đề cập và nơi tìm chúng. Phần nền về VMD, việc
            đặt ion và vật lý của bản đồ (§21.1, pp. 513–514) là một đoạn ở trang bắt đầu, và hình phân tử được kết xuất ở Fig.
            21.1 (p. 514) không được trình bày. Hai đoạn mã C tuần tự, Figs. 21.3 và 21.4 (pp. 515–516), được mô tả bằng lời,
            phép hoán đổi vòng lặp và dz² cùng dy² được đưa ra ngoài vòng lặp, không chạy qua từng bước. Luồng dữ liệu theo lát
            và chunk của host (p. 517) có hai câu ở trang 21.2 và nằm trong các hàm khởi chạy của lab 21.1.
          </p>
        }
      >
        <p>
          The chapter is condensed in this course, so here is what is not covered and where to find it. The VMD background,
          ion placement and the physics of the map (§21.1, pp. 513–514) are one paragraph on the start page, and the
          rendered molecule of Fig. 21.1 (p. 514) is not shown. The two sequential C listings, Figs. 21.3 and 21.4 (pp.
          515–516), are described in prose, the loop interchange and the hoisted dz² and dy², not stepped through. The
          host's slice-and-chunk data flow (p. 517) gets two sentences on page 21.2 and lives in lab 21.1's launch functions.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Từ §21.5, hình học 3D ở cấp Ångström của Fig. 21.13 và cách dựng từ 512 vòng tròn (p. 525) được thay bằng trình khám
            phá 2D và các tổng 3D được tính ra. Kernel phân ngăn, mà sách chỉ đưa dưới dạng phác thảo (p. 526), được giải thích và
            có hoạt ảnh nhưng không được xây trong một lab. Các thể tích con cho phép lượt xử lý phần tràn của host chồng lấn với
            device (p. 527) có một câu. Phiên bản lưu vector của bài tập 4 (p. 528) được bàn và hỏi ở trang 21.2 nhưng không
            thuộc lab. Phần tóm tắt (§21.6, pp. 527–528) được gộp vào trang này, và ba tài liệu tham khảo (p. 528) không được
            chép lại. Không phần nào phía sau trong khóa học phụ thuộc vào những gì bị bỏ qua.
          </p>
        }
      >
        <p>
          From §21.5, the Ångström-level 3D geometry of Fig. 21.13 and the construction from 512 circles (p. 525) are
          replaced by the 2D explorer and the computed 3D totals. The binned kernel, which the book gives only as an outline
          (p. 526), is explained and animated but not built in a lab. The sub-volumes that let the host's overflow pass
          overlap the device (p. 527) get one sentence. The vector-store revision of exercise 4 (p. 528) is discussed and
          questioned on page 21.2 but is not part of the lab. The summary (§21.6, pp. 527–528) is folded into this page, and
          the three references (p. 528) are not reproduced. Nothing later in the course depends on what was left out.
        </p>
      </Translated>

      <h2>
        <Translated vi="Dự án đến lúc này">The project so far</Translated>
      </h2>
      <p>
        <Translated
          vi={
            <>
              Chương này không thêm mốc dự án nào vào dự án xuyên suốt. Quyết định đầu tiên của nó đã có sẵn trong các mốc bạn
              có: các phép nhân ma trận của M2 và M3 giao cho mỗi luồng các đầu ra của riêng nó, cách phân rã mà chương 22 gọi
              là lấy đầu ra làm trung tâm. Lab 21.1 là tùy chọn.
            </>
          }
        >
          This chapter adds no milestone to the through-line project. Its first decision is already in the milestones you
          have: the matrix multiplications of M2 and M3 give each thread its own outputs, the decomposition chapter 22 calls
          output-centric. Lab 21.1 is optional.
        </Translated>{' '}
        <span data-lab211={passing.join(',') || 'none'}>
          {backends.length === 0 && !reports['emu']
            ? tr('No lab results are recorded on this machine yet.', 'Máy này chưa ghi lại kết quả lab nào.')
            : passing.length
              ? tr(
                  `Your own version passes on ${passing.map(where).join(' and ')} on this machine.`,
                  `Phiên bản của bạn qua trên ${passing.map(where).join(' và ')} trên máy này.`,
                )
              : tr('Your own version has not passed on this machine yet.', 'Phiên bản của bạn chưa qua trên máy này.')}
        </span>{' '}
        <Translated vi={<>Xem <a href={href('ch21', 'labs')}>lab</a>.</>}>
          See the <a href={href('ch21', 'labs')}>lab</a>.
        </Translated>
      </p>
      <Translated
        vi={
          <p>
            Chương 22 lùi lại khỏi từng kernel riêng lẻ để nhìn vào các quyết định đằng sau chúng. Nó dùng phân ngăn theo bán
            kính cắt làm ví dụ về việc đổi một chút độ chính xác lấy một thuật toán tốt hơn (<a href={href('ch22', '22.1')}>§22.1</a>),
            phân loại mọi mẫu trong sách thành gather hoặc scatter (<a href={href('ch22', '22.2')}>§22.2</a>), và kết thúc bằng
            batching, độ trễ so với thông lượng, chính là mốc dự án M8. Bắt đầu ở <a href={href('ch22', 'start')}>chương 22</a>.
          </p>
        }
      >
        <p>
          Chapter 22 steps back from single kernels to the decisions behind them. It uses cutoff binning as its example of
          trading a little accuracy for a better algorithm (<a href={href('ch22', '22.1')}>§22.1</a>), sorts every pattern in
          the book into gather or scatter (<a href={href('ch22', '22.2')}>§22.2</a>), and ends with batching, latency
          against throughput, which is milestone M8. Start at <a href={href('ch22', 'start')}>chapter 22</a>.
        </p>
      </Translated>
    </>
  );
}
