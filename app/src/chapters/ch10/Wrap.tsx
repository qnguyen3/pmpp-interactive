import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Beyond, Source } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { functionSource } from '../../sim/toolchain';
import { STATUS } from './status';

// The kernel for the first question is a snippet of our own, loaded at run time, so there
// is no copy to drift: labs/ch10/snippets/wrap_centre.cu. Wrap.snippets.test.ts ties its
// helpers to lab 10.1, runs it on the course's emulator, and pins the numbers below.
export const CENTRE_FILE = 'ch10/snippets/wrap_centre.cu';
export const CENTRE = { block: 256, cols: 1000 } as const;
/** The columns of a row that thread 0 writes in centreRows: 0, block, 2 block, ... */
export const threadZeroColumns = (block: number, cols: number): number[] =>
  Array.from({ length: Math.ceil(cols / block) }, (_, k) => k * block).filter((c) => c < cols);

/**
 * Fig. 10.10's tree on a block of any size: thread t first adds inputs t and t + B, then
 * for stride = B/2, B/4, ... down to 1 (integer halving) thread t < stride adds slot
 * t + stride into slot t. Returns how many of the 2B inputs slot 0 ends up holding.
 */
export function sharedTreeCovers(block: number): number {
  const slots = Array.from({ length: block }, () => 2);
  for (let s = Math.floor(block / 2); s >= 1; s = Math.floor(s / 2)) for (let t = 0; t < s; t++) slots[t] += slots[t + s];
  return slots[0];
}
export const ODD_BLOCK = 96;

export const questionsWrap: Record<string, Question> = {
  'ch10/wrap/q-centre': {
    kind: 'choice',
    id: 'ch10/wrap/q-centre',
    prompt: (
      <Translated
        vi={
          <>
            Kernel ở trên căn giữa từng hàng của một ma trận <code>[rows × {CENTRE.cols}]</code>, <code>y = x − mean(x)</code>,
            bước đầu tiên của LayerNorm, với một khối {CENTRE.block} luồng cho mỗi hàng. Nó cộng hàng bằng phép rút gọn warp
            hai giai đoạn của mục 10.8, rồi mỗi luồng chia giá trị <code>s</code> nó đang giữ cho độ dài hàng và ghi các cột
            của mình. Nó biên dịch và chạy được. Những đầu ra nào đúng?
          </>
        }
      >
        The kernel above centres each row of a <code>[rows × {CENTRE.cols}]</code> matrix, <code>y = x − mean(x)</code>,
        the first step of LayerNorm, with one block of {CENTRE.block} threads per row. It sums the row with the two-stage
        warp reduction of section 10.8, then every thread divides the <code>s</code> it holds by the row length and
        writes its columns. It compiles and runs. Which outputs are right?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Tất cả: rào chắn trước giai đoạn 2 khiến mọi luồng đều thấy được tổng.">
            All of them: the barrier before stage 2 makes the total visible to every thread.
          </Translated>
        ),
        why: (
          <Translated vi="Rào chắn đó sắp thứ tự cho việc bàn giao các tổng của warp. Tổng chung được tính sau nó, trong thanh ghi của warp 0, và không có gì trao nó trở lại.">
            That barrier orders the hand-over of the warp sums. The total is computed after it, in warp 0's registers, and nothing hands it back.
          </Translated>
        ),
      },
      {
        label: (
          <Translated
            vi={`Chỉ những cột mà luồng 0 ghi: ${threadZeroColumns(CENTRE.block, CENTRE.cols).join(', ')} của mỗi hàng.`}
          >
            {`Only the columns that thread 0 writes: ${threadZeroColumns(CENTRE.block, CENTRE.cols).join(', ')} of each row.`}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Các cột mà warp 0 ghi: mọi lane của warp 0 đều đã chạy warp_reduce thứ hai.">
            The columns warp 0 writes: every lane of warp 0 ran the second warp_reduce.
          </Translated>
        ),
        why: (
          <Translated vi="Một cây shuffle-down chỉ để tổng của mọi đầu vào ở lane 0. Lane 1 kết thúc với phần của các lane 1 đến 31, lane 2 với ít hơn, và cứ thế.">
            A shuffle-down tree leaves the sum of all its inputs in lane 0 only. Lane 1 ends with lanes 1 to 31's share, lane 2 with less, and so on.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không cái nào: giai đoạn 2 đọc partial_s trước khi mọi warp đã ghi tổng của mình.">
            None: stage 2 reads partial_s before every warp has written its sum.
          </Translated>
        ),
        why: (
          <Translated vi="Lệnh __syncthreads() giữa các lần ghi của lane 0 và các lần đọc của warp 0 có ở đó.">
            The __syncthreads() between the lane-0 writes and warp 0's reads is there.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              <code>warp_reduce</code> dựng trên shuffle-down chỉ để tổng ở lane 0 (p. 240), và giai đoạn 2 chỉ chạy trong
              warp 0, nên khi nó kết thúc, tổng của hàng chỉ tồn tại trong một thanh ghi: thanh ghi của luồng 0. Mọi luồng
              khác chia giá trị cuối cùng nó giữ, là một tổng bộ phận của giai đoạn 2 trong warp 0, và ở những nơi khác là
              phần của warp mình hoặc ít hơn. Luồng 0 ghi các cột {threadZeroColumns(CENTRE.block, CENTRE.cols).join(', ')};{' '}
              {CENTRE.cols - threadZeroColumns(CENTRE.block, CENTRE.cols).length} cột còn lại của mỗi hàng đều sai.
            </p>
            <p>
              Kết quả của một khối phải được trao lại theo cách các tổng của warp đã được bàn giao: luồng 0 ghi nó vào bộ
              nhớ chia sẻ, một rào chắn, rồi mọi luồng đọc nó. Softmax cũng cần như vậy cho max và cho tổng của nó, đó là lý
              do các kernel của lab 10.3 làm thế. <code>simd_sum</code> của Metal trao tổng cho mọi lane của một SIMD group,
              không phải cho các SIMD group khác, nên ở đó cũng cần bước trao lại.
            </p>
          </>
        }
      >
        <>
          <p>
            <code>warp_reduce</code> built on shuffle-down leaves the sum in lane 0 only (p. 240), and stage 2 runs only in
            warp 0, so when it ends the row's total exists in one register: thread 0's. Every other thread divides what it
            last held, a partial sum of stage 2 in warp 0 and its own warp's share or less elsewhere. Thread 0 writes
            columns {threadZeroColumns(CENTRE.block, CENTRE.cols).join(', ')}; the other{' '}
            {CENTRE.cols - threadZeroColumns(CENTRE.block, CENTRE.cols).length} of each row are wrong.
          </p>
          <p>
            A block's result has to be handed back the way the warp sums were handed over: thread 0 writes it to shared
            memory, a barrier, then every thread reads it. A softmax needs the same for its max and for its sum, which is
            why lab 10.3's kernels do it. Metal's <code>simd_sum</code> gives the sum to every lane of one SIMD-group, not
            to the other SIMD-groups, so the hand-back is needed there too.
          </p>
        </>
      </Translated>
    ),
    source: 'ours, from ch. 10 §10.7 p. 240 (only lane 0\'s value is the sum) and §10.8 pp. 241–243; run on the course emulator by Wrap.snippets.test.ts; notes/today/ch10-facts.md C1',
    vi: {
      source:
        'của khóa học này, từ ch. 10 §10.7 p. 240 (chỉ giá trị của lane 0 là tổng) và §10.8 pp. 241–243; chạy trên trình giả lập của khóa học bởi Wrap.snippets.test.ts; notes/today/ch10-facts.md C1',
    },
  },
  'ch10/wrap/q-oddblock': {
    kind: 'numeric',
    id: 'ch10/wrap/q-oddblock',
    prompt: (
      <Translated
        vi={
          <>
            Fig. 10.10, kernel dùng bộ nhớ chia sẻ, được biên dịch và khởi chạy với{' '}
            <code>BLOCK_DIM = blockDim.x = {ODD_BLOCK}</code> (ba warp) trên một đoạn {2 * ODD_BLOCK} giá trị. Vòng lặp của nó
            chia đôi bước nhảy bằng phép chia nguyên. <code>*output</code> bao gồm bao nhiêu trong {2 * ODD_BLOCK} giá trị?
          </>
        }
      >
        Fig. 10.10, the shared-memory kernel, is compiled and launched with <code>BLOCK_DIM = blockDim.x = {ODD_BLOCK}</code>{' '}
        (three warps) on a segment of {2 * ODD_BLOCK} values. Its loop halves the stride with integer division. How many
        of the {2 * ODD_BLOCK} values does <code>*output</code> include?
      </Translated>
    ),
    answer: sharedTreeCovers(ODD_BLOCK),
    unit: 'values',
    explain: (
      <Translated
        vi={
          <p>
            Bước nạp để lại {ODD_BLOCK} ô nhớ, mỗi ô 2 giá trị. Các bước nhảy 48, 24, 12 và 6 chia đôi gọn, để lại các ô 0 đến
            5 với 32 giá trị mỗi ô. Ở bước nhảy 3, các luồng 0 đến 2 cộng các ô 3 đến 5: ba ô 64. Bước nhảy tiếp theo là 3 / 2
            = 1, nên luồng 0 cộng ô 1 còn ô 2 không bao giờ được đọc lại. Kết quả chứa {sharedTreeCovers(ODD_BLOCK)} trong{' '}
            {2 * ODD_BLOCK} giá trị, và không rào chắn hay kiểm tra biên nào nhận ra được. Mọi bước nhảy lẻ lớn hơn 1 đều bỏ
            rơi một ô. Cách sửa là dùng khối có kích thước lũy thừa của hai, điều mà các launcher của lab 10.1 bắt buộc, hoặc
            đệm mảng chia sẻ tới lũy thừa của hai kế tiếp bằng phần tử đơn vị.
          </p>
        }
      >
        <p>
          The load leaves {ODD_BLOCK} slots of 2 values each. Strides 48, 24, 12 and 6 halve cleanly, leaving slots 0 to 5
          with 32 values each. At stride 3, threads 0 to 2 add slots 3 to 5: three slots of 64. The next stride is 3 / 2 =
          1, so thread 0 adds slot 1 and slot 2 is never read again. The result holds {sharedTreeCovers(ODD_BLOCK)} of the{' '}
          {2 * ODD_BLOCK} values, and no barrier or bounds check can notice. Every odd stride above 1 strands a slot. The
          fixes are a power-of-two block, which is what lab 10.1's launchers insist on, or padding the shared array to the
          next power of two with the identity.
        </p>
      </Translated>
    ),
    source: 'our analysis of Fig. 10.10, p. 237; the book assumes power-of-two blocks without saying so (notes, "Unclear in the PDF")',
    vi: {
      unit: 'giá trị',
      source:
        'phân tích của chúng tôi về Fig. 10.10, p. 237; sách ngầm giả định khối có kích thước lũy thừa của hai mà không nói ra (notes, "Unclear in the PDF")',
    },
  },
  'ch10/wrap/q-columns': {
    kind: 'choice',
    id: 'ch10/wrap/q-columns',
    prompt: (
      <Translated
        vi={
          <>
            Bạn cần tổng theo batch của một activation thứ tự hàng (row-major) <code>[64 × 4096]</code>: một kết quả cho mỗi
            cột, 4,096 tổng, mỗi tổng 64 giá trị, đúng dạng của một phép lấy trung bình theo token hay một gradient bias.
            Kernel nào?
          </>
        }
      >
        You need the sum over the batch of a row-major <code>[64 × 4096]</code> activation: one result per column, 4,096
        sums of 64 values each, the shape of a mean over tokens or a bias gradient. Which kernel?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một khối 64 luồng cho mỗi cột, rồi cây dùng bộ nhớ chia sẻ của mục 10.6.">
            One block of 64 threads per column, then the shared-memory tree of section 10.6.
          </Translated>
        ),
        why: (
          <Translated vi="64 lần nạp của mỗi khối cách nhau 4,096 float, nên không hai lần nào chung một yêu cầu bộ nhớ, và cây tốn rào chắn cho chỉ 64 giá trị.">
            Each block's 64 loads are 4,096 floats apart, so no two share a memory request, and the tree spends barriers on 64 values.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một luồng cho mỗi cột, mỗi luồng cộng 64 giá trị của cột mình trong một vòng lặp đi xuống các hàng.">
            One thread per column, each adding its column's 64 values in a loop down the rows.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Một luồng cho mỗi phần tử, mỗi luồng cộng giá trị của mình vào kết quả của cột bằng một atomic.">
            One thread per element, each adding its value into its column's result with an atomic.
          </Translated>
        ),
        why: (
          <Translated vi="262,144 atomic, 64 cái trên mỗi kết quả, đến theo không thứ tự cố định nào: chậm hơn một vòng lặp, và các tổng float thay đổi giữa các lần chạy.">
            262,144 atomics, 64 on every result, landing in no fixed order: slower than a loop, and the float sums change from run to run.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chuyển vị ma trận trước, rồi giao cho mỗi hàng trong 4,096 hàng một khối, như lab 10.3 làm.">
            Transpose the matrix first, then give each of the 4,096 rows one block, as lab 10.3 does.
          </Translated>
        ),
        why: (
          <Translated vi="Phép chuyển vị đọc và ghi toàn bộ ma trận thêm một lần để đạt tới một bố cục mà vòng lặp không cần: các luồng kề nhau vốn đã đọc các địa chỉ kề nhau.">
            The transpose reads and writes the whole matrix once more to reach a layout the loop does not need: neighbouring threads already read neighbouring addresses.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đã có sẵn 4,096 phép rút gọn độc lập, tức 4,096 luồng công việc trước khi có bất kỳ cây nào. Với một luồng cho
            mỗi cột, 32 luồng của một warp đọc 32 float kề nhau của cùng một hàng ở mỗi bước lặp: được hợp nhất, đúng như mục
            10.5 mong muốn. Mỗi vòng lặp là gộp việc cho luồng của mục 10.10 được đẩy tới tận cùng, không rào chắn, không bộ
            nhớ chia sẻ, không lane rảnh và với thứ tự cộng cố định. Cây chỉ có lợi khi có ít phép rút gọn dài; ở đây có
            nhiều phép ngắn, và phép rút gọn chạy dọc theo chiều không liên tục.
          </p>
        }
      >
        <p>
          There are already 4,096 independent reductions, 4,096 threads of work before any tree. With one thread per column,
          the 32 threads of a warp read 32 neighbouring floats of one row at every step of their loops: coalesced, as
          section 10.5 wants. Each loop is section 10.10's coarsening taken to its end, with no barrier, no shared memory,
          no idle lanes and a fixed order of additions. A tree pays when there are few long reductions; here there are many
          short ones, and the reduction runs along the dimension that is not contiguous.
        </p>
      </Translated>
    ),
    source: 'ours, from ch. 10 §10.5 pp. 234–235 and §10.10 pp. 246–248',
    vi: { source: 'của khóa học này, từ ch. 10 §10.5 pp. 234–235 và §10.10 pp. 246–248' },
  },
  'ch10/wrap/q-bits': {
    kind: 'choice',
    id: 'ch10/wrap/q-bits',
    prompt: (
      <Translated
        vi={
          <>
            Bốn phép rút gọn trên cùng một mảng lớn, mỗi phép theo thiết kế nhiều khối của sách: một cây cho mỗi khối, rồi
            một lần commit atomic cho mỗi khối. (1) Tổng float. (2) Max float, commit bằng vòng lặp compare-and-swap. (3) Đếm
            số nguyên các token, không thể tràn số. (4) |x| lớn nhất, cho thang đo lượng tử hóa. Bạn chạy mỗi phép hai lần
            trên cùng đầu vào. Phép nào có thể cho các bit khác ở lần thứ hai?
          </>
        }
      >
        Four reductions over the same large array, each with the book's multi-block design: a tree per block, then one
        atomic commit per block. (1) A float sum. (2) A float max, committed with a compare-and-swap loop. (3) An integer
        count of tokens that cannot overflow. (4) The largest |x|, for a quantisation scale. You run each twice on the
        same input. Which can give different bits the second time?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Cả bốn: các khối commit theo thứ tự khác nhau ở mỗi lần chạy.">
            All four: the blocks commit in a different order on every run.
          </Translated>
        ),
        why: (
          <Translated vi="Thứ tự khác chỉ làm thay đổi kết quả khi toán tử làm tròn hoặc phụ thuộc vào thứ tự.">
            A different order changes the result only when the operator rounds or depends on order.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ (1): tổng float làm tròn sau mỗi lần commit.">
            Only (1): the float sum rounds after every commit.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="(1) và (2): cả hai đều là dấu phẩy động.">(1) and (2): both are floating point.</Translated>,
        why: (
          <Translated vi="Max chọn một trong các đầu vào của nó. Nó không bao giờ làm tròn, nên thứ tự nào cũng chọn cùng một giá trị (với các giá trị hữu hạn).">
            A max selects one of its inputs. It never rounds, so any order selects the same one (for finite values).
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không phép nào: mỗi lần commit atomic đều chính xác.">None: each atomic commit is exact.</Translated>
        ),
        why: (
          <Translated vi="Đúng, mỗi lần commit là không thể chia cắt. Điều thay đổi giữa các lần chạy là thứ tự các lần commit, và tổng float làm tròn sau mỗi lần.">
            Each commit is indivisible, yes. What changes between runs is the order of the commits, and a float sum rounds after each one.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một atomic giữ trọn mỗi lần commit cho riêng nó; nó không cho các lần commit thứ tự nào (p. 245). Vì vậy kết quả
            lặp lại được đúng khi toán tử không quan tâm đến thứ tự. Phép cộng số nguyên có tính kết hợp, miễn là không thể
            tràn số (footnote 2, p. 224). Max và max của |x| chọn một đầu vào, nên không có gì để làm tròn (trừ NaN, footnote
            3, p. 225). Phép cộng float làm tròn sau mỗi bước, nên một thứ tự khác có thể kết thúc với các bit cuối khác, và
            đó là điều mà phép đếm tính tất định của lab 10.1 đo trên GPU của bạn.
          </p>
        }
      >
        <p>
          An atomic gives each commit to itself; it gives the commits no order (p. 245). So the result is repeatable exactly
          when the operator does not care about order. Integer addition is associative, as long as it cannot overflow
          (footnote 2, p. 224). Max and the max of |x| select an input, so there is nothing to round (NaN aside, footnote 3,
          p. 225). Float addition rounds after every step, so another order can end in other last bits, which is what lab
          10.1's determinism count measures on your GPU.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.2 pp. 224–225 and §10.9 p. 245; the four cases are ours',
    vi: { source: 'ch. 10 §10.2 pp. 224–225 và §10.9 p. 245; bốn trường hợp là của khóa học này' },
  },
  'ch10/wrap/q-greedy': {
    kind: 'choice',
    id: 'ch10/wrap/q-greedy',
    prompt: (
      <Translated
        vi={
          <>
            Ở chế độ greedy, một bộ lấy mẫu tính <code>probs = softmax(logits / T)</code> với T &gt; 0, rồi trả về{' '}
            <code>argmax(probs)</code>. Thực sự nó cần những phép rút gọn nào trên hàng?
          </>
        }
      >
        In greedy mode a sampler computes <code>probs = softmax(logits / T)</code> with T &gt; 0, then returns{' '}
        <code>argmax(probs)</code>. Which reductions over the row does it actually need?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một max, một tổng, rồi một argmax: ba lượt qua hàng.">
            A max, a sum, then an argmax: three passes over the row.
          </Translated>
        ),
        why: (
          <Translated vi="Cách đó tính ra các xác suất, thứ mà decode greedy không bao giờ dùng.">
            That computes the probabilities, which greedy decoding never uses.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một argmax duy nhất trên các logits thô, với quy tắc phá hòa của nó.">
            A single argmax over the raw logits, with its tie rule.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Một max trên các logits: logit lớn nhất chính là token.">
            A max over the logits: the largest logit is the token.
          </Translated>
        ),
        why: (
          <Translated vi="Max cho giá trị; token là chỉ số của nó. Việc đó cần một argmax, một phép rút gọn trên các cặp (giá trị, chỉ số).">
            The max gives the value; the token is its index. That takes an argmax, a reduction of (value, index) pairs.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một tổng: hệ số chuẩn hóa quyết định xác suất nào lớn nhất.">
            A sum: the normaliser decides which probability is largest.
          </Translated>
        ),
        why: (
          <Translated vi="Mọi xác suất trong hàng đều được chia cho cùng một tổng, điều này không đổi thứ tự nào.">
            Every probability in the row is divided by the same sum, which changes no order.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Chia cho T &gt; 0, trừ đi max, lấy <code>exp</code> và chia cho một tổng chung đều giữ nguyên thứ tự các phần
            tử, nên chúng không bao giờ thay đổi phần tử nào lớn nhất. Decode greedy là một argmax trên các logits: một phép
            rút gọn trên các cặp (giá trị, chỉ số) với một quy tắc phá hòa, chính là argmax của lab 10.3 và thứ mà bộ decode
            của nó gọi. Tính xác suất trước còn tệ hơn là phí công: việc làm tròn trong <code>exp</code> và trong phép chia
            có thể biến hai logit gần nhau thành hai xác suất bằng nhau, và khi đó quy tắc phá hòa chọn một token mà các
            logits vốn không hòa.
          </p>
        }
      >
        <p>
          Dividing by T &gt; 0, subtracting the max, taking <code>exp</code> and dividing by one shared sum all keep the
          order of the entries, so they never change which one is largest. Greedy decoding is an argmax over the logits:
          one reduction of (value, index) pairs with a tie rule, which is lab 10.3's argmax and what its decoder calls.
          Computing the probabilities first is worse than wasted work: rounding in <code>exp</code> and in the division
          can turn two close logits into equal probabilities, and then the tie rule picks a token the logits did not tie
          on.
        </p>
      </Translated>
    ),
    source: 'ours; the tie rule from ch. 10 §10.2 p. 225 and notes/today/ch10-facts.md D4',
    vi: { source: 'của khóa học này; quy tắc phá hòa từ ch. 10 §10.2 p. 225 và notes/today/ch10-facts.md D4' },
  },
};

/** The kernel of the first question, from the snippet file. */
function CentreKernel() {
  const file = useLabFile(CENTRE_FILE);
  const fn = file && !file.startsWith('// could not load') ? functionSource(file, 'centreRows') : null;
  return (
    <Code
      tabs={[
        {
          label: tr('CUDA (not a lab file: for this question only)', 'CUDA (không phải tệp của lab: chỉ cho câu hỏi này)'),
          code: fn ? fn.text : file ?? `// loading labs/${CENTRE_FILE}`,
          status: STATUS.wrong,
          tiers: { x: 'global', y: 'global', row: 'global', out: 'global', partial_s: 'shared', s: 'reg', mean: 'reg', c: 'reg' },
        },
      ]}
      caption={
        <Translated
          vi={
            <>
              Từ <code>labs/{CENTRE_FILE}</code>. Kiểu mask, <code>warpIdx</code>, <code>laneIdx</code> và{' '}
              <code>warp_reduce</code> của nó là của lab 10.1. Các unit test chạy nó trên trình giả lập của khóa học.
            </>
          }
        >
          <>
            From <code>labs/{CENTRE_FILE}</code>. Its mask type, <code>warpIdx</code>, <code>laneIdx</code> and{' '}
            <code>warp_reduce</code> are lab 10.1's. The unit tests run it on the course's emulator.
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
  const m4 = backends.filter((b) => labStatus(reports[b], 'ch10/lab3_softmax_argmax', 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch10/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>
        <Translated vi="Những gì mang theo">What to carry forward</Translated>
      </h2>
      <Translated
        vi={
          <ul>
            <li>
              Một phép rút gọn là một vòng lặp qua một toán tử hai ngôi, bắt đầu từ phần tử đơn vị của nó. Một cây chỉ cho
              ra đáp án của vòng lặp nếu toán tử có tính kết hợp, và chỉ cho phép mọi thứ tự toán hạng nếu nó còn có tính
              giao hoán. Phép cộng float có tính giao hoán nhưng, một khi đã làm tròn, thì không có tính kết hợp, nên một
              cây khác, số khối khác hay thứ tự commit khác có thể làm thay đổi các bit cuối.
            </li>
            <li>
              Bắt đầu max từ −FLT_MAX hoặc từ phần tử đầu tiên, không phải từ −∞, vì một backend có thể không tôn trọng giá
              trị vô cực. Một argmax rút gọn các cặp (giá trị, chỉ số), với một quy tắc phá hòa làm cho phép kết hợp có tính
              giao hoán.
            </li>
            <li>
              Vị trí của các luồng đang hoạt động quyết định chi phí. Hãy giữ chúng ở cạnh nhau, để cả warp cùng rảnh một
              lúc và các truy cập của một warp nằm kề nhau.
            </li>
            <li>
              Giữ các kết quả bộ phận ở gần: bộ nhớ chia sẻ trước bộ nhớ toàn cục, thanh ghi và shuffle trước bộ nhớ chia
              sẻ, một rào chắn cho mỗi lần bàn giao. Khi dữ liệu đã được nạp, một phép rút gọn chủ yếu là chờ đợi hơn là
              tính toán.
            </li>
            <li>
              Mã warp có thể chuyển sang backend khác: lane mask có kiểu, độ rộng đọc lúc chạy, lần đọc ở giai đoạn 2 có
              kiểm tra, và không lane nào rời đi trước các lệnh shuffle. Một cây shuffle-down chỉ để kết quả ở lane 0.
            </li>
            <li>
              Giữa các khối, một atomic không sắp thứ tự gì cả; một lượt thứ hai mới cố định thứ tự. Gộp việc cho luồng tới
              khi chỉ còn vừa đủ khối để lấp đầy thiết bị, và nhớ rằng một khối cho mỗi hàng ở batch 1 còn xa mới đủ.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            A reduction is one loop over a binary operator, started at its identity. A tree gives the loop's answer only if
            the operator is associative, and any order of operands only if it is also commutative. Float addition is
            commutative but, once rounded, not associative, so a different tree, block count or commit order can change
            the last bits.
          </li>
          <li>
            Start a max at −FLT_MAX or at the first element, not at −∞ where a backend may not honour infinities. An
            argmax reduces (value, index) pairs, with a tie rule that makes the combine commutative.
          </li>
          <li>
            Where the active threads sit decides the cost. Keep them together, so that whole warps fall idle at once and a
            warp's accesses are neighbours.
          </li>
          <li>
            Keep partial results close: shared memory before global memory, registers and shuffles before shared memory,
            one barrier per hand-off. Once the data is loaded, a reduction is waiting more than computing.
          </li>
          <li>
            Warp code that ports: a typed lane mask, the width read at run time, the stage-2 read guarded, and no lane
            leaving before the shuffles. A shuffle-down tree leaves the result in lane 0 only.
          </li>
          <li>
            Across blocks an atomic orders nothing; a second pass fixes the order. Coarsen until there are just enough
            blocks to fill the device, and remember that one block per row at batch 1 is far from enough.
          </li>
        </ul>
      </Translated>
      <Source section="10.11" pages="249" />

      <h2>
        <Translated vi="Một bài kiểm tra tổng hợp">A mixed check</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Các câu này dựa trên cả chương và trên những tình huống mà các mục chưa trình bày.
            {asked
              ? ` Tới giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu hỏi, tính theo câu trả lời gần nhất của bạn cho mỗi câu.`
              : ''}
          </p>
        }
      >
        <p>
          These draw on the whole chapter and on situations the sections did not show.
          {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Kernel và các tình huống dưới đây là của khóa học này.</p>}>
          <p>The kernel and the situations below are ours.</p>
        </Translated>
      </Beyond>
      <CentreKernel />
      <Ask q={q['ch10/wrap/q-centre']} label={tr('Find the defect', 'Tìm lỗi')} />
      <Ask q={q['ch10/wrap/q-oddblock']} />
      <Ask q={q['ch10/wrap/q-columns']} />
      <Ask q={q['ch10/wrap/q-bits']} />
      <Ask q={q['ch10/wrap/q-greedy']} />

      <h2>
        <Translated vi="Bài tập của sách">The book's exercises</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương kết thúc với sáu bài tập (pp. 249–250). Mỗi bài đều được giải ở đâu đó trong khóa học; sách không in đáp
            án, nên đáp án ở đây là của chúng tôi.
          </p>
        }
      >
        <p>
          The chapter ends with six exercises (pp. 249–250). Each is worked somewhere in this course; the book prints no
          answers, so the answers here are ours.
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
                <Translated vi="Nơi được giải ở đây">Where it is worked here</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1</td>
              <td>
                <Translated vi="Các warp phân kỳ ở vòng lặp thứ 5 của kernel đơn giản, 1,024 giá trị">
                  Divergent warps in the 5th iteration of the simple kernel, 1,024 values
                </Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      một câu hỏi và bản đồ warp trong <a href={href('ch10', '10.4')}>10.4</a>
                    </>
                  }
                >
                  <>
                    a question and the warp map in <a href={href('ch10', '10.4')}>10.4</a>
                  </>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td>
                <Translated vi="Cũng câu hỏi đó cho kernel hội tụ">The same for the convergent kernel</Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      một câu hỏi trong <a href={href('ch10', '10.4')}>10.4</a>
                    </>
                  }
                >
                  <>
                    a question in <a href={href('ch10', '10.4')}>10.4</a>
                  </>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td>
                <Translated vi="Kernel lật ngược, với tổng nằm ở phần tử cuối">
                  The mirrored kernel, with the total in the last element
                </Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      "Lật ngược nó" và một câu hỏi trong <a href={href('ch10', '10.4')}>10.4</a>
                    </>
                  }
                >
                  <>
                    "Mirror it" and a question in <a href={href('ch10', '10.4')}>10.4</a>
                  </>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td>
                <Translated vi="Fig. 10.20 dưới dạng phép rút gọn max">Fig. 10.20 as a max reduction</Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      bảng commit và hai câu hỏi trong <a href={href('ch10', '10.9')}>10.9</a>, phần văn bản của{' '}
                      <a href={href('ch10', '10.10')}>10.10</a>, và max theo hàng của <a href={href('ch10', 'labs')}>lab 10.3</a>
                    </>
                  }
                >
                  <>
                    the commit table and two questions in <a href={href('ch10', '10.9')}>10.9</a>, the text of{' '}
                    <a href={href('ch10', '10.10')}>10.10</a>, and the row max of <a href={href('ch10', 'labs')}>lab 10.3</a>
                  </>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>5</td>
              <td>
                <Translated vi="Fig. 10.20 cho độ dài bất kỳ">Fig. 10.20 for any length</Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      bộ tính đoạn trong <a href={href('ch10', '10.9')}>10.9</a>, và kernel (c) của{' '}
                      <a href={href('ch10', 'labs')}>lab 10.1</a>
                    </>
                  }
                >
                  <>
                    the segment calculator in <a href={href('ch10', '10.9')}>10.9</a>, and kernel (c) of{' '}
                    <a href={href('ch10', 'labs')}>lab 10.1</a>
                  </>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>6a</td>
              <td>
                <Translated vi={<>Lần theo {'{6, 2, 7, 4, 5, 8, 3, 1}'} qua kernel đơn giản</>}>
                  <>Tracing {'{6, 2, 7, 4, 5, 8, 3, 1}'} through the simple kernel</>
                </Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      trình chạy từng bước và một câu hỏi trong <a href={href('ch10', '10.3')}>10.3</a>
                    </>
                  }
                >
                  <>
                    the stepper and a question in <a href={href('ch10', '10.3')}>10.3</a>
                  </>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>6b</td>
              <td>
                <Translated vi="Cũng vậy qua kernel hội tụ">The same through the convergent kernel</Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      một câu hỏi trong <a href={href('ch10', '10.4')}>10.4</a>, và "Lật ngược nó" với dữ liệu của bài tập
                    </>
                  }
                >
                  <>
                    a question in <a href={href('ch10', '10.4')}>10.4</a>, and "Mirror it" with the exercise's data
                  </>
                </Translated>
              </td>
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
            Bản sách mà khóa học này dựa vào có một số chỗ có vẻ nhầm lẫn trong chương 10, phần lớn ở các tham chiếu dòng và
            hình, và vài đoạn mã hẹp hơn vẻ ngoài của chúng. Chúng được liệt kê để bạn không bị bối rối khi đọc sách song
            song. Ở chỗ nào làm thay đổi mã, các lab và snippet của khóa học có một phiên bản đã sửa và đã kiểm thử.
          </p>
        }
      >
        <p>
          The copy of the book this course was built from has a number of apparent slips in chapter 10, most of them in
          line and figure references, and a few listings that are narrower than they look. They are listed so that you are
          not thrown by them when you read the book alongside. Where one changes code, the course's labs and snippets have
          a corrected, tested version.
        </p>
      </Translated>
      <Translated
        vi={
          <ul data-list="errata">
            <li>
              Fig. 10.10 (p. 237) ghi vào <code>*output</code> nhưng không có tham số <code>output</code>. Kernel (a) của lab
              10.1 bổ sung nó.
            </li>
            <li>
              Figs. 10.16 và 10.18 để mọi lane của warp 0 đọc <code>partialSums_s[t]</code>, mảng có{' '}
              <code>BLOCK_DIM / WARP_SIZE</code> phần tử: chỉ đúng với 1,024 luồng trong warp rộng 32, và không bao giờ đúng
              trên warp rộng 64 của AMD (<a href={href('ch10', '10.8')}>10.8</a>). Các lab kiểm tra lần đọc và đệm bằng phần
              tử đơn vị.
            </li>
            <li>
              Fig. 10.18 không bao giờ dùng <code>N</code> của nó, và không có gì cho thấy <code>*output</code> được đặt về 0
              trước khi nó chạy; phần văn bản gọi nó là vài thay đổi so với Fig. 10.16, dù nó còn viết trực tiếp các hàm trợ
              giúp lane và warp. Fig. 10.20 không có kiểu trả về và dừng ở một dấu ba chấm. Kernel (c) của lab 10.1 là bản
              đầy đủ.
            </li>
            <li>
              Fig. 10.10 trộn <code>BLOCK_DIM</code> lúc biên dịch với <code>blockDim.x</code> lúc khởi chạy, hai giá trị phải
              khớp nhau, và giả định khối có kích thước lũy thừa của hai (<a href={href('ch10', '10.6')}>10.6</a>). Fig. 10.11
              không nói một lane nhận được gì khi lane nguồn của nó nằm ngoài warp hoặc không có trong mask (
              <a href={href('ch10', '10.7')}>10.7</a>). Cả hai đều là cách đọc của chúng tôi.
            </li>
            <li>
              Fig. 10.13 truyền literal <code>0xffffffff</code>, thứ HIP từ chối lúc biên dịch (
              <a href={href('ch10', '10.7')}>10.7</a>; notes/today/ch10-facts.md B1). Nó chỉ dành cho CUDA, không sai với
              CUDA.
            </li>
            <li>
              p. 234 gọi 384 là "gần một nửa của 736", tài nguyên của kernel đơn giản. p. 231 tính ra 864, và phép đếm của
              khóa học cũng vậy (<a href={href('ch10', '10.4')}>10.4</a>). Cùng trang đó nói số vòng lặp phân kỳ giảm từ
              mười xuống năm; với 256 giá trị thì là từ bảy xuống năm. 36 yêu cầu của p. 237 cho kernel hội tụ không kèm
              cách suy ra; chúng khớp với 12 lượt warp, mỗi lượt 3 truy cập (<a href={href('ch10', '10.5')}>10.5</a>).
            </li>
            <li>
              p. 229 nói bước nhảy tăng gấp đôi cho tới khi vượt quá <code>blockIdx.x</code>; mã dùng{' '}
              <code>blockDim.x</code>.
            </li>
            <li>p. 228 hứa sẽ gỡ giới hạn một khối ở mục 10.10; mục 10.9 mới làm điều đó.</li>
            <li>
              p. 222 đưa phần tử đơn vị của toán tử "the minimum (minimum)"; cái thứ hai là maximum. Cùng trang đó nói tổng
              là 34.6 "after iteration 5" của một vòng lặp năm phần tử đếm từ 0.
            </li>
            <li>Khung bên ở p. 227 cho giải đấu 8 đội của nó "a span of four steps"; ba vòng đấu là ba bước.</li>
            <li>
              Các tham chiếu hình và dòng lệch một: "Fig. 10.7" cho kernel của Fig. 10.8 (p. 235), "Fig. 10.9" cho kernel của
              Fig. 10.10 (p. 237), "line 06" cho dòng 05 của Fig. 10.8 (p. 233), "line 04" cho dòng 05 của Fig. 10.10 (p.
              236), và "lines 17–19" cho các dòng 15–17 của Fig. 10.14 (p. 241). p. 236 còn gọi lần ghi cuối là "the final
              write to input[0]"; mã ghi vào <code>*output</code>. p. 234 cho rằng hợp nhất truy cập bộ nhớ (coalescing) nằm
              ở chương 5; ấn bản này dạy nó ở chương 6 (<a href={href('ch06', '6.1')}>6.1</a>).
            </li>
            <li>
              Các vòng lặp được đánh số từ 0 trong Figs. 10.6 và 10.7 và từ 1 trong văn bản của pp. 230–234. Khóa học nói rõ
              cách đánh số mỗi lần.
            </li>
            <li>
              p. 243 nói kernel hai giai đoạn có "more control divergence". <code>warp_reduce</code> không có nhánh nào: thứ
              nó có nhiều hơn là các lane tính những tổng không ai đọc (<a href={href('ch10', '10.8')}>10.8</a>).
            </li>
            <li>
              Các liên kết của sách cho Thrust và CUB (p. 250) không còn hoạt động; các trang đó giờ nằm dưới{' '}
              <code>nvidia.github.io/cccl/unstable/</code> (ch10-facts A12).
            </li>
          </ul>
        }
      >
        <ul data-list="errata">
          <li>
            Fig. 10.10 (p. 237) writes <code>*output</code> but has no <code>output</code> parameter. Lab 10.1's kernel (a)
            restores it.
          </li>
          <li>
            Figs. 10.16 and 10.18 let every lane of warp 0 read <code>partialSums_s[t]</code>, which has{' '}
            <code>BLOCK_DIM / WARP_SIZE</code> entries: right only for 1,024 threads in 32-wide warps, and never on 64-wide
            AMD warps (<a href={href('ch10', '10.8')}>10.8</a>). The labs guard the read and pad with the identity.
          </li>
          <li>
            Fig. 10.18 never uses its <code>N</code>, and nothing shows <code>*output</code> being set to 0 before it runs;
            the text calls it a few changes from Fig. 10.16, though it also writes the lane and warp helpers out inline.
            Fig. 10.20 has no return type and stops at an ellipsis. Lab 10.1's kernel (c) is the whole thing.
          </li>
          <li>
            Fig. 10.10 mixes the compile-time <code>BLOCK_DIM</code> with the launched <code>blockDim.x</code>, which must
            agree, and assumes a power-of-two block (<a href={href('ch10', '10.6')}>10.6</a>). Fig. 10.11 does not say what
            a lane receives when its source lane is past the warp or not in the mask (<a href={href('ch10', '10.7')}>10.7</a>).
            Both are our readings.
          </li>
          <li>
            Fig. 10.13 passes the literal <code>0xffffffff</code>, which HIP rejects at compile time
            (<a href={href('ch10', '10.7')}>10.7</a>; notes/today/ch10-facts.md B1). It is CUDA-only, not wrong for CUDA.
          </li>
          <li>
            p. 234 calls 384 "almost half of 736", the resources of the simple kernel. p. 231 computes 864, and so does the
            course's count (<a href={href('ch10', '10.4')}>10.4</a>). The same page says the divergent iterations fall from
            ten to five; for 256 values the count is seven to five. p. 237's 36 requests for the convergent kernel come
            without a derivation; they match 12 warp-iterations of 3 accesses each (<a href={href('ch10', '10.5')}>10.5</a>).
          </li>
          <li>
            p. 229 says the stride doubles until it exceeds <code>blockIdx.x</code>; the code uses <code>blockDim.x</code>.
          </li>
          <li>
            p. 228 promises to lift the one-block limit in section 10.10; section 10.9 does it.
          </li>
          <li>
            p. 222 gives the identity of "the minimum (minimum)" operator; the second is the maximum. The same page says
            the sum is 34.6 "after iteration 5" of a five-element loop counted from 0.
          </li>
          <li>
            The sidebar on p. 227 gives its 8-team tournament "a span of four steps"; three rounds is three steps.
          </li>
          <li>
            Figure and line references that point one off: "Fig. 10.7" for the kernel of Fig. 10.8 (p. 235), "Fig. 10.9"
            for the kernel of Fig. 10.10 (p. 237), "line 06" for line 05 of Fig. 10.8 (p. 233), "line 04" for line 05 of
            Fig. 10.10 (p. 236), and "lines 17–19" for lines 15–17 of Fig. 10.14 (p. 241). p. 236 also calls the final write
            "the final write to input[0]"; the code writes <code>*output</code>. p. 234 credits coalescing to chapter 5;
            this edition teaches it in chapter 6 (<a href={href('ch06', '6.1')}>6.1</a>).
          </li>
          <li>
            Iterations are numbered from 0 in Figs. 10.6 and 10.7 and from 1 in the text of pp. 230–234. The course says
            which it means each time.
          </li>
          <li>
            p. 243 says the two-stage kernel has "more control divergence". <code>warp_reduce</code> has no branch: what it
            has more of is lanes computing sums nobody reads (<a href={href('ch10', '10.8')}>10.8</a>).
          </li>
          <li>
            The book's links for Thrust and CUB (p. 250) no longer resolve; the pages now live under{' '}
            <code>nvidia.github.io/cccl/unstable/</code> (ch10-facts A12).
          </li>
        </ul>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Đây là cách đọc của chúng tôi đối với một bản PDF, đã đối chiếu với tài liệu hiện hành của các hãng vào
              2026-11-20 (notes/today/ch10-facts.md). Nếu bản của bạn khác, hãy tin bản của bạn.
            </p>
          }
        >
          <p>
            These are our readings of one PDF, checked against current vendor documentation on 2026-11-20
            (notes/today/ch10-facts.md). If your copy differs, trust your copy.
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Một phần của sách được cố ý bỏ qua, để bạn phân biệt được chỗ lược bỏ với chỗ thiếu sót. Các địa điểm và ngày thi
            đấu World Cup 2010 trong khung bên (pp. 226–227) không được nhắc lại; ý tưởng vòng loại trực tiếp như một cây max
            nằm ở <a href={href('ch10', '10.2')}>10.2</a>. Mọi kernel ở đây đều là float32, dù p. 221 có nêu các kiểu khác;
            các phép rút gọn half và 8-bit sẽ quay lại ở module X2, và Metal không có <code>double</code>. Thrust và CUB được
            nêu tên, không được dùng, và max float xuyên khối của bài tập 4 được bàn tới, không được dựng, vì không có atomic
            max float nào khả chuyển. Các hàm bỏ phiếu warp của p. 241 được giải ở chương 12 (
            <a href={href('ch12', '12.3')}>12.3</a>). Softmax một lượt (online) là một phần của flash attention ở chương 20 (
            <a href={href('ch20', '20.5')}>20.5</a>).
          </p>
        }
      >
        <p>
          Some of the book is left out on purpose, so that you can tell an omission from a gap. The 2010 World Cup venues
          and dates of the sidebar (pp. 226–227) are not repeated; the knockout-as-max-tree idea is in{' '}
          <a href={href('ch10', '10.2')}>10.2</a>. Every kernel here is float32, though p. 221 names other types; half and
          8-bit reductions come back in module X2, and Metal has no <code>double</code>. Thrust and CUB are named, not
          used, and exercise 4's cross-block float max is discussed, not built, because no float atomic max is portable.
          The warp vote functions of p. 241 are worked in chapter 12 (<a href={href('ch12', '12.3')}>12.3</a>). The
          single-pass (online) softmax is part of chapter 20's flash attention (<a href={href('ch20', '20.5')}>20.5</a>).
        </p>
      </Translated>

      <h2>
        <Translated vi="Dự án cho tới giờ">The project so far</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Max, softmax và argmax theo hàng của lab 10.3 là mốc dự án M4 của dự án xuyên suốt: decode greedy từ đầu đến cuối
            trên logits <code>[batch × vocab]</code>, tạm thời trên một mô hình bảng tra.{' '}
            <span data-m4={m4.join(',') || 'none'}>
              {backends.length === 0
                ? 'Chưa có kết quả lab nào được ghi lại trên máy này.'
                : m4.length
                  ? `Lab 10.3 của bạn đạt trên ${m4.map(where).join(' và ')} trên máy này.`
                  : 'Lab 10.3 của bạn chưa đạt trên backend nào của máy này.'}
            </span>{' '}
            Xem các <a href={href('ch10', 'labs')}>lab</a>.
          </p>
        }
      >
        <p>
          Lab 10.3's row-wise max, softmax and argmax are milestone M4 of the through-line project: greedy decoding end to
          end over <code>[batch × vocab]</code> logits, on a lookup-table model for now.{' '}
          <span data-m4={m4.join(',') || 'none'}>
            {backends.length === 0
              ? 'No lab results are recorded on this machine yet.'
              : m4.length
                ? `Your own lab 10.3 passes on ${m4.map(where).join(' and ')} on this machine.`
                : 'Your own lab 10.3 does not pass on any backend of this machine yet.'}
          </span>{' '}
          See the <a href={href('ch10', 'labs')}>labs</a>.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chương 11 giữ nguyên cây và đổi thứ nó trả về. Một phép rút gọn cho một giá trị; một tổng tiền tố (prefix sum)
            cho mọi tổng bộ phận trên đường đi, tức tổng chạy mà một bộ lấy mẫu top-p cắt tại ngưỡng xác suất của nó. Sách
            gọi phép rút gọn là nền móng của mẫu đó (p. 249), và chương 11 là nơi nó gặp các thuật toán song song làm nhiều
            việc hơn vòng lặp tuần tự (p. 226). Bắt đầu ở <a href={href('ch11')}>chương 11</a>.
          </p>
        }
      >
        <p>
          Chapter 11 keeps the tree and changes what it returns. A reduction gives one value; a prefix sum gives every
          partial total along the way, the running sum that a top-p sampler cuts at its probability threshold. The book
          calls reduction the foundation of that pattern (p. 249), and chapter 11 is where it meets parallel algorithms
          that do more work than the sequential loop (p. 226). Start at <a href={href('ch11')}>chapter 11</a>.
        </p>
      </Translated>
    </>
  );
}
