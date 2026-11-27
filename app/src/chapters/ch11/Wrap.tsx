import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { functionSource } from '../../sim/toolchain';

// The kernel of the first question is a snippet of our own, loaded at run time, so there is
// no copy to drift: labs/ch11/snippets/wrap_mass_before.cu. Wrap.snippets.test.ts ties its
// helpers to lab 11.1, runs it on the course's emulator, and pins every number below.
export const MASS_FILE = 'ch11/snippets/wrap_mass_before.cu';
/** The shape the first question asks about: one block of 256 threads per row of 256 tokens, 32-lane warps. */
export const MASS = { block: 256, cols: 256, width: 32 } as const;
/** The tokens the first lane of every warp but the first writes: width, 2 width, ... below min(block, cols). */
export const warpStarts = (block: number, cols: number, width: number): number[] =>
  Array.from({ length: Math.ceil(Math.min(block, cols) / width) }, (_, k) => k * width).filter((c) => c > 0);

/**
 * Fig. 11.3 on one block of B slots with any operator: slots at or past values.length are
 * padded with `pad`, then for stride = 1, 2, 4, ... every slot i >= stride combines slot
 * i - stride into itself, all reading the values of the step before. Returns all B slots.
 */
export function koggeStoneBlock(values: number[], B: number, op: (a: number, b: number) => number, pad: number): number[] {
  let cur = Array.from({ length: B }, (_, i) => (i < values.length ? values[i] : pad));
  for (let stride = 1; stride < B; stride *= 2) {
    const prev = cur;
    cur = prev.map((v, i) => (i >= stride ? op(prev[i - stride], v) : v));
  }
  return cur;
}
/** The second question's row: six logits, all negative, in a block of 8 threads. */
export const LOGITS = [-3.5, -1.25, -2, -0.5, -4, -1];
export const LOGITS_BLOCK = 8;

/** The third question's row: one token of 0.9999, then 100,000 tokens of 10^-9. */
export const TAIL = { head: 0.9999, count: 100000, each: 1e-9 } as const;
/** What the two orders give for the last token in float32 (pinned by the unit test against src/sim/scan.ts). */
export const TAIL_RESULT = { loop: '0.9999', koggeStone: '1' } as const;
/** A top-p threshold between the two results, for the explanation. */
export const TAIL_P = 0.99995;

export const questionsWrap: Record<string, Question> = {
  'ch11/wrap/q-mass-before': {
    kind: 'choice',
    id: 'ch11/wrap/q-mass-before',
    prompt: (
      <Translated
        vi={
          <>
            Kernel ở trên ghi khối xác suất trước mỗi token của một hàng đã sắp, tức scan loại trừ mà mặt nạ top-p so với p, cho
            các hàng dài tối đa một khối token. Nó chạy scan khối bao gồm của lab 11.1, rồi lấy giá trị bao gồm của lane bên trái.
            Nó biên dịch và chạy được. Với khối {MASS.block} luồng trên các hàng {MASS.cols} token, trên một GPU có warp{' '}
            {MASS.width} lane, những đầu ra nào sai?
          </>
        }
      >
        <>
          The kernel above writes the mass before each token of a sorted row, the exclusive scan that a top-p mask compares
          with p, for rows of up to one block of tokens. It runs lab 11.1's inclusive block scan, then takes the inclusive
          value one lane to the left. It compiles and runs. With blocks of {MASS.block} threads on rows of {MASS.cols} tokens,
          on a GPU with {MASS.width}-lane warps, which outputs are wrong?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không đầu ra nào: một lần shuffle lên một đưa cho mọi lane giá trị bao gồm của lane bên trái.">
            {'None: a shuffle up by one hands every lane its left neighbour\'s inclusive value.'}
          </Translated>
        ),
        why: (
          <Translated vi="Mọi lane trừ một. Shuffle ở yên trong warp, và lane 0 không có lane nào bên trái nó ở đó.">
            Every lane but one. A shuffle stays inside the warp, and lane 0 has no lane to its left there.
          </Translated>
        ),
      },
      {
        label: tr(
          `Tokens ${warpStarts(MASS.block, MASS.cols, MASS.width).join(', ')} of every row: each holds its own inclusive mass.`,
          `Các token ${warpStarts(MASS.block, MASS.cols, MASS.width).join(', ')} của mỗi hàng: mỗi token giữ khối bao gồm của chính nó.`,
        ),
        correct: true,
      },
      {
        label: <Translated vi="Token 0 của mỗi hàng.">Token 0 of every row.</Translated>,
        why: (
          <Translated vi="Kernel đặt nó bằng 0, điều đó đúng: không có gì xếp trước token đầu tiên.">
            The kernel sets it to 0, which is right: nothing is ranked before the first token.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Mọi token của warp 1 đến 7.">Every token of warps 1 to 7.</Translated>,
        why: (
          <Translated vi="blockScan đã cộng tổng của các warp trước vào mọi lane của các warp đó. Mỗi warp chỉ có một lane đọc qua ranh giới.">
            {'blockScan has already added the earlier warps\' total to every lane of those warps. Only one lane per warp reads across a boundary.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <>
        <Translated
          vi={
            <p>
              Shuffle không bao giờ rời warp của nó, và một lane dưới độ dịch nhận lại chính giá trị của nó (mục{' '}
              <a href={href('ch11', '11.4')}>11.4</a>). Vì vậy lane 0 của mỗi warp nhận khối bao gồm của chính nó, vốn đã gồm xác
              suất của nó. Phép kiểm tra <code>threadIdx.x == 0</code> chỉ sửa điều đó cho warp đầu tiên. Trong một hàng{' '}
              {MASS.cols}, còn lại {warpStarts(MASS.block, MASS.cols, MASS.width).length} token mỗi hàng bị lớn hơn đúng bằng xác
              suất của chính chúng, và đó là những token mà mặt nạ top-p có thể làm sai: khi token vượt ngưỡng nằm ở ranh giới warp,{' '}
              <code>massBefore &lt; p</code> không thỏa và token bị bỏ. Đó là lỗi lệch một của quy tắc bao gồm, ở mỗi vị trí thứ 32.
            </p>
          }
        >
          <p>
            A shuffle never leaves its warp, and a lane below the shift gets its own value back (section{' '}
            <a href={href('ch11', '11.4')}>11.4</a>). Lane 0 of every warp therefore receives its own inclusive mass, which
            includes its own probability. The test <code>threadIdx.x == 0</code> repairs that for the first warp only. In a
            row of {MASS.cols} that leaves {warpStarts(MASS.block, MASS.cols, MASS.width).length} tokens per row too large by
            exactly their own probability, and those are the tokens a top-p mask can get wrong: when the crossing token sits
            at a warp boundary, <code>massBefore &lt; p</code> fails and the token is dropped. That is the inclusive rule's
            off-by-one, at every 32nd position.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Cách sửa là kiểm tra lane, không phải kiểm tra luồng. Lane 0 của warp w lấy tổng đã scan của các warp trước nó,{' '}
              <code>warpSums_s[w − 1]</code>, vẫn còn trong bộ nhớ chia sẻ sau <code>blockScan</code> (0 với warp 0). Trừ xác suất
              của chính token khỏi giá trị bao gồm của nó cũng được, với cái giá là thêm một lần làm tròn. Trên wavefront 64 lane,
              các token sai là 64, 128 và 192. <code>simd_prefix_exclusive_sum</code> của Metal trả về 0 ở lane đầu của mỗi SIMD
              group (ch11-facts C1), nên nó cũng cần cộng thêm cùng offset warp đó.
            </p>
          }
        >
          <p>
            The fix is a lane test, not a thread test. Lane 0 of warp w takes the scanned total of the warps before it,{' '}
            <code>warpSums_s[w − 1]</code>, still in shared memory after <code>blockScan</code> (0 for warp 0). Subtracting
            the token's own probability from its inclusive value also works, at the price of one more rounding. On a
            64-lane wavefront the wrong tokens are 64, 128 and 192. Metal's <code>simd_prefix_exclusive_sum</code> returns 0
            in the first lane of each SIMD-group (ch11-facts C1), so it needs the same warp offset added.
          </p>
        </Translated>
      </>
    ),
    source: 'ours, from ch. 11 §11.1 p. 253 and §11.4 p. 264; shuffle-up rule: notes/today/ch11-facts.md A3, B1, C1; run on the course emulator by Wrap.snippets.test.ts',
    vi: { source: 'của khóa học này, từ ch. 11 §11.1 p. 253 và §11.4 p. 264; quy tắc shuffle-up: notes/today/ch11-facts.md A3, B1, C1; chạy trên trình giả lập của khóa học bởi Wrap.snippets.test.ts' },
  },
  'ch11/wrap/q-maxpad': {
    kind: 'choice',
    id: 'ch11/wrap/q-maxpad',
    prompt: (
      <Translated
        vi={
          <>
            Bạn biến Fig. 11.3 thành phép lấy max chạy (<code>fmaxf</code> thay cho <code>+</code>), nhưng các luồng vượt cuối vẫn
            nạp <code>0.0f</code>, phần tử đơn vị của phép cộng, không phải của max. Bạn chạy nó trên {LOGITS.length} logits, tất
            cả đều âm ({LOGITS.join(', ')}), trong một khối {LOGITS_BLOCK} luồng. Những đầu ra được ghi nào sai?
          </>
        }
      >
        <>
          You turn Fig. 11.3 into a running maximum (<code>fmaxf</code> in place of <code>+</code>), but the threads past
          the end still load <code>0.0f</code>, the identity of addition, not of max. You run it on {LOGITS.length} logits,
          all negative ({LOGITS.join(', ')}), in a block of {LOGITS_BLOCK} threads. Which of the stored outputs are wrong?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Tất cả: 0 lớn hơn mọi logit, nên nó thắng mọi phép max.">All of them: 0 is larger than every logit, so it wins every maximum.</Translated>,
        why: (
          <Translated vi="Sẽ là vậy, nếu có đầu ra được ghi nào đọc một ô đệm. Mỗi bước đọc theo hướng nào?">
            It would, if any stored output ever read a padded slot. Which direction does each step read in?
          </Translated>
        ),
      },
      { label: <Translated vi="Không đầu ra nào.">None of them.</Translated>, correct: true },
      {
        label: <Translated vi="Chỉ đầu ra được ghi cuối cùng: nó nằm cạnh phần đệm.">The last stored output only: it is the one next to the padding.</Translated>,
        why: (
          <Translated vi="Nằm cạnh phần đệm không quan trọng. Mỗi bước đọc ô cách stride về bên trái, không bao giờ về bên phải.">
            Being next to the padding does not matter. Each step reads the slot stride to the left, never to the right.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Chỉ những đầu ra cùng warp với phần đệm.">Only those in the same warp as the padding.</Translated>,
        why: (
          <Translated vi="Ở đây chỉ có một warp, và đáp án không phụ thuộc vào warp: Fig. 11.3 làm việc qua bộ nhớ chia sẻ.">
            There is one warp here, and the answer does not depend on warps: Fig. 11.3 works through shared memory.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi bước của một mạng scan bao gồm đều đọc về bên trái: ô i kết hợp ô i − stride vào chính nó. Các ô đệm nằm bên phải
            mọi ô thật, nên các số 0 của chúng chỉ chảy vào các ô đệm khác, vốn không bao giờ được ghi ra. Các đầu ra là{' '}
            {koggeStoneBlock(LOGITS, LOGITS_BLOCK, Math.max, 0)
              .slice(0, LOGITS.length)
              .join(', ')}
            ; hai ô đệm kết thúc bằng 0 và không ai đọc chúng. Phần đệm sẽ quan trọng ở chỗ khác: một scan loại trừ đặt phần tử đơn
            vị vào ô 0 (p. 253), và một số 0 ở đó khẳng định một giá trị max mà không logit nào có. Một scan chạy từ phải sang trái
            cũng vậy. Dù sao hãy viết đúng phần tử đơn vị thật của toán tử (<code>-FLT_MAX</code> cho max, chương 10); nó không tốn
            gì và vẫn đúng sau lần sửa đổi kế tiếp.
          </p>
        }
      >
        <p>
          Every step of an inclusive scan network reads to the left: slot i combines slot i − stride into itself. The padded
          slots sit to the right of every real one, so their 0s flow only into other padded slots, which are never stored. The
          outputs are{' '}
          {koggeStoneBlock(LOGITS, LOGITS_BLOCK, Math.max, 0)
            .slice(0, LOGITS.length)
            .join(', ')}
          ; the two padded slots end as 0 and nobody reads them. The pad would matter elsewhere: an exclusive scan puts the
          identity in slot 0 (p. 253), and a 0 there claims a maximum no logit had. So would a scan that runs right to left.
          Write the operator's real identity anyway (<code>-FLT_MAX</code> for max, chapter 10); it costs nothing and
          survives the next change.
        </p>
      </Translated>
    ),
    source: 'ours, from ch. 11 §11.1 p. 253 (identity) and Fig. 11.3, p. 256; ch. 10 §10.1 for the identity of max',
    vi: { source: 'của khóa học này, từ ch. 11 §11.1 p. 253 (phần tử đơn vị) và Fig. 11.3, p. 256; ch. 10 §10.1 cho phần tử đơn vị của max' },
  },
  'ch11/wrap/q-tail': {
    kind: 'choice',
    id: 'ch11/wrap/q-tail',
    prompt: (
      <Translated
        vi={
          <>
            Một hàng đã sắp gồm 100,001 xác suất: một token {TAIL.head}, rồi {TAIL.count.toLocaleString('en-US')} token mỗi token
            10<sup>−9</sup>, tổng cộng thêm 10<sup>−4</sup>. Bạn tính scan bao gồm của nó bằng float32 theo hai cách: với vòng lặp
            của Fig. 11.1 và theo thứ tự của Kogge-Stone. Mỗi cách cho ra gì ở token cuối?
          </>
        }
      >
        <>
          A sorted row of 100,001 probabilities: one token of {TAIL.head}, then {TAIL.count.toLocaleString('en-US')} tokens
          of 10<sup>−9</sup>, another 10<sup>−4</sup> in all. You compute its inclusive scan in float32 two ways: with Fig.
          11.1's loop and in Kogge-Stone's order. What does each give for the last token?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Cả hai khoảng 1, chênh vài đơn vị ở vị trí cuối.">Both about 1, within a few units in the last place.</Translated>,
        why: (
          <Translated vi="Đi theo vòng lặp: mỗi bước cộng 10⁻⁹ vào một tổng chạy gần 0.9999. Như vậy có đủ để thay đổi một số float32 ở đó không?">
            Follow the loop: each step adds 10⁻⁹ to a running sum near 0.9999. Is that enough to change a float32 there?
          </Translated>
        ),
      },
      { label: tr(`The loop ${TAIL_RESULT.loop}, Kogge-Stone ${TAIL_RESULT.koggeStone}.`, `Vòng lặp ${TAIL_RESULT.loop}, Kogge-Stone ${TAIL_RESULT.koggeStone}.`), correct: true },
      {
        label: tr(`The loop ${TAIL_RESULT.koggeStone}, Kogge-Stone ${TAIL_RESULT.loop}.`, `Vòng lặp ${TAIL_RESULT.koggeStone}, Kogge-Stone ${TAIL_RESULT.loop}.`),
        why: (
          <Translated vi="Ngược lại. Thứ tự nào cộng các giá trị rất nhỏ với nhau trước khi cộng chúng vào giá trị lớn?">
            The other way round. Which order adds the tiny values to each other before adding them to the large one?
          </Translated>
        ),
      },
      {
        label: tr(`Both ${TAIL_RESULT.loop}: 10⁻⁹ is too small to register next to 0.9999 in float32.`, `Cả hai ${TAIL_RESULT.loop}: 10⁻⁹ quá nhỏ để ghi nhận được cạnh 0.9999 trong float32.`),
        why: (
          <Translated vi="Một số thì đúng là vậy. Hai số cộng lại, rồi bốn, rồi tám, thì không, một khi chúng đủ lớn.">
            One of them is. Two of them added together, then four, then eight, are not, once they are big enough.
          </Translated>
        ),
      },
    ],
    explain: (
      <>
        <Translated
          vi={
            <p>
              Các số float32 ngay dưới 1 cách nhau 2<sup>−24</sup>, khoảng 6 × 10<sup>−8</sup>, nên cộng bất cứ thứ gì nhỏ hơn một
              nửa khoảng đó vào 0.9999 đều làm tròn về 0.9999. Vòng lặp cộng phần đuôi từng token một vào tổng chạy lớn và mất toàn
              bộ 10<sup>−4</sup> đó. Kogge-Stone cộng các phần tử lân cận trước: từng cặp, rồi nhóm 4, 8, và cứ thế, nên các giá trị
              của phần đuôi được cộng với nhau thành những khúc đủ lớn để được tính trước khi gặp phần đầu. Token cuối của nó ra 1,
              đúng như số học chính xác.
            </p>
          }
        >
          <p>
            Float32 numbers just below 1 are 2<sup>−24</sup>, about 6 × 10<sup>−8</sup>, apart, so adding anything under
            half of that to 0.9999 rounds back to 0.9999. The loop adds the tail one token at a time to the large running sum
            and loses all 10<sup>−4</sup> of it. Kogge-Stone adds neighbours first: pairs, then groups of 4, 8, and so on, so
            the tail's values are summed among themselves into chunks large enough to count before they meet the head. Its
            last token comes out as 1, as exact arithmetic says.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Hai bài học. Vòng lặp thường không phải thứ tự chính xác nhất, chỉ là thứ tự tái lập được nhất, đó là lý do các lab chạy
              nó bằng double làm tham chiếu. Và trong một bộ lấy mẫu top-p với p = {TAIL_P}, khối xác suất của vòng lặp float không
              bao giờ đạt p, nên nó giữ cả 100,001 token, trong khi bản dạng cây giữ khoảng một nửa.
            </p>
          }
        >
          <p>
            Two lessons. The plain loop is not the most accurate order, only the most reproducible one, which is why the labs
            run it in double as their reference. And in a top-p sampler with p = {TAIL_P}, the float loop's mass never
            reaches p, so it keeps all 100,001 tokens, where the tree's keeps about half of them.
          </p>
        </Translated>
      </>
    ),
    source: 'ours, from ch. 11 §11.2, pp. 255–256 (the order of Kogge-Stone\'s additions); float addition: notes/today/ch10-facts.md A9; computed with src/sim/scan.ts by Wrap.snippets.test.ts',
    vi: { source: 'của khóa học này, từ ch. 11 §11.2, pp. 255–256 (thứ tự các phép cộng của Kogge-Stone); phép cộng float: notes/today/ch10-facts.md A9; tính bằng src/sim/scan.ts bởi Wrap.snippets.test.ts' },
  },
  'ch11/wrap/q-batch1': {
    kind: 'choice',
    id: 'ch11/wrap/q-batch1',
    prompt: (
      <Translated
        vi={
          <>
            Thiết kế của lab 11.6 giao cho mỗi hàng xác suất một khối 256 luồng. Khi decode một chuỗi, batch là 1: một hàng duy
            nhất gồm 128,256 token. Thay đổi nào cho phép scan dùng toàn bộ GPU?
          </>
        }
      >
        <>
          Lab 11.6's design gives each row of probabilities one block of 256 threads. Decoding one sequence, the batch is 1:
          a single row of 128,256 tokens. What change lets the scan use the whole GPU?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Khởi chạy 1,024 luồng mỗi khối thay vì 256.">Launch 1,024 threads per block instead of 256.</Translated>,
        why: <Translated vi="Vẫn một khối, trên một SM; phần còn lại của thiết bị vẫn rảnh.">Still one block, on one SM; the rest of the device stays idle.</Translated>,
      },
      {
        label: (
          <Translated vi="Chia hàng cho nhiều khối và ghép các scan của chúng, như lab 11.4 làm (hoặc lab 11.5 trên CUDA).">
            Split the row across many blocks and join their scans, as lab 11.4 does (or lab 11.5 on CUDA).
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Tăng hệ số gộp việc, để mỗi luồng scan nhiều phần hơn của hàng.">Raise the coarsening factor, so that each thread scans more of the row.</Translated>,
        why: <Translated vi="Điều đó cho các luồng của một khối ấy nhiều việc hơn. Nó không thêm khối nào.">{'That gives the one block\'s threads more work each. It adds no blocks.'}</Translated>,
      },
      {
        label: <Translated vi="Dùng Brent-Kung bên trong khối, để tiết kiệm phép cộng.">Use Brent-Kung inside the block, to save additions.</Translated>,
        why: (
          <Translated vi="Phép cộng không phải thứ giới hạn một scan bị giới hạn bởi bộ nhớ, và hàng vẫn chỉ được một khối đọc.">
            Additions are not what limits a memory-bound scan, and the row is still read by one block.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi hàng một khối là đúng khi có nhiều hàng: chúng độc lập, nên 64 hàng lấp 64 khối và không cần giao tiếp giữa các
            khối. Ở batch 1, cùng thiết kế đó chạy một khối trên cả thiết bị, và benchmark của lab 11.6 đo đúng dạng đó. Đáp án là
            của mục <a href={href('ch11', '11.9')}>11.9</a>: nhiều khối trên một hàng, ghép lại bằng một scan các tổng của chúng,
            bằng nhiều kernel ở mọi nơi hoặc một kernel trên CUDA. Các dự án cũng làm vậy: trên CUDA, llama.cpp giao một hàng dài so
            với batch của nó, trước hết là một hàng đơn lẻ, cho device scan của CUB, và trên GPU Apple, PyTorch chia một trục dài
            cho các threadgroup trong ba lượt.
          </p>
        }
      >
        <p>
          One block per row is right when there are many rows: they are independent, so 64 rows fill 64 blocks and need no
          communication between blocks. At batch 1 the same design runs one block on the whole device, and lab 11.6's
          benchmark measures exactly that shape. The answer is section <a href={href('ch11', '11.9')}>11.9</a>'s: many blocks
          on the one row, joined by a scan of their totals, in several kernels anywhere or in one kernel on CUDA. The
          projects do the same: on CUDA llama.cpp hands a row that is long for its batch, a single row above all, to CUB's
          device scan, and on Apple GPUs PyTorch splits a long axis across threadgroups in three passes.
        </p>
      </Translated>
    ),
    source: 'ours, from ch. 11 §11.9, pp. 273–277; notes/today/ch11-facts.md D1, D4',
    vi: { source: 'của khóa học này, từ ch. 11 §11.9, pp. 273–277; notes/today/ch11-facts.md D1, D4' },
  },
  'ch11/wrap/q-subtract': {
    kind: 'choice',
    id: 'ch11/wrap/q-subtract',
    prompt: (
      <Translated
        vi={
          <>
            Một batch các chuỗi được đóng gói nối đuôi trong một buffer, và bạn đã có G, scan bao gồm của cả buffer. Chuỗi k chiếm
            các vị trí s đến e. Với một vị trí i bên trong nó, biểu thức nào cho ra scan bao gồm riêng của chuỗi k, tức scan bắt
            đầu lại tại s?
          </>
        }
      >
        <>
          A batch of sequences is packed end to end in one buffer, and you already have G, the inclusive scan of the whole
          buffer. Sequence k occupies positions s to e. For a position i inside it, which gives sequence k's own inclusive
          scan, the one that restarts at s?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="G[i] − G[s − 1], với G[−1] coi là 0.">G[i] − G[s − 1], with G[−1] taken as 0.</Translated>, correct: true },
      {
        label: <Translated vi="G[i] − G[s], tổng tính đến đầu chuỗi.">G[i] − G[s], the total up to the sequence start.</Translated>,
        why: <Translated vi="G[s] đã gồm phần tử đầu của chuỗi, nên cách này bỏ mất nó.">{'G[s] already includes the sequence\'s first element, so this drops it.'}</Translated>,
      },
      {
        label: 'G[i] − x[s].',
        why: <Translated vi="Cách đó bỏ phần tử đầu của chuỗi và giữ mọi thứ trước nó.">That removes the first element of the sequence and keeps everything before it.</Translated>,
      },
      {
        label: <Translated vi="G[e] − G[i], phần của chuỗi tính đến i.">G[e] − G[i], the part of the sequence up to i.</Translated>,
        why: <Translated vi="Đó là khối sau vị trí i trong chuỗi, một hậu tố, không phải tiền tố.">That is the mass after position i within the sequence, a suffix, not a prefix.</Translated>,
      },
    ],
    explain: (
      <>
        <Translated
          vi={
            <p>
              G[s − 1] là tổng của mọi thứ trước chuỗi (một scan loại trừ tại s, p. 253), và trừ nó đi còn lại tổng từ s đến i. Cách
              này được vì phép cộng có thể đảo ngược. Một scan max không thể bắt đầu lại theo cách đó, vì không gì đảo ngược được
              max: khi đó bạn cần một scan đặt lại ở bất cứ đâu có cờ đánh dấu đầu chuỗi, tức segmented scan.
            </p>
          }
        >
          <p>
            G[s − 1] is the total of everything before the sequence (an exclusive scan at s, p. 253), and subtracting it
            leaves the sum from s to i. This works because addition can be undone. A max scan cannot be restarted that way,
            since nothing undoes a max: there you need a scan that resets wherever a flag marks a sequence start, a segmented
            scan.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Với float, phép trừ có cái giá của nó. G tăng dần trên cả batch, và hiệu của hai tổng lớn gần bằng nhau chỉ giữ được số
              bit mà các số lớn còn lại: một chuỗi ngắn ở cuối một batch lớn nhận một scan kém chính xác hơn nhiều so với khi tính
              riêng.
            </p>
          }
        >
          <p>
            In float the subtraction has a price. G grows over the whole batch, and the difference of two large, nearly equal
            totals keeps only the bits the large numbers had left: a short sequence late in a big batch gets a scan far less
            accurate than one computed on its own.
          </p>
        </Translated>
      </>
    ),
    source: 'ours, from the definitions of ch. 11 §11.1, pp. 252–253',
    vi: { source: 'của khóa học này, từ các định nghĩa của ch. 11 §11.1, pp. 252–253' },
  },
};

/** The kernel of the first question, from the snippet file. */
function MassBeforeKernel() {
  const file = useLabFile(MASS_FILE);
  const fn = file && !file.startsWith('// could not load') ? functionSource(file, 'massBeforeRows') : null;
  return (
    <Code
      tabs={[
        {
          label: tr('CUDA (not a lab file: for this question only)', 'CUDA (không phải tệp lab: chỉ cho câu hỏi này)'),
          code: fn ? fn.text : file ?? `// loading labs/${MASS_FILE}`,
          status: tr('front-end-checked and emulator-checked; wrong on purpose', 'front-end-checked và emulator-checked; cố ý sai'),
          tiers: {
            probs: 'global',
            massBefore: 'global',
            row: 'global',
            out: 'global',
            warpSums_s: 'shared',
            p: 'reg',
            inclusive: 'reg',
            before: 'reg',
          },
        },
      ]}
      caption={
        <Translated
          vi={
            <>
              Từ <code>labs/{MASS_FILE}</code>. Kiểu mặt nạ, <code>laneIdx</code>, <code>warpIdx</code>, <code>warpScan</code> và{' '}
              <code>blockScan</code> của nó giống hệt của lab 11.1 đến từng ký tự. Các unit test chạy nó trên trình giả lập của khóa
              học.
            </>
          }
        >
          <>
            From <code>labs/{MASS_FILE}</code>. Its mask type, <code>laneIdx</code>, <code>warpIdx</code>,{' '}
            <code>warpScan</code> and <code>blockScan</code> are lab 11.1's, character for character. The unit tests run it
            on the course's emulator.
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
  const m5 = backends.filter((b) => labStatus(reports[b], 'ch11/lab6_topp', 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch11/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}><Translated vi="Những gì mang theo">What to carry forward</Translated></h2>
      <ul>
        <li>
          <Translated
            vi={
              <>
                Một scan trả về mọi tổng riêng phần, không chỉ tổng cuối. Dạng bao gồm và loại trừ lệch nhau một vị trí: scan loại trừ
                cho các offset bắt đầu và khối xác suất trước một token, và top-p giữ một token khi khối đó còn dưới p.
              </>
            }
          >
            A scan returns every partial total, not just the last. The inclusive and exclusive forms differ by one position:
            the exclusive scan gives start offsets and the mass before a token, and top-p keeps a token while that mass is
            below p.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Mọi scan song song đều nhóm lại các phép cộng, nên nó cần một toán tử có tính kết hợp. Phép cộng float chỉ gần có tính
                kết hợp: hai thứ tự cho các bit cuối khác nhau, và vòng lặp thường thậm chí không phải thứ tự chính xác nhất. Hãy kiểm
                thử với dữ liệu chính xác và một dải, so với một tham chiếu chạy bằng double.
              </>
            }
          >
            Every parallel scan regroups the additions, so it needs an associative operator. Float addition is only nearly
            associative: two orders give different last bits, and the plain loop is not even the most accurate one. Test
            with exact data and a band, against a reference run in double.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Kogge-Stone mất log₂N bước và làm N·log₂N − (N − 1) phép cộng. Rào chắn thứ hai của nó chỉ bảo vệ một phụ thuộc giả,
                mà hai buffer loại bỏ được; shuffle trong warp và scan-scan-cộng để lại hai rào chắn mỗi lần scan khối.
              </>
            }
          >
            Kogge-Stone takes log₂N steps and does N·log₂N − (N − 1) additions. Its second barrier guards only a false
            dependence, which two buffers remove; warp shuffles and scan-scan-add leave two barriers per block scan.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Phần việc thêm có giá khi ít đơn vị thực thi. Gộp việc cho mỗi luồng một dãy tuần tự, tổng cộng khoảng 2N phép cộng,
                và khối vẫn giữ được hợp nhất truy cập bằng cách nạp qua bộ nhớ chia sẻ từng khúc một, rồi làm việc trong thanh ghi.
              </>
            }
          >
            Extra work costs when execution units are few. Coarsening gives each thread a sequential run, about 2N additions
            in all, and the block stays coalesced by loading through shared memory chunk by chunk, then working in registers.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Một scan làm khoảng một phép cộng mỗi 8 byte, nên nó bị giới hạn bởi bộ nhớ: hãy đánh giá nó so với băng thông sao chép
                của bạn. Giữa các khối, các thiết kế khác nhau về số byte mỗi phần tử, 16, 12 hoặc 8.
              </>
            }
          >
            A scan does about one addition per 8 bytes, so it is memory-bound: judge it against your copy bandwidth.
            Across blocks the designs differ in bytes per element, 16, 12 or 8.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Các kernel riêng ghép các khối một cách an toàn trên mọi backend. Một kernel duy nhất cần các khối chờ nhau: CUDA ghi
                nhận đảm bảo tiến triển giúp chỉ số giành qua bộ đếm và các cờ release/acquire an toàn; AMD và Apple không ghi nhận
                gì, nên trên Metal scan toàn cục là nhiều lần dispatch.
              </>
            }
          >
            Separate kernels join blocks safely on every backend. One kernel needs blocks that wait for each other: CUDA
            documents the progress that makes a counter-claimed index and release/acquire flags safe; AMD and Apple document
            none, so on Metal the global scan is several dispatches.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Brent-Kung làm ít việc hơn trong nhiều bước hơn. Bên trong một warp các lane đằng nào cũng có sẵn, nên Kogge-Stone
                thắng ở phần scan song song còn lại sau khi gộp việc.
              </>
            }
          >
            Brent-Kung does less work in more steps. Inside a warp the lanes are there anyway, so Kogge-Stone wins where the
            parallel scan is left after coarsening.
          </Translated>
        </li>
      </ul>
      <Source section="11.11" pages="285–286" />

      <h2><Translated vi="Kiểm tra tổng hợp">A mixed check</Translated></h2>
      <p>
        <Translated vi="Các câu này dựa trên cả chương và trên những tình huống mà các mục chưa trình bày.">
          These draw on the whole chapter and on situations the sections did not show.
        </Translated>
        {asked ? tr(` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.`, ` Đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu hỏi, tính theo câu trả lời gần nhất của mỗi câu.`) : ''}
      </p>
      <Beyond>
        <Translated vi={<p>Kernel và các tình huống dưới đây là của chúng tôi.</p>}>
          <p>The kernel and the situations below are ours.</p>
        </Translated>
      </Beyond>
      <MassBeforeKernel />
      <Ask q={q['ch11/wrap/q-mass-before']} label={tr('Find the defect', 'Tìm lỗi')} />
      <Ask q={q['ch11/wrap/q-maxpad']} />
      <Ask q={q['ch11/wrap/q-tail']} />
      <Ask q={q['ch11/wrap/q-batch1']} />
      <Ask q={q['ch11/wrap/q-subtract']} />

      <h2><Translated vi="Các bài tập của sách">The book's exercises</Translated></h2>
      <Translated
        vi={
          <p>
            Chương kết thúc với bảy bài tập (p. 286). Mỗi bài được giải ở đâu đó trong khóa học này; sách không in đáp án, nên đáp
            án ở đây là của chúng tôi.
          </p>
        }
      >
        <p>
          The chapter ends with seven exercises (p. 286). Each is worked somewhere in this course; the book prints no
          answers, so the answers here are ours.
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
              <td><Translated vi="Kogge-Stone trên 4 6 7 1 2 8 5 2, từng bước">Kogge-Stone on 4 6 7 1 2 8 5 2, step by step</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      trình chạy từng bước và một câu hỏi ở <a href={href('ch11', '11.2')}>11.2</a>; các trình chạy từng bước của{' '}
                      <a href={href('ch11', '11.3')}>11.3</a> chạy trên cùng các giá trị
                    </>
                  }
                >
                  the stepper and a question in <a href={href('ch11', '11.2')}>11.2</a>; the steppers of{' '}
                  <a href={href('ch11', '11.3')}>11.3</a> run on the same values
                </Translated>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td><Translated vi="Phân kỳ chỉ ở warp đầu tiên, với stride đến 16">Divergence only in the first warp, for strides up to 16</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      trình khám phá "Warp nào phân kỳ?" và một câu hỏi ở độ rộng 64 trong <a href={href('ch11', '11.2')}>11.2</a>
                    </>
                  }
                >
                  the "Which warps diverge?" explorer and a question at width 64 in <a href={href('ch11', '11.2')}>11.2</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td><Translated vi="Số phép cộng của Fig. 11.3 trên 2,048 phần tử">Additions of Fig. 11.3 on 2,048 elements</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      một câu hỏi ở <a href={href('ch11', '11.2')}>11.2</a>, với cả hai cách hiểu bài tập
                    </>
                  }
                >
                  a question in <a href={href('ch11', '11.2')}>11.2</a>, with both readings of the exercise
                </Translated>
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td><Translated vi="Nạp vector và không xung đột bank trong Fig. 11.13">Vector loads and no bank conflicts in Fig. 11.13</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      bản đồ bank và các câu hỏi của nó ở <a href={href('ch11', '11.7')}>11.7</a>; một mục "đi xa hơn" chưa kiểm thử trong{' '}
                      <a href={href('ch11', 'labs')}>lab 11.2</a>
                    </>
                  }
                >
                  the bank map and its questions in <a href={href('ch11', '11.7')}>11.7</a>; an untested "go further" in{' '}
                  <a href={href('ch11', 'labs')}>lab 11.2</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>5</td>
              <td>
                <Translated vi={<><code>interBlockScan</code> với decoupled lookback</>}>
                  <code>interBlockScan</code> with decoupled lookback
                </Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      trình mô phỏng "Ai chờ ai" và một câu hỏi ở <a href={href('ch11', '11.9')}>11.9</a>; một mục "đi xa hơn" chưa kiểm thử
                      trong <a href={href('ch11', 'labs')}>lab 11.5</a>
                    </>
                  }
                >
                  the "Who waits for whom" simulator and a question in <a href={href('ch11', '11.9')}>11.9</a>; an untested
                  "go further" in <a href={href('ch11', 'labs')}>lab 11.5</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>6</td>
              <td><Translated vi="Brent-Kung trên 4 6 7 1 2 8 5 2, từng bước">Brent-Kung on 4 6 7 1 2 8 5 2, step by step</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      trình chạy từng bước up-sweep và down-sweep cùng một câu hỏi ở <a href={href('ch11', '11.10')}>11.10</a>
                    </>
                  }
                >
                  the up-sweep and down-sweep stepper and a question in <a href={href('ch11', '11.10')}>11.10</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>7</td>
              <td><Translated vi="Một kernel Brent-Kung mức khối">A block-level Brent-Kung kernel</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      "Exercise 7: a block-level kernel" ở <a href={href('ch11', '11.10')}>11.10</a>, emulator-checked, với một câu hỏi về
                      các rào chắn của nó
                    </>
                  }
                >
                  "Exercise 7: a block-level kernel" in <a href={href('ch11', '11.10')}>11.10</a>, emulator-checked, with a
                  question on its barriers
                </Translated>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2><Translated vi="Những chỗ PDF có vẻ sai">Where the PDF looks wrong</Translated></h2>
      <Translated
        vi={
          <p>
            Bản sách mà khóa học này dựa vào có một số sơ suất rõ ràng ở chương 11, và vài đoạn mã hẹp hơn vẻ ngoài của chúng.
            Chúng được liệt kê để bạn không bị bối rối khi đọc sách song song. Chỗ nào ảnh hưởng đến mã, các lab và snippet có bản
            đã sửa và kiểm thử.
          </p>
        }
      >
        <p>
          The copy of the book this course was built from has a number of apparent slips in chapter 11, and several listings
          that are narrower than they look. They are listed so that you are not thrown by them when you read the book
          alongside. Where one changes code, the labs and snippets have a corrected, tested version.
        </p>
      </Translated>
      <ul data-list="errata">
        <li>
          <Translated
            vi={
              <>
                Figs. 11.12, 11.13 và 11.18 nhận <code>N</code> mà không hề dùng, không kiểm tra biên: chúng chỉ đúng khi N là một số
                nguyên lần đoạn khối, điều mà văn bản không hề nói. Lab 11.2, 11.4 và 11.5 kiểm tra mọi lần nạp và ghi.
              </>
            }
          >
            Figs. 11.12, 11.13 and 11.18 take <code>N</code> and never use it, with no bounds checks: they are right only
            when N is a whole number of block segments, which the text never says. Labs 11.2, 11.4 and 11.5 check every load
            and store.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Fig. 11.18 (p. 281) thiếu dấu phẩy giữa <code>unsigned int* flags</code> và <code>unsigned int N</code>, và sẽ không
                biên dịch được như bản in. Dòng 12 của nó dùng <code>blockDim.x</code> trong khi dòng 14 dùng{' '}
                <code>BLOCK_DIM</code>.
              </>
            }
          >
            Fig. 11.18 (p. 281) has no comma between <code>unsigned int* flags</code> and <code>unsigned int N</code>, and
            would not compile as printed. Its line 12 uses <code>blockDim.x</code> where line 14 uses{' '}
            <code>BLOCK_DIM</code>.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Các cờ của Fig. 11.17 và bộ đếm của Fig. 11.18 phải bằng 0 trước mỗi lần khởi chạy. Sách nói vậy cho bộ đếm (p. 280)
                và chỉ ngụ ý cho các cờ (p. 278). Hàm khởi chạy của lab 11.5 đặt cả hai về 0.
              </>
            }
          >
            The flags of Fig. 11.17 and the counter of Fig. 11.18 must be zero before every launch. The book says so for the
            counter (p. 280) and only implies it for the flags (p. 278). Lab 11.5's launcher zeroes both.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Fig. 11.17 đọc cờ bằng <code>fetch_add(0, acquire)</code>. Cách đó chạy được, nhưng nó ghi ở chỗ chỉ cần đọc; ví dụ
                của chính NVIDIA cho giao thức này dùng <code>load(acquire)</code> (ch11-facts A5), và lab 11.5 cũng vậy.
              </>
            }
          >
            Fig. 11.17 reads the flag with <code>fetch_add(0, acquire)</code>. That works, but it writes where only a read is
            needed; NVIDIA's own example of the protocol uses <code>load(acquire)</code> (ch11-facts A5), and so does lab
            11.5.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Fig. 11.8 truyền hằng <code>0xffffffff</code> và giả định 32 lane, điều mà HIP từ chối lúc biên dịch và một wavefront
                64 lane sẽ scan thành hai nửa (<a href={href('ch11', '11.4')}>11.4</a>; ch11-facts B1). Fig. 11.9 khai báo một mảng{' '}
                <code>__shared__</code> bên trong một hàm <code>__device__</code>, điều mà Metal từ chối và HIP không ghi nhận. Cả hai
                chỉ dành cho CUDA, không sai với CUDA; các lab dùng một mặt nạ có kiểu, <code>warpSize</code>, và một mảng khai báo
                trong kernel.
              </>
            }
          >
            Fig. 11.8 passes the literal <code>0xffffffff</code> and assumes 32 lanes, which HIP rejects at compile time and
            a 64-lane wavefront scans in halves (<a href={href('ch11', '11.4')}>11.4</a>; ch11-facts B1). Fig. 11.9 declares a{' '}
            <code>__shared__</code> array inside a <code>__device__</code> function, which Metal refuses and HIP does not
            document. Both are CUDA-only, not wrong for CUDA; the labs use a typed mask, <code>warpSize</code>, and an array
            declared in the kernel.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Fig. 11.3 (p. 256) in dòng 14 là <code>__ syncthreads();</code> với một dấu cách, và p. 258 có một lần viết{' '}
                <code>__syncthread()</code>.
              </>
            }
          >
            Fig. 11.3 (p. 256) prints line 14 as <code>__ syncthreads();</code> with a space, and p. 258 writes{' '}
            <code>__syncthread()</code> once.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 258 cho giá trị bị hỏng do race condition là 2x₁ + 2x₂ + x₃ + x₄ + x₅ + x₆. Tính kỹ thì là x₀ + 2x₁ + 2x₂ + x₃ +
                x₄ + x₅ + x₆ (<a href={href('ch11', '11.2')}>11.2</a>). Ý chính, x₁ và x₂ bị tính hai lần, vẫn đúng.
              </>
            }
          >
            p. 258 gives the corrupted value of the race as 2x₁ + 2x₂ + x₃ + x₄ + x₅ + x₆. Worked through, it is x₀ + 2x₁ +
            2x₂ + x₃ + x₄ + x₅ + x₆ (<a href={href('ch11', '11.2')}>11.2</a>). The point, x₁ and x₂ counted twice, stands.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 259 cho luồng 1 ghi Σx₁..x₂ và đọc x₁ trong ví dụ double buffering. Ở lần lặp 0, luồng 1 ghi Σx₀..x₁, và chính luồng
                2 mới đọc x₁ và ghi Σx₁..x₂ (<a href={href('ch11', '11.3')}>11.3</a>).
              </>
            }
          >
            p. 259 has thread 1 writing Σx₁..x₂ and reading x₁ in the double-buffering example. In iteration 0 thread 1
            writes Σx₀..x₁, and it is thread 2 that reads x₁ and writes Σx₁..x₂ (<a href={href('ch11', '11.3')}>11.3</a>).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 265 đối chiếu "the code in Fig. 11.5" với "the code in Fig. 11.5"; cái thứ hai là Fig. 11.9. p. 269 gọi scan khối
                là <code>blockSum</code>; hàm đó là <code>blockScan</code>.
              </>
            }
          >
            p. 265 contrasts "the code in Fig. 11.5" with "the code in Fig. 11.5"; the second is Fig. 11.9. p. 269 calls the
            block scan <code>blockSum</code>; the function is <code>blockScan</code>.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 283 đặt pha rút gọn ở "the bottom portion of Fig. 11.19"; p. 282 đặt nó ở nửa trên. Trong Fig. 11.20 (p. 284), cột
                9 và 11 được tô đậm hơn một mức so với chú giải ở hàng đầu, và cột 9 ở hàng thứ hai (
                <a href={href('ch11', '11.10')}>11.10</a>).
              </>
            }
          >
            p. 283 places the reduction phase in "the bottom portion of Fig. 11.19"; p. 282 puts it in the top half. In Fig.
            11.20 (p. 284), columns 9 and 11 are shaded one level darker than the legend gives in the first row, and column 9
            in the second (<a href={href('ch11', '11.10')}>11.10</a>).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Bài tập 3 hỏi số phép cộng trên 2,048 phần tử mà không nói đó là một đoạn (20,481) hay hai khối 1,024 (18,434), vì một
                khối chứa tối đa 1,024 luồng.
              </>
            }
          >
            Exercise 3 asks for the additions on 2,048 elements without saying whether that is one segment (20,481) or two
            blocks of 1,024 (18,434), since a block holds at most 1,024 threads.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Chương không hề nói <code>__shfl_up_sync</code> trả gì cho các lane dưới stride; mã đơn giản là không dùng nó. Cả ba
                nhà cung cấp đều ghi nhận rằng các lane đó nhận lại chính giá trị của mình, đó là lý do phép kiểm tra{' '}
                <code>laneIdx() &gt;= stride</code> là bắt buộc (<a href={href('ch11', '11.4')}>11.4</a>; ch11-facts A3).
              </>
            }
          >
            The chapter never says what <code>__shfl_up_sync</code> returns to the lanes below the stride; the code simply
            does not use it. All three vendors document that those lanes get their own value back, which is why the test{' '}
            <code>laneIdx() &gt;= stride</code> is required (<a href={href('ch11', '11.4')}>11.4</a>; ch11-facts A3).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Phần tóm tắt (p. 285) nói chỉ số khối động ngăn deadlock; phần thân (p. 277) chỉ nói các khối trước phải được lập lịch
                trước các khối sau, và không hề gọi tên deadlock mà trình mô phỏng của <a href={href('ch11', '11.9')}>11.9</a> cho
                thấy.
              </>
            }
          >
            The summary (p. 285) says the dynamic block index prevents deadlocks; the body (p. 277) only says that earlier
            blocks must be scheduled before later ones, and never names the deadlock that{' '}
            <a href={href('ch11', '11.9')}>11.9</a>'s simulator shows.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Lần lặp được đếm từ 1 trong phần văn của §11.2 và từ 0 trong Figs. 11.2 và 11.4. Khóa học nói rõ mỗi lần là cách đếm
                nào.
              </>
            }
          >
            Iterations are counted from 1 in the prose of §11.2 and from 0 in Figs. 11.2 and 11.4. The course says which it
            means each time.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Phần dàn trang ở pp. 283–284 chỉ đặt chữ số đầu của chỉ số hai chữ số xuống dưới ("x₁1" thay cho x₁₁). Lỗi chính tả:
                "wap-level scan" (p. 264), "They key difference" (p. 276), "to eliminated" (bài tập 4).
              </>
            }
          >
            Typesetting on pp. 283–284 subscripts only the first digit of two-digit indices ("x₁1" for x₁₁). Typos: "wap-level
            scan" (p. 264), "They key difference" (p. 276), "to eliminated" (exercise 4).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 272 cho thiết bị ví dụ 66.9 TFLOPS; NVIDIA in 67 (ch11-facts A1). Mọi con số suy ra từ đó vẫn đúng.
              </>
            }
          >
            p. 272 gives the example device 66.9 TFLOPS; NVIDIA prints 67 (ch11-facts A1). Every number derived from it
            still holds.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 273 nói ngay cả một kernel sao chép cũng không đạt 100% băng thông. Không tìm thấy tuyên bố nào của nhà cung cấp về
                điều đó (ch11-facts A2); nó là unverified, và lab 11.1 đo phép sao chép của bạn thay vào đó.
              </>
            }
          >
            p. 273 says even a copy kernel does not reach 100% of the bandwidth. No vendor statement of that was found
            (ch11-facts A2); it is unverified, and lab 11.1 measures your copy instead.
          </Translated>
        </li>
      </ul>
      <Aside>
        <Translated
          vi={
            <p>
              Đây là cách chúng tôi đọc một bản PDF, đối chiếu với tài liệu hiện hành của nhà cung cấp ngày 2026-11-20
              (notes/today/ch11-facts.md). Nếu bản của bạn khác, hãy tin bản của bạn.
            </p>
          }
        >
          <p>
            These are our readings of one PDF, checked against current vendor documentation on 2026-11-20
            (notes/today/ch11-facts.md). If your copy differs, trust your copy.
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Một phần của sách được cố ý bỏ qua, để bạn phân biệt được chỗ lược bỏ với chỗ thiếu sót. Bài tập 4 và bài tập 5 được
            giải thích và mô hình hóa, và được đưa ra như các mục "đi xa hơn" chưa kiểm thử trong lab 11.2 và 11.5, không phải
            thành lab. Multiple lookback được mô tả trong phần văn của mục 11.9 nhưng không được mô phỏng. Trình mô phỏng 11.9 mô
            hình hóa rào chắn toàn lưới của phương pháp (a), single lookback và decoupled lookback, với trường hợp xấu nhất tiến
            gần thái cực mọi-khối-cộng-mọi-thứ của Fig. 11.16(c); không lab nào xây dựng rào chắn đó, và cooperative launch mà nó
            cần thuộc chương 18. Bài tập 7 là một snippet đã kiểm tra trên trang 11.10, không phải một lab. Trong các ứng dụng ở p.
            251, chỉ có lọc (chương 12) và radix sort (chương 14) được theo tiếp; lịch sử mạch cộng và nhận xét về năng lượng mỗi
            thứ được một câu. Thrust và CUB được nêu tên, không được dùng, và trong các tài liệu tham khảo chỉ có báo cáo decoupled
            look-back được trích dẫn, vì CUB và rocPRIM trích dẫn nó.
          </p>
        }
      >
        <p>
          Some of the book is left out on purpose, so that you can tell an omission from a gap. Exercise 4 and exercise 5
          are explained and modelled, and offered as untested "go further" items in labs 11.2 and 11.5, not as labs.
          Multiple lookback is described in the prose of section 11.9 but not simulated. The 11.9 simulator models the
          grid-wide barrier of method (a), single lookback and decoupled lookback, whose worst case approaches the
          every-block-adds-everything extreme of Fig. 11.16(c); none of the labs builds the barrier, and the cooperative
          launch it needs is chapter 18's. Exercise 7 is a checked snippet on the 11.10 page, not a lab. Of the applications on p. 251, only
          filter (chapter 12) and radix sort (chapter 14) are followed; the adder-circuit history and the energy remark get a
          sentence each. Thrust and CUB are named, not used, and of the references only the decoupled look-back report is
          cited, because CUB and rocPRIM cite it.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch11-facts.md A8; ch10-facts A12)', '2026-11-20 (notes/today/ch11-facts.md A8 và ch10-facts A12)')}>
        <Translated
          vi={
            <p>
              Các liên kết Thrust và CUB của sách giờ nằm dưới <code>nvidia.github.io/cccl/unstable/</code>.{' '}
              <code>DeviceScan</code> của CUB đã dùng decoupled look-back từ CUB 1.0.1 (2013).
            </p>
          }
        >
          <p>
            The book's Thrust and CUB links now live under <code>nvidia.github.io/cccl/unstable/</code>. CUB's{' '}
            <code>DeviceScan</code> has used decoupled look-back since CUB 1.0.1 (2013).
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Dự án đến lúc này">The project so far</Translated></h2>
      <p>
        <Translated vi="Mặt nạ giữ top-p của lab 11.6 là phần top-p của mốc dự án M5 trong dự án xuyên suốt: khối xác suất tích lũy trước mỗi token của một hàng đã sắp, quyết định giữ cùng quy tắc đồng hạng được nêu rõ, và số token được giữ.">
          Lab 11.6's top-p keep mask is the top-p piece of milestone M5 of the through-line project: the cumulative mass
          before each token of a sorted row, the keep decision with its stated tie rule, and the count of kept tokens.
        </Translated>{' '}
        <span data-m5={m5.join(',') || 'none'}>
          {backends.length === 0
            ? tr('No lab results are recorded on this machine yet.', 'Máy này chưa ghi nhận kết quả lab nào.')
            : m5.length
              ? tr(`Your own lab 11.6 passes on ${m5.map(where).join(' and ')} on this machine.`, `Lab 11.6 của bạn đã qua trên ${m5.map(where).join(' và ')} ở máy này.`)
              : tr('Your own lab 11.6 does not pass on any backend of this machine yet.', 'Lab 11.6 của bạn chưa qua trên backend nào của máy này.')}
        </span>{' '}
        <Translated vi={<>Xem <a href={href('ch11', 'labs')}>các lab</a>.</>}>
          See the <a href={href('ch11', 'labs')}>labs</a>.
        </Translated>
      </p>
      <Translated
        vi={
          <p>
            Chương 12 dùng scan như một công cụ. Một mặt nạ giữ cho biết token nào còn lại; scan loại trừ của mặt nạ đó cho mỗi
            token còn lại một ô trong một danh sách ngắn, dày đặc, đó là cách các token được giữ của M5 được nén gọn (compaction)
            trước khi rút mẫu. Bắt đầu ở <a href={href('ch12')}>chương 12</a>.
          </p>
        }
      >
        <p>
          Chapter 12 uses the scan as a tool. A keep mask says which tokens survive; an exclusive scan of that mask gives
          each surviving token its slot in a short, dense list, which is how the kept tokens of M5 are compacted before a
          sample is drawn. Start at <a href={href('ch12')}>chapter 12</a>.
        </p>
      </Translated>
    </>
  );
}
