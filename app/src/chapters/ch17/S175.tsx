import { useId, useMemo, useRef, useState, type CSSProperties, type ReactNode, type RefObject } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Beyond, Bench, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr as trText } from '../../lib/i18n';
import { href } from '../../lib/router';
import { fmt, useReports, type Reports } from '../../lib/reports';
import { SEGMENT_MODELS, type SegmentModelId } from '../../sim/coalesce';
import { widthChoices } from '../../sim/warps';
import { countByRow, exclusiveScan, stableScatter } from '../../sim/sparseroute';
import {
  FIG1713_LENGTHS,
  PRESETS,
  amortiseIterations,
  colsOf,
  csr,
  ell,
  hybrid,
  laneStates,
  maxOf,
  preset,
  rowLengths,
  sliceChoices,
  sortedLengths,
  spills,
  storage,
  traffic,
  warps,
  type FormatId,
  type Preset,
  type Shape,
  type ShapeParams,
  type Store,
  type Traffic,
  type WarpStats,
  type Widths,
} from '../../sim/sparsescale';
import { RouteScene } from './RouteScene';

// Page 17.5: padding, spilling and sorting, priced on the explorer "Pad, spill or sort"
// (src/sim/sparsescale.ts, through divergence.ts loopBatch and coalesce.ts warpAccess),
// when preprocessing pays, and mixture-of-experts routing as exercise 3 (the scene
// "Routing tokens to experts", src/sim/sparseroute.ts). Every number stated comes from
// those models.

const W4: Widths = { value: 4, index: 4 };
const BOOK = rowLengths(preset('book').params);
const BOOK_CSR = csr(BOOK, W4);
const BOOK_ELL = ell(BOOK, W4);
const BOOK_RATIO = BOOK_ELL.numbers / BOOK_CSR.numbers; // 19.05
const FIG = [...FIG1713_LENGTHS];
const FIG_ELL = ell(FIG, W4);
const FIG_K2 = hybrid(FIG, 2, W4);
const FIG_SPILL = spills(FIG, 2).flatMap((s, r) => (s ? [`${s} from row ${r}`] : [])); // i18n-ignore: the Vietnamese is built where it is shown
const HK = [...Array(1000).fill(16), 400, 400];
const HK_HYB = hybrid(HK, 16, W4);
const HK_ELL = ell(HK, W4);
const AMORT = amortiseIterations(30, 0.6);
const MOE_IDS = [2, 0, 2, 1, 0, 2];
const MOE_COUNTS = countByRow(MOE_IDS, 3);
const MOE_OFFSETS = exclusiveScan(MOE_COUNTS);
const MOE_SCATTER = stableScatter(MOE_IDS, MOE_OFFSETS);
const MOE_E0 = MOE_SCATTER.order.slice(0, MOE_COUNTS[0]);
const list = (xs: readonly number[]) => '[' + xs.join(', ') + ']';
const big = (n: number) => n.toLocaleString('en-US');
const pct = (x: number) => (x * 100 >= 100 ? big(Math.round(x * 100)) : fmt(x * 100)) + '%';

export const questionsS175: Record<string, Question> = {
  'ch17/17.5/q-hybrid-k': {
    kind: 'numeric',
    id: 'ch17/17.5/q-hybrid-k',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Một ma trận có 1,000 hàng mỗi hàng 16 phần tử khác 0 và 2 hàng dài 400. Bạn dựng một dạng lai có phần ELL rộng 16. Bao nhiêu
              mục đi vào phần COO?
            </>
          }
        >
          A matrix has 1,000 rows of 16 non-zeros and 2 rows of 400. You build a hybrid whose ELL part is 16 wide. How
          many entries go to the COO part?
        </Translated>
      </>
    ),
    answer: HK_HYB.coo,
    unit: 'entries',
    vi: { unit: 'mục' },
    explain: (
      <Translated
        vi={
          <p>
            Mỗi hàng dài giữ 16 mục trong ELL và gửi 384 mục sang COO: 2 × 384 = {big(HK_HYB.coo)}, và không còn vị trí đệm nào,
            vì hàng nào cũng có ít nhất 16. ELL thuần sẽ đệm cả 1,002 hàng lên 400 và chứa {big(HK_ELL.pads)} vị trí đệm.
            Giờ kernel ELL chạy 16 lần lặp ở mọi luồng, thay vì hai luồng giữ chân warp của chúng trong 400 lần lặp.
          </p>
        }
      >
        <p>
          Each long row keeps 16 in ELL and sends 384 to COO: 2 × 384 = {big(HK_HYB.coo)}, and no pad slots are left,
          since every row has at least 16. Pure ELL would pad all 1,002 rows to 400 and hold {big(HK_ELL.pads)} pad slots.
          The ELL kernel now runs 16 iterations in every thread, instead of two threads holding their warps for 400.
        </p>
      </Translated>
    ),
    source: 'ch. 17 §17.5, pp. 414–415', // i18n-ignore: citation
  },
  'ch17/17.5/q-sort-y': {
    kind: 'choice',
    id: 'ch17/17.5/q-sort-y',
    prompt: (
      <>
        <Translated
          vi={
            <>
              JDS sắp xếp các hàng, hàng dài nhất trước, trước khi tính tích. Sau đó, vector nào phải được đưa về thứ tự
              ban đầu, và bằng gì?
            </>
          }
        >
          JDS sorts the rows longest first before the product. Afterwards, which vector must be put back into the original
          order, and with what?
        </Translated>
      </>
    ),
    options: [
      {
        label: (
          <>
            <Translated
              vi={
                <>
                  <code>x</code>, bằng mảng row, vì các luồng đọc nó.
                </>
              }
            >
              <code>x</code>, with the row array, since the threads read it.
            </Translated>
          </>
        ),
        why: <Translated vi="Sắp xếp hàng là sắp lại các phương trình, không phải các ẩn số. Mọi chỉ số cột vẫn chỉ đúng phần tử đó của x.">Sorting rows reorders equations, not unknowns. Every column index still names the same element of x.</Translated>,
      },
      {
        label: (
          <>
            <Translated
              vi={
                <>
                  <code>y</code>, bằng mảng row; <code>x</code> không hề di chuyển.
                </>
              }
            >
              <code>y</code>, with the row array; <code>x</code> never moved.
            </Translated>
          </>
        ),
        correct: true,
      },
      {
        label: (
          <>
            <Translated
              vi={
                <>Cả hai, mỗi vector với hoán vị riêng của nó.</>
              }
            >
              Both, each with its own permutation.
            </Translated>
          </>
        ),
        why: <Translated vi="Chỉ các hàng di chuyển. x được đánh chỉ số theo cột, và không cột nào di chuyển.">Only rows moved. x is indexed by column, and no column moved.</Translated>,
      },
      {
        label: <Translated vi="Không vector nào: sắp xếp hàng không thể làm thay đổi một tích.">Neither: sorting rows cannot change a product.</Translated>,
        why: <Translated vi="Các giá trị thì đúng, nhưng đầu ra r của tích đã sắp xếp thuộc về hàng gốc row[r].">The values are right, but output r of the sorted product belongs to original row row[r].</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Luồng <code>r</code> tính tích vô hướng của hàng đã sắp xếp <code>r</code>, tức là hàng gốc{' '}
            <code>row[r]</code>, nên kết quả của nó thuộc về <code>y[row[r]]</code>. Các chỉ số cột không bị đụng tới, nên{' '}
            <code>x</code> được đọc tại chỗ. Phần chữ của sách sơ suất một lần và nói đến việc sắp lại các phần tử{' '}
            <code>x</code> của nghiệm (p. 417); phần ghi chú có ghi lại điều này, và đây là cách đọc giữ cho phép tính đúng.
          </p>
        }
      >
        <p>
          Thread <code>r</code> computes the dot product of sorted row <code>r</code>, which is original row{' '}
          <code>row[r]</code>, so its result belongs in <code>y[row[r]]</code>. The column indices are untouched, so{' '}
          <code>x</code> is read in place. The book's text slips once and speaks of reordering the solution's{' '}
          <code>x</code> elements (p. 417); the notes record it, and this is the reading that keeps the arithmetic right.
        </p>
      </Translated>
    ),
    source: 'ch. 17 §17.6, p. 417 (notes/ch17.md, "Unclear in the PDF" item 5)',
    vi: { source: 'ch. 17 §17.6, p. 417 (notes/ch17.md, "Unclear in the PDF" mục 5)' },
  },
  'ch17/17.5/q-jds-loop': {
    kind: 'choice',
    id: 'ch17/17.5/q-jds-loop',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Bài tập 5 yêu cầu một kernel JDS. Một luồng mỗi hàng đã sắp xếp <code>r</code>, hàng dài nhất trước; các phần tử của lần lặp{' '}
              <code>t</code> bắt đầu ở <code>iterPtr[t]</code>. Phép kiểm tra nào cho phép luồng <code>r</code> nhận một phần tử
              ở lần lặp <code>t</code>?
            </>
          }
        >
          Exercise 5 asks for a JDS kernel. One thread per sorted row <code>r</code>, rows longest first; iteration{' '}
          <code>t</code>'s elements start at <code>iterPtr[t]</code>. Which test lets thread <code>r</code> take an element
          in iteration <code>t</code>?
        </Translated>
      </>
    ),
    options: [
      {
        label: <code>t &lt; iterPtr[r+1] - iterPtr[r]</code>,
        why: <Translated vi="Đó là hình dạng của CSR, một con trỏ mỗi hàng. iterPtr có một mục mỗi lần lặp, không phải mỗi hàng.">That is CSR’s shape, one pointer per row. iterPtr has one entry per iteration, not per row.</Translated>,
      },
      { label: <code>r &lt; numRows</code>, why: <Translated vi="Mọi luồng đều qua được nó, và đọc vượt quá cuối lần lặp t sang lần lặp t + 1.">Every thread passes it, and reads past the end of iteration t into iteration t + 1.</Translated> },
      {
        label: <code>colIdx[iterPtr[t] + r] &gt;= 0</code>,
        why: <Translated vi="JDS không có phần đệm nào để đánh dấu. Quá cuối lần lặp t, địa chỉ đó thuộc về lần lặp t + 1.">JDS has no pads to mark. Past the end of iteration t, that address belongs to iteration t + 1.</Translated>,
      },
      { label: <code>r &lt; iterPtr[t+1] - iterPtr[t]</code>, correct: true },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>iterPtr[t+1] - iterPtr[t]</code> là số hàng vẫn còn phần tử thứ <code>t</code>.
            Vì các hàng được sắp xếp hàng dài nhất trước, đó chính là các hàng đã sắp xếp từ 0 tới số đó trừ một, nên luồng{' '}
            <code>r</code> tham gia khi <code>r</code> còn nhỏ hơn số đó. Phần tử của nó nằm ở <code>iterPtr[t] + r</code>:
            các luồng liên tiếp đọc các địa chỉ liên tiếp, như ở ELL, mà không cần đệm. Sách để kernel này làm
            bài tập; phép kiểm tra này là lời giải của khóa học này, không phải của sách.
          </p>
        }
      >
        <p>
          <code>iterPtr[t+1] - iterPtr[t]</code> is the number of rows that still have a <code>t</code>-th element.
          Because rows are sorted longest first, those are exactly sorted rows 0 up to that number minus one, so thread{' '}
          <code>r</code> takes part while <code>r</code> is below it. Its element is at <code>iterPtr[t] + r</code>:
          consecutive threads read consecutive addresses, as in ELL, with no padding. The book leaves the kernel as an
          exercise; this test is our answer, not the book's.
        </p>
      </Translated>
    ),
    source: 'ch. 17 exercise 5, p. 422; Figs. 17.14, 17.15',
    vi: { source: 'ch. 17 bài tập 5, p. 422; Figs. 17.14, 17.15' },
  },
  'ch17/17.5/q-jds-align': {
    kind: 'choice',
    id: 'ch17/17.5/q-jds-align',
    prompt: (
      <Translated vi="ELL có thể khiến mọi lần lặp bắt đầu ở một ranh giới đoạn với chi phí nhỏ. Vì sao JDS không làm được như vậy?">
        <>ELL can make every iteration start on a segment boundary at little cost. Why can JDS not do the same?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="JDS lưu các phần tử của mỗi hàng liền nhau, từng hàng một, như CSR.">JDS stores each row's elements together, row by row, as CSR does.</Translated>,
        why: <Translated vi="Nó lưu theo thứ tự cột, như ELL (Hình 17.14).">It is column-major, like ELL (Fig. 17.14).</Translated>,
      },
      {
        label: (
          <>
            <Translated
              vi={
                <>
                  Lần lặp <code>t</code> bắt đầu ở <code>iterPtr[t]</code>, do số hàng đạt tới mỗi độ dài quyết định.
                </>
              }
            >
              Iteration <code>t</code> starts at <code>iterPtr[t]</code>, set by how many rows reach each length.
            </Translated>
          </>
        ),
        correct: true,
      },
      { label: <Translated vi="Sắp xếp các hàng phá vỡ việc hợp nhất truy cập bộ nhớ, nên căn lề sẽ chẳng mua được gì.">Sorting the rows breaks coalescing, so alignment would buy nothing.</Translated>, why: <Translated vi="Trong một lần lặp, các luồng liên tiếp vẫn đọc các phần tử liên tiếp.">Within an iteration, consecutive threads still read consecutive elements.</Translated> },
      { label: <Translated vi="JDS cộng vào y bằng atomic, và lưu lượng atomic không được đếm theo các đoạn đã căn lề.">JDS adds into y with atomics, and atomic traffic is not counted in aligned segments.</Translated>, why: <Translated vi="Mỗi hàng đã sắp xếp thuộc về một luồng: không có atomic nào.">One thread owns each sorted row: there are no atomics.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lần lặp <code>t</code> của ELL bắt đầu ở phần tử <code>t*numRows</code>. Làm cho <code>numRows</code> là bội của 8
            và, với giá trị 4 byte và địa chỉ gốc đã căn lề, mọi điểm bắt đầu đều rơi vào ranh giới 32 byte (bội của 16 hoặc 32 cho
            các dòng 64 hoặc 128 byte của AMD). Các điểm bắt đầu của JDS là tổng các số đếm hàng, những con số tùy ý, và chỉ có đệm mới căn lề được
            chúng, mà đó lại chính là điều JDS sinh ra để tránh. Đơn vị ví dụ của sách, "such as 64-bytes" (chẳng hạn 64 byte) cho các GPU cũ không nêu tên, không có trong
            tài liệu NVIDIA nào được đọc hôm nay (ch17-facts A3). Đơn vị của NVIDIA ngày nay là sector 32 byte.
          </p>
        }
      >
        <p>
          ELL's iteration <code>t</code> starts at element <code>t*numRows</code>. Make <code>numRows</code> a multiple of 8
          and, with 4-byte values and an aligned base, every start falls on a 32-byte boundary (a multiple of 16 or 32 for
          AMD's 64- or 128-byte lines). JDS's starts are sums of row counts, arbitrary numbers, and only padding could align
          them, which is what JDS exists to avoid. The book's example unit, "such as 64-bytes" for unnamed older GPUs, is in
          no NVIDIA document read today (ch17-facts A3). NVIDIA's unit today is the 32-byte sector.
        </p>
      </Translated>
    ),
    source: 'ch. 17 pp. 413, 417; ch17-facts A3, E1', // i18n-ignore: citation
  },
  'ch17/17.5/q-amortise': {
    kind: 'numeric',
    id: 'ch17/17.5/q-amortise',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Chuyển một ma trận CSR sang một định dạng đã sắp xếp, đã đệm tốn thời gian bằng 30 lần SpMV với CSR. Sau đó mỗi SpMV ở định dạng mới
              tốn 0.6 thời gian của CSR. Sau bao nhiêu lần lặp của bộ giải thì việc chuyển đổi tự bù được chi phí của nó?
            </>
          }
        >
          Converting a CSR matrix to a sorted, padded format takes as long as 30 CSR SpMVs. Each SpMV in the new format then
          takes 0.6 of the CSR time. After how many solver iterations has the conversion paid for itself?
        </Translated>
      </>
    ),
    answer: Math.round(AMORT),
    tol: 0.01,
    unit: 'iterations',
    vi: { unit: 'lần lặp', source: 'ch. 17 §17.5 và §17.6, pp. 414–415, 417' },
    explain: (
      <Translated
        vi={
          <p>
            Mỗi lần lặp tiết kiệm 0.4 của một SpMV với CSR, nên cần 30 / 0.4 = {fmt(AMORT)} lần lặp để thu lại 30 lần đó. Một
            bộ giải hội tụ sau 50 lần lặp thì không nên chuyển đổi; một bộ giải chạy 500 lần thì nên. Các thư viện của nhà cung cấp đưa ra
            cùng lời khuyên về phần tiền xử lý của riêng chúng (bên dưới).
          </p>
        }
      >
        <p>
          Each iteration saves 0.4 of a CSR SpMV, so it takes 30 / 0.4 = {fmt(AMORT)} iterations to win back the 30. A
          solver that converges in 50 iterations should not convert; one that runs 500 should. The vendor libraries give
          the same advice about their own preprocessing (below).
        </p>
      </Translated>
    ),
    source: 'ch. 17 §17.5 and §17.6, pp. 414–415, 417',
  },
  'ch17/17.5/q-moe-offset': {
    kind: 'numeric',
    id: 'ch17/17.5/q-moe-offset',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Sáu vị trí token được định tuyến tới các expert {list(MOE_IDS)}, với ba expert. Sau khi đếm, scan loại trừ và
              rải ổn định, vị trí token 4 rơi vào vị trí nào trong thứ tự đã sắp theo expert?
            </>
          }
        >
          Six token slots are routed to experts {list(MOE_IDS)}, with three experts. After counting, an exclusive scan and a
          stable scatter, at which position of the expert-sorted order does slot 4 land?
        </Translated>
      </>
    ),
    answer: MOE_SCATTER.pos[4],
    unit: '',
    explain: (
      <Translated
        vi={
          <p>
            Số đếm {list(MOE_COUNTS)}, độ lệch {list(MOE_OFFSETS)}. Expert 0 giữ các vị trí {MOE_E0.join(' và ')}, theo đúng thứ tự
            đó, nên vị trí 4 rơi vào {MOE_OFFSETS[0]} + {MOE_SCATTER.rank[4]} = {MOE_SCATTER.pos[4]}. Phép scan cho điểm bắt đầu của expert
            0; rank, tức số vị trí trước đó được định tuyến tới cùng expert, cho phần còn lại.
          </p>
        }
      >
        <p>
          Counts {list(MOE_COUNTS)}, offsets {list(MOE_OFFSETS)}. Expert 0 holds slots {MOE_E0.join(' and ')}, in that
          order, so slot 4 lands at {MOE_OFFSETS[0]} + {MOE_SCATTER.rank[4]} = {MOE_SCATTER.pos[4]}. The scan gave expert
          0's start; the rank, the number of earlier slots routed to the same expert, gave the rest.
        </p>
      </Translated>
    ),
    source: 'ours, after ch. 17 exercise 3, p. 422',
    vi: { source: 'của khóa học này, dựa theo ch. 17 bài tập 3, p. 422' },
  },
  'ch17/17.5/q-moe-atomic': {
    kind: 'choice',
    id: 'ch17/17.5/q-moe-atomic',
    prompt: (
      <>
        <Translated
          vi={
            <>
              <code>mm_ids_helper</code> của llama.cpp xếp hạng mỗi mục trong lát của expert của nó bằng một ballot và một popcount
              trên các lane, chứ không bằng một con trỏ tăng nguyên tử cho mỗi expert. Điều đó mua được gì?
            </>
          }
        >
          llama.cpp's <code>mm_ids_helper</code> ranks each entry inside its expert's slice with a ballot and a popcount
          over the lanes, not with an atomically incremented cursor per expert. What does that buy?
        </Translated>
      </>
    ),
    options: [
      { label: <Translated vi="Ít byte phải di chuyển hơn cho mỗi token, vì không có mảng con trỏ nào phải đọc.">Fewer bytes moved per token, since no cursor array is read.</Translated>, why: <Translated vi="Cả hai đều ghi cùng các id vào cùng các lát, và khối của mỗi expert đọc mọi id (ch17-facts D5), nên phiên bản ballot đọc tổng cộng nhiều byte hơn, không phải ít hơn.">Both write the same ids into the same slices, and every expert’s block reads all the ids (ch17-facts D5), so the ballot version reads more bytes in total, not fewer.</Translated> },
      { label: <Translated vi="Không cần đếm các expert khác: popcount cho mỗi lát điểm bắt đầu của nó.">No need to count the other experts: the popcount gives each slice its start.</Translated>, why: <Translated vi="Popcount chỉ đếm các lane khớp với expert này, nên nó đặt một mục vào trong lát của nó. Lát bắt đầu ở đâu là một phép đếm riêng: hàm trợ giúp cộng dồn số mục của các expert có số thứ tự thấp hơn trong cùng lượt quét (nex_prev, D5), và một con trỏ nguyên tử cũng có thể nằm trong lượt quét đó y như vậy.">The popcount counts only this expert’s matching lanes, so it places an entry inside its slice. Where a slice starts is a separate count: the helper tallies the lower-numbered experts’ entries in the same sweep (nex_prev, D5), and an atomic cursor could sit in that sweep just as well.</Translated> },
      { label: <Translated vi="Các token giữ thứ tự đầu vào của chúng trong mọi lát, ở mọi lần chạy.">Tokens keep their input order inside every slice, on every run.</Translated>, correct: true },
      { label: <Translated vi="Một lượt quét qua các id: một con trỏ nguyên tử sẽ cần lượt thứ hai để xếp hạng.">One sweep over the ids: an atomic cursor would need a second pass to rank.</Translated>, why: <Translated vi="Một con trỏ xếp hạng ngay trong cùng lượt quét: mỗi luồng lấy vị trí trống kế tiếp của vùng ghi khi gặp mục của nó, trong lúc nex_prev đang được đếm. Phương pháp xếp hạng chỉ thay đổi thứ tự bên trong một lát.">A cursor ranks in the same sweep: each thread takes the next free slot of the store when it meets its entry, while nex_prev is being counted. The ranking method changes only the order inside a slice.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một con trỏ nguyên tử phát hạng theo thứ tự các atomic của luồng được thực hiện, vốn không xác định, nên thứ tự bên trong
            một lát có thể thay đổi giữa các lần chạy. Một ballot trên các lane bên dưới, tiếp theo là một popcount, xếp hạng mỗi mục theo
            vị trí của nó trong đầu vào, như phép rải ổn định của mô hình: các token giữ thứ tự đầu vào trong lát của mỗi expert.
            Cái giá là một khối phải đi qua mọi id cho expert của nó, từng đoạn một.
          </p>
        }
      >
        <p>
          An atomic cursor hands out ranks in the order the threads' atomics land, which is undefined, so the order inside
          a slice can change from run to run. A ballot over the lanes below, followed by a popcount, ranks each entry by its
          place in the input, as the model's stable scatter does: tokens stay in input order inside each expert's slice.
          The price is that one block works through all the ids for its expert, chunk by chunk.
        </p>
      </Translated>
    ),
    source: 'ch17-facts D5; ballot and popcount: ch. 12; atomics: ch. 9',
    vi: { source: 'ch17-facts D5; ballot và popcount: ch. 12; atomic: ch. 9' },
  },
};

// ---------------------------------------------------------------- explorer state
interface XState {
  preset: Preset['id'] | null;
  params: ShapeParams;
  K: number;
  sorted: boolean;
  S: number;
  widthKey: string;
  value: number;
  index: number;
  model: SegmentModelId;
  /** Iteration shown in the lane grid; -1 until the reader moves the slider. */
  t: number;
}
function fromPreset(id: Preset['id'], prev?: XState, K?: number): XState {
  const params = { ...preset(id).params };
  const len = rowLengths(params);
  return {
    preset: id,
    params,
    K: K ?? maxOf(len),
    sorted: false,
    S: len.length,
    widthKey: prev?.widthKey ?? 'book',
    value: prev?.value ?? 4,
    index: prev?.index ?? 4,
    model: prev?.model ?? 'nvidia',
    t: -1,
  };
}
const SHAPES: { id: Shape; label: string }[] = [
  {
    id: 'uniform',
    get label() {
      return trText('uniform', 'đều');
    },
  },
  {
    id: 'spread',
    get label() {
      return trText('spread from 1 to 2 × mean − 1', 'trải từ 1 đến 2 × trung bình − 1');
    },
  },
  {
    id: 'long',
    get label() {
      return trText('a few long rows, total kept', 'vài hàng dài, giữ nguyên tổng');
    },
  },
  {
    id: 'power',
    get label() {
      return trText('power law (ours)', 'luật lũy thừa (của khóa học này)');
    },
  },
  {
    id: 'fig1713',
    get label() {
      return trText('Fig. 17.13', 'Hình 17.13');
    },
  },
];
const ROW_CHOICES = [256, 1000, 1024, 4096];
const VALUE_BYTES = [4, 2, 1, 0.5];
const INDEX_BYTES = [4, 2];
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const NARROW_OK: CSSProperties = { maxWidth: '100%', minWidth: 0 };
const Wide = ({ children, style }: { children: ReactNode; style?: CSSProperties }) => (
  <div style={{ overflowX: 'auto', maxWidth: '100%', ...style }}>{children}</div>
);
const byteLabel = (b: number) => (b < 1 ? trText('4 bits', '4 bit') : `${b} B`);

const FORMAT_ORDER: FormatId[] = ['dense', 'csr', 'ell', 'hybrid', 'jds', 'sell'];
function formatLabel(s: Store, x: XState): string {
  switch (s.id) {
    case 'dense':
      return trText('dense', 'dày đặc');
    case 'csr':
      return 'CSR';
    case 'ell':
      return trText(`ELL, ${s.width} wide`, `ELL, rộng ${s.width}`);
    case 'hybrid':
      return trText(`hybrid: ELL ${s.width} wide + ${big(s.coo)} COO`, `lai: ELL rộng ${s.width} + ${big(s.coo)} COO`);
    case 'jds':
      return 'JDS';
    case 'sell':
      return trText(`sliced ELL, slices of ${x.S}${x.sorted ? ', sorted' : ''}`, `sliced ELL, lát cỡ ${x.S}${x.sorted ? ', đã sắp xếp' : ''}`);
  }
}

// ---------------------------------------------------------------- explorer panels
function StoragePanel({ table, x, w }: { table: ReturnType<typeof storage>; x: XState; w: Widths }) {
  const dense = table.stores.dense;
  const maxBytes = Math.max(...FORMAT_ORDER.map((f) => table.stores[f].bytes));
  return (
    <div className="panel" data-panel="storage">
      <div className="mx-label">
        <Translated vi="Lưu trữ: giá trị và chỉ số theo cách sách đếm, và số byte ở các độ rộng đã chọn">Storage: values and indices as the book counts them, and bytes at the widths chosen</Translated>
      </div>
      <Wide>
        <table className="threads" aria-label={trText('Storage of each format', 'Lưu trữ của mỗi định dạng')}>
          <thead>
            <tr>
              <th><Translated vi="Định dạng">Format</Translated></th>
              <th><Translated vi="số">numbers</Translated></th>
              <th><Translated vi="byte">bytes</Translated></th>
              <th><Translated vi="so với dày đặc">of dense</Translated></th>
              <th><Translated vi="vị trí đệm">pad slots</Translated></th>
              <th style={{ minWidth: '9rem' }}></th>
            </tr>
          </thead>
          <tbody>
            {FORMAT_ORDER.map((f) => {
              const s = table.stores[f];
              const padBytes = s.pads * (w.value + w.index);
              return (
                <tr key={f} data-format={f}>
                  <td>{formatLabel(s, x)}</td>
                  <td data-out="numbers">{big(s.numbers)}</td>
                  <td data-out="bytes">{big(Math.round(s.bytes))}</td>
                  <td data-out="pct">{pct(s.bytes / dense.bytes)}</td>
                  <td data-out="pads">{f === 'dense' ? '' : big(s.pads)}</td>
                  <td>
                    <div style={{ display: 'flex', height: 10, width: '100%', minWidth: '9rem' }} aria-hidden="true">
                      <div style={{ width: `${(100 * (s.bytes - padBytes)) / maxBytes}%`, background: 'var(--global)' }} />
                      <div style={{ width: `${(100 * padBytes) / maxBytes}%`, background: 'var(--global-wash)', borderRight: padBytes ? '1px dashed var(--global)' : undefined }} />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Wide>
      <p className="muted" style={{ margin: '0.5rem 0 0' }}>
        <Translated
          vi={
            <>
              Đậm: dữ liệu, chỉ số và con trỏ. Nhạt: phần đệm. Các mảng con trỏ (rowPtrs, iterPtr, mảng row, độ lệch của lát)
              chiếm 4 byte mỗi phần tử.
            </>
          }
        >
          Solid: data, indices and pointers. Pale: padding. Pointer arrays (rowPtrs, iterPtr, the row array, slice offsets)
          take 4 bytes each.
        </Translated>
      </p>
    </div>
  );
}

const hatchStyle = (id: string) => `url(#${id})`;
function LaneGrid({ len, width, t }: { len: number[]; width: number; t: number }) {
  const hatch = useId();
  const sorted = useMemo(() => sortedLengths(len), [len]);
  const c = width <= 32 ? 12 : 7;
  const LEFT = 150;
  const rh = 18;
  const rows = [
    { key: 'orig-0', label: trText('row order, warp 0', 'thứ tự hàng, warp 0'), trips: len, warp: 0 },
    { key: 'orig-1', label: trText('row order, warp 1', 'thứ tự hàng, warp 1'), trips: len, warp: 1 },
    { key: 'sort-0', label: trText('sorted, warp 0', 'đã sắp xếp, warp 0'), trips: sorted, warp: 0 },
    { key: 'sort-1', label: trText('sorted, warp 1', 'đã sắp xếp, warp 1'), trips: sorted, warp: 1 },
  ];
  const Wd = LEFT + width * c + 110;
  const Ht = 22 + rows.length * rh + 8;
  return (
    <svg viewBox={`0 0 ${Wd} ${Ht}`} role="img" aria-label={trText(
        `Which lanes of the first two warps still work in iteration ${t}, in row order and sorted`,
        `Những lane nào của hai warp đầu tiên vẫn làm việc ở lần lặp ${t}, theo thứ tự hàng và khi đã sắp xếp`,
      )}
      style={{ width: Wd, display: 'block' }} data-view="lanes">
      <defs>
        <pattern id={hatch} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.2" />
        </pattern>
      </defs>
      <text x={LEFT - 8} y={13} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi="lane">lane</Translated></text>
      {[0, width / 2, width - 1].map((l) => (
        <text key={l} x={LEFT + l * c + c / 2} y={13} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{l}</text>
      ))}
      {rows.map((r, i) => {
        const st = laneStates(r.trips, width, r.warp, t);
        const working = st.filter((s) => s === 'work').length;
        const real = st.filter((s) => s !== 'absent').length;
        const y = 22 + i * rh + (i >= 2 ? 6 : 0);
        return (
          <g key={r.key} data-grid-row={r.key} data-work={working}>
            <text x={LEFT - 8} y={y + 11} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{r.label}</text>
            {st.map((s, l) => (
              <g key={l}>
                <rect x={LEFT + l * c} y={y} width={c - 1} height={rh - 5} rx="1.5"
                  fill={s === 'work' ? 'var(--ink)' : s === 'done' ? 'var(--cell)' : 'none'}
                  stroke={s === 'work' ? 'none' : 'var(--rule)'} strokeWidth="0.6" strokeDasharray={s === 'absent' ? '2 2' : undefined} />
                {s === 'done' ? <rect x={LEFT + l * c} y={y} width={c - 1} height={rh - 5} rx="1.5" fill={hatchStyle(hatch)} pointerEvents="none" /* i18n-ignore: SVG keyword */ /> : null}
              </g>
            ))}
            <text x={LEFT + width * c + 8} y={y + 11} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
              {real ? trText(`${working} of ${real} work`, `${working} trên ${real} làm việc`) : trText('no rows', 'không có hàng')}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function WarpPanel({ wt, len, width, x, setT }: { wt: ReturnType<typeof warps>; len: number[]; width: number; x: XState; setT: (t: number) => void }) {
  const firstTwo = (trips: number[]) => maxOf(trips.slice(0, 2 * width));
  const tMax = Math.max(1, firstTwo(len), firstTwo(sortedLengths(len)));
  // until the reader moves the slider, show the iteration at the mean row length: where lanes start to drop out
  const auto = Math.round(len.reduce((a, b) => a + b, 0) / Math.max(1, len.length));
  const t = Math.max(0, Math.min(x.t < 0 ? auto : x.t, tMax - 1));
  const row = (id: string, name: ReactNode, s: WarpStats) => (
    <tr data-warps={id}>
      <td>{name}</td>
      <td data-out="spent">{big(s.spent)}</td>
      <td data-out="useful">{big(s.useful)}</td>
      <td data-out="efficiency">{pct(s.efficiency)}</td>
      <td data-out="longest">{big(s.longest)}</td>
    </tr>
  );
  return (
    <div className="panel" data-panel="warps">
      <div className="mx-label">
        <Translated vi="Warp: một luồng mỗi hàng, một lần lặp mỗi phần tử khác 0">Warps: one thread per row, one iteration per non-zero</Translated>
      </div>
      <Wide>
        <table className="threads" aria-label={trText('What each order costs a per-row kernel', 'Mỗi thứ tự tốn của một kernel mỗi-hàng-một-luồng bao nhiêu')}>
          <thead>
            <tr>
              <th><Translated vi="Kernel">Kernel</Translated></th>
              <th><Translated vi="lượt warp-lặp">warp-iterations</Translated></th>
              <th><Translated vi="lượt lane-lặp làm việc">lane-iterations of work</Translated></th>
              <th><Translated vi="phần thời gian lane làm việc">lane time that is work</Translated></th>
              <th><Translated vi="số vòng dài nhất">longest trip</Translated></th>
            </tr>
          </thead>
          <tbody>
            {row('original', trText('CSR or ELL, row order', 'CSR hoặc ELL, thứ tự hàng'), wt.original)}
            {row('sorted', trText('JDS, longest first', 'JDS, hàng dài nhất trước'), wt.sorted)}
            {row('hybrid', trText(`hybrid's ELL part, ${x.K} wide`, `phần ELL của dạng lai, rộng ${x.K}`), wt.hybrid)}
          </tbody>
        </table>
      </Wide>
      <label className="field" style={{ marginTop: '0.7rem', width: '100%' }}>
        <span>
          <Translated vi={<>Lần lặp <b data-out="t">{t}</b> của hai warp đầu tiên</>}>
            Iteration <b data-out="t">{t}</b> of the first two warps
          </Translated>
        </span>
        <input type="range" min={0} max={tMax - 1} value={t} aria-label={trText('Iteration shown in the lane grid', 'Lần lặp hiển thị trong lưới lane')} onChange={(e) => setT(Number(e.target.value))} />
      </label>
      <Wide style={{ marginTop: '0.4rem' }}>
        <LaneGrid len={len} width={width} t={t} />
      </Wide>
      <p className="muted" style={{ margin: '0.4rem 0 0' }}>
        <Translated
          vi={
            <>
              Tô kín: lane làm việc ở lần lặp này. Gạch chéo: hàng của nó đã xong và nó chờ hàng dài nhất của warp. Nét đứt: không có hàng.
              Phần COO của dạng lai chạy một luồng mỗi mục và không có vòng lặp.
            </>
          }
        >
          Filled: the lane works this iteration. Hatched: its row is done and it waits for the warp's longest. Dashed: no row.
          The hybrid's COO part runs one thread per entry and has no loop.
        </Translated>
      </p>
    </div>
  );
}

function TrafficPanel({ tr, model }: { tr: Traffic[] | null; model: SegmentModelId }) {
  const m = SEGMENT_MODELS[model];
  if (!tr || m.segmentBytes == null)
    return (
      <div className="panel" data-panel="traffic" data-hidden="1">
        <div className="mx-label"><Translated vi="Các đoạn được yêu cầu">Requested segments</Translated></div>
        <p style={{ margin: 0 }}>
          <Translated
            vi={
              <>
                Apple không công bố kích thước đoạn, kích thước dòng hay quy tắc hợp nhất nào ({m.facts}), nên không có gì để đếm theo.
                Chọn đơn vị của NVIDIA hoặc AMD để xem số đếm; trên Metal, lab sẽ đo thay vào đó.
              </>
            }
          >
            Apple documents no segment size, line size or coalescing rule ({m.facts}), so there is nothing to count against.
            Choose NVIDIA's or AMD's unit to see the counts; on Metal, the lab measures instead.
          </Translated>
        </p>
      </div>
    );
  return (
    <div className="panel" data-panel="traffic">
      <div className="mx-label">
        <Translated vi={<>Các đoạn được yêu cầu của mảng giá trị và mảng chỉ số, {({ nvidia: 'NVIDIA: sector 32 byte trong dòng 128 byte', amd64: 'AMD: dòng cache 64 byte', amd128: 'AMD: dòng cache 128 byte', metal: 'Apple: không có tài liệu' } as Record<string, string>)[m.id] ?? m.label}</>}>
          Requested segments of the value and index arrays, {m.label}
        </Translated>
      </div>
      <Wide>
        <table className="threads" aria-label={trText("Segments each format's kernel requests", 'Các đoạn mà kernel của mỗi định dạng yêu cầu')}>
          <thead>
            <tr>
              <th><Translated vi="Kernel">Kernel</Translated></th>
              <th>
                <Translated vi={<>số {({ sector: 'sector', line: 'dòng', segment: 'đoạn' } as Record<string, string>)[m.unit] ?? m.unit} được yêu cầu, không có cache</>}>{m.unit}s requested, no cache</Translated>
              </th>
              <th>
                <Translated vi={<>số {({ sector: 'sector', line: 'dòng', segment: 'đoạn' } as Record<string, string>)[m.unit] ?? m.unit} khác nhau, cache hoàn hảo</>}>distinct {m.unit}s, perfect cache</Translated>
              </th>
              <th><Translated vi="byte được dùng">bytes used</Translated></th>
              <th><Translated vi="được dùng / được yêu cầu">used / requested</Translated></th>
            </tr>
          </thead>
          <tbody>
            {tr.map((r) => (
              <tr key={r.format} data-traffic={r.format}>
                <td>{r.format === 'csr' ? 'CSR' : r.format === 'ell' ? 'ELL' : 'JDS'}</td>
                <td data-out="requested">{big(r.requested)}</td>
                <td data-out="distinct">{big(r.distinct)}</td>
                <td data-out="needed">{big(Math.round(r.needed))}</td>
                <td data-out="efficiency">{pct(r.efficiency)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Wide>
      <p className="muted" style={{ margin: '0.5rem 0 0' }}>
        <Translated
          vi={
            <>
              Cột thứ nhất đếm mọi lần nạp của mọi warp như thể không có gì được cache; cột thứ hai đếm mỗi {({ sector: 'sector', line: 'dòng', segment: 'đoạn' } as Record<string, string>)[m.unit] ?? m.unit} một lần. Một
              thiết bị thật di chuyển một lượng nằm giữa hai số đó. Kernel của ELL dừng ở độ dài của mỗi hàng (Hình 17.12), nên nó không yêu cầu
              phần đệm nào.
            </>
          }
        >
          The first column counts every warp's every load as if nothing were cached; the second counts each {m.unit} once. A
          real device moves something between the two. ELL's kernel stops at each row's length (Fig. 17.12), so it requests
          no padding.
        </Translated>
      </p>
    </div>
  );
}

function Explorer({ x, setX, anchor }: { x: XState; setX: (f: (s: XState) => XState) => void; anchor: RefObject<HTMLDivElement | null> }) {
  const { reports } = useReports();
  const choices = useMemo(() => widthChoices(reports as Reports), [reports]);
  const chosen = choices.find((c) => c.key === x.widthKey) ?? choices[0];
  const width = chosen.width;
  const len = useMemo(() => rowLengths(x.params), [x.params]);
  const cols = colsOf(x.params);
  const w: Widths = useMemo(() => ({ value: x.value, index: x.index }), [x.value, x.index]);
  const maxLen = maxOf(len);
  const K = Math.min(x.K, maxLen);
  const S = sliceChoices(len.length).includes(x.S) ? x.S : len.length;
  const table = useMemo(() => storage(len, cols, w, K, S, x.sorted), [len, cols, w, K, S, x.sorted]);
  const wt = useMemo(() => warps(len, width, K), [len, width, K]);
  const seg = SEGMENT_MODELS[x.model].segmentBytes;
  const tr = useMemo(
    () => (seg == null ? null : (['csr', 'ell', 'jds'] as const).map((f) => traffic(f, len, width, seg, w))),
    [len, width, seg, w],
  );
  const nnz = table.nnz;
  const mean = nnz / len.length;
  const ratio = table.stores.ell.numbers / table.stores.csr.numbers;
  const hyb = table.stores.hybrid;
  const fixed = x.params.shape === 'fig1713';
  const isLong = x.params.shape === 'long';

  const edit = (p: Partial<ShapeParams>) =>
    setX((s) => {
      const params = { ...s.params, ...p };
      if (params.shape === 'fig1713') params.rows = 8;
      else if (s.params.shape === 'fig1713') params.rows = 1024;
      const before = rowLengths(s.params);
      const after = rowLengths(params);
      // a K at the old longest row means "pure ELL": keep meaning that
      const K = s.K >= maxOf(before) ? maxOf(after) : Math.min(s.K, maxOf(after));
      const S = after.length !== before.length ? after.length : s.S;
      return { ...s, preset: null, params, K, S, t: -1 };
    });

  return (
    <div ref={anchor} className="wide">
      <Bench
        title={trText('Pad, spill or sort', 'Đệm, tràn hay sắp xếp')}
        task={trText("Price ELL's padding, then remove it three ways.", 'Định giá phần đệm của ELL, rồi loại bỏ nó theo ba cách.')}
        foot={
          <Translated
            vi={
              <>
                Chạy trên src/sim/sparsescale.ts: độ dài hàng lấy từ một bộ sinh có hạt giống cố định, lưu trữ được đếm như sách đếm,
                warp qua mô hình vòng lặp của chương 4 và đoạn qua mô hình của chương 6.{' '}
                {chosen.key === 'book'
                  ? 'Độ rộng mà sách nêu cho các thiết bị CUDA cho đến nay (pp. 74, 77), với giới hạn 1024 luồng trong một khối của thiết bị ví dụ của sách (p. 85).'
                  : chosen.key === 'w64'
                    ? 'Độ rộng mà tài liệu HIP của AMD đưa ra cho các thiết bị gfx9 (AMD Instinct). Giới hạn 1024 luồng trong một khối là ví dụ của sách, không phải con số của AMD.'
                    : chosen.note}
              </>
            }
          >
            <>
              Runs on src/sim/sparsescale.ts: row lengths from a seeded generator, storage counted as the book counts it,
              warps through chapter 4's loop model and segments through chapter 6's. {chosen.note}
            </>
          </Translated>
        }
      >
        <div
          data-explorer="pad-spill-sort"
          data-preset={x.preset ?? ''}
          data-rows={len.length}
          data-cols={cols}
          data-nnz={nnz}
          data-max={maxLen}
          data-k={K}
          data-s={S}
          data-sorted={x.sorted ? 1 : 0}
          data-width={width}
          data-value={x.value}
          data-index={x.index}
          data-model={x.model}
        >
          <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
            <div style={FIELD}>
              <Translated vi="Thiết lập sẵn">Preset</Translated>
              <span className="seg" role="group" aria-label={trText('Presets', 'Thiết lập sẵn')} style={{ flexWrap: 'wrap', rowGap: 4 }}>
                {PRESETS.map((p) => (
                  <button key={p.id} aria-pressed={x.preset === p.id} onClick={() => setX((s) => fromPreset(p.id, s))} title={trText(p.note, ({ book: 'Ví dụ tính toán của sách (pp. 412–413): 1% khác 0, một hàng dài 200. Các hàng khác dài 9 hoặc 10, để tổng vẫn là 1%; cụm "fewer than 10" (ít hơn 10) của sách không thể khớp với 1%.', fig1713: 'Tám hàng của Hình 17.13 (p. 415): độ dài 1, 5, 2, 2, 1, 2, 4, 1, 18 phần tử khác 0.', uniform: 'Mọi hàng có cùng độ dài: trường hợp mà ELL được tạo ra để xử lý.', spread: 'Độ dài trải đều từ 1 đến 31, theo thứ tự xáo trộn.', skewed: 'Tám hàng dài 256 trong 4096 hàng; các hàng còn lại chia phần còn lại để trung bình là 16 mỗi hàng.', power: 'Của khóa học này, không phải của sách: dạng Pareto với số mũ 1.5, vài hàng rất dài và nhiều hàng ngắn.' } as Record<string, string>)[p.id] ?? p.note)}>
                    {trText(p.label, ({ book: 'Ví dụ 1000 × 1000 của sách', fig1713: 'Hình 17.13', uniform: 'Đều, 16 mỗi hàng', spread: 'Trải, 1 đến 31', skewed: 'Lệch: cứ 512 hàng có 1 hàng dài gấp 16 lần', power: 'Luật lũy thừa (của khóa học này)' } as Record<string, string>)[p.id] ?? p.label)}
                  </button>
                ))}
              </span>
            </div>
          </div>
          <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
            <label className="field" style={NARROW_OK}>
              <Translated vi="Độ dài hàng">Row lengths</Translated>
              <select value={x.params.shape} aria-label={trText('Shape of the row lengths', 'Hình dạng phân bố độ dài hàng')} onChange={(e) => edit({ shape: e.target.value as Shape })}>
                {SHAPES.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
              </select>
            </label>
            <label className="field">
              <Translated vi="Số hàng (ma trận vuông)">Rows (square matrix)</Translated>
              <select value={len.length} disabled={fixed} aria-label={trText('Rows', 'Số hàng')} onChange={(e) => edit({ rows: Number(e.target.value) })}>
                {(fixed ? [8] : ROW_CHOICES).map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
            </label>
            <label className="field">
              <span>
                <Translated vi={<>Trung bình mỗi hàng: <b data-out="mean-param">{x.params.mean}</b></>}>
                  Mean per row: <b data-out="mean-param">{x.params.mean}</b>
                </Translated>
              </span>
              <input type="range" min={1} max={64} value={x.params.mean} disabled={fixed} aria-label={trText('Mean non-zeros per row', 'Số phần tử khác 0 trung bình mỗi hàng')} onChange={(e) => edit({ mean: Number(e.target.value) })} />
            </label>
            <label className="field">
              <Translated vi="Hàng dài">Long rows</Translated>
              <input type="number" min={0} max={64} value={x.params.longCount} disabled={!isLong} aria-label={trText('Number of long rows', 'Số hàng dài')}
                onChange={(e) => edit({ longCount: Math.max(0, Math.min(64, Math.floor(Number(e.target.value) || 0))) })} style={{ width: '5rem' }} />
            </label>
            <label className="field">
              <span>
                <Translated vi={<>Độ dài hàng dài: <b>{x.params.longLen}</b></>}>
                  Long-row length: <b>{x.params.longLen}</b>
                </Translated>
              </span>
              <input type="range" min={1} max={cols} value={Math.min(cols, Math.max(1, x.params.longLen))} disabled={!isLong} aria-label={trText('Length of the long rows', 'Độ dài của các hàng dài')}
                onChange={(e) => edit({ longLen: Number(e.target.value) })} />
            </label>
          </div>
          <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
            <label className="field">
              <span>
                <Translated vi={<>Độ rộng ELL của dạng lai K: <b data-out="k">{K}</b></>}>
                  Hybrid ELL width K: <b data-out="k">{K}</b>
                </Translated>
              </span>
              <input type="range" min={0} max={maxLen} value={K} aria-label={trText('Hybrid ELL width K', 'Độ rộng ELL của dạng lai K')} onChange={(e) => setX((s) => ({ ...s, K: Number(e.target.value) }))} />
            </label>
            <label className="field">
              <Translated vi="Kích thước lát S">Slice size S</Translated>
              <select value={S} aria-label={trText('Slice size', 'Kích thước lát')} onChange={(e) => setX((s) => ({ ...s, S: Number(e.target.value) }))}>
                {sliceChoices(len.length).map((v) => <option key={v} value={v}>{v === len.length ? trText(`${v} (plain ELL)`, `${v} (ELL thường)`) : v}</option>)}
              </select>
            </label>
            <label className="check">
              <input type="checkbox" checked={x.sorted} aria-label={trText('Sort rows longest first', 'Sắp xếp hàng, hàng dài nhất trước')} onChange={(e) => setX((s) => ({ ...s, sorted: e.target.checked }))} />{' '}
              <Translated vi="sắp xếp hàng, hàng dài nhất trước">sort rows longest first</Translated>
            </label>
          </div>
          <div className="row" style={{ marginBottom: '1rem', alignItems: 'flex-end' }}>
            <label className="field" style={NARROW_OK}>
              <Translated vi="Độ rộng warp">Warp width</Translated>
              <select value={chosen.key} aria-label={trText('Warp width', 'Độ rộng warp')} onChange={(e) => setX((s) => ({ ...s, widthKey: e.target.value, t: -1 }))}>
                {choices.map((c) => <option key={c.key} value={c.key}>{trText(c.label, c.key === 'book' ? 'con số 32 của sách' : c.key === 'w64' ? '64, như HIP ghi trong tài liệu cho gfx9' : c.label.replace(/^yours: /, 'của bạn: '))}</option>)}
              </select>
            </label>
            <div style={FIELD}>
              <Translated vi="Số byte mỗi giá trị">Bytes per value</Translated>
              <span className="seg" role="group" aria-label={trText('Bytes per value', 'Số byte mỗi giá trị')}>
                {VALUE_BYTES.map((b) => (
                  <button key={b} aria-pressed={x.value === b} onClick={() => setX((s) => ({ ...s, value: b }))}>{byteLabel(b)}</button>
                ))}
              </span>
            </div>
            <div style={FIELD}>
              <Translated vi="Số byte mỗi chỉ số">Bytes per index</Translated>
              <span className="seg" role="group" aria-label={trText('Bytes per index', 'Số byte mỗi chỉ số')}>
                {INDEX_BYTES.map((b) => (
                  <button key={b} aria-pressed={x.index === b} onClick={() => setX((s) => ({ ...s, index: b }))}>{byteLabel(b)}</button>
                ))}
              </span>
            </div>
            <label className="field" style={NARROW_OK}>
              <Translated vi="Mô hình đoạn">Segment model</Translated>
              <select value={x.model} aria-label={trText('Segment model', 'Mô hình đoạn')} onChange={(e) => setX((s) => ({ ...s, model: e.target.value as SegmentModelId }))}>
                {Object.values(SEGMENT_MODELS).map((m) => <option key={m.id} value={m.id}>{trText(m.label, ({ nvidia: 'NVIDIA: sector 32 byte trong dòng 128 byte', amd64: 'AMD: dòng cache 64 byte', amd128: 'AMD: dòng cache 128 byte', metal: 'Apple: không có tài liệu' } as Record<string, string>)[m.id] ?? m.label)}</option>)}
              </select>
            </label>
          </div>

          <Translated
            vi={
              <p className="narrate" data-readout="summary" style={{ marginTop: 0 }}>
                {big(len.length)} × {big(cols)}, <b data-out="nnz">{big(nnz)}</b> phần tử khác 0 ({pct(nnz / (len.length * cols))}), hàng
                trung bình dài <b data-out="mean">{fmt(mean)}</b> và hàng dài nhất dài <b data-out="max">{big(maxLen)}</b>. ELL lưu{' '}
                <b data-out="ratio">{fmt(ratio, 4)}</b> lần số lượng số của CSR: độ rộng của nó là hàng dài nhất, không phải trung bình. Ở K ={' '}
                {K}, dạng lai giữ <b data-out="hyb-pads">{big(hyb.pads)}</b> vị trí đệm và chuyển{' '}
                <b data-out="hyb-coo">{big(hyb.coo)}</b> mục sang COO.
              </p>
            }
          >
            <p className="narrate" data-readout="summary" style={{ marginTop: 0 }}>
              {big(len.length)} × {big(cols)}, <b data-out="nnz">{big(nnz)}</b> non-zeros ({pct(nnz / (len.length * cols))}), a
              mean row of <b data-out="mean">{fmt(mean)}</b> and a longest of <b data-out="max">{big(maxLen)}</b>. ELL stores{' '}
              <b data-out="ratio">{fmt(ratio, 4)}</b> times the numbers of CSR: its width is the longest row, not the mean. At K ={' '}
              {K} the hybrid keeps <b data-out="hyb-pads">{big(hyb.pads)}</b> pad slots and moves{' '}
              <b data-out="hyb-coo">{big(hyb.coo)}</b> entries to COO.
            </p>
          </Translated>
          <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'minmax(0, 1fr)' }}>
            <StoragePanel table={table} x={{ ...x, S }} w={w} />
            <WarpPanel wt={wt} len={len} width={width} x={{ ...x, K }} setT={(t) => setX((s) => ({ ...s, t }))} />
            <TrafficPanel tr={tr} model={x.model} />
          </div>
        </div>
      </Bench>
    </div>
  );
}

// ---------------------------------------------------------------- the page
export function S175() {
  const [x, setXRaw] = useState<XState>(() => fromPreset('book'));
  const setX = (f: (s: XState) => XState) => setXRaw(f);
  const anchor = useRef<HTMLDivElement | null>(null);
  const show = (id: Preset['id'], K?: number) => {
    setXRaw((s) => fromPreset(id, s, K));
    anchor.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };
  return (
    <>
      <Translated
        vi={
          <p>
            Trang <a href={href('ch17', '17.2')}>17.2</a> để lại cho ELL hai vấn đề: một hàng dài khiến mọi hàng khác phải đệm, và
            các luồng vẫn phân kỳ. Trang này định giá cả hai trên các hàng có độ dài không đều, rồi mua sự đều đặn theo ba cách: cho
            phần đuôi của các hàng dài tràn sang COO, sắp xếp các hàng theo độ dài, hoặc đệm từng lát thay vì cả ma trận. Mỗi cách tốn một
            lượt tiền xử lý, nên trang kết thúc bằng việc khi nào điều đó đáng giá, và nơi bạn sẽ gặp tất cả những điều này trong suy luận
            LLM.
          </p>
        }
      >
        <p>
          Page <a href={href('ch17', '17.2')}>17.2</a> left ELL with two problems: one long row pads every other row, and
          the threads still diverge. This page prices both on rows of uneven length, then buys regularity three ways: spill
          the tails of long rows into COO, sort the rows by length, or pad slices instead of the whole matrix. Each costs a
          preprocessing pass, so the page ends with when that pays, and with the place you will meet all of it in LLM
          inference.
        </p>
      </Translated>
      <Source section="17.4–17.6" pages="412–418" />

      <h2>
        <Translated vi="Giá của phần đệm do hàng dài nhất quyết định">Padding is priced by the longest row</Translated>
      </h2>
      <Translated
        vi={
          <p>
            ELL đệm mọi hàng cho dài bằng hàng dài nhất. Ví dụ của sách: một ma trận 1000 × 1000 với 1% phần tử khác 0, tức
            trung bình 10 mỗi hàng, ngoại trừ một hàng có 200. Hãy đếm giá trị và chỉ số, như sách làm, và so sánh ELL
            với CSR.
          </p>
        }
      >
        <p>
          ELL pads every row to the length of the longest. The book's example: a 1000 × 1000 matrix with 1% non-zeros, so
          10 per row on average, except that one row has 200. Count values and indices, as the book does, and compare ELL
          with CSR.
        </p>
      </Translated>
      <Source section="17.4" pages="412–413" />
      <Predict
        id="ch17/17.5/p-book1000"
        prompt={
          <Translated vi="Với ví dụ của sách, ELL lưu số lượng số gấp bao nhiêu lần CSR?">
            <>For the book's example, how many times as many numbers does ELL store as CSR?</>
          </Translated>
        }
        unit="×"
      >
        {(v) => (
          <div data-reveal="ch17/17.5/p-book1000">
            <p>
              <Translated
                vi={
                  <>
                    CSR: 2 × 10,000 + 1,001 = <b>{big(BOOK_CSR.numbers)}</b> số, bằng {pct(BOOK_CSR.numbers / 1e6)} ma trận
                    dày đặc. ELL đệm cả 1,000 hàng lên 200: 2 × 200,000 = <b>{big(BOOK_ELL.numbers)}</b>,{' '}
                    {pct(BOOK_ELL.numbers / 1e6)}. Tỉ số là <b data-out="ratio">{fmt(BOOK_RATIO, 4)}</b>; sách làm tròn nó
                    thành 20. {Math.abs(v - BOOK_RATIO) < 1.5 ? 'Bạn đã đúng.' : 'Một hàng quyết định độ rộng của cả 1,000 hàng.'}
                  </>
                }
              >
                CSR: 2 × 10,000 + 1,001 = <b>{big(BOOK_CSR.numbers)}</b> numbers, {pct(BOOK_CSR.numbers / 1e6)} of the dense
                matrix. ELL pads all 1,000 rows to 200: 2 × 200,000 = <b>{big(BOOK_ELL.numbers)}</b>,{' '}
                {pct(BOOK_ELL.numbers / 1e6)}. The ratio is <b data-out="ratio">{fmt(BOOK_RATIO, 4)}</b>; the book rounds it
                to 20. {Math.abs(v - BOOK_RATIO) < 1.5 ? 'You had it.' : 'One row decided the width of all 1,000.'}
              </Translated>
            </p>
            <button onClick={() => show('book')}>
              <Translated vi="Nạp ví dụ của sách vào trình khám phá">Load the book's example in the explorer</Translated>
            </button>
          </div>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Sách cũng nói các hàng khác có "fewer than 10" (ít hơn 10) phần tử khác 0. Điều đó không thể khớp với tổng 1%: 999
            hàng mỗi hàng tối đa 9 cộng một hàng 200 cho tối đa 9,191, thiếu so với 10,000. Trình khám phá giữ 1% và cho các
            hàng khác 9 hoặc 10. Số đếm của ELL không đổi, vì nó chỉ phụ thuộc vào hàng dài nhất.
          </p>
        }
      >
        <p>
          The book also says the other rows have "fewer than 10" non-zeros. That cannot hold together with a 1% total: 999
          rows of at most 9 plus one of 200 make at most 9,191, short of 10,000. The explorer keeps the 1% and gives the
          other rows 9 or 10. ELL's count does not change, since it depends only on the longest row.
        </p>
      </Translated>
      <Beyond title={trText('Our arithmetic', 'Phép tính của chúng tôi')}>
        <Translated
          vi={
            <p>
              Con số 19.05 là của khóa học này, đã kiểm tra trong ch17-facts E1; sách in "20×". Chỗ mâu thuẫn là phép tính của chúng tôi trên các
              con số của sách, 999 × 9 + 200 = 9,191; sách in "fewer than 10" (pp. 412–413).
            </p>
          }
        >
          <p>
            The 19.05 is ours, checked in ch17-facts E1; the book prints "20×". The conflict is our arithmetic on the book's
            figures, 999 × 9 + 200 = 9,191; the book prints "fewer than 10" (pp. 412–413).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Trình khám phá lưu cùng các hàng đó theo sáu cách và chạy một kernel mỗi-hàng-một-luồng trên chúng. Hãy bắt đầu từ ví dụ của sách, rồi
            thay đổi từng điều khiển một. Quan sát ba điều: kích thước của ELL đi theo hàng dài nhất, không phải trung bình; một độ rộng lai
            K gần hàng điển hình loại bỏ gần như toàn bộ phần đệm với cái giá vài mục COO; và sắp xếp các hàng, riêng lẻ
            hay cùng các lát nhỏ, làm tăng phần thời gian lane làm việc. Trên ví dụ của sách, một hàng duy nhất dài 200
            che mất hiệu ứng cuối cùng đó: hãy chuyển sang thiết lập sẵn "Trải, 1 đến 31" để thấy việc sắp xếp tác động thế nào đến thời gian lane.
          </p>
        }
      >
        <p>
          The explorer stores the same rows six ways and runs a per-row kernel over them. Start on the book's example, then
          change one control at a time. Watch three things: ELL's size follows the longest row, not the mean; a hybrid width
          K near the typical row removes almost all the padding for a handful of COO entries; and sorting the rows, alone
          or with small slices, raises the share of lane time that does work. On the book's example the single row of 200
          hides that last effect: switch to the "Spread, 1 to 31" preset to see what the sort does to lane time.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Độ dài hàng được sinh ra, không bao giờ đọc từ ma trận thật. Warp được đếm, không được đo thời gian. Lưu lượng chỉ đếm
              mảng giá trị và mảng chỉ số; <code>x</code>, <code>y</code> và các mảng nhỏ theo hàng bị bỏ ra. Không có
              mô hình cache nào nằm giữa "mọi yêu cầu" và "mỗi đoạn một lần". Kích thước đoạn của Apple không có tài liệu, nên lựa chọn
              đó ẩn khung lưu lượng.
            </p>
          }
        >
          <p>
            Row lengths are generated, never read from real matrices. Warps are counted, not timed. Traffic counts only the
            value and index arrays; <code>x</code>, <code>y</code> and the small per-row arrays are left out. There is no
            cache model between "every request" and "each segment once". Apple's segment size is not documented, so that
            choice hides the traffic panel.
          </p>
        </Translated>
      </Simplification>
      <Explorer x={x} setX={setX} anchor={anchor} />

      <h2>
        <Translated vi="Cho phần đuôi tràn ra: dạng lai ELL-COO">Spill the tail: hybrid ELL-COO</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Khi vài hàng dài hơn hẳn phần còn lại, hãy cắt ngắn chúng. Giữ K phần tử khác 0 đầu tiên của mọi hàng trong ELL, chuyển
            phần còn lại của mỗi hàng dài vào một danh sách COO, rồi chạy SpMV/ELL trên phần ELL và SpMV/COO trên phần thừa. Dùng
            hai định dạng cho một tích là điều sách gọi là phương pháp lai.
          </p>
        }
      >
        <p>
          When a few rows are much longer than the rest, cut them short. Keep the first K non-zeros of every row in ELL, move
          the rest of each long row into a COO list, then run SpMV/ELL on the ELL part and SpMV/COO on the leftovers. Using
          two formats for one product is what the book calls a hybrid method.
        </p>
      </Translated>
      <Source section="17.5" pages="414" figs={trText('Fig. 17.13', 'Hình 17.13')} />
      <Predict
        id="ch17/17.5/p-fig1713"
        prompt={
          <>
            <Translated
              vi={
                <>
                  Tám hàng của Hình 17.13 có {FIG.join(', ')} phần tử khác 0. ELL thuần rộng {FIG_ELL.width} và chứa{' '}
                  {FIG_ELL.pads} vị trí đệm. Cắt phần ELL còn độ rộng 2 và gửi phần còn lại của mỗi hàng sang COO. Còn lại bao nhiêu vị trí
                  đệm?
                </>
              }
            >
              Fig. 17.13's eight rows have {FIG.join(', ')} non-zeros. Pure ELL is {FIG_ELL.width} wide and holds{' '}
              {FIG_ELL.pads} pad slots. Cut the ELL part to width 2 and send the rest of each row to COO. How many pad slots
              are left?
            </Translated>
          </>
        }
        unit={trText('pad slots', 'vị trí đệm')}
      >
        {(v) => (
          <div data-reveal="ch17/17.5/p-fig1713">
            <p>
              <Translated
                vi={
                  <>
                    <b data-out="pads">{FIG_K2.pads}</b>: các hàng 0, 4 và 7 mỗi hàng có một phần tử khác 0 và mỗi hàng cần một phần đệm.{' '}
                    <b data-out="coo">{FIG_K2.coo}</b> mục đi sang COO, {spills(FIG, 2).flatMap((s, r) => (s ? [`${s} từ hàng ${r}`] : [])).join(' và ')}, và giờ mỗi luồng chạy tối
                    đa 2 lần lặp. Phần chữ của sách nói ba mục đó đến từ "row 2" (hàng 2); hình và các số đếm của nó cho thấy là hàng 1
                    (ghi chú, "Unclear in the PDF" mục 2). {v === FIG_K2.pads ? 'Bạn đã đúng.' : ''}
                  </>
                }
              >
                <b data-out="pads">{FIG_K2.pads}</b>: rows 0, 4 and 7 have one non-zero each and need one pad each.{' '}
                <b data-out="coo">{FIG_K2.coo}</b> entries go to COO, {FIG_SPILL.join(' and ')}, and every thread now makes at
                most 2 iterations. The book's text says the three come from "row 2"; its figure and the counts say row 1
                (notes, "Unclear in the PDF" item 2). {v === FIG_K2.pads ? 'You had it.' : ''}
              </Translated>
            </p>
            <button onClick={() => show('fig1713', 2)}>
              <Translated vi="Hiện Hình 17.13 ở K = 2 trong trình khám phá">Show Fig. 17.13 at K = 2 in the explorer</Translated>
            </button>
          </div>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Cả hai lần khởi chạy đều cộng vào <code>y</code>, nên chúng kết hợp được theo thứ tự nào cũng được: lượt COO cộng phần tràn của một hàng lên trên
            bất cứ thứ gì lượt ELL đã ghi. Điều đó dựa vào việc kernel ELL cộng dồn. Đoạn mã của sách dùng phép gán{' '}
            <code>y[row] = sum</code>, điều này sẽ buộc lượt ELL phải chạy trước; kernel đã sửa ở trang{' '}
            <a href={href('ch17', '17.2')}>17.2</a> cộng dồn giống kernel CSR.
          </p>
        }
      >
        <p>
          Both launches add into <code>y</code>, so they compose in either order: the COO pass adds a row's overflow on top
          of whatever the ELL pass wrote. That relies on the ELL kernel accumulating. The book's listing assigns{' '}
          <code>y[row] = sum</code>, which would force the ELL pass to run first; the corrected kernel on page{' '}
          <a href={href('ch17', '17.2')}>17.2</a> accumulates like the CSR one.
        </p>
      </Translated>
      <Source section="17.5" pages="414" figs={trText('Fig. 17.12', 'Hình 17.12')} />
      <Translated
        vi={
          <p>
            Cái giá là khả năng truy cập. Một hàng bị tràn không còn nằm ở một chỗ, và tìm mọi phần tử khác 0 của nó nghĩa là phải
            tìm trong phần COO. Mọi thứ khác đều tốt hơn ELL đơn thuần: ít không gian hơn, ít phân kỳ hơn trong kernel ELL, không có phân kỳ
            trong kernel COO, và cả hai vẫn được hợp nhất truy cập bộ nhớ.
          </p>
        }
      >
        <p>
          The price is accessibility. A row that overflowed is no longer in one place, and finding all its non-zeros means
          searching the COO part. Everything else improves on ELL alone: less space, less divergence in the ELL kernel, none
          in the COO kernel, and both still coalesce.
        </p>
      </Translated>
      <Source section="17.5" pages="415" />
      <Translated
        vi={
          <p>
            Các nhà cung cấp chia rẽ về định dạng này. NVIDIA đã bỏ nó; AMD vẫn cung cấp nó và cho bạn chọn độ rộng của phần
            ELL.
          </p>
        }
      >
        <p>
          The vendors split on this format. NVIDIA dropped it; AMD still ships it and lets you choose the width of its ELL
          part.
        </p>
      </Translated>
      <Today checked={trText('2026-11-20 (ch17-facts A5, B3)', '2026-11-20 (ch17-facts, mục A5, B3)')}>
        <Translated
          vi={
            <p>
              NVIDIA đã gỡ định dạng lai khỏi cuSPARSE trong CUDA 11.0 (bị đánh dấu lỗi thời từ 10.2). rocSPARSE của AMD vẫn cung cấp{' '}
              <code>rocsparse_hybmv</code>, với độ rộng ELL được chọn tự động, do bạn chọn, hoặc bằng hàng dài nhất và không có phần
              COO (<code>auto</code>, <code>user</code>, <code>max</code>).
            </p>
          }
        >
          <p>
            NVIDIA removed the hybrid format from cuSPARSE in CUDA 11.0 (deprecated in 10.2). AMD's rocSPARSE still ships{' '}
            <code>rocsparse_hybmv</code>, with the ELL width chosen automatically, by you, or as the longest row with no COO
            part (<code>auto</code>, <code>user</code>, <code>max</code>).
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS175['ch17/17.5/q-hybrid-k']} />

      <h2>
        <Translated vi="Sắp xếp theo độ dài: JDS">Sort by length: JDS</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Dạng lai loại bỏ các trường hợp cực đoan; sắp xếp loại bỏ sự pha trộn. Sắp xếp các hàng theo độ dài, hàng dài nhất trước, và lưu chúng
            theo thứ tự cột (column-major) không đệm: lần lặp <code>t</code> chứa phần tử thứ <code>t</code> của mọi hàng có
            phần tử đó, và vì các hàng đã được sắp xếp, đó là những hàng đầu tiên. Một mảng <code>iterPtr</code> ghi lại nơi mỗi
            lần lặp bắt đầu, và một mảng <code>row</code> ghi lại mỗi hàng đã sắp xếp vốn là hàng gốc nào. Các luồng kề nhau
            giờ nhận các hàng có độ dài tương tự, nên các luồng của một warp có số vòng lặp tương tự và phân kỳ luồng giảm đi nhiều.
            Sách gọi đây là Jagged Diagonal Storage.
          </p>
        }
      >
        <p>
          A hybrid removes the extremes; sorting removes the mixture. Sort the rows by length, longest first, and store them
          column-major with no padding: iteration <code>t</code> holds the <code>t</code>-th element of every row that has
          one, and because the rows are sorted, those are the first rows. An <code>iterPtr</code> array records where each
          iteration starts, and a <code>row</code> array records which original row each sorted row was. Neighbouring threads
          now get rows of similar length, so the threads of a warp have similar trip counts and divergence is much reduced.
          The book calls this Jagged Diagonal Storage.
        </p>
      </Translated>
      <Source section="17.6" pages="416–417" figs={trText('Figs. 17.14, 17.15', 'Hình 17.14, 17.15')} />
      <Translated
        vi={
          <p>
            Có hai chỗ sai sót của bản PDF ở đây. Phần chữ nói các hàng được sắp xếp "in increasing order" (theo thứ tự tăng dần), trong khi đoạn trước
            đó và Hình 17.14 sắp hàng dài nhất trước; khóa học này theo hình. Và đoạn chấm điểm JDS ở p. 417
            mở đầu bằng "Now we examine the ELL format" (giờ ta xem xét định dạng ELL): ý là JDS. Cả hai đều có trong danh sách những chỗ chưa rõ của phần ghi chú (mục 3
            và 4).
          </p>
        }
      >
        <p>
          Two slips in the PDF sit here. The text says the rows are sorted "in increasing order", while the paragraph before
          it and Fig. 17.14 sort longest first; this course follows the figure. And the paragraph that scores JDS on p. 417
          opens with "Now we examine the ELL format": it means JDS. Both are in the notes' list of unclear places (items 3
          and 4).
        </p>
      </Translated>
      <Source section="17.6" pages="416–417" />
      <Ask q={questionsS175['ch17/17.5/q-sort-y']} />
      <Translated
        vi={
          <p>
            Sắp xếp có hai cái giá ngoài bản thân việc sắp xếp. Tính linh hoạt: thêm một phần tử khác 0 làm thay đổi độ dài của một hàng và có thể buộc phải sắp xếp
            lại. Căn lề: lần lặp <code>t</code> bắt đầu ở <code>iterPtr[t]</code>, ở bất cứ đâu các số đếm hàng đặt nó, nên
            không có cách rẻ nào để mọi lần lặp bắt đầu ở một ranh giới đoạn, như ELL có thể làm bằng cách đệm số hàng của nó.
          </p>
        }
      >
        <p>
          Sorting has two costs besides the sort. Flexibility: adding a non-zero changes a row's length and may force a new
          sort. Alignment: iteration <code>t</code> starts at <code>iterPtr[t]</code>, wherever the row counts put it, so
          there is no cheap way to make every iteration start on a segment boundary, as ELL can by padding its row count.
        </p>
      </Translated>
      <Source section="17.6" pages="417–418" />
      <Ask q={questionsS175['ch17/17.5/q-jds-align']} />
      <Ask q={questionsS175['ch17/17.5/q-jds-loop']} />
      <Translated
        vi={
          <p>
            Con đường trung dung của sách là biến thể chia đoạn: sau khi sắp xếp, cắt các hàng thành từng đoạn và cho mỗi đoạn
            ELL riêng của nó, chỉ đệm tới hàng dài nhất của đoạn đó. Các hàng trong một đoạn có độ dài tương tự, nên có ít
            phần đệm, và mỗi đoạn vẫn đều đặn. Trong trình khám phá, đánh dấu ô sắp xếp và giảm kích thước lát: các vị trí đệm
            gần như biến mất trong khi mỗi lát vẫn là một ELL.
          </p>
        }
      >
        <p>
          The book's middle road is the sectioned variant: after sorting, cut the rows into sections and give each section
          its own ELL, padded only to that section's longest row. Rows in a section have similar lengths, so there is little
          padding, and each section stays regular. In the explorer, tick the sort and lower the slice size: the pad slots
          nearly vanish while every slice stays an ELL.
        </p>
      </Translated>
      <Source section="17.6" pages="417" />
      <Translated
        vi={
          <p>
            Cả hai thư viện của nhà cung cấp đều có ý tưởng này dưới dạng nào đó, và không bên nào cung cấp chính JDS. Sliced ELL của NVIDIA là
            biến thể chia đoạn không sắp xếp. Thuật toán LRB của AMD làm việc sắp xếp dưới dạng chia nhóm: nó xếp các hàng vào các ngăn theo độ dài.
          </p>
        }
      >
        <p>
          Both vendor libraries ship this idea in some form, and neither ships JDS itself. NVIDIA's Sliced ELL is the
          sectioned variant without the sort. AMD's LRB algorithm does the sort as bucketing: it bins rows by length.
        </p>
      </Translated>
      <Today checked={trText('2026-11-20 (ch17-facts A6, B4)', '2026-11-20 (ch17-facts, mục A6, B4)')}>
        <Translated
          vi={
            <p>
              Sliced ELL đệm bằng cột −1. Các ngăn của LRB là lũy thừa của hai; điều đó được đọc từ mã nguồn của AMD.
            </p>
          }
        >
          <p>
            Sliced ELL pads with column −1. LRB's bins are powers of two; that is read from AMD's source code.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Khi nào phần tiền xử lý đáng giá">When preprocessing pays</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Tách, sắp xếp và chia lát đều là công việc làm trước tích đầu tiên, và với một tích duy nhất chúng có thể tốn
            nhiều hơn phần tiết kiệm được. Trong một bộ giải lặp, cùng một <code>A</code> quay lại ở mọi lần lặp, chỉ có{' '}
            <code>x</code> và <code>y</code> thay đổi, nên chi phí được dàn trải qua tất cả các lần lặp.
          </p>
        }
      >
        <p>
          Splitting, sorting and slicing are all work done before the first product, and for a single product they can cost
          more than they save. In an iterative solver the same <code>A</code> comes back every iteration, with only{' '}
          <code>x</code> and <code>y</code> changing, so the cost is spread over all of them.
        </p>
      </Translated>
      <Source section="17.5, 17.6" pages="414–415, 417" />
      <Translated
        vi={
          <p>
            Cả hai thư viện của nhà cung cấp đều đưa lập luận này vào API của chúng: một lời gọi hoặc giai đoạn tiền xử lý chạy một lần cho mỗi ma trận
            và tăng tốc các tích sau đó có cùng mẫu thưa.
          </p>
        }
      >
        <p>
          Both vendor libraries build this argument into their APIs: a preprocessing call or stage that runs once per matrix
          and speeds up the later products that share its sparsity pattern.
        </p>
      </Translated>
      <Today checked={trText('2026-11-20 (ch17-facts A8)', '2026-11-20 (ch17-facts, mục A8)')}>
        <Translated
          vi={
            <p>
              <code>cusparseSpMV_preprocess</code> của cuSPARSE "may accelerate subsequent calls" (có thể tăng tốc các lời gọi sau); CSR thích ứng của rocSPARSE
              "needs to be amortized over many subsequent sparse vector products" (cần được khấu hao qua nhiều tích vector thưa sau đó).
            </p>
          }
        >
          <p>
            cuSPARSE's <code>cusparseSpMV_preprocess</code> "may accelerate subsequent calls"; rocSPARSE's adaptive CSR
            "needs to be amortized over many subsequent sparse vector products".
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS175['ch17/17.5/q-amortise']} />
      <Translated
        vi={
          <p>
            Cùng lập luận đó áp dụng cho việc đóng gói lại trọng số mô hình một lần, khi chúng được nạp, vào bất kỳ cách bố trí nào mà kernel đọc
            nhanh nhất, trước khi nhiều lượt lan truyền xuôi dùng lại chúng.
          </p>
        }
      >
        <p>
          The same argument covers repacking model weights once, when they are loaded, into whatever layout the kernel reads
          fastest, before many forward passes reuse them.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Phép so sánh với việc đóng gói lại trọng số là của khóa học này.</p>}>
          <p>The weight-repacking analogy is ours.</p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Hiệu năng phụ thuộc vào dữ liệu">Performance that depends on the data</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hầu hết các kernel trước đó trong sách chạy như nhau bất kể các con số trong đầu vào. Những kernel này thì không. ELL lưu bao nhiêu
            phần đệm, bao nhiêu lane rảnh trong CSR và các lần lặp của JDS thẳng hàng tốt đến đâu đều phụ thuộc vào chỗ các
            phần tử khác 0 rơi vào. Sách gọi tính toán thưa là một ví dụ đơn giản về hiệu năng phụ thuộc dữ liệu (pp. 421–422).
          </p>
        }
      >
        <p>
          Most earlier kernels in the book ran the same way whatever the numbers in their input. These do not. How much
          padding ELL stores, how many lanes idle in CSR and how well JDS's iterations line up all depend on where the
          non-zeros fall. The book calls sparse computation a simple example of data-dependent performance (pp. 421–422).
        </p>
      </Translated>
      <Source section="17.8" pages="421–422" />
      <Translated
        vi={
          <p>
            Sách thêm rằng mã thưa đạt ít FLOPS hơn nhiều so với mã dày đặc trên CPU và GPU; trên cùng trang đó sách lưu ý rằng
            các định dạng thưa phải đọc chỉ số tường minh và khó phân ô.
          </p>
        }
      >
        <p>
          It adds that sparse code reaches far fewer FLOPS than dense code on CPUs and GPUs; on the same page it notes that
          sparse formats must read explicit indices and are hard to tile.
        </p>
      </Translated>
      <Source section="17.8" pages="422" />
      <Translated
        vi={
          <p>
            Hãy coi tuyên bố về FLOPS là của sách: <a href={href('ch17', 'labs')}>lab</a> đo các kernel của chính bạn so với
            số byte chúng di chuyển, trên các hàng đều, trải rộng, sắp xếp theo độ dài và lệch. Số byte thì bạn
            có thể kiểm tra ở đây. Đặt giá trị của trình khám phá là 4 bit và chỉ số là 2 byte: khi đó CSR đọc 2.5 byte mỗi trọng số
            được giữ so với 0.5 mỗi vị trí cho ma trận dày đặc, nên nó thua khi hơn một phần năm số trọng số được giữ.
            Hãy thử 256 hàng với trung bình 64.
          </p>
        }
      >
        <p>
          Treat the FLOPS claim as the book's: the <a href={href('ch17', 'labs')}>lab</a> measures your own kernels against
          the bytes they move, on rows that are uniform, spread, sorted by length and skewed. The bytes you
          can check here. Set the explorer's values to 4 bits and its indices to 2 bytes: CSR then reads 2.5 bytes per kept
          weight against 0.5 per position for the dense matrix, so it loses once more than a fifth of the weights are kept.
          Try 256 rows with a mean of 64.
        </p>
      </Translated>
      <Aside>
        <Translated vi={<p>ch17-facts A12: không tài liệu nào của nhà cung cấp được đọc cho khóa học này nêu tuyên bố về FLOPS đó.</p>}>
          <p>ch17-facts A12: no vendor document read for this course states the FLOPS claim.</p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Nơi điều này gặp suy luận LLM: định tuyến token tới expert">Where this meets LLM inference: routing tokens to experts</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một lớp mixture-of-experts chạy một router nhỏ chọn, cho mỗi token, vài expert trong số nhiều expert, và mỗi
            expert có trọng số riêng. Đọc đầu ra của router dưới dạng các cặp (expert, token) thì nó là một ma trận thưa ở dạng COO,
            liệt kê theo thứ tự token. Để chạy mỗi expert như một GEMM dày đặc trên các token của nó, các cặp phải được nhóm theo expert,
            và đó chính là bài tập 3: từ COO sang CSR.
          </p>
        }
      >
        <p>
          A mixture-of-experts layer runs a small router that picks, for every token, a few experts out of many, and each
          expert has its own weights. Read the router's output as (expert, token) pairs and it is a sparse matrix in COO,
          listed in token order. To run each expert as one dense GEMM over its tokens, the pairs must be grouped by expert,
          and that is exercise 3: COO to CSR.
        </p>
      </Translated>
      <Source section="17.9, 20.7" pages="422, 510" />
      <Translated
        vi={
          <p>
            Bốn lượt, lượt này nuôi lượt sau. Đếm số cặp mỗi expert, một histogram như ở chương 9. Scan các số đếm thành
            độ lệch, chương 11. Rải mỗi cặp tới độ lệch của expert của nó cộng hạng của nó, và chạy một GEMM cho mỗi lát liền nhau.
            Rồi rải kết quả trở lại qua hoán vị ngược, chính là mảng row của JDS.
          </p>
        }
      >
        <p>
          Four passes, each feeding the next. Count the pairs per expert, a histogram as in chapter 9. Scan the counts into
          offsets, chapter 11. Scatter each pair to its expert's offset plus its rank, and run one GEMM per contiguous slice.
          Then scatter the results back through the inverse permutation, which is JDS's row array.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Phép ánh xạ định tuyến MoE sang bài tập 3 và JDS là của khóa học này; mã mà nó dựa vào nằm ở ch17-facts D5 đến D7.
            </p>
          }
        >
          <p>
            The mapping of MoE routing onto exercise 3 and JDS is ours; the code it rests on is in ch17-facts D5 to D7.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Hoạt hình chạy bốn lượt trên 8 token với định tuyến top-2 qua 4 expert, trong đó có một expert không token nào chọn.
          </p>
        }
      >
        <p>
          The animation runs the four passes on 8 tokens with top-2 routing over 4 experts, one of which no token picks.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi lượt được tính một lần, không có luồng và không có thời gian; trọng số của router và phép tính số học của các GEMM
              bị bỏ ra. Việc đệm mỗi lát lên trọn một ô (tile) cỡ 4 là minh họa của khóa học này: đường CUDA của llama.cpp đệm danh sách id
              nói chung, còn kernel Metal của nó đệm mỗi lát thêm một mục.
            </p>
          }
        >
          <p>
            Every pass is computed in one go, with no threads and no timing; the router's weights and the GEMMs' arithmetic
            are left out. Padding each slice to a whole tile of 4 is our illustration: llama.cpp's CUDA path pads the id list
            as a whole, and its Metal kernel pads each slice by one entry.
          </p>
        </Translated>
      </Simplification>
      <RouteScene />
      <Translated
        vi={
          <p>
            Trên CUDA và HIP, llama.cpp làm việc này trong <code>mm_ids_helper</code>, một khối mỗi expert, xếp hạng các mục bằng một
            ballot và một popcount, và cấp phát danh sách id của nó ở dạng "padded for unconditional memory access" (được đệm để truy cập bộ nhớ không điều kiện). Kernel Metal của nó{' '}
            <code>kernel_mul_mm_id_map0</code> đếm, lấy tổng tiền tố (prefix sum) loại trừ, và đệm mỗi lát thêm một mục để
            mọi luồng thực hiện một lần ghi bất kể token của nó khớp với gì; chú thích nói "conditional stores of the divergent
            lanes are very slow" (các lần ghi có điều kiện của các lane phân kỳ rất chậm). Đó là quan sát của llama.cpp từ chính mã của nó, không phải tài liệu của Apple. mlx-lm sắp xếp
            các token theo expert bằng argsort, giữ argsort của kết quả đó làm hoán vị ngược, và chỉ làm vậy khi có từ 64
            mục trở lên, ngưỡng khấu hao của riêng nó. Bản thân MLX không có kiểu mảng thưa; thay vào đó nó cung cấp{' '}
            <code>gather_mm</code> với cờ <code>sorted_indices</code>.
          </p>
        }
      >
        <p>
          On CUDA and HIP, llama.cpp does this in <code>mm_ids_helper</code>, one block per expert, ranking entries with a
          ballot and a popcount, and allocates its id list "padded for unconditional memory access". Its Metal kernel{' '}
          <code>kernel_mul_mm_id_map0</code> counts, takes an exclusive prefix sum, and pads every slice by one entry so that
          every thread makes one store whatever its token matched; the comment says "conditional stores of the divergent
          lanes are very slow". That is llama.cpp's observation from its own code, not Apple documentation. mlx-lm sorts
          tokens by expert with an argsort, keeps the argsort of that as the inverse, and does so only once there are 64 or
          more entries, its own amortisation threshold. MLX itself has no sparse array type; it offers{' '}
          <code>gather_mm</code> with a <code>sorted_indices</code> flag instead.
        </p>
      </Translated>
      <Today checked={trText('2026-11-20 (ch17-facts D5, D6, D7)', '2026-11-20 (ch17-facts, mục D5, D6, D7)')}>
        <Translated
          vi={
            <p>
              llama.cpp <code>ggml-cuda/mmid.cu</code>, <code>mmq.cu</code> và <code>ggml-metal/kernels/mul_mm.metal</code>;
              mlx-lm <code>mlx_lm/layers/moe.py</code>; MLX 0.32.4.
            </p>
          }
        >
          <p>
            llama.cpp <code>ggml-cuda/mmid.cu</code>, <code>mmq.cu</code> and <code>ggml-metal/kernels/mul_mm.metal</code>;
            mlx-lm <code>mlx_lm/layers/moe.py</code>; MLX 0.32.4.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS175['ch17/17.5/q-moe-offset']} />
      <Ask q={questionsS175['ch17/17.5/q-moe-atomic']} />

      <h2>
        <Translated vi="Các thư viện ngày nay">Libraries today</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Không định dạng giảng dạy nào của sách còn nguyên vẹn ở mọi nơi. Đây là những gì mỗi nhà cung cấp có cho SpMV, và đâu là
            các lựa chọn tất định.
          </p>
        }
      >
        <p>
          None of the book's teaching formats survives intact everywhere. This is what each vendor ships for SpMV, and where
          the deterministic choices are.
        </p>
      </Translated>
      <Today checked={trText('2026-11-20 (ch17-facts A2, A4, A5, B3 to B5, C4)', '2026-11-20 (ch17-facts, mục A2, A4, A5, B3 đến B5, C4)')}>
        <Translated vi={<p>Phiên bản theo báo cáo của các nguồn: cuSPARSE 13.5, rocSPARSE 5.1.0, hipSPARSE 4.8.0, macOS 27.0 SDK.</p>}>
          <p>Versions as the sources reported them: cuSPARSE 13.5, rocSPARSE 5.1.0, hipSPARSE 4.8.0, macOS 27.0 SDK.</p>
        </Translated>
      </Today>
      <table className="data wide" data-libraries>
        <thead>
          <tr>
            <th></th>
            <th>NVIDIA: cuSPARSE{/* i18n-ignore: product and format names */}</th>
            <th>AMD: rocSPARSE, hipSPARSE{/* i18n-ignore: product and format names */}</th>
            <th>Apple{/* i18n-ignore: product and format names */}</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="SpMV trên GPU">GPU SpMV</Translated></td>
            <td>
              <code>cusparseSpMV</code>
            </td>
            <td>
              <Translated vi={<><code>rocsparse_spmv</code>; <code>hipsparseSpMV</code> làm lớp di động</>}>
                <code>rocsparse_spmv</code>; <code>hipsparseSpMV</code> as the portable layer
              </Translated>
            </td>
            <td><Translated vi="Không có tài liệu nào; các hàm thưa của Accelerate chạy trên CPU">None documented; Accelerate's sparse routines run on the CPU</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Định dạng">Formats</Translated></td>
            <td>COO, CSR, CSC, Sliced ELL, BSR, Blocked-ELL{/* i18n-ignore: product and format names */}</td>
            <td>COO, CSR, CSC, ELL, HYB, Sliced ELL, BSR, Blocked ELL{/* i18n-ignore: product and format names */}</td>
            <td><Translated vi="MPSGraph tạo tensor COO, CSC và CSR, nhưng không có phép toán nào được ghi trong tài liệu dùng chúng">MPSGraph creates COO, CSC and CSR tensors, with no documented operation that consumes them</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="ELL, dạng lai, JDS của sách">The book's ELL, hybrid, JDS</Translated></td>
            <td><Translated vi="Không có ELL thường; dạng lai bị gỡ trong CUDA 11.0; không có JDS">No plain ELL; hybrid removed in CUDA 11.0; no JDS</Translated></td>
            <td><Translated vi="ELL và HYB; không có JDS">ELL and HYB; no JDS</Translated></td>
            <td><Translated vi="Không có">None</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="SpMV tất định">Deterministic SpMV</Translated></td>
            <td>
              <code>CSR_ALG2</code>, <code>COO_ALG2</code>, <code>SELL_ALG1</code>, <code>BSR_ALG1</code>
            </td>
            <td><Translated vi="rowsplit, coo, ell, bsr, sell (rocSPARSE); hipSPARSE không ghi tài liệu cái nào">rowsplit, coo, ell, bsr, sell (rocSPARSE); hipSPARSE documents none</Translated></td>
            <td><Translated vi="Không áp dụng">Not applicable</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Dùng lại qua nhiều tích">Reuse across products</Translated></td>
            <td>
              <code>cusparseSpMV_preprocess</code>
            </td>
            <td><Translated vi="một giai đoạn tiền xử lý">a preprocess stage</Translated></td>
            <td><Translated vi="Không áp dụng">Not applicable</Translated></td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Hai sự thật nữa hoàn thiện bức tranh. <code>DeviceSpmv</code> của CUB đã bị gỡ trong CCCL 3.0 kèm lời khuyên "Use
            cuSPARSE instead" (hãy dùng cuSPARSE thay thế). Và backend MPS của PyTorch tự viết kernel thưa của nó: mỗi luồng tìm lát của hàng mình trong
            các mảng COO đã sắp xếp bằng tìm kiếm nhị phân và lặp qua lát đó, không có atomic. Đó là vấn đề khả năng truy cập của trang{' '}
            <a href={href('ch17', '17.2')}>17.2</a> được giải theo cách thứ ba, khôi phục truy cập theo hàng từ COO đã sắp xếp mà không dựng{' '}
            <code>rowPtrs</code>, với cái giá là một lần tìm kiếm mỗi luồng.
          </p>
        }
      >
        <p>
          Two more facts complete the picture. CUB's <code>DeviceSpmv</code> was removed in CCCL 3.0 with the advice "Use
          cuSPARSE instead". And PyTorch's MPS backend writes its own sparse kernel: each thread finds its row's slice of the
          sorted COO arrays by binary search and loops over it, with no atomics. That is the accessibility problem of page{' '}
          <a href={href('ch17', '17.2')}>17.2</a> solved a third way, row access recovered from sorted COO without building{' '}
          <code>rowPtrs</code>, at the price of a search per thread.
        </p>
      </Translated>
      <Today checked={trText('2026-11-20 (ch17-facts A9, D2)', '2026-11-20 (ch17-facts, mục A9, D2)')}>
        <Translated
          vi={
            <p>
              PyTorch <code>aten/src/ATen/native/sparse/mps/kernels/SparseTensorMath.metal</code>, các kernel{' '}
              <code>spmm_addmm_coo</code>; cách đọc nó như một lời giải thứ ba là của khóa học này.
            </p>
          }
        >
          <p>
            PyTorch <code>aten/src/ATen/native/sparse/mps/kernels/SparseTensorMath.metal</code>, the{' '}
            <code>spmm_addmm_coo</code> kernels; the reading as a third solution is ours.
          </p>
        </Translated>
      </Today>
    </>
  );
}
