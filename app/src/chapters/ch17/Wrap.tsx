import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Beyond, Source, Today, type Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { warpAccess } from '../../sim/coalesce';
import { EXERCISE1, FIG17_14, spmvRef, toJDS } from '../../sim/sparse';

// Chapter 17's summary page: what to carry forward, exercise 5's JDS kernel (ours, loaded
// from labs/ch17/snippets/spmv_jds.cu), a mixed check on situations the pages did not show,
// where the book's five exercises are worked, the PDF's apparent slips, what the condensed
// chapter left out, and the project status. Wrap.snippets.test.ts runs the kernel on the
// emulator and pins every number below.

// ---------------------------------------------------------------- exercise 5's kernel
export const JDS_SNIPPET = 'ch17/snippets/spmv_jds.cu';
export const JDS_LOOP_LINE =
  'for (unsigned int t = 0; t < jdsMatrix.numIters && r < jdsMatrix.iterPtr[t + 1] - jdsMatrix.iterPtr[t]; ++t) {';
export const JDS_INDEX_LINE = 'unsigned int i = jdsMatrix.iterPtr[t] + r;';
export const JDS_STORE_LINE = 'y[jdsMatrix.row[r]] += sum;';
const JDS_STATUS = 'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware';
// `value` is left uncoloured: it names both a local float and the struct's device pointer.
const JDS_TIERS: Record<string, Tier> = { x: 'global', y: 'global', sum: 'reg', r: 'reg', t: 'reg', i: 'reg', col: 'reg' };

/** The file without its leading comment block, which the caption replaces. */
export function snippetBody(src: string): string {
  const lines = src.split('\n');
  let k = 0;
  while (k < lines.length && (lines[k].startsWith('//') || lines[k].trim() === '')) k++;
  return lines.slice(k).join('\n');
}
const linesWith = (text: string, needles: readonly string[]): number[] =>
  text.split('\n').flatMap((l, i) => (needles.some((n) => l.trim() === n) ? [i + 1] : []));

function JdsKernel() {
  const src = useLabFile(JDS_SNIPPET);
  const code = src ? snippetBody(src) : '// loading labs/' + JDS_SNIPPET;
  return (
    <Code
      tabs={[{
        label: tr('spmv_jds.cu (ours)', 'spmv_jds.cu (của khóa học này)'),
        code,
        tiers: JDS_TIERS,
        status: JDS_STATUS,
        highlight: src ? linesWith(code, [JDS_LOOP_LINE, JDS_INDEX_LINE, JDS_STORE_LINE]) : [],
      }]}
      caption={
        <>
          <Translated
            vi={
              <>
                Bài tập 5, được giải trong <code>labs/ch17/snippets/spmv_jds.cu</code>. Sách không đưa kernel JDS nào; kernel này và
                struct <code>JDSMatrix</code> của nó là của khóa học này. Được tô sáng: phép kiểm tra vòng lặp dừng luồng <code>r</code> khi
                có ít hơn <code>r + 1</code> hàng đạt tới lần lặp <code>t</code>, chỉ số của phần tử của nó, và lần ghi vào
                hàng gốc. Đã chạy trên trình giả lập của khóa học theo ba thứ tự luồng, trên Hình 17.14, ma trận của bài tập 1, một
                ma trận lệch và một ma trận có hàng rỗng.
              </>
            }
          >
            Exercise 5, answered in <code>labs/ch17/snippets/spmv_jds.cu</code>. The book shows no JDS kernel; this one and
            its <code>JDSMatrix</code> struct are ours. Highlighted: the loop test that stops thread <code>r</code> once
            fewer than <code>r + 1</code> rows reach iteration <code>t</code>, the index of its element, and the store into
            the original row. Run on the course's emulator in three thread orders, on Fig. 17.14, exercise 1's matrix, a
            skewed matrix and one with empty rows.
          </Translated>
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- numbers the questions use
const J14 = toJDS(FIG17_14);
const ONES6 = Array(6).fill(1);
/** Fig. 17.14's row sums with x all ones: what y should hold, starting from zero. */
export const ROW_SUMS = spmvRef(FIG17_14, ONES6, Array(6).fill(0));
/** y when the result goes to the sorted position r instead of row[r]. */
export const UNPERMUTED = J14.row.map((orig) => ROW_SUMS[orig]);
export const UNPERMUTED_RIGHT = UNPERMUTED.filter((v, r) => v === ROW_SUMS[r]).length;
/** Threads at work in each iteration of Fig. 17.14's JDS: [6, 5, 3, 1]. */
export const PER_ITER = J14.iterPtr.slice(1).map((p, t) => p - J14.iterPtr[t]);
const J_EX1 = toJDS(EXERCISE1);

/** One warp over uniform rows of 16 floats: CSR's sector requests and distinct sectors, and ELL's. */
const LEN = 16;
const WARP = 32; // NVIDIA's warp width; the question says it is NVIDIA's model (ch06-facts A2)
const csrIter = (t: number) => warpAccess(Array.from({ length: WARP }, (_, l) => (LEN * l + t) * 4), 4, 32);
const ellIter = (t: number) => warpAccess(Array.from({ length: WARP }, (_, l) => (t * WARP + l) * 4), 4, 32);
const ITERS = Array.from({ length: LEN }, (_, t) => t);
export const SECTORS = {
  csrRequests: ITERS.reduce((n, t) => n + csrIter(t).transactions, 0),
  csrDistinct: new Set(ITERS.flatMap((t) => csrIter(t).segments)).size,
  ellPerIter: ellIter(0).transactions,
  ellDistinct: new Set(ITERS.flatMap((t) => ellIter(t).segments)).size,
};

/** A decode batch padded to its longest sequence. */
export const RAGGED = { seqs: 64, short: 100, long: 2000 };
const RAGGED_REAL = (RAGGED.seqs - 1) * RAGGED.short + RAGGED.long;
const RAGGED_SLOTS = RAGGED.seqs * RAGGED.long;
export const RAGGED_PCT = (100 * RAGGED_REAL) / RAGGED_SLOTS;

const n = (v: number) => v.toLocaleString('en-US');
const list = (xs: readonly number[]) => '[' + xs.join(', ') + ']';

export const questionsWrap: Record<string, Question> = {
  'ch17/wrap/q-jds-unpermuted': {
    kind: 'numeric',
    id: 'ch17/wrap/q-jds-unpermuted',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Kernel ở trên chạy trên Hình 17.14, với <code>x</code> toàn số 1 và <code>y</code> bắt đầu bằng 0, chỉ khác là
              dòng cuối của nó ghi <code>y[r] += sum</code>, tức vị trí đã sắp xếp, thay vì <code>y[jdsMatrix.row[r]]</code>. Bao
              nhiêu trong sáu mục của <code>y</code> ra đúng?
            </>
          }
        >
          The kernel above on Fig. 17.14, with <code>x</code> all ones and <code>y</code> starting at zero, except that its
          last line writes <code>y[r] += sum</code>, the sorted position, instead of <code>y[jdsMatrix.row[r]]</code>. How
          many of the six entries of <code>y</code> come out right?
        </Translated>
      </>
    ),
    answer: UNPERMUTED_RIGHT,
    unit: 'entries',
    vi: { unit: 'mục', source: 'ch. 17 §17.6, pp. 416–417, Fig. 17.14 (giá trị của khóa học này); bài tập 5, p. 422' },
    explain: (
      <Translated
        vi={
          <p>
            Luồng <code>r</code> cộng hàng đã sắp xếp <code>r</code>, tức là hàng gốc <code>row[r]</code>, và{' '}
            <code>row = {list(J14.row)}</code>. Ghi vào <code>y[r]</code> đặt mỗi tổng vào sai chỗ trừ khi{' '}
            <code>row[r] = r</code>, và chỉ <code>r = 4</code> thỏa: hàng gốc 4 cũng đứng thứ năm sau khi sắp xếp. Với a đến o
            đại diện cho 1 đến 15, sáu tổng hàng là {list(ROW_SUMS)}, tất cả khác nhau, nên không mục nào khác đúng do
            trùng hợp: lỗi này cho ra {list(UNPERMUTED)}. Lần chạy trên trình giả lập bản sao đã kiểm tra của trang này cũng cho kết quả đó, 5 trên 6
            sai. Sắp xếp hàng là sắp lại các phương trình, nên chính <code>y</code> phải được đưa về chỗ cũ.
          </p>
        }
      >
        <p>
          Thread <code>r</code> sums sorted row <code>r</code>, which is original row <code>row[r]</code>, and{' '}
          <code>row = {list(J14.row)}</code>. Writing to <code>y[r]</code> lands each sum in the wrong place unless{' '}
          <code>row[r] = r</code>, and only <code>r = 4</code> qualifies: original row 4 also sorts fifth. With a to o
          standing for 1 to 15, the six row sums are {list(ROW_SUMS)}, all different, so no other entry is right by
          coincidence: the defect gives {list(UNPERMUTED)}. The emulator run of this page's checked copy agrees, 5 of 6
          wrong. Sorting rows reorders the equations, so it is <code>y</code> that has to be put back.
        </p>
      </Translated>
    ),
    source: 'ch. 17 §17.6, pp. 416–417, Fig. 17.14 (values ours); exercise 5, p. 422',
  },
  'ch17/wrap/q-transpose': {
    kind: 'choice',
    id: 'ch17/wrap/q-transpose',
    prompt: (
      <>
        <Translated
          vi={
            <>
              A được lưu ở dạng CSR, và một lượt lan truyền ngược cần <code>Aᵀ·g</code>, trong đó <code>g</code> có một mục cho mỗi hàng của A.
              Kế hoạch nào suy ra được từ chương này?
            </>
          }
        >
          A is stored in CSR, and a backward pass needs <code>Aᵀ·g</code>, where <code>g</code> has one entry per row of A.
          Which plan follows from the chapter?
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
                  Đọc các mảng CSR của A như CSC của Aᵀ và chạy SpMV/CSC, kể cả atomic trên đầu ra; nếu tích này
                  lặp lại, dựng CSR của Aᵀ một lần.
                </>
              }
            >
              Read A's CSR arrays as the CSC of Aᵀ and run SpMV/CSC, atomics on the output included; if the product
              repeats, build the CSR of Aᵀ once.
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
                <>
                  Chạy kernel CSR không đổi với <code>g</code> thay cho <code>x</code>: một luồng mỗi hàng, nên không cần
                  atomic.
                </>
              }
            >
              Run the CSR kernel unchanged with <code>g</code> in place of <code>x</code>: one thread per row, so no
              atomics.
            </Translated>
          </>
        ),
        why: <Translated vi="Cách đó tính A·g, vốn cần một mục của g cho mỗi cột của A. Đó là sai tích.">That computes A·g, which would need one entry of g per column of A. It is the wrong product.</Translated>,
      },
      {
        label: (
          <>
            <Translated
              vi={
                <>
                  Đổi vai trò của <code>rowPtrs</code> và <code>colIdx</code>, điều này biến CSR của A thành CSR của Aᵀ,
                  rồi chạy kernel mỗi-hàng-một-luồng không đổi, vốn không cần atomic.
                </>
              }
            >
              Swap the roles of <code>rowPtrs</code> and <code>colIdx</code>, which turns A's CSR into the CSR of Aᵀ,
              then run the unchanged per-row kernel, which needs no atomics.
            </Translated>
          </>
        ),
        why: <Translated vi="rowPtrs chứa m + 1 độ lệch, colIdx chứa z số hiệu cột: độ dài khác nhau và ý nghĩa khác nhau. Nhóm các mục theo cột cần một lượt đi qua chúng.">rowPtrs holds m + 1 offsets, colIdx holds z column numbers: different lengths and different meanings. Grouping the entries by column takes a pass over them.</Translated>,
      },
      {
        label: <Translated vi="Lưu A dưới dạng ELL: cách bố trí theo thứ tự cột của nó vốn đã là chuyển vị, nên kernel ELL tính Aᵀ·g.">Store A as ELL: its column-major layout already is the transpose, so the ELL kernel computes Aᵀ·g.</Translated>,
        why: <Translated vi="ELL xoay thứ tự lưu của hình chữ nhật đã đệm, không phải ma trận. Vị trí t·stride + r vẫn thuộc về hàng r.">ELL turns the storage order of the padded rectangle, not the matrix. Slot t·stride + r still belongs to row r.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            CSC là CSR với hàng và cột đổi chỗ (p. 418), nên các mảng CSR của A chính xác là CSC của Aᵀ. Kernel đọc
            chúng mỗi luồng một cột của Aᵀ là SpMV/CSC, với ưu điểm duy nhất của nó, <code>g</code> được nạp một lần và được hợp nhất, và
            các cái giá của nó: atomic trên đầu ra và phân kỳ do độ dài hàng. Khi tích lặp lại, lời khuyên của sách về việc
            giữ một bản sao thứ hai theo cột (p. 409) trở nên đáng giá, và việc dựng nó là đếm, scan và rải của bài tập 3, lấy khóa
            theo cột. FlexAttention của PyTorch làm điều tương tự cho lượt lan truyền ngược: nó sinh ra <code>q_indices</code>, một
            bản sao chuyển vị của block mask của nó.
          </p>
        }
      >
        <p>
          CSC is CSR with rows and columns swapped (p. 418), so A's CSR arrays are exactly the CSC of Aᵀ. The kernel that reads
          them one column of Aᵀ per thread is SpMV/CSC, with its one virtue, <code>g</code> loaded once and coalesced, and
          its costs: atomics on the output and divergence from row lengths. When the product repeats, the book's advice to
          keep a second, column-oriented copy (p. 409) pays, and building it is exercise 3's count, scan and scatter, keyed
          by column. PyTorch's FlexAttention does the same for its backward pass: it generates <code>q_indices</code>, a
          transposed copy of its block mask.
        </p>
      </Translated>
    ),
    source: 'ch. 17 §17.3, p. 409; §17.7, pp. 418–421; ch17-facts A9, D3; the backward-pass setting is ours',
    vi: { source: 'ch. 17 §17.3, p. 409; §17.7, pp. 418–421; ch17-facts A9, D3; bối cảnh lượt lan truyền ngược là của khóa học này' },
  },
  'ch17/wrap/q-csr-sectors': {
    kind: 'numeric',
    id: 'ch17/wrap/q-csr-sectors',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Trên GPU NVIDIA (warp {WARP} luồng, sector 32 byte, ch06-facts A2): các hàng đều dài {LEN} số float ở dạng CSR,
              mảng bắt đầu ở ranh giới 32 byte. Một warp {WARP} luồng chạy{' '}
              {LEN} lần lặp của nó; ở lần lặp <code>t</code>, luồng <code>l</code> đọc <code>value[{LEN}*l + t]</code>.
              Đếm theo từng lệnh, mỗi lần lặp chạm {csrIter(0).transactions} sector, tổng cộng {SECTORS.csrRequests} yêu cầu
              sector. {LEN} lần lặp đó chạm bao nhiêu sector 32 byte <em>khác nhau</em> của <code>value</code>?
            </>
          }
        >
          On an NVIDIA GPU ({WARP}-thread warps, 32-byte sectors, ch06-facts A2): uniform rows of {LEN} floats in CSR, the
          array starting on a 32-byte boundary. A warp of {WARP} threads runs its{' '}
          {LEN} iterations; in iteration <code>t</code>, thread <code>l</code> reads <code>value[{LEN}*l + t]</code>.
          Counted per instruction, each iteration touches {csrIter(0).transactions} sectors, {SECTORS.csrRequests} sector
          requests in all. How many <em>distinct</em> 32-byte sectors of <code>value</code> do those {LEN} iterations touch?
        </Translated>
      </>
    ),
    answer: SECTORS.csrDistinct,
    unit: 'sectors',
    vi: { unit: 'sector', source: 'của khóa học này, với sector 32 byte của ch06-facts A2; ch. 17 §17.3–17.4, pp. 409–413' },
    explain: (
      <Translated
        vi={
          <p>
            Hàng của mỗi luồng dài {LEN * 4} byte, hai sector, và luồng đọc mọi số float trong cả hai: {WARP} × 2 ={' '}
            {SECTORS.csrDistinct}. ELL trên cùng các hàng đó chạm {SECTORS.ellPerIter} sector mỗi lần lặp,{' '}
            tổng cộng {SECTORS.ellDistinct}. Đếm theo từng lệnh, CSR yêu cầu gấp {SECTORS.csrRequests / SECTORS.ellDistinct}{' '}
            lần; số byte phải lấy từ bộ nhớ thì như nhau. Nếu một cache giữ hai sector của mỗi luồng
            giữa các lần lặp của nó, phần lớn khoảng cách sẽ khép lại. Cả sách lẫn quy tắc hợp nhất của nhà cung cấp đều không nói một
            cache giữ được bao nhiêu, đó là lý do lab 17.1 yêu cầu bạn dự đoán CSR so với ELL trên các hàng đều rồi mới đo.
          </p>
        }
      >
        <p>
          Each thread's row is {LEN * 4} bytes, two sectors, and the thread reads every float in both: {WARP} × 2 ={' '}
          {SECTORS.csrDistinct}. ELL on the same rows touches {SECTORS.ellPerIter} sectors per iteration,{' '}
          {SECTORS.ellDistinct} in all. Counted per instruction CSR asks for {SECTORS.csrRequests / SECTORS.ellDistinct}{' '}
          times as much; the bytes that have to come from memory are the same. If a cache keeps each thread's two sectors
          between its iterations, most of the gap closes. Neither the book nor the vendor's coalescing rule says how much a
          cache keeps, which is why lab 17.1 asks you to predict CSR against ELL on uniform rows and then measure.
        </p>
      </Translated>
    ),
    source: 'ours, with the 32-byte sector of ch06-facts A2; ch. 17 §17.3–17.4, pp. 409–413',
  },
  'ch17/wrap/q-ragged': {
    kind: 'numeric',
    id: 'ch17/wrap/q-ragged',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Một batch decode chứa {RAGGED.seqs} chuỗi: {RAGGED.seqs - 1} chuỗi có {RAGGED.short} token trong cache và một chuỗi có{' '}
              {n(RAGGED.long)}. Một kernel attention đệm cache của mọi chuỗi cho dài bằng chuỗi dài nhất và đi qua mọi vị trí đã đệm. Bao nhiêu
              phần trăm số vị trí nó đi qua chứa một token thật?
            </>
          }
        >
          A decode batch holds {RAGGED.seqs} sequences: {RAGGED.seqs - 1} with {RAGGED.short} cached tokens and one with{' '}
          {n(RAGGED.long)}. An attention kernel pads every sequence's cache to the longest and walks every padded slot. What
          percentage of the slots it walks hold a real token?
        </Translated>
      </>
    ),
    answer: Math.round(RAGGED_PCT * 100) / 100,
    tol: 0.02,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            {RAGGED.seqs - 1} × {RAGGED.short} + {n(RAGGED.long)} = {n(RAGGED_REAL)} token thật trong {RAGGED.seqs} ×{' '}
            {n(RAGGED.long)} = {n(RAGGED_SLOTS)} vị trí: {RAGGED_PCT.toFixed(1)}%. Đây là ELL với một hàng dài, ví dụ
            1000 × 1000 của sách trong bộ áo khác, và các cách khắc phục của chương áp dụng được. Dừng mỗi chuỗi ở độ dài của riêng nó
            (<code>nnzPerRow</code>: việc đi qua giảm đi, lưu trữ thì không). Lưu các chuỗi nối tiếp nhau kèm độ lệch
            (<code>rowPtrs</code>). Nhóm các chuỗi có độ dài tương tự (phép sắp xếp của JDS, với một mảng chỉ số để đưa kết quả về chỗ cũ). Hoặc
            gửi chuỗi dài theo đường riêng của nó (dạng lai).
          </p>
        }
      >
        <p>
          {RAGGED.seqs - 1} × {RAGGED.short} + {n(RAGGED.long)} = {n(RAGGED_REAL)} real tokens in {RAGGED.seqs} ×{' '}
          {n(RAGGED.long)} = {n(RAGGED_SLOTS)} slots: {RAGGED_PCT.toFixed(1)}%. This is ELL with one long row, the book's
          1000 × 1000 example in other clothes, and the chapter's fixes carry over. Stop each sequence at its own length
          (<code>nnzPerRow</code>: the walking shrinks, the storage does not). Store the sequences back to back with offsets
          (<code>rowPtrs</code>). Group sequences of similar length (the JDS sort, with an index to put results back). Or
          send the long one down its own path (the hybrid).
        </p>
      </Translated>
    ),
    source: 'ours, after ch. 17 §17.4–17.6, pp. 412–417',
    vi: { source: 'của khóa học này, dựa theo ch. 17 §17.4–17.6, pp. 412–417' },
  },
  'ch17/wrap/q-moe-combine': {
    kind: 'choice',
    id: 'ch17/wrap/q-moe-combine',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Một lớp mixture-of-experts gửi mỗi token tới 2 expert hàng đầu của nó. Để kết hợp, mỗi kết quả (token, expert) có một luồng
              nhân hàng đầu ra của expert với trọng số của router và cộng nguyên tử nó, từng phần tử một, vào
              trạng thái ẩn của token, vốn đã chứa phần dư (residual). Hai lần chạy trên cùng đầu vào cho ra logits khác nhau ở
              các bit cuối. Thay đổi nào khiến phép kết hợp tái lập được chính xác đến từng bit?
            </>
          }
        >
          A mixture-of-experts layer sends each token to its top 2 experts. To combine, one thread per (token, expert) result
          multiplies the expert's output row by the router weight and atomically adds it, element by element, into the
          token's hidden state, which already holds the residual. Two runs on the same input give logits that differ in the
          last bits. Which change makes the combine reproducible bit for bit?
        </Translated>
      </>
    ),
    options: [
      {
        label: (
          <Translated vi="Cho hàng của mỗi token một chủ sở hữu duy nhất, gom hai kết quả đã nhân trọng số của nó và cộng chúng theo một thứ tự cố định, không dùng atomic.">
            Give each token's row one owner that gathers its two weighted results and adds them in a fixed order, with no atomic.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <>
            <Translated
              vi={
                <>
                  Giữ atomic, với <code>memory_order_seq_cst</code> thay vì relaxed, để mọi luồng thấy các phép cộng theo
                  cùng một thứ tự.
                </>
              }
            >
              Keep the atomic, with <code>memory_order_seq_cst</code> instead of relaxed, so all threads see the adds in
              one order.
            </Translated>
          </>
        ),
        why: <Translated vi="Một thứ tự bộ nhớ ràng buộc cách các thao tác bộ nhớ khác được nhìn thấy quanh phép cộng. Hai phép cộng vào một phần tử vẫn được thực hiện từng cái một theo thứ tự không xác định, và (r + a) + b với (r + b) + a có thể làm tròn khác nhau.">A memory order constrains how other memory operations are seen around the add. The two adds into one element still land one at a time in an undefined order, and (r + a) + b and (r + b) + a can round differently.</Translated>,
      },
      {
        label: (
          <Translated vi="Sắp xếp các cặp (token, expert) theo token trước khi kết hợp, để hai luồng của mỗi token nằm cạnh nhau.">
            Sort the (token, expert) pairs by token before the combine, so each token's two threads are neighbours.
          </Translated>
        ),
        why: (
          <Translated vi="Cách đó sắp thứ tự đầu vào của các luồng, chứ không phải thời điểm mỗi atomic được thực hiện. Các phép cộng vẫn tuần tự hóa theo bất kỳ thứ tự nào phần cứng chạy chúng.">
            That orders the threads' inputs, not the moment each atomic lands. The adds still serialise in whatever order the hardware runs them.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Cộng dồn bằng độ chính xác double với atomic double, và làm tròn về float một lần ở cuối.">Accumulate in double precision with double atomics, and round to float once at the end.</Translated>,
        why: <Translated vi="Tổng double của ba số float chỉ chính xác khi độ lớn của chúng chênh nhau trong khoảng 2^27. Chênh lệch rộng hơn, các tích được tính bằng double, hoặc nhiều expert hơn sẽ đưa việc làm tròn phụ thuộc thứ tự quay lại, và lần làm tròn cuối về float có thể mang nó vào bit cuối. Hiếm hơn không có nghĩa là tái lập được.">The double sum of three floats is exact only while their magnitudes stay within about 2^27 of each other. A wider spread, products formed in double, or more experts bring order-dependent rounding back, and the final rounding to float can carry it into the last bit. Rarer is not reproducible.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đây lại là COO đối đầu CSR. Một luồng mỗi cặp (token, expert) là COO: nhiều luồng nhắm vào một đầu ra, nên
            phép cộng phải là atomic, và các atomic trên một địa chỉ tuần tự hóa theo thứ tự không xác định, nên việc làm tròn số thực phụ thuộc
            vào lần chạy. Một chủ sở hữu mỗi hàng đầu ra là CSR: mã cố định thứ tự các phép cộng của nó. Phần dư là quan trọng: với
            top 2 và đầu ra bắt đầu bằng 0, <code>0 + a + b</code> và <code>0 + b + a</code> bằng nhau, vì phép cộng số thực
            có tính giao hoán; khi có số hạng thứ ba trong tổng thì chúng không nhất thiết bằng nhau. Các thư viện của nhà cung cấp đưa ra cùng sự đánh đổi đó cho
            SpMV: một thuật toán COO dùng atomic nhanh hơn, và một thuật toán tất định.
          </p>
        }
      >
        <p>
          This is COO against CSR again. One thread per (token, expert) pair is COO: several threads target one output, so
          the add must be atomic, and atomics on one address serialise in an undefined order, so the float rounding depends
          on the run. One owner per output row is CSR: the code fixes the order of its additions. The residual matters: with
          top 2 and an output starting at zero, <code>0 + a + b</code> and <code>0 + b + a</code> are equal, since float
          addition commutes; with a third term in the sum they need not be. The vendor libraries offer the same trade for
          SpMV: an atomic COO algorithm that is faster, and a deterministic one.
        </p>
      </Translated>
    ),
    source: 'ch. 17 §17.2, pp. 406–407; ch09-facts A3; ch10-facts A9; ch17-facts A2; the combine kernel is ours',
    vi: { source: 'ch. 17 §17.2, pp. 406–407; ch09-facts A3; ch10-facts A9; ch17-facts A2; kernel kết hợp là của khóa học này' },
  },
};

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports);
  const lab = 'ch17/lab1_spmv';
  const passing = [...backends, 'emu'].filter((b) => labStatus(reports[b], lab, 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch17/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>
        <Translated vi="Những gì cần mang theo">What to carry forward</Translated>
      </h2>
      <ul data-list="carry">
        <li>
          <Translated
            vi={
              <>
                Định dạng lưu trữ quyết định luồng (thread) nào sở hữu phần việc nào, và điều đó quyết định bạn có cần phép toán nguyên tử (atomic) hay không. Một
                luồng mỗi phần tử khác 0 thì va chạm trên đầu ra; một luồng mỗi hàng thì sở hữu nó. Atomic tốn thông lượng và không cố định
                thứ tự nào, nên kết quả số thực có thể thay đổi giữa các lần chạy.
              </>
            }
          >
            The storage format decides which thread owns which piece of work, and that decides whether you need atomics. One
            thread per non-zero collides on the output; one thread per row owns it. Atomics cost throughput and fix no
            order, so float results can change from run to run.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Các lần nạp có được hợp nhất (coalescing) hay không là tính chất của chỉ số mà các luồng kề nhau tính ra.{' '}
                <code>rowPtrs[r] + t</code> của CSR làm chúng rải rác; <code>t·stride + r</code> của ELL xếp chúng thẳng hàng. Đệm cộng với
                cách bố trí theo thứ tự cột mua được điều đó, với cái giá do hàng dài nhất quyết định.
              </>
            }
          >
            Whether loads coalesce is a property of the index neighbouring threads compute. CSR's{' '}
            <code>rowPtrs[r] + t</code> scatters them; ELL's <code>t·stride + r</code> lines them up. Padding plus a
            column-major layout buys that, at a price the longest row sets.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Số vòng lặp đến từ dữ liệu khiến các warp phân kỳ, và riêng ELL không sửa được điều đó. Cho các phần đuôi dài tràn ra
                (dạng lai), sắp xếp hàng theo độ dài (JDS, sliced ELL) hoặc chia nhóm chúng thì sửa được, mỗi cách tốn một chi phí tiền xử lý và,
                khi các hàng di chuyển, cần một mảng chỉ số để đưa kết quả về chỗ cũ.
              </>
            }
          >
            Trip counts that come from data make warps diverge, and ELL alone does not fix it. Spilling the long tails
            (hybrid), sorting rows by length (JDS, sliced ELL) or bucketing them does, each at a preprocessing cost and,
            when rows move, with an index array to put the results back.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Tiền xử lý chỉ đáng giá khi được dùng lại nhiều lần: nhiều lần lặp của bộ giải với một ma trận, hoặc trọng số được đóng gói lại một lần lúc
                nạp trước nhiều lượt lan truyền xuôi.
              </>
            }
          >
            Preprocessing pays only over reuse: many solver iterations with one matrix, or weights repacked once at load
            time before many forward passes.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                SpMV cộng dồn, <code>y = A·x + y</code>. Hãy viết <code>+=</code> trong mọi kernel và đánh dấu phần đệm bằng cột −1
                cùng một phép kiểm tra, để các định dạng kết hợp được theo thứ tự nào cũng được và một <code>x</code> vô cùng không thể lọt vào những hàng
                không dùng nó.
              </>
            }
          >
            SpMV accumulates, <code>y = A·x + y</code>. Write <code>+=</code> in every kernel and mark pads with column −1
            and a test, so that formats compose in either order and an infinite <code>x</code> cannot leak into rows that
            do not use it.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Hiệu năng phụ thuộc vào dữ liệu. Định dạng tốt nhất đi theo chỗ các phần tử khác 0 rơi vào, các mô hình đếm yêu cầu
                bỏ qua cache, và chỉ một phép đo trên thiết bị của bạn mới cho biết mỗi nước đi đáng giá bao nhiêu.
              </>
            }
          >
            Performance depends on the data. The best format follows where the non-zeros fall, the request-counting models
            ignore caches, and only a measurement on your device says how much each move is worth.
          </Translated>
        </li>
      </ul>
      <Source section="17.8" pages="421–422" />

      <h2>
        <Translated vi="Bài tập 5, có lời giải">Exercise 5, answered</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách để kernel JDS làm bài tập. Một luồng mỗi hàng đã sắp xếp, hàng dài nhất trước. Ở lần lặp{' '}
            <code>t</code>, còn <code>iterPtr[t+1] − iterPtr[t]</code> hàng có phần tử, và vì các hàng đã được sắp xếp
            nên đó là những hàng đầu tiên, nên luồng <code>r</code> tham gia khi <code>r</code> còn nhỏ hơn số đó. Phần tử của nó
            nằm ở <code>iterPtr[t] + r</code>: các luồng liên tiếp đọc các vị trí liên tiếp, như ở ELL, mà không cần đệm. Trên Hình
            17.14, <code>iterPtr = {list(J14.iterPtr)}</code>, nên số luồng làm việc ở các lần lặp 0 đến {PER_ITER.length - 1}{' '}
            là {PER_ITER.slice(0, -1).join(', ')} và {PER_ITER[PER_ITER.length - 1]}. Trang{' '}
            <a href={href('ch17', '17.5')}>17.5</a> hỏi về phép kiểm tra vòng lặp dưới dạng một câu hỏi.
          </p>
        }
      >
        <p>
          The book leaves the JDS kernel as an exercise. One thread per sorted row, rows longest first. In iteration{' '}
          <code>t</code>, <code>iterPtr[t+1] − iterPtr[t]</code> rows still have an element, and because the rows are sorted
          they are the first ones, so thread <code>r</code> takes part while <code>r</code> is below that count. Its element
          is at <code>iterPtr[t] + r</code>: consecutive threads read consecutive slots, as in ELL, with no padding. On Fig.
          17.14, <code>iterPtr = {list(J14.iterPtr)}</code>, so the threads at work in iterations 0 to {PER_ITER.length - 1}{' '}
          number {PER_ITER.slice(0, -1).join(', ')} and {PER_ITER[PER_ITER.length - 1]}. Page{' '}
          <a href={href('ch17', '17.5')}>17.5</a> asks for the loop test as a question.
        </p>
      </Translated>
      <Source section="17.6, 17.9" pages="416–417, 422" figs={tr('Figs. 17.14, 17.15', 'Hình 17.14, 17.15')} />
      <JdsKernel />

      <h2>
        <Translated vi="Bài kiểm tra tổng hợp">A mixed check</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Các câu hỏi này dựa vào cả chương và vào những tình huống mà các trang chưa cho thấy.
            {asked ? ` Đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu hỏi, tính theo câu trả lời gần nhất của bạn cho mỗi câu.` : ''}
          </p>
        }
      >
        <p>
          These draw on the whole chapter and on situations the pages did not show.
          {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Lỗi trong kernel, phép chuyển vị, phép đếm cache, batch được đệm và phép kết hợp expert là của khóa học này. Chúng
              dựa trên các trang của sách và trên các sự thật mà mỗi lời giải nêu tên.
            </p>
          }
        >
          <p>
            The defect in the kernel, the transpose, the cache count, the padded batch and the expert combine are ours. They
            rest on the book's pages and on the facts each answer names.
          </p>
        </Translated>
      </Beyond>
      <Ask q={q['ch17/wrap/q-jds-unpermuted']} label={tr('Find the defect', 'Tìm lỗi')} />
      <Ask q={q['ch17/wrap/q-transpose']} />
      <Ask q={q['ch17/wrap/q-csr-sectors']} label={tr('Work it out', 'Tự tính ra')} />
      <Ask q={q['ch17/wrap/q-ragged']} label={tr('Work it out', 'Tự tính ra')} />
      <Ask q={q['ch17/wrap/q-moe-combine']} />

      <h2>
        <Translated vi="Bài tập của sách">The book's exercises</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương kết thúc bằng năm bài tập (p. 422). Sách không in lời giải, nên các lời giải trong khóa học này là của chúng tôi.
          </p>
        }
      >
        <p>
          The chapter ends with five exercises (p. 422). The book prints no answers, so the answers in this course are ours.
        </p>
      </Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto', marginBottom: '1.2rem' }}>
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
              <td><Translated vi="Một ma trận 4 × 4 ở dạng COO, CSR, ELL và JDS">A 4 × 4 matrix in COO, CSR, ELL and JDS</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      trình khám phá định dạng ở <a href={href('ch17', '17.2')}>17.2</a> với ma trận "Bài tập 1" đã được nạp (mọi
                      thẻ), và câu hỏi về phần đệm của nó. JDS của nó là <code>row = {list(J_EX1.row)}</code>,{' '}
                      <code>iterPtr = {list(J_EX1.iterPtr)}</code>
                    </>
                  }
                >
                  the format explorer on <a href={href('ch17', '17.2')}>17.2</a> with the "Exercise 1" matrix loaded (every
                  tab), and its pad question. Its JDS is <code>row = {list(J_EX1.row)}</code>,{' '}
                  <code>iterPtr = {list(J_EX1.iterPtr)}</code>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td><Translated vi="Mỗi định dạng cần bao nhiêu số nguyên, tính từ m, n và z">How many integers each format needs, from m, n and z</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      dòng "Số đếm của bài tập 2" của trình khám phá và câu hỏi về những gì còn thiếu, ở{' '}
                      <a href={href('ch17', '17.2')}>17.2</a>
                    </>
                  }
                >
                  the explorer's "Exercise 2's count" line and the question on what is missing, on{' '}
                  <a href={href('ch17', '17.2')}>17.2</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td><Translated vi="Từ COO sang CSR bằng một histogram và một tổng tiền tố (prefix sum)">COO to CSR with a histogram and a prefix sum</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      câu hỏi về scan ở <a href={href('ch17', 'start')}>trang mở đầu</a>; hoạt hình "Định tuyến token tới
                      expert" và câu hỏi về độ lệch của nó ở <a href={href('ch17', '17.5')}>17.5</a>; bản phác thảo bên dưới
                    </>
                  }
                >
                  the scan question on the <a href={href('ch17', 'start')}>start page</a>; the animation "Routing tokens to
                  experts" and its offset question on <a href={href('ch17', '17.5')}>17.5</a>; the sketch below
                </Translated>
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td><Translated vi="Mã host cho định dạng lai ELL-COO">Host code for the hybrid ELL-COO format</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      lượt dạng lai của <a href={href('ch17', 'labs')}>lab 17.1</a>, với phần COO chạy trên device; câu hỏi về thứ tự
                      ở <a href={href('ch17', '17.2')}>17.2</a>
                    </>
                  }
                >
                  the hybrid pass of <a href={href('ch17', 'labs')}>lab 17.1</a>, with the COO part on the device; the order
                  question on <a href={href('ch17', '17.2')}>17.2</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>5</td>
              <td><Translated vi="Một kernel JDS">A JDS kernel</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      câu hỏi về phép kiểm tra vòng lặp ở <a href={href('ch17', '17.5')}>17.5</a>, và kernel ở trên
                    </>
                  }
                >
                  the loop-test question on <a href={href('ch17', '17.5')}>17.5</a>, and the kernel above
                </Translated>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <Translated
        vi={
          <p>
            Bài tập 3 trong ba lượt. Đếm số mục của mỗi hàng bằng một histogram trên <code>rowIdx</code> (chương 9).
            Lấy scan loại trừ của các số đếm và nối thêm tổng: đó là <code>rowPtrs</code> (chương 11). Rồi rải
            mỗi mục tới <code>rowPtrs[row]</code> cộng hạng của nó trong số các mục trước đó của cùng hàng, sao chép{' '}
            <code>colIdx</code> và <code>value</code>. Một con trỏ nguyên tử mỗi hàng cho hạng đó theo thứ tự không xác định, điều mà
            kernel CSR chấp nhận được (p. 408); một hạng tính từ thứ tự đầu vào giữ các mục của mọi hàng theo đúng thứ tự chúng
            đến. Bài tập 4 yêu cầu phần COO chạy trên host. Nếu kernel ELL dùng phép gán, như Hình 17.12, lượt trên host đó
            phải chạy sau nó; với <code>+=</code> thì thứ tự nào cũng được.
          </p>
        }
      >
        <p>
          Exercise 3 in three passes. Count the entries of each row with a histogram over <code>rowIdx</code> (chapter 9).
          Take the exclusive scan of the counts and append the total: that is <code>rowPtrs</code> (chapter 11). Then scatter
          each entry to <code>rowPtrs[row]</code> plus its rank among the earlier entries of the same row, copying{' '}
          <code>colIdx</code> and <code>value</code>. An atomic cursor per row gives that rank in an undefined order, which
          the CSR kernel tolerates (p. 408); a rank computed from the input order keeps every row's entries in the order they
          arrived. Exercise 4 asks for the COO part on the host. If the ELL kernel assigns, as Fig. 17.12 does, that host pass
          has to come after it; with <code>+=</code> either order works.
        </p>
      </Translated>
      <Source section="17.3, 17.9" pages="408, 410, 422" />
      <Translated
        vi={
          <p>
            Các thư viện của nhà cung cấp làm việc chuyển đổi trong một lời gọi, và SpMV riêng của CUB đã bị gỡ, kèm lời khuyên hãy dùng cuSPARSE.
          </p>
        }
      >
        <p>
          The vendor libraries do the conversion in one call, and CUB's own SpMV is gone, with the advice to use cuSPARSE.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch17-facts.md A9)', '2026-11-20 (notes/today/ch17-facts.md, mục A9)')}>
        <Translated
          vi={
            <p>
              <code>cusparseXcoo2csr</code> và <code>rocsparse_coo2csr</code>; <code>cub::DeviceSpmv</code> đã bị gỡ trong
              CCCL 3.0.
            </p>
          }
        >
          <p>
            <code>cusparseXcoo2csr</code> and <code>rocsparse_coo2csr</code>; <code>cub::DeviceSpmv</code> was removed in
            CCCL 3.0.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Những chỗ bản PDF có vẻ sai">Where the PDF looks wrong</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bản sách mà khóa học này được xây dựng từ đó có vài chỗ có vẻ sai sót trong chương 17. Chúng được liệt kê để bạn
            không bị bối rối khi đọc sách song song. Chỗ đầu tiên và chỗ cuối cùng làm thay đổi mã.
          </p>
        }
      >
        <p>
          The copy of the book this course was built from has a few apparent slips in chapter 17. They are listed so that you
          are not thrown by them when you read the book alongside. The first and the last change code.
        </p>
      </Translated>
      <ul data-list="errata">
        <li>
          <Translated
            vi={
              <>
                Hình 17.12 kết thúc bằng <code>y[row] = sum</code>, trong khi Hình 17.9 có <code>y[row] += sum</code> và phần chữ gọi
                cả hai là phép cộng dồn (pp. 409, 412). SpMV cộng dồn; mọi kernel trong khóa học này viết <code>+=</code>, và câu hỏi về
                thứ tự ở <a href={href('ch17', '17.2')}>17.2</a> cho thấy phép gán làm hỏng điều gì.
              </>
            }
          >
            Fig. 17.12 ends with <code>y[row] = sum</code>, where Fig. 17.9 has <code>y[row] += sum</code> and the text calls
            both an accumulation (pp. 409, 412). SpMV accumulates; every kernel in this course writes <code>+=</code>, and the
            order question on <a href={href('ch17', '17.2')}>17.2</a> shows what the assignment breaks.
          </Translated>
        </li>
        <li>
          <Translated vi={<>p. 413 nói chỉ số ELL "was calculated in Fig. 17.9" (đã được tính trong Hình 17.9); thực ra nó ở Hình 17.12.</>}>p. 413 says the ELL index "was calculated in Fig. 17.9"; it is in Fig. 17.12.</Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 414 nói ba phần tử khác 0 của "row 2" (hàng 2) chuyển sang COO. Hình 17.13 và các số đếm (độ rộng 5 xuống 2, 22 phần đệm xuống 3) cho thấy
                là hàng 1.
              </>
            }
          >
            p. 414 says three non-zeros of "row 2" move to COO. Fig. 17.13 and the counts (width 5 to 2, 22 pads to 3) say
            row 1.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Ví dụ 1000 × 1000 (pp. 412–413) cho các hàng khác "fewer than 10" (ít hơn 10) phần tử khác 0, điều mà theo phép tính của chúng tôi
                không thể cộng lại thành 1% khi có một hàng dài 200: 999 × 9 + 200 = 9,191, thiếu so với 10,000. Trang 17.5 giữ 1%;
                tỉ số là 19.05, sách làm tròn thành 20.
              </>
            }
          >
            The 1000 × 1000 example (pp. 412–413) gives the other rows "fewer than 10" non-zeros, which by our arithmetic
            cannot add up to 1% beside a row of 200: 999 × 9 + 200 = 9,191, short of 10,000. Page 17.5 keeps the 1%; the
            ratio is 19.05, which the book rounds to 20.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 416 sắp xếp các hàng "in increasing order" (theo thứ tự tăng dần); đoạn trước đó và Hình 17.14 sắp hàng dài nhất trước, và khóa
                học này theo cách đó.
              </>
            }
          >
            p. 416 sorts the rows "in increasing order"; the paragraph before it and Fig. 17.14 sort longest first, which this
            course follows.
          </Translated>
        </li>
        <li>
          <Translated vi={<>p. 417 mở đầu phần đánh giá JDS bằng "Now we examine the ELL format" (giờ ta xem xét định dạng ELL).</>}>p. 417 opens the evaluation of JDS with "Now we examine the ELL format".</Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 417 nói đến việc sắp lại các phần tử <code>x</code> của nghiệm. Sắp xếp hàng là sắp lại các phương trình, nên chính{' '}
                <code>y</code> mới phải quay lại qua mảng <code>row</code> (cách đọc của chúng tôi, và câu hỏi đầu tiên ở trên).
              </>
            }
          >
            p. 417 speaks of reordering the solution's <code>x</code> elements. Sorting rows reorders equations, so it is{' '}
            <code>y</code> that goes back through the <code>row</code> array (our reading, and the first question above).
          </Translated>
        </li>
        <li>
          <Translated vi={<>Chú thích của Hình 17.18 ghi "A parallel SpMV/CSR kernel"; đoạn mã là kernel CSC.</>}>Fig. 17.18's caption reads "A parallel SpMV/CSR kernel"; the listing is the CSC kernel.</Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Hình 17.18 đọc <code>rowIdxs</code> và <code>values</code>, trong khi phần chữ và Hình 17.16 nói{' '}
                <code>rowIdx</code> và <code>value</code>. Bản sao ở 17.2 dùng tên số ít.
              </>
            }
          >
            Fig. 17.18 reads <code>rowIdxs</code> and <code>values</code>, where the text and Fig. 17.16 say{' '}
            <code>rowIdx</code> and <code>value</code>. The copy on 17.2 uses the singular names.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Cụm "the disadvantages of CSR over COO" ở p. 410 nghĩa là các nhược điểm của CSR so với COO. Không phải sai sót, nhưng dễ
                đọc nhầm.
              </>
            }
          >
            p. 410's "the disadvantages of CSR over COO" means CSR's disadvantages compared with COO. Not a slip, but easy to
            misread.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Sách không bao giờ nói cột của một vị trí đệm chứa gì, dù vòng lặp qua toàn bộ độ rộng có đọc nó (p. 411). Cột
                0 cho NaN ở mọi hàng có đệm khi <code>x[0]</code> là vô cùng; khóa học này dùng −1 và một phép kiểm tra, như các
                thư viện.
              </>
            }
          >
            The book never says what a pad slot's column holds, though a loop over the full width reads it (p. 411). Column
            0 gives NaN in every padded row when <code>x[0]</code> is infinite; this course uses −1 and a test, as the
            libraries do.
          </Translated>
        </li>
      </ul>
      <Translated vi={<p>Và những gì các đoạn mã để lại cho bạn, mà một bản dựng thật cần:</p>}>
        <p>And what the listings leave to you, which a real build needs:</p>
      </Translated>
      <ul data-list="gaps">
        <li>
          <Translated
            vi={
              <>
                Không có struct <code>COOMatrix</code>, <code>CSRMatrix</code>, <code>ELLMatrix</code> hay <code>CSCMatrix</code> nào, và
                không có mã host hay mã khởi chạy. Các đoạn mã ở 17.2 định nghĩa các struct; lab truyền con trỏ thô.
              </>
            }
          >
            No <code>COOMatrix</code>, <code>CSRMatrix</code>, <code>ELLMatrix</code> or <code>CSCMatrix</code> struct, and
            no host or launch code. The snippets on 17.2 define the structs; the lab passes raw pointers.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Không có <code>#include &lt;cuda/atomic&gt;</code> cho <code>cuda::atomic_ref</code> của Hình 17.5 và 17.18.
              </>
            }
          >
            No <code>#include &lt;cuda/atomic&gt;</code> for the <code>cuda::atomic_ref</code> of Figs. 17.5 and 17.18.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Đơn vị căn lề "such as 64-bytes" (chẳng hạn 64 byte) (p. 413) không có trong tài liệu NVIDIA nào được đọc cho khóa học này; đơn vị ngày nay là
                sector 32 byte, tức một stride là bội của 8 số float.
              </>
            }
          >
            The alignment unit "such as 64-bytes" (p. 413) is in no NVIDIA document read for this course; today's unit is
            the 32-byte sector, a stride that is a multiple of 8 floats.
          </Translated>
        </li>
        <li>
          <Translated vi={<>Không có kernel JDS: bài tập 5. Kernel ở trên là của khóa học này.</>}>No JDS kernel: exercise 5. The one above is ours.</Translated>
        </li>
      </ul>
      <Aside>
        <Translated
          vi={
            <p>
              Đây là cách đọc của chúng tôi về một bản PDF (notes/ch17.md, "Unclear in the PDF"), với các điểm về nhà cung cấp được kiểm tra ngày
              2026-11-20 (notes/today/ch17-facts.md A1, A3, A6, A10, E1). Nếu bản của bạn khác, hãy tin bản của bạn.
            </p>
          }
        >
          <p>
            These are our readings of one PDF (notes/ch17.md, "Unclear in the PDF"), with the vendor points checked on
            2026-11-20 (notes/today/ch17-facts.md A1, A3, A6, A10, E1). If your copy differs, trust your copy.
          </p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Những gì chương này bỏ ra">What this chapter left out</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương này được rút gọn trong khóa học, nên đây là những gì không được đề cập và nơi tìm chúng. Phần nền tảng về bộ giải lặp,
            Conjugate Gradient, ma trận xác định dương và fill-in (pp. 402–403), có một câu ở
            trang mở đầu. Lịch sử đằng sau các tên ELLPACK và JDS (pp. 410, 416) không có ở đây. Biến thể JDS chia đoạn (p.
            417) không có kernel: nó xuất hiện dưới dạng sliced ELL trong trình khám phá ở 17.5 và dưới dạng Sliced ELL của cuSPARSE. Kernel CSC
            (Hình 17.17 và 17.18, pp. 419–420) được trình bày đã sửa và chấm điểm ở 17.2 nhưng không bao giờ chạy trong lab, và
            tích vector–ma trận cùng SpMSpV (p. 421) có một đoạn ở đó. Bạn không bao giờ viết kernel COO (Hình 17.5): nó
            là một đoạn mã ở 17.2 và một kernel cung cấp sẵn trong lượt dạng lai của lab.
          </p>
        }
      >
        <p>
          The chapter is condensed in this course, so here is what is not covered and where to find it. The iterative-solver
          background, Conjugate Gradient, positive-definite matrices and fill-ins (pp. 402–403), gets one sentence on the
          start page. The history behind the names ELLPACK and JDS (pp. 410, 416) is not here. The sectioned JDS variant (p.
          417) has no kernel: it appears as sliced ELL in the explorer on 17.5 and as cuSPARSE's Sliced ELL. The CSC kernel
          (Figs. 17.17 and 17.18, pp. 419–420) is shown corrected and scored on 17.2 but never run in a lab, and
          vector–matrix products and SpMSpV (p. 421) get one paragraph there. You never write the COO kernel (Fig. 17.5): it
          is a snippet on 17.2 and a provided kernel in the lab's hybrid pass.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Bài tập 3 được giải dưới dạng hoạt hình và câu hỏi, không viết mã, và vòng lặp host của bài tập 4 được thay bằng lượt
            trên device của lab. Bài báo của Bell và Garland (p. 421, tài liệu tham khảo 3) chỉ được nêu tên, vì các số liệu thiết bị của nó là từ
            phần cứng năm 2009. Các tuyên bố của sách rằng ma trận thực tế có hàng chục đến hàng trăm phần tử khác 0 mỗi hàng (p. 409) và rằng
            mã thưa đạt ít FLOPS hơn nhiều so với mã dày đặc (p. 422) được trình bày như của sách, điều không trang nào của nhà cung cấp được đọc cho
            khóa học này nêu ra; thay vào đó lab đo các kernel của chính bạn.
          </p>
        }
      >
        <p>
          Exercise 3 is worked as an animation and questions, not coded, and exercise 4's host loop is replaced by the lab's
          device pass. Bell and Garland's paper (p. 421, reference 3) is named only, because its device figures are from
          2009 hardware. The book's claims that real matrices have tens to hundreds of non-zeros per row (p. 409) and that
          sparse code reaches far fewer FLOPS than dense (p. 422) are presented as the book's, which no vendor page read for
          this course states; the lab measures your own kernels instead.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch17-facts.md A12, A13)', '2026-11-20 (notes/today/ch17-facts.md, mục A12, A13)')}>
        <Translated vi={<p>Cả hai khái quát: không thể kiểm chứng. Bài báo của Bell và Garland vẫn được NVIDIA Research lưu trữ.</p>}>
          <p>Both generalisations: could not verify. Bell and Garland's paper is still hosted by NVIDIA Research.</p>
        </Translated>
      </Today>
      <Translated vi={<p>Không phần nào sau này trong khóa học phụ thuộc vào những gì đã bị bỏ ra.</p>}>
        <p>Nothing later in the course depends on what was left out.</p>
      </Translated>

      <h2>
        <Translated vi="Dự án cho đến nay">The project so far</Translated>
      </h2>
      <p>
        <Translated
          vi={
            <>
              Chương này không thêm mốc dự án nào vào dự án xuyên suốt. Một bước suy luận của transformer dày đặc nhân các ma trận
              dày đặc, và không gì trên đường chính của nó chạy SpMV. Những gì chương này dạy quay lại dưới dạng các khuôn mẫu: attention mask thưa theo khối
              được lưu như ELL, và định tuyến mixture-of-experts như một phép chuyển COO sang CSR. Lab 17.1 là tùy chọn.{' '}
            </>
          }
        >
          This chapter adds no milestone to the through-line project. A dense transformer's inference step multiplies dense
          matrices, and nothing on its main path runs an SpMV. What the chapter teaches returns as patterns: block-sparse
          attention masks stored like ELL, and mixture-of-experts routing as a COO to CSR conversion. Lab 17.1 is optional.{' '}
        </Translated>
        <span data-lab171={passing.join(',') || 'none'}>
          {backends.length === 0 && !reports['emu']
            ? tr('No lab results are recorded on this machine yet.', 'Máy này chưa ghi nhận kết quả lab nào.')
            : passing.length
              ? tr(`Your own version passes on ${passing.map(where).join(' and ')} on this machine.`, `Phiên bản của bạn đạt trên ${passing.map(where).join(' và ')} trên máy này.`)
              : tr('Your own version has not passed on this machine yet.', 'Phiên bản của bạn chưa đạt trên máy này.')}
        </span>{' '}
        <Translated vi={<>Xem <a href={href('ch17', 'labs')}>lab</a>.</>}>
          See the <a href={href('ch17', 'labs')}>lab</a>.
        </Translated>
      </p>
      <Translated
        vi={
          <p>
            Chương 18 lưu đồ thị ở cùng ba định dạng đó: CSR cho các cạnh đi ra của một đỉnh, CSC cho các cạnh đi vào,
            COO cho một danh sách cạnh. Các kernel tìm kiếm theo chiều rộng của nó trông giống các kernel SpMV của chương này, và một mức của
            phép tìm kiếm có thể viết thành một SpMV cộng vài phép toán vector. Điều nó thêm vào là công việc có khối lượng và vị trí thay đổi trong khi kernel chạy. Bắt đầu
            ở <a href={href('ch18', 'start')}>chương 18</a>.
          </p>
        }
      >
        <p>
          Chapter 18 stores graphs in the same three formats: CSR for a vertex's outgoing edges, CSC for its incoming ones,
          COO for an edge list. Its breadth-first search kernels look like this chapter's SpMV kernels, and one level of the
          search can be written as an SpMV plus a few vector operations. What it adds is work whose amount and place change while the kernel runs. Start
          at <a href={href('ch18', 'start')}>chapter 18</a>.
        </p>
      </Translated>
      <Source section="18.1, 18.4" pages="427, 438" />
    </>
  );
}
