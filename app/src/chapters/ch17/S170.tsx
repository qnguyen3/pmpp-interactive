import { Ask, Predict, type Question } from '../../components/Ask';
import { Beyond, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { fmt } from '../../lib/reports';
import { warpAccess } from '../../sim/coalesce';
import { FIG171, countByRow, exclusiveScan, fig171Dense, float32Sums } from '../../sim/sparseroute';
import { breakEvenDensity, bytesPerPosition, warpStats } from '../../sim/sparsescale';

// The start page of chapter 17: what SpMV computes, what the course keeps of the chapter,
// the five criteria, a question about bytes before any format, a check on chapters 4, 6,
// 9 and 11 and appendix A, and a map. Every number below comes from src/sim/sparseroute.ts,
// src/sim/sparsescale.ts or src/sim/coalesce.ts.

const DENSE = fig171Dense();
const LENGTHS = countByRow(FIG171.rowIdx, FIG171.rows); // [2, 3, 2, 1]
const ROWPTRS = exclusiveScan(LENGTHS); // [0, 2, 5, 7, 8]
const LOOP = warpStats(LENGTHS, LENGTHS.length); // 8 useful of 12
const LOOP_PCT = Math.round(LOOP.efficiency * 100); // 67
const lanes = (stride: number) => Array.from({ length: 32 }, (_, t) => stride * t * 4);
const STRIDE16 = warpAccess(lanes(16), 4, 32).transactions; // 32
const UNIT = warpAccess(lanes(1), 4, 32).transactions; // 4
const ORDER = float32Sums([1e8, 1, -1e8]); // [0, 1]
const SPACING = Math.pow(2, Math.floor(Math.log2(1e8)) - 23); // float32 spacing at 1e8: 8

// The hook: fp16 pruned to 10% in CSR (fp16 values, int32 indices) against dense fp16 and dense 4-bit.
const HOOK_COLS = 4096;
const PRUNED = bytesPerPosition(0.1, 2, 4, Infinity); // 0.6
const PTR_SHARE = bytesPerPosition(0.1, 2, 4, HOOK_COLS) - PRUNED; // 4 / 4096
const FEWER = 2 / PRUNED; // 3.3
const BREAK_EVEN = breakEvenDensity(0.5, 2, 2) * 100; // 12.5
const HALF = bytesPerPosition(0.5, 2, 2, Infinity); // 2 bytes at half the weights kept
const list = (xs: readonly number[]) => '[' + xs.join(', ') + ']';

export const questionsS170: Record<string, Question> = {
  'ch17/start/q-breakeven': {
    kind: 'numeric',
    id: 'ch17/start/q-breakeven',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Một ma trận trọng số dày đặc được lượng tử hóa xuống 4 bit đọc 0.5 byte mỗi trọng số. Thay vào đó, hãy lưu chính ma trận ấy sau khi tỉa, ở dạng
              CSR với giá trị 16 bit và chỉ số cột 16 bit, và bỏ qua các con trỏ hàng. Dưới mật độ nào, tính bằng phần trăm
              số trọng số được giữ, thì CSR đọc ít byte ma trận hơn bản dày đặc 4 bit?
            </>
          }
        >
          A dense weight matrix quantised to 4 bits reads 0.5 bytes per weight. Store the same matrix pruned instead, in
          CSR with 16-bit values and 16-bit column indices, and leave out the row pointers. Below what density, in percent
          of weights kept, does CSR read fewer bytes of matrix than the 4-bit dense one?
        </Translated>
      </>
    ),
    answer: BREAK_EVEN,
    tol: 0.005,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            Mỗi trọng số được giữ tốn 2 + 2 = 4 byte, nên CSR đọc 4·d byte mỗi vị trí so với 0.5: d = 0.5 / 4 ={' '}
            {fmt(BREAK_EVEN)}%. Tỉa mà giữ một nửa số trọng số thì đọc {fmt(HALF)} byte mỗi vị trí, gấp bốn lần ma trận
            đã lượng tử hóa. Với trọng số dày đặc đã lượng tử hóa, việc tỉa phải cực đoan thì một định dạng có chỉ số mới thắng về byte,
            mà byte lại là phần của lập luận có lợi cho ma trận thưa: trang <a href={href('ch17', '17.5')}>17.5</a> bổ sung
            điều sách nói về phân ô (tiling), và trình khám phá của trang đó cho phép bạn thay đổi số byte mỗi giá trị.
          </p>
        }
      >
        <p>
          Every kept weight costs 2 + 2 = 4 bytes, so CSR reads 4·d bytes per position against 0.5: d = 0.5 / 4 ={' '}
          {fmt(BREAK_EVEN)}%. Pruning that keeps half the weights reads {fmt(HALF)} bytes per position, four times the
          quantised matrix. With quantised dense weights, pruning has to be extreme before an index format wins on bytes,
          and bytes are the part of the argument that favours sparse: page <a href={href('ch17', '17.5')}>17.5</a> adds
          what the book says about tiling, and its explorer lets you change the bytes per value.
        </p>
      </Translated>
    ),
    source: 'ours, after ch. 17 §17.8, p. 422',
    vi: { source: 'của khóa học này, dựa theo ch. 17 §17.8, p. 422' },
  },
  'ch17/start/q-warp-loop': {
    kind: 'numeric',
    id: 'ch17/start/q-warp-loop',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Bốn luồng (thread), mỗi luồng một hàng của ma trận ở trên, mỗi luồng lặp qua các phần tử khác 0 của hàng mình: số vòng lặp là{' '}
              {LENGTHS.join(', ')}. Chúng chạy như một nhóm, nhóm này tiếp tục phát lệnh các lần lặp cho đến khi luồng dài nhất xong, và
              một lane có luồng đã xong thì ngồi rảnh. Bao nhiêu phần trăm số lượt lane-lặp của nhóm làm việc có ích?
            </>
          }
        >
          Four threads, one per row of the matrix above, each loop over their row's non-zeros: trip counts{' '}
          {LENGTHS.join(', ')}. They run as one group that keeps issuing iterations until its longest thread is done, and
          a lane whose thread has finished sits idle. What percentage of the group's lane-iterations do useful work?
        </Translated>
      </>
    ),
    answer: LOOP_PCT,
    tol: 0.01,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            Nhóm chạy {LOOP.longest} lần lặp, bằng số vòng của luồng dài nhất, nên có {LENGTHS.length} × {LOOP.longest} ={' '}
            {LENGTHS.length * LOOP.longest} lượt lane-lặp, trong đó {LOOP.useful} lượt làm việc: {fmt(LOOP.efficiency * 100)}%. Đây
            là vòng lặp của chương 4 với số vòng lặp đến từ dữ liệu (Hình 4.10). Trong chương này, đó là bài toán cân bằng tải
            của kernel CSR, kernel giao cho mỗi luồng một hàng và để hàng quyết định luồng chạy bao lâu (trang{' '}
            <a href={href('ch17', '17.2')}>17.2</a>). Sắp xếp các hàng theo độ dài là cách khắc phục ở trang{' '}
            <a href={href('ch17', '17.5')}>17.5</a>.
          </p>
        }
      >
        <p>
          The group runs {LOOP.longest} iterations, its longest trip, so {LENGTHS.length} × {LOOP.longest} ={' '}
          {LENGTHS.length * LOOP.longest} lane-iterations, of which {LOOP.useful} work: {fmt(LOOP.efficiency * 100)}%. This
          is chapter 4's loop with trip counts that come from data (Fig. 4.10). In this chapter it is the load-balance
          problem of the CSR kernel, which gives each thread a row and lets the row decide how long the thread runs (page{' '}
          <a href={href('ch17', '17.2')}>17.2</a>). Sorting rows by length is the fix on page{' '}
          <a href={href('ch17', '17.5')}>17.5</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.5, p. 81 (Fig. 4.10); ch. 17 Fig. 17.1', // i18n-ignore: citation
  },
  'ch17/start/q-stride16': {
    kind: 'numeric',
    id: 'ch17/start/q-stride16',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Trên GPU NVIDIA, một lần nạp của warp được phục vụ theo các đoạn (segment) 32 byte. 32 lane của một warp mỗi lane nạp một số float 4 byte,
              lane <code>t</code> nạp phần tử ở chỉ số <code>16*t</code> của một mảng bắt đầu ở ranh giới đoạn. Lần nạp này chạm vào
              bao nhiêu đoạn?
            </>
          }
        >
          On an NVIDIA GPU a warp's load is served in 32-byte segments. The 32 lanes of a warp each load a 4-byte float,
          lane <code>t</code> the element at index <code>16*t</code> of an array that starts on a segment boundary. How many
          segments does this one load touch?
        </Translated>
      </>
    ),
    answer: STRIDE16,
    unit: 'segments',
    vi: { unit: 'đoạn', source: 'ch. 6 §6.1, pp. 124–132; đơn vị 32 byte của NVIDIA: ch06-facts A2' },
    explain: (
      <Translated
        vi={
          <p>
            Số float của lane <code>t</code> bắt đầu sau số float của lane <code>t − 1</code> 64 byte, nên không hai lane nào dùng chung một đoạn:{' '}
            {STRIDE16} đoạn, mỗi đoạn mang 4 byte có ích trong 32 byte. Chỉ số <code>t</code> sẽ chạm {UNIT} đoạn. Đây là
            lần lặp đầu tiên của kernel CSR khi mọi hàng có 16 phần tử khác 0: luồng <code>t</code> đọc{' '}
            <code>value[16*t]</code> (Hình 17.8). Cách bố trí theo thứ tự cột (column-major) của ELL biến chính các lần đọc đó thành chỉ số <code>t</code>{' '}
            (trang <a href={href('ch17', '17.2')}>17.2</a>).
          </p>
        }
      >
        <p>
          Lane <code>t</code>'s float starts 64 bytes after lane <code>t − 1</code>'s, so no two lanes share a segment:{' '}
          {STRIDE16} segments, each delivering 4 useful bytes of its 32. Index <code>t</code> would touch {UNIT}. This is the
          CSR kernel's first iteration when every row has 16 non-zeros: thread <code>t</code> reads{' '}
          <code>value[16*t]</code> (Fig. 17.8). ELL's column-major layout turns the same reads into index <code>t</code>{' '}
          (page <a href={href('ch17', '17.2')}>17.2</a>).
        </p>
      </Translated>
    ),
    source: "ch. 6 §6.1, pp. 124–132; NVIDIA's 32-byte unit: ch06-facts A2",
  },
  'ch17/start/q-lost-update': {
    kind: 'choice',
    id: 'ch17/start/q-lost-update',
    prompt: (
      <>
        <Translated
          vi={
            <>
              <code>y[0]</code> bắt đầu bằng 0. Hai luồng mỗi luồng thực thi một <code>y[0] += v</code> thường, một luồng với{' '}
              <code>v = 1</code> và một luồng với <code>v = 7</code>, không có atomic và không có rào chắn (barrier). GPU thực sự có thể cho ra những giá trị cuối nào của{' '}
              <code>y[0]</code>?
            </>
          }
        >
          <code>y[0]</code> starts at 0. Two threads each execute a plain <code>y[0] += v</code>, one with{' '}
          <code>v = 1</code> and one with <code>v = 7</code>, with no atomic and no barrier. Which final values of{' '}
          <code>y[0]</code> can a GPU actually produce?
        </Translated>
      </>
    ),
    options: [
      { label: <Translated vi="1, 7 hoặc 8, tùy cách các lệnh xen kẽ nhau.">1, 7 or 8, depending on the interleaving.</Translated>, correct: true },
      { label: <Translated vi="Lần nào cũng là 8: mỗi += là một lệnh không thể chia cắt.">8 every time: each += is one indivisible instruction.</Translated>, why: <Translated vi="Nó là một lần nạp, một phép cộng và một lần ghi. Luồng kia có thể chạy xen giữa chúng.">It is a load, an add and a store. The other thread can run between them.</Translated> },
      { label: <Translated vi="8, hoặc 0 khi hai lần ghi va chạm nhau.">8, or 0 when the two stores collide.</Translated>, why: <Translated vi="Mỗi luồng ghi một tổng do nó tính ra, 1, 7 hoặc 8. Một va chạm làm mất một trong hai, không phải cả hai.">Each thread stores a sum it computed, 1, 7 or 8. A collision loses one of them, not both.</Translated> },
      { label: <Translated vi="Bất kỳ giá trị nào, vì race là hành vi không xác định.">Any value at all, since a race is undefined.</Translated>, why: <Translated vi="Về hình thức, trong C++ và CUDA, race condition (tranh chấp dữ liệu) là hành vi không xác định. Trên phần cứng, mỗi lần ghi viết trọn một số float do luồng của nó tính ra, nên thứ bạn quan sát được là 1, 7 hoặc 8.">Formally, in C++ and CUDA, a data race is undefined behaviour. On the hardware, each store writes a whole float its thread computed, so what you observe is 1, 7 or 8.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi luồng đọc <code>y[0]</code>, cộng, rồi ghi lại. Nếu cả hai đọc 0 trước khi bên nào ghi, lần ghi sau
            thắng và để lại 1 hoặc 7; nếu một luồng xong trước khi luồng kia đọc, kết quả là 8. Đó là điều phần cứng làm; ngôn ngữ thì
            không hứa gì, vì race condition là hành vi không xác định trong C++ và trong mô hình bộ nhớ của CUDA. Đây là hai phần tử khác 0 của hàng 0 trong ma trận
            ở trên, nhân với một <code>x</code> toàn số 1. Một luồng mỗi phần tử khác 0 (COO) đặt cả hai lên <code>y[0]</code>, nên
            kernel COO của sách cộng bằng một phép toán nguyên tử (atomic) (Hình 17.5); một luồng mỗi hàng (CSR) sở hữu <code>y[0]</code> và không cần
            atomic nào (trang <a href={href('ch17', '17.2')}>17.2</a>).
          </p>
        }
      >
        <p>
          Each thread reads <code>y[0]</code>, adds, and writes back. If both read 0 before either writes, the later write
          wins and leaves 1 or 7; if one finishes before the other reads, 8. That is what the hardware does; the language
          promises nothing, since a data race is undefined behaviour in C++ and in CUDA's memory model. These are row 0's two non-zeros of the matrix
          above, multiplied by an <code>x</code> of ones. One thread per non-zero (COO) puts both on <code>y[0]</code>, so
          the book's COO kernel adds with an atomic (Fig. 17.5); one thread per row (CSR) owns <code>y[0]</code> and needs
          none (page <a href={href('ch17', '17.2')}>17.2</a>).
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.2, pp. 203–209; ch. 17 §17.2, pp. 404–407', // i18n-ignore: citation
  },
  'ch17/start/q-order': {
    kind: 'choice',
    id: 'ch17/start/q-order',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Ba tích của một hàng là 1e8, 1 và −1e8. Ba luồng cộng chúng vào <code>y = 0</code> bằng các phép cộng nguyên tử
              float32, theo bất kỳ thứ tự nào mà phần cứng tuần tự hóa chúng. Những kết quả nào có thể xuất hiện?
            </>
          }
        >
          One row's three products are 1e8, 1 and −1e8. Three threads add them into <code>y = 0</code> with float32 atomic
          adds, in whatever order the hardware serialises them. Which results can appear?
        </Translated>
      </>
    ),
    options: [
      { label: <Translated vi="Lần chạy nào cũng là 1, vì mỗi phép cộng nguyên tử đều chính xác.">1 on every run, because each atomic add is exact.</Translated>, why: <Translated vi="Atomic làm cho mỗi phép cộng không thể chia cắt. Nó không cố định thứ tự, cũng không loại bỏ việc làm tròn.">An atomic makes each add indivisible. It neither fixes the order nor removes rounding.</Translated> },
      { label: <Translated vi="0 hoặc 1, tùy thứ tự các phép cộng được thực hiện.">0 or 1, depending on the order in which the adds land.</Translated>, correct: true },
      { label: <Translated vi="Lần chạy nào cũng là 0, vì float32 không chứa được 1e8 + 1.">0 on every run, because float32 cannot hold 1e8 + 1.</Translated>, why: <Translated vi="Số 1 còn lại khi hai số hạng lớn triệt tiêu nhau trước: (1e8 − 1e8) + 1 = 1.">The 1 survives when the two large terms cancel first: (1e8 − 1e8) + 1 = 1.</Translated> },
      { label: <Translated vi="0, 1 hoặc 2, tùy việc làm tròn lên hay xuống.">0, 1 or 2, as rounding goes up or down.</Translated>, why: <Translated vi="Làm tròn về số gần nhất lần nào cũng đưa 1e8 + 1 về 1e8. Không thứ tự nào cho ra 2.">Round to nearest takes 1e8 + 1 back to 1e8 every time. No order produces 2.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Gần 1e8, các giá trị float32 liên tiếp cách nhau {SPACING}, nên 1e8 + 1 bị làm tròn về 1e8. Các thứ tự cộng số 1
            vào một số hạng lớn trước sẽ kết thúc ở 0; các thứ tự triệt tiêu hai số hạng lớn trước sẽ kết thúc ở 1. Mô hình, thử mọi
            thứ tự: {list(ORDER)}. Các atomic trên cùng một địa chỉ chạy từng cái một theo thứ tự không xác định (chương 9), nên kết quả nào
            cũng có thể xuất hiện ở bất kỳ lần chạy nào. Đây là phản đối thứ hai của sách với atomic trong SpMV (p. 407). Thư viện của NVIDIA và AMD
            đều cung cấp một thuật toán COO chậm hơn nhưng tất định vì lý do này (trang{' '}
            <a href={href('ch17', '17.5')}>17.5</a> liệt kê chúng).
          </p>
        }
      >
        <p>
          Near 1e8, consecutive float32 values are {SPACING} apart, so 1e8 + 1 rounds back to 1e8. Orders that add the 1
          to a large term first end at 0; orders that cancel the two large terms first end at 1. The model, trying every
          order: {list(ORDER)}. Atomics on one address run one at a time in an undefined order (chapter 9), so either result
          can appear on any run. This is the book's second objection to atomics in SpMV (p. 407). NVIDIA's and AMD's
          libraries both offer a slower, deterministic COO algorithm for this reason (page{' '}
          <a href={href('ch17', '17.5')}>17.5</a> lists them).
        </p>
      </Translated>
    ),
    source: 'appendix A §A.6, pp. 593–594; ch. 17 p. 407; ch17-facts A2',
    vi: { source: 'phụ lục A §A.6, pp. 593–594; ch. 17 p. 407; ch17-facts A2' },
  },
  'ch17/start/q-scan': {
    kind: 'numeric',
    id: 'ch17/start/q-scan',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Các hàng của ma trận ở trên chứa {LENGTHS.join(', ')} phần tử khác 0. Lấy scan loại trừ của {list(LENGTHS)}. Phần tử 2
              của nó là bao nhiêu?
            </>
          }
        >
          The rows of the matrix above hold {LENGTHS.join(', ')} non-zeros. Take the exclusive scan of {list(LENGTHS)}. What
          is its element 2?
        </Translated>
      </>
    ),
    answer: ROWPTRS[2],
    explain: (
      <Translated
        vi={
          <p>
            Scan loại trừ là {list(ROWPTRS.slice(0, -1))}: mỗi phần tử là tổng các số đếm đứng trước nó. Nối thêm
            tổng, {ROWPTRS[ROWPTRS.length - 1]}, và bạn có {list(ROWPTRS)}, chính xác là{' '}
            <code>rowPtrs</code> của CSR (Hình 17.7): hàng 2 bắt đầu ở {ROWPTRS[2]}. Bài tập 3 dựng CSR từ COO theo cách này: đếm theo
            hàng, scan, rải. Trang <a href={href('ch17', '17.5')}>17.5</a> chạy cùng ba lượt đó để nhóm token theo
            expert.
          </p>
        }
      >
        <p>
          The exclusive scan is {list(ROWPTRS.slice(0, -1))}: each element is the sum of the counts before it. Append the
          total, {ROWPTRS[ROWPTRS.length - 1]}, and you have {list(ROWPTRS)}, which is exactly CSR's{' '}
          <code>rowPtrs</code> (Fig. 17.7): row 2 starts at {ROWPTRS[2]}. Exercise 3 builds CSR from COO this way: count per
          row, scan, scatter. Page <a href={href('ch17', '17.5')}>17.5</a> runs the same three passes to group tokens by
          expert.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.1, pp. 252–254; ch. 17 Fig. 17.7 and exercise 3, p. 422',
    vi: { source: 'ch. 11 §11.1, pp. 252–254; ch. 17 Fig. 17.7 và bài tập 3, p. 422' },
  },
};

// ---------------------------------------------------------------- Fig. 17.1, redrawn
const C = 36;
const AX = 74;
const AY = 34;
const VX = AX + 4 * C + 34; // x
const YX = VX + C + 34; // y
const OUTX = YX + C + 48; // result y

function Vector({ x, name, label }: { x: number; name: string; label: string }) {
  return (
    <g data-vector={name}>
      <text x={x + C / 2 - 1} y={AY - 10} fontSize="13" fontWeight="600" textAnchor="middle" style={{ fill: 'var(--global)' }}>
        {label}
      </text>
      {[0, 1, 2, 3].map((i) => (
        <g key={i}>
          <rect x={x} y={AY + i * C} width={C - 3} height={C - 3} rx="2" fill="var(--global-wash)" stroke="var(--global)" />
          <text x={x + (C - 3) / 2} y={AY + i * C + 22} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
            {name}
            {i}
          </text>
        </g>
      ))}
    </g>
  );
}

function Fig171() {
  return (
    <figure style={{ margin: '0.4rem 0 1.4rem' }}>
      <svg
        viewBox={`0 0 ${OUTX + C + 6} ${AY + 4 * C + 14}`}
        role="img"
        aria-label={tr(
          `The book's 4 by 4 example: rows ${DENSE.map((r) => '[' + r.join(' ') + ']').join(', ')}, multiplied by x and added to y.`,
          `Ví dụ 4 x 4 của sách: các hàng ${DENSE.map((r) => '[' + r.join(' ') + ']').join(', ')}, nhân với x rồi cộng vào y.`,
        )}
        style={{ width: '100%', maxWidth: OUTX + C + 6, display: 'block', fontFamily: 'var(--sans)' }}
      >
        <text x={AX + 2 * C} y={AY - 10} fontSize="13" fontWeight="600" textAnchor="middle" style={{ fill: 'var(--global)' }}>
          A
        </text>
        {DENSE.map((row, r) => (
          <g key={r} data-row={r}>
            <text x={AX - 10} y={AY + r * C + 22} fontSize="12" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
              <Translated vi={<>hàng {r}</>}>row {r}</Translated>
            </text>
            {row.map((v, c) => (
              <g key={c} data-cell={`${r},${c}`} data-value={v}>
                <rect
                  x={AX + c * C}
                  y={AY + r * C}
                  width={C - 3}
                  height={C - 3}
                  rx="2"
                  fill={v ? 'var(--global-wash)' : 'var(--cell)'}
                  stroke={v ? 'var(--global)' : 'var(--rule)'}
                />
                {v ? (
                  <text x={AX + c * C + (C - 3) / 2} y={AY + r * C + 22} fontSize="14" textAnchor="middle" style={{ fill: 'var(--ink)' }}>
                    {v}
                  </text>
                ) : null}
              </g>
            ))}
          </g>
        ))}
        <text x={VX - 17} y={AY + 2 * C + 4} fontSize="18" textAnchor="middle" style={{ fill: 'var(--ink)' }}>×</text>
        <Vector x={VX} name="x" label="x" />
        <text x={YX - 17} y={AY + 2 * C + 4} fontSize="18" textAnchor="middle" style={{ fill: 'var(--ink)' }}>+</text>
        <Vector x={YX} name="y" label="y" />
        <text x={OUTX - 24} y={AY + 2 * C + 4} fontSize="18" textAnchor="middle" style={{ fill: 'var(--ink)' }}>→</text>
        <Vector x={OUTX} name="y" label={tr('new y', 'y mới')} />
      </svg>
      <figcaption className="muted" style={{ fontFamily: 'var(--sans)', fontSize: '0.8rem' }}>
        <Translated
          vi={
            <>
              Ví dụ xuyên suốt của sách (Hình 17.1), vẽ lại thành toàn bộ phép toán <code>y = A·x + y</code>. Sách tô màu
              từng phần tử khác 0; ở đây thay vào đó các hàng được gắn nhãn, vì trong khóa học này màu sắc cho biết dữ liệu nằm ở đâu. Mọi mảng
              đều mang màu của bộ nhớ toàn cục (global memory): các kernel của chương đọc tất cả chúng từ đó.
            </>
          }
        >
          The book's running example (Fig. 17.1), redrawn as the whole operation <code>y = A·x + y</code>. The book colours
          each non-zero; here rows are labelled instead, because colour in this course means where data lives. Every array
          is in the global-memory colour: the chapter's kernels read them all from there.
        </Translated>
      </figcaption>
    </figure>
  );
}

const CRITERIA: { name: string; asks: string; side: string }[] = [
  {
    get name() {
      return tr('Space efficiency', 'Hiệu quả không gian');
    },
    get asks() {
      return tr('How much memory the matrix takes.', 'Ma trận chiếm bao nhiêu bộ nhớ.');
    },
    get side() {
      return tr('compaction', 'nén gọn');
    },
  },
  {
    get name() {
      return tr('Flexibility', 'Tính linh hoạt');
    },
    get asks() {
      return tr('How easily a non-zero can be added or removed.', 'Thêm hoặc bớt một phần tử khác 0 dễ đến mức nào.');
    },
    get side() {
      return tr('neither', 'không bên nào');
    },
  },
  {
    get name() {
      return tr('Accessibility', 'Khả năng truy cập');
    },
    get asks() {
      return tr('Which questions are cheap: all non-zeros of a given row; the row and column of a given non-zero.', 'Câu hỏi nào rẻ: mọi phần tử khác 0 của một hàng cho trước; hàng và cột của một phần tử khác 0 cho trước.');
    },
    get side() {
      return tr('neither', 'không bên nào');
    },
  },
  {
    get name() {
      return tr('Memory-access efficiency', 'Hiệu quả truy cập bộ nhớ');
    },
    get asks() {
      return tr("Whether the kernel's loads coalesce.", 'Các lần nạp của kernel có được hợp nhất (coalescing) hay không.');
    },
    get side() {
      return tr('regularity', 'sự đều đặn');
    },
  },
  {
    get name() {
      return tr('Load balance', 'Cân bằng tải');
    },
    get asks() {
      return tr('Whether threads get equal work, so that warps do not diverge.', 'Các luồng có nhận khối lượng việc bằng nhau không, để các warp không phân kỳ.');
    },
    get side() {
      return tr('regularity', 'sự đều đặn');
    },
  },
];

export function S170() {
  return (
    <>
      <Translated
        vi={
          <p>
            Chương này nói về một phép tính. Một ma trận <i>A</i> phần lớn là số 0 nhân với một vector dày đặc{' '}
            <i>x</i>, và tích được cộng vào một vector dày đặc <i>y</i>: <code>y = A·x + y</code>. Sách gọi nó là SpMV,
            phép nhân ma trận thưa với vector rồi cộng dồn, và việc cộng dồn là quan trọng: các kernel của nó cộng vào{' '}
            <code>y</code> thay vì ghi đè lên nó.
          </p>
        }
      >
        <p>
          This chapter is about one computation. A matrix <i>A</i> that is mostly zeros multiplies a dense vector{' '}
          <i>x</i>, and the product is added to a dense vector <i>y</i>: <code>y = A·x + y</code>. The book calls it SpMV,
          sparse matrix–vector multiplication and accumulation, and the accumulate matters: its kernels add into{' '}
          <code>y</code> rather than overwrite it.
        </p>
      </Translated>
      <Source section="17.1" pages="402–403" />
      <Fig171 />
      <Translated
        vi={
          <p>
            Lý do sách dành cả một chương là các bộ giải lặp cho hệ phương trình tuyến tính lớn. Mỗi lần lặp tính{' '}
            <code>A·x + y</code> với một phỏng đoán mới cho <code>x</code> và cùng một <code>A</code> như mọi lần lặp khác.
            Hãy nhớ sự lặp lại đó: chính nó khiến việc chuyển đổi hoặc sắp xếp <code>A</code> một lần trước
            tích đầu tiên trở nên đáng chi phí.
          </p>
        }
      >
        <p>
          The book's reason for a whole chapter is iterative solvers for large linear systems. Each iteration evaluates{' '}
          <code>A·x + y</code> with a new guess for <code>x</code> and the same <code>A</code> as every other iteration.
          Keep that repetition in mind: it is what makes it affordable to convert or sort <code>A</code> once before the
          first product.
        </p>
      </Translated>
      <Source section="17.1" pages="402–403" />
      <Translated
        vi={
          <p>
            Các thư viện ngày nay giữ phần cộng dồn và thêm hai số vô hướng: chúng tính <code>y = α·op(A)·x + β·y</code>, trong đó{' '}
            <code>op(A)</code> là <code>A</code> hoặc chuyển vị của nó. SpMV của sách là α = 1, β = 1; một tích thuần là β = 0.
          </p>
        }
      >
        <p>
          Today's libraries keep the accumulate and add two scalars: they compute <code>y = α·op(A)·x + β·y</code>, where{' '}
          <code>op(A)</code> is <code>A</code> or its transpose. The book's SpMV is α = 1, β = 1; a plain product is β = 0.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch17-facts A4, A10)', '2026-11-20 (ch17-facts, mục A4, A10)')}>
        <Translated
          vi={
            <p>
              <code>cusparseSpMV</code> (cuSPARSE 13.5), <code>rocsparse_spmv</code> và <code>hipsparseSpMV</code> đều dùng
              công thức này. Hàm của tiêu chuẩn, Sparse BLAS <code>usmv</code>, có <code>y</code> vừa vào vừa ra và không có β.
            </p>
          }
        >
          <p>
            <code>cusparseSpMV</code> (cuSPARSE 13.5), <code>rocsparse_spmv</code> and <code>hipsparseSpMV</code> all use
            this formula. The standard's routine, Sparse BLAS <code>usmv</code>, has <code>y</code> in-out and no β.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Khóa học này giữ lại gì">What this course keeps</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Suy luận LLM dày đặc hiếm khi chạy SpMV. Nhưng các mẫu thưa có xuất hiện quanh nó: attention mask bỏ qua cả khối,
            các lớp mixture-of-experts gửi mỗi token tới vài expert, trọng số đã tỉa. Điều mang theo được từ
            chương này không phải danh mục định dạng của nó mà là sáu kỹ thuật bạn sẽ dùng lại trong những kernel chẳng liên quan gì
            đến bộ giải tuyến tính:
          </p>
        }
      >
        <p>
          Dense LLM inference rarely runs an SpMV. Sparse patterns do turn up around it: attention masks that skip blocks,
          mixture-of-experts layers that send each token to a few experts, pruned weights. What carries over from the
          chapter is less its catalogue of formats than six techniques you will reuse in kernels that have nothing to do
          with linear solvers:
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cách đặt vấn đề theo LLM là của khóa học này. Mã đứng sau nó, trong PyTorch, llama.cpp và MLX, nằm ở ch17-facts D1 đến D7, và trang
              17.5 đi qua một trường hợp.
            </p>
          }
        >
          <p>
            The LLM framing is ours. The code behind it, in PyTorch, llama.cpp and MLX, is in ch17-facts D1 to D7, and page
            17.5 walks through one case.
          </p>
        </Translated>
      </Beyond>
      <ul>
        <li>
          <Translated
            vi={
              <>
                <b>Cách ánh xạ luồng vào công việc quyết định bạn có cần atomic hay không.</b> Một luồng mỗi phần tử khác 0 thì va chạm trên
                đầu ra; một luồng mỗi hàng thì sở hữu nó. <span className="muted">(pp. 405–409)</span>
              </>
            }
          >
            <b>The thread-to-work mapping decides whether you need atomics.</b> One thread per non-zero collides on the
            output; one thread per row owns it. <span className="muted">(pp. 405–409)</span>
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Số vòng lặp đến từ dữ liệu gây ra phân kỳ luồng (divergence).</b> Khi mỗi luồng lặp qua hàng của riêng nó, các luồng của
                một warp xong vào những thời điểm khác nhau. <span className="muted">(p. 410)</span>
              </>
            }
          >
            <b>Trip counts that come from data cause divergence.</b> When each thread loops over its own row, the threads of
            a warp finish at different times. <span className="muted">(p. 410)</span>
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Đệm cộng với cách bố trí chuyển vị mua được việc hợp nhất truy cập bộ nhớ (coalescing)</b>, với cái giá do hàng dài nhất quyết định.{' '}
                <span className="muted">(pp. 410–413)</span>
              </>
            }
          >
            <b>Padding plus a transposed layout buys coalescing</b>, at a price set by the longest row.{' '}
            <span className="muted">(pp. 410–413)</span>
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Tách phần thân đều đặn khỏi phần đuôi bất thường</b>, và chạy mỗi phần bằng kernel hợp với nó.{' '}
                <span className="muted">(pp. 414–415)</span>
              </>
            }
          >
            <b>Split a regular bulk from an irregular tail</b>, and run each with the kernel that suits it.{' '}
            <span className="muted">(pp. 414–415)</span>
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Sắp xếp hoặc chia nhóm theo độ dài, và giữ hoán vị ngược</b> để đưa kết quả về chỗ cũ.{' '}
                <span className="muted">(pp. 416–417)</span>
              </>
            }
          >
            <b>Sort or bucket by length, and keep the inverse permutation</b> to put the results back.{' '}
            <span className="muted">(pp. 416–417)</span>
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Khấu hao phần tiền xử lý qua nhiều lần dùng lại</b>: một lần chuyển đổi tốn bằng vài tích sẽ đáng giá khi cùng ma trận đó được
                dùng nhiều lần. <span className="muted">(pp. 414–415, 417)</span>
              </>
            }
          >
            <b>Amortise preprocessing over reuse</b>: a conversion that costs several products pays when the same matrix is
            used many times. <span className="muted">(pp. 414–415, 417)</span>
          </Translated>
        </li>
      </ul>
      <Translated
        vi={
          <p>
            Những gì bị bỏ ra: phần nền tảng về bộ giải (Conjugate Gradient, ma trận xác định dương, và các fill-in khiến
            việc nghịch đảo một ma trận thưa lớn là không khả thi) và lịch sử đằng sau các tên ELL và JDS. Không điều nào trong đó thay đổi
            cách viết một kernel.
          </p>
        }
      >
        <p>
          What is left out: the solver background (Conjugate Gradient, positive-definite matrices, and the fill-ins that
          make inverting a large sparse matrix impractical) and the history behind the names ELL and JDS. None of it changes
          how a kernel is written.
        </p>
      </Translated>
      <Source section="17.1" pages="402" />

      <h2>
        <Translated vi="Nén gọn đối đầu sự đều đặn">Compaction against regularity</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mọi định dạng thưa đều bỏ các số 0, điều sách gọi là nén gọn (compaction), và trả giá bằng sự bất thường: dữ liệu không
            còn một hình dạng mà luồng có thể tính toán để tìm đường đi trong đó. Trên GPU, sự bất thường có ba cái giá: băng thông bộ nhớ
            bị bỏ phí, phân kỳ điều khiển, và mất cân bằng tải. Các định dạng nằm ở những điểm khác nhau giữa nén thật chặt và
            giữ sự đều đặn, và định dạng nào nhanh nhất phụ thuộc vào chỗ các phần tử khác 0 rơi vào.
          </p>
        }
      >
        <p>
          Every sparse format drops zeros, which the book calls compaction, and pays for it with irregularity: the data no
          longer has a shape a thread can compute its way around. On a GPU, irregularity has three costs: memory bandwidth
          left unused, control divergence, and load imbalance. Formats sit at different points between compacting hard and
          staying regular, and which one is fastest depends on where the non-zeros fall.
        </p>
      </Translated>
      <Source pages="401" />
      <Translated
        vi={
          <p>
            Sách chấm điểm mọi định dạng theo cùng năm tiêu chí. Chúng là các cột của trình khám phá định dạng ở trang{' '}
            <a href={href('ch17', '17.2')}>17.2</a>, và đáng giữ lại như những câu hỏi đặt ra cho bất kỳ cách bố trí dữ liệu nào bạn
            thiết kế cho một kernel.
          </p>
        }
      >
        <p>
          The book scores every format against the same five criteria. They are the columns of the format explorer on page{' '}
          <a href={href('ch17', '17.2')}>17.2</a>, and they are worth keeping as questions to ask of any data layout you
          design for a kernel.
        </p>
      </Translated>
      <Source section="17.1" pages="403–404" />
      <table className="data" data-criteria>
        <thead>
          <tr>
            <th><Translated vi="Tiêu chí">Criterion</Translated></th>
            <th><Translated vi="Nó hỏi gì">What it asks</Translated></th>
            <th><Translated vi="Phía của sự đánh đổi">Side of the trade</Translated></th>
          </tr>
        </thead>
        <tbody>
          {CRITERIA.map((c) => (
            <tr key={c.name}>
              <td>{c.name}</td>
              <td>{c.asks}</td>
              <td>{c.side}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>
        <Translated vi="Một câu hỏi trước mọi định dạng">A question before any format</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một phép nhân ma trận với vector đọc mỗi trọng số một lần. Tỉa thì bỏ bớt trọng số; lượng tử hóa thì thu nhỏ chúng. Cả hai đều cắt giảm số byte
            phải đọc, và bạn đã biết lượng tử hóa. Hãy so sánh hai cách chỉ dựa trên số byte trước khi bạn gặp bất kỳ định dạng thưa nào.
          </p>
        }
      >
        <p>
          A matrix-vector product reads every weight once. Pruning drops weights; quantising shrinks them. Both cut the bytes
          read, and you already know quantisation. Compare the two on bytes alone before you meet a single sparse format.
        </p>
      </Translated>
      <Predict
        id="ch17/start/p-bytes"
        prompt={
          <>
            <Translated
              vi={
                <>
                  Một ma trận trọng số giá trị fp16 được tỉa để giữ 10% số trọng số. Các trọng số được giữ lưu ở dạng CSR: mỗi trọng số một
                  giá trị fp16 và một chỉ số cột 32 bit, cộng thêm một con trỏ hàng 32 bit cho mỗi hàng. Một phép nhân ma trận với vector đọc bao nhiêu byte ma trận
                  cho mỗi vị trí của ma trận gốc? fp16 dày đặc đọc 2.
                </>
              }
            >
              A weight matrix of fp16 values is pruned to keep 10% of its weights. The kept weights are stored in CSR: an
              fp16 value and a 32-bit column index each, plus one 32-bit row pointer per row. How many bytes of matrix does
              a matrix-vector product read per position of the original matrix? Dense fp16 reads 2.
            </Translated>
          </>
        }
        unit={tr('bytes', 'byte')}
      >
        {(v) => (
          <div data-reveal="ch17/start/p-bytes">
            <p>
              <Translated
                vi={
                  <>
                    Mô hình cho <b data-out="pruned">{fmt(PRUNED)}</b>: 0.1 × (2 + 4). Con trỏ hàng thêm 4 byte mỗi hàng,
                    tức {fmt(PTR_SHARE)} mỗi vị trí trên một hàng {HOOK_COLS} cột, không đáng kể. Tức là ít hơn{' '}
                    <b data-out="fewer">{fmt(FEWER, 2)}</b> lần số byte so với fp16 dày đặc.{' '}
                    {Math.abs(v - PRUNED) < 0.05 ? 'Bạn đã đúng.' : 'Mỗi trọng số được giữ phải trả cho chỉ số của nó cũng như cho giá trị của nó.'}
                  </>
                }
              >
                The model says <b data-out="pruned">{fmt(PRUNED)}</b>: 0.1 × (2 + 4). The row pointer adds 4 bytes per row,
                or {fmt(PTR_SHARE)} per position on a row of {HOOK_COLS} columns, which is noise. That is{' '}
                <b data-out="fewer">{fmt(FEWER, 2)}</b> times fewer bytes than dense fp16.{' '}
                {Math.abs(v - PRUNED) < 0.05 ? 'You had it.' : 'Every kept weight pays for its index as well as its value.'}
              </Translated>
            </p>
            <Translated
              vi={
                <p>
                  Giờ hãy lượng tử hóa thay vì tỉa: một ma trận dày đặc gồm trọng số 4 bit đọc <b>0.5</b> byte mỗi vị trí và hoàn toàn không có
                  chỉ số nào. Tỉa xuống 10% thua nó, vì mỗi trọng số được giữ kéo theo một chỉ số 4 byte. Ngay cả với
                  chỉ số 16 bit, câu hỏi bên dưới cho thấy một ma trận phải thưa đến mức nào thì CSR mới thắng.
                </p>
              }
            >
              <p>
                Now quantise instead of pruning: a dense matrix of 4-bit weights reads <b>0.5</b> bytes per position and no
                indices at all. Pruning to 10% loses to it, because every kept weight drags a 4-byte index along. Even with
                16-bit indices, the question below shows how sparse a matrix has to be before CSR wins.
              </p>
            </Translated>
          </div>
        )}
      </Predict>
      <Ask q={questionsS170['ch17/start/q-breakeven']} label={tr('Before any format', 'Trước mọi định dạng')} />
      <Translated
        vi={
          <p>
            Sách kết thúc chương bằng cùng lời cảnh báo ấy nhìn từ phía bên kia. Một ma trận dày đặc có chỉ số ngầm định và
            dễ phân ô (tiling) theo hệ thống phân cấp bộ nhớ; một định dạng thưa lưu chỉ số tường minh phải được đọc, và khó
            phân ô. Vì vậy một định dạng thưa chỉ đáng giá khi ma trận rất thưa. Quy tắc kinh nghiệm của sách cho các
            ma trận thực tế là hàng nghìn đến hàng triệu hàng, mỗi hàng có hàng chục đến hàng trăm phần tử khác 0 (p. 409). Với một ma trận vuông,
            đó là mật độ tối đa 10% ở đầu nhỏ, và thấp hơn hẳn 1% khi có hàng chục nghìn hàng:
            thưa hơn mức các mô hình đã tỉa thường đạt.
          </p>
        }
      >
        <p>
          The book ends the chapter with the same warning from the other side. A dense matrix has implicit indices and is
          easy to tile for the memory hierarchy; a sparse format stores explicit indices that must be read, and is hard to
          tile. So a sparse format pays off only when the matrix is very sparse. The book's rule of thumb for real
          matrices is thousands to millions of rows with tens to hundreds of non-zeros each (p. 409). For a square matrix
          that is a density of at most 10%, at the small end, and well below 1% once there are tens of thousands of rows:
          sparser than pruned models usually are.
        </p>
      </Translated>
      <Source section="17.3, 17.8" pages="409, 422" />
      <Translated
        vi={
          <p>
            Tài liệu của PyTorch cũng nói vậy: các định dạng thưa "are not always advantageous" (không phải lúc nào cũng có lợi), và khi thử
            một định dạng, bạn "might find your execution time to increase rather than decrease" (có thể thấy thời gian chạy tăng lên thay vì giảm đi).
          </p>
        }
      >
        <p>
          PyTorch's documentation says the same: sparse formats "are not always advantageous", and trying
          one you "might find your execution time to increase rather than decrease".
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch17-facts D1)', '2026-11-20 (ch17-facts, mục D1)')}>
        <Translated vi={<p>Trang về sparse của PyTorch 2.15, trang này cũng gọi tính thưa bán cấu trúc (2:4) của nó là một bản thử nghiệm (prototype).</p>}>
          <p>PyTorch 2.15's sparse page, which also calls its semi-structured (2:4) sparsity a prototype.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Trọng số đã tỉa và đã lượng tử hóa là nơi câu hỏi này gặp công việc của bạn. Khi một định dạng thưa thực sự thắng về byte, phần
            còn lại của chương bàn về việc một kernel có biến được số byte ấy thành tốc độ hay không.
          </p>
        }
      >
        <p>
          Pruned and quantised weights are where this question meets your work. When a sparse format does win on bytes, the
          rest of the chapter is about whether a kernel can turn those bytes into speed.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cầu nối sang trọng số đã tỉa và đã lượng tử hóa, phép tính của câu hỏi mở đầu, và mật độ suy ra từ quy tắc kinh nghiệm
              của sách (giả định ma trận vuông) là của khóa học này.
            </p>
          }
        >
          <p>
            The bridge to pruned and quantised weights, the arithmetic of the hook, and the density read from the book's rule
            of thumb (it assumes a square matrix) are ours.
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Trước khi bắt đầu">Before you start</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Năm câu hỏi về những gì chương này lấy từ các chương 4, 6, 9, 11 và phụ lục A, theo thứ tự chương dùng đến chúng.
            Mỗi câu sẽ quay lại trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Five questions on what this chapter takes from chapters 4, 6, 9 and 11 and appendix A, in the order it uses them.
          Each one comes back in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={questionsS170['ch17/start/q-warp-loop']} label={tr('From chapter 4', 'Từ chương 4')} />
      <Ask q={questionsS170['ch17/start/q-stride16']} label={tr('From chapter 6', 'Từ chương 6')} />
      <Ask q={questionsS170['ch17/start/q-lost-update']} label={tr('From chapter 9', 'Từ chương 9')} />
      <Ask q={questionsS170['ch17/start/q-order']} label={tr('From appendix A', 'Từ phụ lục A')} />
      <Ask q={questionsS170['ch17/start/q-scan']} label={tr('From chapter 11', 'Từ chương 11')} />

      <h2>
        <Translated vi="Chương này trong khóa học">The chapter in this course</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hai trang bài giảng, một lab tùy chọn và một trang tóm tắt. Trang <a href={href('ch17', '17.2')}>17.2</a> dựng COO, CSR,
            ELL và CSC từ ví dụ 4 × 4 và cho thấy mỗi định dạng quyết định kernel của nó: định dạng nào cần atomic, lần nạp nào
            được hợp nhất, luồng nào phân kỳ. Trang <a href={href('ch17', '17.5')}>17.5</a> mua sự đều đặn: đệm, cho
            các hàng dài tràn sang COO, sắp xếp hàng theo độ dài, khi nào phần tiền xử lý đó đáng giá, và những nước đi ấy xuất hiện ở đâu trong suy luận
            LLM. <a href={href('ch17', 'labs')}>Lab</a> chạy SpMV với CSR và ELL trên các hàng lệch trên chính GPU của bạn.{' '}
            <a href={href('ch17', 'wrap')}>Trang tóm tắt</a> có một bài kiểm tra tổng hợp, năm bài tập của sách và những chỗ sai sót trong bản PDF.
          </p>
        }
      >
        <p>
          Two teaching pages, an optional lab and a summary. Page <a href={href('ch17', '17.2')}>17.2</a> builds COO, CSR,
          ELL and CSC from the 4 × 4 example and shows that each format decides its kernel: which one needs atomics, which loads
          coalesce, which threads diverge. Page <a href={href('ch17', '17.5')}>17.5</a> buys regularity: padding, spilling
          long rows to COO, sorting rows by length, when that preprocessing pays, and where the same moves appear in LLM
          inference. The <a href={href('ch17', 'labs')}>lab</a> runs CSR and ELL SpMV on skewed rows on your own GPU. The{' '}
          <a href={href('ch17', 'wrap')}>summary</a> has a mixed check, the book's five exercises and the slips in the PDF.
        </p>
      </Translated>
      <Source section="17.2–17.9" pages="404–422" />
      <Translated
        vi={
          <p>
            Hai chương xây trên chương này. Chương 18 lưu đồ thị ở dạng CSR, CSC và COO, và các kernel duyệt đồ thị của nó trông giống
            các kernel SpMV này. Chương 20 kể tên mixture of experts trong số các cách mở rộng quy mô một mô hình.
          </p>
        }
      >
        <p>
          Two chapters build on this one. Chapter 18 stores graphs in CSR, CSC and COO, and its traversal kernels look like
          these SpMV kernels. Chapter 20 names mixture of experts among the ways to scale a model.
        </p>
      </Translated>
      <Source section="18.1, 18.4, 20.7" pages="427, 438, 510" />
      <Translated
        vi={
          <p>
            Lab chạy trên mọi backend, vì một luồng mỗi hàng không cần atomic và không cần gì riêng của nhà cung cấp.
          </p>
        }
      >
        <p>
          The lab runs on every backend, because one thread per row needs no atomics and nothing vendor-specific.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch17-facts B1, C1)', '2026-11-20 (ch17-facts, mục B1, C1)')}>
        <Translated
          vi={
            <p>
              Các kernel mỗi-hàng-một-luồng không cần cách viết riêng cho HIP (B1); unverified trên HIP. Trong Metal,{' '}
              <code>[[thread_position_in_grid]]</code> cho ra hàng một cách trực tiếp.
            </p>
          }
        >
          <p>
            The per-row kernels need no HIP-specific spelling (B1); unverified on HIP. In Metal,{' '}
            <code>[[thread_position_in_grid]]</code> gives the row directly.
          </p>
        </Translated>
      </Today>
    </>
  );
}
