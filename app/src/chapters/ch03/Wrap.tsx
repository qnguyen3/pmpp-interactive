import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { span } from '../../sim/blur';
import { layout, strides } from '../../sim/mapping';

// A kernel of our own, for one question: it normalises each row of a matrix in place.
// Checked copy: labs/ch03/snippets/scale_rows_defect.cu (Wrap.snippets.test.ts ties the two).
const SCALE_ROWS = `// Divide every row of a rows x cols matrix by that row's first element, in place.
__global__ void scaleRowsKernel(float* X, int rows, int cols) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        X[row*cols + col] = X[row*cols + col] / X[row*cols];
    }
}

void launchScaleRows(float* X, int rows, int cols) {
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((cols + 15)/16, (rows + 15)/16, 1);
    scaleRowsKernel<<<dimGrid, dimBlock>>>(X, rows, cols);
}`;

// The numbers the questions ask for, from the models the sections run on.
/** One token through the 1024 to 4096 layer under lab 3.3's launch: x covers l, y covers j, 16 by 16 blocks. */
const DECODE = layout({ height: 1, width: 4096, blockW: 16, blockH: 16, style: 'blocks' });
/** Valid positions along one axis for an element at index 0 with radius 2; a corner of a volume has three such axes. */
const AXIS_AT_CORNER = span(0, 64, 2);
const AXIS_INSIDE = span(32, 64, 2);
const CORNER_3D = AXIS_AT_CORNER ** 3;
/** Strides of a row-major [batch, j, k] tensor, in units of j and k: (j*k, k, 1). */
const BATCH_STRIDES = strides([8, 3, 5]); // batch 8, j = 3, k = 5

const n = (v: number) => v.toLocaleString('en-US');
// Keeps the spaces of an index expression but lets it wrap at one of them on a phone.
const pre = { whiteSpace: 'pre-wrap' as const };

export const questionsWrap: Record<string, Question> = {
  'ch03/wrap/q-inplace-rows': {
    kind: 'choice',
    id: 'ch03/wrap/q-inplace-rows',
    prompt: (
      <>
        <Translated
          vi={
            <>
              <code>launchScaleRows</code> ở trên được gọi trên một ma trận 4 × 50 trong đó mọi phần tử đều bằng 2.0. Những phần tử
              nào chắc chắn bằng 1.0 sau đó?
            </>
          }
        >
          <code>launchScaleRows</code> above is called on a 4 × 50 matrix in which every element is 2.0. Which elements
          are guaranteed to hold 1.0 afterwards?
        </Translated>
      </>
    ),
    options: [
      {
        label: (
          <Translated vi="Cả 200. Mỗi phần tử có đúng một luồng ghi, nên không thứ tự luồng nào có thể làm thay đổi kết quả.">
            All 200. Each element has exactly one writer, so no order of threads can change the result.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi phần tử một luồng ghi mới là một nửa quy tắc. Nửa kia là không luồng nào đọc thứ mà luồng khác ghi, và ở đây 196 luồng đọc một phần tử mà một luồng khác ghi vào.">
            One writer per element is half of the rule. The other half is that no thread reads what another thread writes, and here 196 threads read an element that a different thread stores to.
          </Translated>
        ),
      },
      { label: <Translated vi="Chỉ bốn phần tử ở cột 0.">Only the four in column 0.</Translated>, correct: true },
      {
        label: (
          <Translated vi="Các cột 0 đến 15: những luồng đó chung khối với luồng của cột 0.">
            Columns 0 to 15: those threads share a block with the thread of column 0.
          </Translated>
        ),
        why: (
          <Translated vi="Chung một khối không sắp thứ tự gì cả. Không có rào chắn (barrier), các luồng của cùng một khối cũng chạy theo thứ tự không được hứa hẹn.">
            Sharing a block orders nothing. Without a barrier the threads of one block run in no promised order either.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Không phần tử nào.">None of them.</Translated>,
        why: (
          <Translated vi="Luồng của cột 0 chỉ đọc phần tử của chính nó, và không luồng nào khác ghi vào đó: 2.0 / 2.0 là 1.0 dù theo thứ tự nào.">
            The thread of column 0 reads only its own element, and no other thread writes it: 2.0 / 2.0 is 1.0 in any order.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Mọi luồng chia cho <code>X[row*cols]</code>, và luồng của cột 0 trong hàng đó ghi đè nó bằng 1.0. Một luồng đọc nó
              trước lần ghi đó sẽ tính 2.0 / 2.0 = 1.0. Một luồng đọc nó sau đó sẽ tính 2.0 / 1.0 = 2.0. Không gì sắp thứ tự cho
              hai việc này, nên mỗi phần tử trong 196 phần tử còn lại có thể ra kết quả nào cũng được. Phép ánh xạ, phép kiểm tra
              và chỉ số phẳng đều đúng; lỗi nằm ở chỗ kernel đọc thứ mà nó ghi.
            </p>
            <p>
              Trên trình giả lập (emulator) CPU của khóa học, vốn chạy từng luồng một, ba thứ tự luồng mà các lab dùng (theo thứ
              tự, đảo ngược, xáo trộn) để lại lần lượt 4, 64 và 37 trên 200 phần tử bằng 1.0 (emulator-checked:{' '}
              <code>python3 labs/ch03/snippets/check.py</code>; không dùng GPU nào). Cách sửa là cách mục 3.3 đã dùng cho phép làm
              mờ: ghi sang một buffer thứ hai, hoặc truyền các số chia trong một mảng riêng mà không luồng nào ghi.
            </p>
          </>
        }
      >
        <>
          <p>
            Every thread divides by <code>X[row*cols]</code>, and the thread of column 0 in that row overwrites it with
            1.0. A thread that reads it before that store computes 2.0 / 2.0 = 1.0. One that reads it afterwards computes
            2.0 / 1.0 = 2.0. Nothing orders the two, so each of the other 196 elements can come out either way. The
            mapping, the guard and the flat index are all correct; the defect is that the kernel reads what it writes.
          </p>
          <p>
            On the course's CPU emulator, which runs one thread at a time, the three thread orders the labs use (in
            order, reversed, shuffled) leave 4, 64 and 37 of the 200 elements at 1.0 (emulator-checked:{' '}
            <code>python3 labs/ch03/snippets/check.py</code>; no GPU was involved). The fix is the one section 3.3 used
            for blur: write to a second buffer, or pass the divisors in an array of their own that no thread writes.
          </p>
        </>
      </Translated>
    ),
    source: 'our kernel; the discipline is that of ch. 3 §3.4, p. 60 (one thread, one output element), and of "Why two buffers" in section 3.3',
    vi: { source: 'kernel của khóa học này; kỷ luật là của ch. 3 §3.4, p. 60 (một luồng, một phần tử đầu ra), và của "Vì sao cần hai buffer" ở mục 3.3' },
  },
  'ch03/wrap/q-batched': {
    kind: 'choice',
    id: 'ch03/wrap/q-batched',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Một lớp tuyến tính được áp dụng cho cả một batch các chuỗi. <code>A</code> là một tensor{' '}
              <code>[batch, j, k]</code> theo thứ tự hàng (row-major), trọng số <code>B</code> là một ma trận <code>[k, l]</code>{' '}
              dùng chung cho mọi phần tử của batch, và <code>C</code> là <code>[batch, j, l]</code>. Một lưới 3D cho mỗi luồng{' '}
              <code>col</code> từ x, <code>row</code> từ y và <code>b</code> từ z. Ở bước <code>i</code> của vòng lặp, luồng cộng
              thêm tích nào?
            </>
          }
        >
          A linear layer is applied to a whole batch of sequences. <code>A</code> is a row-major{' '}
          <code>[batch, j, k]</code> tensor, the weight <code>B</code> is one <code>[k, l]</code> matrix shared by every
          entry, and <code>C</code> is <code>[batch, j, l]</code>. A 3D grid gives each thread <code>col</code> from x,{' '}
          <code>row</code> from y and <code>b</code> from z. Which product does the thread add at step <code>i</code> of
          its loop?
        </Translated>
      </>
    ),
    options: [
      { label: <code style={pre}>A[(b*j + row)*k + i] * B[i*l + col]</code>, correct: true },
      {
        label: <code style={pre}>A[b*j + row*k + i] * B[i*l + col]</code>,
        why: (
          <Translated vi="Chỉ số batch phải nhảy qua nguyên các ma trận j*k phần tử. Ở đây nó chỉ nhảy j phần tử, nên phần tử 1 của batch bắt đầu bên trong phần tử 0.">
            The batch index has to skip whole matrices of j*k elements. Here it skips j elements, so entry 1 starts inside entry 0.
          </Translated>
        ),
      },
      {
        label: <code style={pre}>A[(b*k + row)*j + i] * B[i*l + col]</code>,
        why: (
          <Translated vi="Các kích thước đặt sai chỗ: một hàng của A chứa k phần tử, nên k là bước nhảy (stride) của chỉ số hàng.">
            The sizes are in the wrong places: a row of A holds k elements, so k is the stride of the row index.
          </Translated>
        ),
      },
      {
        label: <code style={pre}>A[(b*j + row)*k + i] * B[(b*k + i)*l + col]</code>,
        why: (
          <Translated vi="B không có chiều batch. Với mọi b lớn hơn 0, biểu thức này đọc vượt quá cuối trọng số.">
            B has no batch dimension. For any b above 0 this reads past the end of the weight.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bước nhảy (stride) của một chỉ số là tích các kích thước ở bên phải nó. Với <code>[batch, j, k]</code> đó là (j·k,
            k, 1): với j = 3 và k = 5, mô hình cho ({BATCH_STRIDES.join(', ')}). Vậy phần tử nằm tại{' '}
            <code>b*j*k + row*k + i</code>, tức là <code>(b*j + row)*k + i</code>, và đó chính là{' '}
            <code>plane*m*n + row*m + col</code> của sách với tên khác. Trọng số được đánh chỉ số y như trong lab 3.3, vì mọi
            phần tử của batch đều nhân với cùng một ma trận. Lần ghi là <code>C[(b*j + row)*l + col]</code>, và phép kiểm tra
            xét cả ba b, row và col.
          </p>
        }
      >
        <p>
          The stride of an index is the product of the sizes to its right. For <code>[batch, j, k]</code> that is (j·k,
          k, 1): with j = 3 and k = 5 the model gives ({BATCH_STRIDES.join(', ')}). So the element is at{' '}
          <code>b*j*k + row*k + i</code>, which is <code>(b*j + row)*k + i</code>, and that is the book's{' '}
          <code>plane*m*n + row*m + col</code> with other names. The weight is indexed exactly as in lab 3.3, because
          every batch entry multiplies by the same matrix. The store is <code>C[(b*j + row)*l + col]</code>, and the
          guard tests all three of b, row and col.
        </p>
      </Translated>
    ),
    source: 'our kernel, built from ch. 3 §3.2, p. 55 (3D form) and §3.4, pp. 60–62 (index reasoning)',
    vi: { source: 'kernel của khóa học này, dựng từ ch. 3 §3.2, p. 55 (dạng 3D) và §3.4, pp. 60–62 (lập luận về chỉ số)' },
  },
  'ch03/wrap/q-decode-surplus': {
    kind: 'numeric',
    id: 'ch03/wrap/q-decode-surplus',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Lab 3.3 khởi chạy các khối 16 × 16, với x của lưới phủ l đầu ra và y phủ j hàng của batch. Một token đi qua một lớp
              có 1024 đầu vào và 4096 đầu ra: j = 1, k = 1024, l = 4096. Bao nhiêu luồng đã khởi chạy dừng lại ở phép kiểm tra?
            </>
          }
        >
          Lab 3.3 launches 16 × 16 blocks, with the grid's x covering the l outputs and its y covering the j rows of the
          batch. One token goes through a layer of 1024 inputs and 4096 outputs: j = 1, k = 1024, l = 4096. How many of
          the launched threads stop at the guard?
        </Translated>
      </>
    ),
    answer: DECODE.surplus,
    unit: 'threads',
    explain: (
      <Translated
        vi={
          <p>
            Lưới là ({DECODE.gridX}, {DECODE.gridY}): {DECODE.blocks} khối 256 luồng, tổng cộng {n(DECODE.threads)}. Chỉ hàng
            luồng đầu tiên của mỗi khối có phần tử của C, nên {n(DECODE.working)} luồng làm việc và{' '}
            {n(DECODE.surplus)} luồng dừng ở phép kiểm tra: cứ 16 thì 15. Lượng phép tính số học vẫn như với mọi cấu hình
            khởi chạy khác, nên riêng con số này không nói gì về thời gian; hàng batch 1 trong benchmark của lab 3.3 mới là phép
            đo. Các khối 256 × 1 sẽ phủ cùng hàng đó mà không có luồng dư nào. Chọn hình dạng khối từ hình dạng dữ liệu và từ
            những gì thiết bị báo cáo là cách chương 4 hoàn thành mốc dự án M1.
          </p>
        }
      >
        <p>
          The grid is ({DECODE.gridX}, {DECODE.gridY}): {DECODE.blocks} blocks of 256 threads, {n(DECODE.threads)} in
          all. Only the first thread row of each block has an element of C, so {n(DECODE.working)} threads work and{' '}
          {n(DECODE.surplus)} stop at the guard: 15 of every 16. The arithmetic done is the same as with any other
          launch, so this count alone says nothing about time; the batch 1 row of lab 3.3's benchmark is the
          measurement. Blocks of 256 by 1 would cover the same row with no surplus thread at all. Choosing the block
          shape from the shape of the data and from what the device reports is how chapter 4 completes milestone M1.
        </p>
      </Translated>
    ),
    source: 'method of ch. 3 §3.2, pp. 50 and 54–55, applied to our lab 3.3',
    vi: { unit: 'luồng', source: 'phương pháp của ch. 3 §3.2, pp. 50 và 54–55, áp dụng cho lab 3.3 của khóa học này' },
  },
  'ch03/wrap/q-corner3d': {
    kind: 'numeric',
    id: 'ch03/wrap/q-corner3d',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Một kernel lấy trung bình (2r + 1)³ giá trị của khối lập phương quanh mỗi phần tử của một khối thể tích 3D và bỏ qua
              các vị trí nằm ngoài khối thể tích, giống cách kernel làm mờ làm ở mép ảnh. Với r = 2, luồng của một phần tử ở góc
              cộng bao nhiêu giá trị?
            </>
          }
        >
          A kernel averages the (2r + 1)³ values of the cube around each element of a 3D volume and skips the
          positions outside the volume, the way the blur kernel does at the edges of a picture. With r = 2, how many values does the
          thread of a corner element add up?
        </Translated>
      </>
    ),
    answer: CORNER_3D,
    unit: 'values',
    explain: (
      <Translated
        vi={
          <p>
            Xét từng trục một. Quanh chỉ số 0, vòng lặp đi qua −2, −1, 0, 1, 2 và {AXIS_AT_CORNER} vị trí trong số đó tồn tại.
            Một góc nằm ở đầu mút của cả ba trục, và các số đếm nhân với nhau: {AXIS_AT_CORNER}³ = {CORNER_3D} trên{' '}
            {AXIS_INSIDE ** 3}. Một phần tử trên một cạnh của khối thể tích được {AXIS_AT_CORNER * AXIS_AT_CORNER * AXIS_INSIDE},
            một phần tử ở giữa một mặt được {AXIS_AT_CORNER * AXIS_INSIDE * AXIS_INSIDE}. Đó là lý do phép kiểm tra bên trong có
            một cặp vế cho mỗi trục, và vì sao số chia phải là một số đếm do luồng tự giữ.
          </p>
        }
      >
        <p>
          Take one axis at a time. Around index 0 the loop visits −2, −1, 0, 1, 2 and {AXIS_AT_CORNER} of those exist.
          A corner is at the end of all three axes, and the counts multiply: {AXIS_AT_CORNER}³ = {CORNER_3D} of{' '}
          {AXIS_INSIDE ** 3}. An element on an edge of the volume gets {AXIS_AT_CORNER * AXIS_AT_CORNER * AXIS_INSIDE},
          one in the middle of a face {AXIS_AT_CORNER * AXIS_INSIDE * AXIS_INSIDE}. That is why the inner test has one
          pair of clauses per axis, and why the divisor has to be a count kept by the thread.
        </p>
      </Translated>
    ),
    source: 'method of ch. 3 §3.3, pp. 58–59; the 3D kernel is ours (the book returns to 3D arrays in ch. 8)',
    vi: { unit: 'giá trị', source: 'phương pháp của ch. 3 §3.3, pp. 58–59; kernel 3D là của khóa học này (sách quay lại mảng 3D ở ch. 8)' },
  },
  'ch03/wrap/q-squaretest': {
    kind: 'choice',
    id: 'ch03/wrap/q-squaretest',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Một kernel cho <code>C[j × l] = A[j × k] · B[k × l]</code> đọc toán hạng thứ hai là{' '}
              <code>B[i*k + col]</code> trong khi <code>B[i*l + col]</code> mới đúng. Nó vượt qua hình dạng kiểm thử nào trong số
              này, viết dưới dạng (j, k, l)?
            </>
          }
        >
          A kernel for <code>C[j × l] = A[j × k] · B[k × l]</code> reads its second operand as{' '}
          <code>B[i*k + col]</code> where <code>B[i*l + col]</code> is right. Which of these test shapes, written (j, k,
          l), does it pass?
        </Translated>
      </>
    ),
    options: [
      {
        label: '(5, 7, 3)',
        why: (
          <Translated vi="k và l khác nhau, nên từ i = 1 trở đi mỗi lần đọc rơi quá xa 4·i phần tử.">
            k and l differ, so from i = 1 on each read lands 4·i elements too far along.
          </Translated>
        ),
      },
      {
        label: '(1, 40, 33)',
        why: (
          <Translated vi="Một hàng của C không giúp gì: bước nhảy thuộc về B, và 40 không phải 33.">
            One row of C does not help: the stride belongs to B, and 40 is not 33.
          </Translated>
        ),
      },
      { label: '(20, 1, 35)', correct: true },
      {
        label: '(33, 20, 1)',
        why: (
          <Translated vi="Ở đây B có 20 phần tử và kernel đọc B[i*20], vượt quá cuối của nó với mọi i lớn hơn 0.">
            B has 20 elements here and the kernel reads B[i*20], which is past its end for every i above 0.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hai biểu thức trùng nhau khi k = l. Chúng cũng trùng nhau khi k = 1, vì khi đó i chỉ bằng 0 và bước nhảy không nhân
            với gì cả. Vậy kernel này vượt qua mọi phép kiểm thử ma trận vuông và tích ngoài, và thất bại với mọi thứ khác. Các
            phép kiểm thử của lab 3.3 có những hình dạng mà cả ba kích thước đều khác nhau chính vì lý do này. Khi lab được xây
            dựng, lỗi này đã được chạy với chúng: trên Metal nó vượt qua bốn hình dạng vuông và hình dạng k = 1, 5 trên 12. Đó
            cũng là điều mà phép kiểm thử đầu tiên của lab 3.5, khối lập phương, có mặt để cho thấy.
          </p>
        }
      >
        <p>
          The two expressions agree when k = l. They also agree when k = 1, because then i is only ever 0 and the stride
          multiplies nothing. So this kernel passes every square test and the outer product, and fails everything else.
          Lab 3.3's tests include shapes in which all three sizes differ for exactly this reason. When the lab was
          built, this defect was run against them: on Metal it passed the four square shapes and the k = 1 shape, 5 of
          12. It is also what the first test of lab 3.5, the cube, is there to show.
        </p>
      </Translated>
    ),
    source: 'our lab 3.3; index reasoning of ch. 3 §3.4, p. 62',
    vi: { source: 'lab 3.3 của khóa học này; lập luận về chỉ số của ch. 3 §3.4, p. 62' },
  },
};

const M1 = 'ch03/lab3_naive_gemm';
// Backends that are not a GPU: the emulator and its race-detection flavour.
const EMULATED = new Set(['emu', 'race']);

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const devices = Object.keys(reports).filter((b) => !EMULATED.has(b));
  const m1 = devices.filter((b) => labStatus(reports[b], M1, 'work') === 'pass');
  const m1emu = labStatus(reports.emu, M1, 'work') === 'pass';
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch03/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  return (
    <>
      <h2 style={{ marginTop: 0 }}><Translated vi="Những gì cần mang theo">What to carry forward</Translated></h2>
      <Translated
        vi={
          <ul>
            <li>
              Một lần khởi chạy là hai giá trị <code>dim3</code>, một lưới (grid) các khối (block) và một khối các luồng
              (thread), mỗi thứ tới ba chiều. Một luồng chỉ biết bốn biến dựng sẵn của nó. Mỗi thứ được lớn đến đâu là thuộc tính
              của thiết bị, và một lần khởi chạy vi phạm giới hạn sẽ thất bại không một lời báo trừ khi host hỏi.
            </li>
            <li>
              <code>row = blockIdx.y*blockDim.y + threadIdx.y</code> và{' '}
              <code>col = blockIdx.x*blockDim.x + threadIdx.x</code>. Hàng lấy từ y và cột lấy từ x, nên x của lưới được định
              kích thước theo chiều rộng và y theo chiều cao. Nhãn và kích thước dữ liệu được viết với chiều cao nhất trước; hàm
              dựng <code>dim3</code> nhận x trước.
            </li>
            <li>
              Một lưới gồm nguyên các khối, nên một số luồng không có phần tử. Mọi kernel kiểm tra mọi chỉ số mà nó tính ra. Trên
              Metal, <code>dispatchThreads</code> làm lưới khớp đúng với dữ liệu và phép kiểm tra không còn cần nữa;{' '}
              <code>dispatchThreadgroups</code> là tình huống của sách.
            </li>
            <li>
              Bộ nhớ là phẳng. Một phần tử của mảng theo thứ tự hàng (row-major) nằm tại <code>row*width + col</code>, và ở ba
              chiều tại <code>plane*m*n + row*m + col</code>: bước nhảy (stride) của một chỉ số là tích các kích thước ở bên phải nó.
            </li>
            <li>
              Một luồng có thể lặp. Kernel làm mờ có hai phép kiểm tra ngăn hai lỗi khác nhau: phép bên ngoài giữ cho luồng dư
              không ghi, phép bên trong giữ cho một vùng lân cận không đọc ra ngoài bức ảnh. Hãy chia cho số đã đếm được, không
              phải cho diện tích vùng lân cận.
            </li>
            <li>
              Mỗi luồng ghi phần tử đầu ra của riêng nó và chỉ đọc những gì không luồng nào ghi. Khi đó không cần rào chắn
              (barrier), khóa hay phép toán nguyên tử (atomic) nào, và không thứ tự luồng nào làm thay đổi kết quả. Một kernel cập
              nhật đầu vào của nó tại chỗ là phá vỡ nửa sau của quy tắc đó.
            </li>
            <li>
              Phép nhân ma trận ngây thơ đọc <code>M[row*Width + k]</code> dọc theo một hàng và <code>N[k*Width + col]</code> xuống
              một cột. Với ma trận chữ nhật, mỗi <code>Width</code> trở thành đúng một trong j, k và l, và một kernel được viết
              cho một quy ước lưu trữ: quy ước của sách, được giữ trong lab 3.3 và 5.3, lưu trọng số dạng (in, out), là chuyển vị
              của thứ PyTorch và ggml lưu.
            </li>
            <li>
              Một đoạn mã in trong sách không phải mã đã kiểm thử. Phép ép kiểu gắn chặt hơn <code>*</code>, trình biên dịch không
              nói gì, và chỉ một phép kiểm thử có đáp án biết trước mới cho thấy điều đó. Hãy kiểm thử với các kích thước khác
              nhau và không phải bội số của kích thước khối.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            A launch is two <code>dim3</code> values, a grid of blocks and a block of threads, each up to three
            dimensions. A thread knows only its four built-ins. How large either may be is a property of the device, and
            a launch that breaks a limit fails without a word unless the host asks.
          </li>
          <li>
            <code>row = blockIdx.y*blockDim.y + threadIdx.y</code> and{' '}
            <code>col = blockIdx.x*blockDim.x + threadIdx.x</code>. Rows come from y and columns from x, so the grid's x
            is sized from the width and its y from the height. Labels and data sizes are written highest dimension
            first; the <code>dim3</code> constructor takes x first.
          </li>
          <li>
            A grid is whole blocks, so some threads have no element. Every kernel guards every index it computed. On
            Metal, <code>dispatchThreads</code> makes the grid exactly the data and the guard goes away;{' '}
            <code>dispatchThreadgroups</code> is the book's situation.
          </li>
          <li>
            Memory is flat. An element of a row-major array is at <code>row*width + col</code>, and in three dimensions
            at <code>plane*m*n + row*m + col</code>: the stride of an index is the product of the sizes to its right.
          </li>
          <li>
            A thread may loop. The blur kernel has two tests that prevent two different failures: the outer one keeps
            surplus threads from storing, the inner one keeps a patch from reading outside the picture. Divide by what
            was counted, not by the patch area.
          </li>
          <li>
            Each thread writes its own output element and reads only what no thread writes. Then no barrier, lock or
            atomic is needed and no order of threads changes the result. A kernel that updates its input in place
            breaks the second half of that rule.
          </li>
          <li>
            The naive matmul reads <code>M[row*Width + k]</code> along a row and <code>N[k*Width + col]</code> down a
            column. For rectangular matrices each <code>Width</code> becomes exactly one of j, k and l, and a kernel is
            written for one storage convention: the book's, kept by labs 3.3 and 5.3, holds the weight as (in, out),
            the transpose of what PyTorch and ggml store.
          </li>
          <li>
            A printed listing is not tested code. A cast binds tighter than <code>*</code>, the compiler says nothing,
            and only a test with a known answer shows it. Test with sizes that differ from one another and that are not
            multiples of the block.
          </li>
        </ul>
      </Translated>
      <Source section="3.5" pages="64" />

      <h2><Translated vi="Bài kiểm tra tổng hợp">A mixed check</Translated></h2>
      <Translated
        vi={
          <p>
            Các câu này dựa trên cả chương và trên những tình huống các mục chưa cho thấy.
            {asked ? ` Đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu hỏi, tính câu trả lời gần nhất của bạn cho mỗi câu.` : ''}
          </p>
        }
      >
        <p>
          These draw on the whole chapter and on situations the sections did not show.
          {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
        </p>
      </Translated>
      <Code
        tabs={[
          {
            label: tr('CUDA (not a lab file: for this question only)', 'CUDA (không phải tệp lab: chỉ cho câu hỏi này)'),
            code: SCALE_ROWS,
            tiers: { X: 'global', row: 'reg', col: 'reg' },
            status: tr('front-end-checked; has a defect on purpose', 'front-end-checked; cố ý có một lỗi'),
          },
        ]}
        caption={
          <Translated
            vi={
              <>
                Của khóa học này, không phải của sách. Bản đã kiểm tra là <code>labs/ch03/snippets/scale_rows_defect.cu</code>.
              </>
            }
          >
            <>
              Ours, not the book's. The checked copy is <code>labs/ch03/snippets/scale_rows_defect.cu</code>.
            </>
          </Translated>
        }
      />
      <Ask q={questionsWrap['ch03/wrap/q-inplace-rows']} label={tr('Find the defect', 'Tìm lỗi')} />
      <Ask q={questionsWrap['ch03/wrap/q-batched']} />
      <Ask q={questionsWrap['ch03/wrap/q-decode-surplus']} />
      <Ask q={questionsWrap['ch03/wrap/q-corner3d']} />
      <Ask q={questionsWrap['ch03/wrap/q-squaretest']} />

      <h2><Translated vi="Bài tập của sách">The book's exercises</Translated></h2>
      <Translated
        vi={
          <p>
            Chương kết thúc bằng năm bài tập (pp. 64–65), in ngay sau phần tóm tắt mà không có tiêu đề riêng. Mỗi bài được giải
            ở đâu đó trong khóa học này; sách không in đáp án, nên đáp án ở đây là của khóa học này.
          </p>
        }
      >
        <p>
          The chapter ends with five exercises (pp. 64–65), printed straight after the summary with no heading of their
          own. Each is worked somewhere in this course; the book prints no answers, so the answers here are ours.
        </p>
      </Translated>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="exercises">
        <thead>
          <tr><th><Translated vi="Bài tập">Exercise</Translated></th><th><Translated vi="Nội dung">About</Translated></th><th><Translated vi="Được giải ở đâu trong khóa học">Where it is worked here</Translated></th></tr>
        </thead>
        <tbody>
          <tr>
            <td>1a, 1b</td>
            <td><Translated vi="Nhân ma trận với mỗi luồng một hàng đầu ra, và mỗi luồng một cột đầu ra">Matmul with one thread per output row, and per output column</Translated></td>
            <td><Translated vi={<>lab 3.4 trên <a href={href('ch03', 'labs')}>trang lab</a>; trình khám phá độ mịn công việc ở <a href={href('ch03', '3.4')}>3.4</a></>}>lab 3.4 on the <a href={href('ch03', 'labs')}>labs page</a>; the granularity explorer in <a href={href('ch03', '3.4')}>3.4</a></Translated></td>
          </tr>
          <tr>
            <td>1c</td>
            <td><Translated vi="Ưu và nhược điểm của hai thiết kế">Pros and cons of the two designs</Translated></td>
            <td><Translated vi={<>một câu hỏi ở <a href={href('ch03', '3.4')}>3.4</a>, rồi phép đo của chính bạn trong lab 3.4</>}>a question in <a href={href('ch03', '3.4')}>3.4</a>, then your own measurement in lab 3.4</Translated></td>
          </tr>
          <tr>
            <td>2</td>
            <td><Translated vi="Ma trận nhân vector, mỗi luồng một phần tử đầu ra">Matrix times vector, one thread per output element</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    một câu hỏi ở <a href={href('ch03', '3.4')}>3.4</a>; phép kiểm thử l = 1 của lab 3.3. Phần mã host mà bài tập cũng
                    yêu cầu là quy trình cấp phát, sao chép, khởi chạy, sao chép về của chương 2, và không được lặp lại.
                  </>
                }
              >
                a question in <a href={href('ch03', '3.4')}>3.4</a>; the l = 1 test of lab 3.3. The host stub the
                exercise also asks for is chapter 2's allocate, copy, launch, copy back, and is not repeated.
              </Translated>
            </td>
          </tr>
          <tr>
            <td>3a, 3b, 3c</td>
            <td><Translated vi="Số luồng mỗi khối, số luồng và số khối trong một lưới, đọc từ mã host">Threads per block, threads and blocks in a grid, read from host code</Translated></td>
            <td><Translated vi={<>câu hỏi cuối của <a href={href('ch03', '3.1')}>3.1</a></>}>the closing question of <a href={href('ch03', '3.1')}>3.1</a></Translated></td>
          </tr>
          <tr>
            <td>3d</td>
            <td><Translated vi="Bao nhiêu luồng thực thi dòng được kiểm tra">How many threads execute the guarded line</Translated></td>
            <td><Translated vi={<>một câu hỏi ở <a href={href('ch03', '3.2')}>3.2</a>, và cấu hình mẫu "exercise 3" của bản đồ luồng ở đó</>}>a question in <a href={href('ch03', '3.2')}>3.2</a>, and the "exercise 3" preset of its thread map</Translated></td>
          </tr>
          <tr>
            <td>4a, 4b</td>
            <td><Translated vi="Chỉ số phẳng theo thứ tự hàng (row-major) và theo thứ tự cột (column-major)">Flat index in row-major and in column-major order</Translated></td>
            <td><Translated vi={<>hai câu hỏi và trình khám phá phép làm phẳng ở <a href={href('ch03', '3.2')}>3.2</a></>}>two questions and the flattening explorer in <a href={href('ch03', '3.2')}>3.2</a></Translated></td>
          </tr>
          <tr>
            <td>5</td>
            <td><Translated vi="Chỉ số phẳng vào một tensor 3D">Flat index into a 3D tensor</Translated></td>
            <td><Translated vi={<>một câu hỏi ở <a href={href('ch03', '3.2')}>3.2</a>, phát biểu bằng bước nhảy; câu hỏi về batch ở trên</>}>a question in <a href={href('ch03', '3.2')}>3.2</a>, stated with strides; the batched question above</Translated></td>
          </tr>
        </tbody>
      </table>

      <h2><Translated vi="Những chỗ PDF có vẻ sai">Where the PDF looks wrong</Translated></h2>
      <Translated
        vi={
          <p>
            Ấn bản này còn mới và bản mà khóa học này dựa vào có những chỗ có vẻ nhầm ở chương 3. Bốn chỗ đầu nằm trong mã của
            Fig. 3.4 và lời gọi của nó: ba chỗ làm thay đổi thứ chương trình tính ra và chỗ thứ tư khiến lời gọi như in không
            biên dịch được, đó là lý do các lab không dùng đoạn mã như sách in. Những chỗ còn lại được liệt kê để bạn không bị
            bối rối khi đọc sách song song.
          </p>
        }
      >
        <p>
          This edition is new and the copy this course was built from has apparent slips in chapter 3. The first four are
          in the code of Fig. 3.4 and its call: three change what the program computes and the fourth stops the printed
          call from compiling, which is why the labs do not use the listing as printed. The rest are listed so that you
          are not thrown by them when you read the book alongside.
        </p>
      </Translated>
      <Translated
        vi={
          <ul data-list="errata">
            <li>
              Fig. 3.4, dòng 19 (p. 53): <code>(unsigned char) 0.21f*r+0.71f*g+0.07f*b</code>. Phép ép kiểu chỉ áp dụng cho{' '}
              <code>0.21f</code>, giá trị này thành 0, nên số hạng màu đỏ bị mất. Cả tổng cần được đặt trong ngoặc.{' '}
              <a href={href('ch03', '3.2')}>Mục 3.2</a> cho bạn tự tìm ra lỗi này, và lab 3.1 có phiên bản đã kiểm thử.
            </li>
            <li>
              Cùng dòng đó dùng 0.21, 0.71 và 0.07, cộng lại là 0.99. Phần văn bản ở p. 52 cho 0.299, 0.587 và 0.114, là những
              giá trị khóa học này dùng.
            </li>
            <li>
              Kernel trong đoạn mã nhận <code>(Pout, Pin, width, height)</code>, còn lời gọi từ host ở p. 50 truyền{' '}
              <code>(Pin_d, Pout_d, m, n)</code>. Như sách in, đầu vào và đầu ra bị đảo chỗ.
            </li>
            <li>
              Đoạn mã đặt tên kernel là <code>colortoGrayscaleConvertion</code>; phần văn bản, chú thích hình và lời gọi từ host
              ghi <code>colorToGrayscaleConversion</code>. Lời gọi như sách in sẽ không biên dịch được với đoạn mã đó.
            </li>
            <li>
              p. 53 dẫn "line 07" cho phép kiểm tra phạm vi, vốn nằm ở dòng 08 của Fig. 3.4. Chú thích ở dòng 11 và 12 ghi
              CHANNEL trong khi mã dùng <code>CHANNELS</code>.
            </li>
            <li>
              Chú thích của Fig. 3.5 ghi "76 × 62 picture". Phần văn bản và Fig. 3.2 ghi 62 × 76, hàng trước, đúng quy ước của
              chính sách (chú thích 1, p. 49). Phần văn bản gọi vùng 2 và 3 là tô đậm vừa và vùng 4 là tô nhạt; trong bản của
              chúng tôi, các vùng 2 đến 4 trông giống nhau, và chính đường viền và chữ số mới phân biệt được chúng.
            </li>
            <li>p. 48 in "threadId.x" thay cho <code>threadIdx.x</code>.</li>
            <li>
              Fig. 3.8 (p. 57): số dòng in ra nhảy từ 07 sang 09, và chú thích ở dòng 09 gọi vùng lân cận là một "BLUR_SIZE x
              BLUR_SIZE box". Vùng lân cận có cạnh 2·BLUR_SIZE + 1, như phần văn bản ở p. 58 nói. Sách không in cấu hình khởi
              chạy nào cho kernel này.
            </li>
            <li>Fig. 3.10 ghi nhãn chiều cao của một ô (tile) là "BLOCK_WIDTHE".</li>
            <li>
              Fig. 3.11 đặt tên kernel là <code>MatrixMulKernel</code>; phần văn bản và chú thích của Fig. 3.12 viết{' '}
              <code>matrixMulKernel</code>. Chương này không bao giờ in cấu hình khởi chạy hay định nghĩa của{' '}
              <code>BLOCK_WIDTH</code> cho nó; cấu hình khởi chạy ở <a href={href('ch03', '3.4')}>mục 3.4</a> là của khóa học này.
            </li>
            <li>
              p. 63 tính hàng và cột của Thread (1,0) của Block (0,0) là "0 ∗ 0 + 1 = 1 and 0 ∗ 0 + 0 = 0". Với các khối rộng 2,
              các tích là 0 ∗ 2; kết quả vẫn như nhau. Cũng ở trang đó, "k ∗ Width + col2 ∗ 4 + 0 = 8" thiếu dấu bằng sau "col".
            </li>
            <li>
              Những chỗ nhỏ hơn: lưới ô của Fig. 3.9 không ghi kích thước, một số nhãn của Fig. 3.13 bị các mũi tên che, và một
              câu trong khung BLAS ở p. 60 thiếu chữ "in".
            </li>
          </ul>
        }
      >
        <ul data-list="errata">
          <li>
            Fig. 3.4, line 19 (p. 53): <code>(unsigned char) 0.21f*r+0.71f*g+0.07f*b</code>. The cast applies to{' '}
            <code>0.21f</code> alone, which becomes 0, so the red term is lost. The whole sum needs parentheses.{' '}
            <a href={href('ch03', '3.2')}>Section 3.2</a> has you find this, and lab 3.1 has the tested version.
          </li>
          <li>
            The same line uses 0.21, 0.71 and 0.07, which add up to 0.99. The text on p. 52 gives 0.299, 0.587 and
            0.114, which is what this course uses.
          </li>
          <li>
            The listing's kernel takes <code>(Pout, Pin, width, height)</code>, and the host call on p. 50 passes{' '}
            <code>(Pin_d, Pout_d, m, n)</code>. As printed, input and output are swapped.
          </li>
          <li>
            The listing names the kernel <code>colortoGrayscaleConvertion</code>; the text, the caption and the host
            call say <code>colorToGrayscaleConversion</code>. The call as printed would not compile against the listing.
          </li>
          <li>
            p. 53 cites "line 07" for the range test, which is on line 08 of Fig. 3.4. The comment on lines 11 and 12
            says CHANNEL where the code has <code>CHANNELS</code>.
          </li>
          <li>
            The caption of Fig. 3.5 says "76 × 62 picture". The text and Fig. 3.2 say 62 × 76, rows first, which is the
            book's own convention (footnote 1, p. 49). The text calls areas 2 and 3 medium shaded and area 4 lightly
            shaded; in our copy areas 2 to 4 look alike, and the borders and numerals are what tell them apart.
          </li>
          <li>p. 48 prints "threadId.x" for <code>threadIdx.x</code>.</li>
          <li>
            Fig. 3.8 (p. 57): the printed line numbers go from 07 to 09, and the comment on line 09 calls the patch a
            "BLUR_SIZE x BLUR_SIZE box". The patch is 2·BLUR_SIZE + 1 on a side, as the text on p. 58 says. The book
            prints no launch for this kernel.
          </li>
          <li>Fig. 3.10 labels the height of a tile "BLOCK_WIDTHE".</li>
          <li>
            Fig. 3.11 names the kernel <code>MatrixMulKernel</code>; the text and the caption of Fig. 3.12 write{' '}
            <code>matrixMulKernel</code>. The chapter never prints the launch or a definition of{' '}
            <code>BLOCK_WIDTH</code> for it; the launch in <a href={href('ch03', '3.4')}>section 3.4</a> is ours.
          </li>
          <li>
            p. 63 computes the row and column of Thread (1,0) of Block (0,0) as "0 ∗ 0 + 1 = 1 and 0 ∗ 0 + 0 = 0". With
            blocks 2 wide the products are 0 ∗ 2; the results are the same. On the same page "k ∗ Width + col2 ∗ 4 + 0 =
            8" is missing an equals sign after "col".
          </li>
          <li>
            Smaller things: the lattice of Fig. 3.9 is not dimensioned, some labels of Fig. 3.13 are hidden under its
            arrows, and a sentence of the BLAS sidebar on p. 60 is missing the word "in".
          </li>
        </ul>
      </Translated>
      <Aside>
        <Translated vi={<p>Đây là cách chúng tôi đọc một bản PDF. Nếu bản của bạn khác, hãy tin bản của bạn.</p>}>
          <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Có một chỗ nhầm không phải của sách. Trang của Apple về cách tính kích thước threadgroup và lưới in "a width of 32, a
            height of 48" threadgroup cho ví dụ 1024 × 768 của nó, trong khi chính mã và các giá trị của nó cho 64 × 24. Cả hai
            đều nhân ra 1,536 threadgroup mà trang đó nêu sau đó.
          </p>
        }
      >
        <p>
          One slip is not the book's. Apple's page on calculating threadgroup and grid sizes prints "a width of 32, a
          height of 48" threadgroups for its 1024 × 768 example, where its own code and values give 64 by 24. Both
          multiply to the 1,536 threadgroups it then states.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, C3)">{/* i18n-ignore: date and notes reference */}
        <Translated
          vi={
            <p>
              Các giá trị của trang là 512 luồng mỗi nhóm và độ rộng thực thi 16, nên các nhóm rộng 16 và cao 32. Phép tính được
              làm lại bằng tay, không chạy.
            </p>
          }
        >
          <p>
            The page's values are 512 threads per group and an execution width of 16, so groups 16 wide and 32 high.
            The arithmetic was redone by hand, not run.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Những gì chương này bỏ qua">What this chapter left out</Translated></h2>
      <Translated
        vi={
          <ul>
            <li>
              Khung "Memory Space" (p. 51) là một câu ở mục 3.2, và khung "Linear Algebra Functions" (pp. 59–60) là một đoạn ở
              mục 3.4. Các bộ giải và thủ tục tính trị riêng không được bàn đến.
            </li>
            <li>
              Làm mờ ảnh dùng để làm gì, và bức ảnh của Fig. 3.6 (p. 56). Khóa học dùng những bức ảnh tổng hợp nhỏ để mọi con số
              trên màn hình đều kiểm tra được bằng tay.
            </li>
            <li>Làm mờ có trọng số và làm mờ Gauss (p. 56): chương 7, cùng với tích chập (convolution).</li>
            <li>Mảng 3D trong kernel stencil (p. 55): chương 8.</li>
            <li>
              Bốn loại khối tốn gì khi các luồng của một warp bất đồng ở phép kiểm tra: chương 4. Lưu lượng bộ nhớ của phép nhân
              ma trận ngây thơ tốn gì: chương 5. Hai kiểu truy cập của nó khác nhau về chi phí ra sao: chương 6.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            The "Memory Space" sidebar (p. 51) is one sentence in section 3.2, and the "Linear Algebra Functions"
            sidebar (pp. 59–60) one paragraph in section 3.4. Solvers and eigenvalue routines are not discussed.
          </li>
          <li>
            What blurring is used for, and the photograph of Fig. 3.6 (p. 56). The course uses small synthetic pictures
            so that every number on screen can be checked by hand.
          </li>
          <li>Weighted and Gaussian blur (p. 56): chapter 7, with convolution.</li>
          <li>3D arrays in stencil kernels (p. 55): chapter 8.</li>
          <li>
            What the four kinds of block cost when a warp's threads disagree at the guard: chapter 4. What the naive
            matmul's memory traffic costs: chapter 5. How its two access patterns differ in cost: chapter 6.
          </li>
        </ul>
      </Translated>

      <h2><Translated vi="Dự án đến lúc này">The project so far</Translated></h2>
      <p>
        <Translated
          vi={
            <>
              Kernel của lab 3.3, một GEMM ngây thơ cho mọi hình dạng với chiều batch đứng đầu, là nửa đầu của mốc dự án M1 trong
              dự án xuyên suốt: phép tính đằng sau mọi lớp tuyến tính của transformer nhỏ mà cuối cùng bạn sẽ chạy, ở dạng đơn giản
              nhất của nó.{' '}
            </>
          }
        >
          Lab 3.3's kernel, a naive GEMM for any shapes with the batch first, is the first half of milestone M1 of the
          through-line project: the computation behind every linear layer of the small transformer you will end up
          running, in its plainest form.{' '}
        </Translated>
        <span data-m1>
          {devices.length === 0 && !reports.emu
            ? tr('No lab results are recorded on this machine yet.', 'Chưa có kết quả lab nào được ghi lại trên máy này.')
            : m1.length
              ? tr(`Your own version passes on ${m1.join(' and ')} on this machine.`, `Phiên bản của bạn vượt qua trên ${m1.join(' và ')} trên máy này.`)
              : m1emu
                ? tr('Your own version passes on the emulator; it has not passed on a device on this machine yet.', 'Phiên bản của bạn vượt qua trên trình giả lập; nó chưa vượt qua trên thiết bị nào của máy này.')
                : tr('Your own version has not passed on this machine yet.', 'Phiên bản của bạn chưa vượt qua trên máy này.')}
        </span>{' '}
        <Translated
          vi={
            <>
              Xem các <a href={href('ch03', 'labs')}>lab</a>. Nửa sau thuộc về chương 4: khối 16 × 16 của chương này là một lựa
              chọn cố định, còn ở đó cấu hình khởi chạy được chọn từ những gì thiết bị báo cáo. Sau đó chương 5 thay kernel đằng
              sau cùng những đối số đó bằng một kernel phân ô, mốc dự án M2.
            </>
          }
        >
          See the <a href={href('ch03', 'labs')}>labs</a>. The second half is chapter 4's: the 16 × 16 block of this
          chapter is a fixed choice, and there the launch configuration is chosen from what the device reports. Chapter
          5 then replaces the kernel behind the same arguments with a tiled one, milestone M2.
        </Translated>
      </p>
      <Translated
        vi={
          <p>
            Chương 4 mở chiếc hộp thiết bị mà chương này coi là kín. Nó cho thấy các khối được lập lịch thế nào, các luồng của
            một khối được gom thành warp ra sao, và một nhánh tốn gì khi các luồng của một warp đi theo hai phía khác nhau của
            nó. Phép kiểm tra của chương này chính là một nhánh như vậy, và bốn loại khối ở mục 3.2 quay lại ở đó làm ví dụ minh
            họa. Phép đo khối rộng so với khối cao trong lab 3.1 được để ngỏ cho đến lúc đó.
          </p>
        }
      >
        <p>
          Chapter 4 opens the device this chapter treated as a box. It shows how blocks are scheduled, how the threads
          of a block are grouped into warps, and what a branch costs when the threads of one warp take different sides
          of it. This chapter's guard is such a branch, and the four kinds of block from section 3.2 return there as
          the worked example. Lab 3.1's measurement of wide blocks against tall ones is left open until then.
        </p>
      </Translated>
    </>
  );
}
