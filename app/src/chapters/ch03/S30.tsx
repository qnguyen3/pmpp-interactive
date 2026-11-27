import { Ask, type Question } from '../../components/Ask';
import { Aside, Source, type Tier } from '../../components/notes';
import { Code, useLabFile } from '../../components/Code';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { coverage, dim3, globalCoord, sizeGrid } from '../../sim/grid';

const VEC_TIERS: Record<string, Tier> = {
  A: 'global', B: 'global', C: 'global', A_d: 'global', B_d: 'global', C_d: 'global',
  i: 'reg', n: 'reg', threads: 'reg', blocks: 'reg',
};

// The numbers the prerequisite questions ask for, from the same model section 3.1 runs on.
const INDEX = globalCoord(dim3(3, 0, 0), dim3(256), dim3(10, 0, 0)).x;
const EX4 = coverage(8000, 1024, sizeGrid(8000, 1024, 'round-up'));

// Prerequisite questions drawn from chapter 2 (from the notes for that chapter).
export const questionsS30: Record<string, Question> = {
  'ch03/start/q-globalindex': {
    kind: 'numeric',
    id: 'ch03/start/q-globalindex',
    prompt: (
      <Translated
        vi={
          <>
            Kernel ở trên được khởi chạy với 256 luồng trong mỗi khối. Luồng có{' '}
            <code>blockIdx.x = 3</code> và <code>threadIdx.x = 10</code> xử lý phần tử nào? Hãy cho giá trị của <code>i</code>.
          </>
        }
      >
        The kernel above is launched with 256 threads in each block. Which element does the thread with{' '}
        <code>blockIdx.x = 3</code> and <code>threadIdx.x = 10</code> handle? Give the value of <code>i</code>.
      </Translated>
    ),
    answer: INDEX,
    explain: (
      <Translated
        vi={
          <p>
            <code>i = blockIdx.x * blockDim.x + threadIdx.x</code> = 3 × 256 + 10 = {INDEX}. Các khối 0, 1 và 2 phủ các
            phần tử 0 đến 767, nên khối 3 bắt đầu tại 768. Chương này dùng đúng công thức đó một lần cho mỗi chiều: một bản
            cho cột, một bản cho hàng.
          </p>
        }
      >
        <p>
          <code>i = blockIdx.x * blockDim.x + threadIdx.x</code> = 3 × 256 + 10 = {INDEX}. Blocks 0, 1 and 2 cover
          elements 0 to 767, so block 3 starts at 768. This chapter uses the same formula once for each dimension: one
          copy for the column, one for the row.
        </p>
      </Translated>
    ),
    source: 'ch. 2 §2.5, p. 34',
  },
  'ch03/start/q-surplus': {
    kind: 'numeric',
    id: 'ch03/start/q-surplus',
    prompt: (
      <Translated
        vi={
          <>
            Một vector 8000 phần tử, mỗi luồng một phần tử, các khối 1024 luồng, và số khối ít nhất đủ phủ nó. Có bao
            nhiêu luồng không qua được phép kiểm tra <code>i &lt; n</code>?
          </>
        }
      >
        A vector of 8000 elements, one element per thread, blocks of 1024 threads, and the fewest blocks that cover
        it. How many threads fail the <code>i &lt; n</code> test?
      </Translated>
    ),
    answer: EX4.surplus,
    unit: 'threads',
    explain: (
      <Translated
        vi={
          <p>
            8000 / 1024 làm tròn lên thành {EX4.blocks} khối, tức là {EX4.threads.toLocaleString('en-US')} luồng.{' '}
            {EX4.working.toLocaleString('en-US')} luồng trong số đó có phần tử; {EX4.surplus} luồng còn lại tính ra một chỉ
            số vượt quá cuối mảng và không được làm gì. Một lưới luôn gồm một số nguyên khối, nên có luồng dư là chuyện
            bình thường. Ở hai chiều, phần dư trở thành nguyên các hàng và cột luồng, và phép kiểm tra có thêm vế thứ hai.
          </p>
        }
      >
        <p>
          8000 / 1024 rounds up to {EX4.blocks} blocks, which is {EX4.threads.toLocaleString('en-US')} threads.{' '}
          {EX4.working.toLocaleString('en-US')} of them have an element; the other {EX4.surplus} compute an index past
          the end and must do nothing. A grid is always a whole number of blocks, so some surplus is normal. In two
          dimensions it becomes whole rows and columns of threads, and the test gets a second half.
        </p>
      </Translated>
    ),
    source: 'ch. 2 §2.5 and §2.6, pp. 37–39; exercise 4, p. 41',
    vi: { unit: 'luồng', source: 'ch. 2 §2.5 và §2.6, pp. 37–39; bài tập 4, p. 41' },
  },
  'ch03/start/q-bytes': {
    kind: 'choice',
    id: 'ch03/start/q-bytes',
    prompt: (
      <Translated
        vi={
          <>
            Bạn cấp phát bộ nhớ device cho một ảnh màu rộng <code>w</code> điểm ảnh, cao <code>h</code>, lưu thành ba{' '}
            <code>unsigned char</code> mỗi điểm ảnh. Đối số kích thước của <code>cudaMalloc</code> là gì?
          </>
        }
      >
        You allocate device memory for a colour image, <code>w</code> pixels wide and <code>h</code> high, stored as
        three <code>unsigned char</code> per pixel. What is the size argument of <code>cudaMalloc</code>?
      </Translated>
    ),
    options: [
      {
        label: <code>w * h * 3</code>,
        correct: true,
        why: (
          <Translated vi="Kích thước tính bằng byte, và một unsigned char là một byte.">
            The size is in bytes, and an unsigned char is one byte.
          </Translated>
        ),
      },
      {
        label: <code>w * h</code>,
        why: (
          <Translated vi="Đó là số điểm ảnh. Mỗi điểm ảnh chiếm ba byte.">
            That is the number of pixels. Each pixel takes three bytes.
          </Translated>
        ),
      },
      {
        label: <code>w * h * 3 * sizeof(float)</code>,
        why: (
          <Translated vi="Gấp bốn lần mức cần trên máy có float 4 byte: các phần tử là unsigned char, không phải float.">
            Four times too much on a machine with 4-byte floats: the elements are unsigned char, not float.
          </Translated>
        ),
      },
      {
        label: <code>(void**)&amp;Pin_d</code>,
        why: (
          <Translated vi="Đó là đối số thứ nhất: địa chỉ của con trỏ sẽ nhận vùng nhớ được cấp phát.">
            That is the first argument: the address of the pointer that will receive the allocation.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>cudaMalloc</code> nhận địa chỉ của một con trỏ, ép kiểu thành <code>(void**)</code>, và một kích thước
            tính bằng <b>byte</b>. Nó ghi con trỏ qua đối số thứ nhất để giá trị trả về được dành cho việc báo lỗi. Kernel
            đầu tiên của chương này đọc đúng một buffer như vậy, và thừa số 3 lại xuất hiện bên trong nó dưới dạng một chỉ số.
          </p>
        }
      >
        <p>
          <code>cudaMalloc</code> takes the address of a pointer, cast to <code>(void**)</code>, and a size in{' '}
          <b>bytes</b>. It writes the pointer through its first argument so that its return value is free to report an
          error. The first kernel of this chapter reads exactly such a buffer, and the factor of 3 turns up again inside
          it as an index.
        </p>
      </Translated>
    ),
    source: 'ch. 2 §2.4, pp. 28–30',
  },
  'ch03/start/q-order': {
    kind: 'choice',
    id: 'ch03/start/q-order',
    prompt: (
      <Translated
        vi={<>Một kernel được khởi chạy trên nhiều khối. Mã của nó có được dựa vào việc khối 0 xong trước khi khối 1 bắt đầu không?</>}
      >
        <>A kernel is launched over many blocks. May its code rely on block 0 finishing before block 1 starts?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Không. Các khối có thể chạy theo bất kỳ thứ tự nào.">No. Blocks may execute in any order.</Translated>,
        correct: true,
      },
      {
        label: (
          <Translated vi="Có. Các khối chạy theo thứ tự tăng dần của blockIdx.x.">
            Yes. Blocks run in increasing order of blockIdx.x.
          </Translated>
        ),
        why: (
          <Translated vi="Sách không hứa hẹn bất kỳ thứ tự nào giữa các khối.">
            The book promises no order at all between blocks.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ trên một GPU nhỏ chạy mỗi lần một khối.">
            Only on a small GPU that runs one block at a time.
          </Translated>
        ),
        why: (
          <Translated vi="Một GPU nhỏ có thể chạy mỗi lần một hoặc hai khối, nhưng khối nào chạy trước vẫn không được hứa hẹn.">
            A small GPU may run one or two blocks at a time, but which one goes first is still not promised.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Có, miễn là lưới là 1D.">Yes, as long as the grid is 1D.</Translated>,
        why: (
          <Translated vi="Hình dạng của lưới không ảnh hưởng gì đến thứ tự chạy các khối của nó.">
            The shape of the grid has no bearing on the order in which its blocks run.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Các khối làm việc trên những phần khác nhau của dữ liệu và có thể chạy theo bất kỳ thứ tự nào, một số chạy cùng
            lúc. Đó là lý do mọi kernel trong chương này giữ cùng một kỷ luật: một luồng chỉ ghi phần tử đầu ra của riêng
            nó và chỉ đọc những đầu vào mà không luồng nào ghi. Không có lần ghi chung nào, thì không thứ tự nào có thể làm
            thay đổi kết quả.
          </p>
        }
      >
        <p>
          Blocks work on different parts of the data and may run in any order, some of them at the same time. That is
          why every kernel in this chapter has the same discipline: a thread writes only its own output element and
          reads only inputs that no thread writes. With no shared writes, no order can change the result.
        </p>
      </Translated>
    ),
    source: 'ch. 2 §2.6, p. 39',
  },
};

export function S30() {
  const vecAdd = useLabFile('ch03/snippets/vec_add_1d.cu');
  return (
    <>
      <Translated
        vi={
          <p>
            Trong PyTorch, một tensor là một hình dạng (shape) và một buffer phẳng. Khi bạn viết <code>x[b, t, h]</code>{' '}
            hay gọi <code>torch.matmul</code>, có hai phép tính số học diễn ra khuất mắt. Một thứ quyết định đơn vị công
            việc nào xử lý phần tử nào, và một thứ biến các chỉ số <code>(b, t, h)</code> thành một offset duy nhất trong buffer. Người viết kernel tự tay viết cả hai. Chương này chính là hai phép tính đó.
          </p>
        }
      >
        <p>
          In PyTorch a tensor is a shape and a flat buffer. When you write <code>x[b, t, h]</code> or call{' '}
          <code>torch.matmul</code>, two pieces of arithmetic happen out of sight. Something decides which unit of work
          handles which element, and something turns the indices <code>(b, t, h)</code> into one offset into the buffer.
          A kernel author writes both by hand. This chapter is those two pieces of arithmetic.
        </p>
      </Translated>
      <Source pages="45" />

      <h2><Translated vi="Chương 2 để lại cho bạn những gì">What chapter 2 left you with</Translated></h2>
      <Translated
        vi={
          <p>
            Một lời gọi kernel khởi chạy một lưới (grid) luồng (thread) một chiều trên các mảng một chiều. Mỗi luồng dựng
            một chỉ số toàn cục duy nhất từ ba biến dựng sẵn, so nó với độ dài, và xử lý một phần tử.
          </p>
        }
      >
        <p>
          One kernel call that launches a one-dimensional grid of threads over one-dimensional arrays. Each thread
          builds a single global index from three built-in variables, tests it against the length, and handles one
          element.
        </p>
      </Translated>
      <Code
        tabs={[{ label: 'CUDA', code: vecAdd ?? tr('// loading…', '// đang tải…'), status: 'front-end-checked', tiers: VEC_TIERS }]}
        highlight={[4, 5]}
        caption={tr(
          'Chapter 2 in two functions. Line 4 is the mapping from a thread to its element and line 5 the test that stops surplus threads. Everything in this chapter is these two lines in more dimensions.',
          'Chương 2 trong hai hàm. Dòng 4 là phép ánh xạ từ một luồng sang phần tử của nó, dòng 5 là phép kiểm tra chặn các luồng dư. Mọi thứ trong chương này là hai dòng đó ở nhiều chiều hơn.',
        )}
      />

      <h2><Translated vi="Chương này thêm gì">What this chapter adds</Translated></h2>
      <Translated
        vi={
          <ul>
            <li>
              <b>Lưới và khối (block) tới ba chiều</b>, đặt bằng hai giá trị <code>dim3</code> lúc khởi chạy, và một biến
              dựng sẵn thứ tư, <code>gridDim</code>, bên cạnh ba biến bạn đã biết (
              <a href={href('ch03', '3.1')}>mục 3.1</a>).
            </li>
            <li>
              <b>Làm phẳng theo thứ tự hàng (row-major)</b>: phần tử <code>[row][col]</code> nằm ở đâu trong một mảng
              phẳng, đó là cách mọi mảng trong chương này đến được kernel của nó (<a href={href('ch03', '3.2')}>mục 3.2</a>).
            </li>
            <li>
              <b>Ba kernel phức tạp dần.</b> Chuyển ảnh màu sang ảnh xám là ánh xạ thuần túy: một luồng, một điểm ảnh, không
              vòng lặp. Làm mờ cho mỗi luồng một vòng lặp qua một vùng lân cận và một phép kiểm tra biên thứ hai bên trong
              nó (
              <a href={href('ch03', '3.3')}>mục 3.3</a>). Phép nhân ma trận cho mỗi luồng trọn một tích vô hướng
              (<a href={href('ch03', '3.4')}>mục 3.4</a>).
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            <b>Grids and blocks of up to three dimensions</b>, set by two <code>dim3</code> values at the launch, and a
            fourth built-in variable, <code>gridDim</code>, beside the three you know (
            <a href={href('ch03', '3.1')}>section 3.1</a>).
          </li>
          <li>
            <b>Row-major flattening</b>: where element <code>[row][col]</code> sits in a flat array, which is how every
            array in this chapter reaches its kernel (<a href={href('ch03', '3.2')}>section 3.2</a>).
          </li>
          <li>
            <b>Three kernels of rising complexity.</b> Colour to grayscale is pure mapping: one thread, one pixel, no
            loop. Blur gives each thread a loop over a neighbourhood and a second boundary test inside it (
            <a href={href('ch03', '3.3')}>section 3.3</a>). Matrix multiplication gives each thread a whole dot product
            (<a href={href('ch03', '3.4')}>section 3.4</a>).
          </li>
        </ul>
      </Translated>
      <Translated
        vi={
          <p>
            Hãy rõ ràng về mục đích của phép nhân ma trận ở đây. Chương 5 mổ xẻ kernel đó vì hiệu năng. Chương này chỉ làm
            cho nó đúng: mọi luồng ghi đúng phần tử từ đúng đầu vào, với mọi kích thước. Đừng đánh giá tốc độ của nó vội.
          </p>
        }
      >
        <p>
          Be clear about what the matrix multiplication here is for. Chapter 5 takes that kernel apart for performance.
          This chapter only gets it correct: every thread writes the right element from the right inputs, for any size.
          Do not judge its speed yet.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              <b>Màu sắc trong khóa học này</b> cho biết dữ liệu nằm ở đâu. Chương này dùng hai trong bốn màu:{' '}
              <span className="chip reg">thanh ghi</span> cho các biến riêng của một luồng và{' '}
              <span className="chip global">bộ nhớ toàn cục</span> cho các mảng. Cùng những màu đó đánh dấu biến trong mã.
              Màu đỏ luôn có nghĩa là có gì đó thất bại.
            </p>
          }
        >
          <p>
            <b>Colour in this course</b> means where data lives. This chapter uses two of the four:{' '}
            <span className="chip reg">register</span> for a thread's private variables and{' '}
            <span className="chip global">global memory</span> for the arrays. The same colours mark variables in code.
            Red always means something failed.
          </p>
        </Translated>
      </Aside>

      <h2><Translated vi="Bạn sẽ làm gì">What you will do</Translated></h2>
      <Translated
        vi={
          <ul>
            <li>Bố trí một lần khởi chạy ở một, hai hoặc ba chiều và nói những hình dạng nào bị thiết bị từ chối.</li>
            <li>Đi từng bước một luồng từ bốn chỉ số của nó đến byte mà nó đọc.</li>
            <li>Tìm ba lỗi trong một đoạn mã kernel như sách in.</li>
            <li>
              Làm <a href={href('ch03', 'labs')}>năm lab</a> trên GPU của chính bạn: ảnh xám, làm mờ, một GEMM ngây thơ cho
              một lớp tuyến tính, mỗi luồng một phần tử so với mỗi luồng một hàng hoặc một cột, và một cuộc săn bốn lỗi chỉ
              số. Lab thứ ba là nửa đầu của mốc dự án M1 trong dự án của khóa học.
            </li>
          </ul>
        }
      >
        <ul>
          <li>Lay out a launch in one, two or three dimensions and say which shapes a device rejects.</li>
          <li>Step one thread from its four indices to the byte it reads.</li>
          <li>Find three errors in a kernel listing as the book prints it.</li>
          <li>
            Work through <a href={href('ch03', 'labs')}>five labs</a> on your own GPU: grayscale, blur, a naive GEMM for
            a linear layer, one thread per element against one per row or column, and a hunt for four index defects.
            The third is the first half of milestone M1 of the course project.
          </li>
        </ul>
      </Translated>

      <h2><Translated vi="Kiểm tra lại chương 2">A check on chapter 2</Translated></h2>
      <Translated
        vi={
          <p>
            Bốn câu hỏi về những gì chương này giả định từ chương 2. Sai bây giờ còn hơn sai về sau; mỗi câu sẽ quay lại
            trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Four questions on what this chapter assumes from chapter 2. They are worth getting wrong now rather than
          later; each one comes back in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={questionsS30['ch03/start/q-globalindex']} label={tr('From chapter 2', 'Từ chương 2')} />
      <Ask q={questionsS30['ch03/start/q-surplus']} label={tr('From chapter 2', 'Từ chương 2')} />
      <Ask q={questionsS30['ch03/start/q-bytes']} label={tr('From chapter 2', 'Từ chương 2')} />
      <Ask q={questionsS30['ch03/start/q-order']} label={tr('From chapter 2', 'Từ chương 2')} />
    </>
  );
}
