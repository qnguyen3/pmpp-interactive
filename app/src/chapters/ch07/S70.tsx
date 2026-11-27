import type { ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// "Before you start" for chapter 7: what the chapter is for, the four kernels it builds,
// one honest bridge to LLM inference, and five prerequisite questions drawn from the
// notes for chapters 3 to 5.

export const questionsS70: Record<string, Question> = {
  'ch07/start/q-blur-corner': {
    kind: 'choice',
    id: 'ch07/start/q-blur-corner',
    prompt: (
      <Translated
        vi={
          <>
            Kernel làm mờ của chương 3 với <code>BLUR_SIZE = 1</code>, so với một phép tích chập (convolution) dùng bộ
            lọc 3×3 có cả chín trọng số đều bằng 1/9 và ô ma bằng không (láng giềng bị thiếu được tính là 0). Hai cách này
            có cho cùng kết quả ở pixel góc (0, 0) không?
          </>
        }
      >
        <>
          Chapter 3&rsquo;s blur kernel with <code>BLUR_SIZE = 1</code>, against a convolution with a 3×3 filter whose
          nine weights are all 1/9 and with zero ghost cells (missing neighbours count as 0). Do the two agree at the
          corner pixel (0, 0)?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có: phép làm mờ cũng tính năm láng giềng bị thiếu là số không và chia tổng cho 9.">
            Yes: the blur also counts the five missing neighbours as zeros and divides its sum by 9.
          </Translated>
        ),
        why: (
          <Translated vi="Phép làm mờ đếm số pixel nó thực sự cộng vào (4 ở góc) và chia cho số đó. Cả hai cộng cùng bốn pixel, nhưng chia cho hai số khác nhau.">
            The blur counts the pixels it actually added (4 at a corner) and divides by that count. They sum the same
            four pixels, but divide by different numbers.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không: phép làm mờ chia tổng cho 4 còn phép tích chập chia cho 9, nên ở đó phép tích chập tối hơn.">
            No: the blur divides its sum by 4 and the convolution by 9, so the convolution is darker there.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không: ở góc, phép tích chập đọc năm pixel nằm ngoài ảnh, nên tổng của nó là rác.">
            No: the convolution reads five pixels outside the image at the corner, so its sum is garbage.
          </Translated>
        ),
        why: (
          <Translated vi="Ô ma bằng không là một giá trị được định nghĩa. Không có gì ngoài ảnh bị đọc; các láng giềng bị thiếu bằng 0 theo định nghĩa.">
            Zero ghost cells are a defined value. Nothing outside the image is read; the missing neighbours are 0 by
            definition.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Kernel làm mờ đếm xem nó thực sự đã cộng bao nhiêu pixel (<code>pixels</code> = 4 ở góc, 6 ở cạnh) và chia
            cho số đó. Một phép tích chập với trọng số cố định 1/9 thực chất cộng thêm năm số không và chia cho 9, nên ở
            góc nó chỉ sáng bằng 4/9. Không cách nào sai cả: một láng giềng bị thiếu đáng giá bao nhiêu là một chính sách
            bạn chọn, và mục 7.1 cho thấy thêm hai chính sách nữa.
          </p>
        }
      >
        <p>
          The blur kernel counts how many pixels it actually added (<code>pixels</code> = 4 at a corner, 6 on an edge)
          and divides by that. A convolution with fixed weights of 1/9 effectively adds five zeros and divides by 9, so
          it is 4/9 as bright at the corner. Neither is wrong: what a missing neighbour is worth is a policy you choose,
          and section 7.1 shows two more.
        </p>
      </Translated>
    ),
    source: 'ch. 3 §3.3, pp. 55–59; ch. 7 §7.1, p. 162',
    vi: { source: 'ch. 3 §3.3, tr. 55–59; ch. 7 §7.1, tr. 162' },
  },
  'ch07/start/q-rowmajor-patch': {
    kind: 'numeric',
    id: 'ch07/start/q-rowmajor-patch',
    prompt: (
      <Translated
        vi={
          <>
            Một ảnh rộng 16 được lưu theo thứ tự hàng (row-major) trong một mảng phẳng. Mảng con 3×3 có tâm tại (hàng 3,
            cột 5) bắt đầu hàng đầu tiên của nó ở chỉ số phẳng nào?
          </>
        }
      >
        <>
          A 16-wide image is stored row-major in a flat array. The 3×3 patch centred on (row 3, column 5) starts its first
          row at which flat index?
        </>
      </Translated>
    ),
    answer: 36,
    explain: (
      <Translated
        vi={
          <p>
            Góc trên bên trái của mảng con nằm cao hơn tâm một hàng và lệch trái một cột: (hàng 2, cột 4). Chỉ số phẳng
            của nó là row·width + column = 2·16 + 4 = 36. Mục 7.2 tính đúng góc này cho mọi luồng (thread), dưới dạng{' '}
            <code>(outRow - r, outCol - r)</code>.
          </p>
        }
      >
        <p>
          The patch&rsquo;s upper-left corner is one row up and one column left of the centre: (row 2, column 4). Its
          flat index is row·width + column = 2·16 + 4 = 36. Section 7.2 computes exactly this corner for every thread, as{' '}
          <code>(outRow - r, outCol - r)</code>.
        </p>
      </Translated>
    ),
    source: 'ch. 3 §3.2, pp. 48–55',
  },
  'ch07/start/q-divergence': {
    kind: 'numeric',
    id: 'ch07/start/q-divergence',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel 1D với 256 luồng mỗi khối chạy trên <code>n = 1000</code> phần tử với lưới (grid) 4 khối, và mỗi
            warp rộng 32 luồng. Có bao nhiêu warp phân kỳ tại <code>if (i &lt; n)</code>?
          </>
        }
      >
        <>
          A 1D kernel with 256 threads per block runs over <code>n = 1000</code> elements with a grid of 4 blocks, and
          warps are 32 threads wide. How many warps diverge at <code>if (i &lt; n)</code>?
        </>
      </Translated>
    ),
    answer: 1,
    unit: 'warps',
    explain: (
      <Translated
        vi={
          <p>
            Lưới có 1024 luồng trong 32 warp. Chỉ warp chứa các phần tử 992 đến 1023 là có cả luồng vượt qua (992 đến
            999) lẫn luồng không vượt qua (1000 đến 1023). Mọi warp trước nó đều vượt qua trọn vẹn. Mục 7.2 đặt cùng câu
            hỏi này cho kernel tích chập, nơi các luồng ở biên bỏ qua số lượng tap của bộ lọc khác nhau.
          </p>
        }
      >
        <p>
          The grid has 1024 threads in 32 warps. Only the warp holding elements 992 to 1023 has both threads that pass
          (992 to 999) and threads that do not (1000 to 1023). Every warp before it passes entirely. Section 7.2 asks the
          same question of the convolution kernel, whose edge threads skip different numbers of filter taps.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.5, pp. 79–82',
    vi: { unit: 'warp' },
  },
  'ch07/start/q-shared-flopb': {
    kind: 'numeric',
    id: 'ch07/start/q-shared-flopb',
    prompt: (
      <Translated
        vi={
          <>
            Một vòng lặp trong tính <code>acc += w_s[k] * x[i + k]</code>, trong đó <code>w_s</code> là một mảng{' '}
            <code>float</code> trong bộ nhớ chia sẻ (shared memory) và <code>x</code> là một mảng <code>float</code> trong
            bộ nhớ toàn cục (global memory). Tỉ lệ giữa số phép toán dấu phẩy động và số byte đọc từ bộ nhớ toàn cục của
            nó là bao nhiêu, chỉ tính vòng lặp?
          </>
        }
      >
        <>
          An inner loop computes <code>acc += w_s[k] * x[i + k]</code>, where <code>w_s</code> is a{' '}
          <code>float</code> array in shared memory and <code>x</code> a <code>float</code> array in global memory. What
          is its ratio of floating-point operations to bytes read from global memory, counting only the loop?
        </>
      </Translated>
    ),
    answer: 0.5,
    tol: 0.001,
    unit: 'FLOP/B',
    explain: (
      <Translated
        vi={
          <p>
            Mỗi lần lặp làm hai phép toán (một phép nhân và một phép cộng) và đọc một <code>float</code> 4 byte từ bộ nhớ
            toàn cục. Lần đọc bộ nhớ chia sẻ không tốn băng thông toàn cục. 2 / 4 = 0.5. Nếu cả hai toán hạng nằm trong bộ
            nhớ toàn cục thì sẽ là 2 / 8 = 0.25, đúng là điểm xuất phát của kernel đầu tiên trong chương này.
          </p>
        }
      >
        <p>
          Each iteration does two operations (a multiply and an add) and reads one 4-byte <code>float</code> from global
          memory. The shared-memory read costs no global bandwidth. 2 / 4 = 0.5. With both operands in global memory it
          would be 2 / 8 = 0.25, which is exactly where this chapter&rsquo;s first kernel starts.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.1–5.2, pp. 93–104',
  },
  'ch07/start/q-constant-copies': {
    kind: 'choice',
    id: 'ch07/start/q-constant-copies',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel dùng <code>__constant__ float table[64]</code> khai báo ở phạm vi tệp,{' '}
            <code>__shared__ float tile[256]</code>, và một biến cục bộ <code>float acc</code>. Nó được khởi chạy trên một
            device với 1000 khối, mỗi khối 256 luồng. Có bao nhiêu bản sao của <code>table</code> tồn tại khi nó chạy?
          </>
        }
      >
        <>
          A kernel uses <code>__constant__ float table[64]</code> declared at file scope,{' '}
          <code>__shared__ float tile[256]</code>, and a local <code>float acc</code>. It is launched on one device with
          1000 blocks of 256 threads. How many copies of <code>table</code> exist while it runs?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Một.">One.</Translated>, correct: true },
      {
        label: <Translated vi="1000, mỗi khối một bản.">1000, one per block.</Translated>,
        why: <Translated vi="Đó là tile: bộ nhớ chia sẻ có phạm vi khối.">That is tile: shared memory has block scope.</Translated>,
      },
      {
        label: <Translated vi="256,000, mỗi luồng một bản.">256,000, one per thread.</Translated>,
        why: (
          <Translated vi="Đó là acc: một biến vô hướng cục bộ nằm trong một thanh ghi của mỗi luồng.">
            That is acc: a local scalar lives in a register of each thread.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Mỗi streaming multiprocessor một bản.">One per streaming multiprocessor.</Translated>,
        why: (
          <Translated vi="Bộ nhớ đệm có thể giữ bản sao các dòng của nó, nhưng bản thân biến chỉ tồn tại một lần trên mỗi device.">
            Caches may hold copies of its lines, but the variable itself exists once per device.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một biến <code>__constant__</code> có phạm vi lưới và sống suốt ứng dụng; tài liệu hướng dẫn của CUDA mô tả nó
            là có &ldquo;một đối tượng riêng cho mỗi device&rdquo;. <code>tile</code> có một bản sao mỗi khối (tổng cộng
            1000) và <code>acc</code> một bản mỗi luồng. Mục 7.4 đặt bộ lọc tích chập vào đúng một biến như vậy.
          </p>
        }
      >
        <p>
          A <code>__constant__</code> variable has grid scope and lives for the whole application; CUDA&rsquo;s guide
          describes it as having &ldquo;a distinct object per device&rdquo;. <code>tile</code> has one copy per block (1000
          in all) and <code>acc</code> one per thread. Section 7.4 puts the convolution filter in exactly such a variable.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.2, pp. 98–104; notes/today/ch07-facts.md A1',
    vi: { source: 'ch. 5 §5.2, tr. 98–104; notes/today/ch07-facts.md A1' },
  },
};

const MAP: { kernel: string; fig: string; fixes: string; flopb: ReactNode; section: string }[] = [
  {
    get kernel() {
      return tr('Basic: one thread per output', 'Cơ bản: mỗi đầu ra một luồng');
    },
    get fig() {
      return tr('Fig. 7.7, p. 166', 'Hình 7.7, tr. 166');
    },
    get fixes() {
      return tr(
        'The starting point: correct at the edges, nothing reused on purpose.',
        'Điểm xuất phát: đúng ở biên, không có gì được tái sử dụng một cách chủ ý.',
      );
    },
    flopb: (
      <Translated vi={<>0.25, bất kể kích thước bộ lọc (tr. 167)</>}>
        <>0.25, whatever the filter size (p. 167)</>
      </Translated>
    ),
    section: '7.2',
  },
  {
    get kernel() {
      return tr('Filter in constant memory', 'Bộ lọc trong bộ nhớ hằng');
    },
    get fig() {
      return tr('Fig. 7.9, p. 170', 'Hình 7.9, tr. 170');
    },
    get fixes() {
      return tr('Filter reads stop costing DRAM bandwidth.', 'Các lần đọc bộ lọc thôi tốn băng thông DRAM.');
    },
    flopb: (
      <Translated vi={<>0.5 (tr. 171)</>}>
        <>0.5 (p. 171)</>
      </Translated>
    ),
    section: '7.4',
  },
  {
    get kernel() {
      return tr('Tiled, halo cells in shared memory', 'Phân ô, ô vùng biên trong bộ nhớ chia sẻ');
    },
    get fig() {
      return tr('Fig. 7.12, p. 174', 'Hình 7.12, tr. 174');
    },
    get fixes() {
      return tr(
        'Each input element is fetched once per block, not once per tap.',
        'Mỗi phần tử đầu vào được nạp một lần cho mỗi khối, không phải một lần cho mỗi tap.',
      );
    },
    flopb: (
      <Translated
        vi={
          <>
            Một công thức theo độ rộng bộ lọc và độ rộng ô, tiến dần tới mức lý tưởng m²/4; 5.42 với bộ lọc 5×5 và ô đầu
            ra 28×28 (tr. 176–177)
          </>
        }
      >
        <>
          A formula in filter width and tile width that approaches the ideal m²/4; 5.42 for a 5×5 filter and 28×28 output
          tiles (pp. 176–177)
        </>
      </Translated>
    ),
    section: '7.5',
  },
  {
    get kernel() {
      return tr('Tiled, halo cells left to the cache', 'Phân ô, ô vùng biên để cho bộ nhớ đệm lo');
    },
    get fig() {
      return tr('Fig. 7.15, p. 178', 'Hình 7.15, tr. 178');
    },
    get fixes() {
      return tr(
        'Input and output tiles the same size, a power of two.',
        'Ô đầu vào và ô đầu ra cùng kích thước, là một lũy thừa của hai.',
      );
    },
    flopb: (
      <Translated vi={<>Sách không đưa ra con số</>}>
        <>The book gives no figure</>
      </Translated>
    ),
    section: '7.6',
  },
];

export function S70() {
  return (
    <>
      <Translated
        vi={
          <p>
            Tích chập là mẫu song song đầu tiên của sách. Nó có mặt khắp xử lý tín hiệu, ảnh và video, và thị giác máy
            tính, thường dưới dạng một bộ lọc; kernel làm mờ của chương 3 là một ví dụ.
          </p>
        }
      >
        <p>
          Convolution is the first of the book&rsquo;s parallel patterns. It runs through signal, image and video
          processing and computer vision, usually as a filter; chapter 3&rsquo;s blur kernel was one.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Chương này mở đầu bằng câu nói rằng bộ lọc Gaussian có thể dùng để làm sắc nét cạnh (tr. 159). Trong các công
              cụ bạn dùng, bộ lọc Gaussian là một phép làm mờ: <code>GaussianBlur</code> của torchvision &ldquo;blurs image
              with randomly chosen Gaussian blur&rdquo; (facts D5). Khóa học này gọi nó là bộ lọc làm mịn.
            </p>
          }
        >
          <p>
            The chapter opens by saying Gaussian filters can be used to sharpen edges (p. 159). In the tools you use a
            Gaussian filter is a blur: torchvision&rsquo;s <code>GaussianBlur</code> &ldquo;blurs image with randomly
            chosen Gaussian blur&rdquo; (facts D5). This course calls it a smoothing filter.
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Đây là một mẫu hữu ích để bắt đầu vì nó vừa dễ vừa khó chịu. Mỗi đầu ra độc lập với mọi đầu ra khác, nên song
            song thừa thãi. Nhưng các đầu ra cạnh nhau đọc gần như cùng các đầu vào, và các đầu ra gần biên cần những đầu
            vào không tồn tại. Xử lý tốt cả hai điều đó là nội dung của chương: phân ô (tiling), và đưa đầu vào lên chip
            trước khi dùng.
          </p>
        }
      >
        <p>
          It is a useful pattern to start with because it is easy and awkward at the same time. Every output is
          independent of every other, so there is parallelism to spare. But neighbouring outputs read almost the same
          inputs, and the outputs near the edges need inputs that do not exist. Handling both well is what the chapter is
          about: tiling, and staging inputs on the chip.
        </p>
      </Translated>
      <Source pages="159" />
      <Translated
        vi={
          <p>
            Trên đường tới suy luận LLM nhanh, bạn sẽ không viết nhiều bộ lọc ảnh. Bạn sẽ dùng hai kỹ thuật mà chương này
            dạy thông qua chúng:
          </p>
        }
      >
        <p>
          You will not write many image filters on the way to fast LLM inference. You will use the two techniques the
          chapter teaches through them:
        </p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>
              <b>bộ nhớ hằng (constant memory)</b> cho một bảng nhỏ chỉ-đọc mà mọi luồng đọc theo cùng một thứ tự;
            </li>
            <li>
              <b>phân ô khi ô đầu vào lớn hơn ô đầu ra</b>: một khối cần một đường viền đầu vào phụ, gọi là{' '}
              <b>vùng biên (halo)</b>, quanh các đầu ra mà nó sở hữu.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            <b>constant memory</b> for a small read-only table that every thread reads in the same order;
          </li>
          <li>
            <b>tiling when the input tile is larger than the output tile</b>: a block needs a border of extra inputs, the{' '}
            <b>halo</b>, around the outputs it owns.
          </li>
        </ul>
      </Translated>
      <Translated
        vi={
          <p>
            Và bạn sẽ luyện phương pháp của chương 5 trên một hình dạng mới: mỗi kernel được đánh giá bằng số phép toán dấu
            phẩy động trên mỗi byte lưu lượng bộ nhớ toàn cục, giờ là một hàm của kích thước bộ lọc và kích thước ô.
          </p>
        }
      >
        <p>
          And you will practise the method from chapter 5 on a new shape: each kernel is judged by its floating-point
          operations per byte of global memory traffic, now as a function of the filter size and the tile size.
        </p>
      </Translated>

      <h2>
        <Translated vi="Bốn kernel">The four kernels</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương này xây một kernel cho mỗi bước. Mỗi hàng cho biết bước đó sửa điều gì và sách tính được gì cho nó. Các
            con số là mô hình của sách, không phải số đo; các lab mới đo.
          </p>
        }
      >
        <p>
          The chapter builds one kernel per step. Each row says what the step fixes and what the book computes for it.
          The numbers are the book&rsquo;s model, not measurements; the labs measure.
        </p>
      </Translated>
      <table className="data wide" data-map style={{ maxWidth: '58rem' }}>
        <thead>
          <tr>
            <th>
              <Translated vi="Kernel">Kernel</Translated>
            </th>
            <th>
              <Translated vi="Trong sách">In the book</Translated>
            </th>
            <th>
              <Translated vi="Nó sửa điều gì">What it fixes</Translated>
            </th>
            <th>
              <Translated vi="FLOP/B, theo cách sách tính">FLOP/B, as the book computes it</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          {MAP.map((m) => (
            <tr key={m.section} data-section={m.section}>
              <td>
                <a href={href('ch07', m.section)}>{m.kernel}</a> <span className="muted">(§{m.section})</span>
              </td>
              <td>{m.fig}</td>
              <td>{m.fixes}</td>
              <td>{m.flopb}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Mục <a href={href('ch07', '7.1')}>7.1</a> định nghĩa thứ mà cả bốn kernel tính, và mục{' '}
            <a href={href('ch07', '7.3')}>7.3</a> giải thích vì sao kernel đầu tiên còn cách xa mức lý tưởng đến vậy. Hai
            lab trên <a href={href('ch07', 'labs')}>trang lab</a> đặt bộ lọc vào bộ nhớ hằng theo ba cách và xây kernel
            phân ô có vùng biên, mỗi lab có một dự đoán để bạn chốt trước khi đo.
          </p>
        }
      >
        <p>
          Section <a href={href('ch07', '7.1')}>7.1</a> defines what all four compute, and section{' '}
          <a href={href('ch07', '7.3')}>7.3</a> explains why the first one is so far from the ideal. Two labs on the{' '}
          <a href={href('ch07', 'labs')}>labs page</a> put the filter in constant memory three ways and build the tiled
          kernel with halos, each with a prediction to commit before you measure.
        </p>
      </Translated>

      <h2>
        <Translated vi="Chương này nằm ở đâu bên dưới các framework bạn dùng">
          Where this sits under the frameworks you use
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            Thứ mà PyTorch gọi là lớp tích chập tính đúng công thức của chương này. <code>torch.nn.Conv2d</code> được tài
            liệu mô tả là &ldquo;valid 2D cross-correlation operator&rdquo;, chính là công thức của sách: nó không lật bộ
            lọc (mục 7.1 cho thấy điều đó nghĩa là gì). Với <code>padding=r</code> và đệm bằng số không, nó là kernel của
            sách với ô ma bằng không.
          </p>
        }
      >
        <p>
          What PyTorch calls a convolution layer computes this chapter&rsquo;s formula. <code>torch.nn.Conv2d</code> is
          documented as the &ldquo;valid 2D cross-correlation operator&rdquo;, which is the book&rsquo;s formula: it does
          not flip the filter (section 7.1 shows what that means). With <code>padding=r</code> and zeros as padding it is
          the book&rsquo;s kernel with zero ghost cells.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Toàn bộ phần này là của khóa học này, được kiểm tra ngày 2026-11-20 với tài liệu của PyTorch 2.15 và mã nguồn
              llama.cpp và MLX (notes/today/ch07-facts.md, D1 đến D3).
            </p>
          }
        >
          <p>
            This whole part is ours, checked on 2026-11-20 against PyTorch 2.15&rsquo;s documentation and the llama.cpp
            and MLX sources (notes/today/ch07-facts.md, D1 to D3).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Các thư viện dùng trong thực tế hiếm khi chạy nó theo cách của sách. Phần lớn chúng biến một phép tích chập
            thành một phép nhân ma trận, là nội dung của chương 19 (<a href={href('ch19', '19.3')}>§19.3</a>):{' '}
            <code>ggml_conv_2d</code> của llama.cpp là <code>ggml_im2col</code>{' '}
            rồi đến <code>ggml_mul_mat</code>, còn MLX chọn giữa Winograd, implicit GEMM, explicit GEMM và một kernel
            depthwise.
          </p>
        }
      >
        <p>
          Production libraries rarely run it the way the book does. They mostly turn a convolution into a matrix
          multiplication, which is chapter 19 (<a href={href('ch19', '19.3')}>§19.3</a>): llama.cpp&rsquo;s <code>ggml_conv_2d</code> is <code>ggml_im2col</code>{' '}
          followed by <code>ggml_mul_mat</code>, and MLX chooses between Winograd, implicit GEMM, explicit GEMM and a
          depthwise kernel.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Trong suy luận LLM, cách dùng trực tiếp của nó rất hẹp: các phần front-end cho thị giác và âm thanh, và phép
            tích chập 1D nhân quả ngắn của các mô hình kiểu Mamba và mô hình lai, mà{' '}
            <a href={href('ch07', '7.5')}>mục 7.5</a> lần theo vào tận llama.cpp: kernel cho prompt dài của nó dùng kỹ
            thuật của mục đó.
          </p>
        }
      >
        <p>
          In LLM inference its direct use is narrow: vision and audio front-ends, and the short causal 1D convolution of
          Mamba-style and hybrid models, which <a href={href('ch07', '7.5')}>section 7.5</a> follows into llama.cpp: its
          kernel for long prompts uses that section&rsquo;s technique.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Những gì mang theo được, và vì sao chương này đáng thời gian của bạn: các giá trị chỉ-đọc mà mọi luồng đọc theo
            cùng một cách (tham số lấy mẫu, hình dạng tensor) nên nằm trong bộ nhớ hằng hoặc trong đối số kernel truyền theo
            giá trị; phân ô có vùng biên áp dụng cho mọi phép toán cửa sổ trượt; và việc cân đối cường độ tính toán với kích
            thước ô chính là cách các ô GEMM được chọn kích thước ở chương 15.
          </p>
        }
      >
        <p>
          What carries over, and why the chapter is worth your time: read-only values that every thread reads the same
          way (sampling parameters, tensor shapes) belong in constant memory or in kernel arguments passed by value;
          halo tiling applies to any sliding-window operation; and the bookkeeping of arithmetic intensity against tile
          size is exactly how GEMM tiles are sized in chapter 15.
        </p>
      </Translated>

      <h2>
        <Translated vi="Trước khi bắt đầu">Before you start</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Năm câu hỏi về những gì chương này giả định từ chương 3 đến 5. Mỗi câu ghi rõ trang nguồn, và mỗi câu sẽ quay
            lại trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Five questions on what this chapter assumes from chapters 3 to 5. Each one names the page it comes from, and
          each comes back in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={questionsS70['ch07/start/q-blur-corner']} label={tr('From chapter 3', 'Từ chương 3')} />
      <Ask q={questionsS70['ch07/start/q-rowmajor-patch']} label={tr('From chapter 3', 'Từ chương 3')} />
      <Ask q={questionsS70['ch07/start/q-divergence']} label={tr('From chapter 4', 'Từ chương 4')} />
      <Ask q={questionsS70['ch07/start/q-shared-flopb']} label={tr('From chapter 5', 'Từ chương 5')} />
      <Ask q={questionsS70['ch07/start/q-constant-copies']} label={tr('From chapter 5', 'Từ chương 5')} />
    </>
  );
}
