import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// Prerequisite questions drawn from the notes of chapters 1 to 3.
export const questionsS40: Record<string, Question> = {
  'ch04/start/q-surplus': {
    kind: 'numeric',
    id: 'ch04/start/q-surplus',
    prompt: (
      <Translated
        vi={
          <>
            Một vector 1000 phần tử được cộng bằng <code>vecAddKernel</code>, khởi chạy theo các khối 256 luồng. Bao nhiêu
            luồng đã khởi chạy không có phần tử nào để xử lý?
          </>
        }
      >
        A vector of 1000 elements is added with <code>vecAddKernel</code>, launched in blocks of 256 threads. How many
        of the launched threads have no element to work on?
      </Translated>
    ),
    answer: 24,
    unit: 'threads',
    explain: (
      <Translated
        vi={
          <p>
            Lần khởi chạy cần <code>ceil(1000 / 256) = 4</code> khối, tức 1024 luồng cho 1000 phần tử. 24 luồng cuối không
            qua được phép kiểm tra <code>if (i &lt; n)</code> và không làm gì. Mục 4.5 hỏi 24 luồng đó gây tốn kém gì cho
            những luồng dùng chung phần cứng với chúng.
          </p>
        }
      >
        <p>
          The launch needs <code>ceil(1000 / 256) = 4</code> blocks, which is 1024 threads for 1000 elements. The last 24
          fail the test <code>if (i &lt; n)</code> and do nothing. Section 4.5 asks what those 24 cost the threads that
          share hardware with them.
        </p>
      </Translated>
    ),
    source: 'ch. 3 §3.1, p. 46 and §3.2, p. 50',
    vi: { unit: 'luồng', source: 'ch. 3 §3.1, p. 46 và §3.2, p. 50' },
  },
  'ch04/start/q-region4': {
    kind: 'numeric',
    id: 'ch04/start/q-region4',
    prompt: (
      <Translated
        vi={
          <>
            Chương 3 phủ một bức ảnh cao 62 pixel, rộng 76 pixel bằng các khối 16 × 16, bốn hàng năm khối. Bao nhiêu luồng
            của khối dưới cùng bên phải qua được phép kiểm tra phạm vi <code>col &lt; width &amp;&amp; row &lt; height</code>?
          </>
        }
      >
        Chapter 3 covered a picture 62 pixels high and 76 wide with 16 × 16 blocks, four rows of five. How many threads
        of the bottom-right block pass the range test <code>col &lt; width &amp;&amp; row &lt; height</code>?
      </Translated>
    ),
    answer: 168,
    unit: 'threads',
    explain: (
      <Translated
        vi={
          <p>
            Cột khối cuối cùng bắt đầu ở cột pixel 64, nên 76 − 64 = 12 trong 16 cột luồng của nó có pixel. Hàng khối cuối
            cùng bắt đầu ở hàng pixel 48, nên 62 − 48 = 14 trong 16 hàng luồng của nó có pixel. Khối ở góc chịu cả hai giới
            hạn: 12 × 14 = 168 trên 256. Bạn sẽ đếm lại khối này ở mục 4.5, lần này theo từng warp.
          </p>
        }
      >
        <p>
          The last column of blocks starts at pixel column 64, so 76 − 64 = 12 of its 16 thread columns hold a pixel. The
          last row of blocks starts at pixel row 48, so 62 − 48 = 14 of its 16 thread rows do. The corner block has both
          limits: 12 × 14 = 168 of 256. You will count this block again in section 4.5, this time warp by warp.
        </p>
      </Translated>
    ),
    source: 'ch. 3 §3.2, pp. 54–55 (Fig. 3.5)',
    vi: { unit: 'luồng' },
  },
  'ch04/start/q-flat': {
    kind: 'numeric',
    id: 'ch04/start/q-flat',
    prompt: (
      <Translated
        vi={
          <>
            Vẫn bức ảnh đó, rộng 76 pixel, khối 16 × 16. Luồng có <code>threadIdx = (x 3, y 2)</code> trong khối có{' '}
            <code>blockIdx = (x 1, y 0)</code> ghi một phần tử của <code>Pout</code>. Chỉ số phẳng của nó là bao nhiêu?
          </>
        }
      >
        Same picture, 76 pixels wide, 16 × 16 blocks. The thread with <code>threadIdx = (x 3, y 2)</code> in the block
        with <code>blockIdx = (x 1, y 0)</code> writes one element of <code>Pout</code>. What is its flat index?
      </Translated>
    ),
    answer: 171,
    explain: (
      <Translated
        vi={
          <p>
            <code>row = blockIdx.y * blockDim.y + threadIdx.y = 0 * 16 + 2 = 2</code> và{' '}
            <code>col = blockIdx.x * blockDim.x + threadIdx.x = 1 * 16 + 3 = 19</code>. Thứ tự hàng (row-major):{' '}
            <code>row * width + col</code> = 2 × 76 + 19 = 171. Mục 4.4 dùng đúng cách làm phẳng này, x thay đổi nhanh
            nhất, để quyết định những luồng nào của một khối chung một warp.
          </p>
        }
      >
        <p>
          <code>row = blockIdx.y * blockDim.y + threadIdx.y = 0 * 16 + 2 = 2</code> and{' '}
          <code>col = blockIdx.x * blockDim.x + threadIdx.x = 1 * 16 + 3 = 19</code>. Row-major: <code>row * width + col</code>{' '}
          = 2 × 76 + 19 = 171. Section 4.4 uses the same flattening, x fastest, to decide which threads of a block share a
          warp.
        </p>
      </Translated>
    ),
    source: 'ch. 3 §3.2, p. 53',
  },
  'ch04/start/q-order': {
    kind: 'choice',
    id: 'ch04/start/q-order',
    prompt: (
      <Translated vi={<>Một lưới có hai khối, 0 và 1. Kernel được phép giả định gì về việc khối nào chạy trước?</>}>
        <>A grid has two blocks, 0 and 1. What may the kernel assume about which of them runs first?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Khối 0 chạy trước: các khối bắt đầu theo thứ tự chỉ số.">
            Block 0 runs first: blocks start in index order.
          </Translated>
        ),
        why: (
          <Translated vi="Mô hình không nói điều đó ở đâu cả, và mục 4.2 cho thấy vì sao không thể như vậy.">
            Nothing in the model says so, and section 4.2 shows why it cannot.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Chúng luôn chạy cùng lúc.">They always run at the same time.</Translated>,
        why: (
          <Translated vi="Có thể. Chúng cũng có thể chạy lần lượt, theo thứ tự nào cũng được.">
            They may. They may also run one after the other, in either order.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không gì cả. Runtime có thể chạy chúng theo thứ tự nào cũng được, hoặc cùng lúc.">
            Nothing. The runtime may run them in either order, or together.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Thứ tự cố định cho mỗi thiết bị, nên có thể đo một lần rồi dựa vào đó.">
            The order is fixed per device, so it can be measured once and relied on.
          </Translated>
        ),
        why: (
          <Translated vi="Một lần thử trên một thiết bị cho thấy điều đã xảy ra lần đó, không phải điều được bảo đảm.">
            A test on one device shows what happened that time, not what is promised.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Chương 2 nêu nó thành một quy tắc cho lập trình viên: các khối làm việc trên những phần dữ liệu khác nhau và có
            thể thực thi theo bất kỳ thứ tự nào. Chương này đưa ra lý do quy tắc ấy tồn tại, và bạn nhận được gì khi tuân
            theo nó.
          </p>
        }
      >
        <p>
          Chapter 2 stated it as a rule for the programmer: blocks work on different parts of the data and may execute in
          any order. This chapter gives the reason the rule exists, and what you get in exchange for obeying it.
        </p>
      </Translated>
    ),
    source: 'ch. 2 §2.6, p. 39',
  },
  'ch04/start/q-throughput': {
    kind: 'choice',
    id: 'ch04/start/q-throughput',
    prompt: (
      <Translated
        vi={
          <>
            CPU dành phần lớn diện tích cho bộ nhớ đệm (cache) lớn và bộ dự đoán rẽ nhánh. GPU thì dành nó cho rất nhiều lõi
            (core) đơn giản. Vì sao đó là một sự đánh đổi hợp lý cho GPU?
          </>
        }
      >
        A CPU spends much of its area on large caches and branch predictors. A GPU spends it on many simple cores
        instead. Why is that a sound trade for a GPU?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Cache và bộ dự đoán chủ yếu cắt giảm độ trễ của một luồng, còn GPU được xây dựng cho thông lượng.">
            Caches and predictors mostly cut one thread's latency, and a GPU is built for throughput.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chương trình GPU hiếm khi rẽ nhánh và hiếm khi dùng lại dữ liệu, nên cache và bộ dự đoán sẽ nằm không.">
            GPU programs rarely branch and rarely reuse data, so caches and predictors would sit idle.
          </Translated>
        ),
        why: (
          <Translated vi="Kernel có rẽ nhánh (mỗi phép kiểm tra phạm vi là một nhánh) và dùng lại dữ liệu liên tục.">
            Kernels branch (every range test is one) and reuse data constantly.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Bộ nhớ GPU nhanh đến mức một cache đặt trước nó sẽ chẳng thêm được gì mà kernel đo được.">
            GPU memory is so fast that a cache in front of it would add nothing a kernel could measure.
          </Translated>
        ),
        why: (
          <Translated vi="Ngược lại: một lần truy cập bộ nhớ toàn cục (global memory) chính là thao tác chậm của chương này.">
            The opposite: an access to global memory is the slow operation of this chapter.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Lõi đơn giản chạy mỗi luồng nhanh hơn một lõi phức tạp, nên diện tích được dùng hiệu quả hơn.">
            Simple cores run each thread faster than a complex core would, so the area is better spent.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi luồng chạy chậm hơn. Chỉ là có nhiều luồng hơn hẳn.">
            Each thread runs slower. There are many more of them.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lập luận của chương 1: giảm một nửa độ trễ tốn diện tích và điện năng hơn nhiều so với tăng gấp đôi thông lượng
            bằng cách thêm đơn vị. Sự đánh đổi này chỉ có tác dụng nếu phần cứng luôn có một luồng khác để chạy trong khi
            một luồng đang chờ. Mục 4.6 cho thấy cơ chế: một SM giữ nhiều luồng hơn hẳn số nó có thể thực thi và chuyển
            qua lại giữa chúng mà không tốn chi phí.
          </p>
        }
      >
        <p>
          Chapter 1's argument: halving latency costs far more area and power than doubling throughput by adding units.
          The trade only works if the hardware always has another thread to run while one waits. Section 4.6 shows the
          mechanism: an SM keeps many more threads than it can execute and switches among them at no cost.
        </p>
      </Translated>
    ),
    source: 'ch. 1 §1.1, pp. 4–5, recalled on p. 67',
    vi: { source: 'ch. 1 §1.1, pp. 4–5, nhắc lại ở p. 67' },
  },
};

export function S40() {
  const q = questionsS40;
  return (
    <>
      <Translated
        vi={
          <p>
            Chương 2 và 3 đã dạy bạn khởi chạy một lưới (grid): bấy nhiêu khối (block), mỗi khối bấy nhiêu luồng (thread),
            mỗi luồng tự tìm phần dữ liệu của mình. Chúng gần như không nói gì về việc thiết bị làm gì với lưới đó khi đã
            nhận được nó. Chương này nói. Sau chương này, hai con số trong <code>&lt;&lt;&lt;grid, block&gt;&gt;&gt;</code>{' '}
            là những quyết định bạn có thể lập luận được, chứ không phải giá trị chép từ một ví dụ.
          </p>
        }
      >
        <p>
          Chapters 2 and 3 taught you to launch a grid: so many blocks of so many threads, each thread finding its own
          piece of the data. They said almost nothing about what the device does with that grid once it has it. This
          chapter does. After it, the two numbers in <code>&lt;&lt;&lt;grid, block&gt;&gt;&gt;</code> are decisions you
          can reason about, not values copied from an example.
        </p>
      </Translated>
      <Source pages="67" />
      <Translated vi={<p>Chương này đi theo một lưới từ trên xuống, theo thứ tự các trang của sách:</p>}>
        <p>The chapter follows a grid downwards, in the order of the book's pages:</p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>
              <b>khối lên phần cứng</b>: thiết bị được tạo nên từ những gì (<a href={href('ch04', '4.1')}>4.1</a>) và
              runtime giao cho nó trọn từng khối như thế nào (<a href={href('ch04', '4.2')}>4.2</a>);
            </li>
            <li>
              <b>rào chắn (barrier)</b>, và vì sao các khối có thể chạy theo bất kỳ thứ tự nào (
              <a href={href('ch04', '4.3')}>4.3</a>);
            </li>
            <li>
              <b>khối được cắt thành warp</b>, những nhóm luồng thực thi cùng nhau (
              <a href={href('ch04', '4.4')}>4.4</a>);
            </li>
            <li>
              <b>warp tại một nhánh rẽ</b>: tốn kém gì khi các luồng của một warp đi theo những hướng khác nhau (
              <a href={href('ch04', '4.5')}>4.5</a>);
            </li>
            <li>
              <b>warp chờ bộ nhớ</b>, và vì sao một thiết bị giữ nhiều warp hơn số nó có thể chạy (
              <a href={href('ch04', '4.6')}>4.6</a>);
            </li>
            <li>
              <b>vừa được bao nhiêu</b>: những giới hạn tài nguyên quyết định điều đó (
              <a href={href('ch04', '4.7')}>4.7</a>);
            </li>
            <li>
              <b>hỏi thiết bị</b> về các kích thước của nó trước khi bạn chọn cấu hình khởi chạy (
              <a href={href('ch04', '4.8')}>4.8</a>).
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            <b>blocks onto hardware</b>: what the device is made of (<a href={href('ch04', '4.1')}>4.1</a>) and how the
            runtime hands it whole blocks (<a href={href('ch04', '4.2')}>4.2</a>);
          </li>
          <li>
            <b>barriers</b>, and why blocks may run in any order (<a href={href('ch04', '4.3')}>4.3</a>);
          </li>
          <li>
            <b>blocks cut into warps</b>, the groups of threads that execute together (
            <a href={href('ch04', '4.4')}>4.4</a>);
          </li>
          <li>
            <b>warps at a branch</b>: what it costs when threads of one warp disagree (
            <a href={href('ch04', '4.5')}>4.5</a>);
          </li>
          <li>
            <b>warps waiting on memory</b>, and why a device keeps more of them than it can run (
            <a href={href('ch04', '4.6')}>4.6</a>);
          </li>
          <li>
            <b>how many fit</b>: the budgets that decide it (<a href={href('ch04', '4.7')}>4.7</a>);
          </li>
          <li>
            <b>asking the device</b> for its sizes before you choose a launch (<a href={href('ch04', '4.8')}>4.8</a>).
          </li>
        </ul>
      </Translated>
      <Translated
        vi={
          <p>
            Tiếp theo là bộ nhớ: chương 5 là bộ nhớ trên chip, chương 6 là bộ nhớ ngoài chip và việc tinh chỉnh rút ra từ cả
            hai.
          </p>
        }
      >
        <p>
          Memory comes next: chapter 5 is the memory on the chip, chapter 6 the memory off it and the tuning that follows
          from both.
        </p>
      </Translated>

      <h2>
        <Translated vi="Chương này nằm ở đâu bên dưới một lệnh gọi framework">
          Where this sits under a framework call
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            Khi <code>torch.matmul</code> chạy trên một tensor GPU, một kernel nào đó làm việc này, và người viết kernel đó
            đã chọn kích thước khối và lưới cho nó. Những lựa chọn ấy được đưa ra bằng chính vốn từ của chương này, và bạn
            có thể thấy nó trong mã nguồn mà khóa học này đang hướng tới. Các kernel CUDA của PyTorch dùng một hằng{' '}
            <code>C10_WARP_SIZE</code>. Backend CUDA của llama.cpp có <code>#define WARP_SIZE 32</code>. Softmax Metal của
            MLX có <code>constexpr int SIMD_SIZE = 32</code>. Mỗi dòng đó là một khẳng định về cách một thiết bị nhóm các
            luồng để thực thi. Đến mục 4.4 bạn sẽ nói được mỗi dòng giả định điều gì, và đến mục 4.8 là lẽ ra nó nên hỏi
            thiết bị điều gì.
          </p>
        }
      >
        <p>
          When <code>torch.matmul</code> runs on a GPU tensor, some kernel does the work, and whoever wrote that kernel
          chose a block size and a grid for it. Those choices are made in this chapter's vocabulary, and you can see it in
          the sources this course is heading for. PyTorch's CUDA kernels use a constant <code>C10_WARP_SIZE</code>.
          llama.cpp's CUDA backend has <code>#define WARP_SIZE 32</code>. MLX's Metal softmax has{' '}
          <code>constexpr int SIMD_SIZE = 32</code>. Each of those lines is a statement about how a device groups threads
          for execution. By section 4.4 you will be able to say what each one assumes, and by section 4.8 what it should
          have asked the device instead.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không nhắc đến PyTorch, llama.cpp hay MLX. Ba hằng số trên được đọc trong mã nguồn của các dự án vào ngày
              2026-11-20 (notes/today/ch04-facts.md, D1, D3, D4).
            </p>
          }
        >
          <p>
            The book does not mention PyTorch, llama.cpp or MLX. The three constants were read in the projects' sources on
            2026-11-20 (notes/today/ch04-facts.md, D1, D3, D4).
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Ba từ, ba nhà cung cấp">Three words, three vendors</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách dạy CUDA, và từ ngữ của nó là của NVIDIA. Cùng ba ý tưởng này xuyên suốt cả chương trên mọi backend, dưới
            những cái tên khác nhau:
          </p>
        }
      >
        <p>
          The book teaches CUDA, and its words are NVIDIA's. The same three ideas carry this whole chapter on every
          backend, under different names:
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Việc đặt các cột cạnh nhau là của khóa học này. Không nhà cung cấp nào vẽ ra những tương đương này, và hàng ở
              giữa chỉ tương đương lỏng lẻo: mục 4.1 nói rõ vì sao. Các tên gọi đã được đối chiếu với tài liệu của nhà cung
              cấp (notes/today/ch04-facts.md, B1, B2, C2).
            </p>
          }
        >
          <p>
            Setting the columns side by side is ours. No vendor draws these equivalences, and the middle row is loose:
            section 4.1 says how. Names checked against vendor documentation (notes/today/ch04-facts.md, B1, B2, C2).
          </p>
        </Translated>
      </Beyond>
      <table
        className="data wide"
        style={{ maxWidth: '58rem' }}
        aria-label={tr('Vocabulary across vendors', 'Từ vựng giữa các nhà cung cấp')}
      >
        <thead>
          <tr>
            <th>
              <Translated vi="Nó là gì">What it is</Translated>
            </th>
            <th>
              <Translated vi="CUDA (sách)">CUDA (the book)</Translated>
            </th>
            <th>
              <Translated vi="HIP trên AMD">HIP on AMD</Translated>
            </th>
            <th>Metal</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <Translated vi="Nhóm luồng bạn khởi chạy cùng nhau, có thể chia sẻ bộ nhớ và gặp nhau tại một rào chắn">
                The group of threads you launch together, which can share memory and meet at a barrier
              </Translated>
            </td>
            <td>
              <Translated vi="block">block</Translated>
            </td>
            <td>
              <Translated vi="block trong mã HIP; workgroup trong tài liệu phần cứng của AMD">
                block in HIP code; workgroup in AMD's hardware documentation
              </Translated>
            </td>
            <td>
              <Translated vi="threadgroup">threadgroup</Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi="Phần cứng mà một nhóm như vậy chạy trên đó">
                The piece of hardware one such group runs on
              </Translated>
            </td>
            <td>
              <Translated vi="streaming multiprocessor (SM)">streaming multiprocessor (SM)</Translated>
            </td>
            <td>
              <Translated vi="compute unit (CU)">compute unit (CU)</Translated>
            </td>
            <td>
              <Translated vi="không được công bố trong tài liệu">not documented</Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi="Các luồng trong một nhóm cùng thực thi một lệnh">
                The threads inside a group that execute one instruction together
              </Translated>
            </td>
            <td>
              <Translated vi="warp">warp</Translated>
            </td>
            <td>
              <Translated vi="warp trong tài liệu của HIP; wavefront theo thuật ngữ tập lệnh của AMD">
                warp in HIP's documentation; wavefront in AMD's instruction-set terms
              </Translated>
            </td>
            <td>
              <Translated vi="SIMD-group">SIMD-group</Translated>
            </td>
          </tr>
        </tbody>
      </table>

      <h2>
        <Translated vi="Bạn sẽ đo những gì">What you will measure</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bốn <a href={href('ch04', 'labs')}>bài thực hành</a> đi kèm chương này: hỏi thiết bị về các kích thước của nó
            và suy ra cấu hình khởi chạy từ đó; đo cái giá của một warp bị phân kỳ; săn ba lỗi quanh một rào chắn; và quét
            kích thước khối của một kernel dành thời gian chờ bộ nhớ.
          </p>
        }
      >
        <p>
          Four <a href={href('ch04', 'labs')}>labs</a> go with the chapter: ask the device for its sizes and derive a
          launch from them; measure what a divergent warp costs; hunt three defects around a barrier; and sweep the block size of a
          kernel that spends its time waiting on memory.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Một giới hạn cần biết trước khi bắt đầu. Các trình giả lập (emulator) CPU chạy logic của kernel: chúng bắt được
            kết quả sai, rào chắn đặt sai chỗ, race condition. Chúng không chạy các luồng của một warp đồng bộ từng bước và
            không chờ bộ nhớ, nên không thể cho thấy phân kỳ luồng (divergence) hay occupancy (mức chiếm dụng) thấp tốn kém
            ra sao. Hai phép đo đó cần một GPU, và các con số bạn nhận được thuộc về GPU đó.
          </p>
        }
      >
        <p>
          One limit to know before you start. The CPU emulators run your kernel's logic: they catch wrong results, a
          barrier in the wrong place, a race. They do not run the threads of a warp in step and they do not wait on memory, so they cannot
          show what divergence or low occupancy costs. Those two measurements need a GPU, and the numbers you get belong
          to that GPU.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              <code>python3 labs/pmpp.py doctor</code> liệt kê các backend máy này chạy được. Mọi điều một bài học nói về
              "thiết bị của bạn" đều lấy từ báo cáo đó.
            </p>
          }
        >
          <p>
            <code>python3 labs/pmpp.py doctor</code> lists the backends this machine can run. Anything a lesson says about
            "your device" comes from that report.
          </p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Trước khi bắt đầu">Before you start</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Năm câu hỏi về những gì chương này giả định từ chương 1 đến 3. Mỗi câu đều được dùng lại ở phần sau của chương,
            và mỗi câu đều quay lại trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Five questions on what this chapter assumes from chapters 1 to 3. Each one is used again later in the chapter,
          and each comes back in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={q['ch04/start/q-surplus']} label={tr('From chapter 3', 'Từ chương 3')} />
      <Ask q={q['ch04/start/q-region4']} label={tr('From chapter 3', 'Từ chương 3')} />
      <Ask q={q['ch04/start/q-flat']} label={tr('From chapter 3', 'Từ chương 3')} />
      <Ask q={q['ch04/start/q-order']} label={tr('From chapter 2', 'Từ chương 2')} />
      <Ask q={q['ch04/start/q-throughput']} label={tr('From chapter 1', 'Từ chương 1')} />
    </>
  );
}
