import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { basicIntensity } from '../../sim/convai';
import { offset, phases, printedPhases } from '../../sim/convlib';

// Prerequisite questions drawn from the notes of chapters 3, 5, 7 and 15. Each explanation
// names where the idea was taught and the section of this chapter that builds on it. The
// numeric answers come from the tested models (convlib, convai), not from typed numbers.
const LENET_INNER = 25; // LeNet C1: C·K·K = 1·5·5 (notes, §19.3 derived)
const TILE = 16;

export const questionsS190: Record<string, Question> = {
  'ch19/start/q-index4d': {
    kind: 'numeric',
    id: 'ch19/start/q-index4d',
    prompt: (
      <Translated
        vi={
          <>
            <code>X</code> chứa một batch ảnh dưới dạng mảng <code>N × C × H × W</code>, lưu theo thứ tự hàng (row-major)
            trong một <code>float*</code> phẳng, với <code>C = 3</code> và <code>H = W = 32</code>. Chỉ số phẳng của{' '}
            <code>X[1][2][5][7]</code> là bao nhiêu?
          </>
        }
      >
        <>
          <code>X</code> holds a batch of images as an <code>N × C × H × W</code> array, stored row-major in a flat{' '}
          <code>float*</code>, with <code>C = 3</code> and <code>H = W = 32</code>. What is the flat index of{' '}
          <code>X[1][2][5][7]</code>?
        </>
      </Translated>
    ),
    answer: offset('NCHW', { C: 3, H: 32, W: 32 }, 1, 2, 5, 7),
    explain: (
      <Translated
        vi={
          <p>
            Bóc từng chiều một: <code>((n·C + c)·H + h)·W + w = ((1·3 + 2)·32 + 5)·32 + 7 = 5287</code>. Chương này viết{' '}
            <code>X[n, c, h + p, w + q]</code> dưới dạng mã giả và để phần chuyển đổi này cho bạn (mục{' '}
            <a href={href('ch19', '19.2')}>19.2</a>), và dòng 28 của Fig. 19.11 chính là biểu thức này với ánh xạ unfold
            nằm bên trong (mục <a href={href('ch19', '19.3')}>19.3</a>). Đã học ở{' '}
            <a href={href('ch03', '3.2')}>ch. 3 §3.2</a>.
          </p>
        }
      >
        <p>
          Peel off one dimension at a time: <code>((n·C + c)·H + h)·W + w = ((1·3 + 2)·32 + 5)·32 + 7 = 5287</code>. The
          chapter writes <code>X[n, c, h + p, w + q]</code> as pseudo-code and leaves this translation to you (section{' '}
          <a href={href('ch19', '19.2')}>19.2</a>), and line 28 of Fig. 19.11 is this very expression with the unfolding
          map inside it (section <a href={href('ch19', '19.3')}>19.3</a>). Taught in{' '}
          <a href={href('ch03', '3.2')}>ch. 3 §3.2</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 3 §3.2, p. 55 (the 3D form), and §3.4; ch. 19 pp. 461, 463, 471',
    vi: { source: 'ch. 3 §3.2, p. 55 (dạng 3D), và §3.4; ch. 19 pp. 461, 463, 471' },
  },
  'ch19/start/q-reuse': {
    kind: 'choice',
    id: 'ch19/start/q-reuse',
    prompt: (
      <Translated
        vi={
          <>
            Một GEMM phân ô (tiling) với các ô <code>T × T</code> tính một đầu ra <code>j × l</code> là{' '}
            <code>Y = A · B</code>. Bỏ qua bộ nhớ đệm (cache), mỗi phần tử của <code>A</code> được nạp từ bộ nhớ toàn cục
            (global memory) bao nhiêu lần?
          </>
        }
      >
        <>
          A tiled GEMM with <code>T × T</code> tiles computes a <code>j × l</code> output <code>Y = A · B</code>. Ignoring
          caches, how many times is each element of <code>A</code> fetched from global memory?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi={<>ceil(l / T): mỗi khối trong hàng khối của nó nạp một lần.</>}><>ceil(l / T): once by each block in its row of blocks.</></Translated>, correct: true },
      {
        label: (
          <Translated vi="Một lần: sau khi khối đầu tiên nạp nó, bộ nhớ chia sẻ (shared memory) phục vụ mọi khối khác.">
            Once: after the first block loads it, shared memory serves every other block.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là trường hợp lý tưởng. Mỗi khối cần phần tử đó đều nạp nó vào bộ nhớ chia sẻ của riêng mình, và các khối không chia sẻ với nhau.">
            That is the ideal. Every block that needs the element loads it into its own shared memory, and blocks do not share.
          </Translated>
        ),
      },
      {
        label: <Translated vi={<>ceil(j / T): mỗi khối trong cột khối của nó nạp một lần.</>}><>ceil(j / T): once by each block in its column of blocks.</></Translated>,
        why: (
          <Translated vi="Cách đó đếm số khối trong một cột của đầu ra. Một phần tử của A cần dùng dọc theo hàng của nó trong Y, hàng này cắt qua các cột.">
            That counts the blocks in one column of the output. An element of A is needed along its row of Y, which crosses the columns.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="T lần: một lần trong mỗi pha trong T pha của khối.">
            T times: once in each of the T phases of its block.
          </Translated>
        ),
        why: (
          <Translated vi="Một khối nạp một phần tử cho trước của A chỉ trong một pha. Phân ô chia số lần nạp cho T so với cách mỗi luồng một đầu ra; phần còn lại là một lần nạp cho mỗi khối cần phần tử đó.">
            A block loads a given element of A in one phase only. Tiling divides the fetches by T compared with one thread per output; what is left is one fetch per block that needs the element.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một phần tử ở hàng r của <code>A</code> góp vào mọi đầu ra ở hàng r của <code>Y</code>, và các đầu ra đó trải
            trên ceil(l / T) khối. Mỗi khối nạp nó một lần. Với lớp đầu tiên của LeNet dưới dạng GEMM, một đầu ra 6 × 784
            với T = 16, mỗi trọng số bộ lọc được nạp {phases(784, TILE)} lần. Ô lớn hơn và ma trận lớn hơn thì lợi hơn, đó
            là lập luận mà mục <a href={href('ch19', '19.3')}>19.3</a> kết thúc và mục{' '}
            <a href={href('ch15', '15.2')}>15.2</a> đẩy xa hơn. Đã học ở <a href={href('ch05', '5.3')}>ch. 5 §5.3</a>{' '}
            và <a href={href('ch15', '15.2')}>ch. 15 §15.2</a>.
          </p>
        }
      >
        <p>
          An element in row r of <code>A</code> feeds every output in row r of <code>Y</code>, and those outputs are spread
          over ceil(l / T) blocks. Each of them fetches it once. For LeNet&rsquo;s first layer as a GEMM, a 6 × 784 output
          with T = 16, every filter weight is fetched {phases(784, TILE)} times. Larger tiles and larger matrices pay
          better, which is the argument section <a href={href('ch19', '19.3')}>19.3</a> ends on and section{' '}
          <a href={href('ch15', '15.2')}>15.2</a> pushes further. Taught in <a href={href('ch05', '5.3')}>ch. 5 §5.3</a>{' '}
          and <a href={href('ch15', '15.2')}>ch. 15 §15.2</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.3, pp. 104–107; ch. 15 §15.2, pp. 351–352; ch. 19 p. 472', // i18n-ignore: citation
  },
  'ch19/start/q-phases': {
    kind: 'numeric',
    id: 'ch19/start/q-phases',
    prompt: (
      <Translated
        vi={
          <>
            Một GEMM phân ô với <code>T = 16</code>, viết kèm các kiểm tra biên của mục 5.5, nhân một ma trận 6 × 25 với
            một ma trận 25 × 784. Vòng lặp pha của nó chạy bao nhiêu lần?
          </>
        }
      >
        <>
          A tiled GEMM with <code>T = 16</code>, written with the boundary checks of section 5.5, multiplies a 6 × 25
          matrix by a 25 × 784 matrix. How many times does its phase loop run?
        </>
      </Translated>
    ),
    answer: phases(LENET_INNER, TILE),
    unit: 'phases',
    vi: { unit: 'pha' },
    explain: (
      <Translated
        vi={
          <p>
            ceil(25 / 16) = 2. Ở pha thứ hai chỉ có 9 trong 16 cột của ô bên trái, và 9 trong 16 hàng của ô bên phải, là
            tồn tại; các phép kiểm tra lấp 7 ô nhớ còn lại bằng số 0, vốn không cộng thêm gì. Hai ma trận này là lớp đầu
            tiên của LeNet, bộ lọc <code>F</code> và đầu vào đã unfold <code>B</code>. Fig. 19.11, như bản in, chạy{' '}
            <code>(C*K*K)/TILE_WIDTH</code> pha, ở đây bị cắt cụt còn {printedPhases(LENET_INNER, TILE)} và lặng lẽ bỏ mất{' '}
            {LENET_INNER - printedPhases(LENET_INNER, TILE) * TILE} trong 25 số hạng (mục{' '}
            <a href={href('ch19', '19.3')}>19.3</a>). Đã học ở <a href={href('ch05', '5.5')}>ch. 5 §5.5</a>.
          </p>
        }
      >
        <p>
          ceil(25 / 16) = 2. In the second phase only 9 of the 16 columns of the left tile, and 9 of the 16 rows of the
          right one, exist; the checks fill the other 7 slots with zeros, which add nothing. These two matrices are
          LeNet&rsquo;s first layer, the filter bank <code>F</code> and the unfolded input <code>B</code>. Fig. 19.11, as
          printed, runs <code>(C*K*K)/TILE_WIDTH</code> phases, which truncates to {printedPhases(LENET_INNER, TILE)} here
          and silently drops {LENET_INNER - printedPhases(LENET_INNER, TILE) * TILE} of the 25 terms (section{' '}
          <a href={href('ch19', '19.3')}>19.3</a>). Taught in <a href={href('ch05', '5.5')}>ch. 5 §5.5</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.5, pp. 112–115 (Fig. 5.13); ch. 19 p. 471', // i18n-ignore: citation
  },
  'ch19/start/q-convratio': {
    kind: 'numeric',
    id: 'ch19/start/q-convratio',
    prompt: (
      <Translated
        vi={
          <>
            Kernel tích chập (convolution) 2D cơ bản của chương 7 giao cho mỗi luồng (thread) một đầu ra và đọc bộ lọc từ
            bộ nhớ toàn cục. Với bộ lọc 5 × 5 ở độ chính xác đơn, vòng lặp trong của nó thực hiện bao nhiêu FLOP cho mỗi
            byte nó yêu cầu từ bộ nhớ toàn cục?
          </>
        }
      >
        <>
          Chapter 7&rsquo;s basic 2D convolution kernel gives each thread one output and reads the filter from global
          memory. With a 5 × 5 filter in single precision, how many FLOP does its inner loop do per byte it requests from
          global memory?
        </>
      </Translated>
    ),
    answer: basicIntensity(4),
    tol: 0.01,
    unit: 'FLOP/B',
    explain: (
      <Translated
        vi={
          <p>
            Mỗi tap nạp một giá trị đầu vào và một giá trị bộ lọc, 8 byte, cho một phép nhân và một phép cộng: 2 / 8 =
            0.25. Kích thước bộ lọc thay đổi số tap, không thay đổi giá của mỗi tap. Kernel trực tiếp của mục{' '}
            <a href={href('ch19', '19.2')}>19.2</a> làm y như vậy cho mỗi tap, trên mọi kênh đầu vào, nên nó thừa hưởng
            0.25, và sách gọi nó là bị giới hạn bởi băng thông bộ nhớ toàn cục. Đã học ở{' '}
            <a href={href('ch07', '7.3')}>ch. 7 §7.3</a>.
          </p>
        }
      >
        <p>
          Each tap loads one input and one filter value, 8 bytes, for one multiply and one add: 2 / 8 = 0.25. The filter
          size changes how many taps there are, not what each one costs. The direct kernel of section{' '}
          <a href={href('ch19', '19.2')}>19.2</a> does the same per tap, over every input channel, so it inherits 0.25, and
          the book calls it limited by global-memory bandwidth. Taught in <a href={href('ch07', '7.3')}>ch. 7 §7.3</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 7 §7.3, p. 167; ch. 19 p. 463', // i18n-ignore: citation
  },
  'ch19/start/q-xcorr': {
    kind: 'choice',
    id: 'ch19/start/q-xcorr',
    prompt: (
      <Translated
        vi={
          <>
            <code>nn.Conv2d</code> của PyTorch có lật bộ lọc trước khi trượt nó trên đầu vào không? Còn công thức tích chập
            của chương 7 thì sao?
          </>
        }
      >
        <>
          Does PyTorch&rsquo;s <code>nn.Conv2d</code> flip the filter before sliding it over the input? Does chapter
          7&rsquo;s convolution formula?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không bên nào lật: cả hai đều tính cái mà các framework gọi là tương quan chéo (cross-correlation).">
            Neither does: both compute what frameworks call cross-correlation.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="PyTorch lật nó, như xử lý tín hiệu vẫn làm; công thức của sách thì không.">
            PyTorch flips it, as signal processing does; the book’s formula does not.
          </Translated>
        ),
        why: (
          <Translated vi="PyTorch ghi trong tài liệu rằng nn.Conv2d là một phép tương quan chéo 2D hợp lệ, vốn không có bước lật.">
            PyTorch documents nn.Conv2d as a valid 2D cross-correlation, which has no flip.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Sách lật nó, như xử lý tín hiệu vẫn làm; PyTorch thì không.">
            The book flips it, as signal processing does; PyTorch does not.
          </Translated>
        ),
        why: (
          <Translated vi="Công thức của chương 7 đi qua bộ lọc và đầu vào theo cùng một hướng, nên không có gì bị đảo ngược.">
            Chapter 7’s formula walks the filter and the input in the same direction, so nothing is reversed.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cả hai đều lật khi kích thước bộ lọc là số lẻ.">Both flip it when the filter size is odd.</Translated>
        ),
        why: (
          <Translated vi="Việc lật không phụ thuộc vào kích thước bộ lọc, và không bên nào lật cả.">
            Flipping does not depend on the size of the filter, and neither of them flips.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tích chập theo nghĩa chặt chẽ đảo ngược bộ lọc; tương quan chéo thì không. Công thức của chương 7, và lớp của
            chương này (<code>X[c, h + p, w + q] * F[m, c, p, q]</code> trong Fig. 19.3), đọc bộ lọc và đầu vào theo cùng
            một hướng. PyTorch ghi trong tài liệu rằng <code>nn.Conv2d</code> là tương quan chéo và gọi cuDNN và MIOpen ở
            chế độ đó, nên trọng số của sách và của PyTorch khớp nhau mà không cần đảo ngược. Mục{' '}
            <a href={href('ch19', '19.4')}>19.4</a> quay lại chuyện thư viện nào ghi rõ điều này và thư viện nào không. Đã
            học ở <a href={href('ch07', '7.1')}>ch. 7 §7.1</a>.
          </p>
        }
      >
        <p>
          A convolution in the strict sense reverses the filter; cross-correlation does not. Chapter 7&rsquo;s formula, and
          this chapter&rsquo;s layer (<code>X[c, h + p, w + q] * F[m, c, p, q]</code> in Fig. 19.3), read filter and input
          in the same direction. PyTorch documents <code>nn.Conv2d</code> as cross-correlation and calls cuDNN and MIOpen in
          that mode, so the book&rsquo;s weights and PyTorch&rsquo;s line up with no reversal. Section{' '}
          <a href={href('ch19', '19.4')}>19.4</a> comes back to which libraries document this and which do not. Taught in{' '}
          <a href={href('ch07', '7.1')}>ch. 7 §7.1</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 7 §7.1, pp. 159–164 (the no-flip reading is the notes’); ch. 19 p. 457; ch07-facts D1',
    vi: { source: 'ch. 7 §7.1, pp. 159–164 (cách hiểu không lật là của ghi chú); ch. 19 p. 457; ch07-facts D1' },
  },
};

// ---------------------------------------------------------------- the chapter at a glance
// A function so the titles are read in the current language when the table renders.
const MAP = (): [string, string, string][] => [
  [
    '19.1',
    tr('Convolutional neural networks', 'Mạng nơ-ron tích chập'),
    tr(
      'Read a layer’s shapes: the input X, the filter bank F, the output Y, and what a batch adds',
      'Đọc kích thước của một lớp: đầu vào X, bộ lọc F, đầu ra Y, và batch thêm vào những gì',
    ),
  ],
  [
    '19.2',
    tr('A CUDA convolutional layer kernel', 'Một kernel lớp tích chập CUDA'),
    tr(
      'Give each thread one output, place the grid, and see why the kernel waits on memory',
      'Giao cho mỗi luồng một đầu ra, đặt lưới (grid), và thấy vì sao kernel phải chờ bộ nhớ',
    ),
  ],
  [
    '19.3',
    tr('Formulating a convolutional layer as GEMM', 'Biểu diễn một lớp tích chập dưới dạng GEMM'),
    tr(
      'Unfold the input into a matrix, first explicitly, then tile by tile inside a GEMM with the index map of Eq. 19.5',
      'Unfold đầu vào thành một ma trận, trước hết tường minh, rồi từng ô một bên trong một GEMM với ánh xạ chỉ số của Eq. 19.5',
    ),
  ],
  [
    '19.4',
    tr('The cuDNN library', 'Thư viện cuDNN'),
    tr(
      'Describe a layer to cuDNN, MIOpen or Apple’s frameworks, and say what each one decides for you',
      'Mô tả một lớp cho cuDNN, MIOpen hoặc các framework của Apple, và nói mỗi thư viện quyết định thay bạn những gì',
    ),
  ],
];

export function S190() {
  const q = questionsS190;
  return (
    <>
      <Translated
        vi={
          <p>
            Khóa học này nói về các kernel đứng sau suy luận LLM, và một khối transformer chỉ có decoder không chứa phép
            tích chập (convolution) nào. Tích chập chạy trước khi mô hình ngôn ngữ thấy bất cứ thứ gì, trong các bộ mã hóa
            biến một ảnh hay một âm thanh thành token cho một mô hình đa phương thức. Patch embedding của một bộ mã hóa thị
            giác là một phép tích chập có bước nhảy (stride) bằng kích thước bộ lọc, nên các cửa sổ của nó không bao giờ
            chồng lên nhau. Một bộ mã hóa âm thanh kiểu Whisper mở đầu bằng các phép tích chập 1D. Một số mô hình lai cũng
            chạy một phép tích chập 1D nhân quả ngắn bên trong các khối của chúng, thứ bạn đã gặp ở mục{' '}
            <a href={href('ch07', '7.5')}>7.5</a>. Vậy là bạn sẽ gặp tích chập, nhưng đó không phải lý do chương này có mặt
            ở đây.
          </p>
        }
      >
        <p>
          This course is about the kernels behind LLM inference, and a decoder-only transformer block contains no
          convolution. Convolutions run before the language model sees anything, in the encoders that turn an image or a
          sound into tokens for a multimodal model. A vision encoder&rsquo;s patch embedding is a convolution whose stride
          equals its filter size, so its windows never overlap. A Whisper-style audio encoder opens with 1D convolutions.
          Some hybrid models also run a short causal 1D convolution inside their blocks, which you met in section{' '}
          <a href={href('ch07', '7.5')}>7.5</a>. You will meet convolutions, then, but they are not why this chapter is
          here.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Những chỗ tích chập xuất hiện trong việc phục vụ LLM được đọc từ mã nguồn của llama.cpp vào 2026-11-20
              (ch19-facts D2; ch07-facts D3).
            </p>
          }
        >
          <p>
            Where convolutions appear in LLM serving was read from llama.cpp&rsquo;s sources on 2026-11-20 (ch19-facts D2;
            ch07-facts D3).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Lý do là một bước đi mà chương thực hiện ở p. 464, được ghi công cho Chellapilla và cộng sự: viết lại toán tử
            thành một phép nhân ma trận duy nhất, để một hàm GEMM đã được tinh chỉnh kỹ phục vụ nó. Mỗi đầu ra của một lớp
            tích chập là tích vô hướng của một số đầu vào với một số trọng số, và một tích ma trận là một bảng các tích vô
            hướng. Xếp các bộ lọc thành các hàng của một ma trận và các vùng đầu vào thành các cột của một ma trận khác, và
            lớp đó là một GEMM. Chương 15 nói các lớp attention cũng được biểu diễn thành những phép nhân ma trận lớn. Khi
            một toán tử bạn cần là một đống tích vô hướng, hãy tìm góc nhìn trong đó nó là một GEMM, và dồn công sức vào
            đúng kernel đó.
          </p>
        }
      >
        <p>
          The reason is one move the chapter makes on p. 464, which it credits to Chellapilla and colleagues: rewrite the
          operator as a single matrix multiplication, so that one heavily tuned GEMM routine serves it. Every output of a
          convolutional layer is a dot product of some inputs with some weights, and a matrix product is a table of dot
          products. Lay the filters out as the rows of one matrix and the input patches as the columns of another, and
          the layer is a GEMM. Chapter 15 says attention layers are formulated as large matrix multiplications too. When
          an operator you need is a pile of dot products, look for the view in which it is one GEMM, and spend your effort
          on that one kernel.
        </p>
      </Translated>
      <Source section="19.3" pages="463–464" />

      <h2>
        <Translated vi="Sách đặt ra mục tiêu gì">What the book sets out to do</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách đặt vấn đề theo góc nhìn lịch sử. Học sâu, tức học máy với những tầng phân cấp sâu của các bộ trích xuất
            đặc trưng và rất nhiều tham số, đã chờ rất lâu cho hai thứ: các tập dữ liệu lớn, và sức tính toán số học của
            GPU. Sách xác định thời điểm nó cất cánh trong công nghiệp là năm 2012. Theo sách, các lớp tích chập hợp với GPU
            vì chúng làm rất nhiều phép tính cho mỗi byte chúng chạm tới và có rất nhiều công việc độc lập.
          </p>
        }
      >
        <p>
          The book&rsquo;s framing is historical. Deep learning, machine learning with deep hierarchies of feature
          extractors and very many parameters, waited a long time for two things: large datasets, and the arithmetic of
          GPUs. The book dates its take-off in industry to 2012. Convolutional layers suit GPUs, it says, because they do
          a lot of arithmetic for every byte they touch and offer plenty of independent work.
        </p>
      </Translated>
      <Source pages="453–454" />
      <Translated
        vi={
          <p>
            Kế hoạch sách nêu ra là một kernel CUDA cơ bản cho lượt lan truyền xuôi của một lớp tích chập, &ldquo;một vài
            tối ưu&rdquo;, rồi lớp đó dưới dạng một phép nhân ma trận. Hãy đọc mục ở giữa cho cẩn thận. Kernel trực tiếp
            không có tối ưu nào của riêng nó: mục 19.2 trình bày nó, nói nó dùng quá nhiều băng thông bộ nhớ toàn cục
            (global memory), và để việc phân ô (tiling) bằng bộ nhớ chia sẻ (shared memory) cho bạn. Tối ưu duy nhất mà
            chương phát triển là cách biểu diễn bằng GEMM.
          </p>
        }
      >
        <p>
          The plan it states is a basic CUDA kernel for the forward path of one convolutional layer, &ldquo;a few
          optimizations&rdquo;, and then the layer as a matrix multiplication. Read the middle item with care. The direct
          kernel gets no optimisation of its own: section 19.2 shows it, says it uses too much global-memory bandwidth,
          and leaves shared-memory tiling to you. The one optimisation the chapter develops is the GEMM formulation.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Phần mở đầu hứa hẹn các tối ưu ở p. 454; phân ô cho kernel trực tiếp được để lại làm bài tập ở p. 463 (ghi chú,
              &ldquo;Unclear in the PDF&rdquo;).
            </p>
          }
        >
          <p>
            The intro promises optimisations on p. 454; tiling the direct kernel is left as an exercise on p. 463 (notes,
            &ldquo;Unclear in the PDF&rdquo;).
          </p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Tổng quan chương">The chapter at a glance</Translated>
      </h2>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-map>
          <thead>
            <tr>
              <th>
                <Translated vi="Mục">Section</Translated>
              </th>
              <th>
                <Translated vi="Bạn sẽ làm được gì">What you will be able to do</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            {MAP().map(([id, title, what]) => (
              <tr key={id}>
                <td>
                  <a href={href('ch19', id)}>
                    {id} {title}
                  </a>
                </td>
                <td>{what}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Translated
        vi={
          <p data-lab>
            Một lab, trên <a href={href('ch19', 'labs')}>trang lab</a>. Lab 19.1 dựng lớp này theo hai cách, như kernel trực
            tiếp của mục 19.2 và như một bước unfold tường minh theo sau là GEMM M2 của dự án, hàm{' '}
            <code>launchLinear</code> của lab 6.4, và đo thời gian cả hai trên thiết bị của bạn. Bạn dự đoán kết quả trước
            khi chạy.
          </p>
        }
      >
        <p data-lab>
          One lab, on the <a href={href('ch19', 'labs')}>labs page</a>. Lab 19.1 builds the layer two ways, as the direct
          kernel of section 19.2 and as an explicit unfold followed by the project&rsquo;s M2 GEMM, lab 6.4&rsquo;s{' '}
          <code>launchLinear</code>, and times both on your device. You predict the outcome before you run it.
        </p>
      </Translated>

      <h2>
        <Translated vi="Bên dưới torch.nn.functional.conv2d">Underneath torch.nn.functional.conv2d</Translated>
      </h2>
      <Translated
        vi={
          <p>
            PyTorch giữ một danh sách backend cho phép tích chập. Trong số đó có cuDNN, MIOpen và MPS (một phép tích chập
            MPSGraph trên Apple silicon), các thư viện của nhà sản xuất ở mục 19.4. Còn có một đường CUDA chậm của riêng
            nó, và đường đó gần như từng dòng là mục 19.3: với mỗi mẫu, nó unfold đầu vào thành một ma trận cột, rồi gọi một
            GEMM với các trọng số. Trọng số không cần sao chép, vì <code>nn.Conv2d</code> lưu chúng với kênh đầu ra đứng
            trước, đúng <code>M × C × K × K</code> của chương.
          </p>
        }
      >
        <p>
          PyTorch keeps a list of backends for a convolution. Among them are cuDNN, MIOpen and MPS (an MPSGraph
          convolution on Apple silicon), the vendor libraries of section 19.4. There is also a slow CUDA path of its own,
          and that one is section 19.3 nearly line for line: for each sample it unfolds the input into a column matrix,
          then calls one GEMM with the weights. The weights need no copy, because <code>nn.Conv2d</code> stores them
          output channels first, the chapter&rsquo;s <code>M × C × K × K</code>.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Đọc từ <code>ConvUtils.h</code>, <code>ConvolutionMM2d.cu</code> của PyTorch và tài liệu bản 2.15 vào
              2026-11-20 (ch19-facts D4).
            </p>
          }
        >
          <p>
            Read from PyTorch&rsquo;s <code>ConvUtils.h</code>, <code>ConvolutionMM2d.cu</code> and the 2.15 docs on
            2026-11-20 (ch19-facts D4).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            llama.cpp hạ cấp phép tích chập theo cùng cách trong đồ thị tổng quát của nó, một bước unfold tường minh theo
            sau là một phép nhân ma trận, và backend CUDA của nó có một kernel chính là Fig. 19.11 chạy trong thực tế. Mục
            19.4 mở cả hai ra xem.
          </p>
        }
      >
        <p>
          llama.cpp lowers convolutions the same way in its generic graph, an explicit unfold followed by a matrix
          multiplication, and its CUDA backend has a kernel that is Fig. 19.11 in production. Section 19.4 opens both.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch19-facts D1, D2)" /* i18n-ignore: citation */>
        <Translated
          vi={
            <p>
              Các công cụ đa phương thức của llama.cpp dựng patch embedding của một mô hình thị giác bằng{' '}
              <code>ggml_conv_2d</code>, với bước nhảy bằng kích thước patch và không padding, và bộ mã hóa Whisper của nó
              chạy các lớp <code>conv1d</code>.
            </p>
          }
        >
          <p>
            llama.cpp&rsquo;s multimodal tools build a vision model&rsquo;s patch embedding with <code>ggml_conv_2d</code>,
            stride equal to the patch size and no padding, and its Whisper encoder runs <code>conv1d</code> layers.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Các kernel của chương chỉ dùng luồng (thread), khối (block), bộ nhớ chia sẻ và rào chắn (barrier), và với những
            thứ đó, một tệp <code>.cu</code> được thiết lập để build cho HIP mà không cần sửa, qua{' '}
            <code>pmpp/vendors/hip.h</code> của khóa học (có trong tài liệu, ch19-facts B1); nó chưa được biên dịch hay chạy
            trên phần cứng AMD ở đây. Điều đó sẽ không còn đúng nếu chúng dùng intrinsic warp <code>_sync</code>,
            cooperative groups, header nửa độ chính xác, functor của CUB, intrinsic matrix-core hoặc <code>__dp4a</code>{' '}
            (HIP không có cái nào), mà chúng thì không dùng. Trên Metal chúng được viết lại bằng MSL với các mảng{' '}
            <code>threadgroup</code> và <code>threadgroup_barrier</code>.
          </p>
        }
      >
        <p>
          The chapter&rsquo;s kernels use only threads, blocks, shared memory and barriers, and for those one{' '}
          <code>.cu</code> file is set up to build for HIP unchanged through the course&rsquo;s{' '}
          <code>pmpp/vendors/hip.h</code> (documented, ch19-facts B1); it has not been compiled or run on AMD hardware here.
          That would stop being true if they used warp <code>_sync</code> intrinsics, cooperative groups, half-precision
          headers, CUB functors, matrix-core intrinsics or <code>__dp4a</code> (HIP has none), which they do not. On Metal they are rewritten in MSL with{' '}
          <code>threadgroup</code> arrays and <code>threadgroup_barrier</code>.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>Sách chỉ đề cập CUDA. Đã đối chiếu với tài liệu của nhà sản xuất (ch19-facts B1, C1).</p>
          }
        >
          <p>The book covers CUDA only. Checked against vendor documentation (ch19-facts B1, C1).</p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Trước khi bắt đầu">Before you start</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Năm câu hỏi về những gì chương này giả định từ các chương 3, 5, 7 và 15. Mỗi lời giải thích chỉ tới nơi ý tưởng
            đó được dạy và tới mục ở đây dùng nó. Chúng sẽ quay lại trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Five questions on what this chapter assumes from chapters 3, 5, 7 and 15. Each explanation points to where the
          idea was taught and to the section here that uses it. They come back in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={q['ch19/start/q-index4d']} label={tr('From chapter 3', 'Từ chương 3')} />
      <Ask q={q['ch19/start/q-reuse']} label={tr('From chapters 5 and 15', 'Từ chương 5 và 15')} />
      <Ask q={q['ch19/start/q-phases']} label={tr('From chapter 5', 'Từ chương 5')} />
      <Ask q={q['ch19/start/q-convratio']} label={tr('From chapter 7', 'Từ chương 7')} />
      <Ask q={q['ch19/start/q-xcorr']} label={tr('From chapter 7', 'Từ chương 7')} />
    </>
  );
}
