import { Ask, type Question } from '../../components/Ask';
import { Aside, Source } from '../../components/notes';
import { Code, MATMUL_TIERS } from '../../components/Code';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// The kernel of the book's Fig. 3.11 as the labs use it: labs/ch05/shared/naive_matmul.cu
// (snippets.test.ts keeps the two identical). It differs from the page only by `const`
// on the inputs and the kernel's name.
const NAIVE = `__global__ void naiveMatmulKernel(const float* M, const float* N, float* P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < Width && col < Width) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = Pvalue;
    }
}`;

// Prerequisite questions drawn from chapters 3 and 4 (from the notes for those chapters).
export const questions50: Record<string, Question> = {
  'ch05/start/q-rowmajor': {
    kind: 'numeric',
    id: 'ch05/start/q-rowmajor',
    prompt: (
      <Translated
        vi={
          <>
            Một ma trận có <code>Width = 6</code> được lưu theo thứ tự hàng (row-major) trong mảng phẳng <code>N</code>.
            Kernel ở trên đi dọc <b>cột 2</b> của <code>N</code>. Giữa hai lần lặp liên tiếp của <code>k</code>, địa chỉ
            dịch đi bao nhiêu phần tử của mảng?
          </>
        }
      >
        A matrix with <code>Width = 6</code> is stored row-major in a flat array <code>N</code>. The kernel above walks
        down <b>column 2</b> of <code>N</code>. By how many array elements does the address move between consecutive
        iterations of <code>k</code>?
      </Translated>
    ),
    answer: 6,
    unit: 'elements',
    explain: (
      <Translated
        vi={
          <p>
            <code>N[k*Width + col]</code>: mỗi bước của <code>k</code> cộng thêm trọn một hàng, nên bước nhảy (stride)
            là <code>Width</code>. Lần đọc <code>M[row*Width + k]</code> dịch đi một. Hãy nhớ sự bất đối xứng này: M được
            đọc dọc theo bộ nhớ, còn N thì cắt ngang nó. Chương 6 biến điều này thành một sự thật về hiệu năng.
          </p>
        }
      >
        <p>
          <code>N[k*Width + col]</code>: each step of <code>k</code> adds one whole row, so the stride is{' '}
          <code>Width</code>. The read of <code>M[row*Width + k]</code> moves by one. Keep this asymmetry in mind: M is
          read along memory, N across it. Chapter 6 turns it into a performance fact.
        </p>
      </Translated>
    ),
    source: 'ch. 3 §3.4, p. 62',
    vi: { unit: 'phần tử' },
  },
  'ch05/start/q-mapping': {
    kind: 'choice',
    id: 'ch05/start/q-mapping',
    prompt: (
      <Translated
        vi={
          <>
            Kernel ngây thơ được khởi chạy với các khối 16×16 trên một ma trận 40×40. Những luồng nào của khối tại{' '}
            <code>blockIdx = (x 2, y 0)</code> thực sự làm việc?
          </>
        }
      >
        The naive kernel is launched with 16×16 blocks on a 40×40 matrix. Which threads of the block at{' '}
        <code>blockIdx = (x 2, y 0)</code> do real work?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Cả 256: mọi luồng của một khối đã khởi chạy đều chạy thân kernel.">
            All 256: every thread of a launched block runs the kernel body.
          </Translated>
        ),
        why: (
          <Translated vi="Mọi luồng đều chạy thân kernel, nhưng chỉ những luồng có cột tồn tại mới vượt qua phép kiểm tra biên. Khối này phủ các cột 32 đến 47, còn ma trận dừng ở cột 39.">
            Every thread runs the body, but only those whose column exists pass the bounds test. This block covers
            columns 32 to 47, and the matrix stops at column 39.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="128 luồng có threadIdx.x từ 0 đến 7.">The 128 with threadIdx.x from 0 to 7.</Translated>
        ),
        correct: true,
        why: (
          <Translated vi="Các cột 32 đến 39 tồn tại; các hàng 0 đến 15 đều tồn tại.">
            Columns 32 to 39 exist; rows 0 to 15 all exist.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="128 luồng có threadIdx.y từ 0 đến 7: ma trận kết thúc sau 8 hàng của khối này.">
            The 128 with threadIdx.y from 0 to 7: the matrix ends 8 rows into this block.
          </Translated>
        ),
        why: (
          <Translated vi="Ở đây x ánh xạ sang cột, y sang hàng. Khối này có blockIdx.y = 0, nên các hàng 0 đến 15 của nó đều tồn tại.">
            x maps to columns here, y to rows. This block has blockIdx.y = 0, so its rows 0 to 15 all exist.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="64 luồng có cả threadIdx.x và threadIdx.y từ 0 đến 7.">
            The 64 with threadIdx.x and threadIdx.y both from 0 to 7.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là khối ở góc, x 2 và y 2. Khối này nằm ở hàng trên cùng của lưới, nơi mọi hàng đều tồn tại.">
            That is the corner block, x 2 and y 2. This block sits in the top row of the grid, where every row exists.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>col = blockIdx.x*blockDim.x + threadIdx.x = 32 + threadIdx.x</code> phải nhỏ hơn 40, nên{' '}
            <code>threadIdx.x ≤ 7</code>. <code>row = threadIdx.y</code> luôn nhỏ hơn 40. 128 luồng còn lại không qua được{' '}
            <code>if</code> và thoát. Ở phần sau của chương này bạn sẽ gặp những luồng không sở hữu phần tử đầu ra nào mà
            vẫn có việc phải làm.
          </p>
        }
      >
        <p>
          <code>col = blockIdx.x*blockDim.x + threadIdx.x = 32 + threadIdx.x</code> must be below 40, so{' '}
          <code>threadIdx.x ≤ 7</code>. <code>row = threadIdx.y</code> is always below 40. The other 128 threads fail the{' '}
          <code>if</code> and exit. Later in this chapter you will meet threads that own no output element and still have
          work to do.
        </p>
      </Translated>
    ),
    source: 'ch. 3 §3.2 and §3.4, pp. 48–54, 61',
    vi: { source: 'ch. 3 §3.2 và §3.4, pp. 48–54, 61' },
  },
  'ch05/start/q-barrier': {
    kind: 'choice',
    id: 'ch05/start/q-barrier',
    prompt: (
      <Translated
        vi={
          <>
            <code>__syncthreads()</code> bảo đảm điều gì?
          </>
        }
      >
        What does <code>__syncthreads()</code> guarantee?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không luồng nào của khối vượt qua nó cho đến khi mọi luồng của khối đó đều đến được nó.">
            No thread of its block passes it until all threads of that block reach it.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không luồng nào của lưới đi tiếp cho đến khi mọi luồng của lưới đều đến được nó.">
            No thread of the grid proceeds until every thread of the grid has reached it.
          </Translated>
        ),
        why: (
          <Translated vi="Phạm vi của nó là một khối. Các khối không thể chờ nhau theo cách này, và chính điều đó cho phép chúng chạy theo bất kỳ thứ tự nào.">
            The scope is one block. Blocks cannot wait for each other this way, which is what lets them run in any
            order.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Sau đó các luồng của khối thực thi các câu lệnh phía sau lần lượt từng luồng một, như một khóa (lock).">
            The threads of the block then execute the statements after it one at a time, like a lock.
          </Translated>
        ),
        why: (
          <Translated vi="Nó là một rào chắn (barrier), không phải khóa: sau nó, các luồng lại chạy đồng thời.">
            It is a barrier, not a lock: after it, threads run concurrently again.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Mọi lần ghi vào bộ nhớ toàn cục trước nó đều trở nên hiển thị với mọi luồng của lưới.">
            Every global-memory write made before it becomes visible to every thread of the grid.
          </Translated>
        ),
        why: (
          <Translated vi="Phạm vi của nó là một khối. Luồng của các khối khác không được nó sắp thứ tự, nên nó không hứa hẹn gì với chúng.">
            Its scope is one block. Threads of other blocks are not ordered by it, so it promises them nothing.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Nó là một rào chắn có phạm vi một khối. Hai hệ quả bạn sẽ dùng hôm nay: mọi luồng của khối phải thực thi cùng
            một rào chắn, và một rào chắn không nói gì về các khối khác.
          </p>
        }
      >
        <p>
          It is a barrier scoped to one block. Two consequences you will use today: every thread of the block must execute
          the same barrier, and a barrier says nothing about other blocks.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.3, pp. 70–72',
  },
  'ch05/start/q-occupancy': {
    kind: 'numeric',
    id: 'ch05/start/q-occupancy',
    prompt: (
      <Translated
        vi={
          <>
            Trong thiết bị ví dụ của sách, một streaming multiprocessor (SM) chứa được 2048 luồng và có 65,536 thanh ghi.
            Một kernel biên dịch ra 40 thanh ghi mỗi luồng. Occupancy cao nhất nó có thể đạt là bao nhiêu phần trăm? Bỏ
            qua độ chi tiết theo kích thước khối.
          </>
        }
      >
        In the book's example device, a streaming multiprocessor can hold 2048 threads and has 65,536 registers. A
        kernel compiles to 40 registers per thread. What is the best occupancy it can reach, in percent? Ignore the
        block-size granularity.
      </Translated>
    ),
    answer: 80,
    tol: 0.01,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            65,536 / 40 = 1638 luồng vừa, và 1638 / 2048 = 80%. Thanh ghi là một trong ba ngân sách giới hạn số luồng một
            SM có thể chứa; chương này thêm ngân sách thứ hai, bộ nhớ chia sẻ. (Với kích thước khối thực tế, con số còn
            thấp hơn nữa: chỉ những khối trọn vẹn mới vừa.)
          </p>
        }
      >
        <p>
          65,536 / 40 = 1638 threads fit, and 1638 / 2048 = 80%. Registers are one of three budgets that cap how many
          threads an SM can hold; this chapter adds a second, shared memory. (With real block sizes the figure is lower
          still: only whole blocks fit.)
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.7, pp. 85–87 (device figures are the book\'s Hopper H100 example)',
    vi: { source: 'ch. 4 §4.7, pp. 85–87 (số liệu thiết bị là ví dụ Hopper H100 của sách)' },
  },
};

export function S50() {
  return (
    <>
      <Translated
        vi={
          <p>
            Khi bạn gọi <code>torch.matmul</code> trên một tensor GPU, framework giao việc cho một hàm nhân ma trận đã được
            tinh chỉnh cho thiết bị đó. Chương này là nơi bạn bắt đầu thấy vì sao những hàm như vậy trông chẳng giống
            chút nào ba vòng lặp lồng nhau mà bạn sẽ viết đầu tiên, và bạn tự viết cải tiến thực sự đầu tiên.
          </p>
        }
      >
        <p>
          When you call <code>torch.matmul</code> on a GPU tensor, the framework hands the work to a matrix
          multiplication routine tuned for that device. This chapter is where you start to see why such routines look
          nothing like the three nested loops you would write first, and you write the first real improvement yourself.
        </p>
      </Translated>
      <Source pages="93" />
      <Translated
        vi={
          <p>
            Kernel từ chương 3 đúng và dùng hàng nghìn luồng, vậy mà trên GPU tham chiếu của sách nó chỉ đạt khoảng một
            phần trăm khả năng tính toán số học của phần cứng. Các luồng không thiếu việc. Chúng đang chờ bộ nhớ. Mọi
            thứ trong chương này đều bắt nguồn từ một quan sát đó:
          </p>
        }
      >
        <p>
          The kernel from chapter 3 is correct and uses thousands of threads, yet on the book's reference GPU it reaches
          about one percent of the arithmetic the hardware can do. The threads are not short of work. They are waiting
          for memory. Everything in this chapter follows from that one observation:
        </p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>
              một cách để <b>đo</b> nó: số phép toán thực hiện trên mỗi byte được nạp, và một hình vẽ (roofline) cho bạn
              biết kernel đang chạm vào giới hạn nào;
            </li>
            <li>
              những <b>nơi dữ liệu có thể nằm</b> trên GPU, và cái giá của từng nơi;
            </li>
            <li>
              một kỹ thuật, <b>phân ô (tiling)</b>, chuyển dữ liệu mà một khối luồng dùng đi dùng lại vào bộ nhớ nằm trên
              chip;
            </li>
            <li>
              và <b>cái giá phải trả</b> cho kỹ thuật đó: rào chắn, kiểm tra biên và một ngân sách bộ nhớ.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            a way to <b>measure</b> it: operations done per byte fetched, and a picture (the roofline) that tells you
            which limit a kernel is up against;
          </li>
          <li>
            the <b>places data can live</b> on a GPU, and what each costs;
          </li>
          <li>
            a technique, <b>tiling</b>, that moves the data a block of threads keeps reusing into memory that is on the
            chip;
          </li>
          <li>
            and the <b>bill</b> for that technique: barriers, bounds checks and a memory budget.
          </li>
        </ul>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              <b>Màu sắc trong khóa học này</b> cho biết dữ liệu nằm ở đâu:{' '}
              <span className="chip reg">thanh ghi</span> <span className="chip shared">bộ nhớ chia sẻ</span>{' '}
              <span className="chip global">bộ nhớ toàn cục</span> <span className="chip const">bộ nhớ hằng</span>. Các
              màu này cũng đánh dấu biến trong mã. Màu đỏ luôn có nghĩa là có gì đó thất bại.
            </p>
          }
        >
          <p>
            <b>Colour in this course</b> means where data lives:{' '}
            <span className="chip reg">register</span> <span className="chip shared">shared memory</span>{' '}
            <span className="chip global">global memory</span> <span className="chip const">constant memory</span>. The
            same colours mark variables in code. Red always means something failed.
          </p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Kernel mà bạn sắp mổ xẻ">The kernel you are about to take apart</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một luồng tính một phần tử của <code>P = M × N</code>. Hãy đọc nó theo màu: mọi thứ trong vòng lặp không phải
            thanh ghi đều là một chuyến đi tới bộ nhớ toàn cục.
          </p>
        }
      >
        <p>
          One thread computes one element of <code>P = M × N</code>. Read it with the colours: everything inside the loop
          that is not a register is a trip to global memory.
        </p>
      </Translated>
      <Source section="3.4" pages="61" figs={tr('Fig. 3.11', 'Hình 3.11')} />
      <Code
        tabs={[{ label: 'CUDA', code: NAIVE, tiers: MATMUL_TIERS, status: 'front-end-checked (labs/ch05/shared/naive_matmul.cu)' }]}
        highlight={[7]}
        caption={tr(
          "The naive kernel of the book's Fig. 3.11, in the form the labs use as their baseline (the book names it MatrixMulKernel and leaves out const). Line 7 is the whole story of this chapter: two loads from global memory for one multiply and one add.",
          'Kernel ngây thơ trong Hình 3.11 của sách, ở dạng các lab dùng làm mốc so sánh (sách đặt tên nó là MatrixMulKernel và bỏ const). Dòng 7 là toàn bộ câu chuyện của chương này: hai lần nạp từ bộ nhớ toàn cục cho một phép nhân và một phép cộng.',
        )}
      />

      <h2>
        <Translated vi="Trước khi bắt đầu">Before you start</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bốn câu hỏi về những gì chương này giả định từ chương 3 và 4. Trả lời sai bây giờ còn hơn để sau này; mỗi câu
            sẽ quay lại trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Four questions on what this chapter assumes from chapters 3 and 4. They are worth getting wrong now rather than
          later; each one comes back in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={questions50['ch05/start/q-rowmajor']} label={tr('From chapter 3', 'Từ chương 3')} />
      <Ask q={questions50['ch05/start/q-mapping']} label={tr('From chapter 3', 'Từ chương 3')} />
      <Ask q={questions50['ch05/start/q-barrier']} label={tr('From chapter 4', 'Từ chương 4')} />
      <Ask q={questions50['ch05/start/q-occupancy']} label={tr('From chapter 4', 'Từ chương 4')} />
    </>
  );
}
