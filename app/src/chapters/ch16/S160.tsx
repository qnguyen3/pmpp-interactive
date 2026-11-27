import { Ask, type Question } from '../../components/Ask';
import { Beyond, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { inputsOf, presetFronts, stepAt } from '../../sim/wavefront';

// The start page of chapter 16: a question before any theory, what a wavefront is and why
// the course keeps it (recurrences along a sequence in real inference code), what the
// course keeps and leaves out, and a check on the ideas from chapters 2, 4, 5, 6 and 11
// that the chapter builds on. Every number about wavefronts comes from src/sim/wavefront.ts.

// Edit distance of two 1,000-token sequences: the 1,000 x 1,000 interior of its table.
const HOOK = presetFronts('smith-waterman', 1000, 1000);
// The small table drawn on this page.
const FIG = presetFronts('smith-waterman', 6, 6);
const FIG_FRONT = FIG.sizes.indexOf(FIG.widest); // the widest step
const FIG_CELL: [number, number] = [3, 4];

export const questionsS160: Record<string, Question> = {
  'ch16/start/q-hook': {
    kind: 'choice',
    id: 'ch16/start/q-hook',
    prompt: (
      <Translated
        vi={
          <>
            Khoảng cách chỉnh sửa (edit distance) giữa hai chuỗi 1,000 token lấp đầy một bảng 1,001 × 1,001 ô. Hàng 0 và
            cột 0 đã biết trước khi bắt đầu. Mỗi ô còn lại cần ba ô: ô ngay trên nó, ô bên trái nó và ô chéo trên bên
            trái. Phát biểu nào đúng?
          </>
        }
      >
        <>
          The edit distance between two sequences of 1,000 tokens fills a table of 1,001 × 1,001 cells. Row 0 and column 0
          are known before you start. Every other cell needs three cells: the one above it, the one to its left and the
          one above-left. Which is true?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nó tuần tự về bản chất: mỗi ô cần ô ngay trước nó, nên mỗi lần chỉ một ô.">
            It is inherently sequential: each cell needs the cell just before it, so one cell at a time.
          </Translated>
        ),
        why: (
          <Translated vi="Nhiều ô không cần gì từ nhau. Hãy xét hai ô có tổng hàng và cột bằng nhau.">
            Many cells need nothing from each other. Look at two cells whose row plus column is the same.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Mỗi hàng có thể tính song song khi hàng trên đã xong: 1,000 bước, mỗi bước 1,000 ô.">
            Each row can be computed in parallel once the row above is done: 1,000 steps of 1,000 cells each.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi ô còn cần ô bên trái, nằm cùng hàng. Một hàng là một chuỗi phụ thuộc.">
            Each cell also needs its left neighbour, which is in the same row. A row is a chain.
          </Translated>
        ),
      },
      {
        label: (
          <Translated
            vi={`Các ô trên cùng một đường chéo phụ không cần gì từ nhau, nên bảng cần ${HOOK.steps.toLocaleString('en-GB')} bước, mỗi bước tối đa ${HOOK.widest.toLocaleString('en-GB')} ô.`}
          >
            {`Cells on one anti-diagonal need nothing from each other, so the table takes ${HOOK.steps.toLocaleString('en-GB')} steps of up to ${HOOK.widest.toLocaleString('en-GB')} cells.`}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Mọi ô đều độc lập khi đã biết hai chuỗi, nên tất cả có thể chạy trong một bước.">
            Every cell is independent once the two sequences are known, so all of them can run in one step.
          </Translated>
        ),
        why: (
          <Translated vi="Một ô cần kết quả của ba ô khác, không chỉ hai token mà nó so sánh.">
            A cell needs the results of three other cells, not only the two tokens it compares.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đánh số các ô theo hàng r và cột q. Ba đầu vào của một ô nằm trên các đường chéo phụ r + q − 1 (trên và trái)
            và r + q − 2 (chéo trên trái), không bao giờ trên đường chéo của chính nó, nên mọi ô có cùng r + q có thể chạy
            cùng lúc khi hai đường chéo phụ trước đó đã xong. Các ô được tính có r và q từ 1 đến 1,000, nên r + q chạy từ 2
            đến 2,000: {HOOK.steps.toLocaleString('en-GB')} bước, bước rộng nhất {HOOK.widest.toLocaleString('en-GB')} ô.
            Trung bình mỗi bước chỉ có khoảng {Math.round(HOOK.mean)} ô, một nửa bước rộng nhất, vì các đường chéo phụ dài
            dần rồi ngắn dần. Trang <a href={href('ch16', '16.6')}>16.6</a> loại bỏ sự lãng phí đó bên trong mỗi ô (tile)
            của một kernel phân ô. Các mặt sóng của các ô vẫn dài dần rồi ngắn dần, và phép trượt (shear) làm chúng nông
            hơn. Tập các ô có thể chạy cùng lúc là một <b>wavefront</b>.
          </p>
        }
      >
        <p>
          Number the cells by row r and column q. The three inputs of a cell lie on the anti-diagonals r + q − 1 (above and
          left) and r + q − 2 (above-left), never on its own, so all cells with the same r + q can run together once the
          two anti-diagonals before them are done. The computed cells have r and q from 1 to 1,000, so r + q runs from 2 to
          2,000: {HOOK.steps.toLocaleString('en-GB')} steps, the widest {HOOK.widest.toLocaleString('en-GB')} cells. On
          average a step has only about {Math.round(HOOK.mean)} cells, half the widest, because the anti-diagonals grow
          and then shrink. Page <a href={href('ch16', '16.6')}>16.6</a> removes that waste inside each tile of a tiled
          kernel. The fronts of tiles still grow and shrink, and shearing makes them shallower. The set of cells that can
          run together is a <b>wavefront</b>.
        </p>
      </Translated>
    ),
    source:
      'ch. 16 §16.3 and §16.5, pp. 376–377, 383–384 (edit distance is our example; its table has the three-neighbour shape of the book\'s Smith-Waterman)',
    vi: {
      source:
        'ch. 16 §16.3 và §16.5, pp. 376–377, 383–384 (edit distance là ví dụ của khóa học này; bảng của nó có dạng ba ô láng giềng như Smith-Waterman của sách)',
    },
  },
  'ch16/start/q-stream-order': {
    kind: 'choice',
    id: 'ch16/start/q-stream-order',
    prompt: (
      <Translated
        vi={
          <>
            Vòng lặp host Floyd-Warshall của sách khởi chạy một kernel cho mỗi bước k và gọi{' '}
            <code>cudaDeviceSynchronize()</code> sau mỗi lần khởi chạy. Lời gọi đó có cần để lần khởi chạy k + 1 thấy được
            những gì lần khởi chạy k đã ghi không?
          </>
        }
      >
        <>
          The book's Floyd-Warshall host loop launches a kernel once for every step k and calls{' '}
          <code>cudaDeviceSynchronize()</code> after each launch. Is that call needed for launch k + 1 to see what launch
          k wrote?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có: thiếu nó, lần khởi chạy k + 1 có thể bắt đầu trước khi lần khởi chạy k xong.">
            Yes: without it, launch k + 1 could start before launch k has finished.
          </Translated>
        ),
        why: (
          <Translated vi="Không phải trong cùng một stream. Các thao tác trong một stream chạy theo thứ tự được phát lệnh, và không thao tác nào vượt được thao tác khác.">
            Not within one stream. Operations in a stream run in the order they were issued, and none can overtake another.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không: các lần khởi chạy trong một stream chạy theo thứ tự. Lời gọi đó chỉ bắt host chờ.">
            No: launches in one stream run in order. The call only makes the host wait.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không, vì một lần khởi chạy kernel không trả về cho đến khi kernel chạy xong.">
            No, because a kernel launch does not return until the kernel has finished.
          </Translated>
        ),
        why: (
          <Translated vi="Một lần khởi chạy trả về trước khi kernel chạy. Vì thế host phải chờ trước khi đọc kết quả, nhưng lần khởi chạy tiếp theo không cần host chờ.">
            A launch returns before the kernel runs. That is why the host must wait before it reads results, but the next
            launch does not need the host to wait.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ khi kernel dùng phép toán nguyên tử (atomic), vì các cập nhật của chúng có thể vẫn đang trên đường khi lần khởi chạy tiếp theo bắt đầu.">
            Only if the kernel uses atomics, whose updates may still be in flight when the next launch starts.
          </Translated>
        ),
        why: (
          <Translated vi="Phép toán nguyên tử sắp thứ tự các truy cập bên trong một kernel. Thứ tự giữa các lần khởi chạy đến từ stream.">
            Atomics order accesses within a kernel. The order between launches comes from the stream.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một lần khởi chạy trả về ngay và công việc được đưa vào một stream; các thao tác của một stream chạy lần lượt,
            theo thứ tự được phát lệnh. HIP ghi nhận cùng quy tắc. Trên Metal, các dispatch trong một compute encoder được
            tạo theo cách thông thường chạy tuần tự theo thứ tự dispatch, và các buffer tạo từ device được theo dõi xung
            đột (hazard-tracked), nên mỗi dispatch thấy được những gì dispatch trước đã ghi. Vậy chính việc khởi chạy lại là
            rào chắn (barrier) giữa các bước, và lời gọi thêm kia chỉ giữ host lại sau mỗi bước. Mã đã sửa ở trang{' '}
            <a href={href('ch16', '16.3')}>16.3</a>{' '}
            bỏ nó đi.
          </p>
        }
      >
        <p>
          A launch returns at once and the work goes into a stream; the operations of one stream run one after another, in
          the order they were issued. HIP documents the same rule. On Metal, dispatches in one compute encoder created
          the plain way run serially in dispatch order, and buffers created from the device are hazard-tracked, so each
          dispatch sees the previous one's writes. So the relaunch is the barrier between steps, and the extra call only
          holds the host back after every step. The corrected listing on page <a href={href('ch16', '16.3')}>16.3</a>{' '}
          leaves it out.
        </p>
      </Translated>
    ),
    source: 'ch. 2 §2.2 and §2.6, pp. 24–25, 37–39; ch02-facts A5, B3, C9 (cited in ch16-facts, "Already settled")',
    vi: {
      source: 'ch. 2 §2.2 và §2.6, pp. 24–25, 37–39; ch02-facts A5, B3, C9 (được trích trong ch16-facts, "Already settled")',
    },
  },
  'ch16/start/q-exit-barrier': {
    kind: 'choice',
    id: 'ch16/start/q-exit-barrier',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel bắt đầu bằng <code>if (col &gt;= n) return;</code>, và các luồng còn lại gọi{' '}
            <code>__syncthreads()</code> vài dòng sau đó. Trong khối cuối của mỗi hàng, một số luồng trả về. Ở đâu điều này
            được tài liệu xác nhận là hoạt động?
          </>
        }
      >
        <>
          A kernel begins with <code>if (col &gt;= n) return;</code>, and the threads that remain call{' '}
          <code>__syncthreads()</code> a few lines later. In the last block of each row some threads return. Where is this
          documented to work?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Trên CUDA, nơi rào chắn chỉ chờ các luồng chưa thoát. HIP không nói gì; Metal chỉ nói vậy cho Apple silicon.">
            On CUDA, whose barrier waits only for threads that have not exited. HIP does not say; Metal says so for Apple
            silicon only.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Trên mọi backend: không thể chờ một luồng đã trả về, nên mỗi rào chắn chỉ đếm các luồng còn đang chạy.">
            On every backend: a thread that has returned cannot be waited for, so each barrier counts only the threads
            still running.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là quy tắc được tài liệu của CUDA ghi nhận. Tài liệu HIP của AMD không nêu nó, còn Apple chỉ nêu cho Apple silicon.">
            That is CUDA's documented rule. AMD's HIP documentation does not state it, and Apple states it for Apple
            silicon only.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không ở đâu cả: rào chắn vẫn đếm mọi luồng mà khối được khởi chạy cùng, nên các luồng còn lại chờ mãi mãi.">
            Nowhere: the barrier still counts every thread the block was launched with, so the threads that remain wait for
            ever.
          </Translated>
        ),
        why: (
          <Translated vi='Trên CUDA, tài liệu xác nhận nó hoạt động: rào chắn chờ "all non-exited threads" (mọi luồng chưa thoát).'>
            On CUDA it is documented to work: the barrier waits for "all non-exited threads".
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ khi các luồng trả về tạo thành những warp trọn vẹn, để mỗi warp đến rào chắn với tất cả lane của nó hoặc không lane nào.">
            Only if the threads that return make up whole warps, so that each warp reaches the barrier with all its lanes
            or none.
          </Translated>
        ),
        why: (
          <Translated vi='Quy tắc của CUDA đếm từng luồng đã thoát một ("all non-exited threads"); nó không nói gì về warp trọn vẹn.'>
            CUDA's rule counts exited threads one by one ("all non-exited threads"); it says nothing about whole warps.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Kernel Floyd-Warshall của sách (Fig. 16.4) làm đúng như vậy, và sách không bình luận gì về nó. Kernel đã sửa ở
            trang 16.3 không bao giờ trả về trước rào chắn của nó: một luồng nằm ngoài phạm vi giữ một cờ (flag) và bỏ qua
            phần việc, điều này không tốn gì và đúng trên mọi backend.
          </p>
        }
      >
        <p>
          The book's Floyd-Warshall kernel (Fig. 16.4) does exactly this, and the book does not comment on it. The
          corrected kernel on page 16.3 never returns before its barrier: an out-of-range thread keeps a flag and skips the
          work, which costs nothing and holds on every backend.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.3, pp. 70–72; ch04-facts A6, B3, C3', // i18n-ignore: citation
  },
  'ch16/start/q-raw-only': {
    kind: 'choice',
    id: 'ch16/start/q-raw-only',
    prompt: (
      <Translated
        vi={
          <>
            Một khối chạy một vòng lặp. Ở mỗi bước, mọi luồng đọc các giá trị mà các luồng láng giềng đã ghi ở bước trước,
            và ghi kết quả của chính nó vào một vị trí bộ nhớ chia sẻ (shared memory) chưa luồng nào dùng trước đó. Mỗi bước
            cần bao nhiêu lời gọi <code>__syncthreads()</code>?
          </>
        }
      >
        <>
          A block runs a loop. In each step every thread reads values that its neighbours wrote in the previous step, and
          writes its own result to a shared-memory location that no thread has used before. How many{' '}
          <code>__syncthreads()</code> calls does each step need?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không cần: mỗi luồng chỉ ghi vị trí của riêng nó.">
            None: each thread writes only its own location.
          </Translated>
        ),
        why: (
          <Translated vi="Các luồng láng giềng đọc vị trí đó ở bước sau. Thiếu rào chắn, chúng có thể đọc trước khi nó được ghi.">
            Its neighbours read that location in the next step. Without a barrier they may read it before it is written.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một, giữa các lần ghi của một bước và các lần đọc của bước sau.">
            One, between one step's writes and the next step's reads.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Hai, một sau các lần ghi và một sau các lần đọc, như trong phép nhân ma trận phân ô.">
            Two, one after the writes and one after the reads, as in the tiled matrix multiplication.
          </Translated>
        ),
        why: (
          <Translated vi="Rào chắn thứ hai ở đó tồn tại vì các ô được tái sử dụng: một lần ghi không được đè lên thứ vẫn đang được đọc. Ở đây không vị trí nào bị ghi hai lần.">
            The second barrier there exists because the tiles are reused: a write must not overwrite what is still being
            read. Here no location is written twice.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không cần, nếu các luồng láng giềng nằm trong cùng một warp.">
            None, if neighbouring threads are in the same warp.
          </Translated>
        ),
        why: (
          <Translated vi="Các luồng ở rìa một warp có láng giềng trong warp khác, và chỉ rào chắn mới sắp thứ tự hai warp của một khối.">
            Threads at the edge of a warp have neighbours in another warp, and only a barrier orders two warps of a block.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Rào chắn bảo vệ một phụ thuộc thật: bên đọc cần thứ bên ghi tạo ra (đọc sau ghi). Phụ thuộc giả, ghi sau đọc,
            chỉ xuất hiện khi một vị trí được tái sử dụng, và ở đây không có vị trí nào như vậy. Đó là rào chắn bên trong một
            wavefront phân ô: mỗi bước ghi các ô con mới của ô, và bước sau đọc chúng (Figs. 16.9 và 16.16 đặt một{' '}
            <code>__syncthreads()</code> ở cuối mỗi bước bên trong một ô).
          </p>
        }
      >
        <p>
          The barrier guards a true dependence: readers need what writers produce (read after write). The false one,
          write after read, appears only when a location is reused, and here none is. That is the barrier inside a tiled
          wavefront: each step writes new cells of the tile, and the next step reads them (Figs. 16.9 and 16.16 put one{' '}
          <code>__syncthreads()</code> at the end of each step inside a tile).
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.4, pp. 110–111; ch. 16 §16.6, pp. 388–389', // i18n-ignore: citation
  },
  'ch16/start/q-stride16': {
    kind: 'numeric',
    id: 'ch16/start/q-stride16',
    prompt: (
      <Translated
        vi={
          <>
            32 luồng của một warp đọc một mảng <code>int</code> chia sẻ tại chỉ số <code>16*t + c</code>, trong đó{' '}
            <code>t</code> là lane và <code>c</code> như nhau với mọi lane. Bộ nhớ chia sẻ có 32 bank, mỗi bank 4 byte.
            Truy cập này là một xung đột bank (bank conflict) n chiều. n bằng bao nhiêu?
          </>
        }
      >
        <>
          The 32 threads of a warp read a shared <code>int</code> array at index <code>16*t + c</code>, where{' '}
          <code>t</code> is the lane and <code>c</code> is the same for every lane. Shared memory has 32 banks of 4 bytes.
          The access is an n-way bank conflict. What is n?
        </>
      </Translated>
    ),
    answer: 16,
    explain: (
      <Translated
        vi={
          <p>
            Bank là (16t + c) mod 32. Các lane chẵn đều rơi vào bank c, các lane lẻ vào bank c + 16: 16 lane trên mỗi bank
            trong hai bank, nên mỗi bank phục vụ 16 word khác nhau lần lượt. Sách đưa ra cùng trường hợp: một ô rộng 16
            cột, đọc dọc theo một cột. Trang 16.6 gặp lại nó: ô đã trượt được lưu dưới dạng hình vuông và đọc dọc theo một
            cột, với bước nhảy (stride) bằng độ rộng của ô, và sách thêm phần đệm vào các hàng để khắc phục.
          </p>
        }
      >
        <p>
          The bank is (16t + c) mod 32. Even lanes all land in bank c, odd lanes in bank c + 16: 16 lanes on each of two
          banks, so each bank serves 16 different words one after another. The book gives the same case: a tile 16 columns
          wide, read down a column. Page 16.6 meets it again: the sheared tile is stored as a square and read down a
          column, with a stride equal to the tile's width, and the book pads the rows to fix it.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.4, pp. 139–141 (the 16-column case, p. 141); infra-facts A10',
    vi: { source: 'ch. 6 §6.4, pp. 139–141 (trường hợp 16 cột, p. 141); infra-facts A10' },
  },
  'ch16/start/q-counter': {
    kind: 'choice',
    id: 'ch16/start/q-counter',
    prompt: (
      <Translated
        vi={
          <>
            Phép scan một lượt của chương 11 lấy chỉ số của mỗi khối từ một bộ đếm nguyên tử thay vì dùng{' '}
            <code>blockIdx.x</code>. Vì sao?
          </>
        }
      >
        <>
          Chapter 11's single-pass scan takes each block's index from an atomic counter instead of using{' '}
          <code>blockIdx.x</code>. Why?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Để một khối chỉ bao giờ chờ một khối đã bắt đầu chạy.">
            So that a block only ever waits for a block that has already started.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Để việc cập nhật cờ là nguyên tử, sao cho một khối đang chờ không bao giờ đọc phải một cờ ghi dở.">
            To make the flag update atomic, so that a waiting block never reads a half-written flag.
          </Translated>
        ),
        why: (
          <Translated vi="Cờ là một biến nguyên tử riêng. Bộ đếm quyết định khối nào chờ khối nào.">
            The flag is its own atomic. The counter decides which block waits for which.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Vì blockIdx.x không duy nhất khi số khối được khởi chạy nhiều hơn số khối có thể chạy cùng lúc.">
            Because blockIdx.x is not unique when more blocks are launched than can run at once.
          </Translated>
        ),
        why: (
          <Translated vi="Nó là duy nhất. Điều không gì bảo đảm là thứ tự các khối bắt đầu.">
            It is unique. What nothing promises is the order in which blocks start.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Để cân bằng công việc giữa các SM.">To balance the work between SMs.</Translated>,
        why: (
          <Translated vi="Mọi khối làm cùng một lượng việc. Điểm mấu chốt là ai chờ ai.">
            Every block does the same work. The point is who waits for whom.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Không gì bảo đảm khối <code>blockIdx.x − 1</code> đã bắt đầu, và một khối đang quay vòng chờ vẫn giữ chỗ của
            nó. Nếu các khối sau chiếm hết mọi chỗ và chờ một khối chưa bắt đầu, sẽ không có gì kết thúc. CUDA ghi nhận rằng
            một khối đã bắt đầu sẽ tiếp tục tiến triển, nên chỉ chờ những khối đã nhận chỉ số sớm hơn là an toàn. Trang 16.3
            áp dụng cùng lập luận cho các ô của một wavefront.
          </p>
        }
      >
        <p>
          Nothing promises that block <code>blockIdx.x − 1</code> has started, and a block that spins keeps its slot. If
          later blocks fill every slot and wait for one that has not started, nothing ever finishes. CUDA documents that a
          block which has started keeps making progress, so waiting only on blocks that claimed their index earlier is
          safe. Page 16.3 applies the same argument to tiles of a wavefront.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.9, pp. 276–282 (the counter, p. 280); ch11-facts A6, C5; ch04-facts A5',
    vi: { source: 'ch. 11 §11.9, pp. 276–282 (bộ đếm, p. 280); ch11-facts A6, C5; ch04-facts A5' },
  },
};

// ---------------------------------------------------------------- a small table, numbered by step
const CELL = 40;
const X0 = 34;
const Y0 = 30;

function SmallTable() {
  const ins = inputsOf(FIG, ...FIG_CELL);
  const cx = (c: number) => X0 + c * CELL + CELL / 2;
  const cy = (r: number) => Y0 + r * CELL + CELL / 2;
  return (
    <figure style={{ margin: '0.4rem 0 1.4rem' }}>
      <svg
        viewBox={`0 0 ${X0 + FIG.cols * CELL + 196} ${Y0 + FIG.rows * CELL + 12}`}
        role="img"
        aria-label={tr(
          `A ${FIG.rows} by ${FIG.cols} table, each cell numbered by the step at which it can run, from 0 to ${FIG.steps - 1}. Step ${FIG_FRONT} is outlined: ${FIG.widest} cells.`,
          `Một bảng ${FIG.rows} × ${FIG.cols}, mỗi ô ghi số bước mà nó có thể chạy, từ 0 đến ${FIG.steps - 1}. Bước ${FIG_FRONT} được viền đậm: ${FIG.widest} ô.`,
        )}
        style={{ width: '100%', maxWidth: 500, display: 'block', fontFamily: 'var(--sans)' }}
      >
        <defs>
          <marker id="s160-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute value */}
            <path d="M0,0 L10,5 L0,10 z" fill="var(--ink)" />
          </marker>
        </defs>
        <text x={X0} y={16} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
          <Translated vi="hàng 0 và cột 0 của bảng đầy đủ nằm ở phía trên và bên trái">
            row 0 and column 0 of the full table sit above and to the left
          </Translated>
        </text>
        {FIG.step.map((s, i) => {
          const r = Math.floor(i / FIG.cols);
          const c = i % FIG.cols;
          const front = s === FIG_FRONT;
          return (
            <g key={i} data-cell={`${r},${c}`} data-step={s}>
              <rect
                x={X0 + c * CELL + 1}
                y={Y0 + r * CELL + 1}
                width={CELL - 2}
                height={CELL - 2}
                rx="2"
                fill="var(--ink)"
                fillOpacity={0.04 + (0.3 * s) / (FIG.steps - 1)}
                stroke={front ? 'var(--ink)' : 'var(--rule)'}
                strokeWidth={front ? 2.4 : 1}
              />
              <text x={cx(c)} y={cy(r) + 4} fontSize="13" textAnchor="middle" style={{ fill: 'var(--ink)' }}>
                {s}
              </text>
            </g>
          );
        })}
        {ins.map(([r, c]) => (
          <line
            key={`${r},${c}`}
            data-input={`${r},${c}`}
            x1={cx(c) + (FIG_CELL[1] - c) * 9}
            y1={cy(r) + (FIG_CELL[0] - r) * 9}
            x2={cx(c) + (FIG_CELL[1] - c) * (CELL - 13)}
            y2={cy(r) + (FIG_CELL[0] - r) * (CELL - 13)}
            stroke="var(--ink)"
            strokeWidth="1.6"
            markerEnd="url(#s160-arrow)"
          />
        ))}
        <text x={X0 + FIG.cols * CELL + 12} y={Y0 + 18} fontSize="12" style={{ fill: 'var(--ink)' }}>
          <tspan x={X0 + FIG.cols * CELL + 12} dy="0"><Translated vi={<>viền đậm: bước {FIG_FRONT},</>}>outlined: step {FIG_FRONT},</Translated></tspan>
          <tspan x={X0 + FIG.cols * CELL + 12} dy="16"><Translated vi={<>{FIG.widest} ô không cần</>}>{FIG.widest} cells that need</Translated></tspan>
          <tspan x={X0 + FIG.cols * CELL + 12} dy="16"><Translated vi="gì từ nhau">nothing from each other</Translated></tspan>
          <tspan x={X0 + FIG.cols * CELL + 12} dy="28"><Translated vi="mũi tên: ba đầu vào">arrows: the three inputs</Translated></tspan>
          <tspan x={X0 + FIG.cols * CELL + 12} dy="16"><Translated vi={<>của ô ở bước {stepAt(FIG, ...FIG_CELL)}</>}>of the cell at step {stepAt(FIG, ...FIG_CELL)}</Translated></tspan>
          <tspan x={X0 + FIG.cols * CELL + 12} dy="28"><Translated vi={<>{FIG.steps} bước trong góc</>}>{FIG.steps} steps in this</Translated></tspan>
          <tspan x={X0 + FIG.cols * CELL + 12} dy="16"><Translated vi={<>{FIG.rows} × {FIG.cols} này</>}>{FIG.rows} × {FIG.cols} corner</Translated></tspan>
        </text>
      </svg>
      <figcaption className="muted" style={{ fontFamily: 'var(--sans)', fontSize: '0.8rem' }}>
        <Translated
          vi={
            <>
              {FIG.rows} × {FIG.cols} ô ở góc trên bên trái của một bảng như vậy, mỗi ô ghi số bước mà nó có thể chạy: lớn
              hơn một so với đầu vào muộn nhất trong ba đầu vào của nó. Càng đậm là càng muộn. Hình vẽ của khóa học này, được
              tính bằng mô hình đứng sau trình khám phá ở trang 16.3.
            </>
          }
        >
          The top-left {FIG.rows} × {FIG.cols} cells of such a table, each numbered by the step at which it can run: one
          more than the latest of its three inputs. Darker means later. Our drawing, computed by the model behind the
          explorer on page 16.3.
        </Translated>
      </figcaption>
    </figure>
  );
}

export function S160() {
  return (
    <>
      <Translated
        vi={
          <p>
            Hầu hết các kernel cho đến giờ đều có sẵn toàn bộ công việc ngay từ đầu: luồng nào cũng có thể chạy trước, theo
            thứ tự bất kỳ. Chương này nói về những phép tính mà điều đó không đúng. Mỗi ô của một bảng cần những ô được tính
            trước nó, nên tại mỗi thời điểm chỉ một phần công việc sẵn sàng. Các ô sẵn sàng cùng lúc tạo thành một{' '}
            <b>wavefront</b>, và kernel chạy hết bảng theo từng wavefront nối tiếp nhau.
          </p>
        }
      >
        <p>
          Most kernels so far had all of their work ready at the start: any thread could run first, in any order. This
          chapter is about computations where that is false. Each cell of a table needs cells computed before it, so at any moment only some
          of the work is ready. The cells that are ready together form a <b>wavefront</b>, and the kernel runs the table one
          wavefront after another.
        </p>
      </Translated>
      <Source section="16 (opening), 16.1" pages="373–374" />

      <h2><Translated vi="Một câu hỏi trước mọi lý thuyết">A question before any theory</Translated></h2>
      <Translated
        vi={
          <p>
            Hãy thử câu này trước khi đọc tiếp. Đó là cả chương gói trong một ví dụ, và nó chỉ cần đến hệ thức truy hồi.
          </p>
        }
      >
        <p>
          Try this before reading on. It is the chapter in one example, and it needs nothing but the recurrence.
        </p>
      </Translated>
      <Ask q={questionsS160['ch16/start/q-hook']} label={tr('Before any theory', 'Trước mọi lý thuyết')} />

      <h2><Translated vi="Chương này nói về gì">What the chapter is about</Translated></h2>
      <Translated
        vi={
          <p>
            Phương tiện của sách là quy hoạch động: một bài toán được chia thành những bài toán con nhỏ hơn chồng lấn nhau,
            nên mỗi đáp án được dùng lại nhiều lần và đáng được giữ trong một bảng. Khi được lấp từ bài toán con nhỏ nhất trở
            lên, bảng trở thành một tập các ô có phụ thuộc lẫn nhau, và những ô có mọi đầu vào sẵn sàng có thể được tính cùng
            lúc. Sách liệt kê đường đi ngắn nhất, nhân chuỗi ma trận, dynamic time warping, thuật toán Viterbi của mô hình
            Markov ẩn và căn chỉnh chuỗi (sequence alignment) là những bài toán thuộc loại này.
          </p>
        }
      >
        <p>
          The book's vehicle is dynamic programming: a problem split into smaller sub-problems that overlap, so that each
          answer is reused many times and is worth keeping in a table. Filled from the smallest sub-problem up, the table
          becomes a set of cells with dependencies between them, and the cells whose inputs are all ready can be computed
          at once. The book lists shortest paths, chain matrix multiplication, dynamic time warping, the Viterbi algorithm
          of hidden Markov models and sequence alignment as problems of this kind.
        </p>
      </Translated>
      <Source section="16.1" pages="373–374" />
      <Translated
        vi={
          <p>
            Đây là bảng của câu hỏi, cắt chỉ còn góc trên bên trái. Mỗi ô ghi số bước mà nó có thể chạy.
          </p>
        }
      >
        <p>
          Here is the table from the question, cut down to its top-left corner. Each cell carries the step at which it can
          run.
        </p>
      </Translated>
      <SmallTable />
      <Translated
        vi={
          <p>
            Fibonacci là trường hợp suy biến. Mỗi số cần hai số trước nó, nên mọi wavefront chỉ có một ô và không gì có thể
            chạy song song. Những bảng đáng dùng GPU có wavefront rộng nhiều ô, và chương này nói về việc giữ GPU bận rộn
            trên chúng trong khi vẫn chờ đúng những gì mỗi ô cần.
          </p>
        }
      >
        <p>
          Fibonacci is the degenerate case. Each number needs the two before it, so every wavefront is one cell and nothing
          can run in parallel. The tables worth a GPU have wavefronts many cells wide, and the chapter is about keeping a
          GPU busy on them while still waiting for exactly what each cell needs.
        </p>
      </Translated>
      <Source section="16.1, 16.2" pages="374, 376" figs={tr('Eq. 16.1', 'Phương trình 16.1')} />

      <h2><Translated vi="Vì sao nó có trong khóa học này">Why it is in this course</Translated></h2>
      <Translated
        vi={
          <p>
            Hai nghiên cứu tình huống của chương, đường đi ngắn nhất Floyd-Warshall và căn chỉnh chuỗi Smith-Waterman, không
            phải là phép toán suy luận, và bạn sẽ không viết cái nào cho một mô hình ngôn ngữ. Lý do để đọc nó là một kiểu
            vòng lặp bạn sẽ gặp trong mã suy luận: một hệ thức truy hồi dọc theo chuỗi, trong đó bước t cần bước t − 1. Trạng
            thái hồi quy của một lớp RWKV hay Mamba và quy tắc gated delta của các mô hình linear-attention thuộc loại này, và
            CTC loss của PyTorch, một hàm mục tiêu huấn luyện, cũng vậy.
          </p>
        }
      >
        <p>
          The chapter's two case studies, Floyd-Warshall's shortest paths and Smith-Waterman's sequence alignment, are not
          inference operations, and you will not write either one for a language model. The reason to read it is a kind of
          loop you will meet in inference code: a recurrence along the sequence, where step t needs step t − 1. The
          recurrent state of an RWKV or Mamba layer and the gated delta rule of linear-attention models are of this kind,
          and so is PyTorch's CTC loss, a training objective.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={<p>Sách không nhắc đến cái nào trong số này. Mối liên hệ, và mã được đọc cho nó, là của khóa học này (ch16-facts, nhóm D).</p>}
        >
          <p>The book mentions none of these. The link, and the code read for it, are ours (ch16-facts, group D).</p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Mỗi cái là một phép tính wavefront, và mỗi dự án chọn một cách chờ khác nhau giữa các bước: một rào chắn khối mỗi
            bước, một warp hay SIMD group cho mỗi chuỗi, hoặc các đoạn nối với nhau bằng các lần khởi chạy. Chương này cho bạn
            từ ngữ để đọc những lựa chọn đó.
          </p>
        }
      >
        <p>
          Each is a wavefront computation, and each project picks a different way to wait between steps: a block barrier
          per step, one warp or SIMD-group per chain, or chunks joined by launches. This chapter gives you the words to read
          those choices.
        </p>
      </Translated>
      <Today checked={tr("2026-11-20 in the projects' source (ch16-facts D1 to D3)", '2026-11-20 trong mã nguồn của các dự án (ch16-facts D1 đến D3)')}>
        <Translated
          vi={
            <p>
              llama.cpp <code>wkv.cu</code>, <code>wkv.metal</code>, <code>ssm-scan.cu</code>; mlx-lm{' '}
              <code>gated_delta.py</code>; CTC loss của PyTorch trong <code>LossCTC.cu</code> và <code>LossOps.metal</code>.
            </p>
          }
        >
          <p>
            llama.cpp <code>wkv.cu</code>, <code>wkv.metal</code>, <code>ssm-scan.cu</code>; mlx-lm{' '}
            <code>gated_delta.py</code>; PyTorch's CTC loss in <code>LossCTC.cu</code> and <code>LossOps.metal</code>.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Khóa học giữ lại gì">What the course keeps</Translated></h2>
      <Translated
        vi={
          <p>
            Hai ý tưởng. Thứ nhất, đồng bộ hóa đúng các phụ thuộc và không hơn, ở mức rẻ nhất đủ bao chúng: một lần khởi chạy
            lại, một rào chắn khối, một cờ giữa hai khối, hoặc không gì cả vì một warp sở hữu toàn bộ chuỗi. Thứ hai, định
            hình lại không gian lặp sao cho mọi bước đều có công việc đủ độ rộng: trước tiên cắt bảng thành các ô, rồi trượt
            các ô sao cho các wavefront của chúng đều dài bằng nhau.
          </p>
        }
      >
        <p>
          Two ideas. First, synchronise exactly the dependencies and no more, at the cheapest level that covers them: a
          relaunch, a block barrier, a flag between two blocks, or nothing at all because one warp owns the whole chain.
          Second, reshape the iteration space so that every step has full-width work: first cut the table into tiles, then
          shear the tiles so that their wavefronts all have the same length.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Trang <a href={href('ch16', '16.3')}>16.3</a> bàn ý thứ nhất: cách đánh số các ô theo bước mà chúng có thể chạy
            (§16.2 và §16.3), một lần khởi chạy mỗi bước của Floyd-Warshall và vì sao nó không cần buffer thứ hai (§16.4), và
            ai chờ ai giữa các ô (§16.8). Trang <a href={href('ch16', '16.6')}>16.6</a> bàn ý thứ hai trên bảng của
            Smith-Waterman: các ô vuông trong bộ nhớ chia sẻ, rồi các ô đã trượt.{' '}
            <a href={href('ch16', 'labs')}>Bài thực hành</a> dựng cả hai loại ô trên GPU của chính bạn.
          </p>
        }
      >
        <p>
          Page <a href={href('ch16', '16.3')}>16.3</a> covers the first: how to number cells by the step at which they can
          run (§16.2 and §16.3), Floyd-Warshall's one launch per step and why it needs no second buffer (§16.4), and who
          waits for whom between tiles (§16.8). Page <a href={href('ch16', '16.6')}>16.6</a> covers the second on
          Smith-Waterman's table: square tiles in shared memory, then sheared ones. The{' '}
          <a href={href('ch16', 'labs')}>lab</a> builds both kinds of tile on your own GPU.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Những gì bị bỏ ra: bối cảnh sinh học hệ gen và sơ đồ chấm điểm của Smith-Waterman (§16.5, pp. 382–384), bước truy
            vết ngược để khôi phục một phép căn chỉnh (p. 384), khung bên lề về stencil (p. 397), và các tối ưu hóa thêm của
            §16.8 dưới dạng mã: phiên bản cooperative groups, kernel dựa trên cờ và lệnh DPX (pp. 397–399) xuất hiện dưới dạng
            ý tưởng và lịch chạy, không phải kernel. Phần <a href={href('ch16', 'wrap')}>tổng kết</a> chỉ ra nơi từng bài
            trong sáu bài tập của sách được giải và liệt kê các chỗ sơ suất trong PDF.
          </p>
        }
      >
        <p>
          What is left out: the genomics background and the scoring scheme of Smith-Waterman (§16.5, pp. 382–384), the
          traceback that recovers an alignment (p. 384), the stencil sidebar (p. 397), and the further optimisations of
          §16.8 as code: the cooperative-groups version, the flag-based kernel and the DPX instruction (pp. 397–399) appear
          as ideas and schedules, not as kernels. The <a href={href('ch16', 'wrap')}>summary</a> maps the book's six
          exercises to where each is worked and lists the slips in the PDF.
        </p>
      </Translated>

      <h2><Translated vi="Trước khi bắt đầu">Before you start</Translated></h2>
      <Translated
        vi={
          <p>
            Năm câu hỏi về những gì chương này lấy từ các chương 2, 4, 5, 6 và 11, theo thứ tự chương dùng đến chúng. Mỗi câu
            sẽ quay lại trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Five questions on what this chapter takes from chapters 2, 4, 5, 6 and 11, in the order the chapter uses them.
          Each one comes back in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={questionsS160['ch16/start/q-stream-order']} label={tr('From chapter 2', 'Từ chương 2')} />
      <Ask q={questionsS160['ch16/start/q-exit-barrier']} label={tr('From chapter 4', 'Từ chương 4')} />
      <Ask q={questionsS160['ch16/start/q-raw-only']} label={tr('From chapter 5', 'Từ chương 5')} />
      <Ask q={questionsS160['ch16/start/q-stride16']} label={tr('From chapter 6', 'Từ chương 6')} />
      <Ask q={questionsS160['ch16/start/q-counter']} label={tr('From chapter 11', 'Từ chương 11')} />
    </>
  );
}
