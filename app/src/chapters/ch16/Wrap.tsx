import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { presetFronts, stencil, wavefronts } from '../../sim/wavefront';
import { launches, stepsPerTile } from '../../sim/wavetile';

// Chapter 16's summary page: what to carry forward, a mixed check on situations the pages
// did not show, where the book's exercises are worked, the PDF's apparent slips, what the
// condensed chapter left out, and the project status. No kernel is shown for a question
// here, so there is no snippet to tie.

// The H.264 question: square tiles over a table whose cells read W and NE. A tile then reads
// the tile on its left, the tile above and the tile above-right: the stencil W, N, NE one
// level up. src/sim/wavefront.ts numbers the tiles.
const H264_T = 4;
const H264_TILES = wavefronts(H264_T, H264_T, stencil(['W', 'N', 'NE']));
const SW_TILES = presetFronts('smith-waterman', H264_T, H264_T);

// Book exercise 2: rectangular tiles of RECT.H rows (one thread each) by RECT.W columns.
const RECT = { Lseq: 1000, H: 16, W: 64 };
const RECT_ROWS = Math.ceil(RECT.Lseq / RECT.H); // 63 tile rows
const RECT_COLS = Math.ceil(RECT.Lseq / RECT.W); // 16 tile columns
const RECT_LAUNCHES = RECT_ROWS + RECT_COLS - 1; // 78
const RECT_STEPS = RECT.H + RECT.W - 1; // 79
const SQ16 = { launches: launches({ Lseq: RECT.Lseq, W: 16, scheme: 'square' }), steps: stepsPerTile({ W: 16, scheme: 'square' }) };
const SQ64 = { launches: launches({ Lseq: RECT.Lseq, W: 64, scheme: 'square' }), steps: stepsPerTile({ W: 64, scheme: 'square' }) };
const RECT_BUSY = Math.round((RECT.H * RECT.W) / RECT_STEPS); // 13
const SQ16_BUSY = Math.round((16 * 16) / SQ16.steps); // 8

export const questionsWrap: Record<string, Question> = {
  'ch16/wrap/q-h264-tiles': {
    kind: 'numeric',
    id: 'ch16/wrap/q-h264-tiles',
    prompt: (
      <Translated
        vi={
          <>
            Một bảng mà các ô đọc láng giềng bên trái và ô phía trên bên phải (mẫu H.264 ở trang 16.3) được cắt thành các ô (tile)
            vuông, một khối mỗi ô, với một lần khởi chạy giữa các mặt sóng của ô như ở trang 16.6. Với {H264_T} ô mỗi cạnh, số lần
            khởi chạy ít nhất để mọi ô luôn chạy sau các ô mà nó đọc là bao nhiêu?
          </>
        }
      >
      <>
        A table whose cells read their left neighbour and the cell above and to the right (the H.264 pattern on page
        16.3) is cut into square tiles, one block per tile, with a launch between fronts of tiles as on page 16.6. With{' '}
        {H264_T} tiles per side, what is the fewest launches that keeps every tile after the tiles it reads?
      </>
      </Translated>
    ),
    answer: H264_TILES.steps,
    unit: 'launches',
    explain: (
      <Translated
        vi={
          <p>
            Một ô đọc từ chính các ô con của nó, từ ô bên trái, từ ô phía trên, và, qua ô con cuối của hàng trên cùng, từ ô phía
            trên bên phải. Ô cuối này có cùng tile_row + tile_col với chính ô đó, nên các mặt sóng của trang 16.6 sẽ đặt hai ô vào
            cùng một lần khởi chạy: ô (1, 0) đọc một ô con của ô (0, 1). Thay vào đó, hãy đánh số các ô theo tile_col + 2·tile_row.
            Số đó chạy từ 0 đến {H264_TILES.steps - 1}: {H264_TILES.steps} lần khởi chạy, mỗi lần tối đa {H264_TILES.widest} ô, so
            với {SW_TILES.steps} lần khởi chạy, mỗi lần tối đa {SW_TILES.widest} ô với stencil ba láng giềng. Đó là lịch chạy của
            ô đã trượt, <code>tile_col = d − 2·tile_row</code>, vì cùng lý do: một phụ thuộc hướng lên trên và sang phải.
          </p>
        }
      >
      <p>
        A tile reads from its own cells, from the tile on its left, from the tile above, and, through the last cell of its
        top row, from the tile above and to the right. That last one has the same tile_row + tile_col as the tile itself,
        so page 16.6's fronts would put the two in one launch: tile (1, 0) reads a cell of tile (0, 1). Number the tiles by
        tile_col + 2·tile_row instead. That runs from 0 to {H264_TILES.steps - 1}: {H264_TILES.steps} launches of at most{' '}
        {H264_TILES.widest} tiles, against {SW_TILES.steps} launches of up to {SW_TILES.widest} for the three-neighbour
        stencil. It is the schedule of sheared tiles, <code>tile_col = d − 2·tile_row</code>, for the same reason: a
        dependency up and to the right.
      </p>
      </Translated>
    ),
    source: 'ch. 16 §16.3, Fig. 16.2(d), and §16.7, p. 392; tiling the H.264 stencil is ours, numbered by wavefront.ts',
    vi: { unit: 'lần khởi chạy', source: 'ch. 16 §16.3, Fig. 16.2(d), và §16.7, p. 392; việc phân ô stencil H.264 là của khóa học này, được đánh số bởi wavefront.ts' },
  },
  'ch16/wrap/q-stencil-barriers': {
    kind: 'choice',
    id: 'ch16/wrap/q-stencil-barriers',
    prompt: (
      <Translated
        vi={
          <>
            Khung bên lề về stencil của sách lặp lại <code>A[i] = 0.5 * (A[i] + A[i+1])</code> cho mọi i, mỗi bước thời gian một
            lần. Một khối giữ A trong bộ nhớ chia sẻ (shared memory), một luồng mỗi i, và cập nhật nó tại chỗ. Mỗi bước thời gian
            cần bao nhiêu <code>__syncthreads()</code>?
          </>
        }
      >
      <>
        The book's stencil sidebar repeats <code>A[i] = 0.5 * (A[i] + A[i+1])</code> for every i, once per time step. One
        block holds A in shared memory, one thread per i, and updates it in place. How many{' '}
        <code>__syncthreads()</code> does each time step need?
      </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Một, sau các lần ghi, để bước thời gian sau đọc được các giá trị bước này đã ghi."}>{'One, after the writes, so that the next time step reads the values this step wrote.'}</Translated>,
        why: <Translated vi={"Luồng i − 1 đọc A[i] trong cùng bước mà luồng i ghi đè nó. Phải có gì đó giữ lần ghi đó lại cho đến khi lần đọc xong."}>{'Thread i − 1 reads A[i] in the same step in which thread i overwrites it. Something must hold that write back until the read is done.'}</Translated>,
      },
      {
        label: <Translated vi="Hai: một giữa các lần đọc của mỗi luồng và lần ghi của nó, và một sau các lần ghi.">{'Two: one between every thread\'s reads and its write, and one after the writes.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Không cần, nếu mỗi luồng đọc A[i] và A[i + 1] vào thanh ghi trước khi ghi."}>{'None, if each thread reads A[i] and A[i + 1] into registers before it writes.'}</Translated>,
        why: <Translated vi={"Điều đó sắp thứ tự lần đọc và ghi của chính một luồng. Nó không làm gì cho luồng i − 1, luồng có thể chưa đọc A[i]."}>{'That orders one thread\'s own read and write. It does nothing for thread i − 1, which may not have read A[i] yet.'}</Translated>,
      },
      {
        label: <Translated vi={"Không cần trong một warp: các luồng của nó chạy cùng nhau, nên chỉ rìa warp cần chú ý."}>{'None inside a warp: its threads run together, so only warp edges need care.'}</Translated>,
        why: <Translated vi={"Không gì trong mô hình lập trình bảo đảm rằng mọi luồng của một warp đọc trước khi bất kỳ luồng nào ghi, và các luồng ở rìa warp đọc từ warp kế tiếp."}>{'Nothing in the programming model promises that every thread of a warp reads before any of them writes, and threads at a warp\'s edge read from the next warp.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hai phụ thuộc, hai rào chắn (barrier). Bước t + 1 đọc những gì bước t đã ghi: đọc sau ghi, một rào chắn sau các lần
            ghi. Bên trong một bước, luồng i ghi đè A[i] trong khi luồng i − 1 vẫn cần giá trị cũ: ghi sau đọc, một rào chắn giữa
            các lần đọc và các lần ghi. Đó chính là cặp rào chắn của phép nhân ma trận phân ô, ở đây trong một wavefront dọc theo
            thời gian. Hai mảng, đọc mảng này và ghi mảng kia rồi hoán đổi mỗi bước (double buffering), loại bỏ phụ thuộc thứ hai
            và chỉ để lại một rào chắn mỗi bước. Ý chính của khung bên lề còn đi xa hơn: các ô được trượt trong mặt phẳng thời
            gian–không gian cần ít lần đồng bộ hóa toàn cục hơn.
          </p>
        }
      >
      <p>
        Two dependences, two barriers. Step t + 1 reads what step t wrote: read after write, one barrier after the writes.
        Inside a step, thread i overwrites A[i] while thread i − 1 still needs the old value: write after read, one barrier
        between the reads and the writes. That is the pair of barriers of the tiled matrix multiplication, here in a
        wavefront along time. Two arrays, reading one and writing the other and swapping them each step (double
        buffering), remove the second dependence and leave one barrier per step. The sidebar's own point goes further:
        tiles sheared in the time–space plane need fewer global synchronisations.
      </p>
      </Translated>
    ),
    source: 'ch. 16 sidebar, p. 397; ch. 5 §5.4, pp. 110–111; ch. 6 double buffering (§16.4 points there, p. 380)',
    vi: { source: 'ch. 16 khung bên lề, p. 397; ch. 5 §5.4, pp. 110–111; ch. 6 double buffering (§16.4 dẫn tới đó, p. 380)' },
  },
  'ch16/wrap/q-viterbi': {
    kind: 'choice',
    id: 'ch16/wrap/q-viterbi',
    prompt: (
      <Translated
        vi={
          <>
            Giải mã Viterbi, một trong các ví dụ về quy hoạch động của sách: ở mỗi bước thời gian, mỗi trạng thái trong S trạng
            thái lấy giá trị lớn nhất qua cả S trạng thái của bước trước. S quá lớn cho một khối, nên mỗi bước được chia cho 4
            khối. Một cờ (flag) cho mỗi khối có cho phép một khối bắt đầu bước t sớm hơn so với khởi chạy lại mỗi bước không?
          </>
        }
      >
      <>
        Viterbi decoding, one of the book's examples of dynamic programming: at every time step each of S states takes a
        maximum over all S states of the step before. S is too large for one block, so each step is split over 4 blocks.
        Would a flag per block let a block start step t earlier than a relaunch per step does?
      </>
      </Translated>
    ),
    options: [
      {
        label:
          <Translated vi={"Không: mỗi khối của bước t cần cả bốn khối của bước t − 1, nên nó vẫn chờ tất cả dù theo cách nào. Cờ tiết kiệm lần khởi chạy, không bớt việc chờ."}>{'No: each block of step t needs all four blocks of step t − 1, so it waits for all of them either way. Flags save launches, not the wait.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Có: với cờ, một khối chỉ chờ các khối mà nó đọc, nên nó bắt đầu khi khối của chính nó ở bước t − 1 đã xong."}>{'Yes: with flags a block waits only for the blocks it reads, so it starts once its own block of step t − 1 is done.'}</Translated>,
        why: <Translated vi={"Nó đúng là chỉ chờ các khối nó đọc, nhưng ở đây nó đọc mọi trạng thái, nên là mọi khối của bước t − 1."}>{'It does wait only for the blocks it reads, but here it reads every state, so every block of step t − 1.'}</Translated>,
      },
      {
        label: <Translated vi={"Có, miễn là các khối lấy việc từ một bộ đếm nguyên tử theo thứ tự bước, để mỗi khối chỉ chờ các khối đã bắt đầu."}>{'Yes, as long as blocks take their work from an atomic counter in step order, so each waits only on blocks that have already started.'}</Translated>,
        why: <Translated vi={"Bộ đếm làm việc chờ an toàn trên CUDA. Nó không thu nhỏ những gì một khối chờ."}>{'The counter makes waiting safe on CUDA. It does not shrink what a block waits for.'}</Translated>,
      },
      {
        label: <Translated vi={"Không, vì Viterbi không có phụ thuộc giữa các bước thời gian khi đã biết điểm phát xạ."}>{'No, because Viterbi has no dependency between time steps once the emission scores are known.'}</Translated>,
        why: <Translated vi={"Có: mỗi trạng thái ở bước t đọc mọi trạng thái ở bước t − 1."}>{'It does: each state at step t reads every state at step t − 1.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đồng bộ hóa đúng các phụ thuộc: ở đây một bước phụ thuộc vào toàn bộ bước trước, tất cả với tất cả, giống hình dạng các
            bước của Floyd-Warshall hơn là các láng giềng của Smith-Waterman. Chờ cả bốn cờ là một rào chắn trên bốn khối, nên nó
            chỉ có thể loại bỏ chi phí khởi chạy, như <code>grid.sync()</code> sẽ làm. Điều thay đổi bức tranh là công việc mỗi lần
            khởi chạy: một khối mỗi chuỗi với một rào chắn khối mỗi bước, như CTC loss của PyTorch làm, và nhiều chuỗi chạy song
            song để mỗi lần khởi chạy lấp đầy GPU.
          </p>
        }
      >
      <p>
        Synchronise exactly the dependencies: here a step depends on the whole previous step, all to all, the shape of
        Floyd-Warshall's steps rather than Smith-Waterman's neighbours. A wait on all four flags is a barrier over the four
        blocks, so it can only remove the launch overhead, as <code>grid.sync()</code> would. What changes the picture is
        the work per launch: one block per sequence with a block barrier per step, as PyTorch's CTC loss does, and many
        sequences side by side so that each launch fills the GPU.
      </p>
      </Translated>
    ),
    source: 'ch. 16 §16.1, p. 374 (Viterbi listed), §16.4 and §16.8, pp. 379, 397–398; the Viterbi analysis is ours',
    vi: { source: 'ch. 16 §16.1, p. 374 (có liệt kê Viterbi), §16.4 và §16.8, pp. 379, 397–398; phân tích Viterbi là của khóa học này' },
  },
  'ch16/wrap/q-causal': {
    kind: 'choice',
    id: 'ch16/wrap/q-causal',
    prompt: (
      <Translated
        vi={
          <>
            Prefill của một lớp causal attention tính một ma trận điểm số tam giác dưới: hàng i dùng các key 0 đến i. Đó có phải
            là một phép tính wavefront theo nghĩa của chương này không?
          </>
        }
      >
      <>
        The prefill of a causal attention layer computes a lower-triangular matrix of scores: row i uses keys 0 to i. Is
        that a wavefront computation in this chapter's sense?
      </>
      </Translated>
    ),
    options: [
      {
        label:
          <Translated vi={"Không: không điểm số nào đọc điểm số khác, nên mọi ô có thể chạy cùng lúc. Hình tam giác làm công việc không đều, không tạo thứ tự."}>{'No: no score reads another score, so every cell can run at once. The triangle makes the work uneven, not ordered.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Có: hàng i cần các hàng phía trên nó, nên các hàng là các mặt sóng, với một rào chắn sau mỗi hàng."}>{'Yes: row i needs the rows above it, so the rows are the fronts, with a barrier after each one.'}</Translated>,
        why: <Translated vi={"Hàng i cần các key 0 đến i, do lớp bên dưới tạo ra. Nó không cần kết quả nào của hàng khác."}>{'Row i needs keys 0 to i, which the layer below produced. It needs no result of another row.'}</Translated>,
      },
      {
        label: <Translated vi={"Có, và trượt hình tam giác, như §16.7 trượt các ô, sẽ làm mọi bước đủ độ rộng."}>{'Yes, and shearing the triangle, as §16.7 shears tiles, would make every step full width.'}</Translated>,
        why: <Translated vi={"Phép trượt lấp những bước mà phụ thuộc buộc phải có. Ở đây không có bước bị buộc, chỉ có những khối có nhiều ô hơn các khối khác."}>{'Shearing fills steps that dependencies force. Here there are no forced steps, only blocks with more cells than others.'}</Translated>,
      },
      {
        label: <Translated vi={"Chỉ khi decode, khi hình tam giác dài thêm một hàng mỗi token và mỗi hàng mới đọc điểm số của hàng trước nó."}>{'Only during decoding, when the triangle grows by one row per token and each new row reads the scores of the row before it.'}</Translated>,
        why: <Translated vi={"Decode là tuần tự, nhưng phụ thuộc chạy qua toàn bộ mô hình, từ đầu ra của một token đến đầu vào của token kế tiếp. Bên trong ma trận điểm số của một lớp, không ô nào đọc ô khác."}>{'Decoding is sequential, but the dependency runs through the whole model, from one token\'s output to the next token\'s input. Inside one layer\'s score matrix no cell reads another.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một wavefront cần những ô đọc kết quả của ô khác. Một điểm số là tích vô hướng của một query và một key; softmax và
            tổng có trọng số sau đó chỉ đọc điểm số của cùng hàng. Vậy mọi hàng đều độc lập, và hình tam giác chỉ làm công việc
            không đều: các khối phụ trách các hàng trên cùng xong sớm. Một hệ thức truy hồi dọc chuỗi, như trạng thái của một lớp
            Mamba hay RWKV hoặc bảng CTC của PyTorch, là một wavefront, với một mặt sóng mỗi token.
          </p>
        }
      >
      <p>
        A wavefront needs cells that read other cells' results. A score is a dot product of a query and a key; the softmax
        and the weighted sum then read only scores of the same row. So every row is independent, and the triangle only
        makes the work uneven: blocks over the top rows finish early. A recurrence along the sequence, such as the state
        of a Mamba or RWKV layer or PyTorch's CTC table, is a wavefront, with one front per token.
      </p>
      </Translated>
    ),
    source: 'ours: the chapter\'s definition of a wavefront (§16.2–16.3, pp. 376–377) applied to attention; ch16-facts D1, D2',
    vi: { source: 'của khóa học này: định nghĩa wavefront của chương (§16.2–16.3, pp. 376–377) áp dụng cho attention; ch16-facts D1, D2' },
  },
  'ch16/wrap/q-ex2': {
    kind: 'numeric',
    id: 'ch16/wrap/q-ex2',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 2 của sách yêu cầu các ô hình chữ nhật. Giữ một luồng mỗi hàng của ô, để một khối {RECT.H} luồng lấp một ô
            {RECT.H} hàng × {RECT.W} cột, và giữ một lần khởi chạy mỗi mặt sóng của ô. Với L_seq ={' '}
            {RECT.Lseq.toLocaleString('en-US')}, bảng cần bao nhiêu lần khởi chạy?
          </>
        }
      >
      <>
        The book's exercise 2 asks for rectangular tiles. Keep one thread per tile row, so a block of {RECT.H} threads
        fills a tile of {RECT.H} rows by {RECT.W} columns, and keep a launch per front of tiles. With L_seq ={' '}
        {RECT.Lseq.toLocaleString('en-US')}, how many launches does the table take?
      </>
      </Translated>
    ),
    answer: RECT_LAUNCHES,
    unit: 'launches',
    explain: (
      <Translated
        vi={
          <p>
            ⌈{RECT.Lseq.toLocaleString('en-US')} / {RECT.H}⌉ = {RECT_ROWS} hàng ô và ⌈{RECT.Lseq.toLocaleString('en-US')} /{' '}
            {RECT.W}⌉ = {RECT_COLS} cột ô; các mặt sóng vẫn là tile_row + tile_col, từ 0 đến {RECT_ROWS - 1} +{' '}
            {RECT_COLS - 1}: {RECT_LAUNCHES} lần khởi chạy, mỗi lần tối đa {Math.min(RECT_ROWS, RECT_COLS)} ô. Bên trong một ô,
            các bước chạy dọc các đường chéo phụ của nó, {RECT.H} + {RECT.W} − 1 = {RECT_STEPS} bước, và {RECT.H * RECT.W} ô con
            giữ trung bình khoảng {RECT_BUSY} trên {RECT.H} luồng bận, so với khoảng {SQ16_BUSY} trong một ô vuông 16 × 16. Ô
            vuông cạnh 16 cần {SQ16.launches} lần khởi chạy; cạnh 64 cần {SQ64.launches} lần khởi chạy nhưng khối 64 luồng và{' '}
            {SQ64.steps} bước mỗi ô. Những gì khác thay đổi: chỉ số trong ô trở thành{' '}
            <code>swTile[r_tile * {RECT.W} + q_tile]</code>, phép kiểm tra biên là <code>q_tile &lt; {RECT.W}</code>, có hai số
            đếm ô thay vì một, và khi lưu, {RECT.H} luồng phủ {RECT.W} cột, mỗi luồng {RECT.W / RECT.H} cột mỗi hàng.
          </p>
        }
      >
      <p>
        ⌈{RECT.Lseq.toLocaleString('en-US')} / {RECT.H}⌉ = {RECT_ROWS} tile rows and ⌈{RECT.Lseq.toLocaleString('en-US')} /{' '}
        {RECT.W}⌉ = {RECT_COLS} tile columns; the fronts are still tile_row + tile_col, from 0 to {RECT_ROWS - 1} +{' '}
        {RECT_COLS - 1}: {RECT_LAUNCHES} launches of at most {Math.min(RECT_ROWS, RECT_COLS)} tiles. Inside a tile the steps
        run along its anti-diagonals, {RECT.H} + {RECT.W} − 1 = {RECT_STEPS} of them, and the {RECT.H * RECT.W} cells keep
        about {RECT_BUSY} of the {RECT.H} threads busy on average, against about {SQ16_BUSY} in a 16 × 16 square. Square tiles of 16 need {SQ16.launches} launches; of 64, {SQ64.launches} launches but blocks of 64 threads
        and {SQ64.steps} steps per tile. What else changes: the tile index becomes <code>swTile[r_tile * {RECT.W} + q_tile]</code>,
        the bounds test <code>q_tile &lt; {RECT.W}</code>, there are two tile counts instead of one, and in the store{' '}
        {RECT.H} threads cover {RECT.W} columns, {RECT.W / RECT.H} per thread per row.
      </p>
      </Translated>
    ),
    source: 'ch. 16 exercise 2, p. 399; the analysis is ours (notes/ch16.md, Exercises)',
    vi: { unit: 'lần khởi chạy', source: 'ch. 16 bài tập 2, p. 399; phân tích là của khóa học này (notes/ch16.md, Exercises)' },
  },
};

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports);
  const lab = 'ch16/lab1_wavefront';
  const passing = [...backends, 'emu'].filter((b) => labStatus(reports[b], lab, 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch16/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}><Translated vi="Những gì mang theo">What to carry forward</Translated></h2>
      <ul>
        <li>
          <Translated
            vi={
              <>
                Wavefront là tập các ô có mọi đầu vào đã xong. Đánh số mỗi ô lớn hơn một so với đầu vào muộn nhất của nó: riêng
                stencil quyết định một bảng cần bao nhiêu bước và mỗi bước rộng bao nhiêu.
              </>
            }
          >
          A wavefront is the set of cells whose inputs are all done. Number every cell one more than its latest input: the
          stencil alone decides how many steps a table takes and how wide each one is.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Đồng bộ hóa đúng các phụ thuộc, ở mức rẻ nhất đủ bao chúng: khởi chạy lại khi cả lưới phải chờ toàn bộ mặt sóng trước,
                rào chắn khối giữa các bước bên trong một khối, một cờ khi một ô chỉ chờ các láng giềng của nó, không gì cả khi một
                warp hay một luồng sở hữu toàn bộ chuỗi.
              </>
            }
          >
          Synchronise exactly the dependencies, at the cheapest level that covers them: a relaunch when the whole grid
          must wait for the whole previous front, a block barrier between steps inside a block, a flag when a tile waits
          only for its neighbours, nothing when one warp or thread owns the whole chain.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Chờ xuyên khối chỉ an toàn với công việc đã bắt đầu: các ô lấy từ một bộ đếm nguyên tử, hoặc một lần khởi chạy
                cooperative, trên CUDA; unverified trên HIP; trên Metal, một dispatch mỗi mặt sóng.
              </>
            }
          >
          Waiting across blocks is safe only on work that has already started: tiles from an atomic counter, or a
          cooperative launch, on CUDA; unverified on HIP; on Metal, one dispatch per front.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Một cập nhật tại chỗ an toàn khi không gì mà một bước đọc thay đổi trong bước đó, như hàng k và cột k của
                Floyd-Warshall không đổi khi dist[k][k] bằng 0 và không trọng số nào âm. Nếu không, dùng hai buffer.
              </>
            }
          >
          An in-place update is safe when nothing a step reads changes during the step, as Floyd-Warshall's row k and
          column k do not when dist[k][k] is 0 and no weight is negative. Otherwise, two buffers.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Một wavefront phân ô có hai mức: các lần khởi chạy giữa các mặt sóng của ô, các rào chắn giữa các bước bên trong một
                ô. Giữ ô trong bộ nhớ chia sẻ và lưu nó với luồng = cột, để mỗi hàng của ô đi ra được hợp nhất (coalesced).
              </>
            }
          >
          A tiled wavefront has two levels: launches between fronts of tiles, barriers between steps inside a tile. Keep
          the tile in shared memory and store it with thread = column, so each tile row leaves coalesced.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Trượt (shear) một ô lấp đầy mọi bước, W bước thay vì 2W − 1, và tốn 3T − 1 lần khởi chạy với ít ô hơn mỗi lần. Đổi
                cách bố trí là dời các xung đột bank (bank conflict): đệm sửa phép duyệt theo cột của ô đã trượt và làm hỏng phép duyệt
                đường chéo phụ của ô vuông. Hãy kiểm tra bước nhảy (stride), và trên Metal, hãy đo.
              </>
            }
          >
          Shearing a tile fills every step, W steps instead of 2W − 1, and costs 3T − 1 launches with fewer tiles each.
          Moving a layout moves bank conflicts: padding fixes the sheared tile's column walk and breaks the square tile's
          anti-diagonal one. Check the stride, and on Metal, measure.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Trong mã suy luận, các wavefront là những hệ thức truy hồi dọc theo chuỗi (trạng thái RWKV và Mamba, quy tắc gated
                delta), và CTC loss của PyTorch, một hàm mục tiêu huấn luyện, cũng là một wavefront. Chúng chạy với một rào chắn khối
                mỗi bước, một SIMD group hay warp mỗi chuỗi, hoặc các đoạn nối với nhau bằng các lần khởi chạy.
              </>
            }
          >
          In inference code the wavefronts are recurrences along the sequence (RWKV and Mamba state, the gated delta
          rule), and PyTorch's CTC loss, a training objective, is one too. They run with a block barrier per step, a
          SIMD-group or warp per chain, or chunks joined by launches.
          </Translated>
        </li>
      </ul>
      <Source section="16.9" pages="399" />

      <h2><Translated vi="Một bài kiểm tra tổng hợp">A mixed check</Translated></h2>
      <Translated
        vi={
          <p>
            Các câu này dựa trên toàn bộ chương và trên những tình huống mà các trang chưa cho thấy.
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
              Việc phân ô stencil H.264, cách chia Viterbi và câu hỏi về attention là của khóa học này. Vòng lặp stencil là khung bên
              lề của sách; bài tập 2 là của sách, lời giải là của khóa học này.
            </p>
          }
        >
        <p>
          Tiling the H.264 stencil, the Viterbi split and the attention question are ours. The stencil loop is the book's
          sidebar; exercise 2 is the book's, its answer ours.
        </p>
        </Translated>
      </Beyond>
      <Ask q={q['ch16/wrap/q-h264-tiles']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={q['ch16/wrap/q-stencil-barriers']} />
      <Ask q={q['ch16/wrap/q-viterbi']} />
      <Ask q={q['ch16/wrap/q-causal']} label={tr('In an LLM', 'Trong một LLM')} />
      <Ask q={q['ch16/wrap/q-ex2']} label={tr('Book exercise 2', 'Bài tập 2 của sách')} />

      <h2><Translated vi="Các bài tập của sách">The book's exercises</Translated></h2>
      <Translated
        vi={
          <p>
            Chương kết thúc bằng sáu bài tập (pp. 399–400), và đặt thêm hai bài trong phần chữ. Tất cả đều yêu cầu viết mã; sách
            không in đáp án, nên đáp án ở đây là của khóa học này. Lab 16.1 dựng hai bài được đặt trong phần chữ; sáu bài ở cuối
            được phân tích, không được dựng, và bảng cho biết ở đâu.
          </p>
        }
      >
      <p>
        The chapter ends with six exercises (pp. 399–400), and sets two more in the text. All of them ask for code; the
        book prints no answers, so the answers here are ours. Lab 16.1 builds the two set in the text; the six at the end are
        analysed, not built, and the table says where.
      </p>
      </Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
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
              <td><Translated vi="Floyd-Warshall với khối vuông, hàng k và cột k trong bộ nhớ chia sẻ">Floyd-Warshall with square blocks, row k and column k in shared memory</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      câu hỏi "Bài tập 1 của sách" ở <a href={href('ch16', '16.3')}>16.3</a>: số lần đọc mỗi ô và những gì phép đếm bỏ qua;
                      không được dựng
                    </>
                  }
                >
                the "Book exercise 1" question in <a href={href('ch16', '16.3')}>16.3</a>: reads per cell and what the count
                leaves out; not built
                </Translated>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td><Translated vi="Ô hình chữ nhật">Rectangular tiles</Translated></td>
              <Translated
                vi={
                  <td>câu hỏi "Bài tập 2 của sách" ở trên; không được dựng</td>
                }
              >
              <td>the "Book exercise 2" question above; not built</td>
              </Translated>
            </tr>
            <tr>
              <td>3</td>
              <td><Translated vi="Cooperative groups thay cho khởi chạy lại">Cooperative groups instead of a relaunch</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      "Ai chờ ai" ở <a href={href('ch16', '16.3')}>16.3</a>: rào chắn toàn lưới, những gì nó tiết kiệm và không
                      tiết kiệm, và dạng của nó trên từng backend (không có trên Metal); API là §18.7 của sách, được giải ở trang{' '}
                      <a href={href('ch18', '18.6')}>18.6, "Hàng đợi riêng và một lần khởi chạy cho mọi mức"</a>; không được dựng
                    </>
                  }
                >
                "Who waits for whom" in <a href={href('ch16', '16.3')}>16.3</a>: the grid barrier, what it saves and does
                not, and its form on each backend (none on Metal); the API is the book's §18.7, worked on page{' '}
                <a href={href('ch18', '18.6')}>18.6, "Private queues and one launch for every level"</a>; not built
                </Translated>
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td><Translated vi="Cờ giữa các ô vuông">Flags between square tiles</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      hoạt hình "Waiting for everyone, or only for your neighbours", trình khám phá "Ai chờ ai" và câu hỏi về hàng
                      thường trú ở <a href={href('ch16', '16.3')}>16.3</a>; không được dựng, vì chỉ CUDA ghi nhận một dạng an toàn
                    </>
                  }
                >
                the animation "Waiting for everyone, or only for your neighbours", the "Who waits for whom" explorer and the persistent-row question in{' '}
                <a href={href('ch16', '16.3')}>16.3</a>; not built, because only CUDA documents a safe form
                </Translated>
              </td>
            </tr>
            <tr>
              <td>5</td>
              <td><Translated vi="Cờ giữa các ô đã trượt">Flags between sheared tiles</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      phụ thuộc nằm ở <a href={href('ch16', '16.6')}>16.6</a> (một ô đã trượt đọc ô bên trái và ô phía trên bên phải); câu
                      hỏi H.264 ở trên có cùng phụ thuộc đó; không được dựng
                    </>
                  }
                >
                the dependency is in <a href={href('ch16', '16.6')}>16.6</a> (a sheared tile reads the tile on its left
                and the tile above and to the right); the H.264 question above has the same one; not built
                </Translated>
              </td>
            </tr>
            <tr>
              <td>6</td>
              <td><Translated vi="Một lệnh DPX cho max4">A DPX instruction for max4</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      câu hỏi về DPX ở <a href={href('ch16', '16.6')}>16.6</a>; README của bài thực hành nói cách thử nó trên một GPU NVIDIA
                    </>
                  }
                >
                the DPX question in <a href={href('ch16', '16.6')}>16.6</a>; the lab's README says how to try it on an
                NVIDIA GPU
                </Translated>
              </td>
            </tr>
            <tr>
              <td>p. 384</td>
              <td><Translated vi="Một lần khởi chạy mỗi đường chéo phụ, một luồng mỗi ô">One launch per anti-diagonal, one thread per cell</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      được cung cấp trong <a href={href('ch16', 'labs')}>lab 16.1</a> làm phương án cơ sở, <code>labs/ch16/shared/</code>
                    </>
                  }
                >
                provided in <a href={href('ch16', 'labs')}>lab 16.1</a> as the baseline, <code>labs/ch16/shared/</code>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>p. 386</td>
              <td><Translated vi="Khởi chạy đúng số khối mà một mặt sóng cần">Launch exactly the blocks a front needs</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      câu hỏi về số khối chính xác ở <a href={href('ch16', '16.6')}>16.6</a>, và cả hai chương trình khởi chạy của{' '}
                      <a href={href('ch16', 'labs')}>lab 16.1</a>
                    </>
                  }
                >
                the exact-blocks question in <a href={href('ch16', '16.6')}>16.6</a>, and both launchers of{' '}
                <a href={href('ch16', 'labs')}>lab 16.1</a>
                </Translated>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2><Translated vi="Những chỗ PDF có vẻ sai">Where the PDF looks wrong</Translated></h2>
      <Translated
        vi={
          <p>
            Bản sách mà khóa học này dựa vào có một số chỗ có vẻ sơ suất ở chương 16. Chúng được liệt kê để bạn không bị bối rối
            khi đọc sách song song. Trước hết, về câu chữ và đánh số:
          </p>
        }
      >
      <p>
        The copy of the book this course was built from has a number of apparent slips in chapter 16. They are listed so
        that you are not thrown by them when you read the book alongside. First, wording and numbering:
      </p>
      </Translated>
      <ul data-list="errata">
        <li>
          <Translated
            vi={
              <>
                Fig. 16.5 ghi cạnh từ b đến f là 6 trong đồ thị và 9 trong bảng. Khóa học dùng bảng như được in (
                <a href={href('ch16', '16.3')}>16.3</a>).
              </>
            }
          >
          Fig. 16.5 labels the edge from b to f 6 in the graph and 9 in the table. The course uses the table as printed (
          <a href={href('ch16', '16.3')}>16.3</a>).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Eq. 16.2 trừ điểm phạt xóa (deletion) khỏi ô phía trên và điểm phạt chèn (insertion) khỏi ô bên trái; Fig. 16.6, phần
                chữ ở p. 383 và kernel gán tên theo chiều ngược lại. Với các điểm phạt bằng nhau thì không gì thay đổi.
              </>
            }
          >
          Eq. 16.2 subtracts the deletion penalty from the cell above and the insertion penalty from the cell on the left;
          Fig. 16.6, the text on p. 383 and the kernel attach the names the other way round. With equal penalties nothing
          changes.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 382 đánh chỉ số H theo một tiền tố của A cho i và của B cho j; p. 383 và Fig. 16.6 dùng i cho B và j cho A, và câu
                "the B_j was incorrectly inserted" ở p. 383 nghĩa là B_i.
              </>
            }
          >
          p. 382 indexes H by a prefix of A for i and of B for j; p. 383 and Fig. 16.6 use i for B and j for A, and p. 383's
          "the B_j was incorrectly inserted" means B_i.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Fig. 16.14(a) ghi các mặt sóng của ô vuông là d = 0, 1, 2, 4, 5; phần chữ đánh số chúng từ 0 đến 4, nên hai số cuối là
                3 và 4.
              </>
            }
          >
          Fig. 16.14(a) labels the square tiles' fronts d = 0, 1, 2, 4, 5; the text numbers them 0 to 4, so the last two are
          3 and 4.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                p. 391 tính phép trượt cho "cell (1, 0) in tile (0, 1)"; phép tính thực ra là cho ô con (2, 0). Ở phần sau trang đó,
                "only tile (1, 1)" ở d = 1 nghĩa là ô (0, 1), như công thức của kernel và các câu tiếp theo cho thấy.
              </>
            }
          >
          p. 391 works the shear for "cell (1, 0) in tile (0, 1)"; the arithmetic is for cell (2, 0). Later on the page,
          "only tile (1, 1)" at d = 1 means tile (0, 1), as the kernel's formula and the next sentences give.
          </Translated>
        </li>
        <li><Translated vi='p. 398 nói max4 nằm "in the kernel of Fig. 16.10"; thực ra nó được gọi trong Figs. 16.9 và 16.16 và được định nghĩa trong Fig. 16.11.'>p. 398 says max4 is "in the kernel of Fig. 16.10"; it is called in Figs. 16.9 and 16.16 and defined in Fig. 16.11.</Translated></li>
        <li><Translated vi="p. 377 nói việc tính F_i cần giữ F_i và F_i−1; Eq. 16.1 cần F_i−1 và F_i−2.">p. 377 says computing F_i needs F_i and F_i−1 kept; Eq. 16.1 needs F_i−1 and F_i−2.</Translated></li>
        <li>
          <Translated
            vi={
              <>
                "up to 1.5× more wavefronts" ở p. 396 nghĩa là nhiều nhất gấp 1.5 lần: (3T − 1) / (2T − 1) tiến tới 1.5.
              </>
            }
          >
          p. 396's "up to 1.5× more wavefronts" means up to 1.5 times as many: (3T − 1) / (2T − 1) approaches 1.5.
          </Translated>
        </li>
      </ul>
      <Translated vi={<p>Và trong các mã in, ở chỗ ảnh hưởng đến mã:</p>}><p>And in the listings, where it affects code:</p></Translated>
      <ul data-list="code">
        <li>
          <Translated
            vi={
              <>
                Fig. 16.4 so sánh một <code>int</code> với <code>INFINITY</code>, một <code>float</code> mà không <code>int</code>{' '}
                nào bằng, nên các lối thoát sớm không bao giờ kích hoạt, và với <code>INT_MAX</code> làm "không có cạnh", tổng bị tràn
                số. Khóa học dùng <code>INT_MAX / 2</code> (<a href={href('ch16', '16.3')}>16.3</a>).
              </>
            }
          >
          Fig. 16.4 compares an <code>int</code> with <code>INFINITY</code>, a <code>float</code> that no <code>int</code>{' '}
          equals, so the early exits never fire, and with <code>INT_MAX</code> as "no edge" the sum overflows. The course
          uses <code>INT_MAX / 2</code> (<a href={href('ch16', '16.3')}>16.3</a>).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Fig. 16.4 cho các luồng ngoài phạm vi trả về trước <code>__syncthreads()</code> của nó. CUDA ghi nhận rằng các luồng
                đã thoát không giữ rào chắn; HIP không nói gì, và Metal chỉ nói vậy cho Apple silicon. Khóa học giữ mọi luồng đến rào
                chắn.
              </>
            }
          >
          Fig. 16.4 returns out-of-range threads before its <code>__syncthreads()</code>. CUDA documents that exited threads
          do not hold a barrier; HIP does not say, and Metal says so for Apple silicon only. The course keeps every thread
          to the barrier.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Figs. 16.9 và 16.16 tính <code>tile_col</code> và <code>q_tile</code> từ các giá trị unsigned và lưu một kết quả có
                thể âm vào một <code>int</code>, một phép chuyển đổi mà C++17 để cho trình biên dịch quyết định. Bài thực hành tính
                bằng <code>int</code>.
              </>
            }
          >
          Figs. 16.9 and 16.16 compute <code>tile_col</code> and <code>q_tile</code> from unsigned values and store a
          possibly negative result in an <code>int</code>, a conversion C++17 leaves to the compiler. The lab computes in{' '}
          <code>int</code>.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Phép lưu của ô đã trượt ở Fig. 16.18 chỉ kiểm tra <code>q &lt; L</code>, và chạy được vì <code>L</code> là unsigned,
                nên một q âm bị quay vòng. Bài thực hành viết <code>q &gt;= 1 &amp;&amp; q &lt; L</code>.
              </>
            }
          >
          Fig. 16.18's sheared store tests only <code>q &lt; L</code>, and works because <code>L</code> is unsigned, so a
          negative q wraps round. The lab writes <code>q &gt;= 1 &amp;&amp; q &lt; L</code>.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Macro <code>pad()</code> ở p. 395 thêm một khoảng trống sau mỗi số int bằng số bank, không phải một khoảng trống mỗi
                hàng của ô: hai cách chỉ giống nhau khi W bằng số bank, và có xung đột 2 chiều ở W = 64 trên 32 bank. Các mã in không
                bao giờ áp dụng nó. Bài thực hành đệm một khoảng trống mỗi hàng qua một bước nhảy (
                <a href={href('ch16', '16.6')}>16.6</a>).
              </>
            }
          >
          The <code>pad()</code> macro of p. 395 adds one gap per bank-count ints, not one per tile row: the same only when
          W equals the number of banks, and a 2-way conflict at W = 64 on 32 banks. The printed listings never apply it. The
          lab pads one gap per row through a stride (<a href={href('ch16', '16.6')}>16.6</a>).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Các mã in không bao giờ định nghĩa <code>INFINITY</code>, <code>T</code>, <code>MATCH</code>, <code>MISMATCH</code>,{' '}
                <code>DELETION</code>, <code>INSERTION</code>, hay <code>threads</code>, <code>L_seq</code>,{' '}
                <code>sw</code>, <code>rea</code> và <code>ref</code> của các đoạn mã host. Các điểm phạt được cộng vào, nên chúng phải
                âm, điều sách ngụ ý nhưng không nói rõ.
              </>
            }
          >
          The listings never define <code>INFINITY</code>, <code>T</code>, <code>MATCH</code>, <code>MISMATCH</code>,{' '}
          <code>DELETION</code>, <code>INSERTION</code>, or the host snippets' <code>threads</code>, <code>L_seq</code>,{' '}
          <code>sw</code>, <code>rea</code> and <code>ref</code>. The penalties are added, so they must be negative, which
          the book implies but does not state.
          </Translated>
        </li>
      </ul>
      <Aside>
        <Translated
          vi={
            <p>
              Đây là cách đọc của chúng tôi về một bản PDF (notes/ch16.md, "Unclear in the PDF"), với các điểm về mã được kiểm tra
              ngày 2026-11-20 (ch16-facts A3, E1; ch04-facts A6, B3, C3). Fig. 16.2(b) và hình của khung bên lề cũng khó đọc ở cỡ
              in; stencil tài chính của trình khám phá là cách đọc của chúng tôi. Nếu bản của bạn khác, hãy tin bản của bạn.
            </p>
          }
        >
        <p>
          These are our readings of one PDF (notes/ch16.md, "Unclear in the PDF"), with the code points checked on
          2026-11-20 (ch16-facts A3, E1; ch04-facts A6, B3, C3). Fig. 16.2(b) and the sidebar's figure are also hard to
          read at print size; the explorer's financial stencil is our reading. If your copy differs, trust your copy.
        </p>
        </Translated>
      </Aside>

      <h2><Translated vi="Những gì chương này bỏ ra">What this chapter left out</Translated></h2>
      <Translated
        vi={
          <p>
            Chương này được rút gọn trong khóa học, nên đây là những gì không được đề cập và nơi tìm chúng. Từ §16.5: bối cảnh
            sinh học hệ gen (p. 382), ý nghĩa của điểm số, cách chọn sơ đồ chấm điểm và chèn, xóa là gì (pp. 382–383, Eq. 16.2,
            Fig. 16.6), trong đó bài thực hành chỉ giữ các giá trị ví dụ làm hằng số, và bước truy vết ngược để khôi phục một phép
            căn chỉnh (p. 384). Từ §16.1 đến §16.4: các mã Fibonacci từng dòng và vết lời gọi của chúng (Fig. 16.1, pp. 374–376),
            các lĩnh vực ứng dụng đứng sau các họ bài toán (p. 374) và sau bốn mẫu của Fig. 16.2 (p. 377), cả hai được nhắc đến
            trong một câu ở đây, và cách đặt vấn đề theo đồ thị cùng mã giả của Floyd-Warshall ngoài mẫu phụ thuộc của nó
            (Fig. 16.3, pp. 377–379).
          </p>
        }
      >
      <p>
        The chapter is condensed in this course, so here is what is not covered and where to find it. From §16.5: the
        genomics background (p. 382), what the score means, how a scoring scheme is chosen and what insertions and
        deletions are (pp. 382–383, Eq. 16.2, Fig. 16.6), of which the lab keeps only the example values as constants, and
        the traceback that recovers an alignment (p. 384). From §16.1 to §16.4: the Fibonacci listings line by line and
        their call trace (Fig. 16.1, pp. 374–376), the application domains behind the families of problems (p. 374) and
        behind Fig. 16.2's four patterns (p. 377), both named in a sentence here, and Floyd-Warshall's graph framing and
        pseudo-code beyond its dependency pattern (Fig. 16.3, pp. 377–379).
      </p>
      </Translated>
      <Translated
        vi={
          <p>
            Từ §16.7: lập luận về cache cho ô đã trượt (Fig. 16.13, pp. 389–390), được nêu trong một đoạn ở trang 16.6 và không
            được mô hình hóa, và khung bên lề về ô đã trượt cho stencil theo thời gian và không gian (p. 397), được nhắc ở đó và
            được dùng cho một câu hỏi ở trên. Từ §16.8: kernel cooperative groups, các kernel dùng cờ và các bài toán nhỏ được giữ
            trong thanh ghi với shuffle (pp. 397–398), tất cả được bàn và mô hình hóa ở trang 16.3 nhưng không được dựng, và DPX
            chỉ dưới dạng một câu hỏi. Bài tập 1 và 2 được phân tích, không được dựng. Trong các tài liệu tham khảo (p. 400), chỉ
            nguồn gốc của hypertiling được trích. Không phần nào sau này trong khóa học phụ thuộc vào những gì bị bỏ ra.
          </p>
        }
      >
      <p>
        From §16.7: the cache argument for sheared tiles (Fig. 16.13, pp. 389–390), stated in a paragraph on page 16.6 and
        not modelled, and the sidebar on sheared tiles for stencils in time and space (p. 397), named there and used by one
        question above. From §16.8: the cooperative-groups kernel, the flag kernels and small problems held in registers
        with shuffles (pp. 397–398), all discussed and modelled on page 16.3 but not built, and DPX as a question only.
        Exercises 1 and 2 are analysed, not built. Of the references (p. 400), only the origin of hypertiling is cited.
        Nothing later in the course depends on what was left out.
      </p>
      </Translated>

      <h2><Translated vi="Dự án đến giờ">The project so far</Translated></h2>
      <Translated
        vi={
          <p>
            Chương này không thêm mốc dự án nào cho dự án xuyên suốt: không gì trên đường chính của một bước transformer
            decoder-only, các phép nhân ma trận, attention, softmax hay lấy mẫu của nó, lấp một bảng quy hoạch động. Các ý tưởng
            của nó áp dụng cho các lớp hồi quy và linear-attention, thứ mà mô hình của dự án không có, và cho việc đọc cách
            llama.cpp, PyTorch và mlx-lm chạy chúng. Lab 16.1 là tùy chọn.{' '}
            <span data-lab161={passing.join(',') || 'none'}>
              {backends.length === 0 && !reports['emu']
                ? 'Chưa có kết quả bài thực hành nào được ghi trên máy này.'
                : passing.length
                  ? `Bản của bạn đạt trên ${passing.map(where).join(' và ')} trên máy này.`
                  : 'Bản của bạn chưa đạt trên máy này.'}
            </span>{' '}
            Xem <a href={href('ch16', 'labs')}>bài thực hành</a>. Mốc tiếp theo, M6 attention, đi cùng{' '}
            <a href={href('ch20', 'start')}>chương 20</a>.
          </p>
        }
      >
      <p>
        This chapter adds no milestone to the through-line project: nothing on the main path of a decoder-only
        transformer's step, its matrix products, attention, softmax or sampling, fills a dynamic-programming table. Its
        ideas apply to recurrent and linear-attention layers, which the project's model does not have, and to reading how
        llama.cpp, PyTorch and mlx-lm run them. Lab 16.1 is optional.{' '}
        <span data-lab161={passing.join(',') || 'none'}>
          {backends.length === 0 && !reports['emu']
            ? 'No lab results are recorded on this machine yet.'
            : passing.length
              ? `Your own version passes on ${passing.map(where).join(' and ')} on this machine.`
              : 'Your own version has not passed on this machine yet.'}
        </span>{' '}
        See the <a href={href('ch16', 'labs')}>lab</a>. The next milestone, M6 attention, comes with{' '}
        <a href={href('ch20', 'start')}>chapter 20</a>.
      </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chương 17 chuyển từ phụ thuộc sang dữ liệu bất quy tắc. Một phép nhân ma trận thưa với vector hoàn toàn không có
            wavefront, nhưng định dạng lưu ma trận quyết định các lần nạp có được hợp nhất không, các luồng có phân kỳ không và các
            khối có nhận lượng việc bằng nhau không, và sự đánh đổi giữa đệm và công việc lãng phí quay lại như cái giá của một
            cách bố trí đều đặn. Bắt đầu ở <a href={href('ch17', 'start')}>chương 17</a>.
          </p>
        }
      >
      <p>
        Chapter 17 turns from dependencies to irregular data. A sparse matrix–vector product has no wavefronts at all, but
        the format the matrix is stored in decides whether loads coalesce, whether threads diverge and whether blocks get
        equal work, and the trade between padding and wasted work comes back as the price of a regular layout. Start at{' '}
        <a href={href('ch17', 'start')}>chapter 17</a>.
      </p>
      </Translated>
    </>
  );
}
