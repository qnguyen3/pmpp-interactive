import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { EX1, outNeighbours, runBFS, toCSR, type Kernel, type LevelRun } from '../../sim/bfs';
import { gridSummary, strideAssignment, treeSummary } from '../../sim/gridbfs';

// Chapter 18's summary page: what to carry forward, a mixed check on situations the pages did
// not show, where the book's three exercises are worked (with Exercise 1's answers computed
// from src/sim/bfs.ts), the PDF's apparent slips, what the condensed chapter left out, and the
// project status. No kernel is shown here, so there is no snippet file; Wrap.test.ts pins
// every number below against the models.

// ---------------------------------------------------------------- Exercise 1, from the model
const EX1_CSR = toCSR(EX1);
/** Exercise 1a: each vertex's out-neighbours, which are the 1s of its row in the adjacency matrix. */
export const EX1_ROWS = Array.from({ length: EX1.n }, (_, v) => outNeighbours(EX1_CSR, v));
export { EX1_CSR };

const run = (k: Kernel) => runBFS(EX1, { variant: k, root: 0, blockSize: 4, warpWidth: 4 });
/** Exercise 1c: one row per kernel and count, one value per call. */
export interface Ex1cRow {
  kernel: Kernel;
  what: string;
  values: number[];
}
const per = (k: Kernel, f: (c: LevelRun) => number) => run(k).calls.map(f);
export const EX1C: Ex1cRow[] = [
  { kernel: 'push', get what() { return tr('threads launched (one per vertex)', 'số luồng khởi chạy (mỗi đỉnh một luồng)'); }, values: per('push', (c) => c.items) },
  { kernel: 'push', get what() { return tr('threads that loop over their out-edges', 'số luồng lặp qua các cạnh ra của mình'); }, values: per('push', (c) => c.pastCheck) },
  { kernel: 'pull', get what() { return tr('threads launched (one per vertex)', 'số luồng khởi chạy (mỗi đỉnh một luồng)'); }, values: per('pull', (c) => c.items) },
  { kernel: 'pull', get what() { return tr('threads that loop (vertex still unvisited)', 'số luồng lặp (đỉnh vẫn chưa thăm)'); }, values: per('pull', (c) => c.pastCheck) },
  { kernel: 'pull', get what() { return tr('threads that label their vertex', 'số luồng gán nhãn cho đỉnh của mình'); }, values: per('pull', (c) => c.labelsMax) },
  { kernel: 'edge', get what() { return tr('threads launched (one per edge)', 'số luồng khởi chạy (mỗi cạnh một luồng)'); }, values: per('edge', (c) => c.items) },
  { kernel: 'edge', get what() { return tr('threads whose source is in the last level', 'số luồng có đỉnh nguồn thuộc mức vừa rồi'); }, values: per('edge', (c) => c.pastCheck) },
  { kernel: 'edge', get what() { return tr('threads that may label (destination unvisited)', 'số luồng có thể gán nhãn (đỉnh đích chưa thăm)'); }, values: per('edge', (c) => c.labelsMax) },
  { kernel: 'frontier', get what() { return tr('threads launched, all of which loop', 'số luồng khởi chạy, tất cả đều lặp'); }, values: per('frontier', (c) => c.items) },
];
const KERNEL_NAME: Record<Kernel, string> = {
  get push() { return tr('Vertex push', 'Push theo đỉnh'); },
  get pull() { return tr('Vertex pull', 'Pull theo đỉnh'); },
  get edge() { return tr('Edge-centric', 'Lấy cạnh làm trung tâm'); },
  get frontier() { return tr('Frontier push', 'Push dùng frontier'); },
};

// ---------------------------------------------------------------- numbers the questions use
/** A 16-ary tree five levels deep, against the 1000 x 1000 grid of page 18.6. */
export const TREE = { b: 16, depth: 5 } as const;
export const TREE_SUM = treeSummary(TREE.b, TREE.depth);
export const GRID_SUM = gridSummary(1000);
/** The emulator's cooperative grid in lab 18.1: 8 blocks of 128 threads. */
export const EMU_GRID = { blocks: 8, threads: 128 } as const;
export const STRIDE_FRONTIER = 2500;
const STRIDE = strideAssignment(STRIDE_FRONTIER, EMU_GRID.blocks * EMU_GRID.threads);
export const STRIDE_MAX = Math.max(...STRIDE.map((s) => s.length));
export const STRIDE_THREE = STRIDE.filter((s) => s.length === STRIDE_MAX).length;

const n = (v: number) => v.toLocaleString('en-US');
const list = (xs: readonly number[]) => '[' + xs.join(', ') + ']';

export const questionsWrap: Record<string, Question> = {
  'ch18/wrap/q-tree-launches': {
    kind: 'numeric',
    id: 'ch18/wrap/q-tree-launches',
    prompt: (
      <Translated
        vi={
          <>
            Một cây trong đó mọi đỉnh có {TREE.b} con, sâu {TREE.depth} mức bên dưới gốc: {n(TREE_SUM.vertices)}{' '}
            đỉnh. Bạn tìm kiếm nó từ gốc với vòng lặp host theo từng mức của Hình 18.15. Phép tìm kiếm thực hiện bao nhiêu
            lần khởi chạy kernel?
          </>
        }
      >
        <>
          A tree in which every vertex has {TREE.b} children, {TREE.depth} levels deep below its root: {n(TREE_SUM.vertices)}{' '}
          vertices. You search it from the root with the per-level host loop of Fig. 18.15. How many kernel launches does
          the search make?
        </>
      </Translated>
    ),
    answer: TREE_SUM.launches,
    unit: 'launches',
    explain: (
      <Translated
        vi={
          <p>
            {TREE_SUM.levels} mức sau gốc, mỗi mức một lần khởi chạy, cộng lần khởi chạy cuối, lần thấy frontier trống:{' '}
            {TREE_SUM.launches}. Frontier tăng {TREE.b} lần sau mỗi mức, tới {n(TREE_SUM.maxFrontier)} ở đáy. Lưới 1000 × 1000
            của trang 18.6 có số đỉnh xấp xỉ như vậy, {n(GRID_SUM.vertices)}, và cần {n(GRID_SUM.launches)}{' '}
            lần khởi chạy, không lần nào có quá {n(GRID_SUM.maxFrontier)} đỉnh công việc. Số lần khởi chạy đi theo độ sâu
            của đồ thị, không theo kích thước của nó, nên chi phí cố định của một mức là không đáng kể với cây nhưng có thể
            chiếm phần lớn thời gian với lưới: đó là sự tương phản giữa mạng xã hội và mạng đường bộ của sách.
          </p>
        }
      >
        <p>
          {TREE_SUM.levels} levels after the root, one launch each, plus the last launch, which finds the frontier empty:{' '}
          {TREE_SUM.launches}. The frontier grows {TREE.b}-fold per level, to {n(TREE_SUM.maxFrontier)} at the bottom. Page
          18.6's 1000 × 1000 grid has about as many vertices, {n(GRID_SUM.vertices)}, and takes {n(GRID_SUM.launches)}{' '}
          launches, none with more than {n(GRID_SUM.maxFrontier)} vertices of work. The number of launches follows the
          depth of the graph, not its size, so the fixed cost of a level is noise for the tree and may be most of the
          time for the grid: the book's contrast between social networks and road networks.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.5, p. 440; §18.7, p. 445; src/sim/gridbfs.ts; the tree is ours',
    vi: {
      unit: 'lần khởi chạy',
      source: 'ch. 18 §18.5, p. 440; §18.7, p. 445; src/sim/gridbfs.ts; cây là của chúng tôi',
    },
  },
  'ch18/wrap/q-exchange': {
    kind: 'choice',
    id: 'ch18/wrap/q-exchange',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel repetition-penalty liệt kê các token phân biệt của một chuỗi: mỗi luồng chạy{' '}
            <code>old = atomicExch(&amp;seen[tok], 1)</code> và nối <code>tok</code> khi <code>old</code> bằng 0. Cách đó
            liệt kê mỗi token một lần. Vì sao Hình 18.14 không thể làm tương tự, exchange giá trị <code>currLevel</code> vào
            và nối khi giá trị cũ là <code>UNVISITED</code>?
          </>
        }
      >
        <>
          A repetition-penalty kernel lists the distinct tokens of a sequence: each thread runs{' '}
          <code>old = atomicExch(&amp;seen[tok], 1)</code> and appends <code>tok</code> when <code>old</code> was 0. That
          lists each token once. Why can Fig. 18.14 not do the same, exchanging in <code>currLevel</code> and appending
          when the old value was <code>UNVISITED</code>?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một exchange luôn ghi: một đỉnh đã được gán nhãn ở mức trước sẽ nhận mức hiện tại thay vào. Còn giá trị 1 lặp lại của một cờ thì vô hại.">
            An exchange always writes: a vertex labelled at an earlier level gets the current level instead. A flag's repeated 1 is harmless.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Exchange không phải atomic trên bộ nhớ device: chỉ compare-and-swap mới là phép đọc-sửa-ghi thực sự ở đó.">
            An exchange is not atomic on device memory: only compare-and-swap is a true read-modify-write there.
          </Translated>
        ),
        why: (
          <Translated vi="atomicExch là một phép đọc-sửa-ghi atomic như các phép khác, trên CUDA và trên HIP, và cuda::atomic_ref cũng có exchange().">
            atomicExch is an atomic read-modify-write like the others, on CUDA and on HIP, and cuda::atomic_ref has exchange() too.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Hai luồng exchange cùng lúc có thể cùng nhận lại UNVISITED, vì mỗi luồng đọc giá trị cũ trước khi luồng kia ghi.">
            Two threads exchanging at the same moment could both get UNVISITED back, because each reads the old value before the other one writes.
          </Translated>
        ),
        why: (
          <Translated vi="Các exchange trên một vị trí diễn ra lần lượt, và mỗi lần trả về giá trị mà lần trước để lại. Chỉ lần đầu thấy UNVISITED, nên frontier sẽ chứa mỗi đỉnh một lần.">
            The exchanges on one location happen one after another, and each returns the value the one before it left. Only the first sees UNVISITED, so the frontier would hold each vertex once.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó sẽ chạy được, với cùng frontier và cùng các mức; compare-and-swap được chọn chỉ vì nó nhanh hơn.">
            It would work, with the same frontier and the same levels; compare-and-swap is chosen only because it is faster.
          </Translated>
        ),
        why: (
          <Translated vi="Tốc độ không phải vấn đề. Một cạnh ngược, như 7 → 0 trong đồ thị của sách, sẽ gán lại cho đỉnh 0 mức hiện tại.">
            Speed is not the problem. A back edge, such as 7 → 0 in the book's graph, would relabel vertex 0 with the current level.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Compare-and-swap chỉ ghi khi giá trị vẫn là giá trị bạn mong đợi; một exchange ghi bất kể ở đó có gì. Với một cờ,
            mọi luồng ghi đều ghi cùng một giá trị, nên lần ghi vô điều kiện là vô hại và chỉ riêng giá trị cũ quyết định ai được
            nối. Với BFS, giá trị được ghi, tức mức, chỉ đúng cho lần thăm đầu tiên: một đỉnh được tìm thấy lại qua một cạnh
            ngược đã giữ một mức nhỏ hơn, và exchange sẽ thay thế nó. Đúng một lần cần điều kiện và lần ghi trong cùng một bước,
            và đó chính là Hình 18.14.
          </p>
        }
      >
        <p>
          Compare-and-swap writes only when the value is still the one you expected; an exchange writes whatever is
          there. For a flag every writer stores the same value, so the unconditional write is harmless and the old value
          alone decides who appends. For BFS the written value, the level, is only right for the first visit: a vertex
          found again through a back edge already holds a smaller level, and the exchange would replace it. Exactly once
          needs the condition and the write in one step, which is what Fig. 18.14 is.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.5, pp. 440–442, Fig. 18.14; infra-facts A1, A2 (atomicExch, exchange); the penalty kernel is ours',
    vi: {
      source: 'ch. 18 §18.5, pp. 440–442, Hình 18.14; infra-facts A1, A2 (atomicExch, exchange); kernel penalty là của chúng tôi',
    },
  },
  'ch18/wrap/q-sampler-order': {
    kind: 'choice',
    id: 'ch18/wrap/q-sampler-order',
    prompt: (
      <Translated
        vi={
          <>
            Một bộ lấy mẫu top-p gom các token trên một ngưỡng bằng các hàng đợi riêng của Hình 18.15, rồi tính tổng dồn xác
            suất của chúng theo thứ tự danh sách và chọn phần tử đầu tiên có tổng dồn vượt <code>u</code>, một số rút từ một
            seed cố định. Cùng logits, cùng seed: hai lần chạy chọn hai token khác nhau. Vì sao, và cách nào khắc phục?
          </>
        }
      >
        <>
          A top-p sampler collects the tokens above a threshold with Fig. 18.15's private queues, then takes a running sum
          of their probabilities in list order and picks the first entry whose running sum passes <code>u</code>, a number
          drawn from a fixed seed. Same logits, same seed: two runs pick different tokens. Why, and what fixes it?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Thứ tự danh sách phụ thuộc vào lịch chạy, nên cùng một u rơi vào token khác. Hãy sắp xếp các token sống sót trước, hoặc dựng danh sách bằng một phép lọc ổn định.">
            The list order depends on the schedule, so the same u lands on another token. Sort the survivors first, or build the list with a stable filter.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Bộ đếm atomic thỉnh thoảng làm rơi một token, nên nội dung các danh sách khác nhau.">
            The atomic counter drops a token now and then, so the lists differ in content.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi fetch_add trao một ô khác nhau, và mỗi token sống sót nhận đúng một ô, riêng hoặc chung. Hai danh sách chứa cùng các token theo thứ tự khác nhau.">
            Each fetch_add hands out a different slot, and every survivor takes exactly one, private or public. The two lists hold the same tokens in different orders.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nhánh tràn đưa một số token vào danh sách chung hai lần.">
            The overflow path puts some tokens in the public list twice.
          </Translated>
        ),
        why: (
          <Translated vi="Một token sống sót đi một đường: một ô riêng dưới dung lượng, hoặc bộ đếm chung. Dù cách nào nó cũng được commit một lần.">
            A survivor takes one path: a private slot below the capacity, or the public counter. It is committed once either way.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Đặt cả hai bộ đếm là memory_order_seq_cst: khi đó có một thứ tự toàn phần trên mọi atomic của chúng, giống nhau ở mọi lần chạy, và do đó cùng thứ tự danh sách.">
            Make both counters memory_order_seq_cst: there is then one total order over all their atomics, the same in every run, and so the same list order.
          </Translated>
        ),
        why: (
          <Translated vi="Một memory order ràng buộc những gì các luồng khác thấy quanh mỗi atomic. Nó không quyết định luồng nào tới bộ đếm trước, và đó mới là thứ đặt ra thứ tự của danh sách.">
            A memory order constrains what other threads see around each atomic. It does not decide which thread reaches the counter first, and that is what sets the order of the list.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hình 18.16 cho thấy frontier chung được nhóm theo khối nhưng không theo thứ tự khối, và trong một khối các ô riêng
            theo thứ tự các atomic đến. Với BFS thứ tự đó không quan trọng: thứ tự nào của một frontier cũng cho cùng các mức.
            Với một lần lấy mẫu theo CDF ngược thì có, vì token được chọn là một vị trí trong danh sách. Hoặc làm cho thứ tự là
            một hàm của dữ liệu (sắp xếp các token sống sót), hoặc không bao giờ để nó phụ thuộc vào lịch chạy (phép lọc ổn định
            của chương 12: một phép scan loại trừ trên các cờ giữ lại cho mỗi token sống sót vị trí của nó). Danh sách expert của
            llama.cpp, ở trang 18.6, ra theo thứ tự token vì chúng xếp hạng bằng ballot thay vì atomic.
          </p>
        }
      >
        <p>
          Fig. 18.16 shows the public frontier grouped by block but not in block order, and inside a block the private
          slots go in the order the atomics land. For BFS that order does not matter: any order of a frontier gives the
          same levels. For an inverse-CDF sample it does, because the token chosen is a position in the list. Either make
          the order a function of the data (sort the survivors) or never let it depend on the schedule (chapter 12's
          stable filter: an exclusive scan of keep flags gives every survivor its position). llama.cpp's expert lists, on
          page 18.6, come out in token order because they rank with a ballot rather than an atomic.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.6, p. 444, Fig. 18.16; ch. 12 §12.2, §12.5, pp. 290–291, 297–298; the sampler is ours',
    vi: {
      source: 'ch. 18 §18.6, p. 444, Hình 18.16; ch. 12 §12.2, §12.5, pp. 290–291, 297–298; bộ lấy mẫu là của chúng tôi',
    },
  },
  'ch18/wrap/q-stride-barrier': {
    kind: 'choice',
    id: 'ch18/wrap/q-stride-barrier',
    prompt: (
      <Translated
        vi={
          <>
            Một đồng nghiệp sửa Hình 18.17 để commit sau mỗi phần tử frontier: rào chắn khối, phần cấp phát và commit được
            chuyển vào trong vòng lặp có bước nhảy{' '}
            <code>for (i = rank; i &lt; numPrevFrontier; i += numThreads)</code>. Mọi bài kiểm tra có frontier là bội số của
            số luồng của lưới đều đạt. Điều gì sai trên một frontier có kích thước khiến một số luồng của một khối phải đi thêm
            một lượt còn các luồng còn lại thì không?
          </>
        }
      >
        <>
          A colleague changes Fig. 18.17 to commit after every frontier element: the block barrier, the allocation and the
          commit move inside the stride loop{' '}
          <code>for (i = rank; i &lt; numPrevFrontier; i += numThreads)</code>. Every test whose frontiers are a multiple
          of the grid's thread count passes. What goes wrong on a frontier whose size gives some threads of one block an
          extra trip and not the rest?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Các luồng của khối đó đi số lượt khác nhau, nên một số luồng tới __syncthreads() nhiều lần hơn các luồng khác: không xác định, trên thực tế là treo hoặc mất phần tử.">
            Threads of that block make different numbers of trips, so some reach __syncthreads() more often than others: undefined, in practice a hang or lost entries.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không có gì: __syncthreads() chỉ chờ các luồng còn ở trong vòng lặp, và các luồng khác đi tiếp tới rào chắn lưới.">
            Nothing: __syncthreads() waits only for the threads still inside the loop, and the others go on to the grid barrier.
          </Translated>
        ),
        why: (
          <Translated vi="Nó chờ mọi luồng của khối. Một luồng đã rời vòng lặp không bao giờ tới rào chắn đó; nó đi tiếp tới rào chắn lưới, nơi các luồng kia không bao giờ tới.">
            It waits for every thread of the block. A thread that has left the loop never arrives at that barrier; it goes on to the grid barrier, where the others never arrive.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Ở đây chỉ rào chắn lưới mới có thể deadlock; một rào chắn khối chỉ khiến các luồng đến sớm chờ lâu hơn một chút.">
            Only the grid barrier can deadlock here; a block barrier just lets the threads that arrive early wait a little longer.
          </Translated>
        ),
        why: (
          <Translated vi="Một rào chắn khối mà một số luồng của khối không bao giờ tới là hành vi không xác định (ch. 4, Fig. 4.4), trên thực tế thường là treo.">
            A block barrier that some threads of the block never reach is undefined behaviour (ch. 4, Fig. 4.4), in practice often a hang.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các lần commit mất tính hợp nhất (coalescing) ở lượt ngắn nhưng vẫn đúng, vì mọi phần tử vẫn được sao chép một lần.">
            The commits lose coalescing on the short trip but stay correct, since every entry is still copied once.
          </Translated>
        ),
        why: (
          <Translated vi="Tính đúng đắn phải đặt lên trước: rào chắn không còn được cả khối tới nữa. Coalescing không liên quan cho đến khi sửa được điều đó.">
            Correctness goes first: the barrier is no longer reached by the whole block. Coalescing is beside the point until that is fixed.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Luồng r của lưới lấy các phần tử r, r + G, r + 2G và cứ thế của một frontier n phần tử. Khi n là bội số của G, mọi
            luồng đi n / G lượt; nếu không, n mod G rank thấp nhất đi thêm một lượt. Khi n mod G không phải bội số của kích
            thước khối, ranh giới rơi vào giữa một khối, khối đó khi ấy chứa cả hai loại, và bất kỳ rào chắn nào trong vòng lặp
            đều được các luồng của khối đó tới với số lần khác nhau. (Với G = 1024, khối 128 và n = 1152, khối 0 đi 2 lượt ở mọi
            luồng còn các khối khác đi 1: không khối nào bị chia, và kích thước này tình cờ cũng đạt.) Đó chính xác là quy tắc
            của chương 4: mọi luồng của khối thực thi cùng một <code>__syncthreads()</code>, hoặc không luồng nào thực thi. Hình
            18.17 giữ các rào chắn khối bên ngoài vòng lặp có bước nhảy, và các bài kiểm tra chỉ dùng kích thước thuận tiện
            không thể phân biệt hai cách.
          </p>
        }
      >
        <p>
          Thread r of the grid takes elements r, r + G, r + 2G and so on of a frontier of n. When n is a multiple of G
          every thread makes n / G trips; otherwise the lowest n mod G ranks make one more. When n mod G is not a multiple
          of the block size, the boundary falls inside a block, which then holds both kinds, and any barrier inside the
          loop is reached a different number of times by threads of that block. (With G = 1024, blocks of 128 and n =
          1152, block 0 makes 2 trips throughout and the others 1: no block is split, and this size happens to pass too.) That is exactly
          the rule of chapter 4: every thread of the block executes the same <code>__syncthreads()</code>, or none does.
          Fig. 18.17 keeps its block barriers outside the stride loop, and tests that only use convenient sizes cannot
          tell the two apart.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.3, p. 72, Fig. 4.4; ch. 18 §18.7, pp. 446–447, Fig. 18.17; the change is ours',
    vi: { source: 'ch. 4 §4.3, p. 72, Fig. 4.4; ch. 18 §18.7, pp. 446–447, Fig. 18.17; thay đổi này là của chúng tôi' },
  },
  'ch18/wrap/q-stride-count': {
    kind: 'numeric',
    id: 'ch18/wrap/q-stride-count',
    prompt: (
      <Translated
        vi={
          <>
            Trên trình giả lập của khóa học, lưới cooperative của lab 18.1 gồm {EMU_GRID.blocks} khối, mỗi khối{' '}
            {EMU_GRID.threads} luồng. Frontier của một mức chứa {n(STRIDE_FRONTIER)} đỉnh. Bao nhiêu luồng của lưới xử lý{' '}
            {STRIDE_MAX} đỉnh trong mức đó?
          </>
        }
      >
        <>
          On the course's emulator, lab 18.1's cooperative grid is {EMU_GRID.blocks} blocks of {EMU_GRID.threads}{' '}
          threads. One level's frontier holds {n(STRIDE_FRONTIER)} vertices. How many of the grid's threads handle{' '}
          {STRIDE_MAX} vertices in that level?
        </>
      </Translated>
    ),
    answer: STRIDE_THREE,
    unit: 'threads',
    explain: (
      <Translated
        vi={
          <p>
            Lưới có {n(EMU_GRID.blocks * EMU_GRID.threads)} luồng, và luồng r lấy các phần tử r, r + 1,024 và r + 2,048. Phần
            tử thứ ba tồn tại khi r + 2,048 &lt; {n(STRIDE_FRONTIER)}, tức với các rank 0 đến{' '}
            {STRIDE_THREE - 1}: {STRIDE_THREE} luồng lấy {STRIDE_MAX}, phần còn lại lấy {STRIDE_MAX - 1}. Lưới thường trú là cố
            định, nên một frontier lớn hơn nó tốn thêm lượt cho mỗi luồng, không phải thêm khối. Đây là số lượt không đều mà câu
            hỏi trước biến thành một lỗi.
          </p>
        }
      >
        <p>
          The grid has {n(EMU_GRID.blocks * EMU_GRID.threads)} threads, and thread r takes elements r, r + 1,024 and r +
          2,048. A third element exists while r + 2,048 &lt; {n(STRIDE_FRONTIER)}, so for ranks 0 to{' '}
          {STRIDE_THREE - 1}: {STRIDE_THREE} threads take {STRIDE_MAX}, the rest take {STRIDE_MAX - 1}. The resident grid
          is fixed, so a frontier larger than it costs more trips per thread, not more blocks. This is the uneven trip
          count that the previous question turns into a bug.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.7, pp. 446–447, Fig. 18.17 lines 21–22; lab 18.1 README (the emulator grid); src/sim/gridbfs.ts',
    vi: {
      unit: 'luồng',
      source: 'ch. 18 §18.7, pp. 446–447, Fig. 18.17 dòng 21–22; README của lab 18.1 (lưới của trình giả lập); src/sim/gridbfs.ts',
    },
  },
};

function Ex1cTable() {
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto', marginBottom: '1.2rem' }}>
      <table className="data" style={{ margin: 0 }} data-table="ex1c">
        <thead>
          <tr>
            <th><Translated vi="Kernel">Kernel</Translated></th>
            <th><Translated vi="Số đếm">Count</Translated></th>
            {EX1C[0].values.map((_, i) => (
              <th key={i} className="num"><Translated vi={<>Lần gọi {i + 1}</>}>Call {i + 1}</Translated></th>
            ))}
          </tr>
        </thead>
        <tbody>
          {EX1C.map((r, i) => (
            <tr key={i} data-kernel={r.kernel}>
              <td>{i === 0 || EX1C[i - 1].kernel !== r.kernel ? KERNEL_NAME[r.kernel] : ''}</td>
              <td>{r.what}</td>
              {r.values.map((v, j) => (
                <td key={j} className="num">{v}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports);
  const lab = 'ch18/lab1_frontier';
  const passing = [...backends, 'emu'].filter((b) => labStatus(reports[b], lab, 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch18/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}><Translated vi="Những gì cần mang theo">What to carry forward</Translated></h2>
      <Translated
        vi={
          <ul data-list="carry">
            <li>
              Chọn ai làm việc quét. Push lặp qua các cạnh ra của mức vừa rồi và ghi vào đỉnh khác; pull lặp qua các cạnh vào
              của các đỉnh chưa thăm và chỉ ghi phần tử của chính nó, nên không cần atomic và có thể dừng ở lần trúng đầu tiên.
              Cách nào rẻ hơn thay đổi theo từng mức.
            </li>
            <li>
              Đếm số luồng lãng phí, không chỉ mức song song. Kiểm tra mọi đỉnh hoặc cạnh ở mọi mức thì đơn giản và không cần
              đồng bộ hóa, nhưng trên một đồ thị sâu gần như toàn bộ là lãng phí. Một frontier chỉ khởi chạy luồng ở nơi có công
              việc.
            </li>
            <li>
              Dựng một danh sách công việc bằng một bộ đếm atomic: chiếm một ô bằng <code>fetch_add</code>, rồi lưu. Thứ tự là
              bất cứ thứ tự nào lịch chạy tạo ra. Ở nơi một bản trùng tốn công việc, hãy làm "kiểm tra và chiếm" thành một
              compare-and-swap, với giá trị expected mới mỗi lần, và lặp ở nơi chỉ có dạng yếu.
            </li>
            <li>
              Riêng tư hóa (privatization) một bộ đếm nóng: một hàng đợi mỗi khối trong bộ nhớ chia sẻ (shared memory), một
              nhánh tràn, một lần đặt chỗ toàn cục mỗi khối và một lần commit được hợp nhất. Mọi lần đặt lại một bộ đếm mà nơi
              khác cập nhật bằng atomic cũng phải là atomic.
            </li>
            <li>
              Nhiều bước ngắn phải trả một chi phí cố định cho mỗi lần khởi chạy. Một rào chắn lưới chỉ loại bỏ được các lần
              khởi chạy với một lưới thường trú hoàn toàn, nên hãy định kích thước nó từ truy vấn occupancy (mức chiếm dụng) và
              khởi chạy nó theo kiểu cooperative, hoặc dùng graph; Metal không có cả rào chắn lẫn truy vấn, và giữ mỗi bước một
              dispatch.
            </li>
            <li>
              Công việc không đều giữa các luồng giữ chân cả lưới. Hãy ánh xạ luồng vào cạnh thay vì đỉnh, hoặc chia nhóm công
              việc theo kích thước vào một luồng, một warp hay một khối.
            </li>
          </ul>
        }
      >
        <ul data-list="carry">
          <li>
            Choose who does the scanning. Push loops over the last level's out-edges and writes to others; pull loops over
            the unvisited vertices' in-edges and writes only its own entry, so it needs no atomic and can stop at the first
            hit. Which is cheaper changes from level to level.
          </li>
          <li>
            Count wasted threads, not only parallelism. Checking every vertex or edge at every level is simple and
            synchronisation-free, and on a deep graph almost all of it is wasted. A frontier launches threads only where
            the work is.
          </li>
          <li>
            Build a work list with an atomic counter: claim a slot with <code>fetch_add</code>, then store. The order is
            whatever the schedule made it. Where a duplicate costs work, make "check and claim" one compare-and-swap,
            with a fresh expected value each time, and loop where only a weak one exists.
          </li>
          <li>
            Privatise a hot counter: a queue per block in shared memory, an overflow path, one global reservation per
            block and a coalesced commit. Every reset of a counter that others update atomically must be atomic too.
          </li>
          <li>
            Many short steps pay a fixed cost per launch. A grid barrier removes the launches only for a grid that is
            entirely resident, so size it from the occupancy query and launch it cooperatively, or use graphs; Metal has
            neither barrier nor query, and keeps one dispatch per step.
          </li>
          <li>
            Uneven work per thread holds up the grid. Map threads to edges instead of vertices, or bucket the work by size
            onto a thread, a warp or a block.
          </li>
        </ul>
      </Translated>
      <Source section="18.9" pages="449" />

      <h2><Translated vi="Kiểm tra tổng hợp">A mixed check</Translated></h2>
      <p>
        <Translated vi="Các câu này dựa trên cả chương và trên những tình huống mà các trang chưa trình bày.">
          These draw on the whole chapter and on situations the pages did not show.
        </Translated>
        {asked
          ? tr(
              ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.`,
              ` Đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu hỏi, tính theo câu trả lời mới nhất của bạn cho mỗi câu.`,
            )
          : ''}
      </p>
      <Beyond>
        <Translated
          vi={
            <p>
              Cái cây, kernel penalty, bộ lấy mẫu, thay đổi của đồng nghiệp và phép đếm bước nhảy là của chúng tôi. Chúng dựa
              trên các trang của sách và trên các dữ kiện mà mỗi đáp án nêu tên.
            </p>
          }
        >
          <p>
            The tree, the penalty kernel, the sampler, the colleague's change and the stride count are ours. They rest on the
            book's pages and on the facts each answer names.
          </p>
        </Translated>
      </Beyond>
      <Ask q={q['ch18/wrap/q-tree-launches']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={q['ch18/wrap/q-exchange']} />
      <Ask q={q['ch18/wrap/q-sampler-order']} />
      <Ask q={q['ch18/wrap/q-stride-barrier']} label={tr('Bug hunt', 'Săn lỗi')} />
      <Ask q={q['ch18/wrap/q-stride-count']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Bài tập của sách">The book's exercises</Translated></h2>
      <Translated
        vi={
          <p>
            Chương kết thúc bằng ba bài tập (pp. 449–450). Sách không in đáp án, nên đáp án trong khóa học này là của chúng
            tôi, và đáp án Bài tập 1 dựa trên cách chúng tôi đọc các đầu mũi tên trong hình vẽ của nó.
          </p>
        }
      >
        <p>
          The chapter ends with three exercises (pp. 449–450). The book prints no answers, so the answers in this course are
          ours, and Exercise 1's rest on our reading of the arrowheads in its drawing.
        </p>
      </Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto', marginBottom: '1.2rem' }}>
        <table className="data" style={{ margin: 0 }} data-table="exercises">
          <thead>
            <tr>
              <th><Translated vi="Bài tập">Exercise</Translated></th>
              <th><Translated vi="Về">About</Translated></th>
              <th><Translated vi="Được giải ở đâu trong khóa học">Where it is worked here</Translated></th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1a</td>
              <td><Translated vi="Ma trận kề của đồ thị 8 đỉnh">The adjacency matrix of the 8-vertex graph</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      lăng kính định dạng trên <a href={href('ch18', 'start')}>trang mở đầu</a> khi chọn "Exercise 1 graph": các
                      cạnh ra của một đỉnh là các số 1 trên hàng của nó. Bên dưới, từng hàng một
                    </>
                  }
                >
                  the format lens on the <a href={href('ch18', 'start')}>start page</a> with the "Exercise 1 graph" chosen: a
                  vertex's out-edges are the 1s of its row. Below, row by row
                </Translated>
              </td>
            </tr>
            <tr>
              <td>1b</td>
              <td><Translated vi="Đồ thị ở dạng CSR, danh sách đỉnh kề đã sắp xếp">The graph in CSR, neighbour lists sorted</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      cùng lăng kính đó; câu hỏi đọc CSR trên <a href={href('ch18', 'start')}>trang mở đầu</a> dạy cách làm.{' '}
                      <code>srcPtrs = {list(EX1_CSR.srcPtrs)}</code>, <code>dst = {list(EX1_CSR.dst)}</code>
                    </>
                  }
                >
                  the same lens; the CSR reading question on the <a href={href('ch18', 'start')}>start page</a> teaches the
                  method. <code>srcPtrs = {list(EX1_CSR.srcPtrs)}</code>, <code>dst = {list(EX1_CSR.dst)}</code>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>1c</td>
              <td>
                <Translated vi="BFS từ 0: số luồng khởi chạy, lặp và gán nhãn ở mỗi lần gọi, cho bốn kernel">
                  BFS from 0: threads launched, looping and labelling per call, for four kernels
                </Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      hai dự đoán và trình khám phá "Một đồ thị, năm cách chạy một mức" trên{' '}
                      <a href={href('ch18', '18.3')}>18.3</a>, với đồ thị Bài tập 1 được nạp. Bảng đầy đủ ở bên dưới
                    </>
                  }
                >
                  the two predictions and the explorer "One graph, five ways to run a level" on{' '}
                  <a href={href('ch18', '18.3')}>18.3</a>, with the Exercise 1 graph loaded. The full table is below
                </Translated>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td><Translated vi="Mã host cho BFS tối ưu theo hướng">Host code for direction-optimised BFS</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      được giải như một câu hỏi quyết định, không phải mã host: câu hỏi Bài tập 2 và chế độ tối ưu theo hướng của
                      trình khám phá với thanh trượt điểm chuyển, trên <a href={href('ch18', '18.3')}>18.3</a>
                    </>
                  }
                >
                  worked as a decision question, not as host code: the Exercise 2 question and the explorer's
                  direction-optimised mode with its switch slider, on <a href={href('ch18', '18.3')}>18.3</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td><Translated vi="Hình 18.17 với một rào chắn lưới mỗi mức">Fig. 18.17 with one grid barrier per level</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      trình chạy từng bước "Ba rào chắn lưới" (chế độ Bài tập 3 của nó), câu hỏi Bài tập 3 và tab "kernel Bài tập
                      3" của đoạn mã khởi chạy, trên <a href={href('ch18', '18.6')}>18.6</a>; phần "Go further" của lab
                    </>
                  }
                >
                  the stepper "Three grid barriers" (its Exercise 3 mode), the Exercise 3 question and the "Exercise 3
                  kernel" tab of the launch listing, on <a href={href('ch18', '18.6')}>18.6</a>; the lab's "Go further"
                </Translated>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <Translated
        vi={
          <p>
            Bài tập 1a, từng hàng một: {EX1_ROWS.map((r, v) => `${v} → {${r.join(', ')}}`).join('; ')}. Đó là 15 số 1 trong một
            ma trận 8 × 8, đúng 15 phần tử tạo nên <code>dst</code>.
          </p>
        }
      >
        <p>
          Exercise 1a, row by row: {EX1_ROWS.map((r, v) => `${v} → {${r.join(', ')}}`).join('; ')}. That is 15 ones in an 8 ×
          8 matrix, the same 15 entries that make up <code>dst</code>.
        </p>
      </Translated>
      <Source section="18.10" pages="449–450" />
      <Translated
        vi={
          <p>
            Bài tập 1c, theo cách mô hình của khóa học đếm, từ gốc 0. Phép tìm kiếm gán nhãn {'{2, 5}'}, rồi {'{1, 3, 7}'}, rồi{' '}
            {'{4, 6}'}, và cần lần gọi thứ tư để thấy không còn gì. Số luồng khởi chạy được đếm mỗi phần tử một luồng, trước khi
            làm tròn lên thành khối đầy đủ.
          </p>
        }
      >
        <p>
          Exercise 1c, as the course's model counts it, from root 0. The search labels {'{2, 5}'}, then {'{1, 3, 7}'}, then{' '}
          {'{4, 6}'}, and needs a fourth call to find that nothing is left. Launched threads are counted one per item,
          before rounding up to whole blocks.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Của chúng tôi, không phải của sách: hướng các cạnh là cách chúng tôi đọc p. 450, và các unit test của mô hình gắn
              chặt các số đếm này với ghi chú.
            </p>
          }
        >
          <p>
            Ours, not the book's: the edge directions are our reading of p. 450, and the model's unit tests pin these counts
            against the notes.
          </p>
        </Translated>
      </Aside>
      <Ex1cTable />
      <Translated
        vi={
          <p>
            Đọc bảng theo một hàng và lập luận về hiệu quả công việc của 18.3 hiện ra ở quy mô nhỏ. Push khởi chạy 8 luồng mỗi
            lần gọi và dùng nhiều nhất 3; kernel lấy cạnh làm trung tâm khởi chạy 15 và, ở lần gọi 3, để bốn luồng ghi hai đỉnh,
            bản trùng vô hại; phiên bản dùng frontier khởi chạy đúng những luồng có lặp. Mảnh còn thiếu của Bài tập 2, khi nào
            chuyển từ push sang pull, sách không đưa ra; quy tắc của cuGraph nằm ở 18.3.
          </p>
        }
      >
        <p>
          Read the table along a row and the work-efficiency argument of 18.3 shows up in small. Push launches 8 threads
          every call and uses at most 3; edge-centric launches 15 and, in call 3, lets four of them write two vertices,
          the benign duplicate; the frontier version launches exactly the threads that loop. Exercise 2's missing piece,
          when to switch from push to pull, the book does not give; cuGraph's rule is on 18.3.
        </p>
      </Translated>

      <h2><Translated vi="Những chỗ PDF có vẻ sai">Where the PDF looks wrong</Translated></h2>
      <Translated
        vi={
          <p>
            Bản sách mà khóa học này dựa vào có vài lỗi nhỏ rõ ràng trong chương 18. Chúng được liệt kê để bạn không bị bối rối
            khi đọc sách song song.
          </p>
        }
      >
        <p>
          The copy of the book this course was built from has a few apparent slips in chapter 18. They are listed so that you
          are not thrown by them when you read the book alongside.
        </p>
      </Translated>
      <Translated
        vi={
          <ul data-list="errata">
            <li>
              p. 441 gọi "the vertex-centric push kernel without frontiers" là Hình 18.12; thực ra đó là Hình 18.6, có dòng 09
              và 10 là phép kiểm tra và phép gán nhãn.
            </li>
            <li>
              p. 439: "the threads whose vertices are not in the current level do not do anything" ý nói mức trước, như dòng 05
              của Hình 18.6 cho thấy.
            </li>
            <li>
              p. 433 đặt <code>UNVISITED</code> bằng <code>cuda::std::numeric_limits&lt;unsigned&gt;</code>, vốn chỉ gọi tên
              trait chứ không phải giá trị của nó; biểu thức đầy đủ là{' '}
              <code>cuda::std::numeric_limits&lt;unsigned int&gt;::max()</code>.
            </li>
            <li>p. 439 in cận của pull với dấu ngoặc không cân bằng, "O(d · n + d · m))".</li>
            <li>Chú thích của Hình 18.5 ghi "breath-first search".</li>
            <li>p. 445: "thread blocks that are not schedule" ý là scheduled.</li>
            <li>
              Hình 18.7, 18.9, 18.11 và 18.13 mô tả lần gọi từ mức 1 sang mức 2 nhưng in mức cuối cùng của mọi đỉnh, kể cả 8,
              đỉnh lúc đó vẫn chưa được thăm. Trình khám phá trên 18.3 hiển thị các mức tại lần gọi đó.
            </li>
            <li>
              Hình 18.15 dòng 29 và Hình 18.17 dòng 35 đặt lại bộ đếm riêng bằng một lần ghi thường trong khi các luồng khác
              của khối có thể đang thực hiện <code>fetch_add</code> trên nó. Đó là một race condition (tranh chấp dữ liệu);
              trang <a href={href('ch18', '18.6')}>18.6</a> và lab ghi qua <code>atomic_ref</code> hoặc kẹp giá trị sau rào
              chắn.
            </li>
            <li>
              p. 431 nói chướng ngại hình chữ J "connects three net terminals", nhưng Hình 18.5 chỉ đánh dấu hai đầu nối của
              dây đang được định tuyến.
            </li>
            <li>
              p. 436 gọi mạng xã hội và đồ thị định tuyến mê cung là vô hướng, sau khi p. 428 mô tả đồ thị người theo dõi với
              các cạnh có hướng. Không phải mâu thuẫn (đồ thị bạn bè và đồ thị người theo dõi khác nhau), nhưng đáng đọc kỹ.
            </li>
          </ul>
        }
      >
        <ul data-list="errata">
          <li>
            p. 441 calls "the vertex-centric push kernel without frontiers" Fig. 18.12; it is Fig. 18.6, whose lines 09 and
            10 are the check and the label.
          </li>
          <li>
            p. 439: "the threads whose vertices are not in the current level do not do anything" means the previous level,
            as line 05 of Fig. 18.6 shows.
          </li>
          <li>
            p. 433 sets <code>UNVISITED</code> to <code>cuda::std::numeric_limits&lt;unsigned&gt;</code>, which names the
            trait and not its value; the full expression is <code>cuda::std::numeric_limits&lt;unsigned int&gt;::max()</code>.
          </li>
          <li>p. 439 prints the pull bound with an unbalanced parenthesis, "O(d · n + d · m))".</li>
          <li>Fig. 18.5's caption reads "breath-first search".</li>
          <li>p. 445: "thread blocks that are not schedule" means scheduled.</li>
          <li>
            Figs. 18.7, 18.9, 18.11 and 18.13 depict the call from level 1 to level 2 but print every vertex's final level,
            8 included, which is still unvisited at that point. The explorer on 18.3 shows levels as of the call.
          </li>
          <li>
            Fig. 18.15 line 29 and Fig. 18.17 line 35 reset the private counter with a plain store while other threads of the
            block may be doing <code>fetch_add</code> on it. That is a data race; page{' '}
            <a href={href('ch18', '18.6')}>18.6</a> and the lab store through the <code>atomic_ref</code> or clamp after the
            barrier.
          </li>
          <li>
            p. 431 says the J-shaped obstacle "connects three net terminals", but Fig. 18.5 marks only the two terminals of
            the wire being routed.
          </li>
          <li>
            p. 436 calls social networks and maze-routing graphs undirected, after p. 428 described follower graphs with
            directed edges. Not a contradiction (friend graphs and follower graphs differ), but worth reading carefully.
          </li>
        </ul>
      </Translated>
      <p>
        <Translated vi="Và những gì các đoạn mã để lại cho bạn, hoặc viết theo cách không mang sang được mọi backend, mà một bản build thật cần:">
          And what the listings leave to you, or write in a way that does not carry to every backend, which a real build needs:
        </Translated>
      </p>
      <Translated
        vi={
          <ul data-list="gaps">
            <li>
              Không kernel nào kiểm tra dung lượng của frontier chung, và không mã host nào cấp phát nó. Với compare-and-swap,{' '}
              <code>numVertices</code> phần tử là đủ; lab kiểm tra các word canh sau mỗi buffer.
            </li>
            <li>
              Không có vòng lặp host cho các kernel theo từng mức: đặt lại bộ đếm, khởi chạy, đọc kích thước về và hoán đổi
              frontier. <code>launchBfsPrivatised</code> của lab là một vòng lặp như vậy.
            </li>
            <li>
              Không có giá trị nào cho <code>PRIVATE_FRONTIER_CAPACITY</code> hay <code>numThreadsPerBlock</code>; lab dùng 256
              và 128.
            </li>
            <li>
              Không có struct <code>CSRGraph</code>, <code>CSCGraph</code> hay <code>COOGraph</code>, và không có{' '}
              <code>#include &lt;cuda/atomic&gt;</code> cho <code>cuda::atomic_ref</code> của Hình 18.12 đến 18.17.
            </li>
            <li>
              <code>grid.thread_rank()</code> và <code>grid.num_threads()</code> trả về giá trị 64-bit trên CUDA, mà Hình 18.17
              thu hẹp thành <code>unsigned int</code> không một lời giải thích; vô hại với mọi lưới đủ nhỏ để thường trú.
            </li>
            <li>
              Đồ thị Bài tập 1 là một hình không đánh số với các đầu mũi tên nhỏ; 15 cạnh dùng ở đây là cách chúng tôi đọc.
            </li>
            <li>
              p. 446 khởi chạy với bốn đối số. Đó là CUDA hợp lệ, có hai tham số cuối mang giá trị mặc định, nhưng không biên
              dịch được trên HIP; khóa học truyền đủ sáu.
            </li>
            <li>
              Hình 18.17 khai báo các biến <code>__shared__</code> bên trong vòng lặp mức. Điều đó hợp lệ, và dù sao chúng cũng
              sống bằng thời gian sống của khối; lab khai báo chúng ở đầu, nơi thời gian sống của chúng hiển nhiên.
            </li>
          </ul>
        }
      >
        <ul data-list="gaps">
          <li>
            No kernel checks the public frontier's capacity, and no host code allocates it. With the compare-and-swap,{' '}
            <code>numVertices</code> entries are enough; the lab checks guard words after each buffer.
          </li>
          <li>
            No host loop for the per-level kernels: resetting the counter, launching, reading the size back and swapping
            the frontiers. The lab's <code>launchBfsPrivatised</code> is one.
          </li>
          <li>
            No value for <code>PRIVATE_FRONTIER_CAPACITY</code> or <code>numThreadsPerBlock</code>; the lab uses 256 and
            128.
          </li>
          <li>
            No <code>CSRGraph</code>, <code>CSCGraph</code> or <code>COOGraph</code> struct, and no{' '}
            <code>#include &lt;cuda/atomic&gt;</code> for the <code>cuda::atomic_ref</code> of Figs. 18.12 to 18.17.
          </li>
          <li>
            <code>grid.thread_rank()</code> and <code>grid.num_threads()</code> return 64-bit values on CUDA, which Fig.
            18.17 narrows to <code>unsigned int</code> without a word; harmless for any grid small enough to be resident.
          </li>
          <li>
            The Exercise 1 graph is an unnumbered figure with small arrowheads; the 15 edges used here are our reading.
          </li>
          <li>
            p. 446 launches with four arguments. That is valid CUDA, whose last two parameters have defaults, but does not
            compile on HIP; the course passes all six.
          </li>
          <li>
            Fig. 18.17 declares its <code>__shared__</code> variables inside the level loop. That is legal, and they live as
            long as the block either way; the lab declares them at the top, where their lifetime is plain to see.
          </li>
        </ul>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Đây là cách chúng tôi đọc một bản PDF (notes/ch18.md, mục "Unclear in the PDF"), với các điểm về nhà cung cấp được
              kiểm tra ngày 2026-11-20 (notes/today/ch18-facts.md A1, A4, A7, A9, B1). Nếu bản của bạn khác, hãy tin bản của bạn.
            </p>
          }
        >
          <p>
            These are our readings of one PDF (notes/ch18.md, "Unclear in the PDF"), with the vendor points checked on
            2026-11-20 (notes/today/ch18-facts.md A1, A4, A7, A9, B1). If your copy differs, trust your copy.
          </p>
        </Translated>
      </Aside>

      <h2><Translated vi="Những gì chương này bỏ qua">What this chapter left out</Translated></h2>
      <Translated
        vi={
          <p>
            Chương này được rút gọn trong khóa học, nên đây là những gì không được đề cập và nơi tìm chúng. Phần nền tảng về
            ma trận kề và phân phối bậc của §18.1 (pp. 425–429) được một đoạn văn và lăng kính định dạng trên trang mở đầu. Định
            tuyến mê cung trong thiết kế chip, §18.2 (pp. 430–431), với các đầu nối, chướng ngại và heuristic phân xử của nó,
            được rút gọn còn một đoạn văn trên trang mở đầu và hoạt hình wavefront trên 18.3. Đồ thị thế giới nhỏ (p. 436) chỉ
            được nêu tên, và điểm chuyển cho BFS tối ưu theo hướng, điều sách không đưa ra, được thay bằng quy tắc của cuGraph
            trong một ghi chú trên 18.3; không chương trình host tối ưu theo hướng nào được viết. Cách phát biểu bằng đại số
            tuyến tính và GraphBLAS (p. 438) được một đoạn văn.
          </p>
        }
      >
        <p>
          The chapter is condensed in this course, so here is what is not covered and where to find it. The adjacency-matrix
          and degree-distribution background of §18.1 (pp. 425–429) gets a paragraph and the format lens on the start page.
          Maze routing in chip design, §18.2 (pp. 430–431), with its net terminals, blockages and tie-breaking heuristics, is
          reduced to a paragraph on the start page and the wavefront animation on 18.3. Small-world graphs (p. 436) are named only, and the book's switch
          point for direction-optimised BFS, which it does not give, is replaced by cuGraph's rule in a note on 18.3; no
          direction-optimised host program is written. The linear-algebra formulation and GraphBLAS (p. 438) get one
          paragraph.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chia nhóm theo bậc, §18.8 (pp. 448–449), chỉ là văn xuôi, không có kernel và không có lab. Đồ thị lớn hơn bộ nhớ GPU
            và sắp xếp lại đỉnh (p. 449) chỉ được nêu tên, và RAPIDS cuGraph (p. 449) là một ghi chú. Các kernel pull, lấy cạnh
            làm trung tâm và tối ưu theo hướng là các đoạn mã và chế độ của trình khám phá trên 18.3, không bao giờ là bài lab;
            lab viết các kernel frontier và cung cấp push cùng frontier chưa riêng tư hóa làm mốc so sánh. Bài tập 2 được giải
            như một câu hỏi quyết định, không phải mã host. Tài liệu tham khảo 1 (Kepner và Gilbert, 2011) và 3 (Luo, Wong và
            Hwu, 2010) ở p. 451 chưa được kiểm tra.
          </p>
        }
      >
        <p>
          Degree bucketing, §18.8 (pp. 448–449), is prose with no kernel and no lab. Graphs larger than GPU memory and
          vertex reordering (p. 449) are named only, and RAPIDS cuGraph (p. 449) is a note. The pull, edge-centric and
          direction-optimised kernels are snippets and explorer modes on 18.3, never lab work; the lab writes the frontier
          kernels and provides push and the unprivatised frontier as baselines. Exercise 2 is worked as a decision
          question, not as host code. References 1 (Kepner and Gilbert, 2011) and 3 (Luo, Wong and Hwu, 2010) on p. 451
          were not checked.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch18-facts.md A11, A12, A13)', '2026-11-20 (notes/today/ch18-facts.md, mục A11, A12, A13)')}>
        <Translated
          vi={
            <p>
              Tài liệu tham khảo 2, 4 và 5 đã được kiểm tra: SuiteSparse:GraphBLAS, bài báo năm 2012 về duyệt đồ thị có khả năng
              mở rộng, và cuGraph.
            </p>
          }
        >
          <p>
            References 2, 4 and 5 were checked: SuiteSparse:GraphBLAS, the 2012 paper on scalable traversal, and cuGraph.
          </p>
        </Translated>
      </Today>
      <p>
        <Translated vi="Không phần nào về sau trong khóa học phụ thuộc vào những gì đã bị bỏ qua.">
          Nothing later in the course depends on what was left out.
        </Translated>
      </p>

      <h2><Translated vi="Dự án cho đến giờ">The project so far</Translated></h2>
      <p>
        <Translated
          vi={
            <>
              Chương này không thêm mốc dự án nào cho dự án xuyên suốt. Một engine suy luận không duyệt đồ thị nào, và BFS không
              nằm trên đường nóng của ai. Những gì chương này dạy quay lại dưới dạng các mẫu: danh sách công việc dựng bằng một
              bộ đếm atomic, như danh sách ứng viên top-p của mốc dự án M5; hàng đợi riêng với một lần commit duy nhất, như việc
              định tuyến expert của llama.cpp; và chi phí của nhiều lần khởi chạy nhỏ, thứ mà decode phải trả ở mỗi token. Lab
              18.1 là tùy chọn.
            </>
          }
        >
          This chapter adds no milestone to the through-line project. An inference engine traverses no graph, and BFS is on
          nobody's hot path. What the chapter teaches returns as patterns: work lists built with an atomic counter, as in
          the top-p candidate list of milestone M5; private queues with a single commit, as in llama.cpp's expert routing;
          and the cost of many small launches, which decoding pays every token. Lab 18.1 is optional.
        </Translated>{' '}
        <span data-lab181={passing.join(',') || 'none'}>
          {backends.length === 0 && !reports['emu']
            ? tr('No lab results are recorded on this machine yet.', 'Chưa có kết quả lab nào được ghi lại trên máy này.')
            : passing.length
              ? tr(
                  `Your own version passes on ${passing.map(where).join(' and ')} on this machine.`,
                  `Phiên bản của bạn đạt trên ${passing.map(where).join(' và ')} trên máy này.`,
                )
              : tr('Your own version has not passed on this machine yet.', 'Phiên bản của bạn chưa đạt trên máy này.')}
        </span>{' '}
        <Translated vi={<>Xem <a href={href('ch18', 'labs')}>lab</a>.</>}>
          See the <a href={href('ch18', 'labs')}>lab</a>.
        </Translated>
      </p>
      <Translated
        vi={
          <p>
            Chương 19 chuyển sang mạng tích chập, và bước đi chính của nó là điều mà chương này chỉ nêu tên: tìm cách nhìn trong
            đó một toán tử là một phép nhân ma trận và giao nó cho một routine đã được tinh chỉnh. Một mức BFS viết thành phép
            nhân ma trận thưa với vectơ là bước đi đó cho đồ thị; một phép tích chập (convolution) trải ra thành GEMM là bước đi
            đó cho ảnh. Bắt đầu ở <a href={href('ch19', 'start')}>chương 19</a>.
          </p>
        }
      >
        <p>
          Chapter 19 turns to convolutional networks, and its main move is one this chapter only named: find the view in
          which an operator is one matrix multiplication and hand it to a tuned routine. A BFS level written as a sparse
          matrix–vector product is that move for graphs; a convolution unfolded into a GEMM is that move for images. Start
          at <a href={href('ch19', 'start')}>chapter 19</a>.
        </p>
      </Translated>
      <Source section="18.4" pages="438" />
    </>
  );
}
