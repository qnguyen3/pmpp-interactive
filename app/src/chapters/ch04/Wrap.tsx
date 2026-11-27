import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Beyond, Source } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { capacity, waves } from '../../sim/blocksched';
import { PRESETS, warpsNeeded } from '../../sim/latency';
import { BOOK, occupancy, pct, type Limits } from '../../sim/occupancy';
import { classifyBlock, dims } from '../../sim/warps';
import { Translated, tr } from '../../lib/i18n';

// A kernel of our own, for one question. Checked copy: labs/ch04/snippets/warp_tail_sum.cu
// (Wrap.snippets.test.ts ties the two and pins the numbers the other questions take from the models).
const WARP_TAIL_SUM = `// Sum of each block's 256 values. The last six steps are written for one warp, with no barrier.
#define BLOCK 256

__global__ void blockSum(const float* in, float* out) {
    __shared__ float s[BLOCK];
    unsigned int t = threadIdx.x;
    s[t] = in[blockIdx.x * BLOCK + t];
    __syncthreads();
    for (unsigned int stride = BLOCK / 2; stride > 32; stride /= 2) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    if (t < 32) {
        volatile float* v = s;
        v[t] += v[t + 32];
        v[t] += v[t + 16];
        v[t] += v[t + 8];
        v[t] += v[t + 4];
        v[t] += v[t + 2];
        v[t] += v[t + 1];
    }
    if (t == 0) out[blockIdx.x] = s[0];
}`;

// ---------------------------------------------------------------- the numbers the questions take from the models
/** A 16 x 8 tile that skips its rows from K on: divergent warps at widths 32 and 64. */
export const TILE = dims(16, 8);
export const ROW_LIMITS = [2, 4, 6] as const;
export const divergentWarps = (k: number, width: number) =>
  classifyBlock(TILE, width, (_x, y) => y < k).states.filter((s) => s === 'divergent').length;

/** Section 4.6's load-heavy preset, then the same loop with four times the arithmetic per load. */
export const BEFORE_PACK = PRESETS.loadHeavy;
export const AFTER_PACK = { work: 4 * PRESETS.loadHeavy.work, latency: PRESETS.loadHeavy.latency };
export const NEEDED_BEFORE = warpsNeeded(BEFORE_PACK.work, BEFORE_PACK.latency);
export const NEEDED_AFTER = warpsNeeded(AFTER_PACK.work, AFTER_PACK.latency);

/** A small grid on a device described only by its limits (no product). */
export const SMALL_GRID = { blocks: 64, threadsPerBlock: 256, units: 40, slotsBefore: 2, slotsAfter: 4 } as const;
export const CAP_BEFORE = capacity(SMALL_GRID.units, SMALL_GRID.slotsBefore);
export const CAP_AFTER = capacity(SMALL_GRID.units, SMALL_GRID.slotsAfter);
export const WARPS_IN_GRID = SMALL_GRID.blocks * (SMALL_GRID.threadsPerBlock / BOOK.width!);

/** One kernel, two sets of limits: the book's example and an SM with 1536 thread slots and 16 block slots. */
export const KERNEL_40 = { threads: 256, regs: 40, shared: 0 };
export const SMALLER_SM: Limits = { ...BOOK, threadSlots: 1536, blockSlots: 16, shared: null };
export const ON_BOOK = occupancy({ ...BOOK, shared: null }, KERNEL_40);
export const ON_SMALLER = occupancy(SMALLER_SM, KERNEL_40);

const n = (v: number) => v.toLocaleString('en-US');

export const questionsWrap: Record<string, Question> = {
  'ch04/wrap/q-warpsync': {
    kind: 'choice',
    id: 'ch04/wrap/q-warpsync',
    prompt: (
      <Translated
        vi={
          <>
            Kernel ở trên bỏ rào chắn cho sáu bước cuối, vì "32 luồng là một warp, và một warp chạy đồng bộ từng bước". Chương
            này nói gì về sáu dòng đó?
          </>
        }
      >
        <>
          The kernel above leaves out the barrier for its last six steps, because "32 threads are one warp, and a warp
          runs in step". What does this chapter say about those six lines?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Chúng đúng: luồng 0 đến 31 tạo thành một warp, và một warp thực thi một lệnh cho mọi luồng của nó cùng lúc.">
            {'They are correct: threads 0 to 31 form one warp, and a warp executes one instruction for all its threads at once.'}
          </Translated>
        ),
        why: (
          <Translated vi="Đó là hình ảnh SIMD của phần 4.4 (p. 76). Phần 4.5 bổ sung rằng từ Volta trở đi bạn không thể giả định các luồng của một warp luôn đi cùng nhau (p. 81), và hướng dẫn của NVIDIA gọi các giả định đồng bộ theo warp là không hợp lệ.">
            {"That is section 4.4's SIMD picture (p. 76). Section 4.5 adds that from Volta on you cannot assume the threads of a warp stay together (p. 81), and NVIDIA's guide calls warp-synchronous assumptions invalid."}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chúng hỏng trên thiết bị có warp rộng 64, nơi luồng 0 đến 31 chỉ là nửa warp.">
            {'They fail on a device with 64-wide warps, where threads 0 to 31 are only half a warp.'}
          </Translated>
        ),
        why: (
          <Translated vi="Nửa warp vẫn nằm trong một warp, và AMD, với thiết bị gfx9 rộng 64, ghi trong tài liệu rằng một warp chạy lockstep. Độ rộng mà mã này không sống sót được là một độ rộng hẹp hơn 32, nơi luồng 0 đến 31 sẽ là hai nhóm.">
            {'Half a warp is still inside one warp, and AMD, whose gfx9 devices are 64 wide, documents that a warp runs in lockstep. The width this code cannot survive is one narrower than 32, where threads 0 to 31 would be two groups.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chúng giả định luồng 0 đến 31 chạy đồng bộ như một nhóm, điều mà chỉ AMD hứa; mỗi bước cần được đồng bộ hóa.">
            {'They assume threads 0 to 31 run in step as one group, which only AMD promises; each step needs synchronising.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chúng đúng vì rào chắn ở cuối vòng lặp đã đồng bộ hóa cả khối.">
            {'They are correct because the barrier at the end of the loop has already synchronised the block.'}
          </Translated>
        ),
        why: (
          <Translated vi="Rào chắn đó sắp thứ tự cho vòng lặp. Trong sáu dòng sau nó, mỗi luồng đọc một giá trị mà một luồng khác của cùng nhóm ghi trong cùng dòng, không có gì ở giữa.">
            {'That barrier orders the loop. In the six lines after it, each thread reads a value that another thread of the same group writes in the same line, with nothing between them.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Ở mỗi dòng, luồng <code>t</code> đọc <code>v[t + d]</code>, thứ mà luồng <code>t + d</code> ghi trong cùng dòng.
              Điều đó chỉ đúng nếu mọi luồng đọc trước khi bất kỳ luồng nào ghi, đó là điều lockstep sẽ đem lại. Phần 4.4 bảo
              hãy giả định các luồng của một khối chạy theo thứ tự bất kỳ và đặt rào chắn giữa các pha (p. 74); phần 4.5 nói
              từ Volta trở đi các luồng của một warp có thể không đi cùng nhau và chỉ tới <code>__syncwarp()</code> (p. 81).
              Bài tập 5 đã hỏi cùng câu hỏi về <code>__syncthreads()</code>.
            </p>
            <p>
              Chỉ AMD hứa lockstep; NVIDIA đã không hứa từ compute capability 7.0, còn Apple không ghi trong tài liệu cả
              lockstep lẫn độ rộng. Trên CUDA, cách sửa là đồng bộ hóa warp tường minh giữa lần đọc và lần ghi của mỗi bước;
              viết cho mọi backend, đó là một rào chắn mà cả khối đều tới, hoặc các hàm cấp warp mà chương 10 đến 12 dùng. Độ
              rộng 32 là giả định thứ hai, và cùng mã nguồn đó chạy trên warp rộng 64 dưới HIP.
            </p>
          </>
        }
      >
        <>
          <p>
            In each line thread <code>t</code> reads <code>v[t + d]</code>, which thread <code>t + d</code> writes in the
            same line. That is correct only if every thread reads before any thread writes, which is what lockstep would
            give. Section 4.4 says to assume the threads of a block run in any order and to put a barrier between phases
            (p. 74); section 4.5 says that from Volta on the threads of one warp may not stay together and points to{' '}
            <code>__syncwarp()</code> (p. 81). Exercise 5 asked the same question about <code>__syncthreads()</code>.
          </p>
          <p>
            Only AMD promises lockstep; NVIDIA has not since compute capability 7.0, and Apple documents neither lockstep
            nor a width. On CUDA the repair is explicit warp synchronisation between the read and the write of each step;
            written for every backend, it is a barrier the whole block reaches, or the warp-level functions chapters 10 to
            12 use.
            The width 32 is a second assumption, and the same source runs on 64-wide warps under HIP.
          </p>
        </>
      </Translated>
    ),
    source:
      'ch. 4 §4.4 p. 74, §4.5 p. 81, exercise 5 p. 91; notes/today/ch04-facts.md A10, B5, C2; the kernel is ours',
    vi: { source: 'ch. 4 §4.4 p. 74, §4.5 p. 81, bài tập 5 p. 91; notes/today/ch04-facts.md A10, B5, C2; kernel là của chúng tôi' },
  },
  'ch04/wrap/q-warpbarrier': {
    kind: 'choice',
    id: 'ch04/wrap/q-warpbarrier',
    prompt: (
      <Translated
        vi={
          <>
            Một khối 128 luồng trên thiết bị có warp rộng 32 chạy{' '}
            <code>if (threadIdx.x &lt; 32) {'{'} ...; __syncthreads(); {'}'}</code>. Điều nào đúng?
          </>
        }
      >
        <>
          A block of 128 threads on a device with 32-wide warps runs{' '}
          <code>if (threadIdx.x &lt; 32) {'{'} ...; __syncthreads(); {'}'}</code>. Which is true?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nó hợp lệ: điều kiện giống nhau với mọi luồng của một warp, nên không warp nào phân kỳ.">
            {'It is legal: the condition is the same for every thread of a warp, so no warp diverges.'}
          </Translated>
        ),
        why: (
          <Translated vi="Không warp nào phân kỳ, đó là câu hỏi của phần 4.5. Quy tắc rào chắn là của phần 4.3, và nó nói về khối.">
            {'No warp diverges, which is section 4.5\'s question. The barrier rule is section 4.3\'s, and it is about the block.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó hợp lệ trên thiết bị mà warp chạy lockstep, và không xác định trên mọi thiết bị khác.">
            {'It is legal on a device where warps run in lockstep, and undefined on every other device.'}
          </Translated>
        ),
        why: (
          <Translated vi="Lockstep là lời hứa về các luồng bên trong một warp. Ba warp còn lại không bao giờ tới rào chắn này.">
            {'Lockstep is a promise about the threads inside one warp. The other three warps never reach this barrier at all.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không warp nào phân kỳ, và nó vẫn không hợp lệ: ba trong bốn warp không bao giờ tới rào chắn.">
            {'No warp diverges, and it is still illegal: three of the four warps never reach the barrier.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nó không hợp lệ chỉ vì điều kiện chia tách warp đầu tiên trên thiết bị có warp rộng 64.">
            {'It is illegal only because the condition splits the first warp on a device with 64-wide warps.'}
          </Translated>
        ),
        why: (
          <Translated vi="Ở độ rộng 64 warp đầu tiên đúng là phân kỳ, nhưng rào chắn đã không hợp lệ ngay ở 32, nơi không có gì phân kỳ.">
            {'At 64 the first warp does diverge, but the barrier is illegal at 32 already, where nothing diverges.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hai câu hỏi dễ bị gộp làm một. Điều kiện có chia tách một warp không? Câu đó quyết định số lượt, và ở đây ở độ rộng
            32 thì không. Mọi luồng của khối có tới cùng một rào chắn không? Câu đó quyết định kernel có được định nghĩa hay
            không, và ở đây 96 luồng không bao giờ tới. NVIDIA cho phép rào chắn trong mã có điều kiện "only if the condition
            evaluates uniformly across the entire thread block" (chỉ khi điều kiện cho cùng kết quả trên toàn bộ khối luồng).
            Một điều kiện có thể đồng nhất theo warp mà vẫn không đồng nhất theo khối.
          </p>
        }
      >
        <p>
          Two questions that are easy to merge. Does the condition split a warp? That decides passes, and here it does
          not at width 32. Does every thread of the block reach the same barrier? That decides whether the kernel is
          defined, and here 96 threads never do. NVIDIA allows a barrier in conditional code "only if the condition
          evaluates uniformly across the entire thread block". A condition can be uniform per warp and still not per
          block.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.3 p. 72 and §4.5 p. 81; notes/today/ch04-facts.md A6',
    vi: { source: 'ch. 4 §4.3 p. 72 và §4.5 p. 81; notes/today/ch04-facts.md A6' },
  },
  'ch04/wrap/q-rowmask': {
    kind: 'choice',
    id: 'ch04/wrap/q-rowmask',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel attention phủ một ô bằng các khối 16 × 8 luồng (<code>blockDim.x = 16</code>,{' '}
            <code>blockDim.y = 8</code>) và bỏ qua các hàng của ô nằm quá cuối một chuỗi bằng{' '}
            <code>if (threadIdx.y &lt; K)</code>. Với K nào thì không warp nào phân kỳ, ở độ rộng 32 và ở độ rộng 64?
          </>
        }
      >
        <>
          An attention kernel covers a tile with blocks of 16 × 8 threads (<code>blockDim.x = 16</code>,{' '}
          <code>blockDim.y = 8</code>) and skips the tile's rows past the end of a sequence with{' '}
          <code>if (threadIdx.y &lt; K)</code>. For which K does no warp diverge, at a width of 32 and at a width of 64?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="K bất kỳ: điều kiện kiểm tra y, còn warp được cắt theo x.">
            {'Any K: the condition tests y, and a warp is cut along x.'}
          </Translated>
        ),
        why: (
          <Translated vi="Một warp được cắt từ chỉ số tuyến tính, x nhanh nhất. Với hàng dài 16 luồng, một warp trải trên nhiều hàng, và một phép kiểm tra trên y có thể chia tách nó.">
            {'A warp is cut from the linear index, x fastest. With rows 16 threads long, one warp spans several rows, and a test on y can split it.'}
          </Translated>
        ),
      },
      {
        label: 'K = 2',
        why: tr(`At width 32 a warp is two rows and K = 2 is a warp boundary. At 64 a warp is four rows, and K = 2 splits the first one: ${divergentWarps(2, 64)} divergent warp.`, `Ở độ rộng 32 một warp là hai hàng và K = 2 là ranh giới warp. Ở 64 một warp là bốn hàng, và K = 2 chia tách warp đầu tiên: ${divergentWarps(2, 64)} warp phân kỳ.`),
      },
      { label: 'K = 4', correct: true },
      {
        label: 'K = 6',
        why: tr(`Fine at 32, where 6 is a multiple of two rows. At 64 rows 4 to 7 are one warp and K = 6 splits it: ${divergentWarps(6, 64)} divergent warp.`, `Ổn ở 32, nơi 6 là bội số của hai hàng. Ở 64, hàng 4 đến 7 là một warp và K = 6 chia tách nó: ${divergentWarps(6, 64)} warp phân kỳ.`),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một hàng rộng 16 đặt {32 / TILE.x} hàng vào một warp rộng 32 và {64 / TILE.x} hàng vào một warp rộng 64. Một phép
            kiểm tra theo hàng không chia tách warp nào khi K là bội số của số hàng mỗi warp ở mọi độ rộng mà mã sẽ gặp. K = 4
            là lựa chọn duy nhất là bội số của cả 2 và 4: {divergentWarps(4, 32)} warp phân kỳ ở 32 và {divergentWarps(4, 64)}{' '}
            ở 64. Độ rộng thuộc về thiết bị, nên một kernel muốn có tính chất này phải đọc nó, như phần 4.4 đã nói.
          </p>
        }
      >
        <p>
          A 16-wide row puts {32 / TILE.x} rows in a 32-wide warp and {64 / TILE.x} in a 64-wide one. A row test splits
          no warp when K is a multiple of the rows per warp at every width the code will meet. K = 4 is the only option
          that is a multiple of both 2 and 4: {divergentWarps(4, 32)} divergent warps at 32 and {divergentWarps(4, 64)}{' '}
          at 64. The width belongs to the device, so a kernel that wants this property has to read it, as section 4.4
          said.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.4 pp. 75–76 (the cut) and §4.5 p. 81 (conditions on threadIdx); notes/today/ch04-facts.md A8, B6; counted with app/src/sim/warps.ts',
    vi: { source: 'ch. 4 §4.4 pp. 75–76 (cách cắt) và §4.5 p. 81 (điều kiện trên threadIdx); notes/today/ch04-facts.md A8, B6; đếm bằng app/src/sim/warps.ts' },
  },
  'ch04/wrap/q-pack': {
    kind: 'numeric',
    id: 'ch04/wrap/q-pack',
    prompt: (
      <Translated
        vi={
          <>
            Trong mô hình của phần 4.6, các warp của một kernel decode phát {BEFORE_PACK.work} lệnh mỗi lần nạp rồi chờ{' '}
            {BEFORE_PACK.latency} chu kỳ, nên {NEEDED_BEFORE} warp giữ một ô phát lệnh luôn bận. Sau đó trọng số được lưu ở 8
            bit và mỗi lần nạp mang về bốn trọng số, nên mỗi lần nạp giờ nuôi {AFTER_PACK.work} lệnh và thời gian chờ không
            đổi. Giờ cần bao nhiêu warp để giữ ô đó luôn bận?
          </>
        }
      >
        <>
          In section 4.6's model, a decode kernel's warps issue {BEFORE_PACK.work} instructions per load and then wait{' '}
          {BEFORE_PACK.latency} cycles, so {NEEDED_BEFORE} warps keep one issue slot busy. The weights are then stored in 8
          bits and four come in with each load, so each load now feeds {AFTER_PACK.work} instructions and the wait is the
          same. How many warps keep the slot busy now?
        </>
      </Translated>
    ),
    answer: NEEDED_AFTER,
    unit: 'warps',
    explain: (
      <Translated
        vi={
          <p>
            Một warp dùng ô đó trong {AFTER_PACK.work} trên mỗi {AFTER_PACK.work + AFTER_PACK.latency} chu kỳ, nên cần{' '}
            {AFTER_PACK.work + AFTER_PACK.latency} / {AFTER_PACK.work}, làm tròn lên, = {NEEDED_AFTER} warp, giảm từ{' '}
            {NEEDED_BEFORE}. Thời gian chờ không ngắn đi. Mỗi lần chờ giờ che được nhiều phép tính hơn, nên cần ít warp hơn để
            che nó, và occupancy mà kernel cần cũng giảm theo.
          </p>
        }
      >
        <p>
          One warp uses the slot for {AFTER_PACK.work} of every {AFTER_PACK.work + AFTER_PACK.latency} cycles, so it takes{' '}
          {AFTER_PACK.work + AFTER_PACK.latency} / {AFTER_PACK.work}, rounded up, = {NEEDED_AFTER} warps, down from{' '}
          {NEEDED_BEFORE}. The wait did not get shorter. Each wait now covers more arithmetic, so fewer warps are needed to
          hide it, and the occupancy the kernel needs falls with them.
        </p>
      </Translated>
    ),
    source: 'our model of ch. 4 §4.6, pp. 83–85 (app/src/sim/latency.ts); the 8-bit weights are ours',
    vi: { unit: 'warp', source: 'mô hình của chúng tôi cho ch. 4 §4.6, pp. 83–85 (app/src/sim/latency.ts); trọng số 8 bit là của chúng tôi' },
  },
  'ch04/wrap/q-smallgrid': {
    kind: 'choice',
    id: 'ch04/wrap/q-smallgrid',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel lúc decode với batch 1 khởi chạy {SMALL_GRID.blocks} khối {SMALL_GRID.threadsPerBlock} luồng trên một
            thiết bị có {SMALL_GRID.units} SM, nơi thanh ghi giới hạn nó ở {SMALL_GRID.slotsBefore} khối mỗi SM. Các warp của
            nó phần lớn thời gian chờ bộ nhớ. Ai đó cắt bớt thanh ghi để {SMALL_GRID.slotsAfter} khối vừa mỗi SM. Điều đó có
            giúp ích không?
          </>
        }
      >
        <>
          A decode-time kernel at batch 1 launches {SMALL_GRID.blocks} blocks of {SMALL_GRID.threadsPerBlock} threads on a
          device with {SMALL_GRID.units} SMs, where registers limit it to {SMALL_GRID.slotsBefore} blocks per SM. Its warps
          wait on memory most of the time. Someone trims its registers so that {SMALL_GRID.slotsAfter} blocks fit per SM.
          Will that help?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có: số khối vừa mỗi SM gấp đôi, nên số warp thường trú để che các lần chờ cũng gấp đôi.">
            {'Yes: twice as many blocks fit on each SM, so twice as many warps are resident to cover the waits.'}
          </Translated>
        ),
        why: tr(`Twice as many could fit. The grid has ${SMALL_GRID.blocks} blocks, and ${CAP_BEFORE} could already run at once.`, `Có thể vừa gấp đôi. Lưới có ${SMALL_GRID.blocks} khối, và ${CAP_BEFORE} khối vốn đã có thể chạy cùng lúc.`),
      },
      {
        label: tr(`No: all ${SMALL_GRID.blocks} blocks already run at once, with room for ${CAP_BEFORE}; the new slots would stay empty.`, `Không: cả ${SMALL_GRID.blocks} khối vốn đã chạy cùng lúc, với chỗ cho ${CAP_BEFORE}; các ô mới sẽ để trống.`),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có: với nhiều khối mỗi SM hơn, đợt cuối ngắn lại, nên hiệu ứng đuôi giảm.">
            {'Yes: with more blocks per SM the last wave gets shorter, so the tail effect shrinks.'}
          </Translated>
        ),
        why: tr(`There is only one wave, and it is partial: ${SMALL_GRID.blocks} of ${CAP_BEFORE} slots.`, `Chỉ có một đợt, và nó chưa đầy: ${SMALL_GRID.blocks} trên ${CAP_BEFORE} ô.`),
      },
      {
        label: tr(`It depends on the order in which the runtime places the blocks on the ${SMALL_GRID.units} SMs.`, `Còn tùy vào thứ tự runtime đặt các khối lên ${SMALL_GRID.units} SM.`),
        why: tr(`Whatever the order, at most ${SMALL_GRID.blocks} blocks exist to be placed.`, `Dù thứ tự thế nào, chỉ có tối đa ${SMALL_GRID.blocks} khối để đặt.`),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            {SMALL_GRID.blocks} khối trên một thiết bị chạy {CAP_BEFORE} khối cùng lúc là {waves(SMALL_GRID.blocks, CAP_BEFORE)}{' '}
            đợt; ở {CAP_AFTER} thì là {waves(SMALL_GRID.blocks, CAP_AFTER)}. Dù thế nào, cả thiết bị giữ{' '}
            {n(WARPS_IN_GRID)} warp của lưới và không hơn. Occupancy là trần của số warp thường trú; lưới quyết định có gì chạm
            tới trần đó hay không. Một lưới nhỏ phải chờ bộ nhớ cần thêm việc song song, không cần thêm chỗ: một lưới có nhiều
            luồng hơn, ví dụ chia đôi công việc của mỗi khối.
          </p>
        }
      >
        <p>
          {SMALL_GRID.blocks} blocks on a device that runs {CAP_BEFORE} at once is {waves(SMALL_GRID.blocks, CAP_BEFORE)} of
          a wave; at {CAP_AFTER} it is {waves(SMALL_GRID.blocks, CAP_AFTER)}. Either way the whole device holds the grid's{' '}
          {n(WARPS_IN_GRID)} warps and no more. Occupancy is a ceiling on resident warps; the grid decides whether
          anything reaches it. A small grid that waits on memory needs more parallel work, not more room: a grid with more
          threads, for example each block's work split in two.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.3 p. 73 (waves) and §4.7 p. 85 (occupancy); the decode setting is ours',
    vi: { source: 'ch. 4 §4.3 p. 73 (đợt) và §4.7 p. 85 (occupancy); bối cảnh decode là của chúng tôi' },
  },
  'ch04/wrap/q-twodevices': {
    kind: 'choice',
    id: 'ch04/wrap/q-twodevices',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel dùng {KERNEL_40.regs} thanh ghi mỗi luồng và khối {KERNEL_40.threads} luồng. Nó chạy trên SM ví dụ của
            sách ({n(BOOK.threadSlots!)} ô luồng, {BOOK.blockSlots} ô khối, {n(BOOK.regs!)} thanh ghi) và trên một SM có{' '}
            {n(SMALLER_SM.threadSlots!)} ô luồng, {SMALLER_SM.blockSlots} ô khối và cùng số thanh ghi. Occupancy của nó là bao
            nhiêu?
          </>
        }
      >
        <>
          A kernel uses {KERNEL_40.regs} registers per thread and {KERNEL_40.threads}-thread blocks. It runs on the book's
          example SM ({n(BOOK.threadSlots!)} thread slots, {BOOK.blockSlots} block slots, {n(BOOK.regs!)} registers) and
          on an SM with {n(SMALLER_SM.threadSlots!)} thread slots, {SMALLER_SM.blockSlots} block slots and the same
          registers. What are its occupancies?
        </>
      </Translated>
    ),
    options: [
      {
        label: tr(`${pct(ON_BOOK.byWarps)}% on both: the kernel, its registers and its blocks are the same on the two SMs.`, `${pct(ON_BOOK.byWarps)}% trên cả hai: kernel, thanh ghi và khối của nó giống nhau trên hai SM.`),
        why: (
          <Translated vi="Occupancy là một phần trong số ô warp của chính mỗi SM, và hai SM có số ô khác nhau.">
            {'Occupancy is a fraction of each SM\'s own warp slots, and the two SMs have different numbers of them.'}
          </Translated>
        ),
      },
      {
        label: tr(`${pct(ON_BOOK.byWarps)}% on the book's example and ${pct(ON_SMALLER.byWarps)}% on the other, so the other SM keeps more warps resident.`, `${pct(ON_BOOK.byWarps)}% trên ví dụ của sách và ${pct(ON_SMALLER.byWarps)}% trên SM kia, nên SM kia giữ nhiều warp thường trú hơn.`),
        why: tr(`Count the warps: ${ON_BOOK.residentWarps} on each.`, `Hãy đếm số warp: ${ON_BOOK.residentWarps} trên mỗi SM.`),
      },
      {
        label: tr(`${pct(ON_BOOK.byWarps)}% on the book's example and ${pct(ON_SMALLER.byWarps)}% on the other, with the same ${ON_BOOK.residentWarps} warps resident on each.`, `${pct(ON_BOOK.byWarps)}% trên ví dụ của sách và ${pct(ON_SMALLER.byWarps)}% trên SM kia, với cùng ${ON_BOOK.residentWarps} warp thường trú trên mỗi SM.`),
        correct: true,
      },
      {
        label: tr(`${pct(ON_SMALLER.byWarps)}% on both: neither the thread limit nor the block limit is reached on either SM.`, `${pct(ON_SMALLER.byWarps)}% trên cả hai: cả giới hạn luồng lẫn giới hạn khối đều không bị chạm tới trên SM nào.`),
        why: tr(`Registers are: a block takes ${n(KERNEL_40.threads * KERNEL_40.regs)} of them, so ${ON_BOOK.blocks} blocks fit, ${ON_BOOK.residentWarps} warps, which is not all of the book's ${ON_BOOK.warpSlots} warp slots.`, `Thanh ghi thì có: một khối lấy ${n(KERNEL_40.threads * KERNEL_40.regs)} thanh ghi, nên vừa ${ON_BOOK.blocks} khối, ${ON_BOOK.residentWarps} warp, chưa phải toàn bộ ${ON_BOOK.warpSlots} ô warp của sách.`),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một khối cần {KERNEL_40.threads} × {KERNEL_40.regs} = {n(KERNEL_40.threads * KERNEL_40.regs)} thanh ghi, nên{' '}
            {ON_BOOK.blocks} khối vừa trong {n(BOOK.regs!)} trên cả hai SM: {ON_BOOK.residentWarps} warp. Trên ví dụ của sách
            đó là {ON_BOOK.residentWarps} trên {ON_BOOK.warpSlots} ô warp, {pct(ON_BOOK.byWarps)}%; trên SM kia là{' '}
            {ON_SMALLER.residentWarps} trên {ON_SMALLER.warpSlots}, {pct(ON_SMALLER.byWarps)}%. Con số phần trăm mô tả SM nhiều
            ngang với mô tả kernel. Hãy so sánh thiết bị bằng số warp thường trú, và đọc mọi giới hạn từ thiết bị.
          </p>
        }
      >
        <p>
          One block needs {KERNEL_40.threads} × {KERNEL_40.regs} = {n(KERNEL_40.threads * KERNEL_40.regs)} registers, so{' '}
          {ON_BOOK.blocks} blocks fit in {n(BOOK.regs!)} on both SMs: {ON_BOOK.residentWarps} warps. On the book's example
          that is {ON_BOOK.residentWarps} of {ON_BOOK.warpSlots} warp slots, {pct(ON_BOOK.byWarps)}%; on the other it is{' '}
          {ON_SMALLER.residentWarps} of {ON_SMALLER.warpSlots}, {pct(ON_SMALLER.byWarps)}%. The percentage describes the
          SM as much as the kernel. Compare devices by warps resident, and read every limit from the device.
        </p>
      </Translated>
    ),
    source: "ch. 4 §4.7 pp. 85–86; per-SM limits differ across today's GPUs (notes/today/ch04-facts.md A12); computed with app/src/sim/occupancy.ts",
    vi: { source: 'ch. 4 §4.7 pp. 85–86; giới hạn mỗi SM khác nhau giữa các GPU hiện nay (notes/today/ch04-facts.md A12); tính bằng app/src/sim/occupancy.ts' },
  },
};

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const m1 = backends.filter(
    (b) =>
      labStatus(reports[b], 'ch03/lab3_naive_gemm', 'work') === 'pass' &&
      labStatus(reports[b], 'ch04/lab1_launchconfig', 'work') === 'pass',
  );
  const where = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch04/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}><Translated vi="Những gì cần mang theo">What to carry forward</Translated></h2>
      <Translated
        vi={
          <ul>
            <li>
              GPU là một mảng các SM (compute unit trên AMD; Apple không mô tả đơn vị tương ứng). Một lưới được giao cho chúng
              từng khối một, mỗi khối trọn vẹn, theo thứ tự bất kỳ. Đó là lý do khối là phạm vi của rào chắn và của bộ nhớ
              chia sẻ, và vì sao một lưới chạy không cần sửa trên thiết bị ở mọi kích thước.
            </li>
            <li>
              Một rào chắn phải được mọi luồng của khối tới, tại cùng một lời gọi. Điều kiện bao quanh nó phải đồng nhất trên
              cả khối, không chỉ trên một warp.
            </li>
            <li>
              Một khối được cắt thành warp từ chỉ số luồng tuyến tính, x nhanh nhất. Độ rộng thuộc về thiết bị, và trên Metal
              thuộc về pipeline đã biên dịch: hãy đọc nó, đừng bao giờ viết 32.
            </li>
            <li>
              Các luồng của một warp bất đồng tại một nhánh tốn thêm lượt. Một phép kiểm tra biên tốn ít khi dữ liệu lớn. Số
              vòng lặp theo từng luồng và mask phụ thuộc vị trí mới là nơi phân kỳ tốn kém.
            </li>
            <li>
              Một SM giữ nhiều warp hơn hẳn số nó thực thi, để một warp đang chờ không tốn gì trong khi có warp khác sẵn sàng.
              Nó cần bao nhiêu warp tùy vào lượng việc một warp làm giữa các lần chờ.
            </li>
            <li>
              Occupancy là số warp thường trú trên số ô warp. Ô khối, ô luồng, thanh ghi và (chương 5) bộ nhớ chia sẻ mỗi thứ
              đều giới hạn nó, khác nhau trên mỗi thiết bị, và một thay đổi nhỏ có thể làm nó tụt dốc đột ngột. Tăng nó chỉ có
              lợi khi còn những lần chờ cần che.
            </li>
            <li>Hãy hỏi thiết bị: struct thuộc tính trên CUDA và HIP, thiết bị và pipeline đã biên dịch trên Metal.</li>
          </ul>
        }
      >
        <ul>
          <li>
            A GPU is an array of SMs (compute units on AMD; Apple documents no counterpart). A grid is handed to them
            block by block, each block whole, in any order. That is why a block is the scope of a barrier and of shared
            memory, and why one grid runs unchanged on a device of any size.
          </li>
          <li>
            A barrier must be reached by every thread of the block, at the same call. A condition around it has to be
            uniform across the block, not just across a warp.
          </li>
          <li>
            A block is cut into warps from its linear thread index, x fastest. The width belongs to the device, and on
            Metal to the compiled pipeline: read it, never write 32.
          </li>
          <li>
            Threads of one warp that disagree at a branch cost extra passes. A boundary check costs little once the data
            is large. Per-thread trip counts and position-dependent masks are where divergence costs.
          </li>
          <li>
            An SM holds many more warps than it executes, so that a waiting warp costs nothing while another is ready.
            How many it needs depends on how much work a warp does between waits.
          </li>
          <li>
            Occupancy is resident warps over warp slots. Block slots, thread slots, registers and (chapter 5) shared
            memory each cap it, on each device differently, and a small change can drop it off a cliff. More of it pays
            only while there are waits left to hide.
          </li>
          <li>Ask the device: the properties struct on CUDA and HIP, the device and the compiled pipeline on Metal.</li>
        </ul>
      </Translated>
      <Source section="4.9" pages="89–90" />

      <h2><Translated vi="Bài kiểm tra tổng hợp">A mixed check</Translated></h2>
      <p>
        <Translated vi="Những câu này dựa trên toàn bộ chương và trên những tình huống mà các phần trước chưa trình bày.">
          These draw on the whole chapter and on situations the sections did not show.
        </Translated>
        {asked ? tr(` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.`, ` Đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu, tính theo câu trả lời gần nhất của bạn cho mỗi câu.`) : ''}
      </p>
      <Beyond>
        <Translated
          vi={
            <p>
              Kernel, bối cảnh attention và decode, và trọng số 8 bit là của chúng tôi, còn số chu kỳ là đầu vào cho mô hình
              của phần 4.6, không phải số đo.
            </p>
          }
        >
          <p>
            The kernel, the attention and decode settings and the 8-bit weights are ours, and the cycle counts are
            inputs to section 4.6's model, not measurements.
          </p>
        </Translated>
      </Beyond>
      <Code
        tabs={[
          {
            label: tr('CUDA (not a lab file: for this question only)', 'CUDA (không phải file lab: chỉ dành cho câu hỏi này)'),
            code: WARP_TAIL_SUM,
            status: tr('front-end-checked; never run', 'front-end-checked; chưa từng chạy'),
            tiers: { in: 'global', out: 'global', s: 'shared', v: 'shared', t: 'reg', stride: 'reg' },
          },
        ]}
      />
      <Ask q={q['ch04/wrap/q-warpsync']} label={tr('Read the code', 'Đọc mã')} />
      <Ask q={q['ch04/wrap/q-warpbarrier']} />
      <Ask q={q['ch04/wrap/q-rowmask']} />
      <Ask q={q['ch04/wrap/q-pack']} />
      <Ask q={q['ch04/wrap/q-smallgrid']} />
      <Ask q={q['ch04/wrap/q-twodevices']} />

      <h2><Translated vi="Bài tập của sách">The book's exercises</Translated></h2>
      <Translated
        vi={
          <p>
            Chương kết thúc bằng chín bài tập (pp. 90–92). Mỗi bài đều được giải ở đâu đó trong khóa học này; sách không in đáp
            án, nên đáp án ở đây là của chúng tôi.
          </p>
        }
      >
        <p>
          The chapter ends with nine exercises (pp. 90–92). Each is worked somewhere in this course; the book prints no
          answers, so the answers here are ours.
        </p>
      </Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }} data-table="exercises">
          <thead>
            <tr>
              <th><Translated vi="Bài tập">Exercise</Translated></th>
              <th><Translated vi="Về">About</Translated></th>
              <th><Translated vi="Được giải ở đâu">Where it is worked here</Translated></th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1 (a, b)</td>
              <td><Translated vi="Số warp mỗi khối và trong lưới cho một kernel cho trước">Warps per block and in the grid for a given kernel</Translated></td>
              <td><Translated vi={<>một câu hỏi trong <a href={href('ch04', '4.4')}>4.4</a></>}>a question in <a href={href('ch04', '4.4')}>4.4</a></Translated></td>
            </tr>
            <tr>
              <td>1 (c, d, e)</td>
              <td><Translated vi="Warp hoạt động và warp phân kỳ, hiệu suất SIMD, số lần lặp">Active and divergent warps, SIMD efficiency, loop iterations</Translated></td>
              <td><Translated vi={<>câu dự đoán mười con số và trình mô phỏng phân kỳ trong <a href={href('ch04', '4.5')}>4.5</a></>}>the ten-number prediction and the divergence simulator in <a href={href('ch04', '4.5')}>4.5</a></Translated></td>
            </tr>
            <tr>
              <td>2, 3</td>
              <td><Translated vi="Số luồng trong một lưới, và các warp phân kỳ tại phép kiểm tra biên của nó">Threads in a grid, and warps that diverge at its boundary check</Translated></td>
              <td><Translated vi={<>hai câu hỏi trong <a href={href('ch04', '4.5')}>4.5</a></>}>two questions in <a href={href('ch04', '4.5')}>4.5</a></Translated></td>
            </tr>
            <tr>
              <td>4</td>
              <td><Translated vi="Thời gian chờ ở một rào chắn">Time spent waiting at a barrier</Translated></td>
              <td><Translated vi={<>dòng thời gian rào chắn và câu dự đoán của nó trong <a href={href('ch04', '4.3')}>4.3</a></>}>the barrier timeline and its prediction in <a href={href('ch04', '4.3')}>4.3</a></Translated></td>
            </tr>
            <tr>
              <td>5</td>
              <td><Translated vi="Bỏ rào chắn trong khối 32 luồng">Leaving out the barrier in 32-thread blocks</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      một câu hỏi trong <a href={href('ch04', '4.5')}>4.5</a>, thí nghiệm của{' '}
                      <a href={href('ch04', 'labs')}>lab 4.3</a>, và câu hỏi đầu tiên ở trên
                    </>
                  }
                >
                  a question in <a href={href('ch04', '4.5')}>4.5</a>, the experiment of{' '}
                  <a href={href('ch04', 'labs')}>lab 4.3</a>, and the first question above
                </Translated>
              </td>
            </tr>
            <tr>
              <td>6</td>
              <td><Translated vi="Kích thước khối đặt được nhiều luồng nhất lên một SM">The block size that puts most threads on an SM</Translated></td>
              <td>
                <Translated vi={<>một câu hỏi trong <a href={href('ch04', '4.7')}>4.7</a>; các bài kiểm thử mô hình của lab 4.4</>}>
                  a question in <a href={href('ch04', '4.7')}>4.7</a>; the model tests of lab 4.4
                </Translated>
              </td>
            </tr>
            <tr>
              <td>7, 8</td>
              <td><Translated vi="Cách phân bổ nào vừa, ở occupancy bao nhiêu, và cái gì giới hạn nó">Which assignments fit, at what occupancy, and what limits it</Translated></td>
              <td>
                <Translated vi={<>trình kiểm tra ở cuối <a href={href('ch04', '4.7')}>4.7</a>; các bài kiểm thử mô hình của lab 4.4</>}>
                  the checker at the end of <a href={href('ch04', '4.7')}>4.7</a>; the model tests of lab 4.4
                </Translated>
              </td>
            </tr>
            <tr>
              <td>9</td>
              <td><Translated vi="Một lần khởi chạy mà thiết bị không thể đã chạy">A launch the device cannot have run</Translated></td>
              <td><Translated vi={<>một câu hỏi trong <a href={href('ch04', '4.8')}>4.8</a></>}>a question in <a href={href('ch04', '4.8')}>4.8</a></Translated></td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2><Translated vi="Những chỗ PDF có vẻ sai">Where the PDF looks wrong</Translated></h2>
      <Translated
        vi={
          <p>
            Bản sách mà khóa học này dựa vào có vài chỗ có vẻ nhầm ở chương 4, và vài phát biểu đã lỗi thời. Chúng được liệt kê
            để bạn không bị vấp khi đọc sách song song.
          </p>
        }
      >
        <p>
          The copy of the book this course was built from has a few apparent slips in chapter 4, and a few statements
          that have dated. They are listed so that you are not thrown by them when you read the book alongside.
        </p>
      </Translated>
      <Translated
        vi={
          <ul data-list="errata">
            <li>
              p. 69 nói giới hạn số khối mỗi SM được bàn ở "Section 4.6". Thực ra chúng ở phần 4.7 (
              <a href={href('ch04', '4.2')}>4.2</a> có ghi chú điều này).
            </li>
            <li>
              Fig. 4.4 (p. 72) thiếu <code>__global__</code>, và phần chữ dẫn số dòng cao hơn một so với số được in (
              <a href={href('ch04', '4.3')}>4.3</a>).
            </li>
            <li>
              Khung bên lề ở pp. 78–79 chỉ tới các mũi tên "in Fig. 4.8"; các mũi tên nằm trong sơ đồ riêng của khung bên lề ở
              p. 79.
            </li>
            <li>
              Fig. 4.9 vẽ phần mã sau nhánh như một lượt hoạt động đầy đủ. Phần chữ một trang sau nói không thể giả định hội tụ
              lại từ Volta trở đi; hãy đọc hình với lưu ý đó.
            </li>
            <li>
              Nhãn luồng của Fig. 4.10 (0, 1, 2, 4, …) không khớp với "threads 1 and 7" của phần chữ. Khóa học dùng số vòng lặp
              được in và nói rõ điều đó (<a href={href('ch04', '4.5')}>4.5</a>).
            </li>
            <li>
              p. 82: "one of 313" warp cho 10,000 phần tử là một trong 314 warp được khởi chạy; 313 warp chứa phần tử. Và hình
              "200 × 150" có 75 hoặc 100 warp phân kỳ trong 1040, tùy con số nào là chiều rộng, chứ không phải 80 như in. Với
              200 hàng, tỉ lệ vượt quá "under 8%" của sách (<a href={href('ch04', '4.5')}>4.5</a>).
            </li>
            <li>p. 81 gọi <code>vecAddKernel</code> của chương 3 bằng một tên khác, <code>addVecKernel</code>.</li>
            <li>
              Occupancy được định nghĩa bằng warp ở p. 85 và bằng luồng ở p. 90. Hai cách khớp nhau khi warp đầy; khóa học dùng
              warp (<a href={href('ch04', '4.7')}>4.7</a>).
            </li>
            <li>p. 86 nói phân tích bộ nhớ chia sẻ nằm "in 5."; ý là chương 5.</li>
            <li>
              "SIMD efficiency" và warp "active" chỉ xuất hiện ở bài tập 1 và không được định nghĩa ở đâu; phần 4.5 đưa ra các
              định nghĩa mà nó dùng.
            </li>
            <li>
              p. 88: vòng lặp thiết bị so sánh một <code>unsigned int</code> với số đếm <code>int</code> và không kiểm tra mã
              trả về nào (<a href={href('ch04', '4.8')}>4.8</a>, lab 4.1).
            </li>
            <li>
              p. 89 mô tả <code>regsPerBlock</code> là thanh ghi của một SM. NVIDIA mô tả nó là giới hạn cho một khối; kho của
              SM là <code>regsPerMultiprocessor</code>. Trường <code>clockRate</code> trong cùng danh sách đã bị gỡ khỏi struct
              trong CUDA 13.0 (<a href={href('ch04', '4.8')}>4.8</a>).
            </li>
            <li>
              p. 88 cho rằng một PC điển hình có hai thiết bị CUDA trở lên vì GPU tích hợp. Không nguồn nào của NVIDIA mà chúng
              tôi tìm được ủng hộ lý do đó, và GPU tích hợp của các hãng khác không phải thiết bị CUDA.
            </li>
          </ul>
        }
      >
        <ul data-list="errata">
          <li>
            p. 69 says the limits on blocks per SM are discussed in "Section 4.6". They are in section 4.7 (
            <a href={href('ch04', '4.2')}>4.2</a> notes this).
          </li>
          <li>
            Fig. 4.4 (p. 72) lacks <code>__global__</code>, and the text cites its lines one higher than they are printed (
            <a href={href('ch04', '4.3')}>4.3</a>).
          </li>
          <li>
            The sidebar on pp. 78–79 points at arrows "in Fig. 4.8"; the arrows are in the sidebar's own diagram on p. 79.
          </li>
          <li>
            Fig. 4.9 draws the code after the branch as one fully active pass. The text a page later says reconvergence
            cannot be assumed from Volta on; read the figure with that.
          </li>
          <li>
            Fig. 4.10's thread labels (0, 1, 2, 4, …) disagree with the text's "threads 1 and 7". The course uses the
            printed trip counts and says so (<a href={href('ch04', '4.5')}>4.5</a>).
          </li>
          <li>
            p. 82: "one of 313" warps for 10,000 elements is one of 314 launched; 313 hold an element. And the "200 × 150"
            picture has 75 or 100 divergent warps of 1040, depending on which number is the width, not the printed 80.
            With 200 rows the share is above the book's "under 8%" (<a href={href('ch04', '4.5')}>4.5</a>).
          </li>
          <li>p. 81 calls chapter 3's <code>vecAddKernel</code> by another name, <code>addVecKernel</code>.</li>
          <li>
            Occupancy is defined with warps on p. 85 and with threads on p. 90. They agree when warps are full; the
            course uses warps (<a href={href('ch04', '4.7')}>4.7</a>).
          </li>
          <li>p. 86 says the shared-memory analysis is "in 5."; chapter 5 is meant.</li>
          <li>
            "SIMD efficiency" and "active" warps appear only in exercise 1 and are defined nowhere; section 4.5 gives the
            definitions it uses.
          </li>
          <li>
            p. 88: the device loop compares an <code>unsigned int</code> with the <code>int</code> count and checks no
            return code (<a href={href('ch04', '4.8')}>4.8</a>, lab 4.1).
          </li>
          <li>
            p. 89 describes <code>regsPerBlock</code> as the registers of an SM. NVIDIA documents it as the limit for one
            block; the SM's pool is <code>regsPerMultiprocessor</code>. The same list's <code>clockRate</code> was removed
            from the struct in CUDA 13.0 (<a href={href('ch04', '4.8')}>4.8</a>).
          </li>
          <li>
            p. 88 expects two or more CUDA devices in a typical PC because of integrated GPUs. No NVIDIA source we found
            supports that reason, and integrated GPUs from other vendors are not CUDA devices.
          </li>
        </ul>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Đây là cách đọc của chúng tôi trên một bản PDF, đối chiếu với tài liệu hiện hành của nhà cung cấp ngày
              2026-11-20 (notes/today/ch04-facts.md). Nếu bản của bạn khác, hãy tin bản của bạn.
            </p>
          }
        >
          <p>
            These are our readings of one PDF, checked against current vendor documentation on 2026-11-20
            (notes/today/ch04-facts.md). If your copy differs, trust your copy.
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Hai điểm về thiết bị ví dụ của sách vẫn còn để ngỏ. Bốn processing block, mỗi block 32, trên mỗi SM không xác nhận
            được theo hướng nào, và các bài học trình bày chúng như con số của sách. Và sách không cho cách chia 132 SM vào 8
            GPC, NVIDIA cũng không công bố.
          </p>
        }
      >
        <p>
          Two points about the book's example device stay open. Its four processing blocks of 32 per SM could not be
          confirmed either way, and the lessons present them as the book's figure. And the book gives no split of its 132
          SMs among 8 GPCs, and NVIDIA does not publish one.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Một số phần của sách được cố ý bỏ qua, để bạn phân biệt được chỗ lược bỏ với chỗ thiếu sót khi đọc song song: danh
            sách trình biên dịch trong khung bên lề về hàm intrinsic (p. 70), phân loại Flynn trong khung bên lề về warp và phần
            cứng SIMD (pp. 78–79), phép so sánh với bưu điện cho khả năng che độ trễ (p. 83) và phép so sánh với khách sạn cho
            việc truy vấn thiết bị (p. 87). Các bài học nêu cùng những điểm đó bằng hình ảnh của riêng mình, và không có gì về
            sau phụ thuộc vào chúng.
          </p>
        }
      >
        <p>
          Some of the book is left out on purpose, so that you can tell an omission from a gap when you read alongside:
          the list of compilers in the sidebar on intrinsic functions (p. 70), Flynn's taxonomy in the sidebar on warps
          and SIMD hardware (pp. 78–79), the post-office analogy for latency tolerance (p. 83) and the hotel analogy for
          querying a device (p. 87). The lessons make the same points with their own pictures, and nothing later depends
          on these.
        </p>
      </Translated>

      <h2><Translated vi="Dự án đến lúc này">The project so far</Translated></h2>
      <p>
        <Translated
          vi={
            <>
              GEMM ngây thơ của lab 3.3 và lần khởi chạy của lab 4.1, được định kích thước từ những gì thiết bị báo cáo, là mốc
              dự án M1 của dự án xuyên suốt: một lớp tuyến tính không có gì về thiết bị được viết cứng trong mã nguồn.{' '}
            </>
          }
        >
          Lab 3.3's naive GEMM and lab 4.1's launch, sized from what the device reports, are milestone M1 of the
          through-line project: a linear layer with nothing about the device written into its source.{' '}
        </Translated>
        <span data-m1={m1.join(',') || 'none'}>
          {backends.length === 0
            ? tr('No lab results are recorded on this machine yet.', 'Chưa có kết quả lab nào được ghi trên máy này.')
            : m1.length
              ? tr(`Your own labs 3.3 and 4.1 both pass on ${m1.map(where).join(' and ')} on this machine.`, `Lab 3.3 và 4.1 của chính bạn đều qua trên ${m1.map(where).join(' và ')} trên máy này.`)
              : tr('Your own labs 3.3 and 4.1 do not both pass on any backend of this machine yet.', 'Lab 3.3 và 4.1 của bạn chưa cùng qua trên backend nào của máy này.')}
        </span>{' '}
        <Translated vi={<>Xem các <a href={href('ch04', 'labs')}>lab</a>.</>}>
          See the <a href={href('ch04', 'labs')}>labs</a>.
        </Translated>
      </p>
      <Translated
        vi={
          <p>
            Chương 5 chuyển sang bộ nhớ trên chip. Thanh ghi và bộ nhớ chia sẻ là nơi kernel giữ dữ liệu nó dùng lại, và chúng
            cũng là hai trong số các ngân sách occupancy của chương này: phần 5.6 thêm bộ nhớ chia sẻ vào phép tính bạn đã làm ở
            đây. Bước đầu tiên của nó là điều mà các lab của chương này liên tục chỉ tới: kernel ngây thơ phải chờ bộ nhớ toàn
            cục, và phân ô (tiling) khiến nó chờ ít hơn. Hãy bắt đầu ở <a href={href('ch05', 'start')}>chương 5</a>.
          </p>
        }
      >
        <p>
          Chapter 5 turns to the memory on the chip. Registers and shared memory are where a kernel keeps the data it
          reuses, and they are also two of this chapter's occupancy budgets: section 5.6 adds shared memory to the
          arithmetic you did here. Its first move is the one this chapter's labs keep pointing at: the naive kernel waits
          on global memory, and tiling makes it wait less. Start at <a href={href('ch05', 'start')}>chapter 5</a>.
        </p>
      </Translated>
    </>
  );
}
