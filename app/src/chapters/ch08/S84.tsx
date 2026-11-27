import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { benchRows, fmt, num, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import {
  basicAI,
  basicCounts,
  blocksBasic,
  blocksCube,
  box,
  boxAI,
  cubeAI,
  denseFlops,
  denseIdealAI,
  denseTaps,
  haloCount,
  haloFraction,
  haloFraction2D,
  idealAIFinite,
  interiorPoints,
  maxCubeSide,
  neverRead,
  points,
  role,
  sharedBytesCube,
  sideForShare,
  stencilFlops,
  stencilIdealAI,
  threadsCube,
  tileElements,
  tileOutputs,
  type Role,
} from '../../sim/stencil3d';
import {
  blockView,
  dims3,
  rowStrip,
  warpView,
  warpsPerBlock,
  type Dims,
  type Lane,
  type WarpConfig,
} from '../../sim/stencilwarp';

// Page 8.4: book §§8.2 to 8.4 (pp. 187–193). The basic kernel and its FLOP/B, why tiling
// buys less for a stencil than for a convolution, the cube-tiled kernel, its budget, and
// one warp's view of its load. Every number comes from src/sim/stencil3d.ts or
// src/sim/stencilwarp.ts; the kernels shown are lab files.

const LAB = 'ch08/lab1_stencil3d';
const BASIC_CU = 'ch08/lab1_stencil3d/solution/kernel.cu';
const BASIC_METAL = 'ch08/lab1_stencil3d/solution/kernel.metal';
const CUBE_CU = 'ch08/shared/tiled_cube.cu';
const CUBE_METAL = 'ch08/shared/tiled_cube.metal';

// Where each identifier of the listings lives. The weights `c` are a kernel parameter in
// CUDA (which the compiler places in constant memory, ch. 7 p. 172) and a `constant`
// buffer in Metal.
const TIERS: Record<string, Tier> = {
  in: 'global', out: 'global', in_s: 'shared', i: 'reg', j: 'reg', k: 'reg', c: 'const',
};

const P7 = points(3, 1); // 7
const IDEAL = stencilIdealAI(P7); // 1.625
const BASIC = basicAI(P7); // 0.40625
const BOOK_THRESHOLD = 20; // the book's example device, ch. 5 p. 94 (facts A8)
// The 32 × 8 × 4 block of p. 193 against the 8 × 8 × 8 cube: loads and outputs per block.
const FLAT_IN = tileElements(box(32, 8, 4)); // 1024
const FLAT_OUT = tileOutputs(box(32, 8, 4), 1); // 360
const CUBE_IN = tileElements(box(8)); // 512
const CUBE_OUT = tileOutputs(box(8), 1); // 216
const r2 = (v: number) => (Math.round(v * 100) / 100).toString();
const pct = (v: number, digits = 1) => (Math.round(v * 100 * 10 ** digits) / 10 ** digits).toString() + '%';
const count = (n: number) => n.toLocaleString('en-US');
const bytesText = (b: number) => (b >= 1024 && b % 1024 === 0 ? `${count(b)} B (${b / 1024} KB)` : `${count(b)} B`); // i18n-ignore: units

// ---------------------------------------------------------------- questions
export const questionsS84: Record<string, Question> = {
  'ch08/8.4/q-reuse': {
    kind: 'choice',
    id: 'ch08/8.4/q-reuse',
    prompt: (
      <Translated
        vi={
          <>
            Một stencil 3D 7 điểm và một phép tích chập (convolution) 3 × 3 × 3 đều chỉ đọc bên trong một hộp 3 × 3 × 3 quanh mỗi đầu
            ra. Vì sao tỉ lệ tốt nhất có thể của stencil lại thấp hơn nhiều đến vậy ({IDEAL} FLOP/B so với {denseIdealAI(3, 1)})?
          </>
        }
      >
        <>
          A 3D 7-point stencil and a 3 × 3 × 3 convolution both read only inside a 3 × 3 × 3 box around each output. Why is
          the stencil's best possible ratio so much lower ({IDEAL} FLOP/B against {denseIdealAI(3, 1)})?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Cùng 8 byte bắt buộc mỗi đầu ra, nhưng 13 phép toán cho stencil so với 54 cho bộ lọc.">{'The same 8 compulsory bytes per output, but 13 operations for the stencil against 54 for the filter.'}</Translated>,
        correct: true,
      },
      { label: <Translated vi="Vùng biên của stencil sâu hơn, nên phần lớn hơn của mỗi ô được nạp mà không bao giờ được tính.">{'The stencil\'s halo is deeper, so more of every tile is loaded and never computed.'}</Translated>, why: <Translated vi="Cả hai đều với ra một điểm theo mọi hướng: cùng bán kính, nên cùng vùng biên.">{'Both reach one point out in every direction: the same radius, so the same halo.'}</Translated> },
      { label: <Translated vi="Stencil không được dùng lại gì từ bộ nhớ chia sẻ, nên mỗi lần đọc trong 7 lần đều đi đến bộ nhớ toàn cục.">{'A stencil gets no reuse from shared memory, so each of its 7 reads goes to global memory.'}</Translated>, why: <Translated vi="Nó có dùng lại các giá trị đã nạp, và trang này xây một kernel chia sẻ chúng. Mức lý tưởng giả định dùng lại hoàn hảo, bất kể bộ nhớ nào cung cấp.">{'It does reuse loaded values, and this page builds a kernel that shares them. The ideal assumes perfect reuse, whatever memory provides it.'}</Translated> },
      { label: <Translated vi="Stencil được tính bằng độ chính xác kép, làm gấp đôi số byte mỗi đầu ra.">{'Stencils are computed in double precision, which doubles the bytes per output.'}</Translated>, why: <Translated vi="Sách so sánh ở 4 byte mỗi giá trị cho cả hai (p. 190).">{'The book makes the comparison at 4 bytes per value for both (p. 190).'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tỉ lệ lý tưởng có cùng mẫu số cho cả hai: dù dùng lại thế nào, mỗi đầu ra cần giá trị đầu vào của riêng nó được nạp một
            lần và kết quả của nó được lưu một lần, 8 byte. Khác biệt nằm ở tử số. Bộ lọc dày đặc làm 27 phép nhân và 27 phép cộng
            mỗi đầu ra; stencil chỉ đọc 7 trong 27 ô của hộp và làm 13 phép toán. Cùng lưu lượng bắt buộc, một phần tư khối lượng
            công việc.
          </p>
        }
      >
        <p>
          The ideal ratio has the same denominator for both: whatever the reuse, each output needs its own input value
          loaded once and its result stored once, 8 bytes. What differs is the numerator. The dense filter does 27 multiplies
          and 27 adds per output; the stencil reads only 7 of the 27 cells of its box and does 13 operations. Same
          compulsory traffic, a quarter of the work.
        </p>
      </Translated>
    ),
    source: 'ch. 8 §8.4, p. 190 (the closed forms are ours)',
    vi: { source: 'ch. 8 §8.4, p. 190 (các dạng đóng là của chúng tôi)' },
  },
  'ch08/8.4/q-cube-max': {
    kind: 'numeric',
    id: 'ch08/8.4/q-cube-max',
    prompt: (
      <Translated
        vi={
          <>
            Một luồng (thread) cho mỗi phần tử của ô, tối đa 1024 luồng mỗi khối (block) và tối đa 64 theo z (các giới hạn của
            CUDA). Cạnh <i>t</i> lớn nhất của khối lập phương mà một khối có thể có là bao nhiêu?
          </>
        }
      >
        <>
          One thread per tile element, at most 1024 threads per block and at most 64 along z (CUDA's limits). What is the
          largest cube side <i>t</i> a block can have?
        </>
      </Translated>
    ),
    answer: maxCubeSide(1024),
    explain: (
      <Translated
        vi={
          <p>
            10 × 10 × 10 = 1000 luồng vừa; 11³ = 1331 thì không. Giới hạn 64 theo z không bao giờ ràng buộc một khối lập phương, vì
            tổng số luồng cạn từ lâu trước đó. Con số 8 của sách là một lựa chọn lũy thừa của hai, không phải giới hạn. Theo chính
            công thức của sách, t = 10 cho{' '}
            {r2(cubeAI(10))} FLOP/B và vùng biên (halo) {pct(haloFraction(10), 0)}, so với {r2(cubeAI(8))} và{' '}
            {pct(haloFraction(8), 0)} ở t = 8. Warp cuối cùng trong 32 warp của nó chứa 8 luồng.
          </p>
        }
      >
        <p>
          10 × 10 × 10 = 1000 threads fit; 11³ = 1331 do not. The z limit of 64 never binds for a cube, because the total
          runs out long before. The book's 8 is a power-of-two choice, not a limit. By the book's own formula, t = 10 gives{' '}
          {r2(cubeAI(10))} FLOP/B and a {pct(haloFraction(10), 0)} halo, against {r2(cubeAI(8))} and{' '}
          {pct(haloFraction(8), 0)} at t = 8. The last of its 32 warps holds 8 threads.
        </p>
      </Translated>
    ),
    source: 'ch. 8 §8.4, p. 192; notes/today/ch08-facts.md A1 (CUDA limits settled in ch05 A5) and B1',
    vi: { source: 'ch. 8 §8.4, p. 192; notes/today/ch08-facts.md A1 (các giới hạn CUDA đã xác định ở ch05 A5) và B1' },
  },
  'ch08/8.4/q-never-read': {
    kind: 'numeric',
    id: 'ch08/8.4/q-never-read',
    prompt: (
      <Translated
        vi={
          <>
            Trong ô đầu vào 8 × 8 × 8 của Hình 8.8, có bao nhiêu trong 512 phần tử được nạp mà không luồng nào từng đọc?
          </>
        }
      >
        <>
          In the 8 × 8 × 8 input tile of Fig. 8.8, how many of the 512 loaded elements does no thread ever read?
        </>
      </Translated>
    ),
    answer: neverRead(8),
    unit: 'elements',
    explain: (
      <Translated
        vi={
          <p>
            Một stencil theo trục chỉ với tới những điểm lệch khỏi tâm theo một tọa độ. Một phần tử của ô nằm trong lớp vùng biên
            theo hai hoặc ba trục thì không bao giờ là láng giềng của ai: 12 cạnh của khối lập phương mỗi cạnh chứa 6 phần tử như vậy
            và 8 góc mỗi góc một, 72 + 8 = 80. Chúng là những góc bị thiếu của Hình 8.7, trong ba chiều. Kernel vẫn nạp chúng, vì
            mọi luồng nạp phần tử của chính nó: 80 trong 512 lần nạp, khoảng 16%, lấy về dữ liệu không ai dùng.
          </p>
        }
      >
        <p>
          An axis stencil only reaches points that are off the centre in one coordinate. A tile element that lies in the
          halo layer along two or three axes is never anyone's neighbour: the 12 edges of the cube hold 6 such elements each
          and the 8 corners one each, 72 + 8 = 80. They are Fig. 8.7's missing corners, in three dimensions. The kernel
          loads them anyway, because every thread loads its own element: 80 of 512 loads, about 16%, fetch data nobody
          uses.
        </p>
      </Translated>
    ),
    source: 'ours, from Fig. 8.8 (p. 191) and the missing corners of Fig. 8.7 (p. 190)',
    vi: { unit: 'phần tử', source: 'của chúng tôi, từ Hình 8.8 (p. 191) và các góc bị thiếu của Hình 8.7 (p. 190)' },
  },
  'ch08/8.4/q-ex1a': {
    kind: 'numeric',
    id: 'ch08/8.4/q-ex1a',
    prompt: (
      <Translated vi={<>Một lượt quét stencil 3D chạy trên lưới 120 × 120 × 120, kể cả các ô biên. Một lượt quét tính bao nhiêu điểm đầu ra?</>}>
        <>A 3D stencil sweep runs on a 120 × 120 × 120 grid, boundary cells included. How many output points does one sweep compute?</>
      </Translated>
    ),
    answer: interiorPoints(120),
    unit: 'points',
    explain: (
      <Translated vi={<p>Chỉ các điểm bên trong được tính: 118 mỗi trục, 118³ = 1,643,032.</p>}>
        <p>Only interior points are computed: 118 per axis, 118³ = 1,643,032.</p>
      </Translated>
    ),
    source: 'ch. 8 exercise 1a, p. 198',
    vi: { unit: 'điểm', source: 'ch. 8 bài tập 1a, p. 198' },
  },
  'ch08/8.4/q-ex1b': {
    kind: 'numeric',
    id: 'ch08/8.4/q-ex1b',
    prompt: (
      <Translated vi={<>Cùng lưới đó. Kernel cơ bản (Hình 8.6) cần bao nhiêu khối 8 × 8 × 8?</>}>
        <>Same grid. How many 8 × 8 × 8 blocks does the basic kernel (Fig. 8.6) need?</>
      </Translated>
    ),
    answer: blocksBasic(120, 8),
    unit: 'blocks',
    explain: (
      <Translated
        vi={
          <p>
            Các luồng của nó phủ mọi điểm lưới, kể cả biên, và <code>if</code> lọc bỏ biên: 120 / 8 = 15 khối mỗi trục, 15³ =
            3,375.
          </p>
        }
      >
        <p>
          Its threads cover every grid point, boundaries included, and the <code>if</code> filters the boundary out:
          120 / 8 = 15 blocks per axis, 15³ = 3,375.
        </p>
      </Translated>
    ),
    source: 'ch. 8 exercise 1b, p. 198',
    vi: { unit: 'khối', source: 'ch. 8 bài tập 1b, p. 198' },
  },
  'ch08/8.4/q-ex1c': {
    kind: 'numeric',
    id: 'ch08/8.4/q-ex1c',
    prompt: (
      <Translated vi={<>Cùng lưới đó. Kernel phân ô (Hình 8.8) cần bao nhiêu khối 8 × 8 × 8?</>}>
        <>Same grid. How many 8 × 8 × 8 blocks does the tiled kernel (Fig. 8.8) need?</>
      </Translated>
    ),
    answer: blocksCube(120, 8),
    unit: 'blocks',
    explain: (
      <Translated
        vi={
          <p>
            Giờ mỗi khối cho ra một ô đầu ra 6 × 6 × 6, nên các khối cách nhau 6: ceil(120 / 6) = 20 mỗi trục, 20³ = 8,000. Hơn
            gấp đôi số khối của kernel cơ bản, cho cùng số đầu ra: các vùng biên chồng lấn là thứ mỗi khối thêm vào phải trả giá để
            nạp.
          </p>
        }
      >
        <p>
          Each block now yields a 6 × 6 × 6 output tile, so blocks are spaced 6 apart: ceil(120 / 6) = 20 per axis, 20³ =
          8,000. More than twice the blocks of the basic kernel, for the same outputs: the overlapping halos are what each
          extra block pays to load.
        </p>
      </Translated>
    ),
    source: 'ch. 8 exercise 1c, p. 198',
    vi: { unit: 'khối', source: 'ch. 8 bài tập 1c, p. 198' },
  },
  'ch08/8.4/q-double': {
    kind: 'choice',
    id: 'ch08/8.4/q-double',
    prompt: (
      <Translated vi={<>Chuyển ô t = 8 từ giá trị 4 byte sang 8 byte. Điều gì thay đổi?</>}>
        <>Switch the t = 8 tile from 4-byte to 8-byte values. What changes?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={`Tỉ lệ giảm một nửa còn ${r2(cubeAI(8, 1, 8))} FLOP/B và ô cần ${count(sharedBytesCube(8, 8))} B bộ nhớ chia sẻ, gấp đôi.`}>{`The ratio halves to ${r2(cubeAI(8, 1, 8))} FLOP/B and the tile needs ${count(sharedBytesCube(8, 8))} B of shared memory, twice as much.`}</Translated>,
        correct: true,
      },
      { label: <Translated vi="FLOP/B không đổi: số phép toán và số phần tử vẫn vậy; chỉ bộ nhớ chia sẻ của ô gấp đôi.">{'Nothing in FLOP/B: the operations and the element counts are the same; only the tile\'s shared memory doubles.'}</Translated>, why: <Translated vi="Số đếm vẫn vậy; số byte mỗi phần tử thì không, và tỉ lệ được tính trên mỗi byte.">{'The counts are the same; the bytes per element are not, and the ratio is per byte.'}</Translated> },
      {
        label: <Translated vi="Tỉ lệ giảm một nửa, nên kernel trở nên bị giới hạn bởi tính toán.">{'The ratio halves, so the kernel becomes compute-bound.'}</Translated>,
        why: <Translated vi="Ít phép toán hơn mỗi byte đẩy kernel về phía bị giới hạn bởi bộ nhớ, chứ không phải xa khỏi nó. Nếu double có lúc nào làm kernel này bị giới hạn bởi tính toán trên phần cứng nào đó, nguyên nhân sẽ là một khối FP64 chậm, không phải tỉ lệ thấp hơn.">{'Fewer operations per byte push a kernel towards the memory-bound side, not away from it. If double ever made this kernel compute-bound on some part, the cause would be a slow FP64 unit, not the lower ratio.'}</Translated>,
      },
      { label: <Translated vi="Ô 8 byte không còn vừa bộ nhớ chia sẻ, nên cạnh phải thu lại còn t = 4.">{'The 8-byte tile no longer fits in shared memory, so the side has to shrink to t = 4.'}</Translated>, why: <Translated vi="4,096 B nằm gọn trong mức một khối được phép dùng.">{'4,096 B is well within what a block may use.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mọi số byte đều gấp đôi, nên mọi con số FLOP/B giảm một nửa và mọi con số bộ nhớ chia sẻ gấp đôi. Phép tính double có
            chậm hơn hay không còn tùy phần cứng: bảng của NVIDIA cho tỉ lệ thông lượng FP32 so với FP64 là 2:1 trên compute
            capability 8.0, 9.0 và 10.0 và lên đến 64:1 trên các loại khác. Metal hoàn toàn không làm được: MSL không có{' '}
            <code>double</code>.
          </p>
        }
      >
        <p>
          Every byte count doubles, so every FLOP/B figure halves and every shared-memory figure doubles. Whether double
          arithmetic is also slower depends on the part: NVIDIA's table gives FP32 to FP64 throughput of 2:1 on compute
          capabilities 8.0, 9.0 and 10.0 and up to 64:1 on others. Metal cannot do it at all: MSL has no{' '}
          <code>double</code>.
        </p>
      </Translated>
    ),
    source: 'ch. 8 §8.1, pp. 184–185; notes/today/ch08-facts.md A7 (no double in MSL: ch03 C6, cited there)',
    vi: { source: 'ch. 8 §8.1, pp. 184–185; notes/today/ch08-facts.md A7 (MSL không có double: ch03 C6, được dẫn ở đó)' },
  },
  'ch08/8.4/q-transactions': {
    kind: 'numeric',
    id: 'ch08/8.4/q-transactions',
    prompt: (
      <Translated
        vi={
          <>
            Một khối 8 × 8 × 8 bên trong của Hình 8.8, lần nạp của warp 0, dữ liệu <code>float</code>, một mảng được căn chỉnh và N
            là bội số của 8. Theo mô hình giao dịch 32 byte của NVIDIA, lần nạp này cần bao nhiêu giao dịch?
          </>
        }
      >
        <>
          An interior 8 × 8 × 8 block of Fig. 8.8, warp 0's load, <code>float</code> data, an aligned array and N a
          multiple of 8. Under NVIDIA's 32-byte transaction model, how many transactions does the load take?
        </>
      </Translated>
    ),
    answer: 8,
    unit: 'transactions',
    vi: { unit: 'giao dịch' },
    explain: (
      <Translated
        vi={
          <p>
            Warp gồm bốn hàng 8 số float, cách nhau N số float. Mỗi hàng bắt đầu ở <code>6·blockIdx.x − 1</code>, một chỉ số lẻ,
            nên 32 byte của nó nằm vắt qua ranh giới một đoạn: hai đoạn mỗi hàng, tổng cộng tám. Dùng 128 byte trên 256 byte lấy về,
            50%. Sách nói "không được hợp nhất"; mô hình của NVIDIA nói hiệu suất một nửa (phép tính của chúng tôi).
          </p>
        }
      >
        <p>
          The warp is four rows of 8 floats, N floats apart. Each row starts at <code>6·blockIdx.x − 1</code>, an odd index,
          so its 32 bytes straddle a segment boundary: two segments per row, eight in all. 128 bytes used of 256 fetched,
          50%. The book says "not coalesced"; NVIDIA's model says half efficient (our arithmetic).
        </p>
      </Translated>
    ),
    source: 'ch. 8 §8.4, p. 193; notes/today/ch08-facts.md A2', // i18n-ignore: citation only
  },
  'ch08/8.4/q-x32': {
    kind: 'choice',
    id: 'ch08/8.4/q-x32',
    prompt: (
      <Translated vi={<>Chạy Hình 8.8 với khối 32 × 8 × 4, như sách gợi ý. Lần nạp của mỗi warp giờ có được hợp nhất hoàn toàn không, và FLOP/B thay đổi ra sao?</>}>
        <>Run Fig. 8.8 with 32 × 8 × 4 blocks, as the book suggests. Is each warp's load now perfectly coalesced, and what happens to FLOP/B?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={`Không: mỗi hàng bắt đầu ở một chỉ số lẻ và trải qua năm đoạn (80%). FLOP/B giảm còn ${r2(boxAI(box(32, 8, 4), 1))}: ${count(FLAT_OUT)} đầu ra cho ${count(FLAT_IN)} lần nạp, so với ${CUBE_OUT} trên ${CUBE_IN} của khối lập phương.`}>{`No: each row starts on an odd index and spans five segments (80%). FLOP/B falls to ${r2(boxAI(box(32, 8, 4), 1))}: ${count(FLAT_OUT)} outputs for ${count(FLAT_IN)} loads, against the cube's ${CUBE_OUT} of ${CUBE_IN}.`}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Có, 100%: mỗi warp giờ đọc một hàng 32 số float, một dải 128 byte duy nhất được căn chỉnh, và FLOP/B tăng vì khối có số luồng gấp đôi khối lập phương.">{'Yes, 100%: each warp now reads one row of 32 floats, a single aligned 128-byte run, and FLOP/B rises because the block has twice the threads of the cube.'}</Translated>,
        why: <Translated vi={`Độ lệch −1 làm mọi hàng bắt đầu ở một chỉ số lẻ. Và một khối dẹt biến một phần nhỏ hơn trong ${count(FLAT_IN)} lần nạp của nó thành đầu ra (${pct(FLAT_OUT / FLAT_IN, 0)}) so với khối lập phương với ${CUBE_IN} (${pct(CUBE_OUT / CUBE_IN, 0)}).`}>{`The −1 offset makes every row start on an odd index. And a flat block turns a smaller share of its ${count(FLAT_IN)} loads into outputs (${pct(FLAT_OUT / FLAT_IN, 0)}) than the cube does with ${CUBE_IN} (${pct(CUBE_OUT / CUBE_IN, 0)}).`}</Translated>,
      },
      { label: <Translated vi="Có, 100%; FLOP/B không đổi, vì mỗi luồng vẫn nạp một phần tử của ô.">{'Yes, 100%; FLOP/B is unchanged, since each thread still loads one element of the tile.'}</Translated>, why: <Translated vi="Hãy đếm các đoạn từ một điểm bắt đầu lẻ: 128 byte từ một điểm bắt đầu không căn chỉnh chạm năm đoạn 32 byte. Và FLOP/B giảm: khối dẹt biến một phần nhỏ hơn trong các lần nạp của nó thành đầu ra.">{'Count the segments from an odd start: 128 bytes from an unaligned start touch five 32-byte segments. And FLOP/B falls: the flat block turns a smaller share of its loads into outputs.'}</Translated> },
      { label: <Translated vi={`Không, vẫn 50%, như khối lập phương; FLOP/B tăng lên ${r2(cubeAI(10))}.`}>{`No, still 50%, as for the cube; FLOP/B rises to ${r2(cubeAI(10))}.`}</Translated>, why: <Translated vi="Một hàng mỗi warp chính là cải thiện; và con số FLOP/B đó thuộc về khối lập phương 10 × 10 × 10.">{'One row per warp is the improvement; and that FLOP/B belongs to the 10 × 10 × 10 cube.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Các lần nạp tốt lên còn việc dùng lại thì kém đi. Một khối rộng 32 đặt một hàng vào mỗi warp, nên warp đọc một dải liền
            kề thay vì bốn, nhưng dải đó bắt đầu trước ranh giới đoạn một phần tử, nên nó chạm năm đoạn cho lượng dữ liệu của bốn
            đoạn. Trong khi đó ô đầu ra của khối dẹt, 30 × 6 × 2 ={' '}
            {count(FLAT_OUT)} điểm, lớn hơn {CUBE_OUT} của khối lập phương nhưng là phần nhỏ hơn trong những gì khối nạp:{' '}
            {pct(FLAT_OUT / FLAT_IN, 0)} của {count(FLAT_IN)} so với {pct(CUBE_OUT / CUBE_IN, 0)} của {CUBE_IN}. Tỉ lệ đi theo phần
            đó, từ {r2(cubeAI(8))} xuống {r2(boxAI(box(32, 8, 4), 1))}. Sách chỉ nói tỉ lệ vẫn thấp (p. 193); con số 80% là phép tính
            của chúng tôi trên mô hình của NVIDIA.
          </p>
        }
      >
        <p>
          The loads get better and the reuse gets worse. A 32-wide block puts one row in each warp, so the warp reads one
          contiguous run instead of four, but the run starts one element before a segment boundary, so it touches five
          segments for four segments' worth of data. Meanwhile the flat block's output tile, 30 × 6 × 2 ={' '}
          {count(FLAT_OUT)} points, is larger than the cube's {CUBE_OUT} but a smaller share of what the block loads:{' '}
          {pct(FLAT_OUT / FLAT_IN, 0)} of {count(FLAT_IN)} against {pct(CUBE_OUT / CUBE_IN, 0)} of {CUBE_IN}. The
          ratio follows the share, from {r2(cubeAI(8))} to {r2(boxAI(box(32, 8, 4), 1))}. The book says only that the
          ratio stays low (p. 193); the 80% is our arithmetic on NVIDIA's model.
        </p>
      </Translated>
    ),
    source: 'ch. 8 §8.4, p. 193; notes/today/ch08-facts.md A2', // i18n-ignore: citation only
  },
};

// ---------------------------------------------------------------- code from the lab
/** The source of one function in a file, from its signature line to its closing brace, with the comment above it. */
function functionText(src: string | null, signature: RegExp): string | null {
  if (!src) return null;
  const lines = src.split('\n');
  const start = lines.findIndex((l) => signature.test(l));
  if (start < 0) return null;
  let first = start;
  while (first > 0 && /^\s*\/\//.test(lines[first - 1])) first--;
  let depth = 0;
  let opened = false;
  for (let i = start; i < lines.length; i++) {
    for (const ch of lines[i]) {
      if (ch === '{') {
        depth++;
        opened = true;
      } else if (ch === '}') depth--;
    }
    if (opened && depth === 0) return lines.slice(first, i + 1).join('\n');
  }
  return null;
}

/** What the cube file calls its two tile sizes, read from its own #defines. */
function cubeDefines(cu: string): { inName: string; outName: string; side: number } | null {
  const a = /#define\s+(\w*IN_TILE_DIM)\s+(\d+)/.exec(cu);
  const b = /#define\s+(\w*OUT_TILE_DIM)\b/.exec(cu);
  return a && b ? { inName: a[1], outName: b[1], side: Number(a[2]) } : null;
}

function CubeCaption({ cu }: { cu: string }) {
  const d = cubeDefines(cu);
  return (
    <Translated
      vi={
        <>
          Kernel phân ô lập phương, được nạp từ <code>labs/ch08/shared/</code>; lab 8.1 cung cấp nó và kiểm thử nó cùng các
          kernel của bạn.{' '}
          {d ? (
            <>
              Nó định nghĩa <code>{d.inName}</code> là {d.side} và <code>{d.outName}</code> là {d.side - 2}, hai kích thước mà
              listing in trong sách dùng mà không định nghĩa
              {d.inName !== 'IN_TILE_DIM' ? <> (sách gọi chúng là <code>IN_TILE_DIM</code> và <code>OUT_TILE_DIM</code>)</> : null}
              , và ghép trọng số với láng giềng như kernel cơ bản.
            </>
          ) : (
            <>Nó ghép trọng số với láng giềng như kernel cơ bản.</>
          )}
        </>
      }
    >
      <>
        The cube-tiled kernel, loaded from <code>labs/ch08/shared/</code>; lab 8.1 provides it and tests it alongside your
        kernels.{' '}
        {d ? (
          <>
            It defines <code>{d.inName}</code> as {d.side} and <code>{d.outName}</code> as {d.side - 2}, the two sizes the
            printed listing uses without defining
            {d.inName !== 'IN_TILE_DIM' ? <> (it calls them <code>IN_TILE_DIM</code> and <code>OUT_TILE_DIM</code>)</> : null}
            , and pairs the weights with neighbours as the basic kernel does.
          </>
        ) : (
          <>It pairs the weights with neighbours as the basic kernel does.</>
        )}
      </>
    </Translated>
  );
}

/** Status of the lab's reference solution, in the labs page's words (LabKit's solutionTabStatus). */
function useLabStatus() {
  const { reports } = useReports();
  return { cu: solutionTabStatus(reports, LAB, 'kernel.cu'), metal: solutionTabStatus(reports, LAB, 'kernel.metal') };
}

function LabListing({
  cuPath,
  metalPath,
  cuSig,
  metalSig,
  caption,
}: {
  cuPath: string;
  metalPath: string;
  cuSig?: RegExp;
  metalSig?: RegExp;
  caption: ReactNode | ((cu: string) => ReactNode);
}) {
  const cu = useLabFile(cuPath);
  const metal = useLabFile(metalPath);
  const status = useLabStatus();
  if (cu === null || metal === null) return (
      <p className="muted">
        <Translated vi="Đang nạp các tệp của lab.">Loading the lab's files.</Translated>
      </p>
    );
  const cuText = (cuSig && functionText(cu, cuSig)) || cu;
  const metalText = (metalSig && functionText(metal, metalSig)) || metal;
  return (
    <Code
      tabs={[
        { label: tr('CUDA and HIP', 'CUDA và HIP'), code: cuText, tiers: TIERS, status: status.cu },
        { label: 'Metal', code: metalText, tiers: TIERS, status: status.metal },
      ]}
      caption={typeof caption === 'function' ? caption(cu) : caption}
    />
  );
}

// ---------------------------------------------------------------- the basic kernel's count
function BasicReveal({ v }: { v: number }) {
  const c = basicCounts(P7);
  const near = (x: number, tol = 0.015) => Math.abs(v - x) <= tol;
  let verdict: ReactNode;
  if (near(BASIC)) verdict = tr('That is the count.', 'Đúng con số đó.');
  else if (near(13 / 28)) verdict = tr('You left out the store. Each thread writes one 4-byte result, and that is traffic too: 28 bytes of loads plus 4 of store.', 'Bạn đã bỏ sót lần lưu. Mỗi luồng ghi một kết quả 4 byte, và đó cũng là lưu lượng: 28 byte nạp cộng 4 byte lưu.');
  else if (near(13 / 8)) verdict = tr('That is 13 FLOP per access: you counted the 8 accesses, not their 32 bytes. (It is also the ideal ratio, the best any kernel could do, which is the next paragraph.)', 'Đó là 13 FLOP mỗi lần truy cập: bạn đã đếm 8 lần truy cập, chứ không phải 32 byte của chúng. (Đó cũng là tỉ lệ lý tưởng, mức tốt nhất mà bất kỳ kernel nào có thể đạt, chính là đoạn tiếp theo.)');
  else if (near(13 / 7, 0.03)) verdict = tr('You divided by the 7 loads as if each were one byte, and left out the store.', 'Bạn đã chia cho 7 lần nạp như thể mỗi lần là một byte, và bỏ sót lần lưu.');
  else if (near(14 / 32)) verdict = tr('Close: the book counts 6 adds, not 7. Summing 7 products takes 6 additions.', 'Gần đúng: sách đếm 6 phép cộng, không phải 7. Cộng 7 tích cần 6 phép cộng.');
  else if (v > 1) verdict = tr('That is above what any kernel could reach for this stencil (the ideal below is 1.625). Recount the bytes.', 'Con số đó cao hơn mức bất kỳ kernel nào có thể đạt cho stencil này (mức lý tưởng bên dưới là 1.625). Hãy đếm lại số byte.');
  else verdict = tr('Recount with the listing in front of you: every term is a multiply, every + is an add, every in[...] is a 4-byte load.', 'Hãy đếm lại với listing trước mặt: mỗi số hạng là một phép nhân, mỗi dấu + là một phép cộng, mỗi in[...] là một lần nạp 4 byte.');
  return (
    <>
      <Translated
        vi={
          <p>
            <b>
              {c.flop} FLOP trên {c.bytes} byte: {BASIC} FLOP/B, mà sách in là 0.41.
            </b>{' '}
            {verdict}
          </p>
        }
      >
        <p>
          <b>
            {c.flop} FLOP over {c.bytes} bytes: {BASIC} FLOP/B, which the book prints as 0.41.
          </b>{' '}
          {verdict}
        </p>
      </Translated>
      <table className="data" style={{ margin: '0.4rem 0 0.6rem', maxWidth: '30rem' }}>
        <tbody>
          <tr>
            <td>
              <Translated vi="Phép toán">Operations</Translated>
            </td>
            <td className="num">
              <Translated vi={<>7 phép nhân + 6 phép cộng = {c.flop}</>}>7 multiplies + 6 adds = {c.flop}</Translated>
            </td>
          </tr>
          <tr><td><Translated vi="Lần nạp">Loads</Translated></td><td className="num">{c.loads} × 4 B = {c.loads * 4} B</td></tr>
          <tr><td><Translated vi="Lần lưu">Store</Translated></td><td className="num">{c.stores} × 4 B = 4 B</td></tr>
          <tr><td><Translated vi="Tỉ lệ">Ratio</Translated></td><td className="num">{c.flop} / {c.bytes} = {BASIC}</td></tr>
        </tbody>
      </table>
    </>
  );
}

const DEVICE_BACKENDS = ['cuda', 'hip', 'metal'] as const;

function YourThreshold() {
  const { reports, loading } = useReports();
  if (loading) return null;
  const found = DEVICE_BACKENDS.filter((b) => reports[b]).flatMap((b) => {
    const row = benchRows(reports[b], 'ch05/lab1_roofline').find((r) => r.name === 'roofline');
    const th = num(row?.threshold_flop_per_byte);
    return th ? [{ b, device: String(reports[b].device.name ?? b), th }] : [];
  });
  if (!found.length)
    return (
      <p data-threshold="none">
        <Translated
          vi={
            <>
              Ngưỡng của chính thiết bị của bạn đến từ lab 5.1, mà máy này chưa ghi lại. Hãy chạy{' '}
              <code>python3 labs/pmpp.py run 5.1 --solution --bench</code> và <code>./verify.sh</code>, và đoạn này sẽ đặt stencil
              cạnh ngưỡng đó.
            </>
          }
        >
          Your own device's threshold comes from lab 5.1, which has not recorded one on this machine yet. Run{' '}
          <code>python3 labs/pmpp.py run 5.1 --solution --bench</code> and <code>./verify.sh</code>, and this paragraph will
          place the stencil against it.
        </Translated>
      </p>
    );
  return (
    <>
      {found.map((f) => (
        <p key={f.b} data-threshold={f.b}>
          <Translated
            vi={
              <>
                Trên <b>{f.device}</b> ({f.b}), lab 5.1 đo được ngưỡng <b>{fmt(f.th)} FLOP/B</b>. Con số {r2(BASIC)} của kernel cơ
                bản {f.th > BASIC ? `thấp hơn ngưỡng ${fmt(f.th / BASIC)} lần` : 'cao hơn ngưỡng'}, và ngay cả mức lý tưởng {IDEAL}{' '}
                cũng {f.th > IDEAL ? `thấp hơn ngưỡng ${fmt(f.th / IDEAL)} lần` : 'cao hơn ngưỡng'}. Đó là một phép đo từ máy của bạn,
                thực hiện bằng hai kernel thăm dò đơn giản, không phải con số của nhà cung cấp.
              </>
            }
          >
            On <b>{f.device}</b> ({f.b}), lab 5.1 measured a threshold of <b>{fmt(f.th)} FLOP/B</b>. The basic kernel's{' '}
            {r2(BASIC)} is {f.th > BASIC ? `${fmt(f.th / BASIC)} times below it` : 'above it'}, and even the ideal {IDEAL}{' '}
            is {f.th > IDEAL ? `${fmt(f.th / IDEAL)} times below it` : 'above it'}. That is a measurement from your machine,
            made with two simple probe kernels, not a vendor figure.
          </Translated>
        </p>
      ))}
    </>
  );
}

// ---------------------------------------------------------------- explorer: stencil or dense filter
const BOOK_P190: { d: number; r: number; label: string }[] = [
  { d: 2, r: 1, get label() { return tr('2D, order 1', '2D, bậc 1'); } },
  { d: 2, r: 2, get label() { return tr('2D, order 2', '2D, bậc 2'); } },
  { d: 2, r: 3, get label() { return tr('2D, order 3', '2D, bậc 3'); } },
  { d: 3, r: 3, get label() { return tr('3D, order 3', '3D, bậc 3'); } },
];

function StencilBox({ d, r }: { d: number; r: number }) {
  const m = 2 * r + 1;
  const cs = m <= 3 ? 30 : m <= 5 ? 22 : 16;
  const slices = d === 3 ? m : 1;
  const perRow = d === 3 ? Math.min(slices, 4) : 1;
  const gap = 18;
  const sw = m * cs;
  const sh = (d === 1 ? 1 : m) * cs;
  const rows = Math.ceil(slices / perRow);
  const W = perRow * sw + (perRow - 1) * gap + 2;
  const H = rows * (sh + 18) + 2;
  const cells: ReactNode[] = [];
  for (let s = 0; s < slices; s++) {
    const ox = 1 + (s % perRow) * (sw + gap);
    const oy = 1 + Math.floor(s / perRow) * (sh + 18) + 16;
    const dz = s - r;
    if (d === 3)
      cells.push(
        <text key={'l' + s} x={ox + sw / 2} y={oy - 5} fontSize="11" textAnchor="middle">
          {dz === 0 ? 'z' : dz > 0 ? `z + ${dz}` : `z − ${-dz}`}
        </text>,
      );
    for (let y = 0; y < (d === 1 ? 1 : m); y++)
      for (let x = 0; x < m; x++) {
        const dx = x - r;
        const dy = d === 1 ? 0 : y - r;
        const off = (dx !== 0 ? 1 : 0) + (dy !== 0 ? 1 : 0) + (dz !== 0 && d === 3 ? 1 : 0);
        const onAxis = off <= 1;
        const centre = off === 0;
        cells.push(
          <g key={`${s}-${x}-${y}`} data-cell={onAxis ? 'stencil' : 'box'}>
            <rect x={ox + x * cs} y={oy + y * cs} width={cs} height={cs} fill="var(--cell)" stroke="var(--rule)" />
            {onAxis ? (
              <circle cx={ox + x * cs + cs / 2} cy={oy + y * cs + cs / 2} r={centre ? cs * 0.34 : cs * 0.24} fill={centre ? 'none' : 'var(--ink)'} stroke="var(--ink)" strokeWidth={centre ? 2 : 0} />
            ) : null}
          </g>,
        );
      }
  }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr(`A ${d}D stencil of order ${r} inside its ${m}-wide box`, `Một stencil ${d}D bậc ${r} bên trong hộp rộng ${m} của nó`)} style={{ width: '100%', maxWidth: Math.min(W, 640), display: 'block' }}>
      {cells}
    </svg>
  );
}

function ShapeExplorer() {
  const [d, setD] = useState(3);
  const [r, setR] = useState(1);
  const p = points(d, r);
  const taps = denseTaps(d, r);
  return (
    <Bench
      title={tr('Stencil or dense filter', 'Stencil hay bộ lọc dày đặc')}
      task={tr('Find the dimension and order at which the stencil reads the smallest share of its box.', 'Tìm số chiều và bậc mà tại đó stencil đọc phần nhỏ nhất của hộp.')}
      foot={
        <Translated
          vi={
            <>
              Mỗi ô vuông là một tap của một bộ lọc dày đặc có cùng phạm vi (chương 7); một chấm là một điểm stencil đọc, vòng tròn là
              tâm của nó. Một hộp 3D được vẽ thành các lát z của nó. Các tỉ lệ giả định giá trị 4 byte và dùng lại hoàn hảo: một lần
              nạp và một lần lưu mỗi đầu ra. Hai hàng dùng hai cách đếm phép toán của sách: chương 7 đếm một phép nhân và một phép cộng
              mỗi tap, chương 8 đếm p phép nhân và p − 1 phép cộng. Đó là lý do các hàng 1D khác nhau đôi chút dù hình dạng giống nhau.
            </>
          }
        >
          <>
            Each square is one tap of a dense filter of the same extent (chapter 7); a dot is a point the stencil reads, the
            ring its centre. A 3D box is drawn as its z-slices. Ratios assume 4-byte values and perfect reuse: one load and
            one store per output. The two rows use the book's two operation counts: chapter 7 counts a multiply and an add
            per tap, chapter 8 counts p multiplies and p − 1 adds. That is why the 1D rows differ slightly although the
            shapes are the same.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
          <Translated vi="Số chiều">Dimension</Translated>
          <span className="seg" data-control="dim">
            {[1, 2, 3].map((v) => (
              <button key={v} aria-pressed={d === v} onClick={() => setD(v)}>{v}D</button>
            ))}
          </span>
        </div>
        <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
          <Translated vi="Bậc r (số điểm mỗi bên)">Order r (points on each side)</Translated>
          <span className="seg" data-control="order">
            {[1, 2, 3].map((v) => (
              <button key={v} aria-pressed={r === v} onClick={() => setR(v)}>{v}</button>
            ))}
          </span>
        </div>
      </div>
      <div className="cols plot">
        <div className="panel">
          <StencilBox d={d} r={r} />
        </div>
        <div className="panel" data-readout="shape">
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th></th>
                <th className="num">
                  <Translated vi="số điểm đọc">points read</Translated>
                </th>
                <th className="num">
                  <Translated vi="FLOP mỗi đầu ra">FLOP per output</Translated>
                </th>
                <th className="num">
                  <Translated vi="FLOP/B lý tưởng">ideal FLOP/B</Translated>
                </th>
              </tr>
            </thead>
            <tbody>
              <tr data-row="stencil">
                <td>
                  <Translated vi="stencil">stencil</Translated>
                </td>
                <td className="num" data-out="points">{p}</td>
                <td className="num" data-out="flop">{stencilFlops(p)}</td>
                <td className="num" data-out="ai">{stencilIdealAI(p)}</td>
              </tr>
              <tr data-row="dense">
                <td>
                  <Translated vi="bộ lọc dày đặc">dense filter</Translated>
                </td>
                <td className="num" data-out="points">{taps}</td>
                <td className="num" data-out="flop">{denseFlops(taps)}</td>
                <td className="num" data-out="ai">{denseIdealAI(d, r)}</td>
              </tr>
            </tbody>
          </table>
          <p style={{ margin: '0.7rem 0 0' }} data-out="share">
            <Translated
              vi={
                <>
                  Stencil đọc <b>{p} trên {taps}</b> ô của hộp ({pct(p / taps, 0)}).{' '}
                  {taps - p > 0 ? <>{taps - p} ô còn lại, lệch tâm theo hai tọa độ trở lên, không bao giờ được đọc.</> : <>Trong 1D, hộp chính là stencil.</>}
                </>
              }
            >
              The stencil reads <b>{p} of the {taps}</b> cells of its box ({pct(p / taps, 0)}).{' '}
              {taps - p > 0 ? <>The other {taps - p}, off-centre in two or more coordinates, are never read.</> : <>In 1D the box is the stencil.</>}
            </Translated>
          </p>
        </div>
      </div>
      <p className="ui muted" style={{ margin: '1rem 0 0.3rem' }}>
        <Translated vi="Bảng của sách (p. 190), tính từ cùng mô hình:">The book's table (p. 190), computed from the same model:</Translated>
      </p>
      <table className="data" data-table="p190" style={{ margin: 0 }}>
        <thead>
          <tr>
            <th>
              <Translated vi="Trường hợp">Case</Translated>
            </th>
            <th className="num">
              <Translated vi="số điểm stencil">stencil points</Translated>
            </th>
            <th className="num">
              <Translated vi="stencil, FLOP/B">stencil, FLOP/B</Translated>
            </th>
            <th className="num">
              <Translated vi="bộ lọc dày đặc">dense filter</Translated>
            </th>
            <th className="num">
              <Translated vi="dày đặc, FLOP/B">dense, FLOP/B</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          {BOOK_P190.map((row) => {
            const m = 2 * row.r + 1;
            const here = row.d === d && row.r === r;
            return (
              <tr key={row.label} style={here ? { fontWeight: 620 } : undefined} aria-current={here ? 'true' : undefined}>
                <td className="nowrap">{row.label}</td>
                <td className="num">{points(row.d, row.r)}</td>
                <td className="num">{stencilIdealAI(points(row.d, row.r))}</td>
                <td className="num nowrap">{row.d === 2 ? `${m} × ${m}` : `${m} × ${m} × ${m}`}</td>
                <td className="num">{denseIdealAI(row.d, row.r)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: a cube-shaped tile
interface Limit {
  key: string;
  label: string;
  threads: number;
  /** Shared or threadgroup memory a block may use, bytes, where known. */
  shared?: number;
  sharedName?: string;
  note: string;
}
const BOOK_LIMIT: Limit = {
  key: 'book',
  get label() {
    return tr("The book's limit: 1024 threads per block", 'Giới hạn của sách: 1024 luồng mỗi khối');
  },
  threads: 1024,
  shared: 48 * 1024,
  get sharedName() {
    return tr("CUDA's static shared memory per block without opt-in", 'bộ nhớ chia sẻ tĩnh mỗi khối của CUDA khi không opt-in');
  },
  get note() {
    return tr(
      'The book quotes 1024 threads per block (p. 192); NVIDIA documents the same on every compute capability, and 48 KB of static shared memory per block without opt-in (facts A1, A6).',
      'Sách nêu 1024 luồng mỗi khối (p. 192); NVIDIA ghi nhận cùng con số trên mọi compute capability, và 48 KB bộ nhớ chia sẻ tĩnh mỗi khối khi không opt-in (facts A1, A6).',
    );
  },
};

const positive = (v: unknown): number => {
  const n = Number(v);
  return isFinite(n) && n > 0 ? Math.floor(n) : 0;
};

/**
 * A Metal pipeline's thread cap from lab 8.1's "limits" rows, if the lab recorded one. The
 * cap belongs to a compiled pipeline (facts C1). A row for the cube kernel itself
 * (kernel: "cube") is used when present; otherwise the lowest cap the lab read from its
 * register-tiled pipelines, which is another kernel's and is labelled so.
 */
function metalCap(reports: Reports): { threads: number; cube: boolean } | null {
  const run = reports['metal'];
  for (const variant of ['solution', 'work']) {
    const rows = benchRows(run, LAB, variant).filter((r) => r.name === 'limits' && positive(r.max_threads));
    if (!rows.length) continue;
    const cube = rows.find((r) => r.kernel === 'cube');
    if (cube) return { threads: positive(cube.max_threads), cube: true };
    return { threads: Math.min(...rows.map((r) => positive(r.max_threads))), cube: false };
  }
  return null;
}

function reportedLimits(reports: Reports): Limit[] {
  const out: Limit[] = [];
  for (const b of DEVICE_BACKENDS) {
    const run = reports[b];
    if (!run) continue;
    const d = run.device;
    const name = String(d.name ?? b);
    if (b === 'metal') {
      const cap = metalCap(reports);
      if (!cap) continue;
      out.push({
        key: b,
        get label() {
          return tr(`Yours: ${name} (metal), a pipeline's cap`, `Của bạn: ${name} (metal), giới hạn của một pipeline`);
        },
        threads: cap.threads,
        shared: positive(d.maxThreadgroupMemoryLength) || undefined,
        sharedName: 'maxThreadgroupMemoryLength',
        get note() {
          return cap.cube
            ? tr(
                `The thread cap lab 8.1 read from the cube kernel's compiled pipeline (maxTotalThreadsPerThreadgroup), and the device's maxThreadgroupMemoryLength, recorded on ${run.generated.slice(0, 10)}.`,
                `Giới hạn số luồng mà lab 8.1 đọc từ pipeline đã biên dịch của kernel lập phương (maxTotalThreadsPerThreadgroup), và maxThreadgroupMemoryLength của thiết bị, ghi lại ngày ${run.generated.slice(0, 10)}.`,
              )
            : tr(
                `The lowest thread cap (maxTotalThreadsPerThreadgroup) lab 8.1 read from its register-tiled kernel's pipelines, and the device's maxThreadgroupMemoryLength, recorded on ${run.generated.slice(0, 10)}. The cap belongs to a compiled kernel, so the cube kernel's own may differ.`,
                `Giới hạn số luồng thấp nhất (maxTotalThreadsPerThreadgroup) mà lab 8.1 đọc từ các pipeline của kernel phân ô thanh ghi, và maxThreadgroupMemoryLength của thiết bị, ghi lại ngày ${run.generated.slice(0, 10)}. Giới hạn thuộc về một kernel đã biên dịch, nên giới hạn của chính kernel lập phương có thể khác.`,
              );
        },
      });
    } else {
      const threads = positive(d.maxThreadsPerBlock);
      if (!threads) continue;
      out.push({
        key: b,
        get label() {
          return tr(`Yours: ${name} (${b})`, `Của bạn: ${name} (${b})`);
        },
        threads,
        shared: positive(d.sharedMemPerBlock) || undefined,
        sharedName: 'sharedMemPerBlock',
        get note() {
          return tr(
            `maxThreadsPerBlock and sharedMemPerBlock as this device reported them when ./verify.sh ran on ${run.generated.slice(0, 10)}.`,
            `maxThreadsPerBlock và sharedMemPerBlock như thiết bị này báo cáo khi ./verify.sh chạy ngày ${run.generated.slice(0, 10)}.`,
          );
        },
      });
    }
  }
  return out;
}

const ROLE_FILL: Record<Role, string> = { output: 'var(--shared)', halo: 'var(--shared-wash)', never: 'url(#ch08-never)' };
const ROLE_NAME: Record<Role, string> = {
  get output() {
    return tr('output (computed here)', 'đầu ra (tính ở đây)');
  },
  get halo() {
    return tr('halo, read by a neighbour', 'vùng biên, do một láng giềng đọc');
  },
  get never() {
    return tr('loaded, never read', 'được nạp, không bao giờ được đọc');
  },
};

function NeverPattern() {
  return (
    <defs>
      <pattern id="ch08-never" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="5" height="5" fill="var(--cell)" />
        <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.6" />
      </pattern>
    </defs>
  );
}

function SliceView({ t, z }: { t: number; z: number }) {
  const S = 252;
  const cs = S / t;
  const cells: ReactNode[] = [];
  for (let y = 0; y < t; y++)
    for (let x = 0; x < t; x++) {
      const ro = role(t, 1, x, y, z);
      cells.push(
        <rect
          key={x + ',' + y}
          data-role={ro}
          x={x * cs}
          y={y * cs}
          width={cs}
          height={cs}
          fill={ROLE_FILL[ro]}
          stroke={ro === 'halo' ? 'var(--shared)' : 'var(--page)'}
          strokeOpacity={ro === 'halo' ? 0.45 : 1}
          strokeWidth={t > 20 ? 0.4 : 1}
        />,
      );
    }
  return (
    <svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label={tr(`Slice z = ${z} of a ${t}-wide tile`, `Lát z = ${z} của một ô rộng ${t}`)} style={{ width: '100%', maxWidth: S, display: 'block' }}>
      <NeverPattern />
      {cells}
    </svg>
  );
}

// Scaled to a phone the chart's labels would be a few pixels tall, so below this width it
// keeps its size and scrolls sideways in its frame (as a scene's stage does).
const PLOT_MIN = 520;

function AICurve({ t, bytes, maxSide }: { t: number; bytes: number; maxSide: number }) {
  const W = 560;
  const H = 230;
  const pad = { l: 46, r: 14, t: 14, b: 36 };
  const t0 = 3;
  const t1 = 32;
  const ideal = stencilIdealAI(P7, bytes);
  const yMax = ideal * 1.12;
  const X = (v: number) => pad.l + ((v - t0) / (t1 - t0)) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - (v / yMax) * (H - pad.t - pad.b);
  const ts = Array.from({ length: t1 - t0 + 1 }, (_, i) => t0 + i);
  const legal = ts.filter((v) => v <= maxSide);
  const beyond = ts.filter((v) => v >= Math.max(maxSide, t0));
  const line = (vs: number[]) => vs.map((v) => `${X(v)},${Y(cubeAI(v, 1, bytes))}`).join(' ');
  const step = ideal > 2 ? 1 : ideal > 1 ? 0.4 : 0.2;
  const yt = Array.from({ length: Math.floor(ideal / step) + 1 }, (_, n) => Math.round(n * step * 10) / 10);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('FLOP per byte against tile side', 'FLOP mỗi byte theo cạnh ô')} style={{ width: '100%', minWidth: PLOT_MIN, maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {yt.map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end">{v}</text>
        </g>
      ))}
      {[4, 8, 12, 16, 20, 24, 28, 32].map((v) => (
        <text key={v} x={X(v)} y={H - pad.b + 15} fontSize="11" textAnchor="middle">{v}</text>
      ))}
      <text x={(W + pad.l) / 2} y={H - 6} fontSize="11.5" textAnchor="middle">
        <Translated vi="cạnh ô t">tile side t</Translated>
      </text>
      <line x1={pad.l} x2={W - pad.r} y1={Y(ideal)} y2={Y(ideal)} stroke="var(--ink)" strokeDasharray="5 4" />
      <text x={W - pad.r - 4} y={Y(ideal) - 5} fontSize="11" textAnchor="end">
        <Translated vi={<>lý tưởng {ideal}</>}>ideal {ideal}</Translated>
      </text>
      {maxSide >= t0 && maxSide < t1 ? (
        <g data-mark="limit">
          <line x1={X(maxSide)} x2={X(maxSide)} y1={pad.t} y2={H - pad.b} stroke="var(--fail)" strokeDasharray="3 3" />
          <text x={X(maxSide) + 5} y={pad.t + 13} fontSize="11" style={{ fill: 'var(--fail)' }}>
            <Translated vi={<>giới hạn số luồng: t ≤ {maxSide}</>}>thread limit: t ≤ {maxSide}</Translated>
          </text>
        </g>
      ) : null}
      {legal.length > 1 ? <polyline points={line(legal)} fill="none" stroke="var(--ink)" strokeWidth="2" /> : null}
      {beyond.length > 1 ? <polyline points={line(beyond)} fill="none" stroke="var(--ink-3)" strokeWidth="1.6" strokeDasharray="4 4" /> : null}
      {bytes === 4 ? (
        <g data-mark="book">
          <circle cx={X(8)} cy={Y(cubeAI(8))} r="4" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.6" />
          <text x={X(8) + 7} y={Y(cubeAI(8)) + 16} fontSize="11">
            <Translated vi={<>t = 8 của sách: {r2(cubeAI(8))}</>}>the book's t = 8: {r2(cubeAI(8))}</Translated>
          </text>
        </g>
      ) : null}
      <circle cx={X(t)} cy={Y(cubeAI(t, 1, bytes))} r="5.5" fill={t <= maxSide ? 'var(--ink)' : 'var(--fail)'} data-mark="current" />
    </svg>
  );
}

function CubeExplorer() {
  const { reports } = useReports();
  const limits = useMemo(() => [BOOK_LIMIT, ...reportedLimits(reports)], [reports]);
  const [t, setT] = useState(8);
  const [bytes, setBytes] = useState(4);
  const [lk, setLk] = useState('book');
  const [slice, setSlice] = useState(4);
  const lim = limits.find((l) => l.key === lk) ?? BOOK_LIMIT;
  const z = Math.min(slice, t - 1);
  const threads = threadsCube(t);
  const over = threads > lim.threads;
  const ai = cubeAI(t, 1, bytes);
  const ideal = stencilIdealAI(P7, bytes);
  const shared = sharedBytesCube(t, bytes);
  const sharedOver = lim.shared != null && shared > lim.shared;
  const maxSide = maxCubeSide(lim.threads);
  const metalMissing = !!reports['metal'] && !limits.some((l) => l.key === 'metal');
  return (
    <Bench
      title={tr('A cube-shaped tile', 'Một ô hình lập phương')}
      task={tr('Find the largest tile your thread limit allows, and how far its FLOP/B still is from the ideal.', 'Tìm ô lớn nhất mà giới hạn số luồng của bạn cho phép, và FLOP/B của nó còn cách mức lý tưởng bao xa.')}
      foot={
        <>
          <Translated
            vi={
              <>
                Một luồng cho mỗi phần tử của ô, như trong Hình 8.8; stencil có bậc 1. Lát cắt là một mặt phẳng z của ô đầu vào. Các tỉ
                lệ tính cho một khối bên trong nạp mỗi phần tử của ô một lần và lưu mỗi đầu ra một lần. {lim.note}
              </>
            }
          >
            One thread per tile element, as in Fig. 8.8; the stencil has order 1. The slice is one z-plane of the input
            tile. Ratios count an interior block that loads every tile element once and stores every output once. {lim.note}
          </Translated>
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field">
          <span>
            <Translated vi={<>Cạnh ô t = <b data-out="t">{t}</b></>}>
              Tile side t = <b data-out="t">{t}</b>
            </Translated>
          </span>
          <input type="range" aria-label={tr('tile side', 'cạnh ô')} min="3" max="32" value={t} onChange={(e) => setT(+e.target.value)} style={{ width: '13rem' }} />
        </label>
        <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
          <Translated vi="Số byte mỗi giá trị">Bytes per value</Translated>
          <span className="seg" data-control="bytes">
            {[
              [4, '4 (float)'], // i18n-ignore: type names
              [8, '8 (double)'], // i18n-ignore: type names
              [2, '2 (half)'], // i18n-ignore: type names
            ].map(([v, l]) => (
              <button key={v} aria-pressed={bytes === v} onClick={() => setBytes(v as number)}>{l}</button>
            ))}
          </span>
        </div>
        <label className="field">
          <Translated vi="Giới hạn số luồng">Thread limit</Translated>
          <select value={lk} onChange={(e) => setLk(e.target.value)} data-control="limit">
            {limits.map((l) => (
              <option key={l.key} value={l.key}>{l.label}</option>
            ))}
          </select>
        </label>
      </div>
      {metalMissing ? (
        <p className="status-line" style={{ marginTop: '-0.4rem' }} data-note="metal-cap">
          <Translated
            vi={
              <>
                Thiết bị Metal của bạn chưa có trong danh sách: giới hạn số luồng của nó thuộc về một pipeline đã biên dịch, không phải
                thiết bị, và lab 8.1 ghi lại nó khi chạy trên Metal.
              </>
            }
          >
            Your Metal device is not in the list yet: its thread cap belongs to a compiled pipeline, not the device, and lab
            8.1 records it when it runs on Metal.
          </Translated>
        </p>
      ) : null}
      <div className="cols plot">
        <div>
          <div style={{ overflowX: 'auto' }} data-scroll="ai-curve">
            <AICurve t={t} bytes={bytes} maxSide={maxSide} />
          </div>
          <div className="row" style={{ marginTop: '0.8rem', alignItems: 'flex-start' }}>
            <div style={{ flex: '0 1 252px', minWidth: '10rem' }}>
              <SliceView t={t} z={z} />
            </div>
            <div style={{ flex: '1 1 12rem' }}>
              <label className="field">
                <span>
                  <Translated vi={<>Lát z = <b data-out="z">{z}</b> trong khoảng 0 đến {t - 1}</>}>
                    Slice z = <b data-out="z">{z}</b> of 0 to {t - 1}
                  </Translated>
                </span>
                <input type="range" aria-label={tr('slice', 'lát cắt')} min="0" max={t - 1} value={z} onChange={(e) => setSlice(+e.target.value)} style={{ width: '11rem' }} />
              </label>
              <span className="legend" style={{ flexDirection: 'column', marginTop: '0.6rem' }}>
                {(['output', 'halo', 'never'] as Role[]).map((ro) => (
                  <span key={ro}>
                    <svg width="12" height="12" style={{ verticalAlign: '-0.1em', marginRight: '0.35em' }} aria-hidden="true">
                      <NeverPattern />
                      <rect width="12" height="12" fill={ROLE_FILL[ro]} stroke="var(--rule)" />
                    </svg>
                    {ROLE_NAME[ro]}
                  </span>
                ))}
              </span>
            </div>
          </div>
        </div>
        <div className="panel" data-readout="cube">
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr>
                <td>
                  <Translated vi="Số luồng mỗi khối, t³">Threads per block, t³</Translated>
                </td>
                <td className="num" data-out="threads">
                  {over ? <span className="chip fail">{count(threads)} &gt; {count(lim.threads)}</span> : <Translated vi={<>{count(threads)} trên {count(lim.threads)}</>}>{count(threads)} of {count(lim.threads)}</Translated>}
                </td>
              </tr>
              <tr>
                <td>
                  <Translated vi="Cạnh hợp lệ lớn nhất">Largest legal side</Translated>
                </td>
                <td className="num" data-out="maxside">{maxSide}</td>
              </tr>
              <tr>
                <td>FLOP/B</td>
                <td className="num" data-out="ai">{r2(ai)} <span className="muted"><Translated vi={<>trên mức lý tưởng {ideal} ({pct(ai / ideal, 0)})</>}>of {ideal} ideal ({pct(ai / ideal, 0)})</Translated></span></td>
              </tr>
              <tr>
                <td>
                  <Translated vi="Vùng biên">Halo</Translated>
                </td>
                <td className="num" data-out="halo">
                  <Translated vi={<>{count(haloCount(t))} trên {count(threads)} ({pct(haloFraction(t))})</>}>
                    {count(haloCount(t))} of {count(threads)} ({pct(haloFraction(t))})
                  </Translated>
                </td>
              </tr>
              <tr>
                <td>
                  <Translated vi="Được nạp, không bao giờ được đọc">Loaded, never read</Translated>
                </td>
                <td className="num" data-out="never">{count(neverRead(t))} ({pct(neverRead(t) / threads)})</td>
              </tr>
              <tr>
                <td>
                  <Translated vi={<>Bộ nhớ chia sẻ, t³ × {bytes} B</>}>Shared memory, t³ × {bytes} B</Translated>
                </td>
                <td className="num" data-out="shared">
                  {sharedOver ? <span className="chip fail">{bytesText(shared)}</span> : bytesText(shared)}
                  {lim.shared != null ? <div className="muted">
                      <Translated vi={<>giới hạn {bytesText(lim.shared)}: {lim.sharedName}</>}>
                        limit {bytesText(lim.shared)}: {lim.sharedName}
                      </Translated>
                    </div> : null}
                </td>
              </tr>
            </tbody>
          </table>
          <p style={{ margin: '0.7rem 0 0' }} data-out="reach">
            {over ? (
              <Translated
                vi={
                  <>
                    Khối này vượt giới hạn và sẽ không khởi chạy được. Mức tốt nhất một khối lập phương hợp lệ đạt được ở đây là t ={' '}
                    {maxSide}: {r2(cubeAI(maxSide, 1, bytes))} FLOP/B.
                  </>
                }
              >
                <>
                  This block is over the limit and would not launch. The best a legal cube does here is t = {maxSide}:{' '}
                  {r2(cubeAI(maxSide, 1, bytes))} FLOP/B.
                </>
              </Translated>
            ) : (
              <Translated
                vi={
                  <>
                    Để đạt 90% mức lý tưởng, một khối lập phương cần t = {sideForShare(0.9)}: {count(threadsCube(sideForShare(0.9) ?? 0))}{' '}
                    luồng trong một khối.
                  </>
                }
              >
                <>
                  To reach 90% of the ideal a cube needs t = {sideForShare(0.9)}: {count(threadsCube(sideForShare(0.9) ?? 0))} threads in
                  one block.
                </>
              </Translated>
            )}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: one warp's load
const SHAPES: { key: string; label: string; block: Dims }[] = [
  { key: '8', label: '8 × 8 × 8', block: dims3(8, 8, 8) },
  { key: '10', label: '10 × 10 × 10', block: dims3(10, 10, 10) },
  { key: '32', label: '32 × 8 × 4', block: dims3(32, 8, 4) },
];
const GRID_N = 128;

function PlaneGrid({ block, z, lanes }: { block: Dims; z: number; lanes: Map<string, Lane> }) {
  const cs = block.x >= 32 ? 14 : 24;
  const W = block.x * cs;
  const H = block.y * cs;
  const cells: ReactNode[] = [];
  for (let y = 0; y < block.y; y++)
    for (let x = 0; x < block.x; x++) {
      const l = lanes.get(`${x},${y},${z}`);
      let fill = 'var(--cell)';
      let stroke = 'var(--rule)';
      let dash: string | undefined;
      let state = 'other';
      if (l) {
        if (l.index !== null) {
          fill = 'var(--global-wash)';
          stroke = 'var(--global)';
          state = 'loads';
        } else {
          stroke = 'var(--ink-2)';
          dash = '2 2';
          state = 'skips';
        }
      }
      cells.push(
        <g key={x + ',' + y} data-lane={l ? l.lane : undefined} data-state={state}>
          <rect x={x * cs + 0.5} y={y * cs + 0.5} width={cs - 1} height={cs - 1} fill={fill} stroke={stroke} strokeDasharray={dash} />
          {l && cs >= 20 ? (
            <text x={x * cs + cs / 2} y={y * cs + cs / 2 + 3.5} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink)' }}>
              {l.lane}
            </text>
          ) : null}
        </g>,
      );
    }
  return (
    <figure style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${W + 1} ${H + 1}`} role="img" aria-label={tr(`Plane z = ${z} of the block`, `Mặt phẳng z = ${z} của khối`)} style={{ width: '100%', maxWidth: W + 1, display: 'block' }}>
        {cells}
      </svg>
      <figcaption className="status-line">
        <Translated vi={<>threadIdx.z = {z}: x theo chiều ngang, y theo chiều dọc</>}>threadIdx.z = {z}: x across, y down</Translated>
      </figcaption>
    </figure>
  );
}

function MemoryStrip({ cfg, view }: { cfg: WarpConfig; view: ReturnType<typeof warpView> }) {
  const cw = 12;
  return (
    <div data-strips>
      {view.rows.map((row) => {
        const strip = rowStrip(cfg, view, row);
        const W = strip.length * cw;
        return (
          <figure key={row.i + ',' + row.j} style={{ margin: '0 0 0.7rem' }} data-row={`${row.i},${row.j}`}>
            <figcaption className="status-line">
              <Translated
                vi={
                  <>
                    hàng i = {row.i}, j = {row.j}: k = {row.kFirst} đến {row.kLast}, {row.segLast - row.segFirst + 1} đoạn
                  </>
                }
              >
                row i = {row.i}, j = {row.j}: k = {row.kFirst} to {row.kLast}, {row.segLast - row.segFirst + 1} segment{row.segLast > row.segFirst ? 's' : ''}
              </Translated>
            </figcaption>
            <svg viewBox={`0 0 ${W + 2} 26`} role="img" aria-label={tr(`Global memory around row ${row.j}`, `Bộ nhớ toàn cục quanh hàng ${row.j}`)} style={{ width: '100%', maxWidth: W + 2, display: 'block' }}>
              {strip.map((c, n) => (
                <rect
                  key={c.index}
                  x={1 + n * cw}
                  y={4}
                  width={cw}
                  height={16}
                  fill={c.lane !== null ? 'var(--global)' : c.fetched ? 'var(--global-wash)' : 'var(--cell)'}
                  stroke="var(--rule)"
                  strokeWidth="0.6"
                  data-mem={c.lane !== null ? 'used' : c.fetched ? 'fetched' : 'untouched'}
                />
              ))}
              {strip.map((c, n) =>
                (c.index * (cfg.bytes ?? 4)) % 32 === 0 ? (
                  <line key={'s' + c.index} x1={1 + n * cw} x2={1 + n * cw} y1={0} y2={24} stroke="var(--ink)" strokeWidth="1.4" />
                ) : null,
              )}
              <line x1={1 + strip.length * cw} x2={1 + strip.length * cw} y1={0} y2={24} stroke="var(--ink)" strokeWidth="1.4" />
            </svg>
          </figure>
        );
      })}
    </div>
  );
}

function Seg<T extends string | number>({
  label,
  name,
  value,
  set,
  options,
}: {
  label: string;
  name: string;
  value: T;
  set: (v: T) => void;
  options: { v: T; label: string; disabled?: boolean; title?: string }[];
}) {
  return (
    <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
      {label}
      <span className="seg" data-control={name}>
        {options.map((o) => (
          <button key={String(o.v)} aria-pressed={value === o.v} disabled={o.disabled} title={o.title} onClick={() => set(o.v)}>
            {o.label}
          </button>
        ))}
      </span>
    </div>
  );
}

function WarpExplorer() {
  const [shape, setShape] = useState('8');
  const [bx, setBx] = useState(1);
  const [warpSel, setWarp] = useState(0);
  const [width, setWidth] = useState(32);
  const [offset, setOffset] = useState(-1);
  const block = (SHAPES.find((s) => s.key === shape) ?? SHAPES[0]).block;
  const nWarps = warpsPerBlock(block, width);
  const warp = Math.min(warpSel, nWarps - 1);
  const base = { block, r: 1, blockIdx: dims3(bx, 1, 1), width, N: GRID_N, offset };
  const cfg: WarpConfig = { ...base, warp };
  const view = warpView(cfg);
  const blk = blockView(base);
  const lanes = new Map(view.lanes.map((l) => [`${l.t.x},${l.t.y},${l.t.z}`, l]));
  const planes = [...new Set(view.lanes.map((l) => l.t.z))];
  const unit = width === 64 ? 'wavefront' : 'warp';
  return (
    <Bench
      title={tr("One warp's load", 'Lần nạp của một warp')}
      task={tr(
        "Count the segments one warp's load touches for each block shape, then find where a row has to start to touch only four.",
        'Đếm số đoạn mà lần nạp của một warp chạm tới với mỗi hình dạng khối, rồi tìm xem một hàng phải bắt đầu ở đâu để chỉ chạm bốn đoạn.',
      )}
      foot={
        <Translated
          vi={
            <>
              Một lưới {GRID_N}³ số float; khối ở blockIdx (x {bx}, y 1, z 1), nên chỉ vị trí x thay đổi. Các lane được đánh số theo
              thứ tự của chương 4: x nhanh nhất, rồi y, rồi z. Các vạch đậm trong dải bộ nhớ là ranh giới đoạn 32 byte; ô đậm là số
              float mà một lane nạp, ô nhạt là phần còn lại của một đoạn mà lần nạp lấy về.
            </>
          }
        >
          <>
            A {GRID_N}³ grid of floats; the block is at blockIdx (x {bx}, y 1, z 1), so only the x position changes. Lanes
            are numbered in the order of chapter 4: x fastest, then y, then z. Thick lines in the memory strips are 32-byte
            segment boundaries; dark cells are floats a lane loads, light cells the rest of a segment the load fetches.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <Seg label={tr('Block shape', 'Hình dạng khối')} name="shape" /* i18n-ignore: data-control key */ value={shape} set={(v) => setShape(v)} options={SHAPES.map((s) => ({ v: s.key, label: s.label }))} />
        <Seg label="blockIdx.x" name="bx" value={bx} set={setBx} options={[0, 1, 2, 3].map((v) => ({ v, label: String(v) }))} />
        <Seg
          label={tr('Warp width', 'Độ rộng warp')}
          name="width" // i18n-ignore: data-control key
          value={width}
          set={setWidth}
          options={[
            { v: 32, label: '32' },
            { v: 64, label: '64 (AMD wave64)' }, // i18n-ignore: product name
            { v: 0, label: 'Metal', disabled: true, title: tr('SIMD-group partition not documented', 'Cách chia SIMD group không được ghi nhận') },
          ]}
        />
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field">
          <span>
            <Translated vi={<>{unit} <b data-out="warp">{warp}</b> trong khoảng 0 đến {nWarps - 1}</>}>
              {unit} <b data-out="warp">{warp}</b> of 0 to {nWarps - 1}
            </Translated>
          </span>
          <input type="range" aria-label={tr('warp', 'warp')} min="0" max={nWarps - 1} value={warp} onChange={(e) => setWarp(+e.target.value)} style={{ width: '11rem' }} />
        </label>
        <Seg
          label={tr('Halo offset', 'Độ lệch vùng biên')}
          name="offset" // i18n-ignore: data-control key
          value={offset}
          set={setOffset}
          options={[
            { v: -1, label: tr('−1 (the book)', '−1 (của sách)') },
            { v: 0, label: tr('0 (compare)', '0 (để so sánh)') },
          ]}
        />
      </div>
      <p className="status-line" style={{ marginTop: '-0.3rem' }}>
        <Translated
          vi={
            <>
              Metal bị làm mờ: Apple nói việc chia một threadgroup thành các SIMD group là do cài đặt quyết định, nên không có warp nào
              để vẽ (bảng facts; ch04 C2).
            </>
          }
        >
          Metal is greyed out: Apple says the partition of a threadgroup into SIMD-groups is implementation-defined, so
          there is no warp to draw (facts table; ch04 C2).
        </Translated>
      </p>
      <div className="cols plot">
        <div>
          <div className="row" style={{ alignItems: 'flex-start', gap: '1rem' }} data-planes={planes.length}>
            {planes.map((z) => (
              <div key={z} style={{ flex: block.x >= 32 ? '1 1 100%' : '0 1 auto', minWidth: 0 }}>
                <PlaneGrid block={block} z={z} lanes={lanes} />
              </div>
            ))}
          </div>
          <div style={{ marginTop: '0.6rem' }}>
            <MemoryStrip cfg={cfg} view={view} />
          </div>
        </div>
        <div className="panel" data-readout="warp">
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr><td><Translated vi="Số hàng của ô">Rows of the tile</Translated></td><td className="num" data-out="rows">{view.rows.length}</td></tr>
              <tr>
                <td>
                  <Translated vi="Số lane nạp">Lanes that load</Translated>
                </td>
                <td className="num" data-out="loading">
                  <Translated vi={<>{view.loading} trên {view.threads}</>}>{view.loading} of {view.threads}</Translated>
                </td>
              </tr>
              <tr><td><Translated vi="Số giao dịch 32 byte">32-byte transactions</Translated></td><td className="num" data-out="transactions">{view.transactions}</td></tr>
              <tr><td><Translated vi="Số byte dùng / lấy về">Bytes used / fetched</Translated></td><td className="num" data-out="bytes">{view.bytesUsed} / {view.bytesFetched}</td></tr>
              <tr><td><Translated vi="Hiệu suất">Efficiency</Translated></td><td className="num" data-out="eff">{pct(view.efficiency)}</td></tr>
              <tr>
                <td>
                  <Translated vi={<>Cả khối ({blk.warps.length} {unit})</>}>Whole block ({blk.warps.length} {unit}s)</Translated>
                </td>
                <td className="num" data-out="block">
                  <Translated vi={<>{blk.transactions} giao dịch, {pct(blk.efficiency)}</>}>
                    {blk.transactions} transactions, {pct(blk.efficiency)}
                  </Translated>
                </td>
              </tr>
            </tbody>
          </table>
          {view.padded > 0 ? (
            <p className="status-line" style={{ margin: '0.6rem 0 0' }} data-out="padded">
              <Translated vi={<>{unit} này chứa {view.threads} luồng; {view.padded} lane là phần đệm.</>}>
                This {unit} holds {view.threads} threads; {view.padded} lanes are padding.
              </Translated>
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page
export function S84() {
  const finite = [64, 128, 512].map((n) => ({ n, ai: idealAIFinite(n) }));
  return (
    <>
      <p>
        <Translated
          vi={
            <>
              Trang này đưa stencil 3D của chương qua hai kernel đầu tiên và đếm xem mỗi kernel đòi hỏi bộ nhớ những gì. Stencil có
              bậc 1: mỗi đầu ra là một tổng có trọng số của đầu vào tại cùng điểm và sáu láng giềng theo trục của nó.
            </>
          }
        >
          This page takes the chapter's 3D stencil through the first two kernels and counts what each asks of memory. The
          stencil has order 1: each output is a weighted sum of the input at the same point and its six axis neighbours.
        </Translated>
      </p>

      <h2>
        <Translated vi="Một luồng cho mỗi điểm">One thread per point</Translated>
      </h2>
      <p>
        <Translated
          vi={
            <>
              Kernel cơ bản cho mỗi điểm lưới một luồng (thread). Cách ánh xạ là cách thông thường, với x nhanh nhất: <code>k</code>{' '}
              đến từ các trường x của <code>blockIdx</code>, <code>blockDim</code> và <code>threadIdx</code>,{' '}
              <code>j</code> từ y và <code>i</code> từ z. Lưới có kích thước N × N × N, lưu theo thứ tự hàng (row-major), nên điểm (i,
              j, k) nằm ở{' '}
              <code>i·N² + j·N + k</code>, và sáu láng giềng của nó nằm ở ±1, ±N và ±N² tính từ đó. Chỉ các điểm bên trong được tính,
              và phép kiểm tra điểm bên trong làm thêm một việc: nó bảo đảm cả sáu chỉ số láng giềng đều nằm trong mảng.
            </>
          }
        >
          The basic kernel gives every grid point a thread. The mapping is the usual one, with x fastest: <code>k</code>{' '}
          comes from the x fields of <code>blockIdx</code>, <code>blockDim</code> and <code>threadIdx</code>,{' '}
          <code>j</code> from y and <code>i</code> from z. The grid is N × N × N, stored row-major, so point (i, j, k) sits at{' '}
          <code>i·N² + j·N + k</code>, and its six neighbours sit at ±1, ±N and ±N² from it. Only interior points are
          computed, and the interior test does a second job: it guarantees that all six neighbour indices are inside the
          array.
        </Translated>
      </p>
      <Source section="8.2" pages="188–189" figs={tr('Fig. 8.6', 'Hình 8.6')} />
      <LabListing
        cuPath={BASIC_CU}
        metalPath={BASIC_METAL}
        cuSig={/__global__\s+void\s+stencilBasicKernel/}
        metalSig={/kernel\s+void\s+stencil_basic\b/}
        caption={
          <Translated
            vi={
              <>
                Kernel cơ bản từ lời giải tham chiếu (solution) của lab 8.1, được nạp từ <code>labs/{LAB}/solution/</code>. Lab 8.1 yêu
                cầu bạn viết nó; hãy thử làm trước khi nghiên cứu bản này. Bản build HIP dùng nguyên tệp CUDA; đường đó chưa được biên
                dịch trên phần cứng AMD.
              </>
            }
          >
            <>
              The basic kernel from the reference solution of lab 8.1, loaded from <code>labs/{LAB}/solution/</code>. Lab
              8.1 asks you to write it; try that before studying this one. The HIP build uses the CUDA file unchanged; that
              path has not been compiled on AMD hardware yet.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Có hai điểm khác với listing in trong sách. Listing dùng <code>c0</code> đến <code>c6</code> mà không khai báo chúng;
            sách để bạn tự viết cứng chúng hoặc đặt chúng vào bộ nhớ hằng (constant memory) (pp. 188–189). Lab truyền chúng dưới dạng
            một struct nhỏ, <code>StencilCoeffs</code>: theo giá trị trong CUDA, cách mà chương 7 liệt kê là cách thứ ba để đưa trọng
            số cố định cho kernel (§7.4, p. 172), và dưới dạng một buffer <code>constant</code> trong Metal. Và việc ghép hệ số với
            láng giềng được cố định như trong Hình 8.6 và 8.8:{' '}
            <code>c1</code>/<code>c2</code> cho x − 1 và x + 1, <code>c3</code>/<code>c4</code> cho y − 1 và y + 1,{' '}
            <code>c5</code>/<code>c6</code> cho z − 1 và z + 1. Các listing của trang 8.5 đổi chỗ <code>c3</code> và{' '}
            <code>c4</code>; lab thì không.
          </p>
        }
      >
        <p>
          Two things differ from the printed listing. The listing uses <code>c0</code> to <code>c6</code> without declaring
          them; the book leaves you to hard-code them or put them in constant memory (pp. 188–189). The lab passes them as
          one small struct, <code>StencilCoeffs</code>: by value in CUDA, which chapter 7 lists as a third way to give a
          kernel fixed weights (§7.4, p. 172), and as a <code>constant</code> buffer in Metal. And the pairing of coefficients with neighbours is fixed as in Figs. 8.6 and 8.8:{' '}
          <code>c1</code>/<code>c2</code> for x − 1 and x + 1, <code>c3</code>/<code>c4</code> for y − 1 and y + 1,{' '}
          <code>c5</code>/<code>c6</code> for z − 1 and z + 1. Page 8.5's listings swap <code>c3</code> and{' '}
          <code>c4</code>; the lab does not.
        </p>
      </Translated>
      <Source section="8.2 and 7.4" pages="188–189, 172" />

      <h2>
        <Translated vi="Hãy đếm">Count it</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hãy đọc kernel ở trên và đếm, cho một luồng: số phép toán, số byte nạp, số byte lưu.
          </p>
        }
      >
        <p>
          Read the kernel above and count, for one thread: operations, bytes loaded, bytes stored.
        </p>
      </Translated>
      <Predict id="ch08/8.4/p-basic" prompt={
          <Translated vi={<>Tỉ lệ giữa số phép toán và số byte lưu lượng bộ nhớ toàn cục của kernel cơ bản là bao nhiêu? Hãy chốt một con số.</>}>
            <>What is the basic kernel's ratio of operations to bytes of global-memory traffic? Commit a number.</>
          </Translated>
        } unit="FLOP/B">
        {(v) => <BasicReveal v={v} />}
      </Predict>
      <Translated
        vi={
          <p>
            Giờ đến mức tốt nhất mà bất kỳ kernel nào có thể làm. Nếu mọi giá trị đầu vào được nạp từ bộ nhớ toàn cục (global memory)
            đúng một lần và mọi đầu ra được lưu một lần, mỗi đầu ra sẽ tốn một lần nạp 4 byte và một lần lưu 4 byte: 13 phép toán trên
            8 byte,{' '}
            <b>{IDEAL} FLOP/B</b>. Trên một lưới hữu hạn thì thấp hơn một chút, vì lớp biên được nạp như đầu vào nhưng không bao giờ
            được tính như đầu ra: sách đếm n³ − 12n + 16 đầu vào (mọi điểm trừ những điểm trên 12 cạnh của khối lập phương, nơi không
            stencil nào với tới) so với (n − 2)³ đầu ra.
          </p>
        }
      >
        <p>
          Now the best any kernel could do. If every input value were loaded from global memory exactly once and every
          output stored once, each output would cost one 4-byte load and one 4-byte store: 13 operations over 8 bytes,{' '}
          <b>{IDEAL} FLOP/B</b>. On a finite grid it is a little lower, because the boundary layer is loaded as input but
          never computed as output: the book counts n³ − 12n + 16 inputs (every point except those on the cube's 12 edges,
          which no stencil reaches) against (n − 2)³ outputs.
        </p>
      </Translated>
      <Source section="8.3" pages="189" />
      <table className="data" data-table="finite" style={{ maxWidth: '30rem' }}>
        <thead>
          <tr>
            <th>
              <Translated vi="Cạnh lưới n">Grid side n</Translated>
            </th>
            {finite.map((f) => <th key={f.n} className="num">{f.n}</th>)}
            <th className="num">
              <Translated vi="n lớn">large n</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr><td><Translated vi="FLOP/B lý tưởng">Ideal FLOP/B</Translated></td>{finite.map((f) => <td key={f.n} className="num">{r2(f.ai)}</td>)}<td className="num">{IDEAL}</td></tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Dù thế nào stencil cũng bị giới hạn bởi bộ nhớ. Trên thiết bị ví dụ của sách, ngưỡng là khoảng {BOOK_THRESHOLD}{' '}
            FLOP/B (chương 5, p. 94): kernel cơ bản thấp hơn ngưỡng {fmt(BOOK_THRESHOLD / BASIC)} lần, và ngay cả mức lý tưởng cũng
            thấp hơn hơn mười lần. Không mức phân ô nào làm kernel này bị giới hạn bởi tính toán ở đó; phân ô chỉ có thể đưa nó từ{' '}
            {r2(BASIC)} tiến về {IDEAL}.
          </p>
        }
      >
        <p>
          Either way the stencil is memory-bound. On the book's example device the threshold is about {BOOK_THRESHOLD}{' '}
          FLOP/B (chapter 5, p. 94): the basic kernel is {fmt(BOOK_THRESHOLD / BASIC)} times below it, and even the ideal is
          more than ten times below it. No amount of tiling makes this kernel compute-bound there; tiling can only move it
          from {r2(BASIC)} towards {IDEAL}.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch08-facts.md, A8)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Thiết bị ví dụ của sách là H100 của NVIDIA; 20 FLOP/B là tốc độ FP32 công bố chia cho băng thông công bố của nó, ngưỡng mà
              chương 5 dùng. Hãy coi nó chỉ là ví dụ của sách.
            </p>
          }
        >
          <p>
            The book's example device is NVIDIA's H100; 20 FLOP/B is its published FP32 rate over its published bandwidth,
            the threshold chapter 5 uses. Treat it as the book's example only.
          </p>
        </Translated>
      </Today>
      <YourThreshold />
      <Translated
        vi={
          <p>
            Như ở chương 5, một phép đo có thể cho kết quả cao hơn mức trần {r2(BASIC)}: các luồng kề nhau đọc nhiều giá trị giống
            nhau, và các bộ nhớ đệm (cache) L1 và L2 phục vụ một phần những lần đọc lặp lại đó. Sách cũng nói vậy (p. 189), và thêm
            rằng phân ô là con đường đáng tin cậy.
          </p>
        }
      >
        <p>
          As in chapter 5, a measurement can come out above the {r2(BASIC)} ceiling: neighbouring threads read many of the
          same values, and the L1 and L2 caches serve some of those repeats. The book says the same (p. 189), and adds that
          tiling is the reliable route.
        </p>
      </Translated>

      <h2>
        <Translated vi="Vì sao phân ô ở đây được lợi ít hơn so với tích chập">Why tiling buys less here than for a convolution</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Thiết kế phân ô gần như là thiết kế tích chập của chương 7: một khối nạp một ô đầu vào, kể cả vùng biên (halo), và tính
            một ô đầu ra từ nó. Khác biệt là hình dạng của những gì mỗi đầu ra đọc. Một bộ lọc dày đặc dùng mọi ô trong hộp của nó;
            một stencil chỉ dùng các ô trên những trục đi qua tâm. Vì vậy ô đầu vào của một stencil 2D 5 điểm không có góc (Hình
            8.7), và trong 3D phần hộp mà stencil dùng còn tiếp tục thu nhỏ.
          </p>
        }
      >
        <p>
          The tiling design is almost the convolution design of chapter 7: a block loads an input tile, halo included, and
          computes an output tile from it. The difference is the shape of what each output reads. A dense filter uses every
          cell of its box; a stencil uses only the cells on the axes through the centre. The input tile of a 2D 5-point
          stencil therefore has no corners (Fig. 8.7), and in 3D the share of the box the stencil uses keeps shrinking.
        </p>
      </Translated>
      <Source section="8.4" pages="189–190" figs={tr('Fig. 8.7', 'Hình 8.7')} />
      <Translated
        vi={
          <p>
            Ít lần đọc hơn mỗi đầu ra nghĩa là mỗi giá trị đã nạp được dùng lại ít hơn, và điều đó chặn tỉ lệ, vì 8 byte bắt buộc mỗi
            đầu ra (một lần nạp, một lần lưu) giữ nguyên trong khi khối lượng công việc thu nhỏ. Hãy chọn số chiều và bậc rồi so sánh.
          </p>
        }
      >
        <p>
          Fewer reads per output means less reuse of each loaded value, and that caps the ratio, because the compulsory 8
          bytes per output (one load, one store) stay the same while the work shrinks. Pick the dimension and the order and
          compare.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>Chỉ các tỉ lệ lý tưởng: dùng lại hoàn hảo, giá trị 4 byte, không kernel, không cache, không đo thời gian.</p>
          }
        >
          <p>Ideal ratios only: perfect reuse, 4-byte values, no kernel, no caches, no timing.</p>
        </Translated>
      </Simplification>
      <Beyond>
        <Translated
          vi={
            <p>
              Các dạng đóng là của chúng tôi: một stencil p điểm làm 2p − 1 phép toán mỗi đầu ra, nên mức lý tưởng của nó là (2p − 1)/8
              FLOP/B; một bộ lọc dày đặc với m<sup>d</sup> tap, m<sup>d</sup>/4. Sách cho các giá trị (p. 190).
            </p>
          }
        >
          <p>
            The closed forms are ours: a p-point stencil does 2p − 1 operations per output, so its ideal is (2p − 1)/8
            FLOP/B; a dense filter with m<sup>d</sup> taps, m<sup>d</sup>/4. The book gives the values (p. 190).
          </p>
        </Translated>
      </Beyond>
      <ShapeExplorer />
      <Ask q={questionsS84['ch08/8.4/q-reuse']} />
      <Translated
        vi={
          <p>
            Hãy ghi nhớ những góc bị thiếu. Ở trang 8.5 chúng trở thành lý do một giá trị chỉ một luồng đọc có thể nằm trong thanh ghi
            (register) của luồng đó.
          </p>
        }
      >
        <p>
          Keep the missing corners in mind. On page 8.5 they become the reason a value read by only one thread can live in
          that thread's registers.
        </p>
      </Translated>

      <h2>
        <Translated vi="Một ô hình lập phương">A cube-shaped tile</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hình 8.8 là <a href={href('ch07', '7.5')}>kernel phân ô của chương 7</a> trong ba chiều. Khối có kích thước bằng ô đầu
            vào, nên mỗi luồng nạp đúng một phần tử. <code>− 1</code> ở mỗi tọa độ là bậc của stencil: các luồng của một khối bắt đầu
            trước ô đầu ra của nó một phần tử. Các lần nạp được chặn trong <code>[0, N)</code> trên mọi trục, nên các luồng của một
            khối ở rìa lưới bỏ qua những phần tử không tồn tại. Sau đó là một rào chắn (barrier), và hai điều kiện độc lập quyết định
            ai tính: điểm phải nằm bên trong lưới, và luồng không được ở lớp ngoài của khối, nơi các luồng chỉ nạp.
          </p>
        }
      >
        <p>
          Fig. 8.8 is <a href={href('ch07', '7.5')}>chapter 7's tiled kernel</a> in three dimensions. The block is the size of the input tile, so each thread
          loads exactly one element. The <code>− 1</code> in each coordinate is the stencil's order: a block's threads start
          one element before its output tile. Loads are guarded to <code>[0, N)</code> on every axis, so the threads of a
          block at the grid's edge skip elements that do not exist. Then a barrier, and two independent guards decide who
          computes: the point must be in the grid's interior, and the thread must not be on the block's outer layer, whose
          threads only load.
        </p>
      </Translated>
      <Source section="8.4" pages="191–192" figs={tr('Fig. 8.8', 'Hình 8.8')} />
      <Translated
        vi={
          <p>
            Không có việc điền số không cho các lần nạp bị bỏ qua, khác với chương 7. Không cần điều đó: đầu ra chỉ được tính ở các
            điểm lưới bên trong, mà sáu láng giềng của chúng luôn nằm trong lưới, nên không luồng nào từng đọc một ô đã bị bỏ qua.
          </p>
        }
      >
        <p>
          There is no zero fill for the skipped loads, unlike chapter 7. None is needed: outputs are computed only at
          interior grid points, whose six neighbours are always inside the grid, so no thread ever reads a slot that was
          skipped.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Lập luận đó là của chúng tôi; sách chỉ gọi các điều kiện là phép chặn ngoài biên. Nó được ghi cùng listing trong ghi chú
              của chương.
            </p>
          }
        >
          <p>
            That argument is ours; the book only calls the conditions out-of-bound guards. It is recorded with the listing
            in the chapter notes.
          </p>
        </Translated>
      </Beyond>
      <LabListing cuPath={CUBE_CU} metalPath={CUBE_METAL} caption={(cu) => <CubeCaption cu={cu} />} />
      <Translated
        vi={
          <p>
            Hãy đếm theo từng khối, như chương 7 đã làm. Với một ô đầu vào cạnh t, ô đầu ra có cạnh t − 2; mỗi luồng trong (t − 2)³
            luồng hoạt động làm 13 phép toán; khối nạp t³ giá trị và lưu (t − 2)³. Tỉ lệ của sách là
          </p>
        }
      >
        <p>
          Count it per block, as chapter 7 did. With an input tile of side t, the output tile has side t − 2; each of the
          (t − 2)³ active threads does 13 operations; the block loads t³ values and stores (t − 2)³. The book's ratio is
        </p>
      </Translated>
      <p style={{ textAlign: 'center' }}>
        <code>13·(t − 2)³ / (4·t³ + 4·(t − 2)³)</code> FLOP/B,{/* i18n-ignore: formula and unit */}
      </p>
      <Translated
        vi={
          <p>
            tỉ lệ này tăng theo t, tiến về {IDEAL}. Vậy hãy làm t thật lớn. Trình khám phá cho thấy điều gì ngăn bạn lại.
          </p>
        }
      >
        <p>
          which rises with t towards {IDEAL}. So make t large. The explorer shows what stops you.
        </p>
      </Translated>
      <Source section="8.4" pages="192" />

      <h2>
        <Translated vi="Ngân sách của khối lập phương">The cube's budget</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách: một khối có thể chứa 1024 luồng, nên với một luồng cho mỗi phần tử "giới hạn thực tế của t là 8", một khối 8 × 8 ×
            8 gồm 512 luồng. Điều đó cho {r2(cubeAI(8))} FLOP/B so với mức lý tưởng {IDEAL}, vì vùng biên chiếm{' '}
            {pct(haloFraction(8), 0)} của ô, so với {pct(haloFraction2D(32), 0)} của ô 32 × 32 ở chương 7. Các phần tử vùng biên được
            dùng lại ít hơn phần tử bên trong, và trong một khối lập phương nhỏ chúng chiếm phần lớn ô.
          </p>
        }
      >
        <p>
          The book: a block may hold 1024 threads, so with one thread per element "the practical limit on t is 8", an 8 × 8
          × 8 block of 512 threads. That gives {r2(cubeAI(8))} FLOP/B against the ideal {IDEAL}, because the halo is{' '}
          {pct(haloFraction(8), 0)} of the tile, against {pct(haloFraction2D(32), 0)} for chapter 7's 32 × 32 tile. Halo
          elements are reused less than interior ones, and in a small cube they are most of the tile.
        </p>
      </Translated>
      <Source section="8.4" pages="192–193" />
      <Translated
        vi={
          <p>
            Hãy kéo cạnh. Rồi chuyển giới hạn số luồng sang thiết bị của chính bạn, nếu có trong danh sách, và số byte mỗi giá trị
            sang 8.
          </p>
        }
      >
        <p>
          Drag the side. Then switch the thread limit to your own device, if it is listed, and the bytes per value to 8.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một khối bên trong nạp mỗi phần tử của ô một lần và lưu mỗi đầu ra một lần: không cache, không khối biên, không đo thời
              gian. Số phần tử không bao giờ được đọc và các công thức cho kích thước phần tử khác là của chúng tôi.
            </p>
          }
        >
          <p>
            An interior block that loads each tile element once and stores each output once: no caches, no boundary blocks,
            no timing. The never-read count and the formulas for other element sizes are ours.
          </p>
        </Translated>
      </Simplification>
      <CubeExplorer />
      <Translated
        vi={
          <p>
            8 là lựa chọn lũy thừa của hai của sách, không phải giới hạn. Một khối 10 × 10 × 10 có 1000 luồng, mức mà mọi giới hạn
            CUDA và HIP được ghi nhận đều cho phép; theo công thức của sách nó đạt {r2(cubeAI(10))} FLOP/B với vùng biên{' '}
            {pct(haloFraction(10), 0)}, và warp cuối cùng của nó dùng 8 trên 32 lane. Trên Metal nó chỉ chạy được nếu giới hạn số
            luồng của pipeline đã biên dịch cho phép 1000. Dù thế nào khối lập phương vẫn cách xa {IDEAL}: muốn tiến gần sẽ cần hàng
            chục nghìn luồng trong một khối.
          </p>
        }
      >
        <p>
          8 is the book's power-of-two choice, not the limit. A 10 × 10 × 10 block has 1000 threads, which every documented
          CUDA and HIP limit allows; by the book's formula it reaches {r2(cubeAI(10))} FLOP/B with a{' '}
          {pct(haloFraction(10), 0)} halo, and its last warp has 8 of 32 lanes in use. On Metal it works only if the
          compiled pipeline's thread cap allows 1000. Either way the cube stays far from {IDEAL}: getting close would need
          tens of thousands of threads in one block.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch08-facts.md, A1, B1, C1)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              CUDA: 1024 luồng mỗi khối, tối đa 1024 × 1024 × 64. HIP: 1024 luồng, và 1024 theo mỗi chiều. Metal: 1024 là mức tối đa
              lý thuyết; giới hạn thực là <code>maxTotalThreadsPerThreadgroup</code> của pipeline, và các giá trị theo từng chiều không
              được ghi nhận.
            </p>
          }
        >
          <p>
            CUDA: 1024 threads per block, at most 1024 × 1024 × 64. HIP: 1024 threads, and 1024 in each dimension. Metal:
            1024 is a theoretical maximum; the real cap is the pipeline's <code>maxTotalThreadsPerThreadgroup</code>, and
            per-dimension values are not documented.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS84['ch08/8.4/q-cube-max']} label={tr('Work it out', 'Tự tính ra')} />
      <Ask q={questionsS84['ch08/8.4/q-never-read']} label={tr('Work it out', 'Tự tính ra')} />
      <Translated
        vi={
          <p>
            Bài tập đầu tiên của sách định kích thước các kernel này cho một lưới thực. Mỗi đáp án là một dòng tính toán, và đáp án
            thứ ba cho thấy cái giá của vùng biên bằng một đơn vị khác: số khối.
          </p>
        }
      >
        <p>
          The book's first exercise sizes these kernels for a real grid. Each answer is one line of arithmetic, and the
          third shows the cost of the halo in another currency: blocks.
        </p>
      </Translated>
      <Ask q={questionsS84['ch08/8.4/q-ex1a']} label={tr('Book exercise 1a', 'Bài tập 1a của sách')} />
      <Ask q={questionsS84['ch08/8.4/q-ex1b']} label={tr('Book exercise 1b', 'Bài tập 1b của sách')} />
      <Ask q={questionsS84['ch08/8.4/q-ex1c']} label={tr('Book exercise 1c', 'Bài tập 1c của sách')} />
      <Translated
        vi={
          <p>
            Độ chính xác là núm vặn còn lại. Phần mở đầu của sách (p. 183) nói mã stencil thường cần dữ liệu độ chính xác cao, vốn tốn
            nhiều bộ nhớ trên chip hơn khi phân ô, và §8.1 (pp. 184–185) thêm rằng độ chính xác kép còn tốn băng thông bộ nhớ. Thiết
            lập 8 byte của trình khám phá cho thấy cả hai.
          </p>
        }
      >
        <p>
          Precision is the other knob. The book's opening (p. 183) says stencil codes tend to need high-precision data,
          which takes more on-chip memory when tiled, and §8.1 (pp. 184–185) adds that double precision also costs memory
          bandwidth. The explorer's 8-byte setting shows both.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch08-facts.md, A7)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Sách nói độ chính xác đơn và bán chính xác có thông lượng số học "cao hơn nhiều" so với double. Bảng của NVIDIA cho tỉ lệ
              FP32 so với FP64 từ 2:1 (compute capability 8.0, 9.0 và 10.0) đến 64:1, nên nó tùy phần cứng. Metal không có{' '}
              <code>double</code>.
            </p>
          }
        >
          <p>
            The book says single and half precision have "much higher" arithmetic throughput than double. NVIDIA's table
            gives FP32 to FP64 ratios from 2:1 (compute capabilities 8.0, 9.0 and 10.0) to 64:1, so it depends on the part.
            Metal has no <code>double</code>.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS84['ch08/8.4/q-double']} />

      <h2>
        <Translated vi="Một warp nạp khối lập phương ra sao">How one warp loads the cube</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lời phàn nàn thứ hai của sách về một khối lập phương nhỏ là hợp nhất truy cập bộ nhớ (coalescing). Trong một khối 8 × 8 ×
            8, một warp 32 luồng là bốn hàng 8 luồng (câu hỏi đầu tiên của chương), nên một lệnh nạp đọc bốn dải bộ nhớ ngắn cách
            nhau N phần tử. Khối rộng 32 sửa điều đó: một warp đọc một hàng 32 số float liền kề. Sách gợi ý 32 × 8 × 4 và lưu ý rằng ô
            đầu ra, và do đó tỉ lệ, vẫn nhỏ.
          </p>
        }
      >
        <p>
          The book's second complaint about a small cube is coalescing. In an 8 × 8 × 8 block, a warp of 32 threads is four
          rows of 8 (the first question of the chapter), so one load instruction reads four short runs of memory N elements
          apart. Blocks 32 wide fix that: a warp reads one row of 32 contiguous floats. The book suggests 32 × 8 × 4 and notes
          that the output tile, and so the ratio, stays small.
        </p>
      </Translated>
      <Source section="8.4" pages="193" />
      <Translated
        vi={
          <p>
            NVIDIA ghi nhận một mô hình bạn có thể dùng để đếm: lần nạp của một warp được phục vụ bằng các giao dịch 32 byte, và thước
            đo của hợp nhất truy cập là số byte dùng trên số byte lấy về. Hãy chọn một hình dạng và một warp, rồi đếm.
          </p>
        }
      >
        <p>
          NVIDIA documents a model you can count with: a warp's load is served in 32-byte transactions, and the measure of
          coalescing is bytes used over bytes fetched. Pick a shape and a warp, and count.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Chỉ đơn vị 32 byte của NVIDIA, một mảng được căn chỉnh, N là bội số của 8 để mọi hàng bắt đầu ở cùng mức căn chỉnh. Không
              L1, không L2, không trang DRAM, không dùng lại giữa các warp. Việc bốn hàng cách nhau N số float có tốn hơn bốn đoạn kề
              nhau hay không thì không được ghi nhận; các số đếm ở đây là phép tính của chúng tôi và chưa có gì đo chúng.
            </p>
          }
        >
          <p>
            NVIDIA's 32-byte unit only, an aligned array, N a multiple of 8 so every row starts at the same alignment. No L1,
            no L2, no DRAM pages, no reuse between warps. Whether four rows N floats apart cost more than four adjacent
            segments is not documented; the counts here are our arithmetic and nothing has measured them yet.
          </p>
        </Translated>
      </Simplification>
      <WarpExplorer />
      <Translated
        vi={
          <p>
            Theo mô hình này, lần nạp của khối lập phương đạt một nửa hiệu suất chứ không chỉ đơn giản là "không được hợp nhất": mỗi
            hàng 8 phần tử bắt đầu ở <code>6·blockIdx.x − 1</code>,
            một chỉ số lẻ, nên nó vắt qua hai đoạn. Khối rộng 32 của sách đọc một hàng mỗi warp, nhưng hàng đó bắt đầu ở{' '}
            <code>30·blockIdx.x − 1</code>, cũng lẻ, nên nó chạm năm đoạn cho lượng dữ liệu của bốn đoạn: 80%, không phải 100%. Đặt
            độ lệch về 0 và bạn thấy điều quan trọng là nơi một hàng bắt đầu: chỉ khi bắt đầu ở bội số của 8 số float mới cho bốn
            đoạn, và <code>30·blockIdx.x</code> là bội số của 8 chỉ ở mỗi khối thứ tư. (Kernel cần −1 để nạp vùng biên; giá trị 0
            chỉ để so sánh.)
          </p>
        }
      >
        <p>
          By this model the cube's load is half efficient rather than simply "not coalesced": each row of 8 starts at <code>6·blockIdx.x − 1</code>,
          an odd index, so it straddles two segments. The book's 32-wide block reads one row per warp, but that row starts
          at <code>30·blockIdx.x − 1</code>, also odd, so it touches five segments for four segments' worth of data: 80%,
          not 100%. Set the offset to 0 and you see that what matters is where a row starts: only a start on a multiple of
          8 floats gives four segments, and <code>30·blockIdx.x</code> is a multiple of 8 only in every fourth block. (The
          kernel needs its −1 to load the halo; the 0 is there only for the comparison.)
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch08-facts.md, A2, B2, C4)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              NVIDIA: giao dịch 32 byte, "tối đa hóa tỉ lệ số byte dùng trên số byte truyền". AMD chỉ mô tả hợp nhất truy cập bằng lời
              (burst, cache line) và không cho kích thước. Apple không ghi nhận mô hình hợp nhất truy cập nào; lời khuyên gần nhất của
              họ là làm threadgroup rộng bằng <code>threadExecutionWidth</code>.
            </p>
          }
        >
          <p>
            NVIDIA: 32-byte transactions, "maximize the ratio of bytes used to bytes transferred". AMD describes coalescing
            only in words (bursts, cache lines) and gives no size. Apple documents no coalescing model; its nearest advice
            is to make a threadgroup <code>threadExecutionWidth</code> wide.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS84['ch08/8.4/q-transactions']} label={tr('Work it out', 'Tự tính ra')} />
      <Ask q={questionsS84['ch08/8.4/q-x32']} />

      <h2>
        <Translated vi="Cùng một kernel trên ba backend">The same kernel on three backends</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Kernel phân ô chuyển sang từng dòng một, nhưng những con số định hình nó thì không. Bảng tập hợp những gì trang này đã dựa
            vào.
          </p>
        }
      >
        <p>
          The tiled kernel ports line for line, but the numbers that shaped it do not. The table collects what this page
          relied on.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. Mọi mục HIP và Metal đều lấy từ tài liệu của nhà cung cấp như đã kiểm tra ngày 2026-11-20 (bảng
              "Where the backends really differ" trong notes/today/ch08-facts.md). Chưa phần nào trong đó được chạy trên phần cứng CUDA
              hoặc AMD ở đây.
            </p>
          }
        >
          <p>
            The book uses CUDA only. Every HIP and Metal entry is from vendor documentation as checked on 2026-11-20 (the
            table "Where the backends really differ" in notes/today/ch08-facts.md). None of it has run on CUDA or AMD
            hardware here.
          </p>
        </Translated>
      </Beyond>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-table="backends">
          <thead>
            <tr>
              <th>
                <Translated vi="Mục">What</Translated>
              </th>
              <th>CUDA</th>
              <th>HIP</th>
              <th>Metal</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <Translated vi="Số luồng mỗi khối">Threads per block</Translated>
              </td>
              <td>
                <Translated vi="1024 trên mọi compute capability được liệt kê">1024 on every listed compute capability</Translated>
              </td>
              <td>1024</td>
              <td>
                <Translated vi={<>1024 trên lý thuyết; giới hạn thực tính theo từng pipeline đã biên dịch, <code>maxTotalThreadsPerThreadgroup</code></>}>
                  1024 in theory; the real cap is per compiled pipeline, <code>maxTotalThreadsPerThreadgroup</code>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>
                <Translated vi="Giới hạn theo từng chiều">Per-dimension caps</Translated>
              </td>
              <td>1024 × 1024 × 64</td>
              <td>
                <Translated vi="1024 theo mỗi chiều">1024 in each dimension</Translated>
              </td>
              <td>
                <Translated vi={<><code>maxThreadsPerThreadgroup</code>; giá trị không được ghi nhận</>}>
                  <code>maxThreadsPerThreadgroup</code>; values not documented
                </Translated>
              </td>
            </tr>
            <tr>
              <td>
                <Translated vi="10 × 10 × 10 có hợp lệ không?">Is 10 × 10 × 10 legal?</Translated>
              </td>
              <td>
                <Translated vi="có">yes</Translated>
              </td>
              <td>
                <Translated vi="có">yes</Translated>
              </td>
              <td>
                <Translated vi="chỉ khi giới hạn của pipeline ít nhất là 1000">only if the pipeline's cap is at least 1000</Translated>
              </td>
            </tr>
            <tr>
              <td>
                <Translated vi="Ô">The tile</Translated>
              </td>
              <td><code>__shared__ float in_s[8][8][8]</code></td>
              <td>
                <Translated vi="giống vậy">the same</Translated>
              </td>
              <td>
                <Translated vi={<><code>threadgroup float in_s[8][8][8]</code> trong thân kernel</>}>
                  <code>threadgroup float in_s[8][8][8]</code> in the kernel body
                </Translated>
              </td>
            </tr>
            <tr>
              <td>
                <Translated vi="Mô hình hợp nhất truy cập">Coalescing model</Translated>
              </td>
              <td>
                <Translated vi="được ghi nhận: giao dịch 32 byte">documented: 32-byte transactions</Translated>
              </td>
              <td>
                <Translated vi="chỉ định tính, không có kích thước">qualitative only, no size</Translated>
              </td>
              <td>
                <Translated vi="không được ghi nhận">not documented</Translated>
              </td>
            </tr>
            <tr>
              <td>
                <Translated vi="Các warp của một khối 8 × 8 × 8">Warps of an 8 × 8 × 8 block</Translated>
              </td>
              <td>
                <Translated vi="16 warp, mỗi warp 4 hàng 8 luồng">16 warps, each 4 rows of 8</Translated>
              </td>
              <td>
                <Translated vi="wave64: 8 wavefront mỗi cái 8 hàng; wave32: như CUDA">wave64: 8 wavefronts of 8 rows; wave32: as CUDA</Translated>
              </td>
              <td>
                <Translated vi="cách chia SIMD group do cài đặt quyết định">SIMD-group partition is implementation-defined</Translated>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>
        <Translated vi="Bạn đã gặp kernel này ở đâu">Where you have seen this kernel</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Kernel tích chập depthwise 3D trên CUDA của PyTorch, <code>conv_depthwise3d_cuda_kernel</code>, có hình dạng cơ bản của
            trang này: mỗi luồng tính các đầu ra lần lượt từng cái, đọc mọi tap thẳng từ bộ nhớ toàn cục với một phép kiểm tra biên
            mỗi tap, và không dùng bộ nhớ chia sẻ. Phân ô và gộp việc cho luồng (thread coarsening) của chương này là những bước một
            người đóng góp sẽ thử với nó; chúng có đem lại lợi ích ở đó hay không là một phép đo mà khóa học này chưa thực hiện.
          </p>
        }
      >
        <p>
          PyTorch's CUDA depthwise 3D convolution, <code>conv_depthwise3d_cuda_kernel</code>, has this page's basic shape:
          each thread computes outputs one at a time, reads every tap straight from global memory with a bounds test per
          tap, and uses no shared memory. The tiling and coarsening of this chapter are the moves a contributor would try on it; whether
          they pay off there is a measurement this course has not made.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch08-facts.md, D3)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Đọc trong <code>aten/src/ATen/native/cuda/DepthwiseConv3d.cu</code> trên nhánh mặc định của PyTorch (commit cuối
              2026-11-03). Lưới của nó bị chặn, nên mỗi luồng có thể lặp qua nhiều đầu ra cách nhau cả một lưới, vốn không chung đầu
              vào nào.
            </p>
          }
        >
          <p>
            Read in <code>aten/src/ATen/native/cuda/DepthwiseConv3d.cu</code> on PyTorch's default branch (last commit
            2026-11-03). Its grid is capped, so each thread may loop over several outputs a whole grid apart, which share
            no inputs.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Khối lập phương không thể lớn lên vì nó cần t³ luồng, mỗi phần tử một luồng, và vùng biên của nó chiếm phần lớn những gì
            nó nạp.{' '}
            <a href={href('ch08', '8.5')}>Trang 8.5</a> phá vỡ mối liên hệ giữa độ sâu của ô và số luồng.
          </p>
        }
      >
        <p>
          The cube cannot grow because it needs t³ threads, one per element, and its halo is most of what it loads.{' '}
          <a href={href('ch08', '8.5')}>Page 8.5</a> breaks the link between the tile's depth and the number of threads.
        </p>
      </Translated>
    </>
  );
}
