import { useEffect, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  NEIGHBOURS,
  PRESETS,
  inputsOf,
  presetFronts,
  stencil,
  stepAt,
  wavefronts,
  type NeighbourId,
  type PresetId,
} from '../../sim/wavefront';
import { FIG_16_5, INF, VERTICES, runSteps, stateAfter, stepTrace, type BlockEvent } from '../../sim/floydwarshall';
import { flagsAfter, frontMakespan, numbering, schedule, slotAt, type Rule, type StartOrder } from '../../sim/wavesched';
import { NeighboursScene, SCENE } from './NeighboursScene';

// Section 16.3 of the course: book §16.2 to §16.4 and §16.8 (pp. 374–381, 397–399).
// Four parts: top-down or bottom-up; numbering cells by step (pattern explorer, on
// src/sim/wavefront.ts); Floyd-Warshall in place (stepper, on src/sim/floydwarshall.ts);
// who waits for whom (scene and scheduler explorer, on src/sim/wavesched.ts); and one
// bridge, PyTorch's CTC loss. Every number stated comes from those models.

// The kernel and host loop as shown: both appear verbatim in labs/ch16/snippets/fw_bottomup.cu
// (snippets-s163.test.ts checks it, and pins the line numbers the stepper highlights).
const FW_KERNEL = `#define FW_INF (INT_MAX / 2)  // "no path"; FW_INF + FW_INF still fits in an int

__global__ void FW_bottomup(int k, int* dist, int n_vertices) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    bool inRange = col < n_vertices;   // no return before the barrier
    int row = blockIdx.y;
    int distIndex = n_vertices * row + col;

    __shared__ int dist_k_col;         // dist[row][k], the same for the whole block
    if (threadIdx.x == 0)
        dist_k_col = dist[n_vertices * row + k];
    __syncthreads();

    if (dist_k_col == FW_INF) return;  // no path from row to k: the block retires

    if (inRange) {
        int dist_k_row = dist[k * n_vertices + col];   // dist[k][col]
        if (dist_k_row != FW_INF) {
            int new_distance = dist_k_col + dist_k_row;
            if (new_distance < dist[distIndex])
                dist[distIndex] = new_distance;
        }
    }
}`;
const FW_HOST = `void floydWarshall(int* dist_d, int n_vertices, int threads) {
    dim3 dimBlock(threads);
    dim3 dimGrid((n_vertices + threads - 1) / threads, n_vertices);
    for (int k = 0; k < n_vertices; k++)
        FW_bottomup<<<dimGrid, dimBlock>>>(k, dist_d, n_vertices);  // in order: one stream
}`;
/** Lines of FW_KERNEL the stepper highlights. */
export const FW_LINES = { read: 11, barrier: 12, retire: 14, rowK: 17, compare: 20, write: 21 };
const FW_TIERS: Record<string, Tier> = {
  dist: 'global', dist_d: 'global', dist_k_col: 'shared',
  dist_k_row: 'reg', new_distance: 'reg', col: 'reg', row: 'reg', inRange: 'reg', distIndex: 'reg', k: 'reg',
};

// Numbers the page states, from the models.
const SW8 = presetFronts('smith-waterman', 8, 8);
const H264_8 = presetFronts('h264', 8, 8);
const V = (name: string) => VERTICES.indexOf(name as (typeof VERTICES)[number]);
const FW_A = stepTrace(FIG_16_5, V('a'));
const FW_C = stepTrace(runSteps(FIG_16_5, 2), V('c'));
const FW_C_COL = VERTICES.flatMap((w, i) => (FW_C.before[i][V('c')] === INF || i === V('c') ? [] : [`${w} ${FW_C.before[i][V('c')]}`]));
const cellText = (c: { i: number; j: number; to: number }) => tr(`${VERTICES[c.i]} to ${VERTICES[c.j]} = ${c.to}`, `${VERTICES[c.i]} đến ${VERTICES[c.j]} = ${c.to}`);
const listAnd = (xs: string[]) => (xs.length < 2 ? xs.join('') : xs.slice(0, -1).join(', ') + tr(' and ', ' và ') + xs[xs.length - 1]);
const F2 = SCENE.front2.makespan!;
const C2 = SCENE.counter2.makespan!;
// q-enough-sms asks about a grid the explorer cannot show (it stops at T = 8 and S = 8).
const BIG_T = 10;
const BIG_S = 16;
const F_BIG = frontMakespan(BIG_T, BIG_S);
const C_BIG = schedule(BIG_T, BIG_S, 'counter').makespan!;

export const questionsS163: Record<string, Question> = {
  'ch16/16.3/q-topdown': {
    kind: 'choice',
    id: 'ch16/16.3/q-topdown',
    prompt: (
      <Translated
        vi={
          <>
            Một đồng nghiệp chuyển một bộ giải đệ quy có memo sang GPU: một luồng cho mỗi bài toán con, kết quả được lưu đệm
            trong một bảng băm ở bộ nhớ toàn cục (global memory). Lý do mạnh nhất để chương này đi từ dưới lên là gì?
          </>
        }
      >
      <>
        A colleague ports a memoised recursive solver to the GPU: one thread per sub-problem, results cached in a hash
        table in global memory. What is the strongest reason the chapter goes bottom up instead?
      </>
      </Translated>
    ),
    options: [
      {
        label:
          <Translated vi={"Lần tra cứu của các luồng kề nhau rơi vào những chỗ không liên quan trong bảng, nên các lần nạp của một warp không được hợp nhất, và phép đệ quy chỉ ấn định thứ tự lúc chạy."}>{"Neighbouring threads' lookups land at unrelated places in the table, so a warp's loads do not coalesce, and the recursion fixes the order only at run time."}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Mã GPU không gọi được hàm: mọi hàm device đều được inline, nên một lời gọi đệ quy không có gì để biên dịch thành."}>{'GPU code cannot call functions: every device function is inlined, so a recursive call has nothing to compile into.'}</Translated>,
        why: <Translated vi={"Hàm device có thể được gọi, và có thể đệ quy trên CUDA và HIP; Metal cho phép đệ quy trong kernel từ Metal 2.4."}>{'Device functions can be called, and can recurse on CUDA and HIP; Metal allows recursion in kernels from Metal 2.4.'}</Translated>,
      },
      {
        label: <Translated vi={"GPU không có bảng băm: mã device không có container kết hợp, nên bảng memo sẽ phải nằm trên host."}>{'GPUs have no hash tables: device code has no associative container, so the memo table would have to stay on the host.'}</Translated>,
        why: <Translated vi={"Bảng băm là một mảng cộng một ít phép tính; kernel nào cũng dựng được. Vấn đề là các lần tra cứu của nó rơi vào đâu."}>{'A hash table is an array and some arithmetic; any kernel can build one. The trouble is where its lookups land.'}</Translated>,
      },
      {
        label: <Translated vi={"Đệ quy bị cấm trong mã device trên cả CUDA, HIP và Metal, nên một bộ giải đệ quy không biên dịch được trên bất kỳ backend nào."}>{'Recursion is forbidden in device code on CUDA, HIP and Metal alike, so a recursive solver does not compile on any of them.'}</Translated>,
        why: <Translated vi={"Trên CUDA chỉ riêng kernel (hàm __global__) là không được đệ quy. Hàm device thì được, trên CUDA và HIP với một stack có kích thước do bạn đặt, và Metal cho phép đệ quy từ Metal 2.4."}>{'On CUDA only the kernel itself (the __global__ function) may not recurse. Device functions may, on CUDA and HIP with a stack whose size you set, and Metal allows recursion from Metal 2.4.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cách từ dưới lên thay thế cả hai vấn đề. Một luồng tính địa chỉ ô của nó từ chỉ số của nó, nên các luồng kề nhau đọc
            các ô kề nhau, và thứ tự các bước được ấn định trước khi kernel bắt đầu. Lý do đầu tiên của sách, rằng GPU xử lý kém
            các chuỗi lời gọi sâu, là về chi phí: NVIDIA ghi nhận một stack cho mỗi luồng và AMD một stack cho mỗi warp; AMD và
            Apple cảnh báo về kích thước của nó; không ai cấm đệ quy trong hàm device.
          </p>
        }
      >
      <p>
        Bottom up replaces both problems. A thread computes the address of its cell from its index, so neighbouring
        threads read neighbouring cells, and the order of the steps is fixed before the kernel starts. The book's first
        reason, that GPUs handle deep chains of calls poorly, is about cost: NVIDIA documents a stack per thread and AMD
        one per warp; AMD and Apple warn about its size; none forbids recursion in device functions.
      </p>
      </Translated>
    ),
    source: 'ch. 16 §16.2, pp. 374–376; ch16-facts A1, B1, C1', // i18n-ignore: citation
  },
  'ch16/16.3/q-h264': {
    kind: 'numeric',
    id: 'ch16/16.3/q-h264',
    prompt: (
      <Translated
        vi={
          <>
            Một bảng 8 × 8 trong đó mỗi ô cần ô bên trái và ô chéo phía trên bên phải, mẫu H.264 của Fig. 16.2(d). Nó cần bao
            nhiêu bước được ngăn cách bởi rào chắn? Hãy tính trước khi đặt trình khám phá sang H.264.
          </>
        }
      >
      <>
        An 8 × 8 table in which each cell needs its left neighbour and the cell above and to the right of it, the H.264
        pattern of Fig. 16.2(d). How many barrier-separated steps does it take? Work it out before you set the explorer
        to H.264.
      </>
      </Translated>
    ),
    answer: H264_8.steps,
    unit: 'steps',
    vi: { unit: 'bước' },
    explain: (
      <Translated
        vi={
          <p>
            Hàng 0 là một chuỗi: ô (0, c) chạy ở bước c. Ô (1, 0) cần (0, 1), xong ở bước 1, nên nó chạy ở bước 2, và mỗi hàng
            bắt đầu sau hàng trên hai bước: bước của (r, c) là c + 2r. Ô cuối, (7, 7), chạy ở bước 7 + 14 ={' '}
            {stepAt(H264_8, 7, 7)}, nên các bước là 0 đến {H264_8.steps - 1}: tất cả {H264_8.steps} bước, so với {SW8.steps} của
            stencil ba láng giềng trên cùng bảng, và không bao giờ rộng hơn {H264_8.widest} ô.
          </p>
        }
      >
      <p>
        Row 0 is a chain: cell (0, c) runs at step c. Cell (1, 0) needs (0, 1), done at step 1, so it runs at step 2, and
        every row starts two steps after the row above: the step of (r, c) is c + 2r. The last cell, (7, 7), runs at step
        7 + 14 = {stepAt(H264_8, 7, 7)}, so the steps are 0 to {H264_8.steps - 1}: {H264_8.steps} of them, against{' '}
        {SW8.steps} for the three-neighbour stencil on the same table, and never more than {H264_8.widest} cells wide.
      </p>
      </Translated>
    ),
    source: 'ch. 16 §16.3, pp. 376–377, Fig. 16.2(d)', // i18n-ignore: citation
  },
  'ch16/16.3/q-live': {
    kind: 'numeric',
    id: 'ch16/16.3/q-live',
    prompt: (
      <Translated
        vi={
          <>
            Bạn chỉ cần điểm số cuối cùng của một bảng L × L với stencil bắc, tây và tây bắc, không cần chính phép căn chỉnh.
            Ngoài mặt sóng đang được ghi, bao nhiêu mặt sóng trước đó phải còn nằm trong bộ nhớ?
          </>
        }
      >
      <>
        You need only the final score of an L × L table with the north, west and north-west stencil, not the alignment
        itself. Besides the front being written, how many earlier fronts must stay in memory?
      </>
      </Translated>
    ),
    answer: SW8.reach,
    unit: 'fronts',
    explain: (
      <Translated
        vi={
          <p>
            Bắc và tây nằm trên đường chéo phụ trước đó, tây bắc nằm trên đường chéo trước nữa, và không gì đọc lùi xa hơn ("đầu
            vào xa nhất" của trình khám phá ghi {SW8.reach}). Ba đường chéo phụ, mỗi đường tối đa L ô: bộ nhớ tăng theo L thay vì
            L². Sách giữ toàn bộ bảng của Smith-Waterman chỉ vì bước truy vết ngược, vốn khôi phục phép căn chỉnh, đi lùi qua
            bảng từ ô tốt nhất.
          </p>
        }
      >
      <p>
        North and west lie on the previous anti-diagonal, north-west on the one before it, and nothing reads further back
        (the explorer's "farthest input" reads {SW8.reach}). Three anti-diagonals of at most L cells: memory that grows
        like L instead of L². The book keeps Smith-Waterman's whole table only because the traceback, which recovers the
        alignment, walks back through it from the best cell.
      </p>
      </Translated>
    ),
    source: 'ch. 16 §16.3 and §16.5, pp. 377, 384',
    vi: { unit: 'mặt sóng', source: 'ch. 16 §16.3 và §16.5, pp. 377, 384' },
  },
  'ch16/16.3/q-fw-changes': {
    kind: 'numeric',
    id: 'ch16/16.3/q-fw-changes',
    prompt: (
      <Translated
        vi={
          <>
            Bắt đầu từ bảng của Fig. 16.5 như được in. Chạy bước k = a, rồi bước k = b. Bước k = c thay đổi bao nhiêu ô?
          </>
        }
      >
      <>
        Start from Fig. 16.5's table as printed. Run step k = a, then step k = b. How many cells does step k = c change?
      </>
      </Translated>
    ),
    answer: FW_C.changed.length,
    unit: 'cells',
    explain: (
      <Translated
        vi={
          <p>
            Sau các bước a và b, cột c chứa {listAnd(FW_C_COL)}, và hàng c chỉ chứa c đến e = 10. Mỗi cặp hữu hạn cho một đường
            đi ứng viên qua c, và {FW_C.changed.length} trong số đó tốt hơn phần tử hiện tại:{' '}
            {listAnd(FW_C.changed.map(cellText))}. Cặp e, e sẽ cho 8 + 10 = 18, nhưng e đến e đã là 0. Các hàng{' '}
            {listAnd(FW_C.retired.map((i) => VERTICES[i]))} không có đường đến c, nên khối của chúng kết thúc sớm mà không đọc hàng c.
          </p>
        }
      >
      <p>
        After steps a and b, column c holds {listAnd(FW_C_COL)}, and row c holds only c to e = 10. Each finite pair gives
        a candidate path through c, and {FW_C.changed.length} of them beat the current entry:{' '}
        {listAnd(FW_C.changed.map(cellText))}. The pair e, e would give 8 + 10 = 18, but e to e is already 0. Rows{' '}
        {listAnd(FW_C.retired.map((i) => VERTICES[i]))} have no path to c, so their blocks retire without reading row c.
      </p>
      </Translated>
    ),
    source: 'ch. 16 §16.4, pp. 380–381, Fig. 16.5 (the table as printed); checked by floydwarshall.test.ts',
    vi: { unit: 'ô', source: 'ch. 16 §16.4, pp. 380–381, Fig. 16.5 (bảng như được in); được kiểm tra bởi floydwarshall.test.ts' },
  },
  'ch16/16.3/q-fw-inplace': {
    kind: 'choice',
    id: 'ch16/16.3/q-fw-inplace',
    prompt: (
      <Translated
        vi={
          <>
            Vì sao mọi luồng của bước k có thể cập nhật <code>dist</code> tại chỗ trong khi các luồng khác đang đọc hàng k và
            cột k của cùng bảng đó?
          </>
        }
      >
      <>
        Why can every thread of step k update <code>dist</code> in place while other threads are reading row k and
        column k of the same table?
      </>
      </Translated>
    ),
    options: [
      {
        label:
          <Translated vi={"Với dist[k][k] = 0, một ô của hàng k hoặc cột k được cập nhật thành chính giá trị của nó, nên không gì mà luồng khác đọc bị thay đổi trong bước đó."}>{'With dist[k][k] = 0, a cell of row k or column k is updated to its own value, so nothing that another thread reads changes during the step.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Các luồng của một khối chạy đồng bộ khóa bước (lockstep), nên mọi luồng đọc hàng k và cột k trước khi bất kỳ luồng nào của khối đến được lần ghi của nó."}>{'The threads of a block run in lockstep, so every thread reads row k and column k before any thread of the block gets as far as its own write.'}</Translated>,
        why: <Translated vi={"Các khối khác cũng đọc hàng k, và không gì sắp thứ tự các khối trong một lần khởi chạy."}>{'Different blocks read row k too, and nothing orders blocks within a launch.'}</Translated>,
      },
      {
        label: <Translated vi={"Các lần khởi chạy trong một stream có thứ tự, nên bước k + 1 thấy mọi lần ghi của bước k, bất kể các luồng của bước k chạy theo thứ tự nào."}>{'Launches in one stream are ordered, so step k + 1 sees every write of step k, whatever order the threads of step k ran in.'}</Translated>,
        why: <Translated vi={"Đúng, nhưng điều đó sắp thứ tự các bước. Câu hỏi là về các luồng bên trong một bước."}>{'True, but that orders the steps. The question is about threads within one step.'}</Translated>,
      },
      {
        label: <Translated vi={"min có tính giao hoán và kết hợp, nên dù các luồng cập nhật theo thứ tự nào, mọi ô đều kết thúc với cùng giá trị nhỏ nhất."}>{'min is commutative and associative, so whatever order the threads update in, every cell ends with the same minimum.'}</Translated>,
        why: <Translated vi={"Thứ tự cập nhật trở nên quan trọng ngay khi một giá trị mà ai đó đọc có thể thay đổi. Lập luận phải chỉ ra rằng điều đó không thể xảy ra."}>{'The order of updates matters as soon as a value someone reads can change. The argument has to show that it cannot.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lập luận cần cả hai điều kiện sách nêu: dist[k][k] = 0 và không có trọng số âm. Phá điều kiện thứ nhất (công tắc −1
            của trình chạy từng bước) thì hàng k và cột k thay đổi trong bước k, nên điều một khối đọc phụ thuộc vào việc khối k
            đã chạy hay chưa: kết quả phụ thuộc vào thứ tự. Khi đó bạn cần hai buffer, đọc bảng cũ và ghi bảng mới (double
            buffering, chương 6).
          </p>
        }
      >
      <p>
        The argument needs both conditions the book states: dist[k][k] = 0 and no negative weights. Break the first (the
        stepper's −1 switch) and row k and column k change during step k, so what a block reads depends on whether block k
        has already run: the result depends on the order. Then you need two buffers, reading the old table and writing a
        new one (double buffering, chapter 6).
      </p>
      </Translated>
    ),
    source: 'ch. 16 §16.4, pp. 379–380', // i18n-ignore: citation
  },
  'ch16/16.3/q-infinity': {
    kind: 'choice',
    id: 'ch16/16.3/q-infinity',
    prompt: (
      <Translated
        vi={
          <>
            Fig. 16.4 như được in kiểm tra <code>dist_k_col == INFINITY</code>, trong đó <code>dist_k_col</code> là một{' '}
            <code>int</code> và <code>INFINITY</code> đến từ <code>&lt;cmath&gt;</code>. Chuyện gì xảy ra?
          </>
        }
      >
      <>
        Fig. 16.4 as printed tests <code>dist_k_col == INFINITY</code>, where <code>dist_k_col</code> is an{' '}
        <code>int</code> and <code>INFINITY</code> comes from <code>&lt;cmath&gt;</code>. What happens?
      </>
      </Translated>
    ),
    options: [
      {
        label:
          <Translated vi={"int trở thành float, và không int nào bằng vô cực, nên các lối thoát sớm không bao giờ kích hoạt; với INT_MAX làm \"không có cạnh\", tổng sau đó bị tràn số."}>{'The int becomes a float, and no int equals infinity, so the exits never fire; with INT_MAX as "no edge", the sum then overflows.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Nó hoạt động: INFINITY chuyển thành int lớn nhất, nên phép kiểm tra khớp với các phần tử lưu INT_MAX cho \"không có cạnh\"."}>{'It works: INFINITY converts to the largest int, so the test matches entries that store INT_MAX for "no edge".'}</Translated>,
        why: <Translated vi={"Phép chuyển đổi đi theo chiều ngược lại. Khi một int gặp một float trong phép so sánh, int trở thành float."}>{'The conversion goes the other way. When an int meets a float in a comparison, the int becomes a float.'}</Translated>,
      },
      {
        label: <Translated vi={"Các lối thoát kích hoạt quá thường xuyên: khoảng cách lớn bị làm tròn thành vô cực dưới dạng float, nên những khối có đường đi vẫn kết thúc sớm."}>{'The exits fire too often: large distances round to infinity as floats, so blocks that do have a path retire.'}</Translated>,
        why: <Translated vi={"Ngay cả INT_MAX cũng chuyển thành một float hữu hạn. Không int nào so sánh bằng dương vô cực, nên các lối thoát không bao giờ kích hoạt."}>{'Even INT_MAX converts to a finite float. No int compares equal to positive infinity, so the exits never fire at all.'}</Translated>,
      },
      {
        label: <Translated vi={"Không sao miễn là trọng số nhỏ, vì khi đó không tổng nào của hai phần tử chạm được giới hạn của int."}>{'It is fine as long as the weights are small, because then no sum of two entries can reach the int limit.'}</Translated>,
        why: <Translated vi={"Phép kiểm tra không bao giờ đúng, bất kể trọng số, nên một phần tử \"không có cạnh\" vẫn đi vào tổng."}>{'The test can never be true, whatever the weights, so a "no edge" entry still goes into the sum.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong phép so sánh một <code>int</code> với một <code>float</code>, <code>int</code> được chuyển thành{' '}
            <code>float</code>. <code>INFINITY</code> là một hằng <code>float</code>, dương vô cực, và không{' '}
            <code>int</code> nào chuyển thành nó. Vậy các khối lẽ ra phải kết thúc sớm vẫn làm việc, và{' '}
            <code>dist_k_col + dist_k_row</code> cộng bất cứ giá trị nào đại diện cho "không có cạnh". Với <code>INT_MAX</code>,
            tổng đó tràn số, là hành vi không xác định. Kernel đã sửa dùng giá trị canh số nguyên{' '}
            <code>FW_INF = INT_MAX / 2</code>, nên <code>FW_INF + FW_INF</code> vẫn vừa, và so sánh với nó. Một hệ quả phụ đáng
            để ý: với giá trị canh này, một tổng có chứa <code>FW_INF</code> không bao giờ nhỏ hơn bất kỳ phần tử nào, nên các
            lối thoát chỉ tiết kiệm công việc.
          </p>
        }
      >
      <p>
        In a comparison of an <code>int</code> with a <code>float</code>, the <code>int</code> is converted to{' '}
        <code>float</code>. <code>INFINITY</code> is a <code>float</code> constant, positive infinity, and no{' '}
        <code>int</code> converts to it. So blocks that should retire do the work anyway, and{' '}
        <code>dist_k_col + dist_k_row</code> adds whatever stands for "no edge". With <code>INT_MAX</code> that sum
        overflows, which is undefined behaviour. The corrected kernel uses the integer sentinel{' '}
        <code>FW_INF = INT_MAX / 2</code>, so <code>FW_INF + FW_INF</code> still fits, and compares with it. A side
        effect worth noticing: with this sentinel a sum that involves <code>FW_INF</code> is never smaller than any entry,
        so the exits only save work.
      </p>
      </Translated>
    ),
    source: 'ch. 16 Fig. 16.4, p. 380 (notes/ch16.md, "Unclear in the PDF": INFINITY is undefined in the listing); ch16-facts A3',
    vi: { source: 'ch. 16 Fig. 16.4, p. 380 (notes/ch16.md, "Unclear in the PDF": INFINITY không được định nghĩa trong mã in); ch16-facts A3' },
  },
  'ch16/16.3/q-ex1': {
    kind: 'numeric',
    id: 'ch16/16.3/q-ex1',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 1 của sách. Chạy Floyd-Warshall với các khối 16 × 16: mỗi khối phủ 16 hàng và 16 cột của bảng, nạp 16 giá
            trị của cột k và 16 giá trị của hàng k vào bộ nhớ chia sẻ một lần, rồi mọi luồng cập nhật ô của nó từ đó. Khối thực
            hiện bao nhiêu lần đọc toàn cục hàng k hoặc cột k cho mỗi ô nó cập nhật? (Không tính lần đọc ô của chính mỗi luồng.)
          </>
        }
      >
      <>
        The book's exercise 1. Run Floyd-Warshall with 16 × 16 blocks: each block covers 16 rows and 16 columns of the
        table, loads its 16 values of column k and its 16 values of row k into shared memory once, and then every thread
        updates its cell from there. How many global reads of row k or column k does the block make per cell it updates?
        (Do not count each thread's read of its own cell.)
      </>
      </Translated>
    ),
    answer: 32 / 256,
    tol: 0.01,
    unit: 'reads per cell',
    explain: (
      <Translated
        vi={
          <p>
            32 lần đọc cho 256 ô: 0.125 mỗi ô, mỗi giá trị được dùng 16 lần. Kernel của sách với khối 256 luồng dọc một hàng thực
            hiện 1 lần đọc dist[row][k] và 256 lần đọc hàng k cho 256 ô, khoảng 1.004 mỗi ô: giá trị cột được cả khối chia sẻ,
            các giá trị hàng thì không. Có hai điều phép đếm bỏ qua. 16 giá trị cột nằm ở 16 hàng khác nhau, cách nhau{' '}
            <code>n_vertices</code>, nên các lần đọc đó không được hợp nhất. Và việc kết thúc sớm trở thành một phép kiểm tra
            theo từng luồng, vì một khối không còn phủ một hàng duy nhất. Kernel nào nhanh hơn trên GPU của bạn là chuyện đo
            đạc, không phải điều phép đếm này quyết định được.
          </p>
        }
      >
      <p>
        32 reads for 256 cells: 0.125 per cell, each value used 16 times. The book's kernel with 256-thread blocks along
        one row makes 1 read of dist[row][k] and 256 reads of row k for 256 cells, about 1.004 per cell: the column value
        is shared by the block, the row values are not. Two things the count leaves out. The 16 column values lie in 16
        different rows, <code>n_vertices</code> apart, so those reads do not coalesce. And the early retirement becomes a
        per-thread test, since a block no longer covers a single row. Which kernel is faster on your GPU is a
        measurement, not something this count settles.
      </p>
      </Translated>
    ),
    source: 'ch. 16 §16.10, exercise 1, p. 399; the count is ours (notes/ch16.md, Exercises)',
    vi: { unit: 'lần đọc mỗi ô', source: 'ch. 16 §16.10, bài tập 1, p. 399; phép đếm là của khóa học này (notes/ch16.md, Exercises)' },
  },
  'ch16/16.3/q-enough-sms': {
    kind: 'numeric',
    id: 'ch16/16.3/q-enough-sms',
    prompt: (
      <Translated
        vi={
          <>
            Một lưới {BIG_T} × {BIG_T} ô trên {BIG_S} SM, mỗi ô mất một bước thời gian. Bắt đầu mỗi ô theo hai cờ của nó tiết
            kiệm được bao nhiêu bước thời gian so với khởi chạy lại mỗi mặt sóng?
          </>
        }
      >
      <>
        A {BIG_T} × {BIG_T} grid of tiles on {BIG_S} SMs, each tile taking one time step. How many time steps does
        starting each tile on its two flags save over a relaunch per front?
      </>
      </Translated>
    ),
    answer: F_BIG - C_BIG,
    unit: 'steps',
    explain: (
      <Translated
        vi={
          <p>
            Không bước nào. Mỗi mặt sóng có tối đa {BIG_T} ô, nên với {BIG_S} SM mỗi mặt sóng chạy trong một bước và khởi chạy lại
            mỗi mặt sóng mất {F_BIG} bước, 2T − 1. Cờ cũng mất {C_BIG}, vì không lịch chạy nào thắng được chuỗi ô từ góc này đến
            góc kia, cũng dài 2T − 1. Cờ chỉ có lợi ở giữa hai thái cực: khi máy hẹp hơn mặt sóng rộng nhất, nhưng không hẹp đến
            mức mọi bước đều đầy (với một SM cả hai đều mất T² bước).
          </p>
        }
      >
      <p>
        None. Every front has at most {BIG_T} tiles, so with {BIG_S} SMs each front runs in one step and a relaunch per
        front takes {F_BIG} steps, 2T − 1. Flags take {C_BIG} as well, because no schedule can beat the chain of tiles
        from the corner to the corner, also 2T − 1 long. Flags pay off only between the extremes: when the machine is narrower than the
        widest front, but not so narrow that every step is full anyway (with one SM both take T² steps).
      </p>
      </Translated>
    ),
    source: 'ch. 16 §16.8, pp. 397–399, Fig. 16.21; checked by wavesched.test.ts',
    vi: { unit: 'bước', source: 'ch. 16 §16.8, pp. 397–399, Fig. 16.21; được kiểm tra bởi wavesched.test.ts' },
  },
  'ch16/16.3/q-row-owner': {
    kind: 'choice',
    id: 'ch16/16.3/q-row-owner',
    prompt: (
      <Translated
        vi={
          <>
            Sách giao cho mỗi khối thường trú một hàng ô theo <code>blockIdx.x</code>, và cho nó quay vòng chờ cờ của ô phía trên
            trước mỗi ô. Dưới một lần khởi chạy CUDA thông thường, điều gì sai?
          </>
        }
      >
      <>
        The book gives each persistent block one row of tiles by <code>blockIdx.x</code>, and has it spin on the flag of
        the tile above before each tile. Under an ordinary CUDA launch, what is wrong?
      </>
      </Translated>
    ),
    options: [
      {
        label:
          <Translated vi={"Không gì bảo đảm khối sở hữu hàng phía trên đã bắt đầu, và một khối đang quay vòng chờ giữ SM của nó: nếu các hàng sau bắt đầu trước, không gì kết thúc."}>{'Nothing promises that the block owning the row above has started, and a spinning block keeps its SM: if later rows start first, nothing finishes.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Cờ phải được khai báo volatile, nếu không trình biên dịch giữ nó trong thanh ghi và khối đang quay vòng không bao giờ thấy nó được bật."}>{'The flag must be declared volatile, or the compiler keeps it in a register and the spinning block never sees it set.'}</Translated>,
        why: <Translated vi={"Nó phải được ghi và đọc như một biến nguyên tử ở phạm vi device với release và acquire, điều mà volatile không làm. Và một cờ đúng cũng không giúp gì nếu khối bật nó không bao giờ chạy."}>{'It must be written and read as a device-scope atomic with release and acquire, which volatile is not. And a correct flag does not help if the block that sets it never runs.'}</Translated>,
      },
      {
        label: <Translated vi={"Không có gì sai: các khối bắt đầu theo thứ tự blockIdx, nên hàng phía trên đang chạy trước khi hàng dưới bắt đầu chờ nó."}>{'Nothing: blocks start in blockIdx order, so the row above is running before the row below begins to wait on it.'}</Translated>,
        why: <Translated vi={"Không có thứ tự lập lịch nào được bảo đảm: các khối có thể chạy theo thứ tự bất kỳ (chương 4)."}>{'No scheduler ordering is promised: blocks may run in any order (chapter 4).'}</Translated>,
      },
      {
        label: <Translated vi={"Không có gì sai, miễn là lưới không có nhiều khối hơn số SM của GPU, vì khi đó mọi khối có một SM riêng và không khối nào phải chờ để bắt đầu."}>{'Nothing, as long as the grid has no more blocks than the GPU has SMs, since then every block gets an SM of its own and none is left waiting to start.'}</Translated>,
        why: <Translated vi={"Một lưới vừa khít là điều kiện cho một lần khởi chạy cooperative, và chỉ lần khởi chạy đó mới bảo đảm mọi khối đều tiến triển. Dưới một lần khởi chạy thông thường, không gì đảm bảo điều đó."}>{'A grid that fits is the condition for a cooperative launch, and only that launch promises that every block makes progress. Under an ordinary launch nothing backs it.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hai cách sửa được tài liệu xác nhận an toàn trên CUDA: phát các ô từ một bộ đếm nguyên tử theo thứ tự mặt sóng, để
            mọi ô mà một khối chờ đã được nhận trước bởi một khối đang chạy, hoặc khởi chạy kernel theo kiểu cooperative. Trên
            HIP cùng mã nguồn biên dịch được, nhưng AMD không ghi nhận bảo đảm tiến triển nào giữa các khối, nên thiết kế là
            unverified ở đó. Metal hoàn toàn không ghi nhận bảo đảm tiến triển nào giữa các threadgroup; trên Metal, việc khởi
            chạy lại vẫn giữ nguyên.
          </p>
        }
      >
      <p>
        Two documented-safe fixes on CUDA: hand tiles out from an atomic counter in front order, so every tile a block
        waits on was claimed earlier by a block that is already running, or launch the kernel cooperatively. On HIP the
        same source compiles, but AMD documents no progress guarantee between blocks, so the design is unverified there.
        Metal documents no progress guarantee between threadgroups at all; on Metal the relaunch stays.
      </p>
      </Translated>
    ),
    source: 'ch. 16 §16.8, p. 398; ch16-facts A4, B3, C4; ch11-facts A6, C5; ch04-facts A5', // i18n-ignore: citation
  },
  'ch16/16.3/q-ctc': {
    kind: 'choice',
    id: 'ch16/16.3/q-ctc',
    prompt: (
      <Translated
        vi={
          <>
            CTC loss CUDA của PyTorch giao cho mỗi khối trọn các chuỗi và lặp theo thời gian với một <code>__syncthreads()</code>{' '}
            ở mỗi bước. Vì sao nó không bao giờ khởi chạy lại, hay dùng cờ giữa các khối?
          </>
        }
      >
      <>
        PyTorch's CUDA CTC loss gives each block whole sequences and loops over time with a <code>__syncthreads()</code>{' '}
        at every step. Why does it never relaunch, or use flags between blocks?
      </>
      </Translated>
    ),
    options: [
      {
        label:
          <Translated vi={"Bước t chỉ cần bước t − 1 của cùng chuỗi, và các chuỗi độc lập với nhau, nên toàn bộ phụ thuộc nằm gọn trong một khối."}>{'Step t needs only step t − 1 of the same sequence, and sequences are independent, so the whole dependency stays inside one block.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"CTC không có phụ thuộc giữa các bước thời gian: mỗi phần tử alpha đến từ đầu ra của mạng ở chính bước của nó."}>{'CTC has no dependency between time steps: each alpha entry comes from the network\'s outputs at its own step.'}</Translated>,
        why: <Translated vi={"alpha ở bước t cần ba phần tử của bước t − 1."}>{'alpha at step t needs three entries of step t − 1.'}</Translated>,
      },
      {
        label: <Translated vi={"Nó dùng phép toán nguyên tử thay cho rào chắn: mỗi luồng cộng phần của nó vào bước t + 1 ngay khi có."}>{'It uses atomics instead of a barrier: each thread adds its share to step t + 1 as soon as it has it.'}</Translated>,
        why: <Translated vi={"Vòng lặp có một rào chắn ở mỗi bước thời gian; đó chính là sự đồng bộ hóa."}>{'The loop has a barrier at every time step; that is the synchronisation.'}</Translated>,
      },
      {
        label: <Translated vi={"PyTorch chạy batch từng chuỗi một, mỗi lần khởi chạy một khối, nên rào chắn khối là sự đồng bộ hóa duy nhất nó có thể cần."}>{'PyTorch runs the batch one sequence at a time, one block per launch, so a block barrier is the only synchronisation it could need.'}</Translated>,
        why: <Translated vi={"Các khối xử lý các chuỗi khác nhau song song. Vòng lặp theo thời gian nằm bên trong mỗi khối."}>{'Blocks handle different sequences in parallel. The loop over time is inside each block.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Khi chuỗi đích rộng hơn khối, chính khối đó đi qua các đoạn vị trí theo thứ tự phụ thuộc, từ trái sang phải với
            alpha, nên nó vẫn không bao giờ chờ khối khác. Sự đồng bộ hóa rẻ nhất bao được phụ thuộc là rào chắn khối, và đó là
            thứ nó dùng. Kernel MPS làm tương tự với một threadgroup cho mỗi phần tử của batch.
          </p>
        }
      >
      <p>
        When the target is wider than the block, the same block walks over chunks of positions in dependency order, left
        to right for alpha, so it still never waits on another block. The cheapest synchronisation that covers the
        dependency is a block barrier, and that is what it uses. The MPS kernel does the same with one threadgroup per
        batch item.
      </p>
      </Translated>
    ),
    source: 'ch16-facts D2 (PyTorch LossCTC.cu and LossOps.metal, read 2026-11-20); the mapping to the chapter is ours',
    vi: { source: 'ch16-facts D2 (PyTorch LossCTC.cu và LossOps.metal, đọc ngày 2026-11-20); mối liên hệ với chương là của khóa học này' },
  },
};

// ================================================================ the pattern explorer
/** Lets a row of segment buttons wrap on a narrow screen instead of running off it. */
const WRAP = { flexWrap: 'wrap', rowGap: 4 } as const;

const SEL0: [number, number] = [4, 4];

function PatternExplorer() {
  const [preset, setPreset] = useState<PresetId | null>('smith-waterman');
  const [on, setOn] = useState<NeighbourId[]>(['N', 'W', 'NW']);
  const [rows, setRows] = useState(8);
  const [cols, setCols] = useState(8);
  const [sel, setSel] = useState<[number, number]>(SEL0);
  const f = useMemo(() => wavefronts(rows, cols, stencil(on)), [rows, cols, on]);
  const r0 = Math.min(sel[0], rows - 1);
  const c0 = Math.min(sel[1], cols - 1);
  const s0 = stepAt(f, r0, c0);
  const ins = inputsOf(f, r0, c0);

  const choose = (id: PresetId) => {
    const p = PRESETS.find((x) => x.id === id)!;
    setPreset(id);
    setOn([...p.neighbours]);
    setRows(p.rows);
    setCols(p.cols);
    setSel(p.rows === 1 ? [0, Math.floor(p.cols / 2)] : SEL0);
  };
  const toggle = (id: NeighbourId) => {
    const next = on.includes(id) ? on.filter((x) => x !== id) : [...on, id];
    setOn(next);
    const match = PRESETS.find((p) => p.neighbours.length === next.length && p.neighbours.every((n) => next.includes(n)));
    setPreset(match ? match.id : null);
  };

  const C = 30;
  const P = 8;
  const gw = cols * C + 2 * P;
  const gh = rows * C + 2 * P;
  const cx = (c: number) => P + c * C + C / 2;
  const cy = (r: number) => P + r * C + C / 2;
  const barW = Math.max(6, Math.min(26, Math.floor(440 / f.steps)));
  const barH = 90;
  const sizes = (
    <svg
      viewBox={`0 0 ${f.steps * barW + 40} ${barH + 44}`}
      role="img"
      aria-label={tr(`Cells per step: ${f.sizes.join(', ')}`, `Số ô mỗi bước: ${f.sizes.join(', ')}`)}
      style={{ width: '100%', maxWidth: f.steps * barW + 40, display: 'block' }}
    >
      {f.sizes.map((n, s) => {
        const h = (n / f.widest) * barH;
        return (
          <g key={s} data-bar={s} data-size={n}>
            <rect x={30 + s * barW + 1} y={18 + barH - h} width={barW - 2} height={h} fill="var(--ink)" fillOpacity={s === s0 ? 0.85 : 0.25} />
            {barW >= 16 ? (
              <text x={30 + s * barW + barW / 2} y={18 + barH - h - 3} fontSize="9.5" textAnchor="middle">{n}</text>
            ) : null}
          </g>
        );
      })}
      <text x={0} y={26} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{f.widest}</text>
      <text x={0} y={18 + barH} fontSize="10" style={{ fill: 'var(--ink-2)' }}>0</text>
      <text x={30} y={barH + 38} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>bước 0 đến {f.steps - 1}: số ô trong mỗi bước (đậm: bước của ô đang chọn)</>}>step 0 to {f.steps - 1}: cells in each step (dark: the selected cell's step)</Translated></text>
    </svg>
  );

  return (
    <Bench
      title={tr('Which cells can run together', 'Những ô nào có thể chạy cùng nhau')}
      task={tr('Pick a pattern or build one, then click a cell to see what it waits for.', 'Chọn một mẫu hoặc tự dựng một mẫu, rồi bấm vào một ô để xem nó chờ những gì.')}
      foot={<Translated vi={<>Mỗi số là bước mà ô có thể chạy: lớn hơn một so với bước muộn nhất trong các ô nó đọc. Càng đậm là càng muộn. Viền dày: ô đang chọn; viền nét đứt: các ô nó đọc; viền mảnh: các ô cùng bước với nó.</>}><>Each number is the step at which the cell can run: one more than the latest step among the cells it reads. Darker means later. Thick outline: the selected cell; dashed outline: the cells it reads; thin outline: the cells that share its step.</></Translated>}
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label={tr('Pattern', 'Mẫu')} style={WRAP}>
          {PRESETS.map((p) => (
            <button key={p.id} aria-pressed={preset === p.id} onClick={() => choose(p.id)} title={tr(p.figure, ({ 'smith-waterman': 'Hình 16.2(c)', checkerboard: 'Hình 16.2(a)', financial: 'Hình 16.2(b), theo cách đọc của chúng tôi', h264: 'Hình 16.2(d)', fibonacci: 'Phương trình 16.1, p. 374' } as Record<PresetId, string>)[p.id])}>
              <Translated vi={({ 'smith-waterman': 'Smith-Waterman', checkerboard: 'Bàn cờ', financial: 'Tài chính; cả CTC', h264: 'H.264', fibonacci: 'Fibonacci' } as Record<PresetId, string>)[p.id]}>{p.label}</Translated>
            </button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted"><Translated vi="mỗi ô đọc:">each cell reads:</Translated></span>
        {NEIGHBOURS.map((n) => (
          <label key={n.id} className="check">
            <input type="checkbox" checked={on.includes(n.id)} onChange={() => toggle(n.id)} data-neighbour={n.id} /> <Translated vi={({ N: 'trên', W: 'trái', NW: 'chéo trên trái', NE: 'chéo trên phải', NWW: 'trên, lùi hai cột trái', WW: 'lùi hai cột trái' } as Record<NeighbourId, string>)[n.id]}>{n.label}</Translated>
          </label>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi={"\"Số hàng\""}>
          Rows
          </Translated>
          <select value={rows} onChange={(e) => { setRows(+e.target.value); setPreset(null); }} aria-label={tr('Rows', 'Số hàng')}>
            {Array.from({ length: 16 }, (_, i) => i + 1).map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi={"\"Số cột\""}>
          Columns
          </Translated>
          <select value={cols} onChange={(e) => { setCols(+e.target.value); setPreset(null); }} aria-label={tr('Columns', 'Số cột')}>
            {Array.from({ length: 16 }, (_, i) => i + 1).map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <span className="muted">{preset ? tr(PRESETS.find((p) => p.id === preset)!.figure, ({ 'smith-waterman': 'Hình 16.2(c)', checkerboard: 'Hình 16.2(a)', financial: 'Hình 16.2(b), theo cách đọc của chúng tôi', h264: 'Hình 16.2(d)', fibonacci: 'Phương trình 16.1, p. 374' } as Record<PresetId, string>)[preset]) : tr('your own stencil', 'stencil của riêng bạn')}</span>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(16rem, 1fr))' }}>
        <div>
          <svg
            viewBox={`0 0 ${gw} ${gh}`}
            role="img"
            aria-label={tr(`A ${rows} by ${cols} table numbered by step`, `Một bảng ${rows} × ${cols} được đánh số theo bước`)}
            data-grid={`${rows}x${cols}`}
            style={{ width: '100%', maxWidth: Math.max(160, gw * 1.15), display: 'block' }}
          >
            {f.step.map((s, i) => {
              const r = Math.floor(i / cols);
              const c = i % cols;
              const isSel = r === r0 && c === c0;
              const same = s === s0;
              return (
                <g key={i} data-cell={`${r},${c}`} data-step={s} onClick={() => setSel([r, c])} style={{ cursor: 'pointer' }}>
                  <rect
                    x={P + c * C + 1}
                    y={P + r * C + 1}
                    width={C - 2}
                    height={C - 2}
                    rx="2"
                    fill="var(--ink)"
                    fillOpacity={0.04 + (0.32 * s) / Math.max(1, f.steps - 1)}
                    stroke={isSel || same ? 'var(--ink)' : 'var(--rule)'}
                    strokeWidth={isSel ? 2.8 : same ? 1.6 : 1}
                  />
                  <text x={cx(c)} y={cy(r) + 4} fontSize="11" textAnchor="middle">{s}</text>
                </g>
              );
            })}
            {ins.map(([r, c]) => (
              <rect
                key={`${r},${c}`}
                data-input={`${r},${c}`}
                x={P + c * C + 3.5}
                y={P + r * C + 3.5}
                width={C - 7}
                height={C - 7}
                rx="2"
                fill="var(--ink)"
                fillOpacity={0.14}
                stroke="var(--ink)"
                strokeWidth="2"
                strokeDasharray="4 2.5"
                pointerEvents="none" // i18n-ignore: SVG attribute value
              />
            ))}
          </svg>
          <p className="muted" style={{ margin: '0.4rem 0 0' }} data-selected>
            <Translated
              vi={
                <>
                  Ô ({r0}, {c0}) chạy ở bước {s0}.{' '}
                  {ins.length
                    ? `Nó đọc ${ins.map(([r, c]) => `(${r}, ${c}) ở bước ${stepAt(f, r, c)}`).join(', ')}.`
                    : 'Nó chỉ đọc giá trị biên.'}{' '}
                  {f.sizes[s0]} ô cùng bước với nó.
                </>
              }
            >
            Cell ({r0}, {c0}) runs at step {s0}.{' '}
            {ins.length
              ? `It reads ${ins.map(([r, c]) => `(${r}, ${c}) at step ${stepAt(f, r, c)}`).join(', ')}.`
              : 'It reads only boundary values.'}{' '}
            {f.sizes[s0]} cells share its step.
            </Translated>
          </p>
        </div>
        <div>
          {sizes}
          <table className="data" style={{ marginTop: '0.8rem' }}>
            <tbody>
              <tr><td><Translated vi="số bước (ngăn cách bởi rào chắn)">steps (barrier-separated)</Translated></td><td data-k="steps">{f.steps}</td></tr>
              <tr><td><Translated vi="bước rộng nhất">widest step</Translated></td><td data-k="widest"><Translated vi={<>{f.widest} ô</>}>{f.widest} cells</Translated></td></tr>
              <tr><td><Translated vi="số ô trung bình mỗi bước">cells per step on average</Translated></td><td data-k="mean">{f.mean.toFixed(2)}</td></tr>
              <tr><td><Translated vi="đầu vào xa nhất, lùi bao nhiêu mặt sóng">farthest input, fronts back</Translated></td><td data-k="reach">{f.reach}</td></tr>
              <tr>
                <td><Translated vi="hàng r + 1 bắt đầu sau hàng r">row r + 1 starts after row r by</Translated></td>
                <td data-k="lag">{f.rowLag === null ? tr('one row only', 'chỉ có một hàng') : tr(`${f.rowLag} step${f.rowLag === 1 ? '' : 's'}`, `${f.rowLag} bước`)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ================================================================ the Floyd-Warshall stepper
const N = VERTICES.length;
const HEAD = { background: 'transparent', border: 0, fontWeight: 650, color: 'var(--ink-2)' } as const;

function describeBlock(b: BlockEvent, k: number, before: number[][]): { text: string; stale: boolean } {
  const v = VERTICES[b.row];
  const kv = VERTICES[k];
  const head = tr(`Block for row ${v} (blockIdx.y = ${b.row}): thread 0 reads dist[${v}][${kv}] into the shared dist_k_col`, `Khối của hàng ${v} (blockIdx.y = ${b.row}): luồng 0 đọc dist[${v}][${kv}] vào biến chia sẻ dist_k_col`);
  if (b.retired)
    return {
      text: tr(`${head}. It is FW_INF: there is no path from ${v} to ${kv}, so nothing in this row can improve. After the barrier the whole block returns.`, `${head}. Nó là FW_INF: không có đường từ ${v} đến ${kv}, nên không gì trong hàng này có thể cải thiện. Sau rào chắn, cả khối trả về.`),
      stale: false,
    };
  const stale = b.distKCol !== before[b.row][k] || b.rowK.some((x, col) => x !== before[k][col]);
  const reads = b.rowK.flatMap((x, col) => (x === INF ? [] : [`${VERTICES[col]} = ${x}`]));
  const writes = b.writes.map((w) => tr(`${v} to ${VERTICES[w.col]}: ${w.from === INF ? 'no path' : w.from} becomes ${w.to}`, `${v} đến ${VERTICES[w.col]}: ${w.from === INF ? 'không có đường' : w.from} thành ${w.to}`));
  return {
    text:
      tr(`${head}: ${b.distKCol}. Its threads read row ${kv} (${reads.join(', ') || 'no paths'}) and add. `, `${head}: ${b.distKCol}. Các luồng của nó đọc hàng ${kv} (${reads.join(', ') || 'không có đường nào'}) và cộng. `) +
      (writes.length ? tr(`Smaller sums are written: ${writes.join('; ')}.`, `Các tổng nhỏ hơn được ghi: ${writes.join('; ')}.`) : tr('No sum is smaller than the current entry, so nothing changes.', 'Không tổng nào nhỏ hơn phần tử hiện tại, nên không gì thay đổi.')),
    stale,
  };
}

function FloydStepper() {
  const [k, setK] = useState(0);
  const [m, setM] = useState(0);
  const [neg, setNeg] = useState(false);
  const diag = neg ? -1 : undefined;
  const base = useMemo(() => runSteps(FIG_16_5, k), [k]);
  // The book's kernel: one table, updated in place, blocks run top row first. The two-buffer
  // run is computed only to compare against.
  const trace = useMemo(() => stepTrace(base, k, { mode: 'in-place', diag }), [base, k, diag]);
  const two = useMemo(() => stepTrace(base, k, { mode: 'two-buffers', diag }), [base, k, diag]);
  useEffect(() => setM(0), [k, neg]);

  const table = stateAfter(trace, m);
  const cur = m > 0 ? trace.blocks[m - 1] : undefined;
  const done = trace.blocks.slice(0, m);
  const retiredSoFar = new Set(done.filter((b) => b.retired).map((b) => b.row));
  const changedSoFar = table.flatMap((row, i) => row.flatMap((x, j) => (x !== trace.before[i][j] ? [[i, j]] : [])));
  const complete = m >= N;
  const desc = cur ? describeBlock(cur, k, trace.before) : null;
  const hl = cur ? [cur.retired ? FW_LINES.retire : cur.writes.length ? FW_LINES.write : FW_LINES.compare] : [];
  const kv = VERTICES[k];
  const kStill = !trace.rowKChanged && !trace.colKChanged;
  const same = JSON.stringify(trace.after) === JSON.stringify(two.after);

  return (
    <Bench
      title={tr('One Floyd-Warshall step', 'Một bước Floyd-Warshall')}
      task={tr('Step through the blocks of one k, then set dist[k][k] to −1 and watch the blocks that run after block k.', 'Đi qua từng khối của một k, rồi đặt dist[k][k] thành −1 và theo dõi các khối chạy sau khối k.')}
      foot={
        <>
          <Translated
            vi={
              <>
                Bảng của Fig. 16.5 như được in; một ô trống là FW_INF, không có đường. Một bảng, cập nhật tại chỗ, các khối chạy
                hàng trên cùng trước. Hàng k và cột k được tô màu: mọi khối đều đọc chúng. Một ô được tô đậm đã thay đổi trong bước
                này. <TierLegend tiers={['reg', 'shared', 'global']} />
              </>
            }
          >
          Fig. 16.5's table as printed; a blank cell is FW_INF, no path. One table, updated in place, blocks run top row
          first. Row k and column k are tinted: every block reads them. A filled cell changed during this step. <TierLegend tiers={['reg', 'shared', 'global']} />
          </Translated>
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted"><Translated vi="bước k =">step k =</Translated></span>
        <span className="seg" role="group" aria-label={tr('Step k', 'Bước k')}>
          {VERTICES.map((w, i) => (
            <button key={w} aria-pressed={k === i} onClick={() => setK(i)}>{w}</button>
          ))}
        </span>
        <label className="check">
          <input type="checkbox" checked={neg} onChange={(e) => setNeg(e.target.checked)} data-neg /> <Translated vi="đặt dist[k][k] thành −1 cho bước này">set dist[k][k] to −1 for this step</Translated>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setM(0)} disabled={m === 0}><Translated vi="Bắt đầu lại bước">Restart the step</Translated></button>
        <button onClick={() => setM(m - 1)} disabled={m === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => setM(m + 1)} disabled={complete}><Translated vi="Khối tiếp theo">Next block</Translated></button>
        <button onClick={() => setM(N)} disabled={complete}><Translated vi="Chạy cả bước">Run the step</Translated></button>
        <button onClick={() => setK(Math.min(N - 1, k + 1))} disabled={k === N - 1}><Translated vi="k tiếp theo">Next k</Translated></button>
        <span className="muted" aria-live="polite" data-progress>
          <Translated
            vi={
              <>số khối đã chạy: {m} trên {N}; kết thúc sớm: {retiredSoFar.size}; số ô đã đổi: {changedSoFar.length}</>
            }
          >
          blocks run: {m} of {N}; retired: {retiredSoFar.size}; cells changed: {changedSoFar.length}
          </Translated>
        </span>
      </div>

      <div className={'narrate' + (desc?.stale ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }} data-narrate>
        {desc ? (
          <>
            {desc.text}
            {desc.stale ? (
              <>
                <br />
                <b className="t-fail">
                  <Translated
                    vi={
                      <>
                        Khối {kv} đã thay đổi hàng {kv} hoặc cột {kv}, và khối này đã đọc giá trị mới. Với hai buffer nó sẽ đọc giá trị cũ:
                        kết quả phụ thuộc vào ai chạy trước.
                      </>
                    }
                  >
                  Block {kv} has already changed row {kv} or column {kv}, and this block read the new values. With two
                  buffers it would read the old ones: the result depends on who ran first.
                  </Translated>
                </b>
              </>
            ) : null}
          </>
        ) : (
          <>
            <Translated
              vi={
                <>
                  Bước k = {kv}: mọi khối sẽ đọc dist[row][{kv}] (cột {kv}) một lần, và các luồng của nó sẽ đọc hàng {kv}.
                  {neg ? ` dist[${kv}][${kv}] giờ là −1, điều này phá vỡ giả định của sách.` : ''} Bấm Khối tiếp theo.
                </>
              }
            >
            Step k = {kv}: every block will read dist[row][{kv}] (column {kv}) once, and its threads will read row {kv}.
            {neg ? ` dist[${kv}][${kv}] is now −1, which breaks the book's assumption.` : ''} Press Next block.
            </Translated>
          </>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 4fr) minmax(0, 6fr)' }}>
        <div>
          <div className="mx-label t-global"><Translated vi={<>dist (bộ nhớ toàn cục), trong bước k = {kv}</>}>dist (global memory), during step k = {kv}</Translated></div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${N + 1}, auto)` }} data-table>
            <span style={HEAD} />
            {VERTICES.map((w) => <span key={w} style={HEAD}>{w}</span>)}
            {table.map((row, i) => {
              const ran = done.some((b) => b.row === i);
              const retired = retiredSoFar.has(i);
              return [
                <span
                  key={`h${i}`}
                  data-row={VERTICES[i]}
                  data-retired={retired ? 1 : 0}
                  style={{ ...HEAD, fontWeight: cur?.row === i ? 800 : 650, textDecoration: cur?.row === i ? 'underline' : undefined }}
                  title={retired ? tr('block retired', 'khối đã kết thúc sớm') : ran ? tr('block ran', 'khối đã chạy') : ''}
                >
                  {VERTICES[i]}
                </span>,
                ...row.map((x, j) => {
                  const changed = x !== trace.before[i][j];
                  const kLine = i === k || j === k;
                  const cls = changed ? (kLine ? 'bad' : 'hot') : kLine ? 'on' : '';
                  return (
                    <span
                      key={`${i},${j}`}
                      className={cls}
                      data-cell={`${i},${j}`}
                      data-v={x === INF ? '' : x}
                      data-changed={changed ? 1 : 0}
                      style={retired ? { opacity: 0.35 } : undefined}
                    >
                      {x === INF ? '' : x}
                    </span>
                  );
                }),
              ];
            })}
          </div>
          <p className="muted" style={{ margin: '0.5rem 0 0' }}>
            <Translated
              vi={
                <>
                  {retiredSoFar.size
                    ? `Hàng mờ: khối đã kết thúc sớm, vì dist[row][${kv}] là FW_INF (${[...retiredSoFar].sort((a, b) => a - b).map((i) => VERTICES[i]).join(', ')}).`
                    : 'Chưa khối nào kết thúc sớm.'}
                  {cur && !cur.retired ? (
                    <> <span className="t-shared">dist_k_col</span> chia sẻ của khối vừa chạy: {cur.distKCol}.</>
                  ) : null}
                </>
              }
            >
            {retiredSoFar.size
              ? `Faded rows: the block retired, because dist[row][${kv}] is FW_INF (${[...retiredSoFar].sort((a, b) => a - b).map((i) => VERTICES[i]).join(', ')}).`
              : 'No block has retired yet.'}
            {cur && !cur.retired ? (
              <> Shared <span className="t-shared">dist_k_col</span> of the last block: {cur.distKCol}.</>
            ) : null}
            </Translated>
          </p>
          {complete ? (
            <p style={{ margin: '0.7rem 0 0' }} data-checks>
              <span className={'chip ' + (kStill ? 'ok' : 'fail')} data-check="row-col">
                {kStill ? tr(`row ${kv} and column ${kv} unchanged during this step`, `hàng ${kv} và cột ${kv} không đổi trong bước này`) : tr(`row ${kv} or column ${kv} changed during this step`, `hàng ${kv} hoặc cột ${kv} đã đổi trong bước này`)}
              </span>{' '}
              <span className={'chip ' + (same ? 'ok' : 'fail')} data-check="same">
                {same ? tr('in place equals two buffers', 'tại chỗ bằng hai buffer') : tr('in place differs from two buffers', 'tại chỗ khác hai buffer')}
              </span>
            </p>
          ) : (
            <p className="muted" style={{ margin: '0.7rem 0 0' }}><Translated vi="Chạy cả bước để xem hai phép kiểm tra.">Run the whole step to see the two checks.</Translated></p>
          )}
        </div>
        <div>
          <Code tabs={[{ label: tr('FW_bottomup, corrected', 'FW_bottomup, đã sửa'), code: FW_KERNEL, tiers: FW_TIERS }]} highlight={hl} />
        </div>
      </div>
    </Bench>
  );
}

// ================================================================ who waits for whom: the explorer
const RULES: { id: Rule; label: string }[] = [
  { id: 'front', get label() { return tr('relaunch per front', 'khởi chạy lại mỗi mặt sóng'); } },
  { id: 'counter', get label() { return tr('flags, tiles from a counter', 'cờ, ô lấy từ bộ đếm'); } },
  { id: 'rows', get label() { return tr('one persistent block per tile row', 'một khối thường trú mỗi hàng ô'); } },
];

function WaitExplorer() {
  const [T, setT] = useState(4);
  const [S, setS] = useState(2);
  const [rule, setRule] = useState<Rule>('front');
  const [start, setStart] = useState<StartOrder>('index');
  const [t, setTime] = useState(0);
  const sch = useMemo(() => schedule(T, S, rule, start), [T, S, rule, start]);
  const counter = useMemo(() => schedule(T, S, 'counter'), [T, S]);
  const front = frontMakespan(T, S);
  const len = sch.stuck ? sch.timeline.length + 3 : sch.timeline.length;
  useEffect(() => setTime(0), [T, S, rule, start]);
  const num = numbering(T);
  const flags = flagsAfter(sch, t);
  const last = t > 0 ? Array.from({ length: S }, (_, s) => slotAt(sch, t - 1, s)) : [];
  const running = new Set(last.filter((c) => c?.state === 'run').map((c) => c!.tile));
  const waiting = new Set(last.filter((c) => c?.state === 'spin').map((c) => c!.tile));
  let idleSoFar = 0;
  for (let u = 0; u < t; u++) for (let s = 0; s < S; s++) if (slotAt(sch, u, s)?.state !== 'run') idleSoFar++;
  const stuckShown = sch.stuck && t > sch.stuckAt!;

  const G = 36;
  const gsz = T * G + 4;
  const cw = Math.max(12, Math.min(34, Math.floor(600 / len)));
  const rh = 26;
  const chartW = 52 + len * cw + 4;
  const chartH = 22 + S * rh + 4;

  return (
    <Bench
      title={tr('Who waits for whom', 'Ai chờ ai')}
      task={tr('Choose a rule, then step through time. Where do the SMs sit idle?', 'Chọn một quy tắc, rồi đi qua từng bước thời gian. SM ngồi không ở đâu?')}
      foot={
        <>
          <Translated
            vi={
              <>
                Mỗi ô cần ô phía trên và ô bên trái nó, và mất một bước. Một ô có ô vuông nhỏ được tô đã bật cờ của nó. Trong biểu
                đồ, một ô đỏ nét đứt là một SM không làm việc: trống, hoặc đang giữ một ô đang chờ.
              </>
            }
          >
          Each tile needs the tile above and the tile to its left, and takes one step. A tile with a filled square has set
          its flag. In the chart, a dashed red slot is an SM doing no work: empty, or holding a tile that waits.
          </Translated>
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label={tr('Rule', 'Quy tắc')} style={WRAP}>
          {RULES.map((r) => (
            <button key={r.id} aria-pressed={rule === r.id} onClick={() => setRule(r.id)}>{r.label}</button>
          ))}
        </span>
        {rule === 'rows' ? (
          <label className="field">
            <Translated vi={"\"Các khối bắt đầu\""}>
            Blocks start
            </Translated>
            <select value={start} onChange={(e) => setStart(e.target.value as StartOrder)} aria-label={tr('Blocks start', 'Các khối bắt đầu')}>
              <option value="index">{tr('in index order', 'theo thứ tự chỉ số')}</option>
              <option value="reversed">{tr('in reverse order', 'theo thứ tự ngược')}</option>
            </select>
          </label>
        ) : null}
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi={"\"Số ô mỗi cạnh, T\""}>
          Tiles per side, T
          </Translated>
          <select value={T} onChange={(e) => setT(+e.target.value)} aria-label={tr('Tiles per side', 'Số ô mỗi cạnh')}>
            {[2, 3, 4, 5, 6, 7, 8].map((v) => <option key={v} value={v}>{v}{v === 4 ? tr(" (the book's example)", ' (ví dụ của sách)') : ''}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi={"\"SM, S\""}>
          SMs, S
          </Translated>
          <select value={S} onChange={(e) => setS(+e.target.value)} aria-label={tr('SMs', 'SM')}>
            {[1, 2, 3, 4, 5, 6, 7, 8].map((v) => <option key={v} value={v}>{v}{v === 2 ? tr(" (the book's example)", ' (ví dụ của sách)') : ''}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setTime(0)} disabled={t === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => setTime(t - 1)} disabled={t === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => setTime(t + 1)} disabled={t >= len}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => setTime(len)} disabled={t >= len}><Translated vi="Chạy đến hết">Run to the end</Translated></button>
        <span className="muted" aria-live="polite" data-time={t}><Translated vi={<>sau bước {t}{sch.stuck ? '' : ` trên ${len}`}</>}>after step {t}{sch.stuck ? '' : ` of ${len}`}</Translated></span>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 3fr)' }}>
        <div>
          <div className="mx-label"><Translated vi="các ô theo thứ tự mặt sóng">tiles in front order</Translated></div>
          <svg viewBox={`0 0 ${gsz} ${gsz}`} role="img" aria-label={tr('Tiles and their flags', 'Các ô và cờ của chúng')} style={{ width: '100%', maxWidth: Math.min(220, gsz * 1.3), display: 'block' }}>
            {num.flatMap((row, r) =>
              row.map((n, c) => {
                const isDone = flags.has(n);
                const run = running.has(n);
                const wait = waiting.has(n);
                return (
                  <g key={n} data-tile={n} data-done={isDone ? 1 : 0} data-now={run ? 'run' : wait ? 'wait' : ''}>
                    <rect x={2 + c * G} y={2 + r * G} width={G - 3} height={G - 3} rx="2"
                      fill={isDone ? 'var(--global-wash)' : 'var(--cell)'}
                      stroke={wait ? 'var(--fail)' : run ? 'var(--ink)' : 'var(--rule)'}
                      strokeWidth={run || wait ? 2.4 : 1}
                      strokeDasharray={wait ? '4 3' : undefined} />
                    <text x={2 + c * G + G / 2 - 2} y={2 + r * G + G / 2 + 3} fontSize="12" textAnchor="middle">{n}</text>
                    <rect x={2 + c * G + G - 11} y={2 + r * G + G - 11} width="6" height="6"
                      fill={isDone ? 'var(--global)' : 'none'} stroke={isDone ? 'var(--global)' : 'var(--rule)'} />
                  </g>
                );
              }),
            )}
          </svg>
        </div>
        <div>
          <div className="mx-label"><Translated vi="mỗi SM làm gì, từng bước một">what each SM does, step by step</Translated></div>
          <div style={{ overflowX: 'auto' }}>
          <svg viewBox={`0 0 ${chartW} ${chartH}`} role="img" aria-label={tr('SMs by time', 'Các SM theo thời gian')} style={{ width: '100%', minWidth: Math.min(chartW, 900), maxWidth: Math.max(chartW * 1.6, 320), display: 'block' }} data-chart>
            {Array.from({ length: len }, (_, u) => (
              <text key={u} x={52 + u * cw + cw / 2} y={13} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>
                {cw >= 16 || u % 2 === 0 ? u + 1 : ''}
              </text>
            ))}
            {Array.from({ length: S }, (_, s) => (
              <g key={s}>
                <text x={0} y={22 + s * rh + rh / 2 + 4} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{tr(`SM ${s}`, `SM ${s}`)}</text>
                <rect x={52} y={22 + s * rh} width={len * cw} height={rh} fill="none" stroke="var(--rule-soft)" />
                {Array.from({ length: Math.min(t, len) }, (_, u) => {
                  const cell = slotAt(sch, u, s)!;
                  const x = 52 + u * cw + 1;
                  const y = 22 + s * rh + 2;
                  return (
                    <g key={u} data-slot={`${u},${s}`} data-state={cell.state} data-tile={cell.tile ?? ''}>
                      <rect x={x} y={y} width={cw - 2} height={rh - 4} rx="2"
                        fill={cell.state === 'run' ? 'var(--ink)' : cell.state === 'spin' ? 'var(--fail-wash)' : 'none'}
                        fillOpacity={cell.state === 'run' ? 0.12 : 1}
                        stroke={cell.state === 'run' ? 'var(--ink-2)' : 'var(--fail)'}
                        strokeDasharray={cell.state === 'run' ? undefined : '3 2'} />
                      {cw >= 18 && cell.tile !== null ? (
                        <text x={x + (cw - 2) / 2} y={y + rh / 2 + 2} fontSize="10" textAnchor="middle"
                          style={{ fill: cell.state === 'run' ? 'var(--ink)' : 'var(--fail)' }}>{cell.tile}</text>
                      ) : null}
                    </g>
                  );
                })}
              </g>
            ))}
            {rule === 'front'
              ? sch.barriers.filter((b) => b < t - 1).map((b) => (
                  <line key={b} data-barrier={b} x1={52 + (b + 1) * cw} y1={16} x2={52 + (b + 1) * cw} y2={chartH - 2} stroke="var(--ink-2)" strokeDasharray="2 3" />
                ))
              : null}
          </svg>
          </div>
          <table className="data" style={{ marginTop: '0.8rem' }}>
            <tbody>
              <tr><td><Translated vi="khởi chạy lại mỗi mặt sóng">relaunch per front</Translated></td><td data-k="front"><Translated vi={<>{front} bước</>}>{front} steps</Translated></td></tr>
              <tr><td><Translated vi="cờ, ô lấy từ bộ đếm">flags, tiles from a counter</Translated></td><td data-k="counter"><Translated vi={<>{counter.makespan} bước</>}>{counter.makespan} steps</Translated></td></tr>
              <tr>
                <td><Translated vi="quy tắc này">this rule</Translated></td>
                <td data-k="makespan">{sch.stuck ? tr('never finishes', 'không bao giờ xong') : tr(`${sch.makespan} steps`, `${sch.makespan} bước`)}</td>
              </tr>
              <tr>
                <td><Translated vi="số SM-bước không có việc, đến giờ">SM-steps with no work, so far</Translated></td>
                <td data-k="idle"><Translated vi={<>{idleSoFar}{sch.stuck ? '' : ` trên ${sch.idle}`}</>}>{idleSoFar}{sch.stuck ? '' : ` of ${sch.idle}`}</Translated></td>
              </tr>
            </tbody>
          </table>
          {stuckShown ? (
            <p className="t-fail" style={{ margin: '0.6rem 0 0' }} data-stuck>
              <Translated
                vi={
                  <>
                    <b>Bị kẹt.</b> Mọi SM đều giữ một khối đang chờ, trực tiếp hoặc qua một khối đang chờ khác, một hàng không có SM nào,
                    và không khối nào nhường SM của mình.
                  </>
                }
              >
              <b>Stuck.</b> Every SM holds a block that waits, directly or through another waiting block, for a row
              that has no SM, and no block gives its SM up.
              </Translated>
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

// ================================================================ the page
export function S163() {
  return (
    <>
      <Translated
        vi={
          <p>
            Hai quyết định định hình mọi kernel wavefront: những ô nào đi cùng nhau trong một bước, và một bước chờ bước trước
            như thế nào. Trang này lần lượt xét từng quyết định, với nghiên cứu tình huống đầu tiên của sách, Floyd-Warshall, ở
            giữa. Trang <a href={href('ch16', '16.6')}>16.6</a> áp dụng cả hai cho các ô (tile).
          </p>
        }
      >
      <p>
        Two decisions shape every wavefront kernel: which cells go together in one step, and how one step waits for the
        one before. This page takes them in turn, with the book's first case study, Floyd-Warshall, in between. Page{' '}
        <a href={href('ch16', '16.6')}>16.6</a> applies both to tiles.
      </p>
      </Translated>

      <h2><Translated vi="Từ trên xuống hay từ dưới lên">Top down or bottom up</Translated></h2>
      <Translated
        vi={
          <p>
            Quy hoạch động xây dựng đáp án từ đáp án của những bài toán con nhỏ hơn chồng lấn nhau, và giữ lại từng đáp án để
            nó chỉ được tính một lần. Có hai cách tổ chức việc đó. Từ trên xuống là phép đệ quy bạn sẽ viết đầu tiên: để có
            F(n), hỏi F(n − 1) và F(n − 2), nhưng trước hết tra từng giá trị trong một bảng băm chứa các đáp án đã tìm được
            (memoisation). Từ dưới lên làm ngược lại: lấp một bảng từ bài toán con nhỏ nhất trở lên, để mọi bài toán con tìm
            thấy đầu vào đã có sẵn trong bảng (tabulation).
          </p>
        }
      >
      <p>
        Dynamic programming builds an answer from the answers to smaller sub-problems that overlap, and keeps each one so
        that it is computed once. There are two ways to organise that. Top down is the recursion you would write first:
        to get F(n), ask for F(n − 1) and F(n − 2), but look each one up first in a hash table of answers found so far
        (memoisation). Bottom up turns it round: fill a table from the smallest sub-problem upward, so that every
        sub-problem finds its inputs already in the table (tabulation).
      </p>
      </Translated>
      <Source section="16.2" pages="374–376" figs={tr('Fig. 16.1', 'Hình 16.1')} />
      <Translated
        vi={
          <p>
            Sách đưa ra hai lý do chống lại cách từ trên xuống trên GPU. Kernel không xử lý hiệu quả các chuỗi lời gọi hàm sâu:
            hàm device thường được inline, và một lời gọi đệ quy không thể được inline mãi không dừng. Và các luồng của một warp
            truy cập vào bảng băm ở những chỗ rải rác, không được hợp nhất truy cập bộ nhớ (coalescing).
          </p>
        }
      >
      <p>
        The book gives two reasons against top down on a GPU. Kernels do not handle deep chains of function calls
        efficiently: device functions are normally inlined, and a recursive call cannot be inlined without end. And the
        threads of a warp reach into the hash table at scattered places, which do not coalesce.
      </p>
      </Translated>
      <Source section="16.2" pages="376" />
      <Translated
        vi={
          <p>
            Lý do thứ nhất là về chi phí, không phải một quy tắc: mọi backend đều cho phép hàm device đệ quy, và cái giá là bộ
            nhớ stack được dành sẵn cho mỗi luồng hoặc mỗi warp. Một bộ giải từ trên xuống đệ quy sâu bằng chuỗi bài toán con dài
            nhất của nó, nên stack mà mỗi luồng hoặc warp cần tăng theo kích thước bài toán, và hướng dẫn của AMD chỉ ra rằng
            tính song song của GPU nhân nó lên.
          </p>
        }
      >
      <p>
        The first reason is about cost, not a rule: every backend allows a recursive device function, and what it costs
        is stack memory reserved for every thread or warp. A top-down solver recurses as deep as its longest chain of
        sub-problems, so the stack each thread or warp needs grows with the problem, and AMD's guide points out that the
        GPU's parallelism multiplies it.
      </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch16-facts A1, B1, C1)', 'ngày 2026-11-20 (ch16-facts A1, B1, C1)')}>
        <Translated
          vi={
            <p>
              CUDA: hàm <code>__device__</code> có thể đệ quy, kernel thì không (<code>cudaLimitStackSize</code>). HIP: một stack
              cho mỗi warp (<code>hipLimitStackSize</code>). Metal: từ 2.4; Apple khuyên dùng <code>maxCallStackDepth</code> nhỏ.
            </p>
          }
        >
        <p>
          CUDA: <code>__device__</code> functions may recurse, kernels may not (<code>cudaLimitStackSize</code>). HIP: a
          stack per warp (<code>hipLimitStackSize</code>). Metal: from 2.4; Apple advises a small <code>maxCallStackDepth</code>.
        </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Lý do thứ hai mạnh hơn, và nó không liên quan đến đệ quy. Bảng memo là một bảng băm: các luồng có bài toán con kề
            nhau rơi vào những địa chỉ không liên quan, nên các lần nạp của một warp chạm vào nhiều đoạn bộ nhớ thay vì vài đoạn
            (chương 6). Và bài toán con nào chạy khi nào do phép đệ quy quyết định trong lúc chạy. Từ dưới lên cho mỗi luồng một
            địa chỉ được tính toán trong một mảng thường, và một lịch chạy cố định trước khi kernel bắt đầu. Phần còn lại của
            chương dùng cách từ dưới lên.
          </p>
        }
      >
      <p>
        The second reason is the stronger one, and it is not about recursion. A memo table is a hash table: threads with
        neighbouring sub-problems land at unrelated addresses, so a warp's loads touch many memory segments instead of a
        few (chapter 6). And which sub-problem runs when is decided by the recursion as it goes. Bottom up gives every
        thread a computed address in a plain array, and a schedule fixed before the kernel starts. The rest of the
        chapter is bottom up.
      </p>
      </Translated>
      <Translated
        vi={
          <p>
            Nó cũng cho định nghĩa thực dụng của wavefront: các ô được lấp trong một vòng lặp của vòng lặp từ dưới lên. Với
            Fibonacci, mỗi vòng lặp lấp một ô, nên không có gì để chạy song song. Những bảng đáng dùng GPU lấp nhiều ô mỗi vòng
            lặp, và phần tiếp theo nói về những ô nào.
          </p>
        }
      >
      <p>
        It also gives the working definition of a wavefront: the cells filled in one iteration of the bottom-up loop. For
        Fibonacci each iteration fills one cell, so there is nothing to run in parallel. The tables worth a GPU fill many
        cells per iteration, and the next part is about which ones.
      </p>
      </Translated>
      <Source section="16.2" pages="376" />
      <Ask q={questionsS163['ch16/16.3/q-topdown']} />

      <h2><Translated vi="Đánh số mỗi ô theo thời điểm nó có thể chạy">Number each cell by when it can run</Translated></h2>
      <Translated
        vi={
          <p>
            Mọi thứ trong chương này dựa trên một quy tắc. Một ô có thể chạy khi mọi ô nó đọc đã xong. Vậy hãy cho mỗi ô một số
            bước: lớn hơn một so với bước muộn nhất trong các ô nó đọc, hoặc 0 nếu nó chỉ đọc giá trị biên. Các ô có cùng số tạo
            thành một wavefront. Chúng không đọc gì của nhau và có thể chạy cùng lúc, và giữa bước này với bước sau phải có một
            rào chắn (barrier) nào đó, vì bước s đọc những gì bước s − 1 đã ghi.
          </p>
        }
      >
      <p>
        Everything in this chapter rests on one rule. A cell can run once every cell it reads is done. So give each cell a
        step number: one more than the latest step among the cells it reads, or 0 if it reads only boundary values. Cells
        with the same number form a wavefront. They read nothing from each other and can all run at once, and between one
        step and the next there must be a barrier of some kind, because step s reads what step s − 1 wrote.
      </p>
      </Translated>
      <Source section="16.3" pages="376–377" figs={tr('Fig. 16.2', 'Hình 16.2')} />
      <Translated
        vi={
          <p>
            Sách vẽ bốn mẫu. Ở hai mẫu, bàn cờ và tài chính, mọi ô chỉ đọc hàng ngay trên, nên một bước là cả một hàng và mọi mặt
            sóng có cùng kích thước. Floyd-Warshall, ở dưới, cũng có mặt sóng không đổi: cả một bảng mỗi bước. Ở Smith-Waterman và
            H.264, các mặt sóng dài dần rồi ngắn dần, khiến các luồng rảnh rỗi gần các góc của bảng.
          </p>
        }
      >
      <p>
        The book draws four patterns. In two of them, checkerboard and financial, every cell reads only the row above, so
        a step is a whole row and every front has the same size. Floyd-Warshall, below, has constant fronts too: one whole
        table per step. In Smith-Waterman and H.264 the fronts grow and then shrink, which leaves threads idle near the
        corners of the table.
      </p>
      </Translated>
      <Source section="16.3" pages="377" />
      <Predict
        id="ch16/16.3/p-sw8"
        prompt={
          <Translated
            vi={
              <>
                Một bảng 8 × 8 trong đó mỗi ô đọc ô ngay trên, ô bên trái và ô chéo trên bên trái (mẫu Smith-Waterman). Nó cần bao
                nhiêu bước được ngăn cách bởi rào chắn?
              </>
            }
          >
          <>
            An 8 × 8 table in which each cell reads the cell above, the cell to its left and the cell above-left (the
            Smith-Waterman pattern). How many barrier-separated steps does it take?
          </>
          </Translated>
        }
        unit={tr('steps', 'bước')}
      >
        {(v) => (
          <Translated
            vi={
              <p>
                {v === SW8.steps ? 'Đúng: ' : `Nó cần ${SW8.steps}: `}bước của ô (r, c) là r + c, từ 0 đến {SW8.steps - 1}, và nói
                chung một bảng R × C cần R + C − 1 bước. Bước rộng nhất có {SW8.widest} ô và trung bình là {SW8.mean.toFixed(1)}, nên
                một kernel cho mỗi bước {SW8.widest} luồng giữ chúng bận hơn một nửa thời gian một chút. Hãy kiểm tra trong trình
                khám phá.
              </p>
            }
          >
          <p>
            {v === SW8.steps ? 'Right: ' : `It takes ${SW8.steps}: `}the step of cell (r, c) is r + c, from 0 to{' '}
            {SW8.steps - 1}, and in general an R × C table takes R + C − 1 steps. The widest step has {SW8.widest} cells and
            the average is {SW8.mean.toFixed(1)}, so a kernel that gives every step {SW8.widest} threads keeps them a little
            over half busy. Check it in the explorer.
          </p>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Trình khám phá đánh số bảng bất kỳ theo quy tắc này. Chọn một mẫu hoặc tự đánh dấu tập đầu vào của bạn, đổi kích
            thước, và bấm vào một ô để xem nó đọc những ô nào.
          </p>
        }
      >
      <p>
        The explorer numbers any table by this rule. Choose a pattern or tick your own set of inputs, change the size, and
        click a cell to see which cells it reads.
      </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi ô tốn một đơn vị, số luồng bằng số ô trong một bước, và bộ nhớ không tốn gì: trình khám phá đếm bước, không đếm
              thời gian. Mẫu tài chính là cách đọc của chúng tôi về Fig. 16.2(b), vốn in rất nhỏ.
            </p>
          }
        >
        <p>
          Every cell costs one unit, there are as many threads as cells in a step, and memory is free: the explorer counts
          steps, not time. The financial pattern is our reading of Fig. 16.2(b), which is small in print.
        </p>
        </Translated>
      </Simplification>
      <PatternExplorer />
      <Translated
        vi={
          <p>
            Ba điều cần rút ra. Riêng stencil quyết định một bảng cần bao nhiêu bước và chúng rộng bao nhiêu; giá trị trong các ô
            không đóng vai trò gì. Một stencil chỉ đọc hàng ngay trên cho mặt sóng không đổi. Và một đầu vào ở phía trên bên
            phải, như trong H.264, khiến mỗi hàng bắt đầu sau hàng trên hai bước thay vì một, nên một bảng vuông cần số bước
            khoảng gấp ba cạnh của nó thay vì gấp hai.
          </p>
        }
      >
      <p>
        Three things to take from it. The stencil alone decides how many steps a table needs and how wide they are; the
        values in the cells play no part. A stencil that reads only the row above gives constant fronts. And an input
        above and to the right, as in H.264, makes each row start two steps after the row above instead of one, so a
        square table takes about three times its side in steps instead of two.
      </p>
      </Translated>
      <Ask q={questionsS163['ch16/16.3/q-h264']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Quy tắc này cũng cho bạn biết cần giữ bao nhiêu phần của bảng. Một bước đọc lùi tối đa một số cố định mặt sóng, và
            mọi thứ cũ hơn có thể bỏ. Sách ghi rằng Fibonacci chỉ cần hai giá trị cuối, Floyd-Warshall giữ một mặt phẳng của
            không gian ba chiều, và Smith-Waterman giữ toàn bộ bảng chỉ để truy vết ngược (traceback).
          </p>
        }
      >
      <p>
        The rule also tells you how much of a table must be kept. A step reads at most a fixed number of fronts back, and
        anything older can go. The book notes that Fibonacci needs only its last two values, that Floyd-Warshall keeps one
        plane of its three-dimensional space, and that Smith-Waterman keeps its whole table only for the traceback.
      </p>
      </Translated>
      <Source section="16.3" pages="377" />
      <Ask q={questionsS163['ch16/16.3/q-live']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Floyd-Warshall: một lần khởi chạy mỗi bước, một bảng">Floyd-Warshall: one launch per step, one table</Translated></h2>
      <Translated
        vi={
          <p>
            Floyd-Warshall tìm đường đi ngắn nhất giữa mọi cặp đỉnh của một đồ thị có hướng với cạnh có trọng số. Gọi d(i, j, k)
            là khoảng cách ngắn nhất từ i đến j chỉ dừng dọc đường ở các đỉnh 0 đến k. Hoặc đường tốt nhất như vậy tránh k, hoặc
            nó đi đến k rồi đi tiếp từ k: d(i, j, k) = min(d(i, j, k − 1), d(i, k, k − 1) + d(k, j, k − 1)). Bắt đầu từ trọng số
            các cạnh, với "không có đường" ở chỗ không có cạnh, và áp dụng hệ thức truy hồi cho k = 0, 1, …, N − 1. Các bài
            toán con lấp đầy một không gian N × N × N, nhưng mỗi mặt phẳng chỉ cần mặt phẳng trước nó, nên mỗi lúc chỉ phải giữ
            N × N giá trị.
          </p>
        }
      >
      <p>
        Floyd-Warshall finds the shortest path between every pair of vertices of a directed graph with weighted edges. Let
        d(i, j, k) be the shortest distance from i to j that stops on the way only at vertices 0 to k. Either the best such
        path avoids k, or it goes to k and on from k: d(i, j, k) = min(d(i, j, k − 1), d(i, k, k − 1) + d(k, j, k − 1)).
        Start from the edge weights, with "no path" where there is no edge, and apply the recurrence for k = 0, 1, …, N −
        1. The sub-problems fill an N × N × N space, but each plane needs only the one before it, so only N × N values
        have to be kept at a time.
      </p>
      </Translated>
      <Source section="16.4" pages="377–378" figs={tr('Fig. 16.3', 'Hình 16.3')} />
      <Translated
        vi={
          <p>
            Chỉ nhìn vào các phụ thuộc. Bước k không đọc gì ngoài hàng k và cột k của bước k − 1, nên N² ô của một bước không cần
            gì từ nhau: mọi ô nằm trong mọi mặt sóng, N bước gồm N² ô, một mặt sóng không đổi rộng bằng cả bảng. Sự đồng bộ hóa
            duy nhất cần có là giữa các bước, và sách đạt được nó theo cách đơn giản nhất: một luồng mỗi ô, một lần khởi chạy
            kernel mỗi k. Các lần khởi chạy trong một stream chạy lần lượt, nên việc kernel kết thúc chính là rào chắn.
          </p>
        }
      >
      <p>
        Look only at the dependencies. Step k reads nothing but row k and column k of step k − 1, so the N² cells of one
        step need nothing from each other: every cell is in every front, N steps of N² cells, a constant front as wide as
        the table. The only
        synchronisation needed is between steps, and the book gets it in the simplest way: one thread per cell, one
        kernel launch per k. Launches in one stream run one after another, so ending the kernel is the barrier.
      </p>
      </Translated>
      <Source section="16.4" pages="378–379" />
      <Code
        tabs={[
          { label: tr('kernel', 'kernel'), code: FW_KERNEL, tiers: FW_TIERS, status: tr('emulator-checked and front-end-checked; unverified on CUDA and HIP hardware', 'emulator-checked và front-end-checked; unverified trên phần cứng CUDA và HIP') },
          { label: tr('host loop', 'vòng lặp host'), code: FW_HOST, tiers: FW_TIERS, status: tr('emulator-checked and front-end-checked; unverified on CUDA and HIP hardware', 'emulator-checked và front-end-checked; unverified trên phần cứng CUDA và HIP') },
        ]}
        caption={
          <Translated
            vi={
              <>
                Fig. 16.4 của sách, đã sửa, từ <code>labs/ch16/snippets/fw_bottomup.cu</code>; <code>check.py</code>{' '}
                ở đó chạy nó so với một Floyd-Warshall trên CPU bằng trình giả lập (emulator). Nó khác trang sách ở ba chỗ, được mô
                tả bên dưới.
              </>
            }
          >
          <>
            The book's Fig. 16.4, corrected, from <code>labs/ch16/snippets/fw_bottomup.cu</code>; <code>check.py</code>{' '}
            there runs it against a CPU Floyd-Warshall on the emulator. It differs from the page in three places,
            described below.
          </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Mỗi khối phủ một phần của một hàng trong bảng, nên mọi luồng của nó cần cùng một giá trị, dist[row][k]. Luồng 0 đọc
            nó một lần vào một biến chia sẻ, và sau rào chắn mọi luồng dùng nó. Nếu đó là giá trị canh (sentinel), không có đường
            từ đỉnh của hàng này đến k, không gì trong hàng có thể cải thiện, và cả khối trả về. Ngược lại, mỗi luồng đọc phần
            tử của riêng nó trong hàng k, và các lần đọc này được hợp nhất trên toàn khối.
          </p>
        }
      >
      <p>
        Each block covers part of one row of the table, so all its threads need the same value, dist[row][k]. Thread 0
        reads it once into a shared variable, and after the barrier every thread uses it. If it is the sentinel, there is
        no path from this row's vertex to k, nothing in the row can improve, and the whole block returns. Otherwise each
        thread reads its own entry of row k, and these reads are coalesced across the block.
      </p>
      </Translated>
      <Source section="16.4" pages="381" />
      <Translated vi={<p>Những chỗ mã này khác với trang sách:</p>}><p>Where the listing differs from the page:</p></Translated>
      <ul>
        <li>
          <Translated
            vi={
              <>
                <b>Giá trị canh.</b> Trang sách so sánh một <code>int</code> với <code>INFINITY</code>, vốn là một{' '}
                <code>float</code>; không <code>int</code> nào bằng nó. Mã này dùng số nguyên{' '}
                <code>FW_INF = INT_MAX / 2</code>, để một phép cộng hai giá trị canh vẫn vừa trong một <code>int</code>.
              </>
            }
          >
          <b>The sentinel.</b> The page compares an <code>int</code> with <code>INFINITY</code>, which is a{' '}
          <code>float</code>; no <code>int</code> ever equals it. The listing uses the integer{' '}
          <code>FW_INF = INT_MAX / 2</code>, so that one addition of two sentinels still fits in an <code>int</code>.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Không trả về trước rào chắn.</b> Trang sách cho các luồng ngoài phạm vi trả về ngay đầu, trước{' '}
                <code>__syncthreads()</code>. Mã này giữ chúng đến rào chắn bằng một cờ <code>inRange</code>. Lệnh{' '}
                <code>return</code> sau rào chắn chỉ phụ thuộc vào giá trị chia sẻ, nên nó như nhau với cả khối.
              </>
            }
          >
          <b>No return before the barrier.</b> The page returns out-of-range threads at the top, before{' '}
          <code>__syncthreads()</code>. The listing keeps them to the barrier with an <code>inRange</code> flag. The{' '}
          <code>return</code> after the barrier depends only on the shared value, so it is the same for the whole block.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Host không chờ giữa các lần khởi chạy.</b> Trang sách gọi <code>cudaDeviceSynchronize()</code> sau mỗi lần khởi
                chạy. Thứ tự các lần khởi chạy trong một stream đã cho mỗi bước thấy những gì bước trước ghi.
              </>
            }
          >
          <b>No host wait between launches.</b> The page calls <code>cudaDeviceSynchronize()</code> after every launch.
          The order of launches in a stream already gives each step the previous one's writes.
          </Translated>
        </li>
      </ul>
      <Today checked={tr('2026-11-20 (ch02-facts A5, B3, C9; ch04-facts A6, B3, C3)', 'ngày 2026-11-20 (ch02-facts A5, B3, C9; ch04-facts A6, B3, C3)')}>
        <Translated
          vi={
            <p>
              Một stream chạy các lần khởi chạy theo thứ tự trên CUDA và HIP. Trên Metal, các dispatch trong một compute encoder
              được tạo theo cách thông thường chạy theo thứ tự và thấy được những gì nhau đã ghi.
            </p>
          }
        >
        <p>
          One stream runs launches in order on CUDA and HIP. On Metal, dispatches in one compute encoder created the
          plain way run in order and see each other's writes.
        </p>
        </Translated>
        <Translated
          vi={
            <p>
              Một luồng đã trả về không giữ <code>__syncthreads()</code> lại trên CUDA. HIP không ghi nhận điều này, và Metal chỉ
              ghi nhận cho Apple silicon.
            </p>
          }
        >
        <p>
          A thread that has returned does not hold <code>__syncthreads()</code> on CUDA. HIP does not document it, and
          Metal documents it for Apple silicon only.
        </p>
        </Translated>
      </Today>
      <Ask q={questionsS163['ch16/16.3/q-infinity']} label={tr("The page's listing", 'Mã in trong sách')} />

      <h3><Translated vi="Vì sao một bảng là đủ">Why one table is enough</Translated></h3>
      <Translated
        vi={
          <p>
            Kernel ghi vào bảng trong khi các luồng khác đọc hàng k và cột k từ đó. Điều này an toàn chỉ khi hàng k và cột k
            không đổi trong bước k, và với hai điều kiện thì chúng không thể đổi: dist[k][k] bằng 0, và không trọng số nào âm.
            Luồng sở hữu dist[i][k] tính min(dist[i][k], dist[i][k] + dist[k][k]), lại chính là dist[i][k], và điều tương tự
            đúng dọc hàng k. Vậy không luồng nào đọc một giá trị mà luồng khác đang thay đổi, và không cần buffer thứ hai. Ở chỗ
            một bước có thay đổi những giá trị mà luồng khác đọc, bạn cần hai buffer: đọc bảng cũ, ghi bảng mới, rồi hoán đổi
            (double buffering, chương 6).
          </p>
        }
      >
      <p>
        The kernel writes into the table while other threads read row k and column k from it. That is safe only if row k
        and column k do not change during step k, and under two conditions they cannot: dist[k][k] is 0, and no weight is
        negative. The thread that owns dist[i][k] computes min(dist[i][k], dist[i][k] + dist[k][k]), which is dist[i][k]
        again, and the same holds along row k. So no thread reads a value that another is changing, and no second buffer
        is needed. Where a step does change values that others read, you need two buffers: read the old table, write a new
        one, swap (double buffering, chapter 6).
      </p>
      </Translated>
      <Source section="16.4" pages="379–380" />
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy một bước của kernel đã sửa trên ví dụ tám đỉnh của sách, mỗi hàng một khối. Đầu tiên hãy
            đi qua k = a và đếm số khối kết thúc sớm. Rồi chạy a và b, và tìm các ô mà bước c thay đổi. Cuối cùng, đánh dấu
            dist[k][k] = −1 và đi lại qua c, theo dõi các khối chạy sau khối c.
          </p>
        }
      >
      <p>
        The stepper runs one step of the corrected kernel on the book's eight-vertex example, one block per row. Step
        through k = a first and count the blocks that retire. Then run a and b, and find the cells that step c changes.
        Last, tick dist[k][k] = −1 and step through c again, watching the blocks that run after block c.
      </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các khối chạy lần lượt, hàng trên cùng trước, mỗi khối chạy đến hết; trên GPU chúng đan xen theo những cách bạn không
              chọn. Đó là một trong các thứ tự GPU có thể tạo ra, đủ để cho thấy một kết quả phụ thuộc vào thứ tự. Bảng là bảng của
              Fig. 16.5 như được in, trong đó b đến f là 9; đồ thị bên cạnh ghi 6.
            </p>
          }
        >
        <p>
          Blocks run one after another, top row first, each to completion; on a GPU they interleave in ways you do not
          choose. That is one of the orders a GPU may produce, enough to show a result that depends on the order. The table is Fig. 16.5's as printed,
          where b to f is 9; the graph beside it says 6.
        </p>
        </Translated>
      </Simplification>
      <FloydStepper />
      <Translated
        vi={
          <p>
            Ở k = a, {FW_A.retired.length} trong 8 khối kết thúc ngay: chỉ a có đường đến a. Ở k = c, sau a và b,{' '}
            {FW_C.retired.length} khối kết thúc sớm. Với dist[k][k] = −1, hàng k và cột k thay đổi ngay trong bước của chúng,
            mọi khối chạy sau khối k đọc giá trị mới, và bảng khác với bảng mà hai buffer cho ra. Điều một khối đọc phụ thuộc
            vào việc khối k có chạy trước nó hay không: một race condition (tranh chấp dữ liệu), thứ mà hai buffer sẽ loại bỏ.
          </p>
        }
      >
      <p>
        At k = a, {FW_A.retired.length} of the 8 blocks retire at once: only a has a path to a. At k = c, after a and b,{' '}
        {FW_C.retired.length} retire. With dist[k][k] = −1, row k and column k change during their own step, every block
        that runs after block k reads the new values, and the table differs from the one two buffers give. What a block
        reads depends on whether block k ran before it: a race, which two buffers would remove.
      </p>
      </Translated>
      <Ask q={questionsS163['ch16/16.3/q-fw-changes']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS163['ch16/16.3/q-fw-inplace']} />
      <Ask q={questionsS163['ch16/16.3/q-ex1']} label={tr('Book exercise 1', 'Bài tập 1 của sách')} />

      <h2><Translated vi="Ai chờ ai">Who waits for whom</Translated></h2>
      <Translated
        vi={
          <p>
            Floyd-Warshall không cần gì mịn hơn một lần khởi chạy lại: mọi khối của bước k + 1 đọc hàng k + 1 và cột k + 1, vốn do
            các khối khắp bước k ghi. Một bảng mà các ô chỉ phụ thuộc vào láng giềng gần xứng đáng được đối xử tốt hơn. Hãy hỏi
            với mọi thiết kế: một đơn vị công việc chờ điều gì, và điều đó có nhiều hơn mức nó cần không? Từ kiểu chờ thô nhất
            đến mịn nhất:
          </p>
        }
      >
      <p>
        Floyd-Warshall needs nothing finer than a relaunch: every block of step k + 1 reads row k + 1 and column k + 1,
        which blocks all over step k wrote. A table whose cells depend only on near neighbours deserves better. Ask of every design: what does one unit of work
        wait for, and is that more than it needs? From the coarsest wait to the finest:
      </p>
      </Translated>
      <Source section="16.8" pages="397–399" figs={tr('Fig. 16.21', 'Hình 16.21')} />
      <ol>
        <li>
          <Translated
            vi={
              <>
                <b>Cả lưới chờ toàn bộ mặt sóng trước.</b> Khởi chạy lại mỗi mặt sóng làm điều này; các kernel Smith-Waterman phân ô
                của sách khởi chạy lại một lần cho mỗi đường chéo phụ của các ô. Một rào chắn toàn lưới,{' '}
                <code>grid.sync()</code> dưới một lần khởi chạy cooperative, tiết kiệm các lần khởi chạy nhưng không bớt việc chờ, và
                cần mọi khối thường trú cùng lúc (API này là §18.7 của sách, pp. 445–448, được giải ở trang{' '}
                <a href={href('ch18', '18.6')}>18.6, "Hàng đợi riêng và một lần khởi chạy cho mọi mức"</a>).
              </>
            }
          >
          <b>The whole grid waits for the whole previous front.</b> A relaunch per front does this; the book's tiled
          Smith-Waterman kernels relaunch once per anti-diagonal of tiles. A grid-wide barrier,{' '}
          <code>grid.sync()</code> under a cooperative launch, saves the launches but not the wait, and needs every block
          resident at once (the API is the book's §18.7, pp. 445–448, worked on page{' '}
          <a href={href('ch18', '18.6')}>18.6, "Private queues and one launch for every level"</a>).
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Một đơn vị công việc chỉ chờ các láng giềng của nó.</b> Một cờ (flag) cho mỗi ô, được bật khi ô đã được lưu. Một
                ô bắt đầu khi cờ của ô phía trên và ô bên trái nó đã bật. Sách ví nó với phép scan của chương 11.
              </>
            }
          >
          <b>A unit of work waits only for its neighbours.</b> One flag per tile, set once the tile is stored. A tile
          starts when the flags of the tile above and the tile to its left are set. The book likens it to the scan of
          chapter 11.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <b>Không gì chờ xuyên khối</b>, vì một khối, hay một warp, sở hữu toàn bộ chuỗi. Với chuỗi ngắn, một khối hay thậm
                chí một warp cho mỗi bài toán là đủ, với rào chắn hoặc shuffle bên trong (shuffle ở trang{' '}
                <a href={href('ch10', '10.7')}>10.7</a>).
              </>
            }
          >
          <b>Nothing waits across blocks</b>, because one block, or one warp, owns the whole chain. For short sequences one
          block or even one warp per problem is enough, with barriers or shuffles inside (shuffles are on page{' '}
          <a href={href('ch10', '10.7')}>10.7</a>).
          </Translated>
        </li>
      </ol>
      <Source section="16.6, 16.8" pages="384, 398" />
      <Translated
        vi={
          <p>
            Hoạt hình và trình khám phá sau nó chạy trên cùng một mô hình các ô và SM, chỉ đếm bước và không gì khác.
          </p>
        }
      >
      <p>
        The animation and the explorer after it run on one model of tiles and SMs, which counts steps and nothing else.
      </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi ô mất một bước, khởi chạy và rào chắn không tốn gì, và một khối đang chờ chỉ tốn SM của nó. S là số ô chạy cùng
              lúc, không phải số SM của một thiết bị. Kết cục bị kẹt là một lịch chạy mà việc thiếu bảo đảm về thứ tự cho phép,
              không phải mô hình bộ lập lịch của GPU nào.
            </p>
          }
        >
        <p>
          Every tile takes one step, launches and barriers cost nothing, and a waiting block costs its SM and nothing else.
          S is how many tiles run at once, not a device's SM count. The stuck outcome is one schedule that the missing
          ordering guarantee allows, not a model of any GPU's scheduler.
        </p>
        </Translated>
      </Simplification>
      <Predict
        id="ch16/16.3/p-flags-2sm"
        prompt={
          <Translated
            vi={
              <>
                Một lưới 4 × 4 ô trên một GPU có 2 SM; khởi chạy lại mỗi mặt sóng mất 10 bước thời gian. Nếu thay vào đó mỗi ô bắt
                đầu ngay khi ô phía trên và ô bên trái nó xong, sẽ mất bao nhiêu bước thời gian?
              </>
            }
          >
          <>
            A 4 × 4 grid of tiles on a GPU with 2 SMs; a relaunch per front takes 10 time steps. If instead each tile
            starts as soon as the tile above it and the tile to its left are done, how many time steps does it take?
          </>
          </Translated>
        }
        unit={tr('steps', 'bước')}
      >
        {(v) => (
          <Translated
            vi={
              <p>
                {v === C2 ? 'Đúng: ' : `Nó cần ${C2}: `}ít hơn một bước so với {F2} của việc khởi chạy lại mỗi mặt sóng, như trong
                Fig. 16.21 của sách. Hoạt hình cho thấy bước đó đến từ đâu, và trình khám phá sau nó cho bạn đổi lưới và số SM.
              </p>
            }
          >
          <p>
            {v === C2 ? 'Right: ' : `It takes ${C2}: `}one step fewer than the {F2} of a relaunch per front, as in the
            book's Fig. 16.21. The animation shows where the step comes from, and the explorer after it lets you change the
            grid and the number of SMs.
          </p>
          </Translated>
        )}
      </Predict>
      <NeighboursScene />
      <Ask q={questionsS163['ch16/16.3/q-enough-sms']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Trong trình khám phá, khởi chạy lại mỗi mặt sóng để SM rảnh ở bất cứ chỗ nào một mặt sóng có ít ô hơn số SM, hoặc có
            số ô không chia đều. Với cờ, một SM rảnh lấy ô tiếp theo theo thứ tự mặt sóng và chỉ chờ hai láng giềng của ô đó, nên
            một mặt sóng sau có thể bắt đầu trong khoảng trống. Hãy thử một SM, rồi số SM bằng số ô mỗi cạnh.
          </p>
        }
      >
      <p>
        In the explorer, a relaunch per front leaves SMs idle wherever a front has fewer tiles than there are SMs, or a
        number that does not divide evenly. With flags, a free SM takes the next tile in front order and waits only for
        that tile's two neighbours, so a later front can start in the gap. Try one SM, then as many SMs as tiles per side.
      </p>
      </Translated>
      <WaitExplorer />
      <Translated
        vi={
          <p>
            Thiết kế cờ của chính sách giao cho mỗi khối thường trú (persistent) cả một hàng ô, theo <code>blockIdx.x</code> của
            nó, nên một khối chỉ kiểm tra cờ của ô phía trên. Đặt trình khám phá thành một khối mỗi hàng, khởi động các khối
            theo thứ tự ngược, và cho nó ít SM hơn số hàng.
          </p>
        }
      >
      <p>
        The book's own flag design gives each persistent block a whole row of tiles, by its <code>blockIdx.x</code>, so a
        block checks only the flag of the tile above. Set the explorer to one block per row, start the blocks in reverse,
        and give it fewer SMs than rows.
      </p>
      </Translated>
      <Source section="16.8" pages="398" />
      <Ask q={questionsS163['ch16/16.3/q-row-owner']} />
      <Translated
        vi={
          <p>
            Điều làm việc chờ an toàn cũng giống như ở chương 11: một khối chỉ được chờ công việc đã bắt đầu. Một bộ đếm ô theo
            thứ tự mặt sóng cho đúng điều đó, vì mọi ô mà một khối chờ đã được nhận trước, bởi một khối đang chạy.
          </p>
        }
      >
      <p>
        What makes waiting safe is the same as in chapter 11: a block may wait only for work that has already started. A
        tile counter in front order gives exactly that, because every tile a block waits for was claimed earlier, by a
        block that is already running.
      </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch16-facts A4, B3, C4; infra-facts A4, B4)', 'ngày 2026-11-20 (ch16-facts A4, B3, C4; infra-facts A4, B4)')}>
        <Translated
          vi={
            <p>
              CUDA: cờ được tài liệu xác nhận là an toàn với release và acquire ở phạm vi device (<code>cuda::atomic_ref</code>) và
              các ô được nhận từ một bộ đếm nguyên tử theo thứ tự mặt sóng, hoặc dưới một lần khởi chạy cooperative với cả lưới
              thường trú.
            </p>
          }
        >
        <p>
          CUDA: flags are documented-safe with device-scope release and acquire (<code>cuda::atomic_ref</code>) and tiles
          claimed from an atomic counter in front order, or under a cooperative launch with the whole grid resident.
        </p>
        </Translated>
        <Translated
          vi={
            <p>
              HIP: cùng mã nguồn biên dịch được qua libhipcxx, và <code>hipLaunchCooperativeKernel</code> có tồn tại, nhưng AMD
              không ghi nhận bảo đảm tiến triển nào giữa các khối: unverified.
            </p>
          }
        >
        <p>
          HIP: the same source compiles through libhipcxx, and <code>hipLaunchCooperativeKernel</code> exists, but AMD
          documents no progress guarantee between blocks: unverified.
        </p>
        </Translated>
        <Translated
          vi={
            <p>
              Metal: không có rào chắn toàn lưới, không có khởi chạy cooperative, không có bảo đảm tiến triển giữa các threadgroup.
              Một dispatch mỗi mặt sóng là thiết kế duy nhất được tài liệu xác nhận an toàn xuyên threadgroup.
            </p>
          }
        >
        <p>
          Metal: no grid barrier, no cooperative launch, no progress guarantee between threadgroups. One dispatch per
          front is the only documented-safe design across threadgroups.
        </p>
        </Translated>
      </Today>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th><Translated vi="Cái gì chờ">What waits</Translated></th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="cả lưới, khi khởi chạy lại mỗi mặt sóng">the grid, on a relaunch per front</Translated></td>
            <td><Translated vi="một lần khởi chạy mỗi mặt sóng, một stream">one launch per front, one stream</Translated></td>
            <td><Translated vi="như trên">same</Translated></td>
            <td><Translated vi="một dispatch mỗi mặt sóng trong một compute encoder">one dispatch per front in one compute encoder</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="cả lưới, ở một rào chắn toàn lưới">the grid, on a grid barrier</Translated></td>
            <Translated
              vi={
                <td><code>grid.sync()</code> dưới <code>cudaLaunchCooperativeKernel</code>, lưới thường trú</td>
              }
            >
            <td><code>grid.sync()</code> under <code>cudaLaunchCooperativeKernel</code>, grid resident</td>
            </Translated>
            <Translated
              vi={
                <td><code>hipLaunchCooperativeKernel</code>, đủ cả sáu đối số</td>
              }
            >
            <td><code>hipLaunchCooperativeKernel</code>, all six arguments</td>
            </Translated>
            <td><Translated vi="không có">none</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="một ô, chờ cờ của các ô láng giềng">a tile, on its neighbours' flags</Translated></td>
            <td><Translated vi="release và acquire ở phạm vi device, các ô lấy từ một bộ đếm nguyên tử">release and acquire at device scope, tiles from an atomic counter</Translated></td>
            <td><Translated vi="cùng mã nguồn; tiến triển không được tài liệu ghi nhận, unverified">same source; progress not documented, unverified</Translated></td>
            <td><Translated vi="không an toàn: không có bảo đảm tiến triển giữa các threadgroup">unsafe: no progress guarantee between threadgroups</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="một bước, bên trong một khối">a step, inside one block</Translated></td>
            <td><code>__syncthreads()</code></td>
            <td><Translated vi="như trên">same</Translated></td>
            <Translated
              vi={
                <td><code>threadgroup_barrier</code>, với <code>mem_device</code> nếu chuỗi nằm trong bộ nhớ device</td>
              }
            >
            <td><code>threadgroup_barrier</code>, with <code>mem_device</code> if the chain is in device memory</td>
            </Translated>
          </tr>
          <tr>
            <td><Translated vi="một bước, bên trong một warp">a step, inside one warp</Translated></td>
            <Translated
              vi={
                <td><code>__shfl_*_sync</code> với lane mask 32-bit</td>
              }
            >
            <td><code>__shfl_*_sync</code> with a 32-bit lane mask</td>
            </Translated>
            <Translated
              vi={
                <td>cùng tên, lane mask 64-bit</td>
              }
            >
            <td>same names, 64-bit lane mask</td>
            </Translated>
            <Translated
              vi={
                <td><code>simd_shuffle_*</code>, không có mask; đọc độ rộng từ <code>[[threads_per_simdgroup]]</code></td>
              }
            >
            <td><code>simd_shuffle_*</code>, no mask; read the width from <code>[[threads_per_simdgroup]]</code></td>
            </Translated>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p className="muted" style={{ fontFamily: 'var(--sans)', fontSize: '0.8rem' }}>
            Từ ch16-facts, "Where the backends really differ", và infra-facts A3, B3, C2, C5. Đừng bao giờ viết độ rộng warp là
            hằng số 32: nó là 32 trên NVIDIA, 64 hoặc 32 trên AMD, và được đọc từ pipeline trên Metal.
          </p>
        }
      >
      <p className="muted" style={{ fontFamily: 'var(--sans)', fontSize: '0.8rem' }}>
        From ch16-facts, "Where the backends really differ", and infra-facts A3, B3, C2, C5. Never write the warp width as
        a literal 32: it is 32 on NVIDIA, 64 or 32 on AMD, and read from the pipeline on Metal.
      </p>
      </Translated>

      <h2><Translated vi="Điều này nằm ở đâu trong PyTorch">Where this sits in PyTorch</Translated></h2>
      <Translated
        vi={
          <p>
            CTC loss của PyTorch, <code>torch.nn.functional.ctc_loss</code>, lấp một bảng alpha(t, s) trong đó mỗi phần tử cần
            alpha(t − 1, s), alpha(t − 1, s − 1) và alpha(t − 1, s − 2). Đó là một mặt sóng không đổi mỗi bước thời gian, chính
            stencil của mẫu "Financial; also CTC" trong trình khám phá. Kernel CUDA của nó giao cho mỗi khối trọn các chuỗi và
            lặp theo t với một <code>__syncthreads()</code> ở mỗi bước, nên các khối không bao giờ chờ nhau. Khi chuỗi đích rộng
            hơn một khối, khối đi qua các đoạn vị trí theo thứ tự phụ thuộc. Kernel MPS làm tương tự với một threadgroup cho mỗi
            phần tử của batch, và rào chắn của nó là{' '}
            <code>threadgroup_barrier(mem_flags::mem_device)</code>, vì bảng nằm trong bộ nhớ device và một rào chắn chỉ trên bộ
            nhớ threadgroup sẽ không sắp thứ tự được nó.
          </p>
        }
      >
      <p>
        PyTorch's CTC loss, <code>torch.nn.functional.ctc_loss</code>, fills a table alpha(t, s) in which each entry needs
        alpha(t − 1, s), alpha(t − 1, s − 1) and alpha(t − 1, s − 2). That is a constant front per time step, the
        stencil of the explorer's "Financial; also CTC" pattern. Its CUDA kernel gives each block whole sequences and loops over t with a{' '}
        <code>__syncthreads()</code> at every step, so blocks never wait for each other. When the target is wider than a
        block, the block walks over chunks of positions in dependency order. The MPS kernel does the same with one
        threadgroup per batch item, and its barrier is{' '}
        <code>threadgroup_barrier(mem_flags::mem_device)</code>, because the table lives in device memory and a barrier
        on threadgroup memory alone would not order it.
      </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không bàn về CTC. Các kernel được đọc ngày 2026-11-20 (ch16-facts D2, C3: <code>LossCTC.cu</code>,{' '}
              <code>LossOps.metal</code>); mối liên hệ với chương này là của khóa học này.
            </p>
          }
        >
        <p>
          The book does not discuss CTC. The kernels were read on 2026-11-20 (ch16-facts D2, C3: <code>LossCTC.cu</code>,{' '}
          <code>LossOps.metal</code>); the mapping to this chapter is ours.
        </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS163['ch16/16.3/q-ctc']} label={tr('In PyTorch', 'Trong PyTorch')} />
      <Translated
        vi={
          <p>
            <a href={href('ch16', 'labs')}>Bài thực hành</a> đưa ý tưởng thứ hai của chương này, các bước đủ độ rộng, lên GPU của
            chính bạn, và trang <a href={href('ch16', '16.6')}>16.6</a> giải thích nó trước.
          </p>
        }
      >
      <p>
        The <a href={href('ch16', 'labs')}>lab</a> puts the second idea of this chapter, full-width steps, on your own GPU,
        and page <a href={href('ch16', '16.6')}>16.6</a> explains it first.
      </p>
      </Translated>
    </>
  );
}
