import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { defineLabs, solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { causalSoftmaxProblem } from '../../sim/attnref';
import {
  CORRECT_SOFTMAX,
  NEG_IDENTITY,
  SCHEDULES,
  SCHEDULE_LABEL,
  VARIANTS,
  columnsOf,
  runAllRows,
  runRow,
  stateAt,
  unfusedTraffic,
  type Schedule,
  type SoftmaxConfig,
  type Step,
} from '../../sim/masksoftmax';
import { functionSource } from '../../sim/toolchain';

export const questionsS203: Record<string, Question> = {
  'ch20/20.3/q-cols': {
    kind: 'numeric',
    id: 'ch20/20.3/q-cols',
    prompt: (
      <Translated
        vi={
          <>
            Fig. 20.4 với <code>BLOCK_SIZE = 4</code> và N = 4096, quá lớn cho trình chạy từng bước. Trong khối 1000 (hàng
            1000), luồng 3 đọc bao nhiêu cột trong vòng lặp max?
          </>
        }
      >
        <>
          Fig. 20.4 with <code>BLOCK_SIZE = 4</code> and N = 4096, too large for the stepper. In block 1000 (row 1000),
          how many columns does thread 3 read in the max loop?
        </>
      </Translated>
    ),
    answer: 250,
    unit: 'columns',
    explain: (
      <Translated
        vi={
          <p>
            <code>idx</code> bắt đầu từ <code>threadIdx.x</code> = 3 và bước 4: 3, 7, …, 999, rồi 1003 không thỏa{' '}
            <code>idx &lt;= blockIdx.x</code> = 1000. Tức là 3 + 4k ≤ 1000 với k = 0 đến 249, vậy 250 cột. Hãy kiểm tra
            quy tắc trên một trường hợp nhỏ trong trình chạy từng bước: N = 8, hàng 5, luồng 1 đọc các cột 1 và 5.
          </p>
        }
      >
        <p>
          <code>idx</code> starts at <code>threadIdx.x</code> = 3 and steps by 4: 3, 7, …, 999, then 1003 fails{' '}
          <code>idx &lt;= blockIdx.x</code> = 1000. That is 3 + 4k ≤ 1000 for k = 0 to 249, so 250 columns. Check the rule
          on a small case in the stepper: N = 8, row 5, thread 1 reads columns 1 and 5.
        </p>
      </Translated>
    ),
    source: 'ch. 20 Fig. 20.4, pp. 487–488',
    vi: { unit: 'cột' },
  },
  'ch20/20.3/q-idle': {
    kind: 'choice',
    id: 'ch20/20.3/q-idle',
    prompt: (
      <Translated vi={<>Trong khối 2 (với BLOCK_SIZE = 4), luồng 3 không đọc cột nào cả. Nó đóng góp gì cho hai phép rút gọn?</>}>
        <>In block 2 (with BLOCK_SIZE = 4), thread 3 reads no column at all. What does it contribute to the two reductions?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Các phần tử đơn vị: float nhỏ nhất cho max, 0 cho tổng.">
            The identities: the lowest float for the max, 0 for the sum.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không gì cả: một luồng không có cột nào bỏ qua cả hai phép rút gọn.">
            Nothing: a thread with no columns skips both reductions.
          </Translated>
        ),
        why: (
          <Translated vi="BlockReduce là một thao tác tập thể trên cả khối: mọi luồng của khối phải gọi nó.">
            BlockReduce is a block-wide collective: every thread of the block must call it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="S[2][3], cột duy nhất nó lẽ ra đã đọc nếu không có mặt nạ.">
            S[2][3], the one column it would have read without the mask.
          </Translated>
        ),
        why: (
          <Translated vi="Thân vòng lặp không bao giờ chạy, nên nó không đọc gì; các giá trị khởi đầu của nó được đưa vào.">
            The loop body never runs, so it reads nothing; its starting values go in.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="0 cho cả hai phép rút gọn, vì nó không có giá trị riêng nào để cộng.">
            0 to both reductions, since it has no values of its own to add.
          </Translated>
        ),
        why: (
          <Translated vi="Phần tử đơn vị 0 cho max là sai với một hàng có logits đều âm.">
            A max identity of 0 is wrong for a row whose logits are all negative.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Max riêng của nó giữ giá trị khởi đầu (<code>NEG_INFINITY</code> của sách, <code>-FLT_MAX</code> trong kernel
            đã sửa) và tổng riêng của nó vẫn là 0. Không giá trị nào làm thay đổi kết quả. Trong trình chạy từng bước, hãy
            chọn hàng 2 và nhìn vào hàng của luồng 3 trong bảng trước dòng 11.
          </p>
        }
      >
        <p>
          Its private max keeps its starting value (the book&rsquo;s <code>NEG_INFINITY</code>, <code>-FLT_MAX</code> in
          the corrected kernel) and its private sum stays 0. Neither changes the result. In the stepper, pick row 2 and
          look at thread 3&rsquo;s row of the table before line 11.
        </p>
      </Translated>
    ),
    source: 'ch. 20 p. 488; notes on lines 6–9 of Fig. 20.4',
    vi: { source: 'ch. 20 p. 488; ghi chú về các dòng 6–9 của Fig. 20.4' },
  },
  'ch20/20.3/q-barrier': {
    kind: 'choice',
    id: 'ch20/20.3/q-barrier',
    prompt: (
      <Translated
        vi={
          <>
            Trong bảng kết quả, bỏ rào chắn ở dòng 19 vẫn để mọi thứ tự luồng cho kết quả đúng. Dòng 21 vẫn ghi đè giá
            trị max trong <code>max_or_sum</code> bằng tổng. Vì sao điều đó không bao giờ có thể xảy ra trước khi một
            luồng chậm đọc max ở dòng 14?
          </>
        }
      >
        <>
          In the outcome table, removing the barrier at line 19 leaves every thread order right. Line 21 still overwrites
          the max in <code>max_or_sum</code> with the sum. Why can that never happen before a slow thread has read the max
          at line 14?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Luồng 0 trước hết cần mọi tổng riêng, mỗi tổng được tạo sau dòng 14 của luồng đó.">
            Thread 0 needs every private sum first, each made after that thread's line 14.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Rào chắn ở dòng 22 giữ lại lần ghi ở dòng 21 của luồng 0 cho đến khi mọi luồng đã đọc.">
            The barrier at line 22 holds thread 0's line-21 write back until every thread has read.
          </Translated>
        ),
        why: (
          <Translated vi="Dòng 22 đến sau dòng 21; nó không thể giữ lại một lần ghi đứng trước nó.">
            Line 22 comes after line 21; it cannot hold back a write that precedes it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Vì max_or_sum chỉ được đọc một lần, nên không thể có ghi-sau-đọc.">
            Because max_or_sum is read only once, so a write-after-read is impossible.
          </Translated>
        ),
        why: (
          <Translated vi="Nó được đọc hai lần mỗi luồng (dòng 14 và 23) và được ghi hai lần; về nguyên tắc mối nguy là có thật.">
            It is read twice per thread (lines 14 and 23) and written twice; the hazard is real in principle.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó có thể xảy ra; chỉ là các thứ tự luồng của mô hình chưa bao giờ tạo ra nó.">
            It can happen; the model's thread orders just never produce it.
          </Translated>
        ),
        why: (
          <Translated vi="Không thứ tự nào làm được: giá trị luồng 0 ghi chưa tồn tại cho đến khi mọi luồng đã đóng góp.">
            No order can: the value thread 0 writes does not exist until every thread has contributed.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Tổng mà luồng 0 ghi ở dòng 21 là kết quả của phép rút gọn ở dòng 20, vốn cần <code>sum_thread</code> của
              mọi luồng, và mỗi luồng chỉ tính giá trị đó sau khi đọc max ở dòng 14. Luồng dữ liệu xếp lần ghi đứng sau mọi
              lần đọc. Rào chắn ở dòng 13 cũng thỏa quy tắc của CUB về việc dùng lại <code>temp_store</code> (facts A2),
              nên dòng 19 không phải là thứ giữ cho kernel này đúng; dòng 10 không bảo vệ gì trong một kernel xử lý một
              hàng mỗi khối.
            </p>
            <p>
              Rào chắn thực sự quan trọng ở vị trí đó là rào chắn trong lab 10.3, nơi ô phát tán cũng là vùng nháp của
              phép rút gọn (câu hỏi tiếp theo).
            </p>
          </>
        }
      >
        <>
          <p>
            The sum thread 0 writes at line 21 is the result of the reduction at line 20, which needs every thread&rsquo;s{' '}
            <code>sum_thread</code>, and each thread computes that only after reading the max at line 14. The data flow
            orders the write after every read. The barrier at line 13 also satisfies CUB&rsquo;s rule for reusing{' '}
            <code>temp_store</code> (facts A2), so line 19 is not what keeps this kernel right; line 10 protects nothing in
            a kernel that handles one row per block.
          </p>
          <p>
            The barrier that does matter in that position is the one in lab 10.3, where the broadcast slot is also the
            reduction&rsquo;s scratch space (next question).
          </p>
        </>
      </Translated>
    ),
    source: 'our analysis of Fig. 20.4; the model behind the outcome table; facts A2',
    vi: { source: 'phân tích của chúng tôi về Fig. 20.4; mô hình đứng sau bảng kết quả; facts A2' },
  },
  'ch20/20.3/q-war-lab': {
    kind: 'choice',
    id: 'ch20/20.3/q-war-lab',
    prompt: (
      <Translated
        vi={
          <>
            Lab 20.1 dùng lại phép rút gọn M4 của bạn. <code>blockReduceMax</code> trao kết quả cho mọi luồng qua{' '}
            <code>scratch[0]</code>; <code>blockReduceSum</code> bắt đầu bằng việc lane 0 của mỗi warp ghi{' '}
            <code>scratch[warp]</code>. Bỏ <code>__syncthreads()</code> giữa hai lời gọi. Điều gì có thể hỏng?
          </>
        }
      >
        <>
          Lab 20.1 reuses your M4 reduction. <code>blockReduceMax</code> hands its result to every thread through{' '}
          <code>scratch[0]</code>; <code>blockReduceSum</code> begins with lane 0 of each warp writing{' '}
          <code>scratch[warp]</code>. Remove the <code>__syncthreads()</code> between the two calls. What can go wrong?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một warp 0 nhanh ghi đè scratch[0] trước khi một warp chậm đọc max.">
            A fast warp 0 overwrites scratch[0] before a slow warp reads the max.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không gì cả, vì cùng lý do mà rào chắn ở dòng 19 của kernel trong sách là không cần thiết.">
            Nothing, for the same reason the barrier at line 19 of the book's kernel is not needed.
          </Translated>
        ),
        why: (
          <Translated vi="Lần ghi đầu tiên của warp 0 chỉ cần tổng của chính warp 0, không cần của mọi luồng: không có luồng dữ liệu nào giữ nó lại.">
            Warp 0's first write needs only warp 0's own sum, not every thread's: no data flow holds it back.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Max đến scratch[0] quá muộn với các luồng đọc nó.">
            The max arrives in scratch[0] too late for the threads that read it.
          </Translated>
        ),
        why: (
          <Translated vi="blockReduceMax kết thúc bằng một rào chắn trước khi có ai đọc scratch[0].">
            blockReduceMax ends with a barrier before anyone reads scratch[0].
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ các lane của warp 0 thấy max sai; các warp khác an toàn.">
            Only lanes of warp 0 see a wrong max; the other warps are safe.
          </Translated>
        ),
        why: (
          <Translated vi="Warp 0 là bên ghi. Nạn nhân là các warp khác, những warp đọc ô mà nó ghi đè.">
            Warp 0 is the writer. The victims are the other warps, which read the slot it overwrites.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Ở đây lần ghi đến trước trong phép rút gọn thứ hai chỉ phụ thuộc vào một warp. Một warp 0 nhanh có thể đến đó
            khi một warp khác chưa thực hiện lần đọc <code>scratch[0]</code> của nó: một ghi-sau-đọc, cùng mối nguy như
            rào chắn thứ hai của phép nhân ma trận phân ô. Chú thích của lab 10.3 ở rào chắn đó nói đúng điều này.
          </p>
        }
      >
        <p>
          Here the write that comes first in the second reduction depends on one warp only. A fast warp 0 can reach it
          while another warp has not executed its read of <code>scratch[0]</code>: a write-after-read, the same hazard as
          the tiled matmul&rsquo;s second barrier. Lab 10.3&rsquo;s comment at that barrier says exactly this.
        </p>
      </Translated>
    ),
    source: 'labs/ch10/lab3_softmax_argmax/solution/kernel.cu; ch. 5 §5.4', // i18n-ignore: citation
  },
  'ch20/20.3/q-griddim': {
    kind: 'choice',
    id: 'ch20/20.3/q-griddim',
    prompt: (
      <Translated
        vi={
          <>
            Sách nói một lưới 1D có <code>gridDim.y</code> và <code>gridDim.z</code> bằng 0 (p. 487). Một kernel tính số
            khối của nó là <code>gridDim.x * gridDim.y * gridDim.z</code>. Nó nhận được gì với một lần khởi chạy 1D gồm N
            khối?
          </>
        }
      >
        <>
          The book says a 1D grid has <code>gridDim.y</code> and <code>gridDim.z</code> equal to 0 (p. 487). A kernel
          computes its block count as <code>gridDim.x * gridDim.y * gridDim.z</code>. What does it get for a 1D launch of
          N blocks?
        </>
      </Translated>
    ),
    options: [
      { label: 'N.', correct: true }, // i18n-ignore: a symbol
      {
        label: '0.', // i18n-ignore: a number
        why: (
          <Translated vi="Đó là điều phát biểu của sách sẽ cho ra, và là lý do phát biểu đó là một sơ suất.">
            That is what the book's statement would give, and why the statement is a slip.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không xác định: các chiều không dùng không được khởi tạo.">
            Undefined: unused dimensions are not initialised.
          </Translated>
        ),
        why: (
          <Translated vi="dim3 gán giá trị mặc định cho mọi thành phần không được cho.">
            dim3 defaults every component it is not given.
          </Translated>
        ),
      },
      {
        label: <Translated vi="N trên HIP, 0 trên CUDA.">N on HIP, 0 on CUDA.</Translated>,
        why: <Translated vi="Cả hai đều mặc định là 1.">Both default to 1.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Các chiều không dùng bằng 1: &ldquo;the default value of all components of dim3 is 1&rdquo; (CUDA
            Programming Guide 5.4.2.2), và hàm dựng <code>dim3</code> của HIP cũng mặc định mỗi thành phần là 1. Trên
            Metal không có giá trị mặc định: host truyền cả ba thành phần của kích thước lưới.
          </p>
        }
      >
        <p>
          Unused dimensions are 1: &ldquo;the default value of all components of dim3 is 1&rdquo; (CUDA Programming
          Guide 5.4.2.2), and HIP&rsquo;s <code>dim3</code> constructor defaults each component to 1 as well. On Metal
          there is no default: the host passes all three components of the grid size.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch20-facts.md A1', // i18n-ignore: citation
  },
  'ch20/20.3/q-sp-bytes': {
    kind: 'numeric',
    id: 'ch20/20.3/q-sp-bytes',
    prompt: (
      <Translated vi={<>32 head, N = 4096, fp32: S và P cùng nhau chiếm bao nhiêu bộ nhớ cho một lớp của pipeline không gộp?</>}>
        <>32 heads, N = 4096, fp32: how much memory do S and P together take for one layer of the unfused pipeline?</>
      </Translated>
    ),
    answer: 4,
    unit: 'GiB',
    explain: (
      <Translated
        vi={
          <p>
            Mỗi ma trận là 4096² × 4 B = 64 MiB mỗi head; hai ma trận cho 32 head là 4 GiB, cho một lớp, ở một ngữ cảnh mà
            một cuộc trò chuyện đạt tới rất nhanh. Hãy đặt trình khám phá về các giá trị này.
          </p>
        }
      >
        <p>
          Each is 4096² × 4 B = 64 MiB per head; two of them for 32 heads is 4 GiB, for one layer, at a context a chat
          reaches quickly. Set the explorer to these values.
        </p>
      </Translated>
    ),
    source: 'our arithmetic from pp. 486–488; the explorer in this section',
    vi: { source: 'phép tính của chúng tôi từ pp. 486–488; trình khám phá trong mục này' },
  },
  'ch20/20.3/q-cub': {
    kind: 'choice',
    id: 'ch20/20.3/q-cub',
    prompt: (
      <Translated vi={<>Bài tập 2 của sách, ngày nay: bạn chuyển Fig. 20.4 sang một bản CUB hiện tại. Cái gì phải thay đổi?</>}>
        <>The book&rsquo;s Exercise 2, today: you port Fig. 20.4 to a current CUB. What must change?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Chỉ các functor (cub::Max(), cub::Sum()); luồng 0 vẫn giữ kết quả.">
            Only the functors (cub::Max(), cub::Sum()); thread 0 still holds the result.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không gì cả: các lời gọi CUB trong hình vẫn biên dịch được như khi in.">
            Nothing: the CUB calls in the figure still compile as printed.
          </Translated>
        ),
        why: (
          <Translated vi="cub::Max và cub::Sum đã bị gỡ bỏ trong CCCL 3.0.">
            cub::Max and cub::Sum were removed in CCCL 3.0.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Phần phát tán: BlockReduce giờ trả kết quả cho mọi luồng của khối.">
            The broadcast: BlockReduce now returns its result to every thread of the block.
          </Translated>
        ),
        why: (
          <Translated vi="Tài liệu vẫn ghi kết quả của nó chỉ hợp lệ ở luồng 0.">
            Its result is still documented as valid in thread 0 only.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Mọi thứ: BlockReduce đã bị gỡ bỏ để nhường cho cooperative groups.">
            Everything: BlockReduce was removed in favour of cooperative groups.
          </Translated>
        ),
        why: (
          <Translated vi="BlockReduce và WarpReduce vẫn còn hiện hành; chỉ cách viết functor thay đổi.">
            BlockReduce and WarpReduce are current; only the functor spellings changed.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Truyền <code>cuda::maximum&lt;&gt;{'{}'}</code> và <code>cuda::std::plus&lt;&gt;{'{}'}</code> (hướng dẫn
            chuyển đổi CCCL 3.0). Kết quả vẫn chỉ hợp lệ ở luồng 0, nên việc phát tán qua <code>max_or_sum</code> vẫn
            giữ. Trên HIP các lớp là <code>hipcub::</code>, với cách viết functor riêng. Đoạn mã trình chạy từng bước hiển
            thị dùng cách viết hiện tại.
          </p>
        }
      >
        <p>
          Pass <code>cuda::maximum&lt;&gt;{'{}'}</code> and <code>cuda::std::plus&lt;&gt;{'{}'}</code> (CCCL 3.0 migration
          guide). The result is still valid in thread 0 only, so the broadcast through <code>max_or_sum</code> stays. On
          HIP the classes are <code>hipcub::</code>, with their own functor spellings. The snippet the stepper shows has
          the current spellings.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch20-facts.md A2; infra-facts.md A11, B11', // i18n-ignore: citation
  },
  'ch20/20.3/q-decode-row': {
    kind: 'choice',
    id: 'ch20/20.3/q-decode-row',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel nhân quả cho nq truy vấn trên nk key (nq ≤ nk) cho hàng r thấy các cột tới r + nk − nq. Với nq = 1
            và nk = 500, hàng duy nhất thấy những cột nào?
          </>
        }
      >
        <>
          A causal kernel for nq queries against nk keys (nq ≤ nk) lets row r see columns up to r + nk − nq. With nq = 1
          and nk = 500, which columns does the single row see?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Cả 500: một bước decode một token không cần mặt nạ.">
            All 500: a one-token decode step needs no mask.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Chỉ cột 0.">Column 0 only.</Translated>,
        why: (
          <Translated vi="Đó là cách căn theo góc trên bên trái mà is_causal của PyTorch dùng cho một truy vấn ngắn: một lỗi decode.">
            That is the upper-left alignment PyTorch's is_causal uses for a short query: a decode bug.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Các cột 0 và 1.">Columns 0 and 1.</Translated>,
        why: (
          <Translated vi="r + nk − nq = 0 + 500 − 1 = 499, cột cuối cùng.">
            r + nk − nq = 0 + 500 − 1 = 499, the last column.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không cột nào; mặt nạ che mọi thứ sau đường chéo.">
            None; the mask hides everything after the diagonal.
          </Translated>
        ),
        why: (
          <Translated vi="Với cách căn theo góc dưới bên phải, đường chéo kết thúc ở góc dưới bên phải.">
            With lower-right alignment the diagonal ends in the bottom-right corner.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            r + nk − nq = 0 + 499: mọi cột. Mặt nạ được căn theo góc dưới bên phải, nên truy vấn mới nhất thấy mọi key, như
            trong Fig. 20.6, vốn không có mặt nạ. FlashAttention từ bản 2.1 và MLX dùng cách căn này;{' '}
            <code>is_causal</code> của PyTorch căn theo góc trên bên trái.
          </p>
        }
      >
        <p>
          r + nk − nq = 0 + 499: every column. The mask is aligned to the lower right, so the newest query sees every key,
          as in Fig. 20.6, which has no mask. FlashAttention since 2.1 and MLX use this alignment; PyTorch&rsquo;s{' '}
          <code>is_causal</code> aligns to the upper left.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch20-facts.md A8, D4, D5; ch. 20 Fig. 20.6', // i18n-ignore: citation
  },
};

// ---------------------------------------------------------------- the stepper
const TIERS: Record<string, Tier> = {
  S: 'global', P: 'global', L: 'global', S_row: 'global',
  temp_store: 'shared', max_or_sum: 'shared',
  max_val_thread: 'reg', max_val_row: 'reg', sum_thread: 'reg', sum_row: 'reg', idx: 'reg',
};
// Lab 20.1's names (labs/ch20/lab1_causal_softmax/solution/kernel.cu and kernel.metal).
const LAB_TIERS: Record<string, Tier> = {
  S: 'global', P: 'global', L: 'global', S_row: 'global', P_row: 'global',
  scratch: 'shared',
  m: 'reg', s: 'reg', v: 'reg', inv: 'reg', c: 'reg', row: 'reg', item: 'reg', visible: 'reg', val: 'reg',
};
const SNIPPET = 'ch20/snippets/s203_softmax_corrected.cu';
const fmtV = (x: number | null | undefined, digits = 4) => {
  if (x === null || x === undefined) return '';
  if (Number.isNaN(x)) return 'NaN';
  if (x === Infinity) return '∞';
  if (x === NEG_IDENTITY) return '−FLT_MAX';
  return String(Number(x.toPrecision(digits))).replace('-', '−');
};

const BARRIER_NOTE: Record<number, string> = {
  get 10() {
    return tr('Nothing before it needed protecting: this block has done nothing shared yet.', 'Không có gì trước nó cần bảo vệ: khối này chưa làm gì chung cả.');
  },
  get 13() {
    return tr('Thread 0 has stored the max; now every thread may read it.', 'Luồng 0 đã lưu max; giờ mọi luồng có thể đọc nó.');
  },
  get 19() {
    return tr('Every thread has read the max and finished its sum loop.', 'Mọi luồng đã đọc max và xong vòng lặp tính tổng.');
  },
  get 22() {
    return tr('Thread 0 has stored the sum; now every thread may read it.', 'Luồng 0 đã lưu tổng; giờ mọi luồng có thể đọc nó.');
  },
};

function describe(s: Step, cfg: SoftmaxConfig, row: number): string {
  const t = s.thread;
  const th = t >= 0 ? s.after.threads[t] : undefined;
  switch (s.op) {
    case 'release':
      return tr(
        `The barrier at line ${s.line} opens: every thread of the block has arrived. ${BARRIER_NOTE[s.line] ?? ''}`,
        `Rào chắn ở dòng ${s.line} mở: mọi luồng của khối đã đến. ${BARRIER_NOTE[s.line] ?? ''}`,
      );
    case 'reduced':
      return s.line === 11
        ? tr(
            `BlockReduce at line 11 completes: the row max is ${fmtV(s.value)}, and only thread 0 has it. The others' max_val_row is undefined until line 14.`,
            `BlockReduce ở dòng 11 hoàn tất: max của hàng là ${fmtV(s.value)}, và chỉ luồng 0 có nó. max_val_row của các luồng khác không xác định cho đến dòng 14.`,
          )
        : tr(
            `BlockReduce at line 20 completes: the row sum is ${fmtV(s.value)}, again in thread 0 only.`,
            `BlockReduce ở dòng 20 hoàn tất: tổng của hàng là ${fmtV(s.value)}, lại chỉ ở luồng 0.`,
          );
    case 'barrier':
      return tr(
        `Thread ${t} reaches the barrier at line ${s.line} and waits for the rest of the block.`,
        `Luồng ${t} đến rào chắn ở dòng ${s.line} và chờ phần còn lại của khối.`,
      );
    case 'reduceMax': {
      const none = columnsOf(cfg, row, t).length === 0;
      return tr(
        `Thread ${t} enters the max reduction (line 11) with its private max ${fmtV(th!.privMax)}${none ? ': it read no column, so this is the identity' : ''}.`,
        `Luồng ${t} vào phép rút gọn max (dòng 11) với max riêng ${fmtV(th!.privMax)}${none ? ': nó không đọc cột nào, nên đây là phần tử đơn vị' : ''}.`,
      );
    }
    case 'reduceSum':
      return tr(
        `Thread ${t} enters the sum reduction (line 20) with its private sum ${fmtV(th!.privSum)}.`,
        `Luồng ${t} vào phép rút gọn tổng (dòng 20) với tổng riêng ${fmtV(th!.privSum)}.`,
      );
    case 'max':
      return tr(
        `Thread ${t}, line 8: S[${row}][${s.idx}] = ${fmtV(s.value)}${s.masked ? ', a column after the row' : ''}; its private max is now ${fmtV(th!.privMax)}.`,
        `Luồng ${t}, dòng 8: S[${row}][${s.idx}] = ${fmtV(s.value)}${s.masked ? ', một cột sau hàng' : ''}; max riêng của nó giờ là ${fmtV(th!.privMax)}.`,
      );
    case 'writeMax':
      return tr(
        `Thread 0, line 12: stores the row max ${fmtV(s.value)} in max_or_sum.`,
        `Luồng 0, dòng 12: lưu max của hàng ${fmtV(s.value)} vào max_or_sum.`,
      );
    case 'readMax':
      return tr(
        `Thread ${t}, line 14: reads max_or_sum and gets ${fmtV(s.value)}.`,
        `Luồng ${t}, dòng 14: đọc max_or_sum và nhận ${fmtV(s.value)}.`,
      );
    case 'sum':
      return tr(
        `Thread ${t}, line 17: exp(S[${row}][${s.idx}]${cfg.subtractMax ? ' − max' : ''}) = ${fmtV(s.value)}; its private sum is now ${fmtV(th!.privSum)}.`,
        `Luồng ${t}, dòng 17: exp(S[${row}][${s.idx}]${cfg.subtractMax ? ' − max' : ''}) = ${fmtV(s.value)}; tổng riêng của nó giờ là ${fmtV(th!.privSum)}.`,
      );
    case 'writeSum':
      return tr(
        `Thread 0, line 21: stores the row sum ${fmtV(s.value)} in max_or_sum.`,
        `Luồng 0, dòng 21: lưu tổng của hàng ${fmtV(s.value)} vào max_or_sum.`,
      );
    case 'readSum':
      return tr(
        `Thread ${t}, line 23: reads max_or_sum and gets ${fmtV(s.value)}.`,
        `Luồng ${t}, dòng 23: đọc max_or_sum và nhận ${fmtV(s.value)}.`,
      );
    case 'out':
      return s.masked
        ? tr(
            `Thread ${t}, line 25: column ${s.idx} is after the row, so P[${row}][${s.idx}] = 0, written explicitly.`,
            `Luồng ${t}, dòng 25: cột ${s.idx} nằm sau hàng, nên P[${row}][${s.idx}] = 0, được ghi tường minh.`,
          )
        : tr(
            `Thread ${t}, line 25: P[${row}][${s.idx}] = exp(S${cfg.subtractMax ? ' − max' : ''}) / sum = ${fmtV(s.value)}.`,
            `Luồng ${t}, dòng 25: P[${row}][${s.idx}] = exp(S${cfg.subtractMax ? ' − max' : ''}) / sum = ${fmtV(s.value)}.`,
          );
    case 'lse':
      return tr(
        `Thread 0, line 27: L[${row}] = ${cfg.subtractMax ? 'max + log(sum)' : 'log(sum)'} = ${fmtV(s.value)}.`,
        `Luồng 0, dòng 27: L[${row}] = ${cfg.subtractMax ? 'max + log(sum)' : 'log(sum)'} = ${fmtV(s.value)}.`,
      );
  }
  return '';
}

function Stepper() {
  const [n, setN] = useState(8);
  const [block, setBlock] = useState(4);
  const [row, setRow] = useState(5);
  const [schedule, setSchedule] = useState<Schedule>('in-order');
  const [b10, setB10] = useState(true);
  const [b13, setB13] = useState(true);
  const [b19, setB19] = useState(true);
  const [b22, setB22] = useState(true);
  const [lost, setLost] = useState(false);
  const [subtract, setSubtract] = useState(true);
  const [offset, setOffset] = useState(0);
  const [k, setK] = useState(0);
  const [playing, setPlaying] = useState(false);

  const src = useLabFile(SNIPPET);
  const fn = src && !src.startsWith('// could not load') ? functionSource(src, 'softmax_kernel') : null;
  const r = Math.min(row, n - 1);
  const cfg: SoftmaxConfig = useMemo(
    () => ({ ...CORRECT_SOFTMAX, n, block, barrier10: b10, barrier13: b13, barrier19: b19, barrier22: b22, bound: lost ? 'n' : 'row', subtractMax: subtract }),
    [n, block, b10, b13, b19, b22, lost, subtract],
  );
  const input = useMemo(() => causalSoftmaxProblem(n, offset), [n, offset]);
  const trace = useMemo(() => runRow(cfg, input, r, schedule, 1), [cfg, input, r, schedule]);
  const total = trace.steps.length;
  const at = Math.min(k, total);
  const st = stateAt(cfg, trace, at);
  const cur = at > 0 ? trace.steps[at - 1] : undefined;
  const firstBad = trace.violations.length ? trace.violations[0].step + 1 : null;

  useEffect(() => {
    setK(0);
    setPlaying(false);
  }, [cfg, input, r, schedule]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setK((x) => x + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  const nextEvent = () => {
    for (let i = at; i < total; i++) if (trace.steps[i].thread < 0) return setK(i + 1);
    setK(total);
  };
  const doing = (t: number) => {
    const x = st.threads[t];
    if (x.done) return tr('finished', 'đã xong');
    if (x.waitingAt !== null)
      return x.waitingAt === 11 || x.waitingAt === 20
        ? tr(`in the reduction at line ${x.waitingAt}`, `trong phép rút gọn ở dòng ${x.waitingAt}`)
        : tr(`waiting at line ${x.waitingAt}`, `đang chờ ở dòng ${x.waitingAt}`);
    const last = [...trace.steps.slice(0, at)].reverse().find((s) => s.thread === t);
    return last ? tr(`after line ${last.line}`, `sau dòng ${last.line}`) : tr('not started', 'chưa bắt đầu');
  };
  const ref = input.refP[r];

  return (
    <Bench
      title={tr('Step one row', 'Chạy từng bước một hàng')}
      task={
        <Translated vi="Xem luồng 0 mang từng kết quả rút gọn qua max_or_sum. Rồi bỏ từng rào chắn một.">
          Watch thread 0 carry each reduction result through max_or_sum. Then remove one barrier at a time.
        </Translated>
      }
      foot={
        <>
          <Translated
            vi={
              <>
                Một khối của lưới: khối r sở hữu hàng r của S và ghi hàng r của P. Các ô nét đứt là các cột sau hàng, mà
                mặt nạ loại ra.
              </>
            }
          >
            <>
              One block of the grid: block r owns row r of S and writes row r of P. Dashed cells are columns after the row,
              which the mask keeps out.
            </>
          </Translated>{' '}
          <TierLegend />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          N
          <select value={n} onChange={(e) => setN(+e.target.value)} aria-label="N">
            {[4, 5, 6, 7, 8, 9, 10].map((x) => (
              <option key={x} value={x}>
                {x}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          BLOCK_SIZE
          <select value={block} onChange={(e) => setBlock(+e.target.value)} aria-label="BLOCK_SIZE">
            {[2, 4, 8].map((x) => (
              <option key={x} value={x}>
                {x}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <Translated vi="Khối (hàng)">Block (row)</Translated>
          <select value={r} onChange={(e) => setRow(+e.target.value)} aria-label={tr('row', 'hàng')}>
            {Array.from({ length: n }, (_, i) => (
              <option key={i} value={i}>
                {i}
              </option>
            ))}
          </select>
        </label>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Thứ tự luồng">Thread order</Translated>
          <select value={schedule} onChange={(e) => setSchedule(e.target.value as Schedule)} aria-label={tr('Thread order', 'Thứ tự luồng')} style={{ maxWidth: '100%' }}>
            {SCHEDULES.map((s) => (
              <option key={s} value={s}>
                {tr(SCHEDULE_LABEL[s], { lockstep: 'mọi luồng chạy đồng bộ từng bước (lockstep)', 'in-order': 'mỗi luồng chạy xa nhất có thể, luồng 0 trước', reversed: 'mỗi luồng chạy xa nhất có thể, luồng cuối trước', shuffled: 'xen kẽ ngẫu nhiên' }[s])}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted">
          <Translated vi="Rào chắn:">Barriers:</Translated>
        </span>
        {(
          [
            [10, b10, setB10],
            [13, b13, setB13],
            [19, b19, setB19],
            [22, b22, setB22],
          ] as const
        ).map(([line, on, set]) => (
          <label key={line} className="check">
            <input type="checkbox" checked={on} onChange={(e) => set(e.target.checked)} />{' '}
            <Translated vi={<>dòng {line}</>}>line {line}</Translated>
          </label>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="check">
          <input type="checkbox" checked={lost} onChange={(e) => setLost(e.target.checked)} />{' '}
          <Translated vi={<>cận vòng lặp idx &lt; N (mất mặt nạ)</>}>loop bound idx &lt; N (mask lost)</Translated>
        </label>
        <label className="check">
          <input type="checkbox" checked={subtract} onChange={(e) => setSubtract(e.target.checked)} />{' '}
          <Translated vi="trừ max của hàng">subtract the row max</Translated>
        </label>
        <label className="check">
          <input type="checkbox" checked={offset === 90} onChange={(e) => setOffset(e.target.checked ? 90 : 0)} />{' '}
          <Translated vi="logits + 90">logits + 90</Translated>
        </label>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setK(0); }} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => { setPlaying(false); setK(at - 1); }} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => { setPlaying(false); setK(at + 1); }} disabled={at >= total}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => { setPlaying(false); nextEvent(); }} disabled={at >= total}><Translated vi="Chạy đến rào chắn tiếp theo">Run to next barrier</Translated></button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Chạy')}</button>
        {firstBad ? <button onClick={() => { setPlaying(false); setK(firstBad); }}><Translated vi="Đến lỗi đầu tiên">Go to the first failure</Translated></button> : null}
        <span className="muted" aria-live="polite"><Translated vi={<>bước {at} trên {total}</>}>step {at} of {total}</Translated></span>
      </div>

      <div className={'narrate' + (cur?.violation ? ' bad' : '')} role="status" data-narrate style={{ marginBottom: '0.9rem' }}>
        {cur ? (
          <>
            {describe(cur, cfg, r)}
            {cur.violation ? (
              <>
                <br />
                <b className="t-fail">
                  <Translated
                    vi={
                      <>
                        Lỗi:{' '}
                        {((v) => {
                        const t = v.thread;
                        switch (v.kind) {
                          case 'masked-read':
                            return tr(v.message, `luồng ${t} đã đọc S[${r}][${cur.idx}], một cột thuộc tương lai của hàng ${r}`);
                          case 'read-before-write':
                            return v.message.includes('row max')
                              ? tr(v.message, `luồng ${t} đọc max_or_sum trước khi luồng 0 lưu max của hàng vào đó`)
                              : tr(
                                  v.message,
                                  `luồng ${t} đọc max_or_sum trước khi luồng 0 lưu tổng của hàng vào đó (nó vẫn còn giữ ${v.message.includes('still held the max') ? tr('the max', 'giá trị max') : tr('nothing', 'chưa có gì')})`,
                                );
                          case 'overwritten-before-read':
                            return tr(v.message, `luồng ${t} cần giá trị max, nhưng luồng 0 đã ghi đè max_or_sum bằng tổng`);
                          case 'overflow': {
                            // The message reads "exp(<x>) is beyond ...": reuse the argument as the simulator formatted it.
                            const arg = /^exp\(([^)]*)\)/.exec(v.message)?.[1] ?? '';
                            return tr(v.message, `exp(${arg}) vượt quá phạm vi của float32 (trên khoảng 88.72), nên nó là vô cực`);
                          }
                          case 'temp-reuse':
                            return tr(
                              v.message,
                              'phép rút gọn tổng dùng lại temp_store mà không có rào chắn nào kể từ khi phép rút gọn max kết thúc; CUB yêu cầu phải có một',
                            );
                        }
                          return v.message;
                        })(cur.violation)}
                        .
                      </>
                    }
                  >
                    Failure: {cur.violation.message}.
                  </Translated>
                </b>
              </>
            ) : null}
          </>
        ) : (
          <Translated
            vi={
              <>
                Chưa có gì chạy. Khối {r} sở hữu hàng {r}: {r + 1} cột nhìn thấy, chia cho {block} luồng. Nhấn Bước.
              </>
            }
          >
            <>
              Nothing has run yet. Block {r} owns row {r}: {r + 1} visible column{r ? 's' : ''}, shared among {block} threads.
              Press Step.
            </>
          </Translated>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
        <div style={{ overflowX: 'auto' }}>
          <table className="threads" aria-label={tr('Threads of the block', 'Các luồng của khối')}>
            <thead>
              <tr>
                <th><Translated vi="luồng">thread</Translated></th>
                <th><Translated vi="cột">columns</Translated></th>
                <th>max_val_thread{/* i18n-ignore: identifier */}</th>
                <th>max_val_row{/* i18n-ignore: identifier */}</th>
                <th>sum_thread{/* i18n-ignore: identifier */}</th>
                <th>sum_row{/* i18n-ignore: identifier */}</th>
              </tr>
            </thead>
            <tbody>
              {st.threads.map((x, t) => (
                <tr key={t} className={cur?.thread === t ? 'cur' : ''} data-thread={t}>
                  <td title={doing(t)}>{t}</td>
                  <td>{columnsOf(cfg, r, t).join(', ') || <span className="muted"><Translated vi="không có">none</Translated></span>}</td>
                  <td className="t-reg">{fmtV(x.privMax)}</td>
                  <td className="t-reg">{Number.isNaN(x.maxRow) ? <span className="muted"><Translated vi="không xác định">undefined</Translated></span> : fmtV(x.maxRow)}</td>
                  <td className="t-reg">{fmtV(x.privSum)}</td>
                  <td className="t-reg">{Number.isNaN(x.sumRow) ? <span className="muted"><Translated vi="không xác định">undefined</Translated></span> : fmtV(x.sumRow)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.4rem 0 0.9rem', fontSize: '0.78rem' }}>
            {st.threads.map((_, t) => tr(`thread ${t}: ${doing(t)}`, `luồng ${t}: ${doing(t)}`)).join(' · ')}
          </p>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <div className="row" style={{ alignItems: 'flex-start', gap: '1rem', marginBottom: '0.8rem' }}>
            <div>
              <div className="mx-label t-shared">
                <Translated vi="max_or_sum (chia sẻ)">max_or_sum (shared)</Translated>
              </div>
              <div className="mx shared" data-shared={st.shared.holds}>
                <span className={cur && (cur.op === 'writeMax' || cur.op === 'writeSum' || cur.op === 'readMax' || cur.op === 'readSum') ? (cur.violation ? 'bad' : 'hot') : ''}>
                  {st.shared.holds === 'unset' ? '·' : fmtV(st.shared.value)}
                </span>
              </div>
              <div className="muted" style={{ fontSize: '0.74rem' }}>
                {tr(
                  `holds ${st.shared.holds === 'unset' ? 'nothing yet' : 'the ' + st.shared.holds}`,
                  `chứa ${st.shared.holds === 'unset' ? 'chưa có gì' : st.shared.holds === 'max' ? 'giá trị max' : 'tổng'}`,
                )}
              </div>
            </div>
            <div>
              <div className="mx-label t-global">
                <Translated vi={<>L[{r}] (toàn cục)</>}>L[{r}] (global)</Translated>
              </div>
              <div className="mx global">
                <span className={st.L === null ? '' : trace.lOk || at < total ? 'on' : 'bad'}>{st.L === null ? '' : fmtV(st.L)}</span>
              </div>
            </div>
          </div>
          <div className="mx-label t-global">
            <Translated vi={<>S hàng {r} (toàn cục)</>}>S row {r} (global)</Translated>
          </div>
          <div className="mx global" data-srow style={{ gridTemplateColumns: `repeat(${n}, auto)`, marginBottom: '0.7rem' }}>
            {input.S[r].map((v, j) => (
              <span key={j} className={cur?.idx === j && (cur.op === 'max' || cur.op === 'sum') ? (cur.masked ? 'bad' : 'hot') : j > r ? 'void' : 'on'}>
                {fmtV(v, 3)}
              </span>
            ))}
          </div>
          <div className="mx-label t-global">
            <Translated vi={<>P hàng {r} (toàn cục)</>}>P row {r} (global)</Translated>
          </div>
          <div className="mx global" data-prow style={{ gridTemplateColumns: `repeat(${n}, auto)` }}>
            {st.P.map((v, j) => (
              <span
                key={j}
                data-v={v === null ? '' : String(v)}
                className={v === null ? (j > r ? 'void' : '') : Math.abs(v - ref[j]) <= 2e-6 * Math.max(1, Math.abs(ref[j])) ? 'good' : 'bad'}
              >
                {v === null ? '' : fmtV(v, 3)}
              </span>
            ))}
          </div>
          {at >= total ? (
            <p style={{ margin: '0.9rem 0 0' }} data-result={trace.correct ? 'correct' : 'wrong'}>
              {trace.correct ? (
                <span className="chip ok">
                  <Translated vi={<>Hàng {r} khớp với softmax tham chiếu.</>}>Row {r} matches the reference softmax.</Translated>
                </span>
              ) : (
                <span className="chip fail">
                  <Translated
                    vi={
                      <>
                        Hàng {r} sai{trace.wrongCols.length ? ` ở ${trace.wrongCols.length} trên ${n} cột` : ''}
                        {!trace.lOk ? (trace.wrongCols.length ? ' và ở L' : ' ở L') : ''}.
                      </>
                    }
                  >
                    Row {r} is wrong{trace.wrongCols.length ? ` in ${trace.wrongCols.length} of ${n} columns` : ''}
                    {!trace.lOk ? (trace.wrongCols.length ? ' and in L' : ' in L') : ''}.
                  </Translated>
                </span>
              )}{' '}
              {trace.violations.length ? (
                <span className="muted">
                  <Translated vi={<>{trace.violations.length} vấn đề được đánh dấu trên đường đi.</>}>
                    {trace.violations.length} problem{trace.violations.length > 1 ? 's' : ''} flagged along the way.
                  </Translated>
                </span>
              ) : null}
            </p>
          ) : null}
        </div>
      </div>
      {fn ? (
        <Code
          tabs={[
            {
              label: tr('softmax_kernel, corrected', 'softmax_kernel, đã sửa'),
              code: fn.text,
              tiers: TIERS,
              status: tr('front-end-checked; unverified on CUDA and HIP', 'front-end-checked; unverified trên CUDA và HIP'),
            },
          ]}
          highlight={cur ? [cur.line] : []}
        />
      ) : (
        <p className="muted">
          {src ? tr(`Could not load labs/${SNIPPET}.`, `Không tải được labs/${SNIPPET}.`) : tr('Loading the kernel.', 'Đang tải kernel.')}
        </p>
      )}
    </Bench>
  );
}

const KIND_TEXT: Record<string, string> = {
  get 'read-before-write'() {
    return tr('read before write', 'đọc trước khi ghi');
  },
  get 'overwritten-before-read'() {
    return tr('overwritten before read', 'bị ghi đè trước khi đọc');
  },
  get 'masked-read'() {
    return tr('read the future', 'đọc tương lai');
  },
  get overflow() {
    return tr('exp overflowed', 'exp bị tràn');
  },
  get 'temp-reuse'() {
    return tr('temp_store reused unguarded', 'temp_store bị dùng lại không được bảo vệ');
  },
};

function Outcomes() {
  const N = 8;
  const B = 4;
  const inputs = useMemo(() => ({ 0: causalSoftmaxProblem(N, 0), 90: causalSoftmaxProblem(N, 90) }), []);
  const rows = useMemo(
    () =>
      VARIANTS.map((v) => ({
        v,
        cells: SCHEDULES.map((s) => runAllRows({ ...CORRECT_SOFTMAX, n: N, block: B, ...v.over }, inputs[v.offset === 90 ? 90 : 0], s, 1)),
      })),
    [inputs],
  );
  return (
    <Bench
      title={tr('Every thread order at once', 'Mọi thứ tự luồng cùng lúc')}
      task={
        <Translated vi="Trước khi nhìn: bạn nghĩ rào chắn nào sẽ quan trọng, và dưới những thứ tự nào?">
          Before you look: which barriers do you expect to matter, and under which orders?
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Mỗi ô chạy cả {N} khối của một bài toán {N} × {N} với BLOCK_SIZE = {B} trong mô hình ở trên và so sánh P và L
              với bản tham chiếu.
            </>
          }
        >
          <>Each cell runs all {N} blocks of an {N} × {N} problem with BLOCK_SIZE = {B} in the model above and compares P and L with the reference.</>
        </Translated>
      }
    >
      <div style={{ overflowX: 'auto' }}>
        <table className="data" data-table="outcomes" style={{ background: 'var(--page)', margin: 0 }}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>
                <Translated vi="Kernel">Kernel</Translated>
              </th>
              {SCHEDULES.map((s) => (
                <th key={s}>{tr(SCHEDULE_LABEL[s], { lockstep: 'mọi luồng chạy đồng bộ từng bước (lockstep)', 'in-order': 'mỗi luồng chạy xa nhất có thể, luồng 0 trước', reversed: 'mỗi luồng chạy xa nhất có thể, luồng cuối trước', shuffled: 'xen kẽ ngẫu nhiên' }[s])}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ v, cells }) => (
              <tr key={v.id} data-variant={v.id}>
                <td style={{ paddingLeft: '0.6rem' }}>{tr(
                    v.label,
                    (
                      {
                        'as-corrected': 'cả bốn rào chắn',
                        no10: 'không có rào chắn ở dòng 10',
                        no13: 'không có rào chắn ở dòng 13',
                        no19: 'không có rào chắn ở dòng 19',
                        no22: 'không có rào chắn ở dòng 22',
                        bound: 'cận vòng lặp idx < N (mất mặt nạ)',
                        nomax: 'không trừ max',
                        nomax90: 'không trừ max, logits + 90',
                      } as Record<string, string>
                    )[v.id] ?? v.label,
                  )}</td>
                {cells.map((c, i) => (
                  <td key={i} data-outcome={c.correct ? 'correct' : 'wrong'}>
                    {c.correct ? (
                      <span className="chip ok">
                        <Translated vi="đúng">correct</Translated>
                      </span>
                    ) : (
                      <span className="chip fail">
                        <Translated vi={<>{c.wrongRows.length} trên {N} hàng sai</>}>
                          {c.wrongRows.length} of {N} rows wrong
                        </Translated>
                      </span>
                    )}
                    {c.kinds.map((k) => (
                      <div key={k} className="muted">
                        {KIND_TEXT[k]}
                      </div>
                    ))}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- what the unfused pipeline moves
const fmtBytes = (b: number) => {
  if (b >= 2 ** 30) return (b / 2 ** 30).toPrecision(3) + ' GiB';
  if (b >= 2 ** 20) return (b / 2 ** 20).toPrecision(3) + ' MiB';
  return (b / 2 ** 10).toPrecision(3) + ' KiB';
};

function Traffic() {
  const [logN, setLogN] = useState(12);
  const [d, setD] = useState(128);
  const [heads, setHeads] = useState(32);
  const [bytes, setBytes] = useState(4);
  const n = 2 ** logN;
  const t = unfusedTraffic(n, d, heads, bytes);
  const parts = [
    { key: 'gemm1', label: tr('S = Q K^T (writes S)', 'S = Q K^T (ghi S)'), v: t.gemm1.read + t.gemm1.write, alpha: 0.85 },
    {
      key: 'softmax',
      label: tr(
        'mask + softmax (reads the visible half of S three times, writes P)',
        'mặt nạ + softmax (đọc nửa nhìn thấy của S ba lần, ghi P)',
      ),
      v: t.softmax.read + t.softmax.write,
      alpha: 0.55,
    },
    { key: 'gemm2', label: tr('O = P V (reads P)', 'O = P V (đọc P)'), v: t.gemm2.read + t.gemm2.write, alpha: 0.28 },
  ];
  const ratio = t.total / t.irreducible;
  return (
    <Bench
      title={tr('What the unfused pipeline moves', 'Pipeline không gộp di chuyển những gì')}
      task={
        <Translated vi="Đặt 32 head ở N = 4096 với fp32 và đọc dung lượng bộ nhớ mà S và P chiếm. Rồi nhân đôi N.">
          Set 32 heads at N = 4096 in fp32 and read the memory S and P hold. Then double N.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Cách đếm của chúng tôi: lượng tối thiểu mỗi kernel phải di chuyển như đã viết, mỗi đầu vào đọc một lần và mỗi
              đầu ra ghi một lần. L được tính ở fp32.
            </>
          }
        >
          <>Our count: the minimum each kernel must move as written, every input read once and every output written once. L is counted in fp32.</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field" style={{ minWidth: '12rem' }}>
          <Translated vi="số token N:">tokens N:</Translated> <b data-n>{n.toLocaleString('en-US')}</b>
          <input type="range" min={7} max={15} step={1} value={logN} onChange={(e) => setLogN(+e.target.value)} aria-label={tr('N (log scale)', 'N (thang log)')} />
        </label>
        <label className="field">
          <Translated vi="độ rộng head d">head width d</Translated>
          <select value={d} onChange={(e) => setD(+e.target.value)} aria-label={tr('head width', 'độ rộng head')}>
            <option value={64}>64</option>
            <option value={128}>128</option>
          </select>
        </label>
        <label className="field" style={{ minWidth: '10rem' }}>
          <Translated vi="số head:">heads:</Translated> <b data-heads>{heads}</b>
          <input type="range" min={1} max={64} value={heads} onChange={(e) => setHeads(+e.target.value)} aria-label={tr('heads', 'số head')} />
        </label>
        <span className="seg" role="group" aria-label={tr('Bytes per element', 'Số byte mỗi phần tử')}>
          {[4, 2].map((b) => (
            <button key={b} aria-pressed={bytes === b} onClick={() => setBytes(b)}>
              {b === 4 ? 'fp32' : '16-bit'}
            </button>
          ))}
        </span>
      </div>
      <div className="mx-label t-global">
        <Translated vi="Lưu lượng bộ nhớ toàn cục, một lớp">Global-memory traffic, one layer</Translated>
      </div>
      <div data-traffic style={{ display: 'flex', height: '1.6rem', border: '1px solid var(--rule)', borderRadius: 3, overflow: 'hidden' }}>
        {parts.map((p) => (
          <div key={p.key} data-part={p.key} title={`${p.label}: ${fmtBytes(p.v)}`} style={{ width: `${(100 * p.v) / t.total}%`, background: 'var(--global)', opacity: p.alpha }} />
        ))}
      </div>
      <div className="legend" style={{ marginTop: '0.4rem', display: 'flex', flexWrap: 'wrap', gap: '0.3rem 1rem' }}>
        {parts.map((p) => (
          <span key={p.key}>
            <i style={{ background: 'var(--global)', opacity: p.alpha }} />
            {p.label}: {fmtBytes(p.v)}
          </span>
        ))}
      </div>
      <div className="mx-label t-global" style={{ marginTop: '0.9rem' }}>
        <Translated vi="Đọc Q, K, V một lần và ghi O một lần">Read Q, K, V once and write O once</Translated>
      </div>
      <div style={{ display: 'flex', height: '1.6rem', border: '1px solid var(--rule)', borderRadius: 3, overflow: 'hidden' }}>
        <div data-part="irreducible" style={{ width: `${Math.max(0.4, (100 * t.irreducible) / t.total)}%`, background: 'var(--global)' }} />
      </div>
      <p style={{ margin: '0.9rem 0 0' }} data-readout>
        <Translated
          vi={
            <>
              Pipeline di chuyển <b data-total={t.total}>{fmtBytes(t.total)}</b>, gấp{' '}
              <b data-ratio={ratio.toFixed(2)}>{ratio.toFixed(1)}</b> lần {fmtBytes(t.irreducible)} mà mọi kernel attention
              đều phải di chuyển. Giữa các kernel, S và P chiếm <b data-sp={t.sp}>{fmtBytes(t.sp)}</b> bộ nhớ device.
            </>
          }
        >
          The pipeline moves <b data-total={t.total}>{fmtBytes(t.total)}</b>, <b data-ratio={ratio.toFixed(2)}>{ratio.toFixed(1)}</b> times the{' '}
          {fmtBytes(t.irreducible)} any attention kernel must move. Between the kernels S and P occupy{' '}
          <b data-sp={t.sp}>{fmtBytes(t.sp)}</b> of device memory.
        </Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- lab 20.1's reference solution
const [LAB1] = defineLabs('ch20', [{ num: '20.1', dir: 'lab1_causal_softmax', name: 'causal softmax', vi: { name: 'softmax nhân quả' } }]);

function LabListing() {
  const { reports } = useReports();
  const cu = useLabFile('ch20/lab1_causal_softmax/solution/kernel.cu');
  const metal = useLabFile('ch20/lab1_causal_softmax/solution/kernel.metal');
  const mm = useLabFile('ch20/lab1_causal_softmax/solution/launch.mm');
  const m = solutionTabStatus(reports, LAB1, 'kernel.metal');
  if (!cu || !metal || !mm)
    return (
      <p className="muted">
        <Translated vi="Đang tải lời giải tham chiếu của lab 20.1.">Loading lab 20.1&rsquo;s reference solution.</Translated>
      </p>
    );
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu, tiers: LAB_TIERS, status: solutionTabStatus(reports, LAB1, 'kernel.cu') },
        { label: tr('Metal kernel', 'Kernel Metal'), code: metal, tiers: LAB_TIERS, status: m },
        { label: tr('Metal host', 'Host Metal'), code: mm, tiers: {}, status: m },
      ]}
      caption={
        <Translated
          vi={
            <>
              Lời giải tham chiếu của lab 20.1, được tải từ <code>labs/ch20/lab1_causal_softmax/solution/</code>. Hãy tự
              viết bản của bạn trong lab trước. Bản build HIP được thiết lập để đưa tệp này qua{' '}
              <code>pmpp/vendors/hip.h</code>, điều chỉ khả thi vì mã warp của nó dùng một lane mask có kiểu và{' '}
              <code>warpSize</code>, không bao giờ dùng hằng <code>0xffffffff</code> hay một số 32 cố định; nó chưa được
              biên dịch hay chạy trên phần cứng AMD (unverified).
            </>
          }
        >
          <>
            The reference solution of lab 20.1, loaded from <code>labs/ch20/lab1_causal_softmax/solution/</code>. Write
            your own in the lab first. The HIP build is set up to take this file through <code>pmpp/vendors/hip.h</code>,
            which can work only because its warp code uses a typed lane mask and <code>warpSize</code>, never a{' '}
            <code>0xffffffff</code> literal or a fixed 32; it has not been compiled or run on AMD hardware (unverified).
          </>
        </Translated>
      }
    />
  );
}

export function S203() {
  return (
    <>
      <Translated
        vi={
          <p>
            Một head attention gồm hai phép nhân ma trận, một phép co giãn, một mặt nạ và một softmax. Các phép nhân có
            thể là kernel của chính bạn hoặc một GEMM thư viện (cuBLAS gộp 1/√d vào α của nó); sách nói mảnh mới duy nhất
            là softmax và chỉ trình bày phần đó. Trong các mốc dự án của bạn, pipeline là ba lời gọi, với S và P được ghi
            ra bộ nhớ toàn cục (global memory) giữa chúng:
          </p>
        }
      >
        <p>
          One head of attention is two matrix products, a scaling, a mask and a softmax. The products can be your own
          kernels or a library GEMM (cuBLAS folds the 1/√d into its α); the book says the only new piece is the softmax
          and shows only that. In your milestones the pipeline is three calls, with S and P written to global memory
          between them:
        </p>
      </Translated>
      <Source section="20.3" pages="486" />
      <ul>
        <li>
          <Translated
            vi={
              <>
                <code>launchLinear(Q, K, S, N, d, N)</code> (M3, lab 15.6): S = Q K<sup>T</sup>, vì K vốn đã được lưu như
                một trọng số <code>nn.Linear</code> với N đầu ra;
              </>
            }
          >
            <code>launchLinear(Q, K, S, N, d, N)</code> (M3, lab 15.6): S = Q K<sup>T</sup>, because K is already stored
            like an <code>nn.Linear</code> weight with N outputs;
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                một kernel mặt nạ và softmax trên S, ghi P: mục này, và lab 20.1 (tự áp dụng 1/√d, vì{' '}
                <code>launchLinear</code> không có α);
              </>
            }
          >
            a mask-and-softmax kernel over S, writing P: this section, and lab 20.1 (which applies the 1/√d itself, since{' '}
            <code>launchLinear</code> has no α);
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <code>launchGemm(P, V, O, N, N, d)</code> (M2, lab 5.3): O = P V, cả hai toán hạng theo thứ tự hàng
                (row-major) như khi lưu.
              </>
            }
          >
            <code>launchGemm(P, V, O, N, N, d)</code> (M2, lab 5.3): O = P V, both operands row-major as stored.
          </Translated>
        </li>
      </ul>

      <h2>
        <Translated vi="Một khối cho mỗi hàng">One block per row</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Kernel của sách khởi chạy một khối (block) cho mỗi hàng của S, nên <code>blockIdx.x</code> là hàng. Mỗi luồng
            (thread) nhận các cột <code>threadIdx.x</code>, <code>threadIdx.x + BLOCK_SIZE</code>, v.v. của hàng, phép
            rút gọn có gộp việc của chương 10. Mặt nạ nhân quả không gì khác ngoài cận vòng lặp: các vòng lặp max và tổng
            dừng ở <code>idx &lt;= blockIdx.x</code>, nên các phần tử bị che không bao giờ được đọc, và không có −∞ nào
            được cộng vào. Rồi một max trên khối, một tổng trên khối của exp(S − max), và một vòng lặp cuối qua cả N cột
            ghi các xác suất và các số 0 tường minh phía trên đường chéo. Gộp mặt nạ vào softmax tiết kiệm một lượt đọc
            qua S.
          </p>
        }
      >
        <p>
          The book&rsquo;s kernel launches one block per row of S, so <code>blockIdx.x</code> is the row. Each thread takes
          the row&rsquo;s columns <code>threadIdx.x</code>, <code>threadIdx.x + BLOCK_SIZE</code> and so on, chapter
          10&rsquo;s coarsened reduction. The causal mask is nothing but the loop bound: the max and sum loops stop at{' '}
          <code>idx &lt;= blockIdx.x</code>, so masked entries are never read, and no −∞ is ever added. Then a block max,
          a block sum of exp(S − max), and a final loop over all N columns that writes the probabilities and explicit
          zeros above the diagonal. Folding the mask into the softmax saves a pass over S.
        </p>
      </Translated>
      <Source section="20.3" pages="486–488" figs={tr('Fig. 20.4', 'Hình 20.4')} />
      <Translated
        vi={
          <p>
            <code>BlockReduce</code> chỉ trả kết quả cho luồng 0. Vì thế luồng 0 công bố mỗi kết quả qua biến vô hướng
            chia sẻ <code>max_or_sum</code>, một rào chắn (barrier) cho mọi luồng bắt kịp, và mọi luồng đọc lại nó. Đoạn
            mã trong trình chạy từng bước là Fig. 20.4 từng dòng một, được sửa ở chỗ trang sách đã lỗi thời (bảng phía
            dưới liệt kê từng thay đổi). Hãy chạy từng bước một hàng qua nó.
          </p>
        }
      >
        <p>
          <code>BlockReduce</code> returns its result to thread 0 only. So thread 0 publishes each result through the
          shared scalar <code>max_or_sum</code>, a barrier lets everyone catch up, and every thread reads it back. The
          listing in the stepper is Fig. 20.4 line for line, corrected where the page is out of date (the table further
          down lists each change). Step one row through it.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một luồng tiến từng dòng mã nguồn một. BlockReduce là một bước tập thể: mọi luồng đến, rồi kết quả xuất hiện
              ở luồng 0. Bộ nhớ tạm của nó không được mô hình hóa. Bộ nhớ chia sẻ chưa được ghi đọc ra NaN. Không có warp,
              không có thời gian.
            </p>
          }
        >
          <p>
            A thread advances one source line at a time. BlockReduce is one collective step: every thread arrives, then the
            result appears in thread 0. Its temporary storage is not modelled. Unwritten shared memory reads as NaN. No
            warps, no timing.
          </p>
        </Translated>
      </Simplification>
      <Stepper />
      <Ask q={questionsS203['ch20/20.3/q-cols']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Các khối mất cân bằng nặng, và sách chấp nhận điều đó: khối r có r + 1 cột còn hiệu lực, nên khối cho hàng 0
            gần như không làm gì trong khi khối cuối đọc cả hàng. Trong một khối, các luồng nằm quá cuối hàng mang phần tử
            đơn vị vào cả hai phép rút gọn.
          </p>
        }
      >
        <p>
          Blocks are badly balanced, and the book accepts it: block r has r + 1 live columns, so the block for row 0 does
          almost nothing while the last block reads the whole row. Inside a block, threads past the row&rsquo;s end bring
          the identities to both reductions.
        </p>
      </Translated>
      <Ask q={questionsS203['ch20/20.3/q-idle']} />

      <h2>
        <Translated vi="Kernel cần những rào chắn nào">Which barriers the kernel needs</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Fig. 20.4 có bốn rào chắn, ở các dòng 10, 13, 19 và 22. Hai trong số đó bảo vệ một phụ thuộc thật. Dòng 13 ngăn
            mọi luồng đọc <code>max_or_sum</code> trước khi luồng 0 lưu max vào đó, và dòng 22 làm điều tương tự cho tổng:
            đọc-sau-ghi, rào chắn đầu tiên của chương 5.
          </p>
        }
      >
        <p>
          Fig. 20.4 has four barriers, at lines 10, 13, 19 and 22. Two of them guard a true dependence. Line 13 stops
          any thread from reading <code>max_or_sum</code> before thread 0 has stored the max in it, and line 22 does the
          same for the sum: read-after-write, chapter 5&rsquo;s first barrier.
        </p>
      </Translated>
      <Source section="20.3" pages="487–488" />
      <Translated
        vi={
          <p>
            Hai rào chắn còn lại ít hiển nhiên hơn, và sách không nói chúng bảo vệ gì. Dòng 19 nằm ở chỗ một ghi-sau-đọc
            lẽ ra được bảo vệ: luồng 0 ghi đè max bằng tổng ở dòng 21. Nhưng luồng 0 không thể đến đó sớm. Tổng nó ghi là
            kết quả của một phép rút gọn trên tổng riêng của mọi luồng, và mỗi luồng chỉ tạo tổng riêng sau khi đọc max.
            Dòng 13 cũng là rào chắn mà CUB yêu cầu giữa hai lần dùng cùng một <code>temp_store</code>. Dòng 10 đến trước
            khi có bất cứ điều gì chung xảy ra. Hãy bỏ từng cái trong trình chạy từng bước dưới thứ tự &ldquo;luồng cuối
            trước&rdquo;, rồi kiểm tra bảng.
          </p>
        }
      >
        <p>
          The other two are less obvious, and the book does not say what they protect. Line 19 sits where a
          write-after-read would be guarded: thread 0 overwrites the max with the sum at line 21. But thread 0 cannot get
          there early. The sum it writes is the result of a reduction over every thread&rsquo;s private sum, and each
          thread makes its private sum only after reading the max. Line 13 is also the barrier CUB asks for between two
          uses of the same <code>temp_store</code>. Line 10 comes before anything shared has happened. Remove each one in
          the stepper under the &ldquo;last thread first&rdquo; order, then check the table.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Những rào chắn nào cần thiết là phân tích của chúng tôi về đoạn mã, được mô hình kiểm tra ở mọi thứ tự luồng.
              Quy tắc dùng lại của CUB lấy từ tài liệu của nó (facts A2). Với một vòng lặp ngoài qua các hàng, cùng lập
              luận đó áp dụng cho dòng 10.
            </p>
          }
        >
          <p>
            Which barriers are needed is our analysis of the listing, checked by the model in every thread order. CUB&rsquo;s
            reuse rule is from its documentation (facts A2). With an outer loop over rows, the same argument covers line 10.
          </p>
        </Translated>
      </Beyond>
      <Outcomes />
      <Ask q={questionsS203['ch20/20.3/q-barrier']} />
      <Translated
        vi={
          <p>
            Đừng kết luận rằng một rào chắn giữa hai phép rút gọn không bao giờ cần thiết. Lab 20.1 dùng lại phép rút gọn
            M4 của bạn, vốn phát tán qua <code>scratch[0]</code>, chính ô mà phép rút gọn tiếp theo ghi đầu tiên. Lần ghi
            đó chỉ cần giá trị của một warp, nên không gì khác sắp thứ tự nó, và rào chắn giữa hai lời gọi là thiết yếu ở
            đó.
          </p>
        }
      >
        <p>
          Do not conclude that a barrier between two reductions is never needed. Lab 20.1 reuses your M4 reduction, which
          broadcasts through <code>scratch[0]</code>, the same slot the next reduction writes first. That write needs only
          one warp&rsquo;s values, so nothing else orders it, and the barrier between the two calls is essential there.
        </p>
      </Translated>
      <Ask q={questionsS203['ch20/20.3/q-war-lab']} />
      <Translated
        vi={
          <p>
            Hai lỗi còn lại trong trình chạy từng bước thất bại như nhau ở mọi thứ tự. Với <code>idx &lt; N</code>, max và
            tổng thu nạp cả tương lai, nên mọi hàng trừ hàng cuối đều ra sai, dù vòng lặp đầu ra vẫn ghi số 0 phía trên
            đường chéo. Không trừ max, kernel đúng với logits bình thường và chuyển thành NaN khi một logit vượt khoảng
            88.7, giới hạn float32 của exp: hãy đánh dấu &ldquo;logits + 90&rdquo;.
          </p>
        }
      >
        <p>
          The two other defects in the stepper fail every order alike. With <code>idx &lt; N</code> the max and the sum
          take in the future, so every row but the last comes out wrong, although the output loop still writes zeros above
          the diagonal. Without the max subtraction the kernel is right at ordinary logits and turns to NaN once a logit
          passes about 88.7, the float32 limit of exp: tick &ldquo;logits + 90&rdquo;.
        </p>
      </Translated>

      <h2>
        <Translated vi="Như khi in và như lab viết">As printed and as the lab writes it</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Như khi in, đoạn mã cần một <code>NEG_INFINITY</code> mà CUDA không công bố và các functor CUB mà CCCL 3.0 đã
            gỡ bỏ, và văn bản xung quanh nói rằng các chiều lưới không dùng bằng 0. Lab 20.1 yêu cầu bạn viết cùng kernel
            cho ba backend, nên nó giải quyết từng điểm:
          </p>
        }
      >
        <p>
          As printed, the listing needs a <code>NEG_INFINITY</code> that CUDA does not document and CUB functors that
          CCCL 3.0 removed, and the text around it says that unused grid dimensions are 0. Lab 20.1 asks you to write the
          same kernel for three backends, so it settles each point:
        </p>
      </Translated>
      <table className="data wide" data-table="corrections" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr>
            <th>
              <Translated vi="Như khi in">As printed</Translated>
            </th>
            <th>
              <Translated vi="Như lab viết">As the lab writes it</Translated>
            </th>
            <th>
              <Translated vi="Lý do">Why</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <Translated
                vi={
                  <>
                    <code>gridDim.y</code> và <code>gridDim.z</code> của một lưới 1D &ldquo;bằng 0&rdquo; (p. 487)
                  </>
                }
              >
                A 1D grid&rsquo;s <code>gridDim.y</code> and <code>gridDim.z</code> &ldquo;are 0&rdquo; (p. 487)
              </Translated>
            </td>
            <td>
              <Translated
                vi={
                  <>
                    Chúng bằng 1; <code>dim3</code> của HIP cũng mặc định là 1; một host Metal truyền cả ba
                  </>
                }
              >
                They are 1; HIP&rsquo;s <code>dim3</code> also defaults to 1; a Metal host passes all three
              </Translated>
            </td>
            <td>facts A1{/* i18n-ignore: fact reference */}</td>
          </tr>
          <tr>
            <td>
              <code>NEG_INFINITY</code>
            </td>
            <td>
              <Translated
                vi={
                  <>
                    <code>-FLT_MAX</code>, phần tử đơn vị của M4; che theo chỉ số, nên không bao giờ lưu vô cực
                  </>
                }
              >
                <code>-FLT_MAX</code>, M4&rsquo;s identity; masking by index, so no infinity is ever stored
              </Translated>
            </td>
            <td>
              <Translated vi="không phải tên của CUDA (A5); vô cực là không xác định dưới fast math của Metal (C5)">
                not a CUDA name (A5); infinities are undefined under Metal&rsquo;s fast math (C5)
              </Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated
                vi={
                  <>
                    <code>cub::BlockReduce</code> với <code>cub::Max()</code>, <code>cub::Sum()</code>
                  </>
                }
              >
                <code>cub::BlockReduce</code> with <code>cub::Max()</code>, <code>cub::Sum()</code>
              </Translated>
            </td>
            <td>
              <Translated
                vi={
                  <>
                    Phép rút gọn hai giai đoạn viết tay của M4 (lane mask có kiểu, <code>warpSize</code>); trên Metal,{' '}
                    <code>simd_max</code> và <code>simd_sum</code> với bộ nhớ threadgroup
                  </>
                }
              >
                M4&rsquo;s hand-written two-stage reduction (typed lane mask, <code>warpSize</code>); on Metal,{' '}
                <code>simd_max</code> and <code>simd_sum</code> with threadgroup memory
              </Translated>
            </td>
            <td>
              <Translated vi="các functor bị gỡ bỏ trong CCCL 3.0 (infra A11); hipCUB khác và cần mask 64-bit (B1); C3">
                functors removed in CCCL 3.0 (infra A11); hipCUB differs and needs a 64-bit mask (B1); C3
              </Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated
                vi={
                  <>
                    Ghi <code>D</code>, mẫu số của hàng
                  </>
                }
              >
                Writes <code>D</code>, the row&rsquo;s denominator
              </Translated>
            </td>
            <td>
              <Translated
                vi={
                  <>
                    Ghi <code>L = m + log(D)</code>
                  </>
                }
              >
                Writes <code>L = m + log(D)</code>
              </Translated>
            </td>
            <td>
              <Translated vi="Chỉ riêng D, không có max của hàng, không thể dựng lại P; FlashAttention-2 lưu log-sum-exp (A7)">
                D alone, without the row max, cannot rebuild P; FlashAttention-2 stores the log-sum-exp (A7)
              </Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi="Vuông N × N, hàng r thấy các cột 0 đến r">Square N × N, row r sees columns 0 to r</Translated>
            </td>
            <td>
              <Translated vi="nq ≤ nk truy vấn và key; hàng r thấy các cột tới r + nk − nq, nên nq = 1 là một bước decode không có mặt nạ">
                nq ≤ nk queries and keys; row r sees columns up to r + nk − nq, so nq = 1 is a decode step with no mask
              </Translated>
            </td>
            <td>
              <Translated vi="căn theo góc dưới bên phải, như FlashAttention 2.1 và MLX (A8, D4)">
                lower-right alignment, as FlashAttention 2.1 and MLX (A8, D4)
              </Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi="S đến nơi đã được co giãn theo 1/√d; một chuỗi, một head">
                S arrives already scaled by 1/√d; one sequence, one head
              </Translated>
            </td>
            <td>
              <Translated
                vi={
                  <>
                    Kernel nhân với <code>scale</code>, như lab 10.3; trục y của lưới đi qua một batch các phần tử (chuỗi,
                    head)
                  </>
                }
              >
                The kernel multiplies by <code>scale</code>, as lab 10.3 does; grid y walks a batch of (sequence, head)
                items
              </Translated>
            </td>
            <td>
              <Translated
                vi={
                  <>
                    S đến thẳng từ <code>launchLinear</code>, vốn không có α; sách cho phép đặt ở chỗ nào cũng được (p.
                    486)
                  </>
                }
              >
                S comes straight from <code>launchLinear</code>, which has no α; the book allows either place (p. 486)
              </Translated>
            </td>
          </tr>
        </tbody>
      </table>
      <LabListing />
      <Ask q={questionsS203['ch20/20.3/q-griddim']} />
      <Ask q={questionsS203['ch20/20.3/q-decode-row']} />
      <Translated
        vi={
          <p>
            Thêm một khẳng định ở p. 487: một khối cho mỗi hàng nghĩa là N không thể vượt giới hạn của lưới theo x. Giới hạn
            đó là 2<sup>31</sup> − 1 khối, xa hơn mọi ngữ cảnh; chỗ nó gây khó là y và z, ở 65,535, đó là nơi một kernel
            theo batch đặt các head hoặc chuỗi (lab 20.1 đặt chúng trên y). Cách sửa của sách vẫn áp dụng được: một vòng
            lặp ngoài bước qua các hàng theo <code>gridDim.x</code>.
          </p>
        }
      >
        <p>
          One more claim on p. 487: one block per row means N cannot exceed the grid&rsquo;s limit in x. That limit is 2
          <sup>31</sup> − 1 blocks, far beyond any context; where it bites is y and z, at 65,535, which is where a batched
          kernel puts heads or sequences (lab 20.1 puts them on y). The book&rsquo;s fix still applies: an outer loop that
          strides rows by <code>gridDim.x</code>.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch05-facts.md A5; ch20-facts.md A2, B1, C3; infra-facts.md A11)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              Bài tập 2 yêu cầu bạn đọc tài liệu về phép rút gọn của CUB. Tóm lại: <code>WarpReduce</code> và{' '}
              <code>BlockReduce</code> để kết quả ở lane 0 hoặc luồng 0; <code>cub::Max</code> và <code>cub::Sum</code>{' '}
              không còn nữa; hãy cho mỗi warp <code>TempStorage</code> riêng, và đặt <code>__syncwarp()</code> hoặc{' '}
              <code>__syncthreads()</code> giữa hai lần dùng một bộ nhớ tạm.
            </p>
          }
        >
          <p>
            Exercise 2 asks you to read CUB&rsquo;s reduction documentation. In short: <code>WarpReduce</code> and{' '}
            <code>BlockReduce</code> leave the result in lane 0 or thread 0; <code>cub::Max</code> and{' '}
            <code>cub::Sum</code> are gone; give each warp its own <code>TempStorage</code>, and put{' '}
            <code>__syncwarp()</code> or <code>__syncthreads()</code> between two uses of one.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS203['ch20/20.3/q-cub']} />

      <h2>
        <Translated vi="Pipeline không gộp di chuyển những gì">What the unfused pipeline moves</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hãy đếm byte cho ba kernel. GEMM đầu tiên ghi toàn bộ S, vì một GEMM thư viện không biết đến mặt nạ. Softmax
            đọc nửa nhìn thấy của S ba lần (max, tổng, đầu ra) và ghi toàn bộ P. GEMM thứ hai đọc P. Mọi thứ khác là Q, K,
            V và O, thứ mà mọi cách cài đặt đều phải di chuyển. Đo so với phần đó, phần N × N tăng theo N / d.
          </p>
        }
      >
        <p>
          Count bytes for the three kernels. The first GEMM writes all of S, since a library GEMM knows no mask. The
          softmax reads the visible half of S three times (max, sum, output) and writes all of P. The second GEMM reads P.
          Everything else is Q, K, V and O, which any implementation has to move. Measured against that, the N × N part
          grows like N / d.
        </p>
      </Translated>
      <Source section="20.3" pages="486–488" />
      <Translated
        vi={
          <p>
            Ở N = 4096 và d = 128, S của một head lớn gấp 32 lần Q của nó. Hãy dùng trình khám phá để tìm hiểu điều đó có
            nghĩa gì với cả một lớp.
          </p>
        }
      >
        <p>
          At N = 4096 and d = 128 one head&rsquo;s S is 32 times the size of its Q. Use the explorer to find out what
          that means for a whole layer.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các con số là của chúng tôi: lưu lượng bắt buộc của mỗi kernel như đã viết, không có bộ nhớ đệm hỗ trợ. Sách
              nêu ý này bằng lời: các kernel riêng rẽ nghĩa là nhiều lượt đi qua bộ nhớ toàn cục (pp. 492–493).
            </p>
          }
        >
          <p>
            The counts are ours: the compulsory traffic of each kernel as written, with no cache help. The book makes the
            point in words: separate kernels mean several trips through global memory (pp. 492–493).
          </p>
        </Translated>
      </Simplification>
      <Traffic />
      <Ask q={questionsS203['ch20/20.3/q-sp-bytes']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            4 GiB đó là cho một lớp, ở một ngữ cảnh mà một cuộc trò chuyện đạt tới rất nhanh, và chúng tăng gấp bốn với
            mỗi lần N tăng gấp đôi. Mục <a href={href('ch20', '20.4')}>20.4</a> loại bỏ việc tính lại qua các bước decode;
            mục <a href={href('ch20', '20.5')}>20.5</a> loại bỏ hẳn S và P khỏi bộ nhớ.
          </p>
        }
      >
        <p>
          Those 4 GiB are for one layer, at a context a chat reaches quickly, and they grow fourfold with every doubling
          of N.
          Section <a href={href('ch20', '20.4')}>20.4</a> removes the recomputation across decode steps; section{' '}
          <a href={href('ch20', '20.5')}>20.5</a> removes S and P from memory altogether.
        </p>
      </Translated>
    </>
  );
}
