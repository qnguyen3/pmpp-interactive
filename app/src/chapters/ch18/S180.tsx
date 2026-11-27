import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { href } from '../../lib/router';
import { BOOK, EX1, backtrace, inNeighbours, levels, showLevel, storage, toCOO, toCSC, toCSR, treeEdges, UNVISITED } from '../../sim/bfs';
import { BfsGraph } from './BfsGraph';

// Before you start (book pp. 425-431, §18.1 and §18.2, plus the chapter's purpose from
// pp. 425, 445 and 449). What the book says comes from notes/ch18.md; vendor facts from
// notes/today/ch18-facts.md (D3, D5, D6). Prerequisite questions draw on the notes of
// chapters 4, 9, 11, 12 and 17. Models: src/sim/bfs.ts.

export const questionsS180: Record<string, Question> = {
  'ch18/start/q-csr-read': {
    kind: 'choice',
    id: 'ch18/start/q-csr-read',
    prompt: (
      <Translated
        vi={
          <>
            <code>srcPtrs = [0,2,4,7,9,11,12,13,15,15]</code> và <code>dst = [1,2,3,4,5,6,7,4,8,5,8,6,8,0,6]</code>. Những
            cạnh nào rời đỉnh 4, và có bao nhiêu cạnh rời đỉnh 8?
          </>
        }
      >
        <>
          <code>srcPtrs = [0,2,4,7,9,11,12,13,15,15]</code> and <code>dst = [1,2,3,4,5,6,7,4,8,5,8,6,8,0,6]</code>. Which
          edges leave vertex 4, and how many leave vertex 8?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="4 → 5 và 4 → 8; không cạnh nào rời 8.">4 → 5 and 4 → 8; none leave 8.</Translated>, correct: true },
      {
        label: <Translated vi="4 → 4 và 4 → 8; một cạnh rời 8.">4 → 4 and 4 → 8; one leaves 8.</Translated>,
        why: (
          <Translated vi="Cách đó đọc dst từ srcPtrs[3] = 7 trở đi, tức đoạn của đỉnh 3. Đoạn của đỉnh v bắt đầu tại srcPtrs[v].">
            That reads dst at srcPtrs[3] = 7 and onwards, the slice of vertex 3. Vertex v's slice starts at srcPtrs[v].
          </Translated>
        ),
      },
      {
        label: <Translated vi="4 → 5 và 4 → 8; hai cạnh rời 8, tới 0 và 6.">4 → 5 and 4 → 8; two leave 8, to 0 and 6.</Translated>,
        why: (
          <Translated vi="dst[13] và dst[14] thuộc về đỉnh 7: srcPtrs[7] = 13 và srcPtrs[8] = 15.">
            dst[13] and dst[14] belong to vertex 7: srcPtrs[7] = 13 and srcPtrs[8] = 15.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ 4 → 5: đoạn là từ srcPtrs[4] = 9 đến srcPtrs[5] = 11, và đầu cuối bị loại trừ, nên chỉ một phần tử.">
            Only 4 → 5: the slice is srcPtrs[4] = 9 to srcPtrs[5] = 11, and the end is excluded, so one entry.
          </Translated>
        ),
        why: (
          <Translated vi="Từ 9 đến 11, không gồm 11, là hai phần tử: dst[9] và dst[10].">
            From 9 up to, not including, 11 is two entries: dst[9] and dst[10].
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Các cạnh ra của một đỉnh là đoạn của <code>dst</code> từ <code>srcPtrs[v]</code> đến, không gồm,{' '}
            <code>srcPtrs[v+1]</code>. Với đỉnh 4 đó là <code>dst[9..10]</code> = 5, 8. Với đỉnh 8 là từ 15 đến 15: một đoạn
            rỗng, nên 8 không có cạnh ra. Phần tử cuối thêm vào của <code>srcPtrs</code> tồn tại để quy tắc này cũng đúng
            cho đỉnh cuối cùng. Đây là cách làm của Bài tập 1b.
          </p>
        }
      >
        <p>
          A vertex's out-edges are the slice of <code>dst</code> from <code>srcPtrs[v]</code> up to, not including,{' '}
          <code>srcPtrs[v+1]</code>. For 4 that is <code>dst[9..10]</code> = 5, 8. For 8 it is from 15 to 15: an empty
          slice, so 8 has no out-edges. The extra last entry of <code>srcPtrs</code> exists so this rule works for the last
          vertex too. This is Exercise 1b's method.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.1, p. 427, Fig. 18.3',
  },
  'ch18/start/q-csc-in': {
    kind: 'choice',
    id: 'ch18/start/q-csc-in',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel pull cần mọi đỉnh trỏ tới đỉnh 6 trong đồ thị của sách. Những mảng nào cho danh sách đó dưới dạng một
            đoạn liên tục, và trong đó có gì?
          </>
        }
      >
        <>
          A pull kernel needs every vertex that points to vertex 6 in the book's graph. Which arrays give that list as
          one contiguous slice, and what is in it?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="CSC: dstPtrs[6] = 8 đến dstPtrs[7] = 11, nên src[8..10] = 2, 5, 7.">
            CSC: dstPtrs[6] = 8 to dstPtrs[7] = 11, so src[8..10] = 2, 5, 7.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="CSR: srcPtrs[6] = 12 đến srcPtrs[7] = 13, nên đoạn là dst[12] = 8, đỉnh duy nhất nối với 6.">
            CSR: srcPtrs[6] = 12 to srcPtrs[7] = 13, so the slice is dst[12] = 8, the one vertex joined to 6.
          </Translated>
        ),
        why: <Translated vi="Đó là nơi 6 trỏ tới (6 → 8), không phải ai trỏ tới 6.">That is where 6 points (6 → 8), not who points at 6.</Translated>,
      },
      {
        label: (
          <Translated vi="COO: các cạnh vào 6 nằm cạnh nhau trong đó, nên một đoạn của mảng src cho ra 2, 5 và 7.">
            COO: the edges into 6 sit side by side in it, so one run of its src array gives 2, 5 and 7.
          </Translated>
        ),
        why: (
          <Translated vi="COO của sách được sắp theo đỉnh nguồn, nên các phần tử có dst 6 nằm rải rác (phần tử 5, 11 và 14): bạn phải quét cả 15 để tìm ra chúng.">
            The book's COO is sorted by source, so the entries with dst 6 are scattered (entries 5, 11 and 14): you scan all 15 to find them.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            CSC nhóm các cạnh theo đỉnh đích, nên một cột là một đoạn. Với CSR bạn phải tìm số 6 trong mọi hàng; với COO bạn
            phải quét cả 15 cạnh. Định dạng bạn giữ quyết định câu hỏi nào rẻ, và do đó bạn viết được kernel nào.
          </p>
        }
      >
        <p>
          CSC groups edges by destination, so a column is one slice. With CSR you would search every row for a 6; with
          COO you would scan all 15 edges. Which format you keep decides which question is cheap, and so which kernel
          you can write.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.1, pp. 428–429, Fig. 18.3',
  },
  'ch18/start/q-path': {
    kind: 'choice',
    id: 'ch18/start/q-path',
    prompt: (
      <Translated vi={<>Với gốc 2, lần ngược từ đỉnh 1. Chuỗi nào là một đường đi ngắn nhất?</>}>
        <>With root 2, back-trace from vertex 1. Which chain is a shortest path?</>
      </Translated>
    ),
    options: [
      { label: '1 ← 0 ← 7 ← 2', correct: true },
      {
        label: '1 ← 0 ← 7 ← 6 ← 2',
        why: (
          <Translated vi="6 và 7 đều ở mức 1, và cạnh giữa chúng đi 7 → 6. Mỗi bước phải đi tới một đỉnh kề vào thấp hơn đúng một mức.">
            6 and 7 are both at level 1, and the edge between them runs 7 → 6. A step must go to an in-neighbour exactly one level lower.
          </Translated>
        ),
      },
      {
        label: '1 ← 4 ← 5 ← 2',
        why: (
          <Translated vi="1 → 4 là một cạnh rời 1, nên 4 là con của 1, không phải cha. Lần ngược đi theo các cạnh vào.">
            1 → 4 is an edge leaving 1, so 4 is a child of 1, not a parent. Back-tracing follows incoming edges.
          </Translated>
        ),
      },
      {
        label: '1 ← 0 ← 2',
        why: (
          <Translated vi="Không có cạnh 2 → 0; với gốc này 0 ở mức 2.">There is no edge 2 → 0; 0 is at level 2 with this root.</Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi bước đi tới một đỉnh kề vào thấp hơn một mức: đỉnh kề vào duy nhất của 1 là 0 (mức 2), của 0 là 7 (mức 1),
            của 7 là 2. Vậy đường đi là 2 → 7 → 0 → 1. Phép lần ngược đọc các cạnh vào, nên nó cần CSC, hoặc một đỉnh cha
            được ghi lại cho mỗi đỉnh trong lúc tìm kiếm.
          </p>
        }
      >
        <p>
          Each step goes to an in-neighbour one level lower: 1's only in-neighbour is 0 (level 2), 0's is 7 (level 1),
          7's is 2. So the path is 2 → 7 → 0 → 1. The walk reads incoming edges, so it needs CSC, or a predecessor
          recorded for each vertex during the search.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.2, p. 430, Fig. 18.4(b)', // i18n-ignore: citation
  },
  'ch18/start/q-divergence': {
    kind: 'numeric',
    id: 'ch18/start/q-divergence',
    prompt: (
      <Translated
        vi={
          <>
            Trong một warp, lane 0 chạy một vòng lặp 100 lần và mọi lane khác chạy nó một lần. Warp phải đi qua thân vòng lặp
            bao nhiêu lần?
          </>
        }
      >
        <>
          In one warp, lane 0 runs a loop 100 times and every other lane runs it once. How many times does the warp step
          through the loop body?
        </>
      </Translated>
    ),
    answer: 100,
    unit: 'passes',
    explain: (
      <Translated
        vi={
          <p>
            Một warp tiếp tục chạy cho đến khi lane chạy lâu nhất của nó xong. Sau lượt đầu, mọi lane trừ một đều rảnh, và các
            lane rảnh đó vẫn chiếm phần cứng. Ở trang sau, một đỉnh có nhiều cạnh gây ra đúng điều này cho warp của nó, và
            kernel lấy cạnh làm trung tâm là một lối thoát.
          </p>
        }
      >
        <p>
          A warp keeps going until its longest-running lane is done. After the first pass every lane but one is idle,
          and those idle lanes still occupy the hardware. On the next page a vertex with many edges does exactly this to
          its warp, and the edge-centric kernel is one way out.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.5, p. 81 (loops with different trip counts, Fig. 4.10)',
    vi: { unit: 'lượt', source: 'ch. 4 §4.5, p. 81 (vòng lặp có số lần lặp khác nhau, Hình 4.10)' },
  },
  'ch18/start/q-resident': {
    kind: 'choice',
    id: 'ch18/start/q-resident',
    prompt: (
      <Translated
        vi={
          <>
            Một lưới (grid) 64 khối chạy trên một thiết bị chứa được 8 khối cùng lúc. Mỗi khối bật một cờ, rồi quay vòng chờ
            đến khi cả 64 cờ đều được bật. Điều gì xảy ra?
          </>
        }
      >
        <>
          A grid of 64 blocks runs on a device that holds 8 blocks at a time. Every block sets a flag, then spins until
          all 64 flags are set. What happens?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nó treo: 8 khối thường trú chờ những khối không thể bắt đầu cho đến khi một khối thường trú kết thúc.">
            It hangs: the 8 resident blocks wait for blocks that cannot start until a resident block finishes.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nó có thể xong, chậm hơn: phần cứng có thể đẩy các khối đang quay vòng ra để cho các khối khác vào.">
            It may finish, more slowly: the hardware can swap the spinning blocks out to let the others in.
          </Translated>
        ),
        why: (
          <Translated vi="Các khối được nhận vào khi các khối trước kết thúc, không bị đẩy ra khi đang chạy. Ở đây không có gì kết thúc.">
            Blocks are admitted as earlier ones finish, not swapped out while they run. Nothing here finishes.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó xong với tốc độ tối đa: 8 khối thường trú quay vòng trong khi 56 khối còn lại chạy song song với chúng.">
            It finishes at full speed: the 8 resident blocks spin while the other 56 run alongside them.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi lúc chỉ vừa 8 khối, và không khối nào rời vòng quay của nó, nên 56 khối kia không bao giờ bắt đầu và không bao giờ bật cờ.">
            Only 8 fit at a time, and none of them leaves its spin loop, so the other 56 never start and never set their flags.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Runtime giữ một danh sách các khối còn phải chạy và giao một khối khi một khối đang chạy kết thúc. Một khối chờ
            một khối chưa bắt đầu có thể chờ mãi mãi. Phép scan một lượt của chương 11 tránh điều này bằng cách lấy chỉ số
            khối theo thứ tự các khối thực sự bắt đầu. Mục 18.7 cần một rào chắn (barrier) trên toàn lưới, và đối mặt trực
            tiếp với đúng vấn đề này.
          </p>
        }
      >
        <p>
          The runtime keeps a list of blocks still to run and assigns one when a running block finishes. A block that
          waits for a block that has not started can wait for ever. Chapter 11's single-pass scan avoided this by taking
          block indices in the order blocks actually start. Section 18.7 needs a barrier across the whole grid, and
          meets the same problem head on.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.2, p. 69; ch. 11 §11.9, p. 280', // i18n-ignore: citation
  },
  'ch18/start/q-workeff': {
    kind: 'choice',
    id: 'ch18/start/q-workeff',
    prompt: (
      <Translated
        vi={
          <>
            Một phép scan song song xong trong log₂N bước nhưng thực hiện khoảng N·log₂N phép cộng, trong khi vòng lặp tuần tự
            chỉ cần N − 1. Theo cách nói của chương 11, nó có vấn đề gì, và khi nào điều đó làm bạn tốn kém?
          </>
        }
      >
        <>
          A parallel scan finishes in log₂N steps but performs about N·log₂N additions, where the sequential loop does N
          − 1. In chapter 11's terms, what is wrong with it, and when does that cost you?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nó không hiệu quả về công việc: các phép cộng thừa tốn thời gian khi khối lượng công việc vượt quá số đơn vị thực thi, và luôn tốn năng lượng.">
            It is not work-efficient: the extra additions cost time once the work exceeds the execution units, and energy always.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không có gì: với đủ luồng, ít bước hơn nghĩa là kernel nhanh hơn, bất kể số phép cộng.">
            Nothing: with enough threads, fewer steps means a faster kernel, whatever the additions.
          </Translated>
        ),
        why: (
          <Translated vi="Chỉ khi có đủ đơn vị thực thi. Khi có ít, các phép cộng thừa phải trả bằng thời gian.">
            Only with enough execution units. With few, the extra additions are paid for in time.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó không hiệu quả về công việc, điều này tốn thời gian trên CPU có ít lõi nhưng không tốn trên GPU, nơi các phép cộng thừa chạy song song.">
            It is not work-efficient, which costs time on a CPU with a few cores but not on a GPU, where the extra additions run in parallel.
          </Translated>
        ),
        why: (
          <Translated vi="GPU có nhiều đơn vị nhưng không phải vô hạn, và mỗi phép toán thừa đều tốn năng lượng.">
            A GPU has many units but not unlimited ones, and every extra operation costs energy.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hiệu quả công việc so sánh tổng khối lượng công việc với thuật toán tuần tự tốt nhất. Các kernel BFS của trang sau
            đều xong một mức trong một lần khởi chạy, và phần lớn chúng không hiệu quả về công việc: chúng kiểm tra mọi đỉnh,
            hoặc mọi cạnh, ở mọi mức. Mục 18.5 đếm chi phí đó và khắc phục nó.
          </p>
        }
      >
        <p>
          Work efficiency compares the total work with the best sequential algorithm's. The BFS kernels of the next
          page all finish a level in one launch, and most of them are not work-efficient: they check every vertex, or
          every edge, at every level. Section 18.5 counts the cost and fixes it.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.5, pp. 265–267',
  },
  'ch18/start/q-slot': {
    kind: 'choice',
    id: 'ch18/start/q-slot',
    prompt: (
      <Translated
        vi={
          <>
            Mỗi luồng vượt qua đều chạy <code>i = counter.fetch_add(1, relaxed); out[i] = key;</code>. Hai lần chạy trên cùng
            đầu vào cho đầu ra theo thứ tự khác nhau. Vì sao không ô nào bị ghi hai lần hay bị bỏ sót?
          </>
        }
      >
        <>
          Every passing thread runs <code>i = counter.fetch_add(1, relaxed); out[i] = key;</code>. Two runs on the same
          input give outputs in different orders. Why is no slot ever written twice or skipped?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mỗi fetch_add trả về một giá trị cũ khác nhau, 0, 1, 2 và cứ thế, không có chỗ trống; chỉ việc ai nhận giá trị nào là thay đổi.">
            Every fetch_add returns a distinct old value, 0, 1, 2 and so on with no gaps; only who gets which one varies.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Thứ tự relaxed buộc các luồng lần lượt theo thứ tự luồng, nên mỗi luồng nhận ô kế tiếp.">
            Relaxed ordering makes the threads take their turns in thread order, so each gets the next slot.
          </Translated>
        ),
        why: (
          <Translated vi="Thứ tự relaxed không hứa hẹn gì về thứ tự. Các thứ tự đầu ra khác nhau cho thấy điều đó.">
            Relaxed ordering promises nothing about order. The different output orders show that.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các ô có thể bị ghi hai lần khi hai luồng va chạm; hai lần chạy này tình cờ không va chạm.">
            Slots can be written twice when two threads collide; these two runs happened not to collide.
          </Translated>
        ),
        why: (
          <Translated vi="Hai fetch_add trên cùng một bộ đếm không bao giờ trả về cùng một giá trị: đó chính là điều làm chúng nguyên tử (atomic).">
            Two fetch_adds on one counter never return the same value: that is what makes them atomic.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tập các ô và số đếm cuối cùng là cố định; luồng nào nhận ô nào thì không. Đó là phép lọc không ổn định của chương
            12, và đó chính xác là cách một frontier BFS được dựng ở trang sau.
          </p>
        }
      >
        <p>
          The set of slots and the final count are fixed; which thread gets which slot is not. That is chapter 12's
          unstable filter, and it is exactly how a BFS frontier is built on the next page.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.2, pp. 290–291',
  },
  'ch18/start/q-scope': {
    kind: 'choice',
    id: 'ch18/start/q-scope',
    prompt: (
      <Translated
        vi={
          <>
            Bộ đếm riêng của một khối nằm trong bộ nhớ chia sẻ (shared memory) và chỉ khối đó chạm vào nó. Sau đó khối cộng số
            đếm của mình vào một bộ đếm trong bộ nhớ toàn cục (global memory) mà mọi khối đều cộng vào. Phạm vi (scope) nào là
            đúng?
          </>
        }
      >
        <>
          A block's private counter lives in shared memory and only that block touches it. The block then adds its count
          to a counter in global memory that every block adds to. Which scopes are right?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Phạm vi khối cho bộ đếm chia sẻ, phạm vi thiết bị cho bộ đếm toàn cục.">
            Block scope for the shared counter, device scope for the global one.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Phạm vi khối cho cả hai: mỗi khối chỉ cập nhật bộ đếm toàn cục một lần.">
            Block scope for both: each block makes only one update to the global counter.
          </Translated>
        ),
        why: (
          <Translated vi="Các khối khác cập nhật cùng bộ đếm toàn cục đó. Một atomic phạm vi khối trên một đối tượng mà các khối khác chạm vào là một race condition (tranh chấp dữ liệu).">
            Other blocks update the same global counter. A block-scope atomic on an object other blocks touch is a data race.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Phạm vi thiết bị cho cả hai: một atomic trên bộ nhớ chia sẻ cần phạm vi thiết bị để được nhìn thấy sau rào chắn.">
            Device scope for both: a shared-memory atomic needs device scope to be seen after the barrier.
          </Translated>
        ),
        why: (
          <Translated vi="Chỉ khối đó nhìn thấy bộ nhớ chia sẻ của nó, và rào chắn sắp thứ tự cho nó. Phạm vi thiết bị ở đó vẫn đúng nhưng đòi hỏi nhiều hơn mức cần; phạm vi khối mới là vừa khớp.">
            Only the block can see its shared memory, and the barrier orders it. Device scope is correct there but asks for more than it needs; block scope is the right fit.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Phạm vi chỉ ra tập các luồng đồng bộ hóa thông qua phép atomic. Mục 18.6 dùng đúng cặp này: một bộ đếm phạm vi
            khối cho hàng đợi riêng và một <code>fetch_add</code> phạm vi thiết bị cho mỗi khối trên hàng đợi chung.
          </p>
        }
      >
        <p>
          The scope names the set of threads that synchronise through the atomic. Section 18.6 uses exactly this pair: a
          block-scope counter for the private queue and one device-scope <code>fetch_add</code> per block on the public
          one.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.4, pp. 212–213; notes/today/ch09-facts.md A2', // i18n-ignore: citation
  },
  'ch18/start/q-rowthread': {
    kind: 'choice',
    id: 'ch18/start/q-rowthread',
    prompt: (
      <Translated
        vi={
          <>
            Nhân ma trận thưa với vectơ: so sánh mỗi luồng một hàng CSR với mỗi luồng một phần tử khác không của COO, trên một
            ma trận có một hàng chứa 10,000 phần tử khác không và mọi hàng khác khoảng 3. Phiên bản nào bị mất cân bằng tải,
            và phiên bản kia phải trả giá gì thay vào đó?
          </>
        }
      >
        <>
          Sparse matrix–vector multiplication: compare one thread per CSR row with one thread per COO non-zero, on a
          matrix where one row holds 10,000 non-zeros and every other row about 3. Which version suffers load imbalance,
          and what does the other pay instead?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mỗi luồng một hàng bị mất cân bằng; mỗi luồng một phần tử khác không thì cân bằng công việc nhưng cần atomic trên y.">
            One thread per row is imbalanced; one thread per non-zero balances the work but needs atomics on y.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Mỗi luồng một phần tử khác không bị mất cân bằng, vì các phần tử khác không của hàng dài dồn lại; mỗi luồng một hàng phải trả bằng atomic trên y.">
            One thread per non-zero is imbalanced, since the long row's non-zeros pile up; one thread per row pays with atomics on y.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi phần tử khác không là cùng một lượng công việc. Hàng có 10,000 phần tử khác không mới là việc dài.">
            Each non-zero is the same amount of work. The row with 10,000 non-zeros is the long job.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không cái nào: phần cứng tự cân bằng công việc giữa các luồng.">
            Neither: the hardware balances the work across threads.
          </Translated>
        ),
        why: (
          <Translated vi="Một luồng làm phần việc mà chỉ số của nó giao cho; không có gì phân phối lại.">
            A thread does the work its index gives it; nothing redistributes it.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Luồng sở hữu hàng dài chạy 10,000 lần lặp trong khi các luồng cùng warp chờ. Mỗi luồng một phần tử khác không cho
            mỗi luồng một phép nhân, nhưng giờ nhiều luồng cùng cộng vào một <code>y[row]</code>. Trang sau thực hiện đúng sự
            đánh đổi này giữa BFS lấy đỉnh làm trung tâm và BFS lấy cạnh làm trung tâm.
          </p>
        }
      >
        <p>
          The thread that owns the long row runs 10,000 iterations while its warp mates wait. One thread per non-zero
          gives every thread one multiply, but several threads now add into the same <code>y[row]</code>. The next page
          makes the same trade between vertex-centric and edge-centric BFS.
        </p>
      </Translated>
    ),
    source: 'ch. 17 §17.2 and §17.3, pp. 404–409; ch. 6 on re-mapping threads to work (cited by ch. 18, p. 438)',
    vi: { source: 'ch. 17 §17.2 và §17.3, pp. 404–409; ch. 6 về việc ánh xạ lại luồng với công việc (được ch. 18 trích, p. 438)' },
  },
};

// ---------------------------------------------------------------- the format lens
type Pick = { kind: 'vertex'; v: number } | { kind: 'edge'; e: number } | null;

function ArrayRow({ id, name, values, hot, on }: { id: string; name: string; values: number[]; hot: Set<number>; on: Set<number> }) {
  return (
    <div data-array={id} style={{ marginBottom: '0.55rem' }}>
      <div className="mx-label t-global">{name}</div>
      <div className="mx global" style={{ gridTemplateColumns: `repeat(${values.length}, auto)` }}>
        {values.map((x, i) => (
          <span key={i} data-i={i} className={hot.has(i) ? 'hot' : on.has(i) ? 'on' : ''} style={{ minWidth: '1.9rem', height: '1.8rem' }}>
            {x}
            <sub>{i}</sub>
          </span>
        ))}
      </div>
    </div>
  );
}

const range = (a: number, b: number) => new Set(Array.from({ length: Math.max(0, b - a) }, (_, k) => a + k));
const slice = (a: number, b: number) => (b > a ? (b - 1 > a ? `${a}..${b - 1}` : `${a}`) : '');

function FormatLens() {
  const [which, setWhich] = useState<'book' | 'ex1'>('book');
  const g = which === 'book' ? BOOK : EX1;
  const csr = useMemo(() => toCSR(g), [g]);
  const csc = useMemo(() => toCSC(g), [g]);
  const coo = useMemo(() => toCOO(g), [g]);
  const st = storage(g);
  const [pick, setPick] = useState<Pick>({ kind: 'vertex', v: 3 });
  const none = new Set<number>();
  let csrHot = none, csrOn = none, cscHot = none, cscOn = none, cooOn = none, csrDstHot = none, cscSrcHot = none;
  let text: ReactNode = tr('Click a vertex or an edge of the graph.', 'Bấm vào một đỉnh hoặc một cạnh của đồ thị.');
  if (pick?.kind === 'vertex') {
    const v = pick.v;
    const [a, b] = [csr.srcPtrs[v], csr.srcPtrs[v + 1]];
    const [c, d] = [csc.dstPtrs[v], csc.dstPtrs[v + 1]];
    csrHot = new Set([v, v + 1]);
    csrOn = range(a, b);
    cscHot = new Set([v, v + 1]);
    cscOn = range(c, d);
    const outs = csr.dst.slice(a, b);
    const ins = csc.src.slice(c, d);
    text = (
      <Translated
        vi={
          <>
            <b>Đỉnh {v}.</b> CSR: <code>srcPtrs[{v}] = {a}</code> và <code>srcPtrs[{v + 1}] = {b}</code>,{' '}
            {b > a ? (
              <>
                nên các cạnh ra của nó là <code>dst[{slice(a, b)}]</code> = {outs.join(', ')}: <span data-out>{outs.map((u) => `${v} → ${u}`).join(', ')}</span>.
              </>
            ) : (
              <>một đoạn rỗng: không cạnh nào rời {v}.</>
            )}{' '}
            CSC: <code>dstPtrs[{v}] = {c}</code> và <code>dstPtrs[{v + 1}] = {d}</code>,{' '}
            {d > c ? (
              <>
                nên các cạnh vào của nó là <code>src[{slice(c, d)}]</code> = {ins.join(', ')}: <span data-in>{ins.map((u) => `${u} → ${v}`).join(', ')}</span>.
              </>
            ) : (
              <>một đoạn rỗng: không cạnh nào đi tới {v}.</>
            )}{' '}
            COO không có con trỏ theo từng đỉnh: để tìm các cạnh này ở đó, bạn phải quét cả {g.edges.length} phần tử.
          </>
        }
      >
        <>
          <b>Vertex {v}.</b> CSR: <code>srcPtrs[{v}] = {a}</code> and <code>srcPtrs[{v + 1}] = {b}</code>,{' '}
          {b > a ? (
            <>
              so its out-edges are <code>dst[{slice(a, b)}]</code> = {outs.join(', ')}: <span data-out>{outs.map((u) => `${v} → ${u}`).join(', ')}</span>.
            </>
          ) : (
            <>an empty slice: no edges leave {v}.</>
          )}{' '}
          CSC: <code>dstPtrs[{v}] = {c}</code> and <code>dstPtrs[{v + 1}] = {d}</code>,{' '}
          {d > c ? (
            <>
              so its in-edges are <code>src[{slice(c, d)}]</code> = {ins.join(', ')}: <span data-in>{ins.map((u) => `${u} → ${v}`).join(', ')}</span>.
            </>
          ) : (
            <>an empty slice: no edges arrive at {v}.</>
          )}{' '}
          COO has no per-vertex pointers: to find these edges there you would scan all {g.edges.length} entries.
        </>
      </Translated>
    );
  } else if (pick?.kind === 'edge') {
    const e = pick.e;
    const [s, t] = g.edges[e];
    const k = csc.dstPtrs[t] + inNeighbours(csc, t).indexOf(s);
    cooOn = new Set([e]);
    csrDstHot = new Set([e]);
    cscSrcHot = new Set([k]);
    text = (
      <Translated
        vi={
          <>
            <b>Cạnh {s} → {t}</b> là phần tử COO thứ {e}: <code>src[{e}] = {s}</code>, <code>dst[{e}] = {t}</code>. Hai lần đọc,
            không tìm kiếm. Trong CSR, cùng cạnh đó là <code>dst[{e}]</code>, mà bạn chỉ tới được qua hàng của đỉnh nguồn; trong
            CSC nó là <code>src[{k}]</code>, tới được qua cột của đỉnh đích.
          </>
        }
      >
        <>
          <b>Edge {s} → {t}</b> is COO entry {e}: <code>src[{e}] = {s}</code>, <code>dst[{e}] = {t}</code>. Two reads, no
          search. In CSR the same edge is <code>dst[{e}]</code>, which you reach only through its source's row; in CSC it is{' '}
          <code>src[{k}]</code>, reached through its destination's column.
        </>
      </Translated>
    );
  }
  const pickedVertex = pick?.kind === 'vertex' ? pick.v : -1;
  return (
    <Bench
      title={tr('Format lens', 'Lăng kính định dạng')}
      task={tr('Click a vertex: which slices light up? Then click an edge.', 'Bấm vào một đỉnh: những đoạn nào sáng lên? Rồi bấm vào một cạnh.')}
      foot={
        <Translated
          vi={
            <>
              Các ô màu hổ phách là những phần tử mảng mà một luồng sẽ đọc: cặp đậm hơn là các con trỏ, dải nhạt hơn là đoạn
              chúng giới hạn. Số nhỏ là chỉ số. Mũi tên đậm nét liền là cạnh ra của đỉnh được chọn, mũi tên đậm nét đứt là cạnh
              vào của nó.
            </>
          }
        >
          <>
            Amber cells are the array entries a thread would read: the darker pair are the pointers, the lighter run is
            the slice they bound. Small numbers are indices. Solid bold arrows are out-edges of the chosen vertex,
            dashed bold arrows its in-edges.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Graph', 'Đồ thị')}>
          <button aria-pressed={which === 'book'} onClick={() => { setWhich('book'); setPick({ kind: 'vertex', v: 3 }); }}><Translated vi="đồ thị của sách (Hình 18.1)">the book's graph (Fig. 18.1)</Translated></button>
          <button aria-pressed={which === 'ex1'} onClick={() => { setWhich('ex1'); setPick({ kind: 'vertex', v: 0 }); }}><Translated vi="đồ thị Bài tập 1 (cách chúng tôi đọc p. 450)">Exercise 1 graph (our reading of p. 450)</Translated></button>
        </span>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 7fr)' }}>
        <div>
          <BfsGraph
            g={g}
            label={tr(
              `The ${which === 'book' ? "book's" : 'Exercise 1'} graph; click a vertex or an edge`,
              `Đồ thị ${which === 'book' ? 'của sách' : 'Bài tập 1'}; bấm vào một đỉnh hoặc một cạnh`,
            )}
            look={(v) => (v === pickedVertex ? 'root' : 'plain')}
            edgeLook={(i, s, d) =>
              pick?.kind === 'edge' ? (i === pick.e ? 'pick' : 'dim') : s === pickedVertex ? 'out' : d === pickedVertex ? 'in' : 'dim'
            }
            onVertex={(v) => setPick({ kind: 'vertex', v })}
            onEdge={(e) => setPick({ kind: 'edge', e })}
          />
          <p className="muted" data-storage style={{ margin: '0.6rem 0 0' }}>
            <Translated
              vi={
                <>
                  Dung lượng lưu trữ không tính mảng dữ liệu: ma trận dày đặc <b>{st.dense}</b>, CSR <b>{st.csr}</b> ({g.n + 1} + {g.edges.length}),
                  CSC <b>{st.csc}</b>, COO <b>{st.coo}</b> ({g.edges.length} + {g.edges.length}).
                </>
              }
            >
              Storage without the data array: dense matrix <b>{st.dense}</b>, CSR <b>{st.csr}</b> ({g.n + 1} + {g.edges.length}),
              CSC <b>{st.csc}</b>, COO <b>{st.coo}</b> ({g.edges.length} + {g.edges.length}).
            </Translated>
          </p>
        </div>
        <div style={{ minWidth: 0 }}>
          <div className="narrate" role="status" style={{ marginBottom: '0.8rem' }}>{text}</div>
          <div style={{ overflowX: 'auto' }}>
            <div className="mx-label"><Translated vi="CSR: các cạnh ra">CSR: outgoing edges</Translated></div>
            <ArrayRow id="csr-srcPtrs" name="srcPtrs" values={csr.srcPtrs} hot={csrHot} on={none} />
            <ArrayRow id="csr-dst" name="dst" values={csr.dst} hot={csrDstHot} on={csrOn} />{/* i18n-ignore: array names */}
            <div className="mx-label" style={{ marginTop: '0.4rem' }}><Translated vi="CSC: các cạnh vào">CSC: incoming edges</Translated></div>
            <ArrayRow id="csc-dstPtrs" name="dstPtrs" values={csc.dstPtrs} hot={cscHot} on={none} />
            <ArrayRow id="csc-src" name="src" values={csc.src} hot={cscSrcHot} on={cscOn} />{/* i18n-ignore: array names */}
            <div className="mx-label" style={{ marginTop: '0.4rem' }}><Translated vi="COO: mỗi cạnh một phần tử">COO: one entry per edge</Translated></div>
            <ArrayRow id="coo-src" name="src" values={coo.src} hot={cooOn} on={none} />{/* i18n-ignore: array names */}
            <ArrayRow id="coo-dst" name="dst" values={coo.dst} hot={cooOn} on={none} />{/* i18n-ignore: array names */}
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- move the root
/** The explorer shows the answer, so it opens only once a prediction is committed. */
function RootGate() {
  const { progress } = useProgress();
  if (!progress.answers['ch18/start/p-root2']) {
    return (
      <p className="muted ui">
        <Translated vi="Trình khám phá sẽ mở khi bạn đã chốt một dự đoán.">The explorer opens when you have committed a prediction.</Translated>
      </p>
    );
  }
  return <RootExplorer />;
}

function RootExplorer() {
  const [root, setRoot] = useState(2);
  const [target, setTarget] = useState(4);
  const g = BOOK;
  const lv = useMemo(() => levels(g, root), [g, root]);
  const lv0 = useMemo(() => levels(g, 0), [g]);
  const csc = useMemo(() => toCSC(g), [g]);
  const tree = useMemo(() => new Set(treeEdges(g, root).map(([s, d]) => s + ',' + d)), [g, root]);
  const path = useMemo(() => backtrace(g, root, target), [g, root, target]);
  const onPath = new Set((path ?? []).slice(1).map((v, i) => path![i] + ',' + v));
  const changed = lv.filter((L, v) => L !== lv0[v]).length;
  // The walk back, step by step, reading each vertex's in-edges (CSC).
  const steps = path
    ? path.slice(1).reverse().map((v) => {
        const ins = inNeighbours(csc, v);
        const next = path[path.indexOf(v) - 1];
        return { v, ins, next };
      })
    : [];
  return (
    <Bench
      title={tr('Move the root', 'Di chuyển gốc')}
      task={tr('Click a vertex to make it the root. Pick a vertex to walk back from.', 'Bấm vào một đỉnh để biến nó thành gốc. Chọn một đỉnh để lần ngược từ đó.')}
      foot={tr(
        'Bold arrows are the BFS tree; amber arrows the path recovered backwards. Each vertex shows its level.',
        'Mũi tên đậm là cây BFS; mũi tên màu hổ phách là đường đi được khôi phục theo chiều ngược. Mỗi đỉnh hiển thị mức của nó.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Lần ngược từ">Walk back from</Translated>
          <select value={target} onChange={(e) => setTarget(+e.target.value)} aria-label={tr('Walk back from', 'Lần ngược từ')}>
            {Array.from({ length: g.n }, (_, v) => (
              <option key={v} value={v}>{tr(`vertex ${v}`, `đỉnh ${v}`)}</option>
            ))}
          </select>
        </label>
        <span className="muted" data-root><Translated vi={<>gốc {root}</>}>root {root}</Translated></span>
        <button onClick={() => setRoot(0)} disabled={root === 0}><Translated vi="Gốc 0, như trong Hình 18.4(a)">Root 0, as in Fig. 18.4(a)</Translated></button>
        <button onClick={() => setRoot(2)} disabled={root === 2}><Translated vi="Gốc 2, như trong Hình 18.4(b)">Root 2, as in Fig. 18.4(b)</Translated></button>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 6fr)' }}>
        <div>
          <BfsGraph
            g={g}
            label={tr(`The book's graph with root ${root}`, `Đồ thị của sách với gốc ${root}`)}
            look={(v) => (v === root ? 'root' : lv[v] === UNVISITED ? 'unvisited' : 'visited')}
            tag={(v) => showLevel(lv[v])}
            ring={(v) => (v === target ? 'pick' : null)}
            edgeLook={(_, s, d) => (onPath.has(s + ',' + d) ? 'path' : tree.has(s + ',' + d) ? 'tree' : 'dim')}
            onVertex={setRoot}
          />
        </div>
        <div style={{ minWidth: 0 }}>
          <table className="threads" aria-label={tr('Levels by root', 'Các mức theo gốc')}>
            <thead>
              <tr>
                <th><Translated vi="đỉnh">vertex</Translated></th>
                {Array.from({ length: g.n }, (_, v) => <th key={v}>{v}</th>)}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><Translated vi="gốc 0">root 0</Translated></td>
                {lv0.map((L, v) => <td key={v}>{showLevel(L)}</td>)}
              </tr>
              <tr>
                <td><Translated vi={<>gốc {root}</>}>root {root}</Translated></td>
                {lv.map((L, v) => (
                  <td key={v} data-level={v} style={L !== lv0[v] ? { fontWeight: 650 } : undefined}>{showLevel(L)}</td>
                ))}
              </tr>
            </tbody>
          </table>
          <p data-changed={changed} style={{ margin: '0.6rem 0' }}>
            {root === 0 ? tr('This is root 0.', 'Đây là gốc 0.') : <Translated vi={<><b>{changed} / {g.n}</b> đỉnh có mức khác so với khi gốc là 0.</>}><><b>{changed} of {g.n}</b> vertices have a different level than with root 0.</></Translated>}
          </p>
          <div className="narrate" data-path={path ? path.join(' → ') : 'none'}>
            {path ? (
              path.length === 1 ? (
                <Translated vi={<>Đỉnh {target} là gốc: đường đi chỉ là {target}.</>}><>Vertex {target} is the root: the path is just {target}.</></Translated>
              ) : (
                <>
                  {steps.map((s) => (
                    <div key={s.v}>
                      <Translated vi={<>{s.v} (mức {lv[s.v]}): cạnh vào từ {s.ins.map((u) => `${u} (mức ${showLevel(lv[u])})`).join(', ')}; bước tới {s.next}.</>}>
                        {s.v} (level {lv[s.v]}): in-edges from {s.ins.map((u) => `${u} (level ${showLevel(lv[u])})`).join(', ')}; step to {s.next}.
                      </Translated>
                    </div>
                  ))}
                  <div style={{ marginTop: '0.3rem' }}>
                    <Translated vi={<><b>Đường đi: {path.join(' → ')}</b>, {path.length - 1} cạnh.</>}>
                      <b>Path: {path.join(' → ')}</b>, {path.length - 1} edges.
                    </Translated>
                  </div>
                </>
              )
            ) : (
              <Translated vi={<>Không thể tới đỉnh {target} từ {root}: nó không có mức và không có đường đi.</>}>
                <>Vertex {target} cannot be reached from {root}: it has no level and no path.</>
              </Translated>
            )}
          </div>
        </div>
      </div>
    </Bench>
  );
}

export function S180() {
  return (
    <>
      <Translated
        vi={
          <p>
            Sách dùng tìm kiếm theo chiều rộng, BFS, làm ví dụ cho loại công việc mà khối lượng và vị trí thay đổi trong khi
            chương trình chạy: những đỉnh nào cần xử lý ở bước sau phụ thuộc vào những gì bước hiện tại tìm ra. Bản thân BFS
            không thuộc stack suy luận LLM, và khóa học này coi nó là một phương tiện. Nó mang theo ba kỹ thuật mà bạn sẽ dùng:
          </p>
        }
      >
        <p>
          The book uses breadth-first search, BFS, as its example of work whose amount and place change while the
          program runs: which vertices need attention at the next step depends on what the current step found. BFS
          itself is not part of an LLM inference stack, and this course treats it as a vehicle. It carries three
          techniques that you will use:
        </p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>
              <b>chỉ khởi chạy luồng cho những gì đã thay đổi</b>: một frontier, danh sách các đỉnh mà bước vừa rồi phát hiện
              (§18.5);
            </li>
            <li>
              <b>dựng danh sách đó trong một hàng đợi riêng của mỗi khối</b> và ghi nó ra bằng một atomic toàn cục cho mỗi
              khối, tức là phép lọc riêng tư hóa (privatization) của chương 12 cộng thêm một lối thoát khi hàng đợi riêng bị
              đầy (§18.6);
            </li>
            <li>
              <b>chạy nhiều bước phụ thuộc nhau trong một lần khởi chạy</b>, ngăn cách bởi một rào chắn (barrier) trên toàn
              lưới (§18.7).
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            <b>launch threads only for what changed</b>: a frontier, the list of vertices the last step discovered
            (§18.5);
          </li>
          <li>
            <b>build that list in a queue private to each block</b> and commit it with one global atomic per block,
            which is chapter 12's privatised filter plus a way out when the private queue fills up (§18.6);
          </li>
          <li>
            <b>run many dependent steps inside one launch</b>, separated by a barrier across the whole grid (§18.7).
          </li>
        </ul>
      </Translated>
      <Source pages="425, 449" />
      <Translated
        vi={
          <p>
            Giải mã từng token một có đúng hình dạng mà sách mô tả cho các frontier nhỏ (cách chúng tôi đọc): mỗi bước khởi
            chạy nhiều kernel ngắn, và mỗi lần khởi chạy có một chi phí cố định mà lượng công việc nhỏ bên trong không che
            được. Cách chữa của sách là một lần khởi chạy cho nhiều mức. Các engine suy luận dùng cùng cách chữa ở dạng khác:
            ghi lại các lần khởi chạy của một bước một lần rồi phát lại chúng.
          </p>
        }
      >
        <p>
          Token-by-token decoding has the shape the book describes for small frontiers (our reading): every step
          launches many short kernels, and each launch has a fixed cost that the little work inside it cannot hide. The
          book's remedy is one launch for many levels. Inference engines reach for the same remedy in other forms,
          recording a step's launches once and replaying them.
        </p>
      </Translated>
      <Beyond title={tr('Bridge (ours)', 'Cầu nối (của khóa học này)')}>
        <Translated
          vi={
            <p>
              Sách chỉ bàn về chi phí khởi chạy cho BFS trên các đồ thị có frontier nhỏ, như mạng lưới đường bộ (p. 445). Việc
              liên hệ với decode là của chúng tôi.
            </p>
          }
        >
          <p>
            The book discusses launch overhead only for BFS on graphs with small frontiers, such as road networks (p.
            445). The mapping to decoding is ours.
          </p>
        </Translated>
      </Beyond>
      <Today checked={tr('2026-11-20 (notes/today/ch18-facts.md D3, D5, D6)', '2026-11-20 (notes/today/ch18-facts.md, mục D3, D5, D6)')}>
        <Translated
          vi={
            <p>
              llama.cpp bật CUDA graph theo mặc định khi bạn tự build llama.cpp, và bật HIP graph theo mặc định. Metal không
              có API graph, nên backend của nó mã hóa bằng các dispatch đồng thời và rào chắn bộ nhớ, và chia việc mã hóa cho
              nhiều luồng CPU. CUDA graph của PyTorch cấm các thao tác đồng bộ hóa với CPU, chẳng hạn{' '}
              <code>.item()</code>: chính là lần đọc kích thước frontier ở mỗi mức mà vòng lặp host của BFS thực hiện. Tài
              liệu của MLX gọi công việc trên các mảng nhỏ là "overhead bound" (bị giới hạn bởi chi phí phụ).
            </p>
          }
        >
          <p>
            llama.cpp turns CUDA graphs on by default when you build llama.cpp itself, and HIP graphs on by default.
            Metal has no graph API, so its backend encodes with concurrent dispatches and memory barriers and spreads the
            encoding over CPU threads. PyTorch's CUDA graphs prohibit operations that synchronise with the CPU, such as{' '}
            <code>.item()</code>: exactly the per-level read of the frontier size a BFS host loop makes. MLX's
            documentation calls work on small arrays "overhead bound".
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Đồ thị trong các định dạng của chương 17">Graphs in chapter 17's formats</Translated></h2>
      <Translated
        vi={
          <p>
            Một đồ thị là một tập đỉnh được nối bởi các cạnh. Các cạnh trong chương này có hướng: một cạnh đi từ một đỉnh
            nguồn tới một đỉnh đích, và một quan hệ hai chiều là hai cạnh. Đánh số các đỉnh từ 0 thì đồ thị là một ma trận
            vuông có số 1 ở hàng <i>i</i>, cột <i>j</i> cho mỗi cạnh từ <i>i</i> tới <i>j</i>. Một hàng chứa các cạnh ra của
            một đỉnh, một cột chứa các cạnh vào. Gần như mọi phần tử đều là 0: ví dụ xuyên suốt có 9 đỉnh và 15 cạnh, trong
            khi một đồ thị liên thông đầy đủ 9 đỉnh sẽ có 72.
          </p>
        }
      >
        <p>
          A graph is a set of vertices joined by edges. The chapter's edges are directed: an edge goes from a source
          vertex to a destination vertex, and a two-way relation is two edges. Number the vertices from 0 and the graph is
          a square matrix with a 1 in row <i>i</i>, column <i>j</i> for each edge from <i>i</i> to <i>j</i>. A row holds a
          vertex's outgoing edges, a column its incoming ones. Almost every entry is 0: the running example has 9
          vertices and 15 edges, where a fully connected graph of 9 vertices would have 72.
        </p>
      </Translated>
      <Source section="18.1" pages="425–427" figs={tr('Figs. 18.1, 18.2', 'Hình 18.1, 18.2')} />
      <Translated
        vi={
          <p>
            Vì vậy các định dạng ma trận thưa của <a href={href('ch17', '17.2')}>chương 17</a> áp dụng được, dưới tên mới. CSR
            giữ các hàng dưới dạng <code>srcPtrs</code> (offset của hàng) và <code>dst</code> (chỉ số cột). CSC giữ các cột
            dưới dạng <code>dstPtrs</code> và <code>src</code>. COO giữ mỗi cạnh dưới dạng một cặp <code>src</code>,{' '}
            <code>dst</code>. Mọi giá trị được lưu đều sẽ là 1, nên mảng dữ liệu bị bỏ: một chỉ số tồn tại nghĩa là một cạnh
            tồn tại. Như vậy CSR còn 25 số, 10 offset và 15 đỉnh đích, so với 81 cho ma trận dày đặc.
          </p>
        }
      >
        <p>
          So the sparse formats of <a href={href('ch17', '17.2')}>chapter 17</a> apply, under new names. CSR keeps the rows as{' '}
          <code>srcPtrs</code> (row offsets) and <code>dst</code> (column indices). CSC keeps the columns as{' '}
          <code>dstPtrs</code> and <code>src</code>. COO keeps each edge as a <code>src</code>, <code>dst</code> pair.
          Every stored value would be 1, so the data array is dropped: an index existing means an edge exists. That
          leaves 25 numbers for CSR, 10 offsets and 15 destinations, against 81 for the dense matrix.
        </p>
      </Translated>
      <Source section="18.1" pages="427–428" figs={tr('Fig. 18.3', 'Hình 18.3')} />
      <Translated
        vi={
          <p>
            Hãy đọc nó theo cách sách làm: <code>srcPtrs[3] = 7</code> và <code>srcPtrs[4] = 9</code>, nên các cạnh ra của
            đỉnh 3 là phần tử 7 và 8 của <code>dst</code>, chứa 4 và 8. Các cạnh là 3 → 4 và 3 → 8. Kiểm tra điều đó trong
            lăng kính, rồi bấm vào một đỉnh chỉ nhận cạnh, và một cạnh.
          </p>
        }
      >
        <p>
          Read it the way the book does: <code>srcPtrs[3] = 7</code> and <code>srcPtrs[4] = 9</code>, so vertex 3's
          outgoing edges are entries 7 and 8 of <code>dst</code>, which hold 4 and 8. The edges are 3 → 4 and 3 → 8.
          Check that in the lens, then click a vertex that only receives edges, and an edge.
        </p>
      </Translated>
      <FormatLens />
      <Translated
        vi={
          <p>
            Mỗi định dạng làm cho một câu hỏi trở nên rẻ. CSR trả lời "từ <i>v</i> tôi đi được tới đâu?" bằng một đoạn. CSC
            trả lời "ai trỏ tới <i>v</i>?" bằng một đoạn. COO trả lời "cạnh <i>e</i> bắt đầu và kết thúc ở đâu?" bằng hai lần
            đọc. Hỏi nhầm định dạng thì bạn phải tìm kiếm: dùng CSR để tìm các đỉnh trỏ tới 6 nghĩa là phải xem qua mọi hàng.
            Trang sau dựng một kernel BFS trên mỗi định dạng, và mỗi kernel hỏi đúng câu hỏi mà định dạng của nó trả lời rẻ.
          </p>
        }
      >
        <p>
          Each format makes one question cheap. CSR answers "where can I go from <i>v</i>?" with one slice. CSC answers
          "who points at <i>v</i>?" with one slice. COO answers "where does edge <i>e</i> start and end?" with two reads.
          Ask the wrong format and you search: CSR for the vertices pointing at 6 means looking through every row. The
          next page builds a BFS kernel on each format, and each kernel asks the question its format answers cheaply.
        </p>
      </Translated>
      <Source section="18.1" pages="428–429" />
      <Translated
        vi={
          <p>
            Kernel nào hợp với một đồ thị còn tùy vào cách các cạnh của nó phân bố. Bậc của một đỉnh là số cạnh của nó. Mạng
            lưới đường bộ có bậc thấp và đều: một giao lộ nối vài con đường. Đồ thị người theo dõi thì rộng, với một vài tài
            khoản nổi tiếng có số cạnh nhiều hơn phần còn lại nhiều bậc độ lớn. Trang sau chạy cả hai loại.
          </p>
        }
      >
        <p>
          Which kernel suits a graph also depends on how its edges are spread. A vertex's degree is its number of edges.
          Road networks have low, even degrees: an intersection joins a handful of roads. Follower graphs are broad, with
          a few popular accounts holding orders of magnitude more edges than the rest. The next page runs both kinds.
        </p>
      </Translated>
      <Source section="18.1" pages="428" />
      <Ask q={questionsS180['ch18/start/q-csr-read']} label={tr('Read the arrays', 'Đọc các mảng')} />
      <Ask q={questionsS180['ch18/start/q-csc-in']} label={tr('Read the arrays', 'Đọc các mảng')} />

      <h2><Translated vi="Tìm kiếm theo chiều rộng">Breadth-first search</Translated></h2>
      <Translated
        vi={
          <p>
            BFS tìm số cạnh ít nhất từ một gốc tới mọi đỉnh khác, và gán cho mỗi đỉnh con số đó, gọi là mức của nó. Gốc ở mức
            0, các đỉnh kề ra của nó ở mức 1, các đỉnh kề ra chưa thăm của chúng ở mức 2, và cứ thế. Với gốc 0, ví dụ có 1 và
            2 ở mức 1, 3 đến 7 ở mức 2, và 8 ở mức 3.
          </p>
        }
      >
        <p>
          BFS finds the smallest number of edges from a root to every other vertex, and labels each vertex with that
          number, its level. The root is level 0, its out-neighbours level 1, their unvisited out-neighbours level 2, and
          so on. With root 0, the example has 1 and 2 at level 1, 3 to 7 at level 2, and 8 at level 3.
        </p>
      </Translated>
      <Source section="18.2" pages="429–430" figs={tr('Fig. 18.4(a)', 'Hình 18.4(a)')} />
      <Translated
        vi={
          <p>
            Trước khi di chuyển gốc, hãy dự đoán việc dời gốc đi một cạnh ảnh hưởng thế nào tới một đỉnh. Sau đó trình khám
            phá sẽ mở, và bạn có thể di chuyển gốc tới bất cứ đâu.
          </p>
        }
      >
        <p>
          Before you move the root, predict what one edge's move does to a single vertex. Then the explorer opens, and you
          can move the root anywhere.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các mức của trình khám phá lấy từ một BFS tuần tự thông thường. Khi một đỉnh có nhiều đỉnh kề vào thấp hơn một
              mức, phép lần ngược lấy đỉnh có số nhỏ nhất: quy tắc đó là của chúng tôi, và lựa chọn nào cũng cho một đường đi
              ngắn nhất.
            </p>
          }
        >
          <p>
            The explorer's levels come from an ordinary sequential BFS. Where a vertex has several in-neighbours one level
            lower, the walk back takes the lowest-numbered: that rule is ours, and any choice gives a shortest path.
          </p>
        </Translated>
      </Simplification>
      <Predict
        id="ch18/start/p-root2"
        prompt={
          <Translated
            vi={
              <>
                Dời gốc đi một cạnh, từ 0 sang 2 (0 → 2 là một cạnh). Giờ đỉnh 4 ở mức nào? Trước đó nó ở mức 2.
              </>
            }
          >
            <>
              Move the root one edge along, from 0 to 2 (0 → 2 is an edge). At what level is vertex 4 now? It was at level
              2.
            </>
          </Translated>
        }
      >
        {(v) => (
          <Translated
            vi={
              <>
                <p>
                  {v === 4 ? 'Đúng: ' : 'Nó ở '}mức 4, theo 2 → 7 → 0 → 1 → 4. Từ 2, phép tìm kiếm phải đi đường vòng, quay lại
                  qua 0, để tới 1 rồi 4. Giờ hãy bấm vào bất kỳ đỉnh nào trong trình khám phá bên dưới để biến nó thành gốc.
                </p>
              </>
            }
          >
            <>
              <p>
                {v === 4 ? 'Right: ' : 'It is at '}level 4, by 2 → 7 → 0 → 1 → 4. From 2 the search must go the long way
                round, back through 0, to reach 1 and then 4. Now click any vertex in the explorer below to make it
                the root.
              </p>
            </>
          </Translated>
        )}
      </Predict>
      <RootGate />
      <Translated
        vi={
          <p>
            Các cạnh mà phép tìm kiếm dùng để tới mỗi đỉnh tạo thành cây BFS, vẽ đậm: một cạnh vào mỗi đỉnh tới được, trừ gốc.
            Để khôi phục một đường đi ngắn nhất tới một đỉnh, bắt đầu từ đó và bước tới một đỉnh kề vào bất kỳ thấp hơn đúng
            một mức, cho đến khi tới gốc. Với gốc 0, từ đỉnh 8: các đỉnh kề vào 3, 4 và 6 của nó đều ở mức 2, nên đỉnh nào cũng
            được; trình khám phá lấy 3, rồi 1, rồi 0. Hãy để ý phép lần ngược đọc gì: các cạnh vào của mỗi đỉnh. Đó là câu hỏi
            của CSC, không phải của CSR.
          </p>
        }
      >
        <p>
          The edges a search uses to reach each vertex form the BFS tree, drawn bold: one edge into every reached vertex
          but the root. To recover a shortest path to a vertex, start there and step to any in-neighbour exactly one level
          lower, until you reach the root. With root 0, from vertex 8: its in-neighbours 3, 4 and 6 are all at level 2,
          so any of them will do; the explorer takes 3, then 1, then 0. Notice what the walk reads: each vertex's
          incoming edges. That is the CSC question, not the CSR one.
        </p>
      </Translated>
      <Source section="18.2" pages="430" />
      <Ask q={questionsS180['ch18/start/q-path']} />
      <Translated
        vi={
          <p>
            Thiết kế chip dùng trực tiếp điều này. Trong định tuyến mê cung (maze routing), lưới đi dây là đồ thị: mỗi ô trống
            là một đỉnh, mỗi bước một dây có thể đi tới ô bên cạnh là một cạnh, và các ô đã dùng là chướng ngại phải đi vòng.
            Một phép tìm kiếm từ một đầu nối lan ra như một wavefront cho đến khi chạm đầu kia, và phép lần ngược là dây ngắn
            nhất. Trang sau phát wavefront đó dưới dạng hoạt hình.
          </p>
        }
      >
        <p>
          Chip design uses this directly. In maze routing, the wiring grid is the graph: each free block is a vertex,
          each step a wire can take to a neighbouring block is an edge, and blocks already used are blockages to route
          around. A search from one terminal spreads as a wavefront until it reaches the other, and the walk back is the
          shortest wire. The next page plays that wavefront as an animation.
        </p>
      </Translated>
      <Source section="18.2" pages="430–431" figs={tr('Fig. 18.5', 'Hình 18.5')} />

      <h2><Translated vi="Chương này trong khóa học">The chapter in this course</Translated></h2>
      <p>
        <Translated vi="Chương này được rút gọn. Đây là nơi các mục của sách được đặt.">This chapter is condensed. Here is where the book's sections went.</Translated>
      </p>
      <table className="data">
        <thead>
          <tr>
            <th><Translated vi="Sách">Book</Translated></th>
            <th><Translated vi="Ở đây">Here</Translated></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>§18.1, §18.2 (pp. 425–431)</td>
            <td><Translated vi="trang này">this page</Translated></td>
          </tr>
          <tr>
            <td>
              <Translated vi="§18.3 đến §18.5 (pp. 431–442)">§18.3 to §18.5 (pp. 431–442)</Translated>
            </td>
            <td>
              <Translated
                vi={
                  <>
                    <a href={href('ch18', '18.3')}>18.3 Đỉnh, cạnh hay một frontier</a>: push và pull, lấy cạnh làm trung tâm,
                    hiệu quả công việc, frontier, compare-and-swap
                  </>
                }
              >
                <a href={href('ch18', '18.3')}>18.3 Vertices, edges or a frontier</a>: push and pull, edge-centric, work
                efficiency, frontiers, compare-and-swap
              </Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi="§18.6 đến §18.8 (pp. 442–449)">§18.6 to §18.8 (pp. 442–449)</Translated>
            </td>
            <td>
              <a href={href('ch18', '18.6')}>
                <Translated vi="18.6 Hàng đợi riêng và một lần khởi chạy cho mọi mức">18.6 Private queues and one launch for every level</Translated>
              </a>
            </td>
          </tr>
          <tr>
            <td><Translated vi="của khóa học này">ours</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    <a href={href('ch18', 'labs')}>Lab 18.1</a>, tùy chọn: BFS dùng frontier với một hàng đợi riêng tư hóa
                  </>
                }
              >
                <a href={href('ch18', 'labs')}>Lab 18.1</a>, optional: frontier BFS with a privatised queue
              </Translated>
            </td>
          </tr>
          <tr>
            <td>§18.9, §18.10 (pp. 449–450)</td>
            <td>
              <a href={href('ch18', 'wrap')}>
                <Translated vi="Tóm tắt và bài tập">Summary and exercises</Translated>
              </a>
            </td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Rút gọn còn một câu hoặc bỏ qua: các heuristic mà bộ định tuyến dùng để chọn giữa những dây dài bằng nhau (p. 431),
            đồ thị thế giới nhỏ (small-world) (p. 436), cách phát biểu BFS đầy đủ bằng đại số tuyến tính (p. 438), và các bài
            toán mở về đồ thị lớn hơn bộ nhớ GPU và về sắp xếp lại các đỉnh (p. 449).
          </p>
        }
      >
        <p>
          Cut to a sentence or left out: the heuristics routers use to choose between equally short wires (p. 431),
          small-world graphs (p. 436), the full linear-algebra formulation of BFS (p. 438), and the open problems of
          graphs larger than GPU memory and of reordering vertices (p. 449).
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Chương 17 cũng được rút gọn. Các định dạng mà trang này mượn nằm ở trang{' '}
              <a href={href('ch17', '17.2')}>17.2 Định dạng quyết định kernel</a> của chương đó.
            </p>
          }
        >
          <p>
            Chapter 17 is condensed too. The formats this page borrows are on its page{' '}
            <a href={href('ch17', '17.2')}>17.2 Formats decide the kernel</a>.
          </p>
        </Translated>
      </Aside>

      <h2><Translated vi="Trước khi bắt đầu">Before you start</Translated></h2>
      <Translated
        vi={
          <p>
            Sáu câu hỏi về những gì chương này giả định từ các chương 4, 9, 11, 12 và 17. Mỗi câu sẽ quay lại trong phần{' '}
            <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Six questions on what this chapter assumes from chapters 4, 9, 11, 12 and 17. Each comes back in{' '}
          <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={questionsS180['ch18/start/q-divergence']} label={tr('From chapter 4', 'Từ chương 4')} />
      <Ask q={questionsS180['ch18/start/q-resident']} label={tr('From chapters 4 and 11', 'Từ chương 4 và 11')} />
      <Ask q={questionsS180['ch18/start/q-workeff']} label={tr('From chapter 11', 'Từ chương 11')} />
      <Ask q={questionsS180['ch18/start/q-slot']} label={tr('From chapter 12', 'Từ chương 12')} />
      <Ask q={questionsS180['ch18/start/q-scope']} label={tr('From chapter 9', 'Từ chương 9')} />
      <Ask q={questionsS180['ch18/start/q-rowthread']} label={tr('From chapters 17 and 6', 'Từ chương 17 và 6')} />
    </>
  );
}
