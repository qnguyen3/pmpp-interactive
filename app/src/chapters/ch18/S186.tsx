import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { siteLine } from '../../sim/bfspriv';
import { gridSummary } from '../../sim/gridbfs';
import { region } from '../../sim/histpriv';
import { functionSource } from '../../sim/toolchain';
import { GridBarrierStepper } from './GridBarrierStepper';
import { OneLaunchScene } from './OneLaunchScene';
import { PRIV_TIERS, PrivateQueueStepper } from './PrivateQueueStepper';

// Section 18.6 of this course: the book's §18.6 to §18.8 (pp. 442-449). What the book says
// comes from notes/ch18.md; vendor facts from notes/today/ch18-facts.md (A1-A5, A7, A8, A10,
// A11, A13, B1-B6, C1-C4, C6, D1-D5, E1). Models: src/sim/bfspriv.ts (the private-queue
// stepper) and src/sim/gridbfs.ts (the grid-barrier stepper, "Does it fit?", the prediction,
// the scene). Listings: labs/ch18/snippets/s186_privatised_variants.cu and
// labs/ch18/snippets/s186_coop_swap.cu, loaded at run time; S186.snippets.test.ts ties the
// code this page quotes in its prose to those files.

const GRID = gridSummary(1000);
const fmt = (n: number) => n.toLocaleString('en-US');

export const questionsS186: Record<string, Question> = {
  'ch18/18.6/q-atomics': {
    kind: 'numeric',
    id: 'ch18/18.6/q-atomics',
    prompt: (
      <Translated
        vi={
          <>
            Một mức nối thêm 3,000 đỉnh từ 24 khối, và không hàng đợi riêng của khối nào bị tràn. Hình 18.12 thực hiện 3,000
            phép toán nguyên tử (atomic) trên bộ đếm toàn cục, mỗi đỉnh một phép. Hình 18.15 thực hiện bao nhiêu phép?
          </>
        }
      >
        <>
          One level appends 3,000 vertices from 24 blocks, and no block's private queue overflows. Fig. 18.12 makes 3,000
          atomic operations on the global counter, one per vertex. How many does Fig. 18.15 make?
        </>
      </Translated>
    ),
    answer: 24,
    unit: 'atomics',
    explain: (
      <Translated
        vi={
          <p>
            Mỗi khối một phép: lần đặt chỗ của luồng 0 cho toàn bộ khoảng của khối. 3,000 lần nối vẫn diễn ra, nhưng trên 24 bộ
            đếm riêng trong bộ nhớ chia sẻ (shared memory), mỗi bộ đếm chỉ bị tranh chấp bởi chính khối của nó. Một lần tràn sẽ
            thêm một atomic toàn cục cho mỗi phần tử không vừa.
          </p>
        }
      >
        <p>
          One per block: thread 0's reservation of the block's whole range. The 3,000 appends still happen, but on 24
          private counters in shared memory, each contended only by its own block. An overflow would add one global atomic
          per entry that does not fit.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.6, p. 442; src/sim/bfspriv.ts (levelAtomics)', // i18n-ignore: citation
    vi: { unit: 'atomic' },
  },
  'ch18/18.6/q-overflow': {
    kind: 'numeric',
    id: 'ch18/18.6/q-overflow',
    prompt: (
      <Translated
        vi={
          <>
            Dung lượng 8, với cách đặt lại bộ đếm đã sửa. Một khối 4 luồng phát hiện 13 đỉnh mới trong một mức. Bao nhiêu đỉnh
            trong số đó đi thẳng vào frontier chung?
          </>
        }
      >
        <>
          Capacity 8, with the corrected counter reset. A block of 4 threads discovers 13 new vertices in one level. How
          many of them go straight to the public frontier?
        </>
      </Translated>
    ),
    answer: 5,
    unit: 'entries',
    explain: (
      <Translated
        vi={
          <p>
            8 lần tăng đầu tiên trả về 0 đến 7 và nhận các ô riêng. 5 lần còn lại trả về 8 trở lên, đặt bộ đếm trở lại 8 và
            nối vào frontier chung bằng một atomic phạm vi thiết bị. Mức có sẵn hub của trình chạy từng bước với dung lượng 8
            và 1 khối 4 luồng chính là trường hợp này.
          </p>
        }
      >
        <p>
          The first 8 increments return 0 to 7 and get private slots. The other 5 return 8 or more, put the counter back
          to 8 and append to the public frontier with a device-scope atomic. The hub preset of the stepper with capacity
          8 and 1 block of 4 threads is this case.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.6, pp. 443–444',
    vi: { unit: 'phần tử' },
  },
  'ch18/18.6/q-overflow-counter': {
    kind: 'numeric',
    id: 'ch18/18.6/q-overflow-counter',
    prompt: (
      <Translated
        vi={
          <>
            Cùng khối đó: dung lượng 8, 13 đỉnh mới, cách đặt lại bộ đếm đã sửa. Bộ đếm riêng <code>numCurrFrontier_s</code>{' '}
            giữ giá trị bao nhiêu sau rào chắn 2, khi luồng 0 đọc nó để đặt chỗ trong frontier chung?
          </>
        }
      >
        <>
          The same block: capacity 8, 13 new vertices, the corrected counter reset. What does the private counter{' '}
          <code>numCurrFrontier_s</code> hold after barrier 2, when thread 0 reads it to reserve public slots?
        </>
      </Translated>
    ),
    answer: 8,
    unit: 'entries',
    explain: (
      <Translated
        vi={
          <p>
            Mỗi lần tăng vượt dung lượng đều được theo sau bởi lần ghi dung lượng của chính luồng đó, nên dù xen kẽ thế nào bộ
            đếm cũng kết thúc ở min(13, 8) = 8: đúng số phần tử riêng cần sao chép ra. Mười ba sẽ đặt chỗ 5 ô chung mà không ai
            lấp. Phần "dù xen kẽ thế nào" là kiểm tra của riêng chúng tôi, bằng cách liệt kê mọi cách xen kẽ, không phải của nhà
            cung cấp.
          </p>
        }
      >
        <p>
          Every increment past capacity is followed by that thread's store of the capacity, so whatever the interleaving
          the counter ends at min(13, 8) = 8: exactly the number of private entries to copy out. Thirteen would reserve 5
          public slots that nobody fills. The "whatever the interleaving" is our own check, by enumerating every
          interleaving, not a vendor's.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.6, pp. 443–444; notes/today/ch18-facts.md E1 (ours)',
    vi: { unit: 'phần tử', source: 'ch. 18 §18.6, pp. 443–444; notes/today/ch18-facts.md E1 (của chúng tôi)' },
  },
  'ch18/18.6/q-line29': {
    kind: 'choice',
    id: 'ch18/18.6/q-line29',
    prompt: (
      <Translated
        vi={
          <>
            Dòng <code>numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;</code> của sách cho số đếm đúng trong mọi bài kiểm tra
            bạn chạy. Vì sao phải sửa nó?
          </>
        }
      >
        <>
          The book's <code>numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;</code> gives the right counts in every test you
          run. Why change it?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nó tranh chấp với fetch_add của các luồng khác trên cùng bộ đếm: một race condition (tranh chấp dữ liệu), không xác định theo mô hình bộ nhớ ở pp. 433–434.">
            It races with other threads' fetch_add on the same counter: a data race, undefined under the memory model of pp. 433–434.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nó làm mất các lần tăng: một luồng cộng vào giữa fetch_add và lần ghi của luồng khác bị xóa mất, nên số đếm ra thấp hơn.">
            It loses increments: a thread that adds between another's fetch_add and its store is wiped out, so the count comes out low.
          </Translated>
        ),
        why: (
          <Translated vi="Chỉ luồng ở mức hoặc vượt dung lượng mới ghi, và nó ghi chính dung lượng. Mọi cách xen kẽ đều kết thúc ở min(N, capacity), đó là con số đúng. Lỗi nằm ở chính race.">
            Only a thread at or past capacity stores, and it stores the capacity. Every interleaving ends at min(N, capacity), which is the right number. The defect is the race itself.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó chậm hơn một lần ghi atomic, vì lần ghi thường buộc bộ đếm rời khỏi bộ nhớ chia sẻ.">
            It is slower than an atomic store, because the plain store forces the counter out of shared memory.
          </Translated>
        ),
        why: (
          <Translated vi="Tốc độ không phải vấn đề; một lần ghi thường không chậm hơn, và bộ đếm vẫn ở trong bộ nhớ chia sẻ. Vấn đề là nó là hành vi không xác định.">
            Speed is not the issue; a plain store is no slower, and the counter stays in shared memory. The issue is that it is undefined behaviour.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không cần sửa gì: lần ghi là vô hại, vì mọi luồng chạm tới nó đều ghi cùng một dung lượng.">
            Nothing needs to change: the store is benign, because every thread that reaches it writes the same capacity.
          </Translated>
        ),
        why: (
          <Translated vi="Bản thân sách nói rằng các race lũy đẳng vẫn vi phạm mô hình bộ nhớ (pp. 433–434), và ở đây lần ghi còn tranh chấp với các phép đọc-sửa-ghi, không chỉ với các lần ghi cùng giá trị.">
            The book itself says idempotent races still violate the memory model (pp. 433–434), and here the store also races with read-modify-writes, not only with stores of the same value.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một race là không xác định, dù các con số có ra đúng trên phần cứng bạn thử hay không. Bảng race của trình chạy từng
            bước báo dòng này đối với <code>fetch_add</code>, và backend race của khóa học cũng báo mỗi khi hai luồng của một
            khối cùng tràn một lúc. Trên Metal, bộ đếm là một <code>threadgroup atomic_uint</code>, không thể gán trực tiếp,
            nên chỉ viết được dạng atomic.
          </p>
        }
      >
        <p>
          A race is undefined whether or not the numbers come out right on the hardware you test. The stepper's race
          panel reports this line against the <code>fetch_add</code>, and the course's race backend does whenever two
          threads of a block overflow at the same time. On Metal the
          counter is a <code>threadgroup atomic_uint</code>, which cannot be assigned plainly, so only the atomic form can
          be written.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.6, p. 444; facts A7, C4, E1',
    vi: { source: 'ch. 18 §18.6, p. 444; dữ kiện A7, C4, E1' },
  },
  'ch18/18.6/q-two-runs': {
    kind: 'choice',
    id: 'ch18/18.6/q-two-runs',
    prompt: (
      <Translated vi={<>Hai lần chạy Hình 18.15 trên cùng đồ thị cho ra hai frontier chung khác nhau. Điều gì có thể khác giữa chúng?</>}>
        <>Two runs of Fig. 18.15 on the same graph produce different public frontiers. What can differ between them?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Vị trí khoảng của mỗi khối, thứ tự bên trong nó, và nơi các phần tử tràn rơi vào; không phải tập đỉnh hay kích thước của nó.">
            Where each block's range sits, the order inside it, and where overflow entries land; not the set of vertices or its size.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Lẽ ra không có gì khác: các khối đặt chỗ khoảng của mình theo thứ tự khối, nên một khác biệt nghĩa là có lỗi.">
            Nothing should differ: the blocks reserve their ranges in block order, so a difference means a bug.
          </Translated>
        ),
        why: (
          <Translated vi="Các atomic trên một địa chỉ được phục vụ không theo thứ tự cố định nào, nên các khối đặt chỗ theo bất kỳ thứ tự nào chúng đến. Một phép lọc không ổn định được phép ra theo bất kỳ thứ tự nào.">
            Atomics on one address are served in no fixed order, so blocks reserve in whatever order they arrive. An unstable filter is allowed to come out in any order.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Số lượng: phần tử nào tràn phụ thuộc vào thời điểm, nên một đỉnh có thể bị nối hai lần, một lần qua hàng đợi riêng và một lần thẳng vào frontier chung.">
            The count: which entries overflow depends on timing, so a vertex can be appended twice, once through its private queue and once straight to the public frontier.
          </Translated>
        ),
        why: (
          <Translated vi="Phần tử nào tràn phụ thuộc vào thời điểm; số đỉnh được nối thì không. Mỗi đỉnh mới đi một đường, riêng hoặc chung, và compare-and-swap nối nó đúng một lần.">
            Which entries overflow depends on timing; how many vertices are appended does not. Each new vertex takes one path, private or public, and the compare-and-swap appends it exactly once.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hình 18.16 vẽ đúng điều này: các phần tử của mỗi khối nằm trong một khoảng liên tục, các khoảng theo thứ tự đặt
            chỗ. Trong trình chạy từng bước, hãy đổi thứ tự khối và so sánh frontier chung: cùng tập, khác cách sắp xếp.
          </p>
        }
      >
        <p>
          Fig. 18.16 draws exactly this: each block's entries in one contiguous range, the ranges in reservation order.
          In the stepper, change the block order and compare the public frontier: same set, different arrangement.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.6, p. 444, Fig. 18.16; notes/today/ch18-facts.md "Already settled" (ch09 A3, B1)',
    vi: { source: 'ch. 18 §18.6, p. 444, Hình 18.16; notes/today/ch18-facts.md, mục "Already settled" (ch09 A3, B1)' },
  },
  'ch18/18.6/q-commit-if': {
    kind: 'choice',
    id: 'ch18/18.6/q-commit-if',
    prompt: (
      <Translated
        vi={
          <>
            Một đồng nghiệp commit bằng{' '}
            <code>if (threadIdx.x &lt; numCurrFrontier_s) currFrontier[start + threadIdx.x] = currFrontier_s[threadIdx.x];</code>.
            Nó qua mọi bài kiểm tra với dung lượng 64 và 128 luồng mỗi khối, và thất bại với dung lượng 256. Vì sao?
          </>
        }
      >
        <>
          A colleague commits with{' '}
          <code>if (threadIdx.x &lt; numCurrFrontier_s) currFrontier[start + threadIdx.x] = currFrontier_s[threadIdx.x];</code>.
          It passes every test with capacity 64 and 128 threads per block, and fails with capacity 256. Why?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mỗi luồng một phần tử: các phần tử từ chỉ số blockDim.x trở lên không bao giờ được sao chép, và dung lượng 64 thì không có phần tử nào như vậy.">
            One entry per thread: entries from index blockDim.x up are never copied, and capacity 64 has none.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Dung lượng 256 làm tràn bộ nhớ chia sẻ của một khối, nên lần khởi chạy thất bại hoặc hàng đợi riêng quay vòng.">
            Capacity 256 overflows the shared memory of a block, so the launch fails or the private queue wraps around.
          </Translated>
        ),
        why: (
          <Translated vi="256 word là 1 KB; lỗi là mất phần tử, không phải lỗi khởi chạy hay quay vòng.">
            256 words is 1 KB; the failure is lost entries, not a launch error or a wrap.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Phần commit không còn được hợp nhất (coalescing): với 256 phần tử, các luồng cạnh nhau ghi vào những địa chỉ cách xa nhau.">
            The commit is no longer coalesced: with 256 entries, neighbouring threads write addresses far apart.
          </Translated>
        ),
        why: (
          <Translated vi="Chúng vẫn liên tiếp. Vấn đề là tính đúng đắn: các ô được đặt chỗ cho phần tử 128 đến 255 không bao giờ được ghi.">
            They are still consecutive. The problem is correctness: slots reserved for entries 128 to 255 are never written.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Vòng lặp có bước nhảy (chỉ số <code>threadIdx.x + k·blockDim.x</code>) cho phép một khối sao chép hàng đợi dài bao
            nhiêu cũng được. Hãy thử trong trình chạy từng bước: commit bằng một <code>if</code>, 2 luồng mỗi khối và dung lượng
            4.
          </p>
        }
      >
        <p>
          The strided loop (index <code>threadIdx.x + k·blockDim.x</code>) lets a block copy a queue of any length. Try it in
          the stepper: commit with an <code>if</code>, 2 threads per block and capacity 4.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.6, p. 444; src/sim/bfspriv.ts', // i18n-ignore: citation
  },
  'ch18/18.6/q-toolarge': {
    kind: 'choice',
    id: 'ch18/18.6/q-toolarge',
    prompt: (
      <Translated
        vi={
          <>
            Một thiết bị có S SM, và truy vấn occupancy (mức chiếm dụng) cho biết k khối mỗi SM. Bạn yêu cầu S·k + 1 khối qua
            cooperative launch. Điều gì xảy ra trên CUDA, và trên HIP?
          </>
        }
      >
        <>
          A device has S SMs, and the occupancy query says k blocks per SM. You request S·k + 1 blocks through the
          cooperative launch. What happens on CUDA, and on HIP?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Cả hai đều từ chối khi khởi chạy: cudaErrorCooperativeLaunchTooLarge và hipErrorCooperativeLaunchTooLarge.">
            Both refuse it at launch: cudaErrorCooperativeLaunchTooLarge and hipErrorCooperativeLaunchTooLarge.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nó khởi chạy, và grid.sync() đầu tiên bị deadlock: khối thừa chờ một chỗ không bao giờ được giải phóng.">
            It launches, and the first grid.sync() deadlocks: the extra block waits for a slot that never frees.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là rủi ro sách mô tả (p. 445), nhưng API cooperative kiểm tra lưới với cùng tích số đó trước khi khởi chạy bất cứ thứ gì.">
            That is the risk the book describes (p. 445), but the cooperative API checks the grid against the same product before launching anything.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="CUDA từ chối nó với cudaErrorCooperativeLaunchTooLarge; HIP khởi chạy nó và bị deadlock.">
            CUDA refuses it with cudaErrorCooperativeLaunchTooLarge; HIP launches it and deadlocks.
          </Translated>
        ),
        why: (
          <Translated vi="Runtime của HIP so sánh lưới với giới hạn dựa trên occupancy của riêng nó và trả về hipErrorCooperativeLaunchTooLarge (cách chúng tôi đọc mã nguồn của nó).">
            HIP's runtime compares the grid with its own occupancy-based limit and returns hipErrorCooperativeLaunchTooLarge (our reading of its source).
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Deadlock ở p. 445 là thứ bạn chỉ gặp rủi ro khi đi vòng qua API cooperative. Hãy kiểm tra mã trả về của lần khởi
            chạy: đó là báo cáo duy nhất bạn nhận được.
          </p>
        }
      >
        <p>
          The deadlock of p. 445 is what you would risk only by going around the cooperative API. Check the return code
          of the launch: it is the only report you get.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.7, pp. 445–446; notes/today/ch18-facts.md A3, B3', // i18n-ignore: citation
  },
  'ch18/18.6/q-barrier2': {
    kind: 'choice',
    id: 'ch18/18.6/q-barrier2',
    prompt: (
      <Translated vi={<>Bỏ <code>grid.sync()</code> thứ hai của phép hoán đổi frontier. Điều gì có thể sai?</>}>
        <>Remove the second <code>grid.sync()</code> of the frontier swap. What can go wrong?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Luồng 0 có thể đặt lại bộ đếm trước khi một luồng khác đọc nó: luồng đó đọc được 0, rời vòng lặp, và các luồng còn lại treo ở grid.sync() kế tiếp.">
            Thread 0 may reset the counter before another thread reads it: that thread reads 0, leaves the loop, and the rest hang at the next grid.sync().
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Các lần nối của mức sau có thể tới bộ đếm trước khi luồng 0 đặt lại nó, nên lần đặt lại xóa mất chúng và những đỉnh đó không bao giờ được mở rộng.">
            Appends of the next level may reach the counter before thread 0 resets it, so the reset wipes them out and those vertices are never expanded at all.
          </Translated>
        ),
        why: <Translated vi="Đó là điều rào chắn thứ ba ngăn chặn.">That is what the third barrier prevents.</Translated>,
      },
      {
        label: (
          <Translated vi="Một luồng có thể đọc bộ đếm trước khi mọi khối nối xong, và nhận được số đếm chưa đầy đủ.">
            A thread may read the counter before every block has finished appending, and get a partial count.
          </Translated>
        ),
        why: <Translated vi="Đó là việc của rào chắn thứ nhất.">That is the first barrier's job.</Translated>,
      },
      {
        label: (
          <Translated vi="Không có gì: mọi luồng đều đã giữ số đếm, nên rào chắn thứ hai chỉ sắp xếp công việc cho nhanh.">
            Nothing: every thread already holds the count, so the second barrier only orders work for speed.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi rào chắn trong ba cái bảo vệ một thứ tự khác nhau. Hãy bỏ chọn nó trong trình chạy từng bước, theo thứ tự.">
            Each of the three guards a different ordering. Untick it in the stepper, in order.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong trình chạy từng bước, bỏ chọn grid.sync() 2 và chạy theo thứ tự (luồng 0 trước): luồng 0 đọc, đặt lại, và
            luồng đọc kế tiếp thấy 0. Một luồng rời kernel sẽ không bao giờ tới rào chắn kế tiếp, mà một rào chắn chờ mọi luồng
            của lưới.
          </p>
        }
      >
        <p>
          In the stepper, untick grid.sync() 2 and run in order (thread 0 first): thread 0 reads, resets, and the next
          reader sees 0. A thread that leaves the kernel never reaches the next barrier, and a barrier waits for every
          thread of the grid.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.7, pp. 447–448; src/sim/gridbfs.ts', // i18n-ignore: citation
  },
  'ch18/18.6/q-ex3': {
    kind: 'choice',
    id: 'ch18/18.6/q-ex3',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 3: cho mỗi mức một bộ đếm riêng, tất cả được đặt về 0 trước lần khởi chạy. Mỗi mức cần bao nhiêu rào chắn
            lưới, và ở đâu?
          </>
        }
      >
        <>
          Exercise 3: give each level its own counter, all zeroed before the launch. How many grid barriers does each
          level need, and where?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một, sau các lần nối và trước khi đọc counter[L]. Không có gì bị đặt lại, nên không lần đọc nào có thể tranh chấp với một lần đặt lại.">
            One, after the appends and before reading counter[L]. Nothing is reset, so no read can race with a reset.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Hai: một sau các lần nối, trước khi đọc counter[L], và một sau lần đọc, trước các lần nối kế tiếp.">
            Two: one after the appends, before reading counter[L], and one after the read, before the next appends.
          </Translated>
        ),
        why: (
          <Translated vi="Rào chắn sau lần đọc bảo vệ lần đặt lại. Không có đặt lại thì không có gì để nó bảo vệ.">
            The barrier after the read protected the reset. With no reset there is nothing for it to protect.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không cái nào: mỗi mức có bộ đếm riêng, nên không lần đọc nào gặp một lần ghi.">
            None: each level has its own counter, so no read can meet a write.
          </Translated>
        ),
        why: (
          <Translated vi="Một luồng vẫn không được đọc counter[L] trước khi mọi luồng đã nối vào nó.">
            A thread must still not read counter[L] before every thread has appended to it.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cái giá là giới hạn số mức: các bộ đếm được cấp phát trước lần khởi chạy, mỗi mức mà phép tìm kiếm có thể tới một
            bộ đếm. Sách gọi đây là một dạng tổng quát hơn của double buffering (p. 448). Kernel Bài tập 3 của trình chạy từng
            bước đúng với mọi thứ tự khi có một rào chắn của nó, và sai khi không có.
          </p>
        }
      >
        <p>
          The price is a bound on the number of levels: the counters are allocated before the launch, one per level the
          search may reach. The book calls this a more general form of double buffering (p. 448). The stepper's Exercise 3
          kernel is correct in every order with its one barrier, and wrong without it.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.7, p. 448; Exercise 3, p. 450; src/sim/gridbfs.ts',
    vi: { source: 'ch. 18 §18.7, p. 448; Bài tập 3, p. 450; src/sim/gridbfs.ts' },
  },
  'ch18/18.6/q-hip-launch': {
    kind: 'choice',
    id: 'ch18/18.6/q-hip-launch',
    prompt: (
      <Translated
        vi={
          <>
            Lời gọi <code>cudaLaunchCooperativeKernel((void*)bfs_kernel, numBlocks, numThreadsPerBlock, kernelArgs);</code>{' '}
            của sách biên dịch được cho CUDA nhưng không cho HIP. Dòng duy nhất nào chạy được cho cả hai thông qua header của
            khóa học?
          </>
        }
      >
        <>
          The book's <code>cudaLaunchCooperativeKernel((void*)bfs_kernel, numBlocks, numThreadsPerBlock, kernelArgs);</code>{' '}
          compiles for CUDA and not for HIP. Which single line works for both through the course's header?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi={'Cùng lời gọi đó, thêm ", 0, 0" cho kích thước bộ nhớ chia sẻ và stream.'}>
            The same call with ", 0, 0" added for the shared-memory size and the stream.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Cùng lời gọi đó với hip thay cho cuda: header đổi tên nó.">
            The same call with hip in place of cuda: the header renames it.
          </Translated>
        ),
        why: (
          <Translated vi="Header có đổi tên nó, nhưng hipLaunchCooperativeKernel không có đối số mặc định, nên bốn đối số vẫn không biên dịch được.">
            The header does rename it, but hipLaunchCooperativeKernel has no default arguments, so four arguments still do not compile.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="hipLaunchCooperativeKernel(bfs_kernel, numBlocks, numThreadsPerBlock, 0, 0, 0), như trong ví dụ của AMD.">
            hipLaunchCooperativeKernel(bfs_kernel, numBlocks, numThreadsPerBlock, 0, 0, 0), as in AMD's example.
          </Translated>
        ),
        why: (
          <Translated vi="Cách đó truyền 0 làm mảng đối số cho một kernel có sáu tham số. Chính ví dụ của AMD có lỗi nhỏ này; đừng sao chép nó.">
            That passes 0 as the argument array of a kernel that has six parameters. AMD's own example has this slip; do not copy it.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trên CUDA, lời gọi bốn đối số được phân giải thành một overload C++ có hai tham số cuối mặc định là 0, nên dạng sáu
            đối số cũng hợp lệ ở đó. llama.cpp truyền đủ sáu vì cùng lý do.
          </p>
        }
      >
        <p>
          On CUDA the four-argument call resolves to a C++ overload whose last two parameters default to 0, so the
          six-argument form is valid there too. llama.cpp passes all six for the same reason.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch18-facts.md A1, B1, D2', // i18n-ignore: citation
  },
  'ch18/18.6/q-metal-plan': {
    kind: 'choice',
    id: 'ch18/18.6/q-metal-plan',
    prompt: (
      <Translated vi={<>Bạn port BFS một lần khởi chạy sang Metal. Kế hoạch nào chạy được?</>}>
        <>You port the single-launch BFS to Metal. Which plan works?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mỗi mức một dispatch trên một compute encoder tuần tự của MTLCommandQueue, vốn sắp thứ tự chúng và theo dõi hazard, mỗi dispatch có kích thước lấy từ một indirect dispatch dựa trên số đếm do GPU ghi.">
            One dispatch per level on a serial compute encoder of an MTLCommandQueue, which orders them and tracks hazards, each sized by an indirect dispatch from a GPU-written count.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Một dispatch, với một threadgroup barrier ở cuối mỗi mức để mọi luồng thấy frontier mới.">
            One dispatch, with a threadgroup barrier at the end of each level so every thread sees the new frontier.
          </Translated>
        ),
        why: (
          <Translated vi="Một threadgroup barrier đồng bộ hóa một threadgroup, không phải cả lưới. Metal không có rào chắn trên toàn lưới.">
            A threadgroup barrier synchronises one threadgroup, not the grid. Metal has no grid-wide barrier.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một dispatch có kích thước lấy từ truy vấn occupancy, mọi threadgroup quay vòng trên một device atomic giữa các mức.">
            One dispatch sized from the occupancy query, every threadgroup spinning on a device atomic between levels.
          </Translated>
        ),
        why: (
          <Translated vi="Metal không có truy vấn occupancy và không ghi nhận bảo đảm tiến triển nào giữa các threadgroup, nên một rào chắn quay vòng có thể không bao giờ được giải phóng.">
            Metal has no occupancy query and documents no forward progress between threadgroups, so a spin barrier may never be released.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Mỗi mức một dispatch trên một concurrent encoder, không có rào chắn giữa chúng, để các mức có thể chồng lên nhau.">
            One dispatch per level on a concurrent encoder, with no barriers between them, so the levels can overlap.
          </Translated>
        ),
        why: (
          <Translated vi="Trên một concurrent encoder các dispatch có thể chồng lên nhau, và bạn chịu trách nhiệm sắp thứ tự chúng, bằng memoryBarrier(scope:).">
            On a concurrent encoder the dispatches may overlap, and you are responsible for ordering them, with memoryBarrier(scope:).
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>dispatchThreadgroups(indirectBuffer:...)</code> đọc số threadgroup ngay trước khi lưới bắt đầu, nên CPU không
            bao giờ đọc kích thước frontier. Số đếm 0 không được tài liệu hóa: hãy ghi ít nhất 1 và để kernel return sớm khi
            frontier trống. Thứ tự được bảo đảm trên <code>MTLCommandQueue</code>, không phải trên{' '}
            <code>MTL4CommandQueue</code>, vốn không theo dõi hazard nào.
          </p>
        }
      >
        <p>
          <code>dispatchThreadgroups(indirectBuffer:...)</code> reads the threadgroup count just before the grid starts,
          so the CPU never reads the frontier size. A count of 0 is not documented: write at least 1 and let the kernel
          return early on an empty frontier. The ordering holds on <code>MTLCommandQueue</code>, not on{' '}
          <code>MTL4CommandQueue</code>, which tracks no hazards.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch18-facts.md C1, C2, C3', // i18n-ignore: citation
  },
  'ch18/18.6/q-hubs': {
    kind: 'choice',
    id: 'ch18/18.6/q-hubs',
    prompt: (
      <Translated
        vi={
          <>
            Một frontier chứa 10,000 đỉnh bậc 2 và ba đỉnh bậc 50,000, mỗi đỉnh một luồng. Điều gì sai, và cách nào khắc phục?
          </>
        }
      >
        <>
          A frontier holds 10,000 vertices of degree 2 and three of degree 50,000, one thread per vertex. What goes wrong,
          and what fixes it?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Ba luồng làm việc lâu hơn phần còn lại 25,000 lần; cho mỗi hub một khối, hoặc mỗi cạnh một luồng.">
            Three threads work 25,000 times longer than the rest; give each hub a block, or a thread per edge.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không có gì: 10,000 đỉnh nhỏ giữ cho thiết bị bận trong khi ba luồng hub làm hết danh sách của chúng.">
            Nothing: the 10,000 small vertices keep the device busy while the three hub threads work through their lists.
          </Translated>
        ),
        why: (
          <Translated vi="Mức kết thúc khi luồng chậm nhất của nó kết thúc. Các đỉnh nhỏ xong gần như ngay lập tức, rồi ba luồng làm việc một mình.">
            The level ends when its slowest thread ends. The small vertices finish almost at once, and then three threads work alone.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="150,000 cạnh của các hub làm tràn các hàng đợi riêng; hãy làm mỗi hàng đợi đủ lớn để chứa hết những gì chúng tìm thấy.">
            The hubs' 150,000 edges overflow the private queues; make each queue large enough to hold all they find.
          </Translated>
        ),
        why: (
          <Translated vi="Nhánh tràn đã xử lý việc đó. Vấn đề là thời gian một luồng phải bỏ ra để duyệt 50,000 cạnh.">
            The overflow path handles that already. The problem is the time one thread spends walking 50,000 edges.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đó là chia nhóm theo bậc (luồng, warp hoặc khối tùy theo bậc) hoặc cách ánh xạ lấy cạnh làm trung tâm của §18.4 chỉ
            áp dụng cho các đỉnh nặng, đúng như phân đoạn bậc cao của cuGraph làm.
          </p>
        }
      >
        <p>
          That is degree bucketing (thread, warp or block by degree) or the edge-centric mapping of §18.4 applied to the
          heavy vertices only, which is what cuGraph's high-degree segment does.
        </p>
      </Translated>
    ),
    source: 'ch. 18 §18.8, pp. 448–449; notes/today/ch18-facts.md D1', // i18n-ignore: citation
  },
};

/** The three forms of the overflow branch, from the snippet file, with the reset line marked. */
function ResetForms() {
  const src = useLabFile('ch18/snippets/s186_privatised_variants.cu');
  if (!src) return null;
  // rows: how many source lines the marked statement spans (the atomic store and the clamp take two).
  const tab = (label: string, kernel: string, site: 'reset' | 'clamp', rows: number, status: string) => {
    const f = functionSource(src, kernel);
    const text = f ? f.text : `// ${kernel} not found`;
    const line = f ? siteLine(text, site) : -1;
    const highlight = line >= 0 ? Array.from({ length: rows }, (_, k) => line + 1 + k) : [];
    return {
      label: tr(label, ({ 'corrected: atomic store': 'đã sửa: ghi atomic', 'corrected: clamp': 'đã sửa: kẹp giá trị', "the book's line 29": 'dòng 29 của sách' } as Record<string, string>)[label] ?? label),
      code: text,
      tiers: PRIV_TIERS,
      highlight,
      status,
    };
  };
  const checked = tr(
    'emulator-checked (emu and race) and front-end-checked; unverified on CUDA and HIP hardware',
    'emulator-checked (emu và race) và front-end-checked; unverified trên phần cứng CUDA và HIP',
  );
  return (
    <Code
      tabs={[
        tab('corrected: atomic store', 'bfsPrivatised', 'reset', 2, checked), // i18n-ignore: the label is translated inside tab()
        tab('corrected: clamp', 'bfsPrivatisedClamp', 'clamp', 2, checked), // i18n-ignore: the label is translated inside tab()
        tab("the book's line 29", 'bfsPlainStore', 'reset', 1, tr('emulator-checked; the race emulator reports this line', 'emulator-checked; trình giả lập race báo dòng này')), // i18n-ignore: the label is translated inside tab()
      ]}
      caption={
        <Translated
          vi={
            <>
              Hình 18.15 theo ba cách, từ <code>labs/ch18/snippets/s186_privatised_variants.cu</code>. Được đánh dấu: dòng khác
              biệt. Dung lượng là một tham số template, để một bản build có thể thử từ 1 đến 8.
            </>
          }
        >
          <>
            Fig. 18.15 three ways, from <code>labs/ch18/snippets/s186_privatised_variants.cu</code>. Marked: the line that
            differs. The capacity is a template parameter, so that one build can try 1 to 8.
          </>
        </Translated>
      }
    />
  );
}

/** The host code of pp. 445-446 and the vendor-guarded include, from the cooperative snippet. */
function LaunchCode() {
  const src = useLabFile('ch18/snippets/s186_coop_swap.cu');
  if (!src) return null;
  const header = region(src, 'header');
  const launch = region(src, 'launch');
  const ex3 = functionSource(src, 'bfsCoopEx3');
  const status = tr(
    'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware',
    'emulator-checked và front-end-checked; unverified trên phần cứng CUDA và HIP',
  );
  return (
    <Code
      tabs={[
        { label: tr('sizing and launch', 'định kích thước và khởi chạy'), code: launch ? launch.text : '// region launch not found', status },
        { label: tr('include, CUDA and HIP', 'include, CUDA và HIP'), code: header ? header.text : '// region header not found', status },
        { label: tr('Exercise 3 kernel', 'kernel Bài tập 3'), code: ex3 ? ex3.text : '// bfsCoopEx3 not found', tiers: PRIV_TIERS, status },
      ]}
      caption={
        <Translated
          vi={
            <>
              Từ <code>labs/ch18/snippets/s186_coop_swap.cu</code>. Lần khởi chạy truyền sáu đối số trong khi sách truyền bốn,
              để cùng dòng đó biên dịch được cho HIP; include được chọn theo nhà cung cấp. Kernel đầy đủ, gồm cả hàng đợi riêng,
              là lời giải tham chiếu (solution) của trang lab.
            </>
          }
        >
          <>
            From <code>labs/ch18/snippets/s186_coop_swap.cu</code>. The launch passes six arguments where the book passes
            four, so the same line compiles for HIP; the include is chosen by vendor. The full kernel, private queue and
            all, is the labs page's reference solution.
          </>
        </Translated>
      }
    />
  );
}

export function S186() {
  return (
    <>
      <Translated
        vi={
          <p>
            Kernel frontier của trang trước còn hai chi phí: một bộ đếm mà mọi luồng nối vào trong lưới đều chạm tới, và một
            lần khởi chạy, kèm một lần đọc về, cho mỗi mức. Trang này loại bỏ chi phí thứ nhất bằng một hàng đợi riêng của mỗi
            khối và chi phí thứ hai bằng một rào chắn (barrier) trên toàn lưới, sửa một dòng có race trong mã của sách, và nói
            rõ mỗi backend làm được và không làm được gì.
          </p>
        }
      >
        <p>
          The frontier kernel of the last page has two costs left: one counter that every appending thread in the grid
          hits, and one launch, with a read-back, per level. This page removes the first with a queue private to each
          block and the second with a barrier across the whole grid, fixes one racy line in the book's code, and says
          what each backend can and cannot do.
        </p>
      </Translated>

      <h2><Translated vi="Bộ đếm là điểm nóng">The counter is the hot spot</Translated></h2>
      <Translated
        vi={
          <p>
            Trong Hình 18.12, mỗi đỉnh mới tốn một <code>fetch_add</code> trên <code>numCurrFrontier</code>, một địa chỉ trong bộ
            nhớ toàn cục (global memory), và sách dự kiến tranh chấp ở đó sẽ cao. Nối vào một frontier là phép lọc không ổn
            định của chương 12, và chương 12 có hai cách chữa cho bộ đếm của nó: gộp các atomic của một warp thành một, và riêng
            tư hóa (privatization). Với cách thứ nhất, sách nói bạn có thể giả định trình biên dịch sẽ gộp chúng. Hãy hiểu đó là
            "có thể".{' '}
            <a href={href('ch12', '12.3')}>Mục 12.3</a> đã cho thấy cách làm thủ công: một ballot, một phép đếm bit, một atomic
            mỗi warp, một lần broadcast.
          </p>
        }
      >
        <p>
          In Fig. 18.12 every new vertex costs one <code>fetch_add</code> on <code>numCurrFrontier</code>, one address in
          global memory, and the book expects contention there to be high. Appending to a frontier is chapter 12's
          unstable filter, and chapter 12 had two cures for its counter: combine the atomics of a warp into one, and
          privatise. For the first, the book says you can assume the compiler combines them. Read that as "may".{' '}
          <a href={href('ch12', '12.3')}>Section 12.3</a> showed how to do it by hand: a ballot, a population count, one
          atomic per warp, a broadcast.
        </p>
      </Translated>
      <Source section="18.6" pages="442" />
      <Today checked={tr('2026-11-20 (ch18 facts A8, D1)', '2026-11-20 (dữ kiện ch18 A8, D1)')}>
        <Translated
          vi={
            <p>
              NVIDIA chỉ nêu việc tự động gộp atomic theo warp trong một bài blog năm 2017. Backend LLVM của AMD làm điều đó cho
              các địa chỉ đồng nhất trong wavefront, như bộ đếm này. Apple không ghi nhận gì. BFS của cuGraph gộp thủ công bằng
              một ballot.
            </p>
          }
        >
          <p>
            NVIDIA states automatic warp aggregation only in a 2017 blog post. AMD's LLVM backend does it for
            wavefront-uniform addresses, such as this counter. Apple documents nothing. cuGraph's BFS aggregates by hand
            with a ballot.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Cách chữa thứ hai là mục này. Mỗi khối giữ hàng đợi và bộ đếm riêng trong bộ nhớ chia sẻ (shared memory), như
            histogram của chương 9 giữ các ngăn riêng. Các luồng chỉ tranh chấp với chính khối của mình, atomic trên bộ nhớ chia
            sẻ rẻ hơn, và lần sao chép ra frontier chung ở cuối có thể được hợp nhất (coalescing).
          </p>
        }
      >
        <p>
          The second cure is this section. Each block keeps its own queue and counter in shared memory, as chapter 9's
          histogram kept private bins. Threads contend only with their own block, shared-memory atomics are cheaper, and
          the copy to the public frontier at the end can be coalesced.
        </p>
      </Translated>
      <Source section="18.6" pages="442" />
      <Ask q={questionsS186['ch18/18.6/q-atomics']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Một hàng đợi riêng cho mỗi khối">A private queue per block</Translated></h2>
      <p>
        <Translated vi="Hình 18.15 làm việc này trong bốn giai đoạn, ngăn cách bởi ba rào chắn khối:">
          Fig. 18.15 does it in four phases, separated by three block barriers:
        </Translated>
      </p>
      <Translated
        vi={
          <ol>
            <li>
              Luồng 0 đặt bộ đếm chia sẻ <code>numCurrFrontier_s</code> về 0. Rào chắn 1.
            </li>
            <li>
              Duyệt như trước. Một luồng thắng compare-and-swap thực hiện <code>fetch_add(1)</code> trên bộ đếm riêng, với phạm
              vi khối. Dưới <code>PRIVATE_FRONTIER_CAPACITY</code>, nó lưu đỉnh vào{' '}
              <code>currFrontier_s</code>. Ngược lại hàng đợi đã đầy: nó đặt bộ đếm trở lại dung lượng và nối đỉnh thẳng vào
              frontier chung, bằng một atomic phạm vi thiết bị. Rào chắn 2.
            </li>
            <li>
              Luồng 0 đặt chỗ khoảng của khối bằng một <code>fetch_add</code> số đếm riêng lên bộ đếm chung, và giữ giá trị cũ
              trong <code>currFrontierStartIdx</code>, trong bộ nhớ chia sẻ. Rào chắn 3.
            </li>
            <li>
              Commit: luồng t sao chép các phần tử riêng t, t + <code>blockDim.x</code>, và cứ thế, tới{' '}
              <code>currFrontierStartIdx</code> cộng chỉ số của phần tử. Trong mỗi lượt các luồng liên tiếp ghi vào các địa chỉ
              liên tiếp, nên các lần ghi được hợp nhất, như trong <a href={href('ch06', '6.1')}>chương 6</a>.
            </li>
          </ol>
        }
      >
        <ol>
          <li>
            Thread 0 zeroes the shared counter <code>numCurrFrontier_s</code>. Barrier 1.
          </li>
          <li>
            Traverse as before. A thread that wins the compare-and-swap does <code>fetch_add(1)</code> on the private
            counter, with block scope. Below <code>PRIVATE_FRONTIER_CAPACITY</code> it stores the vertex in{' '}
            <code>currFrontier_s</code>. Otherwise the queue is full: it puts the counter back to the capacity and appends
            the vertex straight to the public frontier, with a device-scope atomic. Barrier 2.
          </li>
          <li>
            Thread 0 reserves the block's range with one <code>fetch_add</code> of the private count on the public
            counter, and keeps the old value in <code>currFrontierStartIdx</code>, in shared memory. Barrier 3.
          </li>
          <li>
            Commit: thread t copies private entries t, t + <code>blockDim.x</code>, and so on, to{' '}
            <code>currFrontierStartIdx</code> plus the entry's index. In each pass consecutive threads write consecutive
            addresses, so the stores coalesce, as in <a href={href('ch06', '6.1')}>chapter 6</a>.
          </li>
        </ol>
      </Translated>
      <Source section="18.6" pages="443–444" figs={tr('Fig. 18.15', 'Hình 18.15')} />
      <Translated
        vi={
          <p>
            Hai bộ đếm cần hai phạm vi khác nhau: phạm vi khối cho bộ đếm trong bộ nhớ chia sẻ, mà chỉ khối đó thấy, và phạm vi
            thiết bị cho bộ đếm chung, mà mọi khối đều cập nhật. Một phạm vi hẹp hơn tập luồng chạm vào một đối tượng là một
            race condition (tranh chấp dữ liệu).
          </p>
        }
      >
        <p>
          The two counters need different scopes: block scope for the one in shared memory, which only the block sees,
          and device scope for the public one, which every block updates. A scope narrower than the set of threads that
          touch an object is a data race.
        </p>
      </Translated>
      <Today
        checked={tr('2026-11-20 (ch18 facts, “Already settled”: ch09 A2)', '2026-11-20 (dữ kiện ch18, mục “Already settled”: ch09 A2)')}
      >
        <Translated
          vi={
            <p>
              <code>cuda::thread_scope_block</code> và <code>thread_scope_device</code> là tên của libcu++; HIP có cùng những
              thứ này qua libhipcxx. Các phép đọc-sửa-ghi atomic của MSL không nhận phạm vi.
            </p>
          }
        >
          <p>
            <code>cuda::thread_scope_block</code> and <code>thread_scope_device</code> are libcu++'s names; HIP gets the
            same through libhipcxx. MSL's atomic read-modify-writes take no scope.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Kết quả, vẽ trong Hình 18.16: các phần tử của mỗi khối nằm trong một khoảng liên tục của frontier chung, nhưng các
            khoảng theo thứ tự các khối đặt chỗ, không theo thứ tự khối, vì các atomic trên một địa chỉ được phục vụ không theo
            thứ tự cố định nào. Các phần tử tràn rơi vào bất cứ đâu mà atomic của chính chúng đặt.
          </p>
        }
      >
        <p>
          The result, drawn in Fig. 18.16: each block's entries sit in one contiguous range of the public frontier, but
          the ranges are in the order the blocks reserved, not block order, because atomics on one address are served in
          no fixed order. Overflow entries land wherever their own atomic put them.
        </p>
      </Translated>
      <Source section="18.6" pages="444" figs={tr('Fig. 18.16', 'Hình 18.16')} />
      <Ask q={questionsS186['ch18/18.6/q-two-runs']} />
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy một mức của kernel đã sửa trên 1 đến 4 khối. Sáu điều cần tìm. Bắt đầu ở dung lượng 8 trên
            mức fan-out và đọc bảng atomic; rồi giảm dung lượng cho đến khi các khối tràn, và tìm ra lúc tràn bắt đầu. Đổi thứ
            tự khối và so sánh frontier chung. Chọn lần ghi thường của sách: mọi thứ tự đều đúng, nhưng bảng race vẫn báo nó.
            Bỏ chọn từng rào chắn và tìm một thứ tự làm hỏng: rào chắn 2 theo thứ tự xuôi, rào chắn 3 theo thứ tự ngược, và rào
            chắn 1 chỉ khi có giá trị sót lại nhỏ (giá trị lấp 0xCD của trình giả lập lớn hơn mọi dung lượng rất nhiều, nên các
            lần nối sớm bị tràn và không có gì bị mất). Commit bằng một <code>if</code> và cho một khối nhiều phần tử hơn số
            luồng. Tắt phép kiểm tra tràn.
          </p>
        }
      >
        <p>
          The stepper runs one level of the corrected kernel on 1 to 4 blocks. Six things to find. Start at capacity 8 on
          the fan-out level and read the atomics panel; then drop the capacity until blocks overflow, and work out when
          overflow starts. Change the block order and compare the public frontier. Choose the book's plain store: every
          order is right, and the race panel still reports it. Untick each barrier and find an order that fails: barrier 2
          in order, barrier 3 reversed, and barrier 1 only with the small leftover (the emulator's 0xCD fill is far above
          any capacity, so early appends overflow and nothing is lost). Commit with an <code>if</code> and give a block
          more entries than threads. Turn off the overflow check.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một bước là một câu lệnh của một luồng, và các khối chạy lần lượt, như trên trình giả lập của khóa học. Không có
              thời gian, warp hay bộ nhớ đệm (cache). Phép kiểm tra của bảng race là loại mà ThreadSanitizer thực hiện; nó bỏ
              qua phạm vi, như trình giả lập.
            </p>
          }
        >
          <p>
            One step is one statement of one thread, and blocks run one after another, as on the course's emulator. No
            time, warps or caches. The race panel's check is the kind ThreadSanitizer makes; it ignores scopes, as the
            emulator does.
          </p>
        </Translated>
      </Simplification>
      <PrivateQueueStepper />
      <Ask q={questionsS186['ch18/18.6/q-overflow']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS186['ch18/18.6/q-overflow-counter']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS186['ch18/18.6/q-commit-if']} label={tr('Bug hunt', 'Săn lỗi')} />

      <h2><Translated vi="Dòng có race duy nhất">The one racy line</Translated></h2>
      <Translated
        vi={
          <p>
            Dòng 29 của Hình 18.15, <code>numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;</code>, là một lần ghi thường vào một
            bộ đếm mà các luồng khác của khối có thể đang cập nhật bằng <code>fetch_add</code> vào đúng lúc đó. Sách gọi việc
            này là khôi phục bộ đếm. Theo mô hình bộ nhớ mà chính sách viện dẫn cho các race vô hại ở pp. 433–434, một truy cập
            thường xung đột với một truy cập atomic đồng thời là một race condition, và một race condition là hành vi không xác
            định. Hình 18.17 lặp lại dòng này (dòng 35 của nó).
          </p>
        }
      >
        <p>
          Line 29 of Fig. 18.15, <code>numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;</code>, is a plain store to a
          counter that other threads of the block may be updating with <code>fetch_add</code> at that moment. The book
          calls it restoring the counter. Under the memory model the book itself invokes for the benign races of pp.
          433–434, a plain access that conflicts with a concurrent atomic one is a data race, and a data race is
          undefined behaviour. Fig. 18.17 repeats the line (its line 35).
        </p>
      </Translated>
      <Source section="18.6" pages="444" />
      <Today checked={tr('2026-11-20 (ch18 facts A7)', '2026-11-20 (dữ kiện ch18 A7)')}>
        <Translated
          vi={
            <p>
              Định nghĩa của NVIDIA: hai hành động xung đột có thể đồng thời, ít nhất một không phải atomic, là một race
              condition, và mọi race condition đều không xác định. Sách không bàn về dòng này.
            </p>
          }
        >
          <p>
            NVIDIA's definition: two potentially concurrent conflicting actions, at least one not atomic, are a data
            race, and any data race is undefined. The book does not discuss this line.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Có hai cách sửa. Ghi qua cùng tham chiếu, <code>numCurrFrontier_s_ref</code>, bằng một <code>store</code>{' '}
            relaxed, vốn là atomic; hoặc bỏ lần ghi và để luồng 0 kẹp số đếm sau rào chắn 2, khi không ai khác chạm vào nó. Cách
            nào thì bộ đếm mà lần đặt chỗ đọc cũng là min(N, capacity) cho N lần phát hiện, với mọi cách xen kẽ.
          </p>
        }
      >
        <p>
          There are two fixes. Store through the same reference, <code>numCurrFrontier_s_ref</code>, with a relaxed{' '}
          <code>store</code>, which is atomic; or drop the store and let thread 0 clamp the count after barrier 2, when
          nobody else touches it. Either way the counter that
          the reservation reads is min(N, capacity) for N discoveries, in every interleaving.
        </p>
      </Translated>
      <Beyond title={tr('Our check', 'Kiểm tra của chúng tôi')}>
        <Translated
          vi={
            <p>
              Khẳng định min(N, capacity) đến từ việc liệt kê mọi cách xen kẽ cho tối đa 5 luồng và dung lượng 4, không phải từ
              một nhà cung cấp. Mô hình của trình chạy từng bước lặp lại phép liệt kê này trong các unit test của nó.
            </p>
          }
        >
          <p>
            The min(N, capacity) claim comes from enumerating every interleaving for up to 5 threads and capacity 4, not
            from a vendor. The stepper's model repeats the enumeration in its unit tests.
          </p>
        </Translated>
      </Beyond>
      <ResetForms />
      <Translated
        vi={
          <p>
            Sao phải bận tâm, khi các con số vẫn ra đúng? Vì một race condition là hành vi không xác định: các bài kiểm tra đạt
            với trình biên dịch và GPU hôm nay không hứa hẹn gì về những thế hệ sau. Và các công cụ có báo nó. Backend{' '}
            <code>race</code> của khóa học, chạy các luồng của một khối cùng lúc, báo dòng này đối với{' '}
            <code>fetch_add</code> mỗi khi hai luồng cùng tràn; bảng race của trình chạy từng bước, kiểm tra mọi thứ tự nó chạy,
            thì luôn báo. Trên Metal câu hỏi này không đặt ra: bộ đếm là một <code>threadgroup atomic_uint</code>, không thể gán
            trực tiếp, nên chỉ viết được dạng atomic.
          </p>
        }
      >
        <p>
          Why bother, when the numbers come out right? Because a data race is undefined behaviour: tests that pass with
          today's compiler and GPU promise nothing about the next ones. And the tools flag it. The course's{' '}
          <code>race</code> backend, which runs a block's threads at the same time, reports this line against the{' '}
          <code>fetch_add</code> whenever two threads overflow together; the stepper's race panel, which checks every
          order it runs, always does. On Metal the question does not arise: the
          counter is a <code>threadgroup atomic_uint</code>, which cannot be assigned plainly, so only the atomic form can
          be written.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch18 facts C4)', '2026-11-20 (dữ kiện ch18 C4)')}>
        <Translated
          vi={
            <p>
              Đã biên dịch, chưa chạy, bằng toolchain Metal trên máy: lần đặt lại bộ đếm riêng bằng{' '}
              <code>atomic_store_explicit</code> trước một <code>threadgroup_barrier</code>.
            </p>
          }
        >
          <p>
            Compiled, not run, with the local Metal toolchain: the private counter reset with{' '}
            <code>atomic_store_explicit</code> before a <code>threadgroup_barrier</code>.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS186['ch18/18.6/q-line29']} />
      <Translated
        vi={
          <p>
            Thêm một điều sách bỏ qua: không kernel nào kiểm tra dung lượng của frontier chung, và lần cấp phát trên host không
            bao giờ được trình bày. Với compare-and-swap, mỗi đỉnh được nối nhiều nhất một lần mỗi quá trình tìm kiếm, nên một
            buffer <code>numVertices</code> phần tử là đủ. Cận đó là của chúng tôi, suy ra từ mã; lab đặt các word canh sau mỗi
            buffer frontier và kiểm tra chúng.
          </p>
        }
      >
        <p>
          One more thing the book leaves out: no kernel checks the public frontier's capacity, and the host allocation is
          never shown. With the compare-and-swap each vertex is appended at most once per search, so a buffer of{' '}
          <code>numVertices</code> entries is enough. That bound is ours, derived from the code; the lab puts guard words
          after each frontier buffer and checks them.
        </p>
      </Translated>
      <Source section="18.6" pages="443–444" />

      <h2><Translated vi="Cùng hàng đợi đó trong llama.cpp">The same queue in llama.cpp</Translated></h2>
      <Translated
        vi={
          <p>
            Một lớp mixture-of-experts phải gom, cho mỗi expert, các token được định tuyến tới nó: một danh sách công việc cho
            mỗi expert, dựng trên GPU, không biết trước mỗi danh sách dài bao nhiêu. Backend CUDA và HIP của llama.cpp dựng
            những danh sách đó trong <code>mm_ids_helper</code> theo đúng mẫu của trang này: một vùng lưu riêng trong bộ nhớ chia
            sẻ cho mỗi khối, một lần xả rồi tiếp tục khi nó đầy (nhánh tràn, làm theo cách khác), và một lần commit có bước nhảy.
            Có hai khác biệt đáng chú ý. Ô của mỗi phần tử đến từ thứ hạng ballot thay vì một atomic, nên danh sách giữ đúng thứ
            tự token, điều mà hàng đợi của Hình 18.16 không làm. Và mỗi khối tính điểm bắt đầu khoảng của mình từ số đếm của các
            expert có số thứ tự thấp hơn thay vì giành nó bằng một <code>fetch_add</code>.
          </p>
        }
      >
        <p>
          A mixture-of-experts layer has to gather, for each expert, the tokens routed to it: a work list per expert,
          built on the GPU, with no idea in advance how long each list is. llama.cpp's CUDA and HIP backend builds those
          lists in <code>mm_ids_helper</code> with this page's pattern: a private store in shared memory per block, a flush
          and resume when it fills (the overflow path, done differently), and a strided commit. Two differences are worth
          noticing. Each entry's slot comes from a ballot rank rather than an atomic, so the list stays in token order,
          which Fig. 18.16's queue does not. And each block computes where its range starts from the counts of
          lower-numbered experts instead of claiming it with a <code>fetch_add</code>.
        </p>
      </Translated>
      <Beyond title={tr('Bridge (ours)', 'Cầu nối (của khóa học này)')}>
        <Translated
          vi={
            <p>
              Từ <code>ggml/src/ggml-cuda/mmid.cu</code> của llama.cpp, đọc ngày 2026-11-20 (dữ kiện D4). Không có trong sách.
            </p>
          }
        >
          <p>
            From llama.cpp's <code>ggml/src/ggml-cuda/mmid.cu</code>, read on 2026-11-20 (facts D4). Not in the book.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Một lần khởi chạy cho mọi mức">One launch for every level</Translated></h2>
      <Translated
        vi={
          <p>
            Mọi kernel cho đến giờ chạy một lần cho mỗi mức. Giữa các mức host chờ, sao chép kích thước frontier về, rồi khởi
            chạy lại. Nhận định của sách: với frontier lớn, như trong mạng xã hội, chi phí phụ đó không đáng kể; với frontier
            nhỏ, như trong mạng đường bộ, nó có thể chiếm phần lớn.
          </p>
        }
      >
        <p>
          Every kernel so far runs once per level. Between levels the host waits, copies the frontier size back, and
          launches again. The book's judgement: with large frontiers, as in social networks, that overhead is
          negligible; with small ones, as in road networks, it may dominate.
        </p>
      </Translated>
      <Source section="18.7" pages="445" />
      <Predict
        id="ch18/18.6/p-grid-levels"
        prompt={
          <Translated
            vi={<>Một đồ thị giống mạng đường bộ: một lưới 1000 × 1000, mỗi đỉnh nối với 4 đỉnh bên cạnh, tìm kiếm từ một góc. Phép tìm kiếm có bao nhiêu mức sau gốc?</>}
          >
            <>A road-like graph: a 1000 × 1000 grid, each vertex linked to its 4 neighbours, searched from a corner. How many levels does the search have after the root?</>
          </Translated>
        }
        unit={tr('levels', 'mức')}
      >
        {() => (
          <p data-reveal>
            <Translated
              vi={
                <>
                  {fmt(GRID.levels)} mức: mức k là đường chéo thứ k, và góc xa nhất cách 999 + 999 bước. Vòng lặp host theo từng
                  mức thực hiện {fmt(GRID.launches)} lần khởi chạy, lần cuối thấy frontier trống, và không frontier nào chứa quá{' '}
                  {fmt(GRID.maxFrontier)} trong {fmt(GRID.vertices)} đỉnh của lưới. Mỗi lần khởi chạy trả cùng một chi phí cố
                  định. Hàng "floor" của <a href={href('ch18', 'labs')}>lab</a> đo chi phí đó trên thiết bị của bạn.
                </>
              }
            >
              {fmt(GRID.levels)} levels: level k is the k-th diagonal, and the far corner is 999 + 999 hops away. The
              per-level host loop makes {fmt(GRID.launches)} launches, the last one finding the frontier empty, and no
              frontier ever holds more than {fmt(GRID.maxFrontier)} of the grid's {fmt(GRID.vertices)} vertices. Each
              launch pays the same fixed cost. The <a href={href('ch18', 'labs')}>lab's</a> "floor" row measures that cost
              on your device.
            </Translated>
          </p>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Một lần khởi chạy cho cả quá trình tìm kiếm cần một rào chắn trên toàn lưới giữa các mức. Rào chắn đó có một vướng
            mắc: một lưới có thể có nhiều khối hơn số khối thường trú được cùng lúc, và phần cứng đưa khối vào khi các khối khác
            kết thúc (<a href={href('ch04', '4.2')}>chương 4</a>). Một khối thường trú chờ ở rào chắn lưới cho một khối chưa bắt
            đầu sẽ không bao giờ kết thúc, và khối chưa bắt đầu chờ một chỗ không bao giờ được giải phóng: deadlock. Vì vậy mọi
            khối phải thường trú, nghĩa là không khởi chạy nhiều hơn mức thiết bị chứa được. Phép scan một lượt của chương 11 thì
            tránh vấn đề theo cách khác: một khối chỉ chờ các khối trước, và lấy chỉ số khi nó bắt đầu, nên nó không bao giờ chờ
            một khối chưa được lập lịch (<a href={href('ch11', '11.9')}>mục 11.9</a>).
          </p>
        }
      >
        <p>
          One launch for the whole search needs a barrier across the grid between levels. That barrier has a catch: a
          grid may have more blocks than can be resident at once, and the hardware brings blocks in as others finish (
          <a href={href('ch04', '4.2')}>chapter 4</a>). A resident block waiting at a grid barrier for a block that has not
          started never finishes, and the block that has not started waits for a slot that is never freed: a deadlock.
          So every block must be resident, which means launching no more than the device can hold. Chapter 11's
          single-pass scan avoided the problem instead: a block waits only for earlier blocks, and takes its index when
          it starts, so it never waits for one that has not been scheduled (<a href={href('ch11', '11.9')}>section
          11.9</a>).
        </p>
      </Translated>
      <Source section="18.7" pages="445" />
      <Aside>
        <Translated vi={<p>Lỗi chính tả ở p. 445: "thread blocks that are not schedule" ý là scheduled.</p>}>
          <p>Typo on p. 445: "thread blocks that are not schedule" means scheduled.</p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Sách định kích thước lưới bằng số SM nhân số khối mỗi SM, lấy từ{' '}
            <code>cudaOccupancyMaxActiveBlocksPerMultiprocessor</code> và <code>multiProcessorCount</code> của thiết bị, và khởi
            chạy qua <code>cudaLaunchCooperativeKernel</code> với các đối số trong một mảng <code>void*</code>, vì cú pháp{' '}
            <code>&lt;&lt;&lt; &gt;&gt;&gt;</code> không thể yêu cầu một cooperative launch.
          </p>
        }
      >
        <p>
          The book sizes the grid as SMs times blocks per SM, from{' '}
          <code>cudaOccupancyMaxActiveBlocksPerMultiprocessor</code> and the device's <code>multiProcessorCount</code>,
          and launches through <code>cudaLaunchCooperativeKernel</code> with the arguments in a <code>void*</code> array,
          because the <code>&lt;&lt;&lt; &gt;&gt;&gt;</code> syntax cannot ask for a cooperative launch.
        </p>
      </Translated>
      <Source section="18.7" pages="445–446" />
      <Today checked={tr('2026-11-20 (ch18 facts A1, A2)', '2026-11-20 (dữ kiện ch18 A1, A2)')}>
        <Translated
          vi={
            <p>
              NVIDIA đưa ra cùng công thức đó, và runtime kiểm tra đúng tích số đó. Lời gọi bốn đối số là CUDA hợp lệ: một
              overload C++ đặt mặc định cho hai đối số cuối.
            </p>
          }
        >
          <p>
            NVIDIA prints the same recipe, and the runtime checks exactly that product. The four-argument call is valid
            CUDA: a C++ overload defaults the last two.
          </p>
        </Translated>
      </Today>
      <LaunchCode />
      <Translated
        vi={
          <p>
            Bên trong kernel, <code>this_grid()</code> trả về lưới dưới dạng một nhóm. Giờ số luồng là cố định, nên một frontier
            có thể có nhiều phần tử hơn số luồng: mỗi luồng bắt đầu ở <code>grid.thread_rank()</code> và bước theo{' '}
            <code>grid.num_threads()</code>. Phần còn lại của một mức là Hình 18.15, gồm cả hàng đợi riêng. Trên CUDA cả hai lời
            gọi đều trả về <code>unsigned long long</code>; <code>unsigned int</code> của sách thu hẹp chúng, điều vô hại với mọi
            lưới đủ nhỏ để thường trú.
          </p>
        }
      >
        <p>
          Inside the kernel, <code>this_grid()</code> returns the grid as a group. The thread count is now fixed, so a
          frontier may have more elements than threads: each thread starts at <code>grid.thread_rank()</code> and steps by{' '}
          <code>grid.num_threads()</code>. The rest of a level is Fig. 18.15, private queue and all. On CUDA both calls
          return <code>unsigned long long</code>; the book's <code>unsigned int</code> narrows them, which is harmless for
          any grid small enough to be resident.
        </p>
      </Translated>
      <Source section="18.7" pages="446–447" figs={tr('Fig. 18.17', 'Hình 18.17')} />
      <Today checked={tr('2026-11-20 (ch18 facts A4)', '2026-11-20 (dữ kiện ch18 A4)')}>
        <Translated vi={<p>Các kiểu trả về lấy từ hướng dẫn của NVIDIA.</p>}>
          <p>The return types are from NVIDIA's guide.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Điều sách không nói: dưới API cooperative, một lưới quá lớn không bao giờ được khởi chạy. Lời gọi trả về{' '}
            <code>cudaErrorCooperativeLaunchTooLarge</code>. Deadlock là thứ bạn chỉ gặp rủi ro khi đi vòng qua API đó, và{' '}
            <code>grid.sync()</code> làm gì trong một kernel khởi chạy bằng <code>&lt;&lt;&lt; &gt;&gt;&gt;</code> thì không được
            tài liệu hóa: <code>grid.is_valid()</code> cho bạn biết lưới có thể đồng bộ hóa hay không. Trình giả lập của khóa học
            dừng một kernel như vậy với thông báo nêu tên cooperative launch; đó là hành vi của trình giả lập, không phải của GPU.
          </p>
        }
      >
        <p>
          What the book does not say: under the cooperative API a grid that is too large is never launched. The call
          returns <code>cudaErrorCooperativeLaunchTooLarge</code>. The deadlock is what you risk only by going around that
          API, and what <code>grid.sync()</code> does in a kernel launched with <code>&lt;&lt;&lt; &gt;&gt;&gt;</code> is
          not documented: <code>grid.is_valid()</code> tells you whether the grid can synchronise. The course's emulator
          stops such a kernel with a message naming the cooperative launch; that is the emulator's behaviour, not a GPU's.
        </p>
      </Translated>
      <Today checked={tr("2026-11-20 (ch18 facts A3; the emulator's source)", '2026-11-20 (dữ kiện ch18 A3; mã nguồn của trình giả lập)')}>
        <Translated
          vi={
            <p>
              Cooperative launch cần compute capability 6.0 trở lên. Trình giả lập từ chối một lưới cooperative có hơn 1,024
              luồng, một giới hạn của riêng nó.
            </p>
          }
        >
          <p>
            Cooperative launch needs compute capability 6.0 or later. The emulator refuses a cooperative grid of more
            than 1,024 threads, a limit of its own.
          </p>
        </Translated>
      </Today>
      <Simplification>
        <Translated
          vi={
            <p>
              Hoạt hình vẽ thời gian theo số lần khởi chạy và số mức, không bao giờ theo giây. Một lượt đi về được vẽ bằng công
              việc của 10 đỉnh trong mọi dải: một giả định cho hình vẽ, không phải một phép đo.
            </p>
          }
        >
          <p>
            The animation draws time in launches and levels, never in seconds. A round trip is drawn as the work of 10
            vertices in every strip: an assumption for the picture, not a measurement.
          </p>
        </Translated>
      </Simplification>
      <OneLaunchScene />
      <Ask q={questionsS186['ch18/18.6/q-toolarge']} />

      <h2><Translated vi="Ba rào chắn, ba nhiệm vụ">Three barriers, three jobs</Translated></h2>
      <Translated
        vi={
          <p>
            Ở cuối mỗi mức, các luồng hoán đổi frontier (Hình 18.17, dòng 65 đến 74). <code>grid.sync()</code> thứ nhất chờ đến
            khi mọi lần nối đã vào. Sau đó mọi luồng đọc <code>*numCurrFrontier</code> vào <code>numPrevFrontier</code> của
            riêng nó. Rào chắn thứ hai chờ đến khi mọi luồng đã đọc. Luồng 0 của lưới đặt bộ đếm về 0. Rào chắn thứ ba làm lần
            đặt lại hiển thị trước khi bất kỳ ai nối cho mức sau. Rồi mỗi luồng hoán đổi hai con trỏ của riêng nó. Mỗi luồng giữ
            bản sao riêng của kích thước và các con trỏ, và mọi luồng đều đọc cùng một giá trị, nên tất cả cùng rời vòng lặp.
          </p>
        }
      >
        <p>
          At the end of each level the threads swap frontiers (Fig. 18.17, lines 65 to 74). The first{' '}
          <code>grid.sync()</code> waits until every append is in. Every thread then reads <code>*numCurrFrontier</code>{' '}
          into its own <code>numPrevFrontier</code>. The second waits until everyone has read it. Grid thread 0 resets the
          counter to 0. The third makes the reset visible before anyone appends for the next level. Then each thread swaps
          its own two pointers. Each thread keeps private copies of the size and the pointers, and every thread reads the
          same value, so they all leave the loop together.
        </p>
      </Translated>
      <Source section="18.7" pages="447–448" figs={tr('Fig. 18.17', 'Hình 18.17')} />
      <Today checked={tr('2026-11-20 (ch18 facts A5)', '2026-11-20 (dữ kiện ch18 A5)')}>
        <Translated
          vi={
            <p>
              <code>sync()</code> làm các lần ghi trước đó hiển thị với cả nhóm. Hãy giữ mỗi <code>atomic_ref</code> tới bộ đếm
              bên trong <code>if</code> thực hiện việc nối, để không cái nào còn sống trong lúc đọc và đặt lại thường.
            </p>
          }
        >
          <p>
            <code>sync()</code> makes earlier writes visible to the whole group. Keep each <code>atomic_ref</code> to the
            counter inside the <code>if</code> that appends, so none is alive during the plain read and reset.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy cả quá trình tìm kiếm trong một lần khởi chạy gồm 2 hoặc 3 khối, mỗi khối 2 luồng. Bỏ chọn
            từng rào chắn một và tìm thứ tự làm hỏng nó: một cái cho một luồng số đếm chưa đầy đủ, một cái làm một luồng đọc số 0
            sau khi đặt lại và rời vòng lặp, và một cái vứt bỏ các lần nối được thực hiện trước một lần đặt lại muộn. Sau đó
            chuyển sang Bài tập 3. Tab thứ hai, "Có vừa không?", đặt một lưới G khối lên R chỗ thường trú.
          </p>
        }
      >
        <p>
          The stepper runs the whole search in one launch of 2 or 3 blocks of 2 threads. Untick one barrier at a time and
          find the order that breaks it: one gives a thread a partial count, one makes a thread read the reset 0 and leave
          the loop, and one throws away appends made before a late reset. Then switch to Exercise 3. The second tab, "Does
          it fit?", puts a grid of G blocks on R resident slots.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Hàng đợi riêng được bỏ qua: mỗi đỉnh mới được nối thẳng vào bộ đếm chung, nên một luồng chạy từ rào chắn lưới này
              đến rào chắn kế tiếp trong một bước. Kernel được gắn chặt với{' '}
              <code>labs/ch18/snippets/s186_coop_swap.cu</code> và chạy trên trình giả lập. Không có thời gian; tab thường trú đếm
              khối, nó không tính occupancy.
            </p>
          }
        >
          <p>
            The private queue is left out: each new vertex is appended straight to the public counter, so a thread runs
            from one grid barrier to the next in one step. The kernel is pinned to{' '}
            <code>labs/ch18/snippets/s186_coop_swap.cu</code> and run on the emulator. No time; the residency tab counts
            blocks, it does not compute occupancy.
          </p>
        </Translated>
      </Simplification>
      <GridBarrierStepper />
      <Ask q={questionsS186['ch18/18.6/q-barrier2']} label={tr('Bug hunt', 'Săn lỗi')} />
      <Translated
        vi={
          <p>
            Hai trong ba rào chắn tồn tại vì một bộ đếm được dùng lại. Bài tập 3 loại bỏ việc dùng lại: cấp phát một bộ đếm cho
            mỗi mức, đặt tất cả về 0 trước lần khởi chạy, nối vào bộ đếm L và đọc bộ đếm L sau một rào chắn. Sách gọi đây là một
            dạng tổng quát hơn của double buffering; mã nằm ở tab thứ ba của đoạn mã phía trên.
          </p>
        }
      >
        <p>
          Two of the three barriers exist because one counter is reused. Exercise 3 removes the reuse: allocate a counter
          per level, zero them all before the launch, append to counter L and read counter L after one barrier. The book
          calls it a more general form of double buffering; the code is in the third tab of the listing above.
        </p>
      </Translated>
      <Source section="18.7" pages="448" />
      <Ask q={questionsS186['ch18/18.6/q-ex3']} label={tr('Exercise 3, p. 450', 'Bài tập 3, p. 450')} />
      <Translated
        vi={
          <p>
            Sách nói có hai lợi ích khác. CPU rảnh trong khi phép tìm kiếm chạy. Và trạng thái có thể ở lại trong bộ nhớ chia sẻ
            và thanh ghi (register) từ mức này sang mức sau thay vì đi qua bộ nhớ toàn cục, điều mà kernel này không dùng nhưng
            một kernel gộp thì có thể. Phép scan một lượt của chương 11 gộp các giai đoạn của nó bằng một lần chờ một
            chiều vào khối trước; đây là một rào chắn đầy đủ theo cả hai chiều.
          </p>
        }
      >
        <p>
          Two other benefits, says the book. The CPU is free while the search runs. And state can stay in shared memory
          and registers from one level to the next instead of going through global memory, which this kernel does not
          use but a fused kernel can. Chapter 11's single-pass scan fused its phases with a one-way wait on the previous
          block; this is a full barrier in both directions.
        </p>
      </Translated>
      <Source section="18.7" pages="448" />

      <h2><Translated vi="Trên HIP và Metal">On HIP and Metal</Translated></h2>
      <Translated
        vi={
          <p>
            HIP có cùng cooperative groups với cùng tên, sau một header khác,{' '}
            <code>&lt;hip/hip_cooperative_groups.h&gt;</code>, nên một file <code>.cu</code> cần một <code>#if</code> quanh
            include, như llama.cpp làm. <code>hipLaunchCooperativeKernel</code> nhận đủ sáu đối số và không có giá trị mặc định,
            nên lời gọi bốn đối số của sách không biên dịch được cho HIP; hãy truyền <code>0, 0</code> cho bộ nhớ chia sẻ và
            stream trên cả hai. Chính ví dụ của AMD truyền <code>0</code> làm mảng đối số cho một kernel có tham số: đừng sao
            chép nó. Các rank của HIP là 32-bit. Runtime từ chối một lưới lớn hơn giới hạn occupancy của nó với{' '}
            <code>hipErrorCooperativeLaunchTooLarge</code>.
          </p>
        }
      >
        <p>
          HIP has the same cooperative groups with the same names, behind a different header,{' '}
          <code>&lt;hip/hip_cooperative_groups.h&gt;</code>, so one <code>.cu</code> file needs an <code>#if</code> around
          the include, as llama.cpp has. <code>hipLaunchCooperativeKernel</code> takes all six arguments and has no
          defaults, so the book's four-argument call does not compile for HIP; pass <code>0, 0</code> for shared memory
          and stream on both. AMD's own example passes <code>0</code> as the argument array of a kernel with parameters:
          do not copy it. HIP's ranks are 32-bit. The runtime refuses a grid larger than its occupancy limit with{' '}
          <code>hipErrorCooperativeLaunchTooLarge</code>.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. Các dữ kiện về HIP và Metal lấy từ file dữ kiện (B1 đến B3, C1 đến C3) và chưa được chạy trên
              phần cứng AMD hay Apple. Việc HIP kiểm tra cùng tích số như CUDA là cách chúng tôi đọc mã nguồn runtime của nó.
              AMD không nêu bảo đảm bộ nhớ nào cho <code>sync()</code>: unverified.
            </p>
          }
        >
          <p>
            The book uses CUDA only. HIP and Metal facts are from the facts file (B1 to B3, C1 to C3) and were not run on
            AMD or Apple hardware. That HIP checks the same product as CUDA is our reading of its runtime source. AMD
            states no memory guarantee for <code>sync()</code>: unverified.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Metal không có rào chắn lưới, không có cooperative launch và không có truy vấn occupancy, nên lần khởi chạy duy nhất
            không có dạng Metal. Thứ mang sang được là mỗi mức một dispatch. Trên một <code>MTLCommandQueue</code> có tài nguyên
            được theo dõi và một compute encoder thường, các dispatch chạy theo thứ tự và mỗi dispatch thấy các lần ghi của
            dispatch trước; trên <code>MTL4CommandQueue</code>, vốn không theo dõi hazard nào, điều đó không đúng. Và Metal làm
            được một việc mà CUDA và HIP không làm được dưới dạng một lần khởi chạy:{' '}
            <code>dispatchThreadgroups(indirectBuffer:...)</code> lấy kích thước của mức sau từ một buffer do GPU ghi, nên CPU
            không bao giờ đọc nó. Số threadgroup bằng 0 không được tài liệu hóa; hãy ghi ít nhất 1 và return sớm khi frontier
            trống.
          </p>
        }
      >
        <p>
          Metal has no grid barrier, no cooperative launch and no occupancy query, so the single launch has no Metal form.
          What carries over is one dispatch per level. On an <code>MTLCommandQueue</code> with tracked resources and a
          plain compute encoder, the dispatches run in order and each sees the previous one's writes; on{' '}
          <code>MTL4CommandQueue</code>, which tracks no hazards, that does not hold. And Metal can do something CUDA and
          HIP cannot do as a launch: <code>dispatchThreadgroups(indirectBuffer:...)</code> takes the next level's size from
          a buffer the GPU wrote, so the CPU never reads it. A threadgroup count of 0 is not documented; write at least 1
          and return early on an empty frontier.
        </p>
      </Translated>
      <Today
        checked={tr(
          "2026-11-20 (ch18 facts C1 to C3, and the course's Metal runtime)",
          '2026-11-20 (dữ kiện ch18 C1 đến C3, và runtime Metal của khóa học)',
        )}
      >
        <Translated
          vi={
            <p>
              Runtime Metal của khóa học dùng một <code>MTLCommandQueue</code> và một compute encoder thường, nên các lab có hành
              vi có thứ tự và được theo dõi hazard.
            </p>
          }
        >
          <p>
            The course's Metal runtime uses an <code>MTLCommandQueue</code> and a plain compute encoder, so the labs get
            the ordered, hazard-tracked behaviour.
          </p>
        </Translated>
      </Today>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th><Translated vi="Tính năng">Feature</Translated></th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="Rào chắn trên toàn lưới">Grid-wide barrier</Translated></td>
            <td>
              <Translated vi={<><code>this_grid().sync()</code> dưới một cooperative launch</>}>
                <code>this_grid().sync()</code> under a cooperative launch
              </Translated>
            </td>
            <td>
              <Translated vi={<>cùng tên; header <code>&lt;hip/hip_cooperative_groups.h&gt;</code></>}>
                same names; header <code>&lt;hip/hip_cooperative_groups.h&gt;</code>
              </Translated>
            </td>
            <td><Translated vi="không có; mỗi mức một dispatch">none; one dispatch per level</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Cooperative launch">Cooperative launch</Translated></td>
            <td>
              <Translated vi={<><code>cudaLaunchCooperativeKernel</code>; hai đối số cuối mặc định là 0</>}>
                <code>cudaLaunchCooperativeKernel</code>; last two arguments default to 0
              </Translated>
            </td>
            <td>
              <Translated vi={<><code>hipLaunchCooperativeKernel</code>, bắt buộc đủ sáu</>}>
                <code>hipLaunchCooperativeKernel</code>, all six required
              </Translated>
            </td>
            <td><Translated vi="không có">none</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Lưới quá lớn">Grid too large</Translated></td>
            <td>
              <Translated vi={<>bị từ chối: <code>cudaErrorCooperativeLaunchTooLarge</code></>}>
                refused: <code>cudaErrorCooperativeLaunchTooLarge</code>
              </Translated>
            </td>
            <td>
              <Translated vi={<>bị từ chối: <code>hipErrorCooperativeLaunchTooLarge</code></>}>
                refused: <code>hipErrorCooperativeLaunchTooLarge</code>
              </Translated>
            </td>
            <td><Translated vi="không áp dụng">not applicable</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Định kích thước một lưới thường trú">Sizing a resident grid</Translated></td>
            <td>
              <Translated vi={<><code>multiProcessorCount</code> × truy vấn occupancy</>}>
                <code>multiProcessorCount</code> × occupancy query
              </Translated>
            </td>
            <td>
              <Translated vi="cùng tích số (CU, hoặc WGP, được điều chỉnh trong runtime)">
                same product (CUs, or WGPs, adjusted in the runtime)
              </Translated>
            </td>
            <td><Translated vi="không có số lõi, không có truy vấn occupancy">no core count, no occupancy query</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Rank và kích thước">Rank and size</Translated></td>
            <td>
              <Translated vi={<><code>thread_rank()</code>, <code>num_threads()</code> 64-bit</>}>
                64-bit <code>thread_rank()</code>, <code>num_threads()</code>
              </Translated>
            </td>
            <td>32-bit{/* i18n-ignore: a width */}</td>
            <td><Translated vi="không có">none</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Khả năng hiển thị qua rào chắn">Visibility across the barrier</Translated></td>
            <td>
              <Translated vi={<>được tài liệu hóa cho <code>sync()</code></>}>
                documented for <code>sync()</code>
              </Translated>
            </td>
            <td><Translated vi="không được nêu">not stated</Translated></td>
            <td>
              <Translated vi={<>tại ranh giới dispatch: có thứ tự, hazard được theo dõi trên <code>MTLCommandQueue</code></>}>
                at the dispatch boundary: ordered, tracked hazards on <code>MTLCommandQueue</code>
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Kích thước mức sau mà không cần CPU đọc">Next level's size without a CPU read</Translated></td>
            <td>
              <Translated vi="node điều kiện WHILE của graph (CUDA 12.3 trở lên)">graph conditional WHILE node (CUDA 12.3 and later)</Translated>
            </td>
            <td><Translated vi="không có">none</Translated></td>
            <td><code>dispatchThreadgroups(indirectBuffer:...)</code></td>
          </tr>
          <tr>
            <td><Translated vi="Đặt lại bộ đếm riêng (dòng 29)">Private counter reset (line 29)</Translated></td>
            <td>
              <Translated vi="ghi atomic hoặc kẹp giá trị; ghi thường là race">atomic store or clamp; the plain store is a race</Translated>
            </td>
            <td><Translated vi="như trên">same</Translated></td>
            <td><Translated vi="chỉ viết được lần ghi atomic">only an atomic store can be written</Translated></td>
          </tr>
        </tbody>
      </table>
      <Ask q={questionsS186['ch18/18.6/q-hip-launch']} />
      <Ask q={questionsS186['ch18/18.6/q-metal-plan']} />

      <h2><Translated vi="Những cách khác để giảm số lần khởi chạy">Other ways to cut launches</Translated></h2>
      <Translated
        vi={
          <p>
            Một cooperative kernel không phải câu trả lời duy nhất cho một vòng lặp dài gồm các lần khởi chạy nhỏ, và trong
            llama.cpp nó là ngoại lệ. Sinh token có hình dạng frontier nhỏ (cách chúng tôi đọc): nhiều kernel ngắn cho mỗi
            token, mỗi kernel do host khởi chạy. CUDA graph ghi lại nhiều lần khởi chạy và phát lại chúng với chi phí của một
            lần. Một node điều kiện WHILE (CUDA 12.3 trở lên) còn lặp ngay trên device: một kernel kết thúc vòng lặp bằng{' '}
            <code>cudaGraphSetConditional</code>, điều sẽ loại bỏ lần đọc về của host trong BFS, và cooperative launch được
            phép trong thân vòng lặp mà không cần MPS. Programmatic dependent launch (compute capability 9.0 trở lên) cho phép
            kernel sau bắt đầu trước khi kernel trước kết thúc. HIP có graph, nhưng không có hai thứ còn lại.
          </p>
        }
      >
        <p>
          A cooperative kernel is not the only answer to a long loop of small launches, and in llama.cpp it is the
          exception. Token generation has the small-frontier shape (our reading): many short kernels per token, each
          launched by the host.
          CUDA graphs record many launches and replay them for the cost of one. A conditional WHILE node (CUDA 12.3 and
          later) even loops on the device: a kernel ends the loop with <code>cudaGraphSetConditional</code>, which would
          remove the BFS host's read-back, and cooperative launches are allowed in the body without MPS. Programmatic
          dependent launch (compute capability 9.0 and later) lets the next kernel start before the last one ends. HIP has
          graphs, but neither of the other two.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch18 facts A10, B5)', '2026-11-20 (dữ kiện ch18 A10, B5)')}>
        <Translated vi={<p>Không điều nào trong số này có trong sách.</p>}>
          <p>None of this is in the book.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Các engine làm gì: llama.cpp dùng CUDA graph và HIP graph, và programmatic dependent launch chỉ trên CUDA. Trên
            Metal, vốn không có API graph, nó dùng concurrent encoder với rào chắn bộ nhớ và mã hóa command buffer trên nhiều
            luồng. Cooperative kernel duy nhất tìm thấy trong backend CUDA và HIP của nó (tìm ngày 2026-11-20) là một softmax cho
            các hàng rất dài, mỗi SM một khối, khởi chạy với sáu đối số. CUDA graph của PyTorch cấm kiểu đọc{' '}
            <code>.item()</code> mà vòng lặp host của BFS thực hiện, trừ khi qua <code>torch.cond</code>.
          </p>
        }
      >
        <p>
          What the engines do: llama.cpp uses CUDA and HIP graphs, and programmatic dependent launch on CUDA only. On
          Metal, which has no graph API, it uses concurrent encoders with memory barriers and encodes command buffers on
          several threads. The one cooperative kernel found in its CUDA and HIP backend (searched 2026-11-20) is a softmax
          for very long rows, one block per SM, launched with six arguments. PyTorch's CUDA graphs forbid the <code>.item()</code>-style read a BFS host loop makes, except
          through <code>torch.cond</code>.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch18 facts D2, D3, D5)', '2026-11-20 (dữ kiện ch18 D2, D3, D5)')}>
        <Translated vi={<p>Đọc trong mã nguồn của llama.cpp và tài liệu của PyTorch 2.15.</p>}>
          <p>Read in llama.cpp's source and PyTorch 2.15's docs.</p>
        </Translated>
      </Today>

      <h2><Translated vi="Các hub, và phần còn lại của §18.8">Hubs, and the rest of §18.8</Translated></h2>
      <Translated
        vi={
          <p>
            Trong một kernel lấy đỉnh làm trung tâm, công việc của một luồng là bậc của đỉnh nó phụ trách. Trong một đồ thị mạng
            xã hội, vài đỉnh có bậc lớn hơn phần còn lại nhiều bậc độ lớn, và luồng của chúng giữ chân cả lưới. Một cách chữa là
            cách ánh xạ lấy cạnh làm trung tâm của §18.4. Cách kia là chia nhóm theo bậc: sắp frontier vào các nhóm theo bậc và
            chạy mỗi nhóm với một nhóm luồng có kích thước phù hợp, một luồng cho bậc nhỏ, một warp cho bậc vừa, một khối cho bậc
            lớn. Sách cho rằng ba tầng này là của Merrill, Garland và Grimshaw (2012).
          </p>
        }
      >
        <p>
          In a vertex-centric kernel a thread's work is its vertex's degree. In a social graph a few vertices have degrees
          orders of magnitude above the rest, and their threads hold up the whole grid. One remedy is the edge-centric
          mapping of §18.4. The other is degree bucketing: sort the frontier into buckets by degree and run each bucket
          with a group of the right size, a thread for small degrees, a warp for medium, a block for large. The book
          attributes the three tiers to Merrill, Garland and Grimshaw (2012).
        </p>
      </Translated>
      <Source section="18.8" pages="448–449" />
      <Translated
        vi={
          <p>
            Bài báo đó làm điều linh hoạt hơn: bên trong một kernel, nó giao các danh sách dài nhất cho một khối, các danh sách
            trung bình cho một warp, và trải phần còn lại lên khối bằng một tổng tiền tố (prefix sum), và nó dựng các hàng đợi
            bằng tổng tiền tố thay vì atomic. cuGraph ngày nay chia các đỉnh tại kích thước warp và tại 1,024, và chạy phân đoạn
            bậc cao của nó theo kiểu lấy cạnh làm trung tâm.
          </p>
        }
      >
        <p>
          That paper does something more dynamic: inside one kernel it gives the longest lists a block, the middle ones a
          warp, and spreads the rest over the block with a prefix sum, and it builds its queues with prefix sums rather
          than atomics. cuGraph today splits vertices at warp size and at 1,024, and runs its high-degree segment
          edge-centric.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch18 facts A11, D1)', '2026-11-20 (dữ kiện ch18 A11, D1)')}>
        <Translated vi={<p>Đọc trong bài báo năm 2012 và trong mã nguồn của cuGraph.</p>}>
          <p>Read in the 2012 paper and in cuGraph's source.</p>
        </Translated>
      </Today>
      <Ask q={questionsS186['ch18/18.6/q-hubs']} />
      <Translated
        vi={
          <p>
            Sách nêu tên, nhưng không phát triển, các đồ thị lớn hơn bộ nhớ GPU và việc tiền xử lý sắp xếp lại các đỉnh để tăng
            tính song song, tính cục bộ hay cân bằng tải. Cho phân tích đồ thị trong môi trường production, sách chỉ tới RAPIDS
            cuGraph, mà tài liệu giờ nằm ở docs.nvidia.com/cugraph, với <code>cugraph.bfs</code>. hipGRAPH và rocGRAPH của AMD
            đang ở giai đoạn truy cập sớm, và không tìm thấy thư viện nào của Apple. Lưu ý cái tên: "graph" trong CUDA graph,
            HIP graph và MPSGraph nghĩa là công việc GPU được ghi lại, không phải một đồ thị để duyệt.
          </p>
        }
      >
        <p>
          The book names, and does not develop, graphs larger than GPU memory and preprocessing that reorders vertices for
          parallelism, locality or load balance. For production graph analytics it points to RAPIDS cuGraph, whose
          documentation now lives at docs.nvidia.com/cugraph, with <code>cugraph.bfs</code>. AMD's hipGRAPH and rocGRAPH
          are in early access, and no Apple library was found. Mind the name: "graph" in CUDA graphs, HIP graphs and
          MPSGraph means recorded GPU work, not a graph to traverse.
        </p>
      </Translated>
      <Source section="18.8–18.9" pages="449" />
      <Today checked={tr('2026-11-20 (ch18 facts A13, B6, C6)', '2026-11-20 (dữ kiện ch18 A13, B6, C6)')}>
        <Translated vi={<p>Tình trạng thư viện tính đến ngày kiểm tra.</p>}>
          <p>Library status as of the check date.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            <a href={href('ch18', 'labs')}>Lab</a> đưa hàng đợi riêng tư hóa và cooperative kernel vào làm việc trên một lưới
            giống mạng đường bộ và một đồ thị phân phối lũy thừa, so với một mức sàn đo chi phí cố định của một mức trên thiết
            bị của bạn.
          </p>
        }
      >
        <p>
          The <a href={href('ch18', 'labs')}>lab</a> puts the privatised queue and the cooperative kernel to work on a
          road-like grid and a power-law graph, against a floor that measures the fixed cost of a level on your device.
        </p>
      </Translated>
    </>
  );
}
