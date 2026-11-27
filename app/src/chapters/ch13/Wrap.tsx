import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// Chapter 13's summary page: what to carry forward, a mixed check on situations the
// sections did not show, where the book's four exercises are worked, the PDF's apparent
// slips, what the condensed chapter left out, and the project status. No kernel is shown
// for a question here, so there is no snippet to tie.

// Exercise 4 (p. 328), computed rather than typed.
const EX4 = { m: 1_030_400, n: 608_000, per: 8, block: 1024 };
const EX4_THREADS = (EX4.m + EX4.n) / EX4.per; // 204,800
const EX4_BLOCKS = EX4_THREADS / EX4.block; // 200
// The input-split question.
const SPLIT = { m: 1000, n: 1000, threads: 4 };
const SPLIT_BUSIEST = SPLIT.m / SPLIT.threads + SPLIT.n; // 1,250
const SPLIT_EVEN = (SPLIT.m + SPLIT.n) / SPLIT.threads; // 500
// The top-k question: the lab's basic kernel with total < m + n.
const TOPK = { len: 64, k: 50, per: 8, block: 128 };
const TOPK_WORKING = Math.ceil(TOPK.k / TOPK.per); // 7

const n = (v: number) => v.toLocaleString('en-US');

export const questionsWrap: Record<string, Question> = {
  'ch13/wrap/q-input-split': {
    kind: 'numeric',
    id: 'ch13/wrap/q-input-split',
    prompt: (
      <Translated
        vi={
          <>
            Một đồng nghiệp chia một phép trộn theo đầu vào thay vì theo đầu ra. A và B mỗi mảng có {n(SPLIT.m)} phần tử, và
            mọi phần tử của B đều nhỏ hơn A[0]. Mỗi luồng trong {SPLIT.threads} luồng lấy {SPLIT.m / SPLIT.threads} phần tử
            liên tiếp của A, tìm kiếm nhị phân trong B những phần tử thuộc về khoảng giữa chúng, rồi trộn. Luồng bận nhất ghi
            bao nhiêu đầu ra?
          </>
        }
      >
        A colleague splits a merge by input instead of by output. A and B each have {n(SPLIT.m)} elements, and every
        element of B is smaller than A[0]. Each of {SPLIT.threads} threads takes {SPLIT.m / SPLIT.threads} consecutive
        elements of A, binary-searches B for the elements that belong between them, and merges. How many outputs does the
        busiest thread write?
      </Translated>
    ),
    answer: SPLIT_BUSIEST,
    unit: 'outputs',
    explain: (
      <Translated
        vi={
          <p>
            Toàn bộ B đứng trước A[0], nên luồng sở hữu A[0..{SPLIT.m / SPLIT.threads}) cũng nhận mọi phần tử của B:{' '}
            {SPLIT.m / SPLIT.threads} + {n(SPLIT.n)} = {n(SPLIT_BUSIEST)} đầu ra, trong khi ba luồng kia mỗi luồng ghi{' '}
            {SPLIT.m / SPLIT.threads}. Cả phép trộn phải chờ một luồng đó. Chia đầu ra cho mọi luồng {SPLIT_EVEN}, bất kể dữ
            liệu; chỉ các lát đầu vào thay đổi. Đó là lý do cho cách phân rã lấy đầu ra làm trước của chương.
          </p>
        }
      >
        <p>
          All of B comes before A[0], so the thread that owns A[0..{SPLIT.m / SPLIT.threads}) also gets every element of B:{' '}
          {SPLIT.m / SPLIT.threads} + {n(SPLIT.n)} = {n(SPLIT_BUSIEST)} outputs, while the other three write{' '}
          {SPLIT.m / SPLIT.threads} each. The whole merge waits for that one thread. Splitting the output gives every
          thread {SPLIT_EVEN}, whatever the data; only the input slices vary. That is the reason for the chapter's
          output-first decomposition.
        </p>
      </Translated>
    ),
    source: 'ch. 13 §13.3, pp. 306–308; ch. 22 §22.2, p. 535 (load balance); the input-split variant is ours',
    vi: {
      unit: 'đầu ra',
      source: 'ch. 13 §13.3, pp. 306–308; ch. 22 §22.2, p. 535 (cân bằng tải); biến thể chia theo đầu vào là của khóa học này',
    },
  },
  'ch13/wrap/q-descending': {
    kind: 'choice',
    id: 'ch13/wrap/q-descending',
    prompt: (
      <Translated
        vi={
          <>
            Hai danh sách top-k đến đã được sắp xếp lớn nhất trước, và bạn muốn kết quả trộn của chúng cũng lớn nhất trước,
            ổn định, hòa thì về A. Cần thay đổi gì trong <code>co_rank</code> và <code>merge_sequential</code>?
          </>
        }
      >
        Two top-k lists arrive sorted largest first, and you want their merge largest first, stable, with ties going to
        A. What has to change in <code>co_rank</code> and <code>merge_sequential</code>?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated
            vi={
              <>
                Chỉ phép so sánh của phép trộn, trở thành <code>A[i] &gt;= B[j]</code>; phép tìm giữ nguyên.
              </>
            }
          >
            <>
              Only the merge's comparison, which becomes <code>A[i] &gt;= B[j]</code>; the search stays.
            </>
          </Translated>
        ),
        why: (
          <Translated vi="Phép tìm vẫn sẽ tìm điểm chia của một phép trộn tăng dần: các phép thử của nó sẽ coi i là quá cao đúng lúc nó quá thấp.">
            The search would still look for the split of an ascending merge: its tests would call i too high exactly when
            it is too low.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Lật phép thử của phép trộn và cả hai phép thử của phép tìm; mỗi phép giữ nguyên tính chặt của nó.">
            Flip the merge test and both search tests; each keeps its strictness.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Đảo ngược cả hai danh sách, trộn chúng theo thứ tự tăng dần, rồi đảo ngược kết quả trộn.">
            Reverse both lists, merge them ascending, and reverse the merged result.
          </Translated>
        ),
        why: (
          <Translated vi="Các khóa ra đúng thứ tự và mỗi danh sách giữ thứ tự của riêng nó, vì nó bị đảo hai lần. Nhưng quy tắc hòa-về-A của phép trộn tăng dần cuối cùng bị đảo ngược, nên với khóa bằng nhau, phần tử của B giờ đứng trước. Đó không phải phép trộn ổn định bạn yêu cầu, trừ khi phép trộn tăng dần gửi phần hòa về B.">
            The keys come out in the right order and each list keeps its own order, since it is reversed twice. But the
            ascending merge's A-first tie rule ends up reversed, so on equal keys B's element now comes first. That is not
            the stable merge you asked for, unless the ascending merge sends ties to B.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Hoán đổi vai trò của A và B trong cả phép tìm lẫn phép trộn.">
            Swap the roles of A and B in both the search and the merge.
          </Translated>
        ),
        why: (
          <Translated vi="Điều đó chỉ gửi phần hòa về B. Các khóa vẫn ra nhỏ nhất trước.">
            That only sends ties to B. The keys would still come out smallest first.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong một phép trộn giảm dần, "đứng trước" nghĩa là "lớn hơn", nên mọi phép so sánh đều lật:{' '}
            <code>A[i] &gt;= B[j]</code> trong phép trộn, <code>A[i-1] &lt; B[j]</code> và <code>B[j-1] &lt;= A[i]</code>{' '}
            trong phép tìm. Quy tắc phá hòa phải giữ nguyên ở cả ba chỗ: phép trộn lấy A khi khóa bằng nhau, phép thử thứ nhất
            không coi i là quá cao khi hòa, phép thử thứ hai thì coi j là quá cao khi hòa. Đổi dấu các khóa rồi trộn tăng dần
            cho cùng kết quả. Trộn các danh sách top-k theo từng khối theo cách này là cách top-k trên Metal của llama.cpp hoạt
            động với k nhỏ.
          </p>
        }
      >
        <p>
          In a descending merge "comes first" means "is larger", so every comparison flips:{' '}
          <code>A[i] &gt;= B[j]</code> in the merge, <code>A[i-1] &lt; B[j]</code> and <code>B[j-1] &lt;= A[i]</code> in
          the search. The tie rule must stay the same in all three places: the merge takes A on equal keys, the first test
          does not call i too high on a tie, the second does call j too high on one. Negating the keys and merging
          ascending gives the same result. Merging per-block top-k lists this way is how llama.cpp's Metal top-k works for
          small k.
        </p>
      </Translated>
    ),
    source: 'ch. 13 §13.4, p. 311 (the tie rule in the tests); notes/today/ch13-facts.md D3; the descending case is ours',
    vi: {
      source: 'ch. 13 §13.4, p. 311 (quy tắc phá hòa trong các phép thử); notes/today/ch13-facts.md D3; trường hợp giảm dần là của khóa học này',
    },
  },
  'ch13/wrap/q-topk-threads': {
    kind: 'numeric',
    id: 'ch13/wrap/q-topk-threads',
    prompt: (
      <Translated
        vi={
          <>
            Kernel cơ bản của lab 13.1 trộn hai danh sách đã sắp xếp, mỗi danh sách {TOPK.len} phần tử, nhưng bạn chỉ muốn{' '}
            <code>total = {TOPK.k}</code> đầu ra đầu tiên. Với {TOPK.per} đầu ra mỗi luồng và các khối {TOPK.block} luồng, bao
            nhiêu luồng chạy một phép tìm co-rank?
          </>
        }
      >
        Lab 13.1's basic kernel merges two sorted lists of {TOPK.len} elements each, but you want only the first{' '}
        <code>total = {TOPK.k}</code> outputs. With {TOPK.per} outputs per thread and blocks of {TOPK.block} threads, how
        many threads run a co-rank search?
      </Translated>
    ),
    answer: TOPK_WORKING,
    unit: 'threads',
    explain: (
      <Translated
        vi={
          <p>
            Lần khởi chạy được định cỡ theo <code>total</code>, không theo m + n: ⌈{TOPK.k} / ({TOPK.per} × {TOPK.block})⌉ = 1
            khối. Luồng t sở hữu các đầu ra từ {TOPK.per}t, nên các luồng 0 đến {TOPK_WORKING - 1} bắt đầu dưới {TOPK.k}{' '}
            (luồng {TOPK_WORKING - 1} chỉ sở hữu các đầu ra {TOPK.per * (TOPK_WORKING - 1)} và{' '}
            {TOPK.per * (TOPK_WORKING - 1) + 1}, vì <code>k_next</code> của nó bị kẹp ở {TOPK.k}). Các luồng {TOPK_WORKING}{' '}
            đến {TOPK.block - 1} bắt đầu tại hoặc quá <code>total</code> và trả về trước khi tìm. {TOPK_WORKING} luồng, mỗi
            luồng hai phép tìm, và không gì sau <code>C[{TOPK.k - 1}]</code> từng được ghi.
          </p>
        }
      >
        <p>
          The launch is sized for <code>total</code>, not m + n: ⌈{TOPK.k} / ({TOPK.per} × {TOPK.block})⌉ = 1 block. Thread t
          owns outputs from {TOPK.per}t, so threads 0 to {TOPK_WORKING - 1} start below {TOPK.k} (thread {TOPK_WORKING - 1}{' '}
          owns only outputs {TOPK.per * (TOPK_WORKING - 1)} and {TOPK.per * (TOPK_WORKING - 1) + 1}, because its{' '}
          <code>k_next</code> is clamped to {TOPK.k}). Threads {TOPK_WORKING} to {TOPK.block - 1} start at or past{' '}
          <code>total</code> and return before searching. {TOPK_WORKING} threads, two searches each, and nothing past{' '}
          <code>C[{TOPK.k - 1}]</code> is ever written.
        </p>
      </Translated>
    ),
    source: 'lab 13.1 (the total mode is ours); ch. 13 Fig. 13.9, p. 313',
    vi: { unit: 'luồng', source: 'lab 13.1 (chế độ total là của khóa học này); ch. 13 Fig. 13.9, p. 313' },
  },
  'ch13/wrap/q-ex4': {
    kind: 'choice',
    id: 'ch13/wrap/q-ex4',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 4 của sách: các mảng {n(EX4.m)} và {n(EX4.n)} phần tử, {EX4.per} đầu ra mỗi luồng, {n(EX4.block)} luồng
            mỗi khối. Bao nhiêu luồng tìm kiếm nhị phân trên bộ nhớ toàn cục (global memory) trong kernel cơ bản (Fig. 13.9);
            và trong kernel phân ô (Figs. 13.11 to 13.13), bao nhiêu luồng tìm trên bộ nhớ toàn cục và bao nhiêu luồng tìm
            trên bộ nhớ chia sẻ (shared memory)?
          </>
        }
      >
        The book's exercise 4: arrays of {n(EX4.m)} and {n(EX4.n)} elements, {EX4.per} outputs per thread, {n(EX4.block)}{' '}
        threads per block. How many threads binary-search global memory in the basic kernel (Fig. 13.9); and in the
        tiled kernel (Figs. 13.11 to 13.13), how many search global memory and how many search shared memory?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi={`${n(EX4_THREADS)}; rồi ${n(EX4_BLOCKS)} và ${n(EX4_THREADS)}.`}>
            {`${n(EX4_THREADS)}; then ${n(EX4_BLOCKS)} and ${n(EX4_THREADS)}.`}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi={`${n(EX4_THREADS)}; rồi ${n(EX4_THREADS)} và 0.`}>{`${n(EX4_THREADS)}; then ${n(EX4_THREADS)} and 0.`}</Translated>
        ),
        why: (
          <Translated vi="Trong Fig. 13.11 chỉ luồng 0 của mỗi khối chạy co_rank cấp khối trên bộ nhớ toàn cục, và mọi luồng tìm trên các ô.">
            In Fig. 13.11 only thread 0 of each block runs the block-level co_rank on global memory, and every thread
            searches the tiles.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi={`${n(EX4.m + EX4.n)}; rồi ${n((EX4.m + EX4.n) / EX4.block)} và ${n(EX4.m + EX4.n)}.`}>
            {`${n(EX4.m + EX4.n)}; then ${n((EX4.m + EX4.n) / EX4.block)} and ${n(EX4.m + EX4.n)}.`}
          </Translated>
        ),
        why: (
          <Translated vi="Đó là mỗi đầu ra một luồng. Mỗi luồng ở đây tạo ra 8.">
            That is one thread per output. Each thread here produces 8.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi={`${n(EX4_THREADS)}; rồi ${n(EX4_BLOCKS)} và ${n(EX4_BLOCKS)}.`}>{`${n(EX4_THREADS)}; then ${n(EX4_BLOCKS)} and ${n(EX4_BLOCKS)}.`}</Translated>
        ),
        why: (
          <Translated vi="Các dòng 41 và 43 chạy trong mọi luồng, trên các ô chia sẻ, và dòng 48 cũng vậy.">
            Lines 41 and 43 run in every thread, on the shared tiles, and so does line 48.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            {n(EX4.m)} + {n(EX4.n)} = {n(EX4.m + EX4.n)} đầu ra với {EX4.per} đầu ra mỗi luồng là {n(EX4_THREADS)} luồng.
            Trong kernel cơ bản, mỗi luồng gọi <code>co_rank</code> hai lần trên bộ nhớ toàn cục. Trong kernel phân ô, các luồng
            đó tạo thành {n(EX4_BLOCKS)} khối, mỗi khối {n(EX4.block)}; chỉ luồng 0 của mỗi khối tìm trên bộ nhớ toàn cục, và
            mọi luồng tìm trên các ô chia sẻ, ba lần mỗi lần lặp nếu tính cả dòng 48. Cách đếm này giả định, như bài tập gợi ý,
            rằng 8 đầu ra mỗi luồng cố định số luồng; bài tập không cho <code>tile_size</code>. Các thư viện đi xa hơn: một
            kernel phân chia riêng thực hiện một phép tìm toàn cục cho mỗi ranh giới ô.
          </p>
        }
      >
        <p>
          {n(EX4.m)} + {n(EX4.n)} = {n(EX4.m + EX4.n)} outputs at {EX4.per} per thread is {n(EX4_THREADS)} threads. In the
          basic kernel each of them calls <code>co_rank</code> twice on global memory. In the tiled kernel those threads
          form {n(EX4_BLOCKS)} blocks of {n(EX4.block)}; only thread 0 of each searches global memory, and every thread
          searches the shared tiles, three times per iteration counting line 48. The count assumes, as the exercise
          suggests, that 8 outputs per thread fixes the number of threads; it gives no <code>tile_size</code>. The
          libraries go further: a separate partition kernel makes one global search per tile boundary.
        </p>
      </Translated>
    ),
    source: 'ch. 13 exercise 4, p. 328; Figs. 13.9, 13.11, 13.13',
    vi: { source: 'ch. 13 bài tập 4, p. 328; Figs. 13.9, 13.11, 13.13' },
  },
  'ch13/wrap/q-cache': {
    kind: 'choice',
    id: 'ch13/wrap/q-cache',
    prompt: (
      <Translated
        vi={
          <>
            Trên một thiết bị nào đó, kernel cơ bản chạy với 8 đầu ra mỗi luồng nhanh xấp xỉ như với 1, dù theo cách đếm của
            trang 13.3, mỗi lần ghi của nó chạm tới số đoạn 32 byte gấp 8 lần khi ở mức 8. Lý do khả dĩ nhất là gì?
          </>
        }
      >
        On some device the basic kernel runs about as fast at 8 outputs per thread as at 1, although by page 13.3's count
        each of its stores touches 8 times as many 32-byte segments at 8. What is the likeliest reason?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Qua tám bước, mỗi lane lấp đầy đoạn của riêng nó, và bộ nhớ đệm (cache) có thể gộp các lần ghi đó.">
            Over eight steps each lane fills its own segment, and a cache can combine those stores.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Cách đếm sai: với 8 đầu ra mỗi luồng, các lane kề nhau vẫn ghi vào các địa chỉ kề nhau.">
            The count is wrong: at 8 outputs per thread neighbouring lanes still store to neighbouring addresses.
          </Translated>
        ),
        why: (
          <Translated vi="Lane t ghi vào C[8t + s] ở bước s: cách nhau 32 byte với khóa 4 byte, mỗi lane một đoạn. Cách đếm đúng cho một lệnh.">
            Lane t stores to C[8t + s] at step s: 32 bytes apart for 4-byte keys, one segment each. The count is right
            for one instruction.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Với 8 đầu ra mỗi luồng, kernel trở nên bị giới hạn bởi tính toán, nên bộ nhớ không còn quan trọng.">
            At 8 outputs per thread the kernel became compute-bound, so memory no longer matters.
          </Translated>
        ),
        why: (
          <Translated vi="Một phép trộn thực hiện một phép so sánh mỗi đầu ra và không có phép tính số học nào trên dữ liệu; nó vẫn bị giới hạn bởi bộ nhớ và bởi các nhánh của nó.">
            A merge does one comparison per output and no arithmetic on the data; it stays limited by memory and by its
            branches.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Thiết bị hợp nhất các truy cập từ các warp khác nhau thành một yêu cầu.">
            The device coalesces accesses from different warps into one request.
          </Translated>
        ),
        why: (
          <Translated vi="Hợp nhất truy cập bộ nhớ (coalescing) gộp các truy cập của một lệnh của một warp. Gộp lưu lượng về sau là việc của bộ nhớ đệm, và điều đó không được bảo đảm.">
            Coalescing combines the accesses of one warp's instruction. Merging later traffic is what a cache does, and it
            is not guaranteed.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cách đếm theo từng lệnh hỏi một lần ghi của một warp chạm tới những đoạn nào, và nó giả định không gì ghi nhớ gì
            giữa các lệnh. Mỗi luồng đi qua lát của riêng nó theo thứ tự, nên qua tám bước mỗi lane lấp đầy hoàn toàn đoạn của
            riêng nó, và số byte phải tới bộ nhớ giống như với một đầu ra mỗi luồng; chỉ số yêu cầu riêng lẻ là khác, và một
            bộ nhớ đệm giữ dòng đó có thể hấp thụ phần lớn khác biệt. Các phép tìm mà việc gộp việc tiết kiệm được bù cho một
            phần còn lại. Dự đoán thứ nhất của lab 13.1 đo điều này trên thiết bị của bạn. Hãy dự đoán từ mô hình, rồi đo.
          </p>
        }
      >
        <p>
          The per-instruction count asks which segments one store of one warp touches, and it assumes nothing remembers
          anything between instructions. Each thread walks its own slice in order, so over eight steps each lane fills
          its own segment completely, and the bytes that must reach memory are the same as at one output per thread; only
          the number of separate requests differs, and a cache that keeps the line can absorb much of that. The searches
          that coarsening saves pay for some of the rest. Lab 13.1's first prediction measures it on your device. Predict
          from the model, then measure.
        </p>
      </Translated>
    ),
    source: 'ch. 13 §13.5, p. 314, and §13.8, p. 327; the cache reasoning is ours (page 13.3, "What it costs")',
    vi: {
      source: 'ch. 13 §13.5, p. 314, và §13.8, p. 327; lập luận về bộ nhớ đệm là của khóa học này (trang 13.3, "Nó tốn gì")',
    },
  },
};

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports);
  const lab = 'ch13/lab1_merge';
  const passing = [...backends, 'emu'].filter((b) => labStatus(reports[b], lab, 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch13/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>
        <Translated vi="Những gì cần mang theo">What to carry forward</Translated>
      </h2>
      <ul>
        <Translated
          vi={
            <li>
              Khi những đầu vào mà một luồng cần phụ thuộc vào dữ liệu, hãy chia đầu ra, không chia đầu vào. Mọi luồng có cùng
              số đầu ra, và một phép tìm tìm ra các khoảng đầu vào tạo ra chúng.
            </li>
          }
        >
          <li>
            When which inputs a thread needs depends on the data, split the output, not the input. Every thread gets the
            same number of outputs, and a search finds the input ranges that produce them.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Co-rank chính là phép tìm đó: một phép tìm kiếm nhị phân trên các cách chia i + j = k, được dẫn hướng bởi các
              phần tử ở hai bên chỗ cắt, với kiểm tra biên đứng trước các lần đọc. Nó mất khoảng log2 N bước. Các thư viện gọi
              nó là merge path hoặc merge partition.
            </li>
          }
        >
          <li>
            The co-rank is that search: a binary search over the splits i + j = k, steered by the elements on either side
            of the cut, with the bounds tests ahead of the reads. It takes about log2 N steps. The libraries call it merge
            path or merge partition.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Phép tìm và phép trộn tuần tự phải phá hòa theo cùng một cách, nếu không phép trộn mất tính ổn định. Chỉ một bài
              kiểm tra với giá trị có gắn nhãn, chẳng hạn các cặp khóa–giá trị được kiểm tra với <code>std::merge</code>, mới
              thấy được sự khác biệt.
            </li>
          }
        >
          <li>
            The search and the sequential merge must break ties the same way, or the merge stops being stable. Only a test
            with tagged values, such as key–value pairs checked against <code>std::merge</code>, can see the difference.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Mọi luồng đều trả cho các phép tìm của nó, nên hãy giao cho mỗi luồng nhiều đầu ra. Trong kernel cơ bản, điều đó
              làm các truy cập của mỗi warp giãn ra; mô hình theo từng lệnh và phép đo có thể không khớp nhau, vì bộ nhớ đệm
              (cache) thấy nhiều hơn một lệnh.
            </li>
          }
        >
          <li>
            Every thread pays for its searches, so give each thread many outputs. In the basic kernel that spreads each
            warp's accesses apart; the per-instruction model and the measurement can disagree, because caches see more
            than one instruction.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Đưa các lần đọc phụ thuộc dữ liệu qua bộ nhớ chia sẻ (shared memory): nạp vào bằng các lần nạp được hợp nhất,
              truy cập không đều ở đó. Khi một ô không thể biết trước nó sẽ dùng những đầu vào nào, hoặc nạp cho trường hợp xấu
              nhất và lãng phí một nửa (sách), hoặc tìm trước rồi nạp chính xác (mọi phép trộn phân ô mà khóa học này đã đọc:
              CUB, rocPRIM, MLX, PyTorch MPS; phép trộn Metal của llama.cpp hoàn toàn không đưa vào bộ nhớ chia sẻ).
            </li>
          }
        >
          <li>
            Stage data-dependent reads through shared memory: coalesced loads in, irregular accesses there. When a tile
            cannot know in advance which inputs it will use, either load for the worst case and waste half (the book), or
            search first and load exactly (every tiled merge read for this course: CUB, rocPRIM, MLX, PyTorch MPS;
            llama.cpp's Metal merge does not stage at all).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Đừng bao giờ lấy <code>ceil</code> của một thương số nguyên. Hãy viết <code>(a + b - 1) / b</code>.
            </li>
          }
        >
          <li>
            Never take <code>ceil</code> of an integer quotient. Write <code>(a + b - 1) / b</code>.
          </li>
        </Translated>
      </ul>
      <Source section="13.9" pages="327" />

      <h2>
        <Translated vi="Một bài kiểm tra tổng hợp">A mixed check</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Những câu này dựa trên cả chương và trên những tình huống mà các trang chưa cho thấy.
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
              Câu chia theo đầu vào, phép trộn giảm dần, lần khởi chạy top-k và câu hỏi về bộ nhớ đệm là của chúng tôi. Bài
              tập 4 là của sách.
            </p>
          }
        >
          <p>
            The input split, the descending merge, the top-k launch and the cache question are ours. Exercise 4 is the
            book's.
          </p>
        </Translated>
      </Beyond>
      <Ask q={q['ch13/wrap/q-input-split']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={q['ch13/wrap/q-descending']} />
      <Ask q={q['ch13/wrap/q-topk-threads']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={q['ch13/wrap/q-ex4']} label={tr('Book exercise 4', 'Bài tập 4 của sách')} />
      <Ask q={q['ch13/wrap/q-cache']} />

      <h2>
        <Translated vi="Các bài tập của sách">The book's exercises</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương kết thúc với bốn bài tập (pp. 327–328). Sách không in đáp án, nên đáp án trong khóa học này là của chúng
            tôi.
          </p>
        }
      >
        <p>
          The chapter ends with four exercises (pp. 327–328). The book prints no answers, so the answers in this course are
          ours.
        </p>
      </Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }} data-table="exercises">
          <thead>
            <tr>
              <th>
                <Translated vi="Bài tập">Exercise</Translated>
              </th>
              <th>
                <Translated vi="Về">About</Translated>
              </th>
              <th>
                <Translated vi="Được giải ở đâu trong khóa học">Where it is worked here</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1</td>
              <td>
                <Translated vi="Co-rank của C[8] trong ví dụ xuyên suốt">The co-ranks of C[8] in the running example</Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      trình khám phá co-rank ở <a href={href('ch13', '13.3')}>13.3</a>: đặt k = 8 và chạy phép tìm
                    </>
                  }
                >
                  the co-rank explorer in <a href={href('ch13', '13.3')}>13.3</a>: set k = 8 and run the search
                </Translated>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td>
                <Translated vi="Hoàn tất phép tìm co-rank cho luồng 2 của ví dụ ba luồng">
                  Finish the co-rank search for thread 2 of the three-thread example
                </Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      trình khám phá ở <a href={href('ch13', '13.3')}>13.3</a> tại k = 6 (nó dừng ngay), và phần 5 của hoạt
                      hình "Tìm điểm chia", cho thấy cách chia của mọi luồng
                    </>
                  }
                >
                  the explorer in <a href={href('ch13', '13.3')}>13.3</a> at k = 6 (it stops at once), and part 5 of the
                  animation "Searching for the split", which shows every thread's split
                </Translated>
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td>
                <Translated vi="Chỉ nạp những phần tử mà một lần lặp sẽ dùng, với một lời gọi co-rank thêm">
                  Load only the elements an iteration will use, with an extra co-rank call
                </Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      mục "Nạp đúng những gì một ô dùng" ở <a href={href('ch13', '13.3')}>13.3</a>; phần (b) của{' '}
                      <a href={href('ch13', 'labs')}>lab 13.1</a>, theo dạng mà các thư viện phát hành; bản phác thảo bên
                      dưới cho chính vòng lặp của sách
                    </>
                  }
                >
                  "Load exactly what a tile uses" in <a href={href('ch13', '13.3')}>13.3</a>; part (b) of{' '}
                  <a href={href('ch13', 'labs')}>lab 13.1</a>, in the shape the libraries ship; the sketch below for the
                  book's own loop
                </Translated>
              </td>
            </tr>
            <tr>
              <td>4a, 4b, 4c</td>
              <td>
                <Translated vi="Các luồng tìm kiếm nhị phân trên bộ nhớ toàn cục và bộ nhớ chia sẻ trong hai kernel">
                  Threads that binary-search global and shared memory in the two kernels
                </Translated>
              </td>
              <td>
                <Translated vi='câu hỏi "Bài tập 4 của sách" ở trên'>the "Book exercise 4" question above</Translated>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <Translated
        vi={
          <p>
            Bài tập 3 trong chính vòng lặp của sách: trước các vòng nạp, một luồng tính co-rank của điểm cuối các đầu ra của
            lần lặp này trên bộ nhớ toàn cục,{' '}
            <code>co_rank(min(C_curr + C_completed + tile_size, C_next), A, m, B, n)</code>, và chia sẻ nó qua bộ nhớ chia sẻ
            sau một rào chắn (barrier). Trừ đi <code>A_curr + A_consumed</code> cho số phần tử A mà lần lặp này cần, và phần
            còn lại của ô đến từ B. Hai số đếm đó giới hạn các vòng nạp và trở thành độ dài trong các lời gọi co-rank theo từng
            luồng, và <code>A_consumed</code> được đặt trực tiếp từ kết quả, nên dòng 48 bị bỏ. Mỗi lần lặp trả thêm một phép
            tìm trên bộ nhớ toàn cục và thêm một rào chắn, và nạp ít đi một nửa. Nếu giữ là hai mảng mỗi mảng{' '}
            <code>tile_size</code>, các ô vẫn cần chỗ cho trường hợp xấu nhất; nếu dồn vào một mảng, phần của A rồi phần của
            B như lab 13.1 làm, chúng cần một nửa chỗ.
          </p>
        }
      >
        <p>
          Exercise 3 in the book's own loop: before the load loops, one thread computes the co-rank of the end of this
          iteration's outputs on global memory, <code>co_rank(min(C_curr + C_completed + tile_size, C_next), A, m, B, n)</code>,
          and shares it through shared memory behind a barrier. Subtracting <code>A_curr + A_consumed</code> gives the
          number of A elements this iteration needs, and the rest of the tile comes from B. Those two counts bound the load
          loops and become the lengths in the per-thread co-rank calls, and <code>A_consumed</code> is set from the result
          directly, so line 48 goes. Each iteration pays one more search on global memory and one more barrier, and loads
          half as much. Kept as two arrays of <code>tile_size</code> each, the tiles still need room for the worst case;
          packed into one array, A's part and then B's as lab 13.1 does, they need half the space.
        </p>
      </Translated>
      <Source section="13.6, 13.10" pages="316, 328" />

      <h2>
        <Translated vi="Những chỗ bản PDF có vẻ sai">Where the PDF looks wrong</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bản sách mà khóa học này dựa vào có một số chỗ có vẻ nhầm trong chương 13, phần lớn nằm trong các đoạn mã buffer
            vòng mà khóa học này bỏ qua. Chúng được liệt kê để bạn không bị bối rối khi đọc sách song song. Tám mục đầu là lỗi
            về câu chữ hoặc đánh số; phần lớn các mục còn lại ảnh hưởng đến mã.
          </p>
        }
      >
        <p>
          The copy of the book this course was built from has a number of apparent slips in chapter 13, most of them in the
          circular-buffer listings that this course leaves out. They are listed so that you are not thrown by them when
          you read the book alongside. The first eight are slips of wording or numbering; most of the rest affect code.
        </p>
      </Translated>
      <ul data-list="errata">
        <Translated vi={<li>p. 306 nói phần thứ hai của hàm "is on the right Fig. 13.2"; đoạn mã chỉ có một cột.</li>}>
          <li>p. 306 says the second part of the function "is on the right Fig. 13.2"; the listing is a single column.</li>
        </Translated>
        <Translated
          vi={
            <li>
              Fig. 13.3(b) ghi trường hợp C[6] là i = 4, j = 2. Văn bản và p. 308 cho co-rank 5 và 1, đúng với những gì quy
              tắc phá hòa và Fig. 13.5 cho (<a href={href('ch13', '13.3')}>13.3</a>).
            </li>
          }
        >
          <li>
            Fig. 13.3(b) labels the C[6] case i = 4, j = 2. The text and p. 308 give co-ranks 5 and 1, which is what the
            tie rule and Fig. 13.5 give (<a href={href('ch13', '13.3')}>13.3</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 309 in các lát của luồng t là "A[i_t] to A[i_t + 1]", với + 1 nằm ngoài chỉ số dưới. Ý là từ A[i_t] đến ngay
              trước A[i_(t+1)], và tương tự với B.
            </li>
          }
        >
          <li>
            p. 309 prints thread t's slices as "A[i_t] to A[i_t + 1]", the + 1 outside the subscript. It means from A[i_t]
            up to just before A[i_(t+1)], and the same for B.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Trong Fig. 13.5 (và Fig. 13.19), chú thích trên dòng <code>j_low</code> ghi <code>i_low = max(0,k-m)</code>, và
              chú thích trên dòng delta thiếu cân bằng dấu ngoặc. Bản của khóa học sửa cả hai chú thích.
            </li>
          }
        >
          <li>
            In Fig. 13.5 (and Fig. 13.19) the comment on the <code>j_low</code> line reads <code>i_low = max(0,k-m)</code>,
            and the comment on the delta line has unbalanced parentheses. The course's copy fixes both comments.
          </li>
        </Translated>
        <Translated vi={<li>p. 311 nói "In Fig. 13.7, for iteration 0"; lần lặp 0 là Fig. 13.6.</li>}>
          <li>p. 311 says "In Fig. 13.7, for iteration 0"; iteration 0 is Fig. 13.6.</li>
        </Translated>
        <Translated vi={<li>Chú thích của Fig. 13.8 nói "thread 0"; văn bản và Figs. 13.6 và 13.7 nói về luồng 1.</li>}>
          <li>Fig. 13.8's caption says "thread 0"; the text and Figs. 13.6 and 13.7 are about thread 1.</li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 314 liệt kê các lần đọc đầu tiên của kernel cơ bản là "A[0], A[2], and B[0]". Theo đoạn mã, luồng 0 so sánh
              A[0] với B[0], luồng 1 (không có phần tử B) đọc A[2], và luồng 2 (không có phần tử A) đọc B[1].
            </li>
          }
        >
          <li>
            p. 314 lists the basic kernel's first reads as "A[0], A[2], and B[0]". By the listing, thread 0 compares A[0]
            with B[0], thread 1 (no B elements) reads A[2], and thread 2 (no A elements) reads B[1].
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 315 nói lát A của luồng 1 trong ví dụ của Fig. 13.8 nhỏ hơn nhiều so với lát B của nó; ở đó luồng 1 lấy ba
              phần tử của A và không phần tử nào của B. p. 321 dẫn kernel phân ô là "Figs. 13.12 through 13.14" (thực ra là
              Figs. 13.11 to 13.13) và nói các ô bắt đầu "inside A_S[0] and B_S[0]" (bên trong các mảng A_S và B_S); p. 322
              một lần gọi kernel phân ô là <code>merge_tile_kernel</code>, và giới thiệu hai câu lệnh modulo của kernel vòng (
              <code>A_S_start</code> và <code>B_S_start</code>) là "the if-statement below".
            </li>
          }
        >
          <li>
            p. 315 says thread 1's A slice in the Fig. 13.8 example is much smaller than its B slice; there thread 1 takes
            three elements of A and none of B. p. 321 cites the tiled kernel as "Figs. 13.12 through 13.14" (it is Figs.
            13.11 to 13.13) and says tiles start "inside A_S[0] and B_S[0]" (inside the arrays A_S and B_S); p. 322 calls
            the tiled kernel <code>merge_tile_kernel</code> once, and introduces the circular kernel's two modulo statements
            (<code>A_S_start</code> and <code>B_S_start</code>) as "the if-statement below".
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>ceil</code> của một thương số nguyên ở Fig. 13.9 dòng 03, Fig. 13.11 dòng 05 và 06, và Fig. 13.12 dòng
              21: phép chia cắt cụt trước, nên <code>ceil</code> không thay đổi gì. Với chính các con số của sách, kernel phân
              ô chạy 3 lần lặp trong khi p. 317 nói 4, và có đầu ra không được ghi (<a href={href('ch13', '13.3')}>13.3</a>).
            </li>
          }
        >
          <li>
            <code>ceil</code> of an integer quotient in Fig. 13.9 line 03, Fig. 13.11 lines 05 and 06, and Fig. 13.12 line
            21: the division truncates first, so <code>ceil</code> changes nothing. With the book's own numbers the tiled
            kernel runs 3 iterations where p. 317 says 4, and outputs go unwritten (
            <a href={href('ch13', '13.3')}>13.3</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Fig. 13.13 dòng 48 đếm <code>A_consumed</code> như thể cả hai ô đều đầy. Sách thừa nhận nó sai sau lần lặp cuối
              (p. 320). Bản giả lập của chính chúng tôi cho thấy nó cũng sai ở một lần lặp sớm hơn có ô chỉ nạp một phần; dòng
              50 của kernel vòng, với độ dài <code>min</code>, là đúng (<a href={href('ch13', '13.3')}>13.3</a>).
            </li>
          }
        >
          <li>
            Fig. 13.13 line 48 counts <code>A_consumed</code> as if both tiles were full. The book allows that it is wrong
            after the last iteration (p. 320). Our own emulation shows it also goes wrong on an earlier iteration with a
            partly loaded tile; the circular kernel's line 50, with <code>min</code> lengths, is correct (
            <a href={href('ch13', '13.3')}>13.3</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>__syncthreads()</code> thứ hai của Fig. 13.11 (dòng 16) không được giải thích. Cách đọc của chúng tôi:
              mọi luồng phải đọc xong <code>A_S[0]</code> và <code>A_S[1]</code> trước khi các lần nạp ô ghi đè chúng.
            </li>
          }
        >
          <li>
            The second <code>__syncthreads()</code> of Fig. 13.11 (line 16) is not explained. Our reading: every thread must
            have read <code>A_S[0]</code> and <code>A_S[1]</code> before the tile loads overwrite them.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 322 nói <code>A_S_consumed</code> và <code>B_S_consumed</code> bắt đầu từ 0; Fig. 13.16 cho chúng bắt đầu từ{' '}
              <code>tile_size</code>, điều mà lần nạp đầu tiên cần.
            </li>
          }
        >
          <li>
            p. 322 says <code>A_S_consumed</code> and <code>B_S_consumed</code> start at 0; Fig. 13.16 starts them at{' '}
            <code>tile_size</code>, which the first fill needs.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Fig. 13.16 dòng 37 tính ô đích của B bằng <code>A_S_consumed</code>; ý là <code>B_S_consumed</code>.
            </li>
          }
        >
          <li>
            Fig. 13.16 line 37 computes B's destination slot with <code>A_S_consumed</code>; <code>B_S_consumed</code> is
            meant.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Kernel vòng không có <code>__syncthreads()</code> nào giữa các lần nạp (Fig. 13.16, kết thúc ở dòng 39) và các
              phép tìm (Fig. 13.18, từ dòng 40), trong khi kernel phân ô có một. Cần một rào chắn vì cùng lý do.
            </li>
          }
        >
          <li>
            The circular kernel has no <code>__syncthreads()</code> between its loads (Fig. 13.16, ends at line 39) and the
            searches (Fig. 13.18, from line 40), where the tiled kernel has one. One is needed for the same reason.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Lần nạp lại của Fig. 13.16 đọc từ <code>A[A_curr + A_consumed + i + threadIdx.x]</code>, vốn nạp lại những phần
              tử vẫn còn trong buffer. Trong bản giả lập của chính chúng tôi, cộng thêm <code>tile_size − A_S_consumed</code>{' '}
              vào độ lệch nguồn và điều kiện canh làm các phép trộn đúng; Fig. 13.15(c) vẽ lần nạp lại bắt đầu sau ô trước,
              điều này khớp.
            </li>
          }
        >
          <li>
            Fig. 13.16's refill reads from <code>A[A_curr + A_consumed + i + threadIdx.x]</code>, which reloads elements
            still in the buffer. In our own emulation, adding <code>tile_size − A_S_consumed</code> to the source offset and
            the guard made the merges correct; Fig. 13.15(c) draws the refill starting after the previous tile, which
            agrees.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Fig. 13.19 tính <code>j_m_1_cir</code> từ <code>B_S_start+i-1</code>; ý là <code>j-1</code>. Vài dòng của nó và
              một dòng trong Fig. 13.20 có dấu <code>)</code> thừa, và cả hai đoạn mã đều không có số dòng.
            </li>
          }
        >
          <li>
            Fig. 13.19 computes <code>j_m_1_cir</code> from <code>B_S_start+i-1</code>; <code>j-1</code> is meant. Several
            of its lines and one in Fig. 13.20 have an unmatched <code>)</code>, and neither listing has line numbers.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Fig. 13.18 dòng 48 gọi <code>merge_sequetial_circular</code>. Fig. 13.17(b) ghi mũi tên B là "B_Start", và pp.
              323–324 viết <code>A_curr</code> và <code>B_curr</code> trong khi ý là <code>a_curr</code> và <code>b_curr</code>{' '}
              ở cấp luồng.
            </li>
          }
        >
          <li>
            Fig. 13.18 line 48 calls <code>merge_sequetial_circular</code>. Fig. 13.17(b) labels the B arrow "B_Start",
            and pp. 323–324 write <code>A_curr</code> and <code>B_curr</code> where the thread-level{' '}
            <code>a_curr</code> and <code>b_curr</code> are meant.
          </li>
        </Translated>
      </ul>
      <Translated vi={<p>Và những gì các đoạn mã để lại cho bạn, mà một bản build thật cần:</p>}>
        <p>And what the listings leave to you, which a real build needs:</p>
      </Translated>
      <ul data-list="gaps">
        <Translated
          vi={
            <li>
              <code>co_rank</code> và <code>merge_sequential</code> không mang từ khóa nào. Trên CUDA và HIP chúng cần{' '}
              <code>__device__</code>; một hàm không có từ khóa chỉ chạy trên host.
            </li>
          }
        >
          <li>
            <code>co_rank</code> and <code>merge_sequential</code> carry no qualifier. On CUDA and HIP they need{' '}
            <code>__device__</code>; an unqualified function is host-only.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Fig. 13.9 kẹp <code>k_next</code> nhưng không kẹp <code>k_curr</code>, và dựa vào cách khởi chạy để giữ các luồng
              trong C. Hãy kẹp cả hai, và không khởi chạy gì cho một phép trộn rỗng: HIP từ chối một lưới không khối nào.
            </li>
          }
        >
          <li>
            Fig. 13.9 clamps <code>k_next</code> but not <code>k_curr</code>, and relies on the launch to keep threads inside
            C. Clamp both, and launch nothing for an empty merge: HIP rejects a grid of zero blocks.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>extern __shared__ int shareAB[]</code> không có kích thước trong kernel; nó đến từ đối số khởi chạy thứ
              ba, thứ mà chương không cho thấy.
            </li>
          }
        >
          <li>
            <code>extern __shared__ int shareAB[]</code> has no size in the kernel; it comes from the third launch argument,
            which the chapter does not show.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>min</code> số nguyên khác nhau giữa các backend: CUDA có các overload trộn có dấu và không dấu trả về{' '}
              <code>unsigned</code>, header của HIP không có, và MSL muốn cả hai đối số cùng một kiểu. Các biểu thức ba ngôi
              của đoạn mã tránh được vấn đề này.
            </li>
          }
        >
          <li>
            Integer <code>min</code> differs between backends: CUDA has mixed signed and unsigned overloads that return{' '}
            <code>unsigned</code>, HIP's header has none, and MSL wants both arguments of one type. The listings' ternaries
            avoid the question.
          </li>
        </Translated>
      </ul>
      <Aside>
        <Translated
          vi={
            <p>
              Đây là cách đọc của chúng tôi về một bản PDF (notes/ch13.md, "Unclear in the PDF"), với các điểm về nhà cung
              cấp đã được kiểm tra vào 2026-11-20 (notes/today/ch13-facts.md A4, A5, B1, C1; ch02-facts). Các mục ghi là bản
              giả lập của chúng tôi là suy luận, không phải của sách. Nếu bản của bạn khác, hãy tin bản của bạn.
            </p>
          }
        >
          <p>
            These are our readings of one PDF (notes/ch13.md, "Unclear in the PDF"), with the vendor points checked on
            2026-11-20 (notes/today/ch13-facts.md A4, A5, B1, C1; ch02-facts). Items marked as our emulation are inferences,
            not the book's. If your copy differs, trust your copy.
          </p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Những gì chương này bỏ qua">What this chapter left out</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương này được rút gọn trong khóa học, nên đây là những gì không được đề cập và nơi tìm chúng. §13.7, kernel buffer vòng (pp. 321–327, Figs. 13.15 to 13.20), có một đoạn ở trang 13.3 và các chỗ nhầm của nó ở trên: không có
            phần tương tác, không có mã và không có lab, vì không thư viện hay dự án nào được đọc cho khóa học này dùng nó.
            Phần đi qua từng dòng của kernel phân ô (Figs. 13.11 to 13.13, pp. 316–320) được tóm tắt ở trang 13.3; phần ghi
            sổ của nó, <code>A_consumed</code>, <code>C_completed</code> và các điều kiện canh ở đuôi, nằm trong trình khám
            phá đã nạp so với đã dùng, và lab dạy phiên bản nạp chính xác. Hai trường hợp của Observation 1 (pp. 306–307) được
            gộp vào Observation 2, và vết chạy của phép trộn tuần tự (p. 306) được gộp vào một đoạn ở trang bắt đầu.
            Map-reduce và Hadoop (p. 305) cùng phép giao và hợp tập hợp (p. 303) được sách nhắc qua và không có ở đây.
          </p>
        }
      >
        <p>
          The chapter is condensed in this course, so here is what is not covered and where to find it. §13.7, the
          circular-buffer kernel (pp. 321–327, Figs. 13.15 to 13.20), gets one paragraph on page 13.3 and its slips above:
          no interactive, no code and no lab, because no library or project read for this course uses one. The
          line-by-line walk through the tiled kernel (Figs. 13.11 to 13.13, pp. 316–320) is summarised on page 13.3; its
          bookkeeping, <code>A_consumed</code>, <code>C_completed</code> and the tail guards, lives in the loaded-versus-used
          explorer instead, and the lab teaches the exact-load version. Observation 1's two cases (pp. 306–307) are folded
          into Observation 2, and the sequential merge's trace (p. 306) into a paragraph on the start page. Map-reduce and
          Hadoop (p. 305) and set intersection and union (p. 303) are mentioned by the book in passing and not here.
        </p>
      </Translated>
      <Translated vi={<p>Không phần nào sau này trong khóa học phụ thuộc vào những gì bị bỏ qua.</p>}>
        <p>Nothing later in the course depends on what was left out.</p>
      </Translated>

      <h2>
        <Translated vi="Dự án đến giờ">The project so far</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương này không thêm mốc dự án nào cho dự án xuyên suốt: không gì trên đường chính của một bước suy luận, các
            phép nhân ma trận, attention hay phép tính lượng tử hóa của nó, trộn hai danh sách đã sắp xếp. Dù vậy, phép trộn
            có nằm dưới việc lấy mẫu. Chế độ <code>total</code> của lab 13.1 là bước kết hợp các danh sách top-k theo từng
            khối, như backend Metal của llama.cpp làm. Module X1 xây top-k của nó bằng radix select thay vào đó (
            <a href={href('x1', 'X1.4')}>X1.4, "Top-k là một phép chọn, không phải sắp xếp"</a>), nên mốc dự án của nó không
            phụ thuộc vào lab này. Lab 13.1 là tùy chọn.{' '}
            <span data-lab131={passing.join(',') || 'none'}>
              {backends.length === 0 && !reports['emu']
                ? 'Chưa có kết quả lab nào được ghi trên máy này.'
                : passing.length
                  ? `Phiên bản của bạn qua trên ${passing.map(where).join(' và ')} trên máy này.`
                  : 'Phiên bản của bạn chưa qua trên máy này.'}
            </span>{' '}
            Xem <a href={href('ch13', 'labs')}>lab</a>.
          </p>
        }
      >
        <p>
          This chapter adds no milestone to the through-line project: nothing on the main path of an inference step, its
          matrix products, attention or quantised arithmetic, merges two sorted lists. The merge does sit under sampling,
          though. Lab 13.1's <code>total</code> mode is the step that combines per-block top-k lists, as llama.cpp's Metal
          backend does. Module X1 builds its top-k as a radix select instead (
          <a href={href('x1', 'X1.4')}>X1.4, "Top-k is a selection, not a sort"</a>), so its milestone does not depend on
          this lab. Lab 13.1 is optional.{' '}
          <span data-lab131={passing.join(',') || 'none'}>
            {backends.length === 0 && !reports['emu']
              ? 'No lab results are recorded on this machine yet.'
              : passing.length
                ? `Your own version passes on ${passing.map(where).join(' and ')} on this machine.`
                : 'Your own version has not passed on this machine yet.'}
          </span>{' '}
          See the <a href={href('ch13', 'labs')}>lab</a>.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chương 14 biến phép trộn thành phép sắp xếp. Merge sort của nó là phép trộn của chương này áp dụng hết giai đoạn
            này đến giai đoạn khác, và co-rank quyết định phần của mỗi khối trong một phép trộn bắt đầu ở đâu. Tuy vậy, phần
            lớn chương đó là radix sort, vốn hoàn toàn không so sánh khóa: mỗi lượt đặt mọi khóa bằng một scan loại trừ từ
            chương 11. Bắt đầu ở <a href={href('ch14', 'start')}>chương 14</a>.
          </p>
        }
      >
        <p>
          Chapter 14 turns the merge into a sort. Its merge sort is this chapter's merge applied stage after stage, and the
          co-rank decides where each block's share of a merge starts. Most of that chapter, though, is radix sort, which
          compares no keys at all: each pass places every key with an exclusive scan from chapter 11. Start at{' '}
          <a href={href('ch14', 'start')}>chapter 14</a>.
        </p>
      </Translated>
    </>
  );
}
