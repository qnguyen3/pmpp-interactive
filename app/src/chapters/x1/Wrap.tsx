import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, NoteGroup, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { LAB } from '../../sim/chain';

// Module X1's summary: what to carry forward, a mixed check on situations the sections did
// not show, where each of the module's tasks is worked (the book has no exercises for it),
// where the sources disagree (the module has no PDF errata), and milestone M5.

export const questionsWrap: Record<string, Question> = {
  'x1/wrap/q-fuse-select': {
    kind: 'numeric',
    id: 'x1/wrap/q-fuse-select',
    prompt: (
      <Translated
        vi={
          <>
            Bạn thử bỏ bớt lượt áp phạt. Thay vì một kernel đọc từng logit cùng số đếm của nó rồi ghi logit đã phạt trở lại,
            phép chọn của lab X1.1 áp phạt ngay khi đọc: mỗi lần trong năm lần nó đọc một logit (bốn lượt histogram và lượt
            gom) cũng đọc số đếm của token đó. Tính trên mỗi logit, bỏ qua k cặp đầu ra, thiết kế gộp di chuyển bao nhiêu
            byte?
          </>
        }
      >
      <>
        You try to save the penalty pass. Instead of one kernel that reads each logit and its count and writes the
        penalised logit back, lab X1.1's select applies the penalties on the fly: each of its five reads of a logit (four
        histogram passes and the gather) also reads that token's count. Per logit, ignoring the k output pairs, how many
        bytes does the fused design move?
      </>
      </Translated>
    ),
    answer: 40,
    unit: 'bytes',
    explain: (
      <Translated
        vi={
          <p>
            Năm lần đọc một logit 4 byte và một số đếm 4 byte: 5 × 8 = 40 byte. Hai kernel tách riêng di chuyển 12 cho lượt áp
            phạt (đọc, đọc, ghi) và 20 cho phép chọn, tổng cộng 32. Gộp có lợi khi nó loại bỏ một vòng đi về qua bộ nhớ; ở
            đây nó bỏ được một lần ghi và một lần đọc nhưng nhân số lần đọc số đếm lên năm. Hãy đếm trước khi gộp.
          </p>
        }
      >
      <p>
        Five reads of a 4-byte logit and a 4-byte count: 5 × 8 = 40 bytes. The two kernels apart move 12 for the penalty
        pass (read, read, write) and 20 for the select, 32 in all. Fusing pays when it removes a round trip through memory;
        here it removes one write and one read but multiplies the reads of the counts by five. Count before you fuse.
      </p>
      </Translated>
    ),
    source: 'ours: the byte counts of X1.3 (12 per penalty pass) and X1.4 (20 per key for the select)',
    vi: {
      unit: 'byte',
      source: 'của khóa học này: số byte của X1.3 (12 cho mỗi lượt áp phạt) và X1.4 (20 cho mỗi khóa với phép chọn)',
    },
  },
  'x1/wrap/q-near-boundary': {
    kind: 'choice',
    id: 'x1/wrap/q-near-boundary',
    prompt: (
      <Translated
        vi={
          <>
            Lab X1.3 qua trên một máy CUDA và trên Mac của bạn. Bạn ghi lại mọi token của cả hai lần chạy và thấy 3 lần rút,
            trong số nhiều nghìn, mà hai backend chọn token khác nhau. Tham chiếu CPU đã đánh dấu cả ba là gần một biên. Điều đó
            cho bạn biết gì?
          </>
        }
      >
      <>
        Lab X1.3 passes on a CUDA machine and on your Mac. You log every token of both runs and find 3 draws, out of many
        thousands, where the two backends chose different tokens. The CPU reference had flagged all three as near a
        boundary. What does that tell you?
      </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Không có gì sai: exp, thứ tự cộng và cách làm tròn khác nhau theo backend, và những lần rút đó nằm cách một biên trong phạm vi khác biệt ấy.">Nothing is wrong: exp, addition order and rounding differ by backend, and those draws sat within that difference of a boundary.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Hai backend đã rút các số ngẫu nhiên khác nhau cho các hàng đó, vì các hãng phát hành thư viện Philox khác nhau.">The two backends drew different random numbers for those rows, because the vendors ship different Philox libraries.</Translated>,
        why: <Translated vi="M5 không dùng bộ sinh nào của thư viện. Philox riêng của nó qua các vector đáp án đã biết trên từng backend, và u được so sánh từng bit.">M5 uses no library generator. Its own Philox passes the known-answer vectors on each backend, and u is compared bit for bit.</Translated>,
      },
      {
        label: <Translated vi="Một trong hai kernel có lỗi thật, một lỗi mà các bài test tình cờ không bắt được trên các hàng này.">One of the two kernels has a real defect, one that the tests did not happen to catch on these rows.</Translated>,
        why: <Translated vi="Một lỗi sẽ lộ ra ở các lần rút dứt khoát, vốn phải khớp chính xác. Ba lần này nằm trong dải mà tham chiếu đã ghi lại.">A defect shows on decisive draws, which must match exactly. These three sat inside the band the reference recorded.</Translated>,
      },
      {
        label: <Translated vi="Tham chiếu CPU mới là bên sai, vì nó tính bằng double.">The CPU reference is the one that is wrong, because it computes in double.</Translated>,
        why: <Translated vi="Double là bên gần đúng nhất trong ba. Dải tồn tại vì cả hai thiết bị đều tính bằng float, mỗi bên theo cách riêng.">Double is the closer of the three. The band exists because both devices compute in float, each in its own way.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            u là cùng các bit ở mọi nơi; trọng số tích lũy thì không. <code>expf</code> của CUDA và <code>exp</code> nhanh của
            Metal có cận sai số khác nhau, hai phép rút gọn theo khối cộng theo thứ tự khác nhau, và Metal có thể làm tròn về
            phía 0. Một u rơi vào trong phạm vi khác biệt đó quanh một biên có thể chọn bên nào cũng được, và cả hai đều là mẫu
            đúng. Đó là lý do M5 chỉ hứa cùng token cho các lần rút dứt khoát, và lý do digest chỉ bao gồm những lần đó.
          </p>
        }
      >
      <p>
        u is the same bits everywhere; the cumulative weights are not. CUDA's <code>expf</code> and Metal's fast{' '}
        <code>exp</code> have different error bounds, the two block reductions add in different orders, and Metal may
        round toward zero. A u that lands within that difference of a boundary can pick either neighbour, and both are
        correct samples. That is why M5 promises the same token only for decisive draws, and why the digest covers those
        alone.
      </p>
      </Translated>
    ),
    source: 'X1.8 and lab X1.3 README; notes/today/x1-facts.md C5; ch10-facts.md A11, C4',
    vi: { source: 'X1.8 và README của lab X1.3; notes/today/x1-facts.md C5; ch10-facts.md A11, C4' },
  },
  'x1/wrap/q-pad-vocab': {
    kind: 'choice',
    id: 'x1/wrap/q-pad-vocab',
    prompt: (
      <Translated
        vi={
          <>
            Phép chiếu đầu ra của một mô hình được đệm thêm, nên mỗi hàng logits có nhiều hơn số token của bộ từ vựng 128 cột, và
            phép nhân ma trận để lại 0.0 trong các cột đó. Bạn gọi <code>launchSampleRows</code> với <code>vocab</code> đặt
            bằng độ rộng đã đệm. Trước hết nên đặt gì vào các cột đệm, để không backend nào có thể rút trúng một cột?
          </>
        }
      >
      <>
        A model's output projection is padded, so every logits row has 128 more columns than the vocabulary has tokens,
        and the matmul leaves 0.0 in them. You call <code>launchSampleRows</code> with <code>vocab</code> set to the padded
        width. What should go into the padding columns first, so that no backend can ever draw one?
      </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Một logit âm hữu hạn lớn như −1e4: trọng số của nó underflow về 0, và một trọng số 0 không bao giờ thắng lần rút.">A large finite negative logit such as −1e4: its weight underflows to 0, and a zero weight never wins the draw.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Không gì cả: để nguyên 0.0, vì top-k và top-p sẽ loại chúng.">Nothing: leave them at 0.0, since top-k and top-p will remove them.</Translated>,
        why: <Translated vi="Logit 0 là một giá trị bình thường. Với các logit gần 0, hoặc với top-k 0, một id đệm có thể sống sót và bị rút trúng.">A logit of 0 is an ordinary value. With logits near 0, or with top-k 0, a padding id can survive and be drawn.</Translated>,
      },
      {
        label: <Translated vi="Âm vô cùng, giá trị mask mà các sampler dạng graph dùng.">Minus infinity, the mask value the graph samplers use.</Translated>,
        why: <Translated vi="Dưới chế độ fast math mặc định của Metal, hành vi của vô cùng không xác định (ch10-facts C3), nên mask này không an toàn trên mọi backend.">Under Metal's default fast math the behaviour of infinity is undefined (ch10-facts C3), so the mask is not safe on every backend.</Translated>,
      },
      {
        label: <Translated vi="−FLT_MAX, số float hữu hạn âm nhất.">−FLT_MAX, the most negative finite float.</Translated>,
        why: <Translated vi="Phép rút tính (l − l₀) / T. Với T dưới 1, −FLT_MAX − l₀ chia cho T tràn thành âm vô cùng, tức lại là vấn đề trước.">The draw computes (l − l₀) / T. At T below 1, −FLT_MAX − l₀ divided by T overflows to minus infinity, which is the previous problem again.</Translated>,
      },
      {
        label: <Translated vi="Để nguyên các logit và truyền kích thước bộ từ vựng thật làm vocab.">Leave the logits alone and pass the real vocabulary size as vocab.</Translated>,
        why: <Translated vi="vocab vừa là độ dài hàng vừa là bước nhảy (stride) giữa các hàng, nên từ hàng 1 trở đi mọi hàng sẽ bị đọc từ sai chỗ.">vocab is both the row length and the row stride, so from row 1 on every row would be read from the wrong place.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            −1e4 chia cho bất kỳ temperature nào sampler chấp nhận vẫn hữu hạn, và exp của nó nhỏ hơn rất xa số float nhỏ nhất,
            nên trọng số là 0 trên mọi backend làm tròn hợp lý. Các bài test của lab X1.3 cài đúng những logit như vậy và kiểm
            tra rằng chúng không bao giờ thắng, và điều đó đúng trên Metal và trên các trình giả lập; riêng cận độ chính xác của
            các hãng thì không hứa một số 0 chính xác. Số đếm của các id đệm là 0, nên các phép phạt không đụng đến chúng.
          </p>
        }
      >
      <p>
        −1e4 divided by any temperature the sampler accepts stays finite, and exp of it is far below the smallest float, so
        the weight is 0 on every backend that rounds sensibly. Lab X1.3's tests plant exactly such logits and check that
        they never win, and that held on Metal and on the emulators; vendor accuracy bounds alone do not promise an exact
        0. The counts of the padding ids are 0, so the penalties leave them alone.
      </p>
      </Translated>
    ),
    source: 'ours; lab X1.3 README (test 3); notes/today/ch10-facts.md C3',
    vi: { source: 'của khóa học này; README của lab X1.3 (test 3); notes/today/ch10-facts.md C3' },
  },
  'x1/wrap/q-n-samples': {
    kind: 'choice',
    id: 'x1/wrap/q-n-samples',
    prompt: (
      <Translated
        vi={
          <>
            Một yêu cầu xin bốn phần hoàn thành cho một prompt. Server của bạn cho chúng bốn hàng của một batch, seed của yêu
            cầu, và bước 0 cho mỗi hàng. Điều gì phải khác nhau giữa bốn hàng để M5 tạo ra bốn phần hoàn thành khác nhau?
          </>
        }
      >
      <>
        A request asks for four completions of one prompt. Your server gives them four rows of one batch, the request's
        seed, and step 0 for each. What has to differ between the four rows for M5 to produce four different
        completions?
      </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Id stream: cùng seed, stream, step và logits cho cùng u và cùng token ở mọi bước.">The stream id: the same seed, stream, step and logits give the same u and the same token at every step.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Không gì cả: bốn hàng chạy trong bốn khối khác nhau, nên mỗi hàng rút số ngẫu nhiên riêng.">Nothing: the four rows run in four different blocks, so each draws its own random number.</Translated>,
        why: <Translated vi="Chỉ số khối không phải đầu vào của u trong M5; đó là lựa chọn của FlashInfer. u chỉ phụ thuộc vào (seed, stream, step).">The block index is not an input to M5's u; that was FlashInfer's choice. u depends on (seed, stream, step) alone.</Translated>,
      },
      {
        label: <Translated vi="Temperature: cho mỗi hàng trong bốn hàng một T hơi khác, để phân phối của chúng khác nhau.">The temperature: give each of the four rows a slightly different T, so their distributions differ.</Translated>,
        why: <Translated vi="Việc đó thay đổi phân phối một chút. Với cùng u và trọng số gần như giống nhau, các hàng vẫn rơi vào cùng một token gần như mọi lần.">That changes the distributions a little. With the same u and nearly the same weights, the rows still land on the same token almost every time.</Translated>,
      },
      {
        label: <Translated vi="Không gì cả, miễn là top-k bằng 0, vì đường xử lý cả hàng sắp xếp từng hàng riêng.">Nothing, provided top-k is 0, because the whole-row path sorts each row on its own.</Translated>,
        why: <Translated vi="Phép sắp xếp là tất định. Các hàng giống hệt nhau được sắp xếp giống hệt nhau và cho các ứng viên giống hệt nhau.">The sort is deterministic. Identical rows sort identically and give identical candidates.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Phép rút của M5 là một hàm thuần của dữ liệu và của (seed, stream, step). Bốn hàng giống hệt nhau với cùng ba số đó
            là cùng một phép tính lặp bốn lần. Hãy cho mỗi phần hoàn thành một stream riêng: các bài test của lab X1.3 làm đúng
            như vậy, mỗi hàng một stream. Chính tính chất này giúp một phần hoàn thành tái lập được khi một yêu cầu di chuyển
            giữa các vị trí trong batch.
          </p>
        }
      >
      <p>
        M5's draw is a pure function of the data and of (seed, stream, step). Four identical rows with the same three
        numbers are the same computation four times. Give each completion its own stream: lab X1.3's tests do exactly that,
        one stream per row. The same property is what makes one completion reproducible when a request moves between batch
        slots.
      </p>
      </Translated>
    ),
    source: 'ours; X1.7 (M5’s mapping); notes/today/x1-facts.md A6',
    vi: { source: 'của khóa học này; X1.7 (ánh xạ của M5); notes/today/x1-facts.md A6' },
  },
  'x1/wrap/q-hip-popc': {
    kind: 'choice',
    id: 'x1/wrap/q-hip-popc',
    prompt: (
      <Translated
        vi={
          <>
            Bạn port phép gom của lab X1.1 sang HIP và xếp hạng các khóa được giữ bằng <code>__popc(vote &amp; lanesBelow)</code>,
            trong đó <code>vote</code> là giá trị 64-bit mà <code>__ballot_sync</code> trả về ở đó. Trên một wavefront rộng 64,
            điều gì sai?
          </>
        }
      >
      <>
        You port lab X1.1's gather to HIP and rank the kept keys with <code>__popc(vote &amp; lanesBelow)</code>, where{' '}
        <code>vote</code> is the 64-bit value <code>__ballot_sync</code> returns there. On a 64-wide wavefront, what goes
        wrong?
      </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Các lane 32 đến 63 chỉ đếm các khóa được giữ ở lane 0 đến 31, nên hai lane có thể dùng chung một vị trí và một khóa bị mất.">Lanes 32 to 63 count only the keepers in lanes 0 to 31, so two lanes can share a slot and one key is lost.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Nó không biên dịch: HIP từ chối đối số 64-bit cho __popc.">It does not compile: HIP rejects a 64-bit argument to __popc.</Translated>, why: <Translated vi="Việc cắt bớt diễn ra âm thầm: giá trị 64-bit bị chuyển thành 32 bit (ch12-facts B1).">The cut is silent: the 64-bit value is converted to 32 bits (ch12-facts B1).</Translated> },
      { label: <Translated vi="Không có gì sai: HIP trả về vote 32-bit kể cả trên target rộng 64, nên __popc thấy mọi lane.">Nothing goes wrong: HIP returns a 32-bit vote even on 64-wide targets, so __popc sees every lane.</Translated>, why: <Translated vi="Vote là 64 bit trên mọi target AMD (infra-facts B3).">The vote is 64 bits on every AMD target (infra-facts B3).</Translated> },
      {
        label: <Translated vi="Chỉ thứ tự đầu ra thay đổi; tập khóa được giữ vẫn đúng.">Only the order of the output changes; the set of kept keys stays right.</Translated>,
        why: <Translated vi="Các vị trí trùng nhau nghĩa là một khóa bị mất và một vị trí còn giữ giá trị canh (sentinel), và tổng của các warp cũng bị cắt thiếu.">Colliding slots mean a lost key and a slot left holding the sentinel, and the warp totals are cut short as well.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hạng của lane 40 phải đếm các khóa được giữ ở lane 0 đến 39. <code>__popc</code> chỉ thấy bit 0 đến 31, nên nếu lane
            32 cũng giữ, cả hai lane tính ra cùng một vị trí. Tổng của warp, <code>__popc(vote)</code>, bị cắt theo cùng cách,
            nên offset của mọi warp phía sau cũng sai. Cách sửa là cách của lab: một lane mask có kiểu và một popcount khớp với
            nó, <code>__popcll</code> trên HIP.
          </p>
        }
      >
      <p>
        Lane 40's rank should count the keepers in lanes 0 to 39. <code>__popc</code> sees only bits 0 to 31, so if lane 32
        also keeps, both lanes compute the same slot. The warp's total, <code>__popc(vote)</code>, is cut the same way, so
        every later warp's offset is wrong too. The fix is the lab's: a typed lane mask and a popcount that matches it,{' '}
        <code>__popcll</code> on HIP.
      </p>
      </Translated>
    ),
    source: 'notes/today/ch12-facts.md B1; infra-facts.md B3; lab X1.1 README',
    vi: { source: 'notes/today/ch12-facts.md B1; infra-facts.md B3; README của lab X1.1' },
  },
};

// ---------------------------------------------------------------- milestone M5
/** Every lab that supplies a kernel of M5, from chain.ts's LAB list, with its directory. */
const M5: { num: string; chapter: string; title: string; dir: string }[] = [
  { ...LAB.counts, dir: 'ch09/lab3_tokencounts' },
  { ...LAB.softmax, dir: 'ch10/lab3_softmax_argmax' },
  { ...LAB.topp, dir: 'ch11/lab6_topp' },
  { ...LAB.compact, dir: 'ch12/lab4_candidates' },
  { ...LAB.sort, dir: 'ch14/lab3_sortrows' },
  { ...LAB.radix, dir: 'x1/lab1_topk' },
  { ...LAB.filters, dir: 'x1/lab2_filters' },
  { ...LAB.m5, dir: 'x1/lab3_sampler' },
];
const where = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
/** Backends (devices first, then the emulator) on which the reader's own work of a lab passes. */
const passingOn = (reports: Reports, lab: string) =>
  [...realBackends(reports), 'emu'].filter((b) => labStatus(reports[b], lab, 'work') === 'pass');

// ---------------------------------------------------------------- the page
export function Wrap() {
  const { progress } = useProgress();
  const { reports, loading } = useReports();
  const ran = Object.keys(reports).length > 0;
  const own = passingOn(reports, 'x1/lab3_sampler');
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('x1/') && !k.includes('/p-'));
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
              Một sampler là một chuỗi các giai đoạn theo hàng: phép phạt thay đổi logits, top-k, top-p và min-p loại bỏ ứng
              viên, temperature định hình lại phần còn lại, và một lần rút chọn ra một token. Top-k chỉ phụ thuộc vào thứ tự của
              các logit; top-p và min-p phụ thuộc vào chỗ temperature được áp dụng. Các dự án không thống nhất về thứ tự, nên hãy
              nói rõ nó.
            </li>
            <li>
              Mọi giai đoạn đều bị giới hạn bởi bộ nhớ. Câu hỏi quan trọng là hàng được đọc và ghi bao nhiêu lần, và bao nhiêu
              phần của thiết bị đang bận. Với một khối mỗi hàng, batch 1 chỉ chiếm một multiprocessor.
            </li>
            <li>
              Chỉ phép phạt và top-k đọc toàn bộ từ vựng. Top-k là một phép chọn: radix select thu hẹp dần về khóa thứ k từng
              byte một và di chuyển khoảng một phần tư số byte của một phép sắp xếp đầy đủ. Mọi thứ sau đó làm việc trên k giá
              trị.
            </li>
            <li>
              Top-p và min-p mỗi cái giữ một tiền tố của các ứng viên đã sắp xếp, nên cùng nhau chúng giữ tiền tố ngắn hơn.
              Top-p giữ token cắt qua p (một scan loại trừ dưới p); min-p là một phép so sánh với logit lớn nhất cộng
              ln(min-p), tính một lần.
            </li>
            <li>
              Cùng seed, cùng bit: một bộ sinh dựa trên bộ đếm như Philox với khóa (seed, stream, step), kiểm tra với các
              vector đã công bố, và một phép chuyển bit sang float chính xác ở mọi nơi. Cùng bit không có nghĩa là cùng token:
              trọng số float khác nhau giữa các backend, nên chỉ hứa sự thống nhất cho các lần rút không nằm trong phạm vi làm
              tròn quanh một biên, và test đúng điều đó.
            </li>
            <li>Một quy tắc phân định hòa, áp dụng ở mọi giai đoạn, là cách duy nhất để hai backend thống nhất trên một hàng có các logit bằng nhau.</li>
          </ul>
        }
      >
      <ul>
        <li>
          A sampler is a chain of row-wise stages: penalties change logits, top-k, top-p and min-p remove candidates,
          temperature reshapes what is left, and a draw picks one. Top-k depends only on the order of the logits; top-p and
          min-p depend on where temperature runs. Projects disagree on the order, so name it.
        </li>
        <li>
          Every stage is memory-bound. The questions that matter are how many times the row is read and written, and how
          much of the device is busy. With one block per row, batch 1 occupies one multiprocessor.
        </li>
        <li>
          Only the penalties and top-k read the whole vocabulary. Top-k is a selection: a radix select narrows in on the
          k-th key a byte at a time and moves about a quarter of a full sort's bytes. Everything after it works on k values.
        </li>
        <li>
          Top-p and min-p each keep a prefix of the sorted candidates, so together they keep the shorter prefix. Top-p keeps
          the token that crosses p (an exclusive scan below p); min-p is one comparison against the largest logit plus
          ln(min-p), computed once.
        </li>
        <li>
          Same seed, same bits: a counter-based generator such as Philox keyed by (seed, stream, step), checked against
          published vectors, and a bit-to-float conversion that is exact everywhere. Same bits do not mean the same token:
          float weights differ between backends, so promise agreement only for draws that are not within rounding of a
          boundary, and test exactly that.
        </li>
        <li>One tie rule, applied at every stage, is the only way two backends agree on a row with equal logits.</li>
      </ul>
      </Translated>
      <NoteGroup>
        <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
          <Translated
            vi={
              <p>
                Transformer của sách kết thúc ở phân phối trên các token của lớp un-embedding. Mọi thứ ở trên là của riêng
                module này.
              </p>
            }
          >
          <p>
            The book's transformer ends at the un-embedding layer's distribution over tokens. Everything above is this
            module's own.
          </p>
          </Translated>
        </Beyond>
        <Source section="20.1" pages="481" />
      </NoteGroup>

      <h2><Translated vi="Kiểm tra tổng hợp">A mixed check</Translated></h2>
      <p>
        <Translated
          vi={
            <>
              Các câu này dựa trên toàn bộ module, với những tình huống các mục chưa trình bày.
              {asked ? ` Đến giờ trong module này bạn đã trả lời đúng ${right} trên ${asked} câu hỏi, tính theo câu trả lời gần nhất của bạn cho mỗi câu.` : ''}
            </>
          }
        >
        These draw on the whole module, on situations the sections did not show.
        {asked ? ` So far in this module you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
        </Translated>
      </p>
      <Ask q={q['x1/wrap/q-fuse-select']} label={tr('Count the bytes', 'Đếm số byte')} />
      <Ask q={q['x1/wrap/q-near-boundary']} />
      <Ask q={q['x1/wrap/q-pad-vocab']} />
      <Ask q={q['x1/wrap/q-n-samples']} />
      <Ask q={q['x1/wrap/q-hip-popc']} label={tr('Find the defect', 'Tìm lỗi')} />

      <h2><Translated vi="Bài tập của module">The module's exercises</Translated></h2>
      <Translated
        vi={
          <p>
            Sách không có bài tập cho module này, vì sách không có phần lấy mẫu. Mục tiêu của kế hoạch cho X1 là xây dựng lấy
            mẫu greedy, top-k và top-p trên GPU và test chúng với một tham chiếu CPU. Đây là nơi từng phần của việc đó được
            giải.
          </p>
        }
      >
      <p>
        The book has no exercises for this module, because it has no sampling. The plan's goal for X1 is to build greedy,
        top-k and top-p sampling on the GPU and test them against a CPU reference. Here is where each part of that is
        worked.
      </p>
      </Translated>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="exercises">
        <thead>
          <tr><th><Translated vi="Nhiệm vụ">Task</Translated></th><th><Translated vi="Được giải trong một mục">Worked in a section</Translated></th><th><Translated vi="Được xây trong một lab">Built in a lab</Translated></th></tr>
        </thead>
        <tbody>
          <tr><td><Translated vi="Gọi tên các giai đoạn và cố định thứ tự của chúng">Name the stages and fix their order</Translated></td><td><Translated vi={<>trình khám phá chuỗi và scene trong <a href={href('x1', 'X1.1')}>X1.1</a></>}>the chain explorer and scene in <a href={href('x1', 'X1.1')}>X1.1</a></Translated></td><td><Translated vi="lab X1.3, mọi thiết lập">lab X1.3, every setting</Translated></td></tr>
          <tr><td><Translated vi="Greedy, và quy tắc phân định hòa của nó">Greedy, and its tie rule</Translated></td><td><a href={href('x1', 'X1.2')}>X1.2</a></td><td><Translated vi="lab X1.3 ở temperature 0 (argmax của lab 10.3)">lab X1.3 at temperature 0 (lab 10.3's argmax)</Translated></td></tr>
          <tr><td><Translated vi="Temperature">Temperature</Translated></td><td><Translated vi={<><a href={href('x1', 'X1.2')}>X1.2</a> và vị trí của nó trong <a href={href('x1', 'X1.6')}>X1.6</a></>}><a href={href('x1', 'X1.2')}>X1.2</a> and its placement in <a href={href('x1', 'X1.6')}>X1.6</a></Translated></td><td><Translated vi="phép rút của lab X1.3">lab X1.3's draw</Translated></td></tr>
          <tr><td><Translated vi="Phạt lặp lại, phạt tần suất và phạt hiện diện">Repetition, frequency and presence penalties</Translated></td><td><Translated vi={<>trình khám phá phép phạt trong <a href={href('x1', 'X1.3')}>X1.3</a></>}>the penalties explorer in <a href={href('x1', 'X1.3')}>X1.3</a></Translated></td><td><Translated vi="lab X1.3, bài test 4 và 5">lab X1.3, tests 4 and 5</Translated></td></tr>
          <tr><td><Translated vi="Top-k không cần sắp xếp">Top-k without a sort</Translated></td><td><Translated vi={<>trình chạy từng bước, scene và trình khám phá chi phí trong <a href={href('x1', 'X1.4')}>X1.4</a></>}>stepper, scene and cost explorer in <a href={href('x1', 'X1.4')}>X1.4</a></Translated></td><td><Translated vi="lab X1.1">lab X1.1</Translated></td></tr>
          <tr><td>Top-p</td><td><a href={href('x1', 'X1.5')}>X1.5</a></td><td><Translated vi="lab X1.2 phần A, rồi gộp ở phần B">lab X1.2 part A, then fused in part B</Translated></td></tr>
          <tr><td>Min-p</td><td><a href={href('x1', 'X1.6')}>X1.6</a></td><td><Translated vi="lab X1.2">lab X1.2</Translated></td></tr>
          <tr><td><Translated vi="Số ngẫu nhiên thống nhất giữa các backend">Random numbers that agree across backends</Translated></td><td><Translated vi={<>scene các vòng và các trình khám phá trong <a href={href('x1', 'X1.7')}>X1.7</a></>}>the rounds scene and explorers in <a href={href('x1', 'X1.7')}>X1.7</a></Translated></td><td><Translated vi="lab X1.3, bài test 1 và 2">lab X1.3, tests 1 and 2</Translated></td></tr>
          <tr><td><Translated vi="Phép rút">The draw</Translated></td><td><Translated vi={<>trình chạy từng bước và scene trong <a href={href('x1', 'X1.8')}>X1.8</a></>}>stepper and scene in <a href={href('x1', 'X1.8')}>X1.8</a></Translated></td><td><Translated vi="lab X1.3, bài test 3">lab X1.3, test 3</Translated></td></tr>
          <tr><td><Translated vi="Test với một tham chiếu CPU, trên các backend">Test against a CPU reference, across backends</Translated></td><td><Translated vi={<>"Thống nhất chính xác giữa các backend" trong <a href={href('x1', 'X1.8')}>X1.8</a></>}>"Exact agreement across backends" in <a href={href('x1', 'X1.8')}>X1.8</a></Translated></td><td><Translated vi={<>lab X1.3, bài test 5 đến 7, và digest trên <a href={href('x1', 'labs')}>trang labs</a></>}>lab X1.3, tests 5 to 7, and the digest on the <a href={href('x1', 'labs')}>labs page</a></Translated></td></tr>
          <tr><td><Translated vi="Gỡ lỗi một bộ lọc gộp">Debug a fused filter</Translated></td><td><Translated vi={<>các quy tắc biên của <a href={href('x1', 'X1.5')}>X1.5</a></>}>the boundary rules of <a href={href('x1', 'X1.5')}>X1.5</a></Translated></td><td><Translated vi="lab X1.4">lab X1.4</Translated></td></tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Các bài tập của chính sách về từng phần được ánh xạ trên trang tổng kết của các chương tương ứng: histogram ở{' '}
            <a href={href('ch09', 'wrap')}>chương 9</a>, phép rút gọn ở <a href={href('ch10', 'wrap')}>chương 10</a>, scan ở{' '}
            <a href={href('ch11', 'wrap')}>chương 11</a>, lọc ở <a href={href('ch12', 'wrap')}>chương 12</a> và
            sắp xếp ở <a href={href('ch14', 'wrap')}>chương 14</a>.
          </p>
        }
      >
      <p>
        The book's own exercises on the parts are mapped on their chapters' summary pages: histograms in{' '}
        <a href={href('ch09', 'wrap')}>chapter 9</a>, reduction in <a href={href('ch10', 'wrap')}>chapter 10</a>, scan in{' '}
        <a href={href('ch11', 'wrap')}>chapter 11</a>, filtering in <a href={href('ch12', 'wrap')}>chapter 12</a> and
        sorting in <a href={href('ch14', 'wrap')}>chapter 14</a>.
      </p>
      </Translated>

      <h2><Translated vi="Nơi các nguồn không thống nhất">Where the sources disagree</Translated></h2>
      <Translated
        vi={
          <p>
            Module này không có PDF nào để sai: nó không có trong sách. Thay vào đó, nó có các dự án, bài báo và thư viện của
            các hãng không thống nhất với nhau, và với điều một người đọc sẽ giả định. Đây là những bất đồng mà module được xây
            dựng quanh đó, tất cả đã được kiểm tra trong lượt kiểm tra dữ kiện.
          </p>
        }
      >
      <p>
        This module has no PDF to be wrong: it is not in the book. What it has instead are projects, papers and vendor
        libraries that disagree with each other, and with what a reader would assume. These are the disagreements the
        module was built around, all checked in the facts pass.
      </p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>
              <b>Thứ tự.</b> Chuỗi mặc định của llama.cpp và mlx-lm cắt bớt trước temperature; transformers áp dụng temperature
              trước; mlx-lm chạy top-p, rồi min-p, rồi top-k; bài báo min-p áp dụng min-p sau temperature.
            </li>
            <li>
              <b>Hòa.</b> Top-k của llama.cpp giữ các cột thấp nhất; transformers giữ mọi token hòa với token thứ k và có thể
              trả về nhiều hơn k; top-p của mlx-lm giữ mọi token hòa tại biên; argmax Metal của llama.cpp trả về chỉ số hòa cao
              nhất. M5 đưa mọi trường hợp hòa về id thấp nhất.
            </li>
            <li>
              <b>Phạt lặp lại.</b> Bài báo CTRL chia bất kể dấu. llama.cpp nhân một logit nhỏ hơn hoặc bằng 0 và chia một
              logit dương; transformers và mlx-lm kiểm tra dưới 0, cho cùng kết quả tại 0.
            </li>
            <li>
              <b>Phép rút.</b> Phép rút phía host của llama.cpp lấy tổng chạy đầu tiên lớn hơn hoặc bằng u × total, đường
              backend của nó lấy tổng đầu tiên lớn hơn hẳn. <code>torch.multinomial(probs, 1)</code> là một cuộc đua mũ, không
              phải CDF nghịch đảo. <code>categorical</code> của MLX đổi thuật toán theo hình dạng batch, nên cùng một khóa có
              thể cho các token khác nhau.
            </li>
            <li>
              <b>Số ngẫu nhiên.</b> <code>hiprand_uniform</code> trên AMD và <code>curand_uniform</code> biến cùng 32 bit thành
              các số float khác nhau với 1 trên 256 đầu vào (theo cách đếm của chúng tôi). Bảng của AMD nói Philox của rocRAND
              không phải chuỗi của cuRAND, trong khi header thiết bị của nó khởi tạo theo cùng cách theo cách chúng tôi đọc:
              chưa giải quyết. Philox MPS của PyTorch chạy 9 vòng theo cách chúng tôi đọc mã nguồn. MSL không có hàm sinh số
              ngẫu nhiên, và có thể làm tròn số học float về phía 0.
            </li>
          </ul>
        }
      >
      <ul>
        <li>
          <b>Order.</b> llama.cpp's default chain and mlx-lm truncate before temperature; transformers applies temperature
          first; mlx-lm runs top-p, then min-p, then top-k; the min-p paper applies min-p after temperature.
        </li>
        <li>
          <b>Ties.</b> llama.cpp's top-k keeps the lowest columns; transformers keeps every token tied with the k-th and can
          return more than k; mlx-lm's top-p keeps every token tied at the boundary; llama.cpp's Metal argmax returns the
          highest tied index. M5 sends every tie to the lowest id.
        </li>
        <li>
          <b>Repetition penalty.</b> The CTRL paper divides regardless of sign. llama.cpp multiplies a logit at or below 0
          and divides a positive one; transformers and mlx-lm test below 0, which gives the same result at 0.
        </li>
        <li>
          <b>The draw.</b> llama.cpp's host draw takes the first running sum at or above u × total, its backend path the
          first strictly above. <code>torch.multinomial(probs, 1)</code> is an exponential race, not an inverse CDF. MLX's{' '}
          <code>categorical</code> switches algorithm with the batch shape, so the same key can give different tokens.
        </li>
        <li>
          <b>Random numbers.</b> <code>hiprand_uniform</code> on AMD and <code>curand_uniform</code> turn the same 32 bits
          into different floats for 1 input in 256 (our count). AMD's table says rocRAND's Philox is not cuRAND's sequence,
          while its device header initialises the same way by our reading: unresolved. PyTorch's MPS Philox runs 9 rounds by
          our reading of its source. MSL has no random number function, and may round float arithmetic toward zero.
        </li>
      </ul>
      </Translated>
      <Today checked={tr('2026-11-20 in the facts pass (notes/today/x1-facts.md B1, B2, C1, C5, D1 to D9, E2; ch10-facts.md D2; ch11-facts.md D3; ch14-facts.md D2)', '2026-11-20 trong lượt kiểm tra dữ kiện (notes/today/x1-facts.md B1, B2, C1, C5, D1 đến D9, E2; ch10-facts.md D2; ch11-facts.md D3; ch14-facts.md D2)')}>
        <Translated vi={<p>Phần "Summary of corrections" trong notes/today/x1-facts.md liệt kê mọi mục cùng nguồn của nó.</p>}>
        <p>The "Summary of corrections" in notes/today/x1-facts.md lists every item with its source.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Hai chỗ mà chính tệp dữ kiện và kế hoạch của chúng tôi đã sai, được phát hiện khi xây dựng các lab. Ghi chú của tệp
            dữ kiện về hai phép chuyển đổi của thư viện nói rằng các mẫu nó in ra khớp nhau từ 2²⁴ trở lên; câu đó chưa đầy đủ,
            và E2 giờ mang một bản đính chính có ghi ngày. Theo cách đếm binade của chúng tôi, đúng 2²⁴ trong 2³² đầu vào khác
            nhau, trong đó có một phần tư số đầu vào trong [2²⁴, 2²⁵), và <code>app/src/sim/philox.test.ts</code> kiểm tra điều
            đó. Còn kế hoạch thì kỳ vọng trình giả lập bắt được mọi rào chắn bị thiếu của X1.8 và lab X1.4. Nó bắt được rào chắn
            A của X1.8. Một lần ghi sau một lần đọc giữa các warp, với mọi lần đọc diễn ra trước điểm hẹn cấp warp đầu tiên (rào
            chắn B của X1.8, lỗi 3 của lab X1.4), qua được cả ba thứ tự của nó, và chỉ backend race báo cáo điều đó.
          </p>
        }
      >
      <p>
        Two places where our own facts file and plan were wrong, found while building the labs. The facts file's note on
        the two library conversions said the samples it printed agree from 2²⁴ upward; that sentence was incomplete, and E2
        now carries a dated correction. By our binade count exactly 2²⁴ of the 2³² inputs differ, a quarter of the inputs
        in [2²⁴, 2²⁵) among them, and <code>app/src/sim/philox.test.ts</code> checks it. And the plan expected the emulator to catch every missing
        barrier of X1.8 and lab X1.4. It catches X1.8's barrier A. A write after a read across warps, with every read
        before the first warp-level rendezvous (X1.8's barrier B, lab X1.4's defect 3), passes in all three of its orders,
        and only the race backend reports it.
      </p>
      </Translated>
      <Aside>
        <Translated vi={<p>Đây là cách chúng tôi đọc các nguồn vào ngày 2026-11-20. Các dự án thay đổi; hãy kiểm tra tệp trước khi dựa vào một dòng nào.</p>}>
        <p>These are our readings of the sources on 2026-11-20. Projects move; check the file before you rely on a line.</p>
        </Translated>
      </Aside>

      <h2><Translated vi="Dự án đến lúc này">The project so far</Translated></h2>
      <p>
        <Translated
          vi={
            <>
              <code>launchSampleRows</code> của lab X1.3 hoàn thành mốc dự án M5: logits và số đếm token{' '}
              <code>[rows × vocab]</code> đi vào, một id token mỗi hàng đi ra, cùng token cho cùng (seed, stream, step) trên mọi
              backend với mọi lần rút dứt khoát.{' '}
            </>
          }
        >
        Lab X1.3's <code>launchSampleRows</code> completes milestone M5: <code>[rows × vocab]</code> logits and token counts
        in, one token id per row out, the same token for the same (seed, stream, step) on every backend for every decisive
        draw.{' '}
        </Translated>
        <span data-m5-own={loading ? 'loading' : own.join(',') || 'none'}>
          {loading
            ? tr('Reading reports.', 'Đang đọc báo cáo.')
            : !ran
              ? tr('No lab results are recorded on this machine yet.', 'Chưa có kết quả lab nào được ghi lại trên máy này.')
              : own.length
                ? tr(`Your own sampler passes on ${own.map(where).join(' and ')} on this machine.`, `Sampler của chính bạn qua trên ${own.map(where).join(' và ')} trên máy này.`)
                : tr('Your own sampler does not pass on any backend of this machine yet.', 'Sampler của chính bạn chưa qua trên backend nào của máy này.')}
        </span>{' '}
        <Translated vi={<>Xem <a href={href('x1', 'labs')}>các lab</a>.</>}>
        See the <a href={href('x1', 'labs')}>labs</a>.
        </Translated>
      </p>
      <table className="data" data-table="m5">
        <thead>
          <tr><th><Translated vi="Phần của M5">Piece of M5</Translated></th><th>Lab</th><th><Translated vi="Bản làm của bạn qua trên">Your own work passes on</Translated></th></tr>
        </thead>
        <tbody>
          {M5.map((m) => {
            const on = passingOn(reports, m.dir);
            return (
              <tr key={m.dir} data-m5={m.num} data-pass={loading ? 'loading' : on.join(',') || 'none'}>
                <td>{tr(m.title, ({
                  '9.3': 'số đếm token cho phép phạt lặp lại',
                  '10.3': 'softmax và argmax trên logits (M4)',
                  '11.6': 'mask giữ của top-p',
                  '12.4': 'nén gọn ứng viên',
                  '14.3': 'ứng viên đã sắp xếp, mọi hàng giảm dần và ổn định',
                  'X1.1': 'top-k bằng radix select, đo so với sắp xếp toàn bộ',
                  'X1.2': 'top-p và min-p: ghép từ các lab trước, rồi gộp',
                  'X1.3': 'mốc dự án M5: một sampler GPU theo batch',
                } as Record<string, string>)[m.num] ?? m.title)}</td>
                <td className="nowrap">
                  <a href={href(m.chapter, 'labs')}><Translated vi={<>lab {m.num}</>}>lab {m.num}</Translated></a>
                </td>
                <td>{loading ? tr('reading reports', 'đang đọc báo cáo') : on.length ? on.map(where).join(', ') : <span className="muted"><Translated vi="chưa có">nothing yet</Translated></span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Tiếp theo trong thứ tự của khóa học này là chương 20 của sách, về các mô hình ngôn ngữ lớn (pp. 477–512): các kernel
            attention của mốc dự án M6 và KV cache của M7. Mốc dự án M10 của module X3 sau đó chạy một mô hình nhỏ kiểu Llama
            trên bước decode của M7 và, trên GPU khi có lời giải của lab X1.3, có thể rút các token của nó bằng sampler này thay
            cho argmax.
          </p>
        }
      >
      <p>
        Next in this course's order is chapter 20 of the book, on large language models (pp. 477–512): the attention
        kernels of milestone M6 and the KV cache of M7. Module X3's milestone M10 then runs a small Llama-style model on
        M7's decode step and, on a GPU when lab X1.3's solution is present, can draw its tokens with this sampler in place
        of the argmax.
      </p>
      </Translated>
    </>
  );
}
