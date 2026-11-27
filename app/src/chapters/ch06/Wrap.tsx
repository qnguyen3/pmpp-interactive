import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { warpAccess } from '../../sim/coalesce';
import { Translated, tr } from '../../lib/i18n';

// A kernel of our own, for two questions. Checked copy: labs/ch06/snippets/wrap_scores.cu
// (snippets-wrap.test.ts ties the two and pins the numbers below).
const SCORES = `// Attention scores for one query per head: s[h][t] = dot(q[h], K[t][h]).
// The key cache K is stored (token, head, dim), row-major. One thread per (head, token).
__global__ void scoresKernel(const float* q, const float* K, float* s, int T, int H, int D) {
    int h = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T) {
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) {
            acc += q[h * D + d] * K[(t * H + h) * D + d];
        }
        s[h * T + t] = acc;
    }
}`;

// ---------------------------------------------------------------- the numbers the questions take from the model
// NVIDIA's model (notes/today/ch06-facts.md A2): one warp's load costs the distinct
// 32-byte sectors its lanes touch. Byte addresses of one warp's 32 lanes, at one d.
const LANES = 32;
const SECTOR = 32;
export const SHAPE = { H: 32, D: 128, h: 5, d: 7 } as const;
const lanes = (f: (lane: number) => number) => Array.from({ length: LANES }, (_, l) => 4 * f(l));
const { H, D, h, d } = SHAPE;
/** The kernel as written: lane = token, so neighbouring lanes are H × D floats apart. */
export const SCORES_K = warpAccess(lanes((t) => (t * H + h) * D + d), 4, SECTOR);
/** The q load: every lane reads the same float. */
export const SCORES_Q = warpAccess(lanes(() => h * D + d), 4, SECTOR);
/** The same lanes reading a float4 of their own row instead (d a multiple of 4). */
export const SCORES_K4 = warpAccess(lanes((t) => (t * H + h) * D + 4), 16, SECTOR);
/** Remapped: one token per warp, lanes split d. */
export const SCORES_REMAPPED = warpAccess(lanes((l) => (3 * H + h) * D + l), 4, SECTOR);

/** Permuting (T, H, D) to (H, T, D): one warp of consecutive output elements, read and write side. */
export const PERMUTE = { T: 1000, H: 32, D: 128, first: 4096 * 37 + 64 } as const;
const outToIn = (o: number) => {
  const { T, H: nh, D: nd } = PERMUTE;
  const dd = o % nd;
  const t = Math.floor(o / nd) % T;
  const hh = Math.floor(o / (nd * T));
  return (t * nh + hh) * nd + dd;
};
export const PERMUTE_WRITE = warpAccess(lanes((l) => PERMUTE.first + l), 4, SECTOR);
export const PERMUTE_READ = warpAccess(lanes((l) => outToIn(PERMUTE.first + l)), 4, SECTOR);

/** Does every row of a K-float row-major matrix start on a 16-byte boundary (base aligned)? */
export const rowsAligned = (K: number) => (4 * K) % 16 === 0;
/** How many of the first `rows` rows start on a 16-byte boundary. */
export const alignedRows = (K: number, rows: number) => Array.from({ length: rows }, (_, r) => (4 * r * K) % 16 === 0).filter(Boolean).length;

export const questionsWrap: Record<string, Question> = {
  'ch06/wrap/q-scores-sectors': {
    kind: 'numeric',
    id: 'ch06/wrap/q-scores-sectors',
    prompt: (
      <Translated
        vi={
          <>
            Lấy <code>scoresKernel</code> ở trên với H = {H} head và D = {D}. Ở một vòng lặp <code>d</code>, lần nạp từ <code>K</code> của một warp 32 lane chạm tới bao nhiêu sector 32 byte, theo mô hình của NVIDIA?
          </>
        }
      >
        <>
          Take <code>scoresKernel</code> above with H = {H} heads and D = {D}. At one iteration <code>d</code>, how many
          32-byte sectors does one 32-lane warp&rsquo;s load from <code>K</code> touch, in NVIDIA&rsquo;s model?
        </>
      </Translated>
    ),
    answer: SCORES_K.transactions,
    unit: 'sectors',
    vi: { unit: 'sector', source: 'quy tắc của ch. 6 §6.1, pp. 126–129; mô hình sector, notes/today/ch06-facts.md A2; kernel là của chúng tôi' },
    explain: (
      <Translated
        vi={
          <p>
            Các lane kề nhau là các token kề nhau, và <code>t</code> được nhân với H × D = {H * D} float, nên các lane cách nhau {4 * H * D} byte: mỗi lane cần một sector riêng, tổng cộng {SCORES_K.transactions}, và chỉ dùng 4 trong 32 byte của nó ({Math.round(SCORES_K.efficiency * 1000) / 10}%). Lần nạp từ <code>q</code> là trường hợp ngược lại: mọi lane đọc cùng một float, {SCORES_Q.transactions} sector cho cả warp. Cùng quy tắc với N thứ tự cột (column-major) của mục 6.1, trong một kernel bạn sẽ gặp ở attention.
          </p>
        }
      >
        <p>
          Neighbouring lanes are neighbouring tokens, and <code>t</code> is multiplied by H × D = {H * D} floats, so the
          lanes are {4 * H * D} bytes apart: each needs a sector of its own, {SCORES_K.transactions} in all, and uses 4 of
          its 32 bytes ({Math.round(SCORES_K.efficiency * 1000) / 10}%). The load from <code>q</code> is the opposite
          case: every lane reads the same float, {SCORES_Q.transactions} sector for the whole warp. Same rule as
          section 6.1&rsquo;s column-major N, in a kernel you will meet in attention.
        </p>
      </Translated>
    ),
    source: 'rule of ch. 6 §6.1, pp. 126–129; sector model, notes/today/ch06-facts.md A2; the kernel is ours',
  },
  'ch06/wrap/q-scores-fix': {
    kind: 'choice',
    id: 'ch06/wrap/q-scores-fix',
    prompt: (
      <Translated
        vi={
          <>
            Bạn không được thay đổi cách lưu key cache: các kernel khác ghi vào nó. Thay đổi nào đối với <code>scoresKernel</code> làm các lần đọc <code>K</code> của nó được hợp nhất?
          </>
        }
      >
        <>
          You may not change how the key cache is stored: other kernels write it. Which change to{' '}
          <code>scoresKernel</code> makes its reads of <code>K</code> coalesce?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Mỗi warp một token, 32 lane của nó chia nhau vòng lặp d, rồi cộng các kết quả từng phần của chúng.">One token per warp, its 32 lanes splitting the d loop, then a sum of their partial results.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Đọc K bằng các lần nạp float4, bốn giá trị d mỗi lane, để mỗi sector được dùng trọn.">Read K with float4 loads, four values of d per lane, so that each sector is fully used.</Translated>,
        why: <Translated vi={`Khi đó mỗi lane dùng 16 byte của sector thay vì 4, nhưng lane bên cạnh vẫn cách ${4 * H * D} byte: ${SCORES_K4.transactions} sector mỗi warp, như trước.`}>{`Each lane then uses 16 bytes of its sector instead of 4, but its neighbour is still ${4 * H * D} bytes away: ${SCORES_K4.transactions} sectors per warp, as before.`}</Translated>,
      },
      {
        label: <Translated vi="Đệm mỗi hàng của K thêm một float, để các token kề nhau rơi vào các bank khác nhau.">Pad each row of K by one float, so that neighbouring tokens fall in different banks.</Translated>,
        why: <Translated vi="Đệm giúp dời một cột ra khỏi một bank của bộ nhớ chia sẻ. Trong bộ nhớ toàn cục các lane vẫn cách nhau H × D + 1 float, mỗi lane trong một sector riêng.">Padding moves a column off one bank of shared memory. In global memory the lanes would still be H × D + 1 floats apart, each in its own sector.</Translated>,
      },
      {
        label: <Translated vi="Trung chuyển K vào bộ nhớ chia sẻ trước, với cùng cách ánh xạ luồng, rồi tính từ đó.">Stage K in shared memory first, with the same thread mapping, and compute from there.</Translated>,
        why: <Translated vi="Lần nạp vào bộ nhớ chia sẻ vẫn là lần nạp rải rác y như vậy. Xoay góc hiệu quả vì các luồng thay đổi phần tử chúng nạp, và đó là một phép ánh xạ lại.">The load into shared memory would be the same scattered load. Corner turning works because the threads change which element they load, and that is a remapping.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đây là cách khắc phục thứ nhất của sách: thay đổi luồng nào xử lý dữ liệu nào. Với các lane chạy dọc theo <code>d</code>, 32 lane đọc 32 float liên tiếp trong key của một token, {SCORES_REMAPPED.transactions} sector, mọi byte đều được dùng. Cái giá là một phép cộng qua các lane của warp ở cuối, việc mà chương 10 làm hiệu quả. Cách khắc phục thứ hai của sách, một bố cục khác như (head, dim, token), cũng sẽ được, với cái giá đổ lên mọi kernel ghi vào cache.
          </p>
        }
      >
        <p>
          This is the book&rsquo;s first remedy: change which thread handles which data. With lanes along <code>d</code>,
          32 lanes read 32 consecutive floats of one token&rsquo;s key, {SCORES_REMAPPED.transactions} sectors, all bytes
          used. The price is a sum across the lanes of a warp at the end, which chapter 10 does efficiently. The
          book&rsquo;s second remedy, a different layout such as (head, dim, token), would also work, at the cost of every
          kernel that writes the cache.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.1, p. 129 (three strategies) and Fig. 6.13 row 4, p. 148; the kernel is ours',
    vi: { source: 'ch. 6 §6.1, p. 129 (ba chiến lược) và Fig. 6.13 hàng 4, p. 148; kernel là của chúng tôi' },
  },
  'ch06/wrap/q-permute': {
    kind: 'choice',
    id: 'ch06/wrap/q-permute',
    prompt: (
      <Translated
        vi={
          <>
            Tách head: một kernel sao chép Q từ hình dạng (T, H, D) sang (H, T, D), float theo thứ tự hàng, D = {PERMUTE.D}, mỗi luồng một phần tử, với các luồng liên tiếp ghi các phần tử đầu ra liên tiếp. Những truy cập nào của nó được hợp nhất?
          </>
        }
      >
        <>
          Splitting heads: a kernel copies Q from shape (T, H, D) to (H, T, D), row-major floats, D = {PERMUTE.D}, one
          thread per element, with consecutive threads writing consecutive output elements. Which of its accesses
          coalesce?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Cả lần đọc lẫn lần ghi.">Both the reads and the writes.</Translated>, correct: true },
      {
        label: <Translated vi="Chỉ lần ghi: lần đọc nhảy theo H × D.">Only the writes: the reads jump by H × D.</Translated>,
        why: <Translated vi={`Các phần tử đầu ra liên tiếp khác nhau trước hết ở d, và d cũng là chiều trong cùng của đầu vào. Một warp đọc ${PERMUTE_READ.transactions} sector.`}>{`Consecutive output elements differ first in d, and d is the innermost dimension of the input too. A warp reads ${PERMUTE_READ.transactions} sectors.`}</Translated>,
      },
      {
        label: <Translated vi="Chỉ lần đọc.">Only the reads.</Translated>,
        why: <Translated vi="Lần ghi liên tiếp theo cách xây dựng: luồng liên tiếp, phần tử đầu ra liên tiếp.">The writes are consecutive by construction: consecutive threads, consecutive output elements.</Translated>,
      },
      {
        label: <Translated vi="Không cái nào: đó là một phép chuyển vị.">Neither: it is a transpose.</Translated>,
        why: <Translated vi="Nó chỉ hoán đổi hai chiều ngoài. Trường hợp mà một phía trở thành truy cập theo bước nhảy là một phép chuyển vị di chuyển chiều trong cùng, như của lab 6.1.">It swaps the two outer dimensions only. The case where one side turns strided is a transpose that moves the innermost dimension, like lab 6.1's.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Phần tử đầu ra (h, t, d) đến từ phần tử đầu vào (t, h, d). Với {PERMUTE.D} phần tử đầu ra liên tiếp, h và t cố định còn d chạy từ 0 đến {PERMUTE.D - 1}, nên các địa chỉ đầu vào cũng liên tiếp. Một warp 32 lane ghi {PERMUTE_WRITE.transactions} sector và đọc {PERMUTE_READ.transactions}. Một hoán vị giữ nguyên chiều trong cùng sẽ sao chép các đoạn liền nhau có độ dài bằng chiều đó; chỉ khi các đoạn ngắn hơn một warp thì một warp mới vắt qua nhiều đoạn.
          </p>
        }
      >
        <p>
          Output element (h, t, d) comes from input element (t, h, d). For {PERMUTE.D} consecutive output elements h and t
          are fixed and d runs from 0 to {PERMUTE.D - 1}, so the input addresses are consecutive as well. A 32-lane warp
          writes {PERMUTE_WRITE.transactions} sectors and reads {PERMUTE_READ.transactions}. A permutation that keeps the
          innermost dimension in place copies contiguous runs of that dimension&rsquo;s length; only when the runs are
          shorter than a warp does a warp straddle several of them.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.1, pp. 126–128; data layout strategies, Fig. 6.13, p. 148; the example is ours',
    vi: { source: 'ch. 6 §6.1, pp. 126–128; các chiến lược bố cục dữ liệu, Fig. 6.13, p. 148; ví dụ là của chúng tôi' },
  },
  'ch06/wrap/q-rowalign': {
    kind: 'choice',
    id: 'ch06/wrap/q-rowalign',
    prompt: (
      <Translated
        vi={
          <>
            Một ma trận K float mỗi hàng đến từ <code>cudaMalloc</code>, và mỗi luồng đọc hàng của nó qua các lần nạp <code>float4</code> bắt đầu từ phần tử đầu tiên của hàng. Với K nào thì mọi hàng đều bắt đầu ở chỗ cho phép một lần nạp <code>float4</code>?
          </>
        }
      >
        <>
          A matrix of K floats per row comes from <code>cudaMalloc</code>, and each thread reads its row through{' '}
          <code>float4</code> loads starting at the row&rsquo;s first element. For which K does every row start where a{' '}
          <code>float4</code> load is allowed?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="K là bội của 4.">K a multiple of 4.</Translated>, correct: true },
      { label: <Translated vi="K bất kỳ: vùng cấp phát được căn chỉnh theo 256 byte.">Any K: the allocation is aligned to 256 bytes.</Translated>, why: <Translated vi="Chỉ hàng đầu tiên bắt đầu ở vùng cấp phát. Hàng r bắt đầu sau đó 4·r·K byte.">Only the first row starts at the allocation. Row r starts 4·r·K bytes later.</Translated> },
      { label: <Translated vi="K là lũy thừa của hai.">K a power of two.</Translated>, why: <Translated vi="Đủ khi từ 4 trở lên, nhưng không cần thiết: K = 12 cũng được.">Sufficient from 4 upwards, but not necessary: K = 12 works too.</Translated> },
      { label: <Translated vi="K là bội của 32, độ rộng warp.">K a multiple of 32, the warp width.</Translated>, why: <Translated vi="Căn chỉnh là thuộc tính của từng địa chỉ, không phải của warp. 16 byte là 4 float.">Alignment is a property of each address, not of the warp. 16 bytes is 4 floats.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hàng r bắt đầu sau địa chỉ gốc đã căn chỉnh 4·r·K byte, và một truy cập <code>float4</code> cần một biên 16 byte, nên 16 phải chia hết 4·K: K là bội của 4. Với K = 4097 chỉ cứ mỗi hàng thứ tư mới bắt đầu ở một biên 16 byte ({alignedRows(4097, 8)} trong 8 hàng đầu); với K = 4100 thì mọi hàng đều vậy ({alignedRows(4100, 8)} trên 8). Mã thực tế canh đúng điều này: phép nhân ma trận Metal của llama.cpp lùi về đọc từng phần tử khi độ dài hàng phá vỡ căn chỉnh của kiểu vector, và PyTorch chọn độ rộng vector dựa trên căn chỉnh của từng con trỏ lúc chạy.
          </p>
        }
      >
        <p>
          Row r starts 4·r·K bytes after the aligned base, and a <code>float4</code> access needs a 16-byte boundary, so 16
          must divide 4·K: K a multiple of 4. With K = 4097 only every fourth row starts on a 16-byte boundary
          ({alignedRows(4097, 8)} of the first 8); with K = 4100 every row does ({alignedRows(4100, 8)} of 8). Real code
          guards exactly this: llama.cpp&rsquo;s Metal matrix multiply falls back to element-wise reads when
          the row length breaks the vector type&rsquo;s alignment, and PyTorch picks its vector width from each
          pointer&rsquo;s alignment at run time.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch06-facts.md A3 (256-byte allocations), A11 (alignment to the vector size), D2 and D4',
    vi: { source: 'notes/today/ch06-facts.md A3 (cấp phát theo 256 byte), A11 (căn chỉnh theo kích thước vector), D2 và D4' },
  },
  'ch06/wrap/q-threebuf': {
    kind: 'choice',
    id: 'ch06/wrap/q-threebuf',
    prompt: (
      <Translated
        vi={
          <>
            Lab 6.5 cho mỗi toán hạng hai ô và giữ một rào chắn mỗi giai đoạn, sau các lần nạp. Một đồng nghiệp cho mỗi toán hạng ba ô, chọn bằng <code>ph % 3</code>, và bỏ luôn rào chắn cuối cùng đó. Có đúng không?
          </>
        }
      >
        <>
          Lab 6.5 gives each operand two tiles and keeps one barrier per phase, after the loads. A colleague gives each
          operand three tiles, chosen by <code>ph % 3</code>, and removes that last barrier too. Correct?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Không: rào chắn đó canh một phụ thuộc thật, thứ mà không số buffer nào loại bỏ được.">No: that barrier guards a true dependence, which no number of buffers removes.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Có: với ba buffer, không luồng nào có thể ghi đè một ô mà luồng khác vẫn đang đọc.">Yes: with three buffers no thread can overwrite a tile someone is still reading.</Translated>,
        why: <Translated vi="Đó là phụ thuộc giả, và hai buffer đã loại bỏ nó. Rào chắn còn lại liên quan đến việc đọc một giá trị trước khi nó được ghi.">That was the false dependence, and two buffers already removed it. The remaining barrier is about reading a value before it is written.</Translated>,
      },
      {
        label: <Translated vi="Có, trên phần cứng chạy một warp theo kiểu đồng bộ từng bước (lockstep).">Yes, on hardware that runs a warp in lockstep.</Translated>,
        why: <Translated vi="Một khối 16 × 16 là tám warp 32 luồng, và không có gì bảo đảm các lần nạp của warp này xảy ra trước các lần đọc của warp khác.">A 16 × 16 block is eight warps of 32, and nothing orders one warp's loads before another warp's reads.</Translated>,
      },
      {
        label: <Translated vi="Không: ba buffer cần hai rào chắn mỗi giai đoạn.">No: three buffers need two barriers per phase.</Translated>,
        why: <Translated vi="Thêm buffer không bao giờ thêm rào chắn. Chúng chỉ làm cho việc chờ ghi-sau-đọc trở nên không cần thiết.">More buffers never add barriers. They only make the write-after-read wait unnecessary.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Buffer tách biệt vị trí, và điều đó chỉ loại bỏ các lần chờ ghi-sau-đọc. Rào chắn sau các lần nạp là một lần chờ đọc-sau-ghi: tích vô hướng đọc các ô nhớ trong tile mà các luồng khác ghi trong cùng giai đoạn. Điều mà thêm buffer có thể làm là cho các lần nạp của giai đoạn tiếp theo bắt đầu sớm, việc này cần một phép sao chép mà các luồng không phải chờ; đó là hướng chương 15 đưa buffer kép tới.
          </p>
        }
      >
        <p>
          Buffers separate locations, and that removes write-after-read waits only. The barrier after the loads is a
          read-after-write wait: the dot product reads tile slots that other threads write in the same phase. What more
          buffers can do is let the next phase&rsquo;s loads start early, which needs a copy the threads do not wait for;
          that is where chapter 15 takes double buffering.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.7, pp. 146–147 (true and false dependences); ch. 5 §5.4, pp. 110–111',
    vi: { source: 'ch. 6 §6.7, pp. 146–147 (phụ thuộc thật và giả); ch. 5 §5.4, pp. 110–111' },
  },
  'ch06/wrap/q-bottleneck': {
    kind: 'choice',
    id: 'ch06/wrap/q-bottleneck',
    prompt: (
      <Translated
        vi={
          <>
            Một profiler báo cáo hai điều về một kernel: thông lượng DRAM của nó ở mức 92% đỉnh của thiết bị, và 40% số truy cập bộ nhớ chia sẻ của nó có xung đột bank. Bạn thay đổi gì trước?
          </>
        }
      >
        <>
          A profiler reports two things about one kernel: its DRAM throughput is at 92% of the device&rsquo;s peak, and 40%
          of its shared-memory accesses have bank conflicts. What do you change first?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Di chuyển ít byte hơn từ DRAM, rồi profile lại.">Move fewer bytes from DRAM, then profile again.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Đệm các ô trong bộ nhớ chia sẻ: 40% là rất nhiều xung đột.">Pad the shared tiles: 40% is a lot of conflicts.</Translated>,
        why: <Translated vi="Giải tỏa một tài nguyên không phải điểm nghẽn chẳng mang lại gì (p. 153). Ở mức 92% đỉnh, DRAM mới là thứ đang thiếu.">Relieving a resource that is not the bottleneck buys nothing (p. 153). At 92% of its peak, DRAM is the one that is short.</Translated>,
      },
      {
        label: <Translated vi="Chuyển sang nạp dạng vector, để di chuyển cùng số byte với ít lệnh hơn.">Switch to vector loads, to move the same bytes in fewer instructions.</Translated>,
        why: <Translated vi="Giới hạn là số byte, không phải số lệnh di chuyển chúng, và chẳng còn bao nhiêu băng thông để việc phát lệnh nhanh hơn tận dụng.">The bytes are the limit, not the instructions that move them, and there is little bandwidth left for faster issue to use.</Translated>,
      },
      {
        label: <Translated vi="Tăng occupancy, để che giấu thêm độ trễ.">Raise occupancy, to hide more latency.</Translated>,
        why: <Translated vi="Thêm yêu cầu đang chờ phục vụ chỉ có ích khi băng thông còn để không. Ở đây nó gần như đã dùng hết.">More requests in flight help while bandwidth sits idle. Here it is nearly used up.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Gần mái bộ nhớ, đòn bẩy duy nhất còn lại là ít byte hơn: tái sử dụng nhiều hơn, hoặc kiểu dữ liệu nhỏ hơn. Các xung đột bank là có thật, và một khi lưu lượng DRAM giảm chúng có thể trở thành điểm nghẽn: đó là lý do bạn profile lại sau mỗi thay đổi và đánh giá nó bằng thời gian chạy của kernel, không phải bằng một con số phần trăm.
          </p>
        }
      >
        <p>
          Near the memory roof the only lever left is fewer bytes: more reuse, or smaller types. The bank conflicts are real, and once DRAM traffic falls
          they may become the bottleneck: that is why you profile again after each change and judge it by the
          kernel&rsquo;s duration, not by a percentage.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.9, pp. 153–154; Nsight Compute Compute Triage Guide, notes/today/ch06-facts.md A17; the numbers are made up',
    vi: { source: 'ch. 6 §6.9, pp. 153–154; Nsight Compute Compute Triage Guide, notes/today/ch06-facts.md A17; các con số là giả định' },
  },
};

// ---------------------------------------------------------------- the page
export function Wrap() {
  const { progress } = useProgress();
  const { reports, loading } = useReports();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const m2 = backends.filter((b) => labStatus(reports[b], 'ch06/lab4_linear', 'work') === 'pass');
  const name = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch06/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>
        <Translated vi="Những gì cần mang theo">What to carry forward</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương này khép lại Phần 1 của sách. Với nó, bạn có thể lập luận về việc thời gian của bất kỳ kernel nào đi đâu, và đây là những phần cần giữ lại.
          </p>
        }
      >
        <p>
          The chapter closes Part 1 of the book. With it you can reason about where any kernel&rsquo;s time goes, and
          these are the parts to keep.
        </p>
      </Translated>
      <Source section="6.10" pages="154" />
      <ul>
        <Translated
          vi={
            <li>
              Một lần nạp của warp tốn bằng số đơn vị bộ nhớ khác nhau mà các lane của nó chạm tới, không phải bằng số lane. Các lane đọc địa chỉ kề nhau dùng chung đơn vị; nhân chỉ số luồng với bất cứ số nào lớn hơn một thì con số đó tăng lên. Đơn vị là của nhà sản xuất: sector 32 byte trên NVIDIA, dòng 64 hoặc 128 byte trên AMD, Apple không công bố.
            </li>
          }
        >
          <li>
            A warp&rsquo;s load costs the distinct memory units its lanes touch, not the number of lanes. Lanes that read
            neighbouring addresses share units; multiply the thread index by anything larger than one and the count
            grows. The unit is the vendor&rsquo;s: 32-byte sectors on NVIDIA, 64- or 128-byte lines on AMD, none documented
            by Apple.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Khi cách ánh xạ tự nhiên không được hợp nhất, có ba lối ra: thay đổi luồng nào lấy dữ liệu nào, thay đổi bố cục, hoặc đưa dữ liệu vào bộ nhớ chia sẻ theo thứ tự được hợp nhất rồi thực hiện truy cập khó khăn ở đó (xoay góc). Một lớp tuyến tính có trọng số lưu dạng (out, in) chính là trường hợp đó.
            </li>
          }
        >
          <li>
            When the natural mapping does not coalesce there are three ways out: change which thread takes which data,
            change the layout, or bring the data into shared memory in a coalesced order and do the awkward access there
            (corner turning). A linear layer with its weight stored (out, in) is that case.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              DRAM chỉ cho băng thông của nó khi có nhiều yêu cầu đang chờ phục vụ, trải trên các kênh và bank. Các warp thường trú cung cấp chúng, và các yêu cầu rộng hơn cũng vậy: occupancy và nạp dạng vector là công cụ cho băng thông.
            </li>
          }
        >
          <li>
            DRAM gives its bandwidth only to many requests in flight, spread over channels and banks. Resident warps
            supply them, and so do wider requests: occupancy and vector loads are bandwidth tools.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Bộ nhớ chia sẻ được chia thành các bank, và các lane rơi vào cùng một bank được phục vụ lần lượt từng cái. Một bước nhảy có chung ước số với số bank là thủ phạm thường gặp; một cột đệm dời mỗi hàng đi một bank. Số bank là của NVIDIA và AMD, không phải của Apple.
            </li>
          }
        >
          <li>
            Shared memory is split into banks, and lanes that land in the same bank are served one after another. A stride
            that shares a factor with the bank count is the usual culprit; one column of padding moves each row by one
            bank. The bank counts are NVIDIA&rsquo;s and AMD&rsquo;s, not Apple&rsquo;s.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Gộp việc, trải vòng lặp và buffer kép, mỗi thứ tiêu một thứ (tính song song, thanh ghi, bộ nhớ chia sẻ) để loại bỏ một thứ (chi phí phụ, các lần chờ, một rào chắn). Mỗi thứ chỉ có lợi khi thứ nó loại bỏ đang tốn thời gian.
            </li>
          }
        >
          <li>
            Coarsening, unrolling and double buffering each spend something (parallelism, registers, shared memory) to
            remove something (overhead, stalls, a barrier). Each pays only when what it removes was costing time.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Tìm điểm nghẽn, giải tỏa nó, đo, lặp lại, và dừng khi gần mái. Thời gian chạy của kernel là sự thật, và thiết bị là thứ quyết định: cùng một quy tắc có thể đáng giá gấp nhiều lần trên GPU này và chẳng đáng gì trên GPU khác.
            </li>
          }
        >
          <li>
            Find the bottleneck, relieve it, measure, repeat, and stop near the roof. The kernel&rsquo;s duration is the
            ground truth, and the device decides: the same rule can be worth a factor on one GPU and nothing on another.
          </li>
        </Translated>
      </ul>

      <h2>
        <Translated vi="Bài kiểm tra tổng hợp">A mixed check</Translated>
      </h2>
      <p>
        <Translated vi="Các câu này dùng cả chương cho những tình huống các mục chưa trình bày, phần lớn lấy từ các kernel suy luận.">
          These use the whole chapter on situations the sections did not show, most of them from inference kernels.
        </Translated>
        {asked ? tr(` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.`, ` Đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu hỏi, tính theo câu trả lời gần nhất của bạn cho mỗi câu.`) : ''}
      </p>
      <Beyond>
        <Translated
          vi={
            <p>
              Các kernel và hình dạng trong những câu hỏi này là của chúng tôi. Bố cục key cache trong câu đầu là một lựa chọn khả dĩ, được chọn cho câu hỏi.
            </p>
          }
        >
          <p>
            The kernels and shapes in these questions are ours. The key-cache layout in the first is one possible choice,
            picked for the question.
          </p>
        </Translated>
      </Beyond>
      <Code
        tabs={[
          {
            label: tr('CUDA (not a lab file: for these questions only)', 'CUDA (không phải tệp lab: chỉ dùng cho các câu hỏi này)'),
            code: SCORES,
            status: tr('front-end-checked; never run', 'front-end-checked; chưa từng chạy'),
            tiers: { q: 'global', K: 'global', s: 'global', acc: 'reg', t: 'reg', d: 'reg' },
          },
        ]}
        caption={<Translated vi={<>Từ <code>labs/ch06/snippets/wrap_scores.cu</code>.</>}>From <code>labs/ch06/snippets/wrap_scores.cu</code>.</Translated>}
      />
      <Ask q={q['ch06/wrap/q-scores-sectors']} />
      <Ask q={q['ch06/wrap/q-scores-fix']} />
      <Ask q={q['ch06/wrap/q-permute']} />
      <Ask q={q['ch06/wrap/q-rowalign']} />
      <Ask q={q['ch06/wrap/q-threebuf']} />
      <Ask q={q['ch06/wrap/q-bottleneck']} />

      <h2>
        <Translated vi="Các bài tập của sách">The book&rsquo;s exercises</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương kết thúc với năm bài tập (pp. 154–155). Mỗi bài đều được giải ở đâu đó trong khóa học này; làm thêm theo đúng cách diễn đạt của sách cũng đáng công.
          </p>
        }
      >
        <p>
          The chapter ends with five exercises (pp. 154–155). Each is worked somewhere in this course; doing the
          book&rsquo;s own wording as well is worth it.
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
              <td><Translated vi="Phân loại chín truy cập của một kernel thành được hợp nhất, không được hợp nhất hoặc không áp dụng">Classify nine accesses of a kernel as coalesced, uncoalesced or not applicable</Translated></td>
              <td>
                <Translated vi={<>bài luyện bài tập 1 và ba câu hỏi ôn tập trong <a href={href('ch06', '6.1')}>6.1</a></>}>
                  the exercise 1 drill and three review questions in <a href={href('ch06', '6.1')}>6.1</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td><Translated vi="Một phép nhân ma trận có xoay góc">A matrix multiplication with corner turning</Translated></td>
              <td>
                <Translated vi={<>trình chạy từng bước xoay góc trong <a href={href('ch06', '6.1')}>6.1</a>, rồi lab 6.4 trên <a href={href('ch06', 'labs')}>trang lab</a>, với bố cục trọng số (out, in), cũng là bài toán truy cập đó</>}>
                  the corner-turning stepper in <a href={href('ch06', '6.1')}>6.1</a>, then lab 6.4 on the{' '}
                  <a href={href('ch06', 'labs')}>labs page</a>, in the (out, in) weight layout, which is the same access
                  problem
                </Translated>
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td><Translated vi="Độ rộng ô nào tránh được truy cập không hợp nhất trong kernel của chương 5">Which tile widths avoid uncoalesced accesses in chapter 5&rsquo;s kernel</Translated></td>
              <td>
                <Translated vi={<>câu hỏi và các giá trị T = 8, 16, 32 của trình chạy từng bước trong <a href={href('ch06', '6.1')}>6.1</a>, theo cả hai cách hiểu của bài tập</>}>
                  the question and the stepper&rsquo;s T = 8, 16, 32 in <a href={href('ch06', '6.1')}>6.1</a>, under both
                  readings of the exercise
                </Translated>
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td><Translated vi="Phép cộng vector dùng nạp dạng vector, kèm các biên của nó">Vector-load vector addition with its boundaries</Translated></td>
              <td>
                <Translated vi={<>trình lập kế hoạch đoạn trong <a href={href('ch06', '6.3')}>6.3</a>, rồi lab 6.2 trên <a href={href('ch06', 'labs')}>trang lab</a></>}>
                  the chunk planner in <a href={href('ch06', '6.3')}>6.3</a>, then lab 6.2 on the{' '}
                  <a href={href('ch06', 'labs')}>labs page</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>5</td>
              <td><Translated vi="Bank và xung đột cho bảy bước nhảy">Banks and conflicts for seven strides</Translated></td>
              <td>
                <Translated vi={<>bảng của bài tập 5 và trình khám phá bank trong <a href={href('ch06', '6.4')}>6.4</a>; lab 6.3 đo thời gian cùng các bước nhảy đó</>}>
                  the exercise 5 table and the bank explorer in <a href={href('ch06', '6.4')}>6.4</a>; lab 6.3 times the
                  same strides
                </Translated>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>
        <Translated vi="Chỗ PDF có vẻ sai">Where the PDF looks wrong</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bản sách mà khóa học này dựa vào có vài chỗ có vẻ nhầm và vài điểm còn bỏ ngỏ ở chương 6. Chúng được liệt kê để bạn không bị bối rối khi đọc sách song song.
          </p>
        }
      >
        <p>
          The copy this course was built from has a few apparent slips and loose ends in chapter 6. They are listed so
          that you are not thrown by them when you read the book alongside.
        </p>
      </Translated>
      <Aside>
        <Translated vi={<p>Đây là cách chúng tôi đọc một bản PDF. Nếu bản của bạn khác, hãy tin bản của bạn.</p>}>
          <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
        </Translated>
      </Aside>
      <ul data-list="errata">
        <Translated
          vi={
            <li>
              pp. 140–141 in phép dịch ngược chiều: nhân với 32 là <code>i &lt;&lt; 5</code>, và nhân với 33 là <code>(i &lt;&lt; 5) + i</code>. Như được in, <code>i &gt;&gt; 5 + i</code> là phép dịch phải 5 + i bit. Xem <a href={href('ch06', '6.4')}>6.4</a>.
            </li>
          }
        >
          <li>
            pp. 140–141 print the shift the wrong way round: multiplying by 32 is <code>i &lt;&lt; 5</code>, and by 33 is{' '}
            <code>(i &lt;&lt; 5) + i</code>. As printed, <code>i &gt;&gt; 5 + i</code> is a right shift by 5 + i bits. See{' '}
            <a href={href('ch06', '6.4')}>6.4</a>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 147 nói buffer kép loại bỏ rào chắn &ldquo;thứ hai&rdquo;; theo chính phân tích và mã của sách, rào chắn bị loại bỏ là cái thứ nhất, sau lần đọc. Trong vòng lặp của lab 6.5, vốn nạp trước, thì đó là cái thứ hai. Xem <a href={href('ch06', '6.7')}>6.7</a>.
            </li>
          }
        >
          <li>
            p. 147 says double buffering eliminates the &ldquo;second&rdquo; barrier; the one it removes, by its own
            analysis and code, is the first, after the read. In lab 6.5&rsquo;s loop, which loads first, it is the second.
            See <a href={href('ch06', '6.7')}>6.7</a>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 133 ghi xung nhịp bus HBM3E là &ldquo;4.9 GHz/s&rdquo; (ý là GHz) và kết quả là 1 TB/s. Chính các số liệu đầu vào của sách cho 8 byte × 2 × 4.9 GHz × 16 = 1254.4 GB/s, khoảng 1.25 TB/s, cho một stack. Xem <a href={href('ch06', '6.2')}>6.2</a>.
            </li>
          }
        >
          <li>
            p. 133 gives the HBM3E bus clock as &ldquo;4.9 GHz/s&rdquo; (GHz is meant) and the result as 1 TB/s. Its own
            inputs give 8 bytes × 2 × 4.9 GHz × 16 = 1254.4 GB/s, about 1.25 TB/s, for one stack. See{' '}
            <a href={href('ch06', '6.2')}>6.2</a>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 146 nói mảng cục bộ mặc định nằm trong &ldquo;global memory&rdquo; (bộ nhớ toàn cục); chương 5 gọi chỗ đó là bộ nhớ cục bộ, và NVIDIA cũng vậy, đặt nó về mặt vật lý trong bộ nhớ device và liệt kê khi nào một mảng bị đưa vào đó. Xem <a href={href('ch06', '6.6')}>6.6</a>.
            </li>
          }
        >
          <li>
            p. 146 says local arrays live in &ldquo;global memory&rdquo; by default; chapter 5 calls the place local
            memory, and so does NVIDIA, which puts it physically in device memory and lists when an array goes there.
            See <a href={href('ch06', '6.6')}>6.6</a>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Fig. 6.12 không có kiểm tra biên và không bao giờ dùng tham số <code>N</code>; văn bản để phần biên cho Exercise 4, nên đoạn mã như được in là chưa hoàn chỉnh. Chú thích của nó ghi &ldquo;vector loads&rdquo; (nạp dạng vector), dù dòng 10 cũng là một lần ghi dạng vector.
            </li>
          }
        >
          <li>
            Fig. 6.12 has no bounds checks and never uses its parameter <code>N</code>; the text leaves the boundaries to
            Exercise 4, so the listing is not complete as printed. Its caption says &ldquo;vector loads&rdquo;, though
            line 10 is a vector store as well.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 137 yêu cầu bạn kiểm tra rằng ma trận 8 × 8 &ldquo;use all four channels&rdquo; (dùng cả bốn kênh). Với cách ánh xạ của Fig. 6.9, chúng có dùng trên toàn bộ lượt chạy, nhưng các lần nạp M của mỗi giai đoạn chỉ dùng một kênh, ít kênh cùng lúc hơn trường hợp 4 × 4. Sách không nói ý nó là cái nào.
            </li>
          }
        >
          <li>
            p. 137 asks you to check that 8 × 8 matrices &ldquo;use all four channels&rdquo;. With Fig. 6.9&rsquo;s
            mapping they do across the run, but each phase&rsquo;s M loads use one channel, fewer at once than the 4 × 4
            case. The book does not say which it means.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Chú giải của Fig. 6.8 gọi các đoạn màu nhạt là &ldquo;idle time&rdquo; (thời gian rảnh) trong khi văn bản gọi chúng là độ trễ truy cập, và hình vẽ một tỉ lệ khoảng 2 trên 1 trong khi văn bản làm việc với 20 trên 1.
            </li>
          }
        >
          <li>
            Fig. 6.8&rsquo;s legend calls the light segments &ldquo;idle time&rdquo; while the text calls them access
            latency, and it draws a ratio of about 2 to 1 where the text works with 20 to 1.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Tên gọi không thống nhất: Fig. 6.4 gọi các ô trong bộ nhớ chia sẻ là <code>A_s</code> và <code>B_s</code> cho ma trận M và N, §6.4 dùng <code>a</code> và <code>TILE_DIM</code>, còn Exercise 3 hỏi về một <code>BLOCK_SIZE</code> mà kernel của chương 5 không có (nó dùng <code>TILE_WIDTH</code>).
            </li>
          }
        >
          <li>
            The names drift: Fig. 6.4 calls the shared tiles <code>A_s</code> and <code>B_s</code> for matrices M and N,
            §6.4 uses <code>a</code> and <code>TILE_DIM</code>, and Exercise 3 asks about a <code>BLOCK_SIZE</code> that
            chapter 5&rsquo;s kernel does not have (it uses <code>TILE_WIDTH</code>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Exercise 3 có hai đáp án đều bảo vệ được. Nếu &ldquo;coalesced&rdquo; (được hợp nhất) nghĩa là một dãy 32 địa chỉ mỗi warp, chỉ độ rộng ô 32 là đạt; theo quy tắc sector của NVIDIA, một ô rộng 8 hoặc 16 được căn chỉnh cũng không lãng phí gì. &ldquo;number of bank conflicts&rdquo; (số xung đột bank) của Exercise 5 có thể nghĩa là số lane mỗi bank hoặc số lượt thêm; khóa học này dùng số lane mỗi bank, theo cách dùng &ldquo;k-way&rdquo; của sách.
            </li>
          }
        >
          <li>
            Exercise 3 has two defensible answers. If &ldquo;coalesced&rdquo; means one run of 32 addresses per warp,
            only a tile width of 32 qualifies; under NVIDIA&rsquo;s sector rule an aligned 8- or 16-wide tile wastes nothing
            either. Exercise 5&rsquo;s &ldquo;number of bank conflicts&rdquo; could mean lanes per bank or extra passes;
            this course gives lanes per bank, the book&rsquo;s &ldquo;k-way&rdquo; usage.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Danh sách kiểm tra (Fig. 6.13) và phần văn bản của nó khác nhau ở hai hàng: mục trong bảng của phân ô thanh ghi chỉ nói về bộ nhớ chia sẻ, còn văn bản nói về bộ nhớ toàn cục hoặc bộ nhớ chia sẻ; và mục trong bảng của các primitive cấp warp khẳng định ít lưu lượng toàn cục hơn, điều mà văn bản không giải thích.
            </li>
          }
        >
          <li>
            The checklist (Fig. 6.13) and its text differ in two rows: register tiling&rsquo;s table entry speaks of shared
            memory only, the text of global or shared memory; and warp-level primitives&rsquo; table entry claims less
            global traffic, which the text does not explain.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Chương không bao giờ đưa ra kích thước burst hay dòng cache cho một GPU thật, cũng không nói điều gì xảy ra với các truy cập liên tiếp một phần như bước nhảy 2. Mục <a href={href('ch06', '6.1')}>6.1</a> bổ sung điều đó từ tài liệu của NVIDIA và AMD.
            </li>
          }
        >
          <li>
            The chapter never gives a burst or cache-line size for a real GPU, nor what happens to partly consecutive
            accesses such as a stride of 2. Section <a href={href('ch06', '6.1')}>6.1</a> fills that in from NVIDIA&rsquo;s
            and AMD&rsquo;s documents.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Phần tóm tắt (p. 154) bỏ qua nạp dạng vector và xung đột bank, dù mục 6.3 và 6.4 có trình bày chúng.
            </li>
          }
        >
          <li>
            The summary (p. 154) leaves out vector loads and bank conflicts, although sections 6.3 and 6.4 cover them.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 147 không bao giờ cho thấy <code>inBuffer</code> và <code>outBuffer</code> nằm ở đâu, kiểu của chúng, hay <code>swap</code> là gì; nó phải hoán đổi các con trỏ riêng của từng luồng. Đoạn mã của mục <a href={href('ch06', '6.7')}>6.7</a> viết rõ chúng ra.
            </li>
          }
        >
          <li>
            p. 147 never shows where <code>inBuffer</code> and <code>outBuffer</code> live, their types, or what{' '}
            <code>swap</code> is; it has to exchange each thread&rsquo;s own pointers. Section{' '}
            <a href={href('ch06', '6.7')}>6.7</a>&rsquo;s snippet writes them out.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Fig. 6.4 tô mỗi ma trận bằng một dải chuyển màu mà văn bản không bao giờ giải thích; chúng tôi hiểu nó là chiều tăng của địa chỉ (M và P thứ tự hàng, N thứ tự cột).
            </li>
          }
        >
          <li>
            Fig. 6.4 shades each matrix with a gradient the text never explains; we read it as the direction of
            increasing address (row-major M and P, column-major N).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Lỗi in nhỏ: khung bên được dẫn là &ldquo;Why is DRAM so slow?&rdquo; nhưng có tiêu đề &ldquo;Why are DRAMs so slow?&rdquo; (p. 124); &ldquo;the accesses data wall span&rdquo; (p. 126); &ldquo;Bank 0 or Channel 1&rdquo; thay vì &ldquo;of&rdquo; (p. 135); &ldquo;integer of floating point&rdquo; thay vì &ldquo;or&rdquo; (p. 139); &ldquo;connected a large number of banks&rdquo;, thiếu &ldquo;to&rdquo; (p. 133).
            </li>
          }
        >
          <li>
            Small typos: the sidebar is cited as &ldquo;Why is DRAM so slow?&rdquo; but titled &ldquo;Why are DRAMs so
            slow?&rdquo; (p. 124); &ldquo;the accesses data wall span&rdquo; (p. 126); &ldquo;Bank 0 or Channel 1&rdquo;
            for &ldquo;of&rdquo; (p. 135); &ldquo;integer of floating point&rdquo; for &ldquo;or&rdquo; (p. 139);
            &ldquo;connected a large number of banks&rdquo;, missing &ldquo;to&rdquo; (p. 133).
          </li>
        </Translated>
      </ul>
      <Translated
        vi={
          <p>
            Một trong các phát biểu gắn với thời điểm của sách nay đã bị phủ nhận: &ldquo;one to eight channels&rdquo; (một đến tám kênh) (p. 133) đánh giá thấp các GPU hiện nay. Hai phát biểu khác không xác nhận được theo hướng nào: NVIDIA không đưa ra độ trễ bộ nhớ chia sẻ tính theo chu kỳ, cho câu &ldquo;a few clock cycles&rdquo; (vài chu kỳ xung nhịp) (p. 139), và không trang nào của nhà sản xuất mà chúng tôi đọc đưa ra độ trễ DRAM tính theo nano giây, cho câu &ldquo;tens of nanoseconds&rdquo; (hàng chục nano giây) (p. 124). Cả hai con số đều là của sách.
          </p>
        }
      >
        <p>
          One of the book&rsquo;s dated statements is contradicted today: &ldquo;one to eight channels&rdquo; (p. 133)
          understates current GPUs. Two could not be confirmed either way: NVIDIA gives no shared-memory latency in
          cycles, for &ldquo;a few clock cycles&rdquo; (p. 139), and no vendor page we read gives DRAM latency in
          nanoseconds, for &ldquo;tens of nanoseconds&rdquo; (p. 124). Both figures are the book&rsquo;s.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md A1, A7, A12)">{/* i18n-ignore: date and notes reference */}
        <Translated
          vi={
            <p>
              NVIDIA đưa ra độ trễ bộ nhớ toàn cục là &ldquo;hundreds of clock cycles&rdquo; (hàng trăm chu kỳ xung nhịp). AMD gọi 8 đến 16 bộ điều khiển bộ nhớ độc lập là điển hình.
            </p>
          }
        >
          <p>
            NVIDIA gives global-memory latency as &ldquo;hundreds of clock cycles&rdquo;. AMD calls 8 to 16
            independent memory controllers typical.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Những gì khóa học này rút gọn">What this course shortened</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mỗi mục của chương đều có trang riêng ở đây. Ba đoạn ngắn hơn trong sách: khung bên về lý do DRAM chậm (p. 125) chỉ còn một đoạn văn không kèm hình mạch điện, các ảnh đi chung xe và chính sách giao thông (Figs. 6.5 và 6.6, pp. 131–132) chỉ còn một câu, và các tham chiếu tới từng chương sau của §6.8 (pp. 149–153) trở thành cột &ldquo;returns in&rdquo; (xuất hiện lại ở) của công cụ danh sách kiểm tra. Tài liệu tham khảo duy nhất (p. 155) được thay bằng các ghi chú có ghi ngày bên cạnh mỗi mục. Sao chép bất đồng bộ, các lần nạp rộng hơn 16 byte và hướng dẫn dùng profiler chỉ được nhắc tới; chúng thuộc về chương 15.
          </p>
        }
      >
        <p>
          Every section of the chapter has its own page here. Three passages are shorter than in the book: the sidebar on
          why DRAM is slow (p. 125) is one paragraph without its circuit figure, the carpool photographs and traffic
          policies (Figs. 6.5 and 6.6, pp. 131–132) are one sentence, and the per-chapter forward references of §6.8
          (pp. 149–153) are the checklist tool&rsquo;s &ldquo;returns in&rdquo; column. The one reference (p. 155) is
          replaced by the dated notes beside each section. Asynchronous copies, loads wider than 16 bytes and profiler
          walkthroughs are only mentioned; chapter 15 is where they belong.
        </p>
      </Translated>

      <h2>
        <Translated vi="Dự án cho đến giờ">The project so far</Translated>
      </h2>
      <p>
        <Translated vi={<>Lab 6.4 là mốc dự án M2 &ldquo;đã đo và tinh chỉnh&rdquo;: lớp tuyến tính của dự án, nay với bố cục trọng số mà PyTorch và llama.cpp dùng, có ô trọng số đã xoay góc và đệm, cùng một benchmark so với phiên bản không được hợp nhất.{' '}</>}>
          Lab 6.4 is milestone M2 &ldquo;measured and tuned&rdquo;: the project&rsquo;s linear layer, now in the weight
          layout PyTorch and llama.cpp use, with a corner-turned, padded weight tile and a benchmark against the
          uncoalesced version.{' '}
        </Translated>
        <span data-m2={loading ? 'loading' : m2.join(',') || 'none'}>
          {loading
            ? tr('Reading reports.', 'Đang đọc báo cáo.')
            : backends.length === 0
              ? tr('No lab results are recorded on this machine yet.', 'Chưa có kết quả lab nào được ghi trên máy này.')
              : m2.length
                ? <Translated vi={<><code>linearKernel</code> của bạn đã qua trên {m2.map(name).join(' và ')} trên máy này.</>}><>Your own <code>linearKernel</code> passes on {m2.map(name).join(' and ')} on this machine.</></Translated>
                : <Translated vi={<><code>linearKernel</code> của bạn chưa qua trên máy này.</>}><>Your own <code>linearKernel</code> has not passed on this machine yet.</></Translated>}
        </span>{' '}
        <Translated vi={<>Xem <a href={href('ch06', 'labs')}>các lab</a>. Chương 15 gộp việc và phân ô thanh ghi cho cùng kernel này, với cùng chữ ký, thành mốc dự án M3 (lab 15.6 trên <a href={href('ch15', 'labs')}>trang lab của chương 15</a>).</>}>
          See the <a href={href('ch06', 'labs')}>labs</a>. Chapter 15 coarsens and register-tiles the same kernel, with the
          same signature, into milestone M3 (lab 15.6 on the <a href={href('ch15', 'labs')}>chapter 15 labs page</a>).
        </Translated>
      </p>
      <Translated
        vi={
          <p>
            Chương 7 mở đầu Phần 2 của sách, các mẫu song song, bằng tích chập (convolution). Mỗi đầu ra ở đó cần một vùng lân cận của đầu vào, nên các khối kề nhau cần các ô đầu vào chồng lên nhau, và các ô bạn nạp lớn hơn các ô bạn tạo ra. Chương đó cũng bổ sung bộ nhớ hằng (constant memory), và đánh giá mỗi kernel trong bốn kernel của nó bằng số phép toán trên mỗi byte, phương pháp của chương 5. Bắt đầu ở <a href={href('ch07', 'start')}>chương 7</a>.
          </p>
        }
      >
        <p>
          Chapter 7 opens Part 2 of the book, the parallel patterns, with convolution. Every output there needs a
          neighbourhood of inputs, so neighbouring blocks need overlapping input tiles, and the tiles you load are larger
          than the tiles you produce. It also adds constant memory, and it judges each of its four kernels by operations
          per byte, the method of chapter 5. Start at <a href={href('ch07', 'start')}>chapter 7</a>.
        </p>
      </Translated>
      <Source section="6.8" pages="151, 153" figs={tr("the checklist's forward references to chapter 7", 'các tham chiếu tới chương 7 trong danh sách kiểm tra')} />
    </>
  );
}
