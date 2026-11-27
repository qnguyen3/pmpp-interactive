import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source } from '../../components/notes';
import { defineLabs, preferredBench, whose } from '../../components/LabKit';
import { useProgress } from '../../lib/progress';
import { fmt, labStatus, num, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';

// The mixed check's numbers are ours, chosen for the arithmetic; none is a device figure.
// Host amortisation: kernels 10 ms at batch 1 and 14 ms at batch 32, a fixed 2 ms of host work per step.
export const AMORT = { k1: 10, k32: 14, host: 2, batch: 32 };
export const AMORT_GAIN = (AMORT.batch / (AMORT.k32 + AMORT.host)) / (1 / (AMORT.k1 + AMORT.host)); // 24
// A latency limit: step times per batch, and the most each user may wait for a token.
export const SLO_STEPS: [number, number][] = [
  [8, 20],
  [16, 22],
  [32, 30],
  [64, 55],
];
export const SLO_LIMIT = 40;
const sloBest = SLO_STEPS.filter(([, ms]) => ms <= SLO_LIMIT).reduce((a, b) => (b[0] / b[1] > a[0] / a[1] ? b : a));
export const SLO_TPS = (1000 * sloBest[0]) / sloBest[1]; // 1,066.7 at batch 32
const [lastB, lastMs] = SLO_STEPS[SLO_STEPS.length - 1];
export const SLO_OVER_TPS = (1000 * lastB) / lastMs; // 1,163.6 at batch 64

export const questionsWrap: Record<string, Question> = {
  'ch22/wrap/q-host-amortise': {
    kind: 'numeric',
    id: 'ch22/wrap/q-host-amortise',
    prompt: (
      <Translated
        vi={
          <>
            Các kernel của một bước decode mất {AMORT.k1} ms ở batch 1 và {AMORT.k32} ms ở batch {AMORT.batch}. Mỗi bước còn tốn
            thêm {AMORT.host} ms trên host để khởi chạy chúng, bất kể batch là bao nhiêu, và không có gì chồng lấp. Batch{' '}
            {AMORT.batch} cho số token mỗi giây gấp bao nhiêu lần batch 1?
          </>
        }
      >
        <>
          A decode step's kernels take {AMORT.k1} ms at batch 1 and {AMORT.k32} ms at batch {AMORT.batch}. Every step also
          spends {AMORT.host} ms on the host launching them, whatever the batch, and nothing overlaps. How many times the
          tokens per second of batch 1 does batch {AMORT.batch} give?
        </>
      </Translated>
    ),
    answer: AMORT_GAIN,
    tol: 0.02,
    unit: '×',
    explain: (
      <Translated
        vi={
          <p>
            Batch 1: một token mỗi {AMORT.k1 + AMORT.host} ms. Batch {AMORT.batch}: {AMORT.batch} token mỗi{' '}
            {AMORT.k32 + AMORT.host} ms. Tỷ số là {AMORT.batch} × {AMORT.k1 + AMORT.host} / {AMORT.k32 + AMORT.host} ={' '}
            {fmt(AMORT_GAIN)}. Riêng các kernel sẽ cho {AMORT.batch} × {AMORT.k1} / {AMORT.k32} ={' '}
            {fmt((AMORT.batch * AMORT.k1) / AMORT.k32)}: batching còn chia chi phí cố định của host cho các chuỗi, nên phần mà định
            luật Amdahl nói bạn không thể tăng tốc cũng co lại tính trên mỗi token. Mỗi người dùng vẫn phải chờ lâu hơn cho mỗi
            token, {AMORT.k32 + AMORT.host} ms so với {AMORT.k1 + AMORT.host}.
          </p>
        }
      >
        <p>
          Batch 1: one token per {AMORT.k1 + AMORT.host} ms. Batch {AMORT.batch}: {AMORT.batch} tokens per{' '}
          {AMORT.k32 + AMORT.host} ms. The ratio is {AMORT.batch} × {AMORT.k1 + AMORT.host} / {AMORT.k32 + AMORT.host} ={' '}
          {fmt(AMORT_GAIN)}. The kernels alone would give {AMORT.batch} × {AMORT.k1} / {AMORT.k32} ={' '}
          {fmt((AMORT.batch * AMORT.k1) / AMORT.k32)}: batching also shares the host's fixed cost among the sequences, so
          the part Amdahl's law said you could not speed up shrinks per token too. Each user still waits longer per token,{' '}
          {AMORT.k32 + AMORT.host} ms against {AMORT.k1 + AMORT.host}.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.3, pp. 537–538, and §22.5, p. 539; the numbers are ours',
    vi: { source: 'ch. 22 §22.3, tr. 537–538, và §22.5, tr. 539; các con số là của chúng tôi' },
  },
  'ch22/wrap/q-rope': {
    kind: 'choice',
    id: 'ch22/wrap/q-rope',
    prompt: (
      <Translated
        vi={
          <>
            Một rotary position embedding xoay mỗi cặp (q[2i], q[2i+1]) của một vector query một góc phụ thuộc vào vị trí của
            token và vào i. Một kernel giao cho mỗi luồng một cặp là lấy đầu ra làm trung tâm hay lấy đầu vào làm trung tâm, và
            tiêu chí nào của sách quyết định điều đó?
          </>
        }
      >
        <>
          A rotary position embedding rotates each pair (q[2i], q[2i+1]) of a query vector by an angle that depends on the
          token's position and on i. Is a kernel that gives each thread one pair output-centric or input-centric, and which
          of the book's criteria decides?
        </>
      </Translated>
    ),
    options: [
      {
        label: tr('Input-centric, and atomics decide: a thread that reads a pair scatters its rotation into outputs.', 'Lấy đầu vào làm trung tâm, và atomic quyết định: một luồng đọc một cặp sẽ scatter phép xoay của nó vào các đầu ra.'),
        why: tr('A pair contributes only to the same pair of outputs, and no other thread writes those two elements. There is nothing to collide.', 'Một cặp chỉ đóng góp vào đúng cặp đầu ra tương ứng, và không luồng nào khác ghi vào hai phần tử đó. Không có gì để va chạm.'),
      },
      {
        label: tr('Output-centric, and load balance decides: each output must cost the same.', 'Lấy đầu ra làm trung tâm, và cân bằng tải quyết định: mỗi đầu ra phải tốn như nhau.'),
        why: tr('Every pair does cost the same, one sine, one cosine and four multiplications, whichever way you label the thread. Nothing here is unbalanced.', 'Mọi cặp đúng là tốn như nhau, một sin, một cos và bốn phép nhân, dù bạn gọi luồng theo cách nào. Không có gì mất cân bằng ở đây.'),
      },
      {
        label: tr('Both at once: each thread reads one pair and writes that same pair, so, as with odd-even sort, the label does not matter.', 'Cả hai cùng lúc: mỗi luồng đọc một cặp và ghi đúng cặp đó, nên, như với odd-even sort, cách gọi tên không quan trọng.'),
        correct: true,
      },
      {
        label: tr('Output-centric, and parallelism decides: one thread per output element instead of per pair gives the device twice as many threads.', 'Lấy đầu ra làm trung tâm, và mức song song quyết định: một luồng cho mỗi phần tử đầu ra thay vì mỗi cặp cho thiết bị gấp đôi số luồng.'),
        why: tr('Twice the threads, each reading both inputs of its pair and computing the same sine and cosine as its partner: the extra threads redo work. And each still writes only its own element, so the split changes the grain, not the decomposition.', 'Gấp đôi số luồng, mỗi luồng đọc cả hai đầu vào của cặp mình và tính cùng sin và cos như luồng bạn của nó: các luồng thêm vào làm lại việc. Và mỗi luồng vẫn chỉ ghi phần tử của riêng nó, nên việc chia nhỏ thay đổi độ mịn, không thay đổi cách phân rã.'),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Các tiêu chí của sách (tr. 533) chọn giữa hai cách phân rã khác nhau ở chỗ ai ghi. Khi mỗi đầu vào chỉ cấp cho đúng các
            đầu ra ở chính vị trí của nó, hai cách trùng nhau, đó là nhận xét của sách về odd-even sort. Câu hỏi còn lại là độ mịn:
            mỗi luồng một cặp giữ hai lần đọc và góc dùng chung trong một luồng. RoPE là ví dụ của khóa học này, không phải của
            sách.
          </p>
        }
      >
        <p>
          The book's criteria (p. 533) choose between two decompositions that differ in who writes. When every input feeds
          exactly the outputs at its own position, the two coincide, which is the book's remark about odd-even sort. The
          question left is the grain: a pair per thread keeps the two reads and the shared angle in one thread. RoPE is the
          course's example, not the book's.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.2, p. 533 (odd-even sort); the rotary embedding example is ours',
    vi: { source: 'ch. 22 §22.2, tr. 533 (odd-even sort); ví dụ rotary embedding là của chúng tôi' },
  },
  'ch22/wrap/q-greedy': {
    kind: 'choice',
    id: 'ch22/wrap/q-greedy',
    prompt: (
      <Translated
        vi={
          <>
            Một bộ decode greedy tính softmax trên 128,000 logits rồi lấy argmax của các xác suất. Thay đổi nào bỏ bớt việc mà
            không làm token được chọn kém chính xác đi chút nào?
          </>
        }
      >
        <>
          A greedy decoder computes a softmax over 128,000 logits and then the argmax of the probabilities. Which change
          removes work without making the chosen token any less exact?
        </>
      </Translated>
    ),
    options: [
      { label: tr('Take the argmax of the logits and drop the softmax: it keeps the order of the values.', 'Lấy argmax của logits và bỏ softmax: nó giữ nguyên thứ tự của các giá trị.'), correct: true },
      {
        label: tr('Compute the softmax in 16 bits: it halves the bytes it moves, and the argmax reads only the order of the values.', 'Tính softmax bằng 16 bit: nó giảm một nửa số byte phải di chuyển, và argmax chỉ đọc thứ tự của các giá trị.'),
        why: tr('It does halve the bytes, but rounding to 16 bits can turn two close probabilities into the same value, and a tie then goes to the lower index, which may be the smaller logit.', 'Nó đúng là giảm một nửa số byte, nhưng làm tròn về 16 bit có thể biến hai xác suất gần nhau thành cùng một giá trị, và khi hòa thì chỉ số nhỏ hơn được chọn, mà đó có thể là logit nhỏ hơn.'),
      },
      {
        label: tr('Select the top 50 first, then take the argmax of those.', 'Chọn top 50 trước, rồi lấy argmax của chúng.'),
        why: tr('The largest of the top 50 is the largest of all, so the token is the same, but a top-k selection does strictly more work than the argmax it contains.', 'Giá trị lớn nhất trong top 50 là lớn nhất trong tất cả, nên token vẫn như cũ, nhưng một phép chọn top-k làm nhiều việc hơn hẳn phép argmax nằm bên trong nó.'),
      },
      {
        label: tr('Skip the LM-head rows of tokens that are rarely chosen.', 'Bỏ qua các hàng LM-head của những token hiếm khi được chọn.'),
        why: tr('Nothing tells you which logits will not be the largest without computing them. This one changes tokens.', 'Không có gì cho bạn biết logit nào sẽ không phải lớn nhất nếu không tính chúng. Cách này làm thay đổi token.'),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Câu hỏi đặt ra cho bộ decode là token nào lớn nhất, không phải mỗi xác suất bằng bao nhiêu. Hàm mũ và mẫu số chung không
            bao giờ đảo thứ tự hai giá trị, nên trong số học chính xác đáp án là như nhau; với số thực dấu phẩy động, softmax chỉ có
            thể thêm sai số làm tròn, có thể khiến hai logit gần nhau thành hai xác suất bằng nhau. Đặt lại câu hỏi sao cho một phép
            tính rẻ hơn trả lời được nó chính là phát biểu bài toán (mục 22.4) ở quy mô nhỏ. Lab 22.2 vẫn tính các xác suất, vì bộ
            lấy mẫu của mốc dự án M5 đọc chúng.
          </p>
        }
      >
        <p>
          The question asked of the decoder is which token is largest, not what each probability is. The exponential and the
          shared denominator never reverse two values, so in exact arithmetic the answer is the same; in floats, the softmax
          can only add rounding, which can make two close logits equal probabilities. Re-posing the question so that a
          cheaper computation answers it is problem formulation (section 22.4) at a small scale. Lab 22.2 still computes the
          probabilities, because the sampler of milestone M5 reads them.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.4, p. 538 (re-posing the problem); the example is ours; labs/ch22/lab2_batching/README.md',
    vi: { source: 'ch. 22 §22.4, tr. 538 (đặt lại bài toán); ví dụ là của chúng tôi; labs/ch22/lab2_batching/README.md' },
  },
  'ch22/wrap/q-slo': {
    kind: 'numeric',
    id: 'ch22/wrap/q-slo',
    prompt: (
      <Translated
        vi={
          <>
            Các bước decode đo được:{' '}
            {SLO_STEPS.map(([b, ms], i) => (
              <span key={b}>
                {i ? ', ' : ''}batch {b} mất {ms} ms
              </span>
            ))}
            . Mỗi người dùng phải nhận được một token ít nhất mỗi {SLO_LIMIT} ms. Số token mỗi giây lớn nhất bạn có thể phục vụ
            trong giới hạn đó là bao nhiêu?
          </>
        }
      >
        <>
          Measured decode steps:{' '}
          {SLO_STEPS.map(([b, ms], i) => (
            <span key={b}>
              {i ? ', ' : ''}batch {b} takes {ms} ms
            </span>
          ))}
          . Each user must get a token at least every {SLO_LIMIT} ms. What is the most tokens per second you can serve within
          that limit?
        </>
      </Translated>
    ),
    answer: Math.round(SLO_TPS),
    tol: 0.02,
    unit: 'tokens/s',
    vi: { unit: 'token/s', source: 'ch. 22 §22.5, tr. 539; các con số là của chúng tôi' },
    explain: (
      <Translated
        vi={
          <p>
            Thời gian của một bước là thời gian mỗi token của từng người dùng. Batch {sloBest[0]} mất {sloBest[1]} ms, nằm trong
            giới hạn: {sloBest[0]} token mỗi {sloBest[1]} ms là {fmt(SLO_TPS)} token mỗi giây. Batch {lastB} sẽ phục vụ được{' '}
            {fmt(SLO_OVER_TPS)}, nhưng mỗi người dùng sẽ phải chờ {lastMs} ms. Đây là ví dụ công ty đầu tư có khung thời gian của
            sách: yêu cầu, chứ không phải phần cứng, chọn điểm trên đường cong. Các server thể hiện điều này thành một mức trần cho
            batch (mục 22.5).
          </p>
        }
      >
        <p>
          A step's time is each user's time per token. Batch {sloBest[0]} takes {sloBest[1]} ms, within the limit:{' '}
          {sloBest[0]} tokens every {sloBest[1]} ms is {fmt(SLO_TPS)} tokens per second. Batch {lastB} would serve{' '}
          {fmt(SLO_OVER_TPS)}, but every user would wait {lastMs} ms. This is the book's investment firm with a time window:
          the requirement, not the hardware, picks the point on the curve. Servers expose it as a cap on the batch (section
          22.5).
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.5, p. 539; the numbers are ours',
  },
  'ch22/wrap/q-router': {
    kind: 'choice',
    id: 'ch22/wrap/q-router',
    prompt: (
      <Translated vi="Ở batch 1, bạn kỳ vọng split-K hơn nhiều nhất so với gather một warp mỗi đầu ra ở lớp nào?">
        <>At batch 1, on which layer would you expect split-K to gain the most over a gather with one warp per output?</>
      </Translated>
    ),
    options: [
      {
        label: tr('An LM head, 4096 → 128,000.', 'Một LM head, 4096 → 128,000.'),
        why: tr('Gather already has 128,000 warps there. Splitting adds atomics and buys parallelism the device cannot use.', 'Ở đó gather đã có 128,000 warp. Chia nhỏ thêm atomic và mua thêm mức song song mà thiết bị không dùng được.'),
      },
      { label: tr('A mixture-of-experts router, 4096 → 8.', 'Một router mixture-of-experts, 4096 → 8.'), correct: true },
      {
        label: tr('An MLP up-projection, 4096 → 14,336.', 'Một phép chiếu lên (up-projection) của MLP, 4096 → 14,336.'),
        why: tr('Gather has 14,336 warps: enough, on most devices, to keep many loads in flight.', 'Gather có 14,336 warp: trên hầu hết thiết bị, đủ để giữ nhiều lệnh nạp đang chạy cùng lúc.'),
      },
      {
        label: tr('None more than another: a matrix-vector product is limited by the bytes of W, and split-K reads the same bytes.', 'Không lớp nào hơn lớp nào: một phép nhân ma trận-vector bị giới hạn bởi số byte của W, và split-K đọc đúng chừng đó byte.'),
        why: tr('The bytes are the same, but memory delivers them fast only when enough warps have loads in flight, and eight warps leave almost all of a GPU idle.', 'Số byte là như nhau, nhưng bộ nhớ chỉ giao chúng nhanh khi đủ warp có lệnh nạp đang chạy, và tám warp để gần như cả GPU rảnh rỗi.'),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mức song song của gather là số đầu ra. Với 8 đầu ra, 8 warp gánh cả lớp, còn S × 8 warp của split-K đọc cùng số byte đó
            đồng thời hơn nhiều, với cái giá là S phép cộng nguyên tử cho mỗi đầu ra: sự đánh đổi của sách giữa mức song song được
            bộc lộ và atomic (tr. 533). Lab 22.1 đã đo một lớp có 16 đầu ra.
          </p>
        }
      >
        <p>
          Gather's parallelism is the number of outputs. With 8 outputs, 8 warps carry the whole layer, and split-K's{' '}
          S × 8 warps read the same bytes far more concurrently, at the price of S atomic additions per output: the book's
          trade between exposed parallelism and atomics (p. 533). Lab 22.1 measured a layer of 16 outputs.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.2, p. 533; lab 22.1; the layer shapes are ours',
    vi: { source: 'ch. 22 §22.2, tr. 533; lab 22.1; các kích thước lớp là của chúng tôi' },
  },
};

// ---------------------------------------------------------------- this machine's lab 22.1 numbers for the router question
const [GEMV] = defineLabs('ch22', [{ num: '22.1', dir: 'lab1_gemv', name: 'gather and split-K', vi: { name: 'gather và split-K' } }]);
const [M8] = defineLabs('ch22', [{ num: '22.2', dir: 'lab2_batching', name: 'batching study (M8)', vi: { name: 'nghiên cứu batching (M8)' } }]);

function OnThisMachine() {
  const { reports, loading } = useReports();
  if (loading) return null;
  const lines = preferredBench(reports, GEMV).flatMap((s) => {
    const rows = s.rows.filter((r) => r.name === 'gemv' && num(r.in) === 4096 && num(r.out) === 16 && num(r.batch) === 1 && num(r.seconds));
    const gather = rows.find((r) => r.kernel === 'gather');
    const best = rows.filter((r) => r.kernel === 'splitk').sort((a, b) => num(a.seconds)! - num(b.seconds)!)[0];
    if (!gather || !best) return [];
    return [{ s, ratio: num(gather.seconds)! / num(best.seconds)!, splits: String(best.splits) }];
  });
  return lines.length ? (
    <>
      {lines.map(({ s, ratio, splits }) => (
        <Translated
          vi={
            <p key={s.backend} data-machine={s.backend} data-ratio={fmt(ratio)}>
              Số liệu của chính bạn, {s.device} ({s.backend}), {whose(s)}: trên lớp 4096 → 16 của lab 22.1 ở batch 1, split-K nhanh
              nhất (S = {splits}) chạy nhanh gấp <b>{fmt(ratio)}</b> lần gather.{' '}
              <a href={href('ch22', 'labs')}>Trang lab</a> có mọi kích thước.
            </p>
          }
        >
          <p key={s.backend} data-machine={s.backend} data-ratio={fmt(ratio)}>
            Your own numbers, {s.device} ({s.backend}), {whose(s)}: on lab 22.1's 4096 → 16 layer at batch 1, the fastest
            split-K (S = {splits}) ran <b>{fmt(ratio)}</b> times as fast as gather. The{' '}
            <a href={href('ch22', 'labs')}>labs page</a> has every shape.
          </p>
        </Translated>
      ))}
    </>
  ) : (
    <Translated
      vi={
        <p className="muted" data-machine="none">
          Để đặt thiết bị của bạn cạnh câu hỏi này, bạn cần các dòng benchmark của lab 22.1:{' '}
          <code>python3 labs/pmpp.py run 22.1 --bench</code>.
        </p>
      }
    >
      <p className="muted" data-machine="none">
        To put your own device beside this question you need lab 22.1's benchmark rows:{' '}
        <code>python3 labs/pmpp.py run 22.1 --bench</code>.
      </p>
    </Translated>
  );
}

// ---------------------------------------------------------------- the page
export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports);
  const where = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  const passing = (id: string) => [...backends, 'emu'].filter((b) => labStatus(reports[b], id, 'work') === 'pass');
  const m8 = passing(M8.id);
  const gemv = passing(GEMV.id);
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch22/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  const none = backends.length === 0 && !reports['emu'];
  return (
    <>
      <h2 style={{ marginTop: 0 }}><Translated vi="Những điều cần mang theo">What to carry forward</Translated></h2>
      <Translated
        vi={
          <ul>
            <li>
              Bốn quyết định đi trước mọi tối ưu hóa: đặt bài toán thế nào, dùng thuật toán nào, phân rã nó thành việc cho các
              luồng ra sao, và chỉ sau đó mới đến làm cho nó nhanh thế nào.
            </li>
            <li>
              Các thuật toán cho cùng một bài toán đánh đổi số bước, mức song song được bộc lộ, tính tổng quát và độ chính xác.
              Không thuật toán nào thắng cả bốn; lựa chọn đúng phụ thuộc vào phần cứng, và các dạng lai như gộp việc cho luồng
              (thread coarsening) làm mềm sự đánh đổi.
            </li>
            <li>
              Một luồng có thể sở hữu một đầu ra và gather, hoặc sở hữu một đầu vào và scatter. Gather tích lũy trong thanh ghi
              (register) và không cần atomic; scatter cần atomic ở bất cứ đâu hai luồng có thể ghi cùng một đầu ra. Atomic chỉ là
              một tiêu chí: mức song song được bộc lộ, độ dễ ánh xạ đầu vào sang đầu ra, và cân bằng tải cũng quyết định.
            </li>
            <li>
              Tốc độ tăng của một ứng dụng bị giới hạn bởi phần vẫn chậm. Chuyển phần còn lại lên device, cho nó chồng lấp, hoặc
              chạy các mảnh nhỏ song song cạnh nhau; khi decode, hãy giữ vòng lặp token trên device.
            </li>
            <li>
              Những cải thiện lớn nhất đến từ việc đặt lại bài toán, và chúng phải trả giá: độ chính xác, bộ nhớ, hoặc một lập
              luận chuyên ngành về lý do phép xấp xỉ là an toàn.
            </li>
            <li>
              Batching đổi độ trễ của từng yêu cầu lấy thông lượng của cả nhóm. Trọng số của một bước decode được đọc một lần mỗi
              bước, nên dưới điểm gãy các chuỗi thêm vào đi kèm gần như miễn phí; một giới hạn độ trễ, chứ không phải phần cứng,
              cho biết nên đi xa đến đâu.
            </li>
            <li>
              Không nhà cung cấp nào đưa ra con số so sánh atomic với thanh ghi, và điểm gãy phụ thuộc vào thiết bị của bạn. Hãy dự
              đoán từ mô hình, rồi đo.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            Four decisions come before any optimisation: how to pose the problem, which algorithm to use, how to decompose it
            into work for threads, and only then how to make that fast.
          </li>
          <li>
            Algorithms for one problem trade steps, exposed parallelism, generality and accuracy. None wins on all four; the
            right one depends on the hardware, and hybrids such as coarsening soften the trade.
          </li>
          <li>
            A thread can own an output and gather, or own an input and scatter. Gather accumulates in registers and needs no
            atomics; scatter needs one wherever two threads can write one output. Atomics are only one criterion: exposed
            parallelism, how easily inputs map to outputs, and load balance decide too.
          </li>
          <li>
            An application's speed-up is capped by what stays slow. Move the remainder onto the device, overlap it, or run
            the small pieces side by side; in decoding, keep the token loop on the device.
          </li>
          <li>
            The largest gains come from re-posing the problem, and they cost something: exactness, memory, or a domain
            argument for why the approximation is safe.
          </li>
          <li>
            Batching trades each request's latency for the group's throughput. A decode step's weights are read once per
            step, so below the knee extra sequences ride almost free; a latency limit, not the hardware, says how far to go.
          </li>
          <li>
            No vendor puts a number on atomics against registers, and the knee depends on your device. Predict from the
            model, then measure.
          </li>
        </ul>
      </Translated>
      <Source section="22.6" pages="540" />

      <h2><Translated vi="Bài kiểm tra tổng hợp">A mixed check</Translated></h2>
      <Translated
        vi={
          <p>
            Các câu này dựa trên cả chương và trên những tình huống mà các trang chưa trình bày.
            {asked ? ` Cho đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu, tính theo câu trả lời gần nhất của bạn cho mỗi câu.` : ''}
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
            <p>Mọi con số và mọi ví dụ trong năm câu hỏi này đều là của chúng tôi.</p>
          }
        >
          <p>Every number and every example in these five questions is ours.</p>
        </Translated>
      </Beyond>
      <Ask q={q['ch22/wrap/q-host-amortise']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={q['ch22/wrap/q-rope']} />
      <Ask q={q['ch22/wrap/q-greedy']} />
      <Ask q={q['ch22/wrap/q-slo']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={q['ch22/wrap/q-router']} />
      <OnThisMachine />

      <h2><Translated vi="Bài tập của sách">The book's exercises</Translated></h2>
      <Translated
        vi={
          <p>
            Chương này không có bài tập nào: mục lục của nó đi thẳng từ phần tóm tắt đến tài liệu tham khảo duy nhất (tr. 529). Phần
            thực hành nằm ở các câu hỏi trên mỗi trang, các trình khám phá, và hai lab. Nơi các ví dụ của chính sách được làm:
          </p>
        }
      >
        <p>
          This chapter has none: its contents go from the summary straight to the one reference (p. 529). The practice is in
          the questions on each page, the explorers, and the two labs. Where the book's own examples are worked:
        </p>
      </Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }} data-table="examples">
          <thead>
            <tr>
              <th><Translated vi="Ví dụ của sách">The book's example</Translated></th>
              <th><Translated vi="Trang">Pages</Translated></th>
              <th><Translated vi="Được làm ở đâu trong khóa học này">Where it is worked here</Translated></th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><Translated vi="Brent-Kung so với Kogge-Stone; odd-even sort, radix sort và merge sort; tổng trực tiếp so với tổng cutoff">Brent-Kung against Kogge-Stone; odd-even, radix and merge sort; direct against cutoff summation</Translated></td>
              <td>530–531</td>
              <td>
                <Translated vi={<>trình khám phá scan và các câu hỏi ở <a href={href('ch22', '22.1')}>22.1</a></>}>
                  the scan explorer and questions on <a href={href('ch22', '22.1')}>22.1</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td><Translated vi="Cutoff với bin và danh sách tràn">Cutoff binning and the overflow list</Translated></td>
              <td>531</td>
              <td>
                <Translated vi={<>trình khám phá sức chứa bin ở <a href={href('ch22', '22.1')}>22.1</a></>}>
                  the bin-capacity explorer on <a href={href('ch22', '22.1')}>22.1</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td><Translated vi="Lấy đầu ra làm trung tâm so với lấy đầu vào làm trung tâm (Hình 22.1)">Output-centric against input-centric (Fig. 22.1)</Translated></td>
              <td>532–533</td>
              <td>
                <Translated
                  vi={
                    <>
                      hoạt hình gather và scatter, trình khám phá phân rã và trình chạy từng bước split-K ở{' '}
                      <a href={href('ch22', '22.2')}>22.2</a>; lab 22.1
                    </>
                  }
                >
                  the gather and scatter animation, the decomposition explorer and the split-K stepper on{' '}
                  <a href={href('ch22', '22.2')}>22.2</a>; lab 22.1
                </Translated>
              </td>
            </tr>
            <tr>
              <td><Translated vi="Chuyến điểm qua mọi mẫu trong sách">The tour of every pattern in the book</Translated></td>
              <td>533–536</td>
              <td>
                <Translated vi={<>các thẻ phân loại ở <a href={href('ch22', '22.2')}>22.2</a></>}>
                  the classification cards on <a href={href('ch22', '22.2')}>22.2</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td><Translated vi="Động lực học phân tử, 95% ở 100× (Hình 22.2); song song mức tác vụ">Molecular dynamics, 95% at 100× (Fig. 22.2); task-level parallelism</Translated></td>
              <td>537–538</td>
              <td>
                <Translated vi={<>trình khám phá module và trình khám phá stream ở <a href={href('ch22', '22.3')}>22.3</a></>}>
                  the module explorer and the streams explorer on <a href={href('ch22', '22.3')}>22.3</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td><Translated vi="Cutoff với bin như một phép phát biểu bài toán">Cutoff binning as problem formulation</Translated></td>
              <td>538</td>
              <td>
                <Translated vi={<>trình khám phá phát biểu bài toán ở <a href={href('ch22', '22.4')}>22.4</a></>}>
                  the formulation explorer on <a href={href('ch22', '22.4')}>22.4</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td><Translated vi="Batching các truy vấn LLM; hai công thức thông lượng; CPU so với GPU">Batching LLM queries; the two throughput formulas; CPU against GPU</Translated></td>
              <td>539–540</td>
              <td>
                <Translated vi={<>trình khám phá batching và hai hoạt hình ở <a href={href('ch22', '22.5')}>22.5</a>; lab 22.2</>}>
                  the batching explorer and the two animations on <a href={href('ch22', '22.5')}>22.5</a>; lab 22.2
                </Translated>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2><Translated vi="Những chỗ PDF có vẻ sai">Where the PDF looks wrong</Translated></h2>
      <Translated
        vi={
          <p>
            Bản sách mà khóa học này dựa vào có vài chỗ có vẻ nhầm lẫn ở chương 22. Chúng được liệt kê ra để bạn không bị bối rối
            khi đọc sách song song.
          </p>
        }
      >
        <p>
          The copy of the book this course was built from has a few apparent slips in chapter 22. They are listed so that you
          are not thrown by them when you read the book alongside.
        </p>
      </Translated>
      <Translated
        vi={
          <ul data-list="errata">
            <li>
              Tr. 534 dẫn "scan (Chapter 12)". Scan là chương 11, như chính chương này nói ở tr. 530 (
              <a href={href('ch22', '22.2')}>22.2</a>).
            </li>
            <li>
              Tr. 536 nói bản đồ tĩnh điện ưu tiên phân rã lấy đầu ra làm trung tâm "because of the benefit over the gather
              pattern of the scatter pattern". Phần còn lại của đoạn văn mang nghĩa ngược lại: lợi thế của gather so với scatter (
              <a href={href('ch22', '22.2')}>22.2</a>).
            </li>
            <li>
              Tr. 537 in 1 / 5.95% thành 17×. Đúng ra là 16.8×; 20× khi có chồng lấp là chính xác (
              <a href={href('ch22', '22.3')}>22.3</a>).
            </li>
            <li>
              Tr. 537 để kernel tính lực "for all the grid points" trong một ví dụ về nguyên tử, và dùng lẫn "non-bonded" với
              "non-bonding" cho cùng một loại lực (<a href={href('ch22', '22.3')}>22.3</a>).
            </li>
            <li>
              Tr. 531 không nói các nguyên tử tràn được xử lý ở đâu sau kernel. Chương 21 có nói: trên host (tr. 527;{' '}
              <a href={href('ch22', '22.1')}>22.1</a>).
            </li>
            <li>
              Tr. 539 liệt kê "double buffering, warp-level shuffling register tiling, and look-back". Mục ở giữa đọc như hai kỹ
              thuật bị viết dính vào nhau; <a href={href('ch22', '22.5')}>22.5</a> đưa ra cách hiểu của chúng tôi.
            </li>
            <li>
              Phần văn của tr. 539 nói phép nhân "vector-matrix" còn công thức thông lượng của nó nói "matrix-vector". Đó là cùng
              một phép toán.
            </li>
            <li>
              Tr. 539 nói batching làm tăng cường độ tính toán "of the attention heads". Chương 20 (tr. 505) nói attention không
              được lợi từ batching. Cả hai đều đúng nếu "attention" ở tr. 539 bao gồm phép chiếu QKV, như câu tiếp theo của nó nói:
              attention thực sự đọc key và value riêng của từng yêu cầu và không được lợi gì (
              <a href={href('ch22', '22.5')}>22.5</a>).
            </li>
            <li>
              Các lỗi nhỏ: "well-define comparison operator" (tr. 530), "due to it scatter memory access behavior" (tr. 531), "by
              each thread or collection" (tr. 532). Hai khung của Hình 22.1 có số ô khác nhau và tô màu các hàng khác nhau, không
              có chú giải; các con số đếm có vẻ ngẫu nhiên.
            </li>
          </ul>
        }
      >
        <ul data-list="errata">
          <li>
            p. 534 cites "scan (Chapter 12)". Scan is chapter 11, as the same chapter says on p. 530 (
            <a href={href('ch22', '22.2')}>22.2</a>).
          </li>
          <li>
            p. 536 says the electrostatic map favours output-centric decomposition "because of the benefit over the gather
            pattern of the scatter pattern". The rest of the paragraph means the reverse: gather's benefit over scatter (
            <a href={href('ch22', '22.2')}>22.2</a>).
          </li>
          <li>
            p. 537 prints 1 / 5.95% as 17×. It is 16.8×; 20× with overlap is exact (
            <a href={href('ch22', '22.3')}>22.3</a>).
          </li>
          <li>
            p. 537 has the kernel compute forces "for all the grid points" in an example about atoms, and alternates
            "non-bonded" and "non-bonding" for one kind of force (<a href={href('ch22', '22.3')}>22.3</a>).
          </li>
          <li>
            p. 531 does not say where the overflow atoms are processed after the kernel. Chapter 21 does: on the host (p.
            527; <a href={href('ch22', '22.1')}>22.1</a>).
          </li>
          <li>
            p. 539 lists "double buffering, warp-level shuffling register tiling, and look-back". The middle item reads like
            two techniques run together; <a href={href('ch22', '22.5')}>22.5</a> gives our reading.
          </li>
          <li>
            p. 539's prose says "vector-matrix" multiplication and its throughput formula "matrix-vector". It is one operation.
          </li>
          <li>
            p. 539 says batching raises the arithmetic intensity "of the attention heads". Chapter 20 (p. 505) says attention
            does not benefit from batching. Both hold if p. 539's "attention" includes the QKV projection, as its next sentence
            says: attention proper reads each request's own keys and values and gains nothing (
            <a href={href('ch22', '22.5')}>22.5</a>).
          </li>
          <li>
            Small slips: "well-define comparison operator" (p. 530), "due to it scatter memory access behavior" (p. 531),
            "by each thread or collection" (p. 532). Fig. 22.1's two panels have different numbers of cells and colour
            different rows, with no legend; the counts look incidental.
          </li>
        </ul>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>Đây là cách đọc của chúng tôi trên một bản PDF (notes/ch22.md). Nếu bản của bạn khác, hãy tin bản của bạn.</p>
          }
        >
          <p>These are our readings of one PDF (notes/ch22.md). If your copy differs, trust your copy.</p>
        </Translated>
      </Aside>

      <h2><Translated vi="Những gì khóa học này bỏ qua">What this course left out</Translated></h2>
      <Translated
        vi={
          <p>
            Các loại lực động lực học phân tử của Hình 22.2 trở thành các hộp module; phần vật lý và các kernel của bản đồ tĩnh điện
            ở lại <a href={href('ch21')}>chương 21</a>. Các định dạng ma trận thưa và các biến thể BFS trong chuyến điểm qua chỉ
            xuất hiện dưới dạng thẻ phân loại; chúng được làm ở chương 17 và 18. Stream chỉ được mô hình hóa như nhiều kernel nhỏ
            chạy cùng lúc; API của chúng thuộc chương 23. Định nghĩa thuật toán (tr. 530) và ví dụ công ty đầu tư (tr. 539) mỗi thứ
            được một đoạn, còn những nhận xét khép lại về cách con người học (tr. 540) được một câu. Attention bị bỏ khỏi bước của lab
            22.2, còn lấy mẫu và trọng số lượng tử hóa mỗi thứ có một câu hỏi: chúng thuộc các mốc dự án M6 và M7, M5 và M9.
          </p>
        }
      >
        <p>
          The molecular-dynamics force types of Fig. 22.2 become module boxes; the electrostatic map's physics and kernels
          stay in <a href={href('ch21')}>chapter 21</a>. The sparse-matrix formats and BFS variants of the tour appear only
          as classification cards; they are worked in chapters 17 and 18. Streams are modelled only as several small kernels
          at once; their API is chapter 23's. The definition of an algorithm (p. 530) and the investment-firm example (p.
          539) get a paragraph each, and the closing remarks on how people learn (p. 540) one sentence. Attention is left out
          of lab 22.2's step, and sampling and quantised weights get one question each: they belong to milestones M6 and
          M7, M5 and M9.
        </p>
      </Translated>

      <h2><Translated vi="Dự án cho đến giờ">The project so far</Translated></h2>
      <Translated
        vi={
          <p>
            Bước của lab 22.2 là mốc dự án M8: mốc đầu tiên chạy trọn một bước decode, được xây từ M0, M2, M3, M4 và gather của lab
            22.1.{' '}
            <span data-m8={m8.join(',') || 'none'}>
              {none
                ? 'Chưa có kết quả lab nào được ghi lại trên máy này.'
                : m8.length
                  ? `Phiên bản của bạn qua được trên ${m8.map(where).join(' và ')} trên máy này.`
                  : 'Phiên bản của bạn chưa qua được trên máy này.'}
            </span>{' '}
            <span data-lab221={gemv.join(',') || 'none'}>
              {none
                ? ''
                : gemv.length
                  ? `Gather của lab 22.1 của bạn qua được trên ${gemv.map(where).join(' và ')}.`
                  : 'Gather của lab 22.1 của bạn chưa qua được trên máy này.'}
            </span>{' '}
            Xem <a href={href('ch22', 'labs')}>các lab</a>.
          </p>
        }
      >
        <p>
          Lab 22.2's step is milestone M8: the first milestone that runs a whole decode step, built from M0, M2, M3, M4 and
          lab 22.1's gather.{' '}
          <span data-m8={m8.join(',') || 'none'}>
            {none
              ? 'No lab results are recorded on this machine yet.'
              : m8.length
                ? `Your own version passes on ${m8.map(where).join(' and ')} on this machine.`
                : 'Your own version has not passed on this machine yet.'}
          </span>{' '}
          <span data-lab221={gemv.join(',') || 'none'}>
            {none
              ? ''
              : gemv.length
                ? `Your lab 22.1 gather passes on ${gemv.map(where).join(' and ')}.`
                : 'Your lab 22.1 gather has not passed on this machine yet.'}
          </span>{' '}
          See the <a href={href('ch22', 'labs')}>labs</a>.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chương 23 tiếp nối chỗ mục 22.3 dừng ở "nhiều kernel nhỏ cùng lúc": stream, sao chép bất đồng bộ và event, được dạy trên
            một GPU (<a href={href('ch23', '23.3')}>23.3</a>). Trong dự án, module X2 thêm trọng số lượng tử hóa, mốc dự án M9, thứ
            dời điểm gãy bạn vừa đo bằng cách thay đổi số byte mà mỗi trọng số tốn.
          </p>
        }
      >
        <p>
          Chapter 23 takes up what section 22.3 left at "several small kernels at once": streams, asynchronous copies and
          events, taught on one GPU (<a href={href('ch23', '23.3')}>23.3</a>). In the project, module X2 adds quantised
          weights, milestone M9, which moves the knee you just measured by changing the bytes each weight costs.
        </p>
      </Translated>
    </>
  );
}
