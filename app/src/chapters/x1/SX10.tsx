import type { CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Beyond, NoteGroup, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { DRAW, GREEDY, M5_ORDER, VOCAB, rowBytes, stage, type KernelStep } from '../../sim/chain';

// Module X1's start page: where the sampler sits, the chain as a map of the earlier labs
// that supply its kernels (drawn from chain.ts's STAGES, the same list the X1.1 explorer
// runs), what the reader's own stack does today, the size of the problem, what "same seed,
// same token" will mean, and a prerequisite check drawn from chapters 9 to 14.

const n = (v: number) => v.toLocaleString('en-US');
const FACTS = 'notes/today/x1-facts.md';

export const questionsSX10: Record<string, Question> = {
  'x1/start/q-max-subtract': {
    kind: 'choice',
    id: 'x1/start/q-max-subtract',
    prompt: (
      <Translated
        vi={
          <>
            Softmax của lab 10.3 trừ đi giá trị lớn nhất của hàng trước khi gọi <code>exp</code> (
            <a href={href('ch10', 'labs')}>lab 10.3</a>). Một hàng logits đạt tới 95. Trong float32, điều gì hỏng nếu bạn
            bỏ phép trừ đó?
          </>
        }
      >
        Lab 10.3's softmax subtracts the row maximum before it calls <code>exp</code> (
        <a href={href('ch10', 'labs')}>lab 10.3</a>). A row of logits reaches 95. What goes wrong in float32 if you
        skip the subtraction?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="exp(95) tràn thành vô cùng, nên tổng là vô cùng và mọi xác suất đều ra 0 hoặc NaN.">
            exp(95) overflows to infinity, so the sum is infinite and every probability comes out as 0 or NaN.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không gì cả: float32 đạt tới khoảng 3.4 × 10³⁸, lớn hơn exp(95).">
            Nothing: float32 reaches about 3.4 × 10³⁸, which is more than exp(95).
          </Translated>
        ),
        why: (
          <Translated vi="exp(95) xấp xỉ 1.8 × 10⁴¹. exp vượt qua số float32 hữu hạn lớn nhất ở đối số khoảng 88.7.">
            exp(95) is about 1.8 × 10⁴¹. exp passes the largest finite float32 at an argument of about 88.7.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các xác suất mất vài chữ số chính xác nhưng vẫn dùng được.">
            The probabilities lose a few digits of precision but stay usable.
          </Translated>
        ),
        why: (
          <Translated vi="Tràn số không phải là mất chữ số. Kết quả là vô cùng, và vô cùng chia vô cùng là NaN.">
            Overflow is not a loss of digits. The result is infinity, and infinity divided by infinity is NaN.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ token lớn nhất bị ảnh hưởng: nó nhận xác suất 1, các token còn lại nhận 0, và greedy decoding vẫn chọn nó.">
            Only the largest token is affected: it gets probability 1, the rest get 0, and greedy decoding still picks it.
          </Translated>
        ),
        why: (
          <Translated vi="Tử số của nó và tổng đều là vô cùng, và thương của chúng là NaN, không phải 1.">
            Its numerator and the sum are both infinite, and their quotient is NaN, not 1.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trừ đi giá trị lớn nhất m của hàng giữ nguyên mọi xác suất (cả tử số lẫn mẫu số đều được nhân với
            e<sup>−m</sup>) và khiến mọi số mũ không vượt quá 0. Khi đó số hạng lớn nhất đúng bằng 1 và tổng ít nhất là 1,
            nên không có gì tràn và phép chia an toàn. Lab 10.3 kiểm tra các hàng được cộng thêm +90 chính vì điều này.
          </p>
        }
      >
        <p>
          Subtracting the row maximum m leaves every probability unchanged (numerator and denominator are both
          multiplied by e<sup>−m</sup>) and makes every exponent at most 0. The largest term is then exactly 1 and the sum is
          at least 1, so nothing overflows and the division is safe. Lab 10.3 tests rows offset by +90 for exactly this.
        </p>
      </Translated>
    ),
    source: 'ch. 20 §20.2, Eq. 20.2, p. 484; lab 10.3 README',
    vi: { source: 'ch. 20 §20.2, Eq. 20.2, p. 484; README của lab 10.3' },
  },
  'x1/start/q-exclusive': {
    kind: 'numeric',
    id: 'x1/start/q-exclusive',
    prompt: (
      <Translated
        vi={
          <>
            Các xác suất đã sắp xếp 0.5, 0.25, 0.125, 0.125 và p = 0.75. Quy tắc của lab 11.6 giữ token i khi khối xác suất
            xếp hạng trước nó nhỏ hơn p (<a href={href('ch11', '11.1')}>mục 11.1</a>). Nó giữ bao nhiêu token?
          </>
        }
      >
        Sorted probabilities 0.5, 0.25, 0.125, 0.125 and p = 0.75. Lab 11.6's rule keeps token i when the mass ranked
        before it is below p (<a href={href('ch11', '11.1')}>section 11.1</a>). How many tokens does it keep?
      </Translated>
    ),
    answer: 2,
    unit: 'tokens',
    explain: (
      <Translated
        vi={
          <p>
            Khối xác suất trước mỗi token là 0, 0.5, 0.75 và 0.875. Token 0 và 1 bắt đầu dưới 0.75 nên được giữ; phần của
            chính token 1 đưa tổng lên đúng 0.75, nên tập đã đạt p và token 2, bắt đầu tại 0.75, là không cần. Đây là scan
            loại trừ của mục 11.1, và là quy tắc mà các sampler của llama.cpp tính.
          </p>
        }
      >
        <p>
          The mass before each token is 0, 0.5, 0.75 and 0.875. Tokens 0 and 1 start below 0.75 and are kept; token 1's own
          share brings the total to exactly 0.75, so the set already reaches p and token 2, which starts at 0.75, is not
          needed. This is the exclusive scan of section 11.1, and it is the rule llama.cpp's samplers compute.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.1, p. 253; notes/today/ch11-facts.md D2', // i18n-ignore: citation
    vi: { unit: 'token' },
  },
  'x1/start/q-stable': {
    kind: 'choice',
    id: 'x1/start/q-stable',
    prompt: (
      <Translated
        vi={
          <>
            Lab 12.4 dựng danh sách ứng viên của mỗi hàng bằng một bộ lọc ổn định, tức một scan trên các cờ giữ lại (
            <a href={href('ch12', '12.5')}>mục 12.5</a>), thay vì bằng một bộ đếm nguyên tử. Vì sao sampler quan tâm đến điều này?
          </>
        }
      >
        Lab 12.4 builds each row's candidate list with a stable filter, a scan of keep flags (
        <a href={href('ch12', '12.5')}>section 12.5</a>), rather than with an atomic counter. Why does a sampler care?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một bộ đếm nguyên tử phát các vị trí theo thứ tự đến, nên danh sách, và mọi trường hợp hòa được phân định theo vị trí, có thể thay đổi giữa các lần chạy.">
            An atomic counter hands out slots in arrival order, so the list, and any tie broken by position, can change from run to run.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Atomic quá chậm cho một hàng cỡ bộ từ vựng, nơi hàng nghìn luồng cùng đập vào một bộ đếm.">
            Atomics are too slow for a vocabulary-sized row, where thousands of threads hit one counter.
          </Translated>
        ),
        why: <Translated vi="Ở đây tốc độ không phải lý do loại chúng. Thứ tự mới là lý do.">Speed is not what rules them out here. Order is.</Translated>,
      },
      {
        label: (
          <Translated vi="Metal không có atomic trên bộ nhớ device, và bộ lọc phải chạy trên cả ba backend, nên nó buộc phải dùng scan.">
            Metal has no atomics on device memory, and the filter must run on all three backends, so it has to use a scan.
          </Translated>
        ),
        why: (
          <Translated vi="Metal có atomic số nguyên 32-bit (notes/today/ch09-facts.md C1).">
            Metal has 32-bit integer atomics (notes/today/ch09-facts.md C1).
          </Translated>
        ),
      },
      {
        label: <Translated vi="Scan cần ít bộ nhớ hơn một bộ đếm.">A scan needs less memory than a counter.</Translated>,
        why: (
          <Translated vi="Scan cần nhiều hơn: một offset cho mỗi phần tử so với một bộ đếm cho mỗi hàng.">
            A scan needs more: an offset per element against one counter per row.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Thứ tự đầu ra của một bộ lọc dùng atomic là bất kỳ thứ tự nào mà các luồng tình cờ đến. Một sampler phân định
            trường hợp hòa theo token id, và được bạn mong đợi cho ra cùng một token với cùng một seed, thì cần cùng một
            danh sách ở mọi lần chạy. Một bộ lọc ổn định ghi mỗi token được giữ vào vị trí bằng số token được giữ đứng trước nó, điều chỉ
            phụ thuộc vào dữ liệu.
          </p>
        }
      >
        <p>
          The order of an atomic filter's output is whatever order the threads happened to arrive in. A sampler that breaks
          ties by token id, and that you want to give the same token for the same seed, needs the same list every run.
          A stable filter writes each kept token at the number of kept tokens before it, which depends only on the data.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.5, pp. 297–298; lab 12.4 README',
    vi: { source: 'ch. 12 §12.5, pp. 297–298; README của lab 12.4' },
  },
  'x1/start/q-vocab-kib': {
    kind: 'numeric',
    id: 'x1/start/q-vocab-kib',
    prompt: (
      <Translated
        vi={
          <>
            Một bộ đếm <code>uint32</code> cho mỗi token trong bộ từ vựng {n(VOCAB.qwen3)} id của Qwen3-8B: một bản sao
            riêng của các bộ đếm chiếm bao nhiêu KiB (<a href={href('ch09', '9.4')}>mục 9.4</a>)?
          </>
        }
      >
        One <code>uint32</code> count per token for Qwen3-8B's vocabulary of {n(VOCAB.qwen3)} ids: how many KiB is one
        private copy of the counts (<a href={href('ch09', '9.4')}>section 9.4</a>)?
      </Translated>
    ),
    answer: (VOCAB.qwen3 * 4) / 1024,
    tol: 0.001,
    unit: 'KiB',
    explain: (
      <Translated
        vi={
          <p>
            {n(VOCAB.qwen3)} × 4 byte = {n(VOCAB.qwen3 * 4)} byte = 593.5 KiB. Con số đó vượt mọi giới hạn on-chip cho mỗi
            khối có trong tài liệu, vì vậy lab 9.3 đếm thẳng vào bộ nhớ toàn cục (global memory) với một hàng bộ đếm cho
            mỗi request và không có bản sao riêng trong bộ nhớ chia sẻ (shared memory).
          </p>
        }
      >
        <p>
          {n(VOCAB.qwen3)} × 4 bytes = {n(VOCAB.qwen3 * 4)} bytes = 593.5 KiB. That is more than any documented per-block
          on-chip limit, which is why lab 9.3 counts straight into global memory with one row of counts per request and
          no shared-memory private copy.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.4, pp. 211–213, and Fig. 9.10, p. 214; notes/today/ch09-facts.md D7',
    vi: { source: 'ch. 9 §9.4, pp. 211–213, và Fig. 9.10, p. 214; notes/today/ch09-facts.md D7' },
  },
  'x1/start/q-msd-bin': {
    kind: 'numeric',
    id: 'x1/start/q-msd-bin',
    prompt: (
      <Translated
        vi={
          <>
            Một MSD radix select tìm 20 khóa lớn nhất của một hàng đọc histogram 8 bit cao nhất của nó từ bin cao nhất trở
            xuống: 2, 5, 30, 100, … khóa (<a href={href('ch14', '14.9')}>mục 14.9</a>). Nó còn phải chọn bao nhiêu khóa từ
            bên trong bin chứa khóa lớn thứ 20?
          </>
        }
      >
        An MSD radix select for the 20 largest keys of a row reads its top-8-bit histogram from the highest bin down: 2,
        5, 30, 100, … keys (<a href={href('ch14', '14.9')}>section 14.9</a>). How many keys must it still pick from inside
        the bin where the 20th largest key lies?
      </Translated>
    ),
    answer: 13,
    unit: 'keys',
    explain: (
      <Translated
        vi={
          <p>
            Các tổng tích lũy là 2, 7, 37. Khóa thứ 20 nằm ở bin thứ ba, chứa 30 khóa. 7 khóa phía trên bin đó đều được
            chọn, nên cần thêm 13 khóa từ bên trong nó, và lượt tiếp theo chỉ xét 30 khóa đó, từng chữ số một. Lab X1.1 dựng
            phép chọn này; nó không bao giờ sắp xếp hàng.
          </p>
        }
      >
        <p>
          The running totals are 2, 7, 37. The 20th key lies in the third bin, which holds 30 keys. The 7 keys above that
          bin are all in, so 13 more are needed from inside it, and the next pass looks only at those 30 keys, digit by
          digit. Lab X1.1 builds this selection; it never sorts the row.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.9, p. 346; notes/today/ch14-facts.md D2', // i18n-ignore: citation
    vi: { unit: 'khóa' },
  },
  'x1/start/q-one-block': {
    kind: 'choice',
    id: 'x1/start/q-one-block',
    prompt: (
      <Translated
        vi={
          <>
            Mọi kernel của mốc dự án M5 chạy một khối cho mỗi hàng. Ở batch 1, trên một thiết bị có S multiprocessor, một
            kernel như vậy giữ bận được tối đa bao nhiêu phần các multiprocessor (<a href={href('ch10', '10.10')}>mục 10.10</a>)?
          </>
        }
      >
        Every kernel of milestone M5 runs one block per row. At batch 1, on a device with S multiprocessors, what share of
        the multiprocessors can one such kernel keep busy at most (<a href={href('ch10', '10.10')}>section 10.10</a>)?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="1/S: lưới là một khối, và một khối chạy trên một multiprocessor">
            1/S: the grid is one block, and a block runs on one multiprocessor
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Tất cả, vì một hàng cỡ bộ từ vựng đủ dài để lấp đầy thiết bị">
            All of them, because a vocabulary-sized row is long enough to fill the device
          </Translated>
        ),
        why: (
          <Translated vi="Độ dài của hàng quyết định một khối đó làm việc bao lâu, không phải nó chiếm bao nhiêu multiprocessor.">
            The length of the row decides how long the one block works, not how many multiprocessors it occupies.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một nửa, phần còn lại được giữ cho kernel kế tiếp trong stream">
            Half of them, with the rest held back for the next kernel in the stream
          </Translated>
        ),
        why: (
          <Translated vi="Không có gì trong lần khởi chạy đặt việc lên một nửa thiết bị.">
            Nothing in the launch puts work on half the device.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Tùy kích thước bộ từ vựng: một hàng dài hơn trải ra trên nhiều multiprocessor hơn">
            It depends on the vocabulary size: a longer row spreads over more of them
          </Translated>
        ),
        why: (
          <Translated vi="Bộ từ vựng quyết định khối làm bao nhiêu việc. Dù thế nào lưới cũng chỉ có một khối.">
            The vocabulary sets how much work the block does. The grid has one block either way.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một khối được giao cho một multiprocessor và ở lại đó. Một hàng cho một khối, nên phần còn lại của thiết bị
            không có gì để làm. Cách tăng lên là thêm hàng (một batch) hoặc chia một hàng cho nhiều khối rồi gộp kết quả
            của chúng, như mục 10.9 làm. Trong M5, batch chính là lưới.
          </p>
        }
      >
        <p>
          A block is assigned to one multiprocessor and stays there. One row gives one block, so the rest of the device has
          nothing to do. The ways up are more rows (a batch) or splitting a row across blocks and combining their results,
          as section 10.9 does. In M5 the batch is the grid.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.2, p. 69; ch. 10 §10.10, pp. 246–248', // i18n-ignore: citation
  },
};

// ---------------------------------------------------------------- the map
const BOX: CSSProperties = {
  border: '1px solid var(--rule)', // i18n-ignore: CSS
  borderRadius: 4,
  background: 'var(--page)',
  padding: '0.6rem 0.7rem', // i18n-ignore: CSS
  fontFamily: 'var(--sans)',
  fontSize: '0.86rem',
  lineHeight: 1.4,
  margin: 0,
  minWidth: 0,
};
const LIST: CSSProperties = { listStyle: 'none', margin: '0.3rem 0 0', padding: 0 }; // i18n-ignore: CSS

function KernelLine({ k }: { k: KernelStep }) {
  return (
    <li style={{ margin: '0 0 0.3rem' }} data-lab={k.lab.num}>
      {k.what}: <a href={href(k.lab.chapter, 'labs')}>{tr('lab', 'lab')} {k.lab.num}</a>
    </li>
  );
}

/** The chain as M5 builds it, drawn from STAGES in M5's order, then the draw. */
function ChainMap() {
  return (
    <ol className="wide" data-part="chain-map" aria-label={tr('The sampler chain and the labs that supply it', 'Chuỗi sampler và các lab cung cấp nó')}
      style={{ listStyle: 'none', padding: 0, margin: '0.4rem 0 1.4rem', display: 'grid', gap: '0.5rem', gridTemplateColumns: 'repeat(auto-fit, minmax(12.5rem, 1fr))', maxWidth: '62rem' }}>
      {M5_ORDER.map((id, i) => {
        const s = stage(id);
        return (
          <li key={id} style={BOX} data-stage={id}>
            <b style={{ fontSize: '0.95rem' }}>{i + 1}. {s.label}</b>{' '}
            <span className="muted">{s.kind === 'truncate' ? tr('removes candidates', 'loại bớt ứng viên') : tr('changes logits', 'thay đổi logits')}</span>
            <ul style={LIST}>{s.kernels.map((k) => <KernelLine key={k.what} k={k} />)}</ul>
          </li>
        );
      })}
      <li style={{ ...BOX, borderColor: 'var(--ink-3)' }} data-stage="draw">
        <b style={{ fontSize: '0.95rem' }}>{M5_ORDER.length + 1}. <Translated vi="rút">draw</Translated></b> <span className="muted"><Translated vi="một token id cho mỗi hàng">one token id per row</Translated></span>
        <ul style={LIST}>
          <KernelLine k={DRAW} />
          <li className="muted" style={{ margin: 0 }} data-lab={GREEDY.lab.num}>
            <Translated vi="khi T ≤ 0 thì thay vào đó:">at T ≤ 0 instead:</Translated> {GREEDY.what}, <a href={href(GREEDY.lab.chapter, 'labs')}>{tr('lab', 'lab')} {GREEDY.lab.num}</a>
          </li>
        </ul>
      </li>
    </ol>
  );
}

export function SX10() {
  const q = questionsSX10;
  const one = rowBytes(VOCAB.qwen3);
  return (
    <>
      <Translated
        vi={
          <p>
            Phép nhân ma trận cuối cùng của một bước decode để lại cho bạn logits: một tensor{' '}
            <code>[batch × vocabulary]</code> gồm các điểm float32, mỗi request một hàng. Kernel kế tiếp phải biến mỗi hàng
            thành một token id. Kernel đó là sampler, và module này dựng nó trên GPU, trên CUDA, HIP và Metal, từ những
            kernel bạn đã viết.
          </p>
        }
      >
        <p>
          The last matrix multiplication of a decode step leaves you with logits: a <code>[batch × vocabulary]</code> tensor
          of float32 scores, one row per request. The next kernel has to turn each row into one token id. That kernel is the
          sampler, and this module builds it on the GPU, on CUDA, HIP and Metal, from kernels you have already written.
        </p>
      </Translated>
      <NoteGroup>
        <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
          <Translated
            vi={
              <p>
                Transformer của sách kết thúc ở lớp un-embedding, lớp biến đầu ra của lớp cuối thành một phân phối xác suất
                trên các token. Cách chọn ra một token không có trong sách. Module X1 là của khóa học này.
              </p>
            }
          >
            <p>
              The book's transformer ends at the un-embedding layer, which turns the last layer's output into a probability
              distribution over tokens. How one token is chosen is not in the book. Module X1 is ours.
            </p>
          </Translated>
        </Beyond>
        <Source section="20.1" pages="481" />
      </NoteGroup>
      <Translated
        vi={
          <p>
            Một sampler là một chuỗi ngắn các phép toán theo hàng: phạt những token đã được dùng, chỉ giữ lại các ứng viên
            khả dĩ nhất theo ba cách khác nhau, làm sắc nét hoặc làm phẳng phần còn lại, và rút một token. Mỗi phép toán đều
            nhỏ. Phần thú vị là thứ tự chúng chạy, các quy tắc phân định khi hòa, và làm sao để cùng một seed cho cùng một
            token trên ba backend.
          </p>
        }
      >
        <p>
          A sampler is a short chain of row-wise operations: penalise tokens that were already used, keep only the most
          likely candidates in three different ways, sharpen or flatten what is left, and draw one. Each operation is small.
          The interesting parts are the order they run in, the tie rules, and making the same seed give the same token on
          three backends.
        </p>
      </Translated>

      <h2>
        <Translated vi="Chuỗi, và ai cung cấp mỗi giai đoạn">The chain, and who supplies each stage</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mốc dự án M5 chạy các giai đoạn dưới đây, theo đúng thứ tự này, một khối cho mỗi hàng. Mỗi ô ghi tên lab trước
            đó có kernel mà nó dùng lại. Ba lab mới lấp các chỗ trống: một top-k chọn thay vì sắp xếp (X1.1), một bộ lọc min-p
            và một kernel gộp top-p với min-p (X1.2), và bước rút cùng các số ngẫu nhiên của nó (X1.3). Mục{' '}
            <a href={href('x1', 'X1.1')}>X1.1</a> giải thích vì sao là thứ tự này mà không phải thứ tự khác.
          </p>
        }
      >
        <p>
          Milestone M5 runs the stages below, in this order, one block per row. Every box names the earlier lab whose kernel
          it reuses. Three new labs fill the gaps: a top-k that selects instead of sorting (X1.1), a min-p filter and a fused
          top-p and min-p kernel (X1.2), and the draw with its random numbers (X1.3). Section{' '}
          <a href={href('x1', 'X1.1')}>X1.1</a> explains why this order and not another.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Sơ đồ được vẽ từ cùng danh sách giai đoạn mà trình khám phá của X1.1 chạy, nên hai bên không thể lệch nhau.
              Chuỗi của llama.cpp có thêm bốn giai đoạn mà M5 không dựng; X1.1 cho thấy chúng.
            </p>
          }
        >
          <p>
            The map is drawn from the same stage list the explorer of X1.1 runs, so the two cannot disagree. llama.cpp's
            chain has four more stages that M5 does not build; X1.1 shows them.
          </p>
        </Translated>
      </Aside>
      <ChainMap />

      <h2>
        <Translated vi="Thứ bạn đang chạy hôm nay">What you run today</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Nếu bạn gọi <code>generate</code> trong Hugging Face transformers, các logits processor chạy dưới dạng các phép
            toán tensor và token được rút bằng <code>torch.multinomial</code>, trên bất kỳ thiết bị nào chứa logits.
            llama.cpp mặc định lấy mẫu trên host, với <code>std::mt19937</code> và một lần đi inverse-CDF; nó có một nhánh
            thử nghiệm <code>--backend-sampling</code> chỉ bật mặc định cho các bản nháp của speculative decoding. mlx-lm
            chạy mọi bộ lọc dưới dạng phép toán MLX, tức là trên GPU qua Metal. vLLM lấy mẫu trên GPU, với một kernel Triton
            tìm một pivot thay vì sắp xếp, và FlashInfer là một tùy chọn trên CUDA.
          </p>
        }
      >
        <p>
          If you call <code>generate</code> in Hugging Face transformers, the logits processors run as tensor operations
          and the token is drawn by <code>torch.multinomial</code>, on whatever device the logits are on. llama.cpp samples
          on the host by default, with <code>std::mt19937</code> and an inverse-CDF walk; it has an experimental{' '}
          <code>--backend-sampling</code> path that is on by default only for the drafts of speculative decoding. mlx-lm
          runs every filter as MLX operations, so on the GPU under Metal. vLLM samples on the GPU, with a Triton kernel that
          finds a pivot instead of sorting, and FlashInfer as an option on CUDA.
        </p>
      </Translated>
      <Today
        checked={tr(
          `2026-11-20 in the facts pass (${FACTS}, D1, D3, D4, D5, D8; notes/today/ch14-facts.md D7)`,
          `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, D1, D3, D4, D5, D8; notes/today/ch14-facts.md D7)`,
        )}
      >
        <Translated
          vi={
            <p>
              llama.cpp: <code>bool backend_sampling = false;</code> cho sinh văn bản, <code>true</code> cho bản nháp
              speculative. transformers: <code>torch.multinomial(probs, num_samples=1)</code>. mlx-lm: các bộ lọc là phép
              toán MLX. vLLM: mặc định là một kernel Triton dựa trên pivot; <code>flashinfer.sampling</code> là tùy chọn.
            </p>
          }
        >
          <p>
            llama.cpp: <code>bool backend_sampling = false;</code> for generation, <code>true</code> for the speculative
            draft. transformers: <code>torch.multinomial(probs, num_samples=1)</code>. mlx-lm: filters are MLX ops. vLLM: a
            pivot-based Triton kernel by default; <code>flashinfer.sampling</code> optional.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Kích thước của bài toán">The size of the problem</Translated>
      </h2>
      <Predict
        id="x1/start/p-bytes"
        unit={tr('bytes', 'byte')}
        prompt={
          <Translated
            vi={
              <>
                Một bước lấy mẫu cho một hàng của Qwen3-8B: {n(VOCAB.qwen3)} logits float32, cộng với {n(VOCAB.qwen3)} bộ
                đếm token <code>uint32</code> mà các phép phạt cần. Bước đó phải đọc ít nhất bao nhiêu byte từ bộ nhớ toàn
                cục (global memory) cho hàng đó?
              </>
            }
          >
            One sampling step for one row of Qwen3-8B: {n(VOCAB.qwen3)} float32 logits, plus the {n(VOCAB.qwen3)}{' '}
            <code>uint32</code> token counts the penalties need. What is the least number of bytes the step must read from
            global memory for that row?
          </Translated>
        }
      >
        {(c) => (
          <div data-reveal="bytes">
            <Translated
              vi={
                <p>
                  Mỗi token đóng góp 4 byte logit và 4 byte bộ đếm, nên mức sàn là {n(VOCAB.qwen3)} × 8 ={' '}
                  <b>{n(one)} byte</b> cho mỗi hàng
                  {c === one ? ', đúng bằng con số của bạn' : Math.abs(c - one) / one < 0.05 ? ', gần với con số của bạn' : ''}.
                  Ở batch 64 đó là {n(one * 64)} byte, {(one * 64 / 1e6).toFixed(1)} MB, được đọc để ra một token cho mỗi
                  request.
                </p>
              }
            >
              <p>
                Every token contributes 4 bytes of logit and 4 bytes of count, so the floor is {n(VOCAB.qwen3)} × 8 ={' '}
                <b>{n(one)} bytes</b> per row
                {c === one ? ', exactly your number' : Math.abs(c - one) / one < 0.05 ? ', close to your number' : ''}. At batch
                64 that is {n(one * 64)} bytes, {(one * 64 / 1e6).toFixed(1)} MB, read for one token per request.
              </p>
            </Translated>
            <Translated
              vi={
                <p>
                  So với điều đó, mỗi giai đoạn chỉ làm vài phép toán trên mỗi token: một phép so sánh, một phép nhân, một
                  phép trừ, một hàm mũ. Không giai đoạn nào đến gần mức số phép tính trên mỗi byte mà tại đó GPU thôi phải
                  chờ bộ nhớ (roofline của <a href={href('ch05', '5.1')}>mục 5.1</a>; áp dụng nó cho lấy mẫu là của khóa
                  học này, và băng thông của chính bạn đến từ lab 5.1, không phải từ trang này). Mọi giai đoạn đều bị giới
                  hạn bởi bộ nhớ, và những câu hỏi đáng hỏi là hàng được đọc và ghi bao nhiêu lần, và thiết bị có bận chút
                  nào không.
                </p>
              }
            >
              <p>
                Against that, each stage does a handful of operations per token: a compare, a multiply, a subtraction, an
                exponential. None comes close to the arithmetic per byte at which a GPU stops waiting for memory (the
                roofline of <a href={href('ch05', '5.1')}>section 5.1</a>; applying it to sampling is ours, and your own
                bandwidth comes from lab 5.1, not from this page). Every stage is memory-bound, and the questions worth
                asking are how many times the row is read and written, and whether the device is busy at all.
              </p>
            </Translated>
          </div>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Câu hỏi thứ hai đó có một câu trả lời khó chịu. Mỗi kernel trong module này giao một khối cho một hàng: hình dạng
            của softmax ở lab 10.3, của kernel lấy mẫu của FlashInfer và của top-k Metal của llama.cpp. Ở batch 1 đó là một
            khối trên một multiprocessor, và phần còn lại của GPU ngồi không, như{' '}
            <a href={href('ch10', '10.10')}>mục 10.10</a> đã cho thấy với softmax. Cái giá đó bao nhiêu là điều bạn sẽ đo
            trong lab X1.3, lab đo thời gian một bước lấy mẫu ở batch 1 và batch 64 và so với một luồng CPU.
          </p>
        }
      >
        <p>
          That second question has an awkward answer. Each kernel in this module gives one block to one row: the shape of
          lab 10.3's softmax, of FlashInfer's sampling kernel and of llama.cpp's Metal top-k. At batch 1 that is one block on
          one multiprocessor, and the rest of the GPU idles, as <a href={href('ch10', '10.10')}>section 10.10</a> showed for
          the softmax. How much that costs is something you will measure in lab X1.3, which times a sampling step at
          batch 1 and batch 64 and against one CPU thread.
        </p>
      </Translated>
      <Today
        checked={tr(
          `2026-11-20 in the facts pass (${FACTS}, A6; notes/today/ch12-facts.md D2)`,
          `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, A6; notes/today/ch12-facts.md D2)`,
        )}
      >
        <Translated
          vi={
            <p>
              <code>SamplingFromProbKernel</code> của FlashInfer: một khối cho mỗi hàng,{' '}
              <code>curand_init(philox_seed, bx, ...)</code>. Top-k Metal của llama.cpp: một threadgroup cho mỗi hàng.
            </p>
          }
        >
          <p>
            FlashInfer's <code>SamplingFromProbKernel</code>: one block per row, <code>curand_init(philox_seed, bx, ...)</code>.
            llama.cpp's Metal top-k: one threadgroup per row.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi={'"Cùng seed, cùng token" sẽ có nghĩa là gì'}>What "same seed, same token" will mean</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đến cuối module, với một seed cho trước, M5 hứa hai điều. Thứ nhất, cùng các bit ngẫu nhiên trên mọi backend, từ
            một bộ sinh dựa trên bộ đếm được khóa bởi seed, hàng và bước decode (X1.7). Thứ hai, cùng một token, trừ khi số
            ngẫu nhiên rơi vào trong một biên độ đã nêu quanh ranh giới giữa phần xác suất của hai token (X1.8). Lời hứa thứ
            hai yếu hơn một cách có chủ ý.
          </p>
        }
      >
        <p>
          By the end of the module, M5 promises two things for a given seed. First, the same random bits on every backend,
          from a counter-based generator keyed by the seed, the row and the decode step (X1.7). Second, the same token,
          except when the random number lands within a stated margin of a boundary between two tokens' shares of the
          probability (X1.8). The second promise is weaker on purpose.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Lý do là những dữ kiện bạn có thể kiểm tra. cuRAND và hipRAND trên AMD biến cùng 32 bit thành các số float khác
            nhau với khoảng 0.4 % đầu vào. Ngôn ngữ shading của Metal hoàn toàn không có hàm sinh số ngẫu nhiên. Và Metal có
            thể làm tròn phép tính float về phía 0, nên một tổng tích lũy không nhất thiết khớp với của CPU ở bit cuối.
          </p>
        }
      >
        <p>
          The reasons are facts you can check. cuRAND and hipRAND on AMD turn the same 32 bits into different floats for
          about 0.4 % of inputs. Metal's shading language has no random number function at all. And Metal may round float
          arithmetic toward zero, so a cumulative sum need not match the CPU's in the last bit.
        </p>
      </Translated>
      <Today
        checked={tr(
          `2026-11-20 in the facts pass (${FACTS}, B2, C1, C5)`,
          `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, B2, C1, C5)`,
        )}
      >
        <Translated
          vi={
            <p>
              <code>curand_uniform</code>: x·2⁻³² + 2⁻³³; rocRAND trên AMD: 2⁻³² + x·2⁻³². MSL 4.1: không có hàm random,
              noise hay hash. MSL §8.2: "Either round ties to even or round toward zero rounding mode may be supported".
            </p>
          }
        >
          <p>
            <code>curand_uniform</code>: x·2⁻³² + 2⁻³³; rocRAND on AMD: 2⁻³² + x·2⁻³². MSL 4.1: no random, noise or hash
            function. MSL §8.2: "Either round ties to even or round toward zero rounding mode may be supported".
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Trước khi bắt đầu">Before you start</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sáu câu hỏi về những gì module này giả định từ các chương 9 đến 14 và các lab của chúng. Mỗi câu liên kết tới mục
            mà nó xuất phát, và mỗi câu quay lại trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Six questions on what this module assumes from chapters 9 to 14 and their labs. Each links to the section it comes
          from, and each comes back in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={q['x1/start/q-max-subtract']} label={tr('From chapter 10', 'Từ chương 10')} />
      <Ask q={q['x1/start/q-exclusive']} label={tr('From chapter 11', 'Từ chương 11')} />
      <Ask q={q['x1/start/q-stable']} label={tr('From chapter 12', 'Từ chương 12')} />
      <Ask q={q['x1/start/q-vocab-kib']} label={tr('From chapter 9', 'Từ chương 9')} />
      <Ask q={q['x1/start/q-msd-bin']} label={tr('From chapter 14', 'Từ chương 14')} />
      <Ask q={q['x1/start/q-one-block']} label={tr('From chapters 4 and 10', 'Từ chương 4 và 10')} />

      <h2>
        <Translated vi="Các lab">The labs</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bốn lab, tất cả đều ở <a href={href('x1', 'labs')}>trang lab</a>. Mỗi lab yêu cầu một dự đoán trước khi bạn đo
            và kiểm tra so với một bản tham chiếu CPU.
          </p>
        }
      >
        <p>
          Four labs, all on the <a href={href('x1', 'labs')}>labs page</a>. Each asks for a prediction before you measure
          and tests against a CPU reference.
        </p>
      </Translated>
      <ul>
        <Translated vi={<li><b>Lab X1.1</b>, top-k bằng radix select, đo so với một lần sắp xếp toàn bộ.</li>}>
          <li><b>Lab X1.1</b>, top-k by radix select, measured against a full sort.</li>
        </Translated>
        <Translated vi={<li><b>Lab X1.2</b>, top-p và min-p: ghép từ các lab trước, rồi gộp thành một kernel.</li>}>
          <li><b>Lab X1.2</b>, top-p and min-p: composed from earlier labs, then fused into one kernel.</li>
        </Translated>
        <Translated vi={<li><b>Lab X1.3</b>, mốc dự án M5: một sampler GPU theo batch, từ logits và bộ đếm tới một token cho mỗi hàng.</li>}>
          <li><b>Lab X1.3</b>, milestone M5: a batched GPU sampler, from logits and counts to one token per row.</li>
        </Translated>
        <Translated vi={<li><b>Lab X1.4</b>, săn lỗi bộ lọc: ba lỗi trong một kernel gộp top-p và min-p.</li>}>
          <li><b>Lab X1.4</b>, filter hunt: three defects in a fused top-p and min-p kernel.</li>
        </Translated>
      </ul>
    </>
  );
}
