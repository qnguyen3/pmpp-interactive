import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import { visibleCount } from '../../sim/attnref';
import { BOOK, METAL_DOCUMENTED, plan } from '../../sim/flashplan';
import { KiB, eq212 } from '../../sim/kvmath';
import { mergeD } from '../../sim/onlinesoftmax';

// ---------------------------------------------------------------- the numbers the questions take from the models
/** Two blocks' partial (max, denominator) for one row, merged by Eq. 20.4 (src/sim/onlinesoftmax.ts). */
export const SPLIT = { mA: 2, DA: 3, mB: 5, DB: 1.5, ...mergeD(2, 3, 5, 1.5) };
/** Verifying 4 drafted tokens against 1000 cached keys: what the first query sees (src/sim/attnref.ts). */
export const SPEC = { nq: 4, nk: 1000, first: visibleCount(0, 4, 1000), last: visibleCount(3, 4, 1000) };
/** Lab 20.2's tiles with K^T and V double-buffered, per B_c (src/sim/flashplan.ts). */
const doubled = (bc: number) => {
  const p = plan({ ...BOOK, bc });
  return { single: p.bytes.total, double: p.bytes.total + p.bytes.kt + p.bytes.v };
};
export const DOUBLE = { bc16: doubled(16), bc8: doubled(8), limit: METAL_DOCUMENTED };
/** A GQA model's cache per token in llama.cpp's q8_0 (34 bytes per 32 values, facts D3), Eq. 20.12. */
export const Q8 = { l: 32, hq: 32, hkv: 8, d: 128, bytes: eq212(1, 1, 32, 32, 4, 128, 34 / 32) };
const r3 = (v: number) => Math.round(v * 1000) / 1000;

export const questionsWrap: Record<string, Question> = {
  'ch20/wrap/q-split-merge': {
    kind: 'numeric',
    id: 'ch20/wrap/q-split-merge',
    prompt: (
      <Translated
        vi={
          <>
            Flash-Decoding chia các key của một head cho hai khối. Khối A kết thúc với giá trị lớn nhất đang chạy {SPLIT.mA} và
            mẫu số {SPLIT.DA}, khối B với giá trị lớn nhất {SPLIT.mB} và mẫu số {SPLIT.DB}, mỗi mẫu số tính tương đối theo giá
            trị lớn nhất của chính khối đó. Mẫu số của cả hàng, tính tương đối theo giá trị lớn nhất của hàng, là bao nhiêu?
          </>
        }
      >
        <>
          Flash-Decoding splits one head&rsquo;s keys between two blocks. Block A ends with running max {SPLIT.mA} and
          denominator {SPLIT.DA}, block B with max {SPLIT.mB} and denominator {SPLIT.DB}, each relative to its own max. What
          is the denominator of the whole row, relative to the row&rsquo;s max?
        </>
      </Translated>
    ),
    answer: r3(SPLIT.D),
    tol: 0.004,
    explain: (
      <Translated
        vi={
          <p>
            Giá trị lớn nhất của hàng là {SPLIT.m}. Các số hạng của khối A được đo so với {SPLIT.mA}, nên mỗi số hạng bằng e
            <sup>{SPLIT.mA} − {SPLIT.m}</sup> = {r3(SPLIT.aFactor)} lần giá trị của nó khi đo so với {SPLIT.m}: {SPLIT.DA} ×{' '}
            {r3(SPLIT.aFactor)} + {SPLIT.DB} × 1 = {r3(SPLIT.D)}. Đó là Eq. 20.4 với hai khối là hai tập rời nhau A và B. Các
            đầu ra trộn với cùng hai hệ số đó, và logsumexp của hàng là {SPLIT.m} + ln{' '}
            {r3(SPLIT.D)} = {r3(SPLIT.m + Math.log(SPLIT.D))}, con số duy nhất cho mỗi hàng và mỗi phần chia mà Flash-Decoding
            ghi ra. Lab 20.4 thực hiện phép trộn này giữa các warp của một khối; mục <a href={href('ch20', '20.5')}>20.5</a> có
            phần đại số.
          </p>
        }
      >
        <p>
          The row&rsquo;s max is {SPLIT.m}. Block A&rsquo;s terms were measured against {SPLIT.mA}, so each is e
          <sup>{SPLIT.mA} − {SPLIT.m}</sup> = {r3(SPLIT.aFactor)} of what it is against {SPLIT.m}: {SPLIT.DA} ×{' '}
          {r3(SPLIT.aFactor)} + {SPLIT.DB} × 1 = {r3(SPLIT.D)}. That is Eq. 20.4 with the two blocks as the disjoint sets A
          and B. The outputs merge with the same two factors, and the row&rsquo;s logsumexp is {SPLIT.m} + ln{' '}
          {r3(SPLIT.D)} = {r3(SPLIT.m + Math.log(SPLIT.D))}, the one number per row and split that Flash-Decoding writes.
          Lab 20.4 does this merge across the warps of a block; section <a href={href('ch20', '20.5')}>20.5</a> has the
          algebra.
        </p>
      </Translated>
    ),
    source: 'Eq. 20.4, p. 494; Flash-Decoding, notes/today/ch20-facts.md A8; the numbers are ours',
    vi: { source: 'Eq. 20.4, p. 494; Flash-Decoding, notes/today/ch20-facts.md A8; các con số là của chúng tôi' },
  },
  'ch20/wrap/q-spec-mask': {
    kind: 'numeric',
    id: 'ch20/wrap/q-spec-mask',
    prompt: (
      <Translated
        vi={
          <>
            Speculative decoding kiểm chứng {SPEC.nq} token nháp trong một lượt: attention với nq = {SPEC.nq} query so với nk =
            {' '}{SPEC.nk} key, kể cả các token nháp. Với mặt nạ nhân quả căn góc dưới bên phải của lab 20.1, query đầu tiên trong{' '}
            {SPEC.nq} query thấy bao nhiêu key?
          </>
        }
      >
        <>
          Speculative decoding verifies {SPEC.nq} drafted tokens in one pass: attention with nq = {SPEC.nq} queries against
          nk = {SPEC.nk} keys, the drafted tokens included. Under lab 20.1&rsquo;s lower-right causal mask, how many keys does
          the first of the {SPEC.nq} queries see?
        </>
      </Translated>
    ),
    answer: SPEC.first,
    tol: 0,
    unit: 'keys',
    vi: {
      unit: 'key',
      source: 'Fig. 20.18, p. 507; README của lab 20.1 (căn góc dưới bên phải); facts A8, D5; bối cảnh là của chúng tôi',
    },
    explain: (
      <Translated
        vi={
          <p>
            Hàng r thấy các cột c ≤ r + (nk − nq). Với r = 0, đó là các cột 0 đến {SPEC.nk - SPEC.nq}: {SPEC.first} key, mọi
            thứ trước các bản nháp và chính bản nháp đầu tiên. Query cuối thấy cả {SPEC.last}. Một mặt nạ căn góc trên bên trái,
            tức <code>is_causal</code> của PyTorch khi nq &lt; nk, sẽ chỉ cho query đầu thấy một key. Hình 20.18 kiểm các bản
            nháp song song; đây là mặt nạ khiến mỗi lần kiểm thấy đúng phần tiền tố của chính nó.
          </p>
        }
      >
        <p>
          Row r sees columns c ≤ r + (nk − nq). For r = 0 that is columns 0 to {SPEC.nk - SPEC.nq}: {SPEC.first} keys,
          everything before the drafts and the first draft itself. The last query sees all {SPEC.last}. An upper-left mask,
          PyTorch&rsquo;s <code>is_causal</code> when nq &lt; nk, would let the first query see one key. Fig. 20.18 checks
          the drafts in parallel; this is the mask that makes each check see exactly its own prefix.
        </p>
      </Translated>
    ),
    source: 'Fig. 20.18, p. 507; lab 20.1 README (lower-right alignment); facts A8, D5; the setting is ours',
  },
  'ch20/wrap/q-vocab': {
    kind: 'choice',
    id: 'ch20/wrap/q-vocab',
    prompt: (
      <Translated
        vi={
          <>
            Sau lớp cuối, bạn cần softmax của 128,000 logits từ vựng cho một chuỗi. Bạn gọi <code>launchCausalSoftmax</code> của
            lab 20.1{' '}với batch 1, nq = 1, nk = 128,000. Nó đúng. Vì sao nó chậm?
          </>
        }
      >
        <>
          After the last layer you need the softmax of 128,000 vocabulary logits for one sequence. You call lab 20.1&rsquo;s{' '}
          <code>launchCausalSoftmax</code> with batch 1, nq = 1, nk = 128,000. It is correct. Why is it slow?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Lưới chỉ có một khối: 256 luồng gánh cả hàng trong khi GPU ngồi không.">{'The grid is one block: 256 threads take the whole row while the GPU idles.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Mặt nạ góc dưới bên phải che phần lớn bộ từ vựng, nên hàng bị đọc nhiều lần.">{'The lower-right mask hides most of the vocabulary, so the row is read many times.'}</Translated>,
        why: <Translated vi="Với nq = 1, hàng duy nhất thấy mọi cột, và mỗi cột được đọc ba lần, như ở bất kỳ hàng nào.">{'With nq = 1 the single row sees every column, and each column is read three times, as in any row.'}</Translated>,
      },
      {
        label: <Translated vi="Logits rộng như vậy làm tràn fp32, nên kernel lùi về phép tính số học chậm hơn.">{'Logits that wide overflow fp32, so the kernel falls back to slower arithmetic.'}</Translated>,
        why: <Translated vi="Trừ đi giá trị lớn nhất giữ mọi số mũ ở mức 0 hoặc thấp hơn, bất kể hàng dài bao nhiêu.">{'Subtracting the max keeps every exponent at or below 0, whatever the row length.'}</Translated>,
      },
      {
        label: <Translated vi="Hai phép rút gọn khối của nó tốn hơn nhiều so với ba lượt đi qua hàng.">{'Its two block reductions cost far more than its three passes over the row.'}</Translated>,
        why: <Translated vi="Mỗi phép rút gọn chỉ là vài bước shuffle và một rào chắn, một lần cho mỗi hàng. Các lượt đi qua hàng là 500 lần đọc mỗi luồng.">{'Each reduction is a few shuffle steps and one barrier, once per row. The passes are 500 reads per thread.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lưới có một khối cho mỗi hàng, và ở đây chỉ có một hàng: 256 luồng mỗi luồng đọc 500 cột, ba lần. Cách chữa là cách
            chương này cứ quay lại: chia hàng cho nhiều khối, để mỗi khối tạo giá trị lớn nhất và mẫu số riêng, rồi trộn chúng
            bằng Eq. 20.4, như Flash-Decoding làm với các key. Module X1 xây các kernel lấy mẫu đi sau softmax này; sách chỉ áp
            dụng Hình 20.4 cho điểm số attention.
          </p>
        }
      >
        <p>
          The grid is one block per row, and here there is one row: 256 threads read 500 columns each, three times. The cure is the one this chapter keeps returning to:
          split the row across many blocks, let each produce its own max and denominator, and merge them with Eq. 20.4, as
          Flash-Decoding does for keys. Module X1 builds the sampling kernels that follow this softmax; the book applies
          Fig. 20.4 only to attention scores.
        </p>
      </Translated>
    ),
    source: 'Fig. 20.4, pp. 486–488; Eq. 20.4, p. 494; the setting is ours',
    vi: { source: 'Fig. 20.4, pp. 486–488; Eq. 20.4, p. 494; bối cảnh là của chúng tôi' },
  },
  'ch20/wrap/q-double-buffer': {
    kind: 'choice',
    id: 'ch20/wrap/q-double-buffer',
    prompt: (
      <Translated
        vi={
          <>
            Một đồng nghiệp dùng buffer đôi (double buffering) cho các ô K<sup>T</sup> và V của lab 20.2, mỗi loại hai ô, chọn
            bằng <code>j &amp; 1</code>, và xóa rào chắn sau vòng lặp warp, giữ lại rào chắn sau mỗi lần nạp. Trên một thiết bị
            có 32 KB bộ nhớ threadgroup từng chạy kernel của bạn ở B_c = 16, chuyện gì xảy ra?
          </>
        }
      >
        <>
          A colleague double-buffers lab 20.2&rsquo;s K<sup>T</sup> and V tiles, two of each chosen by <code>j &amp; 1</code>,
          and deletes the barrier after the warp loop, keeping the one after each load. On a device with 32 KB of
          threadgroup memory that ran your kernel at B_c = 16, what happens?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Kết quả sai: một warp nhanh nạp vào một ô mà các warp chậm vẫn đang đọc.">{'Wrong results: a fast warp loads into a tile that slow warps are still reading from.'}</Translated>,
        why: <Translated vi="Ô j + 1 được nạp vào buffer được đọc lần cuối ở vòng j − 1, và một warp chỉ vượt qua rào chắn nạp của vòng j sau khi mọi warp đã xong vòng j − 1.">{'Tile j + 1 goes into the buffer last read in iteration j − 1, and a warp passes iteration j\'s load barrier only after every warp has finished iteration j − 1.'}</Translated>,
      },
      {
        label: <Translated vi="Đúng, nhưng các ô vượt quá 32 KB ở B_c = 16; host phải chọn B_c = 8.">{'Correct, but the tiles outgrow 32 KB at B_c = 16; the host must pick B_c = 8.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Đúng, và không gì khác thay đổi: bớt được một rào chắn mỗi ô thuần túy là lợi.">{'Correct, and nothing else changes: one barrier per tile is simply a gain.'}</Translated>,
        why: <Translated vi="Mỗi loại hai ô là gấp đôi bộ nhớ chia sẻ cho K^T và V, và giới hạn thì không xê dịch.">{'Two of each tile is twice the shared memory for K^T and V, and the limit does not move.'}</Translated>,
      },
      {
        label: <Translated vi="Kết quả sai trừ khi có một __syncwarp() thay chỗ cho rào chắn đã xóa.">{'Wrong results unless a __syncwarp() takes the deleted barrier\'s place.'}</Translated>,
        why: <Translated vi="Mối nguy mà rào chắn bị xóa bảo vệ nằm giữa các warp; một rào chắn warp không sắp thứ tự gì giữa chúng.">{'The hazard the deleted barrier guarded was between warps; a warp barrier orders nothing between them.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Rào chắn nạp của vòng j đã chứng minh rằng mọi warp xong vòng j − 1, vốn là lần đọc cuối cùng của buffer mà vòng j + 1
            ghi đè: cùng lập luận như buffer đôi của lab 15.5. Mỗi warp chỉ chạm vào các hàng của riêng nó trong S_i, nên ô đó
            không cần bản sao thứ hai. Cái giá là không gian: ở B_c = 16 các ô tăng từ{' '}
            {DOUBLE.bc16.single.toLocaleString('en-US')} lên {DOUBLE.bc16.double.toLocaleString('en-US')} byte, vượt quá{' '}
            {DOUBLE.limit.toLocaleString('en-US')}; ở B_c = 8 chúng chiếm {DOUBLE.bc8.double.toLocaleString('en-US')}. Mã host
            của lab 20.2 sẽ chọn 8. Việc bớt một rào chắn có bù được cho các ô hẹp đi một nửa hay không là chuyện phải đo.
          </p>
        }
      >
        <p>
          The load barrier of iteration j already proves that every warp finished iteration j − 1, which is the last reader
          of the buffer iteration j + 1 overwrites: the same argument as lab 15.5&rsquo;s double buffering. Each warp
          touches only its own rows of S_i, so that tile needs no second copy. The cost is space: at B_c = 16 the tiles grow
          from {DOUBLE.bc16.single.toLocaleString('en-US')} to {DOUBLE.bc16.double.toLocaleString('en-US')} bytes, over{' '}
          {DOUBLE.limit.toLocaleString('en-US')}; at B_c = 8 they take {DOUBLE.bc8.double.toLocaleString('en-US')}. Lab
          20.2&rsquo;s host code would pick 8. Whether one barrier fewer pays for tiles half as wide is a measurement.
        </p>
      </Translated>
    ),
    source: 'Fig. 20.9 line 39, p. 496; double buffering, ch. 15 §15.8; facts C1; the variant is ours (src/sim/flashplan.ts)',
    vi: { source: 'Fig. 20.9 dòng 39, p. 496; double buffering, ch. 15 §15.8; facts C1; biến thể là của chúng tôi (src/sim/flashplan.ts)' },
  },
  'ch20/wrap/q-q8-cache': {
    kind: 'numeric',
    id: 'ch20/wrap/q-q8-cache',
    prompt: (
      <Translated
        vi={
          <>
            Một mô hình GQA có {Q8.l} lớp, {Q8.hq} head query và {Q8.hkv} head K/V rộng {Q8.d}. llama.cpp lưu cache của nó ở dạng
            q8_0: các khối 32 giá trị trong 34 byte. Cache của một token chiếm bao nhiêu KiB?
          </>
        }
      >
        <>
          A GQA model has {Q8.l} layers, {Q8.hq} query heads and {Q8.hkv} K/V heads of {Q8.d}. llama.cpp stores its cache
          as q8_0: blocks of 32 values in 34 bytes. How many KiB does one token&rsquo;s cache take?
        </>
      </Translated>
    ),
    answer: Q8.bytes / KiB,
    tol: 0.01,
    unit: 'KiB',
    explain: (
      <Translated
        vi={
          <p>
            Eq. 20.12 cho mỗi token: 2 × {Q8.l} lớp × {Q8.hkv} head K/V × {Q8.d} = 65,536 giá trị, mỗi giá trị 34/32 byte:{' '}
            {Q8.bytes.toLocaleString('en-US')} byte, {Q8.bytes / KiB} KiB. Cùng mô hình đó với 32 head K/V ở fp16 sẽ chiếm 512
            KiB mỗi token: gom nhóm cho 4×, lưu 8-bit thêm gần 2× nữa. Cả hai cũng cắt giảm số byte một bước decode đọc cho mỗi
            token, thứ quyết định thời gian của nó (mục <a href={href('ch20', '20.6')}>20.6</a>). Module{' '}
            <a href={href('x2', 'X2.7')}>X2</a> (X2.7) lượng tử hóa trọng số của một bước decode không có attention, nên không có
            cache nào để lượng tử hóa; không lab nào trong khóa học xây một cache đã lượng tử hóa.
          </p>
        }
      >
        <p>
          Eq. 20.12 per token: 2 × {Q8.l} layers × {Q8.hkv} K/V heads × {Q8.d} = 65,536 values, at 34/32 bytes each:{' '}
          {Q8.bytes.toLocaleString('en-US')} bytes, {Q8.bytes / KiB} KiB. The same model with 32 K/V heads in fp16 would
          take 512 KiB per token: grouping gives 4×, 8-bit storage almost 2× more. Both also cut the bytes a decode step
          reads per token, which is what sets its time (section <a href={href('ch20', '20.6')}>20.6</a>). Module{' '}
          <a href={href('x2', 'X2.7')}>X2</a> (X2.7) quantises the weights of a decode step that has no attention, so it has
          no cache to quantise; no lab in the course builds a quantised cache.
        </p>
      </Translated>
    ),
    source: 'Eq. 20.12, p. 509; q8_0 block size, notes/today/ch20-facts.md D3; the model shape is ours',
    vi: { source: 'Eq. 20.12, p. 509; kích thước khối q8_0, notes/today/ch20-facts.md D3; kích thước mô hình là của chúng tôi' },
  },
};

// ---------------------------------------------------------------- the page
export function Wrap() {
  const { progress } = useProgress();
  const { reports, loading } = useReports();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const name = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  const pass = (lab: string) => backends.filter((b) => labStatus(reports[b], lab, 'work') === 'pass');
  const m6 = pass('ch20/lab1_causal_softmax').filter((b) => pass('ch20/lab2_flash').includes(b));
  const m7 = pass('ch20/lab4_decode');
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch20/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}><Translated vi="Những gì cần mang theo">What to carry forward</Translated></h2>
      <Translated
        vi={
          <p>
            Phần tóm tắt của sách chỉ là một đoạn: attention không gộp, rồi KV cache, rồi flash attention, rồi cache tốn bao
            nhiêu bộ nhớ và các mô hình thu nhỏ nó thế nào. Những phần cần giữ:
          </p>
        }
      >
        <p>
          The book&rsquo;s summary is a paragraph: unfused attention, then the KV cache, then flash attention, then what the
          cache costs in memory and how models shrink it. The parts to keep:
        </p>
      </Translated>
      <Source section="20.8" pages="510" />
      <ul>
        <Translated
          vi={
            <li>
              Attention là hai GEMM với một softmax theo hàng ở giữa. Khi không gộp, S và P là các ma trận N × N được ghi vào
              rồi đọc lại từ bộ nhớ thiết bị; kernel softmax là một phép rút gọn mỗi khối một hàng với mặt nạ làm cận của vòng lặp.
            </li>
          }
        >
          <li>
            Attention is two GEMMs with a row softmax between them. Unfused, S and P are N × N matrices written to and read
            back from device memory; the softmax kernel is a block-per-row reduction with the mask as its loop bound.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Một softmax trên các mảnh rời nhau có thể được trộn chính xác nếu mỗi mảnh mang theo giá trị lớn nhất của nó: co giãn
              mẫu số và đầu ra của mỗi mảnh theo e<sup>m_piece − m_all</sup> rồi cộng lại. Các ô của một hàng, các warp của một
              khối, các khối của một bước decode và các phần chia của bộ từ vựng đều trộn theo đúng một quy tắc đó.
            </li>
          }
        >
          <li>
            A softmax over disjoint pieces can be merged exactly if each piece carries its max: rescale each piece&rsquo;s
            denominator and output by e<sup>m_piece − m_all</sup> and add. Tiles of a row, warps of a block, blocks of a
            decode step and splits of a vocabulary all merge with that one rule.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Flash attention giữ một giá trị lớn nhất, một mẫu số và một đầu ra chưa chuẩn hóa đang chạy cho mỗi hàng và chia một
              lần ở cuối. S và P chỉ tồn tại từng ô một trong bộ nhớ chia sẻ, nên lưu lượng N × N biến mất; thứ còn lại là hai
              phép nhân ma trận nhỏ cho mỗi ô.
            </li>
          }
        >
          <li>
            Flash attention keeps a running max, denominator and unnormalised output per row and divides once at the end.
            S and P exist only one tile at a time in shared memory, so the N × N traffic disappears; what is left is two
            small matrix products per tile.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Rào chắn đi theo dữ liệu, không đi theo mã. Listing của sách bỏ sót rào chắn warp giữa lần ghi P và{' '}
              <code>compute_O</code>, và một GPU có thể qua mọi bài kiểm tra mà không cần nó: shuffle không sắp thứ tự bộ nhớ,
              nhưng cũng chẳng có gì buộc các lane tách nhau ra.
            </li>
          }
        >
          <li>
            Barriers follow data, not code. The book&rsquo;s listing omits the warp barrier between the P write and{' '}
            <code>compute_O</code>, and a GPU may pass every test without it: shuffles do not order memory, but nothing forces the lanes apart either.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Với KV cache, thêm một token làm thay đổi một hàng của Q, K, V, S, P và O. Prefill là một GEMM và bị giới hạn bởi tính
              toán; decode là một lượt nhân ma trận-vectơ trên toàn bộ cache, khoảng một phép toán mỗi byte, và batching không giúp
              được nó vì mỗi chuỗi có cache riêng.
            </li>
          }
        >
          <li>
            With a KV cache, adding a token changes one row of Q, K, V, S, P and O. Prefill is a GEMM and compute-bound;
            decode is a matrix-vector pass over the whole cache, about one operation per byte, and batching does not help it
            because every sequence has its own cache.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Cache tốn 2 · số lớp · số head K/V · d · số byte cho mỗi token. MQA, GQA, MLA và các kiểu dữ liệu hẹp hơn thu nhỏ nó
              và nâng cường độ của decode; phân trang loại bỏ phần bộ nhớ giữ chỗ cho những token không bao giờ đến.
            </li>
          }
        >
          <li>
            The cache costs 2 · layers · K/V heads · d · bytes per token. MQA, GQA, MLA and narrower types shrink it and
            raise decode&rsquo;s intensity; paging removes the memory reserved for tokens that never come.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Mọi con số ở đây không phải phép tính số học đều đến từ các lab của bạn. Sách không in số đo thời gian nào, và flash
              attention viết bằng vòng lặp warp fp32 không tự động nhanh hơn các GEMM đã tinh chỉnh.
            </li>
          }
        >
          <li>
            Every number here that is not arithmetic came from your labs. The book prints no timing, and flash attention in
            fp32 warp loops is not automatically faster than tuned GEMMs.
          </li>
        </Translated>
      </ul>

      <h2><Translated vi="Một bài kiểm tra tổng hợp">A mixed check</Translated></h2>
      <p>
        <Translated
          vi={
            <>
              Các câu này dùng cả chương cho những tình huống mà các mục chưa cho thấy.
              {asked ? ` Đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu, tính theo câu trả lời mới nhất của bạn cho mỗi câu.` : ''}
            </>
          }
        >
          These use the whole chapter on situations the sections did not show.
          {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
        </Translated>
      </p>
      <Beyond>
        <Translated
          vi={
            <p>Các bối cảnh và biến thể của người đồng nghiệp trong những câu hỏi này là của chúng tôi, chọn cho các câu hỏi.</p>
          }
        >
          <p>The settings and the colleague&rsquo;s variant in these questions are ours, picked for the questions.</p>
        </Translated>
      </Beyond>
      <Ask q={q['ch20/wrap/q-split-merge']} />
      <Ask q={q['ch20/wrap/q-spec-mask']} />
      <Ask q={q['ch20/wrap/q-vocab']} />
      <Ask q={q['ch20/wrap/q-double-buffer']} />
      <Ask q={q['ch20/wrap/q-q8-cache']} />

      <h2><Translated vi="Các bài tập của sách">The book&rsquo;s exercises</Translated></h2>
      <Translated
        vi={
          <p>
            Chương kết thúc với bốn bài tập (p. 511), tất cả về kernel flash attention, và sách không đưa lời giải cho bài nào.
            Bài nào cũng được giải ở đây.
          </p>
        }
      >
        <p>
          The chapter ends with four exercises (p. 511), all about the flash attention kernel, and the book gives no
          solution for any of them. Each is worked here.
        </p>
      </Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }} data-table="exercises">
          <thead>
            <tr>
              <th><Translated vi="Bài tập">Exercise</Translated></th>
              <th><Translated vi="Nội dung">About</Translated></th>
              <th><Translated vi="Được giải ở đâu">Where it is worked here</Translated></th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1</td>
              <td><Translated vi="Mã host khởi chạy kernel của Hình 20.9">Host code that launches the kernel of Fig. 20.9</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      <code>launchFlashAttention</code> trong lab 20.2, và mã host Metal của nó, cũng là nơi chọn B_c; trên{' '}
                      <a href={href('ch20', 'labs')}>trang lab</a>
                    </>
                  }
                >
                  <code>launchFlashAttention</code> in lab 20.2, and its Metal host code, which also chooses B_c; on the{' '}
                  <a href={href('ch20', 'labs')}>labs page</a>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td><Translated vi={<><code>WarpReduce</code> của CUB: các kiểu, các đối số, các toán tử của nó</>}>CUB&rsquo;s <code>WarpReduce</code>: its types, its arguments, its operators</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      mục <a href={href('ch20', '20.3')}>20.3</a> (những gì đã đổi trong CUB, và câu hỏi về việc port Hình 20.4), câu hỏi
                      ở trang bắt đầu về nơi kết quả nằm lại, và bảng các chỉnh sửa của mục{' '}
                      <a href={href('ch20', '20.5')}>20.5</a>
                    </>
                  }
                >
                  section <a href={href('ch20', '20.3')}>20.3</a> (what changed in CUB, and the question on porting Fig.
                  20.4), the start page&rsquo;s question on where the result lands, and section{' '}
                  <a href={href('ch20', '20.5')}>20.5</a>&rsquo;s table of corrections
                </Translated>
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td>
                <Translated
                  vi={
                    <><code>initialize</code>, và vì sao O<sub>i</sub> và D<sub>i</sub> bắt đầu từ 0</>
                  }
                >
                  <code>initialize</code>, and why O<sub>i</sub> and D<sub>i</sub> start at 0
                </Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      câu hỏi và trình chạy từng bước online-softmax ở <a href={href('ch20', '20.5')}>20.5</a> (thử khởi đầu bằng âm vô
                      cực); lab 20.2
                    </>
                  }
                >
                  the question and the online-softmax stepper in <a href={href('ch20', '20.5')}>20.5</a> (try the
                  minus-infinity start); lab 20.2
                </Translated>
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td><Translated vi="Mọi lần nạp từ K và V trong Hình 20.11 đều được hợp nhất">Every load from K and V in Fig. 20.11 is coalesced</Translated></td>
              <td>
                <Translated
                  vi={
                    <>
                      câu hỏi ở <a href={href('ch20', '20.5')}>20.5</a>; câu hỏi tương tự về cách bố trí ở{' '}
                      <a href={href('ch20', 'start')}>trang bắt đầu</a> cho Q
                    </>
                  }
                >
                  the question in <a href={href('ch20', '20.5')}>20.5</a>; the same layout question on the{' '}
                  <a href={href('ch20', 'start')}>start page</a> for Q
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
            Bản PDF mà khóa học này dựa vào có những chỗ có vẻ nhầm ở chương 20, một số trong mã và một số trong văn bản. Những
            chỗ trong mã làm thay đổi thứ bạn viết, nên các lab dùng các dạng đã sửa và đã kiểm thử.
          </p>
        }
      >
        <p>
          The copy this course was built from has apparent slips in chapter 20, some in the code and some in the text. The
          code ones change what you write, so the labs use corrected, tested forms.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>Đây là cách chúng tôi đọc một bản PDF. Nếu bản của bạn khác, hãy tin bản của bạn.</p>
          }
        >
          <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
        </Translated>
      </Aside>
      <ul data-list="errata">
        <Translated
          vi={
            <li>
              p. 487: &ldquo;gridDim.y và gridDim.z bằng 0&rdquo; cho một lưới một chiều. Các chiều không dùng bằng 1. Xem{' '}
              <a href={href('ch20', '20.3')}>20.3</a>.
            </li>
          }
        >
          <li>
            p. 487: &ldquo;gridDim.y and gridDim.z are 0&rdquo; for a one-dimensional grid. Unused dimensions are 1. See{' '}
            <a href={href('ch20', '20.3')}>20.3</a>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Hình 20.4, 20.12 và 20.15 dùng <code>NEG_INFINITY</code>, thứ không header CUDA nào định nghĩa, và{' '}
              <code>cub::Max()</code> và <code>cub::Sum()</code>, những thứ CUB hiện tại không còn. Hình 20.12 và 20.15 còn cho cả
              16 warp dùng chung một <code>WarpReduce::TempStorage</code>, không phải cách dùng CUB ghi trong tài liệu. Các lab
              dùng giá trị khởi đầu hữu hạn và các phép rút gọn tự viết (<a href={href('ch20', '20.3')}>20.3</a>,{' '}
              <a href={href('ch20', '20.5')}>20.5</a>).
            </li>
          }
        >
          <li>
            Figs. 20.4, 20.12 and 20.15 use <code>NEG_INFINITY</code>, which no CUDA header defines, and{' '}
            <code>cub::Max()</code> and <code>cub::Sum()</code>, which current CUB no longer has. Figs. 20.12 and 20.15
            also give all 16 warps one <code>WarpReduce::TempStorage</code>, which is not CUB&rsquo;s documented use. The
            labs use finite start values and hand-written reductions (<a href={href('ch20', '20.3')}>20.3</a>,{' '}
            <a href={href('ch20', '20.5')}>20.5</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Hình 20.13 và 20.15: mỗi lane ghi một phần tử vào hàng P của warp mình rồi đọc cả hàng, không có rào chắn ở giữa. Cần
              một <code>__syncwarp()</code>; lab 20.3 nói về điều này.
            </li>
          }
        >
          <li>
            Figs. 20.13 and 20.15: each lane writes one element of its warp&rsquo;s P row and then reads the whole row, with
            no barrier between. A <code>__syncwarp()</code> is needed; lab 20.3 is about it.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Hình 20.9 chia N cho B_r và B_c, nên N phải là bội của cả hai, điều mà văn bản không nói. Các lab làm tròn lên và che
              mặt nạ theo chỉ số.
            </li>
          }
        >
          <li>
            Fig. 20.9 divides N by B_r and B_c, so N must be a multiple of both, which the text does not say. The labs round
            up and mask by index.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Hình 20.16 ghi mẫu số cuối D &ldquo;cho huấn luyện&rdquo;. Không có giá trị lớn nhất của hàng, D không dựng lại được
              P; FlashAttention-2 lưu logsumexp m + log D, và các lab cũng vậy.
            </li>
          }
        >
          <li>
            Fig. 20.16 writes the final denominator D &ldquo;for training&rdquo;. Without the row max, D cannot rebuild P;
            FlashAttention-2 stores the logsumexp m + log D, and so do the labs.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 495: ở dòng đầu của phần tiếp theo trong phép suy ra Eq. 20.6, số mũ thứ hai được in là l − m<sub>B</sub>{' '}
              trong khi ý là l − m<sub>A∪B</sub> (<a href={href('ch20', '20.5')}>20.5</a>).
            </li>
          }
        >
          <li>
            p. 495: in the first line of the continued derivation of Eq. 20.6, the second exponent is printed l − m<sub>B</sub>{' '}
            where l − m<sub>A∪B</sub> is meant (<a href={href('ch20', '20.5')}>20.5</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 502: dòng 6 của Hình 20.14 được nói là tính số hạng B của Eq. 20.4; thực ra nó co giãn mẫu số cũ, tức số hạng A.
              Cùng trang đó, tổng trong Hình 20.15 được nói là dùng m<sub>B</sub>; mã dùng giá trị lớn nhất đã cập nhật (
              <a href={href('ch20', '20.5')}>20.5</a>).
            </li>
          }
        >
          <li>
            p. 502: line 6 of Fig. 20.14 is said to compute the B term of Eq. 20.4; it rescales the old denominator, the A
            term. On the same page the sum of Fig. 20.15 is said to use m<sub>B</sub>; the code uses the updated max (
            <a href={href('ch20', '20.5')}>20.5</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 499: văn bản gọi hàm nạp là <code>load_KTV()</code>, listing gọi là <code>load_KT_and_V</code>; V được gắn dấu
              số 5, mà trong Hình 20.8 là ô P; và <code>addr()</code> được nói là lấy từ chương 16, nơi macro tên là{' '}
              <code>pad()</code>.
            </li>
          }
        >
          <li>
            p. 499: the text calls the loader <code>load_KTV()</code>, the listing <code>load_KT_and_V</code>; V is tagged
            with marker 5, which in Fig. 20.8 is the P tile; and <code>addr()</code> is said to come from chapter 16, where
            the macro is <code>pad()</code>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 510, Eq. 20.15: biểu thức ở giữa phải có 2h<sub>q</sub> + N ở mẫu số. Giới hạn, khoảng 2h
              <sub>q</sub>, không bị ảnh hưởng (<a href={href('ch20', '20.7')}>20.7</a>).
            </li>
          }
        >
          <li>
            p. 510, Eq. 20.15: the middle expression should have 2h<sub>q</sub> + N in the denominator. The limit, about 2h
            <sub>q</sub>, is unaffected (<a href={href('ch20', '20.7')}>20.7</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              pp. 508–509: g<sub>q</sub> được gọi là số nhóm, nhưng Eq. 20.12 lưu h<sub>q</sub>/g<sub>q</sub>{' '}
              head K/V, khiến g<sub>q</sub> là số head query trên mỗi head K/V (<a href={href('ch20', '20.7')}>20.7</a>).
            </li>
          }
        >
          <li>
            pp. 508–509: g<sub>q</sub> is called the number of groups, but Eq. 20.12 stores h<sub>q</sub>/g<sub>q</sub>{' '}
            K/V heads, which makes g<sub>q</sub> the query heads per K/V head (<a href={href('ch20', '20.7')}>20.7</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Chú thích attention của Hình 20.1 kết thúc bằng &ldquo;S x V&rdquo;; thực ra là P × V. p. 481 nói số tham số của một
              mô hình là tích của độ sâu, số chiều head và số head, điều này không đúng (
              <a href={href('ch20', '20.1')}>20.1</a>).
            </li>
          }
        >
          <li>
            Fig. 20.1&rsquo;s attention call-out ends with &ldquo;S x V&rdquo;; it is P × V. p. 481 says a model&rsquo;s
            parameter count is the product of depth, head dimension and head count, which it is not (
            <a href={href('ch20', '20.1')}>20.1</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 489 gọi cột mới của Q K<sup>T</sup> là &ldquo;toàn phần tử bằng 0&rdquo;; các logit bị che là −∞ và chỉ trở thành
              0 sau softmax (<a href={href('ch20', '20.4')}>20.4</a>).
            </li>
          }
        >
          <li>
            p. 489 calls the new column of Q K<sup>T</sup> &ldquo;all zero elements&rdquo;; the masked logits are −∞ and
            become 0 only after the softmax (<a href={href('ch20', '20.4')}>20.4</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 503 nói các đóng góp mới dùng giá trị lớn nhất &ldquo;tính đến hàng mới&rdquo;; có vẻ ý là &ldquo;ô mới&rdquo;
              (<a href={href('ch20', '20.5')}>20.5</a>).
            </li>
          }
        >
          <li>
            p. 503 says the new contributions use the max &ldquo;up to the new row&rdquo;; &ldquo;new tile&rdquo; seems
            meant (<a href={href('ch20', '20.5')}>20.5</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 506, hàng &ldquo;PaLM 2&rdquo;: kích thước của PaLM 2 chưa bao giờ được công bố, và các con số là của PaLM 62B,
              mô hình dùng multi-query attention với một head K/V rộng 256. Cache 16-bit 4,096 token của nó là 256 MiB, không phải 8
              GB (facts E2; <a href={href('ch20', '20.6')}>20.6</a>).
            </li>
          }
        >
          <li>
            p. 506, the &ldquo;PaLM 2&rdquo; row: PaLM 2&rsquo;s shape was never published, and the numbers are PaLM
            62B&rsquo;s, which uses multi-query attention with one K/V head of 256. Its 4,096-token 16-bit cache is 256 MiB,
            not 8 GB (facts E2; <a href={href('ch20', '20.6')}>20.6</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 506, hàng GPT-3: cửa sổ ngữ cảnh của GPT-3 là 2,048 token, nên cache 4,096 token không xảy ra; ở cửa sổ của nó,
              cache là 9 GiB, không phải 18 (facts E2; <a href={href('ch20', '20.6')}>20.6</a>).
            </li>
          }
        >
          <li>
            p. 506, the GPT-3 row: GPT-3&rsquo;s context window is 2,048 tokens, so a 4,096-token cache does not occur; at
            its window the cache is 9 GiB, not 18 (facts E2; <a href={href('ch20', '20.6')}>20.6</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              pp. 509–510, MLA: latent là một cho mỗi token mỗi lớp, không dùng chung giữa các lớp, và latent của DeepSeek-V2 rộng
              4.5 độ rộng head, không phải d, nên Eq. 20.14 đánh giá thấp cache của nó 4.5 lần. K và V được hấp thụ vào các phép
              chiếu khác, không được giải nén trước khi dùng (facts E5; <a href={href('ch20', '20.7')}>20.7</a>).
            </li>
          }
        >
          <li>
            pp. 509–510, MLA: the latent is one per token per layer, not shared across layers, and DeepSeek-V2&rsquo;s is
            4.5 head widths, not d, so Eq. 20.14 understates its cache 4.5 times. K and V are absorbed into the other
            projections, not decompressed before use (facts E5; <a href={href('ch20', '20.7')}>20.7</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 509, PagedAttention: các khối không được nạp từ bộ nhớ hệ thống khi cần. Chúng nằm ở bất kỳ đâu trong bộ nhớ GPU
              thông qua một bảng khối; bộ nhớ host chỉ là nơi các khối bị đẩy ra đi tới khi bộ nhớ chịu áp lực (facts E6;{' '}
              <a href={href('ch20', '20.7')}>20.7</a>).
            </li>
          }
        >
          <li>
            p. 509, PagedAttention: the blocks are not loaded from system memory when needed. They live anywhere in GPU
            memory through a block table; host memory is only where evicted blocks go under memory pressure (facts E6;{' '}
            <a href={href('ch20', '20.7')}>20.7</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              p. 506, in-flight batching: không phải &ldquo;nhét một yêu cầu mới vào batch có khối lượng việc thấp nhất&rdquo;. Các
              yêu cầu đã xong rời batch và các yêu cầu mới gia nhập sau mỗi vòng lặp, như Orca và TensorRT-LLM mô tả (facts E6;{' '}
              <a href={href('ch20', '20.6')}>20.6</a>).
            </li>
          }
        >
          <li>
            p. 506, in-flight batching: not &ldquo;slotting a new request into the batch with the lowest workload&rdquo;.
            Finished requests leave the batch and new ones join after every iteration, as Orca and TensorRT-LLM describe
            it (facts E6; <a href={href('ch20', '20.6')}>20.6</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Những chỗ nhầm nhỏ: Hình 20.3 vẽ một dấu số 4 mà văn bản không hề nhắc tới; kích thước KV cache ở p. 506 tính bằng
              gigabyte nhị phân.
            </li>
          }
        >
          <li>
            Small slips: Fig. 20.3 draws a marker 4 that the text never mentions; the KV-cache sizes on p. 506 are in binary
            gigabytes.
          </li>
        </Translated>
      </ul>
      <Translated
        vi={
          <p>
            Các chỉnh sửa về mã dựa trên tài liệu hiện tại của các nhà sản xuất, được trích trong tệp facts, không dựa trên trí nhớ.
          </p>
        }
      >
        <p>
          The code corrections rest on the vendors&rsquo; current documentation, quoted in the facts file, not on memory.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md A1, A2, A3, A5, A7, E2, E3, E5, E6; infra-facts A11)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Mỗi chỉnh sửa về mã ở trên được kiểm tra với tài liệu hiện tại của CUDA, CUB và FlashAttention-2, và mỗi chỉnh sửa về
              dữ kiện được kiểm tra với các bài báo được trích; tệp facts có các trích dẫn.
            </p>
          }
        >
          <p>
            Each code correction above is checked against current CUDA, CUB and FlashAttention-2 documentation, and each
            factual one against the cited papers; the facts file gives the quotations.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Những gì khóa học này rút gọn">What this course shortened</Translated></h2>
      <Translated
        vi={
          <p>
            Mỗi mục của chương có trang riêng. Chỉ giữ lại một câu hoặc một chỉ dẫn: lịch sử và các chú thích thuật ngữ của
            chương (pp. 477–478); tokenizer, embedding và mã hóa vị trí (pp. 479–480), những thứ bạn đã biết và không kernel nào
            ở đây phụ thuộc vào; vì sao các head chuyên môn hóa (p. 482); retrieval-augmented generation (p. 481); lưới bền
            (persistent grid) với rào chắn toàn lưới (p. 493), mà cơ chế được <a href={href('ch18', '18.6')}>mục
            18.6</a> trình bày; các kỹ thuật Hopper của FlashAttention-3 (p. 504); và mixture of experts (p. 510), một đoạn và
            một câu hỏi. Không đề cập: huấn luyện và lượt backward (pp. 482, 493), mà sách để dành cho tài liệu tham khảo [6].
            LayerNorm và mạng feed-forward của Hình 20.1 được phân loại ở <a href={href('ch20', '20.1')}>20.1</a>, không được
            cài đặt.
          </p>
        }
      >
        <p>
          Every section of the chapter has its own page. Kept to a sentence or a pointer: the chapter&rsquo;s history and
          terminology footnotes (pp. 477–478); tokenizer, embedding and positional encoding (pp. 479–480), which you know
          and no kernel here depends on; why heads specialise (p. 482); retrieval-augmented generation (p. 481); the
          persistent grid with a grid-wide barrier (p. 493), whose mechanism <a href={href('ch18', '18.6')}>section
          18.6</a> covers; FlashAttention-3&rsquo;s
          Hopper techniques (p. 504); and mixture of experts (p. 510), one paragraph and a question. Not covered: training
          and the backward pass (pp. 482, 493), which the book defers to its reference [6]. LayerNorm and the feed-forward
          of Fig. 20.1 are classified in <a href={href('ch20', '20.1')}>20.1</a>, not implemented.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Không có trong sách và không được xây trong chương này, dù được nhắc tên ở chỗ chúng quan trọng: attention nửa độ
            chính xác và dùng tensor core, decode tách KV giữa các khối, kernel cho cache phân trang, cửa sổ trượt, rotary
            embedding và cache đã lượng tử hóa. Trong số này, chỉ rotary embedding được xây ở nơi khác trong khóa học, trong
            module X3 (<a href={href('x3', 'X3.7')}>X3.7, bước Llama: RMSNorm, RoPE và SwiGLU</a>, và lab đầu tiên của nó).
          </p>
        }
      >
        <p>
          Not in the book and not built in this chapter, though named where they matter: half-precision and tensor-core
          attention, split-KV decoding across blocks, paged-cache kernels, sliding windows, rotary embeddings and quantised
          caches. Of these, only rotary embeddings are built elsewhere in the course, in module X3 (
          <a href={href('x3', 'X3.7')}>X3.7, the Llama step: RMSNorm, RoPE and SwiGLU</a>, and its first lab).
        </p>
      </Translated>

      <h2><Translated vi="Dự án đến giờ">The project so far</Translated></h2>
      <Translated
        vi={
          <p>
            Lab 20.1 và 20.2 là mốc dự án M6: attention nhân quả cho một lượt prefill, không gộp và gộp, cả hai đều ghi
            cùng một logsumexp. Lab 20.4 và 20.5 là mốc dự án M7: một bước decode với KV cache, theo từng head và theo nhóm.{' '}
            <span data-milestones={loading ? 'loading' : `m6:${m6.join(',') || 'none'};m7:${m7.join(',') || 'none'}`}>
              {loading
                ? 'Đang đọc báo cáo.'
                : backends.length === 0
                  ? 'Máy này chưa ghi lại kết quả lab nào.'
                  : `${m6.length ? `M6 của bạn đạt trên ${m6.map(name).join(' và ')} trên máy này` : 'M6 của bạn chưa đạt trên máy này'}; ${m7.length ? `M7 của bạn đạt trên ${m7.map(name).join(' và ')}` : 'M7 của bạn chưa đạt'}.`}
            </span>{' '}
            Xem <a href={href('ch20', 'labs')}>các lab</a>. Với M2 đến M7, dự án đã có các phép nhân ma trận và attention của
            một lớp decoder, cho một lượt prefill và cho từng bước decode một.
          </p>
        }
      >
        <p>
          Labs 20.1 and 20.2 are milestone M6: causal attention for a prefill, unfused and fused, both writing the same
          logsumexp. Labs 20.4 and 20.5 are milestone M7: one decode step with a KV cache, per head and grouped.{' '}
          <span data-milestones={loading ? 'loading' : `m6:${m6.join(',') || 'none'};m7:${m7.join(',') || 'none'}`}>
            {loading
              ? 'Reading reports.'
              : backends.length === 0
                ? 'No lab results are recorded on this machine yet.'
                : `${m6.length ? `Your M6 passes on ${m6.map(name).join(' and ')} on this machine` : 'Your M6 does not pass on this machine yet'}; ${m7.length ? `your M7 passes on ${m7.map(name).join(' and ')}` : 'your M7 does not pass yet'}.`}
          </span>{' '}
          See the <a href={href('ch20', 'labs')}>labs</a>. With M2 to M7 the project has the matrix products and the
          attention of a decoder layer, for a prefill and for one decode step at a time.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <a href={href('ch21')}>Chương 21</a> là nghiên cứu tình huống tiếp theo của sách, về scatter so với gather. Với dự
            án, các điểm dừng tiếp theo là <a href={href('ch22')}>chương 22</a>, nơi nghiên cứu về batching, mốc dự án M8, đo
            việc batching một bước decode mang lại bao nhiêu thông lượng và tốn bao nhiêu độ trễ, và{' '}
            <a href={href('x1')}>module X1</a>, nơi lấy mẫu token tiếp theo từ softmax mà các lớp của chương này cung cấp.
          </p>
        }
      >
        <p>
          <a href={href('ch21')}>Chapter 21</a> is the book&rsquo;s next case study, on scatter versus gather. For the
          project, the next stops are <a href={href('ch22')}>chapter 22</a>, whose batching study, milestone M8, measures
          what batching a decode step buys in throughput and costs in latency, and{' '}
          <a href={href('x1')}>module X1</a>, which samples the next token from the softmax this chapter&rsquo;s layers
          feed.
        </p>
      </Translated>
    </>
  );
}
